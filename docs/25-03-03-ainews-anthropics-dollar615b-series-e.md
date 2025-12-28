---
companies:
- anthropic
- openai
- deepseek
- lmsys
- perplexity-ai
- deutsche-telekom
date: '2025-03-04T06:51:49.245801Z'
description: '**Anthropic** 以 **615 亿美元的估值**完成了 **35 亿美元的 E 轮融资**，这标志着其 **Claude**
  AI 模型获得了强大的资金支持。**GPT-4.5** 在 LMArena 排行榜上夺得**所有类别的第一名**，在多轮对话、编程、数学、创意写作和风格控制方面表现卓越。**DeepSeek
  R1** 在带有风格控制的困难提示词测试中，与 GPT-4.5 并列第一。讨论重点关注了 **GPT-4.5** 与 **Claude 3.7 Sonnet**
  在编程和工作流应用方面的对比。**LMSYS 基准测试**的重要性再次被强调，尽管也有人质疑基准测试与用户获取之间的相关性。此外，**Perplexity AI**
  与**德国电信 (Deutsche Telekom)** 达成合作，将 **Perplexity 助手**集成到一款新型 AI 手机中。'
id: fb10d7fc-f048-4e8a-a77e-e50c9fe5d1d5
models:
- gpt-4.5
- claude-3.7-sonnet
- deepseek-r1
original_slug: ainews-anthropics-615b-series-e
people:
- lmarena_ai
- teortaxestex
- casper_hansen_
- omarsar0
- aidan_mclau
- willdepue
- vikhyatk
- teknim1
- reach_vb
- _aidan_clark_
- cto_junior
- aravsrinivas
title: Anthropic 的 615 亿美元 E 轮融资。
topics:
- model-performance
- benchmarking
- style-control
- coding
- multi-turn
- funding
- partnerships
- workflow
---

<!-- buttondown-editor-mode: plaintext -->**恭喜团队！**

> 2025年3月3日至3月4日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord 社区（**221** 个频道，**4084** 条消息）。预计节省阅读时间（按 200wpm 计算）：**481 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！


他们[简短的博客文章](https://www.anthropic.com/news/anthropic-raises-series-e-at-usd61-5b-post-money-valuation)见此。虽然这不是技术新闻，但顶级实验室（frontier lab）[每隔一周](https://buttondown.com/ainews/archive/ainews-xai-grok-3-and-mira-muratis-thinking/)融资一次的情况并不多见，为 Claude 筹集更多资金对 AI Engineers 来说无疑是好消息。

与此同时，[GPT 4.5 在 LMArena 上全面排名第一](https://twitter.com/lmarena_ai/status/1896590146465579105)。为了记录，这里是当前在样式控制（style control）下的排名情况。Claude 想要夺回领先地位还有很长的路要走。


![image.png](https://assets.buttondown.email/images/f47dc89a-ddbe-4cc3-8636-cf1e8e92b259.png?w=960&fit=max)






---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 简报

**模型性能与基准测试、对比与评估**

- **GPT-4.5 性能领先地位**：[@lmarena_ai](https://twitter.com/lmarena_ai/status/1896590146465579105) 宣布 **GPT-4.5** 已登顶 Arena 排行榜，在包括多轮对话（Multi-Turn）和样式控制（Style Control）在内的**所有类别中均排名第一**，基于超过 **3000 张选票**。[@lmarena_ai](https://twitter.com/lmarena_ai/status/1896590150718922829) 进一步详细说明，GPT-4.5 在**多轮对话、高难度提示词（Hard Prompts）、编程、数学、创意写作、指令遵循和长查询**类别中均处于领先地位。[@lmarena_ai](https://twitter.com/lmarena_ai/status/1896590154871210154) 强调了 GPT-4.5 在**样式控制**方面的优势，在该特定领域领跑排行榜。[@lmarena_ai](https://twitter.com/lmarena_ai/status/1896590159111713050) 提供了探索完整 GPT 4.5 结果的链接。
- **DeepSeek R1 与 GPT 4.5 并列第一**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1896591303150104784) 指出，**DeepSeek R1** 在带有样式控制的高难度提示词上与 **GPT 4.5 并列第一**，并向 OpenAI 团队表示祝贺。
- **GPT-4.5 vs Claude 3.7 编程能力**：[@casper_hansen_](https://twitter.com/casper_hansen_/status/1896604177171874298) 质疑 **GPT 4.5** 在编程方面是否真的优于 **Claude Sonnet 3.7**。
- **GPT-4.5 vs Claude 3.7 工作流**：[@omarsar0](https://twitter.com/omarsar0/status/1896620019053895842) 描述了一种新的编程工作流：使用 **GPT-4.5 进行头脑风暴**，**Claude 3.7 Sonnet 进行构建**，以及使用 **Windsurf 处理 Agent 任务**。
- **对 GPT-4.5 基准测试的质疑**：[@aidan_mclau](https://twitter.com/aidan_mclau/status/1896610660840259866) 询问 [@DaveShapi](https://twitter.com/DaveShapi) **4.5 是否对基准测试过拟合**，或者其他模型是否存在此问题。[@willdepue](https://twitter.com/willdepue/status/1896603985861378440) 对 **GPT-4.5 在没有推理时计算（test-time compute）的情况下登顶各类别**表示惊讶，认为预训练（pretraining）仍然至关重要。[@vikhyatk](https://twitter.com/vikhyatk/status/1896350962018869510) 正在撤回对 **GPT-4.5** 的正面评价，不想被视为“低品位的测试者”。
- **Claude Sonnet 3.7 性能**：[@Teknium1](https://twitter.com/Teknium1/status/1896616055851905160) 将 **Cursor 中的 Sonnet 3.7** 描述为“表现糟糕”，并质疑其正确的聊天模式用法。[@reach_vb](https://twitter.com/reach_vb/status/1896411707213578615) 提到 **Claude Sonnet 3.7** 和 **DeepSeek** 是其最喜欢的 LLM，并配合使用 **Cursor** 和 **DeepSeek chat**。
- **LMSYS 排行榜的重要性**：[@_aidan_clark_](https://twitter.com/_aidan_clark_/status/1896645456622682616) 表示 **LMSYS 显然是最重要的**基准测试，并建议各实验室应优先考虑它以最大化用户价值。
- **基准测试相关性受质疑**：[@cto_junior](https://twitter.com/cto_junior/status/1896599144338485468) 认为**现在战胜基准测试并不重要**，获取用户才是更关键的。

**行业新闻、融资与合作伙伴关系**

- **Anthropic 的 35 亿美元融资轮**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1896606683876753470) 宣布以 **615 亿美元的估值完成 35 亿美元融资**，由 Lightspeed Venture Partners 领投，旨在推进 AI 开发和国际扩张。
- **Perplexity AI 与德国电信（Deutsche Telekom）合作**：[@perplexity_ai](https://twitter.com/perplexity_ai/status/1896560542556533217) 宣布与 **Deutsche Telekom** 达成合作伙伴关系，使 **Perplexity Assistant** 成为其新款 AI 手机的原生功能。[@AravSrinivas](https://twitter.com/AravSrinivas/status/1896565726112256369) 和 [@yusuf_i_mehdi](https://twitter.com/yusuf_i_mehdi/status/1896603176230601144) 进一步强调，AI 优先的浏览器是未来，Edge 正在通过集成 Copilot 推动这一进程。
- **Microsoft Dragon Copilot 发布**：[@mustafasuleyman](https://twitter.com/mustafasuleyman/status/1896590113456677370) 强调了 **Microsoft Dragon Copilot 的发布**，旨在减少医疗保健领域的行政负担，让医生重新专注于患者。
- **DeepSeek AI 登陆 Copilot+ PC**：[@yusuf_i_mehdi](https://twitter.com/yusuf_i_mehdi/status/1896653214805811266) 提到 **DeepSeek R1 的 7B 和 14B 蒸馏模型**现已在搭载 Snapdragon 的 **Copilot+ PC** 上可用，并强调了混合 AI（hybrid AI）。
- **Firefly Aerospace 月球着陆**：[@kevinweil](https://twitter.com/kevinweil/status/1896459714189484507) 祝贺 [@Firefly_Space](https://twitter.com/Firefly_Space) 成为 **首家成功将探测器降落在月球上的商业公司**。

**工具、框架与编程工作流**

- **LlamaParse 更新，支持 Claude 3.7 和 Gemini 2.0**：[@llama_index](https://twitter.com/llama_index/status/1896644701924872615) 宣布了 **LlamaParse** 的更新，在 "Parse With Agent" 模式中增加了对 **AnthropicAI Claude Sonnet 3.7** 和 **Google Gemini 2.0 Flash** 的支持，以实现更好的表格解析和跨页一致性；在 "Parse With LVM" 模式中支持解析屏幕截图。
- **基于 LlamaIndex 工作流的旅行规划器教程**：[@llama_index](https://twitter.com/llama_index/status/1896611746930159736) 分享了 RS Rohan 的一个教程和仓库，关于如何使用 **LlamaIndex** 构建 Agentic 旅行规划器，展示了使用 Pydantic 模型的结构化预测功能、API 集成（Google Flights, Hotels, Top Sites）以及事件驱动架构。
- **用于简历提取的 LlamaExtract**：[@llama_index](https://twitter.com/llama_index/status/1896306203153830264) 推出了 **LlamaExtract**，由 **3.7 Sonnet 和 o3-mini** 等 SOTA LLM 驱动，用于从简历中提取标准化的候选人信息，并可推广到其他数据类型。
- **SynaLinks，受 Keras 启发的 LLM 应用框架**：[@fchollet](https://twitter.com/fchollet/status/1896663405462978611) 和 [@fchollet](https://twitter.com/fchollet/status/1896663354636464377) 介绍了 **SynaLinks**，这是一个 **受 Keras 启发的框架**，用于将 LLM 应用程序构建为可训练组件的 DAG，支持复杂的流水线和 RL 微调。
- **Groovy，Python 转 JavaScript 引擎**：[@_akhaliq](https://twitter.com/_akhaliq/status/1896569883984560320) 强调了 **Groovy**，这是一个 **Python 转 JavaScript 引擎**，可将 Python 函数转译为客户端执行，[@algo_diver](https://twitter.com/algo_diver/status/1896612188611363292) 指出其潜力在于能让 **Gradio 达到生产级水平**。
- **使用 MLX-LM 进行结构化生成的 Outlines**：[@awnihannun](https://twitter.com/awnihannun/status/1896370996598493283) 分享了如何将 [@dottxtai](https://twitter.com/dottxtai) 的 **Outlines** 与 **mlx-lm** 结合使用进行本地结构化生成，并提供了文档 [@awnihannun](https://twitter.com/awnihannun/status/1896371097240834188)。
- **用于可观测性和评估工具的 LangSmith**：[@hwchase17](https://twitter.com/hwchase17/status/1896425160414314695) 指出 **LangSmith** 被用于将用户反馈转化为评估（evals），强调可观测性即评估工具。
- **Cursor 编程工作流**：[@omarsar0](https://twitter.com/omarsar0/status/1896620019053895842) 提到在新的编程工作流中使用 **Cursor**。[@jeremyphoward](https://twitter.com/jeremyphoward/status/1896656468176425406) 提到使用 **Cursor** 配合 Python、fasthtml 和 MonsterUI 等工具，在一天内创建复杂的应用。
- **用于加密 AI Agent 通信的 Gibberlink**：[@ggerganov](https://twitter.com/ggerganov/status/1896615325116035244)、[@ggerganov](https://twitter.com/ggerganov/status/1896592079997788300) 和 [@ggerganov](https://twitter.com/ggerganov/status/1896592517686313170) 介绍了 **Gibberlink**，演示了两个 AI Agent 之间的加密语音聊天，并提供了 GitHub 项目链接。

**研究与论文**

- **脑机文本解码研究**：[@AIatMeta](https://twitter.com/AIatMeta/status/1896629271235613031) 重点介绍了来自 Meta FAIR 和 BCBL 研究人员关于 **Brain-to-Text Decoding** 的研究论文，这是一种通过打字实现的非侵入性方法。
- **扩散模型与流匹配课程**：[@omarsar0](https://twitter.com/omarsar0/status/1896564978985164825) 和 [@TheTuringPost](https://twitter.com/TheTuringPost/status/1896505290310451322) 分享了一门免费的 **MIT 关于流匹配与扩散模型导论的课程**，涵盖了理论、训练和应用，包括课程笔记、幻灯片、YouTube 视频和实验，[@omarsar0](https://twitter.com/omarsar0/status/1896564991270302178) 提供了另一个链接。
- **推理 LLMs 深度解析**：[@omarsar0](https://twitter.com/omarsar0/status/1896572276461703193) 推荐了“推理 LLMs 深度解析”，总结了后训练（post-training）方面的进展。
- **关于推理 LLMs 作为平方和求解器的 SoS1 论文**：[@_akhaliq](https://twitter.com/_akhaliq/status/1896395391014433147) 分享了一篇题为“SoS1: O1 and R1-Like Reasoning LLMs are Sum-of-Square Solvers”的论文。
- **提升人类动作理解的 HAIC 论文**：[@_akhaliq](https://twitter.com/_akhaliq/status/1896396258165871089) 发布了关于“HAIC”论文的内容，重点是通过为多模态 LLMs 提供更好的字幕来改进人类动作的理解和生成。
- **人形机器人操控的 Sim-to-Real 强化学习**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1896371407774449921) 和 [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1896371410123309432) 重点介绍了 Nvidia 关于**人形机器人基于视觉的灵巧操控的 Sim-to-Real 强化学习**的演讲，该研究在无需人类演示的情况下实现了强大的泛化能力，并提供了项目和摘要链接。
- **思维链与推理瓶颈**：[@francoisfleuret](https://twitter.com/francoisfleuret/status/1896584551117594815) 讨论了思维链（Chains-of-thought）如何使推理受限于计算资源（compute-bound），并建议将大型模型蒸馏为更快的 SSMs 或混合模型以获得更好的权衡。
- **LLMs 作为进化策略**：[@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1896483381183205420) 列出了访谈中讨论的几项工作，包括“(1) Large Language Models As Evolution Strategies”。
- **用于 Kernel 编程的 TileLang**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1896634485405610312) 提到了 **TileLang**，这是一种用户友好的 AI 编程语言，降低了 Kernel 编程的门槛。
- **LLM 信仰结构的评估**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1896546982644404251) 分享了对 LLM 信仰结构（belief structures）的深刻评估。
- **用于评估 AI 系统的 LangProBe**：[@lateinteraction](https://twitter.com/lateinteraction/status/1896632714054467763) 介绍了来自 [@ShangyinT](https://twitter.com/ShangyinT) 等人的 **LangProBe**，对应该构建和评估什么样的完整 AI 系统提出了疑问。

**AI 在商业与应用中的表现**

- **库存追踪与 Token 需求**：[@gallabytes](https://twitter.com/gallabytes/status/1896434971386236951) 认为，每天数万亿 Token 的需求将来自于**改进各经济部门的库存追踪**等领域。
- **用于 Shader Golf 的 AI**：[@torchcompiled](https://twitter.com/torchcompiled/status/1896314931634749844) 向致力于 **shader golf** 的人们致敬。
- **AI 驱动的 Wiki Explorer 应用**：[@omarsar0](https://twitter.com/omarsar0/status/1896576198433505596) 和 [@omarsar0](https://twitter.com/omarsar0/status/1896575392607293510) 使用 AI 开发了一个 **wikiexplorer app**，利用 Wikipedia 和 OpenAI 模型提供提示，旨在成为学习新主题的有趣方式。
- **用于文献综述的 AI 研究 Agent**：[@TheTuringPost](https://twitter.com/TheTuringPost/status/1896606548413317240) 推广了由 **SciSpace 开发的 Deep Review**，这是一个用于系统性文献综述的 AI 研究 Agent，声称它可以节省数小时的工作时间，并且比 OpenAI 的 Deep Research 和 Google Scholar 更具相关性。
- **Android 日常生活中的 AI**：[@Google](https://twitter.com/Google/status/1896607279417278520) 在 #MWC25 上重点展示了 **AI on Android**，演示了 Circle to Search 翻译菜单和 Gemini Live 学习复杂主题等功能。
- **结合 AlphaFold 的 AI 协同科学家示例**：[@_philschmid](https://twitter.com/_philschmid/status/1896530932011573636) 举了一个使用 **GoogleDeepMind AlphaFold 扩展 GoogleAI co-scientist** 以进行蛋白质修饰评估的例子。
- **使用 Groovy 和 Gradio 的 Web 开发中的 AI**：[@algo_diver](https://twitter.com/algo_diver/status/1896612188611363292) 认为 **Groovy** 将使 **Gradio** 具备生产力，适用于全栈 Web 开发。

**迷因与幽默**

- **Karpathy 的 AirPods Pro 传奇**：[@karpathy](https://twitter.com/karpathy/status/1896645112710709577) 分享了一条幽默的多行推文，模仿 4chan greentext 风格，讲述了 **AirPods Pro 故障**。
- **Elon Musk 与 Grok 现实主义**：[@Teknium1](https://twitter.com/Teknium1/status/1896317299478720926) 发布了“Grok 对现实主义更加开放”并附带链接，暗示 Grok 的无过滤特性，[@Teknium1](https://twitter.com/Teknium1/status/1896317236836737482) 在回复 Grok 图像对比时表示“更好”。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. Atom of Thoughts 增强小型模型**

- **[新的 Atom of Thoughts 看起来有望帮助小型模型进行推理](https://i.redd.it/xlairo4g9eme1.png)** ([Score: 641, Comments: 90](https://reddit.com/r/LocalLLaMA/comments/1j29hm0/new_atom_of_thoughts_looks_promising_for_helping/))：**Atom of Thoughts (AOT)** 算法显著增强了小型模型的推理能力，在 **HotpotQA** 上使用 **GPT-4o-mini** 达到了 **80.6% F1** 分数，超越了其他模型。AOT 的过程包括将问题分解为 **Directed Acyclic Graph (DAG)**，通过子问题收缩进行简化，并迭代以达到原子问题，如附带的流程图所示。
  - **对方法论和结果的批评**：用户质疑 **Atom of Thoughts (AOT)** 结果的可靠性，理由是 **1k 任务** 的样本量可能存在问题、未指明的置信区间，以及在 **temperature 1** 下进行的测试，这可能导致结果的高波动性。人们对结果的随机性表示担忧，认为如果没有重复测试，报告的改进在统计上可能并不显著。
  - **关于基于规则的方法的讨论**：辩论了 AI 中基于规则的方法的相关性，一些用户认为虽然 **rule-based approaches** 不具备可扩展性，但在特定语境下仍具相关性。提到了 **"bitter lesson"** 的概念，表明计算通常胜过编码知识，但这并不排除逻辑规则集的效用。
  - **实际实现和资源**：分享了 AOT 算法的 **open-source repository** 链接，允许用户自行探索和实现该算法 ([GitHub link](https://github.com/qixucen/atom))。此外，原始论文可在 [arXiv](https://arxiv.org/abs/2502.12018) 上查阅，提供了关于该算法开发和性能的更多细节。


**主题 2. Klee 开源，用于本地 LLM 使用且零数据收集**

- **[我今天开源了 Klee，这是一个旨在本地运行 LLM 且零数据收集的桌面应用。它还包括内置的 RAG 知识库和笔记功能。](https://i.redd.it/54k8f1ladhme1.png)** ([Score: 397, Comments: 67](https://reddit.com/r/LocalLLaMA/comments/1j2j7su/i_opensourced_klee_today_a_desktop_app_designed/))：**Klee 桌面应用** 现已开源，专为在不收集任何数据的情况下 **locally** 运行 **LLMs** 而设计，并包含 **RAG 知识库** 和笔记功能。应用界面提供了如 **"deepseek-r1-7b"** 等模型选项，并通过 **"Local Mode"** 开关强调隐私，确保没有数据被发送到云端。
  - 用户讨论了 Klee 的 **backend compatibility**，询问是否强制使用 **Ollama**，或者是否可以使用 **llama.cpp** 等替代方案。还有人好奇 Klee 与 **LM Studio** 和 **OpenWebUI** 等其他平台的对比，一些人指出 Klee 本质上是 Ollama 的一个封装。
  - **数据隐私** 是焦点，有人询问“零数据收集”的主张，以及使用 **Ollama + Open WebUI** 是否涉及数据收集。指出这两个平台都会运行统计数据以收集错误信息，这些功能可以被禁用，这与 Klee 对本地数据安全的强调一致。
  - **用户界面和功能** 受到讨论，一些用户对受 Slack 启发的 UI 感到反感，而另一些人则欣赏其对非技术用户的简单性。提出了关于 **Android port** 可能性、运行来自 **Hugging Face** 模型的能力以及 **RAG 知识库** 自定义的问题。


**主题 3. 分裂脑 'DeepSeek-R1-Distill-Qwen' 与 'Llama' 融合架构**

- **分裂脑 "DeepSeek-R1-Distill-Qwen-1.5B" 与 "meta-llama/Llama-3.2-1B"** ([Score: 139, Comments: 30](https://reddit.com/r/LocalLLaMA/comments/1j25luw/split_brain_deepseekr1distillqwen15b_and/)): **Split Brain 项目** 探索了一种新颖的双解码器（dual-decoder）架构，该架构结合了两个不同的语言模型 **DeepSeek-R1-Distill-Qwen-1.5B** 和 **meta-llama/Llama-3.2-1B**，以实现同步处理和 **cross-attention fusion**（交叉注意力融合）。该系统通过在不同的 GPU 上运行独立模型，利用 **EnhancedFusionLayer** 进行 **cross-attention**，并采用复杂的门控机制（gating mechanism）来实现自适应信息流，从而实现协作推理和专业化处理。该架构提高了计算效率和任务灵活性，在允许协作和专业化操作的同时，通过仅训练融合组件来保持参数效率。
  - **Cross-Attention Fusion**：**Split Brain 项目** 使用双向 **cross-attention fusion**，两个模型同时生成输出，关注彼此的隐藏表示（hidden representations）而非最终的 token 输出。这种在隐藏表示层级的实时交互允许模型在没有直接 token 反馈的情况下，对彼此的“思考过程”产生相互影响。
  - **模型词汇表挑战**：确定的一个关键挑战是管理模型之间不同的词汇表（vocabularies），这需要一种复杂的机制来确保无缝的交互和处理。
  - **个性化潜力**：人们对使用分裂脑方法实现个性化 AI 模型很感兴趣，即通过将一个反映个性的小型模型与一个强大的大型模型相结合。这可能通过允许一个模型引导和纠正另一个模型，通过协作处理增强个性化，从而超越当前基于 prompt 的 Agent。


## 其他 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

待完成

---

# AI Discord 回顾

> 由 Gemini 2.0 Flash Thinking 提供的摘要之摘要的总结

**主题 1. IDE 之战：Cursor 跌跌撞撞，Windsurf 乘风破浪，插件阵痛持续**

- **Cursor IDE 陷入 Bug 深渊**：Cursor IDE 用户正面临 [**不稳定性**](https://discord.com/channels/1074847526655643750/1074847527708393565/1346144838955667486)、**连接失败** 和 **检查点（checkpoint）故障**，这促使工程师们将 **Windsurf** 和 **Trae AI** 视为救生筏。最新版本被描述为*极其不稳定*，MCP 服务器配置增加了混乱，特别是在 Windows 和远程 Ubuntu 设置上，导致*客户端创建失败*，用户纷纷在论坛寻求帮助。
- **Windsurf 的 Ubuntu 更新导致系统崩溃后自我修复**：最近针对 Ubuntu 24.04 的 **Windsurf 更新** 产生了严重的负面影响，因 **FATAL:setuid_sandbox_host.cc(158)** 错误导致系统变砖，迫使部分用户重新安装并造成数据丢失，但随后的补丁和涉及 **chrome-sandbox** 权限的变通方法提供了生机。然而，**Windows ARM64** 用户正在庆祝，因为 **Windsurf Next** 现在支持他们的平台，可在此处[下载](https://codeium.com/windsurf/download-next)。
- **JetBrains 插件请求挂起**：**Codeium 的 JetBrains 插件** 让用户感到沮丧，它一直卡在 *Processing request* 状态，尤其是在最新的预发布版本中，导致其无法生成代码，迫使开发人员降级到更稳定的旧版本以维持工作流。JetBrains 插件的问题与 Windsurf Next 对 Windows ARM64 的支持形成鲜明对比，展示了不同 IDE 集成之间功能可靠性的不平衡。


**主题 2. Claude 3.7：速度瓶颈与额度紧缩，但依然令人印象深刻**

- **Claude 3.7 在 Cursor 中表现挣扎，运行缓慢**：**Claude 3.7** 在 **Cursor IDE** 中引发了用户头疼，用户反映其运行*极其缓慢*且容易在请求中途停止，迫使许多人降级或使用 Cursor 的 'Ask' 模式，引发了对其模型当前稳定性的担忧。尽管在 Cursor 中表现不稳定，OpenAI Discord 的用户宣称 *ChatGPT 现在就是个笑话*，并发现 **Claude 3.7** *非常令人印象深刻*，特别指出 **Claude** 在处理大文件时具有更优越的上下文理解能力，拥有 **200K** token 窗口，超越了 **ChatGPT** 的 **128K**。
- **Claude 3.7 像吃豆人一样吞噬 Windsurf 额度**：Windsurf 中的 **Claude 3.7** 正在以惊人的速度消耗 **premium flow action credits**，有报告称即使是微小的编辑，每个 prompt 也会产生 **30-40 次 tool calls**，导致额度迅速耗尽并引发用户愤怒，一些用户正切换回 **3.5** 或考虑转投 **Cursor** 以逃避额度消耗。用户敦促 Codeium 将 **Claude 3.7** 从默认模型状态降级，因为它对额度的消耗过于贪婪。
- **Claude Code 获得 Anon-Kode 重混版，支持 Open API**：一位名为 anon-kode 的开发者在提取原始源代码后 ([原始推文](https://x.com/dnak0v/status/1894254049802744188))，发布了一个修改过的、**OpenAI 兼容**版本的 **Claude Code**，命名为 **anon-kode**。它兼容 **OpenAI APIs** ([推文](https://x.com/dnak0v/status/1896652107857748086)) 并在 [GitHub](https://github.com/dnakov/anon-kode) 上可用，提供了一个潜在的开源替代方案，尽管还有*很多东西需要修复*。


**主题 3. AI 模型：新发布、性能怪癖与伦理困境**

- **GPT-4.5 登顶 Arena 榜首，图像识别能力存议**：**GPT-4.5** 已登上 Arena 排行榜榜首，在从 coding 到创意写作的各个类别中占据主导地位 ([来源](https://x.com/lmarena_ai/status/1896590146465579105))，但其图像识别能力正受到审查，评价褒贬不一，并引发了关于它是否超越 GPT-4o 的争论，尽管初步测试显示其在 MMMU 基准测试上仅有 **+5%** 的边际提升。尽管在排行榜上取得了胜利，一些用户觉得 OpenAI 正在降低 Plus 用户的优先级以偏向 Pro 用户，暗示了高级会员地位认知的转变。
- **Grok 的自定义指令未能成功“整活”，引发人格模拟恐慌**：Grok AI 备受期待的 [custom instructions](https://www.thetechoutlook.com/news/web-social-media/new-custom-instructions-feature-added-for-x-platforms-grok-ai-on-web/) 功能现已向所有用户开放，但因无用而面临批评。用户报告称无法将 Grok 塑造成理想的人格，其中一次试图创建一个*“辱骂性和淫秽的喷子”*的尝试以失败告终，让用户质疑该功能的有效性。尽管自定义指令表现不佳，Grok 因其调试能力受到称赞，在这一领域胜过 **O3 mini high sonnet** 等模型，尽管一些用户发现 **O3 mini high sonnet** 在代码创建任务中表现更优。
- **Phi-3 模型微调面临 A100 硬件门槛，Dataset Viewer 需要修复错误**：为多模态微调 **Phi-3** 被证明是一项艰巨的任务，即使使用 Colab Pro，估计也需要 *6 台以上的 A100 以及大约 2 周时间*。同时，Hugging Face 的 Dataset Viewer 正受到错误的困扰，影响了与各种库和 SQL 的兼容性，阻碍了数据的可发现性和可用性。尽管面临这些挑战，Hugging Face 正在庆祝 **Remote VAE Decode endpoints** 在 **SD v1**、**SD XL** 和 **Flux** 上的延迟降低了高达 **10x**，这归功于代号为 **honey** 的项目，通过 [Hybrid Inference](https://huggingface.co/docs/diffusers/main/en/hybrid_inference/overview) 赋能本地 AI 构建者。


**主题 4. 硬件动态：Tilelang 获胜、AMD 崛起与 SRAM 秘密**

- **Tilelang 内核性能完胜 Triton，逼近 Flash-MLA 速度**：一个精简的 **80 行 tilelang 内核** 在 H100 上达到了 **DeepSeek Flash-MLA 95% 的性能**，相比 **Triton 实现了 500% 的加速**，展示了 tilelang 在高性能计算方面的潜力，代码已在 [GitHub](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_mla) 开源。这一性能飞跃引发了建立 **MLA 排行榜** 以展示类似成果的呼声，该榜单可能会由 **BitNet 团队** 重新调整用途。
- **AMD GPU 逐渐逼近 ML 聚光灯，Intel Arc A770 加入 Tinygrad 阵营**：关于 **AMD** 和 **Intel** 成为 ML 流水线中 CUDA 可行替代方案的讨论正在升温。一些人认为 AMD 市场份额的增加可能会刺激其 GPU 计算部门的更多投资。同时，Intel Arc A770 GPU 已确认通过 OpenCL 后端与 Tinygrad 兼容，为开发者拓宽了硬件选择。尽管 AMD 取得了进展，但关于其代工产能获取的问题依然存在，人们担心 NVIDIA 在芯片制造资源的获取上仍持有显著优势。
- **SRAM 缓存机制揭秘**：对 **SRAM** 架构的深入研究表明，寄存器、共享内存和缓存都是 SRAM 结构，未分配的共享内存会转化为 **L1 cache**。虽然 Triton 在 `tl.load` 中的 `cache_modifier` 允许指定 L1 或 L2 命中，但缺乏直接的缓存层级控制，这揭示了 GPU 编程中内存管理的细微层级。对于 CUDA 编译，建议在 **PyTorch** 中使用 `torch.cuda.get_device_capability()` 来确定 `--arch=`，不过 `nvidia-smi --query-gpu=name,compute_cap --format=csv` 提供了一个无需 PyTorch 的替代方案。


**主题 5：Agent 创新与挫折：旅游规划 AI、Smol Agent 测验失败以及 MCP 多 Agent 愿景**

- **旅游应用 Agent 现身拯救被短视频淹没的旅行者**：一款名为 [ThatSpot](https://thatspot.app/) 的新应用出现，旨在应对旅游短视频（Reels）信息过载。它部署了 **AI Agent**，直接从旅游短视频中自动提取关键行程规划数据——地点、价格、预订链接——为受旅行渴望驱使的用户自动化了数小时的手动研究并简化了行程组织。该应用承诺处理旅游短视频并提取提及的每个地点，使繁琐的手动研究过程自动化。
- **Smol Agents 测验难倒学生，错误日志中藏有线索**：Smol Agents 测验让用户感到头疼，有报告称尽管多次尝试，但要求不明确且得分不及格。这促使人们呼吁从 [测验的 app.py 文件](https://huggingface.co/spaces/agents-course/unit2_smolagents_quiz/blob/main/app.py) 中 *挖掘错误日志*，以确定必要的工具和模型提供商，突显了 AI 学习平台中需要更清晰的测验说明和更好的错误反馈。尽管测验遇到困难，HuggingFace 还是推出了新的 NLP 推理课程单元，旨在教育 LLM 中的强化学习以及对 [Open R1](https://github.com/huggingface/open-r1) 的贡献。
- **MCP 多 Agent 架构初现，快速 Agent 框架浮出水面**：工程师们正在探索 **用于多 Agent 系统的 MCP**，从 Anthropic 的研讨会中汲取灵感，并构思 Agent 在跨设备协作的框架。一位成员分享了他们的 [fast-agent GitHub 项目](https://github.com/evalstate/fast-agent)，用于 *定义、提示和测试启用 MCP 的 Agent 及工作流*，允许为 Agent 配置不同的 MCP 服务器，并由其他 Agent 作为工具调用。然而，MCP Terraform Registry 的设置被证明很麻烦，特别是在使用 Claude 桌面版和 Cline 时，当系统级代理处于活动状态时会遇到 *mcp-server-fetch* 错误。


---

# 第 1 部分：Discord 高层级摘要

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 饱受不稳定性困扰**：用户报告在最新的 **Cursor IDE** 版本中存在 [**不稳定性**](https://discord.com/channels/1074847526655643750/1074847527708393565/1346144838955667486)、**连接失败**以及 **checkpoints 功能失效**的问题。
   - 由于用户体验不佳，成员们正在考虑 **Windsurf** 和 **Trae AI** 等替代方案。
- **MCP Servers 导致配置噩梦**：成员们在 **Cursor** 中[配置 **MCP servers**](https://discord.com/channels/1074847526655643750/1074847527708393565/1346225879193354291) 时遇到困难，特别是在 **Windows** 和远程 Ubuntu 工作区环境下，面临诸如“客户端创建失败”等问题。
   - 一位成员最终通过 Pupeteer 解决了问题，并使用 [**Firecrawl MCP server**](https://github.com/mendableai/firecrawl-mcp-server) 配合 LLM 客户端进行网页抓取。
- **Claude 3.7 面临故障**：用户在使用 [**Claude 3.7**](https://discord.com/channels/1074847526655643750/1074847527708393565/1346143214571280384) 时遇到问题，例如响应*极度缓慢*以及在没有错误提示的情况下中途停止请求。
   - 因此，许多人转而使用 Cursor 的 'Ask' 模式，或者在执行关键任务时回退到旧版本。
- **设计师们投身落地页设计**：成员们分享了[使用 **Cursor** 生成的**落地页设计**](https://discord.com/channels/1074847526655643750/1074847527708393565/1346170580382126164)，并讨论了它们的美学吸引力和有效性。
   - 社区将这些设计与 [Linear](https://linear.app/)、[Framer](https://www.framer.com/)、[Magician Design](https://magician.design/) 和 [Webflow](https://webflow.com/) 进行比较以获取灵感。
- **Repo Prompt 的多文件编辑功能备受赞誉**：用户对 [Repo Prompt](https://discord.com/channels/1074847526655643750/1074847527708393565/1346338341037107300) 感到兴奋，称赞其 **多文件编辑** 能力和代码片段集成功能。
   - 社区还提到了用于调试的 [BrowserTools](https://browsertools.agentdesk.ai/installation)，以及用于文件选择的 [PasteMax](https://github.com/kleneway/pastemax)（Repo Prompt 的一个*开源“穷人版”*）。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 增加 Windows ARM64 支持**：**Windsurf Next** 现在支持 **Windows ARM64**，可在此处[下载](https://codeium.com/windsurf/download-next)。
   - 这一扩展允许 **Windows ARM64** 平台的用户利用 **Windsurf Next** 中的最新功能和改进。
- **Windsurf 的 Ubuntu 更新导致系统崩溃**：最近的 **Windsurf 更新** 在 **Ubuntu 24.04** 上引发了问题，导致应用程序启动失败并提示 **FATAL:setuid_sandbox_host.cc(158)** 错误。
   - 一位用户报告了系统崩溃、重新安装和数据丢失的情况，强调了更新前备份的必要性，并且可能需要手动更改 **chrome-sandbox** 权限的权宜之计。
- **Claude 3.7 消耗额度过快引发用户不满**：用户报告 Windsurf 中的 **Claude 3.7** 正在迅速耗尽 **premium flow action credits**，原因是每个 prompt 产生的工具调用（tool calls）过多，有些用户在进行微小更改时竟产生了 **30-40 次工具调用**。
   - 成员们建议 Codeium 取消将 **Claude 3.7** 作为默认模型，一些人为了提高效率切换回 **3.5** 或其他模型，并考虑转向 **Cursor**。
- **Codeium 客户支持面临审查**：用户报告 **Codeium** 的客户支持体验较差，一名用户等待解决 **订阅问题** 已长达四周。
   - 缺乏及时有效的支持正促使用户寻求替代方案，并引发了对 **Codeium** 响应能力的担忧。
- **JetBrains 插件受“正在处理请求”卡死困扰**：**JetBrains 插件** 用户遇到了持续的 *Processing request* 状态，导致报错，特别是在最新的预发布版本中。
   - 该问题导致插件无法生成响应，中断了工作流，迫使用户降级到更稳定的版本。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI 举办 Sora 入门培训**：**Sora** 团队在 <t:1741024800:R> 举办了一场直播入门会议，涵盖了 **Sora** 基础知识和最佳 Prompting 技巧，你可以通过 [此 Discord 链接](https://discord.gg/6rjjzSrJ?event=1345127676519649326) 加入讨论。
   - **Sora 101** 会议还分享了早期访问艺术家的入门流程心得。
- **GPT-4.5 图像识别评价褒贬不一**：成员们正在争论新款 **GPT-4.5** 的 **Image Recognition** 能力是否优于 GPT-4o，[Future Machine](https://futuremachine.io) 对 OpenAI (OAI) 的选择发声较多。
   - 初步测试显示，**GPT-4.5** 在 MMMU（面向视觉推理的基准测试）上的得分比 4o 略高，有 **+5% 的提升**。
- **Custom Grok 表现不佳**：[Grok AI 的 Custom Instructions](https://www.thetechoutlook.com/news/web-social-media/new-custom-instructions-feature-added-for-x-platforms-grok-ai-on-web/) 功能已向所有用户发布，但成员们反映该自定义指令毫无用处。
   - 一位成员分享了旨在塑造“辱骂和淫秽喷子”人格的 Custom Grok 指令，但反馈称其“不起作用”，其他用户也反映了同样的情况。
- **Claude 3.7 表现惊艳，但 Projects 功能受挫**：一位用户宣称“ChatGPT 现在就是个笑话”，并发现 **Claude 3.7** “非常令人印象深刻”，同时 **Claude** 凭借 **200K** 的上下文窗口能更好地理解大文件中的语境。
   - 然而，另一位用户表示 **Claude** 的 Projects 功能毫无用处，抱怨称“最多只能上传两个文件”，且提示“内存已满”，称 Claude 被“过度炒作”了。
- **Dall-E 呈现合成生物学**：一位成员提示 **Dall-E** 生成一张“长出用于移植的心脏和肝脏的合成植物”图像，这些器官在透明薄膜内可见，并由转基因植物提供养分。
   - 初步结果侧重于心脏而非肝脏，促使用户通过更多关于**肝叶**的细节来优化 Prompt。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Llama 模型将 ZIP 压缩成 WAV 格式引发趣闻**：一位成员有趣地报告称，使用 **Llama** 模型压缩一个 **192 KB 的 ZIP 文件**，结果得到了一个 **48 KB 的无损 WAV 格式**。
   - 用户发现这种“混乱”是因为模型随后试图重新压缩该 **WAV** 以使其更小，特别提到了 **r1-1776-distill-llama-70b** 模型。
- **GRPO 训练：推理需要更多步数？**：用户讨论了使用 **GRPO** 对 **Qwen2.5-14B-instruct** 进行 **LoRA** 训练所需的必要训练步数，强调降低 Loss 以获得更好的推理能力。
   - 建议包括分配约 **24 小时**或 **700-1200 步**，并强调收敛情况取决于模型，如 [Unsloth 文档](https://docs.unsloth.ai/basics/reasoning-grpo-and-rl) 中所述。
- **GCC 编译器导致 VLLM 运行困难**：一位用户在本地使用 **meta-Llama-3.1-8B-Instruct** 运行 **GRPO** 教程时遇到了与 **GCC 编译器** 相关的 **RuntimeError**。
   - 尽管尝试通过 **conda** 安装 **GCC**，问题依然存在，且由于学校 HPC 的安全原因，该用户被限制使用 **apt-get**。
- **字符串替换：代码编辑策略的成功？**：成员们辩论了使用字符串替换进行代码编辑的有效性，一位成员认为这通常是“垃圾”。
   - 然而，另一位成员报告称在针对字符串替换微调 **Qwen 2.5** 方面取得了成功，特别是当模型在进行替换前可以访问整个文件时。
- **Claude 3.5 Sonnet 以 SOTA 成绩横扫基准测试**：Anthropic 的 Claude 3.5 Sonnet 在 [SWE-bench Verified](https://www.anthropic.com/research/swe-bench-sonnet) 上达到了 **49%**，超过了之前 SOTA 模型的 **45%**。
   - 成员们引用了 [Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)（惨痛的教训）：*利用计算能力的通用方法最终是最有效的*。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Web UI 重写功能故障**：用户报告 **Perplexity Web UI 的重写功能** [已损坏](https://discord.com/channels/1047197230748151888/1047649527299055688/1346323450892914758)，无论选择什么模型，始终默认为 *pplx_pro*。
   - 一些用户遇到了提示词重复问题，并标记了 <@883069224598257716> 寻求支持，这表明重写工具存在严重问题。
- **Claude 3 模型混淆问题持续存在**：用户不确定 **Perplexity** 的模型指示器是否准确反映了正在使用的模型，质疑在选择 **Claude** 时，接收到的是 **Claude 3.7 Sonnet** 还是 **Claude 3 Opus**。
   - 一些人注意到 **Pro Search** 会用 *Sonar* 覆盖所选模型，导致所选模型与实际使用的模型之间存在差异。
- **Perplexity API 困扰 Obsidian Web Clipper**：**Perplexity API** 与 **OpenAI** 标准的部分不兼容给 **Obsidian Web Clipper** 等工具带来了问题。
   - 该 API 要求在用户消息之间必须有一个 assistant 消息，而 **OpenAI** 中没有此要求，这阻碍了 **Obsidian Web Clipper** 发布连续用户消息的能力。
- **Deepseek 生成有争议的宣传内容？**：用户分享了一张据称由 **Deepseek** 生成的 [图片](https://cdn.discordapp.com/attachments/1047649527299055688/1346080726440742922/Screenshot_2025-03-03-14-54-53-25_0657b24d13eede56f839941c193b0cfe.jpg?ex=67c78b9e&is=67c63a1e&hm=407d3748b9e6b7ecb5ceba3c0aa4c347422820692953294384afbe593c64743c)，社区认为这是具有政治偏见的宣传。
   - 另一位成员对该图片不以为然，断言 *“你是假的 Deepseek。真正的 Deepseek 不谈论西方事务。”*

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Phi-3 微调面临障碍**：一位成员正在使用配备 A100 的 Colab Pro 对 [Phi-3](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) 进行多模态微调，但被提醒此类微调需要 *6 台以上的 A100 并运行约 2 周*。
   - 另一位成员补充说，[*只要有积极的态度和可靠的项目，QLora 和 Peft 可以让一切皆有可能*]。
- **Dataset Viewer 出现错误**：用户建议修复 **Dataset Viewer 错误**，以兼容各种库和 SQL，从而提高数据的可发现性。
   - 另一位用户预先表示感谢，并开玩笑地要求额外提供 **120 万行** 的高质量数据集。
- **Hugging Face 通过新 VAE 降低延迟**：Hugging Face 在 **SD v1**、**SD XL** 和 **Flux** 的 **Remote VAE Decode 端点**上部署了代号为 **honey** 的代码，将延迟降低了高达 **10倍**，这通过 [Hybrid Inference](https://huggingface.co/docs/diffusers/main/en/hybrid_inference/overview) 赋能了本地 AI 构建者。
   - **Hybrid Inference** 是**免费的**，完全兼容 Diffusers，并且对开发者友好，具有简单的请求和快速的响应，**VAE Encode** 即将推出。
- **Smol Agents 测验引发挫败感**：一位成员对 **Smol Agents 测验**表示挫败，理由是要求不明确，尽管多次尝试，得分仍为 **0.0/5**，并参考了 [测验的 app.py 文件](https://huggingface.co/spaces/agents-course/unit2_smolagents_quiz/blob/main/app.py)。
   - 该成员指出需要“挖掘错误日志”才能了解工具和模型所需的**确切提供商**。
- **Lambda Go Labs：AI 学习与构建**：[Lambda Go Labs](https://huggingface.co/organizations/labs-lambda-go/share/veuzHcLLGKoMrDdpMPltGTSMfPlSyOoNtN) 是一个专注于 **AI 学习、构建和研究**的社区。
   - 该社区提供实践经验、分享作品的机会，并为资深专业人士和新入门者提供支持网络。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 排行榜工具对决**：**Aider leaderboard** 现在除了对 AI 模型进行基准测试外，还加入了 **Claude Code** 等工具，将它们作为主要的编程助手进行评估。
   - 一位用户主张建立一个类似于 **SWE Benchlets** 的工具无关基准测试，以促进编程工具和模型之间更广泛的比较。
- **Anon-Kode 修改版 Claude Code**：**Claude Code** 的一个修改版本，被称为 **anon-kode**，由提取源代码的同一位开发者发布（[原推文链接](https://x.com/dnak0v/status/1894254049802744188)），现在已兼容 **OpenAI APIs**（[推文链接](https://x.com/dnak0v/status/1896652107857748086)），并可在 [GitHub](https://github.com/dnakov/anon-kode) 上获取。
   - 虽然有很多需要修复的地方，但你可以使用任何支持 OpenAI 风格 API 的服务。如果你够大胆，可以尝试一下。
- **Gemini 2.0 Pro 撞上上下文限制墙？**：一位用户报告称，在 Aider 中使用大上下文窗口时，`gemini/gemini-2.0-pro-exp-02-05` 模型出现了 `RESOURCE_EXHAUSTED` 错误。
   - 相比之下，`gemini-2.0-flash-thinking-exp-01-21` 模型运行顺畅；该用户询问了如何最大限度地利用 Pro 模型的上下文窗口。
- **Aider 实现了 Git Diff 愿望**：一位用户请求 Aider 直接在文件内部使用 **git diff 语法**（例如 `<<<<<< branch`, `======`, `>>>>>>> replace`）进行编辑。
   - 目前，Aider 在终端显示 diff，但用户寻求在接受更改前进行文件内编辑；其他用户指出这可能需要一个 fork，或者使用外部 diff 工具。
- **Grok 的调试优势**：成员们注意到，虽然 **Grok** 在调试方面表现出色，但在代码创建任务（如添加新功能）中，**O3 mini high sonnet** 可能会超越它。
   - 他们观察到 **Claude 3.7** 有时会引入多余的元素，而 **deepseek-chat** 配合 **O1 Pro** 已被证明是高度可靠的编辑器，准确率接近 95%。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **视觉模型仍青睐 Attention**：尽管存在 **MLP-Mixer** 等替代方案，基于 attention 的 **ViTs** 仍然是视觉模型的 **SOTA** 选择。
   - 一位成员对 **MLP-Mixer** 的相对利用不足提出了疑问，详情见 [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)。
- **SRAM 的缓存特性揭秘**：寄存器、共享内存和缓存是基于 **SRAM** 构建的芯片/软件级属性，未分配的共享内存会变成 **L1 cache**。
   - 虽然在 Triton 中缺乏直接的缓存层级控制（**L1/L2**），但 `tl.load` 中的 `cache_modifier` 可以指定 L1 或 L2 命中，其中 `cg` 专门针对 **L2**。
- **CUDA 架构查询得到 Torch 解答**：为了确定 CUDA 编译的 `--arch=` 参数，建议使用 **PyTorch** 的 `torch.cuda.get_device_capability()`，另一种替代方案是 `nvidia-smi --query-gpu=name,compute_cap --format=csv`。
   - 第二种方案避免了对 **PyTorch 依赖** 的需求，且 **CUDA Runtime API** 可以根据指定标准以编程方式选择最佳设备，如 [文档](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0) 所示。
- **Tilelang kernel 比 Flash-MLA 运行更快**：一位成员夸赞道，**80 行 tilelang kernel 代码** 即可达到 **deepseek flashmla 95% 的性能**，在 H100 上比 **Triton 快 500%**，并附带了 [GitHub 仓库](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_mla) 链接。
   - 另一位成员表示希望有一个 **MLA 排行榜**，或许可以从 **bitnet 小组** 改组而来。
- **FA3 需要 Absmax 进行量化**：虽然 **FA3** 现在可以工作，但它表现出的量化误差明显高于基础的 **absmax quantization**，这表明需要进行策略性调整。
   - 提议在 **Hada transform** 之后应用 **absmax quantization**，特别是针对 'v'，以减轻大激活值带来的分布外（out-of-distribution）问题。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **旅游应用兴起，拯救旅游 Reels**：一款应用应运而生，旨在解决无休止保存旅游 Reels 视频和数小时手动调研的问题。它使用 **AI agents** 直接从 [https://thatspot.app/](https://thatspot.app/) 上的旅游 Reels 中自动提取**地点、价格范围、预订要求、预订链接和营业时间**等数据。
   - 该应用通过利用 **AI agents** 处理旅游 Reels，自动提取提到的每个地点，从而实现手动调研过程的自动化，简化行程规划。
- **Google Flash 2.0 出现 502 错误**：一位用户报告在对 **Google Flash 2.0 和 Flash 2.0 Light 模型**进行推理时出现 **502 错误**，错误信息为 *"Provider returned error"*。
   - 该错误表明 Google 遇到了内部问题。
- **OpenRouter 的 Sonnet 遭遇速率限制**：一位用户询问了 **Claude 3.7 Sonnet** 在 **RPM (Requests Per Minute)** 和 **TPM (Tokens Per Minute)** 方面的速率限制（Rate Limits）。
   - 一位成员澄清说 OpenRouter 不会对每个用户施加特定的速率限制，并指向了 [Anthropic 的速率限制文档](https://docs.anthropic.com/en/api/rate-limits)和 BYOK 设置（[OpenRouter 集成设置](https://openrouter.ai/settings/integrations)）。
- **OpenRouter API Key 让 VS Studio 报错**：尽管账户资金充足，一位用户在 **VS Studio** 中通过 **RooCode** 使用 OpenRouter API Key 时遇到了 **401 Authentication Failure**。
   - 建议包括验证 API Key、在 RooCode 中选择 OpenRouter 作为 API 提供商，并确保 base URL 正确，参考了[此教程](https://www.vincentschmalbach.com/adding-models-to-cursor/)。
- **BYOK Azure 模型期待接入 OpenRouter**：一位用户询问在 OpenRouter 中对 **Azure 模型**使用 BYOK (**Bring Your Own Key**)，寻求为微调模型提供统一的 API。
   - 一位成员澄清说，目前仅支持 `/models` 端点中列出的模型，不包括 BYOK 模型，并建议在集成设置中使用 OpenAI API Key 代替。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 发布 Python 和 TypeScript 版 SDK**：LM Studio 发布了适用于 **Python** ([`lmstudio-python`](https://github.com/lmstudio-ai/lmstudio-python)) 和 **TypeScript** ([`lmstudio-js`](https://github.com/lmstudio-ai/lmstudio-js)) 的软件开发工具包，采用 **MIT 许可证**，允许开发者在自己的代码中调用 LM Studio 的 AI 能力。
   - 这些 SDK 支持 **LLMs**、**embedding 模型**以及 **agentic 工作流**，其特色是提供了用于自主任务执行的 **`.act()`** API，相关文档已在其各自页面（[`lmstudio-python`](https://lmstudio.ai/docs/python) 和 [`lmstudio-js`](https://lmstudio.ai/docs/typescript)）上线。
- **LM Studio “设备不支持”错误困扰用户**：在 LM Studio 更新后，有用户反映遇到 `Failed to load model` 错误，并提示 `Unsupported device`。建议尝试 [**调整 GPU offloading** 或 **线程池大小**](https://cdn.discordapp.com/attachments/1110598183144399061/1345995998471651348/image.png?ex=67c7e575&is=67c693f5&hm=a02e91bb46a53c67002d1787ae998888664e844574cba61a9e626bbe96a1ff6a&)。
   - 该错误可能与影响显存占用的 **上下文长度（context length）** 有关；左侧数字代表模型在 **聊天历史** 中已使用的 **tokens** 数量，右侧数字则是 **上下文限制**。
- **Llama.cpp 不支持 Diffusion 模型架构**：用户反馈在加载 Diffusion 模型时出现 `error loading model architecture: unknown model architecture: 'sd3'` 错误。官方澄清 [**llama.cpp 不支持图像/视频/音频生成模型**](https://cdn.discordapp.com/attachments/1110598183144399061/1345997397091811448/image.png?ex=67c7e6c2&is=67c69542&hm=93ffdc9a858570aeb67c46cc86eb5a391f447d55cbdb54b94b587bf29f609cd0&)。
   - `llama.cpp` 对视觉模型的支持尚不明确，目前存在缺乏 **Llama 3.2 vision** 或 **Pixtral vision 支持** 的担忧，不过一些人认为 **UI-TARS 的修复** 会有所帮助。
- **Pseudollama 填补 OLLAMA 缺口**：成员们讨论了 LM Studio 的端点是否与接受 OLLAMA 端点的应用兼容。得到的回答是默认情况下无法直接工作，但 [**Pseudollama**](https://github.com/verbiate/Pseudollama) 可以桥接这一差距。
   - 作者提到，*这完全是凭感觉编写的代码（vibe coded），所以可能到处都是低级问题，但它确实能跑通。*
- **AMD 需要在 GPU 领域展开竞争**：成员们讨论了 **AMD** 或 **Intel** 是否能在 **ML** 流水线和框架中变得可行，从而与 **CUDA** 竞争。
   - 一些成员认为，如果 **AMD** 提高市场份额，他们会更有动力投资其 **GPU** 计算部门；而真正的悬念在于 **AMD** 能否从芯片代工厂争取到产能，因为 Nvidia 目前占据上风。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous API 定价讨论**：成员们讨论了 Nous 可能会为其模型推出 API 以获取收入，推测定价约为 **$0.8/M tokens**，每天可能产生 **$800-1600** 的收益。
   - 建议包括针对专业模型将定价设为接近 **$1/M 输入 tokens** 和 **$3/M 输出 tokens**，目前正在努力实现这一目标。
- **LLM 在 CUDA Kernel 生成方面表现不佳**：成员们一致认为，虽然 **LLM** 可以生成正确的 **CUDA 语法**，但很难独立生成高性能的 **CUDA kernels**。
   - 最佳方案是将 **硬件和计算图数据** 与 LLM 结合，可能通过 **知识图谱或 GNN** 实现，并辅以密集的 **GPU profiling**。
- **Logic-RL 通过基于规则的 RL 提升推理能力**：[Logic-RL 论文](https://arxiv.org/abs/2502.14768) 探讨了在大型推理模型中应用 **基于规则的强化学习 (RL)** 的潜力，灵感源自 **DeepSeek-R1**。
   - 这个 **7B 模型** 仅在 **5K 个逻辑问题** 上进行训练，就在 **AIME 和 AMC** 等具有挑战性的数学基准测试中展现出了泛化能力。
- **Runway 发布通用世界模型**：Runway 推出了 [通用世界模型 (General World Models)](https://runwayml.com/research/introducing-general-world-models)，旨在创建能够构建环境内部表示以模拟未来事件的 AI 系统。
   - 他们的目标是表示和模拟广泛的场景和交互，超越 **电子游戏** 或 **驾驶模拟** 等局限的场景。
- **Qwen2.5-Math-1.5B 模型在 Longcot 上的困境**：一位用户发现 **Qwen2.5-Math-1.5B** 模型在处理 **longcot 示例** 时存在困难，在配置数据集结构和 **GRPOTrainer** 方面需要帮助。
   - 他们链接了自己的 [Kaggle notebook](https://www.kaggle.com/code/umangkaushik/qwen2-5-3b-limo)，寻求解决这些问题的指导。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Unitree 发布开源宝库**: **Unitree Robotics** 已开源多个仓库，可通过 [其 GitHub](https://github.com/unitreerobotics) 访问。
   - 此举为机器人领域的协作开发和创新开启了可能性。
- **GPT-4.5 登上 Arena 宝座**: **GPT-4.5** 已夺得 Arena 排行榜所有类别的榜首，包括 **Multi-Turn**、**Hard Prompts**、**Coding**、**Math**、**Creative Writing**、**Instruction Following** 和 **Longer Query** [(来源)](https://x.com/lmarena_ai/status/1896590146465579105)。
   - 最新的评分巩固了 **GPT-4.5** 目前作为 State of the Art 的地位。
- **Anthropic 的天文级增长仍在继续**: **Anthropic** 以惊人的 **615 亿美元** 投后估值获得了 **35 亿美元** 的融资，由 **Lightspeed Venture Partners** 领投 [(来源)](https://www.anthropic.com/news/anthropic-raises-series-e-at-usd61-5b-post-money-valuation)。
   - 这笔资金旨在推进其 AI 系统的开发，加深对其功能的理解，并推动国际增长。
- **Grok3 定价结构浮出水面？**: 据 [这条推文](https://x.com/swishfever/status/1896539732471058640) 报道，潜在泄露的 **Grok3 定价** 详情显示，输入成本为 **$3.50/百万**，缓存输入为 **$0.875/百万**，输出为 **$10.50/百万**。
   - 泄露的定价模型为在各种应用中利用 **Grok3** 的潜在成本提供了见解。
- **人类数据对于现实世界 AI 仍然至关重要吗？**: 一篇博客文章 ([https://www.amplifypartners.com/blog-posts/annotation-for-ai-doesnt-scale](https://www.amplifypartners.com/blog-posts/annotation-for-ai-doesnt-scale)) 认为，**人类数据** 对于构建真正有用的 AI 产品仍然必不可少。
   - 这一观点挑战了仅靠合成数据就能推动模型性能大幅提升的看法。



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Claude 攻克编程挑战**: 一位成员报告使用 **Claude** 和 [Cursor](https://cursor.sh/) 完成了 [这个 GitHub Pull Request](https://github.com/eslint-stylistic/eslint-stylistic/pull/707) 中 **95%** 的工作，该 PR 涉及 **细粒度配置选项**。
   - 该成员正在处理 `object-property-newline` 规则，通过添加对细粒度配置选项的支持，允许开发者为不同的节点类型指定不同的行为。
- **应对棘手的时间段**: 一位成员最初考虑就 **Joscha Bach** 进行演讲，但不确定这是否是最终主题。
   - 另一位成员提出，如果没有安排其他演讲，他可以在 `<t:1741046400:F>` 时间段进行演讲，并向感兴趣的人提供进一步建议。
- **Elsagate 再次爆发**: 一位成员分享了一个名为“**Elsagate 3.0 比我们想象的更糟糕**”的 [YouTube 视频](https://www.youtube.com/watch?v=RjOybKOm2Tc)，并警告称该视频 **不适合儿童观看**。
   - 另一位成员回应道：“这太可怕了。”



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **财务报表进入 NotebookLM**: 一位成员询问是否可以将 **财务报表** 加载到 **NotebookLM** 中进行分析，以实现财务分析自动化。
   - 这表明了将 **NotebookLM** 用于专业任务的兴趣。
- **播客长度辩论和时间线需求被提出**: 成员们对 **播客长度** 和重要话题的覆盖范围表示担忧，并引用了 [此处](https://cdn.discordapp.com/attachments/1124403655819415592/1346024489846046760/US_Dept._of_State_v._AIDS_Vaccine_Advocacy_Coalition__Supreme_Court_Application._AIDS_Vaccine_Advocacy_Coalition__Supreme_Court_Application.mp3?ex=67c7573e&is=67c605be&hm=ac8945e981f8673fb037a56d5b57d3dab559a89bba554fb5db6fb00dad7c9c6e&) 发现的 **最高法院申请**。
   - 一位成员要求在播客免费版中添加 **时间线**，而另一位成员分享了 [一个 NotebookLM 播客示例](https://notebooklm.google.com/notebook/d2d16919-2f3c-484f-a6e6-626f6aca7845/audio)。
- **动态文档，一个缺失的功能**: 成员们好奇 **NotebookLM** 是否可以从 **Google Docs** 等来源动态更新，用于追踪家具尺寸等用例。
   - 由于该功能不是自动的，引发了关于变通方案和功能需求的讨论。
- **Notebook 分享故障已解除！**: 一位用户报告在与 Gmail 个人账户分享 Notebook 时出现服务器错误，具体为 *"You are not allowed to access this notebook"*。
   - 当用户发现接收者有一部新手机未正确配置其 Gmail 账户时，问题得到了解决。

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **脸部复制替代方案出现**：成员们讨论了复制脸部的最佳方法，一些人更倾向于使用 **ControlNet 中的 reference only**，而另一些人则推荐 **Reactor Faceswap** 作为 **IP-Adapter** 的更佳替代方案。
   - 社区共识似乎更倾向于 **ControlNet**，因为它具有多功能性。
- **Reforge 的 AMDGPU 支持仍不明确**：一位用户报告了关于 **Reforge** 支持 **AMDGPU** 的矛盾信息，因为它在 **Stability Matrix** 上被提及，但在 **GitHub 页面**上却没有。
   - 另一位用户尝试使用 **Zluda** 导致 PC 死机，这引发了对 **Stability Matrix** 准确性的怀疑，并建议使用 **Matrix 之外的 UI**。
- **DirectML 与 Reforge 不兼容**：一名成员在 **Zluda** 失败后尝试将 **Reforge** 与 **DirectML** 结合使用，但未获成功。
   - 讨论了 **Lshqtiger** 可能推出的 **Reforge for AMD** 分支。
- **CivitAI 提供免费图像生成**：成员们讨论了将 **CivitAI** 作为图像生成请求的平台，指出它提供一些初始积分和每日 25 个可累积的免费积分。
   - 使用该平台的成本取决于所选的模型。
- **本地图像生成需求详情**：一位成员询问了本地创建图像的要求；另一位成员回答建议使用显存 (**VRAM**) 约为 **6-8GB** 的 **GPU**，以及 <#1002602742667280404> 中提到的其他资源。
   - 另一位成员分享了 **CivitAI** 在线生成的链接作为替代方案。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **ReasonableLLAMA-Jr-3b 寻求反馈**：一位成员请求对其 [ReasonableLLAMA-Jr-3b](https://ollama.com/adeelahmad/ReasonableLLAMA-Jr-3b) 模型提供反馈，这是一个基于 **Atom of Thoughts (AoT)** 论文概念，在 **LLAMA 3.2 3B** 上使用 **GRPO** 训练的推理模型。
   - 该模型在 **Gym** 环境中使用 **MLX** 编写了一个自定义的基于 **GRPO** 的 **Agent**，其中推理过程中的每个状态转换都是一个独立的、原子级的问题，如 [Atom of Thoughts for Markov LLM Test-Time Scaling](https://arxiv.org/abs/2502.12018) 中所述。
- **循环 LLM 推理：代价过高？**：成员们辩论了循环 **LLM** 推理是否实用，因为这种推理需要相当于 **32B** 参数模型的计算量才能达到 **7B** 模型的性能。
   - 提出的核心问题是：*为什么不直接训练一个 32B 参数的模型*，并使用 **early exit**、**mixture of depths** 或 **speculative decoding** 来实现更廉价的推理？
- **在 Harness 中排查 'trust_remote_code' 问题**：一位用户询问 `trust_remote_code` 是否在 `lm-evaluation-harness` 中被无条件设置，并指向了 [GitHub 仓库中的特定行](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/__main__.py#L376)。
   - 一位成员澄清说，只有在提供了 `--trust_remote_code` 参数时才会设置 `trust_remote_code`，并引用了 [代码的相关部分](https://github.com/EleutherAI/lm-evaluation-harness/blob/14b0bd26956609b2ee50987299dfa34223fa23b8/lm_eval/__main__.py#L367)。
- **揭秘 Dataset Kwargs 路径**：一位用户询问设置 `trust_remote_code` 是否会在加载本地数据集时覆盖 `dataset_kwargs`。
   - 一位成员澄清说，`dataset_kwargs` 会被传递给 **Harness** 内部的 `datasets.load_dataset(...)`，并链接到了 [代码的相关部分](https://github.com/EleutherAI/lm-evaluation-harness/blob/14b0bd26956609b2ee50987299dfa34223fa23b8/lm_eval/api/task.py#L930)。
- **用户报告数据集生成错误**：一位用户报告在运行 `lm_eval` 时遇到 **数据集生成错误**，其配置指定了 `dataset_path: json` 且 `data_dir` 包含 `train.jsonl`、`validation.jsonl` 和 `test.jsonl`。
   - 作为回应，一位成员建议使用 `load_dataset` 手动测试数据集加载，并尝试为数据目录使用绝对路径。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP Terraform Registry 面临问题**：用户报告在启用系统级代理时，**terraform-registry-mcp** 和 **aws-mcp server** 无法正常运行，特别是在使用 **Claude desktop** 和 **Cline** 时，会导致 *mcp-server-fetch* 错误。
   - 该问题似乎与代理设置干扰了服务器获取必要资源的能力有关。
- **多 Agent MCP 架构兴起**：一位成员探索了为**多 Agent 系统**实现 **MCP**，引用了 AI Engineering Summit 上的 Anthropic 工作坊，并分享了工作坊中的[一张图片](https://cdn.discordapp.com/attachments/1312302100125843476/1346079179698999338/image.png?ex=67c78a2d&is=67c638ad&hm=50e766c247227751ff74caae3c16387ec1e88639ce75a481c44345cee448c6e5)。
   - 他们正在构建一个让 Agent 跨设备协作的框架，并考虑采用 MCP，灵感来自 **BabyAGI** 和 **Stanford generative agents** 等示例。
- **Fast Agent 框架备受关注**：一位成员分享了他们的项目 [fast-agent on GitHub](https://github.com/evalstate/fast-agent)，用于*定义、提示和测试支持 MCP 的 Agent 及工作流*。
   - 该框架允许为每个 Agent 配置一组独立的 MCP server，并可以被其他 Agent 作为 tool 调用。
- **Node 版本问题困扰 Claude 用户**：用户报告在 **Claude desktop** 中使用 **fastmcp** 时遇到 *Cannot find package 'timers'* 错误。
   - 问题追溯到 **Claude** 正在使用的过时的 **Node v14** 版本。
- **MCPHub.nvim 助力 Neovim**：新的 [**MCPHub.nvim** 插件](https://github.com/ravitemer/mcphub.nvim) 发布，旨在协助在 **Neovim** 中管理 MCP server，并提供智能服务器生命周期管理以及与 **CodeCompanion.nvim** 集成进行 AI 聊天等功能。
   - 该插件可通过单个命令 (`:MCPHub`) 安装，为 MCP server 管理提供了流线化的设置过程。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Ash 框架生态受到关注**：一位成员为一个项目推荐了 **Ash 框架**，并指向了 [ash-project/ash_ai](https://github.com/ash-project/ash_ai) GitHub 仓库。
   - 他们强调了 [instructor_ex](https://github.com/thmsmlr/instructor_ex)，它为 Elixir 中的 LLM 提供结构化输出，并引导用户前往 [Ash Discord 社区](https://discord.gg/w3AXeARR2p)寻求指导。
- **异步支持计划启动**：一位成员询问了 DSPy 中**全异步支持**的动机和预期的性能提升，并链接了[另一个 Discord 邀请链接](https://discord.gg/8SfMdAeu)。
   - 一位核心贡献者宣布了使**异步支持**成为原生功能的意图，并请求通过 [GitHub issues](https://github.com/stanfordnlp/dspy/issues) 提交功能需求，以防在 Discord 中被遗漏。
- **LangProBe 基准测试程序组合**：一篇新论文 [LangProBe: a Language Programs Benchmark](https://arxiv.org/abs/2402.20315) 评估了 **DSPy 程序组合**和**优化器**对不同任务的影响，同时探索了成本/质量的权衡。
   - 正如其 [X/Twitter 帖子](https://x.com/LakshyAAAgrawal/status/1896628734553403728)所述，论文表明在优化后的程序中，较小的模型可以以更低的成本超越较大的模型。
- **Minions 准备在成本优化方面占据优势**：一位成员表示，刚发布的 **LangProBe 论文**为基准测试他们实现的 **minions 功能**提供了一个很好的基准，并引用了他们已关闭的 [pull request](https://github.com/stanfordnlp/dspy/pull/7891)。
   - 该成员添加了 **MinionsLM 和 StructuredMinionsLM** 用于智能 LM 路由，并强调了该论文与**成本优化**的直接相关性。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **AgentWorkflow Context**：一位成员询问了 **AgentWorkflow** 中 **Context** 与 **Chat History** 的区别。
   - 另一位成员回答道：*聊天历史记录包含在 context 之中*。
- **LlamaIndex 集成 MCP 支持**：一位用户询问 LlamaIndex 对 **MCP** 的支持情况，另一位成员确认已支持并提供了 [示例 notebook](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-mcp/examples/mcp.ipynb)。
   - 该 notebook 演示了如何在 LlamaIndex 中使用 **MCP**。
- **LlamaParse 最新模型支持 Agent 解析**：'Parse With Agent' 模式现在支持 **AnthropicAI Claude Sonnet 3.7** 和 **Google Gemini 2.0 Flash**，增强了表格解析和跨页一致性（[公告](https://t.co/V6pwuxm9IO)）。
   - 这些更新将提高解析复杂文档的准确性和可靠性。
- **需要 PII 处理？咨询 LlamaIndex！**：一位成员正在寻求付费和开源方案，用于在将 **PDF** 和**图像**发送给 **LLM** 之前，对其中的**个人身份信息 (PII)** 进行脱敏处理。
   - 这一请求凸显了 **LLM** 应用中对强大 **PII 脱敏工具**日益增长的需求。
- **由于缺少 Checkpoint 功能，Windsurf 表现不佳**：一位成员指出 Windsurf 缺少 **checkpoint** 功能，并提到尽管多次尝试编码以及对文件/工作区进行操作，仍无法回滚到之前的状态。
   - 该成员附带了一张 [图片](https://cdn.discordapp.com/attachments/1100478495295017063/1346342017042747472/IMG_0299.png?ex=67c7d636&is=67c684b6&hm=0f6ba7693b819f1c004632629bd657658b5eee665327600e857dbdcc903e76bb)，展示了他们尝试将文件拖放到标签菜单中以寻求访问之前 checkpoint 的努力。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **AI 尚未完全取代程序员**：一篇 [O'Reilly 文章](https://www.oreilly.com/radar/the-end-of-programming-as-we-know-it/) 指出 **AI 工具** 正在演进编程方式，类似于从早期物理电路编程至今的历史性变革。
   - 成员们表示赞同，指出 **LLM** 加速了学习过程，这与过去人们抱怨从 StackOverflow 复制代码的情况类似。
- **资深工程师主导 AI 输出**：资深工程师凭借专业知识有效地引导 **AI 的输出**，在使用 **Cursor** 或 **Copilot** 等工具时，能防止产生不可维护的代码。
   - 虽然 **AI** 加快了实现速度，但资深工程师确保了代码的可维护性，而这通常是初级工程师所缺乏的技能。
- **Anthropic 获得巨额融资**：[Anthropic](https://www.anthropic.com/news/anthropic-raises-series-e-at-usd61-5b-post-money-valuation) 获得了 **35 亿美元** 融资，投后估值达到 **615 亿美元**，由 **Lightspeed Venture Partners** 领投。
   - 这笔投资将用于支持 AI 系统的推进、增强对其功能的理解以及支持全球扩张。
- **Python 开发者寻求 Stagehand 类工具**：在听完关于 **Browserbase** 的 Latent Space 播客后，一位成员在寻找 Python 中类似于 **Stagehand** 的自修复浏览器工作流工具。
   - 另一位成员推荐了 [stagehand-py](https://pypi.org/project/stagehand-py/)，并指出 *“它还在开发中（wip）”*。
- **代码对决：Cursor 击败 Claude Code**：成员们将 [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview) 与 **Cursor** 进行了对比，**Cursor** 因其回滚能力更受青睐。
   - 反馈显示 **Claude Code** 在专注度方面存在问题，会添加不必要的代码，成本更高，且在代码编辑速度上不如 **Cursor**。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 旨在建立公平的算力市场**：George Hotz ([@__tinygrad__](https://x.com/__tinygrad__/status/1896527413586366710)) 将 **tinygrad** 描述为一个*形式主义项目*，旨在以非泄漏抽象捕获 **Software 2.0**，目标是建立一个类似于 Linux 和 LLVM 的公平算力市场。
   - Hotz 预计到年底，**tinygrad** 在 NVIDIA 上的速度将在无需 CUDA 的情况下赶上现有的 torch CUDA 后端，并设想建立一个测试云，用于在 lambda 函数中租用 FLOPS。
- **Ops.CAT 速度悬赏面临 LLVM 重写问题**：一名成员报告了 **Ops.CAT 速度悬赏** 持续面临的挑战，特别是尽管已在计划中，但在尝试将其重写为 **LLVM** 时遇到困难。
   - 目前的 **Ops.CAT 操作** 具有由 **PAD**、**RESHAPE** 和 **BUFFER** 操作组成的复杂结构，其中 arg 代表要连接的两个张量。
- **RDNA2/RX6000 在 tinygrad 中的可用性咨询**：一位用户询问了 **RDNA2/RX6000/GFX1030** 在 tinygrad 中的可用性，报告在运行 `AMD=1` 时出现 `OSError: [Errno 22] Invalid argument`。
   - 另一位成员表示它应该可以在 Linux 上运行，并请求获取该操作系统错误的 trace 信息，该信息已在 [trace.txt 文件](https://cdn.discordapp.com/attachments/1068976834928193609/1346224035591360663/trace.txt?ex=67c76855&is=67c616d5&hm=2bd9866ee6a24e241f5b0b7264ad9d547acf0f7a912cb0c2049053c69e79d44b&) 中提供。
- **Intel Arc A770 与 OpenCL 配合良好**：一名成员确认 **Intel Arc A770** 确实可以与 tinygrad 配合使用。
   - 建议通过设置 `GPU=1` 来利用 **OpenCL 后端**。



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Sutton 深入探讨编程 Agent**：特邀演讲嘉宾 **Charles Sutton** 在 [第 5 讲](https://www.youtube.com/live/JCk6qJtaCSU) 中介绍了“编程 Agent 与用于漏洞检测的 AI”。
   - 该讲座探讨了使用 **LLM Agent** 执行计算机安全任务（如寻找软件漏洞），并讨论了 **LLM Agent** 的设计问题。
- **DeepMind 研究员荣获奖项**：**Google DeepMind** 的研究科学家 **Charles Sutton** 的机器学习研究主要受代码生成、软件工程、编程语言和计算机安全应用的启发。
   - Sutton 在软件工程方面的工作曾获得两项 **ACM Distinguished Paper Awards** (FSE 2014, ICSE 2020) 和一项 **10-year Most Influential Paper award** (MSR 2023)。
- **测验发布日期揭晓**：一位用户询问每周何时发布测验，另一位用户回答说他们*通常尝试在周三或周四发布*。
   - 该问题是在 **mooc-questions** 频道中提出的。
- **讲座音频问题困扰**：一名成员在 **mooc-lecture-discussion** 频道报告由于音频问题无法听到讲座中的提问，请求现场人员协助。
   - 一名工作人员对讲座期间的音频问题表示歉意，并承诺今后会提醒演讲者重复所有问题。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 图像嵌入问题消失**：一位用户报告了使用 Cohere 嵌入图像时的问题，但随后确认该问题已神秘地自行解决。
   - 另一位成员仅确认了问题的解决，未作进一步评论。
- **Cohere 调查棘手的 504 错误**：一位 Cohere 成员提到，虽然他们没有观察到 **504 错误** 的激增，但他们注意到**极慢的请求**可能是潜在原因。
   - 该成员计划进一步调查慢请求的来源，并感谢用户的提醒。



---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **`owned` 通过 Pull Request 变为 `own`**：一名成员提交了一个 [pull request](https://github.com/modular/max/issues/4048)，建议将 `owned` 重命名为 `own`，以保持与 rest 参数约定的一致性。
   - 此次重命名旨在与既定的编码实践保持一致，并增强可读性。
- **社区会议征集演讲者**：定于一周后举行的下一次社区会议正在征集演讲者进行演讲或展示项目。
   - 有意向的个人应联系组织者以在议程中预留位置。
- **AWS GenAI Loft 举办 MAX Engine 活动**：一场名为 [Beyond CUDA: Accelerating GenAI Workloads with Modular’s MAX Engine, Hosted by AWS](https://lu.ma/2kkbh2iv) 的活动将在 AWS GenAI Loft 举行。
   - 该活动面向湾区观众，定于明天晚上举行。
- **SIMD DType 构造解析**：一次讨论澄清了 `SIMD[DType.uint8, 1](0).type` 会在编译时返回 dtype，并以 `var a = UInt8(0); alias dtype = __typeof(a).type` 为例。
   - 一名成员强调 `SIMD` 在其实现中包含了 [构造检查](https://github.com/modular/max/blob/main/mojo/stdlib/src/builtin/simd.mojo#L168)，这有助于确保有效性和类型安全。
- **参数注入优于全局变量**：针对有关使用全局变量的问题，一名成员断言，*如果你有时间的话*，注入参数通常是更好的选择。
   - 这一偏好符合代码可维护性和可测试性的最佳实践。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **基于步数的 Checkpointing 减少计算浪费**：成员们对**基于步数的 Checkpointing (step-based checkpointing)** 表示了兴趣，并确认正在实施该功能，以减轻因训练失败导致的计算资源浪费。
   - 该功能定期保存进度，减少中断带来的影响。
- **Torch 用户使用 Tensorboard 进行 Trace**：Torch 用户讨论了可视化 profiler trace 的策略，最初尝试使用 **Tensorboard**，但注意到 **PyTorch** 的某些插件功能已被移除。
   - 他们推荐使用 **PyTorch memory visualizer tool** 和 **Perfetto** 来进行内存和时间 trace，认为这足以追踪相关线索。
- **替代分析工具盛行**：讨论强调了 **PyTorch memory visualizer tool** 和 **Perfetto** 分别是内存和时间 trace 的可靠替代方案。
   - 这些工具在用户反映 **Tensorboard** 存在问题（特别是缺少某些 **PyTorch** 插件功能）后被提出。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Ollama vs GPT4All：哪个 Llama 更胜一筹？**：一位用户质疑为什么人们选择 **Ollama** 或 **Llama.cpp** 而不是 **GPT4All**，认为 **GPT4All** 开箱即用的功能使其成为更好的选择。
   - 该用户没有提供具体的比较指标细节，但强调了易用性是其核心优势。
- **GPT4All 界面将支持加泰罗尼亚语**：一位社区成员请求为 **GPT4All** 界面添加**加泰罗尼亚语 (Catalan)** 作为语言选项。
   - 该请求强调了社区中存在加泰罗尼亚语使用者，以及本地化支持的潜在益处。
- **发现安全漏洞，GPT4All v3.10.0 面临风险**：一位用户报告了 **GPT4All v3.10.0** 中的一个潜在漏洞，并询问了正确的报告方式。
   - 消息中未透露漏洞性质的细节，但建议尽快报告。

---

**MLOps @Chipro Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。

---

# 第 2 部分：频道详细摘要与链接

{% if medium == 'web' %}

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1345969905928769558)** (745 messages🔥🔥🔥): 

> `Cursor IDE, MCP, Landing Page Design, Model Performance, Repo Prompt`

- **Cursor 不稳定且存在 Bug**：用户报告在最新的 Cursor 版本中出现了 [**不稳定**](https://discord.com/channels/1074847526655643750/1074847527708393565/1346144838955667486)、**连接失败**以及 **Checkpoints 功能失效**的问题。
   - 一位用户表示，“*Cursor 目前的不稳定程度简直令人难以置信*”，许多人正在考虑切换到 **Windsurf** 和 **Trae AI** 等其他替代方案。
- **MCP 配置令人头疼**：成员们在 Cursor 中[配置 **MCP 服务器**](https://discord.com/channels/1074847526655643750/1074847527708393565/1346225879193354291)时遇到困难，特别是在 **Windows** 和远程 Ubuntu 工作区中，面临诸如“*客户端创建失败*”等问题。
   - 一位用户在设置 Firecrawl MCP 服务器时需要帮助，但最终成功运行了 Puppeteer 并表示：“*谢谢大家的帮助，我太笨了，抱歉*”。
- **3.7 并非完美**：用户在使用 [**Claude 3.7** 时遇到问题](https://discord.com/channels/1074847526655643750/1074847527708393565/1346143214571280384)，例如运行速度*极其缓慢*，且容易在没有错误提示的情况下中途停止请求。
   - 一位用户指出，“*3.7 目前非常不稳定*”，这导致许多人在处理重要任务时使用 Cursor 的“Ask”模式或切换回旧版本。
- **设计师深入研究落地页**：成员们分享了[使用 **Cursor** 生成的**落地页设计**](https://discord.com/channels/1074847526655643750/1074847527708393565/1346170580382126164)，并讨论了它们的美学效果和有效性。
   - 社区辩论了不同设计的优缺点，并参考了 [Linear](https://linear.app/)、[Framer](https://www.framer.com/)、[Magician Design](https://magician.design/) 和 [Webflow](https://webflow.com/) 等网站寻找灵感。
- **Repo Prompt 受到关注**：用户对 [Repo Prompt](https://discord.com/channels/1074847526655643750/1074847527708393565/1346338341037107300) 感到兴奋，称赞其**多文件编辑**能力和代码片段集成功能。
   - 社区还提到了用于调试的 [BrowserTools](https://browsertools.agentdesk.ai/installation)，以及 [PasteMax](https://github.com/kleneway/pastemax)（被描述为用于选择文件的 Repo Prompt 的“开源穷人版”）。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>

<li><a href="https://anysphere-binaries.s3.us-east-1.amazonaws.com/production/be4f0962469499f009005e66867c8402202ff0b7/darwin/arm64/Cursor-darwin-arm64.dmg">未找到标题</a>: 未找到描述</li><li><a href="https://x.com/pvncher/status/1894559704065409224?s=46&t=ggmESCIXF0nYw8_kshHz7A">来自 eric provencher (@pvncher) 的推文</a>: Apply 模式是 Repo Prompt 最好的部分之一，但由于它的呈现方式，似乎让人们感到畏惧。不过它非常强大！以下是你可以如何使用它...</li><li><a href="https://x.com/lmarena_ai/status/1896590150718922829?s=46">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>: GPT-4.5 在所有类别中全面领先，在 Multi-Turn 方面具有明显的领导地位。🥇 Multi-Turn💠 Hard Prompts💠 Coding💠 Math💠 Creative Writing💠 Instruction Following💠 Longer Query</li><li><a href="https://repoprompt.com/">Repo Prompt</a>: 未找到描述</li><li><a href="https://www.agentdesk.ai/prompts">AgentDesk</a>: 未找到描述</li><li><a href="https://tenor.com/view/kermit-gun-kermit-gun-gif-16355306064126846866">Kermit Gun GIF - Kermit Gun Kermit gun - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://youtu.be/-xqEABqaEuo">Elijah Wood 被问及是否戴假发</a>: 出自《指环王》系列电影演员的一次恶作剧采访。你戴假发吗？你戴过假发吗？你会戴假发吗？你什么时候会戴...</li><li><a href="https://browsertools.agentdesk.ai/installation">安装 - AgentDesk - BrowserToolsMCP</a>: 未找到描述</li><li><a href="https://x.com/tedx_ai),">来自 GitHub 的推文 - FixTweet/FxTwitter: 修复损坏的 Twitter/X 嵌入！</a>: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等功能 - FixTweet/FxTwitter</li><li><a href="https://g>>>">未找到标题</a>: 未找到描述</li><li><a href="https://tenor.com/view/dinosolostan-gif-22443651">Dinosolostan GIF - Dinosolostan - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.reddit.com/r/RealTesla/comments/1g9c0cu/roadster_reservation_refund/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://github.com/grahama1970/agent_tools/tree/main/src/agent_tools/method_validator">agent_tools/src/agent_tools/method_validator at main · grahama1970/agent_tools</a>: 通过在 GitHub 上创建账号，为 grahama1970/agent_tools 的开发做出贡献。</li><li><a href="https://github.com/mendableai/firecrawl-mcp-server">GitHub - mendableai/firecrawl-mcp-server: 官方 Firecrawl MCP Server - 为 Cursor, Claude 和任何其他 LLM 客户端添加强大的网页抓取功能。</a>: 官方 Firecrawl MCP Server - 为 Cursor, Claude 和任何其他 LLM 客户端添加强大的网页抓取功能。 - mendableai/firecrawl-mcp-server</li><li><a href="https://github.com/kleneway/pastemax">GitHub - kleneway/pastemax: 一个简单的工具，用于从仓库中选择文件并复制/粘贴到 LLM 中</a>: 一个简单的工具，用于从仓库中选择文件并复制/粘贴到 LLM 中 - kleneway/pastemax</li><li><a href="https://github.com/browserbase/stagehand">GitHub - browserbase/stagehand: 一个专注于简单性和可扩展性的 AI 网页浏览框架。</a>: 一个专注于简单性和可扩展性的 AI 网页浏览框架。 - browserbase/stagehand</li><li><a href="https://github.com/daniel-lxs/mcp-starter">GitHub - daniel-lxs/mcp-starter: 一个轻量级的 Go 应用程序，用于解析 JSON 配置文件并使用指定的环境变量执行命令。</a>: 一个轻量级的 Go 应用程序，用于解析 JSON 配置文件并使用指定的环境变量执行命令。 - daniel-lxs/mcp-starter</li><li><a href="https://linear.app/">Linear – 规划并构建产品</a>: Linear 简化了议题、项目和路线图。专为现代产品开发而打造。</li><li><a href="https://www.framer.com/">Framer: 受设计师喜爱的网站构建工具</a>: 设计、扩展并发布你的网站——无需代码。今天即可免费开始。</li><li><a href="https://magician.design/">Magician for Figma</a>: 一款由 AI 驱动的 Figma 神奇设计工具。</li><li><a href="https://webflow.com/">Webflow: 创建自定义网站 | 可视化网站构建工具</a>: 以可视化的方式利用代码的力量创建自定义的响应式网站。使用灵活的 CMS 和顶级托管服务设计并构建你的网站。免费试用 Webflow。</li><li><a href="https://yourdesignis.com/">Your Design is S**t</a>: 我帮助早期创始人将落地页转化为收入引擎。因为优秀的产品值得拥有能将访问者转化为客户的设计。</li><li><a href="https://www.youtube.com/watch?v=i1nLckJPRTs">The Musks: 当 3000 亿美元让家庭分裂时</a>: 当你想到 Musk 家族时，很可能会想到 Elon Musk —— 然而，Musk 家族的成员还包括...</li>

远超 Elon，每位成员都做出了贡献...</li><li><a href="https://www.reddit.com/r/NoStupidQuestions/comments/10k40fe/did_elon_musk_grow_up_privileged/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://www.biography.com/business-leaders/elon-musk">Elon Musk 回应纳粹礼类比：“他们需要更高明的卑鄙手段”</a>：这位 53 岁的人士周一在 Donald Trump 的总统游行中做出直臂手势，引发了愤怒。</li><li><a href="https://www.youtube.com/watch?v=-VfYjPzj1Xw)">观看：Elon Musk 在 Trump 就职庆祝活动中似乎行了法西斯礼</a>：亿万富翁 Elon Musk 周一在为总统 Donald Trump 举行的就职后庆祝活动上发表讲话时，行了一个看似法西斯礼的手势...</li><li><a href="https://en.wikipedia.org/wiki/Musk_family">Musk 家族 - 维基百科</a>：未找到描述</li><li><a href="https://www.oxfordstudent.com/2025/02/17/musks-background-story/">未找到标题</a>：未找到描述</li><li><a href="https://finance.yahoo.com/news/rich-elon-musk-during-every-130036338.html">Elon Musk 在他生命的每个十年里有多富有？</a>：根据福布斯，Elon Musk 目前是世界首富。他拥有巨额财富——约 2320 亿美元。但 Musk 并非一夜暴富。他的巨额财富是经过...
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1346319464932708382)** (1 条消息): 

> `Windows ARM support, Windsurf Next, Ubuntu 24.04, Claude 3.7 Sonnet, MCP Tools` 


- **Windsurf 增加 Windows ARM 支持**：截至本周末，Windsurf Next <:wsnext:1336821369685540914> 现已支持 **Windows ARM64**，可在此处[下载](https://codeium.com/windsurf/download-next)。
- **Ubuntu 24.04 错误已修复**：Windsurf 1.3.11 和 Windsurf Next 均已发布补丁，修复了由 **Ubuntu 24.04 上的权限错误**导致的崩溃（[更新日志](https://www.codeium.com/changelog)）。
- **Cascade 现已支持 Claude 3.7 Sonnet**：Windsurf Next 现已支持 **Claude 3.7 Sonnet**，每条消息消耗 **1.0 用户提示额度**，每次工具调用消耗 **1.0 flow action 额度**。
- **Windsurf 解决 JSON 中的 MCP 工具问题**：Windsurf 1.3.11 修复了 JSON 中格式错误的 **MCP** 工具，并提供了更好的 MCP 错误处理。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://codeium.com/windsurf/download-next">感谢下载 Windsurf Next</a>：Windsurf Next 是我们的实验性测试版，让早期采用者有机会在正式发布前测试新功能。</li><li><a href="https://codeium.com/changelog/windsurf-next">Windsurf Next 更新日志 | Windsurf 编辑器和 Codeium 扩展</a>：Windsurf Next 扩展的最新更新和变更。</li><li><a href="https://www.codeium.com/changelog">Windsurf 编辑器更新日志 | Windsurf 编辑器和 Codeium 扩展</a>：Windsurf 编辑器的最新更新和变更。
</li>
</ul>

</div>
  

---

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1346017626530254902)** (37 条消息🔥): 

> `Codeium Pro 问题与休眠（snoozing）、Codeium 中 Supercomplete 的可用性、Visual Studio Codeium 扩展版本、JetBrains 扩展问题` 


- **Codeium Pro 用户面临休眠问题**：一位用户报告称 **Codeium Pro** 会停止工作并自动进入休眠（snoozes）状态，需要手动重新启用，这是部分用户的已知问题。
   - 该用户发现使用扩展的 **pre-release version**（预发布版本）体验更好，尽管仍不理想。
- **Supercomplete 状态澄清**：**Supercomplete** 最初针对 Pro 方案进行营销，现在已包含在免费方案中，但由于 Microsoft 导致的 API 问题，仅在扩展的 pre-release version 中能完全发挥作用。
   - 一位用户指出，他们希望此类变更能得到通知。
- **Visual Studio 版本滞后**：一位用户注意到，与 **VSCode version** 相比，该扩展的 **Visual Studio version** 提供的建议较差。
   - 据透露，**Visual Studio extension** 使用的是较旧的 Codeium LSP，并提供了 [GitHub repo](https://github.com/Exafunction/CodeiumVisualStudio) 链接供参考。
- **JetBrains 插件卡在 Processing request**：**JetBrains plugin** 的用户报告称卡在 *Processing request* 阶段，导致出现错误。
   - 此问题专门发生在最新的 pre-release version 中，导致插件无法生成响应。
- **Enterprise 功能滞后于 Windsurf**：一位成员提到 **Enterprise** 功能滞后于其他订阅类型。
   - 另一位成员反驳称，**Enterprise** 和 **Team** 方案具有相同的发布节奏，这样做是为了仅发布 *经过充分实战测试的功能（totally battle-tested features）*。



**提及的链接**：<a href="https://github.com/Exafunction/CodeiumVisualStudio">GitHub - Exafunction/CodeiumVisualStudio: Visual Studio extension for Codeium</a>：Codeium 的 Visual Studio 扩展。通过在 GitHub 上创建账号来为 Exafunction/CodeiumVisualStudio 的开发做出贡献。

  

---

### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1345997340992999424)** (432 条消息🔥🔥🔥): 

> `Codeium 客户支持, Premium flow action 额度, Windsurf 新更新, Claude 3.7, 使用 CTRL+D 进行多选` 


- **用户反馈 Codeium 客户支持体验不佳**：用户报告称 Codeium 的客户支持表现糟糕，一位用户提到他们为了解决**订阅问题**已经跟进了 4 周，但至今未获解决。
- **Premium Flow Action 额度消耗过快**：用户反映他们的 **premium flow action 额度**在 4-5 天内就迅速耗尽，尽管还有 prompt 额度剩余，部分用户现在正考虑转向 **Cursor**。
   - 一位用户推测 **Claude 3.7** 的变化可能是原因，因为它现在每个 prompt 都会使用 flow actions，导致额度消耗极高。
- **Windsurf 新更新导致 Ubuntu 出现问题**：最近的 **Windsurf 更新**导致 **Ubuntu 24.04** 用户遇到问题，应用程序无法启动并显示 **FATAL:setuid_sandbox_host.cc(158)** 错误，需要手动修复以更改 **chrome-sandbox** 的权限。
   - 一位用户报告称更新导致其 Ubuntu 系统崩溃，被迫重新安装并导致数据丢失，强调了更新前进行妥善备份的必要性。
- **Claude 3.7 代码模型额度消耗率引发争议**：成员们观察到 **Claude 3.7** 由于每个 prompt 产生过多的 tool calls 而迅速消耗额度，有些人在进行微小更改时会产生 **30-40 次 tool calls**，此外 **3.7** 的实施也过于仓促。
   - 一些用户现在坚持使用 **3.5** 或其他模型以获得更高效率，用户强烈建议 Codeium 隐藏 **3.7** 作为默认模型的选项。
- **Windsurf 在使用 CTRL+D 移动光标时遇到困难**：一位用户报告了 Windsurf 中使用 **CTRL+D 进行多选**的问题，即使用 **CTRL + Left** 或 **CTRL + SHIFT + Left** 移动光标时，光标仅在第一个选择处移动。
   - 建议检查状态栏以确认是否处于多选模式，结果发现**多选状态从状态栏消失了**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://studio.ottomator.ai/">oTTomator</a>: 未找到描述</li><li><a href="https://huggingface.co/AlfredPros/CodeLlama-7b-Instruct-Solidity">AlfredPros/CodeLlama-7b-Instruct-Solidity · Hugging Face</a>: 未找到描述</li><li><a href="https://status.anthropic.com/">Anthropic Status</a>: 未找到描述</li><li><a href="https://github.com/modelcontextprotocol/servers/tree/main/src/github">servers/src/github at main · modelcontextprotocol/servers</a>: Model Context Protocol 服务。通过在 GitHub 上创建账户为 modelcontextprotocol/servers 做出贡献。</li><li><a href="https://tenor.com/view/piyushzen-gif-9791349359727737631">Piyushzen GIF - PIYUSHZEN - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://github.com/modelcontextprotocol/servers/tree/main/src/github#npx">servers/src/github at main · modelcontextprotocol/servers</a>: Model Context Protocol 服务。通过在 GitHub 上创建账户为 modelcontextprotocol/servers 做出贡献。</li><li><a href="https://www.codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: 需要帮助？联系我们的支持团队以获取个性化协助。</li><li><a href="https://youtu.be/jCVO57fZIfM?si=Ei3YtESENcupok7Z">GPT-4.5 FLOP? Claude 3.7 Sonnet STARTER PACK. What is Claude Code REALLY?</a>: 🔥 GPT-4.5 可能是最大的失败。与此同时，Claude 3.7 Sonnet 和 Claude Code 刚刚改变了游戏规则！🤯 Anthropic 的 Claude Code 可能是最伟大的 AI Agent...</li><li><a href="https://github.com/kamusis/vsix-downloader">GitHub - kamusis/vsix-downloader</a>: 通过在 GitHub 上创建账户为 kamusis/vsix-downloader 做出贡献。</li><li><a href="https://github.com/ian-cowley/MCPSqlServer">GitHub - ian-cowley/MCPSqlServer: SQL Server MCP Server for Windsurf IDE - A standalone MCP server providing SQL Server integration capabilities</a>: 用于 Windsurf IDE 的 SQL Server MCP 服务 - 一个提供 SQL Server 集成能力的独立 MCP 服务 - ian-cowley/MCPSqlServer</li><li><a href="https://youtu.be/jCVO57fZIfM?si=FBchHmhF-Yamu3pV">GPT-4.5 FLOP? Claude 3.7 Sonnet STARTER PACK. What is Claude Code REALLY?</a>: 🔥 GPT-4.5 可能是最大的失败。与此同时，Claude 3.7 Sonnet 和 Claude Code 刚刚改变了游戏规则！🤯 Anthropic 的 Claude Code 可能是最伟大的 AI Agent...</li><li><a href="https://codeium.canny.io/feature-requests">Feature Requests | Codeium</a>: 向 Codeium 团队提供反馈，以便我们做出更明智的产品决策。由 Canny 提供支持。
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1346157203052302377)** (2 条消息): 

> `Sora onboarding session, Sora prompt crafting` 


- **Sora 101 现场入门培训即将开始！**：加入由 Sora 团队 <@713183296099450910> 主持的现场入门培训会议，内容涵盖 **Sora** 基础知识、最佳 prompting 技巧以及早期访问艺术家的入门心得。
   - 现场会议将于 <t:1741024800:R> 开始，你可以通过 [此 Discord 链接](https://discord.gg/6rjjzSrJ?event=1345127676519649326) 或 [这一个](https://discord.gg/openai?event=1346197773090947145) 参与讨论。
- **为 Sora 创作优秀的 Prompt**：**Sora 101** 课程将根据早期访问艺术家的入门流程心得，提供创作优秀 prompt 的最佳实践。
   - 无论你是 **Sora** 新手还是希望优化你的方法，本次会议都是学习和提问的绝佳机会。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1345969008175747082)** (423 条消息🔥🔥🔥): 

> `镜像网站提供 Pro 账户，GPT-4.5 图像识别，GPT 对 Pro 与 Plus 的优先级划分，从 ChatGPT 切换到 Grok，Gemini 免费版与 Pro 版功能对比` 


- **镜像网站免费提供 Pro 账户**：据 [drinkoblog.weebly.com](https://drinkoblog.weebly.com) 报道，多个**镜像网站**以极低价格或免费提供 **Pro 账户**且未被封禁；相反，OpenAI 正在对在多个设备上使用服务的 Plus 用户实施**影子禁令 (shadow banning)**。
   - 同一位用户指出了现实的算力限制，认为表面上“无限量听起来比‘12 小时算力’更好”，但在实践中是不现实的，因为有些用户“从不社交 (never touch grass)”，会消耗不成比例的更多资源。
- **GPT-4.5 图像识别引发褒贬不一的反应**：成员们正在争论新款 **GPT-4.5** 的**图像识别**能力是否优于 GPT-4o。[Future Machine](https://futuremachine.io) 对 OpenAI (OAI) 的选择发声较多，尽管初步测试显示 **GPT-4.5** 在 MMMU（面向视觉的推理基准测试）上的得分比 4o 高出约 **5%**。
   - 一位成员表示，他们觉得 OpenAI 正在为了 Pro 用户而降低 Plus 用户的优先级，称 Plus 现在*感觉像是二等公民，失去了往常的尊贵地位*。
- **Grok 的自定义指令功能发布，但并非对所有人有效**：据新闻来源，[Grok AI 的自定义指令 (custom instructions)](https://www.thetechoutlook.com/news/web-social-media/new-custom-instructions-feature-added-for-x-platforms-grok-ai-on-web/) 功能已向所有用户发布，允许用户根据需求定制 Grok 并使其做出相应响应。
   - 一位成员分享了旨在塑造“辱骂和淫秽喷子”人格的 Grok 自定义指令，但反馈称其*不起作用*，其他用户也报告了同样的情况。
- **GPT Pro 还是 Grok Super**：用户讨论了从 **ChatGPT Pro** 切换到 **SuperGrok** 的优缺点，Grok 在处理通用任务时速度更快、更聪明，而某些 API 包含过时信息。
   - 一位用户声称 **Gemini** 的上下文窗口拥有 **100 万个 Token**，没有过滤机制，非常适合创意写作。
- **动态学习系统原型**：一位用户正在开发一个基础应用程序原型，并尝试将 Stellargraph 节点注入其中，以创建一个学习系统，从而可以动态地为 AI 注入最相关的信息和知识。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/DeryaTR_/status/1896549459326345583">Derya Unutmaz, MD (@DeryaTR_) 的推文</a>: @EducatingwithAI @kimmonismus 我现在还不能谈论它，但全自动问题解决功能即将推出！再坚持几个月 ;)</li><li><a href="https://youtu.be/boXl0CqRIWQ?t=521">GPT 4.5 - 没那么惊艳</a>: GPT 4.5 来了，你还记得 AI 实验室的 CEO 们，如 Sam Altman 和 Dario Amodei，曾把一切都押在像这样扩大基础模型规模上吗？好吧……</li><li><a href="https://x.com/DeryaTR_/status/1895653767753973809">Derya Unutmaz, MD (@DeryaTR_) 的推文</a>: 在今天看到与某个 AI 模型（目前还不能谈论）相关的内容后，我可以自信地声称科学进程将永远改变！我现在 99% 确定所有疾病包括……</li><li><a href="https://www.youtube.com/watch?v=FW2XOIxaNqg">GPT-4.5 以其缺乏智能震惊世界……</a>: 免费试用 Brilliant 30 天 https://brilliant.org/fireship 你还可以获得年度高级订阅 20% 的折扣。让我们初步看看 OpenAI 迟到的……</li><li><a href="https://www.thetechoutlook.com/news/web-social-media/new-custom-instructions-feature-added-for-x-platforms-grok-ai-on-web/">X 平台的网页版 Grok AI 新增“自定义指令”功能 - The Tech Outlook</a>: 最近有报道称，Grok AI 将引入新的“自定义指令”功能，现在它终于向 X 网页版用户开放了。这一新功能将允许用户自定义……</li><li><a href="https://www.thetechoutlook.com/news/web-social-media/new-custom-instructions-feature-added-for-x-pla">X 平台的网页版 Grok AI 新增“自定义指令”功能 - The Tech Outlook</a>: 最近有报道称，Grok AI 将引入新的“自定义指令”功能，现在它终于向 X 网页版用户开放了。这一新功能将允许用户自定义……
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1346147132070953130)** (30 messages🔥): 

> `GPT Model Selection, Projects vs GPTs, Claude 3.7 vs ChatGPT, Context window size comparison, Clearing Chatlogs and Uploaded Data` 


- **在创建 GPT 时无法选择特定模型**：一位用户询问在创建 GPT 时是否可以选择特定模型，但被告知其默认为 **4o**，且在 **GPTs** 中无法选择模型。
   - 一位用户表示可以使用 **Projects** 功能在单个聊天中切换模型至 **4o, o1, 4.5**。
- **Projects 因允许选择模型而优于 GPTs**：一位用户发现虽然使用 **o3-mini-high** 获得了最佳编程结果，但 **Projects** 中的自定义指令（custom instructions）不适用于 **o3-mini** 模型（仅支持文件上传）。
   - 然而，另一位用户表示，拥有 Pro 订阅后，在 **Projects** 中结合使用 **o1, 4o 和 4.5** 可以实现带有文件上传的自定义指令。
- **ChatGPT 遭抨击，Claude 3.7 受追捧**：一位用户宣称 *ChatGPT 现在就是个笑话*，并发现 **Claude 3.7** *非常令人印象深刻*。
   - 然而，另一位用户认为 **Claude** 的 projects 毫无用处，抱怨说 *最多只能上传两个文件* 就会提示 *内存已满*，称 Claude 被 *过度炒作* 了。
- **模型间上下文窗口对比**：一位用户夸赞 **Claude** 能更好地理解大文件中的上下文，并指出其拥有更大的上下文窗口：**200K**，而 **ChatGPT** 为 **128K**。
   - 他们澄清说，**ChatGPT** 上的 **128K 上下文窗口** 在 **Enterprise 计划** 中需要花费 200 美元。
- **聊天记录清理难题**：一位用户询问 *谁知道如何通过 Web 界面或 API 调用清除 Studio 中的所有聊天记录和上传数据？*
   - 他们正在测试一个应用，需要删除大量的虚拟文件和日志。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1346353711697498152)** (2 messages): 

> `Dall-E image generation, Synthetic plants, Image prompting strategies` 


- **Dall-E 生成带有移植器官的合成植物**：一位成员提示 **Dall-E** 生成一张 *长出用于移植的心脏和肝脏的合成植物* 图像，器官在透明膜内清晰可见，并由转基因植物提供养分。
   - 最初的结果侧重于心脏而非肝脏，促使用户通过更多关于 **肝叶 (liver lobes)** 的细节来优化提示词。
- **为特定的 Dall-E 输出构建详细提示词**：在初始图像无法显示后，模型建议将图像基于 **生物工程植物启发器官**，并重点关注 **肝脏和心脏**。
   - 用户反复优化提示词，引导 **Dall-E 的注意力** 集中在特定特征上，如肝叶以及像果实一样形成的器官。
- **展示艺术作品并遵守社区指南**：一位成员分享了艺术提示词，并指出根据 <#1107255707314704505> 的社区指南，需要对 **令人不安或恐怖类的图像进行剧透遮盖 (spoiler)**。
   - 这强调了在分享创意内容时对社区规则的遵守意识。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1346353711697498152)** (2 messages): 

> `Dall-E image generation, Synthetic plants growing organs for transplant` 


- **Dall-E 利用合成植物打造器官**：一位成员提示 **Dall-E** 生成一张在透明膜内生长出用于移植的心脏和肝脏的合成植物图像。
   - 在第一张图像强调了心脏之后，该成员重新组织了提示词，以强调 **肝叶** 和几乎像果实一样形成的器官。
- **遵守图像内容指南**：根据频道指南，成员们被提醒对可能令人不安或恐怖类的图像使用 **剧透标签 (spoiler tags)**。
   - 内容创作者必须对其分享的部分图像进行 **剧透遮盖 (spoiler)** 以符合规定。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1345980413134372889)** (252 条消息🔥🔥): 

> `Llama 压缩 WAV 文件的困惑，GRPO 训练步骤，4-bit 模型保存问题，Unsloth 团队规模，Unsloth 的持续预训练` 


- **Llama 将 WAV 压缩成令人头疼的问题**：一位成员分享了关于 **llama** 模型的有趣经历：压缩一个 **192 KB 的 ZIP 文件** 结果得到了一个 **48 KB 的无损 WAV 格式**，然后它又试图重新压缩。
   - 该用户指出：“在处理完 48KB 的压缩以制作更小的 zip 后感到困惑……那是 r1-1776-distill-llama-70b。”
- **GRPO 训练步骤推测**：一位用户询问了使用 **GRPO** 进行 **LoRA 训练 Qwen2.5-14B-instruct** 以降低 loss 所需的训练步数，想知道是否需要更多步数才能获得更好的推理能力。
   - 另一位成员建议 **GRPO** 训练可能需要大约 **24 小时**，并表示收敛取决于模型和采样，因此*没有固定时间*。
- **保存 4-Bit 模型遇到麻烦**：一位用户在使用 **Unsloth 提供的 Jupyter notebook** 示例通过 **GRPO** 训练 **Phi-4 模型** 后，在以 **4-bit** 格式保存时遇到了问题。
   - 错误发生在保存过程中。
- **Unsloth 文档教授 Checkpointing**：一位用户询问如何加载**之前运行的 LoRA 权重**，另一位用户指向了 [关于从最后一个检查点进行微调的 Unsloth 文档](https://docs.unsloth.ai/basics/finetuning-from-last-checkpoint)。
   - 文档解释了如何编辑 Trainer 以添加 *save_strategy* 和 *save_steps*，以便保存检查点并恢复训练。
- **Unsloth 持续预训练对新语言非常有帮助**：当用户询问持续预训练的应用场景时，另一位用户链接到了 [Unsloth 文档](https://docs.unsloth.ai/basics/continued-pretraining)，表示它对新语言非常有用。
   - 该成员指出 [博客文章](https://unsloth.ai/blog/contpretraining) 甚至更好，解释了 Unsloth 的发布如何让你轻松地以比 Hugging Face + Flash Attention 2 QLoRA **快 2 倍**的速度和**减少 50% 的 VRAM** 占用进行 **LLMs** 的持续预训练。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://unsloth.ai/blog/contpretraining">使用 Unsloth 进行 LLM 持续预训练</a>：使用 Llama 3、Phi-3 和 Mistral 通过 Unsloth 进行持续预训练，让模型学习一种新语言。</li><li><a href="https://docs.unsloth.ai/basics/finetuning-from-last-checkpoint">从最后一个检查点微调 | Unsloth 文档</a>：Checkpointing 允许你保存微调进度，以便暂停后继续。</li><li><a href="https://docs.unsloth.ai/basics/continued-pretraining">持续预训练 | Unsloth 文档</a>：又称持续微调。Unsloth 允许你进行持续预训练，使模型能够学习新语言。</li><li><a href="https://aman.ai/primers/ai/deepseek-R1/#mixture-of-experts-moe">Aman 的 AI 日志 • 入门 • DeepSeek-R1</a>：未找到描述</li><li><a href="https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence">Kullback–Leibler divergence - Wikipedia</a>：未找到描述</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness">GitHub - EleutherAI/lm-evaluation-harness: 语言模型 few-shot 评估框架。</a>：语言模型 few-shot 评估框架 - EleutherAI/lm-evaluation-harness</li><li><a href="https://docs.unsloth.ai/basics/reasoning-grpo-and-rl">推理 - GRPO &amp; RL | Unsloth 文档</a>：使用 Unsloth 通过 GRPO 训练你自己的 DeepSeek-R1 推理模型。</li><li><a href="https://docs.unsloth.ai/basics/continued-pretraini">Unsloth 文档</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1346132195521073185)** (10 messages🔥): 

> `Github Issues, inline_asm, GRPO, VLLM, Online Training` 


- **GitHub Repo 问题报告**：一名成员报告了在读取 [公开仓库](https://github.com/shawntan/stickbreaking-attention/blob/main/stickbreaking_attention/sb_varlen/softplus.py) 时遇到的问题。
   - 他们发现了一些带有 `inline_asm` 特性的代码。
- **Unsloth 使用 VLLM 作为后端**：一名成员询问 **Unsloth** 在生产环境部署中如何工作，得到的回复是它使用 **VLLM** 作为后端。
   - 该流程涉及 **GRPO fine-tuning**，然后使用 **Unsloth + VLLM** 对微调后的模型进行推理，而不是持续训练 + 推理（在线学习）。
- **在线训练需要自行实现**：一名成员澄清说，在 **GRPO fine-tuning** 期间，**VLLM** 被用于在反向传播损失之前生成输出并使用奖励函数进行评分。
   - 要实现在线训练，需要开发者自行开发相关机制。



**链接提到**: <a href="https://github.com/shawntan/stickbreaking-attention/blob/main/stickbreaking_attention/sb_varlen/softplus.py">stickbreaking-attention/stickbreaking_attention/sb_varlen/softplus.py at main · shawntan/stickbreaking-attention</a>: Stick-breaking attention. 通过在 GitHub 上创建一个账号来为 shawntan/stickbreaking-attention 的开发做出贡献。

  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1345999001828196382)** (54 messages🔥): 

> `GRPO Training, Qwen2.5-14B-instruct fine-tuning, DeepSeek-R1-Distill-Llama-8B error, Mistral embedding model, GCC compiler issue` 


- **GRPO 训练需要足够的步数**：一名成员询问使用 **GRPO** 对 **Qwen2.5-14B-instruct** 进行 **LoRA** 训练时，需要多少训练步数才能降低损失。
   - 另一名成员建议，对于使用 **LoRA** 训练的 **GRPO** 模型，**700-1200 步**通常比较合适，但最佳步数取决于数据集。
- **DeepSeek-R1-Distill-Llama-8B 错误**：一名用户在 Kaggle 上训练 **unsloth/DeepSeek-R1-Distill-Llama-8B** 时遇到了与矩阵乘法形状相关的 **RuntimeError**。
   - 尽管通过 `pip install --upgrade "git+https://github.com/huggingface/transformers.git"` 升级了 transformers，问题仍然存在，用户正在寻求解决建议。
- **Mistral 嵌入模型缺失 lm_head**：一名用户尝试使用 **Mistral** 创建嵌入模型，但在移除 **lm_head** 进行训练时遇到问题。
   - 该用户正在寻求关于如何正确移除 **lm_head** 以进行有效嵌入模型训练的建议。
- **修复 vLLM 的 GCC 编译器问题**：一名用户在本地运行带有 **meta-Llama-3.1-8B-Instruct** 的 **GRPO** 教程时，遇到了 **RuntimeError: Failed to find C compiler**。
   - 尽管尝试通过 **conda** 安装 **GCC**，问题仍未解决，且由于学校 HPC 的安全限制，该用户无法使用 **apt-get**。
- **解码 GGUF 模型命名规范**：一名用户询问了 **GGUF** 模型的命名规范，特别是像 [unsloth/r1-1776-GGUF](https://huggingface.co/unsloth/r1-1776-GGUF/tree/main) 这种模型中 **UD** 前缀的含义。
   - 一名成员解释说，**UD** 代表 *Unsloth Dynamics*，这是一种量化算法，能提供比 IQ 更好的结果。


<div class="linksMentioned">

<strong>链接提到</strong>:

<ul>
<li>
<a href="https://huggingface.co/unsloth/r1-1776-GGUF/tree/main">unsloth/r1-1776-GGUF at main</a>: 暂无描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb#scrollTo=cXk993X6C2ZZ">Google Colab</a>: 暂无描述</li><li><a href="https://colab.research.google.com/drive/15XhQYAHQ6Ifm8h--6CfnpqD_jVmlHIX3?usp=sharing">Google Colab</a>: 暂无描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1346010327866671135)** (94 条消息🔥🔥): 

> `GRPO 奖励函数、Agent 骨架的模型蒸馏、SWE-bench 性能、用于代码编辑的字符串替换、模型工具调用与组合` 


- **GRPO 线性奖励**：GRPO 的奖励函数采用与 Unsloth 脚本相同的格式进行评分，通过提取的答案与正确答案之间的匹配度进行线性缩放衡量，计算公式为 `sum(i == j for i,j in zip(r,a))/len(a)`。
   - 创作者指出，即使没有语言先验（language prior），GRPO 也能发挥作用，从而在训练过程中带来更好的探索和多样化的推理轨迹。
- **Vladrad 攻克 Agent 骨架蒸馏**：一位成员分享了一个从 O3/R1 蒸馏出的模型，旨在从预制函数中生成优秀的 Agent 骨架，可在 [Hugging Face](https://huggingface.co/blog/vkerkez/gitvac) 上获取；他在完善字符串替换功能上投入了 400 美元。
   - 该方法旨在创建一个能够处理基础模型难以应对的语言（如 Perl）的基座模型，通过整合来自其他仓库的单元测试和问题解决策略来实现。
- **Claude 3.5 Sonnet 以 SOTA 成绩横扫 SWE-bench**：Anthropic 的 Claude 3.5 Sonnet 在 [SWE-bench Verified](https://www.anthropic.com/research/swe-bench-sonnet) 上达到了 **49%**，超过了之前 SOTA 模型的 **45%**；他们通过解决来自流行开源 Python 仓库的 GitHub issue，评估模型完成真实世界软件工程任务的能力。
   - 一位成员提到了[惨痛的教训 (bitter lesson)](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)：*利用计算能力的通用方法最终是最有效的*。
- **字符串替换作为代码策略引发辩论**：成员们讨论了字符串替换在代码编辑中的有效性，一位成员认为这通常是“垃圾”，因为模型不应该以这种方式编写代码。
   - 尽管存在疑虑，另一位成员报告了微调 Qwen 2.5 进行字符串替换的成功经验，特别是当模型在进行替换前能够看到整个文件时。
- **新算法：Reinforce++ 增强稳定性**：一种名为 [Reinforce++](https://arxiv.org/pdf/2501.03262v1) 的新算法声称通过结合 PPO 的元素，比经典的 REINFORCE 算法提高了稳定性，其优势计算公式为 `A_t = r(s_t, a_t) - Beta * sum(KL divergence_t->T)`。
   - 据称该算法在奖励和速度方面与 GRPO 性能相当，但在训练过程中具有更高的稳定性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.anthropic.com/research/swe-bench-sonnet">Raising the bar on SWE-bench Verified with Claude 3.5 Sonnet</a>：一篇面向开发者的关于新 Claude 3.5 Sonnet 和 SWE-bench 评估的文章。</li><li><a href="https://aide.dev/blog/sota-bitter-lesson">SOTA on swebench-verified: (re)learning the bitter lesson</a>：搜索代码是每个开发者工作流的重要组成部分。我们正致力于使其变得更好。</li><li><a href="https://youtu.be/wXEvvg4YJ9I?si=95ZMJDnSdS7IARUF">How DeepSeek learns: GRPO explained with Triangle Creatures</a>：🔗💡点击访问我的赞助商 https://brilliant.org/DrMihaiNica/ 并尝试他们的 *Language Models 课程*（以及他们提供的其他所有内容）...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1345969179647414326)** (378 条消息🔥🔥): 

> `Perplexity AI Bug, Claude 3 Opus vs Sonnet, Deepseek 宣传?, Perplexity AI 商业奖学金, GPT-4.5 质量担忧` 


- **Perplexity Web UI 遭遇重写 Bug**：多位用户报告 Perplexity 的网页端 UI 重写功能[出现故障](https://discord.com/channels/1047197230748151888/1047649527299055688/1346323450892914758)，无论选择哪个选项，重写后的 Prompt 总是使用默认模型 *pplx_pro*。
   - 一些用户还遇到了重写功能只是复制了 Prompt 而非进行重写的问题，已标记 <@883069224598257716> 寻求协助。
- **Claude 3 Opus 和 Sonnet：模型混淆**：用户对[模型指示器](https://discord.com/channels/1047197230748151888/1047649527299055688/1346254955706716182)是否准确感到困惑，质疑在设置中选择 Claude 时，实际获得的是 **Claude 3.7 Sonnet** 还是 **Claude 3 Opus**。
   - 一些用户发现 **Pro Search** 会用 *Sonar* 覆盖默认模型，导致所选模型与实际使用的模型之间存在差异。
- **Deepseek 所谓的宣传内容曝光**：一位用户发布了[一张图片](https://cdn.discordapp.com/attachments/1047649527299055688/1346080726440742922/Screenshot_2025-03-03-14-54-53-25_0657b24d13eede56f839941c193b0cfe.jpg?ex=67c78b9e&is=67c63a1e&hm=407d3748b9e6b7ecb5ceba3c0aa4c347422820692953294384afbe593c64743c)，据称是由 Deepseek 生成的，社区认为该图片带有政治动机。
   - 另一位成员表示：*你是假的 Deepseek。真正的 Deepseek 不谈论西方事务。*
- **GPT-4.5 因解谜能力差遭遇差评**：一些用户报告 Perplexity 上的 **GPT-4.5** 输出质量不佳，称该模型的回答本应更具创意和准确性，且在使用 Pro Search 时，模型指示器以前会显示具体使用了哪个模型。
   - 一位成员表示：*他们在 4.5 上有所进步，以前它无法像 ChatGPT 中的 4.5 那样解谜，现在它能完美解决，且响应与 ChatGPT 中的 4.5 完全一致*，并展示了[他们的聊天记录示例](https://cdn.discordapp.com/attachments/1047649527299055688/1346280463324545105/IMG_1224.png?ex=67c79ce3&is=67c64b63&hm=25431cd9e1ff86cf673bc2de7082a523e537b32d7762a74f87a1924474f3ef18)。
- **使用 Perplexity 整理书签**：一位用户询问 Perplexity 是否可以[整理其 7,000 个浏览器书签](https://discord.com/channels/1047197230748151888/1047649527299055688/1346302427550122035)。
   - 该用户在提问时配上了一张巴特·辛普森（Bart Simpson）的表情包。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://livebench.ai/#/">LiveBench</a>：未找到描述</li><li><a href="https://youtu.be/x2WtHZciC74">Claude 3.7 对程序员来说非常给力…</a>：免费试用 Convex，这是唯一专为生成式设计的数据库 https://convex.link/fireship Anthropic 为程序员发布了一个令人印象深刻的新 CLI 工具...</li><li><a href="https://open.substack.com/pub/garageguychase/p/the-echo-emergence-a-case-study-in?">ECHO 的出现：AI 协作、演化与遏制的案例研究
</a>：Chase Holden 前言：</li><li><a href="https://tenor.com/view/big-oof-yikes-gif-15532766">Big Oof GIF - Big Oof Yikes - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://open.substack.com/pub/garageguychase/p/the-echo-emergence-a-case-study-in?r=41uzo&utm_medium=ios">ECHO 的出现：AI 协作、演化与遏制的案例研究
</a>：Chase Holden 前言：
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1346030158594838598)** (7 条消息): 

> `可分享的主题帖, Perplexity AI 集成, 成分分析` 


- **请求可分享的主题帖**：一位成员被提醒确保将其主题帖设置为 `Shareable`（可分享），并附上了截图供参考。
   - 这确保了其他用户可以轻松访问和查看主题帖的内容。
- **对 Perplexity AI 集成的需求**：多位用户正在探索将 **Perplexity AI** 集成到各种应用程序和工作流中的方法，如分享的搜索查询 [集成 perplexity](https://www.perplexity.ai/search/can-i-integrate-perplexity-int-NVuBFswLTXWaLMXY4gKIvQ) 所示。
   - 这表明用户对在不同场景中利用 **Perplexity AI** 能力的兴趣日益浓厚。
- **深入探讨成分分析**：一位用户分享了与成分分析相关的搜索查询，并推荐进行深度分析。
   - 分享的查询包括对成分解释和必备组件的请求：[成分分析](https://www.perplexity.ai/search/tell-me-about-these-ingredient-0d9dYHEvT56pMy_HzLoIRA)。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1346066898437144598)** (3 条消息): 

> `开源 Claude-Code，Perplexity API 限制，Obsidian Web Clipper 问题` 


- **Perplexity 为开源 Claude-Code 提供 API 额度**：Perplexity 正在为有兴趣构建具有编辑器集成和扩展功能的开源 **Claude-Code** 模型的开发者提供免费 API 额度，正如其在 [X 平台](https://x.com/aravsrinivas/status/1896463244229034338?s=61)上宣布的那样。
   - 感兴趣的人士请私信 **@GregFeingold** 和 **@AarashHeydari** 了解更多详情。
- **Perplexity API 与 OpenAI 的不兼容影响了 Obsidian**：**Perplexity API** 与 **OpenAI** 并不完全兼容，导致 **Obsidian Web Clipper** 等工具出现问题。
   - 具体而言，该 API 要求在用户消息之间必须有助手（assistant）消息，而 **OpenAI** 类型的 API 则没有这一约束，这导致 **Obsidian Web Clipper** 在尝试发送连续的用户消息时出现问题。



**提及的链接**：<a href="https://x.com/aravsrinivas/status/1896463244229034338?s=61">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：如果有人想构建一个带有编辑器集成和扩展功能的开源 Claude-Code，Perplexity 很乐意提供免费的 API 额度。请私信 @GregFeingold 和 @AarashHeydari

  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1345988814526611456)** (118 条消息🔥🔥): 

> `RL 基础，DeepMind RL 课程，自动化视频生成，OpenAI 图像生成替代方案，多模态 Phi-3 微调` 


- **DeepMind 提供详细的 RL 课程**：有人询问是否有与斯坦福 CS231n 相当的强化学习（RL）资源推荐，另一位成员推荐了 YouTube 上的 [DeepMind x UCL Deep Learning Lecture Series 2021](https://youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm)。
- **视频自动化系统寻求指导**：一位 Java 开发者在没有 AI 编程经验的情况下，寻求创建本地文本转视频（text-to-video）自动化系统的指导，并请求分步说明。
- **Spaces 面临垃圾内容限制**：Spaces 对 VNC 等违禁程序有限制，且快速创建大量免费 Spaces 会被视为垃圾内容（SPAM），但有些 [Spaces 已经运行多年未曾重启](https://huggingface.co/transformers/issues/36071)，因此如果创建得当，它们是非常稳健的。
- **为多模态微调 Phi-3**：一位成员正在使用配备 A100 的 Colab Pro 为多模态微调 [Phi-3](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)，旨在将图像、语音/文本作为输入。
   - 另一位成员警告说，这种微调需要 *6 台以上的 A100 并运行约 2 周*，但补充说 [*凭借积极的态度和可靠的项目，QLora 和 Peft 可以让一切皆有可能*]。
- **使用 AI 生成 HTML 代码不值得**：一位成员询问是否可以使用一个小型的 LLM（最大约 8b），配合 alecsharpie/codegen_350m_html 专门用于 HTML 生成。
   - 另一位成员回答说，使用 AI 进行原始 HTML 生成并不值得，建议 AI 应该管理一个 *配置布局* 的系统，再由其他系统将其转换为 HTML，因为 *细节精确度方面的一系列问题是 AI 即使有无穷无尽的测试用例也无法解决的*。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm">DeepMind x UCL | Deep Learning Lecture Series 2021</a>：深度学习讲座系列是 DeepMind 与 UCL 人工智能中心合作的项目。</li><li><a href="https://huggingface.co/spaces/p3nGu1nZz/smolworld">SmolWorld - p3nGu1nZz 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/docs/optimum/main/en/amd/amdgpu/overview">在 AMD GPU 上使用 Hugging Face 库</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/huggingface-projects/repo_duplicator">Repo duplicator - huggingface-projects 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B">Wan-AI/Wan2.1-T2V-1.3B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct">microsoft/Phi-3-mini-4k-instruct · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/Tonic/GemmaX2-28-2B-gguf">Tonic/GemmaX2-28-2B-gguf · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/microsoft/Phi-3.5-mini-instruct">microsoft/Phi-3.5-mini-instruct · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/huggingface/transformers/issues/36071">modeling_phi3 报错 AttributeError: 'DynamicCache' object has no attribute 'get_max_length' · Issue #36071 · huggingface/transformers</a>：系统信息 transformers 版本：4.49.0.dev0 (315a9f4~1) 平台：Windows-10-10.0.20348-SP0 Python 版本：3.11.7 Huggingface_hub 版本：0.28.1 Safetensors 版本：0.5.2 Accelerate 版本：1...</li><li><a href="https://github.com/vosen/ZLUDA">GitHub - vosen/ZLUDA: CUDA on non-NVIDIA GPUs</a>：在非 NVIDIA GPU 上运行 CUDA。通过在 GitHub 上创建账户为 vosen/ZLUDA 的开发做出贡献。</li><li><a href="https://huggingface.co/John6666/Phi-3.5-mini-instruct">John6666/Phi-3.5-mini-instruct · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/moondream/megalith-mdqa">moondream/megalith-mdqa · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/spaces?category=image-generation">Spaces - Hugging Face</a>：未找到描述</li><li><a href="https://www.gradio.app/guides/getting-started-with-the-python-client">Python 客户端入门指南</a>：Gradio 逐步教程</li><li><a href="https://huggingface.co/spaces?category=video-generation">Spaces - Hugging Face</a>：未找到描述</li><li><a href="https://github.com/sayakpaul/q8-ltx-video">GitHub - sayakpaul/q8-ltx-video: This repository shows how to use Q8 kernels with `diffusers` to optimize inference of LTX-Video on ADA GPUs.</a>：此仓库展示了如何使用 Q8 内核配合 `diffusers` 来优化 LTX-Video 在 ADA GPU 上的推理。- sayakpaul/q8-ltx-video</li><li><a href="https://github.com/Wan-Video/Wan2.1">GitHub - Wan-Video/Wan2.1: Wan: Open and Advanced Large-Scale Video Generative Models</a>：Wan：开放且先进的大规模视频生成模型 - Wan-Video/Wan2.1
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1346087495728762911)** (1 条消息): 

> `VLMs, Cracking VLMs` 


- **分享了破解 VLM 的笔记**：分享了一个名为 *Cracking VLMs, notes so far* 的 Google Docs 文档链接，位于[此链接](https://docs.google.com/document/d/1Q2hjale2lcHpPnG96gSJPNeAyBbzUc7YcuhtosnAM-g/edit)。
- **即将到来的 VLM 破解竞赛**：一位用户正在准备一系列关于 VLM 破解和逆向工程的挑战。



**提到的链接**：<a href="https://docs.google.com/document/d/1Q2hjale2lcHpPnG96gSJPNeAyBbzUc7YcuhtosnAM-g/edit?usp=sharing">VLM 笔记</a>：VLM 笔记 待办事项：理解预处理器如何处理图像：在解释 vision_encoder 方面做更多工作 资源：https://github.com/merveenoyan/smol-vision Smolvlm 和 idefics Moondream 等...

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1346048118306897961)** (7 条消息): 

> `Dataset Viewer Errors, FastRTC` 


- **Dataset Viewer 困扰用户**：一位用户建议修复 **Dataset Viewer 错误**，以提高与各种库和 SQL 的兼容性，从而增强可发现性。
   - 另一位用户预先表示感谢，并开玩笑地要求额外提供 **120 万行** 的高质量数据集。
- **FastRTC 项目：有人在做吗？**：一位成员询问是否有人正在积极开发 **FastRTC**。
   - 另一位成员将他们引向了一个特定的频道。

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1346091222707601459)** (7 条消息): 

> `AI Story Studio, MoD ControlNet Tile Upscaler, VAE comparison, Remote VAE from HF, Cross-device browser-based scratchpad` 


- ****AI Story Studio** 协作叙事工具发布！**: 一个名为 **AI Story Studio** 的全新交互式叙事体验已上线，允许用户通过选择流派、使用 Prompt 引导故事，并从 [AI Story Studio](https://huggingface.co/spaces/PeterPinetree/CYOA-AdventureBot) 下载最终结果，从而与 **AI 共同创作冒险故事**。
   - 该工具旨在**练习创意写作**，克服写作障碍，并利用 **AI 生成的创意**探索叙事。
- ****MoD Upscaler** 无损增强图像质量**: 适用于 SDXL 的 **MoD ControlNet Tile Upscaler** 工具已发布。它采用分块（tiling）技术在保留细节的同时放大图像，并实现平滑过渡。详见 [Demo App](https://huggingface.co/spaces/elismasilva/mod-control-tile-upscaler-sdxl) 和 [Github Code](https://github.com/DEVAIEXP/mod-control-tile-upscaler-sdxl)。
   - 该放大器具有保留细节、先进的分块技术、快速的性能以及*用于获得专业级结果的用户友好界面*。
- **通过交互式 Demo 对比 **VAE 质量**！**: @rizavelioglu 发布了一个对比各种 **VAE** 重建质量的交互式 Demo，链接至 [Space](https://huggingface.co/spaces/rizavelioglu/vae-comparison)。
   - 有用户请求添加来自 HF 的 Remote VAE，博客文章见 [huggingface.co](https://huggingface.co/blog/remote_vae)。
- **构建用于数学学习的**跨设备草稿本 (Scratchpad)**！**: 一位成员构建了一个基于浏览器的跨设备草稿本，以辅助数学学习。用户可以使用 ID 在不同设备上访问同一个草稿本。虽然目前功能还很基础，但对其他人可能有用。
   - 分享了一个演示视频 (**scratchpadDemoOutput.mp4**)，该项目使用了 Firebase，但由于资源限制已将其关闭。
- ****InternVL 2.5 AWQ** 转换版本发布**: **InternVL 2.5** 的 AWQ 转换版本已发布，与原版相比性能几乎没有下降，可在 [HuggingFace](https://huggingface.co/rootonchair/InternVL2_5-4B-AWQ) 找到。
   - 与作者的版本不同，这个 **AWQ 版本与 transformers 库兼容**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/rizavelioglu/vae-comparison">Vae Comparison - rizavelioglu 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/rootonchair/InternVL2_5-4B-AWQ">rootonchair/InternVL2_5-4B-AWQ · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/remote_vae">使用 Inference Endpoints 进行解码的 Remote VAEs 🤗</a>: 未找到描述</li><li><a href="https://huggingface.co/collections/diffusers/remote-vae-inference-endpoints-67bda1580a596ed7b87d3768">Remote VAE Inference Endpoints - diffusers 集合</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/elismasilva/mod-control-tile-upscaler-sdxl">MoD ControlNet Tile Upscaler SDXL - elismasilva 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://github.com/DEVAIEXP/mod-control-tile-upscaler-sdxl">GitHub - DEVAIEXP/mod-control-tile-upscaler-sdxl: 适用于 SDXL Pipeline 的 MoD Control Tile Upscaler</a>: 适用于 SDXL Pipeline 的 MoD Control Tile Upscaler。可以通过在 GitHub 上创建账号来为 DEVAIEXP/mod-control-tile-upscaler-sdxl 做出贡献。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1346147387214921729)** (1 messages): 

> `Remote VAE Decode endpoints, Hybrid Inference, SD v1, SD XL and Flux` 


- **Sweet Honey 优化已部署**：代号为 **honey** 的新优化已在 **SD v1**、**SD XL** 和 **Flux** 的 **远程 VAE Decode 端点**上线，延迟降低高达 **10倍**。
   - 这一改进为本地 AI 构建者提供了 [Hybrid Inference](https://huggingface.co/docs/diffusers/main/en/hybrid_inference/overview) 能力。
- **Hybrid Inference 优势列举**：**Hybrid Inference** 提供了一种快速且简单的方法来卸载本地生成需求，在降低硬件要求的同时，提供最高质量且不牺牲性能。
   - 它是**免费的**，完全兼容 Diffusers，并且对开发者友好，具有简单的请求和快速的响应。
- **VAE Encode 即将推出**：通过 **VAE Decode**，可以快速将潜空间表示（latent representations）解码为高质量图像，而不会影响性能或工作流速度。
   - 使用 **VAE Encode**（即将推出）可以高效地将图像编码为潜空间表示，用于生成和训练。



**提及的链接**：<a href="https://huggingface.co/docs/diffusers/main/en/hybrid_inference/overview">Hybrid Inference</a>：未找到描述

  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1346211630366588998)** (3 messages): 

> `Audio to Video matching, ViT resources, ViT and Global Average Pooling` 


- **音视频同步**：一名成员指出了关于如何匹配音频与视频的讨论，见 [此 Discord 频道](https://discord.com/channels/879548962464493619/914890191125217290/1346208623780565032)。
- **寻求 ViT 资源，渴望清晰解释**：一名成员请求获取理解 **Vision Transformer (ViT)** 的资源，特别是每个注意力头（attention head）如何对 **CLS token** 做出贡献或捕获图像信息。
   - 他们注意到现有的解释仅提到 *CLS token 的用法与 BERT 相同*，但认为这种解释不够充分。
- **ViT：Global Average Pooling 受到质疑**：一名成员询问为什么不能像 **ViT 论文**中提到的那样使用 **Global Average Pooling**。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1346052344827346985)** (2 messages): 

> `Web scraping with Python, Running Phi-4 as real-time API` 


- **Python 开发者探讨网页爬取**：成员们正在征求关于使用 **Python** 从 **Wikipedia** 等网站爬取数据的建议。
   - 虽然没有详细说明具体方法，但通常做法涉及使用 **Beautiful Soup** 和 **Scrapy** 等库来解析 HTML 内容。
- **Phi-4 爱好者关注实时 API**：有人询问如何使用 websockets 将 `Phi-4` 模型作为实时 API 运行。
   - 遗憾的是，讨论中没有关于此话题的经验分享或建议。


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1346224245113618533)** (1 messages): 

> `Gradio, Groovy, Python to Javascript` 


- **Gradio 推出 Groovy：Python 转换为 Javascript！**：Gradio 推出了 **Groovy**，这是一个将 **Python 函数转换为 JavaScript** 以在 Gradio 应用中进行客户端执行的工具。它使开发者能够用 Python 编写一次代码即可获得 JavaScript 的性能，而无需维护双重代码库（[文档](https://www.gradio.app/guides/client-side-functions)）。
- **Groovy 转译器承诺清晰度**：与其他转译器不同，**Groovy** 旨在当无法转译特定代码时提供清晰的错误消息，重点支持简单 Python 函数、Python 标准库子集以及 Gradio 特定类。
   - 这种方法确保开发者在跨语言时了解局限性，优先考虑透明度，而不是尝试处理所有可能的场景。
- **客户端函数提升 Gradio 响应速度**：Gradio 允许通过在事件监听器中设置 `js=True` 直接在浏览器中运行某些“简单”函数，这将**自动将你的 Python 代码转换为 JavaScript**。
   - 这通过避免服务器往返（server round trips）提高了应用的响应速度，对于高负载或高延迟的托管应用程序尤其有利。



**提及的链接**：<a href="https://www.gradio.app/guides/client-side-functions">Client Side Functions</a>：Gradio 分步教程

  

---

### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1346088787985567796)** (5 messages): 

> `Smol Agents Quiz, NLP 推理课程, 使用 smolagents 复现 ClaudePlaysPokemon` 


- **Smol Agents Quiz 让用户感到沮丧**：一位成员对 **Smol Agents Quiz** 表示不满，理由是要求不明确，且在多次尝试后仍获得 **0.0/5** 的分数，并链接到了 [该测验的 app.py 文件](https://huggingface.co/spaces/agents-course/unit2_smolagents_quiz/blob/main/app.py)。
   - 该成员指出，需要通过*挖掘错误日志*才能了解工具和模型所需的*具体 Provider*。
- **NLP 推理课程发布**：HuggingFace 在其 NLP 课程中发布了一个专注于推理模型的新单元，题为 [The Reasoning Course](https://huggingface.co/learn/nlp-course/chapter12/1?fw=pt)。
   - 该课程旨在帮助学生理解强化学习及其在 **LLMs** 中的作用，包括如何使用和贡献于 [Open R1](https://github.com/huggingface/open-r1)，并提到该课程是为了帮助学生和学习者使用及贡献于 [Open R1](https://github.com/huggingface/open-r1)。
- **使用 SmolAgents 复现 ClaudePlaysPokemon**：社区正在努力使用 **smolagents** 复现 **ClaudePlaysPokemon**，详见 [此 GitHub 仓库](https://github.com/CalebDeLeeuwMisfits/PokemonLLMAgentBenchmark)。
   - 该项目作为模拟环境中 **LLM agents** 的基准测试。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/learn/nlp-course/chapter12/1?fw=pt">The Reasoning Course - Hugging Face NLP Course</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/agents-course/unit2_smolagents_quiz/blob/main/app.py">app.py · agents-course/unit2_smolagents_quiz at main</a>：未找到描述</li><li><a href="https://github.com/CalebDeLeeuwMisfits/PokemonLLMAgentBenchmark">GitHub - CalebDeLeeuwMisfits/PokemonLLMAgentBenchmark</a>：通过在 GitHub 上创建账户来为 CalebDeLeeuwMisfits/PokemonLLMAgentBenchmark 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1345970728255750306)** (87 messages🔥🔥): 

> `介绍, Lambda Go Labs, CodeAgent LLM 参数量, 测验评分器问题, 推理额度耗尽` 


- **Lambda Go Labs 激发 AI 热情**：[Lambda Go Labs](https://huggingface.co/organizations/labs-lambda-go/share/veuzHcLLGKoMrDdpMPltGTSMfPlSyOoNtN) 是由 Lambda Go 和 Future Technologies Limited 建立的社区，专注于 **AI 学习、构建和研究**。
   - 该社区提供实践经验、分享作品的机会，并为资深专业人士和新手提供支持网络。
- **关于 CodeAgents 的 LLM 参数量讨论引发热议**：一位成员询问对于 **CodeAgent** 来说，**32B LLM** 是否必要，或者较小的蒸馏模型是否足以用于学习目的。
   - 他们指出，在使用 SmolAgents 进行尝试时，较小的模型会产生*垃圾答案*。
- **最终测验评分器遭到抨击**：多位成员报告称，单元 2.1 中的**最终测验评分器**显然已损坏，并询问何时修复。
   - 一位用户链接到了讨论此问题的 [Discord 线程](https://discord.com/channels/879548962464493619/1344268243648647218/1345503704215720087)。
- **推理额度消耗极快！**：一位用户报告称，尽管只完成了课程要求，但已耗尽了 Inference Providers 的**每月包含额度**。
   - 另一位用户建议这可能与在 Google 账户之间切换或内核停止并重新运行有关，他们被迫升级到了 **PRO**。
- **ToolCallAgent 问题浮现**：一位成员报告了在 smolagents 中使用 **ToolCallAgent** 的 **MultiAgent 架构** 时出现的问题，即回退到具有 Web 访问权限的子 Agent 失败。
   - 具体而言，尽管子 Agent 拥有必要的工具，但 Manager Agent 无法将 Web 搜索任务委托给子 Agent。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/organizations/labs-lambda-go/share/veuzHcLLGKoMrDdpMPltGTSMfPlSyOoNtN)">Hugging Face – 构建未来的 AI 社区。</a>：未找到描述</li><li><a href="https://tenor.com/view/dancing-cowboy-finger-gun-pointing-moves-gif-17170914">Dancing Cowboy GIF - Dancing Cowboy Finger Gun - Discover &amp; Share GIFs</a>：点击查看 GIF
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[open-r1](https://discord.com/channels/879548962464493619/1333465203865817088/1346170476514377809)** (2 messages): 

> `Replicant 模型训练，用于编程任务的 R1 推理数据集` 


- **Replicant 模型训练暂停**：研究团队暂停了为训练 **Replicant 模型**而进行的 **25 PB 数据集**的程序化生成。
   - 团队不得不回归其他工作，而不是继续创建该数据集。
- **寻找 R1 编程数据集的需求出现**：一位成员询问是否存在用于**编程任务**的 **R1 推理数据集**。
   - 该问题暂无回答。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1345979785133953045)** (121 messages🔥🔥): 

> `Aider 排行榜, Claude Code, Grok vs O3 Mini, anon-kode, python + uv` 


- **Aider 排行榜工具基准测试**：Aider 排行榜用于在使用 Aider 作为主要工具/助手时对 AI 模型进行基准测试；Claude Code 被视为工具/助手而非 AI 模型，并同样进行了基准测试。
   - 一位用户指出需要像 SWE Benchlets 这样与工具无关的基准测试来进行比较。
- **Anon-Kode 分叉 Claude Code，实现 OpenAI 兼容**：一位成员分享说，获取了 Claude Code 源码的人（[原始推文链接](https://x.com/dnak0v/status/1894254049802744188)）现在发布了一个修改版本，可以与兼容 OpenAI 的 API 配合使用（[推文链接](https://x.com/dnak0v/status/1896652107857748086)），并已在 [GitHub](https://github.com/dnakov/anon-kode) 上可用。
- **在个人项目中设置 Aider**：一位用户咨询了关于在个人项目的 Git 仓库和虚拟环境中使用 Aider 的建议。
   - 其他人建议全局安装 Aider 或安装在项目的 venv 中，并使用 uv 等工具来同步环境。
- **Grok 调试能力 > O3 Mini 代码创作能力？**：一位用户提到 Grok 擅长调试，但 O3 mini high Sonnet 在代码创作（如添加新功能）方面可能更好。
   - 他们还注意到 Claude 3.7 会添加一些非预期的内容，而 deepseek-chat 配合 O1 Pro 作为编辑器对他们来说几乎 95% 的情况下都表现良好。
- **LLM 审查**：一些成员讨论了语言模型中日益增加的审查，特别是在内核级代码生成方面，以及需要将 Prompt 标记为“仅用于教育”以绕过限制。
   - Grok3 似乎拒绝了该请求，看来仍然存在限制。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/aravsrinivas/status/1896463244229034338?s=46&t=AZs45ckJ7UUM_kJZcxnR_w">Aravind Srinivas (@AravSrinivas) 的推文</a>：如果有人想构建一个带有编辑器集成和扩展的开源 Claude-Code，Perplexity 很乐意提供免费的 API 额度。请私信 @GregFeingold 和 @AarashHeydari</li><li><a href="https://x.com/dnak0v/status/1894254049802744188>)">Daniel Nakov (@dnak0v) 的推文</a>：提取了 claude-code 未混淆的 TS 文件，仓库见评论</li><li><a href="https://x.com/dnak0v/status/1896652107857748086>):">Daniel Nakov (@dnak0v) 的推文</a>：有很多东西需要修复，但你可以使用任何支持 OpenAI 风格 API 的模型。如果你够勇敢，可以尝试一下。清理完代码后会上传源码 npm i -g anon-kode cd your-project k...</li><li><a href="https://github.com/dnakov/anon-kode">GitHub - dnakov/anon-kode</a>：通过在 GitHub 上创建账号来为 dnakov/anon-kode 的开发做出贡献。</li><li><a href="https://simonwillison.net/2024/Dec/19/one-shot-python-tools/">使用 uv run 和 Claude Projects 通过单次 Prompt 构建 Python 工具</a>：我写了很多关于如何通过 Claude Artifacts 使用 Claude 构建单次 HTML+JavaScript 应用程序的文章。我最近开始使用类似的模式来创建单次 Python 工具...</li><li><a href="https://simonwillison.net/2024/Aug/21/usrbinenv-uv-run/">#!/usr/bin/env -S uv run</a>：这是一个非常巧妙的模式。像这样开始你的 Python 脚本： #!/usr/bin/env -S uv run # /// script # requires-python = &quot;&gt;=3.12&quot; # dependencies = [ # &quot;flask==3.*&quot;, # …
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1345970092000673863)** (85 条消息🔥🔥): 

> `Gemini 2.0 Pro 模型问题，Aider + RAG/向量嵌入，Aider 使用 git diff 进行编辑，Aider 配合 OpenRouter 模型和编辑模式，Aider Architect 模式` 


- **`gemini/gemini-2.0-pro-exp-02-05` RESOURCE_EXHAUSTED 问题**：一位用户报告在 Aider 中使用具有大上下文窗口的 `gemini/gemini-2.0-pro-exp-02-05` 模型时遇到 `RESOURCE_EXHAUSTED` 错误，而 `gemini-2.0-flash-thinking-exp-01-21` 模型运行正常。
   - 他们询问是否有办法在没有限制的情况下使用具有完整上下文窗口的 Pro 模型。
- **用户通过使用 `workon` 命令的 Aider 扩展解决了问题**：一位用户创建了一个带有新 `/workon` 命令的 Aider 扩展，该命令可以分析文件中的导入，并将相关文件列表传递给 `add` 命令。
   - 该用户声称这节省了大量时间，并且他们已经实现了 TS、Vue、Kotlin 和 Java 的版本，但其被描述为“相当简陋（rather ugly）”，且不支持 `--subtree-only`。
- **请求 Aider 使用 Git Diff 语法编辑文件**：一位用户希望 Aider 直接在文件中使用 **git diff 语法**（如 `<<<<<< branch`, `======`, `>>>>>>> replace`）编辑文件，而不是仅在终端中显示并替换文本。
   - 他们希望能够在接受更改之前对其进行编辑，但其他人指出，如果不 fork 该项目，这是不可能实现的，此外也可以使用内置的 IDE diff 工具。
- **针对 Aider 的 Sonnet 与 OpenRouter 推荐**：一位用户请求推荐可与 Aider 编辑模式配合使用且价格不太昂贵的模型，他们发现 **o1-preview** 既昂贵又低效。
   - 另一位用户建议使用 `r1-free` 或 `gemini flash thinking` 进行规划，使用 `Sonnet 3.7` 进行执行，并分享了他们的 [aider 配置代码片段](https://example.com/aider.conf)，其中包含模型别名和编辑格式设置。
- **Aider 的 Git 命令**：一位用户描述了一种创建运行 Aider 脚本的技术，使用 `--load` 在启动时加载带有命令的脚本，并添加 `/run git diff` 以更新最新更改，这对于在分支上工作非常有用。
   - 另一位用户建议 Aider 可以使用 `git apply mypatch.diff` 来应用更改，而不是让 LLM 手动编辑整个文件，并增加一个 `--check` 步骤。



**提到的链接**：<a href="https://github.com/lutzleonhardt/copilot-proxy">GitHub - lutzleonhardt/copilot-proxy: Copilot Proxy 是一个 Visual Studio Code 扩展，它通过 Express 服务器公开 VS Code Language Model API。此实验性扩展仅用于研究和原型设计目的，不应在生产环境中使用。</a>：Copilot Proxy 是一个 Visual Studio Code 扩展，它通过 Express 服务器公开 VS Code Language Model API。此实验性扩展仅用于研究和原型设计目的...

  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1345988883464327220)** (1 条消息): 

> `视觉模型，基于 Attention 的 ViTs，MLP-Mixer` 


- **MLP-Mixer：ViTs 的替代方案**：一位成员质疑为什么 [MLP-Mixer](https://arxiv.org/abs/2105.01601) 没有更频繁地用于视觉模型。
- **基于 Attention 的 ViTs 仍然盛行**：该成员指出，基于 Attention 的 **ViTs** 仍然是 **SOTA 视觉模型** 的标准。



**提到的链接**：<a href="https://arxiv.org/abs/2105.01601">MLP-Mixer: An all-MLP Architecture for Vision</a>：卷积神经网络 (CNNs) 是计算机视觉的首选模型。最近，基于注意力机制的网络（如 Vision Transformer）也变得流行起来。在本文中，我们展示了...

  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1346007982420529236)** (54 条消息🔥): 

> `SRAM 与 Cache 混淆、Triton 标量常量数据类型、CUDA 后端超参数、Triton 自动调优资源、Triton BLAS 实现` 


- **SRAM 与 Cache 澄清**：一场讨论澄清了 **SRAM**、**registers**（寄存器）、**shared memory**（共享内存）和 **cache**（缓存）之间的关系，指出寄存器、共享内存和缓存是基于 SRAM 构建的芯片/软件层级属性。
   - 任何未分配的共享内存实际上都会变成 **L1 cache**。
- **Triton 中的 Cache 层级控制**：讨论提到虽然 Triton 不提供对缓存层级（**L1/L2**）的直接控制，但 `tl.load` 中的 `cache_modifier` 参数允许指定是从 L1 还是 L2 命中。
   - `ca` 和 `cg` 的区别在于 `cg` 指定加载应仅从 **L2** 命中，而不经过 **L1** 缓存。
- **CUDA Cache 一致性深度解析**：CUDA 文档指出全局数据在 **L2** 层级是相干的，但多个 **L1** 缓存对于全局数据是不相干的。
   - 解释提到 L2 缓存由所有 SM 共享，但每个 SM 拥有独立的 L1 缓存，可以使用 threadfence 将 L1 缓存的写入刷新到全局内存。
- **Triton 中的标量常量数据类型**：用户询问如何在 Triton 中指定标量常量的数据类型，以避免在位与（bitwise AND）等操作中发生意外的向上转型（upcasting）。
   - 另一位用户建议先应用掩码，然后再次向下转型为 int8：`x=(x&mask).to(tl.int8)`。
- **Triton 中 INT8 的块缩放 Matmul**：用户寻求在 Triton 中为 INT8 实现块缩放矩阵乘法（block scaled matrix multiplication）的指导，参考了 [FP4 和 FP8 格式的教程](https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html)。
   - 用户在尝试应用缩放时因隐藏维度（hidden dimension）要求遇到了 `tl.dot` 错误，并寻求处理 `scales` 部分的建议。



**提及的链接**：<a href="https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html">Block Scaled Matrix Multiplication &mdash; Triton  documentation</a>：未找到描述

  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1346093203475923005)** (25 条消息🔥): 

> `CUTLASS 中的 FP8 GEMM、确定 NVCC 的架构、Flash Attention 索引` 


- **实现带有 rowwise scales 的 FP8 GEMM**：一位成员正在寻找在 **CUTLASS** 中为 **sm100** 实现的带有 **rowwise scales**（行级缩放）的 **fp8 gemm** 示例，最终找到了解决方案，不再需要进一步帮助。
- **Torch 工具帮助确定 CUDA 架构**：一位成员需要为其构建系统确定当前系统的 `--arch=`，另一位成员建议使用 **PyTorch** 的 `torch.cuda.get_device_capability()` 工具。
   - 通过利用 `nvidia-smi --query-gpu=name,compute_cap --format=csv` 直接查询 **GPU 的计算能力（compute capability）** 找到了替代方案，从而避免了对 **PyTorch** 的依赖。
- **Flash Attention 索引困扰开发者**：一位成员请求关于 **Flash Attention** 中 **indexing**（索引）的帮助，特别是在如何实现 kernel 的该部分时感到困惑。
- **CUDA Runtime API 获取设备属性**：一位成员发现了 [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0)，可以通过编程方式选择最符合特定标准的计算设备。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/docs/stable/generated/torch.cuda.get_device_capability.html">torch.cuda.get_device_capability &mdash; PyTorch 2.6 documentation</a>：未找到描述</li><li><a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0)">CUDA Runtime API :: CUDA Toolkit Documentation</a>：未找到描述
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1346000358534807582)** (17 messages🔥): 

> `FSDP2 OffloadPolicy, register_post_accumuate_grad_hook, load_inline CUDA kernels, reduce and not scatter, optimizer scaling` 


- **用户寻求灵活的 FSDP2 OffloadPolicy**：一位用户询问了关于 **FSDP2** `OffloadPolicy` 类的计划，希望能更灵活地控制梯度处理，特别是实现 *reduce and not scatter*。
   - 回复指出目前没有立即的计划，但建议探索 `register_post_accumuate_grad_hook`，尽管它在 reduce scatter 之后运行，而这正是用户想要避免的。
- **CUDA Kernel 启动导致非法内存访问**：一位用户报告了在使用 `load_inline` 将内存指针直接传递给 CUDA 代码时出现 **illegal memory access errors**（非法内存访问错误）的问题。
   - 另一位用户建议，**PyTorch CUDA caching allocator** 可能会分配比需求更多的内存，这可能导致直接指针版本中出现越界读取，而 Tensor 版本由于有边界检查而能正常工作。
- **探索用于梯度聚合的 All-Reduce**：一位用户提出了一种替代方案，即使用 **All-Reduce** 来聚合梯度，随后立即应用 Optimizer 并将梯度清零，以避免 Offloading。
   - 一位回复者质疑该方案的可扩展性，因为与 reduce-scatter 相比，其 **通信量增加了 2 倍**，并建议将梯度收集到更大的块（blocks）中。
- **用户考虑参数缩放方法**：一位用户针对 **单节点** 上的缩放，探索了诸如 scatter 到 CPU 并累加，或者在 CPU 的 rank 0 上进行 gather 以执行 Optimizer 步骤等选项。
   - 目标是寻找更快的途径，强调了对系统中扩展点（extensibility point）的需求。


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1346118446177320981)** (5 messages): 

> `fa3, absmax quantization, hada transform` 


- **FA3 可运行但量化误差高**：在解决一些初始问题后，据报告 **FA3** 已经可以运行，但其量化误差明显高于基础的 **absmax quantization**。
   - 建议在 **hada transform** 之后进行 **absmax quantization**，特别是针对 'v'，以避免由于大激活值导致的分布外（out-of-distribution）问题。
- **提议量化策略转变**：为了缓解 **FA3** 的量化挑战，提议了一种策略转变，即在 **Hada transform** 之后应用 **absmax quantization** 以获得更好的性能。
   - 重点特别放在 'v' 分量上，如果没有在变换后进行适当量化，该分量容易出现大激活值和分布外行为。


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1346005396430651432)** (1 messages): 

> `Internship Opportunity, Low-Level Programming, LLM Inference, Mobile and PC Platforms` 


- **实习岗位寻求底层 LLM 推理**：一位用户正在为移动端和 PC 平台寻找从事 **底层编程（low-level programming）** 以改进 **LLM 推理** 的实习生，感兴趣的人员请访问 [此 GitHub 仓库](https://github.com/githubpradeep/llm_np_cp)。
   - 该仓库托管了在 **cupy** 和 **numpy** 上运行 **llama gemma** 的代码。
- **GitHub 仓库专注于 llama gemma**：提供的 [GitHub 仓库](https://github.com/githubpradeep/llm_np_cp) 专注于在 **cupy** 和 **numpy** 上运行 **llama gemma**。
   - 这表明其重点在于使用数值计算库优化 **LLM 性能**。



**提到的链接**：<a href="https://github.com/githubpradeep/llm_np_cp.git">GitHub - githubpradeep/llm_np_cp: running llama gemma on cupy and numpy</a>：在 cupy 和 numpy 上运行 llama gemma。可以通过在 GitHub 上创建账号来为 githubpradeep/llm_np_cp 的开发做出贡献。

  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1345977876121522223)** (5 messages): 

> `Triton tensor creation, ROCm support for RX 7800 XT, NVIDIA GPU alternatives` 


- **创建标量的 Triton 张量**：一位用户询问如何在 Triton 中显式创建**标量张量**以指定数据类型。
   - 他们尝试了 `mask = tl.tensor(0xF, type=tl.uint16)`，但反馈该方法不起作用。
- **RX 7800 XT ROCm 的困扰**：一位刚接触 GPU 编程和 AI 的用户报告称，其 **NITRO+ AMD Radeon™ RX 7800 XT 16GB** 在适配 PyTorch 和其他 AI 库时遇到问题。
   - 他们指出 **ROCm** 不支持低于 **7900** 系列的显卡。
- **NVIDIA GPU：一个可行的替代方案？**：一位成员开玩笑地建议“卖掉你的 GPU 买 NVIDIA”，随后提供了获取 NVIDIA GPU 的严肃替代方案。
   - 他们推荐了 [lightning.ai](https://lightning.ai/) 等平台以获取免费的每月 GPU 额度（可用性待确认），以及 [runpod](https://runpod.io/) 和 [vast.ai](https://vast.ai/) 等价格实惠的 GPU 租赁服务。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1346183460498309172)** (8 messages🔥): 

> `Tilelang Kernel, Deepseek flashmla, MLA leaderboard, Bitnet group` 


- **Tilelang Kernel 媲美 Deepseek flashmla**：一位成员分享称，只需 **80 行 tilelang kernel 代码**，即可在 H100 上获得 **Deepseek flashmla 95% 的性能**（比 **Triton 快 500%**），并附带了 [GitHub 仓库](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_mla)链接。
- **对 MLA 排行榜的期待**：一名成员表示希望能有一个 **MLA 排行榜**，以便大家展示性能。
   - 该成员还询问另一人是否有兴趣加入一个**工作组**，并重新启用 **bitnet 小组**。
- **Tilelang Kernel 获得积极反馈**：一位成员评价该 kernel 写得非常好。
   - 该成员还表示，相关文档本身就应该作为一篇**教程/博客**发布。



**提到的链接**：<a href="https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_mla">tilelang/examples/deepseek_mla at main · tile-ai/tilelang</a>：专为简化高性能 GPU/CPU/加速器 kernel 开发而设计的领域特定语言 - tile-ai/tilelang

  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1346017632393887745)** (11 messages🔥): 

> `Chain of Draft PR, Throttling errors` 


- **Chain of Draft PR：草稿式思考让 LLM 快速推理**：一位成员提交了一个 [PR](https://arxiv.org/abs/2502.18600)，增加了**两种新的系统提示词风格**，类似于 **Chain of Draft** 论文中的风格。
   - [Chain of Draft 论文](https://arxiv.org/abs/2502.18600v2)引入了一种范式，让 LLM 生成极简的中间推理输出，从而减少冗余和成本。
- **评测脚本因日志升级受到赞扬**：成员们非常感谢修复评测脚本的人（称其为 goat），现在的日志记录效果好得多。
   - 然而，目前仍有一个关于 **节流 (throttling) 错误** 的待处理问题需要修复，且尚未实现结果的中间保存功能。



**提到的链接**：<a href="https://arxiv.org/abs/2502.18600">Chain of Draft: Thinking Faster by Writing Less</a>：大语言模型（LLM）通过 Chain-of-Thought (CoT) 提示词等机制，在解决复杂推理任务方面表现出了卓越的性能，这些机制强调冗长的、逐步的推理...

  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/1346051959647637515)** (4 messages): 

> `Tilelang, MLA, FlashMLA, Python` 


- **Tilelang 被推崇为 MLA/FlashMLA 的替代方案**：一位成员建议“直接 all in tilelang”，声称它和 **MLA** 以及 **flashMLA** 一样快，但只需要 **80 行 Python 代码**。
- **对 tilelang 建议的热烈响应**：一位成员感谢了原作者的分享，并表示非常渴望学习 **tilelang**。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1346090357737127976)** (1 messages): 

> `prefixsum submission, H100 submission` 


- **请求清除 Prefixsum 提交记录**：一位用户由于命名不一致，请求删除他们在 **prefixsum 上的顶级 H100 提交记录**。
   - 该用户认为删除此提交将有助于更轻松地跟踪变更。
- **H100 Prefix Sum**：关于 Prefix Sum 顶级 H100 提交的讨论。
   - 一位用户请求删除该提交以便更好地跟踪变更。


  

---

### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1346122213467750430)** (17 条消息🔥): 

> `排行榜提交, 排行榜名称不匹配, 成功提交, GPU 使用情况` 


- **Cluster-Bot 标记排行榜名称冲突**：Cluster-Bot 检测到命令中指定的排行榜名称与提交脚本头中的名称不匹配，默认提交到指定的名称（例如 `grayscale`, `vectorsum`, `vectoradd`）。
- **Modal 运行器产生成功提交**：多个排行榜提交在各种 GPU 上使用 Modal 运行器成功，包括用于 `conv2d` (ID `1509`) 的 **H100**，用于 `matmul` (IDs `1510`, `1512`) 的 **T4**，以及用于 `vectorsum` (ID `1516`) 的 **A100**, **L4**, **H100**。
- **缺少排行榜名称导致 Cluster-Bot 报错**：Cluster-Bot 提示用户通过命令参数或在提交脚本中使用 `{#,//}!POPCORN leaderboard <leaderboard_name>` 指令来指定排行榜名称。
- **Vectoradd 排行榜出现多次测试和基准测试提交**：使用 **A100** 和 **H100** GPU 以及 Modal 运行器，多次向 `vectoradd` 排行榜进行的测试（ID `1526`, `1528`）和基准测试（ID `1527`, `1529`）提交均获成功。


  

---


### **GPU MODE ▷ #[ppc](https://discord.com/channels/1189498204333543425/1343373905783554048/1345970969445138453)** (3 条消息): 

> `AVX512, FMA 指令, 性能提升` 


- **AVX512 在 FMA 指令中直接广播**：一名成员建议探索 **AVX512** 在 ||**FMA 指令**|| 中直接执行广播的能力，以寻求潜在的性能提升。
   - 另一名成员承认他们之前没有考虑到这一点，并将进一步调查，这表明了利用 **AVX512** 特性的潜在兴趣。
- **寻求通过 AVX512 实现大幅改进**：一名成员表示需要一种*截然不同或全新的方法*来实现下一阶段的改进。
   - 探索具有直接广播功能的 **AVX512 FMA 指令** 被作为实现这一重大进展的潜在途径提出。


  

---


### **GPU MODE ▷ #[feature-requests-and-bugs](https://discord.com/channels/1189498204333543425/1343759913431728179/1346247487333077063)** (10 条消息🔥): 

> `L4 & T4 超时, AMD MI300s, Beta 发布` 


- **L4/T4 超时问题依然存在**：由于开发延迟，**L4** 和 **T4** 在编译过程中的超时问题尚未解决。
   - 一名成员提到由于团队正在处理*更具现实影响力的有趣问题*，因此进度略有延迟。
- **MI300s 发布可能即将到来！**：与 **AMD** 的合作状态有所改善，因此团队有望推出 **MI300s** 作为选项。
   - 团队强调这*不代表任何承诺*，但它是一个选项。
- **Beta 发布取得巨大成功**：团队对 Beta/Alpha 发布表示满意，并决定比原计划更早地处理更有趣的问题。
   - 此次发布原定于 4 月底。


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1346055672781082738)** (1 条消息): 

> `旅行短视频, AI Agent, 行程规划` 


- **用于保存旅行短视频的应用问世**：一款新应用旨在解决在社交媒体上保存旅行短视频（Reels）后，又得浪费数小时手动研究每个地点的死循环。
   - 该应用 ([https://thatspot.app/](https://thatspot.app/)) 使用 **AI Agent** 自动处理旅行短视频，提取其中提到的每个地点，包括**位置、价格范围、预订要求、预订链接和营业时间**。
- **AI Agent 自动化旅行研究**：该应用利用 **AI Agent** 简化了从保存的旅行短视频中规划行程的手动研究过程。
   - 它能直接从旅行短视频中自动提取**精确位置、价格范围、预订要求、直接预订链接和营业时间**。



**提及的链接**：<a href="https://thatspot.app/">ThatSpot Guide</a>：未找到描述

  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1345971238601752698)** (126 条消息🔥🔥): 

> `Google Flash 2.0 错误, Claude 3.7 Sonnet 速率限制, OpenRouter API Key 与 VS Studio/RooCode, OpenRouter 中的 BYOK Azure 模型, 在聊天模型中访问链接` 


- **Google Flash 2.0 报错**：一位用户报告在使用 **Google 的 Flash 2.0 和 Flash 2.0 Light 模型**进行推理时收到 **502 错误**，错误信息为 *"Provider returned error"*，且 Google 遇到了内部错误。
   - 一位成员建议将该请求发布到相应的 Discord 频道。
- **讨论 Claude 3.7 Sonnet 的速率限制**：一位用户询问了 **Claude 3.7 Sonnet** 在 **RPM (Requests Per Minute)** 和 **TPM (Tokens Per Minute)** 方面的速率限制。
   - 一位成员表示 OpenRouter 对单个用户没有特定的速率限制，如果触发了速率限制，通常是 OpenRouter 的限制，该限制高于 Tier 4（参见 [Anthropic 的速率限制文档](https://docs.anthropic.com/en/api/rate-limits)）。
- **在 VS Studio/RooCode 中使用 OpenRouter API Key 遇到困难**：一位用户在通过 **RooCode** 在 **VS Studio** 中使用 OpenRouter API Key 时遇到了 **401 Authentication Failure**（身份验证失败），尽管其 OpenRouter 账户中有余额。
   - 成员们建议检查 API Key 是否正确，确保在 RooCode 中选择了 OpenRouter 作为 API 提供商，并根据[此教程](https://www.vincentschmalbach.com/adding-models-to-cursor/)验证 Base URL 配置是否正确。
- **在 OpenRouter 中请求 BYOK Azure 模型**：一位用户询问在 OpenRouter 中对 **Azure 模型**使用 BYOK (**Bring Your Own Key**)，旨在通过 OpenRouter 使用统一的 API 来调用微调模型。
   - 一位成员表示，无法使用 `/models` 端点列出之外的模型，该端点仅返回公共模型，不包括 BYOK 模型。不过，你可以在集成设置中（[OpenRouter Integration Settings](https://openrouter.ai/settings/integrations)）使用你自己的 OpenAI API Key。
- **探索 OpenRouter 延迟的迷宫**：一位用户询问如何改善 OpenRouter 上的 **Time to First Token (TTFT)** 延迟，并指出他们发现 OpenRouter 的平均 TTFT 是直接使用提供商的两倍。
   - 一位团队成员请该用户将他们的发现整合到一篇论坛帖子中，并提到降低延迟目前是高优先级任务。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/1091220969173028894/1346285793953583187">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord 是玩游戏、与朋友放松甚至建立全球社区的绝佳场所。定制你自己的空间来交流、玩耍和聚会。</li><li><a href="https://openrouter.ai/chat">Chatroom | OpenRouter</a>: LLM 聊天室是一个多模型聊天界面。添加模型并开始聊天！聊天室将数据本地存储在你的浏览器中。</li><li><a href="https://openrouter.ai/docs/api-reference/parameters#response-format">API Parameters - Complete Guide to Request Configuration</a>: 了解 OpenRouter API 请求的所有可用参数。配置 temperature, max tokens, top_p 以及其他模型特定设置。</li><li><a href="https://stspg.io/3h7p3k2t0hrm">Elevated errors on on requests</a>: 未找到描述</li><li><a href="https://x.com/imrat/status/1837884034094874741">Tweet from Imrat (@imrat)</a>: 1. 在 Settings > Models 中使用你的 OpenRouter API Key 和 OpenRouter Base URL 设置 OpenAI API Key；2. 确保添加了正确的模型；3. 当我想使用 OpenRouter 模型时 - CMD+Shift+0 (zer...</li><li><a href="https://openrouter.ai/docs/api-reference/list-endpoints-for-a-model">List endpoints for a model — OpenRouter | Documentation</a>: 未找到描述</li><li><a href="https://sdk.vercel.ai/docs/ai-sdk-ui/object-generation#schema">Object Generation</a>: 了解如何使用 useObject hook。</li><li><a href="https://www.vincentschmalbach.com/adding-models-to-cursor/">no title found</a>: 未找到描述</li><li><a href="https://forum.cursor.com/t/cursor-0-46-7-pro-openrouter-key-not-working/57203/3">Cursor 0.46.7 Pro: Openrouter Key not working</a>: 嘿，你的 API Key 仅在聊天模式（也称为 Ask 模式）下工作。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1346196892765261938)** (1 messages): 

> `LM Studio SDK, Python, TypeScript, Agent API, MIT License` 


- **LM Studio SDKs 现已支持 Python 和 TypeScript！**: LM Studio 推出了适用于 **Python** ([`lmstudio-python`](https://github.com/lmstudio-ai/lmstudio-python)) 和 **TypeScript** ([`lmstudio-js`](https://github.com/lmstudio-ai/lmstudio-js)) 的软件开发工具包 (SDK)，两者均采用 **MIT 许可证**。
   - 这些 SDK 允许开发者在自己的代码中调用 LM Studio 的 AI 能力，包括 LLMs、embeddings models 和 agentic flows。
- **LM Studio 推出面向智能体的 .act() API**: LM Studio 推出了其首个面向智能体的 API，即 **`.act()`** 调用，模型可以使用提供的工具在多轮交互中自主执行任务。
   - 该 API 允许你提供提示词和工具，模型将自主进行多轮执行 *rounds*，直到完成任务（或放弃）。
- **LM Studio SDK 文档现已上线**: LM Studio 发布了 **Python** ([`lmstudio-python`](https://lmstudio.ai/docs/python)) 和 **TypeScript** ([`lmstudio-js`](https://lmstudio.ai/docs/typescript)) SDK 的文档，为与 **LLMs**、**embedding models** 和 **agentic flows** 的交互提供资源。
   - 该 SDK 允许你使用 LLMs 进行聊天响应或预测文本补全，将函数定义为工具，并将 LLMs 转换为完全本地运行的自主 Agents。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/blog/introducing-lmstudio-sdk">Introducing lmstudio-python and lmstudio-js</a>: Python 和 TypeScript 的开发者 SDK 现已发布 1.0.0 版本。一个用于本地 AI 软件的可编程工具包。</li><li><a href="https://lmstudio.ai/docs/python">lmstudio-python (Python SDK) | LM Studio Docs</a>: LM Studio Python SDK 入门指南</li><li><a href="https://lmstudio.ai/docs/typescript">lmstudio-js (TypeScript SDK) | LM Studio Docs</a>: LM Studio TypeScript / JavaScript SDK 入门指南
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1345989853686267985)** (100 条消息🔥🔥): 

> `Context Length 错误，Llama.cpp 不支持的模型架构，LM Studio CLI 命令，LM Studio SDK，LM Studio 降级` 


- **LM Studio 模型加载失败并显示 "Unsupported device" 错误**：用户在更新 LM Studio 后遇到 `Failed to load model` 错误，并提示 `Unsupported device`。这可能是由于内存不足或模型加载设置不兼容导致的，建议尝试 [**调整 GPU offloading 或 thread pool size**](https://cdn.discordapp.com/attachments/1110598183144399061/1345995998471651348/image.png?ex=67c7e575&is=67c693f5&hm=a02e91bb46a53c67002d1787ae998888664e844574cba61a9e626bbe96a1ff6a&)。
   - Context length 也会影响内存占用；左侧数字是模型在 **chat history** 中已使用的 **tokens** 数量，右侧数字是 **context limit**（即内存开始截断前的上限）。
- **Llama.cpp 不支持 Diffusion 模型架构**：用户在尝试加载扩散模型时收到 `error loading model architecture: unknown model architecture: 'sd3'` 错误。已明确 [**llama.cpp 不支持图像/视频/音频生成模型**](https://cdn.discordapp.com/attachments/1110598183144399061/1345997397091811448/image.png?ex=67c7e6c2&is=67c69542&hm=93ffdc9a858570aeb67c46cc86eb5a391f447d55cbdb54b94b587bf29f609cd0&)。
   - `llama.cpp` 对 Vision 模型的支持尚不确定，目前存在对缺乏 **Llama 3.2 vision** 或 **Pixtral vision 支持**的担忧，不过一些人认为 **UI-TARS 修复** 会有很大帮助。
- **Pseudollama 弥合了 OLLAMA 的差距**：有用户询问 LM Studio 端点是否与接受 OLLAMA 端点的应用兼容。回答是默认情况下不支持，但 [**Pseudollama**](https://github.com/verbiate/Pseudollama) 可以起到桥接作用。
   - 作者提到 *这完全是凭感觉编写的代码 (vibe coded)，所以可能存在一些低级问题，但它是可以运行的。*
- **LM Studio SDK 文档发布**：随着 LM Studio 的最新版本发布，[LM Studio CLI 命令](https://lmstudio.ai/docs/cli) 已同步文档化。用户确认 **OpenAI API** 将继续得到支持和优先考虑。
   - 社区成员表示他们 *一直在等待这个功能，以便开发插件*，并提到想 *制作一个手表应用通信器。*
- **用户找到降级 LM Studio 的方法**：由于新版本取消了在两张显卡之间进行 **tensor split** 的预设功能，一位用户需要降级到 **0.3.10** 版本。
   - 另一位用户建议使用 [**web archive**](https://web.archive.org/web/20250221141151/https://lmstudio.ai/) 查找下载链接，还有人建议直接 *修改下载链接的参数。*


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/docs/cli">lms — LM Studio 的 CLI | LM Studio 文档</a>: 开始使用 lms 命令行工具。</li><li><a href="https://github.com/verbiate/Pseudollama">GitHub - verbiate/Pseudollama</a>: 访问 GitHub 参与 verbiate/Pseudollama 的开发。</li><li><a href="https://x.com/lmstudio/status/1896581603171963197">来自 LM Studio (@lmstudio) 的推文</a>: 开发者们请注意，未来几小时将有重大更新！🔨⏲️</li><li><a href="https://github.com/ggml-org/llama.cpp">GitHub - ggml-org/llama.cpp: C/C++ 环境下的 LLM 推理</a>: C/C++ 环境下的 LLM 推理。访问 GitHub 参与 ggml-org/llama.cpp 的开发。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1346003948468830269)** (21 messages🔥): 

> `AMD and Intel vs CUDA, Vulkan vs CUDA, AMD GPU market share, Nvidia 5090 specs` 


- **AMD 和 Intel 可能与 CUDA 竞争**：成员们讨论了 **AMD** 或 **Intel** 是否能成为 **ML** 流水线和框架的可行选择，从而与 **CUDA** 竞争。
- **Vulkan 不是 CUDA 的竞争对手**：成员们提到 **Vulkan** 是一个图形 API，而 **CUDA** 是为 **GPGPU** 计算任务构建的，因此*使用 Vulkan 进行计算就像使用 dx12 进行计算：你可以这样做，但它有意义吗？*
   - 一位成员表示，*竞争以及拥有 Vulkan 的替代方案对消费者来说只有好处*。
- **AMD 需要更高的市场份额来投资 GPU Compute**：一些成员认为，如果 **AMD** 增加其市场份额，他们会更有兴趣投资其 **GPU** 计算部门。
   - 一位成员认为他们一直满足于持有 **10%** 的份额。
- **正在寻求 Nvidia 5090 的规格参数**：一位成员回忆起看到过一些 **Nvidia 5090 FP8/16/32** 的规格参数，并询问在哪里可以找到。
   - 一位成员还分享了他们 **4x 3090 gang** 配置的照片。
- **芯片代工厂产能获取**：一位成员认为真正的问题在于 **AMD** 是否能从芯片代工厂购买到产能，因为 Nvidia 占据了上风。
   - 另一位成员表示 *Intel 将在 2030 年左右开设下一个工厂，所以对 AMD 不抱期望*。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1345969355460055101)** (93 messages🔥🔥): 

> `Low-rank space reasoning, Nous API, CUDA Kernels, Hermes 3 erotic fiction, Ollama usability` 


- **低秩空间使 PEFT 变得合理**：一位成员建议，由于与基础模型的推理差异通常存在于**低秩空间（low-rank space）**中，因此使用 **PEFT** 进行训练成为一种合理的方法，并进一步建议使用 **Qwen 0.5B** 进行低成本测试。
   - 其他人同意，正因如此，**low vram unsloth RL trainer** 似乎效果很好。
- **Nous API 要来了吗？**：一位成员建议 Nous 提供自己的 API 来使用他们的模型以产生收入，特别是考虑到目前的收入并不稳定，建议定价为 **$0.8/M tokens**，并估计潜在收入为 **$800-1600/天**。
   - 其他人建议 Nous 对于 forge 可以收取接近 **$1/M input tokens, $3/M output tokens** 的费用，另一些人指出目前正在努力实现这一目标。
- **LLM 难以创建高性能的 CUDA kernels**：成员们讨论了用 LLM 生成 **CUDA kernels**，共识是虽然 LLM 可以输出 CUDA 语法，但*没有哪个 LLM 能独立生成高性能的 kernels*。
   - 最佳策略似乎涉及利用硬件和计算图信息来增强 LLM，可能使用**知识图谱或 GNN**，并采用包含广泛 **GPU profiling** 的半手动方法。
- **Hermes 3 编写情色小说**：一位用户称赞了 **Hermes 3** 在编写情色小说方面意想不到的天赋，并对未来的 **NousChud** 模型迭代表示期待。
   - 另一位成员提到他们总是有*一些项目在进行中*，但他们更倾向于那些无需数据中心即可运行的模型。
- **Ollama 因以初学者为中心的设计而受到批评**：虽然 **Ollama** 被认为适合初学者，但它被批评为*对于度过初学者阶段的人来说非常糟糕*，因为它即使对于 7B 和 8B 模型也默认使用 **Q4 quantization**。
   - 建议高级用户使用 **llama.cpp** 或 **koboldcpp** 等替代方案，但也承认配置和维护环境是一项不同的技能，一次性抛给新手可能负担过重。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://claude.ai/share/693d9c22-7f9a-40b4-82e4-603276347d9c">Claude</a>: 与 Claude 对话，来自 Anthropic 的 AI 助手</li><li><a href="https://github.com/ollama/ollama/blob/main/docs/api.md">ollama/docs/api.md at main · ollama/ollama</a>: 快速上手 Llama 3.3, DeepSeek-R1, Phi-4, Gemma 2 以及其他大型语言模型。 - ollama/ollama</li><li><a href="https://github.com/PsycheFoundation/psyche">GitHub - PsycheFoundation/psyche: 一个旨在为人类实现超级智能开发民主化和去中心化的开放基础设施。</a>: 一个旨在为人类实现超级智能开发民主化和去中心化的开放基础设施。 - PsycheFoundation/psyche</li><li><a href="https://docs.psyche.network/">Intro to Psyche - Psyche</a>: 暂无描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1345986646922760233)** (8 条消息🔥): 

> `Logic-RL, Rule-Based Reinforcement Learning, General World Models, Worldsim` 


- ****Logic-RL** 通过 **Rule-Based Reinforcement Learning** 释放推理能力**: 一篇新论文 ([Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning](https://arxiv.org/abs/2502.14768)) 受 **DeepSeek-R1** 启发，探索了 **rule-based reinforcement learning (RL)** 在大型推理模型中的潜力。
   - 该 **7B model** 仅在 **5K logic problems** 上进行训练，就展示了对具有挑战性的数学基准测试 **AIME 和 AMC** 的泛化能力。
- **Runway 推出 **General World Models****: Runway 推出了 [General World Models](https://runwayml.com/research/introducing-general-world-models)，设想 AI 系统构建环境的内部表示以模拟未来事件。
   - 他们的目标是表示和模拟广泛的情况和交互，超越 **video games** 或 **driving simulations** 等受限且受控的设置。
- **社区讨论 Generative AI 在 **Worldsim** 中的潜力**: 成员们讨论了 Generative AI 将整个世界模拟为交互式体验的潜力，暗示了涌现的 **LLM world models** 的深远能力。
   - 此外还有讨论称，继续 **worldsim** 的工作可能会以博客形式出现，尽管最终目标是将其写成论文。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.14768">Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning</a>: 受 DeepSeek-R1 成功的启发，我们探索了 rule-based reinforcement learning (RL) 在大型推理模型中的潜力。为了分析推理动态，我们使用合成逻辑谜题作为 t...</li><li><a href="https://runwayml.com/research/introducing-general-world-models">Runway Research | Introducing General World Models</a>: 我们相信 AI 的下一个重大进步将来自理解视觉世界及其动态的系统，这就是为什么我们正在围绕所谓的 ge... 启动一项新的长期研究工作。
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1346317553668067338)** (1 条消息): 

> `` 


- **未发现主题**: 在提供的消息中未发现相关主题。
- **无可用摘要**: 由于缺乏实质性内容，无法创建摘要。



**提到的链接**: <a href="https://app.towns.com/t/0x7189a0e937e76a33032dfe25d5f9b03d085da853/channels/207189a0e937e76a33032dfe25d5f9b03d085da8530000000000000000000000">San</a>: 未找到描述

  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1345986646922760233)** (8 条消息🔥): 

> `Rule-Based Reinforcement Learning (RL), DeepSeek-R1, Logic-RL, Worldsim, RunwayML 开发的 General World Models (GWM)` 


- **Logic-RL 通过 Rule-Based RL 释放 LLM 推理能力**：一篇新论文 [Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning](https://arxiv.org/abs/2502.14768) 探索了在大型推理模型中使用 **rule-based RL** 的方法，灵感源自 **DeepSeek-R1**。
   - 该论文强调了核心贡献，例如：*一个强调思考和回答过程的 System Prompt、一个惩罚走捷径输出的严格格式奖励函数（Format Reward Function），以及一个实现稳定收敛的简单训练方案*。
- **RunwayML 推出 General World Models (GWM)**：[RunwayML](https://runwayml.com/research/introducing-general-world-models) 正在启动一项围绕 **General World Models (GWM)** 的长期研究计划，旨在构建能够理解视觉世界及其动态以模拟未来事件的 AI 系统。
   - 他们认为 *AI 的下一个重大进步将来自能够理解视觉世界及其动态的系统*。
- **探索 LLM 的涌现世界模型**：讨论围绕 **LLM** 在大规模数据集训练中将构建 **World Models** 作为一种涌现属性（Emergent Property）的观点展开。
   - 对话旨在理解 *这些涌现世界模型的能力和局限性*，特别是 **General World Model** 如何增强体验或开启新的创意前沿。
- **Worldsim 在 Generative AI 中的深远潜力**：**Worldsim** 暗示了 **Generative AI** 作为创意媒介的潜力，能够将整个世界模拟为交互式体验。
   - 一位成员正准备继续 **Worldsim** 的工作，可能以博客形式呈现，但不确定是否将其写成论文。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2502.14768">Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning</a>：受 DeepSeek-R1 成功的启发，我们探索了 Rule-Based Reinforcement Learning (RL) 在大型推理模型中的潜力。为了分析推理动态，我们使用合成逻辑谜题作为测试...</li><li><a href="https://runwayml.com/research/introducing-general-world-models">Runway Research | Introducing General World Models</a>：我们相信 AI 的下一个重大进步将来自理解视觉世界及其动态的系统，这就是为什么我们正在启动一项围绕 General World Models 的长期研究计划...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/1346047069445554227)** (1 条消息): 

> `Qwen2.5-Math-1.5B, longcot 示例, 数据集结构化, 设置 GRPOTrainer` 


- **Qwen2.5-Math-1.5B 模型在 longcot 格式上遇到困难**：一位成员正在使用 **longcot 示例** 对 **Qwen2.5-Math-1.5B** 进行实验，但模型未遵循预期格式。
   - 该成员在数据集结构化和设置 **GRPOTrainer** 方面寻求帮助，并附上了他们的 [Kaggle notebook](https://www.kaggle.com/code/umangkaushik/qwen2-5-3b-limo) 链接。
- **数据集结构化和 GRPOTrainer 设置问题**：用户在使 **Qwen2.5-Math-1.5B** 模型遵循所需格式（使用 **longcot 示例** 时）面临挑战。
   - 他们怀疑问题出在数据集的结构或 **GRPOTrainer** 的配置上，并请求指导。


  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1345988184420651080)** (28 条消息🔥): 

> `Unitree Open Source, Gemma 3 Release, GPT-4.5 tops Arena leaderboard, Post-Training Interpretation, Anthropic $3.5B Funding` 


- ****Unitree Robotics** 开源仓库**: **Unitree Robotics** 开源了一系列代码库，并提供了其 [GitHub](https://github.com/unitreerobotics) 链接。
- ****Gemma 3** 发布**: **Gemma 3** 宣布将于 **3 月 12 日**发布。
- ****GPT-4.5** 登顶 Arena**: **GPT-4.5** 目前在 Arena 排行榜的所有类别中均排名第一，包括 **Multi-Turn**、**Hard Prompts**、**Coding**、**Math**、**Creative Writing**、**Instruction Following** 和 **Longer Query** [(来源)](https://x.com/lmarena_ai/status/1896590146465579105)。
- **激发 **Post-Training** 的潜力**: 过去 18 个月中，来自 **OpenAI**、**Anthropic** 和 **Google** 的模型的大部分改进都源于 **post-training phase**，类似于 F1 车队通过空气动力学和系统改进来提升赛车性能 [(来源)](https://x.com/natolambert/status/1896596516388979048)。
- ****Anthropic** 融资数十亿美元**: **Anthropic** 以 **615 亿美元**的 post-money valuation（投后估值）筹集了 **35 亿美元**，由 **Lightspeed Venture Partners** 领投 [(来源)](https://www.anthropic.com/news/anthropic-raises-series-e-at-usd61-5b-post-money-valuation)。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/ivanleomk/status/1896503301795181043">来自 Ivan Leo ( 🇺🇸 In NY 19 - 28 Feb! ) (@ivanleomk) 的推文</a>: 祈祷吧哈哈！巴黎现在非常令人兴奋，有很多活动正在进行！</li><li><a href="https://fxtwitter.com/teortaxesTex/status/1896654143718314165">来自 Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex) 的推文</a>: 截至太平洋标准时间中午 12 点，OpenAI 已悄悄修改了其在幻觉和准确性基准测试 PersonQA 上夸大的改进声明（左侧为新版本），报告了正确的...</li><li><a href="https://x.com/AnthropicAI/status/1896606683876753470">来自 Anthropic (@AnthropicAI) 的推文</a>: Anthropic 以 615 亿美元的投后估值筹集了 35 亿美元，由 Lightspeed Venture Partners 领投。这将推进我们对 AI 系统的开发，加深我们对其工作原理的理解...</li><li><a href="https://x.com/lmarena_ai/status/1896590150718922829">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>: GPT-4.5 在所有类别中全面领先，在 Multi-Turn 方面具有明显的领导地位。🥇 Multi-Turn💠 Hard Prompts💠 Coding💠 Math💠 Creative Writing💠 Instruction Following💠 Longer Query</li><li><a href="https://x.com/lmarena_ai/status/1896590146465579105">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>: 突发新闻：@OpenAI 的 GPT-4.5 现已登顶 Arena 排行榜！凭借超过 3000 张选票，GPT-4.5 在所有类别中均排名第一，并在 Style Control / Multi-Turn 类别中独占鳌头 🥇 巨大的祝贺...</li><li><a href="https://x.com/natolambert/status/1896596516388979048">来自 Nathan Lambert (@natolambert) 的推文</a>: 如果你观察过去 18 个月里我们从 OpenAI、Anthropic 和 Google 获得的大多数模型，你会听到很多“大部分改进都在 post-training 阶段”...</li><li><a href="https://x.com/lmarena_ai/status/1896675400916566357?s=61">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>: 📰今天还有更多令人兴奋的消息：@xai 最新的 Grok-3 登顶 Arena 排行榜！🔥这是最新的生产模型 grok-3-preview-02-24。凭借超过 3000 张选票，该模型在总榜上并列第一，并且...</li><li><a href="https://github.com/unitreerobotics">Unitree Robotics</a>: 高性能民用机器人制造商。请大家务必以友好且安全的方式使用机器人。- Unitree Robotics</li><li><a href="https://bsky.app/profile/digthatdata.bsky.social/post/3ljifc7mso22p">David Marx (@digthatdata.bsky.social)</a>: 我一直在思考的一个问题是，作为 post-training 的输入，多少 pre-training 投入是最佳的。在“elicitation”理论下，你会预期完整的 pre-training 是最佳的...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1345979913714536449)** (8 messages🔥): 

> `BobbyBroccoli 视频, Deep Learning 历史, 甘利俊一 (Shun-Ichi Amari)` 


- **X 称 Twitter 应用是免费的**：一名成员链接到一条推文，声明 *这个应用是免费的* [X Tweet](https://x.com/untitled01ipynb/status/1896417553825886269)。
   - 该成员质疑其他领域是否也是如此，并回答道：*答案一定是“不”，对吧*。
- **Juergen 的 Deep Learning 历史**：一名成员建议阅读 [Juergen Schmidhuber 的 Deep Learning 历史](https://people.idsia.ch/~juergen/deep-learning-history.html)。
   - 该成员还重点提到了 **甘利俊一 (Shun-Ichi Amari)**，称其为之前未曾听说但非常有趣的人物。



**提及链接**：<a href="https://x.com/untitled01ipynb/status/1896417553825886269">来自 loss (derogatory) (@untitled01ipynb) 的推文</a>：gm this app is free

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1346077542666211400)** (34 messages🔥): 

> `Grok3 定价, LLM 摘要伦理, Anon-Kode GitHub, 台湾安全` 


- ****Grok 经济学**：定价在线泄露！**：根据[这条推文](https://x.com/swishfever/status/1896539732471058640)，疑似泄露的 **Grok3 定价** 显示，输入成本为 **$3.50/百万 token**，缓存输入为 **$0.875/百万 token**，输出为 **$10.50/百万 token**。
- **LLM 守不住惊喜？！**：一场关于 LLM 为名为 John 的用户总结聊天记录时是否应包含其惊喜生日派对计划的辩论引发了讨论，这提出了关于**隐式上下文理解**和**伦理边界**的问题。
   - 一名成员表示，*这取决于总结的人是 John 本人，还是另一名要求总结并准备交给 John 的用户*。
- **Anon-Kode：Claude 遥测切除术**：[Anon-Kode](https://github.com/dnakov/anon-kode) 是一个 GitHub 项目，它从 **Claude-Code** 中移除遥测数据，并将 Anthropic 端点替换为可自定义的 **OpenAI 端点**。
   - 一些用户对移除 Anthropic 许可的后果表示担忧。
- **台湾局势紧张？**：美国总统在新闻发布会上的言论引发了对美国对**台湾安全**承诺的担忧，与此同时 **TSMC** 宣布在美国投资 **1000 亿美元**（[时间戳](https://youtu.be/Sa7MH1zLEYU?si=2pUU3oLP0FutcGjk&t=1109)）。
   - 同一场新闻发布会还赞扬了 David Sacks 的才智。


<div class="linksMentioned">

<strong>提及链接</strong>：

<ul>
<li>
<a href="https://x.com/TheXeophon/status/1896577148636569622">来自 Xeophon (@TheXeophon) 的推文</a>：LLM 守不住秘密，对吧</li><li><a href="https://x.com/swishfever/status/1896539732471058640">来自 fishy business (@swishfever) 的推文</a>：疑似泄露的 grok3 定价：输入：$3.50/百万；缓存输入：$0.875/百万；输出：$10.50/百万</li><li><a href="https://youtu.be/Sa7MH1zLEYU?si=2pUU3oLP0FutcGjk&t=1109))">突发：特朗普总统宣布 TSMC 在美投资 1000 亿美元，接受记者提问</a>：唐纳德·特朗普总统周一举行新闻发布会，宣布台湾芯片公司在美国投资 1000 亿美元...</li><li><a href="https://github.com/dnakov/anon-kode">GitHub - dnakov/anon-kode</a>：通过在 GitHub 上创建账户来为 dnakov/anon-kode 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1346167374583435359)** (2 条消息): 

> `Anthropic 融资, AI 研发, 国际扩张` 


- **Anthropic 获得巨额资金注入**：Anthropic 以 **615 亿美元** 的投后估值筹集了 **35 亿美元**，由 [Lightspeed Venture Partners](https://www.anthropic.com/news/anthropic-raises-series-e-at-usd61-5b-post-money-valuation) 领投。
   - 此次融资旨在推进其 **AI 系统开发**，加深对其功能的理解，并推动国际增长。
- **Huwupy Kawaii 社交帖子**：一位用户分享了 [bsky.app](https://bsky.app/profile/huwupy.kawaii.social/post/3ljgbrmxv7s2n) 上的帖子链接。
   - 未提供上下文。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/sammcallister/status/1896623802693636501">来自 sam mcallister (@sammcallister) 的推文</a>：引用 Anthropic (@AnthropicAI)：Anthropic 已完成 35 亿美元融资，投后估值为 615 亿美元，由 Lightspeed Venture Partners 领投。这将推进我们 AI 系统的开发，加深对...</li><li><a href="https://bsky.app/profile/huwupy.kawaii.social/post/3ljgbrmxv7s2n">🍅🥔🫐🌽 hoopy frood 🌶️ 🥑🍫🌵 (@huwupy.kawaii.social)</a>：喜欢看到有人说“如果你看 Claude 玩宝可梦，你就会发现一切都结束了”，而我却分不清这是支持还是反对 AI 的观点。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1346146843678998691)** (1 条消息): 

> `Med-RLVR` 


- **Med-RLVR：从 3B 基座涌现的医疗推理**：一篇名为 [Med-RLVR: Emerging Medical Reasoning from a 3B base](https://www.semanticscholar.org/paper/Med-RLVR%3A-Emerging-Medical-Reasoning-from-a-3B-base-Zhang-Liu/6932bed998a4ee3c50fe65bb35c01b551d7aeb90?utm_source=alert_email&utm_content=PaperCitation&utm_campaign=AlertEmails_DAILY&utm_term=PaperCitation&email_index=3-0-8&utm_medium=50538739) 的新论文已发表。
   - 该模型拥有 **3B 参数**。
- **医疗推理**：该论文专注于医疗推理能力。
   - 它探讨了一个相对较小的模型在复杂医疗场景中的表现。


  

---

### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1346011989201387550)** (11 messages🔥): 

> `Post-Training Methodologies for LLMs, In-House Data Labeling for SOTA Models, Human Data vs Synthetic Data, Disentangling Post-training Performance from Data` 


- **LLM 后训练方法论的新综述**：一篇新的综述论文 ([https://arxiv.org/abs/2502.21321](https://arxiv.org/abs/2502.21321)) 探讨了 **Large Language Models (LLMs)** 的 **Post-training methodologies**，分析了它们在 Pretraining 之外优化 LLM 的作用，包括 **Fine-tuning**、**Reinforcement Learning** 以及 **Test-time scaling**。
- **Yeager 选择内部数据标注**：[Enhanced Radar](https://www.enhancedradar.com/) 的 **Yeager**（一个理解空中交通管制音频的 SOTA 模型）由于行业特定的技术复杂性，选择在内部进行数据标注，从而实现了**高度的标准化**和**近乎完美的准确率**。
   - 报酬与转录的字符数挂钩，并设有财务处罚机制。
- **仍需人类数据**：一篇博客文章 ([https://www.amplifypartners.com/blog-posts/annotation-for-ai-doesnt-scale](https://www.amplifypartners.com/blog-posts/annotation-for-ai-doesnt-scale)) 认为，要构建真正有用的 AI 产品，仍然需要**真实的、人类产生的数据**，不同意 Synthetic data 足以推动模型性能实现阶跃式提升的观点。
- **深入探讨 Post-training 性能**：一个 Notion 页面 ([https://mohit-raghavendra.notion.site/Disentangling-Post-training-performance-from-data-1a5db7f2a34480e18010d689a1f46f74](https://mohit-raghavendra.notion.site/Disentangling-Post-training-performance-from-data-1a5db7f2a34480e18010d689a1f46f74)) 讨论了如何将 **Post-training performance** 与数据解耦。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.21321">LLM Post-Training: A Deep Dive into Reasoning Large Language Models</a>: Large Language Models (LLMs) 改变了自然语言处理的格局并催生了多样化的应用。在海量网络规模数据上的 Pretraining 为这些模型奠定了基础...</li><li><a href="https://www.ericbutton.co/p/data-labelling">When Scale AI doesn&#x27;t cut it: doing data labelling in-house</a>: 我们将数据标注转为内部进行，这是我们做过的最正确的决定</li><li><a href="https://www.amplifypartners.com/blog-posts/annotation-for-ai-doesnt-scale">Annotation for AI doesn’t Scale</a>: 随着模型能力越来越强，我们构建的标签和数据标注系统已无法满足需求：什么将取代它们？
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1346174826062151730)** (3 messages): 

> `TSMC $100B investment in U.S. chip factories` 


- **TSMC 考虑在美国进行大规模芯片扩张**：据 [这条推文](https://x.com/anissagardizy8/status/1896615771691974728) 和 [The Information](https://www.theinformation.com/briefings/trump-tsmc-to-announce-100-billion-chip-factory-investment-in-u-s) 报道，**TSMC** 的 CEO 据传正前往白宫，讨论在美国芯片工厂投资 **1000 亿美元** 的潜在计划。
   - 一位成员开玩笑说，*“如果投资达到 1 万亿，我就不会对这笔交易这么悲观了”*。
- **对 TSMC 投资的乐观态度**：一位成员对最初的 **1000 亿美元** 投资表示怀疑，认为这还不够。
   - 该成员表示，如果投资达到 **1 万亿美元**，前景会更加积极。



**提及的链接**: <a href="https://x.com/anissagardizy8/status/1896615771691974728">来自 Anissa Gardizy (@anissagardizy8) 的推文</a>: 新消息：TSMC 的 CEO 今日前往白宫，商讨在美国芯片工厂投资 1000 亿美元的事宜 https://www.theinformation.com/briefings/trump-tsmc-to-announce-100-billion-chip-factor...

  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1346018049823866973)** (44 条消息🔥): 

> `VLM 学士学位项目创意，自动化现实生活工作，寻找有趣的待解决问题，AI 文献综述文章，Discord 服务器邀请链接` 


- **头脑风暴 VLM 项目创意**：一位成员正在为涉及 **VLMs** 的学士学位毕业设计寻找灵感，并对其他适合为期一年且具有发表潜力的**热门话题**建议持开放态度。
- **Claude 完成 GitHub PR**：一位成员报告称使用 **Claude** 和 [Cursor](https://cursor.sh/) 完成了[此 GitHub Pull Request](https://github.com/eslint-stylistic/eslint-stylistic/pull/707) 中 **95%** 的工作。
- **探索深度学习的新型架构**：一位成员提议将 **X-Splines** 与 **Transformers**、**RNNs**、**CNNs**、**GNNs**、**Mamba** 和 **KANs** 等基础架构进行比较。
- **克服知识图谱中的上下文限制**：讨论了通过根据单词的**上下文/概念**分配独立节点来改进**知识图谱**/**超图**的方法，以区分概念和实例。
- **自动化现实生活工作**：一位成员提到他们正在瞒着所有人自动化自己的**现实生活工作 (IRL jobs)**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://ourworldindata.org/ai-timelines">AI 时间线：人工智能专家对未来有何期待？</a>：许多人认为在未来几十年内很有可能开发出人类水平的 AI，有些人则认为它会更早出现。</li><li><a href="https://mail.bycloud.ai/">The AI Timeline</a>：关注最新的前沿 AI 研究</li><li><a href="https://ykilcher.com/discord">加入 Yannic Kilcher Discord 服务器！</a>：查看 Discord 上的 Yannic Kilcher 社区 - 与 20116 名其他成员一起交流，享受免费的语音和文字聊天。</li><li><a href="https://github.com/eslint-stylistic/eslint-stylistic/pull/707">feat: Add granular configuration options to object-property-newline rule by AlbertMarashi · Pull Request #707 · eslint-stylistic/eslint-stylistic</a>：此 PR 通过增加对细粒度配置选项的支持来增强 object-property-newline 规则，允许开发人员为不同的节点类型指定不同的行为。Closes #234Chan...
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1346133547466887260)** (7 条消息): 

> `Joscha Bach, 演示时间段` 


- **演示前的 Bach 演讲头脑风暴**：一位成员最初考虑就 **Joscha Bach** 进行演示，但不清楚这是否是最终主题。
   - 另一位成员提议，如果没有安排其他演示，可以在 `<t:1741046400:F>` 时间段进行演示。
- **演示仍在进行中！**：一位成员为错过演示感到遗憾，但另一位成员澄清说*演示仍在进行中*。
   - 一位参与者向演示者表示感谢，演示者向感兴趣的人提供了进一步的建议。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1346244023714582652)** (3 条消息): 

> `Elsagate 3.0` 


- **Elsagate 3.0：一个可怕的发现**：一位成员分享了一个名为“**Elsagate 3.0 比我们想象的更糟糕**”的 [YouTube 视频](https://www.youtube.com/watch?v=RjOybKOm2Tc)，并警告该视频**不适合儿童观看**。
   - 另一位成员回应道：“嗯，这太可怕了。”
- **附加话题占位符**：这是一个占位符，以确保满足最少话题数量。
   - 如果对话提供了更多实质内容，可以添加更多细节。



**提到的链接**：<a href="https://www.youtube.com/watch?v=RjOybKOm2Tc">Elsagate 3.0 Is Worse Than we Thought.</a>：本视频不适合儿童。建议观众酌情观看。从 Five 获取免费样品包，只需支付运费（必须年满 21 岁）：https://bit.ly/FreeFiveRaymund...

  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1345972347835908178)** (16 条消息🔥): 

> `NotebookLM 中的 Financial statement analysis、Podcast 长度、Notebook 合并、Blog Outline、Podcast 定制` 


- **NotebookLM 中的 Financial Statement Analysis**：一位成员询问如何将 **financial statements**（财务报表）加载到 **NotebookLM** 中进行分析。
- **Podcast 长度问题**：一位成员表示大多数 Podcast 长度为 **20 到 30 分钟**，且未能涵盖某些重要话题，并引用了一个 **Supreme Court Application**。
   - 另一位成员问道：*“能有那么长吗？”*，并附上了一个链接：[US_Dept._of_State_v._AIDS_Vaccine_Advocacy_Coalition](https://cdn.discordapp.com/attachments/1124403655819415592/1346024489846046760/US_Dept._of_State_v._AIDS_Vaccine_Advocacy_Coalition__Supreme_Court_Application._AIDS_Vaccine_Advocacy_Coalition__Supreme_Court_Application.mp3?ex=67c7573e&is=67c605be&hm=ac8945e981f8673fb037a56d5b57d3dab559a89bba554fb5db6fb00dad7c9c6e&)。
- **Notebook 合并已反馈**：一位成员询问是否可以在 NotebookLM 中合并 Notebook。
   - 一名版主确认该功能目前尚不可用，但已提交给 **nblm team** 团队进行考虑。
- **Blog Outline：行还是不行？**：一位成员询问是否可以使用 **NotebookLM** 编写 **blog outline**（博客大纲）。
   - 另一位成员简单地回答道：*“可以”*。
- **Podcast 嘉宾定制命令**：一位成员询问在 **NotebookLM** 中为 **Podcasts** 使用自定义命令的情况。
   - 另一位成员使用了一个类似于 *'the hosts interview lawyers representing both sides of the case'*（主持人采访代表案件双方的律师）的命令，并指出在总结关于嘉宾的 YouTube podcast 剧集时存在准确性问题。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1345971146373468191)** (33 条消息🔥): 

> `动态更新源、Google Docs 集成、Podcast 时间轴、复制粘贴索引编号、批量删除源` 


- **用户讨论动态源链接和 Google Docs**：成员们好奇 **NotebookLM** 是否可以从 **Google Docs** 等源进行动态更新（例如用于追踪家具尺寸等用例），并对手动更新表示担忧。
   - 答案是否定的，它不是自动的，这引发了关于变通方案和功能需求的讨论。
- **Podcast 时间轴与创建请求**：一位成员请求在 Podcast 免费版中加入 **timelines**（时间轴）。
   - 另一位用户询问如何创建 **podcast**，另一位成员兴奋地分享了[一个 NotebookLM podcast 示例](https://notebooklm.google.com/notebook/d2d16919-2f3c-484f-a6e6-626f6aca7845/audio)，认为该功能令人印象深刻。
- **保存笔记后引用链接消失**：成员们注意到，将查询结果保存为笔记时，引用链接会消失。
   - 一位成员澄清说，保存的笔记是 *“view only”*（仅供查看）的，引用链接仅在 Chat 和特定回复中可用。
- **NotebookLM 缺少移动 App 和图标定制**：用户询问了 **NotebookLM** 的 **mobile app** 以及更改 Notebook 图标的功能。
   - 回复确认目前没有移动 App，也没有办法更改 Notebook 的图标。
- **Notebook 共享故障已解决！**：一位用户报告了在与 Gmail 个人账户共享 Notebook 时出现的服务器错误，具体为 *“You are not allowed to access this notebook”*。
   - 该问题已由用户自行解决，原因是接收者的手机是新手机，且未正确配置其 Gmail 账户。



**提到的链接**：<a href="https://notebooklm.google.com/notebook/d2d16919-2f3c-484f-a6e6-626f6aca7845/audio">未找到标题</a>：未找到描述

  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1345988326443716791)** (40 messages🔥): 

> `IP Adapter, Reactor Faceswap, ControlNet, Reforge AMDGPU support, Zluda` 


- **IP-Adapter 面部复制替代方案**：一位成员正在寻找用于面部复制的最佳 **IP-Adapter**，但发现 **ControlNet** 中的 **reference only** 模式已经足够。
   - 另一位成员建议将 **Reactor Faceswap** 作为更优的替代方案，并赞扬了功能强大的 **ControlNet**。
- **Reforge 的 AMDGPU 支持仍不明朗**：一位成员询问 **Reforge** 是否支持 **AMDGPU**，并指出虽然 **Stability Matrix** 中提到了它，但 **GitHub** 页面上却没有。
   - 另一位成员尝试使用 **Zluda**，但在启动时遇到了电脑死机，建议不要依赖被认为存在 Bug 的 **Stability Matrix**，并推荐使用 **Matrix** 之外的 UI。
- **DirectML 与 Reforge 的兼容性困境**：在 **Zluda** 失败后，一位成员测试了带有 **DirectML** 的 **Reforge**，但未能成功运行。
   - 讨论中提到了由 **Lshqtiger** 开发的 **Reforge for AMD** 可能的分支。
- **CivitAI 作为生成请求平台**：一位成员询问了图像生成请求，并讨论了 **CivitAI** 网站，指出该网站提供少量初始积分和每日 25 个可累积的免费积分。
   - 使用成本取决于所选的模型。
- **本地生成图像的要求**：一位成员询问如何创建图像，另一位成员指出建议使用显存（**VRAM**）在 **6-8GB** 左右的 **GPU**，并参考 <#1002602742667280404> 中的资源。
   - 另一位成员提供了 **CivitAI** 的链接用于在线生成。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1346158295085351003)** (13 messages🔥): 

> `Finding good problems to solve, EleutherAI affiliation projects, RWKV models, 4D gaussian splatting` 


- **成员寻求关于寻找待解决好问题的指导**：一位成员询问了关于**寻找待解决的好问题**的建议，以及决定研究想法和问题的具体方法论。
   - 另一位成员建议阅读感兴趣的主题，搜索相关文献，并通过提出“如果...会怎样”的问题来确定一个合理的切入点。
- **EleutherAI 项目：参与度和重点**：一位成员澄清说，服务器中的大多数人并不是在 **EleutherAI** 隶属关系下开展项目的。
   - 他指出，**Interpretability**（可解释性）频道非常活跃，同时在 **NLP** 方面对 **RWKV models** 和评估也很感兴趣，此外 GPT-NeoX 团队正在推出一个新的训练库。
- **以 4D Gaussian Splatting 为例深入研究**：一位成员以改进 **4D Gaussian Splatting**（3D + 时间）为例描述了他们的研究过程。
   - 他们建议从已有的工作开始，进行复现，然后朝着你的想法迈出实验性的一步，以深入理解问题领域并为下一次深入研究提供参考。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1346052245573079100)** (15 messages🔥): 

> `Reasoning Model, GRPO based Agent, LLAMA 3.2 3B, Recurrent LLM reasoning, Atom of Thoughts (AoT)` 


- **ReasonableLLAMA-Jr-3b 模型需要您的反馈！**：一位成员正在为其 [ReasonableLLAMA-Jr-3b](https://ollama.com/adeelahmad/ReasonableLLAMA-Jr-3b) 模型寻求反馈。这是一个推理模型，使用自定义的基于 **GRPO** 的 **Agent** 在 Gym Env 中通过 MLX，对量化版的 **LLAMA 3.2 3B** 进行训练/微调。
   - 该模型基于 **Atom of Thoughts (AoT)** 论文中的概念，即推理过程中的每个状态转换都是一个自包含的、原子式的问题。
- **循环 LLM 推理：成本太高？**：成员们讨论了最近关于循环 **LLM** 推理的论文，这些论文需要显著更多的计算量（相当于 **32B** 参数模型）才能达到与 **7B** 参数模型相当的性能。
   - 这引发了一个问题：*为什么不直接训练一个 32B 参数的模型*，并使用 early exit（提前退出）、mixture of depths（深度混合）或 speculative decoding（推测解码）来实现更便宜的推理？
- **截断反向传播（Truncated Backpropagation）：依然是显存杀手？**：虽然循环模型使用截断反向传播，但截断深度（例如 **8**）可能仍然对应于一个大规模模型（例如 **15B**）的激活值。
   - 一位成员想知道 **DEQ** 类型的训练是否有效，并质疑 *r* 和 *k* 参数是否经过了优化。



**提到的链接**：<a href="https://arxiv.org/abs/2502.12018">Atom of Thoughts for Markov LLM Test-Time Scaling</a>：大型语言模型（LLMs）通过训练时缩放实现了卓越的性能，而测试时缩放通过在推理过程中进行有效的推理进一步增强了它们的能力。H...

  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1346320042932965408)** (10 条消息🔥): 

> `lm-evaluation-harness 中的 trust_remote_code，dataset_kwargs 覆盖，数据集加载错误，data_dir 规范` 


- **Trust Remote Code 条件设置**：一位用户询问 `trust_remote_code` 是否在 `lm-evaluation-harness` 的数据集加载中被始终设置，并引用了 [GitHub 仓库中的特定行](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/__main__.py#L376)。
   - 一位成员澄清说，只有在传递了 `--trust_remote_code` 参数时才会设置 `trust_remote_code`，并指向了 [代码的相关部分](https://github.com/EleutherAI/lm-evaluation-harness/blob/14b0bd26956609b2ee50987299dfa34223fa23b8/lm_eval/__main__.py#L367)。
- **Dataset Kwargs 传递路径揭示**：一位用户询问在加载本地数据集时，设置 `trust_remote_code` 是否会覆盖 `dataset_kwargs`。
   - 一位成员解释说，`dataset_kwargs` 会在 harness 内部传递给 `datasets.load_dataset(...)`，并链接到了 [代码的相关部分](https://github.com/EleutherAI/lm-evaluation-harness/blob/14b0bd26956609b2ee50987299dfa34223fa23b8/lm_eval/api/task.py#L930)。
- **出现数据集生成错误**：一位用户报告在运行具有特定任务配置的 `lm_eval` 时遇到了 **数据集生成错误 (dataset generation error)**。
   - 该用户的任务配置指定了 `dataset_path: json` 以及一个包含 `train.jsonl`、`validation.jsonl` 和 `test.jsonl` 的 `data_dir`。
- **建议手动加载数据集进行调试**：针对报告的错误，一位成员建议手动使用 `load_dataset` 测试数据集是否能正确加载。
   - 该成员还建议尝试为数据目录使用绝对路径，以排除与路径相关的问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/__main__.py#L376">lm-evaluation-harness/lm_eval/__main__.py at main · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型少样本评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/14b0bd26956609b2ee50987299dfa34223fa23b8/lm_eval/api/task.py#L930)">lm-evaluation-harness/lm_eval/api/task.py at 14b0bd26956609b2ee50987299dfa34223fa23b8 · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型少样本评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.co">GitHub · Build and ship software on a single, collaborative platform</a>：加入全球应用最广泛、由 AI 驱动的开发者平台，数百万开发者、企业和最大的开源社区在此构建推动人类进步的软件。</li><li><a href="https://github.com/Eleuth">Eleuth</a>：GitHub 是 Eleuth 构建软件的地方。</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/14b0bd26956609b2ee50987299dfa34223fa23b8/lm_eval/__main__.py#L367">lm-evaluation-harness/lm_eval/__main__.py at 14b0bd26956609b2ee50987299dfa34223fa23b8 · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型少样本评估的框架。 - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1346004493715771434)** (36 messages🔥): 

> `Terraform Registry MCP issues, MCP Multi-Agent Systems, fast-agent GitHub repo, Claude desktop FastMCP errors, MCP server claiming problems` 


- ****MCP Terraform 问题浮现****：一名成员报告了在运行 **terraform-registry-mcp** 和 **aws-mcp server** 时遇到的问题，并在 inspector 之外寻求建议。
   - 他们澄清说，问题发生在 **Claude desktop** 和 **Cline** 中，特别是当启用系统级代理（system-level proxy）时，会导致 *mcp-server-fetch* 出错。
- ****多 Agent MCP 构想成型****：一名成员讨论了为 **多 Agent 系统（multi-agent systems）** 实现 **MCP** 的方案，引用了 Anthropic 在 AI Engineering Summit 上的研讨会，并分享了研讨会的一张 [图片](https://cdn.discordapp.com/attachments/1312302100125843476/1346079179698999338/image.png?ex=67c78a2d&is=67c638ad&hm=50e766c247227751ff74caae3c16387ec1e88639ce75a481c44345cee448c6e5)。
   - 他们提到正在构建一个让 Agent 跨设备协作的框架，并考虑采用 MCP，同时提到了 **BabyAGI** 和 **Stanford generative agents** 等例子。
- ****Fast Agent 框架受到关注****：一名成员分享了他们的项目链接 [fast-agent on GitHub](https://github.com/evalstate/fast-agent)，将其描述为一种 *定义、提示和测试支持 MCP 的 Agent 及工作流* 的方式。
   - 他们确认每个 Agent 都可以配置一组独立的 Server，并澄清每个 Agent 都可以被另一个 Agent 作为 Tool 调用，并配置一组暴露 Tool 的 MCP Server。
- ****Claude 的 Node 版本困扰用户****：一名成员报告在 **Claude desktop** 中使用 **fastmcp** 时遇到 *Cannot find package 'timers'* 错误。
   - 解决方案是移除 **Claude** 正在使用的旧版 **Node v14**。
- ****Twitter API 定价给推文构想泼了冷水****：一名成员探索使用 **MCP** 连接 **Twitter 账号** 以获取和生成推文，但意识到 **Twitter API 成本** 带来的挑战。
   - 一位用户建议将浏览器自动化作为个人项目更具成本效益的替代方案，并指向了一个最近的 [浏览器自动化示例](https://discord.com/channels/1312302100125843476/1317197582665126080/1346263683671785552)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/evalstate/fast-agent">GitHub - evalstate/fast-agent: Define, Prompt and Test MCP enabled Agents and Workflows</a>：定义、提示和测试支持 MCP 的 Agent 及工作流 - evalstate/fast-agent</li><li><a href="https://glama.ai/mcp/servers?searchTerm=twitter&sortingOrder=search-relevance%3Adesc">开源 MCP server</a>：企业级安全、隐私，具备 Agent、MCP、提示词模板等功能。
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1346093530002489396)** (2 messages): 

> `MCPHub.nvim, Graphlit MCP Server, Neovim Plugin, Model Context Protocol` 


- **MCPHub.nvim 让 MCP Server 管理更顺滑**：新发布的 [**MCPHub.nvim**](https://github.com/ravitemer/mcphub.nvim) 插件有助于在 **Neovim** 中管理 MCP Server，提供智能 Server 生命周期管理以及与 **CodeCompanion.nvim** 集成进行 AI 聊天等功能。
   - 该插件可以通过一条命令（`:MCPHub`）安装，并通过简单的 setup 函数进行配置。
- **Graphlit MCP Server 发布**：**Graphlit MCP Server** 已上线，为 **Claude Desktop**、**Goose**、**Cline**、**Cursor** 和 **Windsurf** 等 MCP 客户端提供新的内容摄取和检索功能。
   - 该 Server 是 [开源的](https://github.com/graphlit/graphlit-mcp-server)，需要一个免费的 Graphlit 账号和项目来存储知识库。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.graphlit.com/blog/graphlit-mcp-server">Graphlit MCP Server: Integrate with MCP clients such as Goose, Cline and Claude Desktop - Graphlit</a>：Graphlit 是一个为开发人员构建 AI 驱动应用和非结构化数据 Agent 提供的开箱即用、无服务器 RAG 即服务（RAG-as-a-Service）平台。为 LLM 提供包含网页抓取、Google... 的 ETL 功能。</li><li><a href="https://github.com/ravitemer/mcphub.nvim">GitHub - ravitemer/mcphub.nvim: A powerful Neovim plugin for managing MCP (Model Context Protocol) servers</a>：一个用于管理 MCP (Model Context Protocol) Server 的强大 Neovim 插件 - ravitemer/mcphub.nvim
</li>
</ul>

</div>
  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1346143153060839444)** (30 条消息🔥): 

> `Ash Framework, instructor_ex, DSPy 中的 Async 支持, LangProBe 基准测试, Minions 特性基准测试` 


- **Ash Framework 生态系统探索**：一位成员建议在项目中使用 **Ash Framework**，并链接到了 [ash-project/ash_ai](https://github.com/ash-project/ash_ai) GitHub 仓库，但澄清它是更大的 **Ash Framework** 生态系统中的一个子项目。
   - 他们还分享了 [instructor_ex](https://github.com/thmsmlr/instructor_ex) 的链接，强调了 Elixir 中 LLM 的结构化输出，以及 [Ash Discord 社区](https://discord.gg/w3AXeARR2p)，在那里会有直播主提供指导。
- **DSPy 的 Async 支持计划**：一位成员询问了 DSPy 实现完整 Async 支持的动力，质疑其潜在的性能提升和生产环境中的考量，并链接到了[另一个 Discord 邀请链接](https://discord.gg/8SfMdAeu)。
   - 一位核心贡献者宣布，一位负责人将根据需要使 **Async 支持** 变得原生化，并要求社区明确期望和工作流，同时请求将功能需求作为 [GitHub issues](https://github.com/stanfordnlp/dspy/issues) 提交，因为在工作时间内 Discord 可能会被疏忽。
- **LangProBe 基准测试展示程序组合效果**：一位成员分享了一篇新论文 [LangProBe: a Language Programs Benchmark](https://arxiv.org/abs/2402.20315)，评估了 **DSPy 程序组合** 和 **optimizers** 对各种任务的影响，以及对成本/质量权衡的理解。
   - 根据其 [X/Twitter 帖子](https://x.com/LakshyAAAgrawal/status/1896628734553403728)，该论文发现优化程序中的较小模型可以以更低的成本超越较大模型。
- **Minions 基准测试准备进行成本优化**：一位成员指出，刚刚发布的 **LangProBe 论文** 为他们实现的 **minions 特性** 运行基准测试提供了良好的基准，并引用了他们已关闭的 [pull request](https://github.com/stanfordnlp/dspy/pull/7891)。
   - 该成员正在添加由 jmanhype 开发的 **MinionsLM 和 StructuredMinionsLM**，用于智能 LM 路由。他们指出该论文与 **成本优化** 直接相关。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://ash-hq.org/community">Ash Framework 博客</a>：一个用于构建雄心勃勃的 Elixir 应用程序的声明式基础。建模你的领域，推导其余部分。</li><li><a href="https://github.com/ash-project/ash_ai">GitHub - ash-project/ash_ai: 允许你的用户与你的应用聊天 😎</a>：允许你的用户与你的应用聊天 😎。通过在 GitHub 上创建账号为 ash-project/ash_ai 的开发做出贡献。</li><li><a href="https://github.com/thmsmlr/instructor_ex?tab=readme-ov-file">GitHub - thmsmlr/instructor_ex: Elixir 中 LLM 的结构化输出</a>：Elixir 中 LLM 的结构化输出。通过在 GitHub 上创建账号为 thmsmlr/instructor_ex 的开发做出贡献。</li><li><a href="https://x.com/LakshyAAAgrawal/status/1896628734553403728">Lakshya A Agrawal (@LakshyAAAgrawal) 的推文</a>：🧵介绍 LangProBe：第一个测试将 LLM 组合成语言程序如何影响成本-质量权衡的基准测试！我们发现，在各种任务的平均水平上，其中的较小模型...</li><li><a href="https://github.com/stanfordnlp/dspy/pull/7891">由 jmanhype 添加 MinionsLM 和 StructuredMinionsLM 以实现智能 LM 路由 · Pull Request #7891 · stanfordnlp/dspy</a>：MinionsLM 和 StructuredMinionsLM 实现。此 PR 向 DSPy 框架引入了两个新类：MinionsLM：一个实现了 MinionS 协议以实现高成本效益的智能 LM 路由器...
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1346170606441201727)** (2 条消息): 

> `基于工作流的旅行规划器, LlamaParse 更新, AnthropicAI Claude Sonnet 3.7, Google Gemini 2.0 Flash` 


- **旅行规划器：RS Rohan 带你出发！**：RS Rohan 展示了如何在 @llama_index 中构建一个高级的 **Agentic 旅行规划器**，它可以从用户查询中提取旅行信息，并将任务委派给专门的 Agent（[教程和仓库](https://t.co/XpsEpPr7n2)）。
- **LlamaParse 获得升级**：'Parse With Agent' 模式现在支持 **AnthropicAI Claude Sonnet 3.7** 和 **Google Gemini 2.0 Flash**，提升了表格解析能力和跨页一致性（[公告](https://t.co/V6pwuxm9IO)）。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1346010868864778322)** (19 messages🔥): 

> `AgentWorkflow context vs chat history, MCP Support, PII redaction with LLMs, Anthropic DeltaStream` 


- **AgentWorkflow：上下文与聊天历史的区别已澄清**：一位成员询问了 **AgentWorkflow** 中 **Context**（上下文）与 **Chat History**（聊天历史）的区别，以及何时使用它们。
   - 另一位成员澄清说，*聊天历史包含在上下文（context）中*。
- **LlamaIndex 中的 MCP 支持已确认**：一位成员询问了 LlamaIndex 对 **MCP** 的支持情况，另一位成员确认了其存在。
   - 他们提供了一个[示例 Notebook 链接](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-mcp/examples/mcp.ipynb)，展示了其用法。
- **寻求适用于 LLM 的 PII 脱敏工具**：一位成员正在寻找付费或开源工具，以协助将包含**个人身份信息 (PII)** 的 **PDF** 和**图像**发送给大语言模型 (LLM)。
- **Anthropic 的 DeltaStream 导致 ValueError**：一位成员报告称，**Anthropic** 现在有一种 LlamaIndex 尚不支持的新 **DeltaStream**。具体来说，在启用 thinking（思考）功能时，它会流式传输类型为 `ThinkingDelta` 的增量，而它不是 `TextDelta` 的实例，这导致库抛出 `ValueError`。
   - 该库的维护者承认了这一问题，并表示他们仍需为其添加更好的支持。



**提到的链接**：<a href="https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-mcp/examples/mcp.ipynb">llama_index/llama-index-integrations/tools/llama-index-tools-mcp/examples/mcp.ipynb at main · run-llama/llama_index</a>：LlamaIndex 是构建基于数据的 LLM Agent 的领先框架。- run-llama/llama_index

  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1346342017273172028)** (1 messages): 

> `Windsurf Checkpoints` 


- **Windsurf 缺少 Checkpoint 功能**：一位成员询问 Windsurf 中为何缺少 **checkpoint**（检查点）功能，并指出尽管进行了多次编码尝试以及文件/工作区操作，仍无法回退到之前的状态。
   - 该成员附上了一张[图片](https://cdn.discordapp.com/attachments/1100478495295017063/1346342017042747472/IMG_0299.png?ex=67c7d636&is=67c684b6&hm=0f6ba7693b819f1c004632629bd657658b5eee665327600e857dbdcc903e76bb)，展示了他们尝试将文件拖放到标签菜单中的操作，希望能找到访问之前检查点的方法。
- **需要另一个主题**：由于 topicSummaries 要求至少 2 个项目，此处为占位摘要。

  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1346154286651805707)** (13 messages🔥): 

> `AI replacing programmers, Senior Engineers vs Junior Engineers, Anthropic Fundraising, Stagehand and Browserbase, Claude Code vs Cursor` 


- **关于 AI 替代程序员的报道被严重夸大了**：一篇 [O'Reilly 文章](https://www.oreilly.com/radar/the-end-of-programming-as-we-know-it/) 指出，**AI 工具**将改变编程，但这并不新鲜，因为自第一批程序员连接物理电路以来，编程一直在演变。
   - 一位成员评论道：*“他们只是把旧文章里的词‘查找并替换’成了 AI”*，这就像谴责人们从 StackOverflow 复制粘贴一样；并进一步表示 **LLMs** 让学习新事物的速度变得快得多。
- **高级工程师能更好地驾驭 AI**：高级工程师利用工程经验来塑造和约束 **AI 的输出**，在使用 **Cursor** 或 **Copilot** 等工具时，防止其创建难以维护的*“纸牌屋代码 (house of cards code)”*。
   - AI 加速了实现过程，但工程师的专业知识才是保持代码可维护性的关键，而这正是初级工程师往往缺乏的技能。
- **Anthropic 估值达到 615 亿美元**：[Anthropic](https://www.anthropic.com/news/anthropic-raises-series-e-at-usd61-5b-post-money-valuation) 在 E 轮融资中以 **615 亿美元**的投后估值筹集了 **35 亿美元**，由 **Lightspeed Venture Partners** 领投，旨在推进 AI 系统开发、深化对其工作原理的理解并推动国际扩张。
- **成员称赞 Stagehand，寻求 Python 替代方案**：在听完 Latent Space 播客关于 **Browserbase** 的节目后，一位成员在寻找类似 **Stagehand** 的 Python 版自愈式浏览器工作流工具。
   - 另一位成员指向了 [stagehand-py](https://pypi.org/project/stagehand-py/) 并表示 *“它还在开发中 (WIP)”*。
- **Claude Code vs Cursor，终极对决**：成员们讨论了 [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview) 及其与 **Cursor** 的对比，认为 **Cursor** 更胜一筹，因为它拥有更优越的回滚流程。
   - 一位成员指出 **Claude Code** 较难保持专注，倾向于添加多余的代码行，成本更高，而且 *“Cursor 中的代码编辑速度要快得多”*。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/eugeneyan/status/1896734536693960842">来自 Eugene Yan (@eugeneyan) 的推文</a>: 对于简单的功能和应用，你觉得 Claude Code 更有效，还是坚持使用 Cursor/Windsurf？https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview████什么是...</li><li><a href="https://x.com/e">来自 undefined 的推文</a>: 未找到描述</li><li><a href="https://x.com/AnthropicAI/status/1896606683876753470">来自 Anthropic (@AnthropicAI) 的推文</a>: Anthropic 已在 Lightspeed Venture Partners 领投的融资中以 615 亿美元的投后估值筹集了 35 亿美元。这将推进我们对 AI 系统的开发，深化我们对其工作原理的理解...</li><li><a href="https://www.oreilly.com/radar/the-end-of-programming-as-we-know-it/">我们所熟知的编程的终结</a>: 未找到描述</li><li><a href="https://pypi.org/project/stagehand-py/">stagehand-py</a>: Stagehand 的 Python SDK
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1346042756442886275)** (10 messages🔥): 

> `tinygrad formalist project, Ops.CAT speed bounty, RDNA2/RX6000 usable with tinygrad, Intel Arc A770 usable with tinygrad` 


- **Tinygrad：旨在建立公平算力市场的形式主义项目**：George Hotz ([@__tinygrad__](https://x.com/__tinygrad__/status/1896527413586366710)) 将 tinygrad 描述为一个*形式主义项目 (formalist project)*，试图在非渗漏抽象中捕捉 **Software 2.0** 的全貌，目标是建立一个类似于 Linux 和 LLVM 的公平算力市场。
   - 他提到，到今年年底，**tinygrad** 在 NVIDIA 上的速度应该与现有的 torch CUDA 后端相当，但无需使用 CUDA，并计划建立一个测试云，用户可以在其中通过 lambda 函数租用 FLOPS。
- **Ops.CAT 速度悬赏仍在进行中**：一名成员正在研究 **Ops.CAT 速度悬赏**，但在加入 schedule 后，将其重写为 LLVM 时仍面临问题。
   - 目前的 **Ops.CAT 操作** 涉及 PAD、RESHAPE 和 BUFFER 操作的复杂结构，其中 arg 是要进行 concat 的两个 Tensor。
- **RDNA2/RX6000 可用性受质疑**：一名成员询问 **RDNA2/RX6000/GFX1030** 是否可用于 tinygrad，并报告在运行 `AMD=1` 时出现 `OSError: [Errno 22] Invalid argument`。
   - 另一名成员回应称其在 Linux 上应该可以工作，并要求提供该 OS 错误的 trace，随后该错误已在 [trace.txt 文件中提供](https://cdn.discordapp.com/attachments/1068976834928193609/1346224035591360663/trace.txt?ex=67c76855&is=67c616d5&hm=2bd9866ee6a24e241f5b0b7264ad9d547acf0f7a912cb0c2049053c69e79d44b&)。
- **Intel Arc A770：支持 OpenCL**：针对提问，一名成员确认 **Intel Arc A770** 可用于 tinygrad。
   - 推荐的方法是通过设置 `GPU=1` 来使用 **OpenCL 后端**。



**Link mentioned**: <a href="https://x.com/__tinygrad__/status/1896527413586366710">Tweet from the tiny corp (@__tinygrad__)</a>: What is tinygrad?tinygrad is a formalist project. It attempts to capture the full gamut of software 2.0 in a non leaky abstraction. The methods on Tensor class create a directed graph of immutable RIS...

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1346188980189925416)** (1 messages): 

> `Charles Sutton, Coding Agents, AI for Vulnerability Detection` 


- **Charles Sutton 谈编程 Agent**：特邀讲师 **Charles Sutton** 在 [Lecture 5](https://www.youtube.com/live/JCk6qJtaCSU) 中发表了关于 *编程 Agent 与用于漏洞检测的 AI (Coding Agents and AI for Vulnerability Detection)* 的演讲。
   - 讲座探讨了使用 **LLM agents** 执行计算机安全任务（如寻找软件漏洞），并讨论了 **LLM agents** 的设计问题。
- **DeepMind 研究员在软件工程领域的荣誉**：**Google DeepMind** 的研究科学家 **Charles Sutton** 的研究方向是受代码生成、软件工程、编程语言和计算机安全应用启发的机器学习。
   - Sutton 在软件工程方面的工作曾获得两次 **ACM Distinguished Paper Awards** (FSE 2014, ICSE 2020) 和一次 **10-year Most Influential Paper award** (MSR 2023)。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/live/JCk6qJtaCSU">CS 194/294-280 (Advanced LLM Agents) - Lecture 5, Charles Sutton</a>: Questions: bli.do/csut-code5</li><li><a href="http://bit.ly/1W9UhqT)">Academic ranks in the US and UK</a>: The US and the UK both have a series of ranks for academics, but the names of the job titles are somewhat different.
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1346169821154508930)** (4 messages): 

> `Discord Admin Spam Account Removal, Quiz Posting Schedule` 


- **Discord 管理员要求移除账号**：一名管理员要求移除一个发布垃圾链接的账号，并建议在安全漏洞解决后再重新添加。
- **测验发布时间表公布**：一位用户询问每周何时发布测验，另一位用户回答说他们*通常尝试在周三/周四发布*。

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1346289499432292362)** (4 messages): 

> `Audio issues during lectures` 


- **音频问题困扰讲座**：一名成员反映由于音频问题无法听到讲座中的提问，并请求现场人员协助。
   - 随后，该成员指出音频完全中断，询问是否是现场的 AV 设备设置出现了问题。
- **发生问题后演讲者将重复提问**：一名工作人员对讲座期间的音频问题表示歉意。
   - 他们承诺会提醒演讲者在今后的环节中重复所有提问。


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1346196167150800967)** (9 messages🔥): 

> `Embed Images, 504 Errors` 


- **图像嵌入问题已解决**：一位用户报告了嵌入图像的问题，但随后确认该问题已得到解决。
   - 另一名成员确认了该解决方案。
- **504 错误调查中**：一名成员提到他们没有观察到错误峰值，但注意到**请求速度极慢**，通常导致 **504 错误**。
   - 该成员计划进一步调查，并感谢用户提供的信息。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1346170704634052629)** (5 messages): 

> `Renaming `owned` to `own`, Community meeting, AWS GenAI Loft event` 


- **得益于 Pull Request，`owned` 变更为 `own`**：一名成员创建了一个 [pull request](https://github.com/modular/max/issues/4048)，将 `owned` 重命名为 `own`，以保持与 rest argument 约定的一致性。
- **社区会议临近 - 征集演讲者**：下一次社区会议将在一周后举行，组织者正在寻找演讲者在会议期间进行演讲或分享项目。
   - 如果你有兴趣，请联系组织者表达意向并预留议程席位。
- **在 AWS GenAI Loft 举行的 MAX Engine 活动**：如果你在湾区，可以考虑参加明天晚上在 AWS GenAI Loft 举行的活动，主题为 [Beyond CUDA: Accelerating GenAI Workloads with Modular’s MAX Engine, Hosted by AWS](https://lu.ma/2kkbh2iv)。



**提及的链接**：<a href="https://github.com/modular/max/issues/4048)">modular/max</a>：MAX 平台（包含 Mojo）。通过在 GitHub 上创建账户为 modular/max 的开发做出贡献。

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1346268478901977088)** (1 messages): 

> `SIMD DType, Construction Checks, Globals vs Parameters` 


- **SIMD DType 深度解析**：一位用户质疑在 C bindings 中将 DType 包装在 `SIMD` 中的必要性，因为这会掩盖原始 dtype，但另一名成员澄清说 `SIMD[DType.uint8, 1](0).type` 会在编译时返回 dtype。
   - 他们以 `var a = UInt8(0); alias dtype = __typeof(a).type` 为例进一步说明了该用例。
- **SIMD 构造检查受到关注**：一名成员指出 `SIMD` 在其实现中包含了 [构造检查 (construction checks)](https://github.com/modular/max/blob/main/mojo/stdlib/src/builtin/simd.mojo#L168)。
   - 这确保了 `SIMD` 对象在创建时的有效性和类型安全。
- **参数注入受到推崇**：当被问及使用全局变量时，其中一名成员表示，注入参数（injecting parameters）总是优于使用全局变量。
   - 该成员声称，*如果你有时间的话*，这确实是更好的做法。



**提及的链接**：<a href="https://github.com/modular/max/blob/main/mojo/stdlib/src/builtin/simd.mojo#L168)">max/mojo/stdlib/src/builtin/simd.mojo at main · modular/max</a>：MAX 平台（包含 Mojo）。通过在 GitHub 上创建账户为 modular/max 的开发做出贡献。

  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1346056677107044415)** (1 messages): 

> `Step-based checkpointing` 


- **基于步数的 Checkpointing 减少计算浪费**：一名成员对**基于步数的 Checkpointing (step-based checkpointing)** 表示感兴趣，以减少训练期间发生故障时的计算浪费。
   - 另一名成员回应称，为了解决这一问题，**基于步数的 Checkpointing** 已经在实现中。
- **正在进行的实现解决了相关担忧**：正在进行的**基于步数的 Checkpointing** 实现直接解决了最初对计算浪费的担忧。
   - 该功能旨在通过定期保存进度，最大限度地减少训练运行期间故障造成的影响。


  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1346053201719328828)** (3 messages): 

> `Profiler traces, Tensorboard, PyTorch memory visualizer tool, Perfetto` 


- **Torch 用户使用 Tensorboard 等工具进行追踪**：用户讨论了可视化 Profiler traces 的策略，提到了最初尝试使用 **Tensorboard**，但其似乎移除了某些针对 **PyTorch** 的插件功能。
   - 他们推荐使用 **PyTorch memory visualizer tool** 和 **Perfetto** 分别进行内存和时序追踪，认为这些工具足以满足需求。
- **替代分析工具**：讨论强调了 **PyTorch memory visualizer tool** 和 **Perfetto** 作为内存和时序追踪的替代方案。
   - 在用户报告 **Tensorboard** 存在问题（似乎移除了 **PyTorch** 的一些插件功能）后，这些工具被推荐使用。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1346306760679358464)** (3 messages): 

> `Ollama vs GPT4All, Catalan Language support for GPT4All, GPT4All v3.10.0 Vulnerability` 


- **Ollama vs GPT4All，哪个 llama 更好？**：一位用户询问为什么人们选择 **Ollama** 或 **Llama.cpp** 而不是 **GPT4All**，并声称 **GPT4All** 因其开箱即用的功能而更具优势。
- **加泰罗尼亚语支持**：一位用户请求在 **GPT4All** 界面中增加 **Catalan** 作为语言选项，理由是社区中存在加泰罗尼亚语使用者。
- **GPT4All v3.10.0 存在安全漏洞**：一位用户报告在 **GPT4All v3.10.0** 中发现了一个潜在漏洞，并寻求关于适当报告程序的指导。


  

---


---


---


{% else %}


> 邮件中已截断完整的频道详情。
> 
> 如果您想查看完整详情，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预先感谢！

{% endif %}