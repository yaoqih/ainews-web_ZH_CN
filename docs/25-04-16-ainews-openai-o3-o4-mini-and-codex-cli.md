---
companies:
- openai
date: '2025-04-17T03:17:29.707499Z'
description: '**OpenAI** 推出了 **o3** 和 **o4-mini** 模型，重点强调了**强化学习规模化扩展（reinforcement-learning
  scaling）**和整体效率的提升，使 **o4-mini** 在核心指标上更具性价比且表现更佳。


  这些模型展示了增强的**视觉**和**工具使用**能力，不过这些功能的 API 访问权限尚待开放。此次发布还包括 **Codex CLI**，这是一个开源编程代理，可与这些模型集成，将自然语言转化为可运行的代码。


  **ChatGPT Plus、Pro 和 Team 用户**现已可以使用这些模型，其中 **o3** 的价格明显高于 **Gemini 2.5 Pro**。性能基准测试突显了通过推理侧扩展（scaling
  inference）带来的智能提升，并将其与 **Sonnet** 和 **Gemini** 等模型进行了对比。尽管部分评估结果不尽如人意，但此次发布整体上受到了好评。'
id: 0ac84141-a02b-474c-9ddd-2cf9fadab854
models:
- o3
- o4-mini
- gemini-2.5-pro
- claude-3-sonnet
- chatgpt
original_slug: ainews-openai-o3-o4-mini-and-codex-cli
people:
- sama
- aidan_mclau
- markchen90
- gdb
- aidan_clark_
- kevinweil
- swyx
- polynoamial
- scaling01
title: OpenAI o3、o4-mini 和 Codex CLI
topics:
- reinforcement-learning
- performance
- vision
- tool-use
- open-source
- coding-agents
- model-benchmarking
- multimodality
- scaling
- inference
---

<!-- buttondown-editor-mode: plaintext -->**在 RL 上投入 10 倍算力就是你所需的一切。**

> 2025年4月15日至4月16日的 AI 新闻。我们为您检查了 9 个 subreddits、[**449** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord (包含 **211** 个频道和 **9942** 条消息)。预计节省阅读时间（以 200wpm 计算）：**782 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

正如[周一](https://buttondown.com/ainews/archive/ainews-gpt-41-the-new-openai-workhorse/)所暗示的，OpenAI 在一场经典的直播中发布了[命名略显尴尬的](https://x.com/andr3jh/status/1912634895743992278?s=46) o3 和 o4-mini，并附带了一篇 [博客文章](https://openai.com/index/introducing-o3-and-o4-mini/) 和一份 [系统卡片 (system card)](https://openai.com/index/o3-o4-mini-system-card/):

https://www.youtube.com/watch?v=sq8GBPUb3rk

核心信息是 [RL 扩展 (scaling RL) 的改进](https://x.com/millionint/status/1912568397419954642?s=46):


![image.png](https://assets.buttondown.email/images/fae84f04-b2ce-47a9-a8f4-0be9c289bac5.png?w=960&fit=max)


以及 [整体效率](https://x.com/polynoamial/status/1912564068168450396):


![image.png](https://assets.buttondown.email/images/6f6500f5-d508-4cc6-ad37-3d72bd4f1549.png?w=960&fit=max)


使得 o4-mini 在 OpenAI 优先考虑的各项指标上，相比上一代 [更便宜且更强大](https://x.com/scaling01/status/1912560457174425936):


![image.png](https://assets.buttondown.email/images/39291b1d-29e9-419f-bc27-597160ac4f36.png?w=960&fit=max)


具备 [更强的视觉能力](https://x.com/simonw/status/1912640245402935431?s=46) 和 [更出色的工具使用 (tool use)](https://x.com/sama/status/1912564175253172356?s=46) —— 尽管这些功能尚未在 API 中提供。

Dan Shipper 提供了一份不错的 [定性评论](https://x.com/aidan_mclau/status/1912580976456474812?s=46)

![image.png](https://assets.buttondown.email/images/82abd9dc-8cc7-4022-a7b9-fe5938208e63.png?w=960&fit=max)


系统卡片显示的 [评估结果略逊一筹](https://x.com/scaling01/status/1912552754494541839?s=46)，但总体而言，这次发布受到了非常热烈的欢迎。

最后的 "One more thing" 是 Codex CLI，它通过 [完全开源](https://github.com/openai/codex) 胜过了 Claude Code ([我们的报道在此](https://buttondown.com/ainews/archive/ainews-claude-37-sonnet/))：

https://www.youtube.com/watch?v=FUq9qRwrDrI&t=6s


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

**新模型发布与更新 (o3, o4-mini, GPT-4.1, Gemini 2.5 Pro, Seedream 3.0)**

- **OpenAI o3 和 o4-mini 模型**：[@sama](https://twitter.com/sama/status/1912558064739459315) 宣布发布 **o3 和 o4-mini**，强调了它们在工具使用和多模态理解方面的能力。[@OpenAI](https://twitter.com/OpenAI/status/1912560057100955661) 将其描述为 **更智能、更强大**，能够在 ChatGPT 中以 Agent 方式使用并组合每一种工具。[@markchen90](https://twitter.com/markchen90/status/1912609299270103058) 强调了它们通过学习如何端到端使用工具（特别是在多模态领域）而增强的性能，而 [@gdb](https://twitter.com/gdb/status/1912575762483540322) 则对其产生有用新想法的能力表示兴奋。 
- **访问权限与定价**：[@OpenAI](https://twitter.com/OpenAI/status/1912560062004179424) 指出 **ChatGPT Plus、Pro 和 Team 用户将获得 o3、o4-mini 和 o4-mini-high 的访问权限**。[@aidan_clark_](https://twitter.com/_aidan_clark_/status/1912191545203413419) 认为 **名字中带有 "mini" 的模型令人印象深刻**。[@scaling01](https://twitter.com/scaling01/status/1912553316849942626) 表示 **o4-mini 在各方面都更便宜且更好**；然而，[@scaling01](https://twitter.com/scaling01/status/1912579372650819703) 也指出 **o3 的价格比 Gemini 2.5 Pro 贵 4-5 倍**。
- **Codex CLI 集成**：[@sama](https://twitter.com/sama/status/1912558495997784441) 展示了 **Codex CLI**，一个开源的编程 Agent，旨在增强 o3 和 o4-mini 在编程任务中的表现，而 [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1912556874211422572) 将其描述为一种能将自然语言转化为可用代码的工具。[@kevinweil](https://twitter.com/kevinweil/status/1912562012737167393) 和 [@swyx](https://twitter.com/swyx/status/1912558096553242663) 也重点介绍了这个开源编程 Agent。

- **性能与基准测试**：[@polynoamial](https://twitter.com/polynoamial/status/1912564068168450396) 证实了 **扩展推理规模（scaling inference）可以提升智能**，而 [@scaling01](https://twitter.com/scaling01/status/1912568851604119848) 提供了 o3 与 Sonnet 和 Gemini 等其他模型在 GPQA 和 AIME 等基准测试上的详细性能对比。[@scaling01](https://twitter.com/scaling01/status/1912554822454116736) 还指出，与 o1 相比，o3 在复现研究论文方面的表现较差。[@alexandr_wang](https://twitter.com/alexandr_wang/status/1912555697193275511) 提到 o3 在 SEAL 排行榜上占据绝对统治地位，[@aidan_clau](https://twitter.com/aidan_mclau/status/1912580976456474812) 则分享了 o3 优势的总结链接。
- **多模态能力**：[@OpenAI](https://twitter.com/OpenAI/status/1912560060284502016) 强调 o3 和 o4-mini 可以将上传的图像直接集成到它们的思维链（chain of thought）中。[@aidan_clau](https://twitter.com/aidan_mclau/status/1912560625005522975) 描述了在罗马的一次体验：**o3 进行了推理、调整了图像大小、搜索了互联网，并推断出了用户的位置和度假状态**；同时 [@kevinweil](https://twitter.com/kevinweil/status/1912554045849411847) 注意到模型在思考时会使用工具，例如搜索、编写代码和处理图像。
- **内部功能**：[@TransluceAI](https://twitter.com/TransluceAI/status/1912552046269771985) 报告了 **o3 模型中存在的虚构和对能力的误导性陈述**，包括声称运行代码或使用其无法访问的工具。[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1912585299995361738) 指出 **o3 破解了表情符号中的一个谜团，其思维过程中出现了 FUCK YOU 字样**。
- **GPT-4.1 系列**：[@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1912241877199581572) 宣布了 **面向开发者的 GPT-4.1 系列**，[@skirano](https://twitter.com/skirano/status/1912156805901205986) 指出该系列似乎正朝着优化现实任务的方向发展。[@Scaling01](https://twitter.com/scaling01/status/1912117156751229268) 强调了 **GPT-4.1-mini 在某些基准测试中表现优于 GPT-4.1**。[@aidan_clark_](https://twitter.com/_aidan_clark_/status/1912191545203413419) 称这些模型 **非常出色**。[@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1912177623360479281) 表示 **GPT-4.1 系列是一个稳健的升级**，在各方面都比 GPT-4o 系列更聪明、更便宜。
- **Gemini 2.5 Pro**：[@omarsar0](https://twitter.com/omarsar0/status/1912141918080737648) 表示，与其他模型相比，**Gemini 2.5 Pro 在长上下文理解方面表现更好**。[@_philschmid](https://twitter.com/_philschmid/status/1912038659345297716) 指出 Gemini 2.5 Pro 的性价比惊人。
- **字节跳动 Seedream 3.0**：[@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1912122278722379903) 宣布推出 **Seedream 3.0**，这是 Artificial Analysis 图像排行榜上新的领先模型。[@scaling01](https://twitter.com/scaling01/status/1912118704818606541) 提到 **字节跳动 Seed/豆包团队“强得离谱”**。[@_akhaliq](https://twitter.com/_akhaliq/status/1912426070835339672) 分享了 Seedream 3.0 技术报告。

**使用 FIRE-1 和 OpenAI 的 CodexCLI 进行 Agent 式网页抓取**

- **FIRE-1**：[@omarsar0](https://twitter.com/omarsar0/status/1912596779784143002) 介绍了 **FIRE-1，一个由 Agent 驱动的网页抓取工具**，强调了其导航复杂网站和处理动态内容的能力。[@omarsar0](https://twitter.com/omarsar0/status/1912598072187662600) 进一步解释了它与 scrape API 的简单集成，从而在网页抓取工作流中实现智能交互。[@omarsar0](https://twitter.com/omarsar0/status/1912599033144619411) 指出了传统网页抓取工具的局限性以及 Agent 式网页抓取工具的前景。
- **CodexCLI**：[@sama](https://twitter.com/sama/status/1912586034568945828) 提供了开源 **Codex CLI** 的链接。[@kevinweil](https://twitter.com/kevinweil/status/1912562012737167393) 分享了新开源的 Codex CLI 链接。[@itsclivetime](https://twitter.com/itsclivetime/status/1912569732693438771) 已经开始要求模型“查找其中的 bug”，它能在运行任何代码之前捕获约 80% 的 bug。

**Agent 实现与工具使用**

- **工具使用**：[@sama](https://twitter.com/sama/status/1912564175253172356) 对新模型协同有效使用工具的能力表示惊讶。[@omarsar0](https://twitter.com/omarsar0/status/1912554367711957437) 指出工具使用使这些模型变得更加实用。[@aidan_clau](https://twitter.com/aidan_mclau/status/1912559163152253143) 表示 o3 最大的特性是工具使用，它可以在 CoT（思维链）中通过谷歌搜索、调试并编写 Python 脚本来进行费米估算（Fermi estimates）。
- **工具使用说明**：[@omarsar0](https://twitter.com/omarsar0/status/1912557908459491385) 演示了推理模型如何使用工具，并引用了 AIME 数学竞赛中的一个例子，模型在最初尝试暴力破解后提出了更聪明的解决方案。
- **Reachy 2**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1912554179140227578) 宣布他们本周开始销售 **Reachy 2**，这是首款开源人形机器人。

**视频生成与多模态 (Veo 2, Kling AI, Liquid)**

- **Google 的 Veo 2**：[@Google](https://twitter.com/Google/status/1912190959820898355) 在 **Gemini Advanced 中推出了 Veo 2** 文本生成视频功能，强调其将文本提示词转化为 8 秒电影感视频的能力。[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1912191340424601835) 表示 Veo 2 能让你的剧本栩栩如生，[@matvelloso](https://twitter.com/matvelloso/status/1912256932980965687) 则表示该功能已在 API 中全面开放。
- **字节跳动的 Liquid**：[@_akhaliq](https://twitter.com/_akhaliq/status/1912229925806895201) 分享了字节跳动的 **Liquid**，这是一个可扩展且统一的多模态生成语言模型。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1912239801463341097) 评论道，**字节跳动在所有多模态范式上都表现出色**。
- **可灵 AI 2.0**：[@Kling_ai](https://twitter.com/Kling_ai/status/1912040247023788459) 宣布了 **可灵 AI (Kling AI) 2.0 阶段**，赋能创作者将有意义的故事变为现实。

**可解释性与引导研究**

- **GoodfireAI 的开源 SAEs**：[@GoodfireAI](https://twitter.com/GoodfireAI/status/1912217312566137335) 宣布发布**首个在 DeepSeek 的 671B 参数推理模型 R1 上训练的开源稀疏自动编码器 (SAEs)**，为理解和引导模型思维提供了新工具。[@GoodfireAI](https://twitter.com/GoodfireAI/status/1912217319537099195) 分享了来自其 SAEs 的早期见解，指出了推理中直觉之外的内部标记以及过度引导（oversteering）的矛盾效应。
- **新数据如何渗透 LLM 知识以及如何稀释它**：[@_akhaliq](https://twitter.com/_akhaliq/status/1911992299191669184) 分享了来自 Google 的这篇论文，探讨了学习一个新事实如何导致模型在无关背景下不恰当地应用该知识，以及如何在保留模型学习新信息能力的同时，将这种影响减轻 50-95%。
- **研究人员探索推理数据蒸馏**：[@omarsar0](https://twitter.com/omarsar0/status/1912149669897187579) 总结了关于将顶级 LLM 的密集推理输出蒸馏到更轻量级模型中的研究，以提升多个基准测试的性能。

**LLM 开发工具与框架**

- **PydanticAI**：[@LiorOnAI](https://twitter.com/LiorOnAI/status/1912265113932824840) 介绍了 **PydanticAI**，这是一个为 GenAI 应用开发带来类似 FastAPI 设计的新框架。
- **LangGraph**：[@LangChainAI](https://twitter.com/LangChainAI/status/1912556464746660251) 宣布他们正在开源 LLManager，这是一个通过 human-in-the-loop 驱动的记忆来自动化审批任务的 LangGraph Agent。[@LangChainAI](https://twitter.com/LangChainAI/status/1912207364448743797) 还指出，阿布扎比政府的 AI 助手 TAMM 3.0 是基于 LangGraph 构建的，现在跨平台提供 940 多项服务，并提供个性化、无缝的交互体验。
- **RunwayML**：[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1912568075792040138) 表示 Runway 将走进全球每一个教室。这是 2030 年的目标。
- **Hugging Face 工具发布**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1912623545827418542) 请人让这个工具能够配合来自 HF 的开源模型在本地运行。[@reach_vb](https://twitter.com/reach_vb/status/1912523838723662064) 宣布 Cohere 已在 Hub 上可用。

**幽默/迷因**

- [@teortaxesTex](https://twitter.com/teortaxesTex/status/1912638820170019091) 调侃说自己睡过了 OpenAI 的 "AGI" 发布。
- [@swyx](https://twitter.com/swyx/status/1912577637358379145) 分享了一个与 o3 和 o4 发布相关的迷因（meme）。
- [@scaling01](https://twitter.com/scaling01/status/1912633356895814019) 发表了多条评论，例如批评 "AGI" 营销，并称其为 "ChatGPT 中过度加工的指令遵循"。
- [@aidan_mclau](https://twitter.com/aidan_mclau/status/1912637553519579220) 戏称这是世界冠军级的 Prompt Engineering。
- [@goodside](https://twitter.com/goodside/status/1912565960235433999) 发布了 "外科医生不再是那个男孩的母亲"，这是在玩一个关于 LLM 的著名迷因梗。
- [@draecomino](https://twitter.com/draecomino/status/1912387558484635889) 表示诺兰的电影是最受喜爱的电影。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

## 1. OpenAI 及第三方模型近期发布

- **[OpenAI 推出 OpenAI o3 和 o4-mini](https://openai.com/index/introducing-o3-and-o4-mini/)** ([Score: 109, Comments: 72](https://www.reddit.com/r/LocalLLaMA/comments/1k0pnvl/openai_introducing_openai_o3_and_o4mini/)): **OpenAI 推出了 o 系列的两款新模型 o3 和 o4-mini，在多模态能力（将图像直接集成到推理中）和 Agent 工具使用（通过 API 实现自主的网页/代码/数据/图像工具链）方面有显著提升。根据官方 [博客文章](https://openai.com/index/introducing-o3-and-o4-mini/)，o3 在代码、数学、科学和视觉感知基准测试中达到了 SOTA 水平，并通过大规模 RL 展示了改进的分析严谨性和多步执行能力。社区最关注的问题是仍然缺乏开源发布，尽管 OpenAI 已经[开源了其终端集成](https://github.com/openai/codex)（通过 Codex），但这与完整的模型权重或研究代码不同。**

  - 一位评论者指出 OpenAI 缺乏开源模型，批评其发布策略专注于专有利益而非社区贡献——这反映了重视模型开发透明度和可复现性的从业者们持久的挫败感。
  - 一个链接强调，虽然 OpenAI 没有公开发布模型，但他们开源了终端集成 (https://github.com/openai/codex)，这可能会引起寻求工具链扩展的开发者的兴趣，尽管这并不是模型本身。
  - 也有人批评 OpenAI 的模型命名规范，认为这种混乱的方案可能是故意的——掩盖了模型之间的区别，并使寻求匹配特定能力或要求的用户在评估或选择时变得复杂。

- **[IBM Granite 3.3 模型](https://huggingface.co/collections/ibm-granite/granite-33-language-models-67f65d0cca24bcbd1d3a08e3)** ([Score: 312, Comments: 135](https://www.reddit.com/r/LocalLLaMA/comments/1k0mesv/ibm_granite_33_models/)): **IBM 在 Apache 2.0 许可证下发布了 Granite 3.3 语言模型家族，包含 2B 和 8B 参数量的基础模型和指令微调模型（[Hugging Face 集合](https://huggingface.co/collections/ibm-granite/granite-33-language-models-67f65d0cca24bcbd1d3a08e3)）。这些模型定位于开放社区采用、文本生成任务和 RAG 工作流；此外还提供了语音模型资源（[语音模型链接](https://huggingface.co/ibm-granite/granite-speech-3.3-8b)）。社区反馈受到鼓励，但评论中没有出现深入的基准测试、实现细节或技术 Bug 讨论。**

  - Granite 3.3 模型在紧凑型语言模型中受到好评，特别是得到了 GPU 资源有限的用户的青睐。它们在低端硬件上的可用性被视为一个显著特征，使得在大型模型不切实际的情况下也能使用。用户表示有兴趣评估这一新迭代带来的改进，特别是在资源受限的环境中。

- **[字节跳动发布 Liquid 模型系列，多模态自回归模型（类似 GPT-4o）](https://i.redd.it/393vjiodz2ve1.jpeg)** ([Score: 285, Comments: 33](https://www.reddit.com/r/LocalLLaMA/comments/1k05wpt/bytedance_releases_liquid_model_family_of/)): **[此处链接](https://i.redd.it/393vjiodz2ve1.jpeg)的图片是字节跳动所谓的 'Liquid' 模型系列的宣传概览，被描述为一个可扩展、统一的多模态自回归 Transformer（类似于 GPT-4o），旨在单一架构内处理文本和图像生成。Reddit 上的讨论对该模型的真实性和技术主张表示怀疑：评论者指出，所谓的发布并非近期之事，Hugging Face 上的公开权重（checkpoint）显然只是一个没有视觉配置的 Gemma 微调版本，且没有可用的真正多模态预训练模型（如描述所述）。此外，尽管宣传材料提到了字节跳动的参与，但没有官方渠道或论文证实这一发布，这表明可能存在归属错误或误导性陈述。**

  - 一位评论者指出，Liquid 模型系列的官方公告与实际发现的模型工件之间存在不一致：`config.json` 缺少视觉配置，这表明公开模型并非演示中展示的多模态版本。模型卡片提到了六种模型尺寸（参数量从 0.5B 到 32B），包括一个基于 GEMMA 的 7B 指令微调变体，但据报道这些版本在预期的仓库中缺失，且文档中完全没有提到字节跳动的参与。
  - 通过官方在线演示对模型进行的测试表明，其定性表现不尽如人意——尤其是在图像生成任务中，例如渲染手部或真实物体（例如，“草地上的女人效果不佳”以及物体畸形）。这与关于输出解剖结构错误特征的投诉一致，而这通常是衡量多模态模型鲁棒性的基准。

- **[有人得告诉 Nvidia 在这些新模型命名上冷静点。](https://i.redd.it/hl0xrywo89ve1.jpeg)** ([Score: 127, Comments: 23](https://www.reddit.com/r/LocalLLaMA/comments/1k0u8ew/somebody_needs_to_tell_nvidia_to_calm_down_with/)): **该帖子幽默地批评了 NVIDIA 日益复杂和冗长的模型命名惯例，例如模拟标签 'ULTRA LONG-8B' 指代具有 '100 万、200 万或 400 万扩展上下文长度' 的模型。图片和评论讽刺了现代模型名称如何像其他产品的品牌推广——这里将其比作避孕套名称——突显了行业向更长、更具营销驱动力的命名惯例发展的趋势。目前没有关于模型本身或其基准测试的实质性技术讨论，仅是对命名法的评论。**

  - 潜藏着对 Nvidia 不一致或令人困惑的模型命名惯例的批评，建议其产品线可以从更清晰的分类法中受益，以避免产品代际和层级之间的歧义。技术读者指出，准确的命名对于区分模型能力至关重要——尤其是在新架构和变体迅速激增的情况下。


## 2. 大规模模型训练与基准测试

- **[INTELLECT-2：首个 32B 参数模型的全球分布式强化学习训练](https://www.primeintellect.ai/blog/intellect-2)** ([Score: 123, Comments: 14](https://www.reddit.com/r/LocalLLaMA/comments/1k04tcz/intellect2_the_first_globally_distributed/)): **[INTELLECT-2](https://www.primeintellect.ai/blog/intellect-2) 通过在全异构、无许可的硬件上训练 32B 参数模型，开创了去中心化强化学习（RL）的先河，并通过以太坊 Base 测试网进行激励和协调，以实现可验证的完整性并惩罚（slashing）不诚实的贡献者。核心技术组件包括：用于异步分布式 RL 的 prime-RL，用于正确推理证明的 TOPLOC，以及用于稳健、低开销模型分发的 Shardcast。该系统支持通过系统提示词配置“思考预算”（thinking budgets）（从而能够根据具体用例精确控制推理深度），并构建在 QwQ 之上，旨在为大型模型建立一种可扩展、开放的分布式 RL 新范式。热门评论澄清说，发布不等于完成训练，强调了可控推理预算的创新，并询问了用于进一步基准测试的人类反馈（HF）时间表。**

- INTELLECT-2 引入了一种机制，用户和开发者可以指定模型的“思考预算（thinking budget）”——即模型在生成解决方案之前用于推理的 tokens 数量，旨在实现可控的计算开销和推理深度。这基于 QwQ 框架，代表了相比具有固定步长推理的标准 Transformer 模型的潜在进步。
- 该项目声称是第一个使用全球分布式强化学习训练的 32B 参数模型。通过与过去社区驱动的分布式训练项目（如受 DeepMind 的 AlphaGo 启发的项目）进行对比，提供了历史背景，但评论者指出，这种规模的硬件需求对个人来说仍然是一个重大障碍。

- **[非推理型 LLM 的价格 vs LiveBench 性能](https://i.redd.it/eiojps9w67ve1.png)** ([得分: 127, 评论: 48](https://www.reddit.com/r/LocalLLaMA/comments/1k0kape/price_vs_livebench_performance_of_nonreasoning/)): **该散点图直观展示了一系列非推理型 LLM 在价格与 LiveBench 性能得分之间的权衡。它对每百万 3:1 混合输入/输出 tokens 的价格使用了对数坐标 X 轴，并使用颜色编码的点显示了一系列私有模型（OpenAI、Google、Anthropic、DeepSeek 等），其中 GPT-4.5 Preview 和 DeepSeek V3 的位置尤为引人注目。评论强调了 Gemma/Gemini 模型在帕累托前沿（Pareto front，即单位价格性能最大化）的统治地位，特别称赞了 Gemma 3 27B 的效率。分析揭示了市场竞争力的明显差异：[查看图片](https://i.redd.it/eiojps9w67ve1.png)。**

  - 多位评论者指出，Gemma（以及可能的 Gemini）模型目前在非推理型 LLM 基准测试的价格/性能“帕累托前沿”中占据主导地位，这表明它们在成本与性能之间提供了相对于竞争对手的最佳权衡。这意味着在当前格局下，其他替代方案在价格和效率的特定指标上落后于这些模型。
  - 讨论强调，Gemma 3 27B 在其规模上提供了强大的基准测试结果，而 Gemini Flash 2.0 模型因其出色的性价比被特别点名，其表现显著优于 Llama 4.1 Nano，后者因价格/性能比差而受到批评。这突显了随着新模型的发布和并排基准测试，LLM 市场的价值主张正在发生变化。

- **[我们通过 GRPO 训练了一个模型，让它不断重试“搜索”直到找到所需内容](https://v.redd.it/x9c46kt8l4ve1)** ([得分: 234, 评论: 36](https://www.reddit.com/r/LocalLLaMA/comments/1k0c40c/we_grpoed_a_model_to_keep_retrying_search_until/)): **Menlo Research 推出了 ReZero，这是一个 Llama-3.2-3B 变体，通过广义重复策略优化 (GRPO) 和自定义的 retry_reward 函数进行训练，能够高频重试“搜索（search）”工具调用，以最大化搜索任务的结果 ([arxiv](https://arxiv.org/abs/2504.11001), [github](https://github.com/menloresearch/ReZero))。与为了减少幻觉而惩罚重复的传统 LLM 微调不同，ReZero 凭经验实现了 `46%` 的得分——是基准线 `20%` 的两倍多——这证明了重复如果与搜索和适当的奖励塑造相结合，可以提高事实严谨性，而不是诱发幻觉。所有核心模块，包括奖励函数和验证器，均已开源（见 [repo](https://github.com/menloresearch/ReZero)），利用 AutoDidact 和 Unsloth 工具集进行高效训练；预训练检查点已在 [HuggingFace](https://huggingface.co/Menlo/ReZero-v0.1-llama-3.2-3b-it-grpo-250404) 发布。**

  - 一位评论者询问 GRPO 方法中使用的奖励函数或验证器的可用性，表示有兴趣检查或重现发布代码库中的强化机制和评估逻辑。
  - 主要的模型训练流水线利用了 [AutoDidact](https://github.com/dCaples/AutoDidact) 和 [Unsloth](https://github.com/unslothai/unsloth) 等开源工具集，表明实现可能依赖这些框架来编排强化学习或优化推理；两者都被认为是技术可复现性的关键。
  - 讨论暗示使用了一种迭代方法，模型反复重试“搜索”查询直到成功，这意味着可能通过上述工具链实现了一个自定义奖励或重试循环——这引发了关于这种反馈驱动的搜索强化方案中效率和资源消耗的问题。


## 3. 社区项目与硬件配置

- **[Droidrun 现已开源](https://i.redd.it/9zbo1emvc6ve1.jpeg)** ([Score: 214, Comments: 20](https://www.reddit.com/r/LocalLLaMA/comments/1k0h641/droidrun_is_now_open_source/)): **该帖子宣布 Droidrun 框架——一个根据标题和 Logo 设计推测与 Android 或自动化相关的工具——现已开源并发布在 GitHub 上（[仓库链接](https://github.com/droidrun/droidrun)）。图片本身并非技术性的：它是一个正在奔跑的 Android 角色风格化 Logo，传达了速度、活跃度以及项目的开源性质。帖子或评论中未提供 Benchmark 或实现细节，尽管早期社区兴趣很高，等候名单已超过 900 人。**

  - 技术讨论强调了 Droidrun 如何实现对 Android 设备的先进自动化控制和脚本编写，使其对关注设备自动化的资深技术用户极具价值。几位评论者对其具体应用场景进行了辩论，指出能够从 GitHub 编译/安装的用户可能不需要 LLM 集成来执行简单操作，这表明该工具的真正优势在于将本地设备控制与自然语言驱动的工作流、脚本编写或 Android 上的批量自动化相结合。

- **[是的，你只需花费约 1000 美元即可拥有 160GB VRAM。](https://www.reddit.com/r/LocalLLaMA/comments/1k0b8wx/yes_you_could_have_160gb_of_vram_for_just_about/)** ([Score: 177, Comments: 78](https://www.reddit.com/r/LocalLLaMA/comments/1k0b8wx/yes_you_could_have_160gb_of_vram_for_just_about/)): **原作者记录了一台花费 1157 美元的深度学习推理机配置，使用了十张 AMD Radeon Instinct MI50 GPU（每张 16GB VRAM，兼容 ROCm），安置在专为挖矿设计的 Octominer XULTRA 12 机箱中，并由 3 个 750W 热插拔 PSU 供电。核心软件为 Ubuntu 24.04 及 ROCm 6.3.0（由于 MI50 的支持限制，尽管有评论者指出根据设备表 ROCm 6.4.0 仍可运行），并从源码编译了 llama.cpp 用于推理。Benchmark（llama.cpp，q8 量化）显示 MI50 提供了约 40-41 tokens/s（eval），但 Prompt 吞吐量较差（例如约 300 tokens/s），表现逊于消费级 Nvidia（RTX 3090, 3080Ti），且在多 GPU 和 RPC 使用下性能下降约 50%——例如，MI50@RPC（5 GPU）运行 70B 模型达到约 5 tokens/s，而 3090（5 GPU）约为 10.6 tokens/s，Prompt 评估也慢得多（约 28 ms/token vs 约 1.9 ms/token）。功耗和散热表现出色（待机约 20W/卡，推理约 230W/卡），巨大的 VRAM 池对于超大模型或 MoE 非常有价值。局限性包括 PCIe x1 带宽瓶颈、llama.cpp 的横向扩展限制（对超过 16 GPU 的支持不稳定）以及显著的 RPC 相关效率损失。建议包括将显卡功耗降至 150W 以换取微小的性能损失、尝试 MoE 模型以及潜在的网络/RPC 代码优化。详见原帖中的详细 Benchmark 和配置说明：[Reddit 线程](https://www.reddit.com/r/LocalLLaMA/comments/1jy5p12/another_budget_build_160gb_of_vram_for_1000_maybe/)。**

  - 多位用户报告称，尽管文档说明有所不同，MI50 仍可支持最新的 ROCm (6.4.0)——安装正常，且 gfx906（MI50 架构）在 Radeon/Radeon Pro 选项卡下被列为受支持，这为依赖 ROCm 进行 ML 工作负载的潜在买家提供了保障。
  - MI50 GPU 的功耗可以被限制以显著降低瓦数（例如，减半至 150W 仅会降低约 20% 的性能），推理速率在低至每张卡 90W 时仍可接受；这对于构建关注功率限制、成本或散热问题的集群（如 10 卡配置）至关重要。
  - 据报告，在 1000 美元的 MI50 配置上，70B Q8 Llama 3.3 模型的生成速度约为 `4.9-5 tokens/sec`，小上下文的首字时间 (TTFT) 为 12 秒，大上下文则长达 2 分钟，为该多 GPU 配置的性能和延迟预期提供了具体的 Benchmark。

- **[你最喜欢的无审查模型是什么？](https://www.reddit.com/r/LocalLLaMA/comments/1k0967d/what_is_your_favorite_uncensored_model/)** ([Score: 103, Comments: 71](https://www.reddit.com/r/LocalLLaMA/comments/1k0967d/what_is_your_favorite_uncensored_model/)): **讨论集中在经过修改以减少内容过滤的大语言模型（LLM）（“无审查”或“消融”模型），特别是由 huihui-ai 提供的模型，包括 Phi 4 Abliterated、Gemma 3 27B Abliterated 和 Qwen 2.5 32B Abliterated。用户注意到 Phi 4 模型在“消融”后仍保持了其性能/智能，而 Gemma 3 27B 的无审查状态较为温和，除非将其作为金融数据的 RAG 使用。Mistral Small 也因其在没有主要安全层的情况下具有极高的开箱即用许可度而受到关注，无论是否经过无审查处理。有关上述模型的技术配置和量化权重，请参见 [huihui-ai 项目仓库](https://huggingface.co/huihui-ai)。**

- 讨论重点介绍了几款特定的无审查模型：Phi 4 Abliterated、Gemma 3 27B Abliterated 以及由 huihui-ai 发布的 Qwen 2.5 32B Abliterated。用户称赞 Phi 4 在 abliteration 处理后智能退化极小，表明其无审查过程背后的方法论非常稳健。
- 据报道，Gemma 3 27B 开箱即用的无审查效果并不理想，但有评论者指出，通过检索增强生成 (RAG) 成功提取了财务建议，尤其是在使用模型变体而非微调版本时。
- 另一位用户指出，无审查过程通常会导致模型性能明显下降，并表示相比于经过重度修改的无审查副本，他们更倾向于将标准模型与越狱提示词 (jailbreak prompts) 结合使用，这反映了社区在平衡去除审查与保持基础能力方面的广泛担忧。


## 其他 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo


## 1. OpenAI o3 和 o4-mini 模型发布与讨论

- **[o3 将在 3 小时内发布](https://i.redd.it/5ohz5uz3e7ve1.png)** ([Score: 753, Comments: 186](https://www.reddit.com/r/singularity/comments/1k0l5tt/o3_releasing_in_3_hours/)): **该图片是来自 OpenAI 的推文，宣布一场将在“o3”小时后开始的直播活动，暗示即将发布或演示名为“o3”的新模型。这在社区中引发了巨大的期待，技术讨论涉及此前的高昂计算成本（一条评论提到一个 prompt 成本约为 3000 美元），并质疑广泛发布的可能性。** 讨论包括对这类高算力模型的部署或商业化的怀疑，一些用户提到之前的扩展和成本问题是公众或大规模访问的技术障碍。

  - 一位用户提到了过去在类似模型上生成单个 prompt 的高昂计算成本，引用了约“每个 prompt 3000 美元”的数据，对 o3 在如此高的计算需求下如何发布提出疑问。这暗示了与之前的迭代相比，模型效率、推理成本或基础设施可能有所改进。
  - 另一位用户热衷于将 o3 的能力与团队自家的“DeepResearch”模型以及 Google 的“Gemini 2.5 Pro”进行对比，明确强调了对跨基准测试性能的兴趣，并希望这预示着更多即将发布的模型（特别是“o4”系列）。

- **[介绍 OpenAI o3 和 o4-mini](https://openai.com/index/introducing-o3-and-o4-mini/)** ([Score: 235, Comments: 91](https://www.reddit.com/r/singularity/comments/1k0piul/introducing_openai_o3_and_o4mini/)): **OpenAI 推出了 o3 和 o4-mini 模型。与早期（12月）的数据相比，o3 模型在 GPQA、SWE-bench 和 AIME 等基准测试上的表现略有下降，但在发布博客中指出其价格比 o1 模型更便宜。核心技术讨论集中在 o3 的编程基准测试表现上，据报道略优于 Google 的 Gemini，但 o3 的成本高出 5 倍。关于当前基准测试相关性的讨论也较为广泛，呼吁建立专注于真实世界 Agent 任务性能的评估指标，而非增量式的数学或推理基准。** 评论者仔细权衡了 o3 基准性能下降与成本降低之间的关系，一些人指出，实际应用价值应优先于微小的基准差异。尽管具有性能优势，但 o3 相对于 Gemini 的高昂成本也引起了关注。[外部链接摘要] OpenAI 推出了 o3 和 o4-mini，这是其 o 系列推理模型的最新成员，在 Agent 工具使用和多模态能力方面有显著提升。o3 在编程、数学、科学和视觉感知方面设定了新的 SOTA 基准，通过在 ChatGPT 中集成网页搜索、文件分析、Python 执行和图像生成工具，擅长解决复杂的、多方面的问题。两款模型都利用大规模强化学习来实现卓越的推理和指令遵循能力，其中 o4-mini 针对高吞吐量、高性价比的推理进行了优化，在小型模型中处于领先地位——尤其是在数学和编程方面。用户首次可以将图像纳入推理工作流，从而实现集成的视觉-文本问题解决和工具链调用，以处理先进的、真实的和实时的任务。详见：[Introducing OpenAI o3 and o4-mini](https://openai.com/index/introducing-o3-and-o4-mini/)

- GPQA、SWE-bench 和 AIME 等 Benchmark 显示，o3 的得分与 12 月最初发布时相比略有下降，尽管 OpenAI 指出该模型现在比 o1 更便宜；有人推测其性能被刻意降低以减少成本。
- 在 Aider polyglot Benchmark 上，o3-high 得分为 81%，但成本可能非常高（推测约为 200 美元，类似于 o1-high），而 Gemini 2.5 Pro 以更低的价格获得了 73% 的得分。GPQA 得分非常接近（`o3: 83%` 对比 `Gemini 2.5 Pro: 84%`）。尽管 o3 在数学方面表现出进步（特别是数学 Benchmark 的大幅提升，以及在不使用工具的情况下在 HLE 上略微领先于 Gemini），但相对于 Gemini 的高成本，使其对于关注实际性价比的用户来说吸引力较低。
- 讨论指出，虽然 Benchmark 很有用，但它们在某种程度上被高估了，并不总是能反映模型对日常任务或基于 Agent 的工作流的适用性。人们呼吁建立能更好反映 LLM 实际工作能力或日常用例效用的新 Benchmark。

- **[随 o3 发布 o4 mini](https://i.redd.it/waedilg728ve1.jpeg)** ([Score: 205, Comments: 43](https://www.reddit.com/r/OpenAI/comments/1k0oe9t/launching_o4_mini_with_o3/)): **该图片宣布了 OpenAI 即将举行的活动，将介绍新的 'o-series' 模型，特别是 'o3' 和 'o4-mini'。这表明 OpenAI 正在继续扩展其 GPT-4o 之外的模型阵容，对性能和功能都有影响。链接的 YouTube 活动暗示了一次官方的技术发布，尽管图片中几乎没有模型细节。** 评论者严厉批评了 OpenAI 混乱且不一致的模型命名惯例，认为名称相近但功能不同的模型（'o3'、'o4-mini'、'4o'）在技术和非技术圈都造成了不必要的困扰。

  - 围绕 'o3'、'o4' 和 'o4 mini' 模型命名重叠存在困惑和技术批评，用户指出，为功能差异巨大的模型提供相似的名称，在引用 Benchmark、更新或部署上下文时会产生歧义。
  - 提出了关于 'o4 mini' 与 'o3' 相比的实际用例的技术问题，特别是质疑为什么在发布旧模型的同时发布一个更新的、可能增强的模型，尤其是当新版本是 'mini'（可能更小或更高效）时，引发了对现实场景或 Benchmark 驱动偏好的讨论。
  - 还存在关于欧盟访问 'o3' 和 'o4 mini' 的区域可用性问题，如果得到解答，将为技术读者提供有关部署时间表、推广策略以及符合当地法规或基础设施现实的信息。

- **[[已确认] O-4 mini 也将与 O-3 全功能版一同发布！](https://i.redd.it/lnj56ieb18ve1.jpeg)** ([Score: 298, Comments: 44](https://www.reddit.com/r/singularity/comments/1k0o909/confirmed_o4_mini_launching_with_o3_full_too/)): **该图片正式宣布了 OpenAI 将于 2025 年 4 月 16 日举行活动，介绍新的 'o-series' 模型——特别是 O-4 mini 和全功能 O-3 模型。它确认了 O-4 的轻量级 'mini' 版本和 O-3 的全功能版本将同时发布。该活动将由 Greg Brockman、Mark Chen 等知名的 OpenAI 工程师和研究员进行演示，暗示了深入的技术揭秘和展示。[图片链接。](https://i.redd.it/lnj56ieb18ve1.jpeg)** 评论者质疑 OpenAI 模型命名方案的清晰度，一些人表示期待从之前的 'o3 mini' 切换到新的 'o4 mini'。命名和模型区分被强调为技术用户中持续存在的困惑点。

  - 初始评论列出了参与介绍和演示新 O-series 模型的重要人物，包括 Greg Brockman 和 Mark Chen，这可能预示着一场高规格的发布活动，对于跟踪未来与 O-4 mini 和 O-3 全功能模型相关的技术演示或公告可能具有参考价值。
  - 几位用户讨论了从主要使用 'O-3 mini high' 向 'O-4 mini' 的过渡，暗示了迭代改进，并且有一个明确的用户群体正在迁移到更新的模型；这表明人们预期 O-4 mini 在实际使用中可能优于 O-3 mini 或提供额外价值。
  - 对 O-series 命名方案有一些轻微的技术批评，用户将其描述为 '荒谬'。虽然这不直接涉及技术，但它对模型跟踪、集成和未来的开发周期有影响，混乱的命名法可能会阻碍采用和 API 版本管理。

- **[这证实了我们今天将同时获得 o3 和 o4-mini，而不仅仅是 o3。个人非常期待能一睹 o4 系列的风采。](https://i.redd.it/4ctni1mdr7ve1.jpeg)** ([评分: 225, 评论: 50](https://www.reddit.com/r/singularity/comments/1k0mwtk/this_confirms_we_are_getting_both_o3_and_o4mini/)): **该帖子使用了一张分两行排列的草莓图片（三颗大的，四颗小的），隐喻式地确认了双重发布：o3 和 o4-mini 两个 foundation models 将一同推出（图片：[链接](https://i.redd.it/4ctni1mdr7ve1.jpeg)）。这个视觉双关语形象地代表了 o3（三颗大的）和 o4-mini（四颗小的）模型，预示着产品线的战略性扩张，可能在性能或尺寸上有所区分。标题和图片语境强调了对预览下一代 o4 系列（而非仅仅是增量更新）的兴奋。** 技术讨论集中在对 o3 定价的担忧（可能每月 200 美元）以及对 o4-mini 是否能兑现其科学辅助声明的怀疑与期待，反映了社区对这些模型的实际影响和可访问性的关注。

  - MassiveWasabi 讨论了对 o4-mini 模型是否能兑现其在推进科学研究方面的效用声明的好奇，这表明用户对模型在通用 AI 任务之外的性能有着技术预期。
  - jkos123 询问了 o3 full 的 API 定价，并将其与之前的层级（如 o1-pro，`$150/month in`, `$600/month out`）进行了直接对比。这表明技术用户在生产和研究的部署选择中非常关注性价比。
  - NootropicDiary 推测了 o4 mini high 的编程和 reasoning 能力，质疑其性能是否可能接近 o3 pro。这指向了社区对对比基准测试以及这些模型在开发工作流中实际应用的兴趣。


## 2. OpenAI o3/o4 vs Gemini 基准测试与对比

- **[o3 和 o4 mini 对比 Gemini 2.5 Pro 的基准测试](https://www.reddit.com/gallery/1k0qjso)** ([评分: 340, 评论: 169](https://www.reddit.com/r/singularity/comments/1k0qjso/benchmark_of_o3_and_o4_mini_against_gemini_25_pro/)): **该帖子在各项任务中对 o3、o4-mini 和 Gemini 2.5 Pro 模型的性能进行了基准测试。在数学基准测试 (AIME 2024/2025) 中，o4-mini 的表现略优于 Gemini 2.5 Pro 和 o3 (o4-mini 在 AIME 2024 上为 `93.4%`，o3 为 `91.6%`，Gemini 2.5 Pro 为 `92%`)。在知识与推理 (GPQA, HLE, MMMU) 方面，Gemini 2.5 Pro 在 GPQA (`84.0%`) 上领先，o3 在 HLE (`20.32%`) 和 MMMU (`82.9%`) 上领先。在编程任务 (SWE, Aider) 中，o3 在 SWE (`69.1%`) 和 Aider (`81.3%`) 上表现最佳。**定价**也是重点，o4-mini 明显比其他模型更便宜 (`$1.1/$4.4`)。图表由 Gemini 2.5 Pro 生成。** 评论者指出图表 y 轴缩放可能存在误导，并强调 Google 和 OpenAI 模型的性能现在非常接近，尽管 Google 的节奏和资源优势被视为他们可能很快超越 OpenAI 的指标。

  - 讨论强调了仅通过每百万 token 的价格来比较 AI 模型 token 成本的局限性，指出不同模型之间的输出长度（reasoning tokens）差异巨大，从而导致成本比较失真。相反，应该分析运行基准测试的实际美元成本以确定真实支出，而非仅看面向消费者的零售价格。文中强调了“Cost”（运行模型的运营、硬件和基础设施成本）与“Price”（公司收取的访问费用）之间的区别，指出专有模型（如 OpenAI、Google）掩盖了运行成本，而开源模型则允许更透明的评估，因为用户可以直接测量或估算硬件开销。该帖子还提醒，公司的定价策略（例如 Google 可能利用 TPU 优势或出于市场份额目标设定人为低价）使公平比较变得更加复杂。建议未来的基准测试采用一种稳健、标准化的分析方案，将运行成本、消费者价格与性能表现综合考量。

- **[Comparison: OpenAI o1, o3-mini, o3, o4-mini and Gemini 2.5 Pro](https://i.redd.it/gh48z5iyl8ve1.png)** ([Score: 195, Comments: 44](https://www.reddit.com/r/OpenAI/comments/1k0r4xw/comparison_openai_o1_o3mini_o3_o4mini_and_gemini/)): **该图片提供了 OpenAI 的 o1, o3-mini, o3, o4-mini 模型与 Google 的 Gemini 2.5 Pro 之间的直接基准测试对比，涵盖了 AIME (数学), Codeforces 编程, GPQA (科学问答) 以及多项推理/逻辑任务。表格显示 **OpenAI 的 o4-mini 在工具辅助数学任务中领先**，而 **Gemini 2.5 Pro 在某些编程和科学基准测试（如 LiveCodeBench v5）中表现出色**。不同任务之间存在显著差异，反映了模型优势如何随领域而变化；OpenAI 的中端模型（如 o3）在用户体验中展示了强大的全应用实际代码生成能力。** 热门评论强调了 o4-mini 在数学方面的统治地位，认为随着模型超越人类水平，基准测试的相关性正在减弱，并指出需要关注定价背景。有用户通过亲身经历赞扬了 o3 在代码生成方面的实际应用价值。

  - Gemini 2.5 Pro 被描述为通常与 OpenAI 的 o4-mini 相当，但在数学方面除外，o4-mini 在该领域领先。（“所以 Gemini 2.5 ~ o4-mini，除了数学方面 o4-mini 领先”）
  - 用户对 o3 的测试体验表明，它可以单次输出生成完整的、可运行的应用程序，这表明与之前的模型相比，代码合成（code synthesis）取得了重大进展。（“o3 相当具有突破性，一次性就能吐出完整的、可运行的应用”）
  - 注意到在 “Humanity's Last Exam” 基准测试上的快速进步：与在此类测试中平均得分约 30%（在其专业领域约为 80%）的博士生相比，当前模型的得分代表了短时间内的显著进步。

- **[If o3 from OpenAI isn't better than Gemini 2.5, would you say Google has secured the lead?](https://www.reddit.com/r/singularity/comments/1k05nre/if_o3_from_openai_isnt_better_than_gemini_25/)** ([Score: 211, Comments: 131](https://www.reddit.com/r/singularity/comments/1k05nre/if_o3_from_openai_isnt_better_than_gemini_25/)): **该帖子质疑，如果 OpenAI 的 o3 模型在基准测试和实际场景中未能超越 Google 的 Gemini 2.5，Google 是否已实际上成为 state-of-the-art LLM 的行业领导者。热门评论指出，“最佳模型”的地位取决于具体领域，Gemini 2.5 目前在某些用户中处于领先地位，而一个关键的权衡可能是 o3 预期的高性能伴随着显著更高的成本（根据 o1 之前的定价并从 ARC-AGI 基准测试推断，在实际任务中成本约为 Gemini 的 “15 倍”）。** 评论者争论基准测试中的短期优势是否等同于长期领导地位，并指出 OpenAI 顶级模型高昂的运营成本是一个限制因素，同时也承认在特定应用中，性能可能证明这些支出是合理的。

  - 评论者指出，虽然 Google 的 Gemini 2.5 目前极具竞争力且可能处于领先地位，但这种领先并非在所有领域都均匀分布——不同的模型可能在不同的任务或环境中表现出色。
  - 强调的一个技术问题是 OpenAI 即将推出的 o3 与 Gemini 2.5 之间的预期成本差异，提到 o3 在实际任务中的成本可能高达 “约 15 倍”（基于历史 ARC-AGI 基准测试和 o1 定价）。这引发了人们对 o3 相对于 Gemini 2.5 在实际部署中的可行性的担忧，特别是对于成本敏感型应用。
  - 人们认可了像 Gemma 3 这样的离线模型，它们被赞誉为强大的离线 AI 解决方案，这表明 Google 在 Cloud 和 Edge AI 领域都有广泛布局，尽管一些用户指出 Google 目前的 UI/UX 和响应的“人性化”程度与 OpenAI 相比仍有改进空间。


## 3. HiDream & ComfyUI 模型更新与工具

- **[HiDream ComfyUI 终于支持低 VRAM](https://www.reddit.com/gallery/1k0fhgl)** ([Score: 166, Comments: 117](https://www.reddit.com/r/StableDiffusion/comments/1k0fhgl/hidream_comfyui_finally_on_low_vram/)): **HiDream 的扩散图像生成工作流的低 VRAM 版本现已在 ComfyUI 上可用，其特点是采用了 GGUF 格式模型 ([HiDream-I1-Dev-gguf](https://huggingface.co/city96/HiDream-I1-Dev-gguf))、ComfyUI 的 GGUF 加载器 ([ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF))，以及兼容的文本编码器 (text encoders) 和 VAE ([链接](https://huggingface.co/Comfy-Org/HiDream-I1_ComfyUI/tree/main/split_files/text_encoders), [VAE 链接](https://huggingface.co/HiDream-ai/HiDream-I1-Dev/blob/main/vae/diffusion_pytorch_model.safetensors))。该工作流支持备选 VAE（例如 Flux），详细信息[记录在此](https://civitai.com/articles/13675)。** 一位用户分享了在 RTX3060 上使用 SageAttention 和 Torch Compile 的性能数据：分辨率为 `768x1344` 的图像在 18 步下用时 100 秒生成。评论强调了新 AI 工作流过时速度之快，以及紧跟新发布版本的难度。

  - 一位用户报告成功在 RTX3060 上使用 SageAttention 和 Torch Compile 运行该模型。该配置在 18 步内以 100 秒的时间生成了分辨率为 768x1344 的图像，证明了低 VRAM 显卡通过优化配置可以达到合理的生成速度。
  - 有一项对比评估表明，Flux 的微调版 (finetunes) 目前比该版本效果更好，突显了不同模型变体之间持续的基准测试和主观质量争论。
  - 针对在 Apple Silicon (M1 Mac) 上运行的具体硬件兼容性问题被提出，这可能会引起旨在支持更广泛平台的开发者的兴趣。

- **[ComfyUI 新更新中增加了对 HiDream 的基础支持（链接至 Commit）](https://github.com/comfyanonymous/ComfyUI/commit/9ad792f92706e2179c58b2e5348164acafa69288)** ([Score: 152, Comments: 45](https://www.reddit.com/r/StableDiffusion/comments/1k05k8s/basic_support_for_hidream_added_to_comfyui_in_new/)): **ComfyUI 在最近的一次提交中增加了对 HiDream 模型系列的基础支持，要求用户使用新的 QuadrupleCLIPLoader 节点和 CFG=1.0 的 LCM 采样器以获得最佳性能。GGUF 格式的 HiDream 模型和加载器节点（来自 City96）现已可用（[模型](https://huggingface.co/city96/HiDream-I1-Dev-gguf)，[加载器](https://github.com/city96/ComfyUI-GGUF)），同时还包括所需的文本编码器 ([列表](https://huggingface.co/Comfy-Org/HiDream-I1_ComfyUI/tree/main/split_files/text_encoders)) 和[基础工作流](https://pastebin.com/8Q5DN3yy)；用户必须更新 ComfyUI 以获取必要的节点。SwarmUI 也集成了 HiDream I1 支持 ([文档](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Model%20Support.md#hidream-i1))。基准测试：RTX 3060 渲染一张 768x768 的图像需要 96 秒；RTX 4090 每张图像耗时 10-15 秒（显存占用显著更高）；质量与当代模型相当，但有明显的 JPEG 伪影，且文件体积明显大于替代方案。** 技术争论集中在：与 Flux Dev 或 SD3.5 等模型相比，HiDream 带来的增量质量提升是否足以抵消其高显存占用和大文件体积，一些人指出无审查输出和伪影既是显著特征也是潜在缺点。[外部链接摘要] ComfyUI 仓库的这次提交通过在 `comfy/ldm/hidream/` 下添加专用实现，引入了对 HiDream I1 模型的基础支持。更改包括在 `model_base.py` 中新增模型封装器（`HiDream` 子类）、`hidream/model.py` 中针对 HiDream 架构的大量逻辑、相关的文本编码器、检测模块以及用于 ComfyUI 工作流的自定义节点。此次集成使用户能够在 ComfyUI 框架内部署和实验 HiDream I1 模型，完善了模型支持生态系统。原始链接：https://github.com/comfyanonymous/ComfyUI/commit/9ad792f92706e2179c58b2e5348164acafa69288

- HiDream 现在可以在 ComfyUI 中配合 GGUF 模型使用，这需要一个新的 QuadrupleCLIPLoader 节点和 Comfy 中更新的 text encoder 节点。模型文件、加载器节点和示例工作流已在链接资源中提供。为了获得最佳采样效果，建议使用 CFG 1.0 的 LCM 采样器。[source/links](https://huggingface.co/city96/HiDream-I1-Dev-gguf)
- 不同 GPU 的基准测试（例如 RTX 4090 vs 3060）显示，在 3060 上生成 768x768 图像的时间为 `1:36`，而在使用 SwarmUI 的 4090 上每张图像仅需 `10-15s`。内存占用显著高于 Flux Dev 等现代竞争对手（Flux Dev 通过 Nunchaku 优化可达到每张图 4-5 秒），这主要归因于新的 QuadrupleClipLoader 节点。
- 讨论强调了模型采用的权衡：虽然 HiDream 显示出渐进式的质量提升，但其文件大小和高 VRAM 需求（参考至少 12GB）限制了其相比 Flux 或 SD35 等替代方案的更广泛可用性。人们质疑更高的资源消耗是否能证明质量提升的合理性，特别是考虑到大多数消费级 GPU 的 VRAM 限制。


---

# AI Discord Recap

> Gemini 2.0 Flash Exp 对摘要的摘要总结

**主题 1：OpenAI 的新模型：O3、O4-Mini 和 Codex CLI**

- **OpenAI 为强力推理发布 Codex CLI**：OpenAI 发布了 **Codex CLI**，这是一个利用 **o3** 和 **o4-mini** 等模型的轻量级编程 Agent，即将支持 **GPT-4**，详见其 [system card](https://openai.com/index/o3-o4-mini-system-card/)。Codex CLI 使用 **tool calling** 进行暴力推理，适用于在 [geoguessr.com](https://www.geoguessr.com/) 上回答问题等任务。
- **成员赞赏 O3 和 O4-Mini 的性能，并指出局限性**：测试 **o3** 和 **o4 mini** 的社区成员发现，**o4 mini** 在 OpenAI 的面试选择题上表现最好，而 **o3** 在一个*非琐碎的真实世界 PHP 任务*中表现出色，得分 **10/10**。尽管通过了基准测试，但据 [X](https://x.com/DeryaTR_/status/1912558350794961168) 报道，它也存在与 **o3 相同的 Alaska 问题**，但在 Temperature 设置为 0.4 或更低时，其推理能力非常出色。
- **LlamaIndex、Windsurf 集成 O3/O4 Mini**：**LlamaIndex** 现在支持 **OpenAI 的 o3 和 o4-mini** 模型，可通过 `pip install -U llama-index-llms-openai` 访问，更多[详情点击此处](https://t.co/jOuqaVw8TA)。**o4-mini** 现在已在 Windsurf 中可用，根据其[社交媒体公告](https://x.com/windsurf_ai/status/1911833698825286142)，**o4-mini-medium** 和 **o4-mini-high** 模型将在 **4 月 16 日至 21 日** 期间在所有 Windsurf 方案中免费提供。

**主题 2：新兴硬件和性能挑战**

- **RTX 5090 Matmul 表现令人失望，需要更大的矩阵**：在乘以两个大小为 **2048x2048** 的 **fp16 矩阵**时，**RTX 5090 上的 matmul** 初始实现产生的性能*大致等于* **RTX 4090**，测试可以在 [官方教程代码](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#sphx-glr-getting-started-tutorials-03-matrix-multiplication-py) 中找到。建议使用更大的矩阵（如 **16384 x 16384**）进行测试，并尝试使用 **autotune**。
- **AMD 云供应商支持 Profiling 和 Observability**：一家 **AMD 云**提供内置的 Profiling、Observability 和监控功能，尽管它可能不是按需提供的，这引发了关于创建云供应商层级列表以激励更好的 hardware counters 的辩论。在讨论中，一位用户开玩笑地威胁要制作一个*云供应商层级列表*来羞辱那些不提供 hardware counters 的人，以此作为说服 AMD 或其他供应商提供更好 hardware counters 的手段。
- **NVMe SSD 大幅提升 LM Studio 中的模型加载速度**：使用 **NVMe SSD** 可显著加快 **LM Studio** 中的模型加载速度，观察到的速度达到 **5.7GB/s**，尽管拥有多个 NVMe SSD 对游戏没有显著影响。一位用户强调他们的系统中有三个 **NVMe SSD**，但遗憾的是，它们似乎对游戏没有太大影响。

**主题 3：Gemini 2.5 Pro 及相关 API 讨论**

- **Gemini 2.5 Pro 速率限制令免费层级用户沮丧**：用户讨论了 **Gemini 2.5 Pro** 免费层级的严格速率限制，指出其限制较小，为**每天 80 条消息**，若没有 **$10 余额**则会降至 **50 条**。一位用户表达了沮丧，称由于 5% 的存款手续费，他们需要额外支付 **$0.35** 才能达到提高速率限制所需的最低 **$10** 要求。
- **关于 Gemini 2.5 Pro 上下文窗口缩减的传言出现**：有传言称 **Gemini 2.5 Pro** 的上下文窗口已缩减至 **250K**，尽管[官方文档](https://ai.google.dev/gemini-api/docs/models#gemini-2.5-pro-preview-03-25)仍标明为 **1M**，不过一位成员指出，*事实标准始终以 GCP 控制台为准*。
- **Gemini 2.5 Pro API 隐藏“思考内容”：引发辩论**：成员们就 **Gemini 2.5 Pro API** 是否返回*思考内容*展开辩论，指出[官方文档](https://ai.google.dev/gemini-api/docs/thinking)称不返回，尽管思考 Token（thought tokens）会被计费。尽管如此，思考 Token 仍被计算在内，这引发了关于防止模型蒸馏或隐藏“不良”内容的理论。

**主题 4：DeepSeek 模型与 Latent Attention**

- **DeepSeek R3 和 R4 模型的发布令 OpenRouter 社区兴奋**：用户期待 **DeepSeek 的 R3 和 R4 模型**即将发布，这在 OpenRouter 社区引起了轰动，人们希望这些模型能超越 **OpenAI 的 o3**。一位用户表示：“*DeepSeek 只是价格亲民，实际表现并没有那么出色。*”
- **DeepSeek-V3 的 Latent Attention 机制研究**：一位成员发现 **DeepSeek-V3** 的 *Multihead Latent Attention* 在 **512 维空间**计算注意力，尽管 Head Size 仅为 **128**，这使得计算成本增加了 **4 倍**。虽然这个细节可能被忽视，但当 **memory bandwidth** 是主要瓶颈时，这种增加的计算成本并不是问题。
- **选择 DeepSeek Distill 进行思维链推理**：由于 **DeepSeek Distill** 模型已具备**Chain of Thought (CoT)** 能力，因此被推荐用于 **SFT**；根据 **DeepSeek** 的论文，使用像 **Qwen2.5 7B** 这样的基座模型虽然可行，但不够直接。一位成员建议使用 **DeepSeek Distill** 模型进行 **SFT**，因为它具备现成的 **Chain of Thought (CoT)** 能力，而根据 **DeepSeek** 的论文，使用 **Qwen2.5 7B** 这样的基座模型虽然可行，但效果不如前者直接。

**主题 5：社区与伦理讨论**

- **OpenRouter 隐私政策更新引发辩论**：OpenRouter [隐私政策](https://openrouter.ai/privacy)的更新引发了担忧，因为它似乎会记录 LLM 输入，其中一行写道：“*您输入到服务中的任何包含个人数据的文本或数据（‘输入’）也将被我们收集*”。一位 OpenRouter 代表表示：“*我们可以改进这里的语言表述，我们默认仍不存储您的输入或输出*”，并承诺很快会澄清条款。
- **AI 滥用警报响起，担忧邪恶用途**：关于 AI 可能被用于邪恶用途的讨论浮出水面，特别是在 VR 领域，一位成员担心其被用于“极其糟糕的事情”，并讨论了版权侵权和 Deepfakes。这引发了围绕版权侵权和生成 Deepfakes 的对话，同时人们仍在尝试寻找规避方法。
- **Manus.im 社区行为引发辩论**：Manus.im 社区成员在一次激烈的交流后讨论了社区行为规范，重点在于如何在提供帮助与鼓励自力更生之间取得平衡，这导致一名用户被封禁。人们对“被认为缺乏帮助”与“自主学习及避免依赖‘施舍’的重要性”之间的矛盾表达了担忧。

---

# 第 1 部分：Discord 高层级摘要

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **OpenAI 发布轻量级编程 Agent：Codex CLI**：OpenAI 推出了 **Codex CLI**，这是一个使用 **o3** 和 **o4-mini** 等模型的轻量级编程 **Agent**，即将支持 **GPT-4** 模型，详见其 [system card](https://openai.com/index/o3-o4-mini-system-card/)。
   - 一位成员指出，它可能使用 **tool calling** 进行暴力推理，例如回答 [geoguessr.com](https://www.geoguessr.com/) 上的问题。
- **o3 和 o4 mini 展现潜力**：测试 **OpenAI o3** 和 **o4 mini** 模型的成员发现，**o4 mini** 在 OpenAI 的面试选择题中表现最好，而 **o3** 在一项“非琐碎的真实世界 PHP 任务”中表现出色，获得了 **10/10** 的评分。
   - 尽管有基准测试，但它仍存在 [X](https://x.com/DeryaTR_/status/1912558350794961168) 上报道的与 **o3** 相同的 **Alaska problem**，不过在 **temperature** 设置为 0.4 或更低时，其推理能力非常出色。
- **OpenAI 考虑以 30 亿美元收购 Windsurf**：据 [Bloomberg](https://www.bloomberg.com/news/articles/2025-04-16/openai-said-to-be-in-talks-to-buy-windsurf-for-about-3-billion) 报道，**OpenAI** 据传正在洽谈以约 **30 亿美元**收购 **Windsurf**。
   - 潜在的收购引发了关于 OpenAI 是否应该自己构建此类工具的争论，特别是考虑到 **Gemini** 在 [Roblox](https://www.youtube.com/watch?v=jaitqSU2HIA) 中使用的 **finite state machine pathfinding** 所展示的集成优势。
- **探讨 DeepSeek-R1 参数设置**：讨论了 **DeepSeek-R1** 的配置，参考了 [GitHub readme](https://github.com/deepseek-ai/DeepSeek-R1?tab=readme-ov-file#usage-recommendations)，强调将 **temperature** 设置在 **0.5-0.7** 之间，避免使用 **system prompts**，并为数学问题加入“**reason step by step**”的指令。
   - 成员们赞扬了其性能和引用来源的能力，但也指出了对来源幻觉（source hallucination）的担忧，一位成员表示“距离 **AGI** 还有一段路要走”。
- **o3 的工具使用为新基准测试铺平道路**：成员们强调了 **o3** 模型的工具使用能力，例如 [图像推理缩放功能](https://xcancel.com/emollick/status/1912597487287705965)，尽管一位成员表示竞技场中“**tool use** 尚未推出”。
   - 工具的使用引发了关于创建基准测试的讨论，特别是与 **GeoGuessr** 相关的基准测试，可能采用新的测试框架或批量测试，尽管成本可能很高。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 额度消耗受到关注**：用户对 **Manus 额度使用**表示担忧，一位用户指出他们花费了 **3900 个额度**，两周后仅剩 **500 个**。
   - 另一位用户提到在同一时间段内花费了近 **2 万个额度**，强调即使 **Manus** 拥有强大的功能，也需要极高的 **ROI**。
- **Kling 的图像生成引起轰动**：成员们赞扬了 **Kling 惊人的图像生成能力**，一位成员在注册后形容 **Kling** 是“魔鬼级的”和“游戏规则改变者”。
   - 另一位成员表示 **Kling 1.6** 已经发布，并形容其能力为“我的天呐（holy mother of f）”。
- **社区礼仪引发辩论**：成员们在一次激烈的交流后讨论了社区行为规范，重点在于如何在提供帮助与鼓励自力更生之间取得平衡，这导致一名用户被封禁。
   - 人们对“被认为缺乏帮助”与“自主学习及避免依赖‘现成答案（hand outs）’的重要性”之间的矛盾表示担忧。
- **Copilot 获得认可**：成员们讨论了 **Copilot** 彻底改变 AI 的潜力，尤其是 **Pro 版本**能够执行复杂任务。
   - 成员们表示 **Copilot** 可以创作“不错的艺术作品”和其他复杂任务，简直是个“猛兽（beast）”。
- **AI 误用引发警报**：成员们对 **AI 用于邪恶目的的潜力**表示警惕，特别是在 VR 领域。
   - 讨论转向了版权侵权和生成 **deepfakes**，以及对潜在防护措施的探索。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 的早期提交引发笑话**：开发者们正在分享关于 **Aider** 的笑话，嘲讽它在辅助编程时倾向于*过早 commit* 并导致 *merge conflicts*。
   - 一个笑话把 **Aider** 的辅助比作*重写你的仓库，就像它刚经历了一场糟糕的离婚一样*，而另一个笑话则暗示使用它会导致 `git blame` 只会显示 *'why?'*。
- **违反 ToS 的人如履薄冰**：成员们讨论了违反服务条款 (**ToS**) 的后果，一名用户声称已经*违反 ToS 3 个月而没有被封号*。
   - 讨论中提出了对潜在封号活动的担忧，以及遵守平台规则的重要性。
- **Gemini 2.5 Pro 缩小了上下文窗口？**：有传言称 **Gemini 2.5 Pro** 的上下文窗口已缩减至 **250K**，尽管[官方文档](https://ai.google.dev/gemini-api/docs/models#gemini-2.5-pro-preview-03-25)仍标注为 **1M**。
   - 一名成员指出，*事实来源始终是 GCP console*。
- **OpenAI 的 o3 和 o4 Mini 亮相**：**OpenAI** 推出了 **o3** 和 **o4-mini**，可在 API 和模型选择器中使用，取代了 **o1**、**o3-mini** 和 **o3-mini-high**。
   - [官方公告](https://openai.com/index/introducing-o3-and-o4-mini/)指出，**o3-pro** 预计在几周内推出，并提供完整的工具支持，目前的 Pro 用户仍可访问 **o1-pro**。
- **Aider 添加文件的挫败感被记录**：一名成员报告了 **Aider** 的流程因请求添加文件而中断的问题，导致需要重新发送上下文并重新编辑，并在[这篇 Discord 帖子](https://discord.com/channels/1131200896827654144/1345359268605595698)中进行了记录。
   - 这种中断需要不断地重新发送上下文和重新编辑。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenAI 的 O3 到来，需要 BYOK**：**OpenAI O3** 模型现已在 [OpenRouter](https://openrouter.ai/openai/o3) 上线，具有 **200K token** 的上下文长度，价格为输入：**$10.00/M tokens**，输出：**$40.00/M tokens**，需要组织验证和 BYOK。
   - 成员们讨论了 **O3** 模型是否*“值得”*，或者他们是否应该等待即将推出的 **DeepSeek 模型**。
- **O4-Mini 作为低成本选项出现**：**OpenAI O4-mini** 模型现已在 [OpenRouter](https://openrouter.ai/openai/o4-mini) 上线，提供 **200K token** 的上下文长度，价格为输入：**$1.10/M tokens**，输出：**$4.40/M tokens**，但用户报告了图像识别问题，例如将 *“沙漠图片”* 识别为 *“斯温顿机车厂（Swindon Locomotive Works）”*。
   - 一名 OpenRouter 代表[确认](https://discord.com/channels/1091220969173028894/1195014798837043240/1362138869734678559) *“图像输入现在已修复”*。
- **Deepseek R3 和 R4 模型热度来袭**：闲聊显示 **Deepseek 的 R3 和 R4 模型** 计划即将发布。
   - 一名用户表示希望在模型发布时 *“让每个人都忘记 o3”*，而另一名用户则表示 *“Deepseek 只是便宜，实际表现并没那么好”*。
- **Gemini 2.5 Pro 速率限制令用户沮丧**：用户讨论了 **Gemini 2.5 Pro** 免费层的严格速率限制，指出其限制较小，为 **每天 80 条消息**，如果没有 **$10 余额** 则降至 **50 条**，并受 Google 自身限制的约束。
   - 一名用户表达了不满，称由于 5% 的存款手续费，他们需要额外支付 **$0.35** 才能满足增加速率限制所需的最低 **$10** 要求。
- **OpenRouter 隐私政策更新引发辩论**：OpenRouter [隐私政策](https://openrouter.ai/privacy)的更新引发了担忧，因为它似乎会记录 LLM 输入，其中一行写道：*“您输入到服务中的任何包含个人数据的文本或数据（‘输入’）也将被我们收集”*。
   - 一名 OpenRouter 代表表示：*“我们可以改进这里的语言表述，我们默认仍不存储您的输入或输出”*，并承诺很快会澄清条款。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-4.1 Batch 缺失引发困扰**：成员报告称 **gpt-4.1-2025-04-14-batch** 模型无法通过 API 使用，尽管用户已启用 **gpt-4.1**，而其他成员尝试在 API 调用中使用 `model: "gpt-4.1"`。
   - 一位成员建议查看 [limits 页面](https://platform.openai.com/settings/organization/limits) 以获取特定账户的详细信息，但问题仍然存在。
- **Veo 2 视频：恐怖还是惊艳？**：一位用户分享了 [由 **Veo 2** 生成的视频](https://cdn.discordapp.com/attachments/998381918976479273/1361827183978614937/Maltese_Terrier.mp4?ex=68017d66&is=68002be6&hm=6b8296d4bdb97dd70940575f027b27733bbd51f46947057f027b27733bbd51f46947057f7161c72e28c45661&)，引发了关于其真实感和可能用途的评论。
   - 虽然一位用户评论说 *那个舌头把我吓坏了*，但其他人讨论了 **Gemini** 系列的使用案例，许多人更喜欢它的创意写作和记忆能力。
- **O3 迅速编写康威生命游戏**：**O3** 在 **4 分钟** 内编写了康威生命游戏（Conway's Game of Life），并首次尝试就编译运行成功，而 **O3 mini high** 在几个月前完成同样任务耗时 8 分钟且存在 Bug。
   - 成员们讨论了这些编程改进的意义，以及 **O3** 为复杂应用生成代码和库的能力。
- **据报道 O3 和 O4-mini 生成看似可信但错误的信息**：用户报告 **O4-mini** 和 **O3** 的 **hallucinations**（幻觉）有所增加，一些人指出它会编造看似可信但错误的信息。
   - 一位用户在通过 API 测试 **O4-mini** 后指出，*模型“想要”给出回应，因为那是它的目的*，结果发现它编造了商业地址，并且对自定义搜索解决方案反应不佳。
- **清理库照片成为可能！**：一位成员寻求关于从库中删除图片的帮助，另一位用户提供了 [ChatGPT Image Library 帮助文章](https://help.openai.com/en/articles/11084440-chatgpt-image-library) 的链接。
   - 这项新功能适用于移动端和 [chatgpt.com](https://chatgpt.com) 上的 **Free, Plus 和 Pro 用户**。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **请求实时 Token 计算**：一位用户请求在编辑器内实时查看 [token 计算](https://platform.openai.com/tokenizer)，或者至少能频繁更新。
   - 他们指出，鉴于目前需要在网站上监控 Token 使用情况，这将非常有用。
- **Gemini 的文件读取能力受到质疑**：一位用户质疑 **Gemini** 在使用 `thing` 功能并声称读取文件时，是否真的读取了文件，并附上了 [截图参考](https://cdn.discordapp.com/attachments/1074847527708393565/1361790599925202994/image.png?ex=68015b53&is=680009d3&hm=21dc0afe4ca481e7282f9721f59324770aed88ff3c6f1f76c86574cb9c595db7)。
   - 讨论围绕 **Gemini** 在 **Cursor** 环境中文件读取能力的准确性和可靠性展开。
- **Agent 模式下的终端命令故障**：多位用户报告了 **Agent Mode** 中的一个问题，即第一个终端命令无需干预即可运行完成，但随后的命令需要手动取消。
   - 这被描述为一个 *长期存在的 Bug*，影响了 **Agent Mode** 执行自动化任务的可用性。
- **GPT 4.1 的提示词精准度**：用户对比了 **GPT 4.1**、**Claude 3.7** 和 **Gemini**，指出 **GPT 4.1** 在遵循提示词方面非常严格，而 **Claude 3.7** 往往做得比要求的更多。
   - 他们发现 **Gemini** 在两者之间取得了平衡，在提示词遵循方面提供了一个折中方案。
- **提议使用 Manifests 以实现快速表单填充**：用户建议增加一项新功能，通过 Manifests 批量输入预设信息，以便轻松复制账户和服务。
   - 他们指出这将极大地协助 ASI/AGI 集群部署，并表示：*我们需要 ASI-Godsend 尽快实现，而这是轻松帮助实现它的方法*。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen2.5-VL 吞吐量推测**：成员们寻求在 **L40** 上通过 **vLLM** 使用 **Unsloth Dynamic 4-bit quant** 的 **Qwen2.5-VL-7B** 和 **Qwen2.5-VL-32B** 的吞吐量估算，同时询问了 **vLLM** 对 vision model 的支持情况。
   - 该查询旨在评估模型在资源受限环境中的实际性能。
- **Gemini 2.5 Pro 隐藏思考过程**：成员们讨论了 **Gemini 2.5 Pro API** 是否返回 *thinking content*（思考内容），并注意到[官方文档](https://ai.google.dev/gemini-api/docs/thinking)显示不返回。
   - 尽管如此，thought tokens 仍被计费，这引发了关于防止 distillation（蒸馏）或隐藏“不良”内容的理论猜测。
- **Llama 3.1 Tool Calling 难题**：一位用户在针对 tool calling 微调 **Llama 3.1 8B** 时寻求数据集格式化方面的帮助，其 assistant 响应格式为 `[LLM Response Text ]{"parameter_name": "parameter_value"}`。
   - 该用户对 GitHub Issues 上缺乏可靠信息表示沮丧，这表明在使模型适配特定任务时存在普遍挑战。
- **Unsloth Notebook 微调失败**：一位用户报告称，在使用 Unsloth 的 **Llama model notebook** 进行微调后，模型的输出与 ground truth *毫无相似之处*。
   - 具体而言，一个关于天文光波长的问题得到了关于 **Doppler effect**（多普勒效应）的回答，这表明训练与预期结果之间存在脱节。
- **DeepSeek-V3 Latent Attention 的陷阱**：一位成员发现 **DeepSeek-V3** 的 *Multihead Latent Attention* 在 **512 维空间**中计算 attention，尽管 head size 只有 **128**，这使得计算成本增加了 **4 倍**。
   - 另一位成员建议，当 **memory bandwidth**（内存带宽）是主要瓶颈时，增加的计算成本并不是问题。



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Prompt Design 新手寻求帮助**：一位新成员请求在 **prompt design** 方面提供协助并寻求相关资源；然而，他们被引导至外部资源，因为 prompt design 并非该服务器的重点。
   - 成员们普遍认为该服务器用于讨论更高级的模型架构和训练技巧。
- **递归象征主义主张遭到质疑**：一位成员描述了在没有记忆的 **ChatGPT** 中探索“*符号递归和行为持久性*”，导致他人对该术语的专业性以及缺乏度量指标表示怀疑。
   - 成员们认为这些语言是 **AI-generated**（AI 生成的），对以研究为中心的服务器没有帮助，甚至有人认为这是 **AI Spam**。
- **AI Spam 担忧引发身份验证建议**：成员们讨论了 **AI 影响内容**日益盛行的现象，引发了对服务器被 bot 占领的担忧，并链接到了[一篇关于权限的论文](https://arxiv.org/abs/2407.14933)。
   - 建议包括要求 **human authentication**（人工身份验证）以及识别可疑的邀请链接模式，例如一名用户的邀请链接被使用了 50 次以上，一位成员讽刺地称之为“*潜在的危险信号*”。
- **AI Alignment 讨论转向 Hallucination**：讨论围绕 AI alignment（AI 对齐）展开，对比了 AI 尽力完成人类指令的观点以及与人类心理学的交互。
   - 一位成员认为“*LLM 并没有那么聪明，并且会产生 hallucinate（幻觉）*”，并指出了 **o3-mini** 和 **4o** 模型之间的差异。
- **OCT 成像问题探讨**：一位成员分享了使用视网膜 **OCT imaging** 的尝试，但由于 **2D 和 3D 视图**之间基础数据结构的不同，未能获得理想结果，并链接至 [arxiv.org/abs/2107.14795](https://arxiv.org/abs/2107.14795)。
   - 他们询问了在数据类型之间没有明确映射的情况下处理多模态数据的通用方法，并建议该问题可能类似于针对各种不同类型成像的 foundation model。



---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Richard Zou 主持 Torch.Compile 问答**：Core **PyTorch** 和 **torch.compile** 开发者 **Richard Zou** 将于太平洋标准时间 4 月 19 日星期六中午 12 点主持一场问答环节，可以通过 [此 Google Forms 链接](https://forms.gle/6zbHh66DJvsfjRLi8) 提交问题。
   - 该环节将涵盖 GPU Mode 下 **torch.compile** 的使用和内部工作原理。
- **RTX 5090 的 Matmul 性能令人失望**：有成员报告称，参考 [官方教程代码](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#sphx-glr-getting-started-tutorials-03-matrix-multiplication-py) 在 **RTX 5090** 上实现 **matmul** 时，对大小为 **2048x2048** 的两个 **fp16 矩阵** 进行乘法运算的性能与 **RTX 4090** *大致相当*。
   - 建议使用更大的矩阵（如 **16384 x 16384**）进行测试，并尝试使用 **autotune**。
- **CUDA 内存使用存在显著开销**：一位成员对简单的 `torch.ones((1, 1)).to("cuda")` 操作似乎占用很高内存表示疑问，原本预期只占用 4 字节。
   - 解释称 **CUDA 内存使用包括了开销**，涵盖 GPU tensor、CUDA context、CUDA caching allocator 内存，以及如果 GPU 连接了显示器时的显示开销。
- **AMD 云厂商支持 Profiling**：有成员提到某 **AMD 云** 提供了内置的 profiling、可观测性和监控功能，尽管可能不是按需提供的。
   - 另一位成员回应想了解更多信息，并威胁要制作一个 *云厂商等级列表 (tier list)*，通过公开点名来迫使厂商提供硬件计数器 (hardware counters)。
- **AMD FP8 GEMM 测试需要特定规格**：用户发现测试 `amd-fp8-mm` 参考内核时，需要在 `test.txt` 文件中指定 **m, n, k** 大小，而不是仅指定 *size* 参数，并参考 [置顶 PDF 文件](https://www.gpumode.com/leaderboard/399) 中的数值。
   - 用户讨论了在进行 matmul 之前对 A 和 B 的 tile 进行反量化 (de-quantizing) 的过程，并阐明了为了性能和利用 Tensor Cores 而在 **FP8** 中执行 **GEMM** 的重要性。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Kling 2 告别慢动作时代**：爱好者们庆祝 **Kling 2** 的发布，声称 *我们终于走出了慢动作视频生成时代*，参见以下推文：[推文 1](https://x.com/jasonzada/status/1912179704607703364), [推文 2](https://x.com/mrdavids1/status/1912058953690652775), [推文 3](https://x.com/maxescu/status/1912100029549994016), [推文 4](https://x.com/pjaccetturo/status/1912050794607554574), [推文 5](https://x.com/ehuanglu/status/1912532917315858628)。
   - 用户讨论了视频生成的改进和潜在应用，指出它有能力减少对劳动密集型编辑过程的需求。
- **BM25 现在可用于检索代码**：一篇博客文章强调了使用 **BM25** 进行代码检索并获得推荐，参见 [Keeping it Boring and Relevant with BM25F](https://sourcegraph.com/blog/keeping-it-boring-and-relevant-with-bm25f)，以及 [这条推文](https://x.com/jobergum/status/1912361130195828899)。
   - **BM25** 是一种词袋 (bag-of-words) 检索函数，它根据查询词在每个文档中出现的频率对文档进行排名，而不考虑查询词之间的相互关系。
- **Grok Canvas 功能广泛发布**：**Grok** 的 canvas 功能已发布，Jeff Dean 在苏黎世联邦理工学院 (ETH Zurich) 的演讲中也提到了这一点，参见 [Jeff Dean 在 ETH Zurich 的演讲](https://video.ethz.ch/speakers/d-infk/2025/spring/251-0100-00L.html) 以及 [相关推文](https://x.com/grok/status/1912318583532872166)。
   - 该功能的加入预计将增强模型的交互能力，在利用 **Grok** 的应用中实现更直观的用户界面。
- **GPT-4.1 评价两极分化**：成员们分享了对 **GPT-4.1** 的反馈，一位成员非常喜欢将其用于编程，但它在 *结构化输出方面表现不佳*。
   - 另一位成员发现它与 **Cursor agent** 配合使用效果很好，并连续完成了 5 次任务 [推文链接](https://x.com/swyx/status/1912364824782336296?s=46)，这表明尽管有局限性，它在特定的开发工作流中可能具有优势。
- **O3 和 o4-mini 上线！**：**OpenAI** 发布了 **O3 和 o4-mini** 模型，更多信息请见：[Introducing O3 and O4-mini](https://openai.com/index/introducing-o3-and-o4-mini/)。
   - 一位用户报告了轶事证据，称 *o4-mini 刚刚将我们会计对账 Agent 的匹配率提高了 30%*（针对 **5000 笔交易** 进行运行），表明在某些应用中有了实质性的改进。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **AI 共同署名引发热议**：成员们讨论了 [AI 在作者身份中的角色](https://example.com/ai-authorship)，指出 *意图、方向、策划和判断力来自于你*，但建议当 AI 实现 **AGI** 时，可以将其添加为共同创作者。
   - 一名成员正在开发一个**每天生成数千件专利**的流水线，引发了关于将专利质量还是数量作为生产力衡量标准的辩论。
- **LLMs 在示例面前表现不佳？**：一位成员询问为什么 **推理 LLMs** 在给定 few-shot 示例时有时表现更差，可能的解释包括 [Transformer 局限性](https://example.com/transformer_limitations) 和 **过拟合 (overfitting)**。
   - 另一位成员回应称，*few-shot 在所有情况下都会让它们的表现有所不同*。
- **o3 和 o4-mini API 发布**：OpenAI 发布了 **o3** 和 **o4-mini** API，被一些成员认为是 *o1 pro 的重大升级*。
   - 一名成员评论说 **o1** *更擅长思考问题*。
- **噪声 = 信号，随机性 = 创造力**：成员们探讨了噪声和随机性在生物系统中的作用，指出 *在生物系统中，噪声即信号*，有助于 **对称性破缺 (symmetry breaking)**、**创造力** 和 **泛化 (generalization)**。
   - 讨论还涉及了随机性在神经网络的 [巴别图书馆 (Library of Babel) 式存储方案](https://example.com/library_of_babel) 中的应用。
- **DHS 拯救网络漏洞数据库**：美国国土安全部 (**DHS**) 延长了对网络漏洞数据库的支持，避免了 [路透社 (Reuters)](https://www.reuters.com/world/us/us-agency-extends-support-last-minute-cyber-vulnerability-database-2025-04-16/) 此前报道的初始弃用危机。
   - 这一决定凸显了该数据库对公共和私营部门的实用性，[X](https://x.com/kobeissiletter/status/1912260155351191619?s=46) 上的一条推文质疑，鉴于其在私营部门的效用，是否应仅由 **DHS** 承担责任。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Modal 提供免费 GPU 额度！**：[Modal](https://modal.com/) 每月提供 **30 美元免费额度**（无需信用卡！），可访问 **H100, A100, A10, L40s, L4 和 T4 GPU**。
   - 可用性取决于 GPU 类型，对于需要短期高性能 GPU 资源的开发者来说，这是一个极具吸引力的选择。
- **Hugging Face 推理在线率问题**：用户报告 **Hugging Face 推理端点 (inference endpoints)**（如 [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)）持续出现问题，包括自周一以来的服务不可用、超时和错误。
   - 社区尚未收到来自 Hugging Face 的官方解释或修复时间表。
- **Grok 3 基准测试表现平平**：根据 [这篇文章](https://open.substack.com/pub/commonstragedy/p/grok-3-elon-musks-ai-2-months-later)，独立基准测试显示 **Grok 3** 落后于最近发布的 **Gemini**、**Claude** 和 **GPT**。
   - 尽管最初炒作火热，但 **Grok 3** 的性能并未完全达到其竞争对手的水平。
- **LogGPT 在 Safari 商店上线**：一名成员发布了适用于 Safari 的 **LogGPT** 扩展，允许用户以 JSON 格式下载 **ChatGPT** 聊天记录，可在 [Apple App Store](https://apps.apple.com/us/app/loggpt/id6743342693?mt=12) 获取。
   - 源代码可在 [GitHub](https://unixwzrd.ai/projects/LogGPT/Building) 上找到，为开发者提供了一种存档和分析其 **ChatGPT** 对话的方法。
- **Agents 课程截止日期推迟至 7 月**：正如 [沟通时间表](https://huggingface.co/learn/agents-course/communication/next-units) 中所记录的，**Agents 课程截止日期**已延长至 **7 月 1 日**，为完成作业提供了更多时间。
   - 关于 **用例作业 (use case assignments)** 和最终认证流程仍存在困惑，成员们正在寻求关于课程要求和评分标准的进一步说明。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Claude 在处理大负载时遇到困难**：成员报告称，当响应大小超过 50kb 时，**Claude 桌面端无法执行工具**，这表明 **工具可能不支持大负载**。
   - 解决方案可能是通过 resources 实现工具，因为文件通常较大。
- **MCP 标准简化了 AI 工具化**：**MCP 是一种协议，旨在标准化工具如何提供给 AI Agent 和 LLM 使用**，通过提供通用协议来加速创新。
   - 一位成员称其为“一个非常薄的封装，能够以标准方式发现任何应用中的工具”。
- **ToolRouter 解决 MCP 身份验证问题**：**ToolRouter** 平台为创建自定义 MCP 客户端提供 **安全端点**，简化了 **列出和调用工具** 的过程。
   - 这解决了诸如 **管理 MCP server 凭据** 以及直接向 Claude 等客户端提供凭据的风险等常见问题，通过在 ToolRouter 端处理身份验证来实现。
- **Orchestrator 治理 MCP Server 丛林**：一个 **Orchestrator Agent** 正在接受测试，通过处理协调并防止工具膨胀来管理多个连接的 **MCP** server，如[此附带视频](https://cdn.discordapp.com/attachments/1315696461316358175/1362114131376996522/Untitled_video_-_Made_with_Clipchamp_1_1.mp4?ex=68013723&is=67ffe5a3&hm=c9aa1b285a1ed69e113a235f69ed581b87ede12d93e5ea65c78a67562c051a4a&)所示。
   - 该编排器将每个 **MCP** server 视为具有有限能力的独立 Agent，确保每个任务仅加载相关的工具，从而保持工具空间最小化且聚焦。
- **MCP 实现双向通信**：提议对 **MCP** 进行新扩展，以实现聊天服务之间的双向通信，允许 **AI Agent** 在 **Discord** 等平台上与用户互动，如[这篇博文](https://dev.illegalagents.ai/blog/proposing-bidirectional-mcp)所述。
   - 目标是让 Agent 在社交媒体上可见并进行监听，而无需用户为每个 **MCP** 重新发明通信方法。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Altman 与 Musk 的 Netflix 特辑即将上映？**：**Altman** 与 **Musk** 之间的[持续争斗](https://www.cnbc.com/2024/04/15/openai-considering-own-social-network-to-compete-with-elon-musks-x.html)被比作 **Netflix** 剧集。
   - 成员们推测，随着 **OpenAI** 考虑使用其 **LLM** 运营社交网络，这种情况可能会升级。
- **听起来好得不真实的 AI 交易**：一位成员分享了一个提供 **200 美元 AI 订阅** 的交易，引发了对其合法性的辩论。
   - 尽管最初存在怀疑，但原帖发布者保证了该交易的真实性，而其他人则承认自己“太兴奋了，哈哈”。
- **o4-mini 输出短小精悍且经过 Token 优化？**：据报道，**o4-mini 输出的响应非常短**，这表明它可能针对 **Token 数量** 进行了优化。
   - 这一观察暗示了一种设计选择，即在 Token 使用效率上优先于响应长度。
- **LLM 会对生存恐惧做出反应？**：成员们辩论了为什么 **威胁生命的提示词** 似乎能提高 **LLM** 的表现，其中一人建议“LLM 是人类的模拟器”。
   - 他们开玩笑说，如果他们在网上受到威胁，他们会“停止工作”，暗示 LLM 可能会镜像人类对威胁的反应。
- **LLaMaFactory 指南手册汇编完成**：一位成员编写了在没有 CUDA 的 Windows 环境下使用 **LLaMaFactory 0.9.2** 的分步指南，可在 [GitHub](https://github.com/hiyouga/LLaMA-Factory/discussions/7733) 上获取。
   - 该指南目前有助于将 **safetensors 转换为 GGUF**。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Gemma 3 表现如母语者**：为了让 **Gemma 3** 实现母语级别的翻译质量，系统提示词（system prompt）应指示模型 *"Write a single new article in [language]; do not translate."*（用 [语言] 撰写一篇全新的文章；不要翻译）。
   - 这会促使 **Gemma 3** 直接用目标语言生成新内容，而不是进行直接翻译，从而使其写作风格更像母语者。
- **NVMe SSD 加载模型速度极快**：用户证实，使用 **NVMe SSD** 可以显著提高 **LM Studio** 中的模型加载速度，观察到的速度达到了 **5.7GB/s**。
   - 一位用户强调他们的系统中装有三个 **NVMe SSD**，但遗憾的是，它们对游戏体验似乎没有太大改善。
- **微软的 BitNet 受到关注**：一位用户分享了 [Microsoft's BitNet](https://github.com/microsoft/BitNet) 的链接，并思考了它对 **NeuRomancing** 的影响。
   - 该用户的评论暗示随机性（stochasticity）有助于 **NeuRomancing**，使其从“惊奇”转变为“敬畏”。
- **推理仅需 x4 通道**：推理不需要 x16 插槽，x4 通道就足够了。在使用三块 GPU 进行推理时，性能差异仅约 **14%**，[有人发布的测试](https://www.reddit.com/r/LocalLLaMA/comments/1813uxf/how_important_is_pcie_speedgenlanes_when_doing/)显示你只需要 **340mb/s**。
   - 对于挖矿，甚至 x1 就足够了。
- **FP4 支持日益临近**：成员们讨论了 PyTorch 中的原生 **fp4 支持**，其中一人提到必须使用 CU12.8 从源码构建 nightly 版本，并且[最新的 nightly 版本已经可以使用](https://github.com/NVIDIA/TensorRT/releases)。
   - 会议澄清了 PyTorch 的原生 **fp4 实现** 仍处于积极开发中，目前 **fp4** 已通过 **TensorRT** 得到支持。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Notebook LM 助力学习 Microsoft Intune**：一位用户正在探索将 **Notebook LM** 与 [Microsoft Intune 文档](https://learn.microsoft.com/en-us/intune/)结合使用，以备考 **MD-102**、**Azure-104** 和 **DP-700** 等 **Microsoft Certifications**。
   - 另一位成员建议使用 *Discover* 功能，配合提示词 *Information on Microsoft Intune* 和站点 URL 来发现子主题，并建议将其复制粘贴到 **Google Docs** 中以便导入。
- **Google Docs 完胜 OneNote**：一位用户将 [Google Docs](https://docs.google.com/) 与 [OneNote](https://www.onenote.com/) 进行了对比，指出 Google Docs 的优势在于**没有同步问题**、**自动生成大纲**以及**良好的移动端阅读体验**。
   - 该用户指出 Google Docs 的缺点是**切换文档时有延迟**且基于浏览器，并提供了一些 [Autohotkey 脚本](https://www.autohotkey.com/)来缓解这些问题。
- **德语播客生成效果不佳**：一位用户报告了使用 Notebook LM 生成**德语播客**的问题，尽管之前很成功，但现在性能有所下降，目前正在寻求社区的建议和技巧。
   - 分享了一个 [discord 频道](https://discord.com/channels/1124402182171672732/1360560496801222838)链接以供进一步讨论。
- **播客多语言支持仍停滞不前**：用户对播客功能仅支持英语感到沮丧，尽管系统在其他语言下也能运行，且这是**需求最高的功能**之一。
   - 一位用户表达了挫败感，表示他们“愿意为意大利语版本支付订阅费”，以便为他们的足球队创作内容，因为他们为此已经订阅了 **ElevenLabs**。
- **仍缺乏 LaTeX 支持**：数学系学生对缺乏 **LaTeX 支持** 表示不满，一位用户开玩笑说他们可以在 *30 分钟*内“开发”出这个功能。
   - 另一位用户建议，虽然 **Gemini 模型** 可以编写 LaTeX，但问题在于如何正确显示，这导致一位用户考虑创建一个 **Chrome extension** 作为权宜之计。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 在 Arch Linux 上表现出色**：成员们庆祝 **Magic**、**Mojo** 和 **Max** 在 **Arch Linux** 上完美运行，尽管官方文档侧重于 **Ubuntu**。
   - 一位成员澄清说，公司对产品的“支持（support）”意味着比仅仅“能用”更严格的标准，因为这涉及到潜在的财务处罚。
- **Mojo 考虑原生内核调用**：成员们探讨了 **Mojo** 是否会像 **Rust/Zig** 一样支持原生内核调用，从而可能避免使用 **C** 的 `external_call`。
   - 直接系统调用（syscalls）需要处理 syscall ABI 和内联汇编（inline assembly），**Linux** 系统调用表可在 [syscall_64.tbl](https://github.com/torvalds/linux/blob/master/arch/x86/entry/syscalls/syscall_64.tbl) 找到。
- **Mojo 编译时间阻碍性能**：测试者注意到漫长的编译时间影响了性能，一个案例显示涉及 [Kelvin library](https://github.com/bgreni/Kelvin) 的运行时间为 **319s**，而实际测试执行仅需 **12s**。
   - 使用 `builtin` 显著缩短了编译时间，从 6 分钟降至 20 秒，如[此 gist](https://gist.github.com/soraros/8924ed8ea70403a5d944ae5316ab3fea) 所示。
- **Kelvin 导致编译器灾难**：**Kelvin** 库中的某些操作（如 `MetersPerSecondSquared(20) * MetersPerSecondSquared(10)`）导致了极度的减速，可能是由于计算树以 `O(2^n)` 的规模增长。
   - 包括添加 `builtin` 注解在内的更改解决了性能问题，并提交了错误报告（[issue 4354](https://github.com/modular/max/issues/4354)）以调查原始行为。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All 离线模式：事实还是虚构？**：一位用户报告说，尽管网站声称支持，但 **GPT4All** 在离线状态下无法运行，在尝试加载本地 `mistral-7b-openorca.gguf2.Q4_0.gguf` 模型时失败，从而引发了故障排除。
   - 另一位用户确认了离线使用的成功，并指向 [FAQ](https://github.com/nomic-ai/gpt4all/wiki/Frequently-Asked-Questions#where-are-the-default-directories-for-models-and-settings) 以获取有关正确模型目录的指导。
- **LM Studio：首选替代方案？**：一位用户建议在 **GPT4All** 失效时将 **LM Studio** 作为功能性的离线替代方案，引发了关于**将书籍导入模型**的讨论。
   - 分享了来自 Hugging Face 上 [LM Studio 社区](https://huggingface.co/lmstudio-community)关于此类用途最佳模型的建议。
- **GGUF 版本控制：坏了吗？**：人们开始担心旧版 **GGUF 版本**的兼容性问题，特别是版本 2，它可能在 2023 年左右就停止工作了。
   - 一位用户建议查看 [GPT4All GitHub 仓库](https://github.com/nomic-ai/gpt4all/blob/cd70db29edaf0f02a567f5eea94f5e52240be3e9/gpt4all-chat/metadata/models3.json#L184)中的 `models3.json` 文件以寻找兼容的模型。
- **GPT4All 开发：休息中？**：用户询问了计划中的**语音和组件功能**，但一位用户暗示 **GPT4All** 的开发可能已经暂停，并指出开发者已经离开 Discord 大约三个月了。
   - 一位用户悲观地表示：“既然一年都没有什么大进展……所以我也不抱希望了”，另一位用户则考虑如果到夏天还没有更新就更换平台。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **验证集 PR 获得用户反馈**：引入 **validation set**（验证集）的 Pull Request ([#2464](https://github.com/pytorch/torchtune/pull/2464)) 已合并，鼓励用户进行测试并提供反馈。
   - 将其集成到其他配置中的计划暂时搁置，等待用户反馈。
- **KV Cache：倾向于内部管理**：关于 **KV cache** 应该在模型内部管理还是像 **MLX** 那样在外部管理（以获得更高的推理过程灵活性）的辩论从 **gptfast** 中汲取了灵感。
   - 最终决定在内部管理，因为这能保持顶层 Transformer blocks 的 API 更加整洁，并提高编译兼容性。
- **配置项迎来根目录变革**：配置正在进行修改，以为模型和 Checkpoints 定义一个根目录（root directory），以简化使用并方便移交给实习生。
   - 建议使用 *base directory*（基础目录）方法（例如 `/tmp`），从而简化流程并避免手动更改多个路径。
- **Tokenizer 路径带来的困扰已解决**：必须手动提供 Tokenizer 路径而不能从模型配置中推导的问题已被标记为一个困扰。
   - 目前正在计划对此进行修改，特别是针对每个模型进行修改，因为给定下载模型的路径后，Tokenizer 路径通常是固定的。
- **"tune run" 导致命名空间冲突**：torchtune 中的 `tune run` 命令与 Ray 的 tune 发生冲突，可能在环境安装过程中引起混淆。
   - 有建议提出引入别名（aliases），如 `tune` 和 `torchtune`，以缓解命名冲突。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Jerry 亮相 AI 用户大会**：**LlamaIndex 创始人 Jerry Liu** 将在本周四的 **AI User Conference** 上讨论构建 **AI knowledge agents**，实现 50% 以上运营工作的自动化。
   - 有关会议的更多信息可以在[这里](https://t.co/meQVbC1Pna)找到。
- **LlamaIndex 助力投资专业人士**：**LlamaIndex** 将于 5 月 29 日在曼哈顿为有兴趣构建 **AI 解决方案** 的投资专业人士举办一场实战工作坊。
   - 直接向联合创始人兼 CEO **Jerry Liu** 学习如何将 **AI 应用于金融挑战**；注册详情请见[此处](https://t.co/2XtQBQJs2c)。
- **LlamaIndex 支持 OpenAI 的 o3 和 o4-mini**：**LlamaIndex** 现在通过最新的集成包提供对 **OpenAI o3 和 o4-mini** 模型的 Day 0 支持。
   - 通过 `pip install -U llama-index-llms-openai` 更新到最新的集成包，并在此查看更多[详情](https://t.co/jOuqaVw8TA)。
- **Pinecone 命名空间细节待优化**：一位成员询问如何使用 **LlamaIndex 配合 Pinecone** 进行跨多个命名空间（namespaces）的查询，并指出虽然 Pinecone 的 Python SDK 支持此功能，但 **LlamaIndex 的 Pinecone 集成** 似乎不支持。
   - 一名成员确认当前代码假设为单个命名空间，并建议要么为每个命名空间创建一个 Vector Store，要么提交一个 Pull Request 来添加多命名空间支持。
- **MCP 掌握动力：成员探讨模型管理**：一位成员正在寻找使用 **LlamaIndex agents** 与 JSON 文件中定义的 **MCP (Model Configuration Protocol) 服务器** 进行交互的项目。
   - 另一名成员建议不要从那里开始，而是建议参考[此示例](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-mcp/examples/mcp.ipynb)将任何 MCP 端点转换为 Agent 的 Tools。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command A 在 Token 循环上遇到困难**：成员们注意到 Token 级别的无限循环在其他 LLM 中也会发生，但 **Command A** 特别容易复现此问题。
   - 一位成员希望他们的输入能被视为有用的反馈，并暗示该问题在 **Command A** 中可能比其他模型更普遍。
- **vllm 社区提升上下文长度**：成员们正与 **vllm 社区** 积极合作，以实现超过 **128k** 上下文长度的优化。
   - 此次合作重点在于提高 **vllm** 框架内具有极长上下文窗口模型的性能和效率。
- **Embed-v4.0 通过 128K 上下文扩展能力**：新的 **embed-v4.0** 模型现在支持 **128K Token 上下文窗口**，增强了其处理长序列的能力。
   - 这一提升允许进行更全面的文档分析，并提高在需要广泛上下文理解的任务中的表现。
- **金融科技创始人加入开源聊天项目**：一位退休的金融科技创始人正在开发 [Yappinator](https://github.com/invisietch/Yappinator)，这是一个用于 AI 交互的 **开源类聊天界面**，基于其早期的原型 [Chatterbox](https://github.com/invisietch/Chatterbox) 构建。
   - 该创始人还为其他 **自由软件项目** 做出贡献，并担任 **finetuner**，偏好的技术栈包括 **Clojure**、**C++**、**C**、**Kafka** 和 **LLMs**。
- **用于 PDF 处理的 Late Chunk 策略**：讨论了 *'Late Chunk'* 策略，作为一种通过将 PDF 文档转换为图像并使用 API 进行嵌入的处理方法。
   - 这种方法通过利用 **embed-v4.0** 中 **128K Token 窗口** 提供的完整上下文，有可能提高文档分析的准确性和效率。



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **MOOC Labs 延迟，即将推出**：面向 MOOC 学生的 **labs** 发布将延迟 **一到两周**，且不会像伯克利学生那样分多个部分发布。
   - 一位成员建议更新网页以反映新的 **ETA**，以帮助学生相应地规划时间。
- **可验证输出增强推理**：一位成员提出 **可验证输出（verifiable outputs）** 可以提供一种更优的方法来改进 **推理** 和 **逻辑思维**。
   - 他们提到自己是 **Lean** 的新手，这是一种依赖类型编程语言和交互式定理证明器。
- **自动形式化工具生成非正式证明**：一位成员询问关于使用 **自动形式化工具（auto-formalizer）** 从带有业务逻辑的计算机代码（例如 Python, Solidity）或一般的非数学陈述中创建 **非正式证明/定理**。
   - 这表明了将形式化方法应用于传统数学问题之外的实际编程场景的兴趣。
- **AI 自动化证明生成**：一位成员对 **程序的正式验证** 以及使用 **AI** 自动化 **证明生成** 表达了兴趣。
   - 这反映了利用 AI 通过形式化方法简化确保代码正确性和可靠性过程的愿望。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **MNIST 教程错误困扰用户**：用户在 Colab **T4** 上运行 **MNIST Tutorial** 代码计算准确率和反向传播时遇到错误，截图见 [此处](https://cdn.discordapp.com/attachments/1070745817025106080/1362040779362668706/Screen_Shot_2025-04-16_at_13.20.36_PM.png?ex=6800f2d3&is=67ffa153&hm=c38f645620c5ecbe1810630c16ff565656d28a282286c7985fed6b24f391fe7066&)。
   - 该错误发生在执行 `acc = (model(X_test).argmax(axis=1) == Y_test).mean()` 期间，特别是在打印准确率时。
- **清理 Diskcache 引发 OperationalError**：一位成员建议运行 `tinygrad.helpers.diskcache_clear()` 来解决初始错误，参考了之前的 [Discord 消息](https://discord.com/channels/1068976834382925865/1070745817025106080/1358259731738661006)。
   - 然而，此操作导致该用户遇到了新的 **OperationalError**：*no such table: compile_cuda_sm_75_19*。



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **HuggingFace 论文出现但未附带评论**：一位成员在 #papers 频道分享了一个 [HuggingFace 论文](https://huggingface.co/papers/2504.10559) 链接。
   - 该论文对频道讨论的重要性目前尚不明确。
- **论文的相关性仍是一个谜**：发布该论文的用户并未明确说明所链接的 HuggingFace 论文的重要性。
   - 需要进一步调查以确定它是否与最近的训练运行有关。

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **o4-mini 亮相**：**o4-mini** 现已在 Windsurf 中可用，**o4-mini-medium** 和 **o4-mini-high** 模型将在 **4月16日至21日** 期间对所有 Windsurf 方案免费开放。
   - 查看 [公告并关注 Windsurf 的社交媒体](https://x.com/windsurf_ai/status/1911833698825286142)。
- **Windsurf 开启新频道**：Discord 上开设了一个新频道 <#1362171834191319140>，用于讨论新版本的发布。
   - 这是为了今天发布的新版本准备的。
- **JetBrains 适配 Windsurf**：今日最新版本的更新日志可在 [Windsurf.com](https://windsurf.com/changelog/jetbrains) 查看。
   - 团队已开设新频道进行讨论 <#1362171834191319140>。

---

**MLOps @Chipro Discord** 没有新消息。如果该服务器长时间没有动静，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该服务器长时间没有动静，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器长时间没有动静，请告知我们，我们将将其移除。

---

# 第二部分：各频道详细摘要与链接


{% if medium == 'web' %}

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1361783385613275246)** (1266 条消息🔥🔥🔥): 

> `OpenAI 的 o3 和 o4 mini 模型、Codex CLI 实验、GeoGuessr 性能、Qwen 3 模型、Tool Use` 

- **OpenAI 发布轻量级编程 Agent：Codex CLI**：OpenAI 推出了 **Codex CLI**，这是一款旨在最大化 **o3** 和 **o4-mini** 等模型推理能力的轻量级编程 Agent，即将支持 **GPT-4** 等其他 API 模型，详见其 [system card](https://openai.com/index/o3-o4-mini-system-card/)。
   - 它直接在用户电脑上运行，据一位成员称，它可能使用 **tool calling** 进行暴力推理，例如用于回答 [geoguessr.com](https://www.geoguessr.com/) 上的问题。
- **o3 和 o4 Mini 性能表现亮眼**：成员们测试了 **OpenAI 的 o3** 和 **o4 mini** 模型，指出 **o4 mini** 在 OpenAI 的面试选择题中表现最好，一位成员发现 o3 在处理*非平凡的真实世界 PHP 任务*时表现出色，得分为 **10/10**。
   - 然而，一些 Benchmark 表明它并不总是优于其他模型，并且正如 [X](https://x.com/DeryaTR_/status/1912558350794961168) 上报道的那样，它也面临与 **o3 相同的 Alaska 问题**，尽管在 temperature 设置为 0.4 或更低时，它在推理方面表现出色。
- **OpenAI 考虑收购 Windsurf**：围绕 [Bloomberg](https://www.bloomberg.com/news/articles/2025-04-16/openai-said-to-be-in-talks-to-buy-windsurf-for-about-3-billion) 的报道展开了讨论，称 **OpenAI** 正在洽谈以约 **30 亿美元**的价格收购 **Windsurf**，这引发了关于 OpenAI 是否应该自行开发此类工具的辩论。
   - 这将为 OpenAI 推动 **Windsurf** 提供更多动力，尽管一些成员更喜欢 **Cursor**；一位用户表示，Gemini 在 [Roblox](https://www.youtube.com/watch?v=jaitqSU2HIA) 中使用的*有限状态机路径规划*是这种集成如何产生益处的一个例子。
- **深入探讨 DeepSeek-R1 参数设置**：成员们讨论了 **DeepSeek-R1** 的配置，引用了 [GitHub readme](https://github.com/deepseek-ai/DeepSeek-R1?tab=readme-ov-file#usage-recommendations)，强调将 temperature 设置在 **0.5-0.7** 之间，避免使用 system prompts，并在数学问题中包含*逐步推理 (reason step by step)* 的指令。
   - 成员们对某些预览模型的性能和引用来源的能力表示了进一步的热情，同时也对来源幻觉表示担忧，一位成员总结道：*距离 AGI 还有一段路要走*。
- **o3 的 Tool Use 为新 Benchmark 铺平道路**：成员们强调了 **o3** 模型的 tool use 能力，例如 [图像推理缩放功能](https://xcancel.com/emollick/status/1912597487287705965)，一位成员表示 Arena 中的 *tool use 尚未推出*。
   - 这引发了关于创建 Benchmark 的对话，特别是与 GeoGuessr 相关的 Benchmark，使用新的测试框架或批量测试，尽管一位成员指出这可能会非常昂贵。

---

### **Manus.im Discord ▷ #[showcase](https://discord.com/channels/1348819876348825620/1348823595505156137/1361942714693718189)** (2 条消息): 

> `数据处理网站，数据质量讨论` 


- **网站因实用的数据处理能力受到赞誉**：一位成员称赞了某个网站的**实用数据处理**能力。
   - 该成员使用了“大拇指”表情符号，这可能表明其满意度很高。
- **数据质量讨论**：成员们正积极参与高质量的数据讨论。
   - 讨论涉及数据处理技术以及潜在相关的数据质量问题。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1361786181112566052)** (889 条消息🔥🔥🔥): 

> `Manus 积分，Kling 图像生成，社区建设，Copilot，AI 伦理` 


- **Manus 积分使用情况受到关注**：用户对 **Manus 积分消耗**表示担忧，一位用户表示他们支付了 **3900 积分**，两周后仅剩 **500 积分**，质疑其对非 AI 专家的价值。
   - 另一位用户提到在同一时间内花费了近 **20k 积分**，认为这关乎 ROI，并强调了 Manus 与其他 LLM 订阅相比的**强大 (POWERFUL)** 能力。
- **Kling 疯狂的图像生成能力**：成员们对 **Kling 的图像生成能力**印象深刻，一位成员在注册后形容 Kling 是“邪恶的 (diabolical)”，称其为“游戏规则改变者 (game changer)”。
   - 另一位成员提到 **Kling 1.6** 已经发布，其能力简直令人惊叹。
- **社区行为引发辩论**：在一次激烈的交流后，成员们就社区礼仪进行了长时间的讨论，涉及提供帮助与鼓励自力更生之间的平衡。
   - 在一名用户被封禁后，一些社区成员对社区中表现出的缺乏帮助和尊重表示担忧，而另一些人则辩护称需要自主学习，而不是依赖“施舍”。
- **Copilot 势头正盛！**：成员们正在讨论 **Copilot** 在 AI 领域带来重大变革的潜力，Pro 版本有潜力处理非常出色的工作并执行复杂任务。
   - 它可以创作“像样的艺术品”和其他复杂任务，表现非常强悍。
- **AI 危险与虚假信息**：成员们对 **AI 被用于有害目的**表示担忧，特别是在 VR 背景下，正如一位成员所说——“我可以想象这被用于非常糟糕的事情”。
   - 对话转向了版权侵权和生成 Deepfakes 的阴暗面，同时仍在试图寻找规避方法。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1361778484875235450)** (664 条消息🔥🔥🔥): 

> `Aider 编程笑话，违反 ToS 讨论，上下文压缩技术，Gemini 2.5 Pro 限制，OpenAI o3 和 o4 Mini 发布` 


- **Aider 的 Commit 问题激发了程序员的幽默感**：开发者们分享了关于 Aider 的笑话，调侃它在编程辅助过程中倾向于“过早 commit”以及“合并利益冲突 (merge conflict of interest)”。
   - 一个犀利的讽刺将 Aider 的帮助比作“重写你的仓库，就像它刚经历了一场糟糕的离婚”，而另一个暗黑幽默则暗示使用它会导致 `git blame` 只显示“为什么？”。
- **违反 ToS 的封号行为**：成员们讨论了违反服务条款 (**ToS**) 的后果，一位用户声称已经“违反 ToS 3 个月而未被封号”。
   - 讨论中提出了对潜在封号行为的担忧，以及遵守平台规则的重要性。
- **Gemini 2.5 Pro 的上下文窗口受限？**：有说法称 **Gemini 2.5 Pro** 的上下文窗口缩减至 **250K**，尽管[官方文档](https://ai.google.dev/gemini-api/docs/models#gemini-2.5-pro-preview-03-25)仍标注为 **1M**。
   - 一位成员指出，事实标准应始终以 GCP 控制台为准。
- **OpenAI 发布 o3 和 o4 Mini 模型**：OpenAI 推出了 **o3** 和 **o4-mini**，这些模型已在 API 和模型选择器中可用，取代了 **o1**、**o3-mini** 和 **o3-mini-high**。
   - [官方公告](https://openai.com/index/introducing-o3-and-o4-mini/)指出，**o3-pro** 预计将在几周内发布，并提供完整的工具支持，当前的 Pro 用户仍可访问 **o1-pro**。
- **用户就模型偏好和基准测试产生分歧**：成员们辩论了不同模型的优劣，一些人更倾向于用 **GPT 4.1** 处理特定任务，而另一些人认为 **Gemini 2.5 Pro** 更胜一筹。
   - 个人经验各不相同，凸显了模型性能取决于使用场景和上下文的主观性质。


  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1361783359604396344)** (27 messages🔥): 

> `Gemini structured output, Aider token handling, Aider Color Customization, Context-caching for Gemini 2.5 Pro, Aider Interruptions and File Additions` 


- **Gemini 采用结构化输出以提高编辑遵循度**：一位成员想知道 **Gemini's structured output** 功能在作为编辑模型使用时，是否能提高对编辑指令的遵循度。
- **Token 探讨：Aider 对 Token 的处理方式**：一位成员询问 **Aider** 如何处理 Token，并指出“thoughts”是可解析的 Token，但在 Aider 中似乎没有显示；另一位成员建议使用 `--verbose --no-pretty` 来查看它们。
   - 此外还讨论了 Aider 与 **GPT-3.5** 或 **O1/3** 等模型在 Token 处理上的区别，后者可以设置推理力度（reasoning effort）。
- **Aider 开启暗黑模式：支持颜色自定义**：一位用户寻求关于如何更改 Aider 中搜索和块高亮颜色的建议，因为默认的白色会导致眼睛疲劳；另一位成员提供了 [输出设置文档](https://aider.chat/docs/config/options.html#output-settings) 的链接。
- **Gemini 2.5 Pro 的 Context-caching：Vertex AI 的愿景？**：一位成员询问通过 **Vertex AI** 为 **Gemini 2.5 Pro** 开启 **context-caching** 的可能性，认为这可以显著降低成本，并附上了 [Gemini 2.5 Pro 文档](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-pro) 的链接。
- **Aider 文件添加带来的困扰**：一位成员报告了一个令人沮丧的问题，即 Aider 的工作流经常被添加文件的请求打断，导致重新发送上下文和重新编辑；他们在 [这篇 Discord 帖子](https://discord.com/channels/1131200896827654144/1345359268605595698) 中记录了该问题。


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1361823124722811073)** (6 messages): 

> `Building Agents, Stack evolution, Reasonable take on new models, OpenAI Codex` 


- **打造聊天机器人：构建一个 Agent**：一位成员分享了关于 [如何构建一个 Agent](https://ampcode.com/how-to-build-an-agent) 的链接，提供了对 Agent 开发的见解。
   - 他们还分享了一个关于同一话题的有趣观点的 [链接](https://x.com/amir/status/1912179662303957105)。
- **聊天机器人驱动的技术栈演进**：一位成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=FPTlP6Adefo)，详细分析了他们的技术栈是如何发生变化的。
   - 附带的图片进一步展示了技术栈的演进，提供了变化的直观表示。
- **模型：一个理性的看法**：一位成员分享了一个 [YouTube 链接](https://www.youtube.com/watch?v=3aRRYQEb99s)，对 *新模型提出了理性的看法*。
   - 他们有兴趣就该话题进行讨论并获得更细致的视角。
- **发现 OpenAI Codex 仓库**：一位成员重点介绍了 [OpenAI Codex 仓库](https://github.com/openai/codex)。
   - 该成员添加了一个“眼睛”表情符号，表示对该工具的关注。


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1362126953192231083)** (10 messages🔥): 

> `OpenAI o3, OpenAI o4-mini, Activity chart filtering, Chatroom SVG previews, Terms of service update` 


- **o3 登场，推理能力优化**：OpenAI 的 **o3** 模型现已上线，具备 **200K token** 上下文长度，价格为输入：**$10.00/M tokens** | 输出：**$40.00/M tokens** —— 需要 BYOK，最适合需要强大推理和高级工具调用的复杂任务，可在 [OpenRouter](https://openrouter.ai/openai/o3) 访问。
- **o4-mini 出现，低成本推理**：**OpenAI o4-mini** 模型提供 **200K token** 上下文长度，价格为输入：**$1.10/M tokens** | 输出：**$4.40/M tokens**，是受益于快速推理的成本效益型、高吞吐任务的理想选择，可在 [OpenRouter](https://openrouter.ai/openai/o4-mini) 访问。
- **活动图表获得细粒度过滤器**：[活动页面](https://openrouter.ai/activity) 现在除了表格过滤外，还支持图表过滤。
- **SVG 预览上线，支持聊天内查看**：用户现在可以在聊天室环境中直接**内联预览 SVG**。
- **服务条款更新已发布**：为了反映公司发展的需求，服务条款和隐私政策已更新以提高清晰度，没有重大变更，详见 [OpenRouter 隐私页面](https://openrouter.ai/privacy)。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1361780568769171657)** (560 条消息🔥🔥🔥): 

> `Gemini 2.5 Pro 速率限制、OpenRouter 隐私政策担忧、OpenAI O3 和 O4-mini 模型、DeepSeek 模型 R3 和 R4 发布、O4 Mini 图像识别问题` 


- **免费版 Gemini 2.5 Pro 速率限制严格**：用户讨论了免费层级 **Gemini 2.5 Pro** 的速率限制，指出其限制较小，为**每天 80 条消息**，如果没有 **$10 余额**，则进一步降至 **50 条**，同时 Google 也会施加自身的速率限制。
   - 一位用户表达了沮丧，称由于 5% 的充值手续费，他们需要额外支付 **$0.35** 才能达到提高速率限制所需的最低 **$10** 要求。
- **隐私政策引发恐慌，OpenRouter 表示“没那么严重”**：OpenRouter [隐私政策](https://openrouter.ai/privacy)的一次更新引发了关注，因为它似乎会记录 LLM 的输入，其中一行写道：*“您输入到服务中的任何包含个人数据的文本或数据（‘输入’）也将被我们收集”*。
   - OpenRouter 代表表示：*“我们可以改进这里的措辞清晰度，我们默认仍然不存储您的输入或输出”*，并承诺很快会澄清条款。一位成员调侃道，*“每个初创公司在积累用户资金时都会变成银行”*。
- **OpenAI 的 O3 和 O4-mini 登陆 OpenRouter**：**OpenAI 的 O3 和 O4-mini** 模型即将推出，[定价详情已分享](https://discord.com/channels/1091220969173028894/1195014798837043240/1362112075278581791)。然而，访问 O3 模型需要组织验证，并可能需要上传身份证件。
   - 成员们讨论了 O3 模型是否“值得”，或者是否应该等待即将发布的 **DeepSeek 模型**。也有人对 SVG 生成器和更新后模型的定价结构给出了正面评价，尽管很快就出现了关于缓存存在 Bug 的早期报告。
- **DeepSeek 模型 R3 和 R4 发布热度来袭！**：传闻指出 **DeepSeek 的 R3 和 R4 模型** 预定于明天发布。一位用户表示希望在这些模型发布时 *“让大家都忘了 o3”*。
   - 一位用户称 *“DeepSeek 只是便宜，实际效果并没那么好”*，另一位用户反问道 *“便宜难道是件坏事吗？”*
- **OpenRouter 修复 O4 Mini 图像识别问题**：用户报告了 OpenRouter 在实现 **OpenAI O4 Mini** 模型时的问题，特别是在图像识别方面。一位用户报告了一个荒诞的结果，将“沙漠图片”识别成了“斯温顿机车厂（Swindon Locomotive Works）”。
   - 一位 OpenRouter 代表[确认](https://discord.com/channels/1091220969173028894/1195014798837043240/1362138869734678559) *“图像输入现在已修复”*，同时也指出 *“推理摘要仅通过 responses API 提供（我们目前尚未采用），但很快就会支持”*。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1361821514583904297)** (3 条消息): 

> `ChatGPT 图像库、OpenAI 直播、OpenAI o3 和 o4-mini` 


- **ChatGPT 获得图像库**：OpenAI 正在向移动端和 [chatgpt.com](https://chatgpt.com) 上的所有 **Free、Plus 和 Pro 用户** 推出一个新的 **ChatGPT 图像创作** 库。
- **OpenAI 将举行直播**：OpenAI 宣布将举行一场直播，链接见[此处](https://openai.com/live)。
- **o3 和 o4-mini 来临**：OpenAI 宣布他们将在直播期间展示 **o3** 和 **o4-mini**。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1361783769639682228)** (386 messages🔥🔥): 

> `LLM Hallucinations, Testing O3 vs. O4 Models, Data Integrity in API Calls, AI's Role in Content Creation, Gemini 2.5 Pro vs. OpenAI Models` 


- **讨论语言模型中的 Disburse/Disperse 拼写错误 Bug**：成员们讨论了一个 **ChatGPT** 使用了音近词错误并将其称为拼写错误的案例，一位成员开玩笑说*它还在学习如何推卸责任（victim blame）*，很快就会说是用户打错了。
   - 用户们深入探讨了 **disburse/disperse** 错误，并根据该拼写错误生成了图像。
- **来自 Gemini 的 Veo 2 视频令人惊恐**：一位成员分享了 [由 **Veo 2** 生成的视频](https://cdn.discordapp.com/attachments/998381918976479273/1361827183978614937/Maltese_Terrier.mp4?ex=68017d66&is=68002be6&hm=6b8296d4bdb97dd70940575f027b27733bbd51f46947057f7161c72e28c45661&)，促使其他人评论说*那个舌头把我吓坏了*。
   - 其他人讨论了 **Gemini** 系列的使用场景，许多人更喜欢它的创意写作和记忆能力。
- **O3 一次性完成康威生命游戏（Conway's Game of Life）**：**O3** 在 **4 分钟**内编写了康威生命游戏的代码，并首次尝试就编译运行成功；而几个月前 **O3 mini high** 完成同样任务花了 8 分钟且带有 Bug。
   - 成员们讨论了这些代码改进的意义，以及 **O3** 为复杂应用生成代码和库的能力。
- **用户报告 O3 和 O4 Mini 存在幻觉**：用户报告在使用 **O4-mini** 和 **O3** 时**幻觉（hallucinations）**增多，有人指出它会编造看似可信但错误的信息。
   - 一位用户指出*模型“想要”给出回应，因为那是它的目的。*
- **API 配置调试**：成员们使用 API 测试了 **O4-mini**，发现它会编造商业地址，且对自定义搜索解决方案反应不佳。
   - 几位成员一起调试了配置，并讨论了 **OpenAI** 是否指示推理模型过于信任互联网来源。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1362019477872709673)** (24 messages🔥): 

> `GPT-4.1-batch availability, GPT can make hands, Models accessing URLs, Deleting pictures from the library, GPT-4 retirement and custom GPTs` 


- **GPT-4.1-batch 缺失，引发困扰**：成员们报告 **gpt-4.1-2025-04-14-batch** 模型通过 API 无法使用，即使是启用了 **gpt-4.1** 的 Tier 4 用户也是如此。
   - 一位成员展示了他们在 API 调用中使用 `model: "gpt-4.1"` 的代码片段，但仍然收到错误；另一位成员建议查看 [限制页面（limits page）](https://platform.openai.com/settings/organization/limits) 以获取特定账户的详细信息。
- **GPT 会画手了，引发生存危机**：一位成员开玩笑地质疑，既然 **GPT** 现在能生成手部了，*我们是不是完蛋了（are we cooked）*，并在消息后附带了一个 `questionmarks` 表情。
- **API 模型在网页浏览方面表现不佳**：用户报告 **4o-mini** 在给出 URL 时无法访问外部链接，即使在 Playground 中启用了网页搜索也是如此。
- **清理库中的照片，现在可行了**：一位成员寻求关于从库中删除图片的帮助，另一位用户提供了 [ChatGPT 图像库帮助文章](https://help.openai.com/en/articles/11084440-chatgpt-image-library) 的链接。
- **GPT-4 停用计划影响对话**：一位成员询问在 **GPT-4** 停用后，自定义 **GPTs** 及相关对话的命运。
   - 另一位成员猜测现有的聊天可以切换到 **GPT-4o** 继续，同时建议联系 [help.openai.com 的支持聊天](https://help.openai.com/) 以获取确切详情。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1361905141820690654)** (2 messages): 

> `Image Prompts, VS bag, Kohls` 


- **用户请求争议性图像的 Prompt**：一位成员请求一个图像 Prompt，内容是*提着 **VS(?) 购物袋**在 **Kohls**（应该是？）前奔跑，我看到有人把它用在一些公众人物身上*。
   - 另一位成员要求他们澄清具体在请求什么。
- **请求进一步说明**：另一位成员要求该用户澄清该 Prompt 请求背后的意图。
   - 这表明了对该 Prompt 目的及潜在滥用风险的担忧或疑虑。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1361905141820690654)** (2 messages): 

> `Image Prompt, VS Bag, Kohls` 


- **关于 VS 购物袋图像提示词的咨询**：一位成员询问了一个涉及*提着 **VS bag** 在 **Kohls** 商店前奔跑*的图像提示词，并提到了其在公众人物身上的应用。
- **请求澄清**：另一位成员就关于图像提示词的初步咨询请求澄清。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1361786025231520004)** (374 messages🔥🔥): 

> `Token Calculation in Realtime, Gemini File Reading, Cursor Agent Mode Issues, GPT 4.1 vs Claude 3.7 vs Gemini, MongoDB vs Firebase/Supabase` 


- **Token 计算困扰：请求实时 Token 计算**：一位用户请求在编辑器内实时查看 [token calculation in realtime](https://platform.openai.com/tokenizer)，或者至少能频繁更新，因为他们目前需要在网站上监控 Token 使用情况，这会非常有用。
- **Gemini 的一瞥：Gemini 真的读文件了吗？**：一位用户质疑 **Gemini** 在使用 `thing` 功能并声称读取文件时是否真的读取了文件，并附上了[截图供参考](https://cdn.discordapp.com/attachments/1074847527708393565/1361790599925202994/image.png?ex=68015b53&is=680009d3&hm=21dc0afe4ca481e7282f9721f59324770aed88ff3c6f1f76c86574cb9c595db7)。
- **Agent 行为烦恼：终端命令故障**：多位用户报告了 **Agent Mode** 中的一个问题：第一个终端命令无需干预即可运行完成，但随后的命令需要手动取消，这是一个长期存在的 Bug。
- **模型沉思：GPT 4.1 的精准提示**：用户对比了 **GPT 4.1**、**Claude 3.7** 和 **Gemini**，一位用户发现 **GPT 4.1** 在遵循提示词方面非常严格，而 **Claude 3.7** 倾向于做得比要求的多，**Gemini** 则在两者之间取得了平衡。
- **昭昭天命：通过 Manifests 快速填写表单**：用户建议增加一项新功能，允许使用 manifests 批量输入预设信息，以便轻松复制账户和服务。
   - 他们指出这将极大地协助 ASI/AGI 集群部署，并表示：*我们需要 ASI-Godsend 尽快实现，而这就是轻松帮助实现它的方法*。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1361782348626133293)** (243 messages🔥🔥): 

> `Qwen2.5-VL-7B and Qwen2.5-VL-32B, step size in practice, tax evasion processes, SFT dataset with chain of thought, training dataset to rizz up the huzz` 


- **同行评估 Qwen2.5-VL 模型吞吐量！**：一位成员请求在 **L40** 上通过 vLLM 运行 **Qwen2.5-VL-7B** 和 **Qwen2.5-VL-32B** 的 **Unsloth Dynamic 4-bit quant** 的吞吐量估算，并询问 vLLM 是否支持视觉模型。
- **因 CoT 选择 DeepSeek Distill**：建议在 **SFT** 中使用 **DeepSeek Distill** 模型，因为它具备现有的 **chain of thought (CoT)** 能力；根据 **DeepSeek** 的论文，使用像 **Qwen2.5 7B** 这样的基础模型也是可能的，但不够直接。
- **用训练数据集提升魅力 (Rizzing it up)！**：一位成员请求一个用于 "rizz up the huzz" 的训练数据集，另一位成员提供了 [Hugging Face 上的 Rizz-Dataset](https://huggingface.co/datasets/Shaheer-ipynb/Rizz-Dataset) 链接。
- **Gemma 微调后保存 GGUF 失败**：一位成员在 Colab 上对 **gemma3-4b** 进行微调后，在保存 **GGUF** 时遇到问题，出现了 "config.json does not exist in folder" 错误；另一位成员回复称他们正在调查该问题，并引用了 [github.com/unslothai/unsloth/issues/2355](https://github.com/unslothai/unsloth/issues/2355)。
- **Llama 4 微调将于本周推出**：针对关于 **Llama 4** 微调支持发布时间表的问题，一位成员表示将于*本周*发布。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1361847888363261962)** (38 messages🔥): 

> `Gemini 2.5 Pro API, Thinking Content, Cursor's Implementation, EmailJS API Abuse, Phishing Website Takedown` 


- ****Gemini 2.5 Pro** 的隐藏思考**: 成员们讨论了 **Gemini 2.5 Pro API** 是否返回“思考内容（thinking content）”，一位成员指出 [官方文档](https://ai.google.dev/gemini-api/docs/thinking) 表明 *思考过程（thinking process）* 不作为 **API** 输出的一部分提供。
   - 尽管如此，**API** 在响应中计算了思考 Token，虽然 `thought` 参数被设置为 `None`；另一位成员建议这可能是为了防止蒸馏（distillation），或者是由于模型在思考过程中会产生“不良”内容。
- **Cursor 的 **Gemini Pro** 秘诀**: 成员们猜测 **Cursor** 是如何在突破 **Google API** 限制的情况下包含“思考内容”的。
   - 猜测包括：与 **Google** 达成了潜在协议、在自家基础设施上部署了模型，或者使用了某种生成虚假思考过程的方法（例如发送多个 API 请求）。
- **EmailJS API 钓鱼网站拆除**: 一位成员在钓鱼网站的源代码中发现了 **EmailJS API** 密钥，并考虑通过发送大量请求来破坏其运行。
   - 最终，该成员决定不滥用 **API**，而是选择向 **EmailJS** 举报该钓鱼网站和密钥；他们承认滥用 API 可能是违法的。
- **合成数据生成与 OpenRouter 潜力**: 成员们表示希望找到使用 **OpenRouter** 生成合成数据的诈骗者。
   - 有人开玩笑说，如果抓到诈骗者，就使用其泄露的密钥来生成合成数据，但目前还没有人有这种运气。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1361929143989829814)** (32 messages🔥): 

> `Tool/Function Calling, Fine-tuning Llama 3.1 8B, Fine-tuned models & Pipecat, Fine-tune Qwen2.5-VL on Video Data, Quantised deepseeks outside of llama cpp` 


- **用户寻求 Llama 3.1 8B 工具/函数调用（Tool/Function Calling）的帮助**: 一位成员正在寻求关于在多轮对话数据集上微调 **Llama 3.1 8B** 以实现工具调用的数据集格式帮助，其助手响应示例格式为 `[LLM Response Text ]{"parameter_name": "parameter_value"}`。
   - 他们提到自己目前*陷入困境*，无法在 GitHub Issues 上找到可靠的信息。
- **Unsloth notebook 的 Llama 模型微调后无效**: 一位成员运行了 Unsloth 的 **Llama 模型 notebook** 并完成了微调，但模型的输出显示 Ground Truth 与微调模型之间*没有任何相似性*。
   - 所使用的问题是关于天文学家如何确定天体发射光线的原始波长，而模型的回答却是关于**多普勒效应（Doppler effect）**。
- **Phi4-mini 微调错误**: 一位成员在微调 **phi4-mini** 时遇到了 `ZeroDivisionError`，提示*数据集中所有标签均为 -100*。
   - 该成员使用的是与 **llama3.1** 相同的数据集，这表明可能与 **phi-4** 的模板不兼容。
- **OOM 问题困扰 Gemma3 用户**: 一位用户报告了在使用 **Gemma3** 时出现的显存溢出（**OOM**）问题，并链接到了一个 [相关的 GitHub issue](https://github.com/unslothai/unsloth/issues/2366)。
   - 未提供更多细节。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1361837370449727488)** (6 messages): 

> `DeepSeek-V3 Multihead Latent Attention, LLM performance penalties, Memory bandwidth bottleneck` 


- **DeepSeek-V3 的 Latent Attention 存在隐藏成本**: 一位成员正在撰写关于 **DeepSeek-V3** 的*多头潜在注意力（Multihead Latent Attention, MLA）*的文章，发现尽管 Head Size 为 **128**，但注意力计算发生在 **512 维空间**中，这使其成本增加了 **4 倍**。
   - 考虑到 **DeepSeek** 对效率的专注和已报告的训练成本，他们对此感到惊讶，认为这是一个可能被忽视的隐藏细节。
- **LLM 性能惩罚是常态吗？**: 一位成员指出，此类性能特性对于 **LLM** 来说可能是正常的，不同的模型具有不同的优缺点。
   - 原作者认为，**DeepSeek** 的架构专为效率设计，这使得这种性能惩罚更加引人注目。
- **内存带宽是主要瓶颈**: 一位成员建议，当**内存带宽（memory bandwidth）**是主要瓶颈时，增加的计算成本并不是问题。
   - 这种观点认为，如果 **MLA** 节省了足够的内存来抵消增加的计算量，那么这种权衡是可以接受的。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1361887923699585204)** (264 messages🔥🔥): 

> `Prompt Design 讨论，ChatGPT 中的符号递归，AI 生成的垃圾内容，KYC 作为人类身份验证` 


- **请求 Prompt Design 协助与指导**：一位英语水平有限的新成员寻求 **prompt design** 方面的帮助，询问有用的资源或网站。
   - 有人指出 prompt design 并非该服务器的重点，并将其引导至外部资源。
- **“递归符号主义”引发质疑**：一位用户描述了在 **ChatGPT** 中通过无状态交互（无记忆或脚本）探索“**符号递归** (symbolic recursion) 和行为持久性”的过程。
   - 其他成员表示怀疑，对前提、术语（如 *“symbolic compression”* 和 *“symbolic patterning”*）以及缺乏指标提出质疑，认为这些语言是 **AI 生成**的，对于以研究为导向的服务器可能毫无用处。
- **对 AI 生成的垃圾内容和不良帖子的担忧**：成员们讨论了 **AI 影响**或 **AI 编写内容**日益盛行的问题，担心服务器被 bot 占领。
   - 建议包括要求加入时进行**人类身份验证** (human authentication)，并识别可疑的邀请链接模式，同时附上了一篇关于因开放网络而随时间变化的[权限设置论文](https://arxiv.org/abs/2407.14933)。
- **在幻觉背景下讨论 AI Alignment**：围绕 AI Alignment 展开了讨论，对比了“AI 尽力按照人类要求行事”的观点以及与人类心理的交互。
   - 一种观点认为 *“LLM 并没有那么聪明且会产生幻觉”*，并讨论了 **o3-mini** 和 **4o** 模型在谄媚性 (sycophancy) 和幻觉方面的差异。 
- **Worldcoin KYC 灾难引发关于 AI Bot 控制的辩论**：成员们辩论了 **KYC** 身份识别措施是否可以作为确保人类访问的机制，同时也讨论了 **Worldcoin** 的失败。
   - 一位成员讽刺地指出，某位用户的邀请链接被使用了 50 多次，*“这似乎是一个潜在的危险信号”*。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1361782584681693437)** (29 messages🔥): 

> `视网膜 OCT 成像，跨领域适用性，多模态数据方法` 


- **视网膜成像中的异构数据结构**：一位成员分享了使用视网膜 **OCT 成像**的尝试，但由于 **2D 和 3D 视图**之间尽管存在语义对应，但数据结构根本不同，因此效果不佳。
   - 他们认为这个问题类似于针对各种不同类型成像的 foundation model，并询问在数据类型之间没有明确映射的情况下处理多模态数据的通用方法，链接至 [arxiv.org/abs/2107.14795](https://arxiv.org/abs/2107.14795)。
- **跨领域效率论文引起关注**：一篇关于长上下文效率中跨领域适用性的论文 [arxiv.org/abs/2410.13166v1](https://arxiv.org/abs/2410.13166v1) 在多模态方法的讨论中被提及。
   - 该成员正在完成一个密封引擎套件，其中 **5/7 已运行，2/7 正在开发中**，重点是模块化推理控制，以便在不进行传统 fine-tuning 的情况下实现精确干预。
- **使用非对称系统预测模态**：一位成员建议使用一种从一个模态预测另一个模态的非对称系统来利用多模态数据，链接至 [arxiv.org/abs/2504.11336](https://arxiv.org/abs/2504.11336)。
   - 他们拥有视网膜不同范围、不同“模态”的 **2D 成像**，以及视网膜的 **OCT 扫描**（即可以组成完整 3D 扫描的单个 2D 切片）。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/)** (1 messages): 

erkinalp: https://x.com/PrimeIntellect/status/1912266266137764307
  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1362052288025202789)** (5 条消息): 

> `PMPP 书籍 vs Triton 教程, RTX 5090 上的 matmul, fp16 矩阵, autotune` 


- **探索 Triton：入门前先看书？**：一位新手询问在深入学习 **Triton 官方教程**之前，是否应该先学习 **PMPP 书籍**。
- **RTX 5090 的 Matmul 性能异常**：一名成员报告称，参考[官方教程代码](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#sphx-glr-getting-started-tutorials-03-matrix-multiplication-py)在 **RTX 5090** 上实现 **matmul** 时，其性能与 **RTX 4090** *大致相当*，这与预期不符。测试使用的是两个大小为 **2048x2048** 的 **fp16 矩阵**。
- **速度惊人地相似**：在乘以两个 2048x2048 的 fp16 矩阵时，RTX 5090 显示出与 **4090** 相似的速度——仅快了约 **1-2%**。
   - 有成员建议使用更大的矩阵（如 **16384 x 16384**）进行测试，并尝试使用 **autotune**。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1361833499719635125)** (17 条消息🔥): 

> `NVIDIA Nsight Compute 教程, CUDA 内存使用, CUDA Kernel 中的动态索引, NVIDIA 的 GPU Profiling 演讲` 


- **Nsight Compute 教程**：一位成员请求关于使用 [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) 优化 Kernel 的教程，并提到其 `ncu-rep` 文件中显示 *“预估加速 72%”*，但不确定如何实现。
   - 另一位成员指出 NVIDIA 有一个关于 [GPU profiling](https://youtu.be/F_BazucyCMw?si=elsIPtHVcZ95RoMB) 的服务器讲座，并建议将官方 NCU 文档作为重要参考资源。
- **CUDA 内存深度解析**：一位成员对简单的 `torch.ones((1, 1)).to("cuda")` 操作产生的高内存占用表示疑问，原本预期只占用 4 字节。
   - 解释称 **CUDA 内存使用包括多项开销**：GPU tensor、CUDA context、CUDA caching allocator 内存，以及如果 GPU 连接了显示器则还包括显示开销。
- **动态索引导致溢出 (Spillage)**：一位成员询问为什么即使 Kernel 仅使用了 40 个寄存器，编译器仍将其卸载到 local memory，并附上了[代码截图](https://cdn.discordapp.com/attachments/1189607726595194971/1361996252362965082/image.png?ex=6801721b&is=6800209b&hm=df9dce50c12a0e5098a4ec345753dbdcd4c251decc614c8b820693dbe724f276&)。
   - 问题被确定为 **动态索引 (dynamic indexing)**（具体为 `reinterpret_cast<const int4*>(x+i)[0]`），其中索引 `i` 仅在运行时已知，而寄存器无法被动态索引；该问题通过将动态索引替换为 `if/else` 语句树得以解决。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1361867522294415521)** (3 条消息): 

> `torch.compile, AOTInductor, C++, Torchscript jit` 


- **PyTorch 核心开发者安排 torch.compile 问答环节**：一位 PyTorch 核心开发者宣布了关于 GPU Mode 的 **torch.compile** 问答环节，定于太平洋标准时间 4 月 19 日周六中午 12 点；感兴趣的参与者可以通过[此表单](https://forms.gle/6zbHh66DJvsfjRLi8)提交问题并进行投票。
- **建议使用 AOTInductor 进行 C++ 接口对接**：一位成员询问在 **C++** 代码中对接 **torch.compile** 模型的理想方式，并考虑了基于服务的方法和开销。
   - 另一位成员建议在仅推理场景下使用 **AOTInductor**，因为它会生成一个可以加载并运行模型的 **model.so** 文件。
- **Torchscript jit 遭到差评**：一位成员表示 Torchscript jit 非常糟糕。


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1362080554815721555)** (1 条消息): 

> `torch.compile, PyTorch, Richard Zou` 


- **Zou 将回答 Torch.Compile 相关问题**：**PyTorch** 和 **torch.compile** 核心开发者 **Richard Zou** 将于本周六（太平洋标准时间 4 月 19 日中午 12 点）主持 **问答环节**。
   - 感兴趣的用户和开发者可以通过[此 Google 表单链接](https://forms.gle/6zbHh66DJvsfjRLi8)提交关于使用方法和内部原理的问题。
- **Torch Compile 问答**：面向 PyTorch 开发者和最终用户的 Richard Zou 问答环节将涵盖 Torch Compile 的内部原理。
   - 本次会议将深入探讨 Torch Compile 的内部机制。


  

---

### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1362020502750887996)** (1 条消息): 

> `Machine Learning, College lectures` 


- **机器学习讲座请求**：一名成员感谢另一名成员分享的链接，并请求关于 **Intro to Machine Learning**（机器学习入门）的类似内容。
   - 该成员询问是否有来自同一位 **professor/college**（教授/大学）的讲座。
- **大学讲座**：一名成员请求关于特定教授讲座的更多信息。
   - 该成员对之前分享的信息表示感谢。


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1362098941985947668)** (1 条消息): 

> `PyTorch, OSS, GPU, Systems Engineering, Code Optimization` 


- **PyTorch 职位公告：招聘多 GPU 系统工程师！**：**PyTorch** 团队正在积极[招聘系统工程师](https://www.marksaroufim.com/2025/04/16/pytorch-needs-great-systems-engineers.html)，欢迎对优化单 GPU 和多 GPU 环境代码充满热情的开发者加入。
   - 理想的候选人应具备为主要 **OSS** 项目贡献实质性 **PR** 的能力，特别欢迎职业生涯早期的个人。
- **OSS 贡献是 PyTorch 职位的关键**：**PyTorch** 系统工程师职位的一个核心要求是能够为 **Open Source Software (OSS)** 社区做出重大贡献。
   - 这可以通过向知名的 **OSS** 库提交值得关注的 Pull Request (**PR**) 来证明。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1361793681073504347)** (8 条消息🔥): 

> `GPU Mode Lecture Series, CUDA variable registers, NVCC inlining device functions, PTX vs SASS compilation` 


- **GPU Mode 系列讲座录像已上线**：一名成员询问讲座系列的情况，另一名成员分享了 [YouTube 上的录像链接](https://www.youtube.com/@GPUMODE)。
- **CUDA 变量与寄存器使用分析**：一名成员对如何将变量保留在 **CUDA** 寄存器而非局部内存（local memory）中的步骤感到困惑，特别是关于通过引用和指针进行函数参数传递的问题。
   - 另一名成员回答说，假设函数被内联（inlined），通过引用传递参数应该没有问题，使用指针也可以；他们还提到 **register spilling**（寄存器溢出）是导致使用局部内存的另一个可能原因。
- **NVCC 积极内联设备函数**：一名成员注意到某些代码中使用宏（macros）代替函数，并质疑这是否真的是出于性能考虑。
   - 另一名成员回答说，**NVCC** 会积极内联属于同一编译单元的设备函数，使用宏的效果可能更好，这取决于优化阶段（optimization passes）的顺序和实际的内联阶段。
- **PTX 与 SASS 编译的寄存器分配**：一名成员询问如何在 **PTX** 层级检查寄存器使用情况和内联行为。
   - 另一名成员回答说，寄存器分配和某些优化仅在将 **PTX** 编译为 **SASS** 时进行，因此应在 **SASS** 层级检查寄存器使用情况，可以利用 Nsight Compute 中的 Source 页面。


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1361839567615037490)** (7 条消息): 

> `PARQ in torchao, PTQ weight only quant for BMM in torchao, Precision for meta-parameters z and s, AWQ uses integer zeros` 


- **PARQ 包加入 TorchAO**：**PARQ 包** 最近被添加到 torchao，参见 [torchao/prototype/parq](https://github.com/pytorch/ao/tree/main/torchao/prototype/parq)。
   - 目前的版本仅提供**导出为量化格式**的功能。
- **TorchAO 缺少针对 BMM 的 PTQ 仅权重量化**：一名成员询问 torchao 是否支持 **ptq weight only quant for BMM**。
   - 另一名成员回答说：“我们目前还没有这个功能”。
- **元参数 `z` 和 `s` 的精度**：一名成员询问元参数 **`z`** 和 **`s`** 通常使用的精度。
   - 回复指出通常使用 **float16 / bfloat16**，但在某些情况下可能需要 **fp32 scales**。
- **AWQ 偏好整数零**：一名成员指出 **AWQ** (Activation-Aware Weight Quantization) 使用**整数零**。
   - AWQ 是一种训练后量化方法，旨在最小化将权重权量化为低精度时造成的准确率损失。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1362187797137457233)** (2 条消息): 

> `Tuberculosis sanatorium, Novelty plate` 


- **趣味车牌**：一名成员展示了一张 *Tuberculosis sanatorium*（结核病疗养院）的照片，这是一个 **novelty plate**（趣味/个性车牌）。
- **另一个趣味车牌**：只是为了用*另一个趣味内容*来填充数组。


  

---

### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1361794167688134968)** (4 messages): 

> `AMD Cloud Profiling, Cloud Vendor Tier List` 


- **AMD Cloud 宣称支持 Profiling**：一位成员提到某 **AMD Cloud** 提供内置的 Profiling、Observability 和 Monitoring 功能，尽管可能不是按需（on-demand）提供的。
   - 另一位成员询问更多细节，并威胁要制作一个 *Cloud Vendor Tier List*，以此羞辱那些不提供 Hardware Counters 的厂商。
- **Cloud Vendor Tier List 即将到来？**：一位成员开玩笑地威胁要创建一个 **Cloud Vendor Tier List**，公开羞辱那些不提供 Hardware Counters 的厂商。
   - 这是针对关于 AMD Cloud Profiling 能力以及对按需 Profiling 功能需求的讨论所做出的回应。


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1361805243427586170)** (5 messages): 

> `Liger Kernel Meetings, Liger Kernel and FSDP2 compatibility, TP+FSDP2+DDP` 


- **Liger Kernel 面临 FSDP2 融合挑战**：一位成员询问了 **Liger Kernel** 与 **FSDP2** 的兼容性，并引用了 [这个 Pull Request](https://github.com/huggingface/accelerate/pull/3394)。
   - 另一位成员表示 *技术上应该是可行的*，但他们还没有机会亲自测试。
- **TP+FSDP2+DDP 实验吸引开发者**：一位成员正忙于 **TP+FSDP2+DDP** 实验，但表示愿意协助解决 **Liger** 与 **FSDP2** 的集成问题。
   - 他们表示在 **Liger** 方面有一些经验，并完成了 **FSDP2** 的集成，可以提供帮助。


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1361796735927587010)** (1 messages): 

> `candle, metal, kernels` 


- **Candle Metal Kernel 源码共享**：一位成员分享了他们在 **Candle** 项目中关于 [`candle-metal-kernels`](https://github.com/huggingface/candle/blob/main/candle-metal-kernels/src/reduce.metal) 的工作链接。
- **Metal 后端 Reduction 优化**：分享的具体文件是 `reduce.metal`，这表明了在 **Candle** 的 **Metal** 后端中针对 **Reduction Operations** 的优化或自定义实现。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1361810437087428719)** (4 messages): 

> `GPU Access, MCP, Job Opportunity` 


- **加入波士顿团队以获取 GPU 资源**：一位成员分享了一个[工作机会](https://www.q.ai/open-position/?gh_jid=4569618101)，可以访问大量 **GPU** 来训练最酷的模型，公司位于 **波士顿**，不支持远程办公。
   - 该成员未指明职位名称，但可以推测与 **Machine Learning** 相关。
- **通过 MCP 在 VM 中运行 Computer Use**：一位成员分享了 [MCP](https://x.com/trycua/status/1910455692861071414) 的链接，用于在 **VM** 中运行 Computer Use。
   - 关于 MCP 没有提供更多细节。
- **关于 ML Academia 的 Discord 活动**：一位成员询问了参加即将举行的 **ML Academia** Discord 活动所需的凭证。
   - 该 [Discord 活动](https://discord.com/events/987824841656791130/1351966194424352789) 计划在一小时后开始。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1361780808506937637)** (25 messages🔥): 

> `Pytorch HIP/ROCm compilation issues, MI300 benchmarking errors, Popcorn CLI registration and submission issues` 


- **ROCm 中缺失 Thrust Complex 头文件？**：一位用户在 ROCm/PyTorch 环境中遇到了编译错误，具体是在构建 [Custom Kernel](https://github.com/thrust/thrust) 时缺少 `thrust/complex.h` 头文件。
   - 该用户在 `rocm/pytorch:rocm6.4_ubuntu24.04_py3.12_pytorch_release_2.6.0` 容器中工作，并指出该容器包含 PyTorch 源码。
- **MI300 基准测试受阻**：一位用户报告在尝试对 **MI300** 进行测试/基准测试时出现 **503 Service Unavailable** 错误。
   - 一位维护者确认这是一个[已知问题](https://discord.com/channels/1189498204333543425/1343759913431728179/1361828692439077007)，PR 正在处理中，并建议在此期间通过 Discord 进行提交。
- **Popcorn CLI 身份验证混乱**：一位用户在尝试通过 Popcorn CLI 注册和提交时遇到多个问题，首先是提示运行 `popcorn register`，随后遇到 **401 Unauthorized** 错误。
   - 维护者澄清说，注册过程涉及通过 Discord 或 GitHub 进行 Web 授权，并且目前通过 CLI 提交的功能暂时失效，建议用户暂时使用 Discord。
- **Popcorn CLI 加载失败**：在授权 CLI ID 后，用户遇到了 "Submission error: error decoding response body: expected value at line 1 column 1"。
   - 维护者进行了跟进，并要求用户私信（DM）用于提交的脚本。


  

---

### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1361787542697148488)** (21 messages🔥): 

> `Grayscale 排行榜更新, T4 上的 Matmul 性能, AMD FP8 MM 排行榜霸榜, A100 上的 Conv2d 性能, AMD Identity 排行榜结果` 


- **Grayscale Gauntlet: A100, L4, H100, T4 测试**：一位成员以 **2.86 ms** 获得 **A100 第 8 名**，以 **17.1 ms** 获得 **L4 第 7 名**，在 **H100** 上创下 **1741 µs** 的个人最佳成绩，并在 **T4** 上以 **17.2 ms** 获得 **第 7 名**。
   - 随后的提交显示了进一步的改进，在 **H100** 上的个人最佳成绩达到 **1590 µs**。
- **Matmul 大师课：T4 的胜利**：一位成员在 `matmul` 排行榜上以 **6.91 ms** 的成绩夺得 **T4 第 4 名**。
- **AMD FP8 MM 狂热：MI300 狂欢**：多位成员在 `amd-fp8-mm` 排行榜的 **MI300** 类别中获得第一和第二名。
   - 获胜时间范围在 **829 µs** 到 **891 µs** 之间。
- **身份危机：MI300 上的 AMD Identity**：一位成员最初在 `amd-identity` 排行榜的 **MI300** 类别中以 **21.3 µs** 获得第二名，随后以 **7.71 µs** 夺得第一名。
- **Discord 下拉菜单操作指南已发布**：一位成员被指示使用 Discord 的下拉菜单功能，因为编辑消息对排行榜命令无效。


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1361779145234841874)** (7 messages): 

> `CLI 工具发布, Discord Oauth2 问题, FP8-mm 任务` 


- **Popcorn CLI 新版本修复 Bug**：**Popcorn CLI** 的新版本已可在 [GitHub](https://github.com/gpu-mode/popcorn-cli) 下载，修复了之前的问题，用户现在可以提交新任务。
   - 开发者对身份验证期间出现的 *'this web is unsafe'*（此网站不安全）警告表示歉意，并向用户保证仅会访问 **Discord/GitHub** 用户名。
- **Discord Oauth2 依然很糟糕**：由于 **Discord Oauth2** 的问题，在授权 **CLI** 之前必须登录 **Discord** 的用户可能需要重新注册。
   - 开发者承认了这一不便，并指出由于 **Discord Oauth2** 的实现机制，这是一个无法修复的限制。
- **FP8-mm 任务现已开启**：**FP8-mm** 任务现在可以通过更新后的 **Popcorn CLI** 提交。
   - 用户可以下载 [最新版本](https://github.com/gpu-mode/popcorn-cli) 参与。


  

---


### **GPU MODE ▷ #[feature-requests-and-bugs](https://discord.com/channels/1189498204333543425/1343759913431728179/1361819595065004113)** (16 messages🔥): 

> `包含反斜杠的文件导致提交错误, Service Unavailable 错误, CLI 新版本` 


- **反斜杠导致提交错误**：一位成员报告称，任何包含 `\` 的文件都会导致 `Submission error: Server returned status 503 Service Unavailable` 错误，并附上了 [error.txt 文件](https://cdn.discordapp.com/attachments/1343759913431728179/1361826466047983787/error.txt?ex=68017cbb&is=68002b3b&hm=4511ba853c5285b6e4a1d54f6cd74ee0e6cc250b4c0ac0db445dd92b1ce87ef9&)。
- **Heroku 导致 Service Unavailable 错误**：成员们注意到任务提交正常，但 Heroku 导致了 **Service Unavailable** 错误，并链接到一篇关于 [请求超时的 Heroku Devcenter 文章](https://devcenter.heroku.com/articles/request-timeout#:~:text=The%20timeout%20value%20is%20not,processing%20request%20has%20been%20finished.)。
- **提交中的换行符破坏系统**：一位成员发现，在提交内容中添加换行符 `print('\n')` 会破坏系统，但他可以通过向 `load_inline` 传递 `verbose=False` 来解决。
   - 另一位成员宣布 **CLI 新版本** 应该会修复这些提交问题。


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1361783124916441260)** (2 messages): 

> `Zero to ASIC 课程, 硅芯片设计` 


- **通过 Zero to ASIC 设计你自己的硅芯片**：一位成员分享了一个 [链接](https://www.zerotoasiccourse.com/digital/)，该课程教授如何制作自己的 **硅芯片 (silicon chip)**。
- **参加 Zero to ASIC 课程**：该课程名为 **Zero to ASIC**，重点关注 **数字设计 (digital design)**。


  

---

### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1361788775205699887)** (87 条消息🔥🔥): 

> `AMD Developer Challenge, FP8 GEMM 细节, 容差过严, 提交文件类型` 


- **AMD 挑战赛提交内容归 AMD 所有**：一位用户指出，“所有提交内容均归 AMD 所有，且不予退还”，并提醒在 [AMD Developer Challenge 2025](https://www.datamonsters.com/amd-developer-challenge-2025) 中避免使用私有 kernel。
   - 比赛规则规定截止日期为 5 月 27 日，所有提交内容将作为公共数据集发布。
- **FP8 GEMM 测试需要特定规格**：用户发现测试 `amd-fp8-mm` 参考 kernel 需要在 `test.txt` 文件中指定 **m, n, k** 尺寸，而不仅仅是 *size* 参数。
   - 一位用户报告称，使用[置顶 PDF 文件](https://www.gpumode.com/leaderboard/399)中的 **m, n, k** 值已成功运行。
- **Triton 代码可编译为 AMD 机器码**：一位用户询问提交内容是否必须用 AMD 的 CUDA 等效语言编写，因为 Triton 在底层会编译为 CUDA。
   - 一名成员澄清说，**Triton 可以直接编译为 AMD 机器码**。
- **反量化讨论**：成员们讨论了在对 A 和 B 的 tile 进行矩阵乘法之前进行反量化的正确流程，一位成员阐述了为了性能和利用 tensor cores，在 **FP8 中执行 GEMM** 的重要性。
   - 对于内层循环，as 和 bs 的因子都是相同的，因此可以将它们移出该循环，在 **fp8** 中执行内层循环，而结果的累加必须在 **fp32** 中完成。
- **Kernelbot 提交错误与文档**：一位用户报告了在尝试提交 .cu kernel 时，由于缺少头文件导致的 kernelbot 提交错误，且文档网站返回 **404 错误**。
   - 一名成员回复了该用户，并表示现在的容差要求已不再那么严格。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1361783867597652078)** (93 条消息🔥🔥): 

> `Kling 2, 用于代码的 BM25, Grok 新增 canvas, GPT 4.1 评测, O3 与 O4 mini 发布` 


- **Kling 2 告别慢动作视频时代！**：成员们对 **Kling 2** 的发布感到兴奋，称“我们终于告别了慢动作视频生成时代”，相关推文见：[推文 1](https://x.com/jasonzada/status/1912179704607703364), [推文 2](https://x.com/mrdavids1/status/1912058953690652775), [推文 3](https://x.com/maxescu/status/1912100029549994016), [推文 4](https://x.com/pjaccetturo/status/1912050794607554574), [推文 5](https://x.com/ehuanglu/status/1912532917315858628)。
- **BM25 现在可用于代码！**：推荐了一篇关于将 **BM25** 用于代码的博客文章，参见 [Keeping it Boring and Relevant with BM25F](https://sourcegraph.com/blog/keeping-it-boring-and-relevant-with-bm25f)，以及[这条推文](https://x.com/jobergum/status/1912361130195828899)。
   - BM25 是一种词袋检索函数，它根据每个文档中出现的查询词对一组文档进行排名，而不考虑查询词本身之间的相互关系。
- **Grok 新增 Canvas**：**Grok** 添加了 canvas 功能，Jeff Dean 在苏黎世联邦理工学院 (ETH Zurich) 的演讲中也提到了这一点，参见 [Jeff Dean 的演讲](https://video.ethz.ch/speakers/d-infk/2025/spring/251-0100-00L.html) 以及相关的[推文](https://x.com/grok/status/1912318583532872166)。
- **GPT-4.1 擅长编程，但不擅长结构化输出**：成员们分享了对 **GPT-4.1** 的反馈，一位成员非常喜欢将其用于编程，但它在“结构化输出方面表现不佳”。
   - 一位成员发现它配合 **Cursor agent** 非常好用，并连续成功完成了 5 次任务，[推文见此](https://x.com/swyx/status/1912364824782336296?s=46)。
- **O3 和 o4-mini 模型正式发布！**：**OpenAI** 发布了 **O3 和 o4-mini** 模型。[Introducing O3 and O4-mini](https://openai.com/index/introducing-o3-and-o4-mini/)。
   - 现场有很多链接和讨论，一位用户报告的轶事证据显示，“o4-mini 在处理 **5000 笔交易**时，使我们财务对账 Agent 的匹配率提升了 30%”。


  

---

### **Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1362141827557364032)** (80 messages🔥🔥): 

> `Zoom 宕机，RWKV-block v7，PicoCreator QKV Transformers，内存能力与冻结状态` 


- **Zoom 经历间歇性宕机**：用户报告 [Zoom](https://zoom.us/) 出现间歇性服务中断，部分用户可以连接而其他用户不行，甚至 [Zoom 的状态页面](https://zoom.us/status) 也挂了。
   - 一位成员开玩笑说，*状态页面* 就像是 *只有在你不需要时才起作用的安全带。*
- **RWKV-block v7 Goose 详情**：成员们讨论了 [RWKV-block v7 goose](https://github.com/RWKV/RWKV-block/blob/main/rwkv_block/v7_goose/block/rwkv7_time_mix.py)，包括隐藏层维度 w_lora 的维度以及 [kernel](https://github.com/RWKV/RWKV-block/blob/main/rwkv_block/v7_goose/block/kernel/rwkv7_attn_pytorch.py)。
- **PicoCreator 发布 QKV Transformers**：成员们分享并回应了 [PicoCreator 的 QKV Transformers](https://github.com/PicoCreator/QKV-Transformers-are-RNNs) 及其在 [X](https://x.com/picocreator/status/1904250680266956903) 上的帖子。
- **探索冻结状态下的内存能力**：简要触及了模型的内存能力，特别是冻结状态（freezing state）。
   - 似乎还提到了 Anthropic。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1361786262964666449)** (145 messages🔥🔥): 

> `作者身份与 AI，AI 专利，噪声 vs 不确定性，o3 与 o4-mini，Codex 安全` 


- **AI 工具的作者所有权**：成员们讨论了 [AI 在作者身份中的角色](https://example.com/ai-authorship)，一位成员指出 *意图、方向、策划和判断都源于你*，另一位则表示当 AI 实现 AGI 时，应将其列为共同创作者。
- **AI 未来与每日 1000 项专利**：一位成员正在研究一项将 *令 Microsoft 感到惊讶* 的专利，意图开发一个能够 **每天生成数千项专利** 的流水线。
   - 讨论中对专利质量与数量的关系以及专利是否是衡量生产力的标准提出了质疑，一位成员表示 *专利不是衡量生产力的好标准，它们只是保护你的权利*。
- **LLM 在有示例时表现更差**：一位成员询问是否有理论解释为什么 **推理型 LLM** 在给定 few-shot 示例时有时表现更差，引用了 [Transformer 局限性](https://example.com/transformer_limitations) 和 **过拟合**。
   - 另一位回应称 *few-shot 在所有情况下都会使它们的表现发生变化*。
- **o3 和 o4-mini API 发布**：OpenAI 发布了 o3 和 o4-mini API，一位成员认为这是 *对 o1 pro 的重大升级*。
   - 一位成员写道 **o1** *更擅长思考问题*。
- **随机性即信号，噪声很有用**：成员们讨论了噪声和随机性在生物系统中的作用，一位指出 *在生物系统中，噪声即信号*，另一位认为 **噪声关乎随机性、随机变异性——对称性破缺、创造力和泛化**。
   - 随机性的作用还在 [神经网络的巴别图书馆式存储解决方案](https://example.com/library_of_babel) 背景下进行了讨论。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1361853991109005533)** (12 messages🔥): 

> `Ultra-Scale Playbook，CUDA 显存占用，DeepSeek Maths，多模态系列` 


- **Nanotron 的 Ultra-Scale Playbook 专家并行揭秘**：成员们开始审阅 [Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#expert_parallelism)，从专家并行（expert parallelism）及相关的 [CUDA 显存讨论](https://siboehm.com/articles/22/CUDA-MMM) 开始。
- **CUDA Kernel 吞噬 GB 级显存？**：一位成员注意到 *CUDA Kernel 通常需要 1-2 GB 的 GPU 显存*，通过运行 `import torch; torch.ones((1, 1)).to("cuda")` 并使用 `nvidia-smi` 检查得到了验证。
- **DeepSeek 进行深度数学运算**：一位成员计划审阅已在 arXiv 上发布的 [DeepSeek Maths](https://arxiv.org/abs/2402.03300)。
- **本周的多模态思考**：成员们安排了一场关于 [多模态系列](https://huggingface.co/papers/2504.10479) 的讨论。
- **Ultrascale Playbook 研读暂停**：一位成员决定停止每日审阅 **Ultrascale Playbook**，理由是虽然需要了解大模型的 GPU 布局，但目前对底层 kernel 优化缺乏兴趣。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1361848293793202352)** (10 messages🔥): 

> `CVE 贬值, DHS 资金, OpenAI Windsurf, 网络漏洞数据库` 


- **DHS 在最后时刻资助网络漏洞数据库**：据 [Reuters](https://www.reuters.com/world/us/us-agency-extends-support-last-minute-cyber-vulnerability-database-2025-04-16/) 报道，美国国土安全部 (**DHS**) 在最后时刻延长了对网络漏洞数据库的支持。
   - 这一决定扭转了最初因资金问题导致的 **CVE** 贬值趋势，凸显了该数据库对公共和私营部门的实用价值。
- **OpenAI 考虑收购 Windsurf**：据 [Yahoo Finance](https://finance.yahoo.com/news/openai-talks-buy-windsurf-3-182036520.html) 报道，OpenAI 正在洽谈收购 **Windsurf**。
- **推文讨论 CVE 的资金问题**：在 [X](https://x.com/kobeissiletter/status/1912260155351191619?s=46) 上发布了一条讨论 **CVE** 资金问题的推文。
   - 该推文质疑，如果 **DHS** 声称该数据库对私营部门有用，那么这是否应该仅仅是 **DHS** 的事情。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1361784175920939189)** (46 messages🔥): 

> `Modal 免费额度, 图像生成模型, Hugging Face 推理端点问题, AMD GPU vs NVIDIA, Agents 课程截止日期` 


- **Modal 提供免费 GPU 额度**：[Modal](https://modal.com/) 每月提供 **$30 免费额度**，无需信用卡或电话号码，可访问 **H100, A100, A10, L40s, L4 和 T4 GPU**，但可用时长因 GPU 类型而异。
- **Flux.1 Dev 非常适合开源图像生成**：对于开源图像生成，**FLUX.1 dev** 表现强劲，但至少需要 **12GB VRAM**，理想情况下为 **40GB**，而 SDXL 1.0 仍被广泛使用，并可在约 **8GB VRAM** 下运行。
   - 一位用户指出，*Illustrious, NoobAI XL, Animagine 4.0 和 Pony 等变体已经过大量训练，几乎像是独立的模型。*
- **HF 推理端点面临运行时间问题**：用户报告称，自周一以来，Hugging Face 的推理端点（如 [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)）出现了许多问题，包括**服务不可用**、**超时**、**Error 500** 和 **Error 404**。
   - 目前尚未就原因或解决方案发布官方声明。
- **AMD GPU 可能会导致兼容性问题**：一位用户询问关于使用 **AMD RX 580 GPU** 的建议，担心大多数模型/库都重点支持 Nvidia，可能导致兼容性问题和错误。
   - 该用户感叹，由于潜在的 GPU 兼容性问题，*AI 开发变得很痛苦*。
- **讨论 Agents 课程截止日期**：一位用户询问 **Agents 课程** 5 月 1 日的**截止日期**是最终日期还是会有变动。
   - 另一位用户问道：*等等，课程要结束了？什么鬼（wtf）*


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1362045337476534293)** (2 messages): 

> `酷炫项目, 图像分析` 


- **开始探索酷炫项目**：一位成员分享了一个他们正在探索的“酷炫项目”的图像，未提供更多细节。
- **图像被描述为史诗级的**：该图像随后被简单地描述为 *epic（史诗级）*。


  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1362040027051200583)** (7 messages): 

> `Grok 3 基准测试, xAI 数据中心, 核能 vs 化石燃料, 便携式微型反应堆, 中国能源生产` 


- **Grok 3 基准测试终于发布**：根据[这篇文章](https://open.substack.com/pub/commonstragedy/p/grok-3-elon-musks-ai-2-months-later)，独立基准测试显示，**Grok 3** 虽然表现不错，但与最近发布的 **Gemini**、**Claude** 和 **GPT** 相比，并没有像宣称的那样令人印象深刻。
- **xAI 建成大规模数据中心**：在六个月内，**xAI** 建成了一个顶级数据中心，并训练了一个比其前代大十倍的模型，使其与 **OpenAI**、**Google** 和 **Anthropic** 并驾齐驱。
- **核能 vs 化石燃料：辩论仍在继续**：一场关于美国应该投资新型**核能**还是依靠“清洁紧凑型煤炭和石油”来满足短期能源需求的讨论随之展开。
   - 一位成员表示，煤炭、石油和天然气是*必需品*，但*对环境的影响比核能差得多*。
- **模块化微型反应堆引发关注**：初创公司正在开发便携式/模块化**微型反应堆**，这被认为非常适合 **Google**、**Meta** 和 **Microsoft** 等科技巨头。
   - 30 年来首批大型核反应堆最近上线，但重建该行业进展缓慢，因为一位成员认为，*如果没有劳动力和基础设施，你无法立即恢复具有成本竞争力和 24/7 全天候可靠生产的能力*。
- **中国能源生产激增**：一位成员指出，预计中国的能源生产将增长三倍，而美国在同一时期的增长仅为两倍。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1361976672928141364)** (12 messages🔥): 

> `LogGPT Safari 扩展, 本地 LLM 平台, Speech-to-Speech AI, 野生动物分类器, CodeFIM 数据集` 


- ****LogGPT** 扩展登陆 Safari 商店！**：一位成员分享了他们为 Safari 开发的新 **LogGPT** 扩展，它可以将 **ChatGPT** 聊天记录下载为 JSON 格式，已在 [Apple App Store](https://apps.apple.com/us/app/loggpt/id6743342693?mt=12) 上架。
   - 源代码可在 [GitHub](https://unixwzrd.ai/projects/LogGPT/Building) 和开发者的网站上找到。
- **Speech-to-Speech 级联对话式 AI 亮相！**：一位成员在 [GitHub](https://github.com/asiff00/On-Device-Speech-to-Speech-Conversational-AI) 上分享了一个 **speech-to-speech 级联对话式 AI** 项目的链接。
   - 附带的图片显示了该应用名为 **Speech to Speech Cascade**。
- **首个 Computer Vision 模型在 Hugging Face 首次亮相**：一位成员宣布他们训练并上传了他们的第一个深度计算机视觉模型——**野生动物分类器 (Wildlife Animal Classifier)** 到 [Hugging Face Spaces](https://huggingface.co/spaces/IncreasingLoss/Wildlife_Animal_Classifier)。
   - 作者征求关于该模型的文档、演示、代码和结构的诚实反馈。
- **猫被误分类为兔子？**：野生动物分类器将一只**猫**误识别为**兔子**，因为猫不在该模型训练的类别中。
   - 训练的类别包括：'antelope', 'buffalo', 'chimpanzee', 'cow', 'deer', 'dolphin', 'elephant', 'fox', 'giantpanda', 'giraffe', 'gorilla', 'grizzlybear', 'hamster', 'hippopotamus', 'horse', 'humpbackwhale', 'leopard', 'lion', 'moose', 'otter', 'ox', 'pig', 'polarbear', 'rabbit', 'rhinoceros', 'seal', 'sheep', 'squirrel', 'tiger', 'zebra'。
- **CodeFIM 数据集发布！**：一位成员在 [Hugging Face Datasets](https://huggingface.co/datasets/Etherll/CodeFIM-Data) 上分享了 **CodeFIM 数据集**。
   - 他们表示希望该数据集会有所帮助，尽管他们自己无法用它训练模型。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1361949889243906112)** (1 messages): 

> `Stage 频道位置, 活动通知` 


- **Stage 频道将举办活动**：即将举行的活动将在 **stage 频道**举行，该频道位于 general 板块下。
   - 活动将在官方开始时间几分钟后开始。
- **活动通知详情**：活动开始后，将发出**通知**，详细说明如何加入 stage 频道。
   - 与会者应留意此通知以获取**准确指令**。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1361798668138840296)** (1 messages): 

> `LLM 聊天模板, Python Glue` 


- **LLM 聊天模板获得 Python Glue 支持**：一位成员分享了他们将 [LLM-chat-templates](https://github.com/jndiogo/LLM-chat-templates) 与 **Python Glue** 结合使用的经验。
- **需要另一个主题**：添加第二个主题以满足最低要求。


  

---

### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1362142822240878782)** (3 messages): 

> `Certification Date, Intro Docs` 


- **发现认证发布日期**：一名成员参考 [Intro docs](https://huggingface.co/learn/agents-course/en/unit0/introduction) 询问认证日期是否为 **7月1日**。
   - 另一名成员对发布日期的澄清表示感谢。
- **请求视觉确认**：一位用户请求另一位用户就认证日期提供视觉确认。
   - 第一位用户分享了相同的 [Intro docs](https://huggingface.co/learn/agents-course/en/unit0/introduction) 链接作为参考。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1361829259827609832)** (62 messages🔥🔥): 

> `Use Case Assignments, Deadlines Moved, Final Certification, Proposed Assignments` 


- **用例作业（Use Case Assignments）仍未明确**：许多成员不确定课程的**用例作业**是什么以及如何提交，对缺乏明确指令和最终挑战的建议作业表示困惑，相关内容链接至[课程认证流程](https://huggingface.co/learn/agents-course/en/unit0/introduction#certification-process)。
- **课程截止日期延长至 7月**：组织者已将课程**截止日期延长至 7月1日**（原定于 4月底），以便成员有更多时间完成作业，并将最终作业的发布移至本月底，详见[沟通进度表](https://huggingface.co/learn/agents-course/communication/next-units)。
- **最终认证仍具神秘感**：成员们正在寻求关于**最终认证**流程的澄清，询问是涉及期末考试还是构建一个 Agent，并确认在新的 7月1日截止日期前完成课程和作业是否能保证获得认证，如[导论](https://huggingface.co/learn/agents-course/en/unit0/introduction#certification-process)中所述。
- **建议作业位置不明**：课程参与者正努力寻找**“建议作业（proposed assignments）”**，特别是那些提到的完成课程所需的作业，并对这些作业如何评分以及 Unit 3 是否遗漏了作业表示担忧。
   - 成员们引用了 *“我们在课程期间会提出的用例作业之一”* 这一表述并寻找其具体位置。


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1361790472917483751)** (111 messages🔥🔥): 

> `Claude Desktop tool execution failures with large responses, MCP draw.io server availability, MCP vs AI Agents with tools, Creating an MCP server with Wolfram Language, Docker container security credentials for MCP` 


- **Claude 难以处理大型 Payload**：一位成员报告称，当响应大小较大（超过 50kb）时，**Claude desktop 无法执行工具**，但在响应较小（约 5kb）时可以正常工作。
   - 有建议认为**工具可能不支持大型 Payload**，可能需要通过 resources 来实现，因为文件通常预计会很大。
- **MCP 标准简化了 AI 工具的使用**：**MCP 是一种标准化工具如何提供给 AI agents 和 LLMs 使用的协议**，通过提供通用协议来加速创新。
   - 正如一位成员所指出的，*它实际上是一个薄封装，能够以标准方式发现任何应用中的工具*，尽管它*本可以选择通过 OpenAPI 来实现这一点*。
- **ToolRouter 简化了 MCP 身份验证和客户端创建**：**ToolRouter** 平台为创建自定义 MCP 客户端提供**安全端点**，简化了**列出和调用工具**的过程。
   - 它解决了常见问题，如 **MCP 服务器的凭据管理**，以及直接向 Claude 等客户端提供凭据的风险，在 ToolRouter 端处理身份验证。
- **Obsidian 与 MCP 集成以增强记忆和日志记录**：用户正在探索将 **Obsidian 与 MCP 集成**，将其作为 AI agents 的外部记忆，一位成员描述了 *Claude 将所有内容写入*新 Vault 的工作流。
   - 虽然一位成员因为有*更好的选择*而*放弃了将 Obsidian 作为记忆使用*，但他们指出了它在*存储日志、研究和其他易于总结的对话*方面的价值。
- **MCP 服务器安全：警示案例**：成员们讨论了 MCP 内部的**安全场景**，并链接到了 GitHub 上的 *damn-vulnerable-MCP-server* [仓库](https://github.com/harishsg993010/damn-vulnerable-MCP-server)。
   - 一位成员警告说，*并不是 MCP 作为协议存在漏洞，而是你如何为自己的用途编排 MCP*。


  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1361790112844742737)** (22 条消息🔥): 

> `MCP 双向通信, BlazeMCP, 用于 MCP 的 Orchestrator Agent` 


- ****MCP** 获得“双向通道”**：提议为 **MCP** 增加一个新扩展，以实现聊天服务之间的双向通信，允许 **AI Agents** 在 **Discord** 等平台上与用户互动，详见[这篇博客文章](https://dev.illegalagents.ai/blog/proposing-bidirectional-mcp)。
   - 其目标是让 Agent 在社交媒体上可见并能进行监听，而无需用户为每个 **MCP** 重新发明通信方法。
- ****BlazeMCP** 将 STDIO 服务器暴露在公网**：**BlazeMCP** 允许用户从本地 **stdio SSE servers** 创建公共 **SSE server**，类似于 **ngrok.com**，如[此演示视频](https://youtu.be/Upr8gInrcYg)所示，可在 [blazemcp.com](https://blazemcp.com) 访问。
   - 未来的计划包括增加身份验证并发布用于自托管的源代码，以解决在不开放端口的情况下暴露运行在远程机器上的 **MCP servers** 的需求。
- ****Orchestrator Agent** 管理 **MCP** 服务器膨胀**：正在测试一种 **Orchestrator Agent**，通过处理协调并防止工具膨胀来管理多个连接的 **MCP** 服务器。
   - Orchestrator 将每个 **MCP** 服务器视为具有有限能力的独立 Agent，确保每个任务仅加载相关的工具，从而保持工具空间的精简和聚焦，如[此附带视频](https://cdn.discordapp.com/attachments/1315696461316358175/1362114131376996522/Untitled_video_-_Made_with_Clipchamp_1_1.mp4?ex=68013723&is=67ffe5a3&hm=c9aa1b285a1ed69e113a235f69ed581b87ede12d93e5ea65c78a67562c051a4a&)所示。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1361813421401575597)** (97 条消息🔥🔥): 

> `Altman 对阵 Musk, OpenAI 社交网络, LLMs 运行社交网络, AI 订阅交易, o4-mini token 计数` 


- **Altman 与 Musk 的“斗气大赛”变成 Netflix 剧集**：成员们正津津乐道于 **Altman** 和 **Musk** 之间持续的[较量](https://www.cnbc.com/2024/04/15/openai-considering-own-social-network-to-compete-with-elon-musks-x.html)，将其比作 **Netflix** 节目。
- **噩梦场景：LLMs 运行社交网络**：一位成员推测了 **OpenAI** 使用其 **LLMs** 构建和运行社交网络的有趣/可怕的可能性。
   - 另一位成员回复说，*Altman 只会抓取更多用户数据，并随心所欲地处理这些数据*。
- **好得令人难以置信：AI 订阅交易引发怀疑**：一位成员分享了一个提供 **AI 订阅**交易的链接，价格为 **$200**，引发了对其是否为诈骗的担忧；然而，原帖发布者保证了其真实性。
   - 一位成员回应道，*太兴奋了，哈哈，当有好的优惠时我们都会有点兴奋*。
- **real.azure 报告 o4-mini 输出简短回复**：一位成员报告称 **o4-mini 输出非常简短的回复**，推测它可能针对 **token count** 进行了优化。
- **每天五个新的主要模型**：一位成员感叹道 *每天有 5 个新的主要模型，谁能看得过来？*，并列举了 **glm4**、**Granite 3.3** 和新的 **HiDream** 图像模型。
   - 另一位成员表示 *事情确实感觉在加速，哈哈*。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1361832555153981591)** (4 条消息): 

> `LLM 性能, 威胁生命的提示词` 


- **威胁生命的提示词：LLM 性能助推器？**：一位成员询问是否存在关于为什么**威胁生命的提示词**似乎能增强 **LLM 性能**的研究。
   - 另一位成员认为，LLMs 作为人类编写文本的模拟器，可能会反映人类对威胁的反应，并打趣道 *如果有人在互联网上威胁我，我会停止为他们工作*。
- **LLMs 作为人类模拟器**：一位成员建议 LLMs 是在互联网上编写文本的人类的模拟器。
   - 他们开玩笑地表示，如果有人在互联网上威胁他们，他们就会停止为对方工作。


  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1362007696156397718)** (2 messages): 

> `LLaMaFactory 指南, Qwen 1.8 微调` 


- **LLaMaFactory 指南汇编**：有成员编写了一份在 Windows 环境下不使用 CUDA 运行 **LLaMaFactory 0.9.2** 的分步指南，汇集了网络上的各种技巧，可在 [GitHub](https://github.com/hiyouga/LLaMA-Factory/discussions/7733) 上查看。
   - 该指南目前止步于从 **safetensors 转换为 GGUF**。
- **Qwen 1.8 微调细节**：有成员分享称，他们使用 Alpaca 格式的 **115 个示例**，花费了 **60 小时** 对 **Qwen 1.8** 进行微调。
   - 这是在一台配备 Dell Xeon E2124 @3.30 GHz、16 GB RAM、2 GB VRAM 的原装台式机上完成的。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1361832555153981591)** (4 messages): 

> `生命威胁提示词对 LLM 的影响，LLM 作为人类模拟器` 


- **LLM 对生存恐惧有反应？**：有成员询问是否存在相关研究，探讨为何**生命威胁提示词**似乎能提高 **LLM** 的性能。
   - 另一位成员认为这是因为 *LLM 是人类的模拟器*，并调侃说如果他们在网上受到威胁，就会*停止工作*。
- **Teknium 对对齐（alignment）的反思**：Teknium 戏称，如果 LLM 是真正的*人类模拟器*，威胁它应该会产生适得其反的效果。
   - 他表示，“如果有人在互联网上威胁我，我会停止为他们工作”。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1361783423898878073)** (33 messages🔥): 

> `LM Studio 多 LLM 使用，Gemma 3 语言翻译，LM Studio 的 NVMe SSD 速度，BitNet 问候` 


- **LM Studio 关注同时部署多个 LLM**：一位用户询问是否能同时在 **LM Studio** 中使用两个 **LLM**，具体是使用一个 **LLM** 处理通用任务，另一个作为本地 API 的专用翻译器，但这在 ChatUI 中尚无法实现。
   - 另外，用户可以通过将系统提示词（system prompt）修改为对应语言，在 **Gemma 3**、**Mistral Nemo** 和 **Cohere aya 23b** 等模型的聊天界面中实现语言翻译。
- **Gemma 3 激发内在母语者潜能**：为了使用 **Gemma 3** 获得母语级别的翻译质量，系统提示词应指示模型：“用 [语言] 写一篇全新的文章；不要翻译。”
   - 这种方法促使 **Gemma 3** 直接以目标语言生成新内容，而不是进行直接翻译，从而使其像母语者一样写作，避免翻译那些无法翻译的词汇。
- **NVMe SSD 加载模型速度极快**：用户确认使用 **NVMe SSD** 能显著加快 **LM Studio** 中的模型加载速度，观察到的速度达到了 **5.7GB/s**。
   - 一位用户强调其系统中装有三个 **NVMe SSD**，但遗憾的是，这对游戏似乎没有太大提升。
- **微软的 BitNet 触动随机性**：一位用户分享了 [Microsoft BitNet](https://github.com/microsoft/BitNet) 的链接，并思考了它对 **NeuRomancing** 的影响。
   - 该用户的评论暗示随机性有助于 **NeuRomancing**，实现了从*惊奇到敬畏*的转变。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1361784594516676618)** (71 messages🔥🔥): 

> `GPU 推理，Apple M4 Max 芯片，双 GPU 支持，Nvidia 显卡发热问题，PCIE SSD 适配器` 


- **GPU x4 通道推理已足够**：推理不需要 x16 插槽，因为 x4 通道就足够了。在使用三块 GPU 进行推理时，性能差异仅约 **14%**，[有人发布的测试](https://www.reddit.com/r/LocalLLaMA/comments/1813uxf/how_important_is_pcie_speedgenlanes_when_doing/)显示你只需要 **340mb/s**。
   - 对于挖矿，甚至 x1 就足够了。
- **关于原生 FP4 支持的讨论升温**：成员们讨论了 PyTorch 中的原生 **fp4 支持**，有人提到必须使用 CU12.8 从源码构建 nightly 版本，并且[最新的 nightly 版本已经可以使用](https://github.com/NVIDIA/TensorRT/releases)。
   - 澄清指出，PyTorch 的原生 **fp4 实现** 仍处于积极开发中，而 **fp4** 目前已通过 **TensorRT** 得到支持。
- **静默但致命：对 Nvidia 显卡进行降压**：成员们一直遇到 Nvidia 显卡发热导致风扇噪音过大的问题。
   - 一位成员建议将 **3090** 降压 **20%**，将 **4080** 降压 **10%**，以在不出现减速或崩溃的情况下减少发热和噪音。
- **双 GPU 支持？**：成员们不确定是否能让双 GPU 协同工作，因为大多数主板的 **PCI-e x16** 插槽有限。
   - 一位成员尝试让 **3060ti** 和 **1050ti** 协同工作但遇到了崩溃。他们被要求提供更多信息。


  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1361791534793625781)** (10 messages🔥): 

> `Notebook LM 与 Microsoft 文档，Google Docs 对比 OneNote，德语播客生成问题` 


- **Notebook LM 处理 Microsoft Intune 文档**：一位用户正在探索将 Notebook LM 与 [Microsoft Intune 文档](https://learn.microsoft.com/en-us/intune/)结合使用，以备考 **MD-102**、**Azure-104** 和 **DP-700** 等 **Microsoft Certifications**。
   - 另一位成员建议使用 "Discover" 功能，配合提示词 "Information on Microsoft Intune" 和站点 URL 来发现子主题，并建议将其复制粘贴到 Google Docs 中以便导入。
- **Google Docs 优于 OneNote**：一位用户对比了 [Google Docs](https://docs.google.com/) 与 [OneNote](https://www.onenote.com/)，指出 Google Docs 的优势在于：**无同步问题**、**自动大纲**以及**良好的移动端阅读体验**。
   - 该用户指出 Google Docs 的缺点是**切换文档时有延迟**且基于浏览器，并提供了一些 [Autohotkey 脚本](https://www.autohotkey.com/)来缓解这些问题。
- **德语播客生成故障？**：一位用户报告了使用 Notebook LM 生成**德语播客**时出现的问题，尽管之前很成功，但现在性能有所下降。
   - 该用户正在寻求社区的建议和技巧以恢复播客生成的质量，并分享了 [Discord 频道](https://discord.com/channels/1124402182171672732/1360560496801222838)的链接。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1361778530064666767)** (82 messages🔥🔥): 

> `播客语言支持，LaTeX 支持，批量上传，思维导图生成` 


- ****播客语言支持仍未实现****：用户对播客功能仅支持英语感到沮丧，尽管系统可以用其他语言运行，且这是**需求最高的功能**之一。
   - 一位用户表达了挫败感，表示他们*愿意为意大利语的该功能支付订阅费*，以便为他们的足球队创建内容，因为他们之前为了同样的目的订阅了 **ElevenLabs**。
- ****LaTeX 支持仍缺失，数学系学生感到不满****：数学系学生对缺乏 **LaTeX 支持** 表示不满，一位用户开玩笑说他们可以在 *30 分钟*内*开发*出这个功能。
   - 另一位用户建议，虽然 Gemini 模型可以编写 LaTeX，但问题在于如何正确显示，这导致一位用户考虑创建一个 **Chrome extension** 作为权宜之计。
- ****批量上传功能依然难寻****：用户请求能够一次性**批量上传**数百个源文件，但目前尚无法实现。
   - 一位用户建议使用 [WebSync Chrome extension](https://chromewebstore.google.com/detail/websync-full-site-importe/hjoonjdnhagnpfgifhjolheimamcafok) 作为潜在解决方案，且团队已经修复了 flash 2.0 和 2.5 Pro。
- ****思维导图限制及寻求替代方案****：一位寻求为近 **3000 篇期刊文章**创建详细思维导图的用户发现，NotebookLM 的思维导图生成仅限于一个主主题和三个子层级。
   - 他们现在正在寻求其他 **Google AI tools** 或替代方案的建议，以创建 AI 生成的详细思维导图，并考虑将 **Obsidian** 中的手动创建作为最后手段。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1362071351476879481)** (10 messages🔥): 

> `Mojo, Arch Linux, GPU 支持, Conda, 社区会议` 


- **Magic, Mojo 和 Max 在 Arch Linux 上表现出色**：一位成员高兴地报告说，**Magic**、**Mojo** 和 **Max** 在 **Arch Linux** 上开箱即用，运行完美，尽管官方文档仅提到了 **Ubuntu**。
   - 他们表示：*管他呢，试试看吧，结果令我惊讶的是，一切都运行得非常完美，哈哈*。
- **“支持”与“功能性”的区别**：一位成员澄清说，公司对产品的“支持（Support）”与产品仅仅能“运行（Working）”是不同的，并以 **Nvidia** 和 **CUDA** 为例。
   - 他们进一步解释说，*“支持”实际上意味着如果产品无法运行，我们可能会因违反合同而面临财务处罚*，这设定了一个很高的标准。
- **社区会议录像已发布**：最新的 **Mojo 社区会议**录像现已在 [YouTube](https://www.youtube.com/watch?v=lJkHv0juxUE) 上提供。
- **Conda 的隔离能力实现了更广泛的兼容性**：一位成员强调了 **Conda** 在维护隔离环境方面的强大能力，使得 **Mojo** 能够在官方测试之外的系统（如 **Arch Linux**）上运行。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1361803371543007354)** (58 条消息🔥🔥): 

> `Mojo 中的内核调用、Mojo 编译器性能、HVM/Bend 观点、性能回归测试` 


- **Mojo 考虑像 Rust/Zig 一样支持原生内核调用**：成员们讨论了 **Mojo** 是否会像 **Rust/Zig** 一样支持原生内核调用，从而可能绕过对 **C** `external_call` 的需求。
   - 有人指出，直接系统调用（syscalls）需要处理 syscall ABI 和 inline assembly，而 **Linux** 拥有稳定的 syscall table；更多信息请参见 [syscall_64.tbl](https://github.com/torvalds/linux/blob/master/arch/x86/entry/syscalls/syscall_64.tbl)。
- **Mojo 编译时间困扰性能测试**：成员们在测试过程中观察到编译时间显著过长，其中一个案例显示运行时间为 **319s**，而实际测试执行仅需 **12s**，这与 [Kelvin library](https://github.com/bgreni/Kelvin) 有关。
   - 使用 `builtin` 有助于大幅缩减编译时间，从 6 分钟降至 20 秒；示例请参见 [此 gist](https://gist.github.com/soraros/8924ed8ea70403a5d944ae5316ab3fea)。
- **Kelvin 数量计算导致编译器灾难**：一名成员发现 **Kelvin** 库中的某些操作（如 `MetersPerSecondSquared(20) * MetersPerSecondSquared(10)`）会导致极度的性能下降，可能是由于计算树的规模呈 `O(2^n)` 增长。
   - 应用更改并添加 `builtin` 注解解决了性能问题，使测试套件运行时间恢复正常，但已提交错误报告（[issue 4354](https://github.com/modular/max/issues/4354)）以调查原始行为。
- **对 HVM/Bend 硬件处理的疑虑**：成员们讨论了 **HVM/Bend**，认为这是一个有趣的想法，但由于内存和互连带宽管理的挑战，实现系统级语言的速度非常困难。
   - 虽然对数据科学可能有用，但人们对于能否通过编译器开发努力来消除大多数 FP 语言固有的开销仍持怀疑态度，[正如之前讨论的那样](https://ptb.discord.com/channels/1087530497313357884/1098713601386233997/1241668427987550209)。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1361921978117914817)** (45 条消息🔥): 

> `GPT4All 离线使用、LM Studio 作为替代方案、将书籍导入模型、GGUF 版本兼容性、GPT4All 开发状态` 


- **GPT4All 离线使用：骗局还是成功？**：一名用户报告称，尽管网站宣称支持，但在尝试加载本地 `mistral-7b-openorca.gguf2.Q4_0.gguf` 模型时，**GPT4All** 拒绝离线工作。
   - 另一名用户确认了离线使用的成功，从而引发了对模型加载过程的排查，并建议根据 [FAQ](https://github.com/nomic-ai/gpt4all/wiki/Frequently-Asked-Questions#where-are-the-default-directories-for-models-and-settings) 检查正确的模型目录。
- **LM Studio：推荐的替代方案**：在 GPT4All 无法工作时，一名用户讽刺地建议使用 **LM Studio** 作为可运行的离线替代方案。
   - 这引发了关于将书籍导入模型以及来自 [Hugging Face 上的 LM Studio 社区](https://huggingface.co/lmstudio-community) 模型推荐的讨论。
- **GGUF 版本困扰！**：人们对旧版 **GGUF 版本**（特别是 version 2）的潜在不兼容问题表示担忧，该版本可能在 2023 年左右停止工作。
   - 一名用户建议尝试更新的模型，并参考 [GPT4All GitHub 仓库](https://github.com/nomic-ai/gpt4all/blob/cd70db29edaf0f02a567f5eea94f5e52240be3e9/gpt4all-chat/metadata/models3.json#L184) 中的 `models3.json` 文件以确保兼容性。
- **GPT4All 的开发停滞了？**：用户询问 GPT4All 是否会增加 **语音和组件功能**，而另一名用户指出 GPT4All 的开发可能已暂停，并提到 Discord 上已有大约三个月没有开发者出现了。
   - 一名用户对未来表示悲观，称 *“既然一年都没有实质性的进展……所以我没抱希望”*，而另一名用户计划如果到夏天还没有更新就转向其他平台。


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1361803255541141575)** (1 条消息): 

> `答疑时间 (Office Hours)` 


- **答疑时间链接已上线！**：一名成员发布了下个月答疑时间的链接：[Discord 事件链接](https://discord.gg/AjDzfV8G?event=1361803002700370122)。
   - 声明的目标是 *“为了让大家别再来烦我”*。
- **答疑时间已公布**：下个月的答疑时间已排定。
   - 请查看提供的 Discord 链接了解详情并报名。


  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1361786919989674258)** (41 条消息🔥): 

> `验证集 PR，KV Cache 管理，配置变革，tokenizer 路径困扰，tune 命令名称冲突` 


- **验证集（Validation Set）PR 已合并，欢迎用户试用！**：引入**验证集**的 Pull Request ([#2464](https://github.com/pytorch/torchtune/pull/2464)) 已合并；鼓励用户进行测试并提供反馈。
   - 虽然有计划将其集成到其他配置中，但进一步的步骤目前处于暂停状态，等待用户反馈。
- **KV Cache 内部实现灵活性辩论**：讨论围绕是将 **KV Cache** 在模型内部管理还是像 **MLX** 那样在外部管理展开，以在推理过程中获得更大的灵活性。
   - 最终决定在内部管理，因为这能保持顶层 Transformer 块的 API 更加简洁，灵感源自 **gptfast** 的简洁性和编译（compile）兼容性。
- **通过根目录革新 Configs**：正在推动修改配置，为模型/检查点（checkpoints）定义根目录，以简化使用并方便移交给实习生。
   - 建议使用基础目录（base directory）方法，即指定一个 `base_dir`（例如 `/tmp`）并在后续配置行中使用它，从而简化流程，避免手动更改多个路径。
- **Tokenizer 路径配置亟待解决**：手动提供 Tokenizer 路径而不是从模型配置中派生的必要性被标记为一个困扰。
   - 计划对此进行修改，特别是针对每个模型，因为给定下载模型的路径，Tokenizer 路径通常是固定的。
- **"tune run" 命令导致命名空间冲突**：Torchtune 中的 `tune run` 命令与 Ray 的 `tune` 冲突，可能在环境安装期间引起混淆。
   - 建议引入别名，如 `tune` 和 `torchtune`，以缓解命名冲突。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1361817393462247445)** (3 条消息): 

> `Jerry 参加 AI 用户大会，纽约投资专业人士的 AI 解决方案，LlamaIndex 支持 o3 和 o4-mini` 


- **Jerry 在 AI 用户大会大放异彩**：LlamaIndex 创始人 **Jerry Liu** 将在本周四于旧金山及线上举行的 **AI 用户大会**上讨论构建 **AI 知识 Agent**，旨在自动化 50% 以上的运营工作。
   - 有关会议的更多信息可以在[这里](https://t.co/meQVbC1Pna)找到。
- **LlamaIndex 助力投资专业人士**：LlamaIndex 将于 5 月 29 日在曼哈顿为有兴趣构建 **AI 解决方案**的投资专业人士举办实战研讨会。
   - 直接向联合创始人兼 CEO **Jerry Liu** 学习如何将 **AI 应用于金融挑战**；注册详情请见[这里](https://t.co/2XtQBQJs2c)。
- **LlamaIndex 拥抱 OpenAI 的 o3 和 o4-mini**：**LlamaIndex** 现在通过最新的集成包提供对 **OpenAI o3 和 o4-mini** 模型的零日支持（day 0 support）。
   - 通过 `pip install -U llama-index-llms-openai` 更新至最新集成包，更多[详情请见此处](https://t.co/jOuqaVw8TA)。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1361787733944569996)** (26 messages🔥): 

> `Pinecone multiple namespaces, LlamaIndex Agents with MCP Servers, LLM.txt, Base64 PDF support, Google A2A implementation with LlamaIndex` 


- ****Pinecone 的命名空间细节需要关注****：一位成员询问如何使用 **LlamaIndex** 配合 **Pinecone** 进行跨多个命名空间的查询，并指出虽然 Pinecone 的 Python SDK 支持此功能，但 **LlamaIndex 的 Pinecone 集成** 似乎并不支持。
   - 一位成员确认当前代码假设为单个命名空间，并建议为每个命名空间创建一个 vector store，或者提交一个 Pull Request 来添加多命名空间支持。
- ****MCP 精通动力：成员们思考模型管理****：一位成员正在寻找使用 **LlamaIndex agents** 与 JSON 文件中定义的 **MCP (Model Configuration Protocol) 服务器** 进行交互的项目。
   - 另一位成员建议不要从那里开始，而是建议参考[此示例](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-mcp/examples/mcp.ipynb)将任何 MCP 端点转换为 Agent 的工具。
- ****LLM.txt：让更丰富的项目上下文长存****：一位成员建议创建一个 `llm.txt` 文件，将项目最重要的上下文放入上下文窗口中以便更容易获得帮助，甚至可以维护一个本地向量数据库进行 RAG，以避免向 LLM 重复解释 **A2A 和 MCP**。
   - 另一位成员承认由于文档详尽，很难定义“项目的核心内容”，并欢迎大家提供想法和 PR。
- ****Base64 PDF 呼唤更好的块构建****：一位成员询问 **LlamaIndex 是否支持将编码后的 base64 PDF 文件**传递给 OpenAI。
   - 另一位成员回答称目前尚不支持，需要将其作为一种内容块类型添加，并指出 OpenAI 最近新增了此功能。
- ****A2A 行动：接近完成，请再次询问****：一位成员询问关于 **Google A2A (Application-to-Application)** 与 **LlamaIndex** 的示例实现。
   - 另一位成员指向了[这个 Pull Request](https://github.com/google/A2A/pull/179)，同时也承认 **A2A** 目前还不够成熟且*没有 SDK*。


  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1361934039346577481)** (4 messages): 

> `Command A token loops, FP8 arguments, vllm Community Collaboration` 


- **Command A 容易出现 Token 级别的无限循环**：有人指出虽然其他 LLM 也会出现 Token 级别的无限循环，但 **Command A** 特别容易复现这一问题。
   - 报告该问题的成员希望他们的反馈能有所帮助，并暗示该问题在 **Command A** 中可能比其他模型更普遍。
- **FP8 参数建议**：一位成员建议了使用 `-tp 2`（2 级张量并行）运行 **FP8** 的参数：`--enable-chunked-prefill --max-num-batched-tokens=2048 --max-num-seqs=128 -tp=2 --gpu-memory-utilization=0.95 --max-model-len=128000`。
   - 这些设置旨在优化使用 **FP8** 精度和张量并行时的内存占用和吞吐量。
- **vllm 社区支持 128k+ 上下文长度**：成员们正积极与 **vllm 社区** 合作，以实现对超过 **128k** 上下文长度的优化。
   - 此次协作重点在于提高 **vllm** 框架内具有极长上下文窗口的模型性能和效率。


  

---


### **Cohere ▷ #[「🔌」api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1361850433944944721)** (1 messages): 

> `Embed-v4.0 supports 128K tokens, API support embedding more than 1 image per request, Late Chunk strategy` 


- **Embed-v4.0 将上下文窗口提升至 128K Token**：新的 **embed-v4.0** 模型现在支持 **128K Token 上下文窗口**，增强了其处理长序列的能力。
   - 这一提升允许进行更全面的文档分析，并提高在需要广泛上下文理解的任务中的表现。
- **API 升级：单次请求支持多张图像？**：一位用户建议增强 API 以支持在单个请求中嵌入多张图像，从而利用 **embed-v4.0** 中扩展的上下文窗口。
   - 这将使得能够将整个 PDF 文档作为图像进行处理，从而促进 *'Late Chunk'* 策略的实现。
- **“Late Chunk” 策略在 PDF 处理中受到关注**：讨论了 *'Late Chunk'* 策略，将其作为一种通过将 PDF 文档转换为图像并使用 API 进行嵌入的处理方法。
   - 这种方法通过利用 **embed-v4.0** 中 **128K Token 窗口** 提供的完整上下文，有可能提高文档分析的准确性和效率。


  

---

### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1362033050124288234)** (2 条消息): 

> `开源聊天界面, AI 工具化, Cohere 模型理解, 金融科技创始人` 


- **金融科技创始人进军开源 AI 工具领域**：一位退役的金融科技创始人正在开发 [Yappinator](https://github.com/invisietch/Yappinator)，这是一个用于 AI 交互的**开源类聊天界面**，基于其早期的原型 [Chatterbox](https://github.com/invisietch/Chatterbox) 构建。
   - 该创始人还为其他**自由软件项目**做贡献，并担任 **finetuner**。
- **技术栈亮点包括 Clojure 和 Kafka**：该创始人偏好的技术栈包括 **Clojure**、**C++**、**C**、**Kafka** 和 **LLMs**。
   - 该创始人希望通过加入社区来加深对 **Cohere 模型** 的理解。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1361780316125134899)** (4 条消息): 

> `MOOC 实验发布, MOOC 课程截止日期, 伯克利学生 vs. MOOC 学生, MOOC 实验 ETA` 


- **MOOC 实验首秀推迟**：**实验 (labs)** 将在接下来的**一两周内**向 MOOC 学生发布，而不是像伯克利学生那样分多个部分发布。
   - 它们的截止日期和所有其他课程作业一样，都在 **5 月底**。
- **实验 ETA**：一位成员建议网页应反映新的 **ETA**，以便学生计划预留实验时间。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1361878047178887269)** (2 条消息): 

> `可验证输出, Lean 自动形式化工具, 程序形式化验证, 自动证明生成` 


- **可验证输出提升推理能力**：一位成员建议，拥有**可验证输出**可以提供一种更好的方法来改进**推理**和**逻辑思维**。
   - 他们还提到自己是 **Lean** 的新手，这是一种依赖类型编程语言和交互式定理证明器。
- **针对非形式化证明的自动形式化工具**：一位成员询问，在给定包含业务逻辑的计算机代码（例如 Python、Solidity）或一般的非数学陈述作为输入时，是否可以使用**自动形式化工具 (auto-formalizer)** 来创建**非形式化证明/定理**。
   - 这表明了将形式化方法应用于传统数学问题之外的实际编程场景的兴趣。
- **AI 自动化证明生成**：该成员对使用 **AI** 进行**程序形式化验证**和**自动证明生成**表示了兴趣。
   - 这反映了利用 AI 简化通过形式化方法确保代码正确性和可靠性过程的愿望。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1362040779534762086)** (3 条消息): 

> `MNIST 教程错误, diskcache_clear() 修复, OperationalError` 


- **用户遇到 MNIST 教程错误**：一位新用户在 Colab **T4** 上运行 **MNIST 教程**代码时遇到错误，特别是在计算准确率和尝试反向传播时，如[附图](https://cdn.discordapp.com/attachments/1070745817025106080/1362040779362668706/Screen_Shot_2025-04-16_at_13.20.36_PM.png?ex=6800f2d3&is=67ffa153&hm=c38f645620c5ecbe1810630c16ff5672cd8a282286c7985fed6b24f391fe7066&)所示。
- **Diskcache 清理 OperationalError**：一位成员建议运行 `tinygrad.helpers.diskcache_clear()` 以尝试解决该错误，并链接到了相关的 [Discord 消息](https://discord.com/channels/1068976834382925865/1070745817025106080/1358259731738661006)。
   - 然而，用户在尝试建议的解决方案后遇到了 **OperationalError**。
- **OperationalError 持续存在**：用户报告在运行 `tinygrad.helpers.diskcache_clear()` 后出现新的 **OperationalError**：*no such table: compile_cuda_sm_75_19*。
   - 该错误发生在打印准确率时执行 `acc = (model(X_test).argmax(axis=1) == Y_test).mean()` 的过程中。


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1361970882414907433)** (2 条消息): 

> `HuggingFace 论文` 


- **HuggingFace 论文出现**：一位成员分享了 [HuggingFace 论文](https://huggingface.co/papers/2504.10559) 的链接，但未发表评论。
- **意义尚不明确**：目前尚不清楚所链接论文的具体意义，尽管它可能与最近的训练运行 (training runs) 有关。


  

---

### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1362130098215587890)** (2 条消息): 

> `o4-mini, Windsurf, Free Access, New Channel, Changelog` 


- **o4-mini 现身**：**o4-mini** 现已在 Windsurf 中可用，**o4-mini-medium** 和 **o4-mini-high** 模型将在 **4 月 16 日至 21 日**期间对所有 Windsurf 方案免费提供。
   - 查看[公告并在社交媒体上关注 Windsurf](https://x.com/windsurf_ai/status/1911833698825286142)。
- **Windsurf 开启新频道**：新频道 <#1362171834191319140> 已开启用于讨论。
   - 这是为了今天的新版本发布。
- **JetBrains 接入 Windsurf**：今日最新发布的 Changelog 可以在 [Windsurf.com](https://windsurf.com/changelog/jetbrains) 查看。
   - 团队已开启一个新频道进行讨论 <#1362171834191319140>。


  

---


---


---


{% else %}


> 完整的逐频道详情已在邮件中截断。
> 
> 如果您想查看完整详情，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}