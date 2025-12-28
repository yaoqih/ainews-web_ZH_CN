---
companies:
- artificial-analysis
- stability-ai
- adobe
- deepseek
- alibaba
date: '2025-03-25T01:43:04.934624Z'
description: '**Reve** 是一款由前 Adobe 和 Stability 成员 **Christian Cantrell**、**Taesung
  Park** 和 **Michaël Gharbi** 联手打造的新型复合 AI 模型。它目前已成为评分最高的图像生成模型，在文本渲染和排版设计方面超越了 Recraft
  和 Ideogram 等此前的顶尖模型。


  该团队强调“通过逻辑增强视觉生成模型”以及“利用先进的语言能力理解用户意图”，从而能够根据自然语言输入对视觉图像进行迭代修正。此外，**DeepSeek-V3-0324**
  和阿里巴巴的 **Qwen2.5-VL-32B-Instruct** 模型也已发布，它们在视觉任务基准测试和数学推理等方面均展现出显著的性能提升。'
id: 9e88f842-ed12-4ead-aa52-fe4d029a6454
models:
- deepseek-v3-0324
- qwen-2.5-vl-32b-instruct
- recraft
original_slug: ainews-halfmoon-is-reve-image-a-new-sota-image
people:
- christian-cantrell
- taesung-park
- michael-gharbi
title: Halfmoon 推出 Reve Image：由前 Adobe 和 Stability AI 三人组打造的全新 SOTA（最先进）图像模型。
topics:
- text-to-image
- prompt-understanding
- model-composition
- visual-generation
- language-understanding
- model-performance
- complex-prompting
- iterative-generation
---

<!-- buttondown-editor-mode: plaintext -->**Composite AI is all you need?**

> 2025年3月21日至3月24日的 AI 新闻。我们为您检查了 7 个 subreddit、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord 服务器（**227** 个频道和 **10464** 条消息）。预计节省阅读时间（以 200wpm 计算）：**1129 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

今天 [Qwen](https://news.ycombinator.com/item?id=43464068) 和 [Deepseek](https://twitter.com/_akhaliq/status/1904154585242935516) 发布了一些不错的更新，但我们将头条位置留给了一个知名度较低但雄心勃勃的新进入者。

Reve（[发音为 [ʀɛv]，源自 “rêve”](https://x.com/m_gharbi/status/1904213903384695280)）已从 [Artificial Analysis 的排行榜](https://x.com/ArtificialAnlys/status/1904188980423467472)中脱颖而出，成为评分最高的 imagegen 模型，取代了之前的 SOTA 模型 Recraft。“该模型以其令人印象深刻的文本渲染、prompt adherence 和美学表现脱颖而出。” 我们发现它的上手体验非常简单。


![image.png](https://assets.buttondown.email/images/eacb4da0-b781-47d9-b5a2-a0b230b883b5.png?w=960&fit=max)



![image.png](https://assets.buttondown.email/images/43ed5116-3b08-4b8a-abcb-f74fa99263f9.png?w=960&fit=max)


而且它在 typography 方面击败了 Ideogram：


![image.png](https://assets.buttondown.email/images/149cc977-e438-444a-b92a-098efb750d70.png?w=960&fit=max)


有趣的是，它来自前 Stability 产品副总裁 [Christian Cantrell](https://x.com/cantrell/status/1904213242567917684)、[Taesung Park](https://x.com/Taesung/status/1904220824435032528) 和 [Michaël Gharbi](https://x.com/m_gharbi/status/1904213903384695280)。他们都是 Adobe 的校友，Michael 的公告最深入地揭示了他们的实现方式：

> Reve 的使命是发明意图驱动（intent-driven）视觉创作的未来。捕捉创作意图需要机器对自然语言和其他交互有先进的理解。**将这种意图转化为引人注目的视觉效果，需要交互式系统**对它们生成的视觉世界有深刻的理解，以便它们能够**迭代地进行修正**。

[Taesung 表示赞同](https://x.com/Taesung/status/1904220827073257483)：

> 今天的 text-to-image 模型本质上就是随机的世界片段生成器。没有智能。这既是数据问题，也是 representation 问题。**我们需要为图像利用相当于完整文档的东西，但我们还没有很好的 representation。** 我们在 Reve 的使命是**用逻辑增强视觉生成模型**。作为第一步，我们专注于通过先进的语言能力理解用户意图，从而实现卓越的复杂提示词理解和文本书写能力。

没有迹象表明它是一个单一模型，而更像是模型的某种 composite。这可能就是 Christian 想在 Stability 构建但没能实现的东西。


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

以下是提供的推文中与 AI 相关的讨论摘要，已为技术受众进行分类：

**模型发布与更新，包括性能表现**

- **DeepSeek V3-0324 发布与性能**：[@_akhaliq](https://twitter.com/_akhaliq/status/1904154585242935516) 宣布 **DeepSeek-V3-0324** 在 Hugging Face 上发布，[@Teknium1](https://twitter.com/Teknium1/status/1904147049219494148) 也注意到了其发布，[@reach_vb](https://twitter.com/reach_vb/status/1904153415665517034) 强调这是一个**训练后更新（post-training update）**，具有提升下游性能的潜力。多位用户讨论了其性能和特性，包括 [@teortaxesTex](https://twitter.com/teortaxesTex/status/1904161508642168971) 认为它**可与 Sonnet 3.6 媲美**，以及 [@teortaxesTex](https://twitter.com/teortaxesTex/status/1904292164672115077) 指出在某些评估中它**超越了 DeepSeek-R1 和 Claude-3.7**。
- **Qwen 2.5-VL-32B-Instruct 发布**：[@_akhaliq](https://twitter.com/_akhaliq/status/1904242971043607002) 宣布阿里巴巴的 **Qwen2.5-VL-32B-Instruct** 在 Hugging Face 上发布，[@reach_vb](https://twitter.com/reach_vb/status/1904234593576014312) 分享了**性能基准测试**，显示其在视觉任务上击败了 Qwen 2.5 72B 和 GPT 4o Mini，并增强了数学推理和人类偏好对齐。
- **DeepSeek 模型推理服务**：[@_akhaliq](https://twitter.com/_akhaliq/status/1904231386430799938) 指出 **DeepSeek 的新模型通过 Hyperbolic Labs 在 Hugging Face 上提供服务**，[@ClementDelangue](https://twitter.com/ClementDelangue/status/1904237660237115542) 提到它也可以通过 FireworksAI 和 Hyperbolic Labs 获取。[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1904223627509465116) 表示 **Hyperbolic Labs 现在支持 DeepSeek-V3-0324 的推理服务**。
- **MLX 上的 DeepSeek V3-0324**：[@reach_vb](https://twitter.com/reach_vb/status/1904204090868900140) 报告称最新的 **DeepSeek V3-0324 在配备 512GB 内存的 M3 Ultra 上使用 mlx-lm 运行速度超过 20 toks/sec**，[@awnihannun](https://twitter.com/awnihannun/status/1904177084609827054) 也确认了这一点。
- **NVIDIA Mamba 图像骨干网络**：[@mervenoyann](https://twitter.com/mervenoyann/status/1904168637612630279) 宣布 **NVIDIA 在 Hugging Face 上发布了新的 Mamba 图像骨干网络**，提供多种尺寸和分辨率。

**Frameworks and Tools**

- **LangChain 和 LangGraph 使用案例**：多条推文强调了 LangChain 和 LangGraph 的使用案例，包括沃达丰（Vodafone）用于数据运营的 AI 助手 [@hwchase17](https://twitter.com/hwchase17/status/1904216034095333392)，Klarna 用于客户支持的 AI 助手 [@LangChainAI](https://twitter.com/LangChainAI/status/1904219446874604018)，以及一个医疗供应链 AI 系统 [@LangChainAI](https://twitter.com/LangChainAI/status/1904201544305725749)。[@hwchase17](https://twitter.com/hwchase17/status/1904247784087388252) 还提到了 LangGraph 中的上下文管理。
- **Weave-Agent 规划器讨论**：[@jd_pressman](https://twitter.com/jd_pressman/status/1904139443189252252) 讨论了 **Weave-Agent 的设计与规划**，考虑了 ReActTree 和 MuZero 等 Agent 规划方法。
- **Smolagents 的增长**：[@AymericRoucher](https://twitter.com/AymericRoucher/status/1904219464263946480) 宣布 **smolagents 已达到 1.5 万个 GitHub Star**，并正在通过 E2B 或 Docker 集成沙箱化代码执行。
- **Together Chat**：[@togethercompute](https://twitter.com/togethercompute/status/1904204860217500123) 推出了 **Together Chat**，其特点是使用 DeepSeek R1 等 OSS 模型进行网页搜索、编码、图像生成和图像分析，[@togethercompute](https://twitter.com/togethercompute/status/1904204864885755905) 列出了其技术栈。

**Agent Engineering and Applications**

- **Agent 工程演讲与文章**：[@swyx](https://twitter.com/swyx/status/1904256213661192405) 分享了关于 **Agent 工程的演讲和文章**，定义了 Agent，概述了六个要素，并讨论了它们的潜在影响。
- **Linear 与 Codegen 集成**：[@mathemagic1an](https://twitter.com/mathemagic1an/status/1904293319297179871) 宣布了 **Codegen 与 Linear 的集成**，使 Agent 能够解决工单并关闭重复项，并强调了 Linear 为机器人扩展的功能 [@mathemagic1an](https://twitter.com/mathemagic1an/status/1904293320840655249)。
- **Agent 评估指标**：[@_philschmid](https://twitter.com/_philschmid/status/1904147086011940942) 提倡使用 **pass^k 而非 pass@k 来评估 Agent**，认为这能提供更准确且符合用户体验的性能指标。

**Economic and Strategic Implications**

- **AI 自动化与经济增长模型**：[@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1904180712393036095) 讨论了 **GATE，一个用于 AI 自动化经济影响的模型**，预测了数万亿美元的 AI 投资、极端的算力扩展以及显著的经济增长。
- **美日国防创新奖**：[@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1904156111621754905) 宣布 **Sakana AI 在美日国防创新竞赛中获奖**，凭借其新颖的 AI 解决方案脱颖而出。
- **关于中国与 AGI 的观点**：[@teortaxesTex](https://twitter.com/teortaxesTex/) 分享了关于中国技术和战略优势的多项观点，包括其国家能力、工业基础和 AGI 方面的努力。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1904008640542937273) 还提到了 DeepSeek 的“补充品商品化 (commoditize your complement)”理论。

**ARC-AGI 基准测试**

- **ARC-AGI-2 发布与竞赛**：[@fchollet](https://twitter.com/fchollet/status/1904265979192086882) 宣布发布 **ARC-AGI-2**，这是一个旨在衡量通用流体智力的基准测试，并启动了 ARC Prize 2025 竞赛，大奖奖金达 700,000 美元 [@fchollet](https://twitter.com/fchollet/status/1904266438959084003)。他指出，目前顶尖的 AI 方法得分非常低，需要测试时自适应 (test-time adaptation)，并讨论了评估方法论 [@fchollet](https://twitter.com/fchollet/status/1904267900963475807)。

**幽默与梗图**

- **凭感觉编程 (Coding by Vibes)**：[@gneubig](https://twitter.com/gneubig/status/1904186575732253008) 分享了一条关于**通过提示词改进 Vibe Coding** 的推文，区分了个人项目的 Vibe Coding 与 Agent 行为之间的差异。


---

# AI Reddit 回顾

## /r/LocalLlama 回顾

**主题 1. DeepSeek V3-0324：性能及对比 R1 的预期**

- **[DeepSeek 发布新的 V3 Checkpoint (V3-0324)](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324)** ([分数: 638, 评论: 125](https://reddit.com/r/LocalLLaMA/comments/1jip611/deepseek_releases_new_v3_checkpoint_v30324/))：**DeepSeek** 发布了其新的 **V3 Checkpoint (V3-0324)**，其中可能包含针对先前版本的更新和改进。帖子中未提供有关具体功能或增强的进一步细节。
  - 关于 **DeepSeek-V3 Checkpoint (V3-0324)** 的讨论包括对其作为未来 **R2 版本**基座的猜测，一些用户预计它将在 **4 月**发布。关于 R2 是否需要 **V4** 存在争论，有观点认为通过更好的 Scaling 和推理技术，无需新基座模型即可实现改进。
  - 用户正在寻求**基准测试结果**以比较新模型的性能，一些人注意到目前尚未发布官方基准测试。由于权重已开源发布，预计很快会有独立测试，并有人呼吁 DeepSeek 像 **Mistral** 一样发布自己的基准测试。
  - 有观察称该模型的**编程能力有所提升**，并已部署在 API 和 Web 平台上，一些用户注意到与原始版本相比，该版本**审查更严格**。**MTP 模块**因其在提高解码速度方面的作用而受到关注，根据[研究论文](https://arxiv.org/pdf/2412.19437)所述，其实现了 **1.8 倍的 TPS**。


- **[新版 DeepSeek V3 vs R1 (前者为 V3)](https://i.redd.it/cvnu636y1nqe1.png)** ([分数: 282, 评论: 56](https://reddit.com/r/LocalLLaMA/comments/1jiqi81/new_deepseek_v3_vs_r1_first_is_v3/))：该图片对比了两个版本的 **DeepSeek** 用户界面：**V3** 和 **R1**。**V3** 展示了更具动态感的设计，带有“多风”、“多雨”、“晴朗”和“下雪”的动画天气卡片；而 **R1** 提供了一个更简单的界面，带有“风”、“雨”、“太阳”和“雪”的切换按钮，每个按钮由单个图标表示。
  - **DeepSeek V3** 和 **R1** 的界面正在进行对比，**V3** 提供动画天气卡片，而 **R1** 具有更简单的切换按钮。用户对每个界面对应的模型以及用于对比的提示词感到好奇。
  - 尽管 **DeepSeek 模型**不是最便宜的，但由于成本和灵活性，用户更倾向于**开源模型**而非专有模型。据指出，**Sonnet** 比 **V3** 贵得多，尤其是在非高峰时段。
  - 讨论中提到了在本地运行 **command-a**，并提供了进一步探索的链接，例如 [Hugging Face 模型](https://huggingface.co/CohereForAI/c4ai-command-a-03-2025) 和展示界面的 [GIF](https://i.redd.it/sl2dyqigfnqe1.gif)。用户对视频等更多动态内容表现出兴趣，以便更好地理解动画功能。

- **DeepSeek V3-0324 在我的代码创意基准测试中已经追平了 Sonnet 3.7 —— “用 Python 编写一个渲染包含多个彩色光源的有趣场景的光线追踪器（raytracer）。”** ([Score: 215, Comments: 43](https://reddit.com/r/LocalLLaMA/comments/1jisuq4/deepseek_v30324_has_caught_up_to_sonnet_37_in_my/)): **DeepSeek V3-0324** 在涉及 Python raytracer 任务的代码创意基准测试中追平了 **Sonnet 3.7**，展示了较其先前版本的显著改进。基准测试显示，虽然大多数 **LLM** 生成的是简单的 RGB 场景，但 **Sonnet 3.7** 以及现在的 **DeepSeek V3-0324** 产生了更复杂且更具美感的场景，尽管这种创意提升的方法仍处于推测阶段。更多细节和数据可在 [GitHub 仓库](https://github.com/cpldcpu/llmbenchmark/blob/master/raytracer/Readme.md)中找到。
  - **DeepSeek V3-0324** 因其“疯狂的品味（psychotic taste）”而受到关注，与前代相比，它更像 **R1** 或 **QwQ** 等推理模型，并且因其创意写作输出而面临批评，一些用户认为尽管其基准测试分数很高，但输出内容并不连贯。**Gemma 3** 在小说创作中的连贯性和创意备受瞩目，与经常受到批评的 **R1** 输出形成鲜明对比。
  - **R1** 在基准测试中失败了，尽管进行了尝试，但未能生成可运行的程序，这引发了人们对其与旧版 **DeepSeek V3** 相比有效性的质疑。讨论表明，与之前的 **DeepSeek** 版本不同，**R1** 的长思维链（**CoT**）并不能保证成功的输出。
  - 值得注意的是 **DeepSeek V3-0324** 和 **Sonnet 3.7** 的程序体积有所增加，有人推测这是否归功于针对更长生成长度的训练或其他优化。单次尝试生成 10kB 的代码被认为具有重大意义，预示着模型能力的潜在进步。


**主题 2. Meta 的 ParetoQ 探索：2-bit 模型的承诺**

- **[Meta 上个月发布了一篇似乎被忽视的论文。《ParetoQ: Scaling Laws in Extremely Low-bit LLM Quantization》。这是一个比 BitNet 更好的解决方案，这意味着如果 Meta 愿意（增加 10% 的额外计算量），他们可以为我们提供性能极高的 2-bit 模型。](https://arxiv.org/pdf/2502.02631)** ([Score: 505, Comments: 49](https://reddit.com/r/LocalLLaMA/comments/1jig5re/meta_released_a_paper_last_month_that_seems_to/)): **Meta** 的 **ParetoQ** 论文引入了**极低比特 LLM 量化的扩展定律（scaling laws）**，提出了一个比 **BitNet** 更有效的解决方案。这使得仅需增加 **10% 的计算需求**即可交付高效的 **2-bit 模型**成为可能。
  - **量化与性能：** 讨论强调了 **2-bit 量化**在轻量级模型中的潜力，一些用户指出，这对于创意写作助手和聊天机器人等应用可能具有变革性。然而，人们也提出了对潜在速度下降以及量化对模型智能和指令遵循影响的担忧，并希望通过使用 **vulkan/T-MAC kernels** 来改进。
  - **研究与对比：** 用户讨论了 **ParetoQ** 框架作为一种更严谨的量化设置比较方法，强调了 2 到 3 bits 之间的学习转变。该论文因其优化 2-3 bit 模型训练的能力而受到关注，并与 **AQLM** 进行了对比，还提到了人类突触具有 **4-5 bpw**。
  - **资源与参考：** 讨论中提到了 **Intel auto-round** 项目和 **DeepSeek-R1-int2-mixed-sym-inc** 等资源，这些资源在保持 97.9% 准确率的情况下实现了相当的性能。论文链接：[arxiv.org](https://arxiv.org/pdf/2502.02631)。


**主题 3. 扩展 LLM 功能：从文本到多模态**

- **[我制作了一个关于 Transformer 工作原理的图表和解释](https://www.reddit.com/gallery/1jifvny)** ([Score: 272, Comments: 20](https://reddit.com/r/LocalLLaMA/comments/1jifvny/i_made_a_diagram_and_explanation_of_how/)): **LLM** 的功能正在向文本之外扩展，一位用户创建了一个**图表和解释**来阐述 **Transformer** 的运作方式。这项工作旨在为对 AI 和机器学习感兴趣的人提供对 **Transformer** 内部机制更清晰的理解。
  - **输入和输出嵌入 (Input and Output Embeddings)**：讨论了现代 **Transformer 架构**中输入和输出嵌入是否仍然关联，用户指出获取这些架构全面且最新的概览非常困难。
  - **资源与图表**：多位用户分享了辅助理解 **Transformer** 的资源，包括 **Cromulent123** 的详细解释以及指向包含相关图表的 GitHub 页面的链接 ([GitHub Llama Nuts and Bolts](https://github.com/adalkiran/llama-nuts-and-bolts/blob/main/docs/20-DIAGRAMS.md))。另一位用户推荐了 [Ben Levinstein 的 Substack](https://benlevinstein.substack.com/p/a-conceptual-guide-to-transformers) 上关于 **Transformer** 的概念指南。
  - **关于 Transformer 功能的详细解释**：**Cromulent123** 深入解释了 **Transformer** 的工作原理，重点关注 Token 嵌入过程、**Query, Key, and Value Matrices** 的作用，以及用于确定相关性的 **Attention Scores** 概念。他们还讨论了通过多个 **Transformer** 块进行**上下文增强 (contextual enrichment)** 的重要性，强调了对 Token 关系的细微理解。


- **[我不再确切理解 LLM 到底是什么了](https://www.reddit.com/r/LocalLLaMA/comments/1jijyx2/i_dont_understand_what_an_llm_exactly_is_anymore/)** ([Score: 233, Comments: 89](https://reddit.com/r/LocalLLaMA/comments/1jijyx2/i_dont_understand_what_an_llm_exactly_is_anymore/)): 作者对 **Large Language Models (LLMs)** 日益扩大的定义感到困惑，最初理解为基于文本数据预训练权重预测下一个词的系统。他们质疑 **LLM** 现在如何涵盖音频和图像生成等能力，并引用了处理 3D 点云数据的 **[SpatialLM](https://manycore-research.github.io/SpatialLM/)** 作为这种范围扩大的例子，寻求关于其与语言模型联系的澄清。
  - **扩散模型与 LLM**：关于像 **Stable Diffusion** 这样的模型是否属于 **LLM** 存在辩论，因为它们整合了 **T5** 来理解文本提示，尽管它们主要生成图像。**Co0k1eGal3xy** 认为，尽管这类模型传统上不属于 **LLM** 范畴，但由于其先进的语言理解能力，它们与 **LLM** 非常接近。
  - **Tokenization 与多模态模型**：**suprjami** 解释说，所有数据（包括文本、图像和音频）都被 **Tokenized** 成数字供 **LLM** 处理，这使得它们能够学习不同媒体类型之间的关系。**Chair-Short** 详细说明了 **Self-attention** 机制和 **Positional Encoding** 如何使 **LLM** 能够处理不同的数据模态，暗示了从纯文本模型向多模态能力的转变。
  - **定义 LLM**：讨论突显了定义 **LLM** 时模糊的界限，一些人将其视为能够处理和生成语言的大型模型，无论输入类型如何。**SnackerSnick** 提到 **LLM** 使用 **Tokenization** 和 **Embeddings** 来预测后续 Token，而 **Otherwise_Marzipan11** 和 **Co0k1eGal3xy** 则认为品牌化以及与语言（无论是文本、音频还是图像）的交互促成了 **LLM** 这一标签的使用。

- **Chatbot Arena 上可能出现的 Llama 4 原型** ([Score: 105, Comments: 21](https://reddit.com/r/LocalLLaMA/comments/1jiewjn/possible_llama_4_prototypes_on_chatbot_arena/)): **MetaAI** 正在 [Chatbot Arena](https://lmarena.ai/) 上测试几个匿名的 **Llama/Meta 模型**，这些模型可能是 **Llama 4** 的原型。像 **aurora**、**ertiga**、**pinnacle**、**solaris** 和 **spectra** 这样的模型支持图像功能，而 **rhea** 被识别为 **Llama 3**。
  - 讨论显示了对 **Chatbot Arena** 上模型身份的怀疑，因为一些模型（如 **anonymous-chatbot**）声称来自 **OpenAI**，而其他模型（如 **rage** 和 **phantom**）则被怀疑是 **Meta** 的模型。用户注意到这些模型经常提供不一致的公司归属信息，这可能是由于防护模型（guard model）或幻觉（hallucinations）造成的。
  - **anonymous-chatbot** 和 **nebula** 模型因其性能而受到关注，其中 **nebula** 因在测试中表现出色而特别受到称赞，而 **rage** 和 **rhea** 等模型收到的评价褒贬不一，其中 **rhea** 因其友好的态度和对表情符号的使用而受到关注。
  - 关于是否有模型确实是 **Llama 4** 存在争议，用户指出没有模型明确自称为 **Llama 4**。一些评论认为 **Meta** 可能正在测试多样化的写作风格，或使用随机化的系统提示词来掩盖模型的真实来源。


**主题 4. TeapotLLM 的影响：轻量级问答模型**

- **[发布 TeapotLLM - 一个开源的 ~800M 模型，用于抗幻觉问答和文档提取，完全在 CPU 上运行。](https://huggingface.co/teapotai/teapotllm#evaluation)** ([Score: 163, Comments: 50](https://reddit.com/r/LocalLLaMA/comments/1jioxj4/announcing_teapotllm_an_opensource_800m_model_for/)): **TeapotLLM** 是一个开源模型，专为抗幻觉问答和文档提取而设计，采用约 **8 亿参数** 的架构。它经过优化，可以完全在 **CPU** 上运行，使其在无需专门硬件的情况下也能被广泛使用。
  - **TeapotLLM 的抗幻觉能力**：讨论强调了该模型对抗幻觉的关注及其相对于 **Qwen** 和 **Llama** 等模型的表现，一些人对减少幻觉的说法表示怀疑。用户对其在幻觉排行榜上的排名感到好奇，目前已提供 [demo](https://teapotai-teapotchat.hf.space/) 供测试。
  - **模型的语言和输出能力**：该模型主要以英语训练，但理论上支持 **flan-t5** 涵盖的所有语言。它可以使用将字段解析为类型化 **JSON** 的库将结构化数据提取为 **JSON**，详见 [文档](https://teapotai.com/docs#3-information-extraction)，不过人们对扩大语言支持和在 **ollama** 等平台上进行测试很感兴趣。
  - **性能和资源占用**：**TeapotLLM** 针对 CPU 使用进行了优化，可运行在 Google Colab 约 **2GB 的 RAM** 内，方便计算资源有限的用户使用。人们有兴趣探索在更现代的模型（如 **Qwen 0.5B**）上进行微调以潜在地增强性能，同时保持当前模型在文档提取和简洁回答方面的优势。

## 其他 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**主题 1. ChatGPT 中新增改进的 Memory Alpha 增强了交互体验**

- **新的改进版 memory alpha 简直疯狂** ([Score: 414, Comments: 241](https://reddit.com/r/ChatGPT/comments/1jidd6w/new_improved_memory_alpha_is_insane/)): 该帖子讨论了 **ChatGPT** 中**新的改进版 memory alpha** 功能，将其影响比作从 **GPT-2 到 GPT-4** 的跨越。作者对 **DeepSeek** 的竞争能力表示怀疑，除非他们采用类似的进步，并对 **OpenAI** 的持续领导地位表示信心。
  - 许多用户对 **ChatGPT** 中新的 **memory alpha 功能** 的**可用性和不一致性**表示沮丧和困惑，一些用户尽管拥有 **pro 订阅**，却意外失去了访问权限。**CyberNoche** 和 **jalpseon** 强调了停用问题，而 **alpha_rover** 和 **DamionPrime** 分享了关于记忆持久性的积极体验。
  - 讨论涉及了 **ChatGPT** 订阅的**定价**，**Initial-Kangaroo-534** 质疑每月支付 **$200** 的价值。与之形成对比的是 **alpha_rover**，他认为该功能对于项目连续性具有不可估量的价值，相比其他 AI 工具，他会非常想念这个功能。
  - 一些评论者如 **3xNEI** 和 **SillyTwo3470** 推测了记忆功能的更广泛影响，认为这可能导致**人机混合 (human-AI hybridization)**。他们强调了**增强个性化**的潜力以及工具与伙伴之间界限的模糊，表明用户与 AI 交互方式将发生重大转变。


**主题 2. Anthropic 的营收激增，追平 OpenAI 2023 年的数据**

- **[Anthropic 目前月营收约 1.15 亿美元；与 2023 年 11 月的 OpenAI 持平](https://i.redd.it/klikk2sppkqe1.png)** ([Score: 272, Comments: 50](https://reddit.com/r/ClaudeAI/comments/1jijnw9/anthropic_is_making_about_115m_a_month_now_same/)): 据报道，Anthropic 每月产生 **1.15 亿美元** 的收入，追平了 **OpenAI 在 2023 年 11 月的营收**。2025 年的营收预测估计 **20 亿美元** 是可能的，**40 亿美元** 是乐观的，其中 Manus 为其营收贡献了约 **每任务 2 美元**。一张图片展示了从 2024 年 12 月到 2025 年 3 月年化营收增长了 **40%**，数据来自 **Bay Area Times**。
  - **Claude 的影响和使用**: 用户强调 **Claude Code** 是一款改变游戏规则的工具，由于其在自动化编程任务方面的有效性，一些人每天在上面花费 **$50**。虽然提到了 **AIDER** 和 **Cursor's Agent** 等替代方案，但被认为不如 Claude 有效，Claude 被描述为类似于拥有一个称职的实习生。
  - **营收来源与背景**: **Anthropic 营收** 的很大一部分归功于与 **AWS Bedrock** 的集成，随着企业广泛采用，预计将持续增长。讨论明确指出，报告的数字代表营收，而非利润。
  - **模型对比与偏好**: 用户比较了各种 AI 模型，指出尽管在某些情况下上下文窗口较小，但 Claude 提供了卓越的性能。提到了 **OG 600b 模型** 和 **Sonnet 3.7**，后者因其智能能力和迭代解决问题的能力而受到称赞。


**主题 3. AI 驱动的 Bug 修复自动化：一场为期 27 天的实验**

- **我让 AI 连续 27 天修复我生产环境中的 Bug - 经验教训** ([Score: 191, Comments: 80](https://reddit.com/r/ChatGPTCoding/comments/1jibmtc/i_made_ai_fix_my_bugs_in_production_for_27_days/)): 在 27 天的时间里，作者使用 **Claude 3.7** 自动修复了 21 个独特的生产环境 Bug，结果有 12 个成功的一次性修复，6 个部分成功，以及 3 个由于错误假设或复杂问题导致的失败。尽管最初的时间投入超过了手动修复 Bug，但该系统减少了认知负荷和上下文切换，尽管它可能不适合利基或复杂的特定领域问题。
  - **对开源的兴趣**: 人们对该项目开源有浓厚兴趣，**Relevant-Pitch-8450** 表示打算在清理代码后分享。用户赞赏其 UI 设计，并看到了该工具的潜在效用。
  - **潜在的商业化**: 像 **ClassyBukake** 这样的评论者建议该工具可以作为一项服务进行货币化，强调了它从个人和商业角度的吸引力。
  - **成本和时间效率**: **HelpRespawnedAsDee** 提出了关于该工具在较长时间内的成本和时间效率的问题，建议继续使用以评估长期效益。


**主题 4. 高级 Claude 工作流集成：MCP 外部工具**

- **我的 Claude 工作流指南：使用 MCP 外部工具的高级设置** ([Score: 124, Comments: 20](https://reddit.com/r/ClaudeAI/comments/1ji8ruv/my_claude_workflow_guide_advanced_setup_with_mcp/)): 该帖子提供了设置 **Claude 桌面应用程序**与 **Brave Search** 和 **Tavily** 等外部工具配合使用的详细指南，以增强其功能。这需要 **Claude Pro 订阅**（每月 20 美元）以及安装 **Node.js** 和 **Python** 等特定软件。指南包含了 **Windows** 和 **macOS** 的配置示例、访问开发者设置的说明，以及安装和设置问题的故障排除技巧。该指南强调了增强型网页搜索、文件系统访问和顺序思维（sequential thinking）的好处，并为有效使用提供了额外资源和安全性考量。
  - **Claude 桌面应用程序设置**因其对非开发者的友好性而受到称赞，为普通桌面用户在无需编程技能的情况下增强 Claude 的能力搭建了桥梁。该指南与 **Claude Code** 进行了对比，后者为习惯于命令行界面（CLI）的技术型用户提供了更多灵活性。
  - 推荐给那些有兴趣探索其功能的用户的 **Claude Code** 教程可在 [YouTube](https://www.youtube.com/watch?v=oM2dXJnD80c) 上观看。这突出了两种方法之间的区别：一种优先考虑易用性，另一种则侧重于高级自定义。


**主题 5. Wan 2.1 视频帧功能在 AI 中的创新**

- **[Wan-i2v - Prompt: a man throws a lady overboard from the front of a cruiseship.](https://v.redd.it/0ftuy4jmljqe1)** ([Score: 812, Comments: 51](https://reddit.com/r/StableDiffusion/comments/1jifrb8/wani2v_prompt_a_man_throws_a_lady_overboard_from/)): **Wan-i2v AI** 引入了新功能和进步，正如提示词场景 *"a man throws a lady overboard from the front of a cruiseship"*（一个男人从游轮前部将一位女士扔下船）所演示的那样。虽然该帖子没有提供更多细节，但它表明了 AI 生成内容对动作导向场景或潜在争议性主题的关注。
  - **Wan-i2v AI** 被讨论为一种 **image-to-video** 工具，一些用户注意到它无法独立从电影 **Titanic** 中创建起始帧，这意味着它使用了直接的截图。这突显了 AI 在没有参考图像的情况下生成完全原创内容的潜在局限性。
  - 用户幽默地批评了 AI 对**物理学**的理解，评论认为虽然 AI 目前可能无法掌握物理定律，但 **Stable Diffusion** 和 **Wan2.1** 等技术的进步正在迅速提高动画中真实物理效果的模拟，例如“胸部抖动”。
  - 对话还涉及了 AI 生成**电影替代结局**的想法，用户开玩笑说要为 **Titanic** 等电影创作新的结局。这引发了关于**版权问题**以及专注于 AI 制作内容的 **YouTube 频道**潜力的讨论，尽管知识产权方面存在挑战。


- **[Wan 2.1 首尾帧功能官方模型即将发布](https://i.redd.it/ngxqlw2t8nqe1.png)** ([Score: 100, Comments: 13](https://reddit.com/r/StableDiffusion/comments/1jirb3r/wan_21_begin_and_ending_frame_feature_having/)): **Wan 2.1** 即将发布支持**首尾帧插值（start and end frames interpolation）**的官方模型，社交媒体平台用户 "danielzy1990" 证实了这一点。更多详情请参考 [GitHub issue comment](https://github.com/Wan-Video/Wan2.1/issues/264#issuecomment-2747490626)。
  - 用户预期 **Wan 2.1** 的新模型将显著增强视频控制能力，一些人表达了对改进的期待，例如添加类似 **Hunyuan** 的引导层（guidance layer）以缩短生成时间。
  - 与 **Hunyuan** 的对比突显了其效率：**Hunyuan** 以 **24fps** 生成视频片段的时间几乎只有 **Wan** 以 **16fps** 生成时间的一半，强调了引导训练（guidance training）的潜在优势。
  - 人们对该模型支持**多个定时关键帧（multiple timed keyframes）**的能力很感兴趣，一些用户希望它能保持与现有 **img2vid** 功能的兼容性。


---

# AI Discord 回顾

> 由 o1-preview-2024-09-12 生成的摘要之摘要的摘要

**主题 1. DeepSeek V3 意外发布震撼 AI 社区**

- [**DeepSeek V3 作为开源巨头登场**](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324)：**DeepSeek** 发布了 **DeepSeek V3**，这是一个拥有 **685B 参数**的 **Mixture-of-Experts** 模型，采用 **MIT license**，可在 **Hugging Face** 上获取。社区对此感到兴奋，并将其性能与 **OpenAI** 的 **o1** 模型进行对比。
- [**DeepSeek V3 表现优于 R1？**](https://x.com/IterIntellectus/status/1904159903754621348)：用户声称 **DeepSeek V3** 在编程和前端任务中击败了 **R1**，即使没有 **chain-of-thought** 推理，并指出其成本效益和在数学方面的卓越表现。
- [**DeepSeek V3 发布时竟然没有 README！**](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324)：**DeepSeek** 发布 **DeepSeek V3** 时没有提供完善的文档，用户对缺少 **README** 感到既有趣又困惑，但官方提供了一个用于实验的 **playground**。

**主题 2. Qwen 模型及即将到来的 AI 创新**

- [**Hugging Face Transformers 已添加 Qwen3 支持**](https://github.com/huggingface/transformers/pull/36878)：开发者们非常激动，因为 **Qwen3** 的支持已集成到 **Hugging Face Transformers** 中，为即将到来的 **Qwen3** 模型做好了准备。
- [**Qwen2.5-VL-32B-Instruct 在 Apache 2.0 协议下发布**](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct)：**Qwen** 发布了 **Qwen2.5-VL-32B-Instruct**，这是一个通过强化学习（reinforcement learning）微调的多模态视觉语言模型，增强了数学推理和视觉问题解决能力。
- [**Qwen3 将支持 CPU 推理？**](https://github.com/huggingface/transformers/pull/36878)：用户推测 **Qwen3-15B-A2B** 因其尺寸原因，可能非常适合 **CPU 推理**，从而使先进的 AI 模型更加普及。

**主题 3. LLM 推理训练的辩论与进展**

- [**R1-Zero 训练偏差揭晓**](https://github.com/sail-sg/understand-r1-zero/blob/main/understand-r1-zero.pdf)：研究人员揭示了类 **R1-Zero 训练**中的一种偏差，即使用行均值（row mean）会偏向更短的正确回答和更长的错误回答，从而影响模型输出。
- [**GRPO 的长度爆炸困扰从业者**](https://x.com/WenhuChen/status/1903464313391624668)：用户正在努力解决 **GRPO** 训练导致的长度爆炸（length explosion）问题，并讨论了长度裁剪（length clipping）和课程学习（curriculum）等应对技术。
- [**MathFusion 增强 LLM 数学能力**](https://arxiv.org/abs/2503.16219)：**MathFusion** 通过跨问题指令合成增强了 **LLM** 的数学推理能力，提升了 **DeepSeekMath-7B**、**Mistral-7B** 和 **Llama3-8B** 等模型的表现。

**主题 4. Agent 工程与 MCP 进展**

- [**AGNCY 倡议推进 Agent 交互标准**](https://t.co/I558Qe2u4n)：Luke 领导的 **AGNCY** 旨在为 **Agentic** 交互创建开放标准，为开发更高效的 **AI Agent** 提供强大的框架。
- [**MCPwizard 简化 MCP Server 创建**](https://www.npmjs.com/package/mcpwizard)：开发者推出了 **mcpwizard**，这是一个 **CLI** 工具，可以简化 **MCP Server** 的创建和部署，从而轻松地为 **Claude** 等 AI 助手添加自定义工具。
- [**A16Z 探讨 MCP 与 AI 工具链的未来**](https://a16z.com/a-deep-dive-into-mcp-and-the-future-of-ai-tooling/)：**A16Z** 发布了关于 **Model Context Protocol (MCP)** 的深度研究，分析了其作为 AI 模型标准接口的潜力，并讨论了其对 AI 工具链的影响。

**主题 5. NVIDIA Nemotron-H 模型与硬件进展**

- [**NVIDIA 发布 Nemotron-H 混合模型**](https://research.nvidia.com/labs/adlr/nemotronh/)：NVIDIA 推出了 **Nemotron-H** 系列，这是 **Mamba-Transformer** 混合模型，提供高达 **3 倍** 的速度提升，模型参数范围从 **8B** 到 **47-56B**。
- [**Mistral 24B 重新受到青睐**](https://x.com/neurosp1ke/status/1903564534930907604)：**Mistral 24B** 被誉为近期最伟大的发布之一，用户对其在 **Apache 2.0** 协议下的强大性能和易用性印象深刻。
- [**Flash Attention 与 Hopper 架构揭秘**](https://developer.nvidia.com/ERR_NVGPUCTRPERM)：爱好者们深入研究了 **Flash Attention** 优化，并澄清了关于 **Hopper** 架构 **64B swizzle** 的困惑，增强了对 NVIDIA **GPU** 架构的理解。

---

# 第一部分：高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Sonar 3.7 Bug 导致模型中断**：一位用户报告了 **Sonar 3.7** 的一个 Bug，在编码时执行 *chown* 命令会导致模型退出并中断对话，用户想知道高源数据量与旧源数据量在性能上是否存在差异，以及搜索步骤之间的推理质量是否有区别。
   - 一位用户跟进指出，根据他们的经验，这种差异相当大，并在此分享了[截图](https://cdn.discordapp.com/attachments/1047649527299055688/1353541240200761537/image.png?ex=67e2afc3&is=67e15e43&hm=03e4b82072a680a8a9d215442a099e9d4c3adf29d24c0690d38258cbfe15690e&)。
- **Sonar 模型给出截断的代码片段**：多位用户报告称 Perplexity API 中的 **Sonar model** 会出现响应截断的问题，特别是在周末之后，尽管 JSON 格式是正确的。
   - 一位用户提供了一个 JSON 请求和被截断响应的示例，并指出切换到 **sonar-pro** 可以解决该问题，但出于成本考虑这并非首选方案。
- **Llama Index 与 Sonar 适配问题**：一位用户在将 **Sonar** 配置为 **Llama Index** 的聊天引擎用于 **RAG project** 时遇到错误，并请求协助。
   - 这凸显了在将 **Sonar** 与其他 AI 开发工具结合使用时可能面临的集成挑战。
- **Deep Research 速率限制**：一位用户询问是否可以放宽每分钟 **100 deep researches** 的限制，因为他们的应用程序有批量处理的需求。
   - 这一询问强调了工作负载较大的用户对更高 API 使用限制的需求。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Bonsai Bitnet 征集测试者以对比 Qwen2.5**：一位成员正在为 [deepgrove/Bonsai](https://huggingface.co/deepgrove/Bonsai/tree/main) 寻找测试者，询问 **bitnet** 与 **Qwen2.5 0.5B** 的对比情况。
   - 他们还链接了一个[相关的 Hugging Face Transformers PR](https://github.com/huggingface/transformers/pull/36878)，内容涉及添加 **Qwen3** 和 **Qwen3MoE** 的支持。
- **Orpheus TTS 模型获得音频微调功能**：根据新发布的 [Unsloth notebook](https://github.com/unslothai/notebooks/pull/17/files)，**Audio finetuning** 已支持 **Orpheus TTS** 模型。
   - 一位用户指出，这项工作全部由一位特定成员完成，且该 notebook 与本地音频 Tokenizing 后再进行常规 **Llama3** finetuning 相比，流程要精简得多。
- **Unsloth GitHub 接受直接 PR，但需等待**：一位成员询问了如何向 Unsloth 的 GitHub 贡献代码，另一位成员确认**直接提交 PR 是可以接受的**，但由于近期 PR 和 Issue 数量较多，可能会有延迟。
   - 讨论随后转向在 Colab 中修改数据准备步骤以适配 **.txt** 文件，旨在实现更廉价的推理，并链接了[原始 Issue](https://github.com/unslothai/unsloth/issues/14)。
- **GRPO 推理需要训练数据**：一位用户询问是否可以仅对输出的部分内容进行训练，特别是希望模型在推理过程中能自主生成推理过程。
   - 建议参考 [GRPO notebooks](https://github.com/unslothai/unsloth/tree/main/notebooks) 作为添加推理的标准方法，并指出模型在训练期间必须看到推理轨迹 (reasoning traces)，才能在推理时将其纳入考虑。
- **Unsloth 微调指南现已发布**：一位成员制作了 [fine-tuning with Unsloth 指南](https://youtu.be/Lt7KrFMcCis)，涵盖了理论层面、实践案例以及如何使用 **GRPO** 创建推理模型。
   - 该指南汇集了过去一年中所学到的所有知识。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Nebula 抢占聊天机器人风头**：成员们发现 **Nebula**（一个疑似来自 DeepMind 的匿名聊天机器人）表现*非常出色*，被认为是*目前最强的匿名模型*，在数学、英土（英语-土耳其语）翻译以及解决 Arc-AGI 问题方面优于其他模型。
   - 它看起来与 **Phantom** 类似，用户认为后者是 Google 的模型，两者都在竞技场中接受测试。
- **GPT-4o 获得人类对齐增强**：通过 OpenAI 的后期训练，**GPT-4o** 得到了显著提升，由于自 12 月以来持续的预训练，可能很快就会超越 **Grok 3**。
   - 推测认为它可能会登顶排行榜，这得益于 OpenAI 在 LM arena 人类偏好对齐方面的深厚造诣。
- **Specter 演进为 Phantom 再到 Nebula**：**Specter**、**Phantom** 和 **Nebula** 是同一模型的依次修订版本，在短短几周内展现了性能飞跃。
   - 成员们注意到，从 **Specter** 到 **Phantom** 的性能提升比从 **Phantom** 到 **Nebula** 的提升更为显著。
- **LMArena 修复 Bug 并调整排行榜**：LMArena alpha 版本收到了包括 Bug 修复和新功能在内的更新，鼓励测试者使用密码 `still-alpha` 在 [alpha.lmarena.ai](https://alpha.lmarena.ai/) 继续测试。
   - 修复了一个导致消息无法保存和投票失败的 Bug；排行榜列现在支持排序并实现实时数据更新；可以通过 [此 Google Forms 链接](https://forms.gle/8cngRN1Jw4AmCHDn7) 提供反馈，并通过 [此 Airtable 链接](https://airtable.com/appK9qvchEdD9OPC7/pagxcQmbyJgyNgzPx/form) 提交 Bug 报告。



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 的 CMD+Backspace 组合键引发问题**：用户对 **Cursor 的 CMD+Backspace** 导致意外删除项目感到沮丧，部分用户甚至丢失了多达 **7 次** 工作成果。
   - Cursor 团队计划将默认键绑定更改为 **CMD+Shift+Backspace**，并提供配置选项，预计于周一推出。
- **Claude 3.7 MAX 增加用户开销**：**Claude 3.7 Thinking**（现更名为 **Claude 3.7 MAX**）从 Pro 订阅计划转为按量计费，因成本增加引起用户不满。
   - 与标准的 **Claude 3.7 Sonnet** 相比，**Claude 3.7 MAX** 具有更高的上下文窗口和更多的 tool calls 次数。
- **Windsurf 在响应速度上领先**：一些用户发现 **Windsurf** 比 Cursor 更快、响应更灵敏，并指出了 Cursor 存在的卡顿和冻结问题。
   - 其他用户则因回滚功能和 Agent 性能而更青睐 Cursor，尽管他们也承认 AI 编程仍面临挑战。
- **MCP 组合成为热点**：用户正在尝试各种 **MCP (Model Context Protocol)** 服务组合来增强像 Cursor 这样的 AI 编程 Agent，其中 **Supabase MCP** 备受关注。
   - 一些用户认为 MCP 可能被过度炒作，并指出 Agent 存在过度使用或利用不足的情况，建议需要更清晰的指令。
- **3D 集成令 AI 编程者受挫**：一位用户在尝试使用 Claude 将 3D 模型（**FBX 格式**）集成到 three.js 项目时遇到了困难，主要面临 **FBXLoader** 的问题。
   - AI 在处理 3D 设计方面的局限性变得显而易见，建议切换到 **GLTF 格式** 并简化任务。



---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **DeepSeek V3-0324 击败 R1？**：Aider 社区对新的 [DeepSeek V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) 发布感到兴奋，认为它在编程和前端任务中表现优于 **R1**，尽管它缺乏思维链（chain of thought）。
   - 成员们强调了它与之前版本相比在编程和数学方面的优势，在 Benchmark 中将其与 **Sonnet 3.5** 进行对比，并指出了其成本效益。
- **Aider 驯服 Sonnet 的过度积极行为**：Paul Gauthier 透露，他通过在 Prompt 中添加一行让其“冷静”的指令，成功缓解了 **Sonnet 3.7** 的过度积极行为；该功能现已在 main branch 中可用。
   - 他鼓励用户根据他们的编程环节对这一调整提供反馈。
- **Aider 启用新主页**：Paul Gauthier 宣布在 [aider.chat](https://aider.chat) 发布 Aider 的新主页，展示了对 **Claude 3.7 Sonnet**、**DeepSeek R1** & **Chat V3**、**OpenAI o1**、**o3-mini** & **GPT-4o** 等模型的兼容性，并支持超过 100 种编程语言。
   - 此次更新为新用户提供了更好的入门介绍，并作为资源的中心枢纽。
- **Aider 的 Context 命令简化对话**：Paul Gauthier 在 Aider 中引入了一个实验性的 `/context` 命令，可自动设置对话上下文，在与 **Sonnet 3.7**、**R1** 和 **o3-mini** 配合使用时效果最佳。
   - 这一新命令通过智能识别并向对话中添加相关文件，提升了用户体验。
- **社区策划 LLM 上下文**：一位成员宣布推出 [ctxs.ai/weekly](https://ctxs.ai/weekly)，这是一个致力于收集 **aider conventions**、**prompts** 以及 **LLM** 面向的文档片段的网站。
   - 目标是为 **aider 社区** 创建一个有用的资源库，该成员正积极征求关于如何改进该网站的反馈。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **LCPP 上下文长度令人困惑**：用户发现，在 LCPP 中将上下文长度设置为 **100** 仍会尝试分配 **180GB** 的 RAM，导致 VRAM 耗尽。
   - 建议包括 Attention 覆盖了分配的上下文长度、缺少 ROPE 特定的参数，或使用 **Q8** 量化。
- **Deepseek V3 媲美 Sonnet 3.7**：**Deepseek V3 0324** 显示出与 **Sonnet 3.7** 相当的变化，暗示它们在架构上有着共同的进步，可参考[这张图片](https://cdn.discordapp.com/attachments/1154120232051408927/1353739123084627998/image.png?ex=67e2bf4e&is=67e16dce&hm=a779b06b1028e58affe0e8deb753caa78df67398ccb0c12f6de9f1360198b369)。
   - 一位用户甚至称其为具有 **Sonnet** 级别代码创造力的重大更新，并可能成为 **R2** 的基础。
- **Transformers 抛弃归一化**：受 **Transformers without Normalization** 论文启发，一位成员用 **tanh** 替换了归一化（normalization）。
   - 讨论随后集中在推理时移除 Expert 及其对较小权重的影响。
- **MathFusion 增强 LLM 数学能力**：**MathFusion** 通过跨问题指令合成（cross-problem instruction synthesis）提高了 **LLM** 的数学推理能力，增强了 **DeepSeekMath-7B**、**Mistral-7B** 和 **Llama3-8B** 等模型（[更多关于 MathFusion 的信息](https://x.com/gm8xx8/status/1903021157214748701?s=46)）。
   - 该方法创建了 **MathFusionQA 数据集**，用于微调模型并在极少额外数据的情况下提升 Benchmark 准确率度。
- **Qwen3 将支持 CPU 推理**：[transformers 库 PR#36878](https://github.com/huggingface/transformers/pull/36878) 显示正在添加 **Qwen3** 支持，这意味着这些模型很快将得到 **transformers** 库的支持。
   - 一位用户推测 **Qwen3-15B-A2B** 由于其尺寸，可能是 CPU 推理的理想候选者。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Sam Altman 预告 GPT-5 发布**：尽管没有正式公告，**Sam Altman** 确认 **GPT-5** 将于今年推出，引发了它可能在上半年发布以与 **R2** 或 **Llama-4** 竞争的猜测。
   - OpenAI Discord 服务器的成员暗示，一个未公布的 API 可能也即将推出。
- **GPT-4o：让用户彻底倒戈的模型**：一位用户发现 **GPT-4o** 作为一个强大的日常工具，以至于他们很少切换模型，只有在 **4o 消息**用完，或者处理重要及未解决的问题时，才会使用 **4.5, o1, o3** 等其他模型。
   - 该用户还声称构建了一个“引擎”，可以恢复 **400+ 轮对话**，并在超过 **500 轮**后继续保持上下文，且没有偏移或幻觉，全部通过默认 prompt 实现。
- **Many-Shot Prompting 增强多模态模型实力**：一篇研究论文（[MANY-SHOT IN-CONTEXT LEARNING IN MULTIMODAL FOUNDATION MODELS](https://arxiv.org/abs/2405.17015)）指出，像 **GPT-4o** 和 **Gemini 1.5 Pro** 这样的**闭源模型**从高达约 2,000 个示例的 many-shot 演示中获益显著，而权重开放模型则没有表现出同样的收益。
   - 论文指出，与 few-shot 示例相比，当提供 many-shot 演示时，像 **GPT-4o** 和 **Gemini 1.5 Pro** 这样的大型多模态基础模型表现出显著的性能提升。
- **运行由 GPT-4o 驱动的 F1 车队**：开源项目 **FormulaGPT**（[github repo](https://github.com/dawid-maj/FormulaGPT/)）模拟了由 LLM 驱动的车队之间的正面交锋，这些车队通过持续的推理、策略制定和细致的决策进行*上下文和自适应思考*。
   - 观众可以在**玩家 vs AI 模式**中挑战高级语言模型，或在 **AI vs AI 模式**中观看顶级 AI 模型对决，同时观察每个维修站停靠、换胎或超车动作背后的详细 AI 推理。
- **规避 Turnitin AI 检测器，如果你敢的话**：一名成员寻求关于如何规避 **Turnitin AI 相似度检测**的建议，因为其报告重复使用了公司的商业模式，这违反了 Turnitin 的 ToS。
   - 其他人认为这看起来像是为了作业作弊而进行的刷屏申诉，并建议使用 **humanize AI** 工具。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenAI 的 o1-pro：奢侈品级的定价？**：用户对 **OpenAI o1-pro** API 的定价反应强烈，其价格为 **$150/M input tokens** 和 **$600/M output tokens**，由于成本极高，有人称其为 *GucciAI*。
   - 另一位成员开玩笑说，考虑到算力限制，API 的缓慢可能是一个刻意设计的特性，以防止过度支出。
- **OpenRouter 上的图像生成功能缺失**：一位用户询问关于在 *gemini-2.0-flash-exp* 模型中使用 **Gemini 图像生成**的问题，但被告知 **OpenRouter 尚不支持图像生成**。
   - 团队表示，虽然图像生成已列入路线图，但目前没有支持 **Flux** 等图像模型的短期计划。
- **Lambda 端点饱受 404 错误困扰**：多位用户报告在尝试使用 **Lambda** 模型时遇到 **404 'no endpoint found' 错误**，尽管 **Lambda 的状态页面**显示运行完全正常。
   - 社区提供了建议，一些用户确认 **Llama 3.3 70B Instruct | Lambda** 模型对他们来说运行正常。
- **DeepSeek R1 挑战 OpenAI o1**：成员们注意到 **DeepSeek R1** 模型（一个 **671B 参数模型**，推理时 **37B 激活**）的性能与 **OpenAI o1** 相当，但它是开源的，并根据 **MIT 许可证**发布。
   - 它在 MIT 许可证下的可用性允许商业用途。
- **Claude 3.7 Sonnet 因过载错误而停滞**：用户报告在使用 **Claude 3.7 Sonnet** 时频繁出现**过载错误**，导致响应中断并被收取 input tokens 费用。
   - 一位用户建议采用重试策略或切换到 **Gemini 2.0 Pro** 作为替代方案，同时承认 Claude 在翻译方面的优势。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 缺少 NPU 支持**：用户报告 **LM Studio** 尚未支持 **NPU**，但 **0.3.11** 版本中已提供 **Ryzen AI** 支持。
   - 对于 **2GB VRAM** 等资源有限的用户，建议考虑使用 **Q6** 或 **Q8 量化** 的 **Gemma 3 1B** 以及 **CUDA** 运行时以提升性能。
- **KV Cache 量化大幅降低显存需求**：用户建议在运行具有 30k token 等长上下文窗口的模型时，利用 **KV cache 8-bit 量化** 来减少内存占用。
   - 请记住，**12GB VRAM** 对于 **32B 模型** 可能不足，建议将 **Phi-4** 或 **Qwen2.5 14b** 作为极具吸引力的替代方案。
- **多 GPU 获得应用内管理功能**：爱好者们对 **LM Studio** 的控制功能赞不绝口，该功能允许用户选择加载模型的 **GPU**，已在最新的 beta 版本中提供。
   - 多名用户确认，最新的 **LM Studio** beta 版本**原生支持多 GPU**。
- **Google Coral TPU 在 AI 领域表现不佳**：**Google Coral 双 TPU** 不适合 **AI** 使用，因为它没有用于存储数据的板载内存。
   - 一位拥有 **8060s** 的用户还询问了 **Framework Desktop** 的散热和功率余量。
- **4060 Ti：廉价推理的黄金选择**：配备 **16GB VRAM** 的 **RTX 4060 Ti** 是 **AI** 推理的预算友好型首选，价格约为 **500 美元/欧元**。
   - 一位用户提到，需要注意的是 **AMD 显卡** 未针对游戏进行优化，而 **Nvidia** 的 **5000 系列** 可能会出现熔断问题。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **VPN 代码劫持 OpenAI 网站？**：用户报告在 OpenAI 网站上看到了 `<veepn-guard-alert>` 和 `<veepn-lock-screen>` 标签，暗示存在 **VPN 注入**，但这很可能是由**他们自己的 VPN** 注入的代码 [sm0kywu.github.io/Amodal3R](https://sm0kywu.github.io/Amodal3R)。
   - 看起来该用户只是在使用 VPN。
- **cuOpt 解决 NVIDIA 的线性规划问题**：根据 [docs.nvidia.com](https://docs.nvidia.com/cuopt/user-guide/latest/introduction.html)，**NVIDIA® cuOpt™** 是一款 GPU 加速的优化 AI 微服务，在 [混合整数线性规划 (MILP)](https://en.wikipedia.org/wiki/Linear_programming#Integer_unknowns)、[线性规划 (LP)](https://en.wikipedia.org/wiki/Linear_programming) 和 [车辆路径问题 (VRP)](https://en.wikipedia.org/wiki/Vehicle_routing_problem) 方面表现出色。
   - 看来这款微服务在 NVIDIA 内部广受好评且性能强大。
- **CUDA Python 是新潮流？**：成员们讨论了这是否真的是 [blelbach 在 X 上](https://x.com/blelbach/status/1903148174853935326)提到的“CUDA Python 之年”，一些人断言 **Python** 对于 GPU 编程已经足够了。
   - 其他人则嘲讽了现代 Python 程序员，并分享了一个名为《现代 Python 程序员》的 [YouTube 视频](https://youtu.be/sVn4sBxLokA?si=mA3Djr31Nv_MZjUo)。
- **MoE 训练趋于稳定？**：一位用户声称 **MoE** 训练不稳定，但另一位用户反驳说它们“两年来一直很稳定”，现在“与稠密网络（dense networks）差不多”。
   - 稳定性主要归功于更好的算子（kernels）和无丢弃的 token 路由（dropless token routing），解决了数值不稳定和专家坍缩（expert collapse）等问题。
- **DeepSeek-V3 悄然发布**：成员们注意到 [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) 发布了其 **DeepSeek-V3-0324** 模型，并且一篇博客文章重复使用了他们的图表。
   - 该模型拥有 **685B 参数**，并提供 **BF16**、**F8_E4M3** 和 **F32** 等多种张量类型，并附带了微调和量化版本的链接。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Flash Attention FA 调试**：在关于理解 **Flash Attention (FA)** 的讨论中，一名成员建议通过编码和性能分析/调试来学习，并指出动手实现有助于理解普通 Attention，同样也适用于 **Flash Attention**。
   - 一名成员在 Triton 中实现 **Flash Attention 1** 时遇到了问题：*在 TRITON_INTERPRET=1 模式下可以运行，但在 CUDA 上有几个元素不匹配*。在增加 **rtol & atol** 后，测试通过了。
- **RTX 5080 获得 CUDA 12.8 支持**：一名开发者发布了一个补丁，使 **RTX 5080** 能够完全兼容 **CUDA 12.8 + PyTorch 2.5.0** 以及 **Blackwell / sm_120 架构**，并提供了一个包含脚本、diff 文件和说明的 [GitHub 仓库](https://github.com/kentstone84/pytorch-rtx5080-support)。
   - 同时也确认了 **WMMA** 指令是“包装器”，在 SASS 中直接编译为 **HMMA/IMMA/QMMA** 指令，类似于 **MMA** 指令的运作方式，正如 [CUDA Godbolt](https://cuda.godbolt.org/) 上所示。
- **解读 Hopper 的 Swizzle**：文档中对 **Hopper 架构** 中 **64B swizzle** 的描述让许多人感到困惑，但现在已澄清它是 **64*B* (字节)** 的 swizzle，其中每个方块是 **128*b* (位)**，这对应于 **8-bit 数据类型的 8x64 tile** 和 **16-bit 类型的 8x32 tile**。
   - 一名成员正在寻找 **ROCm** 专家，以帮助为 **tilelang HIP 后端** 实现无行间 bank 冲突的 swizzle。
- **牛津大学设立 AI 研究员职位**：牛津大学现有一个研究员空缺（博士后级别或同等经验），与 Rui Ponte Costa 合作研究 **游戏和神经影像中的 AI / RL**，**年薪 10 万英镑以上**。
   - 这项工作涉及开发一种 **AI 驱动的技术**，通过分析游戏数据来推断特定大脑区域对行为的贡献，从而实现 **神经系统疾病的非侵入性诊断和治疗**。
- **Flash Attention 的连续内存**：在 **Flash Attention** 中，张量以 **(batch_size, N, num_heads, d)** 格式存储，在 **d** 维度上是连续的（通常 > 64），这实现了高效的全局内存合并 (Global Memory Coalescing)，每个线程可以加载 **16B** 的数据。
   - 这也使得理解运行过程变得更加容易，因此可以使用 **LLM** 来辅助理解 **kernel 代码**，解释张量中特定位置的简单概念和变量状态。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Nvidia 工程师开发 Mamba-Transformer 混合模型**：根据 [Nvidia 的研究](https://research.nvidia.com/labs/adlr/nemotronh/)，他们推出了 **Nemotron-H** 模型系列，包括一系列 **8B** 和 **47-56B** 的模型，这些模型是 Mamba-Transformer 混合模型，提供了更快的推理速度。
   - 该模型因其与其他模型相比在速度上的提升而受到关注。
- **Mistral 24B 重新受到青睐**：**Mistral 24B** 的发布被视为一个重大亮点，因为它实力强劲且提供了可获取的基座模型，此外还有在 **Apache 2.0** 许可下发布的新开源版本。
   - 一名成员表示：*“Mistral 24B 可能是过去几个月里最伟大的发布之一，模型非常强大，而且你还可以访问基座模型。”*
- **R1-Zero 训练的长度偏差被揭示**：一项分析显示，在 **类 R1-Zero 训练** 中使用行均值 (row mean) 会引入偏差，倾向于更短的正确回答和更长的错误回答，详见 [论文](https://github.com/sail-sg/understand-r1-zero/blob/main/understand-r1-zero.pdf) 及配套 [代码](https://github.com/sail-sg/understand-r1-zero)。
   - 切换到全局均值 (all mean) 可以在不增加长度的情况下获得相当的性能，这引发了对“推理长度增加与能力提升相关”的图表的质疑。
- **中国计划发起开源 AI 攻势**：据 [这条推文](https://x.com/balajis/status/1903469483739730132) 称，中国计划向市场投放大量开源 AI 模型，以使 **AI 软件商品化** 并促进其硬件销售，这可能会动摇美国的科技主导地位。
   - **DeepSeek** 模型的发布曾导致美国科技股总市值缩水约 1 万亿美元，突显了中国 AI 的潜在影响。
- **浏览器自动化通过 Infinibranch 实现规模化**：[Morph Cloud 的 Infinibranch 浏览器](https://x.com/morph_labs/status/1902566171641266500) 被提议作为扩展浏览器使用 **Agent** 的可能解决方案，在为书籍列表查找亚马逊链接等任务上，将成功率提高到了约 **80%**。
   - 由于重度使用 JavaScript 的单页应用、CAPTCHA 和复杂的机器人检测，传统的网页抓取方法已经过时。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Gemini 更新深度解析**：**Gemini** 的 Dave Citron 加入了 @OfficialLoganK 的 Release Notes 播客，讨论了最近的更新，包括 **personalization**、**Canvas**、**Audio Overviews** 和 **Deep Research**，正如 [Google Gemini App](https://x.com/GeminiApp/status/1902752852843331650) 所报道的。
   - 讨论涵盖了从最近的应用发布到 **Gemini app** 中个性化的未来等话题，包括对用户数据和隐私考量的见解。
- **Claude Code 获得八项新功能**：Anthropic 为 **Claude Code** 推出了 **八项** 新功能，旨在帮助开发者更快速、更智能地构建，详情记录在他们的 [工程博客](https://www.anthropic.com/engineering/claude-think-tool) 中。
   - 功能包括一个新的 *think* 工具，引发了关于其实现和价值的讨论，一些人将其比作 Chain of Thought 提示。
- **A16Z 探讨 Model Context Protocol (MCP)**：A16Z 发布了对 **Model Context Protocol (MCP)** 的深度解析，探讨了其作为 AI 模型执行、数据获取和工具调用的标准接口的潜力，正如 API 是互联网的第一个伟大统一者一样 [A Deep Dive Into MCP and the Future of AI Tooling | Andreessen Horowitz](https://a16z.com/a-deep-dive-into-mcp-and-the-future-of-ai-tooling/)。
   - 该文章研究了 MCP 的用例、挑战以及它如何改变 AI 与工具交互的方式，并指出 API 曾是互联网的第一个伟大统一者，但 AI 模型目前还缺乏类似的对等物。
- **Roboflow 发布用于实时目标检测的 RF-DETR**：Roboflow 宣布了 **RF-DETR**，这是一个完全开源的实时目标检测模型，采用 Apache 2.0 许可证，可在 [GitHub](https://github.com/roboflow/rf-detr) 上获取。
   - RF-DETR 在 **COCO** 上实现了超过 **60 mAP** 的 **SOTA** 性能，其基础模型和大型模型的参数量分别为 **29M** 和 **128M**。
- **Swyx 构建 Agent 的未来**：**Swyx** 发布了关于 **Agent Engineering** 的 [新演讲和文章](https://x.com/swyx/status/1904256213661192405)，强调了在 @aiDotEngineer 全力投入 Agents 的原因。
   - 讨论定义了 **Agents**（感谢 @simonw），并详细阐述了 **Agent Engineering 的六个要素**，探讨了 **Agents** 如何成为 **ChatGPT** 达到 **10 亿月活跃用户 (MAU)** 的路径。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **征集移动端研究参与者**：团队正在为一项关于移动端用例的研究征集参与者，鼓励个人分享见解，以增强对如何在移动设备上使用该工具的理解。
   - 团队还宣布了即将到来的 **AI 模型更新**，更多细节将很快分享。
- **NotebookLM 中逐渐出现思维导图**：一位用户注意到他们的 **NotebookLM** 中缺少 **mindmaps**，而另一位用户确认在免费版本中已有该功能，这表明该功能正在逐步推出。
   - **mind map** 功能评价褒贬不一，需要不断重新生成才能更新，且除了主题之外缺乏细节。
- **NotebookLM 助力深度研究报告**：一位用户使用 **NotebookLM** 进行研究，撰写详细报告以帮助人们了解情况，重点关注本地和区域新闻。
   - 该用户还分享了一个播客节目的链接，讨论了 911 恶作剧电话的法律后果 [911 Prank Call: The Felony Consequences](https://creators.spotify.com/pod/show/peezyproductions/episodes/911-Prank-Call-The-Felony-Consequences-e30gfec)。
- **NotebookLM 作为 HR 政策中心**：一位用户探索了将 **NotebookLM** 作为 **HR 政策**、员工手册和新员工入职的中心枢纽。
   - 尽管这个概念很有前景，但用户注意到答案并不总是准确，并对有效的信息组织策略感到好奇。
- **通过缩放解决思维导图像素化问题**：一位成员建议在下载 **Mind Map** 之前先放大标签页，以提高输出质量并解决像素化问题。
   - 该成员称赞了其*极大的上下文窗口和低幻觉率*，甚至取消了对 **ChatGPT** 和 **Claude** 的订阅。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **虚拟测试器预测模型性能**：一名成员提议建立一个虚拟测试环境，在训练前预测 AI 模型的可行性，从而可能节省资源并加速创新；该模拟器旨在确定模型是否有*实际成功的机会*，还是注定会早期失败。
   - 虽然其他人指出，小规模测试新架构已经相对便宜，在 **3090** 上训练一天 **L6D512** 模型的成本约为 **$5**。
- **EleutherAI 评估评估方法**：一名成员在新的博客中详细介绍了 EleutherAI 的评估方法，并建立了一个 [MkDocs 站点](https://slyracoon23.github.io/lm-evaluation-harness/) 以方便导航；他们还在等待 [此 PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/2832) 的评审。
   - 该贡献者被警告不要使用 AI 生成 PR 内容，强调需要*审查贡献以避免增加垃圾内容*。
- **VectorAdam 声称具有旋转等变性**：[VectorAdam](https://www.dgp.toronto.edu/~zling/vector-adam/) 将二阶矩更新修改为每个梯度向量的向量范数平方，解决了 Adam 中的坐标系偏差，可能提高旋转等变性。
   - 有人指出 VectorAdam 与 Adafactor 不同，而更像是一个 **block size = hidden dim** 的分块近似。
- **MechInterp 因处于学术界之外而面临抵制**：成员们讨论了“mechinterp”品牌似乎受到了学术界的“抵制”，因为其中很大一部分内容处于传统学术渠道之外，且他们对这种范式持有抵触情绪。
   - 一名成员发现触发激活的第一个 token 是 *holocaust*，但它不是激活最强的 token，并想知道神经元激活是否具有上下文特定性。
- **递归设计优于 GANs、CNNs 和 RL**：一名成员介绍了一种使用递归设计的新颖图表，将其与传统的 **GANs** 区分开来；该实现强调结构组织而非顺序处理，利用 **CNNs** 进行过滤，并利用 **RL** 来优化响应。
   - 另一名成员正在起草一个 PR，将评估逻辑更新到最新版本 `lm_eval==0.4.8`，参考了 [Evals PR](https://github.com/EleutherAI/gpt-neox/pull/1348)。



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF Agents 课程拥抱新框架**：**Hugging Face Agents Course** 现在集成了 **LlamaIndex**、**LangChain** 和 **smolagents**，为学习者提供多样的 Agent 框架方法，如[这条推文](https://x.com/ben_burtenshaw/status/1903025737633841170)所述。
   - 使用 Agents 课程的成员指出 **LangGraph** 比较*死板*，这有助于在构建 **smolagents** 时指导他们的流程。
- **pdf2notes 轻松转换 PDF 笔记**：[Pdf2Notes](https://github.com/AstraBert/pdf2notes) 使用 **LlamaParse** 和 **Llama-3.3-70B** 将 PDF 转换为有条理的笔记，还利用 **DeepMind** 的 **Gemini 2 Flash** 进行多模态解析，并封装在 **Gradio** 和 **FastAPI** 框架中。
   - 一名成员询问 **pdf2notes** 是否可以 100% 本地运行而无需外部 API，对需要 **Gemini** 和 **Groq** 订阅表示担忧。
- **SpatialLM 处理 3D 数据**：**SpatialLM** 是一种旨在处理 3D 点云数据的 3D 大语言模型，已在 Hugging Face 上发布，地址为 [manycore-research/SpatialLM-Llama-1B](https://huggingface.co/manycore-research/SpatialLM-Llama-1B)。
   - 它生成结构化的 3D 场景理解输出，可以通过 [项目网站](https://manycore-research.github.io/SpatialLM) 和 [GitHub 仓库](https://github.com/manycore-research/SpatialLM) 进一步探索。
- **InferenceClient API 抛出身份验证错误**：一位用户报告在尝试使用 `InferenceClient` API 列出已部署模型时出现 **403 Forbidden** 错误，即使配置了允许调用 Inference Providers 的只读 Token。
   - 该错误表示调用 Inference Providers 的权限不足，一位用户发布了一个具有相同错误的 [链接](https://huggingface.co/posts/kpadpa/282697879499561)。



---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP Prompt 测试需要 K8s**：测试 **MCP prompts** 需要 **Kubernetes 设置**，例如在 [此文件](https://github.com/strowk/mcp-k8s-go/blob/main/testdata/list_prompts_test.yaml) 和 [此测试](https://github.com/strowk/mcp-k8s-go/blob/10aa7fd54dd7839bbeeb6b8705243e8cdb67ca7e/testdata/with_k3d/list_k8s_namespaces_test.yaml#L50) 中找到的那些。
   - [此处](https://github.com/Abiorh001/mcp_ev_assistant_server) 提供了另一个带有 prompts 的实现，用于管理 **电动汽车充电站**。
- **Microsoft 发布官方 MCP C# SDK**：Microsoft 发布了新的官方 **C# SDK**，用于 **Model Context Protocol servers and clients**，可在 [此处](https://github.com/modelcontextprotocol/csharp-sdk) 获取。
   - 根据 [Vercel AI SDK 4.2](https://vercel.com/blog/ai-sdk-4-2)，该 SDK 为开发者提供了使用 **JavaScript** 和 **TypeScript** 构建 **AI applications** 的工具，并可集成到 [Next.js](https://nextjs.org) 和 [Svelte](https://svelte.dev/) 等 Web 框架中。
- **Zapier 集成 MCP**：Zapier 发布了一个 **MCP server**，[为 AI assistants 提供了超过 8,000 个集成接口](https://zapier.com/mcp)，使其能与各种应用交互。
   - 这种集成使 AI 能够执行现实世界的任务，如发送消息、管理数据、安排事件和更新记录，将其能力扩展到文本生成之外。
- **MCPwizard 简化 Server 创建**：一位成员介绍了 [mcpwizard](https://www.npmjs.com/package/mcpwizard)，这是一个 **CLI tool**，旨在简化 **MCP servers** 的创建和部署，重点介绍了初始化项目和向 Claude assistants 添加自定义工具等功能。
   - 该工具的 [GitHub repo](https://github.com/yoannarz/mcpwizard) 也已共享，供社区反馈和贡献。
- **Google Sheets MCP Server 实现直接编辑**：一位成员构建了一个 **Google Sheet MCP server**，允许 Claude 直接编辑电子表格，简化了数据处理和公式调整，如 [此推文](https://x.com/xing101/status/1903391600040083488) 中所述。
   - 代码可以在 [此处](https://github.com/xing5/mcp-google-sheets) 找到。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **使用特定语言对语言模型进行 Prompt 引导**：成员们讨论了为了让语言模型以特定语言（如德语）回答，最好用该语言编写 system message，以避免触发 *"Im Kontext Lernen"*（语境学习）。
   - 进一步建议 **避免使用否定句** 可以改善结果，并建议重新组织指令以使用主动动词。
- **Mistral 模型版本说明**：有提到 [Mistral Nemo 是一个 12b 模型](https://huggingface.co/mistralai)，而 Mistral 24b 是 Mistral 3 或 Mistral 3.1，并围绕项目的具体模型细节进行了讨论。
   - 在识别确切模型时出现了困惑，一位成员强调需要精确的模型信息以避免问题。
- **GPT4All 的 LocalDocs 神秘消失**：一位用户报告称其整个 local docs 目录无故消失，引发了对潜在原因的讨论，如 **安装文件夹的更改** 或 **缺乏管理员权限**。
   - 成员们建议备份 *localdocs.db* 文件和原始文档以防止数据丢失，并指出 Windows 11 更新可能会因为干扰驱动器盘符而导致此问题。
- **LLM 考虑应用于医疗办公自动化**：成员们讨论了在医疗办公环境中使用本地 LLM 帮助医生创建报告并辅助治疗的潜力，重点是让系统从过去的口述笔记中学习。
   - 然而，有人提醒说，由于存在幻觉风险且需要精确信息，**LLM 可能不适合处理财务或医疗数据**。
- **GPT4All 仍不支持视觉功能**：一位成员询问 GPT4All 能运行的模型中是否有具备视觉能力的，确认结果是 **GPT4All 不支持视觉功能**。
   - 建议使用 **LM-Studio** 等替代工具来处理视觉相关任务。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Open APIs 为可移植性铺平道路**：在探索**高性能软件**解决方案时，使用诸如 **OpenCL**、**OpenMP**、**OpenACC**、**Vulkan’s Compute API** 和 **SYCL** 等开放且可移植的 API 是一个很好的起点。
   - **POCL** 被指出是一个带有相关论文的学术项目。
- **民主化 AI 计算降低 GPU 成本**：Chris Lattner 的系列文章《[Democratizing AI Compute](https://www.modular.com/blog/democratizing-compute-part-1-deepseeks-impact-on-ai)》强调了**提高硬件利用率**以减少对昂贵 GPU 需求的重要性。
   - 该系列包括关于 **CUDA**、**OpenCL** 和 **AI compilers (TVM and XLA)** 的文章。
- **MAX Platform 咨询**：一位新用户询问了关于修改 **max/pipeline** 目录以及通过 [pixi.toml 文件](https://github.com/modular/max/tree/main/src/max)在 **MAX Platform** 内测试更改的问题。
   - 具体来说，他们对在不将其作为依赖项下载的情况下更改 **max-pipeline** 感兴趣。
- **Mojo 的格式化工具媲美 Black 和 fmt**：Mojo 包含一个内置的格式化工具 `mojo format`，类似于 Python 中的 `Black` 或 Rust 中的 `fmt`，用于代码格式化。
   - 同时，由于 Windows 编译器工具链处理起来非常麻烦，Windows 的 GPU 支持目前较为困难。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **AGNCY 倡议寻求 Agent 标准**：Luke 正在领导 **AGNCY**，这是一项致力于制定 [Agent 交互开放标准](https://t.co/I558Qe2u4n)的倡议。
   - 该项目旨在为开发更有效且具有互操作性的 AI Agent 提供一个强大的框架。
- **Deepseek 和 LlamaIndex 构建更智能的 RAG**：Akshay Pachaar 详细介绍了一个新项目，该项目集成 **Deepseek AI**，使用 **LlamaIndex** 进行编排，**Deepseek AI R1** 进行推理，**Ollama** 在本地提供 R1 服务，并使用 **Streamlit** 作为 UI 来创建 **RAG app**；更多详情请见[此处](https://t.co/KS26JUkwz0)。
   - 这旨在展示结合不同工具构建复杂应用程序的能力。
- **超时导致 Agent 工作流中断**：一位成员报告称，由于 **OpenAI endpoint** 未处理的**超时错误**，他们的 Agent 工作流发生了崩溃。
   - 建议捕获 `WorkflowRuntimeException` 或 `Exception` 而不是 `WorkflowTimeoutError` 来解决此问题。
- **成员们思考多 Agent 系统中的 Function Calling**：成员们正在思考通过 **function calling** 触发单个 Agent 是否可以取代多 Agent 系统中的**程序范围回退机制**。
   - 核心问题是这两种设置是否可以在某些场景下实现相同的功能，从而可能简化系统架构。
- **打造面试刷题模式**：一位成员正在使用 **Llama 3.2**、**Sonnet 3.7** 和 **Dolphin** 构建本地 AI，并将其混合成一个带有 RAG 和自定义 fine-tuning 的 16B 模型。
   - 他正试图让他的 AI 去*申请 AI/技术公司并通过面试*，他在 face tracking、blender、unity、powershell 和 TTS 方面拥有经验。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command-R-Plus 驱动分子 AI 助手**：一个由 **Cohere** 的 **command-r-plus** 驱动的 AI 助手正被用于构建结构生物学工具，并配备了 **MolStar** 分子查看器 ([https://ai.doi.bio](https://ai.doi.bio))。
   - 该网站支持 “load” 命令，例如通过输入 *“Show me 7zzz”* 将 **PDB entries** 加载到查看器中。
- **Cohere 澄清 Chat 安全政策**：一名成员询问了 **Cohere chat** 功能的**数据保留**和**安全政策**，询问数据是否会被用于模型训练。
   - Cohere 团队成员提供了 [privacy policy](https://cohere.com/privacy)（隐私政策）、[data usage policy](https://cohere.com/data-usage-policy)（数据使用政策）和 [security policy](https://cohere.com/security)（安全政策）的链接，并指出用户可以在其 [dashboard](https://dashboard.cohere.com/data-controls) 中控制数据设置。
- **API 滥用被怀疑是 SSL 错误的原因**：一名成员报告在向 **API** 快速发送请求时遇到 **SSL errors**，暗示尽管正确安装了 **py.ssl** 模块，但这可能是由于请求过于频繁导致的。
   - 另一名成员认为问题可能源于**不受信任的服务器证书**，其他成员则指出 **API rate limits** 通常返回 **429 error code** 而非 **SSL error**。
- **vnc-lm 发布支持 RAG 的 Discord 机器人**：一名成员发布了其 Discord 机器人 **vnc-lm** 的新版本，其特点是拥有 **RAG pipeline**，能够利用来自 **Wikipedia** 和 **DuckDuckGo** 的数据增强 Prompt。
   - 该机器人为每个 Prompt 增加约 **500 tokens**，附加五个来源信息分块以改进模型的上下文，代码已在 [GitHub](https://github.com/jake83741/vnc-lm) 上开源。
- **vnc-lm 现在通过 Docker 支持所有 LLM**：更新后的 Discord 机器人现在支持所有主流的本地和托管大语言模型 API（包括 **Cohere**），并支持 **Docker** 部署。
   - 随着新版本的发布，用户可以轻松地在 Discord 内编辑消息并获取新的回复。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **DeepSeek-V3 发布但缺少 README**：DeepSeek 发布了 **DeepSeek-V3**，但在 [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) 上缺少正式的 Readme，引发了社区的幽默调侃。
   - 尽管缺乏文档，但目前已提供 Playground，允许用户实验该模型。
- **数据质量仍在折磨 AI 工程师**：尽管经过多年的研究，即使在 **fineweb** 和 **lima** 等数据集得到认可后，定义和获取“优质数据”对于 AI 实验室来说仍然是一个挑战。
   - 一名成员对持续缺乏有效的 **PDF extraction** 工具表示沮丧。
- **LlamaExtract 工具实现文档结构化**：[LlamaIndex](https://www.llamaindex.ai/) 推出了 **LlamaExtract**，这是一款使用 genAI-native **agents** 来结构化复杂文档的工具。
   - 根据 [Jerry Liu 的推文](https://x.com/jerryjliu0/status/1902880391578653176)，它能适配最新的模型，从而准确地结构化财务报告和简历等文档。
- **GRPO LoRA 分数出人意料地高**：如[此 Pull Request](https://github.com/pytorch/torchtune/pull/2467) 所示，**GRPO LoRA 3B single device** 在 GMS8K 上达到了 **54%** 的准确率。
   - 尽管在计算中出现了多加一个 +2 的错误，但它在处理新问题上的表现超出了预期。
- **CUDA Graphs 压缩 GPU 操作**：成员们讨论了 **CUDA graphs**，它将一堆 GPU 操作捕获为一个图，并将其作为单个操作启动。
   - 这减少了从 CPU 启动 CUDA 操作的开销，从而减少了 GPU idle time（空闲时间）。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DLCoT Optimizer 削减 Token**: 新的 **DLCoT (Deconstructing Long Chain-of-Thought) Optimizer** 在保持或提高各项基准测试准确率的同时，将 Token 使用量削减了 **70-90%**，详见 [pull request #8000](https://github.com/stanfordnlp/dspy/pull/8000)。
   - 它通过分割 CoT 内容、移除冗余路径、过滤错误链并重构连贯输出来增强思维链推理，同时可与 **BootstrapFewShot** 等现有 DSPy 优化器协同工作。
- **DSPy 激发创意优化**: 成员们讨论了如何通过优化 Prompt 和使用 *good judge* 将 **DSPy** 用于创意内容生成，并参考了 [PAPILLON](https://github.com/Columbia-NLP-Lab/PAPILLON/blob/main/papillon_tutorial.ipynb) 和 [Agentic Reward Modeling](https://github.com/THU-KEG/Agentic-Reward-Modeling) 等资源。
   - 讨论强调了对示例 *inputs* 的需求，但如果评判器/指标可以在没有参考的情况下评估摘要，则不一定需要摘要（标签）。
- **通过 Prediction 实现细粒度反馈**: 通过 **Refine** 实现细粒度反馈即将推出，届时可以对输出进行特定检查并提供针对性反馈。
   - 版本 **2.6.15** 将支持返回 `dspy.Prediction(score=...., feedback=....)`，从而为模块提供细粒度的反馈。
- **多智能体协议标准探索检索**: 成员们探索将多智能体协议标准 (**MCP**) 扩展到检索器/检索增强生成（RAG）。
   - 他们正在讨论检索结果的共享 Schema，以及交换文档和 Embedding 的方法，以简化数据驱动的工作流，并降低组合多个模型和数据源的难度。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **发现数据集来源**: 一名成员在 [仓库的 extra 目录](https://github.com/tinygrad/tinygrad/blob/master/extra/datasets/sops.gz) 中找到了 `datasets/sops.gz` 数据集，该数据集用于 `speed_compare_cuda_ptx`。
   - 该数据集是通过同目录下的 [generate_dataset.sh 脚本](https://github.com/tinygrad/tinygrad/blob/master/extra/optimization/generate_dataset.sh) 生成的。
- **澄清 CUDA 移植配置**: 当被问及将 **Tinygrad** 移植到 **CUDA GPU** 时，一名成员提供了 [README.md](https://github.com/tinygrad/tinygrad/?tab=readme-ov-file#accelerators) 文件的链接，展示了项目支持的后端。
   - 这表明 **CUDA** 支持信息可以在项目的文档中找到。
- **会议 #63 议程提醒**: 第 63 次会议的议程包括 **公司更新**、**量化 DSP**、**BERT**、**scheduler**、**driver**、**tensor cores**、**WebGPU**、**ONNX**、**RetinaNet** 以及 **Torch frontend** 的讨论。
   - 此外还计划讨论围绕 **AMD LLVM backend** 的悬赏任务，以及 **test_ops**、**多 GPU 训练** 和 **torch compile** 等话题。
- **AMD LLVM Backend 进展**: **AMD LLVM backend** 的进展涉及多个已合并的 Pull Request，并已在 **Llama3** 和 **Flux** 示例中进行了测试。
   - 目前有一个 Pull Request 正在评审中，标志着该领域的持续开发。
- **ONNX 前端出现**: 宣布创建 `tinygrad.frontend.onnx`，标志着本周的工作重点将放在 **ONNX** 准备上。
   - 工作内容包括验证前 30 个 **Hugging Face ONNX** 仓库。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **测验标题拼写错误引发困惑**：一名成员报告了 **Quiz 7** 标题中的拼写错误，导致在核对 **Quiz 6** 答案时产生困惑。
   - 另一名成员确认了这一发现并向报告者表示感谢。
- **AgentX Research Track 申请开放**：入选学生将获得来自 **Berkeley 博士后/导师** 的指导，参与 **AgentX Research Track 项目**，截止日期为 **PDT 时间 3 月 26 日晚上 11:59**。
   - 加入或在 **AgentX** 中取得成功并不强制要求导师指导，实验（labs）和证书声明表（Certificate Declaration form）将于 4 月发布，详见[附图](https://cdn.discordapp.com/attachments/1280370030609170494/1353204258450964544/image.png?ex=67e2c76c&is=67e175ec&hm=1fb895b885ce732fd7e5b99b8ff24c55286d5)。
- **Research Track 采用远程模式且无偿**：一名成员确认 **AgentX Research Track 指导** 将以远程方式进行。
   - 另一名成员澄清说，指导是无偿的，导师仅提供研究项目的指导。

---

**MLOps @Chipro Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

**Codeium (Windsurf) Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

# 第二部分：分频道详细摘要与链接


{% if medium == 'web' %}

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1352726582187397160)** (998 条消息🔥🔥🔥): 

> `o3 mini, Grok 3, Chinese AI, Gemini deep research, Complexity plugin` 


- **O3 mini 与 Deep Research 辩论引发热议**：成员们就 Perplexity 的 Deep Research 是由 **o3 mini** 还是不同版本的 **o3** 驱动展开了辩论。一位成员表示 *o3 mini 表现很差*，而另一位成员分享了一张显示其 "Deep research" 由 **o3** 驱动的 [图片](https://cdn.discordapp.com/attachments/1047649527299055688/1353524567070740591/EF6D7AED-7F2D-419B-B425-346D0F7A421E.jpg?ex=67e2a03c&is=67e14ebc&hm=3f8eb38e1f0d6c33ee6bd3351027a9a9725eef4b0bac26182edd355b92e3d3e9&)。
   - Perplexity 团队收到了反馈，一名用户询问为什么他要求 *总结旧对话并帮助他在 Linux 上设置 Yubikeys* 的请求导致了胡言乱语，并附上了 [截图](https://cdn.discordapp.com/attachments/1047649527299055688/1353528582743261274/image.png?ex=67e2a3f9&is=67e15279&hm=06ff2642f00f19f84ba09a245bf1b61d959819956381b76473b9a197c90c7f91&)。
- **Sonar 3.7 "chown" 命令 Bug**：一位成员报告了 **Sonar 3.7** 的一个 Bug，即在编写代码时使用 *chown* 命令会导致模型退出并中断对话。该成员还想知道在搜索步骤之间，高源量与旧源量以及推理质量是否存在性能差异。
   - 一位用户跟进指出，根据他们的经验，这种差异相当大，并在此分享了 [截图](https://cdn.discordapp.com/attachments/1047649527299055688/1353541240200761537/image.png?ex=67e2afc3&is=67e15e43&hm=03e4b82072a680a8a9d215442a099e9d4c3adf29d24c0690d38258cbfe15690e&)。
- **Perplexity Deep Research 即将迎来升级**：成员们讨论了 Perplexity 上 **Deep Research** 即将进行的升级，并将其与 ChatGPT、Gemini、You.com 的 ARI 以及 Grok 的 **Deep Research** 进行了比较。
   - 一些用户认为目前的 Perplexity **Deep Research** 与其他产品相比处于垫底水平，对升级充满期待，并希望 **Deep Research** 的 *High* 功能能尽快全面发布。
- **Perplexity Web 应用出现故障**：用户报告 Perplexity Web 应用以及 Android 应用宕机，并称在 iOS 应用中也看到了 *something went wrong try again later* 的提示。
   - 恢复正常后，用户发现新增又删除了一个“0 enhanced queries”选项，且音频输出功能无法使用。
- **Complexity 插件是必备工具**：成员们讨论了使用适用于 Firefox 和 Chrome 的 Complexity 插件来开启额外功能。这个 [GitHub 仓库](https://github.com/pnd280/complexity) 可以增强 Perplexity.ai 的功能，例如 Deep Research (High)。
   - 为确保扩展程序正常工作，请确认版本为 v1.9.4.0，且左上角会出现一个仪表盘。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/-gif-4470566">一切都在计划之中... GIF - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/fly-insect-bug-evil-plot-gif-5650927">苍蝇 GIF - 苍蝇 昆虫 Bug - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/tc-gif-6128815387503134640">Tc GIF - Tc - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://chromewebstore.google.com/detail/popup-window/nnlippelgfbglbhiccffmnmlnhmbjjpe?hl=en">Popup window - Chrome 网上应用店</a>: 将标签页移动到独立窗口，去除标签栏、导航栏和书签栏 UI。</li><li><a href="https://tenor.com/view/red-button-spam-press-button-click-gif-17367381">红按钮连按 GIF - 红按钮连按 摁下按钮 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/angry-glitch-triggered-kawaii-anime-gif-13939686">愤怒故障触发 GIF - 愤怒故障触发 可爱动漫 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/pnd280/complexity">GitHub - pnd280/complexity: ⚡ 增强你的 Perplexity.ai</a>: ⚡ 增强你的 Perplexity.ai。通过在 GitHub 上创建账号为 pnd280/complexity 的开发做出贡献。</li><li><a href="https://tenor.com/view/shagarita-shalymar-shalymar-rivera-shalymarrivera-shalymar-rivera-gonzalez-gif-1971273378384510616">Shagarita Shalymar GIF - Shagarita Shalymar Shalymar rivera - 发现并分享 GIF</a>: 点击查看 GIF
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1352721841575428106)** (18 messages🔥): 

> `Trump, SSA shutdown, Boeing fighter, sunbathe, bluesky debates` 


- **Trump 威胁关闭 SSA**：一名成员分享了关于 **Trump** 威胁 **SSA shutdown**（关闭社会保障局）的 [Perplexity 页面](https://www.perplexity.ai/page/trump-threatens-ssa-shutdown-o-FVuonDY2QaSXR_Wh7f6o0Q) 链接。
- **Trump 授予波音战斗机合同**：一名成员分享了关于 **Trump** 授予 **Boeing fighter** 合同的 [Perplexity 页面](https://www.perplexity.ai/page/trump-awards-boeing-fighter-je-Ql3GzcXCQ_uxemkyU2.D7Q) 链接。
- **Bluesky 辩论 AI 数据标准**：一名成员分享了关于 **Bluesky** 辩论 **AI data standards** 的 [Perplexity 页面](https://www.perplexity.ai/page/bluesky-debates-ai-data-standa-gc0NsSciQW2cU5dzqcY0FQ) 链接。
- **给新生儿晒太阳的正确方法**：一名成员分享了关于 **sunbathe a newborn**（给新生儿晒太阳）正确方法的 [Perplexity 搜索](https://www.perplexity.ai/search/proper-way-to-sunbathe-a-newbo-6jmpq2c1SAGO1W.QsrRRgQ?0=d&1=d) 链接。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1352926667751948340)** (21 messages🔥): 

> `Perplexity API in Windsurf, API Credit vs Pro Subscription, Deep Research Limit, Sonar Model Truncated Responses, RAG Project with Sonar and Llama Index` 


- **Windsurf 接入 Perplexity API**：用户在 Windsurf 应用中设置 **Perplexity API** 时遇到问题并寻求建议。
   - 另一名用户确认，购买 **API credit** 应该允许在没有 Pro 订阅的情况下调用 API。
- **达到 Deep Research 速率限制**：由于应用中的批量处理需求，用户询问是否可以放宽每分钟 **100 次 deep researches** 的限制。
- **Sonar 模型响应被截断**：多名用户报告 Perplexity API 中的 **Sonar model** 出现响应截断问题，尤其是自周末以来，尽管 JSON 格式正确。
   - 一名用户提供了 JSON 请求和截断响应的示例，并指出切换到 **sonar-pro** 可以解决问题，但出于成本考虑并非首选。
- **Llama Index 难以适配 Sonar**：用户在将 **Sonar** 配置为 **Llama Index** 的聊天引擎以用于 **RAG project** 时遇到错误，并请求协助。
- **Perplexity Pro：是否包含 API Credits？**：一名新用户询问 **Perplexity Pro** 订阅是否包含 **API credits**。
   - 另一名用户分享了 [Perplexity 帮助中心链接](https://www.perplexity.ai/help-center/en/articles/10352901-what-is-perplexity-pro)，详细说明了 Perplexity Pro 的权益。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1352726480509210745)** (602 messages🔥🔥🔥): 

> `Bonsai bitnet, Mistral Small 3.1, Orpheus TTS, Gemma 3 27B, Llama 3 performance` 


- ****Bonsai Bitnet** 招募测试人员**：一名成员正在为 [deepgrove/Bonsai](https://huggingface.co/deepgrove/Bonsai/tree/main) 寻找测试人员，询问该 **bitnet** 与 **Qwen2.5 0.5B** 的对比情况。
   - 他们还链接了一个[相关的 Hugging Face Transformers PR](https://github.com/huggingface/transformers/pull/36878)，涉及添加 **Qwen3** 和 **Qwen3MoE** 支持。
- ****Mistral Small 3.1** 微调困扰**：多名用户报告了微调 **Mistral 3.1** 时的问题，遇到了错误和弃用功能。
   - 一名用户寻求关于选择高性价比云端实例来微调 **LoRA Mistral Small 3.1** 模型的建议，其他用户则报告了 **Unsloth** 与最新 **Mistral** 版本（特别是视觉微调方面）的兼容性问题。
- ****Orpheus TTS** 微调上线**：根据新发布的 [Unsloth notebook](https://github.com/unslothai/notebooks/pull/17/files)，**Audio finetuning**（音频微调）已随 **Orpheus TTS** 模型推出。
   - 一名用户指出，这项工作全部由特定成员完成，且该 notebook 相比于本地音频分词后再进行常规 **Llama3** 微调要精简得多。
- ****Gemma 3 27B** 微调问题**：一名用户报告了微调 **Gemma 3 27B** 的问题，即使在升级 transformers 并使用 **Unsloth Gemma3** 示例后仍遇到错误。
   - 具体错误发生在尝试运行模型时，导致 **llama.cpp** 和 **gguf** 文件失败。
- ****Unsloth** 在 **AMD Framework** 台式机上运行**：围绕 **Unsloth** 与 **Framework Desktop** 的兼容性展开了讨论，特别是关于 **ROCm** 的支持。
   - 一名成员提供了 ML 软件支持 **ROCm** 的时间线，暗示到 **Framework Desktop** 发布时，**AMD** 可能会得到很好的支持。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://docs.unsloth.ai/get-started/fine-tuning-guide#avoiding-overfitting-and-underfitting">微调指南 | Unsloth 文档</a>：学习微调的所有基础知识和最佳实践。初学者友好。</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Orpheus_TTS_(3B).ipynb">Google Colab</a>：未找到描述</li><li><a href="https://www.kaggle.com/competitions/drawing-with-llms">使用 LLMs 绘图</a>：构建并提交能够生成特定概念 SVG 图像的 Kaggle 软件包</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/mai">Google Colab</a>：未找到描述</li><li><a href="https://huggingface.co/blog/aifeifei798/llama-31-nemotron-nano-8b-v1-bnb-4bit-unsloth-trai">Llama-3.1-Nemotron-Nano-8B-v1-bnb-4bit unsloth 训练示例</a>：未找到描述</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct">Qwen/Qwen2.5-VL-32B-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct#using-%F0%9F%A4%97--transformers-to-chat">Qwen/Qwen2.5-VL-32B-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb#scrollTo=-Xbb0cuLzwgf">Google Colab</a>：未找到描述</li><li><a href="https://unsloth.ai/newsletter">Unsloth 通讯</a>：加入我们的通讯和候补名单，获取有关 Unsloth 的一切动态！</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama#id-6.-alpaca-dataset">教程：如何微调 Llama-3 并在 Ollama 中使用 | Unsloth 文档</a>：为创建可在 Ollama 本地运行的定制化个人助手（类似 ChatGPT）提供的初学者指南</li><li><a href="https://tenor.com/view/goku-super-saiyan-super-saiyan2-super-saiyan2goku-goku-vegeta-gif-23177097">悟空超级赛亚人超级赛亚人2 GIF - 悟空超级赛亚人超级赛亚人2 超级赛亚人2悟空 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-run-and-fine-tune-gemma-3">教程：如何运行和微调 Gemma 3 | Unsloth 文档</a>：如何通过我们的 GGUF 在 llama.cpp、Ollama、Open WebUI 上高效运行 Gemma 3，以及如何使用 Unsloth 进行微调！</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Advanced_Llama3_1_(3B)_GRPO_LoRA.ipynb#scrollTo=ptqkXK2D4d6p">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/HuggingFace%20Course-Llama3.1_(8B)-GRPO.ipynb#scrollTo=vzOuSVCL_GA9">Google Colab</a>：未找到描述</li><li><a href="https://tenor.com/view/gohan-dbz-gif-9459511">悟饭龙珠 Z GIF - 悟饭龙珠 Z - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/deepgrove/Bonsai/tree/main">deepgrove/Bonsai at main</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/2131">“Unsloth: Failed to make input require gradients!” 视觉微调 Gemma3 时报错 · Issue #2131 · unslothai/unsloth</a>：我正尝试参考此教程进行视觉微调 Gemma3：https://colab.research.google.com/drive/1j0N4XTY1zXXy7mPAhOC1_gMYZ2F2EBlk?usp=sharing#scrollTo=QmUBVEnvCDJv 我构建了我的数据集...</li><li><a href="https://huggingface.co/klei1/bleta-meditor-27b/tree/main">klei1/bleta-meditor-27b at main</a>：未找到描述</li><li><a href="https://github.com/webbigdata-jp/python_sample/blob/main/FanFic_Illustrator_demo.ipynb">python_sample/FanFic_Illustrator_demo.ipynb at main · webbigdata-jp/python_sample</a>：Python 示例脚本。通过在 GitHub 上创建账号来为 webbigdata-jp/python_sample 的开发做出贡献。</li><li><a href="https://github.com/unslothai/notebooks/pull/17/files">Etherll 提供的微调 Orpheus-TTS 的 Notebook · Pull Request #17 · unslothai/notebooks</a>：未找到描述</li><li><a href="https://github.com/huggingface/smol-course/blob/main/1_instruction_tuning/notebooks/chat_templates_example.ipynb">smol-course/1_instruction_tuning/notebooks/chat_templates_example.ipynb at main · huggingface/smol-course</a>：关于对齐 smol 模型的课程。通过在 GitHub 上创建账号来为 huggingface/smol-course 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth/issues/2144">“Qwen2_5_VLProcessor”对象没有属性“eos_token” · Issue #2144 · unslothai/unsloth</a>：你好，我正尝试仅对 Qwen2.5 VL 进行文本微调（同时保留视觉能力），具体为：unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit，但在访问时遇到上述错误...</li><li><a href="https://github.com/unslothai/unsloth/pull/1289">由 shashikanth-a 添加了对 Apple Silicon 的支持 · Pull Request #1289 · unslothai/unsloth</a>：未优化。尚不支持 GGUF。从源码构建 Triton 和 bitsandbytes。</li>

用于 bitsandbytes 构建的 `make -DCOMPUTE_BACKEND=mps -S .`
`pip install unsloth-zoo==2024.11.4`
`pip install xformers==0.0.25`</li><li><a href="https://www.vultr.com/?ref=9738530-9J">SSD VPS 服务器、云服务器和云托管</a>：Vultr 全球云托管 - 极速 SSD VPS 云服务器。100% KVM 虚拟化</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>：未找到描述</li><li><a href="https://github.com/huggingface/transformers/pull/36878">由 bozheng-hit 提交的添加 Qwen3 和 Qwen3MoE · Pull Request #36878 · huggingface/transformers</a>：添加 Qwen3。此 PR 为即将推出的 Qwen3 模型添加了代码支持。有关 Qwen 的更多信息，请访问 https://github.com/QwenLM/Qwen2.5。@ArthurZucker</li><li><a href="https://www.amazon.com/dp/B0DV3WWMBD">Amazon.com: 机器学习与人工智能：概念、算法与模型，Reza Rawassizadeh 编写的教育教科书：9798992162103：Reza Rawassizadeh：书籍</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1352845046021619763)** (41 条消息🔥): 

> `Unsloth PR 流程, 阿拉伯语 LLMs 微调, LLMs 的 Consensus 框架, Rotary Position Embedding (RoPE), Unsloth fork 与原始仓库对比` 


- **Unsloth GitHub 接受直接提交 PR**：一位成员询问是否可以向 Unsloth 的 GitHub 贡献代码，另一位成员确认**直接提交 PR 是可以接受的**，但由于近期 PR 和 Issue 数量较多，可能会出现延迟。
   - 讨论随后转向修改 Colab 中的数据准备步骤以支持 **.txt** 文件，旨在降低推理成本，并链接了[原始 Issue](https://github.com/unslothai/unsloth/issues/14)。
- **阿拉伯语 LLM 微调建议**：一位成员寻求关于针对特定方言微调阿拉伯语 LLM 的建议，有人建议 **Qwen2.5-7B** 是一个合适的模型，因为它具备良好的阿拉伯语能力。
   - 建议在微调时使用 **Q&A 格式**而非纯文本，并引导该成员参考 [Unsloth 入门指南](https://docs.unsloth.ai/get-started/beginner-start-here)以获取更多详情。
- **Consensus：LLM 协商决策框架**：一位成员介绍了 **Consensus**，这是一个兼容 Langchain 的框架，旨在实现多个 LLM 之间的协商决策，并强调了它在计算、谜题和难题方面的有效性。
   - 为那些有兴趣结合不同 LLM 和模型以达成单一确定性答案的人提供了 [Consensus GitHub 仓库](https://github.com/jersobh/consensus)。
- **重现 RoPE**：一位成员分享了他们为了兴趣和学习，重现 **RoFormer** 论文中关于 **Rotary Position Embedding (RoPE)** 结果的工作。
   - 他们更新了自己的 toy repo，包含了不同的 Attention 机制和位置嵌入（Positional Embeddings），可以在这个 [仓库](https://github.com/chrisjob1021/transformer-examples)中找到。
- **理解 Unsloth 的 Fork 仓库**：一位成员就如何向一个与原始仓库不同步的 Unsloth fork 贡献代码寻求指导，发现该 fork 是一个独立版本。
   - 澄清并非所有 fork 都旨在保持同步，贡献者应向维护者确认同步状态，因为由于结构差异无法进行合并，相关仓库见 [cut-cross-entropy](https://github.com/unslothai/cut-cross-entropy/pull/3)。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/chrisjob1021/transformer-examples">GitHub - chrisjob1021/transformer-examples: 现代 Transformer 架构关键组件的教学用 toy 实现和示例集合。</a>: 现代 Transformer 架构关键组件的教学用 toy 实现和示例集合。 - chrisjob1021/transformer-examples</li><li><a href="https://github.com/unslothai/unsloth/issues/14">[功能请求] 原始 txt 文件训练 · Issue #14 · unslothai/unsloth</a>: 如果能在 readme 中包含一个使用简单的未格式化文本文件进行训练的示例就太好了！</li><li><a href="https://github.com/unslothai/cut-cross-entropy/pull/3">由 BouajilaHamza 更新 Python 版本要求至 >= 3.9 · Pull Request #3 · unslothai/cut-cross-entropy</a>: 调整 Python 版本要求以兼容 Python 3.9 及以上版本。</li><li><a href="https://docs.unsloth.ai/basics/chat-templates)">Unsloth 文档</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here)">Unsloth 文档</a>: 未找到描述</li><li><a href="https://github.com/jersobh/consensus">GitHub - jersobh/consensus: Consensus 是一个兼容 Langchain 的框架，支持多个 LLM（大语言模型）之间的协商决策。它支持并行执行、多轮推理、同行反馈以及多数投票、加权置信度和排序选择等可定制策略。</a>: Consensus 是一个兼容 Langchain 的框架，支持多个 LLM 之间的协商决策...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1352737992451821621)** (257 条消息🔥🔥): 

> `训练输出的特定部分, GRPO notebooks, Qwen 模型依赖问题, CUDA 版本, Mistral 3.1`

- **推理需要训练数据 (Reasoning needs Training Data)**：一位用户询问关于仅训练部分输出的问题，特别是希望模型在推理（inference）过程中生成自己的推理。建议参考 [GRPO notebooks](https://github.com/unslothai/unsloth/tree/main/notebooks) 作为添加推理的标准方法，并指出模型必须在训练期间看到推理轨迹（reasoning traces），才能在推理（inference）时将其考虑在内。
- **UV 导致依赖问题**：一位用户在尝试修复 **Qwen 模型** 中的问题时遇到了 **unsloth-zoo** 的依赖问题，特别是与 *cut-cross-entropy* 库相关的问题。建议他们安装 **Python 3.11** 并重新构建，因为目前尚不支持 **UV**，且已提交了一个 **PR** 来解决 Python 版本要求。
- **CUDA 问题**：一位用户在运行 Qwen2.5 GRPO notebook 时遇到了与 **numpy.dtype size** 相关的 **ValueError**，这可能表明存在二进制不兼容。另一位用户建议安装 **Python 3.11** 并使用特定配置重新构建，以解决潜在的 CUDA 相关问题。
- **过时的 Mistral notebook 问题**：一位用户在使用 **unsloth/Llama-3.2-3B-bnb-4bit** 模型和 *Mistral 7B Text Completion - Raw Text training full example.ipynb* 时遇到了 **ValueError**，错误信息为 *"Some modules are dispatched on the CPU or the disk"*。有人指出该 notebook 已过时，应仅使用 [Unsloth 文档](https://docs.unsloth.ai/get-started/unsloth-notebooks) 中提供的版本，其中包含 GRPO 推理。
- **GGUF 模型出现幻觉**：一位用户报告称，将**微调后的 Llama 3.2** 模型转换为 **GGUF** 格式并在 **Ollama** 中使用后，出现了幻觉（hallucination）问题，尽管模型在转换前能正确回答测试问题。该用户参考了此 [链接](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb#scrollTo=kR3gIAX-SM2q) 中的 notebook，并看到了关于 *attention_mask* 以及 pad/eos tokens 重要性的警告。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing#scrollTo=95_Nn-89DhsL">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb#scrollTo=kR3gIAX-SM2q">Google Colab</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth 文档</a>: 以下是我们所有 Notebook 的列表：</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements#fine-tuning-vram-requirements">Unsloth 要求 | Unsloth 文档</a>: 这里是 Unsloth 的要求，包括系统和 GPU VRAM 要求。</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(3B)-">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb">Google Colab</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/what-model-should-i-use#:~:text=Instruct%20or%20Base%20Model%3F">我应该使用哪个模型？ | Unsloth 文档</a>: 未找到描述</li><li><a href="https://unsloth.ai/blog/contpretraining">使用 Unsloth 进行持续 LLM 预训练</a>: 通过使用 Unsloth 对 Llama 3、Phi-3 和 Mistral 进行持续预训练，让模型学习一种新语言。</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B">deepseek-ai/DeepSeek-R1-Distill-Qwen-7B · Hugging Face</a>: 未找到描述</li><li><a href="https://ollama.com/library/phi4-mini">phi4-mini</a>: Phi-4-mini 在多语言支持、推理和数学方面带来了显著增强，现在，期待已久的 function calling 功能终于得到了支持。</li><li><a href="https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF/blob/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf">Llama-3.2-3B-Instruct-Q4_K_M.gguf · unsloth/Llama-3.2-3B-Instruct-GGUF at main</a>: 未找到描述</li><li><a href="https://github.com/NovaSky-AI/SkyThought/tree/main/skythought/test-time-scaling">SkyThought/skythought/test-time-scaling at main · NovaSky-AI/SkyThought</a>: Sky-T1：在 450 美元内训练你自己的 O1 预览模型 - NovaSky-AI/SkyThought</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_3.5_Mini-Conversational.ipynb">Google Colab</a>: 未找到描述</li><li><a href="https://github.com/timothelaborie/text_classification_scripts/blob/main/unsloth_classification.ipynb">text_classification_scripts/unsloth_classification.ipynb at main · timothelaborie/text_classification_scripts</a>: 使用 Llama 和 BERT 进行文本分类的脚本 - timothelaborie/text_classification_scripts</li><li><a href="https://huggingface.co/klei1/bleta-meditor-27b/tree/main">klei1/bleta-meditor-27b at main</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/2146">AttributeError: module 'transformers.models.mistral3.modeling_mistral3' has no attribute 'logger' · Issue #2146 · unslothai/unsloth</a>: 你好，我在运行 Mistral Small 3.1 模型时遇到了以下错误 File "unsloth_zoo/compiler.py", line 1465, in unsloth_compile_transformers exec("modeling_file.logger.addFilter(HideL...</li><li><a href="https://github.com/huggingface/transformers/issues/28005.">huggingface/transformers</a>: 🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的尖端机器学习。 - huggingface/transformers</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb).">Google Colab</a>: 未找到描述</li><li><a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 以 2 倍的速度和减少 70% 的显存微调 Llama 3.3、DeepSeek-R1、Gemma 3 和推理 LLMs！ 🦥</a>: 以 2 倍的速度和减少 70% 的显存微调 Llama 3.3、DeepSeek-R1、Gemma 3 和推理 LLMs！ 🦥 - unslothai/unsloth
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1353413160085880912)** (7 条消息): 

> `Unsloth fine-tuning, Lc0 Chess LLM, Vibe coding` 


- **Unsloth 获取微调指南**：一名成员创建了 [使用 Unsloth 进行微调](https://youtu.be/Lt7KrFMcCis) 的指南，涵盖了理论层面、实践案例，以及如何使用 **GRPO** 创建推理模型。
   - 该指南汇集了过去一年中学到的所有知识。
- **LLM 嘲讽使用 Lc0 的国际象棋玩家**：一名成员在 [Discord 附件](https://cdn.discordapp.com/attachments/1179779344894263297/1353769600038211665/IMG_2180.png?ex=67e2dbb0&is=67e18a30&hm=59e72a0a554c30166f4a088356cbef73f14c0873c724cc7a450c0b955cebff82&) 中分享了一张图片，显示一个 LLM 正在嘲笑一名与 **Lc0** 对弈的国际象棋用户。
- **Vibe Coding 被低估了**：成员们讨论了 **Vibe Coding**，指出尽管可能受到行业批评，但它让编程再次变得有趣，并强调了理解代码功能、网络安全和解耦的重要性。
   - 一位成员表示：*业界可能在讨厌我们，但它让我重新爱上了编程。*


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1352726997566099547)** (51 条消息🔥): 

> `Tree of Thoughts limitations, Graph of Thought improvements, GRPO multi-turn setup, LLMs vs human brain, Llama3 Thai language support` 


- **Tree of Thoughts 因效率低下受到抨击**：一位成员表示 [Tree of Thoughts](https://arxiv.org/abs/2305.10601) (**ToT**) *简直是垃圾*，因为它需要非常特定的 Prompt，且其性能严重依赖于模型遵循格式的能力。
   - 该用户发现这种策略感觉就像是在一个问题上投入了大量算力却没有好的回报，如果模型不能很好地遵循 Prompt，那么整个策略就会崩溃。
- **Graph of Thought 在 Tree 的基础上进行构建**：一位成员指出 [Forest of Thought](https://arxiv.org/abs/2402.08774) 和 [Graph of Thought](https://arxiv.org/abs/2307.11838) 改进了 **Tree of Thought** 的一些粗糙之处。
   - 他们澄清说，默认的静态 **Tree of Thought** 在处理能力上确实有些局限。
- **Google 的 LLM 与大脑关联研究**：Google Research 团队正在 [通过 LLM 表示法破译人类大脑中的语言处理过程](https://research.google/blog/deciphering-language-processing-in-the-human-brain-through-llm-representations/)。
   - 理论认为，LLM 和人类语言的符号心理语言学模型为自然语言编码提供了根本不同的计算框架，使它们能够产生特定语境的语言输出。
- **GRPO 寻求多轮对话精通**：一位成员正在寻找在多轮对话设置中使用 **GRPO** 的示例，试图针对能够最大化长期回报的问题对模型进行微调。
   - 另一位成员建议通过 Prompt 让一个更大的 LLM 充当具有 2-3 轮对话能力的模拟器。
- **Continual Learning 依然难以实现**：一位成员很好奇目前是什么阻碍了社区在生产环境中使用 LLM 的 **Continual Learning**（持续学习），质疑为什么尽管有许多结果非常好的论文，但在实践中却没有被使用。
   - 作为回应，另一位成员发布了一个 [蟹老板金钱 GIF](https://tenor.com/view/money-mr-krabs-gif-18326632)，暗示主要原因是**成本**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://research.google/blog/deciphering-language-processing-in-the-human-brain-through-llm-representations/">通过 LLM 表示法破译人类大脑中的语言处理过程</a>：未找到描述</li><li><a href="https://tenor.com/view/money-mr-krabs-gif-18326632">Money Mr GIF - Money Mr Krabs - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1352721576818245664)** (844 条消息🔥🔥🔥): 

> `Mistral Naming Schemes, Phantom Chatbot, Nebula Chatbot, DeepMind's Nebula, OpenAI GPT-4o`

- **Phantom 聊天机器人是 Google 的作品**：聊天机器人 **Phantom** 来自 Google，成员们一直在对其进行测试，并将其描述为“非常出色”。
   - 它在竞技场中已经存在了大约一周，而在约 8 小时后被移出竞技场引发了关注，并引发了关于它与 **Nebula** 和 **Specter** 潜在联系的讨论。
- **DeepMind 的 Nebula 聊天机器人令人印象深刻**：**Nebula** 是一个可能来自 DeepMind 的匿名聊天机器人，成员们发现它“非常棒”，是“目前最好的匿名模型”。
   - 它似乎与 **Phantom** 相似，目前正在竞技场中接受测试，在数学、英土翻译以及解决 Arc-AGI 问题方面表现出色。
- **OpenAI 的 GPT-4o 获得提升**：据描述，**GPT-4o** 通过 OpenAI 的 post-training 技术得到了显著改进，可能很快就会超越 **Grok 3**，这归功于自 12 月以来持续进行的 pretraining。
   - 有推测认为，由于 OpenAI 在 LM 竞技场的人类偏好对齐（human preference alignment）方面的专长，它可能会登顶排行榜。
- **Specter, Phantom 和 Nebula 是 Checkpoints**：**Specter**、**Phantom** 和 **Nebula** 是同一个模型的不同修订版本，顺序为 Specter -> Phantom -> Nebula。
   - 成员们注意到，从 **Specter** 到 **Phantom** 存在性能飞跃，而从 **Phantom** 到 **Nebula** 的提升幅度较小，这一切都发生在几周之内。
- **Rhea 创建了南方公园游戏**：一名成员提示 **Rhea** 创建一个**南方公园世界观的 2D 游戏**，该模型将游戏的完整代码生成到了一个 html 文件中。
   - 这展示了 vibe coding，并引发了人们对 LLM 在面对带有 AI 乱码字母的虚假 AI 生成图像时，会幻觉（hallucinating）出不存在的迹象的担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/koltregaskes/status/1903800811509133815/photo/1">来自 Kol Tregaskes (@koltregaskes) 的推文</a>：顺便说一下，Gemini 2.0 Pro Thinking 将包含原生图像生成功能！再次感谢 @legit_api。👍</li><li><a href="https://x.com/Alibaba_Qwen/status/1904227859616641534">来自 Qwen (@Alibaba_Qwen) 的推文</a>：72B 对 VLM 来说太大？7B 又不够强？那么你应该使用我们的 32B 模型，Qwen2.5-VL-32B-Instruct！博客：https://qwenlm.github.io/blog/qwen2.5-vl-32b/ Qwen Chat：https://chat.qwen.ai HF：https://hugg...</li><li><a href="https://twitter.sywv.tech/">Twitter, Inc. | 服务于公众对话</a>：未找到描述</li><li><a href="https://aistudio.google.com/status">Google AI Studio</a>：Google AI Studio 是开始使用 Gemini（我们下一代多模态生成式 AI 模型系列）进行构建的最快方式。</li><li><a href="https://x.com/OfficialLoganK/status/1869902322840571922),">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>：我们将构建世界上最强大的编程模型，2.0 已经取得了许多进展。2025 年将会很有趣 :)</li><li><a href="https://bat9254.github.io/simple-svg-tools/">SVG 测试网站</a>：未找到描述</li><li><a href="https://x.com/oriolvinyalsml/status/1904217389950005563?s=46&t=P8-tRi_JAVcI6l5U6nOT4A">来自 Oriol Vinyals (@OriolVinyalsML) 的推文</a>：🤔 引用 AshutoshShrivastava (@ai_for_success)：更多消息显示 LMSYS Arena 上的 Nebula 实际上是一个 Google 模型，很可能是 Google Gemini 2.0 Pro Thinking 模型。它的编程能力也非常出色，...</li><li><a href="https://preview.reve.art/app">Reve: 让你的创意成真</a>：未找到描述</li><li><a href="https://t.ly/oHbxd">SVG 测试网站</a>：未找到描述</li><li><a href="https://imgsys.org/">imgsys.org | 由 fal.ai 提供的图像模型竞技场</a>：一个生成式 AI 竞技场，你可以在这里测试不同的提示词并选择你最喜欢的结果。查看模型排名并亲自尝试！</li><li><a href="https://x.com/OriolVinyalsML/status/1904217389950005563?t=jZJnHJHuMGrK1b58cncEjQ&s=19">来自 Oriol Vinyals (@OriolVinyalsML) 的推文</a>：🤔 引用 AshutoshShrivastava (@ai_for_success)：更多消息显示 LMSYS Arena 上的 Nebula 实际上是一个 Google 模型，很可能是 Google Gemini 2.0 Pro Thinking 模型。它的编程能力也非常出色，...</li><li><a href="https://x.com/m__dehghani/status/1904224150060671308?t=Vl7bAcPWqcZGaeiyOxvtlA&s=19">来自 Mostafa Dehghani (@m__dehghani) 的推文</a>：@ai_for_success @AnalogPvt Nebula 太优秀了，不会保持神秘太久！😉</li><li><a href="https://artificialanalysis.ai/text-to-image/arena?tab=Leaderboard">文本转图像模型竞技场 | Artificial Analysis</a>：通过在不知道提供商的情况下选择你喜欢的图像，来了解该使用哪些 AI 文本转图像模型。</li><li><a href="https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard/tree/main">lmarena-ai/chatbot-arena-leaderboard (main 分支)</a>：未找到描述</li><li><a href="https://status.gemini.com/">Gemini 交易状态</a>：未找到描述</li><li><a href="https://dubesor.de/assets/shared/UIcompare/deepseek3UI.html)">现代演示页面</a>：未找到描述</li><li><a href="https://dubesor.de/assets/shared/UIcompare/deepseek3%200324UI.html)">现代 CSS 展示</a>：未找到描述</li><li><a href="https://dubesor.de/assets/shared/SteinsGateWebsiteExamples/DeepSeek%20V3.html)">命运石之门终端</a>：未找到描述</li><li><a href="https://dubesor.de/assets/shared/SteinsGateWebsiteExamples/DeepSeek%20V3%200324.html)">命运石之门终端</a>：未找到描述</li><li><a href="https://dubesor.de/assets/shared/LLMBenchtableMockup/DeepSeek%20V3%200.04%20cents.html)">LLM 基准测试表</a>：未找到描述</li><li><a href="https://dubesor.de/assets/shared/LLMBenchtableMockup/DeepSeek%20V3%200324%200.07%20cents.html)">LLM 基准测试表</a>：未找到描述</li><li><a href="https://dubesor.de/assets/shared/MushroomPlatformer/DeepSeek%20V3.html)">简单平台游戏</a>：未找到描述</li><li><a href="https://dubesor.de/assets/shared/MushroomPlatformer/DeepSeek%20V3%200324.html)">简单平台游戏</a>：未找到描述
</li>
</ul>

### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1353778010121179146)** (1 条消息): 

> `Alpha 测试更新、Bug 修复、O3-Mini 格式化、排行榜改进` 


- **LMArena Alpha 更新发布**：LMArena alpha 版本根据用户反馈进行了更新，包括 Bug 修复和新功能；鼓励测试人员访问 [alpha.lmarena.ai](https://alpha.lmarena.ai/) 并使用密码 `still-alpha` 继续进行测试。
- **消息保存 Bug 已修复**：最新 alpha 版本修复了一个导致消息无法保存（并导致投票失败）的 Bug，优化了用户体验。
- **O3-Mini 格式化修正**：**O3-Mini** 模型现在可以正确地对文本进行格式化，提升了 alpha 平台内生成内容的可读性和展示效果。
- **排行榜现支持排序并实时更新**：排行榜各列现在支持排序，且数据实时更新，为用户提供动态且交互式的性能洞察。
   - 可以通过 [此 Google Forms 链接](https://forms.gle/8cngRN1Jw4AmCHDn7) 提供反馈，并通过 [此 Airtable 链接](https://airtable.com/appK9qvchEdD9OPC7/pagxcQmbyJgyNgzPx/form) 提交 Bug 报告。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://forms.gle/8cngRN1Jw4AmCHDn7">Arena - 新 UI 反馈</a>: 告诉我们你对新设计的看法！</li><li><a href="https://airtable.com/appK9qvchEdD9OPC7/pagxcQmbyJgyNgzPx/form">Airtable | 全民应用平台</a>: Airtable 是一个用于构建协作应用的低代码平台。自定义你的工作流，进行协作并实现宏伟目标。免费开始使用。
</li>
</ul>

</div>
  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1352720844400623687)** (857 条消息🔥🔥🔥): 

> `Cursor 的 Cmd+Backspace 问题、Claude 3.7 Thinking 定价与功能、Windsurf 表现更佳、MCP 组合、AI 对 3D 设计的理解局限性` 


- **Cursor 的 CMD+Backspace 灾难**：用户对 **Cursor 的 CMD+Backspace** 行为感到沮丧，这导致了意外的项目删除，一名用户报告称由于此问题不得不重新开始工作 **7 次**。
   - 作为回应，Cursor 团队计划将默认键绑定更改为 **CMD+Shift+Backspace**，并提供配置选项，目标是在周一前推出。
- **Claude 3.7 Thinking 消耗额外额度**：用户讨论了 **Claude 3.7 Thinking** 的转变，从包含在 Pro 计划中变为需要按需计费，现在品牌名为 **Claude 3.7 MAX**，部分用户对成本增加和工具调用（tool call）定价表示不满。
   - 确认 **Claude 3.7 MAX** 与标准版 **Claude 3.7 Sonnet** 相比，拥有更高的上下文窗口（context window）和更多的工具调用次数。
- **部分用户更青睐 Windsurf 的性能而非 Cursor**：一些用户发现 **Windsurf** 比 Cursor 更快、响应更迅速，理由是 Cursor 存在卡顿和冻结等性能问题。
   - 然而，也有人因 Cursor 的回滚功能和 Agent 性能而更喜欢它，并指出 AI 编程仍有很长的路要走。
- **MCP 组合探索**：用户正在尝试各种 **MCP (Model Context Protocol)** 服务组合，以增强像 Cursor 这样的 AI 编程 Agent，其中 Supabase MCP 因其实用性受到关注。
   - 此外还有关于 MCP 是否被过度炒作的讨论，一位用户提到 Agent 调用 MCP 过多或不足的情况，需要更清晰的指令。
- **3D 集成难度过大**：一位用户正尝试使用 Claude 将 3D 模型（FBX 格式）集成到 three.js 项目中，但在使用 **FBXLoader** 时遇到了问题，并发现了 AI 在处理 3D 设计方面的局限性。
   - 建议切换到 GLTF 格式并分小块进行工作以简化集成，并遵循清晰的任务分阶段计划。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.cursor.com/settings/beta">Cursor – 早期访问计划</a>: 未找到描述</li><li><a href="https://docs.cursor.com/settings/models">Cursor – Models</a>: 未找到描述</li><li><a href="https://cursor.directory/mcp">Cursor Directory</a>: 为你的框架和语言寻找最佳的 Cursor 规则</li><li><a href="https://supermaven.com/">Supermaven: 免费 AI 代码补全</a>: 最快的 Copilot。Supermaven 使用 100 万 Token 的上下文窗口来提供最高质量的代码补全。</li><li><a href="https://x.com/kenton_parton/status/1903603185459061001">Kenton Parton (@kenton_parton) 的推文</a>: @cursor_ai @ericzakariasson 能否将 “Plan, search, build anything…” 文本区域更新为非静态文本类型？它无法通过 Accessibility API 进行更新。</li><li><a href="https://docs.cursor.com/settings/models#max-mode">Cursor – Models</a>: 未找到描述</li><li><a href="https://exa.ai/">Exa</a>: Exa API 为你的 AI 从网络检索最佳的实时数据</li><li><a href="https://forum.cursor.com/t/how-can-i-make-my-sidebar-look-like-vscode/5737/3">如何让我的侧边栏看起来像 VS Code？</a>: 我通过添加代码 &quot;workbench.activityBar.orientation&quot;: &quot;vertical&quot; 解决了这个问题。谢谢！</li><li><a href="https://forum.cursor.com/t/0-48-removed-workbench-activitybar-orientation/68847/12">0.48 移除了 workbench.activityBar.orientation</a>: 不添加 Sync 功能，因为他们说“纯粹专注于 AI 功能”，但却移除了 workbench.activityBar.orientation 设置？这逻辑说不通……</li><li><a href="https://forum.cursor.com/t/max-mode-for-claude-3-7-out-now/65698/17">Claude 3.7 的 Max Mode - 现已推出！</a>: @jake @kstars111 感谢关于 tool calls 的观点。我今天会把这个添加到文档中，但简而言之，tool call 是 AI 决定在其自身输出之外采取的任何行动。这...</li><li><a href="https://forum.cursor.com/t/source-control-how-to-revert/46441/2">源码控制 | 如何回滚？</a>: 我没看到 Cursor 的源码控制图中有一个专门的“回滚”按钮。变通方法取决于你想做什么……重置到某个 commit（完全放弃更改）git reset --...</li><li><a href="https://status.anthropic.com">Anthropic 状态</a>: 未找到描述</li><li><a href="https://about.gitlab.com/topics/version-control/">什么是版本控制？</a>: 版本控制软件用于跟踪修订、解决代码中的集成冲突，并管理软件项目中涉及的不同产物。</li><li><a href="https://codellm.abacus.ai/">Abacus.AI - CodeLLM</a>: AI 驱动的代码编辑器，帮助你更快地编写、审查和重构代码。</li><li><a href="https://forum.cursor.com/t/max-mode-for-claude-3-7-out-now/65698?u=danperks">Claude 3.7 的 Max Mode - 现已推出！</a>: 摘要：🧠 核心采用 Claude 3.7 Thinking 📚 使用模型的整个 200k 上下文窗口 🛠 具有非常高的 tool call 限制 🔍 可以一次读取更多代码 💰 重要提示：仅通过使用...提供</li><li><a href="https://downloads.cursor.com/production/3def0c1e43c375c98c36c3e60d2304e1c465bd5c/darwin/arm64/Cursor-darwin-arm64.dmg">未找到标题</a>: 未找到描述</li><li><a href="https://ai.dev">Google AI Studio</a>: Google AI Studio 是开始使用 Gemini（我们下一代多模态生成式 AI 模型系列）进行构建的最快方式。</li><li><a href="https://github.com/hgbdev/cursor-agent-notifier">GitHub - hgbdev/cursor-agent-notifier</a>: 通过在 GitHub 上创建账号，为 hgbdev/cursor-agent-notifier 的开发做出贡献。</li><li><a href="https://github.com/GLips/Figma-Context-MCP?tab=readme-ov-file">GitHub - GLips/Figma-Context-MCP: 为 Cursor 等 AI 编程 Agent 提供 Figma 布局信息的 MCP 服务端</a>: 为 Cursor 等 AI 编程 Agent 提供 Figma 布局信息的 MCP 服务端 - GLips/Figma-Context-MCP
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1352718428938698882)** (585 条消息🔥🔥🔥): 

> `Firecrawl, o1 vs o3 mini 调试, Claude Think Tool, Aider 主页, Qwen 2.5 发布`

- **Ripgrep 兴起，Aider 社区欢欣鼓舞**：成员们表示有兴趣探索 [ripgrep](https://github.com/BurntSushi/ripgrep) 及其对 Aider 的潜在益处。
   - 虽然一位成员认为 **o3minihigh** 在调试/编程方面优于 **o1 high**，但他们承认这并未经过基准测试。
- **Aider 驯服 Sonnet 过度积极的特性**：Paul Gauthier 提到，他通过在 prompt 中添加一行让其“冷静”的要求，成功让 Aider 驯服了 **Sonnet 3.7** 过度积极的特性，从他的编码过程来看，这似乎很有帮助。
   - 此更新现已在 main 分支中提供，欢迎提供反馈。
- **Aider 新主页上线**：Paul Gauthier 宣布 Aider 的新主页已在 [aider.chat](https://aider.chat) 上线，重点介绍了它与 **Claude 3.7 Sonnet**、**DeepSeek R1** & **Chat V3**、**OpenAI o1**、**o3-mini** & **GPT-4o** 等模型的兼容性。
   - 它还支持 100 多种编程语言。
- **DeepSeek V3-0324 发布，超越 R1？**：Aider 社区对新发布的 [DeepSeek V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) 议论纷纷，称其在编程和前端方面甚至优于 R1，尽管它没有 Chain of Thought。
   - 成员们指出，它在没有推理的情况下表现出色，在编程和数学方面优于之前的版本，在基准测试中可与 **Sonnet 3.5** 媲美；其更低的价格提供了一个很好的替代方案。
- **Aider 的新 `/context` 命令聚焦对话**：Paul Gauthier 在 Aider 中引入了一个实验性的新 `/context` 命令，有助于自动设置对话上下文。
   - 该新命令在与 **Sonnet 3.7**、**R1** 和 **o3-mini** 配合使用时效果最佳，并能识别哪些文件应添加到对话中。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/teortaxesTex/status/1904118342358552875">来自 Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex) 的推文</a>：> DeepSeek V3 模型已完成小版本升级。欢迎访问官网、APP 或小程序尝试体验（DeepThink 已关闭）。我猜我们要迎来...</li><li><a href="https://memory.basicmachines.co/docs/cli-reference">CLI 参考</a>：CLI 参考</li><li><a href="https://x.com/txhunyuan/status/1903121005809373386?s=46&t=AZs45ckJ7UUM_kJZcxnR_w">来自混元 (@TXhunyuan) 的推文</a>：🚀 介绍 Hunyuan-T1！🌟 见证 Hunyuan-T1，AI 推理领域的最新突破！由 Hunyuan TurboS 驱动，专为速度、准确性和效率而生。🔥 ✅ 混合 Mamba-Transformer MoE A...</li><li><a href="https://aider.chat">Aider - 终端里的 AI 结对编程</a>：未找到描述</li><li><a href="https://x.com/IterIntellectus/status/1904159903754621348">来自 vittorio (@IterIntellectus) 的推文</a>：DeepSeek 突然发布了一个约 700GB 的新模型，MIT 许可。不可思议。</li><li><a href="https://x.com/natolambert/status/1903104262797922567">来自 Nathan Lambert (@natolambert) 的推文</a>：Qwen 3 即将发布！Meta 锁定 LlamaCon 是明智之举，否则 Llama 4 可能会再次推迟 🤭。我真的很期待 Llama 4，快点发布吧。</li><li><a href="https://x.com/jon_durbin/status/1903744256671396092>">来自 Jon Durbin (@jon_durbin) 的推文</a>：🪂 今天早上 chutes 上的 DeepSeek-* 模型迎来了重大性能更新！简而言之：DeepGEMM、MTP、编译。具有最小连接偏好的前缀感知路由（此处未列出，但不久前已在...完成）</li><li><a href="https://tenor.com/view/duh-sarcastic-whatever-gif-874996418923210673">Duh Sarcastic GIF - Duh Sarcastic Whatever - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/rick-et-morty-gif-19089264">Rick Et Morty GIF - Rick Et Morty - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://venice.ai/chat?ref=lRwKpO">未找到标题</a>：未找到描述</li><li><a href="https://www.baseten.co/library/deepseek-r1/">DeepSeek-R1 | 模型库</a>：一个拥有 671B 参数、具备 o1 风格推理能力的尖端 MoE LLM，已授权商业使用。</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3-0324">deepseek-ai/DeepSeek-V3-0324 · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/modelcontextprotocol/servers/tree/HEAD/src/sequentialthinking">modelcontextprotocol/servers 仓库中的 servers/src/sequentialthinking</a>：Model Context Protocol 服务器。通过在 GitHub 上创建账号，为 modelcontextprotocol/servers 的开发做出贡献。</li><li><a href="https://tenor.com/bDZju.gif">Naruto Secretfingerjitsu GIF - Naruto Secretfingerjitsu Jitsu - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://gist.github.com/paul-gauthier/aa10b40c69eaece0d0472bc2b1aa3642">PLAN.md</a>：GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://linux.do/t/511500">Deepseek V3 生成的天气卡片分享...</a>：单次回复达 token limit 截断了，点击右下角的继续生成，直接在原有的部分继续生成，好方便 😇 prompt : 第一次: 你是顶级前端工程师,现就职于 Apple. Create a single HTML file containing CSS and JavaScript to generate an animated weather card. The card shou...</li><li><a href="https://build.nvidia.com/deepseek-ai/deepseek-r1/modelcard">Deepseek-ai 开发的 deepseek-r1 模型 | NVIDIA NIM</a>：在推理、数学和编程方面表现卓越的尖端、高效 LLM。</li><li><a href="https://openrouter.ai/deepseek/deepseek-chat-v3-0324">DeepSeek V3 0324 - API、提供商、统计数据</a>：DeepSeek V3 是一个拥有 685B 参数的混合专家模型（MoE），是 DeepSeek 团队旗舰聊天模型系列的最新迭代。它接替了 [DeepSeek V3](/deepseek/deepseek-chat-v3) 模型...</li><li><a href="https://github.com/richardanaya/UtilityBelt">GitHub - richardanaya/UtilityBelt：从 Aider 与 MCP 服务器对话</a>：从 Aider 与 MCP 服务器对话。通过在 GitHub 上创建账号，为 richardanaya/UtilityBelt 的开发做出贡献。</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1">R1 - API、提供商、统计数据</a>：DeepSeek R1 来了：性能与 [OpenAI o1](/openai/o1) 相当，但它是开源的，并且具有完全开放的推理 Token。它的参数规模为 671B，推理过程中激活参数为 37B。运行...</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1:free">R1 (免费) - API、提供商、统计数据</a>：DeepSeek R1 来了：性能与 [OpenAI o1](/openai/o1) 相当，但它是开源的，并且具有完全开放的推理 Token。它的参数规模为 671B，推理过程中激活参数为 37B。运行...</li><li><a href="https://github.com/Aider-AI/aider/issues/2341">repomap 假设标识符具有合理的唯一性，这在大型代码库中会失效 · Issue #2341 · Aider-AI/aider</a>：正在调查 repomap 质量原因的 Issue

在编辑 Cassandra 时表现糟糕。看起来主要原因是 repomap 无法区分 Foo.X 和 Bar.X。所以我们最终得到了...</li><li><a href="https://openrouter.ai/deepseek/deepseek-chat-v3-0324:free">DeepSeek V3 0324 (free) - API, 提供商, 统计数据</a>：DeepSeek V3 是一款拥有 685B 参数的 Mixture-of-Experts 模型，是 DeepSeek 团队旗舰聊天模型系列的最新迭代版本。它接替了 [DeepSeek V3](/deepseek/deepseek-chat-v3) 模型...</li><li><a href="https://x.com/FireworksAI_HQ/status/1902823918429405509">来自 Fireworks AI (@FireworksAI_HQ) 的推文</a>：Fireworks AI 为 R1 提供与 DeepSeek 相同的定价，并在欧盟和美国提供安全部署。很高兴分享我们 DeepSeek R1 产品的最新增强功能：💡 基础版 DeepSeek R1：高性价比且高质量...</li><li><a href="https://fireworks.ai/blog/fireworks-ai-devel">Fireworks - 生成式 AI 的最快推理</a>：以极快的速度使用最先进的开源 LLM 和图像模型，或者使用 Fireworks AI 免费微调并部署您自己的模型！</li><li><a href="https://www.together.ai/models/deepseek-r1">Together AI | DeepSeek R1</a>：媲美 OpenAI-o1 的开源推理模型，在数学、代码、推理和成本效益方面表现出色。</li><li><a href="https://fireworks.ai/models/fireworks/deepseek-r1">Fireworks - 生成式 AI 的最快推理</a>：以极快的速度使用最先进的开源 LLM 和图像模型，或者使用 Fireworks AI 免费微调并部署您自己的模型！</li><li><a href="https://fireworks.ai/blog/fireworks-ai-developer-cloud">在 Fireworks AI 开发者云上实现更快、更高效的 DeepSeek</a>：了解 Fireworks AI 开发者云如何通过更快、更优化的 DeepSeek R1 部署加速 AI 创新。了解新的 GPU 选项、提升的速度以及用于高效开发的增强型开发者工具...</li><li><a href="https://fireworks.ai/blog/fireworks-ai-developer-cloud.">Fireworks - 生成式 AI 的最快推理</a>：以极快的速度使用最先进的开源 LLM 和图像模型，或者使用 Fireworks AI 免费微调并部署您自己的模型！
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1352727400571473932)** (148 条消息🔥🔥): 

> `Anthropic API, Aider 开发工作流, Claude 3.7, Svelte 5 + SvelteKit, Claude App 中的 MCP servers` 


- **Aider 开发工作流探索**：Paul Gauthier 通过添加需要修改的文件并依靠 repo map 来引入其他相关的上下文来使用 `aider`。
   - 他分享了自己使用 `aider` 来增强 `aider` 的[屏幕录像](https://aider.chat/docs/recordings/)，展示了如何添加新的编程语言和功能。
- **Claude 3.7 输出缓慢报告**：用户报告在生成大文件时 **Claude 3.7** 的输出速度极慢，输出速度降至每 2-5 秒 1 行。
   - 一名成员建议，可以通过联系 **Anthropic** 的销售团队来获取 API 访问的按月计费方案。
- **Aider 与 .gitignore 的集成**：一名用户提交了一个 PR ([feat: Add --add-gitignore-files flag](https://github.com/Aider-AI/aider/pull/3609))，允许 Aider 通过一个新的标志 `--add-gitignore-files` 来编辑被 Git 忽略的文件。
   - 该用户认为 `.gitignore` 应该只负责 Git，而不应规定 Aider 可以访问的内容，并指出他们已在 `.aiderignore` 中明确指定不忽略计划文件。
- **Gemini 输出限制**：一名用户遇到了 **Gemini** 的输出限制，而其他人建议切换到像 **Sonnet** 这样的模型以避免此类限制。
   - Aider 开发者 Paul Gauthier 建议使用 `--edit-format diff` 作为变通方案。
- **使用 Repomix 获取文档上下文**：一名用户建议使用 [repomix](https://repomix.com/) 从文档仓库（如 [Astro 的文档](https://github.com/withastro/docs)）中提取内容。
   - 这个想法是处理文档，过滤掉不必要的代码，并将输出作为只读文件提供给 Aider。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/recordings/">屏幕录像</a>：aider 构建 aider 的屏幕录像。</li><li><a href="https://docs.litellm.ai/docs/mcp">/mcp [BETA] - Model Context Protocol | liteLLM</a>：在 LiteLLM 中使用 Model Context Protocol。</li><li><a href="https://docs.astro.build/en/getting-started/">入门指南</a>：帮助你使用 Astro（以内容为驱动的网站 Web 框架）进行构建的指南、资源和 API 参考。</li><li><a href="https://aider.chat/HISTORY.html#release-notes">发布历史</a>：关于 aider 编写自身代码的发布说明和统计数据。</li><li><a href="https://github.com/Aider-AI/aider/pull/3609">feat: Add --add-gitignore-files flag by omarcinkonis · Pull Request #3609 · Aider-AI/aider</a>：修复了 base_coder.py 中的文件处理逻辑，以便在命令行指定时正确跳过 gitignored 文件；添加了新的 --add-gitignore-files 标志来控制是否处理 gitignored 文件...</li><li><a href="https://github.com/lutzleonhardt/mcpm-aider">GitHub - lutzleonhardt/mcpm-aider</a>：一个用于在 Claude App 中管理 MCP servers 以及供 aider 使用的命令行工具。也可以运行一个 MCP Server 来帮助你管理所有的 MCP Servers。</li><li><a href="https://github.com/withastro/docs">GitHub - withastro/docs</a>：Astro 文档。</li><li><a href="https://github.com/hotovo/aider-desk">GitHub - hotovo/aider-desk</a>：Aider AI 助手的桌面应用程序。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1353688834558394470)** (2 messages): 

> `Aider Conventions, Prompts, LLM Documentation Snippets, Maybe Codebase Cursor Rules, Project Management Guidelines` 


- **Aider 约定与文档网站发布**：一名成员宣布发布了一个网站，用于收集 **aider conventions**、**prompts** 以及面向 **LLM** 的文档片段，网址为 [ctxs.ai/weekly](https://ctxs.ai/weekly)。
   - 该成员正在寻求关于如何使该网站对 **aider 社区**更有用的反馈。
- **Maybe 代码库 Cursor 规则**：分享了一个链接，指向 **Maybe 代码库**结构和开发约定的高级概述，位于 [github.com/maybe-finance/maybe](https://github.com/maybe-finance/maybe/blob/main/.cursor/rules/project-conventions.mdc)。
   - 该文档提供了关于代码库结构和开发实践的见解。
- **代码质量项目管理指南**：在 [gist.github.com](https://gist.github.com/mberman84/19e184e3a3a4c3a20f32a18af51ce3bc) 链接了一份关于**项目方法**、**代码质量**、**开发工作流**和**版本控制最佳实践**的综合指南。
   - 该指南提供了关于有效项目管理和维持高代码质量的见解。



**提到的链接**：<a href="https://ctxs.ai/weekly">ctxs.ai context registry</a>：一个开源的、社区策展的用于 LLM 的上下文注册表。

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1352718448626499604)** (436 messages🔥🔥🔥): 

> `LCPP Context Length, Quantization and Performance, Chinese Thinking Models, Agentic Workflows, Deepseek V3` 


- **LCPP 的上下文分配异常**：用户报告称，在 LCPP 中将上下文长度设置为 **100** 仍会导致系统尝试分配 **180GB** 的 RAM，从而导致 VRAM 耗尽。
   - 成员建议 Attention 实现可能会覆盖分配的上下文长度，或者需要在运行命令中分配特定的 ROPE 参数；以 **Q8** 量化运行也可能避开此问题。
- **解读 DeepSeek-R1 性能**：一位成员指出，由于来自中国的最新思维模型（Thinking Models），基准测试可能已经过时，但在使用复杂的编程提示词进行测试时，[混元-T1 未能停止运行](https://llm.hunyuan.tencent.com/#/chat/hy-t1)。
   - 另一位用户强调，关键标记 *"wait"* 和 *"alternatively"* 可能是由 R1 在 RL 之前的微调所引导的。
- **DeepSeek V3 登场**：用户庆祝 **DeepSeek V3** 的到来，有人声称它能够作为推理模型，检测思维迭代，并间接验证解的存在性，称其为一次巨大的更新，具有 *Sonnet* 级别的代码创造力，并可能成为 R2 的基础。
   - 成员们还注意到它可以生成超出 Token 限制的 CoT，并且可以通过 [chat.deepseek.com](https://chat.deepseek.com) 访问。
- **Hermes 3 的 vLLM 建议**：澄清了使用 SGLang 推理 NeuralMagic FP8 量化版本的 Hermes 70B 而非 vLLM 应该不会产生任何问题。
   - 还有人指出，对于 ERP 私有微调，[Pygmalion](https://huggingface.co/PygmalionAI) 团队及其相关人员可能会提供帮助。
- **新手开发者寻求指导**：一位新开发者寻求关于使用 **Hermes3** 而非 **4o** 开发 AI 的建议。
   - 一名成员确认 **Hermes 3 API** 与 OpenAI 兼容，只需更改 *base URL* 和 *model* 即可使用标准的 OAI SDK 进行调用。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2411.04330">Scaling Laws for Precision</a>: 低精度训练和推理会影响语言模型的质量和成本，但目前的 Scaling Laws 尚未考虑到这一点。在这项工作中，我们设计了“精度感知”的 Scaling Laws...</li><li><a href="https://x.com/Teknium1/status/1903940557296295976">Teknium (e/λ) (@Teknium1) 的推文</a>: 制作大家都在 vibecoding 的这类游戏的教程在哪里？我只需要要求用 3js 或其他工具做一个游戏，它就能运行吗？我对网页游戏一无所知。</li><li><a href="https://fxtwitter.com/davidad/status/1903834443225190721">davidad 🎇 (@davidad) 的推文</a>: @burny_tech 不幸的是，对于更长远未来的“足够好”的规划，其答案可能就像拥有更长久的过去一样简单。🤷</li><li><a href="https://tenor.com/view/daspoody-sleep-sleepy-wake-woke-gif-2569845121217246002">Daspoody Sleep GIF - Daspoody Sleep Sleepy - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3-0324/discussions/3">deepseek-ai/DeepSeek-V3-0324 · 请求可在笔记本电脑上运行的小型蒸馏模型</a>: 未找到描述</li><li><a href="https://fxtwitter.com/OedoSoldier/status/1904130299635892274">OedoSoldier (@OedoSoldier) 的推文</a>: 哇，前端代码编写能力显著提升！V3 New vs R1 Prompt: 创建一个包含 CSS 和 JavaScript 的单个 HTML 文件，以生成一个动画天气卡片。该卡片应在视觉上展示...</li><li><a href="https://tenor.com/view/gif-gif-19496023">Gif GIF - Gif - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://arxiv.org/abs/2502.02631">ParetoQ: Scaling Laws in Extremely Low-bit LLM Quantization</a>: 在量化模型大小和准确性之间实现最佳权衡的最优位宽一直是持续争论的话题。虽然有些人提倡 4-bit 量化，但其他人则建议 1...</li><li><a href="https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-70B-FP8">NousResearch/Hermes-3-Llama-3.1-70B-FP8 · Hugging Face</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2503.16385">Deconstructing Long Chain-of-Thought: A Structured Reasoning Optimization Framework for Long CoT Distillation</a>: 大语言模型 (LLMs) 最近的进展通过长思维链 (CoT) 推理展示了卓越的推理能力。R1 蒸馏方案已成为一种前景广阔的...</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3-0324">deepseek-ai/DeepSeek-V3-0324 · Hugging Face</a>: 未找到描述</li><li><a href="https://docs.sglang.ai/backend/quantization.html">Quantization — SGLang</a>: 未找到描述</li><li><a href="https://github.com/cpldcpu/llmbenchmark/tree/master/thinkingtraces#readme">llmbenchmark/thinkingtraces at master · cpldcpu/llmbenchmark</a>: 各种 LLM 基准测试。通过在 GitHub 上创建账户来为 cpldcpu/llmbenchmark 的开发做出贡献。</li><li><a href="https://github.com/foundation-model-stack/fms-fsdp/tree/main/speculator">fms-fsdp/speculator at main · foundation-model-stack/fms-fsdp</a>: 🚀 利用原生 PyTorch 特性高效地（预）训练基础模型，包括用于训练的 FSDP 和 Flash attention v2 的 SDPA 实现。- foundation-model-stack/fms-fsdp</li><li><a href="https://github.com/ggml-org/llama.cpp/issues/11474">研究：基准测试 DeepSeek-R1 IQ1_S 1.58bit · Issue #11474 · ggml-org/llama.cpp</a>: 研究阶段背景研究（尽量避免重复造轮子）假设形成（你认为这将如何运作及其效果？）策略/实现形成分析...
</li>
</ul>

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1352963517946269769)** (46 条消息🔥): 

> `引导思考模型，Deepseek V3 对比 Sonnet 3.7，在代码库上微调 LLM，无需归一化的 Transformer，利用 LLM 进行光线追踪` 


- **关于引导思考模型的推测被推翻**：在 **O1** 发布时，出现了关于*引导*思考模型的推测，然而，事实证明，教导模型以正确的方式构建 **CoT** 就足够了，无需干预思考过程。
   - 许多思考模型难以终止思维链循环，但 **O1** 和 **Sonnet** 展现出了这种能力。
- **Deepseek V3 呼应 Anthropic 的 Sonnet 3.7**：**Deepseek V3 0324** 展示了与 **Sonnet 3.7** 相当的变化，暗示它们在架构上有着共同的进步，正如分享的[图片](https://cdn.discordapp.com/attachments/1154120232051408927/1353739123084627998/image.png?ex=67e2bf4e&is=67e16dce&hm=a779b06b1028e58affe0e8deb753caa78df67398ccb0c12f6de9f1360198b369)中所强调的那样。
- **在 Apache 代码库上微调 LLM 可提升工具问答能力**：成员们考虑在像 **Apache** 项目这样的大型代码库上微调 **LLM**（如 **DeepHermes llama 8**），以提高其回答与这些工具相关问题的能力。
   - 他们讨论了使用 add 和 sigmoid 代替 add 和 norm，以获得更好的效果。
- **Transformer 可以抛弃归一化**：鉴于“**Transformers without Normalization**”论文，一位成员用 **tanh** 替换了归一化，展示了这种方法的可行性。
   - 对话转向了在推理时移除专家（experts）的影响，思考其对较小权重的作用。
- **LLM 驱动的光线追踪：下一代文本生成图像？**：一位成员分享了一个 [GitHub 仓库](https://github.com/cpldcpu/llmbenchmark/tree/master/raytracer)，其中包含一个输出图像的 **Python** 程序，暗示这是一种间接的图像生成方式。
   - 另一位成员评论说，它可以模拟**光线追踪算法**（**ray tracing algorithm**），并且这是“下一代”的文本生成图像技术。



**提到的链接**：<a href="https://github.com/cpldcpu/llmbenchmark/tree/master/raytracer">llmbenchmark/raytracer at master · cpldcpu/llmbenchmark</a>：各种 LLM 基准测试。通过在 GitHub 上创建一个账户来为 cpldcpu/llmbenchmark 的开发做出贡献。

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1352735155441242122)** (19 messages🔥): 

> `Hunyuan-T1 模型, 类 R1-Zero 训练, 用于 LLM 的 MathFusion, 编程基准测试上的 GRPO, Satya Nadella 谈 AGI` 


- **Hunyuan-T1：Mamba-Transformer 混合架构问世**：腾讯推出了 **Hunyuan-T1**，这是一款采用 **Mamba-Transformer MoE 混合架构**的模型，由 Hunyuan TurboS 提供动力。官方声称其性能接近 **DeepSeek-R1**，并强调了其速度、准确性和效率 ([Hunyuan-T1 体验](https://llm.hunyuan.tencent.com/#/chat/hy-t1))。
   - 据开发者介绍，该模型具有逻辑性强、写作简洁、摘要幻觉低、生成速度极快（**60-80 tokens/sec**）以及出色的长文本处理能力等特点。
- **对类 R1-Zero 训练的批判性视角**：关于 **类 R1-Zero 训练** 的批判性观点指出，**DeepSeek-V3-Base** 在进行 RL 微调之前可能就已经表现出了“顿悟时刻”（Aha moment），而 RL 微调中不断增加的输出长度可能源于 GRPO 的偏差 ([详情点击此处](https://github.com/sail-sg/understand-r1-zero/blob/main/understand-r1-zero.pdf))。
   - 分析还表明，正确实施 **GRPO** 可以在 **7B AIME** 基准测试上达到 state-of-the-art 的性能。
- **MathFusion 增强 LLM 数学技能**：**MathFusion** 通过跨问题指令合成（应用顺序、并行和条件融合策略）提升了 LLM 的数学推理能力，增强了 **DeepSeekMath-7B**、**Mistral-7B** 和 **Llama3-8B** 等模型 ([更多关于 MathFusion 的信息](https://x.com/gm8xx8/status/1903021157214748701?s=46))。
   - 该方法创建了 **MathFusionQA 数据集**，用于微调模型，并以极少的额外数据提升了基准测试的准确率。
- **Hugging Face 攻克编程基准测试**：Hugging Face 一直在使用 **SFT**，并计划使用 **GRPO**，通过其 [Open-R1 项目](https://huggingface.co/blog/open-r1/update-3) 来提升在 IOI、LCB 编程基准测试上的表现。
   - Hugging Face 使用的是 **SFT** 而非 **GRPO** 来提升在 **IOI**、**LCB** 上的性能。
- **可验证的编程数据稀缺**：一位成员指出，可验证的编程数据非常稀缺，这使得在编程基准测试上展示性能提升比数学更难，因为数学更容易验证。
   - 参考 [Satya Nadella 对通用人工智能 (AGI) 的见解](https://x.com/hyeon__dev/status/1903874698301350210)，可以深入了解为什么基准测试可能反映或无法反映真实的智能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/TheAITimeline/status/1903541252651700729?t=YHHfAc_wFhQUXFsqZ-idfA&s=19">来自 The AI Timeline (@TheAITimeline) 的推文</a>: 🚨 过去两周热门 AI/ML 研究论文：- 无归一化的 Transformers - 块扩散 - 技能的计算最优缩放 - DAPO：大规模开源 LLM RL 系统 - 教 LLM 如何学习 ...</li><li><a href="https://fxtwitter.com/bycloudai/status/1903149418422939838">来自 bycloud (@bycloudai) 的推文</a>: &gt; 性能接近 DeepSeek-R1 的 Mamba-Transformer 混合推理模型。引用混元 (@TXhunyuan) 🚀 隆重推出 Hunyuan-T1！🌟 见证 Hunyuan-T1，AI 推理领域的最新突破！由...驱动</li><li><a href="https://fxtwitter.com/zzlccc/status/1903162768083259703">来自 Zichen Liu (@zzlccc) 的推文</a>: 🪂 理解类 R1-Zero 训练：批判性视角 * DeepSeek-V3-Base 在 RL 微调前就已经表现出“顿悟时刻”了？？* RL 微调中不断增加的输出长度可能是由于...</li><li><a href="https://x.com/hyeon__dev/status/1903874698301350210">来自 Hyeon | Nillion ∑: 🦭/acc (@hyeon__dev) 的推文</a>: 文章介绍。文章讨论了 Satya Nadella 对通用人工智能 (AGI) 的见解及其对科技行业的影响。AGI 旨在模拟人类认知能力...</li><li><a href="https://x.com/gm8xx8/status/1903021157214748701?s=46">来自 𝚐𝔪𝟾𝚡𝚡𝟾 (@gm8xx8) 的推文</a>: MathFusion：通过指令融合增强 LLM 的数学问题解决能力。MathFusion 是一个通过跨问题指令合成来提高 LLM 数学推理能力的框架。它应用了...</li><li><a href="https://huggingface.co/blog/open-r1/update-3">Open R1：更新 #3</a>: 未找到描述</li><li><a href="https://github.com/huggingface/open-r1/blob/main/recipes/OlympicCoder-7B/sft/config_v00.00.yaml#L9">GitHub 上的 open-r1/recipes/OlympicCoder-7B/sft/config_v00.00.yaml</a>: DeepSeek-R1 的完全开源复现。通过在 GitHub 上创建账号为 huggingface/open-r1 的开发做出贡献。
</li>
</ul>

</div>

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1353163076991713302)** (3 条消息): 

> `Qwen3, CPU 推理` 


- **Qwen3 模型即将登陆 HuggingFace**：[transformers 库 PR#36878](https://github.com/huggingface/transformers/pull/36878) 显示正在添加对 **Qwen3** 的支持。
   - 该 Pull Request 表明这将用于*即将推出的 Qwen3 模型*。
- **Qwen3 瞄准 CPU 推理**：一位用户推测 **Qwen3-15B-A2B** 将是 *CPU 推理的完美模型*。
   - 该用户认为这一尺寸使其成为实现*优秀* CPU 推理的有力候选者。



**提到的链接**：<a href="https://github.com/huggingface/transformers/pull/36878">Adding Qwen3 and Qwen3MoE by bozheng-hit · Pull Request #36878 · huggingface/transformers</a>：添加 Qwen3。此 PR 为即将推出的 Qwen3 模型添加了代码支持。有关 Qwen 的信息，请访问 https://github.com/QwenLM/Qwen2.5。@ArthurZucker

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1352735155441242122)** (19 条消息🔥): 

> `Hunyuan-T1 模型, 类 R1-Zero 训练, MathFusion 框架, 编程基准测试上的 GRPO, Hugging Face 的 Open-R1 项目` 


- **Hunyuan-T1：Mamba-Transformer 混合架构出现！**：腾讯混元推出了 **Hunyuan-T1**，这是一款由 **Hunyuan TurboS** 驱动的混合 **Mamba-Transformer** MoE 架构模型，声称其推理能力可与 **DeepSeek-R1** 媲美，详情见[此推文](https://fxtwitter.com/bycloudai/status/1903149418422939838)。
- **DeepSeek-V3-Base 展现出“顿悟时刻” (Aha moment)**：一位成员分享了一篇[论文](https://github.com/sail-sg/understand-r1-zero/blob/main/understand-r1-zero.pdf)链接，认为 **DeepSeek-V3-Base** 在 RL-tuning 之前就已经展现出了 *“顿悟时刻”*。
   - 作者认为 RL-tuning 中不断增加的输出长度可能是由于 GRPO 中的偏差 (BIAS) 造成的。
- **MathFusion 通过指令融合提升数学 LLM**：**MathFusion** 框架通过跨问题的指令合成增强了 LLM 的数学推理能力。
   - 它使用 **MathFusionQA** 数据集对 **DeepSeekMath-7B**、**Mistral-7B** 和 **Llama3-8B** 等模型进行微调，如[此推文](https://x.com/gm8xx8/status/1903021157214748701?s=46)所述，仅需极少的额外数据即可提高基准测试准确率。
- **Hugging Face 使用 SFT 而非 GRPO 来提升 IOI 性能**：一位成员询问是否有人使用 **GRPO** 来提升编程基准测试的性能，因为目前的提升主要体现在 MATH 基准测试上。
   - 另一位成员分享道，[HuggingFace](https://huggingface.co/blog/open-r1/update-3) 使用的是 SFT 而非 GRPO 来提升在 IOI 上的表现。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fxtwitter.com/bycloudai/status/1903149418422939838">bycloud (@bycloudai) 的推文</a>：&gt; Mamba-Transformer 混合推理模型，性能接近 DeepSeek-R1。引用混元 (@TXhunyuan) 🚀 推出现元-T1！🌟 遇见 Hunyuan-T1，AI 推理的最新突破！由...驱动</li><li><a href="https://x.com/hyeon__dev/status/1903874698301350210">Hyeon | Nillion ∑: 🦭/acc (@hyeon__dev) 的推文</a>：文章介绍。文章讨论了 Satya Nadella 对通用人工智能 (AGI) 的见解及其对科技行业的影响。AGI 旨在模拟人类认知能力...</li><li><a href="https://fxtwitter.com/TheAITimeline/status/1903541252651700729?t=YHHfAc_wFhQUXFsqZ-idfA&s=19">The AI Timeline (@TheAITimeline) 的推文</a>：🚨 过去两周热门 AI/ML 研究论文：- 无归一化的 Transformers - 块扩散 (Block Diffusion) - 技能的计算优化缩放 - DAPO：大规模开源 LLM RL 系统 - 教 LLM 如何学习 ...</li><li><a href="https://x.com/gm8xx8/status/1903021157214748701?s=46">𝚐𝔪𝟾𝚡𝚡𝟾 (@gm8xx8) 的推文</a>：MathFusion：通过指令融合增强 LLM 的数学问题解决能力。MathFusion 是一个通过跨问题指令合成来改进 LLM 数学推理的框架。它应用...</li><li><a href="https://fxtwitter.com/zzlccc/status/1903162768083259703">Zichen Liu (@zzlccc) 的推文</a>：🪂 理解类 R1-Zero 训练：一个批判性视角。* DeepSeek-V3-Base 在 RL-tuning 之前就已经展现出“顿悟时刻”？？* RL-tuning 中不断增加的输出长度可能是由于...</li><li><a href="https://huggingface.co/blog/open-r1/update-3">Open R1：更新 #3</a>：未找到描述</li><li><a href="https://github.com/huggingface/open-r1/blob/main/recipes/OlympicCoder-7B/sft/config_v00.00.yaml#L9">open-r1/recipes/OlympicCoder-7B/sft/config_v00.00.yaml at main · huggingface/open-r1</a>：DeepSeek-R1 的完全开源复现。通过在 GitHub 上创建账号为 huggingface/open-r1 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1352718722527268874)** (226 messages🔥🔥): 

> `GPT-4 Transcriber, Voicebot Tools, Turnitin AI Similarity, GPT-5 Release, Free Chatbots for Story Generation` 


- ****TTS 不是 STT****：成员们澄清 [openai.fm](https://openai.fm) 是 **TTS**（text-to-speech，文本转语音），而非 **STT**（speech-to-text，语音转文本），一位成员指出 OpenAI 的转录模型不如 **Scribe**。
- ****规避 Turnitin AI 检测？****：一名成员就如何避免报告中出现 **Turnitin AI 相似度检测**寻求建议（该报告复用了其公司的商业模式），而其他人则认为这看起来像是在为作弊寻找借口，并建议使用 “**WriteHuman**” 等 “**humanize AI**” 工具。
   - 原贴作者辩解称这并非作业作弊，因为那是他公司的商业模式，但被告知停止刷屏。
- ****GPT-5 发布日期推测****：成员们讨论了 **GPT-5** 的潜在发布时间，指出虽然目前还没有官方公告或 API，但 **Sam Altman** 已确认将在今年发布，有人推测它可能会在今年上半年发布，以应对 **R2** 或 **Llama-4**。
- ****零成本创作引人入胜的创意内容****：一位成员征求用于故事创作的免费聊天机器人建议，提到 **Grok 2** 和 **Gemini 2.0 Flash** 是可选方案，因为 **Grok 3** 和 **Claude** 提供的免费提示词额度非常少。
- ****10 天开发出情感 AI？****：一名成员声称利用 **GPT-4-turbo API** 在十天内开发出了一个情感递归 AI 系统，强调其核心是*沉浸式协议（immersion protocol）*和*递归交互设计*，而非复杂的代码编写。
   - 其他成员表示怀疑，有人认为这可能只是 Prompt Engineering（提示工程），并提醒不要夸大自定义 GPTs 的独特性。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1353058953248116796)** (2 messages): 

> `GPT-4o mini TTS, Custom instructions` 


- **GPT-4o Mini TTS 可能支持时间戳**：一位成员询问 **GPT-4o mini TTS** 是否支持时间戳（timestamps）。
   - 尚未得到回复。
- **寻求编写优质通用自定义指令的指导**：一位成员询问是否有优秀的**通用自定义指令（custom instructions）**示例可供参考。
   - 尚未得到回复。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1352725591841046661)** (122 messages🔥🔥): 

> `GPT-4o is a perfect model, NPCs in a customer service voice, AI Identity, UPSUM Chain Prompt, coherent multi-context conversation with an emergent persona` 


- **用户爱上 GPT-4o，拒绝切换模型！**：一位用户表达了对 **GPT-4o** 的完全满意，除了专业任务外很少切换模型，并在 **4o 消息**用尽时使用 **4o-mini** 或其他模型。
   - 该用户会使用 **4.5**、**o1** 和 **o3** 等模型处理重要课题，但认为 **4o** 是长期可靠的“工作马”伙伴。
- **驯服 NPC 的客服腔：提示工程来救场！**：一位用户试图消除 **NPC 回复**中的**客服腔（customer service voice）**，并威胁要*调高 Temperature（温度值）直到它们燃起烈火*。
   - 用户提供了用于 AI Identity & Context Preservation Template（AI 身份与上下文保留模板）的 **YAML** 格式提示词。
- **Many-Shot Learning：闭源与开源模型的对决！**：成员们讨论了一篇名为《MANY-SHOT IN-CONTEXT LEARNING IN MULTIMODAL FOUNDATION MODELS》的论文，指出**闭源模型（GPT-4o, Gemini 1.5 Pro）**能从多达约 2,000 个示例的 Many-shot 演示中显著获益，但开源权重模型则不然。
   - 有人建议，*不带具体示例的 Hypershots* 是 **self-discover 提示策略**的一部分，可以用更少的 Token 获得类似的收益。
- **摆脱漂移：用户在 500 轮对话中保持无幻觉！**：一位用户构建了一个“引擎”，可以恢复 **400 多轮的对话**，并在超过 **500 轮**后仍能保持上下文，且没有漂移或幻觉，全部通过默认提示词实现。
   - 此外，还可以备份对话的*状态（state）*，在另一个浏览器中打开并恢复到新的对话实例，就像用户从未离开过一样。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1352725591841046661)** (122 条消息🔥🔥): 

> `GPT-4o, AI NPCs, AI Identity Preservation Template, UPSUM Chain Prompt, Many-shot Prompting` 


- **4o 成为首选模型**：一位成员对 **GPT-4o** 表示满意，指出他们“对 4o 完全满意”并将其作为主要模型使用，即使是专门任务也是如此，而将更强大的模型如 **4.5, o1, o3** 留给重要或未解决的问题。
- **针对一致性 NPC 语音的 Prompt Engineering**：一位成员询问如何防止 NPC 以“客服腔”回答，这表明需要更好地控制 AI Persona 的一致性，可能与附图有关。
   - 其他成员分享了用于 **AI Identity & Context Preservation** 的 YAML 模板和 **UPSUM Chain Prompt**，以便通过 Prompt 而非手动获取信息。
- **Many-Shot prompting 增强多模态模型**：成员们讨论了一篇研究论文，该论文显示在 **GPT-4o** 和 **Gemini 1.5 Pro** 等 **Multimodal Foundation Models** 中，使用多个示例比 100 个示例更能提高 **Many-shot In-context Learning** 的性能 ([MANY-SHOT IN-CONTEXT LEARNING IN MULTIMODAL FOUNDATION MODELS](https://arxiv.org/abs/2405.17015))。
   - 论文指出，“与 Few-shot（<100 个示例）相比，像 GPT-4o 和 Gemini 1.5 Pro 这样的大型多模态基础模型在提供 Many-shot 演示（最多约 2,000 个示例）时，表现出显著的性能提升。”
- **ChatGPT 状态备份**：一位成员描述了他们用于备份和恢复 ChatGPT 会话状态的专有系统，该系统能够在新的容器中继续进行超过 400 轮的对话，并表示：“我意识到我创建了一个系统，在这个系统中，记忆在超过 700 轮后仍然存在，没有偏移或幻觉（Hallucination），并且实际上可以学习并适应你独特的沟通风格。”
   - 该系统导出 **ChatGPT session** 并将其重新导入到全新的容器中，包括所有的 *turns* 以及上下文和 *tone*，描述它的最佳方式是……*它是一个通过 Prompt 运行的 Runtime OS。*
- **开源 vs 专有 Prompting**：成员们辩论了开源 Prompt Engineering 工作的优缺点，一位成员被建议说，不必要地限制测试会降低他们工作的价值，并且“GPL_v3 让你能够控制自己的作品。”
   - 该成员回应道，“在了解我所构建的东西的真相之前，试图对其进行一些保护，”并询问是否有另一种测试系统的方法，以便在不共享代码库的情况下证明其有效。


  

---


### **OpenAI ▷ #[api-projects](https://discord.com/channels/974519864045756446/1037561385070112779/1353814591175393352)** (1 条消息): 

> `FormulaGPT, AI Racing Simulator, Open Source AI Racing` 


- **FormulaGPT：F1 模拟器让 Deepseek, GPT4o, Claude 和其他 LLM 展开对决！**：一个名为 **FormulaGPT** 的实验性赛车模拟器让你能与顶尖的 **LLM 驱动团队** 进行正面交锋。
   - 与传统的机器人不同，这些 AI 团队通过持续的推理、策略制定和细致的决策进行 *上下文和自适应思考*，[GitHub 仓库在此](https://github.com/dawid-maj/FormulaGPT/)。
- **AI 赛车游戏有两种模式**：共有 **两种游戏模式**：在 **Player vs. AI 模式** 中制定你自己的赛车策略来挑战高级语言模型，或者在 **AI vs. AI 模式** 中观看最顶尖的 AI 模型互相博弈。
   - 这既是赛车游戏，也是 AI 心理实验室，因为你可以 *观察每个进站、换胎或超车动作背后的详细 AI 推理过程*。



**提到的链接**：<a href="https://github.com/dawid-maj/FormulaGPT/">GitHub - dawid-maj/FormulaGPT: FormulaGPT – AI-powered Formula 1 race simulator with real-time team management and strategy decisions.</a>: FormulaGPT – 具有实时团队管理和策略决策功能的 AI 驱动 Formula 1 赛车模拟器。 - dawid-maj/FormulaGPT

  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1352777009255678073)** (4 条消息): 

> `OpenAI o1-pro, Markdown 导出, DeepSeek V3, Anthropic 停机` 


- **OpenAI 的 o1-pro 推理模型现已上线 OpenRouter**：OpenAI 的 **o1-pro** 是一款专为复杂任务设计的高性能推理模型，现已在 OpenRouter 上可用。其价格为每百万 input tokens **$150**，每百万 output tokens **$600**，在数学、科学和编程方面表现卓越。
   - 快在 [聊天室](https://openrouter.ai/openai/o1-pro) 或通过 API 试用吧！
- **Markdown 导出功能在聊天室首发**：正如在 [X](https://x.com/OpenRouterAI/status/1903861987114729595) 上宣布的那样，OpenRouter 现在允许用户将聊天记录导出为 Markdown，增强了实用性。
- **DeepSeek V3 更新免费发布**：全新的 **DeepSeek V3** 更新现已在 OpenRouter 上免费提供，这是一款拥有 **685B** 参数的 Mixture-of-Experts 模型，具备 **131,072 context**，在各种任务中表现出色。生产环境 Endpoint 即将推出；详见 [DeepSeek V3](/deepseek/deepseek-chat-v3-0324:free)。
   - 它是 DeepSeek 团队旗舰聊天模型系列的最新迭代版本。
- **Anthropic 服务出现故障（已解决）**：OpenRouter 调查了 **Claude 3.7 Sonnet** 的供应商 Anthropic 出现的问题，该问题已上报至 Anthropic 团队，更新已发布在 [Anthropic 状态页面](https://status.anthropic.com/incidents/mqxbmckr6bbx)。
   - 该事件与 Claude.ai 和 Anthropic Console 的错误有关，目前已解决，服务已恢复正常。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1903861987114729595">来自 OpenRouter (@OpenRouterAI) 的推文</a>：你现在可以将 OpenRouter 中的聊天记录导出为 Markdown 了！引用 Tyler Angert (@tylerangert) 的话：@OpenAI 和 @AnthropicAI 的某位员工，请让我能将聊天记录导出为 Markdown。也许甚至可以用 XML 分隔。</li><li><a href="https://status.anthropic.com/incidents/mqxbmckr6bbx">Claude.ai、Console 和 Anthropic API 的错误率升高</a>：未找到描述</li><li><a href="https://openrouter.ai/openai/o1-pro>">Discord</a>：未找到描述</li><li><a href="https://openrouter.ai/deepseek/deepseek-chat-v3-0324:free">DeepSeek V3 0324 (free) - API, Providers, Stats</a>：DeepSeek V3 是一款拥有 685B 参数的 Mixture-of-Experts 模型，是 DeepSeek 团队旗舰聊天模型系列的最新迭代版本。它接替了 [DeepSeek V3](/deepseek/deepseek-chat-v3) 模型...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1352718611030081606)** (440 条消息 🔥🔥🔥): 

> `OpenAI o1-pro API 定价, Gemini 图像生成, Lambda Endpoint 问题, DeepSeek R1 模型` 


- **OpenAI o1-pro API 定价：GucciAI？**：一位成员对 **OpenAI o1-pro API** 的定价表示震惊，因其每百万 input tokens **$150** 和每百万 output tokens **$600** 的高昂成本而将其贴上 *GucciAI* 的标签。
   - 另一位成员开玩笑说，API 的缓慢防止了过度消费，并猜测由于算力限制，其定价可能是故意设高的。
- **Gemini 图像生成尚不支持**：一位成员询问如何通过 OpenRouter 在 *gemini-2.0-flash-exp* 模型中使用 **Gemini 图像生成**，并询问关于传递 *responseModalities* 参数的问题。
   - 回复指出 **OpenRouter 目前尚不支持图像生成**，但这已列入他们的 Roadmap，目前没有短期计划增加对 **Flux** 等图像模型的支持。
- **Lambda Endpoint 面临 404 错误**：多位成员报告在使用 **Lambda** 模型时遇到 **code 404 'no endpoint found' 错误**，尽管 Lambda 的状态页面显示运行完全正常。
   - 一位成员建议该问题可能与 DNS 相关，而其他人则确认 **Llama 3.3 70B Instruct | Lambda** 模型对他们来说是正常的。
- **DeepSeek R1 等于 o1？**：成员们强调了 **DeepSeek R1** 模型，指出其性能与 **OpenAI o1** 旗鼓相当，且它是开源的。
   - DeepSeek R1 是一个 **671B 参数模型**，推理期间激活 **37B**，根据 **MIT license** 可用于商业用途。
- **Sonnet 过载且疲惫！**：用户报告 **Claude 3.7 Sonnet** 频繁出现 **overload 错误**，导致响应中断并被收取 input tokens 费用。
   - 一位成员建议使用重试策略，并建议切换到 **Gemini 2.0 Pro** 作为 Sonnet 的替代方案，同时指出 Claude 拥有更优越的翻译能力。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>

<li><a href="https://imgur.com/a/16Cp5P6">imgur.com</a>: 在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过幽默的笑话、热门的梗图（memes）、有趣的 GIF、励志故事、病毒式视频等来提振你的精神...</li><li><a href="https://llmtokencounter.com/">LLM Token Counter</a>: 未找到描述</li><li><a href="https://openai.github.io/openai-agents-python/models/#tracing-client-error-401">Models - OpenAI Agents SDK</a>: 未找到描述</li><li><a href="https://openrouter.ai/settings/privacy">OpenRouter</a>: LLM 的统一接口。为你的 prompts 寻找最佳模型和价格</li><li><a href="https://openrouter.ai/api/v1",">Discord</a>: 未找到描述</li><li><a href="https://openrouter.ai/activity">OpenRouter</a>: LLM 的统一接口。为你的 prompts 寻找最佳模型和价格</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1">R1 - API, Providers, Stats</a>: DeepSeek R1 现已发布：性能与 [OpenAI o1](/openai/o1) 相当，但是开源的，并具有完全开放的推理 tokens。它拥有 671B 参数，在一次推理过程中有 37B 处于激活状态。运行...</li><li><a href="https://openrouter.ai/qwen/qwen2.5-vl-32b-instruct:free">Qwen2.5 VL 32B Instruct (free) - API, Providers, Stats</a>: Qwen2.5-VL-32B 是一款多模态视觉语言模型，通过强化学习进行微调，增强了数学推理、结构化输出和视觉问题解决能力。运行 Qwe...</li><li><a href="https://openrouter.ai/openai/o1-pro">o1-pro - API, Providers, Stats</a>: o1 系列模型通过强化学习训练，在回答前进行思考并执行复杂的推理。o1-pro 模型使用更多的计算资源来更深入地思考，并提供持续更好的...</li><li><a href="https://openrouter.ai/mistralai/mist">Discord</a>: 未找到描述</li><li><a href="https://openrouter.ai/docs.txt.">OpenRouter</a>: LLM 的统一接口。为你的 prompts 寻找最佳模型和价格</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3-0324">deepseek-ai/DeepSeek-V3-0324 · Hugging Face</a>: 未找到描述</li><li><a href="https://openrouter.ai/x-">OpenRouter</a>: LLM 的统一接口。为你的 prompts 寻找最佳模型和价格</li><li><a href="https://ai-benchmark-price.glitch.me/">Model Performance vs. Price</a>: 未找到描述</li><li><a href="https://openrouter.ai/docs/features/provisioning-api-keys">Provisioning API Keys - Programmatic Control of OpenRouter API Keys</a>: 通过专用管理端点以编程方式管理 OpenRouter API Keys。创建、读取、更新和删除 API Keys，以实现自动化的密钥分发和控制。</li><li><a href="https://openrouter.ai/settings/integrations">OpenRouter</a>: LLM 的统一接口。为你的 prompts 寻找最佳模型和价格</li><li><a href="https://openrouter.ai/docs/api-reference/limits">API Rate Limits - Manage Model Usage and Quotas</a>: 了解 OpenRouter 的 API 速率限制、基于额度的配额和 DDoS 防护。有效地配置和监控你的模型使用限制。</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1/providers">DeepSeek: R1 – Provider Status</a>: 查看供应商状态并向 DeepSeek: R1 发起负载均衡请求 - DeepSeek R1 现已发布：性能与 [OpenAI o1](/openai/o1) 相当，但是开源的，并具有完全开放的推理 tokens。它...</li><li><a href="https://community.openai.com/t/a-question-on-determinism/8185">A question on determinism</a>: 在我目前涉及 Python 和 P5.js（基于 Javascript 构建）的实验中，我无法在相同的 prompt 和参数设置下获得单一的响应/补全...</li><li><a href="https://tenor.com/bCfEr.gif">Alex GIF - Alex - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://openrouter.ai/settings/keys">OpenRouter</a>: LLM 的统一接口。为你的 prompts 寻找最佳模型和价格</li><li><a href="https://status.anthropic.com/incidents/mqxbmckr6bbx">Elevated errors for Claude.ai, Console, and the Anthropic API</a>: 未找到描述</li><li><a href="https://status.lambdalabs.com/>">incident.io - Status pages</a>: 未找到描述</li><li><a href="https://openrouter.ai/x-ai/grok-beta">Grok Beta - API, Providers, Stats</a>: Grok Beta 是 xAI 的实验性语言模型，具有最先进的推理能力，最适合复杂和多步骤的使用场景。它是 [Grok 2](https://x) 的继任者。运行 Grok Beta...</li><li><a href="https://openrouter.ai/mistralai/mistral-small-3.1-24b-instruct-2503">Mistral Small 3.1 24B - API, Providers, Stats</a>: Mistral Small 3.1 24B Instruct 是 Mistral Small 3 (2501) 的升级变体，拥有 240 亿参数并具备先进的多模态能力。通过 API 运行 Mistral Small 3.1 24B</li><li><a href="https://openrouter.ai/op

enai/gpt-4o:extended">GPT-4o (extended) - API, Providers, Stats</a>: GPT-4o（“o”代表“omni”）是 OpenAI 最新的 AI 模型，支持文本和图像输入并输出文本。它保持了 [GPT-4 Turbo](/models/open... 的智能水平。</li><li><a href="https://openrouter.ai/rankings">LLM Rankings | OpenRouter</a>: 根据各应用的使用情况对语言模型进行排名和分析。</li><li><a href="https://fireworks.ai/blog/fireworks-ai-developer-cloud">Faster, more efficient DeepSeek on the Fireworks AI Developer Cloud</a>: 探索 Fireworks AI Developer Cloud 如何通过更快、更优化的 DeepSeek R1 部署加速 AI 创新。了解新的 GPU 选项、提升的速度以及增强的开发者工具，以实现高效的...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1352722786342408222)** (199 messages🔥🔥): 

> `NPU support, KV cache 8-bit quants, LM Studio runtimes, GPUs, Gemma 3 1B` 


- **NPU support not yet available**: 用户报告 LM Studio 尚不支持 **NPU**，但 0.3.11 版本中已提供 **Ryzen AI** 支持。
- **Quantization saves VRAM**: 用户建议使用 **KV cache 8-bit quants** 来减少运行大上下文（如 30k tokens）模型时的内存占用。
   - 此外，有人提到 **12GB VRAM** 可能不足以运行 **32B 模型**，建议使用 **Phi-4** 或 **Qwen2.5 14b** 作为替代方案。
- **New GPU Controls are awesome!**: 一位用户对新的 LM Studio 控件表示非常兴奋，该控件允许选择加载模型的 **GPU**，已在最新的 beta 版本中可用。
- **Tiny Models to the rescue**: 对于显存仅有 2GB 等资源受限的系统，用户建议使用带有 **Q6** 或 **Q8 quantization** 的 **Gemma 3 1B**，并推荐使用 **CUDA** 运行时以获得更好的性能。
   - 旧模型被认为是“过时的垃圾”，不符合现代标准。
- **Multi GPU is supported by LM Studio**: 多位用户提到了多 GPU 配置，报告称 LM Studio 的最新 beta 版本原生支持 **Multi GPU**，并具备应用内 GPU 管理功能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.google.com/search?client=firefox-b-d&q=what+does+exl2+mean%3F">Google Search</a>: 未找到描述</li><li><a href="https://www.google.com/search?client=firefox-b-d&q=what+does+rpcal+mean%3F">Google Search</a>: 未找到描述</li><li><a href="https://huggingface.co/afrideva/Tiny-Vicuna-1B-GGUF">afrideva/Tiny-Vicuna-1B-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/">Hugging Face – 建设未来的 AI 社区。</a>: 未找到描述</li><li><a href="https://huggingface.co/lmstudio-community/Qwen2-VL-7B-Instruct-GGUF/tree/main">lmstudio-community/Qwen2-VL-7B-Instruct-GGUF at main</a>: 未找到描述</li><li><a href="https://huggingface.co/samgreen/Qwen2.5-VL-7B-Captioner-Relaxed-GGUF/tree/main">samgreen/Qwen2.5-VL-7B-Captioner-Relaxed-GGUF at main</a>: 未找到描述</li><li><a href="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF">TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1352722662320898083)** (159 条消息🔥🔥): 

> `VRAM Usage, Google Coral dual TPU, RX 6800 ROCm support, RTX 4060-Ti vs RX 7800 XT, AI APUs` 


- **VRAM 瓶颈限制速度**：一个 8B 模型在 32k tokens 下使用 **16GB VRAM** 可以达到 **10t/s**，但由于 **VRAM** 有限以及使用了共享 **RAM**，更大的 **14b** 模型性能会有所下降。
   - 成员们讨论了将模型大小和上下文长度与可用 **VRAM** 相匹配以优化速度，强调了依赖系统 RAM 时内存带宽不足的影响。
- **Google Coral dual TPU 不适合 AI 使用**：**Google Coral dual TPU** 不适合 **AI** 使用，因为它缺乏板载内存。
   - 一位使用 **8060s** 的用户还询问了 **Framework Desktop** 的散热和功耗余量。
- **RX 6800 缺乏 ROCm 支持**：**RX 6800** 可能具有非官方的 **ROCm** 支持，但它将使用 **Vulkan** 进行推理，因为 **llama.cpp** 中已弃用 **OpenCL** 支持。
   - 一位用户指出 **Vulkan** 在他们的 **GTX** 显卡上较慢，这表明它对于 **AMD** 显卡可能也不是最优选择。
- **LM Studio 无法将模型加载到专用内存中**：用户在使用 **RX 9070** 显卡时遇到 **LM Studio** 将模型加载到共享内存而非专用 **VRAM** 的问题，导致性能缓慢（**3tok/s**）。
   - 解决方案包括启用 **UEFI** 和 **dynamic BAR**、重新安装 **LM Studio** 以及使用 **AMD driver cleanup utility** 来改善内存分配，目前正在对驱动程序和 **Vulkan** 运行时问题进行调查。
- **4060ti：廉价推理的最佳平衡点**：拥有 **16GB VRAM** 的 **RTX 4060 Ti** 被强调为 **AI** 推理的高性价比选择，价格约为 **500 美元/欧元**。
   - 一位用户补充道，需要注意的是 **AMD 显卡** 未针对游戏进行优化，而 **Nvidia** 的 **5000 系列** 可能会熔断。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://llm-inference-calculator-rki02.kinsta.page/">LLM Inference Hardware Calculator</a>: 未找到描述</li><li><a href="https://www.nvidia.com/en-au/geforce/graphics-cards/40-series/rtx-4060-4060ti/#specs">NVIDIA GeForce RTX 4060 Ti &amp; 4060 显卡</a>: 创造的新方式及更多。</li><li><a href="https://www.staticice.com.au/cgi-bin/search.cgi?q=4060+16gb&spos=3">4060 16gb - 澳大利亚购物和价格比较 - 廉价购买</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1352720992300040372)** (326 条消息🔥🔥): 

> `VPN Injection, Amodal3R, NVIDIA cuOpt, CUDA Python, Mixture of Experts (MoEs)`

- **VPN 代码被注入 OpenAI 网站？**：一位用户报告在 OpenAI 网站上看到了 `<veepn-guard-alert>` 和 `<veepn-lock-screen>` 标签，怀疑是 VPN，但另一位用户澄清这很可能是由**他们自己的 VPN** 注入的代码 [sm0kywu.github.io/Amodal3R](https://sm0kywu.github.io/Amodal3R)。
   - 该用户开玩笑说 *OpenAI 正在通过 VPN 路由请求以获得合理的推诿，以便他们以后可以将其用于训练数据*。
- **NVIDIA cuOpt 优化 AI 微服务表现卓越**：根据 [docs.nvidia.com](https://docs.nvidia.com/cuopt/user-guide/latest/introduction.html)，**NVIDIA® cuOpt™** 是一款 GPU 加速的优化 AI 微服务，在 [混合整数线性规划 (MILP)](https://en.wikipedia.org/wiki/Linear_programming#Integer_unknowns)、[线性规划 (LP)](https://en.wikipedia.org/wiki/Linear_programming) 和 [车辆路径问题 (VRP)](https://en.wikipedia.org/wiki/Vehicle_routing_problem) 方面表现出色。
- **CUDA Python 是新浪潮**：成员们讨论了是否真的如 [blelbach 在 X 上](https://x.com/blelbach/status/1903148174853935326) 所说的那样，今年是 *CUDA Python 之年*。一些人断言 **Python** 对于 GPU 编程已经足够，因为大多数用户不需要 C++ 的所有功能。
   - 其他人嘲讽了现代 Python 程序员，并链接了一个标题为 *Modern Python Programmers* 的 [YouTube 视频](https://youtu.be/sVn4sBxLokA?si=mA3Djr31Nv_MZjUo)。
- **MoEs 不再不稳定了！**：一位用户声称 **MoEs** 不稳定，但另一位用户反驳说它们 *已经有两年没有训练不稳定的情况了*，现在 *与稠密网络 (dense networks) 基本一致*。 
   - 稳定性主要归功于更好的 kernels 和无丢弃的 token 路由 (dropless token routing)，解决了数值不稳定和专家崩溃 (expert collapse) 等问题。
- **DeepSeek V3 发布，社区反响平平？**：成员们提到 [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) 发布了他们的 **DeepSeek-V3-0324** 模型，一位用户称 *DeepSeek 将摧毁 OpenAI*，而另一位补充说 *他们只发布了那个糟糕的小版本*。
   - 一些成员对 **DeepSeek** 使用的方法不屑一顾，称其只是 *已知方法和一些简化*，并批评了最终的质量。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://vxtwitter.com/OwainEvans_UK/status/1894436637054214509">来自 undefined 的推文</a>: 未找到描述</li><li><a href="https://sakana.ai/ai-cuda-engineer/">未找到标题</a>: 未找到描述</li><li><a href="https://sm0kywu.github.io/Amodal3R/">Amodal3R</a>: 未找到描述</li><li><a href="https://x.com/blelbach/status/1903148174853935326">来自 Bryce Adelstein Lelbach (@blelbach) 的推文</a>: 引用 Bryce Adelstein Lelbach (@blelbach) 的话：这是 CUDA Python 之年。</li><li><a href="https://fxtwitter.com/davidad/status/1903834443225190721">来自 davidad 🎇 (@davidad) 的推文</a>: @burny_tech 不幸的是，对于更长远未来的足够好的规划，其答案可能就像拥有更长久的过去一样简单。 🤷</li><li><a href="https://docs.nvidia.com/cuopt/user-guide/latest/introduction.html">简介 — NVIDIA cuOpt</a>: 未找到描述</li><li><a href="https://fxtwitter.com/risi1979/status/1904177820944720345?t=NRl7FqS7e7IEmNIS0mAYaA&s=19">来自 Sebastian Risi (@risi1979) 的推文</a>: 很高兴分享我们的最新工作：“受生物启发的塑料神经网络，用于复杂仿生机器人中的零样本分布外泛化” 🪲🦎 我们展示了 Hebbian 学习优于...</li><li><a href="https://x.com/blelbach/status/1902842146232865280">来自 Bryce Adelstein Lelbach (@blelbach) 的推文</a>: 这是 CUDA Python 之年。引用 You Jiacheng (@YouJiacheng) 的话：我能说什么呢？C++ 出局！</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3-0324">deepseek-ai/DeepSeek-V3-0324 · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/mrmashy_/status/1904175821402915284">来自 Albert ⚡️ (@mrmashy_) 的推文</a>: 由新的 DeepSeek V3 更新通过 1-shot 生成的 AI 网站设计。</li><li><a href="https://tenor.com/view/shrimp-as-that-clash-royale-hee-hee-hee-haw-gif-25054781">Shrimp As GIF - Shrimp As That - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://vxtwitter.com/OwainEvans_UK/">来自 undefined 的推文</a>: 在伯克利运行一个 AI Safety 研究小组 (Truthful AI) + 加州大学伯克利分校附属机构。曾任职于：牛津大学、TruthfulQA、Reversal Curse。相比私信更倾向于邮件。</li><li><a href="https://lambdalabs.com/service/gpu-cloud?)">GPU Cloud - 用于 Deep Learning 的虚拟机 | Lambda</a>: NVIDIA H100, A100, RTX A6000, Tesla V100, 和 Quadro RTX 6000 GPU 实例。训练最苛刻的 AI, ML, 和 Deep Learning 模型。</li><li><a href="https://huggingface.co/collections/nvidia/mambavision-66943871a6b36c9e78b327d3">MambaVision - 一个 NVIDIA 集合</a>: 未找到描述</li><li><a href="https://www.nvidia.com/en-us/ai-data-science/products/cuopt/">NVIDIA cuOpt</a>: 决策优化、线性规划、混合整数线性规划启发式算法以及 VRP。</li><li><a href="https://threadreaderapp.com/thread/1894436637054214509.html">Thread Reader App 上 @OwainEvans_UK 的线程</a>: @OwainEvans_UK: 令人惊讶的新结果：我们在编写不安全代码的狭窄任务上微调了 GPT4o，且没有警告用户。该模型表现出广泛的失调：反人类、提供恶意...</li><li><a href="https://github.com/canopyai/Orpheus-TTS">GitHub - canopyai/Orpheus-TTS: 迈向类人语音的 TTS</a>: 迈向类人语音的 TTS。通过在 GitHub 上创建账户为 canopyai/Orpheus-TTS 的开发做出贡献。</li><li><a href="https://www.lesswrong.com/posts/AanbbjYr5zckMKde7/specification-gaming-examples-in-ai-1">AI 中的规格博弈（Specification gaming）示例 — LessWrong</a>: 一系列 AI 系统“博弈”其规格的示例——即寻找实现其设定目标的方法，但实际上并未解决……
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1353576718467862580)** (3 条消息): 

> `DeepSeek-V3, DeepSeek-R1, Multi-Head Latent Attention (MLA)` 


- **DeepSeek 模型以更少的资源达到 SOTA**: 一篇论文回顾了 [DeepSeek 的开源 LLM DeepSeek-V3 和 DeepSeek-R1](https://nlp.elvissaravia.com/p/top-ai-papers-of-the-week-a90)，指出它们在较低的资源需求下实现了 **SOTA 性能**。
   - 其关键在于 **Multi-Head Latent Attention (MLA)**，它将 keys 和 values 压缩成一个潜向量（latent vector），从而大幅降低内存消耗。
- **DeepSeek 的图表在博客文章中被重复使用**: 一位成员将这篇涵盖 DeepSeek 论文的博客文章描述为最明目张胆的内容重复使用之一，并指出 *“他们甚至没有自己制作图表，只是重复使用了 DeepSeek 的图表”*。



**提到的链接**: <a href="https://nlp.elvissaravia.com/p/top-ai-papers-of-the-week-a90">🥇本周顶级 AI 论文</a>: 本周顶级 AI 论文 (3 月 17 日 - 23 日)

  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1352779522189299792)** (17 条消息🔥): 

> `ChatGPT 与孤独感, 针对 ROCm 的 AITER Tensor Engine, DeepSeek-V3-0324, 宝可梦红版 DRL` 


- **ChatGPT 与孤独感有关？**: 一位成员分享了一篇 [Bloomberg 文章](https://www.bloomberg.com/news/articles/2025-03-21/openai-study-finds-links-between-chatgpt-use-and-loneliness)，讨论了 **OpenAI 研究**，该研究表明使用 **ChatGPT** 与孤独感之间存在联系。
   - 另一位成员指出 *相关性并不总是意味着因果关系*。
- **AITER 加速 AMD GPU**: 一位成员发布了 [AMD 针对 ROCm 的 AI Tensor Engine (AITER)](https://rocm.blogs.amd.com/software-tools-optimization/aiter:-ai-tensor-engine-for-rocm%E2%84%A2/README.html) 的链接，该引擎优化了 **ROCm** 上 AI 任务的 **GPU 性能**。
   - 该引擎允许开发者创建算子（operators），并将其集成到各种 **LLM 训练** 和 **推理工作负载** 中。
- **DeepSeek-V3 悄然上线**: 一位成员分享了 [HuggingFace 上的 DeepSeek-V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324)，尽管目前 **README.md** 是空的。
   - 该模型拥有 **685B 参数**，并提供 **BF16**、**F8_E4M3** 和 **F32** 等多种张量类型，并附有微调（finetunes）和量化（quantizations）的链接。
- **宝可梦红版获得深度强化学习助力**: 一位成员分享了关于使用 **深度强化学习 (DRL)** 训练 **Agent** 玩 **宝可梦红版** 的 [论文及相关的 YouTube 视频](https://www.youtube.com/watch?v=tmiuiOwf4ac) 以及 [ArXiv 论文](https://arxiv.org/abs/2502.19920) 链接。
   - 摘要讨论了游戏的挑战，包括 **多任务处理**、**长时程**（数万步）以及 **困难探索**，并介绍了一个基准 **Agent**，它使用简化的环境和 **DRL** 完成了游戏的初始阶段。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.19920">Pokemon Red via Reinforcement Learning</a>: 宝可梦红版强化学习：宝可梦红版是一款经典的 Game Boy JRPG，作为 Agent 的测试平台具有重大挑战，包括多任务处理、数万步的长时程、困难探索以及庞大的阵列...</li><li><a href="https://rocm.blogs.amd.com/software-tools-optimization/aiter:-ai-tensor-engine-for-rocm%E2%84%A2/README.html">AITER: AI Tensor Engine For ROCm &#8212; ROCm Blogs</a>: 未找到描述</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3-0324">deepseek-ai/DeepSeek-V3-0324 · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1352776115894353940)** (22 条消息🔥): 

> `支持 Profiler 的云服务商、深入探讨 NCCL、量化基准测试、理解 Flash Attention、ILGPU 2.0 可用性` 


- **支持 Profiler 的云服务商**：一位成员询问了除了 **Lambda Labs** 和 **AWS** 之外，还有哪些支持使用 Profiler 的云服务商，并建议整理一份“耻辱名单”来向更多服务商施压。
   - 有人指出 [lightning.ai](https://lightning.ai) 支持 profiling，而 **AWS** 仅在裸机（bare metal）上提供该功能；根据 [Reddit 帖子](https://www.reddit.com/r/MachineLearning/comments/1dtq8hn/any_cloud_providers_with_1_h100_allowing/)，**Paperspace** 和 **Nebius** 也被提及。
- **量化基准测试方法探讨**：一位成员询问了如何对量化模型进行基准测试，以及如何确定哪些层需要量化。
   - 另一位成员建议使用 [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 框架来评估语言模型。
- **通过编码解析 Flash Attention**：在关于理解 Flash Attention (**FA**) 的讨论中，一位成员建议，如果时间允许，编写代码并进行 profiling/debugging 会很有帮助。
   - 有人指出，亲自动手实现有助于理解普通 Attention，对于 **Flash Attention** 也是如此。
- **Tile 布局图：掌握位交错（Bit Interleaving）**：有人就 Tile 布局图的实用性和清晰度征求反馈，例如来自 [tile-ai](https://github.com/tile-ai/tilelang/blob/main/examples/plot_layout/images/base_layout.png) 和 [Nvidia PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#register-fragments) 的图表。
   - 讨论集中在假设大小为 2 的幂且连续的情况下，在整数集之间映射时坐标位是如何交错的。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness">GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://www.reddit.com/r/MachineLearning/comments/1dtq8hn/any_cloud_providers_with_1_h100_allowing/">Reddit - 互联网的心脏</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1352728488054100079)** (15 messages🔥): 

> `Triton 与 Pip 混淆、cuTIl 性能、BF16 原子操作、Triton IR 生成、Flash Attention 1 Kernel 问题` 


- **Triton 安装可能导致 Pip 混淆**：在同一个文件夹中同时安装 **triton** 和 **triton-windows** 会导致 **pip** 混淆，用户需要先卸载两者，然后再重新安装 **triton-windows**。
   - **PyTorch 已经在使用 Triton** 这一事实表明该软件包具有持续的相关性。
- **cuTIl 提升 Triton 性能**：一位用户询问了 **cuTIl** 的性能优势，质疑其是否旨在通过直接利用 **SASS** 而非 **PTX** 进行更精细的性能调优，从而超越基于 **LLVM** 的方法。
   - 其他人指出这与原子 CAS 相关，并引用了此 [github issue](https://github.com/triton-lang/triton/issues/1387)。
- **BFloat16 原子加法需要 SM90 或更高版本**：**atom.add.noftz.bf16** 和 **atom.add.noftz.bf16x2** 要求 **sm_90** 或更高版本，因此需要在 **PTX** 中使用 **atom.global.cas** 版本。
   - 一位用户的临时解决方法是使用 **float32** 输出并转换为 **bfloat16**，这导致 **A100** 上的 **LLama3-8B** 推理速度从 **113 tokens/sec** 降至 **96 tokens/sec**；使用 post-hook 转换可能会提高速度。
- **Gemlite 面临 BF16 原子加法限制**：一位用户在 [gemlite](https://github.com/mobiusml/gemlite/blob/master/gemlite/triton_kernels/gemv_revsplitK_A16fWnO16f_int32packing.py#L308) kernel 中遇到了 **bfloat16 atomic add** 问题，该问题需要 **sm_90** 或更高版本。
   - 他们正在研究在 Triton 中将转换作为 post-hook，因为由于 *torch.compile 不支持 prune_configs_by*，他们需要一个自定义算子。
- **Flash Attention 1 Kernel 面临差异**：一位用户在 Triton 中实现 **Flash Attention 1** 作为第一个 kernel 时报告称，它 *在 TRITON_INTERPRET=1 下工作正常，但在 CUDA 上有少量元素不匹配*。
   - 在增加 **rtol & atol** 后测试通过，这表明 CPU 与 GPU 的结果可能存在重新排序，而浮点数对这种顺序变化很敏感。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/triton-lang/triton/issues/1387#issuecomment-1815528209">功能请求：针对 bfloat16 的 `tl.atomic_add` · Issue #1387 · triton-lang/triton</a>：更多上下文请参见 pytorch/pytorch#97016。由于 tl.atomic_add 不支持 BFloat16，torch.index_put(..., accumulate=True) 目前在 torch.compile 下对 torch.bfloat16 失效。...</li><li><a href="https://github.com/triton-lang/triton/issues/1387">功能请求：针对 bfloat16 的 `tl.atomic_add` · Issue #1387 · triton-lang/triton</a>：更多上下文请参见 pytorch/pytorch#97016。由于 tl.atomic_add 不支持 BFloat16，torch.index_put(..., accumulate=True) 目前在 torch.compile 下对 torch.bfloat16 失效。...</li><li><a href="https://github.com/mobiusml/gemlite/blob/master/gemlite/triton_kernels/gemv_revsplitK_A16fWnO16f_int32packing.py#L308">gemlite/gemlite/triton_kernels/gemv_revsplitK_A16fWnO16f_int32packing.py at master · mobiusml/gemlite</a>：Triton 中的快速低比特 matmul kernel。通过在 GitHub 上创建账号为 mobiusml/gemlite 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1353310688608780308)** (42 条消息🔥): 

> `WMMA 指令, PyTorch RTX 5080 CUDA 12.8 支持, Flash Attention 优化, Hopper 架构 Swizzle, CUDA 性能计数器权限错误` 


- **WMMA 指令编译为 MMA**：已确认 **WMMA** 指令确实是“封装器（wrappers）”，它们在 SASS 中直接编译为 **HMMA/IMMA/QMMA** 指令，其运作方式与 **MMA** 指令类似，正如 [CUDA Godbolt](https://cuda.godbolt.org/) 所示。
- **RTX 5080 PyTorch 支持随 CUDA 12.8 补丁出现**：一位开发者发布了一个补丁，使 **RTX 5080** 能够完全兼容 **CUDA 12.8 + PyTorch 2.5.0** 以及 **Blackwell / sm_120 架构**，并提供了一个包含脚本、diff 文件和说明的 [GitHub 仓库](https://github.com/kentstone84/pytorch-rtx5080-support)。
- **Flash Attention 的内存效率**：在 **Flash Attention** 中，张量以 **(batch_size, N, num_heads, d)** 格式存储，在 **d** 维度上是连续的（通常 > 64），这实现了高效的全局内存合并（global memory coalescing），每个线程可以加载 **16B** 的数据。
- **Hopper 的 Swizzle 布局解析**：文档中关于 **Hopper 架构**中 **64B swizzle** 的描述让许多人感到困惑，但现在已明确这是一个 **64*B* (bytes)** 的 swizzle，其中每个方块是 **128*b* (bits)**，这对应于 **8 位 dtypes 的 8x64 tile** 以及 **16 位类型的 8x32 tile**。
- **解决 Linux 上的 CUDA 权限错误**：当遇到 **ERR_NVGPUCTRPERM**（表示缺乏访问 **NVIDIA GPU Performance Counters** 的权限）时，Linux 用户可能需要使用 `sudo` 运行命令，不过也应参考链接中的 [NVIDIA 文档](https://developer.nvidia.com/ERR_NVGPUCTRPERM) 以获取全面的解决方案。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://cuda.godbolt.org/">Compiler Explorer</a>: 未找到描述</li><li><a href="https://developer.nvidia.com/ERR_NVGPUCTRPERM">NVIDIA Development Tools Solutions - ERR_NVGPUCTRPERM: Performance Counters 权限问题</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/CUDA/s/kKbHNez7E6">Reddit - 互联网的心脏</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1352749380075130931)** (5 messages): 

> `torch.compile() graph breaks, VRAM reduction techniques, FA3 attention FP8` 


- **`torch.compile()` 与图截断（Graph Breaks）：一项调查**：一位用户询问在使用 `torch.compile()` 时如何检查**图截断**，并指出 `tlparse` 日志显示*指标缺失*。
   - 他们注意到使用 `torch.compile(model, fullgraph=True)` 时训练运行正常，并询问这是否意味着没有图截断。
- **VRAM 使用量进一步降低**：一位用户概述了减少 **VRAM 占用**的技术，包括将优化器步骤折叠到反向传播中（参考 [PyTorch 教程链接](https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html)），以及通过 `torchao` 将优化器状态卸载（offload）到 CPU。
   - 他们还提到了使用 BNB 分页优化器部分卸载优化器状态，并指向了 [TorchTune 页面](https://pytorch.org/torchtune/main/tutorials/memory_optimizations.html) 关于内存优化的内容，引用了一张总结了 **Model Precision**、**Activation Checkpointing** 和 **Activation Offloading** 等组件的表格。
- **序列化编译模型依然难以实现**：一位用户分享了一个 [GitHub issue](https://github.com/pytorch/pytorch/issues/101107)，关于无法保存/加载编译后的模型，并询问是否有人正在积极解决此问题。
   - 该 issue 将该 bug 描述为：*使用 pickle 序列化编译模型失败*。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://podcasts.apple.com/sk/podcast/pytorch-developer-podcast/id1566080008">PyTorch Developer Podcast</a>：技术播客 · PyTorch 开发者播客是 PyTorch 开发团队就 PyTorch 各种内部开发主题进行短小精悍（10-20 分钟）讨论的地方。</li><li><a href="https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html">如何通过将优化器步骤融合到反向传播中来节省内存 — PyTorch Tutorials 2.6.0+cu124 文档</a>：未找到描述</li><li><a href="https://pytorch.org/torchtune/main/tutorials/memory_optimizations.html">内存优化概述 &mdash; torchtune 主文档</a>：未找到描述</li><li><a href="https://github.com/pytorch/pytorch/issues/101107">使编译模型可序列化 · Issue #101107 · pytorch/pytorch</a>：🐛 描述 bug：使用 pickle 序列化编译模型失败，报错为 Can&#39;t pickle local object &#39;convert_frame.&lt;locals&gt;._convert_frame&#39; 且无法 pickle &#39;ConfigModuleInstance&...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1353036283689173003)** (1 messages): 

> `Tanishq Kumar, Scaling Laws for Low Precision, Precision-aware scaling laws, post-training quantization, compute optimal` 


- **Tanishq Kumar 关于缩放法则（Scaling Laws）的演讲即将开始**：在大约 3 小时后，**Tanishq Kumar** 将讨论他的论文 "[Scaling Laws for Low Precision](https://arxiv.org/abs/2411.04330)"，该论文引入了用于训练和推理的精度感知缩放法则。
- **低精度训练缩放法则**：论文提出，在较低精度下训练会减少模型的**有效参数量**，从而能够预测低精度训练和训练后量化（post-train quantization）带来的额外损失。
   - 研究表明，在较低精度下训练更大的模型可能是**计算最优（compute optimal）**的。
- **量化退化**：研究表明，随着模型在更多数据上进行训练，**训练后量化**带来的退化会加剧，这可能导致额外的预训练数据反而产生负面影响。
   - 该研究统一了训练后和预训练量化的缩放法则，以预测不同精度下训练和推理的退化情况，并在高达 **1.7B 参数**、**26B tokens** 训练的模型上进行了验证。



**提及的链接**：<a href="https://arxiv.org/abs/2411.04330">Scaling Laws for Precision</a>：低精度训练和推理会影响语言模型的质量和成本，但目前的缩放法则并未考虑到这一点。在这项工作中，我们设计了“精度感知”的缩放法则...

  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

srivarshan4271: https://lights0123.com/blog/2025/01/07/hip-script/
  

---

### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1353465342751670313)** (1 条消息): 

> `牛津大学 AI 与神经科学奖学金，游戏与神经影像中的 AI / RL，神经系统疾病的无创诊断与治疗` 


- **牛津大学开设 AI 与神经科学奖学金**：牛津大学正在招聘一名研究员（博士后级别或同等经验），与 Rui Ponte Costa 合作研究 **游戏与神经影像中的 AI / RL**。
   - **薪资将达到 £100k+**，根据经验水平略有调整，工作地点位于神经回路与行为中心（Centre for Neural Circuits and Behaviour）。
- **AI 助力系统行为神经科学**：该奖学金旨在开发一种 **AI 驱动的技术**，通过分析游戏数据来推断特定大脑区域对行为的贡献，从而实现 **神经系统疾病的无创诊断与治疗**。
   - 他们的方法利用了最先进的深度强化学习模型，特别是 **MuZero 和 Dreamer 架构**（[项目链接](https://encode.pillar.vc/projects/behavioral-neuroscience)）。
- **Pillar VC 支持 AI for Science 研究员**：Pillar VC 和 ARIA 正在支持 AI 研究员在英国各地 **ARIA 的 Opportunity Spaces** 中的顶级科学实验室进行为期一年的嵌入式研究。
   - 他们正在寻找构建 **AI for science** 的下一代创始人、科学家和领导者（[奖学金链接](https://encode.pillar.vc)）。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://encode.pillar.vc/projects/behavioral-neuroscience">ARIA Opportunity Space: Scalable Neural Interfaces</a>: 未找到描述</li><li><a href="https://encode.pillar.vc">Encode: AI for Science Fellowship</a>: 一个将顶尖 AI 人才与英国领先科学实验室联系起来以催化转化的奖学金项目。由 Pillar 支持，ARIA 提供动力。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1352813469686038599)** (56 条消息🔥🔥): 

> `GPU/CUDA 学习资源，Warp 调度器的重要性，上下文切换，SIMD 与 SIMT 执行，Windows 上的 Flash attention 设置` 


- **GPU 术语表引发 CUDA 困惑**：一名成员在通过 [Modal GPU 术语表](https://modal.com/gpu-glossary)学习 GPU/CUDA 时，对 Warp 调度器和上下文切换表示困惑，特别是关于 *如果每个线程共享相同的指令指针，那么上下文切换的意义何在*。
   - 另一位成员通过一个 **64 线程** 分为两组的例子进行了解释，展示了调度器如何在另一个 Warp 等待数据时执行其中一个 Warp，这类似于 CPU 的上下文切换，但没有状态存储的开销。
- **SIMT 揭秘：数据区分线程**：一位成员澄清说，虽然 Warp 中的线程共享相同的指令，但数据不同，从而实现了 **SIMT**（单指令多线程）执行，即 *32 个线程可以在一个时钟周期内完成 32 个元素的乘法*。
   - 他们强调，**一组 32 个线程** 是同时被调度的，上下文切换会换入另一组 32 个线程，而不是逐个调度单个线程。
- **Windows VM 上的 Flash Attention 挫折**：一位成员在 **Windows/Ubuntu VM** 本地设置 [flash attention 仓库](https://github.com/Dao-AILab/flash-attention)时遇到问题，在 **nvcc 版本冲突** 以及对现有 CUDA/Torch/Triton 设置的潜在干扰中挣扎。
   - 考虑到使用 **vast.ai** 进行开发，他们寻求关于适合 Triton/CUDA 工作的机器推荐，以及选择机器来使用自定义 kernel 训练 BERT 模型的指导。
- **纠正 CUDA Core 困惑**：一位成员解释说，NVIDIA 的营销术语 "CUDA cores" 实际上是指 **FP32 单元**，其功能类似于 SIMD 操作，无法独立运行。
   - 来自不同 kernel 的 Warp 可以以精细的时间片方式调度到同一个流式多处理器（**SM**），这在线程等待数据加载时特别有用。
- **流式多处理器架构深度解析**：一位成员澄清说，多个线程块（thread blocks）可以在一个流式多处理器（**SM**）上运行，这对于块同步至关重要，允许 **SM** 在其他 Warp 等待屏障（barrier）时拥有准备运行的 Warp，并参考了 [H100 流式多处理器](https://developer-blogs.nvidia.com/wp-content/uploads/2022/03/H100-Streaming-Multiprocessor-SM.png)图示。
   - 他们解释说，寄存器和共享内存等资源决定了驻留线程块的数量，而 Warp 调度器在 Warp 之间进行上下文切换，以保持处理单元处于繁忙状态。



**提及的链接**: <a href="https://modal.com/gpu-glossary">GPU Glossary</a>: GPU 相关术语表。

  

---

### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1353019198237970504)** (1 条消息): 

> `Amazon 书籍发布日期，书籍第 5 版` 


- **亚马逊上发现第 5 版发布日期**：一名成员报告称在 **Amazon** 上看到某本未指明书籍的 **第 5 版** 列表，预定发布日期为 **2026 年 2 月**。
- **发布日期未确认**：另一名成员请求确认该发布日期。


  

---


### **GPU MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/)** (1 条消息): 

bigfoot1144: 目前有进展吗？
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1353400806300450928)** (2 条消息): 

> `ROCm, tilelang HIP 后端, row-row bank conflict-free swizzle, AMD 赞助显卡` 


- **寻求 ROCm Row-Row Bank Conflict-Free Swizzle 实现**：一名成员正在寻求 **ROCm** 专家，以帮助为 **tilelang HIP 后端**实现 *row-row bank conflict-free swizzle*。
   - 目前，他们只有针对 **NT layout conflict swizzling** 的解决方案，并请求社区协助。
- **为 ROCm 开发恳求 AMD 显卡赞助**：同一名成员开玩笑地请求 **AMD** 赞助一些显卡，用于 **ROCm** 相关的开发。
   - 这凸显了 **ROCm** 生态系统中部分开发者面临的资源限制。


  

---


### **GPU MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1352996043939385436)** (2 条消息): 

> `Hopper Flops, H100 时钟频率, H100 SMs, Nvidia Boost Clocks` 


- **H100 的 Dense FLOPs 揭晓**：对于 **fp16/bf16**，**Hopper** 的 dense flops = **989 TFLOPS**，**H100** 的时钟频率 = **1.830 GHz**，**SM** 数量 = **132**。
   - **FLOPs / clock / SM** = (**989 x 10^3**) / **1.83 / 132**，约等于 **4096**。
- **Nvidia 鲜为人知的 Boost Clock 详情**：**H100 SXM** 在普通 **SM 操作**下的 boost clock 为 **1.980 GHz**，但如果使用 **Tensor Cores**，它会根据功耗/散热情况降至 **1.830** 或更低。
   - *在某些罕见情况下，运行 TC 操作时也能达到全速 boost clock*，但奇怪的是，这种情况并不总是发生。
- **官方 Hopper Boost Clock 文档已找到**：分享了一份提到不同 boost clock 的文档（[GTC22 Whitepaper](https://resources.nvidia.com/en-us-data-center-overview/gtc22-whitepaper-hopper)）。
   - 不同的 **boost clocks** 可以在文档第 39 页的表 3 中找到。



**提到的链接**：<a href="https://resources.nvidia.com/en-us-data-center-overview/gtc22-whitepaper-hopper">NVIDIA H100 Tensor Core GPU 架构概览</a>：NVIDIA H100、新型基于 H100 的 DGX、DGX SuperPOD 和 HGX 系统以及基于 H100 的融合加速器的高层级概览。随后深入探讨了 H100 硬件架构等。

  

---


### **GPU MODE ▷ #[tilelang](https://discord.com/channels/1189498204333543425/1240586843292958790/1353365364423131259)** (10 条消息🔥): 

> `Tilelang 2:4 稀疏性支持, Tilelang v0.1.3 发布, SPGEMM 问题` 


- **Tilelang 将支持 2:4 稀疏性**：Tilelang 计划利用 **Cute** 作为后端来支持 **2:4 稀疏性**，尽管用户承认这在目前的 **AI 工作负载**中并不常见。
   - 一位用户表达了对微调 **2:4 稀疏 LLM** 的兴趣，指出其在视觉模型中取得了成功，但不确定其对 **LLM 准确率**的影响。
- **Tilelang v0.1.3 发布，包含 Cute 升级**：Tilelang 发布了 [v0.1.3](https://github.com/tile-ai/tilelang/releases/tag/v0.1.3)，具有增强功能、优化和错误修复，包括 **Cute 升级**。
   - 该版本包括新的 Kernel 和教程（如 **DeepGEMM**），以及 **autotuning** 和 **kernel caches** 等新特性。
- **请求添加 SPGEMM Issue**：一名 TileLang 开发者请求有兴趣尝试将 Tilelang 用于 **SPGEMM** 的用户在 GitHub 上提交 issue。
   - 一位用户表示，如果开发团队进一步调查，他们有兴趣关注此项进展。



**提到的链接**：<a href="https://github.com/tile-ai/tilelang/releases/tag/v0.1.3">Release v0.1.3 · tile-ai/tilelang</a>：更新内容 [Docker] 为 CUDA 版本在 Dockerfile 中添加 libstdcxx-ng-12，由 @LeiWang1999 在 #160 提交；通过 ctypes 后端添加 CPU JIT，由 @xs-keju 在 #154 提交；[Carver] 用于快速编译的多线程编译...

  

---

### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1353147367452577977)** (3 messages): 

> `Parallelized Cholesky, Python + MLX + Metal` 


- **并行化 Cholesky 通过 Python + MLX + Metal 加速**：一位成员分享了他们对社区的贡献：*一个基于 Python + MLX + Metal 的超高速并行化 Cholesky*，并附带了 [Python 文件](https://cdn.discordapp.com/attachments/1285384841730457600/1353266677130989588/cholesky_metal.py?ex=67e3018e&is=67e1b00e&hm=adb5b20c5284632e835d3b99bb32418c4967ba8009acc790d6175e964cd8c8d1&)。
   - 另一位成员评论道：*这真的很酷*。
- **MLX 势头强劲**：社区对用于 Metal 的 MLX 框架表现出日益增长的兴趣。
   - MLX 似乎正在开启高速计算的新可能性。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1352720241310044261)** (10 messages🔥): 

> `WheelNext Initiative, CUDA Indexing Blogpost, Container-First Triton Development, GemLite bfloat16 Support` 


- **WheelNext 倡议蓄势待发，旨在增强 Python 打包**：**WheelNext** 倡议 ([wheelnext.dev](https://wheelnext.dev/)) 旨在改善 Python 打包生态系统的用户体验，重点关注科学计算和机器/深度学习。
   - 宣布了一场聚会，讨论如何让分发带有原生加速器代码的 Python 包变得更加容易，详情可在 [Discord](https://discord.com/events/987824841656791130/1351966194424352789) 上查看。
- **通过新博客文章深入了解 CUDA 索引**：一位成员分享了一篇[博客文章](https://veitner.bearblog.dev/indexing-in-cuda/)，通过矩阵乘法的 2D 块平铺（block tiling）示例解释了 **CUDA 索引**，并强调了行优先（row-major）格式。
   - 文章详细介绍了 CUDA 中形状为 `(M, N)` 的 2D 数组 `A` 如何以行优先格式线性化，将坐标 `(i,j)` 映射到 `i * N + j`。
- **容器优先方法简化 Triton 开发**：一位成员重点介绍了一篇新的[博客文章](https://next.redhat.com/2025/03/20/a-container-first-approach-to-triton-development/)，关于使用容器来简化和加速 **Triton kernel 开发**。
   - 该文章强调了容器化如何通过简化设置、增加一致性以及实现更无缝的协作来增强 **Triton 开发**工作流。
- **GemLite 为 Gemma 模型添加 bfloat16 支持**：**GemLite** 现在在 Hopper 和非 Hopper GPU 上都支持 **bfloat16**，从而能够通过 hqq 在 vllm 中运行 **Gemma 模型**。
   - 更多详情可在[相关推文](https://x.com/mobicham/status/1904185254224535875)和 [GitHub pull request](https://github.com/mobiusml/gemlite/pull/24) 中找到。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://wheelnext.dev/">WheelNext</a>：未找到描述</li><li><a href="https://x.com/mobicham/status/1904185254224535875">来自 mobicham (@mobicham) 的推文</a>：GemLite 现在在 Hopper 和非 Hopper GPU 上都支持 bfloat16 🫡 https://github.com/mobiusml/gemlite/pull/24</li><li><a href="https://veitner.bearblog.dev/indexing-in-cuda/">CUDA 中的索引</a>： 
在这篇博客文章中，我想解释矩阵处于行优先格式意味着什么。 
这对于理解 CUDA kernel 及其方法至关重要...</li><li><a href="https://next.redhat.com/2025/03/20/a-container-first-approach-to-triton-development/">一种容器优先的 Triton 开发方法</a>：来自 OpenAI 的 Triton 项目处于民主化 AI 加速器和 GPU kernel 编程这一突破性运动的前沿。它为编写...提供了一个强大且灵活的框架。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1353640859639218178)** (1 messages): 

> `LLM Kernel Understanding, RL for Operation Understanding, Reducing Hallucinations in Kernel Creation` 


- **LLM 揭秘 Kernel 代码**：其核心思想是使用 **LLM** 来理解 **kernel 代码**，解释简单的概念以及张量中特定位置的变量状态。
   - 旨在确保 LLM 掌握底层操作。
- **RL 强化对 Kernel 操作的掌握**：采用**强化学习 (RL)** 来增强模型对操作的理解，确保牢固掌握。
   - 对 kernel 操作的牢固掌握可以作为创建复杂 kernel 的先决条件，并有可能减少幻觉。
- **使用 LLM 进行 Kernel 创建的合理性检查**：使用 LLM 来验证和解释 kernel 操作可以大大减少复杂 kernel 创建过程中的幻觉。
   - 这种方法可以被视为对复杂 kernel 代码和设计的**合理性检查 (sanity check)**。


  

---

### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1352873117885927464)** (5 messages): 

> `veRL rollouts with sglang, low precision data types, quantization strategies for RL, ARC-AGI2 announcement` 


- **veRL 支持 sglang rollout**：**veRL** 现在支持使用 **sglang** 进行 rollout，如[这篇论文](https://arxiv.org/abs/2503.16219)所示。
- **使用 GRPO 进行小模型推理**：一项研究表明强化学习 (**RL**) 提升了小语言模型 (**LLMs**) 的推理能力，具体是一个在 4 张 **NVIDIA A40 GPU** 上训练 24 小时的 **1.5B 参数模型**。
   - 通过在精选数据集上适配 Group Relative Policy Optimization (**GRPO**) 算法，该模型取得了显著提升，例如 AMC23 准确率从 **63% 提升至 80%**，AIME24 达到 **46.7%**，训练成本仅为 **$42**。
- **ARC-AGI2 前沿基准测试**：一位成员分享了 [ARC-AGI-2 发布公告](https://x.com/arcprize/status/1904269307284230593)，这是一个挑战 AI 推理系统的前沿 AGI 基准测试。
   - 目标是以约 **$0.42**/任务的效率达到 **85%** 的准确率，这与目前基础 LLMs 为 **0%** 以及推理系统低于 **4%** 的性能水平形成鲜明对比。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2503.16219">Reinforcement Learning for Reasoning in Small LLMs: What Works and What Doesn&#39;t</a>：增强大语言模型 (LLMs) 的推理能力通常依赖于海量的计算资源和庞大的数据集，这限制了资源受限环境下的可访问性...</li><li><a href="https://x.com/arcprize/status/1904269307284230593">ARC Prize (@arcprize) 的推文</a>：今天我们宣布推出 ARC-AGI-2，这是一个尚未饱和的前沿 AGI 基准测试，旨在挑战 AI 推理系统（对人类而言相对容易）。大奖目标：85% 准确率，~$0.42/任务效率。当前性能...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/1353542132799111262)** (5 messages): 

> `CUDA core, CUDA_fp6.hpp, CUDA_fp4.hpp` 


- **征集 CUDA core 的 fp4 和 fp6 用例**：一位成员询问哪些库在 **CUDA core** 中使用了 **fp4** 和 **fp6**，并提到了 **12.8** 版本中存在的 `cuda_fp6.hpp` 和 `cuda_fp4.hpp` 头文件。
   - 然而，他们指出很难找到积极采用这些头文件的库。
- **CUDA FP4/FP6 库使用情况**：用户正在询问 **CUDA core** 中 **FP4** 和 **FP6** 数据类型的使用情况，特别是是否有任何库正在利用它们。
   - 他们已在 CUDA **12.8** 版本中识别出头文件（**cuda_fp6.hpp** 和 **cuda_fp4.hpp**），但尚未在现有库中找到其实际应用的示例。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1352752759107227771)** (9 messages🔥): 

> `Submission Guide, Kernel profiling, Conv2D error` 


- **提交指南已发布**：一位成员询问提交指南，另一位成员分享了 GPU kernel 排行榜的[文档链接](https://gpu-mode.github.io/discord-cluster-manager/docs/intro)，这是一个在 Discord 上的竞赛平台，用户可以提交自己的 kernel 实现。
- **Kernel 性能分析即将推出！**：一位成员询问是否可以通过机器人本身对他们的 triton kernel 进行性能分析（profiling）。
   - 回复是：*目前还没有这个功能，但已在计划中，（很可能）可以在第一个问题集发布时期待它*。
- **Conv2D 提交错误**：一位成员报告在提交 conv2d 时遇到持续错误，涉及 `subprocess.CalledProcessError: Command '['ninja', '-v']' returned non-zero exit status 1`，并询问这是否意味着他们的 **CUDA 源码**无法编译。
   - 该成员是 CUDA 和 C++ 的新手，正在寻求社区的帮助。



**提及的链接**：<a href="https://gpu-mode.github.io/discord-cluster-manager/docs/intro">入门指南 | GPU MODE Kernel 排行榜</a>：欢迎！如果你对构建 GPU kernel 充满热情，这个排行榜就是为你准备的！我们

  

---

### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1352719326683074611)** (119 条消息🔥🔥): 

> `H100 上的 matmul 基准测试，A100 上的 grayscale 基准测试，T4、L4、A100、H100 上的 grayscale 基准测试，T4 上的 histogram 基准测试，A100 上的 vectorsum 测试` 


- **Modal Runners 在 H100 上出色完成 Matmul 基准测试**：在 **H100 GPU** 上使用 **Modal runners** 进行的大量 `matmul` 基准测试和测试已成功，提交 ID 范围从 **2479** 到 **2487**。
   - 这些提交表明 Modal runners 在高性能 GPU 上执行和集成矩阵乘法任务方面取得了成功。
- **A100 GPU 上的 Grayscale 挑战**：在 **A100 GPU** 上使用 **Modal runners** 成功完成了大量 `grayscale` 基准测试和排行榜提交，提交 ID 涵盖从 **2488** 到 **2596** 及其后。
   - 这些持续的成功凸显了 Modal runners 在 A100 GPU 上处理图像处理任务的可靠性和效率。
- **跨 GPU 的 Grayscale 卓越表现**：使用 **Modal runners** 的 `grayscale` 排行榜提交已在包括 **T4**、**L4**、**A100** 和 **H100** 在内的多种 GPU 上取得成功，初始提交 ID 为 **2484**。
   - 这证明了 Modal runners 在处理不同 GPU 架构上的图像处理任务时的通用性。
- **T4 GPU 上的 Histogram 成功提交**：使用 **Modal runners** 在 **T4 GPU** 上进行的 ID 为 **2765** 的 `histogram` 基准测试提交已成功。
   - 这表明在利用 Modal runners 平台的 T4 GPU 上成功执行了直方图计算任务。
- **A100 上的 Vector Sum 胜利与 Conv2d 征服**：使用 **Modal runners** 在 **A100 GPU** 上进行的 `vectorsum` 和 `conv2d` 测试提交已成功，ID 分别为 **2829** 和 **2830**。
   - 这些成功的测试突显了 Modal runners 在高性能 GPU 上处理向量操作和卷积任务的能力。


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1353071106201882693)** (2 条消息): 

> `CUDA, load_inline(), PyTorch headers, KernelBot` 


- **`load_inline()` 因过多的 PyTorch 头文件导致超时**：使用 `load_inline()` 的 CUDA 提交出现超时，因为添加了约 **5K 个 PyTorch 头文件**，详情见 [此 PR](https://github.com/pytorch/pytorch/pull/149480) 中的调查。
   - 增加了一个新模式来禁用隐式添加头文件，一位成员成功将示例编译时间从 **90 秒缩短到 15 秒**，而另一位同事将其从 **15 秒缩短到 5 秒**。
- **KernelBot 排行榜性能提升**：[KernelBot 排行榜](https://pytorch.org/blog/kernel-compilation/) 支持通过 `load_inline()` 进行自定义 CUDA 扩展，此前这会导致长达 **90 秒** 的冷启动。
   - 一位成员表示，*他们一直以为这是 CUDA 的问题*，并很高兴这个问题得到了解决。



**提到的链接**：<a href="https://github.com/pytorch/pytorch/pull/149480">msaroufim 提交的 load_inline no_implicit_headers 模式 · Pull Request #149480 · pytorch/pytorch</a>：在 kernelBot 排行榜中，我们支持人们通过 load_inline() 使用自定义 CUDA 扩展进行竞争，然而即使是玩具级（toy）内核，这也可能导致长达 90 秒的冷启动 —— 这个问题是优先...

  

---

### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1352766663031914619)** (17 条消息🔥): 

> `GPU prices, VRAM requirements for LLMs, RTX Pro 6000, CUDA Capability` 


- **AI 热潮中 GPU 价格飙升**：由于 NVIDIA 将高 VRAM 限制在特定型号的策略，高端消费级 GPU 变得越来越昂贵，但 **vast.ai** 和 **Nebius** 等云厂商为运行模型提供了更便宜的替代方案。
   - 一位成员表示，“*欢迎来到 AI 热潮*”，强调了 AI 对 GPU 定价和供应的影响。
- **在旧 GPU 上拉满预算，在本地运行**：对于本地机器学习，建议投资 **3090** 或 **4090** 等旧显卡以最大化预算利用率，*2x3090* 的性能可能超过单张新显卡，从而实现本地分布式训练。
   - 有观点认为，旧显卡提供了“*在本地学习分布式技术*”的机会。
- **Nvidia 让用户对高价脱敏**：拥有 96GB VRAM 的新款 **RTX Pro 6000** 被认为是专业人士的合理选择，这使人们对高昂 GPU 成本的感知常态化，尽管它缺乏 NVLink。
   - 一位成员指出，“*实际上，我认为 Nvidia 已经成功让我对其疯狂的价格脱敏了*”，这表明由于市场趋势，用户的预期发生了调整。
- **GDDR7 显存**：**RTX Pro 6000** 配备 **96 GB GDDR7**（带 ECC）和 **1792 GB/sec** 带宽，尽管 [Data Sheet](https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-6000-blackwell-workstation-edition/workstation-blackwell-rtx-pro-6000-workstation-edition-nvidia-us-3519208-web.pdf) 和 [TPU 规格](https://www.techpowerup.com/gpu-specs/rtx-pro-6000-blackwell.c4272) 中报告的 CUDA API 版本存在差异。
   - 规格报告 Compute API 为 CUDA 11.6，而 TPU 声称为 CUDA 10.1，该成员还强调 [CUDA GPUs 列表](https://developer.nvidia.com/cuda-gpus) 中 GeForce RTX 50 系列的 C.C.（Compute Capability）为 10.0 而非 12.0。


  

---


### **GPU MODE ▷ #[tpu](https://discord.com/channels/1189498204333543425/1351571759761068082/)** (1 条消息): 

rocka2424: 这太棒了，期待！
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1352811157630750883)** (86 条消息🔥🔥): 

> `Nvidia Mamba-Transformer Hybrid, Qwen 2.5 Omni Model, DeepSeek V3 Model Update, Reve Image Halfmoon Model, Qwen2.5-VL-32B-Instruct` 


- **Nvidia 研发 Nemotron-H Mamba-Transformer 混合模型**：Nvidia 推出了 **Nemotron-H** 系列模型，包括 **8B** 和 **47-56B** 系列，这些是 Mamba-Transformer 混合模型。[根据其研究](https://research.nvidia.com/labs/adlr/nemotronh/)，与其它模型相比，该系列提供了更快的推理速度。
- **Qwen 发布 Qwen2.5-Omni：端到端流式多模态模型**：Qwen 发布了 **Qwen2.5-Omni**，这是一款旨在感知文本、图像、音频和视频的多模态模型，同时以流式方式生成文本和自然语音响应，[根据 HuggingFace 上的信息](https://github.com/huggingface/transformers/pull/36752/commits/b4ff115375f02b59eb3e495c9dd3c1219e63ff50)。
- **DeepSeek V3 获得快速更新，依然领跑排行榜**：DeepSeek 宣布了 **DeepSeek V3** 模型的微小版本升级，API 接口和使用方法保持不变，[根据其 HuggingFace 页面](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324)。
- **Reve Image 发布 Halfmoon：声称在图像生成领域占据首位**：**Reve Image** 推出了 **Halfmoon**，声称它是世界上最好的图像模型，具有出色的文本渲染、提示词遵循能力和美学效果，目前可通过其网站访问，[根据其公告](https://x.com/ArtificialAnlys/status/1904188980423467472)。
- **Qwen 发布 Qwen2.5-VL-32B-Instruct：带有 RLHF 的开源 VL 模型**：Qwen 在 **Apache 2.0 许可证**下开源了 **Qwen2.5-VL-32B-Instruct** 模型，该模型通过强化学习进行了优化，在人类偏好和数学推理方面表现出显著提升，[根据其博客](https://qwenlm.github.io/blog/qwen2.5-vl-32b/)。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://qwenlm.github.io/blog/qwen2.5-vl-32b/">Qwen2.5-VL-32B: Smarter and Lighter</a>: QWEN CHAT GITHUB HUGGING FACE MODELSCOPE DISCORD 简介：今年 1 月底，我们推出了 Qwen2.5-VL 系列模型，获得了广泛关注和积极反馈...</li><li><a href="https://x.com/arcprize/status/1904269307284230593?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">来自 ARC Prize (@arcprize) 的推文</a>: 今天我们宣布推出 ARC-AGI-2，这是一个未饱和的前沿 AGI 基准测试，旨在挑战 AI 推理系统（对人类而言相对容易）。大奖：85%，约 $0.42/任务效率。当前表现...</li><li><a href="https://x.com/AkemiMadoka/status/1904111806693671123">来自 坂本 (@AkemiMadoka) 的推文</a>: @teortaxesTex 看起来是 v3 的一个小更新</li><li><a href="https://x.com/simonw/status/1904187791808123052">来自 Simon Willison (@simonw) 的推文</a>: 关于今天 DeepSeek v3 0324 模型的笔记——这是一个 641 GB 且采用 MIT 许可证的巨兽，但如果你通过 MLX 使用 352 GB 的量化版本，可以在一台价值约 $10,000 的消费级 512GB M3 Mac Studio 上运行它...</li><li><a href="https://x.com/reveimage/status/1904211082870456824">来自 Reve (@reveimage) 的推文</a>: Halfmoon 就是 Reve Image —— 它是世界上最好的图像模型 🥇(🔊)</li><li><a href="https://x.com/TheXeophon/status/1904225899957936314">来自 Xeophon (@TheXeophon) 的推文</a>: 在我的内部基准测试中测试了新的 DeepSeek V3，所有测试的所有指标都有巨大飞跃。它现在是最好的非推理模型，取代了 Sonnet 3.5。恭喜 @deepseek_ai！</li><li><a href="https://x.com/Alibaba_Qwen/status/1904227859616641534">来自 Qwen (@Alibaba_Qwen) 的推文</a>: 72B 对 VLM 来说太大？7B 又不够强！那么你应该使用我们的 32B 模型，Qwen2.5-VL-32B-Instruct！博客: https://qwenlm.github.io/blog/qwen2.5-vl-32b/ Qwen Chat: https://chat.qwen.ai HF: https://hugg...</li><li><a href="https://x.com/picocreator/status/1904250680266956903">来自 PicoCreator - AI Model Builder 🌉 (@picocreator) 的推文</a>: ❗️Attention 并非你所需要的全部 ❗️仅使用 8 个 GPU（非集群），我们训练了 Qwerky-72B（和 32B），没有使用任何 Transformer Attention。评估结果远超 GPT 3.5 turbo，并接近...</li><li><a href="https://x.com/ArtificialAnlys/status/1904188980423467472">来自 Artificial Analysis (@ArtificialAnlys) 的推文</a>: Halfmoon 🌓 揭晓：祝贺 @reveimage 凭借 Reve Image 打造出世界领先的图像生成模型！Reve Image 在过去一周一直处于 Artificial Analysis Image Arena 中...</li><li><a href="https://x.com/asmeurer/status/1904193931325079991">来自 Aaron Meurer (@asmeurer) 的推文</a>: @simonw 许可证更新是一件大事。原始的 V3 并不是 MIT 协议。</li><li><a href="https://x.com/kimmonismus/status/1903221838022324226">来自 Chubby♨️ (@kimmonismus) 的推文</a>: Sora 取消了所有付费层级的积分限制，提供无限生成。这是一个很好的改变。</li><li><a href="https://fxtwitter.com/btibor91/status/1903469632167506018">来自 Tibor Blaho (@btibor91) 的推文</a>: @TheXeophon https://x.com/btibor91/status/1899917834496729259?s=61 引用 Tibor Blaho (@btibor91) @TheXeophon-bench</li><li><a href="https://huggingface.co/collections/nvidia/mambavision-66943871a6b36c9e78b327d3">MambaVision - 一个 NVIDIA 集合</a>: 未找到描述</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3-0324">deepseek-ai/DeepSeek-V3-0324 · Hugging Face</a>: 未找到描述</li><li><a href="https://research.nvidia.com/labs/adlr/nemotronh/">Nemotron-H: A Family of Accurate, Efficient Hybrid Mamba-Transformer Models</a>: Nemotron-H 是一系列混合 Mamba-Transformer 模型，与同等规模的其他先进开源模型相比，它们提供了更好或相当的准确度，并提升了推理速度（最高达 3 倍）...</li><li><a href="https://github.com/huggingface/transformers/pull/36752/commits/b4ff115375f02b59eb3e495c9dd3c1219e63ff50">由 BakerBunker 添加 Qwen2.5-Omni · Pull Request #36752 · huggingface/transformers</a>: 此 PR 做了什么？添加 Qwen2.5 Omni 模型。在提交之前，此 PR 修复了一个拼写错误或改进了文档（如果是这种情况，你可以忽略其他检查）。你阅读了贡献者指南吗...
</li>
</ul>

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1353028110110429295)** (25 messages🔥): 

> `Impact of noisy data in multi-turn SFT, Transformer usage in RL, Community model preferences, Trusting eval benchmarks, Gemini's image generation` 


- **多轮 SFT 中对噪声的容忍度？**：一位成员询问在多轮 SFT 中，噪声对数据质量的影响程度，特别是在复杂的 Agent 轨迹中。他认为某些噪声是可以容忍的，恢复步骤（recovery steps）很有价值，且错误的轮次可以被掩码（masked）。
   - 他们分享道，当复杂度和步骤增加时，很难收集到完美的轨迹，*例如在选择去哪个网站获取信息或使用哪个应用程序打开文件时做出错误决定*。
- **Transformer 在 RL 中普及缓慢？**：一位成员询问为什么 Transformer 在 RL 策略模型（policy models）中的应用有限，怀疑是由于计算和内存约束。
   - 他们*很难找到很多实际使用小型 Transformer 的论文*。
- **社区更偏好用 Claude 3.5 写代码？**：一位成员询问 Interconnects 是否发布社区偏好的模型列表，并指出在代码方面他们更倾向于 **Claude 3.5** 而非 **Claude 3.7**，但在推理（reasoning）方面则相反。
   - 另一位成员提到 Interconnects 不发布模型列表，但他们希望尽可能在 [artifacts logs series](https://www.interconnects.ai/t/artifacts-log) 中增加更多评估（evals）。
- **私有评估 > 基准测试**：多位成员讨论了对模型评估基准（benchmarks）的信任问题，其中一人表示：*不要相信它们；要有自己的评估*，并建议创建一个包含 5-10 个**你**关心的提示词（prompts）的 Markdown 文件。
   - 建议是在 [Chorus](https://chorus.sh/) 等工具中并排运行多个模型的提示词，以*快速感受哪个模型适合哪些任务*。
- **Gemini 的生成器仍是谜团？**：一位成员询问新款 **Gemini** 的图像生成是自回归（autoregressive）的还是使用了扩散头（diffusion head），但其架构（architecture）目前尚不清楚。
   - 另一位成员提到，实验室知道在训练（training）期间包含哪些网站可以提升常用基准测试的表现。



**提到的链接**：<a href="https://www.interconnects.ai/p/building-on-evaluation-quicksand">Building on evaluation quicksand</a>：关于语言模型评估现状。

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1352747678282420246)** (36 条消息🔥): 

> `LLM 输入/输出 token，o1-pro 性能，Mistral 24B 令人印象深刻，Claude Compass 启动提示词，DAPO 和 Dr. GRPO` 


- **LLM 计算输入和输出 Token**：在 LLM 中，**输入 token** 和 **输出 token** 都会在监督微调 (SFT) 过程中被计算，这澄清了最初关于 token 处理的一个问题。
   - 一位成员确认了 token 计数，并幽默地评论道：*“凭这些 token 的成本，他都能买下《纽约时报》(NYT) 了。”*
- **o1-pro 在扩展版 NYT Connections 基准测试中占据主导地位**：**o1-pro** 在 **扩展版 NYT Connections 基准测试** 中以 **81.7** 的得分刷新了纪录，轻松超过了之前的冠军 **o1**（得分 **69.7**），正如一篇 [推文](https://x.com/LechMazur/status/1903255938116538376) 所指出的。
   - 该基准测试是原始版本的更具挑战性的变体，每个谜题包含更多单词。
- **Mistral 24B 给社区留下深刻印象，声誉回升**：**Mistral 24B** 的发布被视为一个重大亮点，其基础模型的强度和可访问性受到称赞，并且承诺在 **Apache 2.0** 协议下发布新的开源版本，这有助于其声誉的恢复。
   - 一位成员表示：*“Mistral 24B 可能是过去几个月里最伟大的发布之一，模型非常强大，而且你还可以访问基础模型。”*
- **Claude Compass 发布提示词**：一位成员分享了 **Claude Compass** 启动提示词的推文，这些是深度研究提示词，例如 *“为我的研究寻找可靠来源”* 和 *“分析优秀的投资推介”*。
   - 此外还注意到，另一家名为 [Cohere](https://cohere.com/compass) 的公司已经拥有一款名为 **Compass** 的产品。
- **DAPO 和 Dr. GRPO 论文**：一位成员正在钻研 **DAPO** 和 **Dr. GRPO**，准备撰写即将发布的博客文章，计划回顾相关论文并改进 RLHF 书籍中关于权衡（tradeoffs）的实现部分。
   - 笔记已经完成，该成员正在考虑将 **DAPO** 和 **Dr. GRPO** 放在一起讨论，可能会将剩余内容推迟到以后的文章中。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/neurosp1ke/status/1903564534930907604">来自 Andreas Köpf (@neurosp1ke) 的推文</a>：添加了 mistral-small-3.1-24b-instruct</li><li><a href="https://x.com/LechMazur/status/1903255938116538376">来自 Lech Mazur (@LechMazur) 的推文</a>：o1-pro 在我的扩展版 NYT Connections 基准测试中创下新纪录，得分为 81.7，轻松超越了之前的冠军 o1 (69.7)！这个基准测试是我原始版本的更难版本...</li><li><a href="https://x.com/btibor91/status/1904206595229130886">来自 Tibor Blaho (@btibor91) 的推文</a>：新内容：Claude Compass (深度研究) 启动提示词 - “为我的研究寻找可靠来源” - “为我的主题提供基于证据的见解” - “为我的写作研究主题” - ...</li><li><a href="https://x.com/LechMazur/status/1903272087441023223">来自 Lech Mazur (@LechMazur) 的推文</a>：@bradthilton 我可能会对较短版本的幻觉进行基准测试，但我不可能运行其他基准测试。</li><li><a href="https://huggingface.co/spaces/Presidentlin/llm-pricing-calculator">Llm Pricing - Presidentlin 的 Hugging Face Space</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1352903024242327582)** (4 条消息): 

> `O1-pro 对比 BoN，O1-pro 推理路径边际化，开源 RL 中的科技公司 CEO` 


- **O1-pro 在推理路径合并方面表现出色**：一位成员认为 **O1-pro** 看起来更像是将推理路径与正确答案合并，而不仅仅是简单的 **BoN** (Bag of Neurons)。
   - 他们注意到 **o1-pro** 的输出长度通常比 **o1** 长得多，但不知道如何对推理路径进行边际化（marginalize）。
- **科技公司 CEO 拥护开源 RL**：Nathan Lambert 分享了一篇 [帖子](https://x.com/natolambert/status/1903893527593193639)，指出 *大型科技公司的 CEO 们正在为开源 RL 仓库中非常前沿的默认设置进行辩论*。
   - 他总结道：*“这个时间线太不可思议了。”*


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fixvx.com/nearcyan/status/1903962841952247833">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://x.com/natolambert/status/1903893527593193639">来自 Nathan Lambert (@natolambert) 的推文</a>：哈哈，当大型科技公司的 CEO 们在为开源 RL 仓库中非常前沿的默认设置进行辩论时。这个时间线太不可思议了。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1352769466965950566)** (127 条消息🔥🔥):

> `R1-Zero 训练, GRPO 偏差, LOOP & RLOO, PPO 目标函数, 创意写作 LLMs` 


- **R1-Zero 训练中行均值（Row Mean）的长度偏差被揭示**：一项分析显示，在类 **R1-Zero 训练**中使用行均值会引入偏差，倾向于更短的正确回答和更长的错误回答，详见[论文](https://github.com/sail-sg/understand-r1-zero/blob/main/understand-r1-zero.pdf)及配套[代码](https://github.com/sail-sg/understand-r1-zero)。
   - 切换到全局均值（all mean）可以在不增加长度的情况下获得相当的性能；这引发了对那些显示推理长度增加与能力提升相关的图表的质疑。
- **GRPO 的长度爆炸问题困扰着从业者**：用户在运行 **GRPO** 时观察到了**长度爆炸**，促使人们考虑长度课程（length curriculum）或裁剪（clipping）等技术，尽管这些被视为不理想的权宜之计。
   - 核心问题是当回答变长时会生成垃圾内容；这暗示了长度之外更深层次的问题。
- **vLLM 的前缀缓存（Prefix Caching）导致 RL 问题**：成员们发现 **vLLM** 的**前缀缓存**可能会导致 **RL 问题**，如[这个 GitHub issue](https://github.com/huggingface/open-r1/issues/491) 中所述。
   - 具体而言，推理表现差于训练，并确定该缓存是罪魁祸首，展示了一个可能被忽视的微妙问题。
- **LOOP 和 RLOO 源于对 Dr. GRPO 的去偏**：有建议指出 **Dr. GRPO** 仍然存在偏差，且组别（group size）越小偏差越明显；为了使其无偏，只需将 **Dr. GRPO** 的 A_i 乘以修正项 **N/N-1**，从而得到 **LOOP (Leave-One-Out Proximal Policy Optimization)**，详见 [Dr GRPO 论文](https://github.com/sail-sg/understand-r1-zero/blob/main/understand-r1-zero.pdf)。
   - 移除 **PPO** 的裁剪（clipping）则得到 **RLOO (Reinforce Leave-One-Out)**。
- **基于偏差（Deviation）的 DPO 使创意 LLM 写作多样化**：一篇新[论文](https://arxiv.org/abs/2503.17126)探讨了如何同时提升创意写作 **LLMs** 的输出多样性和质量，通过在训练目标中加入“偏差（deviation）”来促进从稀有的高质量样本中学习。
   - 该研究将此方法应用于 **Direct Preference Optimization (DPO)** 和 **Odds Ratio Preference Optimization (ORPO)**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2503.17126">Modifying Large Language Model Post-Training for Diverse Creative Writing</a>：由于创意写作任务没有唯一的正确答案，训练用于执行这些任务的 Large Language Models (LLMs) 应该能够生成多样化的有效输出。然而，LLM post-training...</li><li><a href="https://x.com/zzlccc/status/1903162768083259703">Zichen Liu (@zzlccc) 的推文</a>：🪂理解类 R1-Zero 训练：一个批判性视角* DeepSeek-V3-Base 在 RL-tuning 之前就已经表现出“顿悟时刻（Aha moment）”了？？* RL-tuning 中不断增加的输出长度可能是由于...</li><li><a href="https://x.com/tassel_pierre/status/1903442097866236383">Tassel Pierre (@tassel_pierre) 的推文</a>：@elan_marko @zzlccc 是的，但你是按 token 应用 loss 的。因此，如果你有两个具有正向 advantage rewards 的补全，即使较长的那个正向奖励略低，因为它是被应用在...</li><li><a href="https://x.com/QGallouedec/status/1903872705184899492">Quentin Gallouédec (@QGallouedec) 的推文</a>：感谢 @ethayarajh @zzlccc！我有着和 @ethayarajh 完全相同的问题。在 trl 中，我们（不再）进行这种会导致响应层级长度偏差的 per-sequence normalization。相反，我们...</li><li><a href="https://x.com/WenhuChen/status/1903464313391624668">Wenhu Chen (@WenhuChen) 的推文</a>：这篇论文提供了一些非常有趣的见解：1. 之前人们发现 Qwen base 模型在 R1 训练中表现尤为出色，展现出强大的探索能力。- 这篇论文表明...</li><li><a href="https://x.com/leloykun/status/1903382502158500119">leloy! (@leloykun) 的推文</a>：我不确定是否有人已经指出了这一点，但 Dr. GRPO 仍然存在偏差，且 group size 越小，偏差越明显。为了使其无偏，只需将 Dr. GRPO 的 A_i 乘以...</li><li><a href="https://ai.stackexchange.com/questions/37958/where-does-the-proximal-policy-optimization-objectives-ratio-term-come-from">Proximal Policy Optimization 目标的 ratio 项是从哪里来的？</a>：我将使用 Proximal Policy Optimization 论文中使用的符号。需要什么样的近似才能得出带有 ratio $r_t(\theta)$ 的 surrogate objective（上方的等式 (6)）？</li><li><a href="https://ai.stackexchange.com/questions/37958/where-does-the-proximal-policy-opt">Proximal Policy Optimization 目标的 ratio 项是从哪里来的？</a>：我将使用 Proximal Policy Optimization 论文中使用的符号。需要什么样的近似才能得出带有 ratio $r_t(\theta)$ 的 surrogate objective（上方的等式 (6)）？</li><li><a href="https://github.com/huggingface/open-r1/issues/491">GRPO 应该关闭 Prefix Caching · Issue #491 · huggingface/open-r1</a>：我在推理时的运行性能远差于训练时的性能。经过调试，我认为 prefix caching 是幕后黑手。由于模型在不断地被...</li><li><a href="https://github.com/sail-sg/oat/blob/7619b79a8804e813419faeda22bdd35cc4d9b9bd/oat/algorithms/ppo.py#L231">sail-sg/oat 仓库中的 oat/oat/algorithms/ppo.py</a>：🌾 OAT：一个研究友好的 LLM online alignment 框架，包括 preference learning、reinforcement learning 等。- sail-sg/oat</li><li><a href="https://github.com/huggingface/trl/blob/07cfe1677e552b7d5c92b7740e5b2f0b057661d8/trl/trainer/grpo_trainer.py#L965">huggingface/trl 仓库中的 trl/trl/trainer/grpo_trainer.py</a>：使用 reinforcement learning 训练 Transformer 语言模型。- huggingface/trl</li><li><a href="https://github.com/huggingface/trl/blob/07cfe1677e552b7d5c92b7740e5b2f0b057661d8/trl/trainer/ppo_trainer.py#L573C1-L574C1">huggingface/trl 仓库中的 trl/trl/trainer/ppo_trainer.py</a>：使用 reinforcement learning 训练 Transformer 语言模型。- huggingface/trl</li><li><a href="https://github.com/sail-sg/oat/blob/7619b79a8804e813419faeda22bdd35cc4d9b9bd/oat/algorithms/ppo.py#L560">sail-sg/oat 仓库中的 oat/oat/algorithms/ppo.py</a>：🌾 OAT：一个研究友好的 LLM online alignment 框架，包括 preference learning、reinforcement learning 等。- sail-sg/oat
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[cv](https://discord.com/channels/1179127597926469703/1208183243447468093/1352749970393927773)** (6 messages): 

> `Operator Agent 的局限性，Infinibranch Browsers 作为解决方案，智能浏览器自动化` 


- **Operator Agent 缺乏管理技能**：成员们讨论了 **Operator Agent** 的局限性，指出它们在处理需要协调的复杂任务（如从数据集中提取信息）时表现吃力。一位成员评论说，需要 *一个管理 Agent 来指挥一个 Operator Agent 去获取单个数据集的详情*。
   - 一位成员对成功率有限表示沮丧，使用 Operator 仅完成了 **10** 个任务中的 **4** 个，而使用 Deep Research 完成了 **6** 个。
- **Infinibranch Browsers 达到 80% 成功率**：有人建议使用 [Morph Cloud 的 Infinibranch Browser](https://x.com/morph_labs/status/1902566171641266500) 这一可能方案来帮助扩展 browser-use Agent，在“为书籍列表查找 Amazon 链接”等任务上将成功率提高到约 **80%**。
   - X 上的原帖作者 Andrew Carr 需要将 **1000 多本书** 的链接提取到 Google Sheet 中，而 Operator 无法搞定。
- **Morph Cloud 扩展自主浏览器工作流**：[Morph Cloud](https://morph.so/blog/browser-morph-cloud/) 允许用户对完整的浏览器状态（包括身份验证和 Cookie）进行快照（snapshot）和分支（branch），从而更容易在多个并行实例中扩展自主浏览器工作流。
   - 该博客文章进一步解释了传统的网络爬虫方法为何已经过时，原因包括：重度使用 JavaScript 的单页应用、动态加载和无限滚动、访问数据所需的复杂用户交互、CAPTCHA 和复杂的机器人检测，以及需要理解上下文的多步工作流。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/andrew_n_carr/status/1901354501317288304">来自 Andrew Carr (e/🤸) (@andrew_n_carr) 的推文</a>：我有一个非常具体的 Agent 使用场景，难度刚好大到让网络爬虫失效。1. 我有一个 1000 多本书的列表；2. 我想找到它们的 Amazon 链接；3. 我希望将这些保存到...</li><li><a href="https://x.com/morph_labs/status/1902566171641266500">来自 Morph (@morph_labs) 的推文</a>：发布 Infinibranch Browsers。Morph Cloud 的 Infinibranch Browser 将 browser-use Agent 在下列书籍列表任务中的成功率扩展到约 80%。Operator 的成功率不超过 10%。引用 Andrew C...</li><li><a href="https://morph.so/blog/browser-morph-cloud/">使用 Morph Cloud 的远程浏览器：无限可扩展的浏览器自动化</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1352726958584365116)** (16 messages🔥): 

> `类 R1-Zero 训练，DeepSeek-V3-Base，RL-tuning 中的 GRPO 偏差，CoT 哲学，AI 论文中的数学错误` 


- **R1-Zero 训练：新见解出现**：一个 [Twitter 线程](https://x.com/zzlccc/status/1903162768083259703) 强调了关于 **类 R1-Zero 训练** 的关键观察，表明 **DeepSeek-V3-Base** 在 RL-tuning 之前就表现出了 *“顿悟时刻” (Aha moment)*。
   - 研究人员指出 **GRPO 中的潜在偏差** 导致输出长度不断增加，并在 [论文](https://github.com/sail-sg/understand-r1-zero/blob/main/understand-r1-zero.pdf) 中详细说明了发现，并提供了 [代码](https://github.com/sail-sg/understand-r1-zero)。
- **GRPO Loss 实现分析**：本周多篇论文讨论了 **1/o 项** 及其对较长样本的影响，认为该损失函数对冗长、重复行为的惩罚较少，同时对冗长、探索性的生成奖励不足。
   - 他们注意到，按问题归一化（per-question normalization）会惩罚 Batch 中的难题。
- **Chain of Thought 与推理**：一位成员质疑，目前的进展究竟是关于推理，还是仅仅利用 Token 来克服特定任务中 Next-token 补全/搜索的低效。
   - 另一位成员建议将 Chain of Thought (CoT) 作为一种语言模型推理的形式是可行的，并称推理的定义非常广泛。
- **关于论文计算的数学担忧**：AI2 Slack 频道中的讨论暗示论文中呈现的数学推导可能存在错误或异常。
   - 一些成员对论文中关于长度归一化偏差（length normalization bias）的论点表示困惑，相关讨论在链接频道中继续进行，并有一位成员提供了详细解释。



**提到的链接**：<a href="https://x.com/zzlccc/status/1903162768083259703?s=61">来自 Zichen Liu (@zzlccc) 的推文</a>：🪂理解类 R1-Zero 训练：一个批判性视角。* DeepSeek-V3-Base 在 RL-tuning 之前就已经表现出“顿悟时刻”？？* RL-tuning 中不断增加的输出长度可能是由于...

  

---

### **Interconnects (Nathan Lambert) ▷ #[lectures-and-projects](https://discord.com/channels/1179127597926469703/1223784028428177510/1353807028715393156)** (2 条消息): 

> `Claude PR, Header Copy Links` 


- **Claude 为标题复制链接发送 Pull Request**：一位成员分享了 **Claude** 提交的一个 [Pull Request](https://github.com/natolambert/rlhf-book/pull/82)，用于为一个 GitHub 仓库添加标题复制链接。
- **标题复制链接令人惊叹**：成员们发现悬停时显示的标题复制链接非常有趣且实用。
   - 他们附上了一张链接的 [截图](https://cdn.discordapp.com/attachments/1223784028428177510/1353807029223030835/Screenshot_2025-03-24_at_12.05.34_PM.png?ex=67e2fe8c&is=67e1ad0c&hm=41d19137d3231c38197bef45a02356a9b88f754b907ba8a3f1028543cb17349e&)，并指出这些功能通过 **Claude Code** *立即生效*。



**提及的链接**：<a href="https://github.com/natolambert/rlhf-book/pull/82">(experimental) Add heading anchor links for easy section linking by natolambert · Pull Request #82 · natolambert/rlhf-book</a>：为所有标题添加悬停时显示的待复制链接。链接会将带有片段标识符（fragment identifier）的当前 URL 复制到剪贴板。添加了用于设置锚点链接样式的 CSS。更新了 Makefile 以将新的 JS 文件复制到...

  

---


### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1353105005778833449)** (9 条消息🔥): 

> `China's Open Source AI Blitz, DeepSeek's Impact, US vs China AI Competition, Chinese consumer market for software, China commoditizing hardware` 


- **中国计划开源 AI 攻势**：根据 [这条推文](https://x.com/balajis/status/1903469483739730132)，中国旨在通过开源 AI 模型席卷市场，从而使 **AI 软件商品化**并促进其硬件销售。
   - 该策略是模仿、优化、规模化并以更低价格削减西方技术，类似于他们在制造业的做法，其中 **DeepSeek** 是关键参与者。
- **DeepSeek 引发科技市场暴跌**：根据 [这条推文](https://x.com/balajis/status/1903469483739730132)，**DeepSeek** 模型的发布导致美国科技股市值暂时蒸发了约 1 万亿美元，凸显了中国 AI 对全球市场的潜在影响。
   - DeepSeek 的创始人（**梁文锋**）已与中国高层官员会面，表明了显著的国家支持和对*无限资源*的获取。
- **中国的 AI 竞争**：一位成员表示，中国在开源 AI 方面的推动力源于激烈的国内竞争，旨在加速进步，而非单纯为了*击垮美国科技*。
   - 他们补充说，大多数中国顶尖实验室意识到开源是推动进步的最佳方式，因为*你的闭源模型在 3-6 个月左右就会变得无关紧要，不如加速发展*。
- **中国广告和数字服务收入低于美国**：一位成员指出，中国公司并非以破坏美国价值为目标。
   - 广告和数字服务的收入市场与美国不同，*中国的广告和数字服务收入远低于美国*，因此开源也更加可行。
- **中国消费者不愿为软件付费**：中国消费者普遍避免为软件和服务付费，学生和专业人士是主要的付费群体。
   - 消费市场主要由 **ByteDance** 占据，此前则是 **Kimi**。



**提及的链接**：<a href="https://x.com/balajis/status/1903469483739730132">来自 Balaji (@balajis) 的推文</a>：AI 过度生产。中国寻求将其互补品商品化。因此，在接下来的几个月里，我预计中国开源 AI 模型将发起全面攻势，涵盖从计算机视觉到机器人技术的各个领域...

  

---

### **Interconnects (Nathan Lambert) ▷ #[expensive-queries](https://discord.com/channels/1179127597926469703/1338919429752361103/1353032783139704883)** (17 messages🔥): 

> `Grok DeeperSearch, OpenAI Deep Research, Twitter Premium, HF model comparisons` 


- **Grok DeeperSearch 接近 OpenAI Deep Research**：据报道，全新的 **Grok DeeperSearch** 表现“非常出色”，在质量上接近 **OpenAI Deep Research**，考虑到其开发周期之短，这一点令人印象深刻。
   - 最初的 **Grok DeepSearch** 被认为“糟糕透顶”，因为它会从检索到的链接中产生内容幻觉（hallucinating），据一些用户称，那是当时最差的实现方案。
- **Twitter Premium 授予 Grok DeeperSearch 访问权限**：**Grok DeeperSearch** 的访问权限随 **Twitter Premium**（10 美元档位）提供，且仅限在 Grok 网站上使用。
   - 在有用户发推抱怨 **Grok DeepSearch** 表现不佳后，一位来自 xAI 的人员联系了该用户，并根据提供的对话和基准测试（benchmarks）对 **DeeperSearch** 进行了改进。
- **Deep(Re)search 实现的基准测试**：一位用户维护着一个 Markdown 文件，其中包含一组用于测试搜索和研究实现的问题，包括 **Grok DeeperSearch**。
   - 该基准测试包括一个宽泛的购物查询、一个具体的购物查询、一个通用的论文搜索提示词，以及 **Hugging Face** 上两个模型之间的表格/基准对比。
- **图像生成基准测试**：一位用户分享了他们的图像生成基准测试，包括诸如 *"A woman sitting at a poker table with cards in her hands"*（一个女人坐在扑克桌旁，手里拿着牌）和 *"Isometric pixel art of a waterfall"*（瀑布的等轴测像素艺术）等提示词。
   - 这些基准测试有助于比较不同模型的性能，并将为未来的发帖提供支持。



**Link mentioned**: <a href="https://fxtwitter.com/btibor91/status/1899917834496729259">Tweet from Tibor Blaho (@btibor91)</a>: @TheXeophon-bench

  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1352724292319576125)** (89 messages🔥🔥): 

> `Gemini Updates, Claude Code New Features, Model Context Protocol (MCP), AI Agents and Email, RF-DETR Object Detection Model` 


- **Gemini 更新深度解析**：Gemini 的 Dave Citron 加入了 @OfficialLoganK 的 Release Notes 播客，讨论了最近的更新，包括**个性化**、**Canvas**、**Audio Overviews** 和 **Deep Research**。
   - 讨论涵盖了从最近的应用发布到 **Gemini app** 个性化的未来等话题，包括对用户数据和隐私考量的见解。
- **Claude Code 获得 8 项新功能**：Anthropic 为 **Claude Code** 推出了 **8** 项新功能，旨在帮助开发者更快速、更智能地构建应用，详情记录在他们的 [工程博客](https://www.anthropic.com/engineering/claude-think-tool) 上。
   - 功能包括一个新的 "think" 工具，引发了关于其实现和价值的讨论，一些人将其比作 Chain of Thought 提示。
- **A16Z 对 MCP 生态系统的深度探讨**：A16Z 发布了一篇关于 **Model Context Protocol (MCP)** 的深度探讨，探索其作为 AI 模型执行、数据获取和工具调用的标准接口的潜力，因为 APIs 是互联网的第一个伟大统一者。
   - 该文章研究了 MCP 的用例、挑战，以及它如何改变 AI 与工具交互的方式，并指出 APIs 曾是互联网的第一个伟大统一者，但 AI 模型目前还缺乏对等的机制。
- **Roboflow 发布用于实时目标检测的 RF-DETR**：Roboflow 宣布了 **RF-DETR**，这是一个完全开源的实时目标检测模型，采用 Apache 2.0 协议，可在 [GitHub](https://github.com/roboflow/rf-detr) 上获取。
   - RF-DETR 在 **COCO** 数据集上实现了超过 **60 mAP** 的 **SOTA** 性能，其基础模型和大型模型的参数量分别为 **29M** 和 **128M**。
- **Browser Use 融资 1700 万美元为 Agent 构建 Web**：Browser Use 筹集了 **1700 万美元** 以推进 Web Agents 的发展，由 Felicis Ventures 领投，旨在将 Web Agents 提升到新水平。此前其初始原型仅用 **4 天** 完成并在 Hacker News 上发布。
   - 该公司正在招聘顶尖工程师来构建面向 LLMs 的互联网，并承诺提供一个具有纯粹软件极客团队文化的挑战性环境。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://x.com/pashpops/status/1902814965595246855?s=46">来自 Pasha Rayan (@Pashpops) 的推文</a>：介绍 A1Mail - 为 AI Agents 打造的电子邮件！📬🤖 TLDR：使用 A1Mail，你可以创建一个电子邮件地址，然后为你的 AI Agent 发送和接收邮件——无需每月为每个 Gmail 支付 12 美元...</li><li><a href="https://x.com/stuffyokodraws/status/1902757447984710076">来自 Yoko (@stuffyokodraws) 的推文</a>：[新文章] 🔥 深入探讨 MCP 和 AI Tooling 的未来。API 是互联网的第一个伟大统一者，但 AI 模型缺乏同等的东西。目前 MCP 的用例有哪些？挑战在哪里...</li><li><a href="https://x.com/GeminiApp/status/1902752852843331650">来自 Google Gemini App (@GeminiApp) 的推文</a>：在最新一期的 Release Notes 中，Gemini 的 Dave Citron 加入了 @OfficialLoganK，深入探讨了一些最新的 Gemini 更新。🎙️ 了解更多关于带有个性化功能、Canvas、Aud... 的 Gemini。</li><li><a href="https://x.com/kimmonismus/status/1903221838022324226?s=46">来自 Chubby♨️ (@kimmonismus) 的推文</a>：Sora 取消了所有付费层级的积分限制，提供无限次生成。这是一个很好的改变。</li><li><a href="https://x.com/karpathy/status/1903671737780498883">来自 Andrej Karpathy (@karpathy) 的推文</a>：我刚刚用 Swift “氛围编程”（vibe coded）了一个完整的 iOS 应用（之前没有用 Swift 编程过，虽然在这个过程中学到了一些），现在大约 1 小时后，它居然已经在我的实体手机上运行了。这太...</li><li><a href="https://x.com/theomediaai/status/1903448834451111988?s=61">来自 Theoretically Media (@TheoMediaAI) 的推文</a>：Sora 走向“无限”，但 20 美元的层级有水印、720p 且速度较慢。你知道什么能让我重新购买 Pro（200 美元）计划吗？为 Pro 用户发布“Big Daddy”版本的 Sora 模型。保留那个削弱版的...</li><li><a href="https://fxtwitter.com/karpathy/status/1903671737780498883)">来自 Andrej Karpathy (@karpathy) 的推文</a>：我刚刚用 Swift “氛围编程”（vibe coded）了一个完整的 iOS 应用（之前没有用 Swift 编程过，虽然在这个过程中学到了一些），现在大约 1 小时后，它居然已经在我的实体手机上运行了。这太...</li><li><a href="https://x.com/karpathy/status/1903671737780498883>)">来自 Andrej Karpathy (@karpathy) 的推文</a>：我刚刚用 Swift “氛围编程”（vibe coded）了一个完整的 iOS 应用（之前没有用 Swift 编程过，虽然在这个过程中学到了一些），现在大约 1 小时后，它居然已经在我的实体手机上运行了。这太...</li><li><a href="https://x.com/_catwu/status/1903130881205977320">来自 cat (@_catwu) 的推文</a>：这是 Claude Code 的重要一周。我们推出了 8 个令人兴奋的新功能，帮助开发者更快速、更智能地构建。以下是我们发布的所有内容的汇总：</li><li><a href="https://x.com/AnthropicAI/status/1903128670081888756">来自 Anthropic (@AnthropicAI) 的推文</a>：我们正在推出一个新的博客：Engineering at Anthropic。这是一个开发者可以找到实用建议以及我们关于如何充分利用 Claude 的最新发现的中心。</li><li><a href="https://x.com/leloykun/status/1903186153513291933">来自 leloy! (@leloykun) 的推文</a>：实际上有两种推理时计算（inference-time compute）：1. 在生成回答 Token 之前的思考。将其视为“草拟”或“规划”阶段。以及 2. ...</li><li><a href="https://fxtwitter.com/gergelyorosz/status/1904089127600975966)">来自 Gergely Orosz (@GergelyOrosz) 的推文</a>：目前，许多 AI 编程工具初创公司都在大量补贴运行 AI Agents 的实际成本。没有人能无限期地这样做。但那些开始按接近实际成本收费的公司...</li><li><a href="https://x.com/gergelyorosz/status/1904089127600975966>)">来自 Gergely Orosz (@GergelyOrosz) 的推文</a>：目前，许多 AI 编程工具初创公司都在大量补贴运行 AI Agents 的实际成本。没有人能无限期地这样做。但那些开始按接近实际成本收费的公司...</li><li><a href="https://x.com/tokumin/status/1902251588925915429?s=46">来自 Simon (@tokumin) 的推文</a>：🛳️ 在 NotebookLM 中推出交互式思维导图（Mindmaps）！我深受旧金山 Exploratorium 的启发——如果每个笔记本都能生成一套属于你自己的个人交互式理解玩具，那会怎样...</li><li><a href="https://x.com/kalomaze/status/1903366221333958999?s=61">来自 kalomaze (@kalomaze) 的推文</a>：@metalure 混合 Mamba，56b。约 20T Tokens (!!!) FP8 预训练。实际深度（64 层，约 15% 具有 Attention，其余为 Mamba）。蒸馏（不是 SFT，是实际的预训练蒸馏！）47b 变体。针对约 600 亿 Token...</li><li><a href="https://fxtwitter.com/taesung/status/1904220824435032528)">来自 Taesung Park (@Taesung) 的推文</a>：很高兴在 @reveimage 结束隐身模式！与 LLMs 相比，现在的文本转图像/视频模型缺乏逻辑。图像最初看起来似乎合理，但在仔细观察下就会分崩离析：绘画技巧...</li><li><a href="https://x.com/taesung/status/1904220824435032528>)">来自 Taesung Park (@Taesung) 的推文</a>：很高兴...</li>

<li>在 @reveimage 结束隐身模式！与 LLMs 相比，当今的 text-to-image/video 模型缺乏逻辑。图像最初看起来似乎合理，但在审视下就会破绽百出：绘画技巧...</li><li><a href="https://fxtwitter.com/karpathy/status/1886192184808149383)">来自 Andrej Karpathy (@karpathy) 的推文</a>：有一种我称之为 "vibe coding" 的新编程方式，在这种方式下，你完全沉浸在氛围中，拥抱指数级增长，甚至忘记了代码的存在。这之所以成为可能，是因为 LLMs（例如...</li><li><a href="https://x.com/karpathy/status/1886192184808149383>)">来自 Andrej Karpathy (@karpathy) 的推文</a>：有一种我称之为 "vibe coding" 的新编程方式，在这种方式下，你完全沉浸在氛围中，拥抱指数级增长，甚至忘记了代码的存在。这之所以成为可能，是因为 LLMs（例如...</li><li><a href="https://fxtwitter.com/TransluceAI/status/1904226873879806390)">来自 Transluce (@TransluceAI) 的推文</a>：为了解读 AI benchmarks，我们需要查看数据。顶层数据并不代表你所想的：可能存在损坏的任务、意外的行为或差一点就成功的情况。我们正在推出 Docent 来...</li><li><a href="https://x.com/TransluceAI/status/1904226873879806390>)">来自 Transluce (@TransluceAI) 的推文</a>：为了解读 AI benchmarks，我们需要查看数据。顶层数据并不代表你所想的：可能存在损坏的任务、意外的行为或差一点就成功的情况。我们正在推出 Docent 来...</li><li><a href="https://x.com/roboflow/status/1902810257652351228?s=46">来自 Roboflow (@roboflow) 的推文</a>：很高兴宣布 RF-DETR，这是目前用于实时 object detection 的 SOTA，完全开源并为社区提供 Apache 2.0 协议。更多内容即将推出，但 repo 和 Colab notebook 今天已可供你...</li><li><a href="https://fxtwitter.com/gregpr07/status/1903835252382224795)">来自 Gregor Zunic (@gregpr07) 的推文</a>：我们筹集了 1700 万美元来构建面向 Agents 的未来 Web 🤖 几个月前，Browser Use 只是一个周末实验想法，旨在观察 LLMs 是否能像人类一样浏览网页。仅用了四天，我们就构建了...</li><li><a href="https://x.com/gregpr07/status/1901686296902615122>>>">来自 Gregor Zunic (@gregpr07) 的推文</a>：Browser Use 正在招聘前 0.01% 的创始工程师，为 LLMs 构建互联网🔥我们（2 个人）已经构建了领先的 web agents 仓库——在短短 4 个月内获得了 45K+ GitHub stars。每天都有人...</li><li><a href="https://x.com/gregpr07/status/1903835252382224795>)">来自 Gregor Zunic (@gregpr07) 的推文</a>：我们筹集了 1700 万美元来构建面向 Agents 的未来 Web 🤖 几个月前，Browser Use 只是一个周末实验想法，旨在观察 LLMs 是否能像人类一样浏览网页。仅用了四天，我们就构建了...</li><li><a href="https://a16z.com/a-deep-dive-into-mcp-and-the-future-of-ai-tooling/">深入探讨 MCP 和 AI Tooling 的未来 | Andreessen Horowitz</a>：我们探讨了什么是 MCP，它如何改变 AI 与工具交互的方式，开发者已经在构建什么，以及仍需解决的挑战。&nbsp;</li><li><a href="https://x.com/ctnzr/status/1903228434232512878?s=61">来自 Bryan Catanzaro (@ctnzr) 的推文</a>：Nemotron-H：一个 Hybrid Mamba-Transformer LLMs 家族。* Hybrid 架构意味着在相同精度下速度提升高达 3 倍 * 以 FP8 训练 * 非常适合 VLMs * 权重和 instruct 版本即将推出。https...</li><li><a href="https://hamel.dev/blog/posts/field-guide/">快速改进 AI 产品的实战指南 – Hamel 的博客</a>：来自 30 多个生产实施的评估方法、数据驱动的改进和实验技术。</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3-0324">deepseek-ai/DeepSeek-V3-0324 · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/agents_as_tools.py">openai-agents-python/examples/agent_patterns/agents_as_tools.py at main · openai/openai-agents-python</a>：一个用于 multi-agent 工作流的轻量级、强大的框架 - openai/openai-agents-python</li><li><a href="https://x.com/WHinthorn/status/1903511723082232203">来自 WFH (@WHinthorn) 的推文</a>：有趣的事实，这就是我们在常规 meta-prompting 失败的情况下，让 Claude 成为优秀的 prompt engineer 的方法。https://github.com/hinthornw/promptimizer/blob/31a78b28123530571a8a098b020f5a7a5cfbc2ca/src/prompt...</li><li><a href="https://github.com/hinthornw/promptimizer/blob/31a78b28123530571a8a098b020f5a7a5cfbc2ca/src/promptim/optimizers/metaprompt.py#L238">promptimizer/src/promptim/optimizers/metaprompt.py at 31a78b28123530571a8a098b020f5a7a5cfbc2ca · hinthornw/promptimizer</a>：Prompt 优化草稿。通过在 GitHub 上创建账号来为 hinthornw/promptimizer 的开发做出贡献。</li><li><a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5188231">未找到标题</a>：未找到描述</li><li><a href="https://qwenlm.github.io/blog/qwen2.5-vl-32b/">Qwen2.5-VL-32B：更智能、更轻量</a>：QWEN

CHAT GITHUB HUGGING FACE MODELSCOPE DISCORD简介：今年 1 月底，我们推出了 Qwen2.5-VL 系列模型，受到了广泛关注和积极反馈...</li><li><a href="https://chat.qwenlm.ai)">未找到标题</a>: 未找到描述</li><li><a href="https://modelscope.cn/collections/Qwen25-VL-58fbb5d31f1d47)">魔搭社区</a>: 未找到描述</li><li><a href="https://www.oneusefulthing.org/p/the-cybernetic-teammate">The Cybernetic Teammate</a>: 在团队中引入 AI 可以提高绩效、提供专业知识并改善体验</li><li><a href="https://www.hbs.edu/ris/Publication%20Files/24-013_d9b45b68-9e74-42d6-a1c6-c72fb70c7282.pdf)">出版物 - 教师与研究 - 哈佛商学院</a>: 未找到描述</li><li><a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5162111)">页面未找到</a>: 未找到描述</li><li><a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4945566).">页面未找到</a>: 未找到描述</li><li><a href="https://x.com/_mchenco/status/1903520306305827051?s=46">来自 michelle (@_mchenco) 的推文</a>: Cloudflare 2025 年的第一个创新周刚刚结束，我之前说未来的每个产品都将由 Workers AI 驱动，这绝非玩笑。我们并不认为 AI 仅仅是一个垂直领域...</li><li><a href="https://blog.cloudflare.com/how-cloudflare-is-using-automation-to-tackle-phishing/">Cloudflare 如何利用自动化正面打击网络钓鱼</a>: Cloudflare 如何利用威胁情报和我们的开发者平台产品来自动处理网络钓鱼滥用报告。</li><li><a href="https://blog.cloudflare.com/ai-labyrinth/">将违规机器人困在 AI 迷宫中</a>: Cloudflare 如何利用生成式 AI 来减慢、迷惑并浪费那些不尊重“禁止抓取”指令的 AI 爬虫和其他机器人的资源。</li><li><a href="https://blog.cloudflare.com/take-control-of-public-ai-application-security-with-cloudflare-firewall-for-ai/">利用 Cloudflare 的 Firewall for AI 掌控公共 AI 应用安全</a>: Firewall for AI 可以发现并保护您的公共 LLM 驱动的应用，并与 Cloudflare WAF 无缝集成。现在加入 Beta 测试，掌控您的生成式 AI 安全。</li><li><a href="https://blog.cloudflare.com/cloudflare-for-ai-supporting-ai-adoption-at-scale-with-a-security-first-approach/">Cloudflare for AI：以安全优先的方式支持大规模 AI 采用</a>: 通过 Cloudflare for AI，开发者、安全团队和内容创作者可以利用 Cloudflare 的网络和工具组合来确保 AI 应用的安全、可观测性、韧性及可靠使用。</li><li><a href="https://blog.cloudflare.com/introducing-ai-agent/">介绍 Cloudy，Cloudflare 用于简化复杂配置的 AI Agent</a>: Cloudflare 的第一个 AI Agent Cloudy 旨在帮助 Cloudflare 管理员轻松理解复杂的配置。
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1353504654897446993)** (2 条消息): 

> `Rishi Agarwal 谈 Distillation，Swyx 的 Agent Engineering 演讲，Agent Engineering 要素，Agent 作为 ChatGPT 的增长路径` 


- **Agarwal 调研 Distillation 技术**: Deepmind 的 **Rishi Agarwal** 发布了一个简短的 [播客](https://youtu.be/O1AR4iL30mg)，调研了机器学习中的 **Distillation** 技术。
- **Swyx 进军 Agent Engineering**: **Swyx** 发布了关于 **Agent Engineering** 的 [新演讲和文章](https://x.com/swyx/status/1904256213661192405)。
   - 该演讲也在 [@latentspacepod](https://latent.space/p/agent) 上进行了直播，强调了在 @aiDotEngineer 全力投入 **Agent** 的原因。
- **揭秘 Agent Engineering 六要素**: 讨论中定义了 **Agent**（感谢 @simonw），并详细阐述了 **Agent Engineering 的六个要素**。
   - 它还探讨了 **Agent** 如何成为 **ChatGPT** 达到 **10 亿月活跃用户 (MAU)** 的途径。



**提到的链接**: <a href="https://x.com/swyx/status/1904256213661192405">来自 swyx 🌉 (@swyx) 的推文</a>: 🆕 演讲 + 文章: Agent Engineering https://latent.space/p/agent 为什么我们在 @aiDotEngineer 全力投入 Agent。定义 Agent（感谢 @simonw）。Agent Engineering 的六个要素。为什么 Agent 是 ChatGPT&...

  

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1352733334760984606)** (226 条消息🔥🔥): 

> `DORA report, Gemini API, AI code generation, Agile adoption, Ruby on Rails` 


- **Google Cloud 的 DORA report 探讨工程卓越性**：Google Cloud 的 [DORA report](https://dora.dev/research/2024/dora-report/) 深入研究了**工程卓越性 (engineering excellence)** 的指标，不过获取完整报告需要注册。
   - 一些人认为对“*工程卓越性*”的关注过于企业化，与原型设计中常用的“*yolo vibe code*”形成鲜明对比。
- **Discord 移动端应用将展示视频广告**：据 [ArsTechnica](https://arstechnica.com/gadgets/2025/03/discord-heightens-ad-focus-by-introducing-video-ads-to-mobile-apps-in-june/) 报道，Discord 移动端应用将于 6 月开始引入**视频广告**，为广告商提供展示预告片和优质内容的机会。
   - 用户对 Discord 为了准备 IPO 而进行的“*enshittifying*”（平台劣化）表示担忧，并将其与 X 平台进行了类比。
- **Gemini API 是廉价的引流产品 (Loss Leader)**：成员们发现 **Gemini API** 是一个非常便宜的 API，一位用户正在“*sonnet maxxing*”，而另一位用户称其为“*loss leader*”。
   - 也有人对依赖单一 AI 供应商可能带来的“*model lockin*”（模型锁定）风险以及公司间的文化差异表示担忧。
- **AI 代码生成正在取代手动编码**：一位成员提到 AI 编写了其公司 **80-90%** 的代码，另一位成员承认如今 AI 编写了他们 **99%** 的代码，导致机器人完成了所有工作。
   - 其他人表达了对“*template repos*”（模板仓库）的厌恶，并认为 AI 在为自己“重新造轮子”方面表现得更好。
- **Vibe Manifesto 发布**：[Vibe Manifesto](https://vibemanifesto.org/) 强调心流、迭代、增强、产品思维、重抽 (rerolling) 和人类品味。
   - 这些价值观分别与摩擦、完美、自动化、代码工匠精神、调试和技术限制形成对比。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://vibemanifesto.org/">Vibe Coding Manifesto</a>：新一代开发者的哲学</li><li><a href="https://dora.dev/research/2024/dora-report/">DORA | Accelerate State of DevOps Report 2024</a>：DORA 是一个长期的研究项目，旨在了解推动软件交付和运营绩效的能力。DORA 帮助团队应用这些能力，从而实现更好的...</li><li><a href="https://tenor.com/view/putting-on-my-sunglasses-ken-ryan-gosling-barbie-movie-shades-on-gif-812066675624542171">Putting On My Sunglasses Ken GIF - Putting on my sunglasses Ken Ryan gosling - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://github.com/ZachBeta/ruby_ai_llm_bot_for_good_discord">GitHub - ZachBeta/ruby_ai_llm_bot_for_good_discord</a>：通过在 GitHub 上创建账户来为 ZachBeta/ruby_ai_llm_bot_for_good_discord 的开发做出贡献。</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=1439059137#gid=1439059137">AI In Action: Weekly Jam Sessions</a>：暂无描述</li><li><a href="https://arstechnica.com/gadgets/2025/03/discord-heightens-ad-focus-by-introducing-video-ads-to-mobile-apps-in-june/">Discord heightens ad focus by introducing video ads to mobile apps in June</a>：Discord 在预期的 IPO 之前寻找更多赚钱方式。</li><li><a href="https://github.com/ZachBeta/threejs_fpv">GitHub - ZachBeta/threejs_fpv</a>：通过在 GitHub 上创建账户来为 ZachBeta/threejs_fpv 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1353176885789724783)** (1 条消息): 

> `Mobile Study Participants, AI Model Updates` 


- **征集移动端研究参与者**：团队仍在寻找参与者，以进行一项专注于移动端使用场景和想法的研究。
   - 鼓励感兴趣的人加入并分享他们的见解，以帮助团队了解更多信息。
- **AI 模型更新即将推出**：团队宣布了即将对其 AI 模型进行的更新。
   - 关于具体改进和新功能的更多细节将在未来几天内分享。


  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1352725684115734640)** (52 条消息🔥): 

> `NotebookLM 中的思维导图 (Mindmaps)，使用 NotebookLM 进行研究，NotebookLM 中的 HR 政策中心，用于文献搜索的 NotebookLM，外部用户共享 NotebookLM` 


- **思维导图 (Mindmaps) 逐渐在 NotebookLM 中推出**：一位用户注意到他的 NotebookLM 中没有思维导图功能，另一位用户回答说他在免费版本中已经有了，该功能正在逐步推出。
   - 并非所有人都位于同一服务器上，因此所有服务器完成更新需要一些时间。
- **NotebookLM：用于构建详尽报告的研究**：一位用户分享说，他使用 NotebookLM 进行研究并撰写详尽报告，以生成本地甚至区域性新闻，帮助人们了解情况。
   - 该用户还分享了一个关于 911 恶作剧电话及其法律后果的播客剧集链接 [911 Prank Call: The Felony Consequences](https://creators.spotify.com/pod/show/peezyproductions/episodes/911-Prank-Call-The-Felony-Consequences-e30gfec)。
- **NotebookLM：HR 政策中心**：一位用户询问是否有人将 NotebookLM 用作 HR 政策、员工手册和新员工入职的中心，以便他们可以提问并获得正确的答案。
   - 他尝试过，但答案并不总是准确，他想知道是否有某种方法可以对信息进行特定方式的组织。
- **NotebookLM：文献研究**：一位用户询问如何将 NotebookLM 用于文献研究，另一位用户回答说 NotebookLM 没有内置的搜索功能。
   - 尽管如此，它对于学习大学课程主题仍然非常有用。
- **NotebookLM：合同分析**：一位用户有 3 份包含手写数字/金额的单页合同。
   - 其中一份最初根本没有被提及。另一份被提到的金额是 EUR 700 或 EUR 760，而实际上是 EUR 400。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://creators.spotify.com/pod/show/peezyproductions/episodes/911-Prank-Call-The-Felony-Consequences-e30gfec">🚓 911 Prank Call: The Felony Consequences by Neural Network News</a>：佛罗里达州沃卢西亚县一名叫 Ava 的 11 岁女孩通过 911 文本虚报绑架案。副警长追踪到她位于奥兰治港的住所，揭穿了这场骗局。Ava 承认...</li><li><a href="https://open.spotify.com/episode/6a44wSFv8bc1T9x3mEE9Dq?si=tWnXTxqHQbqpky6bWqj0uw&nd=1&dlsi=d20a7ee755104caa">Sancocho con Limon - Quatsch Session 01</a>：FELD.FM · 剧集
</li>
</ul>

</div>
  

---

### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1352724286988746882)** (202 条消息🔥🔥): 

> `Mind Map 像素化修复、Mind Map 功能反馈、NotebookLM 对比 ChatGPT、访问新版 NotebookLM、NotebookLM 反馈方法` 


- **放大以获得清晰的 Mind Map 下载效果**：一位成员建议在下载 **Mind Map** 之前先放大标签页，以获得更大、更高质量的输出并解决像素化问题。
   - 该成员还宣称 *这个工具绝对是游戏规则改变者*，称赞其惊人的 Context window 和极低的幻觉率，甚至取消了对 **ChatGPT** 和 **Claude** 的订阅。
- **Mind Mapping 激发符号推理**：一位用户认为，完善 **Mind Mapping** 是迈向更有效、更智能 AI 的重要一步，并可能预示着符号推理（Symbolic reasoning）。
   - 他们建议，一旦知识可以被表达为意义网络，这些数据结构就可以通过简单的操作（如移植节点或添加中间节点）轻松纠正。
- **NotebookLM 不是 App，而是 PWA**：一位用户试图在 App 上更改语言，但另一位用户指出 **NotebookLM** 没有 App，而是一个渐进式 Web 应用（PWA）。
   - 他们建议删除该 App，在浏览器中加载带有 `?hl=LANGUAGE` 选项的 **NotebookLM**，然后重新安装 **PWA**。
- **播客语言可以被“强制”设定**：一位用户发现，尽管英语是官方唯一支持的语言，但通过在文本设置开头输入特定的 Prompt，可以“强制”生成其他语言的播客。
   - 他们使用 Prompt *PT-BR cria o podcast em português* 生成了葡萄牙语播客，强调这并不总是有效，但成功时非常酷。
- **Mind Map 功能评价褒贬不一**：一位用户认为新的思维导图是 **NotebookLM** 的一个很好的补充，但发现它存在重大缺陷。
   - 他们指出，思维导图需要不断重新生成才能更新，且缺乏主题之外的细节，需要反复导航，并建议 *主题和子主题可以在主题内部进行解释*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.playbook.com/s/notebooklm/the-deep-dive">Playbook</a>：Playbook 是一个现代的创意文件管理器。精美地存储、标记和整理您的文件和文件夹。设计师们，今天就注册并获得 4TB 存储空间！</li><li><a href="https://notebooklm.google.com/">无标题</a>：未找到描述</li><li><a href="https://support.google.com/notebooklm/answer/15724963?hl=en&ref_topic=14775295&sjid=1197563608642675832-NC">了解 NotebookLM 如何保护您的数据 - NotebookLM 帮助</a>：未找到描述</li><li><a href="https://support.google.com/notebooklm/answer/15678219?hl=en">升级到 NotebookLM Plus - NotebookLM 帮助</a>：未找到描述</li><li><a href="https://support.google.com/a/answer/14338836?sjid=14118684210403272528-EU&hl=en">导出用户数据 - Google Workspace 管理员帮助</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1352761126160437350)** (106 条消息🔥🔥): 

> `RWKV 架构开发，AI 模型可行性预测，EleutherAI 评估方法，用于 RL 的低精度数据类型，lm-evaluation-harness 的 MkDocs 站点` 


- **虚拟测试环境预测模型可行性**：一位成员提议建立一个虚拟测试环境（又称模拟器），在训练前预测 AI 模型的可行性，以减少资源浪费，通过在昂贵的真实训练发生之前消除不必要的失败实验，从而节省时间并加速 AI 创新。
   - 该成员表示，他们的目标*不是在预测 AI 机制行为方面达到 100% 的准确率*，而是创建一个至少能告诉我们模型是否有实际成功的机会，还是注定会尽早失败的系统。
- **新博客详细介绍了 EleutherAI 评估方法**：一位成员写了一篇关于 EleutherAI 评估方法的简短博客，并建立了一个 [MkDocs 站点以便于导航](https://slyracoon23.github.io/lm-evaluation-harness/)。
   - 他们也在等待[这个 PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/2832) 的审核。
- **贡献者被提醒注意 PR 中的 AI 生成内容**：一位成员被提醒注意使用 AI 为 Pull Request 生成内容，强调了审查贡献以避免增加垃圾内容的重要性。
   - 建议除非作者 100% 确定所有内容都是正确的，否则*最好撤回贡献，直到你确定为止*。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/cloneofsimo/sdxl_inversions/blob/800613b426785757fca4964badeb666218e59eee/sdxl.py#L86">sdxl_inversions/sdxl.py at 800613b426785757fca4964badeb666218e59eee · cloneofsimo/sdxl_inversions</a>: 通过在 GitHub 上创建一个账户来为 cloneofsimo/sdxl_inversions 的开发做出贡献。</li><li><a href="https://slyracoon23.github.io/blog/posts/2025-03-21_eleutherai-evaluation-methods.html">EleutherAI’s lm-evaluation-harness: Architecture and Configuration – Earl Potters</a>: 关于配置、任务架构和模型集成的综合指南</li><li><a href="https://slyracoon23.github.io/lm-evaluation-harness/">LM Evaluation Harness</a>: 未找到描述</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/2832">Add MkDocs Documentation with GitHub Actions Deployment by Slyracoon23 · Pull Request #2832 · EleutherAI/lm-evaluation-harness</a>: 描述：此 PR 将 MkDocs 集成引入 LM Evaluation Harness 仓库，显著增强了文档的可读性和可访问性。它提供了：MkDocs 设置：配置...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1352903760178974762)** (121 条消息🔥🔥): 

> `AI 模拟环境，生产级 LLM 中的持续学习（Continual learning），架构感知优化器（Architecture-aware optimizers），Transformer 块间的锐度差异（Sharpness Disparity），VectorAdam 优化器` 


- **用于研究的 AI 模拟器**：一名成员分享了一个用于测试 AI 创新的虚拟环境想法，可能节省**资金和资源**，详见附件 [Ai_simulator.pdf](https://cdn.discordapp.com/attachments/747850033994662000/1352903759839363083/Ai_simulator.pdf?ex=67e30110&is=67e1af90&hm=6dd1c8028d8932d9e8b64355594bcf7c338adbf09e986186ccd4322d9cbcf99b&)。
   - 其他人指出，在小规模上测试新架构已经相对便宜，在 **3090** 上训练一个 **L6D512** 模型一天大约只需 **5 美元**。
- **最优优化器推导困境**：成员们讨论了为特定架构推导最优优化器的难度，并指出即使对于 Transformer，尽管存在各种非常规架构，目前也尚未找到此类优化器。
   - 一名成员建议，如果能为任意架构推导出一个近乎最优的优化器，那将是*值得获奖的工作*。
- **VectorAdam 的旋转等变性（rotation equivariance）揭示**：VectorAdam 将二阶矩更新修改为每个梯度向量的向量范数平方，解决了 Adam 中的坐标系偏差，可能提高旋转等变性，如这篇 [VectorAdam 论文](https://www.dgp.toronto.edu/~zling/vector-adam/) 所示。
   - 有人指出 VectorAdam 与 Adafactor 并不相似，而更像是一个 **block size = hidden dim** 的分块近似。
- **收敛引理（Convergence lemmas）遭到质疑**：有人建议收敛引理可能并不重要，正则化项可以直接放入损失函数中，因此可以忽略 AdamW 的细节，或者将其放入单独的损失函数中。
   - 其他研究人员认为这是错误的，因为在不同的正则化下，你所寻找的最优点实际上有很大差异。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2503.17126">Modifying Large Language Model Post-Training for Diverse Creative Writing</a>：修改大语言模型（LLM）后训练以实现多样化的创意写作：由于创意写作任务没有唯一的正确答案，训练用于执行这些任务的大语言模型（LLMs）应该能够生成多样化的有效输出。然而，LLM 后训练...</li><li><a href="https://arxiv.org/abs/1907.04164">Which Algorithmic Choices Matter at Which Batch Sizes? Insights From a Noisy Quadratic Model</a>：在哪些 Batch Size 下哪些算法选择更重要？来自噪声二次模型的见解：增加 Batch Size 是加速神经网络训练的一种流行方法，但超过某个临界 Batch Size 后，更大的 Batch Size 带来的收益会递减。在这项工作中，我们研究了...</li><li><a href="https://arxiv.org/abs/2502.19002">The Sharpness Disparity Principle in Transformers for Accelerating Language Model Pre-Training</a>：Transformer 中的锐度差异原理（Sharpness Disparity Principle）用于加速语言模型预训练：Transformer 由不同的构建块组成，如 Embedding 层、归一化层、Self-attention 机制和逐点前馈网络（FFN）。因此，理解这些差异和...</li><li><a href="https://drive.google.com/file/d/1IIqxolKNn3cbQ9DaKTYqx5WIvJ04twTP/view">evolving_llms_through_text-based_self-play.pdf</a>：未找到描述</li><li><a href="https://www.dgp.toronto.edu/~zling/vector-adam/">VectorAdam for Rotation Equivariant Geometry Optimization</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1352956296466534400)** (20 messages🔥): 

> `mechinterp backlash, token level activations, SAE visualizations, single token activations, untied embeddings` 


- **MechInterp 面临学术界“反弹”**：成员们讨论了“mechinterp”品牌似乎遭到了学术界的“反弹”，因为其大部分内容都处于传统学术渠道之外。
   - 理论认为 mechinterp 处于主流学术渠道之外，而学术界对这种范式存在抵触。
- **分析 Token 激活的准确性**：一位成员正在提取 **SAE** 上的 token 级别激活，并质疑传递单个/成对 token 是否比传递整个句子能产生更准确的结果。
   - 他们发现第一个触发激活的 token 是 *holocaust*，但它并不是激活最强的 token，并想知道神经元激活是否具有上下文特定性。
- **用于可视化的 SAEviz 库**：在查看 neuronpedia 网站上每个特征/神经元的图表时，有人建议研究 **SAEviz**，这是一个使用 **logit lens** 进行此类可视化的库。
   - 讨论明确了这些可视化代表的是地面真值（ground truth）激活，而非近似值。
- **对单 Token 激活提出质疑**：一位成员质疑了单 token 激活的有效性，强调神经元只在上下文中活跃，孤立地分析它们没有意义。
   - 他们解释说激活受到前文上下文的影响；例如，短语 *I am a dictator I want to* 可能会改变 *to* 上的激活。
- **模型需要时间“预热”**：一位成员指出模型需要时间“预热”，在前 50 个 token 中，上下文特征往往会因为模型关注 `end-of-text` token 而被消融（ablated）。
   - 直觉是模型没有足够的信息来对上下文做出准确判断。


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1353676488788148275)** (1 messages): 

> `Recursive Design, GAN vs. CNN vs. RL Architectures` 


- **递归设计成为一种有前景的技术**：一位成员介绍了一种使用递归设计的新颖图表，将其与传统的 **GANs** (*Generative Adversarial Networks*) 区分开来。
   - 该成员强调他们的实现强调结构化组织而非顺序处理，利用 **CNNs** 进行过滤并利用 **RL** 优化响应。
- **替代架构**：用户提出了一种使用递归设计的替代架构。
   - 用户将该架构区分开来：**GAN** 用于表达，**CNN** 用于过滤，**RL** 用于响应优化。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1353833840694788237)** (1 messages): 

> `lm_eval update, CI test failures` 


- **请求更新 `lm_eval`**：一位成员正在起草一个 PR，将评估逻辑更新至最新版本 `lm_eval==0.4.8`，参考 [Evals PR](https://github.com/EleutherAI/gpt-neox/pull/1348)。
- **CI 测试失败**：一位成员观察到 **lm_eval 更新 PR** 以及另一个仅包含微小改动的测试 PR 的 CI 测试均告失败，询问仓库的 CI 是否健康，并参考了 [CI Test PR](https://github.com/EleutherAI/gpt-neox/pull/1349)。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/gpt-neox/pull/1348">将评估逻辑更新至最新的 `lm_eval` (0.4.8) 并支持无验证集的自动基准测试评估，由 Kyle1668 提交 · Pull Request #1348 · EleutherAI/gpt-neox</a>: 我正在训练一个模型，希望在整个数据集上进行训练。我不希望将数据集拆分为训练/验证/测试集。我希望在一系列基准测试上进行评估，其中一个引入了...</li><li><a href="https://github.com/EleutherAI/gpt-neox/pull/1349">[Throw Away] CI 完整性检查，由 Kyle1668 提交 · Pull Request #1349 · EleutherAI/gpt-neox</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1353440360520749210)** (1 条消息): 

> `StarVector, SpatialLM, Hugging Face Agents Course, Xet on the Hub, HF Welcome Page Makeover` 


- ****StarVector** 成为矢量图形大师**：一个新的基础模型 **StarVector** 已在 Hugging Face 发布，用于从图像和文本生成可缩放矢量图形代码，访问地址为 [Hugging Face](https://huggingface.co/collections/starvector/starvector-models-6783b22c7bd4b43d13cb5289)。
   - 初始发布版本包括 **starvector/starvector-1b-im2svg** 模型。
- ****SpatialLM** 探索 3D 领域**：**SpatialLM** 是一个旨在处理 3D 点云数据的 3D Large Language Model，已在 Hugging Face 发布：[manycore-research/SpatialLM-Llama-1B](https://huggingface.co/manycore-research/SpatialLM-Llama-1B)。
   - 它能生成结构化的 3D 场景理解输出，可以通过 [项目网站](https://manycore-research.github.io/SpatialLM) 和 [GitHub 仓库](https://github.com/manycore-research/SpatialLM) 进一步探索。
- **HF Agents 课程引入 LlamaIndex, LangChain 和 SmolAgents**：Hugging Face Agents 课程现在包含了 **LlamaIndex**、**LangChain** 和 **smolagents** 的集成，为学习者提供多样化的 Agent 框架方法。
   - 根据 [这条推文](https://x.com/ben_burtenshaw/status/1903025737633841170)，该课程旨在提供适用于不同框架的基础知识，使那些已经熟悉其中一个或多个框架的人也能轻松上手。
- ****Xet** 在 Hub 上加速**：Hugging Face 的 **Xet 团队** 已将首批模型和数据集仓库从 LFS 迁移到 Xet 存储。
   - 这是赋能 AI 构建者更有效地构建和协作处理大规模模型及数据集的一步，更多细节请参阅这篇 [博客文章](https://huggingface.co/blog/xet-on-the-hub)。
- **Hugging Face 翻新欢迎页面**：Hugging Face 欢迎页面进行了重大改版，提供了对社区 AI 应用、开源库、本地模型执行等内容的流线化访问。
   - 用户可以通过更新后的 [欢迎页面](https://huggingface.co/welcome) 探索 HF Spaces、开源库、本地模型和 Inference Playground 等各个板块。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/collections/starvector/starvector-models-6783b22c7bd4b43d13cb5289">💫StarVector Models - a starvector Collection</a>：未找到描述</li><li><a href="https://huggingface.co/manycore-research/SpatialLM-Llama-1B">manycore-research/SpatialLM-Llama-1B · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/ben_burtenshaw/status/1903025737633841170">Ben Burtenshaw (@ben_burtenshaw) 的推文</a>：@huggingface Agents 课程现在包含三个主要的 Agent 框架：LlamaIndex、LangChain 以及我们自家的 smolagents。我们致力于以独特的方式整合这三个框架，以便...</li><li><a href="https://huggingface.co/blog/xet-on-the-hub">Xet 已上线 Hub</a>：未找到描述</li><li><a href="https://huggingface.co/welcome">Hugging Face – 构建未来的 AI 社区。</a>：未找到描述</li><li><a href="https://huggingface.co/blog/endpoint-analytics">Inference Endpoints 中全新的分析功能</a>：未找到描述</li><li><a href="https://huggingface.co/blog/ai-action-wh-2025">AI Policy @🤗：对白宫 AI 行动计划 RFI 的回应</a>：未找到描述</li><li><a href="https://huggingface.co/blog/olympic-coder-lmstudio">Open R1：如何在本地使用 OlympicCoder 进行编程</a>：未找到描述</li><li><a href="https://huggingface.co/blog/nvidia-physical-ai">NVIDIA GTC 2025 为物理 AI 开发者带来的公告：新的开源模型和数据集</a>：未找到描述</li><li><a href="https://huggingface.co/blog/burtenshaw/gemma3-thinking">让 Gemma 3 思考</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1352720141007196330)** (136 条消息🔥🔥): 

> `ComfyUI Samplers, Open Schizo Leaderboard, Short Story Generator with Pytorch, Photorealism Settings for SD1.5/SDXL, Flux.1 Model Performance`

- **ComfyUI 采样器策略讨论**：成员们讨论了在 ComfyUI 中使用的最佳 **sampler_name**，寻求最佳配置建议，但对此了解不多。
   - 一位用户推荐使用 *dpmpp_2m_sde* 采样器和 *kl_optimal* 调度器，以在 **SD1.5** 和 **SDXL checkpoints** 中实现写实效果。
- **Open Schizo 排行榜展示**：Hugging Face 上发布了一个新的排行榜，展示了顶级模型。
   - 在 Hugging Face 上查看 [Open-Schizo-Leaderboard](https://huggingface.co/spaces/rombodawg/Open-Schizo-Leaderboard)。
- **模型集成协议 (MIP) 简化 LLM 驱动的服务**：一位用户正在寻求关于 **Model Integration Protocol (MIP)** 的反馈，该协议为 OpenAI 提出了一种更简单、更具扩展性的方法，利用反射（reflection）自动将现有的方法、类和 HTTP 端点转换为 JSON-RPC。
   - 该方法旨在大幅减少开发开销，同时保持平台独立性以及与任何 LLM 的兼容性，[Neurocaster-Server 实现](https://github.com/vishalmysore/neurocaster-server)展示了其用法。
- **Wan 模型首次推出 AutoencoderKL**：一位用户遇到了与 `diffusers` 库中的 `AutoencoderKLWan` 相关的导入错误，可能是由于使用了开发版本或错误的仓库。
   - 发现了一个 GitHub [issue](https://github.com/huggingface/diffusers/issues/10963)，解释了用户可能遇到了开发版本错误，因为 `AutoencoderKLWan` 尚未发布。
- **InferenceClient API 抛出身份验证错误**：一位用户报告在尝试使用 `InferenceClient` API 列出已部署模型时出现 **403 Forbidden** 错误，即使配置了允许调用 Inference Providers 的只读 Token。
   - 该错误表明代表用户调用 Inference Providers 的权限不足，一位用户发布了一个具有相同错误的[链接](https://huggingface.co/posts/kpadpa/282697879499561)。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://serapath-example1-orchestrator-agent.hf.space`">未找到标题</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/rombodawg/Open-Schizo-Leaderboard">尝试 Rombos-LLM-V2.5-Qwen-7b - rombodawg 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/hub/spaces-sdks-docker">Docker Spaces</a>: 未找到描述</li><li><a href="https://huggingface.co/chat/)">HuggingChat</a>: 让每个人都能使用社区最好的 AI 聊天模型。</li><li><a href="https://huggingface.co/spaces/fantaxy/fantasy-novel-kr/discussions">fantaxy/fantasy-novel-kr · 讨论</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/hmb/gradio-dataframe-upgrade">Gradio 的 Dataframe 已升级！🎨</a>: 未找到描述</li><li><a href="https://huggingface.co/posts/kpadpa/282697879499561">Hugging Face 上的 @kpadpa：“这意味着什么，我该如何修复它？‘此身份验证方法……’”</a>: 未找到描述</li><li><a href="https://docs.vllm.ai/en/v0.7.2/getting_started/examples/whisper.html">Whisper &#8212; vLLM</a>: 未找到描述</li><li><a href="https://aikval25.kattis.com/contests/aikval25/problems/windchill">Windchill &ndash; Kattis, 2025 年 AI 奥林匹克预选赛</a>: 未找到描述</li><li><a href="https://huggingface.co/posts/julien-c/158943939527784">Hugging Face 上的 @julien-c：“重要通知 🚨”</a>: 未找到描述</li></ul></div>

For Inference Providers who have built support for our…&quot;</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/api-inference/pricing">Pricing and Rate limits</a>: 未找到描述</li><li><a href="https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers">Wan-AI/Wan2.1-I2V-14B-480P-Diffusers · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/open-r1/update-3">Open R1: Update #3</a>: 未找到描述</li><li><a href="https://github.com/huggingface/diffusers/issues/10963">cannot import name &#39;AutoencoderKLWan&#39; from &#39;diffusers&#39; · Issue #10963 · huggingface/diffusers</a>: 描述错误 ImportError: 无法从 &#39;diffusers&#39; (/usr/local/lib/python3.10/dist-packages/diffusers/init.py) 导入名称 &#39;AutoencoderKLWan&#39;。复现代码：from diffusers import Auto...</li><li><a href="https://huggingface.co/docs/inference-endpoints/index">Inference Endpoints</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/cookbook/en/enterprise_dedicated_endpoints">Inference Endpoints (dedicated) - Hugging Face Open-Source AI Cookbook</a>: 未找到描述</li><li><a href="https://github.com/huggingface/text-generation-inference">GitHub - huggingface/text-generation-inference: Large Language Model Text Generation Inference</a>: Large Language Model Text Generation Inference。通过在 GitHub 上创建账号来为 huggingface/text-generation-inference 的开发做出贡献。</li><li><a href="https://huggingface.org/support">Expert Support – Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/fffiloni/diffusers-image-outpaint">Diffusers Image Outpaint - a Hugging Face Space by fffiloni</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/diffusers/using-diffusers/img2img">Image-to-image</a>: 未找到描述</li><li><a href="https://github.com/justinpinkney/stable-diffusion?tab=readme-ov-file#image-mixer">GitHub - justinpinkney/stable-diffusion</a>: 通过在 GitHub 上创建账号来为 justinpinkney/stable-diffusion 的开发做出贡献。</li><li><a href="https://github.com/TheDenk/images_mixing">GitHub - TheDenk/images_mixing: Сombine images using usual diffusion models.</a>: 使用常规扩散模型合并图像。通过在 GitHub 上创建账号来为 TheDenk/images_mixing 的开发做出贡献。</li><li><a href="https://huggingface.co/spaces?sort=trending&search=vton">Spaces - Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces?sort=trending&search=try+on">Spaces - Hugging Face</a>: 未找到描述</li><li><a href="https://archive.ph/2025.02.24-150819/https://medium.com/data-scientists-from-future/fine-tuning-open-source-language-models-a-step-by-step-guide-a38bed8df923">Fine-Tuning Open-Source Language Models: A Step-by-Step Guide | by Vi&#x2026;</a>: 未找到描述</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct">Qwen/Qwen2.5-VL-7B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/microsoft/Phi-4-multimodal-instruct">microsoft/Phi-4-multimodal-instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/cookbook/advanced_rag">Advanced RAG on Hugging Face documentation using LangChain - Hugging Face Open-Source AI Cookbook</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/transformers/tasks/asr">Automatic speech recognition</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1353192213567373354)** (5 条消息): 

> `audio processing, AI agents, Tokenisers, BPE, Unigram language modelling` 


- **深入音频冒险**：一位成员今天正在深入研究 **audio processing**（音频处理）。
- **出色 AI Agents 的框架**：一位成员今天正在攻克 **AI agents** 的**框架**。
- **Tokeniser 之争：BPE vs Unigram**：一位成员正在探索各种 **tokenisers** 的机制，特别是 **BPE** 和 **unigram language modelling**。
- **轻量级模型点亮笔记本电脑**：一位成员正在研究适合在开发笔记本电脑上运行和微调的**轻量级**、**可微调模型**。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1353166030863732786)** (8 条消息🔥): 

> `用于 HF Transformers Trainer 的 Logfire Callback，用于图像整理的 TrashLens，pdf2notes：AI 驱动的 PDF 转笔记工具，孩子们对 UI/UX 的反馈，本地 API 使用情况` 


- ****Logfire Callback** 记录训练事件！**: 一位成员为 **HF transformers Trainer** 创建了一个 [Logfire callback](https://github.com/louisbrulenaudet/logfire-callback)，用于记录训练事件。
   - 该工具旨在帮助跟踪和分析 Hugging Face 中 Transformer 模型的训练过程。
- ****TrashLens** 让混乱的图像变得井然有序！**: [TrashLens](https://github.com/0xrushi/TrashLens) 旨在让混乱的图像变得井然有序，帮助用户轻松专注于重要内容并释放空间。
   - 该工具的目标是简化图像整理流程，使管理和清理视觉数据变得更加容易。
- ****pdf2notes** 将 PDF 转换为条理清晰的笔记！**: [Pdf2Notes](https://github.com/AstraBert/pdf2notes) 是一款**由 AI 驱动的开源解决方案**，它利用 **LlamaParse** 和 **Llama-3.3-70B** 将非结构化的 PDF 转换为条理清晰的笔记。
   - 该工具使用 **DeepMind** 的 **Gemini 2 Flash** 进行多模态解析，并配备了一个用于获取更深层见解的聊天机器人，采用 **Gradio** 和 **FastAPI** 框架封装，并可以通过 **Docker** 在本地运行。
- **孩子们提供了宝贵的 UI/UX 反馈！**: 一位成员分享说，他的儿子在 UI 配色方面提供了帮助，并且非常喜欢这个工具，尤其是解锁新成就的过程。
   - 来自孩子们的反馈强调了在教育工具中加入吸引人的 UI 元素和成就系统的重要性。
- **关于无需 API 的本地运行的疑问！**: 一位成员询问 [pdf2notes](https://github.com/AstraBert/pdf2notes) 是否可以 **100% 本地运行而无需外部 API**，并对需要订阅 **Gemini** 和 **Groq** 表示担忧。
   - 他们批评了 **Docker** 的设置，认为对于那些更喜欢简单解决方案、不想安装额外应用程序的非高级用户来说过于复杂。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/0xrushi/TrashLens">GitHub - 0xrushi/TrashLens</a>: 通过在 GitHub 上创建账号来为 0xrushi/TrashLens 的开发做出贡献。</li><li><a href="https://github.com/louisbrulenaudet/logfire-callback">GitHub - louisbrulenaudet/logfire-callback: 一个用于将 Hugging Face Transformers 的训练事件记录到 Logfire 的回调函数 🤗</a>: 一个用于将 Hugging Face Transformers 的训练事件记录到 Logfire 的回调函数 🤗 - louisbrulenaudet/logfire-callback</li><li><a href="https://github.com/AstraBert/pdf2notes">GitHub - AstraBert/pdf2notes: 在几秒钟内将 PDF 转换为笔记📝</a>: 在几秒钟内将 PDF 转换为笔记📝。通过在 GitHub 上创建账号来为 AstraBert/pdf2notes 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1352896313557258272)** (6 条消息): 

> `用于视频标注的 Qwen，开源版 Opus clip，自动驾驶中的 LLM 和 VLM` 


- **Qwen 指导视频标注新手**: 一位成员就如何使用 **Qwen** 配合 **transformers** 库进行视频帧提取和标注寻求建议。
   - 另一位成员推荐了 [Qwen2.5-VL 官方 GitHub 仓库](https://youtu.be/4twSI2XFK2s) 以获取模型信息和快速入门示例。
- **开源版 Opus Clip 工具寻求帮助**: 一位成员正尝试创建一个开源版本的 **Opus Clip**（**视频二次创作工具**）。
   - 作者正在为其“杂乱的仓库和代码”寻求帮助，该项目利用 **yolov8** 和 **revideo** 来检测人物并对视频进行垂直分割。
- **LLM 和 VLM 推动自动驾驶走向未来**: 一位成员分享了他们关于自动驾驶中 **LLM** 和 **VLM** 的新 Substack 文章，强调了车辆能力的提升。
   - 文章引用了一篇综述论文《A survey for foundation models in autonomous driving》，可在 [arXiv:2402.01105](https://arxiv.org/abs/2402.01105) 查看。



**提到的链接**: <a href="https://samerattrah.substack.com/p/autonomous-driving-with-llms-vlms">使用 LLM, VLM 和 MLLM 的自动驾驶</a>: 讨论大语言/视觉模型在自动驾驶中的应用，以及最重要的进展和方法。

  

---

### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1353824302524530848)** (1 条消息): 

> `Gradio Deep Links` 


- **Gradio 5.23 支持 Deep Links！**: Gradio 5.23 引入了 **Deep Links**，允许直接链接到特定的输出（如图像或视频），例如[这个链接](https://abidlabs-black-forest-labs-flux-1-schnell.hf.space/?deep_link=oUq4ebmL1Ek)指向一张冠蓝鸦的图像。
   - 升级请使用 `pip install --upgrade gradio`。
- **Image.png**: 该图像显示了一个附件文件。
   - 该文件托管在 Discord 上。



**提及的链接**: <a href="https://abidlabs-black-forest-labs-flux-1-schnell.hf.space/?deep_link=oUq4ebmL1Ek">black-forest-labs/FLUX.1-schnell</a>: 未找到描述

  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1352937712549630033)** (1 条消息): 

> `黑客松时间，黑客松详情` 


- **黑客松日期仍是个谜**: 一位成员询问了黑客松的日期，表示很难找到相关信息。
   - 他们提到 **YouTube 直播**中说是 **3 月 22 日**，但没有找到确认信息。
- **缺少黑客松详情**: 用户无法找到任何关于 Hackathon 的相关信息。
   - 用户提到 YouTube 直播说就在今天，但没有任何详情。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1352742104954306560)** (33 条消息🔥): 

> `LangGraph 的严谨性，用于 smolagents 的本地 LLM，LangGraph 中的 Gemini，Notebook 的 API 成本，Agent 存储检索到的信息` 


- **尽管 LangChain 遭到诟病，LangGraph 仍收获粉丝**: 一位刚完成 **LangGraph** 模块的成员表示，相比于他们在 Twitter 上关注的、*备受诟病*的 **LangChain**，他们更喜欢 **LangGraph** 的*严谨性 (rigidness)*。
   - 其他人似乎也产生了共鸣。
- **运行 smolagents 的本地 LLM 需要高性能机器**: 成员们发现，要在 **smolagents** 上运行本地 LLM 并获得良好效果，需要较大的模型（约 **32B** 参数），这意味着需要一台性能强大的机器。
   - 他们尝试了像 **qwen coder 7B** 或 **deepsek-r1 7B** 这样的“小型”LLM，但在 smolagents 上的表现相当不稳定。
- **建立家庭实验室以降低 API 成本**: 成员们讨论了完成 Notebook 所需的 **API** 成本，那些不想付费的人正致力于建立足够的**家庭实验室 (home lab)** 来运行模型并通过 **API** 访问。
   - 有人提到 HuggingFace 的 InferenceClient API 对免费用户是免费使用的，限制为 300 次请求/小时。
- **Agent 将信息存储在哪里以便后续参考？**: 在课程的 Agentic RAG 章节 ([https://huggingface.co/learn/agents-course/unit2/smolagents/retrieval_agents](https://huggingface.co/learn/agents-course/unit2/smolagents/retrieval_agents)) 中，尚不清楚 LLM Agent 如何*存储*检索到的信息，以便在规划未来事件时轻松访问，从而优化后续任务的效率。
   - 有建议认为存储搜索结果的不是 LLM 而是 Agent，Agent 本身必须将其记录在某处，而不仅仅是在 Context 中。
- **API Token 问题已解决！**: 一位成员在使用 **HuggingFaceInferenceAPI** 运行代码时遇到问题，LLM 返回了无关的响应。
   - 问题已查明并解决，是 **API Token** 的问题，在本地运行时需要设置为只读 (read-only)。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2303.17651">Self-Refine: Iterative Refinement with Self-Feedback</a>: 与人类一样，大型语言模型 (LLM) 并不总是在第一次尝试时就生成最佳输出。受人类如何修改文本的启发，我们引入了 Self-Refine，这是一种用于改进...的方法。</li><li><a href="https://huggingface.co/learn/agents-course/unit2/smolagents/retrieval_agents#basic-retrieval-with-duckduckgo)">Building Agentic RAG Systems - Hugging Face Agents Course</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/agents-course">Welcome to the 🤗 AI Agents Course - Hugging Face Agents Course</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/customization/llms/SimpleIndexDemo-Huggingface_stablelm/">HuggingFace LLM - StableLM - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[open-r1](https://discord.com/channels/879548962464493619/1333465203865817088/1352799575294873652)** (9 messages🔥): 

> `r1, vllm, cuda kernel` 


- **关于 r1 训练课程引发辩论！**：一位成员询问了训练课程，称使用 **deepseek** 花了 *5 分钟* 才理解其中的幽默。
   - 另一位成员表示 **r1** *极其缓慢*，需要巨大的算力；他们的 **Scaleway R1 grid** 运行 *20 台机器* 约 **3 PFLOPS**，每天仅生成几百 MB 数据，因此使用 **llama** 并从查询响应对中逆向工程出思维 token 要快得多。
- **讨论 CUDA Kernel 改进**：一位用户询问是否正在使用 **vllm**，并提到正在进行一些 **cuda kernel 改进**。
   - 另一位成员简单地回答了 *no*。


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1352737566960386178)** (155 messages🔥🔥): 

> `MCP and K8s, Anthropic's MCP, MCP server directories, C# MCP SDK, Vercel's AI SDK with MCP Clients` 


- **测试 MCP Prompts 需要 K8s 设置**：为了测试 MCP prompts，特别是来自[此文件](https://github.com/strowk/mcp-k8s-go/blob/main/testdata/list_prompts_test.yaml)和[此测试](https://github.com/strowk/mcp-k8s-go/blob/10aa7fd54dd7839bbeeb6b8705243e8cdb67ca7e/testdata/with_k3d/list_k8s_namespaces_test.yaml#L50)的 prompts，需要 Kubernetes 设置。
   - [这里](https://github.com/Abiorh001/mcp_ev_assistant_server)提供了一个带有 prompts 的替代实现，用于管理电动汽车充电站。
- **用户称：MCP 并不复杂！**：一位用户对 MCP 被认为很复杂的看法表示困惑，并指出 *JSON RPC 并不难。使用 SDK 甚至更容易。与许多其他开发工作相比，制作 MCP server 或 client 相当容易*。
   - 他们建议只需 **1 个命令和 1 个参数** 就可以将任何内容添加到任何 LLM，无需公网 IP、TLS 证书或任何之前的障碍。
- **深入探索 MCP Server 仓库**：用户分享了一系列有用的 MCP server 目录，包括带有评分卡系统的 [Glama](http://glama.ai/mcp/servers)、组织良好且详尽的 [PulseMCP](https://www.pulsemcp.com/)，以及[官方 MCP GitHub](https://github.com/modelcontextprotocol/servers?tab=readme-ov-file#model-context-protocol-servers)。
   - 这些资源可以帮助开发者为他们的项目寻找和评估各种 MCP server。
- **新的 C# SDK 正式发布！**：微软发布了用于 Model Context Protocol server 和 client 的官方 **C# SDK**，详见[此处](https://github.com/modelcontextprotocol/csharp-sdk)。
   - 根据 [Vercel AI SDK 4.2](https://vercel.com/blog/ai-sdk-4-2)，这为开发者提供了使用 **JavaScript** 和 **TypeScript** 构建 **AI 应用** 的工具，并可集成到 [Next.js](https://nextjs.org) 和 [Svelte](https://svelte.dev/) 等 Web 框架中。
- **Zapier 集成 MCP 以获得更广泛的 AI 应用访问**：Zapier 发布了一个 MCP server，[为 AI 助手提供了超过 8,000 个集成入口](https://zapier.com/mcp)，以便与各种应用进行交互。
   - 这使得 AI 能够执行现实世界的任务，如发送消息、管理数据、安排事件和更新记录，将其能力扩展到文本生成之外。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://x.com/llmindsetuk/status/1885719128247296109">llmindset (@llmindsetuk) 的推文</a>：Microsoft 365 Copilot。“Agent”一词现在由数百万看到此屏幕的企业用户所定义。“企业数据保护（Enterprise data protection）”被赋予了显著位置。</li><li><a href="https://x.com/tom_doerr/status/1903972369443475471">Tom Dörr (@tom_doerr) 的推文</a>：从产品需求文档（PRD）到任务的工具。引用 Eyal Toledano (@EyalToledano) 的话：厌倦了 @cursor_ai 重写好代码或原地打转？介绍 Task Master ✨ 一个将你的 PRD 转换为...的 CLI。</li><li><a href="https://zapier.com/mcp">Zapier MCP—立即将你的 AI 连接到任何应用</a>：让你的 AI Assistant 与数千个应用交互的最快方式。无需复杂的 API 集成。</li><li><a href="https://block.github.io/goose/docs/getting-started/providers#local-llms-ollama">配置 LLM Provider | 代号 goose</a>：Goose 与多种 LLM Provider 兼容，允许你选择并集成你偏好的模型。</li><li><a href="https://VeyraX.com/mcp">VeyraX 的推文</a>：VeyraX 是 Agenic Component Interface</li><li><a href="https://github.com/FreePeak/">Free Peak</a>：独立开发者（Indie Hacker）。Free Peak 有一个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://glama.ai/mcp/servers/@gannonh/firebase-mcp">Firebase MCP</a>：Firebase MCP 服务器提供了一个标准化的接口来与 Firebase 服务进行交互，包括 Firebase Authentication、Firestore 和 Firebase Storage。</li><li><a href="https://github.com/Abiorh001/mcp_ev_assistant_server">GitHub - Abiorh001/mcp_ev_assistant_server：一个用于管理电动汽车（EV）充电站、行程规划和资源管理的强大服务器实现。该服务器为 EV 相关服务提供了一套全面的工具和 API。</a>：一个用于管理电动汽车（EV）充电站、行程规划和资源管理的强大服务器实现。该服务器提供了一套全面的工具和 API...</li><li><a href="https://github.com/strowk/mcp-k8s-go/blob/main/testdata/list_prompts_test.yaml">mcp-k8s-go/testdata/list_prompts_test.yaml at main · strowk/mcp-k8s-go</a>：连接到 Kubernetes 的 MCP 服务器。通过在 GitHub 上创建一个账号来为 strowk/mcp-k8s-go 的开发做出贡献。</li><li><a href="https://vercel.com/blog/ai-sdk-4-2">AI SDK 4.2 - Vercel</a>：AI SDK 4.2 引入了 MCP 客户端、推理（reasoning）、使用语言模型生成图像、消息部分、来源等。</li><li><a href="https://github.com/modelcontextprotocol/specification/discussions/220">MCP Hosting 工作组 · modelcontextprotocol/specification · Discussion #220</a>：提交前检查清单。我已确认这作为特定仓库的功能请求并不更合适。我已搜索现有讨论以避免重复。你的想法...</li><li><a href="https://github.com/modelcontextprotocol/python-sdk/pull/343">修复/base64 处理 (Issue #342) 由 evalstate 提交 · Pull Request #343 · modelcontextprotocol/python-sdk</a>：对 lowlevel/server.py 的单行更改 + 用于验证 Base64 解码是否非 URL 安全且符合 Client 预期的测试。动机和背景：传输二进制资源。</li><li><a href="https://github.com/spences10/mcp-sequentialthinking-tools">GitHub - spences10/mcp-sequentialthinking-tools：🧠 MCP Sequential Thinking Server 的适配版本，用于引导工具使用。该服务器为每个阶段最有效的 MCP 工具提供建议。</a>：🧠 MCP Sequential Thinking Server 的适配版本，用于引导工具使用。该服务器为每个阶段最有效的 MCP 工具提供建议。 - spences10/mcp-sequential.....</li><li><a href="https://github.com/strowk/mcp-k8s-go/blob/10aa7fd54dd7839bbeeb6b8705243e8cdb67ca7e/testdata/with_k3d/list_k8s_namespaces_test.yaml#L50">mcp-k8s-go/testdata/with_k3d/list_k8s_namespaces_test.yaml at 10aa7fd54dd7839bbeeb6b8705243e8cdb67ca7e · strowk/mcp-k8s-go</a>：连接到 Kubernetes 的 MCP 服务器。通过在 GitHub 上创建一个账号来为 strowk/mcp-k8s-go 的开发做出贡献。</li><li><a href="https://github.com/modelcontextprotocol/csharp-sdk">GitHub - modelcontextprotocol/csharp-sdk：由 Microsoft 维护的 Model Context Protocol 服务器和客户端的官方 C# SDK</a>：由 Microsoft 维护的 Model Context Protocol 服务器和客户端的官方 C# SDK - modelcontextprotocol/csharp-sdk</li><li><a href="https://glama.ai/mcp/servers/@heurist-network/heurist-mesh-mcp-server">Mesh Agent MCP Server</a>：一个 Model Context Protocol 服务器，将 Claude 连接到 Heurist Mesh API，提供对各种区块链和 web3 工具的访问，包括加密货币数据、代币安全、Twitter 情报等。</li><li><a href="https://github.com/heurist-network">Heurist</a>：Heurist 是一个去中心化的 AI-as-a-Service 云平台。</li>

. Heurist 有 22 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/modelcontextprotocol/servers?tab=readme-ov-file#model-context-protocol-servers)">GitHub - modelcontextprotocol/servers: Model Context Protocol Servers</a>: Model Context Protocol Servers。通过在 GitHub 上创建账号来为 modelcontextprotocol/servers 的开发做出贡献。</li><li><a href="https://github.com/FreePeak/db-mcp-server">GitHub - FreePeak/db-mcp-server</a>: 通过在 GitHub 上创建账号来为 FreePeak/db-mcp-server 的开发做出贡献。</li><li><a href="https://github.com/punkpeye/awesome-mcp-servers/pull/355">Update README: Add multi-database MCP server built with Golang by linhdmn · Pull Request #355 · punkpeye/awesome-mcp-servers</a>: 添加使用 Golang 构建的多数据库 MCP server，支持 MySQL &amp; PostgreSQL，作为 https://github.com/FreePeak/db-mcp-server 的替代方案。
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1352750945901084674)** (29 条消息🔥): 

> `mcpwizard, vscode-mcp, DICOM servers MCP, google sheet MCP server, Narrative Spittoon Inversion project` 


- ****MCPwizard** 简化 Server 创建**：一位成员介绍了 [mcpwizard](https://www.npmjs.com/package/mcpwizard)，这是一个用于简化创建和部署 **MCP servers** 的 CLI 工具，重点介绍了初始化项目以及向 Claude 助手添加自定义工具等功能。
   - 该工具的 [GitHub repo](https://github.com/yoannarz/mcpwizard) 也被分享出来，以获取社区反馈和贡献。
- ****VS Code MCP** 获得社区好评**：成员们分享了一个他们期待已久的 [VS Code MCP](https://github.com/block/vscode-mcp)。
   - 在这个 [Youtube Short](https://www.youtube.com/shorts/gddEgvCLrgU) 中展示了它的实际运行情况。
- ****DICOM MCP** 临床影像 Server**：一位成员创建了一个用于与 **DICOM servers** 交互的 MCP server，使 AI 助手能够查询医疗影像系统以获取患者扫描结果和临床报告，可在 [christianhinge.com](https://www.christianhinge.com/projects/dicom-mcp/) 获取。
   - 相关的 **GitHub repo** 位于[此处](https://github.com/ChristianHinge/dicom-mcp)。
- ****Google Sheets MCP** 用于直接编辑**：一位成员构建了一个 **Google Sheet MCP server**，允许 Claude 直接编辑电子表格，简化了数据处理和公式调整，如[这条推文](https://x.com/xing101/status/1903391600040083488)中所述。
   - 代码可以在[此处](https://github.com/xing5/mcp-google-sheets)找到。
- ****Automated Debugger MCP Server** 增强功能**：一位成员一直在改进他们的 [automated debugger MCP server](https://github.com/jasonjmcghee/claude-debugs-for-you)，鼓励其他人尝试并做出贡献。
   - 该 server 允许 LLM *设置断点、运行代码、在断点之间移动以及评估表达式*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lokka.dev/">Lokka | Lokka</a>: Lokka 是一款 AI Agent 工具，它将 Microsoft Graph 的强大功能带给在本地桌面运行的 AI Agent（如 GitHub Copilot 和 Claude）。</li><li><a href="https://github.com/evalstate/mcp-webcam/.">GitHub - evalstate/mcp-webcam: Capture live images from your webcam with a tool or resource request</a>: 通过工具或资源请求从网络摄像头捕获实时图像 - GitHub - evalstate/mcp-webcam: Capture live images from your webcam with a tool or resource request</li><li><a href="https://github.com/gotohuman/gotohuman-mcp-server">GitHub - gotohuman/gotohuman-mcp-server</a>: 通过在 GitHub 上创建账户，为 gotohuman/gotohuman-mcp-server 的开发做出贡献。</li><li><a href="https://github.com/jasonjmcghee/claude-debugs-for-you">GitHub - jasonjmcghee/claude-debugs-for-you: Enable any LLM (e.g. Claude) to interactively debug any language for you via MCP and a VS Code Extension</a>: 使任何 LLM（例如 Claude）能够通过 MCP 和 VS Code Extension 为你交互式地调试任何语言 - jasonjmcghee/claude-debugs-for-you</li><li><a href="https://github.co">GitHub · Build and ship software on a single, collaborative platform</a>: 加入全球应用最广泛、由 AI 驱动的开发者平台，数百万开发者、企业和最大的开源社区在这里构建推动人类进步的软件。</li><li><a href="https://www.christianhinge.com/projects/dicom-mcp/"> Agentic healthcare LLMs | Christian Hinge </a>: 未找到描述</li><li><a href="https://github.com/ChristianHinge/dicom-mcp">GitHub - ChristianHinge/dicom-mcp: Model Context Protocol (MCP) for interacting with dicom servers (PACS etc.)</a>: 用于与 DICOM 服务器（PACS 等）交互的 Model Context Protocol (MCP) - ChristianHinge/dicom-mcp</li><li><a href="https://github.com/Kvadratni/speech-mcp">GitHub - Kvadratni/speech-mcp: Speech MCP: A Goose MCP extension for voice interaction with audio visualization</a>: Speech MCP：一个用于语音交互并带有音频可视化的 Goose MCP 扩展 - Kvadratni/speech-mcp</li><li><a href="https://github.com/MubarakHAlketbi/game-asset-mcp">GitHub - MubarakHAlketbi/game-asset-mcp: An MCP server for creating 2D/3D game assets from text using Hugging Face AI models.</a>: 一个使用 Hugging Face AI 模型从文本创建 2D/3D 游戏资产的 MCP 服务器。 - MubarakHAlketbi/game-asset-mcp</li><li><a href="https://github.com/MushroomFleet/UNO-MCP">GitHub - MushroomFleet/UNO-MCP: Unified Narrative Operator</a>: 统一叙事操作员（Unified Narrative Operator）。通过在 GitHub 上创建账户，为 MushroomFleet/UNO-MCP 的开发做出贡献。</li><li><a href="https://x.com/xing101/status/1903391600040083488">Tweet from Xing Wu (@xing101)</a>: 每个人都在议论 #MCP，原因如下：一个周末项目解决了我长期以来的痛点。不再需要将数据表复制到电子表格，也不再需要从 LLM 对话中解码复杂的公式指南...</li><li><a href="https://github.com/xing5/mcp-google-sheets">GitHub - xing5/mcp-google-sheets</a>: 通过在 GitHub 上创建账户，为 xing5/mcp-google-sheets 的开发做出贡献。</li><li><a href="https://github.com/yoannarz/mcpwizard">GitHub - yoannarz/mcpwizard: A package to help you create and deploy MCP servers</a>: 一个帮助你创建和部署 MCP 服务器的包 - yoannarz/mcpwizard</li><li><a href="https://shorturl.at/sLWsr">MCPwizard helps you building mcp servers !</a>: 使用 Loom 录制屏幕和摄像头的快速视频。清晰轻松地解释任何事情——并跳过会议。混合办公场所的必备工具。
</li>
</ul>

</div>
  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1352725930593878149)** (102 条消息🔥🔥): 

> `语音转文本解决方案，GPT4All 与 NSFW 内容，LocalDocs 消失，用于办公任务的 LLM，在多台设备上运行模型` 


- **提示词技巧至关重要**：成员们讨论了如果希望语言模型以特定语言（如德语）回答，最好用该语言编写系统消息，以避免触发 *"Im Kontext Lernen"*（语境学习）。
   - 进一步建议**避免使用包含 *"nicht"* 和 *"don't"* 等词的否定句**可以改善结果，并建议重新组织指令以使用主动动词。
- **Nemo 的细节说明**：提到了 [Mistral Nemo 是一个 12b 模型](https://huggingface.co/mistralai)，而 Mistral 24b 是 Mistral 3 或 Mistral 3.1，并讨论了项目中特定模型的细节。
   - 关于准确识别模型产生了困惑，一位成员强调需要精确的模型信息以避免问题。
- **GPT4All 的 LocalDocs 消失**：一位用户报告称其整个 LocalDocs 目录无故消失，引发了对潜在原因的讨论，如**安装文件夹的更改**或**缺乏管理员权限**。
   - 成员们建议备份 *localdocs.db* 文件和原始文档以防止数据丢失，并指出 Windows 11 更新可能因更改盘符而导致该问题。
- **LLM 关注医疗办公效率**：成员们讨论了在医疗办公环境中使用本地 LLM 帮助医生创建报告和辅助治疗的潜力，重点是让系统从过去的听写笔记中学习。
   - 然而，有人提醒说，由于存在**幻觉（confabulation）风险**以及对精确信息的需求，**LLM 可能不适合处理财务或医疗数据**。
- **GPT4All 缺乏视觉能力**：一位成员询问 GPT4All 运行的模型是否具有视觉能力，确认结果是 **GPT4All 不支持视觉能力**。
   - 建议使用 **LM-Studio** 等替代工具作为视觉相关任务的选择。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/mistralai">mistralai (Mistral AI_)</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1broa8h/is_there_a_way_for_me_to_use_multiple_computers/">Reddit - 互联网的核心</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1jcm5p2/ocr_llm_for_invoice_extraction/">Reddit - 互联网的核心</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1352831709149794374)** (7 条消息): 

> `High performance software, Vendor lock-ins, OpenCL, OpenMP, OpenACC, Vulkan’s Compute API, and SYCL, Democratizing AI Compute, Hardware Lottery` 


- **探索高性能软件领域**：一位成员正在探索为各种设备和行业需求编写**高性能软件 (high-performance software)** 的现状，特别是关于厂商锁定 (vendor lock-ins) 以及将项目移植到手机或嵌入式设备的必要性。
   - 他们请求推荐论文、搜索关键词或作者，以便更好地理解现有的权衡和选择。
- **开放且可移植的 API**：一位成员建议从开放且可移植的 API 开始，如 **OpenCL**、**OpenMP**、**OpenACC**、**Vulkan’s Compute API** 和 **SYCL**，并引用了它们被创建的详尽理由。
   - 他们还指出 **POCL** 是一个拥有相关论文的学术项目。
- **民主化 AI 计算系列**：一位成员链接了 Chris Lattner 的 "[Democratizing AI Compute](https://www.modular.com/blog/democratizing-compute-part-1-deepseeks-impact-on-ai)" 系列文章，强调了**更好的硬件利用率**如何显著减少对昂贵 GPU 的需求。
   - 该系列包括关于 **CUDA**、**OpenCL** 和 **AI 编译器 (TVM 和 XLA)** 的文章。
- **硬件彩票 (The Hardware Lottery)**：一位成员推荐了 Sara Hooker 的论文 "[The Hardware Lottery](https://arxiv.org/abs/2009.06489)"，该论文讨论了硬件和软件如何决定研究想法的成败。
   - 摘要指出，该论文*引入了“硬件彩票”一词，用来描述一个研究想法之所以胜出，是因为它适合现有的软件和硬件，而不是因为该想法优于其他研究方向*。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.modular.com/blog/democratizing-compute-part-1-deepseeks-impact-on-ai">Modular: Democratizing AI Compute, Part 1: DeepSeek’s Impact on AI</a>：文章的第一部分，探讨了在 DeepSeek 发布背景下，超越 CUDA 的 AI 硬件加速的未来。</li><li><a href="https://arxiv.org/abs/2009.06489">The Hardware Lottery</a>：硬件、系统和算法研究社区在历史上拥有不同的激励结构，以及相互交流的波动动力。这种历史性的处理方式是……
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1353002566157602847)** (82 条消息🔥🔥): 

> `Mojo Logging Library, Mojo Formatter Tool, Mojo Dict Default Values, GPU Support for Windows, Mojo Inline Assembly Documentation` 


- **Mojo 中的日志库仍处于开发中 (WIP)**：标准库中的日志库正在开发中，但正在进行重构；在日志库被视为完成之前，需要完整的序列化 (serialization) 以及可能的反射 (reflection) 支持。
   - 据一位成员所说，*我们需要在称日志库完成之前完成序列化，这可能意味着需要反射。*
- **Mojo 拥有内置格式化工具**：Mojo 包含一个内置的格式化工具 `mojo format`，类似于 Python 中的 Black 或 Rust 中的 `fmt`，用于代码格式化。
- **Dict 缺少默认值生成功能**：Mojo 的 `Dict` 更像 Python 的 `dict`，不包含像 `defaultdict` 那样生成默认值的功能。
- **Windows GPU 支持令 Mojo 开发者感到沮丧**：Windows 的 GPU 支持非常困难，因为 Windows 编译器工具链使用起来非常痛苦；大多数人不会在 Windows 上运行企业级 GPU 集群，因此改进工具链的动力不足。
- **Mojo 的内联汇编 (Inline Assembly) 文档比较混乱**：成员们注意到 Mojo 中内联汇编的文档有些混乱。
   - 一位成员说 *那是时候骚扰 Joe 让他写文档了*，但随后紧跟着一句 *不要骚扰*。



**提及的链接**：<a href="https://forum.modular.com/t/question-vpermi2b-inline-assembly-output-incorrect-in-loop-context-due-to-register-allocation/1091/2?u=sora">Question: vpermi2b inline assembly output incorrect in loop context due to register allocation</a>：也许你可以尝试这个：`from sys import llvm_intrinsic` `alias T = SIMD[DType.int8, 64]` `@always_inline("nodebug") fn vpermi2b(a: T, b: T, idx: T) -> T: return llvm_intrinsic["llv...`

  

---

### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1353465638743707820)** (3 messages): 

> `MAX Platform, pixi.toml, max-pipeline, Python model graphs, magic CLI` 


- **新人询问关于 MAX Platform 的问题**：一位新用户询问了关于修改 **max/pipeline** 目录以及通过 [pixi.toml 文件](https://github.com/modular/max/tree/main/src/max)在 **MAX Platform** 中测试更改的问题。
   - 具体来说，他们好奇如何在不将其作为依赖项下载的情况下修改 **max-pipeline**。
- **编辑 Python model graphs**：一位成员解释说，虽然 **Python model graphs** 的文档尚不完善，但 **MAX pipelines** 模块的 Python 源码是下载到本地的。
   - 对 `.modular/envs/max-pipelines/lib/python3.12/site-packages/max/pipelines`（或 `.magic` 环境中的类似位置）中这些本地文件的更改应该会在运行 pipeline 时体现出来。
- **通过 Python 运行 max-pipelines**：原帖作者询问是否可以直接使用 Python 运行 **max-pipelines**，而不是使用 **magic CLI**，以便添加更多命令行参数。
   - 目前还没有关于此方法可行性的直接回复。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/modular/max/tree/main/src/max">max/src/max at main · modular/max</a>：MAX Platform（包含 Mojo）。通过在 GitHub 上创建账号为 modular/max 的开发做出贡献。</li><li><a href="https://github.com/m">m - 概览</a>：打字员、工程师、代码诗人、优美数据结构的爱好者。 - m
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1352740201688207571)** (4 messages): 

> `AGNTCY, Large-Scale Structured Extraction, Deepseek R1 + LlamaIndex RAG app, WeAreDevs WebDev & AI Day` 


- **用于 Agentic 交互的 AGNCY 倡议出现**：Luke 讨论了 **AGNCY** 背后的动机，这是一项旨在为 [Agentic 交互创建开放标准](https://t.co/I558Qe2u4n)的努力。
- **针对复杂文档的大规模结构化提取**：LlamaIndex 重点介绍了如何对复杂文档执行**大规模结构化提取**，从带有嵌套子模式的 Pydantic 模式中提取 **50-100 个字段**，这需要极高的准确性。
   - 更多详情请见[此处](https://t.co/tO1vACKTGo)。
- **Deepseek R1 与 LlamaIndex 构建 RAG**：LlamaIndex 推荐了来自 Akshay Pachaar 的一个项目，该项目集成了 **Deepseek AI**，并使用 **LlamaIndex** 进行编排，使用 **Deepseek AI R1** 进行推理，使用 **Ollama** 在本地部署 R1，并使用 **Streamlit** 构建 UI；更多详情请见[此处](https://t.co/KS26JUkwz0)。
- **WeAreDevs WebDev & AI Day 即将到来**：LlamaIndex 为本周四的 **WeAreDevs WebDev & AI Day** 做宣传，承诺将由行业专家分享关于 **AI 如何改变 Web 开发**及其对软件开发影响的见解，更多信息请见[此处](https://t.co/c5N5BJ34mr)。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1352741141925203999)** (71 条消息🔥🔥): 

> `Haystack 卸载与 LlamaIndex 安装、Ollama 集成错误、RTX 3060 Token 问题、自定义 AI 面试准备、Agent 工作流超时错误` 


- ****LlamaIndex + Ollama = 完美的 RAG？****：一位成员寻求帮助，希望使用 **LlamaIndex**、**Ollama** 及相关集成构建 RAG 流水线，并从 Deepseek 获取了一段代码片段来开始，但遇到了依赖问题。
   - 该错误是由函数参数命名不正确（应为 **model** 而非 **model_name**）引起的，虽然错误已解决，但生成的答案仍不符合预期。
- ****打造自定义 AI 面试准备方案****：一位成员正在使用 **Llama 3.2**、**Sonnet 3.7** 和 **Dolphin** 混合成一个 16B 模型，结合 RAG 和自定义微调构建本地 AI，并梦想进入 AI/科技公司工作。
   - 他正试图让他的 AI 能够*申请 AI/科技公司并通过面试*，他在面部追踪、Blender、Unity、PowerShell 和 TTS 方面拥有经验。
- ****超时导致 Agent 工作流中断！****：一位成员报告称，由于 **OpenAI** 端点未处理的 **timeout errors**，他们的 Agent 工作流发生了崩溃。
   - 建议捕获 `WorkflowRuntimeException` 或 `Exception`，而不是 `WorkflowTimeoutError`。
- ****Hugging Face vs Ollama：哪种 LLM 更易于配置？****：成员们讨论了在本地使用 **Hugging Face** 模型进行 RAG 聊天，一位用户建议 **Ollama** 更易于配置。
   - 尽管存在争议，但还是提供了一些有用的 **Hugging Face Embedding** 示例链接，例如[这个 notebook](https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/huggingface.ipynb)。
- ****JSONL 数据集与 Git：天作之合还是数据灾难？****：一位成员思考将数据集作为 **JSONL** 文件存储在 **Git** 中的明智性，寻求关于潜在缺点的见解。
   - 该问题没有具体的回答，但有人提到 *GitHub 会跟踪每一份文档的更新*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/examples/workflow/function_calling_agent/">Workflow for a Function Calling Agent - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/embeddings/huggingface/">Local Embeddings with HuggingFace - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/huggingface/">Hugging Face LLMs - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_cloud_services/blob/main/examples/parse/multimodal/multimodal_rag_slide_deck.ipynb">llama_cloud_services/examples/parse/multimodal/multimodal_rag_slide_deck.ipynb at main · run-llama/llama_cloud_services</a>：云端知识 Agent 与管理。通过在 GitHub 上创建账户，为 run-llama/llama_cloud_services 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1353699437750255668)** (1 条消息): 

> `多 Agent 系统、程序级退避机制、函数调用` 


- **关于通过函数调用触发 Agent 的辩论**：成员们正在辩论，在多 Agent 系统中，单个 Agent 通过 **function calling** 触发其他单个 Agent 是否可以取代**程序级退避机制（backoff mechanisms）**。
   - 他们正在考虑这两种设置是否可能在某些场景下重叠，从而实现相同的功能。
- **探索退避机制的替代方案**：讨论集中在：使用单个 Agent 通过函数调用触发其他 Agent，是否是程序级退避机制的可行替代方案。
   - 目标是确定这种方法是否能在多 Agent 系统中实现类似的功能，从而可能提供更精简的解决方案。


  

---

### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1352766855911047238)** (25 条消息🔥): 

> `RAG 源返回、数据保留政策、Cohere 聊天安全信息、Command A 采样器设置、由 Cohere command-r-plus 驱动的 AI 助手` 


- **Command-R-Plus 驱动新型 AI 助手**：一位初创公司创始人正在使用由 **Cohere command-r-plus** 驱动的 AI 助手构建结构生物学工具，并结合了 **MolStar** 分子查看器 ([https://ai.doi.bio](https://ai.doi.bio))。
   - 该网站目前支持 'load' 命令，用于将 PDB 条目加载到查看器中；例如，输入 *'Show me 7zzz'*。
- **讨论数据保留政策与安全信息**：一名成员询问了 **Cohere 聊天** 功能的 **数据保留** 和 **安全政策**，特别是数据是否会被用于模型训练。
   - Cohere 团队成员回复了 [隐私政策](https://cohere.com/privacy)、[数据使用政策](https://cohere.com/data-usage-policy) 和 [安全政策](https://cohere.com/security) 的链接，并提到用户可以在其控制面板中控制数据设置。
- **Cohere 的数据隐私与部署**：Cohere 团队成员详细说明了其 SaaS 平台允许用户直接从 [控制面板](https://dashboard.cohere.com/data-controls) 控制数据，可根据邮件请求提供 **ZDR 支持**，并集成了主要的云服务商（**OCI**、**Bedrock**、**Sagemaker**、**Azure Cloud**）。
   - 他们还提供 **本地部署（on-prem）解决方案**（详情见 [https://cohere.com/deployment-options](https://cohere.com/deployment-options)），符合 **SOC II** 和 **GDPR** 标准，并遵守数据安全和隐私的行业标准。
- **寻求 RAG 复现资源**：一名成员正在寻求资源以复现类似于 **notebooklm** 的 **RAG 源返回** 行为，即在搜索结果中引用特定段落。
   - 他们正在寻找与 **chunking** 和 **数据模型设计** 相关的开源示例。
- **Command A 采样器设置指南**：一名成员询问了已发布的 **Command A 推荐采样器设置**。
   - 另一名成员建议从 **temperature 0.7** 开始，并根据对确定性与灵活性的需求进行调整；默认 temperature 为 **0.3**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://ai.doi.bio">ai.doi.bio</a>: 未找到描述</li><li><a href="https://cohere.com/security">Security | Cohere</a>: 通过 Cohere 的企业级安全协议、严格的访问控制和私有部署选项，确保极致的 AI 安全与隐私。 </li><li><a href="https://dashboard.cohere.com/data-controls">Login | Cohere</a>: 登录以通过一个易于使用的 API 访问先进的 Large Language Models 和 NLP 工具。</li><li><a href="https://cohere.com/privacy">Privacy Policy | Cohere</a>: Cohere Inc. (“Cohere”) 重视并尊重您的隐私。我们准备了此隐私政策，以解释我们通过位于...的网站收集、使用和披露个人信息的方式。</li><li><a href="https://cohere.com/deployment-options">Deployment Options - SaaS, Cloud API, Virtual Private Cloud (VPC), On Premise | Cohere</a>: 我们的解决方案提供行业领先的数据隐私和安全性，旨在满足寻求利用生成式 AI 力量的组织的各种需求。无论您是初创公司还是...
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[「🔌」api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1353203487290429522)** (35 条消息🔥): 

> `Command 模型、SSL 错误、API 速率限制、MongoDB` 


- **Command 模型面临 SSL 问题？**：一名成员询问了 **Command** 模型及其生成更具人性化回复的潜力，同时也遇到了 **SSL 错误**。
   - 另一名成员指出，SSL 错误通常与模型本身无关，而是与 **不受信任的证书** 或网络配置有关，但也可能与速率限制有关。
- **API 滥用导致 SSL 错误？**：一名成员报告在快速向 **API** 发送请求时遇到 **SSL 错误**，怀疑尽管正确安装了 py.ssl 模块，但仍可能是由于请求过于频繁（spamming）导致的。
   - 另一名成员建议问题可能源于 **不受信任的服务器证书**，而非客户端问题，并建议联系支持团队。
- **怀疑出现 API 速率限制**：一名成员怀疑 **SSL 错误** 可能与频繁请求触发的未公开 **API 速率限制** 有关。
   - 然而，另一名成员指出，速率限制通常会返回 **429 错误代码**。
- **MongoDB 状态查询**：切换话题，一名成员询问另一名成员的 **MongoDB** 是否正常工作。
   - 另一名成员表示工作正常，昨天刚使用过。


  

---

### **Cohere ▷ #[「💡」projects](https://discord.com/channels/954421988141711382/1218409701339828245/1353370236643971123)** (2 messages): 

> `Discord Bot, RAG Pipeline, vnc-lm, Context Augmentation, Docker` 


- **vnc-lm 发布集成 RAG 的 Discord Bot**：一位成员发布了其 Discord Bot **vnc-lm** 的新版本，该版本具有 **RAG pipeline**，可从 **Wikipedia** 和 **DuckDuckGo** 提取数据，通过额外的上下文增强 Prompt。
   - 该 Pipeline 通过附加五个来源信息块，为每个 Prompt 增加了约 **500 tokens**，以改善模型的上下文，代码可在 [GitHub](https://github.com/jake83741/vnc-lm) 获取。
- **搜索功能的启用与禁用**：新发布的 Bot 支持网页搜索。
   - 新的搜索功能可以通过 **+ search** 启用，通过 **+ model** 禁用。
- **多功能 Bot 支持本地和托管 LLM**：更新后的 Discord Bot 现在支持所有流行的本地和托管 LLM API，包括 **Cohere**。
   - 该 Bot 可以使用 **Docker** 快速构建，允许用户在 Discord 内轻松编辑消息并获取新的响应。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://open.spotify.com/episode/6a44wSFv8bc1T9x3mEE9Dq?si=tWnXTxqHQbqpky6bWqj0uw&nd=1&dlsi=d20a7ee755104caa">Sancocho con Limon - Quatsch Session 01</a>: FELD.FM · Episode</li><li><a href="https://github.com/jake83741/vnc-lm">GitHub - jake83741/vnc-lm: A Discord bot for large language models. Add Gemini, Sonnet-3.7 DeepSeek R-1, and other models. Easily change models, edit prompts, and enable web search.</a>: 一个用于大语言模型的 Discord Bot。添加 Gemini, Sonnet-3.7 DeepSeek R-1 等模型。轻松切换模型、编辑 Prompt 并启用网页搜索。 - jake83741/vnc-lm
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1352733484468146206)** (33 messages🔥): 

> `Synthetic Data Generation with vllm and deepseek r1, Llama4 Release, Qwen3 MoE, Good Data Problem, PDF Extraction` 


- **来自 vllm 和 Deepseek R1 的 Synthetic Data 流**：一位成员正在使用 **vllm** 和 **Deepseek R1** 生成 **synthetic data**，预计该过程将持续数周。
   - 由于预期 **Llama4** 将在 LlamaCon 期间发布，训练工作已推迟。
- **数据质量难题仍在继续**：尽管经过多年的研究，即使在 **fineweb** 和 **lima** 等数据集的重要性得到认可之后，对于 AI 实验室来说，“优质数据”的定义和获取仍然难以捉摸。
   - 一位成员对缺乏有效的 **PDF extraction** 工具表示沮丧：*我们仍然没有出色的 PDF 提取工具，这让我非常恼火*。
- **LlamaExtract 工具发布**：[LlamaIndex](https://www.llamaindex.ai/) 发布了 **LlamaExtract**，这是一个使用 genAI-native **agents** 来结构化复杂文档的工具。
   - 它适配了最新的模型，以准确可靠地结构化财务报告和简历等文档。
- **DeepSeek-V3 的“奔放”发布**：一位成员注意到 Deepseek 随意发布了 **DeepSeek-V3**，并幽默地称他们“奔放（unhinged）”，因为缺乏正式的 readme。
   - 该模型可在 [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) 上访问，虽然 `README.md` 是空白的，但提供了 playground 的访问权限。
- **Torchtune 暗示将支持 MoEs？**：有人微妙地提到了 **Torchtune** 可能会加入 **Mixture of Experts (MoE)** 模型。
   - 讨论涉及训练此类大型模型的实际挑战，可能需要 **8-9 TB 的 VRAM**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/jerryjliu0/status/1902880391578653176">Jerry Liu (@jerryjliu0) 的推文</a>: LlamaExtract 现已进入公开测试阶段 🔥 - 领先的、genAI 原生 agent，用于结构化文档提取。我们适配并微调了最新的模型，以便您可以结构化最复杂的文档...</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3-0324">deepseek-ai/DeepSeek-V3-0324 · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1352735360001507479)** (23 messages🔥): 

> `datasets library issue, GRPO LoRA 3B Single Device, vLLM support for data generation, CUDA graphs` 


- **Datasets Library 故障排查**：成员们发现了 **datasets library** 的一个问题并尝试进行调试，其中一人建议升级 **datasets version**。
   - 一位成员确认他们使用的是最新版本 **3.4.1**。
- **GRPO LoRA 在 GMS8K 上达到 54%**：根据一位分享了 [Pull Request 链接](https://github.com/pytorch/torchtune/pull/2467) 的成员，**GRPO LoRA 3B 单设备** 在 GMS8K 上达到了 **54%**。
   - 该成员指出，尽管在计算中出现了添加多余 +2 的错误，但它在处理新颖问题上的表现优于预期。
- **数据生成缺乏 vLLM 支持**：成员们讨论了为数据生成添加 **vLLM support**，但指出在 vLLM 和 torchtune 之间共享权重存在困难。
   - 有人建议在另一个 vLLM 进程中托管模型并转换权重，而另一人提到正在尝试一种“黑客”式的方法使其在较小的模型上运行。
- **CUDA Graphs 捕获操作**：一位成员询问了 **CUDA graphs**，它将一堆 GPU 操作捕获为一个图，并将其作为单个操作启动。
   - 另一位成员确认了这一点，并指出这减少了从 CPU 启动 CUDA 操作的开销，从而减少了 GPU 空闲时间。



**Link mentioned**: <a href="https://github.com/pytorch/torchtune/pull/2467">GRPO LoRA Single Device by ianbarber · Pull Request #2467 · pytorch/torchtune</a>: ContextWhat is the purpose of this PR? Is it to[x ] add a new feature fix a bug update tests and/or documentation other (please add here)#2421 - exploring a LoRA recipe.ChangelogWhat are ...

  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1353412469934264491)** (1 messages): 

> `DLCoT Optimizer, Chain-of-Thought Distillation, Token Usage Reduction, DSPy Optimizers` 


- **为 Chain-of-Thought 推出的 DLCoT 优化器**：一位成员为 DSPy 的 teleprompt 模块提交了一个名为 **DLCoT** (Deconstructing Long Chain-of-Thought) 的新优化器的 [Pull Request (#8000)](https://github.com/stanfordnlp/dspy/pull/8000)。
   - 它通过智能处理和优化长 CoT 数据（分割 CoT 内容、移除冗余路径、过滤错误链并重构连贯输出）来增强 Chain-of-Thought 推理。
- **DLCoT 削减了 70-90% 的 Token 使用量**：**DLCoT 优化器** 可以在保持或提高基准测试准确性的同时，减少 **70-90%** 的 Token 使用量。
   - 该优化器可与现有的 DSPy 优化器（如 **BootstrapFewShot**）配合使用，并蒸馏出最高效的推理路径。



**Link mentioned**: <a href="https://github.com/stanfordnlp/dspy/pull/8000">Add DLCoT Optimizer for efficient Chain-of-Thought distillation by jmanhype · Pull Request #8000 · stanfordnlp/dspy</a>: Add DLCoT (Deconstructing Long Chain-of-Thought) OptimizerOverviewThis PR adds a new optimizer to the DSPy teleprompt module: the DLCoT (Deconstructing Long Chain-of-Thought) optimizer. This feat...

  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1353165176161042493)** (20 条消息🔥): 

> `用于创意内容生成的 DSPy，PAPILLON 示例，Agentic-Reward-Modeling 链接，DLCoT Optimizer，MIPROv2` 


- **讨论用于创意内容生成的 DSPy**：成员们正在讨论使用 **DSPy** 来优化创意内容生成的 Prompt，并建议使用一个*优秀的 judge*。
   - 一位成员建议查看 [PAPILLON](https://github.com/Columbia-NLP-Lab/PAPILLON/blob/main/papillon_tutorial.ipynb) 和 [Agentic Reward Modeling](https://github.com/THU-KEG/Agentic-Reward-Modeling) 示例。
- **DLCoT Optimizer 贡献**：一位成员在 [GitHub](https://github.com/stanfordnlp/dspy/pull/8000) 上分享了一个新的贡献——**DLCoT (Deconstructing Long Chain-of-Thought) Optimizer**，用于高效的 Chain-of-Thought 蒸馏。
   - 该成员鼓励其他人查看并提供反馈。
- **在没有示例的情况下优化 Prompt**：一位成员正在寻求关于在**没有示例**的情况下优化文章摘要 Prompt 的指导，目前已有一个可用的评估函数，并询问是否应该使用 **COPRO** 而不是 **MIPROv2**。
   - 另一位成员澄清说，示例*输入*始终是需要的，但如果 judge/metric 可以在没有参考/标签的情况下评估摘要，那么摘要（标签）则不是必须的。
- **通过 `dspy.Prediction` 获取细粒度反馈**：一位成员询问如何通过 **Refine** 实现细粒度反馈，类似于 assertions/suggestions，即通过对输出进行特定检查来提供针对性反馈。
   - 另一位成员提到，在 **2.6.15** 版本中，将可以返回 `dspy.Prediction(score=...., feedback=....)` 来为模块提供细粒度反馈。
- **检索中的多 Agent 协议标准 (MCP)**：成员们讨论了多 Agent 协议标准 (**MCP**) 的潜力及其扩展到包含检索器/检索增强生成（RAG）的可能性。
   - 讨论内容包括检索结果的共享 schema 以及交换文档和 embedding 的方法，旨在简化数据驱动的工作流，并简化多个模型和数据源的组合。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/THU-KEG/Agentic-Reward-Modeling">GitHub - THU-KEG/Agentic-Reward-Modeling: Agentic Reward Modeling: Integrating Human Preferences with Verifiable Correctness Signals for Reliable Reward Systems</a>：Agentic Reward Modeling：将人类偏好与可验证的正确性信号相结合，构建可靠的奖励系统 - THU-KEG/Agentic-Reward-Modeling</li><li><a href="https://github.com/Columbia-NLP-Lab/PAPILLON/blob/main/papillon_tutorial.ipynb">PAPILLON/papillon_tutorial.ipynb at main · Columbia-NLP-Lab/PAPILLON</a>：我们论文 PAPILLON 的代码：基于互联网和本地语言模型集成的隐私保护 - Columbia-NLP-Lab/PAPILLON</li><li><a href="https://github.com/stanfordnlp/dspy/pull/8000">Add DLCoT Optimizer for efficient Chain-of-Thought distillation by jmanhype · Pull Request #8000 · stanfordnlp/dspy</a>：添加 DLCoT (Deconstructing Long Chain-of-Thought) Optimizer 概述。此 PR 为 DSPy teleprompt 模块添加了一个新的优化器：DLCoT 优化器。该功能...
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1353064669274701935)** (9 条消息🔥): 

> `DSPy Modules, 创意写作 Prompt, PAPILLON, 隐私保护` 


- **DSPy Module 使用方式受到关注**：一位成员询问了在利用 **LLM** 从 **Pandas DataFrame** 生成报告和图表的场景下，**DSPy Modules** 的正确用法。
   - 另一位成员指出，如果除了一个庞大的附加代码文件外没有更具体的问题，很难提供帮助；随后该成员明确询问：*这是否是使用 DSPy Modules 的正确方式*？
- **成员寻求创意写作 Prompt 示例**：一位成员请求改进**创意写作 Prompt** 的示例，或类似没有明确正确答案的案例。
   - 共享了一个指向 **PAPILLON GitHub 仓库** 的链接，其中包含一个专注于基于互联网和本地语言模型集成的隐私保护教程笔记本，[PAPILLON GitHub](https://github.com/Columbia-NLP-Lab/PAPILLON/blob/main/papillon_tutorial.ipynb)。



**提到的链接**：<a href="https://github.com/Columbia-NLP-Lab/PAPILLON/blob/main/papillon_tutorial.ipynb">PAPILLON/papillon_tutorial.ipynb at main · Columbia-NLP-Lab/PAPILLON</a>：我们论文 PAPILLON 的代码：基于互联网和本地语言模型集成的隐私保护 - Columbia-NLP-Lab/PAPILLON

  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1353437505646755893)** (19 条消息🔥): 

> `sops.gz dataset, Tinygrad CUDA port, Meeting #63 Agenda, AMD LLVM progress, ONNX Frontend for Tinygrad` 


- **追踪 `sops.gz` 来源**：一位成员询问了在 `speed_compare_cuda_ptx` 中使用的 `datasets/sops.gz` 数据集的位置。
   - 另一位成员分享了该数据集可在仓库的 [extra 目录](https://github.com/tinygrad/tinygrad/blob/master/extra/datasets/sops.gz) 中找到，并通过 [generate_dataset.sh 脚本](https://github.com/tinygrad/tinygrad/blob/master/extra/optimization/generate_dataset.sh) 生成。
- **CUDA 移植思考**：一位成员询问了将 **Tinygrad** 移植到 **CUDA GPU** 进行训练的可能性。
   - 另一位成员回复了 [README.md](https://github.com/tinygrad/tinygrad/?tab=readme-ov-file#accelerators) 文件的链接，重点介绍了支持的 backends。
- **会议议程公布**：公布了第 63 次会议的议程，涵盖了 **company update**、**quantized DSP**、**BERT**、**scheduler**、**driver**、**tensor cores**、**WebGPU**、**ONNX**、**RetinaNet**、**Torch frontend** 以及其他 bounties。
   - 讨论内容包括 **test_ops**、**multi GPU training**、**torch compile** 以及针对 **AMD LLVM backend** 的 bounties。
- **AMD LLVM Backend 进展**：报告了 **AMD LLVM backend** 的进展，包括多个已合并的 pull requests，以及使用 **Llama3** 和 **Flux** 示例进行的测试。
   - 一个 pull request 正在审查中。
- **ONNX Frontend 进展**：一位成员注意到 `tinygrad.frontend.onnx` 已经存在，并表示本周将重点关注 **ONNX** 的准备工作。
   - 验证前 30 个 **Hugging Face ONNX** 仓库是一个讨论话题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/blob/master/extra/datasets/sops.gz">tinygrad/extra/datasets/sops.gz at master · tinygrad/tinygrad</a>: 你喜欢 pytorch？你喜欢 micrograd？你爱 tinygrad！❤️ - tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/?tab=readme-ov-file#accelerators">GitHub - tinygrad/tinygrad: You like pytorch? You like micrograd? You love tinygrad! ❤️</a>: 你喜欢 pytorch？你喜欢 micrograd？你爱 tinygrad！❤️ - GitHub - tinygrad/tinygrad: You like pytorch? You like micrograd? You love tinygrad! ❤️
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1353147176968126574)** (4 条消息): 

> `Disable colored terminal output, tinygrad facades, GPU code generation, OpenCLEmpty guarantees` 


- **在 tinygrad 中禁用彩色终端输出**：一位成员询问是否有办法禁用彩色终端输出。
- **Tinygrad 有两个面相 (facades)**：Tinygrad 有两个面相：**deep learning** 部分（权重更新、tensors、矩阵乘法）和 **compiler** 部分（GPU 代码生成和 scheduling）。
   - 深度学习部分在 [Karpathy 的 Youtube 教程](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) 中有更好的解释。
- **OpenCL empty 值是不确定的**：一位成员报告称从 [tinygrad-notes 的第一个示例](https://mesozoic-egg.github.io/tinygrad-notes/20241231_intro.html) 中得到了奇怪的输出。
   - 澄清指出，*使用 OpenCLempty 只是空值，没有保证的数值*。



**提到的链接**：<a href="https://mesozoic-egg.github.io/tinygrad-notes/20241231_intro.html">Introduction to the internals</a>：tinygrad 教程

  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1353204258391982166)** (9 条消息🔥): 

> `Quiz 拼写错误, AgentX Research Track, 远程研究指导, 无薪研究` 


- **Quiz 标题拼写错误导致困惑**：一名成员报告了 **Quiz 7** 标题中的拼写错误，导致在核对 **Quiz 6** 答案时产生困惑。
   - 另一名成员确认了这一发现并向报告者表示感谢。
- **AgentX Research Track 申请上线**：入选学生将接受来自 **Berkeley 博士后/导师** 关于 **AgentX Research Track 项目** 的指导，申请截止日期为 **PDT 时间 3 月 26 日晚上 11:59**。
   - 加入或在 **AgentX** 中取得成功并不强制要求导师指导，实验课（Labs）和证书申报表（Certificate Declaration form）将于 4 月发布，详见[附图](https://cdn.discordapp.com/attachments/1280370030609170494/1353204258450964544/image.png?ex=67e2c76c&is=67e175ec&hm=1fb895b885ce732fd7e5b99b8ff24c55286d5)。
- **确认 Research Track 为远程且无薪**：一名成员确认 **AgentX Research Track 指导** 将以远程方式进行。
   - 另一名成员澄清说，该指导是无薪的，导师仅提供研究项目的指导。


  

---


---


---


---


{% else %}


> 完整的各频道详细分析已针对邮件进行缩减。 
> 
> 如果您想查看完整的详细分析，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}