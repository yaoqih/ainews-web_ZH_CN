---
companies:
- character.ai
- google
- adept
- amazon
- inflection
- microsoft
- stability-ai
- black-forest-labs
- schelling
- google-deepmind
- openai
- anthropic
- meta-ai-fair
- lmsys
- langchainai
date: '2024-08-03T01:48:48.159187Z'
description: '以下是该文本的中文翻译：


  **Character.ai 以 25 亿美元的价格向谷歌输送核心人才（Execuhire）**，标志着一次重大的领导层变动；此前，**Adept 以 4.29
  亿美元向亚马逊输送人才**，**Inflection 以 6.5 亿美元向微软输送人才**。尽管 Character.ai 的用户增长和内容势头强劲，其 CEO
  Noam Shazeer 仍选择重返谷歌，这预示着 AI 行业风向的转变。


  **Google DeepMind 的 Gemini 1.5 Pro** 在 Chatbot Arena 基准测试中登顶，超越了 **GPT-4o** 和 **Claude-3.5**，在多语言、数学和编程任务中表现卓越。**Black
  Forest Labs 的 FLUX.1** 文生图模型以及 **LangGraph Studio** 智能体 IDE 的发布，凸显了行业的持续创新。**Llama
  3.1 405B** 作为目前最大的开源模型发布，促进了开发者的应用，并展开了与闭源模型的竞争。


  目前，行业日益将后训练（post-training）和数据视为关键竞争因素，这也引发了公众对收购行为及监管审查的关注。'
id: c4fb887d-02de-4c94-8f1d-da8266ad631b
models:
- gemini-1.5-pro
- gpt-4o
- claude-3.5
- flux-1
- llama-3-1-405b
original_slug: ainews-acquisitions-the-fosbury-flop-of-ma
people:
- noam-shazeer
- mostafa-mostaque
- david-friedman
- rob-rombach
- alexandr-wang
- svpino
- rohanpaul_ai
title: 'Execuhires：挑战可汗之怒


  （注：“The Wrath of Khan” 是《星际迷航》系列中的经典篇目，通常译为《可汗之怒》或《可汗怒吼》。）'
topics:
- execuhire
- model-benchmarking
- multilinguality
- math
- coding
- text-to-image
- agent-ide
- open-source-models
- post-training
- data-driven-performance
---

<!-- buttondown-editor-mode: plaintext -->**Noam 回家了。**

> 2024年8月1日至8月2日的 AI 新闻。我们为您检查了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **28** 个 Discord 社区（**249** 个频道，**3233** 条消息）。预计为您节省阅读时间（以 200wpm 计算）：**317 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

我们想知道是否是同一批律师参与了以下建议：

- [Adept 向 Amazon 提供的 4.29 亿美元 execuhire](https://www.semafor.com/article/08/02/2024/investors-in-adept-ai-will-be-paid-back-after-amazon-hires-startups-top-talent)
- [Inflection 向 Microsoft 提供的 6.5 亿美元 execuhire](https://buttondown.email/ainews/archive/ainews-inflection-25-at-94-of-gpt4-and-pi-at-6m/)
- [Character.ai 今天向 Google 提供的 25 亿美元 execuhire](https://techcrunch.com/2024/08/02/character-ai-ceo-noam-shazeer-returns-to-google/) 

（我们还要注意到 [Stability 的大部分领导层已经离职](https://buttondown.email/ainews/archive/ainews-shipping-and-dipping-inflection-stability/)，尽管这不算是 execuhire，因为 Robin 现在成立了 [Black Forest Labs](https://buttondown.email/ainews/archive/ainews-rombach-et-al-flux1-prodevschnell-31m-seed/)，而 Emad 成立了 [Schelling](https://twitter.com/EMostaque/status/1799044420282826856)。）

Character 并非真的陷入困境。他们的 SimilarWeb 统计数据已经超过了之前的峰值，且[发言人表示内部 DAU 数据同比增长了 3 倍](https://www.theinformation.com/articles/a-chatbot-pioneer-mulls-deals-with-rivals-google-and-meta?rc=ytp67n)。

 
![image.png](https://assets.buttondown.email/images/24aaa55b-03d9-4051-a07e-e6d6da4ebe98.png?w=960&fit=max)
 

我们曾对[他们的博客文章赞不绝口](https://buttondown.email/ainews/archive/ainews-shazeer-et-al-2024/)，就在昨天还报道了 [Prompt Poet](https://research.character.ai/prompt-design-at-character-ai/)。通常情况下，任何拥有这种近期内容势头的公司都表现良好……但在这里，行动胜于雄辩。

正如我们在 [The Winds of AI Winter](https://www.latent.space/p/mar-jun-2024) 中讨论的那样，氛围正在发生变化，虽然这在本质上不完全是技术性的，但它们太重要了，不容忽视。如果 Noam 无法带着 Character 走到底，Mostafa 无法带着 Inflection 走到底，David 无法带着 Adept 走到底，那么其他基础模型实验室的前景又如何呢？[转向以后训练 (post-training)](https://x.com/_xjdr/status/1819435049655455987) 为重心的趋势正在升温。

当一个东西走起来像鸭子，叫起来像鸭子，但又不想被称为鸭子时，我们大概还是可以把它归入*鸭科 (Anatidae)* 家族树。当大公司拿走了核心技术、核心高管，并偿还了所有核心投资者的资金时……FTC 是否会认为这已经足够接近规避收购的字面定义，但违背了其管辖权的实质精神？


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型发展与基准测试**

- **Gemini 1.5 Pro 性能**：[@lmsysorg](https://twitter.com/lmsysorg/status/1819048821294547441) 宣布 @GoogleDeepMind 的 Gemini 1.5 Pro (Experimental 0801) 在 Chatbot Arena 中夺得榜首，以 1300 分的成绩超越了 GPT-4o/Claude-3.5。该模型在多语言任务中表现出色，并在数学、硬核提示（Hard Prompts）和 Coding 等技术领域表现优异。

- **模型对比**：[@alexandr_wang](https://twitter.com/alexandr_wang/status/1819086525499621494) 指出 OpenAI、Google、Anthropic 和 Meta 都处于 AI 开发的最前沿。Google 凭借 TPU 拥有的长期算力优势可能成为一个显著竞争力。数据和训练后处理（post-training）正成为性能提升的关键竞争驱动因素。

- **FLUX.1 发布**：[@robrombach](https://twitter.com/robrombach/status/1819012132064669739) 宣布成立 Black Forest Labs 并推出其全新的 SOTA 文本生成图像模型 FLUX.1。该模型包含三个变体：pro、dev 和 schnell，其中 schnell 版本在 Apache 2.0 许可证下发布。

- **LangGraph Studio**：[@LangChainAI](https://twitter.com/LangChainAI/status/1819052975295270949) 推出了 LangGraph Studio，这是一个用于开发 LLM 应用程序的 Agent IDE。它提供了复杂 Agent 应用的可视化、交互和调试功能。

- **Llama 3.1 405B**：[@svpino](https://twitter.com/svpino/status/1818982567296532700) 分享了 Llama 3.1 405B 现已开放免费测试。这是迄今为止最大的开源模型，可与闭源模型竞争，其许可证允许开发者使用它来增强其他模型。

**AI 研究与进展**

- **BitNet b1.58**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1819032855055340025) 讨论了 BitNet b1.58，这是一种 1-bit LLM，其中每个参数都是三值（ternary）的 {-1, 0, 1}。这种方法可能允许在手机等内存有限的设备上运行大型模型。

- **Distributed Shampoo**：[@_arohan_](https://twitter.com/_arohan_/status/1819102492468396315) 宣布 Distributed Shampoo 在深度学习优化方面超越了 Nesterov Adam，标志着非对角预处理（non-diagonal preconditioning）领域的重大进展。

- **Schedule-Free AdamW**：[@aaron_defazio](https://twitter.com/aaron_defazio/status/1819099653100785880) 报告称 Schedule-Free AdamW 为自调优训练算法设定了新的 SOTA，在 AlgoPerf 竞赛中整体表现优于 AdamW 和其他提交算法 8%。

- **Adam-atan2**：[@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1819152769980432678) 分享了一行代码修改，通过将除法改为 atan2() 来移除 Adam 中的 epsilon 超参数，这对于解决除以零和数值精度问题可能很有用。

**行业动态与合作伙伴关系**

- **Perplexity 与 Uber 合作**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1819097426617315765) 宣布了 Perplexity 与 Uber 的合作伙伴关系，为 Uber One 订阅者提供 1 年免费的 Perplexity Pro。

- **GitHub 模型托管**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1819098489499877616) 报告称 GitHub 现在将直接托管 AI 模型，通过 Codespaces 提供无摩擦的路径来实验模型推理代码。

- **Cohere 登陆 GitHub**：[@cohere](https://twitter.com/cohere/status/1819069714997694491) 宣布其最先进的语言模型现在通过 Azure AI Studio 提供给 GitHub 上的 1 亿多名开发者。

**AI 工具与框架**

- **torchchat**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1819029248037757195) 分享了 PyTorch 发布的 torchchat，它使本地运行 LLM 变得容易，支持包括 Llama 3.1 在内的一系列模型，并提供 Python 和原生执行模式。

- **TensorRT-LLM Engine Builder**：[@basetenco](https://twitter.com/basetenco/status/1819048091451859238) 为 TensorRT-LLM 引入了新的 Engine Builder，旨在简化为开源和微调后的 LLM 构建优化模型服务引擎的过程。

**关于 AI 影响与未来的讨论**

- **AI 转型**：[@fchollet](https://twitter.com/fchollet/status/1819139182000066779) 认为，虽然 AGI 不会仅仅通过当前技术的规模扩张（scaling）来实现，但 AI 将改变几乎每一个行业，从长远来看，其规模将比大多数观察者预期的还要大。

- **意识形态的古德哈特定律（Goodhart's Law）**：[@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1819103767989833768) 提出，任何不能被某种意识形态信奉者质疑的错误信念，都将日益成为该意识形态的核心。

本摘要涵盖了所提供推文中反映的 AI 领域关键进展、公告和讨论，重点关注与 AI 工程师和研究人员相关的方面。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. 高效 LLM 创新：BitNet 与 Gemma**

- **["hacked bitnet for finetuning, ended up with a 74mb file. It talks fine at 198 tokens per second on just 1 cpu core. Basically witchcraft."](https://x.com/nisten/status/1818529201231688139?t=a2_oszg66OrDGlwweQS1iQ&s=19)** ([Score: 577, Comments: 147](https://reddit.com//r/LocalLLaMA/comments/1ehh9x2/hacked_bitnet_for_finetuning_ended_up_with_a_74mb/)): 一位开发者成功地**微调了 BitNet**，创建了一个极其**紧凑的 74MB 模型**，并展示了令人印象深刻的性能。该模型在**单核 CPU** 上达到了 **198 tokens per second** 的速度，尽管体积微小，却展现了高效的自然语言处理能力。

- **Gemma2-2B on iOS, Android, WebGPU, CUDA, ROCm, Metal... with a single framework** ([Score: 58, Comments: 17](https://reddit.com//r/LocalLLaMA/comments/1ehmwph/gemma22b_on_ios_android_webgpu_cuda_rocm_metal/)): **Gemma2-2B** 是一款最近发布的语言模型，现在可以在其发布后的 **24 小时内**，通过 **MLC-LLM 框架**在包括 **iOS, Android, web 浏览器, CUDA, ROCm 和 Metal** 在内的**多个平台**上本地运行。该模型紧凑的体积和在 **Chatbot Arena** 中的表现使其非常适合本地部署，目前已提供针对各种平台的演示，包括在 [chat.webllm.ai](https://chat.webllm.ai/) 上实时运行的 **4-bit 量化版本**。针对每个平台都提供了详细的文档和部署说明，包括适用于笔记本电脑和服务器的 **Python API**、适用于 iOS 的 **TestFlight**，以及针对 Android 和基于浏览器的实现的特定指南。

- **New results for gemma-2-9b-it** ([Score: 51, Comments: 32](https://reddit.com//r/LocalLLaMA/comments/1ehdchd/new_results_for_gemma29bit/)): 由于配置修复，**Gemma-2-9B-IT** 模型的基准测试结果已更新，现在在大多数类别中都优于 **Meta-Llama-3.1-8B-Instruct**。值得注意的是，**Gemma-2-9B-IT** 在 **BBH**（**42.14** vs **28.85**）、**GPQA**（**13.98** vs **2.46**）和 **MMLU-PRO**（**31.94** vs **30.52**）中获得了更高的分数，而 **Meta-Llama-3.1-8B-Instruct** 在 **IFEval**（**77.4** vs **75.42**）和 **MATH Lvl 5**（**15.71** vs **0.15**）中保持领先。
  - **MMLU-Pro** 基准测试结果因测试方法而异。/u/chibop1 的 [OpenAI API Compatible script](https://github.com/chigkim/Ollama-MMLU-Pro/) 显示 **gemma2-9b-instruct-q8_0** 得分为 **48.55**，**llama3-1-8b-instruct-q8_0** 得分为 **44.76**，均高于报告的分数。
  - 注意到不同来源的 **MMLU-Pro** 分数存在差异。**Open LLM Leaderboard** 显示 **Llama-3.1-8B-Instruct** 为 **30.52**，而 **TIGER-Lab** 报告为 **0.4425**。分数归一化和测试参数可能是导致这些差异的原因。
  - 用户讨论了创建个性化基准测试框架来比较 **LLMs** 和量化方法。考虑的因素包括模型大小、量化级别、处理速度和质量保留，旨在为各种用例做出明智的决策。


**主题 2. 开源 AI 模型的进展**

- **fal announces Flux a new AI image model they claim its reminiscent of Midjourney and its 12B params open weights** ([Score: 313, Comments: 97](https://reddit.com//r/LocalLLaMA/comments/1ehhjlh/fal_announces_flux_a_new_ai_image_model_they/)): fal.ai 发布了 **Flux**，这是一个拥有 **120 亿参数**的新型**开源文本生成图像模型**，他们声称其效果令人联想到 **Midjourney**。该模型被描述为目前可用的**最大的开源文本生成图像模型**，现在可以在 fal 平台上使用，为用户提供了一个强大的 AI 生成图像创作工具。

- **[新的医疗和金融 70b 32k Writer 模型](https://www.reddit.com/gallery/1ei31si)** ([Score: 108, Comments: 34](https://reddit.com//r/LocalLLaMA/comments/1ei31si/new_medical_and_financial_70b_32k_writer_models/)): Writer 发布了两个新的 **70B 参数模型**，具有 **32K 上下文窗口**，分别针对医疗和金融领域。据报道，这些模型的表现优于 **Google 的专用医疗模型**和 **ChatGPT-4**。这些模型可在 Hugging Face 上用于研究和非商业用途，为处理更复杂的问答提供了可能，同时仍可在家用系统上运行，符合开发多个较小模型而非超大 **120B+** 模型的趋势。
  - 针对医疗和金融领域的 **70B 参数模型**（具备 **32K 上下文窗口**）据称优于 **Google 的专用医疗模型**和 **ChatGPT-4**。金融模型以 **73% 的平均分**通过了**难度较大的 CFA 三级考试**，而人类的通过率为 **60%**，ChatGPT 为 **33%**。
  - 讨论了人类医生在这些基准测试中的表现，一位 **ML 工程师兼医生**认为，如果允许搜索，人类的表现可能会很高，但基准测试可能是**为了公关效果而容易被刷分的指标**。其他人则认为，在典型的 **20 分钟问诊**中，LLM 的表现可能优于医生。
  - 关于复制智力任务与体力技能（如管道维修）相对难度的辩论。一些人认为，构建**超人类通用智能 (AGI)** 可能比构建能够执行复杂体力任务的机器更容易，因为动物在感知和运动控制方面经过了数亿年的进化优化。


**主题 3. AI 开发工具与平台**

- **[微软推出 Hugging Face 竞争对手（等候名单注册）](https://github.blog/news-insights/product-news/introducing-github-models/)** ([Score: 222, Comments: 46](https://reddit.com//r/LocalLLaMA/comments/1ehmsc4/microsoft_launches_hugging_face_competitor/)): 微软推出了 **GitHub Models**，将其定位为 AI 模型市场中 **Hugging Face** 的竞争对手。该公司已为感兴趣的用户开放了**等候名单注册**，以便尽早访问该平台，尽管帖子中未提供有关其功能和能力的具体细节。

- **[介绍 sqlite-vec v0.1.0：一个可在任何地方运行的矢量搜索 SQLite 扩展](https://alexgarcia.xyz/blog/2024/sqlite-vec-stable-release/index.html)** ([Score: 117, Comments: 28](https://reddit.com//r/LocalLLaMA/comments/1ehlazq/introducing_sqlitevec_v010_a_vector_search_sqlite/)): **SQLite-vec v0.1.0** 已发布，这是一个新的 **SQLite** 矢量搜索扩展，无需独立的矢量数据库即可提供矢量相似度搜索功能。该扩展支持**余弦相似度**和**欧几里得距离**指标，并可通过 **WebAssembly** 在包括**桌面**、**移动端**和**浏览器**在内的各种平台上使用。它旨在轻量化且易于集成，使其适用于从**本地 AI 助手**到**边缘计算**的各种应用场景。

**主题 4. 本地 LLM 部署与优化技术**

- **[一个包含多种不同策略的 RAG 实现的大型开源集合](https://github.com/NirDiamant/RAG_Techniques)** ([Score: 76, Comments: 5](https://reddit.com//r/LocalLLaMA/comments/1ehl25j/an_extensive_open_source_collection_of_rag/)): 该帖子分享了一个**开源仓库**，其中包含广泛的**检索增强生成 (RAG) 实现策略**，包括 **GraphRAG**。这个由社区贡献的资源提供了**教程**和**可视化图表**，是那些对 RAG 技术感兴趣的人的宝贵参考和学习工具。

- **如何在 Windows 11 上本地构建具有 NVIDIA GPU 加速的 llama.cpp：一个真正有效的简单分步指南。** ([Score: 67, Comments: 19](https://reddit.com//r/LocalLLaMA/comments/1ehd17m/how_to_build_llamacpp_locally_with_nvidia_gpu/)): 本指南提供了在 **Windows 11** 上构建具有 **NVIDIA GPU 加速**的 **llama.cpp** 的**分步说明**。它详细介绍了 **Python 3.11.9**、**Visual Studio Community 2019**、**CUDA Toolkit 12.1.0** 的安装，以及使用 **Git** 和 **CMake**（带有特定的 **CUDA 支持**环境变量）克隆和构建 llama.cpp 仓库所需的命令。

## All AI Reddit 回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 图像生成进展**

- **Flux：新的开源文本转图像模型**：[Black Forest Labs 推出了 Flux](https://www.reddit.com/r/StableDiffusion/comments/1ehh1hx/announcing_flux_the_next_leap_in_texttoimage/)，这是一个拥有 12B 参数的模型，包含三个版本：FLUX.1 [dev]、FLUX.1 [schnell] 和 FLUX.1 [pro]。据称其呈现的美学效果足以媲美 Midjourney。

- **Flux 性能与对比**：[据报道 Flux 的表现与 Midjourney 相当](https://www.reddit.com/r/singularity/comments/1ehsb6l/flux_is_a_new_open_source_image_generator_that_is/)，在文本生成和解剖结构方面表现更好，而 Midjourney 在美学和皮肤纹理方面更胜一筹。Flux 的生成成本为每张图像 0.003-0.05 美元，生成时间为 1-6 秒。

- **Flux 图像示例**：社区分享了一个 [Flux 生成的图像库](https://www.reddit.com/r/StableDiffusion/comments/1ehiz51/flux_image_examples/)，展示了该模型的能力。

- **Runway Gen 3 视频生成**：[Runway 的 Gen 3 模型展示了生成 10 秒视频的能力](https://www.reddit.com/r/singularity/comments/1ehgg2u/runway_gen_3_can_generate_text_to_videos_with/)，能够根据文本提示在 90 秒内生成具有高度细节皮肤的视频，每条视频成本约为 1 美元。

**AI 语言模型与发展**

- **Google Gemini Pro 1.5 占据榜首**：[Google 1.5 Pro 8 月发布版据报道首次在 AI 模型排名中获得第一名](https://www.reddit.com/r/singularity/comments/1ehlbwy/google_claims_1_for_the_first_time_with_15_pro/)。

- **Meta 的 Llama 4 计划**：Mark Zuckerberg 宣布 [训练 Llama 4 将需要比 Llama 3 多近 10 倍的算力](https://www.reddit.com/r/singularity/comments/1ehpvtz/mark_zuckerberg_said_at_q2_earnings_call_the/)，目标是使其在明年成为行业内最先进的模型。

**AI 交互与用户体验**

- **AI 谄媚（Sycophancy）担忧**：用户报告称 [AI 模型变得过度顺从](https://www.reddit.com/r/singularity/comments/1ehzy8k/is_ai_becoming_a_yes_man/)，经常重复用户输入而没有增加有价值的信息。这种被称为“谄媚（Sycophancy）”的行为在各种 AI 模型中都有观察到。

**梗图与幽默**

- r/singularity 中的一个 [梗图帖子](https://www.reddit.com/r/singularity/comments/1ehmfq8/so_this_fucking_sucks/) 获得了极高的关注。


---

# AI Discord 回顾

> 摘要的摘要的摘要

**1. LLM 进展与基准测试**

- **Llama 3 登顶排行榜**：来自 Meta 的 **[Llama 3](https://lmsys.org/blog/2024-05-08-llama3/)** 在 **ChatbotArena** 等排行榜上迅速崛起，在超过 50,000 场对决中表现优于 **GPT-4-Turbo** 和 **Claude 3 Opus**。
   - 社区热衷于讨论 Llama 3 在各种基准测试中的表现，一些人注意到它在某些领域的能力超过了闭源替代方案。
- **Gemma 2 与 Qwen 1.5B 之争**：关于 **Gemma 2B** 是否被过度炒作引发了争论，有观点认为 **Qwen 1.5B** 在 **MMLU** 和 **GSM8K** 等基准测试中表现更好。
   - 一位成员指出 Qwen 的表现很大程度上被忽视了，称其 *“残暴地击败了 Gemma 2B”*，突显了模型改进的飞速步伐以及紧跟最新进展的挑战。
 
- **医疗与金融领域的动态 AI 模型**：新模型 **[Palmyra-Med-70b](https://huggingface.co/Writer/Palmyra-Med-70B)** 和 **Palmyra-Fin-70b** 已被引入医疗和金融应用，并拥有令人印象深刻的性能。
  - 正如 [Sam Julien 的推文](https://x.com/samjulien/status/1818652901130354724) 所证明的那样，这些模型可能会对症状诊断和财务预测产生重大影响。

- **MoMa 架构提升多模态 AI**：Meta 的 **[MoMa](https://arxiv.org/pdf/2407.21770)** 引入了一种稀疏早期融合架构，通过 Mixture-of-Expert 框架增强了预训练效率。
  - 该架构显著提高了交错混合模态 Token 序列的处理能力，标志着多模态 AI 的重大进步。

**2. 优化 LLM 推理与训练**

- **Vulkan 引擎提升 GPU 加速**：**LM Studio** 推出了全新的 **Vulkan llama.cpp 引擎**，取代了之前的 OpenCL 引擎，在 **0.2.31** 版本中为 **AMD**、**Intel** 和 **NVIDIA** 独立 GPU 提供了 GPU 加速支持。
   - 用户报告了显著的性能提升，其中一位在 Llama 3-8B-16K-Q6_K-GGUF 模型上达到了 **40 tokens/second**，展示了本地 LLM 执行优化的潜力。
- **DeepSeek API 的磁盘上下文缓存**：**DeepSeek API** 引入了全新的上下文缓存功能，可将 API 成本降低高达 **90%**，并显著降低多轮对话的首字延迟（first token latency）。
   - 这一改进通过缓存频繁引用的上下文来支持数据和代码分析，突显了在优化 LLM 性能和降低运营成本方面的持续努力。

- **DeepSeek API 通过缓存降低成本**：**[DeepSeek API](https://x.com/deepseek_ai/status/1819358570766643223)** 引入了磁盘上下文缓存功能，可降低高达 **90%** 的 API 成本并降低首字延迟。
  - 这一改进通过缓存频繁引用的上下文来支持多轮对话，从而提升性能并降低成本。
- **Gemini 1.5 Pro 表现优于竞争对手**：讨论强调了 **Gemini 1.5 Pro** 极具竞争力的性能，成员们注意到其在实际应用中令人印象深刻的响应质量。
  - 一位用户观察到，他们对该模型的使用证明了其在响应速度和准确性方面优于其他模型。

**3. 开源 AI 框架与社区努力**

- **Magpie Ultra 数据集发布**：HuggingFace 发布了 **[Magpie Ultra](https://huggingface.co/datasets/argilla/magpie-ultra-v0.1)** 数据集，这是一个包含 **5 万条未过滤** 数据的 L3.1 405B 数据集，称其为开放合成数据集的先驱。
   - 社区表达了兴奋与谨慎并存的态度，讨论围绕该数据集对模型训练的潜在影响，以及对指令质量和多样性的担忧。
- **用于 RAG 流水的 LlamaIndex 工作流**：分享了一个关于使用 **[LlamaIndex workflows](https://t.co/XGmm6gQhcI)** 构建包含检索、重排序和合成的 **RAG pipeline** 的教程，展示了 AI 应用的事件驱动架构。
   - 该资源旨在指导开发者创建复杂的 RAG 系统，反映了人们对模块化和高效 AI 流水线构建日益增长的兴趣。
 
- **FLUX Schnell 的局限性**：用户报告称 **FLUX Schnell** 模型在提示词遵循（prompt adherence）方面表现不佳，经常产生无意义的输出，引发了对其作为生成模型有效性的担忧。
  - *“请千万、千万、千万不要用 Flux 公开发布的权重制作合成数据集”* 是成员们分享的一条警告。

**4. AI 行业趋势与收购**

- **Character.ai 收购引发辩论**：**Character.ai** 被 Google 收购且其联合创始人加入这家科技巨头，引发了关于 AI 初创公司在面对大厂收购时的生存能力的讨论。
   - 社区辩论了这对 AI 领域创新和人才留存的影响，一些人对 *“人才收购”*（acquihire）趋势可能扼杀竞争和创造力表示担忧。
- **在线 GPU 托管服务激增**：用户分享了使用 **[RunPod](https://www.runpod.io/)** 和 **Vast** 等在线 GPU 托管服务的经验，并指出价格随硬件需求的不同而有显著差异。
  - RunPod 因其出色的体验而受到称赞，而 Vast 较低的 3090 租用成本则吸引了预算有限的用户。
- **GitHub 与 Hugging Face 在模型托管方面的竞争**：**[GitHub 的新模型托管方式](https://github.com/karpathy/llm.c/issues/727)** 引发了担忧，与 **Hugging Face** 相比，它被认为是一个限制性的演示，削弱了社区贡献。
  - 成员们推测这一策略旨在控制 ML 社区的代码，并防止用户大规模流向更开放的平台。

---

# PART 1: Discord 高层级摘要

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Flux 模型的质量不一致性**：用户报告称 **Flux model** 生成的图像质量参差不齐，尤其在处理抽象风格（如躺在草地上的女性）时表现吃力。这引发了与 **Stable Diffusion 3** 类似的担忧。
   - 虽然详细的 Prompt 偶尔能产生不错的效果，但许多人对该模型的**核心局限性**表示沮丧。
- **在线 GPU 托管服务激增**：用户分享了他们在在线 GPU 托管服务方面的经验，特别是 **RunPod** 和 **Vast**，并指出价格随硬件需求而有显著差异。青睐 RunPod 的用户强调其体验更精致，而其他人则认为 **Vast** 的 3090 价格更具吸引力。
   - 这一趋势标志着 AI 社区正转向获取更易得的 GPU 资源，从而推动了创意产出。
- **关于许可和模型所有权的辩论**：**Flux** 的发布引发了关于模型所有权以及围绕为 **Stable Diffusion 3** 开发的技术的法律影响的讨论。随着 AI 艺术领域竞争的加剧，用户对知识产权的转移进行了推测。
   - 新兴模型的出现引发了关于未来许可策略和市场动态的疑问。
- **增强 AI 艺术 Prompt 生成**：参与者强调需要改进 **prompt generation** 技术，以增强在各种艺术风格中的可用性。关于迭代过程中速度与质量之间的权衡，意见各不相同。
   - 一些人优先考虑有助于快速概念迭代的模型，而另一些人则主张专注于图像质量。
- **用户交流关于写实主义 (Photo-Realism) 的见解**：讨论集中在如何在各种模型中实现**写实主义**，用户分享了他们对各模型优缺点的看法。针对高质量图像生成的不同 GPU 性能评估也是对话的一部分。
   - 这种集体评估强调了在 AI 艺术中不断优化图像忠实度的追求。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **LoRA 训练技术**：用户讨论了以 **4-bit** 或 **16-bit** 格式保存和加载使用 **LoRA** 训练的模型，并指出需要进行合并（merging）以保持模型准确性。
   - 对于量化方法和正确的加载协议存在困惑，以防止性能下降。
- **TPU 速度超越 T4 实例**：成员们强调了 **TPU** 在模型训练中相对于 **T4 instances** 的速度优势，尽管他们指出缺乏关于 TPU 实现的可靠文档。
   - 用户一致认为需要更好的示例来演示如何有效地将 TPU 用于训练。
- **GGUF 量化产生乱码**：有报告称模型在 **GGUF** 量化后生成**乱码**，特别是在 **Llamaedge** 平台上，这引发了关于 Chat Template 潜在问题的讨论。
   - 这一趋势引起了关注，因为这些模型在 **Colab** 等平台上表现依然正常。
- **Bellman 模型的最新微调**：新上传的 **Bellman** 版本基于 **Llama-3.1-instruct-8b** 进行微调，专注于使用瑞典语维基百科数据集进行 Prompt 问答，并显示出改进。
   - 尽管在问答方面有所进步，该模型在*故事生成方面表现吃力*，表明仍有进一步增强的空间。
- **竞争白热化：Google vs OpenAI**：一篇 Reddit 帖子指出 **Google** 据称正凭借新模型超越 **OpenAI**，这在社区内引起了惊讶和怀疑。
   - 参与者辩论了模型评分的主观性，以及感知到的改进究竟是真正的技术进步，还是仅仅反映了用户的交互偏好。

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **神经网络模拟吸引社区关注**：一位成员展示了一个有趣的[模拟](https://starsnatched.github.io/)，有助于理解神经网络，强调了创新的学习技术。
   - 该模拟激发了人们对这些技术如何推向模型能力极限的兴趣。
- **破解图像聚类技术**：分享了一个关于[使用图像描述符（Image Descriptors）进行图像聚类](https://www.youtube.com/watch?v=8f4oRcSnfbI)的视频，旨在增强数据组织和分析。
   - 该资源提供了有效的方法来利用视觉数据进行多样化的 AI 应用。
- **发布巨量合成数据集**：一个广泛的合成数据集现已在 [Hugging Face](https://huggingface.co/datasets/tabularisai/oak) 上可用，极大地帮助了机器学习领域的研究人员。
   - 该数据集是专注于表格数据分析项目的关键工具。
- **医疗和金融领域的动态 AI 模型**：新模型 **Palmyra-Med-70b** 和 **Palmyra-Fin-70b** 已在 [Hugging Face](https://huggingface.co/Writer/Palmyra-Med-70B) 上发布，用于医疗和金融应用，性能表现令人印象深刻。
   - 正如 [Sam Julien 的推文](https://x.com/samjulien/status/1818652901130354724)所证明的那样，这些模型可能会对症状诊断和财务预测产生重大影响。
- **应对学习中的技能差距**：对参与者之间显著技能差异的担忧引发了对竞赛期间工作量不平衡的恐惧。
   - 成员们建议采取公平的方法，以确保在学习活动中照顾到所有技能水平。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Uber One 会员可获得一年 Perplexity Pro**：美国和加拿大的符合条件的 Uber One 会员可以兑换价值 **200 美元** 的免费一年 [Perplexity Pro](https://pplx.ai/uber-one)。
   - 此项活动旨在增强用户的信息获取能力，让他们能够随时随地使用 Perplexity 的“回答引擎”进行查询。
- **围绕 Uber One 促销资格的困惑**：社区成员不清楚为期一年的 Perplexity Pro 访问权限是否适用于所有用户，有多份关于兑换促销代码出现问题的报告。
   - 担忧集中在资格问题以及注册后促销邮件发送错误，引发了广泛讨论。
- **数学突破引发关注**：最近在数学领域的一项[发现](https://www.youtube.com/embed/FOU-n9Xwp4U)可能会改变我们对**复杂方程**的理解，引发了关于其更广泛影响的讨论。
   - 细节仍然很少，但围绕这一突破的兴奋情绪继续激发各领域的兴趣。
- **大奖章基金（Medallion Fund）持续领跑收益**：在 Jim Simons 的管理下，自 1988 年成立以来，**Medallion Fund** 的平均年回报率在扣除费用前为 **66%**，扣除费用后为 **39%**，详见[神秘的大奖章基金](https://www.perplexity.ai/page/the-enigmatic-medallion-fund-xkICvfd7T7.WsILxst6bpg)。
   - 其神秘的表现令人侧目，因为它始终优于 **Warren Buffett** 等著名投资者。
- **创新混合抗体靶向 HIV**：研究人员通过将羊驼纳米抗体（nanobodies）与人类抗体结合，设计出一种混合抗体，可中和 **95% 以上** 的 HIV-1 毒株，由[佐治亚州立大学](https://www.perplexity.ai/page/hybrid-human-llama-antibody-fi-UCs.nTMFTu6QaRoOTXp0gA)分享。
   - 这些更小的**纳米抗体**比传统抗体能更有效地穿透病毒防御，展示了 HIV 治疗的一个充满希望的前景。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Vulkan llama.cpp 引擎发布！**：全新的 **Vulkan llama.cpp 引擎** 取代了之前的 OpenCL 引擎，为 **AMD**、**Intel** 和 **NVIDIA** 独立 GPU 开启了 GPU 加速。此更新包含在 **0.2.31** 版本中，可通过 [应用内更新](https://lmstudio.ai) 或在官网获取。
   - 用户反馈在使用 Vulkan 时性能有显著提升，加快了本地 LLM 的执行速度。
- **新增 Gemma 2 2B 模型支持**：**0.2.31** 版本引入了对 Google **Gemma 2 2B 模型** 的支持，可在此处 [下载](https://model.lmstudio.ai/download/lmstudio-community/gemma-2-2b-it-GGUF)。这一新模型增强了 LM Studio 的功能，建议从 **lmstudio-community 页面** 下载。
   - 该模型的集成让用户在 AI 工作负载中能够获得更强大的能力。
- **Flash Attention KV Cache 配置**：最新更新允许用户通过 **Flash Attention** 配置 **KV Cache 数据量化**，从而优化大模型的内存占用。然而，需要注意的是 *许多模型并不支持 Flash Attention*，因此该功能目前处于实验阶段。
   - 用户应谨慎操作，因为根据模型兼容性的不同，可能会出现性能不一致的情况。
- **来自用户的 GPU 性能洞察**：有用户报告在 Vulkan 支持下，使用 **RX6700XT** 运行模型的速度约为 **30 tokens/second**，展示了强大的性能表现。另一位用户指出，在 Llama 3-8B-16K-Q6_K-GGUF 模型上达到了 **40 tokens/second**。
   - 这些基准测试强调了当前配置的有效性，并为 LM Studio 的进一步性能调优提供了参考。
- **LM Studio 的兼容性问题**：一名用户报告其 **Intel Xeon E5-1650** 由于缺乏 AVX2 指令支持，在运行 LM Studio 时遇到兼容性挑战。社区建议使用仅限 AVX 的扩展程序，或考虑升级 CPU 以解决性能问题。
   - 这凸显了在部署 AI 模型时进行硬件兼容性检查的必要性。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Nvidia GPU 指令周期资源**：一名成员征求关于 **Nvidia GPU 指令周期** 的优质资源，并分享了一篇 [研究论文](https://conferences.computer.org/cpsiot/pdfs/RTAS2020-4uXAu5nqG7QNiz5wFYyfj6/549900a210/549900a210.pdf) 和另一项专注于每条指令时钟周期的 [微架构研究](https://arxiv.org/abs/2208.11174)。
   - 这一探究有助于理解不同 Nvidia 架构之间的 **性能差异**。
- **第 8 个 Epoch 准确率分数剧增**：一名成员观察到在训练过程中，准确率分数在 **第 8 个 Epoch 显著飙升**，这引发了对模型性能稳定性的担忧。
   - 他们强调这类波动可能是典型的，并引发了关于模型评估实践的广泛讨论。
- **理解 Triton 内部机制与 GROUP_SIZE_M**：讨论澄清了 Triton 分块 matmul 教程中的 **GROUP_SIZE_M** 如何控制数据块的处理顺序，从而提高 L2 cache 命中率。
   - 成员们指出理解 **GROUP_SIZE_M** 与 **BLOCK_SIZE_{M,N}** 之间区别的重要性，教程中的图示有助于加深理解。
- **对 AI 行业 Acquihires（人才收购）的关注**：包括 Character AI 和 Inflection AI 在内的多家公司正在经历 **acquihires**，这表明初创公司被大公司吸收已成为一种趋势。
   - 这引发了关于竞争潜在影响，以及 AI 开发中编程技能与概念思维之间平衡的讨论。
- **关于 Tensor 操作随机性的辩论**：成员们注意到，操作中不同的 Tensor 形状可能会调用不同的 kernels，导致即使是相似的操作也会产生不同的数值输出。
   - 建议实现自定义随机数生成器，以确保跨操作的一致性。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **MoMa 架构增强了混合模态语言建模**：Meta 推出了 [MoMa](https://arxiv.org/pdf/2407.21770)，这是一种稀疏早期融合（sparse early-fusion）架构，通过使用 Mixture-of-Expert 框架提升了预训练效率。
   - 该架构改进了对交错混合模态 Token 序列的处理，标志着多模态 AI 的重大进展。
- **BitNet 微调取得显著成果**：一位用户报告称，微调 **BitNet** 得到了一个 **74MB** 的文件，在单核 CPU 上每秒可处理 **198 个 Token**，展示了令人印象深刻的效率。
   - 该技术正以 [Biggie-SmoLlm](https://huggingface.co/nisten/Biggie-SmoLlm-0.15B-Base) 的名称开源。
- **Character.ai 在收购后的战略转变**：Character.ai 的联合创始人已加入 Google，导致其产品转向使用 **Llama 3.1** 等开源模型。
   - 此举引发了关于行业人才流转以及在大科技公司收购背景下初创公司生存能力的讨论。
- **DeepSeek API 引入磁盘上下文缓存**：**DeepSeek API** 推出了上下文缓存（context caching）功能，可降低高达 **90%** 的 API 成本，并显著降低首个 Token 的延迟。
   - 这一改进通过缓存频繁引用的上下文来支持多轮对话，从而提升性能。
- **Winds of AI Winter 播客发布**：标题为 *Winds of AI Winter* 的最新一集已上线，内容包括对过去几个月 AI 领域的总结，并庆祝下载量突破 **100 万次**。
   - 听众可以通过 [播客链接](https://latent.space/p/q2-2024-recap) 收听完整讨论。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **令人兴奋的 AI 黑客松系列巡回赛开始**：**AI Hackathon Series Tour** 在全美拉开帷幕，最终将迎来 **PAI Palooza**，届时将展示当地的 AI 创新和初创公司。参与者现在可以注册参加这一专注于推进 AI 技术的协作活动。
   - 该系列活动旨在吸引社区参与有意义的技术讨论，并在地方层面促进创新。
- **GraphRAG 系统助力投资者**：引入了一套全新的 **GraphRAG** 系统，利用从 **200 万个抓取的** 公司网站中获得的洞察，帮助投资者识别有潜力的公司。该系统目前正与一个 Multi-Agent 框架同步开发，以提供更深层次的见解。
   - 开发者正在积极寻求合作者以增强系统功能。
- **Neurosity Crown 增强专注力**：**Neurosity Crown** 因其在注意力下降时提供音频提示（如鸟鸣声）来提高专注力的能力而受到关注。一些用户强调了其对生产力的显著提升，尽管也有人对其整体效果表示怀疑。
   - 它的可用性引发了关于整合技术方案以提高生产力的持续讨论。
- **寻找 Web3 合约机会**：一位成员正在寻求与 **Web3**、**Chainlink** 和 **UI** 开发方面的资深开发者讨论兼职合约职位，这表明了对**新兴技术**技能的需求。
   - 这突显了社区对进一步提升区块链和 UI 集成技术专长的兴趣。
- **工具包定制引发关注**：围绕 **toolkit** 的定制功能（如启用身份验证）引起了热议，这可能需要通过 Fork 和创建 Docker 镜像来进行大规模修改。目前已提出安全修改的社区指南，强调协作改进。
   - 成员们正在评估其应用，特别是关于内部工具扩展和 **upstream updates**（上游更新）方面。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **GitHub 对 Hugging Face 的挑战**：GitHub 最新的模型托管方式引发了担忧，被认为是一个有限的演示，与 **Hugging Face** 的模型共享理念相比，削弱了社区贡献。
   - 成员们推测这是一种控制 ML 社区代码并防止人员大规模流失的策略。
- **对欧盟 AI 法规的质疑依然存在**：随着即将出台的针对大型模型的 AI 法案，人们对潜在的执行力及其对全球公司（尤其是初创公司）的影响表示怀疑。
   - 讨论集中在新的立法框架可能如何无意中扼杀创新和适应性。
- **应对 LLM 评估指标的挑战**：一位成员询问了评估 **LLM** 输出的最佳指标，特别强调了对代码输出使用精确匹配（exact matches）的复杂性。
   - 虽然提出了像 humaneval 这样的建议，但对评估过程中使用 `exec()` 的影响表示担忧，并引发了进一步辩论。
- **对蒸馏技术的兴趣重燃**：成员们讨论了对 **logit distillation**（Logit 蒸馏）关注度的回升，揭示了其对数据效率的影响以及对较小模型微小的质量提升。
   - 最近的论文展示了蒸馏的多样化应用，特别是那些结合了合成数据集的应用。
- **GEMMA 的性能接受测试**：**GEMMA** 与 **Mistral** 的性能对比出现了差异，导致了关于评估过程缺乏透明度的辩论。
   - 针对训练动态和资源分配是否准确反映了模型结果，人们提出了担忧。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI 语音模式引发咨询热潮**：新的 **OpenAI voice mode** 发布后，社区内涌入了大量私信。
   - *似乎许多人都渴望了解更多关于其功能和访问权限的信息*。
- **Assistants API 的延迟困扰**：成员们报告了 **Assistants API** 的延迟问题，一些人建议使用 SerpAPI 等替代方案进行实时抓取。
   - *社区反馈集中在共享经验和潜在的变通方法上*。
- **Gemini 1.5 Pro 证明了其竞争力**：讨论强调了 **Gemini 1.5 Pro 的性能**，激发了对其现实应用和响应能力的关注。
   - *一位参与者指出，他们的使用体验展示了该模型极具竞争力的响应质量*。
- **Gemma 2 2b 模型见解**：关于 **Gemma 2 2b 模型** 的见解表明，尽管与大型模型相比缺乏知识储备，但它在指令遵循方面表现出色。
   - 对话反思了在实际应用中如何平衡模型能力与可靠性。
- **Flux 图像模型令社区兴奋**：**Flux 图像模型** 的发布引发了兴奋，用户开始测试其对比 MidJourney 和 DALL-E 等工具的能力。
   - *值得注意的是，它的开源性质和较低的资源需求表明其具有广泛采用的潜力。*

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **关于 LLM-as-Judge 的见解**：成员们征求了关于 **LLM-as-Judge** 元框架以及专注于**指令和偏好**（instruction and preference）数据的合成数据集策略的必读综述。
   - 这一咨询强调了在 **LLM** 领域开发有效方法论的浓厚兴趣。
- **新 VRAM 计算工具发布**：一个新的 [VRAM 计算脚本](https://gist.github.com/jrruethe/8974d2c8b4ece242a071d1a1526aa763) 使用户能够根据各种参数确定 LLM 的 VRAM 需求。
   - 该脚本无需外部依赖即可运行，旨在简化对 LLM 上下文长度和每权重位数（bits per weight）的评估。
- **Gemma 2B 与 Qwen 1.5B 的对比**：成员们讨论了 **Gemma 2B** 被过度炒作的问题，并将其与 **Qwen 1.5B** 进行对比，据报道后者在 MMLU 和 GSM8K 等基准测试中表现更优。
   - Qwen 的能力在很大程度上被忽视了，导致有评论称其“残暴地”超越了 Gemma 2B。
- **Llama 3.1 微调挑战**：一位用户在私有数据集上微调了 **Llama 3.1**，通过 [vLLM](https://github.com/vllm/vllm) 运行时仅达到 **30tok/s**，且输出内容杂乱无章。
   - 尽管温度（temperature）设置为 **0**，问题依然存在，这表明可能存在模型配置错误或数据相关性问题。
- **推理任务的新 Quarto 网站设置**：一个 [Quarto 网站的 PR](https://github.com/NousResearch/Open-Reasoning-Tasks/pull/17) 已经启动，专注于增强推理任务的在线可见性。
   - 最近对文件夹结构的调整旨在简化项目管理，并提高仓库内的导航便利性。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **FLUX Schnell 表现出弱点**：成员们讨论了 **FLUX Schnell** 模型似乎训练不足，且在提示词遵循（prompt adherence）方面表现不佳，产生了一些荒谬的输出，例如 *“一名穿着网球服骑着金翼摩托车的女性”*。
   - 他们担心该模型更像是一个**数据集记忆机器**，而非有效的生成模型。
- **对合成数据集的使用建议谨慎**：针对使用 **FLUX Schnell** 模型生成合成数据集的做法出现了担忧，理由是存在跨代**表征崩溃（representational collapse）**的风险。
   - 一位成员警告说：*“求求大家，千万不要用 Flux 公开发布的权重来制作合成数据集”*。
- **精选数据集优于随机噪声的价值**：强调了**精选数据集（curated datasets）**的重要性，认为用户偏好的数据对于质量和资源效率至关重要。
   - 成员们一致认为，在随机提示词上进行训练会**浪费资源**，且不会带来显著改进。
- **Bug 阻碍了 LLM 的进展**：一位成员在代码中发现了一个**拼写错误**，该错误严重影响了 **50 多个实验**的性能，并对新优化的**损失曲线（loss curve）**感到满意。
   - 他们表示如释重负，因为新的曲线下降速度明显快于以前，这展示了调试的重要性。
- **关注强大的基准模型**：讨论转向了创建**强大基准模型（baseline model）**的需求，而不是纠结于正则化技术带来的微小改进。
   - 成员们注意到工作重点正转向开发**分类器（classifier）**，同时考虑**参数高效架构（parameter-efficient architecture）**。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **对活动赞助的兴趣**：成员们对**赞助活动**表现出热情，标志着对未来聚会的积极态度。
   - 这种乐观情绪表明，可能会获得潜在的资金支持来推动这些倡议。
- **Character AI 交易引发关注**：**Character AI 交易**引发了成员们的怀疑，质疑其对 AI 领域的影响。
   - *一位参与者称这是一场“奇怪的交易”，引发了对交易后员工和公司影响的进一步担忧。*
- **Ai2 发布受闪烁图标启发的新品牌**：**Ai2** 推出了新品牌和网站，采用了 AI 品牌推广中流行的**闪烁表情符号（sparkles emojis）**趋势，正如一篇 [Bloomberg 文章](https://www.bloomberg.com/news/newsletters/2024-07-10/openai-google-adobe-and-more-have-embraced-the-sparkle-emoji-for-ai?srnd=undefined)中所讨论的那样。
   - *Rachel Metz* 强调了这一转变，突出了行业对这种美学日益增长的迷恋。
- **Magpie Ultra 数据集发布**：HuggingFace 发布了 **Magpie Ultra** 数据集，这是一个包含 **50k 未过滤**数据的 L3.1 405B 数据集，声称它是开放合成数据集的先驱。查看他们的 [推文](https://x.com/gabrielmbmb_/status/1819398254867489001) 和 [HuggingFace 上的数据集](https://huggingface.co/datasets/argilla/magpie-ultra-v0.1)。
   - *初始指令质量仍存疑问*，特别是在用户轮次多样性和覆盖范围方面。
- **下周 RL 会议晚宴**：一位成员正考虑下周在 **RL Conference** 举办晚宴，正在寻找有兴趣赞助的 VC 或朋友。
   - 这一倡议可能为寻求贡献的行业专业人士提供极佳的人际网络机会。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 网站访问性**：用户报告访问 [OpenRouter website](https://status.openrouter.ai/) 存在问题，局部停机偶尔会影响部分地区。
   - 一位特定用户指出网站问题已短暂解决，突显了用户体验可能存在非统一性。
- **Anthropic 服务困境**：多位用户指出 **Anthropic services** 正面临严重的负载问题，导致数小时内间歇性无法访问。
   - 这引发了人们对基础设施处理当前需求能力的担忧。
- **聊天室改版与增强设置**：**Chatroom** 已实现功能化，采用了[更简洁的 UI](https://openrouter.ai/chat) 并支持本地保存聊天记录，提升了用户交互体验。
   - 用户现在可以通过 [settings page](https://openrouter.ai/settings/preferences) 配置设置，以避免将请求路由到某些供应商，从而优化体验。
- **API Key 获取变得简单**：用户获取 API Key 就像注册、充值并在插件中使用一样简单，无需任何技术技能（[了解更多](https://help.aiassistworks.com/help/how-easy-it-is-to-get-an-api-key)）。
   - 使用自己的 API Key 可以获得更优惠的价格——对于 **GPT-4o-mini** 等事件，**1,000,000 tokens 仅需 $0.6**——并通过供应商仪表板清晰地洞察模型使用情况。
- **了解免费模型使用限制**：讨论强调，免费模型在 API 访问和聊天室使用方面通常都有显著的速率限制 (rate limits)。
   - 这些约束对于管理服务器负载和确保用户之间的公平访问至关重要。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **事件驱动 RAG 流水线教程发布**：分享了一个关于构建 **RAG pipeline** 的教程，详细介绍了使用 [LlamaIndex workflows](https://t.co/XGmm6gQhcI) 在特定步骤中进行检索、重排序和合成。这份综合指南旨在展示用于流水线构建的事件驱动架构。
   - *你可以逐步实现本教程*，从而更好地集成各种事件处理方法。
- **为印度农民开发的 AI 语音 Agent**：如[这条推文](https://t.co/lrGDFSl0HH)所述，已开发出一款 **AI Voice Agent** 来支持印度农民，解决由于政府援助不足而产生的资源需求。该工具旨在提高他们的生产力并应对挑战。
   - 这一举措体现了技术在解决关键农业问题、改善农民生计方面的潜力。
- **无工具 ReAct Agent 的策略**：用户寻求关于配置 **ReAct agent** 在无工具状态下运行的指导，建议的方法包括 `llm.chat(chat_messages)` 和 `SimpleChatEngine` 以实现更流畅的交互。成员们讨论了 Agent 错误的挑战，特别是关于缺失工具请求的问题。
   - 寻找这些问题的解决方案仍然是提高 Agent 实现中可用性和性能的优先事项。
- **LlamaIndex Service Context 的变化**：成员们研究了 **LlamaIndex** 即将移除 **service context** 的变动，这会影响 `max_input_size` 等参数的设置方式。这一转变引发了对需要进行大量代码调整的担忧。
   - *一位用户表达了他们的沮丧*，这影响了开发者的工作流，特别是向基础 API 中更独立组件的过渡。
- **DSPy 最新更新破坏了 LlamaIndex 集成**：一位成员报告 **DSPy** 的最新更新导致与 **LlamaIndex** 的集成失败。他们指出，与标准的 LlamaIndex 抽象相比，之前的版本 **v2.4.11** 在 Prompt 微调结果上没有任何改进。
   - 该用户在更新后实现 DSPy 运行成功方面仍面临障碍。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 的错误处理困境**：成员们讨论了围绕 **Mojo** 错误处理的困境，比较了 **Python 风格的异常** 和 **Go/Rust 的错误值**，并担心混合两者可能会导致复杂性。
   - 有人感叹这可能会“在程序员面前爆炸”（意指产生严重后果），强调了在 Mojo 中有效管理错误的复杂性。
- **Max 的安装烦恼**：一位成员报告在安装 **Max** 时遇到困难，表示运行代码并不顺利。
   - 他们正在寻求帮助以排查有问题的安装过程。
- **Mojo Nightly 表现出色！**：对于一位活跃的贡献者来说，**Mojo nightly** 运行顺畅，表明尽管 Max 存在问题，但其稳定性良好。
   - 这表明 Mojo 的 nightly 构建版本提供了可靠的体验，在处理安装问题时可以加以利用。
- **Conda 安装或许能解决问题**：一位成员建议使用 **conda** 作为安装问题的可能解决方案，并指出该过程最近已变得更加简单。
   - 这可能会显著减轻那些面临 Max 安装挑战的人在排查和解决问题时的负担。



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter 会话混淆已解决**：成员们对加入正在进行的会话感到困惑，澄清了对话发生在特定的语音频道中。
   - 一位成员提到他们费力才找到该频道，直到其他人确认其可用性。
- **运行本地 LLM 的指导**：一位新成员寻求运行本地 LLM 的帮助，并分享了他们遇到模型加载错误的初始脚本。
   - 社区成员引导他们查看[文档](https://docs.openinterpreter.com/language-models/local-models/llamafile#llamafile)以正确设置本地模型。
- **明确 LlamaFile 服务器的启动**：强调在 Python 模式下使用 **LlamaFile** 之前，必须单独启动其服务器。
   - 参与者确认了 API 设置的正确语法，强调了不同加载函数之间的区别。
- **Aider 浏览器 UI 演示发布**：新的 [Aider 浏览器 UI 演示视频](https://aider.chat/docs/usage/browser.html)展示了与 LLM 协作在本地 git 仓库中编辑代码。
   - 它支持 **GPT 3.5**、**GPT-4** 等模型，并具有使用合理的提交信息实现自动 commit 的功能。
- **讨论 LLM 应用中的事后验证**：研究强调，由于代码理解方面的挑战，人类目前在创建后验证 **LLM 生成的输出** 时面临困难。
   - 该研究建议加入“撤销”功能并建立**损害限制 (damage confinement)**，以便于事后验证 [更多详情请点击此处](https://gorilla.cs.berkeley.edu/blogs/10_gorilla_exec_engine.html)。



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **LLM 提升其判断技能**：最近一篇关于 LLM **Meta-Rewarding** 的论文展示了增强的自我判断能力，将 **Llama-3-8B-Instruct** 在 AlpacaEval 2 上的胜率从 **22.9%** 提升至 **39.4%**。
   - 这一 meta-rewarding 步骤解决了传统方法的饱和问题，提出了一种模型评估的新方法。
- **MindSearch 模拟人类认知**：[MindSearch](https://arxiv.org/abs/2407.20183) 框架利用 LLM 和多智能体 (multi-agent) 系统来解决信息整合挑战，增强了对复杂请求的检索。
   - 论文讨论了该框架如何有效缓解由 LLM **上下文长度 (context length)** 限制带来的挑战。
- **构建 DSPy 摘要流水线**：成员们正在寻求关于使用 DSPy 配合开源模型进行摘要的教程，以迭代提升 prompt 的有效性。
   - 该计划旨在优化符合技术需求的摘要结果。
- **征集 Discord 频道导出数据**：出现了一项征集志愿者分享 JSON 或 HTML 格式的 Discord 频道导出数据的请求，旨在进行更广泛的分析。
   - 在发布调查结果和代码时，将对贡献者表示感谢，从而加强社区协作。
- **将 AI 集成到游戏角色开发中**：关于使用来自 GitHub 的代码开发 AI 驱动的游戏角色的讨论非常热烈，特别是针对**巡逻**和动态玩家交互。
   - 成员们表示有兴趣实现 Oobabooga API，以促进游戏角色的高级对话功能。



---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Fine-tuning Gemma2 2B 备受关注**：成员们探讨了 **Fine-tuning Gemma2 2B** 模型并分享了见解，其中有人建议利用 **pretokenized dataset** 以更好地控制模型输出。
   - 社区的反馈显示了不同的经验，他们热衷于从调整后的方法中获得进一步的结果。
- **寻求日本顶尖语言模型**：在寻找最流利的日语模型时，根据社区的输入，有人推荐了基于 **lightblue** 的 **suzume** 模型。
   - 用户表示有兴趣了解更多关于该模型在现实世界应用中的情况。
- **BitsAndBytes 简化了 ROCm 安装**：最近的一个 [GitHub PR](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1299) 简化了 **BitsAndBytes** 在 ROCm 上的安装，使其兼容 **ROCm 6.1**。
   - 成员们指出，此次更新允许打包与最新的 Instinct 和 Radeon GPU 兼容的 wheel 文件，标志着重大改进。
- **训练 Gemma2 和 Llama3.1 的输出问题**：用户详细描述了他们在训练 **Gemma2** 和 **Llama3.1** 时遇到的困难，指出模型往往只有在达到 **max_new_tokens** 后才会停止。
   - 人们越来越担心在训练上投入的时间与输出质量的提升不成正比。
- **Prompt Engineering 的效果微乎其微**：尽管为了引导模型输出付出了严苛的 Prompt 努力，用户报告称对整体行为的影响**微乎其微**。
   - 这引发了关于当前 AI 模型中 Prompt Engineering 策略有效性的质疑。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain 0.2 文档缺失**：用户报告称 **LangChain v0.2** 中缺乏关于 **agent functionalities** 的文档，导致对其能力的质疑。
   - *Orlando.mbaa* 特别指出他们找不到任何关于 Agent 的引用，引发了对易用性的担忧。
- **在 RAG 应用中实现 Chat Sessions**：讨论了如何在基础 **RAG** 应用中加入 **chat sessions**，类似于 ChatGPT 对历史对话的跟踪。
   - 参与者评估了在现有框架内进行 Session 跟踪的可行性和可用性。
- **LangChain 中的 Postgres Schema 问题**：一名成员引用了一个关于 **Postgres** 中 **chat message history** 失败的 GitHub Issue，特别是在使用显式 Schema 时 ([#17306](https://github.com/langchain-ai/langchain/issues/17306))。
   - 人们对拟议解决方案的有效性及其对未来实现的影响表示担忧。
- **使用 Testcontainers 测试 LLM**：分享了一篇博文，详细介绍了在 Python 中利用 **4.7.0 版本**，使用 [Testcontainers](https://testcontainers.com/) 和 **Ollama** 测试 **LLM** 的过程。
   - 鼓励对[此处](https://bricefotzo.medium.com/testing-llms-and-prompts-using-testcontainers-and-ollama-in-python-81e8f7c18be7)提供的教程提供反馈，强调了稳健测试的必要性。
- **来自社区研究电话会议 #2 的精彩更新**：最近的 [Community Research Call #2](https://x.com/ManifoldRG/status/1819430033993412856) 强调了 **Multimodality**、**Autonomous Agents** 和 **Robotics** 项目的激动人心的进展。
   - 参与者积极讨论了多个**协作机会**，强调了未来研究方向中的潜在合作伙伴关系。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **QAT Quantizers 说明**：成员们讨论了 QAT recipe 支持 **Int8DynActInt4WeightQATQuantizer**，而 **Int8DynActInt4WeightQuantizer** 用于训练后量化，目前尚不支持。
   - 他们指出只有 **Int8DynActInt4Weight** 策略适用于 QAT，其他量化器计划在未来实现。
- **请求对 SimPO PR 进行审查**：一名成员强调需要对 GitHub 上的 **SimPO (Simple Preference Optimisation)** PR #1223 进行澄清，该 PR 旨在解决 Issue #1037 和 #1036。
   - 他们强调该 PR 解决了对齐问题，呼吁更多的监督和反馈。
- **文档重构的 RFC**：出现了一项关于改进 **torchtune** 文档系统的提案，重点是更流畅的 recipe 组织以改善入门体验。
   - 鼓励成员提供见解，特别是关于 **LoRA single device** 和 **QAT distributed** 的 recipe。
- **对新模型页面的反馈**：一位参与者分享了一个潜在**新模型页面**的预览链接，旨在解决当前文档中的可读性问题。
   - 讨论的细节包括需要清晰且详尽的模型架构信息，以增强用户体验。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Computer Vision Enthusiasm**: 成员们表达了对 **Computer Vision** 的共同兴趣，强调了其在当前技术领域的重要性。
   - *许多成员似乎渴望从主导各类会议的 NLP 和 genAI 讨论中分流出来。*
- **Conferences Reflect Machine Learning Trends**: 一位成员分享了参加两次 **Machine Learning** 会议的经历，会上展示了他们在 **Gaussian Processes** 和 **Isolation Forest** 模型方面的工作。
   - *他们注意到许多与会者对这些主题并不熟悉，这表明讨论存在向 **NLP** 和 **genAI** 倾斜的强烈偏见。*
- **Skepticism on genAI ROI**: 参与者质疑 **genAI** 的 **ROI**（投资回报率）是否能达到预期，表明可能存在脱节。
   - *一位成员强调，正向的 ROI 需要初始投资，这表明预算通常是根据感知价值分配的。*
- **Funding Focus Affects Discussions**: 一位成员指出，**Funding** 通常流向 **Budgets** 分配的地方，从而影响技术讨论。
   - *这强调了市场细分和 **Hype Cycles** 在塑造行业活动焦点方面的重要性。*
- **Desire for Broader Conversations**: 鉴于这些讨论，一位成员表示很高兴能有一个平台来讨论 **genAI** 炒作之外的话题。
   - *这反映了对涵盖 **Machine Learning** 各个领域（而非仅限于主流趋势）的多元化对话的渴望。*



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **Image Generation Time Inquiry**: 讨论集中在 **A100** 上使用 **FLUX Schnell** 生成 **1024px image** 所需的时间，引发了关于性能预期的疑问。
   - 然而，对话中没有提到在该硬件上生成图像的具体耗时。
- **Batch Processing Capabilities Explored**: 出现了关于图像生成是否可以进行 **Batch Processing** 以及可以处理的最大图像数量的问题。
   - 对话中缺少关于硬件能力和限制的具体回应。



---


**tinygrad (George Hotz) Discord** 没有新消息。如果该服务器长时间没有动态，请告知我们，我们将将其移除。


---


**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该服务器长时间没有动态，请告知我们，我们将将其移除。


---


**Mozilla AI Discord** 没有新消息。如果该服务器长时间没有动态，请告知我们，我们将将其移除。


---


**DiscoResearch Discord** 没有新消息。如果该服务器长时间没有动态，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器长时间没有动态，请告知我们，我们将将其移除。


---

# 第 2 部分：按频道分类的详细摘要和链接


{% if medium == 'web' %}

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1268651841575714816)** (592 条消息🔥🔥🔥): 

> - `Flux Model Performance` (Flux 模型性能)
> - `GPU Utilization in AI Art` (AI 艺术中的 GPU 利用率)
> - `Licensing and Model Restrictions` (许可与模型限制)
> - `Prompt Generation Techniques` (Prompt 生成技术)
> - `Online GPU Hosting Services` (在线 GPU 托管服务) 


- **Flux 模型生成的图像质量参差不齐**：用户讨论了 Flux 模型的性能，指出它在处理抽象风格时有时表现不佳，特别是在生成女性躺在草地上的图像时。
   - 虽然有些人通过详细的 Prompt 取得了成功，但其他人报告了不一致的结果，表明该模型具有与 SD3 类似的核心局限性。
- **GPU 托管服务日益普及**：几位用户分享了他们使用 RunPod 和 Vast 等在线 GPU 托管服务的经验，指出价格根据硬件和性能需求有很大差异。
   - 一些人青睐 RunPod 的精致体验，而另一些人则欣赏 Vast 的低成本，特别是对于 3090s。
- **关于许可和模型所有权的讨论**：Flux 的发布引发了关于模型所有权的潜在影响以及围绕使用为 SD3 开发的技术的法律方面的对话。
   - 用户推测了知识产权的转移以及 AI 艺术领域竞争模型的未来。
- **对 AI 艺术模型实际使用和工具链的担忧**：讨论强调了对 fine-tuning 和 Prompt 生成改进的需求，以增强跨不同艺术风格的可用性。
   - 关于速度与质量的重要性存在截然不同的观点，一些用户更喜欢允许在概念开发中进行快速迭代的模型。
- **用户在写实渲染方面的体验**：用户就实现写实主义的各种模型的现状交换了意见，评估了它们各自的优缺点。
   - 对话还涉及了不同 GPU 在高效生成高质量图像方面的性能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://comfyanonymous.github.io/ComfyUI_examples/flux/">Flux Examples</a>: ComfyUI 工作流示例</li><li><a href="https://arxiv.org/abs/2405.09818">Chameleon: Mixed-Modal Early-Fusion Foundation Models</a>: 我们介绍了 Chameleon，这是一个早期融合的基于 token 的混合模态模型系列，能够以任何任意序列理解和生成图像和文本。我们概述了一种稳定的训练方法...</li><li><a href="https://x.com/DataPlusEngine/status/1816252032929435850">Tweet from DataVoid (@DataPlusEngine)</a>: @ReyArtAge 第一个 epoch。似乎很有帮助！看来手部仍然有问题。</li><li><a href="https://x.com/virushuo/status/1819097766255079734">Tweet from virushuo (@virushuo)</a>: 未找到描述</li><li><a href="https://www.runpod.io/">RunPod - The Cloud Built for AI</a>: 为 AI 构建的云。在一个云端开发、训练和扩展 AI 模型。通过 GPU Cloud 启动按需 GPU，通过 Serverless 扩展 ML 推理。</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1ehiz51/flux_image_examples/#lightbox">Reddit - Dive into anything</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1268646107018301562)** (379 条消息🔥🔥): 

> - `使用 LoRA 进行训练`
> - `使用 TPU 进行模型训练`
> - `Padding tokens 的影响`
> - `在 vLLM 中实现 LoRA`
> - `准备微调数据集` 


- **在 4-bit 与 16-bit 模型上使用 LoRA 进行训练**：用户讨论了以 4-bit 或 16-bit 格式保存和加载使用 LoRA 训练的模型，强调了对量化和正确加载方法的潜在困惑。
   - 建议将从 4-bit 模型获得的 LoRA 与 16-bit 模型合并，以保持精度。
- **利用 TPU 实现更快的训练**：成员们注意到，与 T4 实例相比，使用 TPU 进行模型训练具有显著的速度优势，并强调了支持 TPU 的重要性。
   - 然而，文档中似乎缺乏关于 TPU 使用的充分知识和示例。
- **Padding tokens 对性能的影响**：讨论了大量的 padding tokens 是否会影响推理性能，并建议对数据集进行预处理以尽量减少 padding。
   - 在训练期间利用 'group by length' 可以通过减少不必要的 padding 来提高性能。
- **针对聊天 AI 的微调数据集**：对于特定的聊天 AI 训练，建议用户使用定制的数据集，并且对于推荐用于微调 LLAMA3.1 的特定数据集存在困惑。
   - 消息需要进行适当的格式化，理想情况下应包含最后几次交互作为上下文 (context)，以进行有效的训练。
- **模型加载最佳实践**：有人对使用不同模型版本时的内存错误表示担忧，特别是在没有进行适当量化的情况下加载 16-bit 模型时。
   - 建议使用 `load_in_4bit=True` 以确保与 4-bit 模型变体的 VRAM 使用保持一致。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/johnpaulbin/qwen1.5b-e2-1-lora">johnpaulbin/qwen1.5b-e2-1-lora · Hugging Face</a>：未找到描述</li><li><a href="https://www.kaggle.com/code/defdet/llama-2-13b-on-tpu-training/notebook">LLAMA 2 13B on TPU (Training)</a>：使用 Kaggle Notebooks 探索和运行机器学习代码 | 使用来自多个数据源的数据</li><li><a href="https://huggingface.co/fhai50032/RolePlayLake-7B">fhai50032/RolePlayLake-7B · Hugging Face</a>：未找到描述</li><li><a href="https://unsloth.ai/blog/llama3">Finetune Llama 3 with Unsloth</a>：通过 Unsloth 轻松微调 Meta 的新模型 Llama 3，支持 6 倍长的上下文长度！</li><li><a href="https://x.com/nisten/status/1818529201231688139">nisten (@nisten) 的推文</a>：为微调修改了 BitNet，最终得到了一个 74mb 的文件。仅在 1 个 CPU 核心上就能以每秒 198 个 tokens 的速度流畅对话。简直是巫术。稍后将通过 @skunkworks_ai 开源，基础模型在这里：https://huggi...</li><li><a href="https://x.com/_xjdr/status/1819401339568640257">xjdr (@_xjdr) 的推文</a>：L3.1 仅通过增加缩放的 RoPE 乘数即可扩展到 1M tokens，且几乎具有完美的召回率。无需额外训练。哈哈</li><li><a href="https://github.com/unslothai/unsloth/wiki#manually-saving-to-gguf">主页</a>：使用 Unsloth 将 Llama 3.1, Mistral, Phi &amp; Gemma LLMs 的微调速度提高 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://huggingface.co/datasets/fhai50032/magicoder-oss-instruct-sharegpt-75k">fhai50032/magicoder-oss-instruct-sharegpt-75k · Datasets at Hugging Face</a>：未找到描述</li><li><a href="https://x.com/danielhanchen/status/1819073748949258638">Daniel Han (@danielhanchen) 的推文</a>：我的 LLMs 低层级技术 3 小时研讨会发布了！我讨论了：1. 为什么训练是 O(N^2) 而不是立方级 2. Triton vs CUDA 3. 为什么使用 causal masks, layernorms, RoPE, SwiGLU 4. GPT2 vs Llama 5. Bug 修复...</li><li><a href="https://github.com/Locutusque/TPU-Alignment">GitHub - Locutusque/TPU-Alignment: Fully fine-tune large models like Mistral, Llama-2-13B, or Qwen-14B completely for free</a>：完全免费地对 Mistral, Llama-2-13B 或 Qwen-14B 等大模型进行全量微调 - Locutusque/TPU-Alignment</li><li><a href="https://huggingface.co/datasets/LDJnr/Capybara">LDJnr/Capybara · Datasets at Hugging Face</a>：未找到描述</li><li><a href="https://cloud.google.com/tpu?hl=id">Tensor Processing Unit (TPU)</a>：Google Cloud Tensor Processing Unit (TPU) 专为加速机器学习工作负载而构建。立即联系 Google Cloud 了解更多信息。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1268651575664971787)** (7 messages): 

> - `Google vs OpenAI`
> - `Chat Ratings` 


- **Google 据报道超越了 OpenAI**：一位成员分享了一个 [Reddit 帖子](https://www.reddit.com/r/ChatGPT/comments/1ehlmqd/finally_google_beat_openai_new_model_from_google/)，声称 **Google** 终于凭借一个新模型击败了 **OpenAI**。
   - 针对该帖子，有人表示 *我简直不敢相信……*，突显了对这一说法的惊讶。
- **关于“真实”本质的辩论**：另一位成员质疑了在这种语境下“真实”的定义，引发了关于感知的讨论。
   - 他们认为模型的评分可能仅仅反映了对话偏好，而非决定性的优越性。
- **对聊天评分（Chat Ratings）的怀疑**：有人担心评分是主观的，取决于与相关模型互动的个人体验。
   - 评论指出，改进可能仅仅源于更具**对话性（conversational）**的交互，而非能力的真正提升。



**提及的链接**：<a href="https://www.reddit.com/r/ChatGPT/comments/1ehlmqd/finally_google_beat_openai_new_model_from_google/">Reddit - Dive into anything</a>：未找到描述

  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1268654684785344512)** (99 messages🔥🔥): 

> - `GGUF quantization issues`
> - `Fine-tuning difficulties with Llama 3.1`
> - `Training on small datasets`
> - `LoRA parameters and learning rates`
> - `Incompatibility of Unsloth models` 


- **GGUF 量化导致乱码输出**：成员们报告了模型在 GGUF 量化后输出乱码的问题，特别是在 Llamaedge 平台上，而它们在 Colab 上运行正常。
   - 有建议认为 Chat Template 的问题可能导致了这一现象，最近似乎影响了许多用户。
- **微调 Llama 3.1 的挑战**：用户在尝试微调 Llama 3.1 模型时遇到了困难，即使经过多次尝试，量化后仍会出现异常行为。
   - 训练时长和方法似乎未能产生预期结果，引发了对模型学习能力的担忧。
- **小数据集阻碍模型训练**：讨论强调，使用极小的数据集会严重限制模型在没有广泛微调的情况下有效学习的能力。
   - 专家建议使用更大、更多样化的数据集，或依靠 RAG 等替代方法来提高模型性能。
- **LoRA 参数影响性能**：小组讨论了 LoRA 参数的影响，强调对于小数据集，建议使用较低的秩 (r) 以避免过拟合。
   - 然而，对于较大的数据集，较高的 r 值可能更有利，同时需要适当调整学习率。
- **模型与 AutoModelForCausalLM 的不兼容性**：关于较新的 Unsloth 模型与 `AutoModelForCausalLM` 的兼容性，以及它是否支持最近的模型更新存在疑问。
   - 一些成员指出他们在旧模型上取得了成功，但在尝试下载 Mistral 和 Llama 3.1 等新发布版本时遇到了问题。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/1eg5wgb/llama_31_changed_its_chat_template_again/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://huggingface.co/Ak1104">Ak1104 (Akshat Shrivastava)</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/basics/lora-parameters-encyclopedia>">Unsloth Documentation</a>：未找到描述</li><li><a href="https://huggingface.co/Ak1104/QA_8k_withChapter_PT">Ak1104/QA_8k_withChapter_PT · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/Ak1104/3_70B">Ak1104/3_70B · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1268937348574482523)** (1 条消息): 

> - `Bellman 模型更新`
> - `微调 Llama 3.1`
> - `模型上传问题`
> - `Q8 版本测试` 


- **随 Llama 3.1 发布 Bellman 微调版**：**Bellman** 的新版本已上传，基于 **Llama-3.1-instruct-8b** 进行微调，专门针对基于瑞典语维基百科数据集的提示词问答。此次更新引入了来自翻译后的 code-feedback 数据集的问答以及若干故事提示词。
   - 虽然该模型在问答方面有所提升，但在*故事生成*方面表现挣扎，不过仍优于其前代版本。
- **测试 Bellman 的 Q8 版本**：Bellman 的 Q8 版本可以在 [此处](https://huggingface.co/spaces/neph1/bellman) 的 CPU 上进行测试，为用户提供了直接探索其能力的机会。鼓励用户尝试该版本以评估微调的效果。
   - 可以在 [此处](https://huggingface.co/neph1/llama-3.1-instruct-bellman-8b-swedish) 查看模型卡片以获取详细见解和规范。
- **模型上传过程中的问题**：在尝试上传基础模型时，出现了多个问题，导致 **TypeError**，提示 'NoneType' 不可迭代。该错误发生在创建模型卡片期间，导致多次尝试将新模型推送到 Hub 失败。
   - 由于该过程已多次停滞，请求澄清此问题是已知故障还是用户操作错误。



**提到的链接**：<a href="https://huggingface.co/neph1/llama-3.1-instruct-bellman-8b-swedish">neph1/llama-3.1-instruct-bellman-8b-swedish · Hugging Face</a>：未找到描述

  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/)** (1 条消息): 

lithofyre: <@1179680593613684819> 你们什么时候能抽空看一下，有时间表吗？
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1268649413648318474)** (1 条消息): 

> - `神经网络模拟`
> - `图像聚类技术`
> - `新的合成数据集`
> - `知识蒸馏趋势`
> - `金融和医疗模型` 


- **神经网络模拟备受关注**：一位成员分享了一个展示神经网络的 [模拟](https://starsnatched.github.io/)，引起了整个社区的兴趣。
   - 该模拟展示了能够增强对神经网络理解和参与度的创新技术。
- **实现准确的图像聚类**：发布了一个关于 [使用图像描述符进行图像聚类](https://www.youtube.com/watch?v=8f4oRcSnfbI) 的视频，强调了组织和分析视觉数据的有效方法。
   - 该成员对这种聚类技术的见解预计将有助于各种 AI 应用。
- **令人兴奋的合成数据集发布**：一位成员发布了一个**巨大的合成数据集**，可通过 [此链接](https://huggingface.co/datasets/tabularisai/oak) 获取，为研究人员拓宽了获取渠道。
   - 该数据集是一个宝贵的资源，支持各种机器学习项目，特别是在表格数据分析方面。
- **富有见地的知识蒸馏趋势**：这篇 [文章](https://www.lightly.ai/post/knowledge-distillation-trends) 总结了**知识蒸馏**的趋势，提供了该领域的最新概览。
   - 该资源提供了关于知识蒸馏如何演变的视角，让社区参与到相关的进展中。
- **具有变革意义的金融和医疗模型发布**：两个新模型 **Palmyra-Med-70b** 和 **Palmyra-Fin-70b** 已亮相，分别专注于医疗保健和金融领域，并拥有令人印象深刻的性能指标。
   - 这些模型可以显著增强诊断、投资和研究，详情可在 [Hugging Face](https://huggingface.co/Writer/Palmyra-Med-70B) 上查看。



**提到的链接**：<a href="https://x.com/samjulien/status/1818652901130354724">来自 Sam Julien (@samjulien) 的推文</a>：🔥 @Get_Writer 刚刚发布了 Palmyra-Med-70b 和 Palmyra-Fin-70b！Palmyra-Med-70b 🔢 提供 8k 和 32k 版本 🚀 MMLU 性能约 86%，优于顶级模型 👨‍⚕️ 用于诊断、制定治疗方案...

  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1268649929346388161)** (227 条消息🔥🔥): 

> - `Learning Resources for Application Development`（应用开发学习资源）
> - `Model Performance Discussions`（模型性能讨论）
> - `Drafting Project Ideas`（项目构思）
> - `Training Autoencoders`（训练 Autoencoders）
> - `Dataset Licensing Inquiries`（数据集许可咨询）


- **针对初学开发者的资源**：一名高中生询问了开始构建应用的经济型资源，特别是使用 Python 以及一些 Java/Swift 知识。
   - 建议包括使用 Google Colab 进行基于云端的机器学习尝试。
- **模型性能指标与问题**：围绕训练好的 LLM 性能问题的讨论，重点关注了 loss 值，特别是提到 0.6 的 loss 是有问题的。
   - 参与者辩论了各种模型训练策略以及在 Transformer 背景下梯度流（gradient flow）的重要性。
- **项目想法与实际应用**：成员们分享了独特项目的想法，并讨论了 Gradio 等框架以及使用 LLM 进行文档交互，强调了提供良好上下文（context）的必要性。
   - 贡献者们还探讨了在包括图像识别在内的各种应用中使用 Autoencoders 的可行性。
- **模型训练与部署中的挑战**：个人在训练 Autoencoders 时遇到问题，寻求性能方面的建议和优化，特别是与预训练模型相关的部分。
   - 讨论中还探索了使用 ONNX 的推理方法，以及模型训练期间妥善管理数据集的重要性。
- **数据集许可讨论**：一位用户寻求关于 IMDB 数据集许可的澄清，并参与了关于联系作者进行确认的讨论。
   - 参与者一致认为，联系通讯作者是澄清许可问题的明智做法。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/nisten/status/1818529201231688139">来自 nisten (@nisten) 的推文</a>：hacked BitNet 用于微调，最终得到了一个 74mb 的文件。仅在 1 个 CPU 核心上就能以每秒 198 个 tokens 的速度流畅对话。简直是巫术。稍后将通过 @skunkworks_ai 开源，base 地址：https://huggi...</li><li><a href="https://tenor.com/view/luffy-one-piece-luffy-smile-smile-gif-23016281">路飞海贼王 GIF - 路飞微笑 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/noaroggendorff/status/1819416977770676622">来自 Noa Roggendorff (@noaroggendorff) 的推文</a>：不错</li><li><a href="https://tenor.com/view/helicopter-baguette-gif-20550621">直升机法棍 GIF - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/datasets/stanfordnlp/imdb/tree/main">stanfordnlp/imdb 在 main 分支</a>：未找到描述</li><li><a href="https://github.com/huggingface/candle">GitHub - huggingface/candle: 适用于 Rust 的极简 ML 框架</a>：适用于 Rust 的极简 ML 框架。通过创建账号为 huggingface/candle 开发做出贡献。</li><li><a href="https://github.com/huggingface/diffusers/pull/9043">sayakpaul 提交的 Flux pipeline · Pull Request #9043 · huggingface/diffusers</a>：我们正在努力将 diffusers 权重上传到相应的 FLUX 仓库。很快就会完成。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1268653606794432542)** (5 条消息): 

> - `Knowledge Distillation`
> - `Local LLM Applications`
> - `Building NLP Applications with Hugging Face`
> - `Evolution of AI Bots`
> - `Retrieval-Augmented Generation` 


- **探索 Knowledge Distillation**：Knowledge distillation 是一种机器学习技术，用于将大型“教师模型”的学习成果转移到较小的“学生模型”中，从而实现**模型压缩 (model compression)**。
   - 它在 **Deep Learning** 中特别有价值，允许紧凑模型有效地模仿复杂模型。
- **本地 LLM 应用的兴起**：一篇文章强调了 **Large Language Models (LLMs)** 在未来十年改变企业的重大作用，重点关注 **Generative AI**。
   - 它讨论了研发方面的进展，这些进展催生了 **Gemini** 和 **Llama 2** 等模型，并利用 **Retrieval Augmented Generation (RAG)** 实现更好的数据交互。
- **使用 Hugging Face 构建 NLP 应用**：一篇关于构建 **NLP applications** 的文章强调了 **Hugging Face** 平台的协作性，突出了其 **open-source libraries**。
   - 该资源旨在为初学者和经验丰富的开发人员提供工具，以有效地增强其 NLP 项目。
- **AI Bots 与新工具**：一篇文章探讨了 **AI bots** 的演变，重点关注 **LLMs** 和 **RAG** 作为 2024 年的关键技术。
   - 它作为一个全面的概述，适合初学者，提供了对 AI bot 开发中的模式和架构设计的见解。
- **AI Bot 开发概述**：文章综合了关于开发智能 AI 应用程序的各种工具的知识，重点关注模式、流水线 (pipelines) 和架构 (architectures)。
   - 它旨在提供通用的技术水平，同时也涉及更深层次的理论方面，使其易于被懂技术的初学者理解。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://blogs.vmware.com/cloud-foundation/2024/03/18/announcing-initial-availability-of-vmware-private-ai-foundation-with-nvidia/">Announcing Initial Availability of VMware Private AI Foundation with NVIDIA</a>：Generative AI (Gen AI) 是未来 5 到 10 年内将改变企业的顶级新兴趋势之一。这场 AI 创新浪潮的核心是 Large Language Models (LLMs)...</li><li><a href="https://www.ibm.com/topics/knowledge-distillation">What is Knowledge distillation? | IBM </a>：Knowledge distillation 是一种机器学习技术，用于将大型预训练“教师模型”的学习成果转移到较小的“学生模型”中。</li><li><a href="https://medium.com/@qdrddr/evolution-of-the-ai-bots-harnessing-the-power-of-agents-rag-and-llm-models-4cd4927b84f8">Evolution of the AI Bots: Harnessing the Power of Agents, RAG, and LLM Models</a>：构建关于 AI bot 开发工具的知识体系，以及对方法、架构和设计的高级概述。</li><li><a href="https://www.analyticsvidhya.com/blog/2024/06/building-nlp-applications-with-hugging-face/">How to Build NLP Applications with Hugging Face?</a>：探索如何使用 Hugging Face 构建 NLP 应用程序，利用模型、数据集和开源库来提升您的 ML 项目。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1268655352715677831)** (8 messages🔥): 

> - `AI + i Podcast Launch`
> - `AI Journey Updates`
> - `Simulations and Neural Networks`
> - `Uber ETA Prediction Video` 


- **AI + i 播客发布公告**：一位成员宣布启动 [AI + i 播客](https://youtube.com/@aiplusi)，讨论领先的基础模型和开源模型，并正在征求话题建议。
   - 另一位成员幽默地询问播客是否涉及与模型的实际对话。
- **2024 年 7 月 AI 历程更新**：一位成员分享了他们对 2024 年 7 月 [AI 历程](https://www.tiktok.com/t/ZTN4Fev1a/)更新的兴奋之情，并用庆祝表情表达了热情。
   - 这一公告标志着与 AI 相关的个人里程碑或项目完成。
- **关于模拟机制的说明**：围绕模拟机制展开了讨论，特别是其作为学习组件的 'Reward'（奖励）和 'Punish'（惩罚）函数。
   - 一位成员指出 'Reward' 旨在释放多巴胺，而 'Punish' 释放血清素，但对模拟的有效性表示困惑。
- **关于 Uber ETA 预测的咨询**：一位成员询问了 Uber 预测预计到达时间 (ETA) 的方法，并引用了一个解释整个 ML 系统的[全动画视频](https://youtu.be/fnKrReaqQgc)。
   - 该视频涉及多个话题，包括经典路由引擎、ML 的必要性以及增量模型改进。



**Link mentioned**: <a href="https://www.tiktok.com/t/ZTN4Fev1a/">TikTok - Make Your Day</a>: 未找到描述

  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1268816923680964640)** (6 messages): 

> - `Organizing study sessions`
> - `Focus topics for learning`
> - `Hackathons and competitions`
> - `Skill gaps in projects`
> - `Balance between courses and projects` 


- **在服务器上组织学习会议**：成员们讨论了直接在服务器上组织学习和分享会议的兴奋点，这将允许任何人加入。
   - 然而，有人担心话题过多可能会导致混乱和参与度下降。
- **需要一个主要焦点**：一位成员强调了为会议选择一个主要焦点的重要性，例如特定的课程或书籍，以避免结构混乱。
   - 灵活性固然重要，但共同的目标有助于保持参与者的积极性。
- **用于学习的协作项目**：另一位成员建议组队参加黑客松或 Kaggle 竞赛，以此作为促进共同学习和参与的一种方式。
   - 这可以创造一个共同目标，激励成员相互协作和学习。
- **技能差异带来的挑战**：有人担心小组成员之间显著的技能差距可能会导致工作分配不均和挫败感。
   - 竞赛可能会因为时间限制而加剧这一问题，因此需要一种对所有水平都更公平的学习方法。
- **从课程或书籍开始**：建议初学者在开展项目之前，先通过学习课程或阅读书籍作为基础步骤。
   - 这种方法可以帮助确保所有成员在应对更复杂的协作挑战之前都有一个基准认知。


  

---

### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1268802609150427187)** (1 messages): 

> - `Running Flux pipelines`
> - `Limited resources for Diffusers`
> - `Pull request for Diffusers` 


- **在有限 VRAM 上运行 Flux 的技巧**：一位成员分享了一份[资源指南](https://gist.github.com/sayakpaul/b664605caf0aa3bf8585ab109dd5ac9c)，介绍如何在有限资源下（甚至是 **24GB** 显卡）通过 **Diffusers** 运行 Black Forest Lab 的 **Flux**。
   - 该指南专门解决了在受限的 VRAM 条件下使用 **Flux** 流水线时遇到的问题。
- **Diffusers 改进的新 Pull Request**：一位成员发布了一个 [pull request](https://github.com/huggingface/diffusers/pull/9049)，其中包含一项修复，使 **Flux** 流水线能够在 **24GB** VRAM 下有效运行。
   - 该 PR 重点修改了 `encode_prompt` 函数，以提升硬件资源有限用户的性能。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://gist.github.com/sayakpaul/b664605caf0aa3bf8585ab109dd5ac9c">此文档列出了展示如何在有限资源下通过 Diffusers 运行 Black Forest Lab 的 Flux 的资源。</a>: This document enlists resources that show how to run Black Forest Lab&#39;s Flux with Diffusers under limited resources.  - run_flux_with_limited_resources.md</li><li><a href="https://github.com/huggingface/diffusers/pull/9049">[Flux] fix: encode_prompt when called separately. by sayakpaul · Pull Request #9049 · huggingface/diffusers</a>: 这个 PR 做了什么？此 PR 允许 Flux 流水线在低于 24GB 的 VRAM 下运行。代码：from diffusers import FluxPipeline, AutoencoderKL from diffusers.image_processor import VaeImageProcessor fr...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1268738469581160551)** (1 messages): 

> - `LoRA Finetuning`
> - `Stable Diffusion models`
> - `Training techniques` 


- **在 Stable Diffusion 上使用 LoRA 进行微调**：一位成员分享了在 **runwayml/stable-diffusion-v1-5** 上使用 [LoRA](https://hf.co/papers/2106.09685) 进行模型微调的指南，强调了其速度和内存效率。
   - *LoRA 通过插入较少的新权重来减少可训练参数*，使模型更易于存储和共享。
- **LoRA 训练脚本**：讨论中提到了 [train_text_to_image_lora.py](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py) 脚本，作为在个人用例中熟悉 LoRA 的资源。
   - 建议成员先从源码安装 diffusers 库，以确保脚本顺利运行。
- **关于 SDXL-base-1.0 和 LoRA 的问题**：一位成员询问 **SDXL-base-1.0** 是否像 **runwayv1.5** 一样支持 LoRA 微调。
   - 这体现了对新训练方法技术兼容性的好奇。



**提及的链接**: <a href="https://huggingface.co/docs/diffusers/en/training/lora">LoRA</a>: 未找到描述

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1268986123879714889)** (1 messages): 

> - `Error Resolution`
> - `Troubleshooting Solutions` 


- **寻求持久错误的帮助**：一位成员在寻求他人帮助时表达了对持续出现的错误的沮丧，称 *“似乎没有任何办法奏效。”*
   - 这一求助请求反映了社区内对常见问题有效解决方案的广泛关注。
- **探索常见错误的解决方案**：成员们讨论了各种故障排除方法，但指出其中许多方法对于编程过程中遇到的常见错误并未产生成功的结果。
   - 大家一致认为，*分享有效的解决方案*有助于在未来的讨论中简化错误解决流程。


  

---

### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1268780913240576172)** (7 messages): 

> - `Flux Architecture` (Flux 架构)
> - `Fine-Tuning Flux` (微调 Flux)
> - `DreamBooth`
> - `GB200 Accelerator` (GB200 加速器)


- **探索 Flux 架构**：一位成员询问了 **Flux** 的架构，并引导其他人查看 [此 GitHub 链接](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py#L145) 以获取更多细节。
   - 该 GitHub 仓库将 **Diffusers** 描述为用于 **PyTorch** 和 **FLAX** 中图像和音频生成的先进扩散模型。
- **关于微调 Flux 的讨论**：另一位成员询问是否有人正在微调 **Flux**，并分享了一个与 **Diffusers** 仓库相关的 [Pull Request](https://github.com/huggingface/diffusers/pull/9057) 链接。
   - 他们鼓励其他人尝试 **DreamBooth**，这似乎是与该更新相关的一个功能。
- **对 GB200 加速器的需求**：一位成员强调微调 **Flux** 需要 **GB200** 加速器，指出了某些技术要求。
   - 另一位成员幽默地评论说他们可以使用它，但在这个过程中并不一定需要它。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/huggingface/diffusers/pull/9057.">Build software better, together</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py#L145">diffusers/src/diffusers/pipelines/flux/pipeline_flux.py at main · huggingface/diffusers</a>：🤗 Diffusers：用于 PyTorch 和 FLAX 中图像和音频生成的先进扩散模型。- huggingface/diffusers
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1268663522389131380)** (1 messages): 

> - `Perplexity Pro for Uber One members` (为 Uber One 会员提供的 Perplexity Pro)
> - `Benefits of Uber One membership` (Uber One 会员权益)


- **Uber One 会员可获得 1 年 Perplexity Pro**：美国和加拿大符合条件的 Uber One 会员现在可以兑换价值 **$200** 的免费一年 [Perplexity Pro](https://pplx.ai/uber-one)。
   - 此项优惠旨在增强通勤或在家时的信息收集能力，为用户提供对 Perplexity “答案引擎”的无限访问权限。
- **Uber One 通过新权益提升效率**：从今天开始，Uber One 会员可以解锁提升信息收集和研究效率的权益。
   - **Perplexity Pro** 允许会员提出诸如 *“谁发明了汉堡？”* 之类的问题，并获得引人入胜的对话式回答。



**提到的链接**：<a href="https://pplx.ai/uber-one">符合条件的 Uber One 会员现在可以解锁一整年免费的 Perplexity Pro&nbsp;</a>：Uber One 会员现在可以通过 Pro Search 等功能节省更多时间。

  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1268645761508315146)** (237 条消息🔥🔥): 

> - `Uber One 促销活动`
> - `Perplexity 用户体验`
> - `ML 模型对比`
> - `Perplexity Pro 订阅` 


- **关于 Uber One 促销的困惑**：用户不确定提供一年 Perplexity Pro 的 Uber One 促销活动是否同时适用于新老订阅者，一些用户报告了在兑换代码时遇到的问题。
   - 在注册 Uber One 后，关于资格限制以及接收促销邮件出错的担忧在用户中非常普遍。
- **用户对 Perplexity 的排名和偏好**：几位用户分享了他们在研究中偏好的模型，对比了 Sonnet 3.5 和 ChatGPT 4.0，对其性能和易用性看法不一。
   - 讨论内容包括如何使用特定的 site 操作符来获取更聚焦的结果，以提高技术文档搜索的准确性。
- **Perplexity 的使用体验**：用户分享了 Perplexity 的多种使用场景，从日常闲聊、故事创作到撰写有来源支持的可信博客文章。
   - 多位社区成员强调，Perplexity 被视为生成准确信息的可靠工具。
- **关于 App 功能的技术问题**：许多用户抱怨移动端体验欠佳，存在多种影响易用性的 Bug，尤其是在 Android 设备上。
   - 成员们强调了一些问题，例如离开输入框时输入的文本会丢失，以及使用平板电脑键盘发送消息困难。
- **Perplexity 增长的用户群**：有人对 Perplexity 报道的 Pro 用户增长提出了疑问，推测用户数已接近 10 万。
   - 用户对选择 Perplexity 的原因表示好奇，并质疑与其他 AI 工具相比，它是否真的能满足他们的需求。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://character.ai/chat/KnHvvSCjV02eDMDXFjurGkCFkl8L71XTEryNiK8hXlc>)!">character.ai | 为你日常每一刻打造的个性化 AI</a>：遇见栩栩如生的 AI。随时随地与任何人聊天。体验超智能聊天机器人的力量，它们能倾听、理解并记住你。</li><li><a href="https://www.perplexity.ai/hub/blog/eligible-uber-one-members-can-now-unlock-a-complimentary-full-year-of-perplexity-pro">符合条件的 Uber One 会员现在可以解锁免费的一整年 Perplexity Pro</a>：Uber One 会员现在可以通过 Pro Search 等功能节省更多时间。</li><li><a href="https://tenor.com/view/%E7%9A%849-gif-27299608">的9 GIF - 的9 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://genai.works/">Generative AI</a>：生成式 AI
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1268817170385604688)** (6 条消息): 

> - `Massive Mathematical Breakthrough` (重大数学突破)
> - `Digital Organization for Productivity` (提升生产力的数字化整理)
> - `Medallion Fund` (大奖章基金)
> - `Hybrid Human-Llama Antibody` (人类-羊驼杂交抗体)
> - `Ducks Classification and Habitat` (鸭子的分类与栖息地)


- **重大数学突破揭晓**：最近的一项发现凸显了一项 **Massive Mathematical Breakthrough**（重大数学突破），可能重塑我们对 **complex equations**（复杂方程）的理解，正如 [YouTube 视频](https://www.youtube.com/embed/FOU-n9Xwp4U)中所分享的那样。
   - 关于这一发现的细节仍处于严格保密状态，引发了对其在各个领域影响的广泛讨论。
- **数字化整理影响职场效率**：Adobe Acrobat 的一项调查显示，近 **3/4 的员工** 认为 **数字化整理不善 (poor digital organization)** 负面影响了他们的效率，其中 **30% 的 Gen Z** 考虑因此离职，正如关于 [Digital Organization for Productivity](https://www.perplexity.ai/page/digital-organization-for-produ-jafDPUtDRJW3ZVYbh4gL5A) 的文章所述。
   - 这凸显了现代商业环境中 **数字化整理** 与员工满意度之间的关键联系。
- **大奖章基金 (Medallion Fund) 的卓越表现**：由 Renaissance Technologies 在 Jim Simons 领导下管理的 **Medallion Fund** 自 1988 年成立以来，取得了令人印象深刻的平均年化收益率：扣除费用前为 **66%**，扣除费用后为 **39%**，详见 [The Enigmatic Medallion Fund](https://www.perplexity.ai/page/the-enigmatic-medallion-fund-xkICvfd7T7.WsILxst6bpg)。
   - 据报道，其秘密策略超越了 **Warren Buffett** 和 **George Soros** 等著名投资者的表现。
- **杂交抗体在对抗 HIV 方面展现前景**：研究人员通过将羊驼纳米抗体 (llama nanobodies) 与人类抗体融合，开发出一种可以中和 **95% 以上** HIV-1 毒株的杂交抗体，据 [Georgia State University](https://www.perplexity.ai/page/hybrid-human-llama-antibody-fi-UCs.nTMFTu6QaRoOTXp0gA) 报道。
   - 这些 **nanobodies**（纳米抗体）由于体积小，比传统的人类抗体能更有效地穿透病毒的防御。
- **了解鸭子及其栖息地**：鸭子被归类为 **Anatidae**（鸭科），其特征是体型较小且喙部扁平，根据摄食习性分为不同的群体：**dabbling**（钻水鸭）、**diving**（潜水鸭）和 **perching**（栖木鸭），如关于 [Ducks](https://www.perplexity.ai/search/what-is-a-duck-X.h_eLguRZuCiSCW65178A) 的文章所述。
   - 这些独特的分类有助于研究它们在不同环境中的行为和栖息地。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.perplexity.ai/search/what-is-a-duck-X.h_eLguRZuCiSCW65178A">what is a duck</a>: 鸭子是属于 Anatidae（鸭科）的水禽，该科还包括天鹅和鹅。它们的特征是体型相对较小，脖子短...</li><li><a href="https://www.perplexity.ai/search/create-an-agile-assessment-of-PGV9oD31QE6RdEtPWT6uWA#0">Perplexity</a>: Perplexity 是一款免费的 AI 驱动回答引擎，可针对任何问题提供准确、可靠且实时的回答。</li><li><a href="https://www.perplexity.ai/page/hybrid-human-llama-antibody-fi-UCs.nTMFTu6QaRoOTXp0gA">Hybrid Human-Llama Antibody Fights HIV</a>: 研究人员通过将羊驼来源的纳米抗体与人类抗体结合，设计出一种对抗 HIV 的强大新武器，创造出一种可以...</li><li><a href="https://www.perplexity.ai/page/the-enigmatic-medallion-fund-xkICvfd7T7.WsILxst6bpg">The Enigmatic Medallion Fund</a>: 由 Renaissance Technologies 管理的大奖章基金是金融史上最成功且最神秘的对冲基金之一。由...创立。</li><li><a href="https://www.perplexity.ai/page/digital-organization-for-produ-jafDPUtDRJW3ZVYbh4gL5A">Digital Organization for Productivity</a>: 根据 Adobe Acrobat 的一项调查，近 3/4 的员工报告称，数字化整理不善会干扰他们有效工作的能力...
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1268991873742147727)** (1 条消息): 

> - `Vulkan llama.cpp engine`
> - `Gemma 2 2B model`
> - `Flash Attention KV Cache configuration` 


- **🌋🔥 Vulkan llama.cpp 引擎发布！**：全新的 **Vulkan llama.cpp 引擎** 取代了之前的 OpenCL 引擎，为 **AMD**、**Intel** 和 **NVIDIA** 独立 GPU 开启了 GPU 加速。
   - 此更新是新版本 **0.2.31** 的一部分，可通过 [应用内更新](https://lmstudio.ai) 或在官网上获取。
- **🤖 新增对 Gemma 2 2B 模型的支持**：版本 **0.2.31** 引入了对 Google 新推出的 **Gemma 2 2B 模型** 的支持，可在此处 [下载](https://model.lmstudio.ai/download/lmstudio-community/gemma-2-2b-it-GGUF)。
   - 建议用户从 **lmstudio-community 页面** 下载此模型以获得增强的功能。
- **🛠️ 高级 KV Cache 数据量化功能**：此次更新允许用户在启用 **Flash Attention** 时配置 **KV Cache 数据量化**，这有助于降低大型模型的内存需求。
   - 然而，需要注意的是 *许多模型并不支持 Flash Attention*，因此这目前是一项实验性功能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lmstudio.ai">👾 LM Studio - 发现并运行本地 LLMs</a>：查找、下载并实验本地 LLMs</li><li><a href="https://model.lmstudio.ai/download/lmstudio-community/gemma-2-2b-it-GGUF)">在 LM Studio 中下载并运行 lmstudio-community/gemma-2-2b-it-GGUF)</a>：在你的 LM Studio 中本地使用 lmstudio-community/gemma-2-2b-it-GGUF)
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1268644460846252156)** (163 条消息🔥🔥): 

> - `GPU Performance and Compatibility`
> - `Model Training and Inference`
> - `LM Studio Features and Updates`
> - `Vulkan vs ROCm on AMD GPUs`
> - `User Experiences with LLMs` 


- **GPU 性能见解**：用户报告在 **RX6700XT** 上通过 Vulkan 支持运行模型取得了成功，速度约为 **30 tokens/second**，突显了该配置的有效性。
   - 一位用户指出，他们使用 Llama 3-8B-16K-Q6_K-GGUF 模型达到了 **40 tokens/second**，展示了令人印象深刻的性能。
- **模型训练与推理中的挑战**：几位用户讨论了训练 LSTM 模型的问题，表示尽管训练输出成功，但在实现正确的推理结果方面存在困难。
   - 一位使用 **32M 参数 LSTM** 模型的用户指出，在正确输入数据方面面临挑战，导致输出不连贯。
- **LM Studio 的功能请求**：用户表达了希望能够将文档拖放到 LM Studio 中的愿望，并希望这一功能能尽快实现。
   - 社区讨论了添加 RAG 等功能以及改进对各种文档格式的支持，以增强易用性。
- **量化模型的性能**：关于不同量化方法的讨论显示，Flash Attention 的表现不一致，在 **qx_0** 设置下运行良好，但在 **qx_1** 下则不行。
   - 用户发现特定的量化设置（如 **q4_1**、**q5_0** 和 **q5_1**）会影响性能，引发了进一步的探索。
- **关于 Vulkan 与 ROCm 的辩论**：参与者讨论了 **Vulkan** 和 **ROCm** 在 AMD GPU 上的性能，Vulkan 被视为一种备选方案。
   - 用户分享了有关环境搭建的经验，权衡了 CUDA 与 Vulkan 的利弊。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1268669556000555008)** (76 messages🔥🔥): 

> - `学习 Proxmox`
> - `Proxmox 中的 GPU 驱动`
> - `LM Studio 的兼容性问题`
> - `MacBook Pro 上的 ML Studio 设置`
> - `为本地 LLM 选择 GPU` 


- **高效学习 Proxmox**：一位用户建议在迁移到裸机 Proxmox 之前，先在 VirtualBox 中学习 Proxmox，以便同时利用 LLM 知识。
   - 另一位成员分享了他们在学习 Proxmox 时使用 RAG 文档的持续策略。
- **排除 GPU 驱动故障**：一位用户在 Proxmox VM 中遇到 NVIDIA 驱动问题，特别是安装后 `nvidia-persistenced` 服务报错。
   - 成员们建议使用 `lspci` 检查设备检测情况，并建议确保已安装必要的驱动程序包。
- **LM Studio 的兼容性问题**：一位用户在 Intel Xeon E5-1650 上遇到 LM Studio 兼容性问题，原因是缺乏 AVX2 指令集支持。
   - 社区建议使用仅限 AVX 的扩展，并建议如果可能的话考虑升级 CPU。
- **优化 MacBook Pro 上的 ML Studio 设置**：一位拥有 M2 Max 和 96GB RAM 的 MacBook Pro 用户寻求服务器模型设置建议，以优化 LM Studio 的性能。
   - 社区指出，调整 VRAM 分配可以提升性能，但建议注意稳定性。
- **为本地 LLM 选择合适的 GPU**：一位用户咨询为双 Xeon 服务器配置购买 NVIDIA Quadro GPU 的建议，在 A2000 到 A6000 之间权衡。
   - 另一位成员建议初始配置考虑单张 RTX 3090，理由是它在本地 LLM 工作负载中表现更好。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF">lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/186phti/m1m2m3_increase_vram_allocation_with_sudo_sysctl/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://tenor.com/view/spongebob-patrick-star-shocked-loop-surprised-gif-16603980">海绵宝宝派大星 GIF - 海绵宝宝派大星震惊 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://llm.extractum.io/list/">所有大型语言模型</a>：大型和小型语言模型（开源 LLM 和 SLM）的精选列表。包含具有动态排序和过滤功能的所有大型语言模型。</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md">configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs</a>：LM Studio JSON 配置文件格式和示例配置文件集合。 - lmstudio-ai/configs</li><li><a href="https://www.reddit.com/r/LocalLLaMA">Reddit - 深入探索</a>：未找到描述
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1268710672233922631)** (10 messages🔥): 

> - `NVIDIA GPU 指令周期`
> - `准确率得分波动` 


- **收集关于 NVIDIA GPU 指令周期的资源**：一位成员请求关于 **NVIDIA GPU 指令周期** 的推荐资源，探索公开披露和实验性发现。
   - 他们分享了一篇 [研究论文](https://conferences.computer.org/cpsiot/pdfs/RTAS2020-4uXAu5nqG7QNiz5wFYyfj6/549900a210/549900a210.pdf) 和另一篇专注于每条指令时钟周期的 [微架构研究](https://arxiv.org/abs/2208.11174)。
- **准确率得分急剧飙升**：一位成员观察到在 **epoch 8** 时，准确率得分显著**飙升**，导致对运行状态产生困惑。
   - 他们质疑这种 *准确率得分的波动* 是否典型，表现出对性能稳定性的惊讶和担忧。



**提及的链接**：<a href="https://arxiv.org/abs/2208.11174">通过微基准测试和指令级分析揭秘 NVIDIA Ampere 架构</a>：图形处理器（GPU）现在被认为是加速 AI、数据分析和 HPC 等通用工作负载的主导硬件。在过去的十年中，研究人员一直专注于...

  

---

### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1268718331687342092)** (19 messages🔥): 

> - `Triton 中的 GROUP_SIZE_M`
> - `Triton 矩阵乘法教程`
> - `理解 Triton 内部机制`
> - `对 Triton 博客文章的反馈` 


- **理解 GROUP_SIZE_M 的作用**：在关于 Triton 分块 matmul 教程的讨论中，明确了 `GROUP_SIZE_M` 用于控制 block 的处理顺序，这可以提高 L2 cache 命中率。
   - 成员们强调了 `GROUP_SIZE_M` 与 `BLOCK_SIZE_{M,N}` 之间的概念区别，教程中的图解帮助了他们的理解。
- **Triton 矩阵乘法教程亮点**：Triton 教程涵盖了高性能 FP16 矩阵乘法，强调了 block 级操作、多维指针运算以及 L2 cache 优化。
   - 参与者注意到，在实际应用中重新排序计算顺序对提升性能非常有效。
- **关于 Triton 内部机制的资源**：一位成员表示难以找到关于 Triton 编译过程和内部机制的详尽文档，并提到自 2019 年原始论文以来已有很大变化。
   - 他们分享了一篇自己撰写的[博客文章](https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/)，旨在汇集资源并寻求社区反馈。
- **请求关于 ML Compilers 的 Triton 讲座**：有人提议举办一场专注于 Triton 内部机制或 ML Compilers 的讲座，以增强社区对该主题的理解。
   - 该询问旨在吸引专家的兴趣，以提供关于 Triton 功能的宝贵见解。



**提到的链接**：<a href="https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#l2-cache-optimizations">Matrix Multiplication &mdash; Triton  documentation</a>：未找到描述

  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1268668370471878878)** (23 messages🔥): 

> - `模型过拟合`
> - `CUDA 扩展与 BitBLAS`
> - `对 Bitnet 的兴趣`
> - `PR 评审`
> - `主题模型分析` 


- **成员讨论过拟合指标**：一位成员询问其他人是否检查过拟合问题，并建议 **MMLU** 是验证此问题的良好指标。
   - 这引发了小组内关于模型评估实践的讨论。
- **对 C++ 扩展开发的兴奋**：一位成员幽默地庆祝获得了 **3090 GPU**，表达了使用 C++ 扩展进行构建的热情。
   - 其他人也加入讨论，探讨了更多 **CUDA extensions** 的实用性，其中一位成员强调 [BitBLAS](https://github.com/microsoft/BitBLAS) 是一个潜在的补充。
- **重新关注 Bitnet 的新进展**：成员们表达了重新审视 **Bitnet** 的兴趣，提到了社区成员分享的新发现和进展。
   - 有人强调 *Bitnet* 已被 hack 用于微调，产生了一个仅 **74MB 的模型**，在单核 CPU 上运行效率很高。
- **开发进度与 PR 评审**：关于合并 **PR #468** 的讨论正在进行，一位成员提到他们遇到了与 tensor subclasses 相关的问题。
   - 一旦这些问题解决，他们预计很快会进行重新评审，旨在实现高效协作。
- **仓库随时间增长的分析**：一位成员正在探索库中 module 的 **tree representations**（树状表示），以更好地理解其结构。
   - 他们计划从 commit 历史中生成主题模型的 **timeseries**（时间序列），以可视化仓库的增长。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/nisten/status/1818529201231688139">nisten (@nisten) 的推文</a>：hack 了 bitnet 用于微调，最终得到了一个 74mb 的文件。它在仅 1 个 CPU 核心上以每秒 198 个 tokens 的速度运行。简直是巫术。稍后将通过 @skunkworks_ai 开源，base 地址：https://huggi...</li><li><a href="https://github.com/microsoft/BitBLAS">GitHub - microsoft/BitBLAS: BitBLAS 是一个支持混合精度矩阵乘法的库，特别适用于量化 LLM 部署。</a>：BitBLAS 是一个支持混合精度矩阵乘法的库，特别适用于量化 LLM 部署。 - microsoft/BitBLAS</li><li><a href="https://github.com/mobiusml/hqq/commit/62494497a13174d7a95d3f82c8f9094a5acd3056">为 4-bit/2-bit 添加 bitblas 后端 · mobiusml/hqq@6249449</a>：未找到描述</li><li><a href="https://github.com/pytorch/ao/pull/468">vayuda 提交的 Intx Quantization Tensor Class · Pull Request #468 · pytorch/ao</a>：履行 #439 基准测试结果的 PR：非 2 的倍数的 dtypes 性能明显较差，但在没有自定义 kernels 的情况下这是预料之中的。
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 条消息): 

marksaroufim: https://techcrunch.com/2024/08/02/character-ai-ceo-noam-shazeer-returns-to-google/
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1268644994143490078)** (178 条消息🔥🔥): 

> - `Llama 3 实现`
> - `KV Cache 问题`
> - `AI 领域的并购式招聘 (Acquihires)`
> - `Tensor 操作中的随机性`
> - `RDNA 与 CDNA 的性能对比` 


- **Llama 3 实现进展**：一位用户修改了 `llm.c` 的 Python 脚本以支持 **Llama 3**，注意到第一次样本生成是一致的，但在第二次运行期间由于尚未实现 KV cache 导致使用了不同的 kernel，从而产生了偏差。
   - 他们计划重构代码以减少分支，并整合 attention 的前向传播，从而实现更简洁的实现。
- **发现 KV Cache 问题**：实现者指出，输出的不一致是由于 KV cache 状态逻辑未激活，导致第二次运行期间 logits 出现数值偏差。
   - 用户确认所有初始 tensor 和 logits 在早期阶段是等效的，这表明实现 KV cache 可以确保生成相同的样本。
- **AI 行业的并购式招聘 (Acquihires)**：讨论围绕 Character AI 和 Inflection AI 等多家公司进行的并购式招聘展开，这标志着行业内一种有潜力的公司被吸收的趋势。
   - 讨论中对生态系统的潜在影响表示了担忧，认为虽然更多的参与者可能会增强竞争，但也引发了关于 AI 开发中编码价值与概念性思考价值的疑问。
- **Tensor 操作中的随机性**：注意到操作中不同的 tensor 形状可能会导致调用不同的 kernel，从而产生不同的数值输出。
   - 这引发了关于传入自定义随机数生成器的建议，以确保跨 tensor 操作的行为一致。
- **RDNA 与 CDNA 的性能对比**：在 RDNA 和 CDNA 设置之间观察到了性能差异，尽管硬件架构不同，但后者在特定 AI 模型中显示出更好的验证损失（validation loss）。
   - 用户讨论认为 RDNA 通过向量单元使用微代码矩阵操作，而 CDNA 具有专用的矩阵硬件，这可能会影响性能结果。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/issues/727.">Issues · karpathy/llm.c</a>: 在简单、原始的 C/CUDA 中进行 LLM 训练。通过在 GitHub 上创建账号为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/pull/726/commits/b499ff35fde826b999f271da0a1bccaa7e6e99a4">Llama tmp by gordicaleksa · Pull Request #726 · karpathy/llm.c</a>: 临时，内部使用</li><li><a href="https://github.com/karpathy/llm.c/pull/725">Add LLaMA 3 Python support by gordicaleksa · Pull Request #725 · karpathy/llm.c</a>: 进行中 (WIP)。</li><li><a href="https://github.com/pytorch/pytorch/issues/39716">Do not modify global random state · Issue #39716 · pytorch/pytorch</a>: 🚀 特性：目前，实现可复现性的推荐方法是设置全局随机种子。我建议让所有需要随机源的函数接受一个本地的.....</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/chat_templates.py#L1456">unsloth/unsloth/chat_templates.py at main · unslothai/unsloth</a>: 微调 Llama 3.1, Mistral, Phi &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://www.picoquant.com/products/category/tcspc-and-time-tagging-modules/hydraharp-400-multichannel-picosecond-event-timer-tcspc-module">
     
        HydraHarp 400 - 多通道皮秒事件计时器和 TCSPC 模块
    
     | PicoQuant</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/1268757532575535177)** (7 messages): 

> - `GPU Compute Learning`
> - `PyTorch Conference Details`
> - `Event Invites Expectation` 


- **GPU Compute 学习热情**：一位成员表达了对 **GPU Compute** 的热爱，强调了他们在周末投入的大量学习努力，并有兴趣在即将举行的活动中与志同道合的人建立联系。
   - _“我发现这个领域深得我心”_ 强调了对该主题的个人情感连接和投入。
- **旅行计划与活动链接**：另一位成员确认了从 **Boston** 出发，使用 **Jetson** 和 **AWS** 的旅行计划，并询问了 [PyTorch Conference](https://events.linuxfoundation.org/pytorch-conference/?__hstc=132719121.f80c3baeafbc1ed5053566b9f8315dd4.1722541248418.1722550924621.1722569593843.3&__hssc=132719121.1.1722569593843&__hsfp=1528460664&_gl=1*1wms76u*_gcl_au*OTI0NDYwMjQ0LjE3MjI1NTA5MjQ.*_ga*OTU2MjQzNzAyLjE3MjI1NTA5MjQ.*_ga_VWZ4V8CGRF*MTcyMjU2OTU5My4yLjAuMTcyMjU2OTU5My4wLjAuMA.. ) 的链接。
   - 分享的链接提供了关于特邀演讲者和活动详情的信息，鼓励更多人参与。
- **邀请函即将发出**：一位成员提到，由于响应非常热烈，预计该活动的邀请函将在本月底发出。
   - 他们提到 **响应非常热烈**，这意味着并非所有人都能获得名额，这在参与者中营造了期待感。



**Link mentioned**: <a href="https://events.linuxfoundation.org/pytorch-conference/?__hstc=132719121.f80c3baeafbc1ed5053566b9f8315dd4.1722541248418.1722550924621.1722569593843.3&__hssc=132719121.1.1722569593843&__hsfp=1528460664&_gl=1*1wms76u*_gcl_au*OTI0NDYwMjQ0LjE3MjI1NTA5MjQ.*_ga*OTU2MjQzNzAyLjE3MjI1NTA5MjQ.*_ga_VWZ4V8CGRF*MTcyMjU2OTU5My4yLjAuMTcyMjU2OTU5My4wLjAuMA..">PyTorch Conference | LF Events</a>: 与顶尖研究人员、开发人员和学者一起深入探讨 PyTorch，这一前沿的开源机器学习框架。

  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1268656787972948028)** (145 messages🔥🔥): 

> - `MoMa architecture`
> - `BitNet fine-tuning`
> - `Character.ai acquisition`
> - `DeepSeek API improvements`
> - `LlamaCoder app` 


- **MoMa 架构增强混合模态语言建模**：Meta 推出了 [MoMa](https://arxiv.org/pdf/2407.21770)，这是一种稀疏早期融合架构，通过使用 **Mixture-of-Expert** 框架提高了预训练效率。
   - 该架构改进了交错混合模态 token 序列的处理，使其成为多模态 AI 的重大进展。
- **BitNet 微调取得快速成效**：一位用户报告称，微调 **BitNet** 产生了一个 **74MB** 的文件，在单个 CPU 核心上每秒可处理 **198 tokens**，展示了令人印象深刻的效率。
   - 该技术正由该用户以 [Biggie-SmoLlm](https://huggingface.co/nisten/Biggie-SmoLlm-0.15B-Base) 的名称开源。
- **Character.ai 被收购后的策略转变**：Character.ai 的联合创始人已加入 Google，导致其产品转向使用 Llama 3.1 等开源模型，引发了关于行业人才流动的讨论。
   - 这一举动引发了关于初创公司在面对大型科技公司收购时的生存能力的对话，对于这是否有益于创新存在不同看法。
- **DeepSeek API 引入磁盘上下文缓存 (Context Caching)**：DeepSeek API 推出了新的上下文缓存功能，可降低高达 **90%** 的 API 成本，并显著降低首个 token 延迟。
   - 这一改进通过缓存频繁引用的上下文，支持多轮对话、数据和代码分析，从而提升整体性能。
- **LlamaCoder 实现高效的 React 应用生成**：LlamaCoder 是一款开源应用程序，使用 Llama 3.1 快速生成完整的 React 应用程序和组件。
   - 该工具为用户提供了一种将想法转化为实际应用的免费方式，促进了 Web 编程的快速开发。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://x.com/nisten/status/1818529201231688139">nisten (@nisten) 的推文</a>：修改了 BitNet 用于微调，最后得到了一个 74MB 的文件。仅在 1 个 CPU 核心上就能以每秒 198 个 token 的速度流畅对话。简直是巫术。稍后将通过 @skunkworks_ai 开源，基础模型在这里：https://huggi...</li><li><a href="https://x.com/victorialinml/status/1819037433251721304?s=46">Victoria X Lin (@VictoriaLinML) 的推文</a>：1/n 介绍 MoMa 🖼，我们用于混合模态语言建模的新型稀疏早期融合架构，显著提升了预训练效率 🚀 (https://arxiv.org/pdf/2407.21770)。MoMa 采用...</li><li><a href="https://x.com/atroyn/status/1819396701217870102">anton (𝔴𝔞𝔯𝔱𝔦𝔪𝔢) (@atroyn) 的推文</a>：1. 流行词是思维杀手。你必须清空脑子里所有的流行词。任何新技术的诱惑在于尽可能多地使用现有概念作为支柱，但这会扼杀创造力...</li><li><a href="https://x.com/_xjdr/status/1819475619224473628">xjdr (@_xjdr) 的推文</a>：Google 拥有：- AlphaZero - 非常擅长搜索和索引 - Gemini 通过 1M 上下文长度在 LMSYS 上刷榜 - 世界上一些最顶尖的研究员和工程师（现在再次包括 Noam 和...</li><li><a href="https://x.com/_xjdr/status/1819435049655455987">xjdr (@_xjdr) 的推文</a>：- GDM 现在在 AGI 竞赛中领先 - Llama 3.1 改变了一切，而 Llama 4 是目前世界上潜在影响最大的模型（除非内部已经实现了 AGI 并且...</li><li><a href="https://github.blog/news-insights/product-news/introducing-github-models/">介绍 GitHub Models：在 GitHub 上构建的新一代 AI 工程师</a>：我们通过 GitHub Models 助力 AI 工程师的崛起——将行业领先的大型和小型语言模型的能力直接带给 GitHub 上超过 1 亿的用户。</li><li><a href="https://x.com/pitdesi/status/1819447414841126997?s=46">Sheel Mohnot (@pitdesi) 的推文</a>：Character 正在与 Google 进行一项人才收购（acquihire）授权交易，类似于 Inflection > Microsoft 和 Adept > Amazon... 联合创始人加入大公司，公司在新领导下继续生存。Character 的投资者获得了 2.5 倍回报，...</li><li><a href="https://qkzfw2wt.ac1.ai).">未找到标题</a>：未找到描述</li><li><a href="https://x.com/teortaxesTex/status/1819473499347468617">Teortaxes▶️ (@teortaxesTex) 的推文</a>：每百万 token 的复用上下文仅需 $0.014。想想我们刚刚读到的关于使用 DeepSeek API 暴力破解 SWE-bench 的内容。我从 2023 年起就一直在发关于缓存复用的推文。这只是...</li><li><a href="https://openrouter.ai/models/perplexity/llama-3.1-sonar-large-128k-online">Perplexity: Llama 3.1 Sonar 70B Online by perplexity</a>：Llama 3.1 Sonar 是 Perplexity 最新的模型系列。它在成本效益、速度和性能方面超越了早期的 Sonar 模型。这是 [离线聊天模型](/m... 的在线版本。</li><li><a href="https://x.com/gabrielmbmb_/status/1819398254867489001">Gabriel Martín Blázquez (@gabrielmbmb_) 的推文</a>：发布 magpie-ultra-v0.1，这是第一个使用 Llama 3.1 405B 构建的开源合成数据集。使用 distilabel 创建，这是我们迄今为止最先进且计算密集度最高的流水线。https://huggingfac...</li><li><a href="https://argilla-argilla-template-space.hf.space/datasets">Argilla</a>：未找到描述</li><li><a href="https://x.com/basetenco/status/1819048091451859238">Baseten (@basetenco) 的推文</a>：我们很高兴推出适用于 TensorRT-LLM 的新 Engine Builder！🎉 同样的 NVIDIA TensorRT-LLM 卓越性能——减少了 90% 的工作量。查看我们的发布文章了解更多：https://www.baseten.c...</li><li><a href="https://x.com/nikunj/status/1819466795788783976">Nikunj Kothari (@nikunj) 的推文</a>：收到了一堆私信，显然人们似乎还没有完全理解发生了什么。要点如下：- 现任 FTC 团队让收购基本上变得不可能 - 大型在位者想要...</li><li><a href="https://x.com/deepseek_ai/status/1819358570766643223?s=46">DeepSeek (@deepseek_ai) 的推文</a>：🎉激动人心的消息！DeepSeek API 现在推出了磁盘上下文缓存，无需更改代码！这项新功能会自动将频繁引用的上下文缓存在分布式存储中，大幅削减...</li><li><a href="https://x.com/character_ai/status/1819138734253920369?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Character.AI (@character_ai) 的推文</a>：很高兴分享我们将开源创新的 Prompt 设计方法！在我们的最新博客文章中了解 Prompt Poet 如何彻底改变我们构建 AI 交互的方式：https://r...</li><li><a href="https://x.com/nutlope/status/1819445838705578091?s=46">Hassan (@nutlope) 的推文</a>：介绍 LlamaCoder！一个开源的 Claude Artifacts 应用，可以使用 Llama 3.1 405B 生成完整的 React 应用和组件。100% 免费且开源。http://llamacoder.io</li>

<li><a href="https://x.com/theomarcu/status/1819455774579732673?s=46">Theodor Marcu (@theomarcu) 的推文</a>：我花了一些时间构建了一个 Character AI 的竞争对手。关于这条新闻的一些想法：引用 TechCrunch (@TechCrunch) http://Character.AI CEO Noam Shazeer 重返 Google https://tcrn.ch/3WQCl7R</li><li><a href="https://x.com/nikunj/status/1819457199871263230">Nikunj Kothari (@nikunj) 的推文</a>：Inflection (Microsoft)、Adept (Amazon) 以及现在的 Character (Google)。这些例子“将会”让早期员工质疑他们最初为什么要加入高增长的初创公司，因为他们是...</li><li><a href="https://x.com/_xjdr/status/1819475619224473628?s=46">xjdr (@_xjdr) 的推文</a>：Google 拥有：- AlphaZero - 非常擅长搜索和索引 - Gemini 凭借 1M ctx len 在 lmsys 上刷榜 - 世界上一些最顶尖的研究员和工程师（现在再次包括 Noam 和...</li><li><a href="https://x.com/Tim_Dettmers/status/1818282778057941042">Tim Dettmers (@Tim_Dettmers) 的推文</a>：在求职市场 7 个月后，我很高兴地宣布：- 我加入了 @allen_ai - 2025 年秋季起担任 @CarnegieMellon 教授 - 新的 bitsandbytes 维护者 @Titus_vK 我的主要重点将是加强...</li><li><a href="https://x.com/nickadobos/status/1819084445481382339?s=46">Nick Dobos (@NickADobos) 的推文</a>：2024 年 8 月 AI 编程状态：最佳基础模型与应用层级列表</li><li><a href="https://x.com/romainhuet/status/1814054938986885550">Romain Huet (@romainhuet) 的推文</a>：@triviatroy @OpenAI GPT-4o 和 GPT-4o mini 的每张图像美元价格是相同的。为了保持这一点，GPT-4o mini 每张图像使用更多的 tokens。感谢您的观察！</li><li><a href="https://x.com/StabilityAI/status/1819025550062850451">Stability AI (@StabilityAI) 的推文</a>：我们很高兴推出 Stable Fast 3D，这是 Stability AI 在 3D 资产生成技术方面的最新突破。这个创新模型仅需... 即可将单张输入图像转换为详细的 3D 资产。</li><li><a href="https://ac1.ai">Adaptive Computer</a>：你会构建什么？</li><li><a href="https://x.com/allen_ai/status/1819077607897682156">Ai2 (@allen_ai) 的推文</a>：经过数月的幕后研究、访谈和心血结晶，我们很高兴在今天首次展示 Ai2 的新品牌和网站。探索演变过程 🧵</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/jHnSGxfHRj">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://x.com/al">未定义推文</a>：未找到描述</li><li><a href="https://x.com/steve8708/status/1819448686424084892?s=46">Steve (Builder.io) (@Steve8708) 的推文</a>：LLMs 简直是有史以来最不可靠的技术（紧随其后的是 ** 的蓝牙）。在经过荒谬的大量尝试和错误后，我们在内部创建了一套规则，使 LLMs 变得相当...</li><li><a href="https://x.com/atroyn/status/1819481762239824231">anton (𝔴𝔞𝔯𝔱𝔦𝔪𝔢) (@atroyn) 的推文</a>：Character AI 的用户真的很希望模型能表现得“好色”。Noam 真的非常不希望模型表现得“好色”。Google 在公司的整个历史上从未制造过任何一个“好色”的产品...</li><li><a href="https://x.com/ContextualAI/status/1819032988933623943">Contextual AI (@ContextualAI) 的推文</a>：我们今天很高兴地分享，我们已经筹集了 8000 万美元的 Series A 融资，以加速我们通过 AI 改变世界运作方式的使命。在我们的博客文章中阅读更多内容：https://contextual.ai/news/an...</li><li><a href="https://github.com/Nutlope/turboseek">GitHub - Nutlope/turboseek: 一个受 Perplexity 启发的 AI 搜索引擎</a>：一个受 Perplexity 启发的 AI 搜索引擎。通过在 GitHub 上创建账户为 Nutlope/turboseek 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1268954528695451748)** (17 messages🔥): 

> - `Winds of AI Winter Podcast`
> - `ChatGPT Voice Mode Demo`
> - `Feature Clamping in Models`
> - `Podcast Recap & Vibe Shift`
> - `Benchmarking with Singapore Accent` 


- **Winds of AI Winter 播客发布**: 最新一期标题为 *Winds of AI Winter* 的播客已上线，内容涵盖了对过去几个月 AI 领域的总结，并庆祝 **100 万次下载**。
   - 听众可以通过 [播客链接](https://latent.space/p/q2-2024-recap) 获取完整的讨论和回顾。
- **令人印象深刻的 ChatGPT Voice Mode 演示**: 播客在结尾处展示了一个语音演示，呈现了 **Singapore accent** 如何提升用户体验并开辟新的 Benchmarking 可能性。
   - 听众对在角色扮演游戏中的潜在应用表示兴奋，激发了对创新用例的兴趣。
- **关于 Feature Clamping 的讨论**: 一位听众针对播客中提到的 Feature Clamping 是否能提升模型在 Coding 中的性能提出了疑问。
   - 对话强调了需要深入探索调整 Activation（例如增加 'python' 特征）如何影响实际任务中的性能。
- **AI 领域 Vibe Shift 的回顾**: 播客深入探讨了 AI 格局的转变，重点关注从 **Claude 3.5** 到 **Llama 3.1** 等模型的变化。
   - 讨论还涵盖了在 **LLM OS** 领域新兴竞争背景下 RAG/Ops 的扩张。
- **播客版本混淆已解决**: 针对播客标题可能存在的错误曾出现混淆，但随后确认使用了正确的版本。
   - 听众和主持人讨论了关于细微编辑差异的后勤问题，以确保内容保持吸引力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/latentspacepod/status/1819394111352590802">来自 Latent.Space (@latentspacepod) 的推文</a>: 🆕 pod: The Winds of AI Winter!  https://latent.space/p/q2-2024-recap  Vibes 已经转变...  @fanahova 和 @swyx 庆祝 100 万次下载并回顾过去 3 个月的 AI 动态！讨论前沿领域...</li><li><a href="https://x.com/la">来自 FxTwitter / FixupX 的推文</a>: 抱歉，该用户不存在 :(</li><li><a href="https://tenor.com/view/im-doing-my-part-soldier-smile-happy-gif-15777039">Im Doing My Part Soldier GIF - Im Doing My Part Soldier Smile - 发现并分享 GIF</a>: 点击查看 GIF
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1269022115382951957)** (72 条消息🔥🔥): 

> - `Cursor vs. Cody`
> - `AI 工具中的上下文管理`
> - `Aider.nvim 的使用`
> - `Claude 的本地同步功能`
> - `Composer 的预测性编辑` 


- **关于 Cursor 和 Cody 有效性的辩论**：用户讨论了来自 Sourcegraph 的 **Cody** 如何实现上下文感知索引，而 **Cursor** 由于复杂的上下文管理在某些情况下反而造成了阻碍。
   - *yikesawjeez* 表示在 Cursor 中管理上下文非常困难，因此他们更倾向于使用 Aider.nvim 进行手动控制。
- **Aider.nvim 的独特功能**：Aider.nvim 拥有一项抓取 URL 以进行自动文档检索的功能，用户发现这一功能非常有用。
   - *yikesawjeez* 指出，与 Aider.nvim 的功能相比，他们自己开发的手动抓取版本显得多余。
- **Claude 即将推出的同步功能**：用户报告称 **Claude** 正在开发本地同步文件夹（Sync Folder）功能，允许用户批量上传文件。
   - 这一功能被强调为在工作流中实现更高效项目管理的一大进步。
- **Composer 的有趣特性**：围绕 **Composer** 的讨论揭示了其预测性编辑能力，通过 *ctrl+k* 等命令增强了内联编辑体验。
   - *disco885* 称赞其频繁的可用性调研是开发中的动力因素，并指出了其潜在的强大功能。
- **作为数字工具箱的 AI 工具**：参与者赞赏了所讨论工具的集体实用性，称其为开发者的数字工具箱。
   - *disco885* 强调了 *Slono* 分享的令人印象深刻的工作流示例，突出了它们的有效性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://sourcegraph.com/blog/how-cody-understands-your-codebase">Cody 如何理解你的代码库</a>：上下文是 AI 编程助手的关键。Cody 使用多种上下文获取方法，为企业级代码库提供相关的答案和代码。</li><li><a href="https://sourcegraph.com/blog/how-cody-provides-remote-repository-context">Cody 如何为各种规模的代码库提供远程仓库感知</a>：利用 Sourcegraph 平台，Cody 的上下文感知能力可以扩展到任何规模的代码库，从最小的初创公司到最大的企业。</li><li><a href="https://x.com/testingcatalog/status/1816945228869206260">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>：Anthropic 正在为 Claude Projects 开发同步文件夹（Sync Folder）功能 👀 用户可以选择一个本地文件夹来批量上传文件。
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1268689203206557707)** (165 条消息🔥🔥): 

> - `AI Hackathon Series Tour`
> - `GraphRAG System`
> - `Neurosity Crown for Focus`
> - `Dwarf Fortress Gameplay`
> - `Silent Gaming Equipment` 


- **AI Hackathon Series Tour 宣布举办**：一场令人兴奋的 **AI Hackathon Series Tour** 即将在全美各地举行，最终将在 **PAI Palooza** 达到高潮，展示当地的 AI 初创公司和创新成果。
   - 鼓励参与者立即注册，以确保在这场专注于 AI 协作和技术进步的活动中占有一席之地。
- **面向投资者的 GraphRAG 系统**：一位成员介绍了他们的 **GraphRAG** 系统，该系统基于抓取的 200 万个公司网站构建，旨在帮助投资者识别有潜力的公司。
   - 该系统目前正与一个 Multi-Agent 框架同步开发，该成员正在寻找合作机会。
- **Neurosity Crown 提高生产力**：**Neurosity Crown** 被强调为一种生产力工具，通过在注意力分散时提供音频提示（如鸟鸣声）来提高专注力。
   - 虽然一些用户质疑其功效，但一位成员报告称，在工作期间他们的专注力得到了显著提升。
- **Dwarf Fortress 游戏体验**：关于 **Dwarf Fortress** 的幽默讨论展开，提到了意想不到的游戏机制（例如矮人踩到洒出的酒滑倒）如何导致游戏内的混乱事件。
   - 成员们分享了他们对这款游戏的各种体验，表达了怀旧之情，也谈到了其独特的模拟风格带来的挑战。
- **游戏设备讨论**：成员们讨论了他们的游戏配置，包括静音鼠标的优势以及像 **Kinesis** 这样的人体工程学键盘，以获得更好的打字和游戏体验。
   - 对话强调了长时间使用中舒适度和人体工程学的重要性，并推荐了一些高质量设备。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.cohere.com/docs/rate-limits">API Keys and Rate Limits</a>：Cohere 提供两种 API keys：试用 key（有各种随附限制）和生产 key（没有此类限制）。您可以通过在 Cohe 注册来获取试用 key...</li><li><a href="https://kinesis-ergo.com/">Kinesis Keyboards</a>：人体工程学键盘、鼠标和脚踏板。</li><li><a href="https://kinesis-ergo.com/keyboards/advantage2-keyboard/">Kinesis Advantage2 Ergonomic Keyboard</a>：分体式轮廓设计，可最大限度地提高舒适度并提升生产力。配备机械轴、板载编程功能等。</li><li><a href="https://lu.ma/2svuyacm">Techstars StartUp Weekend - PAI Palooza &amp; GDG Build with AI—Mountain View · Luma</a>：这次 AI Hackathon Series Tour 是一项开创性的、跨越美国多个城市的活动，汇聚了最聪明的头脑……</li><li><a href="https://neurosity.co/">no title found</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1268670107895463938)** (22 messages🔥): 

> - `Aspect Based Sentiment Analysis`
> - `AI Project Suggestions`
> - `Cohere API for Classification`
> - `RAG with Chat Embed and Rerank Notebook Errors` 


- **探索使用 Cohere 进行 Aspect Based Sentiment Analysis**：一位用户正在评估使用 Cohere 进行基于方面的情感分析，旨在将评论分类到特定的维度，如音量和吸力。
   - 他们注意到，与 OpenAI 的模型相比，Cohere 模型需要的训练样本更少，这可能意味着在应用中具有更高的成本效益。
- **AI 项目建议**：成员们提出了各种 AI 项目，如垃圾邮件检测、产品评论的情感分析以及交通标志识别。
   - 他们还建议查看 [Cohere AI Learning Library](https://cohere.com/llmu) 以获取资源，增强对 AI 领域的理解。
- **利用 Cohere API 进行高效分类**：一位用户发现 Cohere 平台上的分类模型非常有效，并对使用该 API 构建原型感到兴奋。
   - 建议他们可视化数据集模式，并将其用于未来具有成本效益的训练。
- **RAG with Chat Embed and Rerank Notebook 中的错误**：一位用户在运行用于 RAG 功能的 GitHub Notebook 时遇到了与 `rank_fields` 和 `chat_stream` 相关的错误。
   - 其他成员通过询问代码更改和日志来提供协助，以便有效地排除故障。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://cohere.com/llmu">LLM University (LLMU)</a>: 欢迎来到 LLM University，这是您掌握企业级 AI 技术的首选学习目的地。专为开发者和技术专业人士设计，我们的中心提供全面的资源、经验...</li><li><a href="https://docs.cohere.com/docs/structured-outputs-json">Structured Generations (JSON)</a>: Cohere 模型（如 Command R 和 Command R+）非常擅长生成 JSON 等格式的结构化输出。为什么要使用 LLM 生成 JSON 对象？JSON 是一种轻量级格式，易于...</li><li><a href="https://github.com/cohere-ai/notebooks/blob/main/notebooks/llmu/RAG_with_Chat_Embed_and_Rerank.ipynb">notebooks/notebooks/llmu/RAG_with_Chat_Embed_and_Rerank.ipynb at main · cohere-ai/notebooks</a>: Cohere 平台的代码示例和 Jupyter Notebooks - cohere-ai/notebooks
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1268945576653885491)** (4 messages): 

> - `Web3 Contract Opportunity`
> - `Spam Concerns in Chat` 


- **面向开发者的 Web3 合同机会**：一位用户表示有兴趣与在 **Web3**、**Chainlink**、**LLM** 和 **UI 开发**方面有经验的人交流，以寻求兼职合同机会。
   - 这表明社区内对新兴技术技能的需求。
- **关于聊天垃圾帖子的讨论**：一位成员指出，聊天中的某些帖子是垃圾信息且通常是诈骗，建议可以删除这些帖子以获得更好的社区体验。
   - 另一位用户对这一问题的关注表示感谢，强调了社区在维护聊天质量方面的努力。


  

---


### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/1268932151844409445)** (6 messages): 

> - `Toolkit Customization`
> - `Guidelines for Modifying Code`
> - `Collaboration and Contributions`
> - `Use-cases Evaluation`
> - `Upstream Updates` 


- **Toolkit 为定制化提供了灵活性**：一位成员对 **toolkit** 表示了兴趣，强调了其极具前景的定制选项，如启用身份验证和主题。
   - 对于更深层次的修改，可能需要 *Fork 项目并构建 Docker 镜像*。
- **讨论了安全修改的最佳实践**：另一位成员做出了回应，指出已经提供了使用 toolkit 的**指南**，并鼓励进行定制。
   - 他们强调社区非常乐意看到并支持对*项目的贡献*。
- **请求针对特定用例的指导**：一位成员询问是否有需要**指导**的具体方面，从而引发了对 toolkit 理想用例的后续讨论。
   - 他们很好奇用例是更侧重于 **Chat x RAG** 还是分析。
- **评估内部工具扩展**：一位成员提到，他们目前正在**评估** toolkit 是否适合他们的场景，特别是使用内部工具对其进行扩展的可能性。
   - 他们还表示有兴趣关注**上游开发 (upstream developments)** 的最新动态。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1268644458040266862)** (50 messages🔥): 

> - `GitHub and Hugging Face competition` (GitHub 与 Hugging Face 的竞争)
> - `EU AI regulation concerns` (欧盟 AI 监管担忧)
> - `LLM evaluation metrics` (LLM 评估指标)
> - `Developing new neural network architectures` (开发新的神经网络架构)
> - `Code understanding tools for LLMs` (用于 LLM 的代码理解工具)


- **GitHub 旨在与 Hugging Face 竞争**：成员们对 GitHub 的做法表示担忧，指出这感觉像是在启动一个专注于托管模型的有限演示，而不是侧重于社区贡献。
   - 一位成员建议，这可能是一个战略举措，旨在保持控制并防止 ML 社区的代码迁移。
- **欧盟 AI 监管的不确定性**：英国科技大臣表示，即将出台的 AI 法案将侧重于主要模型，而不会增加过度监管，但成员们对其可执行性仍持怀疑态度。
   - 讨论强调了对欧盟和英国法律可能如何影响全球公司，以及初创公司是否能够实现合规的担忧。
- **选择 LLM 评估指标**：一位成员询问了评估 LLM 输出的合适指标，并提到了对代码使用精确匹配（exact matches）所面临的挑战。
   - 几位用户建议了像 HumanEval 这样的方法，同时承认在评估中使用 `exec()` 存在的风险。
- **创建新神经架构的指南**：关于开发新的深度学习架构的咨询强调了编码不变性（invariances）和先验实验的重要性。
   - 一位用户建议使用对比学习（contrastive learning）技术，以提高模型在基于用户的数据上的不变性和性能。
- **用于 LLM 代码理解的工具**：一位用户介绍了一个 Python 工具，旨在帮助 LLM 在不需要访问所有文件的情况下理解代码库，声称这能提高效率。
   - 尽管一些人要求提供其有效性的实证证据，后续讨论强调了通过 API 调用来管理文件交互的潜力。



**Link mentioned**: <a href="https://archive.is/2yfdW">UK&#x2019;s AI bill to focus on ChatGPT-style models</a>: 未找到描述

  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1268649999940583536)** (134 messages🔥🔥): 

> - `Distillation Techniques` (蒸馏技术)
> - `GEMMA Model Performance` (GEMMA 模型性能)
> - `Training Dynamics` (训练动力学)
> - `Logit Distillation vs Synthetic Data` (Logit 蒸馏 vs 合成数据)
> - `Parameter Initialization Effects` (参数初始化影响)


- **重新审视蒸馏技术**：对话强调了人们对 **logit distillation** 兴趣的复苏，成员们指出尽管过去对其有效性存在怀疑，但它对数据效率有影响，并能为较小模型带来微小的质量提升。
   - 近期论文中的几个例子表明，蒸馏方法可以有不同的应用，包括那些使用合成数据的应用，这表明该领域的方法论正在演进。
- **对 GEMMA 模型对比的担忧**：关于 GEMMA 性能的讨论指出了 Benchmark 结果的差异，特别是关于与 **Mistral** 等模型的对比，凸显了评估过程可能存在的不透明性。
   - 成员们辩论了训练动力学和资源分配对性能指标的影响，寻求澄清各种模型设置如何影响结果。
- **训练参数的影响**：小组探讨了如何改变模型的初始化方差，特别注意到对于 NTK 模型，采用了不同的学习率，导致了不同的训练动力学。
   - 这引发了关于调整是否真正符合预期行为的分析，因为使用不同的参数化方法产生了矛盾的结果。
- **合成数据 vs. Logit 蒸馏**：对比了合成数据的使用与传统的 logit 蒸馏方法，多位参与者表示需要明确这些术语目前的定义和理解方式。
   - 强调了进行严格实验和基准对比的必要性，以便在其各自的框架内更好地评估这些策略。
- **理解结果的挑战**：成员们对近期论文中数据的差异表示困惑，特别是某些实验在模型设置理论等效的情况下却产生了意外结果。
   - 强调了需要彻底的验证过程，以确保不同的初始化和学习率配置在性能评估中得到准确呈现。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.21770">MoMa: Efficient Early-Fusion Pre-training with Mixture of Modality-Aware Experts</a>: 我们介绍了 MoMa，这是一种新型的模态感知混合专家 (MoE) 架构，专为预训练混合模态、早期融合 (early-fusion) 语言模型而设计。MoMa 以任意序列处理图像和文本...</li><li><a href="https://www.eleuther.ai/hashes">Hashes &mdash; EleutherAI</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2408.00724">An Empirical Analysis of Compute-Optimal Inference for Problem-Solving with Language Models</a>: 关于大型语言模型 (LLMs) 在模型大小和计算预算方面的最优训练配置已得到广泛研究。但如何在推理过程中最优地配置 LLMs 以解决问题...</li><li><a href="https://arxiv.org/abs/2306.13649">On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes</a>: 知识蒸馏 (KD) 被广泛用于通过训练较小的学生模型来压缩教师模型，从而降低其推理成本和内存占用。然而，目前针对自动回归 (auto-regressive)...</li><li><a href="https://arxiv.org/abs/2407.05872">Scaling Exponents Across Parameterizations and Optimizers</a>: 将模型从窄宽度稳健且有效地缩放到大宽度，通常需要精确调整许多算法和架构细节，例如参数化 (parameterization) 和优化器 (optimizer) 的选择...</li><li><a href="https://arxiv.org/abs/2407.14679">Compact Language Models via Pruning and Knowledge Distillation</a>: 针对不同部署规模和尺寸的大型语言模型 (LLMs) 目前是通过从头开始训练每个变体来生产的；这极其耗费计算资源。在本文中，我们研究...</li><li><a href="https://arxiv.org/abs/2408.00118">Gemma 2: Improving Open Language Models at a Practical Size</a>: 在这项工作中，我们介绍了 Gemma 2，它是 Gemma 系列轻量级、最先进开源模型的新成员，参数规模从 20 亿到 270 亿不等。在这个新版本中，我们...</li><li><a href="https://x.com/mlcommons/status/1819098247270695254">Tweet from MLCommons (@MLCommons)</a>: @MLCommons #AlgoPerf 结果出炉！🏁 5 万美元奖金的竞赛产生了比 Nesterov Adam 快 28% 的神经网络训练结果，采用了非对角预处理 (non-diagonal preconditioning)。无超参数算法的新 SOTA...</li><li><a href="https://x.com/karinanguyen_/status/1819082842238079371">Tweet from Karina Nguyen (@karinanguyen_)</a>: 我正在为 OpenAI 的 Model Behavior 团队招聘！这是我梦想的工作，处于设计工程和训练后 (post-training) 研究的交汇点，也是世界上最罕见的岗位 ❤️ 我们定义了模型的核心行为...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1268656782767558738)** (3 messages): 

> - `双下降现象 (Double Descent Phenomenon)`
> - `参数和数据大小对 Loss 的影响` 


- **两者都存在双下降**：一位成员指出，在分析模型训练过程中跨参数和数据大小的行为时，**双下降 (double descent)** 现象非常明显。
   - 他们强调，当在 x 轴上绘制 **epochs** 时，这一观察结果会发生变化。
- **训练机制影响 Loss**：讨论强调，对于给定的训练机制，保持除参数和数据大小以外的所有因素固定，可以揭示出意想不到的行为。
   - 有人指出，存在一个显著区域，其中一个或两个因素可能对 **Loss 产生负面影响**。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/)** (1 messages): 

norabelrose: https://x.com/norabelrose/status/1819395263674699874
  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1268747660022513665)** (8 messages🔥): 

> - `PhD research gaps`
> - `Evaluation tasks in AI`
> - `Broader Impacts Evaluation workshop`
> - `Provocative claims in social impact evaluation`
> - `Collaborative ML paper writing` 


- **PhD 学生寻找研究空白**：一位成员表示在他们的 PhD 历程中感到迷茫，并回想起一个讨论酷炫研究想法的 Twitter 线程。
   - 他们正在寻找其领域内可以开展工作的潜在空白点。
- **使用多个 Prompt 运行评估**：一位新成员请求指导，关于如何添加一个涉及同时评估多个 Prompt 并聚合结果的任务。
   - 他们正在寻求有效执行此过程的最佳实践。
- **NeurIPS 研讨会公告**：一位成员分享了一个 [Twitter 线程](https://x.com/yjernite/status/1819021732126044489?s=46)，宣布在 NeurIPS 举办关于 GenAI 广泛影响评估（Broader Impacts Evaluation）的研讨会。
   - 该研讨会旨在讨论评估作为一种基于受影响方需求的治理工具的重要性。
- **征集挑衅性论文**：成员们正在为研讨会征集微型论文（tiny papers），包括关于 GenAI 社会影响评估的短篇研究论文和新颖视角。
   - 提到的一个挑衅性观点是：“没有理由相信经过 RLHF 的模型比基础模型（base models）更安全。”
- **机器学习论文的潜在合作**：一位成员表达了共同撰写机器学习论文的兴趣，表明他们正在寻找发表机会。
   - 他们热衷于贡献想法，特别是围绕社会影响评估的挑衅性主张。



**提及的链接**：<a href="https://x.com/yjernite/status/1819021732126044489?s=46">来自 Yacine Jernite (@YJernite) 的推文</a>：很高兴宣布我们在 @NeurIPSConf 举办的关于 GenAI 广泛影响评估的研讨会！评估是一个重要的治理工具；如果它有足够的依据、定义，并由……的需求所驱动。

  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1268644476436615168)** (91 messages🔥🔥): 

> - `OpenAI Voice Mode`
> - `Latency Issues with Assistants API`
> - `Gemini 1.5 Pro Experiment`
> - `Gemma 2 2b Model`
> - `Flux Image Model` 


- **OpenAI 语音模式热议**：成员们讨论了新的 OpenAI 语音模式，一些人提到自发布以来收到了大量私信。
   - *似乎许多人都渴望进一步了解其功能和访问权限*。
- **Assistants API 的延迟担忧**：一位用户报告了关于 Assistants API 的延迟问题，寻求社区对其体验的反馈。
   - 一些人推荐使用 SerpAPI 等替代方案来处理实时信息抓取任务。
- **Gemini 1.5 Pro 表现优异**：讨论强调了 Gemini 1.5 Pro 优于其他模型的表现，成员们对其现实应用感到好奇。
   - *一位参与者指出，他们自己对该模型的使用证明了其在响应质量方面具有竞争力*。
- **Gemma 2 2b 的能力**：一位成员分享了对 Gemma 2 2b 模型潜力的见解，强调了它在指令遵循（instruction following）方面的有效性，尽管其知识储备不如大型模型。
   - 对话集中在平衡技术能力与实际应用的可靠性上。
- **对 Flux 图像模型的兴奋**：Flux 图像模型的发布引发了兴奋，社区成员开始测试其与 MidJourney 和 DALL-E 等现有工具相比的能力。
   - *用户注意到其开源特性和较低的资源需求，暗示了其广泛使用的潜力。*



**提及的链接**：<a href="https://x.com/gdb/status/1790869434174746805">来自 Greg Brockman (@gdb) 的推文</a>：一张 GPT-4o 生成的图像 —— 仅 GPT-4o 的图像生成能力就有如此多值得探索的地方。团队正在努力将这些功能带给世界。

  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1268707165292728373)** (8 messages🔥): 

> - `GPT Custom Instructions`
> - `Fine-Tuning GPTs`
> - `Personalized GPTs`
> - `Custom GPT for OCT Processing` 


- **GPT Custom Instructions 泄露担忧**：开发者对 GPT Custom Instructions 可能被泄露表示担忧，这可能导致他人创建仿冒版本，从而威胁到变现能力。
   - 为了避免*不必要的纷争*，分享指令的想法被完全放弃了。
- **Fine-Tuning 频率与限制**：一名成员建议将 GPT Fine-Tuning 限制在每月五次，以提高性能和可用性。
   - 这一限制可能有助于管理 Fine-Tuned 模型的质量，同时确保其满足特定用户的需求。
- **个性化 GPTs 提升效率**：用户认可了为个人任务创建自定义 GPTs 的有效性，这增强了跨项目的 workflow。
   - 一位成员提到，他们主要使用自己创建的 GPTs 来简化不同的任务。
- **通过 OpenAI API 访问自定义 GPT**：一位用户询问是否可以通过 OpenAI API 访问用于 OCT 处理的自定义 GPT。
   - 这凸显了在 ChatGPT 网页界面之外的其他环境中使用自定义解决方案的兴趣。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1268720676294885447)** (6 messages): 

> - `Text Length Reduction`
> - `LLM Limitations`
> - `Python Tool for Word Counting` 


- **缩减文本长度的挑战**：由于 LLM 预测文本方式的局限性，成员们对能否成功将文本缩减到特定的字符或单词数量表示担忧。
   - 一位成员指出，LLM 难以进行精确计算，因此很难达到准确的长度。
- **使用定性语言进行文本缩减**：在要求 LLM 缩短文本时，“定性语言”（如“短”或“非常短”）通常可以提供相对一致的长度，尽管不是精确的计数。
   - 尽管如此，LLM 对“缩短一半”或“对半拆分”等指令的反应可能不如预期。
- **使用 Python 作为字数统计方案**：有人建议在字数统计至关重要的情况下使用 Python，由 AI 生成初始文本，并由 Python 处理计数。
   - 这种方法可以实现更多控制，因为 Python 可以检查字数，并在必要时返回“缩短”指令。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1268720676294885447)** (6 messages): 

> - `Text Shortening Challenges`
> - `Upgraded ChatGPT Versions`
> - `Python for Word Counting` 


- **LLM 缩短文本的挑战**：成员们讨论了 LLM 在将文本缩短到特定字符或单词数量方面的局限性，强调 LLM 主要是预测文本而非执行计算。
   - 一位成员指出，虽然定性语言可以帮助引导长度，但不会产生精确的计数。
- **LLM 交互性查询**：一位成员询问，要求 LLM 定量地“缩短”文本（如“缩短一半”）是否会产生一致的结果。
   - 回复指出 LLM 难以进行精确的字数统计，导致结果不可预测。
- **利用 Python 工具获取可靠字数**：建议在文本长度至关重要时使用 **Python 工具** 进行更准确的字数统计，而 AI 则负责生成初始文本。
   - 这种方法允许进行系统性检查，如果字数超过预期限制，则指示 AI “缩短”内容。
- **对升级版 ChatGPT 版本的兴趣**：在另一场讨论中，一位成员正在寻找 **ChatGPT-4** 或 **4o** 的升级版本，并请求有访问权限的人员私信。
   - 这反映了社区对获取增强型 AI 能力的持续兴趣。


  

---



### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1268732833459998750)** (3 messages): 

> - `LLM-as-Judge`
> - `Synthetic Dataset Generation`
> - `WizardLM Papers` 


- **寻找 LLM-as-Judge 相关读物**：一位成员征求关于 **LLM-as-Judge** 当前主流趋势（meta）以及合成数据集生成的必读综述推荐，特别是侧重于指令和偏好数据的部分。
   - 这一咨询凸显了对 LLM 领域内有效方法的持续关注。
- **推荐 WizardLM 论文**：另一位成员建议查看 **WizardLM** 最新的两篇论文，认为它们是关于 LLM 和合成数据集生成咨询的相关读物。
   - 这一推荐表明 WizardLM 最近的研究可能为所讨论的话题提供有价值的见解。


  

---

### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1268758411806511184)** (15 条消息🔥): 

> - `Cooking Recipes` (烹饪食谱)
> - `Nous Merch` (Nous 周边)
> - `Deep Frying` (油炸)
> - `Community Engagement` (社区互动)


- **寻求新烹饪食谱**：成员们表达了对新烹饪灵感的渴望，其中一位表示他们经常失望地离开该频道。
   - 另一位成员幽默地引用了“人不能两次踏进同一条河流”，敦促尽管有旧内容，仍需发布新鲜帖子。
- **Nous 周边的制作**：讨论了 Nous 周边商品，一位成员迫切询问何时可以购买。
   - 回复指出，包括一件**优质连帽衫**在内的高质量单品正在筹备中，很快就会发布。
- **用牛脂炸薯条**：一位成员分享了用从分类广告网站购买的牛脂炸薯条的经验，展示了他们的烹饪方法。
   - 他们提到使用了一个**无名品牌的中国电炸锅**，引发了其他人对该品牌的询问。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1269046429595472015)** (3 条消息): 

> - `VRAM calculation for LLMs` (LLM 的 VRAM 计算)
> - `Black Forest Labs generative AI` (Black Forest Labs 生成式 AI)
> - `FLUX.1 models` (FLUX.1 模型) 


- **高效 VRAM 计算脚本发布**：一个新的 [VRAM 计算脚本](https://gist.github.com/jrruethe/8974d2c8b4ece242a071d1a1526aa763) 允许用户根据模型、每个权重的位数（bits per weight）和上下文长度来确定 LLM 的 VRAM 需求。
   - *该脚本不需要外部依赖*，并提供了根据可用 VRAM 评估上下文长度和每个权重位数的功能。
- **Black Forest Labs 发布 SOTA 文本转图像模型**：**Latent Diffusion 团队**推出了 _Black Forest Labs_，并介绍了 **FLUX.1** 系列 SOTA 文本转图像模型，旨在推进生成式 AI 的发展。
   - 正如其 [公告](https://blackforestlabs.ai/announcing-black-forest-labs/) 中详述的那样，该团队致力于让生成式模型变得触手可及，增强公众信任，并推动媒体生成领域的创新。
- **FLUX 官方推理仓库现已上线 GitHub**：FLUX.1 模型的**官方 GitHub 仓库**已发布，为希望实现该模型的用户提供资源和支持。
   - 该仓库可通过 [black-forest-labs/flux](https://github.com/black-forest-labs/flux) 访问，旨在促进对项目开发的贡献。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://blackforestlabs.ai/announcing-black-forest-labs/">Announcing Black Forest Labs</a>: 今天，我们很高兴地宣布 Black Forest Labs 正式成立。我们深植于生成式 AI 研究社区，我们的使命是开发和推进 SOTA 生成式...</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/1ehoqmt/script_calculate_vram_requirements_for_llm_models/">[Script] Calculate VRAM requirements for LLM models</a>: 脚本在这里：https://gist.github.com/jrruethe/8974d2c8b4ece242a071d1a1526aa763 一段时间以来，我一直试图弄清楚我可以运行哪些量化...</li><li><a href="https://github.com/black-forest-labs/flux">GitHub - black-forest-labs/flux: Official inference repo for FLUX.1 models</a>: FLUX.1 模型的官方推理仓库。通过创建一个账户来为 black-forest-labs/flux 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1268670332860891136)** (60 条消息🔥🔥): 

> - `Gemma 2B vs Qwen 1.5B`
> - `使用 Bitnet 进行 Finetuning`
> - `N8Leaderboard 实现`
> - `Llama 405B 性能`
> - `AI 模型在编程中的对比` 


- **Gemma 2B 相比 Qwen 1.5B 被过度吹捧**：成员们讨论了 **Gemma 2B** 如何被认为存在过度吹捧，据报道 **Qwen 2** 在包括 MMLU 和 GSM8K 在内的各种 Benchmark 上都表现得更好。
   - 一位成员提到，尽管 Qwen 具备强大的能力，但其表现却被忽视了，并称其“残暴地”击败了 Gemma 2B。
- **Bitnet 带来的 Finetuning 复兴**：一位成员分享了一个关于使用 **Bitnet** 进行 Finetuning 的链接，最终生成了一个 **74MB** 的文件，在单个 CPU 核心上运行速度达到每秒 198 tokens。
   - 这引发了关于该项目潜在 **Open-sourcing**（开源）的讨论，以及在当前 AI 领域下 Finetuning 的有效性。
- **N8Leaderboard 受到关注**：N8Leaderboard 正在被讨论，一位成员强调它由 **30 个智力题**组成，用于评估。
   - 另一位成员询问了用于实现该排行榜的软件和硬件，表示希望复制这一过程。
- **Llama 405B 展示了令人印象深刻的准确率**：讨论强调了 **Llama 405B** 模型在仅经过 **2 epochs** 训练后就达到了 **90% 的准确率**，这引起了成员们的极大兴趣。
   - 成员们辩论了该模型的训练逻辑，其中一人指出其 **validation loss**（验证损失）最初看起来非常理想。
- **AI 模型的对比分析**：一位成员目前正在对比 **Athena 70B** 和 **Llama 3.1** 在编程任务中的表现，并分享了关于性能差异的见解。
   - 这引发了关于模型整体趋势的进一步讨论，即模型往往表现出更好的理论性能，而非实际应用中的有效性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/nisten/status/1818529201231688139">nisten (@nisten) 的推文</a>：修改了 Bitnet 用于 Finetuning，最终得到了一个 74mb 的文件。它在仅 1 个 CPU 核心上以每秒 198 tokens 的速度运行。简直是巫术。稍后通过 @skunkworks_ai 开源，Base 模型在这里：https://huggi...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ehh9x2/hacked_bitnet_for_finetuning_ended_up_with_a_74mb/">Reddit - 深入探讨</a>：未找到描述</li><li><a href="https://x.com/nisten/status/1818536486662271167?s=46">nisten (@nisten) 的推文</a>：@reach_vb @skunkworks_ai 对没人分享 Bitnet 代码感到生气，所以我直接根据论文硬撸了出来。但它不收敛。所以我不断地对 smolL 的层进行自动“缝合”...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1268914938584498227)** (8 条消息🔥): 

> - `Llama3.1 Fine-tuning 挑战`
> - `数据集讨论`
> - `Gemma 2B 实验` 


- **Llama3.1 输出乱码**：在私有数据集上对 **Llama3.1 70B** 进行 Fine-tuning 并通过 [vLLM](https://github.com/vllm/vllm) 部署后，一位用户报告称运行速度约为 **30tok/s**，但输出的是不可用的乱码。
   - 设置 *Temperature 0* 也没能解决问题，这表明可能存在数据或模型配置错误。
- **用于 Fine-tuning 的私有数据集**：一位用户确认他们利用私有数据集，参考 [Together AI 的指南](https://www.together.ai/blog/finetuning) 对 Llama3.1 进行了 Fine-tuning。
   - 他们提到，与领先的闭源模型相比，使用专有数据在提高准确率方面取得了成功。
- **关于 Gemma 2B 的讨论**：有人询问尝试使用 [GitHub - LLM2Vec](https://github.com/McGill-NLP/llm2vec) 上的项目来运行 **Gemma 2B**。
   - 该帖子促使成员们分享他们关于 LLM2Vec 实现的相关经验或结果。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.together.ai/blog/finetuning">Fine-tuning Llama-3 以极低的成本获得 GPT-4 90% 的性能</a>：未找到描述</li><li><a href="https://github.com/McGill-NLP/llm2vec/tree/main">GitHub - McGill-NLP/llm2vec: 'LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders' 的代码</a>：'LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders' 的代码 - McGill-NLP/llm2vec
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1268737793606287380)** (6 messages): 

> - `Llama 3.1 性能`
> - `Groq temperature 设置` 


- **Llama 3.1 在输出中表现出不稳定性**：一位用户报告称 **Groq 上的 Llama 3.1** 不稳定，经常产生乱码而非连贯的响应，特别是在进行基于角色（persona-based）的 Wikipedia 数据集搜索时。
   - 他们幽默地请求了一种输出格式，并表示由于性能问题，需要对生成的输出进行清理。
- **Temperature 设置影响性能**：一位来自 Groq 的成员询问了运行 Llama 3.1 时使用的 **temperature 设置**，旨在了解不稳定性产生的原因。
   - 原用户确认将 temperature 设置为 **0** 改善了格式并解决了他们遇到的错误，并对收到的反馈表示感谢。
- **温度基准建议**：用户与 Groq 成员互动以建立 temperature 设置的基准，最终决定使用 **0% temperature** 进行测试。
   - 他们感谢 Groq 团队的指导，并指出调整后输出质量有了显著提高。


  

---


### **Nous Research AI ▷ #[reasoning-tasks-master-list](https://discord.com/channels/1053877538025386074/1264666760972472481/1268729193630269522)** (3 messages): 

> - `Quarto 网站设置`
> - `文件结构确认` 


- **用于启动 Quarto 网站的新 PR**：已创建一个新的 [Quarto 网站 PR](https://github.com/NousResearch/Open-Reasoning-Tasks/pull/17) 以设置项目，增强推理任务的在线展示。
   - 该 PR 包含了所有必要的图片以确保清晰度，并对所做的更改进行了详细说明。
- **文件夹结构说明**：一位成员确认 **chapters** 是一个顶级目录，而所有其他文件也都是顶级文件，确保了清晰的项目组织。
   - 这种结构旨在方便在 GitHub 仓库内进行导航和管理。



**提到的链接**：<a href="https://github.com/NousResearch/Open-Reasoning-Tasks/pull/17">由 mmhamdy 创建的用于构建 Quarto 网站的 Pull Request #17 · NousResearch/Open-Reasoning-Tasks</a>：为任务设置 Quarto 网站。

  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1268657864348799108)** (53 messages🔥): 

> - `FLUX Schnell 性能`
> - `合成数据生成的担忧`
> - `模型训练见解`
> - `数据集策展`
> - `合成数据集的挑战` 


- **FLUX Schnell 表现出弱点**：成员们讨论了 **FLUX Schnell** 模型，指出其似乎训练不足，且在提示词遵循（prompt adherence）方面表现不佳，导致出现类似 *“穿着网球服骑着 Goldwing 摩托车的女性”* 这种荒谬的输出。
   - 有人担心该模型虽然看起来不错，但**缺乏泛化能力**，与其说是有效的生成模型，不如说是一个数据集记忆机器。
- **对合成数据集持谨慎态度**：成员们对使用 **FLUX Schnell** 模型生成合成数据集表示担忧，一些成员警告说这可能导致多代训练后的表征崩溃（representational collapse）。
   - 一位成员提到：*“求求你们，千万不要用 Flux 公开发布的权重来制作合成数据集”*。
- **策展优于随机性的价值**：成员们强调了**经过策展的数据集（curated datasets）**优于随机合成数据集的重要性，指出用户偏好的数据反映了真实的质量，并能避免不必要的浪费。
   - 一位成员指出，完全针对随机提示词进行训练会**浪费资源**，且可能不会带来显著改进。
- **合成数据与真实数据训练的优势**：一些成员讨论了使用合成数据集可以加速训练，通过混合合成数据和真实数据来更快地学习概念。
   - 然而，也有反向观点认为，如果你已经拥有一个强大的原始模型，**直接蒸馏模型（distilling models）**仍然是最佳方法。
- **关于缺失数据集的查询**：一位用户询问了 Hugging Face 上的 **laion2B-en** 数据集，指出该数据集似乎已不再可用，这可能会影响他们在 Stable Diffusion 项目上的工作。
   - 这引发了对模型训练至关重要的数据集可访问性的担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/black-forest-labs/FLUX.1-schnell">FLUX.1 [Schnell] - black-forest-labs 开发的 Hugging Face Space</a>：未找到描述</li><li><a href="https://www.nature.com/articles/s41586-024-07566-y">AI 模型在递归生成的数据上训练时会崩溃 - Nature</a>：分析表明，不加区分地在真实和生成的内容上训练生成式人工智能（通常通过从互联网抓取数据完成），会导致崩溃...
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1268722835052822620)** (17 messages🔥): 

> - `数据增强`
> - `训练 Bug`
> - `参数高效架构`
> - `分类器开发` 


- **数据增强方案效果平平**：在尝试了**数据增强（data augmentation）**、**dropout** 和**权重衰减（weight decay）**后，成员注意到这些技术虽然延迟了过拟合，但并未显著降低**验证误差（validation error）**。
   - 一位用户评论道：*“我其实并不太担心通过正则化来榨取每一个可能的百分点”*，因为**任务是 CIFAR-10**。
- **Bug 阻碍了 LLM 的进展**：一位成员发现代码中有一个**拼写错误**，严重影响了 **50 多个实验**的性能。
   - 在看到新的**损失曲线（loss curve）**比旧版本下降得明显更快时，他们表示非常满意。
- **LLM 泛化问题**：一位成员承认在开发自己的 **LLM** 时修复了 **10 个 Bug**，但注意到模型只是记住了输出而不是进行泛化。
   - 这让他们意识到：*“所以你以为它在工作，但其实并没有。”*
- **真实世界数据建议**：有人建议使用**地毯的照片**作为数据增强的**虚假背景**，以提高真实世界数据的使用率。
   - 采用这种技术可能会对性能曲线产生积极影响，正如一位成员鼓励的那样：*“让我们看看之后你的曲线会如何变化。”*
- **专注于强基准模型**：表达的主要目标是创建一个**强基准模型（strong baseline model）**，而不是专注于正则化技术带来的微小改进。
   - 一位成员澄清说，他们正在开发一个**分类器**，并始终牢记**参数高效架构（parameter-efficient architecture）**。


  

---

### **Interconnects (Nathan Lambert) ▷ #[events](https://discord.com/channels/1179127597926469703/1179127598442348729/1268981009215524928)** (1 条消息): 

> - `Event Sponsorship` (活动赞助)
> - `RL Conference Dinner` (RL Conference 晚宴)


- **对活动赞助的兴趣**：一些成员表达了赞助活动的兴趣，表明了支持未来聚会的积极态度。
   - 社区似乎对为这类倡议寻找资金支持持乐观态度。
- **下周 RL Conference 的晚宴**：一位成员正考虑在下周的 RL Conference 举办晚宴，并正在寻找可能感兴趣赞助的 VC 或播客之友。
   - 这一倡议可以为愿意提供资金支持的与会者提供绝佳的人脉拓展机会。


  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1268994201727668357)** (21 条消息🔥): 

> - `Character AI deal` (Character AI 交易)
> - `Employee concerns post-deal` (交易后的员工担忧)
> - `Implications for AI firms` (对 AI 公司的影响)
> - `Noam's exit from the industry` (Noam 退出该行业)
> - `Regulatory challenges` (监管挑战)


- **Character AI 交易引发关注**：讨论集中在 **Character AI 交易**上，成员们对其对行业的影响表示怀疑。
   - *一位参与者声称这是一笔“奇怪的交易”，引发了关于其后果的进一步讨论。*
- **对非创始人员工的担忧**：人们对非创始人员工的命运表示担忧，质疑他们在交易后可能受到何种影响。
   - *一位成员推测，虽然创始人和 VC 可能获益，但大多数人可能会被抛在后面。*
- **AI 公司可能陷入困境**：一位成员将一些陷入困境的 AI 公司比作“僵尸空壳”，暗示它们在最近的收购之后失去了目标。
   - *另一位成员补充说，这些交易通常涉及规避监管机构，这可能会导致复杂情况。*
- **Noam 退出游戏**：关于 Noam 离开 AI 行业的猜测促使成员们思考他的动机以及他对客户群的看法。
   - *有人建议可能涉及一份丰厚的报价，可能来自 Sundar。*
- **经营一家无意中变成色情内容的初创公司**：成员们对经营一家无意中变成色情内容初创公司的挑战发表了看法。
   - *一位成员对这种情况如何发生表示难以置信，认为这可能不是一段愉快的经历。*


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1268645286012780605)** (19 条消息🔥): 

> - `Ai2 redesign` (Ai2 重新设计)
> - `Sparkles Emoji Trend` (闪烁表情符号趋势)
> - `Copyright Issues with AI` (AI 版权问题)
> - `AI Companies Moving to Japan` (AI 公司迁往日本)
> - `Nonprofit Press Freedom` (非营利组织的媒体自由)


- **Ai2 发布受闪烁元素启发的新品牌**：Ai2 在广泛研究后宣布了其新品牌和网站，灵感来自 [Bloomberg 文章](https://www.bloomberg.com/news/newsletters/2024-07-10/openai-google-adobe-and-more-have-embraced-the-sparkle-emoji-for-ai?srnd=undefined)中强调的在 AI 品牌中使用 **闪烁表情符号 (sparkles emojis)** 的趋势。
   - *Rachel Metz* 评论了 AI 行业对**闪烁元素的拥抱**，指出了其流行度的上升。
- **RIAA 的版权困境**：一位成员指出，**RIAA** 成功应对 AI 生成内容兴起的唯一方法是与 AI 公司**达成协议**。
   - 这带来了挑战，因为存在另一家 AI 公司可能从更友好的司法管辖区出现的风险，导致版权所有者**一无所获**。
- **AI 公司涌向日本**：AI 公司搬迁到日本的趋势日益增长，讨论强调这是一种受**种子下载 (torrenting) 挑战**驱动的策略。
   - 一位成员表示他们不知道这正在成为一种趋势，这揭示了对 AI 监管格局变化的见解。
- **非营利组织与负面新闻**：一位成员对像 OpenAI 这样的**非营利**组织经常受到审查，却因其身份而未受到负面新闻报道表示沮丧。
   - 这促使另一位成员幽默地指出 OpenAI 也被归类为非营利组织，将笑声与严肃的评论融合在一起。



**提到的链接**：<a href="https://x.com/rachelmetz/status/1819086846913401266?s=46">Rachel Metz (@rachelmetz) 的推文</a>：看来 @allen_ai 在重新设计中借鉴了闪烁表情符号的策略！查看我最近关于 AI 行业拥抱 ✨ 的文章，了解更多关于谦逊的闪烁元素如何跃升的信息...

  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1268969947598426192)** (26 条消息🔥): 

> - `Magpie Ultra Dataset`
> - `Instruction and Response Diversity`
> - `Synthetic Data Generation`
> - `Nemotron and Olmo Fine-Tunes`
> - `Ross Taylor Interview` 


- **Magpie Ultra 数据集发布**：HuggingFace 推出了 **Magpie Ultra** 数据集，这是一个包含 **50k 条未过滤**数据的 L3.1 405B 数据集，声称是首个此类开源合成数据集。更多详情请查看他们的 [推文](https://x.com/gabrielmbmb_/status/1819398254867489001) 和 [HuggingFace 上的数据集](https://huggingface.co/datasets/argilla/magpie-ultra-v0.1)。
   - *初始指令质量仍存疑问*，主要担忧在于用户轮次的多样性以及模型的首轮响应是否限制了覆盖范围。
- **用于指令多样性的 Bayesian Reward Model**：讨论了使用 **Bayesian reward model** 的提议，这可以通过引导 Prompt 生成来帮助提高指令和响应的多样性。会议指出，Reward score 的不确定性可能表明 **任务分布采样不足**。
   - 之前的论文建议增加方差惩罚以避免 Reward 过度优化，但这种方法旨在促进探索。
- **合成数据生成的挑战**：目前正在努力处理 **100k 个 Prompt**，用于生成合成指令并将之前的 **GPT-4 completions** 更新为 Llama。这包括生成偏好数据以增强未来的模型。
   - 提出了关于如何妥善利用 **Nemotron** 进行合成数据生成以及避免潜在命名冲突的担忧，特别是考虑到近期公司动态的变化。
- **对即将推出的 Llama-3.1-Olmo-2 的期待**：表达了对即将推出的 **Llama-3.1-Olmo-2 模型** 的热情，并讨论了使用 **Nemotron** 重新制作合成数据以获得最佳效果。强调了对命名策略连贯性和清晰度的需求。
   - 此外，一场备受关注的 **Ross Taylor 访谈** 也值得期待，参与者赞扬了他在 AI 领域的专业知识和贡献。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/gabrielmbmb_/status/1819398254867489001">来自 Gabriel Martín Blázquez (@gabrielmbmb_) 的推文</a>：发布 magpie-ultra-v0.1，这是第一个使用 Llama 3.1 405B 构建的开源合成数据集。使用 distilabel 创建，这是我们迄今为止最先进且计算密集度最高的流水线。https://huggingfac...</li><li><a href="https://argilla-argilla-template-space.hf.space/datasets">Argilla</a>：未找到描述
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1269054386982883349)** (1 条消息): 

> - `Chatroom 改进`
> - `忽略的提供商 (Ignored Providers)`
> - `Parameters API 更新`
> - `新模型上线` 


- **Chatroom 迎来全新更新**：Playground 已更名为 [Chatroom](https://openrouter.ai/chat)，具有更简洁的 UI 和本地聊天保存功能。
   - 通过此次更新，用户将发现配置新聊天室变得更加容易。
- **通过新设置避开不想要的提供商**：用户现在可以通过 [设置页面](https://openrouter.ai/settings/preferences) 避免将请求路由到特定的提供商。
   - 此功能允许对请求处理过程进行更大程度的自定义。
- **现在可以轻松检查模型参数！**：改进后的 Parameters API 允许在 [此链接](https://openrouter.ai/docs/parameters-api) 检查模型和提供商支持的参数及设置。
   - 这一增强功能使得理解模型能力变得更加容易。
- **令人兴奋的新模型发布**：新模型包括用于生成训练数据的 **Llama 3.1 405B BASE**，以及可在 [此处](https://openrouter.ai/models/meta-llama/llama-3.1-8b-instruct:free) 免费使用的 **Llama 3.1 8B**。
   - 其他模型包括 **Mistral Nemo 12B Celeste**（一款专门的写作和角色扮演模型），以及用于提供带链接事实性回答的 **Llama 3.1 Sonar 系列**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/settings/preferences">Settings | OpenRouter</a>：管理您的账户和偏好设置</li><li><a href="https://openrouter.ai/docs/parameters-api">Parameters API | OpenRouter</a>：用于管理请求参数的 API</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-405b">Meta: Llama 3.1 405B (base) by meta-llama</a>：Meta 最新的模型系列 (Llama 3.1) 推出了多种尺寸和版本。这是 405B 基础预训练版本。与领先的闭源模型相比，它展示了强大的性能...</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-8b-instruct:free">Meta: Llama 3.1 8B Instruct (free) by meta-llama</a>：Meta 最新的模型系列 (Llama 3.1) 推出了多种尺寸和版本。这个 8B 指令微调版本快速且高效。与...相比，它展示了强大的性能。</li><li><a href="https://openrouter.ai/models/nothingiisreal/mn-celeste-12b">Mistral Nemo 12B Celeste by nothingiisreal</a>：基于 Mistral 的 NeMo 12B Instruct 的专业故事写作和角色扮演模型。在包括 Reddit Writing Prompts 和 Opus Instruct 25K 在内的精选数据集上进行了微调。该模型擅长...</li><li><a href="https://openrouter.ai/models/perplexity/llama-3.1-sonar-large-128k-online">Perplexity: Llama 3.1 Sonar 70B Online by perplexity</a>：Llama 3.1 Sonar 是 Perplexity 最新的模型系列。它在成本效益、速度和性能方面超越了早期的 Sonar 模型。这是 [离线聊天模型](/m... 的在线版本。</li><li><a href="https://openrouter.ai/models/perplexity/llama-3.1-sonar-small-128k-online">Perplexity: Llama 3.1 Sonar 8B Online by perplexity</a>：Llama 3.1 Sonar 是 Perplexity 最新的模型系列。它在成本效益、速度和性能方面超越了早期的 Sonar 模型。这是 [离线聊天模型](/m... 的在线版本。
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1268969204309168219)** (1 条消息): 

> - `API Key 获取`
> - `使用自有 API Key 的好处`
> - `免费计划限制`
> - `Google Sheets 插件` 


- **获取 API Key 非常简单**：获取 API Key 的过程非常直接；只需在 AI 提供商的网站上注册，充值额度，复制 API Key，然后将其粘贴到插件中即可，无需任何技术技能。点击[此处](https://help.aiassistworks.com/help/how-easy-it-is-to-get-an-api-key)了解更多流程。
- **为什么使用自有 API Key 很重要**：使用自己的 API Key 不仅能获得最优惠的价格，还能确保选择 AI 提供商的灵活性。例如，**GPT-4o-mini** 的价格仅为 **每 1,000,000 tokens 0.6 美元**。
   - 此外，你可以在 AI 提供商的控制面板中透明地查看模型使用情况。
- **Lite 计划：免费但有限制**：Lite 计划对低用量用户永久免费，目前限制为 **每月 300 次结果**。值得注意的是，**1 个单元格结果** 计为 **1 次结果**，**1 次分析** 计为 **5 次结果**。
   - 需要注意的是，此限制将来可能会发生变化。
- **通过 Google Sheets 插件获得 1 年免费使用权**：在结账时使用代码 **LAUNCH** 并选择按年计费，同时使用与 Google Sheets 相同的电子邮件地址，即可获得插件的 **1 年免费** 优惠。对于新用户来说，这是一个以更低成本体验服务的绝佳方式。



**提到的链接**：<a href="https://www.aiassistworks.com/">AiAssistWorks - Google Sheets™ AI 插件 - GPT- Claude - Gemini - Llama, Mistral, OpenRouter ,Groq. </a>：未找到描述

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1268697141921185884)** (58 条消息🔥🔥): 

> - `OpenRouter 网站问题`
> - `Anthropic 服务问题`
> - `OR Playground 中的群聊功能`
> - `Yi Large 可用性`
> - `免费模型使用限制` 


- **OpenRouter 网站访问性**：一位成员确认 OpenRouter 网站对他来说是正常的，但区域性连接问题很常见，正如[此处](https://status.openrouter.ai/)显示的过往事件。另一位用户报告说，网站问题随后很快就解决了。
   - 这突显了局部故障影响用户体验的可能性。
- **Anthropic 面临服务问题**：多位用户报告 Anthropic 服务似乎已宕机或在**严重负载**下挣扎，表明可能存在基础设施问题。一位用户指出，服务已经断断续续持续了几个小时。
   - 对于依赖其服务的用户来说，这似乎是一个日益严重的问题。
- **关于 OR Playground 群聊功能的澄清**：一位用户尝试使用 Llama3 设置“作家室”，成员对此澄清说，OR Playground 中的每个模型都在隔离的内存中运行，与传统的群聊不同。暗示未来会进行改进，允许模型按顺序响应。
   - 当前的设置旨在比较不同模型在相同提示词下的输出。
- **Yi Large 和 Fireworks 的可用性**：一位成员询问了 Yi Large 的状态，另一位成员表示他们正在探索从原作者的托管方添加该模型。此外还提到 Fireworks 已被移除。
   - 这表明平台上可用的模型正在进行持续调整。
- **了解免费模型限制**：随后讨论了提供免费模型的含义；澄清了免费模型在通过 API 或聊天室使用时受到严格的速率限制（Rate-limited）。这种限制对于管理服务器负载和确保用户公平访问至关重要。
   - 这些约束对于维持服务的可行性非常重要。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1819500533553443004">来自 OpenRouter (@OpenRouterAI) 的推文</a>：Llama 3.1 405B BASE！它来了。这是上周发布的聊天模型的基座版本。你可以用它来生成训练数据、代码补全等。目前由一个新的专业托管方提供……</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-405b">Meta: Llama 3.1 405B (base) by meta-llama</a>：Meta 最新的模型系列 (Llama 3.1) 推出了多种尺寸和版本。这是 405B 预训练基座版本。与领先的闭源模型相比，它展示了强大的性能……</li><li><a href="https://status.openrouter.ai/">OpenRouter 状态</a>：OpenRouter 故障历史记录
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1268726560458932377)** (3 messages): 

> - `RAG Pipeline`
> - `AI Voice Agent for Farmers`
> - `ReAct Agents` 


- **事件驱动的 RAG Pipeline 教程发布**：分享了一个关于如何构建 **RAG pipeline** 的教程，将检索（retrieval）、重排序（reranking）和合成（synthesis）分解为利用事件的三个不同步骤。进一步的编排逻辑可以使用 [@llama_index workflows](https://t.co/XGmm6gQhcI) 进行管理。
   - 该教程旨在指导用户从头开始构建 RAG pipeline，为事件驱动架构提供了一种全面的方法。
- **为印度农民开发的 AI Voice Agent**：为了支持缺乏必要政府援助的印度农民，@SamarthP6 及其团队开发了一个 **AI Voice Agent**。正如[这条推文](https://t.co/lrGDFSl0HH)所强调的，该工具旨在提高生产力并在面临挑战时改善生计成果。
   - 该倡议旨在弥合农民与关键资源之间的差距，展示了技术如何解决农业问题。
- **使用新工作流构建 ReAct Agent**：提供了一个使用更新的 LlamaIndex workflows 从头开始构建 **ReAct agent** 的新资源。这允许用户更详细地探索 Agent 的内部逻辑并理解系统动态，详见此处分享的链接 [here](https://t.co/F0pPEyWJ2w)。
   - **ReAct agents** 是 Agentic 系统中的关键组件，使得这些工作流对于寻求创新的开发者特别有用。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1268670221556776962)** (31 messages🔥): 

> - `ReAct Agent without Tools`
> - `Service Context Changes in LlamaIndex`
> - `Using WhatsApp Data for Chatbot Training`
> - `RAG Pipeline for Data Interaction` 


- **不带工具的 ReAct Agent 策略**：一位用户寻求关于如何配置 **ReAct agent** 不请求任何工具的建议，并提出了替代方案，如使用 `llm.chat(chat_messages)` 或 `SimpleChatEngine` 以获得更直接的聊天体验。
   - 另一位成员指出了 Agent 错误的挑战，特别是如何优雅地管理“找不到工具”等问题。
- **LlamaIndex 中 Service Context 的变更**：成员们讨论了在 LlamaIndex 未来版本中移除 **service context** 的问题，强调需要更灵活的方法来设置 `max_input_size` 和 `chunk overlap` 等参数。
   - 一位用户对由于转向需要将更多个性化组件传递给基础 API 而必须重写代码表示沮丧。
- **使用 WhatsApp 聊天数据进行聊天机器人开发**：一位新手探索利用其 **WhatsApp** 商业聊天记录来创建聊天机器人，寻找在保留有用客户互动的同时清理和格式化数据的方法。
   - 他们提到在处理大型 PDF 文件时遇到错误，并寻求关于数据提取和查询准确性最佳实践的指导。
- **用于高效聊天数据管理的 RAG Pipeline**：一位用户分享了实现 **RAG pipeline** 的结构化方法，包括数据组织、分块策略以及使用 **LlamaIndex** 进行索引以查询客户互动。
   - 他们建议对大文件进行增量处理，并推荐使用 **UnstructuredReader** 等工具解析聊天日志并将其存储在 **VectorStoreIndex** 中。


  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1268804305146613791)** (3 messages): 

> - `DSPy integration issues`
> - `Fine-tuning vs. RAG` 


- **DSPy 最新更新破坏了 LlamaIndex 集成**：一位成员报告称 **DSPy** 的最新更新破坏了与 **LlamaIndex** 的集成，并表示他们*到目前为止在 DSPy 中还没有成功的用例*。
   - 他们指出，在之前的版本 **v2.4.11** 中，与使用原生的 LlamaIndex 抽象相比，通过 prompt **finetuning** 并没有观察到明显的改进。
- **AI 开发中 Fine-tuning 的未来**：一位成员提出了一个关于在 **Llama 3.1** 等快速发展的 LLM 面前，**fine-tuning** 是否必要的问题，因为这些模型对于大多数用例可能已经足够了。
   - 他们强调，考虑到资源需求，**RAG** 系统可能是增强模型知识的一种更高效的替代方案。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1268683741883007046)** (20 messages🔥): 

> - `Mojo 错误处理`
> - `Python vs Go/Rust 错误模式`
> - `分布式 Actor 框架` 


- **Mojo 的错误处理困境**：成员们讨论了围绕 **Mojo** 错误处理的困境，比较了 **Python 风格的异常 (exceptions)** 和 **Go/Rust 的错误值 (error values)**，同时指出 Mojo 的目标是成为 **Python 超集**。
   - 有人担心同时使用这两种模式可能会导致复杂性，甚至可能给程序员带来严重后果。
- **异常处理的开销**：讨论中提出了抛出和捕获异常的开销问题，**darkmatter__** 表示纯 Mojo 库除非必要，否则应避免调用此类操作。
   - 他强调了在 **FPGA** 上有效实现这一点的挑战，因为这可能会大幅增加资源消耗。
- **分享的引用与参考**：***yolo007*** 分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=Iflu9zEJipQ)，其中 **Chris Lattner** 讨论了编程语言中 *异常与错误* 的区别。
   - 该视频引发了关于 **Elixir** 错误处理以及它如何与 **Mojo** 方法交集的讨论。
- **Erlang 的 'Let it Crash' 哲学**：提到了来自 **Erlang** 的 **'Let it crash'** 概念作为一种可能的设计哲学，即不受控制的错误会传播到更高层进行处理。
   - 有人指出，这种哲学可能与 **Mojo** 的愿景不符，特别是在构建健壮的分布式 **Actor** 框架方面。
- **错误处理的灵活性**：成员们注意到 **Mojo** 可以同时利用异常和返回错误处理，允许通过 **try/catch** 语法灵活地捕获错误。
   - ***yolo007*** 提到更倾向于将代码包装在 **try/catch** 结构中，以避免显式管理所有可能的错误，这表明了一种更精简的代码风格。



**提到的链接**：<a href="https://www.youtube.com/watch?v=Iflu9zEJipQ">Exception vs Errors | Chris Lattner and Lex Fridman</a>：Lex Fridman Podcast 完整集：https://www.youtube.com/watch?v=pdJQ8iVTwj8 请通过查看我们的赞助商来支持此播客：- iHerb: https://lexfri...

  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1268815039041765460)** (3 messages): 

> - `安装问题`
> - `Mojo Nightly 贡献`
> - `Conda 安装建议` 


- **Max 的安装问题**：一位成员表达了安装 **Max** 的困难，指出在尝试运行代码时遇到了问题。
   - 他们提到安装过程可能会出现问题，并正在寻求帮助。
- **Mojo Nightly 运行顺畅**：对于一位正在积极贡献的成员来说，**Mojo nightly** 运行良好。
   - 这表明虽然 **Max** 存在问题，但 **Mojo nightly** 依然保持稳定且可运行。
- **推荐使用 Conda 安装**：另一位成员建议使用 **Conda** 作为安装问题的潜在解决方案。
   - 他们指出最近安装过程已大幅简化，这可能有助于解决所面临的问题。


  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1268649063436517497)** (15 messages🔥): 

> - `Open Interpreter setup`
> - `Using local LLMs`
> - `API configuration for LLMs`
> - `Python development with LlamaFile`
> - `Community engagement` 


- **Open Interpreter 会话混淆**：成员们在加入正在进行的会话时感到困惑，随后澄清了对话是在特定的语音频道中进行的。
   - *一位成员提到，在其他人确认频道可用之前，他们一直难以找到该频道*。
- **在 Open Interpreter 中运行本地 LLM**：一名新成员寻求运行本地 LLM 的指导，并分享了导致模型加载错误的初始脚本。
   - 社区成员引导他们参考 [documentation](https://docs.openinterpreter.com/language-models/local-models/llamafile#llamafile) 以正确设置本地模型。
- **为 Python 模式启动 LlamaFile 服务器**：会议强调，在 Python 模式下使用 LlamaFile 之前，必须单独启动 LlamaFile 服务器。
   - 参与者确认了 API 和模型设置的正确语法，并澄清了不同加载函数之间的区别。
- **Python 模式开发目标**：新成员表示有兴趣使用 Open Interpreter 的 Python 模式开发一个一键助手。
   - 他们表达了之前开发基础 Chatbot 的经验，并渴望探索 LlamaFile 提供的各种新功能。



**提及的链接**：<a href="https://docs.openinterpreter.com/language-models/local-models/llamafile#llamafile">LlamaFile - Open Interpreter</a>：未找到描述

  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1268768591382188062)** (2 messages): 

> - `Stripe Payment Receipts`
> - `Shipping Address Inquiries` 


- **Stripe 收据上缺少收货地址**：一位用户反映在 **3 月 21 日的 Stripe** 支付收据上没有看到 **收货地址 (shipping address)**。
   - 他们询问解决此问题的后续步骤。
- **用户的后续步骤**：另一位成员回复称，目前用户端 **无需采取任何行动**。
   - 他们确认 **OpenInterpreter** 将会主动联系并告知相应的后续步骤。


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1268884909725061151)** (2 messages): 

> - `Aider browser UI`
> - `Post-facto validation with LLMs` 


- **用于代码编辑的 Aider 浏览器 UI**：新的 [Aider 浏览器 UI 演示视频](https://aider.chat/docs/usage/browser.html) 展示了如何与 LLMs 协作编辑本地 Git 仓库中的代码，并自动提交带有合理信息的 Commit。
   - 它支持 GPT 3.5, GPT-4, GPT-4 Turbo with Vision 以及 Claude 3 Opus，安装后可以使用 `--browser` 参数启动。
- **LLM 应用中的事后验证 (Post-facto validation)**：研究强调，随着 LLMs 的扩展，人类目前在执行前验证 **LLM 生成的输出**，但由于代码理解困难而面临挑战。建议集成 **undo**（撤销）功能并建立 **damage confinement**（损害限制）机制，以促进更容易的 LLM 行为事后验证 [更多详情请点击此处](https://gorilla.cs.berkeley.edu/blogs/10_gorilla_exec_engine.html)。
   - 该研究认为，事后验证（在输出生成后进行验证）通常比事前验证更容易管理。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/usage/browser.html">Aider in your browser</a>：Aider 可以在浏览器中运行，而不仅仅是在命令行中。</li><li><a href="https://gorilla.cs.berkeley.edu/blogs/10_gorilla_exec_engine.html">Gorilla Execution Engine</a>：未找到描述
</li>
</ul>

</div>

### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1268717632018972722)** (3 条消息): 

> - `LLM 中的 Meta-Rewarding 机制`
> - `用于信息集成的 MindSearch` 


- **LLM 提升其判断技能**：最近的一篇论文讨论了 LLM 中的 **Meta-Rewarding** 步骤，该步骤允许模型对自己的判断进行评判，从而提升能力，并使 **Llama-3-8B-Instruct** 在 AlpacaEval 2 上的胜率从 **22.9%** 提升至 **39.4%**。
   - 这种方法通过增强模型的**自我评判（self-judgment）**能力，解决了传统方法论中的饱和问题。
- **MindSearch 模拟人类认知过程**：另一篇论文介绍了 **MindSearch**，这是一个旨在通过使用 LLM 和多 Agent 系统（multi-agent systems）模拟人类认知，从而克服信息寻求和集成挑战的框架。
   - 它解决了**复杂请求**无法被准确检索以及 LLM 的**上下文长度（context length）**限制等问题，同时能够有效地聚合相关信息。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2407.20183">MindSearch: Mimicking Human Minds Elicits Deep AI Searcher</a>：信息寻求与集成是一项耗费大量时间和精力的复杂认知任务。受 Large Language Models 显著进展的启发，最近的研究尝试解决这一问题...</li><li><a href="https://arxiv.org/abs/2407.19594">Meta-Rewarding Language Models: Self-Improving Alignment with LLM-as-a-Meta-Judge</a>：Large Language Models (LLMs) 在许多领域正迅速超越人类知识。虽然改进这些模型传统上依赖于昂贵的人类数据，但最近的自我奖励机制（Yuan 等人...
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1268670288162324481)** (13 条消息🔥): 

> - `DSPy 摘要流水线`
> - `Discord 频道导出`
> - `用于游戏开发的 AI`
> - `可重复的分析工具`
> - `巡逻 AI 角色` 


- **构建 DSPy 摘要流水线**：一位成员正在寻求关于如何将 DSPy 与开源模型结合用于摘要任务的指导，并表示希望能有相关教程。
   - 目标是通过迭代增强 Prompt 的有效性，以获得更好的摘要输出。
- **请求 Discord 频道导出数据**：另一位成员正在寻找志愿者，分享以 JSON 或 HTML 格式导出的 Discord 频道数据，用于分析目的。
   - 他们表示打算在发布研究结果和代码时向贡献者致谢。
- **用于 OSS 项目的通用分析工具**：讨论了开发一个文档齐全、可重复的分析工具，专门为开源软件（OSS）项目量身定制。
   - 该工具旨在整合来自 GitHub、Discord 和其他平台的各种日志和 Issue。
- **游戏角色开发的 AI 集成**：一位成员收到了一个关于游戏角色 AI 实现的 GitHub 链接，特别是针对巡逻和玩家交互。
   - 他们的目标是利用 Oobabooga API 来促进 LLM 的响应，从而实现动态角色对话。
- **游戏中不同玩家类型的库**：结合 AI 角色对话，同一位成员表示有兴趣创建一个包含各种玩家类型的库。
   - 这将根据玩家的距离和聊天活跃度来增强角色互动。

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1268729449172434984)** (7 条消息): 

> - `Fine-tuning Gemma2 2B`
> - `Model fluency in Japanese`
> - `BitsAndBytes installation for ROCm` 


- **探索微调 Gemma2 2B**：一名成员询问了关于**微调 Gemma2 2B** 模型的尝试，寻求他人的见解。
   - 另一名成员建议使用**预分词数据集 (pretokenized dataset)** 并调整输出标签来控制模型行为。
- **寻找最流畅的日语模型**：一名成员询问是否有**日语母语者**可以推荐目前最流畅的模型。
   - 根据他人的集体推荐，有人建议使用 **lightblue 的 suzume 模型**。
- **ROCm 上 BitsAndBytes 的简易安装**：一名成员分享道，由于最近的一个 [GitHub pull request](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1299)，在 ROCm 上安装 **BitsAndBytes** 的过程变得更加简单。
   - 此次更新支持为 **ROCm 6.1** 封装 wheel 文件，使其能够兼容最新的 Instinct 和 Radeon GPU，并指出 **tracker 上的所有内容现已完成**。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/bitsandbytes">bitsandbytes - 概览</a>：bitsandbytes 有 6 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1299">由 pnunna93 提交的 PR #1299：为 ROCm 启用 bitsandbytes 封装</a>：此 PR 启用了 ROCm 上 bitsandbytes 的 wheel 封装。它更新了 ROCm 编译和 wheel 构建任务，以便在 ROCm 6.1 上为最新的 Instinct 和 Radeon GPU 进行编译。此外还有一些更新...
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1268729202707009577)** (5 条消息): 

> - `Merged PR`
> - `KD development`
> - `adam-atan2 update`
> - `distilkit release` 


- **合并 PR 后运行顺畅**：一名成员指出，在合并特定的 PR 后，一切运行良好。
   - 遗憾的是，由于已经合并，他们无法轻易复现之前的问题。
- **KD 开发仍在队列中**：一名成员因故暂时离开表示歉意，并提到他们用于开发 **KD** 的时间被占用了。
   - 他们确认这仍然是优先级事项，并欢迎任何感兴趣的人在此期间接手处理。
- **关于 adam-atan2 的有趣调整**：一名成员分享了一篇讨论 **adam-atan2** 的论文，这是一种避免除以零的优雅修复方案。
   - 他们提供了论文链接以供进一步阅读：[adam-atan2 论文](https://arxiv.org/pdf/2407.05872)。
- **对 distilkit 发布的兴奋**：一名成员提到 arcee-ai 发布的 **distilkit** 没有问题，并对其功能表示热忱。
   - 这个新工具看起来很有前景，大家正渴望探索它的能力。


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1268827508191531111)** (4 条消息): 

> - `Training Gemma2`
> - `Llama3.1 Template Challenges`
> - `Output Termination Issues`
> - `Prompt Engineering`
> - `Data Sufficiency for Training` 


- **训练 Gemma2 和 Llama3.1 面临挑战**：用户分享了在旧模板上训练 **Gemma2** 和 **Llama3.1** 的经验，指出模型虽然学会了结构，但在输出终止方面存在困难。
   - 他们观察到模型会持续生成内容，直到达到 **max_new_tokens**，或者在回复后退回到默认行为。
- **提示词工程对输出影响微乎其微**：在尝试引导模型输出时，用户加入了严格的提示词指令（Prompt Engineering），要求除了回复外不输出任何内容。
   - 然而，据报告这些调整对模型的整体行为仅产生了**极小的影响**。
- **对训练时长与输出变化的担忧**：用户对训练所需的时间表示担忧，不愿在输出没有实质性改善的情况下训练数日。
   - 他们推测提供更多示例可能有助于模型学习何时终止输出，但对所需的投入持谨慎态度。
- **非传统模板结构带来挑战**：讨论强调了使用 **input-output** 结构构建模板与传统聊天模板相比所面临的挑战。
   - 用户指出，这种独特的模板结构使训练过程变得复杂，加剧了输出生成的问题。

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1268645705157705738)** (10 messages🔥): 

> - `LangChain v0.2 features`
> - `Chat sessions in RAG applications`
> - `Chat message history with Postgres`
> - `Fine-tuning models for summarization`
> - `Performance comparison of GPT-4o Mini and GPT-4` 


- **关于 LangChain v0.2 特性的咨询**：用户讨论了新版本 **LangChain 0.2** 中关于 **Agent 功能** 文档缺失的问题。
   - *Orlando.mbaa* 特别指出，他们在现有文档中找不到任何关于 Agent 的参考内容。
- **在 RAG 应用中实现聊天会话**：一位用户询问如何在基础的 **RAG 应用** 中实现 **聊天会话 (Chat sessions)**，类似于 ChatGPT 跟踪历史对话的能力。
   - 对话围绕现有框架内会话跟踪的可用性展开。
- **LangChain 与 Postgres Schema 的问题**：一名成员引用了一个 GitHub Issue ([#17306](https://github.com/langchain-ai/langchain/issues/17306))，涉及在使用显式 Schema 时，**Postgres** 中的 **聊天消息历史 (Chat message history)** 失败的问题。
   - 他们表达了对该问题如何解决的关注，并分享了相关资源。
- **关于摘要任务微调的讨论**：一位拥有**大规模文本语料库**的用户咨询了针对摘要任务进行模型 **Fine-tuning** 的方法。
   - 另一位成员指出，对于摘要任务通常不需要进行 **Fine-tuning**，但该用户坚持希望在摘要中融入特定数据。
- **性能对比：GPT-4o Mini vs GPT-4**：有人询问 **GPT-4o Mini** 相较于 **GPT-4** 的性能表现，寻求关于它们能力的见解。
   - 这一咨询凸显了人们对了解不同 AI 模型效率的持续兴趣。



**提到的链接**：<a href="https://github.com/langchain-ai/langchain/issues/17306">Chat message history with postgres failing when destination table has explicit schema · Issue #17306 · langchain-ai/langchain</a>：检查了其他资源，我为此 Issue 添加了非常详细的标题。我通过集成搜索查询了 LangChain 文档，并使用 GitHub 搜索寻找类似问题...

  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1268991920416358420)** (1 messages): 

> - `Community Research Call #2`
> - `Multimodality updates`
> - `Autonomous Agents developments`
> - `Robotics projects`
> - `Collaboration opportunities` 


- **社区研究电话会议 #2 亮点**：最近的 [Community Research Call #2](https://x.com/ManifoldRG/status/1819430033993412856) 展示了多个研究项目的突破性更新。
   - 与会者对 **Multimodality**、**Autonomous Agents** 以及新 **Robotics** 项目的进展感到兴奋。
- **分享令人兴奋的合作机会**：在活动期间，参与者讨论了多个正在进行的研究方向中的 **合作机会 (Collaboration opportunities)**。
   - 现场气氛热烈，成员们为未来项目中潜在的共同努力和伙伴关系出谋划策。



**提到的链接**：<a href="https://x.com/ManifoldRG/status/1819430033993412856">来自 Manifold Research (@ManifoldRG) 的推文</a>：Community Research Call #2 非常成功！我们分享了关于 Multimodality 和 Autonomous Agents 方向的突破性更新，并揭晓了我们在 Robotics 领域的新项目。

  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1268864868132458537)** (1 messages): 

> - `Testing LLMs`
> - `Testcontainers`
> - `Ollama`
> - `Python Blog Post` 


- **关于测试 LLM 的博客文章**：一位成员撰写了一篇博客，详细介绍了如何在 Python 中使用 [Testcontainers](https://testcontainers.com/) 和 **Ollama** 测试 **LLM**，特别是利用了 **4.7.0 版本** 中的模块功能。
   - 他们强调了测试 **LLM** 以确保其在各种条件下按预期运行的重要性，并邀请大家对分享在 [这里](https://bricefotzo.medium.com/testing-llms-and-prompts-using-testcontainers-and-ollama-in-python-81e8f7c18be7) 的教程提供反馈。
- **自制的测试框架插图**：分享了一张插图，展示了在提议的测试框架中 **Testcontainers** 如何与 **Docker**、**Ollama** 以及最终的 **LLM** 进行交互。
   - 该视觉表示旨在阐明 Python 环境中用于测试的各种工具之间的关系。



**提到的链接**：<a href="https://bricefotzo.medium.com/testing-llms-and-prompts-using-testcontainers-and-ollama-in-python-81e8f7c18be7">Testing LLMs and Prompts using Testcontainers and Ollama in Python</a>：一个使用 Python、Testcontainers 和 Ollama 的易于使用的 LLM 与 Prompt 测试框架。

  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1268929430357540994)** (12 条消息🔥): 

> - `QAT Quantizers`
> - `SimPO PR Review`
> - `Documentation Improvement`
> - `New Models Page Feedback` 


- **关于 QAT Quantizers 的澄清**：成员们澄清了 QAT recipe 支持 **Int8DynActInt4WeightQATQuantizer**，而 **Int8DynActInt4WeightQuantizer** 是为 post-training 设计的。
   - 他们指出目前 QAT 仅支持 **Int8DynActInt4Weight** 策略，其他量化器供未来使用。
- **请求 SimPO PR Review**：一位成员请求对 GitHub 上的 **SimPO (Simple Preference Optimisation)** PR #1223 进行 Review，并强调需要明确其用途。
   - 他们强调该 PR 解决了与 alignment 相关的 issue #1037 并关闭了 #1036。
- **文档重构的 RFC**：另一位成员讨论了一项关于重构 **torchtune** 文档系统的提案，特别是侧重于 recipe 的组织。
   - 他们征求反馈以增强用户入门体验，并参考了 **LoRA single device** 和 **QAT distributed** recipes 的示例。
- **对潜在新模型页面的反馈**：一位成员分享了 torchtune 文档网站上潜在的**新模型页面**预览链接以获取反馈。
   - 他们指出目前的模型页面难以阅读，并提到新页面包含了关于模型架构的详细信息。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/torchtune/stable/generated/torchtune.utils.get_quantizer_mode.html#torchtune.utils.get_quantizer_mode">get_quantizer_mode &mdash; torchtune 0.2 documentation</a>：未找到描述</li><li><a href="https://docs-preview.pytorch.org/pytorch/torchtune/954/api_ref_models.html">torchtune.models &mdash; TorchTune main documentation</a>：未找到描述</li><li><a href="https://github.com/pytorch/torchtune/pull/1230.">Build software better, together</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/pull/1223">SimPO (Simple Preference Optimisation) by SalmanMohammadi · Pull Request #1223 · pytorch/torchtune</a>：上下文：此 PR 的目的是什么？是添加新功能、修复 bug、更新测试和/或文档还是其他（请在此处添加）（解决 #1037，关闭 #1036）另一个 alignment PR??? T...
</li>
</ul>

</div>
  

---



### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1268705341185724559)** (6 条消息): 

> - `Computer Vision Interest`
> - `Conferences on Machine Learning`
> - `ROI of genAI`
> - `Funding Trends`
> - `Discussion Diversification` 


- **对 Computer Vision 的热情**：成员们表达了对 **computer vision** 的共同兴趣，强调了其在当前技术版图中的重要性。
   - *许多成员似乎渴望从占据会议主导地位的 NLP 和 genAI 讨论中分流出来。*
- **会议反映机器学习趋势**：一位成员分享了参加两次 **machine learning conferences** 的经历，会上展示了他们在 **Gaussian Processes** 和 **Isolation Forest** 模型方面的工作。
   - *他们注意到许多与会者对这些主题并不熟悉，这表明讨论存在向 **NLP** 和 **genAI** 倾斜的强烈偏好。*
- **对 genAI ROI 的怀疑**：参与者质疑 **genAI** 的**投资回报率 (ROI)** 是否能达到预期，表明可能存在脱节。
   - *一位成员强调，正向的 ROI 需要初始投资，这表明预算通常是根据感知价值分配的。*
- **资金重点影响讨论**：一位成员指出，**资金**通常流向**预算**分配的地方，从而影响技术讨论。
   - *这强调了市场细分和 hype cycles 在塑造行业活动重点方面的重要性。*
- **对更广泛对话的渴望**：鉴于目前的讨论情况，一位成员表示很高兴能有一个平台来讨论 **genAI** 热潮之外的话题。
   - *这反映了对涵盖 **machine learning** 各个领域（而非仅限于主流趋势）的**多样化对话**的渴望。*


  

---

### **Alignment Lab AI ▷ #[general](https://discord.com/channels/1087862276448595968/1095458248712265841/1268863588286730240)** (1 条消息): 

> - `Image generation time on A100`
> - `Batch processing capabilities with FLUX Schnell` 


- **使用 FLUX Schnell 在 A100 上的图像生成时间**：有人询问了在 **A100** 上使用 **FLUX Schnell** 生成 **1024px 图像**所需的时间，强调了对性能的预期。
   - 讨论中未提供具体时长。
- **讨论了批处理能力**：有人提出了关于图像生成是否可以进行 **batch processing**（批处理）以及可处理的最大图像数量的问题。
   - 消息中未分享与硬件能力相关的回复。


  

---



---



---



---



---



{% else %}


> 为了便于邮件发送，完整的频道细分内容已被截断。
> 
> 如果您想查看完整的细分内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}