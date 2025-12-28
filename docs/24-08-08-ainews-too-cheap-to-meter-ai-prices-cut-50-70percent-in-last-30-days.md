---
companies:
- llamaindex
- together-ai
- deepinfra
- deepseek-ai
- mistral-ai
- google-deepmind
- lg-ai-research
- llamaindex
- llamaindex
- llamaindex
date: '2024-08-09T04:27:56.926040Z'
description: '**Gemini 1.5 Flash** 降价约 **70%**，并提供极具竞争力的免费层级（每分钟 **100 万 token**），价格低至
  **$0.075/百万 token**，进一步加剧了 AI 模型的价格战。其他显著降价的模型还包括：**GPT-4o**（降价约 50%，至 **$2.50/百万
  token**）、**GPT-4o mini**（降价 70-98.5%，至 **$0.15/百万 token**）、**Llama 3.1 405b**（降价
  46%，至 **$2.7/百万 token**）以及 **Mistral Large 2**（降价 62%，至 **$3/百万 token**）。**Deepseek
  v2** 引入了上下文缓存（context caching），将输入 token 成本降低了高达 **90%**，降至 **$0.014/百万 token**。


  新发布的模型包括 **Llama 3.1 405b**、**Sonnet 3.5**、**EXAONE-3.0**（由 LG AI Research 开发的 7.8B
  指令微调模型）以及 **MiniCPM V 2.6**（结合了 SigLIP 400M 和 Qwen2-7B 的视觉语言模型）。基准测试显示，**Mistral
  Large** 在 ZebraLogic 上表现优异，而 **Claude-3.5** 在 LiveBench 中处于领先地位。**FlexAttention**
  作为一个新的 PyTorch API，简化并优化了注意力机制。**Andrej Karpathy** 对 RLHF（人类反馈强化学习）进行了分析，指出了其与传统强化学习相比的局限性。此外，文中还总结了
  Google DeepMind 关于计算最优扩展（compute-optimal scaling）的研究。'
id: 7d8d0c98-cad0-44de-a76d-4882abd5a690
models:
- gpt-4o
- gpt-4o-mini
- llama-3-1-405b
- mistral-large-2
- gemini-1.5-flash
- deepseek-v2
- sonnet-3.5
- exaone-3.0
- minicpm-v-2.6
- claude-3.5
- gpt-4o-2024-08-06
original_slug: ainews-too-cheap-to-meter-ai-prices-cut-50-70-in
people:
- rohanpaul_ai
- akhaliq
- mervenoyann
- sophiamyang
- chhillee
- karpathy
title: '**便宜到无需计费：过去 30 天 AI 价格下调 50-70%**'
topics:
- price-cuts
- context-caching
- instruction-tuning
- vision
- benchmarks
- pytorch
- attention-mechanisms
- reinforcement-learning-from-human-feedback
- compute-optimal-scaling
---

<!-- buttondown-editor-mode: plaintext -->**Gemini Flash 就足够了吗？**

> 2024年8月7日至8月8日的 AI 新闻。我们为你检查了 7 个 subreddits、[**384** 个 Twitters](https://twitter.com/i/lists/1585430245762441216) 和 **28** 个 Discords（**249** 个频道和 **2423** 条消息）。预计节省阅读时间（按 200wpm 计算）：**247 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

一份过去 30 天内 **AI 领域所有降价情况**的简单列表（以 "mtok" 即 "每百万 token" 衡量——大部分成本通常是输入），按 LMsys Elo/排名排序：

- **Elo 1286 Rank 2**: [GPT-4o 从 5 月到 8 月降价约 50%](https://buttondown.com/ainews/archive/ainews-gpt4o-august-100-structured-outputs-for/) (**$2.50/mtok**)
- **Elo 1277 Rank 3**: [GPT-4o mini](https://buttondown.com/ainews/archive/ainews-lskjd/) 实际上**降价了 70-98.5%**，取决于你是与 GPT3.5T 还是 GPT4T 比较 (**$0.15/mtok**)
- **Elo 1264 Rank 4**: [Llama 3.1 405b](https://buttondown.email/ainews/archive/ainews-llama-31-the-synthetic-data-model/) 最初由 Together AI 提供 $5/15 的价格——在 48 小时内被 DeepInfra [降价 46% 至 $2.7/mtok](https://x.com/openrouterai/status/1816234833896694270?s=46)，[Lepton 紧随其后](https://x.com/jiayq/status/1816246934925107393?s=46) (**$2.7/mtok**) 
- **Elo 1249 Rank 8**: [Mistral Large 2](https://buttondown.email/ainews/archive/ainews-mistral-large-2/) 相比 [2 月的 Large v1](https://x.com/gblazex/status/1762127672673468566) 降价了 62% (**$3/mtok**)
- **Elo 1228 Rank 17**: [Gemini 1.5 Flash 降价约 70%](https://x.com/OfficialLoganK/status/1821601298195878323) —— 此外还有其现有的 [每分钟 100 万 token 的免费层级](https://reddit.com//r/LocalLLaMA/comments/1em9545/best_summarizing_llms_for_average_pcs/?utm_source=ainews&utm_medium=email) (**$0.075/mtok**)
- **Elo 1213 Rank 17**: [Deepseek v2 在 Context Caching 的 GA 发布上击败了 Gemini](https://x.com/rohanpaul_ai/status/1820833952149487898)，将缓存命中的输入 token 价格最高降低了 90% (**$0.014/mtok**（没写错）)。这是在他们[最初的 $0.14/mtok 定价之后，该定价可能引发了上个月的价格战](https://x.com/EMostaque/status/1813991810823340521)

鉴于 Gemini 1.5 极其慷慨的免费层级，LMsys 排名 17 以下的所有模型——目前包括 Gemma 2, Nemotron 4, GLM 4, Reka Flash, Llama 3 7b, Qwen 72B 等——对于大多数个人和团队使用场景来说，实际上是一发布就过时了。

[价格-智能前沿（Price-Intelligence frontier）](https://x.com/swyx/status/1815892458519289946)在又一个季度中再次推进了一个数量级。



---

{% if medium == 'web' %}

**目录**

[TOC]

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}

---

# AI Twitter 综述

> 所有综述均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型开发与发布**

- **新模型与新功能**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1821214544284282947) 报道了 Llama3.1 405b 和 Sonnet 3.5 的发布，用户可以通过 Google Cloud 的 300 美元赠金免费使用。[@_akhaliq](https://twitter.com/_akhaliq/status/1821327180497842205) 宣布了来自 LG AI Research 的 EXAONE-3.0，这是一个 7.8B 的指令微调模型，在同尺寸的最先进开源模型中表现出极具竞争力的性能。[@mervenoyann](https://twitter.com/mervenoyann/status/1821103721213722683) 重点介绍了 MiniCPM V 2.6，这是一款结合了 SigLIP 400M 和 Qwen2-7B 的视觉语言模型，在多项基准测试中超越了专有模型。

- **模型性能与基准测试**：[@sophiamyang](https://twitter.com/sophiamyang/status/1821119082432712938) 指出，尽管 Mistral Large 比其他模型尺寸更小，但在 ZebraLogic 基准测试中表现出色。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1821326039714246902) 分享道，在针对新版 GPT-4o-2024-08-06 的 LiveBench 基准测试中，Claude-3.5 依然稳居榜首。

- **AI 工具与框架**：[@cHHillee](https://twitter.com/cHHillee/status/1821253769147118004) 介绍了 FlexAttention，这是一个新的 PyTorch API，允许仅用几行 PyTorch 代码就能让多种 Attention 变体享受融合算子（fused kernels）的加速。该开发旨在简化和优化神经网络中的各种 Attention 机制。

**AI 研究与洞察**

- **RLHF 与模型训练**：[@karpathy](https://twitter.com/karpathy/status/1821277264996352246) 对来自人类反馈的强化学习（RLHF）进行了深入分析，讨论了其局限性并将其与传统强化学习进行了比较。他认为 RLHF“仅仅勉强算是强化学习”，并强调了将其应用于大语言模型时的挑战。

- **计算最优扩展（Compute-Optimal Scaling）**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1821314621417922949) 总结了 Google DeepMind 关于大语言模型推理时计算（test-time computation）的计算最优扩展论文。该研究引入了根据提示词难度自适应分配推理时计算资源的方法，有可能让较小的基础模型表现超越大得多的模型。

- **模型合并技术**：[@cwolferesearch](https://twitter.com/cwolferesearch/status/1821250560508465387) 解释了各种模型合并技术，包括线性合并（linear merging）、任务向量（task vectors）、TIES 合并和 DARE 合并。这些方法允许在不需要额外训练数据或计算资源的情况下，结合多个 LLM 的能力。

**AI 应用与工具**

- **用于对象分割的 SAM 2**：[@AIatMeta](https://twitter.com/AIatMeta/status/1821229074754498667) 宣布了 SAM 2，这是一个用于图像和视频中实时、可提示对象分割的统一模型。[@swyx](https://twitter.com/swyx/status/1821298796841541956) 强调，据估计仅在图像处理方面，SAM 1 在一年内就为用户节省了约 35 年的时间。

- **AI 数字人**：[@synthesiaIO](https://twitter.com/synthesiaIO/status/1821152878418944260) 推出了个人 AI 数字人（avatars），并在一个有 4000 多人参加的直播活动中展示了其逼真度。

- **LlamaIndex 进展**：[@llama_index](https://twitter.com/llama_index/status/1821227063812223338) 分享了一个构建文档聊天机器人的教程，该教程使用 Firecrawl 进行网页抓取，并使用 Qdrant 进行向量存储和检索。

**AI 伦理与政策**

- **结构化输出与安全性**：[@AlphaSignalAI](https://twitter.com/AlphaSignalAI/status/1821226608314777708) 报道了 OpenAI 发布了其性能最强的 GPT-4o 助手模型，其特点是具有 100% 可靠性的结构化输出（structured outputs），并改进了 Token 限制和价格。

- **AI 安全担忧**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1821151485293437237) 总结了一篇关于使用人类流畅提示词对经过安全微调的 LLM 进行越狱（jailbreaking）的论文，该方法在保持低困惑度（perplexity）的同时实现了极高的攻击成功率。

---

# AI Reddit 综述

## /r/LocalLlama 综述

**主题 1. 免费访问高级 LLM：Llama 3.1 405B 和 Sonnet 3.5**

- **免费使用 Llama3.1 405b + Sonnet 3.5** ([分数: 304, 评论: 108](https://reddit.com//r/LocalLLaMA/comments/1emddb4/llama31_405b_sonnet_35_for_free/))：Google Cloud 正通过其 Vertex AI Model Garden 提供 **Llama 3.1 405B** 和 **Sonnet 3.5** 模型的**免费访问**，提供价值高达 **300 美元** 的 API 使用额度，对于每个 Google 账号，这大约相当于 **2000 万个输出 Token** 的 Sonnet 3.5 使用量。一个相关的项目 **Open Answer Engine** 展示了如何利用这项 API 服务创建一个具有 Google 搜索功能的 **405B 模型**，详情见 Weights & Biases 的报告。

- **[Experimenting llama3-s: An early-fusion, audio & text, multimodal model](https://homebrew.ltd/blog/can-llama-3-listen)** ([Score: 92, Comments: 16](https://reddit.com//r/LocalLLaMA/comments/1emjyq0/experimenting_llama3s_an_earlyfusion_audio_text/)): **Llama3-s** 是一款集成了**音频和文本**的早期融合（early-fusion）多模态模型，现已发布供实验使用。该模型在 **1.4 trillion tokens** 的文本和 **700 billion tokens** 的音频上进行了训练，展示了在**转录**、**翻译**和**音频理解**任务中的能力，同时在纯文本基准测试中保持了强劲性能。

**Theme 2. Optimized Inference and Quantization for ARM-based Processors**

- **Snapdragon X CPU inference is fast! (Q_4_0_4_8 quantization)** ([Score: 83, Comments: 39](https://reddit.com//r/LocalLLaMA/comments/1emd3bg/snapdragon_x_cpu_inference_is_fast_q_4_0_4_8/)): **Snapdragon X CPU** 在使用 **Q_4_0_4_8 量化**运行 **Llama 3.1 8B** 时表现出令人印象深刻的推理速度，在搭载 **10 核 Snapdragon X Plus 芯片**的 **Surface Pro 11** 上达到了 **15.39 tokens per second**。该帖子提供了优化性能的说明，包括使用 **-win-llvm-arm64.zip** 发布版本、将 Windows 电源模式设置为“**最佳性能**”，以及使用 **llama-quantize.exe** 命令将现有的 GGUF 模型重新量化为 **Q4_0_4_8**，并指出这些结果与 **MacBook M2 和 M3** 的性能水平相当。

- **[LG AI releases Exaone-3.0, a 7.8b SOTA model](https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct)** ([Score: 144, Comments: 77](https://reddit.com//r/LocalLLaMA/comments/1emfm03/lg_ai_releases_exaone30_a_78b_sota_model/)): LG AI 发布了 **Exaone-3.0**，这是一个拥有 **7.8 billion 参数**的语言模型，在多个基准测试中达到了 SOTA 性能。该模型在**韩语和英语**方面表现出卓越的能力，在某些任务上超越了像 **GPT-3.5** 这样更大规模的模型，而体积却小得多。

**Theme 3. Summarization Techniques and Model Comparison for Large Texts**

- **Best summarizing LLMs for average PCs?** ([Score: 68, Comments: 72](https://reddit.com//r/LocalLLaMA/comments/1em9545/best_summarizing_llms_for_average_pcs/)): 该帖子讨论了与**消费级硬件**兼容的**摘要生成 LLM**，特别是针对 **Nvidia RTX 3060 12GB** GPU 和 **32GB DDR5 RAM** 的配置。作者推荐使用 **Qwen2**、**InternLM**，有时也使用 **Phi3 mini 和 medium 128k** 来处理 **20,000 到 25,000 字的文本块**，并指出更大的 LLM 与其配置不兼容，且 **Llama 3.1** 在此任务中表现不佳。
  - **Llama3.1** 和 **GLM-4-9b** 被用于摘要 YouTube 视频转录文本。该过程包括创建章节大纲，然后为每个项目生成详细描述，这种滚动窗口（rolling window）方法对于长内容效果良好。
  - **Gemini 1.5 Flash** 的免费层级提供了令人印象深刻的摘要能力，拥有 **1 million token context window** 和每分钟 **1 million 免费 tokens** 的额度，正如一位用户链接到 [Google AI 价格页面](https://ai.google.dev/pricing)所澄清的那样。
  - **Obsidian 的 Copilot 插件**允许使用本地 LLM 轻松对选定文本进行摘要，提供了一个直接在应用程序内保存摘要的流线化流程。


**Theme 4. Repurposing Mining Hardware for AI Workloads**

- **[Picked up a mining rig for testing . . .](https://i.redd.it/ikzm89f14dhd1.jpeg)** ([Score: 143, Comments: 62](https://reddit.com//r/LocalLLaMA/comments/1emw6eq/picked_up_a_mining_rig_for_testing/)): 一位用户购买了一台装有 **7x 3060 GPU** 的**矿机**进行测试，发现它是一台完整的 PC，虽然处理器和 RAM 较弱，而不仅仅是电源和转接卡。他们正在寻求关于如何在这台机器上**加载 AI 模型**并将**输出分发到宿主 LLM 应用程序**的建议，旨在将矿机硬件重新用于 AI 推理任务。
  - **llama.cpp** 可以在该设备的 **84GB VRAM** 上运行 **LLaMA 3.1 70B Q8**，如果使用 Q6 量化则可获得更多上下文空间。用户建议先尝试较小的模型，从 **2B** 开始逐步增加以测试性能。
  - 建议升级**主板**和 **CPU**，推荐使用**双路 E5 v3/v4 服务器 CPU** 以及支持多个 PCIe 插槽的主板。**PCIe 分叉（Bifurcation）扩展卡**可以允许一个 16x 插槽处理多个 GPU。
  - 推荐使用 **vLLM** 进行分布式设置，而 **ExLlamaV2** 提供了内置的生成器/队列功能。该矿机**每个 GPU 仅占用一个 PCIe 通道**可能是个瓶颈，但一旦模型加载到 VRAM 中，CPU 和系统 RAM 的占用率极低。

## AI Reddit 全回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**AI 模型改进与技术**

- **Flux 结合 LoRA 显著提升照片写实感**：在 r/StableDiffusion 中，一篇文章展示了使用 Flux 结合 LoRA 如何显著[增强生成图像的写实感](https://www.reddit.com/r/StableDiffusion/comments/1emrprx/feel_the_difference_between_using_flux_with/)，特别是皮肤纹理和面部细节。用户注意到第一张图像看起来与真实照片无异。

- **Midjourney 到 Runway 的视频生成效果令人印象深刻**：[r/singularity 的一篇文章](https://www.reddit.com/r/singularity/comments/1emnyxq/midjourney_to_runway_is_scary_good/)展示了将 Midjourney 图像作为输入用于 Runway 视频生成的惊人能力，突显了 AI 生成视频的快速进步。

**OpenAI 动态与推测**

- **Project Strawberry 预热**：OpenAI 在社交媒体上暗示 [“Project Strawberry”](https://www.reddit.com/r/OpenAI/comments/1emwh93/whats_going_on/) 的帖子引发了讨论和推测。一些用户认为这可能与改进 ChatGPT 统计单词（如 “strawberry”）中字母数量的能力有关，这是一个已知的问题。

- **潜在的新推理技术**：分享的一篇 [Reuters 文章](https://www.reuters.com/technology/artificial-intelligence/openai-working-new-reasoning-technology-under-code-name-strawberry-2024-07-12/)表明，OpenAI 正在开发代号为 “Strawberry” 的新推理技术。

**AI 模型行为与局限性**

- **ChatGPT 在字母计数方面表现挣扎**：多位用户测试了 ChatGPT 统计 “strawberry” 中 ‘r’ 数量的能力，模型始终给出错误答案。这突显了在某些类型的推理任务中持续存在的局限性。

- **Tokenization 对模型性能的影响**：一些专业用户指出，字母计数问题与语言模型如何对单词进行 Tokenization 有关，解释了为什么 ChatGPT 在处理这个看似简单的任务时会遇到困难。

**社区反应与讨论**

- **对 OpenAI 营销手段的质疑**：几位用户对 OpenAI 的营销策略表示不满，认为 “Strawberry” 的预热过于夸大，或者分散了对其他问题的注意力。

- **关于 AI 进展的辩论**：这些帖子引发了关于 AI 能力现状的讨论，一些用户对图像和视频生成的快速进步感到印象深刻，而另一些用户则指出了推理任务中持久存在的局限性。

---

# AI Discord 全回顾

> 由 GPT4O-Aug (gpt-4o-2024-08-06) 生成的总结之总结的摘要

**1. 模型性能与优化**

- **BiRefNet 超越 RMBG1.4**：**BiRefNet** 在背景移除方面表现出优于 **RMBG1.4** 的性能，具有增强的高分辨率图像分割能力，详见 [arXiv 论文](https://arxiv.org/pdf/2401.03407)。
  - 该模型由 [南开大学](https://huggingface.co/ZhengPeng7/BiRefNet) 开发，采用双边参考技术，显著优化了图像处理任务。
- **Torchao v0.4.0 提升优化性能**：**torchao v0.4.0** 的发布引入了 **KV cache 量化**和**量化感知训练 (QAT)**，增强了低比特优化器支持。
  - 社区讨论了关于 Intx Tensor Subclasses 的 GitHub issue，邀请在追踪器上提供更多输入，以实验低比特量化。
- **RoPE 优化简化代码**：成员们分析了 **RoPE** 的实现，主张通过转向直接的三角函数运算而非复数来简化代码。
  - 这一调整被视为在保留训练逻辑功能完整性的同时，迈向增强代码清晰度的一步。


**2. 开源 AI 发展**

- **Harambe 彻底改变漏洞猎取**：开源漏洞猎取工具 **Harambe** 的推出，旨在利用 LLM 生成 API 端点建议，从而简化 API 分析。
  - 这种从传统 fuzzing 技术向新方法的转变，为识别代码中的潜在问题提供了一种更高效的方式。
- **EurekAI 平台为研究人员发布**：[EurekAI](https://eurekai-web-app.vercel.app/signup) 作为一个面向研究人员的跨协作平台推出，旨在通过 AI 功能简化研究流程并提高生产力。
  - 目前处于 alpha 阶段，它承诺提供诸如项目创建和集成日志记录（journaling）等功能，旨在促进研究参与度。
- **Midjourney CEO 批评开源**：Midjourney CEO 对 **开源持怀疑态度**，认为本地模型无法与使用 **64 GPUs** 的服务竞争，并将 **ControlNet** 视为唯一的成功案例。
  - 批评者反驳称，Midjourney 的产品类似于开源所能实现的 **劣质版本**，并指出了 **Flux** 中的 **overfitting** 问题：*“它看起来有一种塑料感。”*


**3. AI 基础设施与市场动态**

- **Hugging Face 通过收购 XetHub 进行扩张**：Hugging Face 宣布收购 [XetHub](https://www.forbes.com/sites/richardnieva/2024/08/08/hugging-face-xethub-acquisition/)，以增强其大模型的协作基础设施，旨在实现更好的数据集管理。
  - CEO Clem Delangue 强调，此举对于扩展 AI 模型开发和统一其运营策略至关重要。
- **OpenAI 的降价引发竞争**：据报道，OpenAI 正在对其 GPT-4o 模型实施 **70% 的降价**，引起了行业的广泛关注。
  - 这种剧烈的价格变动可能会导致 AI 模型领域竞争对手调整其定价策略。
- **Vercel 故障影响 OpenRouter**：Vercel 目前面临间歇性故障，影响了 OpenRouter 服务，详见其 [状态更新](https://x.com/OpenRouterAI/status/1821267624228966781)。
  - 经过多次更新，服务在 **东部时间下午 3:45** 恢复稳定，目前仍在持续监控中。


**4. Prompt Engineering 与微调**

- **Self-Discover Prompting 受到关注**：一位成员强调了 **Self-Discover** prompting 的潜力，断言其力量和有效性超越了传统的 Chain-of-Thought (CoT) 方法。
  - 他们强调了其在构建定制化 prompt 以产生更好输出方面的适用性。
- **RAG Pipeline 需要增强可观测性**：有观点认为 **RAG pipelines** 需要更好的可观测性来捕获查询时的追踪（query-time traces），以及合理的文档分块（document chunking）的重要性。
  - 正如一条 [推文](https://twitter.com/llama_index/status/1821332562310205918) 所强调的，不恰当的上下文分块可能会导致检索问题。
- **为 LLM 优化聊天记录**：讨论集中在实现一个 **自定义函数** 来限制 LLM 应用的聊天记录，旨在提高性能。
  - 维护特定用户的上下文被认为是简化跨不同用户交互的聊天保留的关键因素。


**5. AI 应用与工具**

- **SAM 2 Pod 启动上线**：最新一期的 [Latent Space podcast](https://x.com/latentspacepod/status/1821296511260504408) 以 **SAM 2** 为主题，邀请了 **Nikhila Ravi** 和 **Joseph Nelson** 分享见解。
  - 听众了解到，在 RoboFlow 上使用 **SAM** 标记了 **4900 万张图像**，估计节省了用户 **35 年** 的时间。
- **Stable Diffusion 在 Python 中进行优化**：成员们讨论了利用 **Diffusers** 库在 Python 中实现 **Stable Diffusion**，重点是优化性能和 VRAM 使用。
  - 他们强调了正确设置参数以达到 **理想输出质量** 的重要性。
- **MiniCPM-V 2.6 在性能测试中表现出色**：据报道，**MiniCPM-V 2.6** 的表现优于其竞争对手，包括 **Gemini 1.5 Pro**、**GPT-4V** 和 **Claude 3.5 Sonnet**，特别是在多图应用中。
  - 欲了解更多详情，成员们分享了其 [Hugging Face 页面](https://huggingface.co/openbmb/MiniCPM-V-2_6) 和 [GitHub 仓库](https://github.com/OpenBMB/MiniCPM-V) 的链接。

---

# PART 1: 高层级 Discord 摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **4bit GGUF 模型面临加载挑战**：围绕 **4bit GGUF 模型**展开了讨论，指出在模型加载过程中使用 `load_in_4bit` 可能导致**精度损失**，并强调了如果不开启此选项会出现 OOM 错误。
   - 虽然 **4bit** 降低了 VRAM 消耗，但在实施前需要仔细权衡性能方面的折中。
- **PPO Trainer 实现中出现问题**：一名成员报告在尝试将自定义二进制奖励函数与 **PPO Trainer** 结合使用时，出现了**负 KL 散度错误**。
   - 成员们探讨了将 **DPO** 作为更简单的替代方案，但也对其与 **PPO** 相比的性能表现表示担忧。
- **Unsloth 推出多 GPU 支持**：确认向受信任的 Unsloth 用户推出**多 GPU 支持**（multi-GPU support），这可能导致 VRAM 消耗降低并提高处理速度。
   - 随后引发了关于该功能是会在开源仓库中开放，还是仅限付费订阅用户使用的辩论。
- **Mistral 模型成功量化**：分享了关于量化 **123B Mistral-Large-Instruct-2407** 模型的见解，使用 **EfficientQAT** 算法在减小模型体积的同时实现了极小的精度下降。
   - 这种优化进一步证明了在不产生实质性输出降级的情况下提高模型效率的可行性。
- **Harambe：新型 Bug 猎取助手**：介绍了 **Harambe**，这是一个开源的 Bug 猎取工具，旨在利用 LLM 生成 API 端点建议，从而简化 API 分析。
   - 这种从传统模糊测试（fuzzing）技术的转变，为识别代码中的潜在问题提供了一种更高效的方法。



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **BiRefNet 超越 RMBG1.4**：**BiRefNet** 在背景去除方面的表现优于 **RMBG1.4**，具有增强的高分辨率图像分割能力，详见 [arXiv 论文](https://arxiv.org/pdf/2401.03407)。
   - 该模型由 [南开大学](https://huggingface.co/ZhengPeng7/BiRefNet) 开发，采用双边参考技术，显著优化了图像处理任务。
- **EurekAI 平台发布**：[EurekAI](https://eurekai-web-app.vercel.app/signup) 作为一个面向研究人员的跨协作平台推出，旨在通过 AI 功能简化研究流程并提高生产力。
   - 目前处于 alpha 阶段，它承诺提供项目创建和集成日志等功能，旨在促进研究参与度。
- **AI 模型性能评估**：成员们对比了预训练翻译模型，如 [Facebook 的 M2M100](https://huggingface.co/facebook/m2m100_418M) 和 [SeamlessM4T](https://huggingface.co/docs/transformers/model_doc/seamless_m4t_v2)，这些模型在多语言翻译中展现出良好的前景。
   - 讨论强调了 SeamlessM4T-v2 与 Whisper 模型在转录能力上的差异，重点关注实际可用性。
- **Gradio v4.41 的精彩更新**：**Gradio v4.41** 的发布引入了显著功能，例如为 `gr.Image` 提供**全屏图像**显示，通过改进的用户交互机制增强了输出查看体验。
   - 此次更新还加强了针对未授权访问和 XSS 攻击的安全性，为部署应用程序提供了更健壮的框架。
- **Papers with Code 资源见解**：一名成员强调 [Papers with Code](https://paperswithcode.com/sota) 是总结计算机视觉领域 State-of-the-art (SOTA) 性能的重要资源，包含 **11,272 个基准测试**和 **137,097 篇带有代码的论文**。
   - 这个宝贵的平台有助于用户探索各种机器学习应用，增强了文献的可理解性。



---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **用于 CUDA Profiling 的 BPF 见解**：一位成员询问是否有人正在使用 **BPF** 对 **CUDA** 进行 profiling，部分成员表示 **eBPF** 缺乏对 GPU 活动的可见性，仅限于 OS kernel。
   - 成员们对其有效性提出了质疑，并建议使用 **Nsight Compute** 和 **Nsight Systems** 等替代方案来进行全面的 GPU 应用监控。
- **Attention Gym 链接与 FlexAttention**：成员们报告了 **Attention Gym** 的链接失效，并对其关于 softcapping 的详细内容表示赞赏。
   - 此外，还出现了关于将 **FlexAttention** 集成到 HF 模型中的讨论，表示计划等待 PyTorch **2.5** 版本以实现更顺畅的集成。
- **torchao v0.4.0 发布**：**torchao v0.4.0** 的发布带来了 **KV cache 量化**和**量化感知训练 (QAT)** 等增强功能，其对低比特优化器的支持令人振奋。
   - 社区参与包括一个关于用于低比特量化实验的 Intx Tensor Subclasses 的 GitHub issue，并邀请在 tracker 上提供进一步的输入。
- **内存使用与 KV Cache 优化**：一位成员对 **KV Cache** 的实现优化了内存使用，实现了在单个 **80GB GPU** 上进行全量 bfloat16 微调，尽管已接近内存极限。
   - 讨论建议探索托管内存以缓解限制，同时准备专注于代码清理和可维护性的 pull requests。
- **RoPE 优化讨论**：成员们分析了 **RoPE** 的实现，主张通过转向直接的三角运算而非复数来简化。
   - 这一调整被视为在保留训练逻辑功能完整性的同时，增强代码清晰度的一种举措。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro 降低每日限制**：**Perplexity Pro** 用户报告每日限制从 **600 次降至 450 次**，引发了对变更沟通方式的不满。
   - 一位成员表示不信任，称他们事先没有收到关于此变动的任何通知。
- **API 停机导致访问问题**：用户正面临 **Perplexity API** 的**重大停机**，引发了对问题范围的担忧。
   - 报告显示，一些用户通过 VPN 连接到不同地区解决了问题，这表明可能存在基于地理位置的差异。
- **Google 反垄断裁决震动市场**：**2024 年 8 月 5 日**，美国法院裁定 Google 维持非法垄断，这是 **Department of Justice** 的重大胜利。
   - 裁决确认“Google 是垄断者”，并概述了其维持市场主导地位的非法做法。
- **关于神经科学中量子理论的讨论**：对大脑中**量子纠缠**的研究引发了辩论，特别是围绕 **Orch-OR** 等暗示认知影响的理论。
   - 怀疑论者认为，大脑**温暖、潮湿**的环境可能无法支持持续的量子态。
- **非英语回答缺乏连贯性**：用户注意到非英语语言的 prompt 经常产生**不连贯**的回答，突显了多语言处理方面的局限性。
   - 一个法语案例导致了重复的输出，引发了对模型在不同语言下鲁棒性的担忧。

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **在 Python 项目中优化 Stable Diffusion**：成员们讨论了利用 **Diffusers** 库在 Python 中实现 **Stable Diffusion**，重点在于优化性能和 VRAM 占用。
   - 他们强调了正确设置参数以达到**期望输出质量**的重要性。
- **为 AI 工作升级旧电脑**：一位用户寻求关于升级陈旧电脑配置以处理 AI 任务的建议，寻找不需要彻底翻新且价格合理的组件。
   - 建议包括使用 **Fiverr** 寻求组装协助，以及考虑将准系统预装电脑作为替代方案。
- **在 Intel CPU 上进行换脸**：一位用户请求推荐兼容 **Intel CPU** 的换脸技术，并表示愿意为专家指导付费。
   - 这凸显了硬件配置较低的用户对实用解决方案的需求。
- **使用 SAM 工作流增强图像**：社区分享了利用 **SAM** 检测器改进**图像细节**的心得，从而实现更强大的工作流。
   - 一位成员强调细节处理不应局限于人物，还应包括背景和结构，从而拓宽了潜在的应用场景。
- **在 Mac 上生成 NSFW 内容 - 需要网页工具**：一位用户询问在配备 **16GB RAM 的 MacBook Air M2** 上能高效运行的、用于 **NSFW 内容生成** 的最佳网页工具。
   - 讨论内容包括与模型复杂度相关的性能影响，以及基于硬件能力进行本地安装的优势。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **NVIDIA 显卡未受当前问题影响**：当前的性能问题仅影响 **CPU**，成员们对他们的 **NVIDIA 显卡** 表示放心。
   - 讨论强调了对 CPU 与 GPU 配置的偏好，展示了 CPU 驱动型工作负载的优势。
- **CPU 使用率报告引发困惑**：关于 CPU 使用率数字超过 100% 的讨论出现，解释为应用程序根据核心数量报告总使用率。
   - 成员们指出不同操作系统之间的报告标准各异，导致了普遍的误解。
- **双 GPU 未能提升推理速度**：成员们确认 **LM Studio** 支持双 GPU，但推理速度仍与单 GPU 配置相似。
   - 出现了关于硬件改进以增强 Token 吞吐量从而获得更好性能的建议。
- **性能辩论：4090 对比 3080**：用户对 **4090** 的表现与 **3080** 相似表示不满，其训练速度优势仅为 **每 epoch 20 毫秒**。
   - 虽然 **4090** 在游戏方面表现出色，但其他人强调了 **3080** 在处理 8B 以下模型时的高效性。
- **有限的 VRAM 阻碍了模型选择**：**2GB VRAM** 被证明不足以运行大多数模型，导致低 VRAM 选项的性能不佳。
   - 用户注意到必须将较大的模型拆分到 **VRAM** 和**系统内存**中，这显著限制了效率。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI 发布 GPT-4o System Card**：OpenAI 分享了 [GPT-4o System Card](https://openai.com/index/gpt-4o-system-card/)，详细介绍了旨在追踪**前沿模型风险**的评估，并概述了带有预设语音的音频功能。
   - 该 System Card 确保了针对有害内容的妥善防护，增强了用户的信任和理解。
- **免费用户可访问 DALL·E 3**：ChatGPT 免费用户现在每天可以使用 **DALL·E 3** 创建最多**两张图片**，使内容生成更加普及。
   - 这一功能通过无缝请求，为演示文稿和定制卡片等项目提供了个性化的创意输出。
- **持续的网站访问问题**：多位用户报告了访问 OpenAI 主站时的连接问题，导致持续的错误和间歇性的可访问性。
   - 这种情况证实了成员们日益增长的挫败感以及社区内意想不到的困难。
- **对消息配额的困惑**：成员们对使用平台（尤其是涉及 **GPT-4o** 时）较早达到消息配额限制表示不满。
   - 这种体验引发了关于意外达到限制的不一致性的讨论，影响了用户交互。
- **在使用 OpenAI Python SDK 时遇到困难**：用户在尝试使用 OpenAI Python SDK 复现结果时面临挑战，特别是在遇到 Python 版本差异时。
   - 这表明了可能存在的兼容性问题，阻碍了跨不同编码环境的准确输出。

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **MindSearch AI 增强信息检索**：论文 *MindSearch: Mimicking Human Minds Elicits Deep AI Search* 介绍了一种 Agentic AI，通过 *WebPlanner* 和 *WebSearcher* 的双系统方法改进了信息检索，超越了当前的搜索模型。
   - 这种创新结构有效处理复杂查询，展示了在智能信息寻求方面的显著增强。
- **Tavus Phoenix 模型席卷视频生成领域**：Tavus 推出了 **Phoenix** 模型，该模型利用先进技术创建超写实的数字人视频 (talking head videos)，并能够同步**自然面部动作**。
   - 开发者可以通过 Tavus 的 **Replica API** 访问 *Phoenix* 模型，实现视频内容的多样化和高级定制。
- **模型在处理倒置文本时崩溃**：Mistral 和 ChatGPT 等多种模型无法生成连贯的倒置文本，而 Claude Opus 和 Sonnet 3.5 则能轻松处理并提供准确输出。
   - 这些观察结果突显了 Claude 模型卓越的能力，特别是在无错误地生成和重写倒置文本方面。
- **社区讨论 AI Discord 资源**：一名成员分享了一个 [Reddit 帖子](https://www.reddit.com/r/nousresearch/comments/1elmrjr/most_helpful_ai_discordscommunities/?share_id=L2tAJZE66RY4dOPfIbiMw)，列出了几个有用的 AI Discord 频道，包括 **Replete-AI** 和 **Unsloth**。
   - 这些资源为在 Discord 中探索 AI 领域的学习者提供了多样的见解和支持。
- **Claude API 面临服务器过载问题**：用户指出 Claude API 在使用高峰期频繁给出过载消息，干扰了他们的工作流。
   - 目前尚不确定这些问题是源于服务器限制还是影响访问的封禁。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **LM Harness 数据集要求明确**：一名成员询问了用于 **LM Evaluation Harness** 的数据集所需格式，特别是必要的字典键 (dictionary keys)。他们被引导至 [YAML 文件](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/gsm8k) 以获取关于键设计的结构化指导。
   - 这强调了格式化的灵活性，这对于从事数据集集成的开发者至关重要。
- **辩论 AI 模型的 CBRN 风险**：成员们讨论了模型是否可以在不产生 CBRN 风险的情况下提供化学建议，并担心过滤可能会损害科学能力。
   - 讨论指出，知识渊博的用户可能仍会提取有害信息，这对当前过滤策略的有效性提出了挑战。
- **过滤预训练数据的后果**：参与者认为，删除“坏”数据可能会削弱模型的整体理解能力和对齐效果。
   - 有人提到，缺乏负面示例可能会阻碍模型避免有害活动的能力，引发了对能力退化的担忧。
- **对 AI 新闻报道的挫败感**：成员们表达了对记者描述 AI 方式的不满，认为他们通常在缺乏足够背景的情况下强调煽动性的风险。
   - 这引发了对 AI 输出安全叙事及其潜在误导的更广泛担忧。
- **寻找开源奖励模型**：有人询问关于用于验证数学任务的有效**开源 Process Based Reward Models**。
   - 这突显了在数学解题领域对可靠验证工具的迫切需求。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Hugging Face 通过收购 XetHub 进行扩张**：Hugging Face 宣布收购 [XetHub](https://www.forbes.com/sites/richardnieva/2024/08/08/hugging-face-xethub-acquisition/)，以增强其针对大型模型的协作基础设施，旨在实现更好的 dataset 管理。
   - CEO Clem Delangue 强调，此举对于扩展 AI 模型开发和统一其运营策略至关重要。
- **Qwen2-Math 在数学任务中占据主导地位**：阿里巴巴新推出的 [Qwen2-Math 模型系列](https://qwenlm.github.io/blog/qwen2-math)在专业数学任务中的表现优于 GPT-4o 和 Claude 3.5。
   - 这标志着数学专用语言模型的重大飞跃，预示着特定领域应用可能发生的转变。
- **AI Infrastructure 独角兽崛起**：一系列讨论揭示了像 Hugging Face 和 Databricks 这样的 AI infrastructure 构建者如何塑造 generative AI 市场。
   - Hugging Face 最近的融资努力使其在 open-source 领域足以与 GitHub 竞争，反映出强劲的增长战略。
- **OpenAI 的降价引发竞争**：据报道，OpenAI 正在对其 GPT-4o 模型实施 **70% 的降价**，引起了行业的极大关注。
   - 这种剧烈的价格变动可能会导致 AI 模型领域竞争对手调整定价策略。
- **关于 GPT-4 Token 数量的澄清**：报告确认 **GPT-4** 使用了 **10 万亿个 tokens**，这一数字得到了聊天频道中多个来源的证实。
   - 尽管达成了共识，成员们仍将 GPT-4 称为**过时的技术 (ancient technology)**，暗示了模型能力的快速演进。



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **修复 AWS Lambda 中的 LangChain 问题**：一位用户在尝试于 Python **3.12** 运行时的 AWS Lambda 中导入 **LangChain** 模块时遇到了 **pydantic** 错误，凸显了潜在的版本冲突。
   - 建议包括仔细检查 **lambda layer** 设置以解决导入问题。
- **为 LLM 优化聊天记录 (Chat History)**：讨论集中在实现一个**自定义函数**来限制 LLM 应用的聊天记录，旨在提高性能。
   - 维护特定用户的 context 被认为是简化不同用户交互中聊天保留的关键因素。
- **LangChain 与其他框架的辩论**：用户表示，由于功能差异，使用 LangChain 从 **OpenAI** 切换到 **Anthropic** 需要大量的代码重写。
   - 参与者一致认为，尽管 LangChain 进行了抽象，但仍需根据单个 LLM 的行为进行特定调整。
- **LLM 可靠性担忧**：有人对 **Claude 3.5** 出现内部服务器错误表示担忧，强调了生产环境中 AI 系统的可靠性。
   - 这引发了关于 LangChain 是否是构建稳定 AI 系统的正确选择的更广泛讨论。



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **GPT-4o 增强了输入和输出能力**：**GPT-4o** 模型可以处理文本、音频、图像和视频，显著提升了通用性和响应速度，类似于人类交互。
   - 它的 API 使用价格也**便宜了 50%**，并在多种语言中表现出更强的性能。
- **Gemini 1.5 Flash 大幅降价**：GoogleAI 将 **Gemini 1.5 Flash** 的价格降低了约 **70%**，使其对开发者更具吸引力。
   - **AI Studio** 现在面向所有 workspace 客户开放，方便更好地进行新语言实验。
- **DALL·E 3 向免费用户开放**：ChatGPT Free 用户现在每天可以使用 **DALL·E 3** 生成**两张图片**，提高了内容创作的可及性。
   - 虽然这一功能受到欢迎，但对其更广泛的应用仍存在一些怀疑。
- **Mistral Agents 扩大了功能集成**：**Mistral Agents** 现在可以在各种工作流中使用 Python，突显了其更强的适应性。
   - 用户对促进 API 调用的功能非常感兴趣，这增强了实际应用。
- **SAM 2 播客上线**：最新一期的 [Latent Space podcast](https://x.com/latentspacepod/status/1821296511260504408) 介绍了 **SAM 2**，并包含了来自 **Nikhila Ravi** 和 **Joseph Nelson** 的见解。
   - 听众了解到，在 RoboFlow 上使用 **SAM** 标记了 **4900 万张图像**，估计为用户节省了 **35 年**的时间。



---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Midjourney CEO 批评开源**：Midjourney CEO 对**开源持怀疑态度**，认为本地模型无法与他们使用 **64 GPUs** 的服务竞争，并将 **ControlNet** 贬低为唯一的成功案例。
   - 批评者反驳称，Midjourney 的产品类似于开源所能实现效果的**劣质版本**，并强调了 **Flux** 中的**过拟合 (overfitting)** 问题：“它看起来有一种塑料感。”
- **ASL 语言模型概念出现**：一位用户提议开发一个将**语音翻译为 ASL** 的应用，并考虑了使用手势图像训练模型的挑战。
   - 建议包括对现有模型进行 **fine-tuning**，另一位用户讨论了改进语音识别模型，使用 **emojis** 来表示手势。
- **合成语音数据集构想提出**：一名成员提议使用 **so-vits-svc** 通过转换音频文件中的声音来创建合成数据集，旨在增强多样性同时保留内容。
   - 这种方法可以促进在语音表达中捕捉更广泛的**情感**，并提高模型在**人口统计分类**中的区分度。
- **Flux 模型讨论持续**：用户对 **Flux** 进行了反思，一些人将其标记为“好玩的玩具”，认为它没有取得重大进展，并对其**过拟合 (overfitting)** 表示担忧。
   - 持续的对话强调了与 Midjourney 相比，Flux 需要更有针对性的 **fine-tuning**。
- **多种辅助功能 AI 应用**：分享了各种旨在增强辅助功能的 AI 建议，包括一个用于语音识别的**尊重隐私**的 IP Relay 应用。
   - 成员们专注于本地 **inference** 技术以帮助听障人士，展示了对具有影响力的 AI 应用的浓厚兴趣。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **多后端重构安装顺利**：一位成员确认他们成功安装了 **multi-backend-refactor**，没有出现任何问题，并准备关注后续进展。
   - 这种顺利的安装过程增强了对其在进行中项目的稳定性和实用性的信心。
- **Google Gemini 大幅降价**：一位成员分享了一个名为“Google Gemini 疯狂降价！！！”的 [YouTube 视频](https://www.youtube.com/watch?v=3ICC4ftZP8Y)，重点介绍了 **Gemini 1.5 Flash** 的降价。
   - 视频概述了大幅度的折扣，观众可以在 [Google Developers blog](https://developers.googleblog.com/en/gemini-15-flash-updates-google-ai-studio-gemini-...) 中找到更多细节。
- **在 Metaverse 中呼吁 H100**：有人幽默地建议 **Zuck** 需要在 **Metaverse** 中提供更多 **H100** GPU，强调了对先进资源的需求。
   - 这一言论强调了虚拟环境中对高性能计算的持续需求。
- **使用 38k 数据集进行训练**：一位成员报告称，他们使用 **38k** 条目的数据集训练模型，在 **RTX 4090** 上耗时 **32 小时**。
   - 他们担心当前设置中的 **learning rate** 可能过高。
- **正确 Prompt 格式讨论**：成员们强调了在 **inference** 期间针对特定任务的 Prompt 必须使用 **Alpaca 格式**，以确保一致性。
   - 他们强调，聊天时的输出必须镜像 **fine-tuning** 中使用的格式，以获得最佳效果。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Almosnow 寻求 API 文件上传指导**：一位成员希望使用 API 在 coral.cohere.com 上复制 UI 中的 PDF 查询功能，但难以找到相关文档。
   - *出现错误：could not convert string to float: 'float_'* 表明输入格式存在底层问题。
- **Mapler 提供 RAG 资源**：Mapler 回复了关于通过 Cohere API 使用 **Retrieval-Augmented Generation (RAG)** 的资源，并链接到了一篇 [博客文章](https://cohere.com/llmu/rag-start) 和额外文档。
   - 他们分享了一个用于生成可靠回答的代码片段，增强了对 RAG 使用的理解。
- **Azure AI Search 集成困扰**：用户报告称，尽管向量化数据已成功索引，但 **Cohere embeddings** 在 Azure AI Search 中的结果并不一致。
   - [使用来自 Azure AI Studio 的模型进行集成向量化](https://learn.microsoft.com/en-sg/azure/search/vector-search-integrated-vectorization-ai-studio?tabs=cohere) 被强调为解决问题的潜在资源。
- **Cohere-toolkit 增强工具激活**：讨论了通过在 preamble 中添加 `always use the <tool> tool` 来默认启用 **Cohere-toolkit** 中的工具。
   - 有人指出，必须列出该工具才能在调用期间正常运行。
- **用户在自定义部署中遇到障碍**：一位成员分享了在模型选择受限的自定义部署中修改 `invoke_chat_stream` 以实现默认工具加载的尝试。
   - 由于 UI 差异显示工具未激活而产生困惑，强调了模型反馈中需要进一步澄清。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 公告即将发布**：LlamaIndex 的公告定于 **5 分钟** 后发布，这在公告频道成员中引发了热议。
   - 成员们正热切期待此次活动可能带来的亮点或更新。
- **RAG 流水线需要增强可观测性**：有观点认为 **RAG pipelines** 需要更好的可观测性来捕获查询时的 traces 以及正确文档 chunking 的重要性。
   - 正如一条 [推文](https://twitter.com/llama_index/status/1821332562310205918) 所强调的，不当的上下文 chunking 可能会导致检索问题。
- **LongRAG 论文对比引发讨论**：分享的 **LongRAG 论文** 表明，在资源充足的情况下，长上下文模型优于 **RAG**，这引发了对其方法论的讨论。
   - 成员们表达了对涉及 **Claude 3.5** 的对比以及来自 LangChain 的 Lance 的见解的渴望，增强了社区讨论。
- **Self-Routing 技术革新效率**：LongRAG 论文中介绍的 **Self-Route 方法** 根据 self-reflection 路由查询，在保持性能的同时降低了成本。
   - 利用元数据进行 **parent-document retrieval** 的提议浮出水面，以增强检索系统，同时也强调了元数据标记中的可靠性挑战。
- **Workflows 抽象引发关注**：团队展示了使用 **Workflows** 构建复杂 AI 应用程序的简便性，特别是在 [新视频](https://twitter.com/llama_index/status/1821575082516660440) 中展示的重建 LlamaIndex 的 Sub-Question Query Engine。
   - 这使得 Workflows 能够有效地在生成式 AI 应用中部署复杂的查询引擎。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **对 LLAMA 3 生成质量的担忧**：在使用 **LLAMA 3 8B instruct model** 时，一位成员发现使用“anything”进行提示会导致意外输出，引发了对生成质量的担忧。
   - 他们引导其他人分享经验，或参考 [GitHub issue #1285](https://github.com/pytorch/torchtune/issues/1285) 进行进一步讨论。
- **评估 RTX A4000 和 A2000 的 Fine Tuning 表现**：讨论强调了 **RTX A4000** 和 **RTX A2000**（均配备 **16GB** 显存）的性能特征，显示在 **1.5B** 模型上的 Fine Tuning 结果不尽如人意。
   - 一位成员建议增加默认的 **batch size** 以更好地管理内存开销，可能使工作负载适应 **12GB** 环境。
- **内存优化参数正在审查中**：目前关于内存优化参数存在一些**推测**，虽然 **LoRA** 非常有效，但目前并未被优先考虑。
   - 优化的潜力显而易见，特别是对于使用 **8GB VRAM** GPU 的成员，性能提升可能超过 **2x**。
- **关于 RLHF 清理工作的讨论**：一位成员提出了关于在公开分享前对 **RLHF** 进行必要清理的问题，并回顾了早期关于所需调整的笔记。
   - 他们表示愿意合作编写 **tutorial** 或 **blog post**，并承认这需要投入大量精力。
- **宣传和记录工作的计划**：一位成员渴望发起关于**公开宣传**其工作并开发**文档**或 **tutorials** 的讨论，并概述了一个初步路线图。
   - 他们欢迎社区的投入和协助以增强这些努力，表明将采取集体协作的方式。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **在 AI Infrastructure 上构建的自由**：成员们讨论了只要没有商业化意图，在 **AI Infrastructure** 上部署任何内容都是可以接受的，并引用了 [pricing page](https://link.to.pricing)。
   - 内部工具的使用似乎没问题，只要目标不是商业化，但相关指南仍有些模糊。
- **VS Code + WSL：Mojo 的黄金搭档**：一位用户探索了在 Windows 开发环境中使用 WSL 上的 **Mojo Max** 运行 **Mojo**，并推荐使用 **VS Code** 无缝桥接 Windows 和 Linux。
   - 利用这种设置，*你几乎会忘记自己是在 Linux 中开发*，尽管在可复现性方面存在一些限制。
- **FancyZones 提升工作流管理**：一位成员介绍了 [FancyZones utility](https://learn.microsoft.com/en-us/windows/powertoys/fancyzones)，通过将应用程序吸附到定义区域来增强 Windows 上的窗口管理，从而提高生产力。
   - 该工具允许高效利用屏幕，帮助开发者在多窗口设置中简化工作流。
- **Active Directory：并非真正的 distributed database**：关于是否将 **Active Directory** 称为 **distributed database** 展开了一场幽默的辩论，成员们指出尽管它被贴上了此类标签，但缺乏真正的 consistency 等特征。
   - 随后出现了关于 Windows 上现有 **distributed databases** 的进一步讨论，展示了社区对澄清术语的兴趣。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **用于 LLM 评估的 Inspect 工具预告**：一位成员询问了关于 [Inspect](https://github.com/UKGovernmentBEIS/inspect_ai) 在 **LLM observability** 方面的应用，寻求与 **DSPy** 集成的见解。
   - 虽然尚未有经验分享，但该工具的定位似乎是为了增强 **large language model evaluations**。
- **DSPy 相比 Langgraph 的优势**：一个明显的区别是 **DSPy** 优化了 prompt space 指令，而 **LangGraph** 在 **LangChain** 架构中处于较低层级。
   - 从本质上讲，**DSPy** 侧重于性能提升，而 **LangGraph** 处理系统级接口。
- **optimize_signature 胜过 COPRO**：用户报告称，在 GSM8K 的 **Chain of Thought** 任务中，**optimize_signature** 的表现优于 **COPRO**，获得了 **20/20** 的分数。
   - 相比之下，**COPRO** 难以获得 zero-shot 指令解决方案，最高分仅为 **18/20**。
- **用户寻求 DSPy-Multi-Document-Agent 的帮助**：一位成员在查找 **DSPy-Multi-Document-Agent** 的 **requirements.txt** 时遇到困难，询问是否遗漏了关键文件。
   - 这一询问指向了潜在的文档缺失或资源链接不清晰的问题。
- **对使用 qdrant_dspy 进行高级检索的兴趣**：[qdrant_dspy GitHub repository](https://github.com/vardhanam/qdrant_dspy) 的链接强调了使用 **Gemma-2b**、**DSPy** 和 **Qdrant** 构建 **RAG pipelines**。
   - 另一个资源 [dspy/retrieve/qdrant_rm.py](https://github.com/stanfordnlp/dspy/blob/main/dspy/retrieve/qdrant_rm.py) 强调了 **DSPy** 在本地 **VectorDB** 编程中的实用性。

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **getenv 函数触发 ValueError**: 一位用户在导入时遇到了 `ValueError`，具体为 **'invalid literal for int() with base 10: WARN'**，这指向了一个**环境变量**问题。
   - 一位成员建议检查**环境变量**会有所帮助，并确认将 **DEBUG** 变量设置为 **'WARN'** 是问题的根源。
- **DEBUG 变量引发麻烦**: 尽管用户的 Python 脚本运行良好，但在 notebook 环境中将 **DEBUG** 环境变量设置为 **'WARN'** 会导致 getenv 函数出现问题。
   - 这凸显了 tinygrad 在 notebook 和独立脚本环境之间潜在的兼容性差异。
- **Tinygrad Tensor Puzzles 挑战发布**: 成员们介绍了 **Tinygrad Tensor Puzzles**，这是一个包含 **21 个有趣谜题**的集合，旨在从第一性原理出发掌握 tinygrad 等张量库，避免使用魔法函数。
   - 该项目基于 **Sasha 的 PyTorch Tensor-Puzzles**，鼓励新手和资深开发者共同参与，培养问题解决者社区。
- **探索 Tinygrad 内部机制的教程**: 分享了一套[教程](https://mesozoic-egg.github.io/tinygrad-notes/)，旨在增强对 tinygrad 内部机制的理解并促进贡献，同时还提供了一份[快速入门指南](https://github.com/tinygrad/tinygrad/blob/master/docs/quickstart.md)以获取基础见解。
   - 虽然这些资源并非完全针对初学者，但为希望有效参与 tinygrad 开发的开发者提供了必要的知识。
- **利用计算机代数技术优化 Tinygrad**: 最近的讨论涉及与 tinygrad 优化过程相关的[计算机代数学习笔记](https://github.com/mesozoic-egg/computer-algebra-study-notes/tree/main)，增强了潜在的性能洞察。
   - 这种整合展示了可以支持开发者改进 tinygrad 能力的有价值的方法论。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **寻求开源视觉模型**: 成员们正积极寻求适用于**视觉任务**的**开源模型**建议，并询问有关本地和 API 实现方案。
   - 一位成员通过询问社区内此类模型的可用性和性能表现表达了好奇。
- **MiniCPM-V 2.6 在性能测试中脱颖而出**: 据报道，**MiniCPM-V 2.6** 的表现优于竞争对手，包括 **Gemini 1.5 Pro**、**GPT-4V** 和 **Claude 3.5 Sonnet**，特别是在多图应用中。
   - 欲了解更多详情，成员们分享了其 [Hugging Face 页面](https://huggingface.co/openbmb/MiniCPM-V-2_6)和 [GitHub 仓库](https://github.com/OpenBMB/MiniCPM-V)的链接。
- **询问发货更新**: 一位成员提出了关于**发货更新**的问题，表示对时间表和状态感兴趣。
   - 尽管未提供具体答案，但分享了一个相关的 [Discord 频道](https://discord.com/channels/1146610656779440188/1194880263122075688/1266055462063964191)链接以供潜在讨论。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Llama 团队在 arXiv 上参与互动**: **Llama 团队**正在 [arXiv 讨论论坛](https://alphaxiv.org/abs/2407.21783v1)上回答问题，提供了直接进行技术交流的机会。
   - 这一举措有助于更深入地了解 **Llama 3** 模型及其应用。
- **Quora 启动 Poe Hackathon**: Quora 正在举办一场线下和线上的 [Hackathon](https://x.com/poe_platform/status/1820843642782966103)，重点是利用 **Poe** 的新 **Previews 功能**构建机器人。
   - 参与者将利用 **GPT-4o** 和 **Llama 3.1 405B** 等先进的 LLM 开发创新的聊天内生成式 UI 体验。
- **探索非生成式 AI 应用**: 一位成员发起了关于**非生成式 AI** 重要性的对话，鼓励他人分享想法。
   - “你心目中关注哪些类型的 AI 应用？”这一问题激发了探索各种应用的兴趣。
- **确定了多样化的 AI 应用**: 成员们纷纷建议将**计算机视觉**、**预测**、**推荐系统**和 **NLP** 作为非生成式 AI 的关键领域。
   - 这些例子说明了 AI 技术在生成式模型之外服务于各个细分领域的**广泛频谱**。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Vercel 的停机影响 OpenRouter**：Vercel 目前面临间歇性停机，影响了 OpenRouter 服务，详见其 [状态更新](https://x.com/OpenRouterAI/status/1821267624228966781)。经过多次更新，服务已于 **东部时间下午 3:45** 恢复稳定。
   - Vercel 继续监控该问题，并确保更新将发布在 [Vercel 状态页面](https://www.vercel-status.com/)。
- **Anthropic 的高错误率已缓解**：Anthropic 一直在解决影响 **3.5 Sonnet** 和 **3 Opus** 模型的高错误率问题，并实施了缓解策略，截至 **PDT 时间 8 月 8 日 17:29**，成功率已恢复正常。
   - 他们提供了更新，确保 **Claude.ai** 免费用户的访问现已恢复，同时继续密切监控情况。

---

**Alignment Lab AI Discord** 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。

---

# 第 2 部分：频道详细摘要与链接

{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1270821680217985176)** (80 messages🔥🔥): 

> - `4bit and GGUF Models`
> - `PPO Trainer Challenges`
> - `Multi-GPU Support in Unsloth`
> - `Continuous Batching with lmdeploy and vllm`
> - `Quantization of Mistral Models` 

- **4bit 和 GGUF 模型的挑战**：关于加载合并的 **4bit GGUF 模型** 的讨论引发了关于在模型加载函数中使用 `load_in_4bit` 选项是否会导致潜在精度损失的疑问。
   - 成员们指出，使用 **4bit** 有助于减少 VRAM 消耗，但不带此选项加载 GGUF 通常会导致 OOM 错误。
- **PPO Trainer 遇到的问题**：一位成员报告了在使用 **PPO Trainer** 时遇到的困难，特别是在尝试实现自定义二进制奖励函数时收到了负 KL divergence 错误。
   - 建议包括探索 DPO 作为一种可能更简单的替代方案，尽管有人对其与 PPO 相比的性能表示担忧。
- **Unsloth 扩展到 Multi-GPU 支持**：已确认 **multi-GPU 支持** 正在向受信任的 Unsloth 支持者推出，提供诸如 VRAM 减少和速度提升等优势。
   - 人们对该功能是集成到开源仓库中还是仅限于付费订阅表示关注。
- **lmdeploy 和 vllm 中的 Continuous Batching**：成员们询问了 **lmdeploy** 中的 **continuous batching** 功能，以及在处理过程中如何处理异步请求。
   - 澄清指出，在 **vllm**（推测也包括 **lmdeploy**）中使用异步引擎意味着不需要为 batching 请求进行额外的实现。
- **Mistral 模型的 Quantization**：一位用户分享了他们成功对 **123B Mistral-Large-Instruct-2407** 模型进行 Quantization 的见解，在精度下降很小的情况下实现了显著的体积缩减。
   - 使用 **EfficientQAT** 算法使他们能够在不显著降低输出质量的情况下优化模型性能。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/danielhanchen/status/1821284720606638158">来自 Daniel Han (@danielhanchen) 的推文</a>：Llama 3.1 聊天模板的怪癖：1. “Cutting Knowledge Date” 是可选的吗？官方仓库测试没有添加它。文档添加了它，它是 “cutoff” 吗？2. BOS？我与 HF 合作添加了一个默认值...</li><li><a href="https://huggingface.co/docs/trl/main/en/how_to_train#how-to-generate-text-for-training">训练常见问题解答</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1elbn3q/quantize_123b_mistrallargeinstruct2407_to_35_gb/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://docs.vllm.ai/en/latest/dev/engine/async_llm_engine.html">AsyncLLMEngine — vLLM</a>：未找到描述</li><li><a href="https://youtu.be/TKmfBnW0mQA?si=lz2sHuGY_IXBYbN_">修复 Gemma, Llama, &amp; Phi 3 中的 bug：Daniel Han</a>：我们为 Gemma 修复了 8 个 bug，为 Llama 3 修复了多个 tokenization 问题，修复了一个 sliding window bug 并将 Phi-3 Mistral 化背后的故事，并了解我们如何...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1271027245506629695)** (1 条消息): 

> - `AI 模型中的损失函数`
> - `理解 Token 标签` 


- **澄清 Token 标签中 'y' 的含义**：一位成员询问了 `y` 的含义，质疑当 chunk 中不存在正确的 Token 时，它是否表示损失为 **0**，这与提供的图表和段落信息相对应。
   - 他们表达了对潜在误解的担忧，寻求关于该标签在 chunk 上下文中表示方式的澄清。
- **最后 logsumexp reduction 的合理性**：讨论提出了一个关于是否有必要在所有 chunk 之间执行最终 **logsumexp reduction** 的问题，特别是如果正确的 chunk 已经足以作为最终输出。
   - 该成员建议，由于其他 chunk 的交叉熵损失（cross-entropy loss）为 **0**，仅利用正确的 chunk 可能会使过程更加精简。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1270830145527222428)** (135 条消息🔥🔥): 

> - `模型加载问题`
> - `数据集处理`
> - `Hugging Face 集成`
> - `推理优化`
> - `Colab 限制` 


- **向模型输入时出错**：一位用户成功加载了模型，但在尝试与其聊天时遇到了 “error” 消息。
   - 另一位成员指出，由于目前支持有限，此问题仅发生在 **Llama 3.1 8B Instruct** 以外的模型上。
- **处理较小的数据集**：一位用户在从 Hugging Face 下载整个数据集而非较小子集时遇到了问题。
   - 在查看仓库代码行后，有人建议 fork 代码并进行修改，将处理限制在 **200 个文件**内。
- **Hugging Face 模型集成**：一位用户询问如何将他们的模型上传到 Hugging Face，并在聊天脚本中正确引用它。
   - 针对如何仅将模型权重推送到 Hugging Face 以便在新的 Colab 环境中使用，提供了支持。
- **在 A100 GPU 上优化推理**：一位用户寻求关于在 A100 80 GB 服务器上实现 LLM 快速推理的前沿技术建议。
   - 他们分享了正在使用的 **vLLM** 参数，并请求改进建议或替代方案。
- **Colab 的使用与限制**：有人对 Colab 的磁盘空间限制影响模型加载和训练工作表示担忧。
   - 讨论了升级到 Colab Pro 的选项，以及针对资源消耗较低的测试替代方法。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1vbH6h760iesRfcVQlm4-KVv1zYM5sB9k?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/ncbi/pubmed/blob/main/pubmed.py#L40">pubmed.py · ncbi/pubmed at main</a>: 未找到描述</li><li><a href="https://github.com/unslothai/studio">GitHub - unslothai/studio: Unsloth Studio</a>: Unsloth Studio。通过在 GitHub 上创建账户为 unslothai/studio 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: 微调 Llama 3.1, Mistral, Phi &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1270858117814685808)** (293 条消息🔥🔥): 

> - `用于漏洞猎取的 Harambe 工具`
> - `用于 URL 分析的 LLM`
> - `开源协作`
> - `开发挑战`
> - `生产力与睡眠模式` 


- **介绍 Harambe：漏洞猎取助手**：一名成员介绍了 **Harambe**，这是一个开源工具，旨在通过提供 HTTP 请求和响应查看器，并辅以分析和工具来协助漏洞猎手进行更好的分析。
   - 其目标是通过使用 LLM 生成合理的 API 端点建议，从而简化漏洞猎取流程，减少对传统 fuzzing 技术的依赖。
- **对 LLM 微调支持的需求**：该项目需要对多个 LLM 模型（1B、2B、3B 和 7B 参数）进行微调，以确保针对不同硬件能力进行有效的 URL 分析和词表（wordlist）创建。
   - 开发者寻求社区支持，并强调参与者不会获得金钱奖励，但会在项目文档中予以致谢。
- **开发与生产力见解**：成员们讨论了个人生产力，注意到不同的睡眠和工作安排，并强调了对深夜工作时段的偏好。
   - 一位成员反思了编程习惯，以及早年参与项目如何影响持续的兴趣和职业路径。
- **学习新的编程语言**：关于随着年龄增长学习新编程语言意愿的讨论，成员们分享了他们在高级语言与低级语言之间的经验和偏好。
   - 虽然低级语言编程提供了独特的挑战和成就感，但许多人认为高级语言更适合快速原型设计和项目开发。
- **对 AGI 开发的推测**：成员们推测了 AGI 的未来，强调了开发此类技术的复杂性和不可预测性，并提到了各种开发方法论参差不齐的结果。
   - 他们认识到，虽然追求先进的人工智能雄心勃勃，但当前的工具和社会结构主要集中在专业技能和角色上。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://noclip.website/">noclip</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/howdidtheycodeit/comments/183a5en/how_do_i_recreate_this/">Reddit - Dive into anything</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1270831460345057389)** (9 条消息🔥): 

> - `评估中的提示词分类 (Prompt Classification)`
> - `PyTorch 中的 FlexAttention`
> - `Attention 实现的挑战`
> - `非污染打包 (Non-Contaminated Packing)`
> - `Hugging Face 集成` 


- **提示词分类改进评估方法**：一项研究表明，**提示词 (prompts)** 可以有效地对对话进行分类，有助于为友好或浪漫等特定语境的语气创建评估函数。
   - 这种方法允许模型根据识别出的说话者之间的关系，提供细致的数据集评估。
- **引入 FlexAttention 以支持多种 Attention 变体**：新的 [FlexAttention](https://pytorch.org/blog/flexattention/) API 允许在 *PyTorch* 中轻松实现各种 Attention 机制，而无需进行大量的内核 (kernel) 重写。
   - 该功能解决了 ML 研究人员面临的“软件彩票 (software lottery)”问题，从而支持对融合内核 (fused kernels) 进行更多实验。
- **优化后的 Attention 实现面临的挑战**：尽管 **FlashAttention** 带来了效率提升，但由于需要自定义内核，ML 研究人员现在在实现新的 Attention 变体时面临困难。
   - 结果是在探索现有优化内核之外的变体时，会出现运行速度变慢和 CUDA OOM 问题。
- **集成非污染打包 (Non-Contaminated Packing) 的潜力**：关于 [文档掩码锯齿序列 (document masking jagged sequences)](https://pytorch.org/blog/flexattention/#document-maskingjagged-sequences) 讨论的功能可能会促进在 Unsloth 项目中添加非污染打包。
   - 这可以提高利用 Attention 机制的实现的流程和效率。
- **直接集成 Hugging Face 的提议**：一位成员提议 **Hugging Face** 应该直接集成 FlexAttention 功能以增强易用性。
   - 该建议是在讨论了 FlexAttention 为各种机器学习项目带来的改进之后提出的。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://pytorch.org/blog/flexattention/#document-maskingjagged-sequences">FlexAttention: 兼具 PyTorch 的灵活性与 FlashAttention 的性能</a>:   </li><li><a href="https://pytorch.org/blog/flexattention/">FlexAttention: 兼具 PyTorch 的灵活性与 FlashAttention 的性能</a>:   </li><li><a href="https://x.com/cHHillee/status/1821253769147118004">Horace He (@cHHillee) 的推文</a>: 长期以来，用户一直生活在融合 Attention 实现的“软件彩票”暴政之下。不再如此。现在推出 FlexAttention，这是一个新的 PyTorch API，允许实现许多 Attention 变体...
</li>
</ul>

</div>

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1271232340126863410)** (1 messages): 

> - `BiRefNet Background Removal` (BiRefNet 背景移除)
> - `ActionGemma Model for Function Calling` (用于 Function Calling 的 ActionGemma 模型)
> - `Unity ML-Agents`
> - `Segment Anything Model Insights` (Segment Anything Model 见解)
> - `ArabicWeb24 Dataset` (ArabicWeb24 数据集)


- **BiRefNet 性能超越 RMBG1.4**：感谢 [南开大学](https://huggingface.co/ZhengPeng7/BiRefNet) 团队，**BiRefNet** 在背景移除方面的表现优于 **RMBG1.4**。
   - 该模型利用双边参考（bilateral reference）进行高分辨率二分图像分割，更多细节请参阅 [arXiv 论文](https://arxiv.org/pdf/2401.03407)。
- **推出 ActionGemma 模型**：[ActionGemma](https://huggingface.co/KishoreK/ActionGemma-9B) 是一个经过微调的 9B 模型，专为 Function Calling 定制，结合了 Gemma 的多语言能力与来自 xLAM 数据集的 Function Calls。
   - 该模型由 **KishoreK** 开发，融合了跨多种语言的强劲性能。
- **探索 Unity ML-Agents**：一段新的 [YouTube 视频](https://youtube.com/live/J-de9K_3xDw?feature=share) 详细介绍了如何使用 Unity ML-Agents 从零开始预训练大语言模型（LLM）。
   - 这个引人入胜的教程将引导观众利用尖端技术构建智能聊天机器人。
- **来自 Segment Anything Model 的见解**：最近的博客文章讨论了 **CLIP** 和 **ALIGN** 等视觉模型在分割任务背景下的能力，这些模型的发展速度尚未达到文本模型的水平。
   - 关键讨论包括核心计算机视觉任务在进展中面临的挑战，以及探索通过提示词工程（engineered prompts）来改进结果。
- **高质量阿拉伯语数据集：ArabicWeb24**：一篇新博客介绍了 [ArabicWeb24](https://huggingface.co/blog/MayFarhat/arabicweb24)，这是一个专为高质量阿拉伯语预训练量身定制的数据集。
   - 该资源对于开发需要广泛阿拉伯语理解能力的应用程序至关重要。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/ZhengPeng7/BiRefNet">ZhengPeng7/BiRefNet · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/KishoreK/ActionGemma-9B">KishoreK/ActionGemma-9B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/DamarJati/FLUX.1-DEV-Canny">FLUX.1-DEV Canny - a Hugging Face Space by DamarJati</a>: 未找到描述</li><li><a href="https://youtube.com/live/J-de9K_3xDw?feature=share">Unity ML-Agents | Pretrain an LLM from Scratch with Sentence Transformers| Part 1</a>: 欢迎来到我们使用 Unity ML-Agents 和 Sentence Transformers 创建智能聊天机器人的激动人心之旅！🚀在本视频中，我们将带您了解...</li><li><a href="https://www.lightly.ai/post/segment-anything-model-and-friends">Segment Anything Model and Friends</a>: Segment Anything Model (SAM) 及其继任者在计算机视觉领域取得了重大飞跃，特别是在图像和视频分割方面。伴随着 SAM 创新的可提示（promptable）方法...</li><li><a href="https://huggingface.co/spaces/Delik/Anitalker">Anitalker - a Hugging Face Space by Delik</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/LEGENT/LEGENT">LEGENT - a Hugging Face Space by LEGENT</a>: 未找到描述</li><li><a href="https://www.lightly.ai/post/using-self-supervised-learning-for-dense-prediction-tasks">Using Self-Supervised Learning for Dense Prediction Tasks</a>: 用于目标检测、实例分割和语义分割等密集预测任务的自监督学习 (Self-Supervised Learning) 方法概述。</li><li><a href="https://dev.to/tonic/dockers-testcontainers-are-great-42cl">no title found</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/prithivMLmods/lora-adp-01">Unlocking Creativity with Text-to-Image Generation: Exploring LoRA Models and Styles</a>: 未找到描述</li><li><a href="https://youtu.be/fnKrReaqQgc">How does Uber predict Arrival Times (ETA) for trips? | Uber ML System Design | #systemdesign</a>: 你知道像 Uber、Ola 和 Lyft 这样的网约车公司是如何预测行程的预计到达时间 (ETA) 的吗？在本视频中，我们设计了一个端到端的机器...</li><li><a href="https://huggingface.co/blog/MayFarhat/arabicweb24">ArabicWeb24: Creating a High Quality Arabic Web-only Pre-training Dataset </a>: 未找到描述</li><li><a href="https://github.com/Rivridis/LLM-Assistant">GitHub - Rivridis/LLM-Assistant: Locally running LLM with internet access</a>: 具有互联网访问权限的本地运行 LLM。通过在 GitHub 上创建账号为 Rivridis/LLM-Assistant 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1270819545782685718)** (224 条消息🔥🔥): 

> - `LLM 应用中的 Web 搜索功能`
> - `动画克隆化身`
> - `AI 模型性能`
> - `Discord 与论坛在沟通方面的对比`
> - `Minecraft 服务器体验` 


- **Google 和 Bing 的 Web 搜索替代方案**：成员们讨论了在 LLM 应用中用于 Web 搜索功能的 Google 和 Bing 替代方案，重点介绍了 [DuckDuckGo's API](https://duckduckgo.com)，它在构建自定义搜索模型时被广泛使用且免费。
   - 另一位成员提到 [Brave Search API](https://brave.com/search/api/) 是驱动搜索功能的另一个可行替代方案。
- **使用 AI 创建动画克隆化身**：一位用户询问了能够利用视频数据生成动画克隆化身，并根据文本输入进行对口型（lip-syncing）的 AI 模型，特别提到了 [Rask.ai](https://rask.ai)。
   - 有人建议使用 Wav2Lip，但也有人推荐探索 [SeamlessM4T-v2](https://huggingface.co/docs/transformers/model_doc/seamless_m4t_v2)，因为它在翻译方面具有广泛的功能。
- **AI 模型性能评估**：成员们分享了使用预训练翻译模型（如 [Facebook's M2M100](https://huggingface.co/facebook/m2m100_418M) 和 [SeamlessM4T](https://huggingface.co/docs/transformers/model_doc/seamless_m4t_v2)）的经验，认为它们是多语言翻译的潜在合适选择。
   - 讨论还包括了 SeamlessM4T-v2 与 Whisper 模型在转录能力方面的比较。
- **Discord 聊天与论坛沟通**：用户对使用 Discord 线程进行严肃讨论表达了复杂的情绪，指出与传统论坛相比，跟踪对话较为困难。
   - 大家达成共识，认为采用结构化方法（如自动线程化或链接到消息）可以使实时聊天更易于管理且更具信息量。
- **Minecraft 服务器的怀旧体验**：一位成员分享了与朋友一起运行 Minecraft 服务器的怀旧经历，回忆起重新访问地图擦除后未受影响的旧建筑的乐趣。
   - 他们指出，尽管友谊和动态随时间发生了变化，但对作为个人空间的服务器仍怀有深厚的情感。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://ladybird.org/">Ladybird</a>: Ladybird 是一款真正独立的 Web 浏览器，由非营利组织支持。</li><li><a href="https://brave.com/search/api/">Brave Search API | Brave</a>: 为您的搜索和 AI 应用提供动力，使用自 Bing 以来增长最快的独立搜索引擎。只需一次调用即可访问数十亿页面的索引。</li><li><a href="https://huggingface.co/facebook/m2m100_418M">facebook/m2m100_418M · Hugging Face</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=qbfwadEo_t4">为什么我们将公司命名为 "Hugging Face" 🤗</a>: 在这里观看 Clem Delangue 的完整采访: https://youtu.be/BPExesApMXU #Clem Delangue #HarryStebbings #20VC #shorts #HuggingFace #ai #🤗 #generativeai ...</li><li><a href="https://youtu.be/t-NIB6L_3zk">开发者阅读笔记 2：2 分钟掌握 Django 的 20 个概念</a>: 在这个开发者笔记本系列视频中，我将介绍 Django 的 20 个基本概念，为您提供其定义和工作原理的全面概述。如果您还没有...</li><li><a href="https://python.langchain.com/v0.2/docs/integrations/tools/">工具 | 🦜️🔗 LangChain</a>: 如果您想编写自己的工具，请参阅此指南。</li><li><a href="https://huggingface.co/bakrianoo/sinai-voice-ar-stt">bakrianoo/sinai-voice-ar-stt · Hugging Face</a>: 未找到描述</li><li><a href="https://youtu.be/CoqfMasnk0A?t=1241">社区会议 - 2024 年 8 月</a>: 这是 Firefox 产品社区的月度社区会议，旨在讨论即将发布的产品版本、社区更新以及各种贡献机会...</li><li><a href="https://huggingface.co/docs/transformers/model_doc/seamless_m4t_v2">SeamlessM4T-v2</a>: 未找到描述</li><li><a href="https://github.com/huggingface/datasets/issues/7092)">Issues · huggingface/datasets</a>: 🤗 用于 ML 模型的最现成的、易于使用且高效的数据操作工具数据集中心 - Issues · huggingface/datasets</li><li><a href="https://github.com/hiyouga/LLaMA-Factory/wiki/Performance-Comparison)">首页</a>: 用于高效微调 100 多个 LLM 的 WebUI (ACL 2024) - hiyouga/LLaMA-Factory</li><li><a href="https://www.shadertoy.com/view/XfByW3,">错误 - Shadertoy BETA</a>: 未找到描述</li><li><a href="https://www.shadertoy.com/view/XfBcWc">Shadertoy</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1270839917597036659)** (4 messages): 

> - `Neural Network Optimization`
> - `AI in Healthcare`
> - `Embedding Serialization and Deserialization` 


- **Neural Network Optimization with Layer-Wise Scaling**: 在实施 **layer-wise scaling** 以最小化 **outlier features** 后，模型的 **1b loss** 现在停留在 **3.8**，相比之前的 **5.0** 有所改善。
   - **50m loss** 不再发散但收敛缓慢，并且观察到 **attention entropy** 同时崩溃，这表明收敛速度较慢。
- **Exploring Future of AI in Healthcare**: 观看这段 [YouTube 视频](https://www.youtube.com/watch?v=Z--q7RO2TrU)，埃默里大学的 Andrew Janowczyk 教授在视频中讨论了 **AI in Healthcare**。
   - 他隶属于 **Emory Precision AI for Health Institute**，并分享了他近 **15 年** 丰富经验中的见解。
- **Learning to Serialize and Deserialize Embeddings**: 一位成员正专注于学习在 Python 和 C# 之间 **serialize and deserialize embeddings data** 的技术。
   - 这一过程对于 AI 应用中有效的数据处理至关重要。



**Link mentioned**: <a href="https://www.youtube.com/watch?v=Z--q7RO2TrU">The Future of AI in Healthcare, ft. Professor Andrew Janowczyk (Emory University)</a>: Andrew Janowczyk 博士是 Emory Precision AI for Health Institute 的助理教授，也是日内瓦大学医院的数据分析师。拥有近 15 年的...

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1271023723188584450)** (3 messages): 

> - `Transformers Architecture`
> - `EU AI Regulations`
> - `AI Risk-based Regulation` 


- **Dominance of Transformers Explained**: 一篇资讯丰富的文章详细介绍了为什么 **Transformers** 是当前时代的主导架构，并强调了它们的性能优势。
   - 它提供了关于 **attention mechanisms** 和架构优势的直观见解，支持了它们的广泛采用。
- **EU's New AI Regulations Roll Out**: **欧盟** 的 **AI Act** 已于 2024 年 8 月 1 日正式生效，开启了各种 AI 应用合规的时间表。
   - 这包括一系列交错的合规截止日期，其中第一项是在六个月后开始禁止某些 AI 用途，例如执法中的 **remote biometrics**。
- **Implications of Staggered Compliance Deadlines**: **risk-based regulation** 框架将于 2026 年年中全面实施，影响不同类型的 AI 开发者和应用。
   - 大多数关于 AI 应用的条款都将要求合规，这标志着各行业利用 AI 的方式将发生重大转变。



**Link mentioned**: <a href="https://techcrunch.com/2024/08/01/the-eus-ai-act-is-now-in-force/">The EU&#039;s AI Act is now in force | TechCrunch</a>: 欧盟针对人工智能应用的基于风险的监管条例（risk-based regulation）已于今日起生效。

  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1270860414204252161)** (16 messages🔥): 

> - `EurekAI 平台`
> - `Gemma2 9B 微调`
> - `Flux.1 Dev Controlnet Canny`
> - `文本生成图像扩散模型 (Text-to-Image Diffusion Models)`
> - `TTS 优化` 


- **EurekAI：研究者的 GitHub**：一位成员宣布推出了 [EurekAI](https://eurekai-web-app.vercel.app/signup)，这是一个通过 AI 功能促进研究人员之间跨界协作的平台，旨在简化研究流程。
   - 目前处于 alpha 阶段，该平台承诺提供项目创建和集成日志等功能，以提高研究的生产力和参与度。
- **为 Function Calling 微调 Gemma2**：一位用户正在使用 xLAM 数据集微调 [Gemma2 9B 模型](https://huggingface.co/KishoreK/ActionGemma-9B)，旨在为多种语言创建一个通用的 Function Calling 模型。
   - 有人对该模型的公开访问权限提出了疑问，并指出链接中的表情符号可能导致了问题。
- **Flux.1 的新 Hugging Face Space**：一位成员利用 GitHub 仓库的代码为 [Flux.1 Dev Controlnet Canny](https://huggingface.co/spaces/DamarJati/FLUX) 创建了一个 Hugging Face Space。
   - 该项目旨在以易于访问的方式展示其能力，作为不断增长的 Hugging Face Spaces 集合的一部分。
- **Apple 的高效扩散模型训练**：来自 Apple 的一位研究员介绍了一个用于高效训练文本生成图像扩散模型的新 Python 包，链接指向他们的 [ICLR 2024 论文](https://github.com/apple/ml-mdm)。
   - 该包旨在在有限数据下实现最佳性能，在 Machine Learning 领域提供了一种新颖的方法。
- **TTS 优化博客文章**：一位成员分享了他们的 [Medium 博客](https://medium.com/@mllopart.bsc/optimizing-a-multi-speaker-tts-model-for-faster-cpu-inference-part-1-165908627829)，详细介绍了 TTS 优化，旨在提高 CPU 推理性能。
   - 随后讨论了关于提供的链接可能存在的访问问题和格式问题。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/apple/ml-mdm">GitHub - apple/ml-mdm: 以数据和计算高效的方式训练高质量的文本生成图像扩散模型</a>：以数据和计算高效的方式训练高质量的文本生成图像扩散模型 - apple/ml-mdm</li><li><a href="https://huggingface.co/KishoreK/ActionGemma-9B">KishoreK/ActionGemma-9B · Hugging Face</a>：未找到描述</li><li><a href="https://youtube.com/live/J-de9K_3xDw?feature=share">Unity ML-Agents | 使用 Sentence Transformers 从头开始预训练 LLM | 第 1 部分</a>：欢迎来到我们使用 Unity ML-Agents 和 Sentence Transformers 创建智能聊天机器人的激动人心之旅！🚀 在本视频中，我们将带您了解...</li><li><a href="https://huggingface.co/spaces/DamarJati/FLUX.1-DEV-Canny">FLUX.1-DEV Canny - DamarJati 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://github.com/XLabs-AI/x-flux">GitHub - XLabs-AI/x-flux</a>：通过在 GitHub 上创建账户来为 XLabs-AI/x-flux 的开发做出贡献。</li><li><a href="https://eurekai-web-app.vercel.app/signup).">EurekAI App</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1271056966084464640)** (4 messages): 

> - `Transformers 架构`
> - `AI 实验中的新想法` 


- **为每个人简化的 Transformers**：一位成员分享了一篇 [Medium 文章](https://medium.com/@jatinprrrrt/transformers-made-simpler-a-overview-d71585f45fe6)，用更简单的术语解释了 **Transformers 架构**，并寻求反馈以验证他们的理解。
   - 他们强调了通过分享进行学习的重要性，希望吸引他人提供反馈。
- **具有前景结果的令人兴奋的新实验**：另一位成员对一个尚未尝试过的**极具前景的想法**表示兴奋，并指出他们的第一次实验效果非常好。
   - 虽然没有分享具体细节，但这种热情暗示了该领域潜在的重大进展。


  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1271017097106948208)** (9 条消息🔥): 

> - `Papers with Code 用于 Computer Vision`
> - `将手写内容转换为笔画格式 (Stroke Format)`
> - `IAM On-Line Handwriting Database` 


- **Papers with Code 提供丰富资源**：一位成员推荐了 [Papers with Code](https://paperswithcode.com/sota) 作为总结 Computer Vision 领域最新进展 (SOTA) 的宝贵资源，该平台包含 **11,272 个基准测试 (benchmarks)**、**5,031 个任务**以及 **137,097 篇带有代码的论文**。
   - 该平台汇集的数据可协助用户探索机器学习应用的各个方面。
- **将手写内容转换为笔画格式**：一位成员询问了如何将手写图像转换为由包含颜色和时间细节的单个笔画组成的笔画格式的方法。
   - 他们强调需要将此格式与 SVG 区分开来，目标是实现单笔画表示。
- **IAM On-Line Handwriting Database 详情**：一位成员分享了 IAM On-Line Handwriting Database 的链接，提供了有关其数据格式的信息，重点介绍了存储目录以及包含关键细节的 XML 文件。
   - XML 文件包含唯一的表单 ID、作者 ID 以及全面的时间采集数据，详见提供的 [数据格式文档](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database/data-format)。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database/data-format">Research Group on Computer Vision and Artificial Intelligence &mdash; Computer Vision and Artificial Intelligence</a>: 未找到描述</li><li><a href="https://paperswithcode.com/sota">Papers with Code - Browse the State-of-the-Art in Machine Learning</a>: 11272 个排行榜 • 5031 个任务 • 10360 个数据集 • 137097 篇带有代码的论文。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1271034157610303590)** (2 条消息): 

> - `AutoProcessor 可用性`
> - `InternLM 2.5 特性` 


- **关于模型 AutoProcessor 的查询**：一位成员询问如何检查哪些模型关联了 **AutoProcessor**，并提到了 **InternLM** 的相关问题。
   - 他们尝试使用 [InternLM 的仓库](https://huggingface.co/internlm/internlm2_5-7b-chat)，但在调用 `AutoProcessor` 时遇到了错误，提示缺少识别的处理类。
- **InternLM 2.5 模型亮点**：**InternLM 2.5** 拥有卓越的推理能力，在数学推理任务中表现优于 **Llama3** 和 **Gemma2-9B** 等其他模型。
   - 它具有 **1M 上下文窗口**，在长上下文任务中表现出色，并支持从 **100 多个网页**中收集信息。



**提到的链接**: <a href="https://huggingface.co/internlm/internlm2_5-7b-chat">internlm/internlm2_5-7b-chat · Hugging Face</a>: 未找到描述

  

---

### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1270942893452038154)** (29 messages🔥): 

> - `Flux Transformer Training`
> - `Using Multiple GPUs`
> - `LoRA Training`
> - `CUDA Resource Management` 


- **在 RTX GPU 上运行 Flux Transformer 的挑战**：成员们讨论了在 RTX 4090 上加载大型 **FluxTransformer2DModel** 的困难，特别是在尝试利用多 GPU 时。
   - _CUDA 应该自动处理 VRAM 分配_，但用户仍面临模型大小超过可用显存的问题。
- **关于有限 VRAM 训练的建议**：建议训练 **LoRAs** 而不是全量模型，因为后者需要大量的 GPU 资源，且多 GPU 设置尚未得到高效实现。
   - 一位用户指出训练 LoRAs 并不简单，并对需要多少额外空间表示担忧。
- **关于模型加载设备映射（Device Mapping）的困惑**：对于使用 `device_map` 在 GPU 之间拆分模型存在困惑，共识是目前的实现仅能分布不同的模型，而不能拆分单个模型的片段。
   - 尽管尝试使用 `device_map='auto'`，用户仍遇到提示该功能尚不支持的错误。
- **资源效率讨论**：用户讨论了在 GPU 之间拆分模型与池化资源以实现最佳集成的效率问题，并建议确保正确设置 **CUDA 和 Flash attention**。
   - 一位成员指出，拥有 **120 亿参数**，即使是 fp16 模型也可能因潜在的 VRAM 共享导致性能缓慢。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/RisingSayak/status/1821510483729318197">来自 Sayak Paul (@RisingSayak) 的推文</a>：如果你有多个 GPU 并希望在运行 FLUX 推理时共享它们，这在 diffusers 中是可行的。在这里，我模拟了三个 GPU，分别为 16G、16G 和 24G...</li><li><a href="https://huggingface.co/spaces/tori29umai/sketch2lineart">Sketch2lineart - 由 tori29umai 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/docs/diffusers/main/en/tutorials/inference_with_big_models#device-placement.">处理大模型</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1271174813825765417)** (1 messages): 

> - `Gradio v4.41 Release`
> - `New Features in Gradio`
> - `Security Improvements`
> - `Bug Fixes`
> - `Documentation Enhancements` 


- **Gradio v4.41 发布，带来令人兴奋的特性**：新版本 **v4.41** 引入了多项增强功能，包括为 `gr.Image` 等组件提供 **全屏图像** 查看以及改进的输出查看选项。
   - 此次更新旨在通过新按钮简化用户体验，以便更轻松地查看和复制输出。
- **Gradio 中的绘图改进**：更新版本包含针对图表的 **文档改进**，以及设置 `.double_click()` 事件监听器以实现更好交互的能力。
   - 这一增强功能让开发者能够更好地控制用户与可视化数据的交互方式。
- **实施增强的安全措施**：v4.41 中的 **安全修复** 显著收紧了 CI，防止未经授权的文件访问以及 XSS 攻击，提升了平台的整体安全性。
   - 更改还包括围绕设置 `share=True` 的改进安全机制，确保更安全的用户交互。
- **交付大量 Bug 修复**：此版本解决了包括 **gr.Model3D**、**gr.Chatbot** 和 **gr.Interface** 在内的各种组件中的 Bug，确保运行更加顺畅。
   - 有关修复的完整列表，用户可以参考 [changelog](https://www.gradio.app/changelog)，其中详细列出了所有更改。


  

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1270833978412240956)** (14 messages🔥): 

> - `Profiling CUDA`
> - `BPF wizardry`
> - `Nsight tools`
> - `eBPF for GPU monitoring` 


- **关于使用 BPF 进行 CUDA Profiling 的咨询**：一名成员询问是否有人正在使用 **BPF** 来对 **CUDA** 进行性能分析。
   - 一些成员澄清说，**eBPF** 是操作系统内核的一部分，缺乏对 GPU 活动的可见性。
- **Nsight 工具在 CUDA Profiling 方面非常高效**：另一名成员分享了他们的积极体验，推荐使用 **Nsight Compute** 来了解单个 Kernel 的性能，以及使用 **Nsight Systems** 来监控整个 CPU/GPU 应用程序。
   - 一位用户在提到 **Nsight** 时表示：*它很好用，能完成任务。*
- **eBPF 在 GPU 监控方面的潜力**：有人建议将 **eBPF profiling** 用于在生产环境中监控 GPU 集群（fleets）。
   - 然而，对于 **Nsight Systems** 在此特定用途上的有效性，人们提出了质疑。
- **对 Nsight 偏好的好奇**：一名成员询问了关于 **Nsight** 的看法，促使另一名成员肯定了它在 CUDA 性能评估中的实用性。
   - 成员请求了特定资源的链接，但未获提供。
- **eBPF 的奇技淫巧 (Wizardry)**：讨论中提到了在 CUDA Profiling 中有效实现 eBPF 所需的“奇技淫巧”。
   - 对话反映出人们对 GPU 场景下高级 Profiling 技术的兴趣日益浓厚。


  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1270874788608348170)** (31 messages🔥): 

> - `Attention Gym issues`
> - `Integration of FlexAttention`
> - `Torch serialization challenges`
> - `Flash Attention and Paged Attention connection` 


- **Attention Gym 链接失效**：成员们注意到 softcapping 章节下的 **Attention Gym** 链接目前无法访问。
   - *刚刚读完！* 再次强调了文章解释的全面性和详细程度。
- **FlexAttention 集成计划**：一名成员询问了关于帮助将 **FlexAttention** 集成到 HF 模型中的计划，并提到了之前在使用 PyTorch 新特性时遇到的困难。
   - 他们还讨论了 **HF** 可能会等到 **2.5** 版本发布后才会继续推进。
- **Torch 序列化的复杂性**：成员们表达了对 **Torch serialization** 的担忧，其中一人指出这个问题一直困扰着他们，并提到该问题在 Nightly 版本中已修复。
   - 此外，还提供了一个变通方案，建议使用 `model.compile()` 而不是 `torch.compile()`，以避免 state dict 带来的复杂问题。
- **关于 autocast 设备类型的查询**：一名用户询问在未指定设备类型时，**torch.autocast** 的默认设备类型是否为 CUDA。
   - 这一询问得到了理解和关注，但未提供进一步的澄清。
- **Flash Attention 兼容性**：一名成员询问为什么 **Flash Attention** 不能与 **Paged Attention** 一起使用。
   - 回复指出，**FlashAttention** 的初始实现不支持 Paged Attention，这需要后续对 Kernel 进行修改。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://dev-discuss.pytorch.org/t/how-to-bring-compile-time-down-to-zero-our-plans-and-direction-may-14th-edition/2089">如何将编译时间降至零：我们的计划和方向（5 月 14 日版）</a>：我们很高兴地宣布，在 2024 年上半年，我们一直优先改进 torch.compile 工作流的编译时间。快速迭代和高效的开发周期...</li><li><a href="https://discuss.pytorch.org/t/how-to-serialize-models-with-torch-compile-properly/175626/2">如何正确地使用 torch.compile 序列化模型</a>：目前还没有针对 torch.compile 的序列化解决方案，但这是高优先级事项。关于这两个论坛，这更多是一个面向普通 PyTorch 用户的社区论坛，而开发论坛则更多面向 PyTorch 开发者...</li><li><a href="https://github.com/pytorch-labs/attention-gym/blob/bf80fecf39edee616be620ed6204aec786403b9a/attn_gym/masks/causal.py#L5">attention-gym/attn_gym/masks/causal.py (GitHub)</a>：用于处理 flex-attention 的实用工具和示例 - pytorch-labs/attention-gym</li><li><a href="https://github.com/pytorch/pytorch/pull/120143">由 drisspg 添加滑动窗口 Attention Bias · Pull Request #120143 · pytorch/pytorch</a>：摘要：此 PR 添加了一个新的 attention-bias torch_function，旨在与 SDPA 交互。这实现了滑动窗口并更新了 &quot;aten.sdpa_flash&quot; 以暴露 window_size_left ...
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

iron_bound: 代码终于发布了 https://github.com/Aleph-Alpha/trigrams
  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1271149203116261426)** (1 messages): 

> - `2D Conv Kernels`
> - `Constant Memory Usage`
> - `Dynamic Kernel Sizes` 


- **2D Conv 中的 Kernel 大小导航**：一位成员询问框架在利用 `__constant__` 内存（这需要预先知道 kernel 大小）时，如何提供选择 **kernel sizes** 的灵活性。
   - 他们想知道常见的大小是否是预编译的，以及对于不太常见的变体是否采用了动态 kernel 大小。
- **理解卷积中的内存管理**：讨论强调了在处理 **2D convolutional kernels** 和 constant memory 使用时内存管理的重要性。
   - 参与者分享了关于优化 **filter allocations** 以及通过有效的管理策略减少开销的见解。


  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1270870741104988171)** (6 messages): 

> - `Release of torchao v0.4.0`
> - `Intx Tensor Subclasses Quantization`
> - `Update on issue #577`
> - `ModelRunner.run() complexity` 


- **torchao v0.4.0 正式发布！**：torchao 的 **0.4 版本** 已经发布，增加了 **KV cache quantization** 和 **quantization aware training (QAT)** 等功能，可以提升性能。
   - 强调了该版本包含对 **low bit optimizer** 功能的支持，这对用户来说是一个令人兴奋的更新。
- **关于 Intx Tensor Subclasses Quantization 的讨论**：一个正在进行的 **GitHub issue (#439)** 旨在 PyTorch 中实现 sub byte unsigned integer quantization 基准，以便进行低比特量化实验。
   - 有人请求在 tracker 中添加更多项目，并向社区征求建议。
- **寻求关于 issue #577 进展的澄清**：提供了 **GitHub issue #577** 的更新，深入探讨了在 **ModelRunner.run()** 中使用新的 **MultiTensor** 调用 **call_function** 的复杂性。
   - 作者希望对其在 **annotated_gptq** 上的工作进行审查，寻求指导以确保方向正确。
- **激活的模型执行挑战**：详细解释指出，按顺序为每个线性层运行模型可能会导致执行时间缓慢，从而引发对性能的担忧。
   - 建议的方法是在每个线性层之后暂停以更新权重，然后再继续，这可能会提高效率。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/ao/issues/577:">Issues · pytorch/ao</a>: 用于训练和推理的缺失的 pytorch dtype 和 layout 库 - Issues · pytorch/ao</li><li><a href="https://github.com/pytorch/ao/releases/tag/v0.4.0">Release v0.4.0 · pytorch/ao</a>: v0.4.0 亮点 我们很高兴地宣布 torchao 0.4 版本发布！此版本增加了对 KV cache quantization、quantization aware training (QAT)、low bit optimizer 的支持，以及组合...</li><li><a href="https://github.com/pytorch/ao/issues/439">[RFC] Intx Tensor Subclasses Quantization · Issue #439 · pytorch/ao</a>: 目标：实现 1-7 的 sub byte unsigned integer quantization 基准，使用户能够在 pytorch 中实验低比特量化。Tracker：根据 #391 创建一个 UIntx Tensor Subclass...
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1270819014159110207)** (179 条消息🔥🔥): 

> - `KV Cache 实现`
> - `RoPE 优化`
> - `训练效率`
> - `微调策略`
> - `代码清理与重构` 


- **通过 KV Cache 实现改善内存占用**：一位成员实现了 KV Cache 以优化内存使用，从而允许在单个 80GB GPU 上以 batch size 1 和序列长度 4096 进行全量 bfloat16 微调。
   - 尽管有此增强，它仍触及了内存极限，引发了关于使用 managed memory 等内存优化策略的讨论。
- **RoPE 实现分析**：有关于简化 RoPE 实现以避免复数的讨论，倾向于使用直接的三角函数运算以提高可读性和可维护性。
   - 成员们一致认为，虽然可以对复杂的抽象进行注释，但直接使用 sin/cos 可能更易于理解。
- **训练逻辑与重构机会**：有人对某些训练逻辑提出了担忧，特别是 generate 函数中的一个条件，为了清晰起见，可以将其简化或删除。
   - 确定了重构机会以增强代码的可管理性，特别是在训练循环方面。
- **微调策略与内存管理**：关于微调的讨论显示，由于模型训练期间的内存限制，使用 qlora 等策略已变得流行。
   - 分享了一个 Pull Request，旨在分配 managed memory 以缓解设备内存耗尽时的问题，从而方便在较小的系统上进行较慢的训练。
- **代码改进协作**：成员们计划在初始草案合并后，提交专注于代码清理和重构的小型 PR。
   - 这种方法旨在随着项目的发展增强协作并保持代码质量。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/jrahn/gpt2_350M_edu_hermes">jrahn/gpt2_350M_edu_hermes · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/karpathy/llm.c/pull/709">Allocate managed memory if device memory runs out by ngc92 · Pull Request #709 · karpathy/llm.c</a>：如果设备内存耗尽，则使用 cudaMallocManaged 分配优化器状态，这样即使无法容纳优化器状态，我们仍然可以（缓慢地）进行训练。这基于 #694，应该是 m...</li><li><a href="https://github.com/pytorch/torchchat/blob/fe73ef737e84794694bd7c48a4b6bd0fd9028cb2/build/model.py">torchchat/build/model.py at fe73ef737e84794694bd7c48a4b6bd0fd9028cb2 · pytorch/torchchat</a>：在服务器、桌面和移动设备上本地运行 PyTorch LLM - pytorch/torchchat</li><li><a href="https://github.com/karpathy/llm.c/pull/725/files">Add LLaMA 3 Python support by gordicaleksa · Pull Request #725 · karpathy/llm.c</a>：在我们的 Python 代码中添加 LLaMA 3 支持作为参考。该代码目前仅支持推理，且与 nano llama 3 等效。</li><li><a href="https://github.com/karpathy/llm.c/pull/730">Demo equivalence - tmp by gordicaleksa · Pull Request #730 · karpathy/llm.c</a>：抄送：@karpathy，这是在我 PR 基础上的最小改动，提供了与 commit karpathy/nano-llama31@d0dfb06 中的 nano llama 3 reference.py 等效的代码。步骤：检出此 PR，检出...</li><li><a href="https://github.com/karpathy/llm.c/pull/730/commits/298a49ac61a219f0be4a681ad4c3175ec0a95f2f">Demo equivalence - tmp by gordicaleksa · Pull Request #730 · karpathy/llm.c</a>：抄送：@karpathy，这是在我 PR 基础上的最小改动，提供了与 commit karpathy/nano-llama31@d0dfb06 中的 nano llama 3 reference.py 等效的代码。步骤：检出此 PR，检出...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1270834387050565724)** (179 条消息🔥🔥): 

> - `Perplexity Pro Limits` (Perplexity Pro 限制)
> - `API Usability` (API 可用性)
> - `Changes in Model Availability` (模型可用性的变化)
> - `User Experience with Alternatives` (替代方案的用户体验)
> - `Service Stability Issues` (服务稳定性问题)


- **Perplexity Pro 限制下调**：用户报告称 Perplexity Pro 计划的每日限制在短时间内从 **600 次更改为 450 次**，引起了订阅者的不满。
   - *一位用户表示失望*，因为没有收到这些更改的通知，强调了对平台沟通缺乏信任。
- **对 API 可用性的关注**：几位成员讨论了使用 Perplexity API 的潜在成本，指出对于非重度用户来说，按需付费（pay-as-you-go）可能更具成本效益。
   - 有人提到 API 提供基于网络搜索的回答，与其他模型相比具有优势。
- **关于 Poe 等替代方案的讨论**：用户比较了 Perplexity 与 **Poe** 等其他 AI 服务的体验，指出尽管最近存在疑虑，但 Perplexity 提供了更令人满意的用户体验。
   - 一位用户报告称，由于对前者的界面和限制感到不满，已从 Poe 转向 Perplexity。
- **模型可用性关注**：用户表示有兴趣将新的 **Gemini 1.5 Pro** 添加到 Perplexity 平台，因为竞争对手已经在更新其产品。
   - 用户热衷于紧跟最新模型，以在任务中保持竞争力。
- **服务稳定性和可靠性问题**：讨论揭示了对 Sonnet 和 Opus 等服务稳定性的担忧，特别是在最近影响用户访问的停机期间。
   - 一位成员指出，服务崩溃导致重要任务中断，强调了可靠性能的必要性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://status.perplexity.com/">Perplexity - Status</a>: Perplexity 状态</li><li><a href="https://status.anthropic.com/">Anthropic Status</a>: 无描述</li><li><a href="https://www.perplexity.ai/search/google-anti-trust-lawsuit-806TIg85QL65_n0vbFqLlg#0">Google anti trust lawsuit</a>: 2024 年 8 月 5 日，针对 Google 的反垄断案做出了一项里程碑式的裁决，标志着美国司法部的重大胜利……</li><li><a href="https://x.com/testingcatalog/status/1821298236910374975?s=46&t=JsxhFTRLBknd8RUv1f73bA">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: Perplexity 正越来越接近 UGC 👀 一个微小但重要的变化——现在你可以注意到 Discover feed 上的大量 Perplexity 页面不再仅仅由 Perplexity 策划……</li><li><a href="https://x.com/aravsrinivas/status/1821637031002566671?s=61">Tweet from Aravind Srinivas (@AravSrinivas)</a>: 浅色模式 引用 Aravind Srinivas (@AravSrinivas) 不要作恶 (Don't be evil)
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1270882647958360154)** (11 条消息🔥): 

> - `Quantum Entanglement in the Brain` (大脑中的量子纠缠)
> - `Google Antitrust Lawsuit` (Google 反垄断诉讼)
> - `Perplexity Pro Features` (Perplexity Pro 功能)
> - `Microsoft's Advertising Strategies` (Microsoft 的广告策略)
> - `Node.js Module Exports` (Node.js 模块导出)


- **大脑中的量子纠缠引发争论**：大脑中 **quantum entanglement** 的概念引发了诸如 **Orch-OR** 等理论的研究，声称它可能会影响认知。
   - 然而，许多人认为大脑 **warm, wet**（温暖、潮湿）的环境不利于维持量子状态。
- **Google 面临重大反垄断裁决**：2024 年 8 月 5 日，美国地方法院裁定 **Google** 在在线搜索市场维持了非法垄断，这标志着 **Department of Justice**（司法部）的胜利。
   - 裁决指出，“*Google 是一个垄断者，并为了维持其垄断地位而采取了垄断行为*”，判定其关键做法为非法。
- **Perplexity Pro 提供独特功能**：**Perplexity Pro** 通过**实时信息检索 (real-time information retrieval)** 和透明的来源引用增强了搜索体验，对学术研究非常有益。
   - 它允许用户上传并分析文档，并提供对各种 **AI models** 的访问，以进行定制化搜索。
- **Microsoft 挑战 Apple 的批评**：Microsoft 的“**I'm a PC**”活动反击了 Apple 的“**Get a Mac**”广告，宣传其产品的**多功能性 (versatility)**，以对抗被感知的劣势。
   - 该活动利用**文化评论 (cultural commentary)** 和大量的资金支持，重塑了公众对 Windows 的看法。
- **理解 Node.js 模块导出**：`module.exports` 在 **Node.js** 中对于导出函数和值至关重要，促进了跨文件的模块化编程。
   - 每个模块的 `exports` 对象允许开发人员封装代码，从而实现更好的可维护性和关注点分离。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/embed/Thm7aUsLPaU">YouTube</a>: 未找到描述</li><li><a href="https://www.perplexity.ai/search/google-anti-trust-lawsuit-806TIg85QL65_n0vbFqLlg#0">Google anti trust lawsuit</a>: 2024 年 8 月 5 日，针对 Google 的反垄断案发布了一项具有里程碑意义的裁决，标志着美国司法部的重大胜利...</li><li><a href="https://www.perplexity.ai/search/perplexity-produ-you-shi-yao-q-OTW4JySeRQy.Pbuz1DtF9w">perplexity pro都有什么其他炫酷的功能</a>: Perplexity Pro 是一项高级订阅服务，提供了一系列增强功能，使用户的搜索体验更加丰富和高效。以下是一些主要功能： Pro Searches：每天至少可进行 300 次高级搜索，每次搜索会消耗一个积分，积分在24小时后恢复。 强大的 AI 模型：用户可以选择使用不同的高级 AI 模型，如 GPT-4...</li><li><a href="https://www.perplexity.ai/page/quantum-entanglement-in-the-br-7rokEdmsR4uZQmYOlx5J.A">Quantum Entanglement in the Brain</a>: 大脑中量子纠缠的概念，即粒子以可能影响意识和认知的方式相互关联，引发了...</li><li><a href="https://www.perplexity.ai/search/is-naturland-s-tobotronc-the-l-GVann50ESpqyNuB4wT4qvw">Perplexity</a>: Perplexity 是一款免费的 AI 驱动的回答引擎，可为任何问题提供准确、可信且实时的答案。</li><li><a href="https://www.perplexity.ai/search/what-are-module-exports-gbtvCVuJToC16kRkJ649KQ">what are module.exports</a>: 在 Node.js 中，module.exports 是 CommonJS 模块系统的关键特性，它允许开发人员从一个模块导出函数、对象或值，以便...</li><li><a href="https://www.perplexity.ai/search/generate-an-image-of-a-cat-rid-VSs5RFLnRqytc_sBugsosA">Perplexity</a>: Perplexity 是一款免费的 AI 驱动的回答引擎，可为任何问题提供准确、可信且实时的答案。</li><li><a href="https://www.perplexity.ai/search/how-can-i-use-inline-padding-vSnbnuMzStiI1sZKGs4HhA?__cf_chl_rt_tk=Ssx1APDVq7zXMbPOHijPl8mFfEfk1JbgdUF6zBGNwT8-1723119550-0.0.1.1-8468">Perplexity</a>: Perplexity 是一款免费的 AI 驱动的回答引擎，可为任何问题提供准确、可信且实时的答案。</li><li><a href="https://www.perplexity.ai/search/debian-ChlYBL_WRAW.x6c1nybKHg">Debian</a>: Debian 是一个广受认可的 Linux 发行版，以其对自由和开源软件的承诺而闻名。它由 Ian Murdock 于 1993 年发起，并且...</li><li><a href="https://www.perplexity.ai/search/what-is-the-link-to-your-disco-GBkoe8paT5.QBOhdWhL3nQ">What is the link to your discord?</a>: 加入 Perplexity AI Discord 社区的链接是：[https://discord.com/invite/perplexity-ai](https://discord.com/invite/perplexity-ai)。</li><li><a href="https://www.perplexity.ai/search/is-microsoft-rubbish-at-market-SEzx0MhMSxWXzJZDaJNOhw">is Microsoft rubbish at marketing?</a>: 关于 Microsoft 的营销是否“很烂”的问题是非常主观的，取决于各种观点。然而，通过分析现有数据和...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1270889502541353084)** (17 条消息🔥): 

> - `Perplexity API outage` (Perplexity API 停机)
> - `Geo-based access issues` (基于地理位置的访问问题)
> - `Claude outage impact` (Claude 停机影响)
> - `Non-English language incoherence` (非英语语言的不连贯性)
> - `Google Maps URL issues` (Google Maps URL 问题)


- **Perplexity API 遭遇重大停机**：用户报告无法访问 **Perplexity API**，根据状态页显示，部分用户指出发生了**重大停机**。
   - 一位用户对停机是局部还是大范围表示担忧，其他几位用户也分享了他们无法访问 API 的经历。
- **基于地理位置的访问差异**：一些成员认为 API 停机可能与**地理位置**有关，一位来自美国中部的用户表示无法访问，而另一位用户则提到其可以正常工作。
   - 一位遇到问题的用户建议使用 **VPN 连接到欧洲**作为临时解决方案。
- **Claude 停机可能影响 API 功能**：一位用户推测，他们遇到的 API 问题实际上可能源于 **Claude 停机**，因为他们依赖它来处理结果。
   - 这表明服务之间的相互依赖关系可能会影响对 Perplexity API 的访问。
- **非英语语言处理的不连贯性**：一位成员强调了非英语语言生成的响应存在**不连贯**的问题，并举例说明一个法语 Prompt 导致了重复的结果。
   - 这引发了人们对模型在准确处理多种语言和 Prompt 方面有效性的担忧。
- **Google Maps URL 的挑战**：一位用户询问了在生成旅行路线的准确 **Google Maps URL** 时遇到的困难，并指出许多提供的 URL 都是错误的。
   - 这反映了将实时数据集成到应用程序中并确保获取结果准确性方面所面临的持续挑战。



**提到的链接**：<a href="https://docs.perplexity.ai/discuss">Discussions</a>：未找到描述

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1270823988381089883)** (144 条消息🔥🔥): 

> - `Stable Diffusion 使用`
> - `AI 硬件升级`
> - `换脸技术`
> - `结合 SAM 的工作流`
> - `Mac 网页版推荐` 


- **在 Python 项目中使用 Stable Diffusion**：讨论了如何利用 **Diffusers** 库在 Python 项目中实现 **Stable Diffusion**，以优化性能和 VRAM 占用。
   - 成员们强调了正确设置参数对于获得理想输出质量的重要性。
- **为 AI 任务升级旧电脑**：一位用户描述了其旧电脑配置带来的困扰，并寻求在无需全面更换的情况下升级组件的建议。
   - 建议包括使用 Fiverr 等服务获取组装帮助，以及考虑购买准系统预装电脑。
- **Intel CPU 上的换脸技术**：一位用户询问了专门兼容 **Intel CPU** 的换脸方法，并表示愿意为获得帮助付费。
   - 该咨询强调了对针对低配置硬件用户的实用解决方案的需求。
- **用于图像细节增强的 SAM 工作流**：成员们讨论了利用 **SAM** 检测器来增强图像细节处理能力，从而实现多功能的工作流。
   - 一位成员强调了对图像中除人物以外的各种元素（包括背景和结构）进行细节处理的可能性。
- **Mac 上 NSFW 生成的网页版推荐**：一位用户寻求关于 NSFW 内容生成的最佳网页端工具建议，特别要求兼容配备 16GB RAM 的 **MacBook Air** M2。
   - 讨论转向了模型复杂度对性能的影响，强调了基于硬件能力的本地安装速度。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/tori29umai/sketch2lineart">Sketch2lineart - tori29umai 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/stabilityai/stable-diffusion-3-medium/tree/main/text_encoders">stabilityai/stable-diffusion-3-medium (main 分支)</a>: 未找到描述</li><li><a href="https://github.com/vosen/ZLUDA?tab=readme-ov-file#warning-important-warning">GitHub - vosen/ZLUDA: 在 ??? GPU 上运行 CUDA</a>: 在 ??? GPU 上运行 CUDA。通过在 GitHub 上创建账户为 vosen/ZLUDA 的开发做出贡献。</li><li><a href="https://github.com/TraceMachina/nativelink">GitHub - TraceMachina/nativelink: NativeLink 是一个开源的高性能构建缓存和远程执行服务器，兼容 Bazel、Buck2、Reclient 和其他 RBE 兼容的构建系统。它提供了大幅提升的构建速度、减少了测试的不稳定性，并支持专用硬件。</a>: NativeLink 是一个开源的高性能构建缓存和远程执行服务器，兼容 Bazel、Buck2、Reclient 和其他 RBE 兼容的构建系统。它提供了大幅提升的构建速度...
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1270819394943062177)** (107 条消息🔥🔥): 

> - `NVIDIA 显卡性能`
> - `CPU 占用率困惑`
> - `使用 GPU 进行模型推理`
> - `LM Studio 功能`
> - `Tauri vs Electron` 


- **NVIDIA 显卡不受当前问题影响**：已确认当前的性能问题仅影响 CPU，正如一位用户所指出的，成员们对自己的 NVIDIA 显卡表示放心。
   - 成员们分享了关于 CPU 和 GPU 设置的个人偏好，强调了 CPU 驱动工作负载的优势。
- **关于 CPU 占用率报告的困惑**：讨论中提到了 CPU 占用率显示超过 100% 的情况，并解释说某些应用程序是基于核心数量报告总占用率的。
   - 成员们注意到不同操作系统在报告方式上的差异，强调缺乏统一标准会导致误解。
- **使用双 GPU 提升模型推理能力**：确认 LM Studio 支持双 GPU，但模型推理速度与单 GPU 相同，不过可以加载更稠密的模型。
   - 讨论涉及硬件升级建议，以增加 tokens per second 从而获得更好的性能。
- **LM Studio 的功能与局限性**：成员们澄清 LM Studio 主要在本地运行，没有用于直接在线访问的 REST API，但它确实为服务器选项卡的使用提供了 REST API。
   - 强调了使用第三方应用程序或 UI 进行 GUI 交互的重要性。
- **Tauri 框架 vs. Electron**：关于 Tauri 相对于 Electron 的优势展开了辩论，个人经验表明 Tauri 的开发过程更加精简。
   - 成员们表达了对 Electron 社区响应速度的不满，并分享了使用 Tauri 的积极体验。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/vilarin/Llama-3.1-8B-Instruct">Meta-Llama3.1-8B - a Hugging Face Space by vilarin</a>: 未找到描述</li><li><a href="https://tauri.app">使用 Web 前端构建更小、更快、更安全的桌面应用程序 | Tauri Apps</a>: Tauri 是一个用于为所有主要桌面平台构建极小、极快二进制文件的框架。开发者可以集成任何可编译为 HTML, JS 和 CSS 的前端框架来构建他们的应用...</li><li><a href="https://tenor.com/view/aw-cry-sad-grandpa-gif-14766695">Aw Cry GIF - Aw Cry Sad - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/ggerganov/llama.cpp/commit/5442939fcc5e6ae41abf40612a95fd71377e487e">llama : support small Granite models (#7481) · ggerganov/llama.cpp@5442939</a>: * 为 Granite 模型添加可选的 MLP bias
 
 为 ARCH_LLAMA 添加可选的 MLP bias 以支持 Granite 模型。
 部分解决了 ggerganov/llama.cpp/issues/7116
 仍需一些额外更改以...
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1270894503204749413)** (35 条消息🔥): 

> - `4090 vs. 3080 性能对比`
> - `VRAM 与模型训练需求`
> - `AI 任务中的 Mac vs. Nvidia GPU`
> - `使用 Mac Mini 进行 AI 集群`
> - `Gemma 模型性能` 


- **用户质疑 4090 vs. 3080 的性能**：一位用户对 **4090** 相比 **3080** 的表现表示不满，理由是训练速度每轮（epoch）仅有 **20 ms** 的差异。
   - 其他人观察到，虽然 **4090** 在游戏和 **LLM** 任务中具有优势，但 **3080** 仍然能高效处理 8B 以下的模型。
- **VRAM 不足影响模型选择**：讨论显示 **2GB VRAM** 不足以运行大多数模型，在低 **VRAM** 模式下运行的效果很差。
   - 有人指出，较大的模型必须在 **VRAM** 和系统内存（**system RAM**）之间拆分，这会严重影响性能。
- **Mac vs. Nvidia GPU 在 AI 效率方面的对比**：虽然 **Mac** 提供了 **MLX framework** 等优势，但在纯 AI 任务中，其速度通常慢于高端的 **4090** 设备。
   - 一位用户表示可能会为了 **Mac** 退掉 **4090**，但论坛成员建议保留 **4090** 以获得更好的 AI 训练能力。
- **使用 Mac Mini 构建 AI 集群的可能性**：成员们讨论了在 **AI cluster** 中使用 **Mac Mini** 系统的可行性，并强调了其效率。
   - 这一观点受到了热烈响应，表明 **Mac Mini** 可能是处理 AI 任务的一个可行选择。
- **Gemma 模型在对比更大规模替代方案时表现出色**：建议尝试 **Gemma 2 27B** 模型，其性能优于 **Yi 1.5 34B**。
   - 强调了在资源有限的情况下，最大化利用现有模型的策略。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.macrumors.com/2023/10/31/apple-m3-pro-less-memory-bandwidth/">Apple M3 Pro Chip Has 25% Less Memory Bandwidth Than M1/M2 Pro</a>: 苹果最新的 M3 Pro 芯片在新款 14 英寸和 16 英寸 MacBook Pro 中，其内存带宽比 M1 Pro 和 M2 Pro 芯片减少了 25%...</li><li><a href="https://nanoreview.net/en/cpu-compare/apple-m3-pro-vs-apple-m2-ultra">Apple M3 Pro vs M2 Ultra: performance comparison</a>: 我们在游戏和基准测试中对比了 Apple M3 Pro (4.05 GHz) 与 M2 Ultra (3.5 GHz)。了解哪款 CPU 性能更佳。
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1271154665509097515)** (2 条消息): 

> - `GPT-4o System Card`
> - `面向免费用户的 DALL·E 3 图像生成` 


- **GPT-4o System Card 发布**：OpenAI 分享了 [GPT-4o System Card](https://openai.com/index/gpt-4o-system-card/)，详细介绍了安全评估以及为追踪 **frontier model risks**（前沿模型风险）而采取的措施。
   - 该报告重点介绍了对 **GPT-4o** 音频能力的评估以及针对有害内容的防护措施（guardrails），确保其仅以预设声音生成音频。
- **DALL·E 3 现已面向免费用户开放**：ChatGPT 免费用户现在每天可以使用 **DALL·E 3** 创建最多两张图像，用于幻灯片或个性化卡片等各种用途。
   - 用户只需让 ChatGPT 生成所需的图像，从而增强了内容创作和个性化选项。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1270823230156050534)** (55 条消息🔥🔥): 

> - `Website Access Issues` (网站访问问题)
> - `Quota and Limits` (配额与限制)
> - `Python SDK Issues` (Python SDK 问题)
> - `Patent and Open Source Concerns` (专利与开源顾虑)
> - `Model Performance Queries` (模型性能查询)


- **用户面临网站访问问题**：多名成员报告在访问主站时遇到困难，连接性变差，相关错误已被标记以待调查。
   - 尽管尝试重新加载或使用无痕模式访问，问题依然存在，其他面临类似问题的用户也证实了这一点。
- **对消息限制的困惑**：讨论围绕在平台上发送消息时施加的限制展开，一些用户对过早触发限制表示沮丧。
   - 参与者交流了各自的经历，频繁评论他们在遇到 **4o** 限制时的消息情况以及触发错误时的差异。
- **OpenAI Python SDK 使用困境**：多位用户表示在使用最新的 OpenAI Python SDK 复制示例代码后难以复现结果，尤其是不同版本导致了不同的输出。
   - 一名成员指出所使用的 Python 版本存在差异，并承认这是问题的根源。
- **技术专利与开源的博弈**：一名用户就为使用 ChatGPT 开发的技术申请专利寻求建议，表达了希望将其保留在公共领域的愿望，尽管他了解专利申请的高昂成本。
   - 他们既想保护创意不被剥削，又想确保开放获取的意图，引发了关于在 AI 领域申请专利的可行性和实际性的讨论。
- **关于模型性能的查询**：出现了关于 **Mistral NeMo** 模型性能的问题，特别是关于其在 M1 芯片和 16GB RAM 环境下的处理能力。
   - 此外，一名成员评论了访问 **Claude** 的问题，确认其正经历宕机。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1270819300143272116)** (25 条消息🔥): 

> - `API Key Authentication Issues` (API Key 身份验证问题)
> - `GPT-4o Features for Non-Plus Users` (面向非 Plus 用户的 GPT-4o 功能)
> - `GPT-3.5 Turbo Quota Limits` (GPT-3.5 Turbo 配额限制)
> - `Custom GPT Updates Pending` (自定义 GPT 更新待处理)
> - `Using CSV Files with Langchain` (在 Langchain 中使用 CSV 文件) 


- **API Key 身份验证问题**：一名用户表示在请求中传递 API Key 时遇到困难，尽管在 UI 中进行了设置，仍导致“Missing API Key”错误。
   - 另一名成员澄清说 OpenAI 会自动处理 Key 验证，但在某些情况下用户可能需要设置 'X-Api-Key'。
- **面向非 Plus 用户的 GPT-4o 功能**：一名用户询问 GPT-4o 的功能（如 Vision 和高级语音模式）是否会向非 Plus 用户开放。
   - 另一名成员指出，提供此类功能相关的成本可能会阻碍大规模推广。
- **GPT-3.5 Turbo 配额限制**：一名用户询问在使用免费账户调用 GPT-3.5-turbo 时遇到“quota”错误的问题。
   - 一名成员报告称 GPT-3.5-turbo 已不再包含在免费层级中，导致免费用户出现配额错误。
- **自定义 GPT 更新待处理**：一名用户寻求关于自定义 GPT 显示“Updates pending”消息的澄清，想知道对指令的更改是否已正确保存。
   - 另一名成员建议这是一个临时 Bug，刷新页面通常可以消除该消息，并确保更新已生效。
- **在 Langchain 中使用 CSV 文件**：一名用户询问关于在 Langchain 中将 CSV 文件作为 RAG (Retrieval-Augmented Generation) 文档与 OpenAI 结合使用的资源。
   - 该话题仍处于讨论阶段，聊天中尚未提供具体的资源。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1270853582912098477)** (11 messages🔥): 

> - `Self-Discover prompting strategy`
> - `Reverse prompting techniques`
> - `Custom GPTs development`
> - `Groq API for summary notes` 


- **Self-Discover：一种新的提示工程策略**：一位成员强调 **Self-Discover** 是一种强大的新提示策略，超越了思维链 (CoT) 方法。
   - 该技术自 1 月以来一直被探索，展示了在提示框架内作为工具使用的潜力。
- **极具前景的反向提示技术**：成员们讨论了使用**反向提示 (reverse prompting)** 为特定用例开发更有效提示的潜力。
   - 这些技术允许在自定义 GPT 开发过程中，从抽象模板转向更精确的输出。
- **使用 Builder 创建自定义 GPTs**：一位成员分享了他们使用 Builder 而非配置面板构建自定义 GPTs 的方法，并利用了动态评论系统。
   - 通过在整个过程中移动评论部分，他们有效地管理了源自其模板的自定义指令。
- **寻求 Groq API 封装器的帮助**：一位成员正尝试使用 **Groq API** 设计一个 GPT 封装器，以生成一致的讲义摘要和抽认卡。
   - 另一位成员指出，问题在于 API 的编程而非提示工程，并引导他们参考相关资源。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1270853582912098477)** (11 messages🔥): 

> - `Prompt Engineering Strategies`
> - `Self-Discover Prompting`
> - `Custom GPT Development`
> - `Groq API Integration` 


- **探索提示工程创新**：一位成员强调了提示工程技术的潜力，特别关注将 *prompting* 和 *reverse prompting* 模型作为框架内的工具。
   - 他们指出这些想法仍处于起步阶段，但可能会演变成稳健的策略。
- **Self-Discover 提示法受到关注**：另一位成员分享了关于 *Self-Discover* 策略的见解，断言其力量和有效性超越了传统的思维链 (CoT) 方法。
   - 他们强调了其在制作能产生更好输出的定制提示方面的适用性。
- **使用动态指令开发自定义 GPTs**：一位用户详细介绍了他们创建自定义 GPTs 的方法，使用包含滚动评论部分的 Builder 模板来引导过程。
   - 这种方法允许在完成模板中的自定义指令时进行无缝的指令更新。
- **使用 Groq API 处理讲义的挑战**：一位成员请求在提示工程方面提供协助，以使用 Groq API 构建一个能生成可靠摘要和抽认卡的 GPT 封装器。
   - 另一位成员澄清说，问题可能更多地与 API 编程有关，而非提示工程，并提供了一个寻求进一步帮助的链接。


  

---



### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1271108152330551367)** (2 messages): 

> - `MindSearch AI`
> - `Information Seeking Systems`
> - `Audio Research Communities` 


- **MindSearch AI 桥接 LLMs 和搜索引擎**：论文 *MindSearch: Mimicking Human Minds Elicits Deep AI Search* 介绍了一种 Agentic AI，旨在通过 WebPlanner 和 WebSearcher 的双系统方法模拟人类解决问题的过程，从而改进信息检索。
   - 这种结构使 MindSearch 能够高效处理**复杂问题**，并优于现有的搜索系统，展示了其在智能信息检索领域的潜力。
- **WebPlanner 和 WebSearcher 协同工作**：MindSearch 的特色在于 **WebPlanner**（将复杂查询分解为可管理的子问题）和 **WebSearcher**（通过采用分层检索策略寻找答案，以实现有效的信息综合）。
   - 通过利用这些角色，MindSearch 比传统的 LLMs 能更熟练地应对**多步查询**。
- **社区寻求专注于音频的讨论**：一位成员询问是否有类似于 Nous Research Discord 的音频相关主题社区，并提到旧的 Harmonai Discord 已变得不活跃。
   - 他们表达了希望有一个空间来参与关于具有挑战性的研究问题的深入讨论。



**提到的链接**：<a href="https://x.com/intuitmachine/status/1821498263532429571?s=46">来自 Carlos E. Perez (@IntuitMachine) 的推文</a>：1/n 解锁网络知识：一个能读懂链接背后深意的 Agentic AI。在我们这个信息过载的时代，寻找正确答案往往感觉像大海捞针……

  

---

### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1270834599953567826)** (14 条消息🔥): 

> - `Machine Learning Discord Channels`
> - `Nous Artist Compliments`
> - `Commission Work`
> - `Reddit Recommendations` 


- **Machine Learning Discord 频道推荐**：一位成员分享了一个 [Reddit 帖子](https://www.reddit.com/r/nousresearch/comments/1elmrjr/most_helpful_ai_discordscommunities/?share_id=L2tAJZE66RY4dOPfIbiMw)，其中包含各种 AI Discord 链接，有助于探索社区。
   - 一些重点推荐的频道包括 **Replete-AI**、**Unsloth** 和 **Nous-Research**，提供了 AI 和 ML 领域的多种资源。
- **Nous 艺术家获得赞赏**：**Hy3na_xyz** 称赞了 **Nous 艺术家**，表示他们的审美“非常到位 (on point)”，展示了社区的认可。
   - **Kainan_e** 幽默地指出这是一个称赞，为对话增添了轻松的氛围。
- **关于委托工作的咨询**：**Hy3na_xyz** 询问 **Nous 艺术家** **john0galt** 是否接受工作委托，对方回复称这种情况很少见，且必须是值得做的项目。
   - 这表明了对潜在合作的兴趣，同时也突显了该艺术家委托工作的稀缺性。



**提及的链接**：<a href="https://www.reddit.com/r/nousresearch/comments/1elmrjr/most_helpful_ai_discordscommunities/?share_id=L2tAJZE66RY4dOPfIbiMw">Reddit - 深入探索一切</a>：未找到描述

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1270896144452686040)** (1 条消息): 

> - `Tavus Phoenix`
> - `Real-Time Video Cloning`
> - `Neural Radiance Fields`
> - `Video Generation API` 


- **Tavus 推出 Phoenix 模型**：Tavus 新推出的 **Phoenix** 模型提供极其逼真的写实人脸视频，具有精确的**自然面部动作**和与输入同步的表情。
   - 开发团队利用先进技术绕过了传统方法，从而增强了视频输出的真实感。
- **在 8xH100 服务器上的实时处理能力**：据报道，Tavus 可以在 **8xH100 服务器**上**实时**运行其视频克隆，展示了令人印象深刻的计算效率。
   - 这种能力允许即时生成视频，增强了用户使用该技术时的交互性和参与感。
- **通过 API 访问 Phoenix 模型**：鼓励开发者通过 Tavus 的 **Replica API** 访问 **Phoenix** 模型，该 API 支持高度定制和真实感。
   - *该 API 支持多种应用*，使其成为内容创作者和开发者的多功能工具。
- **Neural Radiance Fields 重新定义视频创作**：使用 **Neural Radiance Fields (NeRFs)** 的新颖方法允许 Tavus 构建动态的三维**面部场景**。
   - 这种方法显著提高了生成视频的质量和真实感，为该领域树立了新标准。



**提及的链接**：<a href="https://www.tavus.io/developer">Tavus | Developers</a>：Tavus 在数字复制、唇形同步、配音、文本转视频方面构建了先进的 AI 模型，开发者可通过 API 访问。

  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1270869426954178570)** (84 条消息🔥🔥): 

> - `Upside Down Poem Generation` (倒置诗歌生成)
> - `Model Comparison` (模型对比)
> - `API vs Chat Interface` (API 与聊天界面)
> - `Training Data for Tokenization` (用于 Tokenization 的训练数据)
> - `Server Overloads` (服务器过载)


- **模型在处理倒置文本时表现挣扎**：包括 Mistral、ChatGPT 和 LLama 405b 在内的多个模型无法生成连贯的倒置文本（Upside Down Text），而是产生随机序列。
   - 相比之下，Claude Opus 和 Sonnet 3.5 能够持续生成准确且连贯的倒置消息。
- **Claude 模型的预期性能**：Claude 模型，特别是 Sonnet 3.5，似乎具有更优越的能力，包括为倒置诗歌进行反向文本规划。
   - 它们还能一致地识别自己生成的倒置文本并准确重写，而不像其他模型经常生成错误的解释。
- **训练数据与 Tokenization 讨论**：用户讨论了需要多样化且有趣的训练数据，以提高在生成倒置文本等任务上的表现。
   - 有人推测现有的 Tokenization 问题是否会影响模型输出，特别是在处理较长行时。
- **关于 API 使用和服务器可靠性的评论**：用户指出 Claude API 偶尔会出现过载消息，影响了高需求期间的使用和性能。
   - 对于是被封禁还是服务器问题导致的访问困难，目前尚不确定。
- **对 System Prompts 和 antthinking 的思考**：讨论了 System Prompts 和 antthinking 功能在 Claude 写作过程中的作用。
   - 这一功能似乎对倒置诗歌的生成没有显著影响，因为用户报告了来自不同界面的不同反应。


---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1270988480272404491)** (91 条消息🔥🔥): 

> - `与 LM Harness 的数据集兼容性`
> - `CBRN 风险与模型响应`
> - `过滤预训练数据`
> - `知识移除对模型性能的影响`
> - `新闻界在 AI 生成内容方面面临的挑战` 


- **为 LM Harness 制作数据集**：一位成员询问 LM Harness 的数据集是否必须为 .jsonl 格式，并想了解所需的字典键（dictionary keys）。
   - 另一位成员建议参考 [YAML 文件](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/gsm8k) 以获取更结构化的指导，并强调了键设计的灵活性。
- **探讨 AI 模型的 CBRN 风险**：成员们辩论了是否可以设计一个既能提供化学建议又不构成 CBRN 风险的模型，并担心过滤知识可能会阻碍科学能力。
   - 讨论强调，即使移除了相关知识，聪明的用户仍可能推导出有害信息，从而对这种过滤方法的有效性提出了质疑。
- **过滤预训练数据的影响**：参与者认为，从模型中移除“坏”信息可能会导致整体理解能力和对齐（alignment）有效性的下降。
   - 有人指出，训练中缺乏负面示例可能会损害模型识别和避免有害活动的能力，从而可能导致能力的退化。
- **新闻界对 AI 的看法**：成员们对记者描述 AI 模型的方式表示沮丧，认为他们往往专注于有关潜在风险的耸人听闻的故事，而缺乏足够的背景信息。
   - 这种看法加剧了人们对围绕 AI 输出及其误导性陈述的安全讨论的普遍担忧。
- **平衡 AI 中的知识与风险**：一位成员描述了在确保 AI 模型提供准确科学指导的同时减轻滥用风险的复杂性，强调了知识可用性与安全性之间的紧张关系。
   - 对话强调了在支持科学讨论与防止传播有害方法之间的微妙界限，表明简单的过滤可能不足以进行有效的风险管理。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/gsm8k">lm-evaluation-harness/lm_eval/tasks/gsm8k at main · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/wikitext/wikitext.yaml#L9)">lm-evaluation-harness/lm_eval/tasks/wikitext/wikitext.yaml at main · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/wikitext/preprocess_wikitext.py#L5)">lm-evaluation-harness/lm_eval/tasks/wikitext/preprocess_wikitext.py at main · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1270821750535356477)** (6 条消息): 

> - `使用 RNN 进行感知查询`
> - `开源的 Process Based Reward Models`
> - `Pythia Checkpoints 与 WandB 日志`
> - `同步模型课程` 


- **使用 RNN 更新感知查询**：一位用户建议实现 **RNNs** 来随时间更新感知查询（perception queries），这可能会提高效率。
   - 这种方法可以增强模型在处理信息时的适应性。
- **寻找开源奖励模型**：有人询问是否有合适的开源 **Process Based Reward Models** 来验证数学任务。
   - 这凸显了在数学解题中对有效验证工具的需求。
- **匹配 Pythia Checkpoints 与 WandB 日志**：一位用户询问是否有简便的方法将 **Pythia checkpoints** 与其对应的 **WandB logs** 关联起来。
   - 目前，似乎还没有直接的解决方案来匹配这些资源。
- **同步模型的课程（Curricula）**：有人询问是否可以同步两个不同模型的课程，以保持相同的 minibatches。
   - 另一位用户建议记录训练数据的顺序和分组，以便在第二个模型中实现这一点。


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/)** (1 条消息): 

brain4brain: 噢噢噢我明白了，谢谢提供信息
  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1270824681443954772)** (3 条消息): 

> - `Model Parallelism`
> - `GPU Data Splitting` 


- **理解 GPU 系统中的 Model Parallelism**：已确认 Model Parallelism 涉及将模型拆分到多个 GPU 上，然后在剩余的 GPU 上创建这些分段层的副本。
   - 这种方法可以有效利用可用的硬件资源，从而提升模型性能。
- **GPU 间的数据分布**：在拥有 8 个 GPU 的情况下，运行 4 个独立的进程意味着 GPU 0 和 1 将处理 **1/4** 的数据，且模型副本在它们之间拆分；而 GPU 2 和 3 则以类似方式管理另外 **四分之一** 的数据。
   - 这种系统化的切分确保了每组 GPU 都能有效地处理其对应部分的数据，优化了资源利用。


  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1270841961976168571)** (49 messages🔥): 

> - `Hugging Face 收购 XetHub`
> - `Qwen2-Math 模型发布`
> - `AI 基础设施独角兽`
> - `OpenAI 降价`
> - `文本生成图像排行榜` 


- **Hugging Face 收购 XetHub**: Hugging Face 宣布收购 [XetHub](https://www.forbes.com/sites/richardnieva/2024/08/08/hugging-face-xethub-acquisition/)，以扩展大模型的协作并增强其基础设施。
   - CEO Clem Delangue 强调，此次收购旨在大幅改进 AI 开发流程，重点关注更大规模的数据集。
- **Qwen2-Math 模型表现优于竞争对手**: 阿里巴巴发布了全新的 [Qwen2-Math 模型系列](https://qwenlm.github.io/blog/qwen2-math)，其旗舰模型在数学任务上的表现优于 GPT-4o 和 Claude 3.5。
   - 该模型系列代表了在专业语言处理性能方面的重大进步。
- **洞察 AI 基础设施独角兽**: 一个新系列讨论了 Hugging Face 和 Databricks 等各种基础设施构建者，它们支持着不断增长的生成式 AI 市场。
   - Hugging Face 的巨额融资旨在加强其在开源机遇中类似于 GitHub 的地位。
- **OpenAI 宣布降价**: 最近的讨论显示 OpenAI 为其 GPT-4o 模型提供了 **70% 的降价**，引起了广泛关注。
   - 此举可能会重塑行业内关于 AI 模型的定价策略。
- **文本生成图像领域的新领导者**: 来自 bfl_ml 的 FLUX.1 登顶文本生成图像排行榜（Text to Image Leaderboard），将长期占据榜首的 Midjourney 挤下。
   - 这一转变表明了 AI 图像生成领域的竞争态势，突显了开源权重（open-weight）模型的兴起。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.forbes.com/sites/richardnieva/2024/08/08/hugging-face-xethub-acquisition/">AI 独角兽 Hugging Face 收购了一家初创公司，最终将托管数亿个模型</a>: 以未披露的金额收购，Hugging Face 认为此次收购将帮助开发者构建与 OpenAI 和 Google 齐名的大规模模型。 </li><li><a href="https://x.com/swishfever/status/1821284583171887236?s=46">fishy business (@swishfever) 的推文</a>: 新 Anthropic 模型的截止日期（极有可能）是 2024 年 8 月 31 日</li><li><a href="https://x.com/artificialanlys/status/1821569675370930395?s=46">Artificial Analysis (@ArtificialAnlys) 的推文</a>: 祝贺 @bfl_ml 和 @midjourney 席卷 Artificial Analysis 文本生成图像排行榜！欢迎来到新前沿：🥇来自 @bfl_ml 的 FLUX.1 [pro] 🥈来自 @mid... 的 Midjourney v6.1</li><li><a href="https://huggingface.co/blog/introducing-private-hub">介绍 Private Hub：一种构建机器学习的新方式</a>: 未找到描述</li><li><a href="https://www.turingpost.com/p/databricks">Databricks：企业领域生成式 AI 的未来</a>: 探索 Databricks 不寻常的历史、其对企业级生成式 AI 领域的贡献，以及该公司的战略和对 AI 行业的愿景。</li><li><a href="https://x.com/minimaxir/status/1821597473103905025?s=46">Max Woolf (@minimaxir) 的推文</a>: OpenAI 刚刚泄露了《黑镜》下一季的情节。https://openai.com/index/gpt-4o-system-card/</li><li><a href="https://x.com/Alibaba_Qwen/status/1821553401744015816">Qwen (@Alibaba_Qwen) 的推文</a>: 今天我们发布了一个针对数学专用语言模型的新系列 Qwen2-Math，它基于 Qwen2。旗舰模型 Qwen2-Math-72B-Instruct 的表现优于包括 GPT-4 在内的专有模型...</li><li><a href="https://huggingface.co/blog/the-partnership-amazon-sagemaker-and-hugging-face">合作伙伴关系：Amazon SageMaker 与 Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/julien_c/status/1821540661973160339?s=46">Julien Chaumond (@julien_c) 的推文</a>: 我非常激动地宣布我们已经收购了 XetHub！🎊 @xetdata 团队开发的技术使 Git 能够扩展到 TB 级规模的仓库。在底层，他们一直是...</li><li><a href="https://www.turingpost.com/p/huggingfacechronicle">AI 民主化：Hugging Face 让机器学习触手可及的理念</a>: Hugging Face 从聊天机器人构建者到机器学习传道者历程的内幕故事
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1270822224139517983)** (23 messages🔥): 

> - `Tokens vs Epochs`
> - `GPT-4 Token Count`
> - `Anthropic CI Debates`
> - `RLHF Critique`
> - `Recruitment Challenges` 


- **关于 Tokens 和 Epochs 的传闻混淆**：有推测认为，在讨论中**传闻可能搞混了**对 **Tokens vs Epochs** 的理解。
   - 几位成员注意到在线论坛中经常出现这种混淆，但未提供具体细节。
- **GPT-4 Token 数量确认**：据报道 **GPT-4** 使用了 **10 万亿 (10 trillion) Tokens**，这一数字得到了聊天中多位成员的回应。
   - 成员们达成共识，认为这个数字看起来是准确的，尽管有人评论说 GPT-4 现在已被视为**过时技术 (ancient technology)**。
- **Anthropic 对辩论训练的使用**：讨论涉及 **Anthropic** 的辩论训练如何贡献于其 **Claude 3** 模型，该模型可能已经使用了 **Agent** 辩论。
   - 有人提到关于此话题的更新已发布在 alignment forum 上，引起了成员们的极大兴趣。
- **RLHF 的批判性概述**：一位成员引用了 **Karpathy** 的话，指出 **RLHF 仅仅勉强算是 RL**，因为它严重依赖于一个模仿人类偏好的奖励模型 (reward model)。
   - 该评论强调了 **RLHF** 的潜在陷阱，断言它与 **AlphaGo** 等系统中看到的真实强化学习技术有本质区别。
- **招聘中的挑战**：招聘方面存在紧迫感，一位成员表示需要具有实际训练 **LLMs** 经验的人才。
   - 在此期间，出现了一些幽默的评论，调侃了 **Karpathy** 的视频及其在招聘中的作用，强调了轻松的社区氛围。



**Link mentioned**: <a href="https://x.com/karpathy/status/1821277264996352246?s=46">Tweet from Andrej Karpathy (@karpathy)</a>: # RLHF is just barely RL  Reinforcement Learning from Human Feedback (RLHF) is the third (and last) major stage of training an LLM, after pretraining and supervised finetuning (SFT). My rant on RLHF i...

  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1270891264606343332)** (6 messages): 

> - `Gary Marcus Predictions`
> - `Audience Capture`
> - `Contrarian Perspectives on AI` 


- **Gary Marcus 预测 AI 泡沫即将崩溃**：[Gary Marcus](https://x.com/GaryMarcus/status/1819525054537126075) 对他预测 AI 泡沫将在 2025 年崩溃表示遗憾，声称：*“将是几天或几周后，而不是几个月。”*
   - 这一预测引发了关于他公信力的讨论，观点指出此类言论似乎缺乏严肃性。
- **Nathan 对 Gary Marcus 的看法**：Nathan 将 Gary Marcus 描述为一个受无关见解和古怪观点历史驱动的 *bozo*，暗示其评论中存在一定程度的受众绑架 (**audience capture**)。
   - 他表达了担忧，指出：*“我很难接受那些以技术为职业却长期仇视技术的人。”*
- **对 Marcus 见解的复杂看法**：另一位成员承认，虽然 Gary Marcus 在 **LLMs** 方面提供了一些合理的批评，但他也有唱反调的倾向。
   - 他们注意到，他真实的观点往往被一种想要证明自己正确的欲望所掩盖，总是在寻找机会宣称：*“我早就告诉过你了。”*



**Link mentioned**: <a href="https://x.com/GaryMarcus/status/1819525054537126075">Tweet from Gary Marcus (@GaryMarcus)</a>: I just wrote a great piece for WIRED predicting that the AI bubble will in collapse in 2025, and now I wish I hadn’t.  Clearly, I got the year wrong. It’s going to be days or weeks from now, not month...

  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/)** (1 messages): 

chygao: https://youtu.be/6QWuJRvMtxg?si=SYXsRvYbfcdtYLC2
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/)** (1 messages): 

SnailBot News: <@&1216534966205284433>
  

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1270987563343024200)** (75 条消息🔥🔥): 

> - `LangChain 在 AWS Lambda 中的问题`
> - `在 LLM 应用中限制聊天历史`
> - `在 Slack RAG 应用中管理特定用户的历史记录`
> - `LangChain 与其他框架的对比`
> - `不同 LLM 面临的挑战` 


- **排除 AWS Lambda 中 LangChain 的故障**：一位用户报告了在 AWS Lambda 函数中使用 **LangChain imports** 时遇到的困难，特别是尝试导入 LangChain 模块时出现的 **pydantic** 相关错误。
   - 他们确认使用了 Python **3.12** 运行时并正确设置了 lambda layer，这引发了关于潜在版本冲突的讨论。
- **LLM 的聊天历史管理**：讨论了用户在 LLM 应用中限制聊天历史以提高性能并维持用户上下文的实现方式。
   - 会议指出，需要一个 **custom function** 来标准化跨用户保留在聊天历史中的消息数量。
- **LangChain 与其他框架**：一位用户表示，由于 LLM 功能的差异，使用 LangChain 从 **OpenAI** 切换到 **Anthropic** 需要重写代码，这让他感到沮丧。
   - 大家一致认为，虽然 LangChain 提供了一些抽象，但 LLM 仍需要根据其特性进行特定的调整。
- **LLM 停机带来的挑战**：另一位用户强调 **Anthropic** 的 **Claude 3.5** 遇到了内部服务器错误，对 AI 系统在生产环境中的可靠性表示担忧。
   - 这引发了关于 AI 系统整体就绪程度以及 LangChain 是否是满足其需求的正确选择的疑问。
- **寻求 B2B 销售指导**：一位用户询问如何获取源代码并寻找导师，以进一步学习 **B2B sales**。
   - 这表明了对在商业领域进行销售导航的实践指导和资源的渴望。



**提到的链接**：<a href="https://python.langchain.com/v0.2/docs/how_to/trim_messages/">How to trim messages | 🦜️🔗 LangChain</a>：本指南假设你已熟悉以下概念：

  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/)** (1 条消息): 

_johnny1984: 愚蠢的垃圾邮件机器人（spambot）在我的说唱歌手 AI 面前毫无胜算：
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1270845417889861702)** (61 条消息🔥🔥): 

> - `GPT-4o 能力`
> - `Gemini 1.5 Flash 更新`
> - `DALL·E 3 免费访问`
> - `Mistral Agents`
> - `AI 学术论文` 


- **GPT-4o 提供广泛的输入和输出能力**：GPT-4o 模型可以处理文本、音频、图像和视频的任何组合，显著增强了其通用性，且响应速度与人类相似。
   - 它在跨语言性能上也有所提升，且 API 使用成本比之前的模型便宜 50%。
- **Gemini 1.5 Flash 为开发者带来重大更新**：GoogleAI 将 Gemini 1.5 Flash 的价格降低了约 70%，使其对开发者更具吸引力。
   - 此外，AI Studio 现已向所有工作空间客户开放，允许进行更大规模的实验以及与新语言的集成。
- **OpenAI 向免费用户推出 DALL·E 3**：ChatGPT 免费用户现在每天可以使用 DALL·E 3 创建最多两张图像，提高了普通用户的可访问性。
   - 这一功能使用户能够轻松创建个性化内容，尽管对其更广泛的使用仍存在一些怀疑。
- **Mistral Agents 扩展功能**：Mistral Agents 现在能够利用 Python 并集成到各种工作流中，展示了其多功能性。
   - 用户对允许 API 消耗以及在实际场景中应用这些功能表示了兴趣。
- **对学术 AI 研究论文的见解**：围绕近期学术论文（如与 llama3 模型相关的论文）的讨论表明，人们对多模态模型的兴趣日益浓厚。
   - 作者们正在公开分享他们的研究，提供代码、数据和权重（weights）的访问权限，以促进社区协作。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://buildship.com)">未找到标题</a>: 未找到描述</li><li><a href="https://news.ycombinator.com/item?id=41184559">未找到标题</a>: 未找到描述</li><li><a href="https://news.ycombinator.com/item?id=41188647">未找到标题</a>: 未找到描述</li><li><a href="https://docs.mistral.ai/capabilities/agents/">Agents | Mistral AI 大语言模型</a>: 什么是 AI agents？</li><li><a href="https://homebrew.ltd/blog/can-llama-3-listen">Homebrew</a>: 构建运行在节能硬件上的增强人类能力的 AI。</li><li><a href="https://x.com/OfficialLoganK/status/1821601298195878323">Logan Kilpatrick (@OfficialLoganK) 的推文</a>: @GoogleAI 开发者们的好消息：- Gemini 1.5 Flash 价格现已降低约 70% ($0.075 / 1M) - Gemini 1.5 Flash tuning 向所有人开放 - API 新增支持 100 多种语言 - AI Studio i...</li><li><a href="https://x.com/karpathy/status/1821286855310242020">Andrej Karpathy (@karpathy) 的推文</a>: 公平地说，我无法通过快速的 Google 搜索找到那样的图片。我本想花点时间做一张，但我担心这会有以另一种方式产生误导的风险。在 Go 中...</li><li><a href="https://x.com/karpathy/status/1821277264996352246?s=46&t=6FDPaNxZcbSsELal6Sv7U">Andrej Karpathy (@karpathy) 的推文</a>: # RLHF 几乎算不上 RL。来自人类反馈的强化学习 (RLHF) 是训练 LLM 的第三个（也是最后一个）主要阶段，排在预训练和有监督微调 (SFT) 之后。我对 RLHF 的吐槽是...</li><li><a href="https://x.com/vboykis/status/1821527144922566745">vicki (@vboykis) 的推文</a>: 现在的护城河是模型，一年后的护城河将是记忆（系统对你及其查询的记忆程度，以及选择正确 prompt 和模型的能力，也就是说我们很快又要回到 recsys 了....</li><li><a href="https://x.com/mckaywrigley/status/1821307469114769903?s=46">Mckay Wrigley (@mckaywrigley) 的推文</a>: 这是一个关于 LLMs 高级 prompting 技术的 17 分钟深度探讨。在一个真实的多步骤 AI 工作流中进行了完整演示。观看以获取完整解析。</li><li><a href="https://x.com/OpenAI/status/1821644904843636871">OpenAI (@OpenAI) 的推文</a>: 我们正在向 ChatGPT 免费用户推出每天使用 DALL·E 3 创建最多两张图片的功能。只需让 ChatGPT 为幻灯片创建图片、为朋友定制卡片，或展示...</li><li><a href="https://x.com/clementdelangue/status/1821559961555554469?s=46">clem 🤗 (@ClementDelangue) 的推文</a>: 这是真正的 🍓 - 欢迎来到 @xetdata - 我们才刚刚开始！</li><li><a href="https://x.com/openai/status/1821595015279472736?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">OpenAI (@OpenAI) 的推文</a>: 我们正在分享 GPT-4o System Card，这是一份端到端的安全评估，概述了我们在跟踪和解决安全挑战方面所做的工作，包括根据我们的 Preparedness 框架应对前沿模型风险...</li><li><a href="https://qwenlm.github.io/blog/qwen2-math/">Qwen2-Math 介绍</a>: GITHUB HUGGING FACE MODELSCOPE DISCORD 🚨 该模型主要支持英文。我们很快将发布双语（中英文）数学模型。简介 在过去的一年里，我们投入了大量...</li><li><a href="https://x.com/karpathy/status/1821277264996352246?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Andrej Karpathy (@karpathy) 的推文</a>: # RLHF 几乎算不上 RL。来自人类反馈的强化学习 (RLHF) 是训练 LLM 的第三个（也是最后一个）主要阶段，排在预训练和有监督微调 (SFT) 之后。我对 RLHF 的吐槽是...</li><li><a href="https://x.com/_clashluke/status/1820810798693818761">Lucas Nestler (@_clashluke) 的推文</a>: http://x.com/i/article/1820791134500642816</li><li><a href="https://x.com/karpathy/status/1821257161726685645?s=46">Andrej Karpathy (@karpathy) 的推文</a>: 不久前，自回归语言模型的论文也是那样的。公式化联合似然，对其进行因子分解，推导最大似然估计，讨论与...</li><li><a href="https://x.com/sama/status/1821207141635780938">Sam Altman (@sama) 的推文</a>: 我爱花园里的夏天
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1270857754445221990)** (5 条消息): 

> - `SAM 2 Pod 发布`
> - `SAM 的用户统计数据`
> - `对 SAM 2 的未来预测`
> - `SAM 2 中的视频内容`
> - `与往期节目的关联` 


- **SAM 2 Pod 的发布**：[Latent Space 播客](https://x.com/latentspacepod/status/1821296511260504408) 的最新一期现已上线，重点关注 **SAM 2**。
   - 鼓励听众收听由 **Nikhila Ravi** 和 **Joseph Nelson** 带来的深度见解。
- **SAM 用户统计数据揭示巨大影响力**：来自客座联合主持人的有趣引用提到，在 RoboFlow 上使用 **SAM** 标注了 **4900 万张图片**，估计节省了用户 **35 年** 的时间。
   - 仅在过去的 **30 天** 内，就有 **500 万** 张图片被标注，突显了持续的用户参与度。
- **对 SAM 2 效率的预测**：由于 **SAM 2** 的速度提高了 **6 倍** 并且能够处理视频，人们对其可能为用户节省的时间感到兴奋。
   - *鉴于过去的成功，社区想知道凭借新的功能，还能实现多少时间节省。*
- **更加强调视频内容**：SAM 2 这一期包含比平时更多的 **视频内容**，标志着播客呈现风格的转变。
   - 听众可以观看随附的 [YouTube 演示](https://www.youtube.com/watch?v=lOO_gH4kAn8) 以获取实操见解。
- **引用往期的成功节目**：播客主持人将本次内容与之前对各种 AI 工具的报道联系起来，将 **SAM 2** 置于其持续的叙事中。
   - 诸如 [Segment Anything 1](https://www.latent.space/p/segment-anything-roboflow) 等节目被提及为他们不断深入讨论的基础。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/latentspacepod/status/1821296511260504408">来自 Latent.Space (@latentspacepod) 的推文</a>: http://x.com/i/article/1821295838208909312</li><li><a href="https://x.com/swyx/status/1821298796841541956">来自 swyx 🍓 (@swyx) 的推文</a>: 我们与 @nikhilaravi 合作的 SAM 2 pod 上线了！来自客座联合主持人 @josephofiowa 的有趣 SAM 1 引用：“我最近从过去一年 @RoboFlow 中 SAM 的使用情况中提取了统计数据。用户...”</li><li><a href="https://www.latent.space/p/sam2">Segment Anything 2: 演示优先的模型开发</a>: 别再费劲保持绝对静止了：这个视觉模型现在有了记忆！与 Facebook AI Research 的 Nikhila Ravi 以及特别回归的客座主持人 Roboflow 的 Joseph Nelson 一起探讨 SAM 2</li><li><a href="https://youtu.be/lOO_gH4kAn8">Segment Anything 2: 记忆 + 视觉 = 客体恒常性 —— 与 Nikhila Ravi 和 Joseph Nelson</a>: 别再费劲保持绝对静止了：这个视觉模型现在有了记忆！与 Facebook AI Research 的 Nikhila Ravi 以及特别回归的客座主持人...
</li>
</ul>

</div>
  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1270826421563953152)** (47 条消息🔥): 

> - `Midjourney CEO 对开源的态度`
> - `ASL 模型讨论`
> - `合成语音数据集创建`
> - `Flux 图像生成`
> - `用于无障碍辅助的 AI 应用` 


- **Midjourney CEO 批评开源**：Midjourney CEO 表达了**对开源的怀疑态度**，认为本地模型无法与他们使用 **64 个 GPU** 的服务竞争。他声称开源模型缺乏连贯性且没有商业意义，并将 **ControlNet** 视为唯一的成功案例。
   - 批评者反驳称，Midjourney 的产品类似于开源所能实现的“劣质版本”，并指出了 Flux 中的**过拟合 (overfitting) 问题**：“它看起来有一种塑料感。”
- **ASL 语言模型概念出现**：一位用户询问关于开发一款将**语音转换为 ASL (美国手语)** 的应用，考虑到使用手语图像训练模型的挑战。有人建议对现有模型进行微调 (fine-tune)，以创建一个有效的 **ASL 翻译工具**。
   - 另一位用户就改进语音识别模型提出了建议，探讨它们是否能有效地**利用** emoji 来**表示手势**。
- **提出合成语音数据集构想**：一名成员提议使用 **so-vits-svc** 通过转换音频文件中的声音来创建合成数据集，在保留内容的同时增加多样性。这可能使模型能够捕捉到更广泛的语音**情感 (emotions)** 表达。
   - 该提议强调生成更多样化的数据集，以便在模型难以区分基本**人口统计分类 (demographic classifications)** 的场景中获得更好的结果。
- **Flux 模型讨论继续**：用户之间关于 **Flux** 的对话仍在继续，一些人将其描述为“有趣的玩具”，但表示它没有取得显著进展。人们对其明显的**过拟合**表示担忧，并对其在图像生成方面的整体有效性提出了质疑。
   - 讨论强调了 **Flux** 与 Midjourney 相比的认知度，并评论了更具创新性和针对性的**微调 (fine-tuning)** 的重要性。
- **多种用于无障碍辅助的 AI 应用**：出现了各种旨在增强无障碍功能的 AI 应用建议，例如一个用于语音识别的**尊重隐私**的 IP Relay 应用。讨论强调利用**本地推理技术 (local inference techniques)**，通过在本地将语音转换为文本来帮助听障人士。
   - 众包创意展示了利用 AI 产生深远影响的浓厚兴趣，特别是专注于超越简单技术解决方案的现实应用。


  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1270903254041235507)** (2 条消息): 

> - `频率空间分析`
> - `视觉数据对比` 


- **频率空间分析待开展**：一位成员表示他们尚未探索**频率空间中的原始数据 (raw data in frequency space)**，并表示有必要这样做。
   - 这暗示了数据集中一个潜在的深度调查领域。
- **视觉数据对比似乎一致**：另一位成员怀疑频率空间中的数据在**肉眼看来**几乎是完全相同的。
   - 这种认知引发了关于数据变化的视觉可辨别性的疑问。


  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1270999957884895243)** (3 条消息): 

> - `多后端重构`
> - `Google Gemini 降价`
> - `元宇宙中的 H100` 


- **多后端重构安装顺利**：一位成员确认他们成功安装了 **multi-backend-refactor**，没有遇到任何问题。
   - 他们表示准备好关注后续的进展。
- **Google Gemini 大幅降价**：一位成员分享了一段名为“Google Gemini Insane Price Cuts!!!”的 [YouTube 视频](https://www.youtube.com/watch?v=3ICC4ftZP8Y)，强调了 **Gemini 1.5 Flash** 的新降价政策。
   - 视频指出了价格的大幅下调，并引导观众前往 [Google Developers 博客](https://developers.googleblog.com/en/gemini-15-flash-updates-google-ai-studio-gemini-...) 获取详细信息。
- **呼吁在元宇宙中提供 H100**：一位成员幽默地表示，**Zuck** 需要在元宇宙中提供 **H100** GPU。
   - 这突显了虚拟环境对先进计算资源的持续需求。



**提到的链接**：<a href="https://www.youtube.com/watch?v=3ICC4ftZP8Y">Google Gemini Insane Price Cuts!!!</a>：Google Gemini 1.5 Flash 迎来了疯狂降价！🔗 链接 🔗 详情 - https://developers.googleblog.com/en/gemini-15-flash-updates-google-ai-studio-gemini-...

  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1271078532713287692)** (29 条消息🔥): 

> - `Training Dataset Size` (训练数据集大小)
> - `Prompt Formatting for Inference` (推理时的 Prompt 格式化)
> - `LoRA Import Errors` (LoRA 导入错误)
> - `Using Alpaca Format for Fine-tuning` (使用 Alpaca 格式进行微调)
> - `Llama 3 Model Information` (Llama 3 模型信息)


- **使用 38k 数据集进行训练**：一位成员提到他们使用 **38k** 条目的数据集训练模型，在 **RTX 4090** 上耗时 **32 小时**。
   - 他们推测其配置中的 **learning rate**（学习率）可能过高。
- **正确的 Prompt 格式化**：多位成员讨论了在推理过程中针对特定任务的 Prompt 使用 **Alpaca format** 的重要性。
   - 他们强调，对于聊天场景，输出需要遵循与 fine-tuning（微调）期间使用的相同格式。
- **LoRA 导入错误**：一位用户在导入其 **LoRA** 时遇到错误，不得不从其 adapter 配置中删除两个数值。
   - 另一位成员建议将 LoRA 合并到基础模型中，以潜在地解决此问题。
- **配置澄清**：对 fine-tuning 配置与对话 Prompt 要求之间的区别进行了澄清。
   - 会议强调，使用正确的输出结构对于有效的模型性能至关重要。
- **Llama 3 训练详情**：一位成员询问了 **Llama 3.1 70B** 模型的训练细节，以及在其 fine-tuning 期间使用的数据/掩码（masks）。
   - 他们注意到，现有的 Token 被重命名以充当 special tokens（特殊 Token），而不是创建新的 Token。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/axolotl-ai-co/llama-3-8b-chatml">axolotl-ai-co/llama-3-8b-chatml · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/oobabooga/text-generation-webui">GitHub - oobabooga/text-generation-webui: A Gradio web UI for Large Language Models.</a>：一个用于 Large Language Models 的 Gradio web UI。可以通过在 GitHub 上创建账号来为 oobabooga/text-generation-webui 做出贡献。
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1271064477319692339)** (8 条消息🔥): 

> - `Querying PDFs via API` (通过 API 查询 PDF)
> - `Retrieval-Augmented Generation` (检索增强生成)
> - `Cohere Documentation` (Cohere 文档)
> - `Cohere and Fujitsu Partnership` (Cohere 与 Fujitsu 的合作伙伴关系)
> - `Langchain Integration` (Langchain 集成)


- **Almosnow 寻求 API 文件上传指导**：一位成员希望通过 API 复制 coral.cohere.com UI 上的 PDF 查询功能，但找不到相关文档。
   - 他们注意到各种 POST 端点（如 `/v1/conversations/upload_file`），但不确定这些是否属于官方文档。
- **Mapler 提供有用资源**：另一位成员 mapler 提供了关于通过 Cohere API 使用 Retrieval-Augmented Generation 的资源，并链接到一篇 [博客文章](https://cohere.com/llmu/rag-start)。
   - 他们还分享了 RAG 的文档和一个代码片段，以演示如何生成有依据的回答（grounded answers）。
- **Almosnow 感谢帮助**：Almosnow 对 mapler 分享的资源表示感谢，并表示将查看提供的材料。
   - 这展示了成员们渴望在咨询中互相帮助的协作氛围。
- **Rashmi 的自我介绍**：一位名为 rashmi 的成员通过简单的问候介绍了自己，表示参与到了讨论中。
   - 这突显了社区的一面，新成员在这里受到欢迎。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://cohere.com/llmu/rag-start">Getting Started with Retrieval-Augmented Generation</a>：LLM University 关于 Retrieval-Augmented Generation 模块的第一部分。</li><li><a href="https://docs.cohere.com/docs/retrieval-augmented-generation-rag">Retrieval Augmented Generation (RAG) - Cohere Docs</a>：Retrieval Augmented Generation (RAG) 是一种利用外部数据源生成文本以提高准确性的方法。Chat API 结合 Command 模型可以帮助生成有依据的文本...
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1270978870241132588)** (5 条消息): 

> - `Cohere embeddings 错误`
> - `结合 Llamaparse 的 RAG 模型`
> - `Azure AI Search 集成问题`
> - `embed 端点返回 500 错误`
> - `在 Prompt 中使用 preamble ID` 


- **Cohere embeddings 遇到问题**：一位用户在使用带有提供 API 密钥的 `CohereEmbeddings` 时，遇到了 ValueError 消息，提示无法将字符串转换为浮点数。
   - *An error occurred: could not convert string to float: 'float_'* 表明输入格式存在底层问题。
- **RAG 模型在配合 Llamaparse 时遇到困难**：一位用户询问如何成功将 RAG 模型与 Embedding 和 Reranking 流程以及 Llamaparse 结合使用，其中行内容可能包含混合的不完整句子。
   - 他们寻求社区关于在数据处理过程中如何理解不相关文本块的见解。
- **Azure AI Search 集成结果不一致**：另一位用户报告称，尽管成功将数据向量化，但在将 Cohere embeddings 集成到 Azure AI Search 索引时，查询结果不一致。
   - 即使 'value' 中有得分文档，'@search.answers' 也经常返回为空，这阻碍了 RAG 的有效使用。
- **embed 端点出现 500 错误**：一位用户提到 embed 端点间歇性出现 500 错误，并提供了一个特定的错误 ID 用于追踪。
   - 他们请求协助确定是否是他们提交的数据导致了这些错误。
- **寻求使用 preamble ID 的帮助**：一位用户寻求关于利用 preamble ID 在整个输入文本中泛化 Prompt 的帮助。
   - 他们的咨询表明需要关于有效 prompt engineering 技术的指导。



**提到的链接**：<a href="https://learn.microsoft.com/en-sg/azure/search/vector-search-integrated-vectorization-ai-studio?tabs=cohere).">使用 Azure AI Studio 模型进行集成向量化 - Azure AI Search</a>：了解如何在 Azure AI Search 索引过程中使用 AI Studio 模型对内容进行向量化。

  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1270831369727115375)** (16 条消息🔥): 

> - `Cohere-toolkit`
> - `默认工具激活`
> - `Preamble 调整`
> - `自定义部署` 


- **在 Cohere-toolkit 中默认启用工具**：一位成员讨论了如何在 **Cohere-toolkit** 中默认启用工具，建议在 preamble 中添加 `always use the <tool> tool`。
   - 他们指出，必须将该工具包含在工具列表中，它才能正常工作。
- **首次尝试默认工具加载**：另一位成员分享了他们修改 `cohere_platform.py` 中的 `invoke_chat_stream` 以实现默认工具加载并添加 preamble 的经验。
   - 他们还表达了创建自定义部署的意图，该部署将提供带有默认工具的有限模型选择。
- **UI 实现的异常情况**：有成员提到，虽然模型使用了该工具，但 UI 并没有显示其已激活，这导致了一些困惑。
   - 这种差异引发了关于 UI 反馈与模型实际功能之间关系的疑问。
- **正常运行所需的工具列表要求**：成员们确认，为了让模型利用该工具，还必须在调用 `co.chat` 时将其添加到工具列表中。
   - 无论是否为自定义模型，这都是必要的。


  

---



### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/)** (1 条消息): 

jerryjliu0: 5 分钟后开始！ ^^
  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1270892087906402334)** (3 条消息): 

> - `RAG pipeline 可观测性`
> - `Workflows 抽象`
> - `LlamaIndex Sub-Question Query Engine`
> - `使用 Workflows 进行 Agent 调试` 


- **RAG Pipeline 需要增强的可观测性**：对于 **RAG pipelines** 来说，一个被低估的必要性是不仅要捕获查询时的追踪（traces），还要检查源文档是如何被分块（chunked）的。有人对与不当上下文分块相关的检索问题表示担忧，询问内容是否在中间被切断。
   - 正如最近的一条 [推文](https://twitter.com/llama_index/status/1821332562310205918) 所述，这强调了确保最佳分块以提高检索性能的重要性。
- **对用于 Gen AI 的 Workflows 感到兴奋**：团队继续对 **Workflows** 保持热情，这是一种用于构建复杂 Agentic 生成式 AI 应用的新抽象。他们通过在一段 [新视频](https://twitter.com/llama_index/status/1821575082516660440) 中展示重新构建 LlamaIndex 内置的 Sub-Question Query Engine，证明了其在处理实际工作流方面的便捷性。
   - 这为有效部署复杂的查询引擎和工作流奠定了坚实的基础。
- **使用 Workflows 构建 Agent 的力量**：来自 **@ArizePhoenix** 的一篇博客文章强调了使用 **Workflows** 来观察和调试 Agent 的优势。该 [文章](https://twitter.com/llama_index/status/1821617012080308543) 详细介绍了与传统方法相比，基于事件的架构如何实现更灵活和循环的设计。
   - 对于那些希望使用 **Phoenix** 平台增强 Agent 构建能力的人来说，这是一个宝贵的资源。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1271011022538276928)** (20 messages🔥): 

> - `LongRAG 论文`
> - `Self-routing 技术`
> - `LLM 评估基准`
> - `Token 大小测量`
> - `在 LlamaIndex 中使用 API` 


- **LongRAG 论文对比引发关注**：分享了 LongRAG 论文，该论文对比了检索增强生成 (RAG) 和长上下文 LLM，指出在资源充足的情况下，**长上下文模型**的表现优于 RAG。
   - 成员们表示希望看到涉及 **Claude 3.5** 的对比，并讨论了 *LangChain 的 Lance 提出的* 关于长上下文的方法论。
- **Self-routing 技术提高效率**：LongRAG 论文中提出的 **Self-Route 方法** 根据自我反思 (self-reflection) 将查询路由到 RAG 或长上下文模型，在保持性能的同时显著降低了计算成本。
   - 成员们建议利用元数据 (metadata) 进行 **父文档检索 (parent-document retrieval)** 等创新来增强检索系统，并强调了创建可靠元数据标签的挑战。
- **评估基准引发担忧**：讨论围绕评估数据集对 **数据泄漏 (data leakage)** 的敏感性展开，特别是那些基于已知数据集的评估，这会影响模型评估的准确性。
   - 有人指出，许多数据集的格式过于整洁，无法反映现实世界的应用，因此需要更 **真实的评估指标**。
- **推荐 Token 计数工具**：一位成员询问测量文档 Token 大小的工具，大家推荐了 OpenAI 的 **tiktoken** 作为热门选择，特别是对于使用 OpenAI 模型的用户。
   - 社区对寻找一个可供大家使用的此类工具表现出兴趣，并分享了 GitHub 资源。
- **将 API 与 LlamaIndex 集成**：一位成员寻求关于在 LlamaIndex 中使用 **serpapi** 或 **serperapi** 以增强查询结果的指导，旨在实现包含网页爬取能力的集成。
   - 作为回应，建议用户可以创建利用 serpapi 的自定义引擎，或者使用内置 serpapi 工具的 Agent。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/abacaj/status/1821626025584828747?s=46&t=lqHyE7PE7Ct6jUJNcJPPxg">anton (@abacaj) 的推文</a>：一个 42 页的 PDF（显示为 30k tokens），包含图像和文本，丢进 gemini-flash 1.5，每个答案都正确……彻底结束了</li><li><a href="https://arxiv.org/abs/2407.16833">检索增强生成还是长上下文 LLM？一项综合研究与混合方法</a>：检索增强生成 (RAG) 一直是大型语言模型 (LLM) 高效处理超长上下文的强大工具。然而，最近的 LLM 如 Gemini-1.5 和 GPT-4 表现出了卓越的……</li><li><a href="https://www.youtube.com/watch?v=UlmyyYQGhzc)">RAG 真的死了吗？测试 GPT4-128k 中的多事实检索与推理</a>：长上下文 LLM 检索最受欢迎的基准之一是 @GregKamradt 的“大海捞针 (Needle in A Haystack)”：将一个事实（针）注入到……（干草堆）中。</li><li><a href="https://github.com/openai/tiktoken">GitHub - openai/tiktoken: tiktoken 是一款用于 OpenAI 模型的快速 BPE 分词器。</a>：tiktoken 是一款用于 OpenAI 模型的快速 BPE 分词器。 - openai/tiktoken
</li>
</ul>

</div>
  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1270862078797811853)** (1 messages): 

> - `LLAMA 3 模型性能`
> - `GitHub Issue #1285` 


- **对 LLAMA 3 生成质量的担忧**：一位成员报告在使用 **LLAMA 3 8B instruct 模型**时，输入提示词 "anything" 导致了意外的输出。
   - *如果我这边还需要提供其他输入，请告诉我*，这促使其他人分享他们的经验或在 [GitHub issue 页面](https://github.com/pytorch/torchtune/issues/1285)上讨论该问题。
- **关于 GitHub Issue #1285 的讨论**：该成员引用了一个特定的 [GitHub issue](https://github.com/pytorch/torchtune/issues/1285)，该 issue 解决了该模型的生成质量问题。
   - 他们邀请其他人就该话题发表评论或见解，强调了集体反馈的必要性。



**提到的链接**：<a href="https://github.com/pytorch/torchtune/issues/1285">生成质量 · Issue #1285 · pytorch/torchtune</a>：我使用 LLAMA 3 8B instruct 模型并输入提示词 "anything"，得到了以下结果：chat_format: null checkpointer: _component_: torchtune.utils.FullModelMetaCheckpointer checkpoin...

  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1270849984954237081)** (17 messages🔥): 

> - `RTX A4000 和 A2000 性能`
> - `内存优化技术`
> - `RLHF 清理讨论`
> - `Torchchat 生成优化`
> - `文档和教程计划` 


- **评估 RTX A4000 和 A2000 的微调性能**：成员们讨论了 **RTX A4000** 和 **RTX A2000** 的性能特征，两者均配备 **16GB** 显存。注意到尽管内存指标良好，但在 **1.5B** 模型上进行全量微调（full fine-tune）的性能似乎较低。
   - 一位成员建议通过增加默认 **batch size** 来管理内存开销，有可能将工作负载适配到 **12GB** 显存中。
- **内存优化参数审查中**：一位成员提到了对内存优化参数的**推测**，指出虽然像 **LoRA** 这样的配置很有效，但他们目前并未将重点放在这上面。
   - 他们承认其他人可能拥有 **8GB VRAM** 的 GPU，其性能可能快 **2x** 以上，这暗示了更广泛的优化潜力。
- **关于 RLHF 清理的讨论**：一位成员询问了在向公众广泛分享之前对 **RLHF** 进行必要清理的情况，并回顾了之前提到的需要调整的地方。
   - 有迹象表明，成员们愿意合作编写**教程**或**博客文章**，并意识到这需要投入大量精力。
- **宣传和文档计划**：同一位成员表达了启动关于**宣传**工作以及开发**文档**或**教程**讨论的热情，并已有一个大致的路线图。
   - 他们欢迎社区提供任何意见和帮助，以增强这些工作。
- **Torchchat 生成加速**：一位成员提到有兴趣研究 **torchchat** 如何实现更快的模型生成速度，寻求对其优化方案的见解。
   - 这一询问与目前正在进行的简化和加速所讨论的不同模型性能的努力相一致。



**提到的链接**: <a href="https://wandb.ai/salman-mohammadi/torchtune/?nw=nwusersalmanmohammadi">salman-mohammadi</a>: Weights & Biases，机器学习开发者工具

  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1270845213241376928)** (4 messages): 

> - `AI 基础设施部署`
> - `商业化担忧`
> - `内部工具使用` 


- **在 AI 基础设施上构建的自由**：成员们讨论了根据 [定价页面](https://link.to.pricing)，只要没有将 AI 基础设施平台商业化的意图，就可以自由构建任何内容并进行部署。
   - *只要不是商业用途* 且确实符合内部工具的定义，似乎是可以接受的。
- **关于内部工具的澄清**：一位成员表示，只要不以商业化为目标，内部工具可能是没问题的，尽管他们并非该问题的权威。
   - 他们重申，使用这些工具的明确指南仍然有些模糊。
- **商业化协助提议**：一位成员幽默地指出，如果有人确实想将 AI 基础设施平台商业化，应该联系 **Modular** 寻求帮助。
   - 这反映了社区在应对商业化格局的同时，对创新保持开放态度。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1270854292747714682)** (13 messages🔥): 

> - `Running Mojo in Windows DEV Environment` (在 Windows 开发环境中运行 Mojo)
> - `VS Code with WSL Support` (支持 WSL 的 VS Code)
> - `FancyZones Utility` (FancyZones 工具)
> - `Active Directory and Distributed Databases` (Active Directory 与分布式数据库)


- **Using VS Code with WSL for Mojo Development**: 一位用户询问在 WSL 上安装 **Mojo Max** 后如何在 Windows 开发环境中运行 **Mojo**。另一位用户建议使用支持 WSL 的 **VS Code**，这样可以在 Windows 中编辑，同时在 Linux 中进行构建。
   - 使用这种配置时，*你几乎会忘记自己是在 Linux 中开发*。
- **Benefits and Limitations of WSL**: 讨论强调 **WSL** 提供了一个远离杀毒软件干扰的隔离开发环境，尽管它仍然运行在 **C** 盘。成员们指出了与 *reproducibility*（可复现性）相关的限制以及 WSL 提供的其他优势。
   - 一位成员指出了在 Windows 和 Linux 环境之间平衡的奇特情况：*你只需要过上这种“双重生活”*。
- **FancyZones Utility for Windows**: 一位成员分享了 [FancyZones utility](https://learn.microsoft.com/en-us/windows/powertoys/fancyzones) 的链接，这是一个帮助排列和吸附窗口到高效布局以改善工作流的工具。该工具允许自定义区域位置，以便在 Windows 上进行更好的窗口管理。
   - 将窗口拖入定义好的区域会自动调整大小并重新定位，从而提高开发效率。
- **Debate on Active Directory as Distributed Database**: 一位成员幽默地评论说，将 **Active Directory** 称为分布式数据库是对真正分布式数据库的侮辱。他们详细说明了其同步性质，提到它只提供可用性，而没有真正的强一致性或分区容错性。
   - 另一位成员确认 Microsoft 确实在 Windows 上运行分布式数据库，引发了关于该话题的进一步讨论。



**Link mentioned**: <a href="https://learn.microsoft.com/en-us/windows/powertoys/fancyzones">PowerToys FancyZones utility for Windows</a>: 用于将窗口排列和吸附到高效布局的窗口管理器工具。

  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/)** (1 messages): 

seanchatmangpt: https://www.loom.com/share/0ffc1312c47c45fdb61a2ad00102b3da
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1270907182262652931)** (10 messages🔥): 

> - `Inspect for LLM observability` (用于 LLM 可观测性的 Inspect)
> - `DSPy vs. Langgraph` (DSPy 对比 Langgraph)
> - `Performance of optimize_signature vs. COPRO` (optimize_signature 与 COPRO 的性能对比)


- **Inspect for LLM brings eval capabilities**: 一位用户询问了使用 [Inspect](https://github.com/UKGovernmentBEIS/inspect_ai) 进行 **LLM** 可观测性的经验，以及它与 **DSPy** 的集成效果。
   - 虽然没有分享具体的经验，但该工具似乎专注于大语言模型评估。
- **DSPy and Langgraph operate at different levels**: 一位成员解释说，**DSPy** 在 Prompt 空间优化指令和示例，而 **LangGraph** 则是与 **LangChain** 协作的更底层接口。
   - 从本质上讲，**DSPy** 增强性能，而 **LangGraph** 专注于系统架构。
- **Optimize_signature outperforms COPRO on GSM8K**: 一位用户报告称，在 **GSM8K** 的 **Chain of Thought (CoT)** 任务中，使用 **optimize_signature** 的效果优于 **COPRO**，迅速达到了 **20/20** 的分数。
   - 相比之下，**COPRO** 未能达到 zero-shot 指令解决方案，最高分为 **18/20**。
- **Users share experiences with DSPy in production**: 用户询问是否有人在生产环境中使用 **DSPy**，以及 **Langgraph** 和 **DSPy** 之间的区别。
   - 讨论强调虽然这两个工具是互补的，但 **DSPy** 更侧重于优化。



**Link mentioned**: <a href="https://github.com/UKGovernmentBEIS/inspect_ai">GitHub - UKGovernmentBEIS/inspect_ai: Inspect: A framework for large language model evaluations</a>: Inspect：一个用于大语言模型评估的框架 - UKGovernmentBEIS/inspect_ai

  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1270941482240442471)** (1 messages): 

> - `DSPy-Multi-Document-Agent`
> - `requirements.txt file` 


- **User struggles to find requirements.txt**: 一位成员表示难以找到 **DSPy-Multi-Document-Agent** 的 **requirements.txt** 文件。
   - *“我遗漏了什么吗？”* 是其留下的疑问，表明文档或指南中可能存在缺失。
- **Clarification sought about missing files**: 同一位成员专门询问是否遗漏了与 **DSPy-Multi-Document-Agent** 设置相关的任何内容。
   - 这一询问表明所提供的资源可能存在混淆或缺乏清晰度。

### **DSPy ▷ #[colbert](https://discord.com/channels/1161519468141355160/1250300504462856265/1270828835977433088)** (3 条消息): 

> - `qdrant_dspy`
> - `ColBERT and FastEmbed` 


- **探索 qdrant_dspy GitHub 仓库**：一位成员分享了 [qdrant_dspy GitHub 仓库](https://github.com/vardhanam/qdrant_dspy)的链接，重点介绍了如何使用 **Gemma-2b**、**DSPy** 和 **Qdrant** 设计 RAG 流水线。
   - 该仓库展示了这些技术的集成，重点在于构建高效的检索增强生成（RAG）系统。
- **分享 DSPy 框架资源**：另一位成员提供了 [dspy/retrieve/qdrant_rm.py](https://github.com/stanfordnlp/dspy/blob/main/dspy/retrieve/qdrant_rm.py) 的链接，这是用于编程基础模型的 **DSPy** 框架的一部分。
   - 该资源强调了 DSPy 的多功能性和能力，增强了对本地 VectorDB 交互的理解。
- **对结合 ColBERT 和 FastEmbed 的 STORM 表现出兴趣**：一位成员表示希望在运行 **STORM** 的同时，利用 **ColBERT** 和 **FastEmbed** 在本地 VectorDB 上进行搜索。
   - 这种方法反映了人们对于结合多种先进技术以优化本地向量搜索任务日益增长的兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/stanfordnlp/dspy/blob/main/dspy/retrieve/qdrant_rm.py">dspy/dspy/retrieve/qdrant_rm.py at main · stanfordnlp/dspy</a>: DSPy：用于编程（而非提示）基础模型的框架 - stanfordnlp/dspy</li><li><a href="https://github.com/vardhanam/qdrant_dspy">GitHub - vardhanam/qdrant_dspy: Designing a RAG pipeline using Gemma-2b, DSPy, and Qdrant</a>: 使用 Gemma-2b、DSPy 和 Qdrant 设计 RAG 流水线 - vardhanam/qdrant_dspy
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1270823029110210712)** (6 条消息): 

> - `DEBUG environment variable issue`
> - `Tinygrad Tensor Puzzles`
> - `getenv function ValueError` 


- **getenv 函数中的 ValueError**：一位用户在导入时遇到了 `ValueError`，提示 **'invalid literal for int() with base 10: WARN'**，这与一个环境变量有关。
   - 另一位成员建议检查 **environment variables**（环境变量），因为某些设置被设为了 **WARN**，该用户随后确认了这一点。
- **DEBUG 环境变量被设置为 WARN**：用户发现 **DEBUG** 环境变量被设置为 **'WARN'**，这可能导致了 getenv 函数的问题。
   - 他们注意到其 Python 脚本运行正常，这意味着该问题可能仅限于 notebook 环境。
- **介绍 Tinygrad Tensor Puzzles**：一位成员介绍了 **Tinygrad Tensor Puzzles**，这是一套包含 **21 个有趣且具挑战性的谜题** 的集合，旨在从第一性原理出发掌握张量库。
   - 该项目将 **Sasha 的 PyTorch Tensor-Puzzles** 适配到了 tinygrad，并鼓励初学者和资深用户参与贡献和编写谜题。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/obadakhalili/status/1821587868562940179">Obada Khalili (@obadakhalili) 的推文</a>: 介绍 @__tinygrad__ Tensor Puzzles 🎉 一套包含 21 个有趣且具挑战性的张量谜题集合，旨在从第一性原理出发掌握像 tinygrad 这样的张量库，而不依赖于魔法函数或...</li><li><a href="https://github.com/obadakhalili/tinygrad-tensor-puzzles">GitHub - obadakhalili/tinygrad-tensor-puzzles: Solve puzzles to improve your tinygrad skills</a>: 通过解谜提高你的 tinygrad 技能。通过创建账号参与 obadakhalili/tinygrad-tensor-puzzles 的开发。
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1270944437761085440)** (2 messages): 

> - `tinygrad notes`
> - `fine-tuning tutorials`
> - `computer algebra optimization` 


- **通过有用的笔记探索 tinygrad**：分享了一系列 [教程/学习笔记](https://mesozoic-egg.github.io/tinygrad-notes/)，旨在帮助理解 **tinygrad** 的内部机制并开始贡献。
   - 此外还展示了 [快速入门指南](https://github.com/tinygrad/tinygrad/blob/master/docs/quickstart.md) 等资源，虽然对初学者可能不太友好，但提供了很好的基础见解。
- **用于优化的计算机代数笔记**：最近的更新包括 [计算机代数学习笔记](https://github.com/mesozoic-egg/computer-algebra-study-notes/tree/main)，虽然不直接与 tinygrad 挂钩，但与其优化过程显著相关。
   - 计算机代数技术的整合可能为寻求提升 tinygrad 性能的开发者提供宝贵的视角。
- **关于 fine-tuning 教程的咨询**：一名成员询问了可用的 **fine-tuning** 教程，强调了该特定领域对资源的需求。
   - 交流的信息中未提到具体的教程，表明寻求此类信息的用户可能面临资源缺口。



**Link mentioned**: <a href="https://mesozoic-egg.github.io/tinygrad-notes/">Tutorials on Tinygrad</a>: Tutorials on tinygrad

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1270916275131912214)** (4 messages): 

> - `Open Source Vision Models`
> - `MiniCPM-V 2.6 Performance` 


- **开源视觉模型的查询**：成员们正在寻求适用于 **vision tasks** 的开源模型建议，询问了本地和 API 选项。
   - *一位成员在提出初始问题时表达了好奇。*
- **MiniCPM-V 2.6 超越竞争对手**：一名成员指出 **MiniCPM-V 2.6** 模型在多图应用中表现优于 **Gemini 1.5 Pro**、**GPT-4V** 和 **Claude 3.5 Sonnet**，并分享了相关资源链接。
   - 他们提供了 [Hugging Face 页面](https://huggingface.co/openbmb/MiniCPM-V-2_6) 和 [GitHub 仓库](https://github.com/OpenBMB/MiniCPM-V) 的链接以供进一步探索。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/monkey-monkey-eating-monkey-eating-strawberries-kardie-gif-gif-22488578">Monkey Monkey Eating GIF - Monkey Monkey Eating Monkey Eating Strawberries - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/openbmb/MiniCPM-V-2_6">openbmb/MiniCPM-V-2_6 · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/OpenBMB/MiniCPM-V">GitHub - OpenBMB/MiniCPM-V: MiniCPM-V 2.6: A GPT-4V Level MLLM for Single Image, Multi Image and Video on Your Phone</a>: MiniCPM-V 2.6: 手机端支持单图、多图和视频的 GPT-4V 级 MLLM - OpenBMB/MiniCPM-V
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1271131103822086248)** (2 messages): 

> - `Shipping updates` 


- **关于发货更新的咨询**：一名成员对任何 **发货更新** 表示关注。
   - 未提供具体回复，但分享了一个相关的 Discord 线程链接以供参考。
- **Mikebirdtech 回应**：一名成员分享了一个可能讨论发货更新的 Discord 频道链接。
   - 该链接指向 [此 Discord 消息](https://discord.com/channels/1146610656779440188/1194880263122075688/1266055462063964191) 以获取更多上下文。


  

---

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1271173919029727252)** (3 条消息): 

> - `Llama 团队问答`
> - `Poe 黑客松` 


- **Llama 团队在 arXiv 上回答问题**：如果有兴趣，**Llama 团队**正在 [arXiv 讨论论坛](https://alphaxiv.org/abs/2407.21783v1)上回答问题。
   - 这是一个直接从团队获取技术查询和见解的机会。
- **Quora 举办机器人开发黑客松**：构建 **Poe** 的 Quora 正在举办一场线下和虚拟的 [黑客松](https://x.com/poe_platform/status/1820843642782966103)，重点是利用新的 **Previews feature** 构建机器人。
   - 参与者将使用最新的 LLM（如 **GPT-4o** 和 **Llama 3.1 405B**）创建创新的聊天内生成式 UI 体验。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/poe_platform/status/1820843642782966103">来自 Poe (@poe_platform) 的推文</a>: 我们很高兴宣布与 @agihouse_org 围绕我们新的 Previews 功能举办为期一天的黑客松！竞争使用最新的 LL... 创建最具创新性和实用性的聊天内生成式 UI 体验。</li><li><a href="https://alphaxiv.org/abs/2407.21783v1">Llama 3 模型群 | alphaXiv</a>: 现代人工智能 (AI) 系统由基础模型驱动。本文介绍了一组新的基础模型，称为 Llama 3。它是一个原生支持... 的语言模型群。</li><li><a href="https://x.co">出售域名 | 购买域名 | 停放域名</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1271024432315367445)** (2 条消息): 

> - `通用 AI vs 非通用 AI`
> - `AI 应用类型` 


- **探索生成式之外的 AI**：一位成员发起了一场讨论，表达了他们对非生成式 **AI** 的欣赏，并邀请其他人分享想法。
   - *你脑海中想到了哪些类型的 AI 应用？*
- **建议的多样化 AI 应用**：另一位成员回应了一些建议，包括 **computer vision**、**forecasting**、**recommendation systems** 和 **NLP**，这些都属于非生成式的 AI 类型。
   - 这些应用突显了超出生成能力之外的 **AI** 技术的广阔频谱。


  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1270823164359868490)** (3 条消息): 

> - `Vercel 故障`
> - `Anthropic 错误率` 


- **Vercel 经历间歇性停机**：Vercel 目前正面临故障，影响了 OpenRouter 服务，正如其 [状态更新](https://x.com/OpenRouterAI/status/1821267624228966781) 中所述。经过多次更新，据报告服务在东部时间下午 3:45 恢复稳定。
   - 随着 Vercel 实施修复，监控仍在继续，并将持续更新 [Vercel 状态页面](https://www.vercel-status.com/)。
- **Anthropic 应对高上游错误率**：Anthropic 报告其服务错误率升高，特别是 3.5 Sonnet 和 3 Opus，并已实施了缓解措施和临时解决方案。截至太平洋夏季时间 8 月 8 日 17:29，成功率已恢复正常水平，Claude.ai 免费用户的访问也已恢复。
   - 他们正在密切监控情况，并随着问题的解决继续提供更新。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://status.anthropic.com/">Anthropic 状态</a>: 未找到描述</li><li><a href="https://x.com/OpenRouterAI/status/1821267624228966781">来自 OpenRouter (@OpenRouterAI) 的推文</a>: 注意：由于 @Vercel 平台故障，我们正面临停机，该故障未显示在他们的状态页面上。我们的状态将在 https://status.openrouter.ai/ 上可见。
</li>
</ul>

</div>
  

---



---



---



---



---



{% else %}


> 完整的频道详情已为邮件格式进行截断。
> 
> 如果你想查看完整详情，请访问此邮件的网页版本：[{{ email.subject }}]({{ email_url }})!
>
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}