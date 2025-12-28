---
companies:
- anthropic
- zep
- nvidia
date: '2024-10-23T02:08:12.665635Z'
description: '**Anthropic** 发布了全新的 Claude 3.5 模型：**3.5 Sonnet** 和 **3.5 Haiku**。新模型显著提升了编程性能，其中
  Sonnet 在 **Aider** 和 **Vectara** 等多个编程基准测试中位居榜首。


  全新的 **Computer Use API** 能够通过视觉能力控制计算机，其评分显著高于其他 AI 系统，展示了 AI 驱动的计算机交互技术的进步。**Zep**
  推出了用于 AI 智能体记忆管理的云端版本，并强调了**多模态记忆**面临的挑战。此外，本次更新还提到了来自 **NVIDIA（英伟达）** 的 **Llama
  3.1** 和 **Nemotron** 模型。'
id: 92b32e84-5e69-4ee5-8bd5-b272bd08060c
models:
- claude-3.5-sonnet
- claude-3.5-haiku
- llama-3.1
- nemotron
original_slug: ainews-claude-35-sonnet-new-gets-computer-use
people:
- philschmid
- swyx
title: Claude 3.5 Sonnet (新版) 获“电脑使用” (Computer Use) 功能。
topics:
- coding
- benchmarks
- computer-use
- vision
- multimodal-memory
- model-updates
- ai-integration
---

<!-- buttondown-editor-mode: plaintext -->**更好的模型命名就是我们所需要的一切。**

> 2024/10/21-2024/10/22 的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter](https://twitter.com/i/lists/1585430245762441216) 和 **32** 个 Discord（**232** 个频道和 **3347** 条消息）。预计节省阅读时间（以 200wpm 计算）：**341 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

Anthropic 并没有发布广受期待（且现已[无限期推迟](https://news.ycombinator.com/item?id=41920044)）的 Claude 3.5 Opus，而是发布了新款 3.5 Sonnet 和 3.5 Haiku，为每个模型都带来了提升。


![image.png](https://assets.buttondown.email/images/1d761c93-a21f-49cb-b423-895800d5ee12.png?w=960&fit=max)


3.5 Sonnet 已经在代码编写方面带来了显著提升。新款 3.5 Haiku（基准测试见 [model card](https://assets.anthropic.com/m/1cd9d098ac3e6467/original/Claude-3-Model-Card-October-Addendum.pdf)）在“许多评估中达到了 Claude 3 Opus 的性能，且成本与上一代 Haiku 相同，速度也相近”。


![image.png](https://assets.buttondown.email/images/4cf7338a-9345-4c0c-90d7-fae6013a7957.png?w=960&fit=max)

值得注意的是，在编程方面，它在 SWE-bench Verified 上的[性能提升](https://x.com/AnthropicAI/status/1848742740420341988)从 33.4% 增加到 **49.0%**，**得分高于 o1-preview 的 41.4%**，且无需任何复杂的推理步骤。然而，**在数学方面**，3.5 Sonnet 27.6% 的最高纪录与 o1-preview 的 83% 相比仍显逊色。

**其他基准测试：**

- **Aider**：新款 Sonnet 在 aider 的代码[编辑排行榜](https://x.com/paulgauthier/status/1848808149945618933)中以 **84.2%** 位居榜首，并在 aider [要求更高](https://x.com/paulgauthier/status/1848839965201076618)的重构基准测试中以 **92.1%** 的得分创下 SOTA！
- **Vectara**：在 Vectara 的 Hughes 幻觉评估模型中，Sonnet 3.5 [从](https://github.com/vectara/hallucination-leaderboard) **8.6 降至 4.6**。


**Computer Use**

Anthropic 新推出的 Computer Use API（[文档在此](https://docs.anthropic.com/en/docs/build-with-claude/computer-use)，[演示在此](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo)）引用了 [OSWorld](https://os-world.github.io/) 作为其相关的屏幕操作基准测试——在仅限截图类别中得分 **14.9%**，明显优于排名第二的 AI 系统（7.8%）。


![image.png](https://assets.buttondown.email/images/a2a32df1-bb32-47d8-913f-d7775e7a4725.png?w=960&fit=max)


当被允许使用更多步骤来完成任务时，Claude 的得分为 22.0%。这仍远低于人类 70 多分的表现，但值得关注，因为这本质上是 Adept 此前通过其 Fuyu 模型宣布但从未广泛发布的功能。从简化的角度来看，“computer use”（通过视觉控制计算机）与标准的“tool use”（通过 API/函数调用控制计算机）形成了对比。

示例视频：

[供应商请求表单](https://www.youtube.com/watch?v=ODaHJzOyVCQ)、[通过视觉编写代码](https://www.youtube.com/watch?v=vH2f7cjXjKI)、[Google 搜索和 Google 地图](https://www.youtube.com/watch?v=jqx18KgIzAE)

Simon Willison 对 [GitHub 快速入门](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo)进行了[进一步测试](https://simonwillison.net/2024/Oct/22/computer-use/)，包括在 C 语言中编译并运行 hello world（它已经内置了 gcc，所以直接成功）以及安装缺失的 Ubuntu 软件包。

Replit 也能将 [Claude 作为人类反馈的替代方案](https://x.com/pirroh/status/1848752337080488177)接入 @Replit Agent。

---

**[由 Zep 赞助]** Zep 今天刚刚发布了[他们的云版本](https://shortclick.link/uu8gwd)！Zep 是一个为 AI Agent 和助手提供的低延迟记忆层，能够对随时间变化的事实进行推理。[加入 Discord](https://shortclick.link/wgo7bi) 探讨知识图谱和记忆的未来！

> swyx 评论：随着 Claude 升级后的视觉模型正式支持 computer use，Agent 的记忆存储需要如何改变？你可以看到 Anthropic 简单的[图像记忆](https://github.com/anthropics/anthropic-quickstarts/blob/a306792de96e69d29f231ddcb6534048b7e2489e/computer-use-demo/computer_use_demo/loop.py#L144)实现，但目前还没有**多模态记忆**的解决方案……这是 [Zep Discord](https://shortclick.link/wgo7bi) 的一个热门话题。

---

{% if medium == 'web' %}

**目录**

[TOC]

{% else %}

**目录**和**频道摘要**已移至此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}

---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型更新与发布**

- **Llama 3.1 和 Nemotron**：[@_philschmid](https://twitter.com/_philschmid/status/1848392932627476646) 报道称，NVIDIA 的 Llama 3.1 Nemotron 70B 在 Arena Hard (85.0) 和 AlpacaEval 2 LC (57.6) 中登顶，向 GPT-4 和 Claude 3.5 发起挑战。

- **IBM Granite 3.0**：IBM 发布了 [Granite 3.0 模型](https://twitter.com/rohanpaul_ai/status/1848367865302229466)，参数范围从 4 亿到 8B，在 Hugging Face 的 OpenLLM 排行榜上超越了同等规模的 Llama-3.1 8B。该模型在 12 种语言和 116 种编程语言的 12+ 万亿 token 上进行了训练。

- **xAI API**：[xAI API Beta 版现已上线](https://twitter.com/ibab/status/1848407059961627021)，允许开发者将 Grok 集成到他们的应用程序中。

- **BitNet**：Microsoft 开源了 [bitnet.cpp](https://twitter.com/rohanpaul_ai/status/1848436490159673569)，实现了 1.58-bit LLM 架构。这使得在 CPU 上以每秒 5-7 个 token 的速度运行 100B 参数模型成为可能。

**AI 研究与技术**

- **量化 (Quantization)**：一种[新的线性复杂度乘法 (L-Mul) 算法](https://twitter.com/rohanpaul_ai/status/1848383106736398456)声称可以将 LLM 中逐元素张量乘法的能耗降低 95%，点积能耗降低 80%。

- **合成数据 (Synthetic Data)**：[@omarsar0](https://twitter.com/omarsar0/status/1848445736591163886) 强调了合成数据对于改进 LLM 及其构建系统（Agent、RAG 等）的重要性。

- **Agentic Information Retrieval**：分享了一篇[介绍 Agentic Information Retrieval 的论文](https://twitter.com/omarsar0/status/1848396596230127655)，讨论了 LLM Agent 如何塑造检索系统。

- **RoPE 频率**：[@vikhyatk](https://twitter.com/vikhyatk/status/1848433397842252212) 指出截断最低的 RoPE 频率有助于 LLM 的长度外推 (length extrapolation)。

**AI 工具与应用**

- **Perplexity Finance**：[Perplexity Finance 已在 iOS 上推出](https://twitter.com/AravSrinivas/status/1848480838390059325)，提供财务信息和股票数据。

- **LlamaIndex**：分享了多种使用 LlamaIndex 的应用，包括[报告生成](https://twitter.com/llama_index/status/1848421745101050358)和 [Serverless RAG 应用](https://twitter.com/llama_index/status/1848509130631151646)。

- **Hugging Face 更新**：宣布了新功能，如[面向企业 Hub 订阅的代码库分析](https://twitter.com/ClementDelangue/status/1848410771350249497)以及 [diffusers 中的量化支持](https://twitter.com/RisingSayak/status/1848373306233364847)。

**AI 伦理与社会影响**

- **法律服务**：OpenAI 的 CPO Kevin Weil 讨论了[法律服务领域潜在的颠覆性变化](https://twitter.com/rohanpaul_ai/status/1848381082015580640)，AI 可能将成本降低 99.9%。

- **AI 审计**：宣布将于 10 月 28 日举行关于[第三方 AI 审计、红队测试 (Red Teaming) 和评估的在线研讨会](https://twitter.com/ShayneRedford/status/1848418137110192471)。

**迷因与幽默**

- 分享了关于 ChatGPT 即将到来的生日及潜在礼物的各种推文，包括 [@sama 发出的这一条](https://twitter.com/sama/status/1848486309376991316)。

- 分享了关于 [Google Meet 中 AI 生成背景](https://twitter.com/rohanpaul_ai/status/1848371582231597433)以及 [AI 对视频编辑影响](https://twitter.com/rohanpaul_ai/status/1848504671608180854)的笑话。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. Moonshine：新型开源语音转文本模型挑战 Whisper**

- **[Moonshine 新型开源语音转文本模型](https://petewarden.com/2024/10/21/introducing-moonshine-the-new-state-of-the-art-for-speech-to-text/)** ([Score: 54, Comments: 5](https://reddit.com//r/LocalLLaMA/comments/1g9bll3/moonshine_new_open_source_speech_to_text_model/)): **Moonshine** 是一款新型的**开源语音转文本模型**，声称在保持与 **Whisper** 相当的准确率的同时，速度比其更快。由 **Sanchit Gandhi** 和 **Hugging Face 团队**开发，Moonshine 基于 **wav2vec2**，在 CPU 上的处理速度比 Whisper 快 **30 倍**。该模型已在 [Hugging Face Hub](https://huggingface.co/spaces/sanchit-gandhi/moonshine) 上提供，并可以使用 Transformers 库轻松集成到项目中。
  - **Moonshine** 针对 **Raspberry Pi** 等资源受限的平台，目标是在转录句子时仅使用 **8MB RAM**，而 Whisper 的最低要求为 **30MB**。该模型专注于微控制器和 DSP 的效率，而非与 Whisper large v3 竞争。
  - 用户对尝试 Moonshine 表示兴奋，并指出了 **Whisper 3** 在准确率和幻觉（hallucinations）方面的问题。然而，Moonshine 目前是**仅限英文的模型**，限制了其在多语言应用中的使用。
  - 该项目已在 [GitHub](https://github.com/usefulsensors/moonshine) 上发布，并包含一份[研究论文](https://github.com/usefulsensors/moonshine/blob/main/moonshine_paper.pdf)。一些用户报告了安装错误，可能是由于 Windows 上的 Git 相关问题导致的。


**Theme 2. Allegro: 新型 SOTA 开源文本转视频模型**



- **新型文本转视频模型: Allegro** ([Score: 99, Comments: 8](https://reddit.com//r/LocalLLaMA/comments/1g99lms/new_texttovideo_model_allegro/)): **Allegro** 是一款新型的**开源文本转视频模型**，已发布[详细论文](https://arxiv.org/abs/2410.15458)和 [Hugging Face 实现](https://huggingface.co/rhymes-ai/Allegro)。该模型基于开发者之前的**开源视觉语言模型 (VLM) Aria** 构建，后者为监控定位（surveillance grounding）和推理等任务提供了全面的微调指南。
  - **Allegro** 被誉为新的本地文本转视频 **SOTA** (State of the Art)，其 **Apache-2.0 license** 尤其受到欢迎。该模型的开源性质被视为本地视频生成领域的一个积极进展。
  - 讨论了模型的 **VRAM 需求**，选项从 **9.3GB** (含 CPU offload) 到 **27.5GB** (不含 offload) 不等。用户建议将 **T5 model** 量化为较低精度 (fp16/fp8/int8)，以适配 24GB/16GB VRAM 的显卡。
  - 强调了模型使用的灵活性，可以权衡生成质量以减少 VRAM 占用并缩短生成时间 (可能为 **10-30 分钟**)。一些用户讨论了在初始 prompt encoding 后更换 T5 model 的选项，以优化资源利用。


**Theme 3. 字节跳动 AI 破坏事件引发安全担忧**



- **[TikTok 母公司解雇破坏 AI 项目的实习生](https://news.ycombinator.com/item?id=41900402)** ([Score: 153, Comments: 50](https://reddit.com//r/LocalLLaMA/comments/1g8lzqp/tiktok_owner_sacks_intern_for_sabotaging_ai/)): ByteDance，即 **TikTok** 的母公司，据报道**解雇了一名实习生**，原因是其通过插入恶意代码蓄意**破坏 AI 项目**。这起发生在**中国**的事件凸显了与 AI 开发相关的**安全风险**以及科技公司内部威胁的可能性。ByteDance 在例行代码审查中发现了破坏行为，强调了 AI 开发过程中稳健的安全措施和代码审计的重要性。
  - 据称，该实习生通过在 **checkpoint models** 中植入 **backdoors**、插入 **random sleeps** 以减慢训练速度、终止训练任务以及逆转训练步骤来**破坏 AI 研究**。据报道，这是由于对 **GPU resource allocation** 不满所致。
  - ByteDance **已于 8 月解雇了该实习生**，通报了其所在大学和行业机构，并澄清该事件仅影响了**商业技术团队的研究项目**，而非官方项目或大模型。关于“**8,000 张显卡和数百万损失**”的传言被夸大了。
  - 一些用户对该实习生据称缺乏 AI 经验表示质疑，因为他有能力逆转训练过程。其他人指出这是“**职业自杀**”，并推测其可能会被主要科技公司列入**黑名单**。


**Theme 4. PocketPal AI: 移动端本地模型的开源应用**

- **PocketPal AI 已开源** ([Score: 434, Comments: 78](https://reddit.com//r/LocalLLaMA/comments/1g8kl5e/pocketpal_ai_is_open_sourced/)): **PocketPal AI** 是一款用于在 **iOS** 和 **Android** 设备上运行 **local models** 的应用程序，现已 **开源**。该项目的源代码目前已在 [GitHub](https://github.com/a-ghorbani/pocketpal-ai) 上发布，允许开发者探索并为移动平台的设备端 AI 模型实现做出贡献。
  - 用户报告了 **Llama 3.2 1B** 模型令人印象深刻的性能，在 **iPhone 13** 上达到了 **20 tokens/second**，在 **Samsung S24+** 上达到了 **31 tokens/second**。**iOS 版本** 使用了 **Metal acceleration**，这可能是提升速度的关键因素。
  - 社区对该应用的开源表示感谢，许多人称赞其便利性和性能。一些用户建议增加 **donation section** 以支持开发，并请求增加如 **character cards** 集成等功能。
  - 用户将 **PocketPal** 与另一个开源移动端 LLM 应用 **ChatterUI** 进行了比较。PocketPal 因其用户友好性和在 App Store 的可用性而受到关注，而 ChatterUI 则提供更多自定义选项和 API 支持。


- **[🏆 GPU-Poor LLM Gladiator Arena 🏆](https://huggingface.co/spaces/k-mktr/gpu-poor-llm-arena)** ([Score: 137, Comments: 38](https://reddit.com//r/LocalLLaMA/comments/1g8nepp/the_gpupoor_llm_gladiator_arena/)): **GPU-Poor LLM Gladiator Arena** 是一个比较可以在消费级硬件上运行的小型语言模型的竞赛。鼓励参与者提交参数量最大为 **3 billion parameters** 且能在 **24GB VRAM** 或更低配置设备上运行的模型，目标是在保持效率和可访问性的同时，在各种 benchmark 上实现高性能。
  - 用户对 **GPU-Poor LLM Gladiator Arena** 表现出极大的热情，一些人建议加入更多模型，如 **allenai/OLMoE-1B-7B-0924-Instruct** 和 **tiiuae/falcon-mamba-7b-instruct**。该项目因简化了小型模型的比较而受到称赞。
  - 讨论中提到了 **Gemma 2 2B** 的表现，一些用户注意到它与更大模型相比具有强劲的性能。关于 Gemma 友好的对话风格是否会影响人类评估结果也存在争议。
  - 改进建议包括为评估增加 **tie button**（平局按钮）、计算 **ELO ratings** 而非原始胜率，以及引入更强大的统计方法来考虑样本量和对手强度。


**主题 5：开源权重 AI 模型许可证趋于严格的趋势**



- **近期发布的开源权重模型具有更严格的许可证** ([Score: 36, Comments: 10](https://reddit.com//r/LocalLLaMA/comments/1g8olso/recent_open_weight_releases_have_more_restricted/)): 近期发布的 **open-weight AI model**，包括 **Mistral small**、**Ministral**、**Qwen 2.5 72B** 和 **Qwen 2.5 3B**，与早期的 **Mistral Large 2407** 等发布相比，显示出 **restricted licenses** 增加的趋势。随着 AI 模型性能的提高和运行成本效益的提升，许可证条款明显转向严格，未来 **open-weight releases** 可能主要来自 **academic laboratories**。
  - **Mistral** 对小型模型采取更严格的许可可能会损害其品牌，可能导致公司范围内禁用 Mistral 模型，并降低对其 API-only 大型模型的兴趣。用户对缺乏评估模型质量的本地参考点表示担忧。
  - 不发布 **Mistral 3B 模型** 权重的决定被视为 **开源 AI 的负面信号**。这一趋势表明，公司可能会越来越多地将性能良好的小型模型保持私有，以维持竞争优势。
  - 讨论围绕 **Mistral 对盈利的需求** 以维持运营展开，这与 **Meta** 等能够负担得起公开模型的大型公司形成鲜明对比。一些用户认为 Mistral 的做法是为了生存，而另一些人则认为这是 AI 模型许可中令人担忧的趋势的一部分。

## 其他 AI Subreddit 回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**AI 模型开发与发布**

- **ComfyUI V1 桌面应用程序发布**：ComfyUI 宣布推出全新的封装桌面应用，支持一键安装、自动更新，并配备了包含模板工作流和节点模糊搜索的新 UI。它还包含一个拥有 600 多个已发布节点的 Custom Node Registry。[来源](https://www.reddit.com/r/StableDiffusion/comments/1g8ny9o/introducing_comfyui_v1_a_packaged_desktop/)

- **OpenAI 的 o1 模型在增加算力后表现出更强的推理能力**：OpenAI 研究员 Noam Brown 分享道，o1 模型在数学问题上的推理能力随着测试时算力（test-time compute）的增加而提升，且在对数尺度上“没有停止的迹象”。[来源](https://www.reddit.com/r/singularity/comments/1g8nv30/openais_noam_brown_says_the_o1_models_reasoning/)

- **Advanced Voice Mode 在欧盟发布**：OpenAI 的 Advanced Voice Mode 现已在欧盟正式可用。用户反馈其在口音处理方面有所改进。[来源](https://www.reddit.com/r/OpenAI/comments/1g99wbu/advanced_voice_mode_officially_out_in_eu/)

**AI 研究与行业洞察**

- **微软 CEO 谈 AI 开发加速**：Satya Nadella 表示，由于 Scaling Laws 范式，计算能力现在每 6 个月翻一番。他还提到 AI 开发已进入递归阶段，即利用 AI 来构建更好的 AI 工具。[来源 1](https://www.reddit.com/r/singularity/comments/1g90c8k/microsoft_ceo_satya_nadella_says_computing_power/), [来源 2](https://www.reddit.com/r/singularity/comments/1g93nk7/microsoft_ceo_satya_nadella_says_ai_development/)

- **OpenAI 谈 o1 模型的可靠性**：OpenAI 应用研究负责人 Boris Power 表示，o1 模型的可靠性已足以支持 Agent。[来源](https://www.reddit.com/r/singularity/comments/1g947s8/boris_power_head_of_applied_research_at_openai/)

**AI 伦理与社会影响**

- **Sam Altman 谈技术进步**：OpenAI CEO Sam Altman 发推称，“并不是未来会发生得太快，而是过去发生得太慢”，引发了关于技术进步速度的讨论。[来源](https://www.reddit.com/r/singularity/comments/1g988gz/its_not_that_the_future_is_going_to_happen_so/)

**机器人技术进展**

- **Unitree 机器人训练**：一段展示 Unitree 机器人日常训练的视频被分享，展示了机器人在移动性和控制方面的进步。[来源](https://www.reddit.com/r/singularity/comments/1g8y5q8/daily_training_of_robots_unitree/)

**迷因与幽默**

- 一篇题为“训练更多 AI 的 AI”的帖子引发了关于递归式 AI 改进的幽默讨论。[来源](https://www.reddit.com/r/singularity/comments/1g95mrf/an_ai_that_trains_more_ai/)


---

# AI Discord 摘要回顾

> 由 O1-preview 生成的摘要之摘要的摘要

**主题 1. Claude 3.5 在 Computer Use 领域取得突破**

- [**Claude 3.5 成为你的硅基管家**](https://www.anthropic.com/news/3-5-models-and-computer-use)：Anthropic 的 **Claude 3.5 Sonnet** 推出了 beta 版的 'Computer Use' 功能，允许它像人类助手一样在你的电脑上执行任务。尽管存在一些小瑕疵，用户对这一模糊了 AI 与人类交互界限的实验性功能感到兴奋。
- [**Haiku 3.5 跃升至编程巅峰**](https://www.anthropic.com/news/3-5-models-and-computer-use)：全新的 **Claude 3.5 Haiku** 超越了其前代产品，在 SWE-bench Verified 上获得了 **40.6%** 的评分，表现优于 Claude 3 Opus。随着 Haiku 3.5 树立了 AI 辅助编程的新标准，程序员们欢欣鼓舞。
- **Claude 操控电脑，用户挑战极限**：虽然 'Computer Use' 功能具有开创性，但 Anthropic 警告称它是实验性的，并且“*有时容易出错*”。但这并没有削弱社区挑战极限的热情。

**主题 2. Stable Diffusion 3.5 点亮 AI 艺术**

- [**Stability AI 发布 Stable Diffusion 3.5——艺术家的盛宴**](https://stability.ai/news/introducing-stable-diffusion-3-5)：**Stable Diffusion 3.5** 发布，提升了图像质量和 Prompt 遵循能力，对于年收入低于 100 万美元的商业用途免费。它已在 [Hugging Face](https://huggingface.co/stabilityai) 上线，是送给艺术家和开发者的礼物。
- **SD 3.5 Turbo 疾速领先**：全新的 **Stable Diffusion 3.5 Large Turbo** 模型在不牺牲质量的情况下提供了极快的推理速度。用户对这种速度与性能的结合感到兴奋。
- **艺术家辩论：SD 3.5 对决 Flux——谁能问鼎？**：社区正在热议 SD 3.5 是否能在图像质量和美感上取代 **Flux**。早期测试者的评价褒贬不一，但竞争正在升温。

**主题 3. AI 视频生成随 Mochi 1 和 Allegro 升温**

- [**GenmoAI 的 Mochi 1 带来震撼视频**](https://x.com/genmoai/status/1848762405779574990)：**Mochi 1** 树立了开源视频生成的新标准，在 480p 分辨率下提供逼真的动态效果和 Prompt 遵循能力。在 **2840 万美元**资金的支持下， GenmoAI 正在重新定义写实视频模型。
- [**Allegro 在 Text-to-Video 领域表现亮眼**](https://github.com/rhymes-ai/Allegro)：**Rhymes AI** 推出了 **Allegro**，能将文本转化为 15 FPS、720p 的 6 秒视频。早期采用者可以在[此处](https://forms.gle/JhA7BaKvZoeJYQU87)加入等待名单，抢先体验。
- **视频大战开启：Mochi 对决 Allegro——愿最强帧胜出**：随着 Mochi 1 和 Allegro 的加入，创作者们热切期待哪个模型将在 AI 驱动的视频内容领域领先。

**主题 4. Cohere 将图像嵌入多模态搜索**

- [**Cohere 的 Embed 3 终于将图像接入搜索！**](https://cohere.com/blog/multimodal-embed-3)：**Multimodal Embed 3** 支持混合模态搜索，在检索任务中表现出色。现在，你可以将文本和图像数据存储在一个数据库中，使 RAG 系统变得异常简单。
- **图像与文本，终成眷属**：新的 Embed API 增加了一个名为 `image` 的 `input_type`，让开发者可以与文本一起处理图像。虽然每次请求限制一张图像，但这是统一数据检索的一大飞跃。
- [**与 Embed 专家面对面**](https://discord.com/events/954421988141711382/1298319720868745246)：Cohere 正与其 Embed 高级产品经理举行答疑会，以提供对新功能的见解。参加活动，直接从源头获取内部消息。

**主题 5. 黑客松热潮：伯克利提供超过 20 万美元奖金**

- [**LLM Agents MOOC 黑客松提供 20 万美元重奖**](https://docs.google.com/forms/d/e/1FAIpQLSevYR6VaYK5FkilTKwwlsnzsn8yI_rRLLqDZj0NH7ZL_sCs_g/viewform)：伯克利 RDI 发起了一场黑客松，奖金超过 **200,000 美元**，从 10 月中旬持续到 12 月中旬。该活动向所有人开放，设有应用、基准测试等赛道。
- **OpenAI 和 GoogleAI 全力支持黑客松**：**OpenAI** 和 **GoogleAI** 等主要赞助商支持此次活动，增加了声望和资源。参与者还可以在比赛期间探索职业和实习机会。
- **五大赛道，无限可能**：黑客松包括 **Applications**、**Benchmarks**、**Fundamentals**、**Safety** 以及 **Decentralized & Multi-Agents** 等赛道，邀请参与者突破 AI 边界并释放创新。

---

# 第一部分：高层级 Discord 摘要

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **AI DJ 软件展示潜力**：用户讨论了一个创新的 **AI DJ Software** 概念，该软件可以像 **Spotify** 那样实现自动歌曲过渡和混音。
  - 提到了 [rave.dj](https://rave.dj) 等工具用于创建有趣的混搭（mashups），尽管输出结果尚不完美。
- **Hugging Face 模型查询引发安全担忧**：一位用户寻求关于如何通过 `huggingface_hub` 安全下载 Hugging Face 模型权重而不泄露权重的建议。
  - 社区成员提供了关于使用环境变量进行身份验证以维护隐私的见解。
- **OCR 工具受到关注**：讨论了从 PDF 中提取结构化数据的有效 **OCR 解决方案**，特别是针对建筑领域的应用。
  - 推荐包括 Koboldcpp 等模型，以提高文本提取的准确性。
- **Granite 3.0 模型发布备受瞩目**：新的 [端侧 Granite 3.0 模型](https://huggingface.co/spaces/Tonic/on-device-granite-3.0-1b-a400m-instruct) 引起了用户的兴奋，突显了其便捷的部署特性。
  - 该模型的属性被赞誉为增强了快速集成的可用性。
- **LLM 最佳实践网络研讨会吸引关注**：一位 META 高级 ML 工程师宣布了一个专注于 LLM 导航的 [网络研讨会](https://shorturl.at/ulrCN)，目前已有近 **200 人报名**。
  - 该会议承诺提供关于 Prompt Engineering 和模型选择的可操作见解。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Claude 3.5 Sonnet 展示了令人印象深刻的基准测试结果**：新发布的 **Claude 3.5 Sonnet** 在基准测试中取得了显著提升，且用户无需更改代码。更多信息请参见官方公告 [此处](https://www.anthropic.com/news/3-5-models-and-computer-use)。
  - 成员指出，通过悬停在供应商旁边的信息图标可以轻松跟踪升级，从而增强了用户体验。
- **Llama 3.1 Nitro 带来闪电般的性能提升**：随着 **70%** 的速度提升，**Llama 3.1 405b Nitro** 现已上线，承诺吞吐量约为 **120 tps**。查看新端点：[405b](https://openrouter.ai/meta-llama/llama-3.1-405b-instruct:nitro) 和 [70b](https://openrouter.ai/meta-llama/llama-3.1-70b-instruct:nitro)。
  - 用户被该模型带来的性能优势所吸引，使其成为一个极具吸引力的选择。
- **Ministral 强大的模型阵容**：**Ministral 8b** 已推出，在 **128k** 上下文下达到 **150 tps**，目前在技术提示词（tech prompts）中排名 **#4**。经济型的 **3b 模型** 可在 [此处](https://openrouter.ai/mistralai/ministral-8b) 访问。
  - 这些模型的性能和定价在用户中引起了极大的兴奋，满足了不同的预算需求。
- **Grok Beta 扩展功能**：**Grok Beta** 现在支持增加到 **131,072** 的上下文长度，费用为 **$15/m**，取代了旧的 `x-ai/grok-2` 请求。这一更新受到了期待增强性能的用户们的热烈欢迎。
  - 社区讨论反映了对新定价模式下功能改进的期望。
- **社区对 Claude 自我审查端点的反馈**：发起了一项投票，以收集对 **Claude 自我审查（self-moderated）** 端点的意见，该端点目前在排行榜上排名第一。成员可以点击 [此处](https://discord.com/channels/1091220969173028894/1107397803266818229/1298353935500836957) 参与投票。
  - 用户参与表明他们对影响这些端点的开发和用户体验有着浓厚的兴趣。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Claude 3.5 Sonnet 统治基准测试**：升级后的 **Claude 3.5 Sonnet** 在 Aider 的排行榜上获得了 **84.2%** 的分数，在架构师模式（architect mode）下与 DeepSeek 配合使用时达到了 **85.7%**。
  
  - 该模型不仅增强了编码任务，还保留了之前的价格结构，令许多用户感到兴奋。
- **DeepSeek 是高性价比的编辑器替代方案**：DeepSeek 的成本为 **每 1M 输出 token $0.28**，与价格为 **$15** 的 Sonnet 相比，是一个更便宜的选择。
  
  - 用户注意到它与 Sonnet 配合良好，尽管关于 token 成本变化影响性能的讨论也随之出现。
- **Aider 配置文件需要明确说明**：用户询问了如何设置 `.aider.conf.yml` 文件，指定如 `openrouter/anthropic/claude-3.5-sonnet:beta` 之类的类型作为编辑器模型。
  
  - 用户寻求关于 Aider 在运行时从何处提取配置细节的澄清，以实现最佳设置。
- **令人兴奋的 computer use 测试版公告**：Anthropic 新的 **computer use** 功能允许 Claude 执行移动光标等任务，目前处于公开测试阶段，被描述为实验性的。
  
  - 开发者可以引导其功能，这标志着与 AI 交互方式的转变，并提高了编码环境中的可用性。
- **DreamCut AI - 新颖的视频编辑解决方案**：[DreamCut AI](http://dreamcut.ai) 已发布，允许用户利用 Claude AI 进行视频编辑，由 MengTo 历时 **3 个月** 编写 **5 万行代码** 开发而成。
  
  - 目前处于早期访问阶段，用户可以通过免费账号体验其 AI 驱动的功能。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Diffusion 3.5 发布震惊用户**：**Stable Diffusion 3.5** 发布，包含适用于消费级硬件的可定制模型，根据 [Stability AI Community License](https://stability.ai/community-license-agreement) 提供。用户对 **3.5 Large** 和 **Turbo** 模型感到兴奋，这些模型已在 [Hugging Face](https://huggingface.co/stabilityai) 和 [GitHub](https://github.com/Stability-AI/sd3.5) 上线，**3.5 Medium** 将于 **10 月 29 日** 发布。
  
  - 这一公告让许多人措手不及，引发了关于其意外发布以及预期比之前版本性能提升的讨论。
- **SD3.5 与 Flux 图像质量大对决**：社区评估了 **SD3.5** 在图像质量方面是否能击败 **Flux**，重点关注微调和美学。初步印象表明 **Flux** 在这些领域可能仍具有优势，引发了对数据集有效性的好奇。
  
  - 讨论强调了模型之间基准测试比较的重要性，特别是在建立图像生成的市场标准时。
- **新的许可细节引发疑问**：参与者对 **SD3.5** 的许可模式表示担忧，特别是与 **AuraFlow** 相比在商业环境下的应用。在可访问性与 **Stability AI** 的盈利需求之间取得平衡成为了热门话题。
  
  - 这一讨论凸显了确保模型既对开发者开放，又能让生产者可持续发展的挑战。
- **社区支持促进技术采用**：发现 **Automatic1111's Web UI** 问题的用户在支持频道获得了指导，体现了社区内的协作精神。一位成员迅速获得了直接帮助，展示了与新用户的积极互动。
  
  - 这种主动的支持方式有助于确保用户能够有效地利用新模型和可用的集成工具。
- **LoRA 应用激发艺术家热情**：为 **SD3.5** 引入的 **LoRA** 模型让用户开始尝试提示词并分享结果，展示了其在增强图像生成方面的有效性。社区一直积极展示他们的作品并鼓励进一步的实验。
  
  - 这些举措反映了旨在最大限度提高 AI 艺术社区内新发布功能影响力的参与策略。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Nightly Transformers 修复了梯度累积 Bug**：最近的更新显示，[梯度累积（gradient accumulation）Bug](https://www.reddit.com/r/MachineLearning/comments/1g8ymrn/r_gradient_accumulation_bug_fix_in_nightly/) 已被修复，并将包含在 nightly transformers 和 Unsloth 训练器中，纠正了损失曲线计算中的不准确之处。
  
  - *此修复增强了各种训练设置中性能指标的可靠性。*
- **关于 LLM 训练效率的见解**：成员们讨论了使用短语输入训练 LLM 会生成多个子样本，从而最大限度地提高训练效果并使模型能够高效学习。
  
  - *这种方法允许更丰富的训练数据集，从而提升模型能力。*
- **模型性能与基准测试的挑战**：针对新的 **Nvidia Nemotron Llama 3.1 模型** 出现了一些担忧，尽管其基准测试分数与 Llama 70B 相似，但对其性能是否优于后者表示怀疑。
  
  - *Nvidia 基准测试的不一致性引发了对其模型性能评估的质疑。*
- **创建研究生申请文书编辑器**：一位成员寻求开发 **研究生申请文书编辑器** 的帮助，在实现 AI 模型时面临复杂 Prompt 导致输出过于平庸的挑战。
  
  - *专家被召集来提供微调模型的策略，以增强输出的相关性。*
- **在 CSV 数据上微调 LLaMA**：根据 [Turing 文章](https://www.turing.com/resources/understanding-llm-evaluation-and-benchmarks) 中分享的方法，请求澄清如何使用 CSV 数据微调 **LLaMA 模型** 以处理特定的事件查询。
  
  - *社区反馈在制定有效模型测试方法方面发挥了关键作用。*

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **LLM 中的灾难性遗忘**：讨论集中在大型语言模型（LLM）在持续指令微调过程中的 **灾难性遗忘（catastrophic forgetting）**，特别是在 **1B 到 7B** 参数规模的模型中。成员们指出，微调可能会显著降低性能，详见 [这项研究](https://arxiv.org/abs/2308.08747)。
  
  - 参与者分享了将自己的模型与成熟模型进行基准测试对比的个人经验，揭示了 LLM 训练中固有的挑战。
- **关于 LLM 基准测试性能的见解**：用户指出模型规模显著影响性能，并提到如果没有适当的优化，数据限制会导致较差的结果。一位参与者讨论了他们的 **1B 模型** 相对于 **Meta 模型** 较低的分数，强调了基准对比的重要性。
  
  - 这引发了关于某些模型在缺乏足够训练资源的情况下，如何在竞争环境中表现不佳的进一步思考。
- **对研究论文可靠性的担忧**：最近的一项研究显示，大约 **1/7 的研究论文** 存在严重错误，削弱了其可信度。这引发了关于误导性研究可能导致研究人员无意中建立在错误结论之上的讨论。
  
  - 成员们指出，评估研究完整性的传统方法需要更多资金和关注来纠正这些问题。
- **模型微调：一把双刃剑**：围绕微调大型基础模型（foundation models）有效性的辩论强调了为特定目标而降低广泛能力的风险。成员们推测，微调需要细致的超参数优化才能获得丰硕成果。
  
  - 针对社区缺乏关于微调最佳实践的既定知识的担忧浮现，引发了对自去年以来最新进展的疑问。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio v0.3.5 功能亮点**：**LM Studio v0.3.5** 的更新引入了 **headless mode** 和**按需模型加载 (on-demand model loading)**，优化了本地 LLM 服务功能。
  
  - 用户现在可以使用 CLI 命令 `lms get` 轻松下载模型，增强了模型获取的便利性和可用性。
- **GPU Offloading 性能大幅下降**：一位用户发现，在最近的更新后，GPU Offloading 性能骤降，仅利用了 **4.2GB**，而非预期的 **15GB**。
  
  - 恢复到旧版本的 ROCm 运行时 (runtime) 版本后恢复了正常性能，这表明更新可能改变了 GPU 的利用方式。
- **出现模型加载错误**：一位用户报告了与 GPU Offload 设置调整相关的 “Model loading aborted due to insufficient system resources” 错误。
  
  - 禁用加载防护机制 (loading guardrails) 被提作为一种变通方法，尽管通常不推荐这样做。
- **讨论 AI 模型性能指标**：社区就衡量性能进行了详细讨论，强调了加载设置对吞吐量 (throughput) 和延迟 (latency) 的影响。
  
  - 值得注意的是，在重度 GPU Offloading 下，吞吐量降至 **0.9t/s**，表明可能存在效率低下的问题。
- **咨询游戏图像增强工具**：用户开始探索将游戏图像转换为写实艺术的选项，**Stable Diffusion** 被列为候选工具。
  
  - 对话引发了关于各种图像增强器在转换游戏视觉效果方面有效性的关注。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Anthropic 发布 Claude 3.5**：Anthropic 推出了升级版的 **Claude 3.5 Sonnet** 和 **Claude 3.5 Haiku** 模型，并加入了一项新的 Beta 功能 **computer use**，允许模型像人类一样与计算机交互。
  
  - 尽管具有创新能力，但有用户报告它不能有效地遵循 Prompt，导致用户体验参差不齐。
- **Mochi 1 重新定义视频生成**：**GenmoAI** 推出了 **Mochi 1**，这是一个旨在实现高质量视频生成的开源模型，以其在 480p 分辨率下的逼真动态和 Prompt 遵循能力而著称。
  
  - 该项目利用大量资金进行进一步开发，旨在为写实视频生成设定新标准。
- **CrewAI 完成 1800 万美元 A 轮融资**：**CrewAI** 在由 Insight Partners 领投的 A 轮融资中筹集了 **1800 万美元**，专注于利用其开源框架实现企业流程自动化。
  
  - 该公司声称每月执行超过 **1000 万个 Agent**，服务于很大一部分财富 500 强公司。
- **Stable Diffusion 3.5 上线**：**Stability AI** 发布了 **Stable Diffusion 3.5**，这是一个高度可定制的模型，可在消费级硬件上运行，并免费用于商业用途。
  
  - 用户现在可以通过 [Hugging Face](https://huggingface.co/) 访问它，预计未来还会推出更多变体。
- **Outlines 库的 Rust 移植版提升效率**：Dottxtai 宣布了 **Outlines** 库的 **Rust 移植版**，它为结构化生成 (structured generation) 任务提供了更快的编译速度和轻量化设计。
  
  - 此次更新显著提升了开发者的效率，并包含多种编程语言的绑定。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 中的语言混淆**：用户报告称，尽管提供了英文文档，NotebookLM 的回答仍默认为荷兰语，建议调整 [Google 账户语言设置](https://myaccount.google.com/language)。一位用户在德语输出方面遇到困难，遇到了意想不到的“外星”方言。
  
  - 这突显了 NotebookLM 目前在语言处理方面的局限性以及潜在的改进方向。
- **共享 Notebooks 的挫败感**：几位成员在尝试共享 notebook 时遇到问题，面临永久的“Loading...”屏幕，导致协作失效。这引发了对该工具稳定性和可靠性的担忧。
  
  - 用户正在敦促解决此问题，表明迫切需要一个强大的共享功能来促进团队合作。
- **多语言音频效果参差不齐**：尝试创建各种语言的音频概览（audio overviews）结果不尽如人意，特别是在荷兰语方面，发音和母语般的质量明显不足。一些用户成功生成了荷兰语内容，为改进带来了希望。
  
  - 这一讨论揭示了社区对增强多语言能力以实现更广泛可用性的浓厚兴趣。
- **使用 NotebookLM 的播客体验**：一位用户兴奋地分享了他们成功上传了一个 90 页的区块链课程，并生成了有趣的音频。反馈表明，输入的变化会导致意想不到且具有娱乐性的输出。
  
  - 这展示了 NotebookLM 在播客领域的多种应用，尽管一致的质量仍是需要增强的话题。
- **文档上传问题依然存在**：用户面临文档无法在 Google Drive 中显示的问题，以及处理延迟，引发了关于潜在文件损坏的讨论。建议通过刷新操作来解决这些上传挑战。
  
  - 这些技术障碍强调了 NotebookLM 内部需要可靠的文档管理功能。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Claude 3.5 模型引发热议**：用户热切讨论新的 **Claude 3.5 Sonnet** 和 **Claude 3.5 Haiku**，希望在 [AnthropicAI 的公告](https://x.com/AnthropicAI/status/1848742740420341988)之后，它们能迅速集成到 Perplexity 中。关键特性包括 Claude 能够像人类一样使用计算机。
  
  - 这种兴奋情绪反映了之前的发布情况，并表明了对 AI 不断进化的能力的浓厚兴趣。
- **API 功能引发挫败感**：用户担心 Perplexity API 在收到提示时无法返回来源的完整 URL，导致用户对其易用性感到困惑。一位特定用户表达了尽管按照说明操作，但在获取这些 URL 方面仍面临挑战。
  
  - 这一问题引发了关于 AI 产品中 API 能力以及需要更清晰文档的更大范围讨论。
- **Perplexity 面临竞争挑战**：随着 **Yahoo** 推出 AI 聊天服务，围绕 Perplexity 竞争优势的讨论变得普遍。然而，用户强调 **Perplexity** 的可靠性和资源丰富度是其优于竞争对手的关键优势。
  
  - 尽管竞争加剧，但对质量和性能的承诺仍然是用户的基石。
- **用户反馈突显优势**：多位用户对 Perplexity 的表现给予了积极评价，称赞其高质量的信息传递。一位用户强调了满意度，表示：*“我超级喜欢 PAI！我在工作和生活中一直在用它。”*
  
  - 此类反馈突显了该平台在 AI 社区中的声誉。
- **增强事实核查的资源共享**：关于 **AI 驱动的事实核查** 策略的合集强调了伦理考量以及 LLM 在 [Perplexity](https://www.perplexity.ai/collections/advanced-ai-driven-fact-checki-a3cMcPR.QsKkCRZ79UKFLQ) 误导信息管理中的作用。该资源讨论了来源可信度和偏见检测的重要性。
  
  - 分享此类资源反映了社区在提高信息传播准确性方面的积极努力。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **新的开源 SAE 解释流水线发布**：可解释性团队发布了一个新的开源 [pipeline](https://github.com/EleutherAI/sae-auto-interp)，用于自动解释 LLMs 中的 **SAE features** 和神经元，该流水线引入了五种评估解释质量的技术。
  
  - 这一举措有望大规模增强可解释性，展示了利用 LLMs 进行特征解释方面的进展。
- **集成国际象棋 AI 与 LLMs 以增强交互性**：一项将国际象棋 AI 与 LLM 结合的提案旨在创建一个能够理解自身决策的对话式 Agent，从而提升用户参与度。
  
  - 构想中的模型力求实现连贯的对话，使 AI 能够清晰地阐述其国际象棋走法背后的推理。
- **SAE 研究思路引发讨论**：一名本科生寻求关于 **Sparse Autoencoders (SAEs)** 的项目思路，引发了关于当前研究工作和协作机会的讨论。
  
  - 成员们分享了资源，包括一篇用于深入探索的 [Alignment Forum 帖子](https://www.alignmentforum.org/posts/CkFBMG6A9ytkiXBDM/sparse-autoencoders-future-work)。
- **Woog09 为 ICLR 2025 的 Mech Interp 论文评分**：一名成员分享了一份 [电子表格](https://docs.google.com/spreadsheets/d/1TTHbONFo4OV35Bv0KfEFllnkP-aLGrr_fmzwfdBqBY0/edit?gid=0#gid=0)，对 ICLR 2025 的所有机械解释性（mechanistic interpretability）论文进行了评分，采用 1-3 分的质量量表。
  
  - 他们的重点是提供经过校准的评分，以引导读者阅读提交的论文。
- **调试 Batch Size 配置**：成员们讨论了在设置 `batch_size` 后 `requests` 无法正确进行批处理的问题，强调了在模型层面处理此配置的必要性。
  
  - 针对指定 `batch_size` 的目的产生了困惑，随后有人澄清了其与模型初始化的联系。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Allegro 模型实现文生视频**：Rhymes AI 发布了其新的开源模型 **Allegro**，能以 15 FPS 和 720p 分辨率从文本生成 6 秒的视频，并提供了包括 [GitHub 仓库](https://github.com/rhymes-ai/Allegro) 在内的探索链接。用户可以加入 [Discord 等候名单](https://forms.gle/JhA7BaKvZoeJYQU87) 以获取早期访问权限。
  
  - *这一创新为内容创作打开了新大门，* 既引人入胜又易于获取。
- **Stability AI 凭借 SD 3.5 升温**：**Stability AI** 推出了 **Stable Diffusion 3.5**，为年收入低于 100 万美元的企业提供三个免费商用版本，并增强了如 Query-Key Normalization 等优化功能。Large 版本现已在 Hugging Face 和 GitHub 上线，Medium 版本定于 10 月 29 日发布。
  
  - 该模型标志着一次实质性的升级，因其独特的功能吸引了社区的极大关注。
- **Claude 3.5 Haiku 在编程领域树立高标准**：Anthropic 推出了 **Claude 3.5 Haiku**，在编程任务中超越了 Claude 3 Opus，在 SWE-bench Verified 上得分为 **40.6%**，可通过 [此处](https://docs.anthropic.com/en/docs/build-with-claude/computer-use) 的 API 获取。用户对各种 Benchmark 中突出的进步印象深刻。
  
  - *该模型的性能正在重塑标准*，使其成为编程相关任务的首选。
- **Factor 64 的启示**：一位成员对涉及 **Factor 64** 的突破感到兴奋，事后看来这似乎是“显而易见”的。这一时刻引发了关于其影响的更深层次讨论。
  
  - *这一发现激发了进一步的参与*，暗示了后续的协作或新探索。
- **Hackernews 社区反馈的疏离感**：关于 **Hackernews** 沦为 **流量彩票（views lottery）** 的担忧表明，讨论缺乏实质内容，更多是噪音而非真正的反馈。成员们将其描述为**非常嘈杂且带有偏见**，质疑其参与价值。
  
  - *该平台被越来越多的人认为效率低下*，从而引发了关于替代反馈机制的对话。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Unsloth 讲座上线**：[Unsloth 演讲](https://www.youtube.com/watch?v=hfb_AIhDYnA)现已发布，展示了极高的信息密度，许多观众对其紧凑的节奏表示赞赏。
  
  - *“我正以 0.5 倍速回看，但感觉还是很快”*，这反映了讲座的深度。
- **梯度累积（Gradient Accumulation）见解**：关于梯度累积的讨论强调了 batch 间重缩放以及对大梯度使用 **fp32** 的重要性。
  
  - *“通常所有 batch 无法保持相同大小是有原因的”*，强调了训练的复杂性。
- **GitHub AI 项目揭晓**：一位用户分享了他们的 [GitHub 项目](https://github.com/shaRk-033/ai.c)，其特点是用 **纯 C 语言实现 GPT**，激发了关于深度学习的讨论。
  
  - 该计划旨在通过易于理解的实现来增强对深度学习的理解。
- **解码 Torch Compile 输出**：来自 `torch.compile` 的指标显示了矩阵乘法的执行时间，从而澄清了如何解释 `SingleProcess AUTOTUNE` 的结果。
  
  - *SingleProcess AUTOTUNE 完成耗时 30.7940 秒*，引发了关于运行时分析（runtime profiling）的深入讨论。
- **Meta 的 HOTI 2024 聚焦生成式 AI**：分享了来自 **Meta HOTI 2024** 的见解，[此环节](https://www.youtube.com/watch?v=zk9XFw1s99M&list=PLBM5Lly_T4yRMjnHZHCXtlz-_AKZWW1xz&index=15)讨论了具体问题。
  
  - 关于“驱动 Llama 3”的主旨演讲揭示了对于理解 Llama 3 集成至关重要的基础设施见解。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AGI 辩论升温**：成员们讨论了我们实现 **AGI** 的困境是否源于所提供的数据类型，一些人认为 **二进制数据** 可能会限制进展。
  
  - 一位成员断言，改进的算法可以使 **AGI** 无论数据类型如何都能实现。
- **澄清 GPT 术语**：术语“GPTs”引起了混淆，因为它通常指代 **自定义 GPTs**，而不是涵盖像 **ChatGPT** 这样的模型。
  
  - 参与者强调了区分通用 **GPTs** 及其具体实现的重要性。
- **量子计算模拟器见解**：一位成员指出，有效的 **量子计算模拟器** 应该产生与真实量子计算机 **1:1** 的输出，尽管其有效性仍存争议。
  
  - 各家公司都在开发模拟器，但它们的实际应用仍在讨论中。
- **Anthropic 的 TANGO 模型引发关注**：**TANGO 说话头像模型** 因其唇形同步能力和开源潜力而受到关注，成员们渴望探索其功能。
  
  - 讨论包括 **Claude 3.5 Sonnet** 对阵 **Gemini Flash 2.0** 的性能，关于谁更占优势意见不一。
- **ChatGPT 在电视节目方面表现不佳**：一位成员分享了 **ChatGPT** 错误识别电视节目剧集标题和编号的挫败感，指出训练数据中存在缺口。
  
  - 对话强调了数据中的观点可能会如何使娱乐相关查询的结果产生偏差。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 模型受到青睐**：成员们讨论了在 playground 中积极使用 **Cohere models** 的情况，强调了它们的多样化应用和探索尝试。一位成员特别强调，在探索 **multi-modal embeddings** 时，需要使用不同模型重新运行推理。
  
  - 这引发了人们对这些模型在现实场景中广泛能力的关注。
- **Multimodal Embed 3 发布！**：**Embed 3** 模型发布，在检索任务上具有 SOTA 性能，支持混合模态和多语言搜索，允许文本和图像数据共同存储。更多详情请参阅 [博客文章](https://cohere.com/blog/multimodal-embed-3) 和 [发布说明](https://docs.cohere.com/changelog/embed-v3-is-multimodal)。
  
  - 该模型将成为创建统一数据检索系统的游戏规则改变者。
- **微调 LLM 需要更多数据**：针对使用极小数据集进行 **fine-tuning** LLM 的担忧被提出，重点在于潜在的过拟合问题。建议的策略包括扩大数据集规模和调整超参数，并参考了 [Cohere 微调指南](https://cohere.com/llmu/fine-tuning-for-chat)。
  
  - 成员们在面临挑战时，寻求通过有效的调整来优化模型性能。
- **多语言模型遭遇延迟峰值**：多语言 embed 模型报告了 **30-60s** 的延迟问题，在 **15:05 CEST** 左右飙升至 **90-120s**。用户注意到情况有所改善，并敦促报告持续存在的故障。
  
  - 延迟问题凸显了进行进一步技术评估以确保最佳性能的必要性。
- **Agentic Builder Day 宣布**：Cohere 和 OpenSesame 将于 11 月 23 日共同举办 **Agentic Builder Day**，邀请才华横溢的开发者使用 Cohere 模型创建 AI Agent。参与者可以申请参加这场为期 **8 小时的黑客松**，并有机会赢取奖品。
  
  - 该竞赛鼓励渴望为有影响力的 AI 项目做出贡献的开发者进行协作，申请链接见 [此处](https://www.opensesame.dev/hack)。

 

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 引入自定义数组结构 (SoA)**：你可以使用 Mojo 的语法构建自己的 **Structure of Arrays (SoA)**，尽管它尚未原生集成到语言中。
  
  - 虽然目前可以使用 **slice type**，但用户发现它有一定的局限性，预计 Mojo 不断演进的类型系统会有所改进。
- **Mojo 的 Slice 类型需要改进**：虽然 Mojo 包含 slice 类型，但它基本上仅限于作为标准库结构体，只有部分方法返回 slice。
  
  - 随着 Mojo 的进一步发展，成员们期待重新审视这些 slice 功能。
- **二进制文件剥离显示出显著的体积减小**：剥离一个 **300KB 的二进制文件** 可以显著减小到仅 **80KB**，这表明了强大的优化可能性。
  
  - 成员们指出，这种*显著的下降*对于未来的二进制管理策略是令人鼓舞的。
- **Comptime 变量导致编译错误**：有用户报告在 `@parameter` 作用域之外使用 `comptime var` 会触发编译错误。
  
  - 讨论强调，虽然 **alias** 允许编译时声明，但实现直接的可变性仍然很复杂。
- **Node.js 与 Mojo 在 BigInt 计算中的对比**：对比显示 Node.js 中的 BigInt 操作耗时 **40 秒**，这表明 Mojo 可能会更好地优化这一过程。
  
  - 成员们指出，完善任意宽度整数库是提升性能基准的关键。

 

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **LLVM Renderer 重构提案**：一位用户提议使用模式匹配风格重写 **LLVM renderer** 以增强功能，这可能会提高清晰度和效率。
  
  - *这种方法旨在简化开发并使集成更容易。*
- **提升 Tinygrad 的速度**：讨论强调了在过渡到使用 uops 后增强 **Tinygrad 性能** 的要求，这对于紧跟计算进步至关重要。
  
  - *建议通过优化算法和减少开销来努力实现这些速度目标。*
- **将梯度裁剪集成到 Tinygrad**：社区讨论了 `clip_grad_norm_` 是否应该成为 Tinygrad 的标准，这是深度学习框架中常见的方法。
  
  - George Hotz 指出，在进行此集成之前必须先进行梯度重构才能生效。
- **Action Chunking Transformers 的进展**：一位用户报告了 **ACT 训练** 的收敛情况，在几百步后实现了低于 **3.0** 的 loss，并附带了 [源代码](https://github.com/mdaiter/act-tinygrad) 和相关研究的链接。
  
  - *这一进展表明，基于当前模型性能，仍有进一步优化的潜力。*
- **探索使用 .where() 进行张量索引**：围绕对布尔张量使用 `.where()` 函数展开了讨论，揭示了使用 `.int()` 索引时的非常规结果。
  
  - *这引发了关于不同场景下张量操作预期行为的询问。*

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Hume AI 加入**：一位成员宣布在 **phidatahq** 通用 Agent 中添加了 **Hume AI** 语音助手，通过精简的 UI 以及在 Mac 上创建和执行 AppleScripts 的能力增强了功能。
  
  - *“非常喜欢新的 @phidatahq UI”*，指出了通过此次集成实现的改进。
- **Claude 3.5 Sonnet 开启实验性功能**：Anthropic 正式发布了具有计算机使用（computer usage）公开 beta 测试权限的 **Claude 3.5 Sonnet** 模型，尽管它被描述为仍处于实验阶段且容易出错。
  
  - 成员们表达了兴奋之情，同时指出此类进步强化了 AI 模型日益增长的能力。更多详情请参阅 [Anthropic 的推文](https://x.com/AnthropicAI/status/1848742747626226146)。
- **Open Interpreter 借力 Claude 升级**：人们对使用 **Claude** 增强 **Open Interpreter** 充满热情，成员们讨论了运行新模型的实际实现和代码。
  
  - 一位成员报告了使用特定模型命令的成功经验，鼓励其他人尝试。
- **Screenpipe 受到关注**：成员们称赞 **Screenpipe** 工具在构建日志中的实用性，注意到其有趣的落地页以及社区贡献的潜力。
  
  - 一位成员鼓励更多地参与该工具，并引用了 [GitHub](https://github.com/OpenInterpreter/open-interpreter/blob/development/examples/screenpipe.ipynb) 上链接的一个有用配置文件。
- **货币化与开源的结合**：围绕通过允许用户从源码构建或为预构建版本付费来使公司盈利的讨论展开，以平衡贡献和使用。
  
  - 成员们对这种模式表示赞同，强调了开发者贡献和付费用户共同带来的好处。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **新版本即将到来**：一位成员对创建 **新版本** 而不是修改现有版本表示兴奋，计划在周一进行直播。
  
  - 这种热情得到了共鸣，社区成员纷纷响应即将举行的会议，届时还将涵盖当前的功能。
- **DSPy 文档面临问题**：成员们哀叹新的文档结构中缺少了 **小 AI 助手**，这导致了广泛的失望。
  
  - 聊天中反映了社区的情绪，强调了失去这一重要功能是一种损失。
- **死链警报**：报告了 DSPy 文档中导致 404 错误的大量 **死链**，引起了用户的不满。
  
  - 至少有一位用户迅速采取行动通过 PR 修复了此问题，赢得了同行的感谢。
- **文档机器人回归**：随着 **文档机器人** 的回归，社区爆发了庆祝活动，恢复了用户非常欣赏的功能。
  
  - 聊天中充满了由衷的表情符号和肯定，展示了社区对该机器人重要存在的宽慰和支持。
- **征求对 3.0 版本的看法**：一位成员询问了即将发布的 3.0 版本的 **总体氛围**，表现出对社区反馈的渴望。
  
  - 然而，回应寥寥无几，使得集体的感受笼罩在不确定性之中。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **VividNode：在桌面端与 AI 模型聊天**：**VividNode** 应用允许桌面用户与 **GPT**、**Claude**、**Gemini** 和 **Llama** 进行聊天，具有高级设置以及使用 **DALL-E 3** 或各种 Replicate 模型生成图像的功能。更多详情请参阅[公告](https://twitter.com/llama_index/status/1848484047607239041)。
  
  - *该应用程序简化了与 AI 的沟通*，为用户提供了一个强大的聊天界面。
- **用 9 行代码构建 Serverless RAG 应用**：一个教程演示了仅用 **9 行代码** 即可使用 **LlamaIndex** 部署 Serverless **RAG 应用**，使其成为相比 **AWS Lambda** 更具成本效益的解决方案。更多见解请参考此[推文](https://twitter.com/llama_index/status/1848509130631151646)。
  
  - *易于部署和成本效益是开发者使用此方法的关键亮点*。
- **通过知识管理增强 RFP 响应**：讨论集中在利用向量数据库索引文档以增强 **RFP（征求建议书）响应生成**，从而实现超越简单聊天回复的高级工作流。关于该主题的更多内容可以在此[帖子](https://twitter.com/llama_index/status/1848759935787803091)中找到。
  
  - *这种方法强化了向量数据库在支持复杂 AI 功能中的作用*。
- **加入 Llama Impact 黑客松！**：在旧金山举办的 **Llama Impact Hackathon** 为参与者提供了一个使用 **Llama 3.2** 模型构建解决方案的平台，设有 **15,000 美元的奖金池**，其中包括为最佳使用 **LlamaIndex** 提供的 **1,000 美元奖金**。活动详情可见此[公告](https://twitter.com/llama_index/status/1848807401971192041)。
  
  - 黑客松将于 **11 月 8 日至 10 日**举行，同时支持线下和线上参与者。
- **CondensePlusContextChatEngine 自动初始化记忆**：讨论澄清了 **CondensePlusContextChatEngine** 现在会自动为连续问题初始化记忆，从而改善用户体验。之前的版本行为不同，导致了一些用户的困惑。
  
  - *这一变化简化了持续聊天中的记忆管理*，增强了用户交互。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **LLM Agents MOOC 黑客松启动**：Berkeley RDI 将于 10 月中旬至 12 月中旬启动 **LLM Agents MOOC Hackathon**，奖金总额**超过 200,000 美元**。参与者可以通过[注册链接](https://docs.google.com/forms/d/e/1FAIpQLSevYR6VaYK5FkilTKwwlsnzsn8yI_rRLLqDZj0NH7ZL_sCs_g/viewform)报名。
  
  - 该黑客松设有五个赛道，旨在吸引 **Berkeley** 学生和公众参与，并得到了 **OpenAI** 和 **GoogleAI** 等主要赞助商的支持。
- **TapeAgents 框架介绍**：ServiceNow 新推出的 **TapeAgents 框架** 通过结构化日志记录促进了 Agent 的优化和开发。该框架增强了控制力，能够实现[论文](https://www.servicenow.com/research/TapeAgentsFramework.pdf)中详述的逐步调试。
  
  - 该工具提供了关于 Agent 性能的有价值见解，强调了如何记录每次交互以进行全面分析。
- **LLM 中的 Function Calling 详解**：有一场关于 LLM 如何处理将任务拆分为 **Function Calling** 的讨论，强调了对代码示例的需求。澄清说明了理解这一机制对未来发展的重要性。
  
  - 成员们探讨了架构选择对 Agent 能力的影响，同时研究了这些方法如何提高功能性。
- **关于企业级 AI 的讲座见解**：Nicolas Chapados 在第 7 讲中讨论了**企业级生成式 AI** 的进展，强调了像 **TapeAgents** 这样的框架。会议回顾了在 AI 应用中集成安全性和可靠性的重要性。
  
  - Chapados 和客座演讲者的关键见解强调了实际应用以及 AI 改变企业工作流的潜力。
- **模型蒸馏技术与资源**：成员们分享了关于 [AI Agentic Design Patterns with Autogen](https://learn.deeplearning.ai/courses/ai-agentic-design-patterns-with-autogen/lesson/1/introduction) 的课程，提供了学习模型蒸馏和 Agent 框架的资源。该课程为掌握 Autogen 技术提供了结构化的方法。
  
  - 此外，还讨论了一个有用的 [GitHub 仓库](https://github.com/ServiceNow/TapeAgents)，以及一个探讨 TapeAgents 框架的精彩[线程](https://threadreaderapp.com/thread/1846611633323291055.html)。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **PyTorch Core 出现警告**：一位用户报告称 **PyTorch** 中的一个 **warning** 现在会在 **float16** 上触发，但在 **float32** 上不会，建议使用不同的 kernel 进行测试以评估性能影响。有人推测 [PyTorch 源代码](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/MHA.cpp#L677) 中的特定行可能会影响 JIT 行为。
  
  - 社区预计解决此问题可能会带来显著的性能洞察。
- **分布式训练错误令人头疼**：一位用户在使用 `tune` 命令进行分布式训练并设置 **CUDA_VISIBLE_DEVICES** 时遇到了无报错信息的 **stop**。移除该设置后问题仍未解决，暗示存在更深层次的配置问题。
  
  - 这表明可能需要调查环境设置以查明根本原因。
- **对 Torchtune 配置文件产生困惑**：关于 **.yaml** 扩展名导致 **Torchtune** 误解本地配置的问题引发了困惑。强调了验证文件命名的重要性，以避免操作期间出现意外行为。
  
  - 参与者指出，微小的细节可能导致严重的运行时问题。
- **Flex 性能讨论升温**：围绕 **Flex** 在 **3090s** 和 **4090s** 上的成功运行展开了讨论，并提到了在 **A800s** 上优化的内存使用。对话涉及随着模型规模扩大，更快的 **out-of-memory** 操作。
  
  - 优化的内存管理被视为有效处理大型模型的关键。
- **训练硬件设置受到审查**：一位用户在讨论训练性能问题时确认使用了 **8x A800** GPU。社区讨论了通过减少 GPU 数量进行测试，作为有效排查持续错误的一种手段。
  
  - 对不同硬件设置的讨论突显了训练环境中扩展的细微差别。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Langchain Open Canvas 探索兼容性**：一位成员询问 **Langchain Open Canvas** 是否可以集成 **Anthropic** 和 **OpenAI** 之外的 LLM 提供商，反映了对更广泛兼容性的需求。
  
  - 这一询问表明社区对扩展该应用与各种工具的可用性有着浓厚兴趣。
- **使用 Langchain 的 Agent 编排能力**：关于 **Langchain** 促进 **OpenAI Swarm** 进行 Agent 编排的潜力引发了讨论，并询问是否需要自定义编程。
  
  - 这引发了关于支持编排功能的现有库的回应。
- **策划输出链重构策略**：一位用户正在考虑是重构其 **Langchain** 工作流，还是切换到 **LangGraph** 以增强复杂工具使用中的功能。
  
  - 当前设置的复杂性使得这一战略决策对于实现最佳性能至关重要。
- **Langchain 0.3.4 的安全疑虑**：一位用户标记了 **PyCharm** 关于 **Langchain 0.3.4** 依赖项的 **malicious** 警告，引发了对潜在安全风险的警报。
  
  - 他们寻求社区确认此警告是否为常见现象，担心这可能是误报。
- **寻求本地托管解决方案的建议**：在为企业应用寻求模型的 **local hosting** 过程中，一位用户正在探索使用 **Flask** 或 **FastAPI** 构建 **inference container**。
  
  - 他们的目标是通过发现社区内更好的解决方案来避免冗余。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **2.5.0 带来实验性 Triton FA 支持**：版本 **2.5.0** 为 **gfx1100** 引入了实验性的 **Triton Flash Attention (FA)** 支持，通过 `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` 激活，这在 **Navi31 GPU** 上导致了 **UserWarning**。
  
  - 正如 [GitHub issue](https://github.com/ROCm/aotriton/issues/16#issuecomment-2346675491) 中讨论的那样，该警告最初让用户感到困惑，以为它与 **Liger** 有关。
- **利用指令微调模型进行训练**：一位成员提议利用像 **llama-instruct** 这样的指令微调模型进行指令训练，并指出只要用户接受其先前的微调，这样做是有益的。
  
  - 他们强调了进行*实验*以发现最佳方法的必要性，可能会在训练中混合多种策略。
- **对灾难性遗忘（Catastrophic Forgetting）的担忧**：关于在领域特定指令数据或与通用数据混合之间进行选择以防止训练期间出现**灾难性遗忘**的担忧浮出水面。
  
  - 成员们讨论了训练的复杂性，并鼓励探索多种策略以找到最有效的方法。
- **预训练与指令微调之争**：讨论重点在于，是应该从基础模型开始对原始领域数据进行预训练，还是依赖指令微调模型进行微调。
  
  - 一位成员主张，如果有原始数据，最初应使用原始数据以提供更强大的基础。
- **从原始文本生成指令数据**：一位成员分享了他们使用 **GPT-4** 从原始文本生成指令数据的计划，同时也承认了可能产生的潜在偏差。
  
  - 这种方法旨在减少对人工生成的指令数据的依赖，同时意识到其局限性。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **对用于函数调用的微调模型感到兴奋**：一位用户在为 **function calling** 专门微调了一个模型并成功创建了自己的推理 API 后，对 **Gorilla 项目** 表达了极大的热情。
  
  - 他们寻求对自定义端点进行基准测试（benchmarking）的方法，并请求有关该过程的相关文档。
- **分享添加新模型的说明**：针对询问，一位成员引导用户查看一个 [README 文件](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#contributing)，该文件概述了如何在 **Gorilla** 生态系统内的排行榜中添加新模型。
  
  - 对于旨在为 **Gorilla 项目** 做出有效贡献的用户来说，这份文档非常有价值。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **参加关于 LLM 的免费网络研讨会**：一位来自 Meta 的高级 ML 工程师正在举办一场关于 **构建 LLM 的最佳实践** 的免费网络研讨会，目前已有近 **200 人报名**。点击[此处](https://shorturl.at/ulrCN)注册，获取关于高级 Prompt Engineering 技术、模型选择和项目规划的见解。
  
  - 与会者可以期待深入探讨针对现实场景量身定制的 LLM 实际应用，从而增强其部署策略。
- **关于 Prompt Engineering 的见解**：网络研讨会包括对优化模型性能至关重要的**高级 Prompt Engineering 技术**的讨论。参与者可以利用这些见解更有效地执行 LLM 项目。
  
  - 会议还将讨论性能优化方法，这对于成功部署 LLM 项目至关重要。
- **探索检索增强生成（RAG）**：**Retrieval-Augmented Generation (RAG)** 将是一个重点话题，展示它如何增强 LLM 解决方案的能力。微调策略也将是最大化模型效能的关键讨论点。
  
  - 本次会议旨在为工程师提供在项目中有效实施 RAG 所需的工具。
- **在 Analytics Vidhya 上发表文章**：网络研讨会参与者的优秀文章将发表在 **Analytics Vidhya 的博客空间**，从而提高他们的专业知名度。这为在数据科学社区内分享见解提供了一个极佳的平台。
  
  - 这种曝光可以显著增强他们贡献的影响力，并促进社区参与。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Mozilla 对 AI 访问挑战的见解**：Mozilla 发布了两份关键研究报告：‘[External Researcher Access to Closed Foundation Models](https://blog.mozilla.org/wp-content/blogs.dir/278/files/2024/10/External-researcher-access-to-closed-foundation-models.pdf)’ 和 ‘[Stopping Big Tech From Becoming Big AI](https://blog.mozilla.org/wp-content/blogs.dir/278/files/2024/10/Stopping-Big-Tech-from-Becoming-Big-AI.pdf)’，揭示了 AI 开发的控制权问题。
  
  - 这些报告强调了为创建更公平的 AI 生态系统而进行变革的必要性。
- **总结 AI 研究结果的博客文章**：欲了解更多深入见解，[此处的博客文章](https://discord.com/channels/1089876418936180786/1298015953463808102)详细阐述了受委托的研究及其影响。
  
  - 文章讨论了这些发现对大型科技巨头之间 AI 竞争格局的影响。

 

---

**Alignment Lab AI Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

# PART 2: 频道详细摘要与链接

{% if medium == 'web' %}

 

### **HuggingFace ▷ #**[**general**](https://discord.com/channels/879548962464493619/879548962464493622/1297998334174302341) (586 messages🔥🔥🔥):

> - `AI DJ 软件`
> - `Hugging Face 模型查询`
> - `OCR 工具`
> - `训练 TTS 模型`
> - `LLM 中的结构化输出`

- **探索 AI DJ 软件**：用户讨论了 AI 像 DJ 一样在歌曲之间进行过渡的潜力，建议开发类似于 Spotify 但具有自动混音功能的功能。
  
  - 提到了 [rave.dj](https://rave.dj) 等工具，用户可以通过组合多首歌曲来创建混音（mashups），强调了其趣味性，即使结果并不完美。
- **Hugging Face 模型查询**：一位用户询问如何在不暴露模型的情况下下载 Hugging Face 模型权重，寻求关于在私有仓库中使用 `huggingface_hub` 是否合适的澄清。
  
  - 社区回应了关于如何安全管理和下载模型同时保持架构隐藏的建议，并利用环境变量进行身份验证。
- **用于数据提取的 OCR 工具**：用户询问了从 PDF 中提取结构化数据的有效 OCR 解决方案，特别是在建筑背景下。
  
  - 有人建议利用 Koboldcpp 等模型以及各种方法来提高文本提取的准确性。
- **为特定语言训练 TTS 模型**：讨论了训练 TTS 模型的要求，重点是数据收集以及微调现有模型是否能产生高质量结果。
  
  - 参与者强调了拥有合适数据集的重要性，同时质疑对于冷门语言需要多少训练数据。
- **结构化输出实现**：社区交流了关于 LLM 结构化输出的想法，包括利用 `lm-format-enforcer` 等现有库来维持特定格式。
  
  - 建议倾向于使用 Cmd-R 等模型进行结构化响应，而非 Llama，并强调了集成这些能力的挑战。

**Links mentioned**:

- [Google Colab](https://colab.research.google.com/drive/1fxGqfg96RBUvGxZ1XXN07s3DthrKUl4-?usp=sharing): 未找到描述
- [未找到标题](https://tenor.com/view/everybody%27s-so-creative-tiktok-tanara-tanara-double-chocolate-so-creative-g): 未找到描述
- [Suno AI](https://suno.com/about): 我们正在构建一个任何人都能创作出优美音乐的未来。无需乐器，只需想象力。让你的灵感化作音乐。
- [为什么 AI 永远无法真正理解我们：人类意识的隐藏深度](https://medium.com/@ryanfoster_37838/why-ai-will-never-truly-understand-us-the-hidden-depths-of-human-awareness-fbbd3868b649): 引言
- [Everybody'S So Creative Tiktok GIF - Everybody's so creative Tiktok Tanara - 发现并分享 GIF](https://tenor.com/view/everybody%27s-so-creative-tiktok-tanara-tanara-double-chocolate-so-creative-gif-14126964961449949264): 点击查看 GIF
- [Soobkitty Rabbit GIF - Soobkitty Rabbit Bunny - 发现并分享 GIF](https://tenor.com/view/soobkitty-rabbit-bunny-jump-gif-19501897): 点击查看 GIF
- [未找到标题](https://manifund.org/projects/singulrr-10,): 未找到描述
- [Wait What Wait A Minute GIF - Wait What Wait A Minute Huh - 发现并分享 GIF](https://tenor.com/view/wait-what-wait-a-minute-huh-gif-17932668): 点击查看 GIF
- [stabilityai/stable-diffusion-3.5-large · Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-3.5-large): 未找到描述
- [Dirty Docks Shawty GIF - Dirty Docks Shawty Triflin - 发现并分享 GIF](https://tenor.com/view/dirty-docks-shawty-triflin-shawty-triflin-she-gif-22455514): 点击查看 GIF
- [Lol Tea Spill GIF - Lol Tea Spill Laugh - 发现并分享 GIF](https://tenor.com/view/lol-tea-spill-laugh-lmao-spit-take-gif-15049653): 点击查看 GIF
- [Gifmiah GIF - Gifmiah - 发现并分享 GIF](https://tenor.com/view/gifmiah-gif-19835013): 点击查看 GIF
- [Fawlty Towers John Cleese GIF - Fawlty Towers John Cleese Basil Fawlty - 发现并分享 GIF](https://tenor.com/view/fawlty-towers-john-cleese-basil-fawlty-wake-awake-gif-5075198): 点击查看 GIF
- [I Have No Enemies Dog GIF - I have no enemies Dog Butterfly - 发现并分享 GIF](https://tenor.com/view/i-have-no-enemies-dog-butterfly-gif-7312308025622510390): 点击查看 GIF
- [来自 Anthropic (@AnthropicAI) 的推文](https://x.com/AnthropicAI/status/1848742740420341988): 推出升级版的 Claude 3.5 Sonnet 以及新模型 Claude 3.5 Haiku。我们还推出了一项处于 Beta 阶段的新功能：computer use。开发者现在可以指示 Claude 像人类一样使用电脑...
- [You You Are GIF - You You Are Yes - 发现并分享 GIF](https://tenor.com/view/you-you-are-yes-this-guy-your-good-gif-15036437): 点击查看 GIF
- [下载文件](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/file_download#download-a-single-file): 未找到描述
- [Oxidaksi vs. Unglued - Ounk](https://www.youtube.com/watch?v=PFKFNtUDj8g): #混音 #PSY #DNB #音乐 #SPEEDSOUND 使用 Rave.dj 创建。版权所有：©2021 Zoe Love
- [未找到标题](https://medium.com/@ryanfoster_37838/why-ai-will-never-truly-understand-us-the-hidden-depths-of-huma): 未找到描述
- [How Bro Felt After Writing That Alpha Wolf GIF - How bro felt after writing that How bro felt Alpha wolf - 发现并分享 GIF](https://tenor.com/view/how-bro-felt-after-writing-that-how-bro-felt-alpha-wolf-alpha-alpha-meme-gif-307456636039877895): 点击查看 GIF
- [RaveDJ - 音乐混音器](https://rave.dj/)): 使用 AI 一键将任何歌曲混合在一起
- [llama.cpp/grammars/README.md at master · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md): C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号来为 ggerganov/llama.cpp 的开发做出贡献。
- [GitHub - noamgat/lm-format-enforcer: 强制执行语言模型的输出格式 (JSON Schema, Regex 等)](https://github.com/noamgat/lm-format-enforcer): 强制执行语言模型（JSON Schema、Regex 等）的输出格式 - noamgat/lm-format-enforcer
- [GitHub - sktime/sktime: 一个统一的时间序列机器学习框架](https://github.com/sktime/sktime): 一个统一的时间序列机器学习框架 - sktime/sktime
- [accelerate/src/accelerate/utils/fsdp_utils.py at main · huggingface/accelerate](https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/fsdp_utils.py#L256-L326): 🚀 一种在几乎任何设备和分布式配置上启动、训练和使用 PyTorch 模型的简单方法，支持自动混合精度（包括 fp8），以及易于配置的 FSDP 和 DeepSpeed 支持...

### **HuggingFace ▷ #**[**today-im-learning**](https://discord.com/channels/879548962464493619/898619964095860757/1298227386374164552) (18 messages🔥):

> - `2021 lecture series`
> - `Creating virtual characters`
> - `Path to becoming an ML Engineer`
> - `3blue1brown's educational resources`
> - `Manim animation engine`

- **2021 系列讲座开启**：一位成员确认 **2021 lecture series** 将于下周开始，并表达了期待。
  
  - 成员们互相分享了 *All the best* 的祝福以示支持。
- **在 Instagram 上扩展虚拟角色**：一位成员使用 **Civitai** 为 Instagram 个人资料创建了一个虚拟角色，并寻求通过逼真的 Reels 和照片来扩大规模。
  
  - 他们强调自己缺乏编程经验和资源，请求入门建议。
- **应用数学专业学生的 ML Engineer 之路**：一位来自乌克兰的大学生表达了成为 **ML Engineer** 的兴趣，并寻求职业路径指导。
  
  - 成员们建议观看 [3blue1brown 关于 Transformer 和 LLM 的播放列表](https://www.youtube.com/playlist?list=PLUl4u3cNGP63gFHB6xb-kVBiQHYe_4hSi)。
- **3blue1brown 对 ML 的重要性**：强调了 **3blue1brown** 教育资源的重要性，并分享了一门来自 MIT 的特定课程以供进一步探索。
  
  - 成员们鼓励通过观看这些内容来理解人工智能的深层含义。
- **发现用于动画的 Manim**：一位成员询问了 3blue1brown 使用的动画工具，得知是 **Manim**，这是一个自定义动画引擎。
  
  - 分享了 [GitHub 链接](https://github.com/3b1b/manim)，展示了这一用于制作数学解释视频的资源。

**提到的链接**：

- [MIT 6.034 Artificial Intelligence, Fall 2010](https://www.youtube.com/playlist?list=PLUl4u3cNGP63gFHB6xb-kVBiQHYe_4hSi)：查看完整课程：http://ocw.mit.edu/6-034F10 讲师：Patrick Winston。在这些讲座中，Patrick Winston 教授介绍了来自 6.034 的材料...
- [GitHub - 3b1b/manim: Animation engine for explanatory math videos](https://github.com/3b1b/manim)：用于解释性数学视频的动画引擎。通过在 GitHub 上创建账号来为 3b1b/manim 的开发做出贡献。

---

### **HuggingFace ▷ #**[**cool-finds**](https://discord.com/channels/879548962464493619/897390579145637909/) (1 messages):

capetownbali: 发现得不错...

---

### **HuggingFace ▷ #**[**i-made-this**](https://discord.com/channels/879548962464493619/897390720388825149/1298020151316185191) (10 messages🔥):

> - `Granite 3.0 model release`
> - `Webinar on LLM Best Practices`
> - `Evolution of Contextual Embeddings`
> - `ZK Proofs for Chat History Ownership`
> - `PR Merged for HuggingFace.js`

- **Granite 3.0 模型引起轰动**：一款全新的 [端侧 Granite 3.0 模型](https://huggingface.co/spaces/Tonic/on-device-granite-3.0-1b-a400m-instruct) 已发布，并展示了精美的缩略图。
  
  - 用户对其功能以及为快速部署提供的便利性感到兴奋。
- **向 Meta 学习 LLM 最佳实践**：一位 META 高级 ML 工程师正在主持一场关于 LLM 导航的 [网络研讨会](https://shorturl.at/ulrCN)，目前已吸引近 **200 人报名**。
  
  - 该会议承诺将分享关于 Prompt Engineering 和有效选择模型的见解。
- **关于 Self-Attention 演进的文章**：分享了一篇讨论 **从静态到动态上下文嵌入（Contextual Embeddings）演进** 的文章，探讨了从传统向量化到现代方法的创新。
  
  - 作者旨在达到入门级水平，同时也对社区关于补充更多模型的反馈表示认可。
- **用于 ChatGPT 历史记录所有权的 ZK Proofs**：介绍了一个 **Proof of ChatGPT** 的演示，允许用户使用 ZK Proofs 拥有自己的聊天记录，这可能会增加开源模型的训练数据。
  
  - 该应用旨在通过 OpenBlock 的 Universal Data Protocol 增强数据的来源可追溯性和互操作性。
- **HuggingFace.js PR 成功合并**：一个支持 **pxia** 库的 Pull Request 已合并至 [HuggingFace.js](https://github.com/huggingface/huggingface.js/pull/979)。
  
  - 此次更新带来了 AutoModel 支持以及目前的两种架构，增强了该库的功能。

**提到的链接**：

- [On Device Granite 3.0 1b A400m Instruct - a Hugging Face Space by Tonic](https://huggingface.co/spaces/Tonic/on-device-granite-3.0-1b-a400m-instruct)：未找到描述
- [Self-Attention in NLP: From Static to Dynamic Contextual Embeddings](https://medium.com/@d.isham.ai93/self-attention-in-nlp-from-static-to-dynamic-contextual-embeddings-4e26d8c49427)：在自然语言处理（NLP）领域，我们表示单词和句子的方式对性能有着深远的影响……
- [Tweet from OpenBlock (@openblocklabs)](https://x.com/openblocklabs/status/1848805457290572199)：1/ 介绍 Proof of ChatGPT，这是基于 OpenBlock 的 Universal Data Protocol (UDP) 构建的最新应用。此数据证明赋予用户掌握其 LLM 聊天历史的所有权，标志着一个重要的……
- [Add pxia by not-lain · Pull Request #979 · huggingface/huggingface.js](https://github.com/huggingface/huggingface.js/pull/979)：此 PR 将支持我的 pxia 库，该库可在 https://github.com/not-lain/pxia 找到，该库目前提供 AutoModel 支持以及 2 种架构。如果您有任何问题，请告诉我……
- [Explore the Future of AI with Expert-led Events](https://shorturl.at/ulrCN)：Analytics Vidhya 是领先的分析、数据科学和 AI 专业人士社区。我们正在培养下一代 AI 专业人士。获取最新的数据科学、机器学习和 AI……

---

### **HuggingFace ▷ #**[**core-announcements**](https://discord.com/channels/879548962464493619/1014557141132132392/) (1 messages):

sayakpaul: <@&1014517792550166630> 请享用：  
[https://huggingface.co/blog/sd3-5](https://huggingface.co/blog/sd3-5)

---

### **HuggingFace ▷ #**[**NLP**](https://discord.com/channels/879548962464493619/922424173916196955/1298013277846245386) (8 条消息🔥):

> - `Tensor conversion bottleneck` (Tensor 转换瓶颈)
> - `Dataset device bottleneck` (数据集设备瓶颈)
> - `CPU and GPU usage during inference` (推理期间的 CPU 和 GPU 使用情况)
> - `Evaluating fine-tuned LLMs` (评估微调后的 LLM)
> - `Managing evaluation results` (管理评估结果)

- **解决 Tensor 转换瓶颈**：有成员对 Tokenization 解码迭代中潜在的 **Tensor 转换瓶颈**表示担忧，特别是在推理过程中添加到上下文并编码为 float 16 时。
  
  - 建议深入研究工作流（包括解码、打印和将数据传递给模型），以寻找提高效率的改进点。
- **识别到潜在的数据集设备瓶颈**：一位成员质疑是否存在 **数据集设备瓶颈**，因为他们注意到尽管使用了 CUDA，CPU 内存仍飙升至 **1.5 GB**。
  
  - 建议检查是否误将 **UHD Graphics 显卡**（集成显卡）作为主要推理驱动程序，而非独立 GPU。
- **设置 CUDA 设备环境变量**：一位成员提议设置 **CUDA_VISIBLE_DEVICES** 环境变量，以通过以下代码片段优化目标 GPU 的性能：`os.environ["CUDA_VISIBLE_DEVICES"]="1"`。
  
  - 这将有助于确保利用正确的 GPU 执行推理任务，从而实现更好的资源分配。
- **评估微调 LLM 的方法**：讨论了微调 LLM 的评估方法，重点在于通过 deepval 等库实现**自动化**以及由专家进行手动评估。
  
  - 一位成员询问了用于管理不同版本结果以便于比较的工具，并表示感觉 **Google Sheets** 由于其手动特性可能不是最佳选择。
- **对评估工具中协作问题的担忧**：强调了有效管理评估结果的必要性，特别是在多名协作者可能导致 **Google Sheets** 环境中出现错误的情况下。
  
  - 成员们正在寻求更高效的对比分析工具，这表明在共享文档中保持准确性和便利性存在挑战。

 

**提到的链接**：[Annoyed Cat GIF - Annoyed cat - Discover & Share GIFs](https://tenor.com/view/annoyed-cat-gif-17984166845494923336)：点击查看 GIF

 

---

### **HuggingFace ▷ #**[**diffusion-discussions**](https://discord.com/channels/879548962464493619/1009713274113245215/1298096397379309621) (29 条消息🔥):

> - `Kaggle and GPU Usage` (Kaggle 与 GPU 使用)
> - `Model Downloading Techniques` (模型下载技术)
> - `Learning Rate and Training Insights` (学习率与训练见解)
> - `Diffusers Callbacks for Image Generation` (用于图像生成的 Diffusers 回调)
> - `Cultural Connections in AI` (AI 中的文化联系)

- **Kaggle GPU 资源分配的挑战**：用户分享了使用 **Kaggle 双 15GB GPU** 的经验，注意到在模型下载期间一个 GPU 被完全占用，而另一个则处于闲置状态。
  
  - *一位用户询问是否可以跨两个 GPU 对模型进行分片 (sharding)* 以合并资源，而另一位用户确认这种功能可能会降低性能。
- **高效的模型下载策略**：一位成员建议使用 [huggingface_hub](https://huggingface.co/docs/huggingface_hub/index) 库来下载模型，允许用户通过代码控制下载过程。
  
  - 另一位用户指出，如果默认方法出现问题，直接使用 HTTP 请求可以作为一种替代方案。
- **训练中的学习率问题**：提出了关于训练时合适**学习率 (learning rate)** 的担忧，强调了一种根据所用 GPU 数量进行调整的策略。
  
  - 此外，一位用户在完成 **3,300 steps** 后，寻求关于其模型是过拟合还是欠拟合的澄清。
- **在 Diffusers 中实现回调 (Callbacks)**：为了记录图像生成步骤，建议用户利用带有 `callback_on_step_end` 的**回调函数**，以便在去噪循环期间进行实时调整。
  
  - *虽然标准日志记录可以追踪数值，* 但回调函数在追踪每一步生成的图像时提供了更高的灵活性。
- **AI 社区中的文化联系**：一位用户表达了在社区中找到拉美裔同行的热情，庆祝 AI 领域中共同的文化纽带。
  
  - 这一时刻展示了因对 AI 开发的共同兴趣而产生的同志情谊和全球联系。

**提到的链接**：

- [Pipeline callbacks](https://huggingface.co/docs/diffusers/using-diffusers/callback#display-image-after-each-generation-step)：未找到描述
- [GitHub - huggingface/diffusers: 🤗 Diffusers: State-of-the-art diffusion models for image and audio generation in PyTorch and FLAX.](https://github.com/huggingface/diffusers)：🤗 Diffusers：在 PyTorch 和 FLAX 中用于图像和音频生成的先进扩散模型。- huggingface/diffusers

---

### **OpenRouter (Alex Atallah) ▷ #**[**announcements**](https://discord.com/channels/1091220969173028894/1092729520181739581/1298327226164445236) (2 条消息):

> - `Claude 3.5 Sonnet`
> - `Llama 3.1 Nitro`
> - `Ministral updates`
> - `Grok Beta`
> - `Claude self-moderated endpoints`

- **Claude 3.5 Sonnet 取得基准测试提升**：**Claude 3.5 Sonnet** 在各项基准测试中表现出显著提升，用户无需更改代码即可直接试用。更多详情请参阅发布公告 [此处](https://www.anthropic.com/news/3-5-models-and-computer-use)。
  
  - 成员们注意到，将鼠标悬停在提供商旁边的信息图标上可以查看模型升级的时间，从而轻松追踪改进情况。
- **极速 Llama 3.1 Nitro 现已上线**：**Llama 3.1 405b Nitro** 现已可用，其速度比次快的提供商提升了约 **70%**。提供了新端点的直接链接：[405b](https://openrouter.ai/meta-llama/llama-3.1-405b-instruct:nitro) 和 [70b](https://openrouter.ai/meta-llama/llama-3.1-70b-instruct:nitro)。
  
  - 这些超快且优质的端点承诺约 **120 tps** 的吞吐量，吸引了用户的广泛关注。
- **Ministral 带来强大的新模型**：Mistral 推出了 **Ministral 8b**，支持 **150 tps** 并具备 **128k** 的高上下文，目前在 **技术提示词（tech prompts）排名第 4**。一款经济型的 3b 模型也已通过[此链接](https://openrouter.ai/mistralai/ministral-8b)提供。
  
  - 用户对性能和价格表示兴奋，这两款模型吸引了不同预算范围的用户。
- **Grok Beta 亮相并扩展功能**：**Grok 2** 现已重命名为 **Grok Beta**，其上下文长度增加至 **131,072**，新的输出价格为 **$15/m**。此外，旧有的 `x-ai/grok-2` 请求已设置为 `x-ai/grok-beta` 的别名，以保证用户体验的连续性。
  
  - 社区对此次更新表示欢迎，期待功能的改进以及定价模型的进一步明确。
- **关于 Claude 自我审核端点理想体验的投票**：发起了一项投票，以收集社区对 **Claude self-moderated** (`:beta`) 端点理想体验的反馈，该端点目前位居排行榜榜首。成员可以通过在[此处](https://discord.com/channels/1091220969173028894/1107397803266818229/1298353935500836957)投票来发表意见。
  
  - 用户的参与表明了他们对塑造这些端点未来体验的浓厚兴趣。

**提到的链接**：

- [Llama 3.1 405B Instruct (nitro) - API, Providers, Stats](https://openrouter.ai/meta-llama/llama-3.1-405b-instruct:nitro)：备受期待的 Llama3 400B 级别模型来了！凭借 128k 上下文和令人印象深刻的评估分数，Meta AI 团队继续推动开源 LLM 的前沿。Meta 最新的 c...
- [Llama 3.1 70B Instruct (nitro) - API, Providers, Stats](https://openrouter.ai/meta-llama/llama-3.1-70b-instruct:nitro)：Meta 最新的模型系列 (Llama 3.1) 推出了多种尺寸和版本。通过 API 运行 Llama 3.1 70B Instruct (nitro)。
- [Ministral 8B - API, Providers, Stats](https://openrouter.ai/mistralai/ministral-8b)：Ministral 8B 是一个 8B 参数模型，采用独特的交错滑动窗口注意力（interleaved sliding-window attention）模式，以实现更快、内存效率更高的推理。专为边缘用例设计，支持高达 128k 上下文...

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1297998285503467600) (455 条消息🔥🔥🔥):

> - `New Claude 3.5 Sonnet`
> - `OpenRouter API`
> - `Computer Use 功能`
> - `模型定价`
> - `Haiku 3.5 发布`

- **新版 Claude 3.5 Sonnet 发布**：新版 Claude 3.5 Sonnet 模型已正式发布，并可在 OpenRouter 上使用。
  
  - 用户对其能力和最近的改进表示兴奋，评论中提到了速度和性能。
- **OpenRouter API 密钥与使用**：新用户询问了如何获取和使用 OpenRouter 平台的 API 密钥，并确认密钥可以访问所有可用模型。
  
  - 建议用户使用 OpenRouter Playground 以方便访问和测试。
- **引入 Computer Use 功能**：Anthropic 宣布了一项新的 “Computer Use” 功能，允许用户提供自己的电脑供 AI 操作。
  
  - 这一能力被描述为具有创新性且实用，尽管也引发了对潜在滥用和安全性的担忧。
- **模型定价讨论**：讨论了使用 Claude 等模型的定价，强调某些选项的成本约为每百万 token 18 美元。
  
  - 用户提到在低成本替代方案的背景下，比较了包括 DeepSeek 和 Qwen 在内的各种模型的成本。
- **即将发布的 Haiku 3.5**：新版 Haiku 3.5 模型的发布日期宣布为本月晚些时候，但具体细节仍有待公布。
  
  - 用户期待这一发布，并推测其与现有模型相比的影响和性能。

**提到的链接**：

- [来自 Anthropic (@AnthropicAI) 的推文](https://x.com/AnthropicAI/status/1848742740420341988)：介绍升级后的 Claude 3.5 Sonnet 以及新模型 Claude 3.5 Haiku。我们还推出了一项处于 Beta 阶段的新能力：computer use。开发者现在可以指示 Claude 像人类一样使用电脑...
- [对 Anthropic 新 Computer Use 能力的初步探索](https://simonwillison.net/2024/Oct/22/computer-use/)：Anthropic 今天发布了两个重大公告：新的 Claude 3.5 Sonnet 模型和一种他们称为 computer use 的新 API 模式。（他们还预告了 Haiku 3.5，但那是……）
- [聊天室 | OpenRouter](https://openrouter.ai/chat)：LLM 聊天室是一个多模型聊天界面。添加模型并开始聊天！聊天室将数据本地存储在您的浏览器中。
- [快速入门 | OpenRouter](https://openrouter.ai/docs/quick-start)：开始使用 OpenRouter 进行构建
- [活动记录 | OpenRouter](https://openrouter.ai/activity)：查看您在 OpenRouter 上使用模型的情况。
- [abacusai/Dracarys2-72B-Instruct - Featherless.ai](https://featherless.ai/models/abacusai/Dracarys2-72B-Instruct)：Featherless - 最新的 LLM 模型，无服务器且可根据您的请求随时使用。
- [全栈与 Web3 开发者](https://daniel0629.vercel.app)：我是一名高技能的区块链和全栈开发者，在设计和实现复杂的去中心化应用和 Web 解决方案方面拥有丰富经验。
- [密钥 | OpenRouter](https://openrouter.ai/settings/keys)：管理您的密钥或创建新密钥
- [abacusai/Dracarys2-72B-Instruct · Hugging Face](https://huggingface.co/abacusai/Dracarys2-72B-Instruct)：未找到描述
- [Malding Weeping GIF - Malding Weeping Pov - 发现并分享 GIF](https://tenor.com/view/malding-weeping-pov-malder-nikocado-gif-24866915)：点击查看 GIF
- [模型 | OpenRouter](https://openrouter.ai/models?max_price=0)：在 OpenRouter 上浏览模型
- [OpenRouter](https://openrouter.ai/)：LLM 路由与市场
- [utils.py 第 8123 行未捕获的 APIError · Issue #2104 · Aider-AI/aider](https://github.com/Aider-AI/aider/issues/2104)：Aider 版本：0.59.1 Python 版本：3.11.9 平台：Windows-10-10.0.22631-SP0 Python 实现：CPython 虚拟环境：是 操作系统：Windows 10 (64bit) Git 版本：git version 2.43.0.windo...
- [使用此提供商而非 @ai-sdk/openai 有什么好处？ · Issue #4 · OpenRouterTeam/ai-sdk-provider](https://github.com/OpenRouterTeam/ai-sdk-provider/issues/4)：解释两者之间的区别会很有帮助。我可以通过 @ai-sdk/openai 使用 OpenRouter 的模型，而且它在积极维护中。
- [Reddit - 深入探索一切](https://www.reddit.com/r/ClaudeAI/)：未找到描述

---

### **aider (Paul Gauthier) ▷ #**[**general**](https://discord.com/channels/1131200896827654144/1131200896827654149/1298001236150190110) (290 条消息🔥🔥):

> - `Claude 3.5 Sonnet`
> - `DeepSeek 作为编辑器模型`
> - `模型定价`
> - `模型性能与基准测试`
> - `本地模型集成`

- **Claude 3.5 Sonnet 表现出显著提升**：新版 Claude 3.5 Sonnet 在 Aider 的代码编辑排行榜上以 **84.2%** 位居榜首，在 architect 模式下配合 DeepSeek 达到了 **85.7%**。

- 许多用户对这些增强功能感到兴奋，特别是在编程任务方面，且其定价结构与之前的模型相同。
- **将 DeepSeek 用作 Editor Model**：DeepSeek 因比 Sonnet 便宜得多而受到青睐，其成本为 **每 1M output tokens 0.28 美元**，而 Sonnet 为 **15 美元**。
  
  - 用户报告称，将 DeepSeek 作为 Editor Model 使用时节省了大量费用，并表示在与 Sonnet 搭配使用时表现尚可。
- **关于 Token 成本的担忧**：讨论指出，将 Sonnet 作为 Architect 与 DeepSeek 配合执行，主要是将支出从规划 tokens 转移到了输出 tokens。
  
  - 这引发了一场关于节省的 token 成本是否足以抵消 DeepSeek 较慢性能的辩论。
- **模型性能与本地使用**：有人询问离线模型的有效性，以及它们通过提供解析或错误纠正来辅助 Sonnet 的潜力。
  
  - 用户建议尝试使用更大的本地模型，以在与 Sonnet 集成时增强能力。
- **音频录制与转录**：出现了一个关于转录音频录音是远程提交还是有离线支持的问题。
  
  - 这引发了关于使用 Whisper 等模型实现潜在离线转录能力的讨论。

**提及的链接**：

- [来自 Anthropic (@AnthropicAI) 的推文](https://x.com/AnthropicAI/status/1848742740420341988)：介绍升级版的 Claude 3.5 Sonnet，以及新模型 Claude 3.5 Haiku。我们还推出了一项处于 Beta 阶段的新功能：computer use。开发者现在可以指示 Claude 像...一样使用计算机。
- [模型警告](https://aider.chat/docs/troubleshooting/warnings.html)：aider 是你终端里的 AI 结对编程工具。
- [分离代码推理与编辑](https://aider.chat/2024/09/26/architect.html)：Architect 模型描述如何解决编程问题，而 Editor 模型将其转化为文件编辑。这种 Architect/Editor 方法产生了 SOTA 基准测试结果。
- [Computer use (beta) - Anthropic](https://docs.anthropic.com/en/docs/build-with-claude/computer-use#text-editor-tool)：未找到描述
- [Walk Andrew Tate Walk GIF - Walk Andrew Tate Walk Top G - Discover & Share GIFs](https://tenor.com/view/walk-andrew-tate-walk-top-g-top-g-walk-savage-gif-26857321)：点击查看 GIF
- [YAML 配置文件](https://aider.chat/docs/config/aider_conf.html)：如何使用 YAML 配置文件配置 aider。
- [Aider LLM 排行榜](https://aider.chat/docs/leaderboards/)：LLM 代码编辑能力的定量基准测试。
- [aider/aider/models.py at 1b530f9200078e5653b6de04c2bd9f820bf38380 · Aider-AI/aider](https://github.com/Aider-AI/aider/blob/1b530f9200078e5653b6de04c2bd9f820bf38380/aider/models.py#L297)：aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建一个账户来为 Aider-AI/aider 的开发做出贡献。
- [选项参考](https://aider.chat/docs/config/options.html#--model-model)：关于 aider 所有设置的详细信息。
- [GitHub - nekowasabi/aider.vim: Helper aider with neovim](https://github.com/nekowasabi/aider.vim)：在 neovim 中辅助 aider。通过在 GitHub 上创建一个账户来为 nekowasabi/aider.vim 的开发做出贡献。
- [GitHub - CEDARScript/cedarscript-integration-aider: Allows Aider to use CEDARScript as an edit format](https://github.com/CEDARScript/cedarscript-integration-aider?tab=readme-ov-file#why-use-cedarscript)**)：允许 Aider 使用 CEDARScript 作为编辑格式。通过在 GitHub 上创建一个账户来为 CEDARScript/cedarscript-integration-aider 的开发做出贡献。
- [Update models.py by cschubiner · Pull Request #2117 · Aider-AI/aider](https://github.com/Aider-AI/aider/pull/2117/files)：未找到描述
- [Llama 3.1 405B Instruct (nitro) - API, Providers, Stats](https://openrouter.ai/meta-llama/llama-3.1-405b-instruct:nitro)：备受期待的 400B 级 Llama3 来了！拥有 128k 上下文和令人印象深刻的评估分数，Meta AI 团队继续推动开源 LLM 的前沿。
- [Llama 3.1 70B Instruct (nitro) - API, Providers, Stats](https://openrouter.ai/meta-llama/llama-3.1-70b-instruct:nitro)：Meta 最新的模型系列 (Llama 3.1) 推出了多种尺寸和版本。通过 API 运行 Llama 3.1 70B Instruct (nitro)。
- [Ministral 8B - API, Providers, Stats](https://openrouter.ai/mistralai/ministral-8b)：Ministral 8B 是一个 8B 参数模型，具有独特的交错滑动窗口注意力模式，可实现更快、内存效率更高的推理。专为边缘用例设计，支持高达 128k 的上下文...

### **aider (Paul Gauthier) ▷ #**[**questions-and-tips**](https://discord.com/channels/1131200896827654144/1133060505792159755/1297998482098884618) (20 messages🔥):

> - `Aider Configuration`
> - `Aider Command Issues`
> - `Architect Mode Functionality`
> - `Mistral API Authentication`
> - `Reddit and Claude AI Discussions`

- **设置 Aider 配置**：一位用户请求关于创建 `.aider.conf.yml` 文件的指导，并列出了所需的模型设置：将 `openrouter/anthropic/claude-3.5-sonnet:beta` 作为模型和编辑器模型（editor model），将 `openrouter/anthropic/claude-3-haiku:beta` 作为弱模型（weak model）。
  
  - 另一位成员询问 Aider 在运行时从何处获取这些配置详情。
- **Aider 命令标志（Flag）问题**：一位用户报告了在 Aider 中使用 `--yes` 标志时的问题，命令似乎仍会提议文件并立即退出。
  
  - 有回复建议该行为可能已更改为 `--yes-always`，这可能会影响操作。
- **关于 Architect Mode 的疑问**：一位用户对 Architect Mode 表示困惑，称其会自动添加文件，但随后会提示需要额外文件，却没有明确的指令说明如何先将它们添加到上下文（context）中。
  
  - 其他人建议尝试简单的按键输入如 'Y' 或 'Enter' 来解决问题，同时已提交 Bug 报告以跟踪此情况。
- **Mistral API 身份验证问题**：一位用户在尝试将 Mistral API 与 Aider 配合使用时遇到 401 Unauthorized 错误，表明存在身份验证错误。
  
  - 经过排查，发现他们需要生成一个新的 API key，随后问题得到解决。
- **Reddit 和 Claude AI 见解**：一位用户分享了一个讨论 Claude AI 新功能的 Reddit 链接，包括 Claude 3.5 Sonnet 的新能力，该能力允许直接进行计算机交互。
  
  - 这引发了其他用户关于 Claude 功能的额外见解和确认。

**提到的链接**：

- [Does architect mode prompt to add files? · Issue #2121 · Aider-AI/aider](https://github.com/Aider-AI/aider/issues/2121)：在 Discord 中分享的 Issue：https://discord.com/channels/1131200896827654144/1133060505792159755/1298228879210577931 /architect 示例等等……现在，我们需要更新其他文件以合并……
- [Reddit - Dive into anything](https://www.reddit.com/r/ClaudeAI/s/nwiLAXUDtz)：未找到描述
- [Anthropic (@AnthropicAI) 的推文](https://x.com/AnthropicAI/status/1848742740420341988)：推出升级版的 Claude 3.5 Sonnet，以及新模型 Claude 3.5 Haiku。我们还推出了一项处于 Beta 阶段的新能力：Computer use。开发者现在可以指示 Claude 像人类一样使用计算机……

---

### **aider (Paul Gauthier) ▷ #**[**links**](https://discord.com/channels/1131200896827654144/1268910919057149974/1298301944062021672) (5 messages):

> - `Claude 3.5 Sonnet upgrades`
> - `Claude 3.5 Haiku introduction`
> - `Computer use capability`
> - `DreamCut AI video editor`

- **Claude 3.5 Sonnet 和 Haiku 升级发布**：Anthropic 宣布了**升级版的 Claude 3.5 Sonnet** 和新模型 **Claude 3.5 Haiku**，后者在编程性能方面表现出色，在 SWE-bench Verified 测试中从 **33.4%** 提升至 **49.0%**。
  
  - 据报道，**Claude 3.5 Haiku** 的表现优于其前代产品，同时在多项评估中与之前最大的模型能力持平。
- **关于新 Computer Use 功能的见解**：引入了一项突破性的 **Computer use** 能力，允许开发者指示 Claude 像人类一样操作计算机，例如移动光标和点击按钮。
  
  - 该功能目前处于公开 Beta 阶段，被描述为**实验性**的，虽然可能存在错误，但对可用性具有重大影响。
- **DreamCut AI - 全能 AI 软件构建器**：@MengTo 介绍了一个新的视频编辑平台 [DreamCut AI](http://dreamcut.ai)，该平台是使用 Claude AI 历时 3 个月、编写了 **5 万行代码** 构建而成的。
  
  - 该工具目前处于早期访问阶段，允许用户通过免费账号测试其 AI 功能。

**提到的链接**：

- [Initial explorations of Anthropic’s new Computer Use capability](https://simonwillison.net/2024/Oct/22/computer-use/)：Anthropic 今天发布了两个重大公告：新的 Claude 3.5 Sonnet 模型和一种他们称为 Computer use 的新 API 模式。（他们还预告了 Haiku 3.5，但那是……）
- [Introducing computer use, a new Claude 3.5 Sonnet, and Claude 3.5 Haiku](https://www.anthropic.com/news/3-5-models-and-computer-use)：更新后的、更强大的 Claude 3.5 Sonnet，Claude 3.5 Haiku，以及一项新的实验性 AI 能力：Computer use。
- [Meng To (@MengTo) 的推文](https://x.com/MengTo/status/1848669694800367901)：介绍 http://dreamcut.ai，这是一个我使用 Claude AI 从零开始构建的视频编辑器。这花费了 3 个月时间和超过 5 万行代码。我完全跳过了设计，直接进入编码阶段。目前处于早期……

---

### **Stability.ai (Stable Diffusion) ▷ #**[**announcements**](https://discord.com/channels/1002292111942635562/1002292398703001601/1298294266493407264) (1 条消息):

> - `Stable Diffusion 3.5 Launch`
> - `Performance of Stable Diffusion 3.5 Large`
> - `Stable Diffusion 3.5 Large Turbo`
> - `Community Feedback`
> - `Accessibility of New Models`

- **Stable Diffusion 3.5 发布公告**：**Stable Diffusion 3.5** 的发布包含了多个适用于消费级硬件的可定制变体，并根据 [Stability AI Community License](https://stability.ai/community-license-agreement) **对所有用途免费**。**Stable Diffusion 3.5 Large** 和 **Turbo** 模型目前已在 [Hugging Face](https://huggingface.co/stabilityai) 和 [GitHub](https://github.com/Stability-AI/sd3.5) 上线。
  
  - **3.5 Medium** 模型将于 **10 月 29 日**发布，这体现了在收到之前的社区反馈后对持续开发的承诺。
- **Stable Diffusion 3.5 Large 树立市场新标准**：**Stable Diffusion 3.5 Large** 因其在 **Prompt Adherence**（提示词遵循度）方面的市场领先地位以及媲美更大模型的图像质量而受到赞誉。该版本代表了在听取社区对前一版本反馈后的重大进步。
  
  - *分析显示*，满足社区标准对于确保产品在增强视觉媒体方面的有效性至关重要。
- **Stable Diffusion 3.5 Large Turbo 实现快速推理**：新推出的 **Stable Diffusion 3.5 Large Turbo** 拥有同类产品中**最快的推理时间**，同时保持了极具竞争力的图像质量和提示词遵循度。这使其成为当前产品中一个令人兴奋的选择。
  
  - 许多用户表达了热情，因为这个新的 Turbo 变体符合对模型性能中速度和质量的双重需求。
- **社区参与驱动开发**：Stability AI 团队花时间回应**社区反馈**，而不是匆忙修复，从而开发了 **Stable Diffusion 3.5**。这突显了参与度和响应能力在产品改进中的重要性。
  
  - 事实证明，社区的投入对于塑造旨在**赋能构建者和创作者**的工具至关重要。
- **致力于工具的可访问性**：Stability AI 强调其致力于为构建者提供**广泛可访问**的工具，以用于包括微调和艺术创作在内的各种用例。此次发布鼓励在工作流的各个环节进行**分发和商业化**。
  
  - 这种方法展示了 Stability AI 旨在促进社区内持续创新和创造力的目标。

 

**提到的链接**：[Stable Diffusion 3.5 — Stability AI](https://stability.ai/news/introducing-stable-diffusion-3-5)：今天我们推出了 Stable Diffusion 3.5。这个开放版本包含多个模型变体，包括 Stable Diffusion 3.5 Large 和 Stable Diffusion 3.5 Large Turbo。

 

---

### **Stability.ai (Stable Diffusion) ▷ #**[**general-chat**](https://discord.com/channels/1002292111942635562/1002292112739549196/1298002080618909797) (280 messages🔥🔥):

> - `Stable Diffusion 3.5 Release`
> - `Performance Comparisons with Flux`
> - `Model Licensing`
> - `Technical Support for SD3.5`
> - `Applications of LoRA in AI Art`

- **Stable Diffusion 3.5 发布令社区感到惊喜**：在一段沉寂之后，**SD 3.5** 的发布公告让许多人感到意外，用户们讨论了它的突然发布以及相对于之前版本的潜在改进。
  
  - 一些用户注意到 SD 3.5 改进了提示词遵循（prompt following）能力，而另一些用户则对其与 **Flux** 相比的性能表现表示担忧。
- **关于质量对比的讨论：SD3.5 vs. Flux**：成员们争论 **SD3.5** 是否能与 **Flux** 的图像质量相媲美，并提到了它在微调（fine-tuning）和整体美感方面的表现。
  
  - 初步印象表明 **Flux** 在审美质量上可能仍保持优势，这引发了人们对这两个数据集更多细节的好奇。
- **SD3.5 的新许可详情**：**SD3.5** 许可模式的稳定性引发了疑问，一些参与者对其与 **AuraFlow** 相比的商业用途表示担忧。
  
  - 讨论强调了在模型易用性与允许 **Stability AI** 有效获利之间取得平衡的重要性。
- **使用 Automatic1111 的技术支持**：遇到 **Automatic1111 Web UI** 困难的用户被引导至特定频道寻求支持，这反映了一个积极帮助新人的活跃社区。
  
  - 一位用户很快找到了专门的技术援助频道，显示了成员们积极主动的态度。
- **LoRA 应用探索**：针对 **SD3.5** 的 **LoRA** 模型的推出引起了兴奋，用户分享了提示词和生成结果，突出了其在增强图像生成方面的效用。
  
  - 社区展示了他们的作品，并鼓励尝试新的提示词以发挥 LoRA 的潜力。

**相关链接**：

- [Stability AI - Developer Platform](https://platform.stability.ai/docs/api-reference): 未找到描述
- [Tweet from Stability AI (@StabilityAI)](https://x.com/StabilityAI/status/1848720074057859268): 未找到描述
- [SD3 Examples](https://comfyanonymous.github.io/ComfyUI_examples/sd3/): ComfyUI 工作流示例
- [Shakker-Labs/SD3.5-LoRA-Linear-Red-Light · Hugging Face](https://huggingface.co/Shakker-Labs/SD3.5-LoRA-Linear-Red-Light): 未找到描述
- [Stable Diffusion 3.5 - a stabilityai Collection](https://huggingface.co/collections/stabilityai/stable-diffusion-35-671785cca799084f71fa2838): 未找到描述
- [stabilityai/stable-diffusion-3.5-large · Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-3.5-large): 未找到描述
- [stability-ai/stable-diffusion-3.5-large – Run with an API on Replicate](https://replicate.com/stability-ai/stable-diffusion-3.5-large): 未找到描述
- [fal.ai | The generative media platform for developers](https://fal.ai/): fal.ai 是运行扩散模型最快的方式，提供开箱即用的 AI 推理、训练 API 和 UI Playgrounds。

---

### **Unsloth AI (Daniel Han) ▷ #**[**general**](https://discord.com/channels/1179035537009545276/1179035537529643040/1298009156523130951) (234 messages🔥🔥):

> - `Gradient Accumulation Bug 修复`
> - `LLM 训练效率`
> - `模型性能与基准测试 (Benchmarking)`
> - `与 Meta 的合作`
> - `微调 (Finetuning) 策略`

- **Nightly Transformers 已修复 Gradient Accumulation Bug**：关于 [Gradient Accumulation Bug](https://www.reddit.com/r/MachineLearning/comments/1g8ymrn/r_gradient_accumulation_bug_fix_in_nightly/) 的最新更新显示该问题已修复，并应包含在 Nightly Transformers 和 Unsloth 训练器中。
  
  - 此 Bug 此前导致各种训练器中的 Loss 曲线计算不准确。
- **关于 LLM 训练效率的见解**：成员们讨论了 LLM 训练的效率，强调向模型教授短语会生成多个子样本，而非单个实例。
  
  - 这种方法有效地最大化了训练样本量，使模型能够循序渐进地学习。
- **模型性能与基准测试的挑战**：一位成员对新款 Nvidia Nemotron Llama 3.1 模型的性能表示怀疑，质疑其在基准测试分数相近的情况下是否优于标准 Llama 70B 模型。
  
  - 有人指出 Nvidia 的基准测试可能存在不一致性，影响了其模型的可感知性能。
- **即将与 Meta 展开合作**：Unsloth 团队计划与 Meta 合作，社区成员对潜在成果表示期待。
  
  - 澄清指出，Meta 新发布的模型侧重于预训练 (Pre-training) 和研究，而非与 Unsloth 直接竞争。
- **提升模型性能的微调策略**：讨论围绕微调模型时数据集质量的重要性展开，再次强调针对性数据集通常在特定领域产生更好效果。
  
  - 一位成员分享了尝试使用 Finetome 100k 数据集增强 1B 模型性能的经验，并指出结果褒贬不一。

**提到的链接**：

- [MMLU Pro - a Hugging Face Space by TIGER-Lab](https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro)：未找到描述
- [ibm-granite/granite-3.0-8b-instruct · Hugging Face](https://huggingface.co/ibm-granite/granite-3.0-8b-instruct)：未找到描述
- [Reddit - Dive into anything](https://www.reddit.com/r/MachineLearning/comments/1g8ymrn/r_gradient_accumulation_bug_fix_in_nightly/)：未找到描述
- [Unsloth Notebooks | Unsloth Documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks)：请参阅下方我们所有的 Notebooks 列表：
- [GitHub - facebookresearch/lingua: Meta Lingua: a lean, efficient, and easy-to-hack codebase to research LLMs.](https://github.com/facebookresearch/lingua)：Meta Lingua：一个精简、高效且易于修改的 LLM 研究代码库。- facebookresearch/lingua
- [Reddit - Dive into anything](https://www.reddit.com/r/MachineLea)：未找到描述
- [Windows installation guide in README by timothelaborie · Pull Request #1165 · unslothai/unsloth](https://github.com/unslothai/unsloth/pull/1165/commits/d3aff7e83f44820db2690fbd2f0de693ec66757e)：以前阻碍 Unsloth 在 Windows 上运行的唯一障碍是 Triton，但现在有了 Windows 分支。安装后，Unsloth 立即可以运行，并提供了与 WSL 相同的准确度。
- [gbharti/finance-alpaca · Datasets at Hugging Face](https://huggingface.co/datasets/gbharti/finance-alpaca)：未找到描述

---

### **Unsloth AI (Daniel Han) ▷ #**[**help**](https://discord.com/channels/1179035537009545276/1179777624986357780/1298078088931967009) (34 条消息🔥):

> - `Grad School Application Editor` (研究生申请文书编辑器)
> - `LLaMA Model Fine-Tuning` (LLaMA 模型微调)
> - `Unsloth Installation Issues` (Unsloth 安装问题)
> - `Multi-GPU Support in Unsloth` (Unsloth 中的多 GPU 支持)
> - `CUDA and Library Errors` (CUDA 和库错误)

- **构建研究生申请文书编辑器的挑战**：一位成员表示希望创建一个**研究生申请文书编辑器**，但在处理用于 AI 模型实现的大型详细 Prompt 时遇到困难。他们寻求关于微调模型的指导，以克服输出陈词滥调和 Prompt 结构复杂的问题。
- **在 CSV 数据上微调 LLaMA**：用户咨询了在 CSV 数据上微调 **LLaMA 模型** 是否能使其回答有关事故数据的特定查询。建议包括通过 [Turing 文章](https://www.turing.com/resources/understanding-llm-evaluation-and-benchmarks) 中提供的某些方法论来评估模型性能。
- **本地安装 Unsloth 的问题**：一位用户报告在按照脚本创建 conda 环境安装 **Unsloth** 时，由于批处理文件失效而遇到困难。其他成员建议使用 **WSL2** 来简化安装过程。
- **关于 Unsloth 多 GPU 支持的疑问**：讨论了 Unsloth 中的**多 GPU 支持**，确认该框架目前不支持在多个 GPU 上加载模型。用户正试图理解在当前限制下 `per_device_train_batch_size` 的作用。
- **排查 CUDA 和库错误**：一位用户在运行 Unsloth 时遇到了与 CUDA 库相关的 **ImportError**，引发了对 CUDA 配置损坏的猜测。求助请求强调，解决此类问题需要确保 CUDA 的稳定性和与已安装库的兼容性。

**提到的链接**：

- [未找到标题](https://www.turing.com/resources/understanding-llm-evaluation-and-benchmarks)：未找到描述
- [Reddit - Dive into anything](https://www.reddit.com/r/unsloth/comments/1e4w)：未找到描述
- [Reddit - Dive into anything](https://www.reddit.com/r/unsloth/comments/1e4w3i0/wrote_a_python_script_to_auto_install_unsloth_on/)：未找到描述

---

### **Unsloth AI (Daniel Han) ▷ #**[**community-collaboration**](https://discord.com/channels/1179035537009545276/1180144489214509097/1298368178392535080) (1 条消息):

> - `Unsloth Studio Fixes` (Unsloth Studio 修复)
> - `GitHub Pull Request` (GitHub Pull Request)
> - `Discord Issue Reporting` (Discord 问题报告)

- **修复 Unsloth Studio 的 Pull Request**：由 Erland366 提交的新 [Pull Request #1](https://github.com/unslothai/unsloth-studio/pull/1/files) 解决了 Discord 用户报告的几个 Studio 问题，特别是在导入 Unsloth 期间的问题。
  
  - 据报道，该问题在微调 notebook 中并未出现，这引发了社区的进一步调查。
- **用户在 Discord 上报告问题**：一位用户强调了在 Discord 频道中导入 Unsloth 时触发的问题，表明需要迅速解决。
  
  - 鼓励社区审查该 Pull Request 并提供反馈，以解决报告的问题。

 

**提到的链接**：[Fix/studio by Erland366 · Pull Request #1 · unslothai/unsloth-studio](https://github.com/unslothai/unsloth-studio/pull/1/files)：Studio 中存在几个问题。这些问题是由 Discord 用户提出的。该问题在导入 Unsloth 时触发，但不知为何在微调 notebook 中没有发生……

 

---

### **Nous Research AI ▷ #**[**general**](https://discord.com/channels/1053877538025386074/1149866623109439599/1297998227248779354) (137 条消息🔥🔥):

> - `Fine-Tuning 中的灾难性遗忘`
> - `LLM 在 Benchmark 上的表现`
> - `Nous Research 视频与项目`
> - `Claude 模型更新`
> - `Token as a Service 提供商`

- **灾难性遗忘的探索**：讨论集中在大型语言模型 (LLM) 在持续指令微调 (continual instruction tuning) 过程中观察到的**灾难性遗忘 (catastrophic forgetting)** 现象，特别是在 1B 到 7B 参数规模的模型中。
  
  - 有人指出 Fine-Tuning 可能会显著降低性能，用户分享了个人经验以及将其模型与既有模型进行对比的 Benchmark 结果。
- **Benchmark 性能见解**：用户讨论了模型规模对性能的影响，指出在有限数据上训练且未达到优化状态可能会导致结果不佳。
  
  - 一位参与者强调了他们的 1B 模型与 Meta 的模型相比得分较低，强调了 Baseline 对比的重要性。
- **Nous Research 视频与未来项目**：成员们对最近关于 Forge 的 **Nous Research 视频** 表现出极大的热情，认为这是其项目中一个很有前景的进展。
  
  - 大家对 Forge 项目中知识图谱 (knowledge graph) 的实现感到好奇，展现出对记忆功能如何集成的兴趣。
- **Claude 模型增强**：注意力转向了 **AnthropicAI** 的最新更新，展示了 Claude 3.5 Sonnet 和 Haiku 模型，具备计算机交互能力。
  
  - 参与者注意到了 Sonnet 令人印象深刻的特性，同时讨论了在 Claude 4.0 即将发布的情况下保持竞争优势的意义。
- **关于 Token as a Service 提供商的讨论**：有人询问了支持 Nous 模型的可用 **token as a service** 平台，重点关注 Octo AI 的替代方案。
  
  - 对话延伸到了 OpenRouter 的产品，反映了通过公共 Endpoint 访问 Nous 技术的兴趣。

**提到的链接**：

- [An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning](https://arxiv.org/abs/2308.08747)：大型语言模型在持续微调过程中灾难性遗忘的实证研究。灾难性遗忘 (CF) 是机器学习中发生的一种现象，即模型在获取新知识时遗忘了之前学到的信息。随着大型语言模型 (LLM) 的普及...
- [来自 Nous Research (@NousResearch) 的推文](https://x.com/NousResearch/status/1848397863547515216)：未找到描述
- [来自 Anthropic (@AnthropicAI) 的推文](https://x.com/anthropicai/status/1848742740420341988?s=46)：介绍升级版的 Claude 3.5 Sonnet 以及新模型 Claude 3.5 Haiku。我们还推出了一项处于 Beta 阶段的新功能：computer use。开发者现在可以指导 Claude 像人类一样使用计算机...
- [Apple 研究揭示了 LLM “推理”能力的深层缺陷](https://arstechnica.com/ai/2024/10/llms-cant-perform-genuine-logical-reasoning-apple-researchers-suggest/)：无关的干扰信息 (red herrings) 会导致逻辑推理的“灾难性”失败。
- [GitHub - microsoft/BitNet: Official inference framework for 1-bit LLMs](https://github.com/microsoft/BitNet)：1-bit LLM 的官方推理框架。欢迎通过在 GitHub 上创建账号来为 microsoft/BitNet 的开发做出贡献。
- [Nous Research](https://www.youtube.com/watch?v=7ZXPWTdThAA)：未找到描述
- [Forge by Nous Research @ Nouscon 2024](https://www.youtube.com/watch?v=zmnzW0r_g8k&list=PLjOo65uEP4cYhV7c2whkhDfWy58XFj7yL&index=8&t=514s)：Nous Research 联合创始人 Karan 在 Nouscon 2024 上谈论我们即将推出的项目之一 "Forge"。
- [无标题](https://manifund.org/projects/singulrr-10,)：未找到描述
- [diabolic6045/open-llama-3.2-1B-Instruct · Hugging Face](https://huggingface.co/diabolic6045/open-llama-3.2-1B-Instruct)：未找到描述
- [Open Llama 3.2 1B Instruct - diabolic6045 的 Hugging Face Space](https://huggingface.co/spaces/diabolic6045/open-llama-3.2-1B-Instruct)：未找到描述
- [Category: AI](https://arstechnica.com/ai/2024/10/llms-cant-perform-genuine-logical-reasoning-apple-researchers-sug)：打开舱门……
- [Hermes 3 405B Instruct - API, Providers, Stats](https://openrouter.ai/nousresearch/hermes-3-llama-3.1-405b)：Hermes 3 是一款通用语言模型，相比 Hermes 2 有许多改进，包括先进的 Agent 能力、更好的角色扮演、推理、多轮对话、长上下文一致性...

---

### **Nous Research AI ▷ #**[**ask-about-llms**](https://discord.com/channels/1053877538025386074/1154120232051408927/1298027825730814078) (5 条消息):

> - `Hermes 3 的可用性`
> - `Claude system prompt 增强`
> - `Claude 的问题解决能力`

- **Hermes 3 可能在 Replicate 上不可用**：*Mentallyblue* 询问 **Hermes 3** 是否在 Replicate 上可用，*Teknium* 回应称这似乎与涉及 **8B** 和 **70B** 模型的合作有关。
  
  - 这表明 **Hermes 3** 目前可能无法独立访问。
- **新版 Claude 增强了注意力处理**：最近的讨论强调，新版 **Claude** 更新了 system prompt 以管理 **misguided attention**，明确阐述了谜题的约束条件。
  
  - *Azure2089* 指出，这次更新对 Claude 有所帮助，但也承认它在面对熟悉谜题的细微变化时仍可能出错。
- **Claude 在 CoT 问题上仍显吃力**：尽管有所改进，*Azure2089* 观察到 **新版 Claude** 仍然无法解决那些通过 **Chain of Thought (CoT)** 推理可以轻松处理的问题。
  
  - 这引发了关于 Claude 在特定问题解决场景中局限性的持续讨论。

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1298007861418266716) (11 条消息🔥):

> - `Research Paper Trustworthiness` (研究论文的可信度)
> - `Fine-Tuning Models` (模型微调)
> - `Falsification in Scientific Research` (科学研究中的造假)
> - `Simple Arithmetic for Language Models` (语言模型的简单算术)
> - `AdamW Optimization Techniques` (AdamW 优化技术)

- **1/7 的研究论文被认为不可信**：一项新研究得出结论，大约 **1/7 的研究论文包含严重错误**，导致其不可信。作者强调，用于评估造假的传统方法仍然资金不足，并呼吁政府给予更多关注。
  
  - 成员们讨论了这一发现的影响，指出许多研究人员可能会在不知情的情况下，基于先前有缺陷的研究得出的错误结论进行后续研究。
- **Fine-Tuning 模型的复杂性**：成员们辩论了对大型基础模型进行 Fine-Tuning 的有效性，认为这可能会降低通用能力，以换取特定的目标格式。一位成员推论，Fine-Tuning 需要精细的 Hyperparameter 优化才能获得最佳结果。
  
  - 讨论中提出了关于 Fine-Tuning 艺术性及其缺乏公认社区知识的担忧，引发了对自一年前以来相关进展的好奇。
- **关于科学诚信的争议性观点**：一位成员分享了关于研究人员有时会忽略与同行评审结果不一致的数据的观察，暗示他们可能会重复实验，直到数据符合已建立的共识。这突显了科学研究中潜在的偏见 (Bias) 问题。
  
  - 对话涉及了在研究社区内确保诚信和准确性所面临的持续挑战。
- **语言模型的简单算术**：一位成员提出了一个新颖的想法，即未来的 **Language Models 可能会利用有限域 (Finite Fields) 上的基本算术运算**，而不是传统的浮点运算 (Floating-point Computations)。他们引用了一项研究，该研究表明一种新算法可以大幅降低张量处理 (Tensor Processing) 中的能耗。
  
  - 对话激发了人们对这种模型架构进步的可行性和影响的兴趣。
- **AdamW 优化技术**：在关于优化方法的讨论中，特别关注了 AdamW 及其变体（如 Schedule-free 版本），强调了它们与传统方法相比的性能。成员们注意到了在优化这些算法方面的持续研究。
  
  - 这些新方法的有效性仍然是社区内关注和探索的话题。

**提到的链接**：

- [Addition is All You Need for Energy-efficient Language Models](https://arxiv.org/abs/2410.00907)：大型神经网络的大部分计算都花在浮点张量乘法上。在这项工作中，我们发现浮点乘法器可以用一个高精度的整数加法器来近似...
- [Towards an Improved Understanding and Utilization of Maximum Manifold Capacity Representations](https://arxiv.org/abs/2406.09366)：Maximum Manifold Capacity Representations (MMCR) 是最近的一种多视图自监督学习 (MVSSL) 方法，其表现优于或持平于其他领先的 MVSSL 方法。MMCR 引起关注是因为它并不...
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)：学习表征是现代机器学习系统的核心组件，服务于众多的下游任务。在训练此类表征时，通常会出现计算和统计...
- [Implicit Bias of AdamW: $\ell_\infty$ Norm Constrained Optimization](https://arxiv.org/abs/2404.04454v1)：带有解耦权重衰减的 Adam，也称为 AdamW，因其在语言建模任务中的卓越性能而广受赞誉，在泛化方面超越了带有 $\ell_2$ 正则化的 Adam...
- [The Road Less Scheduled](https://arxiv.org/abs/2405.15682)：现有的不需要指定优化停止步数 T 的学习率调度方案，其表现远不如依赖于 T 的学习率调度方案。我们提出了一种方法...

---

### **Nous Research AI ▷ #**[**interesting-links**](https://discord.com/channels/1053877538025386074/1132352574750728192/1298050141588160553) (15 条消息🔥):

> - `Poe 多模型访问`
> - `关于生成艺术的 Machine Talks`
> - `新 ASR 模型发布`
> - `用于聊天记录所有权的 ZK 证明`
> - `Mina 与 OpenBlock 的对比`

- **Poe 支持访问多个模型**：一位成员询问关于使用 [Poe](https://poe.com/) 访问 **ChatGPT** 和 **Claude 3 Opus** 等各种模型的问题。
  
  - 回复各异，其中一位成员表示他们有时会使用 **OpenRouter**。
- **Machine Talks 探索生成艺术**：一位成员分享了他们由 AI 主持的脱口秀节目 **Machine Talks** 的启动，该节目采访了不同的模型，其中 **Capybara** 是最受欢迎的一个，点击[此处](https://art-dialogues.com/machine-talks/)观看试播集。
  
  - 他们还在 **Vimeo** 上提供了一个预告片链接以提供更多背景信息。
- **快速 ASR 模型 'Moonshine' 发布**：一款名为 [Moonshine](https://github.com/usefulsensors/moonshine) 的新型最先进自动语音识别 (**ASR**) 模型已发布，适用于边缘设备。
  
  - 该项目旨在实现**快速**且**准确**的性能，展示了在边缘设备应用中的潜力。
- **ZK 证明赋予 ChatGPT 用户聊天记录所有权**：一位成员介绍了使用 **ZK 证明**让用户拥有自己的 **ChatGPT** 聊天记录，旨在丰富开源模型的训练数据，点击[此处](https://x.com/openblocklabs/status/1848805457290572199)查看演示。
  
  - 讨论中涉及了速度问题，一位成员指出，现在某些证明可以在不到一秒的时间内完成。
- **对比 Mina 和 OpenBlock 技术**：有人提出了关于 **OpenBlock** 与 **Mina** 对比的问题，一位成员强调了 **Mina** 的轻量级架构和开发者生态系统。
  
  - **Mina** 的体积非常小，仅为 **22kb**，且可以在移动设备上运行，使其成为一个极具吸引力的选择。

**提到的链接**：

- [来自 OpenBlock (@openblocklabs) 的推文](https://x.com/openblocklabs/status/1848805457290572199)：1/ 介绍 Proof of ChatGPT，这是构建在 OpenBlock 通用数据协议 (UDP) 上的最新应用。此数据证明 (Data Proof) 使用户能够掌握其 LLM 聊天记录的所有权，标志着一个重大的...
- [来自 Paul Sengh (@paulsengh) 的推文](https://x.com/paulsengh/status/1846657020868677931)：ZK 技术的发展速度令人难以置信——得益于 @zkemail 的基础设施，现在一些 UDP 证明只需不到一秒的时间。快来尝试：https://bridge.openblocklabs.com/
- [Poe - 快速、实用的 AI 聊天](https://poe.com/)：未找到描述
- [Machine Talks - 预告片](https://vimeo.com/1021629035)：由 Blase 创建，在 art-dialogues.com 观看试播集
- [免费 Hermes AI ai.unturf.com & uncloseai.js – Russell Ballestrini](https://russell.ballestrini.net/free-hermes-ai-unturf-com-uncloseai) ：未找到描述
- [GitHub - usefulsensors/moonshine: 适用于边缘设备的快速准确的自动语音识别 (ASR)](https://github.com/usefulsensors/moonshine)：适用于边缘设备的快速准确的自动语音识别 (ASR) - usefulsensors/moonshine

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1298007861418266716) (11 messages🔥):

> - `Research Paper Trustworthiness`
> - `Falsification of Scientific Data`
> - `Peer Review Concerns`
> - `Fine-tuning Models`
> - `Efficient Computation in Neural Networks`

- **七分之一的论文不可信**：一项新研究得出结论，大约 **1/7 的研究论文** 包含使其不可信的严重错误，正如摘要所述：*“已发表的论文中有 1/7 存在与不可信程度相当的严重错误。”*
  
  - 该研究的方法多种多样，并承认造假率可能因领域而异，作者呼吁在该领域投入更多资金。
- **科学诚信问题**：一位成员分享了轶事证据，表明科学家有时会省略数据或更改实验以符合既定共识，这可能会建立在旧研究的错误结论之上。
  
  - 这引发了人们对科学界同行评审结果可靠性的担忧。
- **关于 Fine-tuning 的理论观点**：一位成员对 Fine-tuning 会导致基础模型退化的观点表示怀疑，认为 Fine-tuning 更多是一门艺术，需要社区知识才能进行有效优化。
  
  - 他们对 Fine-tuning 挑战的潜在答案进行了推测，尽管对第三至第五个问题的细节尚不确定。
- **语言模型中的有限域算术**：一位成员认为，语言模型最终可能会利用有限域上的简单算术运算来构建，并展示了一篇讨论用整数加法器近似浮点乘法的论文。
  
  - 提议的 **L-Mul algorithm** 据报道在实现张量操作高精度的同时，大幅降低了能源成本。
- **对科学伦理和造假的担忧**：成员们强调了科学家数据造假的历史问题，一项旧研究表明 **2%** 的人承认伪造过数据，而现在这被认为是低估了问题的严重性。
  
  - 这提高了人们对潜在科学不端行为以及准确评估研究诚信所面临挑战的认识。

**提到的链接**：

- [Addition is All You Need for Energy-efficient Language Models](https://arxiv.org/abs/2410.00907)：大型神经网络的大部分计算都消耗在浮点张量乘法上。在这项工作中，我们发现浮点乘法器可以用一个高精度的整数加法器来近似……
- [Towards an Improved Understanding and Utilization of Maximum Manifold Capacity Representations](https://arxiv.org/abs/2406.09366)：Maximum Manifold Capacity Representations (MMCR) 是最近的一种多视图自监督学习 (MVSSL) 方法，其效果达到或超过了其他领先的 MVSSL 方法。MMCR 引起关注是因为它并不……
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)：学习到的表示是现代机器学习系统的核心组件，服务于众多的下游任务。在训练此类表示时，通常会出现计算和统计上的……
- [Implicit Bias of AdamW: $\ell_\infty$ Norm Constrained Optimization](https://arxiv.org/abs/2404.04454v1)：具有解耦权重衰减的 Adam（也称为 AdamW）因其在语言建模任务中的卓越性能而广受赞誉，在泛化方面超越了具有 $\ell_2$ 正则化的 Adam……
- [The Road Less Scheduled](https://arxiv.org/abs/2405.15682)：现有的不需要指定优化停止步数 T 的学习率调度方案，其表现远不如依赖于 T 的学习率调度方案。我们提出了一种方法……

---

### **LM Studio ▷ #**[**announcements**](https://discord.com/channels/1110598183144399058/1111797717639901324/1298322162414915707) (1 条消息):

> - `LM Studio v0.3.5 features`
> - `Headless mode`
> - `On-demand model loading`
> - `Pixtral support on Apple MLX`
> - `New CLI command to download models`

- **LM Studio v0.3.5 带来令人兴奋的新特性**：最新更新 LM Studio v0.3.5 引入了诸如作为本地 LLM 服务运行的 **headless mode** 以及 **on-demand model loading** 等特性。
  
  - 用户现在可以使用 CLI 命令 `lms get` 轻松下载模型，简化了模型获取流程。
- **增强的 Apple MLX Pixtral 支持**：用户现在可以在至少配备 **16GB RAM**（建议 **32GB**）的 **Apple Silicon Macs** 上使用 **Pixtral**。
  
  - 得益于针对 Apple 硬件能力的优化，此次集成提升了性能。
- **提升用户体验的 Bug 修复**：0.3.5 版本解决了多个 Bug，包括 **RAG reinjecting documents** 的问题，并修复了 Mission Control 中闪烁的轮廓线。
  
  - Mac 用户还将受益于对 **sideloading quantized MLX models** 的增强支持。
- **社区模型亮点与招聘启事**：LM Studio 展示了像 **Granite 3.0** 这样的社区模型，该模型因其响应多样化查询的能力而受到关注。
  
  - 此外，他们正在招聘一名 **TypeScript SDK Engineer**，以促进设备端 AI 应用的开发。
- **全平台下载链接**：该更新适用于 **macOS**、**Windows** 和 **Linux**，并提供了各平台的具体下载链接。
  
  - 建议用户从其 [官方下载页面](https://lmstudio.ai/download) 获取最新版本。

**提到的链接**：

- [lmstudio-community/granite-3.0-2b-instruct-GGUF · Hugging Face](https://huggingface.co/lmstudio-community/granite-3.0-2b-instruct-GGUF)：未找到描述
- [LM Studio 0.3.5](https://lmstudio.ai/blog/lmstudio-v0.3.5)：Headless mode、on-demand model loading、服务器自动启动、从终端下载模型的 CLI 命令，以及对 Apple MLX 上的 Pixtral 的支持。
- [LM Studio (@LMStudioAI) 的推文](https://x.com/LMStudioAI/status/1848763292191199342)：LM Studio v0.3.5 发布了！👻🎃🥳 - Headless mode（作为本地 LLM 服务运行） - On-demand model loading - 支持基于 🍎MLX 的 @MistralAI Pixtral - 通过 `lms get` 从终端下载模型...

---

### **LM Studio ▷ #**[**general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1298009744375812136) (171 messages🔥🔥):

> - `GPU Offloading Issues`
> - `Model Loading Errors`
> - `AI Model Performance Metrics`
> - `ML Studio Features and Settings`
> - `Game Image Enhancers`

- **ML Studio 中的 GPU Offloading 问题**：一位用户报告称 GPU Offloading 导致性能大幅下降，目前仅使用约 4.2GB，而非之前的 15GB。
  
  - 在切换到旧版本的 ROCm 运行时版本后，性能恢复到正常水平，这表明更新可能影响了 GPU 利用率。
- **模型加载错误与系统资源**：另一位用户在调整 GPU Offload 设置后遇到了“由于系统资源不足，模型加载已中止”的错误。
  
  - 有人指出关闭加载保护机制 (guardrails) 可能会解决该问题，尽管通常不建议这样做。
- **AI 模型的性能指标**：用户讨论了使用吞吐量和延迟指标来衡量性能，加载设置会显著影响整体速度。
  
  - 在高强度的 GPU Offloading 下，吞吐量降至 0.9t/s，表明配置不当时可能存在效率低下的问题。
- **关于游戏图像增强器的咨询**：一位用户询问了将游戏图像增强为写实艺术的可用选项，建议使用 Stable Diffusion。
  
  - 这引发了关于各种工具及其在将游戏画面修改为高质量视觉效果方面的有效性的讨论。
- **ML Studio 中的模型配置意识**：一些用户对他们的模型根据 GPU 配置和量化 (quantization) 设置可以利用多少层感到困惑。
  
  - 讨论内容包括系统 RAM 如何模拟 VRAM，从而影响模型推理 (inference) 期间的性能指标和加载时间。

**提到的链接**：

- [mlx-community/xLAM-7b-fc-r · Hugging Face](https://huggingface.co/mlx-community/xLAM-7b-fc-r)：未找到描述
- [config.json · mlx-community/Mamba-Codestral-7B-v0.1-4bit at main](https://huggingface.co/mlx-community/Mamba-Codestral-7B-v0.1-4bit/blob/main/config.json#L39)：未找到描述
- [TheBloke/meditron-70B-GGUF · Hugging Face](https://huggingface.co/TheBloke/meditron-70B-GGUF)：未找到描述
- [LM Studio - Experiment with local LLMs](https://lmstudio.ai)：在你的电脑上本地运行 Llama, Mistral, Phi-3。
- [AP Workflow 11.0 for ComfyUI | Alessandro Perilli](https://perilli.com/ai/comfyui/#lmstudio)：解锁工业规模的生成式 AI，适用于企业级和消费级应用。

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1298005976393449523) (155 messages🔥🔥):

> - `Anthropic Claude 3.5`
> - `Mochi 1 Video Generation`
> - `CrewAI Series A Fundraising`
> - `Stable Diffusion 3.5 Release`
> - `Outlines Library Rust Port`

- **Anthropic 发布 Claude 3.5**：Anthropic 推出了升级版的 Claude 3.5 Sonnet 和新模型 Claude 3.5 Haiku，其中包括一项用于计算机使用 (computer use) 的 Beta 功能，使模型能够像人类一样与计算机交互。
  
  - 尽管具有创新能力，一些用户发现它在遵循 Prompt 方面效果不佳，导致在实际应用中的体验褒贬不一。
- **Mochi 1 树立视频生成新标准**：GenmoAI 推出了 Mochi 1，这是一款最先进的开源视频生成模型，专注于高质量、逼真的动态效果和细致的 Prompt 遵循能力。
  
  - Mochi 1 专为写实视频生成而设计，目前运行分辨率为 480p，并利用大量资金来加强开发。
- **CrewAI 获得 A 轮融资**：CrewAI 在由 Insight Partners 领投的 A 轮融资中筹集了 1800 万美元，旨在通过其开源框架改变企业的自动化。
  
  - 该公司声称每月执行超过 1000 万个 Agent，为大部分财富 500 强公司提供服务。
- **Stable Diffusion 3.5 发布**：Stability AI 揭晓了 Stable Diffusion 3.5，这是他们迄今为止最强大的模型，具有可定制性并兼容消费级硬件，同时对商业用途免费。
  
  - 用户可以通过 Hugging Face 访问 Stable Diffusion，并期待很快发布更多变体。
- **Outlines 库 Rust 移植版发布**：Dottxtai 宣布发布 Outlines 结构化生成的 Rust 移植版，旨在为各种应用提供更快的编译速度和轻量级库。
  
  - 此次更新提高了开发人员处理结构化生成的效率，并提供多种编程语言的绑定。

**提到的链接**：

- [来自 Anthropic (@AnthropicAI) 的推文](https://x.com/anthropicai/status/1848742740420341988?s=46)：推出升级版的 Claude 3.5 Sonnet，以及新模型 Claude 3.5 Haiku。我们还推出了一项处于 Beta 阶段的新功能：computer use。开发者现在可以指示 Claude 像人类一样使用电脑...
- [来自 Michele Catasta (@pirroh) 的推文](https://x.com/pirroh/status/1848752337080488177?s=46)：我无法形容上一次看到如此令人兴奋的新 AI 功能是什么时候了。我们将 Claude 的 computer use 接入了 @Replit Agent，作为人类反馈的替代方案。而且……它真的有效！我感...
- [对 Anthropic 新功能 Computer Use 的初步探索](https://simonwillison.net/2024/Oct/22/computer-use/)：Anthropic 今天发布了两项重大公告：新的 Claude 3.5 Sonnet 模型和一种被称为 computer use 的新 API 模式。（他们还预告了 Haiku 3.5，但那是……）
- [Intuit 要求我们删除这段 Decoder 节目内容](https://www.theverge.com/2024/10/21/24273820/intuit-ceo-sasan-goodarzi-turbotax-irs-quickbooks-ai-software-decoder-interview)：Intuit 首席执行官 Sasan Goodarzi 声称，有关该公司游说反对免费报税的说法并不准确。
- [来自 Genmo (@genmoai) 的推文](https://x.com/genmoai/status/1848762405779574990)：推出 Mochi 1 预览版。开源视频生成领域的新 SOTA。Apache 2.0 协议。magnet:?xt=urn:btih:441da1af7a16bcaa4f556964f8028d7113d21cbb&dn=weights&tr=udp://tracker.opentrackr.org:1337/annou...
- [来自 swyx (@swyx) 的推文](https://x.com/swyx/status/1848793186220794302)：一些用人类语言表述的实用 Token 数量：基准值 ~3 个单词约 4 个 Token ~每年 50 万分钟 ~一生 5000 万分钟 文本正常人类阅读：每分钟 400 Token，每天 6 小时，即 15 万 Token/...
- [来自 Rhymes.AI (@rhymes_ai_) 的推文](https://x.com/rhymes_ai_/status/1848554123471544711?s=46)：Rhymes 的新模型！✨ 我们很高兴地宣布 Allegro —— 一个小型且高效的开源 text-to-video 模型，可将您的文本转换为令人惊叹的 6 秒视频，帧率为 15 FPS，分辨率为 720p！...
- [来自 Stability AI (@StabilityAI) 的推文](https://x.com/StabilityAI/status/1848729212250951911)：推出 Stable Diffusion 3.5，我们迄今为止最强大的模型。此次开源发布包含多个变体，针对其尺寸具有高度可定制性，可在消费级硬件上运行，且对两者均免费...
- [IBM Granite 3.0：开源、最先进的企业级模型](https://www.ibm.com/new/ibm-granite-3-0-open-state-of-the-art-enterprise-models)：宣布推出 IBM Granite 3.0，这是一系列大语言模型 (LLMs) 和工具，包括 Granite 3.0 8B 和 2B、Granite Guardian 以及 Granite 3.0 MoE 模型。
- [来自 swyx (@swyx) 的推文](https://x.com/swyx/status/1848772118328316341)：因为模型卡显示了 Sonnet 和 Haiku 的提升，我们也可以顺便推测 3.5 Opus 的提升。引用 swyx (@swyx)：笑死，Anthropic 吸收了 @AdeptAILabs 却甚至懒得给...
- [来自 Genmo (@genmoai) 的推文](https://x.com/genmoai/status/1848762410074542278)：我们很期待看到你用 Mochi 1 创作的作品。我们很高兴地宣布获得了由 @NEA, @TheHouseVC, @GoldHouseCo, @WndrCoLLC, @parasnis, @amasad, @pirroh 等领投的 2840 万美元 A 轮融资。
- [Rovo：利用 GenAI 解锁组织知识 | Atlassian](https://www.atlassian.com/software/rovo)：Rovo 是 Atlassian 的新 GenAI 产品，通过 Rovo Search、Rovo Chat 和专门的 Rovo Agents 帮助团队利用组织知识采取行动。
- [Ideogram Canvas, Magic Fill, and Extend](https://about.ideogram.ai/canvas)：Ideogram Canvas 是一个无限的创意画布，用于组织、生成、编辑和组合图像。将您的面部或品牌视觉效果带入 Ideogram Canvas，并使用行业领先的 Magic Fill 和 Ext...
- [Launch YC: Manicode：让你的终端为你编写代码 | Y Combinator](https://www.ycombinator.com/launches/M2Q-manicode-make-your-terminal-write-code-for-you)：“非常 TM 酷”的 AI 代码生成 CLI
- [Aider LLM 排行榜](https://aider.chat/docs/leaderboards/)：LLM 代码编辑能力的定量基准测试。
- [来自 Aravind Srinivas (@AravSrinivas) 的推文](https://x.com/AravSrinivas/status/1846289701822677441)：Perplexity Finance：实时股票价格、深入研究公司财务状况、比较多家公司、研究对冲基金的 13f 等。UI 简直太棒了！
- [来自 Paul Gauthier (@paulgauthier) 的推文](https://x.com/paulgauthier/status/1848795903693336984)：新的 Sonnet 以 84.2% 的成绩登顶 Aider 代码编辑排行榜。使用 --architect 模式，它与 DeepSeek 作为编辑器模型配合，创下了 85.7% 的 SOTA。尝试一下：`pip install -U aider-chat` ...
- [来自 Kenneth Auchenberg 🛠 (@auchenberg) 的推文](https://x.com/auchenberg/status/1848427656598970387)：从收入来源来看，@AnthropicAI 表现为一家基础设施厂商，而 @OpenAI 的运作方式更像是一家面向消费者的公司。看看几年后每家公司的处境将会很有趣...

- [Anthropic (@AnthropicAI) 的推文](https://x.com/AnthropicAI/status/1848742747626226146): 全新的 Claude 3.5 Sonnet 是首个在公开测试版中提供 computer use 功能的前沿 AI 模型。虽然具有开创性，但 computer use 仍处于实验阶段——有时容易出错。我们正在提前发布它...
- [来自 .txt (@dottxtai) 的推文](https://x.com/dottxtai/status/1848783015222169726): 我们一直与 @huggingface 合作，刚刚发布了 Outlines 结构化生成的 Rust 移植版本。👉 更快的编译速度 👉 轻量级库（艾特 @vllm_project）👉 多种语言的绑定...
- [João Moura (@joaomdmoura) 的推文](https://x.com/joaomdmoura/status/1848739310159139161): 很高兴分享 @crewAIInc 筹集了 1800 万美元的资金，我们的 A 轮融资由 @insightpartners 领投，@Boldstartvc 领投了我们的种子轮。我们也很高兴欢迎 @BlitzVent...
- [Threads 上的 Jacky Liang (@jjackyliang)](https://www.threads.net/@jjackyliang/post/DBb5UgIxqhf?xmt=AQGz-YYmFmjzdo_5BBnv3gzMj_4NRw1DDf3ksYXW4L5zow): Cursor 已更新至新的 Claude Sonnet 3.5！它似乎是自动启用的，但你也可以手动关闭（旧的）Claude Sonnet 3.5。不过命名方案有点奇怪...
- [Reddit - 深入了解任何事物](https://www.reddit.com/r/ClaudeAI/comments/1g94a2v/did_claude_just_get_a_super_boost/): 未找到描述
- [Alex Albert (@alexalbert__) 的推文](https://x.com/alexalbert__/status/1848777260503077146): 关于我们开发 computer use 期间的一个有趣故事：我们举行了一次工程 bug bash，以确保发现 API 的所有潜在问题。这意味着将几名工程师带到一个房间里...
- [Genmo。最好的开源视频生成模型。](https://www.genmo.ai/): Genmo 训练世界上最好的开源视频生成模型。在 Genmo 使用 AI 创作令人惊叹的视频
- [Alex Albert (@alexalbert__) 的推文](https://x.com/alexalbert__/status/1848743018075189719): 我很高兴分享我们最近在 Anthropic 的工作成果。- Computer use API - 新的 Claude 3.5 Sonnet - Claude 3.5 Haiku 让我们一起来看看：
- [GitHub - genmoai/models: 最好的 OSS 视频生成模型](https://github.com/genmoai/models): 最好的 OSS 视频生成模型。通过在 GitHub 上创建账号为 genmoai/models 的开发做出贡献。
- [GitHub 的推文 - FixTweet/FxTwitter: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等](https://x.com/tomas_hk)): 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等 - FixTweet/FxTwitter
- [Not Diamond 如何通过 LLM 路由引擎节省 75 万美元 · Zoom · Luma](https://lu.ma/x6ll1wxt): 适合开发者、构建者以及任何希望通过更智能的 LLM 路由来简化 LLM 部署、降低成本并提高性能的人。📅 : 2024 年 10 月 22 日 ⏰ :…

---

### **Notebook LM Discord ▷ #**[**use-cases**](https://discord.com/channels/1124402182171672732/1124403655819415592/1297997949040594985) (45 条消息🔥):

> - `NotebookLM 实验`
> - `播客制作`
> - `语言学习`
> - `WallStreetBets 分析`
> - `AI 生成内容`

- **使用 NotebookLM 制作引人入胜的播客**：一位成员分享了他们使用 NotebookLM 制作每日播客的经验，分析了来自 [WallStreetBets subreddit](https://youtu.be/ZjN0wMKF_ZA) 的最新谈话和情绪。他们讨论了如何输入 Reddit 的热门帖子来分析趋势话题和股票走势。
  
  - 另一位用户展示了他们使用 NotebookLM 进行的深度探索，包括通过调整 Prompt 制作更长的播客剧集，并为中级语言学习者推荐了高级用例。
- **行为艺术中的荒诞幽默**：行为艺术家 Crank Sturgeon 使用 NotebookLM 的播客功能进行了一项实验，创作了一段荒诞的音频作品，可在 SoundCloud ([Unentitled Notbook](https://soundcloud.com/user6719786/unentitled-notbook)) 上收听。这展示了 AI 生成内容在幽默和实验性方面的潜力。
  
  - 讨论强调了利用 NotebookLM 进行创意和喜剧叙事的有趣可能性。
- **AI 驱动的诗歌朗诵**：成员们对使用 NotebookLM 进行诗歌的戏剧性朗诵表现出兴趣，并引用了 Edgar Allan Poe 的《乌鸦》（*The Raven*）等例子。一位用户指出，通过生成多个音频概览并拼接最佳片段，可以获得令人震撼的效果。
  
  - 这表明了一种通过 AI 探索文学作品的趋势，旨在打造迷人的听觉体验。
- **AI 在语言学习中的创新应用**：一位用户介绍了一种语言学习的“深度探索”方法，即用目标语言写作并获得 AI 专家的纠正。该模式针对中级学习者，鼓励互动式语言练习。
  
  - 这种方法启发了其他人将 AI 视为语言学习中的私人导师，从而提高参与度和熟练度。
- **对 AI 生成内容的思考**：成员们分享了利用 NotebookLM 进行播客和内容创作的反馈，表达了其中的挑战与成功。一位成员提到如何将冗长的传记浓缩成 12 分钟的播客剧集，突显了该工具的高效性。
  
  - 这段对话强调了社区在应对 AI 局限性的同时，对利用 AI 进行内容生产的热情。

**提到的链接**：

- [未找到标题](https://notebooklm.google.com/notebook/863f7546-651c-4814-8bd8-8225c54e0d43/audio)：未找到描述
- [NotebooLM 视频：AI 播客转 AI 视频](https://notebooklm.video)：未找到描述
- [NotebookLM 尝试 Sleep Token](https://www.youtube.com/watch?v=B5JyKftj5vI)：我让 NotebookLM 分析了 Sleep Token 全部五张专辑的歌词，它的表现……还算过得去。
- [VotebookLM - 在 AI 的帮助下成为明智的选民](https://youtu.be/MvF1OBbMjyc)：NotebookLM 是一种上传主题信息并在 AI 帮助下查找其中信息的方式。在这个视频中，你将看到如何使用它来创建……
- [Unentitled Notbook](https://soundcloud.com/user6719786/unentitled-notbook?si=ba9e7985cb864b41a35b86d54385ce8f&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing&fbclid=IwY2xjawF6nhhleHRuA2FlbQIxMQABHeTrlIr8PQrBLCuljCedPULUWcPZf8dBoLJv6iIoUeu3F_r-MKbnX1MdZQ_aem_gd_EkobaHOKb-CMEV1uY2w)：https://cranksturgeon.com/tentil.pdf
- [未找到标题](https://notebooklm.google.com/notebook/245692ed-9a2b-4396-89b3-44b04bf24b0b/audio)：未找到描述
- [《东京梦华录》](https://www.youtube.com/watch?v=5Krvvc9jE7Y)：关于《东京梦华录》的讨论。
- [每周深度探索 2024年10月21日](https://youtu.be/A0-oZBgomuU)：2025年每股收益增长、中国刺激政策、收益率曲线、电动汽车价格
- [WallStreetBets 每日播客 - 2024年10月22日](https://youtu.be/ZjN0wMKF_ZA)：NVDA 冲向月球！🚀（但要小心 Theta 损耗！）😱 加入我们，一起剖析今日疯狂的市场行情！NVDA 的疯狂涨势！到 11 月真的能达到 200 美元吗……
- [GitHub - mandolyte/discord-notebooklm: 聊天导出分析](https://github.com/mandolyte/discord-notebooklm)：聊天导出分析。通过在 GitHub 上创建账号为 mandolyte/discord-notebooklm 的开发做出贡献。
- [Reddit - 深入探索一切](https://www.reddit.com/r/notebooklm/comments/1g9d3ub/ai_discovery_once_you_hear_this_youll_never/)：未找到描述
- [虚幻谜团 3：木星的云中泳者](https://www.youtube.com/watch?v=WPrFCGWFDrw)：David 和 Hannah 带你前往木星，去见识居住在那里的悬浮外星太空鲸鱼。了解这与 1972 年以及 Rove 有什么关系……
- [DeepDive](https://www.spreaker.com/organization/deepdive--13017306)：在海量信息中迷失很容易，但找到那些智慧的结晶让一切都变得值得。🌟

---

### **Notebook LM Discord ▷ #**[**general**](https://discord.com/channels/1124402182171672732/1124402182909857966/1297998481692299336) (103 条消息🔥🔥):

> - `NotebookLM 语言设置问题`
> - `共享与协作挑战`
> - `多语言音频概览 (Audio Overviews)`
> - `播客制作体验`
> - `文档上传问题`

- **NotebookLM 中的语言混淆**：用户报告称，尽管提供了英文文档，NotebookLM 的回答仍默认显示为荷兰语，一些建议指出应调整 Google 账号的语言设置。
  
  - 一位用户发现很难获得一致的德语结果，有时生成的音频会出现意想不到的“外星”方言。
- **共享笔记本遇到麻烦**：几位用户对无法共享其笔记本表示沮丧，在尝试共享时一直处于“Loading...”加载界面。
  
  - 这一问题引发了对该工具功能的担忧，使其对于寻求协作的用户来说变得无效。
- **多语言音频概览效果参差不齐**：用户尝试创建各种语言的音频概览，并注意到发音和地道程度存在不一致，尤其是在荷兰语方面。
  
  - 尽管面临挑战，但由于一些用户成功生成了荷兰语音频内容，人们对未来多语言支持的改进持乐观态度。
- **播客创作体验**：一位用户分享了成功上传 90 页区块链课程的兴奋之情，而其他用户则讨论了生成音频的趣味性，称其“非常滑稽”。
  
  - 几位用户反馈了某些指令如何导致意想不到或有趣的输出，表明结果因输入而异。
- **文档上传问题**：用户遇到了文档未在 Google Drive 中显示以及上传文件处理延迟的问题。
  
  - 讨论中提到了潜在原因（如文件损坏），并建议通过刷新操作来解决问题。

**提到的链接**：

- [Account settings: Your browser is not supported.](https://myaccount.google.com/language)：未找到描述
- [NotebookLM for Lesson Planning at Meshed/XQ's 2024 AI+EDU Symposium at Betaworks](https://www.youtube.com/watch?v=TPJKhZM0O5U)：未找到描述
- [Frequently Asked Questions - Help](https://support.google.com/notebooklm/answer/14278184?hl=en)：未找到描述
- [DeepDive](https://www.spreaker.com/organization/deepdive--13017306)：在海量信息中很容易迷失，但找到那些智慧的火花让一切都变得值得。🌟
- [AI+ EDU Symposium: INTRO Google Labs Editorial Director & Science Author, Steven Johnson](https://youtu.be/ROdujIR-A2M?si=3_UWy8MppO4loTAM)：未找到描述
- [AI+ EDU Symposium: Q&A Google Labs Editorial Director & Science Author, Steven Johnson, NotebookLM](https://www.youtube.com/watch?v=Jcxvd8ZAIS0)：未找到描述

---

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1298002642420629564) (103 条消息🔥🔥):

> - `新的 AI 模型和功能`
> - `API 功能和用户关注点`
> - `Perplexity 的竞争和市场地位`
> - `用户体验和反馈`
> - `支持和功能查询`

- **对新 AI 模型的兴奋**：用户正在热烈讨论最近推出的 **Claude 3.5 Sonnet** 和 **Claude 3.5 Haiku**，一些人表示希望它们能尽快集成到 Perplexity 中。
  
  - 参阅来自 [AnthropicAI](https://x.com/AnthropicAI/status/1848742740420341988) 的公告，详细介绍了新功能，包括指挥 Claude 像人类一样使用电脑（computer use）。
- **API 功能问题**：用户报告了 Perplexity API 的问题，特别是当在 Prompt 中请求时，它无法返回来源的 URL。
  
  - 一位用户提到，尽管在 Prompt 中有明确指令，但他们仍难以获取完整的 URL，这引发了关于有效使用方法的咨询。
- **Perplexity 面临竞争**：成员们注意到 **Yahoo** 推出了 AI 聊天服务，引发了关于 Perplexity 在市场中竞争优势的讨论。
  
  - 尽管有这些发展，用户对 Perplexity 的能力仍保持信心，强调其可靠性和资源丰富性。
- **Perplexity 的用户体验**：多位用户称赞 Perplexity 能够持续提供高质量信息和可靠来源，表达了对该服务的满意。
  
  - 一位用户评论道：“我太喜欢 PAI 了！我在工作和生活中一直在用它”，反映了整体的积极情绪。
- **支持和故障排除方面的挫败感**：用户对支持响应速度以及在各种问题上联系客服的困难表示不满。
  
  - 一位用户对报告问题的挑战表示担忧，质疑支持渠道的效率。

**提到的链接**：[来自 Anthropic (@AnthropicAI) 的推文](https://x.com/AnthropicAI/status/1848742740420341988)：推出升级版的 Claude 3.5 Sonnet 以及新模型 Claude 3.5 Haiku。我们还推出了一项处于 Beta 阶段的新功能：computer use。开发者现在可以指挥 Claude 像……一样使用电脑。

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1298079895808638977) (11 条消息🔥):

> - `大学路径和学位`
> - `Snapdragon 8 Elite 概览`
> - `Galaxy Z Fold 特别版`
> - `金价上涨`
> - `AI 驱动的事实核查`

- **大学路径探索**：一位用户在 [Perplexity](https://www.perplexity.ai/search/please-compare-a-math-major-to-Pz8ZGcMWR.2vS3cdLBXC7g) 上分享了一个将数学专业与其他大学学位进行比较的资源，强调了其在学术规划中的效用。
  
  - 该工具辅助用户对未来的教育规划做出明智决策。
- **Snapdragon 8 Elite 详解**：分享了一个关于 Snapdragon 8 Elite 的信息页面，在 [Perplexity](https://www.perplexity.ai/page/snapdragon-8-elite-explained-PqOaRL__RAiWtKUDcgBtng) 上详细解释了其功能及在科技行业的重要性。
  
  - 这种深度解析有助于利益相关者了解移动处理技术的进步。
- **Galaxy Z Fold 特别版发布**：讨论了 Galaxy Z Fold 特别版，在 [Perplexity](https://www.perplexity.ai/page/galaxy-z-fold-special-edition-HHKn46BTS22CYJLOeF5Wug) 上展示了其独特的功能和设计。
  
  - 该版本旨在吸引对高端折叠屏智能手机感兴趣的用户。
- **金价创历史新高**：可以在 [Perplexity](https://www.perplexity.ai/page/gold-s-record-high-KsT3E5EoSBGr0dkLKiYUkg) 找到关于近期金价创历史新高趋势的讨论。
  
  - 这种飙升背后的市场动态和经济因素值得潜在投资者关注。
- **AI 驱动的事实核查集合**：分享了一个致力于高级 AI 驱动事实核查策略的集合，强调了在该过程中 LLM 的使用和伦理考量，详见 [Perplexity](https://www.perplexity.ai/collections/advanced-ai-driven-fact-checki-a3cMcPR.QsKkCRZ79UKFLQ)。
  
  - 该资源讨论了来源可信度和偏见检测等关键方面，为改进虚假信息处理提供了见解。

---

### **Eleuther ▷ #**[**announcements**](https://discord.com/channels/729741769192767510/794042109048651818/1298034322003197974) (1 条消息):

> - `SAE interpretation pipeline`
> - `Evaluation techniques for explanations`
> - `Causal feature explanation`
> - `Feature alignment using Hungarian algorithm`
> - `Open-source tools for LLMs`

- **全新开源 SAE 解释流水线发布**：可解释性团队正在发布一个新的开源 [pipeline](https://github.com/EleutherAI/sae-auto-interp)，利用 LLMs 自身来自动解释 LLMs 中的 **SAE features** 和神经元。
  
  - 该计划引入了 **五种新技术** 用于评估解释质量，从而实现大规模的可解释性增强。
- **Causal Effects 带来更好的特征解释**：首次证明了可以基于引导特征的 **causal effect** 来生成特征解释，这与传统的基于上下文的方法有所不同。
  
  - 这种方法为以前被认为无法解释的特征提供了见解，标志着该领域的重大进展。
- **使用 Hungarian algorithm 对齐 SAE 特征**：团队发现可以使用 **Hungarian algorithm** 对齐 **不同 SAE 的特征**，重点关注那些在同一网络的不同层上训练的特征。
  
  - 该方法揭示了在 **residual stream** 相邻层上训练的 SAEs 表现出几乎相同的特征（MLPs 除外）。
- **大规模分析确认 SAE Latents 的可解释性**：他们的分析确认，**SAE latents** 的可解释性显著高于神经元，即使是使用 **top-k postprocessing** 进行稀疏化处理后的神经元也是如此。
  
  - 这一发现鼓励进一步探索 autoencoders 以实现 LLMs 更好的可解释性。
- **提供合作机会与资源**：鼓励感兴趣的合作者查看频道，了解与 SAE 解释项目相关的持续工作。
  
  - 团队感谢各位成员的贡献，并分享了诸如 [研究论文](https://arxiv.org/abs/2410.13928) 和 [Hugging Face 上的数据集](https://huggingface.co/datasets/EleutherAI/auto_interp_explanations) 等资源。

**提到的链接**：

- [Automatically Interpreting Millions of Features in Large Language Models](https://arxiv.org/abs/2410.13928)：虽然深度神经网络中神经元的激活通常没有简单的人类可理解的解释，但可以使用 sparse autoencoders (SAEs) 将这些激活转换为...
- [GitHub - EleutherAI/sae-auto-interp](https://github.com/EleutherAI/sae-auto-interp)：通过在 GitHub 上创建账户，为 EleutherAI/sae-auto-interp 的开发做出贡献。
- [EleutherAI/auto_interp_explanations · Datasets at Hugging Face](https://huggingface.co/datasets/EleutherAI/auto_interp_explanations)：未找到描述
- [来自 Nora Belrose (@norabelrose) 的推文](https://x.com/norabelrose/status/1848469111073886326)：@AiEleuther 可解释性团队正在发布一个新的开源流水线，用于使用 LLMs 自动解释 LLMs 中的 SAE 特征和神经元。我们还引入了五种新的、高效的技术...

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1298002645864419339) (28 条消息🔥):

> - `Non-archival workshops`
> - `Chess AI model integration`
> - `Chess move explainability`
> - `Stockfish analysis speed`
> - `Research goals in AI development`

- **应对 Non-archival Workshop 投稿**：只要规则允许，似乎可以将同一篇论文提交给多个 workshop，特别是那些不会干扰会议投稿的 non-archival 类型。
  
  - 然而，一些会议可能不接受之前在这些 workshop 上展示过的论文，因此核实各个会议的具体政策至关重要。
- **集成 Chess AI 与 LLM 以增强交互**：一位成员提议将国际象棋 AI 与 LLM 结合，以实现一个能够理解自身决策的对话模型，而不仅仅是简单的查询-响应设置。
  
  - 这种设计旨在创建一个更连贯的系统，使 Chess AI 的推理与其对话能力保持一致，从而能够对其走法进行更深入的对话。
- **象棋走法可解释性的复杂性**：讨论围绕棋手解释引擎所做出的顶级走法的能力展开，一些人认为许多走法通常被视为缺乏明确理由的“computer stuff（计算机风格）”。
  
  - 这凸显了人类理解与引擎逻辑之间的鸿沟，因为即使是专家在直播解说期间也可能难以将某些高评分走法合理化。
- **Stockfish 卓越的分析能力**：一位成员提到，某个版本的 Stockfish 每秒可以评估高达 **2800 万个节点**，这表明其在评估局面方面具有强大的分析能力。
  
  - 这一统计数据引发了对不同引擎能力的讨论，强调了现代象棋引擎令人印象深刻的性能。
- **明确 AI 集成的研究目标**：一位成员鼓励明确 AI 的研究目标，询问让机器人完成某些任务背后的动机。
  
  - 这一询问旨在揭示集成 Chess AI 预期带来的收获和益处，促进对其潜在用例的深入探索。

 

**提到的链接**：[How many positions per second should my homemade engine calculate?](https://chess.stackexchange.com/questions/30505/how-many-positions-per-second-should-my-homemade-engine-calculate): 我的程序打印了执行执行/撤回走法函数所花费的时间，两者合计平均为 00.0002 秒。这意味着我的引擎每秒最多可以分析 5000 个位置...

 

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1298018890185707541) (59 条消息🔥🔥):

> - `使用 RAG 实现 1B 上下文长度`
> - `SAEs 的鲁棒性`
> - `Transformer 模型中的 LayerNorm`
> - `独立研究发表经验`
> - `研究分享中的伦理问题`

- **1B 上下文长度的突破**：最近的一篇帖子讨论了一个检索系统，通过一种基于 **sparse graphs** 的新颖方法将 LLMs 的上下文长度扩展到了 **10 亿 (1B)**，在 Hash-Hop 基准测试中实现了 state-of-the-art 的性能。
  - 与传统的稠密嵌入 RAG 系统相比，该方法在计算和内存方面被认为**更高效**。
- **SAE 项目想法讨论**：一名本科生正在寻求与 **Sparse Autoencoders (SAEs)** 相关的项目想法，引发了关于该领域当前研究进展和资源链接的讨论。
  - 一位成员分享了对协作项目的见解，并提供了一个 [Alignment Forum 帖子](https://www.alignmentforum.org/posts/CkFBMG6A9ytkiXBDM/sparse-autoencoders-future-work) 的链接以供进一步探索。
- **GPT2 中 LayerNorm 的移除**：一位成员分享了一篇帖子，强调了通过 Fine-tuning 从 GPT2 中移除 **LayerNorm** 的研究，展示了在有无 LayerNorm 的情况下基准测试中的细微性能差异。
  - 这项工作由 Apollo Research 完成，指出了 LayerNorm 给 **Mechanistic Interpretability** 带来的挑战。
- **独立研究者与论文发表**：讨论了独立研究者发表论文的可行性，强调如果工作足够优秀，确实可以被会议接收。
  - 成员们分享了个人经验，强调协作可以缓解研究过程中的挑战。
- **研究分享中的伦理担忧**：研究社区中出现了关于分享想法的伦理担忧，讨论了想法在未署名的情况下被挪用的案例。
  - 会议强调解决此类问题非常复杂，并鼓励成员报告任何此类事件以寻求支持。

**提到的链接**：

- [Transformer Architecture: The Positional Encoding - Amirhossein Kazemnejad's Blog](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)：未找到描述
- [Zyphra](https://www.zyphra.com/post/reaching-1b-context-length-with-rag)：未找到描述
- [Sparse Autoencoders: Future Work — AI Alignment Forum](https://www.alignmentforum.org/posts/CkFBMG6A9ytkiXBDM/sparse-autoencoders-future-work)：大部分是我写的，除了由 @Aidan Ewart 编写的“Better Training Methods”部分。…
- [You can remove GPT2’s LayerNorm by fine-tuning for an hour — LessWrong](https://www.lesswrong.com/posts/THzcKKQd4oWkg4dSP/you-can-remove-gpt2-s-layernorm-by-fine-tuning-for-an-hour)：这项工作由 Apollo Research 完成，基于最初在 MATS 进行的研究。编辑：arXiv 版本见 [https://arxiv.org/abs/2409.13710](https://arxiv.org/abs/2409.13710) …
- [You can remove GPT2’s LayerNorm by fine-tuning for an hour — LessWrong](https://www.lesswrong.com/posts/THzcKKQd4oWkg4dSP/you-can-remove-gpt2-s-layernorm-by-fine-tuning-for)：这项工作由 Apollo Research 完成，基于最初在 MATS 进行的研究。编辑：arXiv 版本见 [https://arxiv.org/abs/2409.13710](https://arxiv.org/abs/2409.13710) …

---

### **Eleuther ▷ #**[**interpretability-general**](https://discord.com/channels/729741769192767510/1052314805576400977/1298310476500959273) (2 条消息):

> - `Mech Interp 论文评分`
> - `分享研究`
> - `Twitter 影响力`

- **Woog09 为 ICLR 2025 的 Mech Interp 论文评分**：一位成员对提交给 ICLR 2025 的所有 Mech Interp 论文进行了评分，并分享了他们的 [电子表格](https://docs.google.com/spreadsheets/d/1TTHbONFo4OV35Bv0KfEFllnkP-aLGrr_fmzwfdBqBY0/edit?gid=0#gid=0)，评分标准清晰：**3** 代表优秀 (outstanding)，**2** 代表亮点 (spotlight)，**1** 代表有潜力 (promising)，未评分代表可能被忽视。
  - 他们强调了**经过校准的评分**，以帮助引导读者了解提交论文的质量。
- **呼吁更多研究分享**：一位成员表示希望有更多关于 Mech Interp 论文评分的分享，并指出这些评分在 Discord 等私密环境之外缺乏强大的影响力。
  - 他们旨在通过建立自己的 **Twitter 影响力** 来改变这一现状，并鼓励其他人帮助传播。

**提到的链接**：[Alice Rigg (@woog09) 的推文](https://x.com/woog09/status/1848703344405057587)：我为所有提交给 ICLR 2025 的 Mech Interp 论文评分了：[https://docs.google.com/spreadsheets/d/1TTHbONFo4OV35Bv0KfEFllnkP-aLGrr_fmzwfdBqBY0/edit?gid=0#gid=0](https://docs.google.com/spreadsheets/d/1TTHbONFo4OV35Bv0KfEFllnkP-aLGrr_fmzwfdBqBY0/edit?gid=0#gid=0)。评分经过校准：3 - 优秀...

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1298252159460180002) (8 条消息🔥):

> - `Batch Size 配置`
> - `Model Initialization 处理`

- **调试无 Batch 问题**：一名成员询问如何调试一个问题，即尽管设置了 `batch_size`，`requests` 仍然是一个包含所有 instance 的巨大列表。
  
  - 根据另一位成员的说法，*这似乎确实需要由模型（model）来处理*。
- **输入处理与 Batch Size**：同一位成员质疑，如果模型本身处理 `batch_size` 参数，输入是否就不会被分批（batched）。
  
  - *否则，如果不被正确利用，设置* `batch_size` 参数就毫无意义，这凸显了对该功能的困惑。
- **Model Initialization 的作用**：针对这些疑虑，一名成员澄清说 `batch_size` 会被传递给 Model Initialization。
  
  - 这一澄清让最初的提问者开始思考这种设置背后的基本原理。

---

### **Interconnects (Nathan Lambert) ▷ #**[**news**](https://discord.com/channels/1179127597926469703/1179128538679488533/1298138199045963807) (74 条消息🔥🔥):

> - `Allegro 模型发布`
> - `Stability AI 的 Stable Diffusion 3.5`
> - `Anthropic 的 Claude 3.5 Haiku`
> - `Computer Use API`
> - `新型视频生成模型`

- **Allegro 模型实现文本转视频**：Rhymes AI 发布了其全新的开源模型 **Allegro**，该模型能以 15 FPS 和 720p 分辨率将文本生成 6 秒视频，目前已通过包括 [GitHub repository](https://github.com/rhymes-ai/Allegro) 在内的多个链接开放体验。
  
  - 鼓励用户加入 [Discord 等候名单](https://forms.gle/JhA7BaKvZoeJYQU87)，成为首批试用 **Allegro** 的用户。
- **Stability AI 凭借 SD 3.5 升温**：**Stability AI** 发布了 **Stable Diffusion 3.5**，包含三个版本，年收入低于 100 万美元的企业可免费商用，并展示了用于定制化的 Query-Key Normalization 等高级特性。
  
  - **Large 版本** 现已在 Hugging Face 和 GitHub 上线，**Medium 版本** 预计将于 10 月 29 日发布。
- **Claude 3.5 Haiku 在编程领域树立高标杆**：Anthropic 推出了 **Claude 3.5 Haiku**，其性能超越了 Claude 3 Opus。它在编程任务中表现尤为出色，在 SWE-bench Verified 测试中得分 **40.6%**，可通过 [此处](https://docs.anthropic.com/en/docs/build-with-claude/computer-use) 的 API 获取。
  
  - 用户强调了其能力的显著提升，同时认可了该模型在各项基准测试中的卓越表现。
- **Computer Use API 的令人兴奋的进展**：Anthropic 的 **Computer Use API** 引起了轰动，用户正在尝试其新功能，包括指挥 Claude 在计算机上执行任务，引发了进一步测试的热潮。
  
  - 反馈突显了新 API 背后的功能性和趣味性，正如 [GitHub](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo) 上演示的那样。
- **新型视频生成模型的涌现**：**Mochi 1** 作为一款最先进的开源视频生成模型推出，延续了该领域与现有模型并行的创新趋势。
  
  - 讨论围绕 **Mochi** 和 **Sora** 等模型的快速发展展开，表明了视频生成技术领域的竞争态势。

**提到的链接**：

- [Paul Gauthier (@paulgauthier) 的推文](https://x.com/paulgauthier/status/1848795903693336984)：新的 Sonnet 在 aider 的代码编辑排行榜上名列前茅，得分为 84.2%。使用 --architect 模式并配合 DeepSeek 作为编辑器模型时，它创下了 85.7% 的 SOTA 纪录。尝试方法：pip install -U aider-chat ...
- [Genmo (@genmoai) 的推文](https://x.com/genmoai/status/1848762405779574990)：推出 Mochi 1 预览版。开源视频生成的新 SOTA。Apache 2.0。magnet:?xt=urn:btih:441da1af7a16bcaa4f556964f8028d7113d21cbb&dn=weights&tr=udp://tracker.opentrackr.org:1337/annou...
- [Aidan McLau (@aidan_mclau) 的推文](https://x.com/aidan_mclau/status/1848752392935809263)：呃，Claude 真的跟我一模一样 (fr)
- [Simon Willison (@simonw) 的推文](https://x.com/simonw/status/1848791371341304258)：看起来我今天早上对新 Computer Usage 模型的实验花费了略高于 4 美元
- [Tibor Blaho (@btibor91) 的推文](https://x.com/btibor91/status/1848727528799707187)：Stability AI 发布了 Stable Diffusion 3.5，这是他们迄今为止最强大的图像生成模型，提供三个具有不同能力的版本，年收入 100 万美元以下免费商用，具备高级功能...
- [介绍 computer use、全新的 Claude 3.5 Sonnet 以及 Claude 3.5 Haiku](https://www.anthropic.com/news/3-5-models-and-computer-use)：更新后的、更强大的 Claude 3.5 Sonnet，Claude 3.5 Haiku，以及一项新的实验性 AI 能力：computer use。
- [Rhymes.AI (@rhymes_ai_) 的推文](https://x.com/rhymes_ai_/status/1848554123471544711)：Rhymes 的新模型！✨ 我们激动地宣布 Allegro —— 一个小型且高效的开源文本转视频模型，能将您的文本转化为令人惊叹的 6 秒视频，15 FPS 且达到 720p！...
- [Simon Willison (@simonw) 的推文](https://x.com/simonw/status/1848758104076521681)：好的，这个新的 "computer use" API 玩起来简直太有趣了。你可以使用这个仓库中的 Docker 示例来启动它：https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-d...
- [𝑨𝒓𝒕𝒊𝒇𝒊𝒄𝒊𝒂𝒍 𝑮𝒖𝒚 (@artificialguybr) 的推文](https://x.com/artificialguybr/status/1848769004908761110)：笑死，什么鬼。Twitter/X.AI 自发布以来将 grok-2 的价格提高了一倍，哈哈

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-drama**](https://discord.com/channels/1179127597926469703/1181746144821387334/1298218525743382550) (7 messages):

> - `AI-generated papers` (AI 生成的论文)
> - `Viral content on social media` (社交媒体上的病毒式内容)
> - `Feedback mechanisms in tech communities` (技术社区的反馈机制)

- **AI 论文引发关注**：一位成员指出，目前在 Twitter 和 Hackernews 上流传的一篇热门论文很可能**完全是 AI 生成的**，并强调其中关于 **ORPO** 的章节是幻觉且错误的。
  
  - 批评指出，作者可能也不理解 **Odds Ratio Preference Optimization**。
- **Hackernews 是个噪音机器**：有人对 **Hackernews** 表示担忧，认为那里的讨论被视为一场**流量彩票**，缺乏作为反馈机制的真实价值。
  
  - 成员们形容该平台**非常嘈杂且带有偏见**，质疑其在社区参与方面的有用性。
- **对病毒式垃圾内容的批评**：成员们指责了某些病毒式的在线内容，明确将其描述为 **slop**（垃圾内容），并提到了与 LinkedIn 等平台的关联。
  
  - 这一评论反映了人们对这些渠道中分享和消费的信息质量日益增长的挫败感。

**提及的链接**：

- [Sam Paech (@sam_paech) 的推文](https://x.com/sam_paech/status/1848332471953448972)：@rohanpaul_ai 供参考，这篇论文至少部分、甚至可能全部是 AI 生成的。例如，关于 ORPO 的整个章节都是幻觉。模型显然不知道 ORPO 是什么（它实际上...）
- [Xeophon (@TheXeophon) 的推文](https://x.com/TheXeophon/status/1848660191140618348)：@Dorialexander 这篇垃圾论文里写的不是 LinkedIn 上的那些垃圾内容吗

---

### **Interconnects (Nathan Lambert) ▷ #**[**random**](https://discord.com/channels/1179127597926469703/1183121795247779910/1298003024027058187) (7 messages):

> - `Factor 64`
> - `Blog readership` (博客阅读量)
> - `Reasoning tokens` (推理 Token)
> - `CARDS method in LLMs` (LLM 中的 CARDS 方法)

- **Factor 64 的启示**：一位成员对 **Factor 64** 相关的突破表示兴奋，强调现在看来它是多么“显而易见”。
  
  - 这一领悟时刻引发了关于其影响的进一步讨论。
- **需要博客阅读量**：一位成员感叹**读他们博客的人不够多**，表达了对更多互动的渴望。
  
  - 他们指出，*在拥挤的数字空间中获得关注是一个挑战。*
- **对推理 Token 的怀疑**：有人担心 **reasoning tokens** 可能会产生误导，暗示它们仅仅是一种*近似值*。
  
  - 这种怀疑突显了关于 AI 模型推理效能的持续争论。
- **关于更长推理段落的讨论**：一位成员提到了一种名为 **CARDS** 的 LLM 解码时对齐方法，认为更长的推理块可能是有益的。
  
  - 他们强调该方法实现了 *5 倍的文本生成速度* 且无需重新训练，详见提供的[论文](https://arxiv.org/abs/2406.16306)。

**提及的链接**：[Ruqi Zhang (@ruqi_zhang) 的推文](https://x.com/ruqi_zhang/status/1810690177498595761)：介绍 CARDS，一种新的 LLM 解码时对齐方法：✨文本生成速度提升 5 倍，在 GPT-4/Claude-3 评估中胜率/平局率达 99% ✨可证明生成高奖励、高似然的文本 ✨无需重新训练...

---

### **Interconnects (Nathan Lambert) ▷ #**[**memes**](https://discord.com/channels/1179127597926469703/1187551504995987576/1298374595371335743) (2 messages):

> - `Jeremy Howard's tweet` (Jeremy Howard 的推文)
> - `Tek's angry man arc` (Tek 的愤怒男人阶段)

- **微软 CEO 被动漫头像账号 Ratioed**：在一条[推文](https://x.com/jeremyphoward/status/1848813387242999847)中，**Jeremy Howard** 指出 **Microsoft** 的 CEO 正在被一个动漫头像的账号 Ratioed（评论数远超点赞数）。
  
  - *一些成员觉得这很有趣，* 突显了社交媒体上对企业高管意想不到的反应。
- **Tek 持续的愤怒男人阶段**：一位成员观察到 **Tek** 已经表现出好几个月的愤怒迹象了。
  
  - *这一持续的趋势已成为讨论的话题，* 其他人也注意到了 Tek 举止上的明显转变。

**提及的链接**：[Jeremy Howard (@jeremyphoward) 的推文](https://x.com/jeremyphoward/status/1848813387242999847)：Microsoft 的 CEO 被一个动漫头像账号 Ratioed 了...

---

### **GPU MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1189498205101109300/1298070746467274815) (7 条消息):

> - `Unsloth 讲座发布`
> - `Gradient Accumulation 见解`
> - `GitHub AI 项目`
> - `工程技巧讨论`

- **Unsloth 讲座已发布！**：我们的 [Unsloth 演讲](https://www.youtube.com/watch?v=hfb_AIhDYnA) 现已上线！许多人称赞了整个环节中引人入胜的内容和密集的信息。
  
  - 一位观众评论道：*“我正以 0.5 倍速回看，但感觉还是很快”*，突显了讲座内容的丰富性。
- **深入探讨 Gradient Accumulation**：一位成员分享了关于 Gradient Accumulation 的详细见解，解释了在不同 batch 之间进行正确重缩放（rescaling）的重要性。提供的代码阐明了潜在的陷阱，并强调使用 **fp32** 等更高精度格式来避免大梯度带来的问题。
  
  - 他们指出：*“通常所有 batch 无法保持相同大小是有原因的”*，强调了训练场景中的复杂性。
- **深度学习 GitHub 项目**：一位用户分享了他们在 GitHub 上的项目：
  
  - 这是一个 [用纯 C 语言编写的 GPT 实现](https://github.com/shaRk-033/ai.c)，鼓励他人加深对深度学习的理解。

**提到的链接**：

- [Lecture 32: Unsloth](https://www.youtube.com/watch?v=hfb_AIhDYnA)：未找到描述
- [GitHub - shaRk-033/ai.c: gpt written in plain c](https://github.com/shaRk-033/ai.c)：用纯 C 编写的 GPT。通过在 GitHub 上创建账号来为 shaRk-033/ai.c 的开发做出贡献。

---

### **GPU MODE ▷ #**[**triton**](https://discord.com/channels/1189498204333543425/1189607595451895918/1298064102828015728) (19 条消息🔥):

> - `Torch Compile 输出解读`
> - `Softplus Triton Kernel 优化`
> - `Kernel 编译资源`

- **解读 Torch Compile 输出**：一位用户分享了运行 `torch.compile(model, mode='max-autotune')` 的输出，其中的指标显示了矩阵乘法操作的各种执行时间。另一位成员请求澄清如何解读这些 autotuning 结果和耗时。
  
  - *SingleProcess AUTOTUNE 耗时 30.7940 秒* 完成。
- **优化 Softplus Triton Kernels**：一位用户讨论了开发 *Softplus* Triton Kernel 的过程，但在每次启动时都遇到了 JIT 编译，寻求避免运行时检查的方法。他们考虑在不同的 block size 下缓存 kernel 以提高效率。
  
  - 他们确认，如果始终使用固定的 `BLOCK_SIZE`，则可以重复使用相同的 kernel 而无需重新编译。
- **探索 Kernel 编译资源**：有人询问了关于编译 Triton Kernel 的资源，特别是教程或仓库。一位成员建议 *Triton 文档* 可能是理解 kernel 使用的最佳起点。
  
  - 他们强调，在自定义实现中加入 dtype 提示可能会影响编译期间的性能。

 

---

### **GPU MODE ▷ #**[**torch**](https://discord.com/channels/1189498204333543425/1189607750876008468/1298021905026121760) (1 条消息):

> - `Meta HOTI 2024`
> - `Llama 3 基础设施`

- **Meta 的 HOTI 2024 讨论生成式 AI**：目前正在进行关于 **Meta HOTI 2024** 演讲的讨论，该演讲强调了活动中的挑战和见解。
  
  - 参与者指出，由演讲者 **Pavan Balaji** 主持的[这一环节](https://www.youtube.com/watch?v=zk9XFw1s99M&list=PLBM5Lly_T4yRMjnHZHCXtlz-_AKZWW1xz&index=15)解决了一些特定问题。
- **Powering Llama 3 主旨演讲亮点**：题为“Powering Llama 3”的 **主旨演讲环节** 揭示了 Meta 用于生成式 AI 的庞大基础设施。
  
  - 演讲中的见解对于理解 **Llama 3** 在行业中的集成和性能至关重要。

 

**提到的链接**：[Day 2 10:00: Keynote: Powering Llama 3: Peek into Meta’s Massive Infrastructure for Generative AI](https://www.youtube.com/watch?v=zk9XFw1s99M&list=PLBM5Lly_T4yRMjnHZHCXtlz-_AKZWW1xz&index=15)：演讲者：Pavan Balaji (Meta)

 

---

### **GPU MODE ▷ #**[**beginner**](https://discord.com/channels/1189498204333543425/1191300313928433664/1298026703091798016) (3 条消息):

> - `FA2 padded inputs`
> - `CUDA project ideas`
> - `CUDA accelerated regression`

- **FA2 填充输入与可变序列长度**：一位成员提出了关于如何在 **FA2** 中处理可变序列长度的 **填充输入 (padded inputs)** 的问题，并提到了一个名为 `flash_attn_varlen_qkvpacked_func` 的函数。
  
  - 他们表示很难找到一种简单的方法将 **填充的批处理张量 (padded batched tensor)** 转换为所需的输入格式。
- **寻求实习用的 CUDA 项目想法**：一位刚开始学习 **CUDA** 的用户表示有兴趣通过做项目来丰富简历，以便在明年夏天寻找实习机会。
  
  - 他们向社区征求关于可以开展的 **酷炫项目** 的建议。
- **CUDA 加速回归实现**：同一位用户分享了实现 **CUDA 加速的线性回归和逻辑回归** 的计划，但遭到了朋友的质疑。
  
  - 作为对该实现方案的回应，他们的朋友提供了一个包含项目想法的 **服务器链接**。

---

### **GPU MODE ▷ #**[**torchao**](https://discord.com/channels/1189498204333543425/1205223658021458100/1298061219218853950) (14 条消息🔥):

> - `torchao v0.6.1 Release`
> - `Compatibility of torchao Optimizers with HF Trainer`
> - `Implementing Quantization Aware Training with Older Torch Versions`
> - `Dynamic Masking during Training in torchao`

- **torchao v0.6.1 发布，引入新功能**：今天，**torchao v0.6.1** 发布，引入了诸如 **AWQ**、**Auto-Round** 和 **Float8 Axiswise 缩放训练** 等令人兴奋的新功能。更多详情，请查看[此处的发布说明](https://github.com/pytorch/ao/releases/tag/v0.6.1)。
  
  - 社区因其持续的贡献和参与而受到赞赏。
- **torchao 与 HF Trainer 的兼容性问题**：有一个关于 **torchao 优化器** 与 **HF Trainer** 兼容性的问题，普遍共识是它们应该可以工作，但可能会遇到问题。一位成员指出，由于潜在的冲突，使用 **HF 的 adamw 配合 int8 混合精度** 会出现速度变慢的情况。
  
  - 另一位成员提到 **CPUOffloadOptimizer** 可能会导致问题，因为它并不完全是一个普通的优化器。
- **量化感知训练 (Quantization Aware Training) 的挑战**：一位用户表达了对在旧版本 Torch 中实现 **量化感知训练** 的担忧，特别提到 **torch 1.9** 已经过时。有人建议尝试通过绕过 CPP 的命令从源码构建，但警告说由于版本之间框架的重大变化，这可能会导致问题。
  
  - 后续讨论指出 **torch.quantization** 中存在用于自定义量化方案的有用函数，从而引发了关于 **torchao** 是否是更健壮的重构（特别是在硬件支持方面）的讨论。
- **torchao 中的动态权重掩码**：一位用户询问了 **torchao** 中的 **sparsifier.step()** 函数，寻求关于它是否在训练期间动态查找权重掩码的澄清。澄清结果是，虽然它会将配置保留为目标，但它会持续更新掩码。

**提到的链接**：

- [深入探讨 PyTorch 量化 - Chris Gottbrath](https://www.youtube.com/watch?v=c3MT2qV5f9w&ab_channel=PyTorch.)：了解更多：https://pytorch.org/docs/stable/quantization.html。在开发时，有效利用服务器端和设备端计算资源非常重要...
- [Release v0.6.1 · pytorch/ao](https://github.com/pytorch/ao/releases/tag/v0.6.1)：亮点 我们很高兴宣布 torchao 0.6.1 版本发布！此版本增加了对 Auto-Round 的支持、Float8 Axiswise 缩放训练、BitNet 训练方案以及一个实现...

---

### **GPU MODE ▷ #**[**llmdotc**](https://discord.com/channels/1189498204333543425/1227345713348870156/) (1 条消息):

apaz：他们正在研究 llama3，但确实如此。

---

### **GPU MODE ▷ #**[**rocm**](https://discord.com/channels/1189498204333543425/1233704710389764236/1298044150054260737) (3 条消息):

> - `ROCm 6.2 Docker Image`
> - `GitHub Actions for AMD Cluster`
> - `Torch + ROCm with Poetry`
> - `Difference between ROCm and Official PyTorch Images`
> - `Job Queue Setup`

- **在 MI250 上测试 ROCm 6.2 Docker 镜像性能**：一位成员构建了一个新的 [ROCm 6.2 Docker 镜像](https://github.com/michaelfeil/infinity/pull/434/files)，并渴望在 MI250 上测试其性能。
  
  - 该 Pull Request 包含了针对 **NVIDIA**、**CPU** 和 **AMD** 等多种环境的 Docker 配置重大更新。
- **为 AMD Cluster 提交 GitHub Actions**：另一位成员鼓励通过 [此链接](https://github.com/gpu-mode/amd-cluster/tree/main/.github/workflows) 提交 GitHub Actions，以方便在 AMD Cluster 中执行作业。
  
  - 他们强调了为 **gpu-mode/amd-cluster** 仓库的开发做出贡献的重要性。
- **ROCm 镜像与官方 PyTorch 镜像的区别**：一位参与者对个人 ROCm 镜像与 Docker Hub 上提供的 **官方 ROCm PyTorch 镜像** 之间的区别表示好奇。
  
  - 这一询问突显了用户在转向基于 ROCm 的设置时对清晰指导的需求。
- **作业队列设置讨论**：一位成员表示他们正尝试设置作业队列，以便在环境中管理任务。
  
  - 这反映了在 ROCm 工作流中优化资源利用和作业管理的兴趣日益增长。
- **寻求使用 Poetry 安装 Torch + ROCm 的解决方案**：用户对使用 **Poetry** 进行依赖管理来集成 **Torch 和 ROCm 安装** 的解决方案感兴趣。
  
  - 一场公开讨论正征集简化安装过程的实用方法。

**提到的链接**：

- [amd-cluster/.github/workflows at main · gpu-mode/amd-cluster](https://github.com/gpu-mode/amd-cluster/tree/main/.github/workflows)：向 AMD Cluster 提交作业的仓库。通过在 GitHub 上创建账号来为 gpu-mode/amd-cluster 的开发做出贡献。
- [无标题](https://hub.docker.com/r/rocm/pytorch)：未找到描述
- [New docker image for rocm / torch cpu by michaelfeil · Pull Request #434 · michaelfeil/infinity](https://github.com/michaelfeil/infinity/pull/434/files)：该 Pull Request 包含了针对各种环境（包括 NVIDIA、CPU、AMD 和 TensorRT）的 Docker 配置和依赖管理的重大更新。这些更改旨在简化...

---

### **GPU MODE ▷ #**[**bitnet**](https://discord.com/channels/1189498204333543425/1240586843292958790/1298248538844561459) (7 条消息):

> - `Bitnet Implementation Weights`
> - `Packed Weights`
> - `Ternary Weights`

- **澄清 Bitnet 权重**：一位成员询问为什么 Bitnet 实现中的 **权重 (weights)** 不是 **三值 (ternary)** {-1,0,1}。
  
  - 另一位成员建议权重可能是 **打包 (packed)** 的，并建议检查形状 (shape) 以进一步澄清。
- **理解打包权重 (Packed Weights)**：一位成员解释说，权重看起来是 **打包** 过的，其中一个维度是其应有大小的 1/4，这表明是将 **4 个 2-bit 打包** 到了 **1 个 8-bit** 中。
  
  - 这一细节暗示了权重的有效表示方式，强调了实现的复杂性。
- **对三值权重的认识**：在得到澄清后，最初的询问者表示理解了 **三值权重 (ternary weights)**。
  
  - 在细节确认后，成员表达了简单的感谢。

---

### **GPU MODE ▷ #**[**sparsity-pruning**](https://discord.com/channels/1189498204333543425/1247663759434977453/1298357659472494725) (1 messages):

> - `TorchAO Sparsity Future Plans`
> - `Advancements in Sparsity & Pruning`
> - `Collaborative Opportunities`

- **TorchAO 分享了 Sparsity 的未来计划**：在反思近期进展后，[GitHub](https://github.com/pytorch/ao/issues/1136) 上分享了一份关于 **torchao** 中 Sparsity 未来计划的提案。
  
  - 讨论重点在于增强对 **distillation** 实验和 **fast compilable sparsification** 例程的支持。
- **Sparsity 准确率方面的进展**：Sparsity 和 Pruning 在准确率方面的关键进展包括 **distillation** 和 **activation sparsity** 的发展。
  
  - 该帖子征求社区兴趣和协作的反馈，询问这些进展是否与当前的优先事项产生共鸣。
- **呼吁在 Sparsity 工作上进行协作**：作者表达了希望社区在提议的 Sparsity 项目上进行协作的愿望，并表示这是一个参与的好机会。
  
  - 对其他重要话题的讨论持开放态度，为社区成员的意见输入营造了包容的氛围。

 

**提到的链接**：[[RFC] Sparsity Future Plans · Issue #1136 · pytorch/ao](https://github.com/pytorch/ao/issues/1136)：在 PTC / CUDA-MODE 之后，我有机会进行了反思，并想分享一些关于 torchao 中 Sparsity 未来计划的想法。现状：Sparsity 有两个组成部分，准确率（accuracy）和加速（accelerat...

 

---

### **GPU MODE ▷ #**[**liger-kernel**](https://discord.com/channels/1189498204333543425/1275130785933951039/1298073617862754334) (11 条消息🔥):

> - `Liger Kernel Inference` (Liger Kernel 推理)
> - `Cross Entropy Issues` (Cross Entropy 问题)
> - `Pull Request for Batch Norm` (Batch Norm 的 Pull Request)
> - `Transformers Monkey Patching` (Transformers Monkey Patching)
> - `Loss Function References` (Loss Function 参考资料)

- **Liger Kernel 在 Llama 推理上表现不佳**：一位成员报告称，在 **Llama 3.2** 上使用 **3k token** 的 prompt 时，使用 liger 会导致推理延迟增加，而不是性能提升。
  
  - *感谢！我没有看到性能提升……至少在 3B 模型上是这样。*
- **建议调整 Cross-Entropy 设置**：另一位成员建议尝试将 **cross_entropy = True** 和 **fused_linear_cross_entropy = False** 进行设置，以寻求潜在的性能改进。
  
  - 成员们讨论了 **liger** 的默认设置可能不适合推理需求，因为它主要是针对 LLM 训练进行优化的。
- **新增 Batch Norm 的 Pull Request**：一位成员宣布了一个为 Liger-Kernel 添加 **batch norm** 的 Pull Request，并将其性能与 **Keras** 的 batch norm 进行了对比。
  
  - 该 PR 旨在增强功能，并包含了在 **4090** 设备上的测试结果。
- **针对最新版 Transformers 的 Cross Entropy 补丁**：关于 Transformers 的 **cross-entropy monkey patching** 展开了讨论，怀疑它在最新的 GA 版本上无法正常工作。
  
  - 讨论指出，目前大多数 CausalLMs 使用 *self.loss_function* 而不是 **CrossEntropyLoss**，这可能会影响当前的 patch 策略。
- **提供了 Loss Functions 参考链接**：成员们分享了 Transformers 中使用的 **loss functions** 的关键链接，详细介绍了它们的实现和用法。
  
  - Hugging Face 中的根 **cross-entropy function** 可以在[这里](https://github.com/huggingface/transformers/blob/049682a5a63042f087fb45ff128bfe281b2ff98b/src/transformers/loss/loss_utils.py#L26)找到。

**提到的链接**：

- [added batch norm by vulkomilev · Pull Request #321 · linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel/pull/321)：摘要：由 vulkomilev 添加了 batchNorm。测试已完成，我已将其与 Keras 的 batch norm 进行了对比。使用了 4090 硬件。类型：[ X] 运行 make test 以确保正确性 [X ] 运行 make checkstyle 以确保...
- [transformers/src/transformers/loss/loss_utils.py at 049682a5a63042f087fb45ff128bfe281b2ff98b · huggingface/transformers](https://github.com/huggingface/transformers/blob/049682a5a63042f087fb45ff128bfe281b2ff98b/src/transformers/loss/loss_utils.py#L26)：🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的最先进机器学习库。 - huggingface/transformers
- [transformers/src/transformers/models/gemma2/modeling_gemma2.py at 049682a5a63042f087fb45ff128bfe281b2ff98b · huggingface/transformers](https://github.com/huggingface/transformers/blob/049682a5a63042f087fb45ff128bfe281b2ff98b/src/transformers/models/gemma2/modeling_gemma2.py#L1071))：🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的最先进机器学习库。 - huggingface/transformers
- [transformers/src/transformers/loss/loss_utils.py at 049682a5a63042f087fb45ff128bfe281b2ff98b · huggingface/transformers](https://github.com/huggingface/transformers/blob/049682a5a63042f087fb45ff128bfe281b2ff98b/src/transformers/loss/loss_utils.py#L32))：🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的最先进机器学习库。 - huggingface/transformers
- [Liger-Kernel/src/liger_kernel/transformers/monkey_patch.py at 99599091373f178e8ad6a69ecb1b32351d1d5c1f · linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel/blob/99599091373f178e8ad6a69ecb1b32351d1d5c1f/src/liger_kernel/transformers/monkey_patch.py#L457))：用于 LLM 训练的高效 Triton 内核。通过在 GitHub 上创建账户为 linkedin/Liger-Kernel 的开发做出贡献。

---

### **GPU MODE ▷ #**[**self-promotion**](https://discord.com/channels/1189498204333543425/1288557096404516945/1298352304147402803) (1 条消息):

> - `Model Quantization`
> - `IEEE IPTA Conference`

- **IEEE IPTA 上的 Model Quantization 教程**：在 **IEEE IPTA 会议**上演示了一个关于 **Model Quantization** 的教程，重点介绍了该领域的关键技术和应用。
  
  - 欲了解更多详情，感兴趣的学习者可以查阅[演讲幻灯片](https://docs.google.com/presentation/d/17sPe-DtCWaZf9Y3omgZZr95A9_EPRe1KSdQ2vXx4C4k)。
- **IPTA 2024 概览**：**IPTA 2024** 会议是即将举行的盛会，专注于技术进步，特别是在 **Model Quantization** 等领域。
  
  - 鼓励参与者参加各种教程和演讲，这可能会对当前的研究趋势产生更深刻的见解。

**提到的链接**：[IPTA 2024 - Quantization Tutorial](https://docs.google.com/presentation/d/17sPe-DtCWaZf9Y3omgZZr95A9_EPRe1KSdQ2vXx4C4k/)：高效 Transformer 模型的 Model Quantization 技术，Hicham Badri 博士，Mobius Labs GmbH 首席研究科学家，IEEE IPTA 2024 - 摩洛哥拉巴特。

---

### **GPU MODE ▷ #**[**project-popcorn**](https://discord.com/channels/1189498204333543425/1298372518293274644/1298375128786276484) (6 条消息):

> - `LLM for Efficient Kernels`
> - `Scaling Test Time Compute`
> - `Kernel Dataset Competition`
> - `HidetScript DSL`

- **为高效 Kernel 创建 LLM**：一位成员概述了公开创建一个用于生成高效 Kernel 的 LLM 的计划，目标是在 2024 年 12 月的 NeurIPS 上发布 MVP，向人类和 LLM 解释 **GPU 的工作原理**。
  
  - 该基线将使用 **ncu** 进行大规模采样和验证，同时从现有来源收集全球最大的 Kernel 数据集。
- **引入 Kernel 数据集竞赛**：计划包括创建一个竞赛来为新 Token 构建**数据飞轮 (data flywheel)**，旨在通过在 Discord 上透明地开展所有工作来吸引更多人参与，并由公开赞助商资助。
  
  - MVP 还将明确如何衡量 Kernel 编写的复杂性，并确保输出代码使用适当的抽象。
- **参与简单的 Prompt Engineering**：针对 CUDA 和 Triton 应用相关的任务，提出了使用 **few-shot** 示例且无需 **finetuning** 的简单 **Prompt Engineering**。
  
  - 这种方法旨在利用现有知识，同时尝试不同的 Kernel 生成方法。
- **HidetScript 用于 Kernel 程序的潜力**：一位成员建议探索 [HidetScript](https://hidet.org/docs/stable/hidet-script/examples/index.html#hidet-script-examples) 作为编写 Kernel 程序的 DSL，它直接生成 **CUDA 代码**，而不是像 Triton 那样生成 PTX。
  
  - 他们建议，由于其流行度，将其功能扩展到 **Metal**、Modular 的 Kernel 定义语言以及 **TVM** 也是值得的。

**提到的链接**：

- [Examples — Hidet Documentation](https://hidet.org/docs/stable/hidet-script/examples/index.html#hidet-script-examples)：未找到描述
- [TK + Monkeys + CUDAGen](https://docs.google.com/presentation/d/1JtxGXv80ciIne-bFxySZ25q0J2mAwsXlb9uuST9naqg/edit?usp=sharing)：ThunderKittens，一个简单的 AI Kernel 框架
- [Monkeys_for_Meta_v3.pptx](https://docs.google.com/presentation/d/14jlbVPyohnWuQgFikr74cnaj-mzoEMPT/edit?usp=sharing&ouid=111422880520483065413&rtpof=true&sd=true)：Large Language Monkeys: 通过重复采样扩展推理时间计算 (Scaling Inference-Time Compute with Repeated Sampling)，作者：Brad Brown*, Jordan Juravsky*, Ryan Ehrlich*, Ronald Clark, Quoc Le, Chris Ré, Azalia Mirhoseini
- [META KERNELS - Google Drive](https://drive.google.com/drive/folders/1nt2KcRRKb8YdySxkRxUu5PR4c7UPM_rK)：未找到描述

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1298020964671684678) (47 messages🔥):

> - `AGI 挑战`
> - `自定义 GPT 混淆`
> - `量子计算观点`
> - `Anthropic AI 发布`
> - `电视剧识别问题`

- **关于 AGI 可行性的辩论**：成员们讨论了我们是否因为提供的数据类型而难以实现 **AGI**，质疑 **binary data**（二进制数据）是否可能抑制了进展。
  
  - 一位成员认为，虽然存在学习限制，但通过改进算法仍然可以实现 AGI，而不受数据类型的束缚。
- **澄清自定义 GPT 术语**：参与者指出，“GPTs”一词可能会引起混淆，它通常指代 **custom GPTs**（自定义 GPT），而非包括 ChatGPT 等模型在内的更广泛类别。
  
  - 讨论强调了在区分通用 **GPTs** 与特定实现时需要保持清晰。
- **关于量子计算模拟器的见解**：一位成员提出，为了使量子计算模拟器具有实际用途，理想情况下它们应该产生与真实量子计算机 **1:1** 的输出。
  
  - 虽然有公司正在开发此类模拟器，但其有效性和实际应用仍是辩论的话题。
- **Anthropic 的新 AI Agents**：**TANGO talking head model** 因其对口型和执行身体动作的能力而受到关注，引发了对其开源能力的兴趣。
  
  - 另一位成员分享说，**Claude 3.5 Sonnet** 在 Agent 性能基准测试中表现出色，尽管其他人认为 **Gemini Flash 2.0** 可能会超越它。
- **ChatGPT 在电视剧方面的局限性**：一位成员讲述了 **ChatGPT** 在识别电视剧正确集数和标题方面遇到困难的经历，表明存在训练数据缺口。
  
  - 讨论指出，主观观点可能占据了数据主导地位，从而影响了特定 **TV show** 查询的准确性。

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1298044513570263131) (4 messages):

> - `o1-preview 使用限制`
> - `讨论中的用户引导`

- **o1-preview 回复限制引发询问**：新成员 @sami_16820 询问了 **o1-preview** 的使用限制，指出他们在 **2024 年 10 月 29 日** 切换到另一个模型之前还剩余 **5 次回复**。
  
  - 作为回应，一位用户澄清说 o1-preview 的限制是 **每周 50 次回复**。
- **新用户寻求指导**：在自我介绍中，@sami_16820 表达了对平台的不熟悉，并寻求有关 **o1-preview** 的信息。
  
  - 这次交流凸显了社区在协助新人熟悉平台时的友好氛围。

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/1298034031161507892) (11 messages🔥):

> - `Prompting 中的上下文顺序`
> - `CSV 数据的错误纠正`
> - `使用 GPT 解决问题`
> - `强调 Prompt 细节`
> - `用于独立思考的结构化 Prompt`

- **上下文顺序至关重要**：一位成员强调，为了在指令中 **强调重要信息**，其位置应放在 Prompt 的开头或结尾。
  
  - 另一位参与者建议使用 **table of contents**（目录）来更好地构建长指令的结构。
- **使用照片纠正 CSV 中的错误**：一位成员询问如何编写 Prompt，以纠正使用 gpt-4o 从菜单照片生成的 CSV 中 **价格不准确** 的问题。
  
  - 他们收到建议，去编辑数据生成过程中开始出现 **hallucinations**（幻觉）的原始 Prompt。
- **使用 GPT 独立解决问题**：一位成员请求协助为 ChatGPT 开发一个 Prompt，使其能够独立确定一个正 **decagon**（十边形）的线条可以将平面分割成多少个区域。
  
  - 另一位成员建议将此查询转化为 **structured prompt**（结构化 Prompt），引导 GPT 按程序化步骤处理问题。
- **强调 Prompt 中的细节**：随后讨论了强调 Prompt 部分内容的方法，以便更有效地澄清请求。
  
  - 参与者一致认为，上下文顺序在确保清晰度和关注重要元素方面起着关键作用。

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/1298034031161507892) (11 条消息🔥):

> - `Importance of Context Order in Instructions`（指令中上下文顺序的重要性）
> - `Using Table of Contents for Emphasis`（使用目录进行强调）
> - `Error Correction on CSV from Menu Photos`（对菜单照片生成的 CSV 进行纠错）
> - `Structured Prompts for Problem Solving`（用于问题解决的结构化 Prompt）

- **上下文顺序至关重要**：一位成员强调，在指令中，**上下文顺序 (order of context)** 对于突出重要性至关重要，尤其是在较长的 Prompt 中。
  
  - *Cheers!* 是对这一见解的简短认可。
- **提议使用索引进行强调**：讨论了使用**目录和索引**来帮助维持对 Prompt 关键部分的强调。
  
  - 一位成员确认，结构化部分可以帮助缓解 Prompt 清晰度的问题。
- **菜单 CSV 的纠错**：一位成员分享说，他们有一份由 GPT-4 从菜单照片生成的 **700 行 CSV**，但有些价格需要修正。
  
  - 他们寻求通过反馈照片和 CSV 来进行一轮纠错的 Prompt。
- **修改初始 Prompt 以提高准确性**：另一位成员建议修改初始 Prompt，以解决之前输出中出现 **hallucinations**（幻觉）的地方。
  
  - 这突显了在生成准确结果时进行 Prompt 优化的必要性。
- **用于独立思考的结构化 Prompt**：一位成员询问如何使用 Prompt 让 ChatGPT **独立解决**一个涉及正十边形的几何问题。
  
  - 其他人建议构建结构化的 Prompt，以引导 GPT 进行深思熟虑的、程序化的问题解决。

 

---

### **Cohere ▷ #**[**discussions**](https://discord.com/channels/954421988141711382/954421988783444043/1298029082671579136) (38 条消息🔥):

> - `Cohere models usage`（Cohere 模型使用）
> - `Multi-modal embeddings`（多模态 Embeddings）
> - `Cohere for AI + Embed - V3`
> - `Event scheduling issues`（活动调度问题）
> - `Performance tuning in LLMs`（LLM 性能调优）

- **Cohere 模型受到青睐**：成员们讨论了在 Playground 中积极使用 **Cohere 模型**，强调了它们的多样化应用和尝试。
  
  - 一位成员特别指出，在处理 **multi-modal embeddings** 时，需要使用不同的模型重新运行推理。
- **对多模态 Embeddings 的兴奋**：讨论转向了 **multi-modal embeddings**，成员们对其兼容性和性能表示兴奋和好奇。
  
  - 有人询问图像 Embedding 是否与文本 Embedding 共享**相同的潜空间 (latent space)**，并承认它们可能有所不同。
- **对即将举行的活动的困惑**：关于活动时间存在困惑，一位成员幽默地指出它被标记为“即将推出：昨天晚上 8 点”。
  
  - 管理员澄清这是 Discord 的一个 Bug，活动将在 22 分钟后开始。
- **对 Cohere for AI + Embed - V3 的推测**：成员们对即将推出的 **Cohere for AI + Embed - V3** 表示好奇，一位参与者称其为潜在的**多模态 Command 模型**。
  
  - 另一位成员回复并确认它是一个 **Global connection model**，旨在跨不同模态连接用户。
- **提高 LLM 性能**：成员们分享了模型性能调优的经验，特别注意到行顺序如何影响结果。
  
  - 一位成员指出，在调整代码后，平均 Loss 从 2.5 大幅下降到 1.55，这表明了一种探索性的编程方法。

 

**提到的链接**：[Vsauce Michael GIF - Vsauce Michael Or Is It - Discover & Share GIFs](https://tenor.com/view/vsauce-michael-or-is-it-gif-19808095)：点击查看 GIF

 

---

### **Cohere ▷ #**[**announcements**](https://discord.com/channels/954421988141711382/996880279224451154/1298316076182147123) (1 条消息):

> - `Multimodal Embed 3 Release` (Multimodal Embed 3 发布)
> - `RAG Systems Integration` (RAG 系统集成)
> - `API Changes in Embed 3` (Embed 3 中的 API 变更)
> - `Image Processing Enhancements` (图像处理增强)

- **Multimodal Embed 3 正式发布！**：全新的 **Embed 3** 模型在检索任务上拥有 SOTA 性能，并且在混合模态和多语言搜索方面表现卓越，允许用户在单个数据库中存储文本和图像数据。
  
  - 更多详情请查看 [博客文章](https://cohere.com/blog/multimodal-embed-3) 和 [发布说明](https://docs.cohere.com/changelog/embed-v3-is-multimodal)。
- **让 RAG 系统变得简单**：**Embed 3** 支持在各种数据源（如统计图、图表和产品目录）上构建快速、准确的 **RAG 系统** 和搜索应用。
  
  - 这种集成方法降低了复杂性并增强了数据交互。
- **API 变更简化了图像处理**：**Embed API** 现在支持名为 `image` 的新 `input_type`，并引入了用于处理图像的 `images` 参数，从而简化了用户体验。
  
  - 值得注意的是，目前的 API 限制每次请求仅限一张图像，最大尺寸为 **5mb**。
- **参加 Office Hours 获取见解**：Cohere 正在举办 Office Hours 活动，邀请 **Embed 高级产品经理** 分享见解，旨在帮助用户了解新功能。
  
  - 参与者可以点击 [此处](https://discord.com/events/954421988141711382/1298319720868745246) 加入活动，直接向专家学习。

**提到的链接**：

- [Introducing Multimodal Embed 3: Powering AI Search](https://cohere.com/blog/multimodal-embed-3)：Cohere 发布了一款最先进的多模态 AI 搜索模型，为图像数据释放真正的商业价值。
- [Embed v3.0 Models are now Multimodal — Cohere](https://docs.cohere.com/changelog/embed-v3-is-multimodal)：为我们的 Embed 模型推出多模态嵌入，并提供了一些入门代码。

---

### **Cohere ▷ #**[**questions**](https://discord.com/channels/954421988141711382/1168411509542637578/1298161042412671006) (10 条消息🔥):

> - `LLM Model Fine-Tuning` (LLM 模型微调)
> - `Parallel Request Handling` (并行请求处理)
> - `Cohere Command R Features` (Cohere Command R 特性)
> - `Playground Usage in Professional Settings` (在专业环境中使用 Playground)

- **微调 LLM 需要更多数据**：一位成员分享了使用小数据集微调 LLM 的见解，指出了潜在的过拟合问题并寻求策略指导。
  
  - 回复者建议增加数据集大小并调整学习率等超参数，并参考了 [Cohere 微调指南](https://cohere.com/llmu/fine-tuning-for-chat)。
- **本地设置中的并行请求问题**：一位成员正在测试 Cohere Command R 的并发请求处理，但反馈请求是按顺序处理的。
  
  - 他们请求关于如何为其概念验证 (POC) 目的启用并行处理的指导。
- **Cohere Command R+ 的图像读取能力**：一位成员询问 Command R+ 何时能够读取图像，表示对扩展功能的兴趣。
  
  - 这凸显了用户对 Cohere 模型中多模态能力的广泛兴趣。
- **关于在临床环境中使用 Playground 的担忧**：一位成员对在诊所电脑上使用 Playground 表示疑虑，理由是担心其专业适用性。
  
  - 尽管存在担忧，另一位成员澄清说，虽然这并非禁止行为，但在专业用途中是非常不鼓励且不受支持的。

 

**提到的链接**：[Starting the Chat Fine-Tuning — Cohere](https://docs.cohere.com/docs/chat-starting-the-training#parameters)：了解如何通过 Cohere Web UI 或 Python SDK 为聊天微调 Command 模型，包括数据要求、定价和调用模型的方法。

 

---

### **Cohere ▷ #**[**api-discussions**](https://discord.com/channels/954421988141711382/1168578329423642786/1298168781524439060) (19 条消息🔥):

> - `Multilingual Model Latency` (多语言模型延迟)
> - `API Token Usage` (API Token 使用)
> - `Read Timeout Issues` (读取超时问题)

- **多语言模型遭遇延迟峰值**：多名成员报告 embed 多语言模型出现 **30-60秒延迟**，部分用户在 **15:05 CEST** 左右甚至经历了 **90-120秒** 的延迟。
  - 尽管最初存在担忧，但问题似乎已得到改善，官方敦促成员报告任何持续存在的故障。
- **关于 API Token 使用的澄清**：一名成员询问在 API 请求中是否有必要使用 `<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>`，并质疑它们对响应质量的影响。
  - 官方澄清，对于 **chat requests**，不需要包含这些 Token，因为它们很可能会被忽略。
- **读取超时问题持续存在**：一名成员报告了持续的读取超时问题，特别提到了来自其 cohere 命令的超时消息。
  - 作为回应，团队成员表示他们正在部署修复程序，并承诺在未来一小时内解决。

**提及链接**：[incident.io - Status pages](https://status.cohere.com/,): 未找到描述

---

### **Cohere ▷ #**[**projects**](https://discord.com/channels/954421988141711382/1218409701339828245/1298368461776617534) (1 条消息):

> - `Agentic Builder Day`
> - `OpenSesame collaboration`
> - `Cohere Models competition`

- **Agentic Builder Day 宣布举办**：Cohere 和 OpenSesame 将于 11 月 23 日共同举办 **Agentic Builder Day**，邀请优秀的开发者竞争使用 Cohere Models 创建 AI Agent。
  - 参与者可以申请加入这场 **8 小时的黑客松 (hackathon)**，在展示技能的同时有机会赢取奖品。
- **黑客松招募 AI 开发者**：该活动寻求渴望合作与竞争的资深开发者，为在多伦多构建有影响力的 AI 产品提供平台。
  - 鼓励感兴趣的人士[立即申请](https://www.opensesame.dev/hack)加入这个以社区为中心的竞赛。

**提及链接**：[OpenSesame | Build Better AI Agents](https://www.opensesame.dev/hack): OpenSesame 简化了从构建到评估的整个 AI Agent 生命周期。我们的平台使企业能够轻松创建、共享和实施 AI Agent 并检测幻觉，使 AI...

---

### **Modular (Mojo 🔥) ▷ #**[**general**](https://discord.com/channels/1087530497313357884/1098713601386233997/1298245892767350795) (5 条消息):

> - `Mojo Language Structure of Arrays`
> - `Mojo Language Slices`
> - `Community Reflections on Collections`

- **Mojo 语言支持自定义结构数组 (Structure of Arrays)**：虽然 Mojo 语言本身没有内置，但你可以使用优雅的语法在 Mojo 中轻松创建自己的 **Structure of Arrays (SoA)**。
  - 目前存在 **slice type**，但受到一定限制，预计随着类型系统的发展将得到增强。
- **关于 Mojo 语言 Slices 的讨论**：虽然 Mojo 包含 slice 类型，但它只是标准库中的一个 struct，虽然有一些方法返回 slices，但尚未完全集成。
  - 成员们表示，随着语言的演进，这一限制将被重新审视。
- **社区对 SOA 和 Reflection API 的见解**：之前的社区会议讨论了在 Mojo 的 **reflection API** 中实现 **自动化 SOA 转换** 的潜力，这可能允许在各种集合中进行自动转换。
  - 虽然前景广阔，但设计这些自动转换需要非常先进的编译器或巧妙的 reflection 技术。

---

### **Modular (Mojo 🔥) ▷ #**[**mojo**](https://discord.com/channels/1087530497313357884/1151418092052815884/1298101472927551500) (40 messages🔥):

> - `Performance of Binary Stripping`
> - `Comptime Variables in Mojo`
> - `Using Tuple Arguments`
> - `BigInt Operations Comparison`
> - `Arbitrary Width Integer Libraries`

- **二进制剥离显著减小体积**：对一个 **300KB 的二进制文件**进行剥离（Stripping）可以将其减至仅 **80KB**，展示了巨大的优化潜力。
  
  - 成员们注意到这一过程带来的*令人印象深刻的体积下降*。
- **参数范围外的 Comptime 变量**：有用户询问在 `@parameter` 范围外使用 `comptime var` 的问题，并提到遇到了编译错误。
  
  - 讨论指出，**alias** 允许编译时声明，但直接的可变性（mutability）并不直观。
- **Mojo 中元组参数的问题**：涉及元组参数操作的代码导致编译器崩溃，表明在数组中使用 **StringSlice** 可能存在潜在问题。
  
  - 成员们讨论了改进 Trait 实现的必要性，以增强此类场景下的可用性。
- **Node.js 与 Mojo 的 BigInt 计算对比**：用户对比了 Node.js 中耗时约 **40 秒**的 BigInt 计算，认为在 Mojo 中可以进行优化。
  
  - 讨论显示，优化任意宽度整数库（arbitrary width integer library）对于性能对比至关重要。
- **理解整数库**：成员们讨论了任意宽度整数库在处理超出标准整数范围的计算时的重要性。
  
  - 提到为了适应 **1026 bits** 的运算，需要专门的库来弥补计算差距。

---

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1298065767690141728) (15 messages🔥):

> - `LLVM Renderer Refactor`
> - `Tinygrad Performance Improvements`
> - `Gradient Clipping Integration`
> - `ACT Training Progress`
> - `TinyJit Decorator Queries`

- **LLVM 渲染器重构提议**：有用户建议以模式匹配（pattern matcher）风格重写 **LLVM renderer**，以增强其功能。
  
  - 这可能会显著提高代码的清晰度和效率。
- **提升 Tinygrad 的速度**：讨论强调了在转向使用 uops 后增强 **Tinygrad 性能** 的必要性。
  
  - 这对于紧跟计算能力的进步至关重要。
- **在 Tinygrad 中集成** `clip_grad_norm_`：有用户提出 `clip_grad_norm_` 是否应成为 Tinygrad 的标准部分，理由是它在深度学习代码中频繁出现。
  
  - George Hotz 指出，在推进此集成之前，需要先进行 grad 重构。
- **Action Chunking Transformers 的进展**：有用户报告 **ACT 训练** 在几百步后趋于收敛，损失值（loss）降至 **3.0** 以下。
  
  - 他们分享了 [源代码](https://github.com/mdaiter/act-tinygrad) 和研究论文的链接以供进一步了解。
- **关于 TinyJit 装饰器功能的查询**：有人询问 `@TinyJit` 装饰器是否适用于带有字典键和 Tensor 值的批次输入。
  
  - 同时也对 TinyJit 多次重复使用同一输入的旧行为表示了担忧。

**提到的链接**：

- [Tensor.gradient 应该是一个函数（方法）· Issue #7183 · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/issues/7183)：grad_weights = loss.gradient(\*weights) 应该确认此 API 是否也适用于二阶导数，但我认为没有理由不支持。目前受阻于实际计算 grad 的大图（big graph）...
- [GitHub - facebookresearch/lingua: Meta Lingua: a lean, efficient, and easy-to-hack codebase to research LLMs.](https://github.com/facebookresearch/lingua)：Meta Lingua：一个精简、高效且易于修改的 LLM 研究代码库。
- [GitHub - mdaiter/act-tinygrad: Action Chunking Transformers in Tinygrad](https://github.com/mdaiter/act-tinygrad)：Tinygrad 中的 Action Chunking Transformers。通过在 GitHub 上创建账号为 mdaiter/act-tinygrad 做出贡献。

---

### **tinygrad (George Hotz) ▷ #**[**learn-tinygrad**](https://discord.com/channels/1068976834382925865/1070745817025106080/1298015686907658321) (15 条消息🔥):

> - `Tensor Indexing Techniques` (Tensor 索引技术)
> - `Python Compatibility with MuJoCo` (Python 与 MuJoCo 的兼容性)
> - `Intermediate Representation Inspection` (中间表示检查)
> - `Custom Compiler Development` (自定义编译器开发)

- **探索使用 .where() 进行 Tensor 索引**：讨论了在布尔 Tensor 上使用 `.where()` 函数的方法，建议采用 `m.bool().where(t, None)` 作为一种方案。
  
  - 然而，有人指出使用 `.int()` 进行索引会导致结果为 `[2,1,2]`，不符合预期。
- **MuJoCo 首选 Python3.10**：一位用户发现 **Python3.10** 是运行 **MuJoCo** 唯一兼容的版本，而 **Python3.12** 会导致其功能失效。
  
  - 这引发了关于兼容性问题和特定版本约束的讨论。
- **获取中间表示 (Intermediate Representation)**：用户表示有兴趣在编译前获取线性化输出，以检查中间表示。
  
  - 提到设置 `DEBUG=6` 可以打印线性化的 UOps 以供检查。
- **开发自定义编译器后端**：另一位用户热衷于构建自定义编译器实现，并通过其后端运行输出。
  
  - 社区分享了获取线性化输出和渲染器函数的资源与示例，以辅助调试。

---

### **OpenInterpreter ▷ #**[**general**](https://discord.com/channels/1146610656779440188/1147665339266650133/1298003755928653836) (24 条消息🔥):

> - `Hume AI Voice Assistant` (Hume AI 语音助手)
> - `Claude 3.5 Sonnet Release` (Claude 3.5 Sonnet 发布)
> - `Open Interpreter and Claude Integration` (Open Interpreter 与 Claude 集成)
> - `Screenpipe Tool` (Screenpipe 工具)
> - `Open Source Monetization Models` (开源盈利模式)

- **Hume AI 加入阵营**：一名成员宣布在 **phidatahq** 通用 Agent 中加入了 **Hume AI** 语音助手，通过精简的 UI 以及在 Mac 上创建和执行 applescripts 的能力增强了功能。
  
  - *“非常喜欢新的 @phidatahq UI”*，评论指出这一集成带来了显著改进。
- **Claude 3.5 Sonnet 开启实验性功能**：Anthropic 正式发布了 **Claude 3.5 Sonnet** 模型，并开放了计算机使用（computer use）的公开测试版访问权限，尽管该功能仍被描述为实验性且容易出错。
  
  - 成员们表达了兴奋之情，同时指出这类进步进一步增强了 AI 模型不断增长的能力。
- **Open Interpreter 结合 Claude 实力大增**：社区对使用 **Claude** 增强 **Open Interpreter** 充满热情，成员们讨论了运行新模型的实际实现和代码。
  
  - 一名成员报告了使用特定模型命令的成功经验，并鼓励其他人尝试。
- **Screenpipe 受到关注**：成员们称赞 **Screenpipe** 工具在构建日志中的实用性，并注意到其有趣的落地页和社区贡献潜力。
  
  - 一名成员引用了 GitHub 上一个有用的 Profile 链接，鼓励更多人参与该工具。
- **盈利模式与开源的结合**：围绕通过允许用户从源码构建或为预构建版本付费来使公司盈利的模式展开了讨论，旨在平衡贡献与使用。
  
  - 成员们对这种模式表示赞同，强调了开发者贡献和付费用户双重带来的好处。

**提到的链接**：

- [Anthropic (@AnthropicAI) 的推文](https://x.com/AnthropicAI/status/1848742747626226146)：新的 Claude 3.5 Sonnet 是首个在公开测试版中提供计算机使用（computer use）功能的尖端 AI 模型。虽然具有开创性，但计算机使用功能仍处于实验阶段——有时会出错。我们正在提前发布它...
- [Jacob@AIwithBenefits (@AIwithBenefits) 的推文](https://x.com/AIwithBenefits/status/1848161437828415578)：在 @phidatahq 通用 Agent 中添加了 @hume_ai 语音助手，并从 @OpenInterpreter 系统提示词中获得了一些帮助。引用 Jacob@AIwithBenefits (@AIwithBenefits) 的话：非常喜欢新的 @phid...
- [open-interpreter/examples/screenpipe.ipynb at development · OpenInterpreter/open-interpreter](https://github.com/OpenInterpreter/open-interpreter/blob/development/examples/screenpipe.ipynb)：计算机的自然语言界面。通过在 GitHub 上创建账户来为 OpenInterpreter/open-interpreter 的开发做出贡献。

---

### **OpenInterpreter ▷ #**[**ai-content**](https://discord.com/channels/1146610656779440188/1149229778138824765/) (1 条消息):

facelessman: [https://youtu.be/VgJ0Cge99I0](https://youtu.be/VgJ0Cge99I0) -- 喜欢这一集 —— 喜欢这些人！！！

---

### **DSPy ▷ #**[**show-and-tell**](https://discord.com/channels/1161519468141355160/1202371242519441499/1298298965791473745) (2 条消息):

> - `新版本创建`
> - `升级流程`
> - `当前系统功能`

- **宣布创建新版本**：一位成员表示，与其修改现有的杰作，不如创建一个**新版本**，对此表现出极大的热情。
  
  - *非常感谢，这对我们意义重大* —— 另一位成员确认他们将在周一的直播中创建新版本。
- **讨论当前系统功能**：创建者计划在即将到来的直播环节中深入探讨**当前系统的工作原理**。
  
  - 他们还提到将讨论其**升级流程**，以清晰地展示正在进行的改进。

---

### **DSPy ▷ #**[**general**](https://discord.com/channels/1161519468141355160/1161519469319946286/1298035435880185898) (22 条消息🔥):

> - `AI 助手文档`
> - `失效链接`
> - `文档机器人回归`
> - `3.0 版本的总体氛围`

- **新文档中未实现 AI 助手**：一位成员注意到**小 AI 助手**没有实现在新的文档结构中，并对此表示失望。
  
  - *它消失了，非常难过* —— 这引起了社区的共鸣。
- **报告大量失效链接**：多位用户指出，在各个 DSPy 文档页面中存在导致 404 错误的**失效链接**。
  
  - 一位用户确认他们已经提交了 PR 来解决这个问题，得到了其他人的快速响应和感谢。
- **文档机器人回归**：成员们庆祝**文档机器人**的回归，对其功能表示热烈欢迎和感谢。
  
  - 社区反应积极，纷纷通过爱心表情和肯定言论支持机器人的存在。
- **询问 3.0 版本的总体氛围**：一位成员询问了关于 DSPy 3.0 版本的**总体氛围**，表现出对社区如何看待此次更新的兴趣。
  
  - 目前尚未有详细回应，社区情绪在很大程度上仍未得到充分讨论。

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1298043723938136074) (4 条消息):

> - `VividNode 桌面应用`
> - `Serverless RAG 应用`
> - `RFP 知识管理`
> - `Llama Impact Hackathon`
> - `向量数据库中的文档索引`

- **VividNode：在桌面端与 AI 模型对话**：**VividNode** 应用允许用户从桌面与 **GPT**、**Claude**、**Gemini** 和 **Llama** 进行交互，具有快速搜索和高级设置功能。此外，它还包括使用 **DALL-E 3** 或各种 Replicate 模型的图像生成功能，详见[公告](https://twitter.com/llama_index/status/1848484047607239041)。
  
  - 它旨在为寻求无缝 AI 通信体验的用户提供强大的聊天界面。
- **用 9 行代码构建 Serverless RAG 应用**：由 **@DBOS_Inc** 提供的教程展示了如何仅用 **9 行代码**部署使用 **LlamaIndex** 的 Serverless **RAG 应用**，与 **AWS Lambda** 相比显著降低了成本。如这篇 [推文](https://twitter.com/llama_index/status/1848509130631151646) 所述，该过程经过简化，可实现具有持久执行能力的弹性 AI 应用程序。
  
  - 该教程强调了开发者构建 AI 应用时的部署便捷性和成本效益。
- **通过知识管理增强 RFP 响应**：讨论强调了在向量数据库中索引文档如何辅助 **RFP 响应生成**，从而实现超越简单聊天回复的复杂工作流。这种方法允许 **LLM Agent** 生成具有上下文相关性的产物和响应，如 [文章](https://twitter.com/llama_index/status/1848759935787803091) 中所述。
  
  - 它强调了向量数据库在支持高级 AI 功能方面的多功能性。
- **加入 Llama Impact Hackathon！**：参与者可以参加在旧金山举行的为期 3 天的 **Llama Impact Hackathon**，重点是使用 **Llama 3.2** 模型构建解决方案。参赛团队有机会赢取 **15,000 美元奖金池**的一部分，其中包括为最佳使用 **LlamaIndex** 团队提供的 **1,000 美元奖金**，详见此 [公告](https://twitter.com/llama_index/status/1848807401971192041)。
  
  - 活动将于 **11 月 8 日至 10 日**举行，提供线下和线上参与选项。

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1298076417069678603) (14 条消息🔥):

> - `CondensePlusContextChatEngine 内存初始化`
> - `在 LlamaIndex 中限制 TPM 和 RPM`
> - `在动态数据中使用 GraphRag`
> - `使用 LlamaIndex API 解析 .docx 文件`
> - `在 Workflow 中持久化上下文`

- **CondensePlusContextChatEngine 自动初始化内存**：用户询问在连续提问时是否需要在 **CondensePlusContextChatEngine** 中初始化内存，并指出之前的版本在没有初始化的情况下也能正常工作。
  
  - 一位成员确认内存是*自动初始化*的，从而简化了用户体验。
- **在 LlamaIndex 中限制 TPM 和 RPM**：一位成员询问如何在 LlamaIndex 中限制 **TPM** 和 **RPM**，寻求一种自动化的解决方案。
  
  - 另一位成员澄清说，由于目前没有自动化的方法，用户必须手动限制索引速度或查询频率。
- **在动态数据中高效使用 GraphRag**：一位成员就如何在数据变化时高效使用 **GraphRag** 寻求建议，希望避免每次数据更新时都创建新的图。
  
  - 在收集到的讨论中，针对该问题没有提供直接的解决方案。
- **使用 LlamaIndex API 解析 .docx 文件**：成员们讨论了使用 **LlamaIndex API** 解析 **.docx** 文件是在本地还是在服务器上进行的。
  
  - 已确认解析数据将被发送到 **LlamaCloud** 进行处理。
- **在多次 Workflow 运行中持久化上下文**：一位用户询问如何实现同一 Workflow 在多次执行之间的上下文保留。
  
  - 一位成员提供了代码片段，展示了如何使用 `JsonSerializer` 序列化上下文并在以后恢复。

 

---

### **LlamaIndex ▷ #**[**ai-discussion**](https://discord.com/channels/1059199217496772688/1100478495295017063/1298297198324482109) (1 条消息):

> - `LaBSE 性能`
> - `sentence-transformers/多语言模型`

- **LaBSE 性能未达用户预期**：一位成员提到他们在大约一年前尝试过 **LaBSE**，发现其性能不尽如人意。
  
  - 他们特别提到该模型在处理其数据时**未能达到预期**。
- **多语言 MPNet 模型的问题**：同一位成员对 **sentence-transformers/paraphrase-multilingual-mpnet-base-v2** 表示失望，指出它在处理其新数据时也表现不佳。
  
  - 这进一步加剧了人们对某些**多语言模型**在处理多样化数据集时有效性的担忧。

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**hackathon-announcements**](https://discord.com/channels/1280234300012494859/1280236929379602493/1298035780807168021) (2 条消息):

> - `LLM Agents MOOC Hackathon`
> - `Hackathon FAQ`
> - `Sponsorship`
> - `Tracks of Hackathon`
> - `Sign-Up Details`

- **LLM Agents MOOC Hackathon 宣布举办**：Berkeley RDI 将于 10 月中旬至 12 月中旬举办 **LLM Agents MOOC Hackathon**，提供超过 **$200,000** 的奖金和额度。该活动对 **Berkeley** 学生和公众开放，旨在鼓励 AI 领域的创新。
  
  - 参与者可以通过提供的 [注册链接](https://docs.google.com/forms/d/e/1FAIpQLSevYR6VaYK5FkilTKwwlsnzsn8yI_rRLLqDZj0NH7ZL_sCs_g/viewform) 报名，并在 Hackathon 期间探索 **职业** 和 **实习** 机会。
- **赞助商致谢**：特别鸣谢包括 **OpenAI**、**GoogleAI**、**AMD** 等在内的赞助商对本次 Hackathon 的支持。他们的参与展示了 AI 领域重要参与者的强力支持，提升了活动的公信力。
  
  - @dawnsongtweets 分享的推文强调了 Hackathon 启动的热烈反响，并凭借强大的社区支持鼓励大家积极参与。
- **推出五个令人兴奋的 Hackathon 赛道**：邀请参与者探索五个不同的赛道：**Applications**、**Benchmarks**、**Fundamentals**、**Safety** 以及 **Decentralized & Multi-Agents**。每个赛道都代表了一个深入研究 LLM Agents 和 AI 性能各方面的独特机会。
  
  - 这使参与者能够基于 **前沿** 技术进行构建，并解决 AI 开发中的关键挑战。
- **创建了 Hackathon FAQ**：已创建一份全面的 **LLM Agents Hackathon FAQ** 以解答常见问题，可通过提供的 [FAQ 链接](https://docs.google.com/document/d/1P4OBOXuHRJYU9tf1KH_NQWvaZQ1_8wCfNi3MOnCw6RI/edit?usp=sharing) 访问。
  
  - 该资源将帮助潜在参与者解答疑问，并提升他们在活动开始前的体验。

**提到的链接**：

- [Hackathon FAQ](https://docs.google.com/document/d/1P4OBOXuHRJYU9tf1KH_NQWvaZQ1_8wCfNi3MOnCw6RI/edit?usp=sharing)：LLM Agents Hackathon FAQ。Hackathon 官网是什么？https://rdi.berkeley.edu/llm-agents-hackathon/ 我在哪里报名？https://docs.google.com/forms/d/e/1FAIpQLSevYR6VaYK5FkilTKwwlsnzsn8yI_rRLLqD...
- [来自 Dawn Song (@dawnsongtweets) 的推文](https://x.com/dawnsongtweets/status/1848431882498937295>)：🎉 为我们的 LLM Agents MOOC 展现出的惊人热情感到激动——已有 1.2 万+ 注册学员和 5000+ Discord 成员！📣 很高兴今天启动 LLM Agents MOOC Hackathon，面向所有人开放，提供 $200K+ 的奖金...

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-announcements**](https://discord.com/channels/1280234300012494859/1280369709623283732/1298003872929021972) (2 messages):

> - `Lecture 7 Announcement`
> - `LLM Agents MOOC Hackathon`
> - `TapeAgents Framework`
> - `WorkArena++ Benchmark`
> - `Hackathon Tracks and Sponsors`

- **第 7 讲直播即将开始**：今天的讲座由客座讲师 Nicolas Chapados 和 Alexandre Drouin 主讲，主题为 **AI Agents for Enterprise Workflows**（企业工作流中的 AI Agents），定于太平洋标准时间下午 3:00 开始，直播链接见 [此处](https://www.youtube.com/live/-yf-e-9FvOc)。
  
  - 本次会议将介绍 **TapeAgents framework**，并讨论能够自主使用浏览器的 **web agents**，以及该领域的开放性问题。
- **令人兴奋的 LLM Agents MOOC Hackathon 正式启动**：Berkeley RDI 宣布了 **LLM Agents MOOC Hackathon**，活动从 10 月中旬持续到 12 月中旬，为参赛者提供超过 **200,000 美元的奖金和额度 (credits)**。报名详情请见 [此处](https://docs.google.com/forms/d/e/1FAIpQLSevYR6VaYK5FkilTKwwlsnzsn8yI_rRLLqDZj0NH7ZL_sCs_g/viewform)。
  
  - 本次 Hackathon 向所有人开放，设有**五个赛道**，重点关注应用、Benchmarks、安全性等，并得到了 **OpenAI** 和 **GoogleAI** 等主要赞助商的支持。
- **Nicolas Chapados：企业的 AI**：ServiceNow Inc. 研究副总裁 Nicolas Chapados 将在讲座中分享关于推进企业级 **generative AI** 的见解。他的背景包括在 2021 年 ServiceNow 收购 **Element AI** 之前共同创立了多家机器学习初创公司。
  
  - 演讲将强调 **TapeAgents** 等框架的重要性，并解决 AI 中诸如**安全性与可靠性**等关键问题。

**提到的链接**：

- [CS 194/294-196 (LLM Agents) - Lecture 7, Nicolas Chapados and Alexandre Drouin](https://www.youtube.com/live/-yf-e-9FvOc.)：未找到描述
- [来自 Dawn Song (@dawnsongtweets) 的推文](https://x.com/dawnsongtweets/status/1848431882498937295>)：🎉 为我们的 LLM Agents MOOC 展现出的巨大热情感到激动——已有 1.2 万+ 注册学员和 5000+ Discord 成员！📣 很高兴今天启动 LLM Agents MOOC Hackathon，向所有人开放，奖金超过 20 万美元...

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-questions**](https://discord.com/channels/1280234300012494859/1280370030609170494/1298064448854163537) (7 messages):

> - `Workflow understanding for AI agents`
> - `Assignment deadlines`
> - `Learning resources for auto gen`

- **理解人类工作流可增强 AI 解决方案**：一场讨论强调了在应用 AI 解决方案时理解任务**当前工作流**的重要性，建议转而关注 Agent 可用的**能力 (capabilities)** 和工具。
  
  - 会上指出，Agent 可能不需要直接复制人类的任务。
- **文章作业截止日期确认**：一位成员询问了 **Written Article Assignment** 的截止日期，并得到确认所有作业均需在 **太平洋标准时间 12 月 12 日晚上 11:59** 之前提交。
  
  - 这为所有参与者明确了提交时间表。
- **分享了学习 AutoGen 的课程**：一位成员寻求学习 **AutoGen** 的资源，另一位成员向其推荐了 [AI Agentic Design Patterns with Autogen](https://learn.deeplearning.ai/courses/ai-agentic-design-patterns-with-autogen/lesson/1/introduction) 专项课程。
  
  - 这为有兴趣掌握 AutoGen 的人员提供了结构化的学习机会。

 

**提到的链接**：[AI Agentic Design Patterns with AutoGen - DeepLearning.AI](https://learn.deeplearning.ai/courses/ai-agentic-design-patterns-with-autogen/lesson/1/introduction)：使用 AutoGen 框架构建具有多样化角色和能力的 multi-agent 系统，以实现复杂的 AI 应用。

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-lecture-discussion**](https://discord.com/channels/1280234300012494859/1282734248112947210/1298045919941492738) (4 messages):

> - `Lecture Start Time`
> - `YouTube Stream Issues`

- **今日讲座开始确认**：在最初的一些不确定之后，成员们确认讲座已经开始。
  
  - 最新的消息指出“现在刚刚开始”，消除了所有疑虑。
- **YouTube 直播流没有声音**：一位成员报告收到了 YouTube 信号，但遇到了**没有声音且画面不动**的问题。
  
  - 不过，他们随后更新称视频现在已经开始，标志着其流媒体问题已得到解决。

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-readings-discussion**](https://discord.com/channels/1280234300012494859/1282735578886181036/1298092534777511978) (4 条消息):

> - `Function Calling in LLMs`
> - `TapeAgents Framework`
> - `Agent Development`
> - `Model Distillation Techniques`

- **理解 LLMs 中的 Function Calling**：关于 LLMs 如何将任务拆分为 function calls 似乎存在困惑，有人请求提供代码示例。
  
  - 一位成员澄清说，目前的讨论涉及 LLMs 中 function calling 的概念。
- **介绍 TapeAgents 框架**：来自 ServiceNow 的团队介绍了 **TapeAgents**，这是一个旨在进行 Agent 开发和优化的新框架，它使用一种称为 tape 的结构化 Agent 日志。
  
  - 该框架实现了细粒度控制、逐步调试、可恢复会话和流式传输，正如讨论和链接的 [论文](https://www.servicenow.com/research/TapeAgentsFramework.pdf) 中所述。
- **为 Agents 使用 Tapes 的好处**：作为 TapeAgents 框架的一部分，tape 充当粒度化的结构化日志，增强了对 Agent 会话的控制和优化。
  
  - 有人指出，所有交互都通过这个 tape 日志进行，从而提供对 Agent 性能和配置的全面见解。
- **Agent 框架资源**：成员们分享了宝贵的资源，包括与 TapeAgents 框架相关的 [GitHub 仓库](https://github.com/ServiceNow/TapeAgents) 以及讨论该论文的有用的 [讨论帖](https://threadreaderapp.com/thread/1846611633323291055.html)。
  
  - 这些资源旨在支持社区探索先进的 Agent 框架和方法论。

**提到的链接**：

- [@DBahdanau 在 Thread Reader App 上的讨论帖](https://threadreaderapp.com/thread/1846611633323291055.html)：@DBahdanau：🚨 新的 Agent 框架！🚨 我在 @ServiceNowRSRCH 的团队正在发布 TapeAgents：一个用于 Agent 开发和优化的整体框架。其核心是 tape：一个结构化的 Agent 日志...
- [OpenAI Cookbook](https://cookbook.openai.com/)：使用 OpenAI API 构建应用的开源示例和指南。浏览代码片段、高级技术和演练集合。分享你自己的示例和指南。

---

### **Torchtune ▷ #**[**general**](https://discord.com/channels/1216353675241590815/1216353675744641096/1298005342604759071) (16 messages🔥):

> - `PyTorch Core Issue`
> - `Error in Distributed Training`
> - `Config File Format in Torchtune`
> - `Flex Performance on GPUs`
> - `Hardware Setup for Training`

- **带有警告的 PyTorch 核心问题**：一位用户分享了一个开始出现的警告，现在该警告在 **float16** 上也会触发，但在 **float32** 上不会，建议通过使用不同的 **kernel** 来测试性能影响。
  
  - 有推测认为 [PyTorch 源码](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/MHA.cpp#L677) 中的某些行可能会影响 **JIT** 行为。
- **分布式训练中遇到的错误**：一位用户报告了在指定 **CUDA_VISIBLE_DEVICES** 运行 `tune` 命令时遇到错误，称程序在代码的某一行停止，且没有进一步的消息。
  
  - 在移除 **CUDA_VISIBLE_DEVICES** 指定后，错误仍然存在，这表明配置或设置中存在更深层次的问题。
- **对配置文件格式的困惑**：有人指出，为配置文件使用 `.yaml` 扩展名可能会误导 **Torchtune**，使其将其错误地解释为本地配置。
  
  - 这强调了验证文件命名的必要性，以避免运行时的意外问题。
- **Flex 在 800 GPU 上的性能**：讨论中提到 **Flex** 在 **3090s** 和 **4090s** 上运行良好，一位用户提到了在更大的 GPU（如 **A800s**）上优化内存使用的潜力。
  
  - 对话中提到了更快的显存溢出（**oom**）操作，特别是在 **head dimensions** 较大时。
- **训练的硬件设置**：一位用户确认拥有 **8x A800** GPU，并讨论了在该环境下训练设置中的性能问题。
  
  - 另一位用户询问是否可以用更少的 GPU 进行测试，以便更有效地排查报告的错误。

**提到的链接**：

- [pytorch/aten/src/ATen/native/cudnn/MHA.cpp at main · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/MHA.cpp#L677)：Python 中具有强 GPU 加速的 Tensor 和动态神经网络 - pytorch/pytorch
- [(WIP)feat: add gemma2b variants by Optimox · Pull Request #1835 · pytorch/torchtune](https://github.com/pytorch/torchtune/pull/1835#discussion_r1808975136)：上下文 此 PR 的目的是什么？是添加新功能、修复 bug、更新测试和/或文档还是其他（请在此处添加）。这与添加 gemma2 支持相关 #1813 更新日志...
- [Issues · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/133254)：Python 中具有强 GPU 加速的 Tensor 和动态神经网络 - Issues · pytorch/pytorch

---

### **Torchtune ▷ #**[**dev**](https://discord.com/channels/1216353675241590815/1236040539409879170/1298020196157624370) (1 messages):

> - `Hermes 2.5.0 Release`
> - `Recommendations for nightlies`

- **Hermes 2.5.0 发布引发讨论**：随着 **Hermes 2.5.0** 的发布，成员们讨论了是否继续为某些 **recipes** 推荐 **nightly builds**，以避免显存溢出（**OOM**）错误。
  
  - 提出了 *移除 nightly 推荐* 的建议，以增强用户体验并减轻潜在问题。
- **对 Nightly Builds 的担忧**：一位成员对 **recipes** 使用 **nightly builds** 表示担忧，担心这可能导致系统不稳定和性能问题。
  
  - 讨论强调了稳定版本优于实验性 **nightly** 版本的重要性，以减少技术困难。

---

### **LangChain AI ▷ #**[**general**](https://discord.com/channels/1038097195422978059/1038097196224086148/1298237516280762399) (6 条消息):

> - `Langchain Open Canvas 兼容性`
> - `使用 Langchain 进行 Agent 编排`
> - `最终输出链重构`
> - `Langchain 0.3.4 恶意软件警告`
> - `企业级应用的本地托管`

- **Langchain Open Canvas 寻求兼容性**：一名成员询问 **Langchain Open Canvas** 是否可以与 **Anthropic** 和 **OpenAI** 之外的 LLM 提供商协作。
  
  - 这反映了用户对扩展不同提供商兼容性的持续关注。
- **使用 Langchain 进行 Agent 编排的可能性**：另一名成员询问 **Langchain** 是否可以辅助使用 **OpenAI Swarm** 进行 Agent 编排，还是需要进行自定义编程。
  
  - 回复指出已有可用的库支持此功能。
- **重构输出链以提升功能性**：一位用户在讨论是重构其现有的 **Langchain** 工作流，还是迁移到 **LangGraph** 以获得更好的功能。
  
  - 他们目前的设置涉及复杂的工具使用并输出 JSON 响应，因此需要进行策略性调整。
- **对 Langchain 0.3.4 恶意警告的担忧**：一位用户报告称 **PyCharm** 对 **Langchain 0.3.4** 中的依赖项发出了**恶意软件 (malicious)** 警告，并指出了重大的安全风险。
  
  - 他们询问是否有人遇到过类似问题，并对潜在的误报表示担忧。
- **企业级应用的本地托管解决方案**：一位用户就如何在没有互联网连接的情况下为企业级应用**本地托管**推理模型寻求建议。
  
  - 他们考虑使用 **Flask** 或 **FastAPI** 构建**推理容器 (inference container)**，同时希望如果有更好的成熟方案，尽量避免重复造轮子。

---

### **LangChain AI ▷ #**[**share-your-work**](https://discord.com/channels/1038097195422978059/1038097372695236729/1298290801603772456) (2 条消息):

> - `NumPy 文档改进`
> - `转型咨询领域`

- **增强关于浮点精度的 NumPy 文档**：一名成员庆祝他们成功为 [NumPy library](https://github.com/numpy/numpy/pull/27602) 做出贡献，重点是改进关于**浮点精度 (floating-point precision)** 的文档。
  
  - 他们增加了一个章节来解释浮点运算的细微差别，以帮助用户（尤其是初学者）处理计算中的**微小误差**。
- **资深工程师转型咨询**：另一名成员介绍自己是一位拥有 10 多年经验的**高级软件工程师**，目前正在从编码工作转型为咨询角色。
  
  - 他们邀请其他人直接联系寻求帮助，并展示了其 [GitHub profile](https://github.com/0xdeity) 以提供更多背景信息。

**提到的链接**：

- [0xdeity - 概览](https://github.com/0xdeity)：技术爱好者 | 软件架构师 | 开源贡献者 - 0xdeity
- [由 amitsubhashchejara 更新关于浮点精度和行列式计算的文档 · Pull Request #27602 · numpy/numpy](https://github.com/numpy/numpy/pull/27602)：此拉取请求更新了 NumPy 中与浮点精度相关的文档，特别是解决了某些矩阵的行列式计算不正确的问题。增加了一个注意点...

---

### **LangChain AI ▷ #**[**tutorials**](https://discord.com/channels/1038097195422978059/1077843317657706538/1298124573522268243) (1 条消息):

> - `Self-Attention`
> - `动态上下文嵌入`

- **NLP 中 Self-Attention 的探索**：一名成员分享了一篇 [Medium 文章](https://medium.com/@d.isham.ai93/self-attention-in-nlp-from-static-to-dynamic-contextual-embeddings-4e26d8c49427)，详细介绍了 NLP 中 **Self-Attention** 机制从静态到**动态上下文嵌入 (dynamic contextual embeddings)** 的演变。
  
  - 文章讨论了这种转变如何通过使模型更好地捕捉上下文细微差别来增强性能。
- **动态上下文嵌入的转型**：文章强调了**动态上下文嵌入**在提高 NLP 任务中模型性能和适应性方面的重要性。
  
  - 它重点介绍了证明有效实施的案例研究，这些实施相比静态方法有了显著改进。

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general**](https://discord.com/channels/1104757954588196865/1104757955204743201/1298063945814249524) (2 messages):

> - `Experimental Triton FA support`
> - `User Warning on Flash Attention`

- **2.5.0 带来实验性 Triton FA 支持**：版本 **2.5.0** 通过 aotriton 为 **gfx1100** 添加了实验性的 **Triton Flash Attention (FA)** 支持，需使用 `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`。
  
  - 该设置已启用，但导致了关于 **Navi31 GPU** 上的 Flash Attention 支持仍处于实验阶段的 **UserWarning**。
- **对 Flash Attention 警告的误解**：用户收到一条 **UserWarning**，指出 Navi31 上的 Flash Attention 是实验性的，需要通过 `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` 启用。
  
  - 最初，他们误以为该警告与 **Liger** 有关，因此忽略了其重要性，正如在 [GitHub issue](https://github.com/ROCm/aotriton/issues/16#issuecomment-2346675491) 中进一步讨论的那样。

**提及的链接**：[[Feature]: Memory Efficient Flash Attention for gfx1100 (7900xtx) · Issue #16 · ROCm/aotriton](https://github.com/ROCm/aotriton/issues/16#issuecomment-2346675491)：建议描述：开始使用 torchlearn 在我的 gfx1100 显卡上使用 pytorch 训练模型，但收到警告称 torch 编译时未包含 memory efficient flash attention。我看到有...

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general-help**](https://discord.com/channels/1104757954588196865/1110594519226925137/1298212331515285557) (6 messages):

> - `Instruction-Tuned Models`
> - `Domain-Specific Instruction Data`
> - `Catastrophic Forgetting`
> - `Raw Domain Data`
> - `GPT-4 Generated Instruction Data`

- **利用指令微调模型进行训练**：一位成员建议使用像 **llama-instruct** 这样的指令微调模型进行指令训练，并强调如果用户不介意其之前的微调，这将具有优势。
  
  - 他们推荐了混合策略，但也承认*实验*对于找到正确的平衡至关重要。
- **对灾难性遗忘的担忧**：一位成员提出了关于是仅使用特定领域的指令数据，还是混合通用数据以避免**灾难性遗忘 (Catastrophic Forgetting)** 的担忧。
  
  - 建议是探索各种方法以确定最佳方案，这反映了模型训练的复杂性。
- **预训练 vs 指令微调**：讨论重点在于，是应该从基座模型 (base model) 开始在原始领域数据上进行持续预训练，然后再进行指令微调，还是直接使用指令微调模型。
  
  - 一位成员建议，如果可以获得原始数据，应在开始时利用它，因为这可能提供更坚实的基础。
- **从原始文本生成指令数据**：一位成员表达了他们计划使用 **GPT-4** 从原始文本生成指令数据的计划，并指出了潜在的偏见和覆盖范围限制。
  
  - 这种方法可以减轻对人工创建的指令数据的依赖，同时也承认了可能的缺点。

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #**[**discussion**](https://discord.com/channels/1111172801899012102/1111353033352294440/1298210259902402560) (2 messages):

> - `Function Calling Model Fine-tuning`
> - `Benchmarking Custom Endpoints`
> - `Gorilla Project Documentation`

- **用于函数调用的微调模型**：一位用户在为函数调用 (function calling) 微调模型并创建了自己的推理 API 后，分享了发现 **Gorilla 项目** 的兴奋之情。
  
  - 他们询问了对自定义端点进行基准测试的方法，并寻求有关该过程的文档。
- **添加新模型的说明**：作为回应，一位成员强调了一个 [README 文件](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#contributing)，其中提供了关于如何将新模型添加到排行榜 (leaderboard) 的说明。
  
  - 这些完整的文档支持用户有效地为 **Gorilla** 项目做出贡献。

**提及的链接**：[gorilla/berkeley-function-call-leaderboard at main · ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#contributing)：Gorilla：训练和评估用于函数调用（工具调用）的 LLM - ShishirPatil/gorilla

---

### **LAION ▷ #**[**resources**](https://discord.com/channels/823813159592001537/991938328763056168/1298218797450264637) (1 条消息):

> - `Webinar on LLM Best Practices` (LLM 最佳实践研讨会)
> - `Prompt Engineering Techniques` (Prompt Engineering 技术)
> - `Performance Optimization` (性能优化)
> - `Retrieval-Augmented Generation` (Retrieval-Augmented Generation)
> - `Analytics Vidhya Blog Articles` (Analytics Vidhya 博客文章)

- **加入免费的 LLM 研讨会**：一位来自 Meta 的高级 ML Engineer 正在主持一场关于 **构建 LLM 最佳实践** 的免费研讨会，目前已有近 **200 人报名**。
  
  - 您可以通过点击此 [链接](https://shorturl.at/ulrCN) 注册活动，以获取关于高级 Prompt Engineering 技术、模型选择和项目规划的见解。
- **关于 Prompt Engineering 的见解**：研讨会将涵盖 **高级 Prompt Engineering 技术**，帮助参与者提升技能并学习如何做出战略决策。
  
  - 参与者还将深入了解对于有效部署 LLM 项目至关重要的 **性能优化** 方法。
- **探索 Retrieval-Augmented Generation**：您将学习 **Retrieval-Augmented Generation (RAG)** 以及它如何增强 LLM 解决方案的有效性。
  
  - 研讨会还将讨论 **Fine-tuning**，将其作为最大化模型性能的关键策略。
- **Analytics Vidhya 专题文章**：研讨会参与者的优秀文章将发表在 **Analytics Vidhya 的博客空间**，从而获得曝光和认可。
  
  - 这一机会为那些希望与更广泛受众分享见解的人提升了会议的价值。

 

**提到的链接**：[与专家主导的活动一起探索 AI 的未来](https://shorturl.at/ulrCN)：Analytics Vidhya 是 Analytics、Data Science 和 AI 专业人士的领先社区。我们正在培养下一代 AI 专业人士。获取最新的 Data Science、Machine Learning 和 A...

 

---

### **Mozilla AI ▷ #**[**announcements**](https://discord.com/channels/1089876418936180786/1089876419926032396/1298017406215786526) (1 条消息):

> - `AI access challenges` (AI 访问挑战)
> - `Competition in AI` (AI 领域的竞争)
> - `External researcher access` (外部研究人员访问)
> - `Big Tech and AI control` (大型科技公司与 AI 控制权)

- **Mozilla 关于 AI 访问挑战的研究**：Mozilla 委托了两项研究报告：来自 AWO 的《[外部研究人员对封闭基础模型的访问](https://blog.mozilla.org/wp-content/blogs.dir/278/files/2024/10/External-researcher-access-to-closed-foundation-models.pdf)》以及来自 Open Markets Institute 的《[阻止大型科技公司成为大型 AI](https://blog.mozilla.org/wp-content/blogs.dir/278/files/2024/10/Stopping-Big-Tech-from-Becoming-Big-AI.pdf)》。
  
  - 这些报告揭示了谁在控制 AI 的发展，并概述了建立公平开放的 AI 生态系统所需的变革。
- **关于 AI 研究结果的博客文章**：有关受托研究的更多细节可以在 [此处的博客文章](https://discord.com/channels/1089876418936180786/1298015953463808102) 中找到。
  
  - 该文章强调了这些发现对未来 AI 格局以及主要参与者之间竞争的影响。

 

---

---

---

---

---

{% else %}

> 为了便于邮件阅读，完整的频道细分内容已被截断。
> 
> 如果您想查看完整细分，请访问此邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
> 
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}