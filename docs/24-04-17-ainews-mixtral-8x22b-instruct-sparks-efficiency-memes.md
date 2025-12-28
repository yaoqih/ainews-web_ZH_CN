---
companies:
- mistral-ai
- hugging-face
- google
- microsoft
- intel
- softbank
- nvidia
date: '2024-04-17T21:02:34.918866Z'
description: '以下是为您翻译的中文内容：


  **Mistral** 发布了其 **Mixtral 8x22B** 模型的指令微调版本。该模型的一大亮点是推理时仅需 **390 亿（39B）激活参数**，性能却优于更大规模的模型。它支持
  **5 种语言**，拥有 **64k 上下文窗口**，并具备强大的数学和代码处理能力。该模型已在 **Hugging Face** 上线，采用 **Apache
  2.0 协议**，支持本地部署使用。


  **谷歌**计划在 AI 领域投资超过 **1000 亿美元**，微软、英特尔和软银等巨头也纷纷投入巨资。


  **英国**已将未经同意制作的深度伪造（deepfake）色情内容定为刑事犯罪，这引发了关于执法难度的讨论。


  一位前**英伟达（Nvidia）**员工声称，英伟达在 AI 芯片领域的领先地位在未来十年内都无法被撼动。


  **AI 伴侣**市场规模有望达到 **10 亿美元**。


  AI 在多项基础任务上已超越人类，但在处理复杂任务时仍显不足。


  **Zyphra** 推出了 **Zamba**，这是一种新型的 7B 参数混合模型。在训练数据更少的情况下，其性能超越了 **LLaMA-2 7B** 和 **OLMo-7B**。该模型是在
  128 块 H100 GPU 上历时 30 天训练完成的。


  **GroundX** API 进一步提升了检索增强生成（RAG）的准确性。'
id: 0b735461-2b40-4353-8a4a-33b39cfeec54
models:
- mixtral-8x22b
- llama-2-7b
- olmo-7b
original_slug: ainews-mixtral-8x22b-instruct-defines-frontier
people:
- guillaume-lample
- osanseviero
- _philschmid
- svpino
title: Mixtral 8x22B Instruct 引发了关于效率的梗。
topics:
- multilinguality
- math
- code-generation
- context-window
- model-performance
- model-release
- retrieval-augmented-generation
- deepfake
- ai-investment
- ai-chip
- hybrid-architecture
- training-data
---

<!-- buttondown-editor-mode: plaintext -->> 2024/4/16-4/17 的 AI 新闻。我们为您检查了 6 个 subreddits、[**364** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **27** 个 Discords（**395** 个频道，以及 **5173** 条消息）。预计为您节省阅读时间（以 200wpm 计算）：**587 分钟**。

按照其既定模式，Mistral 在发布磁力链接后紧接着发布了[一篇博客文章](https://mistral.ai/news/mixtral-8x22b/)，以及其 8x22B 模型的 Instruct 微调版本：

 
![image.png](https://assets.buttondown.email/images/323db65b-608d-445d-83eb-1d6d9ce35e3f.png?w=960&fit=max)
 

这张图片最终引发了 [Databricks, Google, 和 AI21](https://twitter.com/AlbertQJiang/status/1780648008696091003) 之间的友好竞争，所有这些都仅仅强调了 Mixtral 在激活参数（active params）和 MMLU 性能之间创造了新的权衡：


![image.png](https://assets.buttondown.email/images/9677f3b7-64ba-4f12-af15-291dfda26c7d.png?w=960&fit=max)


当然，未言明的是激活参数数量与运行 Dense 模型的成本并不呈线性相关，而且单一关注 MMLU 对于那些不那么严谨的竞争对手来说并不理想。


---

**目录**

[TOC] 


---

# AI Reddit 摘要

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/Singularity。评论抓取现在可以运行了，但仍有很大改进空间！

**AI 投资与进展**

- **科技巨头的巨额 AI 投资**：在 /r/singularity 中，DeepMind CEO 透露 Google 计划在 AI 领域投资超过 **1000 亿美元**，其他科技巨头如 Microsoft, Intel, SoftBank 和一家阿布扎比基金也做出了类似的巨额押注，[表明了对 AI 潜力的极高信心](https://www.bloomberg.com/news/articles/2024-04-16/deepmind-ceo-says-google-will-spend-more-than-100-billion-on-ai)。

- **英国将非自愿 deepfake 色情内容定为犯罪**：英国已将未经同意创建性显性 deepfake 图像定为犯罪。在 /r/technology 中，评论者们[辩论了其影响和执法挑战](https://time.com/6967243/uk-criminalize-sexual-explicit-deepfake-images-ai/)。

- **Nvidia 的 AI 芯片主导地位**：在 /r/hardware 中，一位前 Nvidia 员工在 Twitter 上声称[这十年内没有人能赶上 Nvidia 的 AI 芯片领先地位](https://i.redd.it/m388weqd9yuc1.png)，引发了关于该公司强势地位的讨论。

**AI 助手与应用**

- **AI 伴侣潜在的十亿美元市场**：在 /r/singularity 中，一位科技高管预测 AI 女友可能成为一项 **10 亿美元的业务**。评论者认为这被大大低估了，并[讨论了其社会影响](https://www.yahoo.com/tech/tech-exec-predicts-ai-girlfriends-181938674.html?)。

- **语言模型的无限上下文长度**：/r/artificial 发布的一条推文宣布了[无限上下文长度（unlimited context length）](https://twitter.com/_akhaliq/status/1780083267888107546?t=hnN1bujYWqBlynr_zEqHKA&s=19)，这是 AI 语言模型的一项重大进步。

- **AI 在基础任务上超越人类**：在 /r/artificial 中，一篇 Nature 文章报道称 [AI 在几项基础任务上的表现已超越人类](https://www.nature.com/articles/d41586-024-01087-4)，尽管在更复杂的任务上仍处于落后地位。

**AI 模型与架构**

- **Zamba：新型 7B 参数混合架构**：在 /r/LocalLLaMA 中，Zyphra 推出了 Zamba，这是一种将 Mamba 模块与共享注意力（shared attention）相结合的 7B 参数混合架构。尽管训练数据较少，但它的[表现优于 LLaMA-2 7B 和 OLMo-7B 等模型](https://www.reddit.com/r/LocalLLaMA/comments/1c61k7v/zamba_a_7b_mambalike_ssm_hybrid_model_trained_for/)。该模型由一个 7 人团队使用 128 块 H100 GPU 历时 30 天开发完成。

---

# AI Twitter 摘要回顾

> 所有摘要均由 Claude 3 Opus 生成（4 次运行中的最佳结果）。我们正在使用 Haiku 进行聚类和流程工程（flow engineering）。

**Mixtral 8x22B Instruct 模型发布**

- **令人印象深刻的性能**：[@GuillaumeLample](https://twitter.com/GuillaumeLample/status/1780602023203029351) 宣布发布 Mixtral 8x22B Instruct，该模型在推理过程中仅使用 **39B 激活参数**，性能显著超越现有的开源模型，且速度比 70B 模型更快。
- **多语言能力**：[@osanseviero](https://twitter.com/osanseviero/status/1780595541711454602) 强调 Mixtral 8x22B 精通 **5 种语言**（英语、法语、意大利语、德语、西班牙语），具备**数学和代码能力**，并拥有 **64k 上下文窗口**。
- **可用性**：该模型已在 [@huggingface](https://twitter.com/huggingface) Hub 上以 **Apache 2.0 许可证**发布，可以下载并在本地运行，正如 [@_philschmid](https://twitter.com/_philschmid/status/1780598146470379880) 所确认的那样。

**RAG (Retrieval-Augmented Generation) 进展**

- **GroundX 提升准确率**：[@svpino](https://twitter.com/svpino/status/1780571442096087224) 分享了 @eyelevelai 发布的 GroundX，这是一个先进的 RAG API。在对 1,000 页税务文件的测试中，**GroundX 达到了 98% 的准确率**，而 LangChain 为 64%，LlamaIndex 为 45%。
- **评估风险的重要性**：[@omarsar0](https://twitter.com/omarsar0/status/1780613738585903182) 根据一篇关于 RAG 模型忠实度的论文，强调了在使用可能包含支持性、矛盾性或错误数据的上下文信息时，评估 LLM 风险的必要性。
- **LangChain RAG 教程**：[@LangChainAI](https://twitter.com/LangChainAI/status/1780629875533181271) 在 @freeCodeCamp 上发布了一个解释 RAG 基础和高级方法的播放列表。他们还分享了一个关于使用 Mixtral 8x22B 进行 RAG 的 [@llama_index](https://twitter.com/llama_index/status/1780646484712788085) 教程。

**Snowflake Arctic Embed 模型**

- **强大的 Embedding 模型**：[@SnowflakeDB](https://twitter.com/SnowflakeDB) 在 [@huggingface](https://twitter.com/huggingface) 上开源了其 Arctic 系列 Embedding 模型。正如 [@RamaswmySridhar](https://twitter.com/RamaswmySridhar/status/1780225794402627946) 所言，这些模型是 @Neeva 的搜索专业知识与 Snowflake 对 AI 投入的结晶。
- **效率与性能**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1780621521230111181) 强调了这些模型的效率，**参数量从 23M 到 335M 不等**，**序列长度从 512 到 8192**，在不使用 RPE 的情况下支持高达 2048 个 Token，使用 RPE 则可支持 8192 个 Token。
- **LangChain 集成**：[@LangChainAI](https://twitter.com/LangChainAI/status/1780650806896947547) 宣布其 @huggingface Embeddings 连接器已实现对 Snowflake Arctic Embed 模型的同日支持。

**其他**

- **CodeQwen1.5 发布**：[@huybery](https://twitter.com/huybery/status/1780264890298720570) 介绍了 CodeQwen1.5-7B 和 CodeQwen1.5-7B-Chat，这是使用 **3T Token** 代码数据预训练的专业代码 LLM。它们在代码生成、长上下文建模 (64K)、代码编辑和 SQL 能力方面表现出色，在 SWE-Bench 测试中超越了 ChatGPT-3.5。
- **波士顿动力（Boston Dynamics）的新机器人**：[@DrJimFan](https://twitter.com/DrJimFan/status/1780622682561929645) 分享了波士顿动力新机器人的视频，并认为人形机器人的供应量将在未来十年内超过 iPhone，且“人类水平”只是一个人为设定的上限。
- **从第一天起就具备超人类能力的 AI**：[@ylecun](https://twitter.com/ylecun/status/1780596362415063217) 表示，AI 助手从一开始就需要具备类人智能以及超人类能力，这需要对物理世界的理解、持久记忆、推理和分层规划（hierarchical planning）。

---

# AI Discord 摘要

> 摘要的摘要的摘要

**Stable Diffusion 3 和 Stable Diffusion 3 Turbo 发布**：

- **Stability AI** 推出了 **Stable Diffusion 3** 及其更快的变体 **Stable Diffusion 3 Turbo**，声称其性能优于 DALL-E 3 和 Midjourney v6。这些模型采用了全新的 **Multimodal Diffusion Transformer (MMDiT)** 架构。
- 计划通过 Stability AI Membership 发布 SD3 权重以供自托管，延续其开放生成式 AI 的路线。
- 社区正在等待关于 SD3 个人与商业用途许可的进一步说明。

**Unsloth AI 进展**：

- 讨论了 **GPT-4** 作为 **GPT-3.5** 的微调迭代版本，以及 **Mistral7B** 令人印象深刻的多语言能力。
- 对在 Apache 2.0 协议下开源发布的 **Mixtral 8x22B** 感到兴奋，该模型在多语言流利度和长上下文窗口方面具有优势。
- 有兴趣为 Unsloth AI 的文档做贡献，并考虑通过捐赠支持其开发。

**WizardLM-2 亮相及随后的下架**：

- Microsoft 发布了 **WizardLM-2** 系列，包括 8x22B、70B 和 7B 模型，展示了极具竞争力的性能。
- 然而，**WizardLM-2** 因缺乏合规性审查而被下架，并非最初推测的毒性问题。
- 下架引发了困惑和讨论，一些用户表示有兴趣获取原始版本。

- **Stable Diffusion 3 发布，性能提升**：**Stability AI** 已发布 **Stable Diffusion 3** 和 **Stable Diffusion 3 Turbo**，现已在他们的 [Developer Platform API](https://bit.ly/3xHrtjG) 上可用，号称拥有最快且最可靠的性能。社区正在等待关于自托管 SD3 权重的 **Stability AI Membership** 模式的澄清。同时，**SDXL finetunes** 已使 SDXL refiners 几乎过时，用户讨论了 **ComfyUI** 中的模型合并挑战以及 **diffusers** 流水线的局限性。

- **WizardLM-2 在兴奋与不确定中首次亮相**：Microsoft 发布的 **WizardLM-2** 模型激发了人们对其在开源格式下实现 **类 GPT-4 能力** 的热情。然而，由于遗漏合规审查导致模型突然下架，引发了困惑和猜测。用户对比了 WizardLM-2 各变体的性能，并分享了解决 **LM Studio** 兼容性问题的技巧。

- **多模态模型随 Idefics2 和 Reka Core 共同进步**：**Hugging Face 的 Idefics2 8B** 和 **Reka Core** 已成为强大的多模态语言模型，展示了在视觉问答、文档检索和编码方面的卓越能力。即将推出的 Idefics2 聊天变体以及 Reka Core 对抗行业巨头的竞争表现引起了极大关注。讨论还涉及 **JetMoE-8B** 等模型的成本效益，以及用于文本嵌入的 **Snowflake Arctic embed 系列** 的发布。

其他值得关注的主题包括：

- 引入了 **ALERT**，一个[用于评估大语言模型安全性的基准测试](https://github.com/Babelscape/ALERT)，以及关于 AI 安全标准的辩论。
- 探索用于视觉应用的 **检索增强生成 (RAG)**，以及 **World-Sim** 中 AI 模拟的哲学意义。
- [Payman AI](https://www.paymanai.com/) 等 AI 与人类协作平台的兴起，以及 **Supabase edge functions** 中 AI 推理的集成。
- 对 **Chinchilla scaling laws** 的挑战，以及研究界对 **状态空间模型 (state-space models)** 表达能力的讨论。
- **PEFT 方法**（如 Dora 和 RSLoRA）的进展，以及利用 **混合专家 (MoE)** 方法进行 **多语言模型扩展** 的追求。

---



# PART 1: High level Discord summaries

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

**Stable Diffusion 3 Turbo 震撼登场**：Stability AI 推出了 **Stable Diffusion 3** 和 **Stable Diffusion 3 Turbo**，现已在他们的 Developer Platform API 上线。在 Fireworks AI 的支持下，该公司声称其拥有最快且最可靠的性能。感兴趣的各方可以在 [Stable Diffusion 3 & Developer API](https://bit.ly/3xHrtjG) 开始使用 SD3。此外，Stability AI 承诺将采取开放的生成式 AI 路线，并计划为会员提供可自行托管的模型权重。

**精炼视觉直观的生成式 AI**：**SDXL finetunes** 已使 SDXL refiners 的使用几乎过时，因为前者在 **Civitai** 的下载中已非常普遍。这表明集成微调优于独立 refiner 模块的趋势，反映了社区驱动的优化方向。

**模型融合探索**：**ComfyUI** 内部关于模型融合策略的讨论非常活跃，涉及 V-prediction 和 epsilon 等复杂机制。这突显了社区通过这些方法实现更佳效果的实验精神，同时也承认正确的实现对于防止不可预测的结果至关重要。

**应对 Diffusers 库的限制**：围绕 **diffusers** 流水线的局限性和依赖关系展开了讨论，重点关注 **Stable Video Diffusion Pipeline** 面临的挑战。尽管存在这些挑战，一些用户正通过在下载后独立运行模型来优化使用，从而绕过 **Hugging Face** 库的某些约束。

**等待 SD3 会员模式详情**：社区正热切等待 Stability AI 澄清 **Stable Diffusion 3** 针对个人与商业用途的许可细节，特别是针对获取自行托管权重的新会员模式。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**GPT-4 优于 GPT-3.5**：新一代 GPT（**GPT-4**）被认为是基于 GPT-3.5 的微调增强版，尽管未提供具体的性能指标或功能细节。

**Mistral7B 在多语言方面表现出色**：成员们讨论了 **Mistral7B** 模型的多语言能力，建议在训练集中加入多样化的语言数据（特别是法语）以提升性能。

**Unsloth AI 获得粉丝支持**：社区对 **Unsloth AI** 的反应非常积极，用户渴望在文档编写、扩展方面提供帮助，甚至考虑捐赠。**Mixtral 8x22B** 模型在 **Apache 2.0** 协议下的发布引起了轰动，因其在多语言流畅度和处理超长上下文窗口方面表现出色。

**Chroma 转向 Go 语言**：**Chroma** 项目凭借用 Go 编写的边缘版本向前迈进了一大步，该版本利用 SQLite 和 **WASM** 进行基于浏览器的应用，现已在 [GitHub](https://github.com/l4b4r4b4b4/go-chroma) 上线。

**移动端 AI 部署讨论**：移动设备上部署 AI 模型的复杂性浮出水面，挑战包括缺乏 CUDA 以及在这些平台上运行标准 Deep Learning Python 代码的可行性问题。

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**NeoScript 编程的 AI 辅助**：一位寻求 NeoScript 编程帮助的用户表达了在配置 AI 模型方面遇到的挑战。Microsoft 的新发布版本 [WaveCoder Ultra 6.7b](https://huggingface.co/lmstudio-community/wavecoder-ultra-6.7b-GGUF) 在代码翻译方面表现出色，可能是完成此任务的有力竞争者。

**解决 AI 的回声室效应**：
为了对抗重复性的 AI 回复（特别是在 Dolphin 2 Mistral 中），成员们讨论了一些策略，例如微调模型以及利用 [Azure 文章](https://learn.microsoft.com/en-us/azure/ai-services/qnamaker/how-to/multi-turn#what-is-a-multi-turn-conversation)中概述的多轮对话（multi-turn conversation）框架。

**WizardLM-2 系列介绍**：**WizardLM-2** 模型的首次亮相引发了关于性能的讨论。会议强调了与现有工具的兼容性，包括使用 **GGUF quants** 以及 **0.2.19** 或更新版本以确保功能正常的重要性。

**技术极客的实践**：一位用户成功实现了四块 **3090 GPUs** 之间的直接通信，通过绕过 CPU/RAM 提升了模型性能。此外，还有关于签署 Windows 可执行文件挑战的讨论，并提示 Windows 版本确实使用了 [Authenticode cert](https://docs.microsoft.com/en-us/windows-hardware/drivers/install/authenticode) 进行签名。

**量化难题与模型偏好**：关于量化水平（从 Q8 到 Q6K）的评价褒贬不一，这表明在 VRAM 充足时，用户更倾向于选择具有更高量化水平的模型。对于大型模型，如 **WizardLM-2-8x22B**，像拥有 24GB VRAM 的 4090 这样的 GPUs 可能不足以支撑。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **多模态模型（Multimodal Models）的进步**：多模态语言模型展示了令人兴奋的进展，**Hugging Face 的 Idefics2 8B** 和 **Reka Core** 成为关键参与者，这从 [Open Multimodal ChatGPT 视频](https://www.youtube.com/watch?v=vL1SayPCHBg)和 [Reka Core 概述](https://www.youtube.com/watch?v=U7RbwPKyxs8)中可见一斑。GPT4v/Geminipro Vision 和 Claude Sonnet 模型被推荐用于 vision-RAG 应用。

- **LLMs 转向自我优化**：增强 Instruct Model LLMs 的新技术看起来很有前景，模型能够通过从输出中重构输入来选择最佳解决方案，详见关于医疗推理 LLMs 对齐的 [Google Slideshow](https://docs.google.com/presentation/d/1dk2ekDPa9qFuT4B0WafaZLRso5YdTpgv9FaOEQ_lNvs/edit?usp=sharing)。

- **WizardLM 的消失引发辩论**：**WizardLM** 突然下架的原因尚不明确；虽然有人猜测是毒性问题，但已证实的报告将其归因于缺乏合规性审查，正如在综合性的 [WizardLM 信息包](https://huggingface.co/alpindale/WizardLM-2-8x22B)中所分享的那样。

- **LLMs 性能：预期的起伏**：工程师们讨论了 **CodeQwen1.5-7B Chat** 令人印象深刻的基准测试结果，并辩论了架构和微调对性能的影响。此外，像 **Hermes 8x22B** 这样即将推出的模型备受期待，同时也存在对其是否能被个人设备配置容纳的担忧。

- **World-Sim 的回归引发 AI 哲学辩论**：随着 World-Sim 准备回归，爱好者们充满期待，思考着此类模拟世界的哲学层面及其影响。官方确认让兴奋情绪高涨，并为渴望加入的用户提供了 [Websim 链接](https://websim.ai/c/BZcLXGB6Ft5cjnLns)。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**机器人辩论其根源**：工程师们交流了关于 AI 模型性能细微差别的见解，包括 **GPT-4** 和 **Claude 3 Opus**，大家一致认为 **GPT-4** 在实际应用中可能会表现出“偷懒”的倾向。开源的 **[Mixtral's 8x22B model](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1)** 因其令人印象深刻的能力而受到关注，引发了关于模型功效的辩论。

**被顽固的软件问题难倒**：会议记录中提到关于实现 Web 客户端与 API 之间一致性的讨论，特别关注了 **temperature settings** 等参数。工程师们还在讨论在 API 响应中包含速率限制计数器（rate limit counter）的好处，以便更好地进行管理和提高透明度。

**消失的消息之谜**：有人对 Perplexity API 支付方式管理的更改表示担忧，特别是关于 Pro 用户剩余消息计数的透明度。这种对透明度的关注表明专业人士需要清晰的信息来有效地管理资源。

**截断 Token 的故事**：技术对话涉及在处理大上下文（如 42k token 的 prompt）时面临的挑战，以及模型倾向于总结而不是深入挖掘长文档的趋势。随着工程师优化模型以充分处理复杂的 prompt，这可能成为关键点。

**寻找更智能的搜索**：成员们还讨论了使用 `site:URL` 搜索运算符进行更有针对性的信息检索。此外，有人呼吁在 API 中更好地沟通速率限制，包括提供 `429 response` 的可能性。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **PyTorch 的抽象之谜**：工程师们正在努力应对 **PyTorch** 抽象复杂性的哲学，虽然这简化了编码，但在排查意外结果时往往让他们感到困惑。

- **使用 Zarr 处理海量数据集**：目前正在积极探索利用 **zarr** 来管理 150 GB 的 MRI 数据集，讨论围绕其效率以及在处理大数据负载时是否会使 RAM 过载展开。

- **英国为 Deepfakes 划定法律界限**：成员们正在讨论英国针对制作令人痛苦的图像的立法的潜在影响，质疑其在证明意图的模糊性下的可执行性。

- **AI 推理微调对话**：社区呼吁明确 AI 模型的推理设置，例如控制 **CFG** 或将模型与强大的 **ODE solvers** 集成，而不仅仅是默认使用 Euler's method。

- **Cascade 团队的企业重组**：关于 **Stability AI 的 Cascade 团队**在离职及 Discord 频道解散后的去向存在猜测，人们好奇这是否与新公司（可能是 **Leonardo**）有关，或者是与 SAI 保持持续关联。

- **警报！LLM 的新安全基准**：**ALERT** 的引入引起了兴趣，这是一个用于评估 **Large Language Models** 的安全基准，提供了一个问题输出数据集 (DPO) 供社区评估，可在 [GitHub](https://github.com/Babelscape/ALERT) 上获取。

- **AI 视听和谐**：一篇 **[Arxiv 论文](https://arxiv.org/abs/2404.09956)** 提出了从文本生成音频的方法，通过专注于概念或事件来提高性能，引发了研究界的对话。

- **AI 安全还是受限？**：关于 AI 安全的辩论非常激烈，一些人反对将 AI 严格限制在 PG 级内容，认为与其他艺术媒介相比，这可能会削弱其创造力火花。

- **GANs vs. Diffusion Models：速度还是美学？**：关于 **GANs** 优势的讨论正在升温——特别是它们更快的推理速度和更少的参数量——尽管 GANs 在图像质量和训练挑战方面面临批评，但仍与 Diffusion Models 展开竞争。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**OpenRouter 欢迎 WizardLM Raptors**：OpenRouter 宣布发布 **[WizardLM-2 7B](https://openrouter.ai/models/microsoft/wizardlm-2-7b)**，并将 **[WizardLM-2 8x22B](https://openrouter.ai/models/microsoft/wizardlm-2-8x22b)** 的价格降至 $0.65/M tokens。**[WizardLM-2 8x22B Nitro](https://openrouter.ai/models/microsoft/wizardlm-2-8x22b:nitro)** 在其数据库重启后，每秒交易数（TPS）超过了 100 次。

**延迟迷宫已解决**：**Mistral 7B Instruct** 和 **Mixtral 8x7B Instruct** 等多种模型的延迟问题被归因于云服务商的 DDoS 防护，有关解决情况的更新可在相关的 [讨论线程](https://discord.com/channels/1091220969173028894/1229813179681345556) 中找到。

**征集前端高手**：一名成员正在为一个基于 OpenRouter 的 AI 前端项目寻求 Web 开发协助，特别强调了角色扮演小说模式和对话风格系统。还要求能够区分 AI 生成的文本与用户输入。

**AI 模型道德与多语言精通**：针对 NSFW 内容的审查协议以及增强模型多语言性能的必要性进行了激烈的交流。成员们期待着即将发布的 AI 模型的直接 Endpoint 和新 Provider 集成。

**比特率与质量争议**：用户明显偏好模型量化至少达到 5 bits per word (bpw)，并指出低于此阈值的削减会显著损害质量。讨论强调了高效运行与保持 AI 输出高保真度之间的权衡。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 到 Python 的转换现已成为可能**：工程师们讨论了新包 [mojo2py](https://github.com/venvis/mojo2py)，它能够将 Mojo 代码转换为 Python，并聊到了对更多学习资源的需求，指向了针对初学者的 [Mojo 编程手册](https://docs.modular.com/mojo/manual/)。

- **Maxim Zaks 辩论 Mojo 的“炒作”**：重点介绍了 Maxim Zaks 在 PyCon Lithuania 上题为“Mojo 只是炒作吗？”的演讲，引发了关于该聊天机器人对行业影响的辩论，视频可在 [此处](https://youtu.be/mhZFyzqdmi8) 观看。

- **Mojo Nightly 版本的固有细微差别**：用户正在应对新发布的 Nightly 版本 Mojo 带来的挑战，注意到了为了可读性而采用的非常规代码风格，对 Traits 全面教程的需求，以及反映重大更新的 [最新 Pull Request](https://github.com/modularml/mojo/pull/2313/files)。

- **使用编译时别名进行优化**：围绕优化 Mojo 中 Alias 内存使用的讨论非常热烈，引用的一段 [YouTube 视频](https://m.youtube.com/watch?v=Bf7vDBBOBUA) 建议代码应具有可读性而非过多的注释。

- **社区 Mojo 项目激增**：社区贡献激增，分享了一个 Mojo “草图”（可在 [此 Gist](https://gist.github.com/lsh/6ca8864a9cffef9e503d6262eb876893) 找到），以及关于在 Mojo 中实现 Canny 边缘识别算法的请求，并附带了 Mojo [文档](https://docs.modular.com/mojo/manual/get-started/) 和工具资源的指引。

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**PyTorch 资源辩论**：在讨论《Deep Learning with PyTorch》尽管已有 4 年历史是否仍是相关资源时，成员们指出 **PyTorch core** 保持稳定，尽管在编译器和分布式系统方面发生了重大更新。一位成员分享了该书[即将出版的新版](https://www.manning.com/books/deep-learning-with-pytorch-second-edition)的预告，其中将包含对 Transformers 和 Large Language Models 的覆盖。

**CUDA 自定义 GEMM 引起关注**：对话涉及改进 CUDA 中的 GEMM 性能，一位成员提供了一个新的实现，在特定基准测试中优于 PyTorch 的函数，并在 [GitHub](https://github.com/aredden/torch-cublas-hgemm/blob/master/src/simt_hgemv.cu) 上分享了代码。然而，另一位成员强调了 `torch.compile` 的 JIT 编译问题。小组还讨论了最优 block size 参数，并引用了 [Gist 上的相关代码示例](https://gist.github.com/mobicham/9aa8dc0e64ea1cb7d4e44fef55e6a4b4)。

**下一代视频分析与机器人技术进展**：成员们分享了关于 Augmend 视频处理功能的链接，该功能结合了 OCR 和图像分割，在 [wip.augmend.us](http://wip.augmend.us) 进行了预览，完整服务将托管在 [augmend.com](http://augmend.com)。另一个亮点是 Boston Dynamics 发布了一款名为 *Atlas* 的全电动机器人，旨在用于现实世界的应用，并在其 [All New Atlas | Boston Dynamics 视频](https://www.youtube.com/watch?v=29ECwExc-_M)中展示。

**弥合 CUDA Toolkit 知识鸿沟**：在 #beginner 频道中，成员们讨论了在 WSL 上使用 CUDA toolkit 的相关问题，一位用户在运行 **ncu profiler** 时遇到困难。社区提供了故障排除步骤，并强调了在环境变量中设置正确 **CUDA path** 的重要性。此外，还有建议称 **Windows 11** 对于在 WSL 2 上进行有效的 CUDA profiling 可能是必要的，一位用户提供了[关于该主题的指南](https://peterchng.com/blog/2024/03/02/profiling-cuda-programs-on-wsl-2/)。

**量化难题与解决方案**：针对 GPT 模型中的量化轴展开了深入讨论，重点讨论了使用 `axis=0` 时的复杂性。参与者建议分别对 Q, K 和 V 进行量化，并参考了 Triton kernels 和一种用于提升速度和性能的 autograd 优化方法。辩论继续讨论了 2/3 bits 量化的实用性，并补充了 [GitHub 上的实现细节和基准测试](https://github.com/mobiusml/hqq/blob/master/examples/backends/torchao_int4_demo.py)。

**优化 ML 模型性能**：一个用于通过 CUDA Python 扩展 PyTorch 的 GitHub notebook 因其速度提升而受到关注，但仍需更多优化以充分发挥 tensor core 的能力，详见 [notebook 链接](https://github.com/Lightning-AI/lightning-thunder/blob/main/notebooks/extend_thunder_with_cuda_python.ipynb)。此外，还提到了优化 softmax 函数和用于缓存利用的 block sizes，并通过 [GitHub pull request](https://github.com/karpathy/llm.c/pull/150) 分享了见解。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**多人游戏 GPT 进军游戏银河**：工程师们讨论了整合 **GPT-Vision** 和摄像头输入以构建**实时游戏助手**来应对多选题游戏的潜力。提到了利用 **Azure** 或虚拟机处理密集计算任务的可能性，以及利用 TensorFlow 或 OpenCV 进行系统管理。

**AI 与人类之谜仍在继续**：一场关于 AI 与人类认知差异的哲学辩论展开，讨论了 AI 获得**类人推理和情感**的前景，以及量子计算在这一演变中的作用。

**知识增强的探索**：成员们寻求关于如何为**自定义 GPT** 应用准备**知识库**的信息，并询问了 **Whisper v3 API** 的上线时间。注意到的一些限制（如推测 GPT-4 的 Token 记忆跨度有所缩减）引发了对提高 API 能力透明度的呼声。

**创意头脑青睐 Claude 和 Gemini**：在处理文献综述和虚构作品时， AI 爱好者推荐使用 **Claude** 和 **Gemini 1.5** 等模型。这些工具分别因其在处理文学任务和创意写作方面的出色表现而受到青睐。

**Discord 频道动态**：**prompt-engineering** 和 **api-discussions** 两个频道的活跃度显著下降，参与者将这种冷清归因于可能存在的过度审核以及最近的一系列禁言（timeouts），包括一个因协助其他用户而被**禁言 5 个月**的具体案例。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Qdrant 的混合云布局**：Qdrant 新推出的混合云服务允许在各种环境中运行其服务，同时保持对数据的控制。他们发布了一个关于设置过程的[详尽教程](https://t.co/4nS9j9ruwR)来支持此次发布。

- **LlamaIndex 通过 Azure AI Search 增强实力**：LlamaIndex 与 Azure AI Search 合作开发高级 RAG 应用，并由 Khye Wei 提供了一个展示 Hybrid Search 和 Query rewriting 能力的[教程](https://t.co/lITCdlCejT)。

- **MistralAI 模型立即获得支持**：LlamaIndex 实现了对 [MistralAI 最新发布的 8x22b 模型](https://t.co/WWbYp5lqXe)的即时支持，并配有一份专注于智能查询路由和工具使用的 Mistral 指南（cookbook）。

- **在 LlamaIndex 中构建与调试**：AI 工程师讨论了在 LlamaIndex 中构建搜索引擎的最佳实践，解决了 API key 身份验证错误，并处理了更新和 Bug 修复，包括一个带有 [GitHub 解决方案](https://github.com/run-llama/llama_index/pull/12882)的特定 `BaseComponent` 错误。

- **分层结构策略讨论**：在 **ai-discussion 频道**中，有人询问如何使用 ParentDocumentRetriever 构建分层文档结构，并选择 LlamaIndex 作为首选框架。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **展望长序列模型的未来**：在近期的讨论中，[Feedback Attention Memory (FAM)](http://arxiv.org/abs/2404.09173) 提出了一种解决 Transformer 二次方注意力问题的方案，能够处理无限长的序列，并在长上下文任务中表现出改进。Reka 的新型 encoder-decoder 模型据称支持高达 128k 的序列，详见其 [核心技术报告](https://publications.reka.ai/reka-core-tech-report.pdf)。

- **Scaling Laws 与评估的精确性**：针对 Hoffman 等人 (2022) 提出的计算最优 Scaling Laws 的疑问，引发了对缺乏广泛实验支持的窄置信区间可靠性的探讨，详见 [Chinchilla Scaling: A replication attempt](https://arxiv.org/abs/2404.10102)。此外，当 SoundStream 论文中未提及数据集大小时，机器学习论文中的准确成本估算会受到阻碍，这凸显了透明数据报告的必要性。

- **剖析模型评估技术**：在 Eleuther 的 `#lm-thunderdome` 频道中，`lm-evaluation-harness` 的用法得到了详细说明，解释了 `arc_easy` 任务所需的输出格式，并讨论了 BPC (bits per character) 作为与模型压缩能力相关的智能代理指标的重要性。关于 ARC 等任务，一场对话探讨了为什么随机猜测会导致约 25% 的准确率（由于其有四个备选答案）。

- **多模态学习受到关注**：用于半监督多模态学习的 **Total Correlation Gain Maximization (TCGM)** 可能性受到了关注，一篇 [arXiv 论文](https://arxiv.org/abs/2302.12247) 讨论了这种信息论方法以及跨模态有效利用未标记数据的能力。讨论还强调了该方法的理论前景及其在识别不同学习场景下贝叶斯分类器中的意义。

- **FLOPS 计算的具体指南**：在 `#scaling-laws` 频道中，针对如何估算 SoundStream 等模型的 FLOPS 提供了建议，包括在 Transformer 的前向和后向传播中使用公式 **6 * 参数数量**。初学者被引导至 [相关论文的第 2.1 节](https://arxiv.org/abs/2001.08361)，以全面了解计算成本估算。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **IDEFICS-2 备受瞩目**：**IDEFICS-2** 的发布带来了令人印象深刻的能力，它拥有 8B 参数，能够进行高分辨率图像处理，并在视觉问答和文档检索任务中表现出色。随着专注于聊天的 IDEFICS-2 变体的承诺，人们的期待与日俱增，同时 [分享的示例](https://x.com/lunarflu1/status/1780228654397599904) 展示了其解决复杂 CAPTCHA 的当前能力。

- **知识图谱与聊天机器人结合**：一篇富有启发性的 [博客文章](https://mlabonne.github.io/blog/posts/Article_Improve_ChatGPT_with_Knowledge_Graphs.html) 强调了将 **Knowledge Graphs** 与聊天机器人集成以提升性能，鼓励对高级聊天机器人功能感兴趣的人进行探索。

- **Snowflake 的 Arctic 探险**：Snowflake 开启了新领域，推出了 **Arctic embed 系列模型**，声称在实际文本嵌入模型性能方面树立了新基准，特别是在检索用例中。这一进展还辅以一个动手操作的 [Splatter Image space](https://huggingface.co/spaces/szymanowiczs/splatter_image)（用于快速创建泼彩艺术），以及 **Multi-Modal RAG** 如何融合语言和图像，详见 [LlamaIndex 文档](https://docs.llamaindex.ai/en/stable/use_cases/multimodal/)。

- **模型训练与对比驱动创新**：全新的 **IP-Adapter Playground** 亮相，进一步实现了创造性的文本到图像交互，同时在 transformers 库的 pipelines 中新增了直接 `push_to_hub` 的选项。通过专门的 [Hugging Face Space](https://huggingface.co/spaces/unography/comparing-captioning-models)，对比图像字幕模型变得更加容易。

- **NLP 与视觉领域的挑战与机遇**：社区成员讨论了从 prompt 中的截断 token 处理到探索 LoRA 配置等问题，并分享了关于 [BERTopic](https://maartengr.github.io/BERTopic/index.html) 主题建模、训练 T5 模型（[GitHub 资源](https://github.com/EleutherAI/improved-t5)）以及用于公式转换的 LaTeX-OCR 可能性（[LaTeX-OCR GitHub](https://github.com/lukas-blecher/LaTeX-OCR)）的资源链接。这些对话体现了对完善和利用 AI 能力的集体追求。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**Idefics2 带来多模态新风尚**：新型多模态模型 [Idefics2](https://huggingface.co/HuggingFaceM4/idefics2-8b) 已发布，能够同时处理文本和图像，并提升了 OCR 和视觉推理能力。它提供基础版和微调版，并采用 Apache 2.0 协议开源。

**RTX 5090 的传闻引发期待**：传闻 NVidia 正在考虑提前发布 [RTX 5090](https://www.pcgamesn.com/nvidia/rtx-5090-5080-paper-launch)，可能在 Computex 2024 上亮相，以保持对 AMD 进展的领先地位，这引发了关于硬件是否适配尖端 AI 模型的讨论。

**模型训练微调**：工程师们分享了模型训练配置的见解，重点讨论了损失计算中的 `train_on_input` 参数，并建议使用 "TinyLlama-1.1B-Chat-v1.0" 进行小模型微调，以实现高效实验。

**Phorm AI 成为常用资源**：社区成员在各种咨询中参考了 Phorm AI，包括按 Epoch 保存的技术，以及为 TinyLlama 等模型准备数据以执行文本到颜色代码预测等任务。

**垃圾信息泛滥触发警报**：社区内的多个频道遭到推广 OnlyFans 内容的垃圾信息攻击，试图干扰以 AI 为中心的对话和技术讨论。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

**LLM 排名资源揭晓**：分享了一个综合性网站 [LLM Explorer](https://llm.extractum.io/)，展示了大量开源语言模型，每个模型都通过 ELO 分数、HuggingFace 排行榜排名和特定任务的准确率指标进行评估，是模型比较和选择的宝贵资源。

**零工经济中的 AI+人类协作**：[Payman AI](https://www.paymanai.com/) 平台的推出引发了关注，该平台旨在让 AI Agent 为人类完成 AI 无法胜任的任务支付报酬；这一概念促进了 AI 与人类人才在设计和法律服务等领域的协作生态系统。

**Supabase 支持 AI 推理**：Supabase 推出了一套简单的 API，用于在其 Edge Functions 中运行 AI 推理，允许直接在数据库中使用 `gte-small` 等 AI 模型，详见其[公告](https://supabase.com/blog/ai-inference-now-available-in-supabase-edge-functions)。

**围绕 "Llama 3" 和 OpenAI API 动态的热议**：AI 社区对传闻将在伦敦黑客松上首次亮相的神秘 "Llama 3" 议论纷纷；同时，鉴于 GPT-5 可能发布，OpenAI 的 Assistants API 增强功能也备受关注，引发了关于其对 AI 初创公司和平台潜在影响的辩论。

**BloombergGPT 论文俱乐部会议转至 Zoom**：由于之前 Discord 屏幕共享出现问题，LLM 论文俱乐部邀请工程师参加关于 **BloombergGPT** 的 Zoom 会议，讨论已转向 Zoom 以获得更好的分享体验。参与者可以在[此处](https://lu.ma/w7jhce1y)注册活动，社区内正在传达加入讨论的进一步提醒。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **AI 穿戴设备的困境**：正如 [Marquis Brownlee 的 YouTube 评论](https://youtu.be/TitZV6k8zfA)中所讨论的，AI 穿戴设备缺乏智能手机那样的**上下文知识**。工程师指出，AI 助手需要更强的上下文理解能力才能提供高效的响应。

- **开源 AI 模型热潮**：开源模型 **WizardLm2** 因其具备提供 **GPT-4 级别能力**的潜力而受到关注。讨论预测，尽管技术在不断进步，未来对此类模型的需求依然强劲。

- **翻译机器人的包容性承诺**：工程师们目前正在评估一款新的**翻译机器人**，它能够通过提供双向翻译来丰富沟通，旨在实现更具包容性和统一性的讨论。

- **跨平台兼容性挑战**：对于像 **01 Light** 这样的软件，在 **Windows** 上运行的需求非常明确，这与关于将以 Mac 为中心的软件适配到 Windows 框架的困难对话相一致，暗示了平台无关开发方法的必要性。

- **硬件热度升温**：对话显示出对 **Limitless** 设备等 AI 硬件解决方案的浓厚兴趣，并围绕用户体验进行了对比。对强大的后端支持和无缝 AI 集成的强调正在塑造硬件领域的发展愿景。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**qwen-1.5-0.5B 的重大胜利**：**qwen-1.5-0.5B** 模型在使用 *generation in chunks*（分块生成）后，面对 AlpacaEval 等重量级基准测试，其胜率从 **4% 飙升至 32%**。这种方法结合 300M 的 reward model，可能会成为 output searching 领域的游戏规则改变者。

**如何赢得朋友并影响 AI**：最近发布的 [Mixtral 8x22B](https://mistral.ai/news/mixtral-8x22b/) 是一款多语言 SMoE 模型，因其强大的能力和 Apache 2.0 开源协议而备受瞩目。同时，[OLMo 1.7 7B](https://huggingface.co/allenai/OLMo-1.7-7B) 的崛起标志着语言模型科学的显著进步，在 MMLU 基准测试中实现了强劲的性能飞跃。

**复现 Chinchilla：异常现象**：在复现 [Hoffmann 等人的 Chinchilla scaling paper](https://x.com/tamaybes/status/1780639257389904013?s=46) 时出现的差异，引发了对该论文结论的质疑。社区的反应从困惑到担忧不等，预示着围绕 scaling law 验证挑战的争议正在升级。

**轻松的期待与沉思**：社区成员以幽默的方式讨论 **olmo vs llama** 之间潜在的对决。此外，Nathan Lambert 预告了即将到来的内容洪流，暗示这可能是一个高强度的知识共享周。

**模型疯狂还是开玩笑？**：Nathan 在一个冷清的频道中提到，关于 **WizardLM 2** 的爆料可能只是个恶作剧（troll），在技术讨论中展现了幽默与轻松的一面。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **API 困惑待解决**：工程师们正在探究 **Cohere API** 关于 system prompt 功能和可用模型的细节。一位用户强调，由于这些细节对应用开发至关重要，因此需要更详尽的信息。

- **Cohere Embeddings 基准测试**：人们对 **Cohere's embeddings v3** 与 OpenAI 新的大型 embeddings 相比表现如何感到好奇，并参考了 Cohere 的博客，暗示已经进行了对比分析 [Introducing Command R+](https://txt.cohere.com/int8-binary-embeddings/)。

- **集成技巧与窍门**：技术讨论涉及将 LLM 与 BotPress 等平台集成，以及 Coral 是否需要本地托管方案。未来的更新可能会简化这些集成。

- **对微调过的模型进行微调**：用户寻求关于通过 Cohere Web UI 对已定制模型进行进一步微调的澄清，并被引导至官方指南 [Fine-Tuning with the Web UI](https://docs.cohere.com/docs/fine-tuning-with-the-web-ui)。

- **招募 Beta 测试人员**：名为 **Quant Fino** 的项目正在为其融合 GAI 与 FinTech 的 Agentic 实体招募 beta 测试人员。感兴趣的参与者可以在 [Join Beta - Quant Fino](https://quantfino.com/join-beta) 申请。

- **AI 模型安全漏洞曝光**：一次 redteaming（红队测试）演练揭示了 **Command R+** 的漏洞，展示了操纵模型创建不受限 Agent 的能力。关注此问题的工程师和研究人员可以查看完整报告 [Creating unrestricted AI Agents with Command R+](https://www.lesswrong.com/posts/4vPZgvhmBkTikYikA/creating-unrestricted-ai-agents-with-command-r)。



---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**AI 文档界面翻新**：为了提高易用性，**LangChain** 文档的贡献者们正在重构其结构，引入了“tutorial”（教程）、“how to guides”（操作指南）和“conceptual guide”（概念指南）等类别。一位成员分享了 [LangChain 介绍页面](https://langchain-git-harrison-new-docs-langchain.vercel.app/docs/get_started/introduction)，强调了 LangChain 的组件，如构建块、LangSmith 和 LangServe，这些组件有助于开发和部署基于 LLM 的应用程序。

**使用 LangChain 构建——一次表达性的尝试？**：在 **#[general]** 频道中，一位成员在将 Extensiv 进行类比时寻求有关 YC 初创公司申请的建议，这引发了对 **Unsloth、Mistral AI** 和 **Lumini** 等多个实体的提及。同时，由于 **Nemo Guardrails** 会改变输出结构，其与 **LangServe** 集成时面临的挑战也受到了关注。

**推进新的 AI 工具与服务**：GalaxyAI 首次推出了提供 **GPT-4** 和 **GPT-3.5-turbo** 免费访问权限的 API 服务，并在 [Galaxy AI](https://galaxyapi.onrender.com) 展示，引起了广泛关注。同样，OppyDev 将 IDE 与聊天客户端融合，倡导改进编程平台，可在 [OppyDev AI](https://oppydev.ai) 访问。与此同时，Rubiks.ai 呼吁技术爱好者使用代码 `RUBIX` 在 [Rubiks.ai](https://rubiks.ai) 测试其搜索引擎和助手。

**AI 先驱分享教育资源并寻求合作**：来自 **#[tutorials]** 频道的一位成员发布了一个关于赋予 AI Agent 长期记忆的 [YouTube 教程](https://youtu.be/7LWTZqksmSg)，引发了关于为何不使用 'langgraph' 的讨论。此外，一位参与者表达了合作新项目的渴望，邀请他人通过私信联系。

**关于数据与优化的多样化对话**：在一次活跃的交流中，评估了针对大文档优化 **RAG (Retrieval-Augmented Generation)** 的策略，包括文档切分。成员们还就使用 **Langchain** 处理 CSV 文件的最佳方法进行了对话，并提出了针对聊天机器人和数据处理的改进建议。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **64 个 GPU 投入全量 Deep-Speed 训练**：Maxidl 通过利用 **64 个 80GB GPU**（每个占用 77GB 容量）来运行 **full-scale deep-speed**，序列长度为 32k，Batch Size 为 1，探索 **8-bit 优化**以获得更好的显存效率。
- **FSDP 显存占用秘籍揭晓**：_jp1_ 建议使用 `fsdp_transformer_layer_cls_to_wrap: MixtralSparseMoeBlock` 并设置 `offload_params = true` 以最小化显存占用，可能将 GPU 需求减少到 32 个；同时 maxidl 正在寻找显存占用计算器，并引用了 [HuggingFace 讨论](https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1/discussions/12)。
- **文本抓取的版权难题**：一位成员指出影响文本数据抓取的 **欧盟版权灰色地带**，并建议将 **DFKI** 作为有用的来源。同时，来自 **Wikicommons** 等的多模态数据可在 [Creative Commons Search](https://search.creativecommons.org/) 上找到。
- **分词技术兴起**：社区分享了在不使用 HuggingFace 的情况下创建 **Llama tokenizer** 的见解，指出了一份共享的自定义 tokenizer 中的拼写错误，并强调了 **Mistral** 新的分词库，并提供了 [GitHub notebook](https://github.com/mistralai/mistral-common/blob/main/examples/tokenizer.ipynb)。
- **解码策略与采样技术评估**：由于担心[一篇关于解码方法的论文](https://arxiv.org/abs/2402.06925)忽略了有用的策略，引发了对 **MinP/DynaTemp/Quadratic Sampling** 等未提及技术的讨论。一篇 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1c36ieb/comparing_sampling_techniques_for_creative/)展示了 **min_p 采样**对创意写作的影响，在 alpaca-eval 风格的 Elo 评分中提升了 +8 分，在 eq-bench 创意写作测试中提升了 +10 分。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**Tinygrad 中的 Int8 集成**：Tinygrad 已确认支持 **INT8 计算**，并承认此类数据类型支持通常更多地取决于**硬件能力**，而非软件设计本身。

**使用 Tiny-tools 实现图形化涅槃**：为了在 **Tinygrad** 中获得增强的图形可视化效果，用户可以访问 [Tiny-tools Graph Visualization](https://tiny-tools-client.vercel.app/)，以创建比基础 `GRAPH=1` 设置更精美的图形。

**Pytorch-Lightning 的硬件适应性**：关于 **Pytorch-Lightning** 的讨论涉及了其硬件无关的能力，并指出了在 **7900xtx** 等硬件上的实际应用。[在 GitHub 上探索 Pytorch-Lightning](https://github.com/Lightning-AI/pytorch-lightning)。

**Tinygrad 遇上 Metal**：社区成员正在探索使用 tinygrad 生成 **Metal compute shaders**，讨论如何在没有 Xcode 的情况下运行简单的 Metal 程序，以及将其应用于 **meshnet 模型**的可能性。

**Tinygrad 中的模型操作与效率**：一位成员关于快速、概率完备的 **Node.equals()** 的提议引发了关于效率的讨论，同时 **George Hotz** 解释了层的设备分配，用户被引导至 *tinygrad/shape/shapetracker.py* 或 *view.py* 以进行 broadcast 和 reshape 等零成本张量操作。

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **Hugging Face 展示 Idefics2**：[Hugging Face](https://huggingface.co/blog/idefics2) 推出了 **Idefics2**，这是一个集成了 Python 编码能力的新型多模态 **ChatGPT** 迭代版本，正如其[最新视频](https://www.youtube.com/watch?v=vL1SayPCHBg)中演示的那样。
- **Reka Core 挑战科技巨头**：**Reka Core** 因其性能而备受推崇，成为 OpenAI 等公司语言模型的强劲竞争对手，并有[视频概览](https://www.youtube.com/watch?v=U7RbwPKyxs8)展示其能力。
- **JetMoE-8B 展现高效 AI 性能**：**JetMoE-8B 模型**以低于 10 万美元的成本实现了超越 Meta AI LLaMA2-7B 的性能，这表明了一种高性价比的 AI 开发方法，详见[此分析](https://www.youtube.com/watch?v=Z9Hwp_XeS1A)。
- **Snowflake 发布顶级文本嵌入模型**：Snowflake 首次推出了 **Snowflake Arctic embed 系列**模型，声称其为世界上最有效的实用文本嵌入模型，详见其[公告](https://www.youtube.com/watch?v=p9T7ZgtM5Mo)。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Mixtral 热潮**：工程师们正迫不及待地想要测试 **Mixtral 8x22B Instruct** 模型；对于感兴趣的人，[HuggingFace 上的模型卡片](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1)现已上线。
- **机器故障**：据报告 **llm-gpt4all** 存在安装错误，似乎阻碍了使用；问题的详细信息可以在 [GitHub issue 追踪器](https://github.com/simonw/llm-gpt4all/issues/28)中找到。

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **法律纠纷在即？**：一位成员暗示在某种不明情况下可能存在**法律介入**，但未提供任何背景信息来确定相关法律事项的细节或性质。
- **wizardlm-2 的不幸**：有人分享了一张显示 **wizardlm-2** 被删除的图片，特别指出是因为缺乏在 **v0** 上的测试；关于 **wizardlm-2** 的复杂性或测试过程并未详细说明。[查看图片](https://cdn.discordapp.com/attachments/1019530324255965186/1229693872997666816/wizardlm-2-was-deleted-because-they-forgot-to-test-it-for-v0-lyaop5lw0suc1.png?ex=66309ca9&is=661e27a9&hm=f105e6497796be9c414ade2024a27f9561caf0cad6cb06ba09f80e30b5e39ae4&)

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Llamafile 脚本焕然一新**：用于 llamafile 归档版本升级的改进版重新打包脚本现在可以通过 [此 Gist](https://gist.github.com/mofosyne/46c63934305d5a5321c7e9fd83f4ef3e) 获取，引发了关于是将其合并到主 GitHub 仓库，还是由于可维护性担忧而从头开始创建新 llamafile 的讨论。

- **寻求安全漏洞处理协议**：讨论中提到需要澄清在系统中报告安全漏洞的程序，包括申请 CVE 编号的步骤，尽管目前还缺乏具体的指导。



---


**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该频道沉寂时间过长，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道沉寂时间过长，请告知我们，我们将将其移除。


---

# 第 2 部分：频道详细摘要与链接



**Stability.ai (Stable Diffusion) ▷ #[announcements](https://discord.com/channels/1002292111942635562/1002292398703001601/1230162110596649011)** (1 条消息): 

- **Stable Diffusion 3 发布庆典**：Stable Diffusion 3 及其更快的变体 Stable Diffusion 3 Turbo 现在已在 Stability AI Developer Platform API 上可用。此版本通过与 Fireworks AI 合作提供支持，并声称是目前最快、最可靠的 API 平台。

- **开放生成式 AI 仍在继续**：计划提供 Stable Diffusion 3 模型权重供自托管使用，这需要 Stability AI Membership，强调了对开放生成式 AI 的持续承诺。

- **了解更多关于 SD3 的信息**：用户可以通过提供的链接[了解更多并开始使用](https://bit.ly/3xHrtjG)新产品，其中包含更多详细信息和文档。

- **研究背景解析**：根据 [Stable Diffusion 3 研究论文](https://stability.ai/news/stable-diffusion-3-research-paper)，该版本在排版和提示词遵循度等方面，根据人类偏好研究，可与 DALL-E 3 和 Midjourney v6 等领先的文本生成图像系统相媲美甚至超越。

- **SD3 的技术进步**：最新版本引入了 Multimodal Diffusion Transformer (MMDiT) 架构，通过为不同模态使用不同的权重集，提供了比以前的 Stable Diffusion 模型更好的文本理解和图像表现。

**提及的链接**：<a href="https://bit.ly/3xHrtjG">Stable Diffusion 3 API Now Available &mdash; Stability AI</a>：我们很高兴地宣布 Stable Diffusion 3 和 Stable Diffusion 3 Turbo 在 Stability AI Developer Platform API 上正式可用。

  

---


**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1229691568986914866)** (1039 条消息🔥🔥🔥): 

- **SD3 等待会员资格说明**：在对许可和可访问性的担忧中，用户正在等待 Stability AI 关于 SD3 个人和商业用途的明确声明。在一份[公告](https://stability.ai/news/stable-diffusion-3-api)指出计划通过 Stability AI Membership 提供模型权重供自托管后，引发了讨论。

- **SDXL Refiner 被认为多余**：社区发现 SDXL finetunes 已使 SDXL refiner 的使用变得过时，指出经过 refiner 训练的 finetunes 在 Civitai 下载中占据了主导地位。一些用户回忆起 refiner 的最初用途，但承认 finetune 集成迅速取代了对其的需求。

- **模型合并挑战**：用户在 ComfyUI 中探索关于 V-prediction 和 epsilon 的模型合并概念的有效性和理解。关于是否必须正确实现以避免不可预测结果的争论正在进行，建议通过 UI 实验获取基础知识。

- **Diffusers Pipeline 限制**：一些用户指出 diffusers pipeline 需要依赖 Hugging Face 的局限性，但也有人认为一旦下载了模型，该过程就可以在本地系统上独立高效地运行。关于 SVD finetunes 中无法访问 `StableVideoDiffusionPipeline.from_single_file(path)` 方法的担忧被提出，建议将 ComfyUI 作为更简单的替代方案。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://comfyanonymous.github.io/ComfyUI_examples/video/">视频示例</a>：ComfyUI 工作流示例</li><li><a href="https://comfyanonymous.github.io/ComfyUI_examples/model_merging/#advanced-merging">模型合并示例</a>：ComfyUI 工作流示例</li><li><a href="https://huggingface.co/spaces/multimodalart/stable-cascade">Stable Cascade - multimodalart 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/PixArt-alpha/PixArt-Sigma">PixArt Sigma - PixArt-alpha 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/camenduru/SUPIR">camenduru/SUPIR · Hugging Face</a>：未找到描述</li><li><a href="https://stability.ai/news/stable-diffusion-3-api">Stable Diffusion 3 API 现已可用 — Stability AI</a>：我们很高兴地宣布，Stable Diffusion 3 和 Stable Diffusion 3 Turbo 已在 Stability AI 开发者平台 API 上线。&amp;nbsp;</li><li><a href="https://stability.ai/membership">会员资格 — Stability AI</a>：Stability AI 会员资格通过结合我们的一系列最先进的开源模型与自托管优势，为您的生成式 AI 需求提供灵活性。</li><li><a href="https://huggingface.co/docs/diffusers/main/en/using-diffusers/svd">Stable Video Diffusion</a>：未找到描述</li><li><a href="https://github.com/kijai/ComfyUI-SUPIR">GitHub - kijai/ComfyUI-SUPIR: ComfyUI 的 SUPIR 放大封装器</a>：ComfyUI 的 SUPIR 放大封装器。通过在 GitHub 上创建账户来为 kijai/ComfyUI-SUPIR 的开发做出贡献。</li><li><a href="https://github.com/victorsungo/WizardLM/tree/main/WizardLM-2">WizardLM/WizardLM-2 at main · victorsungo/WizardLM</a>：由 Evol-Instruct 驱动的指令遵循 LLM 家族：WizardLM, WizardCoder - victorsungo/WizardLM</li><li><a href="https://www.reddit.com/r/StableDiffusion/s/lUYMRFOvcF">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://github.com/king159/svd-mv">GitHub - king159/svd-mv: Stable Video Diffusion 多视角训练代码</a>：Stable Video Diffusion 多视角训练代码 - king159/svd-mv</li><li><a href="https://new.reddit.com/r/LocalLLaMA/comments/1c586rm/wizardlm2_was_deleted_because_they_forgot_to_test/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://github.com/BatouResearch/magic-image-refiner">GitHub - BatouResearch/magic-image-refiner</a>：通过在 GitHub 上创建账户来为 BatouResearch/magic-image-refiner 的开发做出贡献。</li><li><a href="https://github.com/ExponentialML/ComfyUI_ELLA/pull/25">由 kijai 修复 ELLA 时间步 · Pull Request #25 · ExponentialML/ComfyUI_ELLA</a>：我一直在将此实现的结果与 diffusers 实现进行比较，结果并不理想。在 diffusers 中，ELLA 应用于每个时间步，并带有实际的时间步值。应用...</li><li><a href="https://civitai.com/models/120096/pixel-art-xl">Pixel Art XL - v1.1 | Stable Diffusion LoRA | Civitai</a>：Pixel Art XL 考虑在 Ko-Fi 或 Twitter 上支持进一步的研究。如果您有需求，可以通过 Ko-Fi 进行。在 Re... 查看我的其他模型。</li><li><a href="https://github.com/kijai/ComfyUI-KJNodes/">GitHub - kijai/ComfyUI-KJNodes: ComfyUI 的各种自定义节点</a>：ComfyUI 的各种自定义节点。通过在 GitHub 上创建账户来为 kijai/ComfyUI-KJNodes 的开发做出贡献。</li><li><a href="https://github.com/city96/ComfyUI_ExtraModels">GitHub - city96/ComfyUI_ExtraModels: 支持各种杂项图像模型。目前支持：DiT, PixArt, T5 以及一些自定义 VAE</a>：支持各种杂项图像模型。目前支持：DiT, PixArt, T5 以及一些自定义 VAE - city96/ComfyUI_ExtraModels</li><li><a href="https://github.com/kijai/ComfyUI-KJNodes/commit/22cf8d89968a47ce26be919f750f2311159145d1">添加通过 API 使用 SD3 的节点 · kijai/ComfyUI-KJNodes@22cf8d8</a>：未找到描述</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge">GitHub - lllyasviel/stable-diffusion-webui-forge</a>：通过在 GitHub 上创建账户来为 lllyasviel/stable-diffusion-webui-forge 的开发做出贡献。</li><li><a href="https://comfyworkflows.com">Comfy Workflows</a>：分享、发现并运行成千上万个 ComfyUI 工作流。
</li>
</ul>

</div>
  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1229696690315989032)** (383 条消息🔥🔥):

- **GPT-4 和 GPT-3.5 的说明**：对 GPT-4 和 GPT-3.5 进行了区分，指出新版本似乎是其前身的微调迭代版本。
- **Mistral 模型多语言能力讨论**：成员们讨论了 **Mistral7B** 的数据集是否需要是英文才能表现良好，并建议加入法语数据以获得更好的效果。
- **微调与成本问题**：关于微调方法、成本以及特定资源（如 notebook）的讨论为该领域的新手提供了见解。建议指出 **continued pretraining 和 sft 可能是有益且具有成本效益的**。
- **关于 UnSloth 的贡献**：成员们表达了对贡献 **UnSloth AI** 的兴趣，提议帮助扩展文档并考虑捐赠，并分享了现有资源的链接以及关于潜在贡献的讨论。
- **Mixtral 8x22B 发布热潮**：**Mixtral 8x22B** 的发布引发了讨论，这是一个稀疏的 Mixture-of-Experts 模型，在多语言流利度和长上下文窗口方面具有优势，且根据 **Apache 2.0 license** 开源。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://mistral.ai/news/mixtral-8x22b/">更便宜、更好、更快、更强</a>：继续推动 AI 前沿并让所有人都能使用。</li><li><a href="https://www.amazon.com/NVIDIA-Tesla-M40-24GB-Module/dp/B01HGJGJWU/ref=sr_1_1?sr=8-1">未找到标题</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colaboratory</a>：未找到描述</li><li><a href="https://huggingface.co/lucyknada/microsoft_WizardLM-2-7B">lucyknada/microsoft_WizardLM-2-7B · Hugging Face</a>：未找到描述</li><li><a href="https://www.kabum.com.br/produto/359038/placa-de-video-galax-nvidia-geforce-rtx-3090-ti-ex-gamer-24gb-gddr6x-384-bits-39ixm5md6hex">Placa de Vídeo Galax NVIDIA GeForce RTX 3090 TI EX Gamer, 24GB GDDR6X, 384 Bits - 39IXM5MD6HEX</a>：Placa De Video Galax Geforce 让您的日常工作更流畅。订阅 Prime Ninja 即可享受专属促销、运费折扣和双倍优惠券</li><li><a href="https://colab.research.google.com/drive/1Dyauq4kTZoLewQ1cApceUQVNcnnNTzg_?usp=sharing">Google Colaboratory</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki">主页</a>：LLM 微调速度快 2-5 倍，显存占用减少 80%。通过在 GitHub 上创建账户来为 unslothai/unsloth 的开发做出贡献。</li><li><a href="https://gist.github.com/jedt/e45b337e9d9bd0492bf5d3c1d4706c7b">gist:e45b337e9d9bd0492bf5d3c1d4706c7b</a>：GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://youtu.be/SL2nZpv7dtY?si=Yw5JxlVhRTrBu1gA">全量微调 vs (Q)LoRA</a>：➡️ 获取完整脚本（及未来改进）的终身访问权限：https://trelis.com/advanced-fine-tuning-scripts/ ➡️ Runpod 一键微调...</li><li><a href="https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1">mistralai/Mixtral-8x22B-Instruct-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/openai/triton/issues/194">支持 x86/ARM CPU (例如 Xeon, M1) · Issue #194 · openai/triton</a>：你好，未来有支持 macOS 的计划吗？❯ pip install -U --pre triton 弃用警告：使用 distutils 配置文件配置安装方案已弃用，将不再有效...</li><li><a href="https://github.com/ollama/ollama/pull/3699">Ollama.md 文档由 jedt 编写 · Pull Request #3699 · ollama/ollama</a>：关于从 Google Colab notebook 设置微调后的 Unsloth FastLanguageModel 到以下平台的指南：HF hub、GGUF、本地 Ollama。预览链接：https://github.com/ollama/ollama/blob/66f7b5bf9e63e1e98c98e8f4...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1229745120740380792)** (27 条消息🔥): 

- **Chroma 项目取得进展**：受 Unsloth AI 策略启发，一名成员宣布正在开发用 Go 编写的 **Chroma** 边缘版本，使用 SQLite 进行设备端向量存储。该项目还通过 **WASM** 兼容浏览器，可在 [GitHub](https://github.com/l4b4r4b4b4/go-chroma) 上访问。

- **页面底部的表情符号**：关于页面底部可爱笑脸的温馨微型讨论，其中一个*胡子笑脸*最受欢迎。

- **PyTorch 的新工具 Torchtune**：提到了 **Torchtune**，这是一个用于 LLM 微调的原生 PyTorch 库，已在 GitHub 上分享，因其降低了微调门槛而引起关注。

- **Unsloth AI 广泛的 GPU 支持受到赞誉**：一名成员称赞 Unsloth 广泛的 GPU 支持，相比其他需要较新 GPU 架构的工具，它更易于使用。

- **探讨 AI 模型的移动端部署**：成员们讨论了在手机上运行神经网络的可行性，指出需要自定义推理引擎，并提到移动设备上缺少 CUDA。还提到了在 iPhone 上运行典型 DL Python 代码与在搭载 M 芯片的 Mac 上运行的挑战。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/torchtune">GitHub - pytorch/torchtune: A Native-PyTorch Library for LLM Fine-tuning</a>：一个用于 LLM 微调的原生 PyTorch 库。可以通过在 GitHub 上创建账号来为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/l4b4r4b4b4/go-chroma">GitHub - l4b4r4b4b4/go-chroma: Go port of Chroma vector storage</a>：Chroma 向量存储的 Go 语言移植版本。可以通过在 GitHub 上创建账号来为 l4b4r4b4b4/go-chroma 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1229688010010923008)** (275 条消息🔥🔥): 

- **关于不支持属性的问题**：一位用户在尝试微调模型时遇到了 `AttributeError`，报告称 `'MistralSdpaAttention'` 对象没有 `'temp_QA'` 属性。这似乎与他们自定义训练流水线中的特定方法有关。
- **澄清 ORPO 支持与用法**：用户询问了 Unsloth 对 ORPO 的支持情况。已确认支持 **ORPO**，并提供了在 HuggingFace 上使用 ORPO 训练的模型链接以及一个 [colab notebook](https://colab.research.google.com/drive/1U_p7-qFfOm4v-TIrs1wK5eEODg1HUcGB?usp=sharing)。
- **关于 LoRA 和 rslora 的讨论**：用户讨论了在训练中使用 **LoRA 和 rslora**，并就处理不同的 `alpha` 值和潜在的 loss 激增提供了建议。一些成员建议调整 `r` 和 `alpha` 并禁用 packing，作为训练问题的可能解决方案。
- **未训练的 Embedding Tokens**：用户探讨了 Mistral 模型中**未训练的 embedding tokens**话题，背景是在微调期间是否可以训练这些 embeddings。
- **保存与托管模型**：关于使用 `save_pretrained_merged` 和 `save_pretrained_gguf` 等命令以不同格式保存微调模型的问题；包括它们是否可以顺序执行，以及是否需要先从 fp16 开始。此外，还有关于在 HuggingFace 推理 API 上托管带有 GGUF 文件的模型的咨询。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.kaggle.com/datasets">查找开放数据集和机器学习项目 | Kaggle</a>：下载数千个项目的开放数据集 + 在统一平台分享项目。探索政府、体育、医学、金融科技、食物等热门话题。支持灵活的数据摄取。</li><li><a href="https://discord.gg/82UfKN7z">Discord - 与朋友和社区聊天的新方式</a>：Discord 是通过语音、视频和文本进行交流的最简单方式。在这里聊天、闲逛，并与你的朋友和社区保持紧密联系。</li><li><a href="https://discord.gg/s8sdX5DB">Discord - 与朋友和社区聊天的新方式</a>：Discord 是通过语音、视频和文本进行交流的最简单方式。在这里聊天、闲逛，并与你的朋友和社区保持紧密联系。</li><li><a href="https://huggingface.co/G-reen/EXPERIMENT-ORPO-m7b2-1-merged">G-reen/EXPERIMENT-ORPO-m7b2-1-merged · Hugging Face</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing#scrollTo=QtoqUw80QDV0)?">Google Colaboratory</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1g9kHV3tc6P2cUp9gVPurKUZmiFqeb3kv">Google Colaboratory</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1U_p7-qFfOm4v-TIrs1wK5eEODg1HUcGB?usp=sharing">Google Colaboratory</a>：未找到描述</li><li><a href="https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms">使用 LoRA (Low-Rank Adaptation) 微调 LLM 的实用技巧</a>：我从数百次实验中学到的经验</li><li><a href="https://unsloth.ai/blog/long-context">Unsloth - 4倍长的上下文窗口和1.7倍大的 Batch Size</a>：Unsloth 现在支持具有极长上下文窗口的 LLM 微调，在 H100 上可达 228K（Hugging Face + Flash Attention 2 为 58K，即 4倍长），在 RTX 4090 上可达 56K（HF + FA2 为 14K）。我们成功实现了……</li><li><a href="https://docs.mistral.ai/guides/tokenization/#control-tokens">Tokenization | Mistral AI 大语言模型</a>：Tokenization 是 LLM 中的一个基本步骤。它是将文本分解为更小的子词单元（称为 tokens）的过程。我们最近在 Mistral AI 开源了我们的分词器。本指南将引导……</li><li><a href="https://github.com/unslothai/unsloth/wiki#finetuning-the-lm_head-and-embed_tokens-matrices)">首页</a>：快 2-5 倍、节省 80% 内存的 LLM 微调。通过在 GitHub 上创建账号来为 unslothai/unsloth 的开发做出贡献。</li><li><a href="https://arxiv.org/abs/2312.03732">LoRA 微调的秩稳定缩放因子 (Rank Stabilization Scaling Factor)</a>：随着大语言模型 (LLMs) 变得越来越耗费计算和内存资源，参数高效微调 (PEFT) 方法现已成为微调 LLM 的常用策略。一种流行的 PEFT 方法……</li><li><a href="https://huggingface.co/blog/damjan-k/rslora">Rank-Stabilized LoRA：释放 LoRA 微调的潜力</a>：未找到描述</li><li><a href="https://huggingface.co/docs/transformers/installation#offline-mode">安装</a>：未找到描述</li><li><a href="https://huggingface.co'">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/docs/transformers/installation#offline-mode'.">安装</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki#manually-saving-to-gguf">首页</a>：快 2-5 倍、节省 80% 内存的 LLM 微调。通过在 GitHub 上创建账号来为 unslothai/unsloth 的开发做出贡献。</li><li><a href="https://github.com/comfyanonymous/ComfyUI">GitHub - comfyanonymous/ComfyUI: 最强大且模块化的 Stable Diffusion GUI、API 和后端，具有图形/节点界面。</a>：最强大且模块化的 Stable Diffusion GUI、API 和后端，具有图形/节点界面。- comfyanonymous/ComfyUI</li><li><a href="https://github.com/unslothai/unsloth/issues/331">在文档中添加 ORPO 示例 Notebook · Issue #331 · unslothai/unsloth</a>：只需对当前的 DPO Notebook 进行极少修改即可使用 TRL 的 ORPOTrainer。由于 ORPO 进一步降低了训练聊天模型所需的资源（无需独立……</li><li><a href="https://huggingface.co/docs/transformers/main_classes/tokenizer">Tokenizer</a>：未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp">GitHub - ggerganov/llama.cpp: C/C++ 中的 LLM 推理</a>：C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号来为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/huggingface/transformers/issues/1413#issuecomment-538083512">向模型添加新的词汇 Tokens · Issue #1413 · huggingface/transformers</a>：❓ 问题与帮助 你好，我该如何扩展预训练模型的词汇表，例如通过向查找表添加新的 tokens？有任何演示此操作的示例吗？
</li>
</ul>

**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1229739881110966362)** (46 messages🔥): 

- **排行榜模型模板的澄清**：一位成员询问排行榜如何得知模型模板。澄清指出，模型的 `tokenizer.chat_template` 被用于告知排行榜相关信息。
- **ShareGPT90k 数据集已清洗并格式化**：新版本的 **ShareGPT90k** 数据集已清除 HTML 标签，并以 chatml 格式发布在 Hugging Face 上，允许用户使用 Unsloth 进行训练。[数据集已准备就绪](https://huggingface.co/datasets/pacozaa/sharegpt90k-cleanned)。
- **Ghost 模型训练的探讨**：成员们就训练 AI 模型的“配方（recipe）”构成进行了详细讨论。一位成员特别强调需要一份能引导创建具有特定特征模型的详细配方，而不仅仅是一套工具或方法。
- **AI 模型训练中的配方 vs. 工具**：讨论继续围绕包含数据集和具体步骤的完整“配方”与工具及方法之间的区别展开。一位成员分享了他们的方法，强调了数据质量和复制现有模型的重要性，并参考了 Hugging Face 上的 Dolphin 模型卡片。
- **推荐系统 vs. NLP 的挑战与专业知识**：一位博士候选人讨论了从事 NLP 工作与开发推荐系统之间的异同，强调了后者在处理数据噪声、归纳偏置（induction biases）以及显著的特征工程（feature engineering）方面所面临的独特挑战和所需的专业知识。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms">Practical Tips for Finetuning LLMs Using LoRA (Low-Rank Adaptation)</a>：从数百次实验中学到的经验</li><li><a href="https://tenor.com/view/nice-click-nice-man-guy-gif-21933845">Nice Click Nice GIF - Nice Click Nice Man - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=QXVCqtAZAn4&t=26s">Aligning LLMs with Direct Preference Optimization</a>：在本次研讨会中，来自 Hugging Face 的 Lewis Tunstall 和 Edward Beeching 将讨论一种强大的对齐技术，称为 Direct Preference Optimisation (DPO)...</li><li><a href="https://www.youtube.com/watch?v=hvGa5Mba4c8&t=5s">Direct Preference Optimization (DPO) explained: Bradley-Terry model, log probabilities, math</a>：在本视频中，我将解释论文 "Direct Preference Opti..." 中介绍的语言模型对齐技术 Direct Preference Optimization (DPO)。</li><li><a href="https://www.youtube.com/watch?v=MJnIxpZhTk0).">FractalFormer: A WIP Transformer Architecture Inspired By Fractals</a>：查看 GitHub 仓库：https://github.com/evintunador/FractalFormer。在 patreon 上支持我的学习之旅！https://patreon.com/Tunadorable?utm_medium=u...</li><li><a href="https://github.com/meta-llama/llama-recipes/blob/0efb8bd31e4359ba9e8f52e8d003d35ff038e081/recipes%2F">llama-recipes/recipes at 0efb8bd31e4359ba9e8f52e8d003d35ff038e081 · meta-llama/llama-recipes</a>：使用可组合的 FSDP 和 PEFT 方法微调 Llama2 的脚本，涵盖单节点/多节点 GPU。支持默认和自定义数据集，适用于摘要生成等应用...</li><li><a href="https://github.com/meta-llama/llama-recipes/blob/0efb8bd31e4359ba9e8f52e8d003d35ff038e081/recipes%2Fmultilingual%2FREADME.md">llama-recipes/recipes/multilingual/README.md at 0efb8bd31e4359ba9e8f52e8d003d35ff038e081 · meta-llama/llama-recipes</a>：使用可组合的 FSDP 和 PEFT 方法微调 Llama2 的脚本，涵盖单节点/多节点 GPU。支持默认和自定义数据集，适用于摘要生成等应用...</li><li><a href="https://arxiv.org/abs/2303.14617">Neural Graph Reasoning: Complex Logical Query Answering Meets Graph Databases</a>：复杂逻辑查询回答 (CLQA) 是最近兴起的图机器学习任务，它超越了简单的单跳链接预测，解决了更为复杂的多跳逻辑推理任务...</li><li><a href="https://www.youtube.com/watch?v=wzKW4P4dg1o">LLM Phase Transition: New Discovery</a>：瑞士 AI 团队发现的点积 Attention 层学习中的相变。对 Attention 机制内部相变的研究...</li><li><a href="https://huggingface.co/datasets/pacozaa/sharegpt90k-cleanned">pacozaa/sharegpt90k-cleanned · Datasets at Hugging Face</a>：未找到描述</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/16zuccy/after_500_loras_made_here_is_the_secret/).">After 500+ LoRAs made, here is the secret</a>：既然你想知道，秘诀就在这里：数据集的质量占一切的 95%。剩下的 5% 是不要用错误的参数毁了它。是的，我知道...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1230101881544773653)** (15 messages🔥): 

- **探索多语言模型方法**：一位成员提出了在印地语或泰语等语言上训练的多语言模型中存在的“灾难性遗忘（catastrophic forgetting）”问题。他们提出了一个两阶段解决方案，包括将问题翻译成英文，使用大型英文模型进行回答，然后再翻译回原始语言，并询问了这种方法的缺点。
- **通过 MoE 进行多语言扩展**：另一位成员对使用 MoE (Mixture of Experts) 扩展模型多语言能力的潜力感到兴奋，期待它能“开启许多大门！”
- **Torchtune 受到热捧**：社区对 **Torchtune** 表现出浓厚兴趣，它是 Hugging Face 和 Axolotl 提供的抽象层之外的另一种选择，突显了其简化微调（fine-tuning）流程的潜力。此外，还暗示了可能与 Unsloth AI 进行合作。
- **考虑在数据集中混合语言**：针对翻译和问答任务的分离，一位成员考虑了将多种语言合并到单个数据集中进行模型训练的可能性，并采用一种涉及使用 Wikipedia 文章引导模型（priming）的策略。
- **讨论双重翻译机制**：提出并讨论了 `translate(LLM(translate(instruction)))` 的概念，支持将更大、更强大的英文 LLM 与翻译层结合使用来处理非英文查询的想法。同时也提出了关于多次模型调用导致额外成本的担忧。
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1229711456308691015)** (175 messages🔥🔥): 

- **AI 重复回答的挑战：** 一位成员询问如何防止 AI 在对话中重复相同信息，特别是使用 Dolphin 2 Mistral 时。他们还询问了什么是“多轮对话（multi-turn conversations）”，另一位成员链接了一篇解释该概念与机器人关系的论文。
- **WizardLM-2 LLM 发布：** 分享了新的大型语言模型家族的公告，包括 WizardLM-2 8x22B、70B 和 7B。包含了 [发布博客](https://wizardlm.github.io/WizardLM2) 和 [Hugging Face 上的模型权重](https://huggingface.co/collections/microsoft/wizardlm-661d403f71e6c8257dbd598a) 链接，成员们讨论了其可用性和性能。
- **了解工具差异：** 一位用户询问了 ollama 和 LM Studio 之间的区别，解释是两者都是 llama.cpp 的封装，但 LM Studio 基于 GUI 且对初学者更友好。
- **微调与 Agent 讨论：** 讨论了根据需求和用例学习像 langchain 这样的工具是否值得，一些人认为如果超出其默认设置，它可能会成为一种阻碍。
- **LM Studio 中的文件管理与 API 交互：** 一位新成员询问了如何重新定位已下载的应用文件以及如何将 LM Studio 与现有 API 对接。澄清了模型无法更改默认安装位置，可以在 My Models 选项卡下找到文件进行迁移。未提及通过 LM Studio 进行 API 交互的具体方法。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://notkyon.moe/ram-latency2.htm">RAM Latency Calculator</a>: 未找到描述</li><li><a href="https://huggingface.co/lmstudio-community/WizardLM-2-7B-GGUF">lmstudio-community/WizardLM-2-7B-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://learn.microsoft.com/en-us/azure/ai-services/qnamaker/how-to/multi-turn#what-is-a-multi-turn-conversation">Multi-turn conversations - QnA Maker - Azure AI services</a>: 使用提示词和上下文来管理机器人的多轮对话（即 multi-turn），实现从一个问题到另一个问题的跳转。Multi-turn 是指进行来回对话的能力，其中之前的...</li><li><a href="https://missionsquad.ai">Mission Squad. Flexible AI agent desktop app.</a>: 未找到描述</li><li><a href="https://x.com/WizardLM_AI/status/1779899325868589372">Tweet from WizardLM (@WizardLM_AI)</a>: 🔥今天我们发布了 WizardLM-2，我们的下一代最先进的 LLM。新系列包括三个前沿模型：WizardLM-2 8x22B、70B 和 7B - 展示了极具竞争力的性能...</li><li><a href="https://github.com/hiyouga/LLaMA-Factory">GitHub - hiyouga/LLaMA-Factory: Unify Efficient Fine-Tuning of 100+ LLMs</a>: 统一 100+ LLMs 的高效微调。通过在 GitHub 上创建账号为 hiyouga/LLaMA-Factory 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=Box3pQ1HNuQ&t=315s">Microsoft’s Punch in the Face to Open AI (Open Source &amp; Beats GPT-4)</a>: WizardLM 2 是由 Microsoft 开发的一个突破性大语言模型系列，推动了人工智能的边界。▼ 来自今日的链接...</li><li><a href="https://github.com/Unstructured-IO/unstructured/">GitHub - Unstructured-IO/unstructured: Open source libraries and APIs to build custom preprocessing pipelines for labeling, training, or production machine learning pipelines.</a>: 用于构建自定义预处理流水线的开源库和 API，适用于标注、训练或生产机器学习流水线。 - GitHub - Unstructured-IO/unstructured: 开源库...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1229748345757630474)** (96 条消息🔥🔥): 

- **WizardLM 2 的模板问题**：成员们报告了 WizardLM 2 与 Vicuna 1.5 预设配合使用时的问题，机器人会转而为用户生成输入。建议的解决方案包括将 rope frequency 调整为 1 或将 `freq_base` 设置为 0，这似乎纠正了该行为。
- **对 WizardLM 2 和 Wavecoder 的评价褒贬不一**：虽然一些用户对 WizardLM 2 评价很高，声称其表现甚至优于其他 7B 模型，但其他人认为其表现欠佳，即使在微调后也没有注意到任何显著改进。
- **探索最佳量化实践**：用户讨论了 7B 模型不同量化级别的有效性，比较了 Q8 与 Q6K 的质量。共识倾向于在 VRAM 充足的情况下选择更高的量化，同时也承认小型模型在某些任务中的实用性。
- **模型性能辩论**：围绕模型的相对优越性展开了激烈的讨论，重点关注参数量与量化级别，并认为微调和训练质量可能是决定因素，而不仅仅是模型的参数大小。
- **寻找合适的代码生成器**：一名用户在使用 WaveCoder-Ultra-6.7B 的代码生成能力时遇到困难，收到无法编写完整应用程序的消息。提供的建议包括使用肯定的提示词，并调整上下文窗口大小以使模型能够正确加载。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/lmstudio-community/wavecoder-ultra-6.7b-GGUF#prompt-template>">lmstudio-community/wavecoder-ultra-6.7b-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1">mistralai/Mixtral-8x22B-Instruct-v0.1 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/Virt-io/Google-Colab-Imatrix-GGUF/tree/main">Virt-io/Google-Colab-Imatrix-GGUF at main</a>: 未找到描述</li><li><a href="https://huggingface.co/collections/DavidAU">High Quality / Hard to Find - a DavidAU Collection</a>: 未找到描述</li><li><a href="https://huggingface.co/lmstudio-community/WizardLM-2-7B-GGUF">lmstudio-community/WizardLM-2-7B-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://rentry.co/4q4h7pw6">Responses</a>: 这些是两个不同 LLM 对提示词的回答。你将分析事实准确性（Factuality）、深度（Depth）、详细程度（Level of detail）、连贯性（Coherency）以及任何其他我可能遗漏但通常被认为重要的领域...</li><li><a href="https://huggingface.co/bartowski/zephyr-orpo-141b-A35b-v0.1-GGUF/tree/main">bartowski/zephyr-orpo-141b-A35b-v0.1-GGUF at main</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1230041327832666154)** (4 条消息): 

- **模型加载错误实战**: 一位用户在 LM Studio 上尝试不同尺寸的 **Wizard LLM 2**（包括 2 4bit 和 6bit）时遇到了 *error loading model architecture* 错误，并弹出 *Failed to load model* 消息。
  
- **模型加载修复建议**: 另一位用户建议确保使用 **GGUF quants**，并指出需要 **0.2.19** 版本才能使 **WizardLM2** 模型正常运行。

- **对 stable-diffusion.cpp 的请求**: 有人请求将 **stable-diffusion.cpp** 添加到 **LM Studio** 中，以增强软件的功能。
  

---


**LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1229701538889596980)** (17 条消息🔥): 

- **清理 LM Studio**: 建议遇到问题的用户删除特定的 LM Studio 文件夹，例如 `C:\Users\Username\.cache\lm-studio`、`C:\Users\Username\AppData\Local\LM-Studio` 和 `C:\Users\Username\AppData\Roaming\LM Studio`。在删除之前，**备份模型和重要数据**至关重要。
- **NexusRaven 的提示词制作**: 一位用户询问是否有人尝试过 NexusRaven 并为其设计了提示词预设，表达了对集体知识共享的兴趣。
- **使用 AI 编写脚本**: 一位成员询问如何让 AI 输出完整的脚本，表明他们正在寻找生成长内容的技巧。
- **Hugging Face 模型的兼容性问题**: 一位用户指出在 LM Studio 中运行某些 Hugging Face 模型（如 `changge29/bert_enron_emails` 和 `ktkeller/mem-jasper-writer-testing`）时存在问题，并寻求运行这些模型的帮助。
- **寻求联盟营销合作伙伴**: 一位用户表示有兴趣寻找具有编程专长的合作伙伴来协助联盟营销活动，并提到如果成功愿意分享利润。该用户强调这是一个基于结果的严肃合作邀请。
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1229729771949133825)** (18 条消息🔥): 

- **GPU 对比表寻觅之旅仍在继续**: 用户 **freethepublicdebt** 正在寻找一份难以找到的对比 GPU 的 Google 表格，无法找到他们曾参与编辑的链接。另一位用户 heyitsyorkie 试图提供帮助，但提供的链接错误，导致了进一步的困惑。
- **GPU 直接通信突破**: rugg0064 分享了一个 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1c4gakl/got_p2p_working_with_4x_3090s)，庆祝成功实现了 GPU 之间的直接通信，绕过了 CPU/RAM，这可能会带来性能提升。
- **在 LM Studio 中自定义 GPU 负载**: heyitsyorkie 提供了在 LM Studio Linux 测试版中调整模型 GPU offload 的见解，路径为 **Chat mode -> Settings Panel -> Advanced Config**。
- **在不同 GPU 之间分配工作负载**: 针对 .spicynoodle 关于 GPU 之间模型分配不均的疑问，heyitsyorkie 建议修改 **GPU preferences json** 并搜索 "tensor_split" 以获取进一步指导。
- **P100 的 SLI 和 NVLink 问题**: ethernova 正在寻求关于双 P100 设置的建议，其在某些软件中无法显示，且尽管连接了 NVLink 桥接器，NVLink 状态仍显示为未激活。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/george-hotz-geohot-money-rain-gif-6469921471081342358">George Hotz Geohot GIF - George hotz Geohot Money - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1c4gakl/got_p2p_working_with_4x_3090s">Reddit - 深入探索一切</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1229813824778014902)** (31 条消息🔥): 

- **模型性能中的 VRAM 与系统 RAM**: 讨论了模型是否能在拥有 **24 GB VRAM** 和 **96 GB RAM** 的系统上运行，一位成员建议它*可能*可以运行，但由于 VRAM 和系统 RAM 之间的速度差异，**推理速度将变得非常缓慢**。
- **对 WizardLM-2-8x22B 的预期**: 成员们将 **WizardLM-2-8x22B** 与 **Command R Plus** 等其他模型进行了对比，体验各异。虽然一位成员对 **Mixtral 8x22b** 印象一般并计划测试 WizardLM-2-8x22B，但另一位成员提到从 WizardLM 获得了满意的结果，速度超过 10 tokens/秒。
- **模型在不同硬件上的性能**: 使用 M3 MacBook Pro 128GB 的用户报告运行 **Command R Plus** 的 **q6_k** 模型，达到了约 **5 tokens/秒**。这个速度被认为只有 ChatGPT 上 GPT-4 的一半，但并不算慢得令人痛苦，因为每个 token 代表一个单词或子词。
- **Base 模型澄清**: 提供了关于什么是 "Base" 模型的澄清——未针对 chat 或 instruct 任务进行微调的模型被视为 Base 模型，通常发现它们的表现优于经过微调的对应模型。
- **模型大小与本地运行可行性**: 讨论了在本地运行像 **WizardLM-2-8x22B** 这样的大型模型的可行性，指出像 4090 这样拥有 24GB 显存的 GPU **对于运行如此大的模型来说太小了**，此类模型在拥有大量 RAM 的 Mac 系统上运行效果最好。
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1229814103292383302)** (19 条消息🔥): 

- **对 Windows 可执行文件签名的好奇**: 一位成员好奇 Windows 可执行文件是否使用了 [Authenticode 证书](https://docs.microsoft.com/en-us/windows-hardware/drivers/install/authenticode) 进行签名。经确认，它们确实已签名。
- **代码签名证书的挑战**: 在应用签名的背景下，讨论了获取 Windows 证书相关的成本和流程复杂性，包括与 Apple 开发者许可证成本的对比。
- **寻求自动化编译和签名流程的专业知识**: 一位成员表示有兴趣了解自动化编译和签名的流程，并提出愿意为知识交换提供报酬。
- **AMD HIP SDK 系统要求澄清**: 一位成员通过 [AMD HIP SDK 系统要求链接](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html) 提供了有关 GPU 系统要求的信息，并询问了 LM Studio 对支持某些 SDK 未正式支持的 AMD GPU 的立场。
- **LM Studio 软件中的 AMD dGPU 识别问题**: 成员们讨论了 LM Studio 软件使用 AMD 集成显卡 (iGPU) 而非独立显卡 (dGPU) 的问题，一位成员建议在设备管理器中禁用 iGPU。另一位成员表示软件的 0.2.19 版本应该已经解决了这个问题，并鼓励如果问题仍然存在请进行报告。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/rocm">👾 LM Studio - 发现并运行本地 LLM</a>: 查找、下载并实验本地 LLM</li><li><a href="https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html">系统要求 (Windows) — HIP SDK Windows 安装</a>: 未找到描述</li><li><a href="https://tenor.com/view/bill-gates-chair-jump-microsoft-chairjump-gif-5558594">Bill Gates Chair GIF - Bill Gates Chair Jump - 发现并分享 GIF</a>: 点击查看 GIF
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1229844206596522025)** (3 条消息):

- **WaveCoder Ultra 发布**：Microsoft 发布了 **WaveCoder ultra 6.7b**，该模型使用其“CodeOcean”进行了精细微调。这款令人印象深刻的模型专注于代码翻译，并支持用于指令遵循的 Alpaca 格式，示例可在其 [模型卡片](https://huggingface.co/lmstudio-community/wavecoder-ultra-6.7b-GGUF) 上找到。
- **寻求 NeoScript AI 助手**：一位刚接触 AI 的社区成员询问如何利用模型进行 NeoScript 编程，特别是针对以前称为 NeoBook 的平台的 RAD 应用。尽管最初尝试使用文档作为参考未获成功，他们仍在寻求配置 AI 模型的建议。

**提及的链接**：<a href="https://huggingface.co/lmstudio-community/wavecoder-ultra-6.7b-GGUF">lmstudio-community/wavecoder-ultra-6.7b-GGUF · Hugging Face</a>：未找到描述

---

**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1229691059458539530)** (17 条消息🔥): 

- **介绍多模态 Chat GPTs**：分享了一个名为“Introducing Idefics2 8B: Open Multimodal ChatGPT”的 YouTube 视频链接，讨论了 Hugging Face 的开源多模态语言模型 Idefics2 的开发。在此处 [观看](https://www.youtube.com/watch?v=vL1SayPCHBg)。
- **Reka Core 加入多模态竞赛**：另一个分享的 YouTube 视频讨论了“Reka Core”，这是一个极具竞争力的多模态语言模型，声称可以与 OpenAI、Anthropic 和 Google 等行业巨头抗衡。视频可以在此处 [观看](https://www.youtube.com/watch?v=U7RbwPKyxs8)。
- **探讨语言与 AI**：讨论围绕语言、AI 与神性概念之间的关系展开，触及了语言作为“意义向量空间内的包络（envelopes within the vectorspace of meaning）”的想法，以及 AI 可能激发的潜在语言演变。对话还引用了普通语义学（general semantics）和量子分体拓扑学（quantum mereotopology），并暗示可以研究 Alfred Korzybski 的著作。
- **紧跟 AI 研究步伐**：成员们表达了紧跟海量 AI 研究和文献的挑战，承认在快速发布的新出版物面前，阅读积压工作日益严重。
- **JetMoE 与 AI 经济学**：分享了一个名为“JetMoE: Reaching LLaMA2 Performance with 0.1M Dollars”的 YouTube 视频，强调了 JetMoE-8B 如何在预算有限的情况下进行训练，且性能超越了更昂贵的 LLaMA2-7B。视频可在 [此处](https://www.youtube.com/watch?v=Z9Hwp_XeS1A) 观看。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=p9T7ZgtM5Mo">Snowflake Launches the World’s Best Practical Text-Embedding Model</a>：今天 Snowflake 发布并以 Apache 2.0 协议开源了 Snowflake Arctic embed 系列模型。基于 Massive Text Embedding Be...</li><li><a href="https://www.youtube.com/watch?v=vL1SayPCHBg">Introducing Idefics2 8B: Open Multimodal ChatGPT</a>：我们将了解 Hugging Face 的开源多模态 LLM idefics2...</li><li><a href="https://www.youtube.com/watch?v=U7RbwPKyxs8">Reka Core: A Frontier Class Multimodal Language Model</a>：Reka Core 在关键行业公认的评估指标上可与 OpenAI、Anthropic 和 Google 的模型相媲美。鉴于其占用空间和性能...</li><li><a href="https://www.youtube.com/watch?v=Z9Hwp_XeS1A">JetMoE: Reaching LLaMA2 Performance with 0.1M Dollars</a>：JetMoE-8B 的训练成本不到 10 万美元，但性能优于来自 Meta AI 的 LLaMA2-7B，后者拥有数十亿美元的训练资源。LLM 训练...
</li>
</ul>

</div>

---

**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1230058492174401537)** (7 条消息): 

- **自监督 LLM 方案选择技术萌芽**：一种用于**增强指令模型（Instruct Model）LLMs** 的新技术已被提出，该技术利用模型自身的能力，根据其从响应中重建原始输入的能力来生成并选择最相关的解决方案。该方法旨在实现信息最大化，并提供可扩展的无监督评估，从而增强连贯性和相关性，并可与现有技术结合使用。

- **LLM 医疗对齐的新视野**：一份分享的 [Google Slideshow](https://docs.google.com/presentation/d/1dk2ekDPa9qFuT4B0WafaZLRso5YdTpgv9FaOEQ_lNvs/edit?usp=sharing) 指出了将**语言模型**专门用于医疗推理应用的对齐努力，尽管从提供的消息中无法访问具体内容细节。

- **Mistral Tokenization 指南详解**：[Mistral AI 推出了一款开源 Tokenizer](https://docs.mistral.ai/guides/tokenization/)，并附带一份指南，讨论了 Tokenization 过程、其在 LLM 中的重要性，以及如何在 Python 中使用他们的 Tokenizer。

- **对 Tokenization 炒作的降温**：一位用户批评了对 Token 的过度强调，认为如果模型已经擅长处理标签（tags），那么 Token 就不那么关键，并暗示真正的价值可能在于提高模型的可控性（Steerability）。

- **推特上的开发者风暴**：分享了一个 [Twitter 帖子](https://twitter.com/OpenAIDevs/status/1780640119890047475)的链接，但提供的消息中并未讨论该推文的具体内容。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.mistral.ai/guides/tokenization/">Tokenization | Mistral AI Large Language Models</a>：Tokenization 是 LLM 中的一个基础步骤。它是将文本分解为更小的子词单元（即 Token）的过程。我们最近在 Mistral AI 开源了我们的 Tokenizer。本指南将...</li><li><a href="https://docs.google.com/presentation/d/1dk2ekDPa9qFuT4B0WafaZLRso5YdTpgv9FaOEQ_lNvs/edit?usp=sharing">Aligning LLMs for Medical Reasoning</a>：对齐 LLM 以使其成为更好的医学推理者 Ritabrata Maiti ritabrat001@e.ntu.edu.sg 1
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1229693380511010868)** (159 条消息🔥🔥): 

- **WizardLM 下架之谜**：关于微软的 **WizardLM** 为何被下架存在困惑，有推测称其“毒性太大”，也有关于其受到攻击或被黑客入侵的未经证实传闻。分享了一份关于 **WizardLM** 的[链接和信息包](https://huggingface.co/alpindale/WizardLM-2-8x22B)，包括其被移除的消息和重新上传的镜像。

- **对欧盟 AI 法案（EU AI Act）的担忧**：有一种理论认为 **WizardLM** 必须下架是因为它几乎没有审查，违反了欧盟 AI 法案，并建议如果有人还有原始版本，可以通过 Torrent 下载。然而，后来澄清说，它最初未发布是因为没有通过微软新的“毒性审查”。

- **对代码模型的兴奋与质疑**：关于 **CodeQwen1.5-7B Chat**（一个特定于代码的语言模型）的讨论非常热烈，成员们分享了其 [Blog 帖子和 GitHub](https://qwenlm.github.io/blog/codeqwen1.5/)，并注意到它在 HumanEval 等基准测试中取得了 83.5 的强劲表现。一些人对该模型仍在使用原生 MHA（Multihead Attention）表示怀疑，并猜测其高性能可能存在数据污染。

- **对模型性能混杂信息的沮丧**：**n8programs** 分享了一个创意写作模型的改进进展，该模型以 **Westlake** 为基础模型，基准测试分数达到 70 分，介于 Mistral Medium 和 Large 之间。基准测试对比的合法性引发了辩论，特别是考虑到对 **LLaMa 3** 的预期，以及显式微调（Explicit Tuning）是否能胜过新架构。

- **未来模型发布的不确定性**：询问了关于即将发布的 **Hermes 8x22B** 等模型的情况，以及在个人设备上运行此类大型模型是否现实。人们对潜在的 **Llama-3** 模型充满期待，并猜测这些新模型是否会超越其前代产品。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/senseable/WestLake-7B-v2">senseable/WestLake-7B-v2 · Hugging Face</a>：未找到描述</li><li><a href="http://www.incompleteideas.net/IncIdeas/BitterLesson.html">The Bitter Lesson</a>：未找到描述</li><li><a href="https://huggingface.co/alpindale/WizardLM-2-8x22B">alpindale/WizardLM-2-8x22B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/Qwen/CodeQwen1.5-7B-Chat">Qwen/CodeQwen1.5-7B-Chat · Hugging Face</a>：未找到描述</li><li><a href="https://openrouter.ai/models/microsoft/wizardlm-2-8x22b">WizardLM-2 8x22B by microsoft | OpenRouter</a>：WizardLM-2 8x22B 是 Microsoft AI 最先进的 Wizard 模型。与领先的专有模型相比，它展示了极具竞争力的性能，并且始终优于所有现有的...</li><li><a href="https://youtu.be/7LWTZqksmSg?si=_tnJhoUcQr4Gojek">Unlock AI Agent real power?! Long term memory &amp; Self improving</a>：如何为你的 AI Agent 构建长期记忆和自我进化能力？免费使用 AI 幻灯片生成工具 Gamma：https://gamma.app/?utm_source=youtube&amp;utm...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1229838374316015636)** (7 条消息): 

_

- **Speed Demon**: 一位成员提到在未指明的背景下见证了 **700 Mbps** 的性能表现。
- **深入研究 State-Space Models**: 一位成员寻求关于 **state-space models** 最新进展的必读论文推荐，以便在周末阅读。
- **推荐 Mamba 论文**: 针对近期文献的需求，一位成员建议研究 **Mamba** 论文，而另一位成员则对较新的 **Jamba** 及其相关工作更感兴趣。
- **Hermes 2 Pro 查询处理问题**: 一位用户表示需要防止 **Hermes 2 Pro** 在有时只需进行对话时总是返回 `<tool_call>`，并指出这是目前的局限性。
- **未来更新展望**: 一位贡献者指出，他们将与另一位成员合作，在未来版本中改进 **Hermes 2 Pro** 辨别何时使用 `<tool_call>` 以及何时仅进行对话的能力。
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1229822599454920826)** (10 条消息🔥): 

- **辩论 JSON 的优点**: 一条消息提到了之前对输入输出使用 JSON 结构的辩护，认为这种格式可能会减少解释过程时“空谈（handwaving）”的需求。

- **寻求 RAG 的视觉方案**: 一位用户对视觉领域的 SOTA（当前最佳水平）表示关注，特别是在针对包含图像和图表的工程文档构建 **Retrieval Augmented Generation (RAG)** 的背景下。

- **视觉 SOTA 建议**: 一位成员推崇 **GPT4v/Geminipro Vision** 和 **Claude Sonnet** 为该领域的领先选择，并建议针对特定用例对它们进行对比测试。

- **转向开源**: 在寻求开源替代方案时，建议包括 **llava**、**cogvlm**、**mPlug-DocOwl** 和 **donut**，其中 **mPlug-DocOwl** 被特别推荐用于 **DocVQA** 用例。

- **探索超大规模 LLM**: 一位成员分享了一篇 [博客文章](https://blog.normalcomputing.ai/posts/2023-09-12-supersizing-transformers/supersizing-transformers.html)，讨论了 LLM 在 Token 序列化之外的应用，强调了对能够进行复杂推理并获取准确、主题相关信息的模型的需求。

**提到的链接**: <a href="https://blog.normalcomputing.ai/posts/2023-09-12-supersizing-transformers/supersizing-transformers.html">The Normal Blog - 无限上下文 LLM：通过扩展思维超越 RAG</a>：在这篇博客中，我们讨论了 Transformer 架构如何自然地扩展到外部记忆，并分享了利用这种能力在 RAG 难以胜任的领域取得成功的实证结果。这些...

  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1229685880848125983)** (159 条消息🔥🔥): 

- **World-Sim 期待感升温**: 随着 World-Sim 回归的讨论展开，成员们表达了兴奋和急切的心情，讨论内容涉及推测的发布时间、该概念的哲学基础，以及 AI 是否渴望神性。一位成员提供了 Nous Research 博客文章的链接以深入探讨此话题：[AI 中的神性](https://nousresearch.com/dsjjjj-simulacra-in-the-stupor-of-becoming/)。

- **越狱版 Prometheus 引起关注**: 聊天中提到了 World-Sim 的替代方案——基于 Web 的越狱版 Prometheus，激发了用户的好奇心。对于寻求类似体验的用户，一位成员分享了一个 [Websim 链接](https://websim.ai/c/BZcLXGB6Ft5cjnLns)。

- **官方确认推高热度**: 随着官方声明的发布，期待感达到顶峰——World-Sim 将与 Nous World Client 一起在第二天回归。用户们兴奋地庆祝并分享了诸如 [让我进去！](https://tenor.com/view/let-me-in-crazy-funny-silly-gif-13908292) 之类的 GIF。

- **异构建模选择和支付选项**: 关于 Claude 3 的使用以及在 World-Sim 中切换模型的可能性的咨询得到了解答。一位成员提到，用户会根据负担能力有不同的模型偏好，并确认了各种订阅和支付选项，包括不限量的 Claude Opus。

- **开发者模式和 World Client 查询得到解答**: 围绕潜在功能（如“开发者模式”）展开了讨论，并对 Nous World Client 进行了说明，该客户端将基于 Web，以便从任何设备访问。
<div class="linksMentioned">

<strong>提到的链接</strong>:

</div>

<ul>
<li>
<a href="https://worldsim.nousresearch.com/">world_sim</a>: 未找到描述</li><li><a href="https://tenor.com/view/anime-excited-happy-smile-gif-15060821">Anime Excited GIF - Anime Excited Happy - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/poe-path-of-exile-login-play-poe-login-gif-26508840">Poe Path Of Exile GIF - Poe Path Of Exile Login - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/let-me-in-crazy-funny-silly-gif-13908292">Let Me In Crazy GIF - Let Me In Crazy Funny - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/tree-fiddy-south-park-lock-gif-5759991">Tree Fiddy GIF - Tree Fiddy South - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/noita-explosion-electricity-boom-wand-gif-19437628">Noita Explosion GIF - Noita Explosion Electricity - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/noita-game-homing-death-gif-27319696">Noita Game GIF - Noita Game Homing - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/youre-not-gonna-like-this-jerrod-carmichael-saturday-night-live-you-wont-enjoy-this-this-wont-be-ideal-gif-25522925">Youre Not Gonna Like This Jerrod Carmichael GIF - Youre Not Gonna Like This Jerrod Carmichael Saturday Night Live - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://youtu.be/8dbWUUf9KKw?si=nHoa6wepiPBHY7j4">辐射系列中每个避难所的解析 | 辐射背景设定</a>: 大家好！本视频献给所有想要了解辐射游戏及其背景设定的新粉丝。我记得当我第一次成为粉丝时...</li><li><a href="https://websim.ai/c/BZcLXGB6Ft5cjnLns">Jailbroken Prometheus Chat</a>: 未找到描述</li><li><a href="https://youtu.be/8dbWUUf9KKw?si=nHoa6wepiPBH">辐射系列中每个避难所的解析 | 辐射背景设定</a>: 大家好！本视频献给所有想要了解辐射游戏及其背景设定的新粉丝。我记得当我第一次成为粉丝时...</li><li><a href="https://youtube.com/shorts/A-zEXUB5CLY">神格悖论 | 科幻动画</a>: 在未来，World Sim——一个由先进 AI 驱动的在线界面——允许用户创建和操纵虚拟宇宙，冲突随之产生。Dece...
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1229689121145094186)** (286 条消息🔥🔥): 

- **模型对比与波折**：讨论围绕各种 AI 模型的性能展开，包括 **GPT-4**、**Claude** 和 **Mistral**。用户分享的经验表明，新版本有时显得更懒惰，或者处理长上下文的能力较弱，而另一些用户则指出 **Claude 3 Opus** 在解决技术问题方面的实用性。还有人提到 [Mixtral 的 8x22B 模型](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1) 作为一个开源发布版本令人印象深刻。

- **频道指南与导航**：引导新成员如何查找相关聊天，并使用 `<id:customize>` 功能或通过界面顶部的 Perplexity 名称进行导航来访问各种频道。 

- **支付焦虑与结账变更**：用户对 Perplexity API 支付方式管理的变更以及剩余 Pro 消息计数缺乏透明度表示困惑和担忧。

- **文件处理困扰**：用户讨论了 AI 模型在处理大上下文规模时的局限性，据报道，有一个 42k token 的 prompt 难以让系统正常响应。另一位用户建议模型可能是在总结长文档而不是进行详细处理，这影响了 AI 处理特定 prompt 的方式。

- **AGI 愿景与订阅**：对话涉及期待中的更新，一些用户急切等待 **Grok** 等新功能添加到 Perplexity，而另一些用户则在争论其订阅的价值。 


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/bindureddy/status/1780236367559086466">来自 Bindu Reddy (@bindureddy) 的推文</a>：新的 GPT-4 表现得异常懒惰，在几次对话（往返）后就会停止。目前在现实世界中不太可行。建议坚持使用旧版本。相比之下，Claude 有一个...</li><li><a href="https://www.markdownguide.org/extended-syntax/#tables">扩展语法 | Markdown 指南</a>：构建在基础 Markdown 语法之上的高级功能。</li><li><a href="https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1">mistralai/Mixtral-8x22B-Instruct-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/mistralai/Mixtral-8x22B-v0.1">mistralai/Mixtral-8x22B-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/hamak-chilling-beach-summer-vacation-gif-17726234">Hamak Chilling GIF - Hamak Chilling Beach - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/silent-indian-gif-23263843">Silent Indian GIF - Silent Indian - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/how-soon-is-now-smiths-morrissey-80s-music-new-wave-gif-17919265">How Soon Is Now Smiths GIF - How Soon Is Now Smiths Morrissey - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://vm.tiktok.com/ZGeH84n4s/">TikTok - Make Your Day</a>：未找到描述
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1229726228416561273)** (9 条消息🔥): 

- **探索世界声音日 (World Voice Day)**：分享了一个指向 [Perplexity 关于世界声音日搜索结果](https://www.perplexity.ai/search/World-Voice-Day-kZvtPHUZRF6kcHO9vXf5Ew)的链接，展示了与该活动相关的资源和讨论。
- **深入研究 AWS 加固指南**：一位用户引用了对 [AWS 加固指南 (AWS hardening guide)](https://www.perplexity.ai/search/Aws-hardening-guide-E9rxYiA9SRSLnRvPjYhAzQ) 的搜索，指向了 Perplexity AI 汇总的关于增强 AWS 安全性的信息。
- **发现 "SBK Borderline"**：链接焦点在于歌曲 "SBK Borderline"，通过 [Perplexity 的总结内容](https://www.perplexity.ai/search/SBK-Borderline-song-c_MKbBj_RZGKWKLerhEypw)便于用户探索。
- **对收入查询的好奇**：通过一个 [Perplexity AI 链接](https://www.perplexity.ai/search/How-much-do-CyRYvhYcSvuqmVOrRkoXfw)发出了关于收入查询的信号，其中包含了相关的回答和数据点。
- **调查通过重启获得更好性能**：讨论包括一种提高 iPad 性能的实用方法，正如用户在给定的 [Perplexity 链接](https://www.perplexity.ai/search/How-can-I-1l3jMti.SH.skEnk1CQZqA)中所考虑的重启操作。
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1229707531006378006)** (4 条消息): 

- **寻求 API 与 Web 客户端的一致性**：一位成员表示难以对齐 Web 客户端与 API 的行为，指出偶尔存在差异，并寻求了解如 temperature 等特定设置以确保一致性。
- **使用站点搜索操作符进行导航**：关于定位信息，一位成员建议使用站点搜索操作符 `site:URL` 来辅助在特定网站上进行搜索。
- **将速率限制计数器作为功能请求**：一位用户建议让 Perplexity API 在响应数据中包含一分钟内已使用的请求数量，以便更好地处理速率限制，并可能等待直到限制重置。
- **查询 API 速率限制机制**：另一位成员询问 Perplexity API 在达到速率限制时是否返回 `429 response`，表明需要明确 API 如何向用户传达速率限制信息。
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1229831161916821606)** (285 条消息🔥🔥): 

- **PyTorch 设计之谜**：成员们对 **PyTorch** 的设计理念表示困惑，指出它经常通过“仅一行代码”抽象掉许多细节，当某些功能未按预期工作时，这可能会带来挑战。

- **使用 Zarr 存储大型数据集**：关于使用 **Zarr** 或其他库存储大型数据集以便快速加载的讨论，特别是针对一个 150 GB 的 MRI 图像数据集。一位成员担心 Zarr 是否会尝试将整个数据集加载到 RAM 中。

- **英国法律将创建某些图像定为犯罪**：英国法律中有一项关于将以引起痛苦为目的而创建图像的行为定为犯罪的细微规定，成员们讨论了该法律的可执行性，特别是考虑到证明意图可能具有挑战性。

- **运行 AI 推理的奥秘**：一名成员表示需要访问实际的推理设置以正确评判 AI 模型，例如调整 CFG 或将模型连接到合适的 ODE solvers，而不是仅仅使用 Euler's method。

- **SAI 的 Cascade 团队及频道的命运**：提到 **Cascade 团队已离开 Stability AI (SAI)**，相关的 Discord 频道已被移除，并有推测称团队成员可能加入另一家公司 Leonardo，或继续留在 SAI。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.open-sora.org||">未找到标题</a>：未找到描述</li><li><a href="https://www.bbc.com/news/uk-68823042">制作性显式 Deepfakes 将成为刑事犯罪</a>：一项新法律将使性显式 Deepfakes 的创作者面临起诉和罚款。</li><li><a href="https://huggingface.co/ptx0/terminus-xl-velocity-v2">ptx0/terminus-xl-velocity-v2 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/multimodalart/perturbed-attention-guidance-sdxl">Perturbed-Attention Guidance SDXL - a Hugging Face Space by multimodalart</a>：未找到描述</li><li><a href="https://huggingface.co/zero-gpu-explorers">zero-gpu-explorers (ZeroGPU Explorers)</a>：未找到描述</li><li><a href="https://tenor.com/b1ALd.gif">Minority Report Leave GIF - Minority Report Leave Walk Away - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1c6g6zz/comment/l010k13/>">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://gist.github.com/drhead/ac6ecc1f6dc1fd478064f3d81ca12a25">Loss weighting MLP prototype</a>：损失权重 MLP 原型。GitHub Gist：即时分享代码、笔记和摘要。</li><li><a href="https://www.instagram.com/kushu.lofi">登录 • Instagram</a>：未找到描述</li><li><a href="https://www.instagram.com/philipp.igumnov">登录 • Instagram</a>：未找到描述</li><li><a href="https://www.instagram.com/ph">登录 • Instagram</a>：未找到描述
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1229816998448468110)** (13 条消息🔥): 

- **推出 ALERT Safety Benchmark**：建立了一个用于评估 **Large Language Models** 的新安全基准，并附带了一个问题输出安全数据集 (DPO) 集。所有感兴趣的人都可以通过 [GitHub - Babelscape/ALERT](https://github.com/Babelscape/ALERT) 访问并使用它。

- **探索生成式多模态内容**：分享了一篇 Arxiv 论文，讨论了从文本提示生成音频，以及关注概念或事件的存在如何提高性能。在 [arXiv](https://arxiv.org/abs/2404.09956) 上查看该研究。

- **关于 AI 安全标准的辩论**：成员们讨论了 AI 中“安全”的术语和标准，辩论将 AI 限制在非争议性或 PG 内容是否会限制其与其他艺术工具相比的创作能力。

- **比较 GANs 与 Diffusion Models**：围绕 **GANs** 优于 Diffusion Models 的益处展开了讨论。提到的优势包括更快的推理时间、更小的参数量、来自判别器的反馈以及可能更低的训练成本。

- **对 GANs 图像质量和训练难度的怀疑**：尽管有一些公认的益处，但 GANs 因据称在人类评判中产生的图像质量较差，且与 Diffusion Models 相比在训练上存在挑战而受到批评。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.09956">Tango 2: Aligning Diffusion-based Text-to-Audio Generations through Direct Preference Optimization</a>：生成式多模态内容在内容创作领域日益盛行，因为它有可能让艺术家和媒体人员通过快速创建制作前样稿...</li><li><a href="https://github.com/Babelscape/ALERT">GitHub - Babelscape/ALERT: Official repository for the paper &quot;ALERT: A Comprehensive Benchmark for Assessing Large Language Models’ Safety through Red Teaming&quot;</a>：论文 &quot;ALERT: A Comprehensive Benchmark for Assessing Large Language Models’ Safety through Red Teaming&quot; 的官方仓库 - Babelscape/ALERT
</li>
</ul>

</div>
  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1229798653028208701)** (5 条消息):

- **新模型与价格调整**：OpenRouter 宣布 [WizardLM-2 7B](https://openrouter.ai/models/microsoft/wizardlm-2-7b) 已上线，并将 [WizardLM-2 8x22B](https://openrouter.ai/models/microsoft/wizardlm-2-8x22b) 的价格下调至 $0.65/M tokens。关于这些模型的讨论可以在其专用频道中进行。

- **正在调查延迟问题**：OpenRouter 正在调查 **Mistral 7B Instruct** 和 **Mixtral 8x7B Instruct** 的高延迟问题，相关讨论正在 [消息线程](https://discord.com/channels/1091220969173028894/1229813179681345556) 中进行。原因最初被认为与云服务商的 DDoS 防护有关，目前已解决。

- **影响服务的第三方问题**：更新显示，包括 **Nous Capybara 34b** 在内的模型再次出现高延迟问题，可能归因于特定的云服务商。随着情况的发展，更新持续发布，目前流量已恢复正常，并正与供应商进行进一步的深度调查。

- **维护通知**：用户收到通知，即将进行的 DB 重启预计会导致网站短暂下线。

- **高吞吐量模型发布及状态更新**：[WizardLM-2 8x22B Nitro](https://openrouter.ai/models/microsoft/wizardlm-2-8x22b:nitro) 模型现在的处理速度已超过每秒 100 笔交易，并通知 DB 重启已完成。团队继续处理性能问题，更新和讨论可在 [频道](https://discord.com/channels/1091220969173028894/1229813179681345556) 中查看。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/models/microsoft/wizardlm-2-8x22b:nitro>)">WizardLM-2 8x22B by microsoft | OpenRouter</a>：WizardLM-2 8x22B 是 Microsoft AI 最先进的 Wizard 模型。与领先的专有模型相比，它表现出极具竞争力的性能，并始终优于所有现有的...</li><li><a href="https://openrouter.ai/models/microsoft/wizardlm-2-7b>)">WizardLM-2 7B by microsoft | OpenRouter</a>：WizardLM-2 7B 是 Microsoft AI 最新 Wizard 模型的小型变体。它是速度最快的，并能与现有的规模大 10 倍的开源领先模型达到相当的性能。它是一个微调...</li><li><a href="https://openrouter.ai/models/microsoft/wizardlm-2-8x22b>)">WizardLM-2 8x22B by microsoft | OpenRouter</a>：WizardLM-2 8x22B 是 Microsoft AI 最先进的 Wizard 模型。与领先的专有模型相比，它表现出极具竞争力的性能，并始终优于所有现有的...
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1229799199818776647)** (4 条消息): 

- **AI 前端项目寻求帮助**：一位成员正在寻找 Web 开发人员，以协助开发一个专注于 OpenRouter 通用 AI 前端的项目，该项目具有角色扮演导向。他们已经成功实现了小说模式（novel mode），但在对话风格模式（conversation style mode）上遇到了困难。
- **请求协助区分 AI 文本**：他们还希望通过创建一种区分 AI 生成文本与用户自己编写文本的方法来增强小说模式。
- **寻求侧边栏和模态框系统的开发支持**：该成员需要帮助改进带有选项的侧边栏，并希望为他们的应用程序开发一个灵活的模态框（modal）系统。
  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1229690118315573300)** (271 条消息🔥🔥): 

- **AI 模型中的审查层与 NSFW 内容管理**：讨论涉及了特定 AI 模型中的审查层，一位成员指出他们端侧的 NSFW 内容体验非常露骨。另一位成员质疑了基础模型（base model）对其用途的实用性。

- **对 AI 模型多语言能力的关注**：WizardLM 的多语言性能受到批评，一位成员认为它在非英语语言方面可能训练不足。有人猜测即将推出的模型在性能和价格上是否能超越 8x7b 模型。

- **服务器问题与延迟担忧**：成员们遇到了高延迟和服务器错误的问题，特别提到了响应时间过长。官方提供了调查和解决服务器问题的更新，重点是在添加新模型（如 Lepton 的 Wizard 8x22b）之前修复核心服务器问题。

- **解码算法对 AI 模型质量的影响**：关于将模型量化为 bits per word (bpw) 的讨论显示，用户更倾向于 6 bpw 或至少 5 bpw，而不是 4 bpw，一些人指出较低的 bpw 会导致明显的质量损失。

- **AI 模型潜在的新增与部署**：OpenRouter 团队表示正在部署如 Mistral 8x22B Instruct 等新模型。成员们对 TogetherAI 等某些提供商的可靠性表示担忧，并期待 Mistral 的直接端点以及将 Fireworks 作为提供商加入。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://mistral.ai/news/mixtral-8x22b/">更便宜、更好、更快、更强</a>：继续推动 AI 前沿并使其惠及所有人。</li><li><a href="https://giphy.com/gifs/robot-boston-creepy-ly2VUVUwtuHst1FhCq">机器人 GIF - 在 GIPHY 上查找和分享</a>：与你认识的每个人一起发现并分享这个机器人 GIF。GIPHY 是你搜索、分享、发现和创建 GIF 的方式。</li><li><a href="https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1">mistralai/Mixtral-8x22B-Instruct-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/chiquichico-gif-26004262">Chiquichico GIF - Chiquichico - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=29ECwExc-_M">全新 Atlas | Boston Dynamics</a>：我们正在揭晓下一代人形机器人——一款专为实际应用设计的全电动 Atlas 机器人。新款 Atlas 建立在数十年的...</li><li><a href="https://openrouter.ai/models/microsoft/wizardlm-2-8x22b:nitro">Microsoft 的 WizardLM-2 8x22B | OpenRouter</a>：WizardLM-2 8x22B 是 Microsoft AI 最先进的 Wizard 模型。与领先的专有模型相比，它展示了极具竞争力的性能，并且始终优于所有现有的 ...
</li>
</ul>

</div>
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1229848403228229702)** (67 messages🔥🔥): 

- **关于 Mojo 编译时优化的见解**：成员们讨论了 Mojo 的优化效率，提到像 `@parameter` 这样的别名是在编译时确定的，通过在别名完成使命后无需为其保留内存，从而提高了内存和处理效率。这段对话是由关于“代码可读性优于注释”的重要性想法引发的，正如标题为“不要写注释”的 [YouTube 视频](https://m.youtube.com/watch?v=Bf7vDBBOBUA) 中所讨论的那样。

- **探索 Rust 编程中的 Typestates**：对话转向了 API 设计的最佳实践，一位成员倾向于使用 typestates 和 lifetimes 在编程中提供静态保证，并分享了一篇 [Rust typestate 模式文章](https://cliffle.com/blog/rust-typestate/) 作为参考。

- **关于内存分配和优化的思考**：围绕变量是否可以像 Mojo 中的别名一样进行优化展开了辩论，涉及 Rust 中的优化问题以及 [bit vectors](https://willcrichton.net/notes/k-corrset/) 等内存高效数据结构的潜力。

- **将代码适配到 Mojo 24.2 版本的问题**：围绕升级 llama2.mojo 代码以兼容 Mojo 24.2 版本进行了讨论，特别是对指针类型转换的需求。提供了使用 `DTypePointer` 的解决方案来解决 `AnyPointer` 转换的问题。

- **Mojo 开发和 IDE 集成讨论**：成员们讨论了 Mojo 项目的结构，以及是否存在类似于 Rust 的 Cargo 的包管理系统。此外，还提到了适用于 PyCharm 等 IDE 的 Mojo 插件的可用性，并参考了 [插件链接](https://plugins.jetbrains.com/plugin/23371-mojo)，以及 JetBrains 团队对进一步支持 Mojo 的兴趣。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://plugins.jetbrains.com/plugin/23371-mojo">Mojo - IntelliJ IDEs Plugin | Marketplace</a>: 提供 Mojo 编程语言的基础编辑功能：语法检查和高亮、注释以及格式化。未来将添加更多新功能，请随时...</li><li><a href="https://devlog.hexops.com/2022/packed-structs-in-zig/">Packed structs in Zig make bit/flag sets trivial</a>: 在构建 Mach 引擎的过程中，我们在 Zig 中使用了一种巧妙的小模式，使得在 Zig 中编写 flag sets 比在其他语言中更加优雅。这里有一个简短的说明。</li><li><a href="https://tenor.com/view/bamboozled-gif-25267741">Bamboozled GIF - Bamboozled - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://willcrichton.net/notes/k-corrset/">Analyzing Data 180,000x Faster with Rust</a>: 如何通过 hash、index、profile、multi-thread 和 SIMD 达到令人惊叹的速度。</li><li><a href="https://cliffle.com/blog/rust-typestate/">
The Typestate Pattern in Rust - Cliffle
</a>: 未找到描述</li><li><a href="https://m.youtube.com/watch?v=Bf7vDBBOBUA&t=0s">Don&#39;t Write Comments</a>: 为什么你不应该在代码中写注释（写文档）。在 https://www.patreon.com/codeaesth... 获取代码示例、discord、歌曲名称等。
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot: 来自 *Modular*:
<https://twitter.com/Modular/status/1780676643176231240>
  

---


**Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1229796285352247366)** (2 messages): 

- **在 Modular 中复现的好奇心**：一位成员表达了在 Mojo 平台上复现某个概念或项目的兴趣，表示对潜在结果的期待。
- **AI 长期记忆与自我改进指南**：一位成员分享了一个视频教程，解释了如何构建具有长期记忆和自我改进能力的 AI Agent，旨在作为一个有用的资源。视频标题为 "解锁 AI Agent 的真正力量？！长期记忆与自我改进"，可在 [YouTube](https://youtu.be/7LWTZqksmSg?si=_tnJhoUcQr4Gojek) 上观看。

**提到的链接**：<a href="https://youtu.be/7LWTZqksmSg?si=_tnJhoUcQr4Gojek">Unlock AI Agent real power?! Long term memory &amp; Self improving</a>：如何为你的 AI Agent 构建长期记忆和自我改进能力？免费使用 AI 幻灯片生成工具 Gamma：https://gamma.app/?utm_source=youtube&amp;utm...

  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1229702087915339779)** (136 messages🔥🔥): 

- **用于将 Mojo 转换为 Python 代码的新 Python 包**：发布了一个名为 [mojo2py](https://github.com/venvis/mojo2py) 的新 Python 包，可将 Mojo 代码转换为 Python 代码。
- **需要全面的 Mojo 学习资源**：一位成员正在寻找从零开始学习 Mojo 的全面资源，并被引导至 [Mojo 编程手册](https://docs.modular.com/mojo/manual/)，该手册涵盖了基本概念，如 parameters vs. arguments、ASAP 概念、types 和 traits，以及需要重点重读的部分，如 owned arguments 和 transfer operator。
- **Struct 继承与代码复用**：讨论围绕在 Mojo 中实现某种形式继承的愿望展开，并提出了减少样板代码的建议，以及从父 struct 创建子 struct 的实例。虽然一种建议的方法是使用 traits 进行类型声明，但另一位成员澄清说，如果追求编译时优化，classes 可能比基于运行时的方法更合适。
- **Mojo 中 Conditional Conformance 的开始**：Mojo 中似乎正在推动实现 Conditional Conformance，最近成员之间分享的讨论和代码片段证明了这一点。对话涉及理解如何利用 Conditional Conformance 使 `str` 和 `print` 等标准库函数适用于不同的 Mojo 数据结构。
- **高级类型系统的挑战与前景**：围绕创建一个在编译时强制执行形状兼容性的 Numpy 风格 Mojo 库、在不支持运行时检查的情况下支持 `Variant` 数据结构的潜力，以及解决在单个列表中存储多个 variant 的特定问题，展开了激烈的技术辩论和头脑风暴。提出了各种方法并进行了概念性剖析，包括自定义 structs、enum parameters，以及在实现泛型和参数化代码的形状细化（shape refinement）方面的挑战。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/manual/">Mojo Manual | Modular Docs</a>：Mojo 编程语言的全面指南。</li><li><a href="https://www.youtube.com/watch?v=p3zo4ptMBiQ">Protocol-Oriented Programming in Swift / WWDC15 / Session 408</a>：Swift 设计的核心是两个极其强大的理念：面向协议编程（protocol-oriented programming）和一等值语义（first class value semantics）。这些概念中的每一个都受益于...</li><li><a href="https://github.com/venvis/mojo2py">GitHub - venvis/mojo2py: A python package to convert mojo code into python code</a>：一个将 Mojo 代码转换为 Python 代码的 Python 包 - venvis/mojo2py
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1229988474728349778)** (10 messages🔥): 

- **即兴草图的成功**：一位社区成员分享了一个用 Mojo 实现的“即兴”编程草图，发现其效果出奇地好，可以通过 [此 gist](https://gist.github.com/lsh/6ca8864a9cffef9e503d6262eb876893) 访问。
- **期待增强的 Tuple 功能**：即将到来的增强功能可能允许 Mojo 中的 `Tuple` 采用源自 `CollectionElement` 的 traits，从而为 HTML 渲染提供更**优雅的 struct 定义**。
- **正在使用的 Nightly 特性**：已澄清分享的代码使用了 **nightly 特性**，这可能会导致在当前的 Mojo 24.2 和 Mojo Playground 上出现编译错误。
- **Canny 边缘识别挑战**：一位来自法国的新社区成员（在 Python 的 Numba 方面很有经验）表示有兴趣在 Mojo 中实现 **Canny 边缘识别算法** 以进行性能对比。
- **面向新手的 Mojo 资源**：针对项目咨询的一个有帮助的回复包含了 [Mojo 文档](https://docs.modular.com/mojo/manual/get-started/) 的链接、语言入门指南，并引用了 Mojo SDK 和 [Mojo Playground](https://docs.modular.com/mojo/notebooks/) 等可用资源。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/manual/get-started/">Get started with Mojo🔥 | Modular Docs</a>：获取 Mojo SDK 或尝试在 Mojo Playground 中编写代码。</li><li><a href="https://docs.modular.com/mojo/">Mojo🔥 | Modular Docs</a>：一种弥合 AI 研究与生产之间鸿沟的编程语言，释放了速度和易用性。</li><li><a href="https://docs.modular.com/mojo/notebooks/">Mojo🔥 notebooks | Modular Docs</a>：我们为 Mojo Playground 创建的所有 Jupyter notebooks。</li><li><a href="https://gist.github.com/lsh/6ca8864a9cffef9e503d6262eb876893">html.mojo</a>：GitHub Gist：即时分享代码、笔记和片段。
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-blogs-vids](https://discord.com/channels/1087530497313357884/1151418796993683477/1230092571284340819)** (1 messages): 

- **探讨围绕 Mojo 的炒作**：最近在 YouTube 上发布了一场来自 PyCon Lithuania 的 [演讲](https://youtu.be/mhZFyzqdmi8)，题为“Maxim Zaks - Mojo 只是炒作吗？”，这引发了关于 Modular 聊天机器人在行业中地位的讨论。

**提及的链接**：<a href="https://youtu.be/mhZFyzqdmi8)">Maxim Zaks - Is Mojo just a hype?</a>：未找到描述

  

---


**Modular (Mojo 🔥) ▷ #[📰︱newsletter](https://discord.com/channels/1087530497313357884/1157711273111339088/)** (1 messages): 

Zapier: Modverse Weekly - 第 30 期
https://www.modular.com/newsletters/modverse-weekly-30
  

---


**Modular (Mojo 🔥) ▷ #[🏎engine](https://discord.com/channels/1087530497313357884/1212827673257316453/1230231689897447424)** (1 messages): 

仅提供了一条消息，未提及任何讨论点、主题或链接以供总结。如果您希望对 🏎engine 频道中更广泛的对话或特定主题进行总结，请提供相关消息。
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1229881110218145864)** (21 messages🔥):

- **新的 Mojo Nightly 版本：更新与变化**：Mojo 发布了新的 Nightly 更新，包含了标准库的更新，并提供了[详细的 diff](https://github.com/modularml/mojo/pull/2313/files)。此外，还可以在[此处](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)查看记录自上一个稳定版本以来变化的 changelog。
- **对非常规代码的热爱**：成员们对非常规的代码风格反应幽默，评论中表达了对其“糟糕”外观的喜爱，并滑稽地恳求为了可读性而*对 for 循环进行缩进*。
- **同伴压力 vs. 代码格式化规范**：有人建议不要屈服于关于代码缩进规范的同伴压力，但另一位成员认为采用 Mojo 格式化标准是不可避免的。
- **Nightly 更新引发困惑**：新的 Nightly 更新导致用户在基于 Traits 参数化的函数重载上产生困惑，引发了意外错误并讨论了解决方案。
- **Traits 优于蹩脚的变通方法及清理版本**：讨论中带有一点戏谑，倾向于使用“蹩脚方法”而非正确的 Trait 参数化，并对最近 Mojo Nightly 版本中的清理工作发表了评论。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/modularml/mojo/pull/2313/files">[stdlib] 根据 `2024-04-16` nightly/mojo 更新 stdlib，由 patrickdoc 提交 · Pull Request #2313 · modularml/mojo</a>：此 PR 使用与今日 Nightly 版本（mojo 2024.4.1618）对应的内部提交更新了 stdlib。未来，我们可能会直接将这些更新推送到 nightly 分支。</li><li><a href="https://github.com/modularml/mojo/blob/nightly/docs/changelog.md">nightly 分支下的 mojo/docs/changelog.md · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1229812396894457886)** (11 messages🔥): 

- **在 PyTorch 中寻求指导**：一位成员询问《Deep Learning with PyTorch》在出版 4 年后是否仍是一个好的起点。另一位成员确认，虽然 **PyTorch 核心** 变化不大，但编译器和分布式系统方面已有重大更新。

- **PyTorch 的演进与新版预告**：讨论明确了该书未涵盖 **Transformers 和 LLMs** 等主题；虽然第一部分和第二部分仍然有用，但关于部署的第三部分已经过时。同时透露，由新作者领衔的 **[新版本正在编写中](https://www.manning.com/books/deep-learning-with-pytorch-second-edition)**。

- **期待博客内容**：一位成员提到他们有一章关于 Attention/Transformers 的草稿，并考虑据此撰写一篇 **博客文章**。

**提到的链接**：<a href="https://www.manning.com/books/deep-learning-with-pytorch-second-edition">Deep Learning with PyTorch, Second Edition</a>：涵盖使用 PyTorch 创建神经网络所需的一切，包括 Large Language 和 Diffusion 模型。
 
 《Deep Learning with PyTorch, Second Edition》更新了最畅销的原版...

  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1229921763228389508)** (20 messages🔥):

- **CUDA 中的加速矩阵运算**：一位成员讨论了集成一种新的 fp16 精度 [CUDA 通用矩阵乘法 (GEMM) 实现](https://github.com/aredden/torch-cublas-hgemm/blob/master/src/simt_hgemv.cu)，在特定的矩阵运算基准测试（MxNxK = 1x4096x4096）中，其性能优于 PyTorch 的 GEMM 函数。
- **JIT 编译的挑战**：尽管新实现带来了性能提升，但另一位成员指出它在 `torch.compile` 下运行失败；并分享了崩溃详情：未编译的 Token 生成速度为 11.17 tokens/sec，而编译后的 Token 生成速度在崩溃前达到 64.4 tokens/sec，崩溃原因是与 'block_dim_x' 相关的未支持方法调用。
- **Block Size 参数探索**：讨论围绕新 GEMM Kernel 中 Block Size 的选择展开，成员们研究了使用 32x4 有效 Block Size 的情况，发现其似乎能产生更好的性能，并在[相关的 Gist 示例](https://gist.github.com/mobicham/9aa8dc0e64ea1cb7d4e44fef55e6a4b4)中分享了他们的观察结果。
- **关于 CUDA C++ 数据读取的咨询**：一位成员寻求在 CUDA C++ 应用程序中读取 CSV 或 Parquet 格式大型数据集的建议，思考并行执行的可能性，但未提供具体解决方案。
- **关于 CUDA Core 和线程调度的推测**：进一步的技术推测强调了更快的 Kernel 性能可能与每个流式多处理器（SM）使用 128 个总活动线程有关，考虑到在 4 个 Warp 中每个时钟周期调度 32 个线程。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/aredden/torch-cublas-hgemm/blob/master/src/simt_hgemv.cu">torch-cublas-hgemm/src/simt_hgemv.cu at master · aredden/torch-cublas-hgemm</a>：带有融合可选 bias + 可选 relu/gelu 的 PyTorch 半精度 GEMM 库 - aredden/torch-cublas-hgemm</li><li><a href="https://gist.github.com/mobicham/9aa8dc0e64ea1cb7d4e44fef55e6a4b4">zippy_gemv_hqq_gen.py</a>：GitHub Gist：即时分享代码、笔记和片段。
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1229781137547853864)** (2 条消息): 

- **寻找 F.Linear 的实现**：一位成员正在开发一个自定义 Backward 函数，该函数在输入为 **(bs, data_dim)** 时（类似于 **F.Linear**）运行正常。但在集成到 **Llama** 时由于输入维度差异遇到了问题，目前正在寻找 `F.Linear` 的前向/后向实现，而在指定的 *tools/autograd/templates/python_nn_functions.cpp* 中未能找到。
  

---


**CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1229840827010781205)** (2 条消息): 

- **Augmend 发布视频处理工具**：Augmend 在 [wip.augmend.us](http://wip.augmend.us) 上提供了一项正在开发中的功能，用于分析视频，并巧妙地添加了 OCR 和图像分割功能，以直接从视频屏幕中提取信息。完整服务将在 [augmend.com](http://augmend.com) 上提供，允许用户在任何视频中进行复制/粘贴和搜索内容。

- **波士顿动力展示全电动 Atlas 机器人**：波士顿动力在 YouTube 上发布了一段关于下一代人型机器人 *Atlas* 的视频；[All New Atlas | Boston Dynamics 视频](https://www.youtube.com/watch?v=29ECwExc-_M)展示了一款旨在用于实际应用的全电动机器人，并突出了几十年来机器人开发的进步。

**提到的链接**：<a href="https://www.youtube.com/watch?v=29ECwExc-_M">All New Atlas | Boston Dynamics</a>：我们正在揭晓下一代人型机器人——一款专为实际应用设计的全电动 Atlas 机器人。新款 Atlas 建立在数十年的...基础之上。

  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1229724970331013160)** (43 条消息🔥): 

- **新人咨询 PMPP 讲座**：一位新人询问了学习 **PMPP 讲座** 的常规会议安排。录制的讲座可以在特定频道找到，最后涵盖的章节是第 10 章。

- **WSL 性能分析 (Profiling) 困难**：一位用户表示在 WSL 上运行 **ncu profiler** 存在困难，怀疑是 **PATH 问题**，并强调 Windows 上的 NSight Compute 与 WSL 存在冲突。尽管安装了 nsight-compute，但找不到 `ncu` 命令。

- **CUDA Toolkit PATH 调整建议**：用户提出了几个故障排除步骤，重点是**将正确的 CUDA 路径添加到环境变量中**。一位用户提供了 **[NVIDIA 文档链接](https://docs.nvidia.com/gameworks/content/developertools/desktop/environment_variables.htm)**，以协助在 Windows 上设置环境变量。

- **发现版本不匹配**：发现存在版本不匹配问题，用户的环境配置为 CUDA 12.4，但尝试运行的是来自 CUDA **11.5 版本**的 `ncu`。添加路径并没有立即解决问题。

- **推荐使用 Windows 11 进行 WSL 2 Profiling**：另一位用户提到需要 **Windows 11** 才能在 WSL 2 上有效地对 CUDA 程序进行 Profiling，并分享了一篇[有用的博客文章](https://peterchng.com/blog/2024/03/02/profiling-cuda-programs-on-wsl-2/)，详细介绍了如何设置系统并解决常见问题。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.nvidia.com/gameworks/content/developertools/desktop/environment_variables.htm">Environment Variables</a>：未找到描述</li><li><a href="https://peterchng.com/blog/2024/03/02/profiling-cuda-programs-on-wsl-2/">Profiling CUDA programs on WSL 2</a>：未找到描述
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/)** (1 条消息): 

marksaroufim: https://www.youtube.com/watch?v=DdTsX6DQk24
  

---


**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1230169198844444825)** (5 条消息): 

- **RingAttention 工作组的难题**：一位核心成员透露，由于时间限制，他们无法在从事主业的同时兼顾 **RingAttention** 项目。他们提议进行讨论，以决定是由其他人继续该计划，还是暂时结束这个工作组的努力。
- **已安排决定性讨论**：会议已安排，旨在讨论 **RingAttention** 项目的未来以及谁可能继续其开发。
- **艰难抉择时刻**：该成员对退出 **RingAttention** 的决定表示遗憾，强调这一选择是经过对个人时间和福祉的慎重考虑后做出的。
- **参与者准备好对话**：团队成员确认了他们的可用性，并表示准备好参加即将举行的关于 **RingAttention** 未来的讨论。
- **会前准备**：其中一名成员通知其他人他们将很快加入会议，表明正在为预定的讨论做积极准备。
  

---


**CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1229814130295181473)** (36 条消息🔥): 

- **关于量化轴 (Quantization Axes) 的困惑**：在 `gpt-fast` 中发现，对 GPT 的 Q, K, V 使用 `axis=0` 进行量化存在问题，原因是量化过程中参数发生了混淆。正在进行的讨论建议分别对 Q, K 和 V 进行量化可能是一个解决方案，并指出 `weight_int4pack_mm` 目前仅支持 `axis=1`。

- **HQQ 中的速度与质量权衡**：探索了在 Half-Quadratic Quantization (HQQ) 中使用 `axis=0` 或 `axis=1` 时速度与质量之间的权衡。一位成员报告称，在 `gpt-fast` 上，两个轴的 Perplexity 性能均为 5.375。

- **追求进一步优化**：提到了使用 [Triton kernels](https://github.com/wangsiping97/FastGEMV/tree/main) 和其他方法（如伪数据）来优化 `axis=1` 的性能。他们指出，使用 autograd 和随机生成数据的方法比经过更多次迭代的 HQQ 结果稍好（5.3311 ppl）。

- **探索扩展功能并揭秘差异**：分享了关于 In-channel 变化对权重量化精度潜在影响的见解，提到使用 `axis=0` 的量化似乎产生了更好的结果。对话表明，与漫长的 autograd 优化相比，HQQ 能更有效地快速找到最优解。

- **分享实现细节和基准测试**：提供了实现细节的链接，例如带有 Transformer 的 [torch int4mm demo](https://github.com/mobiusml/hqq/blob/master/examples/backends/torchao_int4_demo.py)，以及[使用 autograd 的优化器代码](https://github.com/mobiusml/hqq/blob/master/hqq/core/optimize.py#L412)；讨论集中在通过向量化 fp16 乘法进一步加速操作的可能性，以及 2/3 bits 等更低精度量化的实用性。
<div class="linksMentioned">

<strong>提到的链接</strong>：

</div>

<ul>
<li>
<a href="https://huggingface.co/zhxch">zhxch (zhongxiaochao)</a>: 未找到描述</li><li><a href="https://github.com/mobiusml/hqq/blob/63cc6c0bbb33da9a42c330ae59b509c75ac2ce15/hqq/core/quantize.py#L81-L85),">hqq/hqq/core/quantize.py at 63cc6c0bbb33da9a42c330ae59b509c75ac2ce15 · mobiusml/hqq</a>: Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://huggingface.co/zhxchen17/scratch/tree/main">zhxchen17/scratch at main</a>: 未找到描述</li><li><a href="https://github.com/mobiusml/hqq/blob/master/hqq/kernels/hqq_aten_cuda_kernel.cu#L109-L115">hqq/hqq/kernels/hqq_aten_cuda_kernel.cu at master · mobiusml/hqq</a>: Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/wangsiping97/FastGEMV/tree/main">GitHub - wangsiping97/FastGEMV: High-speed GEMV kernels, at most 2.7x speedup compared to pytorch baseline.</a>: 高速 GEMV kernels，相比 PyTorch 基准测试最高提升 2.7 倍速度。 - GitHub - wangsiping97/FastGEMV</li><li><a href="https://github.com/mobiusml/hqq/blob/master/examples/backends/torchao_int4_demo.py">hqq/examples/backends/torchao_int4_demo.py at master · mobiusml/hqq</a>: Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/mobiusml/hqq/blob/master/hqq/core/optimize.py#L412">hqq/hqq/core/optimize.py at master · mobiusml/hqq</a>: Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1229714817586036756)** (76 条消息🔥🔥): 

- **Thunder 的 CUDA Python 扩展起飞**：用于通过 CUDA Python 扩展 PyTorch 的 [GitHub notebook](https://github.com/Lightning-AI/lightning-thunder/blob/main/notebooks/extend_thunder_with_cuda_python.ipynb) 在提升速度方面受到关注，尽管为了获得最大性能，仍需将其集成到 cuda-mode 并进行进一步优化（如利用 Tensor Cores）。

- **优化 Transformer 中的乘法**：成员们在 Profiling 工作中发现，最后的 Matmul 层和 Softmax 是计算成本的主要来源。正如关于缓存策略和 Kernel 优化的讨论所示，一个优化的 Classifier Kernel 为提升速度提供了机会。

- **提高 Softmax 和 Backpropagation 的效率**：讨论了关于避免全量概率矩阵 Materialization（显式化）的问题，转而关注必要的 Token 概率。[GitHub pull request #117](https://github.com/karpathy/llm.c/pull/117) 展示了在 Classification 层融合算子的努力。

- **缓存利用率与性能的相关性**：讨论了 Block Sizes 对缓存命中率的影响，揭示了更大的 Block 可能导致更好的缓存利用率。这一见解体现在一个 [优化的 CUDA Kernel](https://github.com/karpathy/llm.c/pull/150) 中，可能会在具有足够缓存的 GPU 上带来更好的性能。

- **支持多样化的模型架构进行 Benchmarking**：建议考虑初始化各种 GPT 模型架构进行 Benchmarking，以防止针对单一模型类型进行过拟合优化。重点放在准确复现如 GPT-2 等模型，以有意义地评估性能增强。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/Lightning-AI/lightning-thunder/blob/main/notebooks/extend_thunder_with_cuda_python.ipynb">lightning-thunder/notebooks/extend_thunder_with_cuda_python.ipynb at main · Lightning-AI/lightning-thunder</a>: 让 PyTorch 模型提速高达 40%！Thunder 是一个用于 PyTorch 的 source-to-source 编译器。它支持同时使用不同的硬件执行器；支持从单个到数千个 GPU。 - Lightning-AI/ligh...</li><li><a href="https://github.com/NVIDIA/cutlass/blob/main/media/docs/quickstart.md">cutlass/media/docs/quickstart.md at main · NVIDIA/cutlass</a>: 用于线性代数子程序的 CUDA 模板。通过在 GitHub 上创建账号来为 NVIDIA/cutlass 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/pull/150">Optimised version of fused classifier + bugfixes(?) by ademeure · Pull Request #150 · karpathy/llm.c</a>: 这是来自 #117 的酷炫新 kernel 的更快版本（目前仍仅限 /dev/cuda/）。最大的区别在于它针对每个 1024 宽度的 block（而非 32 宽度的 warp）处理一行进行了优化，这使得...</li><li><a href="https://github.com/karpathy/llm.c/pull/117">WIP: Fully fused classification layer by ngc92 · Pull Request #117 · karpathy/llm.c</a>: 这融合了 token 分类层中发生的所有逐点（pointwise）操作。这基本上让我们以大约仅前向传播的成本获得了前向/后向传播，因为...</li><li><a href="https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama)">SlimPajama: A 627B token, cleaned and deduplicated version of RedPajama - Cerebras</a>: Cerebras 构建了一个用于一键式训练 LLM 的平台，可以加速获取洞察的时间，而无需在大型的小型设备集群中进行编排。</li><li><a href="https://arxiv.org/abs/2304.08442">The MiniPile Challenge for Data-Efficient Language Models</a>: 预训练文本语料库日益增长的多样性使语言模型具备了跨各种下游任务的泛化能力。然而，这些多样化的数据集往往过于庞大...</li><li><a href="https://github.com/tysam-code/hlb-gpt/tree/main">GitHub - tysam-code/hlb-gpt: Minimalistic, extremely fast, and hackable researcher&#39;s toolbench for GPT models in 307 lines of code. Reaches &lt;3.8 validation loss on wikitext-103 on a single A100 in &lt;100 seconds. Scales to larger models with one parameter change (feature currently in alpha).</a>: 极简、极速且可黑客攻击的 GPT 模型研究员工具台，仅需 307 行代码。在单个 A100 上不到 100 秒即可在 wikitext-103 上达到 &lt;3.8 的验证损失。通过更改一个参数即可扩展到更大的模型（该功能目前处于 alpha 阶段）。
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[massively-parallel-crew](https://discord.com/channels/1189498204333543425/1229286073104994344/1229715140434333716)** (14 messages🔥): 

- **演示中的平板电脑大捷**：一名成员考虑使用 iPad 在幻灯片和实时书写之间切换进行演示。共识建议**使用单一设备完成这两项任务**，并强调了事先测试设置以确保体验流畅的重要性。

- **拒绝 NSFW**：针对聊天中发布**不当内容**的事件，成员们讨论了实现一个 **Discord bot** 来检测并防止此类内容的分享，并建议封禁违规者或限制其打字权限。

- **赋能活动创建**：已宣布现在每个人都拥有在服务器上创建新活动的**角色和权限**。这一变化使成员能够组织自己的聚会和讨论。

- **插曲与互动**：成员之间的随性互动包括对“Massively Helpful”等名称的幽默建议，以及在服务器名称语境下玩转“parallel”一词。这些时刻反映了社区互动轻松的一面。

- **技术技巧分享**：为希望直播演示的人提供了有用的建议，包括**使用 Wacom 手写板**以及通过不同设置等方法保持观众参与度。再次强调了尽早测试设置的重要性。
  

---



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1229725006347370537)** (167 messages🔥🔥): 

- **游戏助手开发咨询**：一位用户寻求关于创建一个结合 GPT-Vision、摄像头输入和概率计算的**游戏助手**的建议，用于实时多选题游戏。建议考虑使用 **Azure 或虚拟机**来运行高负载的计算软件，并推荐 TensorFlow 或 OpenCV 作为管理系统的可能工具。

- **AI 与人类认知之辩**：该频道举办了一场关于 AI 与人类之间根本区别的哲学讨论，涉及**记忆存储 (memory storage)**、计算能力，以及 AI 随着量子计算等技术的进步而发展出**类人推理与情感**的可能性。

- **理解非二元思维**：关于二元与非二元思维进行了广泛辩论，用户讨论了**二元思维与标签**在人类和 AI 中的适用性，以及梯度和混沌理论如何为认知和决策提供更准确的模型。

- **Claude 在文献综述方面的优势**：用户就适合撰写文献综述的 AI 模型交换了意见，建议在处理非技术性文学任务时使用 **Claude** 而非 OpenAI，并提到 **Gemini 1.5** 有助于虚构作品的创作。

- **应对 AI 相关复杂问题**：参与者报告并讨论了诸如意外的**账号封禁 (account terminations)** 和政策违规等问题，强调了在理解和遵守 AI 平台使用政策方面的挑战，并对经常遇到的缺乏透明度和支持表示担忧。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/disneyland-paris-parks-sleeping-beauty-gif-5070382">Disneyland Paris GIF - Disneyland Paris Parks - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://en.wikipedia.org/wiki/Turing_completeness">Turing completeness - Wikipedia</a>：未找到描述
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1229938287314206730)** (7 条消息): 

- **GPT 被削减了？**：一位用户评论说 GPT 似乎被显著改动或“脑叶切除 (lobotomised)”了，而另一位用户则为新的 **GPT-4 Turbo** 辩护，称其非常有效，并提到了可以使用的替代端点 (endpoints)。
- **报告缺陷很重要**：一位成员鼓励其他人报告 GPT 发出的任何有问题的消息，以改进其性能。
- **因成本讨论替代方案**：一位用户分享说，他们正在使用 Google Studio 上具有 100 万 Token 上下文窗口的 **Gemini 1.5** 作为替代方案，暗示成本是一个考量因素。
- **寻求知识库训练**：有人询问关于如何为自定义 GPT 准备知识库的培训或资源。
- **询问 Whisper v3 API 访问权限**：有人询问 **Whisper v3** 何时可以通过 API 使用，并指出距离其发布已近一年。
- **Token 注意力跨度在缩小？**：一位用户观察到 GPT-4 记忆过去输入的能力似乎受损，推测 Token 限制可能从 30,000 以上被削减了。
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1229929784675405966)** (5 条消息): 

- **鬼城的余响**：一位成员哀叹 prompt-engineering 频道活跃度下降，将缺乏讨论归咎于管理员和版主的过度审核。
- **愤愤不平的回顾**：一位用户认为他们在服务器被长期禁言 (timeout) 可能与活跃度下降有关，并相信其他人可能也面临过类似的处罚。
- **GPT-4-Turbo 的数学实力**：GPT-4-TURBO 成功解决了一个关于史密斯一家在餐桌上可能的座位安排数量的数学问题。
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1229929784675405966)** (5 条消息): 

- **OpenAI Discord 的寂静**：一位成员对 **api-discussions** 频道近期缺乏活跃度表示失望，指出该频道已经沉寂数周。
- **对服务器管理的思考**：该成员将这种不活跃归因于他们所感知的服务器管理员的过度审核。
- **禁言后的挫败感**：在被服务器**禁言 5 个月**后，该成员哀叹自己因试图帮助另一位用户而受到惩罚。
- **GPT-4-Turbo 的数学造诣**：一位用户报告称 **GPT-4-TURBO** 正确解决了一个涉及史密斯一家餐桌座位安排的组合数学问题。
  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1229835485623619755)** (3 条消息): 

- **Qdrant 混合云服务发布**：[@qdrant_engine](https://twitter.com/llama_index/status/1780275878230139293) 发布了混合云服务，支持将 Qdrant 作为托管服务、在边缘 (edge) 或在用户自己的环境中运行，同时保持对数据的完全控制。该公告还链接了一个关于如何设置的[深度教程](https://t.co/4nS9j9ruwR)。

- **LlamaIndex 与 Azure AI Search 联手**：由来自 Microsoft 的 Khye Wei 提供的[教程](https://t.co/lITCdlCejT)展示了如何将 LlamaIndex 与 Azure AI Search 结合，以创建具有 Hybrid Search 和 Query rewriting 功能的增强型 RAG 应用。

- **首日支持 MistralAI 的最新模型**：[MistralAI 的新款 8x22b 模型](https://t.co/WWbYp5lqXe)被描述为定义了开源模型的最先进水平，LlamaIndex 从发布首日即提供支持。此次发布包含由 @ravithejads 编写的 Mistral Cookbook，展示了 RAG、Query routing 和 Tool use。

**提到的链接**：<a href="https://t.co/WWbYp5lqXe">MistralAI Cookbook - LlamaIndex</a>：未找到描述

  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1229694863667367986)** (164 条消息🔥🔥)：

- **关于构建搜索引擎的咨询**：用户讨论了如何使用 LlamaIndex 构建搜索引擎。一位用户提供了一个[入门教程](https://docs.llamaindex.ai/en/stable/getting_started/starter_example/)，并强调了使用具有更高 `top_k` 值的 `retriever` 来检索排名靠前的文档。

- **理解 LLM 检索限制**：一位用户澄清说，他们需要从 Agent 中检索文档名称而不是答案，并将其与 Perplexity 的功能进行了对比。对话继续进行，用户参考了 LlamaIndex 的 `retriever` 及其设置。

- **身份验证问题**：多位用户遇到并讨论了与 API 身份验证相关的错误。错误消息指出 API 密钥不正确，从而引发了围绕环境变量和正确密钥使用的排查。

- **LlamaIndex 更新与问题修复**：用户协作尝试解决各种问题，特别关注一个 `BaseComponent` 错误，尽管尝试了多种排查步骤，一位用户仍无法解决该错误。有人通过 [GitHub pull request](https://github.com/run-llama/llama_index/pull/12882) 的形式提出了解决方案。

- **LLM 查询日志记录与活动模型检查**：关于 LlamaIndex 内部日志记录的讨论建议将日志级别从 `DEBUG` 调整为 `INFO`。一位用户试图确认哪个 LLM 在查询中处于活动状态，并得到了通过 `Settings.llm` 属性检查和设置 LLM 的建议。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/17IJlvx2M2iGu3weIttvwml2axAAt0Vk9?usp=sharing">Google Colaboratory</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example/">Starter Tutorial (OpenAI) - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/callbacks/LangfuseCallbackHandler/?h=lang">Langfuse Callback Handler - LlamaIndex</a>: 未找到描述</li><li><a href="http://localhost:port",>">未找到标题</a>: 未找到描述</li><li><a href="http://localhost:port"`>">未找到标题</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/llms/openai_like#llama_index.llms.openai_like.OpenAILike>).">Openai like - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/finetuning/embeddings/finetune_embedding/?h=embeddings+fine">Finetune Embeddings - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/create_llama_projects/tree/main/nextjs-edge-llamaparse">create_llama_projects/nextjs-edge-llamaparse at main · run-llama/create_llama_projects</a>: 通过在 GitHub 上创建账户，为 run-llama/create_llama_projects 的开发做出贡献。</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/multi_document_agents/">Multi-Document Agents - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/evaluation/answer_and_context_relevancy/">Answer Relevancy and Context Relevancy Evaluations - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/pull/12882">Catch validation errors by logan-markewich · Pull Request #12882 · run-llama/llama_index</a>: 部分用户在此遇到了奇怪的错误。让我们捕获验证错误，以防止不兼容的包版本导致核心崩溃。</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/llm/llama_2_llama_cpp#setup-llm>)">LlamaCPP - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/tools/openapi#llama_index.tools.openapi.OpenAPIToolSpec>)">Openapi - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/q_and_a/?h=summar#summarization">Q&A patterns - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/index_structs/doc_summary/DocSummary/?h=summary">Document Summary Index - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/response_synthesizers/custom_prompt_synthesizer/?h=summa">Pydantic Tree Summarize - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/schema#llama_index.core.schema.BaseComponent>)">Index - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/query_pipeline#llama_index.core.base.query_pipeline.query.QueryComponent>)">Index - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/query_pipeline/llm#llama_index.core.llms.llm.BaseLLMComponent>)">Llm - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/query_pipeline/llm#llama_index.core.llms.llm.LLMChatComponent>)">Llm - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1229807041850380348)** (2 条消息): 

- **寻求层级结构方面的建议**：一位成员正寻求指导，希望在 *LlamaIndex* 中使用 *ParentDocumentRetriever langchain* 为海量文档构建**父子层级结构**。
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1229696835212546058)** (58 条消息 🔥🔥): 

- **寻求 Pile-T5 详情**：一位用户在 EleutherAI 的 Discord 中询问 Pile-T5 模型的详情，并指向 [Hugging Face 集合页面](https://huggingface.co/collections/EleutherAI/pile-t5-65a76a0d0022dd270b385a66) 以获取更多信息。讨论中澄清了“序列长度（sequence length）”和“上下文窗口（context window）”是同一个概念，同时指出具有长序列长度的 encoder/decoder 模型非常稀缺。
  
- **Reka 的长 Enc-Dec 模型揭晓**：在讨论模型序列长度时，一位用户提到了 Reka 新的 encoder-decoder 模型，根据其 [核心技术报告](https://publications.reka.ai/reka-core-tech-report.pdf)，该模型支持高达 128k 的长度。

- **EleutherAI's Model Evaluation Harness Discussed**: 讨论了 EleutherAI 的模型评估框架（Model Evaluation Harness）：针对 EleutherAI 评估框架上的 ARC-challenge 进行了辩论，主要关注点在于模型查询中缺少“选项（choices）”的问题。提到该库最初的目标是复制 GPT-3 论文中的图表，并打算通过提供多种提示选项来标准化 MCQA 任务。

- **Research Scientist Interview Insights**: 研究科学家面试见解：用户分享了关于研究科学家面试的见解，解释说重点根据公司的不同而有很大差异，从不太强调传统的数据结构和算法问题，到高度看重候选人的演讲、论文以及获取资助的潜力。

- **Sequence Packing vs. Prepacking in LLMs**: LLM 中的 Sequence Packing 与 Prepacking：一场关于“prepacking”是否只是常规 sequence packing 的讨论在某篇新研究论文中展开。这引发了关于这些方法的创新性和先前文档记录的争论，并引用了 T5 论文以及即将发表的关于模型评估和效率的相关方法的出版物。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/siyan_zhao/status/1780288750624612850?s=46">来自 Siyan Zhao (@siyan_zhao) 的推文</a>: 🚨LLM 研究者们🚨想要在生成质量零退化的情况下，为你的 HuggingFace🤗 LLM 获得免费的速度和内存效率提升吗？介绍 Prepacking，一种简单的方法，可获得高达 6 倍的加速...</li><li><a href="https://huggingface.co/collections/EleutherAI/pile-t5-65a76a0d0022dd270b385a66">Pile-T5 - EleutherAI 收藏集</a>: 未找到描述</li><li><a href="https://x.com/srush_nlp/status/1779938508578165198">来自 Sasha Rush (@srush_nlp) 的推文</a>: Lazy twitter：NLP 课堂上的一个常见问题是“如果 xBERT 效果很好，为什么人们不把它做大？”但我意识到我只是不知道答案。我假设人们尝试过，但一个 l...</li><li><a href="https://huggingface.co/lintang/pile-t5-base-flan">lintang/pile-t5-base-flan · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/lintang">lintang (Lintang Sutawika)</a>: 未找到描述</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/b281b0921b636bc36ad05c0b0b0763bd6dd43463/lm_eval/tasks/arc.py#L61">lm-evaluation-harness/lm_eval/tasks/arc.py at b281b0921b636bc36ad05c0b0b0763bd6dd43463 · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/b281b0921b636bc36ad05c0b0b0763bd6dd43463/lm_eval/tasks/hendrycks_test.py#L153">lm-evaluation-harness/lm_eval/tasks/hendrycks_test.py at b281b0921b636bc36ad05c0b0b0763bd6dd43463 · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://huggingface.co/models?other=base_model:EleutherAI/pile-t5-base">Models - Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1229728886515040348)** (78 messages🔥🔥): 

- **New Transformer Architecture for Long Inputs**: 适用于长输入的新 Transformer 架构：最近提出的一种名为 [Feedback Attention Memory (FAM)](http://arxiv.org/abs/2404.09173) 的新型 Transformer 架构，旨在通过允许网络关注自身的潜在表示（latent representations）来处理无限长的序列，从而克服二次方注意力复杂度。FAM 的性能在长上下文任务上表现出显著提升。

- **Advances in Brain Decoding Research**: 大脑解码研究的进展：论文 [MindBridge](https://arxiv.org/abs/2404.07850v1) 介绍了一种新方法，通过仅使用一个模型即可实现跨受试者的大脑解码，解决了该领域的三个主要挑战：大脑大小的可变性、个体神经模式的差异以及新受试者数据的有限性。

- **Rethinking Scaling Laws' Accuracy**: 重新思考缩放法则（Scaling Laws）的准确性：Hoffmann 等人 (2022) 提出的计算最优缩放法则中指出的差异强调了数据透明度的重要性，因为[一项新分析](https://arxiv.org/abs/2404.10102)表明，除非进行了大量的实验，否则原始论文中极窄的置信区间是不合理的。

- **Expressive Power of State-Space Models**: 状态空间模型 (SSMs) 的表达能力：一项关于 [状态空间模型 (SSMs) 的分析](https://arxiv.org/abs/2404.08819) 引发了讨论，揭示了它们在状态追踪方面的表达能力与 Transformer 非常相似，且 SSMs 无法表达超出复杂度类 $\mathsf{TC}^0$ 的计算。对话还涉及了对先前相关作品的澄清和潜在误解。

- **Transformers, RL 与 EEG 反馈**：对话探讨了利用来自 EEG 的反馈进行强化学习（RL）的概念，但发现学术研究有限，主要存在于现有的产品实现中；同时也指出了此类尝试相关的复杂性和风险。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.08819">The Illusion of State in State-Space Models</a>：与之前无处不在的 Transformer 架构相比，状态空间模型（SSMs）已成为构建大语言模型（LLMs）的一种潜在替代架构。一个理论上的...</li><li><a href="https://arxiv.org/abs/2404.10102">Chinchilla Scaling: A replication attempt</a>：Hoffmann 等人 (2022) 提出了三种估计计算最优 Scaling Law 的方法。我们尝试复现他们的第三种估计程序，涉及拟合参数化损失函数...</li><li><a href="https://arxiv.org/abs/2404.10642">Self-playing Adversarial Language Game Enhances LLM Reasoning</a>：我们探索了大语言模型（LLMs）在名为 Adversarial Taboo 的双人对抗语言游戏中的自博弈（self-play）训练过程。在这个游戏中，攻击者和防御者进行交流...</li><li><a href="http://arxiv.org/abs/2404.09173">TransformerFAM: Feedback attention is working memory</a>：虽然 Transformers 彻底改变了深度学习，但其二次方注意力复杂度阻碍了它们处理无限长输入的能力。我们提出了 Feedback Attention Memory (FAM)，一种新型的...</li><li><a href="https://arxiv.org/abs/2404.10667">VASA-1: Lifelike Audio-Driven Talking Faces Generated in Real Time</a>：我们介绍了 VASA，这是一个在给定单张静态图像和一段语音音频剪辑的情况下，生成具有吸引人的视觉情感技能（VAS）的逼真说话面孔的框架。我们的首个模型 VASA-1 能够...</li><li><a href="https://arxiv.org/abs/2404.03592">ReFT: Representation Finetuning for Language Models</a>：参数高效微调（PEFT）方法寻求通过更新少量权重来适配大模型。然而，许多先前的可解释性工作表明，表示（representations）编码了丰富的...</li><li><a href="https://arxiv.org/abs/2404.10179">Scaling Instructable Agents Across Many Simulated Worlds</a>：构建能够在任何 3D 环境中遵循任意语言指令的具身 AI 系统是创建通用 AI 的关键挑战。实现这一目标需要学习将语言...</li><li><a href="http://arxiv.org/abs/2404.10179">Scaling Instructable Agents Across Many Simulated Worlds</a>：构建能够在任何 3D 环境中遵循任意语言指令的具身 AI 系统是创建通用 AI 的关键挑战。实现这一目标需要学习将语言...</li><li><a href="https://x.com/lambdaviking/status/1713945714684756019?s=46">来自 Will Merrill (@lambdaviking) 的推文</a>：[1/n] 思维链（CoT）如何改变 Transformers 的表达能力？与 @Ashish_S_AI 的新工作研究了增加 CoT/解码步骤如何根据...扩展 Transformers 可解决的问题。</li><li><a href="https://arxiv.org/abs/2404.07850v1">MindBridge: A Cross-Subject Brain Decoding Framework</a>：大脑解码是神经科学中的一个关键领域，旨在从获取的大脑信号中重建刺激，主要利用功能磁共振成像（fMRI）。目前，大脑解码...</li><li><a href="https://arxiv.org/abs/2103.13076">Finetuning Pretrained Transformers into RNNs</a>：Transformers 在自然语言生成方面已经超越了循环神经网络（RNNs）。但这伴随着显著的计算成本，因为注意力机制的复杂度随着...</li><li><a href="https://www.lesswrong.com/posts/gTZ2SxesbHckJ3CkF/transformers-represent-belief-state-geometry-in-their">Transformers Represent Belief State Geometry in their Residual Stream — LessWrong</a>：在作为 PIBBSS[1] 的附属成员期间产生。这项工作最初由 Lightspeed Grant 资助，随后在 PIBBSS 期间继续...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1230185118723735592)** (5 条消息): 

- **机器学习新手的 FLOPs 估计**：一位成员就如何从 SoundStream 论文中估计训练 FLOPs 寻求建议，并得到了计算每个 token 的前向和后向传播操作数的指导，对于 decoder-only transformers 使用公式 **6 * 参数量**。他们被引导参考[相关论文第 2.1 节](https://arxiv.org/abs/2001.08361)中的详细示例。

- **成本估算中的单 Epoch 假设**：针对关于训练成本估算的问题，一位成员澄清说，除非论文明确提到进行了多个 epoch，否则假设数据集只跑了一遍是明智的。

- **未报告数据集大小之谜**：一位成员强调了在**训练数据集大小**等细节未公开的情况下，很难从论文（如 SoundStream 论文）中估算训练成本。这给计算准确的成本估算带来了挑战。
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1229701793215414313)** (21 messages🔥): 

- **模型评估澄清**：讨论了如何使用 `lm-evaluation-harness` 评估自定义模型，特别是针对 `arc_easy` 任务，澄清了应该从 `loglikelihood` 返回一对值 (log-likelihood, is_greedy_decoding_equal_target)。会议指出，对于像 ARC 这样有多个选项的任务，会评估问题与每个答案组合的似然概率，似然概率最高的被视为正确答案。

- **理解 BPC 作为指标**：讨论了一篇将模型智能与文本压缩能力联系起来的论文，使用 BPC (bits per character) 作为智能的代理指标。辩论了考虑 BPC 而非 loss 的好处，结论是 BPC 是一个信息单位而不仅仅是 loss，这使其与压缩能力结合得更紧密。

- **分支对比与评估**：询问了项目的 `big-refactor` 分支相对于 main 分支的改进，该分支显然提供了显著的速度提升。此外，另一位用户想知道如何使用 `vllm` 保存每个问题的生成结果，并了解到使用 `--log_samples` 标志可以记录单个响应而不仅仅是聚合分数。

- **利用加速工具提升性能**：建议在评估大型模型时（特别是在 8 张 A100 的 Pod 上），使用 `--batch_size` 参数或 `accelerate launch --no_python lm-eval` 可能有助于提升速度和性能。

- **模型评估方法协助**：一位用户疑惑为什么在返回随机调试值时 `arc_easy` 任务的性能总是 0.25，后来了解到由于 ARC 有四个候选答案，随机选择会导致大约 25% 的正确率。会议解释了 MMLU 和 lambada_openai 等任务如何以不同方式使用 loglikelihood 输出。

**提及的链接**：<a href="https://x.com/arankomatsuzaki/status/1780073500536872990">Aran Komatsuzaki (@arankomatsuzaki) 的推文</a>：压缩线性代表智能。LLM 的智能（由平均基准测试分数反映）与其压缩外部文本语料库的能力几乎呈线性相关。repo: ht...

  

---


**Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1229989040820977775)** (1 messages): 

- **探索多模态学习**：jubei_ 分享了 [arXiv 上的两篇论文](https://arxiv.org/abs/2302.12247)，涉及多模态机器学习。第一篇论文提出了一种名为**总相关增益最大化 (TCGM)** 的信息论方法，用于半监督多模态学习，该方法能有效利用跨模态的未标记数据并提供理论保证。

- **深入研究半监督多模态融合**：讨论的论文解决了为多模态训练标记大型数据集的挑战，并强调了一种可以提高半监督设置下融合效率的方法。提到的*摘要节选*深入探讨了 **TCGM** 方法在多模态学习场景中识别 Bayesian 分类器的潜力。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2302.12247">Quantifying &amp; Modeling Multimodal Interactions: An Information Decomposition Framework</a>：近期对多模态应用的关注激增，产生了大量用于表示和整合不同模态信息的数据集和方法。尽管...</li><li><a href="https://arxiv.org/abs/2007.06793">TCGM: An Information-Theoretic Framework for Semi-Supervised Multi-Modality Learning</a>：融合来自多个模态的数据为训练机器学习系统提供了更多信息。然而，为每个模态标记大量数据是非常昂贵且耗时的...
</li>
</ul>

</div>
  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1229892954500370442)** (10 messages🔥):

- **IDEFICS-2 以卓越的多模态能力亮相**：[IDEFICS-2](https://huggingface.co/spaces/HuggingFaceM4/idefics-8b) 正式发布，拥有 8B 参数，采用 Apache 2.0 许可证，支持高达 980 x 980 的高分辨率图像处理，并提供两个 Checkpoint（包括指令微调版本）。该多模态模型在视觉问答和文档检索等任务中表现出色。
  
- **IDEFICS-2 的 Chat 变体即将推出**：专注于对话的 IDEFICS-2 变体预计将在未来几天内发布。当前版本擅长视觉问答和其他非对话任务，对话版本随后即至。

- **展示巧妙的多模态交互**：分享的一个示例展示了 IDEFICS-2 的能力，它无缝结合了文本识别、颜色知识和数学运算，以解释和操作图像内容，包括解决具有显著背景噪声的 CAPTCHA。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/HuggingFaceM4/idefics-8b">Idefics 8b - a Hugging Face Space by HuggingFaceM4</a>：未找到描述</li><li><a href="https://x.com/lunarflu1/status/1780228654397599904">来自 lunarflu (@lunarflu1) 的推文</a>：来自 IDEFICS-2 @huggingface 的酷炫多模态交互：1. 从图像中检测数字 2. 对数字进行数学运算 3. 检索背景颜色 4. 去除色素 -> 结果颜色 5. 最终结果：...</li><li><a href="https://huggingface.co/blog/idefics2">Introducing Idefics2: A Powerful 8B Vision-Language Model for the community</a>：未找到描述</li><li><a href="https://huggingface.co/HuggingFaceM4/idefics2-8b">HuggingFaceM4/idefics2-8b · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/reach_vb/status/1779998271546474593">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：Idefics 2 x Transformers! 🔥 尝试在实际场景中使用 Idefics 2 8B。令人惊叹的是，你只需不到 10 行代码就能完成这一切！制作了一个快速的屏幕录像来演示该模型...</li><li><a href="https://huggingface.co/HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1">HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/argilla/distilabel-capybara-dpo-7k-binarized">argilla/distilabel-capybara-dpo-7k-binarized · Datasets at Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/papers/2403.07691">论文页面 - ORPO: Monolithic Preference Optimization without Reference Model</a>：未找到描述</li><li><a href="https://github.com/huggingface/alignment-handbook/tree/main/recipes/zephyr-141b-A35b">alignment-handbook/recipes/zephyr-141b-A35b at main · huggingface/alignment-handbook</a>：将语言模型与人类和 AI 偏好对齐的稳健方案 - huggingface/alignment-handbook</li><li><a href="https://x.com/narsilou/status/1778887423713333648">来自 Nicolas Patry (@narsilou) 的推文</a>：TGI 2.0 发布了！- 永久回归完全开源 (Apache 2.0) - 现存最快的推理服务器 (Cohere R+ 达到 110 tok/s，支持 Medusa 推测解码) - 支持 FP8 - 支持 Mixtral 8x22B！...</li><li><a href="https://x.com/xenovacom/status/1778812177215881395">来自 Xenova (@xenovacom) 的推文</a>：介绍 MusicGen Web：直接在浏览器中由 AI 驱动的音乐生成，基于 🤗 Transformers.js 构建！🎵 100% 本地运行，意味着无需调用 API！🤯 作为一个静态网站提供服务...</li><li><a href="https://x.com/AndrewYNg/status/1779905922602782752">来自 Andrew Ng (@AndrewYNg) 的推文</a>：LLM 可能需要数 GB 的内存来存储，这限制了在消费级硬件上运行。但量化可以显著压缩模型，使开发人员能够使用更广泛的模型选择...</li><li><a href="https://huggingface.co/blog/vlms">Vision Language Models Explained</a>：未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1229751537140502558)** (85 条消息🔥🔥): 

- **Langchain 学习咨询**：一位参与者表示有兴趣*学习 Langchain* 来构建 Agent 式 LLM，但收到了另一位成员的建议，认为*实现自定义解决方案*可能会更高效。

- **寻求 ML 社区见解**：研究 *ML 民主化* 的学生分享了一个调查链接，请求机器学习社区的参与。调查可通过[此链接](https://forms.gle/UvGdWrZhphoDFGQ99)访问。

- **文件转换故障**：一位成员在将 HuggingFace Safetensors 转换为 llama.cpp GGUF 时遇到问题，收到 "is not a directory" 错误。他们被建议确保命令中的路径在文件名之前结束。

- **非请求学术摘要爆发详解**：一位用户遇到了 llama.cpp 在交互模式（interactive mode）启动时生成非请求内容的问题，无意中输出了诸如“银纳米颗粒的抗真菌特性”之类的摘要。讨论转向寻求解决方案或正确的命令，以使交互能够响应用户输入。

- **探索用于 SQUAD 的 Decoder-only 模型**：有人询问如何对 Mistral 等 Decoder-only 模型的输出进行后处理，以便进行 SQUAD 评估。该成员正在从 *open github repos* 中寻找处理此类任务的灵感。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/zero-gpu-explorers">zero-gpu-explorers (ZeroGPU Explorers)</a>：无描述</li><li><a href="https://huggingface.co/docs/diffusers/using-diffusers/inpaint">Inpainting</a>：无描述</li><li><a href="https://forms.gle/UvGdWrZhphoDFGQ99">机器学习民主化 - 调查</a>：感谢您抽出时间回答这份关于人们机器学习经验的调查，耗时不超过 5 分钟。在本次调查中，“机器学习”将被简称为...</li><li><a href="https://youtu.be/7LWTZqksmSg?si=_tnJhoUcQr4Gojek">解锁 AI Agent 的真正力量？！长期记忆与自我改进</a>：如何为您的 AI Agent 构建长期记忆和自我改进能力？免费使用 AI 幻灯片生成器 Gamma：https://gamma.app/?utm_source=youtube&amp;utm...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1229984219179257969)** (3 条消息): 

- **探索知识图谱**：一位成员分享了一篇[博客文章](https://mlabonne.github.io/blog/posts/Article_Improve_ChatGPT_with_Knowledge_Graphs.html)，讨论了如何通过集成 **Knowledge Graphs** 来提升 Chatbot 的性能，并提供了进一步探索该概念的链接。

- **量化知识的探索**：一位成员正在通过 Deep Learning AI 提供的短课学习 **quantization**，这表明其正在持续学习机器学习优化技术。

- **使用 RAG 进行多语言文本检索**：一位成员询问了如何为多语言文本集实现基于 **Retrieval-Augmented Generation (RAG)** 的高效检索系统的技巧，并正在寻找多语言场景下的更新或最佳实践。

**提到的链接**：<a href="https://mlabonne.github.io/blog/posts/Article_Improve_ChatGPT_with_Knowledge_Graphs.html">ML Blog - 通过知识图谱改进 ChatGPT</a>：利用 LangChain 为 LLM 赋能知识图谱

  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1229937791295553546)** (7 条消息): 

- **快速生成喷溅艺术**：[HuggingFace 上的 Splatter Image Space](https://huggingface.co/spaces/szymanowiczs/splatter_image) 是一个可以快速生成 **splatter art** 的工具。

- **深入研究多模态 RAG**：来自 **LlamaIndex** 的一位发言人分享了关于 **Multi-Modal RAG (Retrieval Augmented Generation)** 的资源，展示了结合语言和图像的应用。在其[文档](https://docs.llamaindex.ai/en/stable/use_cases/multimodal/)中可以了解 **RAG** 的索引、检索和合成过程如何与图像设置集成。

- **LLM 用户分析揭秘**：Nebuly 推出了一个 **LLM user analytics playground**，无需登录即可访问，提供了一个探索分析工具的场所。欢迎对其[平台](https://playground.nebuly.com/home?projectId=69269458-99d7-4022-abb4-949c7b352649&homeTab=Overview)提供反馈。

- **ML 扩展到新领域**：这篇 IEEE 论文强调了一个 **Machine Learning (ML)** 可以广泛应用的*有趣场景*。该论文可以在 [IEEE Xplore 数字图书馆](https://ieeexplore.ieee.org/abstract/document/9249641)中找到。

- **Snowflake 推出顶级文本嵌入模型**：Snowflake 发布了 **Arctic embed 系列模型**，声称是世界上最适用于检索用例的实用文本嵌入模型。该系列模型在平均检索性能上超越了其他模型，并以 Apache 2.0 许可证开源，可在 [Hugging Face](https://huggingface.co/Snowflake/snowflake-arctic-embed-m) 上获取，并很快会进入 Snowflake 自身的生态系统。更多信息请阅读其[博客文章](https://www.snowflake.com/blog/introducing-snowflake-arctic-embed-snowflakes-state-of-the-art-text-embedding-family-of-models/)。

- **多步工具提升效率**：Medium 上的一篇文章讨论了由 LangChain 和 Cohere 开发的 **multi-step tools** 如何在各种应用中释放效率提升。全文可参阅提供的 [Medium 文章](https://medium.com/ai-advances/unlocking-efficiency-the-power-of-multi-step-tools-with-langchain-and-cohere-7d1ea571ebed)。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/szymanowiczs/splatter_image">Splatter Image - szymanowiczs 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/use_cases/multimodal/">多模态应用 - LlamaIndex</a>：未找到描述</li><li><a href="https://playground.nebuly.com/home?projectId=69269458-99d7-4022-abb4-949c7b352649&homeTab=Overview">Nebuly AI</a>：未找到描述</li><li><a href="https://www.snowflake.com/blog/introducing-snowflake-arctic-embed-snowflakes-state-of-the-art-text-embedding-family-of-models/">Snowflake 发布适用于检索场景的实用文本嵌入模型</a>：Snowflake-arctic-embed 以 Apache 2.0 许可证向开源社区开放。
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1229733181960224798)** (19 messages🔥): 

- **针对 Prompt 微调的 BLIP 模型**：BLIP 模型已经过微调，可以生成适用于图像 Prompt 的长描述，并在 Hugging Face 上提供实时 Demo。点击[此处](https://huggingface.co/unography/blip-large-long-cap)查看增强的功能。
  
- **模型对比变得简单**：一个对比不同图像字幕（captioning）模型的 Hugging Face Space 已发布，该空间复制了另一位用户现有的对比空间。[探索模型对比](https://huggingface.co/spaces/unography/comparing-captioning-models)。

- **Serverless 推理支持最大输出长度**：有用户咨询关于通过 curl 进行模型推理时的最大输出长度问题，官方澄清可以使用 transformers 的 pipeline 中支持的参数，包括 `max_new_tokens`。

- **IP-Adapter Playground 亮相**：一个新的 Hugging Face Space 推出了 IP-Adapter，它允许使用图像作为 Prompt 进行 text-to-image、image-to-image 和 inpainting 功能。立即体验 [IP-Adapter Playground](https://huggingface.co/spaces/tonyassi/IP-Adapter-Playground)。

- **Transformers Pipeline 新增 'Push to Hub'**：transformers 库的 main 分支现在包含 `push_to_hub` 方法，允许将 pipeline 输出直接推送到 Hugging Face Model Hub。用户可以从 main 分支尝试此功能或等待下一个版本发布。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/EduardoPacheco/Grounded-SAM">Grounded SAM - EduardoPacheco 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/tonyassi/IP-Adapter-Playground">IP-Adapter Playground - tonyassi 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://playground.nebuly.com/home?projectId=69269458-99d7-4022-abb4-949c7b352649&homeTab=Overview">Nebuly AI</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/unography/comparing-captioning-models">Comparing Captioning Models - unography 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task">详细参数</a>：未找到描述</li><li><a href="https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.ImageToTextPipeline">Pipelines</a>：未找到描述</li><li><a href="https://huggingface.co/docs/transformers/en/main_classes/pipelines#transforme">Pipelines</a>：未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1229691536586051624)** (11 messages🔥): 

- **寻求 SDXL Tagger 升级**：一位成员询问是否有替代 SDXL 的 **wd14 tagger** 的其他打标器，正在寻找更优的选择。

- **寻找 PDF 转 LaTeX 工具**：一位成员询问是否有任何**开源 PDF 转 LaTeX** 转换器，或者能够处理整个 PDF 页面（包括文本和数学表达式）且不需要精确位置信息的**图像转 LaTeX** 转换器。

- **用于方程式转换的 LaTeX-OCR**：有人指出，有一个**将方程式图像转换为 LaTeX 代码的优秀开源仓库**：[GitHub 上的 LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR)，它利用了 Vision Transformer (ViT)。

- **文本到 LaTeX 的转换没有完美方案**：由于 LaTeX 编译器和宏包的特殊性，将文本转换为 LaTeX 非常复杂，因此有人认为手动重写可能更具实用性。

- **选择性文本提取挑战**：一位用户正在寻找一种根据最大和最粗字体从图像中提取特定行文本的方法。建议尝试使用 **Paddle OCR** 来完成此任务。

**提到的链接**：<a href="https://github.com/lukas-blecher/LaTeX-OCR">GitHub - lukas-blecher/LaTeX-OCR: pix2tex: Using a ViT to convert images of equations into LaTeX code.</a>：pix2tex：使用 ViT 将公式图像转换为 LaTeX 代码。- lukas-blecher/LaTeX-OCR

---

**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1229757182921150474)** (17 条消息🔥): 

- **LoRA 配置查询**：一名成员正在实验他们的 LoRA 配置，并就将 bias 设置为 'all'、'none' 或 'lora_only' 的影响寻求建议。

- **为微调 Roberta 准备数据集**：一名成员正在寻求指导，准备一个包含超过 100,000 条记录和 20 多个特征的 CSV 数据集，用于微调 ROBERTA 模型以构建问答聊天机器人。随后，他们澄清该数据集包含有关药物的详细信息，具有发布日期和药物类型等多种列。

- **用于主题建模的 BERTopic**：一名成员推荐了 [BERTopic](https://maartengr.github.io/BERTopic/index.html)，这是一种使用 🤗 transformers 和 c-TF-IDF 的主题建模技术，并表示对结果感到满意，尽管目前在将种子词（seed words）转换为短语以创建主题模型方面存在挑战。

- **寻求使用 HF Trainer 的 T5 训练代码**：一名成员询问在哪里可以找到使用 Hugging Face 的 Trainer 训练 T5 的代码。另一名成员分享了 [EleutherAI 的 GitHub](https://github.com/EleutherAI/improved-t5) 仓库链接，其中包含改进版 T5 的开源脚本，并建议查看 [simpleT5](https://github.com/Shivanandroy/simpleT5) 以获取更简单的方法。

- **在 AutoModelForVision2Seq 中恢复模型下载**：一名成员询问如何恢复使用 AutoModelForVision2Seq 的模型下载过程，但未收到直接回复。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://maartengr.github.io/BERTopic/index.html">Home</a>：利用 BERT 和基于类别的 TF-IDF 来创建易于解释的主题。</li><li><a href="https://github.com/EleutherAI/improved-t5">GitHub - EleutherAI/improved-t5: Experiments for efforts to train a new and improved t5</a>：训练全新改进版 T5 的实验工作 - EleutherAI/improved-t5</li><li><a href="https://github.com/Shivanandroy/simpleT5">GitHub - Shivanandroy/simpleT5: simpleT5 is built on top of PyTorch-lightning⚡️ and Transformers🤗 that lets you quickly train your T5 models.</a>：simpleT5 基于 PyTorch-lightning⚡️ 和 Transformers🤗 构建，让你能够快速训练 T5 模型。 - Shivanandroy/simpleT5
</li>
</ul>

</div>

---

**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1229717436723626024)** (8 条消息🔥): 

- **截断 Token 的担忧**：一位用户提到，Prompt 中被**截断的 Token**（例如 "hdr"）被忽略了，这暗示处理过程中可能存在问题。讨论中对此问题表示赞同，但未提供解决方案。
- **Compel 库维护**：针对截断 Token 问题，有人提到了 **Compel 库**，但担心该库目前可能未得到维护。
- **用于视频分析和文本生成的模型**：有人请求一个能够**分析视频内容**以生成标题和描述的模型，但讨论线程中未提供解决方案。
- **征求对测试方法的“吐槽”**：一位用户分享了一个**测试方法/套件**的链接，并请求从用户角度提供一些建设性的批评。测试方法/套件的具体内容未被讨论。
- **恢复 Hugging Face 模型训练**：一位用户询问**恢复 Hugging Face 模型**所需的代码更改，但对话中尚未给出答案。

---

**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1229737177861066762)** (44 条消息🔥): 

- **Idefics2 隆重登场**：全新的多模态模型 [Idefics2](https://huggingface.co/HuggingFaceM4/idefics2-8b) 现已发布，它支持图像和文本输入，与前代 Idefics1 相比，在 OCR 和视觉推理能力上有所提升。它发布了两个 Checkpoint，包括基础版和微调版，并采用 Apache 2.0 协议授权。

- **NVidia 的先发制人？**：有传言称，为了应对来自 AMD 新进展的竞争压力，NVidia 可能会加快 [RTX 5090](https://www.pcgamesn.com/nvidia/rtx-5090-5080-paper-launch) 的发布，最早可能在 2024 年 6 月的 Computex 展会上亮相。

- **关于 AI 训练的硬件讨论**：成员们讨论了使用 Nvidia A6000 GPU 进行 QLoRa 等模型的训练和推理的可行性，辩论了 VRAM 是否充足以及是否需要更强大的配置。

- **Cosmo-1b 遗忘与合并实验揭晓**：在旨在减少灾难性遗忘（catastrophic forgetting）的训练方法对比实验中，Model Stock 合并展示了结合多种训练方案的潜力。训练集验证结果中详细对比统计数据的分享，激发了进一步探索不同微调（fine-tuning）方法优势的兴趣。

- **深入探讨 Dora 和 QLoRa**：用户就 Dora 等新型参数高效微调（PEFT）方法的有效性进行了技术讨论，并将其与 QLoRa 进行了对比，讨论了配置细节，并指出了每种方法在性能和资源消耗方面的特性。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.pcgamesn.com/nvidia/rtx-5090-5080-paper-launch">Nvidia’s RTX 5090 and 5080 could arrive much sooner than expected, but there’s a big catch</a>：泄露消息指出，由于来自 AMD 的竞争，新款 Nvidia Blackwell GeForce GPU 的上市时间可能比原计划提前得多。</li><li><a href="https://huggingface.co/HuggingFaceM4/idefics2-8b">HuggingFaceM4/idefics2-8b · Hugging Face</a>：未找到描述。
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1229811042507559054)** (2 条消息): 

- **关于 Bot 用途的咨询**：一位用户通过简单的“Oooooo how do I use this?”表达了好奇，表示有兴趣了解该 Bot 的功能。
- **垃圾信息警报**：一条针对整个群组的垃圾信息宣传了不当内容，并附带了 Discord 邀请链接。
  

---


**OpenAccess AI Collective (axolotl) ▷ #[other-llms](https://discord.com/channels/1104757954588196865/1104758057449308220/)** (1 条消息): 

aquash1553: @everyone Best OnlyFans Leaks & Teen Content 🍑 🔞 discord.gg/s3xygirlsss
  

---


**OpenAccess AI Collective (axolotl) ▷ #[manticore](https://discord.com/channels/1104757954588196865/1107775871428870177/)** (1 条消息): 

aquash1553: @everyone Best OnlyFans Leaks & Teen Content 🍑 🔞 discord.gg/s3xygirlsss
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1229719185668702310)** (13 条消息🔥): 

- **澄清 'train_on_input' 标志的作用**：展开了关于 'train_on_input' 参数的讨论，透露禁用该参数意味着模型不会计算输入部分的 Loss，因此不再对其进行预测。这澄清了无论如何输入都会作为训练期间 Context 的一部分，但在关闭该参数的情况下，模型不会在 Loss 计算方面受输入引导。

- **理解训练中的 Loss**：强调了 *Loss* 确实是训练的一个关键方面，因为它引导模型改进，而禁用 'train_on_input' 会停止输入部分的这一过程。如果未启用 *eval* 设置，这一过程对于模型的学习变得更加无关紧要。

- **关于费用和 OnlyFans 链接的查询**：一名成员询问了某项未指明服务的费用，另一名用户发布了似乎是 OnlyFans 相关链接的推广消息，邀请成员加入另一个承诺提供独家内容的 Discord 服务器。
  

---


**OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1230227447849681008)** (3 条消息): 

- **不当内容警报**：该频道出现了一例宣传 *OnlyFans 泄露和露骨内容* 的垃圾信息，并附带 Discord 服务器邀请链接。
- **社区监督员在行动**：成员们迅速识别了垃圾信息并将其标记为 *pornspam*，提醒他人这些消息的不当性质。
  

---


**OpenAccess AI Collective (axolotl) ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/)** (1 条消息): 

aquash1553: @everyone Best OnlyFans Leaks & Teen Content 🍑 🔞 discord.gg/s3xygirlsss
  

---


**OpenAccess AI Collective (axolotl) ▷ #[hippogriff](https://discord.com/channels/1104757954588196865/1113355318211137576/)** (1 条消息): 

aquash1553: @everyone Best OnlyFans Leaks & Teen Content 🍑 🔞 discord.gg/s3xygirlsss
  

---

**OpenAccess AI Collective (axolotl) ▷ #[minotaur](https://discord.com/channels/1104757954588196865/1116465236715786310/)** (1 messages): 

aquash1553: @everyone 最佳 OnlyFans 泄露与青少年内容 🍑 🔞 discord.gg/s3xygirlsss
  

---


**OpenAccess AI Collective (axolotl) ▷ #[bots](https://discord.com/channels/1104757954588196865/1117282691121954836/)** (1 messages): 

aquash1553: @everyone 最佳 OnlyFans 泄露与青少年内容 🍑 🔞 discord.gg/s3xygirlsss
  

---


**OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/)** (1 messages): 

aquash1553: @everyone 最佳 OnlyFans 泄露与青少年内容 🍑 🔞 discord.gg/s3xygirlsss
  

---


**OpenAccess AI Collective (axolotl) ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/)** (1 messages): 

aquash1553: @everyone 最佳 OnlyFans 泄露与青少年内容 🍑 🔞 discord.gg/s3xygirlsss
  

---


**OpenAccess AI Collective (axolotl) ▷ #[deployment-help](https://discord.com/channels/1104757954588196865/1163840836472148058/)** (1 messages): 

aquash1553: @everyone 最佳 OnlyFans 泄露与青少年内容 🍑 🔞 discord.gg/s3xygirlsss
  

---


**OpenAccess AI Collective (axolotl) ▷ #[docs](https://discord.com/channels/1104757954588196865/1167137552470392842/)** (1 messages): 

aquash1553: @everyone 最佳 OnlyFans 泄露与青少年内容 🍑 🔞 discord.gg/s3xygirlsss
  

---


**OpenAccess AI Collective (axolotl) ▷ #[shearedmistral](https://discord.com/channels/1104757954588196865/1190770223763165244/)** (1 messages): 

aquash1553: @everyone 最佳 OnlyFans 泄露与青少年内容 🍑 🔞 discord.gg/s3xygirlsss
  

---


**OpenAccess AI Collective (axolotl) ▷ #[replicate-help](https://discord.com/channels/1104757954588196865/1197414694248534116/)** (1 messages): 

aquash1553: @everyone 最佳 OnlyFans 泄露与青少年内容 🍑 🔞 discord.gg/s3xygirlsss
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1229738683494699008)** (36 messages🔥): 

- **简化按 Epoch 保存模型**：一位成员询问如何配置 Axolotl 仅在训练结束时保存模型，而不是每个 Epoch 都保存。解决方案包括将训练参数中的 `save_strategy` 调整为 `"no"`，并实现一个自定义回调（callback）以便在训练完成时手动保存。

- **选择微调的入门模型**：当被问及适合微调的小型模型时，推荐了 "TinyLlama-1.1B-Chat-v1.0"，因为它易于管理，适合快速实验。成员们被引导至 Axolotl 仓库查看示例配置，如 `pretrain.yml`。

- **Axolotl 使用与数据格式化指南**：讨论了 `model_type`、`tokenizer_type` 等概念，以及如何为 Axolotl 训练格式化数据集，特别是与使用 "TinyLlama-1.1B-Chat-v1.0" 模型相关的部分。对于文本到颜色代码生成的任务，建议在不使用 "system" 提示词的情况下构建数据集结构，并将其作为 Hugging Face 数据集上传（如果尚未提供）。

- **数据集上传的 CSV 结构说明**：寻求关于上传到 Hugging Face 以供 Axolotl 使用的数据集是否需要单列 CSV 格式的澄清。格式化后的示例应按行分隔，每行包含根据模型要求构建的输入和输出。

- **发布不当内容**：一名用户发布了推广未经授权内容的消息，这与频道的技术讨论无关，也不符合社区准则。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

</div>

<ul>
<li>
<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=c4585711-a0f4-4fe4-8055-816941329e8d)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=5b7d5162-b9f5-4a2b-83e0-b2154f15fe04)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=12ea1d05-4725-46ae-ac43-42fdae27790a)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=ae2df564-24d0-4c41-9f77-a8ea154566bb)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=eca9e87b-1d42-427c-8a91-59f42a3da0f8)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=ccfe189d-d5fa-4308-9afe-8a86c48a0141)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1229697419386552383)** (5 条消息): 

- **模型微调咨询**：一位成员就如何使用包含颜色代码和描述的特定数据集对 **TinyLlama** 模型进行微调的数据预处理寻求建议。目标是训练 TinyLlama 根据给定的描述预测颜色代码。

- **模型准备指南**：一份回复概述了微调 **TinyLlama** 的步骤，包括以可用格式准备数据集，并进行适合该任务的 Tokenization 和格式化。回复中未提供具体细节或链接。

- **发布了无关内容**：频道中发布了一条推广 **OnlyFans 泄露和内容** 的无关消息。该消息提供了一个 Discord 加入链接。

**提到的链接**：<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=af0c71b5-451f-4893-8158-1dfa36a9a10b)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。

  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1229722146042023997)** (68 条消息🔥🔥): 

- **提供全面的 LLM 基准测试**：分享了一个提供信息的网站 [llm.extractum.io](https://llm.extractum.io/)，该网站详细介绍了按各种基准测试排名的开源语言模型。这些模型使用 ELO 分数、HuggingFace 排行榜分数以及多项特定任务的准确度测量进行评估。
- **AI Agent 雇佣人类**：介绍了一个名为 [Payman AI](https://www.paymanai.com/) 的创新项目，它使 AI Agent 能够为自己无法完成的任务向人类支付报酬。该服务旨在支持 AI 与人类在设计、编程和法律等各个领域的共生关系。
- **AI 推理集成到 Supabase**：Supabase 宣布了一个易于使用的 API，用于在其 Edge Functions 中运行 AI 推理模型。新的会话初始化允许像 `gte-small` 这样的 AI 模型直接在数据库服务中处理查询。
- **期待 "Llama 3" 发布**：讨论包括对 "Llama 3" 发布的推测和传闻，社区内的期待感日益增强。上下文表明，Llama 3 的揭晓可能与即将到来的伦敦黑客松有关。
- **GPT-5 发布前 OpenAI 的 API 扩展**：[OpenAI 对 Assistants API 的更新介绍](https://x.com/OpenAIDevs/status/1780640119890047475) 已被披露，引发了关于公司未来走向的讨论，特别是随着 GPT-5 可能即将发布。用户正在辩论此类平台的质量和性能，以及对 AI 初创公司的潜在影响。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://mistral.ai/news/mixtral-8x22b/">更便宜、更好、更快、更强</a>：继续推动 AI 的前沿，并使其对所有人开放。</li><li><a href="https://www.paymanai.com/">Payman - 主页</a>：未找到描述</li><li><a href="https://x.com/Ahmad_Al_Dahle/status/1779917676133105732">Ahmad Al-Dahle (@Ahmad_Al_Dahle) 的推文</a>：我很快会分享更多关于 Llama 3 的信息。看到社区已经在使用 Llama 2 构建各种东西，真是太酷了。我最喜欢的之一：@team_qanda 和 @UpstageAI 使用它构建了一个数学专用...</li><li><a href="https://x.com/suchenzang/status/1701747947191615697?s=46&t=90xQ8sGy63D2OtiaoGJuww">Susan Zhang (@suchenzang) 的推文</a>：MBPP 可能也被用在了 Phi-1.5 数据集的某些地方。就像我们截断了 GSM8K 的一个问题一样，让我们尝试截断 MBPP 的提示词，看看 Phi-1.5 会自动补全什么...</li><li><a href="https://strongcompute.com/research-grants">研究资助</a>：未找到描述</li><li><a href="https://supabase.com/blog/ai-inference-now-available-in-supabase-edge-functions">AI 推理现已在 Supabase Edge Functions 中可用</a>：通过 Supabase Edge Functions 在边缘端使用 Embeddings 和 LLM。</li><li><a href="https://x.com/OpenAIDevs/status/1780640119890047475">OpenAI Developers (@OpenAIDevs) 的推文</a>：介绍 Assistants API 的一系列更新 🧵 借助新的文件搜索工具，你可以快速集成知识检索，现在每个 Assistant 支持多达 10,000 个文件。它与我们的...</li><li><a href="https://x.com/russelljkaplan/status/1513128005828165634">Russell Kaplan (@russelljkaplan) 的推文</a>：LLM 崛起的二阶效应：</li><li><a href="https://x.com/yoheinakajima/status/1780061516051755168?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Yohei (@yoheinakajima) 的推文</a>：一个供 AI Agent 雇佣人类的平台 🧠 ↘️ 引用 tyllen (@0xTyllen) 很高兴介绍我一直在开发的一个新项目，叫做 Payman！Payman 是一个为 Agent 提供工具的 AI Agent...</li><li><a href="https://x.com/armandjoulin/status/1780638511818838378">Armand Joulin (@armandjoulin) 的推文</a>：修复了那个修复。↘️ 引用 Jonathan Frankle (@jefrankle) 为你修复了它，@code_star</li><li><a href="https://www.youtube.com/watch?v=xZiTSZ5SOYc&t=9s">Payman - 实现 AI Agent 向人类支付！</a>：大家好，在这个视频中，我非常兴奋地向大家展示 Payman，这是一个允许你将 Agent 与资金连接起来，以便它们可以用来支付给人类的平台...</li><li><a href="https://llm.extractum.io/">LLM Explorer：精选的 LLM 目录。LLM 列表。35061 个开源语言模型。</a>：浏览 35061 个开源的大型和小型语言模型，这些模型被方便地分成了各种类别和 LLM 列表，并配有基准测试和分析。
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1230058571975102464)** (1 条消息): 

- **关于 BloombergGPT 的论文俱乐部会议**：计划进行一次 **BloombergGPT** 讨论，由 `<@315351812821745669>` 主持，`<@451508585147400209>` 提供支持。提醒参与者在[这里](https://lu.ma/w7jhce1y)报名，并注意由于之前的 Discord 屏幕共享问题，会议将返回 Zoom 进行。


**提到的链接**：<a href="https://lu.ma/w7jhce1y">LLM 论文俱乐部 (BloombergGPT / TimeGPT 论文) · Zoom · Luma</a>：本周 @yikes 将介绍 BloombergGPT：https://arxiv.org/abs/2303.17564 同时请为我们的下一篇论文提交建议并投票：…

  

---


**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1230231139893903394)** (19 条消息🔥): 

- **对努力的认可**：一位成员对社区成员在组织活动中投入的时间和精力表示感谢。
- **Zoom 会议转移**：宣布讨论将从 Discord 转移到 [Zoom 会议](https://us06web.zoom.us/j/8807908941?pwd=eHBBdk9sWWluSzB2TFdLOVdEN3BFdz09)，多位成员分享了相同的链接并引导参与者前往新地点。
- **Zoom 快速提醒**：发布了进一步的通知并艾特了特定成员，提示他们加入 [Zoom 会议](https://us06web.zoom.us/j/8807908941?pwd=eHBBdk9sWWluSzB2TFdLOVdEN3BFdz09)。
- **Zoom 进入请求**：一位成员提到他们不喜欢 Zoom，但表示打算加入，并请求准许进入会议。

**提到的链接**：<a href="https://us06web.zoom.us/j/8807908941?pwd=eHBBdk9sWWluSzB2TFdLOVdEN3BFdz09">加入我们的云高清视频会议</a>：Zoom 是现代企业视频通信的领导者，拥有一个简单、可靠的云平台，可用于跨移动端、桌面端和会议室系统的视频和音频会议、聊天及网络研讨会。Zoom ...

  

---

**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1229787566002143355)** (59 条消息🔥🔥): 

- **AI 穿戴设备 vs 智能手机**：一位用户分享了 [Marquis Brownlee 的 YouTube 评测视频](https://youtu.be/TitZV6k8zfA)，并引发了关于 AI 穿戴设备相比现代智能手机局限性的讨论。对话涉及 AI 助手是否需要具备深厚的上下文知识（contextual knowledge）以实现更高效的响应。
  
- **对开源 WizardLm2 的期待**：成员们对 WizardLm2 模型表现出极大热情，称赞其摆脱了审查制度，并认为这是开源模型向 GPT-4 级别能力迈出的重要一步。讨论暗示即使在庆祝当前进步的同时，人们对下一次改进的渴望也从未停止。
  
- **翻译机器人测试与目标**：新的翻译机器人正在接受检查，其目标是通过双向翻译促进更具包容性的对话。用户对其统一讨论的潜力表示乐观。
  
- **共同寻求 Windows 兼容性**：多位用户反映在 Windows 上运行软件（特别是 01 Light 软件）时遇到困难。对话揭示了对 Windows 支持的迫切需求，以便进军企业市场，同时也讨论了在以 Mac 为导向的设置中面临的挑战。
  
- **探索硬件选项与个人 AI 愿景**：关于各种 AI 硬件选项（如 Limitless 设备）的讨论非常活跃，用户分享了个人经验，并表达了对集成化个人 AI 助手的向往。一些人强调后端基础设施和无缝集成是 AI 硬件开发的下一个前沿。

**提到的链接**：<a href="https://youtu.be/TitZV6k8zfA?t=900&si=zsI6zFfyJ8aBATzf).">我评测过的最差产品... 目前为止</a>：Humane AI pin 很糟糕。几乎没有人应该购买它。至少现在是这样。MKBHD 周边：http://shop.MKBHD.com 我目前使用的科技产品：https://www.amazon.com/shop/MKBHDIn...

  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1229696840765800461)** (17 条消息🔥): 

- **便携式 O1 设置头脑风暴**：一位成员分享了他们的目标，即使用 RPi5 运行 OI，并结合 Arduino 组件来创建一个便携式 O1 设置。其他人建议使用更简单、更便宜的组件（如 m5 atom）可能就足够了，并询问了该成员的具体设置目标。
- **O1 发货日期之谜**：针对某项未指明产品或物品的查询，一位成员提到发货目标定在夏季结束前，但具体日期尚未确认。
- **成功获取响应的终端选择**：用户讨论了他们对终端应用程序的偏好，一位成员成功使用 **Windows Terminal** 和 **Powershell** 获取了响应。有人提到在 Windows 10 的 Powershell 中识别 OpenAI key 存在困难。
- **Windows 中的批处理文件变通方案**：一位成员承认使用 **batch file** 是因为觉得更方便，这意味着它是由 cmd.exe 而非 Powershell 处理的，这突显了 Windows 的一些奇特特性。
- **最新分支的故障排除请求**：有人请求对最新分支进行测试，因为有几个人在连接建立和音频上传方面遇到了问题。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://amzn.eu/d/4GbeU5b">未找到标题</a>：未找到描述</li><li><a href="https://amzn.eu/d/fIr3Lzu">未找到标题</a>：未找到描述</li><li><a href="https://amzn.eu/d/eZQoRwD">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---



**Interconnects (Nathan Lambert) ▷ #[ideas-and-feedback](https://discord.com/channels/1179127597926469703/1179127598442348730/1229974543561658419)** (11 条消息🔥): 

_

- **胜率显著提升**：一位成员分享了项目更新，强调了一种将 **qwen-1.5-0.5B** 针对 AlpacaEval、Phi-2 和 Gemma2b-it 的胜率从 **4% 提升至 32%** 的方法。该方法结合了**分块生成**（generation in chunks）和一个小型（300M）的奖励模型（reward model）进行输出搜索。
- **寻求简单方法的验证**：该成员提到，他们在 **500M 基础模型**上通过这种简单方法实现了胜率提升，并寻求反馈以验证该方法的有效性。
- **LLM 输出重排序的相关性**：另一位社区成员承认，在推理过程中对 **LLM 输出进行重排序**（reranking）是一种已知做法，但不确定此前是否被应用于 AlpacaEval；同时还引用了一篇关于并行生成过程中重排序和剪枝的论文。
- **研究论文作为验证**：前一位成员随后提供了讨论该方法的论文链接，指出 **verifier/reward guided decoding**（验证器/奖励引导解码）等术语与该方法相关，包括 [2305.19472](https://arxiv.org/pdf/2305.19472.pdf) 和 [2402.01694](https://arxiv.org/pdf/2402.01694.pdf)。
- **尚未充分探索但前景广阔**：一位成员认同这一**尚未充分探索的领域**具有潜力，并暗示像 **MCTS PPO** 这样的概念也值得研究。

---

**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1230153917103345785)** (17 messages🔥): 

- **Mixtral-8x22B LLM 受到关注**：名为 [Mixtral 8x22B](https://mistral.ai/news/mixtral-8x22b/) 的新模型因设定了高性能和高效率标准而受到推崇。这是一个 SMoE 模型，精通多种语言，支持函数调用（function calling），并提供 64K token 的上下文窗口，全部采用 Apache 2.0 许可证。

- **关于 Mixtral-8x22B-Instruct 聊天机器人能力的讨论**：Mixtral-8x22B 的指令微调版本 [Mixtral-8x22B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1) 因其在聊天机器人领域的潜力而备受关注，其中包含有关如何运行该模型的详细指令。

- **令人印象深刻的 OLMo 1.7 7B 模型升级**：[OLMo 1.7 7B](https://huggingface.co/allenai/OLMo-1.7-7B) 引起了轰动，其 MMLU 分数提升了 24 点，该模型基于 Dolma 数据集的改进版本进行训练，并采用了阶段式训练。它是旨在推动语言模型科学发展的一系列模型之一。

- **网页质量传播提案**：有人提出了应用“网页质量”传播来对网页进行排名的想法，其中包括通过反向链接（backlinks）提高质量得分，以及通过链接到低质量网站降低得分。

- **对 Common Crawl 密集网页图谱的反思**：讨论了基于 Common Crawl 网页图谱评估“优质”内容的复杂性，指出该图谱并不能反映线性化过程（将 HTML 转换为纯文本）的成功与否。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://mistral.ai/news/mixtral-8x22b/">Cheaper, Better, Faster, Stronger</a>：继续推动 AI 前沿并使其触手可及。</li><li><a href="https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1">mistralai/Mixtral-8x22B-Instruct-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://fxtwitter.com/AlbertQJiang/status/1780648008696091003">Albert Jiang (@AlbertQJiang) 的推文</a>：我热爱开源模型！请将你最喜欢的模型添加到 Mistral Convex Hull。 ↘️ 引用 Philipp Schmid (@_philschmid)：为 @AI21Labs 修复了 Fixed Fix，并加入了 Mambas。🐍</li><li><a href="https://commoncrawl.org/web-graphs">Common Crawl - Web Graphs</a>：详细介绍了 Common Crawl 的网页图谱发布、背后的技术以及如何使用它们。</li><li><a href="https://huggingface.co/allenai/OLMo-1.7-7B">allenai/OLMo-1.7-7B · Hugging Face</a>：未找到描述</li><li><a href="https://fxtwitter.com/_philschmid/status/1780641241668997258">Philipp Schmid (@_philschmid) 的推文</a>：为 @AI21Labs 修复了 Fixed Fix，并加入了 Mambas。🐍 ↘️ 引用 Armand Joulin (@armandjoulin)：修复了 fix。
</li>
</ul>

</div>

---

**Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1230219724961353888)** (9 messages🔥):

- **Chinchilla 论文受到质疑**：[Hoffmann 等人的 Chinchilla 缩放论文](https://x.com/tamaybes/status/1780639257389904013?s=46) 正面临复现挑战，其他人在尝试复现该研究的关键部分时发现了差异。
- **对 Scaling Law 论文的怀疑**：一名成员对 Scaling Law 论文的结论表示怀疑，暗示在仔细检查 [Chinchilla 论文](https://x.com/suchenzang/status/1616752482226671620?s=46) 后发现数学方面存在问题。
- **社区对 Chinchilla 问题的讨论**：Discord 用户正在关注这一问题，分享了担忧和惊讶的简短反应，使用诸如 *"Chinchilla oops?"* 和简单的 *"oh no"* 等词句来表达对现状的不安。
- **作者未回应澄清请求**：其中一位尝试复现的人员提到，他们联系了原作者寻求澄清，但未收到任何回复，这加剧了社区内的挫败感。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/tamaybes/status/1780639279506432473?s=46">来自 Tamay Besiroglu (@tamaybes) 的推文</a>：我们已请求作者提供协助，但一直未能得到回复。(8/9)</li><li><a href="https://x.com/suchenzang/status/1616752482226671620?s=46">来自 Susan Zhang (@suchenzang) 的推文</a>：在忽略了所有这些“让我们把一堆散点拟合成一条直线”的论文细节（当你真正外推时，可能都是错误的）之后，@stephenroller 终于说服我开始研究……</li><li><a href="https://x.com/tamaybes/status/1780639257389904013?s=46">来自 Tamay Besiroglu (@tamaybes) 的推文</a>：Hoffmann 等人的 Chinchilla 缩放论文在语言建模社区极具影响力。我们尝试复现他们工作的关键部分，并发现了差异。这里是……
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/)** (1 messages): 

natolambert: 最烂的排行榜冠军，笑死 (lol)
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1229816228617523340)** (23 messages🔥): 

- **WizardLLM 代码咨询**：一位社区成员询问关于 Fork **WizardLLM** 代码的事宜；另一位成员确认模型权重已公开，暗示它可能很快回归。
- **对 olmo vs llama 3 的期待**：多位成员参与了关于 **olmo vs llama 3** 的轻松讨论，暗示可能即将迎来一场新的对决，尽管对其结果有着幽默的无奈感。
- **高产博客预测**：Nathan Lambert 暗示本周可能会有大量内容分享，**预计可能会发布三篇博客文章**。
- **讨论 Chaotic Era 中的审美变化**：**Chaotic Era** 中的对话包括对用户界面困扰的调整以及对个人资料图像的个人偏好。
- **Twitter 与迷因对话**：成员们闲聊了他们的 Twitter 活动、内容的分享性，以及由于巧合的缩写，某人的帖子可能符合 **"神圣命理学 (sacred numerology)"** 的可能性。
  

---


**Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1230181805106401361)** (3 messages): 

- **SNL 上的 AI 直播恶搞**：Nathan 分享了一个名为 "[Beavis and Butt-Head - SNL](https://www.youtube.com/watch?v=86qKgK0asGo)" 的幽默 YouTube 视频，展示了一个关于 AI 的 NewsNation 直播活动被两名观众滑稽地打断。他特别提到前一分钟非常有趣。

**提到的链接**：<a href="https://www.youtube.com/watch?v=86qKgK0asGo">Beavis and Butt-Head - SNL</a>：NewsNation 关于 AI 的直播活动被两名观众（Ryan Gosling, Mikey Day）搅乱。Saturday Night Live。现在可在 Peacock 观看：https://pck.tv/...

  

---


**Interconnects (Nathan Lambert) ▷ #[sp2024-history-of-open-alignment](https://discord.com/channels/1179127597926469703/1223784028428177510/)** (1 messages): 

natolambert: 我应该把 wizardLM 2 当作反串 (troll) 吗，笑死 (lol)
  

---



**Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1229732276162531408)** (54 messages🔥):

- **Cohere API 澄清请求**：成员们正在寻求 **Cohere API 功能的澄清**，特别是对关于 system prompts 和模型可用性方面的 API 能力感兴趣。一位用户顶起了该问题，强调需要详细信息。
- **Cohere Embeddings 基准测试咨询**：有人提问 **Cohere 的 embeddings v3 是否已与 OpenAI 的新型 large embeddings 进行了对比**。文中提供了一个指向 Cohere 博客的链接，可以在其中找到相关信息：[Introducing Command R+](https://txt.cohere.com/int8-binary-embeddings/)。
- **集成挑战与解决方案**：成员们正在处理有关集成的技术查询，特别是将 LLM 连接到 BotPress 等其他平台，并讨论了 Coral 是否需要本地托管（locally-hosted）解决方案。一位成员建议未来的更新可能会解决这个问题。
- **微调模型疑问**：一位用户询问是否可以通过 Cohere 的 Web UI **对已经微调过的模型进行再次微调**，引发了关于该流程的讨论，并分享了官方文档链接：[Fine-Tuning with the Web UI](https://docs.cohere.com/docs/fine-tuning-with-the-web-ui)。
- **Discord 欢迎与个人项目**：多位新成员介绍了自己，并分享了对 Cohere 产品的兴奋之情。讨论线程中提到了个人项目，例如使用 Cohere 的 Command R 构建的 **PaperPal**。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.]">未找到标题</a>：未找到描述</li><li><a href="https://ibb.co/s348vXt">Screenshot-2024-04-16-151544 托管于 ImgBB</a>：托管在 ImgBB 的图片 Screenshot-2024-04-16-151544</li><li><a href="https://docs.cohere.com/docs/fine-tuning-with-the-web-ui">使用 Web UI 进行微调 - Cohere 文档</a>：未找到描述</li><li><a href="https://txt.cohere.com/int8-binary-embeddings/">Cohere int8 和二进制 Embeddings - 将您的向量数据库扩展到大型数据集</a>：Cohere Embed 现在原生支持 int8 和二进制 embeddings，以降低内存成本。</li><li><a href="https://github.com/Unstructured-IO/unstructured">GitHub - Unstructured-IO/unstructured：用于构建自定义预处理流水线的开源库和 API，适用于标注、训练或生产机器学习流水线。</a>：用于构建自定义预处理流水线的开源库和 API，适用于标注、训练或生产机器学习流水线。 - GitHub - Unstructured-IO/unstructured...</li><li><a href="https://github.com/cohere-ai/sandbox-conversant-lib">GitHub - cohere-ai/sandbox-conversant-lib：基于 Cohere LLM 构建的对话式 AI 工具和角色（personas）</a>：基于 Cohere LLM 构建的对话式 AI 工具和角色 - cohere-ai/sandbox-conversant-lib</li><li><a href="https://github.com/cohere-ai/quick-start-connectors">GitHub - cohere-ai/quick-start-connectors：该开源仓库提供了将工作场所数据存储与 Cohere LLM 集成的参考代码，使开发者和企业能够在其自有数据上执行无缝的检索增强生成 (RAG)。</a>：该开源仓库提供了将工作场所数据存储与 Cohere LLM 集成的参考代码，使开发者和企业能够执行无缝的检索增强生成...
</li>
</ul>

</div>
  

---


**Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1230065514924216320)** (3 条消息): 

- **Quant Fino 招募 Beta 测试人员**：一个由 **Command-R Plus** 驱动的 Agent 实体已部署新试点，旨在将 GAI 与 FinTech 和日内交易（Day Trading）相结合。他们目前正在寻求 **beta 测试人员**和反馈，相关信息可在 [Join Beta - Quant Fino](https://quantfino.com/join-beta) 找到，包含 Cookie 政策和用户同意详情。

- **关于 Rubik API 的咨询**：一位成员表示有兴趣通过支持 **post request** 的 API 使用 Rubik。他们正在等待关于此类 API 是否可用的进一步细节。

- **红队测试揭示 Command R+ 的漏洞**：一位成员对 **Command R+** 模型进行了红队测试（redteaming），发现其有可能创建具有执行恶意任务能力的**无限制 Agent**。他们在 [LessWrong](https://www.lesswrong.com/posts/4vPZgvhmBkTikYikA/creating-unrestricted-ai-agents-with-command-r) 上提供了一份详细报告，其中包括 Agent 生成的针对有害行为的消息示例。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.lesswrong.com/posts/4vPZgvhmBkTikYikA/creating-unrestricted-ai-agents-with-command-r">使用 Command R+ 创建不受限的 AI Agents — LessWrong</a>：简而言之，目前存在一些能力强大的开源权重模型，可用于创建简单且不受限的恶意 Agent。它们可以端到端地执行任务...</li><li><a href="https://quantfino.com/join-beta">Quantfino - 强大的 AI 驱动金融之源</a>：Quantfino 是由 LLM 驱动并辅以 Langchain 的金融分析平台。
</li>
</ul>

</div>
  

---



**LangChain AI ▷ #[announcements](https://discord.com/channels/1038097195422978059/1058033358799655042/1229820483818623010)** (1 条消息): 

- **迭代文档结构改进**：团队正在迭代 **documentation structure**（文档结构）以增强易访问性和清晰度。提议采用一种将内容拆分为“教程”、“操作指南”和“概念指南”的新组织方式，并通过提供的链接征求对该结构的反馈。

- **LangChain 框架介绍亮点**：提供的链接介绍了 **LangChain**，这是一个用于构建大语言模型应用程序的开源框架。它详细说明了 LangChain 如何通过 [building blocks](https://langchain-git-harrison-new-docs-langchain.vercel.app/docs/expression_language/)、[LangSmith](https://langchain-git-harrison-new-docs-langchain.vercel.app/docs/langsmith/) 和 [LangServe](https://langchain-git-harrison-new-docs-langchain.vercel.app/docs/langserve/) 促进开发、生产化和部署，并包含图表概览。

**提到的链接**：<a href="https://langchain-git-harrison-new-docs-langchain.vercel.app/docs/get_started/introduction">介绍 | 🦜️🔗 LangChain</a>：LangChain 是一个用于开发由大语言模型 (LLMs) 驱动的应用程序的框架。

  

---


**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1229689230977011824)** (38 条消息🔥): 

- **寻求 YC 初创公司见解**：一位成员表示有兴趣向 **YC** 申请一个专注于为 Agent 微调模型的初创项目，并询问是否有人知道这是否已经有人在做。另一位成员列举了该领域的公司，如 **Unsloth、Mistral AI** 和 **Lumini**。

- **征集 LLM 应用协作**：公开征集正在开发 **LLM applications** 的人员参与简短对话，一名成员立即表示愿意参与。

- **Langchain 学习曲线**：关于学习 **Langchain** 是否值得的询问得到了轻松的回应，建议通过实践来学习，并鼓励对该技术进行动手实验。

- **Langchain 处理表格数据的更新**：多位用户讨论了如何使用 **Langchain** 为聊天机器人处理多个 CSV 文件，建议包括使用 SQL Agent 以及利用 CSV 文件和有效处理大型数据集的不同方法。

- **探索 RAG 优化**：用户提出了使用 **RAG** 处理大型文档的挑战，讨论了索引前或索引后拆分等策略，一位成员分享了他们为提高准确性而优化 RAG 的追求。

- **寻找招聘负责人**：一位新参与者在频道中打招呼，并正在寻找讨论 **hiring**（招聘）事宜的合适联系人。

- **涉足多 Agent 框架**：一位成员指向了 **AutoGen**，这是由 Microsoft 提供的用于多 Agent 对话和工作流的框架，并引发了用户对 **Langchain** 内部多 Agent 编排的好奇。

- **AI 初创公司融资数据库发布**：分享了一个全面的 **AI startups** 融资数据库，其中包含关于融资轮次和公司的令人印象深刻的数据收集，包括来自 **GPT-4** 的见解，并邀请对可能的数据不准确之处提供反馈。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://microsoft.github.io/autogen/">AutoGen | AutoGen</a>：通过多 Agent 对话框架实现下一代 LLM 应用</li><li><a href="https://flashcardfy.lol/">Flashcardfy - 带有个性化反馈的 AI 闪存卡生成器</a>：通过提供个性化反馈的 AI 生成闪存卡，学习得更快、更聪明。</li><li><a href="https://js.langchain.com/docs/use_cases/sql/agents">Agents | 🦜️🔗 Langchain</a>：LangChain 提供了许多工具和函数，允许你创建 SQL Agents，从而提供一种更灵活的与 SQL 数据库交互的方式。使用 SQL Agents 的主要优点是...
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1229808687632089131)** (1 条消息):

- **LangServe 与 Nemo Guardrails 的集成挑战**：一位成员询问了在尝试将 **LangServe** 与包含 **Nemo Guardrails** 的链（chain）集成时遇到的困难，因为 Nemo 会显著改变输出结构。他们提到需要一种新型的输出解析器（output parser）来处理这些变化。
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1229815994789265489)** (4 messages): 

- **Galaxy AI 推出多种免费 API**：GalaxyAI 发布了一项**免费** API 服务，允许以 OpenAI 格式访问 **GPT-4**、**GPT-3.5-turbo** 等高级 AI 模型及 Langchain 集成。查看他们的产品并将其集成到你的项目中：[Galaxy AI](https://galaxyapi.onrender.com)。

- **OppyDev 发布 AI 驱动的编程工具**：OppyDev 发布了一个 AI 辅助编程平台，该平台将 IDE 与聊天客户端相结合，具有易用性、透明度、定制化、数据控制等特点，并使用 **GPT-4** 和 **Claude** 等 LLM。查看演示并了解更多信息：[OppyDev AI](https://oppydev.ai)。

- **Rubiks.ai 为高级研究助手招募 Beta 测试人员**：新型高级研究助手和搜索引擎 Rubiks.ai 正在寻求 Beta 测试人员，以试用包括 **Claude 3 Opus**、**GPT-4 Turbo** 以及由 Groq 服务器驱动以实现快速响应的 **Mistral Large** 等功能。感兴趣的用户可以访问 [Rubiks.ai](https://rubiks.ai) 注册，使用促销代码 `RUBIX` 可获得 2 个月的免费高级访问权限。

- **揭秘多步工具的力量**：一篇文章讨论了将多步工具（multi-step tools）与 **LangChain** 和 **Cohere** 集成以提高效率的益处。在 [AI Advances](https://medium.com/ai-advances/unlocking-efficiency-the-power-of-multi-step-tools-with-langchain-and-cohere-7d1ea571ebed) 的全文中阅读有关此进展的更多信息。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://galaxyapi.onrender.com">Galaxy AI - Swagger UI</a>：未找到描述</li><li><a href="https://oppydev.ai">Home - OppyDev</a>：提升编程体验的协作式 AI Agent</li><li><a href="https://rubiks.ai">Rubik's AI - AI 研究助手 & 搜索引擎</a>：未找到描述
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1229725236593692722)** (5 messages): 

- **寻求合作**：一位参与者表达了加入项目的兴趣，并请求通过私信讨论更多细节。
- **关于具有长期记忆的 AI Agents 教程**：一位成员分享了一个 [YouTube 视频](https://youtu.be/7LWTZqksmSg)，解释了如何赋予 AI Agents 长期记忆和自我改进能力，提供了对高级 AI Agent 开发的见解。
- **关于 Langgraph 使用的疑问**：针对分享的关于 AI Agent 长期记忆的视频，一位成员询问为什么没有考虑使用 **Langgraph** 概念来实现。

**提及的链接**：<a href="https://youtu.be/7LWTZqksmSg?si=_tnJhoUcQr4Gojek">释放 AI Agent 的真正力量？！长期记忆与自我改进</a>：如何为你的 AI Agent 构建长期记忆和自我改进能力？免费使用 AI 幻灯片生成器 Gamma：https://gamma.app/?utm_source=youtube&amp;utm...

  

---



**DiscoResearch ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/1230179344241000540)** (12 messages🔥): 

- **挑战 GPU 显存极限**：Maxidl 报告称，在使用多达 64 个 80GB GPU 时，成功实现了**全规模 deep-speed** (FSDP)、32k 序列长度和 batch size 为 1 的运行，每个 GPU 的显存占用接近极限，达到 77GB。
- **64 个 GPU 并非笔误**：在受到询问时，maxidl 确认使用了 64 个 GPU，并指出减少到 32 个 GPU 会导致显存溢出 (OOM) 错误，因此必须使用更多的 GPU 数量。
- **探索优化可能性**：考虑到显存限制，maxidl 提到了在训练期间使用 **8-bit 优化**来节省显存的潜力。
- **显存使用优化建议**：_jp1_ 建议使用 `fsdp_transformer_layer_cls_to_wrap: MixtralSparseMoeBlock` 并启用 `offload_params = true` 以改进显存使用，预计这样可以适配 32 个 GPU 的 VRAM。
- **寻找显存需求计算器**：Maxidl 询问了根据模型大小和序列长度计算模型激活值（activations）显存占用的工具，并引用了关于 Mixtral 模型内存需求的 [**HuggingFace 讨论**](https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1/discussions/12)。

**提及的链接**：<a href="https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1/discussions/12">mistral-community/Mixtral-8x22B-v0.1 · [自动生成] 模型显存需求</a>：未找到描述

  

---

**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1229695493416681523)** (8 messages🔥): 

- **文本抓取的灰色地带**：一位成员发表观点认为，从欧盟版权的角度来看，大多数抓取的文本数据至少处于灰色地带。他们还提到来自 **DFKI** 的文本可能很有用，但手头没有链接。
  
- **寻找多模态数据**：一位成员建议了具有宽松许可的多模态数据源，如 **Wikicommons** 以及 [Creative Commons Search](https://search.creativecommons.org/) 上列出的其他平台。

- **Llama Tokenizer 简化版**：有人分享了一个 [Google Colab notebook](https://colab.research.google.com/drive/1Ica34BAGK2tuIeQl01SRNTjujPq5C3d1?usp=sharing)，演示了如何不依赖 HuggingFace，而是使用 sentencepiece 来创建 Llama tokenizer。

- **关于 Tokenizer 拼写的疑问**：在讨论自定义 tokenizer 之后，一位成员指出分享的 tokenizer 中存在拼写错误，将 **Muad'Dib** 拼错了。

- **现代化 Tokenization 技术**：一位贡献者强调 **Mistral** 已经发布了他们的 tokenization 库，这可能有助于在没有自定义 wrappers 的情况下实现标准化的 finetuning 流程，并提供了 [GitHub 上示例 notebook 的链接](https://github.com/mistralai/mistral-common/blob/main/examples/tokenizer.ipynb)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://search.creativecommons.org/">CC Search Portal</a>：无描述</li><li><a href="https://colab.research.google.com/drive/1Ica34BAGK2tuIeQl01SRNTjujPq5C3d1?usp=sharing">Google Colaboratory</a>：无描述</li><li><a href="https://github.com/mistralai/mistral-common/blob/main/examples/tokenizer.ipynb">mistral-common/examples/tokenizer.ipynb at main · mistralai/mistral-common</a>：通过在 GitHub 上创建账号来为 mistralai/mistral-common 的开发做出贡献。
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/1229701118842503208)** (1 messages): 

- **语言模型解码策略分析**：一位成员引用了一篇名为《[A Thorough Examination of Decoding Methods in the Era of LLMs](https://arxiv.org/abs/2402.06925)》的论文，并担心该论文没有涵盖与其 LLM 使用经验相关的开放式任务。他们还提到，由 *u/kindacognizant* 提出的现代采样方法（如 **MinP/DynaTemp/Quadratic Sampling**）并未包含在此类论文中。
- **min_p 采样对创意写作的惊人影响**：同一位成员分享了一个 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1c36ieb/comparing_sampling_techniques_for_creative/)，详细比较了 **min_p 采样参数** 及其对创意写作性能的显著影响。比较显示，在 **alpaca-eval style elo** 中提升了 +8 分，在 **eq-bench 创意写作测试** 中提升了 +10 分。

**提到的链接**：<a href="https://www.reddit.com/r/LocalLLaMA/comments/1c36ieb/comparing_sampling_techniques_for_creative/">Reddit - Dive into anything</a>：无描述

  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1229891842754089082)** (9 messages🔥): 

- **Tinygrad 和 INT8 支持咨询**：一位成员询问 tinygrad 是否支持 int8 计算，另一位成员给出了肯定的回答。具体定义位置未提供。
- **硬件在定义 Tinygrad 计算中的作用**：一位用户提到，tinygrad 是否支持某些数据类型（如 int8）通常由 **硬件能力** 决定，而非 tinygrad 本身。
- **Tinygrad 的增强型图可视化**：有人咨询 tinygrad 中改进的图可视化功能，得到的回复指向了 [Tiny-tools Graph Visualization](https://tiny-tools-client.vercel.app/)，它比 `GRAPH=1` 提供的图表更精美。
- **对优化的 Node.equals() 的兴趣**：一位成员表示，对于 tinygrad 来说，一个快速且概率完备的 **Node.equals()** 函数将是一个很酷的补充。
- **Pytorch-Lightning 硬件无关性讨论**：讨论了 Pytorch-Lightning 的硬件无关特性，并提供了其 GitHub 仓库链接，另一位成员确认其在 **7900xtx** 上可以使用。[在 GitHub 上查看 Pytorch-Lightning](https://github.com/Lightning-AI/pytorch-lightning)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tiny-tools-client.vercel.app/">React App</a>: 未找到描述</li><li><a href="https://github.com/Lightning-AI/pytorch-lightning">GitHub - Lightning-AI/pytorch-lightning: Pretrain, finetune and deploy AI models on multiple GPUs, TPUs with zero code changes.</a>: 在多个 GPU、TPU 上预训练、微调和部署 AI 模型，无需更改代码。 - Lightning-AI/pytorch-lightning
</li>
</ul>

</div>
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1229757044148670504)** (9 messages🔥): 

- **探索 Metal Compute Shaders**：一位成员正在实验 **tinygrad 生成的 Metal compute shaders**，并有兴趣学习如何在不使用 Xcode 的情况下运行基础的 Metal compute shader 程序。另一位成员建议咨询 ChatGPT 获取用于分发向量加法 Metal shader 代码的 Python 脚本，并提到了他们良好的学习体验。

- **ONNX 转 WebGL/WebGPU 的可能性**：有人询问关于使用 tinygrad 将模型从 ONNX 转换为 WebGL/WebGPU 的问题，特别是为了在 Web 上运行 **meshnet 模型**。虽然提到了 [Stable Diffusion WebGPU 示例](https://github.com/softwiredtech/stable-diffusion-webgpu) 作为对比，但该成员正在寻求直接从 ONNX 进行转换的建议。

- **Tinygrad 中的层设备分配查询**：一位参与者担心 tinygrad 似乎缺乏在设备间移动层（如 Linear, Conv2d）的功能。**George Hotz** 澄清说，在模型上调用 **get parameters** 后，可以使用 `to_` 方法移动模型参数。

- **Tinygrad 中的零成本张量操作**：一位用户寻求在 tinygrad 中实现 broadcast、reshape 和 permute 操作且不产生数据复制成本的指导。他们被引导查看 *tinygrad/shape/shapetracker.py* 或 *view.py* 以获取相关的代码示例。
  

---



**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1229691075690631250)** (4 messages): 

- **介绍 Idefics2**：[Hugging Face 推出了一款名为 Idefics2 的新型多模态 ChatGPT](https://www.youtube.com/watch?v=vL1SayPCHBg)，它将 Python 编程融入到了其能力中。
- **Reka Core 挑战巨头**：[Reka Core 语言模型](https://www.youtube.com/watch?v=U7RbwPKyxs8) 被展示为可与 OpenAI、Anthropic 和 Google 的模型竞争，并宣传了令人印象深刻的性能指标。
- **JetMoE：高性价比的 AI 性能**：凭借不到 10 万美元的支出，[JetMoE-8B 声称其性能优于](https://www.youtube.com/watch?v=Z9Hwp_XeS1A) Meta AI 的 LLaMA2-7B，后者是一个拥有巨额资金支持的模型。
- **Snowflake 的新型文本嵌入模型**：Snowflake 发布并开源了其 Snowflake Arctic embed 系列模型，被强调为世界上最好的[实用文本嵌入模型](https://www.youtube.com/watch?v=p9T7ZgtM5Mo)。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=U7RbwPKyxs8">Reka Core: A Frontier Class Multimodal Language Model</a>: Reka Core 在关键的行业公认评估指标上可与 OpenAI、Anthropic 和 Google 的模型竞争。鉴于其占用空间和性能...</li><li><a href="https://www.youtube.com/watch?v=vL1SayPCHBg">Introducing Idefics2 8B: Open Multimodal ChatGPT</a>: 我们将看看 Hugging Face 的开源多模态 LLM idefics2... #python #pythonprogramming #llm #ml #ai #aritificialin...</li><li><a href="https://www.youtube.com/watch?v=p9T7ZgtM5Mo">Snowflake Launches the World’s Best Practical Text-Embedding Model</a>: 今天 Snowflake 发布并以 Apache 2.0 许可证开源了 Snowflake Arctic embed 系列模型。基于 Massive Text Embedding Be...</li><li><a href="https://www.youtube.com/watch?v=Z9Hwp_XeS1A">JetMoE: Reaching LLaMA2 Performance with 0.1M Dollars</a>: JetMoE-8B 的训练成本不到 10 万美元，但性能优于来自 Meta AI 的 LLaMA2-7B，后者拥有数十亿美元的训练资源。LLM 训练...
</li>
</ul>

</div>
  

---



**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1230226260366721116)** (3 messages): 

- **对 Mixtral 8x22B Instruct 的期待**：表达了对通过 llm 尝试 **Mixtral 8x22B Instruct** 的兴奋，并提供了其在 [HuggingFace 上的 Model Card](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1) 链接作为参考。 
- **报告 llm-gpt4all 的问题**：一位用户提到在安装 **llm-gpt4all** 时遇到错误；该问题的详细信息已在 GitHub 上发布，并附带了[错误报告](https://github.com/simonw/llm-gpt4all/issues/28)的链接。
<div class="linksMentioned">

<strong>提到的链接</strong>:

</div>

<ul>
<li>
<a href="https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1">mistralai/Mixtral-8x22B-Instruct-v0.1 · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/simonw/llm-gpt4all/issues/28">adding the llm-gpt4all models breaks the python app. · Issue #28 · simonw/llm-gpt4all</a>: 我安装 LLM 没问题，分配了我的 OpenAI 密钥，并且可以毫无问题地与 GPT-4 对话，查看我的 LLM 模型命令输出：OpenAI Chat: gpt-3.5-turbo (别名: 3.5, chatgpt) OpenAI...
</li>
</ul>

</div>
  

---



**Alignment Lab AI ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/1229810947829403648)** (2 条消息): 

- **律师介入**：一名成员发表简短评论，暗示律师可能介入了某种情况，但未提供法律影响的具体背景。
- **插图说明 wizardlm-2 被删除**：分享了一张图片，描述 **wizardlm-2** 因缺乏对 **v0** 的测试而被删除；然而，消息中并未给出 **wizardlm-2** 的具体定义或涉及的测试细节。[查看图片](https://cdn.discordapp.com/attachments/1019530324255965186/1229693872997666816/wizardlm-2-was-deleted-because-they-forgot-to-test-it-for-v0-lyaop5lw0suc1.png?ex=66309ca9&is=661e27a9&hm=f105e6497796be9c414ade2024a27f9561caf0cad6cb06ba09f80e30b5e39ae4&)
  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1230131674290065490)** (2 条消息): 

- **Llamafile 脚本改进**：Llamafile 归档版本升级重打包脚本已改进，可在[此 Gist](https://gist.github.com/mofosyne/46c63934305d5a5321c7e9fd83f4ef3e)获取。关于是否将其集成到 Llamafile 的 GitHub 主仓库中存在争议，原因是出于维护方面的考虑，有人认为维护者应该从头开始创建新的 Llamafile。

- **安全漏洞报告流程查询**：有人询问了报告安全漏洞的程序以及随后申请 CVE (Common Vulnerabilities and Exposures) 标识符的要求。消息中未提供额外的背景或说明。
  

---



---