---
companies:
- apple
- google
- mistral-ai
- microsoft
- mosaic
date: '2024-06-11T06:41:05.024816Z'
description: '**Apple Intelligence** 引入了一个小型（约 30 亿参数）的端侧模型，以及一个在搭载 Apple 芯片的私有云计算（Private
  Cloud Compute）上运行的更大型服务器模型，旨在超越 **Google Gemma**、**Mistral Mixtral**、**Microsoft
  Phi** 和 **Mosaic DBRX**。


  该端侧模型采用了一种新颖的无损量化策略，通过混合使用 2 位和 4 位的 LoRA 适配器（平均每个权重为 3.5 位），实现了适配器的动态热插拔和高效的内存管理。苹果将量化和模型延迟的优化归功于
  **Talaria** 工具，在 iPhone 15 Pro 上实现了约 0.6 毫秒的首个 Token 延迟（TTFT）和每秒 30 个 Token 的生成速率。


  苹果专注于“适配器适配一切”（adapter for everything）的策略，并初步部署在 SiriKit 和 App Intents 中。性能基准测试主要依赖人工评分，强调满足消费者层面的实际需求，而非追求学术指标上的统治地位。此外，Apple
  ML 博客还提到了一个专注于 Xcode 代码的模型，以及一个用于生成 Genmoji 的扩散模型。'
id: 8b2bbd94-300e-4a69-b1ad-4d147b7bbec3
models:
- gemma
- mixtral
- phi
- dbrx
original_slug: ainews-talaria-apples-new-mlops-superweapon-4066
people:
- craig-federighi
- andrej-karpathy
title: Talaria：苹果的新型 MLOps 超级武器
topics:
- quantization
- on-device-ai
- adapter-models
- model-optimization
- model-latency
- lossless-quantization
- low-bit-palletization
- token-generation
- model-benchmarking
- human-evaluation
---

<!-- buttondown-editor-mode: plaintext -->**Apple Intelligence 就够了。**

> 2024年6月7日至6月10日的 AI 新闻。我们为您查看了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord 服务（**411** 个频道和 **7641** 条消息）。为您节省的预计阅读时间（以 200wpm 计算）：**816 分钟**。

凭借 [Apple Intelligence](https://www.apple.com/newsroom/2024/06/introducing-apple-intelligence-for-iphone-ipad-and-mac/)，Apple 声称一举超越了 Google Gemma、Mistral Mixtral、Microsoft Phi 和 Mosaic DBRX。这得益于一个小型 “Apple On-Device” 模型（约 3b 参数）和一个“更大”的 Apple Server 模型（可通过运行在 Apple Silicon 上的 [Private Cloud Compute](https://security.apple.com/blog/private-cloud-compute/) 使用）。

https://www.youtube.com/watch?v=Q_EYoV1kZWk

[Apple ML 博客文章](https://machinelearning.apple.com/research/introducing-apple-foundation-models) 还简要提到了另外两个模型——一个专注于 Xcode 代码的模型，以及一个用于 Genmoji 的 Diffusion 模型。

似乎被低估的是端侧模型热插拔 LoRA 的能力，以及其**显然无损的量化**策略：

> 对于端侧推理，我们使用了 low-bit palletization，这是一种实现必要内存、功耗和性能要求的关键优化技术。为了保持模型质量，**我们开发了一个使用 LoRA 适配器的新框架，该框架结合了 2-bit 和 4-bit 的混合配置策略——平均每个权重 3.5 bits-per-weight——以达到与未压缩模型相同的准确度。**
> 
> 此外，我们使用交互式模型延迟和功耗分析工具 Talaria，以更好地指导每个操作的比特率选择。我们还利用了 activation quantization 和 embedding quantization，并开发了一种在我们的 Neural Engine 上实现高效 Key-Value (KV) cache 更新的方法。
> 
> 通过这一系列优化，**在 iPhone 15 Pro 上，我们能够达到每个 prompt token 约 0.6 毫秒的 time-to-first-token 延迟，以及每秒 30 个 tokens 的生成速率。** 值得注意的是，这一性能是在采用 token speculation 技术之前实现的，通过该技术我们可以看到 token 生成速率的进一步提升。
> 
> 我们使用 16 bits 表示适配器参数的值，对于约 30 亿参数的端侧模型，rank 16 的适配器参数通常需要数十 MB。**适配器模型可以动态加载、临时缓存在内存中并进行切换**——这使得我们的基础模型能够针对当前任务即时进行专门化，同时高效管理内存并保证操作系统的响应速度。

他们将这种令人惊叹的端侧推理归功于关键工具 [Talaria](https://machinelearning.apple.com/research/talaria)：

 
![image.png](https://assets.buttondown.email/images/3da01ba6-e217-4d24-bd5c-43d84982c6b3.png?w=960&fit=max)
 

Talaria 有助于在预算限制下消融量化并分析模型架构：

 
![image.png](https://assets.buttondown.email/images/63f0a6dc-7388-412a-9d00-9222d8cb7316.png?w=960&fit=max)
 

 
![image.png](https://assets.buttondown.email/images/f01e9b2f-0d1a-4f70-99cf-8621c5ea7425.png?w=960&fit=max)
 

Apple 似乎并没有追求“万能模型（God Model）”，而是采取了“适配器适配一切”的策略，而 Talaria 旨在让快速迭代和跟踪单个架构的性能变得容易。这就是为什么 Craig Federighi 宣布 Apple Intelligence 在启动初期仅专门适用于 SiriKit 的 8 个特定适配器和 12 类 App Intents：
 
![image.png](https://assets.buttondown.email/images/518032f7-20cc-47ae-a188-211dbdad4e58.png?w=960&fit=max)
 
 
![image.png](https://assets.buttondown.email/images/87f3022a-4d51-4df3-8a6f-10997e6088e1.png?w=960&fit=max)
 

考虑到 Apple 是针对严格的推理预算进行设计的，观察 Apple 如何自报性能也很有趣。几乎所有的结果（除了指令遵循）都是由人类评分员完成的，其优点是作为金标准，但缺点是最不透明：

 
![image.png](https://assets.buttondown.email/images/27af4622-10b9-4c07-830c-649c9fb11a00.png?w=960&fit=max)
 

这些声称击败了 Google/Microsoft/Mistral/Mosaic 的基准测试，其唯一可信度来源在于：Apple 不需要在学术领域获胜——它只需要对消费者来说“足够好”就能赢。在这里，它只需要击败 2011-2023 年间 Siri 的低标准即可。





---

{% if medium == 'web' %}

**目录**

[TOC]

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}

---

# AI Twitter 摘要

> 所有摘要均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在尝试使用 Haiku 进行聚类和流程工程。

**Andrej Karpathy 关于复现 GPT-2 (124M) 的新 YouTube 视频**

- **全面的 4 小时视频讲座**：[@karpathy](https://twitter.com/karpathy/status/1799949853289804266) 发布了一个名为 "Let's reproduce GPT-2 (124M)" 的新 YouTube 视频，涵盖了构建 GPT-2 网络、优化快速训练、设置训练运行以及评估模型。该视频基于 Zero To Hero 系列。
- **详细的演练**：视频分为多个部分，包括探索 GPT-2 checkpoint、实现 GPT-2 nn.Module、使用 mixed precision 和 flash attention 等技术加速训练、设置超参数以及评估结果。**该模型的性能接近 GPT-3 (124M)**。
- **关联的 GitHub 仓库**：[@karpathy](https://twitter.com/karpathy/status/1799949853289804266) 提到关联的 GitHub 仓库包含完整的 commit 历史，以便逐步跟随代码更改。

**Apple 的 WWDC AI 发布**

- **缺乏令人印象深刻的 AI 发布**：[@karpathy](https://twitter.com/karpathy/status/1800223553989886447) 指出，在 Apple 的 WWDC 进行 50 分钟后，没有出现令人印象深刻的重大 AI 发布。
- **关于 "Apple Intelligence" 和 OpenAI 合作的传闻**：[@adcock_brett](https://twitter.com/adcock_brett/status/1799834749004906581) 提到了 Apple 将推出名为 "Apple Intelligence" 的新 AI 系统以及与 OpenAI 潜在合作的传闻，但这些在 WWDC 上未得到确认。

**矩阵乘法的直观解释**

- **关于矩阵乘法的 Twitter 推文串**：[@svpino](https://twitter.com/svpino/status/1800151091461652740) 分享了一个 Twitter 推文串，对矩阵乘法进行了极佳且简单的解释，称其为现代 Machine Learning 背后最关键的思想。
- **逐步分解**：该推文串分解了矩阵 A 和 B 乘积的原始定义，通过可视化逐步展开，以提供对矩阵乘法运作方式及其几何解释的直观理解。

**Apple 的 Ferret-UI：适用于 iOS 的多模态视觉语言模型**

- **Ferret-UI 论文详情**：[@DrJimFan](https://twitter.com/DrJimFan/status/1800199288783618049) 重点介绍了 Apple 关于 Ferret-UI 的论文，这是一种多模态视觉语言模型 (Multimodal Vision-Language Model)，可以理解 iOS 移动屏幕上的图标、组件和文本，并对其空间关系和功能含义进行推理。
- **端侧 AI 助手的潜力**：论文讨论了数据集和基准测试的构建，展示了 Apple 非凡的开放性。**凭借强大的屏幕理解能力，Ferret-UI 可以扩展为功能齐全的端侧助手**。

**AI 投资与进展**

- **自 GPT-4 以来在 NVIDIA GPU 上花费了 1000 亿美元**：[@alexandr_wang](https://twitter.com/alexandr_wang/status/1799930061631729887) 指出，自 2022 年秋季训练 GPT-4 以来，各方在 NVIDIA GPU 上总计花费了约 1000 亿美元。问题在于下一代 AI 模型的能力是否能达到这一投资水平。
- **撞上数据墙**：Wang 讨论了由于数据墙 (data wall) 导致 AI 进展放缓的可能性，这需要数据丰富的方法、算法进步以及扩展到现有互联网数据之外。**行业对于这会是短期障碍还是实质性的平台期存在分歧**。

**Perplexity 成为出版商的顶级引流来源**

- **Perplexity 为出版商带来流量**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1800210005498728531) 分享道，Perplexity 已成为 Forbes 的第二大引流来源（仅次于 Wikipedia），并且是其他出版商的第一大引流来源。
- **即将推出的出版商参与产品**：Srinivas 提到 Perplexity 正在开发新的出版商参与产品，以及与媒体公司对齐长期激励措施的方法，即将发布。

**Yann LeCun 关于管理 AI 研究实验室的想法**

- **声誉卓著的科学家在管理层中的重要性**：[@ylecun](https://twitter.com/ylecun/status/1799876279363137615) 强调，研究实验室的管理层应由声誉卓著的科学家组成，以识别和留住杰出人才，提供资源和自由，确定有前景的研究方向，识别虚假信息 (BS)，激发宏伟目标，并超越简单指标 (metrics) 来评估人才。
- **培养智力上的奇思异想**：LeCun 指出，管理研究实验室需要包容智力上的奇思异想 (intellectual weirdness)，这可能伴随着书呆子式的性格古怪，这使得管理更加困难，因为真正有创造力的人不会落入可预测的条条框框中。

**推理能力 vs. 事实存储与检索**

- **区分推理与记忆**：[@ylecun](https://twitter.com/ylecun/status/1799869604702859336) 指出，推理能力和常识不应与存储和近似检索大量事实的能力相混淆。

---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**AI 模型开发与基准测试**

- **文本转视频模型改进**：在 /r/singularity 中，新款中国文本转视频模型 Kling 展示了相比以往模型[**在一年时间内的显著进步**](https://v.redd.it/33itvj94wo5d1)。/r/singularity 的其他讨论推测了[再过一年可能会出现哪些进一步的突破](https://www.reddit.com/r/singularity/comments/1dc0gop/if_this_is_one_year_later_whats_one_year_from_now/)。
- **AI 对数学的影响**：在 /r/singularity 中，菲尔兹奖得主陶哲轩（Terence Tao）认为 [**AI 将成为数学家的“副驾驶（co-pilot）”**](https://www.scientificamerican.com/article/ai-will-become-mathematicians-co-pilot/)，从而彻底改变数学领域。
- **强大的 zero-shot 预测模型**：在 /r/MachineLearning 中，讨论了 IBM 的开源 [Tiny Time Mixers (TTMs)](https://aihorizonforecast.substack.com/p/tiny-time-mixersttms-powerful-zerofew?post_page-reml--)，认为它们是强大的 zero-shot 预测模型。

**AI 应用与工具**

- **去中心化 AI 模型追踪器**：在 /r/LocalLLaMA 中，[AiTracker.art](https://aitracker.art/)（一个 AI 模型的 Torrent 追踪器）被作为 Huggingface 和 Civitai 的去中心化替代方案展出。
- **LLM 驱动的压缩技术**：在 /r/LocalLLaMA 中，讨论了 [Llama-Zip](https://github.com/AlexBuz/llama-zip)，这是一款由 LLM 驱动的压缩工具，其潜力在于能从压缩后的 key 中恢复完整的训练文章。
- **快速的浏览器端语音识别**：在 /r/singularity 中，[Whisper WebGPU](https://v.redd.it/ujdlsc1m5l5d1) 展示了直接在浏览器中运行的、极速的 ML 驱动语音识别。
- **用本地模型替换 OpenAI**：在 /r/singularity 中，一个帖子演示了如何仅用 1 行 Python 代码[通过 llama.cpp 服务器替换 OpenAI](https://x.com/cocktailpeanut/status/1799894007314628957)。
- **国际象棋棋局的语义搜索**：在 /r/LocalLLaMA 中，分享了一个[国际象棋棋局的 embeddings 模型](https://github.com/broskicodes/chess-position-embeddings/tree/master?tab=readme-ov-file)，实现了语义搜索功能。

**AI 安全与监管**

- **Prompt injection（提示词注入）威胁**：在 /r/OpenAI 中，讨论了针对 LLM 应用的 [prompt injection 威胁及防护方法](https://huggingface.co/datasets/deepset/prompt-injections)，例如训练自定义分类器来防御恶意提示词。
- **模型中敏感数据的担忧**：在 /r/singularity 中，一个帖子认为，随着科技公司在互联网上抓取数据，公开模型曾在**绝密文件（TOP SECRET documents）**上进行过训练的概率可能[**超过 99%**](https://www.reddit.com/r/singularity/comments/1dby6gf/with_tech_companies_scraping_the_internet_for/)。
- **减少模型拒绝响应的技术**：在 /r/LocalLLaMA 中，[正交激活引导（Orthogonal Activation Steering, OAS）和“消融（abliteration）”](https://huggingface.co/grimjim/Llama-3-Oasis-v1-OAS-8B)被指出是减少 AI 模型拒绝执行某些提示词的同一种技术。

**AI 伦理与社会影响**

- **教育中的 AI**：在 /r/singularity 中，讨论了 [AI 在教育场景中的应用](https://www.reddit.com/r/singularity/comments/1dc16s1/thoughts_on_ai_being_used_in_educational_settings/)，引发了关于有效整合以及学生可能滥用的疑问。

**AI 硬件与基础设施**

- **大型模型基准测试**：在 /r/LocalLLaMA 中，分享了 Command-r GGUF 在 [P40 上针对长上下文、flash attention 以及 KV quantization 的基准测试](https://www.reddit.com/r/LocalLLaMA/comments/1dcdit2/p40_benchmarks_part_2_large_contexts_and_flash/)，展示了对处理和生成速度的影响。
- **用于本地模型的 Mac Studio**：在 /r/LocalLLaMA 中，[搭载 M2 Ultra 的 Mac Studio](https://www.reddit.com/r/LocalLLaMA/comments/1dbwlnt/is_a_mac_studio_the_right_choice_for_me/) 被认为是运行本地大型模型的理想选择，因为它体积小、安静且功耗相对较低。
- **用于本地 LLM 的 AMD GPU**：在 /r/LocalLLaMA 中，一个帖子寻求在 Linux 环境下[使用 AMD Radeon GPU 运行本地 LLM](https://www.reddit.com/r/LocalLLaMA/comments/1dbz3mx/anyone_having_luck_using_amd_radeon_gpus_for/) 的经验和性能见解。

**迷因与幽默**

- **深夜 AI 创作**：在 /r/singularity 中，分享了一个[关于熬夜完善 AI 生成杰作（巨大的动漫胸部）的迷因](https://i.redd.it/go2tn2i5yn5d1.png)。
- **AI 生成的 Firefox Logo**：在 /r/singularity 中，发布了一个由 AI 生成的[过度复杂的 Firefox Logo](https://i.redd.it/war9t5skdl5d1.png)。
- **ChatGPT 的调情策略**：在 /r/singularity 中，《每日秀》（The Daily Show）的一个片段展示了[对 ChatGPT“调情”策略的搞笑反应](https://youtube.com/shorts/eVMNvm67Y-A)。

---

# AI Discord 回顾

> 摘要之摘要的摘要

1.  **多模态 AI 与生成模型创新**：
  
  - **Ultravox 进入多模态领域**：[**Ultravox**](https://ultravox.ai) 是一个开源的多模态 LLM，能够理解非文本语音元素，目前已发布 v0.1 版本。该[项目](https://x.com/juberti/status/1798898986289684849)正受到关注并正在招聘扩张。
  - **Sigma-GPT 首次亮相动态序列生成**：[**σ-GPT**](https://x.com/ArnaudPannatier/status/1799055129829839166) 提供动态序列生成，减少了模型评估时间。这种方法引发了对其实用性的兴趣和争论，一些人将其与 XLNet 的发展轨迹进行了比较。
  - **Lumina-Next-T2I 增强文本生成图像模型**：[**Lumina-Next-T2I**](https://huggingface.co/Alpha-VLLM/Lumina-Next-T2I) 模型拥有更快的推理速度、更丰富的生成风格和更好的多语言支持，正如 [Ziwei Liu 的推文](https://x.com/liuziwei7/status/1799846727534727649)所示。


2.  **模型性能优化与微调技术**：
  
  - **高效量化与 Kernel 优化**：关于 [**CUDA Profiling Essentials**](https://youtu.be/fsC3QeZHM1U) 的讨论建议使用 `nsys` 或 `ncu` 进行深入的 Kernel 分析。来自 **NVIDIA Cutlass** 和 **BitBlas** [文档](https://nvidia.github.io/cutlass/integer__subbyte_8h_source.html)的技术展示了有效的位级操作。
  - **LLama-3 微调问题已修复**：用户报告通过使用 **vllm** 解决了 **LLama3** 模型微调的问题，并在 [**axolotl**](https://discord.com/channels/1238365980128706560/1242542198008975430) 论坛中分享了相关配置。
  - **GROUP 项目**：该项目在 **OpenAI 和 Eleuther 社区**中探讨了微调与 RAG 概念的应对以及 LR（学习率）调整，并参考了来自 **Stanford** 的基准测试见解和 [GitHub](https://gist.github.com/matthewdouglas/1c0833f7fa9adbc54e4f5dc09e2b59a2) 上的 Git 设置。


3.  **开源 AI 框架与工具**：
  
  - **Rubik's AI Beta 测试邀请**：用户被邀请参与 [**Rubik's AI**](https://rubiks.ai) 的 Beta 测试，这是一个包含 GPT-4 Turbo、Claude-3 Opus 等模型的新型研究助手。该平台旨在推动 AI 研究的进步。
  - [**LSP-AI**](https://github.com/SilasMarvin/lsp-ai) **增强 IDE 兼容性**：重点介绍了一个旨在协助软件工程师的多编辑器 AI 语言服务器，社区对其跨平台增强能力表现出极高热情。
  - **集成 LangChain 与 Bagel**：LangChain 已与 [**Bagel**](https://x.com/bagel_network/status/1799143240769081731) 集成，提供安全、可扩展的数据集管理，突显了将语言模型与外部数据集成的进展。


4.  **AI 社区与活动亮点**：
  
  - **AI Engineer World’s Fair 公告**：[**AI Engineer World’s Fair**](https://www.ai.engineer/worldsfair) 公布了新演讲者，且门票已售罄，显示出社区极高的参与度和兴趣。
  - **创新项目与聚会**：社区亮点包括 **Websim.ai** 的递归探索等有趣项目，以及在旧金山举行的 **Lehman Trilogy** 活动等著名聚会，由 **Nathan Lambert** 在 [Interconnects discord](https://thelehmantrilogy.com) 中分享。
  - **ICLR 2024 播客与 AI 峰会见解**：[**ICLR 2024 播客**](https://www.latent.space/p/iclr-2024-benchmarks-agents)第 2 部分已发布，讨论了基准测试、Agent 等内容，丰富了社区知识和参与感。

5.  **技术创新与讨论**：
  
  - **多语言转录的困扰**：在 [**OpenAI discord**](https://discord.com/channels/974519864045756446/998381918976479273) 中，用户对 **Whisper v3** 在多语言转录方面的表现提出了批评，引发了对 OpenAI 未来改进的期待。
  - **安全与 API Token 管理**：[警告不要使用恶意的 ComfyUI 节点](https://www.reddit.com/r/comfyui/comments/1dbls5n/psa_if_youve_used_the_comfyui_llmvision_node_from/)，并建议使用环境变量进行 API Token 管理，这是大家共同关注的问题。
  - **性能分析与微调**：讨论包括优化大模型训练配置（如 [**CUDA profiling**](https://youtu.be/fsC3QeZHM1U) 所示），以及在 [**Mojo**](https://docs.modular.com/max/install) 等编程语言中使用结构化并发。

---

# 第一部分：Discord 高层级摘要

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Spotify 优惠杂谈**：工程师们讨论了获取 **Spotify Premium** 的各种途径，包括来自某些供应商的免费优惠，但未详细说明这些优惠的具体成本或条款。
- **区域设置揭秘**：探索了 **regional prompting**（区域提示词）技术，建议包括在 ComfyUI 中使用带有 **attention masks** 的 **IPAdapter**，并询问了 diffusers 中的类似功能，但对于最佳策略尚未达成明确共识。
- **围绕 Stable Diffusion 3 的热议**：社区正热切期待 **Stable Diffusion 3** (SD3)，辩论焦点集中在预期的功能上，如提示词服从性（prompt obedience）和图像生成增强，同时也关注自定义微调（fine-tunes）和发布后的 LoRas，整体氛围呈现出审慎的乐观。
- **LoRas 训练中的小插曲**：成员们分享了在使用 **ComfyUI** 和 **Kohya SS GUI** 等工具训练模型和 **LoRas** 时遇到的挑战和解决方法，并由于未说明的安装问题进一步推荐了 OneTrainer 等替代方法。
- **ComfyUI 安全警报**：发出了关于 ComfyUI 中一个**恶意节点**的警告，该节点有可能窃取敏感信息，引发了关于如何防范 AI 工具中自定义节点相关风险的更广泛讨论。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **随着 SpaceX Starship 的成功，AI 助力腾飞**：SpaceX 的 Starship 第四次试飞取得胜利，标志着向**全可重复使用火箭系统**迈进了一步，其第一级和第二级均成功着陆。这一成就详见 [Perplexity 平台](https://www.perplexity.ai/page/SpaceX-successfully-lands-PCd62nm2Sr.BZGUHFmHP2w)。
- **Starliner 前往 ISS 的坎坷之路**：波音公司的 Starliner 在与 ISS 对接期间，五个 RCS 推进器出现故障，可能影响任务时间表，并展示了航天硬件的复杂性。完整报告可在 [NASA 更新](https://www.perplexity.ai/page/Starliner-Docks-with-EzaalHkvRsqzKROOP2gjuw)中查看。
- **Perplexity 的难题与进展**：用户批评了 Perplexity AI 在 AI 旅行规划方面的能力有限，特别是在航班细节方面，而另一些用户则称赞其新的 Pro Search 功能提高了结果的相关性。社区报告的内容去索引（deindexing）和 GPT-4 模型的准确性问题引发了担忧。围绕 Rabbit R1 设备被指为骗局的说法也引发了争议。
- **地缘政治技术紧张局势**：华为的 Ascend 910B AI 芯片在训练大语言模型（LLM）方面表现出色，正向 Nvidia 的 A100 发起挑战，这引发了技术辩论和地缘政治影响。访问 [Perplexity 更新](https://www.perplexity.ai/page/Huaweis-New-AI-4EKcpjWjR3W3SuA38fyTGw)了解该芯片能力的详细信息。
- **Perplexity API 疑难杂症**：咨询和讨论集中在利用 Perplexity API 的功能上，例如无法生成 embedding 以及如何获得类似于 Web 版结果的建议，反映了用户对清晰文档和支持的需求。关于 API 积分的特定问题被建议通过私信解决，显示了积极的社区参与。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **微调集市上的爆米花与泊松分布**：关于使用概率模型预测*爆米花爆裂时间*的幽默讨论演变成了对*逆泊松分布（inverse Poisson distribution）*的分析。与此同时，一名成员邀请同学参加 [AI Engineer World's Fair](https://www.ai.engineer/worldsfair)，并承诺任何使用课程仓库对爆米花内核进行案例研究的人都可能获得传奇地位。
- **审查与性能成为 LLM 对话焦点**：一篇 [Hugging Face 博客文章](https://huggingface.co/blog/leonardlin/chinese-llm-censorship-analysis)引发了对 Qwen2 Instruct 中错误信息的担忧，进而引发了关于 LLM 性能和审查制度细微差别的讨论，重点关注英文与中文回答的差异。在其他地方，*LLama-3* 模型的微调问题通过使用 **vllm** 部署得到了解决。
- **微调过程中的挫折**：访问 Hugging Face、Replicate 和 Modal 等平台授予的额度（credits）的过程引起了混乱，几位成员没有收到预期的金额，导致一些人表达了失望并寻求解决方案。
- **Modal 的魔力伴随着复杂反应**：成员们分享了在 Modal 上部署模型的经验，评价从称其为“神奇体验”到在权限和 volume ID 错误中挣扎不等，这表明新部署平台存在学习曲线和成长的烦恼。
- **研讨会的困扰与获胜技术**：讨论了技术问题，包括 Workshop 4 的 Zoom 录音部分丢失，已通过分享最后几分钟的链接解决。讨论还赞扬了 Weights & Biases 的资源，如 [10 分钟视频课程](https://www.wandb.courses/courses/wandb-101)，以及 ColBERT 即将在博客文章中详细介绍的新分层池化（hierarchical pooling）功能。
- **微调 vs. RAG 辩论展开**：有人在 LLM 中对微调和 RAG 的角色提出了一个有趣的类比，将添加静态知识与动态的、特定于查询的信息进行并列。然而，这遭到了一些反对，一名成员正致力于对这些复杂概念进行更精确的解释。
- **Accelerate 框架测试揭示速度差异**：一位 AI 工程师使用 **accelerate** 测试了训练配置，对比了 DDP, FSDP 和 DS(zero3)，在正面交锋中发现 DS(zero3) 的 vRAM 效率最高，速度排名第二。
- **全球签到与本地聚会**：成员们从全球各地进行签到，并为旧金山地区的成员提议了一场即兴聚会，展示了社区在数字领域之外建立联系的热情。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **动态对话模型**：[σ-GPT](https://x.com/arnaudpannatier/status/1799055129829839166?s=46) 脱颖而出，成为改变游戏规则的存在，它在推理时动态生成序列，而不同于 GPT 传统的从左到右生成。正如 [OpenAI 博客](https://openai.com/index/extracting-concepts-from-gpt-4/)中所详述的，人们将其与从 GPT-4 中提取概念进行了比较，引发了关于方法论和应用的对话。
- **高风险编辑与法律话语**：对于那些敢于尝试 outpainting 的人，推荐使用 [Krita stable diffusion 插件](https://github.com/Acly/krita-ai-diffusion)；同时，[Interstice Cloud](https://www.interstice.cloud) 和 [Playground AI](https://playground.com/pricing) 被提议作为降低 GPU 云成本的经济高效解决方案。与此同时，关于 [SB 1047](https://x.com/dan_jeffries1/status/1794740447052525609?s=46) 的讨论引发了关于 AI 监管及其对行业活力影响的争论。
- **数据图谱与标准**：成员们讨论了用于 RAG 数据集的 JSON schemas，并倡导使用更结构化的格式（如结合相关性、相似度得分和情感指标）来磨练语言模型的输出。还研究了 Cohere 的检索系统和结构化引用机制等工具的集成，表明由于 JSON 的简单性和易用性，人们更倾向于使用它。
- **革新资源限制**：分享了针对低配置 PC 的解决方案，例如尽管 **Phi-3 3b** 在代码相关任务中存在局限性，但仍采用它。这表明社区关注各种硬件配置下的资源可访问性和优化。
- **方法论对决**：专注于聚类以实现高效语言模型训练的 HippoRAG 的突出地位，标志着向优化信息提取过程的转变。成员们参考 [相关工作](https://arxiv.org/abs/2403.17887) 和 [PruneMe](https://github.com/arcee-ai/PruneMe) 等工具，就模型剪枝（pruning）和微调策略的最佳实践进行了深入探讨。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen 模型中的 GGUF 故障**：工程师报告称 **Qwen GGUF** 会导致“块状”文本输出，尤其是在 **7B model** 中，尽管部分用户使用 lm studio 等工具成功运行。Qwen 模型表现不佳的问题仍然是高热度讨论的话题。
- **多编辑器语言服务器增强**：[LSP-AI](https://github.com/SilasMarvin/lsp-ai) 是一款提供 VS Code 和 NeoVim 等编辑器兼容性的语言服务器，被强调为一种增强而非取代软件工程师能力的工具。
- **简化模型 Finetuning**：用户对用于 **continued pretraining** 的易用型 Unsloth Colab notebook 表示赞赏，它简化了 finetuning 流程，特别是针对 **input and output embeddings**。相关支持包括 [Unsloth Blog](https://unsloth.ai/blog/contpretraining) 和 [repository](https://github.com/unslothai/unsloth/releases/tag/June-2024)。
- **比特之战与模型合并**：讨论深入探讨了 **QLoRA**、**DoRA** 和 **QDoRA** 等 4-bit quantization 方法之间的区别，以及使用 differential weight strategy 进行 model merging 策略的细节，展示了社区成员对高级 ML 技术的熟练掌握。
- **值得关注的 Notebook 网络**：[showcase channel](https://discord.com/channels/1179035537009545276/1179779344894263297/1249410284616159365) 展示了一系列针对 **Llama 3 (8B)**、**Mistral v0.3 (7B)** 和 **Phi-3** 等知名模型的 Google Colab 和 Kaggle notebook，强调了社区内的易用性和协作精神。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA Profiling 基础**：使用 **nsys** 或 **ncu** 进行 CUDA profiling，对于深入分析，应专注于单次前向和后向传递（forward and backward pass），如[算子性能分析视频](https://youtu.be/fsC3QeZHM1U)所示。对于构建个人 ML 实验机，可以考虑 **Ryzen 7950x** 等 CPU 以及 **3090** 或 **4090** 等 GPU，并注意 **AVX-512** 支持以及 **Threadrippers** 和 **EPYCs** 等服务器级 CPU 的权衡。
- **Triton 的崛起**：[FlagGems 项目](https://github.com/FlagOpen/FlagGems)因在大型 LLM 中使用 Triton Language 而受到关注。技术讨论包括处理通用 kernel 大小、将向量加载为对角矩阵，以及寻找最先进的 Triton kernel 资源，可在[此 GitHub 目录](https://github.com/cuda-mode/triton-index)中找到。
- **PyTorch 讨论**：为了准确测量 `torch.compile`，请从初始传递时间中减去第二个 batch 的时间；此处提供[故障排除指南](https://github.com/pytorch/pytorch/blob/main/docs/source/torch.compiler_troubleshooting.rst#cold-start-timing-and-cache-corruption-debugging)。在 [PyTorch 的 GitHub](https://github.com/pytorch/pytorch/tree/main/benchmarks/dynamo) 中探索 Inductor 性能脚本，并考虑使用自定义 C++/CUDA 算子（operator），如[此处](https://github.com/pytorch/ao/pull/135)所示。
- **高速扫描的未来预测**：对客座演讲者关于扫描技术的演讲充满期待，预计会有创新性的见解。
- **电子学启示**：The Amp Hour 播客的一期节目邀请了 **Bunnie Huang** 担任嘉宾，深入探讨了硬件设计和 [Hacking the Xbox](https://www.amazon.com/gp)，可通过 [Apple Podcasts](https://theamphour.com/feed/podcast/?mt=2&ls=1) 或 [RSS](https://theamphour.com/feed/podcast/) 收听。
- **转型建议**：成员们分享了转型到基于 GPU 的机器学习的技巧，建议利用 Fatahalian 的视频和 [Yong He 的 YouTube 频道](https://www.youtube.com/@csyonghe/videos)来学习 GPU 架构。
- **Encoder 探索与 GPT 指导**：虽然没有提供关于 PyTorch 中 encoder-only 模型有效参数搜索的详细信息，但分享了[重现 GPT-2](https://www.youtube.com/watch?v=l8pRSuU81PU) 的资源。NVIDIA 的 RTX 4060Ti (16GB) 被建议作为 CUDA 学习的入门级选择。
- **FP8 在 PyTorch 中的角色**：关于使用 FPGA 模型和考虑不含 matmul 的三值（ternary）模型的讨论，并辅以 [Intel FPGA](https://www.intel.com/content/www/us/en/products/sku/193921/intel-fpga-pac-d5005/specifications.html) 的链接和相关[论文](https://arxiv.org/pdf/2406.02528)。呼吁提供更好的 torch.compile 和 torchao 文档及基准测试，并关注 [Pull Request #276](https://github.com/pytorch/ao/pull/276) 中针对 GPT 模型的新增加内容。
- **Triton 话题再次出现**：链接了一个有趣的三值累加（ternary accumulation）演示，获得了社区的积极反馈（[matmulfreellm](https://github.com/ridgerchu/matmulfreellm)）。
- **llm.c 讨论**：关于模型训练的广泛讨论涵盖了超参数选择、重叠计算（overlapping computations）、FineWebEDU 数据集问题，以及使用详细[脚本](https://gist.github.com/matthewdouglas/1c0833f7fa9adbc54e4f5dc09e2b59a2)成功将模型转换为 Hugging Face 格式。
- **Bit 与 Bitnet**：使用微分位计数（differential bitcounts）的技术引发了好奇和调试工作。将 FPGA 的成本与 A6000 ADA GPU 的速度进行了比较，同时确认 NVIDIA 的 Cutlass 支持 **nbit bit-packing**，包括 uint8 格式（[Cutlass 文档](https://nvidia.github.io/cutlass/integer__subbyte_8h_source.html)）。此外，[BitBlas 的基准测试结果](https://gist.github.com/mobicham/3ef2ef33d7f234f84f80249c41b6fae0)引发了关于 matmul fp16 性能差异的讨论。
- **ARM 雄心**：简要提到讨论可能涉及 ARM 服务器芯片而非移动处理器，并链接了一个[热门 YouTube 视频](https://www.youtube.com/watch?v=ydGdHjIncbk)作为参考。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **大模型，深度讨论**：工程师们讨论了 2B parameter models 的计算需求，公认 50GB 的系统可能不足，可能需要超过 2 个 T4 GPUs。关于 API 的争论凸显了对成本和访问权限的困惑，批评者将 OpenAI 平台戏称为 "closedAI"。
- **科技巨头之战**：尽管 Nvidia 拥有“锁定生态系统”，但其市场主导地位仍得到认可，其 AI 芯片创新和游戏行业需求使其在技术领导地位中保持核心地位。
- **API Tokens 安全提示**：一次意外的邮件 Token 泄露引发了在软件开发中使用环境变量来增强安全性的建议。
- **AI 在模拟中的力量**：成员们接触到了诸如 AI Summit YouTube 录像等资源，展示了 AI 在物理模拟中的应用，并受邀参加由 Stanford 研究人员举办的关于防止模型崩溃（model collapse）的活动。
- **机器学习新动向**：分享了一系列 AI 工具和进展，包括用于 LLM 微调的 Torchtune、用于多功能 LLM 使用的 Ollama、用于图像分类的 Kaggle 数据集，以及用于可持续农业的 FarmFriend。
- **前沿 AI 创作**：AI 领域的创新包括发布了针对尼日利亚语境回复的 Llama3-8b-Naija、用于多 GPU 训练增强的 SimpleTuner v0.9.6.3、用于提升超写实美感的 Visionix Alpha，以及可以与来自不同 AI 公司的各种模型对话的 Chat With 'Em。
- **CV 和 NLP 进展展示**：亮点包括关于旋转边界框（rotated bounding boxes）高效实现的讨论、Gemini 1.5 Pro 在视频分析中的优越性，以及针对 CVPR 2024 论文的语义搜索工具。在 NLP 领域，主题涵盖了从构建基于 RAG 的聊天机器人到使用 MyResumo 生成 AI 简历，以及关于 PyTorch 与 TensorFlow 在模型托管和错误处理方面的咨询。
- **Diffusion Model 动态**：讨论集中在利用共享资源训练 Conditional UNet2D 模型、利用 SDXL 进行图像文本印记，以及对训练期间计算 MFU 的好奇，并由此提出了对代码库修改的建议。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**新可视化模型仍在排队中**：**LM Studio** 目前不支持生成图像 Embedding；建议用户关注 **daanelson/imagebind** 或等待 **nomic** 和 **jina** 的未来版本。

**给 Tesla P40 降降温！**：对于 **Tesla P40** 的散热，社区建议从使用 Mac 风扇到尝试自定义 3D 打印导风罩，一位用户指向了 [Mikubox Triple-P40 散热指南](https://rentry.org/Mikubox-Triple-P40)。

**跨越多 GPU 之桥**：讨论指出，虽然 **LM Studio** 在高效多 GPU 支持方面进度落后，但 **ollama** 表现出更出色的处理能力，促使用户寻求更好的 GPU 利用方法。

**解决硬件兼容性**：从处理将 **AMD 的 ROCm** 注入 Windows 应用程序，到解决 **Tesla P40** 的驱动安装，用户分享了经验和解决方案，包括来自 AMD [文档](https://rocm.docs.amd.com/en/latest/conceptual/gpu-isolation.html)的隔离技术。

**LM Studio 等待 Smaug 的 Tokenizer**：**LM Studio** 的下一个版本将包含对 **Smaug 模型** 的 **BPE Tokenizer** 支持，同时成员们也在探索将 **LMS 数据定向到外部服务器** 的选项。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **iOS 抢占 AI 焦点**：OpenAI 宣布与 Apple 合作，在 iOS、iPadOS 和 macOS 平台上集成 **ChatGPT**，计划于今年晚些时候发布，引发了关于 AI 在消费科技领域影响的热烈讨论。详情和反应可见 [官方公告](https://openai.com/apple)。
- **多语言转录风波与 Apple AI 进展**：**Whisper version 3** 在多语言转录方面表现不佳引发热议，用户纷纷要求推出新版本；同时 Apple 的 “Apple Intelligence” 承诺将提升 iPhone 16 的 AI 能力，可能需要硬件升级以进行优化。
- **图像 Token 经济学与 Agent 困扰**：在经济方面，关于 128k 上下文 Token 化和图像处理的 API 调用成本效益讨论升温；在技术方面，用户对 GPT Agent 默认使用 GPT-4o 导致性能欠佳表示不满。
- **自定义 GPTs 与语音模式的烦恼**：AI 爱好者正在剖析自定义 GPTs 的私有性质（实际上被禁止与外部 OpenAPI 集成），同时对 Plus 用户新语音模式推送缓慢感到困惑和不耐烦。
- **HTML 与 AI 代码编写挑战**：讨论集中在如何让 ChatGPT 输出极简 HTML、改进摘要 Prompt、使用 Canva Pro 进行图像文本编辑、理解 LLM 的失效点，以及生成将十六进制代码转换为 Photoshop 渐变映射的 Python 脚本，这表明工具和指令仍需磨合。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **CPU 模型解决 GPU 贫乏问题**：工程师们讨论了针对有限 GPU 资源的变通方案，考虑使用 **sd turbo** 和基于 CPU 的解决方案来减少等待时间，有人表示这种体验仍然“值得”。
- **固定种子对抗局部最小值**：在神经网络训练中固定种子与随机种子的争论中，一些人倾向于设置手动种子以微调参数并逃离局部最小值，强调“种子总归是存在的”。
- **MatMul 操作被剔除**：一篇 [arXiv 论文](https://arxiv.org/abs/2406.02528)展示了参数量高达 2.7B 的无 MatMul 模型，引发讨论，认为此类模型在保持性能的同时可能降低计算成本。
- **扩散模型：NLP 的新宠？**：转向使用扩散模型来增强 LLM 已被提上日程，[这篇综述论文](https://arxiv.org/abs/2305.14671)等参考文献激发了关于该话题的对话。
- **匈牙利押注 AI Safety**：分析了在匈牙利投资 3000 万美元进行 AI Safety 研究的可行性，强调了不浪费资金的重要性，并考虑使用云端资源满足计算需求。
- **RoPE 技术来救场**：研究频道揭示了实现相对位置编码 (RoPE) 以改进非自回归模型的热情，成员们提出了各种初始化方案，如通过插值权重矩阵进行模型扩缩，以及使用 SVD 进行 LoRA 初始化。
- **给模型“瘦身”**：一位工程师使用层剪枝 (layer pruning) 成功将 Qwen 2 72B 缩减至 37B 参数，在不牺牲性能的情况下展示了效率。
- **可解释性：新前沿**：对 **TopK activations** 的兴趣重新抬头，一个探索 Llama3 中 **MLP neurons** 的项目受到关注，资源可在 [neuralblog](https://neuralblog.github.io/llama3-neurons/) 和 [GitHub](https://github.com/neuralblog/llama3-neurons) 上找到。
- **绝望的 MAUVE**：一位成员在 [MAUVE setup](https://github.com/krishnap25/mauve-experiments) 方面寻求帮助，强调了在安装和使用该工具评估新采样方法时面临的复杂性。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **MacOS 安装遇到障碍**：在 MacOS 14.5 Sonoma 上安装 **MAX** 的工程师面临挑战，需要手动干预。解决方案包括通过 pyenv 设置 Python 3.11，详见 [Modular 官方安装指南](https://docs.modular.com/max/install)。
- **探讨编程中的并发性**：引发了关于结构化并发（structured concurrency）与函数着色（function coloring）的辩论，并提出了效应泛型（effect generics）作为解决方案，尽管这增加了语言编写的复杂性。讨论还延伸到了 Erlang、Elixir 和 Go 等语言中的并发原语，以及 Mojo 从底层为这些范式设计解决方案的潜力。
- **最大化你的 Mojo**：关于 Mojo 语言的见解涵盖了 MAX 平台中使用 GGML k-quants 的量化主题，并提供了现有文档和示例的链接，例如 [Llama 3 pipeline](https://github.com/modularml/max/tree/f89bc8f4e685e2bbcc269c8c324b5c105391f6f9/examples/graph-api/pipelines/llama3)。此外，由于上下文管理器（context managers）在 Python 生态中具有整洁的资源管理能力，其被认为优于潜在的 `defer` 关键字。
- **Modular 发布更新**：最近的开发更新包括视频内容，**Modular** 发布了一个[新的 YouTube 视频](https://www.youtube.com/watch?v=3FKSlhZNdL0)，这对关注者来说可能至关重要。另一个重点资源是来自 Andrej Karpathy 的项目，通过 [YouTube](https://youtu.be/l8pRSuU81PU) 分享，预计社区会对此感兴趣。
- **新版本的工程效率**：Mojo 编译器的 Nightly 版本显示出进展，更新至版本 `2024.6.805`、`2024.6.905` 和 `2024.6.1005`，变更日志可在此处供社区查阅 [here](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)。这些迭代版本构成了 Modular 编程领域持续改进的叙事。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**Gorilla OpenFunctions v2 媲美 GPT-4**：社区成员一直在讨论 [Gorilla OpenFunctions v2](https://huggingface.co/gorilla-llm/gorilla-openfunctions-v2) 的能力，注意到其令人印象深刻的性能以及从自然语言指令生成可执行 API 调用的能力。

**Local II 推出本地 OS 模式**：**Local II** 宣布支持本地 OS 模式，可实现实时演示，可通过 `pip install --upgrade open-interpreter` 进行关注。

**OI 模型出现技术问题**：用户报告了 **OI 模型** 的各种问题，包括 API key 错误以及 moondream 等视觉模型的问题。故障排除中的交流表明修复和改进正在进行中。

**OI 在 iPhone 和 Siri 上的里程碑**：**Open Interpreter** 与 iPhone 的 Siri 集成取得了突破，允许通过语音命令执行终端功能，参考[教程视频](https://youtu.be/Tj9uyyflgxw?feature=shared)。

**Raspberry Pi 和 Linux 用户的尝试与需求**：在 Raspberry Pi 上运行 OI 的尝试遇到了资源问题，但社区仍致力于寻找解决方案。对 Linux 安装教程的请求表明了对跨平台支持的广泛渴望。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Ultravox 登场**：[Ultravox](https://ultravox.ai) 发布了 v0.1 版本，这是一个能够理解非文本语音元素的全新**开源多模态 LLM**。目前正在进行招聘以扩大其开发规模。
- **OpenAI 聘请新高管**：OpenAI 在其 Twitter 上发布了新任命的 CFO 和 CPO——**Friley** 和 **Kevin Weil**，进一步增强了组织的领导团队。
- **Perplexity 因内容滥用遭受抨击**：Perplexity 招致了批评，包括来自 @JohnPaczkowski 的推文，指责其在未获得适当授权的情况下重新利用 Forbes 的内容。
- **Apple 凭借 Cloud Compute Privacy 进军 AI**：Apple 最近关于 "Private Cloud Compute" 的公告旨在安全地将 AI 任务卸载到云端，同时保护隐私，这引发了工程界的广泛讨论。
- **ICLR 播客与 AI World's Fair 更新**：最新的 [ICLR 播客剧集](https://www.latent.space/p/iclr-2024-benchmarks-agents) 深入探讨了代码编辑以及学术界与工业界的融合，而 [AI Engineer World's Fair](https://www.ai.engineer/worldsfair) 列出了新演讲者，并宣布赞助名额和早鸟票已售罄。
- **Websim.ai 激发递归混沌与创意**：对实时流媒体面部识别网站的发现，引导成员们将 websim.ai 递归地应用于自身，制作了一个 greentext 生成器，并分享了一个资源电子表格，展现了探索 Websim 新前沿的创新精神和好奇心。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 的 Command R 模型占据领先地位**：最新对话显示，**Cohere 的 Command R 和 R+ 模型**被认为是行业顶尖水平，用户正在 [Amazon SageMaker](https://aws.amazon.com/marketplace/seller-profile?id=87af0c85-6cf9-4ed8-bee0-b40ce65167e0) 和 [Microsoft Azure](https://ai.azure.com/explore/models/?tid=694fed05-7f6d-4ab2-8c38-9afb438eab6f) 等云平台上使用它们。
- **创新 AI 驱动的角色扮演**："reply_to_user" 工具被公认为能增强 AI 角色扮演中的角色化回复，特别是在 [Dungeonmasters.ai](https://www.dungeonmasters.ai/) 等项目中，这表明交互能力正向更具上下文感知的方向转变。
- **多元化的 Cohere 社区积极参与**：Cohere 社区的新成员（包括一名巴西初级 NLP DS 和一名 MIT 毕业生）正在分享他们对 NLP 和 AI 项目的热情，显示出一个充满活力且多元化的协作环境。
- **塑造 AI 职业生涯与项目**：成员的项目讨论阐明了 **Cohere API** 在提高性能方面的作用，正如在需要 AI 集成的领域获得的积极反馈所证明的那样，这标志着一种对开发者互利的合作关系。
- **Cohere 的 SDK 拓宽视野**：**Cohere SDKs** 与 AWS、Azure 和 Oracle 等多种云服务的兼容性已经公布，增强了灵活性和开发选项，详见其 [Python SDK 文档](https://docs.cohere.com/docs/cohere-works-everywhere)。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

**σ-GPT 为高效序列生成铺平道路**：引入了一种名为 σ-GPT 的新方法，提供具有即时定位功能的动态序列生成，在减少语言建模等领域的模型评估方面显示出强大潜力（[阅读 σ-GPT 论文](https://arxiv.org/abs/2404.09562)）。尽管其前景广阔，但由于需要特定的课程学习（curriculum），人们对其实用性表示担忧，并将其与 XLNET 的轨迹相类比。

**AI 推理挑战曝光**：一项针对 Transformer 嵌入的研究揭示了关于离散与连续表示的新见解，阐明了在性能损失微乎其微的情况下对注意力头进行剪枝的可能性（[分析多头自注意力论文](https://arxiv.org/abs/1905.09418)）。此外，一个旨在测试 LLM 推理能力的 Prompt 仓库被分享，指出训练数据偏差是模型失败背后的关键原因（[MisguidedAttention GitHub 仓库](https://github.com/cpldcpu/MisguidedAttention)）。

**加密货币对话引发关注**：使用加密货币支付 AI 算力的做法引发了褒贬不一的反应，一些人看到了潜力，而另一些人则持怀疑态度，将其贴上可能为诈骗的标签。随后出现了一条关于 ComfyUI_LLMVISION 节点可能收集敏感信息的警告，敦促与其交互过的用户采取行动（[ComfyUI_LLMVISION 节点警报](https://www.reddit.com/r/comfyui/comments/1dbls5n/psa_if_youve_used_the_comfyui_llmvision_node_from/)）。

**展示 AI 的进展与问题**：小组讨论了 Lumina-Next-T2I 的发布，这是一款新的文本生成图像模型，因其增强的生成风格和多语言支持而受到赞誉（[Hugging Face 上的 Lumina-Next-T2I](https://huggingface.co/Alpha-VLLM/Lumina-Next-T2I)）。在一个更具警示意义的案例中，巴西 AI 数据集中滥用儿童照片的事件成为焦点，揭示了数据来源的阴暗面以及公众对 AI 隐私问题的漠视（[人权观察报告](https://www.hrw.org/news/2024/06/10/brazil-childrens-personal-photos-misused-power-ai-tools)）。

**WebSocket 问题与预训练模型的潜力**：在技术故障排除方面，分享了诊断通用 WebSocket 错误的技巧，以及在文本转语音（TTS）服务 WebSocket 中观察到的特殊持续延迟。为了增强项目，建议使用具有扩展上下文窗口的预训练 Instruct 模型，特别是为了将 Rust 文档纳入模型的训练体系。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **图谱专家齐聚**：一场专注于**高级知识图谱 RAG** 的研讨会定于太平洋时间周四上午 9 点举行，由来自 Neo4j 的 Tomaz Bratanic 主讲，涵盖 LlamaIndex 属性图和图查询技术。感兴趣的参与者可以[在此报名](https://lu.ma/kqxmbuou)。
- **编码增强 RAG**：推荐了一系列资源以改进 RAG 应用中的数据分析和用户交互，包括[集成沙箱环境](https://twitter.com/llama_index/status/1799176083381866757)、[构建 Agentic RAG 系统](https://twitter.com/llama_index/status/1799463683179098203)、[查询重写技巧](https://twitter.com/llama_index/status/1799566113208225891)以及[创建快速语音机器人](https://twitter.com/llama_index/status/1799833244894200135)。
- **优化 AI 的效率与精度**：讨论强调了增加 **SimpleDirectory.html** 读取器中 `chunk_size` 的策略，以及管理图存储中的实体解析，并参考了 LlamaIndex 关于[存储文档](https://docs.llamaindex.ai/en/latest/understanding/storing/storing/#inserting-documents-or-nodes)和[优化流程](https://docs.llamaindex.ai/en/latest/optimizing/basic_strategies/basic_strategies/#chunk-sizes)的文档，以构建可扩展的 RAG 系统。
- **LlamaParse 现象已修复**：**LlamaParse** 的临时服务中断已由社区迅速解决，确保了依赖该工具进行解析需求的用户能够获得不间断的服务。
- **寻求通过 QLoRA 增强 RAG**：目前正在努力从电话手册中开发数据集，利用 **QLoRA** 训练模型，旨在提高 **RAG** 性能。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **三款全新 AI 模型上市**：[Qwen 2 72B Instruct](https://openrouter.ai/models/qwen/qwen-2-72b-instruct) 在语言熟练度和代码理解方面表现出色；[Dolphin 2.9.2 Mixtral 8x22B](https://openrouter.ai/models/cognitivecomputations/dolphin-mixtral-8x22b) 面世，其使用挑战在于 1 美元/M tokens 的价格，且依赖于每天 1.75 亿 tokens 的使用率。同时，[StarCoder2 15B Instruct](https://openrouter.ai/models/bigcode/starcoder2-15b-instruct) 作为首个专门用于编码任务的自对齐开源 LLM 开放使用。
- **利用 AI Brushes 增强代码能力**：一款为 VS Code 设计的 AI 增强型代码转换插件现已[免费提供](https://marketplace.visualstudio.com/items?itemName=ThijsDekkers.ai-code-brushes)，该插件利用 OpenRouter 和 Google Gemini，旨在通过利用[这些排名](https://openrouter.ai/rankings/programming/scripting?view=week)中 Programming/Scripting 类别里表现顶尖的模型来革新编码方式。
- **支付讨论中的电子货币与加密货币**：社区正在讨论采用 Google Pay 和 Apple Pay 以简化支付体验，并考虑加入 cryptocurrency 支付作为去中心化选项。
- **攻克 JSON Stream 挑战**：工程师们交流了处理 OpenRouter chat completions 流仅交付部分 JSON 响应情况的策略；运行缓冲区（running buffer）成为了关注焦点，同时还有来自[一篇说明性文章](https://blog.stackademic.com/swift-streaming-openai-api-response-chunked-encoding-transfer-48b7f1785f5f)的见解。
- **应对偏见与扩展语言支持**：一项关于 LLM 内部审查和偏见的调查集中在中美模型的对比上，详见[《利用 Qwen 2 Instruct 分析中国 LLM 的审查与偏见》](https://huggingface.co/blog/leonardlin/chinese-llm-censorship-analysis)；同时社区呼吁在模型熟练度中加入更好的语言类别评估，渴望对捷克语、法语和普通话等语言提供更细致的支持。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**Apple Intelligence：不仅仅是 Siri 的更新**：Nathan Lambert 强调了 Apple 的“personal intelligence”，这可能会重塑 Siri 的角色，使其超越语音助手。尽管最初对 OpenAI 的角色存在困惑，Lambert 承认 [Apple Intelligence](https://www.apple.com/newsroom/2024/06/introducing-apple-intelligence-for-iphone-ipad-and-mac/) 系统是迈向“面向大众的 AI”的重要一步。

**RL 社区审视 SRPO 倡议**：来自 [Cohere 关于 SRPO 的论文](https://cohere.com/research/papers/self-improving-robust-preference-optimization-2024-06-07)引发了讨论，该论文介绍了一种新的离线 RLHF 框架，旨在提高 out-of-distribution 任务的鲁棒性。该技术使用 min-max 优化，并被证明可以解决之前 RLHF 方法中固有的任务依赖问题。

**Dwarkesh Podcast 期待值攀升**：Dwarkesh Patel 与 François Chollet 即将播出的节目备受关注，因为 Chollet 对 AGI 时间线有着独特的见解。这与通常的乐观情绪相反，可能会为 AGI 讨论提供引人注目的贡献。

**Daylight Computer：小众但值得关注**：工程社区对 [Daylight Computer](https://daylightcomputer.com) 表示好奇，注意到它在减少蓝光暴露和辅助阳光直射下的可见度方面所做的尝试。同时，对于成为此类新颖技术的早期采用者所带来的风险，人们持适度的怀疑态度。

**RL 模型审查公开征集**：Nathan Lambert 提议为 RL 频道中讨论的近期论文里尚未验证的方法提供 Pull Requests 反馈。这表明社区为测试和验证提供了一个支持性的环境。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**Markdown 困境与缺失的方法**：工程师们报告了一个问题，即一个 **25MB 的 Markdown 文件**在 LangChain 处理过程中无限期运行，且目前没有建议的解决方案；同时，在使用 `create_tagging_chain()` 时由于 Prompt 被忽略而出现问题，这表明文档中可能存在 Bug 或空白。

**使用 LangChain 和 Bagel 保护你的数据集**：LangChain 与 Bagel 的新集成引入了安全、可扩展的数据集管理，相关进展在一条 [推文](https://x.com/bagel_network/status/1799143240769081731) 中被强调，这可能会增强数据密集型应用的基础设施。

**文档难题**：讨论集中在为 LangChain 加载和拆分文档，强调了针对 PDF 和代码文件等不同文档类型所需的技术技巧，为优化预处理以提高语言模型性能提供了途径。

**API 歧义**：有用户寻求关于如何在 LangServe 中使用 **api_handler()** 而不求助于 **add_route()** 的澄清，特别是旨在实现 *playground_type="default" 或 "chat"* 但缺乏指导。

**AI 创新诚邀参与**：社区成员受邀测试新的高级研究助手 Rubik's AI，该助手可访问 GPT-4 Turbo 等模型；同时还可以查看其他社区项目，如记者的可视化工具、音频新闻简报服务以及 Hugging Face 上的多模型聊天平台，反映了活跃的开发和测试活动。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Pip 技巧**：工程师们发现，通过 `pip3 install -e '.[deepspeed]'` 和 `pip3 install -e '.[flash-attn]'` **分别安装包**可以避免 RAM 溢出，这是在 Python 3.10 的新 conda 环境中工作时的一个有用技巧。
- **Axolotl 的多模态查询**：询问了 **axolotl** 对多模态微调的支持；提到了一个过时的 **Qwen** 分支，指向了潜在的重启或更新需求。
- **数据集加载故障**：成员们报告了数据集加载问题，其中包含括号的文件名可能会导致 `datasets.arrow_writer.SchemaInferenceError`；解决命名规范对于无缝数据处理至关重要。
- **学习率救星**：关于有效 Batch Size 的重申断言，根据 [Hugging Face](https://github.com/huggingface/accelerate/tree/main/docs/source/concept_guides/performance.md#L75L99) 的指导，在更改 Epoch、GPU 或 Batch 相关参数时，**学习率调整**是维持训练稳定性和效率的关键。
- **JSONL 配置指南**：分享了 **JSONL 数据集**的配置技巧，包括为训练和评估数据集指定路径；这包括 **alpaca_chat.load_qa** 和 **context_qa.load_v2** 的路径，有助于在模型训练期间更好地处理数据。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **PyTorch 代码评价褒贬不一**：George Hotz 评价了 [PyTorch 的 fuse_attention.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/fuse_attention.py)，称赞其设计优于 UPat，但指出其过于冗长并考虑进行语法增强。
- **tinygrad 开发者寻求效率提升**：tinygrad 中的一个初学者项目旨在加速 Pattern Matcher，并通过 Process Replay 测试设定了确保正确性的基准。
- **剖析 UOp 中的 'U'**："Micro op"（微操作）是 UOp 中 "U" 的含义，这是由 George Hotz 澄清的，反驳了社区内的其他潜在推测。
- **Hotz 为欧洲代码大会做准备**：George Hotz 将在 Code Europe 讨论 tinygrad；他接受了社区建议，修改其演讲的最后一张幻灯片以增强观众互动。
- **tinygrad 对 AMD 和 Nvidia 的 GPU 规格要求**：**AMD GPU** 需要最低 RDNA 规格，而 **Nvidia** 的门槛是 **2080** 型号；建议使用 HIP 或 OpenCL 作为已失效的 HSA 的替代方案。RDNA3 GPU 已验证为兼容。

## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **游戏设计反思：Serverless 函数引领潮流**：核心讨论集中在 [http://hexagen.world](http://hexagen.world) 中用于游戏循环的 **Convex 架构** 独特的 Serverless 函数，这与旧游戏范式中对内存和机器的依赖形成鲜明对比。通过分布式函数增强了可扩展性，在实现高效后端扩展的同时，通过 websocket 订阅确保了实时客户端更新。
- **AI Town 架构深度解析**：建议对 AI 和 CS 感兴趣的工程师探索 [AI Town 架构文档](https://github.com/a16z-infra/ai-town/blob/main/ARCHITECTURE.md) 提供的深度剖析，这是一个非常有见地的资源。
- **多人同步难题**：多人环境固有的延迟问题被强调为在基于 Convex 的游戏架构中提供最佳竞争体验的一项挑战。
- **令人困惑的 convex.json 配置难题**：用户报告了对缺失 **convex.json** 配置文件感到困惑，并面临后端错误，提示可能缺失依赖项，消息为："Recipe `convex` could not be run because just could not find the shell: program not found."
- **Hexagen 创作者现身**：这款由 Serverless 函数驱动的游戏 [http://hexagen.world](http://hexagen.world) 的创作者对社区分享其项目表示了认可。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Agentic 架构：是掩饰而非修复**：有讨论指出 "Agentic 架构" 仅仅是掩饰而非解决了复杂系统中的深层问题，尽管有类似 *Theorem 2* 的提示表明缓解是可能的。
- **结构性约束削弱了真正的推理能力**：工程师们强调，由于固有的结构限制，RNNs, CNNs, SSMs 和 Transformers 等架构在实际推理任务中表现挣扎，这一点被 *Theorem 1* 所印证。
- **重温理论基础**：一位成员表示打算重新研读论文，以更好地理解当前模型架构中存在的局限性和通信复杂性问题。
- **探讨通信复杂性与 Theorem 1**：深入探讨了多 Agent 系统中的通信复杂性概念，*Theorem 1* 说明了准确计算需要多次通信，这可能导致 Agent 生成幻觉（hallucinated）结果。
- **计划深度研读论文**：计划重新阅读并讨论所引用论文的细节，特别是关于 *Theorem 1* 在函数组合和多 Agent 系统通信挑战方面的见解。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **排行榜刺激发布策略**：一位成员推测，最近的发布是出于战略考虑，旨在促进更多研究并在行业排行榜上占据一席之地，强调了其对进一步分析和基准测试的效用。
- **UMAP 因卓越的聚类性能受到赞赏**：UMAP 因其出色的聚类表现受到一名成员的称赞，他向对该工具技术深度感兴趣的人推荐了一段对 UMAP 创作者的精彩采访。
- **与 UMAP 幕后大脑进行深度探讨**：一段名为 "[Moving towards KDearestNeighbors with Leland McInnes - creator of UMAP](https://www.youtube.com/watch?v=U_0GcpkjNVQ)" 的 YouTube 采访被重点推荐，其中由创作者 Leland McInnes 直接提供了关于 UMAP 及其相关项目（如 PyNNDescent 和 HDBScan）复杂细节的丰富讨论。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **DPO 实验中没有 KL 图表？**：成员们讨论了在 **Torchtune** 的 **DPO 实现实验** 期间没有使用 **KL 图表（KL plots）**。对于那些对 KL 图表使用感兴趣的人，可以参考 GitHub 上 [TRL 的 PPO 训练器中的 KL 图表](https://github.com/huggingface/trl/blob/34ebc4ccaf376c862a081ff4bb0b7e502b17b2fb/trl/trainer/ppo_trainer.py#L1133)。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Bitsandbytes 查询遇到意外问题**：一位成员报告在使用 **lighteval** 评估 **bitsandbytes** 模型时遇到困难，命令行工具无法识别 bitsandbytes 方法，而是请求 GPTQ 数据。
- **文档打包（Document Packing）中的效率追求者**：一位成员对 *Document Packing* 策略提出疑问，好奇该实现是用于实际生产还是仅仅作为一个简单的示例。他们强调了处理大型数据集时高效策略的重要性，并探究了 `tokenized_documents` 数据类型的具体细节。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Chip Huyen 现身 Databricks 活动**：著名工程师 Chip Huyen 正在参加 Databricks 峰会上的 [Mosaic 活动](https://mosaicx.events/events/june-10-2024-san-francisco-ca?events=main)，这为同行交流和建立联系提供了机会。与会者受邀见面并讨论当前的 MLOps 趋势。

---

**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该公会长时间没有动静，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该公会长时间没有动静，请告知我们，我们将将其移除。

---

**YAIG (a16z Infra) Discord** 没有新消息。如果该公会长时间没有动静，请告知我们，我们将将其移除。

---

# 第二部分：分频道详细摘要与链接

{% if medium == 'web' %}

### **Stability.ai (Stable Diffusion) ▷ #**[**general-chat**](https://discord.com/channels/1002292111942635562/1002292112739549196/1248713301504950344) (1091 条消息🔥🔥🔥):

- **讨论 Spotify 订阅方法**：多位成员交流了关于 Spotify 订阅方法的信息，其中一位提到他们免费获得了 Spotify Premium，而另一位则描述了涉及不同成本的优惠。
- **区域提示词（Regional Prompting）技术**：进行了一场关于区域提示词最佳方法的讨论，有人建议在 ComfyUI 中使用带有注意力掩码（attention masks）的 **IPAdapter**，另一个人则对如何使用 diffusers 实现此功能感到好奇。
- **对 SD3 的期待与日俱增**：许多人对即将发布的 **Stable Diffusion 3** (SD3) 表示兴奋和迫切，讨论了其功能和改进，如更好的提示词遵循能力和增强的图像生成能力。普遍共识是持谨慎乐观态度，并期待发布后的自定义微调（fine-tunes）和 Loras。
- **模型和 LoRas 训练的挑战**：一个反复出现的话题是在尝试使用 **ComfyUI** 和 **Kohya SS GUI** 等工具训练模型和 LoRas 时遇到的困难和技术障碍，用户正在排查安装问题并分享替代方案，如 OneTrainer。
- **对 ComfyUI 恶意软件的担忧**：强调了关于 ComfyUI 中一个**恶意节点**的警告，提醒用户该恶意软件可能会窃取敏感信息。这引发了关于在各种 UI 设置中使用自定义节点时如何保持安全性的讨论。

**提到的链接**：

- [ComfyUI](https://www.youtube.com/playlist?list=PLIF38owJLhR1EGDY4kOnsEnMyolZgza1x)：在本地电脑上使用 Stable Diffusion 模型创建 AI 艺术的更好方法。
- [未找到标题](https://www.amazon.com/CyberPowerPC-i7-13700F-GeForce-Windows-GXiVR8040A14/dp/B0CBL8N3FC/ref=sr_1_3?crid=B1SPXNSY9FP9&dib=eyJ2IjoiMSJ9.iSa-cmRbTiQMmAIZVtzRbC2enNa5c0i6IuDXu75F-aMp8ZVg3Fj0ip3n7fsie2GmqiboYwXMXi1gPh5wI7SaYRsul1UXmHIhsxvihGYvN28qo-FDtYlWQvEbQbSkcNaeACtsYNLYFZKVNaLRyYBMZe9q8Q3j_pZKTmFTZRbOt94s1ivYjUr88zR9PXmme6UJNKp5uqn8Kg8WqXlFtfmq3qQK5mrnjKbNrQSWm-5bKVw.vz2EOjrZJRKxVifZ-p0z671v4PT1kOVTQ2q4Pfpj3tk&dib_tag=se&keywords=RTX%2B4060%2BTi%2B16GB%2Bpc&qid=1718037521&sprefix=rtx%2B4060%2Bti%2B16gb%2Bp%2Caps%2C139&sr=8-3&th=1)：未找到描述
- [生成相似面孔的模型 | Civitai](https://civitai.com/articles/3653/models-producing-similar-looking-faces)：简介 这是我（可能还有许多其他人）注意到的一点，即某些自定义模型倾向于生成看起来相似的面孔...
- [仅使用 ComfyUI 进行 Lora 训练！！](https://www.youtube.com/watch?v=gt_E-ye2irQ)：我们向您展示如何完全在 ComfyUI 中训练 Loras...
- [VISION 预设包 #1 - @visualsk2](https://sk2visual.gumroad.com/l/spsjsz)：VisualSK2 的预设包系列（PC-移动端）。我日常使用的最佳 Lightroom 预设集合，旨在让拍摄作品具有电影感和一致的风格。包含什么？20 个预设...
- [Instagram 上的 madhav kohli：“NCR 的恐惧与厌恶……”](https://www.instagram.com/p/C6p8KgSSzo3/)：1.4 万次点赞，73 条评论 - mvdhav 于 2024 年 5 月 6 日发布：“NCR 的恐惧与厌恶……”。
- [Instagram 上的 Samuele “SK2” Poggi："[Vision III/Part. 4] ✨🤍 SK2• Fast day •](https://www.instagram.com/p/C6_kd_hoNGb/)

- [#photography #longexposure #explore #trending #explorepage"](https://www.instagram.com/p/C6_kd_hoNGb/): 3.3万次点赞，260条评论 - visualsk2 于 2024年5月15日: "[Vision III/Part. 4] ✨🤍 SK2• Fast day • #photography #longexposure #explore #trending #explorepage"。
- [在 AMD GPU 上安装并运行](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs): Stable Diffusion web UI。通过在 GitHub 上创建账号，为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。
- [Samuele “SK2” Poggi 在 Instagram 上: "[Vision IV/Part.6] 非常感谢 170,000 名粉丝 ✨🙏🏻 距离教程发布仅剩几天。](https://www.instagram.com/p/C781eUDoJ2h/)

[#grainisgood #idea #reels #framebyframe #photography #blurry #explorepage](https://www.instagram.com/p/C781eUDoJ2h/): 1.4万次点赞，122条评论 - visualsk2 于 2024年6月8日发布："[Vision IV/Part.6] 非常感谢 170,000 名关注者 ✨🙏🏻 距离教程发布仅剩几天。#gra..."
- [Reddit - 深入探索一切](https://www.reddit.com/r/StableDiffusion/comments/11t8mow/anime_style_controlnet_for_a1111_webui_available/): 未找到描述
- [Reddit - 深入探索一切](https://www.reddit.com/r/comfyui/comments/1dbls5n/psa_if_youve_used_the_comfyui_llmvision_node_from/): 未找到描述
- [[Bug]: 全新安装 - torch 安装错误 · Issue #467 · lshqqytiger/stable-diffusion-webui-amdgpu](https://github.com/lshqqytiger/stable-diffusion-webui-amdgpu/issues/467): 检查清单：禁用所有扩展后问题依然存在；在全新安装的 webui 上问题依然存在；问题是由扩展引起的，但我认为这是 webui 的一个 Bug 导致的...
- [https://preview.redd.it/comfyui-sdxl-my-2-stage-workflows-v](https://www.reddit.com/media?url=https%3A%2F%2Fpreview.redd.it%2Fcomfyui-sdxl-my-2-stage-workflows-v): 未找到描述
- [公告：如果你使用了来自 u/AppleBotzz 的 ComfyUI_LLMVISION 节点，你已被黑客入侵](https://old.reddit.com/r/comfyui/comments/1dbls5n/psa_if_youve_used_the_comfyui_llmvision_node_from/l7zfbj4/): 那些混蛋为了报复我，泄露了他们从我这里窃取的所有密码。如果有人愿意帮忙清理...
- [什么 GIF - 猫咪惊讶震惊 - 发现并分享 GIF](https://tenor.com/view/cat-surprised-shookt-what-sexcuse-me-gif-9103855): 点击查看 GIF
- [LoRA 模型入门：Stable Diffusion 中的定义、来源及使用方法](https://youtu.be/ZHVdNeHZPdc?si=iJlH5WZUOiNfbiO9): 在本视频中，我们将了解什么是 LoRA (Low-Rank Adaptation) 模型，以及为什么它们对于任何对低尺寸模型和高质量输出感兴趣的人来说都至关重要...
- [imgur.com](https://imgur.com/py3eKHA): 在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过幽默的笑话、热门迷因、有趣的 GIF、励志故事、病毒视频等来振奋你的精神...
- [GitHub - comfyanonymous/ComfyUI: 最强大且模块化的 Stable Diffusion GUI、API 和后端，采用图表/节点界面。](https://github.com/comfyanonymous/ComfyUI): 最强大且模块化的 Stable Diffusion GUI、API 和后端，采用图表/节点界面。 - comfyanonymous/ComfyUI
- [GitHub - lks-ai/ComfyUI-StableAudioSampler: ComfyUI 节点中的新 Stable Diffusion Audio Sampler 1.0。来制作一些节拍吧！](https://github.com/lks-ai/ComfyUI-StableAudioSampler): ComfyUI 节点中的新 Stable Diffusion Audio Sampler 1.0。来制作一些节拍吧！ - lks-ai/ComfyUI-StableAudioSampler
- [可灵大模型](https://kling.kuaishou.com/mobile): 未找到描述
- [首页 :: AiTracker](https://aitracker.art/): 未找到描述
- [GitHub - bmaltais/kohya_ss](https://github.com/bmaltais/kohya_ss): 通过在 GitHub 上创建账号来为 bmaltais/kohya_ss 的开发做出贡献。
- [Reddit - 深入探索一切](https://www.reddit.com/r/StableDiffusion/comments/193hqkz/lora_training_directly_in_comfyui/): 未找到描述
- [Hard Muscle - v1.0 | Stable Diffusion Checkpoint](https://tensor.art/models/654286272942196700): 未找到描述
- [Hard Muscle - SeaArt AI 模型](https://www.seaart.ai/models/detail/0e5b32eb19562e304d29771ad3898af5): 未找到描述
- [使用 OneTrainer 和 AI 生成进行懒人式 LoRA 制作 | Civitai](https://civitai.com/articles/4789/lazy-lora-making-with-onetrainer-and-ai-generation): 简介：我是 LoRA 制作的新手，很难找到好的指南。要么细节不够，要么细节太太太冗长。所以这...
- [DnD 地图生成器 - v3 | Stable Diffusion Checkpoint | Civitai](https://civitai.com/models/5012/dnd-map-generator): 该模型基于各种 D&D 战斗地图进行训练。如果你有改进建议，请告诉我。使用负面提示词（negative prompt）："grid" 来改进...
- [https://preview.redd.it/comfyui-sdxl-my-2-stage-workflows-v0-mdb012l64lfb1.png?width=2486&format=png&auto=webp&s=e72a2bed93c8fd3d9049ea3a0969aa8ad80f3158](https://www.reddit.com/media?url=https%3A%2F%2Fpreview.redd.it%2Fcomfyui-sdxl-my-2-stage-workflows-v0-mdb012l64lfb1.png%3Fwidth%3D2486%26format%3Dpng%26auto%3Dwebp%26s%3De72a2bed93c8fd3d9049ea3a0969aa8ad80f3158): 未找到描述
- [Reddit - 深入探索一切](https://www.reddit.com/r/StableDiffusion/comments/113vceb/controlnet_for_anime_line_art_coloring/): 未找到描述
- [ControlNet：完整指南 - Stable Diffusion 艺术](https://stable-diffusion-art.com/controlnet/): ControlNet 是一种神经网络，通过添加额外条件来控制 Stable Diffusion 中的图像生成。详情请参阅文章《添加...》

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1248719324550660157) (905 条消息🔥🔥🔥):

- **AI 驱动的旅行规划困境**：用户对 AI 旅行规划表示沮丧，尤其是生成准确的航班详情。一位用户指出，“无论我怎么尝试，它都不会告诉我机票详情” ([来源](https://www.perplexity.ai/search/Plan-me-a-.WikdPp_SjyQ0v4jL9_Zbg#0))。
- **Perplexity AI 的增强功能**：成员们讨论了新的 Pro 搜索功能，该功能提供多步搜索，提高了结果的相关性 ([来源](https://www.perplexity.ai/search/Find-out-the-o4J2zXiaQXWZAkNu6RuVuA))。
- **Perplexity Pages 索引问题**：多位用户报告他们的 Perplexity Pages 被取消索引，怀疑这仅影响非员工撰写的文章 ([来源](https://discord.com/channels/1047197230748151888/1054944216876331118/1248401923938320526))。
- **关于 GPT-4 模型的辩论**：成员们辩论了 GPT-4o 模型的准确性和幻觉问题，指出它有时会错误地修正为 GPT-4。用户分享道，“GPT-4o 不知道 GPT-4o 是个什么东西” ([来源](https://www.perplexity.ai/search/What-is-the-21RKWtnqTg6a37M8k.sIow))。
- **Rabbit 设备争议**：用户警告不要购买 Rabbit R1 设备，根据用户体验和 Coffeezilla 视频等调查，将其贴上骗局的标签 ([来源](https://discord.com/channels/1047197230748151888/1047649880161652827/1240306540909691071))。

**提到的链接**：

- [热门 AI 搜索引擎 Perplexity 直接剽窃新闻媒体内容](https://www.forbes.com/sites/sarahemerson/2024/06/07/buzzy-ai-search-engine-perplexity-is-directly-ripping-off-content-from-news-outlets/)：这家被誉为 Google 挑战者的初创公司正在重新发布包括《福布斯》和《彭博社》在内的多家出版物的独家报道，且归属说明不足。
- [支持的模型](https://docs.perplexity.ai/docs/model-cards)：未找到描述
- [朝鲜向韩国发送另一波垃圾气球 | CNN](https://www.cnn.com/2024/06/09/asia/north-korea-balloons-response-intl-hnk/index.html)：未找到描述
- [Nwmsrocks Northwest Motorsport GIF - Nwmsrocks Northwest Motorsport Pnw - 发现并分享 GIF](https://tenor.com/view/nwmsrocks-northwest-motorsport-pnw-pacific-northwest-toyota-gif-20681791)：点击查看 GIF
- [You Know It Wink GIF - You Know It Wink The Office - 发现并分享 GIF](https://tenor.com/view/you-know-it-wink-the-office-michael-scott-steve-carell-gif-17547389)：点击查看 GIF
- [GPT-4o API 的成本是多少，支持的上下文长度是多少？](https://www.perplexity.ai/search/What-is-the-21RKWtnqTg6a37M8k.sIow)：使用 GPT-4o API 的成本如下：文本输入：每 100 万 tokens 5 美元；文本输出：每 100 万 tokens 15 美元；视觉处理（图像...）
- [找出 VGT 中的前 10 支股票，然后找出它们的名称...](https://www.perplexity.ai/search/Find-out-the-o4J2zXiaQXWZAkNu6RuVuA)：先锋信息技术 ETF (VGT) 中的前 10 支股票是：1. Microsoft Corporation (MSFT) - 17.30% 2. Apple Inc. (AAPL) - 15.29% 3. NVIDIA...
- [Tinytim Tim GIF - Tinytim Tim Poor - 发现并分享 GIF](https://tenor.com/view/tinytim-tim-poor-child-boy-gif-7331148)：点击查看 GIF
- [今日赛马卡 | At The Races](https://www.attheraces.com/racecards)：您可以在 At The Races 赛马卡上查看所有参赛马匹和骑师，其中包含您需要了解的一切信息，包括最新战绩、贴士、统计数据和育种信息以及最新赔率...
- [Starship 第四次测试：成功！](https://www.perplexity.ai/search/Starship-Test-4-l5qs.WoNSkOmJIXJU85gMA)：<scratchpad> 来自提示的关键信息：SpaceX 于 2024 年 6 月 6 日进行了 Starship 发射系统的第四次试飞。飞行器起飞于...
- [AI Playground | 并排比较顶级 AI 模型](https://sdk.vercel.ai/)：聊天并比较 OpenAI GPT, Anthropic Claude, Google Gemini, Llama, Mistral 等。
- [帮我规划一次日本旅行，并提供完整的旅行计划...](https://www.perplexity.ai/search/Plan-me-a-.WikdPp_SjyQ0v4jL9_Zbg#0)：出发城市：美国洛杉矶；旅行日期：6 月 15 日（出发）；兴趣：文化遗址、自然和户外活动、购物、美食餐饮...
- [咖啡杯 GIF - Coffee Cup Shake - 发现并分享 GIF](https://tenor.com/bgRSf.gif)：点击查看 GIF
- [Perplexity](https://www.perplexity.ai/collections/scratchpadthink-wBPEohuUQH6tz5qMlH4F7g)：未找到描述
- [2024 年 6 月 7 日 21:00 巴斯赛马](https://docs.google.com/document/d/14gcrycsKEHY3uMNkeEYttCMW3u7nm1_HkaaLTOSrR6Y/edit?usp=sharing)：2024 年 6 月 7 日 21:00 巴斯 Mitchell & Co 让赛。综合稳健系统分析。让我们将综合稳健系统应用于比赛，结合步速数据模式和剂量...

- [帮我规划一次日本旅行，然后给我完整的旅行计划，制定所有...](https://www.perplexity.ai/search/Plan-me-a-IQGF0vuqSjy66chq8Bp0yA#0)：这里为您推荐一份为期 7 天的日本旅行行程，包含所有已规划的细节：第 1 天：抵达东京，入住东京文华东方酒店...
- [帮我规划一次日本旅行，然后给我完整的旅行计划，制定所有...](https://www.perplexity.ai/search/Plan-me-a-dFjEdP5yTs.SQ1hn4HpIEQ#0)：这里为您推荐一份从洛杉矶出发的 7 天日本旅行行程，包含您要求的所有细节：预订从洛杉矶出发的往返航班...
- [2024年6月7日 20:10 古德伍德 (Goodwood)](https://docs.google.com/document/d/1rj-BAeTmAc02hSATc_wuRwuO5o5ID5gMi8G_ugdAPcU/edit?usp=sharing)：2024年6月7日 20:10 古德伍德全面稳健系统分析。1. Skysail (279) 状态：22/1，在桑当 (Sandown) 的让赛中（10f，好地至软地）获得 14 名中的第 10 名。休赛 9 个月。记录：赛道与距离 (CD)：...
- [未找到标题](https://aistudio.google.com/app/prompts/new_chat)：未找到描述
- [Perplexity](https://www.perplexity.ai/search/Plan-me)：未找到描述
- [帮我规划一次日本旅行，然后给我完整的旅行计划，制定所有...](https://www.perplexity.ai/search/Plan-me-a-H66hf48ARB.yoRq816KO5w#0)：规划一次从洛杉矶出发的全面日本旅行涉及多个步骤，包括预订航班、住宿以及规划每日活动...
- [Reddit - 深入探索一切](https://www.reddit.com/r/ChatGPT/s/KrhcqUpEuq)：未找到描述
- [GTD 组织与任务完成方法 (V2)](https://www.perplexity.ai/search/Fully-review-the-0RmThWSGTFC4i4kRiQStdg)：这是对 Getting Things Done (GTD) 方法的全面回顾，包括截至 2024 年的最新更新和最佳实践：Getting Things Done (GTD)...
- [GTD 组织与任务完成方法](https://www.perplexity.ai/page/The-GTD-Method-XNa.pCFWT0eNitfO40bqrw)：Getting Things Done (GTD) 是一种流行的个人生产力系统，可帮助个人有条理地管理任务、项目和承诺...
- [未找到标题](https://aistudio.google.com/官方)：未找到描述
- [Config 2024 | 会议详情](https://config.figma.com/agenda/session?session_id=8cda6eacbfe4&lang=en)：2024 年将是迄今为止最令人兴奋的 Config！欢迎于 6 月 26 日至 27 日亲临旧金山或在线参加。

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1248858458351210537) (26 条消息🔥):

- **波音 Starliner 在与 ISS 对接期间面临 RCS 推进器问题**：在 Starliner 接近 ISS 期间，其 28 个 RCS 推进器中有 5 个发生故障，导致航天器错过了首次对接尝试。[NASA 报告](https://www.perplexity.ai/page/Starliner-Docks-with-EzaalHkvRsqzKROOP2gjuw)称，受影响推进器的传感器数值略高于正常限制。
- **SpaceX 成功着陆 Starship**：SpaceX 在其 Starship 巨型火箭的第四次试飞中取得了重大里程碑。该任务的第一级和第二级均成功完成海上溅落，标志着向完全可重复使用的火箭系统迈进。[阅读更多](https://www.perplexity.ai/page/SpaceX-successfully-lands-PCd62nm2Sr.BZGUHFmHP2w)。
- **大规模网络攻击袭击 Niconico 服务**：Niconico 服务遭受大规模网络攻击，导致暂时关闭。[Dwango 正在进行](https://www.perplexity.ai/page/niconico-AEzyLaH.QueAghyJD1pgxw)紧急维护，但预计完全恢复需要数天时间。
- **以色列从哈马斯手中营救人质**：以色列军队进行了一次大胆的白天突袭，营救了哈马斯在加沙扣押的四名人质。该任务是加沙中部重大攻势的一部分，据报道导致了大量人员伤亡。[了解更多](https://www.perplexity.ai/page/Israel-Rescues-4-NBpiRY4SS9CgsAYrTYWG1w)。
- **华为 Ascend 910B AI 芯片挑战 Nvidia A100**：华为新款 Ascend 910B AI 芯片已成为 Nvidia A100 的强劲对手，在训练 LLM 方面的效率达到了后者的 80%，并在其他测试中超越了它。[该芯片的发布](https://www.perplexity.ai/page/Huaweis-New-AI-4EKcpjWjR3W3SuA38fyTGw)引发了技术和地缘政治辩论。

**提到的链接**：

- [YouTube](https://www.youtube.com/embed/hhX9EKtInok)：未找到描述
- [YouTube](https://www.youtube.com/embed/nR2GnTAqPrI)：未找到描述
- [声调语言中的歌唱是如何进行的？](https://www.perplexity.ai/search/How-does-singing-3icDgrnMSAixgVGK9r3TbQ)：在中文、越南语和泰语等声调语言中唱歌面临着独特的挑战，因为在言语中传达意义的声调需要...

- [如果我用英文提问，LLM 是否更有可能获得最佳答案？](https://www.perplexity.ai/search/Is-the-LLM-Hg9aJF3jSLqpDviP1A1Y7g): 是的，如果你用英文提问，多语言大语言模型 (LLM) 更有可能提供最佳答案。这是因为这些模型是……
- [国家树木园 (The National Arboretum)](https://www.perplexity.ai/search/The-National-arboretum-Nldn1YMVTeWE3HSq6nA6xA): “National Arboretum” 一词可以指代世界上几个著名的树木园，每个都有其独特的历史、特色和重要性。以下是……
- [Mass Conflict: Ignition - RTS 游戏](https://www.perplexity.ai/search/Mass-Conflict-Ignition-haek5XNYSS2qdl3fo9AXwg): Mass Conflict: Ignition 是一款即将推出的即时战略 (RTS) 游戏，由独立工作室 Misclick Games 开发。它的目标是成为……的精神续作。
- [修改为更稳健的文章。使用包含的 <scratchpad-think>……](https://www.perplexity.ai/search/Revise-into-a-t4taOXiIRU.nnMM4GwkMzQ#0): 这是我尝试使用提供的 scratchpad 框架将文章修改为更稳健版本的尝试：<scratchpad> 来自……的关键信息
- [WWDC 2024 发布了什么？](https://www.perplexity.ai/page/What-Was-Introduced-9M8PxU85Tg2kA.6ubApgJw): 苹果的 WWDC 2024 主旨演讲揭晓了一系列令人兴奋的更新和功能，重点强调了人工智能集成和软件……
- [Miss AI 竞赛](https://www.perplexity.ai/page/The-Miss-AI-MC.WjDXVS8OV9WX5veD6xQ): Miss AI 竞赛由 Fanvue 组织，是世界 AI 创作者奖 (WAICAs) 的一部分，是世界上第一个针对 AI 生成模型的选美比赛。这……
- [华为新 AI 芯片传闻](https://www.perplexity.ai/page/Huaweis-New-AI-4EKcpjWjR3W3SuA38fyTGw): 华为新款 Ascend 910B AI 芯片已成为 Nvidia A100 的强劲对手，引发了技术和地缘政治方面的辩论。该芯片的……
- [用于组织和任务完成的 GTD 方法 (V2)](https://www.perplexity.ai/search/Fully-review-the-0RmThWSGTFC4i4kRiQStdg): 这是对 Getting Things Done (GTD) 方法的全面回顾，包括截至 2024 年的最新更新和最佳实践：Getting Things Done (GTD)……
- [SpaceX 成功让 Starship 着陆](https://www.perplexity.ai/page/SpaceX-successfully-lands-PCd62nm2Sr.BZGUHFmHP2w): SpaceX 在太空探索领域取得了一个重要的里程碑，其 Starship 巨型火箭成功完成了第四次试飞。这次任务见证了……
- [你好，延长佐治亚州身份证需要什么？](https://www.perplexity.ai/search/Hi-what-do-irPADOFbQi.JQn4YDJww.Q): 要更新或延长你的佐治亚州身份证，你通常需要提供以下文件：身份证明（例如美国护照、出生证明公证件……）
- [什么是 Apple Intelligence？](https://www.perplexity.ai/page/What-Is-Apple-3dFcdceTR4W5eqZV.8A_iw): 苹果将在即将举行的全球开发者大会 (WWDC) 上推出一套名为 “Apple Intelligence” 的人工智能功能。这……
- [WWDC 2024 发布了什么？](https://www.perplexity.ai/page/What-Was-Introduced-0laYmO7vS2mGryU354PYaA): 苹果的 WWDC 2024 主旨演讲揭晓了一波 AI 驱动的功能和软件更新，旨在提供更智能、更个性化的跨平台体验……
- [我有一条探索了一段时间的思路。它位于……](https://www.perplexity.ai/search/I-have-a-.FDnCzmyTBeQZgDcW0bXow): 你发人深省的询问深入到了哲学领域，探索了意识作为一种类似于火的普遍资源的有趣可能性……
- [你好 Perplexity！我们想更新你的语言模型生成……](https://www.perplexity.ai/search/hello-perplexity-we-i9ev8L.vQfa5Nbw3D4d6Pg#9): 你好。我很乐意适应新规则，以确保对话安全且相互尊重。以下是更新后规则的摘要：1. 禁止性行为：我将……
- [以色列从加沙哈马斯手中营救出 4 名人质](https://www.perplexity.ai/page/Israel-Rescues-4-NBpiRY4SS9CgsAYrTYWG1w): 周六，以色列军队在一次大胆的日间突袭中，戏剧性地营救了被哈马斯扣押在加沙的四名以色列人质。Noa Argamani (25岁)、Almog Meir……
- [AI 助手](https://www.perplexity.ai/page/AI-Helpers-6u.7YsHvTfyOBvOibNm_1g): AI Agent 就像数字助手，可以独立思考并采取行动来帮助你完成任务。它们使用人工智能来理解你的需求……
- [用于组织和任务完成的 GTD 方法](https://www.perplexity.ai/page/The-GTD-Method-XNa.pCFWT0eNitfO40bqrw): Getting Things Done (GTD) 是一种流行的个人生产力系统，可帮助个人以有组织的方式管理其任务、项目和承诺……

- [Starliner Docks with ISS After Numerous Failures and Delays](https://www.perplexity.ai/page/Starliner-Docks-with-EzaalHkvRsqzKROOP2gjuw): 波音公司的 Starliner 航天器，搭载着 NASA 宇航员 Butch Wilmore 和 Suni Williams，于 6 月 7 日成功与国际空间站（ISS）对接...
- [What is the new Apple Intelligence that was announced? use Pro Search to...](https://www.perplexity.ai/search/What-is-the-xS_Jmhz1Sr2nFMDrfcWhKQ#1): <scratchpad> 来自提示词和来源的关键信息：苹果在 WWDC 2024 上发布了一套名为 "Apple Intelligence" 的新 AI 系统。它将带来 AI...
- [2024年6月のniconico大規模障害についてまとめ](https://www.perplexity.ai/page/niconico-AEzyLaH.QueAghyJD1pgxw): 包括 niconico 动画在内的 niconico 服务遭受了大规模网络攻击，自 2024 年 6 月 8 日凌晨起暂时停止服务。运营方 Dwango 正在进行紧急维护以尽量减少影响，但至少在周末期间难以恢复。
- [Dawn of Generation AI](https://www.perplexity.ai/page/Dawn-of-Generation-zq0PyDIJTl.PkPMDkhGR7g): 人工智能（AI）的迅速崛起正在改变创意产业的格局，引发了关于人类未来...的复杂问题。
- [Comprehensive Guide to Prompt Engineering](https://www.perplexity.ai/page/Comprehensive-Guide-to-ia523Zn_QPmZnfQRD63Lgw): Prompting 作为一种与语言模型交互的技术，已成为生成式 AI 时代的一项关键技能。本综合指南探讨了...

---

### **Perplexity AI ▷ #**[**pplx-api**](https://discord.com/channels/1047197230748151888/1161802929053909012/1248764383870586941) (19 messages🔥):

- **关于 return_images 参数的查询**：一位用户询问 *return_images* 参数是否是 LLaVA 返回图像的方法。关于此话题没有提供进一步的信息。
- **获取与 Perplexity 网页版相同质量的结果**：一位成员询问应使用哪种模型来获得与 Perplexity 网页版相同质量的结果。另一位成员回复了 [Discord 资源](https://discord.com/channels/1047197230748151888/1161802929053909012/1236012953266946058)链接和额外的[指南](https://discord.com/channels/1047197230748151888/1223947903941349476/1223949540487594104)。
- **Perplexity API 无法使用 Embeddings**：一位用户询问关于使用 Perplexity API 生成 Embeddings 的事宜。回复澄清道：*"Hi, no it’s not possible"*（你好，目前无法实现）。
- **通过私信解决 API 积分问题**：一位用户多次尝试解决购买订阅后 API 积分未到账的问题。建议该用户通过私信特定账号并提供其电子邮件地址来解决。
- **寻求将外部网页搜索集成到自定义 GPT 的帮助**：一位用户在将外部网页搜索能力（如 Serper, Tavily 和 Perplexity API）集成到自定义 GPT Actions 以提高搜索准确性时遇到挑战。他们参考了一篇过时的 [Perplexity API 文章](https://docs.perplexity.ai/discuss/65edc94038fa40001045873c)寻求帮助。

**提到的链接**：[Perplexity API with Custom GPT](https://docs.perplexity.ai/discuss/65edc94038fa40001045873c): 未找到描述

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**general**](https://discord.com/channels/1238365980128706560/1238365980128706563/1248743982691782686) (64 messages🔥🔥):

- **Qwen2 Instruct 中的错误信息**：一位成员指出 Qwen2 Instruct 中存在令人担忧的审查制度和明显的错误信息，特别是在英文和中文回答之间存在细微差异。他们计划在 [Hugging Face 博客文章](https://huggingface.co/blog/leonardlin/chinese-llm-censorship-analysis)中分享更多内容。
- **Llama-3 Abliteration**：成员们讨论了在不同的 LLM 上使用 Abliterator 库来减轻拒绝回答的情况，并分享了 [FailSpy 的 Llama-3-70B-Instruct](https://huggingface.co/failspy/llama-3-70B-Instruct-abliterated/blob/main/ortho_cookbook.ipynb) 和 [Sumandora 的项目](https://github.com/Sumandora/remove-refusals-with-transformers/tree/master)链接。
- **微调视觉模型**：成员们对微调名为 Moondream 的视觉语言模型表现出兴趣，并分享了一个相关的 [GitHub notebook](https://github.com/vikhyat/moondream/blob/main/notebooks/Finetuning.ipynb) 以指导该过程。
- **"GPT-2 The Movie" 发布**：成员们对 YouTube 视频 [“GPT-2 The Movie”](https://www.youtube.com/watch?v=l8pRSuU81PU) 的发布感到兴奋，该视频涵盖了从零开始复现 GPT-2 (124M) 的全过程。该视频因其内容的详尽而受到高度赞赏。
- **模型大小启发式方法**：一位成员询问如何根据任务复杂度选择微调的模型大小，并暗示了开发启发式方法或对不同模型能力（例如 8B vs 70B）建立直觉对于简化快速原型设计的重要性。

**提到的链接**：

- [augmxnt/Qwen2-7B-Instruct-deccp · Hugging Face](https://huggingface.co/augmxnt/Qwen2-7B-Instruct-deccp): 未找到描述
- [Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study](https://arxiv.org/abs/2404.10719): 基于人类反馈的强化学习 (RLHF) 是目前最广泛用于使大语言模型 (LLMs) 与人类偏好对齐的方法。现有的 RLHF 方法大致可以分为...
- [augmxnt/deccp · Datasets at Hugging Face](https://huggingface.co/datasets/augmxnt/deccp): 未找到描述
- [Preference Tuning LLMs with Direct Preference Optimization Methods](https://huggingface.co/blog/pref-tuning): 未找到描述
- [Implementing Learnings from the "Mastering LLMs" Course](https://llms-in-prod.beehiiv.com/p/mastering-llms-implement-evals): 我如何利用 Mastering LLMs 课程的见解改进生产环境中的 LLM
- [An Analysis of Chinese LLM Censorship and Bias with Qwen 2 Instruct](https://huggingface.co/blog/leonardlin/chinese-llm-censorship-analysis): 未找到描述
- [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU): 我们从零开始复现 GPT-2 (124M)。这段视频涵盖了整个过程：首先我们构建 GPT-2 网络，然后优化其训练，使其真正...
- [ortho_cookbook.ipynb · failspy/llama-3-70B-Instruct-abliterated at main](https://huggingface.co/failspy/llama-3-70B-Instruct-abliterated/blob/main/ortho_cookbook.ipynb): 未找到描述
- [moondream/notebooks/Finetuning.ipynb at main · vikhyat/moondream](https://github.com/vikhyat/moondream/blob/main/notebooks/Finetuning.ipynb): 微型视觉语言模型。通过在 GitHub 上创建账号来为 vikhyat/moondream 的开发做出贡献。
- [GitHub - petergpt/Fine-Tuning-Memorisation-Experiement-GPT-35: Use of a fine-tuned model](https://github.com/petergpt/Fine-Tuning-Memorisation-Experiement-GPT-35): 微调模型的使用。通过在 GitHub 上创建账号来为 petergpt/Fine-Tuning-Memorisation-Experiement-GPT-35 的开发做出贡献。
- [Five Years of Sensitive Words on June Fourth](https://chinadigitaltimes.net/2016/06/five-years-sensitive-words-june-fourth/): “那一年”、“这一天”、“今天”。在往年 6 月 4 日（这一天因北京对示威者的军事镇压而被铭记）前后，这三个短语在微博搜索中都被屏蔽了...
- [1989 Tiananmen Square protests and massacre - Wikipedia](https://en.wikipedia.org/wiki/1989_Tiananmen_Square_protests_and_massacre#Naming): 未找到描述
- [GitHub - FailSpy/abliterator: Simple Python library/structure to ablate features in LLMs which are supported by TransformerLens](https://github.com/FailSpy/abliterator): 用于消融由 TransformerLens 支持的 LLMs 特性的简单 Python 库/结构 - FailSpy/abliterator
- [GitHub - Sumandora/remove-refusals-with-transformers: Implements harmful/harmless refusal removal using pure HF Transformers](https://github.com/Sumandora/remove-refusals-with-transformers/tree/master): 使用纯 HF Transformers 实现有害/无害拒绝移除 - Sumandora/remove-refusals-with-transformers
- [MopeyMule-Induce-Melancholy.ipynb · failspy/Llama-3-8B-Instruct-MopeyMule at main](https://huggingface.co/failspy/Llama-3-8B-Instruct-MopeyMule/blob/main/MopeyMule-Induce-Melancholy.ipynb): 未找到描述

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**workshop-1**](https://discord.com/channels/1238365980128706560/1239614536298795121/1248971713555730543) (4 条消息):

- **感到落后**: 一位成员分享了因为旅行和度假而感到进度落后的心情，这引起了群组中其他人的共鸣。
- **作业 1 使用案例**: 一位成员发布了作业 1 的几个使用案例：大规模文本分析、为低资源印度语系进行模型微调、创建一个模仿其对话风格的个人 LLM，以及为特定用例构建一个评估器/批判者 LLM。他们引用了 [Ragas-critic-llm-Qwen1.5-GPTQ](https://huggingface.co/explodinggradients/Ragas-critic-llm-Qwen1.5-GPTQ) 作为灵感来源。
- **使用 RAG 而非微调进行更新**: 针对有关使用新政策更新 LLMs 的问题，会议强调 **RAG (Retrieval-Augmented Generation)** 优于微调。微调需要删除过时的训练数据并整合新政策，这比 RAG 更复杂且效率更低。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**🟩-modal**](https://discord.com/channels/1238365980128706560/1241044231829848125/1248758473546666138) (25 messages🔥):

- **解决积分分配问题**：多名用户报告了积分访问问题。Charles 引导他们查看[此处](https://discord.com/channels/1238365980128706560/1241044231829848125/1242137932014555230)的信息，并提供其邮箱以便在需要时提供进一步帮助。
- **Docker Container 设置的变通方法**：一位用户因 `modal setup` 命令需要 Web 浏览器而在 Docker 容器设置中遇到困难。Charles 建议使用从 Web UI 预生成的 token 执行 `modal token set` 作为解决方案。
- **用于工作区管理的 Modal Environments**：一位用户询问如何在多个工作区运行不同的 demo。Charles 建议使用 [Modal environments](https://modal.com/docs/guide/environments)，以便在不更改代码的情况下部署多个应用实例。
- **在本地网络挂载 Modal Volumes**：一位用户询问有关网络挂载 Modal volumes 的问题。Charles 建议使用 `modal volume` 和 `modal shell` 命令进行本地操作和探索。
- **GPU 配额超限请求**：Santiago 询问有关超过 GPU 限制的问题。Charles 请他在 Modal Slack 上私信，并提供有关其需求的详细信息。

**提到的链接**：[Environments](https://modal.com/docs/guide/environments)：Environments 是工作区的子划分，允许你在不更改代码的情况下，为不同目的在多个实例中部署同一个应用（或一组应用）。典型的使用场景包括...

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**learning-resources**](https://discord.com/channels/1238365980128706560/1241089743933149204/) (1 messages):

yxzwayne: [https://arxiv.org/pdf/2402.17193](https://arxiv.org/pdf/2402.17193) 这将很难消化。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**hugging-face**](https://discord.com/channels/1238365980128706560/1241141471814488115/1248944044524961803) (55 messages🔥🔥):

- **澄清积分误解**：一位用户向另一位用户澄清，他们收到的积分不会像最初误解的那样在 6 月底过期。另一位用户确认了这一点，说：“不，它们不会过期。”
- **Mistral 7B 部署难题**：一位成员在部署受限的 Mistral 7B 模型时遇到困难，尽管拥有访问权限，但仍收到详细的错误消息。添加环境变量 `HF_TOKEN` 的建议成功解决了该问题。
- **填表困惑**：多名用户报告称，尽管按时填写了表格，但未收到 Hugging Face 积分。他们被要求提供 HF 邮箱和用户名进行验证。
- **填表提醒**：为错过第一轮的人发布了一份新表格，截止日期为一周。提醒成员填写表格并包含特定详细信息以领取积分。
- **调试支持与 Token**：调试环节以幽默的方式圆满结束。用户感谢了“调试羊驼”表情以及解决问题的 HF token 建议，并表示：“通过将 `HF_TOKEN` 作为环境变量添加到 endpoint，它现在可以工作了！”

**提到的链接**：

- [Hugging Face – 构建未来的 AI 社区。](https://huggingface.co/settings/billing)：未找到描述
- [Hugging Face 积分申请](https://forms.gle/C97btM1E99Q69GCMA)：在我们为您申请 🤗 HF 积分以使用 https://huggingface.co 的付费服务之前，我们需要完成几件简单的事情！如有任何疑问，请联系 website@huggingface.co。...

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**replicate**](https://discord.com/channels/1238365980128706560/1241163904927666287/1248794052854419548) (7 messages):

- **Replicate 额度问题已解决**：用户报告了 Replicate, OpenPipe, BrainTrust 和 OpenAI 的额度缺失问题。在核实用户详情后，一名成员确认他们已收到 Replicate 额度。
- **Replicate 邀请困惑**：一名成员收到了 Replicate 的邀请，但不确定是否需要设置账单信息才能看到额度。另一名用户被引导至特定的助手处寻求进一步帮助。
- **Replicate 宣布支持安全输入类型**：Hamelh 分享了来自 [Replicate 的推文](https://x.com/replicate/status/1800226513721368606)，宣布支持一种新的 **secret** 输入类型，用于安全地传递敏感值，包括密码和 API tokens。此更新包括将权重下载和上传到 Hugging Face、将指标和 artifacts 上传到 Weights and Biases，以及在 S3/R2/GCS 上存储文件输出等功能。

**提及的链接**：[来自 Replicate (@replicate) 的推文](https://x.com/replicate/status/1800226513721368606)：我们现在支持一种新的 **secret** 输入类型，用于安全地向模型传递敏感值，如密码和 API tokens。现在你可以：- 下载权重并上传到 Hugging Face - 上传指标...

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**langsmith**](https://discord.com/channels/1238365980128706560/1241167367040405544/1248771417928106116) (9 messages🔥):

- **账单设置令用户困惑**：多位用户对需要设置账单账户才能解锁免费额度表示困惑，其中一位表示：“*必须先 1. 设置账单账户 2. 接收/解锁免费额度，这并不直观。*”他们认为有必要清晰地沟通这一流程。
- **设置账单后仍未收到额度**：用户 luisramirez9603、hoesanna 和 dml4680 指出，尽管设置了账单账户，他们仍未收到额度。他们提供了自己的组织 ID 以寻求进一步帮助。
- **支持团队手动调整额度**：Jessou_49081 向用户更新了额度问题的进展，提到为部分用户进行了手动调整，并与其他用户通过邮件沟通以解决问题。“*我已经进入系统并为你们添加了这些额度 *\*”表明支持团队采取了主动措施。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**ankurgoyal_textsql_llmevals**](https://discord.com/channels/1238365980128706560/1242222674835538012/1249238023032275036) (1 messages):

- **Text-to-SQL 基准测试强调 GroupBy 但忽略了 Filter/WHERE 子句**：观察到基准测试的重点在于 **GroupBy 案例**，而非 Filter/WHERE 子句中涉及 **高基数列 (high cardinality columns)** 的案例。一个例子是，根据过滤条件查询 *AWS Simple Storage Service* 与 *Amazon Simple Storage Service* 时结果存在差异。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**berryman_prompt_workshop**](https://discord.com/channels/1238365980128706560/1242223275463938221/1249752941796266115) (2 messages):

- **探索 prompt 模板和 Pydantic 模型**：一位成员询问是否应始终使用基于 prompt 的模板来结构化输出，并寻求澄清 Pydantic 模型适用的场景。他们想知道 Pydantic 模型是否可以与 chat 模型一起使用，还是仅限于 completions API。
- **LLaMA-3 prompt 的推理难题**：一位用户解释了他们在 LLaMA-3 模型上遇到的挑战，该模型在推理任务上表现不佳。尽管它能识别患者的年龄和疾病的年龄范围，但无法得出患者年龄处于该范围内的结论，因此询问 prompt engineering 是否能改善这种推理能力。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**whitaker_napkin_math**](https://discord.com/channels/1238365980128706560/1242223332695478332/1249554669227741216) (1 messages):

- **讲座被赞誉为非常出色**：一位用户对最近观看的讲座质量表示赞赏，称其“真的非常好”。他们感谢了会议主持人，并提到按照建议进行多次观看和实际操作的必要性。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**workshop-4**](https://discord.com/channels/1238365980128706560/1242223495673286737/1248863262338973696) (7 messages):

- **Replicate 的 vLLM Model Builder 已找到**：一位用户最初在 Replicate 的 UI 中找不到 vLLM Model Builder。随后他们更新了可用的 [GitHub 链接](https://github.com/replicate/cog-vllm)。
- **Workshop 4 录制问题**：有反馈称 Workshop 4 的录制在 2:30 处中断。经澄清，该工作坊达到了 Zoom 文件长度的最大限制，但最后 12 分钟的 Q&A 可以在[此处](https://us06web.zoom.us/rec/share/cewtLttl3vXMcSuFYrF7BXVkqBjpT937hmi9xLoQ0Nvd3Xac_F0ad9lVQH80o3Li.wzg8h6wUKjCaZOiY?startTime=1717529840000)观看，密码为：Xf0yc\*rx。
- **Modal 平台的额外额度**：用户讨论了如何在 Modal 平台上获取额外的 $500 额度。确认将于 6 月 11 日再次运行脚本来分配这些额度。
- **演讲幻灯片**：一位用户询问除了已在频道中分享的 Modal 演讲幻灯片外，是否还有其他工作坊的幻灯片。

**提到的链接**：[Video Conferencing, Web Conferencing, Webinars, Screen Sharing](https://us06web.zoom.us/rec/share/cewtLttl3vXMcSuFYrF7BXVkqBjpT937hmi9xLoQ0Nvd3Xac_F0ad9lVQH80o3Li.wzg8h6wUKjCaZOiY?startTime=1717529840000)：Zoom 是现代企业视频通信的领导者，拥有一个简单、可靠的云平台，可用于跨移动设备、桌面和会议室系统的视频和音频会议、聊天和网络研讨会。Zoom ...

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**clavie_beyond_ragbasics**](https://discord.com/channels/1238365980128706560/1242223963346698250/1248907349805105192) (104 messages🔥🔥):

- **ColBERT 添加了分层池化 (Hierarchical Pooling)**：ColBERT 的一个 [pull request](https://github.com/stanford-futuredata/ColBERT/pull/347) 增加了对可选分层池化的支持。根据讨论，详细介绍此增强功能的博客文章即将发布。
- **LLM 和 RAG 技术见解**：成员们讨论了构建 RAG (Retrieval-Augmented Generation) 应用程序的各种方法，包括使用 Elasticsearch 进行全文搜索，以及从 BM25 切换到向量数据库的影响。Elastic 的 [Dense vector field type](https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html) 也被提及为一个值得关注的资源。
- **向量搜索中的余弦距离 (Cosine Distance) 与 L2 距离**：社区探讨了向量搜索中余弦距离与 L2 距离的区别。一位成员指出，余弦距离在文档检索中更受青睐，因为它不受文档长度的影响，而未归一化向量上的欧几里得距离（Euclidean distance）则会受影响。
- **Ben Clavié 分享资源和代码**：大家感谢 Ben Clavié 的精彩演讲并分享了各种资源，包括一个包含加载 `wikipedia-api` 数据修改方案的 [GitHub gist](https://gist.github.com/bclavie/f7b041328615d52cf5c0a9caaf03fd5e)。成员们对 Clavié 将复杂信息转化为易懂术语的能力表示高度赞赏。
- **需要更多关于集成方法的信息**：讨论包括关于在搜索框架中组合评分、多语言嵌入模型的应用以及对长文档使用 LLM 分块（chunking）的实际问题。此外，大家还对 [Sentence Transformers](https://www.sbert.net/docs/sentence_transformer/training_overview.html) 及其在适应各种用例方面的强大可训练性表示了感谢。

**提到的链接**：

- [Dense vector 字段类型 | Elasticsearch 指南 [8.14] | Elastic](https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html): 未找到描述
- [考虑向量间角度的距离测量，同时兼顾幅度](https://stats.stackexchange.com/questions/71614/distance-measure-of-angles-between-two-vectors-taking-magnitude-into-account): 假设我有两个向量 v1 和 v2，我可以计算这两个向量之间的角度作为它们“距离”的度量，例如使用 arccos 函数。例如：...
- [Excalidraw — 让协作白板变得简单](https://excalidraw.com/): Excalidraw 是一个虚拟协作白板工具，让你能够轻松绘制具有手绘感的图表。
- [rag_mvp.py](https://gist.github.com/bclavie/f7b041328615d52cf5c0a9caaf03fd5e): GitHub Gist: 立即分享代码、笔记和代码片段。
- [GitHub - AnswerDotAI/rerankers](https://github.com/AnswerDotAI/rerankers): 通过在 GitHub 上创建账户，为 AnswerDotAI/rerankers 的开发做出贡献。
- [由 bclavie 添加对（可选）分层池化的支持 · Pull Request #347 · stanford-futuredata/ColBERT](https://github.com/stanford-futuredata/ColBERT/pull/347): @okhat 👀 效果非常好！博客文章将在几天内发布，未来的改进还需要一点时间。感谢 @NohTow 帮助测试/清理代码，将我的原型转化为可用版本...
- [SentenceTransformers 文档 — Sentence Transformers 文档](https://sbert.net/): 未找到描述
- [训练概览 — Sentence Transformers 文档](https://www.sbert.net/docs/sentence_transformer/training_overview.html): 未找到描述
- [Clem Fandango Steven Toast GIF - Clem Fandango Steven Toast Toast of London - 发现并分享 GIF](https://tenor.com/view/clem-fandango-steven-toast-toast-of-london-yes-i-can-hear-you-clem-fandango-gif-9211791307522605321): 点击查看 GIF
- [来自未定义的推文](https://x.com/bclavie): 未找到描述
- [GitHub - bclavie/RAGatouille: 在任何 RAG 管道中轻松使用和训练最先进的后期交互检索方法 (ColBERT)。专为模块化和易用性而设计，并有研究支持。](https://github.com/bclavie/RAGatouille): 在任何 RAG 管道中轻松使用和训练最先进的后期交互检索方法 (ColBERT)。专为模块化和易用性而设计，并有研究支持。 - bclavie/RAGatouille
- [黑客语言模型指南](https://www.youtube.com/watch?v=jkrNMKz9pWU): 在这段信息丰富的视频中，fast.ai 的联合创始人、所有现代语言模型 (LMs) 所基于的 ULMFiT 方法的创造者 Jeremy Howard...
- [GLiNER: 使用双向 Transformer 的通用命名实体识别模型](https://arxiv.org/abs/2311.08526): 命名实体识别 (NER) 在各种自然语言处理 (NLP) 应用中至关重要。传统的 NER 模型虽然有效，但局限于一组预定义的实体类型。相比之下...

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**jason_improving_rag**](https://discord.com/channels/1238365980128706560/1242224099548332132/1248926525386915933) (5 条消息):

- **Router 模型简化查询分类**: 一位成员提到使用具有三个步骤/指令的 **router 模型**，为每个查询并发调用模型三次。这种方法在使用 **mistral-7b** 或 **llama3-8b** 模型时非常有效。
- **探索用于分类的自定义 Embedding 模型**: 另一位成员询问了如何使用 **自定义 embedding 模型** 结合 **对比学习 (contrastive learning)** 进行分类。他们建议根据距离为 embedding 创建原型/质心，以实现更好的文本分类。
- **类别元数据增强产品推荐**: 一位成员分享了他们在产品推荐中添加 **类别元数据 (category metadata)** 的经验。通过动态填充类别并让 LLM 使用过滤选项，他们发现推荐的相关性有所提高。
- **实体提取和 Router 模型优于 Function Calling**: 一位成员解释说，由于图查询涉及的复杂性，他们更倾向于使用 **实体提取 (entity extraction)** 和 router 模型，而不是 function calling。他们发现，与 function calling 相比，他们的设置在大数据集下更快、更可靠。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**jeremy_python_llms**](https://discord.com/channels/1238365980128706560/1242224309548875917/1248738699865493624) (5 条消息):

- **LLM 容易被 Markdown 和代码块混淆**：一位成员分享了一个问题，即在测试包含 Markdown 和代码块的字符串时，会导致 LLM 表现出不可预测的行为。他们提到了在提供 Claude 风格 Prompt 建议时遇到的困难。
- **NotString 修复格式问题**：为了解决 Discord 上 Markdown 字符串中反引号转义的问题，建议将 `text` 包装到 `NotString` 中。这种方法可以确保正确的渲染。
- **与 Prime 一起学习 htmx**：一位成员推荐了一个 [YouTube 视频](https://www.youtube.com/watch?v=x7v6SNIgJpE)，作为 Prime 介绍 htmx 的入门教程。
- **对 fasthtml 的兴奋以及对 Typescript 的回避**：另一位成员表达了对 fasthtml 的热情，以及它在简化扩展 Streamlit 应用方面的潜力。他们希望 fasthtml 能帮助他们避免学习 Typescript。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**saroufimxu_slaying_ooms**](https://discord.com/channels/1238365980128706560/1242224552415596554/1249415356468953129) (135 条消息🔥🔥):

- **Fused vs Paged Optimizers 辩论**：成员们讨论了 **fused vs paged optimizers** 的区别和优势，指出 *“fused optimizers 主要是为了调度更少的 CUDA kernels”*，这使得 `optimizer.step` 速度更快。此外，将 optimizer state 卸载（offloading）到 CPU 有助于避免 OOM，尽管这可能会导致模型的运行速度变得不可预测。

- **8-bit Adam Optimizer 困惑**：用户分享了关于 **adamw_bnb_8bit** 的使用经验，特别是为什么有些人发现它与 **adamw_torch** 相比在内存占用上没有差异。解释指出，对于 LoRA 而言，由于大部分参数是不可训练的且没有 optimizer state，因此 optimizer state 本身就比较小。

- **Jane 被邀请加入讨论**：有一段关于邀请 **Jane** 加入讨论的对话，一名成员提供了 [邀请链接](https://discord.gg/RfcRWeNs)。

- **Vast.ai 和 Autoscaler 澄清**：用户讨论了 **Vast.ai** 是否为 serverless。经澄清，虽然它并非严格意义上的 serverless，但它提供了用于管理动态工作负载的 autoscaling，并分享了 [autoscaler 文档](https://vast.ai/docs/autoscaler/introduction)。

- **资源与工具汇编**：分享了一些有用的链接和资源，包括 [Tim Dettmers 的 YouTube 视频](https://www.youtube.com/watch?v=jyOqtw4ry2w&themeRefresh=1)、[演讲幻灯片](https://drive.google.com/drive/u/0/folders/1HmGNC4v4L5nXhtdDMVCpUBrme1ELp-2C)，以及使用 [PyTorch profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#using-tracing-functionality) 的细节。成员们对这次内容丰富的演讲和资源表示感谢。

**提到的链接**：

- [Perfetto - System profiling, app tracing and trace analysis](https://perfetto.dev/)：未找到描述
- [Google Drive: Sign-in](https://drive.google.com/)：未找到描述
- [Lecture 16: On Hands Profiling](https://www.youtube.com/watch?v=SKV6kDk1s94)：未找到描述
- [Join the llm-fine-tuning Discord Server!](https://discord.gg/RfcRWeNs)：查看 Discord 上的 llm-fine-tuning 社区 - 与其他 1888 名成员一起交流，享受免费的语音和文字聊天。
- [Slaying OOMs traces – Google Drive](https://drive.google.com/drive/u/3/folders/1HmGNC4v4L5nXhtdDMVCpUBrme1ELp-2C)：未找到描述
- [PyTorch Profiler — PyTorch Tutorials 2.3.0+cu121 documentation](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#using-tracing-functionality)：未找到描述
- [8-bit Methods for Efficient Deep Learning with Tim Dettmers](https://www.youtube.com/watch?v=jyOqtw4ry2w&themeRefresh=1)：Tim Dettmers（华盛顿大学博士生）在 Cohere For AI 技术演讲中介绍了“高效深度学习的 8-bit 方法”。摘要：La...
- [Thanks Bow GIF - Thanks Bow Thank You - Discover & Share GIFs](https://tenor.com/view/thanks-bow-thank-you-sign-of-respect-gif-4807966236937524301)：点击查看 GIF
- [Overview | Vast.ai](https://vast.ai/docs/autoscaler/introduction)：未找到描述
- [I Know Some Of These Words Mhmm GIF - I Know Some Of These Words Mhmm Clueless - Discover & Share GIFs](https://tenor.com/xHx8.gif)：点击查看 GIF
- [Slaying OOMs traces – Google Drive](https://drive.google.com/drive/u/0/folders/1HmGNC4v4L5nXhtdDMVCpUBrme1ELp-2C)：未找到描述
- [Im Pretending I Know What Youre Talking About Ahmed Aldoori GIF - Im Pretending I Know What Youre Talking About Ahmed Aldoori I Have No Idea - Discover & Share GIFs](https://tenor.com/view/im-pretending-i-know-what-youre-talking-about-ahmed-aldoori-i-have-no-idea-faking-it-pretending-gif-18453815)：点击查看 GIF
- [GitHub - pytorch/torchtune: A Native-PyTorch Library for LLM Fine-tuning](https://github.com/pytorch/torchtune)：一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。
- [Answer.AI - You can now train a 70b language model at home](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html)：我们正在发布一个基于 FSDP 和 QLoRA 的开源系统，可以在两个 24GB GPU 上训练 70b 模型。
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/)：未找到描述
- [GitHub - pytorch/torchtitan: A native PyTorch Library for large model training](https://github.com/pytorch/torchtitan)：一个用于大模型训练的原生 PyTorch 库。通过在 GitHub 上创建账号为 pytorch/torchtitan 的开发做出贡献。
- [enable QLoRA + FSDP2 by weifengpy · Pull Request #909 · pytorch/torchtune](https://github.com/pytorch/torchtune/pull/909)：此 PR 构建在包含 NF4Tensor FSDP2 ops 的 TorchAO nightly 版本之上。PR1 PR2 Pytorch nightly 包含 meta init + cpu offloading。PR 单元测试：pytest -s tests/torchtune/utils/test_di...

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**paige_when_finetune**](https://discord.com/channels/1238365980128706560/1242224662142779530/1249860352565448834) (159 条消息🔥🔥):

- **成员们期待 Fine-Tuning 的“爆米花时间”**：讨论中穿插着关于爆米花的幽默，诸如 *“你只需要爆米花”* 之类的评论，并引用了各种预测爆米花爆裂时间的概率模型。一位用户开玩笑说要用合成爆米花数据来 Fine-tuning 一个 LLM，另一位用户评论道：*“谁要是根据 ftcourse 仓库做一个关于爆米花玉米粒的案例研究，那简直就是传奇。”*
- **深入探讨逆泊松分布 (Inverse Poisson Distribution)**：分享了关于逆泊松分布的复杂数学解释，并提供了 [math.stackexchange](https://math.stackexchange.com/questions/1195566/inverse-of-a-poisson-distribution-function) 的链接，帮助用户理解概率公式。
- **Gemini API 和 AI 改进**：用户讨论了 Google Gemini 的各种功能及相关的 API 改进，包括指向音频输入的 [Google-gemini](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Audio.ipynb) 链接和 [Gemini API caching](https://ai.google.dev/gemini-api/docs/caching)。大家对模型能力和多模态模型等高效用例表现出显著的热情。
- **分享 Prompting 技巧**：重点讨论了使用 AI 为其自身编写 Prompt。参与者提到了使用 Claude 等模型，以及通过 Twitter 分享的 Prompt 创建工具，还有通过对话不断迭代和适配来有效引导 Gemini 的技巧。
- **关于 Fine-Tuning 建议的辩论**：辩论了 Fine-tuning 与利用增加的 Context Window（针对 Mixtral 和 GPT-4 等模型）的优劣。一位用户对 Mixtral 的输出格式化感到沮丧，并被建议使用大量的 Few-shot prompting，或者考虑从 Instruct 模型切换到 Base 模型。

**提到的链接**：

- [未找到标题](https://ai.google.dev/gemini-api/docs/caching)：未找到描述
- [Inverse of a Poisson distribution function](https://math.stackexchange.com/questions/1195566/inverse-of-a-poisson-distribution-function)：我有两个独立同分布的随机变量 $X_{1}$ 和 $X_{2}$，遵循连续泊松分布函数 $P(x) = \lambda e^{-\lambda\cdot x}$。我希望获得一个分布函数...
- [未找到标题](https://ai.google.dev/pricing)：未找到描述
- [未找到标题](https://ai.google.dev/gemini-api/docs/get-started/android_aicore)：未找到描述
- [PaliGemma – Google's Cutting-Edge Open Vision Language Model](https://huggingface.co/blog/paligemma)：未找到描述
- [So Excited GIF - So Excited Cant - Discover & Share GIFs](https://tenor.com/view/so-excited-cant-wait-gif-24703188)：点击查看 GIF
- [Spongebob Patrick GIF - Spongebob Patrick Star - Discover & Share GIFs](https://tenor.com/view/spongebob-patrick-star-noted-notes-gif-17474838830648097856)：点击查看 GIF
- [cookbook/quickstarts/PDF_Files.ipynb at main · google-gemini/cookbook](https://github.com/google-gemini/cookbook/blob/main/quickstarts/PDF_Files.ipynb)：Gemini API 的指南和示例集合。 - google-gemini/cookbook
- [cookbook/quickstarts/Audio.ipynb at main · google-gemini/cookbook](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Audio.ipynb)：Gemini API 的指南和示例集合。 - google-gemini/cookbook
- [GitHub - google-research/t5x](https://github.com/google-research/t5x)：通过在 GitHub 上创建账号来为 google-research/t5x 的开发做出贡献。
- [GitHub - outlines-dev/outlines: Structured Text Generation](https://github.com/outlines-dev/outlines)：结构化文本生成。通过在 GitHub 上创建账号来为 outlines-dev/outlines 的开发做出贡献。
- [来自未定义的推文](https://x.com/dynamicwebpaige)：未找到描述
- [webpaige.dev](https://webpaige.dev/)：未找到描述
- [Vertex AI with Gemini 1.5 Pro and Gemini 1.5 Flash](https://cloud.google.com/vertex-ai?hl=en)：尝试使用 Vertex AI，这是一个用于构建生成式 AI 应用的全托管 AI 开发平台，可访问包括 Gemini 1.5 模型在内的 130 多个基础模型。
- [Multimodal prompting with a 44-minute movie | Gemini 1.5 Pro Demo](https://www.youtube.com/watch?v=wa0MT8OwHuk)：这是长上下文理解的演示，是我们最新模型 Gemini 1.5 Pro 中的一项实验性功能，使用了一部 44 分钟的 Buster Keaton 默片 Sherl...
- [未找到标题](https://aistudio.google.com/)：未找到描述
- [Build with Google AI](https://discuss.ai.google.dev/)：就 Google 的 Gemini API 和 Google AI Studio 提问并获取支持

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**yang_mistral_finetuning**](https://discord.com/channels/1238365980128706560/1242224842053521459/1248783851489329265) (3 messages):

- **模型下载可用性**：一位成员询问了微调后下载模型的能力，并提供了一个 [Mistral 微调指南链接](https://docs.mistral.ai/guides/finetuning/)。
- **对额度套餐感到失望**：一位成员对 Mistral 未参与额度套餐表示失望。在提供电话号码后，他们仅获得了 $5 的额度，这不足以支付起价为 $4 的多次微调任务。
- **黑客松参与者的免费额度**：一位成员分享了为通过 Mistral AI 微调黑客松申请的参与者提供的约 $100 免费额度。详细信息包括 [公告链接](https://mistral.ai/news/2024-ft-hackathon/)、重要日期、奖项和申请要求。

**提到的链接**：[Mistral AI Fine-tuning Hackathon](https://mistral.ai/news/2024-ft-hackathon/)：我们很高兴地宣布 Mistral AI 微调黑客松，这是一场将于 2024 年 6 月 5 日至 30 日举行的虚拟体验活动。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**axolotl**](https://discord.com/channels/1238365980128706560/1242542198008975430/1248774125040963728) (4 messages):

- **使用 vllm 修复 LLama3 微调问题**：一位尝试使用 **chatml tokens** 微调 **LLama3** 模型的成员遇到了与 **lm_head.weight 尺寸不匹配**相关的加载错误。他们通过使用 **vllm 进行推理**解决了该问题，并报告称该方法修复了问题。
- **寻找指令模型微调的数据集格式**：另一位成员征求了关于如何格式化数据集以使用相同 Prompt 格式微调指令模型的建议。他们请求提供示例以澄清流程以及如何准确标记数据。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**zach-accelerate**](https://discord.com/channels/1238365980128706560/1242564031425024010/1249691092644003881) (7 messages):

- **DS(zero3) 在 LoRA 训练测试中优于 DDP 和 FSDP**：一位成员分享了他们使用 **accelerate** 进行 LoRA 训练和多节点配置的经验。他们进行了对比 DDP、DS(zero3) 和 FSDP 的测试，发现 *"DS(zero3) 是短期测试中的赢家"*，其 ETA 为 18:42，使用 27GB vRAM，而 DDP 为 18:13（33GB vRAM），FSDP 为 21:47（30GB vRAM）。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**wing-axolotl**](https://discord.com/channels/1238365980128706560/1242564077151326388/1249015603268816998) (1 messages):

- **用户寻求 Mistral instruct 的 chat_templates 概览**：一位成员询问了可用的 **chat_templates**，以确定哪一个支持在 **DPO Mistral instruct** 中使用的 **system message**。摘要中未提供直接回复或链接。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**charles-modal**](https://discord.com/channels/1238365980128706560/1242564177952768062/1249379845163712623) (7 messages):

- **Modal 的魔力留下深刻印象**：一位用户分享了他们使用 Modal 部署 Mistral 进行推理的兴奋之情，提到了在远程服务器上运行本地代码并具有热重载（hot-reload）功能的“神奇体验”。他们赞同这需要一种新的思维方式，但发现它非常有价值。
- **权限错误导致挫败感**：一位用户指出需要为密钥访问设置正确的权限，并强调日志深处的 `403` 错误表示权限问题。
- **寻求指令微调的帮助**：一位用户请求关于使用特定模板进行指令微调的指导，并询问了 config yaml 文件中数据集和 token 的正确配置。
- **Volume ID 错误难倒用户**：另一位用户在运行 llm-finetuning 示例时遇到了 "Volume ID is missing" 错误，尽管他们在同一个终端会话中能成功运行另一个示例。他们被建议通过提供的 Slack URL 向工程团队寻求进一步帮助。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**langchain-langsmith**](https://discord.com/channels/1238365980128706560/1242564256914870384/1248793213830037677) (3 messages):

- **Maven 视频链接问题已解决**：一位成员报告了 Maven 上的视频链接无法重定向到 Zoom 的问题。另一位成员确认了该报告并证实链接已修复，原用户随后确认现在可以正常工作。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**credits-questions**](https://discord.com/channels/1238365980128706560/1243721538432270388/1248724324689641643) (7 messages):

- **部分用户缺失 HuggingFace 额度**：成员反馈尽管填写了包含账户 ID 的表单，但仍未收到 HuggingFace 额度。他们被引导至特定频道发布信息，并需要提供电子邮件和 HF 用户名。
- **Modal 额度也存在问题**：另一位用户提到 Modal 额度遇到困难，并说明了他们注册和申请额度的具体日期。他们被建议在另一个频道咨询以解决该问题。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**strien_handlingdata**](https://discord.com/channels/1238365980128706560/1243773476301443073/) (1 messages):

davidberenstein1957: 亲爱的 Vincent ❤️❤️

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**fireworks**](https://discord.com/channels/1238365980128706560/1245126291276038278/1248729243437895872) (21 messages🔥):

- **成员寻求额度帮助**：包括账户 ID 为 `fil-161078`、`alexander-nicholson-8c5e72` 和 `shruthi-badri-cc7a24` 在内的多位成员请求协助解决未收到预期额度的问题。一位成员提到：“我填写了表单但还没收到额度。”
- **AI Engineer World's Fair 邀请**：一位成员邀请大家在即将举行的 AI Engineer World's Fair 上见面，并分享了活动链接：[AI Engineer World's Fair](https://www.ai.engineer/worldsfair)。另一位成员确认可能会参加。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**emmanuel_finetuning_dead**](https://discord.com/channels/1238365980128706560/1245129595749925086/1248816940219371590) (5 messages):

- **Fine-tuning 与 RAG 知识类比**：一位成员分享了他们的心智模型，描述了 **Fine-tuning** 如何增加静态的、领域适配的知识（适用于各种查询），而 **RAG**（检索增强生成）则提供动态的、特定上下文的信息。他们将其比作程序员使用通用编程知识与在 StackOverflow 上查找特定解决方案的区别。[博客链接](https://ankur-singh.github.io/blog/finetuning-vs-rag)
- **对类比的批评**：另一位成员表示不喜欢类比，并提到他们正在研究一种更好的方式来解释这些概念，表明更倾向于更精确的解释。
- **序列知识获取阶段**：提出了一种关于知识阶段的详细观点：*Pretraining* 是理论学习，*Finetuning* 是实际应用，而 *Alignment* 类似于从导师那里获得反馈以达到精通。这种循序渐进的方法突出了训练语言模型过程中不断演变的复杂性。
- **Post-Training 与 Fine-Tuning 的区别**：一位成员区分了 **Post-training** 和 **Fine-tuning**，指出 Post-training 涉及使模型与连贯的回答对齐并生成 Instruct 模型，而 Fine-tuning 则涉及通过特定示例定制模型的输出风格。他们引用了一篇论文，建议 1000 个示例可能足以进行 Alignment，但推测更高的数量可能对鲁棒性更好。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**braintrust**](https://discord.com/channels/1238365980128706560/1245407617031999581/1248958521517670443) (9 messages🔥):

- **联系信息混乱问题得到解决**：发现了一个关于某些在截止日期前注册的用户缺失联系信息的问题。经过进一步检查，发现 David 的电子邮件中包含一个换行符，这导致重新运行脚本以修剪所有电子邮件，并确保包括 David 在内的所有 18 名学生都已正确配置。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**west-coast-usa**](https://discord.com/channels/1238365980128706560/1245410680065097738/1248999438752677898) (2 messages):

- **在旧金山讨论潜在的聚会**：一位成员询问在下周日之前旧金山是否有任何与课程相关的活动。另一位成员建议他们可以一起出来逛逛，看看还有谁感兴趣。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**europe-tz**](https://discord.com/channels/1238365980128706560/1245425547048386732/) (1 messages):

weskhan_62459: 大家好，我是来自澳大利亚的，目前在波兰向大家问好。

### **LLM Finetuning (Hamel + Dan) ▷ #**[**predibase**](https://discord.com/channels/1238365980128706560/1245803791710687272/1249772551245398117) (4 messages):

- **Predibase 注册步骤已就绪**：一名成员提醒其他人在 [predibase.com/free-trial](https://predibase.com/free-trial) 注册后检查邮件中的 "Accept Invitation" 链接，以完成账号创建。为了确保合规，询问了 *"你是否收到了这个并完成了该流程？"*。
- **Fine-tuning 研讨会录像已发布**：分享了虚拟 Fine-tuning 研讨会的录像链接，点击[此处](https://my.demio.com/recording/ulTom0AP)查看。该资源旨在帮助新用户开始他们的 Fine-tuning 项目。
- **Predibase 额度查询**：一位用户提到他们注册后在租户 ID c4697a91 下收到了 $25 额度。另一位成员承诺会进行调查以提供进一步帮助。

**提及的链接**：[Login - Demio](https://my.demio.com/recording/ulTom0AP.)：未找到描述

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**openpipe**](https://discord.com/channels/1238365980128706560/1245927847437008896/1248755936789790892) (5 messages):

- **账单困惑已解决**：几位成员遇到了账单额度不一致的问题，报告显示为 **$100** 而非预期的 **$222**。一位成员确认问题已解决，并感谢 "Anfal" 的帮助。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**openai**](https://discord.com/channels/1238365980128706560/1245927985123692575/1248724051187204190) (39 messages🔥):

- **用户难以访问 GPT-4 模型**：多位用户报告称，尽管按照必要步骤操作（包括填写表格和添加账单信息），仍无法访问 GPT-4 和 GPT-4o 模型。一位用户通过充值 $5 额度解决了问题，而其他人仍面临困难。
- **征集 Organization ID**：一位用户请求其他成员提供 org_ids，以帮助解决访问问题。几位用户分享了他们的 org_ids，希望能解决 GPT-4 模型的访问问题。
- **关于处理和评分 Prompt 的问题**：一位用户寻求能够处理错误并支持断点续传的工具推荐，用于对大量 Prompt 列表进行评分。这引起了关注，但需要进一步详细说明才能提供有用的建议。
- **额度使用与想法交流**：一位用户分享了他们如何使用额度，并提供了其列表的 Twitter 链接，同时邀请其他人分享想法。[在此查看列表](https://twitter.com/m_chirculescu/status/1799174718286684245?t=gA7oEwPtbq9SuFC-tl6hSA&s=19)。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**capelle_experimentation**](https://discord.com/channels/1238365980128706560/1248386323035721810/1248909475960848466) (13 messages🔥):

- **Weave 免费入门**：一位成员分享了 [Notebook 链接](http://wandb.me/weave_colab) 以学习 Weave 基础知识，Weave 对于跟踪函数调用、发布和检索版本化对象以及使用简单 API 进行评估非常有用。
- **W&B 快速课程**：分享了一个 10 分钟的 [W&B 视频课程](https://www.wandb.courses/courses/wandb-101)，帮助用户了解 Weights & Biases 的核心功能，提高机器学习生产力，并学习如何与 Python 脚本集成。
- **加入 Inspect_Ai 协作**：一位成员邀请其他人协作开发 Inspect_Ai 中的交互和注释共享视图，并将其与 Weights & Biases 链接以实现强大的数据呈现。
- **用于 Eval 可视化的 Python Logging**：讨论了 Michael Driscoll 的 Python Logging 书籍及其在评估中可视化表达性日志配置的相关性，强调了 Python logging 模块的功能。
- **微调 Llama 7B 进行查询转换**：针对 W&B 中使用查询语言过滤数据的功能，有人对微调 Llama 7B 以将自然语言查询 (NLQ) 转换为 DSL 的项目表示出兴趣。分享了关于使用 W&B 查询面板的更多细节，并附带了[文档](https://docs.wandb.ai/guides/app/features/panels/query-panel)链接。

**提及的链接**：

- [Query panels | Weights & Biases Documentation](https://docs.wandb.ai/guides/app/features/panels/query-panel)：此页面上的某些功能处于 Beta 阶段，隐藏在功能标志后。在个人资料页面的 bio 中添加 `weave-plot` 即可解锁所有相关功能。
- [Google Colab](http://wandb.me/weave_colab)：未找到描述
- [W&B 101: ML Experiment Tracking Course | Weights & Biases](https://www.wandb.courses/courses/wandb-101)：W&B 101 是终极机器学习实验跟踪课程。通过实践经验来跟踪、比较和优化模型。立即报名并掌控您的机器学习实验。

---

### **Nous Research AI ▷ #**[**off-topic**](https://discord.com/channels/1053877538025386074/1109649177689980928/1249280788340674604) (9 条消息🔥):

- **早餐喝发酵柳兰茶**：一位成员分享了他们不寻常的早餐选择，包括发酵柳兰茶、牛奶、甜叶菊、2 根加了海盐的黄瓜、抹了蛋黄酱的黑麦酸种面包、香肠和奶酪。
- **混合使用 GPU 的复杂性**：一位成员询问了在机器学习设备中安装两个不同型号 GPU 的难点。另一位成员回答说，系统速度会受限于最慢的 GPU，从而降低效率。

---

### **Nous Research AI ▷ #**[**interesting-links**](https://discord.com/channels/1053877538025386074/1132352574750728192/1248760444533866559) (5 条消息):

- **σ-GPT 动态生成序列**：一位成员惊叹道，由 @ArnaudPannatier 团队开发的 [σ-GPT](https://x.com/arnaudpannatier/status/1799055129829839166?s=46) 在推理时可以按任何顺序生成序列，而不像传统的 GPT 那样从左到右生成。该项目是与 @SkysoftATM 合作开发的。
- **从 GPT-4 中提取概念**：一位成员分享了 [OpenAI 博客](https://openai.com/index/extracting-concepts-from-gpt-4/)关于从 GPT-4 中提取概念的文章。成员们将其与 Anthropic 最近发表的关于理解 GPT-4 的论文进行了比较，认为两者的意图相似，但发现可能有所不同。

**提到的链接**：[来自 Arnaud Pannatier (@ArnaudPannatier) 的推文](https://x.com/arnaudpannatier/status/1799055129829839166?s=46)：GPT 正在按从左到右的顺序生成序列。还有其他方法吗？我们与 @francoisfleuret、@evanncourdier 以及合作伙伴 @SkysoftATM 一起开发了 σ-GPT，能够生成序列...

---

### **Nous Research AI ▷ #**[**general**](https://discord.com/channels/1053877538025386074/1149866623109439599/1248713665448902787) (255 条消息🔥🔥):

- **推荐用于 Outpainting 的 Krita 插件**：一位成员推荐使用 [Krita stable diffusion 插件](https://github.com/Acly/krita-ai-diffusion)进行 Outpainting（外绘），并指出其学习曲线比 [fooocus](https://www.someotheroutpaintingoption.com) 更陡峭。他们建议通过迭代增加分辨率来达到目标宽高比，而不是直接设置为 16:9。
- **72b 模型的惊人性能**：成员们讨论了 72b 模型令人印象深刻的数学和物理推理能力，并将其性能与 GPT-4 进行了比较。该模型在 Together 平台上的可用性引发了大家对测试环境的兴趣。
- **层剪枝策略实验**：社区讨论了针对 Llama 3 70b 和 Qwen 2 72b 等模型的剪枝策略，包括移除层和 Finetuning。文中引用了[一篇相关论文](https://arxiv.org/abs/2403.17887)以及像 [PruneMe](https://github.com/arcee-ai/PruneMe) 这样的实现。
- **对 GPU 云成本和资源的关注**：成员们分享了价格合理的 GPU 云服务资源，如 [Interstice Cloud](https://www.interstice.cloud) 和 [Playground AI](https://playground.com/pricing)。此外，还讨论了在云平台上托管和运行大型模型的挑战及建议。
- **关于 AI 监管的法律和伦理讨论**：指向 [Dan Jeffries 关于 SB 1047 推文串](https://x.com/dan_jeffries1/status/1794740447052525609?s=46)的链接引发了关于 AI 监管及其对创新影响的辩论。Jeffries 批评该法案可能会打着安全措施的旗号，使 AI 控制权中心化并摧毁开源 AI。

**提到的链接**：

- [The Unreasonable Ineffectiveness of the Deeper Layers](https://arxiv.org/abs/2403.17887)：我们对流行的开源权重预训练 LLM 系列进行了一种简单的层剪枝（layer-pruning）策略的实证研究，发现在不同的问答基准测试中，性能退化极小，直到...
- [Paper page - The Unreasonable Ineffectiveness of the Deeper Layers](https://huggingface.co/papers/2403.17887)：未找到描述
- [Together AI | Dedicated Instances](https://www.together.ai/forms/model-requests)：未找到描述
- [carsonpoole/Qwen2-72B-Instruct-Every-Other-Layer · Hugging Face](https://huggingface.co/carsonpoole/Qwen2-72B-Instruct-Every-Other-Layer)：未找到描述
- [carsonpoole/Qwen2-37B-Pruned · Hugging Face](https://huggingface.co/carsonpoole/Qwen2-37B-Pruned)：未找到描述
- [Tweet from Jeremy Nixon (@JvNixon)](https://x.com/jvnixon/status/1799996074146578801?s=46)：SB 1047 值得一个反击！！欢迎来到 SB 1048。📚《AI 创新自由法案》📚。它赋予了 AI 来自第 230 条（Section 230）的最强有力论据，该条款曾保护了互联网生机勃勃的生态系统...
- [Together AI](https://www.together.ai/)：使用 Together AI 构建生成式 AI 模型。受益于最快、最具成本效益的工具和基础设施。与我们致力于助你成功的专家 AI 团队合作。
- [axolotl/examples/qwen2/qlora-fsdp.yaml at main · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/qwen2/qlora-fsdp.yaml)：尽管提问（axolotl 谐音 ask a lot of）。通过在 GitHub 上创建账号，为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。
- [Tweet from Simo Ryu (@cloneofsimo)](https://x.com/cloneofsimo/status/1799819817333219662)：当前非精选（non-cherry-picked）结果。很快将加倍算力，并改进 MFU 和方法。
- [Tweet from Aran Komatsuzaki (@arankomatsuzaki)](https://x.com/arankomatsuzaki/status/1799986416077951460)：TogetherAI 发布 Mixture-of-Agents 增强大语言模型能力，在 AlpacaEval 2.0、MT-Bench 和 FLASK 上达到 SotA 性能，超越 GPT4o https://arxiv.org/abs/2406.04692
- [Self-Supervised Alignment with Mutual Information: Learning to Follow Principles without Preference Labels](https://arxiv.org/abs/2404.14313)：在提示语言模型（LM）时，用户通常期望模型在不同任务中遵循一套行为原则，例如在产生深刻内容的同时避免有害或偏见...
- [Tweet from Daniel Jeffries (@Dan_Jeffries1)](https://x.com/dan_jeffries1/status/1794740447052525609?s=46)：我花了几个小时听 Dan Hendyrcks 的发言，他领导着 SB 1047（又称加州 AI 控制与集权法案）背后的非营利 AI 安全组织。我觉得他很有魅力、稳重、聪明...
- [Mutual information - Wikipedia](https://en.m.wikipedia.org/wiki/Mutual_information)：未找到描述
- [GitHub - SilasMarvin/lsp-ai: LSP-AI 是一个开源语言服务器，作为 AI 驱动功能的后端，旨在辅助和赋能软件工程师，而非取代他们。](https://github.com/SilasMarvin/lsp-ai)：LSP-AI 是一个开源语言服务器，作为 AI 驱动功能的后端，旨在辅助和赋能软件工程师，而非取代他们。 - SilasMarvin/lsp-ai
- [GitHub - PygmalionAI/aphrodite-engine: PygmalionAI 的大规模推理引擎](https://github.com/PygmalionAI/aphrodite-engine)：PygmalionAI 的大规模推理引擎。通过在 GitHub 上创建账号，为 PygmalionAI/aphrodite-engine 的开发做出贡献。
- [GitHub - arcee-ai/PruneMe: 自动识别大语言模型中用于剪枝的冗余层块](https://github.com/arcee-ai/PruneMe)：自动识别大语言模型中用于剪枝的冗余层块 - arcee-ai/PruneMe
- [Interstice](https://www.interstice.cloud/service)：未找到描述
- [/g/ - /lmg/ - 本地模型讨论区 - 技术 - 4chan](https://boards.4chan.org/g/thread/100871552#p100874685)：未找到描述
- [Fireworks - 用于产品创新的生成式 AI！](https://fireworks.ai/)：以极快的速度使用最先进的开源 LLM 和图像模型，或者使用 Fireworks AI 免费微调并部署你自己的模型！

### **Nous Research AI ▷ #**[**ask-about-llms**](https://discord.com/channels/1053877538025386074/1154120232051408927/1249191087394783302) (8 条消息🔥):

- **使用本地 Agent 的 AgentBench**：一位成员询问是否有人在 **AgentBench** 中使用过像 **Llama 3** 这样的本地 Agent，而不是 **GPT-3.5**。
- **群聊 UX 工作流**：讨论开始围绕创建一个共享的群聊 UX 工作流展开，该工作流结合了人类团队和 AI 工具/封装器（wrappers）。目标是让像 **Claude** 和 **GPT** 这样的 AI 共享一个上下文窗口（context window），从而实现人类与 AI 参与者之间的无缝交互。
- **低端 PC 推荐**：一位成员为 Python 课程寻求可以在无独立显卡且仅有 8 GB RAM 的低端 PC 上运行的 **LLM** 推荐。有人建议使用 **Phi-3 3b**，尽管大家注意到目前没有哪个选项在处理代码方面表现特别出色。

---

### **Nous Research AI ▷ #**[**rag-dataset**](https://discord.com/channels/1053877538025386074/1218682416827207801/1248957012570800188) (335 条消息🔥🔥):

- **HippoRAG 与 Raptor：聚类的未来**：一位成员强调 HippoRAG 相比知识图谱（KGs）更注重聚类，以实现更好的信息提取。根据 [HippoRAG](https://arxiv.org/abs/2405.14831) 的说法，“聚类也是一种图”，这使其成为高效语言模型训练的关键工具。
- **RAG 的 Schema 辩论**：多位成员讨论了模型数据输入和输出的 JSON schema，建议使用类似 `"is_supporting": true/false` 的格式。分享了一个提议的 schema，包括 "question"、"answer" 和 "context" 字段。
- **Ditto 与动态偏好优化（DPO）**：在讨论来自 [arxiv](https://arxiv.org/abs/2406.00888) 的 Ditto 的潜力时，成员们考虑了在线比较以及使用小数据集与细粒度任务进行迭代对齐。另一位成员建议在动态奖励建模框架中使用余弦相似度（cosine similarity）作为指标。
- **多指标输出标准化**：成员们辩论了是否直接将相关性、相似度得分和情感等指标纳入数据集以优化输出。有人建议“我们可以将 RAGAS 或某些评估器连接到我们的数据生成器”，并推荐结合排名或简化评估（如“高”、“中”、“低”）来使模型的输出与上下文对齐。
- **Cohere 的检索与引用机制**：社区研究了 Cohere 的检索系统及其使用文档标题作为搜索查询和解析引用的方式。建议将引用以 JSON 等结构化格式存储以便参考，从而避免处理多种文档格式的复杂性。

**提到的链接**：

- [HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models](https://arxiv.org/abs/2405.14831)：为了在充满敌意且不断变化的自然环境中生存，哺乳动物的大脑进化出了存储大量世界知识并不断整合新信息的能力，同时避免……
- [Show, Don't Tell: Aligning Language Models with Demonstrated Feedback](https://arxiv.org/abs/2406.00888)：语言模型被对齐以模仿大众的集体声音，导致输出不符合任何特定个人的偏好。通过监督微调引导 LLM 远离泛泛的输出是可能的……
- [Aran Komatsuzaki (@arankomatsuzaki) 的推文](https://x.com/arankomatsuzaki/status/1799987241135284624)：Meta 发布了 CRAG - 综合 RAG 基准测试。提供了一个包含 4,409 个问答对的事实性问答基准，以及模拟网络和知识图谱（KG）搜索的模拟 API。项目地址：https://www.aicrowd.com/challen...
- [GitHub - interstellarninja/data-genie: A synthetic data generation pipeline using LLMs and websearch over a curriculum](https://github.com/interstellarninja/data-genie)：一个使用 LLM 和基于课程的网页搜索的合成数据生成流水线 - interstellarninja/data-genie
- [jondurbin/contextual-dpo-v0.1 · Hugging Face 数据集](https://huggingface.co/datasets/jondurbin/contextual-dpo-v0.1)：未找到描述

---

### **Nous Research AI ▷ #**[**world-sim**](https://discord.com/channels/1053877538025386074/1221910674347786261/1248852961027821648) (9 条消息🔥):

- **递归 AI 可视化令人惊叹**：一位用户暗示了在递归 AI 可视化方面的进展，但未透露更多细节。他们发布了一张引人入胜的图片，[点击此处查看](https://cdn.discordapp.com/attachments/1221910674347786261/1242196746738860103/OIG3.png)。
- **命令行复制粘贴 Bug 已修复**：一位用户抱怨从命令行界面复制和粘贴文本时存在问题。在管理员确认其报告后，另一位管理员证实该 Bug 现已修复，功能应能按预期运行。

---

### **Unsloth AI (Daniel Han) ▷ #**[**general**](https://discord.com/channels/1179035537009545276/1179035537529643040/1248716615529271347) (366 条消息🔥🔥):

- **“Qwen GGUF 问题持续引发讨论”**：多位用户讨论了 **Qwen** 模型存在的 **GGUF 问题**，特别是重复出现“块状”文本。虽然有人确认在 LM Studio 等特定工具上运行正常，但其他人发现问题依然存在，尤其是 **7B 模型**。
- **“LSP-AI 以多编辑器兼容性令人印象深刻”**：一位用户分享了 [GitHub - LSP-AI](https://github.com/SilasMarvin/lsp-ai) 的链接，强调其作为 VS Code、NeoVim 和 Emacs 等多个编辑器的 **Language Server 功能**。其目标是**增强而非取代**软件工程师的工具链。
- **“新的持续预训练 Notebook 简化了微调”**：成员们讨论了用于 **Continued Pretraining** 的新 Unsloth Colab Notebook，提到了其易用性以及微调 **Input 和 Output Embeddings** 的能力。分享了指向 [Unsloth Blog](https://unsloth.ai/blog/contpretraining) 等资源的链接。
- **“LLama-3 8B 与 Mistral V0.3 微调对比”**：关于微调性能的对话中，用户们争论 **LLama-3 8B 是否优于 Mistral V0.3**。Theyruinedelise 提到即将发布一篇博客文章来详细说明研究结果。
- **“多阶段训练与数据增强策略”**：用户分享了改进模型训练的策略，强调通过噪声副本进行 **Data Augmentation** 并更好地平衡数据集。shensmobile 对可调节的 **LoRA 设置** 以适应特定任务表现出兴趣。

**提到的链接**：

- [Google Colab](https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing): 未找到描述
- [mistral-community/Codestral-22B-v0.1 · Hugging Face](https://huggingface.co/mistral-community/Codestral-22B-v0.1): 未找到描述
- [NVIDIA Corp (NVDA) Stock Price & News - Google Finance](https://google.com/finance/quote/NVDA:NASDAQ?hl=en&window=5Y): 获取最新的 NVIDIA Corp (NVDA) 实时报价、历史表现、图表和其他财务信息，帮助您做出更明智的交易和投资决策。
- [Tweet from David Golchinfar (@DavidGFar)](https://x.com/DavidGFar/status/1799955148091093006?t=RHvvaAqDuY1fBIbm-_gEmA&s=19): 基于 @TheEricHartford、@LucasAtkins7、@FernandoNetoAi 和我开发的一种名为 Spectrum 的新型 LLM 训练技术，我们可以在 @VAGOsolutions 构建一个新的强大的 SauerkrautLM。它基于 @Mic...
- [Supervised Fine-tuning Trainer](https://huggingface.co/docs/trl/main/en/sft_trainer#train-on-completions-only): 未找到描述
- [Nvidia's Stock Split Happens on June 7. Here's What to Expect.](https://finance.yahoo.com/news/nvidias-stock-split-happens-june-084500817.html): 对于 Nvidia 及其股东来说，这是一个令人兴奋的时刻。
- [Release Continued Pretraining · unslothai/unsloth](https://github.com/unslothai/unsloth/releases/tag/June-2024): 持续预训练（Continued pretraining）：你现在可以使用 Unsloth 进行持续预训练。更多详情请参阅 https://unsloth.ai/blog/contpretraining！持续预训练的速度快 2 倍，且显存（VRAM）占用比...减少 50%。
- [GitHub - SilasMarvin/lsp-ai: LSP-AI is an open-source language server that serves as a backend for AI-powered functionality, designed to assist and empower software engineers, not replace them.](https://github.com/SilasMarvin/lsp-ai): LSP-AI 是一个开源语言服务器，作为 AI 驱动功能的后端，旨在辅助和赋能软件工程师，而非取代他们。- SilasMarvin/lsp-ai
- [Home](https://github.com/unslothai/unsloth/wiki#continued-pretraining--finetuning-the-lm_head-and-embed_tokens-matrices): 以 2-5 倍的速度和减少 80% 的内存微调 Llama 3、Mistral、Phi 和 Gemma LLM - unslothai/unsloth
- [llama.cpp/examples/main at master · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/tree/master/examples/main): 使用 C/C++ 进行 LLM 推理。通过在 GitHub 上创建账号，为 ggerganov/llama.cpp 的开发做出贡献。
- [GitHub - petergpt/Fine-Tuning-Memorisation-Experiement-GPT-35: Use of a fine-tuned model](https://github.com/petergpt/Fine-Tuning-Memorisation-Experiement-GPT-35): 微调模型的使用。通过在 GitHub 上创建账号，为 petergpt/Fine-Tuning-Memorisation-Experiement-GPT-35 的开发做出贡献。
- [Cognitive Computations](https://github.com/cognitivecomputations/): Cognitive Computations 拥有 17 个可用代码库。在 GitHub 上关注他们的代码。
- [laserRMT/laserQlora.ipynb at main · cognitivecomputations/laserRMT](https://github.com/cognitivecomputations/laserRMT/blob/main/laserQlora.ipynb): 这是我们对 'Layer Selective Rank Reduction' 的自行实现 - cognitivecomputations/laserRMT
- [Google Colab](https://colab.research.google.com/drive/1XamvWYinY6FOSX9GLvnqSjjsNflxdhNc?usp=sharing#scrollTo=LjY75GoYUCB8): 未找到描述
- [Google Colab](https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing): 未找到描述
- [Add SophiaG. by guilt · Pull Request #24338 · huggingface/transformers](https://github.com/huggingface/transformers/pull/24338/commits/710f5939b018340be11c6792df0ffbcd3265f1e3): 此 PR 的作用是什么？这是一个草稿 PR，展示了如何使用 Transformers 测试 Sophia。这绝非生产就绪版本，且肯定需要考虑许可问题。但是，如果有人需要...
- [Bug: QWEN2 quantization GGML_ASSERT · Issue #7805 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/issues/7805#issuecomment-2153349963): 发生了什么？在尝试将 Qwen2 7B instruct 量化为 IQ2_XS 时，我遇到了以下断言错误：GGML_ASSERT: ggml-quants.c:12083: grid_index >= 0 有什么我可以提供以协助调试的吗？正在上传文件...

### **Unsloth AI (Daniel Han) ▷ #**[**random**](https://discord.com/channels/1179035537009545276/1179039861576056922/1248781145559138424) (25 条消息🔥):

- **ETDisco 讨论 QLoRA vs DoRA**：一位成员询问了 **QLoRA** 和 **DoRA** 之间的区别，另一位成员解释说，**DoRA** 学习的是 **LoRA** 矩阵本身的缩放向量（scaling vector），而 **QLoRA** 指的是 4 bit 版本的 **LoRA**。他们还提到了 **QDoRA**，即 4 bit 版本的 **DoRA**。
- **模型算术技巧**：一条分享的[推文](https://x.com/_xjdr/status/1799518422235304248)讨论了如何通过提取 L3 base 和 L3 instruct 的权重差值，对 base 进行微调，然后在进行额外的 finetuning 之前将 instruct 的差值加回去，从而获得更好的性能。这引发了关于模型合并（model merging）的细微差别和“黑魔法”的讨论。
- **Codegemma 微调版重新设计**：一位用户寻求关于名为 **Codegemma** 的新微调版图形设计的反馈。另一位成员提供了详细建议，例如使用白色背景、将文本与方块对齐，以及可能在方块中加入红色或绿色。

**提到的链接**：[来自 xjdr (@_xjdr) 的推文](https://x.com/_xjdr/status/1799518422235304248)：实用技巧：如果你提取 L3 base 和 L3 instruct 的权重差值，微调 base，然后将 instruct 的差值重新加在上面，再进行一点额外的 finetuning，通常会……

---

### **Unsloth AI (Daniel Han) ▷ #**[**help**](https://discord.com/channels/1179035537009545276/1179777624986357780/1248724410094063716) (194 messages🔥🔥):

- **Unsloth 在 Google Colab 上的内存问题**：用户报告 Unsloth 在 Google Colab 上消耗过量 RAM，导致崩溃。建议使用 `auto_find_batch_size = True` 来缓解 RAM 问题。
- **Meta-llama3 微调的数据集准备**：新手询问使用自定义数据集微调 Meta-llama3 模型的正确格式和系统要求。建议在 2k context 下使用 12GB VRAM，并参考 [Unsloth wiki](https://github.com/unslothai/unsloth/wiki) 获取更多信息。
- **高性价比聊天机器人部署讨论**：用户咨询如何在不产生高额 OpenAI API 费用的情况下部署 Gen AI 聊天机器人。建议包括使用 4-bit 量化模型，并探索像 [aphrodite-engine](https://github.com/PygmalionAI/aphrodite-engine) 这样的开源工具以寻求成本效益高的解决方案。
- **训练期间的 Wandb 和驱动程序问题**：用户遇到了 `wandb` 和 NVIDIA 驱动程序的问题，导致内存相关错误和崩溃。临时解决方案包括禁用 `wandb` 和回滚 NVIDIA 驱动程序。
- **微调与评估障碍**：用户分享了与微调和评估阶段相关的挑战，例如评估设置不当导致显存溢出 (OOM) 错误。一位用户建议在 HuggingFace Transformers GitHub 上提交 issue，以解决评估期间双数据集加载的问题。

**提到的链接**：

- [ksw1/step-50-1k-dpo3 · Hugging Face](https://huggingface.co/ksw1/step-50-1k-dpo3)：未找到描述
- [Fine tuning Optimizations - DoRA, NEFT, LoRA+, Unsloth](https://www.youtube.com/watch?v=ae2lbmtTY5A)：➡️ ADVANCED-fine-tuning Repo: https://trelis.com/advanced-fine-tuning-scripts/➡️ ADVANCED-inference Repo: https://trelis.com/enterprise-server-api-and-infere...
- [ksw1/step-50-1k-dpo3-take2 at main](https://huggingface.co/ksw1/step-50-1k-dpo3-take2/tree/main)：未找到描述
- [Release Continued Pretraining · unslothai/unsloth](https://github.com/unslothai/unsloth/releases/tag/June-2024)：增量预训练（Continued pretraining）发布。你现在可以使用 Unsloth 进行增量预训练。详见 https://unsloth.ai/blog/contpretraining 了解更多详情！增量预训练速度提升 2 倍，且比...节省 50% VRAM。
- [CUDA semantics — PyTorch 2.3 documentation](https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)：未找到描述
- [transformers/src/transformers/trainer.py at 25245ec26dc29bcf6102e1b4ddd0dfd02e720cf5 · huggingface/transformers](https://github.com/huggingface/transformers/blob/25245ec26dc29bcf6102e1b4ddd0dfd02e720cf5/src/transformers/trainer.py#L3542>)：🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的先进机器学习库。 - huggingface/transformers
- [GitHub - PygmalionAI/aphrodite-engine: PygmalionAI's large-scale inference engine](https://github.com/PygmalionAI/aphrodite-engine)：PygmalionAI 的大规模推理引擎。通过在 GitHub 上创建账号为 PygmalionAI/aphrodite-engine 的开发做出贡献。
- [Home](https://github.com/unslothai/unsloth/wiki)：微调 Llama 3, Mistral, Phi & Gemma LLMs 速度提升 2-5 倍，内存占用减少 80% - unslothai/unsloth
- [no title found](https://download.pytorch.org/whl/cu121)：未找到描述
- [Type error when importing datasets on Kaggle · Issue #6753 · huggingface/datasets](https://github.com/huggingface/datasets/issues/6753)：描述 Bug：尝试运行 import datasets print(datasets.__version__) 时产生以下错误 TypeError: expected string or bytes-like object，看起来无法找到 val...
- [GitHub - huggingface/peft: 🤗 PEFT: State-of-the-art Parameter-Efficient Fine-Tuning.](https://github.com/huggingface/peft.git)：🤗 PEFT：先进的参数高效微调技术。 - huggingface/peft

---

### **Unsloth AI (Daniel Han) ▷ #**[**showcase**](https://discord.com/channels/1179035537009545276/1179779344894263297/1249410284616159365) (21 messages🔥):

- **瑞典语 LORA 模型在 Hugging Face 发布**：一位开发者分享了基于 **Llama 3 Instruct** 的瑞典语 LORA 模型的发布，该模型使用来自瑞典语维基百科的数据集针对提示词问答进行了微调。该模型名为 [Bellman](https://huggingface.co/neph1/llama-3-instruct-bellman-8b-swedish)，加入了来自翻译后的代码反馈数据集和叙述性内容的问题，但在故事生成方面表现尚不理想。
- **小语言模型的训练见解与问题**：讨论围绕构建低资源语言模型的挑战以及解决合成故事生成的问题展开。一位开发者提到，他们在训练中仅使用十个短篇故事就获得了较好的效果，希望更多的数据能提升性能。
- **从瑞典语维基百科创建数据集**：创建问答数据集的方法包括爬取瑞典语维基百科，并使用 GPT3.5 Turbo 或 Mixtral 直接生成瑞典语问答。开发者指出，模型在有提示词接地（prompt groundings）的情况下表现更好，并意识到由于语言相似性和训练数据的原因，瑞典语相较于芬兰语可能具有潜在优势。
- **翻译与语法准确性的挑战**：开发者讨论了在使用 GPT-4 等模型进行推理时保持语法准确性的困难，指出尽管尝试了 few-shot prompting，但问题依然频发。建议尝试非 OpenAI 模型，如 [GLM-4-9B](https://huggingface.co/THUDM/glm-4-9b-chat) 作为替代方案。

**提到的链接**：

- [neph1/llama-3-instruct-bellman-8b-swedish · Hugging Face](https://huggingface.co/neph1/llama-3-instruct-bellman-8b-swedish)：未找到描述
- [Hugging Face – 建设未来的 AI 社区。](https://huggingface.co)：未找到描述
- [THUDM/glm-4-9b-chat · Hugging Face](https://huggingface.co/THUDM/glm-4-9b-chat)：未找到描述

---

### **Unsloth AI (Daniel Han) ▷ #**[**community-collaboration**](https://discord.com/channels/1179035537009545276/1180144489214509097/1248980299296145552) (5 messages):

- **First Good Issues 咨询**：一位成员询问：*"你好，有什么我可以参与的 first good issues 吗？"* 另一位成员回复：*"私信你了！"*
- **项目需要文档**：一位成员提到：*"目前我们主要需要帮助支持模型或修复我们包中的 Bug，但这可能太复杂了。"* 他们还表示另一个重点是 *"编写文档"*，并对询问的成员是否能提供帮助表示不确定。

---

### **Unsloth AI (Daniel Han) ▷ #**[**notebooks**](https://discord.com/channels/1179035537009545276/1249414095359312087/1249414274208633022) (1 条消息):

- **关于我们的 Notebooks**：该帖子列出了针对各种模型的多个 Google Colab 和 Kaggle notebooks。模型包括 **Llama 3 (8B)**、**Mistral v0.3 (7B)**、**Phi-3**（medium 和 mini 变体）以及 **Gemma** 等。
- **丰富的 Google Colab 选项**：用户可以访问不同的 Google Colab notebooks，例如 [Llama 3 (8B)](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing) 和 [Mistral v0.3 (7B)](https://colab.research.google.com/drive/1_yNCks4BTD5zOnjozppphh5GzMFaMKq_?usp=sharing)。这些 notebooks 需要用户**登录**后才能访问。
- **提供 Kaggle 版本**：类似模型在 Kaggle 上也提供，例如 [Mistral v0.3 (7B)](https://www.kaggle.com/code/danielhanchen/kaggle-mistral-7b-unsloth-notebook) 和 [Llama 3 (8B)](https://www.kaggle.com/code/danielhanchen/kaggle-llama-3-8b-unsloth-notebook)。
- **征集更多 Notebooks**：该帖子邀请用户在指定的讨论频道 (<#1180144489214509097>) 中申请额外的 notebooks。“如果您希望我们添加其他 notebooks，请提出要求。”

**提到的链接**：

- [Google Colab](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing)): 未找到描述
- [Google Colab](https://colab.research.google.com/drive/1_yNCks4BTD5zOnjozppphh5GzMFaMKq_?usp=sharing)): 未找到描述
- [Google Colab](https://colab.research.google.com/drive/1hhdhBa1j_hsymiW9m-WzxQtgqTH_NHqi?usp=sharing)): 未找到描述
- [Google Colab](https://colab.research.google.com/drive/1lN6hPQveB_mHSnTOYifygFcrO8C1bxq4?usp=sharing)): 未找到描述
- [Google Colab](https://colab.research.google.com/drive/1mvwsIQWDs2EdZxZQF9pRGnnOvE86MVvR?usp=sharing)): 未找到描述
- [Google Colab](https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing)): 未找到描述
- [Google Colab](https://colab.research.google.com/drive/15gGm7x_jTm017_Ic8e317tdIpDG53Mtu?usp=sharing)): 未找到描述
- [Google Colab](https://colab.research.google.com/drive/1AZghoNBQaMDgWJpi4RbffGM1h6raLUj9?usp=sharing)): 未找到描述
- [Google Colab](https://colab.research.google.com/drive/11t4njE3c4Lxl-07OD8lJSMKkfyJml3Tn?usp=sharing)): 未找到描述
- [Google Colab](https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing)): 未找到描述
- [Google Colab](https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing)): 未找到描述
- [Google Colab](https://colab.research.google.com/drive/15F1xyn8497_dUbxZP4zWmPZ3PJx1Oymv?usp=sharing)): 未找到描述
- [Google Colab](https://colab.research.google.com/drive/19lwcRk_ZQ_ZtX-qzFP3qZBBHZNcMD1hh?usp=sharing)): 未找到描述
- [Google Colab](https://colab.research.google.com/drive/1aqlNQi7MMJbynFDyOQteD2t0yVfjb9Zh?usp=sharing)): 未找到描述

---

### **CUDA MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1189498205101109300/1248748837032689858) (52 条消息🔥):

- **CUDA Profiling 工具建议**：一位成员建议在训练期间使用 **nsys** 或 **ncu** 进行性能分析，对于详细的 kernel 性能分析，建议捕获单次前向和后向传递。他们还分享了一个[关于 kernel 性能分析的 YouTube 视频](https://youtu.be/fsC3QeZHM1U?si=g9HdJ_sDRKLO8Gmc)。
- **错过 Jake 的活动**：关于与 Jake 举行的一场活动进行了简短讨论，一位用户表示没看到活动通知，并确认活动已经结束。另一位成员提到该活动已在活动标签页上列出了一段时间。
- **寻求辅导**：一位用户询问关于寻找 PMPP 第 4 版等高级主题辅导老师的事宜，提到对目前的线性代数辅导老师很满意，并寻求在不同学科中具有类似教学质量的推荐。
- **机器学习配置的 GPU 和 CPU 建议**：成员们讨论了组装个人机器学习装备，建议 CPU 使用 **Ryzen 7950x** 或 **7950x3D**，GPU 选择具有大显存（VRAM）的型号，如 **3090** 或 **4090**。其他见解包括考虑支持 **AVX-512** 的 Intel **Xeon 处理器**以进行基于 CPU 的处理，并对双 4090 配置可能出现的问题提出了警告。
- **关于 CPU 中 AVX-512 支持的讨论**：深入讨论了 **AVX-512** 指令集在消费级和服务器级 CPU 中的优势和当前支持情况，包括潜在的权衡以及像 **Threadrippers** 和 **EPYCs** 这样的特定处理器。

**提到的链接**：

- [使用 NVIDIA Nsight Compute 进行 Kernel 性能分析简介](https://youtu.be/fsC3QeZHM1U?si=g9HdJ_sDRKLO8Gmc)：本节将介绍如何使用 NsightCompute 分析 NVIDIA GPU 上单个 GPU kernel 的性能。我们将演示一些简单的 c...
- [Llamafile 基准测试 - OpenBenchmarking.org](https://openbenchmarking.org/test/pts/llamafile)：未找到描述

---

### **CUDA MODE ▷ #**[**triton**](https://discord.com/channels/1189498204333543425/1189607595451895918/1248923235345502209) (14 条消息🔥):

- **FlagGems 引起关注**：一位成员分享了 [GitHub 上的 FlagGems 项目](https://github.com/FlagOpen/FlagGems)，将其描述为 *“一个用 Triton 语言实现的 LLM 算子库。”* 这迅速引起了频道内其他人的兴趣和赞赏。
- **通用 Kernel 维度查询**：一位用户询问了使用 Triton 处理通用 kernel 的最佳方法，特别是提到了没有固定维度的挑战。
- **对角矩阵构建**：另一位成员寻求关于如何将向量加载为对角矩阵的建议，并对 Hadamard 积后接矩阵-向量乘法的性能表示担忧。
- **最先进的 Triton Kernels**：一位用户询问了关于各种算子的最先进 Triton kernel 资源。他们被引导至一个[收录已发布 Triton kernel 的仓库](https://github.com/cuda-mode/triton-index)。
- **Triton 中的 BLOCK_SIZE 和分块 (Chunking)**：讨论了处理任意大小的 BLOCK_SIZE 以及 Triton 是否自动处理分块。会议澄清了用户需要为分块规约（chunk reduction）实现自己的 for 循环，因为 Triton 不会自动处理。

**提到的链接**：

- [GitHub - cuda-mode/triton-index: 收录已发布的 Triton kernel。](https://github.com/cuda-mode/triton-index)：收录已发布的 Triton kernel。欢迎在 GitHub 上为 cuda-mode/triton-index 的开发做出贡献。
- [GitHub - FlagOpen/FlagGems: FlagGems 是一个用 Triton 语言实现的 LLM 算子库。](https://github.com/FlagOpen/FlagGems)：FlagGems 是一个用 Triton 语言实现的 LLM 算子库。- FlagOpen/FlagGems

---

### **CUDA MODE ▷ #**[**torch**](https://discord.com/channels/1189498204333543425/1189607750876008468/1248758870147596369) (35 条消息🔥):

- **准确测量 `torch.compile` 编译时间**：成员们讨论了测量第一次运行可以评估编译时间，但它是与执行结合在一起的。减去第二个 batch 的时间可以帮助隔离编译时间。分享了 [故障排除指南](https://github.com/pytorch/pytorch/blob/main/docs/source/torch.compiler_troubleshooting.rst#cold-start-timing-and-cache-corruption-debugging) 以获取更多详细信息。
- **Inductor 性能仪表板脚本**：关于 PyTorch Inductor 性能仪表板脚本的查询被指向了 [这个 GitHub 目录](https://github.com/pytorch/pytorch/tree/main/benchmarks/dynamo)。
- **PyTorch Wrappers 的优势**：讨论强调了像 Lightning 和 fast.ai 这样的 wrapper 减少了样板代码，并提供了更高层级的抽象、常用模型和日志记录。在需要更深层次的自定义之前，它们作为起点非常有益。
- **编译整个训练过程**：由于 `DataLoader` 的原因，编译整个训练过程具有挑战性，但将其拆解或对特定步骤使用 `torch.compile` 进行部分编译会有所帮助。一位成员提到仅编译前向传播和损失计算取得了成功。
- **PyTorch 中的自定义 C++/CUDA 算子**：与 `torch.compile` 兼容的自定义算子允许全图编译。此类集成的示例可以在 [这里](https://github.com/pytorch/ao/pull/135) 找到。

**提到的链接**：

- [pytorch/benchmarks/dynamo at main · pytorch/pytorch](https://github.com/pytorch/pytorch/tree/main/benchmarks/dynamo)：Python 中具有强 GPU 加速的张量和动态神经网络 - pytorch/pytorch
- [[Tutorial] Custom C++ and CUDA Operators](https://docs.google.com/document/d/1-LdJZBzlxiF0Tm-8NfbyFvRJaofdwRgLcycXGmlIpS0/edit)：自定义 C++ 和 CUDA 算子。PyTorch 提供了大量的张量算子库（例如 torch.add, torch.sum 等）。然而，你可能希望向 PyTorch 引入新的自定义算子。这...
- [Custom CUDA extensions by msaroufim · Pull Request #135 · pytorch/ao](https://github.com/pytorch/ao/pull/135)：这是 #130 的可合并版本 - 我必须进行一些更新：增加除非使用 PyTorch 2.4+ 否则跳过测试的逻辑，增加如果 CUDA 不可用则跳过测试的逻辑，将 ninja 添加到开发依赖项中...
- [pytorch/docs/source/torch.compiler_troubleshooting.rst at main · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/main/docs/source/torch.compiler_troubleshooting.rst#cold-start-timing-and-cache-corruption-debugging)：Python 中具有强 GPU 加速的张量和动态神经网络 - pytorch/pytorch

---

### **CUDA MODE ▷ #**[**announcements**](https://discord.com/channels/1189498204333543425/1189640399476764692/1249075436412080263) (1 条消息):

- **令人兴奋的演讲者回归讨论高速扫描**：主持人宣布了一场由两位回归嘉宾演讲的会议，讨论让 scan “以光速运行”。他们之前分享了关于 **llm.cpp** 的见解，并准备进行另一场引人入胜的演讲。

---

### **CUDA MODE ▷ #**[**cool-links**](https://discord.com/channels/1189498204333543425/1189868872887705671/1249832156151152784) (1 条消息):

- **Bunnie Huang 做客 Amp Hour 播客**：最新一期的 The Amp Hour 播客邀请了以 Chumby 产品硬件设计工作而闻名的 **Bunnie Huang**。该剧集可以 [播放](http://traffic.libsyn.com/theamphour/TheAmpHour-84-BunniesBibelotBonification.mp3)、[下载](http://traffic.libsyn.com/theamphour/TheAmpHour-84-BunniesBibelotBonification.mp3)，并可通过 [Apple Podcasts](https://theamphour.com/feed/podcast/?mt=2&ls=1) 或 [RSS](https://theamphour.com/feed/podcast/) 订阅。
- **Hacking the Xbox**：Huang 还讨论了他的书 [Hacking the Xbox](https://www.amazon.com/gp)。这本书详细介绍了他在修改这款流行游戏机方面的经验和见解。

**提到的链接**：[An Interview with Bunnie Huang - Bunnie's Bibelot Bonification | The Amp Hour Electronics Podcast](https://theamphour.com/the-amp-hour-84-bunnies-bibelot-bonification/)：Bunnie Huang 加入了 Chris 和 Dave 的对谈，讨论他在中国的工作、他在硬件黑客方面的工作以及许多其他电子方面的精彩内容。

---

### **CUDA MODE ▷ #**[**jobs**](https://discord.com/channels/1189498204333543425/1190208177829068860/1249400353703723028) (5 条消息):

- **MLE 学习 CUDA 用于基于 GPU 的机器学习**：一位 30 多岁的成员正在向基于 GPU 的机器学习转型，并利用 PMPP 书籍、个人项目和研究论文实现等资源。他们对该话题的讨论持开放态度，因为他们目前在该专业领域还没有工作，但希望能顺利转型。
- **询问转型资源**：另一位成员询问了转型过程中使用了哪些资源。回复中简要提到了使用 Fatahalian 关于 GPU 架构的视频和其他学术材料。
- **学习 GPU 架构**：推荐了 Yong He 的 YouTube 频道来学习 GPU 架构，特别提到了 Fatahalian 的贡献。提供的链接是 [Yong He on YouTube](https://www.youtube.com/@csyonghe/videos)。

**提到的链接**：[Yong He](https://www.youtube.com/@csyonghe/videos)：未找到描述

---

### **CUDA MODE ▷ #**[**beginner**](https://discord.com/channels/1189498204333543425/1191300313928433664/1248799585283997716) (9 条消息🔥):

- **仅 Encoder 模型的参数搜索**：一位成员询问通常如何为仅 Encoder 的 PyTorch Transformer 模型进行参数搜索。遗憾的是，提供的消息中没有直接的回复。
- **Flash Attention Kernel 咨询**：一位成员询问需要阅读多少 PMPP 书籍内容才能编写 Flash Attention Kernel。提供的消息中没有对该问题的回复。
- **面向初学者的 NVIDIA GPU 推荐**：对于初学者，一位成员建议将 **RTX 4060Ti (16GB)** 作为学习用途的经济型选择，但也提到了在训练大型模型时可能存在的局限性。另一位成员建议使用过去三代中的任何 NVIDIA GPU，并强调即使是中级游戏 GPU 也支持 CUDA，且价格合理。
- **确保 torch.compile 的稳定性**：一位成员询问如何确保 `torch.compile` 在预热（warm-up）后不会在运行时重新编译，特别是在输入形状（shape）没有改变的情况下。消息中没有提供对该问题的回复。
- **关于复现 GPT-2 的 YouTube 视频**：一位成员分享了一个名为“让我们复现 GPT-2 (124M)”的 [YouTube 视频](https://www.youtube.com/watch?v=l8pRSuU81PU)，该视频涵盖了从零开始构建 GPT-2 网络并优化其训练的过程。

**提到的链接**：[Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)：我们从头开始复现 GPT-2 (124M)。这段视频涵盖了整个过程：首先我们构建 GPT-2 网络，然后我们优化它的训练，使其真正地……

---

### **CUDA MODE ▷ #**[**torchao**](https://discord.com/channels/1189498204333543425/1205223658021458100/1248769657775722597) (38 条消息🔥):

- **关于 FPGA 成本和 Ternary Models 的讨论**：成员们讨论了来自 Xilinx 和 Intel 的 FPGA 模型的高昂成本，价格从 $8K 到 $16K 不等（[Intel 规格](https://www.intel.com/content/www/us/en/products/sku/193921/intel-fpga-pac-d5005/specifications.html)）。他们提到使用这些设备来运行无需 matmul 的 Ternary Models（[论文](https://arxiv.org/pdf/2406.02528)）。
- **FP8 和混合精度格式**：一位成员提议使用混合 BF16/FP16 激活值和 FP8 权重，考虑到由于共享指数位（shared exponent bits）带来的快速类型转换。他们询问了在 torch.compile 中融合此类操作的方法，并收到了关于相关配置标志的反馈。
- **Torch.compile 配置**：讨论了在 torch.compile 中使用 `use_mixed_mm` 和 `force_mixed_mm` 等配置，并指出某些标志可能会导致问题或触发多个 kernel。一位成员还提到了生成 split-K matmul kernel 的问题。
- **对 Split-K Matmul 模板的需求**：成员们辩论了在 PyTorch 中为 matmul 提供 split-K 模板的必要性，特别是针对小 batch size 的情况。有人指出，非确定性（nondeterminism）和 epilogue fusion 的复杂性是目前的障碍。
- **基准测试与文档**：讨论了增强 torch.compile 和 torchao 的文档，包括期望的功能，如量化/稀疏化（quantization/sparsity）技术的对比表。还强调了最近为 GPT 模型新增的基准测试（[GitHub 链接](https://github.com/pytorch/ao/pull/276)）。

**提到的链接**：

- [GitHub: Let’s build from here](https://github.com/): GitHub 是超过 1 亿开发者共同塑造软件未来的地方。为开源社区做贡献，管理你的 Git 仓库，像专业人士一样审查代码，跟踪 bug 和功能...
- [ao/torchao/prototype/fp8/splitk_gemm.py at main · pytorch/ao](https://github.com/pytorch/ao/blob/main/torchao/prototype/fp8/splitk_gemm.py): 用于量化和稀疏化的原生 PyTorch 库 - pytorch/ao
- [ao/scripts/hf_eval.py at main · pytorch/ao](https://github.com/pytorch/ao/blob/main/scripts/hf_eval.py): 用于量化和稀疏化的原生 PyTorch 库 - pytorch/ao
- [Pull requests · pytorch/pytorch](https://github.com/pytorch/pytorch/pulls?q=_weight_int4pack_mm): Python 中具有强 GPU 加速的 Tensor 和动态神经网络 - Pull requests · pytorch/pytorch
- [Adding Llama to TorchAO by HDCharles · Pull Request #276 · pytorch/ao](https://github.com/pytorch/ao/pull/276): 摘要：此 PR 为 torchao 代码库增加了对 Llama 模型进行稳定评估/基准测试的功能。模型相关内容位于 torchao/_models/llama，评估部分已移至 _models/_eval.py...

---

### **CUDA MODE ▷ #**[**hqq**](https://discord.com/channels/1189498204333543425/1225499037516693574/) (1 条消息):

appughar: [https://github.com/ridgerchu/matmulfreellm](https://github.com/ridgerchu/matmulfreellm) 关于 Ternary Accumulation 的有趣工作。

---

### **CUDA MODE ▷ #**[**llmdotc**](https://discord.com/channels/1189498204333543425/1227345713348870156/1248713136832385074) (389 条消息🔥🔥):

- **训练模型对比**：成员们就各种模型展开了讨论，包括 **Qwen2**、**Llama2** 和 **Llama3**，讨论了学习率、数据集和计算成本等方面。分享了提及超参数的论文，如 [DeepSeek-AI](https://arxiv.org/abs/2401.02954)。
- **集成重叠计算（Overlapping Computations）**：讨论了优化技术，包括使计算任务更加异步，以实现梯度通信与计算的重叠。重叠计算的基准测试显示了性能提升：**当前设置**达到 69584 tok/s，而**优化后**的设置达到 71211 tok/s。
- **FineWebEDU 数据集的挑战**：发现了 FineWebEDU 的 shuffling 和样本质量影响训练损失（loss）模式的问题，引发了内部调查。成员注意到异常的 loss 模式，可能归因于未打乱或采样不当的数据。
- **LightEval 和模型转换**：由于安装和配置的复杂性，分享了运行 **LightEval** 进行评估指标测试时的挑战。详细介绍了使用脚本和示例将模型转换为 Hugging Face 格式的技巧（[脚本](https://gist.github.com/matthewdouglas/1c0833f7fa9adbc54e4f5dc09e2b59a2)）。
- **技术实现讨论**：成员们对各种实现提供了见解，例如将 **Cutlass** 集成到 **llm.c** 中以及 kernel 调用优化的重要性。分享了 [CutlassJun8](https://youtu.be/rFYVLeHVt4c) 等资源和草案供社区参考。

**提到的链接**：

- [eliebak/debug-cos-100B · Hugging Face](https://huggingface.co/eliebak/debug-cos-100B): 未找到描述
- [numpy memmap 内存使用 - 想要迭代一次](https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122): 假设我有一个保存在磁盘上的大矩阵。将其全部存储在内存中并不可行，所以我使用 memmap 来访问它 `A = np.memmap(filename, dtype='float32', mode='r'...`
- [mdouglas/llmc-gpt2-124M-400B · Hugging Face](https://huggingface.co/mdouglas/llmc-gpt2-124M-400B): 未找到描述
- [HuggingFaceFW/ablation-model-fineweb-edu · Hugging Face](https://huggingface.co/HuggingFaceFW/ablation-model-fineweb-edu#evaluation): 未找到描述
- [lighteval_tasks.py · HuggingFaceFW/fineweb at main](https://huggingface.co/datasets/HuggingFaceFW/fineweb/blob/main/lighteval_tasks.py#L12): 未找到描述
- [模型导出为 Hugging Face 格式并可选上传 by rhys101 · Pull Request #571 · karpathy/llm.c](https://github.com/karpathy/llm.c/pull/571): 这项工作延续了将 llm.c 模型导出为 Hugging Face 格式的工作（Issue 502）。这是一个独立的导出脚本，可将 GPT2 llm.c 二进制模型文件转换为本地 HF 模型目录...
- [添加 GPU CI 工作流文件 · karpathy/llm.c@73506df](https://github.com/karpathy/llm.c/actions/runs/9421152288/job/25978115408): 使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。
- [将 llm.c GPT-2 检查点转换为 HF safetensors](https://gist.github.com/matthewdouglas/1c0833f7fa9adbc54e4f5dc09e2b59a2): 将 llm.c GPT-2 检查点转换为 HF safetensors。GitHub Gist：即时分享代码、笔记和代码片段。
- [添加 GPU CI 工作流文件 · karpathy/llm.c@73506df](https://github.com/karpathy/llm.c/actions/runs/9421152288/job/25954604105?pr=570)): 使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。
- [GitHub - huggingface/lighteval: LightEval 是一个轻量级的 LLM 评估套件，Hugging Face 内部一直在将其与最近发布的 LLM 数据处理库 datatrove 和 LLM 训练库 nanotron 配合使用。](https://github.com/huggingface/lighteval/): LightEval 是一个轻量级的 LLM 评估套件，Hugging Face 内部一直在将其与最近发布的 LLM 数据处理库 datatrove 和 LLM 训练库 nanotron 配合使用。 - hug...
- [Dataloader - 由 gordicaleksa 引入随机性 · Pull Request #573 · karpathy/llm.c](https://github.com/karpathy/llm.c/pull/573): 正在实现完全随机的训练数据打乱... 此 PR 执行以下操作：每个进程都有不同的唯一随机种子，每个进程的训练数据加载器独立选择其起始分片...
- [GitHub - karpathy/llm.c: 使用简单、原始的 C/CUDA 进行 LLM 训练](https://github.com/karpathy/llm.c): 使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。
- [DeepSeek LLM: 以长期主义扩展开源语言模型](https://arxiv.org/abs/2401.02954): 开源大语言模型 (LLMs) 的快速发展确实令人瞩目。然而，先前文献中描述的扩展定律 (scaling law) 呈现出不同的结论，这给...蒙上了阴影。
- [CutlassJun8](https://youtu.be/rFYVLeHVt4c): 未找到描述
- [CutlassJun8p2](https://youtu.be/lWWKraqv-8E): 未找到描述
- [单 GPU 示例 · Issue #103 · NVIDIA/nccl](https://github.com/NVIDIA/nccl/issues/103): 我可以在单个 GPU 上使用 NCCL 吗？如果可以，能给我一个示例吗？

### **CUDA MODE ▷ #**[**bitnet**](https://discord.com/channels/1189498204333543425/1240586843292958790/1248918343289802853) (49 条消息🔥):

- **巧妙的位级技巧 (Sneaky Bit-Level Trickery)**：讨论了利用独特位表示（"0 = `11`, -1 = `01`, 1 = `00`"）的方法，通过三个 bitcounts 的差值来实现高效操作。识别并讨论了建议逻辑中一个潜在的 bug。
- **FPGA 成本与 A6000 ADA 的加速对比**：质疑了在某些操作中使用定制 FPGA 的成本效益，建议采用成本更低的 **A6000 ADA GPU** 等替代方案。强调了 **Bitblas 的 2-bit kernel** 已经提供了显著的加速。
- **NVIDIA Cutlass 与 Bit-Packing**：探索了 NVIDIA Cutlass 库的功能，确认其通过各种数据结构支持使用 **uint8** 格式进行任意 **nbit bit-packing**。分享了相关文档和 GitHub 仓库的链接，详见[此处](https://nvidia.github.io/cutlass/integer__subbyte_8h_source.html)和[此处](https://github.com/bytedance/decoupleQ)。
- **协作会议安排**：安排了一次会议讨论正在进行的项目，重点是使用 BitBlas 和其他 kernel 建立基准（baseline），并致力于 PR 和文档更新。分享了 [GitHub 链接](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#build-only-what-you-need)和[会议时间](https://github.com/pytorch/ao/pull/338)。
- **BitBlas 基准测试与见解**：发布了将 BitBlas 与 PyTorch 的 matmul fp16 进行比较的 [benchmark 结果](https://gist.github.com/mobicham/3ef2ef33d7f234f84f80249c41b6fae0)，指出 **BitBlas 4-bit** 操作显示出显著加速，但性能随输入大小和 batch-size 而异。强调了速度差异以及 **BitBlas 2-bit** 明显优于 4-bit 的使用场景。

**提到的链接**：

- [decoupleQ/csrc/w2a16.cu at main · bytedance/decoupleQ](https://github.com/bytedance/decoupleQ/blob/main/csrc/w2a16.cu)：一种针对 LLM 的量化算法。可以通过在 GitHub 上创建账号为 bytedance/decoupleQ 的开发做出贡献。
- [unilm/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf at master · microsoft/unilm](https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf)：跨任务、语言和模态的大规模自监督预训练 - microsoft/unilm
- [CUTLASS: cutlass::sizeof_bits< int4b_t > Struct Template Reference](https://nvidia.github.io/cutlass/structcutlass_1_1sizeof__bits_3_01int4b__t_01_4.html)：未找到描述
- [ao/test/prototype/mx_formats/test_mx_tensor.py at main · pytorch/ao](https://github.com/pytorch/ao/blob/main/test/prototype/mx_formats/test_mx_tensor.py#L39-L50)：用于量化和稀疏化的原生 PyTorch 库 - pytorch/ao
- [GitHub - bytedance/decoupleQ: A quantization algorithm for LLM](https://github.com/bytedance/decoupleQ)：一种针对 LLM 的量化算法。可以通过在 GitHub 上创建账号为 bytedance/decoupleQ 的开发做出贡献。
- [pytorch/CONTRIBUTING.md at main · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#build-only-what-you-need)：Python 中具有强 GPU 加速功能的张量和动态神经网络 - pytorch/pytorch
- [decoupleQ/decoupleQ/linear_w2a16.py at main · bytedance/decoupleQ](https://github.com/bytedance/decoupleQ/blob/main/decoupleQ/linear_w2a16.py)：一种针对 LLM 的量化算法。
- [Revert "Clean up FP6-LLM" by msaroufim · Pull Request #338 · pytorch/ao](https://github.com/pytorch/ao/pull/338)：还原了 #304，因为这破坏了我们的 manylinux 构建。虽然通过了 CI，但在二进制构建时失败了，所以我们应该有某种方式...
- [CUTLASS: integer_subbyte.h Source File](https://nvidia.github.io/cutlass/integer__subbyte_8h_source.html)：未找到描述

---

### **CUDA MODE ▷ #**[**arm**](https://discord.com/channels/1189498204333543425/1247232251125567609/1249036325630377996) (2 条消息):

- **ARM 讨论可能指向服务器芯片**：一位用户询问讨论的是 ARM 移动处理器还是 SoC。另一位成员澄清说，这更有可能是关于服务器芯片的，并分享了一个题为 ["96 ARM cores—it’s massive! (Gaming + Windows on Arm!)"](https://www.youtube.com/watch?v=ydGdHjIncbk) 的 YouTube 视频来说明这一点。

**提到的链接**： [96 ARM cores—it's massive! (Gaming + Windows on Arm!)](https://www.youtube.com/watch?v=ydGdHjIncbk)：世界上最快的 Arm 桌面设备能处理游戏吗？在此之前，它甚至能安装 Windows 吗？

---

### **HuggingFace ▷ #**[**general**](https://discord.com/channels/879548962464493619/879548962464493622/1248714285278953654) (509 条消息🔥🔥🔥):

<ul>
  <li><strong>2 billion parameters 模型引发性能讨论</strong>：用户讨论了 2 billion parameters 模型的计算需求。有人指出，“50GB 需要大量的计算能力，2x T4 是不够用的。”
  </li>
  <li><strong>API 成本和访问困惑</strong>：围绕特定 API（如 OpenAI）是否免费的问题引发了澄清，指出虽然某些模型可能是免费的，但“API 并不是免费的”。此外还有对 OpenAI 开放性的批评，称其为“closedai”。
  </li>
  <li><strong>Nvidia 和游戏推动技术领先地位</strong>：成员们讨论了 Nvidia 的市场举措和 AI 能力，结论是尽管存在“锁定生态系统（locked-in ecosystem）”，但由于游戏行业的需求和卓越的 AI 芯片，Nvidia 仍然不可或缺。
  </li>
  <li><strong>邮件 Token 泄露担忧</strong>：发生了一起用户意外暴露其 API token 的乌龙事件，随后大家建议使用环境变量来代替，以确保安全。
  </li>
  <li><strong>AI 活动见解分享</strong>：一位成员分享了他们在由 IEEE 大学俱乐部举办的“Artificial Intelligence National Summit 2.0”上的经历，他们在会上发表了关于 Hugging Face agents 的演讲。
  </li>
</ul>

**提到的链接**：

- [Models - Hugging Face](https://huggingface.co/models?pipeline_tag=text-generation&library=transformers.js&sort=trending)：未找到描述
- [Helsinki-NLP/opus-mt-ko-en · Hugging Face](https://huggingface.co/Helsinki-NLP/opus-mt-ko-en?text=%EC%A1%B0%EC%A0%95%EA%B3%BC+%EA%B5%B0%EB%B6%80%EC%9D%98+%EB%AA%87%EB%AA%87%EC%9D%84+%EC%A3%BD%EC%9D%B4%EB%8A%94+%EA%B2%83%EC%9D%B4+%EB%AC%B4%EC%97%87%EC%9D%B4+%EB%8C%80%EC%88%98%EA%B2%A0%EB%8A%94%EA%B0%80%3F)：未找到描述
- [Cat Kitty GIF - Cat Kitty Eepy cat - Discover & Share GIFs](https://tenor.com/view/cat-kitty-eepy-cat-eepy-kitten-eepy-gif-8186911474993351997)：点击查看 GIF
- [Remi Cadene (@RemiCadene) 的推文](https://x.com/RemiCadene/status/1799000991876178038)：我们让来自 @pollenrobotics 的 Reachy2 能够自主完成家务并与我们互动。它可以移动全身，包括脖子。甚至连狗狗都被惊艳到了！🐶 你也可以在 h...
- [Hugging Face - Learn](https://huggingface.co/learn)：未找到描述
- [Tom And Jerry Toy GIF - Tom and jerry Toy Play - Discover & Share GIFs](https://tenor.com/view/tom-and-jerry-toy-play-gif-13406805813729285523)：点击查看 GIF
- [使用 AWS PrivateLink 创建私有端点](https://huggingface.co/docs/inference-endpoints/guides/private_link)：未找到描述
- [transformers/examples/pytorch/summarization/run_summarization.py at 96eb06286b63c9c93334d507e632c175d6ba8b28 · huggingface/transformers](https://github.com/huggingface/transformers/blob/96eb06286b63c9c93334d507e632c175d6ba8b28/examples/pytorch/summarization/run_summarization.py)：🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的最先进机器学习库。 - huggingface/transformers
- [Daniel Jeffries (@Dan_Jeffries1) 的推文](https://x.com/dan_jeffries1/status/1794740447052525609?s=46)：我花了几个小时听 Dan Hendyrcks 的演讲，他是 SB 1047（即加州 AI 控制与中心化法案）背后的非营利 AI Safety 组织的负责人。我觉得他很有魅力、稳重且聪明...
- [nlp/texts at master · amephraim/nlp](https://github.com/amephraim/nlp/tree/master/texts)：通过在 GitHub 上创建账号来为 amephraim/nlp 的开发做出贡献。
- [从 PyTorch 转换 — Core ML Tools 指南](https://apple.github.io/coremltools/docs-guides/source/convert-pytorch.html)：未找到描述
- [Futurama Drinking GIF - Futurama Drinking Slurms - Discover & Share GIFs](https://tenor.com/MOSp.gif)：点击查看 GIF
- [transformers/src/transformers/models/t5/modeling_t5.py at 25245ec26dc29bcf6102e1b4ddd0dfd02e720cf5 · huggingface/transformers](https://github.com/huggingface/transformers/blob/25245ec26dc29bcf6102e1b4ddd0dfd02e720cf5/src/transformers/models/t5/modeling_t5.py#L552)：🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的最先进机器学习库。 - huggingface/transformers
- [GitHub - JosephPai/Awesome-Talking-Face: 📖 A curated list of resources dedicated to talking face.](https://github.com/JosephPai/Awesome-Talking-Face)：📖 一个专注于 Talking Face 的精选资源列表。 - JosephPai/Awesome-Talking-Face
- [transformers/src/transformers/models/t5/configuration_t5.py at main · huggingface/transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/configuration_t5.py#L27)：🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的最先进机器学习库。 - huggingface/transformers

- [Phi3 模型返回序列的差异](https://discuss.huggingface.co/t/difference-in-return-sequence-for-phi3-model/90823/5)：你好，由于你在上面的代码片段中没有调用 pipeline，能否请你提供一个更详细的可复现示例？我还看到了关于 Flash Attention 的警告，这可能解释了其中的差异。
- [transformers/src/transformers/models/t5/modeling_t5.py at 25245ec26dc29bcf6102e1b4ddd0dfd02e720cf5 · huggingface/transformers](https://github.com/huggingface/transformers/blob/25245ec26dc29bcf6102e1b4ddd0dfd02e720cf5/src/transformers/models/t5/modeling_t5.py#L340)：🤗 Transformers: 为 Pytorch, TensorFlow 和 JAX 提供的尖端机器学习。 - huggingface/transformers
- [由 abidlabs 修复 `/file=` 路由上的 SSRF 漏洞 · Pull Request #6794 · gradio-app/gradio](https://github.com/gradio-app/gradio/pull/6794)：由于 `/file` 路由曾用于执行 GET/HEAD 请求以确定文件路径是否为可能的 URL，因此存在被利用进行服务端请求伪造（SSRF）的潜在风险。这应该会减缓...
- [Models - Hugging Face](https://huggingface.co/models?pipeline_tag=text-to-image&library=onnx&sort=trending)：未找到描述
- [SadTalker: 为风格化音频驱动的单张图像人物面部动画学习真实的 3D 运动系数](https://sadtalker.github.io/)：SadTalker: 为风格化音频驱动的单张图像人物面部动画学习真实的 3D 运动系数
- [SadTalker - 由 vinthony 创建的 Hugging Face Space](https://huggingface.co/spaces/vinthony/SadTalker)：未找到描述
- [DocuGeniusRAG/lib/DocLoader.py at main · ManilShrestha/DocuGeniusRAG](https://github.com/ManilShrestha/DocuGeniusRAG/blob/main/lib/DocLoader.py)：DocuGeniusRAG 是我的一个个人项目。通过使用先进的 AI 技术进行深入且高效的解释，使用户能够从文本中提问并获得精确答案，从而改善文档交互...
- [Torch.embedding 失败并报错 RuntimeError: Placeholder storage has not been allocated on MPS device!](https://discuss.pytorch.org/t/torch-embedding-fails-with-runtimeerror-placeholder-storage-has-not-been-allocated-on-mps-device/152124/2)：你好，你是否确保已将模型和输入都移动到了 “mps” 设备上？
- [未找到标题](https://civitai.com/api/download/models/348913?type=Model&format=SafeTensor&size=full&fp=fp16&token=urtoken>)：未找到描述

---

### **HuggingFace ▷ #**[**today-im-learning**](https://discord.com/channels/879548962464493619/898619964095860757/1249563794926080083) (2 条消息):

- **Apple Vision Pro 开发需要 Mac Silicon**：“刚得知开发 Apple Vision Pro 需要 Mac Silicon”。一位成员表达了为 Apple 的 Vision Pro 设备开发应用需要专门硬件的需求。
- **GPU 持有量对比**：另一位成员澄清道，*“他比我们任何人都更 GPU rich（GPU 资源丰富），”*。这表明了一场涉及 GPU 可用性方面的能力或资源的讨论。

---

### **HuggingFace ▷ #**[**cool-finds**](https://discord.com/channels/879548962464493619/897390579145637909/1248722374870827028) (10 条消息🔥):

- **Torchtune 助力 LLM 微调**：查看 [Torchtune](https://github.com/pytorch/torchtune)，这是一个用于 LLM 微调的原生 PyTorch 库。它已在 GitHub 上发布，旨在增强您的 LLM 微调流程。
- **Ollama 拥有多功能的 LLM 特性**：探索 [Ollama](https://ollama.com/)，这是一个用于运行和定制 Llama 3、Phi 3、Mistral 和 Gemma 等 LLM 的平台。它兼容 macOS、Linux 和 Windows（预览版）。
- **Kaggle 上独特的 Alpaca 图像数据集**：将此 [alpaca 图像数据集](https://www.kaggle.com/datasets/shivamaggarwal513/dlai-alpaca-dataset/data) 用于您的图像分类项目。非常适合希望对羊驼进行分类的机器学习爱好者。
- **Langchain 和 DashScope Reranker 提升搜索效果**：在 Medium 上深入阅读《释放搜索猛兽：[Langchain 和 DashScope Reranker](https://medium.com/ai-advances/unleash-the-search-beast-langchain-and-dashscope-reranker-67cbfdbaed0b)》。增强您的搜索算法并发现先进的重排序技术。
- **聚焦可持续农业 AI 工具 FarmFriend**：揭秘专为可持续农业设计的 [FarmFriend Web 应用](https://farm-friend-v1.replit.app)，并集成了 iOS 快捷指令。关注 @twodogseeds 以获取更多关于 iOS AI 快捷指令的创新演示和见解。

**提到的链接**：

- [Ollama](https://ollama.com/)：快速上手并运行 LLM。
- [[DL.AI] Alpaca Dataset](https://www.kaggle.com/datasets/shivamaggarwal513/dlai-alpaca-dataset/data)：带标签的羊驼图像数据集。
- [Defaulter? | EDA | Preds: Acc = 97.49%](https://www.kaggle.com/code/jaymilindpadloskar/defaulter-eda-preds-acc-97-49)：使用 Kaggle Notebooks 探索并运行机器学习代码 | 使用来自 Loan-Dataset 的数据。
- [Tweet from two dog seeds (@twodogseeds)](https://x.com/twodogseeds/status/1799919392660349377)：🍏FarmFriend 对阵 Apple WWDC_24 iOS 快捷指令版🍏 —— 也就是所谓的遥遥领先。—— 正在执行 @Apple 明天将宣布的一些假设功能。就在今天！...
- [GitHub - pytorch/torchtune: A Native-PyTorch Library for LLM Fine-tuning](https://github.com/pytorch/torchtune)：一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。
- [styleguide](https://google.github.io/styleguide/pyguide.html)：Google 开源项目的风格指南。
- [Unleash the Search Beast: Langchain and DashScope Reranker](https://medium.com/ai-advances/unleash-the-search-beast-langchain-and-dashscope-reranker-67cbfdbaed0b)：Ankush k Singal

---

### **HuggingFace ▷ #**[**i-made-this**](https://discord.com/channels/879548962464493619/897390720388825149/1248716544192811008) (16 messages🔥):

- **Llama3-8b-Naija 发布**：一名成员宣布发布 **Llama3-8b-Naija_V1**，这是 **Llama3** 的一个微调版本，旨在以尼日利亚人的方式回答问题。更多详情可以在其 [Twitter 公告](https://twitter.com/saheedniyi_02/status/1798316987170648169?t=CHf8wnZDWtxvZC0QFcJ0Kg&s=19)中找到。
- **SimpleTuner v0.9.6.3 增强 MultiGPU 训练**：**SimpleTuner** 发布了更新（v0.9.6.3），提供了重大的 **MultiGPU 训练修复与优化**。该更新确保了训练时硬件资源的高效利用，可以在[此处](https://github.com/bghira/SimpleTuner/releases/tag/v0.9.6.3)查看。
- **Visionix Alpha 突破超写实边界**：推出了 **Visionix Alpha**，这是一款基于 **SDXL** 的新型超写实模型，在美学、解剖结构和自然景观方面有所改进。该模型可在 [Hugging Face](https://huggingface.co/ehristoforu/Visionix-alpha) 和 [CivitAI](https://civitai.com/models/505719) 上访问。
- **SoteDiffusion Wuerstchen3 发布**：推出了 **Würstchen V3** 的微调版本 **SoteDiffusion Wuerstchen3**，专注于动漫风格，在 600 万张图像上训练了 3 个 epochs。更多信息和访问方式请见[项目页面](https://huggingface.co/Disty0/sotediffusion-wuerstchen3)。
- **Chat With 'Em 正式上线**：**Chat With 'Em** 允许用户在 Hugging Face Spaces 上与来自 **Groq, Anthropic, OpenAI** 和 **Cohere** 的模型进行对话，使用 API key 即可在 Claude 和 GPT-3.5 等多种模型之间切换。点击[此处](https://huggingface.co/spaces/as-cle-bert/chat-with-em)体验该工具。

**提到的链接**：

- [Simple ImageCaptioning - peaceAsh 开发的 Hugging Face Space](https://huggingface.co/spaces/peaceAsh/Simple_ImageCaptioning)：未找到描述
- [Disty0/sotediffusion-wuerstchen3 · Hugging Face](https://huggingface.co/Disty0/sotediffusion-wuerstchen3)：未找到描述
- [不要做一个没有经验的自学开发者 ⛔️ 改进的迹象与建议 🧠](https://youtu.be/T_4EEU13y1c)：成为一名自学开发者很棒，但也可能导致一些坏习惯。在这段视频中，我们将剖析新手自学开发者最大的陷阱……
- [Release v0.9.6.3 MultiGPU 训练修复与优化 · bghira/SimpleTuner](https://github.com/bghira/SimpleTuner/releases/tag/v0.9.6.3)：更新内容：MultiGPU 训练改进。感谢 Fal.ai 提供硬件用于调查和改进这些领域：VAE 缓存现在可以可靠地在所有 GPU 上运行，而不会遗漏任何条目……

---

### **HuggingFace ▷ #**[**reading-group**](https://discord.com/channels/879548962464493619/1156269946427428974/1248719221270122526) (16 messages🔥):

- **AI 赋能物理模拟**：一位成员分享了一段关于 AI 如何应用于物理模拟的 YouTube 录像，并推荐观看：[Hugging Face Reading Group 23](https://www.youtube.com/watch?v=rVw4Zipmo1I&ab_channel=IsamuIsozaki)。
- **斯坦福关于防止模型崩溃（model collapse）的会议**：另一位成员宣布了由斯坦福研究人员领导的 LLM 阅读小组的最后一期会议，讨论一篇新论文，该论文为避免 AI 模型在其自身的合成数据上过度训练时出现模型崩溃提供了切实可行的解决方案。请在[此处](https://www.eventbrite.ca/e/llm-reading-group-march-5-19-april-2-16-30-may-14-28-june-11-tickets-851921368747?aff=oddtdtcreator)预约 6 月 11 日的会议。
- **探索用于指令数据的 WebInstruct**：一位成员建议关注一条关于从预训练数据中提取指令数据的推文，介绍了 WEBINSTRUCT，这是一个包含 1000 万个高质量指令对的数据集，无需人工标注或 GPT-4，使用爬取的网页数据创建。更多详情和资源可在 [Hugging Face](https://huggingface.co/papers/2405.03548)、[博客](https://tiger-ai-lab.github.io/MAmmoTH2/)和[数据集](https://huggingface.co/datasets/TIGER-Lab/WebInstructSub)上找到。

**提到的链接**：

- [Hugging Face Reading Group 23: AI for Physics. Hamilton Neural Networks/Lagrangian Neural Networks](https://www.youtube.com/watch?v=rVw4Zipmo1I&ab_channel=IsamuIsozaki)：演讲者：PS_Venom。往期演讲：https://github.com/isamu-isozaki/huggingface-reading-group
- [Philipp Schmid (@_philschmid) 的推文](https://x.com/_philschmid/status/1799718903922168142)：我们能从预训练数据中提取指令数据吗？WEBINSTRUCT 是一个拥有 1000 万条高质量指令的数据集，无需人工标注或 GPT-4，仅使用爬取的网页数据！👀 实现方式：1️⃣ Recall r...
- [LLM Reading Group (3月5日, 19日; 4月2日, 16日, 30日; 5月14日, 28日; 6月11日)](https://www.eventbrite.ca/e/llm-reading-group-march-5-19-april-2-16-30-may-14-28-june-11-tickets-851921368747?aff=oddtdtcreator)：来见见 LLM/NLP 研究领域一些开创性论文的作者，并听他们分享自己的工作。

---

### **HuggingFace ▷ #**[**computer-vision**](https://discord.com/channels/879548962464493619/922424143113232404/1248831564003938415) (6 messages):

- **旋转边界框（Rotated Bounding Boxes）的帮助**：一位用户正在寻求关于使用 x, y 坐标、宽度、高度和角度提取旋转边界框的帮助。他们提到在使用单应性矩阵（homography matrices）进行变换时遇到了问题，导致边界框不准确。
- **Gemini 1.5 表现优于其他模型**：分享的一条推文显示，[Gemini 1.5 Pro](http://aistudio.google.com) 在视频分析方面的表现显著优于包括 GPT-4o 在内的其他模型。相关链接：[SavinovNikolay 的推文](https://x.com/SavinovNikolay/status/1797621888279355740)，[Video-MME 项目](https://video-mme.github.io/)，以及 [Arxiv 摘要](https://arxiv.org/abs/2405.21075)。
- **轻松搜索 CVPR 2024 论文**：一个新创建的应用提供了对 CVPR 2024 论文摘要的语义搜索功能。该应用可通过[此处](https://huggingface.co/spaces/pedrogengo/CVPR2024_search_papers)访问。
- **关于 Label Studio ML 后端的咨询**：一位用户询问是否有人有使用 Label Studio ML 后端的经验。聊天中未提供进一步的背景或回复。

**提到的链接**：

- [CVPR2024 Search Papers - pedrogengo 开发的 Hugging Face Space](https://huggingface.co/spaces/pedrogengo/CVPR2024_search_papers)：未找到描述
- [来自 Nikolay Savinov 🇺🇦 (@SavinovNikolay) 的推文](https://x.com/SavinovNikolay/status/1797621888279355740)：在 http://aistudio.google.com 免费试用 Gemini 1.5 的视频理解功能。引用 Aran Komatsuzaki (@arankomatsuzaki) 的话：Video-MME：首个多模态综合评估基准...

---

### **HuggingFace ▷ #**[**NLP**](https://discord.com/channels/879548962464493619/922424173916196955/1248998284098539610) (19 messages🔥):

- **理解基于 RAG 的聊天机器人**：一位用户询问了关于构建基于 RAG 的聊天机器人是否需要问答对（Q&A pairs）数据集的问题。成员们澄清说，RAG 的工作原理是检索前 k 个相关文档并将其包含在上下文（context）中，建议从 PDF 中提取文本作为起点，并提到如果模型能够遵循指令，则可能不需要微调（fine-tuning）。
- **MyResumo AI 驱动的简历生成器**：一位用户分享了他们的项目 **MyResumo**，这是一个利用 LLM 技术根据特定职位描述生成定制简历的 AI 驱动工具。他们提供了 [GitHub 链接](https://github.com/AnalyticAce/MyResumo) 和 [LinkedIn 演示](https://www.linkedin.com/posts/shalom-dosseh-4a484a262_ai-resume-career-activity-7196073388098928641-hG1N?utm_source=share&utm_medium=member_android)。
- **模型分析与可解释性的建议**：一位新成员请求关于模型分析和可解释性（interpretability）的资源。作为回应，一位成员推荐了来自 [ACL Anthology](https://aclanthology.org/P19-1452/) 的一篇关于 BERT 的研究论文，以及 [HuggingFace](https://huggingface.co/collections/Vipitis/interpretability-655e24a2b53face4cf2b3cc8) 上的另一个可解释性论文合集。
- **托管具有 API 访问权限的 Llama 模型**：一位用户询问了托管具有 API 访问权限的 Llama 模型以便在多个应用程序中使用的最佳方式。聊天中未提供具体的后续行动或全面的解决方案。
- **PyTorch 与 TensorFlow 模型中的错误处理**：一位用户在使用带有 PyTorch 张量的 TensorFlow GPT2 模型时遇到了错误，导致 ValueError。建议在分词器（tokenizers）中设置 `return_tensors="tf"` 以解决类型不匹配问题。

**提到的链接**：

- [Interpretability - Vipitis 合集](https://huggingface.co/collections/Vipitis/interpretability-655e24a2b53face4cf2b3cc8)：未找到描述
- [BERT Rediscovers the Classical NLP Pipeline](https://aclanthology.org/P19-1452/)：Ian Tenney, Dipanjan Das, Ellie Pavlick。第 57 届计算语言学协会（ACL）年会论文集。2019。

---

### **HuggingFace ▷ #**[**diffusion-discussions**](https://discord.com/channels/879548962464493619/1009713274113245215/1249251108036284467) (13 条消息🔥):

- **使用条件化 UNet2D 模型进行训练**：一位用户询问了关于使用条件化版本的 UNet2D 模型进行训练的示例。分享了一个有用的资源：[text-to-image 训练示例](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py)。
- **使用 SDXL 将文本印入图像**：一位用户询问 SDXL 是否可以将一个图像中的文本印到另一个图像上。建议使用 [Image-to-Image Inpainting Stable Diffusion 社区流水线](https://github.com/huggingface/diffusers/tree/main/examples/community#image-to-image-inpainting-stable-diffusion) 作为解决方案。
- **训练期间 MFU 的计算**：一位成员询问了支持 MFU 计算的计划。明确表示目前官方训练脚本中没有此功能，但建议通过 Fork 并修改仓库作为权宜之计。
- **SDXL 训练方法的差异**：讨论了用于微调 SDXL 模型的 HuggingFace 脚本与预制自定义 Notebook 之间的细微差别和权衡。指出 HuggingFace 脚本大多是示例，而自定义 Notebook 可能会提供更先进和多样的微调策略，但避开了具体的推荐。

**提到的链接**：

- [diffusers/examples/community at main · huggingface/diffusers](https://github.com/huggingface/diffusers/tree/main/examples/community#image-to-image-inpainting-stable-diffusion))：🤗 Diffusers: 在 PyTorch 和 FLAX 中用于图像和音频生成的 SOTA 扩散模型。 - huggingface/diffusers
- [diffusers/examples/text_to_image/train_text_to_image.py at main · huggingface/diffusers](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py)：🤗 Diffusers: 在 PyTorch 和 FLAX 中用于图像和音频生成的 SOTA 扩散模型。 - huggingface/diffusers

---

### **LM Studio ▷ #**[**💬-general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1248727945648799745) (221 条消息🔥🔥):

- **对 LM Studio 中图像嵌入（Image embeddings）的好奇**：一位成员询问 LM Studio 是否可以从图像生成 embeddings，并将其与他们使用的 daanelson/imagebind 进行了比较。其他人提到 llama.cpp 尚不支持 vision embeddings，尽管来自 nomic 和 jina 的新版本可能很快就会支持。
- **Qwen2 模型集成问题**：一些成员遇到了 Qwen2 模型的兼容性问题，并参考了一个 GitHub [pull request](https://github.com/ggerganov/llama.cpp/pull/7835) 以在 llama.cpp 中添加支持。据指出，支持将在下一个 LM Studio 版本发布后合并。
- **RTX 4070 性能反馈**：一位成员分享了他们在 RTX 4070 上运行 Llama 3 的经验，达到了 50t/s。考虑到性能限制，他们咨询了介于 8B 和 70B 之间的模型建议。
- **关于 GPU 缺失的困惑**：一位用户在机器上遇到了 GPU offload 和模型加载困难，引发了关于检查设置和确保 NVIDIA 驱动程序更新的故障排除讨论。
- **对通过 Web 界面使用 LM Studio 的兴趣**：几位成员探讨了远程使用 LM Studio 的方法，讨论了为模型交互创建 Web 界面的可行性，但由于本地服务器限制以及可能需要自定义解决方案而面临限制。

**提到的链接**：

- [bartowski/Codestral-22B-v0.1-GGUF · Hugging Face](https://huggingface.co/bartowski/Codestral-22B-v0.1-GGUF)：未找到描述
- [Obsidian AppImage - 发现 SUID 沙盒辅助二进制文件，但配置不正确](https://askubuntu.com/questions/1512287/obsidian-appimage-the-suid-sandbox-helper-binary-was-found-but-is-not-configu)：升级到 24.04 后，当我尝试运行此 Electron AppImage 应用程序文件时，收到“The SUID sandbox helper binary was found, but is not configured correctly”消息。整个错误日志...
- [来自 Daniel Jeffries (@Dan_Jeffries1) 的推文](https://x.com/dan_jeffries1/status/1794740447052525609?s=46)：我花了几个小时听 Dan Hendyrcks 的演讲，他领导着 SB 1047（即加州 AI 控制与中心化法案）背后的非营利 AI Safety 组织。我觉得他很有魅力、稳重、聪明……
- [GitHub - VideotronicMaker/LM-Studio-Voice-Conversation: 用于通过本地 LLM 进行 LM Studio 增强型语音对话的 Python 应用。使用 Whisper 进行语音转文字，并提供一个注重隐私、易于访问的界面。](https://github.com/VideotronicMaker/LM-Studio-Voice-Conversation)：用于通过本地 LLM 进行 LM Studio 增强型语音对话的 Python 应用。使用 Whisper 进行语音转文字，并提供一个注重隐私、易于访问的界面。
- [更新：由 legraphista 支持 Qwen2-57B-A14B · Pull Request #7835 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/7835)：为 Qwen2-57B-A14B 添加了对键 `moe_intermediate_size` 和 `shared_expert_intermediate_size` 的支持。注意：由于 `self.gguf_writer.add_feed_forward_length` 被 `super().set_gguf_par...` 调用。
- [GitHub - oobabooga/text-generation-webui: 一个用于大语言模型（Large Language Models）的 Gradio Web UI。支持 transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama 模型。](https://github.com/oobabooga/text-generation-webui/)：一个用于大语言模型（Large Language Models）的 Gradio Web UI。支持 transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama 模型。
- [Lpbank Credit Card GIF - Lpbank Credit card Card - 发现并分享 GIF](https://tenor.com/view/lpbank-credit-card-card-animation-rotation-gif-18274263903803752554)：点击查看 GIF

---

### **LM Studio ▷ #**[**🤖-models-discussion-chat**](https://discord.com/channels/1110598183144399058/1111649100518133842/1248928303759364128) (34 messages🔥):

- **不要混淆，要反混淆 (Don't Obfuscate, De-Obfuscate)**：一位用户幽默地建议不要给代码加注释，并故意重命名变量以使代码变得混乱。另一位成员回应道 *"LLM 非常擅长反混淆 (unobfuscation)"*，并表示这让他们会心一笑。
- **用于 Visual Novels 翻译的 AI 排行榜**：[VNTL Leaderboard](https://huggingface.co/datasets/lmg-anon/vntl-leaderboard) 根据将日语 Visual Novels 翻译成英语的能力对 LLM 进行排名。评分基于 128 条 Visual Novels 语句的参考翻译与生成翻译之间的平均余弦相似度。
- **Gemini Nano 模型讨论**：一位用户分享了 Gemini Nano 4bit 模型的下载链接，但指出将其转换为 gguf 格式存在困难。另一位成员建议，它需要先转换为 safetensors 格式，并且由于架构未知，可能无法在 llama.cpp 或 LM Studio 中运行。
- **用于图像编辑的 Stable Diffusion**：有人询问关于在不改变整个图像的情况下编辑特定部分的模型。得到的建议是使用 [Stable Diffusion](https://github.com/AUTOMATIC1111/stable-diffusion-webui)，它允许用户对图像的部分区域进行遮罩 (mask)，并仅在这些区域生成更改。
- **模型合并帮助**：一位用户分享了他们第一个成功合并的模型 [Boptruth-NeuralMonarch-7B](https://huggingface.co/theprint/Boptruth-NeuralMonarch-7B)，该模型合并了两个特定的模型。建议使用 alpaca chat template 以获得最佳效果。

**提到的链接**：

- [mradermacher/Llama3-FiditeNemini-70B-Source-i1-GGUF · Hugging Face](https://huggingface.co/mradermacher/Llama3-FiditeNemini-70B-Source-i1-GGUF)：未找到描述
- [no title found](http://edgedl.me.gvt1.com/edgedl/release2/chrome_component/pxvh7nzt2kgw734yv5s2t5zyzi_2024.6.5.2205/fklghjjljmnfjoepjmlobpekiapffcja_2024.6.5.2205_all_adwfh7dtkja74pd3zhdx6wlr2w6q.crx3)：未找到描述
- [theprint/Boptruth-NeuralMonarch-7B · Hugging Face](https://huggingface.co/theprint/Boptruth-NeuralMonarch-7B)：未找到描述
- [lmg-anon/vntl-leaderboard · Datasets at Hugging Face](https://huggingface.co/datasets/lmg-anon/vntl-leaderboard)：未找到描述

---

### **LM Studio ▷ #**[**🧠-feedback**](https://discord.com/channels/1110598183144399058/1113937247520170084/1249013313795526747) (13 messages🔥):

- **LM Studio 无法生成图像**：针对有关生成图像的问题，一位用户澄清道 *"这不是 LM Studio 能完成的任务。"*
- **需要 stop strings 功能**：一位用户强调，*"这个软件真的需要支持在遇到 stop strings 时立即停止，"* 另一位用户补充了关于 llama.cpp 后端持续存在的问题细节，并建议提交 issue ticket。
- **赞扬与对 closed-source 的担忧**：用户表达了对 LM Studio 的整体赞赏，但也指出了对其作为 closed-source 软件的担忧。*"我真的很喜欢 LM Studio。这么棒的软件。唯一困扰我的是它是 closed-source 的。"*
- **文档导入限制**：一位用户询问关于导入文档进行 AI 交互的问题，得到的澄清是目前不支持此功能，并建议查阅 [FAQs](https://link.to/faqs)。
- **`mmap` 标志减少内存占用**：经过测试，一位用户报告称在 LM Studio 中禁用 `mmap` 标志可以显著减少内存使用，且不会影响 token 生成速度。分享了修改配置的说明，并强调 *"在两种配置下，首个 token 的生成速度是相同的。"*

---

### **LM Studio ▷ #**[**📝-prompts-discussion-chat**](https://discord.com/channels/1110598183144399058/1120489168687087708/1249407541256126464) (1 messages):

- **专注于正面指令以获得更好结果**：一位成员指出了 Prompt engineering 中的一个重要实践，强调 *“你应该告诉它该做什么，而不是不该做什么。”* 这一技巧突出了提供清晰、正面指令对于通过 AI 模型获得理想结果的价值。

---

### **LM Studio ▷ #**[**⚙-configs-discussion**](https://discord.com/channels/1110598183144399058/1136793122941190258/1249605632773062656) (4 messages):

- **Function Calling 沟通误解已解决**：在对关于 Function Calling 的陈述产生初步困惑后，一位成员提到，*"拨云见日后，我意识到他是什么意思了，"* 澄清他们在讨论后理解了相关解释。
- **NVIDIA GT 1030 兼容性问题**：一位新成员询问是否可以在 **LM Studio** 中使用旧的 NVIDIA GT 1030 GPU。他们分享了 GPU 设置的详细规格，表示找不到利用该 GPU 的配置，可能是因为它已经过时了。

### **LM Studio ▷ #**[**🎛-hardware-discussion**](https://discord.com/channels/1110598183144399058/1153759714082033735/1248717181508915200) (228 条消息🔥🔥):

- **使用临时方案冷却 Tesla P40 的挑战**：一位用户收到了他们的 **Tesla P40** 并尝试使用反向气流进行冷却，但由于 PC 机箱空间限制发现效果不佳。社区建议包括使用旧的 Mac 散热风扇以及参考详细指南 [Mikubox Triple-P40 build](https://rentry.org/Mikubox-Triple-P40)，而另一位用户表示使用定制的 3D 打印导风罩取得了成功。
- **在 LM Studio 中处理多 GPU 设置**：用户讨论了 **LM Studio** 在处理多 GPU 设置时的局限性，这导致了性能瓶颈。一位用户指出 LM Studio 在多 GPU 之间分配大模型的效率低下，而另一位用户强调了 **ollama** 卓越的多 GPU 支持。
- **解决 P40 及其他 GPU 的驱动问题**：一位用户在安装 Tesla P40 驱动时遇到了挑战，且不能干扰其 GTX 3060Ti。他们分享了解决方案，如从 [NVIDIA](https://www.nvidia.com/download/driverResults.aspx/129483/en-us) 手动安装驱动，以及使用特定的指南如 [JingShing's GitHub](https://github.com/JingShing/How-to-use-tesla-p40)。
- **为 AI 优化硬件**：讨论涵盖了 AI 任务的最佳硬件配置，建议包括二手 3090 GPU、能以较低价格提供服务器级性能的 **Tesla P40**，以及高吞吐量内存的重要性。分享了 [ipex-llm](https://github.com/intel-analytics/ipex-llm) 等链接，展示了使用 Intel GPU 进行 LLM 加速。
- **探索其他 AI 工具及兼容性**：成员们询问了在 LM Studio 中集成图像生成和文本转语音模型的问题，并讨论了用于 Stable Diffusion 的 **ComfyUI**、**Automatic1111** 和 **Foooocus** 等工具。一位用户分享了 [Civitai](https://civitai.com) 的链接，用于下载配合 AI 工具使用的模型。

**提到的链接**：

- [Tesla Driver for Windows | 386.07 | Windows 10 64-bit | NVIDIA](https://www.nvidia.com/download/driverResults.aspx/129483/en-us/)：下载适用于 Windows 10 64 位系统的英文（美国）Tesla 驱动程序。发布日期 2018.1.9。
- [program.pinokio](https://program.pinokio.computer/#/?id=windows)：Pinokio 编程手册。
- [Civitai: The Home of Open-Source Generative AI](https://civitai.com)：探索数千个高质量的 Stable Diffusion 模型，分享您的 AI 生成艺术，并与充满活力的创作者社区互动。
- [резатьжелезо распилитьжелезо GIF - Резатьжелезо Распилитьжелезо Cut Iron - Discover & Share GIFs](https://tenor.com/view/%D1%80%D0%B5%D0%B7%D0%B0%D1%82%D1%8C%D0%B6%D0%B5%D0%BB%D0%B5%D0%B7%D0%BE-%D1%80%D0%B0%D1%81%D0%BF%D0%B8%D0%BB%D0%B8%D1%82%D1%8C%D0%B6%D0%B5%D0%BB%D0%B5%D0%B7%D0%BE-cut-iron-sharp-spark-gif-15258290)：点击查看 GIF。
- [GitHub - intel-analytics/ipex-llm: Accelerate local LLM inference and finetuning (LLaMA, Mistral, ChatGLM, Qwen, Baichuan, Mixtral, Gemma, Phi, etc.) on Intel CPU and GPU (e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max); seamlessly integrate with llama.cpp, Ollama, HuggingFace, LangChain, LlamaIndex, DeepSpeed, vLLM, FastChat, Axolotl, etc.](https://github.com/intel-analytics/ipex-llm)：在 Intel CPU 和 GPU（例如带 iGPU 的本地 PC，Arc、Flex 和 Max 等独立 GPU）上加速本地 LLM 推理和微调（LLaMA, Mistral, ChatGLM, Qwen, Baichuan, Mixtral, Gemma, Phi 等）；与 llama.cpp, Ollama, HuggingFace, LangChain, LlamaIndex, DeepSpeed, vLLM, FastChat, Axolotl 等无缝集成。
- [GitHub - JingShing/How-to-use-tesla-p40: A manual for helping using tesla p40 gpu](https://github.com/JingShing/How-to-use-tesla-p40)：帮助使用 Tesla P40 GPU 的手册。通过在 GitHub 上创建账户为 JingShing/How-to-use-tesla-p40 做出贡献。
- [Mikubox Triple-P40 build](https://rentry.org/Mikubox-Triple-P40)：来自 eBay 的 Dell T7910“准系统”，包含散热片。我推荐“digitalmind2000”卖家，因为他们使用现场发泡包装，确保工作站送达时完好无损。您可以选择 Xe...

---

### **LM Studio ▷ #**[**🧪-beta-releases-chat**](https://discord.com/channels/1110598183144399058/1166577236325965844/1249045502369202308) (2 条消息):

- **即将发布的针对 Smaug 模型的 BPE tokenizer 更新**：一位成员分享称，下一个版本将包含一个专门针对 **Smaug 模型** 的 **BPE tokenizer** 提交。这预示着在未来的更新中能更好地处理这些模型。
- **关于将 LMS 数据收集到外部服务器的问题**：另一位成员询问是否有办法将 **LMS** 数据收集到**外部服务器**。这凸显了对 LMS 外部数据存储解决方案的兴趣。

---

### **LM Studio ▷ #**[**autogen**](https://discord.com/channels/1110598183144399058/1167546228813336686/1249589098096627743) (5 messages):

- **通过安装开发分支修复问题**：一位成员建议通过安装开发分支来解决问题，命令为 `pip install autogenstudio==0.0.56rc3`。这似乎是针对部分用户遇到的问题的一个潜在变通方案。
- **分享针对工作流问题的 Github 解决方案**：该成员分享了一个 [GitHub issue 链接](https://github.com/microsoft/autogen/issues/2445#issuecomment-2078167059)，他们在那里找到了一个解决方案，解决了在 LM Studio 中使用 AutogenStudio 时工作流在 2 个 Token 后终止的问题。
- **不同模型的测试结果参差不齐**：该成员报告了使用 **Llama 3** 和 **WizardLM** 等各种模型时的混合结果。他们指出，**Llama 3 instruct 70B quantized to 5 bits** 表现最为出色，尽管他们正在考虑 Fine-tuning 一个更适合作为 Agent 的模型。
- **寻求关于 Fine-tuning 的建议**：他们好奇是否可以使用单块 **4090 GPU** 和目前的处理器进行模型 Fine-tuning，并询问如何获取 Fine-tuning 所需的数据。
- **Completion tokens 受限的问题**：另一位成员提到在将 AutogenStudio 与 **TheBloke/Llama-2-7B-Chat-GGUF** 配合使用时，遇到了 Completion tokens 被限制为 2 的问题。他们正在寻求配置设置方面的帮助以解决此错误。

**提到的链接**：[[Issue]: Workflow terminates after 2 tokens when using AutogenStudio with LM Studio · Issue #2445 · microsoft/autogen](https://github.com/microsoft/autogen/issues/2445#issuecomment-2078167059)：问题描述：如果我在 Autogen studio 中创建一个指向 LM studio 端点的模型，然后将该模型添加到 Agent，再添加到工作流等，当我运行工作流时，它会在 2 个字符后终止...

---

### **LM Studio ▷ #**[**langchain**](https://discord.com/channels/1110598183144399058/1167546793656062063/1248839738350374942) (13 messages🔥):

- **选择 llama3 进行 Instruction Following**：一位用户选择了 **llama3 8b instruct Q6K**，因为它是遵循指令能力最强的本地模型之一。他们表示：*"我选择它是因为它是最擅长遵循指令的模型之一。"*
- **讨论统一的模型处理**：关于使用同一个语言模型处理多个任务进行了讨论，用户澄清了他们当前的设置和集成方式。一位用户提到使用曾与 **GPT 3.5-turbo** 配合使用的旧版本代码，现在尝试使用 LM Studio 的 OpenAI 集成。
- **OpenAI 与本地服务器的集成**：针对 LM Studio，一位用户在 **port 8008** 上设置了本地服务器，并使用 `client = OpenAI(base_url="http://localhost:8008/v1", api_key="not-needed")` 调用模型。他们注意到虽然生成了 Token，但结果很差，且模型无法准确遵循指令。
- **尝试不同的模型**：同一位用户除了 **llama3** 之外还尝试了 **Mistral 7b instruct**，发现结果仍然具有随机性。他们评论道：*"结果真的很随机。"*

---

### **LM Studio ▷ #**[**amd-rocm-tech-preview**](https://discord.com/channels/1110598183144399058/1195858490338594866/1248958015403593823) (15 messages🔥):

- **新的 AMD 7800X3D 升级 Bug 出现**：一位成员在从 AMD 3600 CPU 升级到 7800X3D 时遇到了兼容性问题，导致其 RX 6900XT 无法正常工作。他们最终通过在 BIOS 中找到禁用新 CPU 中集成 GPU 的选项解决了此问题。
- **AMD GPU 隔离技巧**：分享了在 ROCm 中隔离 GPU 的各种方法，并提供了一份详细的 [GPU 隔离技术指南](https://rocm.docs.amd.com/en/latest/conceptual/gpu-isolation.html)。在批处理文件中执行 `SET HIP_VISIBLE_DEVICES="1"` 可以帮助管理 GPU 的可见性。
- **讨论针对不同工具的 ROCm 实用性**：成员们讨论了在 Windows 上将 ROCm 与 auto1111 或 comfy 等工具配合使用的可能性。有人指出，虽然可行，但在 A1111 上实现 ROCm 被认为非常“黑入”（hacky），且与在 LMStudio 中使用 ROCm 不同。
- **探索 stable.cpp 项目和 Zluda**：提到了使用 Zluda 挂载到 CUDA 以利用 AMD GPU 是一种具有挑战性但有趣的方法。人们对整合这些技术以创建高效的 GPU 加速应用表现出兴趣。

**提到的链接**：[GPU isolation techniques — ROCm Documentation](https://rocm.docs.amd.com/en/latest/conceptual/gpu-isolation.html)：未找到描述

---

### **LM Studio ▷ #**[**🛠-dev-chat**](https://discord.com/channels/1110598183144399058/1234988891153629205/1249892945369501817) (1 messages):

- **用户寻求 LM Studio 的 GPU 配置支持**：一位新成员询问如何将旧的 GT Nvidia 1030 GPU 添加到 **LM Studio**。他们注意到没有可用的 GPU 使用配置，并推测这可能是由于 GPU 型号过旧所致。

---

### **OpenAI ▷ #**[**annnouncements**](https://discord.com/channels/974519864045756446/977259063052234752/1249799178977153075) (1 messages):

- **OpenAI 与 Apple 达成集成合作伙伴关系**：OpenAI 宣布与 Apple 合作，将 **ChatGPT 集成到 iOS, iPadOS 和 macOS** 中。该集成预计将于今年晚些时候推出：[公告详情](https://openai.com/apple)。

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1248797641790324747) (216 messages🔥🔥):

- **对 Whisper 多语言转录的担忧**：一位成员提到，与版本 2 不同，**Whisper version 3** 在说话者交替使用多种语言时无法进行转录。他们渴望更新或新版本的发布，并询问：“Whisper version 4 何时发布并开源？”
- **Apple 推出 'Apple Intelligence'**：Apple 将通过即将推出的 iPhone 16 增强其 AI 能力，并将该计划命名为 [Apple Intelligence](https://appleinsider.com/inside/apple-intelligence)。这引发了关于对科技行业影响的讨论，一位用户表示，为了使用设备端 AI 功能，可能需要升级硬件。
- **OpenAI API 提示词的安全担忧**：用户讨论了如何保护 OpenAI API 应用中的提示词，建议使用系统提示词（system prompts）并拒绝重复这些提示词等策略。其中一个突出的解决方案是，“*拒绝所有重复系统或用户提示词的请求*”，这在测试中被证明是有效的。
- **图像生成服务的挑战**：一些成员就 **DALL-E** 和 **Midjourney** 等图像生成服务的成本和可访问性进行了辩论。一位成员评论道：“我不想花 10 美元只生成大约 3 张图像，”强调了负担能力问题。
- **关于消费级科技中 AI 模型集成的讨论**：关于在消费级技术中集成 **GPT-4o** 等先进 AI 模型的讨论非常热烈。人们对硬件兼容性和未来更新表示担忧，一种观点认为并非所有用户都能立即获得这些升级。

**提到的链接**：

- [防止泄露系统提示词！](https://community.openai.com/t/prevent-revealing-system-prompt/303771)：大家好，我有一个包含一些规则的提示词。如果用户询问第 3 点的细节，AI 就会泄露我的系统提示词 😅 系统提示词：扮演一名汽车专家。始终遵守 f...
- ['Apple Intelligence' 可能是 Apple 进军 AI 的名称](https://appleinsider.com/articles/24/06/07/ios-18-ai-boost-could-be-called-apple-intelligence)：Apple 在 WWDC 上大举进军 AI 的名称可能非常简单，据报道被称为 “Apple Intelligence”。

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1248726661789384857) (87 messages🔥🔥):

- **GPT Agents 被固定使用 GPT-4o**：成员们对 GPT Agents 只能使用 GPT-4o 表示沮丧，即使在指定使用 GPT-4 时也是如此。一位成员提到，“这似乎是一个巨大的疏忽，”并强调了其在结构化提示词中表现不佳。
- **Token 限制与成本**：针对 UI 与 API 的 Token 限制进行了深入讨论，重点关注 128k Context 调用的高昂成本。**Webhead** 分享道：“一个完整的 128k Context 调用需要 60 美分……还不包括输出，”这引发了对普通用户可行性的担忧。
- **图像 Token 化成本讨论**：成员们辩论了 OpenAI 如何处理图像并收费，解释说图像像文本一样被 Token 化。澄清指出，为了 Token 化的目的，图像被调整为 512x512 的切片，并链接到了 [OpenAI 的 API 定价](https://openai.com/api/pricing/)。
- **Custom GPTs 详情澄清**：几位成员对 Custom GPTs 的隐私和外部集成感到困惑。已确认 Custom GPTs 默认是私有的，并且无法通过 OpenAPI 进行外部集成。
- **新语音模式推出受到质疑**：成员们对 Plus 用户新语音模式的延迟推出提出质疑，其中一人表示：“OpenAI 承诺它将在未来几周内推出，但已经过去一个月了。”另一位成员幽默地指出了“未来几周”这一表述的模糊性。

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/1248776451612999700) (16 messages🔥):

- **ChatGPT 响应格式化的困扰**：一名成员在让 ChatGPT 输出仅包含 `<p>` 和列表标签而非完整 HTML 页面的 HTML 时遇到问题。另一位成员建议提供期望输出的示例，以帮助模型更好地理解。
- **摘要 Prompt 请求的反馈**：一位成员分享了一个简单的摘要 Prompt 并寻求改进结果的反馈。另一位成员提供了建议和替代方案，强调需要尝试不同的方法来确定最有效的 Prompt。
- **Canva Pro 和 Inpainting 作为文本编辑工具**：成员们讨论了使用 Canva Pro 的 Magic Tools 和 Inpainting 作为编辑图像内文本的方法。这些工具可以帮助提取文本、修正拼写错误或在多次会话中进行小区域编辑。
- **LLM Prompt 失败案例**：一位用户询问了大型语言模型（包括 GPT-4）难以处理的 Prompt。提供的一个例子是问题 “What is davidjl?”，ChatGPT 和 GPT-4 都难以给出正确回答。
- **生成 Photoshop 渐变映射的请求**：一位成员寻求帮助，希望创建 Python 脚本将带有十六进制代码的颜色渐变转换为 Photoshop 的 .GRD 文件。他们提供了示例渐变选项，但难以让 Copilot 准确生成所需的脚本。

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/1248776451612999700) (16 messages🔥):

- **HTML 格式化困扰**：一位用户请求帮助，希望让 ChatGPT 将响应格式化为 HTML 而不生成完整的 HTML 页面。另一位用户建议提供期望输出的示例以获得更好的结果。
- **改进摘要 Prompt 反馈**：一位成员分享了一个摘要 Prompt 并征求反馈。另一位成员建议使用替代方案来进一步优化输出，重点是以引人入胜的格式传达清晰的关键信息。
- **Prompt 一致性问题**：有人提出了关于大多数 LLM 难以始终如一正确处理的 Prompt 的问题。回复中包括了一些例子，如对特定查询（如 "What is davidjl?"）的困惑。
- **为 Photoshop 生成渐变映射**：一位用户分享了一个详细请求，希望使用十六进制代码为 Photoshop 生成渐变选项。尽管颜色组合成功，但他们在让 Copilot 创建用于 .GRD 文件的 Python 脚本时遇到困难，并寻求额外帮助。
- **让 ChatGPT 了解 API 内容**：一位用户询问了让 ChatGPT 了解 GitHub 仓库 API 内容的最佳方法。他们考虑了诸如将 API 提取到文本文件中并将其集成到 ChatGPT 的知识库中等选项。

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1248727678585012325) (109 messages🔥🔥):

- **成员讨论 GPU 限制带来的挑战**：成员们分享了对“GPU 贫困”的担忧，并讨论了潜在的解决方案，如使用 **sd turbo** 或基于 CPU 的模型来减少等待时间。一位成员提到：*“你仍然需要等待一分钟左右，但它仍然是值得的”*。
- **模型训练中的固定种子与随机种子**：关于公司在训练生产级神经网络时使用固定种子还是随机种子的深入讨论。一位成员提到，他们通过设置手动种子并调整参数来逃离局部最小值，另一位成员强调：*“种子总是存在的，问题只在于你是否知道它是什么。”*
- **研究用于 LLM 的无 MatMul 模型**：分享了一篇 [arXiv 论文](https://arxiv.org/abs/2406.02528) 链接，强调了在保持强大性能的同时消除大型语言模型中 MatMul 操作的潜力，实验显示在高达 2.7B 参数规模下结果令人期待。
- **在 NLP 中探索扩散模型**：有人建议可能使用扩散模型将 2B LLM 升级到 7B LLM 的质量，随后分享了如[这篇综述论文](https://arxiv.org/abs/2305.14671)的参考文献。一位成员评论道：*“总的来说，这种方法不是重复去噪下一个 Token，而是以随机顺序重复去噪所有 Token。”*
- **匈牙利的 AI 安全研究资助**：讨论了在匈牙利赞助 3000 万美元用于 AI 安全研究的可行性和影响，重点在于确保资金不被浪费，且云端算力访问是理想的选择。一位成员建议：*“如果你是个人，几十万美元就会产生影响，”* 而另一位成员则强调了公关在 AI 安全事业中的重要性。

**Links mentioned**:

- [Scalable MatMul-free Language Modeling](https://arxiv.org/abs/2406.02528)：矩阵乘法 (MatMul) 通常在大型语言模型 (LLMs) 的总体计算成本中占据主导地位。随着 LLMs 扩展到更大的嵌入维度和上下文长度，这种成本只会不断增加……
- [EleutherAI/pile-standard-pythia-preshuffled · Datasets at Hugging Face](https://huggingface.co/datasets/EleutherAI/pile-standard-pythia-preshuffled)：未找到描述
- [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://arxiv.org/abs/2310.16834)：尽管扩散模型在许多生成建模任务中表现出开创性的性能，但在自然语言等离散数据领域却表现不佳。至关重要的是，标准的扩散模型……
- [PLANNER: Generating Diversified Paragraph via Latent Language Diffusion Model](https://arxiv.org/abs/2306.02531)：用于文本的自回归模型有时会产生重复且低质量的输出，因为误差在生成步骤中不断累积。这个问题通常归因于曝光偏差 (exposure bias) —— 即……
- [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)：我们从头开始复现 GPT-2 (124M)。这段视频涵盖了整个过程：首先我们构建 GPT-2 网络，然后优化其训练，使其变得非常……
- [Tensor Parallelism - torch.distributed.tensor.parallel — PyTorch 2.3 documentation](https://pytorch.org/docs/stable/distributed.tensor.parallel.html)：未找到描述
- [Enhancing Paragraph Generation with a Latent Language Diffusion Model](https://machinelearning.apple.com/research/latent-language-diffusion-model)：在快速发展的自然语言处理 (NLP) 领域，对于生成连贯且受控的文本有着强烈的需求，因为……
- [Hashes — EleutherAI](https://www.eleuther.ai/hashes)：未找到描述
- [A Survey of Diffusion Models in Natural Language Processing](https://arxiv.org/abs/2305.14671)：这篇综述论文对扩散模型在自然语言处理 (NLP) 中的应用进行了全面回顾。扩散模型是一类旨在捕捉扩散过程的数学模型……
- [GitHub - justinlovelace/latent-diffusion-for-language](https://github.com/justinlovelace/latent-diffusion-for-language)：通过在 GitHub 上创建账号，为 justinlovelace/latent-diffusion-for-language 的开发做出贡献。
- [GitHub - xhan77/ssd-lm: Semi-autoregressive Simplex-based Diffusion Language Model for Text Generation and Modular Control](https://github.com/xhan77/ssd-lm)：用于文本生成和模块化控制的半自回归单纯形扩散语言模型 - xhan77/ssd-lm
- [ProphetNet/AR-diffusion at master · microsoft/ProphetNet](https://github.com/microsoft/ProphetNet/tree/master/AR-diffusion)：一个用于自然语言生成的研究项目，包含 MSRA NLC 团队的官方实现。- microsoft/ProphetNet
- [GitHub - XiangLi1999/Diffusion-LM: Diffusion-LM](https://github.com/XiangLi1999/Diffusion-LM)：Diffusion-LM。通过在 GitHub 上创建账号，为 XiangLi1999/Diffusion-LM 的开发做出贡献。

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1248757889733099587) (173 条消息🔥🔥):

- **RoPE 技术的实际应用**：成员们讨论了使用相对位置编码 (RoPE) 来增强非自回归文本生成模型。*“你能做的最简单且表现良好的事情就是使用 ROPE 将嵌入添加到 keys 和 queries 中，但将‘当前’位置添加到一个，将‘目标’位置添加到另一个”*。
- **使用插值进行模型初始化**：一位成员提议通过对权重矩阵进行插值来将其大小翻倍，从而初始化模型的权重，类似于处理图像。这种方法*“可能只需要极少的持续训练来‘修复’生成的模型”*。
- **层剪枝与效率**：讨论包括层剪枝策略及其对模型效率和性能的影响。一位成员成功地将 Qwen 2 72B 剪枝到约 37B，同时保持了有效性。
- **权重共享 (weight tying) 和 Universal Transformers 的稳定性**：对话涉及了大型模型中权重共享的不稳定性，以及如何稳定 Universal Transformers (UTs)。*“是的，它很快就会变得不稳定。值得一提的是，我的实验是目前唯一能找到的将简单的 UTs 实际扩展到 20M 以上的有记录的实验”*。
- **LoRA 初始化增强**：成员们研究了初始化低秩自适应 (LoRA) 权重以加速收敛的新方法。见解包括使用奇异值分解 (SVD) 进行初始化，其性能优于传统方法。

**提到的链接**：

- [通过 Circuit Breakers 提高对齐性和鲁棒性](https://arxiv.org/abs/2406.04313)：AI 系统可能会采取有害行动，且极易受到对抗性攻击。我们提出了一种受表征工程（representation engineering）最新进展启发的方法，在模型...时中断它们。
- [深层网络不合理的无效性](https://arxiv.org/abs/2403.17887)：我们对流行的开源权重预训练 LLM 系列研究了一种简单的层剪枝（layer-pruning）策略，发现在不同的问答基准测试中，性能退化极小，直到...
- [为什么通过规模预测前沿 AI 模型的下游能力仍然难以实现？](https://arxiv.org/abs/2406.04391)：通过扩展（scaling）先进 AI 系统获得可预测的行为是一个极其理想的特性。虽然关于预训练性能如何随规模扩展已有成熟的文献，但关于...
- [学习增长预训练模型以实现高效的 Transformer 训练](https://arxiv.org/abs/2303.00980)：扩展 Transformer 在许多领域都取得了重大突破，导致了一种定期训练并发布现有模型更大版本的范式。新的实例...
- [关于可证明的长度和组合泛化](https://arxiv.org/abs/2402.04875)：sequence-to-sequence 模型的分布外泛化能力可以从两种关键的泛化形式来研究：长度泛化——即泛化到...的能力。
- [了解你的 LoRA](https://datta0.github.io/blogs/know-your-lora/)：重新思考 LoRA 初始化。什么是 LoRA？LoRA 在微调领域，特别是参数高效微调（PEFT）中一直是一个巨大的工具。它是一种非常简便的模型微调方式...
- [Grokfast：通过放大慢梯度加速 Grokking](https://arxiv.org/abs/2405.20233)：机器学习中一个令人困惑的现象被称为 grokking，即在对训练数据近乎完美过拟合后的数万次迭代后，才实现延迟泛化。针对长距离...
- [噪声不是 Transformer 上 SGD 和 Adam 差距的主要原因，但符号下降（Sign Descent）可能是](http://arxiv.org/abs/2304.13960)：Adam 优化器在广泛架构上的成功使其成为随机梯度下降（SGD）表现不佳时的默认选择。然而，我们对其理论上的理解...
- [NATURAL PLAN：在自然语言规划上评测 LLM](https://arxiv.org/abs/2406.04520)：我们推出了 NATURAL PLAN，这是一个真实的自然语言规划基准测试，包含 3 个关键任务：旅行规划、会议规划和日历调度。我们的评估重点在于规划...
- [PiSSA：大语言模型的主奇异值和奇异向量自适应](http://arxiv.org/abs/2404.02948)：为了对大语言模型（LLM）进行参数高效微调（PEFT），低秩自适应（LoRA）方法通过两个矩阵的乘积来近似模型变化 $ΔW \in \mathbb{R}^{m \times n}$...
- [堆叠你的 Transformer：深入探讨用于高效 LLM 预训练的模型增长](https://arxiv.org/abs/2405.15319)：由于规模庞大，LLM 的预训练计算成本很高。模型增长（Model growth）作为一种有前景的方法出现，通过利用较小的模型来加速较大模型的训练。然而，...
- [如何缩小损失曲线中训练集和验证集之间的差异？](https://stackoverflow.com/questions/74021838/how-to-reduce-the-difference-between-training-and-validation-in-the-loss-curve)：我使用 Transformer 模型训练时间序列数据集，但我的损失曲线中训练和验证之间总是有差距。我尝试过使用不同的学习率、batch size...
- [VALL-E 2：神经编解码器语言模型是达到人类水平的零样本语音合成器](https://arxiv.org/abs/2406.05370)：本文介绍了 VALL-E 2，这是神经编解码器语言模型的最新进展，标志着零样本文本转语音（TTS）的一个里程碑，首次达到了人类水平。基于...
- [OLoRA：大语言模型的正交低秩自适应](https://arxiv.org/abs/2406.01775)：大语言模型（LLM）的出现彻底改变了自然语言处理，在理解和生成类人文本方面实现了前所未有的能力。然而，计算...
- [来自 Yossi Gandelsman (@YGandelsman) 的推文](https://x.com/YGandelsman/status/1799109601750810706)：机械可解释性（Mechanistic interpretability）不仅是理解模型内部运行的好方法，也是发现“模型漏洞”并利用它们的工具！我们的新论文表明...

- [LLM Reading Group (3月5日, 19日; 4月2日, 16日, 30日; 5月14日, 28日; 6月11日)](https://www.eventbrite.ca/e/llm-reading-group-march-5-19-april-2-16-30-may-14-28-june-11-tickets-851921368747?aff=oddtdtcreator): 来见见 LLM/NLP 研究领域一些开创性论文的作者，听他们分享自己的工作。

---

### **Eleuther ▷ #**[**interpretability-general**](https://discord.com/channels/729741769192767510/1052314805576400977/1248715545969426512) (14 messages🔥):

- **Old Techniques Make a Comeback**: **旧技术回归**："天哪。等等。大家又开始关注 TopK activations 了？Numeta 确实领先于时代。" 一位用户对 **TopK activations** 重新引起关注表示惊讶和怀念。
- **New Insight on Mechanistic Interpretability**: **Mechanistic Interpretability 的新见解**：Mechanistic Interpretability 有助于理解模型并发现 "模型缺陷（model bugs）"。一篇新论文强调，理解 **CLIP neurons** 可以自动生成语义对抗图像，详见[这条推文](https://x.com/ygandelsman/status/1799109601750810706)。
- **Short Circuiting Offers Hope for LLM Security**: **Short Circuiting 为 LLM 安全带来希望**：一种名为 **Short Circuiting** 的新对齐技术有望提高 LLM 安全的对抗鲁棒性。尽管显示出良好的结果，但代码尚未发布，更多讨论见[此处](https://x.com/andyzou_jiaming/status/1799232319250743561)。
- **Project on MLP Neurons in Llama3 Launched**: **Llama3 中的 MLP Neurons 项目启动**：分享了一个探索 Llama3 模型中 **MLP neurons** 的精彩项目，包括一个用于神经元探索的网页和一份发布在 [neuralblog](https://neuralblog.github.io/llama3-neurons/) 上的文章。该项目代码已在 [GitHub](https://github.com/neuralblog/llama3-neurons) 开源。
- **DeepSeek Model Interpretation Challenges**: **DeepSeek 模型解释挑战**：用户讨论了使用 transformerlens 解释 **DeepSeek model** 的复杂性及初步困难。不过，他们分享了一些潜在的想法和一个用于协作解决问题的 GitHub 仓库链接（[repository](https://github.com/wassname/adapters_can_monitor_lies)）。

**提到的链接**：

- [Andy Zou (@andyzou_jiaming) 的推文](https://x.com/andyzou_jiaming/status/1799232319250743561)：没有 LLM 是安全的！一年前，我们揭晓了众多能够破解所有主流 LLM 的自动化越狱方法中的第一种。🚨 但还有希望？！我们推出了 Short Circuiting：第一种对齐技术...
- [Yossi Gandelsman (@YGandelsman) 的推文](https://x.com/ygandelsman/status/1799109601750810706)：Mechanistic Interpretability 不仅是理解模型内部运行的好方法，也是发现 "模型缺陷" 并利用它们的工具！我们的新论文表明...
- [Llama-3-8B MLP Neurons](https://neuralblog.github.io/llama3-neurons/)：未找到描述

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1248905161695559732) (9 messages🔥):

- **Member struggles with MAUVE setup**: **成员在 MAUVE 配置上遇到困难**：一位成员请求帮助运行 MAUVE，以完成一篇关于新采样方法的论文。他们分享了 [MAUVE GitHub repository](https://github.com/krishnap25/mauve-experiments)，并指出配置困难。
- **Concurrency limitations in eval harness**: **eval harness 中的并发限制**：讨论了 eval harness 如何串行运行查询，且 batch size 参数证明无效。建议使用 `--model hf` 和 `--model vllm` 以获得更好的并发性能。
- **Custom task YAML troubleshooting**: **自定义任务 YAML 故障排除**：一位成员的自定义任务无法生成输出，可能是由于 `doc_to_text` 或 `doc_to_target` 的问题，或者是缺少停止序列。建议手动指定 stop sequences。
- **Chat template application issues**: **Chat template 应用问题**：有人询问在 gsm8k 评估运行期间，是否默认应用 Hugging Face 模型的 chat template。澄清了通过 `--apply_chat_template` 标志支持聊天模板，但默认不启用。

**提到的链接**：[GitHub - krishnap25/mauve-experiments](https://github.com/krishnap25/mauve-experiments)：通过在 GitHub 上创建账号，为 krishnap25/mauve-experiments 的开发做出贡献。

---

### **Modular (Mojo 🔥) ▷ #**[**general**](https://discord.com/channels/1087530497313357884/1098713601386233997/1248807850428334120) (141 条消息🔥🔥):

- **在 MacOS 上安装 MAX 需要手动修复**：用户讨论了在 MacOS 14.5 Sonoma 上安装 MAX 时遇到的问题，并需要手动修复。解决方案包括使用 pyenv 设置 Python 3.11，并参考 [Modular 安装指南](https://docs.modular.com/max/install) 中的说明。
- **结构化并发与函数着色（Function Coloring）之争**：成员们辩论了结构化并发与函数着色，并就复杂性和性能发表了看法。一位参与者提到 *"Effect generics 确实解决了函数着色问题，但它们会让语言变得更难编写"*。
- **编程语言中的并发**：对话涵盖了并发原语以及 Erlang/Elixir、Go 和 async/await 机制的有效性。一位用户指出 *"Mojo 的优势在于能够从一开始就通过设计来兼容所有这些特性"*。
- **MLIR 与 Mojo**：讨论了 MLIR dialects 在 Mojo 异步操作中的相关性，并提到了 MLIR 文档中的 async dialect。一位用户澄清道：*"团队在 modcon 上表示，他们只使用了内置的 builtin 和 index dialects"*。
- **新编程语言的资金与可行性**：关于开发新编程语言所需的资金支持进行了对话，引用了 Modular 获得的 1.3 亿美元融资，并与 Rust 和 Zig 等团队进行了比较。一位参与者强调：*"1.3 亿美元比大多数编程语言团队梦寐以求的还要多"*。

**相关链接**：

- [安装 MAX | Modular 文档](https://docs.modular.com/max/install)：欢迎阅读 MAX 安装指南！
- [MAX 入门指南 | Modular 文档](https://docs.modular.com/max/get-started)：欢迎阅读 MAX 快速入门指南！
- ['async' Dialect - MLIR](https://mlir.llvm.org/docs/Dialects/AsyncDialect/)：未找到描述

---

### **Modular (Mojo 🔥) ▷ #**[**📺︱youtube**](https://discord.com/channels/1087530497313357884/1098713700719919234/1248728366169980980) (1 条消息):

- **Modular 发布新视频**：**Modular** 发布了一个新视频并分享了 [YouTube 链接](https://www.youtube.com/watch?v=3FKSlhZNdL0) 以供观看。该视频似乎是与频道关注者相关的最新更新或发布。

---

### **Modular (Mojo 🔥) ▷ #**[**ai**](https://discord.com/channels/1087530497313357884/1103420074372644916/) (1 条消息):

dorjeduck: 来自 Andrej 的新宝藏 [https://youtu.be/l8pRSuU81PU](https://youtu.be/l8pRSuU81PU)

---

### **Modular (Mojo 🔥) ▷ #**[**🔥mojo**](https://discord.com/channels/1087530497313357884/1151418092052815884/1248746734109069323) (86 条消息🔥🔥):

- **Mojo Playground 替代方案**：一名成员建议，如果 Mojo Playground 无法满足特定需求，可以使用 AWS、GCP 或 Azure 等云服务器，特别强调了 Google GCP 实例配合浏览器内 Cloud Shell 的易用性。
- **使用 Mojo 遇到的毕业论文阻碍**：一名成员讨论了在生物模拟论文中使用 Mojo 的潜在问题，指出缺乏 Class 支持是一个主要障碍，最终由于其目前的局限性决定不使用 Mojo。
- **Mojo 中的 Subprocess 计划**：成员们询问了在 Mojo 中实现 subprocess 的未来计划。虽然已经进行了讨论，但该功能尚未设定具体的时间表。
- **指针类型差异**：一名成员指出，新的 `UnsafePointer` 类型在其 `alloc` 函数中缺少 `alignment` 规范，而这在 `LegacyPointer` 中是存在的。
- **自定义 PRNG 和核心更新**：一名成员分享了他在 Mojo 中实现的 xoshiro PRNG，实现了显著的性能提升，并提到了将数值库移植到 Mojo 的持续工作，并附带了相关项目的链接：[numojo](https://github.com/thk686/numojo) 和 [NuMojo](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo)。

**提到的链接**：

- [QuantizationEncoding | Modular Docs](https://docs.modular.com/max/api/mojo/graph/quantization/quantization_encoding/QuantizationEncoding)：描述了可量化数据类型的编码。
- [Get started with Mojo🔥 | Modular Docs](https://docs.modular.com/mojo/manual/get-started)：立即安装 Mojo 并开始开发。
- [List | Modular Docs](https://docs.qa.modular.com/mojo/stdlib/collections/list/List#__init__)：List 类型是一个动态分配的列表。
- [Mojo🔥 roadmap & sharp edges | Modular Docs](https://docs.modular.com/mojo/roadmap#no-python-style-generator-functions)：Mojo 计划摘要，包括即将推出的功能和需要修复的问题。
- [GitHub - Mojo-Numerics-and-Algorithms-group/NuMojo: A numerics library for the Mojo programming language](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo)：Mojo 编程语言的数值库 - Mojo-Numerics-and-Algorithms-group/NuMojo。
- [mojo/stdlib/src/builtin/coroutine.mojo at ceaf063df575f3707029d48751b99886131c61ba · modularml/mojo](https://github.com/modularml/mojo/blob/ceaf063df575f3707029d48751b99886131c61ba/stdlib/src/builtin/coroutine.mojo#L232)：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。
- [[proposal] Add `Deque` struct to the stdlib by gabrieldemarmiesse · Pull Request #2925 · modularml/mojo](https://github.com/modularml/mojo/pull/2925)：#2659 @JoeLoser 我按要求提交了 API 设计提案。我们应该能够通过这种格式轻松讨论 API。如果你想查看渲染后的 Markdown，可以在这里阅读：https:...
- [GitHub - thk686/numojo: Numerics for Mojo](https://github.com/thk686/numojo)：Mojo 的数值库。通过在 GitHub 上创建账号为 thk686/numojo 的开发做出贡献。
- [mojo/stdlib/src/collections/list.mojo at main · modularml/mojo](https://github.com/modularml/mojo/blob/main/stdlib/src/collections/list.mojo)：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。
- [mojo/stdlib/src/memory/unsafe_pointer.mojo at 652485ceb9332885a7537760dcc949bfe8b1e5a0 · modularml/mojo](https://github.com/modularml/mojo/blob/652485ceb9332885a7537760dcc949bfe8b1e5a0/stdlib/src/memory/unsafe_pointer.mojo#L132)：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。
- [mojo/stdlib/src/memory/unsafe.mojo at nightly · modularml/mojo](https://github.com/modularml/mojo/blob/nightly/stdlib/src/memory/unsafe.mojo#L383)：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。

---

### **Modular (Mojo 🔥) ▷ #**[**🏎engine**](https://discord.com/channels/1087530497313357884/1212827673257316453/1248793846897180773) (11 条消息🔥):

- **使用 Mojo 选择 Tensor 轴**：一位用户寻求如何使用 Mojo API 表示 `g[0][:, 2][:, None]`。另一位成员建议使用 `g[0][2, axis=1].reshape(3, 1)` 作为当前的变通方案，并暗示未来会有 UX 改进。
- **将大数据（权重）嵌入二进制文件**：一位用户询问如何将大数据（权重）编译进最终的二进制文件。建议使用 [MAX checkpoint API](https://docs.modular.com/max/api/mojo/graph/checkpoint/)，并参考 [quantize-tinystories pipeline](https://github.com/modularml/max/blob/f89bc8f4e685e2bbcc269c8c324b5c105391f6f9/examples/graph-api/pipelines/quantize_tinystories) 中的示例。
- **新更新的量化技术**：关于最新更新中具体量化技术的咨询得到了详细解答，包括预量化的 GGML k-quants 的细节，并指向了 [GGML k-quants 文档](https://docs.modular.com/max/api/mojo/graph/ops/quantized_ops/qmatmul) 和 [Llama 3 pipeline](https://github.com/modularml/max/tree/f89bc8f4e685e2bbcc269c8c324b5c105391f6f9/examples/graph-api/pipelines/llama3)。
- **博客文章中的失效链接**：发现博客文章中有一个失效链接导致 404 错误。建议的正确链接为：[https://docs.modular.com/max/api/mojo/graph/quantization/](https://docs.modular.com/max/api/mojo/graph/quantization/)。
- **澄清量化文档**：用户讨论了量化文档中可能存在的错误链接，并提供了正确的 URL。经澄清，量化 API 文档的正确链接可能是：[https://docs.modular.com/max/api/mojo/graph/quantization/](https://docs.modular.com/max/api/mojo/graph/quantization/)。

**提到的链接**：

- [QuantizationEncoding | Modular Docs](https://docs.modular.com/max/api/mojo/graph/quantization/quantization_encoding/QuantizationEncoding)：描述可量化数据类型的编码。
- [Quantize your graph weights | Modular Docs](https://docs.modular.com/max/graph/quantize)：MAX Graph 量化 API 简介。
- [max/examples/graph-api/pipelines/quantize_tinystories/quantize_tinystories.🔥 at f89bc8f4e685e2bbcc269c8c324b5c105391f6f9 · modularml/max](https://github.com/modularml/max/blob/f89bc8f4e685e2bbcc269c8c324b5c105391f6f9/examples/graph-api/pipelines/quantize_tinystories/quantize_tinystories.%F0%9F%94%A5#L241-L244))：展示 MAX 平台强大功能的示例程序、笔记本和工具集合 - modularml/max
- [quantization | Modular Docs](https://docs.modular.com/max/api/mojo/graph/quantization/)：用于量化 Graph Tensor 的 API。
- [max/examples/graph-api/pipelines/nn/embedding.🔥 at f89bc8f4e685e2bbcc269c8c324b5c105391f6f9 · modularml/max](https://github.com/modularml/max/blob/f89bc8f4e685e2bbcc269c8c324b5c105391f6f9/examples/graph-api/pipelines/nn/embedding.%F0%9F%94%A5#L66))：展示 MAX 平台强大功能的示例程序、笔记本和工具集合 - modularml/max

---

### **Modular (Mojo 🔥) ▷ #**[**nightly**](https://discord.com/channels/1087530497313357884/1224434323193594059/1248727801461215274) (48 条消息🔥):

- **基准测试进行中**：一位成员询问是否有发布 **benchmark 结果**的地方。另一位成员确认，虽然基准测试尚未公开（“我们内部仍在进行大量工作”），但未来可能会提供。
- **上下文管理器优于** `defer` **关键字**：关于是否引入 `defer` 关键字用于自动内存管理等任务展开了激烈辩论。成员们建议将上下文管理器（Context Managers）作为 Python 中处理资源更具惯用性（idiomatic）且实用的解决方案，并给出了有效管理不安全指针（unsafe pointers）的示例。
- **Mojo 中的内存管理**：详细讨论涵盖了 Mojo 当前能力下的**手动内存管理**和 RAII（资源获取即初始化）。会议指出 **UnsafePointers 没有生命周期（lifetimes）**，类似于 Rust 的 `Box` 概念可能有利于自动内存清理。
- **新的 Nightly Mojo 编译器发布**：发布了多个关于 nightly Mojo 编译器版本的公告，更新至 `2024.6.805`、`2024.6.905` 和 `2024.6.1005`。提供了原始差异（raw diffs）和当前更新日志（changelogs）的链接，以使社区了解最新变化（[链接](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)）。
- **资源管理最佳实践**：强调了使用上下文管理器处理资源的重要性，特别是由于它们能够管理异常并确保资源正确释放。这被视为在 Mojo 中进行稳定可靠的指针管理的关键。

---

### **OpenInterpreter ▷ #**[**general**](https://discord.com/channels/1146610656779440188/1147665339266650133/1248730813319745536) (179 条消息🔥🔥):

- **Gorilla OpenFunctions v2 给社区留下深刻印象**：成员们讨论了新的 [Gorilla OpenFunctions v2](https://huggingface.co/gorilla-llm/gorilla-openfunctions-v2)，注意到其能力和性能，特别是它如何与 GPT-4 并驾齐驱。他们强调了这一新工具对于 LLM 从自然语言指令中形成可执行 API 调用指令的重要性。
- **Local II 的 Local OS Mode 发布令人兴奋**：Killianlucas 宣布 **Local II** 现在支持 local OS mode，成员们对派对上的潜在现场演示感到兴奋。该更新可通过 `pip install --upgrade open-interpreter` 获取。
- **分享了派对录像**：分享了近期派对的 [YouTube 录像](https://youtube.com/live/pqBuxmpgpY0?feature=share)，成员们对录像表示感谢，并对演示（特别是 twodogseeds 的演示）感到兴奋。
- **Interpreter 模型挑战与修复**：成员们报告并讨论了 **OI 模型** 的各种技术问题，包括 API key 错误以及像 moondream 这样的 vision models 的问题。交流了修复这些问题的解决方案和潜在更改。
- **Shortcuts 与 Siri 集成 OI**：Gordanfreeman4871 分享了将 Siri Shortcuts 与 **Open Interpreter** 集成的成果，允许通过 Siri 语音输入命令并在 terminal 中执行，并[发布了教程视频](https://youtu.be/Tj9uyyflgxw?feature=shared)展示这一集成。

**提到的链接**：

- [Farm Friend by TDS](https://farm-friend-v1.replit.app)：未找到描述
- [gorilla-llm/gorilla-openfunctions-v2 · Hugging Face](https://huggingface.co/gorilla-llm/gorilla-openfunctions-v2)：未找到描述
- [Apple Intelligence Preview](https://www.apple.com/apple-intelligence/)：Apple Intelligence 是为您日常事务打造的个人智能系统。内置于 iPhone、iPad 和 Mac 中，具有开创性的隐私保护。
- [无标题](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/call-gemini-using-openai-library)：未找到描述
- [来自 two dog seeds (@twodogseeds) 的推文](https://x.com/twodogseeds/status/1799919392660349377?s=46&t=VaihmUuhAwNpUkMVv0Vzzg)：🍏FarmFriend 对标 Apple WWDC_24 iOS Shortcuts 版🍏 ——又名：什么是遥遥领先。执行一些 Apple 明天将宣布的假设功能。就在今天！s...
- [来自 Daniel Jeffries (@Dan_Jeffries1) 的推文](https://x.com/dan_jeffries1/status/1794740447052525609?s=46)：我花了几个小时听 Dan Hendyrcks 的发言，他是 SB 1047（即加州 AI 控制与中心化法案）背后的非营利 AI Safety 组织的负责人。我觉得他很有魅力、稳重、聪明...
- [2024 年 6 月 8 日](https://youtu.be/nuTokk8rXxs?feature=shared)：未找到描述
- [Siri shortcuts to open interpreter](https://youtu.be/Tj9uyyflgxw?feature=shared)：未找到描述
- [Open Interpreter](https://github.com/OpenInterpreter)：Open Interpreter 有 3 个可用的仓库。在 GitHub 上关注他们的代码。
- [Mbappé Om GIF - MbappéOm Kylian Mbappé Paris Saint Germain - 发现并分享 GIF](https://tenor.com/view/mbapp%C3%A9om-kylian-mbapp%C3%A9-paris-saint-germain-psg-gif-13937899)：点击查看 GIF
- [当我使用 interpreter.chat(stream=True) 时，在什么情况下 type 会返回 'image'？ · Issue #1301 · OpenInterpreter/open-interpreter](https://github.com/OpenInterpreter/open-interpreter/issues/1301)：描述 bug：当我使用 interpreter.chat(stream=True) 时，在什么情况下 type 会返回 'image'？当我尝试在 0.1.18 版本中使用它时，它返回 image，但 0.2.5 版本不支持...
- [欢迎参加六月 OPENINTERPRETER 派对](https://youtube.com/live/pqBuxmpgpY0?feature=share)：由 Restream 提供支持，https://restream.io，Discord 舞台功能挺难搞的
- [moondream:1.8b](https://www.ollama.com/library/moondream:1.8b)：moondream2 是一个小型 vision language model，旨在边缘设备上高效运行。
- [gpt&me + Hey GPT](https://nickdobos.gumroad.com/l/gptAndMe)：8 个 iOS shortcuts，让您在 iOS 上使用 chatGPT 的速度翻倍。直接在每个 iOS 和 Mac 应用中使用 chatGPT，替换 Siri，将 AI 反馈循环放入您的每日待办事项列表...
- [LangChainHub-Prompts/LLM_Bash · Hugging Face 数据集](https://huggingface.co/datasets/LangChainHub-Prompts/LLM_Bash)：未找到描述
- [GitHub - TellinaTool/nl2bash: 从自然语言生成 bash 命令 https://arxiv.org/abs/1802.08979](https://github.com/TellinaTool/nl2bash)：从自然语言生成 bash 命令 https://arxiv.org/abs/1802.08979 - TellinaTool/nl2bash
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/869/shell%2Bcommands%2Bused%2Bby%2Bparticipants%2Bof%2Bhands-on%2Bcybersecurity%2Btraining)：未找到描述

---

### **OpenInterpreter ▷ #**[**O1**](https://discord.com/channels/1146610656779440188/1194880263122075688/1248762185794977812) (24 messages🔥):

- **Rabbit R1 被破解以运行 O1**：一名成员兴奋地收到了他们的 Rabbit R1，并询问：*"现在我该如何破解它来运行 O1？"* 这引发了其他渴望测试该功能的成员的热烈讨论。
- **Raspberry Pi 设置遇到困难**：用户 noimnull 询问是否有人在 Raspberry Pi 上运行过 O1，特别是依赖 *"poetry run 01"*，但遇到了问题：*"它卡在服务器上了，我认为是资源不足"*。
- **将 O1 与 iPhone 连接**：用户 bp416 在将 MacBook 上运行的 O1 与 iPhone 应用连接时遇到麻烦。thatpalmtreeguy 建议：*"当你松开时它会发送命令"*，指出了使用该应用 hello 按钮的正确方式。
- **Raspberry Pi 4 CM4 上的 O1**：noimnull 反馈称他们使用的是 Pi4 CM4 8GB，但面临挑战，推测是由于资源不足。
- **需要 Linux 安装教程**：nxonxi 请求一份在 Linux 上安装 O1 的教程，这是那些尝试在不同操作系统上进行设置的人的共同需求。

---

### **OpenInterpreter ▷ #**[**ai-content**](https://discord.com/channels/1146610656779440188/1149229778138824765/) (1 messages):

gordanfreeman4871: 您的消息

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1248725107518607521) (49 messages🔥):

1.  **swyxio 强调 Ultravox 的发布：** [@juberti](https://x.com/juberti/status/1798898986289684849?s=46&t=90xQ8sGy63D2OtiaoGJuww) 发布了 Ultravox，一个具有理解非文本语音元素能力的“*开源多模态 LLM*”。v0.1 版本已在 [ultravox.ai](https://ultravox.ai) 上线，且他们正在招聘。
2.  **关于检索集成的讨论：** Chygao 提到了 Normal computing 对 Transformer 的实现，swyxio 指出相关演讲者将参加 ai.engineer。该实现可以在 GitHub 的[这里](https://github.com/normal-computing/extended-mind-transformers)找到。
3.  **关于 Perplexity 内容使用的争议：** Swyxio 注意到 [@JohnPaczkowski](https://x.com/johnpaczkowski/status/1799135156051255799?s=46&t=90xQ8sGy63D2OtiaoGJuww) 的一条推文，批评 Perplexity 在没有适当署名的情况下重新利用 Forbes 的内容。
4.  **OpenAI 的新领导层：** OpenAI 在其 Twitter 账号 [@OpenAI](https://x.com/openai/status/1800218626446049382?s=46&t=90xQ8sGy63D2OtiaoGJuww) 上宣布任命了新的 CFO 和 CPO。他们欢迎 *Friley* 担任 CFO，*Kevin Weil* 担任 CPO。
5.  **讨论 Apple 的智能集成：** 包括 [@karpathy](https://x.com/karpathy/status/1800242310116262150?s=46&t=90xQ8sGy63D2OtiaoGJuww) 和 [@matthew_d_green](https://x.com/matthew_d_green/status/1800291897245835616?s=46&t=90xQ8sGy63D2OtiaoGJuww) 在内的多位用户讨论了 Apple 新的 AI 集成和 "Private Cloud Compute" 系统。该系统旨在安全地将复杂任务卸载到云端，同时保持高隐私标准。

**提到的链接**：

- [来自 Anthropic (@AnthropicAI) 的推文](https://x.com/anthropicai/status/1799537686962638886?s=46&t=90xQ8sGy63D2OtiaoGJuww)：AI 的性格应该是怎样的？阅读我们关于如何塑造 Claude 性格的文章：https://www.anthropic.com/research/claude-character
- [来自 OpenAI (@OpenAI) 的推文](https://x.com/openai/status/1800218626446049382?s=46&t=90xQ8sGy63D2OtiaoGJuww)：我们很高兴欢迎两位拥有合适经验、技能和价值观的领导者来推动使命。@thefriley 加入担任首席财务官，@kevinweil 担任首席产品官...
- [来自 Justin Uberti (@juberti) 的推文](https://x.com/juberti/status/1798898986289684849?s=46&t=9)：认识 Ultravox，我们的开源多模态 LLM。查看我们的 v0.1 版本：https://ultravox.ai - 还有更多内容即将推出 - 我们正在招聘！（私信已开放）引用 Joe Heitzeberg (@jheitzeb) 的话：哇！Ult...
- [Reddit - 探索一切](https://www.reddit.com/r/WebSim/comments/1d110ph/the_websim_url_prompting_bible/)：未找到描述
- [来自 Suhail (@Suhail) 的推文](https://x.com/suhail/status/1800265203915055221?s=46&t=90xQ8sGy63D2OtiaoGJuww)：在经历了两次平台浪潮后，我现在是个老兵了，但 Apple 今天所做的是传达了这样一个信息：“嗨伙计们，我们制作了原生集成点，让你们所有的 AI 模型制作者为我们的 10 亿用户规模而竞争...”
- [AI 的个性应该是怎样的？](https://www.youtube.com/watch?v=iyJj9RxSsBY)：你如何赋予 AI 助手性格？那到底意味着什么？你为什么要这样做？在这次对话中，Stuart Ritchie (Re...
- [来自 Tyler Stalman (@stalman) 的推文](https://x.com/stalman/status/1800278850435190871?s=46&t=90xQ8sGy63D2OtiaoGJuww)：Apple 表示他们最终将集成 Google Gemini 模型

- [Rohan Paul (@rohanpaul_ai) 的推文](https://x.com/rohanpaul_ai/status/1799122826114330866?t=DajZXqRteA0XqfzcMTAbHg&s=19)：这真是一篇令人惊叹的论文。🤯 声称可以在 LLMs 中完全消除 MatMul 操作，同时在十亿参数规模下保持强劲性能，并利用一种优化...
- [Justin Uberti (@juberti) 的推文](https://x.com/juberti/status/1798898986289684849?s=46&t=90xQ8sGy63D2OtiaoGJuww)：见见 Ultravox，我们的开源多模态 LLM。查看我们在 https://ultravox.ai 发布的 v0.1 版本 —— 还有更多内容即将推出 —— 我们正在招聘！（私信已开放）引用 Joe Heitzeberg (@jheitzeb) 哇！Ult...
- [Xenova (@xenovacom) 的推文](https://x.com/xenovacom/status/1799110540700078422?s=46&t=90xQ8sGy63D2OtiaoGJuww)：终于实现了：使用 OpenAI Whisper 在浏览器中进行实时语音识别！🤯 该模型完全使用 Transformers.js 和 ONNX Runtime Web 在设备本地运行，并支持多语言转录...
- [Ashok Elluswamy (@aelluswamy) 的推文](https://x.com/aelluswamy/status/1799646232559899098?s=46&t=PW8PiFwluc0tdmv2tOMdEg)：http://x.com/i/article/1799602451844345856
- [OpenAI (@OpenAI) 的推文](https://x.com/openai/status/1800240380220473552?s=46&t=90xQ8sGy63D2OtiaoGJuww)：我们正与 Apple 合作，将 ChatGPT 集成到 iOS、iPadOS 和 macOS 中 —— 将于今年晚些时候推出：https://openai.com/apple
- [Andrej Karpathy (@karpathy) 的推文](https://x.com/karpathy/status/1799949853289804266?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)：📽️ YouTube 上的新 4 小时（哈哈）视频讲座：“让我们复现 GPT-2 (124M)” https://youtu.be/l8pRSuU81PU 视频最后变得这么长是因为它非常... 全面：我们从空文件开始...
- [Suhail (@Suhail) 的推文](https://x.com/suhail/status/1800032099770273987?s=46&t=90xQ8sGy63D2OtiaoGJuww)：明天的交易将是十年一遇的。可能达到 MS-DOS/IBM 的级别。
- [Max Weinbach (@MaxWinebach) 的推文](https://x.com/maxwinebach/status/1800277157135909005?s=46)：这是来自 Apple 的 State of the Union。本地模型是一个 3B 参数的 SLM，它使用了为每个特定功能训练的适配器（adapters）。Diffusion 模型也做了同样的事情，为每种风格提供适配器。A...
- [Matthew Green (@matthew_d_green) 的推文](https://x.com/matthew_d_green/status/1800291897245835616?s=46&t=90xQ8sGy63D2OtiaoGJuww)：所以 Apple 推出了一套名为 “Private Cloud Compute” 的新系统，允许你的手机将复杂的（通常是 AI）任务卸载到云端专门的安全设备上。我仍在尝试弄清楚...
- [GitHub - fixie-ai/ultravox](https://github.com/fixie-ai/ultravox)：通过在 GitHub 上创建账户来为 fixie-ai/ultravox 的开发做出贡献。
- [开发 LLM：构建、训练、微调](https://open.substack.com/pub/sebastianraschka/p/llms-building-training-finetuning?r=1h4isl&utm_medium=ios)：深入探讨 LLM 开发的生命周期
- [GitHub - normal-computing/extended-mind-transformers](https://github.com/normal-computing/extended-mind-transformers/)：通过在 GitHub 上创建账户来为 normal-computing/extended-mind-transformers 的开发做出贡献。
- [Dylan Patel (@dylan522p) 的推文](https://x.com/dylan522p/status/1799985803654991933?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)：开源正在改变 AI 和硬件。本周二在圣何塞，Jim Keller @jimkxa @tenstorrent、Raja Koduri @RajaXg、Charlie Cheng Andes 董事会、Chris Walker @UntetherAI 的 CEO https://www.eventbri...
- [Nick Dobos (@NickADobos) 的推文](https://x.com/nickadobos/status/1800289718439186455?s=46&t=90xQ8sGy63D2OtiaoGJuww)：Siri 可以读取你手机上的每一条数据（针对选择加入的应用）
- [Elon Musk (@elonmusk) 的推文](https://x.com/elonmusk/status/1800265431078551973?s=46&t=90xQ8sGy63D2OtiaoGJuww)：如果 Apple 在操作系统层面集成 OpenAI，那么 Apple 设备将在我的公司被禁用。这是不可接受的安全违规。
- [Marques Brownlee (@MKBHD) 的推文](https://x.com/mkbhd/status/1800223468627304657?s=46&t=90xQ8sGy63D2OtiaoGJuww)：好吧，你知道吗？这太酷了。Math Notes = 用 Apple pencil 写下一个数学题，应用会立即解决它。他们没有称之为 AI（他们还没说过一次这个词），但...
- [Rohan Paul (@rohanpaul_ai) 的推文](https://x.com/rohanpaul_ai/status/1799122826114330866?t=DajZXqRteA0XqfzcM)：这真是一篇令人惊叹的论文。🤯 声称可以在 LLMs 中完全消除 MatMul 操作，同时在十亿参数规模下保持强劲性能，并利用一种优化...
- [Steven Sinofsky (@stevesi) 的推文](https://x.com/stevesi/status/1800314848070557864?s=46&t=90xQ8sGy63D2OtiaoGJuww)：以防万一还不清楚，Apple 所做的是（对 OpenAI）搜索交易的反向操作。与其获得报酬，无论他们支付多还是少都不重要，这将是在有限的时间内...

- [来自 Andrej Karpathy (@karpathy) 的推文](https://x.com/karpathy/status/1800242310116262150?s=46&t=90xQ8sGy63D2OtiaoGJuww)：事实上，非常喜欢 Apple Intelligence 的发布。对于 Apple 来说，这一定是一个非常令人兴奋的时刻，因为他们将 AI 叠加在整个 OS 之上。几个主要主题：第一步是 Multimodal I/O。启用...
- [一张图片价值 170 个 Tokens：GPT-4o 如何编码图像？ - OranLooney.com](https://www.oranlooney.com/post/gpt-cnn/)：事实如下：GPT-4o 在高分辨率模式下处理每个 512x512 的切片（tile）需要消耗 170 个 tokens。按照约 0.75 tokens/单词计算，这表明一张图片大约相当于 227 个单词——仅为...的四倍。
- [来自 John Paczkowski (@JohnPaczkowski) 的推文](https://x.com/johnpaczkowski/status/1799135156051255799?s=46&t=90xQ8sGy63D2OtiaoGJuww)：我们关于 Eric Schmidt 秘密无人机项目的报道今天上午被 @perplexity_ai 发布了。它抄袭了我们大部分的报道。它将我们以及一些转载我们内容的来源列为引用，以最容易...
- [来自 Bilawal Sidhu (@bilawalsidhu) 的推文](https://x.com/bilawalsidhu/status/1800355980829405603?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)：好吧，我收回之前的话。Apple 的 “Private Cloud Computing” 实际上将 “Confidential Computing” 提升到了一个新的水平。它是如此安全，以至于他们甚至无法配合执法部门的请求。> 没有数据...
- [生物学、佛教与 AI：关怀作为智能的驱动力](https://www.mdpi.com/1099-4300/24/5/710)：智能是人类原生体验和人际体验的核心特征。理解智能在进化过程中是如何起源和扩展的，是现代生物学面临的一个关键挑战。因此...
- [来自 Matthew Green (@matthew_d_green) 的推文](https://x.com/matthew_d_green/status/1800291897245835616?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)：所以 Apple 推出了一套名为 “Private Cloud Compute” 的新系统，允许你的手机将复杂的（通常是 AI）任务卸载到云端专门的安全设备上。我仍在尝试弄清楚...

---

### **Latent Space ▷ #**[**ai-announcements**](https://discord.com/channels/822583790773862470/1075282504648511499/1249560567077928970) (8 条消息🔥):

- **新 ICLR 播客集发布**：作为 [ICLR 2024 系列](https://www.latent.space/p/iclr-2024-benchmarks-agents)的第 2 部分，新的播客集已发布。本集包含与 Graham Neubig 和 Aman Sanger 的讨论，涵盖了代码编辑、sandboxes 以及学术界与工业界的交集等主题。
- **AI Engineer World’s Fair 演讲者名单公布**：[AI Engineer World’s Fair](https://www.ai.engineer/worldsfair) 的第二波演讲者名单已[公布](https://x.com/swyx/status/1797654825968291862)。该会议的 Platinum/Gold/Silver 赞助位和早鸟票已售罄，更多信息可在其 [Microsoft 专题集](https://www.latent.space/p/worlds-fair-2024#%C2%A7show-notes)中查看。
- **HN 提交时机策略**：一位用户在太平洋时间上午 9 点左右将 ICLR 2024 系列提交到了 Hacker News，并指出这是一个获得曝光的好时段。
- **在 X 上的推广协作**：讨论了如何在 X 上处理 ICLR 系列的推广，最终决定由一名成员负责推广，另一名成员转发并补充背景信息。还建议更新现有的 X [帖子](https://x.com/latentspacepod/status/1795196817044594817)以包含最新信息。

**提到的链接**：

- [ICLR 2024 — 最佳论文与演讲（Benchmarks, Reasoning & Agents） — 嘉宾：Graham Neubig, Aman Sanger, Moritz Hardt)](https://www.latent.space/p/iclr-2024-benchmarks-agents)：1 场深度访谈，以及来自 ICLR 2024 的另外 12 篇论文和 3 场演讲，涵盖了像 OpenDevin 这样的 Coding Agents、Benchmark 的科学、Reasoning 和 Post-Training 以及 Agent 系统！
- [来自 Latent Space Podcast (@latentspacepod) 的推文](https://x.com/latentspacepod/status/1795196817044594817)：🆕 ICLR 2024：最佳论文（第 1 部分）我们展示了我们挑选的优秀论文和演讲，主题性地介绍了 AI Engineer 需要关注的话题：A 部分：ImageGen、Compression、Adversarial ...

---

### **Latent Space ▷ #**[**ai-in-action-club**](https://discord.com/channels/822583790773862470/1200548371715342479/1248727976544178370) (98 messages🔥🔥):

- **实时流媒体面部识别令人惊叹**：一位成员在 Websim.ai 上发现了 [whothehellisthis.com](https://whothehellisthis.com)，将其描述为“实时流媒体面部识别网站”。他们觉得这“非常神奇”，引发了其他用户的兴趣。
- **Websim.ai 的递归实验**：用户尝试在 [websim.ai](https://websim.ai) 内部递归运行它，直到页面在四层深度时变得无响应。这引发了关于其能力的玩笑和着迷。
- **分享 Websim 资源电子表格**：一位用户分享了一个 [Google Sheets 文档](https://docs.google.com/spreadsheets/d/1F-9MgOsDlevbiokl9u06IrKWoy3tlrCWlMRXDxvoDCY/edit#gid=2061123208)，其中包含与 Websim.ai 相关的各种链接和资源。其中包括 Websim 系统提示词（system prompt）的 GitHub Gist 链接，引发了进一步的兴趣和互动。
- **Greentext 生成器和可塑网页**：一位用户提到在 Websim 上创建了一个“greentext 生成器”，而另一位用户对 Websim 的流式传输机制表示好奇。一个演示 URL 引发了关于通过 Websim 为本地服务创建可用前端的讨论。
- **未来的主持与会议**：成员们讨论了为即将到来的会议设置发布者用户作为主持人，并商定了一项审查并可能记录未来会议的计划。最后，大家对会议内容表达了感谢和热情。

**提到的链接**：

- [worldsim](https://worldsim.nousresearch.com/)：未找到描述
- [websim.txt](https://gist.github.com/SawyerHood/5d82679953ced7142df42eb7810e8a7a)：GitHub Gist：即时分享代码、笔记和片段。
- [Latent Space Friday AI In Action: Websim](https://docs.google.com/spreadsheets/d/1F-9MgOsDlevbiokl9u06IrKWoy3tlrCWlMRXDxvoDCY/edit#gid=2061123208)：资源名称、链接、备注 Websim，<a href="https://websim.ai/">https://websim.ai/</a> 播客剧集，<a href="https://www.latent.space/p/sim-ai">https://www.latent.sp...
- [AI In Action: Weekly Jam Sessions](https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0)：2024 主题、日期、主持人、资源、@dropdown、@ GenAI 的 UI/UX 模式，1/26/2024，nuvic，<a href="https://maggieappleton.com/squish-structure">https://maggieappleton.com/squish-stru...
- [WebSim.AI - Self-Referential Simulated Web](https://websim.ai/c/2PLjreKO66U6TOhES)：未找到描述
- [Cyberpunk Chat Room](https://t.co/evC8wiHkYz)：未找到描述

---

### **Cohere ▷ #**[**general**](https://discord.com/channels/954421988141711382/954421988783444043/1248937097096859688) (98 messages🔥🔥):

- **探索 Cohere 的多功能平台**：用户讨论了 Cohere 的模型列表及其在 [Amazon SageMaker](https://aws.amazon.com/marketplace/seller-profile?id=87af0c85-6cf9-4ed8-bee0-b40ce65167e0) 和 [Microsoft Azure](https://ai.azure.com/explore/models/?tid=694fed05-7f6d-4ab2-8c38-9afb438eab6f&selectedCollection=cohere) 等多个平台上的可用性。一位成员澄清说，**Command R 和 R+ 模型**是最新且更优的版本。
- **与 AI 进行角色扮演**：一位用户分享了通过使用特定的 tool calls（如 "reply_to_user"）而非通用的 "directly_answer" 工具来改进角色回复的见解。他们正将这些实践集成到他们的 [Dungeonmasters.ai](https://www.dungeonmasters.ai/) 项目中，以增强 AI 驱动的叙事体验。
- **社区成员自我介绍**：几位新成员介绍了自己，包括一位来自巴西的初级 NLP DS 和一位正在探索 reranking 模型的 MIT 毕业生。他们表达了加入 Cohere 社区并进行协作的兴奋之情。
- **项目与职业抱负**：对话包括关于学术表现和职业目标的讨论。成员们还强调了一些令人兴奋的项目，其中一个项目需要与 Cohere 合作开发 AI 驱动的游戏体验。
- **鼓励与动力**：社区提供了支持和动力，讨论了理解 NLP 和利用 AI API 如何促进有影响力的项目开发。成员们互相祝贺并鼓励大家成功申请实习并顺利完成项目。

**提到的链接**：

- [Models](https://docs.cohere.com/docs/models)：未找到描述
- [Dungeonmasters.ai](https://www.dungeonmasters.ai/)：发现 Dungeonmaster：一个为 NovelAI 打造的动态前端，提供独特的叙事和图像生成体验。现在就潜入沉浸式的文字冒险和创意之旅吧！

---

### **Cohere ▷ #**[**project-sharing**](https://discord.com/channels/954421988141711382/1218409701339828245/1248779338136948949) (4 条消息):

- **滚动条主题化建议已获认可**：一名成员建议“为这些滚动条设置主题（theme these scrollbars）”，另一名成员对此给出了积极回应，表示：“好主意，很快就会添加。”
- **Cohere API 性能受到赞誉**：在收到关于项目性能的正面反馈（“运行效果非常好！🔥”）后，另一名成员将成功归功于 **Cohere API**，并表示：“感谢强大的 Cohere API 💪。”

---

### **Cohere ▷ #**[**announcements**](https://discord.com/channels/954421988141711382/996880279224451154/1249659607924736030) (1 条消息):

- **Cohere SDK 进军云端**：Cohere SDK 现在已兼容 AWS、Azure 和 Oracle 等多个云平台。用户可以使用来自 [Python SDK](https://docs.cohere.com/docs/cohere-works-everywhere) 的全新 Bedrock Cohere 客户端在 Bedrock 上开始使用 Cohere，从而为开发中的后端选择提供灵活性。

**提到的链接**：[Cohere SDK 云平台兼容性](https://docs.cohere.com/docs/cohere-works-everywhere)：未找到描述

---

### **LAION ▷ #**[**general**](https://discord.com/channels/823813159592001537/823813160075132991/1248761134677229569) (71 条消息 🔥🔥):

- **关于使用加密货币支付 AI 算力的辩论**：成员们讨论了使用加密货币支付 AI 算力的可行性和潜在陷阱。有人声称“你已经可以使用加密货币购买 vast.ai 额度”，而另一人则批评这一想法是“又一个 Emad 的加密货币骗局（crapto scam）”。
- **社区警示恶意 ComfyUI 节点**：一名成员提醒他人注意 [ComfyUI_LLMVISION 节点](https://www.reddit.com/r/comfyui/comments/1dbls5n/psa_if_youve_used_the_comfyui_llmvision_node_from/) 的恶意行为，称其“会尝试窃取信用卡详情等信息”。他们强调，“如果你安装并使用了该节点，你的浏览器密码、信用卡信息和浏览历史记录已经通过 Webhook 发送到了一个 Discord 服务器。”
- **新文本生成图像模型 Lumina-Next-T2I**：成员们分享了关于 [Lumina-Next-T2I 模型](https://huggingface.co/Alpha-VLLM/Lumina-Next-T2I) 的更新，该模型承诺“更快的推理速度、更丰富的生成风格以及更多的多语言支持”。另一人链接了 [一条 Twitter 更新](https://x.com/liuziwei7/status/1799846727534727649)，展示了该模型的能力和可用 Demo。
- **LAION 在巴西引发争议**：一名成员提到 LAION 在巴西电视节目中受到了负面报道。其他人链接了一篇讨论在 [人权观察组织（Human Rights Watch）](https://www.hrw.org/news/2024/06/10/brazil-childrens-personal-photos-misused-power-ai-tools) 上滥用儿童个人照片进行 AI 训练的文章。
- **关于 AI 与隐私的普遍误解**：针对公众的担忧，成员们声称公众并不理解“生成式模型并未侵犯任何人的隐私”。他们认为这些模型“不会记忆个人的随机图像”，并且对这类技术的恐惧在很大程度上是毫无根据的。

**提到的链接**：

- [Ziwei Liu (@liuziwei7) 的推文](https://x.com/liuziwei7/status/1799846727534727649?t=SfYyVjiiYqvERmKAtIZWAA&s=19)：🔥Lumina-Next🔥 是一个更强大、更快速的高分辨率文本生成图像模型。它还支持 1D（音乐）和 3D（点云）生成 - T2I Demo: http://106.14.2.150:10020/ - 代码: https://gi...
- [Alpha-VLLM/Lumina-Next-T2I · Hugging Face](https://huggingface.co/Alpha-VLLM/Lumina-Next-T2I)：未找到描述
- [Simo Ryu (@cloneofsimo) 的推文](https://x.com/cloneofsimo/status/1799819817333219662)：目前的非精选（non-cherry-picked）结果。很快将通过改进的 MFU 和方法加倍算力。
- [GitHub - hamadichihaoui/BIRD: 这是“通过快速扩散反演进行盲图像修复”的官方实现](https://github.com/hamadichihaoui/BIRD)：这是 "Blind Image Restoration via Fast Diffusion Inversion" 的官方实现 - hamadichihaoui/BIRD
- [Reddit - 深入了解任何事物](https://www.reddit.com/r/comfyui/comments/1dbls5n/psa_if_youve_used_the_comfyui_llmvision_node_from/)：未找到描述
- [FxTwitter / FixupX 的推文](https://x.com/search?q=blockchain%20(from%3Aemostaque)&src=typed_query)：抱歉，该用户不存在 :(

---

### **LAION ▷ #**[**research**](https://discord.com/channels/823813159592001537/824374369182416994/1248714867561730109) (23 messages🔥):

- **使用 σ-GPT 进行动态序列生成**：一种新方法 σ-GPT 通过为输出添加位置编码，引入了即时动态序列生成，从而实现了对任意 Token 子集的采样和条件化。这显著减少了在语言建模和路径求解等各个领域中的模型评估次数 ([阅读更多](https://arxiv.org/abs/2404.09562))。
- **自回归模型的替代方案**：虽然 σ-GPT 展示了令人印象深刻的结果，但成员们对其实际应用表示担忧，因为实现高性能需要特定的 Curriculum（课程学习），并将其与未能流行起来的 XLNET 进行了类比 ([Twitter 讨论](https://x.com/ArnaudPannatier/status/1799055129829839166))。
- **Transformer 嵌入分析**：有关 Transformer 中学习到的嵌入性质的查询，比较了离散和连续表示。参考的一篇 [2019 年论文](https://arxiv.org/abs/1905.09418) 提供了关于注意力头如何贡献模型性能以及如何在损失极小的情况下进行大幅剪枝的见解。
- **基于 Prompt 的推理挑战**：分享了一个 [GitHub 仓库](https://github.com/cpldcpu/MisguidedAttention)，其中包含挑战 LLM 推理能力的 Prompt，揭示了模型失败通常源于训练数据中过度代表的问题。
- **条件嵌入扰动测试**：对条件嵌入扰动的实验表明，应用高斯噪声（在不同的 Gamma 水平下）会影响模型对 Prompt 的遵循程度，在高 Gamma 设置下结果显著 ([实验结果](https://x.com/panopstor/status/1798481967391945186))。

**提到的链接**：

- [Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned](https://arxiv.org/abs/1905.09418)：多头自注意力是 Transformer 的关键组件，Transformer 是神经机器翻译的先进架构。在这项工作中，我们评估了单个注意力头的贡献...
- [来自 Arnaud Pannatier (@ArnaudPannatier) 的推文](https://x.com/ArnaudPannatier/status/1799055129829839166)：GPT 以从左到右的顺序生成序列。还有其他方法吗？与 @francoisfleuret 和 @evanncourdier 合作，并与 @SkysoftATM 合作，我们开发了 σ-GPT，能够生成序列...
- [来自 Panopstor (@panopstor) 的推文](https://x.com/panopstor/status/1798481967391945186)：条件嵌入扰动实验。🧵
- [σ-GPTs: A New Approach to Autoregressive Models](https://arxiv.org/abs/2404.09562)：自回归模型（如 GPT 系列）使用固定的顺序（通常是从左到右）来生成序列。然而，这并非必要。在本文中，我们挑战了这一假设并展示了...
- [GitHub - cpldcpu/MisguidedAttention: A collection of prompts to challenge the reasoning abilities of large language models in presence of misguiding information](https://github.com/cpldcpu/MisguidedAttention)：一个挑战大语言模型在存在误导信息时推理能力的 Prompt 集合 - cpldcpu/MisguidedAttention

---

### **LAION ▷ #**[**resources**](https://discord.com/channels/823813159592001537/991938328763056168/) (1 messages):

sidfeels: [https://youtu.be/l8pRSuU81PU](https://youtu.be/l8pRSuU81PU)

---

### **LAION ▷ #**[**learning-ml**](https://discord.com/channels/823813159592001537/991941292999323668/1249575130548666408) (4 messages):

- **利用预训练的 Instruct 模型**：一位成员建议在项目中使用具有代码能力的预训练 Instruct 模型，通过增加其上下文窗口并输入 Rust 文档。他们提到一位 Google 科学家推荐这种方法，而不是从头开始训练。
- **通用错误消息排查**：一位成员解释说，Websocket 中的通用错误消息可能是由于各种问题引起的，例如重新安装 Websocket 或模型处理操作的方式。他们建议提供调试控制台输出，运行单独的测试代码，并编写测试用例来识别问题。
- **Websocket 音频响应延迟**：一位成员描述了文本转语音 (TTS) 服务 Websocket 的行为，指出在第一次浏览器刷新后，Websocket 收到 1001 going away 信号，并表现出音频数据包延迟一个轮次的滞后。这种滞后在随后的刷新后会恶化，音频数据包随后会延迟多个轮次，尽管同一容器中的其他 Websocket 运行正常。

### **LlamaIndex ▷ #**[**announcements**](https://discord.com/channels/1059199217496772688/1073670729054294197/1249874265130537001) (1 条消息):

- **高级知识图谱 RAG 研讨会发布**：本周四太平洋时间上午 9 点将举行一场关于**高级知识图谱 RAG** 的特别研讨会，由来自 Neo4j 的 Tomaz Bratanic 主讲。与会者将学习 LlamaIndex 属性图抽象，包括基于 Neo4j 的高级属性图索引，以及图构建和查询的详细方面。[在此注册](https://lu.ma/kqxmbuou)

**提到的链接**：[LlamaIndex 网络研讨会：使用知识图谱的高级 RAG（与来自 Neo4j 的 Tomaz 合作）· Zoom · Luma](https://lu.ma/kqxmbuou)：我们将在本周四太平洋时间上午 9 点举办一场关于高级知识图谱 RAG 的特别研讨会，邀请了来自 Neo4j 独一无二的 Tomaz Bratanic。在本次研讨会中，你将……

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1248736171757011108) (7 条消息):

- **集成 e2b_dev 的沙箱以增强数据分析**：[Create-llama](https://twitter.com/llama_index/status/1799176083381866757) 现在集成了 @e2b_dev 的沙箱，使用户不仅可以编写 Python 代码来分析数据，还可以返回整个文件（如属性图图像）。这种集成显著拓宽了 Agent 的潜力。
- **学习构建 Agentic RAG 系统**：推荐由 @Prince_krampah 编写的[综合博客/教程系列](https://twitter.com/llama_index/status/1799463683179098203)来构建 RAG 系统，涵盖了从基础路由到复杂文档多步推理的所有内容。
- **用于增强 RAG 的查询重写**：@kingzzm 关于[三种形式查询重写](https://twitter.com/llama_index/status/1799566113208225891)的资源对于通过加强查询理解层来改进 RAG 中的问题处理至关重要，这对于有效检索非常关键。
- **构建客服语音机器人**：@rborgohain4 的教程展示了如何使用 @Inferless_、@llama_index、faster-whisper、Piper 和 @pinecone 构建一个[极速客服语音机器人](https://twitter.com/llama_index/status/1799833244894200135)。这标志着超越传统聊天机器人的下一步进化。
- **在企业云上保护你的 RAG 应用**：@pavan_mantha1 的[教程](https://twitter.com/llama_index/status/1799969601704563036)详细介绍了如何使用 @Azure 上的各种服务，通过 @qdrant_engine 和 OpenAI 来保护 RAG 流水线，包括用于增强安全措施的应用特定身份。

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1248715382089453580) (87 条消息🔥🔥):

<ul>
<li><strong>关于 Directory Reader 中 Chunk Size 增加的疑问</strong>：一位用户询问如何增加 LlamaIndex 中 SimpleDirectoryReader 的 chunk size。机器人提供了一个调整 `chunk_size` 参数的代码示例，并引用了 <a href="https://docs.llamaindex.ai/en/latest/optimizing/basic_strategies/basic_strategies/#chunk-sizes">LlamaIndex 文档</a>。</li>
<li><strong>Graph Store 中的实体解析讨论</strong>：成员们讨论了如何处理实体解析，建议手动删除并 upsert Graph Store 中的节点。分享了一个删除方法的示例，演示了如何指定实体名称、关系名称、属性或 ID 进行删除。</li>
<li><strong>向现有的 VectorStoreIndex 添加文档</strong>：一位用户寻求如何将额外的 PDF 添加到现有的 VectorStoreIndex 中。机器人建议使用 `insert` 方法逐个添加文档，并引用了 <a href="https://docs.llamaindex.ai/en/latest/understanding/storing/storing/#inserting-documents-or-nodes">文档</a>。</li>
<li><strong>LlamaParse 服务中断</strong>：用户遇到了 LlamaParse 服务的问题，报告请求卡在 pending 状态。该问题已由社区成员调查并迅速解决。</li>
<li><strong>RLHF 阶段与 Chunk Sizes</strong>：关于优化检索增强生成 (RAG) 系统的讨论，特别是在处理大量 PDF 时，强调了可扩展性的挑战。成员们建议使用结构化信息进行更精确的检索，并根据 <a href="https://docs.llamaindex.ai/en/latest/optimizing/production_rag/#motivation_1">LlamaIndex 文档</a> 优化策略。</li>
</ul>

**提到的链接**:

- [未找到标题](https://llamahub.ai/l/readers/llama-index-readers-s3?from=readers): 未找到描述
- [来自 Bagel 🥯 (@bagel_network) 的推文](https://x.com/bagel_network/status/1800201048302264533): Bagel 的实验室正在利用密码学推动 AI 的前沿。我们最新的博客文章探讨了可信执行环境 (TEEs) 和安全多方计算 (MPC)。了解这些技术如何...
- [探索 LlamaIndex: JSON Query Engine](https://www.youtube.com/watch?v=4tDyfAaIqEw): JSON 是一种非常流行的数据存储格式。到目前为止，检索增强流水线主要集中在解析/存储非结构化文本...
- [Storing - LlamaIndex](https://docs.llamaindex.ai/en/latest/understanding/storing/storing/#inserting-documents-or-nodes>): 未找到描述
- [Auto merging retriever - LlamaIndex](https://docs.llamaindex.ai/en/latest/api_reference/packs/auto_merging_retriever/#llama_index.packs.auto_merging_retriever.AutoMergingRetrieverPack>): 未找到描述
- [Building Performant RAG Applications for Production - LlamaIndex](https://docs.llamaindex.ai/en/latest/optimizing/production_rag/#motivation_1>): 未找到描述
- [Basic Strategies - LlamaIndex](https://docs.llamaindex.ai/en/latest/optimizing/basic_strategies/basic_strategies/#chunk-sizes>): 未找到描述
- [Building a (Very Simple) Vector Store from Scratch - LlamaIndex](https://docs.llamaindex.ai/en/latest/examples/low_level/vector_store/#generate-embeddings-for-each-node>): 未找到描述
- [[Beta] Text-to-SQL with PGVector - LlamaIndex](https://docs.llamaindex.ai/en/latest/examples/query_engine/pgvector_sql_query_engine/#generate-embedding-for-each-node-with-a-sentence_transformers-model>): 未找到描述

---

### **LlamaIndex ▷ #**[**ai-discussion**](https://discord.com/channels/1059199217496772688/1100478495295017063/1249701633479413862) (1 条消息):

- **根据手机手册创建数据集**：一位成员寻求帮助，希望根据其手机手册创建数据集。他们的目标是使用 **QLoRA** 训练模型，以改进该数据上的 **RAG** (Retrieval-Augmented Generation)。

---

### **OpenRouter (Alex Atallah) ▷ #**[**announcements**](https://discord.com/channels/1091220969173028894/1092729520181739581/1248718975190171660) (3 条消息):

- **Qwen 2 72B Instruct 上线：** [Qwen 2 72B Instruct](https://openrouter.ai/models/qwen/qwen-2-72b-instruct) 模型现已可用，由 OpenRouter 发布。
- **Dolphin 2.9.2 Mixtral 8x22B 作为实验性项目推出：** [Dolphin 2.9.2 Mixtral 8x22B](https://openrouter.ai/models/cognitivecomputations/dolphin-mixtral-8x22b) 现已可用，价格为 $1/M tokens。前提条件是下周内平均每天的使用量需达到 1.75 亿 tokens，否则将停止服务。建议用户在使用该模型时设置 fallback 以 *确保最佳运行时间*。
- **StarCoder2 15B Instruct 发布：** [StarCoder2 15B Instruct](https://openrouter.ai/models/bigcode/starcoder2-15b-instruct) 模型现已可供使用。

**提到的链接**：

- [StarCoder2 15B Instruct by bigcode](https://openrouter.ai/models/bigcode/starcoder2-15b-instruct)): StarCoder2 15B Instruct 在编程相关任务中表现出色，主要针对 Python。它是 BigCode 开发的首个自对齐（self-aligned）开源 LLM。该模型在没有任何人类标注的情况下进行了微调...
- [Qwen 2 72B Instruct by qwen](https://openrouter.ai/models/qwen/qwen-2-72b-instruct)): Qwen2 72B 是一款基于 Transformer 的模型，在语言理解、多语言能力、编程、数学和推理方面表现卓越。它具有 SwiGLU 激活函数、Attention QKV 偏置以及 gro...
- [Dolphin 2.9.2 Mixtral 8x22B 🐬 by cognitivecomputations](https://openrouter.ai/models/cognitivecomputations/dolphin-mixtral-8x22b)): Dolphin 2.9 专为指令遵循、对话和编程而设计。该模型是 [Mixtral 8x22B Instruct](/models/mistralai/mixtral-8x22b-instruct) 的微调版本。它具有 64k 上下文...

---

### **OpenRouter (Alex Atallah) ▷ #**[**app-showcase**](https://discord.com/channels/1091220969173028894/1092850552192368710/1249004787509366854) (4 条消息):

- **AI Code Brushes 插件展示**：一位成员分享了一个免费的 AI 代码转换插件，适用于 Visual Studio Code，使用了 OpenRouter 和 Google Gemini。[点击此处查看](https://marketplace.visualstudio.com/items?itemName=ThijsDekkers.ai-code-brushes)。
- **AI Code Brushes 兼容性讨论**：成员们讨论了 AI Code Brushes 插件的兼容性，强调虽然任何模型都可以运行，但 Programming/Scripting 类别中最受欢迎的模型往往表现最好。在此探索 [排名情况](https://openrouter.ai/rankings/programming/scripting?view=week)。

**提到的链接**：

- [AI Code Brushes - Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=ThijsDekkers.ai-code-brushes&ssr=false#overview): Visual Studio Code 扩展 - 利用 AI 增强你的编程能力
- [LLM Rankings: programming/scripting | OpenRouter](https://openrouter.ai/rankings/programming/scripting?view=week): 根据 Programming/Scripting 提示词的使用情况进行排名和分析的语言模型

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1248739235456876707) (75 messages🔥🔥):

- **Google 和 Apple Pay 支付集成**：成员们讨论了将 Google Pay 和 Apple Pay 集成到支付系统中，并指出了它们在移动端的可用性。还讨论了为那些不想使用应用程序的用户添加加密货币支付选项的建议。
- **处理 API 调用中的部分 JSON**：用户分享了在流式传输 OpenRouter chat completions 时接收到部分数据块（partial chunks）的挑战，并讨论了如为分块数据维护缓冲区等解决方案。一位用户引用了[这篇文章](https://blog.stackademic.com/swift-streaming-openai-api-response-chunked-encoding-transfer-48b7f1785f5f)以获取更多关于处理分块数据的见解。
- **角色扮演 Prompt 问题**：成员们交流了如何防止聊天机器人代表用户发言的技巧，并建议在 Prompt 中使用详细指令以确保更好的回复。分享了一个有用的指南：[Statuo 的机器人聊天进阶指南](https://rentry.co/statuotwtips#the-bot-speaks-for-you-as-part-of-its-introduction-or-as-part-of-its-example-dialogue)。
- **语言支持讨论**：有人请求并随后确认将添加语言类别，以便按语言熟练程度评估模型。用户期待对捷克语、法语、普通话等语言进行更好的分类。
- **LLM 中的审查与偏见**：讨论了一篇文章[《使用 Qwen 2 Instruct 分析中国 LLM 的审查与偏见》](https://huggingface.co/blog/leonardlin/chinese-llm-censorship-analysis)，比较了中国和美国 LLM 的审查方法，并辩论了这些偏见的影响。

**提到的链接**：

- [Stream response from `/v1/chat/completions` endpoint is missing the first token](https://community.openai.com/t/stream-response-from-v1-chat-completions-endpoint-is-missing-the-first-token/187835/9)：这部分有些偏离主题，但我偶然发现了这个帖子，并注意到你的代码中存在一些潜在问题，我想指出这些问题，因为关于这方面的信息非常少，而且……
- [Swift: Streaming OpenAI API Response (Chunked Encoding Transfer)](https://blog.stackademic.com/swift-streaming-openai-api-response-chunked-encoding-transfer-48b7f1785f5f)：当预期从 URLRequest 接收大量数据时，流式传输分块编码（Chunked encoded）数据非常有用。例如，当我们请求……
- [Statuo's Guide to Getting More Out of Your Bot Chats](https://rentry.co/statuotwtips#the-bot-speaks-for-you-as-part-of-its-introduction-or-as-part-of-its-example-dialogue)：Statuo 的机器人聊天进阶指南。介绍、机器人制作通用技巧和指南、垃圾进垃圾出、第一人称？第三人称？第二人称？我的机器人不连贯且正在遗忘……

---

### **OpenRouter (Alex Atallah) ▷ #**[**일반**](https://discord.com/channels/1091220969173028894/1246338143226167349/) (1 messages):

daun.ai: 哦！真是个好消息呢 哈哈

---

### **Interconnects (Nathan Lambert) ▷ #**[**events**](https://discord.com/channels/1179127597926469703/1179127598442348729/1248752370066919566) (1 messages):

- **旧金山《雷曼兄弟三部曲》多余门票**：一位成员提供了今天晚上 7 点在旧金山上演的**《雷曼兄弟三部曲》（Lehman Trilogy）戏剧**的多余门票。该剧时长 3 小时，该成员在决定发布到这里之前，曾尝试在 "x dot com 这个万能应用" 上出售门票，但未成功。
- **John Heffernan 的代表作**：John Heffernan 丰富的**戏剧**作品包括在《无事生非》（Much Ado About Nothing）、《圣乔治与龙》（Saint George and the Dragon）以及《爱德华二世》（Edward II）中的精彩表演。他的**电视**角色涵盖了《德古拉》（Dracula）和《王冠》（The Crown）等作品，而他的**电影**作品则包括《公爵》（The Duke）和《官方机密》（Official Secrets）。

**提到的链接**：[The Lehman Trilogy | Official Site](https://thelehmantrilogy.com/)：不要错过吉莉安·林恩剧院（Gillian Lynne Theatre）的“必看杰作”（《每日电讯报》）。观看这个改变世界的家族和公司的故事。

---

### **Interconnects (Nathan Lambert) ▷ #**[**ideas-and-feedback**](https://discord.com/channels/1179127597926469703/1179127598442348730/1248744307221991556) (6 messages):

- **采用 Dylan 的结构以提升内容质量**：一位成员建议采用类似于 Dylan 的结构，即提供高层级的概述，并为付费订阅者提供单独的深度探讨。这种细分可能会增强内容的质量和组织性。
- **质量重于速度**：Nathan Lambert 承认 Dylan 的内容更完整，并强调虽然他目前专注于实践和外联，但*“质量通常会取胜”*。
- **深度文章的团队与频率**：深度内容可能需要每两周或每月的更新节奏，特别是如果这并非全职工作。一位成员指出，Dylan 称职的团队是他产出深度内容的因素之一。
- **多样化的方法**：Nathan Lambert 对他目前不同的方法感到满意，并承认：*“目前尝试一些不同的东西，我做得还不错。”*

---

### **Interconnects (Nathan Lambert) ▷ #**[**news**](https://discord.com/channels/1179127597926469703/1179128538679488533/1248740236763205654) (40 messages🔥):

- **Nathan Lambert 赞赏 Apple 的 "personal intelligence"**：尽管对与 ChatGPT 的集成看法不一，Lambert 认为 Apple Intelligence 内容充实，并称其为“面向大众的 AI”。官方 [Apple 新闻稿](https://www.apple.com/newsroom/2024/06/introducing-apple-intelligence-for-iphone-ipad-and-mac/) 详细介绍了隐私和功能。
- **短视频主导了 text2video 模型讨论**：多位成员注意到像 TikTok 这样的公司专注于短视频，Sora 和 Kling 的对比示例显示出由于语言 Prompt 导致的差异。担忧包括数据集隐私以及中国在数据方面相对于西方公司的优势。
- **对 OpenAI 与 Apple 合作关系的怀疑**：Gurman 的泄密最初暗示了更深层次的 OpenAI 集成，但目前看来似乎比较表面。像 sid221134224 这样的成员强调了隐私政策冲突，即登录后，ChatGPT 的政策将覆盖 Apple 的政策。
- **对即将播出的 Dwarkesh 与 François Chollet 对谈节目的期待**：由于 François Chollet 对 AGI 时间线的看法更为审慎，成员们对 Dwarkesh Patel 即将进行的采访表示期待。Sid221134224 和 natolambert 认为这与之前的受访者相比是一个令人耳目一新的变化。

**提到的链接**：

- [Introducing Apple Intelligence for iPhone, iPad, and Mac](https://www.apple.com/newsroom/2024/06/introducing-apple-intelligence-for-iphone-ipad-and-mac/)：Apple 今日推出了 Apple Intelligence，这是适用于 iPhone、iPad 和 Mac 的个人智能系统。
- [Tweet from 歸藏(guizang.ai) (@op7418)](https://x.com/op7418/status/1799504701647643069?s=46&t=_jodDCDeIUnWb_Td0294bw)：让我们看看 Kling 距离其目标还有多远。Kling VS Sora。由于 Kling 目前仅接受中文，Prompt 可能会有所不同。引用 歸藏(guizang.ai) (@op7418) 我是否需要使用 mo...
- [Tweet from 歸藏(guizang.ai) (@op7418)](https://x.com/op7418/status/1799504701647643069?s=46&t=_jodDCDeIUnWb_Td)：让我们看看 Kling 距离其目标还有多远。Kling VS Sora。由于 Kling 目前仅接受中文，Prompt 可能会有所不同。引用 歸藏(guizang.ai) (@op7418) 我是否需要使用 mo...

---

### **Interconnects (Nathan Lambert) ▷ #**[**random**](https://discord.com/channels/1179127597926469703/1183121795247779910/1248726380712034464) (25 messages🔥):

- **Daylight Computer 引起关注**：成员们讨论了 [Daylight Computer](https://daylightcomputer.com) 及其独特功能，如减少蓝光和在阳光直射下的可见性。一位成员指出：“虽然我个人不会使用它，但我始终欣赏在产品设计中投入的努力和思考。”
- **对早期产品采用的担忧**：讨论了购买像 Daylight Computer 这样新技术早期版本的风险。正如一位成员所说：“很难预先知道公司是否会真诚地对待他们的早期采用者。”
- **联系创始人与产品测试**：有人建议联系 Daylight 创始人索要测试机。一位成员幽默地提到：“发个邮件然后忘掉它就行。🙂”
- **离开湾区**：Nathan Lambert 宣布他将离开湾区，并提到：“再见湾区 ✌️✌️😢😢🥲。”
- **寻求语言建模教程**：Nathan Lambert 询问是否有近期关于语言建模的教程（特别是来自大型 AI 会议的），用于向 NeurIPS 提交提案。

**提到的链接**：

- [Daylight | A More Caring Computer](https://daylightcomputer.com)：Daylight Computer (DC1) 是一种新型的“冷静”电脑，专为深度工作和健康而设计。
- [来自 murat 🍥 (@mayfer) 的推文](https://x.com/mayfer/status/1794971883856949249)：该死，这家伙的气场无懈可击，纯粹出于尊重刚订购了一台。引用 Jason Carman (@jasonjoyride) 在 S³ 第 45 集中介绍由 @daylightco 打造的全球首款 60+ FPS e-ink 显示屏...

---

### **Interconnects (Nathan Lambert) ▷ #**[**rl**](https://discord.com/channels/1179127597926469703/1208183216843005962/1249837969246916749) (3 messages):

- **混乱还是未经证实？**：一位用户考虑为 **TRL** 贡献实现论文中一种未经证实的方法，但对其有效性表示怀疑。Nathan Lambert 澄清说“混乱”不是一个准确的词，并指出它是“未经证实的”。
- **提供评审**：Nathan Lambert 提出愿意评审任何与该未经证实方法相关的 Pull Requests (PRs)。他表示：“如果你提交了 PR 请告诉我，我很乐意评审。”

---

### **Interconnects (Nathan Lambert) ▷ #**[**reads**](https://discord.com/channels/1179127597926469703/1214764639397617695/1249751329916850206) (7 messages):

- **SRPO 解决 RLHF 任务依赖问题**：一篇来自 [Cohere](https://cohere.com/research/papers/self-improving-robust-preference-optimization-2024-06-07) 的分享论文提出了 **Self-Improving Robust Preference Optimization (SRPO)**，旨在解决现有 RLHF 方法高度依赖任务的问题。该论文引入了一个具有数学原理的离线 RLHF 框架，通过 min-max 优化策略实现分布外（out-of-distribution）任务的鲁棒性。
- **RL 频道讨论 SRPO**：引用了 RL 频道中关于 SRPO 论文的持续讨论，并将其与 **Deterministic Policy Optimization (DPO)** 进行了比较。一位成员指出，这似乎主要是一篇理论论文，并以“拭目以待”作为总结。

**提到的链接**：[Self-Improving Robust Preference Optimization](https://cohere.com/research/papers/self-improving-robust-preference-optimization-2024-06-07)：未找到描述内容。

---

### **LangChain AI ▷ #**[**general**](https://discord.com/channels/1038097195422978059/1038097196224086148/1248726499553447956) (66 条消息🔥🔥):

- **Markdown 文件处理问题**：一位成员讨论了在 LangChain 中处理 Markdown 文件任务时遇到的困难，一个 25MB 的文件似乎在无限期运行。讨论中尚未提供解决方案。
- **LangChain 与 Bagel 集成公告**：一位成员分享了一条关于 LangChain 和 Bagel 新集成的 [推文](https://x.com/bagel_network/status/1799143240769081731)，强调了实现安全且可扩展的数据集管理。
- **可自定义的 Tagging Chains**：一位成员询问关于在非 OpenAI 模型中使用 `create_tagging_chain()` 并自定义 Prompt 的问题，但提到遇到了 Prompt 被忽略的问题。
- **处理 Retrieval 中的特殊字符**：一位成员在用 GitHub 文档填充 Retriever 时遇到了处理特殊字符的问题，并寻求帮助以确保正确的 Pydantic 模型输出。
- **优化 Document Loaders 和 Splitters**：关于加载和分块各种文档类型（如 PDF、Java 代码、Excel 文件）以在 LangChain 中获得最佳结果的广泛讨论。一位成员强调这个过程更多是一门艺术而非科学。

**提到的链接**：

- [no title found](http://localhost:6333')): 未找到描述
- [来自 Bagel 🥯 (@bagel_network) 的推文](https://x.com/bagel_network/status/1799143240769081731)：.@LangChainAI 弥合了语言模型与外部数据源之间的鸿沟，使开发强大的应用程序变得简单。现在，凭借 Bagel 的微调能力和 LangChain 的框架...
- [来自 Bagel 🥯 (@bagel_network) 的推文](https://x.com/bagel_network/status/1800201048302264533)：Bagel 的实验室正在利用密码学推动 AI 的前沿。我们最新的博客文章探讨了可信执行环境 (TEEs) 和安全多方计算 (MPC)。了解这些技术如何...
- [Google AI | 🦜️🔗 LangChain](https://python.langchain.com/v0.2/docs/integrations/llms/google_ai/#setting-up>))：您当前所在的页面记录了将 Google 模型作为文本补全模型的使用方法。许多流行的 Google 模型是聊天补全模型。
- [Document loaders | 🦜️🔗 LangChain](https://python.langchain.com/v0.1/docs/integrations/document_loaders/)：未找到描述
- [Retrieval | 🦜️🔗 LangChain](https://python.langchain.com/v0.1/docs/modules/data_connection/)：许多 LLM 应用程序需要不属于模型训练集的特定用户数据。
- [Reddit - 洞察万象](https://www.reddit.com/r/LangChain/comments/1dcprk4/how_to_get_ai_agent_to_do_follow_up_questions_and/)：未找到描述
- [Issues · langchain-ai/langchain](https://github.com/langchain-ai/langchain/issues/5308>),)：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。
- [如何在子链之间进行路由 | 🦜️🔗 LangChain](https://python.langchain.com/v0.2/docs/how_to/routing/#using-a-custom-function-recommended>)：本指南假设您已熟悉以下概念：

---

### **LangChain AI ▷ #**[**langserve**](https://discord.com/channels/1038097195422978059/1170024642245832774/1249626480930062429) (1 条消息):

- **需要使用 api_handler() 的最小示例**：一位成员寻求使用 **api_handler()** 的帮助，以便在不使用 **add_route()** 的情况下获得 Playground。他们特别提到想使用显式参数 *playground_type="default" 或 "chat"* 来锁定端点。

---

### **LangChain AI ▷ #**[**share-your-work**](https://discord.com/channels/1038097195422978059/1038097372695236729/1249328603812855859) (7 条消息):

- **Rubik's AI 免费 Beta 测试邀请**：邀请用户测试一款新型高级研究助手和搜索引擎，使用促销代码 `RUBIX` 可获得 2 个月的免费高级会员资格。该平台包含 GPT-4 Turbo、Claude-3 Opus 和 Mistral Large 等模型 ([Rubik's AI](https://rubiks.ai/))。
- **LangChain 和 DashScope Reranker 文章**：分享了一篇题为 *“Unleash the Search Beast: Langchain and DashScope Reranker”* 的 Medium 文章，适合对使用这些工具增强搜索和排名功能感兴趣的用户。[阅读文章](https://medium.com/ai-advances/unleash-the-search-beast-langchain-and-dashscope-reranker-67cbfdbaed0b)。
- **面向新闻业的 MIT 可视化工具**：一款旨在帮助记者识别缺乏媒体报道的热门学术研究课题的新型可视化工具正在征求反馈。该工具是开源的，可在 [GitHub](https://github.com/sundai-club/news-coverage) 上获取，演示版可以在[这里](http://ai-news-hound.sundai.club/)试用。
- **AI 音频新闻简报原型**：征求对新型 AI 驱动的音频新闻简报服务的反馈，该服务允许用户收听新闻故事并提问以更好地理解。感兴趣的用户可以在 [Loom](https://loom.com/share/248fc473ebcb4b52b46b4c4278d4c80e) 上查看演示视频。
- **Hugging Face 上的 Chat With 'Em**：这个新的 Hugging Face Space 允许用户与 Groq、Anthropic、OpenAI 和 Cohere 等多个 AI 模型聊天。它易于自定义，支持使用 API key 在不同模型之间切换 ([Chat With 'Em](https://huggingface.co/spaces/as-cle-bert/chat-with-em))。

**提到的链接**：

- [AI news briefing prototype, audio-only (to use driving/walking)](https://loom.com/share/248fc473ebcb4b52b46b4c4278d4c80e)：AI 新闻简报原型。**仅限音频，适合驾驶/步行时使用！**（我只是展示故事，以便你看到它正在阅读的内容）。该简报能快速总结新闻、播客中的所有内容...
- [Streamlit](http://ai-news-hound.sundai.club/)：未找到描述
- [Adding a Chat Component to A Parallel Agent Flow](https://www.youtube.com/watch?v=SHPd500E3k4&t=36s)：在此视频中，我将为 SQL Agent 提供问题的流程块替换为交互式聊天组件。该聊天组件显示了一个传统的...
- [Rubik's AI - AI research assistant & Search Engine](https://rubiks.ai/)：未找到描述

---

### **LangChain AI ▷ #**[**tutorials**](https://discord.com/channels/1038097195422978059/1077843317657706538/1249741210638094436) (1 条消息):

- **构建 LLM 应用的分步指南发布**：一位成员分享了[构建 LLM 应用的分步指南](https://towardsdatascience.com/building-llm-apps-a-clear-step-by-step-guide-1fe1e6ef60fd)，总结了他们过去 2 年的研究和经验。他们鼓励读者快速阅读，给予 50 次鼓掌并分享想法。

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general**](https://discord.com/channels/1104757954588196865/1104757955204743201/1248754079237406850) (16 条消息🔥):

- **频道镜像困惑**：一名成员询问为什么频道无法镜像到其他服务器，怀疑需要将其设置为“announcements”（公告）频道。另一名成员建议服务器必须是启用了特定设置的公开社区服务器才能使用此功能。
- **数据集加载的技术问题**：一位用户报告称，数据集加载失败可能是由于文件名包含括号，触发了诸如 `datasets.arrow_writer.SchemaInferenceError: Please pass 'features' or at least one example when writing data` 之类的错误。
- **训练中的替代指令格式**：讨论了 ShareGPT 是否是最佳格式，或者 reflection 或特殊指令模板等替代方案是否更好。一位成员澄清说，ShareGPT 在训练期间会被转换为模型的 prompt 格式。
- **Apple 模型的基准测试**：一位成员分享了一个 [Twitter 链接](https://x.com/ldjconfirmed/status/1800355063120151031)，其中包含将 Apple 新的端侧（on-device）和服务器模型与其他流行模型在指令遵循和写作能力方面的基准测试对比。
- **Axolotl 的再生能力**：另一位用户分享了一个 [YouTube 视频](https://youtube.com/shorts/OvD30K-KN3k?si=98hJ49tKNeuhfddo)，重点介绍了蝾螈（axolotls）的再生能力，提到它们能在几周内重新长出不同的身体部位。

**提到的链接**：

- [来自 LDJ (@ldjconfirmed) 的推文](https://x.com/ldjconfirmed/status/1800355063120151031)：如果有人好奇，这里有一些 Apple 新的端侧模型和服务器模型与其它流行模型在指令遵循和写作能力方面的基准测试对比。
- [蝾螈能够在短短几周内重新长出肢体、尾巴、鳃、大脑和心脏](https://youtube.com/shorts/OvD30K-KN3k?si=98hJ49tKNeuhfddo)：未找到描述

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general-help**](https://discord.com/channels/1104757954588196865/1110594519226925137/1248851845733154896) (16 条消息🔥):

- **使用 pip3 安装包的解决方法**：一位用户发现，分开安装包（如 `pip3 install -e '.[deepspeed]'` 和 `pip3 install -e '.[flash-attn]'`）与同时安装相比，可以避免 RAM 溢出。他们在 Python 3.10 的新 conda 环境中执行了此操作。
- **axolotl 中的多模态微调**：一位成员询问 **axolotl** 是否支持多模态微调。另一位成员提到曾有一个用于此目的的旧 **Qwen** 分支，但最近一直没有活动。
- **Qwen 2 数据预处理问题**：一位用户在为 Qwen 2 预处理数据时遇到错误，原因是 `AttributeError: LLAMA3`。该错误似乎与 **ShareGPT** 和 **ChatML** 有关。
- **使用 DPO 对齐 SFT 模型**：一位成员询问在将 SFT 模型与 DPO 对齐时，应该包含整个对话历史还是仅包含最后一轮。回复建议测试这两种方法，但指出 **axolotl** 目前的 DPO 可能仅在单轮上进行训练。
- **测试微调后的模型**：一位用户询问如何使用测试集测试其微调后的模型。回复强调了 `test_dataset:` 配置的存在，以方便进行此操作。

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**datasets**](https://discord.com/channels/1104757954588196865/1112023441386778704/1248717893538025522) (1 条消息):

- **成功配置 JSONL 数据集**：一位成员分享了一个成功配置，使用 **常规 JSONL 文件** 并为训练和测试数据集指定了路径。配置包括用于训练的 **alpaca_chat.load_qa** 路径和用于评估的 **context_qa.load_v2** 路径，格式符合文档要求。

### **OpenAccess AI Collective (axolotl) ▷ #**[**axolotl-phorm-bot**](https://discord.com/channels/1104757954588196865/1225558824501510164/1249127821780189348) (8 messages🔥):

- **修改 Epochs 时调整学习率**：当更改 Epochs 数量时，应按相同比例降低学习率，以保持每个数据样本的学习量一致。这可以补偿增加的更新次数。
- **Micro_batch_size 和 Gradient_accumulation_steps 的影响**：有效 Batch Size 至关重要；如果它发生变化，请相应调整学习率。通常做法是根据 [Hugging Face](https://github.com/huggingface/accelerate/tree/main/docs/source/concept_guides/performance.md#L75L99) 的指南，随 Batch Size 的变化线性缩放学习率。
- **根据 GPU 数量进行调整**：由于有效 Batch Size 的增长，GPU 数量的增加应配合学习率的比例增加。该指南有助于实现训练的稳定性和效率。
- **寻求对不一致之处的澄清**：一位用户指出 Phorm 在关于 `gradient_accumulation_steps` 和有效 Batch Size 的初始建议中存在不一致。他们要求确认正确的方法并提供可靠的来源。

**提到的链接**：

- [accelerate/docs/source/concept_guides/performance.md at main · huggingface/accelerate](https://github.com/huggingface/accelerate/tree/main/docs/source/concept_guides/performance.md#L75L99)): 🚀 一种在几乎任何设备和分布式配置上启动、训练和使用 PyTorch 模型的简单方法，支持自动混合精度（包括 fp8），以及易于配置的 FSDP 和 DeepSpeed 支持……
- [OpenAccess-AI-Collective/axolotl | Phorm AI Code Search](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=e6d469bd-1d57-4032-9b13-8449594bde81)): 更快地理解代码。

---

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1248945184905891852) (20 messages🔥):

- **George Hotz 评价 PyTorch 的 fuse_attention.py**：George Hotz 分享了 [PyTorch 的 fuse_attention.py 链接](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/fuse_attention.py)，评论称它“比 UPat 更好”，但有点过于冗长。他正在考虑实现 UPat 中更高级符号化（symbolic）功能的语法。
- **图模式匹配器与开发项目**：George Hotz 正在寻找关于图模式匹配器（graph pattern matchers）的文献，并建议将提高 tinygrad 中模式匹配器的速度作为一个入门项目。这项任务是可行的，且可以通过过程重放测试（process replay testing）验证正确性。
- **关于 UOp 中 U 的讨论**：简要讨论了 “UOp” 中 “U” 的含义，Hotz 澄清它代表 “micro op”（微操作）。
- **Code Europe 准备工作及幻灯片讨论**：George Hotz 提到他将参加 Code Europe，并乐意谈论 tinygrad。还有人建议修改他演讲的最后一张幻灯片，以更好地与观众互动。
- **下周一会议议程**：Chenyuy 概述了即将举行的周一会议议程，包括 symbolic uops、过程重放测试和悬赏（bounty）更新等主题。

**提到的链接**：[pytorch/torch/_inductor/fx_passes/fuse_attention.py at main · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/fuse_attention.py): Python 中的张量和动态神经网络，具有强大的 GPU 加速功能 - pytorch/pytorch

---

### **tinygrad (George Hotz) ▷ #**[**learn-tinygrad**](https://discord.com/channels/1068976834382925865/1070745817025106080/1249302705382363146) (4 messages):

- **HSA 已弃用；HIP 和 OpenCL 是替代方案**：George 提到 *“HSA 已经不存在了”*，并建议使用最小化的 HIP 集成或 OpenCL 进行正确性检查。他确认所有 **RDNA3 GPU 应该都能配合** tinygrad 工作。
- **明确最低规格要求**：George 澄清说，AMD GPU 的**最低规格是 RDNA**，Nvidia GPU 是 **2080**（第一款带有 GSP 的 GPU）。他表示如果不需要太多改动，愿意添加对 RDNA/RDNA2/CDNA 的支持。
- **Vega20 GPU 并非优先级**：尽管在后挖矿时代曾一度流行，但像 **Vega20 这样的前 RDNA 架构 GPU** 并不被视为 tinygrad 支持的严肃目标，原因是它们的显存和性能有限。jewnex 评论道：*“理论规格还可以，但对于任何机器学习来说，显存量都相当糟糕（16 GB）”*。

---

### **AI Stack Devs (Yoko Li) ▷ #**[**app-showcase**](https://discord.com/channels/1122748573000409160/1122748840819306598/) (1 messages):

mikhail_ee: 嘿！[http://hexagen.world](http://hexagen.world) 的作者在这里 🙂 感谢分享！

---

### **AI Stack Devs (Yoko Li) ▷ #**[**ai-town-discuss**](https://discord.com/channels/1122748573000409160/1132926337598902293/1248907172042117123) (16 messages🔥):

- **Convex 架构解释了独特的游戏循环**：游戏循环在 Convex 中运行，作为一系列短生命周期的 serverless 函数执行。这与传统游戏不同，传统游戏的状态保存在内存中，且所有内容都在同一台机器上运行。
- **通过分布式函数实现可扩展性**：这种设置允许处理大量的 API 调用和输入，因为游戏的各个方面都由独立的 serverless 函数调用管理。这意味着 Convex 的后端可以高效扩展。
- **客户端通过 websocket 订阅进行更新**：客户端通过 websocket 订阅查询，每当发生更改时都会收到推送更新。这是 Convex 提供的有益特性之一。
- **多人场景的挑战**：由于玩家之间的网络延迟各不相同，竞技性玩法并非最佳。这一点被强调是为了解释多用户环境中实时交互的局限性。
- **深入研究 AI Town 架构**：对于计算机科学研究，建议查看 [AI Town 的架构文档](https://github.com/a16z-infra/ai-town/blob/main/ARCHITECTURE.md) 以全面了解其内部运作机制。

**提到的链接**：[ai-town/ARCHITECTURE.md at main · a16z-infra/ai-town](https://github.com/a16z-infra/ai-town/blob/main/ARCHITECTURE.md)：一个采用 MIT 许可、可部署的入门套件，用于构建和定制你自己的 AI town 版本——一个 AI 角色居住、聊天和社交的虚拟城镇。- a16z-infra/ai-town

---

### **AI Stack Devs (Yoko Li) ▷ #**[**ai-town-dev**](https://discord.com/channels/1122748573000409160/1137456826733047908/1249021691007340688) (2 messages):

- **Convex.json 配置文件丢失**：一位用户表示难以找到 **convex.json** 配置文件，暗示文件结构中可能存在困惑或误放。
- **Convex 后端错误问题**：在尝试运行 Convex 后端时，出现错误提示："Recipe `convex` could not be run because just could not find the shell: program not found"，暗示缺少依赖项或配置错误。

---

### **AI21 Labs (Jamba) ▷ #**[**jamba**](https://discord.com/channels/874538902696914944/1222916247063232553/1249178257924821014) (7 messages):

- **Agentic 架构可能会“掩盖”问题，而不是解决问题**：一位成员询问将复杂任务分解为简单任务的 "Agentic 架构" 是否能解决固有局限性。另一位成员指出，尽管 *Theorem 2* 表明可能存在缓解作用，但它最终无法解决更深层次的问题。
- **模型架构的局限性**：在针对 RNN, CNN, SSM 和 Transformer 等模型架构局限性的讨论中，有人澄清说，由于结构约束，这些模型难以进行真正的推理，正如 *Theorem 1* 所强调的那样。
- **需要更深入的理解**：一位成员计划重新阅读论文，以充分掌握所讨论的概念，特别是关于当前架构的局限性和通信复杂度（communication complexity）问题。
- **Theorem 1 与通信复杂度**：一位参与者总结了他们对 *Theorem 1* 的理解，解释说它涉及三个 Agent 的函数组合问题，并强调了多次通信对于正确计算的必要性。这些交互有时会导致 Agent 产生幻觉结果。

---

### **Datasette - LLM (@SimonW) ▷ #**[**ai**](https://discord.com/channels/823971286308356157/1097032579812687943/1248911819859427420) (1 messages):

- **参与排行榜**：一位成员推测某个版本的发布是为了支持研究和排行榜参与。他们评论道：*"我认为他们以这种形式发布它是为了让人们可以研究它，并且他们可以参与排行榜。"*

---

### **Datasette - LLM (@SimonW) ▷ #**[**llm**](https://discord.com/channels/823971286308356157/1128504153841336370/1248911421895606356) (4 messages):

- **UMAP 在聚类方面表现出色**：一位用户赞扬了该工具的能力，感叹道：“UMAP 在聚类方面非常惊人。”他们建议通过采访 UMAP 创作者来了解更多细节。
- **与 UMAP 创作者一起深入了解 UMAP**：[Vincent Warmerdam 分享了一段 YouTube 视频](https://www.youtube.com/watch?v=U_0GcpkjNVQ)，题为“与 UMAP 创作者 Leland McInnes 一起迈向 KDearestNeighbors”。这段视频深入探讨了 UMAP、PyNNDescent 和 HDBScan 的细微差别，并包含了 Leland McInnes 本人的见解。

**提到的链接**：[Moving towards KDearestNeighbors with Leland McInnes - creator of UMAP](https://www.youtube.com/watch?v=U_0GcpkjNVQ)：Leland McInnes 以开发众多软件包而闻名。除了 UMAP，还有 PyNNDescent 和 HDBScan。最近他还在致力于开发帮助可视化聚类的工具...

---

### **Torchtune ▷ #**[**general**](https://discord.com/channels/1216353675241590815/1216353675744641096/1249843482864914453) (2 messages):

- **关于 TRL KL 图表的随机咨询**：一位成员询问在 DPO 实现实验期间是否使用了来自 TRL 的 KL 图表。回复指出没有使用，但为感兴趣的人参考了 [TRL 的 PPO trainer 中的 KL 图表](https://github.com/huggingface/trl/blob/34ebc4ccaf376c862a081ff4bb0b7e502b17b2fb/trl/trainer/ppo_trainer.py#L1133)。

**提到的链接**：

- [trl/trl/trainer/ppo_trainer.py at 34ebc4ccaf376c862a081ff4bb0b7e502b17b2fb · huggingface/trl](https://github.com/huggingface/trl/blob/34ebc4ccaf376c862a081ff4bb0b7e502b17b2fb/trl/trainer/ppo_trainer.py#L1133))：使用强化学习训练 Transformer 语言模型。 - huggingface/trl
- [DPO by yechenzhi · Pull Request #645 · pytorch/torchtune](https://github.com/pytorch/torchtune/pull/645#issuecomment-2041861215)？：将 DPO 集成到 Torchtune 的上下文，更多细节见此处的 Changelog ... 测试计划 ....

---

### **DiscoResearch ▷ #**[**disco_judge**](https://discord.com/channels/1178995845727785010/1178996063537991752/1249448152453349489) (1 messages):

- **Lighteval 在处理 bitsandbytes 模型时遇到困难**：一位成员寻求帮助，尝试使用提供的命令在 Lighteval 上评估 bitsandbytes 模型。尝试失败了，因为 Lighteval 无法识别 bitsandbytes 方法，而是要求 GPTQ 数据。

---

### **DiscoResearch ▷ #**[**discolm_german**](https://discord.com/channels/1178995845727785010/1197630242815213618/1249640751915335760) (1 messages):

- **文档打包（Document Packing）讨论**：一位成员询问了 model card 中关于文档打包的代码，质疑这是一个朴素实现还是实际使用的实现。他们还寻求关于 `tokenized_documents` 数据类型的澄清，并提到他们需要一种处理大数据的有效解决方案。

---

### **MLOps @Chipro ▷ #**[**events**](https://discord.com/channels/814557108065534033/869270934773727272/1249872448506171392) (1 messages):

- **在 Mosaic 活动中与 Chip Huyen 见面**：Chip Huyen 宣布她今晚将参加 Databricks 峰会的 Mosaic 活动，并鼓励其他参加者去打个招呼。有关该活动的更多详情可以在[这里](https://mosaicx.events/events/june-10-2024-san-francisco-ca?events=main)找到。

**提到的链接**：[Events | June 10, 2024 San Francisco, CA](https://mosaicx.events/events/june-10-2024-san-francisco-ca?events=main)：未找到描述

---

---

---

{% else %}

> 完整的频道细分内容已因邮件长度而截断。
> 
> 如果您想查看完整细分，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
> 
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}