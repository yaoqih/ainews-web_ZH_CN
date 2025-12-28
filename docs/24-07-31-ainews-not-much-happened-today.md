---
companies:
- meta-ai-fair
- google-deepmind
- scale-ai
- apple
- canva
- hugging-face
date: '2024-07-31T07:04:15.405372Z'
description: '以下是翻译内容：


  **Meta** 发布了 **SAM 2**，这是一个用于实时物体分割的统一模型，并配备了一个比以往大 4.5 倍、标注量多 53 倍的新数据集。由 **Jeremy
  Howard** 开发的新 Python Web 框架 **FastHTML** 问世，可实现交互式 Web 应用的轻松创建与部署。**Scale AI** 推出了关于对抗鲁棒性的
  SEAL 排行榜，由 **Google DeepMind** 的 **Gemini 1.5 Pro** 摘得桂冠。**苹果 (Apple)** 发布了一份关于其用于端侧和服务器的智能基础语言模型（Intelligence
  Foundation Language Models）的技术报告。**Yann LeCun** 与 Martin Casado 和 Ion Stoica 共同撰文，强调了开源
  AI 的重要性。**Maarten Grootendorst** 撰写的关于高效 LLM 推理的《量化视觉指南》（Visual Guide to Quantization）在网络上走红。**ChatGPT**
  开始向部分用户推出先进的语音和视觉模式。**Leonardo AI** 被 **Canva** 收购。**Jim Fan** 分享了关于 Project Groot
  如何增强机器人人类演示数据的见解。**Midjourney v6.1** 正式发布。'
id: f6ce07b9-cb0b-4d23-8996-04827151f6a2
models:
- sam-2
- gemini-1.5-pro
- chatgpt
- midjourney-v6.1
original_slug: ainews-to-be-named-5098
people:
- jeremyphoward
- demis-hassabis
- ylecun
- maartengrootendorst
- jimfan
title: 今天没发生什么特别的事。
topics:
- object-segmentation
- quantization
- web-development-framework
- adversarial-robustness
- on-device-ai
- open-source
- robotics
- voice
- vision
---

<!-- buttondown-editor-mode: plaintext -->**这是一个平静的一天。**

> 2024年7月29日至7月30日的 AI 新闻。我们为您检查了 7 个 Reddit 社区、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **28** 个 Discord 服务区（**248** 个频道，**2257** 条消息）。预计节省阅读时间（按 200wpm 计算）：**262 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

一些小项目：

- maartengrootendorst 的 [《量化视觉指南》(Visual Guide to Quantization)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) 走红，
- ChatGPT 的高级语音模式 [开始向一小部分用户推送](https://x.com/EthanSutin/status/1818439026401329645) —— 有些人甚至获得了 [启用视觉的版本](https://x.com/manuvision/status/1818412120373182928?s=46)
- [Leonardo AI 被 Canva 收购]( https://x.com/ethan_smith_20/status/1818152222326186260?s=46)
- [Jim Fan 分享了 Project Groot 如何为他们的机器人增强人类演示数据](https://x.com/drjimfan/status/1818302152982343983?s=46)
- [Midjourney v6.1 发布](https://x.com/midjourney/status/1818342703618482265)

我们录制了一个高级语音模式的演示，非常有趣，将在下一期 LS 播客中发布。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

> 所有回顾均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**Meta 发布用于对象分割的 SAM 2**

- [@AIatMeta](https://twitter.com/AIatMeta/status/1818055906179105010) 宣布发布 Meta Segment Anything Model 2 (SAM 2)，这是一个用于图像和视频中实时、可提示对象分割的统一模型。SAM 2 以 Apache 2.0 许可证发布。

- 该模型附带一个新的 SA-V 数据集，比现有的最大视频分割数据集 [大 4.5 倍，标注量多约 53 倍](https://twitter.com/AIatMeta/status/1818055908070773078)。

- SAM 2 可以 [开箱即用地应用于各种现实世界的用例](https://twitter.com/AIatMeta/status/1818055909760975134)。Meta 提供了体验 Demo 和访问代码的链接。

**全新 Web 开发框架：FastHTML**

- [@jeremyphoward](https://twitter.com/jeremyphoward/status/1818036923304456492) 宣布了 FastHTML，这是一种在 Python 中创建现代交互式 Web 应用的新方法。它可以从简单的 6 行代码应用扩展到复杂的生产系统。

- FastHTML 集成了身份验证、数据库、缓存、样式等功能。它提供 [一键部署到 Railway、Vercel 和 Hugging Face 等平台](https://twitter.com/jeremyphoward/status/1818036926827610423) 的功能。

- 该框架旨在通过利用 Web 基础知识而非复杂的框架，[让 Web 编程变得更简单、更强大](https://twitter.com/jeremyphoward/status/1818036930657009888)。

- Jeremy 制作了一个 [1 小时的 FastHTML 迷你课程](https://twitter.com/jeremyphoward/status/1818036938932371605)，展示了如何使用纯 Python 从头开始创建并部署一个完整的交互式 Web 应用。

**AI 模型进展与基准测试**

- [@alexandr_wang](https://twitter.com/alexandr_wang/status/1817956788320530940) 宣布了 Scale 最新的 SEAL 对抗鲁棒性排行榜 (SEAL Leaderboard on Adversarial Robustness)，重点关注具有透明评估方法的通用危害场景。

- [@demishassabis](https://twitter.com/demishassabis/status/1818049561421910345) 强调 Gemini 1.5 Pro 在新的 Scale AI 对抗鲁棒性排行榜中名列前茅。

- Apple 发布了一份关于其 [Intelligence Foundation Language Models 的技术报告](https://twitter.com/awnihannun/status/1817989760729891296)，详细介绍了其端侧和服务器模型的架构及训练过程。

**开源 AI 与算力资源**

- [@ylecun](https://twitter.com/ylecun/status/1818044278029128046) 分享了《经济学人》上一篇关于开源 AI 重要性的文章，该文章由 Martin Casado 和加州大学伯克利分校教授 Ion Stoica 共同撰写。

- 讨论了 [用于 AI 开发的 GPU 资源的可用性和定价](https://twitter.com/far__el/status/1817965343702401363)，一些人注意到可用性有所增加，且需求可能正在下降。

---

# AI Reddit 回顾

## /r/LocalLlama 综述

**主题 1. 高效 LLM 推理的量化技术进展**

- **[A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)** ([Score: 332, Comments: 37](https://reddit.com//r/LocalLLaMA/comments/1eeyab4/a_visual_guide_to_quantization/)): 该文章介绍了 **"A Visual Guide to Quantization"**，对用于减小 **Large Language Models (LLMs)** 大小和计算需求的各种 **quantization techniques** 进行了全面概述。它涵盖了 **INT8**、**INT4** 和 **binary quantization** 等方法，解释了它们的原理以及在模型尺寸减小与性能影响之间的权衡，同时还讨论了 **vector quantization** 和 **mixed-precision quantization** 等高级技术。
  - 作者 **MaartenGr** 解释了创建该视觉指南的初衷，强调了随着更多 **LLMs** 的发布，对 **quantization** 的需求日益增长。该指南涵盖了从基础数值表示到 **GPTQ**、**GGUF** 和 **BitNet** 等高级方法的各种技术。
  - 该指南包含 **60 多个自定义视觉图表**，以增强直观性，使初学者和资深读者都能轻松理解 quantization 技术。它涵盖了 **(a)symmetric quantization**、**dynamic/static quantization** 以及 **quantization-aware training** 等主题。
  - 一位读者称赞该指南是他们见过的 *“关于 quantization 最好的文章之一”*，强调了其卓越的质量和对该主题的全面覆盖。

- **Llama 3.1 405B EXL2 quant results** ([Score: 75, Comments: 31](https://reddit.com//r/LocalLLaMA/comments/1efg2wv/llama_31_405b_exl2_quant_results/)): **Llama 3.1 405B** 模型使用 **EXL2** 进行了针对 GPU 使用的量化，结果显示在 **125-150GB** 的模型尺寸范围内，原始的 EXL2 quantization 性能优于 Meta 蒸馏后的 70B 模型。与 70B 版本和商业 **LLMs** 相比，**405B** 模型在长上下文问答、事实分析和详细故事理解方面表现出更优越的性能，并在接近其 **128K context limit** 时保持一致性。尽管 benchmark 表明 70B 和 405B 模型的性能相似，但后者在实际任务中表现出色，仅在文本中出现多个相似示例时才会遇到困难。
  - **Llama 3.1 405B** 模型的性能随 quantization 级别而变化。在 **2.5bpw** (123GB) 时，它在短上下文中表现连贯，但在超过 **4K tokens** 后表现吃力。在 **3bpw** 时，它能保持连贯性直到 **12K tokens**。
  - 该模型的长上下文性能可能源于 **more MLP params**、**bigger embedding dim**、**more attention layers** 或 **raw training compute**。在 **128K context** 下，**Llama 3.1 70B** 的表现优于内部微调的 Llama 2 和 3 70B。
  - 用户将 **Llama 3.1 405B** 与 **Claude-3.5-Sonnet** 和 **GPT-4** 进行了比较，指出输入成本相似（$3/M），但强调了 Llama 在 finetuning 能力方面的优势。一些人对与 **Mistral Large 2** 和 **DeepSeek-v2-coder** 的比较表示感兴趣。


**Theme 2. Meta 的开源 AI 贡献与影响**

- **[Segment Anything 2 (Meta)](https://github.com/facebookresearch/segment-anything-2)** ([Score: 107, Comments: 7](https://reddit.com//r/LocalLLaMA/comments/1effb4z/segment_anything_2_meta/)): Meta 发布了 **Segment Anything 2 (SA-2)**，这是其图像分割模型的升级版本。SA-2 提供了**改进的性能**，包括对图像和视频中 **3D 对象**进行分割的能力，并能处理高达 **3000x3000 像素**的**高分辨率输入**。该模型还引入了诸如**文本提示 (text prompting)** 和**多模态提示 (multi-modal prompting)** 等新功能，从而实现更灵活、更精确的分割任务。
  - 用户称赞了 **SA-2 的性能**，有人在随机视频上进行了测试，并报告其运行“完美无缺”。其 [Web 演示](https://sam2.metademolab.com/demo) 被描述为“令人惊叹”，特别是它在视频片段中追踪球的能力。
  - 讨论集中在潜在应用上，包括将 SA-2 应用于 **3D 模型**以解决 3D 人体建模中的“无用斑块 (useless blobs)”问题，以及对视频分割中“追踪一切 (Track anything)”能力的推测。
  - 一些用户质疑鉴于 SA-2 的能力，分割问题现在是否已“完全解决”，而另一些人则称赞 **Meta** 和 **Zuckerberg** 对 AI 发展的开源贡献。
- **如果 Meta 开源他们的图像模型会怎样？影响可能是巨大的！** ([Score: 76, Comments: 41](https://reddit.com//r/LocalLLaMA/comments/1efmvf2/what_if_meta_opensources_their_image_model_the/)): Meta 的 AI 图像生成器 **Emu** 在 **11 亿张图像**上进行了训练，并展示了令人印象深刻的速度和质量。虽然尚未公开，但人们猜测它可能会像 Meta 的 **Llama 模型**一样开源，这可能是 AI 图像生成领域的一个重大进展。如果发布，它将为 **Stable Diffusion** 等现有工具提供一种全新的替代方案，可能允许用户在个人电脑上运行图像生成模型。
  - **开源 Meta 的图像模型**可能会推动适用于各种设备的更小、更高效版本的开发。虽然在本地达到 **DALL-E** 或 **MidJourney** 的水平可能具有挑战性，但在高端智能手机上已经可以实现原型设计和物体移除等更简单的任务。
  - 图像生成模型正在影响各行各业，**Activision Blizzard** 已批准将 **Midjourney** 和 **Stable Diffusion** 用于概念艺术和营销。**Klarna** 报告称，使用 **genAI 工具**节省了 600 万美元的图像制作成本，且 90% 的员工将 AI 整合到了日常工作流中。
  - 最近几个月涌现了大量新的图像生成模型，包括 **Kolors**、**SD3**、**Aura**、**Flow**、**Lumia**、**Hunyuan** 和 **Pixart**。这些模型在营销、视频游戏开发和平面设计中都有应用，仅美国平面设计市场价值就约为 **140 亿美元**。


**主题 3. 近期发布的 LLM 性能对比**

- **Mistral NeMo vs Llama3.1 8B** ([Score: 74, Comments: 32](https://reddit.com//r/LocalLLaMA/comments/1eeuo9s/mistral_nemo_vs_llama31_8b/)): 该帖子询问了 **Llama3.1 8B** 和 **Mistral NeMo (12B)** 模型之间的比较，特别关注它们的**多语言能力**。作者对 **Mistral NeMo 充满前景的性能**表示感兴趣，但寻求关于它是否优于 **Llama3.1 8B** 的确认，并请求分享个人经验和基准测试讨论。
  - **Mistral NeMo** 被认为更“聪明”，可与 **Llama3 70B** 媲美，而 **Llama3.1 8B** 在自然语调、风格和创造力方面表现出色。用户建议 **Nemo** 更适合代码和 **function calling**，而 **Llama** 更适合聊天机器人。
  - **Gemma 2 9B** 被认为是这两个模型的强力竞争者，特别是在不需要长上下文的任务中。用户推测，具有改进上下文处理能力的潜在 **Gemma 2.1** 可能会超越 **Llama 3.1** 和 **Mistral Nemo**。
  - 用户指出 **Mistral NeMo** 的内置审查较少，且对提示词的接受度较高，建议在创意写作时将 **temperature 设置在 0.5-1 之间**。官方模型卡宣称其优于“更小或类似”的模型，这一说法被批评为门槛设定过低。

- **Llama 3.1 405B EXL2 量化结果** ([Score: 75, Comments: 31](https://reddit.com//r/LocalLLaMA/comments/1efg2wv/llama_31_405b_exl2_quant_results/))：该帖子比较了 **Llama 3.1 405B** 和 **70B** 模型在**长上下文任务**中的表现，重点关注了用于 GPU 的 405B 模型的 **EXL2 量化**。作者指出，在 **125-150GB 模型大小范围**内，原始 EXL2 量化在**困惑度 (PPL)** 方面优于 Meta 蒸馏后的 70B 模型。尽管基准测试显示两者性能相似，但作者的测试表明，在涉及**长上下文问答**、**事实分析**和**故事细节记忆**的任务中，**405B 模型显著优于 70B 模型**以及 **GPT-4** 和 **Claude Sonnet 3.5** 等闭源 LLM，尤其是在接近 **128K 上下文限制**时。
  - **Llama 3.1 405B** 模型在**长上下文任务**中优于 **70B**，但 405B 的 **2.5bpw 量化**在超过 **4K tokens** 后表现吃力，而 **3bpw** 则能维持到约 **12K tokens**。作者建议这值得进一步调查。
  - 讨论集中在比较不同的**量化级别**和模型大小，并关注 **405B 模型**与 **fp16 70B** 以及 **DeepSeek MoE 模型**的对比。作者指出，**原始算力**和**训练时长**可能是性能提升的原因。
  - 用户对与 **Mistral Large 2** 及其他模型在复杂任务和长上下文使用中的对比表示感兴趣。作者正在努力从内部数据集中提取**开放测试基准**，以便进行更客观的比较。


**主题 4. 本地 LLM 推理的硬件与效率考量**

- **新的 DDR6 是否开启了 CPU 驱动 LLM 的时代？** ([Score: 97, Comments: 87](https://reddit.com//r/LocalLLaMA/comments/1ef0qpb/is_the_new_ddr6_the_era_of_cpupowered_llms/))：据报道，即将推出的 **DDR6 RAM** 标准在超频模式下可能达到高达 **17,000 MHz** 的频率，这引发了关于其对 **CPU 驱动 LLM** 影响的猜测。该帖子质疑这一进步是否能让语言模型完全在 CPU 上运行，从而可能减少此类任务对 GPU 的依赖。

- **你认为 Llama3 405B 能盈利吗？** ([Score: 150, Comments: 102](https://reddit.com//r/LocalLLaMA/comments/1eewqtv/do_you_think_llama3_405b_can_be_profitable/))：该帖子讨论了 **Llama3 405B API** 的**盈利挑战**，引用了 **Jia** 在 Twitter 上关于该话题的讨论。作者提到一位在**云服务公司**工作的朋友，该公司最近推出了该 API，正努力在盈利和客户接受度之间寻找**定价平衡**。
    - **Avianio** 声称以**每百万 tokens 5 美元**的价格托管 **Llama 3 405B** 是可以盈利的，而另一位用户则认为现实的 **H100 SXM** 价格（<$2.5/gpu/hr）使得大多数公司在 405B 和 70B 模型上都能盈利。
    - 提供开源模型的市场被描述为高度**商品化**，面临差异化挑战。像 **OpenAI**、**Anthropic** 和 **Mistral** 这样的公司依靠专有或独家授权的模型来收取溢价。
    - **Meta** 的开源策略被视为试图**削减** **OpenAI** 等潜在竞争对手的利润。一些用户质疑 405B 模型的设计选择，认为 70B 版本对于大多数客户需求是更具成本效益的替代方案。

## 所有 AI Reddit 综述

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

待完成

---

# AI Discord 综述

> 摘要之摘要的摘要

## Claude 3.5 Sonnet


**1. LLM 进展与基准测试**

- **Llama 3.1 以多语言能力令人印象深刻**: Meta 的 **[Llama 3.1](https://x.com/reach_vb/status/1815767864277606762)** 已经发布，模型参数高达 **405B**，在 **MMLU 基准测试中获得了 85.2 分**，并支持 **128K context**。
   - 该模型采用了更宽松的许可证，允许在其输出上训练其他 LLM，使其成为 **GPT-4** 和 **Claude** 的强力竞争对手。用户反馈褒贬不一，一些人称赞其性能，而另一些人则遇到了循环响应等问题。
- **Apple 的 AI 模型展现出潜力**: Apple 的新 AI 论文揭示了其服务器端和设备端模型的显著基准测试结果，设备端模型的 **MMLU 分数为 61.4**，服务器模型为 **75.4**。
   - 论文详细介绍了两阶段预训练过程以及 SFT 和 RLHF 方法。值得注意的是，Apple 表示他们没有使用 **NVIDIA GPU** 进行 AI 模型训练，而是选择了 **TPU**，这使他们成为行业内第二大 TPU 用户。
  


**2. 模型优化与性能调优**

- **量化技术受到关注**: 一份 [量化视觉指南](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) 强调了 Large Language Models (LLMs) 的参数通常超过数十亿，这使得它们难以在消费级硬件上运行。

**3. 开源 AI 发展**

- **SWE-Bench Ultra-Hackathon 挑战极限**: 为期 **6 天的 SWE-Bench 超级黑客松** 正在举办，旨在挑战开源代码生成的极限，参与者将获得来自 StrongCompute 提供的 $1,000 计算资源。
   - 此次活动邀请了包括 [John Yang](https://x.com/jyangballin)、[Carlos E. Jimenez](https://x.com/_carlosejimenez) 和 [Ofir Press](https://x.com/OfirPress) 在内的共同作者进行演讲，旨在提升开源代码生成能力并激发社区的创新方法。
- **SAM 2 增强分割能力**: Meta 发布了 **[Segment Anything Model 2 (SAM 2)](https://ai.meta.com/blog/segment-anything-2/)**，提供图像和视频中的实时可提示对象分割，相比其前身有显著改进。
   - SAM 2 在包含 50,000 个视频的新 SA-V 数据集上进行训练，并采用了全新的 memory attention 技术。[GitHub 仓库](https://github.com/facebookresearch/segment-anything-2) 提供了运行推理的代码、训练好的模型权重以及用于各种分割任务的示例 notebook。
  
**4. AI 行业新闻与合作伙伴关系**

- **Perplexity 推出出版商计划**: Perplexity 宣布了其 **[Publishers Program](https://pplx.ai/publishers)**，与 TIME、Der Spiegel 和 Fortune 等主要机构合作，以确保获取可靠信息并支持出版商。
   - 该计划旨在提供新技术以吸引受众并促进共同成功，并计划在未来几个月内引入 **收入共享** 模式，首先从相关问题的广告开始。
- **Leonardo AI 加入 Canva 大家庭**: [Leonardo.Ai](https://x.com/ethan_smith_20/status/1818152222326186260?s=46) 宣布被 Canva 收购，预计这将增强创意工具并以新方式赋能创作者。
   - 此次整合旨在加速创新并基于 Phoenix 等现有项目进行构建，有可能重塑 AI 驱动的设计工具和创意工作流的格局。
  

---

# PART 1: High level Discord summaries

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Llama 3.1 的多语言功能令人印象深刻**：[Llama 3.1](https://x.com/reach_vb/status/1815767864277606762) 支持拥有 **405B** 参数的模型，并在 **128K context** 下于 MMLU 基准测试中达到 **85.2** 分。
   - 该版本采用了宽松的许可证，允许在其输出上进行训练，使其成为 **GPT4o** 和 **Claude** 的强力竞争对手。
- **Argilla 2.0 推出数据集复制功能**：Argilla 2.0 即将发布的版本包含一项易于的数据集复制功能，旨在提高工作流效率。
   - 该[公告](https://x.com/argilla_io/status/1817945202432061792)受到了社区的积极响应，帮助用户无缝管理多个数据集。
- **PEFT v0.12.0 引入新方法**：[PEFT v0.12.0](https://x.com/julien_c/status/1817837045298978986) 展示了 **OLoRA** 和 **X-LoRA** 等方法，旨在增强模型训练效率。
   - 这些方法对于提高训练期间的性能和资源分配至关重要。
- **在图像生成领域达到 SOTA**：一位成员宣布在图像生成能力上达到了 SOTA，并强调了该领域的进展。
   - 他们分享了[这条推文](https://twitter.com/DataPlusEngine/status/1818358813520441493)作为成就证明，并讨论了图像生成技术的进一步发展。
- **探索语言模型中的量化技术**：[一份视觉指南](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)强调了量化技术对于在消费级硬件上优化 LLM 的重要性。
   - 重点在于创建更小、更高效的模型，以解决与尺寸相关的挑战。



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **升级后出现模型加载问题**：用户报告在升级到 **0.2.29** 版本后出现 GPU 加速失效，表明更新过程中可能存在损坏。
   - 一位用户建议清除应用数据并重新安装 **0.2.28** 版本，而其他用户则强调 **Llama 3.1** 需要 **0.2.29** 版本才能获得最佳性能。
- **Llama 3.1 出现意外的循环响应**：一位用户在升级 LM Studio 后遇到了 **Llama 3.1 8B model** 持续循环响应的问题，建议改用 Llama v2 预设。
   - 这一问题凸显了深入理解提示词格式（prompt formatting）的必要性，以避免 AI 响应中出现此类行为。
- **AI 开发入门资源**：一位寻求进入 AI 开发领域的新用户被引导学习 **Python** 和 **PyTorch**，将其作为核心基础工具。
   - 建议利用 **YouTube** 等平台上的免费资源来帮助理解 AI 涉及的概念。
- **GPU 兼容性问题凸显**：成员们指出 LM Studio 不支持 **Intel Iris Xe Graphics**，必须使用支持 **CUDA** 的 NVIDIA 或支持 **ROCm** 的 AMD 才能正常运行。
   - 讨论了 **Tesla P40** 的性能，指出与当代消费级 GPU 相比，它面临兼容性和速度问题。
- **LM Studio 0.2.29 版本现已支持 ROCm**：关于 LM Studio 0.2.29 在 ROCm 上发布的查询得到了答复，根据 [GitHub release notes](https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md) 确认已可用。
   - 成员们表达了在自己的配置中使用该更新提供的新功能的渴望。



---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 出版商计划发布**：Perplexity 推出了其**出版商计划 (publishers' program)**，与 [TIME](https://pplx.ai/publishers) 和 [Der Spiegel](https://pplx.ai/publishers) 等机构合作，以增强内容溯源。
   - 该计划旨在维持由 [The Texas Tribune](https://pplx.ai/publishers) 等信任源支持的**高质量回答**，同时计划实施**收入共享**模式。
- **Llama-3 模型出现幻觉**：用户报告 **llama-3-sonar-large-32k-online** 模型近期出现了产生幻觉信息的问题。
   - 针对 **Llama 模型**将于 2024 年 8 月 12 日弃用的消息，用户因其可靠性日益下降而表达了担忧。
- **特斯拉充电站警报**：特斯拉发布了关于充电站兼容性的警告，引起了依赖 **Supercharging** 的用户关注。
   - 这一公告引发了关于特斯拉基础设施在长途旅行中可靠性的疑问。
- **AI 模型对比分析**：用户讨论了 **Claude 3.5 Sonnet** 和 **GPT-4o** 的性能对比，强调了它们在各项任务中的各自优势。
   - 虽然 **Claude** 提供了不错的输出，但 **GPT-4o** 在准确性方面受到称赞，尤其是在编程应用中。
- **太空军扩大卫星网络**：**太空军 (Space Force)** 计划扩大其卫星网络，以增强国家安全和通信能力。
   - 这一公告引发了关于轨道上**军事卫星**增加所带来影响的辩论。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Artisan 引入新命令 /style**：**/style** 命令现在允许用户根据指定的风格生成图像，例如**梵高风格的猫**或**日式风格的飞船**。
   - 鼓励成员尝试此功能，并分享了展示其创意潜力的示例。
- **在 Stable Diffusion 中遇到 OutOfMemoryError**：用户在使用 **SD1.5 模型**生成图像时，即使使用 8GB GPU 也会遇到 **OutOfMemoryError**，引发了故障排除讨论。
   - 建议包括更改 CUDA 设置和增加虚拟内存以缓解这些问题。
- **AI 角色一致性难题**：一位用户详细描述了使用 **IP Adapter** 和 **ControlNet** 等工具训练模型以实现一致角色生成的挑战。
   - 他们分享了当前的设置，并寻求进一步的改进方案以获得更可靠的结果。
- **探索 AI 动画工具**：围绕各种 **AI 动画工具**展开了讨论，特别是从静态图像生成极简动画，重点关注 **Live Portrait AI**。
   - 一些人指出 **Runway** 等工具存在质量下降的问题，引发了关于不同任务最佳软件的辩论。
- **引入用于视频分割的 SAM 2**：来自 Meta 的新 **SAM 2 模型**承诺增强静态图像和视频的对象分割，为实时应用铺平道路。
   - 其强大的 Zero-shot 性能可能为动画重混等创意任务带来益处。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 在 Windows 上运行困难**：用户报告在 Windows 上使用 **Unsloth** 时遇到 "No triton module" 错误，并建议切换到 WSL 作为权宜之计。
   - 一位用户幽默地提到，由于游戏偏好，他拒绝从 Windows 切换。
- **模型微调的挑战**：关于微调 **Llama3** 模型的讨论集中在避免灾难性遗忘，引出了合并数据集进行重新训练的想法。
   - 参与者确认，为了减轻与灾难性遗忘相关的风险，完全重新训练是更可取的。
- **使用自定义 Token 的矩阵表示**：一位用户询问如何为其 **Arc-AGI** 项目使用自定义 Token 表示 **30x30 矩阵**，强调需要更多细节。
   - 另一位成员要求进一步澄清，表示更深入的解释将大有裨益。
- **Unsloth 改进了 Rope Scaling 支持**：最近的一次更新确认，截至两周前，以前缺乏 **Rope Scaling** 支持的旧模型现在已在 Unsloth 中实现了此功能。
   - 成员们对这一新功能表示兴奋，并提到了与此增强相关的 **Phi-3 128k 变体**。
- **创建翻译数据集**：一位用户寻求用于微调英文模型的翻译数据集，考虑使用 **DeepL** 来实现此目的，其他人则建议利用 **Wikipedia** 作为资源。
   - 对话强调了全面数据集在增强模型训练中的重要性。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Randomized SVD 简化了大规模问题**：Randomized SVD 将大规模矩阵问题简化为较小的矩阵，提供关键奇异值和向量的近似值，从而实现高效处理。
   - 该技术对于处理海量数据集非常有用，且不会耗尽计算资源。
- **探索 Optimizer CPU Offload**：成员们讨论了一个提议的 `cpu_offload` 标志，用于将优化器状态移动到 CPU，从而在优化步骤中促进参数传输。
   - 针对优化器步骤的阻塞性质是否会影响与 `torch.compile` 进行交错操作的可行性，人们提出了担忧。
- **为 Jeopardy 微调 Llama 3.1**：一位成员正在使用 **Unsloth** 微调 **Llama 3.1 8B**，并对复杂的配置表示困惑。
   - 他们强调更倾向于使用稳定的 bf16 微调过程，以简化训练流水线。
- **WebGPU API：不仅仅是一个浏览器工具**：WebGPU 作为一个对 **WGSL** 具有浅层编译定义的 API，现在已用于浏览器之外的原生应用。
   - 这包括在 **Rust** 和 **Zig** 中的实现，提升了在各种平台上的可用性。
- **对即将举行的活动的期待升温**：即将举行的 **CUDA MODE IRL** 活动引起了热议，与会者对线下见面表现出极大的热情。
   - 成员们强调了注册的必要性，并确认了关于 GPU 访问和主题演讲录制的细节。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **小模型展现出竞争优势**：最近的一篇论文表明，运行一次 **70B 模型** 与从 **13B 模型** 生成五个输出相比，后者在五个任务中可以产生高达 **15% 的增益**。
   - *这引出了一个问题：当两个模型在相同的预算下运行时会发生什么？* 研究结果强调了 **unit-test setups** 对于选择最佳输出的重要性。
- **对 AI Interpretability 时间线的怀疑**：在私有实践之外获得可靠的数据集之前，**AI Interpretability** 可能还需要几年的时间。
   - 成员们表示，更长的公共数据发布时间线可能会促进更稳健的研究结果。
- **Apple AI 模型基准测试见解**：Apple 的新论文展示了服务器端和端侧模型，其 **MMLU** 分数分别为 **61.4** 和 **75.4**。
   - 调查结果详细介绍了两阶段预训练过程以及 SFT 和 RLHF 方法。
- **探索 Hermes 和 Llama 模型合并技术**：讨论集中在 **Hermes** 模型与 **Llama** 的合并技术上，关于有效合并策略的文章正在撰写中。
   - 成员们辩论了各种技术对兼容性和效率的性能影响。
- **Midjourney V6.1 增强功能**：Midjourney 推出了 **V6.1**，具有改进的图像质量和连贯性，以及新的上采样模型。
   - 此次更新是在社区声称在图像生成方面达到 state-of-the-art 结果之后发布的。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI Voice Mode 开始推出**：**Advanced Voice Mode** 正在向选定的 **ChatGPT Plus** 用户群体推出，支持实时对话和自由插话。
   - 指示已通过电子邮件和移动应用发送，预计在**秋季**开放更广泛的访问权限。
- **成员确认获得 Search GPT 访问权限**：用户确认获得了 [Search GPT](https://link.to.searchgpt) 的访问权限，并对其功能表现出不同程度的信心。
   - 一些人认为它很有帮助，而另一些人则质疑其功能性。
- **对 GPT-4o 功能的期待升温**：围绕 **GPT-4o** 先进的 vision 和 voice 功能的预期发布展开了讨论，成员们暗示可能会在**本月底**发布 Alpha 版本。
   - 这表明了用户对更新和潜在时间线调整的关注。
- **DALL-E Bot 命令问题持续存在**：用户在 DALL-E bot 频道执行 `/draw` 命令时遇到问题，部分用户超过 **20 分钟** 无法创建图像。
   - 社区中出现了沮丧的声音，成员们寻求社区协助以排除故障。
- **对 GPT 在 Function Calls 中性能的担忧**：社区成员对使用 **Function Calls** 时 **GPT-4o** 响应质量的下降发出了警报，认为输出的准确性有所降低。
   - 他们对比了完整 Prompt 和 Function Call 提交之间的性能，注意到了显著的差异。

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere API 曾宕机但现已恢复运行**：成员们报告称 **Cohere API** 暂时宕机，遇到了 **503 error**，但通过 [Cohere status page](https://status.cohere.com/) 确认目前已完全恢复运行。
   - 状态页面目前显示端点（endpoints）的 **正常运行时间（uptime）为 99.67%**，文档的 **正常运行时间为 100%**，增强了用户对系统可靠性的信心。
- **庆祝使用 Cohere API 开发的成功项目**：一位成员自豪地展示了他们使用 **Cohere API** 构建的**梦想项目**，该项目具有天气、时间以及部分可用的新闻等功能，引发了社区的热烈响应。
   - 该项目强调了背景氛围的重要性以及对生产效率至关重要的功能。
- **Connector 响应格式困扰**：讨论透露，在 **Cohere chat API** 中将 **unix timestamps** 作为整数返回会导致问题，而字符串表示形式则运行良好，这促使官方对预期数据类型进行了澄清。
   - 有人提到，虽然支持整数，但在 connector 响应格式中它们会被作为字符串处理。
- **网络研讨会访问咨询**：在错过 **Enterprise Workflow Automation with GenAI** 网络研讨会后，一名成员寻求获取录像，被建议联系 [events@cohere.com](mailto:events@cohere.com) 以快速获取。
   - 这突显了 Cohere 推广的结构化方法，以确保参与者即使错过直播课程仍能获取重要内容。
- **探索 tool usage 与 connectors 的对比**：讨论中注意到，受近期 office hours 见解的启发，社区实践正从 connectors 转向 **tool usage**，这表明了社区实践中的战略转向。
   - 虽然 connectors 保持着独特的功能，但目前没有弃用它们的计划，允许用户在方法上保持灵活性。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 社区会议 #5 回顾**：录制的 [Mojo Community Meeting #5](https://youtu.be/1T-MBC9k99M) 讨论了 **GPU programming** 和 Q&A 环节。参与者寻求更集中的讨论，并为未来的活动提出了现场编程（live coding）环节的建议。
   - 深入探索 **Mojo 能力** 的愿望显而易见，这表明在即将举行的会议中需要增强主题的针对性。
- **Stack-PR 的便捷安装**：**Stack-pr** 现在可以通过 `pipx install stack-pr` 进行安装，方便在 GitHub 上创建堆叠式 Pull Requests。成员们讨论了向 conda-forge 提交 feedstock 以简化此过程。
   - 简化 stack-pr 等新工具的安装路径，反映了增强 Mojo 生态系统易用性的更广泛目标。
- **探索 CSV 读取器功能**：关于 **Mojo CSV reader** 的咨询揭示了其现有功能可以与 Python 的 csv 模块相媲美。讨论强调了社区渴望探索全面功能以增强对 Mojo 的理解。
   - 成员们表示，扩展 **CSV 功能** 可以显著拓宽 Mojo 在数据处理中的适用性。
- **在 Mojo 中实现图像解析**：一位贡献者分享了他们在 Mojo 中成功实现 **PNG parsing** 的经历，并链接到了他们的 [GitHub repository](https://github.com/fnands/mimage)。他们计划下一步处理 JPEG 解析。
   - 社区对图像解析库的热情标志着对扩展 Mojo 多媒体能力的兴趣日益浓厚。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 为用户提供 Office Hours**：LlamaIndex 邀请用户报名参加 [Office Hours](https://docs.google.com/forms/d/e/1FAIpQLSefrnmxQWD-1OhSP51kUKtdbw9EGDjrMLefkZFACKD19TKsuQ/viewform)，讨论关于 Agent 的使用案例并领取品牌周边。
   - 参与者可以期待一次 **15-30 分钟的 Zoom 会话**，以探索 LlamaIndex 如何协助 Agent 应用。
- **GraphRAG 技术结合了多种方法**：来自 Microsoft 的 **GraphRAG** 技术将文本提取、网络分析、Prompting 和摘要集成到一个系统中，通过生成的图谱增强数据理解。
   - 更多详情可以在[这里](https://t.co/ZnDtJ731hl)找到，其应用说明见[这里](https://t.co/mx54Su1gYk)。
- **网络研讨会重新安排在下周四**：根据最近的[更新](https://t.co/Zo9zRz528F)，即将举行的网络研讨会现定于 **下周四 8/8 太平洋时间上午 9 点**。
   - *参与者应相应更新其日历。*
- **RAPTOR Pack 更新讨论**：成员们讨论了将 **RAPTOR** 部署到 Pinecone 等托管 Vector DBs，以及在不重新聚类的情况下管理文档插入。
   - *交流了在不损害先前聚类数据的情况下添加新文档的策略。*
- **从 LLM 输出生成 Mermaid 图表**：成员们分享了从 LLM 输出生成 **Mermaid 图表**的工具，特别是 `mmd` 格式的使用以及推荐用于渲染的 **Mermaid CLI**。
   - *提供了一些有用的示例来演示有效的图表生成，并参考了 [Mermaid 语法](https://mermaid.js.org/intro/syntax-reference.html)。*

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **索引过程中的 Transformers 错误**：多位成员报告在使用 **Transformers** 库时出现断言错误：`srcIndex < srcSelectDimSize`，特别是在 **Mistral** 模型配置中。
   - 提议的修复方案包括 **删除缓存** 并重新下载依赖项以解决此问题。
- **Gemma 2 持续输出 Pad Token**：一位用户遇到了其微调后的 **Gemma 2 9b** 模型在部署到 vLLM 后不断输出 `<pad>` Token 的问题。
   - 讨论指向了配置问题，强调需要验证来自 [Hugging Face](https://huggingface.co/google/gemma-2-9b-it/blob/main/special_tokens_map.json) 的 **Special Tokens**。
- **Chat Template 训练配置变更**：**PR #1756** 的引入要求为 `type: chat_template` 添加 `roles_to_train` 字段，这破坏了现有使用 **chat_template** 的示例。
   - 成员们对需要额外的文档和示例来澄清这一变化表示担忧。
- **聊天机器人的 RAG 实现探索**：一位参与者讨论了使用 **Retrieval Augmented Generation (RAG)** 作为其聊天机器人项目替代微调方案的可能性。
   - 他们打算将精力分配在 RAG 和传统微调之间，旨在实现显著的输出增强。
- **Loss 函数卡在零**：一位用户报告其模型训练 Loss 卡在 **0.0**，且 `grad_norm` 显示为 **nan**，这表明存在严重的训练问题。
   - 这种持续的 Loss 可能意味着模型训练动态存在潜在问题，或需要解决配置设置错误。

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Agent Executor 缺乏洞察力**：有用户担心 LangSmith 中的 **Agent Executor** 无法展示其规划过程，限制了用户对决策过程的洞察。
   - 参与者建议，增强可见性可能需要用户层面的实现，以获得更好的透明度。
- **LangGraph 兴起用于规划**：一个关于 **LangGraph** 的共享示例引发了关于其促进 Agent 工作流（超越基础执行）潜力的讨论。
   - 鼓励用户学习 LangGraph 的高级功能，以增强他们的项目。
- **Llama 3.1 全新的 Tool Calling 语法**：**Llama 3.1** 中独特的函数调用支持使用了一种特殊的 Prompt 语法，这与标准的参数设置不同。
   - 有人提出疑问，这种语法是否可能成为 **LangChain** 集成中的规范。
- **图灵测试变得有趣**：一篇文章探索了一种有趣的**图灵测试**形式，三个语言模型竞相说服对方自己是 AI 身份。
   - 这种轻松的方式邀请读者思考机器是否真的可以思考，促进了关于 AI 能力的对话。
- **发布全面的 SWE Agent 指南**：一份关于使用 **CrewAI** 和 **LangChain** 等工具创建 **SWE Agent** 的详细指南发布，推广使用 **swekit** Python 框架。
   - 该指南旨在简化各种 Agent 框架的脚手架和功能，可在此处访问 [here](https://git.new/swe/kit)。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Palm Chat 2 使用量增长 3000%**：Palm Chat 2 的使用量从 **1** 次请求激增至 **30** 次，实现了 **3000% 的增长**。
   - 一位成员幽默地将这次激增比作 *WinRAR 销售额* 的梗，为讨论增添了笑料。
- **新的 GPT-4o 支持超长输出**：实验版本的 **GPT-4o** 每次请求可处理高达 **64K 输出 Token**，约合 **18.2K 单词**。
   - 输出成本估计为每 **64K 回复** **1.15 美元**，这是大规模输出的一个重要考虑因素。
- **寻找 LiteLLM 的替代方案**：一位用户对 **LiteLLM** 混乱的文档表示不满，建议利用 **OpenRouter** 构建类似服务。
   - OpenRouter 通过其 Generations 端点提供成本信息，从而提供更多控制权。
- **Claude 模型与 Instruct 模板的挑战**：讨论了 **Claude 3.5 Sonnet 模型** 是否使用了 Instruct 模板，部分人对此表示怀疑。
   - 有建议称在 **OpenRouter** 中使用 `prompt` 模式可以有效地将 Prompt 转换为可用的用户消息。
- **Fireworks 模型状态确认**：一位成员确认虽然 **Fireworks** 运行正常，但 **Yi-Large 端点** 已因不明原因被移除。
   - 这引发了围绕 **Fireworks** 托管模型稳定性的讨论，以确保功能的持续性。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **SAM 2 发布，功能显著增强**：[Meta Segment Anything Model 2 (SAM 2)](https://ai.meta.com/blog/segment-anything-2/) 已发布，提供图像和视频中的实时可提示对象分割，相比前代产品在性能上有显著提升，达到 state-of-the-art 水平。
   - SAM 2 在包含 **50,000** 个视频的新 SA-V 数据集上进行训练，采用了一种新颖的 memory attention 技术，用于在不同场景下进行分割。
- **Leonardo AI 加入 Canva 大家庭**：[Leonardo.Ai](https://x.com/ethan_smith_20/status/1818152222326186260?s=46) 宣布被 Canva 收购，预计将增强创意工具并以新方式赋能创作者。
   - 此次整合将加速创新，基于 Phoenix 等现有项目进行构建。
- **Kagi 推出新的 LLM 基准测试项目**：[Kagi LLM Benchmarking Project](https://help.kagi.com/kagi/ai/llm-benchmark.html) 通过未受污染的基准测试，评估大语言模型在推理、编码和指令遵循方面的能力。
   - 目前结果显示 **gpt-4o** 在准确性和效率方面领先，强调了对不同供应商进行持续测试的必要性。
- **OpenAI 和 Anthropic 的战略合作机会**：讨论表明 **OpenAI** 和 **Anthropic** 可以通过提供基于聊天提及的分析（类似于 [Google Analytics](https://link.to/google-analytics)）与品牌合作。
   - 这可能与 SearchGPT 等新模型保持一致，在确保数据匿名化的同时展示洞察。
- **Apple Intelligence Beta 版发布**：**Apple Intelligence Beta** 现已在 **macOS** 和 **iPhone** 上可用，为用户提供新的 AI 功能。
   - [Discord](https://discord.com/channels/822583790773862470/1249801456870101013) 上的活跃讨论包括对性能和可用性的反馈。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **探索 Open Interpreter 的用途**：成员们讨论了 Open Interpreter (OI) 的各种 **use cases**，强调了其作为任务管理屏幕助手的潜力。
   - *“我一直在寻找一种方法，让某些东西能随着时间的推移学习我的屏幕动作”*，这展示了开源能力的个性化应用。
- **AI 接管编码**：一位成员宣扬了使用 AI 生成代码的成功，吹嘘自己在没有亲自编写任何代码的情况下获得了奖项。
   - 他们敦促其他人利用 AI 提高编码效率，声称 *“相信我，你也可以做到，朋友”*。
- **对 Wayland 体验的担忧**：一位用户分享了他们在 **Wayland** 上的挣扎，透露了在过渡到该显示服务器期间面临的挑战。
   - 他们的反馈反映了用户在适应新系统时的共同感受。
- **Perplexica：你的新搜索伙伴**：一段名为 [Perplexica + Llama-3.1](https://www.youtube.com/watch?v=V0vx94JYNjI) 的视频演示了如何使用 Llama-3.1 构建一个本地、免费的 Perplexity 替代方案。
   - 该教程强调了安装的简便性以及 AI 驱动搜索解决方案的功能性。
- **预订可用性问题**：一位用户询问了构建 Open Interpreter 单元的 **pre-orders** 状态，对找不到更新表示沮丧。
   - 官方澄清预订已不再接受，促使其他人独立收集零件。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **View 合并任务的清晰度**：该任务旨在证明 `View.__add__` 可以合并任何两个可合并的 view，或者在失败时对其进行修改。当 view 不是成对可合并时，复杂性会增加，从而推动 shape tracker 的规约。悬赏发布者强调了定义的清晰度，以确保最小化 view，从而在最终索引计算中获得更好的性能。
- **YouTube 并行计算之旅**：一位成员分享了来自 UCSC 学术报告会的 [YouTube 视频](https://youtu.be/c52ziyKOArc?si=pAUdzwIQGXCtpk3T)，讨论了并行计算及其影响，并提供了幻灯片。该讲座于 2024 年 4 月 10 日举行，强调了并行计算方法论进步的重要性。
- **TinyJit 干扰梯度**：应用 TinyJit 后，在第三个训练循环步骤中，所有张量返回的梯度均为 **None**，这与之前的步骤形成鲜明对比。此问题似乎源于 **TinyJit** 的激活干扰了正常行为。移除 TinyJit 解决了该问题，成员们讨论确认，将 **optim.step()** 放置在 jitted 函数之外可能是罪魁祸首。
- **决定 Jitting 策略**：一位成员在讨论是仅对模型的 forward 步骤进行 jit，还是对整个 step 函数进行 jit，得到的建议是首选全面的 jitting 方法。社区共识倾向于对完整的 step 函数进行 jit，除非有特定原因。
- **遇到 OpenCL 资源错误**：一位成员表示在 Mac 上使用 OpenCL 生成“资源不足”错误时遇到困难，反而遇到了“无效内核”错误。这表明问题可能与编译有关，而非运行时的资源限制。同行间的共识暗示应进一步探索导致资源管理中这些困惑点的编译场景。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **苹果弃用 NVIDIA 转投 TPU**：苹果官方表示，它没有使用 **NVIDIA GPU** 来训练其 AI 模型，而是选择了 **TPU**，正如[最近的一篇文章](https://www.reuters.com/technology/apple-says-it-uses-no-nvidia-gpus-train-its-ai-models-2024-07-29)所报道的那样。此举使苹果成为行业内 **TPU 的第二大用户**。这一决定反映了减少对 NVIDIA 等竞争对手依赖并推广自身 AI 能力的更广泛战略。
- **Tim Dettmers 加入 Allen Institute**：**Tim Dettmers** 已在 **Allen Institute** 获得职位，并将于 2025 年秋季开始在**卡内基梅隆大学**任教。在经过广泛的求职后，他从 **17 所大学**中获得了 **15 份录取通知**。他的目标是在继续从事 **bitsandbytes** 工作的同时，加强开源贡献。对他专业知识的竞争性关注凸显了 AI 领域对人才的需求，**Anthropic** 和 **Hugging Face** 等公司都表达了招募他的渴望。
- **Sewon Kim 对公司的吸引力**：**Sewon Kim** 的招聘引发了各家公司的极大兴趣，说明了他在该领域日益增长的影响力。这种兴趣的涌入强调了通过**独特的产品/方案**来吸引顶尖人才的重要性。这一趋势反映了 AI 人才招聘中的竞争格局，杰出的候选人会吸引多个机会。
- **Zuck 在 SIGGRAPH 上的精彩言论**：在 **SIGGRAPH** 上，**Zuck** 与 **Jensen** 一起发表了坦率的言论，尤其是那句 *“再给我做一个芝士牛排堡，Jensen，”* 为活动的严肃讨论增添了幽默感。这一时刻凸显了高规格会议中经常出现的轻松与厚重感的交融。
- **Perplexity 为出版商推出创新计划**：**Perplexity** 启动了其**出版商计划 (Publishers Program)**，为媒体机构提供**收入分成**和互动工具等功能，旨在提升媒体来源的质量。合作伙伴包括 **TIME** 和 **Der Spiegel** 等知名机构。该倡议不仅旨在分配利润，还旨在提高其系统的整体响应能力。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **在 Trace 框架中探索 OPTO**：成员们强调了 [Trace 使用 OPTO](https://www.microsoft.com/en-us/research/blog/tracing-the-path-to-self-adapting-ai-agents/) 的影响，重点讨论了其在 AI 应用中的相关性。
   - 讨论指出，人们对自适应 AI 技术表现出浓厚兴趣，特别是与游戏领域相关的技术。
- **神经网络的增长**：对话提到了神经网络向拥有 **billions of parameters**（数十亿参数）的复杂系统演进，例如驱动 [ChatGPT](https://arxiv.org/abs/2303.08774) 的系统。
   - 这些进步彻底重塑了 AI 应用在各个领域的能力。
- **MIRPO 与 DSPy 函数的兼容性**：针对此前的兼容性问题，成员们寻求澄清 **MIRPO** 现在是否支持 **dspy.Suggest** 和 **dspy.Assert**。
   - 目前尚未有更新确认该功能已得到解决。
- **为答案偏差创建惩罚指标**：讨论集中在开发一种随金标准答案（gold answer）距离增加而增加的惩罚指标，主张采用比例惩罚。
   - *其中一个建议涉及利用预测分数与实际分数之差的平方公式。*
- **关于 Language Models 的 ICML 演讲**：一位成员分享了关于 Language Models “物理学”的 **ICML talk** 见解，建议优化器可以利用“名人”示例（'celebrity' exemplars）。
   - 演讲链接可以在 [这里](https://youtu.be/YSHzKmEianc) 找到以供进一步探索。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **长上下文创新征集开发者**：团队正积极寻求开发者利用 Jamba 的 **256k effective length**（有效长度）探索长上下文用例，旨在根据 **enterprise customer feedback**（企业客户反馈）提升输出效果。
   - 他们鼓励参与者分享实验结果，并提供 **credits, swag, and fame**（积分、周边和知名度）作为奖励。
- **企业客户分享正面反馈**：企业客户在测试 Jamba 的能力和功能时，初步反应显示出 **promising results**（令人期待的结果）。
   - 消息呼吁提供更多见解，以促进提升平台的协作努力。
- **新用户对 Jamba 充满热情**：新成员 **artworxai** 在 Discord 中介绍了自己，表达了学习更多关于 **Jamba** 知识的渴望。
   - 这表明新用户对该平台的功能和应用兴趣日益浓厚。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **SWE-Bench 超级黑客松挑战代码生成极限**：一场为期 **6 天的 SWE-Bench 超级黑客松** 正在举办，由 StrongCompute 提供 **$1,000 的算力** 支持。基准测试的改进将获得奖金，活动还包括来自 [John Yang](https://x.com/jyangballin)、[Carlos E. Jimenez](https://x.com/_carlosejimenez) 和 [Ofir Press](https://x.com/OfirPress) 等共同作者的演讲。
   - 该活动旨在提升开源代码生成能力，预计讨论将激发社区内的创新方法和见解。
- **GitHub 托管 Segment Anything Model 2 代码库**：**Segment Anything Model 2 (SAM 2)** 的 [GitHub repository](https://github.com/facebookresearch/segment-anything-2) 已上线，提供运行推理的代码、训练好的模型权重（checkpoints）以及示例 notebooks。该资源增强了开源项目中各种分割任务的可用性。
   - 随着这些易于获取的工具发布，围绕 SAM 2 的参与度预计会增加，鼓励开发者轻松实现复杂的分割解决方案。

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Sentry 讨论 AutoFix 功能**：来自 **Sentry** 的 Jenn 和 Ben 将在即将举行的会议中展示他们的 **AutoFix** 功能。活动详情可以在[这里](https://discord.com/events/1089876418936180786/1245836053458190438)找到。
   - 演示预计将涵盖这一开源功能如何增强开发工作流和故障排除，并提供社区驱动的支持。
- **Sentry 开源功能的优势**：即将进行的讨论将强调开发者使用像 AutoFix 这样的 **开源** 功能的优势。参与者可以期待获得关于社区驱动的更新和支持的宝贵见解。
   - 本次会议旨在加深对协作开发实践的理解，并扩大与 **Sentry** 平台的互动。



---


**Alignment Lab AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**LLM Finetuning (Hamel + Dan) Discord** 没有新消息集。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**Torchtune Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**DiscoResearch Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---

# PART 2: 频道详细摘要与链接


{% if medium == 'web' %}




### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1267936585283010640)** (1 条消息): 

> - `Llama 3.1 发布`
> - `Argilla 2.0 预览`
> - `PEFT v0.12.0`
> - `Hugging Face 视频计划`
> - `与 Nvidia 合作的推理即服务 (Inference as a Service)` 


- **Llama 3.1 凭借多语言功能令人印象深刻**：[Llama 3.1](https://x.com/reach_vb/status/1815767864277606762) 已经发布，模型参数最高达 **405B**，在 MMLU 基准测试中达到 **85.2** 分，并支持 **128K context**。
   - *它配备了更宽松的许可证*，允许在其输出上训练其他 LLM，标志着它成为 **GPT4o** 和 **Claude** 的最新竞争对手。
- **Argilla 2.0 拥有数据集复制功能**：即将发布的 [Argilla 2.0](https://x.com/argilla_io/status/1817945202432061792) 将包含一个备受期待的功能，可轻松复制数据集，有助于管理相似的数据集。
   - 这一增强功能预计将为处理多个数据集的用户简化工作流程。
- **PEFT v0.12.0 引入新方法**：[PEFT v0.12.0](https://x.com/julien_c/status/1817837045298978986) 刚刚发布，展示了 **OLoRA** 和 **X-LoRA** 等新的高效方法，增强了模型训练过程。
   - 这些方法旨在提高各种模型的性能和资源效率。
- **Hugging Face 进军视频内容领域**：Hugging Face 正在推出 [视频功能](https://x.com/micuelll/status/1816851392134586540)，以弥补与现有封闭视频模型之间的差距。
   - 这一初步步骤旨在利用其模型进行视频分析和生成。
- **Nvidia 与 Hugging Face 合作提供 AI 服务**：宣布与 [Nvidia AI](https://x.com/NVIDIAAIDev/status/1818050230392398175) 合作提供推理即服务 (Inference-as-a-Service)，使开发者能够使用开源 AI 模型进行原型设计。
   - 该计划旨在简化 AI 模型在生产环境中的部署。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/reach_vb/status/1815767864277606762)">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：Meta Llama 3.1 405B, 70B 和 8B 发布了 —— 支持多语言、128K 上下文、Tool-use 和 Agent！在竞争中毫不逊色甚至击败了 GPT4o 和 Claude Sonnet 3.5，无疑是目前最强的开源 LLM！🐐 额外福利：它还附带...</li><li><a href="https://x.com/reach_vb/status/1818218875239977000)">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：Llama 3.1 8B 在 Mac 上运行，100% 本地化，由 llama.cpp 驱动 🔥 只需两步：1. brew install llama.cpp 2. llama-cli --hf-repo reach-vb/Meta-Llama-3.1-8B-Instruct-Q6_K-GGUF \ --hf-file meta-llama-3.1...</li><li><a href="https://x.com/argilla_io/status/1817945202432061792)">来自 Argilla (@argilla_io) 的推文</a>：💫 对 Argilla 2.0 的发布感到兴奋吗？请关注即将发布的更新！与此同时，我们很高兴能提前展示一个备受期待的功能：便捷的数据集复制...</li><li><a href="https://x.com/julien_c/status/1817837045298978986)">来自 Julien Chaumond (@julien_c) 的推文</a>：以防你错过了上周的消息：PEFT v0.12.0 刚刚发布 🔥 包含了一些酷炫的新参数高效（param-efficient）方法，如 OLoRA、X-LoRA、FourierFT 等</li><li><a href="https://x.com/micuelll/status/1816851392134586540)">来自 Miquel Farré (@micuelll) 的推文</a>：Hugging Face 进军视频领域！我们希望缩小与闭源视频模型的差距，这是我们的第一步。权重：https://huggingface.co/mfarre/Video-LLaVA-7B-hf-CinePile 代码：https://github.com/mfarre/V...</li><li><a href="https://x.com/abidlabs/status/1818034189348053204)">来自 Abubakar Abid (@abidlabs) 的推文</a>：感谢 @mmitchell_ai 提交的优秀 PR，为 Gradio 增加了通过单个参数为 AI 生成的视频添加水印的功能 😎</li><li><a href="https://x.com/davidberenstei/status/1817115209590272021)">来自 David Berenstein (@davidberenstei) 的推文</a>：⚗️ 在 Hugging Face Hub 上查找可重用的合成数据 Pipeline 代码和相应的数据集。找到你的 Pipeline 并使用 `$ distilabel pipeline run --config "hugging_face_dataset_url/pipeline....</li><li><a href="https://x.com/abhi1thakur/status/1816429924233687470)">来自 abhishek (@abhi1thakur) 的推文</a>：🚨 新任务预警：VLM 微调 🚨 AutoTrain 刚刚增加了 VLM 微调功能：支持 PaliGemma 的 Captioning 和 VQA。现在，在自定义数据集上微调 PaliGemma 变得超级简单。下一个模型和任务会是...</li><li><a href="https://x.com/NVIDIAAIDev/status/1818050230392398175)">来自 NVIDIA AI Developer (@NVIDIAAIDev) 的推文</a>：我们与 Hugging Face 合作推出了推理即服务（inference-as-a-service），帮助开发者利用托管在 Hugging Face Hub 上的开源 AI 模型快速构建原型并部署到生产环境。➡️https://...</li><li><a href="https://x.com/RisingSayak/status/1818133546411728903)">来自 Sayak Paul (@RisingSayak) 的推文</a>：随着更大的 Diffusion Transformer 不断涌现，拥有优秀的量化工具变得日益重要。我们展示了一系列实验的研究结果...</li><li><a href="https://x.com/mervenoyann/status/1816857371416887653)">来自 merve (@mervenoyann) 的推文</a>：你知道 Hugging Face 有一个开源的 Cookbook，里面有很多 AI 应用案例（recipes）吗？🤩📖 这里有一些最新贡献的案例 🧶</li><li><a href="https://x.com/_philschmid/status/1816514989982908591)">来自 Philipp Schmid (@_philschmid) 的推文</a>：我听说你喜欢图表。👀 所以，我使用 BigCodeBench 和 Aider（代码编辑）制作了一个针对代码的图表。我们真的应该停止使用 HumanEval 来评估编程能力了！🧑🏻‍💻 > BigCodeBench 评估 LL...</li><li><a href="https://x.com/davidberenstei/status/1816419520447127728)">来自 David Berenstein (@davidberenstei) 的推文</a>：Meta Llama-3.1 模型系列可用于蒸馏和微调，但这需要标注的偏好数据，因此我基于 Gradio 创建了一个人类反馈收集器（Human Feedback Collector），可以直接记录数据...
</li>
</ul>

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1267557819280916513)** (301 条消息🔥🔥): 

> - `Meta LLaMA 的 Token 限制`
> - `Hugging Face Datasets 的问题`
> - `在不同 GPU 上训练模型`
> - `背景去除模型更新`
> - `在 TPU 上使用 Accelerator` 


- **Meta LLaMA Token 限制的困惑**：用户讨论了 **meta/meta-llama-3.1-405b-instruct** 的 Token 限制，有人困惑限制是否为 100 个 Token，而其他人在回复中也报告了大约 100 个 Token 的情况。
   - 一名用户注意到响应被切断，引发了关于推理 API 限制的进一步讨论。
- **Hugging Face Datasets 可靠性问题**：成员们对 Hugging Face Datasets 宕机两天表示沮丧，并讨论了相关错误和不可靠性。
   - 由于成员遇到了 500 错误，有人建议更新并检查数据集的状态。
- **不同 GPU 上的训练问题**：用户分享了在各种 GPU 上训练模型的经验，提到了在 **3060** 上训练时模型冻结和显存溢出（out-of-memory）的问题。
   - 一名用户发现切换到 **A100** 后效果更好，尽管其显存（VRAM）较少。
- **新的背景去除模型可用**：一名成员宣布在 Hugging Face 上合并了一个更好的背景去除模型，大家对优于之前的 **rmbg1.4 model** 的改进感到兴奋。
   - 随后讨论了无法针对特定任务有效地使用旧模型的问题。
- **在 TPU 上使用 Accelerator**：在尝试于 TPU 上利用 Trainer API 时，有人提到如果安装了 **Accelerate**，只需运行脚本就应该能自动识别设备。
   - 然而，存在与 GPU 使用复杂性相关的议题，导致用户寻求替代方案。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/NVIDIAAI/status/1818047933889274159">来自 NVIDIA AI (@NVIDIAAI) 的推文</a>：.@huggingface 已与我们合作推出推理即服务（inference-as-a-service），增强了其平台上开发者的能力。该服务由 NVIDIA NIM 微服务驱动，允许立即访问...</li><li><a href="https://x.com/_philschmid/status/1818286805441286563">来自 Philipp Schmid (@_philschmid) 的推文</a>：Apple 的 AI 策略：没有秘密，基于开源和科学构建，共赢！🚀 看到 @Apple 承认 AI 领域的集体努力，并且不避讳他们如何利用开源构建 Apple Intelligence...</li><li><a href="https://x.com/stevewattsfrey/status/1818033777622532518">来自 Steve Frey (@stevewattsfrey) 的推文</a>：一个大胆的实验：我们正在为 SWE-Bench 举办一场为期 6 天的超级黑客松，以挑战开源代码生成的极限 - 每个人都将获得由 @StrongCompute 提供的 1,000 美元算力 - 多达 50 名研究人员...</li><li><a href="https://huggingface.co/FunAudioLLM/SenseVoiceSmall">FunAudioLLM/SenseVoiceSmall · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/newgen-audiomaker-roblox-6snot-lynxdenis-gif-19984815">Newgen Audiomaker GIF - Newgen Audiomaker Roblox - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/explosion-kitty-komaru-cat-explosion-cat-cat-explosion-gif-4940756872467221811">Explosion Kitty Komaru Cat GIF - Explosion kitty Komaru cat Explosion - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=4rk9fHIOGTU">Llama 8b 测试 - 巨大的退步 📉</a>：LLaMA 3.1 8b 的完整测试。尽管它在基准测试中有巨大提升，但我的测试结果非常令人失望。Vultr 正在赋能下一代...</li><li><a href="https://youtu.be/2PKCOVqhngY?si=DUKS8F0QiBdEHj4R">“我希望 Llama3.1 结合我的私有知识表现提升 10 倍” - 自学习本地 Llama3.1 405B</a>：在 Slack 中构建本地自学习 Llama3.1 Agent。获取关于在工作中采用 AI 的免费 HubSpot 资源：https://clickhubspot.com/7hmy🔗 链接 - 获取完整代码 ...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1267599437673529475)** (4 条消息): 

> - `广泛的测试见解`
> - `LLM 中的 Quantization` 


- **广泛的测试受到好评**：一位成员对**广泛的测试**表示认可，并对分享的**视频**表示感谢。
   - 另一位成员做出了积极回应，表示很高兴这些见解被认为**有用**，并感谢了反馈。
- **探索语言模型的 Quantization**：一位成员分享了一篇关于**量化视觉指南**的文章链接，强调了由于 **Large Language Models (LLMs)** 体积庞大，在消费级硬件上运行它们面临挑战。
   - 文章强调了 Quantization 作为一种使模型更小、更高效的技术的重要性，这是当前**研究**的一个关键领域。



**提到的链接**：<a href="https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization">A Visual Guide to Quantization</a>：探索 LLMs 的内存高效技术

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1267592359852507207)** (9 条消息🔥): 

> - `LLMs 的 Quantization`
> - `关于 AI 的 YouTube 内容`
> - `Diffusion Models`
> - `TikTok AI 趋势` 


- **探索语言模型中的 Quantization**：一篇关于 [Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) 的见解深刻的文章指出，Large Language Models (LLMs) 的参数通常超过数十亿，这使得在消费级硬件上运行它们具有挑战性。
   - 此贴介绍了旨在优化这些模型以获得更好效率和性能的 Quantization 技术。
- **关于 AI 趋势的 YouTube 视频**：分享了几个 [YouTube 视频](https://youtu.be/yJVRXun70dk) 链接，展示了各种 AI 进展，包括一段关于 Diffusion Models 的视频。
   - 其中一段推荐片段非常吸引人，值得 AI 技术爱好者观看。
- **Diffusion Models 的进展**：讨论重点介绍了关于量化 Diffusion Models 能力的 [HuggingFace 博客文章](https://huggingface.co/blog/quanto-diffusers)，该技术改变了高分辨率文本到图像的生成。
   - 随着模型规模的扩大，这种方法解决了与大型基于 Transformer 的扩散流水线相关的内存需求增加的问题。
- **TikTok 上的 AI 趋势**：一段讨论 AI 趋势的 TikTok 视频引起了关注，展示了现代媒体中围绕技术的当代对话。
   - 该视频反映了 AI 如何将其影响力扩展到流行文化和社交媒体平台。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.tiktok.com/@todayin_ai/video/7395098689737444614?is_from_webapp=1&sender_device=pc&web_id=7375199177678243361">TikTok - Make Your Day</a>：未找到描述</li><li><a href="https://huggingface.co/blog/quanto-diffusers">Memory-efficient Diffusion Transformers with Quanto and Diffusers</a>：未找到描述</li><li><a href="https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization">A Visual Guide to Quantization</a>：探索 LLMs 的内存高效技术
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1267623514387779686)** (11 messages🔥): 

> - `CLIP 增强文本搜索`
> - `Unity ML Agent 实验`
> - `Frooty 集成查询`
> - `Guanaco Evolved 数据集`
> - `SAM v2 分割掩码生成器` 


- **CLIP 增强文本搜索展示了创新的图像检索**：一位成员受一篇 [Medium 文章](https://lnkd.in/ebBUnVdr) 启发，使用 **CLIP 算法** 开发了一个名为 **CLIP-Enhanced Text Search on Images** 的图像检索系统，允许用户根据文本查询检索图像。
   - 该技术旨在增强**内容创作**、**数据分析**和**电子商务**，使用户能够高效搜索相关图像。
- **Unity ML Agent 无方向漫游**：一位成员创建了一个新的 **Unity ML Agent**，如 [YouTube 视频](https://youtube.com/live/dcCn4nuKpBs?feature=share) 所示，该 Agent 使用 **SAC** 和基础奖励信号学习在没有路标的场景中漫游。
   - 他们还提到成功将 **CUDA** 与最新的 ML-Agents 集成，并计划将模型上传到 Hugging Face。
- **咨询项目的 Frooty 集成**：一位成员表示有兴趣将其工作与 **Frooty** 集成，而另一位成员则分享了希望攻克游戏开发中 **iplug2** 关卡的愿望，并提到了未来的技术挑战。
   - 在整合 **websockets** 和自动录制功能时出现了问题，引发了对 **FL Studio** 兼容性的不确定性。
- **Guanaco 数据集进化并取得显著成果**：一位成员分享了基于 **Guanaco** 创建 **evolved dataset** 的进展，旨在提高质量和复杂性，同时确定了如评分和 **DEITA filtering** 等潜在改进方向。
   - 初步结果显示在 **MMLU** 上有 **~2% 的提升**，数据集可在 [此处](https://huggingface.co/thesven/SmolLM-360M-Guanaco-Evolved-SFT) 获取。
- **介绍 SAM v2 Mask Generator**：一位成员创建了一个 Space，用于使用最新的 **SAM v2 模型** 生成并导出**分割掩码 (segmentation masks)**，可通过共享的 [Hugging Face 链接](https://huggingface.co/spaces/lightly-ai/SAMv2-Mask-Generator) 访问。
   - 该工具旨在简化图像掩码生成过程，增强用户的实际应用体验。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://lnkd.in/ebBUnVdr]">LinkedIn</a>: 此链接将带您访问 LinkedIn 以外的页面</li><li><a href="https://lnkd.in/eQYY6_rp]">LinkedIn</a>: 此链接将带您访问 LinkedIn 以外的页面</li><li><a href="https://huggingface.co/spaces/lightly-ai/SAMv2-Mask-Generator">SAMv2 Mask Generator - lightly-ai 开发的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://youtube.com/live/dcCn4nuKpBs?feature=share">Unity ML-Agents | 从零开始的 Live Agent 训练</a>: 一个关于 ml agents 和 cuda 的快速小实验
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1267623270870417421)** (2 messages): 

> - `音乐生成模型`
> - `自回归模型技术`
> - `音乐生成相关论文` 


- **对自回归音乐生成的兴趣**：一位成员表示希望专门使用 **autoregressive** 技术而非 diffusion 方法来训练**音乐生成模型**。
   - 他们寻求建议或相关的 [论文推荐](https://link.to/papers)。
- **探索模型替代方案**：另一位成员强调了探索各种 **autoregressive models** 以实现有效音乐生成的重要性。
   - 他们建议查阅关于该主题的现有研究和文献。


  

---

### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1267693369577050163)** (1 messages): 

> - `Quantization Tools for Diffusion Transformers`
> - `Transformer-based Diffusion Models`
> - `Memory Savings in Large Models`
> - `High-Resolution Text-to-Image Generation` 


- **Quantization 增强了 Diffusion Transformers**：最近的实验表明，通过对基于 **diffusion transformers** 的不同扩散流水线进行量化，可以显著节省内存，尽管推理延迟略有增加。
   - 预计这种延迟会随着时间的推移而改善，随着模型规模的增加，量化工具变得至关重要。
- **扩散模型从 UNet 转向 Transformer 架构**：在高分辨率文本生成图像中采用基于 **Transformer-based diffusion backbones**，标志着从之前盛行的 UNet 架构的转变。
   - 这些 Transformer 模型具有从 **0.6B 到 8B** 参数的可扩展性，可以在各种任务中实现更强大的性能。
- **模型缩放增加了内存需求**：随着扩散模型规模的扩大，由于扩散流水线中的多个组件（文本编码器、扩散骨干网络和图像解码器），内存需求变得更加苛刻。
   - 这种复杂性突显了有效的 Quantization 策略在管理资源消耗方面的重要性。



**提及的链接**：<a href="https://huggingface.co/blog/quanto-diffusers">Memory-efficient Diffusion Transformers with Quanto and Diffusers</a>：未找到描述

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1267573209507823769)** (6 messages): 

> - `CachedGISTEmbedLoss function`
> - `Evaluating translated sentences`
> - `Statistical methods for evaluation`
> - `Seq2seq tasks limitations`
> - `Pseudo labels and ontology use` 


- **CachedGISTEmbedLoss 函数实验**：一位成员尝试了 **CachedGISTEmbedLoss** 函数，但发现效果不佳，可能是因为他们已经对数据集进行了充分的清洗。
   - 他们注意到，当被要求专注于 hard negatives 时，训练数据集有了显著改进，有助于模型逐渐细化特征。
- **翻译评估中对评估器的需求**：一位成员询问了如何在没有参考翻译的情况下评估模型的翻译句子。
   - 作为回应，建议包括使用评估器（可能是一个模型），或者让双语者评估输出结果。
- **对统计评估方法的需求**：另一位成员对评估翻译的统计方法表示好奇，利用 **POS tags** 或长度等特征。
   - 他们强调不倾向于使用深度学习模型进行此类评估。
- **Seq2seq 任务的局限性**：一位成员指出，seq2seq 任务通常需要参考或 gold label，这给评估过程增加了限制。
   - 他们还建议使用 pseudo labels 来应对这些挑战，并建议使用词典或 ontology 进行映射。
- **使用 BabbelNET 进行 Ontology 映射**：讨论包括探索一种更奇特的拓扑结构来评估翻译，可能需要 **BabbelNET coverage**。
   - 这种方法允许在翻译任务中映射距离，尽管存在复杂性。


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1267663666564567070)** (4 messages): 

> - `HuggingChat integration`
> - `Knowledge Distillation of 7B model`
> - `SOTA Image Generation`
> - `Compute Resources for Model Training` 


- **寻求将 HuggingChat 集成到 Google App Script 的帮助**：一位成员发布了一个问题，寻求关于使用 API 将 **HuggingChat** 集成到 **Google App Script** 的帮助，并表示其未按预期工作。
   - 他们正在寻求任何有此类集成经验的人的指导。
- **7B 模型的 Knowledge Distillation**：另一位成员请求关于 **7B model** 的 **knowledge distillation** 支持，特别是关于超参数的设置。
   - 他们还询问了该任务所需的计算资源。
- **在图像生成领域达到 SOTA**：一位成员自豪地宣布在内部实现了 **SOTA image generation** 能力，并分享了 [这条推文](https://twitter.com/DataPlusEngine/status/1818358813520441493) 作为亮点。
   - 随后他们又提供了一个链接，展示了图像生成技术的进一步进展。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1267557368787370166)** (195 messages🔥🔥): 

> - `LM Studio Model Loading Issues` (LM Studio 模型加载问题)
> - `Llama 3.1 Model Behavior` (Llama 3.1 模型行为)
> - `AI Development Learning Resources` (AI 开发学习资源)
> - `Thread Count Adjustment in Local Server` (本地服务器线程数调整)
> - `System Prompt Management` (System Prompt 管理)


- **升级后的模型加载问题**：用户报告在将 LM Studio 升级到 0.2.29 版本后出现 GPU 加速失败，之前可以正常运行的模型现在报错，这表明更新过程中可能出现了文件损坏。
   - 一位用户建议清除应用数据并重新安装 0.2.28 版本，而其他人指出 Llama 3.1 需要 0.2.29 版本才能运行。
- **异常的 AI 响应行为**：一位用户在升级 LM Studio 后遇到了 Llama 3.1 8B 模型的循环响应问题，得到的建议是使用 Llama v2 预设以获得更好的性能。
   - 用户承认需要进一步了解 Prompt 格式化，以防止此类循环行为的发生。
- **学习 AI 开发的资源**：一位想要开始 AI 开发的新用户被引导学习 Python、PyTorch 和 Transformer 模型等基础工具。
   - 免费学习资源的建议包括 YouTube 等平台，以帮助掌握这些 AI 核心概念。
- **调整本地服务器的线程数**：一位用户询问如何在本地服务器设置中增加线程数，发现只有代码聊天设置中有该选项，直到升级到最新版本。
   - 更新后，用户确认该选项已出现，这凸显了保持软件更新的重要性。
- **管理 System Prompt**：一位用户在尝试 LM Studio 设置时删除了 System Prompt，并询问此类 Prompt 的常用值。
   - 解释称，通用 Prompt 通常以“You are a helpful XYZ AI assistant”之类的短语开头，用于引导 AI 的回答方向。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.itpro.com/security/python-developers-beware-this-info-stealing-malware-campaign-is-targeting-thousands-of-github-accounts">Python developers beware: This info stealing malware campaign is targeting thousands of GitHub accounts</a>：Python 开发者请注意：这一窃取信息的恶意软件活动正针对成千上万的 GitHub 账号，该恶意软件伪装在流行的 Colorama Python 包中，已导致超过 170,000 名用户的社区受损。</li><li><a href="https://huggingface.co/Groq/Llama-3-Groq-8B-Tool-Use">Groq/Llama-3-Groq-8B-Tool-Use · Hugging Face</a>：未找到描述</li><li><a href="https://www.amuse-ai.com/">Amuse</a>：Stable Diffusion 图像和视频生成</li><li><a href="https://huggingface.co/lmstudio-community/Phi-3.1-mini-128k-instruct-GGUF">lmstudio-community/Phi-3.1-mini-128k-instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/soft-kobe-bryant-no-smh-shaking-my-head-gif-18860898">Soft Kobe Bryant GIF - Soft Kobe Bryant No - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://github.com/meta-llama/llama-agentic-system/blob/main/llama_agentic_system/system_prompt.py">llama-agentic-system/llama_agentic_system/system_prompt.py at main · meta-llama/llama-agentic-system</a>：Llama Stack API 的 Agent 组件。通过在 GitHub 上创建账号为 meta-llama/llama-agentic-system 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1267748792325771305)** (93 条消息🔥🔥): 

> - `GPU Compatibility and Performance` (GPU 兼容性与性能)
> - `LM Studio ROCm Release` (LM Studio ROCm 版本发布)
> - `Motherboard and GPU Upgrades` (主板与 GPU 升级)
> - `AI Streaming Setup` (AI 直播推流配置)
> - `Driver and Cooling Issues` (驱动与散热问题)


- **GPU 兼容性问题**：成员们讨论了 **Intel Iris Xe Graphics** 在 LM Studio 中不受支持，需要配备 CUDA 的 NVIDIA 或配备 ROCm 的 AMD 才能实现兼容。
   - 对话还涉及了 **Tesla P40** 的性能，虽然它拥有更多 CUDA 核心，但与现代消费级 GPU 相比，在速度和兼容性方面面临挑战。
- **LM Studio 0.2.29 版本现已支持 ROCm**：关于 LM Studio 0.2.29 ROCm 版发布的查询得到了回应，指出该版本已经发布，正如 [GitHub 发行说明](https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md)中所示。
   - 成员们对这次更新表示赞赏，并打算在他们的配置中使用这些新功能。
- **升级主板以用于 AI 配置**：一位用户表示打算升级主板，以便容纳两块 RT3060 GPU 用于 AI 直播应用，并强调了对提升 VRAM 容量的需求。
   - 讨论指出，使用第二块 GPU 可能具有成本效益，并且无需彻底更换整机即可提升性能。
- **AI 直播推流考量**：参与者辩论了游戏直播中可能的 AI 集成，一些人建议使用特定的 GPU 配置来促进互动并提升观看体验。
   - 大家达成共识，认为在运行游戏应用的同时运行 AI 时，拥有双 GPU 可能会缓解性能延迟。
- **DIY GPU 散热方案与问题**：一位成员分享了他们为 GPU 散热制作风道的 DIY 经验，提倡使用 3D 打印机以保证质量和便利性。
   - 再次强调了对驱动兼容性和散热效率的担忧，突出了在高负载工作期间保持最佳温度的重要性。



**提到的链接**：<a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md">configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs</a>：LM Studio JSON 配置文件格式及示例配置文件集合。- lmstudio-ai/configs

  

---



### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1267888908994347040)** (1 条消息): 

> - `Perplexity Publishers Program` (Perplexity 出版商计划)
> - `Partnerships with Media Organizations` (与媒体机构的合作伙伴关系)
> - `Revenue Sharing Initiatives` (收入共享计划)


- **Perplexity 推出出版商计划**：Perplexity 推出了其**出版商计划**，与 [TIME](https://pplx.ai/publishers)、[Der Spiegel](https://pplx.ai/publishers) 和 [Fortune](https://pplx.ai/publishers) 等主要机构合作，以确保获取可靠信息。
   - 该计划旨在通过提供新技术来吸引受众并促进共同成功，从而支持出版商。
- **强调可信信息源**：Perplexity 的成功基于提供**高质量的答案**，而这取决于来自 [The Texas Tribune](https://pplx.ai/publishers) 和 [WordPress.com](https://pplx.ai/publishers) 等机构的**可信来源**。
   - 通过在每个答案中加入引用，Perplexity 旨在建立用户信任并确保出版商获得应有的署名。
- **出版商未来的收入共享**：作为出版商计划的一部分，Perplexity 将在未来几个月内推出**收入共享**模式，首先从通过相关问题的广告开始。
   - 此举旨在促进媒体机构的可持续增长，同时让用户从相关内容中受益。



**提到的链接**：<a href="https://pplx.ai/publishers">Introducing the Perplexity Publishers’ Program</a>：从第一天起，我们就在每个答案中包含引用，确保出版商获得应有的署名并建立用户信任。

  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1267560377571610714)** (209 条消息🔥🔥): 

> - `Perplexity AI Down` (Perplexity AI 宕机)
> - `Perplexity Publishers Program` (Perplexity 出版商计划)
> - `AI Language Models Comparison` (AI 语言模型对比)
> - `Impact of Pricing on Services` (定价对服务的影响)
> - `User Support and Experience` (用户支持与体验)


- **Perplexity AI 遭遇宕机**：用户报告 **Perplexity AI** 在不同时段出现宕机且无响应，引发了对其在近期更新后服务可靠性的担忧。
   - 几位用户表达了沮丧，但也指出这类宕机并非任何单一服务所特有。
- **推出 Perplexity Publishers Program**：Perplexity 宣布了一项与出版商共享收入的新计划，以回应针对其内容来源实践的批评。
   - 此举旨在促进合乎道德的合作，同时也应对了新闻机构对内容抓取（scraping）行为的反弹。
- **AI 模型性能对比**：用户对比了包括 **Claude 3.5 Sonnet** 和 **GPT-4o** 在内的不同 AI 模型的有效性，指出了它们在各种任务中的优缺点。
   - 反馈表明，虽然 **Claude** 在特定输出方面表现良好，但 **GPT-4o** 在准确性方面经常受到称赞，尤其是在编程领域。
- **用户对付费服务的担忧**：付费用户担心是否会投放广告，许多人期望其订阅费用能换取无干扰的服务体验。
   - 一些用户表示，付费客户不应被视为“产品”，而应获得无广告的体验。
- **支持与用户查询**：用户就 Perplexity App 的功能以及账户可能存在的问题寻求支持，表达了对有效沟通的需求。
   - 一些用户认为，Perplexity 对于更新或支持问题的沉默令人担忧且感到沮丧。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.theverge.com/2024/7/25/24206488/openais-searchgpt-demo-results-arent-actually-that-helpful">OpenAI 的 SearchGPT 演示结果实际上并没有那么有帮助。</a>：公共 AI 演示中出现幻觉的趋势仍在继续。正如几位记者已经指出的，OpenAI 新的 SearchGPT 引擎演示显示的结果大多要么是错误的，要么是...</li><li><a href="https://arxiv.org/abs/2407.04620">Learning to (Learn at Test Time): RNNs with Expressive Hidden States</a>：Self-attention 在长上下文中表现良好，但具有平方复杂度。现有的 RNN 层具有线性复杂度，但它们在长上下文中的性能受到其表达能力的限制...</li><li><a href="https://www.perplexity.ai/hub/faq/what-is-perplexity-pro">什么是 Perplexity Pro？</a>：浏览 Perplexity 的博客，获取文章、公告、产品更新和优化体验的技巧。保持资讯通畅，充分利用 Perplexity。</li><li><a href="https://x.com/aravsrinivas/status/1818279260517499062?s=61">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：今天，我们宣布推出 Perplexity Publishers Program。我们的成功依赖于确保答案在事实层面基于高质量的信息源。最可扩展且可持续的方式是...</li><li><a href="https://knowyourmeme.com/memes/discord-user-is-a-suspected-terrorist-copypasta">Discord 用户是疑似恐怖分子复制粘贴文 | Know Your Meme</a>：未找到描述</li><li><a href="https://tenor.com/view/second-futurama-scruffy-gif-20187509">Second Futurama GIF - Second Futurama Scruffy - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://one.google.com">获取更多存储空间、更多 AI 功能和更多特性 - Google One</a>：未找到描述</li><li><a href="https://www.perplexity.ai/page/Complexity-Perplexitys-New-yl0q3mHYQz6RhRyuvjvN4w">Complexity：Perplexity 的新扩展</a>：Perplexity AI 的 Complexity 扩展引入了一系列强大的功能，旨在增强用户体验并简化与...的交互。</li><li><a href="https://techcrunch.com/2024/07/30/perplexitys-plan-to-share-ad-revenue-with-outlets-cited-by-its-ai-chatbot/?guccounter=1">Perplexity 详述与 AI 聊天机器人引用的媒体共享广告收入的计划 | TechCrunch</a>：Perplexity AI 很快将开始与新闻出版商共享广告收入，当其聊天机器人在响应用户查询时引用了它们的内容，此举...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1267674278762119168)** (2 条消息): 

> - `Tesla's Charging Warning` (Tesla 充电警告)
> - `Genetically Engineered Flies` (基因工程苍蝇)
> - `Space Force Satellite Expansion` (Space Force 卫星扩张) 


- **Tesla 发布充电警告**：Tesla 发布了关于其充电站潜在问题的警告，提示用户检查兼容性和性能。
   - 这引起了严重依赖 **supercharging** 进行长途旅行的 **Tesla 车主** 的担忧。
- **基因工程苍蝇处理废物**：研究人员开发了可以消耗有机废物的 **基因工程苍蝇**，为废物管理挑战提供了解决方案。
   - 这种创新方法可能有助于减少 **垃圾填埋场废物** 并改善 **回收** 工作。
- **Space Force 计划卫星扩张**：**Space Force** 宣布计划扩大其卫星网络，以增强国家安全和通信能力。
   - 这一举动引发了关于轨道上更多 **军事卫星** 的影响及其对全球治理影响的讨论。



**提到的链接**：<a href="https://www.perplexity.ai/search/why-i-can-use-past-simple-in-p-eP05CN7fSXS7X.qRcV9ZMQ">Perplexity</a>：Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可靠且实时的回答。

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1267859635382124647)** (8 条消息🔥): 

> - `Llama-3 model issues` (Llama-3 模型问题)
> - `Deprecation of Llama models` (Llama 模型弃用)
> - `Request for citations in API` (API 引用请求) 


- **Llama-3 模型产生幻觉信息**：有用户报告无法使用 **llama-3-sonar-large-32k-online**，因为它提供了完全 **幻觉化 (hallucinated)** 的信息，这是直到最近才出现的问题。
   - 另一位用户也对 labs 中新旧在线模型无法产生准确结果表示担忧。
- **即将弃用的 Llama 模型**：成员们注意到，列表中所有的 **Llama 模型**（包括 **llama-3-sonar-small-32k-online**）都将于 2024 年 8 月 12 日 **弃用 (deprecated)**。
   - 由于用户发现这些模型越来越不可靠，人们对其效能提出了担忧。
- **API 引用请求**：一位用户关注了他们在 API 中对 **引用 (citations)** 的请求，称其对 **业务至关重要 (business critical)**，并确认尚未收到回复。
   - 他们分享了[其请求的链接](https://docs.perplexity.ai/discuss/66a8f6b588da9f0024012ab8)，强调了寻求协助的紧迫性。



**提到的链接**：<a href="https://docs.perplexity.ai/discuss/66a8f6b588da9f0024012ab8">Request for citations in API</a>：未找到描述

  

---



### **Stability.ai (Stable Diffusion) ▷ #[announcements](https://discord.com/channels/1002292111942635562/1002292398703001601/1267702752092618792)** (1 条消息): 

> - `Stable Artisan new command` (Stable Artisan 新命令)
> - `Image style generation` (图像风格生成) 


- **Stable Artisan 引入新命令 /style**：**Stable Artisan** 添加了一个新命令 **/style**，允许用户根据指定的风格和提示词生成图像。
   - 例如，用户可以创建像 **梵高风格的猫** 或 **日式风格的宇宙飞船** 之类的图像。
- **体验新图像创建功能**：鼓励成员通过执行 /style 命令来 **体验** 图像创建的新功能。
   - 一位用户分享了一张 **梵高风格可爱猫咪** 的示例图，以展示新命令的潜力。


  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1267558272299176027)** (215 条消息🔥🔥): 

> - `显卡性能`
> - `训练 LoRA 模型`
> - `AI 动画工具`
> - `SAM 2 Segment Anything Model`
> - `Stable Diffusion 配置` 


- **Stable Diffusion 中的 OutOfMemoryError**：用户报告在生成图像时遇到“OutOfMemoryError”，尽管使用的是 8GB 显存的 GPU 和 SD1.5 模型。
   - 建议包括将 CUDA 设置为“prefer system fallback”并增加虚拟内存以缓解内存限制。
- **使用 AI 训练一致性角色**：一位用户分享了在使用包括 IP Adapter 和 ControlNet 在内的各种工具实现一致性角色生成时遇到的困难。
   - 他们提供了自己的 IPAdapter 设置，但寻求关于提高结果一致性的进一步建议。
- **AI 动画工具推荐**：用户讨论了各种用于从静态图像创建极简动画的 AI 动画工具，并指出 Runway 会改变图像质量。
   - 提到了 Live Portrait AI 的功能，尽管它主要专注于面部动画而非通用的图像增强。
- **用于视频分割的 SAM 2 介绍**：来自 Meta 的 SAM 2 模型承诺为图像和视频中的对象提供强大的分割能力，促进实时交互应用。
   - 据指出，SAM 2 提供强大的 zero-shot 性能，可能有利于动画混剪和其他创意尝试。
- **Stable Diffusion 的显卡和内存要求**：讨论强调了在各种 GPU 配置上运行 Stable Diffusion 的用户需要更多 RAM 和有效设置的要求。
   - 建议用户检查其配置是否符合最佳性能的必要建议，特别是对于 AMD 显卡。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://fyrean.itch.io/bgbye-background-remover">BGBye - Background Remover by Fyrean</a>：免费背景移除工具，包含 10 种方法！</li><li><a href="https://www.amuse-ai.com/">Amuse</a>：Stable Diffusion 图像和视频生成</li><li><a href="https://youtu.be/XWuHPMvO-ps">Nvidias Cuda-System Fallback Policy &amp; What It Does! (DO NOT USE) Fortnite Tips&amp; Tricks)</a>：Discord: https://discord.gg/wHM5FyUqTx。在今天的视频中，我解释了 NVIDIA 控制面板中的 CUDA 系统回退策略是什么。这个设置刚刚...</li><li><a href="https://liveportrait.org/">Live Portrait AI - Bring Photos to Life with AI Animation</a>：无描述</li><li><a href="https://github.com/hayden-fr/ComfyUI-Model-Manager">GitHub - hayden-fr/ComfyUI-Model-Manager: Manage models: browsing, donwload and delete.</a>：管理模型：浏览、下载和删除。通过在 GitHub 上创建账户为 hayden-fr/ComfyUI-Model-Manager 的开发做出贡献。</li><li><a href="https://openart.ai/workflows/all">ComfyUI Workflows - Developer Community | OpenArt</a>：在 OpenArt 上发现、分享和运行数千个 ComfyUI 工作流。</li><li><a href="https://github.com/pythongosssss/ComfyUI-Custom-Scripts">GitHub - pythongosssss/ComfyUI-Custom-Scripts: Enhancements &amp; experiments for ComfyUI, mostly focusing on UI features</a>：ComfyUI 的增强与实验，主要侧重于 UI 功能 - pythongosssss/ComfyUI-Custom-Scripts</li><li><a href="https://openart.ai/workflows/home">ComfyUI Workflows - Developer Community | OpenArt</a>：在 OpenArt 上发现、分享和运行数千个 ComfyUI 工作流。</li><li><a href="https://ai.meta.com/SAM2/">无标题</a>：无描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1267560106950656171)** (63 条消息🔥🔥): 

> - `Unsloth 在 Windows 上的使用`
> - `LLM 微调讨论`
> - `使用自定义 Token 的矩阵表示`
> - `对 RoPE Scaling 的支持`
> - `数据集构建工具` 


- **Unsloth 在 Windows 上遇到困难**：一名成员报告在 Windows 上尝试使用 Unsloth 时出现 'No triton module' 错误，对此其他人建议切换到 WSL。
   - 另一位用户幽默地评论说，由于游戏偏好，他们拒绝离开 Windows。
- **使用新数据集微调 LLM**：讨论了在微调 Llama3 模型时如何避免灾难性遗忘（catastrophic forgetting），并提出了合并数据集进行重新训练的建议。
   - 用户确认，为了减轻与灾难性遗忘相关的风险，完全重新训练是更优选的方案。
- **使用自定义 Token 表示矩阵**：一位用户询问如何为他们的 Arc-AGI 项目有效地使用自定义 Token 来表示 30x30 矩阵。
   - 另一位成员要求澄清其请求，表示需要更多细节。
- **RoPE Scaling 支持得到改进**：一名成员分享信息称，旧模型以前缺乏对 RoPE Scaling 的支持，但确认 Unsloth 已在两周前实现了该功能。
   - 用户对这一新功能表示兴奋，其中一人提到了与该功能相关的 Phi-3 128k 变体。
- **使用自定义工具构建数据集**：一位用户寻求从其种子数据（包括 Python 和数学资源）创建自定义数据集文件的工具建议。
   - 他们引用了一个名为 Agent Chef 的 GitHub 工具，以协助数据集的精炼和结构化。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/UnslothAI/status/1818341828254744922">来自 Unsloth AI (@UnslothAI) 的推文</a>：观看 @engineerrprompt 提供的分步视频教程，了解如何使用你自己的数据微调 Llama 3.1：https://www.youtube.com/watch?v=rpAtVIZB72U</li><li><a href="https://x.com/TheXeophon/status/1817991874134569012">来自 Xeophon (@TheXeophon) 的推文</a>：设备端 LLM 为 2.73B 参数，运行在 &lt;4-BIT QUANT 上，使用 LoRA adapters。经过两阶段预训练，随后是 SFT（含合成数据）、RLHF（iTeC, MDLOO（均为新技术））。MMLU：61.4（设备端），75.4（S...</li><li><a href="https://github.com/Leoleojames1/Agent_Chef">GitHub - Leoleojames1/Agent_Chef: Agent Chef 是我们用于数据集精炼、结构化和生成的强大工具。通过利用程序化和合成数据集生成技术，Agent Chef 将使用户能够精炼和清理其微调数据，消除数据投毒和低质量知识库。此外，它还将提供模板和框架。</a>：Agent Chef 是我们用于数据集精炼、结构化和生成的强大工具。通过利用程序化和合成数据集生成技术，Agent Chef 将使用户能够精炼和清理...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1267560593825599590)** (6 条消息): 

> - `Fine-Tuning`
> - `Applied LLMs Course`
> - `Prompt Engineering Video`
> - `LLM Educational Resources` 


- **探索 Fine-Tuning 中的商业机会**：一位成员发起了一场关于专注于模型 **Fine-Tuning** 的 ML 工程师潜在商业机会的讨论。
   - 这促使其他人考虑在该领域进行合作。
- **Applied LLMs Course 免费资源发布**：据 [HamelHusain](https://x.com/HamelHusain/status/1817935895246635362) 称，他们的 **Applied LLMs course** 免费资源已于今日发布，通过增加额外的学习路径增强了课程材料。
   - 这些资源旨在让所有参与者的学习效果最大化。
- **发现 LLM 课程的直接链接**：一位成员分享了 [Parlance Labs](https://parlance-labs.com/education/) 的链接，该网站提供关于 **Fine-Tuning**、**Prompt Engineering** 以及 **LLM** 性能评估等主题的广泛教育材料。
   - 该页面包含多个类别，旨在引导学习 **LLM** 技术的具体应用。
- **关于 Fine-Tuning 的新 Prompt Engineering 视频**：由 [UnslothAI](https://x.com/UnslothAI/status/1818341828254744922) 分享的 **Prompt Engineering** 新视频教程演示了如何使用个人数据 **Fine-Tune** **Llama 3.1**。
   - 可以在 YouTube [此处](https://www.youtube.com/watch?v=rpAtVIZB72U)观看该教程，获取该过程的分步指南。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/UnslothAI/status/1818341828254744922">来自 Unsloth AI (@UnslothAI) 的推文</a>：观看 @engineerrprompt 的分步视频教程，了解如何使用你自己的数据 Fine-Tune Llama 3.1：https://www.youtube.com/watch?v=rpAtVIZB72U</li><li><a href="https://parlance-labs.com/education/">Parlance - 教育资源</a>：关于 LLM 的教育资源</li><li><a href="https://x.com/HamelHusain/status/1817935895246635362">来自 Hamel Husain (@HamelHusain) 的推文</a>：如果你记得我们的 Applied LLMs course，你会喜欢这个。今天，我们将向所有人免费开放所有这些资源！📚 我们额外增加了学习路径、资源以及...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1267563667071959153)** (142 messages🔥🔥): 

> - `Fine-tuning models`
> - `Memory management techniques`
> - `Model conversion`
> - `Model performance metrics`
> - `Multi-GPU usage issues` 


- **使用 Unsloth 进行 Fine-tuning 的故障排除**：用户在 Unsloth 中加载模型时遇到问题，特别是在使用 Fine-tuning 后的 Llama 模型时，出现 `OSError` 提示 unsloth 不是有效的 git 标识符。
   - 根据 GitHub issue 分享了一个解决方法，涉及在 `adapter_config.json` 文件中将 `revision` 设置为 `null`。
- **内存卸载（Memory offloading）能力**：关于 Unsloth 是否支持 RAM 和 GPU 之间的内存卸载有一些讨论，并确认其实现较为复杂。
   - 提到 Gradient checkpointing 是训练中管理内存使用的现有方法。
- **不同设置下的模型性能指标**：几位用户分享了在使用各种模型和框架时的每秒 Token 数 (TPS) 性能指标，并指出 Llama 模型的单次请求 TPS 差异很大。
   - 一位用户报告使用 8-bit 量化的 Llama 模型达到了 150 TPS，而另一位用户分享了通过批处理请求（batched requests）获得更高的 TPS。
- **模型转换和安装问题**：用户询问如何使用 Unsloth 将 `safetensors` 模型转换为 `gguf`，并讨论了在 Yandex DataSphere 等环境中遇到的安装问题。
   - 针对磁盘空间限制的担忧，建议仅保存 LoRA 适配器。
- **多 GPU 使用挑战**：一位用户尝试在不指定 `CUDA_VISIBLE_DEVICES` 的情况下配置多 GPU 设置，面临共享 HPC 资源的限制。
   - 对话强调 Unsloth 目前不支持多 GPU 设置，这限制了用户利用现有硬件的能力。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/rohanpaul_ai/status/1818063089872384348?s=46">Rohan Paul (@rohanpaul_ai) 的推文</a>：📌 LoRA 适配器为特定任务微调基础模型。📌 适配器应用于自注意力层和前馈网络中所有的线性投影矩阵...</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=QmUBVEnvCDJv">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing)">Google Colab</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama#id-12.-saving-the-model">如何微调 Llama-3 并导出到 Ollama | Unsloth 文档</a>：创建可在 Ollama 本地运行的定制化个人助手（如 ChatGPT）的初学者指南</li><li><a href="https://www.unsloth.ai/blog/llama3">使用 Unsloth 微调 Llama 3</a>：通过 Unsloth 轻松微调 Meta 的新模型 Llama 3，上下文长度可延长 6 倍！</li><li><a href="https://github.com/unslothai/unsloth/issues/492">加载 lora 模型时的问题 · Issue #492 · unslothai/unsloth</a>：在尝试加载训练好的 llama3-instruct lora 模型时遇到此错误。然而，两天前它还能正常工作。OSError: unsloth is not a valid git identifier (branch name, tag n...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1267689904742076581)** (4 条消息): 

> - `Translation datasets`
> - `Continued pretraining`
> - `Blockchain Engineer portfolio` 


- **关于翻译数据集的咨询**：一名成员询问是否有可用于将模型从 **English 翻译至任何语言** 进行微调的翻译数据集，并打算使用 DeepL 来完成此任务。
   - 另一名成员建议利用 **Wikipedia** 作为资源。
- **关于 Continued Pretraining 的见解**：一名成员解释说，**Continued Pretraining** (CPT) 使模型能够学习新语言并理解不同领域的知识，正如其 [文档](https://docs.unsloth.ai/basics/continued-pretraining) 中所述。
   - 他们提供了 [text completion notebook](https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing) 和 [continued pretraining notebook](https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing) 的链接，以提供进一步的学习机会。
- **区块链工程师作品集展示**：一名成员分享了他们作为拥有 5 年经验的 **Blockchain Engineer** 的 [作品集](https://alex-portfolio.pages.dev/)，专注于 **Cosmos SDK** 和 **substrate**。
   - 他们强调了在开发桥接协议、zkProof 以及在云平台上配置架构方面的经验，并邀请有兴趣的人员通过 **DM** 联系以获取服务。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://alex-portfolio.pages.dev/">Portfolio - Alex Davis</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/basics/continued-pretraining">Continued Pretraining | Unsloth Documentation</a>：又称 Continued Finetuning。Unsloth 允许你进行 Continued Pretraining，以便模型学习新语言。
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1267787228843606037)** (8 条消息🔥): 

> - `Randomized SVD`
> - `Jacobian method for SVD`
> - `CUBLAS vs CUTLASS`
> - `Accel Venture Capital`
> - `Learning materials for distributed training` 


- **Randomized SVD 简化大型问题**：Randomized SVD 并不取代 SVD，而是将涉及大矩阵的大规模问题简化为更小、更易处理的瘦矩阵 (skinny matrices)，以便进行标准 SVD 处理。
   - 这种方法可以很好地近似原始矩阵的前几个奇异值和向量。
- **Jacobian 方法 Kernel 面临 NaN 问题**：一名成员为使用 Jacobian 方法的 SVD 创建了一个 [Triton kernel](https://link.to/kernel)，但报告称仅得到 NaN 结果。
   - 他们计划很快分享它，并表示这对其他人可能有用。
- **CUBLAS 与 CUTLASS 之争**：一名成员询问，在 **CUTLASS** 已被广泛采用的情况下，是否仍建议使用 **CUBLAS**。
   - 这反映了关于 GPU 计算中最佳工具的持续讨论。
- **了解 Accel 风险投资**：针对一项查询，澄清了 Accel 是一家风险投资公司，并附上了其 [网站](https://www.accel.com/) 链接。
   - 他们以举办活动和支持科技领域的卓越团队而闻名。
- **分布式训练的学习资料**：一名成员征求涵盖分布式训练中 **FSDP**、**TP** 和 **PP** 的学习资料建议。
   - 他们特别寻求能够提高这些领域性能的原理。



**提到的链接**：<a href="https://www.accel.com/">Accel</a>：Accel 是一家全球风险投资公司，是卓越团队从种子轮到 IPO 的首选合作伙伴。Facebook、Flipkart、CrowdStrike、UiPath 和 Spotify 都在 Accel 支持的公司之列...

  

---

### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1267587988582305908)** (3 messages): 

> - `PTX Output Verification`
> - `Libdevice Non-Approximated Exponential`
> - `Code Reflection in Java for Triton`
> - `IR and MLIR Dialects` 


- **需要 PTX 输出验证**：一位用户质疑了某个过程的有效性，并指出应该通过检查其输出的 **PTX (汇编)** 来进行验证。
   - 这突显了 PTX 验证在确认预期功能方面的重要性。
- **Libdevice 拥有非近似指数函数**：另一位成员通过引用 [GitHub 实现](https://github.com/triton-lang/triton/blob/2db56689b0d1268f09dd99cabe4ca940d710da7e/python/triton/language/extra/cuda/libdevice.py#L1156) 确认了 **libdevice** 确实拥有非近似指数函数。
   - 这一引用强调了文档在理解 Triton 功能方面的重要性。
- **探索在 Java 中为 Triton 实现代码反射**：分享了一篇文章，解释了如何使用 **Code Reflection** 在 Java 中实现 [Triton](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html) 编程模型。
   - 该文章是 OpenJDK 项目 [Babylon](https://openjdk.org/projects/babylon/) 的一部分，阐明了 Java 作为 Triton 编程中 Python 替代方案的潜力。
- **IR 和 MLIR Dialects 的细节**：讨论的文章深入探讨了 **IR** 和 **MLIR dialects**，展示了在 Triton 上下文中与代码反射相关的概念。
   - 对 IR 和 MLIR 的关注为 Triton 开发的未来提供了宝贵的见解。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://openjdk.org/projects/babylon/articles/triton">Exploring Triton GPU programming for neural networks in Java</a>：未找到描述</li><li><a href="https://github.com/triton-lang/triton/blob/2db56689b0d1268f09dd99cabe4ca940d710da7e/python/triton/language/extra/cuda/libdevice.py#L1156">triton/python/triton/language/extra/cuda/libdevice.py at 2db56689b0d1268f09dd99cabe4ca940d710da7e · triton-lang/triton</a>：Triton 语言和编译器的开发仓库 - triton-lang/triton
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1267992893512224818)** (2 messages): 

> - `CUDA Memory Alignment`
> - `PyTorch Tensor Alignment` 


- **来自 Caching Allocator 的 CUDA 内存对齐**：有人提出了一个问题，即从 CUDA 缓存分配器返回的 GPU 内存是否总是按某些字节对齐，以便安全地进行重新解释（reinterpretation）。
   - 一位成员回忆说，分配器通常会确保对齐，但警告说 **PyTorch 中的并非所有 Tensor 指针都能保证是对齐的**。
- **关于 PyTorch Tensor 对齐的担忧**：成员们讨论了在执行向量化访问操作时，PyTorch 中 Tensor 指针对齐的影响。
   - 讨论强调了在使用 reinterpret_cast 时可能导致 *CUDA error: misaligned address* 的潜在问题。


  

---


### **CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1267896765118681260)** (1 messages): 

> - `CUDA MODE IRL Meeting`
> - `Keynotes by ML systems leaders`
> - `Working Groups and Hack Leads`
> - `Fireside Chat with Wen-mei Hwu`
> - `Sponsorship and Registration` 


- **加入 CUDA MODE 线下会议！**：CUDA MODE 将于 **9 月 21 日**在旧金山 SoMa 举行 **首次线下会议 (IRL)**，[详情点击此处](https://events.accel.com/cudamode)，共同开发激动人心的项目。
   - 注册人数限制为 **150 名开发者**，感兴趣请尽快申请！
- **ML 偶像的主旨演讲**：与会者将聆听来自 **ML systems** 领域知名领袖的主旨演讲，包括 **Karpathy**、**Tri Dao** 和 **Supriya Rao**。
   - 主旨演讲结束后，参与者将分成工作组协作开发创新项目。
- **获取 Hack 负责人的帮助**：来自服务器的 **熟面孔** 将担任 Hack 负责人，协助参与者在会议期间克服挑战。
   - 这种结构旨在促进整个活动过程中的协作和创造力。
- **炉边谈话和签名会**：当天的亮点包括与 **Wen-mei Hwu** 的 **炉边谈话** 以及 **签名会**。
   - 此外，**Lily Liu** 也将在此时段发表讲话，丰富活动内容。
- **赞助和免费计算额度**：活动由 **Accel**、**NVIDIA** 和 **PyTorch** 赞助，更多赞助商和公告即将发布。
   - 关于 **免费计算额度** 和 **奖金** 的细节将随着日期临近而分享。


  

---

### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1267859325007958137)** (2 条消息): 

> - `Machine Intelligence 进展`
> - `PIM 的重新发现` 


- **机器智能的最新突破**：强调了 **machine intelligence** 的进展，特别是在推荐系统、语音识别和自然语言处理等领域。
   - 最近的一篇文章讨论了这些进展，并提供了指向基础论文的各种 **inline references**，反映了该技术的发展和影响。
- **PIM 的重新发现**：有人对社区最近关注 **PIM (Personal Information Management)** 发表了幽默的评论，暗示它被“重新发现”了。
   - 这引发了关于 PIM 在当今技术领域中的影响和相关性的轻松讨论。



**提到的链接**：<a href="https://www.nature.com/articles/s44335-024-00003-3">Experimental demonstration of magnetic tunnel junction-based computational random-access memory - npj Unconventional Computing</a>：未找到描述

  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1267647819330752622)** (1 条消息): 

> - `CUDA 内存类型`
> - `VRAM 与缓存`
> - `CUDA 中的内存管理` 


- **理解 CUDA 内存类型**：在 CUDA 中，**global memory** 主要由 **VRAM** 支持，在访问时会缓存在 **L2** 中，有时也会缓存在 **L1TEX** 中。
   - 每种内存类型本质上都是一种抽象，**local memory** 尝试映射到寄存器或缓存的 **global memory**。
- **缓存机制详解**：**Texture memory** 具有独特的布局和缓存策略，而 **shared memory** 是 **L1TEX** 的一部分，可以为了效率进行手动管理。
   - **Constant memory** 得益于其不变的特性，从而改进了缓存效果，这使得内存子系统的性能更好。
- **VRAM 使用注意事项**：**global memory** 的大小必须小于 **VRAM**，因为驱动程序和运行时需要占用一部分 VRAM 来实现其自身功能。
   - 因此，驱动程序和缓存会占用一些 VRAM 资源，限制了 global memory 的可用空间。


  

---

### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1267675401228914749)** (19 messages🔥): 

> - `Optimizer CPU Offload`
> - `FSDP and Optimizers`
> - `Low-Bit Optimizers`
> - `CUDA Streams and Transfers`
> - `Pytorch AO Tutorials` 


- **探讨 Optimizer CPU Offload**：成员们讨论了实现 `cpu_offload` 标志，将 Optimizer 状态分配在 CPU 上，在优化步骤中移动参数，然后再将其传回 CUDA。
   - 一位参与者指出，Optimizer step 是阻塞的，这引发了关于与 `torch.compile` 进行交织操作（interleaved operations）可行性的担忧。
- **FSDP 对 CPU Offload 的处理**：会议澄清了 FSDP 在 CPU 上保留参数的“主副本（master copy）”，从而允许高效地移动梯度而无需进行参数传输。
   - 参与者分享了 PyTorch 文档链接，解释了在 Optimizer steps 期间如何管理分片参数（sharded parameters）和梯度。
- **关于 Low-Bit Optimizers 的考虑**：讨论包括在实现 CPU offload 时是否有必要使用 Low-Bit Optimizer，并强调了在节省 CPU RAM 方面的潜在益处。
   - 一位成员提议维护一组主参数（master parameters）以进一步优化 RAM 使用。
- **好奇 CUDA Stream 的功能**：成员们询问了在独立的 CUDA Streams 中排队传输的具体细节，并参考了关于设置和运行 CUDA 操作的资源。
   - 一位参与者分享了一个 GitHub 链接，详细介绍了 CUDA Streams 的作用以及它们如何管理设备上下文（device contexts）。
- **关于量化的博客文章构思**：一位用户提议写一篇博客文章/教程，重点介绍如何使用 PyTorch AO 为 Karpathy 的 build-nanogpt 教程添加量化步骤。
   - 他们寻求关于强调 AO 价值主张的建议，特别是在 GPT 模型的基准测试（benchmarking）和数据类型策略方面。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/msaroufim/tinyoptimizer/blob/master/activation_offload.py">tinyoptimizer/activation_offload.py at master · msaroufim/tinyoptimizer</a>：通过在 GitHub 上创建账号来为 msaroufim/tinyoptimizer 的开发做出贡献。</li><li><a href="https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams">CUDA semantics &mdash; PyTorch 2.4 documentation</a>：无描述</li><li><a href="https://github.com/pytorch/ao/issues/426">The next tutorials · Issue #426 · pytorch/ao</a>：源自我们的 README.md，torchao 是一个用于创建高性能自定义数据类型布局并将其集成到 PyTorch 工作流中的库。到目前为止，我们在构建原语方面做得很好...</li><li><a href="https://github.com/pytorch/pytorch/blob/32c57e78edc46aa71ed19e013741c65b3d777fe9/torch/distributed/_composable/fsdp/_fsdp_param.py#L612-L615">pytorch/torch/distributed/_composable/fsdp/_fsdp_param.py at 32c57e78edc46aa71ed19e013741c65b3d777fe9 · pytorch/pytorch</a>：Python 中具有强大 GPU 加速功能的 Tensor 和动态神经网络 - pytorch/pytorch</li><li><a href="https://github.com/pytorch/pytorch/blob/3864a2d834e3dc84adad791b6fab0c0d401f7e96/torch/distributed/_composable/fsdp/_fsdp_collectives.py#L337-L339">pytorch/torch/distributed/_composable/fsdp/_fsdp_collectives.py at 3864a2d834e3dc84adad791b6fab0c0d401f7e96 · pytorch/pytorch</a>：Python 中具有强大 GPU 加速功能的 Tensor 和动态神经网络 - pytorch/pytorch
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1267565640034553906)** (14 messages🔥): 

> - `SAM 2 demo`
> - `Object Tracking Challenges` 


- **SAM 2 演示令观众印象深刻**：一位成员分享了 [SAM 2](https://sam2.metademolab.com/shared/d8ddf358-7f3a-452b-be9c-f9bd690c9d07.mp4) 的视频演示，展示了其功能，引发了带有多个火表情符号的热烈反应。
   - 一位成员对 **before & after** 对比发表了评论，强调了视觉上的改进。
- **SAM 2 中对象选择的挑战**：成员们讨论了 **SAM 2** 在对象追踪方面的局限性，指出 FPS 和像素分辨率必须与输入匹配才能生效。
   - 有人指出，虽然可以添加正向和负向点，但没有轻松标记多个对象的选项；相反，必须暂停并返回时间线重新选择。



**提到的链接**：<a href="https://sam2.metademolab.com/shared/d8ddf358-7f3a-452b-be9c-f9bd690c9d07.mp4)">SAM 2 Demo | By Meta FAIR</a>：在任何视频中追踪对象，并通过在单帧上进行简单的点击来交互式地创建有趣的效果。

  

---

### **CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1267926065494495416)** (1 messages): 

> - `Apple and HQQ+`
> - `LoRA adapters`
> - `Quantization loss recovery` 


- **Apple “重新发现”了 HQQ+**：Apple 最近在 **HQQ+** 上的工作引发了讨论，特别是考虑到 **4chan** 在三个月前就已经推出了针对 **LoRA** 的量化损失恢复技术。
   - 一位成员分享了一条 [推文](https://x.com/teortaxesTex/status/1818289206948716660)，强调了这一系列事件以及 Apple 采用既有概念的趋势。
- **Apple 关于 LoRA 论文的见解**：在 Apple 的论文中，提到了他们如何在量化模型之上实现 **LoRA adapter**，这有助于恢复准确性。
   - 正如讨论中所表达的，*特定任务的适配器是基于这个准确性恢复基础进行微调的*，这使其成为他们方法论中的一个重要手段。



**提到的链接**：<a href="https://x.com/teortaxesTex/status/1818289206948716660">Teortaxes▶️ (@teortaxesTex) 的推文</a>：顺便说一下，在 Apple 让它变酷的 3 个月前，4chan 就已经做了量化损失恢复 LoRA。引用 Blaze (Balázs Galambosi) (@gblazex) 的话：对我来说，Apple 论文中最有趣的事情之一是……

  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1267558668153520169)** (125 messages🔥🔥): 

> - `Llama 3 Finetuning`
> - `RoPE Integration`
> - `SwiGLU Implementation`
> - `Training Code Developments`
> - `Model Evaluations and Discussions` 


- **为 Jeopardy 比赛微调 Llama 3.1**：一位成员正在使用 **Unsloth** 微调 **Llama 3.1 8B**，由于代码参数复杂，对配置感到不确定。
   - 他们寻求一种更直接的设置，强调偏向于**稳定的 bf16 微调**过程。
- **关于 RoPE 及其实现的讨论**：小组讨论了将 **RoPE** 集成到训练过程中的影响，质疑在 **GPT-2-xl** 未使用它的情况下是否应该包含它。
   - 虽然担心代码会变得过于复杂，但成员们承认 RoPE 对性能的潜在好处。
- **SwiGLU 实现的进展**：成员们分享了实现 **SwiGLU** 的挑战，其中一人确认他们在其上训练了几个模型，并注意到训练动态存在差异。
   - 他们讨论了涉及的复杂性，并考虑使 **SwiGLU** 可训练，权衡其收益与所需的工作量。
- **构建模块化训练代码**：目前正在推动开发一个简洁且模块化的训练设置，并计划单独 fork 并处理 **Llama 3** 的特定更改。
   - 成员们一致认为有必要调整参考 Python 代码，以整合 **GQA**、**SwiGLU** 和 **RoPE** 等特性。
- **Meta 官方代码的挑战**：有人对 **Meta 官方 Llama 3** 代码的可靠性表示担忧，理由是文档中的错误以及对其使用的不确定性。
   - 一位成员专注于创建 **Llama 3 nn.Module** 的简化版本，倾向于独立验证而非依赖外部资源。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/zealandic1/status/1818349493055807929">Anthonix (@zealandic1) 的推文</a>：与 @Yuchenj_UW 发现的最适合原生 GPT2 配置的 0.0018 相比，修改了 SwiGLU+nobiases 的 GPT2 在 2x LR (~0.0036) 下似乎收敛得最好。这些都是 ~124M，FFN 按照 t... 进行了缩放。</li><li><a href="https://github.com/karpath">karpath - 概览</a>：GitHub 是 karpath 构建软件的地方。</li><li><a href="https://hf.co/chat/assistant/66a88b3fc2901bf800fcdeae">GitHub llm.c Repo (L3.1-405b) - HuggingChat</a>：在 HuggingChat 中使用 GitHub llm.c Repo (L3.1-405b) 助手。</li><li><a href="https://github.com/karpathy/llm.c/pull/718">由 gordicaleksa 添加 SwiGLU 支持 · Pull Request #718 · karpathy/llm.c</a>：实现了 SwiGLU - 来自 "GLU Variants Improve Transformer" 论文的 swish GLU 激活函数。注意：由于添加了额外的...，内存占用会有所增加。</li><li><a href="https://github.com/karpathy/llm.c/pull/708">由 gordicaleksa 添加高性能模式 · Pull Request #708 · karpathy/llm.c</a>：添加：当我们进入次优分支时的警告；如果我们没有运行所有最优化分支，将立即退出的高性能模式。还添加了一个将使用的 fwd kernel 配置...</li><li><a href="https://github.com/karpathy/llm.c/pull/715">由 karpathy 提交的 Feature/restore from master · Pull Request #715 · karpathy/llm.c</a>：未找到描述。
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1267969374858383361)** (7 messages): 

> - `WebGPU API`
> - `gpu.cpp 用法`
> - `实时多模态集成`
> - `混合模型计算`
> - `本地设备计算` 


- **WebGPU API：不仅仅是浏览器工具**：WebGPU 作为一个 API 规范，包含了一个名为 **WGSL** (WebGPU Shading Language) 的小型语言定义，允许将其浅层编译为 **Metal**、**Vulkan** 和 **DirectX** 中兼容的着色器。
   - 最初旨在用于浏览器，但现在已被原生应用程序采用，特别是 **Rust** 和 **Zig** 的实现。
- **优先考虑简洁性：gpu.cpp**：**gpu.cpp** 项目在不创建新语言的情况下利用 WebGPU，使用 **WGSL** 编写着色器代码，并简化了 **C++** 项目中的 API 集成。
   - 这种方法旨在抽象掉原始 API 中繁琐的部分，使其更加用户友好。
- **与多模态 IO 的实时集成**：一位用户表示希望利用该能力将模型与**实时多模态（音频和视频）输入/输出**集成，作为主要应用场景。
   - 这种兴趣还延伸到模拟以及探索模型上的分支/条件计算。
- **探索混合模型计算**：讨论中表达了对各种形式的**混合模型计算**的热情，包括将 **CPU SIMD** 与 **GPU** 处理相结合，或集成本地和远程计算。
   - 这种方法可以增强模型在不同环境中部署的性能和通用性。
- **本地设备计算的便捷基底**：一位用户指出，使用 **C++** 中可移植的 GPU API 来探索**本地设备计算**的动力，这为各种新应用提供了便利。
   - 这种结合为实验设备能力和计算技术提供了一种易于访问的方式。


  

---


### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/1267897789938139248)** (13 messages🔥): 

> - `活动注册`
> - `主旨演讲录制`
> - `参会指南`
> - `GPU 访问`
> - `活动期待` 


- **对即将到来的活动充满期待**：成员们表达了他们的热情，其中一位提到他们“对这件事终于发生感到超级兴奋！”
   - *Marksaroufim* 呼应了这一情绪，同样表现出兴奋。
- **主旨演讲将被录制**：计划在服务器和 IRL 现场都对主旨演讲进行录制，确保观众以后可以观看。
   - *Marksaroufim* 确认了记录这些主旨演讲的计划。
- **参会者需要注册**：参会者应确保已注册活动，*Marksaroufim* 鼓励一位成员确认其注册情况。
   - 关于参会的问题得到了解答，明确了注册是必不可少的。
- **讨论 GPU 访问和需求**：一位成员询问是否需要自带 GPU，或者是否会提供计算资源访问。
   - *Marksaroufim* 表示他们正在向赞助商筹集资金，暗示可能会为参会者提供资源。
- **注册者的确认邮件**：在注册 IRL 活动后，成员们很想知道是否会收到关于参会批准的确认邮件。
   - *Marksaroufim* 保证他们很快会向参会者确认批准情况。


  

---



### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

_paradroid: https://arxiv.org/abs/2407.04620
  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/)** (1 messages): 

mautonomy: <:uhhh:1133962718349639721> <:thinking:1134948374760669225>
  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1267855729146990613)** (5 条消息): 

> - `AI Model Size Comparison`
> - `AI Interpretability Concerns` 


- **较小模型展现出竞争优势**：最近的一篇论文指出，运行一次 **70B model** 与从 **13B model** 生成五个输出相比，后者可以产生持续的改进，在 **五个任务中获得高达 15% 的提升**。
   - *这引出了一个问题：当两种模型在相同的预算下运行时会发生什么？* 研究结果强调了 **unit-test setups** 对于选择最佳输出的重要性。
- **对 AI Interpretability 时间表的怀疑**：人们对实现 **AI interpretability** 的时间表表示担忧，一些人认为在私有实践之外获得可靠的数据集可能还需要几年的时间。
   - 这种情绪反映了一种观点，即更长的公共数据发布时间表可能是有益的，可以产生更稳健的研究结果。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.00725">The Larger the Better? Improved LLM Code-Generation via Budget Reallocation</a>: 人们普遍认为大型语言模型 (LLMs) 优于较小尺寸的模型。然而，较大的模型在推理过程中也需要显著更多的时间和计算资源。这引出了一个问题...</li><li><a href="https://tenor.com/view/jim-carrey-gif-12171108510331032271">Jim Carrey GIF - Jim carrey - Discover &amp; Share GIFs</a>: 点击查看 GIF
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1267559570843111565)** (79 messages🔥🔥): 

> - `Apple AI Models`
> - `Hermes Model Merging Techniques`
> - `Midjourney V6.1 Launch`
> - `GPT-4o Capabilities Report`
> - `New MRLM Method` 


- **Apple AI 模型基准测试见解**：一位成员强调了 Apple 新论文中关于其服务器端和端侧模型的有趣细节，其中包含重要的基准测试数据，如端侧模型的 **MMLU** 分数为 **61.4**，服务器模型为 **75.4**。
   - 论文详细介绍了两阶段预训练过程，以及 SFT 和 RLHF 方法。
- **探索 Hermes 和 Llama 模型合并技术**：一位用户询问了将 Hermes 模型与 Llama 合并的技术，并建议准备关于创建有效合并的撰写报告。
   - 随后讨论了各种技术对性能和兼容性的影响。
- **Midjourney V6.1 增强功能**：Midjourney 宣布发布 **V6.1**，承诺在图像质量、连贯性以及新的放大模型方面有显著改进。
   - 此更新是在一位用户声称在图像生成方面达到 state-of-the-art 结果后不久发布的。
- **即将发布的 GPT-4o 能力报告**：人们对预计在 8 月初发布的关于 **GPT-4o** 能力和安全评估的详细报告充满期待。
   - 成员们对性能指标表示好奇，特别是与图像和音频生成相关的指标。
- **用于 Self-RL 模型的新 MRLM 方法**：一位用户讨论了一种用于自我奖励语言模型（self-rewarding language models）的新 **MRLM 方法**，表明其性能优于以往方法，令人印象深刻。
   - 然而，对于缺乏 **MMLU** 等既定指标的基准测试，人们仍存有疑虑。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/midjourney/status/1818342703618482265">来自 Midjourney (@midjourney) 的推文</a>：Midjourney V6.1 现已上线！V6.1 大幅提升了图像质量、连贯性、文本表现，并配备了全新的放大和个性化模型。它更智能、更快速、更清晰、更美观。我们...</li><li><a href="https://mihaiii-trivia.hf.space/">FastHTML 页面</a>：未找到描述</li><li><a href="https://x.com/TheXeophon/status/1817991874134569012">来自 Xeophon (@TheXeophon) 的推文</a>：端侧 LLM 为 2.73B 参数，运行在 <4-BIT 量化下，使用 LoRA 适配器。两阶段预训练，随后是 SFT（使用合成数据）、RLHF（iTeC, MDLOO（两者均为新技术））。MMLU: 61.4 (端侧), 75.4 (服务器)...</li><li><a href="https://x.com/dylan522p/status/1818414482051235994">来自 Dylan Patel (@dylan522p) 的推文</a>：当面对拥有大量算力资源的创始人时，占主导地位的男性会穿上一件更蓬松的皮夹克，以求在吸引配偶的竞争中获胜。这场竞赛类似于...</li><li><a href="https://learn.deeplearning.ai/courses/ai-agents-in-langgraph/lesson/1/introduction">DeepLearning.AI - LangGraph 中的 AI Agents</a>：简介 · 从零开始构建 Agent · LangGraph 组件 · Agentic 搜索工具 · 持久化与流式传输 · Human in the loop · 论文写作 · LangChain 资源 · 结论</li><li><a href="https://til.simonwillison.net/llms/python-react-pattern">一个简单的 LLM ReAct 模式 Python 实现</a>：AI 的一个流行噩梦场景是赋予其访问工具的权限，使其能够进行 API 调用并执行自己的代码，从而普遍突破其初始环境的限制。</li><li><a href="https://tenor.com/view/muahaha-evil-laugh-evil-laugh-futurama-gif-4133163">Professor Farnsworth - 邪恶笑声 GIF - Muahaha Evil Laugh Evil - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1267561183716573298)** (18 条消息🔥): 

> - `NousResearch/Meta-Llama-3.1-8B-Instruct 的差异`
> - `Theta 性能问题`
> - `Hermes 3 与自定义数据集`
> - `BigCodeBench 排行榜` 


- **NousResearch/Meta-Llama-3.1-8B-Instruct 是非门控的**：**NousResearch/Meta-Llama-3.1-8B-Instruct** 模型与原版的主要区别在于 NousResearch 的仓库不是门控（gated）的。
   - 这种开放访问允许用户在没有常规限制的情况下使用该模型。
- **Theta 特有的性能挑战**：**Theta** 目前存在一些问题，尽管它应该使用与 **Llama 3** 等其他模型相同的 system prompt，但其运行表现并不一致。
   - 成员们注意到模型训练中的差异导致了行为的多样化，增加了复杂性。
- **确认 Hermes 3 依赖数据集**：**Hermes 3** 及其 **Pro** 版本将仅使用自定义数据集，这表明其采用了专注的训练方法。
   - 如果后续有其他合并版本，将被命名为 **Hermes 3 Theta**，预示着模型更新的延续。
- **BigCodeBench 拥有排行榜**：一名成员建议关注 **BigCodeBench** 的代码生成任务排行榜，认为它是一个有价值的比较工具。
   - 这引发了关于 **Hugging Face** 排行榜的讨论，参与者对现有选项的熟悉程度各不相同。



**提到的链接**：<a href="https://huggingface.co/spaces/mike-ravkine/can-ai-code-results">Can Ai Code Results - a Hugging Face Space by mike-ravkine</a>：未找到描述

  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/)** (1 条消息): 

teknium: https://x.com/omarsar0/status/1818139150882664696
  

---


### **Nous Research AI ▷ #[reasoning-tasks-master-list](https://discord.com/channels/1053877538025386074/1264666760972472481/)** (1 条消息): 

n8programs: 太棒了
  

---



### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1267912672276512881)** (1 条消息): 

> - `高级语音模式 (Advanced Voice Mode) 推送`
> - `语音对话的安全强化`
> - `GPT-4o 语音能力测试`
> - `ChatGPT Plus 的计划功能`
> - `语音模式中的隐私措施` 


- **高级语音模式开始推送**：高级语音模式（Advanced Voice Mode）开始向一小部分 **ChatGPT Plus** 用户推送，提供更自然、实时的对话体验，并支持随时打断。
   - 参与者已收到电子邮件说明和移动端应用消息，计划在秋季前扩大访问范围。
- **专注于语音对话的安全性**：开发团队在准备推出这项新技术时，一直致力于增强语音对话的**安全性与质量**。
   - 已设置护栏（Guardrails）以拦截涉及**暴力或版权内容**的请求，确保更安全的用户体验。
- **对 GPT-4o 语音功能的广泛测试**：团队邀请了超过 100 名外部红队人员（red teamers），跨越 45 种语言对 **GPT-4o 的语音能力**进行了测试，以评估其性能。
   - 这些测试的经验将为高级语音体验的改进和**安全措施**提供参考。
- **未来功能：视频和屏幕共享**：除了语音功能外，还计划在晚些时候引入**视频和屏幕共享**功能。
   - 这一增强功能旨在进一步丰富 **ChatGPT Plus** 环境下的对话和互动。
- **高级语音模式中的隐私保护**：为了保护用户**隐私**，GPT-4o 将仅使用四种预设语音，并且系统已构建好机制以拦截与这些语音不符的输出。
   - 关于 GPT-4o 能力、局限性及**安全评估**的详细报告预计将于 8 月初发布。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1267570090702802995)** (51 条消息🔥): 

> - `Search GPT 访问权限`
> - `AI-DALLE 3 挑战`
> - `GPT-4o 高级视觉发布`
> - `AGI 讨论`
> - `Midjourney V6.1 发布` 


- **Search GPT 访问权限已开放**：成员确认已获得 [Search GPT](https://link.to.searchgpt) 的访问权限，并向有需要的人提供帮助。
   - *有人认为它很有帮助，而另一些人则对其功能持观望态度。*
- **AI-DALLE 3 挑战相关问题**：一位参与者报告了在访问 **AI-DALLE 3** 挑战时遇到的困难，并在多次尝试后表达了*挫败感*。
   - 他们分享了寻求帮助的细节，以及在解决此访问问题时需要的社区支持。
- **预期的 GPT-4o 高级功能**：关于 **GPT-4o 高级视觉和语音** 功能的预期发布引发了讨论，大家对发布时间看法不一，有人建议可能会推迟到下个月。
   - 另一位成员提到，可能会在**本月底发布 Alpha 版本**。
- **关于 AGI 理解的辩论**：一些成员就 **AGI** 概念的模糊性展开了讨论，指出目前缺乏统一的定义。
   - 观点各异，其中一位强调了其*有趣的本质*和复杂性。
- **对 Midjourney V6.1 的兴奋**：成员们庆祝了 **Midjourney V6.1** 的近期发布，称赞其在图像生成方面的出色能力。
   - 讨论强调了它在文本转换方面的卓越表现和潜在用例，并对*其图像转音频转换的潜力表示了热忱*。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1267567418851721379)** (14 条消息🔥): 

> - `GPT Vision 发布`
> - `OpenAI Assistant 讨论`
> - `GPT 回复质量`
> - `GPT Actions 的 API 响应大小`
> - `GPT 中的 Memory 功能` 


- **关于 GPT Vision 发布的询问**：一位用户询问 **GPT Vision** 是否已经可用，表明对其潜在功能的兴趣日益增加。
   - 对话反映了对 OpenAI 最新更新的期待。
- **关于 OpenAI Assistants 的问题**：一位用户寻求确认该频道是否适合提问有关 **OpenAI assistants** 的问题。
   - 这一询问表明用户希望明确平台的各个支持频道。
- **对 GPT 质量的担忧**：一位用户表示 **GPT 回复** 的质量在过去几周似乎有所下降。
   - 这种情绪得到了其他人的共鸣，表明这是社区内一个较普遍的担忧。
- **GPT Actions 的最大 API 响应大小**：一位成员询问了 **GPT Actions** 的 **API 响应** 的最大限制。
   - 这个问题突出了用户在与 GPT 的 API 交互时的实际考量。
- **关于 Memory 功能的讨论**：出现了关于 GPT 中 **memory 选项** 有效性的问题，一些用户报告了相关故障。
   - 这反映了用户在最新版本中对功能可靠性的持续体验和担忧。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1267700222906073138)** (8 条消息🔥): 

> - `GPTs 训练数据感知`
> - `DALL-E 机器人频道问题`
> - `监护安排脚本问题`
> - `GPT-4 中的 Function call 性能` 


- **GPTs 缺乏训练数据的透明度**：成员们对 GPT 无法清晰说明其训练数据感到沮丧，强调尽管输出详细，但基础模型似乎与其来源脱节。
   - *“在公开版 GPT 上我们还如此落后，这有点荒谬，”* 这表明了对提高透明度的需求。
- **DALL-E 机器人频道功能问题**：一位成员报告了使用 DALL-E 机器人创建图像时的困难，特别指出 `/draw` 命令已数小时无法执行。
   - 他们寻求社区的帮助，并对提供的任何协助表示感谢。
- **监护安排 Python 脚本复杂化**：一位成员试图为 2024 年 8 月创建一个监护安排 Excel 文件，但在跨月边界保持清晰的五天段落时遇到了问题。
   - 通过调整初始 Prompt 明确了指令，最终成功生成了脚本。
- **GPT-4 在使用 Function call 时性能下降**：成员们对使用 **function calls** 时 **GPT-4** 的性能表示担忧，注意到回复的准确性有所下降。
   - 一位成员提到，提交完整 Prompt 的效果比使用 **function calls** 更好。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1267700222906073138)** (8 条消息🔥): 

> - `GPT Training Data Transparency` (GPT 训练数据透明度)
> - `DALL-E Bot Functionality Issues` (DALL-E 机器人功能问题)
> - `Custody Schedule Python Script` (监护时间表 Python 脚本)
> - `Function Call Performance in GPT-4o` (GPT-4o 中的 Function Call 性能)


- **GPT 对训练数据透明度一无所知**：一位成员质疑为什么 GPT 可以访问晦涩的数据点，却无法透露其训练数据，认为这种情况“荒谬”。
   - 他们建议在训练期间增加额外的问答对，以提高对所使用历史数据的清晰度。
- **DALL-E 机器人绘图指令问题**：一位用户对无法在 DALL-E 机器人频道中创建图像表示沮丧，该问题已持续超过 20 分钟。
   - 他们向社区寻求帮助，并对任何支持表示感谢。
- **调整监护时间表脚本的 Prompt**：一位成员在让 ChatGPT 生成监护时间表的 Python 脚本时遇到了困难，但发现通过澄清 Prompt 改善了结果。
   - 他们注意到甚至 Claude 也犯了类似的错误，突显了 Prompt 清晰度的普遍问题。
- **对 GPT-4o Function Call 性能的担忧**：一位成员担心，与使用常规 Prompt 相比，在 GPT-4o 中使用 Function 会导致响应质量下降。
   - 他们询问其他人在使用 Function Call 时是否也遇到了类似的性能退化。


  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1267598596837216306)** (33 条消息🔥): 

> - `API Issues` (API 问题)
> - `Cohere Team Acknowledgment` (Cohere 团队确认)
> - `Project Development with Cohere API` (使用 Cohere API 进行项目开发)
> - `Office Hours Announcement` (Office Hours 公告) 


- **API 宕机，团队正在处理**：一位成员报告 **API 宕机**，遇到了 **503 错误**，并分享了 [Cohere 状态页面](https://status.cohere.com/) 的链接。
   - 另一位成员表示同情并提到：*“抱歉！❤️ 正在内部修复问题！”*。
- **庆祝项目成功**：一位成员兴奋地宣布，他们终于使用 **Cohere API** 构建了他们的**梦想项目**，并收到了热烈回应（包括火表情符号）。
   - 他们提到该项目具有天气、时间、数学和半成品新闻等功能，同时强调了生产环境中背景氛围的重要性。
- **Office Hours 回归！**：一位成员通知大家 **Office Hours** 将回归 Discord，并分享了活动页面的链接。他们表示很想念 Twitter 上的社区，并邀请大家回来。
   - 这引发了关于成员经常错过会议的讨论，以及关于如何确保出席率的建议。



**提及的链接**：<a href="https://status.cohere.com/>">incident.io - Status pages</a>：未找到描述

  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1267723231264116817)** (5 条消息): 

> - `Cohere Status Page` (Cohere 状态页面)
> - `Enterprise Workflow Automation Webinar` (企业工作流自动化网络研讨会) 


- **Cohere 状态页面确认系统运行正常**：[Cohere 状态页面](https://status.cohere.com/) 显示系统运行完全正常，端点可用率为 **99.67%**，文档可用率为 **100%**。目前没有报告影响其系统的故障。
   - 此外，该页面还提供了一个显著的*订阅选项*以获取更新，增强了用户参与度。
- **请求网络研讨会录像**：一位成员在错过 **Enterprise Workflow Automation with GenAI** 网络研讨会后寻求录像。*Sssandra* 建议联系 [events@cohere.com](mailto:events@cohere.com) 以最快获取录制课程。
   - 这表明对于错过直播活动的参与者，有一种结构化的方式来获取重要内容。



**提及的链接**：<a href="https://status.cohere.com/">Cohere Status Page Status</a>：Cohere 状态页面的最新服务状态

  

---

### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1267615990594666559)** (23 条消息🔥): 

> - `Cohere API 停机`
> - `Connector 响应格式`
> - `向 Tool Use 转型` 


- **Cohere API 经历了临时停机**：报告指出 **Cohere API** 出现了响应缓慢和故障，提示用户查看 [status page](https://status.cohere.com/) 获取更新。
   - 一段时间后，确认 API 目前已完全恢复运行。
- **Connector 响应在处理整数时间戳时遇到困难**：一位用户注意到，将 **unix timestamp** 作为整数返回会导致 **Cohere chat API** 不返回结果，而使用字符串表示则运行正常。
   - 官方澄清虽然支持整数，但在 Connector 响应中它们会被视为字符串处理。
- **从 Connector 向 Tool Use 的转变**：讨论强调了向 **Tool Use** 转变的日益增长的趋势，最近的 office hours 洞察也证实了这一点。
   - 尽管如此，官方确认目前没有弃用 Connector 的计划，因为它们与 Tool 并存并提供不同的功能。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.cohere.com/docs/tool-use">Tool Use with Cohere's Models - Cohere Docs</a>：未找到描述</li><li><a href="https://status.cohere.com/">Cohere Status Page Status</a>：Cohere 状态页的最新服务状态
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/1267570341174317056)** (13 条消息🔥): 

> - `Cohere API 性能`
> - `Web Search Tool 实现`
> - `行业炒作周期`
> - `面试准备`
> - `对比测试` 


- **Cohere API 被誉为顶级企业级选择**：一位成员强调 **Cohere API** 是他们唯一使用过且从未经历过停机的 API，称其可能是**最好的企业级选择**。
   - 这一观点得到了其他人的共鸣，引发了关于其与竞争对手相比可靠性的讨论。
- **Web Search Tool 激发创意**：成员们讨论了在聊天界面中使用新的 **Web Search Tool** 创建机器人，其中一人分享了使用 command-r 作为搜索引擎的快速测试。
   - 有人提出了合作邀请，并对在即将到来的面试中对比结果表示热切期待。
- **应对行业炒作周期**：人们对当前行业的**炒作周期**表示担忧，质疑最近的发布是真诚的创新还是仅仅是现有模型的迭代。
   - 一位成员强调致力于确保模型在企业语境中提供真正的价值。
- **面试兴奋感与测试邀请**：一位成员幽默地提到在 Cohere 的面试中使用新工具，引发了其他人提供测试帮助的提议。
   - 这段轻松的对话突显了社区对非技术成员的支持氛围。
- **在竞争对手中重新审视 Cohere 的优势**：一位成员开玩笑说，称 Cohere 为**最好的企业级选择**可能会遭到 OpenAI 的反击，引起了人们对其在社区中良好声誉的关注。
   - 对话表明了对 Cohere 性能优于同领域其他公司的自信且幽默的态度。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1267589651804455003)** (20 条消息🔥): 

> - `Mojo 社区会议 #5`
> - `Stack-PR 安装`
> - `GitHub 文档问题`
> - `社区反馈`
> - `Stack-PR 的 Conda 打包` 


- **Mojo 社区会议 #5 回顾**：今天的第 5 次 [Mojo 社区会议](https://youtu.be/1T-MBC9k99M) 录像已上传至 YouTube，内容包括关于 GPU 编程的讨论和社区问答。
   - 参与者表示希望有更多专注于 *Mojo* 的讨论，并建议在未来的会议中加入现场编程环节。
- **Stack-PR 安装变得简单**：新的命令行工具 **stack-pr** 现在可以通过 `pipx install stack-pr` 轻松安装，支持在 GitHub 上创建多个堆叠的 Pull Request。
   - 成员们讨论了为 stack-pr 工具向 conda-forge 提交 feedstock 的想法，以进一步简化安装。
- **GitHub 文档问题反馈**：一位成员指出了一项关于 Mojo 文档中链接失效的 GitHub Issue，特别提到了 [Issue #3308](https://github.com/modularml/mojo/issues/3308)。
   - 由于文档未注明负责更正的责任方，有人呼吁明确仓库的所有权。
- **建设性的社区反馈**：参与者就社区会议的改进提出了反馈，例如给演讲者更多时间，并确保内容与 Mojo 主题相关。
   - 社区负责人鼓励分享 Discord ID 以获取发言名额，并专注于 Mojo 相关话题的讨论，以提升未来的会议质量。
- **未来演示指南**：一位成员提议为 Mojo 社区会议制定官方指南，确保演示内容与该语言及其工具保持相关。
   - Nick 为未来的演讲建议了一个“试金石”测试：如果一个 Java 程序员也能从中受益，那么讨论的焦点可能需要进一步收敛。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pypi.org/project/stack-pr/">stack-pr</a>: GitHub 的堆叠 PR 命令行工具</li><li><a href="https://pypi.org/project/s">s</a>: 万能的 S 软件包</li><li><a href="https://youtu.be/1T-MBC9k99M.">Mojo 🔥 社区会议 #5</a>: Mojo 社区会议 #5 录像 🔢 Chris Lattner 关于使用 Mojo 进行 GPU 编程 🔥🔀 Async Mojo 🔥 - 10 条简单规则 ❓ 社区问答 完整议程及详情...</li><li><a href="https://www.marcelotrevisani.com/grayskull">Marcelo Duarte Trevisani</a>: 未找到描述</li><li><a href="https://prefix-dev.github.io/rattler-build/latest/converting_from_conda_build/">从 conda-build 转换 - rattler-build</a>: 无</li><li><a href="https://github.com/modularml/mojo/issues/3308">[Docs] Mojo URL 导致 404 · Issue #3308 · modularml/mojo</a>: 问题出在哪里？https://github.com/modularml/mojo 我们能做些什么改进？GitHub 上 Mojo 右上角显示的 URL 不再有效。请将此链接替换为更好的...
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1267565499059929189)** (42 messages🔥): 

> - `Mojo 中 struct 与 class 的区别`
> - `Mojo 中的 CSV 读取器功能`
> - `使用 struct 的性能影响`
> - `Mojo 中的图像解析库`
> - `Mojo class 中的动态行为` 


- **理解 Mojo 中的 Struct vs Class**: 成员们讨论了 Mojo 中缺少 class 的现状以及与 struct 的区别，并指出 Mojo 可能与 C# 和 Swift 具有相似的语义，正如这篇 [参考文章](https://learn.microsoft.com/en-us/dotnet/standard/design-guidelines/choosing-between-class-and-struct) 所强调的。一位贡献者指出，struct 不应仅限于小型不可变类型，还可以用于优化性能。
- **Mojo CSV 读取器讨论**: Affable Honey Badger 询问了 Mojo 的 CSV 读取器，发现它已经存在，但希望能有类似于 Python `csv` 模块的功能。成员们重申，探索此类功能将增强他们对 Mojo 及其 struct 能力的理解。
- **探索 Mojo 中的图像解析**: 一位成员分享了他们在 Mojo 中实现的 PNG 解析，并链接到了他们的 [GitHub 仓库](https://github.com/fnands/mimage)。他们表示打算接下来攻克 JPEG 解析，并参考了一个过时的现有实现，建议将其集成到他们的库中。
- **Mojo Class 中的动态行为**: 关于 Mojo class 是否需要动态行为（类似于其他语言中的引用类型）存在持续的辩论，暗示了对动态分派（dynamic dispatch）和继承的需求。贡献者们表示希望 Mojo 的实现能避免 Objective-C 互操作中出现的复杂性。
- **在 Mojo 中使用 Struct 提升性能**: 成员们讨论了使用 struct 对性能的影响，其中一位成员指出，这需要以在编译时固定成员类型为代价来换取更好的性能。这一观点突显了除非动态行为必不可少，否则更倾向于使用 struct，这与在其他编程语言中的用法形成了对比。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.swift.org/swift-book/documentation/the-swift-programming-language/classesandstructures/">Documentation</a>: 未找到描述</li><li><a href="https://learn.microsoft.com/en-us/dotnet/standard/design-guidelines/choosing-between-class-and-struct">Choosing Between Class and Struct - Framework Design Guidelines</a>: 了解如何决定将类型设计为 class 还是 struct。了解 .NET 中引用类型和值类型的区别。</li><li><a href="https://ruhati.net/mojo/_struct.html">Mojo By Example: A Comprehensive Introduction to the Mojo Programming Language</a>: 未找到描述</li><li><a href="https://github.com/fnands/mimage">GitHub - fnands/mimage: A library for parsing images in Mojo</a>: 一个用于在 Mojo 中解析图像的库。通过在 GitHub 上创建账号为 fnands/mimage 的开发做出贡献。
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1267607257051562137)** (1 messages): 

> - `LlamaIndex Office Hours`
> - `构建 Agent`
> - `深度问题` 


- **LlamaIndex 为用户提供 Office Hours**: LlamaIndex 邀请用户报名参加 [office hours](https://docs.google.com/forms/d/e/1FAIpQLSefrnmxQWD-1OhSP51kUKtdbw9EGDjrMLefkZFACKD19TKsuQ/viewform)，届时可以讨论关于 Agent 的使用案例，并领取 LlamaIndex 品牌的周边礼品作为感谢。
   - Office hours 将包含 15-30 分钟的 Zoom 会谈，以探索 LlamaIndex 如何帮助用户构建 Agent 应用。
- **欢迎深度反馈和使用案例**: 鼓励参与者提出关于使用 LlamaIndex 的深度问题和反馈，特别是对于那些正在构建 Agent 或相关应用的开发者。
   - 对于简短问题，建议用户参考 [Python 文档](https://docs.llamaindex.ai/en/stable/)、[TypeScript 文档](https://ts.llamaindex.ai/) 以及其他文档资源。



**提及的链接**: <a href="https://docs.google.com/forms/d/e/1FAIpQLSefrnmxQWD-1OhSP51kUKtdbw9EGDjrMLefkZFACKD19TKsuQ/viewform">LlamaIndex Community Office Hours</a>: 对 LlamaIndex 的团队有深度问题或反馈？报名参加我们的社区 office hours！我们将联系您安排 15-30 分钟的 Zoom 通话进行交流。我们特别感兴趣...

  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1267576758241329245)** (5 条消息): 

> - `GraphRAG technique` (GraphRAG 技术)
> - `Webinar Scheduling` (Webinar 排期)
> - `Agentic Applications Office Hours` (Agentic Applications 答疑时间)
> - `LlamaCloud QA Assistant Feature` (LlamaCloud QA Assistant 功能)
> - `MLflow in LlamaIndex` (LlamaIndex 中的 MLflow)


- **GraphRAG 结合多种技术进行文本理解**：来自 Microsoft 的 **GraphRAG** 技术将文本提取、网络分析、LLM prompting 和摘要集成到一个系统中。更多详情可以在 [这里](https://t.co/ZnDtJ731hl) 找到，应用说明见 [这里](https://t.co/mx54Su1gYk)。
   - *该过程涉及生成一个增强数据理解的图谱。*
- **Webinar 重新安排至下周四**：计划中的 Webinar 现定于 **下周四 8/8 太平洋时间上午 9 点**。此更新已在最近的 [这条消息](https://t.co/Zo9zRz528F) 中传达。
   - *参与者应在日历中标记新时间。*
- **Agentic Applications 的 Office Hours**：LlamaIndex 邀请 **Agents 或 RAG 应用** 的开发者参加 Office Hours，通过 15-30 分钟的 Zoom 对话讨论其使用案例。感兴趣的人可以在 [这里](https://t.co/o91QKveTWS) 报名，有机会获得免费周边。
   - *这是连接和分享利用 LlamaIndex 见解的好机会。*
- **LlamaCloud 功能增强 QA Assistant**：**LlamaCloud** 的一项新功能支持动态检索，为针对性问题提供 **chunk-level context**（块级上下文），为摘要提供 **document-level context**（文档级上下文）。这增强了构建强大问答助手的潜力，详见 [这里](https://t.co/5WYIx9TcZG)。
   - *关注上下文检索对于有效的问答至关重要。*
- **MLflow 现已支持 LlamaIndex**：**MLflow** 集成了专为 LlamaIndex 量身定制的功能，专注于管理模型开发、部署和管理。关键功能包括跟踪 prompts 和打包引擎，所有依赖项详情见 [这里](https://t.co/BOewMnLklj)。
   - *此增强旨在简化 LlamaIndex 中的模型开发工作流。*


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1267561192239661178)** (55 条消息🔥🔥): 

> - `Spam Issues` (垃圾信息问题)
> - `LlamaIndex Instrumentation` (LlamaIndex Instrumentation 插桩)
> - `RAPTOR Pack Update` (RAPTOR Pack 更新)
> - `Mermaid Diagrams` (Mermaid 图表)
> - `Pydantic Models` (Pydantic 模型) 


- **频道中的垃圾信息问题**：成员们对每个频道中重复出现的垃圾信息表示沮丧，强调需要更好的审核工具。
   - *Discord 使得删除垃圾信息变得容易*，然而成员们对其频率感到担忧。
- **LlamaIndex Instrumentation 的挑战**：讨论集中在为 instrumentation 创建自定义 spans，详细说明了如何使用 LlamaIndex 中新的 `instrumentation` 模块处理 spans。
   - 成员们寻求关于如何有效跟踪 spans 和属性的澄清，强调实际示例将非常有益。
- **RAPTOR Pack 的使用与更新**：关于将 RAPTOR 部署到 Pinecone 等托管向量数据库以及在不重新聚类的情况下处理文档插入的疑问。
   - 成员们讨论了在不丢失之前聚类数据的情况下，管理向 RAPTOR 添加新文档的策略。
- **生成 Mermaid 图表**：成员们分享了从 LLM 输出生成 Mermaid 图表的经验和工具，特别提到了使用 `mmd` 格式。
   - 推荐使用 Mermaid CLI 等工具进行轻松渲染，并分享了示例以便更好地理解。
- **利用 Pydantic 处理结构化数据**：对话强调了在项目中使用 Pydantic 管理结构化数据的优势。
   - 成员们指出，在处理复杂数据模型时，Pydantic 可以轻松执行数据验证。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://mermaid.js.org/intro/syntax-reference.html">Diagram Syntax | Mermaid</a>: 未找到描述</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSefrnmxQWD-1OhSP51kUKtdbw9EGDjrMLefkZFACKD19TKsuQ/viewform">LlamaIndex Community Office Hours</a>: 对 LlamaIndex 有深入的问题或反馈？报名参加我们的社区 Office Hours！我们将联系您安排 15-30 分钟的 Zoom 通话。我们特别感兴趣...</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/observability/instrumentation">Instrumentation - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/mermaid-js/mermaid-cli">GitHub - mermaid-js/mermaid-cli: Command line tool for the Mermaid library</a>: Mermaid 库的命令行工具。通过在 GitHub 上创建一个账户来为 mermaid-js/mermaid-cli 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1267570300384448643)** (8 条消息🔥): 

> - `Transformers 错误`
> - `Q-Galore 状态`
> - `Gemma-2-27B 配置`
> - `Chat Template 训练`
> - `` 


- **Transformers 在索引期间抛出错误**：几位成员在处理 **Transformers** 库（尤其是 **Mistral** 模型）时遇到了断言错误：`srcIndex < srcSelectDimSize`。
   - 一位成员建议**删除缓存**并重新下载所有内容作为潜在的修复方法。
- **Q-Galore 可用性查询**：一位成员询问了 **Q-Galore** 的状态，质疑其目前是否可用。
   - 聊天中没有关于其状态的回复。
- **需要微调 Gemma-2-27B 的配置**：一位成员请求一个用于微调 **gemma-2-27b** 的可用配置，表示需要指导。
   - 回复中没有分享具体的配置或解决方案。
- **chat_template 训练的新要求**：讨论强调了 **PR #1756** 引入了一个新要求，即在使用 `type: chat_template` 时需要 `roles_to_train` 字段。据报道，这一更改破坏了现有的使用 **chat_template** 的示例。
   - 成员们强调需要示例和文档来澄清这一新特性。
- **与 Transformers 库相关的 CUDA Runtime 错误**：报告了一个关于触发 CUDA 设备端断言的 **RuntimeError**，具体发生在 **modeling_mistral.py** 文件中。
   - 当 **attention_mask** 包含 `0.0` 时，似乎会产生此错误，从而导致模型执行中的进一步复杂化。


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1267899972913991740)** (43 条消息🔥): 

> - `Gemma 2 模型微调问题`
> - `使用 RTX 4090 进行聊天机器人训练`
> - `模型微调的最佳实践`
> - `检索增强生成 (RAG)`
> - `训练中的损失函数异常` 


- **Gemma 2 模型输出 pad token**：一位用户遇到了一个问题，其微调后的 **Gemma 2 9b** 模型在合并并部署到 vLLM 后，反复输出 `<pad>` token。
   - 讨论指向了潜在的配置问题，以及验证来自 [Hugging Face](https://huggingface.co/google/gemma-2-9b-it/blob/main/special_tokens_map.json) 的特殊 token 的重要性。
- **RTX 4090 训练聊天机器人的适用性**：另一位用户正在探索在 **RTX 4090** 上训练聊天机器人，提到他们尝试过 **Llama v3.1** 和 sharegpt 数据集，但没有获得理想的结果。
   - 他们表示有兴趣使用 Axolotl 进行微调，并考虑购买第二块 RTX 4090 以增加 VRAM。
- **模型微调的最佳实践**：分享了关于使用优质数据集、配置 LoRA 以实现 VRAM 高效训练以及考虑 batch sizes 的建议，并就如何改进训练结果提出了建议。
   - 再次强调了拥有坚实数据集的重要性，以及可能采用的策略，如使用 Retrieval Augmented Generation (RAG) 来增强训练过程。
- **用于聊天机器人的检索增强生成 (RAG)**：一位参与者还讨论了探索 **Retrieval Augmented Generation (RAG)** 作为其聊天机器人项目微调的替代方案。
   - 他们计划在 RAG 和微调上都投入时间，希望能获得一个良好的微调补丁来强化他们的项目。
- **损失函数停留在零**：一位用户报告他们的训练损失（loss）停留在 **0.0**，且 'grad_norm' 显示为 **nan**，这表明他们的模型训练过程中存在潜在问题。
   - 这种损失问题可能预示着训练动态或配置设置存在问题，需要进行调整。


  

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1267739661644857364)** (45 条消息🔥): 

> - `Agent Executor 与 Toolkits 使用`
> - `LangGraph 功能`
> - `Llama 3.1 Tool Calling`
> - `LangChain Context Caching`
> - `Google Gemini 集成` 


- **Agent Executor 缺乏反思能力 (Reflection)**：一位用户对 Agent Executor 在 LangSmith 中无法显示其规划和思考过程表示担忧，因为这些过程发生在模型端。
   - 回复指出，增强决策过程的可视化可能需要在用户层面进行额外的实现。
- **探索使用 LangGraph 进行规划**：分享了一个 LangGraph 示例链接，作为使用图（graphs）构建 Agent 工作流的一个有前景的起点。
   - 用户讨论了学习 LangGraph 以实现超越基础 Agent 执行的更高级功能的优势。
- **Llama 3.1 独特的 Tool Calling 语法**：Llama 3.1 因其独特的函数调用支持而受到关注，它使用特殊的 Prompt 语法而非典型的参数设置。
   - 产生了关于这种语法是否会被适配到标准 LangChain 用法中的疑问。
- **LangChain 对 Context Caching 的处理**：用户询问了 LangChain 中 Google Gemini 的 Context Caching 集成情况，但目前尚未找到明确答案。
   - 提到 LangChain 提供了对 Gemini 模型的支持，但缺乏关于 Context Caching 特性的具体细节。
- **LLM 决策的最佳实践**：分享了关于让 LLM 执行更小、更清晰的决策任务的建议，并强调此类模型可能会引入不可预测性。
   - 鼓励用户尽可能手动编写逻辑，以更可靠地引导 LLM 输出。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.together.ai/docs/llama-3-function-calling">使用 Llama 3.1 进行函数调用</a>: 未找到描述</li><li><a href="https://github.com/langchain-ai/langchain/pull/24570>).">更好地共同构建软件</a>: GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、分叉并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/migrate_agent/#in-langgraph-1>).">如何从旧版 LangChain agents 迁移到 LangGraph | 🦜️🔗 LangChain</a>: 本指南假设你已熟悉以下概念：</li><li><a href="https://js.langchain.com/v0.2/docs/how_to/streaming_llm/#using-stream>).">如何从 LLM 流式传输响应 | 🦜️🔗 Langchain</a>: 所有 LLM 都实现了 Runnable 接口，该接口带有标准可运行方法的默认实现（即 ainvoke, batch, abatch, stream, astream, astream_events）。</li><li><a href="https://js.langchain.com/v0.2/docs/how_to/migrate_agent/#iterating-through-steps>).">如何从旧版 LangChain agents 迁移到 LangGraph | 🦜️🔗 Langchain</a>: 本指南假设你已熟悉以下概念：-</li><li><a href="https://js.langchain.com/v0.2/docs/how_to/streaming/#chains>).">如何进行流式传输 | 🦜️🔗 Langchain</a>: 本指南假设你已熟悉以下概念：</li><li><a href="https://github.com/langchain-ai/langgraph/blob/main/examples/plan-and-execute/plan-and-execute.ipynb">langgraph/examples/plan-and-execute/plan-and-execute.ipynb at main · langchain-ai/langgraph</a>: 将韧性语言 Agent 构建为图。通过在 GitHub 上创建账号为 langchain-ai/langgraph 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/)** (1 条消息): 

ericyin_41626: https://github.com/langchain-ai/langserve/issues/720
  

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1267761690129207326)** (2 messages): 

> - `Turing Test Implementation`
> - `SWE Agent Development` 


- **Turing Test 的趣味玩法**：一篇新文章探索了一种有趣的 **Turing Test** 方法，涉及三个 Language Models 在游戏形式中竞争，试图说服对方自己是机器。
   - *文章讨论了机器是否能够思考*，并邀请读者通过实验来发现这一点。
- **构建 SWE Agents 指南**：一位用户分享了关于使用 **CrewAI**、**AutoGen**、**LangChain** 和 **LlamaIndex** 等框架创建 **SWE Agents** 的全面指南。
   - 该指南强调利用名为 **swekit** 的 Python 框架，以便在各种 Agentic 框架中轻松进行脚手架搭建和功能实现，访问地址为 [这里](https://git.new/swe/kit)。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://medium.com/taming-the-llama/turing-the-tables-making-language-models-fight-cfa0bc168878?sk=8ba3dcccf2f3f0a294f991fd5e3c167f">Tur(n)ing the Tables — Making Language Models Fight.</a>：对著名实验 Turing Test 的趣味解读，目标相似 —— 测试机器有多聪明（或具有欺骗性）。</li><li><a href="https://git.new/swe/kit">SWE Python Framework - Build SWE Agents </a>：使用 swekit 这个 Python 框架释放 SWE Agent 的力量。轻松构建并搭建与 CrewAI 和 LlamaIndex 等 Agentic 框架兼容的 Agent。利用我们的工具生态系统...
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1267856438127235123)** (2 messages): 

> - `Self Learning Llama3.1`
> - `SWE Agent Guide`
> - `LangChain` 


- **探索 Llama3.1 的自学习**：一段名为《我希望 Llama3.1 凭借我的私有知识表现提升 10 倍》的新 YouTube 视频讨论了在 Slack 中构建本地自学习 **Llama3.1 Agent**，并提供了完整代码和资源的链接。
   - 作者还邀请观众通过[此链接](https://clickhubspot.com/7hmy)获取关于在工作中采用 AI 的免费 HubSpot 资源。
- **构建软件工程 Agent 指南**：一位用户分享了使用 LangChain 创建自己的 **SWE Agent** 的指南，可在[此 GitHub 链接](https://git.new/swe/kit)找到。
   - 该框架 **swekit** 旨在简化软件工程 Agent 的开发，并内置了对 **CrewAI** 和 **LlamaIndex** 等 **Agentic 框架**的兼容性。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://git.new/swe/kit">SWE Python Framework - Build SWE Agents </a>：使用 swekit 这个 Python 框架释放 SWE Agent 的力量。轻松构建并搭建与 CrewAI 和 LlamaIndex 等 Agentic 框架兼容的 Agent。利用我们的工具生态系统...</li><li><a href="https://youtu.be/2PKCOVqhngY?si=DUKS8F0QiBdEHj4R">&quot;I want Llama3.1 to perform 10x with my private knowledge&quot; - Self learning Local Llama3.1 405B</a>：在你的 Slack 中构建本地自学习 Llama3.1 Agent。获取在工作中采用 AI 的免费 HubSpot 资源：https://clickhubspot.com/7hmy 🔗 链接 - 获取完整代码 ...
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1267567980359843912)** (47 messages🔥): 

> - `Palm Chat 2 激增`
> - `GPT-4o 能力`
> - `成本追踪替代方案`
> - `Claude 模型指令模板` 


- **Palm Chat 2 使用量增长 3000%**：一位成员幽默地指出，Palm Chat 2 的使用量从 1 个请求激增到 30 个，导致了 **3000% 的增长**。
   - 另一位成员提到，这种急剧上升让他们想起了 *WinRAR 销售额* 的梗，进一步增添了趣味性。
- **新 GPT-4o 支持超长输出**：实验版本的 **GPT-4o** 每次请求可处理高达 **64K 输出 token**，估计约为 **1.82 万个单词**。
   - 据指出，每 **64K 回复** 的输出价格约为 **$1.15**，这为长响应增加了显著的成本因素。
- **寻找 LiteLLM 替代方案**：一位用户对 LiteLLM 混乱的文档表示沮丧，并建议为类似服务构建潜在方案，最终选择了 **OpenRouter**。
   - 另一位用户指出，OpenRouter 可以提供更多控制权，因为它从 generations 端点提供成本信息。
- **Claude 模型与指令模板的挑战**：关于 Claude 3.5 Sonnet 模型是否使用指令模板（instruct template）展开了讨论，一些人表示它没有模板。
   - 有人暗示在 OpenRouter 中使用 `prompt` 模式可能会将 prompt 转换为用户消息，因此正确引导模型至关重要。
- **Fireworks 模型状态**：一位成员确认，虽然 Fireworks 运行正常，但 **Yi-Large 端点** 已因不明原因被移除。
   - 这引发了关于 Fireworks 托管的其他模型稳定性的讨论，确保大多数模型仍按预期运行。


  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1267576522785423592)** (33 messages🔥): 

> - `SAM 2 发布`
> - `Leonardo AI 加入 Canva`
> - `Kagi LLM 基准测试`
> - `OpenAI 与 Anthropic 与品牌的合作`
> - `白宫关于开源 AI 的报告` 


- **SAM 2 发布，具备增强功能**：[Meta Segment Anything Model 2 (SAM 2)](https://ai.meta.com/blog/segment-anything-2/) 已发布，提供图像和视频中的实时可提示目标分割，在性能上较其前代产品有显著提升，达到业界领先水平。
   - SAM 2 在包含 5 万个视频的新 SA-V 数据集上进行训练，并采用了一种新颖的记忆注意力（memory attention）技术，使其能够在各种场景中分割不同的物体。
- **Leonardo AI 加入 Canva 大家庭**：[Leonardo.Ai](https://x.com/ethan_smith_20/status/1818152222326186260?s=46) 宣布被 Canva 收购，预计这将为用户大规模增强创意工具，并以新方式赋能创作者。
   - 此次整合旨在加速创新，并以 Phoenix 等项目的现有成功为基础。
- **Kagi 启动新 LLM 基准测试项目**：[Kagi LLM Benchmarking Project](https://help.kagi.com/kagi/ai/llm-benchmark.html) 通过未受污染的基准测试对大语言模型在推理、代码编写和指令遵循能力方面进行严格评估。
   - 目前的基准测试结果显示 **gpt-4o** 在准确性和效率方面领先，证明了在不同供应商之间进行持续测试的必要性。
- **OpenAI 和 Anthropic 的战略合作机会**：有讨论涉及 **OpenAI** 和 **Anthropic** 通过提供基于聊天对话中品牌提及的分析数据来与品牌合作的潜力，类似于 [Google Analytics](https://link.to/google-analytics)。
   - 随着 SearchGPT 等新模型的出现，这可能变得越来越重要，以便在确保聚合数据保持匿名性的同时展示洞察。
- **白宫报告倡导开源 AI**：白宫发布了一份报告，强调了 **开源（open-source）** AI 技术的重要性，并反对立即对该类模型实施限制。
   - 这一立场被视为在管理潜在风险的同时推动支持创新，突显了 AI 发展中关于开源模型的持续辩论。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

</div>

<ul>
<li>
<a href="https://x.com/midjourney/status/1818342703618482265">来自 Midjourney (@midjourney) 的推文</a>：Midjourney V6.1 现已上线！V6.1 大幅提升了图像质量、连贯性、文本表现，并配备了全新的 upscaling 和个性化模型。它更智能、更快速、更清晰、更美观。我们...</li><li><a href="https://x.com/hturan/status/1818332375358554133?s=46">来自 harley turan (@hturan) 的推文</a>：嘿！我们在 @cloudflare 构建了一个新的 AI 游乐场，以展示通过链式调用多模态模型可以实现的目标。想象一下——音频 → 文本 → 图像 → 文本，或者组合多个 LLM...</li><li><a href="https://x.com/alexalbert__/status/1817996841923104908?s=46">来自 Alex Albert (@alexalbert__) 的推文</a>：我以前有一个书签文件夹，里面装满了各种小网站，用来执行验证 JSON、检查两个文本之间的差异、格式化 Markdown 等任务。最近我把所有这些书签都替换成了...</li><li><a href="https://x.com/HamelHusain/status/1818040423136510077">来自 Hamel Husain (@HamelHusain) 的推文</a>：TLDR；你必须测试你的 LLM 提供商 - 不同提供商提供的模型很少是“完全相同”的。对于同一个模型，你可能会体验到实质性的结果差异！ - 如果一个模型表现...</li><li><a href="https://apnews.com/article/ai-open-source-white-house-f62009172c46c5003ddd9481aa49f7c3">白宫表示无需限制“开源”人工智能——至少目前如此</a>：白宫表态支持“开源”人工智能技术，在周二的一份报告中辩称，目前没有必要对制造关键组件的公司实施限制...</li><li><a href="https://x.com/swyx/status/1818074658299855262">来自 swyx 🌉 back in SF! (@swyx) 的推文</a>：Memory Attention：通过 50k 美元的算力增加物体持久性。@AIatMeta 继续引领真正的 Open AI。SAM 2 将 SAM 1 从图像分割泛化到视频，发布了任务、模型和数据集...</li><li><a href="https://help.kagi.com/kagi/ai/llm-benchmark.html">Kagi LLM 基准测试项目 | Kagi 文档</a>：未找到描述</li><li><a href="https://x.com/ethan_smith_20/status/1818152222326186260?s=46">来自 Ethan (@Ethan_smith_20) 的推文</a>：我非常高兴地宣布，今天 http://Leonardo.Ai 已加入 Canva 大家庭。这是一段非凡的历程，我想不出比这更好的团队来并肩工作了。我绝对...</li><li><a href="https://x.com/AIatMeta/status/1818055906179105010">来自 AI at Meta (@AIatMeta) 的推文</a>：介绍 Meta Segment Anything Model 2 (SAM 2) —— 首个用于图像和视频中实时、可提示物体分割的统一模型。SAM 2 今天以 Apache 2.0 协议发布，以便任何人...</li><li><a href="https://x.com/nickadobos/status/1818159193037451398?s=46">来自 Nick Dobos (@NickADobos) 的推文</a>：GPT-4o 长输出！？64k 输出！？现在我们要大显身手了。太棒了，冲吧</li><li><a href="https://x.com/drjimfan/status/1818302152982343983?s=46">来自 Jim Fan (@DrJimFan) 的推文</a>：Project GR00T 的激动人心更新！我们发现了一种系统化扩展机器人数据的方法，解决了机器人技术中最痛苦的痛点。想法很简单：人类在真实机器人上收集演示...</li><li><a href="https://www.youtube.com/watch?v=y1WnHpedi2A">你认为 ChatGPT 能够推理吗？</a>：Subbarao Kambhampati 教授认为，虽然 LLM 是令人印象深刻且有用的工具，尤其是在创意任务方面，但它们在逻辑推理方面存在根本性的局限...</li><li><a href="https://ai.meta.com/blog/segment-anything-2/">未找到标题</a>：未找到描述</li><li><a href="https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization">Quantization 视觉指南</a>：探索 LLM 的内存高效技术</li><li><a href="https://github.com/traceloop/openllmetry">GitHub - traceloop/openllmetry：基于 OpenTelemetry 的 LLM 应用开源可观测性</a>：基于 OpenTelemetry 的 LLM 应用开源可观测性 - traceloop/openllmetry</li><li><a href="https://github.com/facebookresearch/segment-anything-2">GitHub - facebookresearch/segment-anything-2：该仓库提供了运行 Meta Segment Anything Model 2 (SAM 2) 推理的代码、下载训练好的模型 checkpoints 的链接，以及展示如何使用该模型的示例 notebooks。</a>：该仓库提供了运行 Meta Segment Anything Model 2 (SAM 2) 推理的代码、下载训练好的模型 checkpoints 的链接，以及展示如何使用该模型的示例...</li><li><a href="https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/">未找到标题</a>：未找到描述
</li>
</ul>

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1267652535083204608)** (1 条消息): 

> - `Apple Intelligence`
> - `macOS updates`
> - `iPhone updates` 


- **Apple Intelligence Beta 发布**: **Apple Intelligence Beta** 现已在 **macOS** 和 **iPhone** 上推出，为用户提供新的 AI 功能访问权限。
   - 随着用户开始体验最新功能，[Discord](https://discord.com/channels/822583790773862470/1249801456870101013) 上的更新和讨论正在持续进行中。
- **Discord 上的 Apple Intelligence 互动**: 用户正在 **Discord** 上积极讨论 **Apple Intelligence Beta** 及其功能，重点关注性能和可用性反馈。
   - 该频道出现了大量关于初始体验以及与早期版本对比的评论。


  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1267565946533445723)** (14 条消息🔥): 

> - `Open Interpreter use cases`
> - `AI for coding`
> - `Speech command technology`
> - `Command line options`
> - `Workflow with Llama 3.1` 


- **探索 Open Interpreter 的用途**: 成员们讨论了 Open Interpreter (OI) 的各种**使用案例**，包括一位成员因健康问题需要屏幕助手来协助完成任务，并认为该技术非常有前景。
   - *我一直在寻找一种方法，让某些东西能够随着时间的推移学习我的屏幕动作*，并能回溯我做过的事情。
- **AI 接管编程**: 一位成员提到他们成功使用 AI 生成代码，并声称自己在没有亲自动手写代码的情况下获得了编程奖项，展示了此类工具的潜力。
   - 他们鼓励其他人利用 AI 进行编程以提高生产力，并表示：*相信我，朋友，你也可以做到*。
- **对 AI 自动化的担忧**: 针对 **Open Interpreter** 的实验性质及其在关键任务中的可靠性，成员们提出了担忧，其中一位成员建议在使用语音命令时保持谨慎。
   - 另一位成员建议使用像 **Whisper** 这样更高精度的工具进行语音转文字，作为更安全的替代方案。
- **澄清命令行选项**: 一位成员询问命令行用法中的 `--local` 和 `--os` 选项是否冗余，并得到了澄清：`--os` 允许在没有提示的情况下控制计算机。
   - `--local` 选项用于本地推理，使本地模型能够在 Open Interpreter 环境中运行。
- **在 Open Interpreter 中使用 Llama 3.1**: 一位成员寻求在使用 **Llama 3.1** 与 Open Interpreter 时的协作流程指导，询问在 OI 启动后是在同一个终端会话中交互还是开启新会话。
   - 他们运行命令时没有遇到问题，并寻求关于向模型提问的最佳实践的澄清。


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1267572093202206762)** (11 条消息🔥): 

> - `Wayland`
> - `Open Interpreter Installation`
> - `Pre-order Status`
> - `Building Parts`
> - `Poetry Version` 


- **对 Wayland 体验的担忧**: 一位成员表达了目前对 **Wayland** 的不满，将其归因于对该系统缺乏经验。
   - 这一见解突显了用户在迁移到新显示服务器时遇到的常见问题。
- **Open Interpreter 的安装环境**: 一位成员询问是否可以在一个虚拟环境中运行 **Open Interpreter** (OI)，同时在另一个环境中使用 **01**，特别是询问关于桌面版与设备版的区别。
   - 澄清安装实践可以帮助新用户高效地管理他们的环境。
- **预订状态问题**: 一位用户询问了当前的预订状态，并表示在网站上很难找到相关信息。
   - 回复澄清说目前已不再接受预订，并鼓励其他人自行采购零件来组装设备。
- **构建资源的访问问题**: 一位用户报告在点击与构建资源相关的共享链接时遇到“无权访问”的消息。
   - 管理员建议，需要在 Discord 服务器中为自己分配 **builder role**（构建者角色）才能获得访问权限。
- **关于 Poetry 版本的讨论**: 一位成员就社区目前使用的 **Poetry** 版本寻求建议。
   - 这表明开发者之间对资源和兼容性信息存在持续需求。


  

---

### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1268004908183982183)** (2 条消息): 

> - `Perplexica`
> - `Llama-3.1`
> - `Open source alternatives` (开源替代方案)


- **Perplexica：你的新搜索伙伴**：一段名为 ["Perplexica + Llama-3.1 (405B, 70B, 8B) : This LOCAL & FREE CLONE of Perplexity BEATS Everyone!"](https://www.youtube.com/watch?v=V0vx94JYNjI) 的 YouTube 视频展示了如何使用 Meta AI 的开源 Llama-3 构建一个本地且免费的 Perplexity 和 SearchGPT 替代方案。
   - 该视频强调了在自托管（self-hosted）设置中安装和访问这些强大工具的便捷性。
- **GitHub 上的 Perplexica：一个开源解决方案**：[GitHub 上的 Perplexica](https://github.com/ItzCrazyKns/Perplexica) 提供了一个 Perplexity AI 的开源替代方案，被设计为一个 AI 驱动的搜索引擎。
   - 其仓库详情概述了 Perplexica 的功能和潜在用途，对于希望增强搜索能力的开发者来说，这是一个宝贵的资源。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=V0vx94JYNjI">Perplexica + Llama-3.1 (405B, 70B, 8B) : This LOCAL &amp; FREE CLONE of Perplexity BEATS Everyone!</a>：在这段视频中，我将告诉你如何利用 Meta AI 新发布的开源 Llama-3 来搭建一个本地且免费的 Perplexity 和 SearchGPT 替代方案……</li><li><a href="https://github.com/ItzCrazyKns/Perplexica">GitHub - ItzCrazyKns/Perplexica: Perplexica is an AI-powered search engine. It is an Open source alternative to Perplexity AI</a>：Perplexica 是一个 AI 驱动的搜索引擎。它是 Perplexity AI 的开源替代方案 - ItzCrazyKns/Perplexica
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1267568219997208669)** (12 条消息🔥): 

> - `View Merging Task` (View 合并任务)
> - `Shape Tracker Reduction` (Shape Tracker 规约)
> - `YouTube Talk on Parallel Computing` (关于并行计算的 YouTube 演讲)
> - `OpenCL Resource Errors` (OpenCL 资源错误)


- **关于 View 合并任务的澄清**：该任务是证明 `View.__add__` 可以合并任何两个可合并的 view，或者如果失败了，则修改它使其工作并提供证明。
   - 在某些情况下，view 并非成对可合并的，但仍然可以规约 shape tracker，这引发了额外的澄清。
- **识别可合并的 View**：悬赏发布者可能会将任务表述为确定并证明两个 view 何时可合并，重点在于定义的清晰度。
   - 目标是尽量减少 view 的数量，以减少最终的索引计算时间，确保最佳性能。
- **关于并行计算的 YouTube 演讲**：一位成员分享了一段名为 [“I want a good parallel computer - UCSC Colloquium”](https://youtu.be/c52ziyKOArc?si=pAUdzwIQGXCtpk3T) 的 YouTube 视频，这是一场关于并行计算的精彩演讲。
   - 该录制的演讲于 2024 年 4 月 10 日在加州大学圣克鲁兹分校（UC Santa Cruz）CSE 研讨会上举行，幻灯片也已公开。
- **生成 OpenCL 资源错误的挑战**：一位成员表示不确定如何在 Mac 上使用 OpenCL 生成“资源耗尽”（out of resources）错误，并报告只收到了“无效内核”（invalid kernels）。
   - 看起来这些错误可能与编译问题有关，而不是运行时资源限制。



**提到的链接**：<a href="https://youtu.be/c52ziyKOArc?si=pAUdzwIQGXCtpk3T">I want a good parallel computer - UCSC Colloquium</a>：这是我在 2024 年 4 月 10 日加州大学圣克鲁兹分校 CSE 研讨会上所做演讲的视频。幻灯片可以在这里找到：https://docs.google.com/presentation/d...

  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1267883182108180582)** (13 条消息🔥): 

> - `Gradients in Training Loop` (训练循环中的梯度)
> - `Using TinyJit` (使用 TinyJit)
> - `Jitting the Step Function` (对 Step 函数进行 Jit)


- **Jitting 后梯度变为 None**：一位成员报告说，在使用 TinyJit 后，训练循环的第三步中所有张量返回的梯度均为 **None**，而前两步是正常的。
   - *TinyJit 在第 3 步开始生效* 可能会导致此问题，这促使该成员进行实验，并最终通过移除它解决了问题。
- **Jit 与梯度的潜在问题**：关于 jitting 是否会影响梯度进行了讨论，一位成员表示不确定并怀疑可能是操作不当（skill issue）。
   - 另一位成员建议，也许 **optim.step()** 位于 jitted 函数之外，这证实了问题的所在。
- **决定对整个 Step 函数进行 Jit**：一位成员考虑是仅对模型的 forward 步骤进行 jit，还是对整个 step 函数进行 jit。
   - 建议是，除非有特定原因不这样做，否则通常最好对整个 step 函数进行 jit。

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1267834416567160923)** (15 messages🔥): 

> - `Apple's AI Model Training` (Apple 的 AI 模型训练)
> - `Tim Dettmers' New Role` (Tim Dettmers 的新角色)
> - `Sewon Kim's Recruitment` (Sewon Kim 的招聘)
> - `TPUs Usage` (TPU 使用情况)
> - `Job Market Insights` (就业市场洞察)


- **Apple 声称未使用 NVIDIA GPU**：Apple 表示，它没有使用 **NVIDIA GPU** 来训练其 AI 模型，而是选择了 **TPU**，正如[最近的一篇文章](https://www.reuters.com/technology/apple-says-it-uses-no-nvidia-gpus-train-its-ai-models-2024-07-29)所指出的。成员们注意到 Apple 是市场上 **TPU 的第二大用户**。
- **Tim Dettmers 加入 Allen Institute**：在求职市场奔波七个月后，**Tim Dettmers** 宣布加入 **Allen Institute**，并将于 2025 年秋季成为 **Carnegie Mellon University** 的教授。他的目标是加强开源贡献以解决现实世界的问题，同时继续维护 **bitsandbytes**。
   - 成员们讨论了对 Dettmers 才华的竞争，强调了来自 **Anthropic** 和 **Hugging Face** 等大公司的兴趣。
- **Sewon Kim 吸引了极高的人才关注**：成员们庆祝了 **Sewon Kim** 的入职，强调许多公司都在争取他。讨论强调了拥有**独特且合理的方案**对于吸引顶尖人才的价值。
- **就业市场洞察**：Tim Dettmers 分享了他在学术就业市场的时间，涉及 **17 所大学**的 **125 场面试**，最终获得了 **15 份工作邀约**。他承诺很快会分享这次经历的见解和教训。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/tim_dettmers/status/1818282779488227575?s=46">Tim Dettmers (@Tim_Dettmers) 的推文</a>：在学术就业市场的六个月非常残酷，但也非常成功。在 17 所大学进行了超过 125 场个人面试，获得了 15 份工作邀约。这是一次独特的经历...</li><li><a href="https://x.com/Tim_Dettmers/status/1818282778057941042">Tim Dettmers (@Tim_Dettmers) 的推文</a>：在求职市场 7 个月后，我很高兴地宣布：- 我加入了 @allen_ai - 2025 年秋季起担任 @CarnegieMellon 教授 - 新的 bitsandbytes 维护者 @Titus_vK 我的主要重点将是加强...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1267609295881769102)** (6 messages): 

> - `Zuck's Stage Performance` (Zuck 的舞台表现)
> - `Perplexity Publishers Program` (Perplexity 出版商计划)
> - `OpenAI Cookbook Controversy` (OpenAI Cookbook 争议)
> - `Email Open Rates Decline` (邮件打开率下降)
> - `iCloud Private Relay Issues` (iCloud Private Relay 问题)


- **Zuck 在 SIGGRAPH 与 Jensen 同台时爆粗口**：在 SIGGRAPH 亮相期间，**Zuck** 与 **Jensen** 同台时因说脏话（f-bombs）成为头条新闻，展现了更坦率的一面。
   - *“再给我做一个芝士牛排堡，Jensen，”* 是一句幽默的评论，在严肃的讨论中引起了共鸣。
- **Perplexity 推出出版商计划**：Perplexity 宣布了其 **Publishers Program**，旨在通过**收入分成**和提高受众参与度的技术来支持媒体机构。
   - 著名的合作伙伴包括 **TIME** 和 **Der Spiegel**，目标是提高响应答案中所使用来源的质量。
- **OpenAI Cookbook 面临法律压力**：围绕 **OpenAI Cookbook** 展开了讨论，有报道称法律诉讼威胁迫使其重新考虑未来的合作伙伴关系。
   - 潜在的后果可能会显著影响项目的方向和社区的参与。
- **调查邮件打开率下降的原因**：自 **7/24** 以来，**邮件打开率**出现了明显的下降，引发了对其原因的调查。
   - 团队认为这种下降与 **Apple iCloud Private Relay** 的故障有关，而不是用户参与度的降低，这特别影响了 Apple Mail 用户。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.perplexity.ai/hub/blog/introducing-the-perplexity-publishers-program">介绍 Perplexity 出版商计划</a>：从第一天起，我们就在每个答案中包含引用，确保出版商获得应有的归属并建立用户信任。</li><li><a href="https://substack.com/@substackwriters/note/c-63739022?r=68gy5&utm_medium=ios&utm_source=notes-share-action">Substack 上的 Substack 作者</a>：我们调查了自 7/24 以来一部分出版商邮件打开率下降的趋势。我们认真对待这些担忧，并认识到这些指标在您建立业务时的重要性...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/)** (1 messages): 

xeophon.: https://152334h.github.io/blog/scaling-exponents/
  

---

### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1267894523531362314)** (2 条消息): 

> - `OPTO 和 Trace`
> - `游戏史中的 AI`
> - `神经网络的演进`
> - `微软的 AI 创新` 


- **在 Trace 框架中探索 OPTO**：一名成员对 [Trace 使用的 OPTO](https://www.microsoft.com/en-us/research/blog/tracing-the-path-to-self-adapting-ai-agents/) 表示好奇，并强调了其在 AI 应用中的意义。
   - 讨论强调了人们对自适应 AI 技术日益增长的兴趣，特别是在游戏行业。
- **游戏作为 AI 创新的前沿**：对话指出，游戏行业历来是 AI 创新的前沿，可以追溯到 2000 年代初期为 [虚拟世界](https://galciv3.fandom.com/wiki/History_of_the_Galactic_Civilizations_franchise) 编写的神经网络程序。
   - 这种演进促成了引人入胜的 AI 角色的开发，显著提升了玩家的互动体验。
- **神经网络的增长**：对话提到了神经网络从简单模型到拥有**数十亿参数**的复杂系统的进步，这些系统现在为 [ChatGPT](https://arxiv.org/abs/2303.08774) 等应用提供动力。
   - 这些发展改变了 AI 能力的格局，实现了更复杂的现实世界应用。
- **微软在 AI 进步中的角色**：提到了微软的举措，包括 [Copilots](https://www.bing.com/chat?q=Microsoft+Copilot&FORM=hpcodx)，它们利用先进的 AI 能力来增强功能。
   - 他们的创新被视为在更广泛的 AI 扩展和应用效率背景下的关键。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.microsoft.com/en-us/research/blog/tracing-the-path-to-self-adapting-ai-agents/">发现 Trace，一个从语言模型到机器人控制的 AI 优化新框架</a>：介绍 Trace，这是微软和斯坦福大学开发的新型 AI 优化框架，现已作为 Python 库提供。Trace 能够动态适应并优化广泛的应用，从...</li><li><a href="https://www.microsoft.com/en-us/research/blog/tracing-the-path-to-se">发现 Trace，一个从语言模型到机器人控制的 AI 优化新框架</a>：介绍 Trace，这是微软和斯坦福大学开发的新型 AI 优化框架，现已作为 Python 库提供。Trace 能够动态适应并优化广泛的应用，从...
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/116151969319946286/1267683115674107935)** (11 条消息🔥): 

> - `MIRPO 更新`
> - `答案评分中的惩罚指标`
> - `DSPy 与 Semantic Kernel`
> - `针对“名人”示例的优化器`
> - `惩罚项中的均方误差 (Mean Squared Error)` 


- **MIRPO 与 dspy 函数的兼容性**：成员们询问 **MIRPO** 是否已更新以支持 **dspy.Suggest** 和 **dspy.Assert**，此前的一个问题表明它尚不支持。
   - 目前还没有关于该功能是否已实现的明确说明。
- **为答案偏差创建惩罚指标**：讨论了开发一种根据与标准答案（gold answer）的距离施加更高惩罚的指标，强调了比例惩罚的概念。
   - *一位成员建议使用涉及预测分数与标准分数之差的平方公式*来实现这一效果。
- **DSPy 与 Semantic Kernel 的兼容性**：**DSPy** 用户对其与 **Semantic Kernel** 的互操作性感到好奇，并正在评估潜在的集成。
   - 聊天中尚未分享关于此兼容性的具体更新或确认。
- **关于优化模型惩罚的见解**：（针对评分指标）**Mean Squared Error**（均方误差）被推荐作为惩罚机器学习中较大误差的常规方法。
   - 一位成员详细说明了如何在 **DSPy** 中使用负惩罚指标来调整分数，并解释了如何有效地优化以降低惩罚。
- **关于语言模型的 ICML 演讲**：一位成员分享了关于 **语言模型“物理学”的 ICML 演讲** 的见解，建议优化器可以使用“名人（celebrity）”示例的潜力。
   - 演讲链接已提供在 [此处](https://youtu.be/YSHzKmEianc) 以供进一步观看。


  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/)** (1 条消息): 

batmanosama: https://github.com/ax-llm/ax
  

---

### **AI21 Labs (Jamba) ▷ #[announcements](https://discord.com/channels/874538902696914944/874538945168408606/1267598282008563754)** (1 条消息): 

> - `Long Context Use Cases`（长上下文用例）
> - `Developer Collaboration`（开发者协作）
> - `Enterprise Customer Feedback`（企业客户反馈）


- **寻找长上下文项目开发者**：团队正在寻找开发者，利用 Jamba 的 **256k 有效长度** 协助处理长上下文用例，旨在根据企业客户的反馈改进结果。
   - 他们邀请正在实验长上下文的人员分享见解，并提供 **Credits、周边礼品和名望** 作为激励。
- **企业客户反馈结果喜人**：来自企业客户的早期反馈表明，在探索 Jamba 的能力时取得了**令人期待的结果**。
   - 消息强调了收集更多见解并加强该领域协作工作的愿望。


  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1267633991289147452)** (4 条消息): 

> - `Joining Discord`（加入 Discord）
> - `Learning about Jamba`（了解 Jamba） 


- **Discord 里的新面孔**：新成员 **artworxai** 宣布加入 Discord。
   - *抱歉在你回复之前我下线了！*
- **对 Jamba 的兴趣**：**artworxai** 表示他们加入 Discord 是为了了解 **Jamba**。
   - 这凸显了新成员对于探索该平台功能和见解的兴趣。


  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1267620727612379177)** (2 条消息): 

> - `SWE-Bench Ultra-Hackathon`（SWE-Bench 超级黑客松）
> - `Segment Anything Model 2` 


- **SWE-Bench 超级黑客松挑战代码生成极限**：一项大胆的实验正在进行中，即为期 **6 天的 SWE-Bench 超级黑客松**，@StrongCompute 为每位参与者提供 $1,000 的算力支持。
   - 改进基准测试或击败现有基准的团队将获得奖品，演讲嘉宾包括合著者 [John Yang](https://x.com/jyangballin)、[Carlos E. Jimenez](https://x.com/_carlosejimenez) 和 [Ofir Press](https://x.com/OfirPress)。
- **GitHub 托管 Segment Anything Model 2 代码库**：[Segment Anything Model 2 (SAM 2)](https://github.com/facebookresearch/segment-anything-2) 的 GitHub 仓库提供了运行推理的代码，以及训练好的模型 Checkpoints 和示例 Notebook。
   - 该资源旨在简化 SAM 2 在开源代码项目各种分割任务中的使用。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/stevewattsfrey/status/1818033777622532518">Steve Frey (@stevewattsfrey) 的推文</a>：一个大胆的实验：我们正在举办为期 6 天的 SWE-Bench 超级黑客松，以挑战开源代码生成的极限 - 每个人都将获得由 @StrongCompute 提供的 $1,000 算力 - 多达 50 名研究人员...</li><li><a href="https://github.com/facebookresearch/segment-anything-2">GitHub - facebookresearch/segment-anything-2: 该仓库提供了使用 Meta Segment Anything Model 2 (SAM 2) 运行推理的代码、下载训练好的模型 Checkpoints 的链接，以及展示如何使用该模型的示例 Notebook。</a>：该仓库提供了使用 Meta Segment Anything Model 2 (SAM 2) 运行推理的代码，下载训练好的模型 Checkpoints 的链接，以及展示如何使用该模型的示例 Notebook...
</li>
</ul>

</div>
  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1267890375628947577)** (1 条消息): 

> - `Sentry`
> - `AutoFix feature`（AutoFix 功能） 


- **Sentry 的 AutoFix 功能演示**：来自 **Sentry** 的 Jenn 和 Ben 将在即将举行的会议中讨论他们的开源功能 **AutoFix**。
   - 活动详情可以在[这里](https://discord.com/events/1089876418936180786/1245836053458190438)找到。
- **Sentry 开源优势**：讨论将重点介绍开发者使用 AutoFix 等**开源**功能的好处。
   - 参与者可以期待获得关于这些功能的社区驱动支持和更新的见解。


  

---



---



---



---



---



{% else %}


> 完整的频道细分内容已在邮件中截断。
> 
> 如果您想查看完整细分，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前致谢！

{% endif %}