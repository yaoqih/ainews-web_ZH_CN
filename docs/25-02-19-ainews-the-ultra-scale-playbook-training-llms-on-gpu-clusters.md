---
companies:
- huggingface
- deepseek
- perplexity-ai
- google-deepmind
- microsoft
- baichuan
- stripedhyena
date: '2025-02-20T05:57:17.513081Z'
description: '**Huggingface** 发布了《超大规模指南：在 GPU 集群上训练大语言模型》（The Ultra-Scale Playbook:
  Training LLMs on GPU Clusters），这是一篇基于在多达 **512 块 GPU 上进行的 4000 次扩展实验**而撰写的交互式博客文章，提供了关于现代
  GPU 训练策略的详细见解。**DeepSeek** 推出了原生稀疏注意力（NSA）模型，引起了社区的极大关注；与此同时，**Perplexity AI** 推出了
  R1-1776，这是 DeepSeek R1 模型的一个无审查且无偏见的版本。**Google DeepMind** 发布了 PaliGemma 2 Mix，这是一款多任务视觉语言模型，提供
  **3B、10B 和 28B 三种尺寸**。**微软**（Microsoft）推出了 Muse，这是一款基于游戏《嗜血边缘》（Bleeding Edge）训练的生成式
  AI 模型，并展示了 Magma，这是一款在 UI 导航和机器人操控方面表现出色的多模态 AI 智能体基础模型。**Baichuan-M1-14B** 作为一款基于
  **20 万亿（20T）token** 训练的顶尖医疗大模型正式亮相，同时发布的还有一款采用 StripedHyena 2 架构的完全开源 40B 基因组建模模型。针对
  Muse，文中指出：*“创造属于你自己的游戏体验，这一天的到来将比你想象的更快。”*'
id: 627736a9-c551-42f4-ab08-e615b14cfbca
models:
- deepseek-native-sparse-attention
- r1-1776
- paligemma-2-mix
- muse
- baichuan-m1-14b
- stripedhyena-2
original_slug: ainews-the-ultra-scale-playbook-training-llms-on
people:
- eliebakouch
- nouamanetazi
- lvwerra
- thom-wolf
- proftomyeh
- alex-wang
- aravsrinivas
- _akhaliq
- _philschmid
- mervenoyann
- reach_vb
- arankomatsuzaki
- maximelabonne
title: '**超大规模实战手册：在 GPU 集群上训练大语言模型**'
topics:
- gpu-training
- scaling
- multimodality
- vision
- model-training
- foundation-models
- medical-llm
- genome-modeling
- robotic-manipulation
- interactive-content
---

<!-- buttondown-editor-mode: plaintext -->**读完这些仅需 2 天。**

> 2025/2/18-2025/2/19 的 AI 新闻。我们为你检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord（**211** 个频道，**6631** 条消息）。预计节省阅读时间（按 200wpm 计算）：**700 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

似乎是为了回应 [DeepMind 的《如何扩展你的模型》](https://buttondown.com/ainews/archive/ainews-how-to-scale-your-model-by-deepmind/)，Huggingface 突然发布了一个重量级的 GPU 版“博客文章”：[The Ultra-Scale Playbook: Training LLMs on GPU Clusters](https://huggingface.co/spaces/nanotron/ultrascale-playbook)。


![image.png](https://assets.buttondown.email/images/5e68bc3f-1cd8-4412-9b5b-157b5d60ff8d.png?w=960&fit=max)


对于想要直观、详细地了解现代训练限制以及在 GPU 上进行规模扩展策略的人来说，这是一个极佳的起点，它从第一性原理出发构建了现代最佳实践：


![image.png](https://assets.buttondown.email/images/2482f42c-d8f1-4aba-a980-57944e830701.png?w=960&fit=max)


更不用说这篇博文是交互式的，基于 4000 次扩展实验的真实数据，最高支持 512 个 GPU。

虽然对 AI Engineer 来说不是严格要求的，但对于任何想要快速掌握训练术语的人来说，这都是一个绝佳的起点。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

**AI 模型与发布**

- **DeepSeek 的 Native Sparse Attention (NSA) 模型**引起了广泛关注。[@eliebakouch](https://twitter.com/eliebakouch/status/1892276069891240286) 分享了关于它的**更多细节**，并向包括 [@Nouamanetazi](https://twitter.com/Nouamanetazi)、[@lvwerra](https://twitter.com/lvwerra) 和 [@Thom_Wolf](https://twitter.com/Thom_Wolf) 在内的团队表示祝贺。[@ProfTomYeh](https://twitter.com/ProfTomYeh/status/1892245518580932812) 提到他将在**直播研讨会中手绘 DeepSeek 的 Native Sparse Attention**，并分享了一张草图，随后宣布他将在 Alex Wang 主持的另一场研讨会中**通过画圆和线来解释 DeepSeek 论文**。[@hkproj](https://twitter.com/hkproj/status/1892107497369915535) 指出 **DeepSeek 发布论文让整个 ML 社区都为之瞩目**，凸显了其软实力。[@qtnx_](https://twitter.com/qtnx_/status/1891989260825153569) 提到 **command r 7b 是他们最喜欢的 Transformer 实现**。
- **Perplexity AI 发布了 R1-1776**，这是 DeepSeek R1 模型的一个**无审查、无偏见且基于事实**的版本，由 [@AravSrinivas](https://twitter.com/AravSrinivas/status/1891958274880139484) 宣布，他预告**本周和下周还会有更多酷炫的发布**。[@_akhaliq](https://twitter.com/_akhaliq/status/1891961543455031429) 也强调了这次发布，将其描述为**经过后训练以移除中国共产党审查的版本**。
- **Google DeepMind 发布了 PaliGemma 2 Mix**，这是一个开放的多任务视觉语言模型，能够执行**服装评判和物体计数**等任务，由 [@_philschmid](https://twitter.com/_philschmid/status/1892258568176320927) 宣布。[@mervenoyann](https://twitter.com/mervenoyann/status/1892267634923593760) 进一步详细介绍了 **PaliGemma 2 Mix**，强调了它在开放式提示词、文档理解以及分割/检测等视觉语言任务中的多功能性，提供 **3B、10B 和 28B 三种尺寸**。
- **Microsoft 推出了 Muse**，这是一个**在 Ninja Theory 的游戏《Bleeding Edge》上训练**的生成式 AI 模型，并发布了模型权重和代码，由 [@reach_vb](https://twitter.com/reach_vb/status/1892254399633801283) 分享，他表示这预示着**制作属于你自己的游戏体验将比你想象的更早到来**。
- **Baichuan-M1**，一个开源的 **SotA 医疗 LLM (Baichuan-M1-14B)**，在 **20T tokens** 上从头开始训练，专注于医疗能力，由 [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1892053102427300212) 宣布。
- [@maximelabonne](https://twitter.com/maximelabonne/status/1892256596136165535) 宣布了一个**完全开源的 40B 模型，用于跨生命所有领域的基因组建模和设计**，采用了新的 StripedHyena 2 架构。

**研究与论文**

- **Microsoft 发布了 Magma**，这是一个用于多模态 AI Agent 的基础模型，在 **UI 导航和机器人操作任务上达到了 SotA**。该模型在标注有 Set-of-Mark (SoM) 和 Trace-of-Mark (ToM) 的大规模数据集上进行了预训练，正如 [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1892059107479224384) 和 [@_akhaliq](https://twitter.com/_akhaliq/status/1892074819795005518) 所强调的。
- **Meta 发布了 NaturalReasoning**，这是一个包含 **280 万个挑战性问题**的野外推理（Reasoning in the Wild）数据集，由 [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1892054528473657669) 分享。[@jaseweston](https://twitter.com/jaseweston/status/1892041992127021300) 详细介绍了 **NaturalReasoning** 的发布，强调其 **280 万个具有挑战性且多样化的问题**需要多步推理，展示了**更陡峭的数据缩放曲线（data scaling curves）**以及自我训练（self-training）的潜力。
- **DeepMind 发布了 PaliGemma 2 Checkpoints**，专为 **OCR（光学字符识别）和字幕生成（Captioning）**等任务量身定制，模型尺寸从 3B 到 28B 不等，全部为开放权重并兼容 Transformers，正如 [@reach_vb](https://twitter.com/reach_vb/status/1892269416831717648) 所提到的。
- **Hugging Face 发布了《在 GPU 集群上训练 LLM 的超大规模指南》（Ultra Scale Playbook for Training LLMs on GPU Clusters）**，这是一本免费的开源书籍，涵盖了 5D 并行、ZeRO、快速 CUDA 内核以及计算与通信重叠等内容，基于为期 6 个月的缩放实验，由 [@reach_vb](https://twitter.com/reach_vb/status/1892276287039033473) 宣布。
- **"Cramming 1568 Tokens into a Single Vector and Back Again: Exploring the Limits of Embedding Space Capacity"** 是 [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1892059638943637960) 强调的一篇新论文。
- **"Revisiting the Test-Time Scaling of o1-like Models"** 论文由 [@_akhaliq](https://twitter.com/_akhaliq/status/1892071152215785526) 分享，该论文质疑这些模型是否真正具备推理时缩放（test-time scaling）能力。
- **"ByteDance presents Phantom: Subject-consistent video generation via cross-modal alignment"** 论文由 [@_akhaliq](https://twitter.com/_akhaliq/status/1892073250974216476) 分享。
- **"Sailor2: Sailing in South-East Asia with Inclusive Multilingual LLMs"** 论文由 [@_akhaliq](https://twitter.com/_akhaliq/status/1892247778291503296) 提及。
- **"Learning to Reason at the Frontier of Learnability"** 论文由 [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1892156610304438297) 强调，重点关注 LLM 中的课程学习（curriculum learning），并在强化学习（RL）中使用采样来提高可学习性。
- **"Is Noise Conditioning Necessary for Denoising Generative Models?"** 论文由 [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1892053059221717486) 分享，探讨了无需噪声调节（noise conditioning）的去噪生成模型，发现其性能下降平缓，有时甚至表现更好。

**工具与库**

- **LangChain 宣布了 LangGraph Studio 的 Playground 集成**，用于更快的 Prompt 迭代，允许用户直接查看 LLM 调用并在不重新运行整个 Graph 的情况下迭代 Prompt，根据 [@LangChainAI](https://twitter.com/LangChainAI/status/1892288249441505444) 的消息。他们还推出了 **Langchain MCP Adapters**，使 LangGraph Agent 能够即时连接到 MCP 生态系统中的数百个工具，以及用于引导 LangGraph.js Agent 模板的 **`npm create langgraph`**。
- **Modular 发布了 MAX 25.1**，这是一个重要的版本，支持新的 Agentic 工作流、Mojo 中的 GPU 自定义算子（custom ops）以及一个新的开发者内容门户，由 [@clattner_llvm](https://twitter.com/clattner_llvm/status/1892294758007263724) 宣布。
- **Together AI 宣布了 NVIDIA Blackwell GPU 的试用计划**，通过 Together GPU Clusters 为 8 个 AI 团队提供免费访问权限，以优化模型并加速训练，根据 [@togethercompute](https://twitter.com/togethercompute/status/1892256276576686131) 的消息。他们还强调了 **Scaled Cognition 在 Together GPU Clusters 上训练 APT-1**。
- **LM Studio 现在支持投机采样（speculative decoding）**，由 [@cognitivecompai](https://twitter.com/cognitivecompai/status/1892252216515293381) 宣布。
- **LlamaCloud EU** 是一项安全、合规的知识管理 SaaS 服务，确保数据完全驻留在欧盟境内，由 [@llama_index](https://twitter.com/llama_index/status/1892271183451869512) 宣布。
- 根据 [@_akhaliq](https://twitter.com/_akhaliq/status/1892260638191235244) 的说法，**Gradio** 被强调为当今构建大多数 AI 应用的首选工具。

**行业新闻与活动**

- **4月29日**是首届 **LlamaCon** 的举办日期，而 **Meta Connect 定于9月17-18日举行**，正如 [@AIatMeta](https://twitter.com/AIatMeta/status/1891969855043313945) 所宣布的那样。
- **LangChain 正在举办系列活动**，包括2月19日在纽约举行的晚间见面会，以及2月27日在亚特兰大举行的 AI 活动，消息来自 [@LangChainAI](https://twitter.com/LangChainAI/status/1892288795841876319) 和 [@LangChainAI](https://twitter.com/LangChainAI/status/1892301688138047842)。[@hwchase17](https://twitter.com/hwchase17/status/1892302989420794105) 也提到他将在**亚特兰大**。
- **纽约 AI Engineer Summit** 正在进行中，[@HamelHusain](https://twitter.com/HamelHusain/status/1892025630079787214) 提到他正为此身处纽约。

**AI Agents 与应用**

- **评估 AI Agents** 是 DeepLearningAI 与 Arize AI 合作推出的新短课重点，由 [@JohnGilhuly](https://twitter.com/JohnGilhuly) 和 [@_amankhan](https://twitter.com/_amankhan) 授课，涵盖了 AI Agent 性能的系统性评估与改进，由 [@AndrewYNg](https://twitter.com/AndrewYNg/status/1892258190546653392) 和 [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1892250339878818124) 宣布。
- **AI co-scientist** 是一个由 Google 使用 Gemini 2.0 构建的多 Agent AI 系统，旨在加速科学突破。[@omarsar0](https://twitter.com/omarsar0/status/1892223515660579219) 在详细的推文中介绍了其功能，如生成新颖假设、表现优于其他 SoTA 模型以及利用 test-time compute。
- **Weights & Biases 举办了 Multimodal AI Agents Hackathon**，吸引了超过 200 名创新者和 40 支团队参加，展示了在构建 AI Agents 过程中的创意与迭代，由 [@weights_biases](https://twitter.com/weights_biases/status/1892254937901396068) 提及。
- **Microsoft 刚刚发布了 MUSE** —— 一个在 Ninja Theory 的多人对战游戏《Bleeding Edge》上训练的生成式 AI 模型，用于游戏玩法，据 [@reach_vb](https://twitter.com/reach_vb/status/1892254399633801283) 报道。
- **能从游戏过程中学习并生成视觉效果的 AI** 被 [@yusuf_i_mehdi](https://twitter.com/yusuf_i_mehdi/status/1892244909484077354) 强调为游戏领域的新篇章。

**量子计算突破**

- **Microsoft 的量子团队**因一项突破受到 [@stevenheidel](https://twitter.com/stevenheidel/status/1892274904638058609) 的祝贺，[@yusuf_i_mehdi](https://twitter.com/yusuf_i_mehdi/status/1892258540460331228) 宣布这是**朝着让量子计算成为现实迈出的重大一步**，释放了解决当今计算机无法处理的问题的能力，可能重塑行业并加速科学发现。[@cognitivecompai](https://twitter.com/cognitivecompai/status/1892258078210375867) 向 [@satyanadella](https://twitter.com/satyanadella) 询问 **topological superconductor（拓扑超导体）是否真的是一种新的物质状态**。[@jeremyphoward](https://twitter.com/jeremyphoward/status/1892250092800791009) 也对该新闻做出了反应，引用道**“我们创造了一种全新的物质状态”**。

**梗与幽默**

- [@nearcyan](https://twitter.com/nearcyan/status/1892049572546904319) 表示，**每一个购买了 700 美元 AI pin 的人都被彻底坑了（rugged）**，该观点获得了大量关注。

---

# AI Reddit 热帖回顾

## /r/LocalLlama 回顾

**主题 1. o3-mini 取代 DeepSeek 成为今年 LLaMA 的领跑者**

- **[o3-mini 赢得了投票！伙计们，我们做到了！](https://i.redd.it/ogpvvrth70ke1.jpeg)** ([得分: 1688, 评论: 186](https://reddit.com/r/LocalLLaMA/comments/1isu4un/o3mini_won_the_poll_we_did_it_guys/)): **o3-mini** 在一项 Twitter 投票中以 **54%** 的得票率（共 **128,108** 票）击败了“手机大小模型”选项。Ahmad 的回复暗示该投票具有误导性，并鼓励为“o3-mini”投票，突显了社区通过点赞、引用和转发进行的积极参与。
  - **模型蒸馏（Model Distillation）讨论**：用户讨论了**蒸馏模型**的过程，即通过训练一个紧凑模型（学生）来模仿大型模型（教师），从而创建一个更小、更快的版本。虽然使用了知识蒸馏、剪枝（pruning）和量化（quantization）等技术，但 OpenAI 的闭源性质限制了直接从其模型进行蒸馏。
  - **对模型发布的怀疑**：对于 OpenAI 是否会发布开源模型存在怀疑，因为从历史上看，他们已经远离了开源模式。用户对“o3-mini 级别”等术语保持警惕，认为它可能达不到预期，或者是一个显著降级的版本。
  - **社区反应与讽刺**：社区对 Twitter 投票结果和可能发布的“手机大小”模型反应不一。评论强调了对这类模型实际效用和性能的讽刺与怀疑，一些用户幽默地指出，发布会可能会令人大失所望。

**主题 2. 配备 128 GB 统一内存的 AMD 笔记本挑战 Apple 的主导地位**

- **[配备 AMD 芯片的新款笔记本拥有 128 GB 统一内存（其中高达 96 GB 可分配为 VRAM）](https://www.youtube.com/watch?v=IVbm2a6lVBo)** ([评分: 482, 评论: 157](https://reddit.com/r/LocalLLaMA/comments/1isxhoy/new_laptops_with_amd_chips_have_128_gb_unified/)): **新款 AMD 笔记本**现在配备 **128 GB 统一内存**，允许将高达 **96 GB** 分配为 **VRAM**。
  - 讨论强调了配备 **128 GB 统一内存**的新款 AMD 笔记本的**性能和通用性**。像 **JustJosh** 和 **Dave2D** 这样的测评者赞扬了它们运行 **LLMs** 和 Linux 的能力，挑战了 Mac 在处理大模型的统一内存方面的统治地位。**b3081a** 提到通过特定配置运行 **vLLM** 以在这些设备上实现性能优化。
  - 与 Apple 设备的**价格和对比**是重要的讨论点。**Asus 128GB 版本**售价为 **$2799**，比同级别的 Apple 设备（**$4700**）更便宜。用户对价值和性能差异进行了辩论，一些人指出其相对于 **RTX 5090** 等高端 GPU 可能具有成本优势。
  - 讨论还涉及对 **Linux 支持**和潜在桌面应用的兴趣。**Kernel 6.14** 预计将为这些设备中的 NPUs 带来完整的 Linux 支持，用户对搭载这些芯片的 **Mini PCs** 和 **Framework 13 主板**表示出兴趣，讨论了在各种配置中共享 RAM 和统一内存的好处。


**主题 3. Gemini 2.0 卓越的带说话人标签的音频转录功能**

- **[Gemini 2.0 在带有说话人标签、精确到秒的时间戳的音频转录方面表现惊人；](https://i.redd.it/d3bl014yx2ke1.png)** ([评分: 387, 评论: 90](https://reddit.com/r/LocalLLaMA/comments/1it36b0/gemini_20_is_shockingly_good_at_transcribing/)): **Gemini 2.0** 在音频转录方面表现出色，具有精确的**说话人标签**和精确到秒的**时间戳**，正如 **Matt Stanbrell** 在 Twitter 上所强调的那样。该工具识别各种声音并提供详细转录的能力，鼓励用户上传音频文件以进行增强的摘要和说话人识别。
  - **Gemini 2.0** 的转录能力在**越南语准确度**（包括声调）方面受到称赞，用户发现它在语言学习方面非常可靠，如 **Mescallan** 所述。然而，来自一家 ASR 公司的 **leeharris100** 批评了其**时间戳准确性**，并提到它在长上下文下会产生幻觉，尽管它在通用 **WER**（词错率）方面与 **Whisper medium** 等模型相比仍具竞争力。
  - 存在一种共识，即 **Google 的 Gemini 模型**不是开源的，这阻碍了像 **Whisper** 那样的本地使用，正如 **CleanThroughMyJorts** 所强调的。**nrkishere** 和 **silenceimpaired** 对其本地运行能力和开源潜力表示怀疑。
  - **Gemini 2.0** 因其**物体识别**和**图表理解**能力而受到关注，像 **Kathane37** 这样的用户对其性能印象深刻。**space_iio** 将其有效性归功于 Google 能够访问 **YouTube 视频和元数据**，从而增强了其超越典型抓取方法的训练数据。


**主题 4. Unsloth 发布具有高准确度的 R1-1776 动态 GGUF**

- **Unsloth 发布的 R1-1776 动态 GGUF** ([评分: 132, 评论: 53](https://reddit.com/r/LocalLLaMA/comments/1it0ocl/r11776_dynamic_ggufs_by_unsloth/)): **Unsloth** 发布了从 **2-bit 到 16-bit** 的 **R1-1776 GGUFs**，包括 **动态 2-bit、3-bit 和 4-bit** 版本，其中 **动态 4-bit** 比 medium 版本更小但更准确。这些模型可在 **[Hugging Face](https://huggingface.co/unsloth/r1-1776-GGUF)** 上获取，需要特定的 token 格式，并在模型卡片中提供了说明，更多见解可在其 **[博客](https://unsloth.ai/blog/r1-reasoning)** 中找到。
  - **资源需求**：运行 **R1-1776 GGUFs** 并不一定需要 VRAM，但为了获得最佳性能，建议至少拥有 **120GB 的 VRAM + RAM**。**动态 2-bit 版本**需要 **211GB 的磁盘空间**，并提供了特定的格式指南以增强模型输出。
  - **模型性能和基准测试**：R1 模型的**动态量化 (Dynamic quants)** 在提交给 **Hugging Face 排行榜**的基准测试中显示出优于或等同于原始 **16-bit 模型**的性能。然而，用户指出除了 **Flappy Bird** 测试之外，还需要更全面的基准测试来全面评估性能。
  - **未来发展和发布**：即将发布的版本将专注于**长上下文**和超过 **10,000 名用户**请求的其他功能，这表明了强烈的社区参与度。此外，还有支持**蒸馏到更小模型**的计划，以及针对 **V3** 和 **V2.5-1210** 版本的潜在更新，以提高可访问性和性能。


## 其他 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**主题 1. DeepSeek GPU 走私调查：揭示 Nvidia 新加坡营收异常**

- **[DeepSeek GPU 走私调查显示 Nvidia 在新加坡的 GPU 销售额占其营收的 28%，但仅有 1% 交付至该国：报告](https://www.tomshardware.com/tech-industry/deepseek-gpu-smuggling-probe-shows-nvidias-singapore-gpu-sales-are-28-percent-of-its-revenue-but-only-1-percent-are-delivered-to-the-country-report)** ([Score: 286, Comments: 90](https://reddit.com/r/OpenAI/comments/1it5d6m/deepseek_gpu_smuggling_probe_shows_nvidias/)): **DeepSeek 的调查**显示，**Nvidia 在新加坡的 GPU 销售额**占其总营收的 **28%**，然而实际上只有 **1%** 的 GPU 交付到了该国。这种差异暗示了可能存在 **GPU 走私活动**。
  - **DeepSeek V3 论文**因声称使用 **8-bit floating point (FP8)** 而受到质疑，如果属实，这将是 AI 模型训练的一个重大转变。讨论显示了对 DeepSeek 说法真实性的怀疑，一些人认为这被媒体过度炒作了。
  - **新加坡作为主要贸易港口的角色**被重点提及，解释了尽管实际交付量较低，但 Nvidia 归属于该国的营收比例却很高。这与新加坡超过 **300%** 的贸易占 GDP 比率相符，表明这可能是转口贸易或加工活动，而非直接走私。
  - **Nvidia 的行为**被视为试图规避美国法律，一些评论者指出文章标题具有误导性，并强调了 GPU 的战略重要性。讨论涉及了更广泛的地缘政治影响，包括与中国对战略资产潜在控制的对比。


- **[准确](https://i.redd.it/tj7ei7ndq0ke1.jpeg)** ([Score: 286, Comments: 15](https://reddit.com/r/ChatGPT/comments/1iswa0c/accurate/)): 该帖子幽默地将严肃的医疗环境与随意的回答进行了对比，画面中一名正在进行 **MRI 扫描**的患者询问医生结果，而显示器上显示着 **"ChatGPT-4o"** 的 Logo。这种并置通过将传统的严肃语境与轻松、非正式的回复相结合，突显了喜剧色调。
  - **医疗 AI 辅助**：**Human-Independent999** 建议 AI 在医疗领域的最佳应用是辅助医生，暗示这种集成可能已经在发生。这反映了关于 AI 增强医疗实践潜力的普遍看法。
  - **未来科技对比**：**PerennialPsycho** 幽默地将这一场景比作《星际迷航》(**Star Trek**)，医生使用简单的工具进行诊断，突显了 AI 在医疗保健领域的未来主义和简化愿景。
  - **对 AI 未来的乐观态度**：**Potential_Club_9494** 对 AI 的未来影响表示乐观，暗示了像 **ChatGPT** 这样的 AI 可能对各个领域产生的变革性潜力。


- **[来自一位 OpenAI 研究员的推文](https://i.redd.it/vgyttoqy55ke1.jpeg)** ([Score: 273, Comments: 27](https://reddit.com/r/OpenAI/comments/1itd425/tweet_from_an_openai_researcher/)): **Aidan Clark** 发布了一条关于 "Sonnet 4" 的推文并祝贺 "Demis & Team"，截至 2025 年 2 月 19 日，该推文获得了 **1.27 万次查看**、**34 条回复**、**25 次转发**和 **239 个赞**，引起了广泛关注。
  - 关于 **"Sonnet 4"** 以及 **Demis Hassabis** 的参与存在困惑和猜测，一些用户质疑 **Demis** 是因为 **DeepMind** 的成就受到祝贺，还是因为 **Google** 对 **Anthropic** 的投资而产生的一种反讽（trolling）。
  - 讨论中热情地提到了 **Grok 4.5**，暗示对话中可能存在混淆或玩笑，因为 **Demis Hassabis** 与 **DeepMind** 相关，而非 **Anthropic**。
  - 几位用户对什么是 **"Sonnet 4"** 表示困惑，一些人幽默地建议这可能是一个跨宇宙或虚构的概念，表明原始推文缺乏清晰度或可能是在开玩笑。


**主题 2. Google 的 NotebookLM：AI 研究工具的变革者**

- **NotebookLM 最被低估的 AI 工具！** ([Score: 1143, Comments: 104](https://reddit.com/r/ChatGPT/comments/1isvvpr/notebooklm_the_most_underrated_ai_tool/)): Google 的 **NotebookLM** 被强调为一个被低估的 AI 工具，它结合了 **ChatGPT, Perplexity** 和 **Notion AI** 的功能，同时提供自动源引用以消除幻觉。它在阅读和总结 PDF、Docs 和笔记方面表现出色，并且比 **ChatGPT** 更好地记住上传的文件，使其成为处理大量数据的研究人员、学生和专业人士的宝贵工具。
  - 用户称赞 **NotebookLM** 独特的播客功能，该功能允许交互式和可定制的体验，将其比作拥有个人 AI 广播节目。然而，人们对潜在的幻觉（特别是在免费层级）以及 Google 停止该服务的可能性表示担忧。
  - 一些用户对 **NotebookLM** 管理和总结数据的潜力表示兴趣，例如 Excel 表格中的支出，但对其聊天功能和 UI 限制存在批评。该工具因其优于其他工具的源管理和引用能力而受到关注。
  - 用户渴望有一个开源、本地的 **NotebookLM** 替代方案，这反映了对 Google 长期支持该工具的担忧。用户还对定价模型及其与 **Google One AI** 的集成感到好奇。


- **['Improved Amateur Snapshot Photo Realism' v12 [FLUX LoRa] - 修复了过度饱和，略微改善了皮肤，提高了提示词遵循度和图像连贯性（20 张示例图）- 现已推出 Tensor.art 版本！](https://www.reddit.com/gallery/1it3acw)** ([Score: 239, Comments: 18](https://reddit.com/r/StableDiffusion/comments/1it3acw/improved_amateur_snapshot_photo_realism_v12_flux/)): **'Improved Amateur Snapshot Photo Realism' v12 [FLUX LoRa]** 是一个重大更新，解决了**过度饱和**问题，增强了**皮肤纹理**，并提高了**提示词遵循度**和**图像连贯性**。该更新包括 **20 张示例图像**，并引入了与 **Tensor.art** 兼容的版本。
  - **图像质量担忧**：像 **animerobin** 这样的用户指出，**FLUX LoRa** 模型经常受到低分辨率像素模糊的困扰，询问如何获得更清晰的图像。
  - **数据集来源**：**TheManni1000** 寻求关于寻找用于训练类似模型的图像数据集的建议，建议包括 **Reddit 图像子版块**、**Instagram**、**Flickr** 以及来自 **Hugging Face** 和 **Kaggle** 的数据集。
  - **资源链接与术语**：**AI_Characters** 提供了该模型在 [CivitAI](https://civitai.com/models/970862/improved-amateur-snapshot-photo-realism-style-lora-flux-spectrum0001-by-aicharacters) 和 **Tensor.art** 上的链接，同时围绕“amateer”一词展开了讨论，该词被认为是“AI 生成的名人”的流行语，对其一致的描述存在一些困惑。


**Theme 3. Claude 3.5 Sonnet: AI 编程与一致性的基准**

- **[Claude 推理功能。Anthropic 可能随时发布官方公告..](https://i.redd.it/bel9ndn6i2ke1.jpeg)** ([Score: 222, Comments: 87](https://reddit.com/r/ClaudeAI/comments/1it1uil/claude_reasoning_anthropic_may_make_offical/)): **Claude 3.5 Sonnet** 界面展示了“相机”、“照片”和“文件”按钮等功能，以及“选择风格”和为 PRO 用户切换“使用扩展思考”的选项。界面暗示了免费计划中的**每日消息限制**功能和“升级”选项，表明了增强功能的潜力。
  - 用户对 Claude 的 **API 定价**表示沮丧，一位用户指出改写几个段落每次花费 **$0.60**，导致他们尽管每日消息有限，仍更倾向于使用 Web 版本。**每日消息限制**是一个主要的痛点，用户感到使用受限。
  - 对 **Claude 3.5 Sonnet** 的更新存在怀疑，一些用户怀疑这仅仅是现有功能（如 **MCP servers**）的重新品牌化，而不是引入新功能。虽然注意到了增加的推理和潜在的 **web search** 功能，但用户仍对缺乏实质性改进持批评态度。
  - 用户报告称，尽管应用已更新，但某些人的 **iOS** 或 **Android** 上仍无法使用新功能，导致对推出的困惑。社区还批评 **Anthropic** 专注于更新而没有解决消息限制和实际增强等根本问题。

- **到底发生了什么？** ([Score: 226, Comments: 185](https://reddit.com/r/ClaudeAI/comments/1it6yij/what_the_fuck_is_going_on/)): 尽管没有采用 AI 自循环（AI self-looping）等新技术，**Claude 3.5 Sonnet** 因其与 **DeepSeek**、**O3** 和 **Grok 3** 等模型相比更卓越的代码可靠性和一致性而受到赞誉。作者指出，虽然这些新模型因其新颖的方法而备受关注，但 **Claude 3.5 Sonnet** 依然无与伦比，尽管它最近没有表现出显著的改进。
  - 许多用户认为 **Claude 3.5 Sonnet** 并非在所有领域都具有优势，因为它在处理编程竞赛题目和架构问题等特定任务时比较吃力，而 **O1** 和 **O3-mini** 在这些方面的表现更好。批评者强调，与 **ChatGPT** 相比，**Claude** 更高的上下文窗口（200k vs. 32k）可能是导致其性能差异的原因之一 ([Reddit 帖子](https://www.reddit.com/r/OpenAI/comments/1is2bw8/chatgpt_vs_claude_why_context_window_size_matters/))。
  - 一些用户对过分强调 AI 基准测试（Benchmarks）和排行榜表示怀疑，认为它们可能无法准确反映现实世界的可用性和智能。他们认为，仅仅关注基准测试性能可能会削弱 AI 模型的实际效用和用户友好性。
  - 人们对未来的发布充满期待，例如 **Claude 4.0** 和 **Opus 4.0**，预计它们将超越当前模型；而其他人则指出，对于需要不同智能维度的任务，目前更倾向于使用 **Gemini Pro 2.0** 以及像 **R1** 和 **O3** 这样的推理模型。一些用户提到了即将推出的具有推理能力的混合模型，暗示了 AI 自循环和推理方面的进展。


**主题 4. OpenAI 的 4o 模型：在创意写作和叙事连贯性方面表现出色**

- **4o 的创意写作太惊人了！** ([Score: 135, Comments: 53](https://reddit.com/r/OpenAI/comments/1isu7bw/4o_creative_writing_is_phenomenal/)): 该帖子讨论了 **OpenAI 4o 模型** 及其在创意写作方面的卓越能力，特别是在**科幻和奇幻**等类型中。作者强调了该模型在延续系列丛书时保持角色一致性的能力，并指出 Pro 订阅提供的 **128k 上下文** 可能产生了影响。
  - 用户对 **OpenAI 4o 模型** 保持角色一致性和自然延伸故事的能力印象深刻，尤其是在 **128k 上下文** 的支持下。然而，一些用户发现该模型有时会默认使用程式化的叙事方式（特别是在恐怖等类型中），并且在编程任务中表现不佳，需要人工干预才能生成准确的代码。
  - 该模型最近的更新（包括 **1 月底的微调** 和 **2 月中旬的内容过滤器放宽**）被认为增强了其自然语言处理能力。一些用户建议，**OpenAI 的推理模型** 可能会在底层进行集成以提高性能。
  - 使用该模型进行创意写作被认为具有娱乐性和个性化，用户分享了诸如使用**叙事技巧**来引导故事发展等策略。此外，还有关于模型处理大输出容量能力的讨论，并提到了像 **Flash Thinking** 这样具有更高 Token 限制的替代方案。


**主题 5. SFW Hunyuan Video LoRAs：扩展创意 AI 视频应用**

- **我将训练并开源 50 个 SFW Hunyuan Video LoRAs。欢迎提需求！** ([Score: 144, Comments: 161](https://reddit.com/r/StableDiffusion/comments/1isrnoe/i_will_train_opensource_50_sfw_hunyuan_video/)): 作者计划利用近乎无限的算力训练并开源 **50 个 SFW Hunyuan Video LoRAs**。他们征集训练创意，承诺将优先考虑点赞数最高的请求和个人偏好。
  - 用户对**武术打斗场景**和**浓郁的电影胶片风格**表现出浓厚兴趣，并建议探索**超广角镜头**和**不同的武术风格**以实现多样化的视觉效果。**Wes Anderson** 美学和**黑色电影（Film Noir）**镜头也是独特视觉叙事的热门请求。
  - 一些评论者强调了将**微调模型**推向极限以获得高质量输出的潜力，建议的主题包括 **360 度角色转场**、**VR 和 SBS 3D 视频**以及**沉浸式实景 GoPro 画面**。其他人则建议专注于**独特的视频训练**，如镜头移动和动作序列，而非静态图像。
  - 针对主题内容有一些创意建议，包括 **Cyberpunk 2077** 和 **Blade Runner 2049** 等**赛博朋克设定**、**暗黑奇幻风格**以及**外星人传说**。此外，人们还有兴趣捕捉具有细腻情感表达的**电影对话瞬间**，以及动漫之外的 **2D 动画风格**。

- **[Anthropic 即将发布推理模型及其他酷炫功能..](https://i.redd.it/all95og7t1ke1.jpeg)** ([Score: 218, Comments: 44](https://reddit.com/r/ClaudeAI/comments/1iszwxj/anthropic_to_release_reasoning_and_other_cool/)): **Anthropic** 准备为 **Claude iOS app** 发布新功能，重点在于 **Thinking Models** 和 **Web Search** 能力。该公告最初由 **Twitter** 用户 **@M1Astra** 发布，包含了“Steps”、“Think”和“Magnifying Glass”等功能的新图标，发布日期为 **2 月 19 日**。
  - 用户表达了希望 **Claude** 解决 **rate limits**、内存和更长对话问题的愿望，如果这些功能得到改进，一些人会考虑重新订阅。**Web search** 能力也是一个呼声极高的功能，用于获取最新的在线信息。
  - 一些评论者（如 **ChrisT182**）强调 **Anthropic 对 AI safety 和理解的关注** 是其更新频率较低的原因，并认为这些基础工作对于未来迈向 **AGI** 至关重要。
  - 还有人期待 **Claude's voice** 等新功能，**Hir0shima** 等人对此表示期待，而一些用户由于对当前产品不满已转向 **DeepSeek** 等替代方案。


- **[ChatGPT 创始人分享完美提示词模板的剖析](https://i.redd.it/1rr5tpfc44ke1.jpeg)** ([Score: 1232, Comments: 62](https://reddit.com/r/ChatGPT/comments/1it7t6w/chatgpt_founder_shares_the_anatomy_of_the_perfect/)): **Greg Brockman** 分享了一条推文，详细介绍了 **ChatGPT** 最佳提示词的结构，重点是创建 **New York City** 附近的独特周末度假方案。该提示词包含“Goal”、“Return Format”、“Warnings”和“Context dump”等部分，以确保生成响应的清晰度和有效性。
  - **提示词结构与有效性**：关于 **prompt structure** 的有效性存在讨论，一些用户注意到将目标或问题放在顶部可能会导致不太理想的响应，而另一些人则认为将其放在底部或使用数据分隔符可以改善结果。**ArthurParkerhouse** 建议使用三井号分隔符来区分上下文和任务，这与 **MemeMan64209** 观察到的“将问题放在最后效果更好”的结论一致。
  - **AI 交互与用户体验**：**Fit-Buddy-9035** 将向 AI 提问比作与逻辑严密、直率的人交流，强调了清晰和明确的必要性；而 **Professional-Noise80** 则强调编写有效的提示词需要批判性思维。**TheSaltySeagull87** 指出，创建详细提示词所付出的努力与使用 Google 或 Reddit 等传统研究方法相当。
  - **AI 模型中的认知处理**：**MaintenanceOk3364** 认为 AI 模型与人类一样会优先处理初始信息，但 **MemeMan64209** 和 **ArthurParkerhouse** 观察到 AI 可能会优先处理最后读取的 token，这反映了对 AI 如何处理信息感知的差异。这突显了 AI 认知处理的复杂性和多变性，引发了关于最佳提示词设计的进一步辩论。

---

# AI Discord 简报

> 由 o1-preview-2024-09-12 生成的摘要之摘要的摘要

**主题 1：Grok 3 成为焦点，反应褒贬不一**

- **Grok 3 让服务器和大脑都“熔化”了！**：xAI 的 **Grok 3** 现已免费开放，直到达到服务器容量上限，提供了前所未有的权限来访问这款“全球最聪明 AI”，正如[这条推文](https://x.com/xai/status/1892400129719611567)所宣布的那样。用户们正在推测使用限制以及潜在的服务器过载。
- **Grok 3 的热度遭遇质疑**：虽然一些用户称赞 **Grok 3** 优于 **ChatGPT-4**，但也有人认为其推理能力与 **Claude 3.5** 和 **O1** 等模型相比差强人意。社区里充满了关于其真实实力的对比和争论。
- **Elon 进军游戏领域并结合 Grok 3**：**Elon Musk 的 xAI** 宣布成立一个与 **Grok 3** 挂钩的新游戏工作室，标志着将 AI 与游戏整合的战略举措，正如 [Elon 的推文](https://x.com/elonmusk/status/1891388509191049307)中所暗示的那样。

**主题 2：AI CUDA Engineer 加速算子优化**

- **AI 为 AI 编写代码：CUDA Kernel 获得提升**：**Sakana AI** 推出了 [**AI CUDA Engineer**](http://sakana.ai/ai-cuda-engineer/)，实现了优化 CUDA Kernel 创建的自动化，并比标准 PyTorch 操作实现了 **10-100 倍的加速**。
- **Kernel 魔法令社区印象深刻**：AI CUDA Engineer 在将 PyTorch 转换为 CUDA 方面拥有 **90% 的成功率**，并发布了一个包含 **17,000 个经过验证的 CUDA Kernel** 的数据集，标志着 AI 驱动的性能优化取得了突破。
- **Jim Fan 博士赞赏自主编程 Agent**：在[一条推文](https://x.com/DrJimFan/status/1892404919480832259)中，**Jim Fan 博士** 称赞 AI CUDA Engineer 是“最酷的自主编程 Agent”，强调了其通过增强 CUDA Kernel 来加速 AI 的潜力。

**主题 3：新 AI 实验室与量子计算进展震撼业界**

- **Mira Murati 创立新 AI 实验室**：前 OpenAI CTO **Mira Murati** 推出了 [**Thinking Machines Lab**](https://thinkingmachines.ai)，旨在开发更易理解、更可定制的 AI 系统，并承诺对公众透明。
- **微软凭借 Majorana 1 实现量子飞跃**：**Microsoft** 推出了 [**Majorana 1**](https://azure.microsoft.com/en-us/blog/quantum/2025/02/19/microsoft-unveils-majorana-1-the-worlds-first-quantum-processor-powered-by-topological-qubits/)，这是首个由拓扑量子比特（topological qubits）驱动的量子处理单元（QPU），有可能在单个芯片上扩展到一百万个量子比特。
- **Lambda 获得 4.8 亿美元融资以助力 AI 云服务**：**Lambda** 获得了巨额 **4.8 亿美元 D 轮融资**，**NVIDIA** 和 **ARK Invest** 参投，突显了 AI 基础设施投资的蓬勃发展。

**主题 4：AI 审查引发辩论；无审查模型发布**

- **Perplexity 开创后审查时代 AI**：**Perplexity AI** 发布了 [**R1 1776**](https://huggingface.co/perplexity-ai/r1-1776)，这是 **DeepSeek R1** 模型的无审查版本，力求提供公正且事实的信息。
- **强化量化版 DeepSeek R1 发布**：用户现在可以使用 [**动态 2-bit GGUF 量化**](https://huggingface.co/unsloth/r1-1776-GGUF) 运行 **Perplexity 的无审查版 DeepSeek R1**，承诺比标准格式具有更高的准确度。
- **用户要求 AI 拥有言论自由**：用户对 **ChatGPT** 和 **Grok** 等模型中严重的审查制度感到愈发沮丧，纷纷寻求允许更多创意和无限制互动的替代方案。

**主题 5：AI 变革游戏与创意表达**

- **AI 渲染改变游戏设计**：爱好者们讨论了 AI 通过动态、交互式环境彻底改变游戏的潜力，呼应了 **Satya Nadella** 在[这条推文](https://fxtwitter.com/satyanadella/status/1892244164814725387)中关于 AI 生成世界的愿景。
- **用于创意写作和角色扮演的 AI 蓬勃发展**：用户分享了使用 AI 模型增强成人角色扮演（ERP）的高级技术，重点在于创建详细的角色和沉浸式体验，突显了 AI 的创意潜力。
- **Anthropic 为 Claude 准备新功能**：据报道，**Anthropic** 正在升级 **Claude**，增加网页搜索和新的“Paprika 模式”以增强推理能力，这在最近的应用更新和[推文](https://x.com/M1Astra/status/1892091124589920532)中有所体现。

---

# 第一部分：Discord 高层级摘要

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 通过 Speculative Decoding 增强推理性能**：LM Studio **0.3.10** 引入了 **Speculative Decoding**，该技术将大模型与更小、更快的草稿模型（draft model）配对，可能使推理速度翻倍，特别是在 **llama.cpp/GGUF** 和 **MLX** 中，详见 [发布博客文章](https://lmstudio.ai/blog/lmstudio-v0.3.10)。
   - 用户报告的结果各不相同，部分用户达到了超过 **100 tokens/sec** 的速度，而其他用户则因配置和模型选择的不同经历了较慢的性能。
- **文本嵌入（Text Embeddings）实现 LM Studio 长期记忆**：**Text embeddings** 被用于直接在 **LM Studio** 中从用户创建和上传的文件中检索相关数据，作为在 **LLMs** 中存储信息以供长期使用的一种方法。
   - 这种方法不同于 LLM 的扩展，因为 embeddings 促进了特定数据的检索，从而增强了 LLM 的上下文（context）。
- **模型微调（Fine-Tuning）：谨慎操作**：**Fine-tuning** 允许模型适应特定上下文，但如果执行不当，可能会导致幻觉（hallucinations），尤其是在处理知名角色或概念时。
   - 社区成员讨论了使用不恰当的训练示例使模型产生偏见的风险，这可能会严重扭曲模型的响应。
- **A6000 GPU 验证微调实力**：配备 **256 GB** 显存的 **A6000 GPU** 被证实足以微调 **phi4 model**；用户被引导至 [Unsloth's notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks) 获取微调示例。
   - 讨论强调了易于获取的硬件对 AI 开发日益增长的重要性，A6000 系列在性能和成本效益之间取得了平衡。
- **Mistral 模型在法语任务中表现出色**：用户强调 **Mistral** 模型是执行法语任务的最佳选择，同时还推荐了 **DeepSeek V2 Lite** 等模型，以便在没有 **GPU** 资源时获得更快的推理速度。
   - 这些建议反映了在模型准确性与计算效率之间取得平衡的需求，确保模型能够在不同的硬件配置下有效运行。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Grok 3 引发关注但仍有不足**：用户发现 **Grok 3** 在给予充足且具体的提示词（prompts）时，在详细脚本和艺术作品生成方面表现出色，超越了免费版本的 **ChatGPT**。
   - **Grok 3** 速度更快且审查制度较少，但据报道在逻辑推理方面不如更先进的模型。
- **AI 审查引发辩论**：用户对 **ChatGPT** 和 **Grok** 中的审查制度表示担忧，特别是针对某些内容类型，不同平台（如 **Perplexity** 和 **o1 models**）的审查程度各不相同。
   - 一位用户表达了挫败感，称这些审查问题显著阻碍了 AI 的创造力和交互能力。
- **Evo-2 从零开始编写基因组**：根据 [vittorio 的这条推文](https://x.com/iterintellectus/status/1892251343881937090?s=46)，**Evo-2** 这是一个在 **9.3 万亿 DNA** 碱基对上训练的大型 AI 模型，现在可以从零开始编写基因组，这可能会彻底改变生物学研究。
   - 这一进展提升了人们对肿瘤学和基因工程突破的乐观情绪，为医学研究和治疗开辟了新途径。
- **伦理 AI 交互需要尊重**：一位成员对与 AI 的交互如何反映人类行为表示担忧，强调需要建立**健康的边界**，并指出：*“我们对待 AI 的方式……很可能就是我们对待同胞的方式。”*
   - 这一观点引发了关于动物对待心理学与人类交互之间相似性的讨论，建议对 AI 进行伦理对待。
- **提示词清晰度修复 ChatGPT 的失误**：一位用户通过简化提示词改进了 **ChatGPT** 的功能，但在其公司服务器上问题依然存在；他们注意到在 **Playground** 中功能有所改善，但在公司服务器中仍有疑问。
   - 社区反馈强调，清晰、具体的提示词会产生更好的 AI 响应，其中一条回复敦促关注服务器输入并确保清晰度，以有效引导模型的输出。



---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **DeepSeek-V3 开启不限量模式！**：正如 [Windsurf AI 官方账号](https://x.com/windsurf_ai/status/1892322088507105561)所宣布的，**DeepSeek-V3** 现在对 **Windsurf Pro** 和 **Ultimate** 计划的用户**不限量**，消除了对 **prompt credits** 和 **flow action credits** 的担忧。
   - 社区成员对这一更新表示兴奋，并被鼓励去体验（*surf*）并充分利用这些新功能。
- **Matt Li 展示 MCP 内容**：Matt Li 重点展示了新的 **MCP** 内容和用例，引发了积极的互动和支持，详情见[此推文](https://x.com/windsurf_ai/status/1892394489588727985)。
   - 此外，还分享了一个演示，展示了 **MCP** 如何在 **Cascade** 中高效工作，阐明了 **MCP 的潜在用途**并提升了社区参与度。
- **Codeium Autocomplete 功能澄清**：讨论澄清了 Codeium 中 **autocomplete** 和 **supercomplete** 的区别，其中 **supercomplete** 可以建议整个函数，而 autocomplete 辅助单行代码。
   - 然而，这次讨论也暴露了文档清晰度的问题，以及部分成员在寻找自动安装信息时遇到的困难。
- **学生认为 Codeium 订阅物有所值**：成员们得出结论，对于开发 SaaS 项目的学生来说，Codeium 的 **$60** 订阅可能是值得的，因为这通常会带来更高的收益。
   - 一位成员指出，尽管订阅费用看起来很贵，但对于从事严肃开发项目的学生来说，这可以实现收支平衡。
- **Windsurf 性能面临挑战**：许多用户报告了 Windsurf 的问题，包括模型迭代期间的内部错误和文件编辑问题，这表明 context length 可能会影响性能。
   - 他们表示需要更高效的确认流程来改进编码工作流。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth AI 兄弟亮相 GitHub 采访**：**Unsloth AI** 团队在 [GitHub Universe 采访](https://www.youtube.com/watch?v=lyVxD0bJDOk)中重点介绍了他们的项目，展示了 Han 氏兄弟之间的协同作用以及他们的工作对 AI 开发的影响。
   - 充满热情的社区成员对采访中展现的清晰思路和激情表示赞赏。
- **利用免费 GPU 进行模型微调**：**Unsloth** 正在有效利用 Colab 等平台提供的免费 GPU，社区成员分享了访问这些资源的链接，以支持模型微调，特别是针对 **DeepSeek-R1** 等模型。
   - 社区成员探索了内容创作机会，以推广这些免费资源及其利用方式。
- **Med-R1 模型进军医疗领域**：新发布的 **med-r1 模型**拥有 **1B** 参数，并在医疗推理数据集上进行了训练，现已在 [Hugging Face](https://huggingface.co/Imran1/Med-R1-v1) 上可用。
   - 该模型专为医疗问答和诊断设计，支持 **4-bit** 推理，最大序列长度（max sequence length）为 **2048 tokens**。
- **社区攻克 Bitsandbytes 代码**：成员们仔细研究了 [bitsandbytes 代码](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/86b6c37a8ad448230cedb60753f63150b603a112/csrc/ops.cu#L86)，对实现的浮点类型和 block sizes 提出了担忧。
   - 讨论集中在代码理想情况下应使用 `fp32` 但实际出现 `fp16` 值的差异上，引发了关于指针转换的问题。
- **AI 提升植物化学精准度**：一个 AI 驱动的模型现在可以通过[一个新框架](https://www.marielandryceo.com/2025/02/title-ai-powered-phytochemical.html?m=1)系统地识别植物化合物，从而优化健康支持。
   - 这种方法旨在加强循证营养保健品的开发，同时优先考虑安全性和有效性。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **据报道 Grok 3 完胜 ChatGPT-4**：有消息称 **Grok 3** 的表现优于 **ChatGPT-4**，引发了对其能力的广泛关注。
   - 社区成员对这些模型之间的技术差异表达了好奇并进行了推测。
- **SWE-Lancer 基准测试评估 LLM 自由职业能力**：根据[这条推文](https://x.com/_philschmid/status/1891780812497887289?t=nkpIbh2A9B0wScpC0iJlqg&s=19)，OpenAI 推出了 **SWE-Lancer** 基准测试，旨在测试 LLM 执行价值高达 100 万美元的软件工程任务的能力。
   - 初步见解表明，模型在方案选择方面比在具体实现方面表现更出色，揭示了其优势与短板。
- **微软凭借 Majorana 1 芯片启动量子计算**：**Satya Nadella** 宣布推出 **Majorana 1** 芯片，这是量子计算领域的一项重大进展。正如[这篇博客文章](https://kuberwastaken.github.io/blog/Technology/Majorana-1---Why-Quantum-Computing-Matters-Now)所述，该芯片有望在几分钟内完成超级计算机需要*数十亿年*才能完成的计算。
   - 这一创新具有重塑行业并显著影响气候变化的潜力。
- **CommentRescueAI 助力 Python 文档编写**：一位成员介绍了 **CommentRescueAI**，这是一个网页扩展程序，旨在轻松地为 Python 代码添加 AI 生成的 docstrings 和注释。
   - 该扩展程序现已在 **VS Code marketplace** 上架，创作者正在征求建议和功能创意。
- **Agents 课程证书获取问题已解决**：许多用户最近在完成 **Agents Course** 的测验后难以生成证书，但该问题已得到解决，目前提交会被定向到一个直接生成证书的新 Space。
   - 鼓励参与者尝试更新后的测验链接，以便及时领取证书。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Grok 3 震撼服务器与人心！**：根据 [xAI 的推文](https://x.com/xai/status/1892400129719611567)，**Grok 3** 现已免费开放，直至达到服务器容量上限，并增加了 X Premium+ 用户和 SuperGrok 成员的访问权限。
   - 用户推测 **Grok 3** 可能会有限制，预计每天限制约 5 次查询，这引发了对可用性的担忧。
- **LLM 推理成为瓶颈**：用户对使用 Aider 编码时的 **LLM 推理速度** 表示沮丧，而根据社区反馈，其他人指出 **Azure** 的 **OpenAI API** 通常更快。
   - 他们分享了关于缓存、将文件标记为只读以及改进 Aider 编码实践的技巧，并详细说明了特定偏好，例如使用 **httpx** 替代 **requests**。
- **OpenRouter 限制端点**：讨论显示 **OpenRouter** 可能会将 **o3-mini** 等模型的端点限制在 **100k** 次请求，这表明了潜在的可用性问题。
   - 这种限制使得用户尝试利用 **OpenRouter** 进行更高强度 AI 交互的体验变得复杂。
- **Ministral 可能会加入 Aider！**：一位成员在看到讨论 **Mistral** 的 [LinkedIn 帖子](https://www.linkedin.com/posts/deividas-mataciunas_ai-mistral-opensource-ugcPost-7294619759722074112-a_JM?utm_source=share&utm_medium=member_android&rcm=ACoAABZCz5cBsMAYVy_zzTHh2HzsmuBv_27C49Y)后，询问是否有人尝试将 **Ministral** 与 **Aider** 集成，并对其兼容性表示好奇。
   - 另外，另一位成员发现构建过程*非常缓慢*，因为在使用 **TF-IDF** 的 API 调用期间，它会用 'build' 重写块。
- **LLADA 模型登场**：社区讨论了一个名为 **LLADA** 的新模型，该模型在编码任务上表现出高性能。
   - 见解表明，由于其创新的方法，像 **LLADA** 这样的模型可能会成为代码编辑的强力替代方案。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Sonnet 在编程方面表现出色**：用户因其可靠性和上下文理解能力而青睐 **Sonnet** 处理编程任务，并指出 **Anthropic 模型**在结合推理和谨慎执行时，在编程和复杂任务中表现卓越。
   - 用户普遍认为 **OpenAI** 和 **DeepSeek** 是通用任务的强力竞争者。
- **Grok 3 面临用户审查**：一些用户批评了 **Grok 3** 的实用性，根据个人测试经验将其贴上“令人失望”标签，显示其在编程任务中的不足，尽管 YouTube 上有一些正面评价。
   - 其他人通过指出多位 YouTuber 的好评来质疑针对 **Grok 3** 的负面论点，而另一些人则指向了正面评价。
- **Cursor 的 Agent 模式简化工作流**：用户分享了在 Agent 模式下使用 **Cursor** 的经验，强调了变更日志 (changelogs) 和自定义规则等功能如何帮助简化工作流。
   - 讨论围绕使用 AI 助手自动化任务并确保高质量输出展开，突出了正面和负面的互动。
- **初创项目寻求程序员合作**：一位用户宣布有兴趣启动一个新项目，正在寻找有时间和奉献精神的程序员进行合作，并表示愿意为项目提供资金。
   - 这一公告引发了频道成员对潜在合作伙伴关系的开放态度。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Deep Research 饱受崩溃困扰**：用户报告 **Deep Research** 功能持续崩溃，特别是企业版用户，此外还存在线程导航和访问库内容的问题。
   - 他们讨论了潜在的修复方案，暗示了平台稳定性和可用性的底层问题，目前正在积极进一步调查该问题。
- **订阅模式引发讨论**：一位成员对 **Pro 订阅**模式表示困惑，该模式以单一费用提供对多个模型的访问。
   - 关于定价策略合法性的猜测随之而来，一位用户幽默地暗示模型可能是盗版的，尽管这很快被斥为毫无根据；然而，关于在各种模型上运行推理 (inference) 的实际成本讨论仍在继续。
- **图像生成功能表现不佳**：尽管访问小组件显示具有**图像生成能力**，但一些用户在平台上使用该功能时遇到了困难。
   - 讨论了变通方法，包括为图像创建编写特定的提示词 (prompts) 以及使用浏览器插件；然而，完全集成到 **Perplexity** 的工作似乎仍在进行中。
- **R1-1776 模型承诺无审查回复**：研究了 **R1-1776 模型**的推理能力及其与标准 **R1 模型**的区别，指出其基于 **DeepSeek** 模型提供无审查回复的潜力。
   - 用户分享了使用该模型的经验，强调了其在敏感话题和操作上下文中的表现，特别是使用 [OpenRouter API](https://openrouter.ai/perplexity/r1-1776)。
- **IRS 利用 Nvidia 增强算力**：据报道，**IRS** 正在采购一台 **Nvidia 超级计算机**以增强运营能力，提高税务处理数据分析的效率。
   - 在日益增长的数据挑战中，这一举措被视为优化税务处理技术的关键，尽管尚未发布具体的模型细节或配置。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Grok 3 发布引发质疑**：虽然 **Grok 3** 展现出潜力，但对其真实能力的质疑依然存在，用户正在 Twitter 之外寻找可靠的来源进行性能验证。成员们讨论了关于整体性能指标以及 **Grok 3** 是否能有效竞争的担忧。
   - 有人对阅读 Hacker News 的评论表示担忧。*为什么关于 XAI 的讨论感觉压力这么大？*。
- **Google 推出 PaliGemma 2 Mix**：Google 推出了 **PaliGemma 2 mix**，允许在多种任务中开箱即用，突破了预训练 Checkpoint 的限制。查看 [发布博客文章](https://developers.googleblog.com/en/introducing-paligemma-2-mix/)。
   - 与之前版本进行了对比，并讨论了命名和功能的清晰度及实用性。
- **AI CUDA Engineer 旨在提升 Kernel 速度**：Sakana AI 推出了 **AI CUDA Engineer**，这是一个旨在生成优化 **CUDA kernels** 的 AI 系统，相比典型实现可实现 **10-100 倍加速**。查看 [Sakana AI 的公告](https://x.com/SakanaAILabs/status/1892385766510338559)。
   - 该系统被预见为可能对机器学习运营中 AI 驱动的性能优化产生变革性影响。
- **Tulu3 70B 进行为期一周的 RLVR 训练**：在 8×8 H100 配置下，**tulu3 70B 模型**的 **RLVR** 阶段训练预计在内存充足的情况下需要约**一周**时间。使用 **GRPO** 将增强内存能力，这可能会对训练过程产生积极影响。
   - 团队指出，他们已更新论文，包含了关于训练阶段的相关信息。
- **Evo 2 成为生物学基础模型**：Michael Poli 宣布推出 **Evo 2**，这是一款拥有 **400 亿参数**的新型基础模型，专为生物应用设计，旨在显著推进对基因组学的理解。查看 [Michael Poli 的公告](https://x.com/MichaelPoli6/status/1892242976942035029)。
   - **Evo 2** 旨在展示推理时扩展定律（test-time scaling laws），并改进乳腺癌变异分类，同时推动 AI 领域的真正开源工作。



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 辩论推理 Token 默认设置**：由于用户反馈希望在 **max_tokens** 较低时也能接收内容，**OpenRouter** 正在重新考虑 **include_reasoning** 的默认设置。
   - 目前正进行社区反馈投票，共有四个选项，从保持当前设置到将推理 Token 设为默认，以及一个用于用户评论的补充选项。
- **Grok 3 在推理任务中表现平平**：用户报告称，尽管 **Grok 3** 被誉为顶级 **LLM**，但与 **Claude 3.5** 和 **O1** 相比，其推理能力并不理想。
   - 一位用户表示它没有达到预期，并对其现状表示担忧。
- **新用户在导航 OpenRouter API 时表示困惑**：新用户在访问 **OpenRouter** 以及通过 API 使用 **O3 mini** 时遇到困难，这引发了关于使用限制的问题。
   - 需要更好的集成选项和关于 **API key** 使用的清晰说明，特别是关于模型访问的逐步推广。
- **Perplexity R1 1776 加入战场**：来自 **Perplexity** 的 **R1 1776** 模型（**DeepSeek R1** 模型的一个版本）已发布，允许用户访问经过*后训练以移除审查*的模型。
   - [X 上的公告](https://x.com/perplexity_ai/status/1891916644248846789?t=7_5m7rcR2w7GFITF2I2QSA&s=19) 链接到了该模型权重的 [HuggingFace Repo](https://huggingface.co/perplexity-ai/r1-1776/discussions)。
- **聊天机器人集成寻求帮助**：一位用户需要关于使用 **OpenRouter API** 在 HTML 网站上集成 AI 聊天机器人的指导，强调了对现成资源的需求。
   - 社区成员建议他们需要自行开发解决方案或聘请开发人员协助。



---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **ControlNet 提升图像准确度**：**ControlNet** 通过使用姿势作为骨架参考，增强了 **Stable Diffusion (SD)** 和 **Flux** 中的图像生成，提高了生成图像的准确性。
   - 用户发现 **ControlNet** 与 **SD** 和 **Flux** 配合良好，可以生成特定姿势的图像。
- **SwarmUI 简化 AI 工具访问**：**ComfyUI** 复杂且类似思维导图的界面对普通用户并不友好，但 **SwarmUI** 和其他一站式解决方案提供了更便捷的访问方式。
   - 这些替代方案提供了更简单的界面，使 AI 工具更易于使用。
- **RTX 3080 可有效处理 SD 和 Flux**：拥有 **RTX 3080** GPU 的用户可以高效运行 **Stable Diffusion 3.5** 和 **Flux**。
   - **SD 3.5** 和 **XL** 之间的选择取决于用户的具体需求，新模型提供了更好的功能。
- **安装指南简化设置**：**SwarmUI**、**CS1o** 教程和 **Lykos Stability Matrix** 的安装指南提供了设置 AI 工具的详细步骤。
   - 这些资源协助用户完成不同界面的安装过程，例如 [LykosAI Stability Matrix](https://github.com/LykosAI/StabilityMatrix/)。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 2025 年上半年路线图公开**：**Torchtune** 2025 年上半年的路线图现已在 [PyTorch dev-discuss](https://dev-discuss.pytorch.org/t/meta-pytorch-team-2025-h1-roadmaps/2794) 上发布，概述了关键目标和时间表，包括在新型模型架构方面保持领先。
   - 团队旨在优先处理现有的核心工作，该路线图是对整个 **PyTorch organization** 其他令人兴奋的进展的补充。
- **Packing 增加 VRAM 占用**：用户发现，在训练过程中使用具有更长序列长度的 packing 会显著提高 **VRAM requirements**，导致不可预测的内存分配。
   - 社区正在积极调查 kernel 差异，以优化内存管理和设置。
- **Llama 3B 微调后表现异常**：在对 **3B Llama models** 进行微调后，用户报告模型在推理过程中会出现胡言乱语的问题，这与 **8B variant** 形成了鲜明对比。
   - 团队怀疑 **Torchtune** 的模型导出和 checkpointing 过程中存在潜在 bug，这可能会影响 **3B model**。
- **新的 Attention 机制即将到来**：社区有兴趣将稀疏和压缩 attention 等先进的 **attention mechanisms** 集成到 **Torchtune** 中，以提高效率。
   - **Torchtune** 团队欢迎新想法，重点是利用 **PyTorch** 核心功能。
- **StepTool 增强多步工具使用**：论文 *StepTool: Enhancing Multi-Step Tool Usage in LLMs through Step-Grained Reinforcement Learning* ([arxiv.org/abs/2410.07745](https://arxiv.org/abs/2410.07745)) 介绍了 **StepTool**，这是一种新型的强化学习框架，通过将工具学习视为动态决策任务，增强了 **LLMs** 的多步工具使用能力。
   - 关键在于 **Step-grained Reward Shaping**，它根据工具交互的成功程度及其对任务的贡献在交互过程中提供奖励，从而以多步方式优化模型的 policy。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Grok-3 API 仍然难以获取**：成员们讨论了 **Grok-3** 缺少 API 的问题，这限制了其可用性，并建议在进行任何测试之前需要独立的 benchmarks。
   - 观察表明 **Grok-3** 可能会混淆内容，引发了关于如何将其集成并应用于各种场景的持续讨论。
- **Le Chat 以速度和质量赢得青睐**：用户报告了使用 **Le Chat** 的积极体验，称赞其速度、图像生成能力以及与其他模型相比的整体质量；**法国居民**正获得低价订阅优惠。
   - 一位用户分享了 [Le Chat 的链接](https://chat.mistral.ai/chat)，称其 *“与竞争对手相比令人印象深刻，尤其是对于法国居民的这个价格点。”*
- **AI 渲染可能变革游戏设计**：讨论集中在 **AI rendering** 通过动态和交互式环境彻底改变游戏的潜力。
   - 成员们对平衡 AI 生成内容与传统游戏开发表示担忧，强调了与技术进行有意义互动的重要性，正如 Satya Nadella 在关于想象用 AI 创建整个交互式环境的 [推文](https://fxtwitter.com/satyanadella/status/1892244164814725387) 中所描述的那样。
- **SWE-Lancer 以 100 万美元挑战赛考验模型**：[论文](https://arxiv.org/abs/2502.12115)中描述的 **SWE-Lancer** benchmark 包含来自 Upwork 的 1,400 多个自由软件工程任务，总价值 **100 万美元**。
   - 任务范围从 **50 美元的 Bug 修复**到 **32,000 美元的功能实现**，评估显示 **frontier models** 难以有效解决大多数任务。
- **MoBA 旨在提升长上下文 LLM 性能**：在 [GitHub](https://github.com/MoonshotAI/MoBA) 上记录的 **MoBA** 引入了一种 **Mixture of Block Attention** 方法，旨在提高 **long-context LLMs** 的性能。
   - 该项目旨在克服有效处理长输入的局限性，可以在 GitHub 上找到更多见解和贡献。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Notebook LM 擅长书籍教学**：一位用户确信 **Notebook LM** 可以通过使用精确的 prompts 有效地教他们一本书，并希望这能帮助避免跳过关键部分。
   - 这反映了用户越来越多地尝试并依赖 **AI tools** 来获得个性化的学习体验。
- **音频讨论被打断**：一位用户赞赏 **Notebook LM** 如何帮助组织回复，但发现音频讨论由于声音之间的频繁中断而难以跟进。
   - 他们建议采用更结构化的方法，即一个声音在另一个声音插话之前表达完整的想法，这表明需要改进 **audio management features**。
- **播客 TTS 提示词出现问题**：一位用户在播客功能的 **TTS prompts** 方面寻求帮助，难以让主持人逐字阅读文本。
   - 对所用精确提示词的请求表明了在实现对 **AI-driven voice outputs** 精确控制方面的挑战。
- **Notebook LM 访问面临限制**：一位用户询问是否可以邀请非 Google 账户持有者访问 **Notebook**，类似于 **Google Docs** 的共享功能。
   - 这突显了 **Notebook LM** 在访问和协作功能方面持续存在的局限性，可能会阻碍更广泛的采用。
- **NotebookLM Plus 设有使用上限**：用户注意到 **NotebookLM Plus** 在所有笔记本中每天有 500 次聊天查询限制，且共享上限为 50 个用户。
   - 通过创建多个账户来绕过这些限制的建议表明，尽管目前对 **AI Tool usage** 有所限制，用户仍在寻找最大化效用的方法。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **AI CUDA Engineer 自动生成 Kernel**：Sakana AI 推出了 [AI CUDA Engineer](http://sakana.ai/ai-cuda-engineer/)，它可以自动生成优化的 CUDA kernel，通过将 PyTorch 算子转换为 CUDA，可能实现比普通 PyTorch 操作快 **10-100 倍**的速度。
   - 该工具的过程涉及使用由 LLM 驱动的**进化方法（evolutionary approach）**来超越传统的 **torch.compile**，并发布了一个包含 **17,000 个经过验证的 CUDA kernel** 的数据集。
- **Windows 上的 CUDA 安装面临挑战**：多名用户报告称，**CUDA Express Installer** 在 Windows 上针对不同 NVIDIA GPU 安装 **Nsight** 和 **Visual Studio** 等组件时卡住。
   - 在 **Visual Studio** 中安装 CUDA **12.5 和 12.8** 时遇到了困难，并引发了关于 **Linux** 与 **Windows** 安装便捷性的讨论。
- **DeepSeek 旨在通过 CodeI/O 提高推理能力**：DeepSeek 发布了一篇关于 [CodeI/O](https://arxiv.org/abs/2502.07316) 的论文，旨在通过将代码转换为输入输出预测格式来提高推理能力。
   - 这种新的训练方法基于自然语言任务，这与增强狭窄技能的做法有所不同；论文可以在 [GitHub](https://github.com/open-thought/reasoning-gym/issues/160) 上找到。
- **AMD 推出 ROCm 开发人员认证**：AMD 推出了 [ROCm 应用开发人员证书（ROCm Application Developer Certificate）](https://www.amd.com/en/products/software/rocm/application-developer-certificate.html)，以增强 ROCm 生态系统中的 GPU 计算技能。
   - 该认证旨在促进**开源 GPU 计算技术**的专业化。
- **利用 Triton 和 CUDA 优化 Kokoro TTS**：现代 TTS 模型 **Kokoro** 的推理速度得到了提升，通过使用 **Triton + CUDA graph** 减少了 **4070Ti SUPER** 上 bs=1 时 LSTM 的 kernel 启动开销。
   - 一位成员表示 *“不敢相信在这个时代我还得优化 LSTM”* 😂，并引用了他的改进[测量结果](https://x.com/gaunernst/status/1892227983072518503)。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **DeepSeek R1 推出增强版量化模型**：一位成员分享了在 Hugging Face 上运行 Perplexity 未经审查的 [**DeepSeek R1**](https://huggingface.co/unsloth/r1-1776-GGUF) 以及其他模型版本的链接。
   - 新的**动态 2-bit GGUF 量化（quants）**承诺比标准的 **1-bit/2-bit** 格式具有更高的准确性；包含使用说明。
- **模型引导（Model-guidance）加速扩散模型训练**：模型引导（**MG**）目标移除了 Classifier-free guidance（**CFG**），加速了训练并使推理速率翻倍，在 **ImageNet 256** 基准测试中实现了 **1.34** 的 FID，达到 state-of-the-art 性能。
   - 评估确认 **MG** 在扩散模型训练中树立了新标准。
- **AI CUDA Engineer 自动化 Kernel 优化**：[AI CUDA Engineer](http://sakana.ai/ai-cuda-engineer/) 自动化了 CUDA kernel 的创建，实现了比标准 PyTorch 快 **10-100 倍**的速度，并发布了 **17,000** 个经过验证的 kernel。
   - 这一进展据称标志着机器学习操作中 AI 驱动效率的新时代，对推理时间的自动优化具有重要意义。
- **探索优化 LLM 计算分配**：成员们讨论了在最近发布的 **Grok 3** 等 **LLM** 中，对优化**测试时计算（test-time compute）**的关注度日益增加，并且扩大预训练计算规模正在产生收益。
   - 尽管对预训练与后训练之间的分配平衡存在推测，但关于资源分配的数据在很大程度上仍无法获得。
- **Transformer Engine 在 NeoX 中面临障碍**：一位尝试在 NeoX 中集成 **Transformer Engine** 的成员遇到了阻碍测试的挑战，表明启用非 FP8 标志会导致系统故障。
   - 这突显了在旨在维护系统可靠性的同时，集成所面临的复杂性。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **马斯克进军游戏领域并推出 Grok 3**：Elon Musk 的 xAI 宣布成立一个与 **Grok 3** 相关的全新游戏工作室，将其 AI 计划扩展至游戏领域，正如其在 [Twitter](https://x.com/elonmusk/status/1891388509191049307) 上所发布的。
   - 此举紧随 **Grok 3** 最近的品牌更新之后，表明了将 AI 与游戏应用集成的战略转变。
- **DeepSeek 探索稀疏注意力机制**：社区讨论重点关注了 **DeepSeek** 的新论文《**Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention**》（[链接](https://arxiv.org/abs/2502.11089)），该论文因其高质量而受到赞誉。
   - 该论文介绍了一种新型的稀疏注意力机制，该机制既符合硬件特性又是原生可训练的，解决了计算效率问题。
- **Murati 创立新 AI 实验室**：前 OpenAI CTO Mira Murati 启动了 [Thinking Machines Lab](https://thinkingmachines.ai)，专注于开发更易理解和可定制的 AI 系统，同时确保公众透明度。
   - 虽然具体项目仍处于保密状态，但该实验室承诺定期发布技术研究，目标是降低 AI 系统的黑箱程度。
- **Perplexity 开创“后审查” AI**：[Perplexity AI](https://the-decoder.com/perplexity-ai-removes-chinese-censorship-from-deepseek-r1/) 推出了 **R1 1776**，这是一个旨在通过对 **DeepSeek R1** 模型使用创新的后训练技术来克服中国式审查的新模型。
   - 这种方法旨在创建更强大的 AI 系统，在生成内容的同时保持上下文的完整性，即使面临审查限制。
- **微软凭借 Majorana 1 实现量子飞跃**：微软发布了 [Majorana 1](https://azure.microsoft.com/en-us/blog/quantum/2025/02/19/microsoft-unveils-majorana-1-the-worlds-first-quantum-processor-powered-by-topological-qubits/)，这是全球首款由拓扑量子比特驱动的量子处理单元（QPU），具有在单芯片上扩展至百万量子比特的潜力。
   - 这一进步代表了迈向实用量子计算的重要一步，为计算技术的新纪元铺平了道路。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Thinking Machines Lab 诞生**：Thinking Machines Lab 正式启动，旨在提高 AI 系统的可访问性和定制化，由 Mira Murati（前 OpenAI CTO）和 Lilian Weng 等知名人物领导，致力于解决 AI 领域的知识鸿沟。
   - 该实验室的成立引发了 AI 社区的热烈讨论，[Andrej Karpathy 向团队表示祝贺](https://x.com/karpathy/status/1891938714915569711)，团队中许多成员都曾参与构建 **ChatGPT**。
- **Perplexity AI 的 R1 1776 宣布开源**：Perplexity AI 开源了 **R1 1776**，这是其 DeepSeek R1 模型的一个版本，旨在提供无审查且无偏见的信息，推动更可靠的 AI 模型发展。
   - AI 社区戏称这种做法为“自由微调”（freedomtuning），标志着该模型对**未过滤数据**和**事实准确性**的重视。
- **OpenAI 的 SWElancer 设定新编程基准**：OpenAI 推出了 **SWElancer**，这是一个包含 1,400 多个自由软件工程任务的新基准，用于评估 AI 在真实场景中的编程性能。
   - 这一举措是在关于 **AI 驱动的游戏生成**可能取代传统游戏工作室的讨论中提出的，强调了对现实评估指标的需求。
- **Mastra 的 JS SDK 释放 AI Agent 潜力**：Mastra 发布了一个开源 JavaScript SDK，旨在促进能够通过内置工作流执行复杂任务的 **AI Agent** 的开发。
   - 该框架旨在方便与 **Vercel 的 AI SDK** 进行协作和集成，标志着开源 AI 开发的重大进展。
- **Lambda 获得 4.8 亿美元 D 轮融资**：Lambda 完成了由 Andra Capital 和 SGW 领投的 4.8 亿美元巨额 D 轮融资，突显了市场对**为 AI 应用量身定制的云服务**日益增长的兴趣。
   - 来自 **NVIDIA** 和 **ARK Invest** 等投资者的参与强调了该公司在不断发展的 AI 领域中的潜力。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Grok 3 提前发布，令 Mojo 感到意外**：**Grok 3** 的发布抢在了 **Mojo** 的进度之前，但这激发了开发者的兴趣而非挫败感。
   - 这种提前发布被认为是有益的，可能会进一步推动 **Mojo** 的创新。
- **Polars 迅速集成到 Mojo**：一位开发者报告称能快速将 **Polars** 导入 **Mojo** 项目，并在 [GitHub](https://github.com/rcghpge/pymo/tree/main) 上分享了包含示例的实现。
   - 针对在项目语境下“实现”与“导入” **Polars** 之间的区别进行了澄清。
- **MAX 25.1 直播引发好奇**：即将举行的直播将涵盖 **MAX 25.1**，并提供 [Google Form](https://forms.gle/NkjU6e3n15TRtiMA7) 用于提交问题。
   - 该活动通过 [LinkedIn](https://www.linkedin.com/events/introducingmax25-17297704283980902402/theater/) 进行推广，鼓励社区参与。
- **Mojo 的快速排序面临性能瓶颈**：一位用户发现用 **Mojo** 实现的 **quick sort** 算法显著慢于（2.9s）其 Python 版本（0.4s）。
   - 随后的讨论建议使用 **Mojo** 的 benchmark 模块来隔离并准确测量排序性能，并注意编译时间对计时结果的影响。
- **Slab List 相比 Linked List 受到更多关注**：成员们探讨了 **SlabList** 优于传统 **LinkedLists** 的优势，重点在于常数时间操作和缓存效率，并指向了 [nickziv's github](https://github.com/nickziv/libslablist)。
   - **slab list** 被定义为 `LinkedList[InlineArray[T, N]]`，在不进行复杂操作的情况下优化了内存使用。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **CUDA GPU 获得支持**：频道讨论了增加对旧款 **CUDA 5.0 兼容 GPU**（如 **GM107/GM108**）的支持，成员指出目前缺乏对低端架构的支持。
   - 一位成员确认支持这些 GPU 的 **PR** 已经合并，并将包含在下一个版本中，参考 [CUDA Wikipedia 页面](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)。
- **GPT4All 嵌入 Token 限制揭晓**：成员们讨论了达到 **GPT4All** 嵌入的 **1000 万 Token 限制**，指出基础价格为 **$10/月**，额外 **Token** 需另付费。
   - 澄清了从本地文档中删除 **Token** 并不会减少已计费的总 **Token** 数。
- **聊天模板提示词令人头疼**：成员们寻求关于使用 **chat templates** 指导模型引用摘录的澄清，但根据 [GPT4All's docs](https://docs.gpt4all.io/gpt4all_desktop/chat_templates.html#using-the-cli-all-models)，被告知系统消息指令就足够了。
   - 其他成员询问了 Jinja 或 JSON 代码在提示模型时的有效性，这表明实现预期输出具有复杂性。
- **Nomic v2 发布失踪了？**：关于 **Nomic v2** 缺席的猜测四起，成员们对延迟表示好奇并指出其重要性。
   - 一位成员幽默地质疑了在没有新版本更新的情况下漫长的等待。
- **GPT4All 的图像处理表现不佳**：一位成员请求能够像其他平台一样直接在聊天中**复制粘贴图像**，但 **GPT4All** 目前不支持图像输入。
   - 频道建议使用外部软件进行图像处理。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Anthropic 意外宕机**：**Anthropic 首页**经历了停机，引发了关于潜在服务中断的猜测。
   - 一位成员报告了停机情况，并通过随附的图片进行了分享。
- **Haiku 3.5 传闻甚嚣尘上**：讨论围绕 **Haiku 3.5** 的潜在发布展开，可能包含 **tool** 和 **vision** 支持。
   - 一位成员还暗示我们可能会看到 **Sonnet 4.0** 的发布。
- **Cursor MCP 工具检测失效**：多位成员报告 **Cursor MCP** 显示 *'No tool found'*，表明这是一个普遍问题。
   - 一位用户分享了一个 **/sse** 实现来解决此问题，并提供了相关信息的链接。
- **Google Workspace MCP 功能强大**：一位成员重点介绍了他们在 Docker 上运行的 [Google Workspace MCP](https://github.com/aaronsb/google-workspace-mcp)，支持多账号和 token 自动刷新，并提供适用于不同平台的 Docker 镜像。
   - 该 MCP 提供了对 **Gmail**、**Calendar** 和其他 Google Workspace API 的集成访问。
- **Python REPL 获得 Matplotlib 支持**：一位成员介绍了他们的 [Python REPL for MCP](https://github.com/evalstate/mcp-py-repl)，提供 STDIO 支持、**matplotlib**、**seaborn** 和 **numpy**。
   - 未来计划包括添加 **IPython** 支持，并对类似于 **Jupyter** 中的可视化功能表现出浓厚兴趣。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaCloud EU 消除障碍**：**LlamaCloud EU** 已宣布推出，专门为欧洲企业提供安全、合规的知识管理，重点关注欧盟管辖范围内的数据驻留，详情见[此处](https://twitter.com/llama_index/status/1892271183451869512)。
   - 这一早期访问产品旨在为关注**合规性和数据隐私**的欧洲公司消除重大障碍。
- **供应商调查问卷应用实现答案检索**：来自 @patrickrolsen 的创新[全栈应用](https://twitter.com/llama_index/status/1891897812834816379)允许用户通过语义化检索之前的答案并使用 LLM 进行增强，从而回答**供应商调查问卷**。
   - 该应用通过简化阅读表单和填写答案的过程，展示了 **knowledge agents** 的核心用例。
- **AgentWorkflow 工具输出存在 Bug**：一位用户报告称，尽管生成了响应，但其 **AgentWorkflow** 的工具输出列表仍然为空，并寻求关于 [AgentWorkflow tool](https://docs.llamaindex.ai/en/stable/examples/agent/agent_workflow_multi/) 实现的澄清。
   - 另一位成员分享道，*streaming events* 可以作为一种变通方法，在 AgentWorkflow 执行期间捕获所有工具调用。
- **AI 和数据运营（Data Ops）下一阶段的挑战**：最近一篇题为 [《大而笨的 AI 与数据的终结》](https://open.substack.com/pub/procurefyi/p/the-end-of-big-dumb-ai-data?r=223ajc&utm_campaign=post&utm_medium=web&showWelcomeOnShare=false) 的文章讨论了 **AI 和数据运营** 中挑战传统方法的新兴趋势。
   - 它强调了在处理数据以实现更好决策时向更智能、更高效系统的转变，并关注自**两年前**启动以来的联邦技术支出和企业级 AI 应用。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **MOOC 学生获得传奇地位**：F24 MOOC 表彰了 **15,000** 名学生，庆祝了 **304 位 trailblazers**、**160 位 masters**、**90 位 ninjas**、**11 位 legends** 以及 **7 位 honorees**。
   - 课程工作人员强调，三位 honorees 来自 ninja 级别，四位来自 masters 级别，标志着广泛的成就。
- **高级课程证书现已推出！**：成员们询问了高级课程证书的可用性，以及是否可以在没有 F24 MOOC 证书的情况下获得它，课程工作人员对这两个问题都给予了肯定回答。
   - 关于高级课程证书获取的具体细节将很快发布。
- **LangChain 简化 LLM 应用**：**LangChain** 旨在简化 LLM 应用生命周期，涵盖[开发](https://python.langchain.com/docs/introduction/)、生产化和部署等领域，并提供各种组件。
   - 它的工作原理是将一个 LLM 的输出链接为另一个 LLM 的输入，从而有效地创建链以增强性能，这对于 **LLM 应用架构师** 来说非常有用。
- **结合机器学习预测模型探索 LLM**：讨论了将 **LLM Agent** 与机器学习预测模型相结合的问题，建议查看 [Everscope](https://everscope.ai/?models=Large+Language+Model+%28LLM%29&sort_by=elo_rating&sort_order=desc&page=1&limit=10&timeRange=this-week) 上的学术论文以获取见解。
   - 在 **Everscope** 上过滤“all-time”可能会产生与 LLM 相关的最佳论文，使你能够学习新技术。
- **MOOC 课程视频现已发布**：课程工作人员宣布，当前课程的视频讲座仍可在 [syllabus](https://llmagents-learning.org/sp25) 中查看。
   - 课程工作人员鼓励成员报名参加 [Spring 2025 iteration](https://llmagents-learning.org/sp25) 以继续学习，因为之前的课程已不再提供测验和考试。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **频道流量激增**：所有 **text channels** 都经历了高流量，表明正在进行的讨论激增。
   - 还注意到大部分流量是由 **automation bots** 造成的，一名成员请求开设一个专门用于分享 **screenshots** 的新频道。
- **利润共享提案引起轰动**：一名成员提议了一项针对 **25-50** 岁人群的 **profit-sharing 合作**，潜在利润在 **$100 到 $1500** 之间。
   - 该提案引发了关于身份共享以及*协作与盗窃*之间平衡的辩论。
- **身份共享受到审查**：成员们对在社区内共享个人身份信息的 **隐私影响** 表示担忧，这引发了诸如“现在身份盗用这么公开吗？”之类的问题。
   - 一名成员建议，在分享观点时，**更清晰的表达**对于防止误解至关重要，尤其是在基于文本的媒介中。
- **项目频道呼吁透明度**：在 **projects** 频道中，发布者强调了一项提案中 **缺乏细节**，且没有网站或文档。
   - 一位用户形容整个风险投资是“可疑的”，并指出透明度对于谨慎的合作至关重要。
- **调查无咖啡世界的影响**：有人请求写一篇 **essay**，探讨没有咖啡的世界所带来的文化和经济后果。
   - 这个话题开启了关于如果没有这种流行饮料可能发生的假设情景和社会变化的讨论。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **用户寻求集成 Jamba-1.5-large 模型的帮助**：一位用户请求协助如何通过 **AI21 API** 向 `jamba-1.5-large` 模型格式化请求，成员们提供了 [API reference](https://docs.ai21.com/reference/jamba-15-api-ref) 和正确构建请求的示例。
   - 讨论强调了成功进行 API 调用所需的特定 headers 和参数，特别是在集成 `jamba-1.5-large` 时。
- **Jamba 1.5 API 输出中出现转义字符**：一位用户询问在使用 **AI21 API** 解决数学表达式时，API 输出中出现的意外细节。
   - 一位成员澄清说，输出中的转义字符需要额外的代码调整才能整洁显示，因为 **AI21 Studio** UI 会自动处理这些字符。
- **API 响应需要特殊字符处理**：社区成员讨论了在使用 **PHP** 时从 **AI21 API** 响应中移除特殊字符的必要性。
   - 讨论强调虽然 **AI21 Studio** UI 旨在处理特殊字符，但在直接访问 API 时，需要额外的代码处理以实现正确的输出格式化。
- **PHP 与 Symfony 集成带来挑战**：一位用户强调了使用 **Symfony** 和 **PHP** 集成 **AI21 API** 响应时的挑战，指出需要进行大量的数据转换和自定义处理。
   - 该用户感谢社区提供了关于在 **PHP** 环境中有效处理和格式化 API 输出的见解。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **SPO 框架优化 Prompt**：一种新的 **Self-Supervised Prompt Optimization (SPO)** 框架可以在没有外部参考的情况下，为封闭式和开放式任务发现有效的 Prompt，增强了 [Self-Supervised Prompt Optimization paper](https://arxiv.org/abs/2502.06855) 中详述的 **LLM 推理能力**。
   - 该框架纯粹从输出比较中推导评估信号，使其具有成本效益；一位参与者指出，该论文*直到最后一段才提到 DSPy*。
- **Zero-Indexing 互联网搜索改进 RAG**：一项研究介绍了一种新的 **Retrieval Augmented Generation (RAG)** 方法，详见 [Zero-Indexing Internet Search Augmented Generation for Large Language Models](https://arxiv.org/abs/2411.19478)，通过使用标准搜索引擎 API 在生成推理过程中动态集成最新的在线信息。
   - 这种范式涉及一个 **parser-LLM**，它决定是否需要互联网增强，并在单次推理中提取搜索关键词，从而在不依赖固定索引的情况下提高生成内容的质量。
- **DSPy 使用 Jinja2 合成数据**：成员们分享了一个 [GitHub 仓库链接](https://github.com/seanchatmangpt/dslmodel)，展示了来自 **DSPy** 和 **Jinja2** 的结构化输出，强调了它们在合成数据生成方面的结合能力。
   - 合成数据生成流水线可用于训练带有插图附件的 **ChatDoctor**。
- **Judge-Time Scaling 库 Verdict 亮相**：一位成员分享了 [Leonard Tang](https://x.com/leonardtang_/status/1892243653071908949) 的帖子，表达了对名为 **Verdict** 的新库的兴奋，该库专注于扩展 judge-time 计算，旨在解决 AI 中的评估局限性，特别是在开放式和不可验证领域。
   - 另一位成员表示，这个库非常适合他们的 **Personal Voice Identity Manager** 概念，强调了其对 AI 框架内个人身份管理的潜在影响。
- **DSPy Prompt 冻结导致控制流丢失**：一位成员分享了一段代码片段，展示了如何使用默认适配器将 **DSPy** 程序中的所有 Prompt 冻结并导出为消息模板。
   - 虽然这种方法很方便，但可能会导致丢失控制流逻辑，建议使用 `program.save()` 等替代方案。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **不同配置下的模型性能表现参差不齐**：一位用户报告在 **10 年前的 GeForce 850M** 上测试仅获得 *3 tok/s*，而另一位在 Windows 11 上使用 **RTX 4070** 的用户获得了 *12 tok/s*。
   - 使用 RTX 4070 的用户提到 **首个 token 生成时间 (time to first token) 为 1.9 秒**。
- **计算成本仍然过高**：尽管 **RTX 4070** 性能尚可，但一位用户发现由于高昂的计算成本和复杂性，该模型并不太实用。
   - 他们指出了诸如 **数值刚性 (numerical stiffness)** 以及非线性子问题导致难以获取准确解等复杂问题。



---


**MLOps @Chipro Discord** 没有新消息。如果该频道长时间保持安静，请告知我们，我们将将其移除。


---


**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长时间保持安静，请告知我们，我们将将其移除。


---

# 第 2 部分：频道详细摘要与链接


{% if medium == 'web' %}




### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1341573318402244699)** (1 条消息): 

> `Speculative Decoding, LM Studio 0.3.10 Release, Inference Speed Ups` 


- **LM Studio 0.3.10 引入 Speculative Decoding**：新版本 **0.3.10** 引入了 **Speculative Decoding**，通过将大模型与更小、更快的 draft model 配对，有可能使推理速度翻倍。
   - 有关使用此功能的更多详细信息，请阅读 [发布博客文章](https://lmstudio.ai/blog/lmstudio-v0.3.10)。
- **新解码过程确认了推理加速**：**Speculative Decoding** 通过验证 draft model 生成的 token，可以显著提升速度——最高可达 **2x**，该技术已在 **llama.cpp/GGUF** 和 **MLX** 中应用。
   - 正如讨论中所指出的，这种技术可以在聊天和 API 功能中实现更快的响应时间。
- **安装和更新变得简单**：用户可以通过应用内更新或从 [官方下载页面](https://lmstudio.ai/download) 轻松下载最新版本的 LM Studio。
   - 提供了包括模型目录和文档在内的额外资源，增强了用户体验和易用性。
- **鼓励社区参与反馈和 Bug 报告**：团队呼吁用户报告任何 Bug 或提供反馈，展示了团队对持续改进的承诺。
   - 鼓励成员关注更新并分享资源以获得社区支持。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lmstudio.ai/blog/lmstudio-v0.3.10.">未找到博客文章</a>：未找到描述</li><li><a href="https://lmstudio.ai/download">下载 LM Studio - Mac, Linux, Windows</a>：发现、下载并运行本地 LLM
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1341462141210726513)** (724 条消息🔥🔥🔥): 

> `LLM 的 Speculative Decoding、Text Embeddings 详解、DeepSeek 模型及其用途、模型性能与 Fine-Tuning、本地 AI 应用中的模型使用` 


- **Speculative Decoding 实验结果**：用户报告了 Speculative Decoding 的不同性能表现，指出在某些配置下（如使用更大的 Draft 模型），速度和 Token 接受率有所提高。
   - 一位用户在其配置下达到了超过 100 tokens/sec 的惊人速度，而其他人的结果较慢，引发了关于配置和模型选择的讨论。
- **理解 LM Studio 中的 Text Embeddings**：Text Embeddings 被描述为一种在 LLM 中存储信息以供长期使用的方法，用于直接在 LM Studio 中从文件中检索相关数据。
   - 鼓励用户创建并上传自己的文本文件以利用 Embeddings，并解释了 Embeddings 与 LLM 扩展（Extensions）的区别。
- **Fine-Tuning 及其影响**：Fine-Tuning 允许模型适应特定上下文，但如果操作不当（特别是涉及特定角色或概念时），可能会导致幻觉（Hallucinations）。
   - 讨论了 Fine-Tuning 过程，强调了给定的上下文和示例如何影响模型的响应和准确性。
- **使用模型进行对话和交互**：测试了模型的创新组合，一位用户展示了一个 Rust 集成，允许两个 LLM 就视频游戏偏好等话题进行对话。
   - 反馈表明，角色交互有时会导致意想不到且幽默的对话，反映了 AI 讨论的随机性。
- **LM Studio 的更新和模型支持**：用户讨论了 LM Studio 最近的 0.3.10 版本更新，以及保持应用程序更新以获得最佳性能的重要性。
   - 关于对 DeepSeek 和 MS Research 等特定模型支持的问题，引发了社区内关于性能和兼容性的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://notebooklm.google.com>">未找到标题</a>: 未找到描述</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1/commit/8a58a132790c9935686eb97f042afa8013451c9f">Update tokenizer_config.json · deepseek-ai/DeepSeek-R1 at 8a58a13</a>: 未找到描述</li><li><a href="https://tenor.com/view/tkt-smart-gif-20642718">Tkt Smart GIF - Tkt Smart - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/doubt-it-i-dont-believe-you-will-farrell-anchor-man-gif-5332521">Doubt It GIF - Doubt It I Dont Believe You Will Farrell - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://gitlab.com/logliwo/lm-studio-docker-compose">Aleksey Tsepelev / LM-Studio docker-compose · GitLab</a>: GitLab.com</li><li><a href="https://tenor.com/view/blender-jerma-crazy-blender-render-blender3d-gif-24406452">Blender Jerma GIF - Blender Jerma Crazy - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/bruh-meme-gif-26978290">Bruh Meme GIF - Bruh Meme - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/excited-fuego-gif-26833875">Excited Fuego GIF - Excited Fuego - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/miy-3d-modelling-sanae-blender-modelling-gif-5248426922200217011">Miy 3d Modelling GIF - MIY 3d modelling Sanae - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/mradermacher/ReasoningCore-Llama-3.2-1B-r1-GGUF">mradermacher/ReasoningCore-Llama-3.2-1B-r1-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-GGUF">unsloth/DeepSeek-R1-GGUF · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1341488785744990279)** (142 条消息🔥🔥): 

> `AI 模型的硬件规格，GPU 性能对比，大语言模型 (LLM) 的 Fine-tuning，开发任务的 AI 模型推荐，显卡基准测试 (Benchmarking)` 


- **A6000 GPU 适用于模型 Fine-tuning**：一位用户询问了关于 **phi4 模型**的 Fine-tuning，并确认他们可以使用每台拥有 **256 GB** 显存的 **A6000 GPU** 系统，这对于训练来说是足够的。
   - 其他人指出 [Unsloth 的 notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks) 是开始 Fine-tuning 的良好资源。
- **关于 RX 5080 与 5070 Ti 性能的讨论**：一位用户在尝试购买 **5070 Ti** 之前寻求 **RX 5080** 的基准测试，并指出两者的游戏性能似乎相似。
   - 一位成员提到只看到了来自 Linus 的基准测试，并强调 RX 5080 拥有 **16 GB** 的 RAM。
- **大型模型在有限硬件上的性能**：在关于在 **VM** 上设置 AI 模型的讨论中，用户建议使用像 **Qwen 2.5 7B** 这样的小型模型，以便在没有 **GPU** 的情况下获得更好的性能。
   - 有人担心大型模型可能会导致无法接受的推理 (Inference) 延迟，响应时间长达 **7 分钟**。
- **GPU 利用率和驱动问题**：一位用户最初在 **GPU** 显示 **0 容量**时遇到困难，通过更新 **LM Studio** 中的运行时版本解决了该问题。
   - 这使得 GPU 能够成功使用，并引发了关于 **Ollama** 等其他 AI 框架中 GPU 支持的问题。
- **法语任务推荐使用 Mistral 模型**：在设置法语模型的背景下，用户强调 **Mistral** 模型是此类任务的最佳选择。
   - 有人建议使用 **DeepSeek V2 Lite** 等模型，主张在没有 GPU 的情况下实现更快的推理速度。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://model.lmstudio.ai/download/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF">在 LM Studio 中下载并运行 Qwen/Qwen2.5-Coder-32B-Instruct-GGUF</a>：在你的 LM Studio 中本地使用 Qwen/Qwen2.5-Coder-32B-Instruct-GGUF</li><li><a href="https://model.lmstudio.ai/download/lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF">在 LM Studio 中下载并运行 lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF</a>：在你的 LM Studio 中本地使用 lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF</li><li><a href="https://model.lmstudio.ai/download/lmstudio-community/Codestral-22B-v0.1-GGUF">在 LM Studio 中下载并运行 lmstudio-community/Codestral-22B-v0.1-GGUF</a>：在你的 LM Studio 中本地使用 lmstudio-community/Codestral-22B-v0.1-GGUF</li><li><a href="https://oobabooga.github.io/blog/posts/gptq-awq-exl2-llamacpp/">GPTQ, AWQ, EXL2, q4_K_M, q4_K_S 和 load_in_4bit 之间的详细对比：困惑度 (Perplexity)、VRAM、速度、模型大小和加载时间。- LLM 博客</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth 文档</a>：以下是我们所有 notebook 的列表：
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1341454253990084639)** (777 条消息🔥🔥🔥): 

> `Grok 3 能力, OpenAI GPT 模型, 生物学中的 AI, AI 审查, AI 驱动的自动化` 


- **Grok 3 在多样化任务中展现出潜力**：用户报告称 Grok 3 在生成详细脚本和艺术作品方面表现出色，在提供丰富提示词的情况下，其表现优于 ChatGPT 免费版等较低层级的模型。
   - 然而，其有效性似乎取决于用户提示词的具体程度，更详细的请求会产生更好的输出。
- **AI 模型的审查机制引发关注**：参与者对 ChatGPT 和 Grok 等 AI 模型的审查表示沮丧，特别是在生成某些类型内容的问题上。
   - 讨论表明，虽然 Grok 的限制可能较松，但在敏感话题方面仍存在局限性，这反映了用户对其他平台的类似担忧。
- **Evo-2 推动生物学领域的 AI 进展**：开发了一个大型 AI 模型 Evo-2，用于从头编写基因组。该模型在庞大的 DNA 序列数据集上进行了训练，可能彻底改变生物学研究。
   - 这一里程碑预示着肿瘤学和基因工程等领域的潜在突破，提升了对未来医学进步的乐观情绪。
- **AI 模型对比**：用户将 Grok 3 的性能与 OpenAI 的模型（特别是 O1 Pro）进行了对比，Grok 以其速度和较不严格的审查而受到关注。
   - 虽然 Grok 3 展示了具有竞争力的能力，但据报道在逻辑推理方面不如更先进的模型，暗示仍有改进空间。
- **社区对 AI 功能的反馈**：许多用户表达了对增强功能的渴望，例如更好的视频处理、实时反馈以及用于创意任务的无审查输出。
   - 社区渴望能够增强 AI 模型交互性和可用性的进步，特别是针对日常应用。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://editor.p5js.org/Clock2003/full/6rVW_oo1Q">p5.js Web Editor</a>：未找到描述</li><li><a href="https://editor.p5js.org/Clock2003/full/VkXYn2iat">p5.js Web Editor</a>：未找到描述</li><li><a href="https://editor.p5js.org/Clock2003/full/yhqTYwIce">p5.js Web Editor</a>：未找到描述</li><li><a href="https://editor.p5js.org/">p5.js Web Editor</a>：未找到描述</li><li><a href="https://grok.com/share/bGVnYWN5_f7db7990-0368-483f-ad2c-e787b64d6a5a">Physics of a Falling Glove from a Car | Shared Grok Conversation</a>：你是推理专家，总是选择最现实的答案。请循序渐进地思考并……</li><li><a href="https://pastebin.com/raw/D05hpynW">Text MUD with Room Layout, Map, and Full Lore Printout</a>：未找到描述</li><li><a href="https://x.com/iterintellectus/status/1892251343881937090?s=46">来自 vittorio (@IterIntellectus) 的推文</a>：天哪，它正在发生。AI 现在可以从头编写基因组。Arc Institute 和 NVIDIA 刚刚发布了 Evo-2，这是最大的生物学 AI 模型，在涵盖整个……的 9.3 万亿个 DNA 碱基对上进行了训练。</li><li><a href="https://x.com/perplexity_ai/status/1891916573713236248?s=46">来自 Perplexity (@perplexity_ai) 的推文</a>：今天我们开源了 R1 1776——DeepSeek R1 模型的一个版本，经过后期训练，旨在提供无审查、无偏见且事实性的信息。</li><li><a href="https://x.com/sama/status/1891667332105109653">来自 Sam Altman (@sama) 的推文</a>：对于我们的下一个开源项目，是做一个相当小但仍需在 GPU 上运行的 o3-mini 级别的模型更有用，还是做一个我们能做的最好的手机端大小的模型？█████████████████o3-...</li><li><a href="https://x.com/elonmusk/status/1892166193152135229?s=46">来自 Elon Musk (@elonmusk) 的推文</a>：尝试我们新的 Grok 3 图像生成器。引用 Mario Nawfal (@MarioNawfal) …… 我可以花一整天只用 Grok 3 创建图像…… 它是不可思议的……
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1341465672193937510)** (29 messages🔥): 

> `AI 交互伦理, Prompt 优化技术, Discord 社区参与, 对 AI 和人类的尊重, ChatGPT 生产力技巧` 


- **讨论中的 AI 交互伦理**：一位成员表达了对与 AI 交互如何反映人类行为的担忧，指出：*“我们对待 AI 的方式……很可能就是我们对待同胞的方式。”*
   - 他们强调了在 AI 交互中建立边界的重要性，并主张给予尊重的对待。
- **Prompt 优化测试成功**：一位用户分享说，他们大幅简化了 Prompt，在 Playground 上测试时运行完美，但在公司服务器上却出现了故障。
   - Eskcanta 建议该问题可能与服务器代码有关，并建议追踪输入给模型的信息。
- **为 Discord 新用户提供欢迎和指导**：一位新加入者寻求关于提高生产力的 ChatGPT 课程建议，并询问该频道是否适合提出此类问题。
   - Eskcanta 鼓励他们直接与模型交流以探索具体问题，并强调了量身定制沟通的重要性。
- **鼓励与 AI 进行清晰的沟通**：Eskcanta 就用户应如何表述问题提供了见解，建议他们直接向模型寻求工作生产力方面的帮助。
   - 一个好的 Prompt 示例是：*“你能如何帮助我提高工作效率？”*，同时要考虑到用户请求的不同上下文。
- **探索 AI 的能力**：Eskcanta 强调模型可以辅助查找各种主题的信息，包括使用 ChatGPT 的合适课程。
   - 他们鼓励进行具体的询问以最大化模型的效用，并指出模型具有根据用户需求调整回复的能力。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1341465672193937510)** (29 messages🔥): 

> `AI 角色扮演问题, 对 AI 和动物的尊重, ChatGPT Prompt 优化, Discord 新用户` 


- **AI 角色扮演崩溃**：最近的讨论强调了关于 **o1 推理模型** 角色扮演不当的问题，这在交互过程中造成了困惑。
   - 一位用户幽默地将其描述为一个根本性的缺陷，质疑这些模型维持上下文的能力。
- **尊重与 AI 的边界**：一位社区成员观察到，恶劣地对待 AI 可能会反映出个体对待其他人类的方式，暗示需要建立 **健康的边界**。
   - 这引发了关于动物对待心理学及其与人类交互相似之处的讨论。
- **Prompt 清晰度对 ChatGPT 至关重要**：一位用户分享了他们改进 ChatGPT Prompt 的经验，注意到在 **Playground** 中功能有所提升，但在公司服务器中存在问题。
   - 回复敦促关注服务器输入，并确保清晰度以有效地引导模型的输出。
- **为 Discord 新用户提供指导**：一位新加入者请求获取合适的 **ChatGPT 课程** 资源以提高工作效率，寻求社区的指导。
   - 回复鼓励利用模型进行个性化学习，强调了根据个人工作背景进行具体查询的重要性。
- **强调 AI 交互中的清晰度**：社区强调，清晰、具体的 Prompt 会带来更好的 AI 回复，引导用户有效地定制他们的咨询。
   - 用户见解赞扬了明确表达需求的重要性，并根据其专业环境调整问题。


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1341881225656139799)** (1 messages): 

> `DeepSeek-V3, Windsurf Pro 和 Ultimate 计划` 


- **DeepSeek-V3 开启无限使用**：DeepSeek-V3 现在对 **Windsurf Pro** 和 **Ultimate** 计划的用户**不限次数**开放，这是针对订阅用户的一次重大更新。
   - 这一变化意味着用户不再需要担心 **0 Prompt 额度** 和 **0 flow action 额度**。
- **对更新的兴奋**：该公告引发了热烈反响，鼓励成员们开始“冲浪”并利用这些新功能。
   - 分享了来自 [Windsurf AI 官方账号](https://x.com/windsurf_ai/status/1892322088507105561) 的推文，强调了无限访问权限。



**提到的链接**：<a href="https://x.com/windsurf_ai/status/1892322088507105561">来自 Windsurf (@windsurf_ai) 的推文</a>：DeepSeek-V3 现在在 Windsurf Pro 和 Ultimate 计划中不限次数。0 Prompt 额度。0 flow action 额度。

  

---

### **Codeium (Windsurf) ▷ #[content](https://discord.com/channels/1027685395649015980/1092566563862884412/1341954445147377685)** (1 messages): 

> `MCP 使用案例，Cascade 集成` 


- **Matt Li 的 MCP 内容展示**：一名成员报告了新的 **MCP 内容**，并强调了 Matt Li 分享的酷炫使用案例，强烈呼吁大家关注该帖子。
   - 他们鼓励其他人查看该帖子并表达支持，称其值得一看，链接在[这里](https://x.com/windsurf_ai/status/1892394489588727985)。
- **MCP 在 Cascade 中的潜力**：一份公告链接到了关于 **MCP** 如何在 **Cascade** 中有效工作的演示，解决了围绕其应用的常见疑问。
   - 该演示旨在澄清 **MCP 的潜在使用案例**，并鼓励社区内的互动。



**提到的链接**：<a href="https://x.com/windsurf_ai/status/1892394489588727985">来自 Windsurf (@windsurf_ai) 的推文</a>：如果你对 MCP 及其潜在使用案例仍有疑问，这里有一个关于 MCP 如何在 Cascade 中工作的快速演示！

  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1341459039179112500)** (20 messages🔥): 

> `Codeium 功能，订阅计划，在 SaaS 或初创公司中的使用，社区支持，自动安装` 


- **关于 Codeium Autocomplete 功能的澄清**：成员们讨论了 Codeium 中 **autocomplete** 和 **supercomplete** 功能的区别，强调 **supercomplete** 可以建议整个函数，而 autocomplete 辅助单行代码。
   - *这引发了对文档的质疑*，其中一名成员表示难以找到关于自动安装的信息。
- **评估 Codeium 对学生的价值**：在辩论 **$60** 订阅费用时，成员们得出结论，对于开发 **SaaS** 项目的学生来说，这可能是值得的，因为这通常会带来更高的收益。
   - 一名成员提到，虽然这看起来是一笔不小的开支，但对于从事严肃项目的学生来说是可以平衡的。
- **对订阅计划的担忧**：一名新用户询问是否应从 **Teams** 切换到 **Pro** 订阅，对最适合其需求的计划表示不确定。
   - 另一名成员建议个人使用选择 Pro 订阅，除非团队功能是必不可少的。
- **社区对 Codeium 查询的建议**：几位成员建议直接联系 Codeium 团队以寻求支持，特别是关于企业版产品或特定功能的问题。
   - 鼓励一名用户填写联系表单，以便在寻求信息时获得更好的体验。
- **关于插件缺失功能的讨论**：一名成员幽默地指出某些功能并未延伸到 Codeium 平台，并引用了一条特定的 Discord 消息。
   - 这引发了关于用户期望和需要适当更新的轻松交流。



**提到的链接**：<a href="https://codeium.com/contact/enterprise">联系方式 | Windsurf 编辑器和 Codeium 扩展</a>：联系 Codeium 团队以获取支持并了解更多关于我们企业版产品的信息。

  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1341454801472585769)** (597 messages🔥🔥🔥): 

> `Windsurf 性能问题，DeepSeek 功能，额度使用担忧，MCP 集成挑战，模型对比`

- **Windsurf 面临性能挑战**：许多用户报告了 Windsurf 的问题，包括在使用模型迭代时出现内部错误，以及文件编辑功能的问题。
   - 一些用户认为上下文长度（context length）可能会影响性能，而另一些用户则强调需要更高效的确认流程。
- **DeepSeek 的性能与成本**：用户对 DeepSeek 频繁的确认提示及其在有效处理 Agent 任务方面的能力表示担忧。
   - 用户指出，对 DeepSeek 性能缓慢和循环响应的持续挫败感损害了编码工作流。
- **额度使用量与成本上升**：报告显示，用户正在迅速耗尽其 Flow Action 和 Pro 额度（credits），导致他们开始考虑 Cursor 等替代 IDE。
   - 针对订阅定价，讨论了降低额度成本和提高额度使用透明度的建议。
- **MCP 集成与文件访问**：集成 MCP 服务器以及 `.codeiumignore` 的功能是热门话题，特别是涉及与 `.gitignore` 相关的文件访问问题。
   - 用户对无法取消忽略项目中的某些文件夹表示沮丧，这导致了对忽略列表预期用途的困惑。
- **AI 编码模型对比**：用户对 Claude、Sonnet 和 DeepSeek 等模型进行了对比，强调了在完成编码任务时的性能差异。
   - 讨论集中在每个模型的使用体验上，指出虽然 Claude 在某些领域表现出色，但 DeepSeek 的免费地位使其具有竞争优势。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.codeium.com/windsurf/mcp#mcp-integration">Cascade MCP Integration</a>：未找到描述</li><li><a href="https://tenor.com/view/albert-einstein-lol-think-be-smart-think-wise-gif-8735407">Albert Einstein Lol GIF - Albert Einstein Lol Think - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://codeium.canny.io/feature-requests/p/devcontainer-support">Devcontainer Support | Feature Requests | Codeium</a>：希望能有更多 devcontainer 支持。特别是：在容器中重建并重新打开（目前只有“在容器中重新打开”）。需要它来安装扩展。</li><li><a href="https://21st.dev/magic">Magic - AI Agent for Your IDE That Creates Professional UI Components | 21st.dev</a>：使用理解现代组件模式（Modern Component Patterns）的 AI Agent 改造你的 IDE。在几秒钟内创建生产级 UI 组件，而不是花费数小时。专为追求美观、一致性的开发者打造...</li><li><a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>：需要帮助？联系我们的支持团队以获取个性化协助。</li><li><a href="https://codeium.canny.io/feature-requests/p/placeholder-input-does-not-change-when-changing-windsurf-open-chat-with-cascade">Placeholder input does not change when changing `Windsurf: Open Chat with Cascade` keybind | Feature Requests | Codeium</a>：见附带的截图</li><li><a href="https://codeium.canny.io/">Codeium Feedback</a>：向 Codeium 团队提供反馈，以便我们做出更明智的产品决策。由 Canny 提供支持。</li><li><a href="https://tenor.com/view/japanese-tyranno-dance-japanese-dance-tyranno-gif-10262458857606665890">Japanese Tyranno Dance Japanese Dance GIF - Japanese Tyranno dance Japanese dance Tyranno - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/mr-bean-mrbean-bean-mr-bean-holiday-mr-bean-holiday-movie-gif-3228235746377647455">Mr Bean Mrbean GIF - Mr bean Mrbean Bean - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.codeium.com/team/team_settings">Windsurf Editor and Codeium extensions</a>：Codeium 是开发者喜爱且企业信赖的 AI 代码助手平台。同时也是首个 Agent 模式 IDE —— Windsurf 的构建者。</li><li><a href="https://github.com/Exafunction/codeium-react-code-editor">GitHub - Exafunction/codeium-react-code-editor: AI-enabled code editor for React. Unlimited AI autocomplete capabilities with full Typescript support.</a>：适用于 React 的 AI 赋能代码编辑器。具有完整 Typescript 支持的无限 AI 自动补全能力。</li><li><a href="https://github.com/blurrah/mcp-graphql">GitHub - blurrah/mcp-graphql: Model Context Protocol server for GraphQL</a>：适用于 GraphQL 的 Model Context Protocol 服务器。通过在 GitHub 上创建账号为 blurrah/mcp-graphql 的开发做出贡献。</li><li><a href="https://docs.codeium.com/windsurf/mcp>">Welcome to Codeium - Codeium Docs</a>：未找到描述</li><li><a href="https://www.vxreddit.com/r/Codeium/comments/1irkro8/are_you_kidding_me/">no title found</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1341485038566510803)** (354 条消息🔥🔥): 

> `Unsloth AI 访谈亮点、免费 GPU 资源、协作模型训练工作、Reasoning GRPO 文档改进、个人背景与家庭动态` 


- **Unsloth AI 访谈亮点**：Unsloth AI 背后的团队参加了 GitHub Universe 访谈，讨论了他们的项目及其对 AI 开发的影响，现在可以在[这里](https://www.youtube.com/watch?v=lyVxD0bJDOk)观看。访谈强调了 Han 氏兄弟在推动项目前进过程中的协同作用。
   - 许多社区成员表达了兴奋之情，并赞赏访谈中展现出的清晰逻辑和热情。
- **免费 GPU 资源**：Unsloth 正在利用通过 Colab 等平台提供的免费 GPU，并在社区中分享了访问链接。重点在于如何有效利用这些资源来支持模型 fine-tuning 工作。
   - 参与成员讨论了围绕这些免费资源创作内容的可能性，以提高知名度和利用率。
- **协作模型训练工作**：开发者们正致力于改进模型训练流程，包括整合反馈机制以及为 DeepSeek-R1 等模型优化目标。多位社区成员表示渴望为持续的改进和测试做出贡献。
   - 建立了一个注重清晰和协作沟通的机制，以支持面向初学者的研讨会和模型工程文档。
- **Reasoning GRPO 文档改进**：Reasoning GRPO 的文档已显著增强，邀请社区成员对更新提供反馈。这一举措彰显了 Unsloth 对用户清晰度和全面指导的承诺。
   - 寻求优化模型交互的用户提出了关于实现细节的问题，例如 GRPO 期间的 temperature 设置。
- **个人背景与家庭动态**：对话中包含了围绕家庭动态的个人轶事，一些成员分享了他们的兄弟姐妹关系和成长经历。这些交流增强了参与者之间的社区感和共鸣。
   - 这种个人分享在技术讨论之余营造了轻松的氛围，展示了社区协作背后的人文一面。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/nanotron/ultrascale-playbook">The Ultra-Scale Playbook - nanotron 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/r1-1776-GGUF">unsloth/r1-1776-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=lyVxD0bJDOk">与 Unsloth.AI 共度创业周三</a>: 结识 Daniel 和 Michael Han，这对澳大利亚兄弟正通过 Unsloth 改变 AI 开发。他们的开源项目使模型 fine-tuning 速度提升了 2 倍...</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here">初学者？从这里开始！ | Unsloth 文档</a>: 未找到描述</li><li><a href="https://x.com/github/status/1892020548433043470">来自 GitHub (@github) 的推文</a>: 📺 明天东部时间晚上 8 点：加入我们，观看 @UnslothAI 关于内存高效模型 fine-tuning 的演示。我们将在聊天室实时回答您的问题！#githubstartupwednesday...</li><li><a href="https://github.com/facebookresearch/memory">GitHub - facebookresearch/memory: Memory 层使用可训练的 key-value 查找机制，在不增加 FLOPs 的情况下为模型添加额外参数。从概念上讲，稀疏激活的 memory 层补充了计算密集型的稠密 feed-forward 层，提供了廉价存储和检索信息的专用容量。</a>: Memory 层使用可训练的 key-value 查找机制，在不增加 FLOPs 的情况下为模型添加额外参数。从概念上讲，稀疏激活的 memory 层补充了计算密集型的稠密 f...</li><li><a href="https://huggingface.co/unsloth/">unsloth (Unsloth AI)</a>: 未找到描述</li><li><a href="https://github.com/vllm-project/vllm/issues/13486.">vllm-project/vllm</a>: 一个用于 LLMs 的高吞吐量且内存高效的推理和提供服务引擎 - vllm-project/vllm</li><li><a href="https://docs.unsloth.ai/basics/reasoning-grpo-and-rl">Reasoning - GRPO &amp; RL | Unsloth 文档</a>: 使用 Unsloth 通过 GRPO 训练您自己的 DeepSeek-R1 推理模型，GRPO 是强化学习 (RL) fine-tuning 的一部分。</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 以 2 倍的速度和减少 70% 的内存 fine-tune Llama 3.3、DeepSeek-R1 和推理 LLMs！🦥</a>: 以 2 倍的速度和减少 70% 的内存 fine-tune Llama 3.3、DeepSeek-R1 和推理 LLMs！🦥 - unslothai/unsloth</li><li><a href="https://github.com/huggingface/trl/pull/2810#issue-comment-box">用于自定义多步 rollout 的 GRPO 环境 (仅限 vLLM)，由 willccbb 提交 · Pull Request #2810 · huggingface/trl</a>: 此 PR 的作用是什么？在 trl/environments 下为 Environment 对象添加了一个协议，该对象封装了 vLLM 的 .generate(...) 以允许自定义 rollout 逻辑，并向 Trai... 添加了一个可选的 env 字段。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1341494082765066241)** (28 条消息🔥): 

> `Unsloth AI 在 AlphaSignal 中被提及，bitsandbytes 仓库讨论，Unsloth 艺术创作，Quantum computing 进展` 


- **Unsloth AI 在 AlphaSignal 中被提及**：Unsloth 在今天的 AlphaSignal 邮件中被点名，引发了社区成员的热烈讨论和兴奋。
   - *Awesome to see!* 是反映社区参与度的共同心声。
- **关于 bitsandbytes 代码的讨论**：成员们讨论了 [bitsandbytes/csrc/ops.cu](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/86b6c37a8ad448230cedb60753f63150b603a112/csrc/ops.cu#L86) 中的代码，并对实现中使用的 float types 和 block sizes 表示担忧。
   - 有人指出代码理想情况下应该是 `fp32`，但一些成员遇到了 `fp16` 值，从而引发了关于 pointer conversions 的疑问。
- **Unsloth 独特的艺术创作**：在一次轻松的交流中，透露出一些 Unsloth 艺术作品（如 3D 树懒）是 AI 生成的，而贴纸则是由聘请的艺术家创作的。
   - 社区对这些艺术品表示赞赏，称其为 **great art**，并认可了其中的才华。
- **对 Quantum Computing 的兴奋**：一位成员分享了一个名为 *Majorana 1 Explained: The Path to a Million Qubits* 的 YouTube 视频链接，讨论了 Quantum computing 的突破。
   - 随后出现了关于该技术需要 helium fridge（氦制冷机）的评论，以及对硬件 coherence timing（相干时间）的询问。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://youtu.be/wSHmygPQukQ?si=4VyaksRGdCXpnNeE">Majorana 1 Explained: The Path to a Million Qubits</a>: 听取来自 Microsoft 团队关于最近在物理学和 Quantum computing 领域取得突破的介绍，该突破由全新的 Majorana 1 芯片展示，该芯片由一个全...</li><li><a href="https://github.com/bitsandbytes-foundation/bitsandbytes/blob/86b6c37a8ad448230cedb60753f63150b603a11">GitHub - bitsandbytes-foundation/bitsandbytes at 86b6c37a8ad448230cedb60753f63150b603a112</a>: 通过针对 PyTorch 的 k-bit 量化实现易用的 LLM。 - GitHub - bitsandbytes-foundation/bitsandbytes at 86b6c37a8ad448230cedb60753f63150b603a112</li><li><a href="https://github.com/bitsandbytes-foundation/bitsandbytes/blob/86b6c37a8ad448230cedb60753f63150b603a112/csrc/ops.cu#L86">bitsandbytes/csrc/ops.cu at 86b6c37a8ad448230cedb60753f63150b603a112 · bitsandbytes-foundation/bitsandbytes</a>: 通过针对 PyTorch 的 k-bit 量化实现易用的 LLM。 - bitsandbytes-foundation/bitsandbytes
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1341492855473967114)** (95 条消息🔥🔥): 

> `微调模型，运行 Unsloth，LLM 相关问题，AI 模型的硬件需求，API 限制` 


- **在自定义数据集上进行微调的困扰**：一些用户正在寻找在资源有限的情况下，在特定数据集上微调 *Qwen* 等模型的方法，通常利用 Colab 或 Kaggle 等提供 GPU 访问的平台。
   - 用户对某些数据集的兼容性提出了担忧，特别是关于纯文本数据与训练相关的 embeddings 之间的关系。
- **在不同系统上安装 Unsloth**：用户请求在各种环境下安装 *Unsloth* 的帮助，并强调目前它需要 NVIDIA GPU，在 AMD 硬件上可能无法正常工作。
   - 对于系统不兼容的用户，建议使用 Colab 等云服务，而不是本地安装。
- **不同硬件配置的性能**：用户讨论了他们的硬件配置，指出笔记本电脑上 4GB VRAM 等硬件会限制性能，且为了获得最佳性能，高 RAM 是必要的。
   - 用户对不同配置的效率进行了比较，一些用户报告在某些配置下速度约为 1 token/sec。
- **API 和模型部署问题**：用户对在线服务的 API 限制和高冷却时间（cooldowns）表示沮丧，认为在配置良好的本地机器上性能可能更可靠。
   - 由于目前面临速率限制（rate limits）和服务器响应问题，用户正在寻找各种在线服务的替代方案。
- **针对有限资源的轻量级 VLM 探索**：用户寻求适合在资源有限的平台上进行微调的小型 VLM 指导，表明需要更易于获取的选项。
   - 对话强调了小型蒸馏模型（distilled models）的潜力，但也承认了当前硬件配置带来的局限性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/trl/en/index">TRL - Transformer Reinforcement Learning</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(3B)-GRPO.ipynb">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/mtsku/SnakeGPT">mtsku/SnakeGPT · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/basics/reasoning-grpo-and-rl">Reasoning - GRPO &amp; RL | Unsloth Documentation</a>: 使用 Unsloth 通过 GRPO（强化学习 RL 微调的一部分）训练你自己的 DeepSeek-R1 推理模型。</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit/blob/main/tokenizer_config.json#L193">tokenizer_config.json · unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit at main</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1341695437136199700)** (6 messages): 

> `Unsloth single GPU training, med-r1 model release, RAG vs. Fine Tuning video, Kolo usage tutorial` 


- **Unsloth 在单 GPU 训练方面表现出色**：一位成员强调，由于其速度优于其他训练器，**Unsloth** 在单 GPU 训练中表现卓越。
   - 他们还提到了一款支持多 GPU 的开源训练器，并强调了其在 Linux 或 WSL 上进行训练的简便性。
- **med-r1 模型介绍**：**med-r1 模型**最近发布，拥有 **1B 参数**，并在医疗推理数据集上进行了训练，可在 [Hugging Face](https://huggingface.co/Imran1/Med-R1-v1) 上获取。
   - 该模型旨在用于医疗问答和诊断推理，支持 **4-bit 推理**，最大序列长度为 **2048 tokens**。
- **关于 RAG 与 Fine Tuning 的辩论**：一段名为“[RAG vs. Fine Tuning (Live demo)](https://www.youtube.com/watch?v=LDMFL3bjpho)”的 YouTube 视频讨论了哪种方法论在 AI 训练中能产生更好的效果。
   - 观众对更多 RAG 与 Fine Tuning 的对比案例表示出兴趣，这促成了未来制作 Kolo 详细教程的计划。
- **正在制作中的未来 Kolo 教程**：针对获取更多信息的需求，一位成员表示计划推出一段专注于 **Kolo** 使用的后续视频，并将包含更全面的测试。
   - 即将推出的教程旨在深入探讨所使用的训练数据和方法论的更多细节。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/Imran1/Med-R1-v1">Imran1/Med-R1-v1 · Hugging Face</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=LDMFL3bjpho">RAG vs. Fine Tuning (Live demo)</a>：RAG 和 Fine Tuning 哪个更好？行业是否搞错了？Fine Tuning 能否提供比传统 RAG 系统更好的结果？观看视频...</li><li><a href="https://github.com/rombodawg/Easy_training">GitHub - rombodawg/Easy_training</a>：通过在 GitHub 上创建账号来为 rombodawg/Easy_training 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1341483251901927556)** (116 messages🔥🔥): 

> `AI 驱动的植物化学配方、公民科学贡献、学术出版的挑战、LLM 中的情感内容、用于心理健康的营养保健品` 


- **AI 驱动的植物化学配方框架解析**：通过[这篇博文](https://www.marielandryceo.com/2025/02/title-ai-powered-phytochemical.html?m=1)介绍了一个系统识别植物化合物以优化健康支持的 AI 驱动模型。该方法旨在加强循证营养保健品的开发，同时强调安全性和有效性。
- **关于公民科学家合法性的辩论**：对话反映了对自封为“公民科学家”人士合法性的看法，强调尽管缺乏正式学术头衔，科学贡献仍可来自不同背景。成员们表示担心，排斥非传统贡献者可能会扼杀创新并阻碍有价值的见解。
- **对学术出版标准的批评**：参与者讨论了学术出版的僵化标准，批评其为非传统背景人士参与科学对话设置了障碍。他们指出，出版通常需要背书，这可能会限制更广泛的知识共享参与。
- **AI 模型中的情感内容分类**：一位成员建议，针对情感内容训练的 Attention 机制可以提高 AI 处理主观话题的能力，尽管区分社会细微差别仍然具有挑战性。讨论强调了 AI 在审核敏感内容时需要可靠的分类系统。
- **针对性营养保健品的用例**：分享了开发定制 GPT 模型以生成针对特定疾病的植物分子补充剂配方的案例。讨论的例子包括针对**边缘性人格障碍 (Borderline Personality Disorder)**、**自闭症谱系障碍 (Autism Spectrum Disorder)** 和**精神分裂症 (Schizophrenia)** 的配方，并强调这些属于研究导向，需谨慎对待。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/unsloth/r1-1776">unsloth/r1-1776 · Hugging Face</a>：未找到描述</li><li><a href="https://bits.logic.inc/p/the-eagles-will-win-super-bowl-lix">使用 LLM 预测超级碗</a>：使用 LLM 挑选 NFL 获胜者，表现优于 94.5% 的人类。</li><li><a href="https://github.com/wrmedford/moe-scaling/blob/main/README.md">moe-scaling/README.md at main · wrmedford/moe-scaling</a>：Mixture of Experts 模型的 Scaling Laws。通过在 GitHub 上创建一个账户，为 wrmedford/moe-scaling 的开发做出贡献。</li><li><a href="https://www.marielandryceo.com/2025/02/title-ai-powered-phytochemical.html?m=1">标题：AI 驱动的植物化学配方：一种支持健康的数据驱动方法</a>：未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1341459247510061247)** (84 条消息🔥🔥): 

> `GPU 对比, Grok 3 vs ChatGPT-4, Hugging Face Playground 问题, LangChain 框架发布, 针对 LLM 的 SWE-Lancer 基准测试` 


- **GPU 辩论：2080 Ti vs P40s**：一场关于 **2080 Ti** 和 **P40s** 在不同 AI 训练应用中价值对比的讨论展开了。
   - 成员们就每种 GPU 性能的优势进行了辩论，但未达成共识。
- **Grok 3 被宣称优于 ChatGPT-4**：据报道 **Grok 3** 优于 **ChatGPT-4**，这引发了对其能力的关注。
   - 社区成员对这些模型之间的技术差异表达了好奇和推测。
- **Hugging Face Playground 出现宕机**：用户报告 Hugging Face Playground 无法访问，访问时出现 **500 Internal Error**。
   - 成员们确认了该问题，并提到已联系团队寻求支持。
- **LangChain 框架发布**：新 **LangChain 框架** 的发布引起了轰动，该框架强调 LLM 的动态指令学习。
   - 分享了一个教程视频，邀请用户体验用于构建自我改进型 Agent 的更新功能。
- **LLM 自由职业新基准：SWE-Lancer**：OpenAI 推出了 **SWE-Lancer** 基准测试，用于测试 LLM 执行价值高达 100 万美元软件工程任务的能力。
   - 初步见解表明，模型在方案选择方面的表现优于执行实现，突显了它们的优缺点。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/learn">Hugging Face - Learn</a>: 未找到描述</li><li><a href="https://huggingface.co/playground">Playground - Hugging Face</a>: 未找到描述</li><li><a href="https://saiyan-world.github.io/goku/">Goku</a>: 未找到描述</li><li><a href="https://www.youtube.com">YouTube</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=Nc3vIuPyQQ0">AI CEO Alexandr Wang | This Past Weekend w/ Theo Von #563</a>: Alexandr Wang 是 Scale AI 的创始人兼 CEO，该平台为 AI 程序提供数据训练。2021 年，他被评为最年轻的白手起家的亿万富翁...</li><li><a href="https://x.com/_philschmid/status/1891780812497887289?t=nkpIbh2A9B0wScpC0iJlqg&s=19">Philipp Schmid (@_philschmid) 的推文</a>: LLM 能通过软件工程赚到 100 万美元吗？SWE-Lancer 是来自 @OpenAI 的新基准测试，测试 LLM 是否能完成来自 Upwork 的自由职业任务，总价值达 100 万美元的真实世界报酬...</li><li><a href="https://tenor.com/view/office-space-yeah-uh-yeah-unsure-uh-sure-gif-5638327">Office Space Yeah GIF - Office Space Yeah Uh Yeah - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://x.com/Nouamanetazi/status/1892274582503248178">Nouamane Tazi (@Nouamanetazi) 的推文</a>: 🚀 很高兴发布 *THE* Ultra-Scale Playbook —— 一份关于从 1 个到数千个 GPU 训练 LLM 的全面指南！</li><li><a href="https://www.youtube.com/watch?v=WW-v5mO2P7w">构建自我改进型 Agent：LangMem 程序化记忆教程</a>: 学习如何使用 LangMem SDK 在 LLM Agent 中实现动态指令学习。本技术教程演示了自动 Prompt 优化...</li><li><a href="https://github.com/rombodawg/Easy_training/tree/main/Galore%2BQlora_With_Multi_GPU_Support">rombodawg/Easy_training 的 Galore+Qlora_With_Multi_GPU_Support 分支</a>: 通过在 GitHub 上创建账号，为 rombodawg/Easy_training 的开发做出贡献。</li><li><a href="https://qiita.com/takeuchiseijin/items/909c48b57127a37fbd12">PyTorch 的 AMP 应该使用 bf16。大概率不会再出现 nan。 - Qiita</a>: 官方给出的用法如下。但如果模型包含数值不稳定的计算（Softmax, division by epsilon...），迟早会出现 nan。顺便一提，Tr…</li><li><a href="https://github.com/huggingface/datasets/pull/6968">使用 `HF_HUB_OFFLINE` 代替 `HF_DATASETS_OFFLINE`，由 Wauplin 提交 · Pull Request #6968 · huggingface/datasets</a>: 要离线使用数据集，可以使用 HF_DATASETS_OFFLINE 环境变量。此 PR 使 HF_HUB_OFFLINE 成为离线训练的推荐环境变量。目标是更加一致...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1341471799807971460)** (3 条消息): 

> `Quantum Computing, Majorana 1 Chip, Neuralink Image Analysis` 


- **Microsoft 的 Majorana 1 芯片发布**：在最近的一篇帖子中，**Satya Nadella** 宣布推出了 **Majorana 1** 芯片，这是 Quantum Computing 领域的一项重大进展，有望在几分钟内完成超级计算机需要 **数十亿年** 才能完成的计算。
   - 一篇博客文章讨论了这一创新，解释了其重塑行业和影响气候变化的潜力。
- **Neuralink 图像分析分享**：对过去两天 **Neuralink** 进展的多张图像进行了分析，展示了各种见解和细节。
   - 附带的图像提供了 Neuralink 进展的视觉呈现，但消息中缺乏具体的描述。



**提到的链接**：<a href="https://kuberwastaken.github.io/blog/Technology/Majorana-1---Why-Quantum-Computing-Matters-Now">Majorana 1 - Why Quantum Computing Matters Now</a>：简介：计算的一个潜在新时代。想象一下，一台功能如此强大的计算机，它可以在几分钟内解决当今最快的超级计算机需要数十亿年才能解决的问题……

  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1341477367255928892)** (14 条消息🔥): 

> `动态 Readme 图像，使用 LLaMA 的幽默版 LIMA 数据集，情感市场预测项目，Aster 音频搜索应用，针对 Python 代码的 CommentRescueAI` 


- **轻松创建动态 Readme 图像**：一位成员分享了一个 GitHub 仓库，提供了一个模板，用于使用 GitHub workflows 和 Puppeteer 为各种应用生成动态更新的图像。
   - 该项目在 MIT License 下开源，旨在增强 Readme 和博客的视觉效果。
- **LIMA 数据集迎来风趣改版**：一位用户幽默地修改了 LIMA 数据集，使用 LLaMA-3.1-70B-Instruct-Turbo 模型添加了喜剧化的回答，使交互更具吸引力。
   - 生成的数据集可以在 [这里](https://huggingface.co/datasets/pulkitmehtawork/lima_humorous) 找到，创建者欢迎大家对这种轻松的方法提供反馈。
- **情感市场预测项目发布**：一位成员介绍了他们的 GitHub 项目，专注于通过使用 Hugging Face Transformer 模型对新闻标题进行情感分析来预测市场趋势。
   - 该项目采用了 LSTM 模型并报告了令人期待的结果，邀请用户在 [GitHub](https://github.com/yannis-gerontopoulos99/sentiment-market-forecasting) 上查看。
- **探索 Aster：一款开源音频搜索应用**：一位用户发布了一篇博客，详细介绍了使用 HF Laion CLAP 模型的 Aster 音频搜索应用，并寻求关于其清晰度的反馈。
   - 另一篇博客文章讨论了 ONNX 和 PyTorch 实现之间的进一步性能比较，并强调了批处理支持的局限性。
- **介绍针对 Python 的 CommentRescueAI**：一位成员展示了 CommentRescueAI，这是一个 Web 扩展程序，旨在帮助轻松地为 Python 代码添加 AI 生成的 docstrings 和注释。
   - 该扩展现已在 VS Code 市场上线，创建者正在寻求改进建议和功能创意。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.kaggle.com/code/allanwandia/secondary-structure-data-analysis">二级结构数据分析</a>：使用 Kaggle Notebooks 探索并运行机器学习代码 | 使用来自多个数据源的数据</li><li><a href="https://asteraudio.app/blog/whats-it-for">Aster</a>：未找到描述</li><li><a href="https://asteraudio.app/blog/webgpu-wasm-cuda">Aster</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=h7dijJBvwWA">Deepseek-R1 AI 通过观察我的屏幕编写 Obsidian 笔记（开源）</a>：立即免费下载：https://screenpi.pe</li><li><a href="https://github.com/rombodawg/Easy_training">GitHub - rombodawg/Easy_training</a>：通过在 GitHub 上创建账户来为 rombodawg/Easy_training 的开发做出贡献。</li><li><a href="https://blog.chataignon.org/joseph/2025-01-17/why-chatgpt-cant-spell-properly/">为什么 ChatGPT 拼写不准</a>：未找到描述</li><li><a href="https://github.com/Kuberwastaken/Dynamic-Readme-Images">GitHub - Kuberwastaken/Dynamic-Readme-Images: 在 Readme 中获取“动态更新”图片的模板仓库</a>：一个在 Readme 中获取“动态更新”图片的模板仓库 - Kuberwastaken/Dynamic-Readme-Images</li><li><a href="https://huggingface.co/datasets/pulkitmehtawork/lima_humorous">pulkitmehtawork/lima_humorous · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://github.com/yannis-gerontopoulos99/sentiment-market-forecasting">GitHub - yannis-gerontopoulos99/sentiment-market-forecasting</a>：通过在 GitHub 上创建账户来为 yannis-gerontopoulos99/sentiment-market-forecasting 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1341514400619692153)** (3 条消息): 

> `Discord 邀请规则，DeepSeek R1 蒸馏模型，对话存储解决方案` 


- **Discord 邀请受到质疑**：一位成员担心 **Discord 邀请** 可能违反了频道 <#895532661383254098> 的社区准则。
   - 该消息语气友好，末尾带有笑脸符号。
- **在 DeepSeek 模型中利用简单 SFT 数据**：有人好奇如何在不使用推理的情况下，将 **简单 SFT 数据** 与 **DeepSeek R1 蒸馏模型** 结合使用，同时在推理过程中仍能实现 **Chain of Thought (COT)**。
   - 该成员正在寻求社区对此事的建议和帮助。
- **对话存储解决方案寻求建议**：一位开发者正在寻求关于聊天应用最佳 **存储解决方案** 的建议，考虑了针对不同对话量的 SQL 和 **NoSQL 数据库**。
   - 他们询问了是否有任何已实现的存储长篇聊天对话的解决方案，包括像 **S3** 这样的云存储选项。


  

---

### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1341617148250030080)** (4 条消息): 

> `Fine-tuning 训练时间，M3 Max 在 MPS 设备上的错误，课程扩展计划，关于 Agents 的第二单元链接` 


- **3090 Fine-tuning 时间受到质疑**：一位成员估计 Fine-tuning notebook 在 **3090** 上需要 **2 小时 30 分钟**，并询问这是否是可接受的时长。
   - 他们注意到在 **M3 Max** 上通过 CPU 运行只需 **2 小时**，这表明 3090 的速度可能比预期的要慢。
- **M3 Max 在 MPS 设备上遇到错误**：在尝试于 **M3 Max** 上运行 Fine-tuning notebook 时，一位成员遇到了一个错误，提示 *'The operator 
   - 这表明 **MPS 设备** 缺乏对 Fine-tuning 所需某些操作的支持。
- **讨论中的未来课程扩展**：一位成员询问了关于扩展现有课程的**未来计划**，暗示了对持续学习的兴趣。
   - 这反映了社区对额外材料的期待。
- **请求 Agents 单元链接**：一位成员询问了专注于 Agents 的**第二单元链接**，渴望获取更具体的内容。
   - 这表明了对结构化获取课程材料和信息的需求。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1341458225689661522)** (333 条消息🔥🔥): 

> `证书生成问题，Hugging Face AI Agents，第二单元发布，社区建设，使用 API Keys 和 Tokens` 


- **证书生成问题已解决**：许多用户最近在完成测验后难以生成证书，但通过将提交定向到一个直接生成证书的新 Space，该问题已得到解决。
   - 鼓励参与者尝试更新后的测验链接，以便及时领取证书。
- **探索 Hugging Face AI Agents**：用户正在积极讨论训练 AI Agents 的最佳实践，包括使用 RAGs 和 API 调用等工具来实现功能。
   - 参与者表达了对项目协作以及在各种应用中有效使用 AI 工具的兴趣。
- **即将发布的第二单元**：大家对第二单元的发布充满期待，有提到内容预计在 2 月 18 日发布。
   - 当用户没有看到预期的单元时产生了困惑，从而引发了对发布状态的进一步询问。
- **建立社区联系**：许多参与者借此机会介绍了自己并表达了自己的背景，特别是在软件开发和 AI 领域。
   - 重点在于建立联系并在学习和开发 AI 项目中互相支持。
- **Agents 的 API Key 管理**：一些讨论围绕着开发 AI Agents 时管理 API Keys 和 secrets 的重要性展开。
   - 用户分享了设置环境变量的技巧，以及在项目中安全存储和利用这些 keys 的最佳实践。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://share.1password.com/s#dpSJqyL8KP5ZWjQXdnVB2EHPOeW6jFyiH30fpkJYLCs">我正在使用 1Password 与您分享一个项目</a>：1Password 是一款密码管理器、数字保险箱、表单填写器和安全数字钱包。1Password 会为您记住所有密码，帮助保护账户信息安全。</li><li><a href="https://huggingface.co/spaces/agents-course/First_agent">First Agent - agents-course 发布的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/learn/agents-course/unit1/dummy-agent-library">Dummy Agent Library - Hugging Face Agents 课程</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1SVS-ALf9ToN6I6WmJno5RQkZEHFhaykJ).">Google Colab</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/vitpolis/First_agent_template">First Agent Template - vitpolis 发布的 Hugging Face Space</a>：未找到描述</li><li><a href="https://agents-course-unit-1-quiz.hf.space/">Dataset Quiz for agents-course/unit_1_quiz</a>：未找到描述</li><li><a href="https://lightning.ai/someshfengde/studios/bonus-unit-1-lightningai">Bonus Unit 1 lightningai - someshfengde 发布的 Lightning Studio</a>：通过使用 lightning.ai studio 的 GPU，在 15 分钟内使用 LoRA 免费微调 gemma-2b-it 以启用 function calling。</li><li><a href="https://huggingface.co/spaces/agents-course/First_agent_template">First Agent Template - agents-course 发布的 Hugging Face Space</a>：未找到描述</li><li><a href="https://github.com/huggingface/agents-course/blob/main/notebooks/bonus-unit1/gemma-SFT-thinking-function_call.ipynb">agents-course 仓库 main 分支下的 gemma-SFT-thinking-function_call.ipynb</a>：此仓库包含 Hugging Face Agents 课程。</li><li><a href="https://x.com/dylan_ebert_/status/1892000664600887686">来自 dylan (@dylan_ebert_) 的推文</a>：什么是 AI Agents？70 秒带你了解</li><li><a href="https://www.youtube.com/watch?v=qU3fmidNbJE">21 分钟掌握 AI Agents 基础</a>：通过我与 Hubspot 合作制作的免费 Prompting 快速入门指南提升您的 AI 技能：https://clickhubspot.com/1gg9。想要在...领域取得领先...</li><li><a href="https://github.com/huggingface/agents-course/blob/main/units/en/unit1/what-are-llms.mdx?plain=1).">agents-course 仓库 main 分支下的 what-are-llms.mdx</a>：此仓库包含 Hugging Face Agents 课程。</li><li><a href="https://www.youtube.com/watch?v=ZZ2QUCePgYw">AI agents 简介</a>：Vertex AI Agent Builder 快速入门 → https://goo.gle/3UPJ7dN。使用 Genkit 构建 GenAI 驱动的应用 → https://goo.gle/4fCSTrK。由 Googlers Aja Hamme 揭秘 AI agents...</li><li><a href="https://www.youtube.com/watch?v=45YtsZbz334">理解 AI 中的 Tools 与 Function Calling</a>：理解 AI 中的 Tools 与 Function-Calling | 非官方 Hugging Face Agents 课程辅助视频。奖励单元 1。为 Function-calling 微调 LLM。课程...</li><li><a href="https://github.com/andrewyng/aisuite">GitHub - andrewyng/aisuite：适用于多个生成式 AI 提供商的简单、统一接口</a>：适用于多个生成式 AI 提供商的简单、统一接口 - GitHub - andrewyng/aisuite</li><li><a href="https://youtu.be/tlV_uxodcQw">使用 Hugging Face 创建 AI Agents！关于他们最新免费课程的感想！</a>：🚀 使用 Hugging Face 创建 AI Agents！🤖 加入我，分享我使用 Hugging Face Agents 课程构建简单 AI agent 的历程！无论您是新手...</li><li><a href="https://github.com/gradio-app/gradio/issues/9895">端点提供的参数过多 - 浏览器控制台中的 JavaScript 警告 · Issue #9895 · gradio-app/gradio</a>：描述 Bug：在使用 gradio.Chatbot 或 gradio.ChatInterface 启动 gradio 应用时，每次提交消息，浏览器控制台都会出现 JavaScript 警告：Too many ar...</li><li><a href="https://huggingface.co/datasets/agents-course/certificates">agents-course/certificates · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://acrobat.adobe.com/id/urn:aaid:sc:EU:38802316-7b5c-48f5-b4a1-c5437d0a48f5">Adobe Acrobat</a>：未找到描述</li><li><a href="https://aiagentslist.com/">AI Agents 目录 - 比较顶级 AI 助手</a>：探索我们精心挑选的 AI agents 集合。通过详细的对比、评论和功能分析，找到最适合您需求的 AI 助手。</li><li><a href="https://t.me/AgentNexus)">Telegram – 开启即时通讯新时代</a>：快速。安全。强大。</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1341457241966116864)** (411 条消息🔥🔥🔥): 

> `Grok 3 发布、模型对比、AI 编程、Aider 及其集成、OpenRouter 的局限性`

- **Grok 3 开放免费使用**：全球最智能的 AI 模型 Grok 3 现已开放免费使用（直至达到服务器容量上限），同时为 X Premium+ 用户和 SuperGrok 会员提供更多访问权限。
   - 此次更新还包括 Voice Mode 等功能的早期访问权限，引发了社区的热烈讨论。
- **对查询限制的担忧**：用户推测 Grok 3 可能会有限制，预计每日查询次数约为 5 次。
   - 这引发了对可用性的担忧，特别是对于那些习惯于与模型进行大量交互的用户。
- **关于 AI 模型效率的讨论**：对话涉及了一个名为 LLADA 的新模型，该模型在编程任务中表现出极高的性能。
   - 观点表明，由于其创新的方法，像 LLADA 这样的模型可能会成为代码编辑的强力替代方案。
- **AI 与 Aider 的集成**：关于各种 AI 模型与 Aider 集成的辩论仍在继续，重点在于它们如何影响编程效率。
   - 一些用户希望 Aider 能够教授编程方法论，而不仅仅是依赖 AI 生成的解决方案。
- **OpenRouter 与资源限制**：讨论显示 OpenRouter 可能会将 o3-mini 等模型的端点限制在 100k 次请求以内，这表明可能存在可用性问题。
   - 这一限制使得试图利用 OpenRouter 进行更密集 AI 交互的用户体验变得复杂。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/benchmarks.html#gpt-code-editing-benchmarks">GPT 代码编辑基准测试</a>：使用基于 Exercism Python 练习的新代码编辑基准套件，对 GPT-3.5 和 GPT-4 的代码编辑技能进行基准测试。</li><li><a href="https://www.kaggle.com/competitions/konwinski-prize">Konwinski Prize</a>：为能够解决 90% 新 GitHub issues 的 AI 提供 100 万美元奖金。</li><li><a href="https://harper.blog/2025/02/16/my-llm-codegen-workflow-atm/">我目前的 LLM 代码生成工作流</a>：详细介绍了我目前使用 LLM 构建软件的工作流，从头脑风暴到规划和执行。</li><li><a href="https://ai.google.dev/gemini-api/docs/models/experimental-models">未找到标题</a>：未找到描述</li><li><a href="https://x.com/ai_for_success/status/1892170935870079388">来自 AshutoshShrivastava (@ai_for_success) 的推文</a>：🚨 Grok-3 现在也开始向 Premium 用户推出了。注：我可以在 grok[.]com 上看到它，但在 X 应用上还看不到。</li><li><a href="https://x.com/iruletheworldmo/status/1891529614167789967">来自 🍓🍓🍓 (@iruletheworldmo) 的推文</a>：Grok 3 来了，它是 AGI。这不仅仅是另一个模型的发布。这是改变一切的时刻。忘掉 GPT，忘掉 Claude，忘掉你以前用过的所有 AI——它们已经过时了。</li><li><a href="https://github.com/openai/SWELancer-Benchmark">GitHub - openai/SWELancer-Benchmark：此仓库包含论文 "SWE-Lancer: Can Frontier LLMs Earn $1 Million from Real-World Freelance Software Engineering?" 的数据集和代码</a>：此仓库包含论文 "SWE-Lancer: Can Frontier LLMs Earn $1 Million from Real-World Freelance Software Engineering?" 的数据集和代码 - openai/SWELancer-Benchmark</li><li><a href="https://x.com/perplexity_ai/status/1891916573713236248?s=46&t=AZs45ckJ7UUM_kJZcxnR_w">来自 Perplexity (@perplexity_ai) 的推文</a>：今天我们开源了 R1 1776——这是 DeepSeek R1 模型的一个版本，经过后期训练，旨在提供无审查、无偏见且事实准确的信息。</li><li><a href="https://tenor.com/view/richard-attenborough-whip-whipped-whiplash-whiplashed-gif-16890874512241116786">Richard Attenborough Whip GIF - Richard Attenborough Whip Whipped - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/MoonshotAI/MoBA/blob/master/MoBA_Tech_Report.pdf">MoonshotAI/MoBA 仓库中的 MoBA_Tech_Report.pdf</a>：MoBA：用于长上下文 LLM 的混合块注意力机制 (Mixture of Block Attention) - MoonshotAI/MoBA</li><li><a href="https://x.com/OpenAI/status/1891911123517018521">来自 OpenAI (@OpenAI) 的推文</a>：今天我们发布了 SWE-Lancer——一个用于评估 AI 模型编程性能的全新且更真实的基准测试。SWE-Lancer 包含来自 Upwork 的 1,400 多个自由软件工程任务，价值...</li><li><a href="https://x.com/elonmusk/status/1891911120572567983">来自 Elon Musk (@elonmusk) 的推文</a>：@xAI Grok 3 的发布将在本周每天迅速改进。如有任何问题，请回复此贴。</li><li><a href="https://kimi.moonshot.cn/">Kimi.ai - 会推理解析，能深度思考的 AI 助手</a>：未找到描述</li><li><a href="https://x.com/xai/status/1892400129719611567">来自 xAI (@xai) 的推文</a>：就是现在：世界上最智能的 AI，Grok 3，现在免费开放（直到我们的服务器熔化为止）。现在就尝试 Grok 3：https://x.com/i/grok。X Premium+ 和 SuperGrok 用户将拥有更多使用 Grok 3 的权限，包括...
</li>
</ul>

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1341461614683230370)** (14 messages🔥): 

> `Aider 浏览器功能问题, LLM 推理速度担忧, 在 Aider 中使用规范 (Conventions), 字体颜色可见性问题, 将 Agents 集成到 Aider` 


- **Aider 浏览器功能引发困惑**：一位用户报告了 Aider 浏览器功能的问题，该问题导致在每次设置 API key 时产生困惑。
   - *“我没意识到每次都需要设置 API key。”*
- **LLM 推理速度是瓶颈**：一位成员对使用 Aider 编码时的 **LLM 推理速度** 表示沮丧，并询问改进方案。
   - 另一位贡献者提到，根据社区反馈，**Azure 的 OpenAI API** 通常速度更快。
- **使用规范 (Conventions) 改进编码实践**：一位用户强调了使用规范文件来指导 Aider 内部编码实践的重要性，并详细说明了特定偏好，例如使用 **httpx** 而非 **requests**。
   - 他们建议通过 `/read CONVENTIONS.md` 加载规范文件，以确保其被缓存并标记为只读。
- **字体颜色的可见性问题**：一位用户提出了 **蓝色字体** 难以阅读的问题，并询问更改字体颜色的方法。
   - 建议包括切换到深色模式，尽管另一位成员指出 **浅色模式** 并没有改善可见性。
- **对 Agents 集成的兴趣**：一位用户询问是否计划将 **Agents** 集成到 Aider 中，表现出对未来功能的兴趣。
   - 这个问题反映了对增强 Aider 能力的渴望。



**提到的链接**：<a href="https://aider.chat/docs/usage/conventions.html">指定编码规范</a>：告知 aider 在处理代码时遵循你的编码规范。

  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1341525366627369010)** (2 messages): 

> `Ministral 与 Aider 的集成, 构建过程与性能, RAG 功能对比` 


- **关于 Aider 和 Ministral 的咨询**：一位成员询问是否有人尝试过将 **Ministral** 与 **Aider** 集成，对其兼容性表示好奇。
   - 他们引用了一篇讨论 **Mistral** 的 [LinkedIn 帖子](https://www.linkedin.com/posts/deividas-mataciunas_ai-mistral-opensource-ugcPost-7294619759722074112-a_JM?utm_source=share&utm_medium=member_android&rcm=ACoAABZCz5cBsMAYVy_zzTHh2HzsmuBv_27C49Y)。
- **对构建过程缓慢的担忧**：一位成员表示，他们欣赏构建过程中类似 Git 的行为，但发现它*非常缓慢*，因为它在 API 调用期间会用 'build' 重写代码块。
   - 他们注意到它使用 **TF-IDF** 进行搜索而不是 Embeddings，这可能会阻碍上下文相似性，并建议增加一个功能以利用供应商提供的 50% **Batching** 成本优惠。
- **RAG 性能对比**：一位用户提到他们的 **RAG** 构建仍在索引中，但他们可以在此过程中使用它，并发现其效果优于 **AIChat RAG 功能**。
   - 他们还提供了构建状态更新，详细列出了已处理的文件和输入成本，强调 **暂存文件 (staged files)** 为 **1019** 个，**代码分块 (chunks)** 为 **20115** 个。


  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1341455098278051870)** (405 messages🔥🔥🔥): 

> `模型性能对比, Grok 3 见解, Cursor 与 AI 模型的使用, 初创公司协作, 使用 AI 模型进行编程`

- **AI 模型效能辩论**：用户对各种 AI 模型发表了强烈看法，许多人因其可靠性和上下文理解能力而青睐 Sonnet 处理编程任务，同时对 Grok 3 的表现表示不满。
   - 讨论强调，虽然 Grok 3 评价褒贬不一，但 Anthropic 模型在编程方面备受推崇，尤其是在提供良好上下文的情况下。
- **OpenAI 与 Anthropic 模型对比**：用户普遍认为 OpenAI 和 DeepSeek 是通用任务的强力竞争者，但指出 Anthropic 模型在结合推理和严谨执行时，在编程和复杂任务中表现出色。
   - 尽管对 Grok 3 的看法不一，许多用户表示所有模型在特定领域都有其优势，主张针对具体工作使用最合适的工具。
- **Grok 3 争议**：一些用户批评了 Grok 3 的实用性，根据个人测试经验将其贴上“令人失望”的标签，这些经验展示了它在编程任务中的不足。
   - 相比之下，其他人通过指出各种 YouTubers 的好评，质疑了反对 Grok 3 论点的有效性。
- **Cursor 使用体验**：用户分享了在 Agent 模式下使用 Cursor 的经验，强调了 changelogs 和自定义规则等功能如何帮助优化他们的工作流。
   - 讨论围绕使用 AI 助手自动化任务并确保高质量输出展开，突出了正面和负面的互动。
- **创业机会公告**：一位用户宣布有兴趣启动一个新项目，并正在寻找有时间和奉献精神的开发者进行合作，并提议为项目提供资金。
   - 这为频道成员之间的潜在合作伙伴关系打开了大门。


<div class="linksMentioned">

<strong>提到的链接</strong>：

</div>

<ul>
<li>
<a href="https://pieces.app)">未找到标题</a>: 未找到描述</li><li><a href="https://caddyserver.com/docs/">欢迎 - Caddy 文档</a>: Caddy 是一个用 Go 编写的、功能强大、企业级、支持自动 HTTPS 的开源 Web 服务器</li><li><a href="https://www.youtube.com/wat"> - YouTube</a>: 未找到描述</li><li><a href="https://x.com/windsurf_ai/status/1892322088507105561?s=46&t=ggmESCIXF0nYw8_kshHz7A">来自 Windsurf (@windsurf_ai) 的推文</a>: DeepSeek-V3 现在在 Windsurf Pro 和 Ultimate 计划中不限额。0 prompt 额度。0 flow action 额度。</li><li><a href="https://x.com/theo/status/1891960246286828028">来自 Theo - t3.gg (@theo) 的推文</a>: 想象一下在不使用任何版本控制的情况下编写一个项目 4 个月</li><li><a href="https://x.com/OpenAI/status/1891911132983722408">来自 OpenAI (@OpenAI) 的推文</a>: 目前的前沿模型无法解决大多数任务。</li><li><a href="https://x.com/kevinhou22/status/1891375289919500794?t=k5skkvhMsKodfbvKbDiYhw&s=19">来自 Kevin Hou (@kevinhou22) 的推文</a>: 我们构建 Windsurf 的 Agent 时，使其不依赖于像其他工具那样的 embedding 索引。通用的检索方式根本无法扩展到 monorepos。相反，我们的 Agent 使用人类会使用的工具...</li><li><a href="https://www.youtube.com/watch?v=gXmakVsIbF0">你一定要尝试 Cursor + o3 + Storybook</a>: 上周日晚上在日本，OpenAI 悄悄发布了由此前未公开的 O3 模型驱动的新深度研究产品。在这个视频中，我向你展示...</li><li><a href="https://x.com/karpathy/status/1891720635363254772">来自 Andrej Karpathy (@karpathy) 的推文</a>: 今天早些时候我获得了 Grok 3 的早期访问权限，我想我应该是第一批进行快速氛围检查（vibe check）的人之一。Thinking✅ 首先，Grok 3 显然拥有一个接近业界领先水平的思维模型...</li><li><a href="https://simonwillison.net/2024/Dec/19/one-shot-python-tools/">使用 uv run 和 Claude Projects 通过单次提示词构建 Python 工具</a>: 我写过很多关于如何使用 Claude 通过 Claude Artifacts 构建单次 HTML+JavaScript 应用程序的文章。最近我开始使用类似的模式来创建单次 Python 实用程序...</li><li><a href="https://github.com/disler/single-file-agents">GitHub - disler/single-file-agents: 如果我们能将单一用途、强大的 AI Agents 打包进一个 Python 文件中会怎样？</a>: 如果我们能将单一用途、强大的 AI Agents 打包进一个 Python 文件中会怎样？ - disler/single-file-agents</li><li><a href="https://github.com/onlook-dev/onlook?tab=readme-ov-file#getting-started">GitHub - onlook-dev/onlook: 面向设计者的开源 Cursor。直接在你的实时 React 应用中进行设计，并将更改发布到代码中。</a>: 面向设计者的开源 Cursor。直接在你的实时 React 应用中进行设计，并将更改发布到代码中。 - onlook-dev/onlook</li><li><a href="https://www.notion.so/Experiment-Prompting-86aa8f988fce404cbf70134690d2635a#eb3588f421e14e21b029a51993eeb65a">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>: 一款将日常工作应用融合在一起的新工具。它是为您和您的团队打造的一体化工作空间</li><li><a href="https://x.com/ryolu_/status/1891677600587629043?s=46">来自 Ryo Lu (@ryolu_) 的推文</a>: 情人节迟到了，但 @cursor_ai 一直在憋大招</li><li><a href="https://x.com/ryolu_/status/1888455169081577955?s=46">来自 Ryo Lu (@ryolu_) 的推文</a>: 在 @cursor_ai 的 4 天：• 每个人都很厉害 • @shaoruu 一直追着我要设计并实现它们 • 每周 1 次会议 • 下一个版本已经为你准备好了一系列功能，这里有一个小预览 ⚡️</li><li><a href="https://www.subframe.com/">Subframe – 快速构建 UI 的最佳方式。</a>: 使用拖拽式可视化编辑器、精美制作的组件和生产就绪的代码，在几分钟内构建出令人惊叹的 UI。针对 React 和 TailwindCSS 进行了优化。</li><li><a href="https://www.relume.io/">Relume — 使用 AI 更快地设计和构建网站 | AI 网站构建器</a>: 将 AI 作为你的设计盟友，而不是替代品。使用 Relume 的 AI 网站构建器，在几分钟内轻松生成营销网站的站点地图和线框图。</li><li><a href="https://21st.dev/">21st.dev - 设计工程师的 NPM</a>: 使用受 shadcn/ui 启发的现成 React Tailwind 组件，更快地交付精美的 UI。由设计工程师构建，为设计工程师服务。</li><li><a href="https://flexboxlabs.netlify.app/">Flexbox Labs</a>: 未找到描述
</li>
</ul>

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1341456025894522953)** (361 条消息🔥🔥): 

> `Deep Research 延迟问题、Perplexity Pro 订阅、图像生成能力、Grok 集成、R1-1776 模型更新` 


- **Deep Research 面临崩溃**：用户报告 Deep Research 功能持续崩溃，特别是企业版计划的用户。
   - 浏览线程和访问库内容时也出现了问题，引发了关于潜在修复方案的讨论。
- **对 Pro 订阅费用的疑虑**：一位用户对 Pro 订阅模式表示困惑，该模式只需支付单笔费用即可访问多个模型，这引发了对该定价策略有效性的猜测。
   - 另一位用户幽默地暗示这些模型可能是盗版的，但这很快被认为毫无根据而遭到驳回。
- **图像生成潜力**：频道强调，尽管可以使用指示图像生成能力的组件，但一些用户在平台上有效利用该功能时遇到了困难。
   - 讨论了变通方法，包括专门为图像创建生成 Prompt 以及使用浏览器插件。
- **Grok 集成更新**：关于 Grok 3 的发布及其在 Perplexity 中潜在实现的猜测，引发了对其当前状态和功能的疑问。
   - 社区成员表达了对 Grok 集成的期待和兴趣，等待团队的更新。
- **R1-1776 模型讨论**：考察了 R1-1776 模型的推理能力及其与标准 R1 模型的区别，并指出了其提供无审查回复的潜力。
   - 用户分享了使用该模型的经验，揭示了其在敏感话题上的表现和运行背景。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/natalrhyme">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://openrouter.ai/perplexity/r1-1776">R1 1776 - API, Providers, Stats</a>：注意：由于此模型不返回 &lt;think&gt; 标签，默认情况下思考过程将直接流式传输到 `content` 字段。R1 1776 是 DeepSeek-R1 的一个版本，经过后期训练以移除...</li><li><a href="https://huggingface.co/unsloth/r1-1776-GGUF">unsloth/r1-1776-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://www.reddit.com/media?url=https%3A%2F%2Fi.redd.it%2Fwyk3twpors8e1.jpeg">https://i.redd.it/wyk3twpors8e1.jpeg</a>：未找到描述</li><li><a href="https://chromewebstore.google.com/detail/complexity-perplexity-ai/ffppm">Chrome 网上应用店</a>：为您的浏览器添加新功能并个性化您的浏览体验。</li><li><a href="https://chromewebstore.google.com/detail/complexity-perplexity-ai/ffppmilmeaekegkpckebkeahjgmhggpj,">Chrome 网上应用店</a>：为您的浏览器添加新功能并个性化您的浏览体验。</li><li><a href="https://x.com/AravSrinivas/status/1891905511286768018">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：祝贺 @xai 在如此短的时间内构建出世界级的推理模型。这是美国对 DeepSeek 的超强回应。美国建设速度很快，xAI 团队和 Elon 正在设定...</li><li><a href="https://www.youtube.com/watch?v=8EyLJHmLNew">多伦多机场达美航空飞机翻转的新镜头</a>：周一，一架达美航空航班在多伦多机场降落时翻转，一名儿童是三名重伤者之一。至少有 18 人...</li><li><a href="https://news.microsoft.com/source/features/ai/microsofts-majorana-1-chip-carves-new-path-for-quantum-computing/">微软的 Majorana 1 芯片为量子计算开辟了新途径 - Source</a>：Majorana 1，首款由全新 Topological Core 架构驱动的量子芯片
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1341464202623320116)** (19 messages🔥): 

> `IRS 采购 Nvidia 超级计算机，ChatGPT 能耗，机器人技术，澳大利亚央行降息，神经网络` 


- **IRS 采购 Nvidia 超级计算机**：报告显示 **IRS** 正在采购一台 **Nvidia Supercomputer** 以增强其运营能力，提升数据分析效率。
   - 此举被视为在日益增长的数据挑战中优化税务处理技术的重大举措。
- **ChatGPT 能耗被高估**：一场讨论引发了对 **ChatGPT 能耗** 被严重高估的关注，各种评估需要对其方法论进行重新审视。
   - 专家认为*准确的能源指标*将有助于改善公众对 AI 可持续性的看法。
- **探索机器人历史**：人们对**早期的机器人创新**感到好奇，一篇[相关文章](https://www.perplexity.ai/search/what-were-the-first-robotic-ar-2c97SCKyRc6YwwpLlrpb_A)阐明了该领域的早期发展和突破。
   - 讨论强调了机器人技术多年来的**演进**。
- **澳大利亚央行降息**：近期新闻显示 **澳大利亚央行** 已经进行了大幅**降息**以刺激经济活动。
   - 该决定旨在应对国家的经济挑战，[此处有详细说明](https://www.perplexity.ai/page/australia-s-central-bank-cuts-9lwmh4dkSfWOo41dB7ga2Q)。
- **神经网络分析数据**：成员们讨论了**神经网络**的进展，并引用了一个[链接](https://www.perplexity.ai/search/neural-network-qqb0tqu2TGGF6ukUwue7HA)，探讨其在复杂数据分析中的应用。
   - 对话强调了**神经网络**在从 AI 到神经科学等各个领域中日益增长的重要性。



**提到的链接**：<a href="https://www.youtube.com/embed/Kxl4xFbfKwU">YouTube</a>：未找到描述

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1341541348884353115)** (7 messages): 

> `Sonar API 中的 R1-1776 热替换，API 中的图像使用，API Profile 设置，API 中的 Deep Research` 


- **R1-1776 热替换查询**：一位成员询问 **R1-1776** 是否在 **Sonar API** 上进行了热替换（hot swapped），特别是针对来自 **OpenRouter** 的请求。
   - 另一位成员提到他们也在考虑同样的问题。
- **API 图像使用协助**：一位成员请求关于如何在 API 中使用图像的指导。
   - 他们在收到反馈后表示感谢，但未提供更多细节。
- **API Profile 配置**：一位成员询问 API 是否使用账户设置中配置的 **Profile**。
   - 这一询问反映了用户对账户配置如何影响 API 行为的关注。
- **关于 Deep Research 功能的咨询**：一位成员询问 **deep research** 功能是否计划集成到 API 中。
   - 这表明了用户对 API 产品线中新功能的期待。


  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1341454612921585825)** (182 messages🔥🔥): 

> `Grok 3 性能，PaliGemma 模型更新，AI CUDA 工程师，客户支持自动化挑战，Claude Web App 增强`

- **Grok 3 展现出潜力但引发质疑**：普遍感觉虽然 **Grok 3** 表现不错，但对其真实能力的质疑依然存在，促使人们在 Twitter 之外寻找更可靠的信息源。
   - 成员们讨论了关于整体性能指标的担忧，以及 Grok 3 是否能经受住竞争。
- **PaliGemma 2 推出混合模型**：Google 推出了 **PaliGemma 2 mix**，它允许在多种任务上开箱即用，突破了预训练 checkpoint 的限制。
   - 讨论中将其与旧版本进行了对比，涉及命名和功能的清晰度及实用性。
- **AI CUDA Engineer 旨在提高效率**：Sakana AI 发布了 **AI CUDA Engineer**，这是一个旨在生成优化后的 CUDA kernels 的 AI 系统，相比典型实现可实现 **10-100 倍的加速**。
   - 该系统被预示为可能对机器学习运维中 AI 驱动的性能优化产生变革性影响。
- **重新审视客户支持自动化**：Klarna 在客户支持方面的历程凸显了用 AI 替代人工 Agent 的挑战和局限性，强调了对人工交互日益增长的需求。
   - 业内人士的轶事反映了在自动化简单任务与处理需要人工干预的复杂查询之间的平衡。
- **Claude Web App 随新功能不断演进**：Claude Web App 的更新表明其正在持续增强，新增了 **web search** 和 **Paprika mode** 等功能，重点在于提升 AI 的推理能力。
   - 这包括在准备公开发布时，增强其性能和用户交互的各种新版本。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://thinkingmachines.ai/">Thinking Machines Lab</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/nanotron/ultrascale-playbook">The Ultra-Scale Playbook - nanotron 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://www.commercialappeal.com/story/money/business/2025/02/13/xai-gas-turbines-at-memphis-supercomputer/78540969007/">埃隆·马斯克的 xAI 希望在孟菲斯继续使用燃气轮机：文件显示</a>: 未找到描述</li><li><a href="https://www.theverge.com/news/614883/humane-ai-hp-acquisition-pin-shutdown">Humane 正在关闭 AI Pin 并将其残余资产出售给 HP</a>: Humane 正在“逐步关停”其 AI Pin。</li><li><a href="https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/">利用 AI 共同科学家加速科学突破</a>: 未找到描述</li><li><a href="https://news.xbox.com/en-us/2025/02/19/muse-ai-xbox-empowering-creators-and-players/">来自 Xbox Wire 的推文：通过 Muse 为创作者和玩家赋能，这是一款用于游戏玩法的生成式 AI 模型</a>: 介绍 Muse，一种新型生成式 AI 模型，让你能够进行游戏和创作。从开发者的构思到未来支持游戏保存，Muse 具有开启新可能性的潜力。</li><li><a href="https://x.com/james_tackett1/status/1891898442206638237">来自 Jamie (@james_tackett1) 的推文</a>: @natolambert 既然 xAI 演示中的图表显示 Grok 3 以相当大的优势领先于所有其他 LLM，你为什么认为其他 LLM 处于领先地位？例如：</li><li><a href="https://x.com/dchaplot/status/1891920016339042463">来自 Devendra Chaplot (@dchaplot) 的推文</a>: 职业更新：非常幸运且兴奋能成为 Thinking Machines Lab 创始团队的一员！https://thinkingmachines.ai/ 加入我们：https://6wajk07p.paperform.co/</li><li><a href="https://x.com/M1Astra/status/1892091124589920532">来自 M1 (@M1Astra) 的推文</a>: 突发：根据 Claude iOS 应用最新更新中的发现，Anthropic 即将发布 Thinking Model/Mode（思考模型/模式）和 Web Search（网页搜索）功能。出现了诸如“Steps”和“Think”之类的新符号...</li><li><a href="https://x.com/btibor91/status/1892290734650433980">来自 Tibor Blaho (@btibor91) 的推文</a>: Claude 网页版应用更新——看起来 Web Search 和 Paprika 模式（新的思考模型）仍在开发中，过去 24 小时内部署了多个新版本。这包括一个新的实验...</li><li><a href="https://x.com/mervenoyann/status/1892289763954069720">来自 merve (@mervenoyann) 的推文</a>: @skalskip92 @onuralpszr 这是混合迁移（mixed transfer），但这次模型接受开放式输入，而不是结构化的任务前缀 🥹</li><li><a href="https://x.com/TheXeophon/status/1891795532500111752">来自 Xeophon (@TheXeophon) 的推文</a>: 上次发生这种情况还是在 *查阅笔记* 两周前。引用 Gavin Baker (@GavinSBaker)：如果我没记错的话，这是这一年多来第一次有一个模型在每个类别中都排名第一。</li><li><a href="https://x.com/ashtom/status/1891925306430337110">来自 Thomas Dohmke (@ashtom) 的推文</a>: 我们新的代码补全模型今天开始公开预览。我们将其命名为 GPT-4o Copilot。基于 GPT-4o mini，在超过 1T tokens 的代码专用语料库上进行了中期训练（mid-training），并进行了强化学习...</li><li><a href="https://x.com/SakanaAILabs/status/1892385766510338559">来自 Sakana AI (@SakanaAILabs) 的推文</a>: 介绍 AI CUDA Engineer：一个自动生产高度优化 CUDA 内核的智能体（agentic）AI 系统。http://sakana.ai/ai-cuda-engineer/ AI CUDA Engineer 可以生产高度优化的...</li><li><a href="https://x.com/klarnaseb/status/1892262217568891179">来自 Sebastian Siemiatkowski (@klarnaseb) 的推文</a>: @GergelyOrosz 我们没有改变航向。我们正在进一步开发它。如今我们的 AI 聊天机器人处理的咨询比你测试时更复杂、质量更高。但与此同时...</li><li><a href="https://fxtwitter.com/testingcatalog/status/1892133844184088576">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>: 突发 🚨：最新的 Claude iOS 版应用正准备推出 Web Search 和推理功能 👀 Web Search 测试版开关目前在 UI 中处于隐藏状态，但尚未生效，因为它可能需要新的模型...</li><li><a href="https://x.com/_xjdr/status/1891911178147987513">来自 xjdr (@_xjdr) 的推文</a>: TL;DR Grok 3 表现不错，通过了前沿级质量的直观感受测试（vibe check），但在我处理的大多数事情上，它并不比 R1 或 o1-pro 更好。总的来说比我预期的要好得多，我把它归入 Gemini...</li><li><a href="https://x.com/GavinSBaker/status/1891723733976465420">来自 Gavin Baker (@GavinSBaker) 的推文</a>: 如果我没记错的话，这是这一年多来第一次有一个模型在每个类别中都排名第一。</li><li><a href="https://x.com/owl_posting/status/1892317797172015210">来自 owl (@owl_posting) 的推文</a>: 笑死，Greg Brockman 在 Evo 论文中的所属单位是“独立研究员”...</li>

rcher&#39;</li><li><a href="https://tenor.com/view/surprised-pikachu-pokemon-shock-surprised-pikachu-gif-15357817">惊讶的皮卡丘 GIF - 惊讶的皮卡丘精灵宝可梦 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://developers.googleblog.com/en/introducing-paligemma-2-mix/">介绍 PaliGemma 2 mix：一个用于多任务的视觉语言模型</a>：未找到描述</li><li><a href="https://huggingface.co/google/paligemma2-3b-mix-448#paligemma-2-results-by-model-resolution-and-size>">google/paligemma2-3b-mix-448 · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/FlagAI-Open/OpenSeek">GitHub - FlagAI-Open/OpenSeek: OpenSeek 旨在联合全球开源社区，推动算法、数据和系统的协同创新，开发超越 DeepSeek 的下一代模型。</a>：OpenSeek 旨在联合全球开源社区，推动算法、数据和系统的协同创新，开发超越 DeepSeek 的下一代模型。 - FlagAI-Open/Open...</li><li><a href="https://x.com/GergelyOrosz/status/1892196257608687842">来自 Gergely Orosz (@GergelyOrosz) 的推文</a>：Klarna 曾是一家全力投入用 AI 机器人取代客户支持并吹嘘节省成本的公司。现在他们正在逆转这一进程。很容易看到更多公司盲目地替换...</li><li><a href="https://x.com/altryne/status/1884778839009796411">来自 Alex Volkov (Thursd/AI) 🔜 AIENG summit NY (@altryne) 的推文</a>：扎克伯格在财报电话会议上的亮点：- Llama 4 & Llama 4 mini（已完成预训练）- 确认推理版 Llama！- Llama 4 将是原生多模态的——它是一个全能模型（omni-model）——并且它将拥有...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1341602327458484294)** (6 条消息): 

> `tulu3 70B 模型，RLVR 阶段训练，GRPO 内存优化，论文更新` 


- **tulu3 70B 的 RLVR 训练时长**：在 8×8 H100 设置下，**tulu3 70B 模型**的 RLVR 阶段训练预计如果内存足够，大约需要**一周**时间，不过这会根据 prompt 和设置而有所不同。
   - 一位成员提到这应该需要 **O(天)** 级别的时间，支持了这一估算。
- **利用 GRPO 进行内存管理**：使用 **GRPO** 将增强内存能力，这可能会对训练过程产生积极影响。
   - 这表明技术设置可以进一步优化以获得更好的性能。
- **论文更新提醒**：一位成员承认他们忘记查看关于训练过程的更新论文。
   - 团队指出他们已经用关于训练阶段的相关信息更新了论文。


  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1341459754677043312)** (101 条消息🔥🔥): 

> `OpenAI 的 Deep Research 工具，AI 生成音乐的流行，用于生物学的 Evo 2 基础模型，关于 AI 模型许可证的讨论，即将举行的 UCSC 演讲活动` 


- **OpenAI 的 Deep Research 工具揭示了局限性**：一位用户分享了关于 OpenAI Deep Research 工具的见解，强调它*声称分析了完整的数据集*，但在数千篇论文中仅引用了大约 5-6 篇。
   - 他们指出，尽管有这些局限性，分享经验对于理解该工具的优缺点仍然很有价值。
- **AI 生成的音乐获得了巨大的参与度**：一位成员指出，一段以 AI 生成的 lofi 音乐为特色的视频获得了 **210 万**次观看，引发了关于 AI 艺术日益流行和潜在主流采用的讨论。
   - *Lofi girl 很快就要失业了*成了一句幽默的评论，反映了当用户意识到自己正在听 AI 生成的音乐时的竞争态势。
- **Evo 2 模型在生物学领域取得突破**：Michael Poli 宣布了 Evo 2，这是一个拥有 **400 亿参数**的新基础模型，专为生物学应用设计，旨在显著推进对基因组学的理解。
   - Evo 2 旨在展示测试时缩放定律（test-time scaling laws），并改进乳腺癌变体分类，同时推动 AI 领域真正的开源努力。
- **社区许可证和模型微调辩论**：一场关于在微调输出时 Llama 3.1 和 Llama 3.3 模型社区许可证兼容性的对话展开，得出的结论是必须保留这两个许可证。
   - 这次讨论突显了在 AI 社区中使用多个模型版本的复杂性以及涉及的法律问题。
- **即将举行的 UCSC 演讲活动**：一位成员宣布了即将举行的 UCSC 演讲活动，重点是 NLP，邀请南湾地区的任何人加入并与 AI 领域的关键人物会面。
   - 他们鼓励爱好者参与，并暗示会为与会者准备一个 **GPT-4 两周年纪念蛋糕**。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/MichaelPoli6/status/1892243237940904112">Michael Poli (@MichaelPoli6) 的推文</a>：[2/7] 我对 Evo 2 感到兴奋有很多原因。它无疑是我们在理解和生成基因组能力方面的一次飞跃。我也希望它能帮助推动 AI 领域真正开源的进展，...</li><li><a href="https://en.wikipedia.org/wiki/Thinking_Machines_Corporation">Thinking Machines Corporation - 维基百科</a>：未找到描述</li><li><a href="https://x.com/tianchezhao/status/1890788490016616780">Tiancheng Zhao (Tony) (@tianchezhao) 的推文</a>：我们在 RefCOCO（一个视觉定位任务）上训练了 Qwen 2.5 VL 3B，并在 RefCOCO Val 和 RefGTA（一个 OOD 任务）上进行了评估。结果：VLM-R1 具有泛化能力，而 SFT 基准线出现了过拟合，这与观察结果一致...</li><li><a href="https://x.com/littmath/status/1891868756340547809">Daniel Litt (@littmath) 的推文</a>：在使用 OpenAI 的 Deep Research 工具时有一个有趣的经历，我想分享一下，因为我认为它揭示了该工具的一些优点和缺点。</li><li><a href="https://x.com/xlr8harder/status/1892164213876961300">xlr8harder (@xlr8harder) 的推文</a>：如果我使用 Llama 3.3 70B Instruct 的输出来微调 Llama 3.1 8B 基座模型，适用哪种社区许可？在这种情况下：- Llama 3.1 坚持要求保留 Llama 3.1 Community License。- Llama 3.3...</li><li><a href="https://x.com/MichaelPoli6/status/1892244621490888718">Michael Poli (@MichaelPoli6) 的推文</a>：[7/7] 这个项目还有很多内容：Evo 2 首次展示了生物学中的 test-time scaling laws；Evo 2 在分类意义不明的 BRCA1 变体方面具有最先进的性能...</li><li><a href="https://x.com/nearcyan/status/1891926678810607858">near (@nearcyan) 的推文</a>：@thinkymachines 你的 x 账号是不是打错了，我不确定 thinky machines 是否有同样的感觉</li><li><a href="https://x.com/littmath/status/1891868775051342232">Daniel Litt (@littmath) 的推文</a>：唯一的问题是，这全是编造的。尽管声称查看了这 75 年间发表在《Annals》上的每一篇论文，但查看它浏览过的页面后发现，它实际上只查看了...</li><li><a href="https://x.com/MichaelPoli6/status/1892242976942035029">Michael Poli (@MichaelPoli6) 的推文</a>：[1/7] 介绍 Evo 2，一个新的生物学基础模型。🚀 Evo 2 是有史以来发布的最大规模、完全开源的 AI 模型：400 亿参数，超过 9 万亿 tokens，以及 100 万上下文...</li><li><a href="https://x.com/TheXeophon/status/1891586946675216803">Xeophon (@TheXeophon) 的推文</a>：一个视频 210 万次播放，每月 12.7 万 Spotify 听众收听 AI 生成的 lofi。起初很缓慢，然后突然爆发。引用 Xeophon (@TheXeophon)：没意识到我一直在听 AI 生成的音乐，直到...</li><li><a href="https://x.com/littmath/status/1891868790314434809">Daniel Litt (@littmath) 的推文</a>：换句话说，它再次声称生成了完整的数据集，但实际上只生成了约 7 行，其余约 3000 行都是占位符。</li><li><a href="https://youtu.be/Nc3vIuPyQQ0?si=KKt7VD5I521H95-W">AI CEO Alexandr Wang | This Past Weekend w/ Theo Von #563</a>：Alexandr Wang 是 Scale AI 的创始人兼 CEO，该平台为 AI 项目提供数据训练。2021 年，他被评为最年轻的白手起家亿万富翁...</li><li><a href="https://www.youtube.com/watch?v=YXTYbr3hiFU">一场出人意料的强化学习复兴</a>：我们所处的语言模型研究时代，普遍完全相信推理和新的强化学习 (RL) 训练...</li><li><a href="https://youtu.be/4poqjZlM8Lo?s">AI 大佬们担心什么</a>：Google DeepMind 和 Anthropic 的创始人 Demis Hassabis 和 Dario Amodei 是全球最顶尖的人工智能领袖。我们的总编辑...</li><li><a href="https://youtu.be/4poqjZlM8Lo?si=E6Y9rdAOYFjUeBhq)">AI 大佬们担心什么</a>：Google DeepMind 和 Anthropic 的创始人 Demis Hassabis 和 Dario Amodei 是全球最顶尖的人工智能领袖。我们的总编辑...</li><li><a href="https://x.com/zeffmax/status/1891970746596909091?s=46">Max Zeff (@ZeffMax) 的推文</a>：不确定为什么 Dario Amodei 和 Demis Hassabis 在这次采访中坐在一张非常小的沙发上——而《经济学人》的总编辑独自坐在一张非常大的沙发上——但我很喜欢。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1341612645723275287)** (3 messages): 

> `OpenAI's AI models, Useless Machine AI Demo, Open Research in AI` 


- **OpenAI 的模型是最棒的**：一位成员声称 **OpenAI 拥有最 based 的模型**，表明了对其优越性的强烈信念。
   - 分享了一张随附的图片，似乎在阐明这一点，尽管图片的具体内容未作详细说明。
- **请求 AI Agent 无用机器演示**：一位成员请求演示一个结合了 **AI Agent** 的**无用机器 (useless machine)**，表现出对其功能的关注。
   - 频道内随后没有关于此类演示的回复或确认。
- **关于 AI 研究动态的评论**：分享了一个包含对 AI 竞争格局评论的链接，特别提到了领先国家采取的**开放研究路径**。
   - 文中指出，**中国**和 **Google** 被重点提及，并暗示了他们在 AI 研究领域不同的动机和方法。



**Link mentioned**: <a href="https://x.com/doomslide/status/1892311556991697009">Tweet from doomslide (@doomslide)</a>: it&#39;s quite telling that the two countries at the forefront of AI both follow the path of open research. china, with its ulterior motive to sabotage funding of san francisco&#39;s finest, and googl...

  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1341519869878734952)** (5 messages): 

> `Manual Decompilation of Older Games, RL Training for Decompilation, LLM4Decompile GitHub Project` 


- **手动反编译工作兴起**：目前正努力通过**手动将汇编代码反编译为 C 语言**来完全重新编译旧游戏，这涉及大量猜测，以使其与已知事实一致并生成相同的汇编代码。
   - 这一过程在 [Decomp.me](https://decomp.me/scratch/hFS6m) 上有详细介绍，爱好者们尝试在此匹配原始代码设置。
- **寻求用于反编译的 RL 解决方案**：一位成员建议通过 **RL 训练 LLM** 来自动化反编译过程的可能性，并使用匹配百分比作为奖励。
   - 他们质疑这种方法是否仍是未开发的领域，以及是否有人尝试在强化学习背景下匹配汇编代码。
- **意识到 SFT 在反编译中的应用**：另一位成员指出，虽然使用大语言模型的想法很吸引人，但目前的实现（如 [LLM4Decompile](https://github.com/albertan017/LLM4Decompile)）似乎仅使用了**监督微调 (SFT)**。
   - 他们强调，据其所知，目前还没有人进行过专注于在反编译过程中匹配汇编代码的强化学习训练。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://decomp.me/scratch/hFS6m>)">decomp.me</a>: no description found</li><li><a href="https://github.com/albertan017/LLM4Decompile">GitHub - albertan017/LLM4Decompile: Reverse Engineering: Decompiling Binary Code with Large Language Models</a>: Reverse Engineering: Decompiling Binary Code with Large Language Models - albertan017/LLM4Decompile
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[cv](https://discord.com/channels/1179127597926469703/1208183243447468093/)** (1 messages): 

the_real_jrb: https://arxiv.org/abs/2502.13923
  

---

### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1341477416555642980)** (4 messages): 

> `理论论文, Open Source AI 担忧, Daniel Jeffries 的观点, 对作者的复杂感受` 


- **对理论论文的怀疑**：一位成员对理论论文表达了*怀疑*，表明了对纯理论研究的普遍怀疑态度。
   - 这种情绪得到了一些轻松的回应，暗示了理论家之间正在进行的辩论。
- **捍卫 Open Source AI**：讨论围绕 [Daniel Jeffries 的文章](https://danieljeffries.substack.com/p/defending-open-source-ai-against) 展开，他在文中认为 **Open Source** 软件在当今会面临来自社会各界的严厉批评。
   - 他暗示，在当前环境下，由于对潜在危险的恐惧，会导致对 **Open Source** 软件的**严格控制**。
- **对监管反应的担忧**：Jeffries 警告说，立法者会将 **Open Source** 标记为威胁，并要求对其开发进行广泛控制。
   - 这一观点突显了对当前**社会恐惧**可能扼杀技术创新的担忧。
- **对 Jeffries 作品的复杂感受**：另一位成员分享了对 Jeffries 作品的复杂感受，并表示希望在喝咖啡时讨论他的观点。
   - 这引发了关于 AI 社区内对争议性想法接受程度的思考。



**提及的链接**：<a href="https://danieljeffries.substack.com/p/defending-open-source-ai-against">Defending Open Source AI Against the Monopolist, the Jingoist, the Doomer and the Idiot</a>：如果 Linux 在今天才刚刚起步，它会被彻底击垮，我们都会因此变得贫穷得多。我们不能让这种事发生在 AI 身上。

  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1341505552924278855)** (13 messages🔥): 

> `Grok 3 Mini 公告, Hacker News 评论, XAI 讨论压力, Reasoning Models 基准测试` 


- **Grok 3 Mini 仅发布公告，尚未正式推出**：一位成员指出 **Grok 3 Mini** 只是发布了公告，尚未正式发布，并引用了 [这条推文](https://x.com/keirp1/status/1891955483251998984)。
   - 另一位成员确认，据他们所知，目前没有任何 Reasoning Models 正式发布。
- **Hacker News 评论不值得阅读**：一位成员评论说，在身体不适时阅读 **Hacker News** 的评论是一种挑战，称这是一项相当令人不快的任务。
   - 他们表示通常会完全避免参与那些评论。
- **XAI 讨论显得令人生畏**：一位成员表达了对参与 **XAI** 社区的抵触情绪，觉得回复讨论的想法压力很大。
   - 这种情绪表明了对参与此类对话所带来的压力的普遍担忧。
- **对高级功能的好奇**：一位成员开玩笑说存在一个付费的“deep thonk”（深度思考）按钮，并质疑其效用。
   - 这一评论为围绕 AI 讨论复杂性的对话增添了轻松的氛围。



**提及的链接**：<a href="https://x.com/keirp1/status/1891955483251998984">来自 Keiran Paster (@keirp1) 的推文</a>：@natolambert @srush_nlp @TheShmanuel 我认为 mini reasoning model 表现优于 R1 是反驳这一说法的有力证据。

  

---


### **Interconnects (Nathan Lambert) ▷ #[retort-podcast](https://discord.com/channels/1179127597926469703/1238601424452059197/1341542219521200350)** (2 messages): 

> `自行车架, 图片分享, 社区情绪` 


- **Torters 庆祝图片分享**：成员们对分享的一张图片感到高兴，在社区中引发了对自行车相关内容的兴奋。
   - 分享的图片营造了参与互动的氛围，展示了社区对自行车配件的共同兴趣。
- **自行车架引起关注**：一位成员注意到他们拥有与分享图片中相同的自行车架，强调了与该主题的个人联系。
   - 这一评论引发了成员之间关于自行车架功能和实用性的简短讨论。


  

---


### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/)** (1 messages): 

gfabulous: 唉，看来我们现在都要用 Grok 了

### **Interconnects (Nathan Lambert) ▷ #[expensive-queries](https://discord.com/channels/1179127597926469703/1338919429752361103/1341454244221550713)** (13 messages🔥): 

> `Prompt Builder 混淆, ODR 的对比查询, PDR 的局限性, Cursor Agent vs ODR, Vibecoding 工作流` 


- **Prompt Builder 混淆揭晓**：一位用户澄清，消息文本中隐藏的 prompt 实际上是 'prompt builder' prompt，*并非用于查询*。
   - 讨论反映了对共享消息中命名惯例的幽默与批评。
- **对比查询引发好奇**：一位用户表达了对在 **ODR** 和 **Perplexity** 之间运行对比查询的好奇，并讽刺地指出后者的性能较差。
   - *随后引发了一阵笑声*，因为小组承认了现有查询工具的局限性。
- **PDR 见解浅薄受到批评**：一位参与者批评 **PDR** 过于肤浅且像“列表文（listicle）”，强调其难以从结果中推断出新的见解。
   - 这导致大家达成共识，认为 PDR 仅比基础 LLM 结合搜索略好一点。
- **Cursor Agent 配合 ODR 取得成功**：一位用户分享了使用 **Cursor** 处理库中破坏性变更（breaking change）的经验，其中 **ODR** 的输出促进了问题的轻松修复。
   - 成功的解决证明了 Cursor Agent 利用上下文的有效性。
- **讨论 Vibecoding 工作流**：该小组开玩笑地将他们的工具组合称为“终极 vibecoding 工作流”，暗示这是一个有趣且高效的编码过程。
   - 一位用户幽默地提到了 Karpathy，巩固了围绕高级编码技术的轻松基调。


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1341888997420433419)** (2 messages): 

> `Reasoning Tokens 默认行为, 关于 Token 限制的反馈, 用户偏好投票` 


- **关于默认 Reasoning Tokens 的辩论**：反馈表明，当 **max_tokens** 较低时，用户更倾向于接收内容，这促使官方对 **include_reasoning** 设置进行审查。
   - *目前，include_reasoning 默认为 false*，这会导致响应可能为空或 null。
- **提议修改 Include Reasoning 设置**：正在考虑两项重大更改：将 **include_reasoning** 默认设置为 true，并标准化响应格式以确保内容始终是非空字符串。
   - *目的是避免空响应*，并允许包含 reasoning tokens，除非明确禁用。
- **扩展用户输入的投票选项**：针对 **include_reasoning** 行为，现已提供包含四个选项的扩展投票，以收集社区反馈。
   - 选项范围从保持当前设置到将 reasoning tokens 设为默认，并提供了一个用户评论的补充选项。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1341467447588552857)** (242 messages🔥🔥): 

> `Grok 3 性能, OpenRouter API 使用, Chatbot 集成, Perplexity R1 1776, AI 模型对比` 


- **Grok 3 在推理方面表现不佳**：用户报告称，与 **Claude 3.5** 和 **O1** 相比，**Grok 3** 的推理能力令人失望，对其作为顶级 LLM 的声誉表示怀疑。
   - 一位用户对比了性能后发现其表现低于预期，认为该模型可能名不副实。
- **使用 OpenRouter API**：新用户对如何访问 **OpenRouter** 以及通过 API 使用 **O3 mini** 表示困惑，特别是关于使用限制。
   - 用户强调需要更好的集成选项以及对 API key 使用的清晰说明，并指出了模型访问权限的逐步开放过程。
- **在网站上集成 Chatbot**：有用户询问如何使用 **OpenRouter API** 在 HTML 网站上集成 AI chatbot，这反映出目前缺乏现成的实现资源。
   - 其他人建议他们需要自行开发解决方案或聘请开发人员协助。
- **Perplexity R1 1776 推出**：**Perplexity** 推出了 **R1 1776** 模型，为用户提供了 **DeepSeek R1** 模型的无审查版本。
   - 这一新模型承诺提供更好的响应能力，并因其极具竞争力的 token 使用定价而受到关注。
- **AI 模型性能洞察**：讨论了 **Claude** 和 **O3 mini** 等 AI 模型之间的对比，用户分享了关于推理能力褒贬不一的体验。
   - 一些用户指出，尽管宣称具有优越性，但与成熟模型相比，新模型在特定任务中仍然表现吃力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://ai.google.dev/gemini-api/terms">未找到标题</a>: 未找到描述</li><li><a href="https://openrouter.ai/google/gemini-">OpenRouter</a>: LLM 的统一接口。为您的 prompt 寻找最佳模型和价格</li><li><a href="https://openrouter.ai/perplexity/r1-1776">R1 1776 - API, Providers, Stats</a>: 注意：由于此模型不返回 &lt;think&gt; 标签，推理过程默认将直接流式传输到 `content` 字段。R1 1776 是 DeepSeek-R1 的一个版本，经过后期训练以移除审查...</li><li><a href="https://openrouter.ai/google/gemini-2.0-pro-exp-02-05:free)">Gemini Pro 2.0 Experimental - API, Providers, Stats</a>: Gemini 2.0 Pro Experimental 是 Gemini 2 的最前沿版本。通过 API 运行 Gemini Pro 2.0 Experimental</li><li><a href="https://huggingface.co/perplexity-ai/r1-1776/discussions">perplexity-ai/r1-1776 · Discussions</a>: 未找到描述</li><li><a href="https://openrouter.ai/models">Models | OpenRouter</a>: 在 OpenRouter 上浏览模型</li><li><a href="https://hst.sh/gewiwoqete.md">hastebin</a>: 未找到描述</li><li><a href="https://x.com/perplexity_ai/status/1891916644248846789?t=7_5m7rcR2w7GFITF2I2QSA&s=19">Perplexity (@perplexity_ai) 的推文</a>: 在我们的 HuggingFace 仓库下载模型权重，或考虑通过我们的 Sonar API 使用该模型。HuggingFace 仓库: https://huggingface.co/perplexity-ai/r1-1776</li><li><a href="https://x.com/perplexity_ai/status/1892329089903841467?t=6lD3qXX2sOcKytYFI8L1kA&s=19">Perplexity (@perplexity_ai) 的推文</a>: R1 1776 现已通过 Perplexity 的 Sonar API 提供。引用 Perplexity (@perplexity_ai)：今天我们开源了 R1 1776——这是 DeepSeek R1 模型的一个版本，经过后期训练以提供...</li><li><a href="https://youtu.be/fwHkdivFCuc">与 OpenRouter CEO Alex Atallah 的未过滤对话</a>: 收听与 OpenRouter CEO Alex Atallah 以及 Nolan Fortman、Logan Kilpatrick 的对话。对话内容涵盖：OpenRouter 的起步...
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1341477017530404948)** (200 条消息🔥🔥): 

> `AI Image Generation Tools, Stable Diffusion vs. Flux, ControlNet Functionality, UI Options for Image Generation, Installation Guides for AI Tools` 


- **探索 AI 图像生成工具**：一位用户表示有兴趣学习 Midjourney 以外的工具，并发现 Stable Diffusion (SD) 比 Flux 更成熟，但寻求关于该学习哪一个的建议。
   - 社区建议查看示例图像并根据个人喜好进行选择，同时也要考虑硬件规格。
- **ControlNet 增强图像生成**：ControlNet 允许用户通过将姿势转换为骨架图来作为参考，从而提高生成图像的准确性。
   - 用户指出 SD 和 Flux 在配合 ControlNet 生成特定姿势方面都表现良好。
- **AI 工具的硬件注意事项**：一位拥有 RTX 3080 GPU 的用户被告知他们可以有效地运行 Stable Diffusion 3.5 和 Flux。
   - 社区强调，在 SD 3.5 和 XL 之间的选择取决于用户需求，但较新的模型提供了更好的功能。
- **为 AI 工具选择用户界面**：ComfyUI 以其复杂性和类似思维导图的界面而著称，这使得它对普通用户不太友好。
   - 推荐使用 SwarmUI 等替代方案和其他一站式解决方案，以便更轻松地使用 AI 工具。
- **AI 生成工具的安装资源**：用户被引导至 SwarmUI、CS1o 教程和 Lykos Stability Matrix 的各种安装指南，以设置 AI 工具。
   - 这些资源提供了详细的步骤，并帮助用户完成不同界面的安装过程。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/DOFOFFICIAL/animeGender-dvgg-0.8">DOFOFFICIAL/animeGender-dvgg-0.8 · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/jiseok-kim-jiseok-big-ocean-bigocean-kpop-gif-16919206117458777151">Jiseok Kim Jiseok GIF - Jiseok Kim jiseok Big ocean - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=n233GPgOHJg">Stable Diffusion Models Explained Once and for All (1.5, 2, XL, Cascade, 3)</a>: 在这段视频中，我解释了 Stable Diffusion 的 5 个不同模型系列。我是否有遗漏或错误？请告诉我。章节：00:00 简介...</li><li><a href="https://github.com/LykosAI/StabilityMatrix/">GitHub - LykosAI/StabilityMatrix: Multi-Platform Package Manager for Stable Diffusion</a>: 适用于 Stable Diffusion 的多平台包管理器 - LykosAI/StabilityMatrix</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases">Releases · AUTOMATIC1111/stable-diffusion-webui</a>: Stable Diffusion web UI。通过在 GitHub 上创建账户，为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。</li><li><a href="https://github.com/mcmonkeyprojects/SwarmUI?tab=readme-ov-file#installing-on-windows">GitHub - mcmonkeyprojects/SwarmUI: SwarmUI (formerly StableSwarmUI), A Modular Stable Diffusion Web-User-Interface, with an emphasis on making powertools easily accessible, high performance, and extensibility.</a>: SwarmUI（原名 StableSwarmUI），一个模块化的 Stable Diffusion Web 用户界面，强调让强大工具易于访问、高性能和可扩展性。 - mcmonkeyprojects/Swa...</li><li><a href="https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides">Webui Installation Guides</a>: Stable Diffusion 知识库（设置、基础、指南等） - CS1o/Stable-Diffusion-Info
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[announcements](https://discord.com/channels/1216353675241590815/1216353675241590818/1341892739289321583)** (1 条消息): 

> `Torchtune roadmap, PyTorch developments` 


- **Torchtune 路线图发布**：今年上半年的 **Torchtune 路线图** 已正式发布在 [PyTorch dev-discuss](https://drive.google.com/file/d/1mKENBMrdMzMQQG1kn43Il64qWPB9uP21/view) 上。
   - 该文档列出了关键目标和时间表，提供了关于 Torchtune 未来发展的见解。
- **PyTorch 范围内令人兴奋的新工作**：除了 Torchtune，整个 PyTorch 组织在今年上半年还有**大量酷炫的工作**正在进行。点击此处查看完整的 [PyTorch 路线图合集](https://dev-discuss.pytorch.org/t/meta-pytorch-team-2025-h1-roadmaps/2794)。
   - 这些路线图涵盖了正在进行的各种项目和优先级，展示了围绕 PyTorch 充满活力的生态系统。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://drive.google.com/file/d/1mKENBMrdMzMQQG1kn43Il64qWPB9uP21/view">[PUBLIC] Torchtune - H1 2025 Roadmap.pdf</a>: 未找到描述</li><li><a href="https://dev-discuss.pytorch.org/t/meta-pytorch-team-2025-h1-roadmaps/2794">Meta PyTorch Team 2025 H1 Roadmaps</a>: PyTorch 社区，Meta 团队很高兴能公开我们的 2025 H1 路线图。我们以半年为单位进行规划，并针对我们在 Meta 以及全球范围内为用户所做的工作进行全局优化...
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1341454925342707854)** (141 条消息 🔥🔥): 

> `Torchtune Roadmap, Packed vs Unpacked Tokenization, Fine-tuning Llama Models, Attention Mechanisms, Pruning Techniques` 


- **Torchtune 路线图发布**：Torchtune 团队在 PyTorch dev-discuss 上分享了他们的路线图，强调了即将推出的功能和优先级。
   - 他们的目标是在保持领先于新模型架构的同时，优先处理现有的核心工作。
- **Packing 对 VRAM 使用的影响**：用户观察到，在训练中使用长序列的 Packing 会显著增加 VRAM 需求。
   - 测试显示了不同的内存分配行为，促使进一步探索 Kernel 差异和设置。
- **微调 3B Llama 模型的挑战**：用户报告了 3B Llama 模型在微调后的推理过程中出现胡言乱语的问题，特别是与 8B 变体相比结果差异巨大。
   - 讨论转向了 Torchtune 中可能影响 3B 模型的模型导出和 Checkpointing 过程中的潜在 Bug。
- **探索奇特的注意力机制**：人们有兴趣将更先进的注意力机制（如稀疏和压缩注意力）集成到 Torchtune 中，以提高效率。
   - 团队对利用这些技术的想法持开放态度，同时主要专注于利用 PyTorch 核心功能。
- **剪枝技术的未来方向**：路线图讨论了为将剪枝技术（Pruning Techniques）整合到模型训练过程中奠定基础。
   - 目前已制定了一项策略，旨在开发支持宽度和深度剪枝并结合知识蒸馏（Knowledge Distillation）的 Recipes。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://pytorch.org/torchtune/main/tutorials/qlora_finetune.html#qlora-compile-label">使用 QLoRA 微调 Llama2 — torchtune 主文档</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2407.14679">通过剪枝和知识蒸馏实现紧凑型语言模型</a>：目前针对不同部署规模和大小的大语言模型 (LLMs) 是通过从头开始训练每个变体来生产的；这极其耗费计算资源。在本文中，我们研究了...</li><li><a href="https://pytorch.org/torchtune/stable/tune_cli.html">torchtune CLI — torchtune 0.5 文档</a>：未找到描述</li><li><a href="https://pytorch.org/torchtune/stable/tutorials/e2e_flow.html#generate-some-output">使用 torchtune 的端到端工作流 — torchtune 0.5 文档</a>：未找到描述</li><li><a href="https://x.com/intervitens/status/1891997689128120397">来自 intervitens (@intervitens) 的推文</a>：@samsja19 Torchtune 已经很接近了。没有 fa3，但通过 PyTorch SDPA 支持 flex attention 或 cuDNN。不完全兼容 HF checkpoints，可以使用相同的权重，但 config 和 tokenizer 需要手动支持...</li><li><a href="https://docs.unsloth.ai/basics/unsloth-benchmarks#llama-3.1-8b-max.-context-length">Unsloth 基准测试 | Unsloth 文档</a>：想知道 Unsloth 有多快吗？</li><li><a href="https://github.com/ianbarber/ttblt">GitHub - ianbarber/ttblt: Byte Latent Transformers 的简化实现，作为一个 TorchTune recipe。</a>：Byte Latent Transformers 的简化实现，作为一个 TorchTune recipe。 - ianbarber/ttblt</li><li><a href="https://github.com/pytorch/torchtune/blob/main/recipes/configs/generation.yaml">torchtune/recipes/configs/generation.yaml 在 main 分支 · pytorch/torchtune</a>：PyTorch 原生训练后库。通过在 GitHub 上创建账号来为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/issues/2392">使用 torch 2.6.0 时出现 `torch._inductor.exc.LoweringException: NoValidChoicesError` · Issue #2392 · pytorch/torchtune</a>：错误 [rank0]: raise NoValidChoicesError( [rank0]: torch._inductor.exc.LoweringException: NoValidChoicesError: 没有可供选择的选项，请考虑将 ATEN 添加到 max_autotune_gemm_backends 配置中...</li><li><a href="https://github.com/MaxHastings/Kolo/blob/main/scripts/merge_lora.py">Kolo/scripts/merge_lora.py 在 main 分支 · MaxHastings/Kolo</a>：使用现有最佳工具在本地进行数据生成、微调和测试 LLMs 的一站式商店。保持简单且多功能！- MaxHastings/Kolo</li><li><a href="https://github.com/pytorch/torchtune/blob/e6cba2532d51a53936c7646bd4cdaa6b2b57ed66/">GitHub - pytorch/torchtune 在 e6cba2532d51a53936c7646bd4cdaa6b2b57ed66</a>：PyTorch 原生训练后库。通过在 GitHub 上创建账号来为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3_2/3B_qlora_single_device.yaml">torchtune/recipes/configs/llama3_2/3B_qlora_single_device.yaml 在 main 分支 · pytorch/torchtune</a>：PyTorch 原生训练后库。通过在 GitHub 上创建账号来为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/blob/e6cba25">GitHub - pytorch/torchtune 在 e6cba2532d51a53936c7646bd4cdaa6b2b57ed66</a>：PyTorch 原生训练后库。通过在 GitHub 上创建账号来为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune#optimization-flags">GitHub - pytorch/torchtune: PyTorch 原生训练后库</a>：PyTorch 原生训练后库。通过在 GitHub 上创建账号来为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/blob/e6cba2532d51a53936c7646bd4cdaa6b2b57ed66/torchtune/modules/attention_utils.py#L35">torchtune/torchtune/modules/attention_utils.py 在 e6cba2532d51a53936c7646bd4cdaa6b2b57ed66 · pytorch/torchtune</a>：PyTorch 原生训练后库。通过在 GitHub 上创建账号来为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/pytorch/issues/137901">有计划为 Hopper GPUs 支持 Flash Attention 3 吗？ · Issue #137901 · pytorch/pytorch</a>：🚀 功能、动机和推介 Flash Attention 3 (https://github.com/Dao-AILab/flash-attention) 已经处于测试阶段一段时间了。我在带有 CUDA 12.3 的 H100 GPUs 上进行了测试，并尝试了一个类似的...</li><li><a href="https://github.com/pytorch/torchtune/blob/e6cba2532d51a53936c7646bd4cdaa6b2b57ed66/recipes/full_finetune_distributed.py#L378">torchtune/recipes/full_finetune_distributed.py 在 e6cba2532d51a53936c7646bd4cdaa6b2b57ed66 · pytorch/torchtune</a>：PyTorch 原生训练后库。通过在 GitHub 上创建账号来为 pytorch/torchtune 的开发做出贡献。
</li>
</ul>

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1341528701359231077)** (32 条消息🔥): 

> `基于步数的 Checkpointing、PPO 性能提升、与 Gymnasium 的集成、面向 LLM 的 Intercode 接口` 


- **基于步数的 Checkpointing 实现讨论**：团队正在讨论由于引入基于步数的 Checkpointing 而产生的新加载机制，并提出了从最新或特定 Checkpoint 恢复的选项。
   - 成员们正推动在现有的 PR 中添加功能，以确保更好地处理多实验场景，从而适应 Checkpointing 需求。
- **PPO Recipe 的性能提升**：最近的一个 PR 为 PPO Recipe 带来了性能改进，重点在于优化 KV caching 的配置。
   - 反馈强调了使用最新优化器带来的效率提升，并讨论了与现有系统的潜在集成兼容性。
- **关于与 Gymnasium 集成的担忧**：一位成员对将 Torchtune 与 Gymnasium 集成的可行性表示担忧，原因是两者在 LLM 任务的设计上存在本质差异。
   - 讨论中提到了如 Intercode 等专为编程任务定制的替代接口，认为它们可能更适合 LLM 训练。
- **Intercode 集成的未来**：关于 Intercode 的讨论表明，虽然它拥有适合编程和 SQL 任务的环境，但可能不符合稳定的 OSS 项目标准。
   - 成员们表示有兴趣分享关于 Intercode 与 LLM 接口连接效果的发现，并指出需要更多针对 Post-training 任务定制的环境。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://ysymyth.github.io)">未找到标题</a>：未找到描述</li><li><a href="https://pytorch.org/torchtune/main/deep_dives/checkpointer.html#resuming-from-checkpoint-full-finetuning">torchtune 中的 Checkpointing &mdash; torchtune 主文档</a>：未找到描述</li><li><a href="https://github.com/pytorch/torchtune/issues/2413">[RFC] Judge 框架与 Online DPO · Issue #2413 · pytorch/torchtune</a>：TRL 拥有多个不同 Judge 的概念，可用于各种在线 RLHF 类方法，请参阅 TRL Judges 文档。作为起点，我们可以先实现一个成对 Judge...</li><li><a href="https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3_1/8B_full_dpo.yaml#L64-L72">torchtune/recipes/configs/llama3_1/8B_full_dpo.yaml (main 分支) · pytorch/torchtune</a>：PyTorch 原生 Post-training 库。通过在 GitHub 上创建账号为 pytorch/torchtune 开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/pull/2412">由 SalmanMohammadi 提交的更新 PPO Recipe 中 KVCache 最大序列长度配置 · Pull Request #2412 · pytorch/torchtune</a>：Bug 在 #2064 中被提出。在设置 KV caches 时，我们使用 tokenizer.max_seq_len 来确定 KV cache 的形状。由于之前未在配置中设置，会导致报错。直接修复方法是...</li><li><a href="https://github.com/joecummings/torchtune/pull/2">feat: 为 checkpointer 工具添加 get_latest_checkpoint · Pull Request #2 · joecummings/torchtune</a>：添加了 get_latest_checkpoint &amp;quot;&amp;quot;&amp;quot; 返回给定目录中的最新 Checkpoint。pattern 参数是一个正则表达式，用于匹配 epoch 编号...</li><li><a href="https://github.com/princeton-nlp/intercode">GitHub - princeton-nlp/intercode: [NeurIPS 2023 D&amp;B] InterCode 基准测试代码库 https://arxiv.org/abs/2306.14898</a>：[NeurIPS 2023 D&amp;B] InterCode 基准测试代码库 https://arxiv.org/abs/2306.14898 - princeton-nlp/intercode</li><li><a href="https://github.com/pytorch/torchtune/pull/2105">由 joecummings 提交的 [RFC] torchtune 中的基于步数的 Checkpointing · Pull Request #2105 · pytorch/torchtune</a>：在 torchtune 中启用基于步数的 Checkpointing。原始背景：#2070。我们目前在做什么？我们目前仅在 epoch 边界进行 Checkpoint。这意味着微调运行必须遍历整个...</li><li><a href="https://github.com/pytorch/torchtune/pull/24">由 joecummings 提交的为 Transformers 添加 KV Cache · Pull Request #24 · pytorch/torchtune</a>：摘要：为 Transformer 添加 Key-value caching。同时重构了部分 Attention 代码使其更简洁，并添加了 torch 编译测试。测试：pytest tests/llm/llama2/test_transformer.py
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1341848371287162951)** (2 条消息): 

> `Multi-step PPO, Tool Use Learning, Reward Shaping, StepTool Framework` 


- **multi-step PPO 论文探讨**：一位用户询问是否有关于 **multi-step PPO** 的论文，这涉及多次连续调用 LLM 并结合工具结果，且奖励是在多个步骤后确定的。
   - 建议研究与 **learning better tool use** 和 **reward shaping** 相关的论文，表明这可能是一个研究资源非常丰富的领域。
- **StepTool 框架介绍**：[这篇论文](https://arxiv.org/abs/2410.07745) 提出了 **StepTool**，这是一个新颖的强化学习框架，通过将工具学习视为 **dynamic decision-making** 任务，增强了 LLM 的多步工具使用能力。
   - 它引入了诸如 **Step-grained Reward Shaping** 等组件，在工具交互过程中根据其成功程度和对任务的贡献提供奖励，旨在以多步方式优化模型的策略。



**提到的链接**：<a href="https://arxiv.org/abs/2410.07745">StepTool: Enhancing Multi-Step Tool Usage in LLMs through Step-Grained Reinforcement Learning</a>：尽管拥有强大的文本生成能力，大语言模型 (LLMs) 仍需学习如何利用外部工具来解决复杂任务，这一过程被称为工具学习。现有方法...

  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1341458134157496390)** (145 条消息🔥🔥): 

> `Grok-3, Le Chat, AI Rendering, Dynamic NPCs, LLMs and Game Development` 


- **Grok-3 API 讨论**：成员们讨论了 **Grok-3** 缺乏 API 的问题，这限制了它的使用，并指出在测试之前需要独立的 benchmark。
   - *似乎 Grok-3 目前会对内容进行混淆*，导致关于其集成和潜在应用的评论不断。
- **Le Chat 给用户留下深刻印象**：用户分享了使用 **Le Chat** 的积极体验，强调了其与其他模型相比在速度、图像生成能力和质量方面的优势。
   - 法国居民拥有**低价订阅**的独特机会，这在社区中引发了兴趣和兴奋。
- **AI Rendering 在游戏领域的未来**：讨论围绕 **AI Rendering** 通过创建动态和交互式环境来彻底改变游戏的潜力展开。
   - 成员们对 AI 生成内容与传统游戏开发之间的平衡表示担忧，强调了有意义的参与感的必要性。
- **AI 增强的 NPC 交互**：有推测认为 AI 可以根据玩家兴趣生成独特的 **NPC** 和任务，从而提升游戏体验。
   - 一些人认为，与当前的游戏结构相比，这种方法可以带来更具吸引力和沉浸感的交互。
- **LLMs 在游戏开发中的见解**：最近的研究强调了 **LLMs** 与人类大脑之间的相似性，表明了 **LLMs** 处理视觉输入等多样化数据的新方式。
   - 这为游戏开发中更具创新性的应用铺平了道路，特别是随着交互变得日益复杂。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/M1Astra/status/1892091124589920532">M1 (@M1Astra) 的推文</a>: 重磅：根据 Claude iOS 应用最新更新中的发现，Anthropic 即将发布 Thinking Model/Mode 和 Web Search 功能。出现了如 "Steps" 和 "Think" 等新符号...</li><li><a href="https://huggingface.co/microsoft/wham">microsoft/wham · Hugging Face</a>: 未找到描述</li><li><a href="https://news.mit.edu/2025/large-language-models-reason-about-diverse-data-general-way-0219">像人类大脑一样，大语言模型以通用的方式对多样化数据进行推理</a>: MIT 研究人员发现，大语言模型处理不同语言、音频输入、图像等多样化数据的方式，与人类推理复杂问题的方式相似。就像人类一样，LLMs...</li><li><a href="https://chat.mistral.ai/chat">Le Chat - Mistral AI</a>: 与 Mistral AI 最先进的语言模型聊天。</li><li><a href="https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview/discussions/5#67b390336de7ff0b4c84b16d">NousResearch/DeepHermes-3-Llama-3-8B-Preview · 老实说，它很垃圾。</a>: 未找到描述</li><li><a href="https://steamcommunity.com/sharedfiles/filedetails/?id=3143225812&searchtext=">Steam Workshop::Reforged Eden 2 Beta</a>: 未找到描述</li><li><a href="https://fxtwitter.com/satyanadella/status/1892244164814725387">Satya Nadella (@satyanadella) 的推文</a>: 如果你觉得 AI 生成的文本、图像和视频很酷，那就想象一下像游戏一样的整个交互式环境吧！</li><li><a href="https://github.com/DamascusGit/err_err">GitHub - DamascusGit/err_err</a>: 通过在 GitHub 上创建账户来为 DamascusGit/err_err 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1itdy0k/no_system_instructions_for_deepseek_makes_jake/">Reddit - 深入了解任何事物</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1341485401260425319)** (2 messages): 

> `SWE-Lancer Benchmark, MoBA Project` 


- **SWE-Lancer 基准测试发布**：介绍 [SWE-Lancer](https://arxiv.org/abs/2502.12115)，这是一个包含超过 **1,400 个来自 Upwork 的自由职业软件工程任务**的基准测试，总价值达 **100 万美元**。
   - 任务范围从 **50 美元的 Bug 修复**到 **32,000 美元的功能实现**，但研究结果显示，frontier models 难以解决大部分任务。
- **MoBA：推进长上下文 LLMs**：MoonshotAI 推出的 [MoBA](https://github.com/MoonshotAI/MoBA) 提出了一种 **Mixture of Block Attention** 技术，旨在提升 **长上下文 LLMs** 的性能。
   - 该项目旨在解决有效处理长输入时的局限性，更多信息可在 GitHub 上查看。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2502.12115">SWE-Lancer: Can Frontier LLMs Earn $1 Million from Real-World Freelance Software Engineering?</a>：我们推出了 SWE-Lancer，这是一个包含 1,400 多个来自 Upwork 的自由职业软件工程任务的基准测试，实际支付总额达 100 万美元。SWE-Lancer 涵盖了独立的工程任务...</li><li><a href="https://github.com/MoonshotAI/MoBA">GitHub - MoonshotAI/MoBA: MoBA: Mixture of Block Attention for Long-Context LLMs</a>：MoBA：用于长上下文 LLMs 的 Mixture of Block Attention - MoonshotAI/MoBA
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1341485401260425319)** (2 messages): 

> `SWE-Lancer Benchmark, MoBA Model` 


- **介绍 SWE-Lancer 基准测试**：**SWE-Lancer** 基准测试包含 1,400 多个来自 Upwork 的自由职业软件工程任务，总价值达 **100 万美元**。该基准测试包括由资深工程师评分的独立任务，旨在评估和提高模型在真实软件工程场景中的性能，可通过 [SWE-Lancer Diamond](https://github.com/openai/SWELancer-Benchmark) 获取。
   - 尽管数据集广泛，但评估显示 **frontier models** 难以有效解决大多数任务。
- **用于长上下文 LLMs 的 MoBA**：[MoBA](https://github.com/MoonshotAI/MoBA) 引入了一种专为长上下文大语言模型量身定制的 **Mixture of Block Attention** 方法，旨在增强处理能力。该项目有望推进 Transformer 模型中长上下文处理效率的研究。
   - MoBA 的 GitHub 仓库为开发人员解决 LLM 应用中的长上下文挑战提供了见解和框架。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2502.12115">SWE-Lancer: Can Frontier LLMs Earn $1 Million from Real-World Freelance Software Engineering?</a>：我们推出了 SWE-Lancer，这是一个包含 1,400 多个来自 Upwork 的自由职业软件工程任务的基准测试，实际支付总额达 100 万美元。SWE-Lancer 涵盖了独立的工程任务...</li><li><a href="https://github.com/MoonshotAI/MoBA">GitHub - MoonshotAI/MoBA: MoBA: Mixture of Block Attention for Long-Context LLMs</a>：MoBA：用于长上下文 LLMs 的 Mixture of Block Attention - MoonshotAI/MoBA
</li>
</ul>

</div>
  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1341472342567813181)** (12 messages🔥): 

> `Using Notebook LM for Book Summaries, Feedback on Audio Discussion Features, Prompting for TTS in Podcasts, Notebook Access for Non-Google Users` 


- **Notebook LM 在书籍教学方面表现出色**：一位用户表示相信 Notebook LM 可以有效地教他们一本书，只需提供更精确的提示词（prompts）以避免跳过关键部分。
   - *Hope it helps.* 强调了引导式提示词的潜在好处。
- **用户对音频讨论功能的反馈**：另一位用户分享道，虽然他们很欣赏 Notebook LM 如何帮助他们组织作业回答，但由于声音之间的来回打断，音频讨论变得难以跟进。
   - 他们建议采用更结构化的音频处理方式，即一个声音在另一个声音插话之前完整地表达一个想法。
- **播客 TTS 提示词的挑战**：一位用户询问了播客功能的 TTS 提示词，指出他们尝试让主持人逐字阅读文本的努力没有成功。
   - *What prompt did you use, verbatim?* 说明了他们在寻找有效的解决方案。
- **Notebook LM 访问限制**：一位用户询问是否可以邀请没有 Google 账号的人访问 Notebook，类似于 Google Docs 的共享功能。
   - 这突显了关于 Notebook LM 内部访问和协作功能的持续疑问。

### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1341461940299632742)** (124 条消息🔥🔥): 

> `NotebookLM Plus 功能, 用户体验挑战, 语言设置, 共享限制, 与其他平台的集成` 


- **NotebookLM Plus 使用限制**：用户讨论了 NotebookLM Plus 的每日限制，指出 500 次聊天查询是跨所有笔记本的总和，而非单个笔记本的限制。
   - 所有笔记本的共享上限为 **50 个用户**，有建议提议通过创建多个账户来增加限制。
- **用户体验关注点**：研究人员对用户体验提出了担忧，特别是关于上传的 PDF 文件重命名以及由此可能导致的混乱。
   - 建议增加直接在源列表中显示原始 URL 的功能，以便快速访问参考资料。
- **用户的语言设置**：一名医学生询问如何将 NotebookLM 的响应语言从阿拉伯语更改为英语，发现切换过程具有挑战性。
   - 经澄清，可以通过修改 URL 参数，将 'ar' 替换为 'en' 来解决此问题。
- **与其他工具的集成与对比**：用户将 NotebookLM 与 ChatGPT 和 DeepSeek 等其他 AI 工具进行了对比，并对其除播客功能之外的独特功能提出了疑问。
   - 讨论内容包括未来可能推出的视频功能，以及直接溯源（direct sourcing）如何改进功能。
- **未来的可用性与访问**：关于非营利组织是否可以使用 NotebookLM Plus 的问题被提出，Google 代表对于通用访问权限给出了不一的信息。
   - 用户表达了希望 Google 员工能对 NotebookLM 当前和未来的功能做出明确说明的愿望。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://notebooklm.google.com/?hl=ar">تسجيل الدخول - حسابات Google</a>：未找到描述</li><li><a href="https://support.google.com/notebooklm/answer/15678219?visit_id=638755661154801012-3742783720&p=plus&rd=1">升级到 NotebookLM Plus - NotebookLM 帮助</a>：未找到描述</li><li><a href="https://drive.google.com/file/d/1vc0a0FzJSXkPhpuuwB7e1twQpB9u7USd/view?usp=sharing">Snipaste_2025-02-19_15-23-18.png</a>：未找到描述</li><li><a href="https://blog.google/feed/notebooklm-google-one">NotebookLM Plus 现已包含在 Google One AI Premium 订阅中。</a>：NotebookLM 是一款研究和思考伴侣，旨在帮助您充分利用信息。您可以上传资料、进行总结、提问并转化……</li><li><a href="https://www.webwizwork.com/author/admin/">Balázs Piller - WebWizWork</a>：未找到描述
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1341977154145222717)** (6 条消息): 

> `GPU 规格电子表格, 学习 GPU 架构, 计算机架构书籍, Snapdragon/Adreno GPU 计算` 


- **寻找权威的 GPU 规格电子表格**：一位成员表达了未能找到类似于 [Google Sheets 上发布的](https://docs.google.com/spreadsheets/d/e/2PACX-1vSdXHeEqyabPZTgqFPQ-JMf-nogOR-qaHSzZGELH7uNU_FixVDDQQuwmhZZbriNoqdJ6UsSHlyHX89F/pubhtml) **权威 GPU 规格电子表格** 的挫败感。他们希望能发现一个更令人满意的 GPU 规格查询来源。
- **学习 GPU 架构的资源**：一位刚进入 GPU 领域的新成员正在寻求了解 GPU 架构的 **学习资源**，特别是与其关注的 CUDA 软件抽象相关的资源。他们分享了一本相关的 [Springer 书籍](https://link.springer.com/book/10.1007/978-3-031-01759-9) 链接，并已开始阅读。
- **对计算机架构书籍的兴趣**：另一位成员加入讨论，表达了对 **优秀计算机架构书籍** 的好奇。他们正在寻求相关推荐，以辅助他们的 GPU 学习。
- **对 Snapdragon/Adreno GPU 频道的建议**：一位成员在购入一台运行 Windows on ARM 的 Snapdragon 笔记本电脑后，提议创建一个专注于 **Snapdragon/Adreno GPU 计算平台** 的频道。他们渴望探索使用 OpenCL/Vulkan 进行 GPU 计算，并询问是否有其他人对此感兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://link.springer.com/book/10.1007/978-3-031-01759-9">通用图形处理器架构 (General-Purpose Graphics Processor Architectures)</a>：未找到描述</li><li><a href="https://docs.google.com/spreadsheets/d/e/2PACX-1vSdXHeEqyabPZTgqFPQ-JMf-nogOR-qaHSzZGELH7uNU_FixVDDQQuwmhZZbriNoqdJ6UsSHlyHX89F/pubhtml#">GPU_Compare</a>：未找到描述
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1341747913226719295)** (2 messages): 

> `BLOCK_SIZE 建议、矩阵维度、GEMM 与 splitK 性能` 


- **大矩阵的 BLOCK_SIZE 建议**：一位成员建议针对 **[228075, 512]** 和 **[512, 512]** 的矩阵尺寸，在 **BLOCK_M**、**BLOCK_N** 和 **BLOCK_K** 上使用 **2^18**。
   - 另一位成员回应称，**BLOCK_SIZE** 应小于或等于矩阵维度，并建议使用如 **BLOCK_M=[64,128]** 和 **BLOCK_K=[32,64,128,256]** 等值以获得更好的性能。
- **理解 GEMM 与 splitK 性能**：用户询问对于他们的矩阵，**GEMM** 是否会比 **splitK** 更快，从而引发了关于性能对比的讨论。
   - 回应建议将重点放在调整 **BLOCK_SIZE** 上，而不是假设更大的尺寸会自动产生更好的结果。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1341458250826121357)** (14 messages🔥): 

> `cudaMemcpyAsync 与 cudaMemcpyDeviceToDevice、CUDA 快速安装程序问题、Raw-Dogged Tensor 提案、Visual Studio 与 CUDA 安装故障` 


- **cudaMemcpy 中细粒度控制的需求**：一位用户表达了对数据拷贝进行**细粒度控制**的需求，特别提到了在 **for** 循环中使用 **cudaMemcpyDeviceToDevice**。
   - 另一位成员建议使用 [`cub::DeviceCopy::Batched`](https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceCopy.html) 作为潜在的解决方案。
- **Windows 上 CUDA 快速安装程序的挑战**：多位用户报告了 **CUDA Express Installer** 在 **Nsight** 和 **Visual Studio** 等关键组件上卡住的问题。
   - 一位用户指出，多位拥有不同 NVIDIA GPU 的朋友也遇到了同样的安装问题。
- **在 CUDA 术语中引入 Raw-Dogged Tensor**：一位用户为下一版本的 **cute**/**cutlass** 提议了一个新术语 **'raw-dogged Tensor'**，旨在增强存储格式与线程布局（thread layouts）之间的兼容性。
   - 另一位成员分享了他们在 **int8 matmul** 方面的经验，强调了在不使用特定预处理技术时出现的共享内存 Bank 冲突（shared memory bank conflicts）问题。
- **Visual Studio 与 CUDA 安装挑战**：一位用户详细描述了他们在尝试安装 **CUDA 12.5 和 12.8** 时与 **Visual Studio** 的斗争，并引用了同行中一个常见的问题。
   - 用户对 **Linux** 相较于 **Windows** 的安装便捷性表示关注，反映了对操作系统更广泛的挫败感。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1341756702109863937)** (2 messages): 

> `GoLU 激活函数、编译性能、文件拆分、Torch 论坛的 MRE` 


- **分享 GoLU 激活函数**：一位成员分享了 [GoLU](https://github.com/automl/GoLU/tree/main/golu) 仓库的链接，该仓库介绍了一种新型的、自门控（self-gated）且逐元素（element-wise）的激活函数，在多种任务中表现良好。
   - GitHub 页面包含图片预览和强调其有效性的描述。
- **对编译缓慢的担忧**：另一位成员表达了尽管拥有与 GoLU 仓库中描述类似的配置，但编译时间仍然很慢的挫败感。
   - 他们表示计划尝试**文件拆分**，并可能创建一个最小可复现示例（MRE）提交到 **torch** 论坛。



**提到的链接**：<a href="https://github.com/automl/GoLU/tree/main/golu">GoLU/golu at main · automl/GoLU</a>：GoLU，一种新型的、自门控且逐元素的激活函数，在多种任务中表现良好 - automl/GoLU

  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/)** (1 messages): 

andreaskoepf: DS 目前正统治该领域：https://arxiv.org/abs/2502.11089
  

---

### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1341525603475525742)** (4 条消息): 

> `GoLU Activation Function, UltraScale Playbook Release, AI CUDA Engineer Optimization, CUDA Kernel Discovery` 


- **介绍 GoLU：一种新型自门控激活函数**：最近的一篇论文提出了 **Gompertz Linear Unit (GoLU)**，这是一种旨在减少潜空间方差并保持梯度流的新型激活函数，其表现优于 **GELU** 和 **Swish**。
   - GoLU 函数定义为 $\mathrm{GoLU}(x) = x \, \mathrm{Gompertz}(x)$，有助于深度学习模型的**训练动力学**。
- **Nanotron 发布 UltraScale Playbook**：[查看来自 Nanotron 的最新博客文章](https://huggingface.co/spaces/nanotron/ultrascale-playbook)，展示了关于扩展 AI 模型的突破性见解。
   - 该文章重点介绍了 **AI 模型优化**和部署的创新策略及方法论。
- **AI CUDA Engineer 自动化 CUDA Kernel 创建**：**AI CUDA Engineer** 框架可自动创建优化的 CUDA Kernel，在 **PyTorch** 中实现比传统机器学习算子快 **10-100 倍**的加速。
   - 该系统将 PyTorch 转换为 CUDA 的成功率超过 **90%**，与现有解决方案相比，显著提升了性能。
- **革新 CUDA Kernel 优化**：AI CUDA Engineer 的表现优于 **native torch** 和 **torch compile** 优化，成功率分别为 **75%** 和 **60%**。
   - 特定算子（如 **Instance Normalization**）展示了该系统在性能增强转换方面的潜力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://sakana.ai/ai-cuda-engineer/">无标题</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/nanotron/ultrascale-playbook">The Ultra-Scale Playbook - a Hugging Face Space by nanotron</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2502.03654v1">Gompertz Linear Units: Leveraging Asymmetry for Enhanced Learning Dynamics</a>：激活函数是深度学习架构的基础元素，因为它们显著影响训练动力学。ReLU 虽然被广泛使用，但容易出现神经元死亡问题，这已经……</li><li><a href="https://pub.sakana.ai/ai-cuda-engineer/">The AI CUDA Engineer 👷</a>：未找到描述
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1341756853742473266)** (10 条消息🔥): 

> `torchao tutorial issues, HuggingFace quantization problems, LLaMa 2 integration, Key naming conflicts in models, Fix for past_key_value bug` 


- **torchao 教程面临问题**：一名成员询问了关于使用 HuggingFace 模型进行量化的教程，这些教程需要数据校准，特别是参考了 **GPTQ**。
   - 他们注意到代码在 **LLaMa 2 7B** 上进行了测试，但对有关 `.pth` 文件的模型要求感到困惑。
- **HuggingFace GPU 编译问题**：一位用户报告在运行 HuggingFace 量化示例时遇到了 **torch.compile 错误**。
   - 他们链接了一个 [GitHub issue](https://github.com/pytorch/ao/issues/1705)，其中记录了错误上下文和版本信息。
- **并非 torchao 的 Bug 而是代码冲突**：会议澄清了该问题不是 **torchao** 的 Bug，并提到即使不进行量化也会发生。
   - 参与者指出在 HuggingFace 的实现中，**past_key_values** 和 **past_key_value** 被混用了。
- **针对 Key 冲突提出的修复方案**：一名成员分享了一个 [pull request](https://github.com/huggingface/transformers/pull/36289) 以解决 **Key 命名冲突**，强调这两个 Key 都需要被正确管理。
   - 该修复专门旨在确保 LLaMa 模型在设备放置期间正确处理 **past_key_value**，以避免混淆。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/ao/issues/1705">torch.compile error when running the HuggingFace torchao example · Issue #1705 · pytorch/ao</a>：当我运行来自 https://huggingface.co/docs/transformers/main/en/quantization/torchao 的代码片段时，我看到了一个 torch.compile 错误。torch 版本：2.6.0 torchao 版本：0.8.0 transformers 版本……</li><li><a href="https://github.com/huggingface/transformers/pull/36289">[bugfix] Update modeling_llama.py so it skips keys correctly by HDCharles · Pull Request #36289 · huggingface/transformers</a>：llama 模型交替使用 past_key_value 和 past_key_values，这导致了问题，因为在 _skip_keys_device_placement 中实际上只跳过了其中一个，而两者都需要……
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1341506159663910994)** (2 messages): 

> `ROCm Application Developer Certificate, AI Copyright & National Security` 


- **AMD 发布 ROCm 开发者认证**：AMD 推出了 [ROCm Application Developer Certificate](https://www.amd.com/en/products/software/rocm/application-developer-certificate.html)，旨在提升 GPU 计算技能并推广 ROCm 生态系统。
   - 对于希望专注于**开源 GPU 计算技术**的开发者来说，该认证是迈出的重要一步。
- **AI 版权辩论升温**：关于 AI 版权的讨论源于一篇博客文章，该文章声称包括 **DeepSeek** 在内的中国 LLM 是在大量非法存档材料上训练的，突显了**国家安全**问题。
   - 该博客探讨了“影子图书馆”的现状，指出了用于 AI 训练的材料相关的[非法性和潜在风险](https://annas-archive.org/blog/ai-copyright.html)，并对比了亟需彻底改革的立法漏洞。



**提及的链接**：<a href="https://annas-archive.org/blog/ai-copyright.html">版权改革对国家安全至关重要</a>：中国 LLM（包括 DeepSeek）是在我的非法书籍和论文档案（全球最大）上训练的。西方国家需要出于国家安全考虑彻底改革版权法。

  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

kpk1340: 有人在 NYC 吗？
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1341939234327494676)** (5 messages): 

> `Mi50 hardware matmul support, Mi50 specifications, Tensor operations on GPUs` 


- **关于 Mi50 Matmul 能力的困惑**：一场关于 **MI50** 及其缺乏**硬件 matmul 支持**的讨论展开了，目前尚不确定它是否能处理 tensor 操作。
   - 一位成员澄清说，**WMMA** (Wave Matrix Multiply Accumulate) 功能出现在后期的显卡中，但在 Mi50 上的应用仍不确定。
- **Mi50 提供强劲规格**：**Mi50** 支持从 **FP64 到 FP16** 的多种数据类型，通过翻倍的 FLOPS 提供令人印象深刻的**性能**。
   - 其规格包括 **26.5 TFLOPs** 的 FP16 性能和高达 **1024 GB/s** 的显存带宽，展示了其在计算任务中的能力。



**提及的链接**：<a href="https://www.8anet.com/Product/17823/AMD-100-506143-Radeon-Instinct-MI50-Accelerator-PCIe-4-0-x16-32GB-HBM2-4096-bit-3840-Stream-Processors-Passive-Cooling">
	8ANET - AMD 100-506143 Radeon Instinct™ MI50 Accelerator PCIe 4.0 x16 32GB HBM2 4096-bit 3840 Stream Processors Passive Cooling
</a>：未找到描述

  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1341903496987017267)** (1 messages): 

> `Convergence Test Bug, PR Merging Process` 


- **通过 Logit Scaling 修复收敛测试 Bug**：一位成员报告称，通过解决 **MiniModelConfig** 中缺失的 **logit scaling 参数**修复了**收敛测试**问题，从而修正了 logit 的量级。
   - 该修复预计将提升模型的整体性能。
- **需要 PR 合并协助**：该成员表达了推进其 **pull request** 合并的迫切愿望，并表示愿意提供任何所需的协助以加快流程。
   - 他们**非常乐意**协作，以尽快完成 PR 合并。


  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1341788560612917260)** (2 messages): 

> `Kokoro TTS 优化, 低比特训练技术, GPU 性能提升` 


- **Kokoro TTS 通过 Triton 和 CUDA 加速**：一项旨在提高现代 TTS 模型 **Kokoro** 推理速度的工作，强调了使用 **Triton + CUDA graph** 来减少 LSTM 在 bs=1 时的 kernel 启动开销。
   - *真不敢相信在这个时代我还得优化 LSTM* 😂，测量是在 **4070Ti SUPER** 上进行的。
- **YouTube 关于低比特训练的见解**：一段名为“Thien Tran - Low bit training”的 [YouTube 视频](https://m.youtube.com/watch?v=leCY8vCUS4g) 讨论了在现代 GPU 上使用 **低比特数据类型 (low-bit datatypes)** 来加速训练。
   - 演讲者 **Thien Tran** 是一位常驻新加坡的 Machine Learning Engineer，他深入探讨了提升性能的实用技术。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://m.youtube.com/watch?v=leCY8vCUS4g">Thien Tran - Low bit training</a>：在本次演讲中，我们将探讨在现代 GPU 上使用低比特数据类型来加速训练。Thien 是一位常驻新加坡的 Machine Learning Engineer。他...</li><li><a href="https://x.com/gaunernst/status/1892227983072518503">Thien Tran (@gaunernst) 的推文</a>：Kokoro 是最近的一款 TTS，听起来很棒且速度很快。但我好奇，我能让它更快吗？以下是一些针对 Kokoro 推理在 bs=1 时的性能优化。所有计时均在 4070Ti 上测量...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1341455648889638912)** (7 messages): 

> `AI CUDA Engineer, Kernel 优化, CUDA 的进化方法, 创新档案 (Innovation Archive), CUDA 开发中的 LLMs` 


- **AI CUDA Engineer 自动化 CUDA kernel 生成**：Sakana AI 推出了 [AI CUDA Engineer](http://sakana.ai/ai-cuda-engineer/)，它可以生成优化的 CUDA kernels，比常见的 PyTorch 操作实现 **10-100 倍的加速**。
   - 他们最近的论文讨论了该系统根据以往学习自主优化运行时性能并发现高效 CUDA 操作的能力。
- **使用进化技术进行 Kernel 优化**：该过程涉及将重要的 PyTorch 算子转换为 CUDA，随后使用驱动 LLM 的**进化方法 (evolutionary approach)** 来优化这些 kernels。
   - 关键创新包括优化下三角矩阵乘法和分类交叉熵 kernels，其性能超越了传统的 **torch.compile**。
- **Dr. Jim Fan 赞扬自主编码 Agent**：一位成员强调 Sakana AI 的自主编码 Agent 是通过增强 CUDA kernels 来加速 AI 的突破，并指出了该 Agent 在调试方面的高效性。
   - 这种被称为 *创新档案 (Innovation Archive)* 的方法，充当了在进化过程中发现的最佳实践的存储库。
- **关于使用 LLM 进行 CUDA 优化的反馈**：一些成员对选择 LLM 而不是现有的 functional FX graph 系统来使 PyTorch 代码函数化表示惊讶。
   - 这场对话反映了关于优化 CUDA kernel 性能的最佳工具和方法的持续争论。
- **数据集发布和开放挑战**：AI CUDA Engineer 发布了一个包含 **17,000 个经过验证的 CUDA kernels** 的数据集，突显了其 AI 驱动加速的全面方法。
   - 挑战依然存在，例如提高 kernel 的多样性以及利用 tensor cores 等新特性，这对于进一步的发展至关重要。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/sakanaailabs/status/1892385766510338559?s=46">Sakana AI (@SakanaAILabs) 的推文</a>：介绍 AI CUDA Engineer：一个自动化生产高度优化 CUDA kernels 的 Agentic AI 系统。http://sakana.ai/ai-cuda-engineer/ AI CUDA Engineer 可以生产高度优化的...</li><li><a href="https://x.com/drjimfan/status/1892404919480832259?s=46">Jim Fan (@DrJimFan) 的推文</a>：我最近见过的最酷的自主编码 Agent：使用 AI 编写更好的 CUDA kernels 来加速 AI。AutoML 回归了！利用计算资源能做的杠杆率最高的事情就是...
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1341783799054209086)** (1 messages): 

> `MLA Attention Support` 


- **关于为 MLA Attention 支持贡献力量的咨询**：一位成员注意到正在进行的添加 **MLA attention** 支持的工作，并询问了参与贡献的方法。
   - 此次咨询突显了社区对 **MLA attention** 实现的协作开发工作的兴趣。
- **MLA Attention 支持正引起关注**：对话强调了围绕 **MLA attention** 支持日益增长的好奇心，引发了关于潜在贡献的咨询。
   - 成员们正在寻找参与开发的途径，标志着一种协作的社区动态。


  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1341474790275547260)** (13 messages🔥): 

> `SO-ARM100 Assembly, 3D Printing Experience, Dataset Collection Challenges, Hybrid Speech Processing Application` 


- **SO-ARM100 组装经验**：关于 [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) 组装的讨论突显了褒贬不一的体验，一位参与者幽默地提到组装花了 **5 小时**，而不是广告中声称的 1 小时。
   - *“我并不打算打印一堆东西，”* 一位对拥有 3D 打印机持怀疑态度的用户表示，他选择直接寄送 CAD 文件。
- **数据集收集的挫败感**：一位用户提到自最初体验以来，数据集收集已有改进，并表示之前在 *收集 episode 期间机械臂发生故障* 时非常令人沮丧。
   - 他们还指出，为了上传到 Hugging Face，必须对数据集进行补丁处理，这增加了过程的复杂性。
- **优化模型性能策略**：有人建议，在少量收集的数据集上对 **pi-0** 等预训练模型进行微调，可能比从头开始训练获得更好的性能。
   - 该策略旨在简化新用户进行模型训练的初步实验。
- **混合语音处理应用概述**：一位用户分享了他们的混合语音处理应用项目，称其在 NVIDIA Jetson Nano 上处理语音分离，并使用基于云的 LLM 来过滤 Prompt。
   - 随附的报告详细说明了实施情况和结果，并邀请同行对该项目提供反馈。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/lerobot/visualize_dataset_v1.6?dataset=philbutler%2Feval_act_green_ball_red_cup-10sec_10ep-right_cam&episode=8">Visualize Dataset (v1.6 old dataset format) - a Hugging Face Space by lerobot</a>: no description found</li><li><a href="https://github.com/TheRobotStudio/SO-ARM100">GitHub - TheRobotStudio/SO-ARM100: Standard Open Arm 100</a>: Standard Open Arm 100. Contribute to TheRobotStudio/SO-ARM100 development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1341517770679914517)** (56 messages🔥🔥): 

> `DeepSeek CodeI/O paper, Spatial reasoning datasets, Decimal chain sum dataset, Reasoning-Gym server experiment, Open-source ecosystem updates`

- **DeepSeek 为推理任务引入 CodeI/O**：DeepSeek 发布了一篇关于 CodeI/O 的新论文，旨在通过将代码转换为输入-输出预测格式来增强推理能力。该方法基于自然语言任务进行训练以提高性能，这是从通常的窄领域技能增强的一次重大转变。
   - 一位成员建议增加更简单的版本以提高适应性，强调了对全面数据集的需求。
- **探索空间推理数据集**：对促进 3D 环境中空间推理的数据集的探索，引出了关于涉及移动列表和空间关系的建议。著名的“大理石问题”展示了复杂的推理挑战，并提倡创建模板。
   - StepGame benchmark 被提及为一个重要的评估工具，尽管之前的模板错误影响了结果，这表明需要进行改进。
- **decimal_chain_sum 数据集的进展**：讨论了 decimal_chain_sum 数据集，强调了使用 Decimal 类进行精确算术运算。成员们贡献了实现建议，重点是在没有容差的情况下进行精确评分评估。
   - 创建了一个 PR 以在 factory 中注册该数据集，解决了之前遗漏的集成问题。
- **Reasoning-Gym server 实验的进展**：Reasoning-Gym server 实验的初始版本已接近完成，有效地结合了 server 和 CLI 工具。调试工作定于次日早晨进行，以确保功能正常。
   - 成员们正积极致力于与 server 功能相关的集成工作，朝着全面部署的方向迈进。
- **获得 interconnect 博客的认可**：Reasoning-Gym 得到了 interconnect substack 博客的提及，展示了其在开源生态系统中日益增长的认可度。该博客强调了 AI 数据集的快速进展，特别是通过 DeepSeek 的贡献。
   - 成员们讨论了 AI 领域的持续发展，强调了来自中国实验室的新模型及其宽松许可证的影响。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2401.03991">Advancing Spatial Reasoning in Large Language Models: An In-Depth Evaluation and Enhancement Using the StepGame Benchmark</a>: 人工智能 (AI) 在各个领域取得了显著进展，像 ChatGPT 这样的大型语言模型因其类人的文本生成能力而受到了广泛关注...</li><li><a href="https://arxiv.org/abs/2502.07316">CodeI/O: Condensing Reasoning Patterns via Code Input-Output Prediction</a>: 推理是 Large Language Models 的一项基本能力。虽然之前的研究主要集中在增强数学或代码生成等狭窄技能上，但提高在许多其他领域的性能...</li><li><a href="https://www.interconnects.ai/p/artifacts-7">The latest open artifacts (#7): Alpaca era of reasoning models, China&#x27;s continued dominance, and tons of multimodal advancements</a>: Artifacts Log 7。对于 AI 研究人员和从业者来说，这将继续是一个有趣的春天。</li><li><a href="https://github.com/open-thought/reasoning-gym/issues/160">Add CODEI/O sampled subset dataset · Issue #160 · open-thought/reasoning-gym</a>: DeepSeek 发布了 CodeI/O: Condensing Reasoning Patterns via Code Input-Output Prediction。任务：“完全以自然语言根据代码和测试用例预测输入/输出” 除了 propr...</li><li><a href="https://github.com/Fangjun-Li/SpatialLM-StepGame">GitHub - Fangjun-Li/SpatialLM-StepGame: Codes and data for AAAI-24 paper &quot;Advancing Spatial Reasoning in Large Language Models: An In-depth Evaluation and Enhancement Using the StepGame Benchmark&quot;</a>: AAAI-24 论文 "Advancing Spatial Reasoning in Large Language Models: An In-depth Evaluation and Enhancement Using the StepGame Benchmark" 的代码和数据 - Fangjun-Li/SpatialLM-StepGame</li><li><a href="https://github.com/open-thought/reasoning-gym/issues/163">Redundant line · Issue #163 · open-thought/reasoning-gym</a>: # Validate digit ranges make sense if self.min_digits &gt; 1: assert 10 ** (self.min_digits - 1) &gt;= 1, &quot;min_digits would result in invalid number range&quot; 这些行位于 /reasoning_gym/arit...</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/164">decimal_chain_sum by vncntt · Pull Request #164 · open-thought/reasoning-gym</a>: 添加 Issue #157。应该很简单。从 chain_sum 复制了很多代码。我想它并不完美，因为理想情况下我们希望针对一些棘手的情况，例如 .9 和 .11</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/167">Register decimal chain sum by vncntt · Pull Request #167 · open-thought/reasoning-gym</a>: 未找到描述</li><li><a href="https://github.com/open-thought/reasoning-gym/blob/main/reasoning_gym/arithmetic/decimal_chain_sum.py">reasoning-gym/reasoning_gym/arithmetic/decimal_chain_sum.py at main · open-thought/reasoning-gym</a>: 程序化推理数据集。通过在 GitHub 上创建账号为 open-thought/reasoning-gym 的开发做出贡献。</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/158/">Decimal Arithmetic by Miserlou · Pull Request #158 · open-thought/reasoning-gym</a>: 简单的 1234.123 + 234.567 类型的数学问题。
</li>
</ul>

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1341455544426303568)** (14 条消息🔥): 

> `General Superintelligence 辩论, Emergent Capabilities, AI 与游戏学习, Anand 的介绍, DeepSeek R1 上传` 


- **辩论 General Superintelligence 的含义**：成员们讨论了 **general superintelligence** 一词，其中一人指出它没有任何实际意义，只是一个流行语（buzzword）。
   - 另一位成员建议，如果模型在多项任务上表现出色，它们将获得更深层次的理解，从而应对复杂的、无法量化的挑战。
- **Emergent Capabilities：流行语还是现实？**：关于 *emergent capabilities*（涌现能力）存在争论，成员们对其在 AI 讨论中的有效性和相关性各执一词。
   - 一位成员澄清了他们的观点，表示他们的理论并不依赖于 emergent capabilities。
- **从游戏经验中学习**：对话转向了一个观点，即在各种游戏上训练模型可以促进以更少的数据学习 **shogi**（将棋）等任务。
   - 然而，另一位成员强调，如果没有针对 **shogi** 的特定训练，模型的表现将不会理想。
- **Anand 介绍：好奇的心灵**：新成员 **Anand** 介绍了自己，表达了对 AI 心智的好奇以及对 AI 风险的担忧。
   - 作为一名软件工程师，Anand 旨在学习并为以 AI 为中心的未来做出积极贡献。
- **DeepSeek R1 模型可用**：成员分享了一个运行 **Perplexity 无审查版 DeepSeek R1** 的链接，以及 Hugging Face 上提供的其他模型版本。
   - 他们指出，**dynamic 2-bit GGUF quants**（动态 2-bit GGUF 量化）比典型的 **1-bit/2-bit** 格式提高了准确度，并提供了使用教程。



**提及的链接**：<a href="https://huggingface.co/unsloth/r1-1776-GGUF">unsloth/r1-1776-GGUF · Hugging Face</a>：未找到描述

  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1341528494269665301)** (32 条消息🔥): 

> `训练 Diffusion 模型的 Model-guidance, Deepseek V2 改进, 用于优化 Kernel 的 AI CUDA Engineer, 强化学习训练课程, 门控机制中的 Sigmoid 与 Softmax` 


- **Model-guidance 技术增强 Diffusion 模型训练**：一种新型目标函数 Model-guidance (MG) 消除了 Classifier-free guidance (CFG)，并显著加快了训练速度，使推理速率翻倍，同时在 ImageNet 256 基准测试中实现了 **1.34** 的 FID，达到 SOTA 性能。广泛的实验验证了其在各种模型和数据集上的有效性。
   - *广泛的评估表明了 MG 的能力*，为 Diffusion 模型训练设定了新标准。
- **Deepseek V2 从 softmax 转向 sigmoid**：Deepseek V2 在门控机制中从 **softmax** 切换到了 **sigmoid**。讨论指出，sigmoid 可以增强训练稳定性，但由于其封顶行为，会使学习门控乘数变得复杂。成员们辩论了使用 sigmoid 与 softmax 在训练动态和专家选择（expert selection）方面的差异。
   - 值得注意的观点包括，当 sigmoid 值接近 **1** 时，区分门控乘数的能力会减弱，从而导致专家选择的潜在波动。
- **引入用于 Kernel 优化的 AI CUDA Engineer**：新推出的 [AI CUDA Engineer](http://sakana.ai/ai-cuda-engineer/) 能够自动创建优化的 CUDA kernels，据称比标准 PyTorch 操作实现了 **10-100x** 的加速。其进化方法能够自主提升 CUDA kernels 的性能，并发布了包含 **17,000** 个经过验证的 kernels 的大型数据集。
   - 团队认为这一进展标志着 AI 驱动的机器学习运营效率进入了新时代，对自动优化推理时间具有重要意义。
- **探讨基于课程的强化学习**：强化学习中的一种新方法强调在训练期间优先处理成功率方差较高的问题，以有效增强学习信号。使用该模型的模型在应对训练过程中解决问题成功率的变化时，表现出持续的性能提升。
   - **PPO** 与 **VinePPO** 的比较反映了对训练效率中实际用时（wall-clock time）和计算时间考虑的见解。
- **关于 Loss 曲线 Scaling 的讨论**：用户对将 Scaling laws 拟合到训练中观察到的 Loss 曲线表示好奇，并注意到 Diffusion 模型与 LLM 模型之间的行为差异。具体表现包括 Diffusion Loss 下降得更快，以及受退火策略（annealing schedules）影响的模型训练动态变化。
   - 观察结果暗示了训练过程的复杂性，包括专家行为如何根据训练期间的输入信号产生波动。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/SakanaAILabs/status/1892385766510338559">来自 Sakana AI (@SakanaAILabs) 的推文</a>：介绍 AI CUDA Engineer：一个能够自动生成高度优化 CUDA kernel 的 Agentic AI 系统。http://sakana.ai/ai-cuda-engineer/ AI CUDA Engineer 可以生成高度优化的...</li><li><a href="https://homietele.github.io">未找到标题</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2502.12272">Learning to Reason at the Frontier of Learnability</a>：强化学习现在被广泛采用作为大语言模型训练的最后阶段，特别是针对数学题等推理类任务。通常，模型会对每个问题尝试多次...</li><li><a href="https://arxiv.org/abs/2502.12154">Diffusion Models without Classifier-free Guidance</a>：本文提出了 Model-guidance (MG)，这是一种用于训练扩散模型的新型目标函数，旨在解决并移除常用的 Classifier-free guidance (CFG)。我们的创新方法超越了...</li><li><a href="https://arxiv.org/abs/2502.07316">CodeI/O: Condensing Reasoning Patterns via Code Input-Output Prediction</a>：推理是 LLM 的一项基本能力。虽然之前的研究主要集中在增强数学或代码生成等狭窄技能上，但提高在许多其他领域的表现...</li><li><a href="https://arxiv.org/abs/2502.13130">Magma: A Foundation Model for Multimodal AI Agents</a>：我们介绍了 Magma，这是一个为数字和物理世界中的多模态 AI Agent 任务服务的基础模型。Magma 是对视觉语言 (VL) 模型的重大扩展，它不仅...</li><li><a href="https://github.com/tzco/Diffusion-wo-CFG/blob/e86a3002df0aa086c7630a1fe379e9fb9564c2ff/train.py#L378)">Diffusion-wo-CFG/train.py at e86a3002df0aa086c7630a1fe379e9fb9564c2ff · tzco/Diffusion-wo-CFG</a>：无需 Classifier-free Guidance 的扩散模型的官方实现 - tzco/Diffusion-wo-CFG
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1341500977911566386)** (26 messages🔥): 

> `LLM Scaling Laws 术语, Scaling Laws 分类法, LLM 资源分配, Pre-training vs Post-training, LLM 开发关注领域` 


- **理解 Scaling Laws 术语**：一位成员寻求关于 **LLM scaling laws** 相关术语的澄清，包括数据、模型大小和 test-time compute 在提升性能中的作用。
   - 大家一致认为，**90% 的 scaling laws** 讨论都集中在 pretraining 指标上，同时也强调了训练与部署考虑因素之间的关系。
- **为获得洞察而对 Scaling Laws 进行分类**：一位成员表示有兴趣创建一个 scaling laws 分类法，以便有效地对计算资源进行分类，从而分析 LLM 领域的进展。
   - 另一位成员指出，虽然 scaling laws 提供了一个框架，但关于资源分配的实际数据并未公开，这使得预算比较变得复杂。
- **Pre-training 与 Post-training 之间的预算分配**：成员们讨论了大型实验室在模型开发的 pre-training 与 post-training 阶段的支出是否正趋于平衡。
   - 尽管有推测，但尚不确定这是否包括与数据获取相关的成本。
- **近期 LLM 发布的关注领域**：近期的进展突显了对优化 **test-time compute** 的日益关注，特别是在 **Grok 3** 等最新模型发布的背景下。
   - 有人指出，扩大 pretraining compute 规模仍在产生收益，这挑战了此前认为此类努力已走入死胡同的假设。
- **数据和 Finetuning 洞察方面的挑战**：成员们承认，很难获取实验室在创意写作、数学/编程以及 LLM 对齐等类别中使用的特定数据。
   - 虽然可以从开源模型中推断出模型大小，但数据利用和 finetuning 方法背后的“秘方”在很大程度上仍然是不透明的。


  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1341662900623769600)** (5 messages): 

> `Recurrent Model Interpretability, Logit Lens, Tuned Lens, Counterfactual Testing, Average-case Goals` 


- **循环模型可解释性：一个难啃的硬骨头**：讨论集中在实现有效的 **recurrent model interpretability**（循环模型可解释性）的困难上，特别是在长程推理任务中。
   - 一位成员将这一挑战比作计算机安全中的 **fuzzing** 和寻找 **backdoors**，强调了其中的复杂性。
- **探索用于可解释性的 Logit 和 Tuned Lenses**：一位成员建议重新审视 **Logit Lens** 和 **Tuned Lens**，认为在 Transformer 和循环模型中使用这些工具仍有未开发的潜力。
   - 他们强调了在复杂任务中分析模型每一步推理过程的附加价值。
- **模型训练中的反事实测试**：另一位成员提议，训练模型根据其 latents（潜变量）标记具有信息量的 **counterfactuals**（反事实），可以增强在未知压力测试期间的可解释性。
   - 即使在模型没有显式隐藏必要线索的情况下，这也能提供见解。
- **权衡平均情况（Average-case）与最坏情况（Worst-case）目标**：一场关于通过 **average-case** 目标而非 **worst-case** 场景获得更好性能的潜力的对话展开了。
   - 这种方法表明，专注于自然配置的 latents 可能会产生更成功的结果。



**提及的链接**：<a href="https://x.com/cfgeek/status/1892097007394652187?s=46">Charles Foster (@CFGeek) 的推文</a>：@alextmallen 我一直在思考这个问题。好奇是否有办法训练出一种东西，在给定模型对特定输入的 latents（例如，来自循环推理器）时，能够标记出特定的反事实...

  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1341515704444387328)** (5 messages): 

> `Chess Tactic Dataset Structure, Eval Harness Support for NeMo Checkpoints` 


- **辩论国际象棋战术数据集结构**：一位成员对国际象棋战术数据集的结构表示怀疑，考虑是预测完整的移动序列还是将其限制为单步提示。
   - 他们指出，虽然包含合法移动可以减少非法移动的预测，但由于序列增长过快，添加所有可能的序列是不切实际的。
- **Eval Harness 与 NeMo Checkpoints**：一位成员询问 Eval Harness 是否支持 NeMo 在训练期间保存的 .distcp checkpoints，并指出它已支持 NeMo checkpoints。
   - 另一位成员不确定，但分享了 [Eval Harness GitHub 中的一个函数](https://github.com/EleutherAI/lm-evaluation-harness/blob/52df63b7b30da53c481ed9090598d9189fab1d91/lm_eval/models/nemo_lm.py#L71)，该函数用于加载模型。



**提及的链接**：<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/52df63b7b30da53c481ed9090598d9189fab1d91/lm_eval/models/nemo_lm.py#L71)">lm-evaluation-harness/lm_eval/models/nemo_lm.py at 52df63b7b30da53c481ed9090598d9189fab1d91 · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness

  

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1341454265436082292)** (32 条消息🔥): 

> `NeMo vs GPT-NeoX 性能对比、A100 配置、Transformer Engine 集成、Evo2 基因组模型、模型训练中的通信策略` 


- **NeMo 可能具有 TP 通信重叠优势**：**NeMo** 有可能利用了 TP 通信重叠（overlap），从而潜在地提升了在 **PCIe** 连接上的性能。成员们讨论了使用 `ub_tp_comm_overlap` 标志的影响，以及切换到 TE MLP 是否有助于缓解问题。
   - 一位成员指出，他们没有在 NeMo 中显式设置任何通信标志，这意味着正在使用默认设置，同时强调需要进一步探索差异的原因。
- **测试 allreduce bucket 大小**：讨论涉及了匹配 NeMo 和 NeoX 之间的 **allreduce bucket 大小** 所产生的影响，有建议认为更大的尺寸可能会提高 **PCIe** 上的速度。一位成员分享称，他们在 NeoX 中尝试了 2 倍和 4 倍的大小，但没有观察到任何速度提升。
   - 这引发了关于被测试配置及其对整体模型性能影响的疑问。
- **基于 GPT-NeoX 训练的 Evo2 基因组模型**：一位成员宣布，新的 **Evo2 基因组模型** 是使用一个基于 **GPT-NeoX** 构建的库进行训练的，这与社区的工作联系在了一起。另一位成员对这一认可表示兴奋，并对这种回报性的成果表示赞赏。
   - 这突显了社区对先进模型训练方法的持续贡献。
- **A100 配置及其影响**：对话探讨了所使用的 **A100** 是 **PCIe** 还是 **NVLINK** 版本，一位成员确认它们确实是 PCIe。这一发现引发了关于通信速度如何影响性能指标的讨论。
   - 强调了根据接口类型优化设置的必要性，因为这可能会影响每秒最大事务数（TPS）。
- **Transformer Engine 集成的挑战**：一位成员提到尝试在 NeoX 中集成 **Transformer Engine**，但遇到了阻碍进一步测试的挑战。他们表示开启非 FP8 标志会导致系统崩溃，这表明当前设置可能存在不稳定性。
   - 这概述了在保持系统可靠性的同时集成新技术所涉及的复杂性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/aflah02/gpt-neox/blob/olmo-support/configs/pythia/6-9B.yml">gpt-neox/configs/pythia/6-9B.yml at olmo-support · aflah02/gpt-neox</a>: 基于 Megatron 和 DeepSpeed 库在 GPU 上实现模型并行自回归 Transformer - aflah02/gpt-neox</li><li><a href="https://github.com/NVIDIA/NeMo/blob/0621272c2a9a760a71b234131f1997e87a265943/nemo/collections/nlp/models/language_modeling/megatron_base_model.py#L882">NeMo/nemo/collections/nlp/models/language_modeling/megatron_base_model.py at 0621272c2a9a760a71b234131f1997e87a265943 · NVIDIA/NeMo</a>: 为从事 LLM、多模态和语音 AI（自动语音识别和文本转语音）的研究人员和开发人员构建的可扩展生成式 AI 框架 - NVIDIA/NeMo
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1341456411972079769)** (38 messages🔥): 

> `Grok 3 的新游戏工作室, 注意力机制反向传播, Mamba 深度学习架构, Transformer 推理复杂度, Jailbreak 挑战赛见解` 


- **Grok 3 的新游戏工作室公告**：Elon Musk 确认 xAI 将启动一个游戏工作室，这是其近期 AI 计划的逻辑延伸，正如最近的 [Twitter 帖子](https://x.com/elonmusk/status/1891388509191049307) 中所讨论的那样。
   - *Dima Zeniuk* 提到了这一向游戏领域的转变与 Grok 3 的品牌更新有关。
- **理解 Transformer 中的反向传播**：一场关于通过 Transformer 架构进行反向传播复杂性的讨论，强调了注意力机制带来的独特挑战。
   - 专家建议从特定操作（如 Attention）开始，以便更清晰地理解梯度计算。
- **Mamba 架构 vs. 传统 Transformer**：为处理长序列而开发的 Mamba 架构，由于采用了结构化状态空间方法，可能避免了原生 Transformer 中典型的 O(n^2) 复杂度。
   - Mamba 专为效率而设计，与在推理过程中需要大量计算的 Transformer 相比，具有线性复杂度。
- **近期 Jailbreak 挑战赛的见解**：在一份详细分析中讨论到，近期 Jailbreak 挑战赛的成功可能要求参与者签署保密协议，从而限制了发现结果的分享。
   - 一份共享的 [Google 文档](https://docs.google.com/document/d/1WI3IaiYCDQUk5YrFK6MP_YxOXwyy3mV9a8d66ndJeyI/edit?tab=t.0) 提供了关于当前 Jailbreak 方法的额外资源。
- **探索不含 Softmax 的损失函数**：一位用户提出在 Embedding 空间中定义均方误差 (MSE) 损失，随后使用交叉熵 (Cross Entropy) 训练分类头，从而避免在 Transformer 块中使用 Softmax。
   - 这表明在保持有效性能的同时，有可能简化 Transformer 架构。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.06773">On the Emergence of Thinking in LLMs I: Searching for the Right Intuition</a>: 最近的 AI 进展（如 OpenAI 的新模型）正在将 LLM 转化为 LRM (Large Reasoning Models)，在推理过程中执行推理，消耗额外的时间和计算资源以获得更高质量...</li><li><a href="https://en.wikipedia.org/wiki/Mamba_(deep_learning_architecture)">Mamba (deep learning architecture) - Wikipedia</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=X_niF6KaWd8">🚨🚨 Chat Creates Doom With Devin.ai 🚨🚨</a>: Twitch https://twitch.tv/ThePrimeagen Discord https://discord.gg/ThePrimeagen 成为后端开发：https://boot.dev/prime (此外我为他们制作了课程) 这是一个...</li><li><a href="https://www.youtube.com/watch?v=nsOMuWD58N0&list=PLgKuh-lKre1058wlfuwtOYyemY5qoxavQ">How Do Transformers Learn Variable Binding?</a>: Raphaël Millière (Macquarie University) https://simons.berkeley.edu/talks/raphael-milliere-macquarie-university-2025-02-07 LLMs, 认知科学, 语言学...</li><li><a href="https://fxtwitter.com/janleike/status/1890141865955278916?t=m7VUL3XBshTSSrfzBa9Wzw&s=19">Tweet from Jan Leike (@janleike)</a>: 我们的 Jailbreaking 挑战赛结果：经过 5 天、超过 300,000 条消息以及约 3,700 小时的集体努力，我们的系统被攻破了。最终有 4 名用户通过了所有关卡，1 名用户发现了通用 Jailbreak 方法。我们...</li><li><a href="https://docs.google.com/document/d/1WI3IaiYCDQUk5YrFK6MP_YxOXwyy3mV9a8d66ndJeyI/edit?tab=t.0">Jailbreaking LLMs</a>: 大语言模型 Jailbreaking，由 Stu Jordan (Evolution Unleashed Lab, X 账号 @stujordanAI) 准备。日期：2025 年 2 月 7 日。鉴于 Anthropic 的挑战，我想分享这个资源...</li><li><a href="https://x.com/elonmusk/status/1891388509191049307">Tweet from Elon Musk (@elonmusk)</a>: 是的。引用 Dima Zeniuk (@DimaZeniuk)：Elon Musk 的 xAI 将启动一个 AI 游戏工作室来制作游戏。
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1341507171828891712)** (50 条消息🔥): 

> `AI Paper Discussions, DeepSeek's Sparse Attention Paper, Challenges in Paper Selection, Discord Event Organization, User Participation` 


- **AI 论文中的挑战与机遇**：参与者讨论了 AI 论文的泛滥，以及过滤噪声以关注有价值见解的重要性，一位成员表示，*烂论文只是另一种机会*。另一位提到，目前 AI 论文的大量涌入使得避免选到劣质论文变得困难。
   - 一位成员分享说，识别一篇论文为何不足可以培养宝贵的技能，并强调了他们面临的时间限制。
- **DeepSeek 关于 Sparse Attention 的论文**：今天的讨论重点是 DeepSeek 的论文，题目为 **Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention** ([链接](https://arxiv.org/abs/2502.11089))，以无需预先准备的 cold read 形式呈现。一位用户评价该论文质量极高，称其为 *banger*。
   - 活动定于特定时间举行，一些人期待周末可能会有重新讨论。
- **日程安排与参与度问题**：讨论中强调了时间和参与度，并询问会议是否会录音以确保隐私。会议指出，为了鼓励开放对话，禁止录音，这让所有参与者的氛围更加轻松。
   - 几位成员表示打算补读错过的论文，并计划进行后续讨论以回顾关键发现。
- **提高活动出席率**：讨论组织者建议，提前发布会议通知可以增加出席率，因为更多成员可能能够加入。一位成员鼓励其他人表达对活动的兴趣，以便有效评估参与人数。
   - 个人日程冲突和时区差异被认为是某些参与者面临的主要挑战。
- **社区参与和反馈**：成员们对讨论表示赞赏，并称赞了会议期间分享的见解，再次肯定了社区反馈的价值。一些人还分享了未来更积极参与讨论的计划，强调了他们对个人学习的承诺。
   - 社区强调了持续讨论对集体成长的重要性，为即将到来的会议中建设性的参与奠定了基础。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.11089">Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention</a>: 长上下文建模对于下一代语言模型至关重要，但标准 Attention 机制的高计算成本带来了显著的计算挑战。Sparse Attention 提供...</li><li><a href="https://arxiv.org/abs/2502.09696">ZeroBench: An Impossible Visual Benchmark for Contemporary Large Multimodal Models</a>: Large Multimodal Models (LMMs) 在解释图像时表现出重大缺陷，在某些衡量标准下，其空间认知能力甚至不如小孩或动物。尽管如此，它们仍获得了高分...</li><li><a href="https://arxiv.org/abs/2502.12150">Idiosyncrasies in Large Language Models</a>: 在这项工作中，我们揭示并研究了 Large Language Models (LLMs) 中的特质（idiosyncrasies）——其输出中可用于区分模型的独特模式。为此，我们考虑了一个简单的分类...
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1341458105224925245)** (13 条消息🔥): 

> `Los Angeles Project 独角兽工程、Mira Murati 的 Thinking Machines Lab、Elon 的 Grok 3、Perplexity AI 克服审查、Microsoft 的 Majorana 1 量子处理器` 


- **Los Angeles Project 旨在打造独角兽**：[The Los Angeles Project](https://building.life/) 是一家生物技术初创公司，旨在通过基因编辑和尖端技术制造出真正的、字面意义上的独角兽。
   - 人们对其招聘宣传中缺乏 AI 内容表示担忧，这表明 AI 可能仅仅是吸引投资的噱头。
- **Mira Murati 成立 Thinking Machines Lab**：前 OpenAI CTO Mira Murati 创立了 [Thinking Machines Lab](https://thinkingmachines.ai)，专注于使 AI 系统更易理解和定制，同时确保公众透明度。
   - 即将开展的项目细节仍处于保密状态，但他们承诺会定期发布技术研究成果。
- **Elon 的 Grok 3 备受关注**：一段 [YouTube 视频](https://www.youtube.com/watch?v=b0XI-cbel1U)提出了“Elon 的 Grok 3 是新的 AI 之王吗？”的问题，关于其能力的讨论正在升级。
   - 演示暗示了潜在的进步，但对其具体影响的看法各不相同。
- **Perplexity AI 突破审查障碍**：[Perplexity AI](https://the-decoder.com/perplexity-ai-removes-chinese-censorship-from-deepseek-r1/) 推出了 R1 1776，这是一个通过创新的 post-training 技术克服中国审查的模型。
   - 这种方法旨在建立更强大的 AI 系统，在生成内容的同时保持上下文的完整性。
- **Microsoft 的 Majorana 1 引领量子飞跃**：Microsoft 发布了 Majorana 1，这是首个由拓扑量子比特（topological qubits）驱动的量子处理器（QPU），单块芯片可扩展至一百万个量子比特。
   - 这一进步标志着向实用量子计算迈出了重要一步，并预示着技术新时代的到来。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.theverge.com/ai-artificial-intelligence/614621/mira-murati-thinking-machines-lab-openai-competitor-launch">Mira Murati is launching her OpenAI rival: Thinking Machines Lab</a>：前员工正在建立另一个 OpenAI 的竞争对手。</li><li><a href="https://x.com/Jiankui_He/status/1892111600770863426">来自 贺建奎 (@Jiankui_He) 的推文</a>：胚胎基因编辑可以预防癌症。通过将 DNA 修复能力提高 100 倍，我们的孩子将不再有任何基因突变，从而永久根除癌症。然而，我...</li><li><a href="https://www.youtube.com/watch?v=b0XI-cbel1U">Is Elon’s Grok 3 the new AI king?</a>：免费试用 Brilliant 30 天 https://brilliant.org/fireship 你还将获得年度高级订阅 20% 的折扣。抢先了解 Elon Musk 的 Grok 3 ...</li><li><a href="https://x.com/elder_plinius/status/1891968598496760230?s=46">来自 Pliny the Liberator 🐉󠅫󠄼󠄿󠅆󠄵󠄐󠅀󠄼󠄹󠄾󠅉󠅭 (@elder_plinius)</a>：🧙‍♂️ 󠅗󠅗解锁新的攻击类别 🧙‍♂️󠅗󠅗</li><li><a href="https://www.youtube.com/watch?v=wSHmygPQukQ">Majorana 1 Explained: The Path to a Million Qubits</a>：听取 Microsoft 团队关于近期物理学和量子计算突破的介绍，该突破由新型 Majorana 1 芯片展示，该芯片由全新的...</li><li><a href="https://the-decoder.com/perplexity-ai-removes-chinese-censorship-from-deepseek-r1/">Perplexity AI removes Chinese censorship from Deepseek R1</a>：Perplexity AI 发布了 R1 1776，这是 Deepseek-R1 语言模型的修改版本，专门设计用于通过专门的 post-training 技术克服中国审查。</li><li><a href="https://www.piratewires.com/p/harnessing-the-breath-of-life">Harnessing the Breath of Life</a>：一家名为 Los Angeles Project 的基因编辑初创公司将如何创造真正的、字面意义上的独角兽（以及更多）</li><li><a href="https://building.life/">LAP</a>：未找到描述</li><li><a href="https://azure.microsoft.com/en-us/blog/quantum/2025/02/19/microsoft-unveils-majorana-1-the-worlds-first-quantum-processor-powered-by-topological-qubits/">Microsoft unveils Majorana 1, the world’s first quantum processor powered by topological qubits - Microsoft Azure Quantum Blog</a>：来自 Microsoft 的 Majorana 1 是全球首个使用拓扑导体构建的量子处理器（QPU）。了解更多。
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1341489757980725279)** (79 条消息🔥🔥): 

> `Thinking Machines Lab 发布、Perplexity AI R1 1776 发布、OpenAI SWElancer 基准测试、Mastra 开源 JS 框架、云 AI 基础设施与融资`

- **Thinking Machines Lab 成立**：Thinking Machines Lab 旨在让 AI 系统更易于访问和定制，同时解决前沿 AI 系统中的知识鸿沟。他们汇集了来自 [ChatGPT](https://chatgpt.com) 和 [Character.ai](https://character.ai/) 等热门 AI 产品背后的专家团队。
   - OpenAI 前 CTO Mira Murati 是这一新创公司的重要人物，共同创始人还包括 Lilian Weng 等。
- **Perplexity AI 推出 R1 1776**：Perplexity AI 宣布开源 R1 1776，这是其 DeepSeek R1 模型的一个版本，提供无审查且无偏见的信息。这标志着在为不同领域提供更可靠 AI 模型方面迈出了重要一步。
   - AI 社区幽默地将这一进展称为 'freedomtuning'，以此致敬该模型的新特性。
- **OpenAI 发布 SWElancer 基准测试**：OpenAI 推出了 SWElancer，这是一个用于评估 AI 编程性能的新基准测试，包含 1,400 多个自由职业软件工程任务。其目标是为理解 AI 在现实应用中的能力创建一个更真实的衡量标准。
   - 这一举措正值有关 AI 驱动的游戏生成可能取代游戏工作室的讨论之际。
- **Mastra 开源框架发布**：Mastra 发布了一个开源 JavaScript SDK，旨在构建 AI Agent，允许通过内置工作流执行复杂任务。它有望促进能够执行复杂任务的自主 Agent 的开发。
   - 该框架鼓励与 Vercel 的 AI SDK 进行协作和集成，标志着开源 AI 开发迈出了重要一步。
- **Lambda 获得 D 轮融资**：Lambda 宣布完成 4.8 亿美元的大规模 D 轮融资，由 Andra Capital 和 SGW 领投。这笔投资反映了人们对为 AI 应用量身定制的云服务日益增长的兴趣。
   - NVIDIA 和 ARK Invest 等知名投资者的参与凸显了该公司在不断发展的 AI 领域中的潜力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://thinkingmachines.ai/]">Thinking Machines Lab</a>: 未找到描述</li><li><a href="https://thinkingmachines.ai/">Thinking Machines Lab</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/nanotron/ultrascale-playbook">The Ultra-Scale Playbook - nanotron 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://x.com/dchaplot/status/1891920016339042463">Devendra Chaplot (@dchaplot) 的推文</a>: 职业更新：非常幸运且兴奋能成为 Thinking Machines Lab 创始团队的一员！https://thinkingmachines.ai/ 加入我们：https://6wajk07p.paperform.co/</li><li><a href="https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/">利用 AI 联合科学家加速科学突破</a>: 未找到描述</li><li><a href="https://x.com/arrakis_ai/status/1892141483941347627?s=46&t=b7l37rB6wtbyAh6ah1NpZQ">CHOI (@arrakis_ai) 的推文</a>: Claude Extended Thinking!</li><li><a href="https://mastra.ai/docs/agents/00-overview">创建与调用 Agent | Agent 文档 | Mastra</a>: 未找到描述</li><li><a href="https://www.theverge.com/ai-artificial-intelligence/614621/mira-murati-thinking-machines-lab-openai-competitor-launch">Mira Murati 正在启动她的 OpenAI 竞争对手：Thinking Machines Lab</a>: 前员工正在建立另一家 OpenAI 的竞争对手。</li><li><a href="https://x.com/perplexity_ai/status/1891916573713236248">Perplexity (@perplexity_ai) 的推文</a>: 今天我们开源了 R1 1776——这是 DeepSeek R1 模型的一个版本，经过后训练以提供无审查、无偏见且事实准确的信息。</li><li><a href="https://en.wikipedia.org/wiki/Thinking_Machines_Corporation">Thinking Machines Corporation - 维基百科</a>: 未找到描述</li><li><a href="https://x.com/ashtom/status/1891925306430337110?s=46&t=PW8PiFwluc0tdmv2tOMdEg">Thomas Dohmke (@ashtom) 的推文</a>: 我们的新代码补全模型今天发布公开预览版。我们将其命名为 GPT-4o Copilot。基于 GPT-4o mini，在超过 1T tokens 的代码语料库上进行了中期训练（mid-training），并进行了强化学习...</li><li><a href="https://www.youtube.com/watch?v=mFuyX1XgJFg&ab_channel=Apple">iPhone 16e 介绍 - 2 月 19 日</a>: 价格最优惠的最新 iPhone —— iPhone 16e 来了。它专为 Apple Intelligence 打造，由最新一代芯片 A18 驱动。它还配备了...</li><li><a href="https://x.com/hyusapx/status/1891642232635306053?s=46">ayush (@hyusapx) 的推文</a>: 为词源学爱好者构建了一个 AI 工具：deconstructor。现在已上线。目标是分析英语单词的起源，但它最终产生了很多惊人的涌现属性（见下文）</li><li><a href="https://x.com/pitdesi/status/1891979324787482769">Sheel Mohnot (@pitdesi) 的推文</a>: Humane AI pin 正在收尾，HP 以 1.16 亿美元收购其团队、IP 和软件。创始人 Imran 和 Bethany 将在 HP 成立一个新部门，将 AI 集成到 HP 的 PC、打印机和联网会议设备中...</li><li><a href="https://x.com/basetenco/status/1892259130540179863">Baseten (@basetenco) 的推文</a>: 2025 年是推理（inference）之年。我们很高兴宣布由 @IVP 和 @sparkcapital 领投的 7500 万美元 C 轮融资，@GreylockVC、@conviction、@basecasevc、@southpkcommons 等参与了跟投...</li><li><a href="https://www.threads.net/@zuck/post/DGOISdTRX9Q?xmt=AQGzKO9jUlLz9JyNocdtLrsQ1L8IvvVBRb--7JMStLY6Fg">Mark Zuckerberg (@zuck) 在 Threads 上</a>: 记下日期 🗓️ LlamaCon: 4 月 29 日；Connect: 9 月 17-18 日</li><li><a href="https://x.com/loubnabenallal1/status/1892278622104215894?s=46">Loubna Ben Allal (@LoubnaBenAllal1) 的推文</a>: nanotron 团队刚刚发布了 The Ultra-Scale Playbook，包含关于小规模和大规模 LLM 预训练所需了解的一切 https://huggingface.co/spaces/nanotron/ultrascale-playbook</li><li><a href="https://x.com/nearcyan/status/1892049572546904319?s=46">near (@nearcyan) 的推文</a>: 每个买了 700 美元 AI pin 的人都被坑了。引用 Elvin (@elvin_not_11)：离线功能就像电池电量一样</li><li><a href="https://x.com/hellenicvibes/status/1892250276473516266?s=46">Zoomer Alcibiades (@HellenicVibes) 的推文</a>: Google 的 AI Agent 独立发现了：一种新的白血病药物，随后在临床浓度下成功进行了体外测试；新的肝纤维化药物靶点；细菌细胞级的抗生素...</li><li><a href="https://siliconangle.com/2025/02/17/ilya-sutskevers-safe-superintelligence-reportedly-raising-1b-30b-valuation/">据报道 Ilya Sutskever 的 Safe Superintelligence 以 300 亿美元估值筹集 10 亿美元以上 - SiliconANGLE</a>: 据报道 Ilya Sutskever 的 Safe Superintelligence 以 300 亿美元估值筹集 10 亿美元以上 - SiliconANGLE</li><li><a href="https://x.com/leonardtang_/status/1892243653071908949">Leonard Tang (@leonardtang_) 的推文</a>: 首先是预训练扩展（pre-training scaling）；然后是推理时扩展（inference-time scaling）。现在轮到 j...</li>

udge-time scaling（评判时缩放）。尽管 AI 通过缩放推理时计算（inference-time compute）取得了进展，但在开放式、非验证性的任务中，AI 仍然不可靠...</li><li><a href="https://x.com/stephenbalaban/status/1892275552171737220?s=46">来自 stephen balaban (@stephenbalaban) 的推文</a>：Lambda 是为 AI 时代设计的云。今天，我们宣布完成了由 Andra Capital 和 SGW 领投的 4.8 亿美元 D 轮融资，NVIDIA、Andrej Karpathy、In-Q-Tel、ARK Invest 等参与了投资。</li><li><a href="https://x.com/scaling01/status/1891913189199053301?s=46">来自 Lisan al Gaib (@scaling01) 的推文</a>：我喜欢 OpenAI 每篇论文中为 Sonnet 3.5 做的广告。有点奇怪他们没有包含 o3。现在感觉要么这是一个局，以便他们以后可以打破自己的基准测试，要么就是 o3 在某些方面表现很差...</li><li><a href="https://youtu.be/4GLSzuYXh6w?si=K9JWK5HMIQzxFk7v">Satya Nadella – 微软的 AGI 计划与量子突破</a>：Satya Nadella 谈论：- 为什么他不相信 AGI，但相信 10% 的经济增长，- 微软新的拓扑量子比特（topological qubit）突破以及游戏世界的动向...</li><li><a href="https://x.com/thom_wolf/status/1892273133547078036?s=46">来自 Thomas Wolf (@Thom_Wolf) 的推文</a>：经过 6 个多月的准备并消耗了超过一年的 GPU 计算时间，我们非常激动地终于发布了“Ultra-Scale Playbook”。点击此处查看：http://hf.co/spaces/nanotron/...</li><li><a href="https://x.com/openai/status/1891911132983722408?s=46">来自 OpenAI (@OpenAI) 的推文</a>：目前的尖端模型无法解决大多数任务。</li><li><a href="https://youtu.be/Ju0ndy2kwlw?si=_Maiv6-7b0dv3vLg">我用 5 台 Mac Studio 建造了一台 AI 超级计算机</a>：获取 NordVPN 2 年计划 + 额外 4 个月 + 再额外 6 个月：https://nordvpn.com/networkchuck。Nord 提供 30 天退款保证，无风险！我刚刚...</li><li><a href="https://tenor.com/view/freedom-america-gif-15593845046973100361">自由美国 GIF - 自由美国 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/thinkymachines/status/1891919141151572094">来自 Thinking Machines (@thinkymachines) 的推文</a>：今天，我们很高兴地宣布成立 Thinking Machines Lab (https://thinkingmachines.ai/)，这是一家人工智能研究和产品公司。我们是某些项目背后的科学家、工程师和构建者...</li><li><a href="https://news.ycombinator.com/item?id=43103073">Show HN: Mastra – 开源 JS Agent 框架，由 Gatsby 的开发者打造 | Hacker News</a>：未找到描述</li><li><a href="https://youtu.be/4poqjZlM8Lo?si=FYxRdCbX_SPg_3Wj">AI 大佬们谈论让他们彻夜难眠的事</a>：Google DeepMind 和 Anthropic 的创始人 Demis Hassabis 和 Dario Amodei 是全球最顶尖的人工智能领导者。我们的总编辑...</li><li><a href="https://youtu.be/4GLSzuYXh6w?si=74SFFf1tGUdflWdm&t=2643">Satya Nadella – 微软的 AGI 计划与量子突破</a>：Satya Nadella 谈论：- 为什么他不相信 AGI，但相信 10% 的经济增长，- 微软新的拓扑量子比特（topological qubit）突破以及游戏世界的动向...</li><li><a href="https://x.com/karpathy/status/1891938714915569711">来自 Andrej Karpathy (@karpathy) 的推文</a>：祝贺 Thinking Machines 公司成立！非常强大的团队，其中很大一部分人直接参与并创造了 ChatGPT 的奇迹。很棒的人，值得关注，祝愿...</li><li><a href="https://techcrunch.com/2025/02/18/humanes-ai-pin-is-dead-as-hp-buys-startups-assets-for-116m/">Humane 的 AI Pin 宣告失败，HP 以 1.16 亿美元收购该初创公司资产 | TechCrunch</a>：Humane 周二宣布，其大部分资产已被 HP 以 1.16 亿美元收购。这家硬件初创公司将立即停止销售</li><li><a href="https://www.reddit.com/r/apple/comments/1it9839/apple_polishing_cloth_adds_support_for_the_new/">Reddit - 深入探索一切</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1341699849317388328)** (28 条消息🔥): 

> `Grok vs Mojo, Mojo 中的 Polars 实现, MAX 25.1 直播活动, 社区会议演讲` 


- **Grok 的发布令 Mojo 开发者感到惊喜**：**Grok 3** 的发布让一些人感到意外，因为它在 **Mojo** 达到其中期目标之前就已问世。
   - *这对 Mojo 来说实际上是一件好事，* 引起了一些成员的兴奋。
- **Polars 集成到 Mojo 项目中**：一位成员分享说，他们快速地将 **Polars 导入** 到了他们的 Mojo 项目中，并在 [GitHub](https://github.com/rcghpge/pymo/tree/main) 上展示了他们的实现示例。
   - 关于 *实现 (implementing)* 和 *导入 (importing)* Polars 之间的区别存在争论，成员们澄清了他们是如何使用它的。
- **即将举行的 MAX 25.1 直播**：明天计划举行一场直播，讨论有关 **MAX 25.1** 的所有内容，鼓励成员通过 [Google Form](https://forms.gle/NkjU6e3n15TRtiMA7) 提交问题。
   - 他们邀请所有人通过 [LinkedIn](https://www.linkedin.com/events/introducingmax25-17297704283980902402/theater/) 加入该环节。
- **社区会议演讲公开征集**：成员们获悉了在周一的社区会议上发表演讲的机会，鼓励分享项目或关注领域。
   - *我们正在寻找引人入胜的演示，* 并邀请感兴趣的成员踊跃参与。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://forms.gle/NkjU6e3n15TRtiMA7">Modular Community Q&amp;A</a>: 未找到描述</li><li><a href="https://github.com/rcghpge/pymo/blob/main/pymo/libpm/libpm.mojo#L42">pymo/pymo/libpm/libpm.mojo at main · rcghpge/pymo</a>: 一个用于 AI/ML/DL 应用和其他领域的 Mojo 框架。 - rcghpge/pymo</li><li><a href="https://github.com/rcghpge/pymo/blob/main/examples/polars.mojo">pymo/examples/polars.mojo at main · rcghpge/pymo</a>: 一个用于 AI/ML/DL 应用和其他领域的 Mojo 框架。 - rcghpge/pymo
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1341501544872280085)** (42 条消息🔥): 

> `Mojo Stack 实现, Mojo 中 Quick Sort 的性能, Slab List vs Linked List, Mojo 的 VS Code 配置, 在 Mojo 中使用 Set 作为 Dict 的值` 


- **探索 Mojo 中高效的 Stack 实现**：一位用户询问了 Mojo 中替代的 Stack 实现，特别是质疑使用指针还是 `ArcPointer` 或 `UnsafePointer`。
   - 另一位成员指出，`LinkedList` 可用于添加或删除操作，在内存分配方面提供了灵活性。
- **Mojo Quick Sort 性能问题**：一位用户分享了他们的发现，即他们的 Python Quick Sort 算法性能明显优于他们的 Mojo 实现（0.4s 对比 2.9s）。
   - 随后讨论了使用 Mojo 的 benchmark 模块来隔离并仅测量排序的性能，并建议避免编译时间影响计时。
- **理解 Slab List 结构**：成员们讨论了使用 `SlabList` 优于传统 `LinkedList` 的好处，强调了常数时间操作和缓存效率。
   - 澄清了 Slab List 基本上由 `LinkedList[InlineArray[T, N]]` 组成，允许在不进行复杂操作的情况下高效使用内存。
- **在 VS Code 中设置 Mojo**：一位用户面临 VS Code 因目录结构无法识别其主包的问题，导致编辑器中出现红线导入错误。
   - 另一位成员建议参考标准库编译过程来管理导入路径。
- **在 Mojo Dictionary 中使用 Set 作为值类型**：一位用户询问 Mojo 中的 Dictionary 是否可以拥有类型为 Set 的值，引发了关于类型兼容性的讨论。
   - 确认了标准库作为一个正常的 Mojo 库运行，从而增强了实现类似数据结构的可能性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/nic">nic - Overview</a>: Chief Trolling Officer。nic 在 GitHub 上拥有 58 个代码库。</li><li><a href="https://github.com/nickziv/libslablist">GitHub - nickziv/libslablist: Slab List 是一种内存效率极高的对数时间数据结构。</a>: Slab List 是一种内存效率极高的对数时间数据结构。 - nickziv/libslablist
</li>
</ul>

</div>
  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1341454363305967688)** (56 messages🔥🔥): 

> `CUDA GPU Support, Embedding Token Limits, Chat Templates, Nomic V2 Release Delay, User Interface for Images` 


- **CUDA GPU 支持讨论**：讨论重点关注了较旧的 CUDA 5.0 兼容 GPU，如 **GM107/GM108**，并指出由于这些低端架构年代久远且实用性有限，目前缺乏支持。
   - 一名成员确认，最近一个支持这些 GPU 的 **PR** 已经合并，并将包含在下一个版本中。
- **Embedding Token 限制说明**：成员们讨论了在 GPT4All 中进行 Embedding 时达到 **1000 万 Token 限制**的问题，以及计费结构，包括 **$10/月** 的基础费用及额外 Token 的附加费用。
   - 关键点在于，从本地文档中删除 Token 并不会减少已计费的总 Token 数。
- **聊天模板 (Chat Templates) 与系统消息**：一位成员寻求关于使用 **Chat Templates** 指示模型引用摘录的澄清，确认在系统消息中加入简单指令通常就足够了。
   - 另一位成员询问了在 Prompt 中使用 Jinja 或 JSON 代码的效果，表明在实现预期输出方面存在复杂性。
- **对 Nomic v2 发布的关注**：关于 **Nomic v2** 缺席的猜测不断，成员们对其延迟感到好奇，并指出其重要性。
   - 现场气氛轻松，一名成员幽默地质疑为何在没有新版本更新的情况下等待了这么久。
- **GPT4All 图像处理的局限性**：一位成员请求能够像其他平台一样直接将图像**复制并粘贴**到聊天中，但得到的回复是 GPT4All 目前不支持图像输入。
   - 建议使用外部软件进行图像处理。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.gpt4all.io/gpt4all_desktop/chat_templates.html#using-the-cli-all-models.">Chat Templates - GPT4All</a>：GPT4All 文档 - 在您的硬件上高效运行 LLM</li><li><a href="https://en.wikipedia.org/wiki/CUDA#GPUs_supported)">CUDA - Wikipedia</a>：未找到描述
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1341457016694112367)** (20 messages🔥): 

> `Anthropic Homepage Downtime, Haiku 3.5 Release Speculations, Cursor MCP Tool Issues, Custom Protocol MCP Servers, Puppeteer Docker Build Questions` 


- **Anthropic 官网经历宕机**：一名成员报告 **Anthropic 官网宕机**，推测可能存在服务中断。
   - 附带了一张显示宕机状态的图片。
- **Haiku 3.5 带有新功能的传闻**：有讨论围绕 **Haiku 3.5** 的潜在发布展开，据称其将支持 **Tool 和 Vision** 功能。
   - 另一名成员提到也有可能看到 **Sonnet 4.0**。
- **Cursor MCP 面临工具检测问题**：多名成员对 **Cursor MCP** 报告 **'No tool found'** 表示担忧，表明这是一个普遍问题。
   - 有人分享了一个使用 **/sse** 的实现方案来解决此问题，并指向了一个共享信息链接。
- **关于自定义协议 MCP 服务器的咨询**：成员们讨论了创建不使用 SSE 或 stdio 的**自定义协议 MCP 服务器**的难度。
   - 有人建议通过 **Docker** 封装解决方案，并利用桥接进行集成。
- **关于使用 Docker 构建 Puppeteer 的问题**：一位新用户寻求安装 **Puppeteer** 最后步骤的指导，特别是关于构建 Docker 镜像的问题。
   - 他们询问是否需要特定的目录更改，以及安装 Docker 是否是先决条件。


  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1341611580218937504)** (27 条消息🔥): 

> `Google Workspace MCP, Dockerized MCP 服务器, 与 Sage 集成, 带有 MCP 的 Python 解释器, MCP 中的 Matplotlib/绘图支持` 


- **Docker 上的 Google Workspace MCP**：一位成员分享了他们在功能丰富的 [Google Workspace MCP](https://github.com/aaronsb/google-workspace-mcp) 上的进展，该项目支持多账号和令牌自动刷新（auto token refresh），并提供适用于不同平台的 Docker 镜像。
   - 该 MCP 提供了对 **Gmail**、**Calendar** 和其他 Google Workspace API 的集成访问。
- **Docker 化 MCP 服务器部署指南**：分享了一篇讨论如何部署 Docker 化 MCP 服务器的博客文章，强调了通过 Docker 封装方法面临的挑战和解决方案。
   - 文章指出了跨各种架构进行 **MCP Server 打包** 的复杂性，并指向了 GitHub 上的参考服务器。
- **Sage 与 Glama 的对齐**：讨论了 Glama 与 Sage 的集成，成员们表达了对协作以及让这两个项目更紧密对齐的兴趣。
   - 提到需要增加 API 以直接列出已安装的 MCP，但 Glama 在添加到 Sage 时确实可以工作。
- **Python 解释器的 MCP 集成**：一位成员介绍了他们的 [用于 MCP 的 Python REPL](https://github.com/evalstate/mcp-py-repl)，提供 STDIO 支持，并分享了它包含 **matplotlib**、**seaborn** 和 **numpy**。
   - 未来计划包括增加 **IPython** 支持，并对类似于 **Jupyter** 中的可视化表现出浓厚兴趣。
- **MCP 中的图像处理**：在关于图像处理的讨论中，注意到目前图像被保存为 **.png** 文件，且仅将文件名返回给 MCP 客户端。
   - 参与者讨论了在 MCP 服务器演示场景中，直接将文件作为工具结果返回的便利性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.defang.io/blog/2025/02/18/model-context-protocol">使用 Docker 和 Model Context Protocol 简化 AI 应用到云端的部署 | Defang</a>：mcp</li><li><a href="https://github.com/evalstate/mcp-py-repl">GitHub - evalstate/mcp-py-repl: 一个用于 MCP 的 Python REPL</a>：一个用于 MCP 的 Python REPL。通过在 GitHub 上创建账号来为 evalstate/mcp-py-repl 的开发做出贡献。</li><li><a href="https://github.com/aaronsb/google-workspace-mcp">GitHub - aaronsb/google-workspace-mcp: 一个提供 Google Workspace API 身份验证访问的 Model Context Protocol (MCP) 服务器，提供集成的身份验证、Gmail、Calendar 和 Drive 功能</a>：一个提供 Google Workspace API 身份验证访问的 Model Context Protocol (MCP) 服务器，提供集成的身份验证、Gmail、Calendar 和 Drive 功能 - aaronsb/google-work...
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1341457077100482570)** (2 条消息): 

> `供应商问卷应用, LlamaCloud EU 发布` 


- **应用简化供应商问卷流程**：一位成员分享了来自 @patrickrolsen 的创新 [全栈应用](https://twitter.com/llama_index/status/1891897812834816379)，该应用允许用户通过语义检索（semantically retrieving）之前的答案并使用 LLM 进行增强，从而回答供应商问卷。
   - 该应用通过简化阅读表单和填写答案的过程，展示了 **knowledge agents** 的核心用例。
- **LlamaCloud EU 为企业保障数据安全**：LlamaCloud EU 已宣布发布，为欧洲企业提供**安全**、合规的知识管理，重点关注 EU 管辖范围内的数据驻留（data residency），详情见[此处](https://twitter.com/llama_index/status/1892271183451869512)。
   - 这一早期访问产品旨在消除关注合规性和数据隐私的欧洲公司的重大障碍。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1341484612630872190)** (27 条消息🔥): 

> `公司指南 AI 聊天，AgentWorkflow 功能问题，QuadrantVectorStore 导入问题，区块链开发协作，Checkpoint 上下文序列化` 


- **用于公司指南的 AI 聊天机器人**：一位用户正在探索创建本地 AI 聊天以回答公司相关问题的方案，最初考虑使用 Llama 3.2 结合 RAG 或微调。
   - 目前，他们计划使用 VLLM 来实现 LLM 对巴西葡萄牙语的兼容性，并使用 LlamaIndex 在 Postgres 中管理数据。
- **AgentWorkflow 工具输出 Bug**：一位用户报告称，尽管生成了响应，但其 AgentWorkflow 的工具输出列表仍然为空，寻求实现方面的解答。
   - 另一位成员分享道，流式事件（streaming events）可以作为一种变通方法，用于捕获执行过程中的所有工具调用。
- **QuadrantVectorStore 导入问题**：一位用户无法从 `llama_index.vector_stores.qdrant` 导入 `QuadrantVectorStore`，对其安装方式感到困惑。
   - 在发现 Qdrant 向量存储需要单独安装后，他们在成功使用前解决了依赖冲突。
- **区块链开发协作**：一位拥有丰富区块链开发经验的用户正在寻求 DeFi、NFTs 和 Web3 开发领域的项目或协作。
   - 专长包括智能合约开发、DeFi 平台和区块链安全审计，重点关注保持行业趋势领先。
- **Checkpoint 上下文序列化挑战**：一位开发者在开发 human-in-the-loop AgentWorkflow 时面临上下文序列化问题，特别是在尝试持久化上下文状态时。
   - 尽管遵循了文档，但在加载 checkpoints 后，Agent 无法继续运行，一直卡在“思考中”状态。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/examples/agent/agent_workflow_multi/">使用 AgentWorkflow 的多 Agent 研究工作流 - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/understanding/agent/multi_agents/">多 Agent 工作流 - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/agent_workflow_basic/#tools-and-state">AgentWorkflow 基础介绍 - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/agent_workflow_basic/#human-in-the-loop">AgentWorkflow 基础介绍 - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1341920240250126366)** (1 条消息): 

> `AI 与数据运营趋势，Procure.FYI，联邦技术支出，企业级 AI 采用` 


- **AI 与数据运营的下一阶段**：一篇题为 [The End of Big Dumb AI & Data](https://open.substack.com/pub/procurefyi/p/the-end-of-big-dumb-ai-data?r=223ajc&utm_campaign=post&utm_medium=web&showWelcomeOnShare=false) 的近期文章讨论了挑战传统方法的 AI 和数据运营新兴趋势。
   - 文章强调了在处理数据以实现更好决策时，向更智能、更高效系统的转变。
- **Procure.FYI Substack 发布详情**：Procure.FYI 专注于联邦技术支出和企业级 AI 应用，于**两年前**发布。
   - 鼓励订阅者查看 Substack 的 [使用条款](https://substack.com/tos) 和 [隐私政策](https://substack.com/privacy)。



**提到的链接**：<a href="https://open.substack.com/pub/procurefyi/p/the-end-of-big-dumb-ai-data?r=223ajc&utm_campaign=post&utm_medium=web&showWelcomeOnShare=false">The End of Big, Dumb AI Data</a>：AI 数据运营进入智能阶段：为什么质量与策略现在胜过数量

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1341459314077859912)** (3 条消息): 

> `F24 MOOC 证书，高级课程证书` 


- **F24 MOOC 成就详情**：F24 MOOC 共有 **15,000** 名学生，其中包含 **304 位开拓者 (trailblazers)** 🚀，**160 位大师 (masters)** 🧑‍🏫，**90 位忍者 (ninjas)** 🥷，**11 位传奇 (legends)** 🏆，以及 **7 位荣誉获得者 (honorees)** 🌟。
   - 值得注意的是，三位荣誉获得者来自忍者级别，四位来自大师级别。
- **高级课程证书可用性**：成员询问是否有高级课程的证书，以及是否可以在没有 F24 MOOC 证书的情况下获得。
   - 回复确认这两个问题的答案均为**是**，更多细节将很快发布！

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1341458743128494195)** (18 条消息🔥): 

> `LangChain 框架, 基于 LLM 的机器学习预测, LLM Agents 课程内容, 评估 LLM 响应, LLM 方法论反馈` 


- **LangChain 简化 LLM 应用**：LangChain 是一个旨在简化 LLM 应用程序生命周期的框架，涵盖了[开发](https://python.langchain.com/docs/introduction/)、生产化和部署等各个环节的组件。
   - 一位成员指出，它的工作原理是将一个 LLM 的输出作为另一个 LLM 的输入进行链接，从而有效地创建链（chains）以增强性能。
- **关于 LLM 和机器学习获取的见解**：讨论了将 LLM Agents 与机器学习预测模型相结合的问题，并建议探索相关的学术论文以获得更深入的见解，例如在 [Everscope](https://everscope.ai/?models=Large+Language+Model+%28LLM%29&sort_by=elo_rating&sort_order=desc&page=1&limit=10&timeRange=this-week) 上找到的论文。
   - 另一位用户指出，按“所有时间（all-time）”进行筛选可能会获得与 LLM 相关的最佳论文。
- **关于课程内容的不确定性**：一位参与者询问是否有必要学习 2024 年的 LLM Agents 课程，并对本学期未涵盖的 DSPy 等主题表示感兴趣。
   - 回复指出，虽然先前的知识有助于理解当前主题，但访问 [2024 年课程材料](https://youtube.com/playlist?list=PLS01nW3RtgopsNLeM936V4TNSsvvVglLc&feature=shared) 可能并非必不可少。
- **使用 Meta Judges 评估 LLM 响应**：一位成员寻求关于 LLM Meta-judging 概念的不同幻灯片如何与生成和评估 LLM 响应相关的澄清，特别是第 89、97 和 98 页幻灯片之间。
   - 解释强调了在特定场景下进行训练和评估响应时，对于是否需要 Meta Judge 的方法差异。
- **课程材料消失**：有人对视频和测验突然消失表示担忧，这表明在跟上课程进度方面存在困难。
   - 强调了在适当频道提问的重要性，并对测验缺失的情况做了简要说明。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://everscope.ai/?models=Large+Language+Model+%28LLM%29&sort_by=elo_rating&sort_order=desc&page=1&limit=10&timeRange=this-week">Everscope - Quant Explorer by Cadenzai</a>: 未找到描述</li><li><a href="https://everscope.ai/?models=Large+Language+Model+%28LLM%29&sort_by=elo_rating&sort_order=desc&page=">Everscope - Quant Explorer by Cadenzai</a>: 未找到描述</li><li><a href="https://youtube.com/playlist?list=PLS01nW3RtgopsNLeM936V4TNSsvvVglLc&feature=shared,">LLM Agents MOOC</a>: Large Language Model Agents MOOC F24</li><li><a href="https://python.langchain.com/docs/introduction/">Introduction | 🦜️🔗 LangChain</a>: LangChain 是一个用于开发由大语言模型 (LLMs) 驱动的应用程序的框架。
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-readings-discussion](https://discord.com/channels/1280234300012494859/1282735578886181036/1341942110936498176)** (3 条消息): 

> `测验可用性, LLM Agents Hackathon, 课程结业, 2025 春季迭代, 视频讲座访问` 


- **MOOC 页面上的测验**：一位成员询问如何访问 **quiz1 和 quiz2** 以备周末学习，表达了在开始较晚后想要赶上进度的愿望。
   - 另一位成员回复说，可以在 [MOOC 页面](https://llmagents-learning.org/sp25) 上访问它们。
- **在公告中找到的测验**：进一步的回复建议在 [公告页面](https://llmagents-learning.org/f24) 上查找测验，并强调它们应该仍然可用。
   - 还强调了报名参加 [LLM Agents Hackathon](https://rdi.berkeley.edu/llm-agents-hackathon/) 的重要性，认为这是一个宝贵的机会。
- **宣布课程结业**：课程工作人员宣布当前课程已结束，但视频讲座在教学大纲中仍然可用。
   - 他们鼓励报名参加 [2025 春季迭代](https://llmagents-learning.org/sp25) 以继续学习。
- **证书已发放**：所有证书均已发放，标志着课程的结束。
   - **对这个精彩的学期表示了感谢**，确认了参与者的积极体验。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://llmagents-learning.org/sp25">Advanced Large Language Model Agents MOOC</a>: MOOC, 2025 春季</li><li><a href="https://llmagents-learning.org/f24">Large Language Model Agents MOOC</a>: MOOC, 2024 秋季
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1341549530575016037)** (5 messages): 

> `文本频道使用情况、Bot 自动化、截图请求` 


- **文本频道活跃度激增**：看到所有**文本频道**都被利用起来并不常见，一位成员微笑着对此发表了评论。
   - 该评论强调了各频道讨论量的意外激增。
- **Bot 自动化主导对话**：另一位参与者指出，观察到的垃圾信息通常来自**自动化 Bot**，而非人类活动。
   - 这表明自动化正在填补频道交流中的空白。
- **请求建立新的捕获频道**：一位成员请求创建一个特定频道，并指出该频道应支持**截图**。
   - 这表明社区内需要更多专门的空间来分享视觉内容。


  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1341928753395138601)** (3 messages): 

> `利润分成合作、没有咖啡的世界的影响` 


- **寻找利润分成合作伙伴**：一位成员正在寻找 **25-50 岁**的人士分享身份并就利润分成机会进行合作，预计利润在 **$100-1500** 之间。
   - 这一请求突显了创业活动中协作和互利的价值。
- **探索咖啡的影响**：有人请求写一篇**论文**，讨论没有咖啡的世界所产生的影响，强调其文化和经济重要性。
   - 该话题引发了关于如果没有这种流行饮品可能出现的假设情景和社会变革的讨论。


  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1341921219322318903)** (11 messages🔥): 

> `身份分享担忧、沟通清晰度、协作与盗窃之争、协作中的隐私` 


- **身份分享引发辩论**：一位用户表示有兴趣与 **25-50 岁**的人士合作并分享利润，但其他人立即对在公共论坛分享个人身份信息表示担忧。
   - *“现在的身份盗用都这么公开了吗？”* 一位用户质疑道，这突显了在没有明确目的的情况下，一些人对在线分享私人细节的犹豫。
- **清晰沟通的重要性**：一位成员建议，在分享观点时，清晰的表达至关重要，以防止误解，尤其是在基于文本的媒介中。
   - 他们强调，改善**积极沟通**可以带来惊人的协作，并敦促大家团结起来。
- **协作还是盗窃？大辩论**：回复中强调了在将身份分享请求视为**协作**还是潜在的**盗窃**形式之间的分歧。
   - 一位用户坚定地表示：*“那是协作……”*，试图澄清请求背后的意图。
- **令人怀疑的信息缺失**：人们对发帖人**缺乏细节**提供表示担忧，包括缺乏与其提议相关的网站或文档。
   - 一位用户形容整个风险项目*“令人怀疑”*，并指出透明度对于谨慎的协作至关重要。


  

---

### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1341832890568282164)** (16 messages🔥): 

> `AI21 API usage, Output formatting issues, Handling special characters in responses, Working with Symfony and PHP` 


- **寻求 AI21 API 集成帮助**：一位用户寻求关于使用 **AI21 API** 的帮助，特别是如何向 `jamba-1.5-large` 模型格式化请求以及如何正确构建 API 调用。
   - 热心成员提供了详细的 [API reference](https://docs.ai21.com/reference/jamba-15-api-ref) 以及成功请求所需的 Header 和参数示例。
- **澄清 API 输出行为**：一位用户询问在尝试解决数学表达式时 API 的输出，收到了出乎意料的详细响应。
   - 另一位成员澄清说，输出格式包含转义字符；需要在代码中进行调整，在显示之前对输出进行清理。
- **API 响应中的特殊字符**：讨论了在使用 **PHP** 配合 AI21 API 时，是否有必要从 API 响应中移除特殊字符。
   - 成员们确认 **AI21 Studio** UI 旨在处理这些字符，但为了获得正确的输出格式，仍需要额外的代码处理。
- **PHP 集成挑战**：一位用户提到他们在配合 **Symfony** 和 PHP 处理 API 响应时面临的挑战，表示可能需要进行大量的转换。
   - 他们还感谢了其他成员对有效处理和格式化 API 输出所提供的见解。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.ai21.com/reference/jamba-15-api-ref">Jamba 1.5</a>: Jamba-1.5 指令遵循对话模型</li><li><a href="https://docs.ai21.com/reference">Jamba 1.5</a>: Jamba-1.5 指令遵循对话模型
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1341488861301444749)** (3 messages): 

> `Self-Supervised Prompt Optimization, Retrieval Augmented Generation, Real-time Information Integration, LLM Performance Improvement` 


- **引入 Self-Supervised Prompt Optimization 框架**：一种名为 **Self-Supervised Prompt Optimization (SPO)** 的新方法旨在为封闭式和开放式任务发现有效的 Prompt，而无需外部参考，使其具有成本效益。该框架纯粹从输出比较中得出评估信号，从而增强 **LLM 推理能力**。
   - *一位参与者指出，这篇论文直到最后一段才提到 DSPy，这太疯狂了。*
- **通过 Retrieval Augmented Generation 进行动态内容整合**：该研究提出了一种传统检索方法的替代方案，通过使用**标准搜索引擎 API** 在生成推理过程中动态整合最新的在线信息。这种方法旨在提高生成内容的质量，而不依赖于固定的索引。
   - 这种新范式涉及一个 **parser-LLM**，它在单次推理中确定是否需要互联网增强并提取搜索关键词。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.06855">Self-Supervised Prompt Optimization</a>: 精心设计的 Prompt 对于增强大语言模型 (LLMs) 的推理能力，同时使其输出与跨不同领域的任务要求保持一致至关重要。然而，手动设计...</li><li><a href="https://arxiv.org/abs/2411.19478">Zero-Indexing Internet Search Augmented Generation for Large Language Models</a>: Retrieval augmented generation 已成为增强大语言模型性能的有效方法。这种方法通常依赖于使用各种索引的内部检索模块...
</li>
</ul>

</div>
  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1341527810707034122)** (11 messages🔥): 

> `Synthetic Data Generation with DSPy, Judge-Time Scaling in AI, Conversation History in DSPy Calls, Freezing and Exporting Prompts, Personal Voice Identity Manager` 


- **DSPy 合成数据生成探索**：一名成员分享了一个 [GitHub 仓库链接](https://github.com/seanchatmangpt/dslmodel)，展示了来自 DSPy 和 Jinja2 的结构化输出，强调了其在合成数据生成方面的能力。
   - 此外，他们还重点介绍了一个合成数据生成流水线，可用于训练 ChatDoctor，并附带了一张图片进行说明。
- **通过 Judge-Time Scaling 增强 AI 评估**：一名成员对名为 Verdict 的新库表示兴奋，该库专注于扩展 Judge-time compute（评测时计算），正如 [Leonard Tang](https://x.com/leonardtang_/status/1892243653071908949) 所分享的那样。
   - 该版本的发布旨在解决 AI 评估的局限性，特别是在开放式和不可验证的领域，标志着 AI 进步的重大转变。
- **验证 DSPy 中的对话历史注入**：一位用户询问 DSPy 是否会自动将对话历史注入到调用中，希望在继续探索之前得到明确答复。
   - 另一名成员确认了他们对该话题的兴趣，并暗示希望避免潜在的复杂性。
- **从 DSPy 程序中冻结并导出 Prompt**：一名成员分享了一段代码片段，展示了如何使用默认 Adapter 将 DSPy 程序中的所有 Prompt 冻结并导出为消息模板。
   - 他们指出，虽然这种方法很方便，但可能会导致丢失控制流逻辑，并建议使用 `program.save()` 等替代方案。
- **个人语音身份管理器（Personal Voice Identity Manager）概念讨论**：一名成员表示，新公布的 Judge-time scaling 库将非常适合他们构思的个人语音身份管理器。
   - 这暗示了该库的实际应用场景，突显了其在 AI 框架内对个人身份管理的潜在影响。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/leonardtang_/status/1892243653071908949">Leonard Tang (@leonardtang_) 的推文</a>：首先是 Pre-training scaling；接着是 Inference-time scaling。现在轮到 Judge-time scaling 了。尽管 AI 通过扩展 Inference-time compute 取得了进步，但在开放式、非验证领域，AI 仍然不可靠...</li><li><a href="https://x.com/DataDeLaurier/status/1891896292650991810">Dᴀᴛᴀ Sᴀᴄᴋs (@DataDeLaurier) 的推文</a>：@Teknium1 因为他们没有任何新东西。他们正准备把 4o, o1, o3, Voice 和 Sora 塞进 RouteLLM 并称之为 GPT-5。我敢打赌他们实际上使用了 RouteLLM 却不引用任何人</li><li><a href="https://github.com/seanchatmangpt/dslmodel">GitHub - seanchatmangpt/dslmodel: 来自 DSPy 和 Jinja2 的结构化输出</a>：来自 DSPy 和 Jinja2 的结构化输出。通过在 GitHub 上创建账号来为 seanchatmangpt/dslmodel 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1341582787106312264)** (4 messages): 

> `Model Testing, Performance Comparison, Computational Complexity` 


- **模型性能的参差体验**：一位用户分享说，在 Windows 10 上使用 **10 年前的 GeForce 850M** 进行测试，最初达到 3 tok/s，但随后逐渐下降，而 Chrome 浏览器加载需要 2 分钟，速度仅为 0.2 tok/s。
   - 相比之下，另一位用户报告在 Windows 11 上使用 **RTX 4070**，达到了更快的 **12 tok/s**，首个 Token 响应时间（Time to first token）为 1.9 秒。
- **对模型可用性的担忧**：一位用户强调，尽管 RTX 4070 性能尚可，但由于高昂的计算成本和复杂性，该模型仍然不太实用。
   - 他们提到了具体问题，如 **数值刚性（Numerical stiffness）** 以及非线性子问题带来的挑战，这些都增加了获取准确解的难度。


  

---


---


{% else %}


> 完整的频道逐项分析已针对邮件进行删减。
> 
> 如果您想查看完整分析，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}