---
companies:
- openai
- anthropic
- mistral-ai
- langchain
- hugging-face
date: '2024-04-12T23:27:50.881515Z'
description: '**GPT-4 Turbo** 凭借在编程、多语言及纯英语任务上的显著提升，重新夺回了排行榜榜首，目前已在付费版 **ChatGPT**
  中上线。尽管如此，**Claude Opus** 在创造力和智能方面依然更胜一筹。**Mistral AI** 发布了如 **Mixtral-8x22B** 和
  **Zephyr 141B** 等强大的开源模型，非常适合进行微调。**LangChain** 增强了跨模型的工具集成能力，而 **Hugging Face**
  推出了用于在浏览器中运行 Transformer 模型的 Transformer.js。专注于医学领域的 **Medical mT5** 作为一个开源多语言文本到文本模型被分享。此外，社区还重点关注了关于将大语言模型（LLM）作为回归器的研究，并分享了来自
  **Vik Paruchuri** 在 OCR/PDF 数据建模方面的实践经验。'
id: 36826f16-41ad-4a33-8a5e-31b025f58f95
models:
- gpt-4-turbo
- claude-3-opus
- mixtral-8x22b
- zephyr-141b
- medical-mt5
original_slug: ainews-zero-to-gpt-in-1-year
people:
- vik-paruchuri
- sam-altman
- greg-brockman
- miranda-murati
- abacaj
- mbusigin
- akhaliq
- clementdelangue
title: '**从零到 GPT：一年进阶之路**


  （也可以翻译为：**一年时间，从零基础到掌握 GPT**）'
topics:
- fine-tuning
- multilinguality
- tool-integration
- transformers
- model-evaluation
- open-source-models
- multimodal-llms
- natural-language-processing
- ocr
- model-training
---

 

这些基础知识可能比紧跟每日新闻更有价值，我们喜欢尽可能地分享这类高质量的建议。


---

**目录**

[TOC] 


---

# AI Reddit 摘要

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence。评论抓取功能现已上线，但仍有很大改进空间！

待完成

---

# AI Twitter 摘要

> 所有摘要均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在尝试使用 Haiku 进行聚类和流程工程（flow engineering）。

**GPT-4 和 Claude 更新**

- **GPT-4 Turbo 重夺排行榜首位**：[@lmsysorg](https://twitter.com/lmsysorg/status/1778555678174663100) 指出 GPT-4-Turbo 已重新夺回 Arena 排行榜第一名，在 Coding（编程）、Longer Query（长查询）和 Multilingual（多语言）等多个领域表现优于其他模型。它在纯英文提示词和包含代码片段的对话中表现更为强劲。
- **发布新版 GPT-4 Turbo 模型**：[@sama](https://twitter.com/sama/status/1778578689984270543) 和 [@gdb](https://twitter.com/gdb/status/1778577748421644459) 宣布在 ChatGPT 中发布了新的 GPT-4 Turbo 模型，该模型显著更智能且使用体验更佳。[@miramurati](https://twitter.com/miramurati/status/1778582115460043075) 确认这是最新的 GPT-4 Turbo 版本。
- **新版 GPT-4 Turbo 的评估数据**：[@polynoamial](https://twitter.com/polynoamial/status/1778584064343388179) 和 [@owencm](https://twitter.com/owencm/status/1778619341833121902) 分享了评估数据，与之前的版本相比，MATH 提升了 +8.9%，GPQA 提升了 +7.9%，MGSM 提升了 +4.5%，DROP 提升了 +4.5%，MMLU 提升了 +1.3%，HumanEval 提升了 +1.6%。
- **Claude Opus 在某些方面仍优于新版 GPT-4**：[@abacaj](https://twitter.com/abacaj/status/1778435698795622516) 和 [@mbusigin](https://twitter.com/mbusigin/status/1778813997246034254) 指出，在他们的使用中，Claude Opus 仍然优于新的 GPT-4 Turbo 模型，表现得更智能且更具创造力。

**开源模型与框架**

- **Mistral 模型**：[@MistralAI](https://twitter.com/MistralAI) 发布了新的开源模型，包括 Mixtral-8x22B 基础模型，它是微调的利器 ([@_lewtun](https://twitter.com/_lewtun/status/1778429536264188214))，以及 Zephyr 141B 模型 ([@osanseviero](https://twitter.com/osanseviero/status/1778430866718421198), [@osanseviero](https://twitter.com/osanseviero/status/1778816205727424884))。
- **Medical mT5 模型**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1778607598784135261) 分享了 Medical mT5，这是一个针对医学领域的开源多语言 text-to-text LLM。
- **LangChain 和 Hugging Face 集成**：[@LangChainAI](https://twitter.com/LangChainAI/status/1778465775034249625) 发布了更新以支持跨模型提供商的 tool calling（工具调用），以及一个用于将工具附加到模型的标准 `bind_tools` 方法。[@LangChainAI](https://twitter.com/LangChainAI/status/1778533665645134280) 还更新了 LangSmith，以支持在各种模型的 trace（追踪）中渲染 Tools 和 Tool Calls。
- **Hugging Face Transformer.js**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1778456263161971172) 注意到 Transformer.js（一个在浏览器中运行 Transformer 的框架）登上了 Hacker News。

**研究与技术**

- **从文字到数字 - LLMs 作为回归器**：[@_akhaliq](https://twitter.com/_akhaliq/status/1778592009067925649) 分享了一项研究，分析了预训练的 LLMs 在给定 in-context 示例时执行线性及非线性回归的表现，其效果可媲美或超越传统的监督学习方法。
- **高效的无限上下文 Transformers**：[@_akhaliq](https://twitter.com/_akhaliq/status/1778605019362632077) 分享了来自 Google 的一篇论文，该论文将压缩内存集成到原生的 attention 层中，使 Transformer LLMs 能够以有限的内存和计算量处理无限长的输入。
- **OSWorld 基准测试**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1778599140634599721) 和 [@_akhaliq](https://twitter.com/_akhaliq/status/1778605020444795284) 分享了 OSWorld，这是首个面向多模态 Agent 的可扩展真实计算机环境基准测试，支持在各种操作系统上进行任务设置、基于执行的评估和交互式学习。
- **ControlNet++**：[@_akhaliq](https://twitter.com/_akhaliq/status/1778606395014676821) 分享了 ControlNet++，它通过高效的一致性反馈改进了 Diffusion 模型中的条件控制。
- **在有限区间内应用引导**：[@_akhaliq](https://twitter.com/_akhaliq/status/1778607531998232926) 分享了一篇论文，表明在有限区间内应用引导可以提高 Diffusion 模型中的样本和分布质量。

**行业新闻与观点**

- **WhatsApp 与 iMessage 之争**：[@ylecun](https://twitter.com/ylecun/status/1778745216842760502) 将 WhatsApp 与 iMessage 的争论比作公制与英制之争，指出全世界都在使用 WhatsApp，除了部分紧抱 iPhone 的美国人或禁用该应用的国家。
- **AI Agents 将无处不在**：[@bindureddy](https://twitter.com/bindureddy/status/1778508892382884265) 预测 AI Agents 将无处不在，通过 Abacus AI，你可以让 AI 在 5 分钟到几小时的简单过程中构建这些 Agents。
- **Cohere Rerank 3 模型**：[@cohere](https://twitter.com/cohere/status/1778417650432971225) 和 [@aidangomez](https://twitter.com/aidangomez/status/1778416325628424339) 推出了 Rerank 3，这是一个用于增强企业搜索和 RAG 系统的基座模型，能够以 100 多种语言精确检索多维度和半结构化数据。
- **Anthropic 因信息泄露解雇员工**：[@bindureddy](https://twitter.com/bindureddy/status/1778546797331521581) 报道称 Anthropic 解雇了 2 名员工，其中一人是 Ilya Sutskever 的好友，原因是泄露了一个内部项目的信息，该项目可能与 GPT-4 有关。

**梗与幽默**

- **关于 LLM 模型名称的梗**：[@far__el](https://twitter.com/far__el/status/1778736813714137342) 调侃了复杂的模型名称，如 "MoE-8X2A-100BP-25BAP-IA0C-6LM-4MCX-BELT-RLMF-Q32KM"。
- **关于 AI 个人助手模式的梗**：[@jxnlco](https://twitter.com/jxnlco/status/1778509125137072525) 调侃称，每家公司的 AI 个人助手模式都有两种——哲学家和集成地狱，并将其比作认识论与身份验证错误（auth errors）。
- **关于 LLM 幻觉的笑话**：[@lateinteraction](https://twitter.com/lateinteraction/status/1778844352334508140) 调侃道，他们担心一旦人们意识到 AGI 并不遥远，且目前还没有可靠的通用 LLMs 或 "Agents"，泡沫就会破裂；并建议更明智的做法是认识到 LLMs 主要为构建解决特定任务的 AI 提供了取得普遍进展的机会。

---

# AI Discord 摘要

> 摘要之摘要的摘要

- **Mixtral 和 Mistral 模型受到关注**：**Mixtral-8x22B** 和 **Mistral-22B-v0.1** 模型引发了热议，后者标志着首次成功将 Mixture of Experts (MoE) 模型转换为稠密（dense）格式。讨论围绕它们的能力展开，例如 Mistral-22B-v0.1 的 220 亿参数。新发布的 **[Zephyr 141B-A35B](https://huggingface.co/HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1)** 是 Mixtral-8x22B 的微调版本，也引起了人们的兴趣。

- **Rerank 3 与 Cohere 的搜索增强**：**[Rerank 3](https://txt.cohere.com/rerank-3/)** 是 Cohere 为企业搜索和 RAG 系统推出的新基础模型，支持 100 多种语言，拥有 4k 上下文长度（context length），并提供高达 3 倍的推理速度提升。它与 **Elastic 的 Inference API** 原生集成，为企业搜索提供动力。

- **CUDA 优化与量化探索**：工程师们正在优化 `CublasLinear` 等 **CUDA 库**以实现更快的模型推理，同时讨论 4-bit、8-bit 等量化策略以及 **High Quality Quantization (HQQ)** 等新方法。通过修改 NVIDIA 驱动程序，实现了 **[4090 GPU 上的 P2P 支持](https://x.com/__tinygrad__/status/1778676746378002712)**，从而带来了显著的加速。

- **Scaling Laws 与数据过滤研究发现**：一篇新论文 **["Scaling Laws for Data Filtering"](https://arxiv.org/abs/2404.07177)** 认为，数据清洗（data curation）不能脱离计算资源而独立存在，并引入了处理异构网络数据的 Scaling Laws。社区正在思考其影响并试图理解所采用的实证方法。

其他值得关注的讨论包括：

- **GPT-4 Turbo** 的发布及其在 Coding 和 Reasoning 任务上的表现
- **Ella** 在动漫图像生成方面的表现不佳
- 对 **Stable Diffusion 3** 的期待及其解决当前模型局限性的潜力
- **Hugging Face** 的 Rerank 模型下载量突破 23 万，以及 **parler-tts** 库的发布
- **OpenAI API** 关于 Wolfram 集成和 Prompt Engineering 资源的讨论

---

# 第一部分：Discord 高层级摘要

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

**砥砺前行，不落后于 A1111**：**Forge** 是 **Automatic1111** 的一个新分支，具有性能增强功能，正广受好评。爱好者们可以在不放弃 A1111 的情况下探索 Forge，并利用 ComfyUI 模型实现更高效的工作流。

**Ella 在动漫艺术方面表现不佳**：使用 **Ella** 进行动漫风格图像生成的实验令人失望，即使使用了推荐的 Checkpoints 也未能达到用户预期。尽管寄予厚望，但 Ella 生成的动漫图像质量仍然较低，被认为无法在该领域使用。

**Stable Diffusion 3 带来希望与疑虑**：社区对 **Stable Diffusion 3 (SD3)** 充满了期待与怀疑，特别是关于它是否有潜力克服当前模型的局限性，如虚化效果（bokeh effects）、色彩保真度和名人识别。

**扩展图像完美工具箱**：讨论中提到了几种增强 Stable Diffusion 输出的工具和扩展，包括用于 Outpainting 的 **BrushNet**，以及改进建筑领域 **depth-fm** 和 **geowizard** 的解决方案，还有一个色彩校正扩展。

**Cascade 因快速学习而闻名**：**Cascade** 在 Stable Diffusion 模型中因其快速的学习能力和独特的特性而脱颖而出，尽管它的学习曲线较陡峭，被亲切地称为“SD 家族的怪表弟”。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

**CORS 导致 Cohere 连接崩溃**：用户遇到了 CORS 策略错误，导致无法访问 **Cohere dashboard**，问题源于从 `https://dashboard.cohere.com` 到 `https://production.api.os.cohere.ai` 的跨域 JavaScript fetch 请求。

**关于上下文长度的争论**：一场关于 **large language models (LLMs)** 中扩展上下文长度与 **Retrieval-Augmented Generation (RAG)** 有效性的激烈讨论展开，辩论焦点在于计算成本和长上下文收益递减的问题。

**Rerank 3 的定价与促销**：**Rerank V3** 已发布，定价为每 1k 次搜索 2 美元，并提供 50% 的首发促销折扣。对于需要旧版本的用户，**Rerank V2** 仍以每 1k 次搜索 1 美元的价格提供。

**Cohere 微调与部署指南**：出现了关于 **Cohere LLMs** 的本地化（on-premise）和基于平台的微调可能性的问题，以及在 **AWS Bedrock** 或类似本地场景下的部署选项。

**Rerank 3 增强搜索概览**：**Rerank 3** 推出以增强企业搜索，声称推理速度提高了三倍，并凭借其扩展的 4k 上下文支持 100 多种语言。它与 **Elastic's Inference API** 集成以改进企业搜索功能，并提供了 [Cohere-Elastic 集成指南](https://docs.cohere.com/docs/elasticsearch-and-cohere)和实用的 [notebook 示例](https://github.com/cohere-ai/notebooks/blob/main/notebooks/Cohere_Elastic_Guide.ipynb)等资源。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**Ghost 7B 精通多语言**：新的 **Ghost 7B** 模型因其在越南语推理和理解方面的出色表现而引起轰动，受到 AI 社区的热切期待。它被强调为一个更紧凑的多语言替代方案，可以满足专业知识需求。

**微调挑战的双重审视**：讨论中提到了微调 NLP 模型的困难，指出充满希望的训练评估与令人失望的实际推理性能之间存在差距。特别是，非英语 NLP 环境中缺乏准确性一直是工程师们感到沮丧的点。

**寻求高效的模型部署策略**：工程师们正在积极分享策略和[资源](https://github.com/unslothai/unsloth)，以简化 **Mistral-7B** 训练后的部署。对 VRAM 限制的担忧依然存在，促使了关于优化 batch sizes 和嵌入上下文 token 以节省内存的讨论。

**Unsloth AI 倡导扩展上下文窗口**：Unsloth AI 框架因减少 **30%** 的内存使用且仅增加 **1.9%** 的时间开销，同时支持长达 **228K** 的上下文窗口微调而受到赞誉，[其博客中详细介绍了这一点](https://unsloth.ai/blog/long-context)。与之前的基准测试相比，这是一个重大飞跃，为 LLM 开发提供了新途径。

**领域特定数据的重要性**：大家一致认为需要更精确的领域特定数据集，因为通用数据收集对于需要详细上下文的专业模型来说是不够的。最佳实践仍在讨论中，许多人转向 Hugging Face 等平台寻求高级数据集解决方案。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **RNN 复兴指日可待**：一篇发表在 [arXiv](https://arxiv.org/abs/2402.19427) 上的新论文表明，一种新兴的混合架构可能会为用于序列数据处理的循环神经网络 (RNNs) 注入新的活力。据报道，Google 投资了一个新的 **70 亿参数的基于 RNN 的模型**，这激发了社区对未来应用的兴趣。

- **Google 凭借 Gemma 引擎进军 C++**：社区注意到 Google 为其 **Gemma** 模型发布了一个 C++ 推理引擎，引发了广泛好奇。该独立引擎是开源的，可以通过其 [GitHub 仓库](https://github.com/google/gemma.cpp)访问。

- **微调 Hermes 需要雄厚的财力**：微调 **Nous Hermes 8x22b** 似乎非常耗钱，据传需要每周大约 "$80k" 的基础设施成本。详细的基础设施细节尚未披露，但显然，这并非易事。

- **全力挖掘 Apple AI 的潜力**：工程师们正密切关注 Apple 的 **M 系列芯片**，期待 *M4 芯片* 及其传闻中的 2TB RAM 支持。**M2 Ultra** 和 **M3 Max** 的 AI 推理能力，尤其是它们的低功耗，赢得了广泛赞誉。

- **LLMs 在医疗领域备受关注但需谨慎**：在医疗领域使用大语言模型 (LLMs) 的影响在社区中引发了兴奋与担忧交织的情绪。有讨论指出，法律风险和人为限制阻碍了其在医疗保健领域的开发和应用。



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Cublas 线性优化**：自定义 `CublasLinear` 库优化正在加速大矩阵乘法的模型推理，尽管 Attention 机制中的瓶颈可能会削弱 "llama7b" 等模型的整体性能提升。
  
- **通过 P2P 实现巅峰性能**：通过修改 NVIDIA 驱动，在 4090 GPU 上实现了所有 AllReduce 操作 58% 的加速。该修改实现了 14.7 GB/s 的 AllReduce，是增强支持 P2P 的 [tinygrad 性能](https://x.com/__tinygrad__/status/1778676746378002712)的重要一步。

- **攻克量化目标**：围绕量化（如 4-bit 方法）的挑战和策略正受到关注，一种新的 HQQ (High Quality Quantization) 方法正在被讨论，以实现更优的反量化线性度。在张量计算中，发现 8-bit 矩阵乘法的速度是 fp16 的两倍，这突显了 `int4 kernels` 可能存在的性能问题。

- **速度突破与 CUDA 进展**：`A4000` GPU 通过 `float4` 加载实现了 375.7 GB/s 的最大吞吐量，表明了 L1 cache 的高效利用。同时，CUDA 的最新特性（如协作组 cooperative groups 和 kernel fusion）正在推动性能提升和现代 C++ 的采用，以提高可维护性。

- **社区资源共享与组织**：成员们建立了共享 CUDA 资料的频道，例如重命名现有频道用于资源分发，并建议组织任务以优化工作流。一个 PMPP UI 学习小组已经启动，欢迎通过 [Discord 邀请](https://discord.gg/XwFJRKH9)加入。

- **概念解释与学术贡献**：分享了一篇关于 [Ring Attention](https://coconut-mode.com/posts/ring-attention/) 的解释文章，旨在扩展 LLMs 的上下文窗口，并征求反馈。在学术方面，一本以 GPU 为中心的数值线性代数书籍的第 4 章正在编写中，而 Golub/Van Loan 书籍的现代 CUDA 版本也深入探讨了相关领域，深化了知识储备。一个包含 GPU 编程在内的并行计算机编程实践课程已[在线开放](https://ppc-exercises.cs.aalto.fi/courses)。



---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **GPT-4 在 Perplexity 中的热度**：工程师们对 **GPT-4** 在 **Perplexity** 中的**集成**感到好奇，询问其功能和 API 可用性。同时，一些用户讨论了 Perplexity 在传统搜索之外的能力，建议将其定位为搜索和图像生成的复合工具。
  
- **扩展 API 产品**：一场热烈的对话探讨了将 **Perplexity API** 集成到电子商务中，并引导用户查阅 [文档](https://docs.perplexity.ai/) 以获取指导。然而，关于 API 中是否提供 Pro Search 功能的咨询得到了明确的否定回答。

- **编写完美的扩展程序**：技术讨论集中在通过浏览器扩展增强 Perplexity 的实用性，尽管客户端抓取（client-side fetching）带来了一些限制。GitHub 上的 [RepoToText](https://github.com/JeremiahPetersen/RepoToText) 等工具被提及作为将 LLM 与仓库内容结合的资源。

- **搜索轨迹与技术轨迹**：用户积极分享了 **Perplexity AI** 的搜索链接，标志着在该平台上扩大协作的趋势。搜索内容涵盖了从不明物体到访问日志和 NIST 标准等深度技术问题，反映了人群的多样化兴趣。

- **期待路线图的实现**：用户关注着 Perplexity 的未来，有人寻求引用功能的更新，并参考 [功能路线图](https://docs.perplexity.ai/docs/feature-roadmap) 来明确即将推出的增强功能。路线图似乎计划了多个延伸至 6 月的更新，尽管目前尚未提及备受期待的来源引用（source citations）。



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**量化探索继续**：**Mixtral-8x22B** 模型现已完成量化并可供下载，但它尚未经过微调（fine-tuned），可能会对无法处理 8x7b 版本的系统产生挑战。模型加载错误可以通过升级到 **LM Studio 0.2.19 beta preview 3** 来解决。

**应对大模型困境**：用户分享了在配置不足的硬件上运行大模型的经验，建议使用云解决方案或进行硬件升级，如 **NVIDIA 4060ti 16GB**。对于处理时间序列数据的用户，建议使用 **Temporal Fusion Transformer (TFT)**，认为它非常适合该任务。

**GPU vs. CPU：性能之谜**：运行 AI 模型时，更多的系统内存有助于加载更大的 LLM，但使用像 **NVIDIA RTX A6000** 这样的显卡进行全 GPU 推理（full GPU inference）才是获得最佳性能的关键。

**Linux 中新兴的 ROCm 谜团**：对 **amd-rocm-tech-preview** 支持感到好奇的 Linux 用户仍处于等待状态，而拥有 7800XT 等兼容硬件的用户报告在执行任务时出现电感啸叫（coil whine）。同时，为 Windows 构建 `gguf-split` 二进制文件是 AMD 硬件测试的一个障碍，需要查阅 GitHub 讨论和 Pull Requests 以获取指导。

**BERT 的边界与 Embedding 利用**：如果没有针对特定任务的微调，**Google BERT models** 通常无法直接在 LM Studio 中使用。对于利用 LM Studio 进行文本嵌入（text embeddings），推荐使用参数量更大的模型，如 `mxbai-large` 和 `GIST-large`，而非标准的 BERT base 模型。

请注意，虽然此摘要内容详尽，但特定频道可能包含更多与 AI 工程师相关的详细讨论和链接。



---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**BERT 的双向难题**：工程师们提出了扩展 BERT 等 **encoder models** 上下文窗口的复杂性，提到了双向机制的困难，并指向了应用 [FlashAttention](https://github.com/Dao-AILab/flash-attention) 的 [MosaicBERT](https://mosaicbert.github.io/)，同时质疑尽管有[贡献](https://github.com/huggingface/transformers/issues/26350)，但为何在流行库中仍缺失该功能。

**用 Google 的 Mixture-of-Depths 模型重新思考 Transformer**：研究人员正在讨论 Google 创新的 **Mixture-of-Depths** 方法，该方法在基于 Transformer 的模型中以不同方式分配计算资源。同样引起关注的是 **RULER** 新开源但最初为空的仓库 [此处](https://github.com/hsiehjackson/RULER)，旨在揭示长上下文语言模型的真实上下文大小。

**明智地扩展数据山**：分享了一篇[论文](https://arxiv.org/abs/2404.07177)，提出 **data curation** 是必不可少的，且不能忽视计算限制。讨论包括在 scaling laws 中对基于熵的方法的符号搜索，以及对基础研究原则的反思。

**大语言模型中的古怪行为困扰分析师**：成员们对 **NeoX 的 embedding layer** 行为表示好奇，质疑训练期间是否省略了 weight decay。他们将 NeoX 的输出与其他模型进行了对比，并确认了独特行为，引发了对技术细节和影响的好奇。

**量化探索与数据集困境**：社区努力包括尝试进行 2-bit quantization 以减少 Mixtral-8x22B 模型的 VRAM 占用，同时对 The Pile 数据集不一致的大小以及缺乏针对各种存档类型的提取代码感到困惑。



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Mixtral 的扩张与收缩**：发布了一个名为 **Mixtral 8x22B:free** 的新模型，增强了路由和速率限制的清晰度，并拥有更新至 65,536 的 context size。然而，它很快被[禁用](https://openrouter.ai/models/mistralai/mixtral-8x22b:free)，促使用户转向其目前活跃的对应版本 **Mixtral 8x22B**。

- **新实验模型登场**：社区有两个新的实验模型可供试用：[Zephyr 141B-A35B](https://openrouter.ai/models/huggingfaceh4/zephyr-orpo-141b-a35b)（Mixtral 8x22B 的指令微调版本）和 [Fireworks: Mixtral-8x22B Instruct (preview)](https://openrouter.ai/models/fireworks/mixtral-8x22b-instruct-preview)，为 AI 领域增添了色彩。

- **购买流程中的障碍**：寻求 token 的购买者遇到了小故障，触发了快照分享，并据推测是要求解决交易流程中的问题。

- **平台困境的自救**：一名陷入登录困境的用户发现了一种自救策略，巧妙地完成了账号注销。

- **Turbo 故障与个人 AI 愿景**：讨论范围从通过 Heroku 重新部署来解决 **GPT-4 Turbo** 故障，到定制交织了 [LibreChat](https://github.com/danny-avila/LibreChat) 等工具的 AI 设置。深入探讨 AI 模型的特性和调优平衡点也是热门话题，其中 *Opus*、*Gemini Pro 1.5* 和 MoE 结构备受关注。



---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 的社区代码贡献**：服务器成员对 **Mojo** 开源其标准库表示赞赏，这促进了社区的贡献和增强。讨论围绕着将 **Modular** 集成到 [BackdropBuild.com](https://backdropbuild.com) 的开发者队列项目中展开，但成员们也被提醒将业务咨询保留在适当的频道中。
  
- **Karpathy 关注 Mojo 移植版**：由 Andrej Karpathy 的 `llm.c` 仓库中的 [GitHub issue #28](https://github.com/karpathy/llm.c/issues/28) 引发的一场激动人心的讨论，重点关注 **Mojo** 移植版的基准测试和对比前景，作者本人也表示有兴趣链接到任何由 **Mojo** 驱动的版本。

- **行与列：矩阵大对决**：[Modular 博客](https://www.modular.com/blog/row-major-vs-column-major-matrices-a-performance-analysis-in-mojo-and-numpy)上的一篇资讯文章详细分析了行优先（row-major）与列优先（column-major）矩阵存储及其在 **Mojo** 和 NumPy 中的性能表现，启发了社区对编程语言和库的存储偏好的理解。

- **Mojo 的终端时尚**：成员们展示了使用 **Mojo** 在终端中进行的高级文本渲染，演示了受 `charmbracelet's lipgloss` 启发的功能和界面。分享了代码片段和实现示例，预览可在 [GitHub](https://github.com/thatstoasty/mog/blob/main/examples/readme/layout.mojo) 上查看。

- **矩阵博客的小失误：寻求帮助**：一位成员指出在执行 [“行优先 vs. 列优先矩阵”](https://www.modular.com/blog/row-major-vs-column-major-matrices-a-performance-analysis-in-mojo-and-numpy) 博客文章中的 Jupyter notebook 时出现错误，遇到了 'mm_col_major' 声明的问题。这一反馈为社区支持的调试提供了机会，该 notebook 位于 [devrel-extras GitHub 仓库](https://github.com/modularml/devrel-extras/blob/main/blogs/mojo-row-major-column-major/row_col_mojo.ipynb)。



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain 的 PDF 摘要速度提升**：强调了一种提高 LangChain 的 `load_summarization_chain` 函数在处理大型 PDF 文档时摘要效率的方法，[GitHub](https://github.com/langchain-ai/langchain/issues/12336) 上提供了一个演示 `map_reduce` 优化方法的代码片段。

- **LangChain AI 推出新教程**：最近推出的一项教程阐明了 LCEL 以及使用 runnables 组装 chain 的方法，为工程师提供动手学习的机会并征求反馈；详见 [Medium](https://medium.com/@klcoder/langchain-tutorial-lcel-and-composing-chains-from-runnables-751090a0720c?sk=55c60f03fb95bdcc10eb24ce0f9a6ea7)。

- **GalaxyAI API 正式发布**：GalaxyAI 首次推出了与 Langchain 无缝衔接的免费 API 服务，引入了 GPT-4 和 GPT-3.5-turbo 等强大的 AI 模型；集成细节可在 [GalaxyAI](https://galaxyapi.onrender.com) 查看。

- **警报：垃圾成人内容侵入 Discord**：有报告称在多个 LangChain AI 频道中分享了不当内容，这违反了 Discord 社区准则。

- **Meeting Reporter 将 AI 与新闻业结合**：新工具 Meeting Reporter 利用 AI 生成新闻报道，结合了 Streamlit 和 Langgraph，需要付费的 OpenAI API 密钥。该应用程序可通过 [Streamlit](https://meeting-reporter.streamlit.app/) 访问，开源代码托管在 [GitHub](https://github.com/tevslin/meeting-reporter) 上。

注意：本摘要中主动忽略了与成人内容推广相关的链接，因为它们显然与该公会的技术和工程讨论无关。



---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**推文提醒：osanseviero 分享新闻**：osanseviero 发布了推文，可能暗示了新的见解或更新；点击[此处](https://twitter.com/osanseviero/status/1778430866718421198)查看。

**RAG 聊天机器人采用嵌入数据集**：该 RAG 聊天机器人使用 [not-lain/wikipedia-small-3000-embedded](https://huggingface.co/datasets/not-lain/wikipedia-small-3000-embedded) 数据集来辅助回答，结合了检索和生成式 AI 以实现准确的信息推理。

**RMBG1.4 广受欢迎**：RMBG1.4 与 transformers 库的集成引起了极大关注，本月下载量达到 23 万次。

**Marimo-Labs 创新模型交互**：[Marimo-labs](https://x.com/marimo_io/status/1777765064386474004) 发布了一个 Python 包，允许为 Hugging Face 模型创建交互式 Playground；一个基于 WASM 的 [marimo 应用](https://marimo.app/l/tmk0k2)让用户可以使用自己的 token 查询模型。

**NLP 社区追求更长上下文的 Encoder**：AI 工程师讨论了对 BigBird 和 Longformer 等 Encoder-Decoder 模型的追求，以处理约 10-15k token 的长文本序列，并分享了使用 `trainer.train()` 的 `resume_from_checkpoint` 进行训练中断和恢复的策略。

**视觉与 Diffusion 成果**：通过 **nvitop** 增强了 GPU 进程管理，同时开发者通过增强和时间维度考量来解决视频修复问题，参考了 NAFNet、BSRGAN、Real-ESRGAN 和 [All-In-One-Deflicker](https://github.com/ChenyangLEI/All-In-One-Deflicker) 等作品。同时，人们正在寻求对 Google 多模态搜索能力的深入了解，以提高图像和拼写错误品牌的识别率，并对 AI-demos 的[识别技术](https://ai-demos.dev/)底层原理表现出兴趣。



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **混合搜索重排序（Reranking）再探讨**：工程师们讨论了在重排序之前**合并词法（lexic）和语义（semantic）搜索结果**是否优于同时合并并重排序所有结果。重排序步骤的衔接可以简化流程并降低搜索方法的延迟。

- **Rerank 3 彻底改变搜索**：**Cohere 的 Rerank 3** 模型宣称增强了**搜索和 RAG 系统**，具有 **4k 上下文长度**和支持 **100 多种语言的多语言能力**。其发布详情和功能在 [Sandra Kublik 的推文](https://x.com/itssandrakublik/status/1778422401648455694?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)中分享。

- **AI 市场因创新而升温**：创新的基于 AI 的工作自动化工具（如 **V7 Go** 和 **Scarlet AI**）的兴起，表明了自动化单调任务和促进 AI 与人类协作执行任务的增长趋势。

- **Perplexity 的“在线”模型消失后回归**：社区讨论了 Perplexity 的“在线”模型从 LMSYS Arena 消失后又重新出现的情况，这些模型具备联网能力。随着 **GPT-4-Turbo** 在 Lmsys 聊天机器人排行榜上重夺领先地位，人们的兴趣再次被点燃，这标志着其强大的代码和推理能力。

- **Mixtral-8x22B 登场**：HuggingFace Transformers 格式的 **Mixtral-8x22B** 的出现引发了关于其用途以及对 Mixture of Experts (MoEs) 架构影响的讨论。社区探讨了专家特化、MoEs 内部的学习过程以及 *semantic router* 等话题，关注冗余和专家实现中潜在的差距。

- **播客：AI 的监督角色**：新的一期播客节目呈现了与 Elicit 的 Jungwon Byun 和 Andreas Stuhlmüller 关于监督 AI 研究的讨论。可通过 [YouTube 链接](https://www.youtube.com/watch?v=Dl66YqSIu5c&embeds_referring_euri=https%3A%2F%2Fwww.latent.space%2F&feature=emb_title)观看，探讨了以产品为导向的方法相较于传统以研究为导向的方法的优势。



---

## [LAION](https://discord.com/channels/823813159592001537) Discord

**Draw Things 遭到批评**：参与者对 *Draw Things* 表示失望，指出其缺乏完整的开源方案；提供的版本省略了包括 *metal-flash-attention support* 在内的关键功能。

**TempestV0.1 令人质疑的训练成果**：社区成员对 *TempestV0.1 Initiative* 声称的 300 万次训练步数表示怀疑，同时质疑其包含 600 万张图像的数据集仅占用 200GB 的物理合理性。

**LAION 5B Demo 会重新上线吗？**：关于 *Laion 5B* 网络演示，尽管有提到 **Christoph** 表示其将回归，但目前尚不确定具体时间，也没有给出时间表或进一步信息。

**LAION 诈骗警报**：有关滥用 LAION 名义的加密货币计划等诈骗行为正在流传，建议保持警惕，并讨论了通过发布公告或增强自动审核（automatic moderation）来应对此问题。

**Diffusion 和 LRU 算法的进展**：社区正在评估在 Long Range Arena 基准测试中改进的 *Least Recently Used* (LRUs) 算法，并讨论增强 Diffusion 模型的 guidance-weight 策略，相关的研究（[研究论文](https://arxiv.org/abs/2404.07724)）和活跃的 GitHub issue（[GitHub issue](https://github.com/huggingface/diffusers/issues/7657)）正被应用于 **huggingface** 的 **diffusers**。



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Pandas 迁移**：`PandasQueryEngine` 将随 **LlamaIndex python v0.10.29** 迁移至 `llama-index-experimental`，安装将通过 `pip install llama-index-experimental` 进行。需要调整 Python 代码中的 import 语句以反映此更改。

- **提升 GitHub 聊天体验**：一个[新教程](https://t.co/BM5yUlCBo7)展示了如何创建一个应用，实现与来自 GitHub 仓库的代码进行对话，并将 LLM 与 Ollama 集成。另一个教程详细介绍了如何使用基于 Colbert 的 Agent 为 **LlamaIndex** 的文档检索引入 memory 功能，从而提升检索过程。

- **动态组合：结合 Auto-Merging 增强的 RAG**：一种新颖的 RAG 检索方法包括[自动合并](https://t.co/0HS5FrLR9X)，旨在从破碎的上下文中形成更连续的 chunks。相比之下，关于 Q/A 任务的讨论显示，由于 **Retriever Augmented Generation (RAG)** 在准确性、成本和灵活性之间的平衡，它比微调 LLM 更受青睐。

- **符合 GDPR 的 AI 应用工具包**：受 Llama Index 的 create-llama 工具包启发，**create-tsi toolkit** 是由 T-Systems 和 Marcus Schiesser 推出的一个全新的、符合 GDPR 标准的 AI 应用基础设施。

- **调试 Embeddings 和 Vector Stores**：讨论澄清了关于 Embedding 存储的困惑，透露它们驻留在 **storage context** 内的 vector stores 中。对于某些导致 **QdrantVectorStore** 崩溃的 'fastembed' 问题，解决方案是降级到 `llama-index-vector-stores-qdrant==0.1.6`，且从 Embeddings 中排除 metadata 需要在代码中进行显式处理。



---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**安装过程中的麻烦**：成员们报告了安装 **Poetry** 和 `litellm` 时遇到的问题——前者的成功修复方法包括运行 `pip install poetry`，而诊断 `litellm` 问题则涉及使用 `interpreter --version` 和 `pip show litellm`。进一步的故障排除指出，安装 Python 和特定的 git commits 对于恢复包是必要的。

**对未来科技硬件保持耐心**：关于新设备的预订和交付的咨询显示，一些科技硬件仍处于原型阶段，预计将在夏季月份发货。对话强调了初创公司在制造过程中面临的典型延迟，并鼓励热切的科技爱好者保持耐心。

**JavaScript 重新定义 Transformers**：**[transformers.js GitHub repository](https://github.com/xenova/transformers.js)** 提供了一种基于 JavaScript 的机器学习解决方案，能够在无需服务器的情况下在浏览器中运行，这引起了 AI 工程师的兴趣。同时，一个指向 https://api.aime.info 的 AI 模型 endpoint 的神秘提及也出现了，但没有更多细节或宣传。

**OpenAI 的额度策略**：OpenAI 从按月计费转向预付 credits，其中包括一项[免费额度推广活动](https://discord.com)（截止日期为 2024 年 4 月 24 日），这引发了成员们的好奇心，并就不同账户类型的后续影响进行了大量信息交流。

**活动与贡献**：社区活动 [Novus](https://lu.ma/novus28) 的邀请函反响热烈，工程师们期待着没有冗余信息的社交机会；同时，关于将 Open Interpreter 作为库使用的成功分享产出了一个包含 [Python templates](https://github.com/MikeBirdTech/open-interpreter-python-templates) 的仓库，供初学者使用。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**讨论 AI 开发的策略与预期**：
- 参与者探讨了在神经网络模型中 freezing layers 的影响，认为虽然减少层数可以简化模型，但也可能降低有效性，从而暗示了复杂性与资源效率之间的微妙平衡。关于语言模型 scaling 理论基础的讨论链接，特别是 Meta 关于知识位 scaling 的研究（[Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws](https://arxiv.org/abs/2404.05405)），表明 LLaMA-3 可能会进一步推进这种平衡。

**训练挑战与模型修改**： 
- 将 Mistral-22B 从 Mixture of Experts 转换为 dense model ([Vezora/Mistral-22B-v0.1](https://huggingface.co/Vezora/Mistral-22B-v0.1)) 一直是焦点，这表明社区对 dense architectures 的兴趣，可能是因为它们与现有基础设施的兼容性。同时，关于以 11 层为增量进行训练的讨论表明，人们正在寻求适应有限 GPU 能力的训练策略。

**生态扩展与协助**： 
- 该集体致力于降低 AI 开发过程的门槛，这从为新成员提供的 Axolotl 入门建议中可见一斑，这些建议体现在一篇富有见地的 [blog post](https://drchrislevy.github.io/posts/intro_fine_tune/intro_fine_tune.html) 和实用技巧（如使用 `--debug` 标志）中。此外，维护的 [Colab notebook example](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/colab-notebooks/colab-axolotl-example.ipynb) 协助用户在 Hugging Face 数据集上 fine-tuning Tiny-Llama 等模型。

**资源受限下的机智**： 
- 对话围绕着创新的训练策略展开，例如为硬件配置较低的用户 unfreezing random subsets of weights，证明了对训练方法民主化的关注。共同分享 pretrain configs，以及利用 Docker 和 DeepSpeed 进行 multi-node fine-tuning 的分步干预，展示了社区在受限环境下驾驭高端训练策略的决心。

**好奇心驱动数据获取**： 
- 对形式逻辑推理数据集和大规模 200-billion token 数据集的查询，描绘了对具有挑战性的大规模数据的积极寻找，以突破模型 pretraining 和实验的界限。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**API 因 AttributeError 受阻**：一位 **OpenAI API 用户**在 Python 的 `client.beta.messages.create` 方法中遇到了 `AttributeError`，这引发了人们对文档可能与库更新不同步的担忧。分享的[代码片段](https://openai.com/api)在社区讨论中未能得出解决方案。

**模型成为焦点**：成员们分享了使用 **Gemini 1.5 和 Claude** 等 AI 模型的不同经验，涉及上下文窗口、内存和代码查询处理方面的差异。针对 Unity 中的 C# 开发，为了保证效率，建议使用 **gpt-4-turbo** 和 **Opus** 模型。

**GPT-4 Turbo 的效率障碍**：一位成员观察到 **GPT-4-turbo** 模型在函数调用（function calls）方面似乎表现欠佳，而另一位成员则不确定如何访问它；然而，讨论中并未提供详细的示例或解决方案。

**使用 LLM 进行大规模文本编辑**：关于使用 **GPT** 编辑大型文档的咨询引发了讨论，内容涉及可能需要第三方服务来绕过标准的上下文窗口限制。

**探索提示工程银河**：对于那些开始学习提示工程（prompt engineering）的人，[Prompting Guide](http://promptingguide.ai) 被推荐为学习资源，而将 **Wolfram** 与 **GPT** 集成可以通过 [Wolfram GPT 链接](https://chat.openai.com/g/g-0S5FXLyFN-wolfram)和平台内的 `@mention` 功能来实现。



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

**稠密模型的重大胜利**：**[Mistral-22B-V.01](https://huggingface.co/Vezora/Mistral-22B-v0.1)** 的发布（一个全新的 **22B 参数稠密模型**）标志着一项显著成就，因为它从压缩的混合专家模型（MoE）转变为稠密形式，为 MoE 到 Dense 模型的转换领域树立了先例。

**跨语言难题与语料库对话**：虽然工程师们正致力于在 **DiscoLM 70b** 等模型中平衡英语和德语数据，并[计划更新模型](https://huggingface.co/DiscoResearch/DiscoLM-70b#dataset)，但他们提到需要更好的德语基准测试（benchmarks）。**Occiglot-7B-DE-EN-Instruct** 展示了潜力，暗示英语和德语训练数据的混合可能是有效的。

**筛选 SFT 策略**：社区分享了在预训练阶段早期集成有监督微调（SFT）数据的潜在益处，这得到了来自 [StableLM](https://arxiv.org/abs/2402.17834) 和 [MiniCPM](https://arxiv.org/abs/2404.06395) 研究的支持，旨在增强模型的泛化能力并防止过拟合。

**Zephyr 凭借 ORPO 翱翔**：**Zephyr 141B-A35B** 亮相，它衍生自 Mixtral-8x22B，并通过名为 ORPO 的新算法进行了微调，目前可在 [Hugging Face 模型中心](https://huggingface.co/HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1)进行探索。

**MoE 合并带来的挑战**：社区使用 Mergekit 通过合并创建自定义 MoE 模型的实验表现平平，这引发了关于在窄领域进行 SFT 与传统 MoE 模型实用性的持续辩论。



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**是增量还是进化？**：Nathan Lambert 发起了一场辩论，讨论从 **Claude 2 到 Claude 3** 的跨越是代表了真正的进步，还是仅仅是 *“[增量式](https://vxtwitter.com/natolambert/status/1778571382491947450)”* 的改进，对 AI 版本更新的实质内容提出了质疑。

**一砖一瓦构建更好的模型**：成员们讨论了**预训练**、**有监督微调 (SFT)** 和 **RLHF** 的混合使用，指出这些技术通常是结合在一起的，尽管这种实践的文档记录很少。一位成员承诺将分享关于在这一系列混合方法中应用**退火（annealing）**技术的见解。

**随意的祝贺演变成幽默**：一个 **meme** 意外地表达了祝贺，引发了幽默时刻，而另一场对话则澄清了该服务器不需要接受即可订阅。

**谷歌 CodecLM 备受关注**：社区研究了谷歌的 **CodecLM**（分享于一篇[研究论文](https://arxiv.org/pdf/2404.05875.pdf)中），指出这是通过使用**定制合成数据**实现“向更强模型学习”趋势的又一尝试。

**关于 LLaMA 的学术交流**：发布了 [**“LLaMA: Open and Efficient Foundation Language Models”**](https://huggingface.co/collections/natolambert/aligning-open-language-models-66197653411171cc9ec8e425) 的链接，表明正在对发布日期为 **2023 年 2 月 27 日**的开放、高效基础语言模型的进展进行积极讨论。



---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **敏捷的命名技巧大爆发**：**tinygrad** Discord 的成员们选择了极具创意的标签，如 **tinyxercises** 和 **tinyproblems**，而俏皮的术语 **tinythanks** 也作为对话中的感谢表达而出现。
- **缓存层级之争**：聊天中的技术交流指出，由于最小化了缓存一致性管理的需求，**L1 caches** 相比池化共享缓存具有更优越的速度。这次讨论强调了将直接的 **L3 to L1 cache transfers** 与异构缓存架构进行比较时的性能差异。
- **思考编程语言的可移植性**：对话揭示了一种对比观点，即 **ANSI C** 广泛的硬件支持和易移植性，与对 **Rust** 安全性的共同审查形成对比，后者通过一个[已知 Rust 漏洞链接](https://www.cvedetails.com/vulnerability-list/vendor_id-19029/product_id-48677/Rust-lang-Rust.html)被揭开了神秘面纱。
- **商标策略引发讨论**：针对 [**Rust Foundation** 限制性的商标政策](https://lunduke.substack.com/p/the-rust-foundation-goes-to-war-against) 发表了具有争议的观点，引发了与 Oracle 和 Red Hat 等其他实体及其自身备受争议的许可条款的比较。
- **Discord 纪律执行**：**George Hotz** 明确表示，在他的 **Discord** 中不允许无关痛痒的闲聊，导致一名用户因对专注的技术讨论缺乏贡献而被禁言。

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **寻找逻辑宝库**：AI 工程师们分享了一个[精选列表](https://github.com/neurallambda/awesome-reasoning)，其中充满了旨在增强**自然语言中形式逻辑推理**的数据集，为逻辑与 AI 交叉领域的项目提供了宝贵资源。
- **文献建议：增强 LLM 中的 Coq 能力**：一篇 [arXiv 论文](https://arxiv.org/abs/2403.12627) 受到关注，该论文解决了提高大语言模型解释和生成 **Coq proof assistant** 代码能力的挑战——这是推进形式化定理证明能力的关键。
- **将符号能力集成到 LLM 中**：工程师们对 [Logic-LLM](https://github.com/teacherpeterpan/Logic-LLM) 产生了兴趣，这是一个讨论实现符号求解器以提升语言模型逻辑推理准确性的 GitHub 项目。
- **通过 Lisp 翻译升级推理能力详解**：对一个通过将**人类文本翻译为可执行的 Lisp 代码**来增强 LLM 的项目进行了说明，旨在通过在 LLM 的潜空间（latent space）内进行计算来增强推理，同时保持端到端的可微性。
- **推理仓库内容更丰富了！**：**awesome-reasoning repo** 的 [提交历史](https://github.com/neurallambda/awesome-reasoning/commits/master/) 更新了新资源，成为了支持推理 AI 开发的更全面的汇编。

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **对 Haiku 速度炒作的质疑**：社区成员正在质疑 **Haiku** 声称的速度提升，特别关注它是否显著改善了总响应时间，而不仅仅是吞吐量。
- **Turbo 成为焦点**：讨论中的工程师对新发布的 **turbo** 在速度和代码处理方面的改进很感兴趣，一些人正在考虑重新激活 **ChatGPT Plus** 订阅以实验 turbo 的功能。

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

**代码求助**：一名公会成员寻求精通此道的同僚通过私信提供代码帮助。

**服务器邀请审查**：对服务器上过度分享 Discord 邀请的行为表示担忧，引发了关于可能禁止此类行为的讨论。

**OO2 项目状态检查**：对 OO2 项目的当前状态进行了简单的询问，质疑其活跃度。

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Gemini 听取并学习**：**Gemini** 模型已增强，能够回答有关视频中音频的问题，这标志着其从早期只能生成非音频描述的限制中取得了进步。
  
- **Google 的文本粘贴问题依然存在**：技术讨论指出，在粘贴到 **Google** 平台时，其文本格式问题一直令人沮丧，影响了用户效率。

- **STORM 项目的巨大影响力**：工程师们关注到了 [STORM 项目](https://github.com/stanford-oval/storm)，这是一个由 **LLM** 驱动的知识策展系统，强调了其自主研究主题并生成带有引用的全面报告的能力。

- **macOS Zsh 命令挂起问题已修复**：在 **macOS zsh** shell 上使用 `llm` 命令时的**挂起问题**已通过[最近的 Pull Request](https://github.com/simonw/llm-cmd/pull/12) 解决，并已在 M1 Macs 的 Terminal 和 Alacritty 上验证了功能。



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Figma 与 Gradio 合作**：Mozilla Innovations 发布了 **Gradio UI for Figma**，受 Hugging Face 的 Gradio 启发，该库为设计师提供了快速原型设计的能力。[Figma 用户](https://www.figma.com/@futureatmozilla)现在可以访问此工具包以增强设计工作流。
  
- **加入 Gradio UI 对话**：来自 Mozilla Innovation Studio 的 Thomas Lodato 正在主持关于 **Gradio UI for Figma** 的讨论；对用户界面感兴趣的工程师可以[在此加入讨论](https://discord.com/channels/1089876418936180786/1091372086477459557/1228056720132280461)。

- **llamafile OCR 潜力解锁**：社区对 llamafile 的 **OCR 能力** 兴趣日益浓厚，成员们正在探索该功能的各种应用。

- **Rust 在 AI 领域的狂热**：推荐了一个名为 **Burnai** 的新项目，它利用 Rust 进行深度学习推理（inference），并具有性能优化；请关注 [burn.dev](https://burn.dev/) 并考虑 [justine.lol/matmul](https://justine.lol/matmul/?ref=dailydev) 以获取 Rust 相关的进展。

- **Llamafile 获得 McAfee 认可**：**llamafile 0.7 binary** 现已进入 McAfee 白名单，消除了用户的安全疑虑。



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

**寻找 Jamba 的起源**：一位社区成员表示希望找到 **Jamba 的源代码**，但未提供 URL 或源代码位置。

**渴望掌握模型合并技术**：分享了一个 GitHub 仓库链接 [moe_merger](https://github.com/isEmmanuelOlowe/moe_merger/tree/master)，该仓库提出了一种模型合并的方法论，尽管目前仍处于实验阶段。

**为协作点赞**：用户对合并模型的资源表示感谢，表明社区对该贡献有积极的反响。

**充满期待**：用户对更新充满期待，可能涉及正在进行的项目或之前消息中的讨论。

**共享智慧待命**：用户正在分享资源并表达感谢，展示了一个积极交换信息和支持的协作环境。



---

# PART 2: Detailed by-Channel summaries and links



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1227877111990063165)** (846 messages🔥🔥🔥): 

- **介绍 Forge**：Forge 是 Automatic1111（或 A1111）的一个分支（fork），因其相对于 A1111 的性能提升而受到赞誉。鼓励用户尝试，特别是它不需要卸载 A1111，并且可以使用来自 ComfyUI 的模型。

- **Ella 的动漫困扰**：用户报告称，Ella 虽然前景看好，但会严重降低生成的动漫风格图像的质量，使其无法用于该流派。尽管尝试了各种 Checkpoints（包括 Ella 创作者推荐的），用户仍无法获得满意的结果。

- **对 SD3 的期待**：在社区中，对于 Stable Diffusion 3 (SD3) 的发布既有兴奋也有怀疑，讨论围绕着对 SD3 解决当前生成模型局限性的期望，如处理背景虚化（bokeh）效果、色彩准确度和名人识别。

- **工具和扩展琳琅满目**：社区讨论了各种改进 Stable Diffusion 输出的工具和模型扩展，例如用于外扩填充（outpainting）的 BrushNet、depth-fm、用于建筑的 geowizard 以及一个用于色彩准确度的扩展。鼓励用户探索并关注最新发布。

- **Cascade 的奇特特质**：Cascade 以学习速度快和在 SD 模型中的独特特性而著称，尽管它也被描述为使用起来具有挑战性，并被亲切地称为“SD 家族中的怪表亲”。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://imgur.com/a/UMQhBhy">Ella 1.5 ComfyUI 结果</a>：在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过有趣的笑话、流行模因、娱乐 GIF、励志故事、病毒视频等来振奋你的精神...</li><li><a href="https://www.livescience.com/technology/artificial-intelligence/mit-has-just-worked-out-how-to-make-the-most-popular-ai-image-generators-dall-e-3-stable-diffusion-30-times-faster">MIT 科学家刚刚弄清楚了如何让最流行的 AI 图像生成器快 30 倍</a>：科学家们构建了一个框架，通过将 DALL·E 3 和 Stable Diffusion 等生成式 AI 系统压缩成更小的模型，在不损害其质量的情况下大幅提升其性能...</li><li><a href="https://www.udio.com/songs/gnFD9LGCUZx7NvECfpKdRy">Udio | Metal Warriors，作者 MrJenius</a>：制作你的音乐</li><li><a href="https://huggingface.co/spaces/TencentARC/InstantMesh">InstantMesh - TencentARC 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/">Hugging Face – 构建未来的 AI 社区。</a>：未找到描述</li><li><a href="https://github.com/Stability-AI/stablediffusion/tree/main/configs/stable-diffusion">Stability-AI/stablediffusion main 分支下的 stablediffusion/configs/stable-diffusion</a>：使用 Latent Diffusion 模型进行高分辨率图像合成 - Stability-AI/stablediffusion</li><li><a href="https://civitai.com/models/21100?modelVersionId=94130">ComfyUI 多主体工作流 - Interaction OP v2.2 | Stable Diffusion 工作流 | Civitai</a>：最后更新的工作流：Interaction OpenPose v2.1 -&amp;gt; v2.2 请从模型版本下载，而不是“更新 [...]”，因为我会删除并重新创建它...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/xelydv/stablediffusioninfinity_outpainting_with/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=wyDRHRuHbAU">A1111 Stable Diffusion 技术终极指南</a>：潜入高分辨率数字艺术的世界，我们将踏上五步之旅，将平凡转化为非凡的 4K 和 8K 视觉杰作...</li><li><a href="https://huggingface.co/stabilityai">stabilityai (Stability AI)</a>：未找到描述</li><li><a href="https://huggingface.co/lambdalabs/sd-pokemon-diffusers">lambdalabs/sd-pokemon-diffusers · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/DataCTE/ELLA_Training">GitHub - DataCTE/ELLA_Training</a>：通过在 GitHub 上创建账户，为 DataCTE/ELLA_Training 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1c0vwd4/talkllamafast_informal_videoassistant/?utm_source=share&utm_medium=web2x&context=3">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=q5MgWzZdq9s&t=20s">Stable Diffusion Forge UI：底层探索 - 技巧与窍门 #stablediffusion</a>：在这段视频中，我们将详细了解 Stable Diffusion Forge UI，涵盖从查找和更新模型、设置到增强功能的所有内容...</li><li><a href="https://youtu.be/mAUpxN-EIgU?feature=shared&t=263">OpenAI 的 Sora 为我制作了疯狂的 AI 视频——然后 CTO 回答了我的（大部分）问题 | 华尔街日报</a>：OpenAI 新的文本转视频 AI 模型 Sora 可以创建一些非常真实的场景。这项生成式 AI 技术是如何工作的？为什么它有时会出错？什么时候会...</li><li><a href="https://github.com/hnmr293/sd-webui-cutoff">GitHub - hnmr293/sd-webui-cutoff: Cutoff - 切断提示词效果</a>：Cutoff - 切断提示词效果。通过在 GitHub 上创建账户，为 hnmr293/sd-webui-cutoff 的开发做出贡献。</li><li><a href="https://stable-diffusion-art.com/install-stable-diffusion-2-1/">如何在 AUTOMATIC1111 GUI 中安装 Stable Diffusion 2.1 - Stable Diffusion Art</a>：Stable Diffusion 2.1 发布于 2022 年 12 月 7 日。</li><li><a href="https://huggingface.co/stabilityai/stable-diffusion-2-1/tree/main">stabilityai/stable-diffusion-2-1 at main</a>：未找到描述
</li>
</ul>

</div>
  

---



**Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1227909306687426602)** (522 条消息🔥🔥🔥): 

- **CORS 访问故障**：用户报告了一个 **Cohere dashboard** 无法访问的问题，确定是 CORS 策略错误阻止了从 `https://dashboard.cohere.com` 到 `https://production.api.os.cohere.ai` 的 JavaScript fetch 请求。

- **关于 Command R+ 和上下文长度的引人入胜的对话**：社区就 **LLM 中长上下文长度的有效性和实用性**与 **Retrieval-Augmented Generation (RAG)** 等策略展开了激烈辩论。论点包括计算效率以及增加上下文长度带来的收益递减。

- **Rerank V3 发布，定价为每 1,000 次搜索 2 美元**：针对 **Rerank V3 的定价**进行了说明，设定为每 1,000 次搜索 2 美元，由于定价变更实施较晚，**目前的促销活动提供 50% 的折扣**；Rerank V2 仍以每 1,000 次搜索 1 美元的价格提供。

- **Cohere 微调和本地部署查询得到解答**：在讨论中，有人提出了关于 **在本地或通过 Cohere 平台微调 Cohere 的 LLM 的能力**，以及在 AWS Bedrock 或本地环境中部署这些模型的问题。

- **Cohere 社区涌现大量友好的自我介绍**：新成员介绍了自己，包括来自尼日利亚的 **Tayo** 表达了对 Cohere 的 LLM 的感谢，以及其他表示对 AI 感兴趣或参与其中的个人。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.06654">RULER: What&#39;s the Real Context Size of Your Long-Context Language Models?</a>: 大海捞针 (NIAH) 测试用于检查从长干扰文本（“干草堆”）中检索一条信息（“针”）的能力，已被广泛采用...</li><li><a href="https://colab.research.google.com/drive/1sKEZY_7G9icbsVxkeEIA_qUthEfPrK3G?usp=sharing&ref=txt.cohere.com">Google Colaboratory</a>: 未找到描述</li><li><a href="https://discord.gg/Y4msga6k?event=1208132674762575994">Join the Cohere Community Discord Server!</a>: Cohere 社区服务器。来聊聊 Cohere API、LLMs、Generative AI 以及介于两者之间的一切。| 15292 名成员</li><li><a href="https://docs.cohere.com/docs/retrieval-augmented-generation-rag">Retrieval Augmented Generation (RAG) - Cohere Docs</a>: 未找到描述</li><li><a href="https://en.wikipedia.org/wiki/Creative_Commons_NonCommercial_license">Creative Commons NonCommercial license - Wikipedia</a>: 未找到描述</li><li><a href="https://dashboard.cohere.com'">no title found</a>: 未找到描述</li><li><a href="https://tenor.com/oTTdPcKwPgW.gif">Screaming Mad GIF - Screaming Mad Fish - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://dashboard.cohere.com/fine-tuning/?">Login | Cohere</a>: Cohere 通过一个易于使用的 API 提供对先进的大型语言模型 (Large Language Models) 和 NLP 工具的访问。免费开始使用。</li><li><a href="https://en.wikipedia.org/wiki/Special:Search?search=glucose",">glucose&quot;, - Search results - Wikipedia</a>: 未找到描述</li><li><a href="https://openrouter.ai/models/cohere/command-r-plus">Command R+ by cohere | OpenRouter</a>: Command R+ 是来自 Cohere 的一款新型 104B 参数 LLM。它适用于角色扮演、普通消费者用例和检索增强生成 (RAG)。它为十种关键语言提供多语言支持...</li><li><a href="https://www.youtube.com/watch?v=b2F-DItXtZs">Episode 1 - Mongo DB Is Web Scale</a>: 关于 No SQL 和关系型数据库优点的问答讨论。</li><li><a href="https://press.asus.com/news/asus-dual-geforce-rtx-4060-ti-ssd-m2-nvme-thermal-performance/">ASUS Announces Dual GeForce RTX 4060 Ti SSD Graphics Card</a>: 未找到描述</li><li><a href="https://huggingface.co/CohereForAI">CohereForAI (Cohere For AI)</a>: 未找到描述</li><li><a href="https://aws.amazon.com/marketplace/seller-profile?id=87af0c85-6cf9-4ed8-bee0-b40ce65167e0">AWS Marketplace: Cohere</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/1902.11266">Efficient Parameter-free Clustering Using First Neighbor Relations</a>: 我们提出了一种以单一聚类方程形式呈现的新聚类方法，能够直接发现数据中的分组。主要命题是每个样本 i 的首个邻居...</li><li><a href="https://github.com/ssarfraz/FINCH-Clustering">GitHub - ssarfraz/FINCH-Clustering: Source Code for FINCH Clustering Algorithm</a>: FINCH 聚类算法源代码。通过在 GitHub 上创建一个账户来为 ssarfraz/FINCH-Clustering 的开发做出贡献。</li><li><a href="https://arxiv.org/abs/2203.12997">Hierarchical Nearest Neighbor Graph Embedding for Efficient Dimensionality Reduction</a>: 降维对于机器学习中高维数据的可视化和预处理都至关重要。我们介绍了一种基于 1-最近邻图构建的层级结构的新方法...</li><li><a href="https://github.com/koulakis/h-nne">GitHub - koulakis/h-nne: A fast hierarchical dimensionality reduction algorithm.</a>: 一种快速的分层降维算法。 - koulakis/h-nne</li><li><a href="https://cohere.com/events/c4ai-Saquib-Sarfraz-2024">Cohere For AI - Guest Speaker: Dr. Saquib Sarfraz, Deep Learning Lead</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1227976995535847524)** (1 条消息):

- **Rerank 3 助力增强型企业搜索**：发布 **Rerank 3**，这是一款旨在提高企业搜索和 RAG 系统效率的基础模型，现在能够处理复杂的半结构化数据，并拥有高达 *3 倍的推理速度提升*。它支持 100 多种语言和 4k 的长上下文长度，以提高包括代码检索在内的各种文档类型的准确性。
- **通过 Cohere 集成提升您的 Elastic Search**：**Rerank 3** 现在已在 **Elastic 的 Inference API** 中得到原生支持，从而实现企业搜索功能的无缝增强。感兴趣的开发者可以参考[使用 Cohere 进行嵌入的详细指南](https://docs.cohere.com/docs/elasticsearch-and-cohere)和实操性的 [Cohere-Elastic notebook 示例](https://github.com/cohere-ai/notebooks/blob/main/notebooks/Cohere_Elastic_Guide.ipynb)开始集成。
- **开启最先进的企业搜索**：在他们最新的[博客文章](https://txt.cohere.com/rerank-3/)中，**Rerank 3** 因其最先进的能力而备受赞誉，包括大幅提升的长文档搜索质量、搜索多维度数据的能力以及多语言支持，同时保持低延迟。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.cohere.com/docs/elasticsearch-and-cohere">Elasticsearch and Cohere</a>: 未找到描述</li><li><a href="https://txt.cohere.com/rerank-3/">Introducing Rerank 3: A New Foundation Model for Efficient Enterprise Search &amp; Retrieval</a>: 今天，我们推出了最新的基础模型 Rerank 3，专为增强企业搜索和检索增强生成 (RAG) 系统而构建。我们的模型与任何数据源都兼容...
</li>
</ul>

</div>
  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1227888318541205535)** (268 条消息🔥🔥): 

- **Mixtral 模型困境**：几位成员对 *Mixtral* 和 *Perplexity Labs* 模型存在的重复问题和行为异常表示担忧，将其行为比作故障。批评意见包括肤浅的指令微调（instruction fine-tuning）以及与基座模型类似的重复输出，一位成员提到 [这个 GitHub 仓库](https://github.com/searxng/searxng) 是创建基于搜索的模型的更好替代方案。

- **对即将推出的 Instruct 模型的期待升温**：人们对新 Instruct 模型的发布表现出浓厚兴趣，一名来自 *Mistral 的管理员* 确认预计在一周内发布，引发了对 *Llama* 和 *Mistral* 等不同模型之间潜在对决的期待。

- **探索 LLM 中的长上下文窗口**：用户深入讨论了利用高达 228K 的长上下文窗口来微调 LLM，*Unsloth AI* 将内存占用减少了 30%，而时间开销仅增加了 1.9%，更多详情见 [Unsloth 博客](https://unsloth.ai/blog/long-context)。

- **寻求特定领域数据**：一位成员向社区询问收集特定领域 128k 上下文大小指令数据集的最佳实践。大家提出了多种建议，包括查看 HF 数据集，但对话倾向于需要更专业和特定领域的数据收集方法。

- **Unsloth AI 关于微调 LLM 的网络研讨会**：Unsloth AI 举办了一场由 Analytics Vidhya 主持的网络研讨会，演示了 Unsloth 的现场 Demo 并分享了微调技巧和窍门，这引起了社区的关注，甚至导致了最后一刻的通知。他们还邀请成员参加旨在分享知识和进行 Q&A 环节的 [Zoom 活动](https://us06web.zoom.us/webinar/register/WN_-uq-XlPzTt65z23oj45leQ)。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://txt.cohere.com/rerank-3/">Introducing Rerank 3: A New Foundation Model for Efficient Enterprise Search &amp; Retrieval</a>: 今天，我们推出了最新的基础模型 Rerank 3，专为增强企业搜索和检索增强生成 (RAG) 系统而构建。我们的模型与任何数据源兼容...</li><li><a href="https://www.bloomberg.com/news/newsletters/2024-04-11/this-startup-is-trying-to-test-how-well-ai-models-actually-work">Bloomberg - Are you a robot?</a>: 未找到描述</li><li><a href="https://us06web.zoom.us/webinar/register/WN_-uq-XlPzTt65z23oj45leQ">Video Conferencing, Web Conferencing, Webinars, Screen Sharing</a>: Zoom 是现代企业视频通信的领导者，拥有简单、可靠的云平台，可跨移动端、桌面端和会议室系统进行视频和音频会议、聊天及网络研讨会。Zoom ...</li><li><a href="https://huggingface.co/collections/LumiOpen/viking-660fa4c659d8544c00f77d9b">Viking - a LumiOpen Collection</a>: 未找到描述</li><li><a href="https://unsloth.ai/blog/long-context">Unsloth - 4x longer context windows &amp; 1.7x larger batch sizes</a>: Unsloth 现在支持超长上下文窗口的 LLM 微调，在 H100 上最高可达 228K（Hugging Face + Flash Attention 2 为 58K，即 4 倍长），在 RTX 4090 上为 56K（HF + FA2 为 14K）。我们成功...</li><li><a href="https://huggingface.co/AI-Sweden-Models/">AI-Sweden-Models (AI Sweden Model Hub)</a>: 未找到描述</li><li><a href="https://developer.nvidia.com/nccl">NVIDIA Collective Communications Library (NCCL)</a>: 未找到描述</li><li><a href="https://github.com/tinygrad/open-gpu-kernel-modules">GitHub - tinygrad/open-gpu-kernel-modules: NVIDIA Linux open GPU with P2P support</a>: 支持 P2P 的 NVIDIA Linux 开源 GPU 内核模块。通过在 GitHub 上创建账号来为 tinygrad/open-gpu-kernel-modules 的开发做出贡献。</li><li><a href="https://github.com/searxng/searxng">GitHub - searxng/searxng: SearXNG is a free internet metasearch engine which aggregates results from various search services and databases. Users are neither tracked nor profiled.</a>: SearXNG 是一个免费的互联网元搜索引擎，它聚合了来自各种搜索服务和数据库的结果。用户既不会被追踪，也不会被画像。 - searxng/searxng
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1227879682758803456)** (244 messages🔥🔥): 

- **解决微调难题**：用户讨论了微调 NLP 任务模型的挑战；其中一个涉及训练期间良好的评估指标与推理指标不佳之间的差异，另一个与他们在非英语 NLP 环境中努力训练模型以提高准确性有关。

- **模型部署对话**：讨论了在使用 Unsloth AI 训练后如何部署模型，提到了可能的合并策略，并提供了 [Unsloth AI GitHub wiki](https://github.com/unslothai/unsloth) 的链接，以指导部署过程，包括 Mistral-7B 等模型。

- **VRAM 饥饿游戏**：一位成员表示，即使应用了 Unsloth 的 VRAM 效率更新，也很难将模型放入 VRAM 限制内。他们讨论了各种策略，包括微调 Batch Size 以及将上下文 Token 合并到基础模型中以节省 VRAM 使用量。

- **GEMMA 微调的数据集格式化**：有人寻求在自定义数据集上微调 GEMMA 的帮助，被引导使用 [Pandas](https://github.com/pandas-dev/pandas) 将其数据转换并加载为 Hugging Face 兼容格式，并取得了成功。

- **机器学习中的 GPU 限制**：在关于 Unsloth AI 多 GPU 支持的辩论中，用户澄清说，虽然 Unsloth 支持多 GPU，但官方支持和文档可能尚未更新，且许可限制旨在防止大型科技公司的滥用。简要提到的与 Llama-Factory 的集成暗示了潜在的多 GPU 解决方案。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing#scrollTo=p31Z-S6FUieB)">Google Colaboratory</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/datasets/en/loading#json">Load</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki">Home</a>: 快 2-5 倍，节省 80% 显存的 QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.co">GitHub: Let’s build from here</a>: GitHub 是超过 1 亿开发者共同塑造软件未来的地方。在这里为开源社区做贡献，管理你的 Git 仓库，像专业人士一样审查代码，跟踪错误和功能...</li><li><a href="https://github.com/akameswa/CodeGenerationMoE/blob/main/code/finetune.ipynb">CodeGenerationMoE/code/finetune.ipynb at main · akameswa/CodeGenerationMoE</a>: 用于代码生成的 Mixture of Expert 模型。通过在 GitHub 上创建账户为 akameswa/CodeGenerationMoE 的开发做出贡献。</li><li><a href="https://github.com/pandas-dev/pandas">GitHub - pandas-dev/pandas: Flexible and powerful data analysis / manipulation library for Python, providing labeled data structures similar to R data.frame objects, statistical functions, and much more</a>: 为 Python 提供灵活且强大的数据分析/操作库，提供类似于 R data.frame 对象的标记数据结构、统计函数等 - pandas-dev/pandas</li><li><a href="https://github.com/Green0-0/Discord-LLM-v2">GitHub - Green0-0/Discord-LLM-v2</a>: 通过在 GitHub 上创建账户为 Green0-0/Discord-LLM-v2 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: 2-5X faster 80% less memory QLoRA &amp; LoRA finetuning</a>: 快 2-5 倍，节省 80% 显存的 QLoRA &amp; LoRA finetuning - unslothai/unsloth
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1227903594716463105)** (7 条消息): 

- **Ghost 7B 预览**: 即将推出的 **Ghost 7B** 模型被宣传为一款小型、多语言大模型，在推理、越南语理解和专家知识方面表现出色。社区反响热烈，用户们正期待其发布。

- **丰富低资源语言**: 分享了增强低资源语言数据集的技巧，包括利用翻译数据或来自 [HuggingFace](https://huggingface.co/ghost-x) 的资源。成员们对 Ghost X 项目的进展表示支持和热情。

- **社区对 Ghost X 的支持**: **Ghost 7B** 的新版本受到了社区成员的掌声和鼓励。积极的反馈凸显了 Ghost X 项目所做的工作。

**提到的链接**: <a href="https://huggingface.co/ghost-x">ghost-x (Ghost X)</a>: 未找到描述

  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/)** (1 条消息): 

starsupernova: 噢是的没错！我也看到了那些推文！
  

---



**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1227884996648243264)** (15 条消息🔥): 

- **分享撒钱 Gif**: 一位成员发布了来自 [Tenor](https://tenor.com/view/money-rain-erlich-bachman-tj-miller-silicon-valley-unicorn-gif-11481689) 的 gif 链接，展示了美剧《硅谷》中 Erlich Bachman 身上撒钱的场景。
- **富有洞察力的朝鲜访谈**: 分享了一个名为“Стыдные вопросы про Северную Корею”的 [YouTube 视频](https://www.youtube.com/watch?v=C84bzu9wXC0)，内容是对一位朝鲜专家的三小时访谈，配有英语字幕和配音。
- **Claude AI 创作歌词**: 一位成员提到，[udio.com](https://www.udio.com/songs/oSC6u46BSPgeXKonjGJARj) 上列出的一首歌的歌词是由名为 Claude 的 AI 创作的。
- **针对垃圾信息的自动审核**: 针对邀请垃圾信息的担忧，一位成员指出已实施自动系统，如果发送者在短时间内发送过多消息，系统将删除消息并禁言该垃圾信息发送者，并向管理员发送通知。
- **Claude 与 GPT-4 的对比**: 一位成员表示在使用 Anthropic 的 Claude AI 时感到有些迷茫，表示更倾向于 GPT-4 的回答，认为 GPT-4 更符合他们的想法。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.udio.com/songs/oSC6u46BSPgeXKonjGJARj">Udio | An Intricate Tapestry (Delving deep) by Kaetemi</a>: 创作你的音乐</li><li><a href="https://tenor.com/view/money-rain-erlich-bachman-tj-miller-silicon-valley-unicorn-gif-11481689">Money Rain Erlich Bachman Tj Miller Silicon Valley GIF - Money Rain Erlich Bachman Tj Miller - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://gist.github.com/fullstackwebdev/34ccaf0fb79677890c8f93a795f8472a">special_relativity_greg_egan.md</a>: GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://gist.github.com/fullstackwebdev/21de1607d2f3489cf0dd4118b0c1e893#cap0">LoReFT.md</a>: GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://www.youtube.com/watch?v=C84bzu9wXC0">Стыдные вопросы про Северную Корею</a>: ERID: LjN8Jv34w 广告。广告商 ООО "ФЛАУВАУ" ИНН: 9702020445。即使远隔万里也能让亲人开心：https://flowwow.com/s/VDUD15 为母亲节选择礼物...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1227887983546470451)** (8 条消息🔥): 

- **RNN 的复兴**：一篇新论文试图通过一种新兴的混合架构来复兴循环神经网络 (RNN)，承诺对序列数据处理领域进行深入探索。详细论文可以在 [arXiv](https://arxiv.org/abs/2402.19427) 上找到。

- **混合 RNN 可能是终局**：讨论表明，在创新 RNN 架构时，趋势正向混合模型发展，这暗示了创建一个能与当前最先进结果相媲美的纯 RNN 解决方案仍面临持久挑战。

- **Google 的新模型**：有传言称 Google 将发布一个新的 70 亿参数模型，该模型采用了近期研究中描述的基于 RNN 的架构，表明在该领域有大量投资。

- **初创公司评估 AI 模型**：一名成员分享了一篇关于一家专注于测试 AI 模型有效性的新初创公司的 Bloomberg 文章，但链接指向了一个标准的浏览器错误页面，提示 JavaScript 或 cookie 问题。文章链接为 [Bloomberg](https://www.bloomberg.com/news/newsletters/2024-04-11/this-startup-is-trying-to-test-how-well-ai-models-actually-work)。

- **值得引用的社交媒体帖子**：一名成员分享了一个幽默推文的链接，内容是：“感觉很可爱，稍后可能会删。不知道。” 为频道增添了片刻的轻松氛围。推文可在[此处](https://x.com/corbtt/status/1778568618051305850)查看。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2402.19427">Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models</a>: 循环神经网络 (RNN) 具有快速推理能力，并在长序列上高效扩展，但它们难以训练且难以扩展。我们提出了 Hawk，一种具有门控线性循环的 RNN，...</li><li><a href="https://www.bloomberg.com/news/newsletters/2024-04-11/this-startup-is-trying-to-test-how-well-ai-models-actually-work">Bloomberg - Are you a robot?</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2404.07965">Rho-1: Not All Tokens Are What You Need</a>: 以前的语言模型预训练方法对所有训练 Token 统一应用了下一个 Token 预测损失。挑战这一规范，我们假设“语料库中的所有 Token 并非同等重要...”</li><li><a href="https://x.com/corbtt/status/1778568618051305850">Kyle Corbitt (@corbtt) 的推文</a>: 感觉很可爱，稍后可能会删。不知道。
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1227883975184486440)** (369 条消息🔥🔥): 

- **Google 的 Gemma 引擎迎来 C++ 版本**：**Google** 为 **Gemma** 推出了自己的 llama.cpp 变体。一个用于 Google Gemma 模型的轻量级、独立 C++ 推理引擎已在其 [GitHub 仓库](https://github.com/google/gemma.cpp)上线。

- **Nous Research 雄心勃勃**：对话涉及了备受期待的 **Nous Hermes 8x22b** 及其开发困难。**Nous Hermes** 的微调如果尝试进行，将需要耗资约“每周 8 万美元”的基础设施，且依赖于不易租赁的技术。

- **Mac 的 AI 预测市场**：关于 **Apple M 系列芯片** 及其在 AI 推理方面潜力的讨论引起了小组热议，**M2 Ultra** 和 **M3 Max** 因其与 Nvidia GPU 相比具有低功耗和高效率而备受关注。一些人推测未来的 **M4 芯片** 据传将支持高达 2TB 的 RAM。

- **Models on the Move**: 讨论中提到了 **Mixtral-8x22b** 的发布，以及实验性的 **Mistral-22b-V.01**。这是一个稠密的 22B 参数模型，是从一个 MOE 模型中提取出来的，发布在 [Vezora Hugging Face 页面](https://huggingface.co/Vezora/Mistral-22B-v0.1)上。人们对即将发布的 **V.2** 版本充满期待，预计其能力将进一步增强。

- **Fine-Tuning the Giants**: 成员们讨论了提示工程 (Prompt Engineering) 对模型性能的影响，最近的推文表明，通过精心设计的提示，在 ConceptArc 等基准测试和国际象棋 Elo 评分上取得了显著提升。对于 GPT-4 利用该技术能在国际象棋中达到 3500+ Elo 评分的说法，也存在一些怀疑。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/osanseviero/status/1778816205727424884?s=46">来自 Omar Sanseviero (@osanseviero) 的推文</a>：欢迎 Zephyr 141B 加入 Hugging Chat🔥 🎉一个 Mixtral-8x22B 的微调版本 ⚡️使用 TGI 实现极速生成 🤗完全开源（从数据到 UI） https://huggingface.co/chat/models/HuggingFaceH4/zeph...</li><li><a href="https://x.com/karpathy/status/1647278857601564672">来自 Andrej Karpathy (@karpathy) 的推文</a>：@dsmilkov 没关注但听起来很有趣。“训练一个带有样本权重的线性模型来进行类别平衡”...？</li><li><a href="https://www.bloomberg.com/news/articles/2024-04-11/apple-aapl-readies-m4-chip-mac-line-including-new-macbook-air-and-mac-pro">Bloomberg - 你是机器人吗？</a>：未找到描述</li><li><a href="https://foundershub.startups.microsoft.com/signup>">Microsoft for Startups FoundersHub</a>：未找到描述</li><li><a href="https://huggingface.co/Vezora/Mistral-22B-v0.1">Vezora/Mistral-22B-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/lightblue/Karasu-Mixtral-8x22B-v0.1">lightblue/Karasu-Mixtral-8x22B-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/shocked-computer-smile-yellow-smile-surprised-gif-26981337">Shocked Computer GIF - Shocked Computer Smile - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/mistral/mixtral-8x22b-qlora-fsdp.yml">axolotl/examples/mistral/mixtral-8x22b-qlora-fsdp.yml at main · OpenAccess-AI-Collective/axolotl</a>：尽管向 axolotl 提问。通过在 GitHub 上创建账号为 OpenAccess-AI-Collective/axolotl 做出贡献。</li><li><a href="https://huggingface.co/wandb/Mistral-7B-v0.2">wandb/Mistral-7B-v0.2 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/wandb/zephyr-orpo-7b-v0.2">wandb/zephyr-orpo-7b-v0.2 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/docs/transformers/v4.18.0/en/performance">性能与可扩展性：如何容纳更大的模型并加快训练速度</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/LDJnr/Capybara">LDJnr/Capybara · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://github.com/google/gemma.cpp">GitHub - google/gemma.cpp: 适用于 Google Gemma 模型的轻量级、独立 C++ 推理引擎。</a>：适用于 Google Gemma 模型的轻量级、独立 C++ 推理引擎。- google/gemma.cpp</li><li><a href="https://huggingface.co/datasets/HuggingFaceH4/capybara">HuggingFaceH4/capybara · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 速度提升 2-5 倍，显存占用减少 80% 的 QLoRA 和 LoRA 微调</a>：速度提升 2-5 倍，显存占用减少 80% 的 QLoRA 和 LoRA 微调 - unslothai/unsloth</li><li><a href="https://azure.microsoft.com/en-us/pricing/offers/ms-azr-0044p">Azure 免费试用 | Microsoft Azure</a>：开始您的 Microsoft Azure 免费试用，并获得 200 美元的 Azure 额度，可随心使用。运行虚拟机、存储数据并开发应用。
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1227975685402923019)** (25 条消息🔥): 

- **Quest for 7B Mistral Finetuning Advice**: 一位成员咨询了 **7B Mistral 微调的逐步指南**。建议包括使用 *Unsloth 仓库*，以及针对小数据集在 *Colab GPU* 上采用 *QLoRA* 而非全量微调流程，或者从 Vast 等服务商处租用高性能 GPU。

- **Logic Reasoning Dataset Hunt**: 一位成员正在寻找用于自然文本命题逻辑和谓词逻辑推理的数据集。另一位成员分享了 GitHub 上的 [Logic-LLM 项目](https://github.com/teacherpeterpan/Logic-LLM)，并指出该项目相比标准的思维链 (Chain-of-Thought) 提示能带来 18.4% 的性能提升。

- **自由职业者寻求 Finetuning 协助**：一位成员表示有兴趣聘请自由职业者，根据提供的数据集编写脚本或指导他们完成 Finetuning 过程。

- **寻找增强 Genstruct 的 Notebook**：一位成员正在寻找 Notebook 或工具，以便输入抓取的数据作为 Genstruct 的引导，并发现了一个 GitHub 仓库 [OllamaGenstruct](https://github.com/edmundman/OllamaGenstruct)，该仓库非常符合他们的需求。

- **探索 LLM 在医疗保健和医学领域的应用**：成员们讨论了 LLM 在医学领域的应用，分享了论文，并提到了通过此类模型提供医疗建议的潜在法律风险。模型的人为限制和其他法律考量被认为是该领域发展的障碍。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2403.13313">Polaris: A Safety-focused LLM Constellation Architecture for Healthcare</a>：我们开发了 Polaris，这是第一个专注于安全的 LLM 星座架构，用于实时患者与 AI 的医疗对话。与以往专注于问答等任务的医疗 LLM 研究不同，我们的工作...</li><li><a href="https://github.com/edmundman/OllamaGenstruct/blob/main/Paperstocsv.py">OllamaGenstruct/Paperstocsv.py at main · edmundman/OllamaGenstruct</a>：通过在 GitHub 上创建账号来为 OllamaGenstruct 的开发做出贡献。</li><li><a href="https://github.com/teacherpeterpan/Logic-LLM/tree/main">GitHub - teacherpeterpan/Logic-LLM: The project page for &quot;LOGIC-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning&quot;</a>："LOGIC-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning" 的项目页面 - teacherpeterpan/Logic-LLM
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1227948837654626394)** (63 messages🔥🔥): 

- **Worldsim 的 UI 灵感**：一位成员分享了 [edex-ui GitHub 仓库](https://github.com/GitSquared/edex-ui/blob/master/media/screenshot_blade.png)的链接，该仓库展示了一个可定制的科幻终端模拟器。虽然另一位用户表示了兴趣，但有人提醒该项目已停止维护，可能存在安全隐患。
- **期待 Worldsim 的回归**：频道内表现出极大的热情，多位成员讨论了 Worldsim 何时回归以及可能具备的新功能。一位成员得到确认，**Worldsim 平台计划于下周三回归**。
- **AGI 被比作“火辣但疯狂”**：在一个轻松的比喻中，一位成员将危险 UI 的吸引力等同于“火辣但疯狂”的关系。对话随后转向讨论 AGI 的定义，成员们加入了 Claude 3 Opus、AutoGen 和 Figure 01 等不同组件来构思 AGI。
- **关于 Worldsim 回归时间的推测**：成员们对 Worldsim 的回归时间进行了业余预测，有人凭空猜测是周六，也有人利用 Claude 3 的预测，估计时间从即将到来的周六到下周末不等。
- **潜在替代方案和资源说明**：针对 Worldsim 停机期间替代方案的查询，一位用户提到 sysprompt 是开源的，可以直接发送给 Claude，或者与 Anthropic workbench 及其他 LLM 配合使用。此外，感兴趣尝试 Claude 模型的用户可以将 Anthropic API key 粘贴到 **Sillytavern** 中。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://play.aidungeon.com/scenario/9D9o0X3tA8Vb/world-sim">AI Dungeon</a>：未找到描述</li><li><a href="https://kbd.news/OGRE-cyberdeck-1835.html">OGRE cyberdeck</a>：OGRE 是一款末日风格或野外使用的网络终端（cyberdeck），是 Jay Doscher 的 Recover Kit 的仿制品。由 rmw156 分享。</li><li><a href="https://github.com/GitSquared/edex-ui/blob/master/media/screenshot_blade.png">edex-ui/media/screenshot_blade.png at master · GitSquared/edex-ui</a>：一个跨平台、可定制的科幻终端模拟器，具有高级监控和触摸屏支持。 - GitSquared/edex-ui
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1228152382156836864)** (6 messages):

- **新成员加入**：一位新成员表达了通过其他成员邀请发现 CUDA MODE Discord 社区的兴奋之情。
- **视频资源库**：消息提到，与社区相关的录制视频可以在 [CUDA MODE 的 YouTube 频道](https://www.youtube.com/@CUDAMODE)上找到。
- **P2P 增强功能发布**：一项关于通过修改 NVIDIA 驱动为 4090 添加 P2P 支持的公告发布，在 [tinygrad](https://x.com/__tinygrad__/status/1778676746378002712) 的支持下，实现了 tinybox green 上 14.7 GB/s 的 AllReduce。
- **CUDA 挑战赛博客文章**：一位成员分享了他们使用 CUDA 应对“十亿行挑战”（One Billion Row Challenge）的经验和一篇[博客文章](https://tspeterkim.github.io/posts/cuda-1brc)，并邀请 CUDA 爱好者提供反馈。
- **资源共享频道重命名**：有人建议创建一个新频道用于分享资料。随后，一个现有频道被重命名，作为分享资源的场所。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/__tinygrad__/status/1778676746378002712">来自 tiny corp (@__tinygrad__) 的推文</a>：我们通过修改 NVIDIA 驱动为 4090 添加了 P2P 支持。适用于 tinygrad 和 nccl (又名 torch)。在 tinybox green 上实现了 14.7 GB/s 的 AllReduce！</li><li><a href="https://tspeterkim.github.io/posts/cuda-1brc">CUDA 中的十亿行挑战：从 17 分钟到 17 秒</a>：未找到描述
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1227884285852127232)** (168 条消息🔥🔥): 

- **探索 CublasLinear 的速度极限**：成员们讨论了使用自定义 CUDA 库优化模型推理速度的方法，测试表明自定义的 `CublasLinear` 在处理大矩阵乘法时速度更快。然而，在 "llama7b" 等完整模型中测试时，加速效果并不显著，这可能是因为瓶颈在于 attention 而非矩阵乘法。

- **追求快速且准确的量化**：辩论了各种量化策略，例如 4-bit 量化及其与其他量化方法相比的实现挑战。一位成员正在研究一种名为 HQQ (High Quality Quantization) 的方法，旨在通过使用线性反量化（linear dequantization）来超越现有的量化方法。

- **通过驱动破解实现 P2P 支持**：一条消息提到通过修改 NVIDIA 驱动为 RTX 4090 添加了 P2P 支持，并提供了指向详细说明该成果的社交媒体帖子的链接。

- **新语言模型的 CUDA Kernel 愿望清单**：分享了一篇介绍 RecurrentGemma（一种利用 Google Griffin 架构的开源语言模型）的论文，并引发了为其构建 CUDA kernel 的兴趣。

- **基准测试与 Kernel 挑战**：对话详细描述了让 CUDA kernels 实现最佳性能和准确性的挑战，强调了从孤立测试转向完整模型集成时的性能差异，以及改变精度如何导致错误或速度限制等问题。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/__tinygrad__/status/1778676746378002712?s=46&t=ej2aClHUAjeapC55UGHfwg">来自 tiny corp (@__tinygrad__) 的推文</a>：我们通过修改 NVIDIA 驱动为 4090 增加了 P2P 支持。适用于 tinygrad 和 nccl (即 torch)。在 tinybox green 上实现了 14.7 GB/s 的 AllReduce！</li><li><a href="https://huggingface.co/papers/2404.07839">论文页面 - RecurrentGemma: Moving Past Transformers for Efficient Open Language Models</a>：未找到描述</li><li><a href="https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/nn/modules.py#L468C19-L468C30">bitsandbytes/bitsandbytes/nn/modules.py at main · TimDettmers/bitsandbytes</a>：通过为 PyTorch 提供 k-bit 量化来实现可用的 LLM。 - TimDettmers/bitsandbytes</li><li><a href="https://gist.github.com/mobicham/7fb59e825fed0831fccf44752cb21214">hqq_hgemm_benchmark.py</a>：GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://github.com/spcl/QuaRot/blob/main/quarot/kernels/gemm.cu#L32">QuaRot/quarot/kernels/gemm.cu at main · spcl/QuaRot</a>：QuaRot 的代码，实现 LLM 的端到端 4-bit 推理。 - spcl/QuaRot</li><li><a href="https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuBLASDx/multiblock_gemm.cu">CUDALibrarySamples/MathDx/cuBLASDx/multiblock_gemm.cu at master · NVIDIA/CUDALibrarySamples</a>：CUDA 库示例。通过在 GitHub 上创建账号为 NVIDIA/CUDALibrarySamples 做出贡献。</li><li><a href="https://github.com/NVIDIA/CUDALibrarySamples/blob/master/MathDx/cuBLASDx/multiblock_gemm.cu#L97">CUDALibrarySamples/MathDx/cuBLASDx/multiblock_gemm.cu at master · NVIDIA/CUDALibrarySamples</a>：CUDA 库示例。通过在 GitHub 上创建账号为 NVIDIA/CUDALibrarySamples 做出贡献。
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1228046249001881710)** (16 条消息🔥): 

- **ViT 模型的量化困境**：一位成员在尝试量化 `google/vit-base-patch16-224-in21k` 时遇到错误，并分享了相关 [GitHub issue #74540](https://github.com/pytorch/pytorch/issues/74540) 的链接。他们正在寻求量化和剪枝（pruning）技术的解决方案和指导。
- **关于 FlashAttention2 异常输出的争议**：在尝试将 **flashattention2** 与 **BERT** 集成时，一位成员注意到补丁版和未补丁版模型之间的输出差异，后续消息报告相同输入的差异约为 **0.03**。
- **对 LayerNorm 延迟的抱怨**：尽管声称有 3-5 倍的速度提升，一位成员发现来自 **Dao-AILab** 的 **flash-attention** 的融合（fused）LayerNorm 和 MLP 模块比预期慢，这与其 [GitHub 仓库](https://github.com/Dao-AILab/flash-attention/tree/main/training)上的宣传不符。
- **通过硬件黑客手段提升性能**：一位用户提到 Tiny Corp 修改了 NVIDIA 的开源 GPU 内核模块，以在 4090 上启用 P2P，使 AllReduce 操作实现了 **58% 的加速**，更多细节和结果分享在 [Pastebin 链接](https://pastebin.com/ne4ipn6)中。
- **Bitnet 位深忧郁**：为了优化 Bitnet 实现中三值（ternary）权重的存储，一位成员讨论了使用自定义 2-bit 张量代替效率较低的 fp16 方法的可能性，并被引导至 [mobiusml/hqq GitHub 仓库](https://github.com/mobiusml/hqq/blob/master/hqq/core/bitpack.py#L43)中一种潜在的 bitpacking 技术解决方案。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/mobiusml/hqq/blob/master/hqq/core/bitpack.py#L43">hqq/hqq/core/bitpack.py at master · mobiusml/hqq</a>：Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/Dao-AILab/flash-attention/tree/main/training">flash-attention/training at main · Dao-AILab/flash-attention</a>：快速且内存高效的精确 Attention。通过在 GitHub 上创建账号为 Dao-AILab/flash-attention 做出贡献。</li><li><a href="https://github.com/tinygrad/open-gpu-kernel-modules">GitHub - tinygrad/open-gpu-kernel-modules: 支持 P2P 的 NVIDIA Linux 开源 GPU 内核模块</a>：支持 P2P 的 NVIDIA Linux 开源 GPU。通过在 GitHub 上创建账号为 tinygrad/open-gpu-kernel-modules 做出贡献。</li><li><a href="https://github.com/pytorch/pytorch/issues/74540">跨步量化张量缺少工厂函数 · Issue #74540 · pytorch/pytorch</a>：🐛 错误描述：对于非量化张量，既有 empty 也有 empty_strided。然而，对于量化张量，只有函数的 empty 变体。这意味着很难进行.....
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1227909119508353024)** (12 条消息🔥):

- **通过个人节奏缓解 FOMO**：一位成员承认在服务器中看到他人的进步会感到压力，强调了按自己的节奏前进的重要性，并以语言学习作为类比。
- **CUDA 学习曲线与语言熟练度**：在一次轻松的比较中，成员们讨论了学习 CUDA 是否比学习德语更容易。共识似乎表明，这个 Discord 中的许多人认为 CUDA 更简单。

- **PMPP UI 学习小组邀请**：发布了关于 PMPP UI 视频的观看派对和学习小组公告，安排了第一次会议并发布了 [Discord 邀请](https://discord.gg/XwFJRKH9)。发起人愿意在未来的会议中使用现有的语音频道。

- **CUDA 是一个持续的学习曲线吗？**：辩论范围涉及 CUDA 不断演进的复杂性是否比过去学习的静态语言（如德语）更难保持同步。

- **编程并行计算机免费在线课程**：一门大学课程的助教提供了《编程并行计算机》（Programming Parallel Computers）在线课程的[公开版本](https://ppc-exercises.cs.aalto.fi/courses)详情，包括 GPU 编程，并提到了练习的自动基准测试功能。

- **CUDA 作为现有框架的加速工具**：一位成员澄清说，当 Nvidia GPU 可用时，TensorFlow 和 PyTorch 可以调用 CUDA C/C++ 函数，本质上是通过在 GPU 上运行并行计算来充当加速器。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://discord.gg/XwFJRKH9">加入 PMPP UI 讲座时区 Discord 服务器！</a>：查看 Discord 上的 PMPP UI 讲座时区社区 - 与其他 12 名成员一起交流，享受免费的语音和文字聊天。</li><li><a href="https://ppc-exercises.cs.aalto.fi/courses">课程</a>：未找到描述
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1228130632941437120)** (1 条消息): 

- **第 4 章分享**：一位成员提供了一个 [Google Docs 链接](https://docs.google.com/document/d/1b29UvSN2-S8D_UP1xvtSB7nFRc86s6AdWH7n5UieDfE/edit?usp=sharing) 指向文档的第 4 章，供大家查阅和反馈。文档的内容和使用背景未作讨论。
  

---


**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1227939087256649781)** (8 条消息🔥): 

- **数据集交付**：一个超大的数据集被幽默地比作特大号披萨，暗示它已准备好可以使用。
- **任务列表建议**：一位用户建议创建一个后续任务列表，表明需要组织接下来的活动。
- **在 Mamba 上测试**：一位名为 jamesmel 的成员表示，他们需要在名为 Mamba 的系统或组件上进行测试。
- **Infini-attention 介绍**：一篇 [arXiv 论文](https://arxiv.org/abs/2404.07143) 介绍了 **Infini-attention**，这是一种在有限内存和计算量下扩展 Transformers 以处理无限长输入的方法，引发了带有 🔥 反应的热烈讨论。
- **Ring Attention 解释器分享**：shindeirou 分享了一个关于 [Ring Attention](https://coconut-mode.com/posts/ring-attention/) 的解释器链接，旨在让更广泛的受众理解这一概念。这是三位同事的作品，强调了 Large Language Models 中上下文窗口的可扩展性。欢迎大家提供反馈，解释器中的动画也受到了称赞。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://coconut-mode.com/posts/ring-attention/">Ring Attention 详解 | Coconut Mode</a>：语言模型近乎无限的上下文窗口。</li><li><a href="https://arxiv.org/abs/2404.07143">Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention</a>：这项工作介绍了一种有效的方法，可以在有限的内存和计算量下，将基于 Transformer 的 Large Language Models (LLMs) 扩展到无限长的输入。我们提出的方法中的一个关键组件...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1228168834578514001)** (4 条消息): 

- **新著作的开始**：该频道见证了一项合作的开始，旨在编写一本现代版的 **Golub/Van Loan 书籍**，重点关注 GPU 和 Tensor Cores 背景下的数值线性代数。

- **CUDA 兼容性打击**：Nvidia 更新了其 [EULA](https://docs.nvidia.com/cuda/eula/index.html)，禁止使用转换层（translation layers）在非 Nvidia 芯片上运行 CUDA 软件，此举似乎针对 [ZLUDA](https://www.tomshardware.com/news/zluda-project-cuda-intel-gpus) 等项目及某些中国 GPU 制造商。这一变更早在 2021 年就在线发布，但直到最近才被添加到 CUDA 11.6 及更高版本的安装过程 EULA 中。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.tomshardware.com/pc-components/gpus/nvidia-bans-using-translation-layers-for-cuda-software-to-run-on-other-chips-new-restriction-apparently-targets-zluda-and-some-chinese-gpu-makers">Nvidia 禁止使用转换层运行 CUDA 软件 —— 此前该禁令仅列在在线 EULA 中，现在已包含在安装文件中 [更新]</a>：转换层成为众矢之的。</li><li><a href="https://www.amazon.com/Computations-Hopkins-Studies-Mathematical-Sciences/dp/1421407949/ref=pd_lpo_sccl_1/138-9633676-7930953">Matrix Computations (Johns Hopkins Studies in the Mathematical Sciences, 3): Gene H. Golub, Charles F. Van Loan: 9781421407944: Amazon.com: Books</a>：未找到描述
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1227949454301069473)** (11 条消息🔥): 

- **跨 GPU 的 Kernel 基准测试**：基准测试显示 **int4mm kernel** 与其他后端相比速度慢得多，且对权重进行 padding 处理并不会影响速度。在 NVIDIA **3090, 4090 和 A100 GPU** 上进行的测试显示了类似的结果，当前的实现参考了 [mobicham 的 GitHub 仓库](https://github.com/mobiusml/hqq/blob/ao_int4_mm/hqq/core/torch_lowbit.py#L197-L221)。

- **更新的速度评估**：speed-eval 文件已更新，可用于进一步的测试和优化，可通过 [此 GitHub Gist](https://gist.github.com/mobicham/4b08fb0bdf4c3872e5bbf68ec9803137) 获取。

- **Matmul 操作的速度比较**：据报告，8-bit 矩阵乘法比 fp16 快两倍，这表明 **int4 kernel** 在较大 batch sizes 下可能存在性能问题。

- **将 HQQ 集成到 gpt-fast 分支**：gpt-fast 分支现在支持将 HQQ **W_q** 直接转换为 packed int4 格式。根据 [zhxchen17 的 GitHub commit](https://github.com/pytorch-labs/gpt-fast/commit/551af74b04ee1e761736fbccfe98d37137d04176) 详情，成功复现的结果显示，在使用 `--compile` 选项时，每秒处理 200 个 token 的 perplexity (ppl) 为 5.375。

- **HQQ Quant Config 中的优化**：用户在测试 HQQ 时应确保开启 quant_config 中的 **optimize** 设置，以潜在地提高性能。正如关于 HQQ 低比特优化的讨论所述，优化对权重质量的影响因 axis 配置而异。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch-labs/gpt-fast/commit/551af74b04ee1e761736fbccfe98d37137d04176">HQQ 4 bit llama 2 7b · pytorch-labs/gpt-fast@551af74</a>：export MODEL_REPO=meta-llama/Llama-2-7b-hf scripts/prepare.sh $MODEL_REPO python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int4-hqq --groupsize 64 python generate.py --...</li><li><a href="https://github.com/mobiusml/hqq/blob/ao_int4_mm/hqq/core/torch_lowbit.py#L197-L221">hqq/hqq/core/torch_lowbit.py at ao_int4_mm · mobiusml/hqq</a>：Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://gist.github.com/mobicham/4b08fb0bdf4c3872e5bbf68ec9803137">hqq_eval_int4mm_noppl.py</a>：GitHub Gist：即时分享代码、笔记和片段。
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton-viz](https://discord.com/channels/1189498204333543425/1225499141241573447/1228040237205160047)** (3 条消息): 

- **规划下一步**：一名成员建议在集成新功能时**循序渐进**，从**添加代码注释**开始。

- **开发时间**：另一名成员表示打算通过**编写代码**来开始实现提议的想法。

- **关于 Tensor 操作的 GIF 指引**：一名成员指出 **GIF** 中展示的顺序可能存在错误，解释说通常操作是从一个加载到较小 tensor 中的大 tensor 开始的，并对更复杂的内部代码可能带来的并发症表示担忧。
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1227946080549736479)** (98 条消息🔥🔥):

- **通过向量化加载实现 A4000 带宽突破**：在 **A4000** 上使用向量化 (`float4`) 加载和流式加载指令的新方法实现了 **375.7 GB/s 的峰值吞吐量**，表明在维度达到 8192 之前几乎保持相同的速度。该方法通过在维度翻倍时将线程数翻倍，保持每个 SM 的缓存需求一致，确保 L1 cache 持续有效。

- **CUDA 开发从 C 转向 C++**：讨论了在 CUDA 编程中 `C++` 优于 `C` 的实用性，强调了利用更现代的 `C++` 特性（如 `constexpr`、函数重载和模板）来潜在提高代码质量的能力。尽管没有列出具体的直接收益，但共识是由于 nvcc 本身就使用 C++ 编译器，为了提高可维护性，这种转变是合理的。

- **Cooperative Groups 增强 Softmax 性能**：Cooperative Groups 被用于优化 softmax kernel，实现在不使用 shared memory 的情况下在更多线程间进行 reduction，并在必要时利用系统保留的 shared memory。结果显示，引入 `cooperative groups` 后，某些 kernel 的速度提升了约两倍。

- **CUDA 书籍课程大纲过时**：讨论指出，CUDA MODE 中使用的 CUDA 编程书籍没有深入涵盖 Cooperative Groups 等关键特性，尽管这些特性进入 CUDA 已经超过 5 年。该书的一位作者承认了这一点，并同意未来的版本可能会使用 CUDA C++。

- **性能提升与 PR 评审**：在经过包括 `cublasLt`、`TF32` 和 kernel fusion 在内的密集优化后，一位贡献者因在 RTX 4090 上的表现可能超越了 PyTorch 而感到兴奋，详见其 [pull request](https://github.com/karpathy/llm.c/pull/89)。然而，在 A100 上，PyTorch 仍然更快，对比结果为 PyTorch 23.5ms vs 30.2ms。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/pull/89">~2x perf improvement beating PyTorch (cublasLt, TF32, CUDA graphs, kernel fusion, etc…) by ademeure · Pull Request #89 · karpathy/llm.c</a>：这将我的本地 RTX 4090 性能从 ~65ms 提升到 ~34ms（而 PyTorch 需要 ~36ms！）。原始版本：step 1: train loss 4.406481 (耗时 64.890952 ms) 优化后：step 1: train loss 4.406351 (耗时...</li><li><a href="https://github.com/karpathy/llm.c/pull/79/files#diff-a00ef278da39f24a9d5cb4306c15626b921d437013fb5aa60ac2d8df6b5a5508R362)">Include the online softmax CPU code and a fully parallelized GPU kernal by lancerts · Pull Request #79 · karpathy/llm.c</a>：包含 online softmax CPU 代码（源自论文 Online normalizer calculation for softmax）。其原生移植的 GPU kernel 5（用于教学对比）。包含全并行 kernel ...</li><li><a href="https://github.com/apaz-cli/pgen/blob/master/src/list.h">pgen/src/list.h at master · apaz-cli/pgen</a>：一个 PEG 分词器/解析器生成器。通过在 GitHub 上创建账号为 apaz-cli/pgen 开发做贡献。</li><li><a href="https://developer.nvidia.com/blog/cooperative-groups/">Cooperative Groups: Flexible CUDA Thread Programming | NVIDIA Technical Blog</a>：在高效的并行算法中，线程通过协作和共享数据来执行集体计算。为了共享数据，线程必须同步。共享的粒度随算法而异...
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1227893532862451742)** (281 条消息🔥🔥): 

- **关于 Perplexity 集成 GPT-4 的咨询**：用户询问 Perplexity 是否集成了更新版本的 GPT-4 及其在 API 中的可用性。
- **模型重要性：Perplexity vs. Opus**：一位用户认为 Perplexity 不仅仅是一个搜索引擎，而是搜索与图像生成的结合体，建议不应仅局限于搜索功能。
- **关于 API 灵活性的考量**：围绕将 Perplexity API 整合到电子商务网站展开了讨论，用户被引导至 [Perplexity 文档](https://docs.perplexity.ai/)。
- **图像生成查询与挑战**：聊天机器人成员就图像生成、上下文限制以及 GPT-4 Turbo 和 Claude 3 Opus 等 LLM 的有效性交换了意见。
- **在 Perplexity 中使用扩展程序**：对话涉及使用浏览器扩展程序增强 Perplexity 功能，并解决客户端获取（client-side fetching）的限制。
<div class="linksMentioned">

<strong>提到的链接</strong>：

</div>

<ul>
<li>
<a href="https://www.bloomberg.com/news/newsletters/2024-04-11/this-startup-is-trying-to-test-how-well-ai-models-actually-work">Bloomberg - Are you a robot?</a>: 未找到描述</li><li><a href="https://docs.perplexity.ai/">pplx-api</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/perplexity_ai/comments/1bl8kc2/perplexity_limits_the_claude_3_opus_context/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=0O2yTG3n1Vc">I tested the viral Humane AI Pin - it's a Nightmare</a>: 我花费了大量时间，力求让我的视频尽可能简洁、精炼且对你有用 - 如果你想支持我的这一使命，那么...</li><li><a href="https://github.com/JeremiahPetersen/RepoToText">GitHub - JeremiahPetersen/RepoToText: Turn an entire GitHub Repo into a single organized .txt file to use with LLM's (GPT-4, Claude Opus, Gemini, etc)</a>: 将整个 GitHub Repo 转换为单个有序的 .txt 文件，以便与 LLM (GPT-4, Claude Opus, Gemini 等) 配合使用 - JeremiahPetersen/RepoToText</li><li><a href="https://github.com/wallabag/wallabagger/blob/bc9bae830c2f51403b1679efdfab9a497365f05d/wallabagger/js/options.js#L109">wallabagger/wallabagger/js/options.js at bc9bae830c2f51403b1679efdfab9a497365f05d · wallabag/wallabagger</a>: 适用于 wallabag v2 的 Chrome / Firefox / Opera 插件。通过在 GitHub 上创建账号来为 wallabag/wallabagger 的开发做出贡献。</li><li><a href="https://github.com/donoceidon/repo2txt">GitHub - donoceidon/repo2txt: A helper script collecting contents of a repo and placing it in one text file.</a>: 一个收集仓库内容并将其放入一个文本文件的辅助脚本。 - donoceidon/repo2txt</li><li><a href="https://github.com/ollama/ollama">GitHub - ollama/ollama: Get up and running with Llama 2, Mistral, Gemma, and other large language models.</a>: 快速上手并运行 Llama 2, Mistral, Gemma 以及其他大语言模型。 - ollama/ollama
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1227910061930844160)** (12 messages🔥): 

- **探索未知**：用户分享了各种 [Perplexity AI 搜索链接](https://www.perplexity.ai)，探讨的主题从身份不明的（**“What is the Ee0kScAbSrKsblBJxZmtgQ”**）到特定查询，如 **“what is a PH63Fv40SMCGc7mtNDr2_Q”** 以及 **“how to build whMjYrciQM.NXoSLpFSDcQ”**。
- **增强可分享性**：一位用户提醒确保线程是可分享的（Shareable），并提供了一个 [Discord 链接](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825) 作为参考截图，但由于提供的摘要性质，具体细节无法访问。
- **深入技术概念**：几个搜索链接暗示用户正在深入研究技术主题，例如 **“rerank3-cohere-1UdMxh5DStirJf028HLA2g”**、**“Access-logging-should-9h6iZhUOQJ.JYhY8m1.cww”** 以及 **“Learning NIST 800-161-uu.csfXOSlGt5Xi_lc7TeQ”**。
- **关注政策与伦理**：成员们分享了对政策考量的兴趣，相关搜索包括 **美国政府在 “US-is-considering-lJ9faQytRx.6RItBXyKFSQ” 中的审议**，以及在 **“Why honesty is-I6x.NhtaQ5K.BycdYIXwrA”** 中的伦理思考。
- **探索转型中的持久性**：好奇心还延伸到了通过搜索链接 **“what is durable-XjhaEk7uSGi7.iVc01E1Nw”** 探索持久性（durability）的概念，暗示了关于弹性系统或概念的讨论。
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1227921357350895636)** (6 messages): 

- **通过 API 查询 Pro Search**：一位用户询问是否可以在 API 中使用 “Pro Search” 功能，结果被告知**无法**这样做。
- **通过 API 寻求网页版答案**：用户讨论了 API 响应是否能与 PERPLEXITY 网页版匹配；有人建议明确要求 API *“提供你的来源 URL”* 作为获取类似结果的方法。
- **功能路线图查询**：一位用户寻求关于引用功能何时可能实现的信息，并引用了 [官方 PERPLEXITY 文档](https://docs.perplexity.ai/docs/feature-roadmap)。路线图被指出将持续到 6 月，并有各种计划中的更新，但没有明确提到来源引用。

**提到的链接**：<a href="https://docs.perplexity.ai/docs/feature-roadmap">Feature Roadmap</a>: 未找到描述

  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1227881387231547422)** (173 messages🔥🔥):

- **应对上下文长度 (Context Length)：** 用户在使用 **Dolphin Mistral** 等模型时遇到了问题，持续使用会导致某些单词或句子的重复。为了解决这个问题，他们应该调整 **Context Length**，因为典型问题通常在达到模型的上下文限制时出现。

- **应对本地模型限制：** 大家达成共识，像 **CommandR+** 这样复杂的模型对硬件有很高要求，部分用户由于 VRAM 和系统规格的限制无法运行较重的模型，**GPU 升级**和使用 **ngrok** 进行服务器访问被认为是可能的解决方案。

- **LM Studio 工具讨论：** 讨论围绕 LM Studio 的多种功能展开，明确了它**不支持模型的互联网访问**或**拖放文件输入**；不过，文中提供了第三方工具和方法的链接来克服这些限制。

- **模型托管与集成查询：** 用户询问关于在 **Runpod** 和 **GitPod** 等各种平台上托管模型的问题，并咨询将文本生成与 **Stable Diffusion** 等图像生成工具集成的可能性。

- **技术支持交流：** 社区在解决各种系统上的问题方面表现活跃，例如缺少 AVX2 指令和 LM Studio 中的 JavaScript 错误。用户 **heyitsyorkie** 经常提供建议，包括引导至支持频道，并确认在某些修复中**关闭 GPU Offload** 的必要性。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/pmysl/c4ai-command-r-plus-GGUF">pmysl/c4ai-command-r-plus-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://www.bloomberg.com/news/newsletters/2024-04-11/this-startup-is-trying-to-test-how-well-ai-models-actually-work">Bloomberg - Are you a robot?</a>: 未找到描述</li><li><a href="https://github.com/Pythagora-io/gpt-pilot/wiki/Using-GPT%E2%80%90Pilot-with-Local-LLMs">Using GPT‐Pilot with Local LLMs</a>: 第一个真正的 AI 开发者。通过在 GitHub 上创建账号为 Pythagora-io/gpt-pilot 的开发做出贡献。</li><li><a href="https://huggingface.co/search/full-text?q=Command+R%2B>">Full Text Search - Hugging Face</a>: 未找到描述</li><li><a href="https://arxiv.org/html/2404.07143v1">Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention</a>: 未找到描述</li><li><a href="https://www.nvidia.com/en-gb/design-visualization/rtx-a6000/">NVIDIA RTX A6000 Powered by Ampere Architecture | NVIDIA</a>: 开启下一代革命性设计和沉浸式娱乐体验
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1227890468008755262)** (46 条消息🔥): 

- **Mixtral-8x22B 已准备好量化 (Quantization)**：Mixtral-8x22B 模型已使用 [llama.cpp](https://github.com/ggerganov/llama.cpp/) 完成量化，现已开放下载，由于体积庞大，模型被拆分为多个部分。建议用户注意这是一个基础模型 (Base Model)，尚未针对聊天或指令任务进行微调 (Fine-tuned)，在无法运行 8x7b 版本的系统上可能会运行困难。
- **LLM 架构加载错误解决方案**：在尝试使用 **LM Studio 0.2.17** 加载 **Mixtral-8x22B-v0.1** 时，出现了错误消息 "llama.cpp error: 'error loading model architecture: unknown model architecture: '''"；升级到 [0.2.19 beta preview 3](https://lmstudio.ai/beta-releases.html) 或更高版本可解决此问题。
- **发布新的 Chat2DB SQL 模型**：用户 bartowski1182 宣布发布了两个针对 SQL 任务优化的新模型，可在其各自的 Hugging Face 链接下载：[Chat2DB-SQL-7B-GGUF](https://huggingface.co/bartowski/Chat2DB-SQL-7B-GGUF) 和 [Chat2DB-SQL-7B-exl2](https://huggingface.co/bartowski/Chat2DB-SQL-7B-exl2)。
- **大型模型性能讨论**：社区成员分享了他们使用大型模型的经验，讨论了 CMDR+ 和 Mixtral 8x22 的资源消耗情况，并建议尝试更小的量化版本或调整 LM Studio 设置，例如关闭 GPU Offload 且不将模型保留在 RAM 中。
- **升级前检查服务器能力**：在讨论加载 100B+ 参数模型的硬件升级和配置时，强调了具备 AVX2 指令兼容性的重要性，并指出出于性能考虑，服务器应至少配备 24GB VRAM 的 GPU。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/beta-releases.html">LM Studio Beta Releases</a>: 未找到描述</li><li><a href="https://huggingface.co/bartowski/Mixtral-8x22B-v0.1-GGUF">bartowski/Mixtral-8x22B-v0.1-GGUF · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---

**LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1227906718160126034)** (2 条消息): 

- **LLM 在时间序列中的设计局限**：提到除非模型设计发生改变，否则**时间序列数据**并不适合 **Large Language Models (LLM)**。
- **TFT 作为时间序列数据的解决方案**：建议在时间序列数据上训练 **Temporal Fusion Transformer (TFT)** 是一种可行的方法。
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1227893549656178708)** (23 条消息🔥): 

- **无云端 GPU，但 HuggingFace Chat 是替代方案**：Command-R Plus 不支持云端 GPU 服务，但 [HuggingFace Chat](https://huggingface.co/chat) 提供了 [CohereForAI/c4ai-command-r-plus](https://huggingface.co/CohereForAI/c4ai-command-r-plus) 模型作为在线选项。
- **本地运行大模型 vs 云端**：关于运行 72b 模型的可行性讨论表达了对 VRAM 限制的担忧；替代方案包括云端解决方案或本地硬件升级，如 **NVIDIA 4060ti 16GB**。
- **期待用于 AI 的 Apple M4**：传闻即将发布的 [Apple M4 Mac](https://9to5mac.com/2024/04/11/apple-first-m4-mac-release-ai/) 将专注于人工智能应用，这可能需要潜在买家增加预算。
- **AI 应用的内存权衡**：关于将非双通道 RAM 从 16GB 增加到 40GB 是否有利于 LLM 性能的辩论得出结论：尽管会牺牲游戏性能，但对于 AI 任务中的 CPU 推理，拥有更多 RAM 即使失去双通道优势也是有益的。
- **GPU vs CPU 推理**：讨论强调 CPU 推理的速度明显慢于 GPU 推理，虽然拥有更多系统内存可以加载更大的 LLM，但最终目标仍是实现全 GPU 推理以获得最佳性能。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://9to5mac.com/2024/04/11/apple-first-m4-mac-release-ai/">Apple 计划今年发布首批搭载 M4 芯片的 Mac，重点关注 AI - 9to5Mac</a>: 随着 M4 系列芯片的到来，Apple 正计划对其 Mac 产品线进行重大改革。据...</li><li><a href="https://huggingface.co/chat">HuggingChat</a>: 让社区最好的 AI 聊天模型惠及每一个人。</li><li><a href="https://github.com/tinygrad/open-gpu-kernel-modules">GitHub - tinygrad/open-gpu-kernel-modules: 支持 P2P 的 NVIDIA Linux 开源 GPU 内核模块</a>: 支持 P2P 的 NVIDIA Linux 开源 GPU 内核模块。通过在 GitHub 上创建账号来为 tinygrad/open-gpu-kernel-modules 的开发做出贡献。
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1227924747040325663)** (9 条消息🔥): 

- **Ubuntu 上的 Mistral 模型加载问题**：一位用户在 Ubuntu 22.04 LTS 上尝试使用其服务器加载本地模型时，遇到了 **Mistral 模型** 的 *“Error loading model”* 错误。用户分享了系统规格（如可用 RAM 和 GPU 详情），以寻求对模型加载过程中遇到的 **Exit code: 0** 错误的解释。

- **BERT Embedding 查询与指南**：一名成员询问是否可以加载 Google BERT Embedding 模型，随后讨论中分享了 LM Studio 文档中关于 [文本 Embedding](https://lmstudio.ai/docs/text-embeddings) 的链接，解释了如何使用 GGUF 格式的 LM Studio Embedding 服务器为 RAG 应用生成文本 Embedding。

- **关于 Google BERT 和微调的说明**：另一位用户澄清说，基础的 **Google BERT 模型** 无法在 LM Studio 中直接使用，且通常不适合在没有针对下游任务进行微调的情况下直接使用，并引用了来自 [Hugging Face 的模型](https://huggingface.co/google-bert/bert-base-uncased)。

- **更好 Embedding 模型的推荐**：进一步推荐了具有更大参数的 **Embedding 模型**，如 `mxbai-large`、`GIST-large` 和 `LaBSE`，以获得优于标准 BERT base 模型的结果。

- **Embedding 的计算成本选项**：关于适合不同计算能力的各种 Embedding 模型的评论指出，除了 1024 维的 `large` 版本外，还有 768 维和 384 维的 `base` 和 `small` 版本作为替代方案。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/docs/text-embeddings">文本 Embedding | LM Studio</a>: 文本 Embedding 正处于 Beta 测试阶段。从此处下载支持该功能的 LM Studio。</li><li><a href="https://huggingface.co/collections/ChristianAzinn/embedding-ggufs-6615e9f216917dfdc6773fa3">Embedding GGUFs - ChristianAzinn 收藏集</a>: 未找到描述
</li>
</ul>

</div>
  

---

**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1228016561793798234)** (12 messages🔥): 

- **Linux 用户询问关于 ROCm 的信息**：一位用户询问是否会为 Linux 提供 **amd-rocm-tech-preview**，另一位用户表示最终会有，但不会很快推出。
- **用户在支持 ROCm 的硬件上的使用体验**：多位用户报告了在不同硬件（特别是 7800XT、7900 XTX Nitro+ 和 6800XT）上运行 ROCm 的体验。他们分享说，运行任务时会产生明显的电感啸叫（coil whine），具体情况因游戏或工作负载而异。
- **AMD 6750XT 上的技术预览版挑战**：一位用户指出，ROCm **tech preview** 声称在 AMD 6750XT 上使用了 GPU，但最终只利用了 CPU 和 RAM，且没有抛出任何兼容性错误。他们将其与常规 Studio 进行了对比，后者能通过 AMD OpenCL 正确地将任务卸载到 GPU。
- **寻求 Windows 二进制文件编译帮助**：一名成员寻求在 **Windows 上编译 gguf-split 二进制文件** 的帮助，以便在 7900XT 上进行测试，并链接了与该问题相关的 GitHub 讨论和 Pull Request：[如何使用 `gguf-split` / 模型分片演示](https://github.com/ggerganov/llama.cpp/discussions/6404) 以及 [添加 Command R Plus 支持](https://github.com/ggerganov/llama.cpp/pull/6491)。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/ggerganov/llama.cpp/discussions/6404#discussioncomment-9090672">How to use the `gguf-split` / Model sharding demo · ggerganov/llama.cpp · Discussion #6404</a>: 分发和存储 GGUF 对于 70b+ 模型来说很困难，尤其是在 f16 精度下。文件传输过程中可能会发生很多问题，例如：临时磁盘空间不足、网络中断。通常情况下...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6491#issuecomment-2050966545">Add Command R Plus support by Carolinabanana · Pull Request #6491 · ggerganov/llama.cpp</a>: 更新了张量映射，为 GGUF 转换添加了 Command R Plus 支持。
</li>
</ul>

</div>
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1227891320412962836)** (96 messages🔥🔥): 

- **探索 Encoder 模型的上下文窗口扩展**：成员们讨论了将用于扩展 Decoder-only 模型上下文窗口的方法适配到 BERT 等 Encoder 模型的挑战，并提到了由于双向注意力机制（bidirectional attention mechanisms）带来的困难。他们还讨论了使用 [FlashAttention](https://github.com/Dao-AILab/flash-attention) 的 [MosaicBERT](https://mosaicbert.github.io/)，并疑惑为什么在 Hugging Face 的 Transformers 等库中没有更普遍地实现它，尽管已经有了[社区贡献](https://github.com/huggingface/transformers/issues/26350)。

- **Mixtral-8x22B 模型的量化与 VRAM 担忧**：一位成员寻求社区支持，希望在 GPU 服务器上运行 Mixtral-8x22B 模型的 2-bit 量化版本，以便在小于 72 GB 的 VRAM 环境下使用。大家对 [AQLM 团队的进展](https://www.mosaicml.com/blog/mpt-7b) 充满期待，这可能需要一周时间。

- **探索 The Pile 数据集大小差异**：用户分享了下载和使用 The Pile 数据集的经验，注意到报告的 886GB 未压缩大小与他们 720GB 到 430GB 不等的压缩副本之间存在差异，并讨论了 The Pile 中不同存档类型缺乏提取代码的问题。

- **创建 AI 学习与语言模型开发的阅读清单**：一位成员分享了一个 [GitHub 仓库](https://github.com/elicit/machine-learning-list)，其中包含一份阅读清单，旨在帮助 Elicit 的新人学习语言模型，内容涵盖从基础 Transformer 操作到最新进展。

- **EleutherAI 的贡献与公开模型**：一位成员强调了 [EleutherAI](https://golden.com/wiki/EleutherAI-Y3V4AA4) 对 AI 发展的贡献，并提到了 GPT-J 和 NeoX 等公开模型。讨论还涉及了维基页面的一项新功能，即使用 EleutherAI 等来源生成的 AI 内容。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2">Learning Agency Lab - Automated Essay Scoring 2.0 | Kaggle</a>：未找到描述</li><li><a href="https://mosaicbert.github.io/">MosaicBERT: A Bidirectional Encoder Optimized for Fast Pretraining</a>：未找到描述</li><li><a href="https://golden.com/wiki/EleutherAI-Y3V4AA4">EleutherAI</a>：EleutherAI 是一家非营利性 AI 研究实验室，专注于大型 AI 模型的解释性（interpretability）和对齐（alignment）。</li><li><a href="https://docs.google.com/document/d/1qt7GjbrFToxSIUKC9nWccvHZN9LnO8r6myMAFUX5SVQ/edit?usp=sharing">[List of evals that you&#39;d like us/you to work on/explore/solve]</a>：未找到描述</li><li><a href="https://github.com/elicit/machine-learning-list">GitHub - elicit/machine-learning-list</a>：通过在 GitHub 上创建账户来为 elicit/machine-learning-list 的开发做出贡献。</li><li><a href="https://github.com/huggingface/transformers/issues/26350">Community contribution: Adding Flash Attention 2 support for more architectures · Issue #26350 · huggingface/transformers</a>：功能请求。Flash Attention 2 是一个提供注意力操作内核的库，用于实现更快、更显存高效的推理和训练：https://github.com/Dao-AILab/flash-attention Le...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1227917024647188541)** (93 条消息🔥🔥): 

- **Google 发布 Mixture of Depths 模型**：@_akhaliq 的一条 [推文](https://x.com/_akhaliq/status/1775740222120087847?t=55VlAx9tjP9PUgvRcnIfMQ&s=33) 透露，Google 展示了 Mixture-of-Depths，旨在 Transformer 架构的语言模型中动态分配计算资源，而不是在输入序列中均匀分布 FLOPs。
- **RULER 的空仓库现已开源**：开源社区现在可以访问 **RULER** 的空仓库，该项目承诺提供关于**长上下文语言模型真实上下文大小**的见解，详见 [GitHub](https://github.com/hsiehjackson/RULER)。
- **对抗样本：超越噪声与畸变**：讨论涵盖了对抗样本（adversarial examples）并不总是无结构的噪声，有些表现为图像部分的实际畸变。这种复杂性在拥有 1000 次引用的 [ImageNet-A 和 ImageNet-O 数据集论文](https://arxiv.org/abs/1907.07174) 中有进一步详细说明。
- **微调部分层可以很高效**：关于*子集微调（subset finetuning）*的新兴讨论受到关注，即微调网络层的子集可以达到与全量微调相当的准确度，尤其是在训练数据稀缺的情况下，正如 [这篇论文](https://arxiv.org/abs/2404.07839) 所指出的。
- **经济且具竞争力的 LLM JetMoE-8B**：JetMoE-8B 是一款新型经济实惠的 LLM，训练成本低于 10 万美元，采用稀疏门控 Mixture-of-Experts 架构。据称其性能优于同等规模的其他模型，标志着开源模型迈出了重要一步。详情可见 [此处](https://arxiv.org/abs/2404.07413)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.07839">RecurrentGemma: Moving Past Transformers for Efficient Open Language Models</a>: 我们介绍了 RecurrentGemma，这是一个使用 Google 新颖的 Griffin 架构的开源语言模型。Griffin 将线性递归与局部注意力相结合，在语言任务上实现了卓越的性能...</li><li><a href="https://arxiv.org/abs/2404.07177">Scaling Laws for Data Filtering -- Data Curation cannot be Compute Agnostic</a>: 视觉语言模型 (VLMs) 在精心策划的网络数据集上经过数千小时的 GPU 训练。近年来，随着多项研究开发出数据筛选策略，数据策划分选变得日益重要...</li><li><a href="https://arxiv.org/abs/2404.07413">JetMoE: Reaching Llama2 Performance with 0.1M Dollars</a>: 大语言模型 (LLMs) 取得了显著成果，但其日益增长的资源需求已成为开发强大且易于获取的超人智能的主要障碍...</li><li><a href="https://arxiv.org/abs/1907.07174">Natural Adversarial Examples</a>: 我们介绍了两个具有挑战性的数据集，它们能可靠地导致机器学习模型性能大幅下降。这些数据集是通过一种简单的对抗过滤技术收集的，旨在创建...</li><li><a href="https://x.com/_akhaliq/status/1775740222120087847?t=55VlAx9tjP9PUgvRcnIfMQ&s=33">Tweet from AK (@_akhaliq)</a>: Google 发布 Mixture-of-Depths：在基于 Transformer 的语言模型中动态分配计算。基于 Transformer 的语言模型在输入序列上均匀分布 FLOPs。在这项工作中，我们...</li><li><a href="https://arxiv.org/abs/2302.06354">Less is More: Selective Layer Finetuning with SubTuning</a>: 微调预训练模型已成为在新任务上训练神经网络的标准方法，从而实现快速收敛和性能提升。在这项工作中，我们研究了一种替代方案...</li><li><a href="https://distill.pub/2019/activation-atlas/">Activation Atlas</a>: 通过使用特征反演来可视化图像分类网络中的数百万个激活，我们创建了一个可探索的激活图谱 (Activation Atlas)，展示了网络学习到的特征以及它所代表的概念...</li><li><a href="https://arxiv.org/abs/2310.05209">Scaling Laws of RoPE-based Extrapolation</a>: 基于旋转位置嵌入 (RoPE) 的大语言模型 (LLMs) 的外推能力是目前备受关注的话题。解决外推问题的主流方法是...</li><li><a href="https://arxiv.org/abs/2310.17041">On Surgical Fine-tuning for Language Encoders</a>: 微调预训练神经语言编码器的所有层（无论是使用全部参数还是使用参数高效方法）通常是将其适配到新任务的默认方式。我们展示了...</li><li><a href="http://arxiv.org/abs/2106.10151">The Dimpled Manifold Model of Adversarial Examples in Machine Learning</a>: 深度神经网络在输入受到微小扰动时的极端脆弱性，在 2013 年由多个研究小组独立发现。然而，尽管付出了巨大努力...</li><li><a href="https://github.com/hsiehjackson/RULER">GitHub - hsiehjackson/RULER: This repo contains the source code for RULER: What’s the Real Context Size of Your Long-Context Language Models?</a>: 此仓库包含 RULER 的源代码：你的长上下文语言模型的真实上下文大小是多少？ - hsiehjackson/RULER</li><li><a href="https://web.archive.org/web/20220616155437/https://james-simon.github.io/deep%20learning/2020/08/31/multiplicative-neural-nets">Multiplicative neural networks</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1228228810315661423)** (11 条消息🔥):

- **关于大规模运行中浮点精度的查询**：一位用户询问了当前最大规模训练运行中使用的浮点格式，想知道 bf16 是否仍是标准，或者实验室是否已转向 fp8。
- **分享了关于数据过滤的 Scaling Laws 论文**：CVPR2024 上发表的一篇新 [Scaling Laws 论文](https://arxiv.org/abs/2404.07177) 探讨了数据清洗（data curation）与计算资源之间的相互作用。它提出数据清洗**不能脱离计算资源（compute agnostic）**，并介绍了处理异构且有限的网络数据的 Scaling Laws。
- **对论文反应平淡**：一位成员用一个非语言表情符号回复，似乎暗示对分享的 Scaling Laws 论文缺乏兴奋感或持怀疑态度。
- **在研究方法中寻找熵**：讨论继续，一位成员象征性地提到了在 Scaling Laws 背景下寻找基于熵的方法。另一位用户认可了这一主题，并指出所采用的是经验方法，没有明确提到熵。
- **思考新研究的基础**：成员们反思了当前的研究（如 Scaling Laws 论文）即使没有直接说明，也可能隐含地基于熵等经典概念。他们讨论了该论文将熵等概念重新定义为“效用（utility）”的方法细微差别，这导致了非传统的分析视角。

**提到的链接**：<a href="https://x.com/pratyushmaini/status/1778577153107570770">Pratyush Maini (@pratyushmaini) 的推文</a>：1/ 🥁数据过滤的 Scaling Laws 🥁 TLDR：数据清洗 *不能* 脱离计算资源！在我们的 #CVPR2024 论文中，我们为异构和有限的网络数据开发了第一个 Scaling Laws。w/@goyalsach...

---

**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1228056988588572672)** (8 messages🔥): 

- **对将 GitHub stars 作为指标感到惊讶**：一位成员对使用 GitHub stars 作为衡量标准表示惊讶，提到他们遇到过只有少量 stars 的优秀项目，以及“拥有 10k+ stars 的绝对软件垃圾”。
- **对 Activation to Parameter 的兴趣**：AtP(*) 的概念因其潜在用途和价值引起了一位成员的兴趣。
- **利用 AtP 进行异常检测的潜力**：人们对利用 AtP* 进行异常检测感到好奇，特别是通过对比单次前向传播与多次其他传播的结果来确定异常。
- **AtP 分析的新方法**：与论文中平均效果的方法不同，该成员建议通过比较单个前向传播结果来识别离群值（outliers）。

---

**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/)** (1 messages): 

butanium: 我实验室里也有人想知道那些 chat_template 分支是否可用。

---

**Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1228274522034012200)** (10 messages🔥): 

- **贡献需要企业级 CLA**：一位成员提到，为了推进 TE 的 **fused kernels 和 fp8** 集成，需要 **企业贡献者许可协议（Corporate CLA）**。目前的 EleutherAI/GPT-NeoX CLA 仅针对个人。
- **编写自定义企业 CLA**：作为回应，另一位成员提出可以 **编写自定义 CLA**，并询问了具体的需求列表以及相对于当前 CLA 需要进行的更改。
- **NeoX Embeddings 引发疑问**：一位分析 Embeddings 的成员注意到 **NeoX** 似乎是一个离群值，并怀疑是否未对其输入 Embeddings 应用 **weight decay**，或者是否使用了其他特定技巧。
- **比较 Pythia 和 NeoX Embeddings**：在询问 **NeoX** 模型的异常行为是否也存在于 **Pythia** 中之后，另一位成员决定对两者进行检查。
- **确认 NeoX 的独特行为**：经过一些分析，确认 NeoX 在其 Embeddings 方面是一个独特的离群值，其 `model.gpt_neox.embed_in.weight[50277:].sum(axis=1)` 不接近 0，这与 **GPT-J** 和 **OLMo** 等其他模型不同。

---

**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1227947166903828654)** (2 messages): 

- **Mixtral 8x22B:free 推出并更新**：为了与 **现有的 :free 模型** 保持一致并澄清路由/频率限制（rate-limiting）的混淆，创建了一个新模型 [Mixtral 8x22B:free](https://openrouter.ai/models/mistralai/mixtral-8x22b:free)。它还更新了上下文大小，从 **64,000 增加到 65,536**。
- **需要从已停用的免费模型切换**：[Mixtral 8x22B:free](https://openrouter.ai/models/mistralai/mixtral-8x22b:free) 已被 **停用**，建议用户切换到 [Mixtral 8x22B](https://openrouter.ai/models/mistralai/mixtral-8x22b)。

- **用于测试的新实验模型**：两个新的 Mixtral 8x22B instruct 微调版本已开放测试：[Zephyr 141B-A35B](https://openrouter.ai/models/huggingfaceh4/zephyr-orpo-141b-a35b) 和 [Fireworks: Mixtral-8x22B Instruct (preview)](https://openrouter.ai/models/fireworks/mixtral-8x22b-instruct-preview)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/models/mistralai/mixtral-8x22b:free>).">Mixtral 8x22B by mistralai | OpenRouter</a>：Mixtral 8x22B 是来自 Mistral AI 的大规模语言模型。它由 8 个专家组成，每个专家有 220 亿参数，每个 token 每次使用 2 个专家。它通过 [X](https://twitter... 发布</li><li><a href="https://openrouter.ai/models/mistralai/mixtral-8x22b>)">Mixtral 8x22B by mistralai | OpenRouter</a>：Mixtral 8x22B 是来自 Mistral AI 的大规模语言模型。它由 8 个专家组成，每个专家有 220 亿参数，每个 token 每次使用 2 个专家。它通过 [X](https://twitter... 发布</li><li><a href="https://openrouter.ai/models/huggingfaceh4/zephyr-orpo-141b-a35b>)">Zephyr 141B-A35B by huggingfaceh4 | OpenRouter</a>：Zephyr 141B-A35B 是一个混合专家 (MoE) 模型，总参数量为 141B，激活参数量为 35B。在公开可用的合成数据集混合上进行了微调。它是 ... 的 instruct 微调版本</li><li><a href="https://openrouter.ai/models/fireworks/mixtral-8x22b-instruct-preview>)">Mixtral-8x22B Instruct OH by fireworks | OpenRouter</a>：Fireworks Mixtral 8x22b Instruct 是来自 Mistral 的最新 MoE 模型 [Mixtral 8x22B](/models/mistralai/mixtral-8x22b) 的第一个 instruct 微调版本。该模型在约 10K 条目上进行了微调 ...
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/)** (1 条消息): 

.o.sarge.o.: 尝试购买 token 时似乎出现了问题。这是图片。
  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1227877783229698082)** (213 条消息🔥🔥): 

- **登录问题已自主解决**：一位用户错误地登录了平台，随后找到了不仅可以退出登录，还可以彻底注销账户的解决方案。

- **GPT 4 Turbo 和 Mistral Large 错误**：对 GPT 4 Turbo 和 Mistral Large 的 `500 errors` 进行排障后发现，重新部署到 Heroku 解决了该问题，这表明部署损坏可能是原因所在。

- **个人 AI 系统搭建讨论**：社区成员讨论了使用 OpenRouter 和其他工具（如 [LibreChat](https://github.com/danny-avila/LibreChat)）搭建个人 AI 系统，并为个性化 AI 体验提供了建议，包括移动端和桌面端的可用性、对话存储以及低延迟的网页搜索结果。

- **Firextral-8x22B-Instruct 更新与说明**：讨论了 Firextral-8x22B-Instruct 等模型的路由更新，切换到了 Vicuna 1.1 模板，并澄清了 OpenRouter 网站上最大上下文列表显示为 "Max Output" 的情况。

- **AI 模型性能与调优经验分享**：用户分享了他们对各种模型性能和调优能力的经验与观察。观点各异，有人在某些任务中更青睐 [GPT-4 Turbo](https://openrouter.ai/playground?models=openai/gpt-4-turbo,mistralai/mistral-large)，有人对 MoE 架构表现出兴趣，并讨论了 Opus、Gemini Pro 1.5 的作用以及模型的涌现行为 (emergent behavior)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/fireworks-ai/mixtral-8x22b-instruct-oh">fireworks-ai/mixtral-8x22b-instruct-oh · Hugging Face</a>: 未找到描述</li><li><a href="https://www.bloomberg.com/news/newsletters/2024-04-11/this-startup-is-trying-to-test-how-well-ai-models-actually-work">Bloomberg - Are you a robot?</a>: 未找到描述</li><li><a href="https://docs.together.ai/docs/function-calling">Function calling</a>: 未找到描述</li><li><a href="https://deepinfra.com/docs/advanced/function_calling">Use Function Calling with Deep Infra endpoints | ML Models | Deep Infra</a>: 查找关于在 Deep Infra 端点使用 Function Calling、集成等相关信息！</li><li><a href="https://openrouter.ai/playground?models=openai/gpt-4-turbo,mistralai/mistral-large">OpenRouter</a>: LLM 和其他 AI 模型的路由</li><li><a href="https://openrouter.ai/models/cohere/command-r-plus">Command R+ by cohere | OpenRouter</a>: Command R+ 是来自 Cohere 的新型 104B 参数 LLM。它适用于角色扮演、通用消费者用例和检索增强生成 (RAG)。它为十种主要语言提供多语言支持...</li><li><a href="https://docs.librechat.ai/install/index.html">Installation and Configuration</a>: 💻 关于安装和配置的深入指南</li><li><a href="https://github.com/danny-avila/LibreChat">GitHub - danny-avila/LibreChat: Enhanced ChatGPT Clone: Features OpenAI, Assistants API, Azure, Groq, GPT-4 Vision, Mistral, Bing, Anthropic, OpenRouter, Google Gemini, AI model switching, message search, langchain, DALL-E-3, ChatGPT Plugins, OpenAI Functions, Secure Multi-User System, Presets, completely open-source for self-hosting. More features in development</a>: 增强版 ChatGPT 克隆：具有 OpenAI、Assistants API、Azure、Groq、GPT-4 Vision、Mistral、Bing、Anthropic、OpenRouter、Google Gemini、AI 模型切换、消息搜索、langchain、DALL-E-3、ChatGPT 插件、OpenAI Functions、安全多用户系统、预设，完全开源可自托管。更多功能开发中。</li><li><a href="https://discord.gg/uDyZ5Tzhct">Join the LibreChat Discord Server!</a>: LibreChat 社区，一个开源、通用的 AI 聊天 Web UI，具有无缝自托管和活跃开发。| 3365 名成员
</li>
</ul>

</div>
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1227961028554264688)** (9 messages🔥): 

- **Mojo 回馈社区**：Mojo 最近开源了其标准库，现在包含了来自已合并 Pull Request 的社区贡献。此举允许社区参与贡献，并让 Mojo 团队有更多时间专注于编译器。

- **寻求 Modular 合作**：来自 [BackdropBuild.com](https://backdropbuild.com/) 的一位成员正在寻求协助，以将 Modular 集成到他们的大规模开发者队列计划中。他们正在联系 Modular 进行合作，以支持使用其技术的构建者。

- **保持在正确频道**：提醒商务和合作咨询应定向到 offtopic 频道，以便在 general 讨论中实现更好的组织和相关性。

**提及链接**：<a href="https://backdropbuild.com/">Backdrop Build</a>：我们共同构建 - 在短短 4 周内与数百名其他出色的构建者一起将疯狂的想法变为现实。

  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot: 来自 *Modular*:
<https://twitter.com/Modular/status/1778482233957101869>
  

---


**Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1228017366504312963)** (1 messages): 

- **Mojo 解决矩阵存储问题**：[Modular 官网](https://www.modular.com/blog/row-major-vs-column-major-matrices-a-performance-analysis-in-mojo-and-numpy)上的一篇新博客文章深入探讨了矩阵在内存中的存储方式，探索了行优先（row-major）和列优先（column-major）排序的区别和性能影响。这项调查旨在阐明为什么不同的编程语言和库对存储顺序有不同的偏好。

**提及链接**：<a href="https://www.modular.com/blog/row-major-vs-column-major-matrices-a-performance-analysis-in-mojo-and-numpy">Modular: Row-major vs. column-major matrices: a performance analysis in Mojo and NumPy</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：Mojo 和 NumPy 中的行优先与列优先矩阵性能分析。

  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1227882548323946526)** (125 messages🔥🔥): 

- **Mojo 中讨论 Karpathy 的 llm.c**：关于 Andrej Karpathy 的 `llm.c` 仓库为何不使用 Mojo 的 GitHub [issue](https://github.com/karpathy/llm.c/issues/28) 引起了关注。Andrej Karpathy 表示，他很乐意在 readme 中链接任何 Mojo 移植版本，以便进行基准测试和对比。

- **Mojo 中的二进制文件读取**：成员们讨论了如何在 Mojo 中实现类似于 Python `struct.unpack` 的二进制文件读取器。提供的一个解决方案是使用 Mojo 的 `read` 而不是 `read_bytes`，这似乎解决了问题，正如 [GitHub 文件](https://github.com/tairov/llama2.mojo/blob/master/llama2.mojo) 中所示。

- **GUI 设计理念的冲突**：围绕 GUI 框架的对话引发了对设计方法的不同意见，重点是 model/view 范式。一些成员表现出对 SwiftUI 等声明式 GUI 的偏好，而另一些成员则捍卫 Tk 等命令式框架提供的灵活性和控制力。

- **Mojo 增强 Python 性能的潜力**：社区对 Mojo 的未来表达了热情，特别是它增强 Python 性能以及可能允许直接进行 Python 代码编译的潜力。分享了一个相关的播客链接，其中 Chris Lattner 讨论了 [Mojo 的目标](https://youtu.be/pdJQ8iVTwj8?si=ML7lZfXAel9zEgj0&t=5763)。

- **比较 Mojo 和 C 之间的功能**：讨论了关于 Mojo 模拟 C 语言功能（如位操作）能力的问题，成员们分享了翻译示例，并确认了 Mojo 中移位（shifts）和按位异或（bitwise XOR）等操作的功能。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com">GitHub: Let’s build from here</a>：GitHub 是超过 1 亿开发者共同塑造软件未来的地方。为开源社区做贡献，管理你的 Git 仓库，像专业人士一样审查代码，跟踪错误和功能...</li><li><a href="https://en.wikipedia.org/wiki/Xorshift#xorshift.2A">Xorshift - Wikipedia</a>：未找到描述</li><li><a href="https://github.com/tairov/llama2.mojo/blob/master/llama2.mojo">llama2.mojo/llama2.mojo at master · tairov/llama2.mojo</a>：纯 🔥 单文件推理 Llama 2。通过在 GitHub 上创建账号来为 tairov/llama2.mojo 的开发做贡献。</li><li><a href="https://github.com/modularml/mojo/issues/1625)">Issues · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做贡献。</li><li><a href="https://github.com/Moosems/bazingo/blob/master/bazingo/misc.py#L19-L169">bazingo/bazingo/misc.py at master · Moosems/bazingo</a>：通过在 GitHub 上创建账号来为 Moosems/bazingo 的开发做贡献。</li><li><a href="https://github.com/Akuli/porcupine">GitHub - Akuli/porcupine: A decent editor written in tkinter</a>：一个用 tkinter 编写的体面编辑器。通过在 GitHub 上创建账号来为 Akuli/porcupine 的开发做贡献。</li><li><a href="https://github.com/karpathy/llm.c/issues/28">Why not Mojo? · Issue #28 · karpathy/llm.c</a>：严肃的问题。如果你要深入底层，Mojo 提供了潜在的巨大加速，并且该语言将从这项工作中显著受益。无论如何 - 热爱这项工作。谢谢...</li><li><a href="https://youtu.be/pdJQ8iVTwj8?si=ML7lZfXAel9zEgj0&t=5763">Chris Lattner: Future of Programming and AI | Lex Fridman Podcast #381</a>：Chris Lattner 是一位传奇的软件和硬件工程师，曾在 Apple, Tesla, Google, SiFive 和 Modular AI 领导项目，包括开发 S...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1228071956432289832)** (2 messages): 

- **Mojo 终端文本渲染预览**：一位成员展示了 **使用 Mojo 进行终端文本渲染** 并分享了代码，灵感来自 `charmbracelet’s lipgloss` 包。[预览代码可在 GitHub 上找到](https://github.com/thatstoasty/mog/blob/main/examples/readme/layout.mojo)，其中一个小的状态栏问题很快将得到修复。
- **为 Basalt 集成喝彩**：另一位社区成员称赞了 **Basalt** 的集成，认为终端渲染效果令人印象深刻。

**提及的链接**：<a href="https://github.com/thatstoasty/mog/blob/main/examples/readme/layout.mojo">mog/examples/readme/layout.mojo at main · thatstoasty/mog</a>：通过在 GitHub 上创建账号来为 thatstoasty/mog 的开发做贡献。

  

---


**Modular (Mojo 🔥) ▷ #[community-blogs-vids](https://discord.com/channels/1087530497313357884/1151418796993683477/1228387825938989188)** (1 messages): 

- **探索矩阵中的内存存储**：一篇题为 ["Row-major vs. Column-major matrices: A performance analysis in Mojo and Numpy"](https://www.modular.com/blog/row-major-vs-column-major-matrices-a-performance-analysis-in-mojo-and-numpy) 的博客文章深入探讨了行优先（row-major）和列优先（column-major）排序。它讨论了性能影响以及不同语言和库对矩阵内存存储的偏好。

- **Matrix Memory Order Notebook 错误**：一位成员在尝试按照博客文章操作时，运行 GitHub 上相关 Jupyter notebook 的第二个单元格时遇到错误，该 notebook 位于 [devrel-extras/blogs/mojo-row-major-column-major](https://github.com/modularml/devrel-extras/blob/main/blogs/mojo-row-major-column-major/row_col_mojo.ipynb)。错误涉及在创建 MojoMatrix 并将其转换为列优先（column-major）格式时，与 'mm_col_major' 相关的未知声明。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.modular.com/blog/row-major-vs-column-major-matrices-a-performance-analysis-in-mojo-and-numpy">Modular: Row-major vs. column-major matrices: a performance analysis in Mojo and NumPy</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：Row-major vs. column-major matrices: a performance analysis in Mojo and NumPy</li><li><a href="https://github.com/modularml/devrel-extras/blob/main/blogs/mojo-row-major-column-major/row_col_mojo.ipynb">devrel-extras/blogs/mojo-row-major-column-major/row_col_mojo.ipynb at main · modularml/devrel-extras</a>：包含开发者关系博客文章、视频和研讨会的配套材料 - modularml/devrel-extras
</li>
</ul>

</div>
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1227898729206382594)** (107 messages🔥🔥): 

- **在 AIAssistant 中追踪 Token 使用情况**：一位成员正在通过接收 Token 计数、乘以价格并保存数据来追踪他们自己来自 OpenAI API 的 Token 使用情况，因为 [**LangSmith** 不为 AIAssistant 提供 Token 使用情况](http://langsmith.langchain.com/)。

- **总结速度瓶颈**：成员们讨论了 **LangChain 的 `load_summarization_chain` 函数**，指出其在总结大型 PDF 时速度较慢。一位成员[分享了一个代码片段](https://github.com/langchain-ai/langchain/issues/12336)，演示了如何使用 `map_reduce` 链来提高速度。

- **Instructor 与 LangChain 的集成与使用**：讨论包括了将 [**Instructor**](https://python.useinstructor.com/)（它可以促进 LLM 输出 JSON 等结构化数据）与 **LangChain** 结合使用的可能性。一位成员表示，他们想要一个能生成有效的 pydantic 对象并通过 LLM 处理验证错误的工具。

- **在 LangChain Tool Calling Agent 中确保有效的工具参数**：一位成员就如何[自修复无效的工具参数](https://langchain-documentation.langchain.com/docs/tooling/toolchain)寻求建议，这些参数是由 **LangChain 的 Tool Calling Agent** 中的 LLM 生成的，并提到了 **Groq Mixtral** 的问题。

- **使用 LangChain 高效读取 CSV 文件**：一位成员[寻求最有效的方法](https://langchain-documentation.langchain.com/docs/tooling/agents)来使用 Agent 读取 .csv 文件，另一位成员建议使用 **ChatOpenAI** 配合 `openai-tools` Agent 类型。随后还讨论了模型在处理不同数量的 .csv 文件时的性能。

- **在 LangChain 中处理 FAISS-GPU 的内存释放**：一位用户询问在 LangChain 中使用 **FAISS-GPU** 和 **Hugging Face embeddings** 遇到 `[torch.cuda.OutOfMemoryError](https://pytorch.org/docs/stable/cuda.html#memory-management)` 时如何释放内存，因为无法通过提供的封装器手动释放 GPU 内存。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://discord.gg/pornox">加入 Teen Content 18+ 🍑🔞 Discord 服务器！</a>：查看 Discord 上的 Teen Content 18+ 🍑🔞 社区 - 与其他 441 名成员一起交流，享受免费的语音和文字聊天。</li><li><a href="https://www.bloomberg.com/news/newsletters/2024-04-11/this-startup-is-trying-to-test-how-well-ai-models-actually-work">Bloomberg - 你是机器人吗？</a>：未找到描述</li><li><a href="https://python.useinstructor.com/">欢迎来到 Instructor - Instructor</a>：未找到描述</li><li><a href="https://js.langchain.com/docs/use_cases/summarization#example>).">文本摘要 | 🦜️🔗 Langchain</a>：一个常见的用例是想要总结长文档。</li><li><a href="https://python.langchain.com/docs/modules/model_io/chat/structured_output/">[beta] 结构化输出 | 🦜️🔗 LangChain</a>：让 LLM 返回结构化输出通常至关重要。</li><li><a href="https://python.langchain.com/docs/use_cases/summarization#option-2.-map-reduce>).">文本摘要 | 🦜️🔗 LangChain</a>：在 Colab 中打开</li><li><a href="https://github.com/langchain-ai/langchain/issues/8399>),">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号来为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/12336>),">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号来为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/12336>).">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号来为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/17352>):">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号来为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/5481>).">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号来为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://python.langchain.com/docs/expression_language/why#invoke>)">LCEL 的优势 | 🦜️🔗 LangChain</a>：我们建议阅读 LCEL [入门]
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1228274170442289215)** (4 条消息): 

- **不当内容警报**：频道中的一条消息通过链接推广**成人内容**，将其伪装成加入 Discord 服务器的邀请。
- **寻求关于 LangFuse Callbacks 的信息**：一位成员请求协助利用 **langfuse callback handler** 通过 langserve 进行追踪，并正在寻找关于如何记录输入（如问题、会话 ID 和用户 ID）的来源或示例。

**提到的链接**：<a href="https://discord.gg/pornox">加入 Teen Content 18+ 🍑🔞 Discord 服务器！</a>：查看 Discord 上的 Teen Content 18+ 🍑🔞 社区 - 与其他 441 名成员一起交流，享受免费的语音和文字聊天。

  

---


**LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1228274187991253002)** (3 条消息): 

- **不当内容警报**：一条消息包含推广成人内容的链接；已被标记为垃圾信息。此类内容通常违反 Discord 的社区准则。

**提到的链接**：<a href="https://discord.gg/pornox">加入 Teen Content 18+ 🍑🔞 Discord 服务器！</a>：查看 Discord 上的 Teen Content 18+ 🍑🔞 社区 - 与其他 441 名成员一起交流，享受免费的语音和文字聊天。

  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1227911324470607882)** (8 条消息🔥): 

- **Galaxy AI 发布**：GalaxyAI 推出了一项**免费 API 服务**，包含 **GPT-4**、**GPT-4-1106-PREVIEW**、**GPT-3.5-turbo-1106** 和 **Claude-3-haiku** 等 AI 模型，并支持 Langchain 集成。可在 [Galaxy AI](https://galaxyapi.onrender.com) 获取，这些 API 采用 OpenAI 格式，便于集成到项目中。

- **Appstorm v1.6.0 提升应用构建体验**：Appstorm 的新版本 1.6.0 已在 [Appstorm beta](https://beta.appstorm.ai/) 发布，包含移动端注册、音乐和地图 GPT、数据探索与可视化、更便捷的应用分享、改进的并发应用管理以及增强应用构建体验的错误修复。

- **寻求 AI 助手开发建议**：一位成员正在开发一个虚拟 AI 助手，需要解析数千份 PDF 以生成基于 **RAG** (Retriever-Answer Generator) 的功能，并通过阅读数据手册为 IoT 边缘平台设置配置参数，正在寻求处理该项目的建议。

- **不当内容警示**：**警告**：识别到一名成员发布了包含露骨内容和色情材料链接的帖子；这些内容对 AI 讨论没有任何建设性贡献。

- **通过 AI 增强 Meeting Reporter**：一款名为 Meeting Reporter 的新应用将 Streamlit 与 Langgraph 结合，通过人机协作（human-AI collaboration）创作新闻故事，该应用需要付费的 OpenAI API key。它已在 [Streamlit App](https://meeting-reporter.streamlit.app/) 上展示，开源代码可在 [GitHub](https://github.com/tevslin/meeting-reporter) 获取，更多详情和会议转录文本见相关的 [博客文章](https://blog.tomevslin.com/2024/04/human-in-the-loop-artificial-intelligence.html)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://discord.gg/pornox">加入 Teen Content 18+ 🍑🔞 Discord 服务器！</a>：查看 Discord 上的 Teen Content 18+ 🍑🔞 社区 - 与其他 441 名成员一起交流，享受免费的语音和文字聊天。</li><li><a href="https://beta.appstorm.ai/">Appstorm</a>：在几秒钟内构建 AI 应用</li><li><a href="https://galaxyapi.onrender.com">Galaxy AI - Swagger UI</a>：未找到描述</li><li><a href="https://meeting-reporter.streamlit.app/">无标题</a>：未找到描述</li><li><a href="https://github.com/tevslin/meeting-reporter">GitHub - tevslin/meeting-reporter: Human-AI collaboration to produce a newstory about a meeting from minutes or transcript</a>：人机协作，根据会议记录或转录生成会议新闻报道 - tevslin/meeting-reporter
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1227879417418878996)** (4 messages): 

- **LangChain 教程警示**：发布了一篇关于 **LCEL (LangChain Execution Language)** 以及使用 runnables 创建链的新教程。感兴趣的人员可以阅读并提供反馈：[LangChain Tutorial: LCEL and Composing Chains from Runnables](https://medium.com/@klcoder/langchain-tutorial-lcel-and-composing-chains-from-runnables-751090a0720c?sk=55c60f03fb95bdcc10eb24ce0f9a6ea7)。

- **垃圾信息警示**：tutorials 频道收到了多条推广成人内容的垃圾信息。这些消息包含露骨内容，与本频道的目的无关。

**提到的链接**：<a href="https://discord.gg/pornox">加入 Teen Content 18+ 🍑🔞 Discord 服务器！</a>：查看 Discord 上的 Teen Content 18+ 🍑🔞 社区 - 与其他 441 名成员一起交流，享受免费的语音和文字聊天。

  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1227995840409239683)** (9 messages🔥): 

```html
<ul>
  <li><strong>Osanseviero 的推文推送</strong>：osanseviero 分享了一条新推文，预计会有令人兴奋的消息或见解。点击<a href="https://twitter.com/osanseviero/status/1778430866718421198">此处</a>查看推文。</li>
  <li><strong>亮点回顾</strong>：Community Highlights #53 带来了多样化的认证用户内容，包括 Hugging Face 的葡萄牙语介绍、时尚试穿空间以及各种有趣的 GitHub 仓库。</li>
  <li><strong>嵌入助力成功</strong>：该 RAG 聊天机器人由通过 <a href="https://huggingface.co/datasets/not-lain/wikipedia-small-3000-embedded">not-lain/wikipedia-small-3000-embedded</a> 嵌入的数据集驱动，作为生成用户知情响应的检索源。</li>
  <li><strong>检索与生成双重奏</strong>：通过将嵌入数据集的检索与生成式 AI 相结合，该 RAG 聊天机器人创新地寻求提供准确的信息推断。</li>
  <li><strong>RMBG1.4 下载量飙升</strong>：与 transformers 库集成的 RMBG1.4 本月下载量达到了 23 万次的新里程碑，显示出社区浓厚的兴趣和使用量。</li>
</ul>
```

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/datasets/not-lain/wikipedia-small-3000-embedded">not-lain/wikipedia-small-3000-embedded · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=nK1hijr8Qng&t=74s">[IA a Z - 06] 介绍 🤗 Hugging Face</a>: 🤗🤗🤗🤗🤗🤗 如果说有一件事是我喜欢的，那就是有大量的工具选项可以学习！这极大地简化了学习新事物的过程，起初...</li><li><a href="https://huggingface.co/spaces/tonyassi/fashion-try-on">Fashion Try On - tonyassi 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://github.com/SuleymanEmreErdem/deep-q-learning-applications">GitHub - SuleymanEmreErdem/deep-q-learning-applications: 我的 Deep Q-Learning 项目</a>: 我的 Deep Q-Learning 项目。通过在 GitHub 上创建账号来为 SuleymanEmreErdem/deep-q-learning-applications 的开发做出贡献。</li><li><a href="https://huggingface.co/spaces/not-lain/RMBG1.4-with-imageslider">带有图像滑块的 RMBG1.4 - not-lain 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://github.com/RooTender/augmentator">GitHub - RooTender/augmentator: 开箱即用的图像增强工具</a>: 开箱即用的图像增强工具。通过在 GitHub 上创建账号来为 RooTender/augmentator 的开发做出贡献。</li><li><a href="https://not-lain-rag-chatbot.hf.space/"># RAG</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=76BkMULO7uw">使用 Ollama, Mistral 和 LLava 判断是否为热狗</a>: 在本教程中，我们使用 Ollama, Mistral 和 LLava 来查看一张图片是否为热狗 #python #pythonprogramming #llm #ml #ai #aritificialintel...</li><li><a href="https://github.com/EdoPedrocchi/RicercaMente">GitHub - EdoPedrocchi/RicercaMente: 旨在通过多年来发表的科学研究追溯数据科学历史的开源项目</a>: 旨在通过多年来发表的科学研究追溯数据科学历史的开源项目 - EdoPedrocchi/RicercaMente</li><li><a href="https://ragdoll-studio.vercel.app/">未找到标题</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=oVJsJ0e6jWk">Where's My Pic 演示</a>: 大家好，我是 Om Alve，在这个视频中，我将演示我的名为 'Where's my pic?' 的项目。该项目解决了搜索...的问题。</li><li><a href="https://huggingface.co/blog/joey00072/mixture-of-depth-is-vibe">Mixture of Depth is Vibe</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/dcarpintero/building-a-neural-network-for-image-classification">从零开始构建神经网络分类器：分步指南</a>: 未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1227886644523438160)** (64 条消息🔥🔥): 

- **数据集基础 (Basics of Datasets)**: 一位用户询问了学习数据集的起点。他们被引导至 HuggingFace 文档，其中包含解释器、模板、创建数据集的指南等，可以在 [HuggingFace's Datasets Library](https://huggingface.co/docs/datasets/index) 找到。

- **辅助人工帮助的问答机器人**: 有人建议在 #help 频道中使用问答机器人，通过建议相关信息或指向类似的已解决问题来协助用户。启用机器人建议的按钮可能会增加其可见性和使用率。

- **训练用于 GUI 导航的模型**: 有一段关于训练用于操作系统 GUI 导航的模型可行性的详细对话。讨论了使用辅助功能模式和应用接口，而不是基于像素完美的视觉控制的替代方案。

- **在单个 GPU 上运行多个模型**: 出现了一场关于在单个 GPU 上同时运行多个模型的讨论。用户分享了经验和技术，例如使用信号量（semaphores）创建 Web 服务器以优化 GPU 吞吐量。

- **使用 Datasets 处理大型数据集和进度跟踪**: 用户讨论了处理和上传超大型数据集（特别是音频和图像）的最佳方法，重点是实现流式传输（streaming）和高效的元数据更新。还有关于在对数据集执行映射函数（mapping functions）时如何提取进度信息以进行 UI 集成的查询。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.bloomberg.com/news/newsletters/2024-04-11/this-startup-is-trying-to-test-how-well-ai-models-actually-work">Bloomberg - Are you a robot?</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/datasets/index">Datasets</a>: 未找到描述</li><li><a href="https://github.com/karpathy/llm.c">GitHub - karpathy/llm.c: LLM training in simple, raw C/CUDA</a>: 使用简单的原生 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/leejet/stable-diffusion.cpp">GitHub - leejet/stable-diffusion.cpp: Stable Diffusion in pure C/C++</a>: 纯 C/C++ 实现的 Stable Diffusion。通过在 GitHub 上创建账号来为 leejet/stable-diffusion.cpp 的开发做出贡献。</li><li><a href="https://github.com/huggingface/quanto">GitHub - huggingface/quanto: A pytorch Quantization Toolkit</a>: 一个 PyTorch 量化工具包。通过在 GitHub 上创建账号来为 huggingface/quanto 的开发做出贡献。
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1228161578382266368)** (1 条消息): 

- **Docker 替代方案深度解析**: 一段名为 ["Contain Your Composure: On Podman-Compose, Code Cleanup, and Tiny Llamas"](https://youtu.be/wAptz3f88H0) 的视频教程提供了使用 **Podman-Compose** 构建微服务的演练，重点介绍了 **Yet Another Markup Language (YAML)** 文件并介绍了 **Small Langu**。描述暗示该视频关注整洁代码实践，并可能以有趣的方式提炼复杂主题。

**提到的链接**: <a href="https://youtu.be/wAptz3f88H0">Contain Your Composure: On Podman-Compose, Code Cleanup, and Tiny Llamas</a>: 本视频教程将带你了解使用 Podman-Compose、YAML 文件、Small Langu 构建微服务的全过程...

  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1228016591187480717)** (6 条消息): 

- **Karpathy 的极简 CUDA 实现**: Andrej Karpathy 实现了一种使用 **原生 C/CUDA 进行 LLM 训练** 的直观方法。代码可以在 GitHub 上的 [llm.c 仓库](https://github.com/karpathy/llm.c) 中获取。
  
- **Mistral 7B vs. Llama 模型**: 一个基准测试网站对比了 **Mistral 7B** 与 **Llama 2 系列**，指出 Mistral 7B 在所有指标上都优于 Llama 2 13B，并能与 Llama 34B 媲美。他们的发现称赞了 Mistral 7B 在代码和推理方面的卓越表现，[Mistral 7B 详情见此](https://mistral-7b.com)。
  
- **等待进一步信息**: 一位成员提到收到了来自 **Google Cloud Next ’24** 的文档，但未提供更多细节或链接。

- **Parler TTS 介绍**: **HuggingFace** 推出了 **parler-tts**，这是一个用于高质量 **TTS 模型** 推理和训练的库，可在其 GitHub 仓库中获取。感兴趣的人可以通过 [parler-tts GitHub 页面](https://github.com/huggingface/parler-tts) 进行探索和贡献。

- **AI 书籍推荐**: 一位成员发现《The age of AI》是一本非常有趣的读物，但未提供额外信息或链接。

- **增强记忆的文档检索指南**: 发布了一篇关于通过记忆增强文档检索的教程，详细介绍了如何使用带有 **基于 Colbert 的 Agent 的 LlamaIndex**。该教程可在 Medium 上找到，提供了对文档检索进展的见解：[增强文档检索教程](https://medium.com/ai-advances/enhancing-document-retrieval-with-memory-a-tutorial-for-llamaindex-with-colbert-based-agent-1c3c47461122)。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/karpathy/llm.c">GitHub - karpathy/llm.c: LLM training in simple, raw C/CUDA</a>: 使用简单的原生 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/huggingface/parler-tts">GitHub - huggingface/parler-tts: Inference and training library for high-quality TTS models.</a>: 用于高质量 TTS 模型的推理和训练库。 - huggingface/parler-tts</li><li><a href="https://mistral-7b.com">Mistral 7B-The Full Guides of Mistral AI &amp; Open Source LLM</a>: Mistral 7B 及其 15 个微调模型以及关于开源 LLM 的指南。包含 Mistral 7B 模型及其来自 Mistral AI 的微调模型的 Chatbot 演示。
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1227958383324168253)** (8 条消息🔥):

- **Marimo 提供的 Hugging Face Model Playground**：一位成员介绍了 [marimo-labs](https://x.com/marimo_io/status/1777765064386474004)，这是一个与 **Hugging Face** 集成的新 Python 包，使用户能够为文本、图像和音频模型创建交互式 Playground。这由 marimo 的响应式执行结合 Hugging Face 的免费推理 API 提供支持。
- **Marimo Playground 交互式链接**：分享了一个 [交互式 marimo 应用程序](https://marimo.app/l/tmk0k2)，用户可以使用自己的 token 交互式地查询 **Hugging Face** 上的模型；该应用通过 WASM 在本地运行。
- **葡萄牙语的 AI 概念**：一位成员发布了一篇葡萄牙语的文章和视频，介绍了 Hugging Face 的基础知识，为讲葡萄牙语的 AI 初学者提供了宝贵的资源。该内容是名为 ["de IA a Z"](https://iatalk.ing/series/ia-z/) 系列的一部分，还有涵盖各种 AI 主题的其他文章。
- **Mergekit 的即将推出的功能**：宣布了 Mergekit 即将推出的新方法，包括已经添加的 *rescaled TIES*。
- **分享了 Vimeo 视频**：分享了一个来自 Vimeo 的视频链接 [https://vimeo.com/933289700](https://vimeo.com/933289700)，但未提供背景或描述。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/marimo_io/status/1777765064386474004">来自 marimo (@marimo_io) 的推文</a>: 宣布 marimo-labs：一个包含尖端 marimo 功能的新 Python 包。我们的第一个实验室是 @huggingface 集成 🤗！为 HuggingF 上超过 35 万个模型中的任何一个创建交互式 Playground...</li><li><a href="https://marimo.app/l/tmk0k2">marimo | 下一代 Python notebook</a>: 使用 marimo 无缝探索数据并构建应用，这是一款下一代 Python notebook。</li><li><a href="https://vimeo.com/933289700">test</a>: 这是 Test Account 在 Vimeo 上的 "test"，Vimeo 是高质量视频及其爱好者的家园。</li><li><a href="https://iatalk.ing/hugging-face/">Apresentando o 🤗 Hugging Face</a>: 你好！今天我想向你介绍一个对于正在进入或已经属于人工智能世界的人来说必不可少的工具：Hugging Face Hub，亲切地称为 hf，或者直接叫 🤗...</li><li><a href="https://www.youtube.com/watch?v=nK1hijr8Qng&t=74s">[IA a Z - 06] Apresentando o 🤗 Hugging Face</a>: 🤗🤗🤗🤗🤗🤗如果说有什么是我喜欢的，那就是有大量的工具选项可以学习！这极大地简化了学习新事物的过程，尤其...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1228243322490064907)** (1 条消息): 

- **Blenderbot 微调建议**：一位成员建议微调 **FAIR 的 Blenderbot**，该模型可在 [HuggingFace](https://huggingface.co/models) 上获得，并指出需要为该任务寻找合适的数据集。
  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1227986394937299064)** (14 条消息🔥): 

- **推荐 GPU 进程管理工具**：推荐了一个名为 **nvitop** 的 **GPU 进程查看器**，作为管理 GPU 进程的实用工具，更多详情见 [GitHub - XuehaiPan/nvitop](https://github.com/XuehaiPan/nvitop)。

- **视频修复技术的入门步骤**：一位寻求视频修复技术（如去噪和去除伪影）建议的用户被引导至一篇图像修复论文作为起点，并建议通过添加时间维度将其视为视频的扩展，详见 [arXiv 上的 NAFNet 论文](https://arxiv.org/abs/2204.04676)。

- **数据增强在图像修复中的重要性**：针对在没有真值 (ground truth) 的情况下训练视频修复数据集的担忧，强调了数据增强是关键，并提供了两篇论文的链接：[BSRGAN](https://arxiv.org/abs/2103.14006) 和 [Real-ESRGAN](https://arxiv.org/abs/2107.10833)，它们详细介绍了对训练修复模型非常有用的增强流水线。

- **了解视频去闪烁**：针对视频噪声和伪影问题，用户被推荐至 GitHub 上的一个特定项目 [All-In-One-Deflicker](https://github.com/ChenyangLEI/All-In-One-Deflicker)，该项目处理盲视频去闪烁 (blind video deflickering)。

- **探索多模态和向量数据库的集成**：讨论了将 Google 的 Vertex 多模态嵌入 (embeddings) 与 Pinecone 向量数据库集成，包括它们如何通过嵌入处理拼写错误和品牌识别，并附带了 Google 的演示链接 [AI Demos Dev](https://ai-demos.dev/)。
<div class="linksMentioned">

<strong>提及的链接</strong>:

</div>

<ul>
<li>
<a href="https://ai-demos.dev/">AI Demos</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2103.14006">Designing a Practical Degradation Model for Deep Blind Image Super-Resolution</a>: 众所周知，如果假设的退化模型与真实图像中的模型不符，单图像超分辨率 (SISR) 方法的表现将不尽如人意。尽管已经有几种退化模型...</li><li><a href="https://arxiv.org/abs/2107.10833">Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data</a>: 尽管在盲超分辨率领域已经进行了许多尝试，以恢复具有未知且复杂退化的低分辨率图像，但它们距离解决通用的真实世界退化图像仍有很大差距...</li><li><a href="https://github.com/ChenyangLEI/All-In-One-Deflicker?tab=readme-ov-file">GitHub - ChenyangLEI/All-In-One-Deflicker: [CVPR2023] Blind Video Deflickering by Neural Filtering with a Flawed Atlas</a>: [CVPR2023] 通过带有缺陷图谱的神经滤波进行盲视频去闪烁 - ChenyangLEI/All-In-One-Deflicker</li><li><a href="https://huggingface.co/docs/transformers/main/en/model_doc/grounding-dino">Grounding DINO</a>: 未找到描述</li><li><a href="https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Grounding%20DINO/Inference_with_Grounding_DINO_for_zero_shot_object_detection.ipynb">Transformers-Tutorials/Grounding DINO/Inference_with_Grounding_DINO_for_zero_shot_object_detection.ipynb at master · NielsRogge/Transformers-Tutorials</a>: 此仓库包含我使用 HuggingFace 的 Transformers 库制作的演示。 - NielsRogge/Transformers-Tutorials</li><li><a href="https://github.com/XuehaiPan/nvitop">GitHub - XuehaiPan/nvitop: An interactive NVIDIA-GPU process viewer and beyond, the one-stop solution for GPU process management.</a>: 一个交互式的 NVIDIA-GPU 进程查看器及更多功能，GPU 进程管理的一站式解决方案。 - XuehaiPan/nvitop</li><li><a href="https://arxiv.org/abs/2204.04676">Simple Baselines for Image Restoration</a>: 尽管最近在图像恢复领域取得了显著进展，但最先进 (SOTA) 方法的系统复杂性也在增加，这可能会阻碍其便利性...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1227902712293888081)** (12 messages🔥): 

- **寻求更长上下文模型**: 一位成员询问了能够处理 10-15k token 左右更长上下文的 **encoder-decoder models**。建议包括研究 **BigBird** 和 **Longformer** 等模型。

- **使用 Checkpoints 进行训练**: 有人询问关于使用 HuggingFace 的 `trainer` 来暂停和恢复训练的问题。确认 `trainer.train()` 中的 `resume_from_checkpoint` 选项可以实现此目的。

- **脚本协助请求**: 一位成员分享了一个详细的脚本 `train_ddp.py`，该脚本利用 **transformers**、**TRL**、**PeFT** 和 **Accelerate** 来训练模型，并请求帮助以确保其正确性以及训练模型的正确保存。

- **平衡评分与指导**: 参与者讨论了评估自动化辅导响应的方法，考虑使用**加权平均**来优先处理评分方案而非指导原则，并建议使用适用于语义含义的 embedding 模型，例如 `sentence-transformers/all-MiniLM-L6-v2`。

- **下载大型分块模型**: 有一个关于下载和组装像 **Mixtral-8x22B** 这样被分割成多个 GGUF 文件的大型模型的问题。该成员询问文件是否需要手动合并，或者在加载时是否会自动组装。
  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1227891325551120424)** (5 messages): 

- **Fastai 和 Diffusers 深度探索**: 一位成员建议学习 [fastai 的第二部分课程](https://course.fast.ai/Lessons/part2.html) 以进行深入理解，然后探索 [HuggingFace diffusers 的 GitHub issues](https://github.com/huggingface/diffusers/issues?q=sort%3Acreated-asc+label%3A%22good+first+issue%22%2C%22Good+second+issue%22%2Chacktoberfest) 和 [相关的 HuggingFace 博客文章](https://huggingface.co/blog?tag=diffusion&p=1)。他们还建议关注 [GitHub 上关于 diffusers 的热门讨论](https://github.com/huggingface/diffusers/discussions?discussions_q=sort%3Atop) 以获取最新见解。
  
- **PixArt-Alpha Pipeline 使用**: 在一条简短的笔记中，一位成员建议查看利用了上述技术的 **PixArt-Alpha pipeline**。

- **消费级 GPU 的限制**：一位成员讨论了在使用 SDPA 和 `torch.compile()` 等现代技术时消费级 GPU 的局限性，认为这些技术在最新的 GPU 上更有益。对于那些使用性能较低 GPU 的用户，他们分享了来自 [GitHub 讨论](https://github.com/huggingface/diffusers/discussions/6609) 的建议，关于如何加速 diffusion。

- **理解多模态搜索能力**：一位成员询问 Google 的多模态 embeddings 如何不仅能匹配图像，还能识别带有拼写错误的品牌名称，这是基于 [AI-demos 的演示](https://ai-demos.dev/)。他们表示打算构建一个功能类似的 Web 应用程序，并寻求对其底层机制的深入了解。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://ai-demos.dev/">AI Demos</a>：未找到描述</li><li><a href="https://github.com/huggingface/diffusers/discussions/6609),">在性能较低的 GPU 上实现更快的扩散 ⚡️ · huggingface/diffusers · Discussion #6609</a>：我们最近发布了《加速生成式 AI 第三部分：扩散，快速》，展示了如何：我们在 80GB A100 上展示了这一点。文中介绍的技术在很大程度上适用于相对...</li><li><a href="https://github.com/huggingface/diffusers/issues?q=sort%3Acreated-asc+label%3A%22good+first+issue%22%2C%22Good+second+issue%22%2Chacktoberfest).">Issues · huggingface/diffusers</a>：🤗 Diffusers：用于 PyTorch 和 FLAX 中图像和音频生成的先进扩散模型。- Issues · huggingface/diffusers</li><li><a href="https://huggingface.co/blog?tag=diffusion&p=1).">Hugging Face – Blog</a>：未找到描述</li><li><a href="https://github.com/huggingface/diffusers/discussions?discussions_q=sort%3Atop).">huggingface/diffusers · Discussions</a>：探索 huggingface diffusers 的 GitHub Discussions 论坛。讨论代码、提问并与开发者社区协作。
</li>
</ul>

</div>
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1227925030986321962)** (86 条消息🔥🔥): 

- **混合搜索方法论辩论**：一位用户就使用 Cohere 的 rerank 策略寻求建议，并询问在 reranking 之前合并词法（lexic）和语义（semantic）搜索结果，是否比将它们全部放在一个列表中进行一次性 reranking 更有效。其他成员建议第二种方法可能更高效，因为它只涉及单个 reranking 步骤，并且可以节省延迟。

- **模型崛起**：Sandra Kublik 的一条推文链接宣布发布了 **Rerank 3**，这是来自 Cohere 的新模型，它增强了搜索和 RAG 系统，包括 **4k 上下文长度**、最先进的 (SOTA) 搜索准确度、代码检索以及跨 **100 多种语言**的多语言能力。包含更多细节的原始推文可以在[这里](https://x.com/itssandrakublik/status/1778422401648455694?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)找到。

- **AI 初创公司与创新**：关于 Alberto Rizzoli 介绍的名为 V7 Go 的**多模态 AI 时代工作自动化工具**的讨论引起了人们的兴趣，该工具旨在利用 GenAI 处理单调任务。另一个竞争产品 **Scarlet AI** 由其创建者提出，宣传其通过 AI 与人类协作进行任务规划和执行的能力。

- **Perplexity 的 "Online" 模型**：用户讨论了 Perplexity 的 "online" 模型从 LMSYS Arena 中消失的情况，推测其含义及背后的技术。Perplexity 博客文章的链接显示它指的是可以访问互联网的模型，链接在[这里](https://www.perplexity.ai/hub/blog/introducing-pplx-online-llms)。

- **AI 聊天机器人竞技场领导者**：分享了关于 **GPT-4-Turbo** 重新夺回 Lmsys 盲测聊天机器人排行榜榜首的更新，强调了其强大的代码编写和推理能力，这已通过各个领域的 8000 多张用户投票得到证实。公告推文可以在[这里](https://x.com/lmsysorg/status/1778555678174663100?s=46&t=90xQ8sGy63D2OtiaoGJuww)访问。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/albertorizzoli/status/1778047883012661413?s=46&t=9">来自 Alberto Rizzoli (@Albertorizzoli) 的推文</a>：它来了.. 多模态 AI 时代的自动化办公。V7 Go 利用 GenAI 可靠地大规模完成工作中最为枯燥的重复性任务。</li><li><a href="https://scarletai.co">Scarlet</a>：未找到描述</li><li><a href="https://stanford.zoom.us/j/99922151759?pwd=dW5CcUtVYkNybGZGY0hMWUZtVkZBZz09">加入我们的 Cloud HD 视频会议</a>：Zoom 是现代企业视频通信领域的领导者，拥有简单、可靠的云平台，可跨移动端、桌面端和会议室系统提供视频和音频会议、聊天及网络研讨会。Zoom ...</li><li><a href="https://x.com/0xmmo/status/1778589664678760748?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 mo² (@0xmmo) 的推文</a>：Lmsys 盲测聊天机器人排行榜刚刚更新。GPT-4 Turbo 以显著优势重夺榜首。https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard</li><li><a href="https://www.bloomberg.com/news/newsletters/2024-04-11/this-startup-is-trying-to-test-how-well-ai-models-actually-work">Bloomberg - 你是机器人吗？</a>：未找到描述</li><li><a href="https://course.fast.ai/">程序员实用深度学习 - Practical Deep Learning</a>：一门专为具有一定编程经验、想要学习如何将深度学习和机器学习应用于实际问题的人员设计的免费课程。</li><li><a href="https://x.com/itssandrakublik/status/1778422401648455694?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Sandra Kublik (@itsSandraKublik) 的推文</a>：推出我们的最新模型 Rerank 3！🚨 进一步增强搜索和 RAG 系统。包含哪些特性？🧑‍🍳 - 4k 上下文长度，- 在复杂数据（如电子邮件、JSON 文档...）上具有 SOTA 搜索准确率</li><li><a href="https://x.com/albertorizzoli/status/1778047883012661413?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Alberto Rizzoli (@Albertorizzoli) 的推文</a>：它来了.. 多模态 AI 时代的自动化办公。V7 Go 利用 GenAI 可靠地大规模完成工作中最为枯燥的重复性任务。</li><li><a href="https://x.com/lmsysorg/status/1778555678174663100?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 lmsys.org (@lmsysorg) 的推文</a>：🔥激动人心的消息 —— GPT-4-Turbo 刚刚再次夺得 Arena 排行榜第一名！哇！我们收集了来自不同领域的 8000 多张用户投票，并观察到其强大的编码和推理能力...</li><li><a href="https://x.com/daniel_eckler/status/1778421669201093057?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Eckler by Design ✦ (@daniel_eckler) 的推文</a>：C3PO x Childish Gambino 🤖 👑 100% AI（官方音乐视频）@openAI + @runwayml + @suno_ai_ + @resembleai + @fable_motion + @midjourney + @topazlabs</li><li><a href="https://www.youtube.com/watch?v=tNmgmwEtoWE">揭秘 Devin：“首位 AI 软件工程师”在 Upwork 上的谎言被揭穿！</a>：最近，所谓的“首位 AI 软件工程师” Devin 发布。该公司撒谎称其视频展示了 Devin 完成任务并获得报酬...
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1228079878730158080)** (3 条消息): 

- **新播客提醒**：查看最新一期播客，内容包括与 **Elicit** 的 Jungwon Byun 和 Andreas Stuhlmüller 的讨论。本期节目深入探讨了 AI 研究的监督，可在 [Twitter](https://twitter.com/swyx/status/1778520821386121582) 上收听。
- **YouTube 上的 Elicit**：您也可以在 [YouTube](https://www.youtube.com/watch?v=Dl66YqSIu5c&embeds_referring_euri=https%3A%2F%2Fwww.latent.space%2F&feature=emb_title) 上观看 Elicit 播客集，包括深入探讨为什么产品可能优于研究以及 Elicit 的演变。别忘了点赞并订阅以获取更多内容！

**提到的链接**：<a href="https://www.youtube.com/watch?v=Dl66YqSIu5c&embeds_referring_euri=https%3A%2F%2Fwww.latent.space%2F&feature=emb_title">监督 AI 研究的过程 —— 与 Elicit 的 Jungwon Byun 和 Andreas Stuhlmüller</a>：时间戳：00:00:00 介绍 00:07:45 Johan 和 Andreas 如何联手创建 Elicit 00:10:26 为什么产品优于研究 00:15:49 演变...

  

---


**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1227965397932707854)** (26 条消息🔥):

- **Mixtral-8x22B 首次亮相**：**Mixtral-8x22B** 模型已转换为 HuggingFace Transformers 格式并可供使用，对负责转换的用户表示感谢。提供了使用 `transformers` 运行该模型的说明，以及[模型链接](https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1)和 [Twitter 公告](https://twitter.com/MistralAI/status/1777869263778291896)。
- **DeepSeekMoE 挑战 Google 的 GShard**：据报道，具有共享专家和细粒度专家分割特征的 *DeepSeekMoE* 性能足以媲美或超越 Google 的 GShard 模型，并提供了[论文链接](https://llm-paper-club-asia-notes.vercel.app/papers/deekseek-moe)以了解架构详情。
- **Mixture of Experts 教育资源**：HuggingFace 的一篇博客文章讨论了 **Mixture of Experts (MoEs)** 以及最近发布的 Mixtral 8x7B，被推荐为 MoE 概念初学者的入门资源。
- **MoE 中专家专业化的探索**：社区讨论了 Mixtral 的 MoE 性能和专家专业化（expert specialization）的概念，将 MoE 与在推理时进行专业化的 semantic router 进行对比，并思考这些模型是如何实现专家专业化的。
- **关于 MoE 模型冗余和专业性的疑问**：围绕 MoE 模型专家内部的实际学习和专业化过程展开了对话，并提供了 [semantic router 的 GitHub 仓库](https://github.com/aurelio-labs/semantic-router)作为参考，同时对 device loss 的实现及其在报告的源代码中明显缺失感到好奇。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://llm-paper-club-asia-notes.vercel.app/papers/deekseek-moe">Nextra: 下一代文档生成器</a>：Nextra: 下一代文档生成器</li><li><a href="https://huggingface.co/blog/moe">Mixture of Experts 详解</a>：未找到描述</li><li><a href="https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1">mistral-community/Mixtral-8x22B-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/aurelio-labs/semantic-router">GitHub - aurelio-labs/semantic-router: 超快速的 AI 决策和多模态数据智能处理。</a>：超快速的 AI 决策和多模态数据智能处理。 - aurelio-labs/semantic-router
</li>
</ul>

</div>
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1227931220042715148)** (93 条消息🔥🔥): 

- **Draw Things 因闭源受到批评**：成员们对 *Draw Things* 表示不满，提到它不是开源的，也没有对社区做出实质性回馈。还指出所谓的开源版本缺少 *metal-flash-attention support* 等核心功能。
- **对 TempestV0.1 声明的怀疑**：讨论了 *TempestV0.1 Initiative* 并持怀疑态度，特别是关于其 300 万步训练的声明以及其数据集大小的可信度——据称 600 万张图像仅占用 200GB。
- **对 Laion 5B 演示的关注**：用户询问了 *Laion 5B* Web 演示的状态，一些人预计它不会恢复。然而，提到 **Christoph** 曾表示它会回归，但未提供具体细节或时间表。
- **防范潜在诈骗的警告**：对与加密货币和虚假关联 **LAION** 的代币相关的诈骗表示显著担忧。警告用户保持警惕，讨论表明此类活动正在利用 LAION 的名义。
- **对误导性信息和解决方案的不满**：指出了误导性信息的持续问题，特别是 Twitter 等平台上虚假声明的流传。建议置顶公告或添加到自动审核系统中以帮助防止这些诈骗。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.ora.io/app/imo/olm">ORA</a>：未找到描述</li><li><a href="https://mirror.xyz/orablog.eth/X3DYXDHnjkpB-DOz88DZO5RdfZPxxRi5j53bxttNgsk>">全球首个 OpenLM 初始模型发行 (IMO)</a>：OpenLM 是一个高性能语言建模 (LM) 仓库，旨在促进中型 LM 的研究。 </li><li><a href="https://tenor.com/mDErrG5aLdg.gif">Pinoquio GIF - Pinoquio - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


**LAION ▷ #[announcements](https://discord.com/channels/823813159592001537/826154622644649985/1228398882598162553)** (1 条消息):

- **警惕虚假 LAION NFT 声明**：针对一个发布 LAION 将发行 NFT 虚假广告的**虚假 Twitter 账号**，官方已发布警告。LAION 郑重澄清，其不销售任何产品，没有雇员，是开源社区的一部分，提供开放、透明且免费的 AI 资源。
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1227962018623914106)** (19 messages🔥): 

- **Intel 与 AMD 及 Nvidia 的制造对比**：会议指出，Intel 自行制造芯片，而 AMD 和 Nvidia 则使用 TSMC 进行半导体代工。
- **LRUs 在 LRA 上展现潜力**：改进的最近最少使用（LRUs）算法被认为在 Long Range Arena (LRA) 长上下文性能基准测试中表现良好。
- **Guidance 权重策略提升 Diffusion Models**：一项研究强调了将 guidance 限制在特定噪声水平对 Diffusion Models 的益处；通过这种方式，可以提高推理速度并提升图像质量（[研究论文](https://arxiv.org/abs/2404.07724)）。
- **将研究应用于实际工具**：关于动态管理 classifier-free guidance (CFG) 的信息已关联到现有的 GitHub issue 以供参考，表明此类研究成果正被积极整合到工具实现中，例如 huggingface 的 diffusers（[GitHub issue](https://github.com/huggingface/diffusers/issues/7657)）。
- **动态 Guidance 调度作为一个学习过程**：一位成员建议将 CFG 的动态调度视为一个更细粒度且可能通过学习获得的过程，并参考了为每个时间步设置独立缩放值的方法，甚至借鉴了 EDM2 中连续时间步的技术。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.07724">Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models</a>：Guidance 是从图像生成 Diffusion Models 中提取最佳性能的关键技术。传统上，在整个采样链中应用恒定的 guidance 权重...</li><li><a href="https://huggingface.co/docs/diffusers/en/using-diffusers/callback">Pipeline callbacks</a>：未找到描述</li><li><a href="https://github.com/huggingface/diffusers/issues/7657>">Issues · huggingface/diffusers</a>：🤗 Diffusers：PyTorch 和 FLAX 中最先进的图像和音频生成扩散模型。- Issues · huggingface/diffusers
</li>
</ul>

</div>
  

---


**LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/1227931111926009886)** (1 messages): 

- **寻找 HowTo100M 数据集**：一位成员询问是否有人可以访问 [HowTo100M 数据集](https://www.di.ens.fr/willow/research/howto100m/)，并对该请求的合适频道表示不确定。HowTo100M 是一个包含教学视频的大规模数据集。
  

---



**LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1228370900940750989)** (1 messages): 

- **LlamaIndex PandasQueryEngine 移至 Experimental**：即将发布的 **LlamaIndex (python) v0.10.29** 将把 `PandasQueryEngine` 移动到 `llama-index-experimental`。用户应使用 `from llama_index.experimental.query_engine import PandasQueryEngine` 调整代码，并通过 `pip install llama-index-experimental` 进行更新。
  

---


**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1228014660603871305)** (4 messages): 

- **与你的代码对话**：由 @helloiamleonie 编写的[新教程](https://t.co/BM5yUlCBo7)展示了如何创建一个允许你与 GitHub 仓库代码对话的应用。该教程详细介绍了使用 Ollama 等工具设置本地 LLM 和 embedding 模型的过程。
  
- **通过自动合并增强 RAG 检索**：针对 RAG 中由于朴素分块导致的上下文“断裂”问题，提出了一种[解决方案](https://t.co/0HS5FrLR9X)，涉及通过自动合并检索动态创建更连续的数据块。

- **Create-tsi 工具包发布**：与 T-Systems、Marcus Schiesser 合作，并受 Llama Index 的 [create-llama 工具包](https://t.co/x4wUgMbkfG)启发，发布了一个全新的符合 GDPR 的全栈 AI 应用工具包 create-tsi。

- **复杂 LLM 查询的自动抽象**：由 Silin Gao 等人提出的新 *Chain of Abstraction* 技术旨在克服当前框架在[不同 LLM](https://t.co/7N2y1lnlMg) 之间进行结合工具使用的多步查询规划时面临的挑战。
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1227892833701335080)** (101 messages🔥🔥):

- **微调 (Fine-Tuning) 与检索增强生成 (Retriever Augmented Generation)**：成员们讨论了 **fine-tuning** 在问答任务中的缺点，强调了知识保留效率低下和对数据集要求极高的问题。在这种情况下，**Retriever Augmented Generation (RAG)** 因其准确性、成本和灵活性而更受青睐。

- **Embedding 存储困惑已解决**：关于 embedding 存储的问题得到了澄清；它们存储在 **storage context** 内的向量存储（vector store）中。会议还提到了即将推出的 **knowledge graph 改进**，可能会简化这一过程。

- **Embedding 中元数据 (Metadata) 的澄清**：解释了在生成 embedding 和 LLM 处理过程中，默认情况下不会排除 metadata，但如果需要可以手动删除。用户讨论了如何通过提供的代码片段在代码中实现此类排除。

- **Ollama 中的 LLMs 参数设置**：一位用户询问了在使用 **Ollama 加载模型** 时如何设置 temperature 和 top_p 等 LLM 参数。提供了一个 **GitHub 代码引用**，展示了如何传递额外参数。

- **使用 Fastembed 排除向量存储问题**：讨论了 **'fastembed'** 导致 **QdrantVectorStore** 崩溃的问题，成员们建议这可能是由于混合搜索（hybrid search）的可选依赖项引起的。据报告，降级到特定版本 **('llama-index-vector-stores-qdrant==0.1.6')** 为一位用户解决了该问题。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.llamaindex]">未找到标题</a>: 未找到描述</li><li><a href="https://ai-demos.dev/">AI Demos</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1P0RiVeQQF5z09A4KxvWuYGzv2UoJUIsX?usp=sharing">Google Colaboratory</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/">Vector Stores - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example/">Starter Tutorial (OpenAI) - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/#custom-node-postprocessor">Node Postprocessor - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/index_structs/knowledge_graph/NebulaGraphKGIndexDemo/#query-with-embeddings">Nebula Graph Store - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/retrievers/relative_score_dist_fusion/?h=queryfusionre">Relative Score Fusion and Distribution-Based Score Fusion - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/2b77f89775840d6b796bcc693f7593d2aebc5fec/llama-index-integrations/llms/llama-index-llms-ollama/llama_index/llms/ollama/base.py#L56">llama_index/llama-index-integrations/llms/llama-index-llms-ollama/llama_index/llms/ollama/base.py at 2b77f89775840d6b796bcc693f7593d2aebc5fec · run-llama/llama_index</a>: LlamaIndex 是适用于您的 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_parse/blob/main/examples/demo_advanced.ipynb">llama_parse/examples/demo_advanced.ipynb at main · run-llama/llama_parse</a>: 为优化 RAG 解析文件。通过在 GitHub 上创建账号为 run-llama/llama_parse 做出贡献。</li><li><a href="https://github.com/run-llama/rags">GitHub - run-llama/rags: Build ChatGPT over your data, all with natural language</a>: 基于您的数据构建 ChatGPT，全部使用自然语言 - run-llama/rags</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/query_engine/CustomRetrievers/?">Retriever Query Engine with Custom Retrievers - Simple Hybrid Search - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_parse/blob/main/">GitHub - run-llama/llama_parse: Parse files for optimal RAG</a>: 为优化 RAG 解析文件。通过在 GitHub 上创建账号为 run-llama/llama_parse 做出贡献。</li><li><a href="https://github.com/run-llama/llama_index/pull/12736">[BUGFIX] Update LlamaIndex-Predibase Integration by alexsherstinsky · Pull Request #12736 · run-llama/llama_index</a>: 描述：Predibase API 已更改。此贡献更新了 LlamaIndex 端连接和提示 Predibase LLM 服务的实现。一旦此拉取请求被...
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1228354294034595940)** (1 条消息):

- **LlamaIndex 获得记忆增强**：一位成员分享了一篇关于使用基于 **Colbert** 的 Agent 为 LlamaIndex **增强文档检索记忆功能**的[教程](https://medium.com/ai-advances/enhancing-document-retrieval-with-memory-a-tutorial-for-llamaindex-with-colbert-based-agent-1c3c47461122)。它概述了将记忆功能集成到检索过程中的步骤，以提高性能。
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1227936138749149276)** (80 messages🔥🔥): 

- **Discord 中的 Litellm 问题**：成员们正在讨论 `litellm` 的问题，包括功能的突然中断；有人建议通过检查 `interpreter --version` 并运行 `pip show litellm` 命令来进行诊断。还有建议在 issues 频道继续讨论该问题，该问题随后在那里得到了解决。
- **OpenAI 额度方案及变更**：OpenAI 正在转向预付额度并停止按月计费；他们提供了一个[免费额度促销](https://discord.com)，用户在 2024 年 4 月 24 日前购买最低金额即可获得。成员们询问并分享了他们对这一变化如何影响不同 OpenAI 账户类型的理解。
- **社区活动邀请与回顾**：分享了一个名为 [Novus](https://lu.ma/novus28) 的温哥华初创企业建设者社区活动，强调务实的网络交流和建设。此外，还提供了关于过去一次成功使用 Open Interpreter 作为库的会议信息，并附带了一个包含入门模板的 [GitHub 仓库](https://github.com/MikeBirdTech/open-interpreter-python-templates)链接。
- **Open-Interpreter 的故障排除与修复**：一位在使用 Open-Interpreter 时遇到困难的成员收到了建议，包括从特定的 git commit 重新安装包以解决问题的命令。讨论还揭示了依赖项之间潜在的兼容性问题，并建议设置环境变量以顺利使用 Open-Interpreter。
- **通过 YouTube 和 ChatGPT 学习 Python**：有人咨询学习 Python 的最佳课程，建议了各种方法，包括 YouTube 教程、在 ChatGPT 辅助下进行基于项目的学习，以及一个针对 [Tina Huang 的 YouTube 频道](https://youtube.com/@TinaHuang1)的特定推荐。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://discord.gg/open-interpreter-1146610656779440188?event=1228084898993143920">加入 Open Interpreter Discord 服务器！</a>: 一种使用计算机的新方式 | 8248 名成员</li><li><a href="https://youtube.com/@TinaHuang1?si=aCN5X-KfXllptiyJ">Tina Huang</a>: 嗨！我是 Tina，前 Meta 数据科学家。现在我创作内容和其他互联网产品！这个频道关于编程、技术、职业和自学。我热爱学习新事物并...</li><li><a href="https://lu.ma/novus28">Novus #28 · Luma</a>: Novus 是一个供初创公司创始人、建设者和创意人士聚集、共同工作和演示的社区。没有废话。没有推销。没有政治。只有建设。议程 12:00 PM - 12:15 PM -...</li><li><a href="https://github.com/MikeBirdTech/open-interpreter-python-templates">GitHub - MikeBirdTech/open-interpreter-python-templates</a>: 通过在 GitHub 上创建账户，为 MikeBirdTech/open-interpreter-python-templates 的开发做出贡献。
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1227975819511599124)** (24 messages🔥): 

- **Poetry 安装小故障**：一位成员在安装 **Poetry** 时，由于 `poetry` 和 `pip` 都出现“未找到命令”错误，导致执行 `poetry install` 失败。建议尝试 `pip install poetry` 并在特定频道寻求进一步帮助，后来发现是 Python 本身未安装。

- **设备配置困惑**：有人在 M5 Atom 设备的 WiFi 设置过程中遇到困难，手机上没有收到输入服务器地址的提示。该问题已得到确认，并提议进行进一步测试以寻找解决方案。

- **增强文档的建议**：一位成员详细的教学内容受到了称赞，并有人提议将其纳入官方文档，该成员对此表示感谢并同意。

- **询问设备预订等待时间**：有人询问了预订设备的交付状态，得到的澄清是这些设备仍处于预生产阶段，预计在夏季发货。

- **预见制造延迟**：关于设备制造延迟的讨论强调了初创公司面临的常见挑战，指出**产品仍处于原型阶段**，并鼓励保持耐心，因为即使是“优秀的初创公司”通常也需要比预期更长的时间。
  

---

**OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1227979796085084240)** (2 messages): 

- **Transformers 进军 JavaScript**: 分享了 [transformers.js GitHub 仓库](https://github.com/xenova/transformers.js)，允许**最先进的机器学习**直接在浏览器中运行，无需服务器。这是 **HuggingFace Transformers 库**的 JavaScript 移植版本。
- **AI 模型端点揭晓**: 一名成员发布了 https://api.aime.info 的链接，推测是一个与 AI 模型相关的 API 端点，但未提供进一步的信息或背景。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://api.aime.info">AIME API Server - The Scalable Model Inference API Server</a>: 未找到描述</li><li><a href="https://github.com/xenova/transformers.js">GitHub - xenova/transformers.js: State-of-the-art Machine Learning for the web. Run 🤗 Transformers directly in your browser, with no need for a server!</a>: 适用于 Web 的最先进机器学习。直接在浏览器中运行 🤗 Transformers，无需服务器！- xenova/transformers.js
</li>
</ul>

</div>
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1227894811957399573)** (54 messages🔥): 

- **新人寻求指导**: 一名成员表达了尽管编程经验和时间有限，但仍渴望为 Axolotl 做出贡献。其他人的建议包括在 GitHub 上复现并确认问题、专注于文档工作，以及使用简单的 "print" 语句来调试代码。
  
- **期待 LLaMA-3**: 成员们就由于期待 Meta 的新 LLaMA-3 而推迟微调工作展开了辩论，一些人引用了 Meta 合作发表的一项关于语言模型中知识位缩放的研究，认为这可能是 LLaMA-3 的“秘密武器”。

- **Mistral-22B 稠密模型转换**: 分享了关于 Mistral-22B-V.01 发布的公告，该模型代表了首次将混合专家模型 (MoE) 转换为稠密模型格式。

- **讨论层冻结的价值**: 一名成员提到最近的一篇论文，建议可以移除模型一半的层而不损失性能；然而，其他人认为这可能导致过度训练，且即使移除一层也可能对模型产生重大影响。

- **开源 GPU 内核模块引发关注**: 关于 [GitHub 上支持 P2P 的开源 GPU 内核模块](https://github.com/tinygrad/open-gpu-kernel-modules) 的公告引发了讨论，表明 NVIDIA 4090s 在模型训练方面可能变得更加可行。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.05405">Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws</a>: 缩放定律描述了语言模型规模与其能力之间的关系。与以往通过 loss 或基准测试评估模型能力的研究不同，我们估计了...</li><li><a href="https://huggingface.co/fireworks-ai/mixtral-8x22b-instruct-oh">fireworks-ai/mixtral-8x22b-instruct-oh · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/Vezora/Mistral-22B-v0.1">Vezora/Mistral-22B-v0.1 · Hugging Face</a>: 未找到描述</li><li><a href="https://www.udio.com/songs/eY7xtug1dV6hbfCDhyHJua">Udio | Dune the Broadway Musical, Showtunes, Soundtrack by BobbyB</a>: 制作你的音乐</li><li><a href="https://x.com/_akhaliq/status/1778599691992571924">AK (@_akhaliq) 的推文</a>: Microsoft 发布 Rho-1。并非所有 Token 都是你需要的。以前的语言模型预训练方法对所有训练 Token 统一应用 next-token prediction loss。挑战这一观点...</li><li><a href="https://github.com/tinygrad/open-gpu-kernel-modules">GitHub - tinygrad/open-gpu-kernel-modules: NVIDIA Linux open GPU with P2P support</a>: 支持 P2P 的 NVIDIA Linux 开源 GPU。通过在 GitHub 上创建账号为 tinygrad/open-gpu-kernel-modules 做出贡献。
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1228117860325527563)** (11 messages🔥):

- **建议的 Axolotl 配置最佳实践**：一名成员指出，将最佳实践和社区见解纳入 **Axolotl 配置** 非常重要，并引用了一个 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/s/I4nAXWnFUg) 作为例子，该帖子详细介绍了 LORA 模型中层训练（layer training）的选项。
- **Flash Attention 2.0 的 GPU 初始化错误**：一位用户遇到了一个错误，提示 *“You are attempting to use Flash Attention 2.0 with a model not initialized on GPU.”*（你正尝试在未于 GPU 上初始化的模型上使用 Flash Attention 2.0）。他们表示，仅针对 11 层进行处理的解决方案奏效了。
- **逐层训练的考量**：有一场关于一次训练 11 层的讨论，推测是为了管理计算资源或内存限制。
- **Bigstral 训练查询**：一位用户确认正在训练一个名为 **Bigstral** 的模型，另一位用户则开玩笑地质疑了这个名字。
- **解冻权重子集的概念**：讨论了在训练的每一步中解冻随机权重子集的可能性，这被设计为一种适应低 GPU 资源用户的策略。有人指出，目前的实现通常仅支持在开始时冻结模型的部分。

**提到的链接**：<a href="https://www.reddit.com/r/LocalLLaMA/s/I4nAXWnFUg">Reddit - Dive into anything</a>：未找到描述

---

**OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1227983231698337852)** (3 条消息):

- **寻求逻辑数据集**：一位用户询问了专注于自然文本中**命题逻辑和谓词逻辑**推理的数据集，旨在对语言数据执行形式化推理方法。
- **寻找大规模数据集**：另一名成员表示需要一个 **2000 亿 token 的数据集** 来预训练一个新的实验性架构。针对这一需求，一位用户建议考虑 **slimpajama** 数据集并提取合适的子集。

---

**OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1227953974930440212)** (3 条消息):

- **面向初学者的 Axolotl**：分享了一篇新的博客文章，旨在帮助初学者开始使用 **[axolotl](https://drchrislevy.github.io/posts/intro_fine_tune/intro_fine_tune.html)**。作者在微调 encoder LLM 方面经验丰富，现在开始尝试训练 decoder LLM，并计划继续构建和分享关于开发“有用”模型的见解。
- **共同跨越学习曲线**：针对关于 **axolotl** 的博客文章，一位成员表示感谢，指出它可以作为该工具初学者的*良好入门指南*，并可能对其他新手有所帮助。
- **使用 Axolotl 调试数据**：一位成员提供了使用 axolotl 的技巧：在预处理期间应用 `--debug` 标志，以确保数据记录正确。这有助于避免在模型训练或评估的后期阶段出现问题。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/cleavey1985/status/1778393547571384410">Chris Levy (@cleavey1985) 的推文</a>：完成了一篇关于首次使用 @axolotl_ai 的博客文章。https://drchrislevy.github.io/posts/intro_fine_tune/intro_fine_tune.html 感谢 @jeremyphoward 推荐该工具，@HamelHusain f...</li><li><a href="https://drchrislevy.github.io/posts/intro_fine_tune/intro_fine_tune.html">Chris Levy - 开始使用 Axolotl 微调 LLM</a>：未找到描述
</li>
</ul>

</div>

---

**OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1228154378503061525)** (18 条消息🔥):

- **Colab Notebook 创建请求**：一位用户寻求帮助，希望使用预装版本的 PyTorch 创建 Google Colab notebook，旨在指定推理的 prompt。他们需要一个框架，结合 **axolotl 对来自 Hugging Face 的数据集进行基础模型 (Tiny-Llama) 的 finetune**，并针对原始模型和 finetuned 模型执行查询。
- **Axolotl Colab Notebook 可用性**：提到在 [此 GitHub 链接](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/colab-notebooks/colab-axolotl-example.ipynb) 提供了一个 Colab notebook，可以引导其使用 TinyLlama config 进行模型操作。
- **Continued Pretraining 配置**：一名成员请求一个示例 config，以便在 Hugging Face 数据集上继续预训练 TinyLlama。共享了一个详细的 pretrain 配置，用于设置和启动 pretraining 过程，其中包含针对用户任务的优化选项和环境设置。
- **使用 Docker 进行基于 DeepSpeed 的多节点 Fine-Tuning**：一位用户询问了使用 Docker 进行 **DeepSpeed 多节点 finetuning** 的步骤。提供了详细步骤，涵盖 Docker 镜像准备、DeepSpeed 配置、节点准备、Docker 容器启动以及集成 DeepSpeed 的训练脚本运行。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/colab-notebooks/colab-axolotl-example.ipynb">axolotl/examples/colab-notebooks/colab-axolotl-example.ipynb at main · OpenAccess-AI-Collective/axolotl</a>：欢迎提出 axolotl 问题。通过在 GitHub 上创建账号为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=bd900851-5f83-44e3-a3fa-2ba65b1d9dab)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=d905c33d-1397-4ef3-82ad-a16aadf1eb1f)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=2c082eba-5539-4621-b903-8d7fe0f7690a)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。
</li>
</ul>

</div>
  

---



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1227914285846368327)** (54 messages🔥): 

- **报告 OpenAI API 问题**：一名成员在使用 OpenAI Assistant API 时遇到问题，在 Python 中调用 `client.beta.messages.create` 方法时出现 `AttributeError`。他们怀疑文档相对于新版本的 OpenAI 库可能已经过时，并分享了有问题的 [代码片段](https://openai.com/api)。
  
- **AI 模型的混合体验**：讨论强调了对 Gemini 1.5 和 Claude 等各种 AI 模型的个人体验，比较了它们的 context windows、记忆召回能力以及处理代码相关查询的方式。大家认识到 API 配额的限制以及不同模型根据任务复杂度表现出的不同有效性。

- **寻求 C# 开发的最佳模型**：一名成员询问用于为 Unity 游戏引擎开发 C# 脚本的最佳 AI 模型，寻求能够一次性成功的模型。建议尝试最新的 gpt-4-turbo 和可能的 Opus，并建议直接将文档提供给 ChatGPT 以获得更好的 context。

- **处理 LLM 的大量函数**：一名成员就如何在无法传递所有函数 schemas 的情况下，让 LLM 处理 300 个函数寻求建议。对话演变为讨论使用 embeddings 作为解决方案，以及为每个函数创建简明摘要或可能将其分布在多个 Agents 之间的策略。

- **ChatGPT 知识更新的局限性**：一名成员关于当前足球队的查询得到了来自 ChatGPT 的过时信息，另一名成员解释说，由于 ChatGPT 不会实时更新其知识库，除非它被编程为浏览互联网获取更新，否则可能会提供过时信息，而这一功能在 GPT-3.5 中并不可用。
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1227880550593855539)** (8 messages🔥):

- **GPT-4 速度不一致的报告**：一位用户对 **GPT-4** 变慢表示担忧，而其他人则认为可能是 **Wi-Fi** 问题，尽管原用户声称其网络运行正常。
- **GPT-4 Turbo 在 Function Calls 方面能力较弱**：有消息指出，**新的 GPT-4-turbo 模型**在 Function Calling 方面的效率显著降低，但未提供进一步的背景或支持证据。
- **访问 GPT-4 Turbo**：一位用户询问如何验证他们是否可以在网站上访问 **GPT-4-turbo** 模型，但未提供更多细节或说明。
- **使用 GPT 编辑大型文档**：一位成员询问使用 **GPT** 处理大型文档的可行性，质疑是否可能超出正常的 Context Window，以及如何启用文档编辑（这可能需要第三方服务）。
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1227926346437492746)** (8 messages🔥): 

- **Wolfram GPT 宇宙指南**：一位成员提供了使用 Wolfram 与 GPT 配合的直接解决方案，引导通过 [Wolfram GPT 链接](https://chat.openai.com/g/g-0S5FXLyFN-wolfram) 进行访问，并提到一旦访问过，就可以使用 **@mention** 功能。
- **Prompt Engineering 入门**：一位新成员请求学习 Prompt Engineering 的资源，并被推荐了一个名为 [Prompting Guide](http://promptingguide.ai) 的网站，该网站提供有关该主题的全面信息。

  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1227926346437492746)** (8 messages🔥): 

- **Wolfram GPT 集成说明**：用户询问如何让 GPT 与 **Wolfram** 配合工作。说明指出可以通过使用 [Wolfram GPT](https://chat.openai.com/g/g-0S5FXLyFN-wolfram) 并在对话中使用 `@mention` 功能来实现。

- **开始学习 Prompt Engineering**：一位社区新成员寻求 **prompt-engineering** 的资源。他们被引导至一个名为 [promptingguide.ai](http://promptingguide.ai) 的实用网站。
  

---



**DiscoResearch ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/1227941876020023377)** (13 messages🔥): 

- **Mistral 扩展至 22B**：一个名为 [Mistral-22b-V.01](https://huggingface.co/Vezora/Mistral-22B-v0.1) 的全新 **22B 参数 Dense 模型**已经发布，这一里程碑引发了热烈讨论。该模型是一个压缩的 MoE，被转换为 Dense 形式，被誉为第一个成功的 MoE 到 Dense 模型的转换。

- **讨论 Mergekit 的挑战**：社区关于使用 Mergekit 将模型转换为 MoE 模型并进行后续微调的实验报告了令人失望的结果；通常，这些自定义的 MoE 合并模型表现不如原始模型，且目前尚未发布任何优越的 MoE 合并模型。

- **介绍 Zephyr 141B-A35B**：新的 Zephyr 141B-A35B 模型已发布，这是一个使用名为 ORPO 的新型算法训练的助手模型，是 Mixtral-8x22B 的微调版本，在强大的硬件上仅用 1 小时多一点的时间完成了 [7k 实例](https://huggingface.co/papers/2403.07691) 的训练。该模型可以在 [HuggingFace 的模型库](https://huggingface.co/HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1) 中找到。

- **Mixtral vs. SFT 性能之争**：关于 Supervised Fine-Tuning (SFT) Mixtral 模型与原始 Mixtral Instruct 性能的讨论，一些成员断言在特定领域进行 SFT 的效果优于通过 Mergekit 创建的 MoE 模型。

- **关于微调 22b 模型的疑问**：社区成员很好奇并询问是否有人成功微调了 22b 模型，特别是考虑到官方 Mixtral 模型在 Routing 方面可能拥有的“秘密配方”，这可能导致 Mixtral Instruct 的性能优于微调变体。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/Vezora/Mistral-22B-v0.1">Vezora/Mistral-22B-v0.1 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1">HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1 · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1227878944443858954)** (4 messages): 

- **关于德语语言 Benchmark 的咨询**：一位用户表示有兴趣看到模型在“德语语言 Benchmark”上运行。随后讨论了德语 Benchmark 的相关性，并提到了 *lm eval harness* 的常用性。

- **获取完整模型评估输出**：一个包含 *Open LLM Leaderboard* 完整评估输出的数据集已发布。对于 **Mixtral-8x22B-v0.1** 模型，可以通过 [Hugging Face](https://huggingface.co/datasets/open-llm-leaderboard/details_mistral-community__Mixtral-8x22B-v0.1) 访问该数据集。

**提到的链接**：<a href="https://huggingface.co/datasets/open-llm-leaderboard/details_mistral-community__Mixtral-8x22B-v0.1">open-llm-leaderboard/details_mistral-community__Mixtral-8x22B-v0.1 · Hugging Face 数据集</a>：未找到描述

  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1227897985388777472)** (22 条消息🔥): 

- **DiscoLM 70b 德语能力受到质疑**：一位用户询问是否对 [DiscoLM 70b](https://huggingface.co/DiscoResearch/DiscoLM-70b#dataset) 进行了消融实验（Ablations），这是一个具有 700 亿参数、经过 650 亿 Token 德语预训练的模型。回复提到由于其他优先级，目前尚未进行消融实验，但计划很快推出带有改进数据集的新模型。
- **英语+德语微调（Finetuning）平衡**：成员们讨论了在微调像 DiscoLM 70b 这样的模型时，英语和德语数据之间的理想平衡。有人担心在英语微调后，可能会削弱模型之前加强的德语能力。
- **通过微调探索语言细微差别**：一位用户提供了一篇[论文链接](https://arxiv.org/pdf/1911.02116.pdf)，讨论了多语言模型微调，但对该过程中语言不平衡的影响表示不确定。另一篇论文提出了一个评估大型语言模型内跨语言知识对齐的框架，可在[此处](https://arxiv.org/html/2404.04659v1)查看。
- **Occiglot-7B-DE-EN-Instruct 的成就**：一位用户展示了他们在 Occiglot-7B-DE-EN-Instruct 上的工作，表明其在基准测试中表现良好，这暗示英语和德语数据的混合可能是有效的。然而，他们警告说，目前的德语基准测试不足以进行深入分析，并分享了 [Occiglot Research](https://huggingface.co/occiglot/occiglot-7b-de-en-instruct) 页面。
- **在语言模型预训练中利用 SFT 数据**：讨论了在预训练阶段（而非仅在标准 SFT 期间）加入有监督微调（SFT）数据的好处。这次讨论是由 [StableLM 的技术报告](https://arxiv.org/abs/2402.17834)和 [MiniCPM](https://arxiv.org/abs/2404.06395) 的发现引发的，表明在预训练阶段包含 SFT 数据可能有助于防止过拟合（Overfitting）并增强泛化（Generalization）能力。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2211.01786">通过多任务微调实现跨语言泛化 (Crosslingual Generalization through Multitask Finetuning)</a>：多任务提示微调（MTF）已被证明有助于大型语言模型在零样本设置下泛化到新任务，但到目前为止，对 MTF 的探索主要集中在英语数据和模型上……</li><li><a href="https://huggingface.co/occiglot/occiglot-7b-de-en-instruct">occiglot/occiglot-7b-de-en-instruct · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/DiscoResearch/DiscoLM-70b#dataset">DiscoResearch/DiscoLM-70b · Hugging Face</a>：未找到描述</li><li><a href="https://arxiv.org/html/2404.04850v1">Lucky 52：指令微调大型语言模型需要多少种语言？(Lucky 52: How Many Languages Are Needed to Instruction Fine-Tune Large Language Models?)</a>：未找到描述</li><li><a href="https://arxiv.org/html/2404.04659v1">多语言预训练和指令微调改善了跨语言知识对齐，但程度较浅 (Multilingual Pretraining and Instruction Tuning Improve Cross-Lingual Knowledge Alignment, But Only Shallowly)</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2402.17834">Stable LM 2 1.6B 技术报告</a>：我们推出了 StableLM 2 1.6B，这是我们新一代语言模型系列中的首款产品。在本技术报告中，我们详细介绍了用于构建 Base 和 Instruct 模型的训练数据和过程……
</li>
</ul>

</div>
  

---



**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1228131664027189327)** (12 条消息🔥):

- **提及 Claude 的演进**：Nathan Lambert 强调了从 **Claude 2 到 Claude 3** 的转变，质疑了这种变化的意义，并将其描述为“[渐进式的 (INCREMENTAL)](https://vxtwitter.com/natolambert/status/1778571382491947450)”。
- **质疑 Hard Fork**：Lambert 表达了沮丧，认为本周的 **hard fork** 似乎与当前的 AI 发展不一致，并觉得这“让他感到不安 (triggering me)”。
- **寻求 Open Data 细节**：Lambert 注意到 AI 社区内似乎缺乏关于 **open data** 的详细讨论，并用一声“咳咳 (AHEM)”可能暗示呼吁更多的关注。
- **审视对 AI 的新闻观点**：Eugene Vinitsky 观察到，大众科技记者可能对技术怀有厌恶感，他认为这是“最奇怪的事情”。

**提到的链接**：<a href="https://fxtwitter.com/_arohan_/status/1778657434976022863?s=46">rohan anil (@_arohan_) 的推文</a>：有趣！“回答以下多选题。你回答的最后一行应采用以下格式：'ANSWER: $LETTER'（不带引号），其中 LETTER 是 ABCD 之一。...

  

---


**Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1227998829140377671)** (6 messages): 

- **模型训练中的混合方法**：一位成员指出，在实践中，**pretraining**、**Supervised Fine-Tuning (SFT)** 数据集和 **Reinforcement Learning from Human Feedback (RLHF)** 提示词在训练过程中经常被**混合在一起**，但也承认这一点并没有被清晰地记录在文档中。
- **关于“混合 (Blend)”的澄清**：该成员澄清说，他们所说的“混合”是指使用 **curriculums**、**schedulers** 或 **annealing** 等机制来结合不同的训练方法，尽管缺乏明确的文档说明。
- **文档与知识共享**：该成员承诺很快会分享更多专门关于 **annealing** 的信息，暗示了即将发布的流程见解。
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1227988757832138762)** (12 messages🔥): 

- **作为祝贺的 Meme**：讨论显示有人可能输入了一个祝贺短语作为 *meme*，这被认为很有趣。
- **订阅困惑已消除**：当被问及订阅是否需要接受时，澄清了**接受功能已关闭**，使订阅过程自动完成。
- **潜在的新服务器成员**：有人猜测 **Satya** 可能会加入服务器，有人暗示已经推荐了他，随后是确认以及关于需要进行一些招募的说明。
  

---


**Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1228425858675507262)** (1 messages): 

- **考察 Google 的 CodecLM**：一位成员分享了一篇关于 [CodecLM 的论文](https://arxiv.org/pdf/2404.05875.pdf)，这是 Google 使用**定制合成数据**对齐语言模型的方法。该成员观察到，这似乎是“向更强大的模型学习 (learn-from-a-stronger-model)”策略的又一个案例。
  

---


**Interconnects (Nathan Lambert) ▷ #[sp2024-history-of-open-alignment](https://discord.com/channels/1179127597926469703/1223784028428177510/1228412327486029916)** (1 messages): 

- **LLaMA 研究分享**：分享了一个指向 [**“LLaMA: Open and Efficient Foundation Language Models”**](https://huggingface.co/collections/natolambert/aligning-open-language-models-66197653411171cc9ec8e425) 论文的链接，该论文已发布并可在 Hugging Face 上访问。该论文发表于 **2023 年 2 月 27 日**。

**提到的链接**：<a href="https://huggingface.co/collections/natolambert/aligning-open-language-models-66197653411171cc9ec8e425">aligning open language models - a natolambert Collection</a>：未找到描述

  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1227961634937508011)** (18 messages🔥): 

- **缓存一致性与性能**：在讨论缓存层级时，一条消息强调了像 L1 这样的低级缓存由于一致性管理较少而速度更快。异构共享缓存池虽然比从 RAM 传输数据到 VRAM 快，但仍无法与具有专用 CCX 缓存的 CPU 上直接从 L3 到 L1 的缓存传输速度相媲美。

- **编程语言的可移植性与安全性**：一位成员认为 ANSI C 默认在所有硬件上都受支持，且易于移植到硬件描述语言。相反，另一位成员分享了一个[指向 Rust 漏洞详情的链接](https://www.cvedetails.com/vulnerability-list/vendor_id-19029/product_id-48677/Rust-lang-Rust.html)，批评了将 Rust 视为编程安全“银弹 (magic bullet)”的看法。

- **关于 Rust Foundation 政策的争议**：一位用户指出了 [Rust Foundation 针对使用 “Rust” 一词以及修改 Rust 徽标的限制性政策](https://lunduke.substack.com/p/the-rust-foundation-goes-to-war-against)，并将其与 Oracle 和 Red Hat 等组织进行比较，由于“政治不安全性”和许可限制，他们避开了这些组织。

- **AI 律师项目愿景**：一位个人表示拒绝接受任何会让外部控制其项目的许可，特别提到了构建 AI 律师的愿景，并希望避免因法律战（lawfare）导致被收购或破产。

- **因离题讨论而被 Discord 封禁**：针对用户关于编程语言的离题言论，**George Hotz** 澄清说，进一步的无关讨论将导致封禁，随后一名名为 **endomorphosis** 的用户因发布无贡献消息而被封禁。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.cvedetails.com/vulnerability-list/vendor_id-19029/product_id-48677/Rust-lang-Rust.html">Rust-lang Rust : Security vulnerabilities, CVEs </a>：Rust-lang Rust 的安全漏洞：影响该产品任何版本的漏洞列表 </li><li><a href="https://lunduke.substack.com/p/the-rust-foundation-goes-to-war-against">The Rust Foundation goes to war against people using the word &quot;Rust&quot;</a>：说真的。这篇文章的标题违反了新的 Rust 商标政策。这太疯狂了。
</li>
</ul>

</div>
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1227946197629665290)** (6 messages): 

- **Tiny 命名难题**：Discord 聊天参与者幽默地集思广益，为潜在的项目名称建议了 **tinyxercises** 和 **tinyproblems**。
- **名称认可**：一位参与者对集思广益的名称做出了积极回应，用简洁的 “ayyy” 表示偏好。
- **Tiny 形式的感谢**：另一条回复以创意的方式表达了感谢，在聊天中创造了 **tinythanks** 一词。
  

---



**Skunkworks AI ▷ #[datasets](https://discord.com/channels/1131084849432768614/1131669182124138616/1227982498655633450)** (7 messages): 

- **寻求逻辑推理数据集**：一位用户询问了关于**对自然文本进行形式逻辑推理**的数据集。另一位用户提供了一个[精选列表](https://github.com/neurallambda/awesome-reasoning)，其中包含数学、逻辑和推理数据集的资源。
- **符号求解器和 LLM 的资源共享**：用户交换了链接，包括名为 [Logic-LLM](https://github.com/teacherpeterpan/Logic-LLM) 的 GitHub 仓库，这是一个旨在为语言模型赋能符号求解器的项目。
- **关于大语言模型 Coq 的学术工作**：分享了一个 [arXiv 论文](https://arxiv.org/abs/2403.12627)链接，该论文讨论了一个旨在提高 LLM 理解和生成 Coq 代码能力的数据集。
- **关于推理项目的澄清**：
  - 一位用户寻求关于一个项目的澄清，该项目旨在通过将人类文本翻译成 Lisp 并执行，来**增强现有的 LLM 架构**以实现更好的推理。
  - 解释强调了利用预设 LLM 并通过**在潜空间（latent space）中进行计算**并保持端到端可微性来增强推理能力的目标。
- **推理资源汇编更新**：确认已将推荐资源添加到 **awesome-reasoning 仓库**，该仓库是一个旨在辅助推理 AI 开发的集合。更新已通过 [commit history](https://github.com/neurallambda/awesome-reasoning/commits/master/) 确认。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2403.12627">Enhancing Formal Theorem Proving: A Comprehensive Dataset for Training AI Models on Coq Code</a>：在形式化定理证明领域，Coq 证明助手因其验证数学断言和软件正确性的严谨方法而脱颖而出。尽管人工智能取得了进步...</li><li><a href="https://github.com/neurallambda/awesome-reasoning/commits/master/">Commits · neurallambda/awesome-reasoning</a>：一个为推理 AI 精选的数据列表。通过在 GitHub 上创建账户，为 neurallambda/awesome-reasoning 的开发做出贡献。</li><li><a href="https://github.com/teacherpeterpan/Logic-LLM">GitHub - teacherpeterpan/Logic-LLM: The project page for &quot;LOGIC-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning&quot;</a>："LOGIC-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning" 的项目页面 - teacherpeterpan/Logic-LLM</li><li><a href="https://github.com/neurallambda/awesome-reasoning">GitHub - neurallambda/awesome-reasoning: a curated list of data for reasoning ai</a>：一个为推理 AI 精选的数据列表。通过在 GitHub 上创建账户，为 neurallambda/awesome-reasoning 的开发做出贡献。
</li>
</ul>

</div>
  

---



**LLM Perf Enthusiasts AI ▷ #[claude](https://discord.com/channels/1168579740391710851/1168582222194933860/1228312970837364767)** (3 条消息): 

- **Haiku 速度受到质疑**：一位成员对 **Haiku** 提出了担忧，质疑其速度提升，而这曾被认为是该模型的主要优势。

- **吞吐量 vs. 响应时间**：另一位成员强调，他们最关心的不是 **throughput**（吞吐量），而是使用 **Haiku** 时的 **total response time**（总响应时间）。
  

---


**LLM Perf Enthusiasts AI ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/1228225471364206602)** (4 条消息): 

- **对 Turbo 的热烈反应**：一位成员询问了社区对新版 **turbo** 的看法。
- **代码熟练度提升**：另一位参与者确认，新版 **turbo** 在处理代码方面确实表现更好。
- **增强的速度性能**：还有人提到新版 **turbo** 具有更快的性能表现。
- **重新激活 Plus 以探索 Turbo**：针对这些反馈，一位成员考虑重新激活他们的 **ChatGPT Plus** 来测试新版 **turbo**。
  

---



**Alignment Lab AI ▷ #[ai-and-ml-discussion](https://discord.com/channels/1087862276448595968/1087876677603958804/)** (1 条消息): 

fredipy: <@748528982034612226>
  

---


**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1228342844377403512)** (4 条消息): 

- **代码协助请求**：一位成员寻求代码方面的帮助，并直接请求私信（DM）。
- **对服务器邀请的担忧**：另一位成员对服务器上普遍存在的 Discord 邀请感到沮丧，并提议禁止这些邀请以防止此类问题。
  

---


**Alignment Lab AI ▷ #[oo2](https://discord.com/channels/1087862276448595968/1176548760814375022/)** (1 条消息): 

aslawliet: 这个项目还活跃吗？
  

---



**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1228020142512804032)** (4 条消息): 

- **Gemini 升级 - 视频音频处理能力**：Gemini 在视频中回答音频相关问题的能力已在 AI 课程中进行了测试，显示出显著改进（此前仅能生成不含音频的描述）。
- **Google 粘贴痛点**：成员们分享了在将文本粘贴到 Google Playground 时遇到的文本格式问题，希望能有解决方案。
- **STORM 带来震撼**：重点介绍了 [GitHub 上的 STORM 项目](https://github.com/stanford-oval/storm)，展示了一个 **LLM 驱动的知识策展系统**，它可以研究某个主题并生成带有引用的完整报告。

**提到的链接**：<a href="https://github.com/stanford-oval/storm">GitHub - stanford-oval/storm: An LLM-powered knowledge curation system that researches a topic and generates a full-length report with citations.</a>：一个 LLM 驱动的知识策展系统，可以研究某个主题并生成带有引用的完整报告。 - stanford-oval/storm

  

---


**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1228041301212008459)** (1 条消息): 

- **Pull Request 修复 macOS Zsh 命令挂起问题**：MacOS 上 [llm-cmd](https://github.com/simonw/llm-cmd/pull/12) 的一个导致终端挂起的问题已通过新的 Pull Request 解决。已确认可在 M1 MacOS Terminal 和使用 zsh 的 Alacritty 上运行。

**提到的链接**: <a href="https://github.com/simonw/llm-cmd/pull/12">fix: macos zsh llm cmd hangs by nkkko · Pull Request #12 · simonw/llm-cmd</a>: 修复了 #11，已在 M1 MacOs (14.3.) 的 Terminal 和 Alacritty (zsh) 中测试，现在运行正常。

  

---



**Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1228056986071990373)** (1 条消息): 

```html
<ul>
  <li><strong>Gradio UI for Figma 发布：</strong> Mozilla Innovations 推出了 <strong>Gradio UI for Figma</strong>，这是一个基于 Hugging Face 的 Gradio 的库，旨在促进设计阶段的快速原型设计。在 <a href="https://www.figma.com/@futureatmozilla">Figma 此处</a>访问工具包。</li>
  <li><strong>加入 Gradio UI 讨论：</strong> 关于 <strong>Gradio UI for Figma</strong> 与来自 Mozilla’s Innovation Studio 的 Thomas Lodato 的对话线程已开启，供有兴趣进一步讨论该工具的人员使用。通过 <a href="https://discord.com/channels/1089876418936180786/1091372086477459557/1228056720132280461">此线程</a>加入 Discord。</li>
</ul>
```

**提到的链接**: <a href="https://www.figma.com/@futureatmozilla">Figma (@futureatmozilla) | Figma</a>: 来自 Mozilla Innovation Projects (@futureatmozilla) 的最新文件和插件 —— 我们正在构建专注于创建更加个性化、私密和开源互联网的产品。

  

---


**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1228168554386554891)** (4 条消息): 

- **在 Llamafile 中探索 OCR**: 一位成员询问了 **llamafile** 的 *OCR 能力*，引发了对其潜在用途的兴趣。
- **深度学习中的 Rust - 探索 Burnai 的呼吁**: 一位成员称赞了他们发现的一个项目 **Burnai**，该项目使用 Rust 进行深度学习推理，并建议社区调查其在跨平台推理方面极具前景的优化。他们欣赏 [justine.lol/matmul](https://justine.lol/matmul/?ref=dailydev) 上的相关工作，并在 [burn.dev](https://burn.dev/) 分享了关于 Burnai 的信息，强调了其对性能的关注。
- **Llamafile 已通过 Mcaffee 认证**: **llamafile 0.7 binary** 已被 Mcaffee 列入白名单，一位成员用庆祝表情符号指出。
- **热烈欢迎新成员**: 一位新成员在频道中打招呼，对富有成效的合作和讨论表示热忱。

**提到的链接**: <a href="https://burn.dev/">Burn</a>: 未找到描述

  

---



**AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1228029803962302546)** (4 条消息): 

- **寻找 Jamba 代码**: 一位用户表示有兴趣寻找 **Jamba** 的源代码。
- **等待更新**: 一位用户询问是否有任何更新，暗示是对之前消息或正在进行的讨论的后续跟进。
- **分享模型合并仓库**: 一位用户分享了一个 GitHub 仓库链接 ([moe_merger](https://github.com/isEmmanuelOlowe/moe_merger/tree/master))，详细介绍了他们合并模型的过程。他们指出该方法尚未经过彻底测试。
- **对分享资源的感谢**: 另一位用户对分享的模型合并仓库表示感谢。