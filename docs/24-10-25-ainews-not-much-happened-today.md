---
companies:
- liquid-ai
- anthropic
- cohere
- openai
- meta-ai-fair
- nvidia
- perplexity-ai
- langchain
- kestra
- ostrisai
- llamaindex
date: '2024-10-26T00:52:03.375131Z'
description: '以下是为您翻译的中文内容：


  **Liquid AI** 举办了发布会，推出了全新的基础模型。**Anthropic** 分享了关于社交偏见和特征控制（feature steering）的后续研究，并展示了其“金门大桥版
  Claude”（Golden Gate Claude）功能。继 Aya Expanse 之后，**Cohere** 发布了多模态 Embed 3 嵌入模型。关于
  **GPT-5/Orion** 的虚假信息已被 **Sam Altman** 澄清。**Meta AI FAIR** 宣布了 **Open Materials
  2024**，推出了用于无机材料发现的新模型和数据集，采用了 EquiformerV2 架构。**Anthropic AI** 展示了如何通过特征控制来平衡社交偏见与模型能力。**NVIDIA**
  的 **Llama-3.1-Nemotron-70B** 凭借风格控制功能在 Arena 排行榜上名列前茅。**Perplexity AI** 的周查询量已扩大至
  1 亿次，并推出了新的财务和推理模式。**LangChain** 强调了与交互式帧插值（frame interpolation）的实际应用集成。**Kestra**
  重点介绍了可扩展的事件驱动型工作流，采用开源且基于 YAML 的编排方式。**OpenFLUX** 通过引导 LoRA 训练将推理速度提高了一倍。关于 AI 安全的讨论包括人类与
  AI 之间的信任动态、AI 自动化的经济影响，以及白宫发布的针对网络和生物风险的 AI 国家安全备忘录。**LlamaIndex** 展示了知识增强型智能体（knowledge-backed
  agents），旨在提升 AI 应用能力。'
id: 979bf92e-0951-413c-86d3-cba6b11c44e8
models:
- llama-3.1-nemotron-70b
- golden-gate-claude
- embed-3
original_slug: ainews-not-much-happened-today-5313
people:
- sam-altman
- lmarena_ai
- aravsrinivas
- svpino
- richardmcngo
- ajeya_cotra
- tamaybes
- danhendrycks
- jerryjliu0
title: 今天没发生什么事。
topics:
- feature-steering
- social-bias
- multimodality
- model-optimization
- workflow-orchestration
- inference-speed
- event-driven-workflows
- knowledge-backed-agents
- economic-impact
- ai-national-security
- trust-dynamics
---

<!-- buttondown-editor-mode: plaintext -->**一个安静的周末正是你所需要的。**

> 2024/10/24-2024/10/25 的 AI 新闻。我们为你检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **32** 个 Discord（**232** 个频道和 **3136** 条消息）。预计节省阅读时间（以 200wpm 计算）：**319 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

- Liquid AI 举办了[发布会](https://www.youtube.com/watch?v=d19jhYtwgCA)（我们的[报道在此](https://buttondown.com/ainews/archive/ainews-liquid-foundation-models-a-new/)）
- Anthropic 分享了一些关于 "Golden Gate Claude" 特征转向（feature steering）的[社会偏见研究后续](https://x.com/AnthropicAI/status/1849840131412296039)
- Cohere 在 [Aya Expanse](https://x.com/CohereForAI/status/1849435983449587796) 之后推出了 [multimodal Embed 3](https://x.com/cohere/status/1848760845641388087?s=46&t=PW8PiFwluc0tdmv2tOMdEg) 嵌入模型。
- 出现了一些关于 [GPT5/Orion 的假新闻](https://x.com/sama/status/1849661093083480123?s=46&t=tMWvmS3OL3Ssg0b9lKvp4Q)。

周末愉快。

---

{% if medium == 'web' %}

**目录**

[TOC]

{% else %}

**目录**和**频道摘要**已移至此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}

---

# AI Twitter 综述

> 所有摘要均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型与研究**

- **Meta FAIR 的 Open Materials 2024**：[@AIatMeta](https://twitter.com/AIatMeta/status/1849843518493135171) 宣布发布 [Open Materials 2024](https://twitter.com/AIatMeta/status/1849843518493135171)，包含用于无机材料发现的新模型和数据集，利用 EquiformerV2 架构并支持关于结构和成分多样性的广泛数据。
  
- **Anthropic AI 的特征转向 (Feature Steering)**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1849840151490420953) 分享了他们关于特征转向的研究，展示了调整模型特征如何影响九个维度的社会偏见评分，同时确定了一个平衡有效性和能力保留的“转向甜点区（steering sweet spot）”。

- **NVIDIA 的 Llama-3.1-Nemotron-70B**：[@lmarena_ai](https://twitter.com/lmarena_ai/status/1849530739194462708) 透露 [Llama-3.1-Nemotron-70B](https://twitter.com/lmarena_ai/status/1849530739194462708) 在带有 Style Control 的情况下，目前在 Arena 排行榜上排名第 9 和第 26，展示了其在人类偏好任务中的竞争力。

- **Perplexity 的模型增强**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1849828465396285892) 强调了 Perplexity 的增长，每周查询量超过 1 亿次，并引入了 Finance 和 Reasoning Mode 等新功能，增强了其能力和用户参与度。

**AI 工具与基础设施**

- **LangChain 的应用集成**：[@hwchase17](https://twitter.com/hwchase17/status/1849562878791254513) 强调了将 [LangChain](https://twitter.com/hwchase17/status/1849562878791254513) 集成到实际应用中，支持交互式帧插值（Interactive Frame Interpolation）等功能以增强部署场景。

- **Kestra 的事件驱动工作流**：[@svpino](https://twitter.com/svpino/status/1849785499587838271) 讨论了采用 [Kestra](https://twitter.com/svpino/status/1849785499587838271) 进行可扩展的事件驱动工作流编排，强调了其开源特性、基于 YAML 的工作流以及处理数百万次执行的能力。

- **OpenFLUX 优化**：[@ostrisai](https://twitter.com/ostrisai/status/1849794125886812409) 探索了使用 OpenFLUX 训练 guidance LoRA，通过消除 CFG 将推理速度提高一倍，展示了 AI 模型的实际优化。

**AI 安全与伦理**

- **人类与 AI 的信任对比**：[@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1849635033482686905) 阐述了人类与 AI 之间的信任动态差异，强调了在 AI 驱动的研究中人类监督的重要性，以确保可靠性并防止滥用。

- **AI 的经济与智力影响**：[@ajeya_cotra](https://twitter.com/ajeya_cotra/status/1849528009956655557) 和 [@tamaybes](https://twitter.com/tamaybes/status/1849575234963374580) 讨论了由 AI 自动化驱动的深刻经济变革，预测了显著的增长率，并强调了人类智能在验证 AI 生成的发现中的关键作用。

- **白宫 AI 国家安全备忘录**：[@DanHendrycks](https://twitter.com/DanHendrycks/status/1849503657156587819) 分享了来自白宫 AI 国家安全战略的见解，重点是减轻攻击性网络操作和生物威胁中的 AI 风险，下划线了 AI 部署中国家安全措施的重要性。

**AI 应用与用例**

- **LlamaIndex 的知识增强型 Agent (Knowledge-Backed Agents)**：[@jerryjliu0](https://twitter.com/jerryjliu0/status/1849577709393056081) 展示了 [LlamaIndex workflows](https://twitter.com/jerryjliu0/status/1849577709393056081) 如何通过引入事件驱动架构和强大的状态管理来增强 AI Agent 应用，从而提升性能和可靠性。

- **Perplexity 的金融搜索 API**：[@virattt](https://twitter.com/virattt/status/1849581597097332940) 介绍了一个全新的 [Financial Search API](https://twitter.com/virattt/status/1849581597097332940)，支持在 20,000 多个股票代码中通过 100 多个过滤器进行搜索，为用户简化了金融数据的处理与分析流程。

- **销售自动化中的 AI Agent**：[@llama_index](https://twitter.com/llama_index/status/1849847301680005583) 展示了一个在 NVIDIA 内部销售 AI 助手中部署 LlamaIndex 的案例研究，详细介绍了其如何利用多 Agent 系统、并行检索和实时推理来提升销售自动化水平和效率。

**AI 社区与活动**

- **AI Agent 大师课 (AI Agents Masterclass)**：[@jerryjliu0](https://twitter.com/jerryjliu0/status/1849577709393056081) 与 [@arizeai](https://twitter.com/arizeai) 共同举办了一场 AI Agent 大师课，涵盖了使用 LlamaIndex workflows 构建知识增强型 Agent 的基础知识，重点关注事件驱动架构和状态管理。

- **播客与研讨会**：[@swyx](https://twitter.com/swyx/status/1849672393016607181) 和 [@maximelabonne](https://twitter.com/maximelabonne/status/1849552170762346774) 推广了即将举行的专注于 AI 开发、社区参与和协作学习的播客与研讨会，旨在培育充满活力的 AI 社区。

- **Meta FAIR 的开放材料研讨会 (Open Materials Workshop)**：[@maximelabonne](https://twitter.com/maximelabonne/status/1849552170762346774) 组织了一场关于 [Meta’s Open Materials](https://twitter.com/maximelabonne/status/1849552170762346774) 的研讨会，邀请 AI 研究人员和爱好者合作，利用开源模型和数据集进行无机材料发现。

**梗/幽默**

- **AI 接管世界的笑话**：[@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1849592010132553887) 幽默地将 AI 提交的内容比作爱因斯坦的作品，并设想了一个 AI 可能密谋接管世界的幽默场景。

- **有趣的 AI 预测**：[@francoisfleuret](https://twitter.com/francoisfleuret/status/1849761387649597871) 对 AI 任务算术 (task arithmetic) 和层复杂度发表了轻松的评论，将技术见解与幽默融合在一起。

- **AI 生成的音乐**：[@suno_ai_](https://twitter.com/suno_ai_/status/1849563704926273620) 分享了一首 AI 生成的歌曲，幽默地将一条推文转化为 bat gothclub 音乐主题，展示了 AI 在内容生成方面的创意和娱乐用途。

- **幽默的 AI 对比**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1849589090833891400) 拿 AI 试图撰写关于智能与秩序的论文开玩笑，强调了 AI 生成内容中那些有趣的局限性。

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

**主题 1. Meta 的量化 Llama 模型：推动端侧 AI 发展**

- **[介绍具有更高速度和更低内存占用的量化 Llama 模型](https://ai.meta.com/blog/meta-llama-quantized-lightweight-models/)** ([Score: 75, Comments: 3](https://reddit.com//r/LocalLLaMA/comments/1gb570b/introducing_quantized_llama_models_with_increased/)): Meta 发布了其 **Llama 2** 模型的**量化版本**，提供 **2-3 倍的推理加速**，并减少了 **40-60% 的内存占用**。这些新模型提供 **4-bit 和 8-bit** 量化版本，在包括 **MMLU**、**HellaSwag** 和 **TruthfulQA** 在内的各种基准测试中，保持了与全精度版本相当的性能。这些量化模型旨在为在资源受限设备上使用 LLM 的开发者提高可访问性和效率。

- **[Zuck 在 Threads 上表示：发布 Llama 1B 和 3B 端侧模型的量化版本。模型体积更小，内存效率更高，速度提升 3 倍，让应用开发更简单。💪](https://www.threads.net/@zuck/post/DBgtWmKPAzs)** ([Score: 404, Comments: 103](https://reddit.com//r/LocalLLaMA/comments/1gb4z63/zuck_on_threads_releasing_quantized_versions_of/)): Meta 发布了 **Llama 1B 和 3B 端侧模型的量化版本**，正如 **Mark Zuckerberg** 在 Threads 上宣布的那样。这些新版本提供了**更小的模型体积**、**更高的内存效率**，并且比前代产品**快 3 倍**，旨在为开发者提供更便捷的应用开发体验。
  - 新模型采用了带有 **LoRA 适配器**的**量化感知训练 (QAT)**，涉及多个训练步骤以实现高质量的量化后效果。由于数据集质量和格式的不确定性，开源社区很难复制这一过程。
  - 量化方案包括针对 Transformer 块中线性层的 **4-bit groupwise** 量化，针对分类和嵌入层的 **8-bit per-channel** 量化，并使用了针对 **Arm CPU** 后端优化的 PyTorch **ExecuTorch** 框架。
  - 用户讨论了官方模型源对企业的重要性，一些人表达了在使用 **Qwen 2.5** 等模型时面临的挑战，原因是其中国背景，特别是在国防合同背景下。
- **Meta 发布量化版 Llama 模型** ([Score: 184, Comments: 25](https://reddit.com//r/LocalLLaMA/comments/1gb5ouq/meta_released_quantized_llama_models/)): Meta 发布了使用**量化感知训练 (QAT)**、**LoRA** 和 **SpinQuant** 技术量化的 **Llama 模型**，这是他们首次发布此类版本。尽管**体积大幅缩小**，这些模型仍表现出令人印象深刻的性能，其**小巧的体积**和**极快的速度**使其适合大规模部署；可以通过 [GitHub 上的 executorch](https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md) 获取并使用。
  - **QLoRA** 变体展示了令人印象深刻的结果，用户讨论了其与 **Tim Dettmers** 论文中 [QLoRA 方法](https://arxiv.org/abs/2305.14314) 的相似之处。关于在流行的量化方法中使用 QLoRA 及其对算力的依赖性也引发了讨论。
  - 大多数**训练后量化 (PTQ)** 方法（如 Q5_0 GGUF）不包含 LoRA 组件。**Meta** 使用原始数据集和早期训练阶段的方法比典型的开源 PTQ 模型具有更高的准确度。
  - 用户询问了将模型转换为 **GGUF 格式**以便在 **LM Studio** 中使用的问题，讨论指出这些较小的模型更适合手机等设备，而非 Mac。还有人对用于 Skyrim 角色扮演等应用的潜在 **128k 上下文长度**模型表示了兴趣。


**主题 2. Cerebras Inference 在 Llama 3.1-70B 上实现 2,100 Tokens/s**


- **Cerebras Inference 现在快了 3 倍：Llama 3.1-70B 突破 2,100 tokens/s** ([Score: 214, Comments: 81](https://reddit.com//r/LocalLLaMA/comments/1gbo68a/cerebras_inference_now_3x_faster_llama3170b/)): **Cerebras Inference** 实现了 **3 倍的性能提升**，现在运行 **Llama 3.1-70B** 的速度达到 **每秒 2,100 个 token**。这一性能比最快的 GPU 解决方案**快 16 倍**，比运行 **Llama 3.1-3B**（体积小 **23 倍**的模型）的 GPU **快 8 倍**，这种提升堪比新一代 GPU 的升级。**Tavus** 和 **GSK** 等公司正在使用 Cerebras Inference 进行视频生成和药物研发，其聊天演示和 API 可在 [inference.cerebras.ai](https://inference.cerebras.ai/) 获取。
  - **Cerebras CS-2** 硬件是一台 **15U 的机器**，功耗为 **23kW**，成本约为 **100-300 万美元**。它拥有 **40GB 的片上 SRAM**，并使用来自台积电的整块**披萨大小的晶圆**，而不是切割后的芯片。一段[服务器拆解视频](https://web.archive.org/web/20230812020202/https://www.youtube.com/watch?v=pzyZpauU3Ig)展示了其独特的架构。
  - 用户报告了 **Cerebras 聊天演示**中令人印象深刻的性能，特别是在**翻译任务**方面。该演示运行 **Llama 3.1 70B 和 8B** 模型，一些用户发现它优于 OpenAI 的产品。然而，也有人对 **API 使用限制**和**首个 token 延迟 (TTFT)** 表示担忧。
  - 讨论涉及了潜在的应用，包括**类 o1 模型的规模化思考**、**推理时计算扩展**以及**更好的采样器**。一些用户质疑了对比指标，建议需要标准化的衡量标准，如**“每百万 token 的瓦数”**，以便进行公平的硬件比较。


**主题 3. 新的开源 LLM 突破了上下文长度和能力的界限**

- **[INTELLECT-1：Prime Intellect AI 本月推出的突破性民主化 100 亿参数 AI 语言模型](https://app.primeintellect.ai/intelligence)** ([Score: 170, Comments: 37](https://reddit.com//r/LocalLLaMA/comments/1gbcgny/intellect1_groundbreaking_democratized/)): Prime Intellect AI 发布了 **INTELLECT-1**，这是一个 **100 亿参数 (10B)** 的 AI 语言模型，标志着民主化 AI 技术的重大进步。该模型于本月推出，旨在为更广泛的用户和开发者提供易于获取且强大的语言处理能力，有望重塑 AI 应用和研究的格局。

- **[CohereForAI/aya-expanse-32b · Hugging Face (上下文长度：128K)](https://huggingface.co/CohereForAI/aya-expanse-32b)** ([Score: 145, Comments: 57](https://reddit.com//r/LocalLLaMA/comments/1gb32p9/cohereforaiayaexpanse32b_hugging_face_context/)): **CohereForAI** 在 Hugging Face 上发布了 **Aya Expanse 32B**，这是一个具有 **128K token 上下文长度**的大语言模型。该模型代表了上下文处理能力的显著提升，能够为各种应用实现更全面、更具上下文感知能力的语言处理。
  - 用户对该模型的性能表示怀疑，许多人要求将其与 **Qwen 2.5** 进行对比。一些人指出，尽管 **Qwen** 在某些用例中拥有**更好的许可证和输出效果**，但**美国和欧洲公司**似乎忽视了它的成就。
  - 讨论中提到了模型可能存在的**配置错误**，因为 `max_position_embeddings` 的值 (**8192**) 与声明的 **128K token 上下文长度**不符。这一问题与 CohereForAI 之前发布的一个版本类似，正如 [Hugging Face 讨论帖](https://huggingface.co/CohereForAI/c4ai-command-r-v01/discussions/12)中所述。
  - 该模型的 **8B 版本** 经过测试后被发现是**高度对齐且带有道德说教倾向**的，会拒绝一些看似平凡的请求。用户指出，该模型的主要目的是用于**翻译任务**，而非通用用途，其 **q8 GGUF 版本**已在 [Hugging Face](https://huggingface.co/NikolayKozloff/aya-expanse-8b-Q8_0-GGUF) 上提供。


**主题 4. 为开发者和移动用户改进 LLM 集成**

- **[VSCode + Cline + VLLM + Qwen2.5 = 快速](https://v.redd.it/rzpcacfg9rwd1)** ([Score: 99, Comments: 29](https://reddit.com//r/LocalLLaMA/comments/1gbb2de/vscode_cline_vllm_qwen25_fast/)): 该帖子描述了集成 **VSCode**、**Cline**、**VLLM** 和 **Qwen2.5** 以实现快速编码辅助的方法。这种配置利用 VLLM 的速度和 Qwen2.5 模型的能力，在 VSCode 环境中实现了快速的本地 AI 驱动代码补全与生成。

- **ChatterUI v0.8.0 发布 - 现支持外部模型加载！** ([Score: 35, Comments: 13](https://reddit.com//r/LocalLLaMA/comments/1gb6jtk/chatterui_v080_released_now_with_external_model/)): **ChatterUI v0.8.0**（一款针对 LLM 的 Android UI）已发布，带来了重大更新，包括**外部模型加载**功能。该应用现在将**远程模式和本地模式**分开，本地模式允许用户自定义和使用设备端模型，而远程模式则支持连接到各种 API。主要改进包括受 **Pocket Pal** 启发的新模型列表，可显示从 **GGUF 文件**中提取的元数据，以及包含 CPU 设置和本地特定应用选项的模型设置页面。

**主题 5. LLM 基准测试与评估工具的进展**

- **只需一行代码即可对 GGUF 模型进行基准测试** ([Score: 45, Comments: 20](https://reddit.com//r/LocalLLaMA/comments/1gb7x5z/benchmark_gguf_models_with_one_line_of_code/)): 该帖子介绍了一个用于通过单行代码对 **GGUF 模型**进行**基准测试**的**开源工具**，解决了本地评估量化模型的挑战。该工具支持**多进程**、**8 个评估任务**，并号称是 GGUF 模型**最快的基准测试**工具。示例显示，在 **4090 GPU** 上使用 **4 个 worker** 对 "ifeval" 数据集进行 **Llama3.2-1B-Instruct Q4_K_M** 模型评估耗时 **80 分钟**。
  - 用户表达了在不上传的情况下**测试自定义模型**的兴趣，特别是用于比较**静态量化与 imatrix 量化**。该工具在评估各种模型类型方面的灵活性受到了关注。
  - 有人提出了在 **MacBook Pro M1** 等设备上测量特定模型**功耗和效率**的可能性，表明了对速度以外性能指标的兴趣。
  - 用户对在不同硬件（包括 **AMD Ryzen GPU**）上测试该基准测试工具表现出热情，这表明他们希望在各种 GPU 架构之间实现更广泛的兼容性和性能比较。
- **[使用 MLC LLM 和 Mistral Large Instruct 2407 q4f16_1 在 4X RTX 3090 上进行功耗扩展测试。测试范围为 150 - 350 瓦。](https://i.redd.it/nlt4dcnz3qwd1.png)** ([Score: 44, Comments: 23](https://reddit.com//r/LocalLLaMA/comments/1gb5ii7/power_scaling_tests_with_4x_rtx_3090s_using_mlc/)): 使用 **4 张 RTX 3090 GPU**、**MLC LLM** 和 **Mistral Large Instruct 2407 q4f16_1** 进行了功耗扩展测试，探索了 **150 到 350 瓦**的功率范围。实验旨在评估这些高端 GPU 在不同功率水平下运行大语言模型（LLM）的性能和效率。
  - **SuperChewbacca** 使用提示词 "Write exactly 100 digits of pi" 进行测试，在聊天模式下运行 **MLC LLM**，并设置 **tensor parallel shards=4**。他们对 MLC LLM 的速度和持续 **100% 的 GPU 利用率**表示赞赏。
  - 用户表示有兴趣将 **MLC LLM 的性能**与 **vLLM** 在 **Mistral-large** 上的表现进行对比，特别是在张量并行效率方面。原作者同意在 vLLM 中进行可比的量化测试。
  - 有人请求在未来的基准测试中加入 **Ollama** 和 **vLLM**，以便在 **4x3090 配置**下对这三种解决方案进行全面的 **tok/s 比较**。

## 其他 AI Subreddit 回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 模型发布与功能**

- **Mochi 1 视频生成模型**：一款名为 Mochi 1 的新型开源 AI 模型展示了令人印象深刻的视频生成能力。经过一些优化后，它可以在单张 24GB VRAM GPU 显卡上运行。该模型可以以 fp8 精度生成 24 fps 的 15 秒视频，或以 bf16 精度生成 2.5 秒视频。[分享了一份详细指南](https://www.reddit.com/r/StableDiffusion/comments/1gb07vj/how_to_run_mochi_1_on_a_single_24gb_vram_card/)，介绍如何设置并在本地运行它。

- **Anthropic 的 Claude 3.5 模型**：Anthropic 发布了具有 "computer use" 功能的新型 Claude 3.5 模型，允许 AI 直接与计算机界面交互。这被视为迈向能够自动化计算机任务的 AI Agent 的重要一步。[此次发布引发了讨论](https://www.reddit.com/r/singularity/comments/1gb5ru4/computer_use_is_a_big_deal/)，关于其对知识工作和自动化的潜在影响。

- **OpenAI 的下一个模型**：关于 OpenAI 计划在 12 月前发布代号为 "Orion" 的新 AI 模型的报道存在矛盾。[虽然一些消息来源对此进行了报道](https://www.reddit.com/r/OpenAI/comments/1gbjyfj/openai_plans_to_release_its_next_big_ai_model_by/)，但 OpenAI CEO Sam Altman [将其斥为 "假新闻"](https://www.reddit.com/r/OpenAI/comments/1gblt88/sam_replies_stating_that_the_article_wasnt_true/)。这些矛盾的信息引发了 AI 社区的诸多猜测。

**AI 研究与技术**

- **Google DeepMind 的多模态学习**：[来自 Google DeepMind 的新论文](https://arxiv.org/html/2406.17711v1) 展示了通过联合样本选择进行数据策展如何加速多模态学习。

- **Microsoft 的 MInference**：[Microsoft 推出了 MInference](https://arxiv.org/abs/2407.02490)，这项技术可以在保持准确性的同时，为长上下文任务实现多达数百万 Token 的推理，并显著提高支持模型的速度。

- **扩展合成数据生成**：一篇关于[扩展合成数据生成](https://www.reddit.com/r/MachineLearning/comments/1dzergu/r_scaling_synthetic_data_creation_with_personas/)的论文利用 LLM 中的多样化视角，从网络数据策展的 10 亿个 Persona 中生成数据。

**AI 模型改进**

- **Salesforce 的 xLAM-1b 模型**：Salesforce 发布了 xLAM-1b，这是一个 10 亿参数的模型，尽管体积相对较小，但在[函数调用（function calling）方面实现了 70% 的准确率，超过了 GPT 3.5](https://www.reddit.com/r/LocalLLaMA/comments/1dz8g10/salesforce_tiny_giant_xlam1b_model_surpasses_gpt/)。

- **Phi-3 Mini 更新**：Rubra AI 发布了更新后的 Phi-3 Mini 模型，[具有函数调用功能](https://www.reddit.com/r/LocalLLaMA/comments/1dzhe38/phi3_mini_june_with_function_calling/)，可与 Mistral-7b v3 竞争，并优于基础版 Phi-3 Mini。

**AI 伦理与社会影响**

- AI 能力的快速进步，特别是在自动化计算机任务方面，[引发了关于潜在工作取代的讨论](https://www.reddit.com/r/singularity/comments/1gb5ru4/computer_use_is_a_big_deal/) 以及对全民基本收入 (UBI) 等解决方案的需求。

- 关于 [AI 权力集中在少数公司手中](https://www.reddit.com/r/OpenAI/comments/1gbjyfj/openai_plans_to_release_its_next_big_ai_model_by/) 的辩论，一些人批评 OpenAI 显然背离了其最初的开源章程。

**硬件与基础设施**

- [据报道，台积电 (TSMC) 亚利桑那州的芯片生产良率超过了台湾](https://www.reddit.com/r/singularity/comments/1gbd2he/tsmcs_arizona_chip_production_yields_surpass/)，这被视为美国半导体制造努力的一次胜利。


---

# AI Discord 简报

> 由 O1-preview 生成的摘要之摘要的总结

**主题 1. AI 模型与硬件取得新突破**

- [**Cerebras 芯片让 GPU 望尘莫及**](https://x.com/CerebrasSystems/status/1849467762932076801)：**Cerebras** 推出了一款芯片，可提供 **3 倍更快的推理速度**，在 **Llama3.1-70B** 上实现了超过 **2,100 tokens/s** 的表现，比最快的 GPU 快了 **16 倍**。这一飞跃使 Cerebras 成为 AI 处理速度领域的重量级选手。
- [**Intel Arc A750 表现令人惊喜**](https://store.intel.com/shop/ca/en_CA/ARC-A750-16GB-Limited)：升级到 **Intel Arc A750** 后，用户发现其在 **LM Studio** 中的表现令人印象深刻，超越了之前的 **6750xt** 等配置。这突显了 Arc 在机器学习任务中的潜力。
- [**Meta 发布极速量化版 Llama 模型**](https://ai.meta.com/blog/meta-llama-quantized-lightweight-models/)：**Meta** 发布了 **Llama 3.2 1B & 3B** 的量化版本，将推理速度提升了高达 **4 倍**。这些模型旨在用于端侧部署，兼顾了速度与性能。

**主题 2. AI 中的伦理挑战与隐私担忧**

- [**Claude 3.5 成为“老大哥”**](https://bdtechtalks.com/2024/10/22/anthropic-claude-computer-use/)：全新的 **Claude 3.5 Sonnet** 可以监控屏幕并控制设备，引发了严重的隐私红旗。用户正在讨论拥有此类侵入性能力的 AI 的伦理问题。
- [**Deepfake 技术真实得令人不安**](https://www.reddit.com/r/notebooklm/comments/1gbf3sk/deep_dive_epicurus_descartes_and_god/)：在 **Notebook LM** 上，关于 Deepfake 技术伦理影响的讨论异常激烈，尤其是涉及知情同意和去人性化的问题。成员们质疑 AI 生成的虚拟形象是否能符合伦理。
- [**AI 审查引发愤怒**](https://openrouter.ai/)：**OpenRouter** 的用户担心 **hermes-3-llama-3.1-405b** 等模型可能受到审查，害怕内容受到限制。社区正在辩论应该在何处划定可接受的 AI 内容审核界限。

**主题 3. AI 工具与平台的用户体验**

- [**LM Studio 用户强烈要求立即支持插件！**](https://github.com/nomic-ai/gpt4all/issues/484)：用户们齐声呼吁 **LM Studio** 支持用户创建的插件，寻求在不增加复杂性的情况下增强功能。与现有工具和 API 的更好集成是一个热门话题。
- [**Aider 迎来升级，用户欢呼**](https://aider.chat/docs/llms.html)：**Aider v0.60.1** 的发布带来了对 **Claude 3** 模型、文件排序和精美新输入标识的支持。用户对这些更新表示赞赏，并注意到通过 Prompt Caching（提示词缓存）实现的成本节约改进。
- [**Perplexity Pro 引发争议**](https://techcrunch.com/2024/10/24/they-wish-this-technology-didnt-exist-perplexity-responds-to-news-corps-lawsuit/)：**Perplexity Pro** 的推出引发了关于其相对于 **Claude** 和 **GPT** 等竞争对手价值的辩论。用户质疑性能与价格的比例，寻求优化订阅的建议。

**主题 4. AI 辅助创意成为焦点**

- [**AI 播客变得个性化且奇特**](https://notebooklm.google.com/notebook/7f340391-ef2a-43e3-a05e-071d0dc588da)：在 **Notebook LM** 上，用户发现为 AI 语音分配名称和角色可以增强生成播客的连贯性。然而，语音角色的局限性也激发了创作挑战。
- [**AI 问答游戏在计分上失误**](https://vocaroo.com/1kBMrHmdLKOp)：开发 AI 驱动的问答游戏的尝试初见成效，但在 AI 无法统计分数时遇到了困难。AI 众所周知的数学难题成了用户间的趣谈。
- [**作家通过 AI 共同作者实现进阶**](https://huggingface.co/spaces/Pixeltable/AI-RPG-Adventure)：作者们使用 AI 来充实角色和场景，发现与 AI 进行“剧本朗读”能深化叙事。这种方法挖掘出了新的背景故事和动机，提升了创意写作水平。

**主题 5. 微调 AI：挑战与最佳实践**

- [**垃圾数据进，垃圾 AI 出：数据集质量至关重要**](https://github.com/unslothai/unsloth/wiki#adding-new-tokens)：**Unsloth AI** 用户强调，微调的成功取决于高质量、平衡的数据集。不平衡的数据会导致性能不佳，凸显了妥善准备数据的必要性。
- [**微调 Llama 3.2 引发讨论**](https://huggingface.co/spaces/mteb/leaderboard)：在 **Eleuther**，成员们讨论了微调 **Llama 3.2** 进行文本分类的最佳方法。建议包括使用简单的分类器和嵌入模型，并对数据集质量保持警惕。
- [**量化技术引起关注**](https://github.com/pytorch/torchtune/blob/main/recipes/quantization.md)：在 **Nous Research AI**，社区研究了 Meta 的新量化模型，权衡了应用量化感知训练（Quantization-aware Training）的收益与复杂性。潜在的性能权衡引发了热烈辩论。

---

# 第一部分：高层级 Discord 摘要

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **H200 服务器碾压 AI 模型性能**：一场围绕使用 **H200 服务器** 运行大型模型的讨论透露，一位用户的生产服务器在处理 **405B 模型时达到了 90 teraflops**。
  
  - 针对典型 AI 应用，此类强力基础设施的成本效益和必要性引发了关注。
- **Transformers 基础与 Reddit 数据生成**：一位成员分享了他们学习 **transformers** 的进展，利用 [Andrej 的视频](https://link.to/video) 通过一个 **10M 参数模型** 取得了成果，并基于 Reddit 数据生成了 **10k tokens**。
  
  - 这一里程碑引发了关于进一步优化以及对其 [DeepLLMs 仓库](https://github.com/its-nmt05/DeepLLMs/blob/main/model_architecture.ipynb) 社区反馈的讨论。
- **引入自动化渗透测试基准**：论文 [“Towards Automated Penetration Testing: Introducing LLM Benchmark, Analysis, and Improvements”](https://arxiv.org/abs/2410.17141) 强调了一个针对使用 LLM 进行渗透测试的基准，评估了 **GPT-4o** 和 **Llama 3.1**。
  
  - 鉴于网络威胁造成了 **6 万亿美元** 的损失，讨论强调了道德黑客攻击以及有效漏洞识别基准的必要性。
- **Streamlit 计算器项目亮相**：一位成员使用 **Streamlit** 复制了一个 [计算器项目](https://github.com/dhruvyadav89300/iOS-Notes-Calculator)，并邀请大家对其实现提供反馈。
  
  - 该项目的热度与利用 HuggingFace 工具进行基因组学蛋白质分析的讨论相辅相成。
- **探索对 Hugging Face Diffusers 的贡献**：对贡献 **Hugging Face Diffusers** 的兴趣促使了阅读 [贡献指南 readme](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md) 以及识别 **good first issues** 的建议。
  
  - 随着讨论的展开，关于在不重新训练的情况下向 tensors 添加噪声的影响的疑问也随之出现，凸显了社区对技术挑战的参与度。

 

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth AI 推进模型支持**：Unsloth 目前缺乏对 **Llama 3.2** 等 **vision model** 的支持，但团队正在开发相关功能，以便在未来将其纳入。
  
  - 在视觉模型集成工作进行期间，敦促用户专注于 **基于文本的 LLM**。
- **字幕微调模型面临挑战**：一位用户报告了在微调模型以纠正 **VTT 字幕** 时的困难，问题源于训练期间时间戳的改动。
  
  - 专家建议从训练数据集中移除时间戳，以避免 **overfitting** 并增强文本纠正能力。
- **数据集质量对微调至关重要**：**LLM 微调** 的成功取决于训练数据集的 **质量和平衡**，不平衡的数据会导致性能欠佳。
  
  - 参与者强调了训练前进行适当 **数据集准备** 的重要性。
- **数据中心蓬勃发展，增长 180%**：讨论中提到 2024 年 **数据中心建设惊人地增长了 180%**，这可能标志着该行业的一个重要趋势。
  
  - 一些成员表示怀疑，认为这可能仅仅意味着 **投资浪费**，而非可持续的增长轨迹。
- **Nvidia 在 AI 领域的强势地位**：关于 Nvidia 市场份额的辩论反映了其历史上对游戏业务的依赖，现在已转型专注于 **AI 加速器**。
  
  - 一位成员断言，即使 **AMD** 的产品免费，企业仍会更倾向于 Nvidia，突显了品牌忠诚度。

 

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **E2B Desktop Sandbox 发布**：**E2B Desktop Sandbox** 目前已进入 Beta 测试阶段，为 LLM 应用量身定制了隔离环境，具有完整的文件系统支持和强大的可定制性。
  
  - 鼓励用户提供反馈，以完善平台并优化其在云环境中的实用性。
- **Claude 3.5 挑战隐私边界**：全新的 **Claude 3.5 Sonnet** 现在可以监控屏幕并控制设备，提供文件搜索和 Web 自动化等功能，这引发了重大的隐私担忧。
  
  - 这一进步标志着 AI 交互复杂性的实质性飞跃，引发了关于伦理使用的讨论。
- **Cerebras 芯片创下新的推理记录**：来自 **Cerebras** 的一款新芯片展示了在 **Llama3.1-70B** 上快出 **3 倍**的推理性能，达到了超过 **2,100 tokens/s**，比最快的 GPUs 快 **16 倍**。
  
  - 这一突破使 Cerebras 成为 AI 处理领域的重要参与者，为竞争对手设定了极高的基准。
- **关于 OpenAI Orion 的猜测引发热议**：OpenAI 暗示将在 12 月前发布名为 **Orion** 的模型，在有关其开发时间表的误导指控中引发了辩论。
  
  - 首席执行官 Sam Altman 对即将推出的技术的评论引发了对实际发布日程的猜测和混乱。
- **Cohere 的 Embed 3 增强了多模态能力**：**Cohere** 推出了其 **Embed 3** 模型，允许企业在文本和图像数据集上进行搜索，极大地提升了 AI 功能。
  
  - 这一进展旨在促进跨不同文档类型的实时数据处理，从而提高效率。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **播客定制化增强了连贯性**：用户发现，通过特定的提示词（如分配姓名和角色），可以实现 **AI 生成播客的连贯性**，使主持人的介绍在各集中保持一致。
  
  - *角色限制变得明显*，因为男声通常扮演主持人，而女声则扮演专家，这使得角色分配的灵活性变得复杂。
- **Deepfake 技术引发伦理担忧**：关于 **Deepfake 技术** 的讨论提出了围绕知情同意的伦理问题，强调了公众理解在避免潜在滥用方面发挥的关键作用。
  
  - 成员们担心 AI 中的**去人性化**问题，质疑是否可以合乎伦理地制作虚拟形象，并认为责任应由内容创作者承担。
- **AI 问答游戏正在开发中**：用户试用了一种利用 AI 进行动态问题交换的问答游戏格式，初步取得了成功，但在准确计算分数方面表现不佳。
  
  - 统计响应时的差异突显了持续存在的挑战，特别是 AI 在**数学准确性**方面的缺陷。
- **AI 辅助角色开发**：利用 AI，成员们正在检查剧本草稿中的角色缺失和发展情况，通过“围读（table reads）”改进故事情节。
  
  - 这种方法通过与 AI 更具参与性的互动，产生了更深刻的叙事见解和潜在的背景故事构思。
- **AI 性能限制暴露了弱点**：参与者承认 AI 存在幻觉（hallucinate）倾向，特别是在计数和事实交付方面，这极大地影响了整体准确性。
  
  - 讨论包括利用 Python 等额外工具来克服 AI 在计算方面的这些缺陷。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **用户渴望 LM Studio 插件功能**：人们对 **LM Studio** 中用户创建插件的潜力越来越感兴趣，这可以在不增加复杂性的情况下增强功能。
  
  - 与现有工具更好的集成以及开放的 API endpoints 可以显著提升用户体验。
- **Mamba-Codestral 模型加载失败**：一位用户报告了加载 **Mamba-Codestral** 模型时的问题，暗示 GPU 错误和驱动程序冲突是主要原因。
  
  - 建议的修复方案包括清理 shader caches（着色器缓存）和修改 GPU offload 百分比，以解决 VRAM 限制问题。
- **大语言模型性能评测**：用户讨论了使用大型 LLM 的经验，指出更大的模型规模可以增强上下文长度，但也会提高硬件需求。
  
  - 性能优化仍然是一个关注点，因为较大的模型可能会由于资源紧张而减慢响应速度。
- **Intel Arc A750 速度令人惊喜**：在升级到 **Intel Arc A750** 后，一位用户发现其在 **LM Studio** 中的表现令人印象深刻，超过了他们之前的 **6750xt** 配置。
  
  - 这引发了关于现代 GPU 能力的讨论，特别是在机器学习场景下。
- **Gemma 2 Token 速率及担忧**：**Gemma 2 2B** 模型达到了 **25 tokens/s**，而 **Gemma 2 9B** 则落后于大约 **6 tokens/s**，引发了对输出错误的警示。
  
  - 这些 Token 速度凸显了可能阻碍模型可用性的问题，需要进一步调查。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **DeepSeek 提供快速性能**：在使用 **DeepSeek** 作为编辑器模型（editor-model）时，用户注意到处理过程中没有明显的延迟，这引发了对该工具效率的热烈讨论。
  
  - 这一积极反馈表明，采用 **DeepSeek** 有潜力实现更流畅的代码编写体验。
- **Aider v0.60.1 新特性**：即将发布的 **Aider v0.60.1** 包括对 **Claude 3** 模型、文件排序的支持，以及一个新的 `--fancy-input` 标志以增强命令处理。
  
  - 关于安装延迟的猜测也随之出现，暗示了一些用户遇到的本地问题。
- **Prompt caching 节省成本**：用户探索了 **Aider** 中的 **prompt caching** 选项，发现它有利于提高性能并降低成本，特别是在使用 **Sonnet model** 时。
  
  - 据报道，启用缓存可以最大限度地减少与本地编码任务相关的费用，使其成为一种首选策略。
- **PearAI 集成 Aider**：围绕 **PearAI** 使用 **Aider** 实现编码功能的讨论展开，引发了关于权限和集成性质的疑问。
  
  - 针对 PearAI 内部可能对 Aider 功能进行的品牌重塑或修改，人们表达了担忧，详见 [PearAI Creator 文章](https://trypear.ai/blog/introducing-pearai-creator-beta)。
- **对 Claude 1022 行为的担忧**：用户报告了 **Claude 1022** 不可预测的输出，通常称其在与 **Cursor** 等工具配合使用时表现出“过度活跃”的行为。
  
  - 输出的不一致性引发了关于需要精炼用户 Prompt 以在交互过程中保持控制的讨论。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous Research 获得收入分成**：**Nous Research** 与 Hyperbolic 合作，分享其 **Hermes 3** 模型的收入，促进了一种协作融资方式。
  
  - 成员们讨论了这种伙伴关系是一种互利安排，并澄清这并非“卖身”行为。
- **AI 炒作周期减弱**：成员们注意到，与今年早些时候相比，AI 的炒作有所减少，可能被即将到来的美国大选等事件掩盖。
  
  - 讨论推测社区可能正处于“期望膨胀”阶段，而非真正的深度参与。
- **模型性能基准测试**：关于 **Llama 4** 模型与 **Claude** 性能对比的辩论非常激烈，人们对当前的 Benchmark（基准测试）方法持怀疑态度。
  
  - 一位成员指出 Llama 4 的性能超过 **120+ tps**，对对比的有效性提出了质疑。
- **探索量化技术**：成员们分析了 Meta 推出的量化模型，辩论了其可行性以及对模型训练的潜在好处。
  
  - 针对应用量化感知训练（quantization-aware training）相关的计算复杂度，人们提出了担忧。
- **Softmax 函数研究**：来自 [Google DeepMind](https://x.com/citizenhicks/status/1849223057757684157?s=46) 的一篇论文揭示，**softmax 函数**在输入增加时难以保持锐度，导致**注意力系数分散**。
  
  - 实验表明，虽然模型在熟悉任务中表现出色，但在更大规模、分布外（out-of-distribution）的情况下，其注意力会减弱。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **NEO 测试显示改进**：**NEO** 模型的本地测试显示，随着重复交互，它变得**更快**且**更聪明**，引发了人们对训练数据集 **Pile** 的兴趣。
  
  - 评论者注意到这些测试中交互的参与性很强。
- **Munkres 被推荐用于拓扑学**：在寻求优秀的**拓扑学 (topology)** 书籍时，成员们迅速推荐了 **Munkres** 作为权威的学习资源。
  
  - 这本书在拓扑学学生中赢得了很高的声誉。
- **微调 Llama 3.2 模型**：一位成员寻求关于微调 **Llama 3.2** 模型以将文本分为 **20 个类别**的指导，特别是关于 **DPO** 的使用。
  
  - 建议包括采用简单的分类器，尽管成员们对数据集可能存在的性能问题表示担忧。
- **对 Classifier-Free Guidance 的质疑**：对 **Classifier-Free Guidance (CFG)** 的有效性产生了怀疑，指出其存在依赖于时间步长 (timestep) 和引导比例 (guidance scales) 的问题。
  
  - 对话中包含了一种潜在的简化方法，即直接从文本输入生成输出。
- **图像描述数据集的挑战**：针对数据集中描述 (caption) 质量差的问题提出了担忧，认为重新标注无法解决准确性和相关性问题。
  
  - 讨论了大规模生成高质量描述的挑战，强调了现有解决方案的局限性。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Opus 3.5 发布时间面临不确定性**：关于 Anthropic 的 **Opus 3.5** 是否会在今年发布的猜测不断，一些人认为可能会推迟到 **2025** 年。
  
  - *有人建议他们可能会直接跳到一个更新的版本。*
- **AGI 与 ANI 之争升温**：成员们就**弱人工智能 (ANI)** 与**通用人工智能 (AGI)** 展开了激烈讨论，评估这些术语对当前 AI 模型的定义和适用性。
  
  - 一些人提议使用 **Emerging AGI** 一词来描述通往开发通用智能的潜在路径。
- **未来 AI 训练方法的推测**：讨论集中在运行**数百万个 H100s** 规模的模型所需的资源上，引发了对下一代 **GPUs** 生产问题的担忧。
  
  - 实现这种规模化可能仍严重依赖现有硬件，一些人估计未来会有巨大的需求。
- **OpenAI 的数据中心雄心引发辩论**：最近的一份报告概述了 OpenAI 计划建设大规模 **5GW 数据中心**以训练先进 AI 模型的计划，引发了关于可行性和规模的对话。
  
  - 怀疑者担心这种广泛计算目标的**生态影响**和实用性。
- **更新后 Co-Pilot 图标失踪**：一位用户在更新后发现 Windows 系统中的 **Co-Pilot** 图标消失了，引发了对原因和可能修复方法的询问。
  
  - 回复从困惑到开玩笑不等，揭示了社区中普遍存在的用户体验问题。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Cerebras API 访问引发兴趣**：用户分享了使用 **Cerebras API** 的经验，指出获取访问权限的时间跨度很大，从一个多月前到无需正式接受即可获取密钥不等。
  
  - 讨论强调了芯片成本与 **API** 预期性能之间的平衡。
- **关于 Hermes-3 审查的推测**：对 **hermes-3-llama-3.1-405b** 潜在审查的担忧被提出，反映了社区对模型内容限制的忧虑。
  
  - 这指向了关于 AI 模型可接受内容阈值的更广泛对话。
- **探索 Prompt Caching 的益处**：讨论了 **OpenRouter** 上 **Sonnet** 模型的 **Prompt Caching** 可用性，用户强调了其优化 **API** 使用的能力。
  
  - 然而，一些人在与 **SillyTavern** 等外部应用程序对接时遇到了实现问题。
- **Token 限制令用户沮丧**：一位用户在拥有 16 美元余额的情况下仍收到最大 **tokens** 限制错误，引发了关于创建新 **API** 密钥的讨论。
  
  - 社区共识倾向于将检查账户余额状态作为故障排除的一部分。
- **OpenRouter 的性能担忧**：用户报告遇到运行缓慢和错误 **520**，对系统可靠性和性能问题发出了警报。
  
  - 讨论指出硬件供应挑战正在影响先进模型的性能。

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Flux 面临漫画创作挑战**：成员们讨论了使用 **FLUX** 进行漫画生成，强调了对特定角色模型进行微调以增强**一致性（consistency）**和**提示词忠实度（prompt fidelity）**的需求。
  
  - *使用标准模型很难达到理想的细节水平*，因此需要针对特定角色的一致性进行进一步训练。
- **Mochi 在视频生成方面表现优异**：用户将 **Mochi 1** 与 **CogVideoX** 在本地视频创作方面进行了对比，结论是虽然 Mochi 更胜一筹，但处理时间较慢。
  
  - 用户推荐使用 **CogVideoX**，因为它功能丰富，尽管在某些场景下效果不如 Mochi。
- **对 Stable Diffusion 3.5 的质疑**：有人对 **Stable Diffusion 3.5** 生成特定提示词（如“一名女性躺在棉花糖池上”）的能力提出了疑问。
  
  - 一位用户指出，使用该提示词创建的图像已出现在另一个频道中供社区反馈。
- **为 House 音乐创作艺术作品**：一位成员正在寻找在 **SoundCloud** 上为 House 轨道设计封面艺术作品的技巧，并分享了对作品风格的具体期望。
  
  - 对初步结果的失望显现出来，表明掌握 AI 驱动的艺术生成存在学习曲线。
- **LoRA 训练依赖优质数据集**：随后讨论了高质量数据集对于 **LoRA** 模型训练的重要性，以确保可靠的输出。
  
  - 参与者建议，关于数据集准备的教程可以在模型微调之前大大提高用户的熟练程度。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro 引发用户辩论**：用户分享了使用 **Perplexity Pro** 的不同经验，辩论其相对于 **Claude** 和 **GPT** 等竞争对手的**价值**。他们寻求有效的设置和资源来优化其订阅。
  
  - *出现了对性能与价值之间关系的担忧*，促使进一步探索最佳使用案例。
- **Gemini 2.0 发布在即**：随着 **Google** 和 **OpenAI** 竞相推出下一代模型，**Gemini 2.0** 预计很快发布，同时人们也对预期的**性能提升**提出了疑问。12 月将成为 AI 发展的重大月份。
  
  - 参与者注意到 AI 能力的迅速进步，但指出*不同平台之间的改进仍然是碎片化的*。
- **关于 Perplexity App 功能的咨询**：用户对 **Perplexity App** 的推理能力及其对 iOS 语音识别的需求表现出浓厚兴趣。讨论强调了管理指令设置以减少 AI **幻觉（hallucinations）**的重要性。
  
  - 用户表达了对确保 App 在更关键的工作流中提供可靠输出的担忧。
- **法律行业利用 AI**：针对 **AI 在法律研究中的作用**，用户表达了挫败感，强调尽管有细致的提示词指令，但仍难以产生可靠的输出。讨论中强调了可靠信息来源的必要性。
  
  - 用户交流了优化提示词的技术，旨在提升 AI 在法律场景中的表现。
- **比特币创始人身份之谜揭晓**：关于**比特币创始人身份**的一个令人震惊的启示已经出现，引发了加密社区的讨论。调查结果可以在这个 [YouTube 视频](https://www.youtube.com/embed/sRUpGVJfNJ4)中查看。
  
  - 这一突破可能会重塑区块链话语中关于比特币起源的对话。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **探索 AI 在兽医学中的应用**：一位成员询问了 **AI 在兽医学中具有前景的应用**，引发了对创新用途的兴趣。
  
  - 这引发了一场没有具体参考文献的开放论坛讨论，突显了该领域尚未开发的潜力。
- **Triton 优化显示出性能挑战**：将 kernel 封装在 `custom_op` 中导致性能从 **23 tokens/sec** 下降到 **16 tokens/sec**，引发了对封装机制的担忧。
  
  - 成员们正在质疑这种方法对 Triton 的开销影响，并考虑进一步的优化。
- **Llama 3.2 模型现已开源**：Meta 发布了 **Llama 3.2 1B 和 3B** 模型，旨在用于 **on-device** 部署，并通过量化技术提高性能。
  
  - 开发者旨在优化内存，同时确保模型在低资源场景下保持其有效性。
- **NanoGPT 的训练增强**：讨论强调 **NanoGPT** 可以通过优化的 **Triton** 操作获得加速，特别是如果仅使用 eager 模式的 **PyTorch**。
  
  - 社区强调了结合 **torch.compile** 以增强模型训练期间性能的重要性。
- **Discord Cluster Manager 开发开始**：**Discord Cluster Manager** 的文档已共享，概述了项目功能和未来的开发需求。
  
  - 计划于 **11 月 3 日** 开始积极开发，目标是在 **11 月 10 日** 之前完成，并邀请社区贡献。

 

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **常规问题引导方向已明确**：提醒成员将有关组织的问题引导至正确的频道 [此处](https://discord.com/channels/1087530497313357884/1098713601386233997)，以获得结构化的支持。
  
  - 此次重组旨在简化查询，确保成员能够有效地找到答案。
- **Kitty Ket 在 LED 矩阵项目上的突破**：Kitty Ket 报告了 **LED 矩阵项目** 的进展，利用 **3D vectors** 和 **数据处理函数** 实现了尖端性能。
  
  - 处理时间的目标是低于 **10 ms**，尽管尚未与 LED 矩阵进行通信，但已展示出令人期待的结果。
- **将 PostgreSQL 与 Mojo 集成**：一位成员提出了关于将 PostgreSQL 的 *libpq.so* 集成到 Mojo 中的问题，特别是关于针对自定义库的 `ffi.external_call`。
  
  - Darkmatter 阐明了 C 语言中 `char*` 的转换，在 x86_64 架构下通常转换为 `Int8`，在 ARM 架构下转换为 `UInt8`，这表明集成过程中需要明确性。
- **关于 Mojo 内存管理的新 Bug 报告**：最近的一份 [bug report](https://github.com/modularml/mojo/issues/3710) 指明，**Mojo** 在引用仍处于活动状态时过早地释放了内存。
  
  - 由于立即释放，用户无法保留 **List** 的地址，这给内存管理带来了持续的挑战。
- **序列化模型摄取用例探索**：成员们讨论了通过 **Graph API** 摄取序列化模型的潜在用例，寻求社区见解。
  
  - 此次参与旨在使模型摄取开发与现实世界的用户需求和应用场景保持一致。

 

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Deterministic GPU Kernels for Metal**: 一位成员询问了如何针对 Metal 创建 **deterministic GPU kernels**，以在 M2 和 M3 等 GPU 上实现一致的输出。另一位成员指出，如果成功，可能值得 fork tinygrad。
  
  - 这一努力与提高 GPU 计算一致性和可靠性的更广泛目标相一致。
- **Floating Point Arithmetic Consistency Challenges**: 关于 MLX 中 **floating-point arithmetic**（浮点运算）不一致性的担忧浮现，引发了关于 tinygrad 实现确定性能力的讨论。用户辩论了这些不一致性对模型可靠性的影响。
  
  - 浮点运算的非结合性质可能会给在各种环境中实现一致输出带来挑战。
- **Tinygrad's Metal Configurations Revealed**: **Tinygrad** 默认禁用 **Metal's fast math mode**，以减轻浮点运算中的差异，这引发了关于其对性能影响的讨论。向 **mathMode** 选项的过渡暗示了提高确定性的潜在路径。
  
  - 成员们承认在进行面向 GPU 的项目时，理解这些配置的重要性。
- **Beam Search in Kernel Space Impresses**: 用户对 **kernel space 中的 beam search** 表现出极大热情，注意到其速度令人印象深刻，尽管目前还无法与 **flash attention** 媲美。这突显了 tinygrad 持续的优化能力。
  
  - 讨论强调了 Kernel 级优化在加速搜索算法方面的有效性。
- **Handling Environment Variables in Notebooks**: 一位用户在 Notebook 中为 **Fashion MNIST** 数据集设置 **environment variables** 时遇到挑战，导致对必要配置产生困惑。George Hotz 澄清了 **os.environ** 的正确用法。
  
  - 这一澄清有助于简化工作流，强调了在记录框架中正确处理环境的重要性。

 

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Build Knowledge-Backed Agents with LlamaIndex**: 在一次 AI Agents 大师课中，创始人详细介绍了如何使用 LlamaIndex 工作流创建 **knowledge-backed agent**，强调了 **LLM router** 和其他核心工具，点击[此处](https://twitter.com/llama_index/status/1849577709393056081)查看该课程。
  
  - 该课程比较了**基于事件和基于图的架构**，大家一致认为 **LLM routers** 因其卓越的性能而更具优势。
- **NVIDIA's Internal AI Assistant Deployment**: **NVIDIA** 宣布其内部 AI 助手利用 **Llama 3.1 405b** 处理简单查询，并使用 **70b model 进行文档搜索**，详情见[此处](https://twitter.com/llama_index/status/1849847301680005583)。
  
  - 该助手集成了多个信息源，包括**内部文档**和 **NVIDIA 网站**，简化了对关键数据的访问。
- **Challenges Selling RAG in Production**: 成员们对向利益相关者证明 **RAG (Retrieval-Augmented Generation)** 在生产环境中的价值之难感到沮丧。
  
  - *“让人们相信这一点太难了”* 捕捉到了在推广 RAG 落地过程中持续存在的挣扎。
- **Strategies for Document Updates**: 管理频繁的文档更新带来了挑战，引发了关于利用向量数据库进行自动化的讨论。
  
  - 建议包括利用 **Qdrant** 进行索引，并安排 **cron jobs** 以促进及时的更新。
- **LlamaDeploy & LlamaIndex Compatibility Confirmed**: 成员们确认 **LlamaDeploy** 与最新版本的 **LlamaIndex Workflow** 兼容，确保了无缝的版本同步。
  
  - 他们指出，由于其异步设计，在 **LlamaDeploy** 中部署多个工作流可以有效地管理并发请求。

 

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 社区非常连贯（Coherent）**：成员们赞扬了 **Cohere 社区**的高质量讨论，将其与缺乏清晰度的其他 AI 社区进行了对比。
  
  - *一位成员在这个充满活力的环境中寻求合作机会*。
- **对 Cohere 研究创新的兴奋**：社区对 **Cohere research** 的近期进展议论纷纷，用户报告了实质性的进步。
  
  - *开发成果正在迅速推出*，标志着团队的一个重要里程碑。
- **理解歌曲 Embedding 功能**：针对 [Song Embedding Notebook](https://github.com/jalammar/jalammar.github.io/blob/master/notebooks/nlp/02_Song_Embeddings.ipynb) 提出了询问，特别是关于如何使用歌曲 ID 计算推荐。
  
  - 成员们讨论了开发这些 Embedding 时选择的方法是 **sentence2vector** 还是 **word2vec**。
- **深入探讨 Aya 与 Command 模型**：讨论明确了 **Aya** 针对多语言任务进行了优化，而 **Command** 则专注于生产环境。
  
  - *成员们注意到 Aya 在多语言能力方面表现尤为出色*，引发了富有成效的讨论。
- **修复奇怪的 JSON 参数 Bug**：一位成员提出了关于函数调用中 JSON 格式错误的担忧，强调了单引号与双引号的问题。
  
  - *对这个奇怪 Bug 的挫败感不断增加*，另一位成员通过示例强调了正确进行 JSON 转义（escaping）的重要性。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter 修复补丁发布**：`interpreter --os` 的最新更新现已在 pip 上提供，邀请用户在启动 **voice mode** 之前测试问题。
  
  - 这些更新旨在提升面临解释器挑战的用户的体验。
- **对 Claude 速率限制（Rate limits）的沮丧**：成员们报告称感到受到了 **Claude** 速率限制的阻碍，这导致了工作流中断。
  
  - 一位成员幽默地指出，速率限制确实在考验他们的耐心。
- **设置自定义 OpenAI API Agent**：关于配置自定义 **OpenAI API** Agent 而非使用 **Claude** 的可行性正在进行讨论。
  
  - 已分享协助用户设置配置的文档以提供实践指导。
- **Clevrr-Computer 赋能 AI 生产力**：[Clevrr-Computer](https://github.com/Clevrr-AI/Clevrr-Computer?ref=producthunt) 提供了 **Anthropic** 的 **Computer** 的开源实现，用于通过 AI Agent 执行基础任务。
  
  - 该项目因其在跨平台自动化任务和提高生产力方面的潜力而受到赞誉。
- **探索 Chrome 内置 AI 功能**：指向 [Chrome's Built-in AI](https://developer.chrome.com/docs/ai/built-in) 资源的链接展示了 Web 活动中 AI 的强大集成。
  
  - 这些功能有望通过直接嵌入浏览器的复杂 AI 工具来改善用户交互。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **观察到视频模型训练瓶颈**：用户报告称，在 **8 个 GPU** 上训练视频分类模型时出现了严重延迟，主要是由于 MP4 文件中 **7M frames** 的数据加载（dataloading）瓶颈。
  
  - 将这些文件转换为 JPEG 会将数据集大小大幅扩展到 **1TB**，从而加剧性能问题。
- **分享 DataLoader 优化技巧**：社区建议强调了通过对比数据获取时间与 GPU 处理时间来监控 **DataLoader** 性能的重要性。
  
  - 实施有效的预取（prefetching）策略对于跟上更快的 GPU 速度、减少瓶颈至关重要。
- **影响训练速度的磁盘 IO 讨论**：关于 SSD 或 HDD 配置是否会导致训练期间显著的**读取速度**或 **IOPS 瓶颈**的担忧。
  
  - 监控磁盘 IO 对于诊断影响 **DataLoader** 性能和整体训练效率的潜在问题至关重要。
- **模型大小对训练效率的重要性**：用户讨论了使用 **50M parameter** 模型在处理较大 Batch Size 时导致延迟的问题，这表明处理视频数据的容量不足。
  
  - 有建议认为增加模型大小可以缓解数据加载问题，从而提升整体性能。
- **关于 LLM 应用最佳实践的新网络研讨会**：由 Meta 的高级 ML 工程师主讲的名为 *Best Practices for Building Successful LLM Applications* 的热门 [YouTube 网络研讨会](https://youtu.be/JVNwAn7bTqY?si=ifDjr3MhKUXlLnTN)在第一天就获得了近 **1000 次观看**。
  
  - 该会议承诺提供针对有效且有影响力的应用量身定制的 **LLM** 实施宝贵见解，鼓励动手学习。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **DPO 评估变得简单**：你可以使用 Axolotl 代码库，通过 `load_prepare_dpo_datasets` 函数加载数据集并将预测结果与 Ground Truth 进行比较，从而对 **Direct Preference Optimization (DPO)** 进行评估。
  
  - *效率与准确性并存*；在生成预测之前，请确保你的 DPO 模型通过 `model.eval()` 运行在评估模式下。
- **生成高效预测**：利用 torch 的 no_grad 上下文从评估数据集中生成预测，通过不跟踪梯度来优化内存使用。
  
  - 这种方法实现了*节省内存的预测*，确保了评估过程的顺畅和高效。
- **轻松计算指标**：生成预测后，使用 scikit-learn 计算**各种指标**（如准确率或 F1 分数），特别是通过 `accuracy_score` 等函数。
  
  - 这实现了预测标签与真实标签之间的精确比较，增强了评估的完整性。
- **集成 Callback 以简化训练流程**：使用 `BenchEvalCallback` 等 Callback 将评估集成到训练中，允许按预定义的时间间隔进行评估。
  
  - 这种指标的平滑整合有助于维持*高效的训练常规*，确保对模型性能的持续监控。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **关于 Mid Training 内容的投票引发讨论**：成员们发起了关于 **Mid Training** 的讨论，询问其中具体包含的内容，从而界定了所涉及的范围和流程。
  
  - *除了 RLHF 之外的所有针对某些数据的专门训练*，这一观点引导了对方法论的深入探索。
- **Epoch 细节：针对编程的训练**：对话中有一个建议，即 Mid Training 可能涉及专门针对编程进行 **1-2 个 Epoch** 的训练，澄清了训练方法论中的区别。
  
  - 这旨在增强对 **Epoch 训练** 如何影响 AI 性能的理解。
- **讨论历史邮件的多样性**：一位成员指出应该在历史邮件中注入多样性，表明了对数据变异及其影响的兴趣。
  
  - 这引发了关于**历史数据集**如何影响当前 AI 模型的思考。
- **Meme 在 AI 领域掀起波澜**：一位成员链接到一条推文，可能强调了 AI 社区内的文化评论。
  
  - 虽然没有提供具体细节，但 *Meme* 通常作为技术讨论的一个独特视角。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **评估来自 PDF 文件的数据集**：一位成员询问了评估和管理数据集的方法，特别是针对 **PDF 数据**，因为他们打算使用 PDF 文件运行评估。
  
  - 这对非结构化格式的结构化评估方法提出了挑战，引发了对潜在方法的讨论。
- **为 AI 高手提供的职位机会**：一位成员正在为即将到来的项目积极寻找**资深的 AI 开发者**，强调了对优秀人才的需求。
  
  - 这种互动引发了关于可能利用此类专业知识的项目构思的提问，营造了一个集思广益的环境。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **提交邮件的时间戳澄清**：一位成员指出**表单邮件的时间戳**为 **PST 时间 9 月 28 日下午 6:50**，为邮件提交背景提供了澄清。
  
  - 这一细节出现在解决邮件提交的具体问题时，强调了时间戳准确性的重要性。
- **邮件混淆问题的进展**：另一位成员确认他们找到了邮件，并对**未来的解决**表示乐观。
  
  - 他们的积极态度表明，围绕邮件问题的困惑即将得到解决。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **MIPROv2 增强 Prompt 生成**：一位成员分享了一个关于使用 **MIPROv2 优化器**技术和 **GSM8K 数据集**进行“自动 Prompt 生成”的[简短推文串](https://x.com/karthikkalyan90/status/1849902254373077196)。
  
  - 该实现包括三个模块，分别用于 Demo 生成、指令创建和最终 Prompt 编译，以简化流程。
- **用于结构化 Prompt 创建的三个模块**：该程序由用于 Demo 的 **Module 1**、用于指令的 **Module 2** 和用于合成最终 Prompt 的 **Module 3** 组成。
  
  - 这种模块化方法专注于 Prompt 生成的效率，利用系统化结构来提高整体效果。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Edgar 的资源检查**：Edgar 对 c123ian 分享的有关 **LLM Finetuning** 的有用资源表示感谢，并计划进行审阅。
  
  - 虽然没有详细说明资源的具体内容，但这次交流凸显了频道内讨论的协作性质。
- **LLM 技术协作**：成员们就 **Finetuning LLMs** 的不同技术和方法论进行了讨论，展示了多样化的专业知识。
  
  - 贡献者强调了分享可操作资源以提高模型性能的必要性。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune GitHub 获得新 Issue**：[Torchtune GitHub](https://github.com/pytorch/torchtune/issues/1901) 上报告了一个涉及各种增强和修复的新 Issue，强调了社区贡献的必要性。
  
  - 鼓励成员参与解决这些增强功能，尽管该 Issue 尚未特别标记为需要社区帮助。
- **呼吁在 Torchtune 上进行协作**：随着成员表示希望就最近关于增强和修复的 Issue 进行协作，对 **Torchtune** 的兴趣正在增长。
  
  - 正在进行的讨论集中在社区如何支持该项目，营造了积极的协作氛围。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **AI 创作者争取报酬权**：互联网上的创作者正面临一场危机，他们的作品在未经许可或未获得报酬的情况下被用于驱动 AI 系统，这凸显了建立有效许可平台的必要性。
  
  - 这一新兴系统旨在让个人能够为其内容授权以进行 AI 训练，有望提高内容创作者的公平性。
- **Human Native AI 推出数据市场**：联合创始人 James Smith 宣布 [Human Native AI](https://www.humannative.ai/) 正在开发一个数据市场，创作者可以在其中汇集他们的作品，并为 AI 训练获得公平的报酬。
  
  - 该倡议旨在解决数据使用中的不平等问题，并为担心作品被剥削的内容创作者提供保障。
- **Mozilla 的 Data Futures Lab 演讲系列活动**：由 James Smith 主讲的讲座是 **Mozilla Data Futures Lab 演讲系列**的一部分，旨在讨论 AI 领域中公平的数据生态系统。
  
  - 鼓励参与者预约参加此次活动，以参与关于数据和生成式 AI 未来的关键讨论。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Gorilla LLM Function Calling 见解**：在关于 [Berkeley Function Calling](https://discord.com/channels/1111172801899012102/1111353033352294440/) 的讨论中，提出了关于 **Gorilla LLM** 及其 **Function Calling** 能力的简明观点，表明其有显著改进。
  
  - *Good catch* 强调了团队正在敏锐地分析最新更新的细微差别，这可能会带来增强的模型交互。
- **潜在增强功能的讨论**：工程师们注意到 LLM 的 **Functionality**（功能性）在不断演进，改进 **Function Calls** 已成为未来版本的重点。
  
  - 这可能会带来进一步的优化，参与者们渴望看到这些讨论的实际成果。

---

**Alignment Lab AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

# PART 2: 频道详细摘要与链接

{% if medium == 'web' %}

### **HuggingFace ▷ #**[**general**](https://discord.com/channels/879548962464493619/879548962464493622/1299085549403115605) (638 messages🔥🔥🔥):

> - `AI Infrastructure`
> - `Usage of Hugging Face and Other Models`
> - `Quantum Computing in AI`
> - `Data Privacy and Ethics`
> - `Video Generation Technology`

- **运行高速 AI 模型**：关于使用 H200 服务器运行大型模型的讨论，一位用户提到他们的生产服务器能以 90 teraflops 的速度处理 405B 模型。
  
  - 用户们反思了此类基础设施的必要性以及是否可能过度配置，并对通用 AI 应用的成本和实用性表示担忧。
- **API 和推理服务**：King.of.kings_ 分享了他们的服务提供 Llama 模型的推理，强调了对快速响应时间的需求。

- 对话涉及优化模型性能与在高性能硬件上投入的权衡。
- **数据使用的伦理与法律**：Hudsong0 讨论了他们收集教育数据的方法以及绕过付费墙的伦理影响。
  - 尽管他们认为自己的使用方式可能符合服务条款，但小组建议对潜在的违规行为保持谨慎。
- **生成式 AI 及其未来应用**：参与者对 AI 生成内容（尤其是视频生成）即将取得的进展表示关注。
  - 随着 AI 视频技术的预期激增，大家讨论了如何利用这一能力进行创意和商业活动。
- **数据管理与处理方法**：Technosourceressextraordinaire 分享了他们使用各种工具将数据集处理成可管理的批次以进行质量控制的方法。
  - 参与者探讨了改进数据处理实践的策略以及与模型训练相关的潜在权衡。

**提到的链接**：

- [Hackers Have Uploaded Thousands Of Malicious Files To AI’s Biggest Online Repository](https://www.forbes.com/sites/iainmartin/2024/10/22/hackers-have-uploaded-thousands-of-malicious-models-to-ais-biggest-online-repository/): 黑客已向 Hugging Face 上传了数千个恶意模型
- [Sana](https://nvlabs.github.io/Sana/): 未找到描述
- [\*Tips Fedora\* | Know Your Meme](https://knowyourmeme.com/memes/tips-fedora#fn15): 未找到描述
- [RareConcepts/FurkinsWorld-SD35-LoKr · Hugging Face](https://huggingface.co/bghira/Furkan-SD3): 未找到描述
- [Time Travel Vanish GIF - Time Travel Vanish Disappear - Discover & Share GIFs](https://tenor.com/view/time-travel-vanish-disappear-on-my-way-im-coming-gif-16482139): 点击查看 GIF
- [Stop Dont Do That GIF - Stop Dont Do That Paparazzi - Discover & Share GIFs](https://tenor.com/view/stop-dont-do-that-paparazzi-blocked-cover-gif-14388166): 点击查看 GIF
- [I Saw W Gus Fring GIF - I Saw W Gus Fring Gus - Discover & Share GIFs](https://tenor.com/view/i-saw-w-gus-fring-gus-gustavo-deleted-gif-25440636): 点击查看 GIF
- [Or Yehuda Edgy GIF - Or Yehuda Edgy Or - Discover & Share GIFs](https://tenor.com/view/or-yehuda-edgy-or-yehuda-edgelord-gif-25925736): 点击查看 GIF
- [Yugioh Anime GIF - Yugioh Anime Omg - Discover & Share GIFs](https://tenor.com/view/yugioh-anime-omg-wtf-cant-unsee-gif-5159766): 点击查看 GIF
- [Laughing Emoji Laughing GIF - Laughing Emoji Laughing Emoji - Discover & Share GIFs](https://tenor.com/view/laughing-emoji-laughing-emoji-animated-laugh-gif-27394849): 点击查看 GIF
- [Forbes Marketplace: The Parasite SEO Company Trying to Devour Its Host](https://larslofgren.com/forbes-marketplace/): 你是否厌倦了 Forbes 出现在搜索结果中？对于 Forbes 并不擅长的领域？这是“最佳宠物保险”的有机排名：Forbes 排名第 2。不确定一家商业网站...
- [Fedora Tipshat GIF - Fedora Tipshat Mlady - Discover & Share GIFs](https://tenor.com/view/fedora-tipshat-mlady-melady-athiest-gif-7191305): 点击查看 GIF
- [Drugs Bye GIF - Drugs Bye Felicia - Discover & Share GIFs](https://tenor.com/view/drugs-bye-felicia-police-call-gif-12264463): 点击查看 GIF
- [Laugh GIF - Laugh - Discover & Share GIFs](https://tenor.com/view/laugh-gif-18097496): 点击查看 GIF
- [Gargoyle Better To Ask Forgiveness GIF - Gargoyle Better to ask forgiveness Disney - Discover & Share GIFs](https://tenor.com/view/gargoyle-better-to-ask-forgiveness-disney-gif-6017339847784900522): 点击查看 GIF
- [What Do You Mean Eric Cartman GIF - What Do You Mean Eric Cartman South Park - Discover & Share GIFs](https://tenor.com/view/what-do-you-mean-eric-cartman-south-park-s10e8-make-love-not-warcraft-gif-20869650): 点击查看 GIF
- [Gifmiah GIF - Gifmiah - Discover & Share GIFs](https://tenor.com/view/gifmiah-gif-19835013): 点击查看 GIF
- [Dave What Do You Think Youre Doing GIF - Dave What Do You Think Youre Doing Overly Attached Girlfriend - Discover & Share GIFs](https://tenor.com/view/dave-what-do-you-think-youre-doing-overly-attached-girlfriend-meme-hal9000-gif-14709229): 点击查看 GIF
- [Stay Calm The Office GIF - Stay Calm The Office - Discover & Share GIFs](https://tenor.com/view/stay-calm-the-office-gif-10942304): 点击查看 GIF
- [Yugioh Should GIF - Yugioh Should Been - Discover & Share GIFs](https://tenor.com/view/yugioh-should-been-gif-23901254): 点击查看 GIF
- [koboldcpp/kcpp_adapters at concedo · LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp/tree/concedo/kcpp_adapters): 使用 KoboldAI UI 轻松运行 GGUF 模型。单个文件，零安装。 - LostRuins/koboldcpp

- [在 Raspberry Pi 上运行本地 LLM](https://www.youtube.com/watch?v=CeKPXZ_8hkI)：查看我们的博客文章，了解如何使用 picoLLM 在 Raspberry Pi 上无损地本地运行 LLM：http://picovoice.ai/blog/local-llm-on-rpi-with-no-compromises/
- [chat_templates/chat_templates/llama-3-instruct.jinja at main · chujiezheng/chat_templates](https://github.com/chujiezheng/chat_templates/blob/main/chat_templates/llama-3-instruct.jinja)：🤗 HuggingFace 大语言模型的聊天模板 - chujiezheng/chat_templates
- [GitHub - p3nGu1nZz/ophrase: generate paraphrase using ollama and python](https://github.com/p3nGu1nZz/ophrase)：使用 ollama 和 python 生成改写（paraphrase）。通过在 GitHub 上创建账号来为 p3nGu1nZz/ophrase 的开发做出贡献。
- [Google Colab](https://colab.research.google.com/drive/1ekNDPjC3CKWWd3jd2_V9QGTJSbvHKIZ2?usp=drive_link)：未找到描述
- [GitHub - GrandaddyShmax/audiocraft_plus: Audiocraft is a library for audio processing and generation with deep learning. It features the state-of-the-art EnCodec audio compressor / tokenizer, along with MusicGen, a simple and controllable music generation LM with textual and melodic conditioning.](https://github.com/GrandaddyShmax/audiocraft_plus)：Audiocraft 是一个用于深度学习音频处理和生成的库。它包含了最先进的 EnCodec 音频压缩器/分词器（tokenizer），以及 MusicGen —— 一个简单且可控的、支持文本和旋律调节的音乐生成 LM。

---

### **HuggingFace ▷ #**[**today-im-learning**](https://discord.com/channels/879548962464493619/898619964095860757/1299096779593089047) (1 条消息):

> - `Transformers 基础`
> - `10M 参数模型`
> - `Reddit 帖子生成`
> - `DeepLLMs 仓库`
> - `学习模型的改进`

- **Transformers 入门**：一位成员刚刚通过观看 [Andrej 的视频](https://link.to/video) 开始学习 **transformers** 的基础知识。他们在 Reddit 帖子数据集上使用 **10M 参数的 transformer** 取得了显著成果。
- **生成 10k Tokens 的输出**：使用一个简单的 transformer 模型，他们成功地从 Reddit 数据中生成了令人印象深刻的 **10k tokens** 输出。
  
  - 这一成就激发了对进一步改进和实现以优化模型性能的兴趣。
- **探索 DeepLLMs 仓库**：用户分享了他们的 [DeepLLMs 仓库](https://github.com/its-nmt05/DeepLLMs/blob/main/model_architecture.ipynb) 链接，该仓库专注于学习 LLMs 和 transformers。
  
  - 该仓库包含了模型架构的详细信息，旨在探索该领域中有趣的应用。
- **寻求改进建议**：该成员表示希望获得关于其 transformer 模型进一步实现和改进的建议。
  
  - 这一反馈请求表明了对社区意见和协作增强的开放态度。

 

**提到的链接**：[DeepLLMs/model_architecture.ipynb at main · its-nmt05/DeepLLMs](https://github.com/its-nmt05/DeepLLMs/blob/main/model_architecture.ipynb)：旨在学习 LLMs 和 transformers 的基础知识，并在此过程中探索其他有趣的内容 - its-nmt05/DeepLLMs

 

---

### **HuggingFace ▷ #**[**cool-finds**](https://discord.com/channels/879548962464493619/897390579145637909/) (1 条消息):

elliotalder50n: [https://lmarena.ai/](https://lmarena.ai/) 你对这个排行榜有什么看法？

---

### **HuggingFace ▷ #**[**i-made-this**](https://discord.com/channels/879548962464493619/897390720388825149/1299249610149335122) (7 条消息):

> - `使用 Streamlit 的计算器项目`
> - `使用 HuggingFace 进行蛋白质和基因组学研究`
> - `自动驾驶中的 Self-Supervised Learning`
> - `AI RPG 冒险`
> - `Stable Diffusion 3.5 Large 画廊`

- **Streamlit 计算器初具规模**：一位成员使用 Streamlit 成功复现了 [计算器项目](https://github.com/dhruvyadav89300/iOS-Notes-Calculator) 的基础版本，并邀请大家对其作品提供反馈。
  
  - 他们表达了对项目的兴奋之情，并鼓励社区查看他们的 GitHub 链接。
- **在基因组学中探索蛋白质表型**：一位参与者分享了他们使用 HuggingFace 工具进行 [Project PhenoSeq](https://huggingface.co/seq-to-pheno) 的初步经验，重点关注蛋白质网络分析和表型结果。
  
  - 他们强调了一个与野生型蛋白质相关的潜在 [数据集](https://huggingface.co/datasets/seq-to-pheno/wildtype_proteins)，认为它是持续研究的宝贵资源。
- **关于自动驾驶 Self-Supervised Learning 的博客**：一位成员撰写了一篇关于 [Self-Supervised Learning](https://www.lightly.ai/post/self-supervised-learning-for-autonomous-driving) 的博客文章，讨论了其在单目深度估计等自动驾驶任务中日益增长的重要性。
  
  - 文章将 **regression-based methods** 与最新进展进行了对比，强调了从 supervised 到 self-supervised 技术的转变。
- **加入 AI RPG 冒险！**：分享了一个引人入胜的 [AI RPG](https://huggingface.co/spaces/Pixeltable/AI-RPG-Adventure) 概念验证，允许玩家在幻想设定中扮演各种角色。
  
  - 创作者邀请其他人基于此概念探索并开发移动或 Web 应用，促进社区内的创意。
- **展示 Stable Diffusion 3.5 Large 的画廊**：一位成员创建了两个画廊，展示 [SD3.5 Large](https://enragedantelope.github.io/Artists-SD35L/) 在解读艺术风格方面的能力，涵盖了 120 多位艺术家。
  
  - 第二个画廊专注于 [艺术风格](https://enragedantelope.github.io/Styles-SD35L/)，展示了由 SD3.5L 生成的 140 种不同风格。

**提及的链接**：

- [AI RPG Adventure - Pixeltable 的 Hugging Face Space](https://huggingface.co/spaces/Pixeltable/AI-RPG-Adventure)：未找到描述
- [SD3.5L 艺术家画廊](https://enragedantelope.github.io/Artists-SD35L/)：未找到描述
- [SD3.5L 风格测试画廊](https://enragedantelope.github.io/Styles-SD35L/)：未找到描述
- [Self-Supervised Learning for Autonomous Driving](https://www.lightly.ai/post/self-supervised-learning-for-autonomous-driving)：本文概述了自动驾驶任务中 Self-Supervised Learning 的最新进展，重点关注三个关键领域：单目深度估计、自我运动估计和...
- [GitHub - dhruvyadav89300/iOS-Notes-Calculator](https://github.com/dhruvyadav89300/iOS-Notes-Calculator)：通过在 GitHub 上创建账户，为 dhruvyadav89300/iOS-Notes-Calculator 的开发做出贡献。
- [seq-to-pheno (Seq-to-Pheno)](https://huggingface.co/seq-to-pheno)：未找到描述
- [seq-to-pheno/wildtype_proteins · Hugging Face 数据集](https://huggingface.co/datasets/seq-to-pheno/wildtype_proteins)：未找到描述

---

### **HuggingFace ▷ #**[**reading-group**](https://discord.com/channels/879548962464493619/1156269946427428974/1299489082241585224) (2 条消息):

> - `Automated Penetration Testing Benchmark`（自动化渗透测试基准）
> - `Ethical Hacking and Cybersecurity Threats`（道德黑客与网络安全威胁）
> - `Performance of LLM in Cybersecurity`（LLM 在网络安全中的表现）

- **引入自动化渗透测试基准**：一篇题为 [“Towards Automated Penetration Testing: Introducing LLM Benchmark, Analysis, and Improvements”](https://arxiv.org/abs/2410.17141) 的论文引入了一个专注于渗透测试中 LLM 表现的新颖基准，旨在解决目前缺乏全面评估工具的问题。
  
  - 该研究使用 **PentestGPT** 工具评估了 **GPT-4o** 和 **Llama 3.1-405B** 等模型，结果显示 Llama 3.1 在特定任务中的表现略优于 GPT-4o。
- **网络安全危机凸显道德黑客需求**：论文强调了黑客攻击构成的严重威胁，全球范围内已造成 **6 万亿美金** 的损失，并强调了通过道德黑客（Ethical Hacking）和渗透测试来识别漏洞的重要性。
  
  - [Hugging Face 上的](https://huggingface.co/blog/Isamu136/ai-pentest-benchmark) 一篇客座博客文章详细阐述了该论文的研究结果以及在网络安全领域建立稳健基准的必要性。
- **关于自动化渗透测试的未来讨论**：计划于下周举行一次会议，展示渗透测试论文的研究结果，旨在根据当前网络安全趋势讨论其影响。
  
  - 该会议旨在为围绕 LLM 在道德黑客中应用的讨论创建一个平台，强调 **自动化评估（automated assessments）** 日益增长的重要性。

**提到的链接**：

- [Towards Automated Penetration Testing: Introducing LLM Benchmark, Analysis, and Improvements](https://arxiv.org/abs/2410.17141)：黑客攻击对网络安全构成重大威胁，每年造成数十亿美元的损失。为了减轻这些风险，采用道德黑客或渗透测试来识别漏洞...
- [Towards Automated Penetration Testing: Introducing LLM Benchmark, Analysis, and Improvements](https://huggingface.co/blog/Isamu136/ai-pentest-benchmark)：未找到描述
- [Towards Automated Penetration Testing: Introducing LLM Benchmark, Analysis, and Improvements](https://isamu-website.medium.com/towards-automated-penetration-testing-introducing-llm-benchmark-analysis-and-improvements-c7d06a2bf963)：你好！这将是关于我们论文《迈向自动化渗透测试：引入 LLM 基准、分析与改进》的博客。为了……

---

### **HuggingFace ▷ #**[**computer-vision**](https://discord.com/channels/879548962464493619/922424143113232404/1299255967577407498) (1 条消息):

> - `Efficient Transformers for High Feature Channels`（适用于高特征通道的高效 Transformer）

- **寻求适用于高特征通道的高效 Transformer**：一名成员请求关于处理具有 **高特征通道数**（超过 **10** 个）输入的最 **高效 Transformer** 的参考文献和见解。
  
  - 重点在于探索能够适应此类高维数据输入的架构调整。
- **围绕提升效率的讨论**：成员们讨论了增强 Transformer 效率的各种策略，特别是针对 **高维特征输入**。
  
  - 他们分享了已测试的各种模型，强调了性能指标和可扩展性。

---

### **HuggingFace ▷ #**[**NLP**](https://discord.com/channels/879548962464493619/922424173916196955/1299239215921631232) (2 条消息):

> - `Uploading models from Google Colab`（从 Google Colab 上传模型）
> - `Using .push_to_hub`（使用 .push_to_hub）

- **咨询从 Google Colab 上传模型的问题**：一名成员询问是否可以直接从 **Google Colab** 将模型上传到 Hugging Face，还是需要先下载到本地存储。
  
  - 这反映了用户在不同平台之间进行模型管理时面临的常见困扰。
- **利用 .push_to_hub 进行模型上传**：另一名成员建议使用 **.push_to_hub** 方法作为上传模型的解决方案。
  
  - 该方法突出了一种直接从 notebook 将模型集成到 Hugging Face 的流线型方法。

---

### **HuggingFace ▷ #**[**diffusion-discussions**](https://discord.com/channels/879548962464493619/1009713274113245215/1299365706407870598) (5 messages):

> - `Contributing to Hugging Face Diffusers`
> - `Community Engagement`
> - `Understanding Noise Effects on Tensors`

- **开始使用 Hugging Face Diffusers**：一位成员表达了对贡献 **Hugging Face Diffusers** 项目的兴趣，并寻求有关入门最佳实践的指导。
  
  - 另一位成员建议阅读 [贡献指南 (contributing readme)](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md) 并搜索 [good first issue 标签](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)。
- **关于向 Tensor 添加噪声的讨论**：有人提问关于向 Tensor 添加噪声的影响，以及是否可以在不重新训练模型或 VAE 的情况下完成。
  
  - 有人寻求对“影响 (effects)”一词具体含义的澄清，表明对该话题存在一些困惑。
- **社区互动咨询**：一位成员询问聊天室中是否有其他人，可能是为了寻求更多互动。
  
  - 尽管询问了社区是否活跃，但随后没有产生实质性的回应。

 

**提到的链接**：<a href="[https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A"good+first+issue")">Issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)%22%3EIssues) · huggingface/diffusers: 🤗 Diffusers: 在 PyTorch 和 FLAX 中用于图像和音频生成的 SOTA 扩散模型。 - Issues · huggingface/diffusers

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**general**](https://discord.com/channels/1179035537009545276/1179035537529643040/1299087923806867580) (233 条消息🔥🔥):

> - `Unsloth AI Capabilities` (Unsloth AI 能力)
> - `Finetuning LLMs` (微调 LLMs)
> - `Dataset Preparation` (数据集准备)
> - `Pandas and cuDF for Data Handling` (用于数据处理的 Pandas 和 cuDF)
> - `AI Tool Selection` (AI 工具选择)

- **Unsloth AI 与视觉支持**：Unsloth 目前不支持像 Llama 3.2 这样的视觉模型，但支持未来功能的开发正在进行中。
  
  - 建议用户在等待视觉支持添加期间，先将该平台用于基于文本的 LLMs。
- **字幕纠错微调中的挑战**：一位用户尝试微调模型以纠正 VTT 字幕，但在模型修改时间戳方面遇到了困难。
  
  - 专家建议从训练数据中移除时间戳，以避免过拟合，并提高模型对文本纠错的关注度。
- **数据集质量的重要性**：微调 LLM 的有效性在很大程度上取决于所提供训练数据集的质量和平衡性。
  
  - 数据集不平衡会导致模型性能不佳，这强调了在训练前进行适当数据准备的必要性。
- **使用 cuDF 和 Pandas 进行数据处理**：在处理大型数据集时，用户更倾向于使用 cuDF，因为它比传统的 Pandas 方法性能更快。
  
  - cuDF 可以显著加速数据操作任务，使其成为数据科学工作流中备受青睐的工具。
- **AI 应用的工具选择**：建议仅在必要时使用 AI 工具，因为对于某些任务（如文本格式化），更简单的方法可能更高效。
  
  - 鼓励用户在 AI 确实能增加价值的地方探索其应用，而不是让简单的数据处理任务复杂化。

**提到的链接**：

- [Oh No GIF - Oh No Computer Saysno - Discover & Share GIFs](https://tenor.com/view/oh-no-computer-saysno-typing-gif-13727386)：点击查看 GIF
- [Welcome to the cuDF documentation! — cudf 24.10.00 documentation](https://docs.rapids.ai/api/cudf/stable/)：未找到描述
- [\==((====))== Unsloth - 2x faster free finetuning | Num GPUs = 1 \\\\ /| - Pastebin.com](https://pastebin.com/Le0CuH9X)：Pastebin.com 自 2002 年以来一直是排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。
- [Build stuck on torch2.5.0 · Issue #1295 · Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention/issues/1295)：我正在 Colab 上安装 flash-attention。在 torch2.4.1 上安装顺利。然而，现在 Colab 的 torch 版本升级到了 2.5.0，它卡在了 "Building wheels for col..."
- [Issues · unslothai/unsloth](https://github.com/unslothai/unsloth/issues/1192,)：以 2-5 倍的速度、减少 80% 的内存微调 Llama 3.2, Mistral, Phi & Gemma LLMs - Issues · unslothai/unsloth
- [Home](https://github.com/unslothai/unsloth/wiki#adding-new-tokens)：以 2-5 倍的速度、减少 80% 的内存微调 Llama 3.2, Mistral, Phi & Gemma LLMs - unslothai/unsloth
- [unsloth (Unsloth AI)](https://huggingface.co/unsloth)：未找到描述
- [GitHub - unslothai/unsloth at 9ca13b836f647e67d6e9ca8bb712403ffaadd607](https://github.com/unslothai/unsloth/blob/9ca13b836f647e67)：以 2-5 倍的速度、减少 80% 的内存微调 Llama 3.2, Mistral, Phi & Gemma LLMs - GitHub - unslothai/unsloth at 9ca13b836f647e67d6e9ca8bb712403ffaadd607
- [unsloth/unsloth/models/llama.py at 9ca13b836f647e67d6e9ca8bb712403ffaadd607 · unslothai/unsloth](https://github.com/unslothai/unsloth/blob/9ca13b836f647e67d6e9ca8bb712403ffaadd607/unsloth/models/llama.py#L2221))：以 2-5 倍的速度、减少 80% 的内存微调 Llama 3.2, Mistral, Phi & Gemma LLMs - unslothai/unsloth
- [Weights & Biases: The AI Developer Platform](https://wandb.ai)：Weights & Biases 是领先的 AI 开发者平台，用于训练和微调模型，管理从实验到生产的模型，并跟踪和评估由 LLMs 驱动的 GenAI 应用。

---

### **Unsloth AI (Daniel Han) ▷ #**[**off-topic**](https://discord.com/channels/1179035537009545276/1179039861576056922/1299097342628073544) (96 messages🔥🔥):

> - `AI 与生活方式的改变`
> - `数据中心建设趋势`
> - `Nvidia 的市场主导地位`
> - `ML 与机器人技术见解`
> - `FPGA vs. GPU 使用场景`

- **AI 正在重塑我们的生活**：一位成员预计，大多数人可能很快就会围绕 **LLM 输出**来规划生活，而不仅仅是将其作为 Google 的替代品，这预示着生活方式的深刻转变。
  
  - *探讨科幻主题*，他们将其与《Daemon》和《Freedom(TM)》等小说进行了类比。
- **数据中心建设增长 180%**：关于 **2024 年数据中心建设增长 180%** 所带来影响的讨论爆发了，这表明它可能代表了一个趋势的开始。
  
  - 其他人则表示怀疑，质疑其重要性，并认为这可能只是反映了**浪费的投资**。
- **Nvidia 不断演变的市场份额**：关于 Nvidia 当前市场定位的辩论强调了他们过去对游戏业务的依赖，并将其与近期对 **AI 加速器**的关注进行了对比。
  
  - 一位成员指出，即使 AMD 免费提供产品，**企业**仍会选择 Nvidia 而非竞争对手。
- **ML 与机器人领域的专业知识**：一位成员分享了他们追溯到 15 年前的丰富 **ML** 经验，强调了其在近期趋势之外的技术历史相关性。
  
  - 他们在**机器人**背景被认为在当今格局中至关重要，并对该领域的未来表示乐观。
- **FPGA 与 GPU 的讨论**：成员们商讨了 **FPGA vs. GPU** 的使用场景，认为由于 CUDA 的强势地位，Nvidia 在 AI 应用中占据了绝对主导。
  
  - 尽管对许可协议感到担忧，一些人表示愿意探索替代方案，同时也承认其中涉及的技术障碍。

 

**提到的链接**：[Reddit - Dive into anything](https://www.reddit.com/r/technology/comments/1gacxos/characterai_faces_lawsuit_after_teens_suicide/): 未找到描述

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**help**](https://discord.com/channels/1179035537009545276/1179777624986357780/1299098878485860383) (23 messages🔥):

> - `通过 DPO 实现简洁性`
> - `Llama 3.2 模型可用性`
> - `基于 MySQL Schema 的 SQL 查询生成`
> - `Gemma 模型推理过程中的错误`
> - `SFTTrainer compute_metrics 示例`

- **通过 DPO 优化回复的简洁性**：一位成员建议在 DPO 微调中对较短的回复给予正向**奖励**以增强简洁性，使用类似 `reward = 1/response_length` 的奖励函数。
  
  - 另一位用户考虑使用*被拒绝的冗长回复*来引导模型，使其明白选定的回复更加直接且简短。
- **Llama 3.2 Vision Instruct 11B 模型的可用性**：一位用户询问从何处获取 **Llama 3.2 Vision Instruct 11B 量化模型**，并直接获得了 Hugging Face 仓库的链接。
  
  - 成员们还强调了可用于访问 Llama 3.2 (3B) 模型的 **Google Colab notebook**。
- **基于 MySQL Schema 生成 SQL 查询**：一位用户寻求关于训练模型使用其 MySQL Schema 生成 **SQL 查询**的指导，重点是生成复杂查询并识别关系。
  
  - 他们询问进行一系列的**持续预训练 (continued pre-training)** 紧接着文本补全微调是否会产生更好的效果。
- **Gemma 模型推理时的问题**：一位用户报告在推理过程中使用 **Gemma 27B bnb** 模型时出现间歇性错误，而较小版本的模型没有问题。
  
  - 另一位用户提供了安装说明，以解决 Kaggle 环境中出现的问题。
- **SFTTrainer compute_metrics 使用示例**：一位成员请求在 **SFTTrainer** 中针对 LLM 分类任务使用 `compute_metrics` 的**实际示例**。
  
  - 这凸显了在模型训练期间对指标计算方面的实践见解的需求。

**提到的链接**：

- [unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit · Hugging Face](https://huggingface.co/unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit): 未找到描述
- [no title found](https://download.pytorch.org/whl/cu121): 未找到描述
- [GitHub - dottxt-ai/outlines: Structured Text Generation](https://github.com/dottxt-ai/outlines): 结构化文本生成。为 dottxt-ai/outlines 的开发做出贡献。

---

### **Unsloth AI (Daniel Han) ▷ #**[**research**](https://discord.com/channels/1179035537009545276/1257011997250424842/1299372917309706300) (5 messages):

> - `Unsloth 的 train_on_completions 方法`
> - `模型权重效率`

- **使用 train_on_completions 提升准确率**：**train_on_completions** 方法被用于仅针对助手输出进行训练，忽略用户输入。这种方法有助于模型更有效地集中其有限的权重，从而提高整体 **accuracy**（准确率）。
  
  - 一位成员评论道：*“如果没有这个，模型最终会将一些有限的权重浪费在预测用户输入中的 token 这一无用任务上”*，强调了该方法的重要性。
- **对话中提供了更多细节**：[一位成员分享了一个链接](https://discord.com/channels/1179035537009545276/1179035537529643040/1188496284789772378)，以获取有关该方法有效性讨论的更多信息。对话中的见解为该方法如何增强训练提供了额外的背景信息。
  
  - 一位成员发现这次讨论*非常有用*，表明了社区对知识共享的支持。

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1299092118672773130) (46 messages🔥):

> - `E2B Desktop Sandbox 发布`
> - `Claude 3.5 Sonnet 特性`
> - `Cerebras 推理芯片性能`
> - `OpenAI Orion 模型计划`
> - `Cohere 多模态 Embedding`

- **E2B Desktop Sandbox 首次亮相**：**E2B Desktop Sandbox** 现已进入测试阶段，提供为 LLM 使用优化的隔离安全环境，具有完整的文件系统和可定制性等特性。
  
  - 随着平台寻求增强云端应用的用户体验，欢迎各方提供反馈。
- **Claude 3.5 Sonnet：令人惊叹的新功能**：Anthropic 的 **Claude 3.5 Sonnet** 可以监控屏幕并控制设备，展示了先进的多步任务交互，这也引发了用户对隐私的担忧。
  
  - 它演示了搜索文件和自动化网页表单等功能，强调了 AI 功能的飞跃。
- **Cerebras 芯片大幅提升性能**：Cerebras 发布了一款新芯片，据报道其推理速度快了 **3 倍**，在 **Llama3.1-70B** 上以超过 **2,100 tokens/s** 的速度打破了记录。
  
  - 据称这一进步比最快的 GPU 解决方案快 **16 倍**，描绘了 AI 处理能力的重大飞跃。
- **关于 OpenAI Orion 模型的推测**：OpenAI 计划在 12 月前发布名为 **Orion** 的模型，但面临着有关其发布的误导信息和矛盾陈述的指责。
  
  - CEO Sam Altman 暗示了即将推出的技术，同时否认了具体的发布计划，增加了社区的困惑。
- **Cohere 推出多模态搜索**：**Cohere** 推出了其 **Embed 3** 模型，实现了跨文本和图像数据源的企业级搜索，增强了 AI 系统的能力。
  
  - 此次更新允许进行适用于各种文档类型的实时数据处理，旨在提高 AI 应用的效率。

**提到的链接**：

- [来自 cohere (@cohere) 的推文](https://x.com/cohere/status/1848760845641388087?s=46&t=PW8PiFwluc0tdmv2tOMdEg)：我们行业领先的 AI 搜索模型现在支持多模态了！Embed 3 使企业能够构建可以准确快速地搜索文本和图像数据源（如复杂的报告...）的系统。
- [来自 Sam Altman (@sama) 的推文](https://x.com/sama/status/1849661093083480123?s=46&t=tMWvmS3OL3Ssg0b9lKvp4Q)：@kyliebytes 假消息失控了
- [来自 Aidan McLau (@aidan_mclau) 的推文](https://x.com/aidan_mclau/status/1849605368768254430?s=46)：@teortaxesTex 不，我的时间线变长了；听起来基本上只有一个真正比 gpt-4 更大的在运行的模型。
- [Google 计划很快发布其下一个 Gemini 模型](https://www.theverge.com/2024/10/25/24279600/google-next-gemini-ai-model-openai-december)：12 月正成为 OpenAI 和 Google 展开 AI 发布对决的一个月。
- [来自 Alessio Fanelli (@FanaHOVA) 的推文](https://x.com/FanaHOVA/status/1849861104111059082)：你如何将一个写满“chicken”这个词的 PDF 变成一个病毒式传播的播客？@raiza_abubakar 和 @usamabinshafqat 来到 @latentspacepod 拆解了为什么 NotebookLM 效果这么好：- Crea...
- [来自 James Wang (@draecomino) 的推文](https://x.com/draecomino/status/1849479392525001210)：Cerebras 刚刚发布了一款新芯片——仅通过一次软件更新。https://x.com/CerebrasSystems/status/1849467759517896955 引用 Cerebras (@CerebrasSystems)：🚨 Cerebras 推理现在快了 3 倍...
- [来自 Boris Power (@BorisMPower) 的推文](https://x.com/borismpower/status/1849495150298530186?s=46)：@jachiam0 等着“十一月的惊喜”吧。

- [来自 Vasek Mlejnsky (@mlejva) 的推文](https://x.com/mlejva/status/1849532254072300028)：今天，我们要发布另一项产品：由 @e2b_dev 开发的 ✶ Desktop Sandbox (beta) ✶。开箱即用的隔离安全环境，带有桌面 GUI。专为 LLMs 使用（即 Computer Use）和运行而优化...
- [来自 Anthropic (@AnthropicAI) 的推文](https://x.com/AnthropicAI/status/1849840131412296039)：Anthropic 的新研究：评估 feature steering。今年 5 月，我们发布了 Golden Gate Claude：由于我们使用了 “feature steering”，这个 AI 对金门大桥（Golden Gate Bridge）非常着迷。我们现在进行了更深入的...
- [来自 Jimmy Apples 🍎/acc (@apples_jimmy) 的推文](https://x.com/apples_jimmy/status/1849629937558602216)：引用 Jimmy Apples 🍎/acc (@apples_jimmy) 的话：好了，回到 10 月。我们应该在 10 月迎来一个 4.x 模型（也许还是叫 4.5，我的老朋友）。那个大家伙 GPT 5，我听说最早在 12 月...
- [来自 Nataniel Ruiz (@natanielruizg) 的推文](https://x.com/natanielruizg/status/1849807021131874583)：我正在分享我们在 Google（与 UNC 合作）制作的一些独特的东西。我们正在发布关于一种新型交互体验的研究，我们称之为 generative infinite games（生成式无限游戏），本质上是视频...
- [来自 Bloomberg (@business) 的推文](https://x.com/business/status/1849534815143370815?s=46)：TSMC 在其亚利桑那州的第一家工厂实现了早期生产良率，超过了台湾的同类工厂。对于这个最初受延误困扰的美国扩张项目来说，这是一个重大突破...
- [来自 roon (@tszzl) 的推文](https://x.com/tszzl/status/1849703928394326189?s=46&t=PW8PiFwluc0tdmv2tOMdEg)：创造昴星和参星，使幽暗变为晨光，使白昼变为黑夜，召唤海水并将其倾倒在地面上的那位——
- [来自 Kylie Robison (@kyliebytes) 的推文](https://x.com/kyliebytes/status/1849625175354233184)：独家新闻：OpenAI 计划在 12 月前发布其下一个前沿模型 Orion。OpenAI 计划首先向与其紧密合作的公司提供访问权限，以便它们构建自己的产品 + ...
- [来自 Andrew Curran (@AndrewCurran_) 的推文](https://x.com/andrewcurran_/status/1849483280195997921?s=46&t=PW8PiFwluc0tdmv2tOMdEg)：“在 180 天内 ... AISI 应在至少两个前沿 AI 模型公开发布或部署之前，对其进行自愿性初步测试，以评估可能构成威胁的能力...”
- [来自 Kylie Robison (@kyliebytes) 的推文](https://x.com/kyliebytes/status/1849857410187526513)：在 CEO Sam Altman 称这个故事为“假新闻”后，OpenAI 发言人 Niko Felix 告诉 The Verge，公司“没有计划在今年发布代号为 Orion 的模型”，但“我们确实计划发布...”
- [关于推进美国在人工智能领域的领导地位；利用人工智能实现国家安全目标；以及促进人工智能的安全、保障和可靠性的备忘录 | 白宫](https://t.co/MpqfF48dDk)：致副总统、国务卿的备忘录
- [新的 Claude 3.5 Sonnet：更好，是的，但不仅仅是以你想象的方式](https://www.youtube.com/watch?v=KngdLKv9RAc)：一个新的 SOTA LLM（至少在创意写作和基础推理方面），但在公布的数据背后隐藏着什么？它是真实的吗，AI 是否...
- [Claude 现在可以控制你的电脑——会出什么问题吗？ - TechTalks](https://bdtechtalks.com/2024/10/22/anthropic-claude-computer-use/)：这有很多出错的可能，但具备 computer use 能力的 Claude 可以成为发现新应用的良好实验工具。
- [TurboML - TurboML 利用新鲜数据进行 ML 的平台](https://www.youtube.com/watch?v=wEU9LvCnnY4)：AI 数据：实时、批处理和 LLMs。了解 TurboML 的平台如何克服实时数据带来的挑战，从而实现更新的特征、更快的模型...
- [TurboML](https://2ly.link/20Wmj)：为实时性重塑的机器学习平台。

---

### **Latent Space ▷ #**[**ai-announcements**](https://discord.com/channels/822583790773862470/1075282504648511499/) (1 条消息):

fanahova: 与 NotebookLM 团队合作的新播客上线了！

https://x.com/FanaHOVA/status/1849861104111059082

---

### **Latent Space ▷ #**[**ai-in-action-club**](https://discord.com/channels/822583790773862470/1200548371715342479/1299462688815972447) (260 条消息🔥🔥):

> - `Cursor Pro Tips`
> - `LLM Integration`
> - `Markdown Generation`
> - `Audio Issues in Discord`
> - `OpenAI Documentation Scraping`

- **Cursor 专业技巧汇集**：参与者讨论了大量使用 Cursor 的专业技巧，重点在于利用命令行工具和 CTRL 快捷键来提高效率。
  
  - 分享的见解包括如何利用现有 Markdown 文件的上下文来改进项目描述和工作流。
- **面临 LLM 集成挑战**：参与者对雇主因代码机密性相关的安全问题而限制使用 LLM 表示担忧。
  
  - 建议使用 AWS Bedrock 和 GCP 等替代方案，以便在保持私有数据的同时与模型进行安全交互。
- **Markdown 文件生成**：讨论中强调了 Cursor 在生成 Markdown 方面的某些问题，导致一些用户转而使用 Claude 的功能来快速创建文档。
  
  - 针对使用 Cursor 进行此类操作的挫败感，大家进行了幽默的调侃，并引发了关于未来改进的讨论。
- **Discord 音频问题**：几位用户在会议期间遇到了间歇性的音频问题，推测可能是 Discord 的性能原因。
  
  - 尽管存在技术挑战，小组仍保持活跃，并继续讨论与 Cursor 和 LLM 相关的各种话题。
- **探索 OpenAI 文档抓取**：参与者表达了对抓取 OpenAI 文档以更轻松获取信息的兴趣，并提到了有效收集数据的工具和技术。
  
  - 创意解决方案包括使用命令行技巧和其他手段，以更高效地浏览现有文档。

**提到的链接**：

- [no title found](https://docs.cursor.com/context/ignore-files)：未找到描述
- [Trolling Is An Honored Profession Leland Townsend GIF - Trolling Is An Honored Profession Leland Townsend Evil - Discover & Share GIFs](https://tenor.com/view/trolling-is-an-honored-profession-leland-townsend-evil-the-demon-of-memes-trolling-is-a-respectable-career-gif-26018384)：点击查看 GIF
- [Hp Harry Potter GIF - Hp Harry Potter Snape - Discover & Share GIFs](https://tenor.com/view/hp-harry-potter-snape-always-stare-gif-17584635)：点击查看 GIF
- [YOLO11 🚀 NEW](https://docs.ultralytics.com/models/yolo11/)：探索 YOLO11，这是最先进的目标检测领域的最新进展，为各种计算机视觉任务提供无与伦比的准确性和效率。
- [yikes’s Substack | Substack](https://butdoesitworktho.substack.com)：我的个人 Substack。点击阅读 yikes’s Substack，一个 Substack 出版物。一年前发布。
- [RDoc Documentation](https://ruby-doc.org/3.3.5/index.html)：未找到描述
- [RDoc Documentation](https://ruby-doc.org/3.3.5/)：未找到描述
- [GitHub - sigoden/aichat: All-in-one LLM CLI tool featuring Shell Assistant, Chat-REPL, RAG, AI tools & agents, with access to OpenAI, Claude, Gemini, Ollama, Groq, and more.](https://github.com/sigoden/aichat)：多合一 LLM CLI 工具，具有 Shell 助手、Chat-REPL、RAG、AI 工具和 Agent，支持访问 OpenAI, Claude, Gemini, Ollama, Groq 等。- sigoden/aichat
- [GitHub - twilwa/crawler.nvim: uses firecrawl, jina, and/or jsondr to render webpages in neovim buffers](https://github.com/twilwa/crawler.nvim)：使用 firecrawl, jina 和/或 jsondr 在 neovim 缓冲区中渲染网页 - twilwa/crawler.nvim
- [Cursor Directory](https://cursor.directory/)：为你的框架和语言寻找最佳的 Cursor 规则

---

### **Notebook LM Discord ▷ #**[**use-cases**](https://discord.com/channels/1124402182171672732/1124403655819415592/1299086083706847273) (71 条消息🔥🔥):

> - `Podcast Customization and AI Interactions`（播客定制与 AI 交互）
> - `Deepfake Technology Discussion`（Deepfake 技术讨论）
> - `Quiz Game Integrations with AI`（AI 问答游戏集成）
> - `Character Development Using AI`（使用 AI 进行角色开发）
> - `AI Performance and Limitations`（AI 性能与局限性）

- **播客定制带来独特的交互体验**：用户讨论了特定的 Prompt（如分配姓名和角色）如何帮助保持 AI 生成播客的连贯性。一位成员成功编写了指令，确保了主持人介绍的一致性，并在整个节目中保留了角色主题。
  
  - 另一位用户指出了角色的局限性：通常男声是主持人，而女声充当专家，这很难轻易更改。
- **Deepfake 技术引发伦理辩论**：围绕 Deepfake 技术的讨论揭示了对其伦理影响的冲突观点，特别是在知情同意不明确的情况下。用户强调了透明同意的重要性，认为公众对 Deepfake 缺乏了解。
  
  - 人们对 AI 的去人性化以及虚拟形象（avatars）的创建是否符合伦理表示担忧，一些成员建议责任应由内容生成者承担，而非技术本身。
- **AI 生成的问答游戏功能**：一位用户尝试使用 AI 创建问答节目格式，由专家和挑战者根据生成的资料交换问题。初步测试显示了生成问题的能力，但在统计正确和错误回答时出现了问题。
  
  - 在计分过程中发现了偏差，凸显了 AI 在数学准确性方面的局限性，这是此类模型的共同挑战。
- **使用 AI 进行角色开发和剧本朗读**：另一位成员提到使用 AI 分析他们的剧本草稿，以寻找故事漏洞或角色动机不足之处。通过调整 Prompt，他们成功进行了场景的“围读（table reads）”，从而获得了更深层次的叙事见解。
  
  - 这种方法不仅有助于完善角色弧线，还通过与 AI 的交互式参与构思了潜在的背景故事元素。
- **AI 性能局限性说明**：参与者观察到 AI 经常出现幻觉（hallucinates），特别是在计数错误和事实准确性方面。这些局限性引发了关于利用 Python 等辅助工具处理需要数学功能的任务的讨论。

**提到的链接**：

- [no title found](https://notebooklm.google.com/notebook/782cdd4f-f030-4868-9d23-93cc42d2b301/audio)：未找到描述
- [Notebooklm GIF - Notebooklm - Discover & Share GIFs](https://tenor.com/view/notebooklm-gif-13936203667734517599)：点击查看 GIF
- [no title found](https://notebooklm.google.com/notebook/a8afe58c-2e9b-4a2f-81fc-deb958c83b9d/audio)：未找到描述
- [Vocaroo | Online voice recorder](https://vocaroo.com/1kBMrHmdLKOp)：未找到描述
- [BIO259 - AI-cast - Muscle Activation](https://www.youtube.com/watch?v=D39tAmo_h_M)：未找到描述
- [The Zombification of JungleTV (serious)](https://www.youtube.com/watch?v=JM5SSfYR5Vs)：阅读来自 JungleTV 创始人 gbl08ma 的完整消息：https://jungletv.live/documents/zombie 播客 (音频): notebooklm.google.com 库存视频: Pexels....

---

### **NotebookLM Discord ▷ #**[**general**](https://discord.com/channels/1124402182171672732/1124402182909857966/1299085475897937930) (186 条消息🔥🔥):

> - `语音和音频问题`
> - `使用 AI 制作播客`
> - `数据存储与限制`
> - `转录生成与时间戳`
> - `NotebookLM 功能与改进`

- **音频上传挑战**：用户报告了从 Android 设备向 NotebookLM 上传音频文件时的困难，特别是在 Media Picker 和整体文件访问权限方面存在问题。
  
  - 建议的临时解决方案包括使用桌面浏览器进行上传，目前正在讨论影响这些功能的已知 Bug。
- **在播客中集成自定义 AI 语音**：参与者讨论了使用来自 Speechify 等工具生成的 AI 语音来替换 NotebookLM 播客中的默认语音，同时保持语调起伏的可能性。
  
  - 此外，提到了 Eleven Labs 作为一个可能复制现有 AI 语音的工具，但用户应注意潜在的版权影响。
- **数据存储查询**：关于 NotebookLM 在何处存储生成数据的问题得到了解答，明确了数据存储在 Google 的内部系统中，且不计入用户的存储限制。
  
  - 注意到有关于具体使用了哪些 Google Cloud 服务进行存储的进一步询问，强调了该服务需要更高的透明度。
- **转录和章节生成挑战**：用户尝试从音频内容中生成准确的转录和章节划分，但遇到了生成的时间戳有时不一致的问题。
  
  - 提出了各种建议，包括使用外部转录提取服务来提高 YouTube 章节生成的准确性。
- **NotebookLM 反馈与增强**：对 NotebookLM 能力的认可凸显了其功能受到的好评，以及其改变内容创作流程的潜力。
  
  - 用户对高级功能和各种配置设置表现出兴趣，同时也讨论了可以提升整体体验的增强功能。

**提到的链接**：

- [NotebookLM](https://youtube.com/playlist?list=PLLY_DJYCPJbvrhRcNztk6L51EKlpQIfmf&si=k03VL62c2Ys6Dfip): 未找到描述
- [未找到标题](https://notebooklm.google.com/notebook/810c42b6-3ed3-464c-a67a-7e1339715673/audio): 未找到描述
- [未找到标题](https://notebooklm.google.com/notebook/c8a760d6-d02c-49ff-ab7d-44556c555d99/audio): 未找到描述
- [未找到标题](https://notebooklm.google.com/notebook/e33ed037-dc5c-461d-910d-75646d58fff2/audio): 未找到描述
- [未找到标题](https://ai.google.dev/gemini-api/docs/vision?lang=python#prompting-video): 未找到描述
- [Help](https://support.google.com/notebooklm#topic=14775295): 未找到描述
- [Reddit - Dive into anything](https://www.reddit.com/r/notebooklm/comments/1gbf3sk/deep_dive_epicurus_descartes_and_god/): 未找到描述
- [未找到标题](https://notebooklm.google.com/notebook/7f340391-ef2a-43e3-a05e-071d0dc588da): 未找到描述
- [未找到标题](https://notebooklm.google.com/notebook/77efa571-52a7-48f5-90df-92db88317111): 未找到描述

---

### **LM Studio ▷ #**[**general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1299085179918225428) (199 条消息🔥🔥):

> - `LM Studio 功能请求`
> - `模型兼容性问题`
> - `大型模型的性能关注`
> - `GPU 与 CPU 的交互`
> - `LM Studio 中的文件管理`

- **关于 LM Studio 插件支持的讨论**：用户对 LM Studio 中用户创建插件的潜力表示了兴趣，强调了其在不增加额外复杂性的情况下扩展功能的重要性。
  
  - 建议通过与现有工具更好的集成来增强用户体验，特别是通过开放的 API 端点。
- **模型加载问题**：一位用户报告了加载 Mamba-Codestral 模型失败的问题，这与指示潜在 GPU 问题或驱动程序冲突的设备错误有关。
  
  - 建议包括清理着色器缓存（shader caches）以及调整 GPU offload 百分比以缓解 VRAM 限制。
- **对模型可用性的担忧**：讨论了运行大型语言模型时不同模型的有效性及其相关的 offloading 能力。
  
  - 有人指出，有限的 VRAM 可能会显著影响模型的性能和可用性。
- **文件管理和应用程序结构**：用户对应用程序文件的组织结构有着共同的担忧，建议将所有文件整合到一个目录中，以提高清晰度和易用性。
  
  - 目前的设置将文件分散在不同位置，这使得管理变得复杂，并需要持续的清理工作。
- **大型语言模型的性能**：用户分享了加载和运行大型 LLM 的经验，指出了在不耗尽系统资源的情况下实现最佳性能的挑战。
  
  - 强调了虽然更大的模型尺寸通常会增加 context length，但它们对硬件的要求也更高，从而导致响应时间变慢。

**提到的链接**：

- [lmstudio-community/MiniCPM-V-2_6-GGUF · Hugging Face](https://huggingface.co/lmstudio-community/MiniCPM-V-2_6-GGUF)：未找到描述
- [Running a Local Vision Language Model with LM Studio to sort out my screenshot mess – Daniel van Strien](https://danielvanstrien.xyz/posts/2024/11/local-vision-language-model-lm-studio.html)：未找到描述
- [desktopCapturer | Electron](https://www.electronjs.org/docs/latest/api/desktop-capturer)：访问有关媒体源的信息，这些媒体源可用于使用 navigator.mediaDevices.getUserMedia API 从桌面捕获音频和视频。
- [Build software better, together](https://github.com/search?q=repo%3Achromium%2Fchromium+desktopcapturer&type=code)：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。
- [Old CPU, trouble with FMA/gemm (and a workaround) · Issue #1327 · huggingface/candle](https://github.com/huggingface/candle/issues/1327)：你好，感谢这个项目。太棒了！我在运行过程中遇到了很多麻烦，所以这里是我的笔记。希望对其他在 GPU-poor 道路上的人有所帮助。请注意...

---

### **LM Studio ▷ #**[**hardware-discussion**](https://discord.com/channels/1110598183144399058/1153759714082033735/1299168262994792501) (33 messages🔥):

> - `Intel Arc A750 performance`
> - `Gemma 2 token speeds`
> - `Mistral 7B usability concerns`
> - `GPU mixing for ML tasks`
> - `AMD SAM impact on performance`

- **Intel Arc A750 在 LM Studio 中表现出人意料地出色**：一位用户报告称，他们的新 **Intel Arc A750** GPU 在 **LM Studio** 中运行得非常好，性能超过了之前的 **6750xt**。
  
  - 这一变化激发了人们对新型 GPU 在 ML 任务中能力的兴趣。
- **Gemma 2 Token 速度报告**：用户分享了使用不同模型的经验，**Gemma 2 2B** 达到了 **25 tokens/s**，而 **Gemma 2 9B** 的速度约为 **6 tokens/s**。
  
  - 用户对 2B 模型输出中的错误表示了担忧。
- **Mistral 7B 偏好与模型问题**：尽管 **Mistral 7B** 是心头好，但最近在性能上表现挣扎，特别是在最新更新之后，导致 **instruct 模型** 出现故障。
  
  - 用户强调需要升级或探索更新的模型，例如 **Ministral 8B**。
- **混合 GPU 进行 ML - 性能考量**：一位用户询问混合使用不同的 GPU（如 **4090** 和 **3090**）是否会比匹配相同型号影响性能。
  
  - 讨论中提到了潜在的瓶颈，回复指出推理速度可能会受限于较慢的 GPU。
- **AMD SAM 对模型加载速度的影响**：一位用户发现启用 **AMD SAM** 可能会通过将部分内容卸载到“Shared GPU Memory”中来减慢模型加载速度。
  
  - 禁用 SAM 带来了性能提升，使他们的模型能以 **33.15 tokens/sec** 的速度运行。

 

**提到的链接**：[snowbin](https://pastes.fmhy.net/M0CZf1)：精心制作的 pastebin。

 

---

### **aider (Paul Gauthier) ▷ #**[**general**](https://discord.com/channels/1131200896827654144/1131200896827654149/1299086390151221259) (121 messages🔥🔥):

> - `DeepSeek performance`
> - `Aider v0.60.1 features`
> - `Prompt caching`
> - `PearAI and Aider integration`
> - `Claude 1022 behavior issues`

- **DeepSeek 提供快速性能**：一位用户报告称，在将 DeepSeek 用于 editor-model 时，处理过程中没有感觉到明显的延迟。
  
  - 另一位用户根据此反馈表达了尝试 DeepSeek 的兴奋之情。
- **Aider v0.60.1 的新功能**：即将发布的 Aider v0.60.1 更新增加了对 Claude 3 模型、文件排序的支持，以及一个新的 --fancy-input 标志以实现更好的命令处理。
  
  - 用户正在推测更新安装的延迟以及潜在的本地问题。
- **Prompt 缓存节省成本**：用户讨论了 Aider 中可用的 Prompt 缓存选项，以在降低成本的同时提高性能，特别是对于 Sonnet 模型。
  
  - 他们强调了启用缓存如何显著减少与本地编码任务相关的费用。
- **PearAI 集成 Aider**：有讨论称 PearAI 据报道在其编程功能中使用了 Aider，这引发了关于权限和集成的疑问。
  
  - 用户对 PearAI 可能对 Aider 的功能进行重新品牌化或重新包装表示了担忧。
- **对 Claude 1022 行为的担忧**：一些用户报告称，Claude 1022 与 Cursor 等工具配合使用时输出不可预测，表明它表现出类似过去版本的过度活跃行为。
  
  - 其他人认为他们的详细 Prompt 是有效的，这表明用户 Prompt 规范可能需要改进以保持控制。

**提到的链接**：

- [Introducing PearAI Creator (Beta) — Powered By aider](https://trypear.ai/blog/introducing-pearai-creator-beta)：PearAI Creator 可以自动为你构建应用、修复 Bug 并实现新功能。了解如何使用这个由 aider 提供支持的强大新功能。
- [Prompt caching](https://aider.chat/docs/usage/caching.html)：Aider 支持 Prompt 缓存，以节省成本并加快编码速度。
- [Release history](https://aider.chat/HISTORY.html#aider-v0601)：关于 aider 编写自身代码的发布说明和统计数据。
- [The League of Gentlemen season 1 episode 1 - local shop](https://www.youtube.com/watch?v=F75d01l5AxM)：未找到描述。

---

### **aider (Paul Gauthier) ▷ #**[**questions-and-tips**](https://discord.com/channels/1131200896827654144/1133060505792159755/1299086039515795571) (53 messages🔥):

> - `Aider installation issues`
> - `Using Aider with Groq/Gemma2`
> - `Aider features and experiences`
> - `AI tools in the workplace`
> - `Aider updates`

- **Aider 安装故障排除**：一位用户在尝试运行来自 Aider 的代码时遇到了 `ModuleNotFoundError`，被建议创建一个虚拟环境并在其中安装 Aider。
  
  - 用户确认命令行功能正常，但在其 Python 环境中遇到问题，引发了对模块访问权限的担忧。
- **将 Aider 连接到 Groq/Gemma2**：一位用户寻求关于如何将 Aider 连接到其公司的 Groq/Gemma2 模型，以及 Groq 提供的 API key 的作用。
  
  - 关于 API key 是否链接到已训练的公司模型存在困惑，建议咨询公司内部以获取具体的集成细节。
- **Aider 与 AI 工具的使用体验**：一位用户分享了 Aider 提高编程效率的经验，并对它提供的学习潜力表示赞赏，特别是通过动手实验。
  
  - 他们提到了在之前的工作环境中使用 AI 工具的挑战，强调了在编程实践中熟悉如何有效使用 AI 的重要性。
- **公司对 AI 编程的态度**：一些用户讨论了关于采用 AI 工具的职场文化，强调了对于使用 AI 辅助是显得不专业还是提高生产力的复杂情绪。
  
  - 一位用户强调，紧跟 AI 工具的发展对开发者至关重要，鼓励他人拥抱新技术以免落后。
- **Aider 的最新更新**：一位用户询问了 Aider 0.60.1 版本的变化，指出该版本发布时缺乏公告。
  
  - 另一位成员澄清说，补丁版本通常是微小的 Bug 修复，这就是它们可能不会公开发布的原因。

**提到的链接**：

- [Connecting to LLMs](https://aider.chat/docs/llms.html)：Aider 可以连接到大多数 LLMs 进行 AI 结对编程。
- [aider/HISTORY.md at main · Aider-AI/aider](https://github.com/Aider-AI/aider/blob/main/HISTORY.md)：Aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建一个账号来为 Aider-AI/aider 的开发做出贡献。

---

### **Nous Research AI ▷ #**[**general**](https://discord.com/channels/1053877538025386074/1149866623109439599/1299088791771807827) (152 条消息🔥🔥):

> - `Nous Research and Corporate Partnerships` (Nous Research 与企业合作伙伴关系)
> - `AI Hype Cycle` (AI 炒作周期)
> - `Model Performance and Benchmarks` (模型性能与基准测试)
> - `Quantization Techniques` (量化技术)
> - `Decentralized AI Development` (去中心化 AI 开发)

- **Nous Research 参与收入共享合作伙伴关系**：Nous Research 宣布与 Hyperbolic 建立合作伙伴关系，分享由其 Hermes 3 模型产生的收入，展示了一种协作式的资金筹集方式。
  
  - 成员们讨论了这一合作伙伴关系的意义，强调这并不等同于“出卖”，而是一种互利共赢的安排。
- **讨论 AI 炒作的现状**：几位成员指出，与今年早些时候相比，AI 的炒作似乎有所降温，可能是受到了即将到来的美国大选等更广泛事件的影响。
  
  - 讨论包括推测 AI 社区是否正处于“期望膨胀”阶段，而非真正的参与。
- **关于模型性能基准测试的辩论**：针对 Llama 4 模型与其他模型（如 Claude）的性能进行了辩论，一些成员对当前的 Benchmark 方法表示怀疑。
  
  - 一位成员强调了 Llama 4 在 120+ tps 下的性能，引发了关于将其与现有 AI 模型进行比较的有效性的讨论。
- **探索量化技术**：成员们讨论了 Meta 最近推出的量化模型，并对在现有模型中应用 Quantization-aware training（量化感知训练）的可行性和潜在收益提出了疑问。
  
  - 一些人对这类技术的计算复杂度表示担忧，特别是对于大型模型。
- **对去中心化 AI 的兴趣日益增长**：目前，人们对去中心化 AI 开发表现出浓厚兴趣，讨论围绕 Prime Intellect 等平台展开，这些平台支持 AI 模型的协作训练。
  
  - 讨论了新一波开放 AI 进展的可能性，重点关注社区驱动的倡议。

**提到的链接**：

- [Introducing Prime Intellect](https://www.primeintellect.ai/blog/introducing-prime-intellect)：Prime Intellect 正在构建大规模去中心化 AI 开发的基础设施。我们整合全球算力，使研究人员能够通过协作训练最先进的模型...
- [来自 nico (@nicochristie) 的推文](https://x.com/nicochristie/status/1849583415194194309?s=46)：我们一直被问及新的 3.5 Sonnet 升级如何影响多 Agent 社会的进展，答案是新模型对长时间自主性有显著提升。我们的 25 个 Agent...
- [来自 adi (@adonis_singh) 的推文](https://fxtwitter.com/adonis_singh/status/1849529291085623372?t=Zeg0OFKmKgWwgycl5O6BNw&s=19)：我让新旧 3.5 Sonnet 进行了一场 Minecraft 建造比赛。这是唯一可靠的基准测试。左：新 3.5 Sonnet，右：旧 3.5 Sonnet。
- [Hyperbolic Partners with Hermes 3 Creators – Nous Research](https://hyperbolic.xyz/blog/hyperbolic-partners-with-nous-research)：今天，我们自豪地宣布与 Hermes 3 大语言模型的创造者 Nous Research 建立收入共享合作伙伴关系。
- [Not A Tunnell Its Dimming GIF - Not A Tunnell Its Dimming Dark Tunnel - Discover & Share GIFs](https://tenor.com/view/not-a-tunnell-its-dimming-dark-tunnel-gif-14376722)：点击查看 GIF
- [关于推进美国在人工智能领域领导地位的备忘录；利用人工智能实现国家安全目标；并促进人工智能的安全、保障和可信度 | 白宫](https://www.whitehouse.gov/briefing-room/presidential-actions/2024/10/24/memorandum-on-advancing-the-united-states-leadership-in-artificial-intelligence-harnessing-artificial-intelligence-to-fulfill-national-security-objectives-and-fostering-the-safety-security/)：副总统、国务卿备忘录。
- [来自 AI at Meta (@AIatMeta) 的推文](https://x.com/AIatMeta/status/1849469912521093360?t=brsdcyrseDaeypjqSyPMjQ&s=19)：我们希望让更多人能更轻松地使用 Llama 进行构建——因此今天我们发布了 Llama 3.2 1B 和 3B 的新量化版本，推理速度提升了 2-4 倍，且平均而言...
- [From Black Holes Entropy to Consciousness: The Dimensions of the Brain Connectome](https://www.youtube.com/watch?v=mlCI5DJM5gc)：提供的文本是一篇科学文章的摘录，探讨了意识、大脑连接组与时空概念之间的联系...
- [GitHub - kolbytn/mindcraft](https://github.com/kolbytn/mindcraft/tree/main)：通过在 GitHub 上创建账户，为 kolbytn/mindcraft 的开发做出贡献。

---

### **Nous Research AI ▷ #**[**ask-about-llms**](https://discord.com/channels/1053877538025386074/1154120232051408927/1299471355531497523) (5 条消息):

> - `Hermes 3 SFT dataset`
> - `OpenHermes 2.5 dataset`
> - `Open source SFT datasets`

- **Hermes 3 SFT 数据集未开源**：一名成员在询问该数据集的可用性后确认，**Hermes 3 SFT 数据集**并未开源。
  
  - 这凸显了一个趋势，即 Hermes 1、2 和 2.5 等早期版本仍保持开放访问。
- **发现 OpenHermes 2.5 数据集**：另一名成员分享了 [OpenHermes 2.5 数据集的链接](https://huggingface.co/datasets/teknium/OpenHermes-2.5)，该数据集驱动了 Hermes 2.5 和 Nous Hermes 2 模型。
  
  - 该数据集是各种开源数据集和自定义合成数据集的集大成之作，展示了 SOTA LLM 的进步。

 

**提到的链接**：[teknium/OpenHermes-2.5 · Datasets at Hugging Face](https://huggingface.co/datasets/teknium/OpenHermes-2.5)：未找到描述

 

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1299396790004486194) (2 条消息):

> - `Softmax function limitations`
> - `Adaptive temperature mechanism`
> - `Linear Attention`
> - `Attention coefficients`

- **Softmax 函数在大规模下失效**：最近的一篇 [Google DeepMind 论文](https://arxiv.org/abs/2410.01104) 揭示，随着输入数量的增加，**softmax 函数**难以保持锐度（sharpness），导致注意力系数分散。
  
  - *实验表明*，虽然模型在熟悉的问题上能实现高度聚焦，但在更大规模的分布外（out-of-distribution）场景中，这种特性会恶化。
- **提出的解决方案：自适应温度**：为了减轻分散现象，作者建议为 softmax 函数引入一种**自适应温度机制**，根据输入熵进行调整以锐化注意力系数。
  
  - 论文还指出，虽然实现零温度可以确保锐度，但对于训练大型语言模型（LLM）来说，这存在实际的挑战。
- **呼吁替代注意力方法**：研究人员敦促进一步探索*替代注意力函数*，以更好地解决 AI 推理引擎中的锐度和泛化挑战。
  
  - 随着模型规模的扩大，这种转变可能会带来各种模型性能的提升。
- **线性注意力超越传统方法**：一名成员提出了一个有趣的观点，认为在足够的规模下，**Linear Attention** 的表现可能优于传统机制，引发了对其有效性的关注。
  
  - *这一见解*反映了关于随着输入规模增长，不同注意力策略效率的持续争论。

 

**提到的链接**：[edward hicksford (@citizenhicks) 的推文](https://x.com/citizenhicks/status/1849223057757684157?s=46)：这篇 @GoogleDeepMind 的论文探讨了人工智能系统中 softmax 函数的局限性，特别是随着输入数量增加，它无法在决策中保持锐度……

 

---

### **Nous Research AI ▷ #**[**interesting-links**](https://discord.com/channels/1053877538025386074/1132352574750728192/1299275150667026483) (2 条消息):

> - `SynthID Text Watermarking`
> - `OmniParser for Screen Parsing`

- **GDM 开源 SynthID 水印方案**：GDM 发布了用于 **Gemini** 的 [SynthID 水印方案](https://github.com/google-deepmind/synthid-text)，使开发者能够轻松集成水印功能。
  
  - 该项目托管在 GitHub 上，鼓励对水印技术的贡献和探索，详情见 [项目页面](https://github.com/google-deepmind/synthid-text)。
- **OmniParser 转换 UI 截图**：**OmniParser** 现已发布，旨在解释 UI 截图并将其转换为结构化格式，以改进基于 LLM 的 UI Agent，其训练数据集侧重于图标检测和描述。
  
  - 了解更多关于其开发中使用的 [YOLOv8](https://arxiv.org/abs/2408.00203) 和 [BLIP-2](https://www.microsoft.com/en-us/research/articles/omniparser-for-pure-vision-based-gui-agent/) 等模型的技术细节。

**提到的链接**：

- [microsoft/OmniParser · Hugging Face](https://huggingface.co/microsoft/OmniParser)：未找到描述
- [GitHub - google-deepmind/synthid-text](https://github.com/google-deepmind/synthid-text)：通过在 GitHub 上创建账号来为 google-deepmind/synthid-text 的开发做出贡献。

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1299396790004486194) (2 messages):

> - `Softmax Function Limitations`
> - `Adaptive Temperature Mechanism`
> - `Linear Attention Performance`

- **Softmax 函数受到审视**：来自 [Google DeepMind](https://x.com/citizenhicks/status/1849223057757684157?s=46) 的一篇论文探讨了 AI 系统中 **Softmax 函数的局限性**，指出随着输入规模的增加，它难以在决策中保持锐度（sharpness）。
  
  - 关键发现表明，即使在训练的问题规模内最大项目能够清晰聚焦，也会出现 **分散的注意力系数（dispersed attention coefficients）**，并且在分布外（out-of-distribution）场景中情况会进一步恶化。
- **用于锐化注意力的自适应温度机制**：作者为 Softmax 提出了一种 **自适应温度机制（adaptive temperature mechanism）** 以对抗分散，根据输入熵调整温度以增强注意力系数。
  
  - 他们建议，虽然 **零温度（zero temperature）** 可以保证锐度，但对于训练大规模 LLM 并不实际，因此值得在训练后阶段进行谨慎探索。
- **Linear Attention 与传统 Attention 的对比**：关于 **Linear Attention** 在足够规模下的性能展开了讨论，表明它可能优于传统的 Attention 方法。
  
  - 这引发了人们对在更大输入场景中采用 Linear Attention 的潜在效率和优势的好奇。

**提及的链接**：[edward hicksford (@citizenhicks) 的推文](https://x.com/citizenhicks/status/1849223057757684157?s=46)：这篇 @GoogleDeepMind 的论文探讨了人工智能系统中 Softmax 函数的局限性，特别是随着输入数量的增加，它无法保持决策的锐度……

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1299174121422327819) (11 messages🔥):

> - `NEO Model Testing`
> - `Topology Book Recommendations`
> - `Fine-Tuning Llama 3.2`
> - `Embedding Models for Classification`
> - `Finetuning Strategies`

- **NEO 测试显示改进**：一位用户报告称，在本地测试 **NEO** 模型的结果是，它在每一轮对话中都变得 **更快** 且 **更聪明**。
  
  - 他们表达了对 **Pile** 数据集的好奇，并发现交互过程非常愉快。
- **拓扑学推荐 Munkres**：一位成员询问优秀的 **Topology** 书籍推荐，另一位成员建议将 **Munkres** 作为可靠的选择。
  
  - 这本书在学习拓扑学的人群中享有盛誉。
- **微调 Llama 3.2 模型**：一位成员咨询关于微调 **Llama 3.2** 模型以将文本分为 **20 个类别** 的问题，并询问是否应该使用 **DPO**。
  
  - 建议包括使用简单的分类器和 Text Embedding 模型提取特征，但对数据集结果的性能表示了担忧。
- **选择 Embedding 模型**：一位成员强调了一个用于评估哪些 **Embedding 模型** 最适合各种任务的资源，链接到了 [Hugging Face 排行榜](https://huggingface.co/spaces/mteb/leaderboard)。
  
  - 该资源可以帮助用户根据特定需求选择合适的模型。
- **微调与推理策略**：一些成员讨论了微调的替代方案，建议使用具有结构化输出的 LLM，或者应用 **常规微调** 技术而非 **DPO**。
  
  - 技巧包括探索 Pause Tokens，以及在投入全面微调之前，先简单地查询模型性能。

**提及的链接**：[MTEB Leaderboard - mteb 的 Hugging Face Space](https://huggingface.co/spaces/mteb/leaderboard)：未找到描述

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1299085061584322750) (116 messages🔥🔥):

> - `Diffusion Models`
> - `Classifier-Free Guidance`
> - `Image Captioning Quality`
> - `Unconditional Generation`
> - `Model Autophagy Disorder`

- **辩论 Classifier-Free Guidance 的有效性**：人们对 **Classifier-Free Guidance (CFG)** 的有效性持怀疑态度，讨论了它对 Timestep 的依赖性，以及在应用高 Guidance Scale 时潜在的饱和问题。
  
  - 成员们建议，理想的解决方案应包括简化流程，直接根据文本输入生成，而无需复杂的权重计算。
- **图像描述数据集的挑战**：一位成员强调了数据集中 Caption 质量差的问题，指出即使是重新标注（Re-captioning）的努力也可能无法解决准确性和相关性的根本问题。

- 这引发了大规模创建高质量 caption 的挑战，人们担心即使是新生成的 caption 可能也无法改善现有问题。
- **调查无条件生成（Unconditional Generation）基准**：人们呼吁为 **unconditional generation** 模型建立更好的基准，特别是强调了条件生成与无条件结果之间的保真度差距。
  
  - 有建议指出，使用像 Imagenet 或 CIFAR 这样不带特定条件的数据集开发基准，可以提供有价值的见解。
- **探索合成数据训练问题**：一篇论文讨论了 **Model Autophagy Disorder (MAD)** 的概念，警告不要在由前几次迭代创建的合成数据上训练生成模型，因为这会产生退化效应。
  
  - 对话中提出，模型可以从经过显式训练的 corrector 模型中获益，从而提高生成输出的质量。
- **比较训练中的优化方法**：围绕 **Adagrad 和 Adafactor** 等优化算法展开了讨论，重点关注它们在不同维度上的效率和影响。
  
  - 成员们得出结论，虽然 **VectorAdam** 在低维度下可能表现更好，但其在高维度环境下的有效性仍不确定。

**提到的链接**：

- [Return of Unconditional Generation: A Self-supervised Representation Generation Method](https://arxiv.org/abs/2312.03701)：无条件生成——在不依赖人工标注标签的情况下对数据分布进行建模的问题——是生成模型中一个长期存在的根本性挑战，创造了一个潜在的...
- [Variational Diffusion Models](https://arxiv.org/abs/2107.00630)：基于扩散的生成模型已展示出令人印象深刻的感知合成能力，但它们也能成为优秀的基于似然（likelihood-based）的模型吗？我们对此给出了肯定的回答，并介绍了...
- [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970)：扩散模型作为生成模型展示了惊人的能力；事实上，它们为目前最先进的文本条件图像生成模型（如 Imagen 和 DALL-E 2）提供了动力。在这篇...
- [Classifier-Free Guidance is a Predictor-Corrector](https://arxiv.org/abs/2408.09000)：我们研究了 Classifier-Free Guidance (CFG) 的理论基础。CFG 是文本到图像扩散模型条件采样的主要方法，但与扩散的其他方面不同...
- [Self-Improving Diffusion Models with Synthetic Data](https://arxiv.org/abs/2408.16333)：人工智能 (AI) 领域正面临用于训练日益庞大的生成模型的真实数据枯竭的问题，导致在合成数据上进行训练的压力不断增大。不幸的是，训练...
- [Inductive Biases and Variable Creation in Self-Attention Mechanisms](https://arxiv.org/abs/2110.10090)：Self-attention 是一种旨在对序列数据中的长程交互进行建模的架构基元，它推动了自然语言处理及其他领域的众多最新突破。这项工作提供...
- [Scaling MLPs: A Tale of Inductive Bias](https://arxiv.org/abs/2306.13575)：在这项工作中，我们重新审视了深度学习中最基础的构建块——多层感知器 (MLP)，并研究了其在视觉任务上性能的极限。对 MLP 的实证见解是...
- [Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models](https://arxiv.org/abs/2410.02416)：Classifier-free guidance (CFG) 对于提高扩散模型的生成质量以及输入条件与最终输出之间的对齐至关重要。虽然高 guidance scale 通常...
- [ComfyUI/comfy/samplers.py at master · comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy%2Fsamplers.py#L688>)：最强大且模块化的扩散模型 GUI、API 和后端，具有图形/节点界面。- comfyanonymous/ComfyUI
- [A Picture is Worth a Thousand Words: Principled Recaptioning Improves Image Generation](https://arxiv.org/abs/2310.16656)：在过去的几年里，文本到图像扩散模型在能力上取得了显著的飞跃，能够根据文本提示实现高质量且多样化的图像合成。然而，即使是最先进的...
- [DreamLIP: Language-Image Pre-training with Long Captions](https://arxiv.org/abs/2403.17007)：语言-图像预训练在很大程度上取决于文本描述其配对图像的精确度和详尽程度。然而在实践中，图像的内容可能非常丰富，以至于需要良好的描述...

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1299089038015336582) (14 条消息🔥):

> - `Raw requests analysis` (原始请求分析)
> - `Model output issues` (模型输出问题)
> - `BOS token requirement` (BOS token 需求)
> - `Pythia model limitations` (Pythia 模型限制)
> - `lm_eval command troubleshooting` (lm_eval 命令故障排除)

- **理解原始请求与处理后的条目**：一位成员澄清说，重点应该放在数据集中的 `arguments` 而不是 `doc`（后者是未处理的数据）。
  
  - 另一位成员评论说，这可能涉及查看预转换（pre-transformed）的数据集。
- **保存生成的模型答案时遇到困难**：有用户报告在使用 `--log_samples` 和 `--write_out` 标志保存模型输出时遇到困难，称只能看到 prompt 但看不到回答。
  
  - 他们注意到响应结构存在问题，特别是 `resps` 键中的值为空。
- **BOS Token 对模型性能的重要性**：一位成员建议某些模型需要 BOS token 才能正常运行，建议在模型参数中设置 `add_bos_token=True`。
  
  - 此外，他们建议对于 instruct 模型，可以使用 `--apply_chat_template` 作为潜在的解决方案。
- **Pythia 模型缺乏 Instruct 微调**：会议强调 Pythia 模型没有经过 instruct 微调，不支持 chat templates 或 BOS tokens。
  
  - 这可能导致立即生成停止序列（stop sequence），对此提出的解决方案包括移除停止序列或使用 few-shot 示例。
- **用户尝试通过 lm_eval 命令进行修复**：一位用户分享了他们尝试使用的 lm_eval 工具命令，其中包含了各种参数以排除输出问题的故障。
  
  - 他们被建议修改命令，以解决在 Pythia 模型上遇到的困难。

 

**提到的链接**：[lm-evaluation-harness/lm_eval/tasks/noticia/noticia.yaml at 7882043b4ee1ef9577b829809c2f4970b0bdba91 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/7882043b4ee1ef9577b829809c2f4970b0bdba91/lm_eval/tasks/noticia/noticia.yaml#L5-L8)：一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness

 

---

### **Eleuther ▷ #**[**gpt-neox-dev**](https://discord.com/channels/729741769192767510/730090096287547444/1299503673508106302) (1 条消息):

> - `Contributing to gpt-neox` (向 gpt-neox 贡献)
> - `GitHub permission issues` (GitHub 权限问题)

- **关于向 gpt-neox 仓库贡献代码的咨询**：一位成员询问如何向 **gpt-neox** 仓库贡献代码，并在尝试向仓库推送（push）更改时遇到了错误。
  
  - *他们报告收到一个* `403` 错误，表明访问仓库时权限被拒绝。
- **关于 GitHub 权限问题的经验**：所描述的问题反映了贡献者在尝试推送到仓库时经常遇到的 **GitHub permission** 挑战。
  
  - 收到 **403** 错误通常意味着用户缺乏必要的访问权限，这引发了社区关于如何解决此类问题的各种讨论。

 

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1299125427440193609) (120 messages🔥🔥):

> - `Anthropic 的 Opus 3.5 发布`
> - `AGI 与 ANI 的讨论`
> - `AI 训练架构`
> - `OpenAI 基础设施报告`
> - `Co-Pilot 功能问题`

- **关于 Anthropic Opus 3.5 发布时间线的推测**：针对 Anthropic 是否会在今年发布 **Opus 3.5** 出现了各种推测，一些人认为可能会推迟到 2025 年。
  
  - *有人建议他们可能会直接跳到一个更新的版本。*
- **关于 AGI 与 ANI 的辩论**：围绕 **Artificial Narrow Intelligence (ANI)** 和 **Artificial General Intelligence (AGI)** 的定义展开了激烈讨论，对于目前的 AI 模型是否符合这些类别，意见不一。
  
  - 一些人提出了 **Emerging AGI**（新兴 AGI）等术语，用来描述通往通用人工智能开发的潜在路径。
- **AI 训练方法与硬件**：成员们讨论了未来 AI 训练的潜力，推测在 **数百万颗 H100** 规模下运行的模型所需的资源。
  
  - 针对下一代 GPU 的生产问题提出了担忧，估计要实现这种规模的扩展，仍需要大量的现有硬件。
- **OpenAI 的基础设施与算力雄心**：OpenAI 最近的一份报告讨论了建造大型 **5GW 数据中心** 以训练先进 AI 模型的宏伟计划，引发了关于此类基础设施的可行性和规模的讨论。
  
  - 一些成员对这些宏大算力目标的实用性和生态影响表示怀疑。
- **更新后 Co-Pilot 图标消失**：一位用户报告称，在更新后其 Windows 系统中的 **Co-Pilot** 图标丢失，引发了关于潜在原因和解决方案的询问。
  
  - 回复从表达困惑到对该情况的幽默评论不等，表明这是用户间的共同经历。

**提到的链接**：

- [microsoft/OmniParser · Hugging Face](https://huggingface.co/microsoft/OmniParser)：未找到描述
- [来自 killian (@hellokillian) 的推文](https://x.com/hellokillian/status/1849248458701705334?t=M5hifaqwGrh417E_rhoflg&s=19)：想用 Claude 来控制你的电脑吗？`pip install open-interpreter interpreter --os`。支持 Windows 和 Mac。玩得开心 :)

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1299145112990912574) (15 messages🔥):

> - `伦理团队沟通`
> - `GPT-4o 记忆功能`
> - `为 AI Agent 使用 API`

- **如何联系 OpenAI 的伦理团队**：一位用户询问如何联系 ChatGPT 的 **伦理团队** 或 **编程团队**，强调了直接联系 OpenAI 支持部门的困难。
  
  - 另一位成员建议使用 [聊天模型反馈表单](https://openai.com/form/chat-model-feedback/) 来报告问题或建议。
- **ChatGPT-4o 记忆访问**：成员们讨论了 **ChatGPT-4o** 在账户开启功能后确实拥有记忆访问权限，而 API 本身并不包含此功能。
  
  - 一位用户澄清说，使用 **Playground** 不会像 Web UI 那样授予记忆访问权限，因为 **API 的运行是独立于** ChatGPT 功能的。
- **通过 API 创建 AI Agent**：一位成员寻求关于使用 API 创建可以对话的 AI Agent 的建议，例如 **CEO 与 CFO** 就某个话题进行对话。
  
  - 另一位成员确认 API 是实现此目标的理想途径，并分享了他们过去通过单个脚本管理 Bot 交互的经验。
- **社区对 AI Agent 实现的支持**：参与者鼓励在指定频道分享问题，以收集关于实现 AI 交互的见解。
  
  - 成员们表示愿意协助提供使用 API 的思路，展示了协作的氛围。

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1299085485011894374) (127 条消息🔥🔥):

> - `Cerebras API 准入`
> - `模型的审查问题`
> - `Prompt Caching`
> - `Token 限制与错误`
> - `新模型的性能`

- **Cerebras API 的准入与使用**：几位用户分享了他们使用 **Cerebras API** 的经验，其中一位提到他们在一个多月前就获得了访问权限，而其他人则报告说在没有正式通过申请的情况下也获得了密钥。
  
  - 讨论内容包括获取 API 密钥的便捷性，以及关于可控的芯片成本与性能之间平衡的潜在问题。
- **Hermes-3 引发审查担忧**：一位用户提出了 **hermes-3-llama-3.1-405b** 是否被审查的问题，反映了社区对模型限制的担忧。
  
  - 这段对话反映了对于 AI 模型可接受内容参数的持续不确定性。
- **Prompt Caching 功能讨论**：社区讨论了 OpenRouter 上 Sonnet 模型的 **prompt caching** 可用性，强调了其在优化 API 使用方面的优势。
  
  - 一位用户指出在实现上存在困难，特别是在与 SillyTavern 等外部应用程序配合使用时。
- **API Token 限制引发困惑**：一位用户表达了在拥有 16 美元余额的情况下仍收到最大 Token 限制错误的沮丧，引发了关于 API 密钥限制和潜在配置的讨论。
  
  - 共识建议将创建新的 API 密钥作为可能的解决方案，并检查账户余额状态。
- **性能问题与 API 可靠性**：用户报告经历了速度变慢并收到错误码 **520**，表明了对系统可靠性的担忧。
  
  - 几项讨论强调了与硬件供应问题相关的挑战，这些问题影响了性能，特别是针对高级模型。

**提到的链接**：

- [关于新 Claude 分析 JavaScript 代码执行工具的笔记](https://simonwillison.net/2024/Oct/24/claude-analysis-tool/)：Anthropic 今天为其面向消费者的 Claude.ai 聊天机器人界面发布了一项名为“分析工具”的新功能。这是他们对 OpenAI ChatGPT Code Interpreter 模式的回应...
- [互联网速度测试 - 衡量网络性能 | Cloudflare](https://speed.cloudflare.com/)：测试您的互联网连接。使用我们的互联网速度测试检查您的网络性能。由 Cloudflare 的全球边缘网络提供支持。
- [聊天室 | OpenRouter](https://openrouter.ai/chat)：LLM 聊天室是一个多模型聊天界面。添加模型并开始聊天！聊天室将数据本地存储在您的浏览器中。
- [Andrej Karpathy (@karpathy) 的推文](https://x.com/karpathy/status/1835024197506187617)：LLM（“大语言模型”）与语言关系不大，这有点令人遗憾且令人困惑；这只是历史原因。它们是用于统计模型的通用技术...
- [额度 | OpenRouter](https://openrouter.ai/credits)：管理您的额度和支付历史。
- [使用 Google Gemini 对图像和 PDF 运行提示词](https://simonwillison.net/2024/Oct/23/prompt-gemini/)：新的 TIL。我一直在尝试使用 Google Gemini API 对图像和 PDF 运行提示词（为最终给 [LLM](https://llm.datasette.io/) 添加多模态支持做准备）...
- [Cerebras (@CerebrasSystems) 的推文](https://x.com/CerebrasSystems/status/1849467762932076801)：我们在 8 月发布 Cerebras Inference 时打破了所有记录。今天，我们将性能从 650 t/s 提高到 2100 t/s，翻了三倍。Cerebras Inference 的速度独占鳌头——比...快 16 倍。
- [Prompt caching (测试版) - Anthropic](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)：未找到描述。
- [Llama 3.1 70B：API 供应商性能基准测试与价格分析 | Artificial Analysis](https://artificialanalysis.ai/models/llama-3-1-instruct-70b/providers)：对 Llama 3.1 Instruct 70B 的 API 供应商在性能指标（包括延迟、首个 Token 时间）、输出速度（每秒输出 Token 数）、价格等方面进行的分析。API 供应商基准测试...
- [Claude 3.5 Sonnet 请求错误率升高](https://status.anthropic.com/incidents/7hfbq9s08kqt)：未找到描述。
- [Prompt Caching | OpenRouter](https://openrouter.ai/docs/prompt-caching)：最高可降低 90% 的 LLM 成本。
- [设置 | OpenRouter](https://openrouter.ai/settings/preferences)：管理您的账户和偏好设置。
- [供应商路由 | OpenRouter](https://openrouter.ai/docs/provider-routing#ignoring-providers)：在多个供应商之间路由请求。
- [OpenRouter 状态](https://status.openrouter.ai/)：OpenRouter 故障历史记录。
- [密钥 | OpenRouter](https://openrouter.ai/settings/keys)：管理您的密钥或创建新密钥。
- [OpenRouter](https://openrouter.ai/)：LLM 路由和市场。

---

### **OpenRouter (Alex Atallah) ▷ #**[**beta-feedback**](https://discord.com/channels/1091220969173028894/1277894087755829278/1299198852590731316) (7 messages):

> - `OpenRouter Integrations`
> - `Anthropic/Claude API Access`

- **用户寻求集成访问权限**：多位用户正在询问如何获取平台上的 **integrations** 访问权限，表达了对利用各种功能的兴趣。
  
  - *许多成员重申了他们的请求*，强调了集成其工作流的共同愿望。
- **将 Anthropic/Claude API 接入 OpenRouter**：一位用户提到打算将其 **Anthropic/Claude API key** 连接到 OpenRouter 以使用 Sonnet，表明了对集成的推动。
  
  - *这显示出在 OpenRouter 环境中利用 API 来增强功能的兴趣日益增长*。

---

### **Stability.ai (Stable Diffusion) ▷ #**[**general-chat**](https://discord.com/channels/1002292111942635562/1002292112739549196/1299094248229965844) (134 messages🔥🔥):

> - `Flux and Comic Creation`
> - `Video Generation with AI`
> - `Stable Diffusion Models`
> - `LoRA Training and Usage`
> - `Art Creation for Music Tracks`

- **Flux 在漫画创作中的挑战**：成员们讨论了使用 **FLUX** 进行漫画生成，以及对特定角色模型进行 fine-tuning 的需求，强调了一致性和 prompt 遵循度。
  
  - *使用标准模型很难达到理想的细节水平，为了保持特定角色的一致性，可能需要训练模型。*
- **Mochi vs. 其他视频生成工具**：用户对比了用于本地视频生成的 **Mochi 1** 和 **CogVideoX**，发现 Mochi 效果更优，但处理速度较慢。
  
  - *尽管在某些任务上被认为不如 Mochi 有效，但由于其特性，仍有建议使用 **CogVideoX**。*
- **Stable Diffusion 3.5 的探索**：成员们质疑 **Stable Diffusion 3.5** 的能力，特别是它生成特定 prompt 的能力，例如“一名女子躺在棉花糖池上”。
  
  - *一位用户表示，使用该 prompt 创建的图像已发布在另一个频道供审查。*
- **为音乐轨道创建艺术作品**：一位成员寻求为 SoundCloud 上的 house 轨道创建封面艺术作品的建议，并提供了关于艺术作品预期的详细 prompt。
  
  - *他们对初步结果与描述不符表示失望，反映了使用 AI 进行艺术创作的学习曲线。*
- **LoRA 训练与数据集的重要性**：讨论围绕着使用高质量数据集为 LoRA 训练模型的必要性展开，以确保高质量的输出。
  
  - *有建议称，在尝试创建或 fine-tune 模型之前，关于数据集准备的教程将使用户大为受益。*

**提到的链接**：

- [Models - Hugging Face](https://huggingface.co/models?other=base_model:adapter:stabilityai/stable-diffusion-3.5-large)：未找到描述
- [Comic Character Loras For Stable Diffusion](https://www.youtube.com/watch?v=KJf67JiOWMY)：在这个史诗级的教程中，学习如何将你自己的角色训练成 Stable Diffusion Lora，以制作出令人惊叹且视觉一致的漫画角色。...
- [Essay Writing Service - Essay Help 24/7 - ExtraEssay.com](https://extraessay.com/?key_wpg=5wpgrd)：最好的论文写作服务，ExtraEssay.com：专业作家，特别折扣，最短期限。我们写论文——你拿高分。

---

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1299089025784614973) (99 条消息🔥🔥):

> - `Perplexity Pro 用户体验`
> - `即将发布的 AI 模型`
> - `Perplexity App 功能`
> - `使用 AI 进行法律研究`
> - `社区故障排除见解`

- **社区讨论 Perplexity Pro 的价值**：成员们分享了关于 **Perplexity Pro** 的使用体验，部分用户对其相对于 Claude 和 GPT 等其他工具的价值提出了疑问。
  
  - 用户表示对最大化 Pro 订阅价值的资源和技巧感兴趣，并强调了有效设置的必要性。
- **对 Gemini 2.0 发布的期待**：随着 Google 和 OpenAI 竞相发布下一代模型，**Gemini 2.0** 预计将很快亮相，尽管人们对性能提升幅度存在担忧。
  
  - 用户注意到 AI 能力的快速进步，但也承认改进分散在来自不同公司的各种模型中。
- **Perplexity App 功能咨询**：出现了关于 **Perplexity App** 功能的问题，包括推理能力以及是否需要访问原生 iOS 语音识别。
  
  - 成员们强调了监控和管理指令设置的重要性，以减少 AI 生成内容中的幻觉（hallucinations）。
- **法律领域的 AI 应用**：讨论了使用 AI 进行**法律研究**的话题，成员们表达了尽管有严格指令但仍难以生成可靠输出的挫败感。
  
  - 分享的经验包括尝试微调 Prompt 以获得更好性能，并反思了对可靠信息来源的需求。
- **寻求故障排除协助**：新用户询问了在哪里发布关于 Perplexity App 的故障排除求助，并遇到了 App 功能方面的问题。
  
  - 社区成员将有关 Bug 报告和支持资源的咨询引导至该 App macOS 版本的相应频道。

**提到的链接**：

- [Lawyer Used ChatGPT In Court—And Cited Fake Cases. A Judge Is Considering Sanctions](https://www.forbes.com/sites/mollybohannon/2023/06/08/lawyer-used-chatgpt-in-court-and-cited-fake-cases-a-judge-is-considering-sanctions/)：律师在法庭上使用 ChatGPT 并引用虚假案例。法官正在考虑制裁。该律师在提交的文件中表示，他不理解 ChatGPT “不是搜索引擎，而是一个生成式语言处理工具”。
- [Google plans to announce its next Gemini model soon](https://www.theverge.com/2024/10/25/24279600/google-next-gemini-ai-model-openai-december)：12 月正逐渐成为 OpenAI 和 Google 展开 AI 发布对决的一个月。
- [TestingCatalog News 🗞 (@testingcatalog) 的推文](https://x.com/testingcatalog/status/1849506668100395310?s=61)：根据最新消息，我们甚至可能在 11 月就看到这一进展。黑色星期五的准备？规模将非常宏大 🔥 https://www.testingcatalog.com/perplexity-progresses-towards-one-click-shopping-with-buy-wit...
- ['They wish this technology didn't exist': Perplexity responds to News Corp's lawsuit | TechCrunch](https://techcrunch.com/2024/10/24/they-wish-this-technology-didnt-exist-perplexity-responds-to-news-corps-lawsuit/)：Perplexity 在周四的一篇博客文章中回击了对 AI 益处持怀疑态度的媒体公司，回应了 News Corp 对该初创公司提起的诉讼。

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1299135991373631551) (8 条消息🔥):

> - `Bitcoin Creator` (比特币创始人)
> - `Carbon Capture Technology` (碳捕集技术)
> - `Space-Based Solar Power` (天基太阳能)
> - `Caffeine Influence` (咖啡因的影响)
> - `Haunted Houses` (闹鬼屋)

- **比特币创始人身份终获确认**：最近的一项发现声称揭示了长期隐匿的**比特币创始人**身份，在加密货币圈引起了轰动。点击此 [YouTube 视频](https://www.youtube.com/embed/sRUpGVJfNJ4)查看完整调查结果。
  
  - 这一发现可能会改变关于 Bitcoin 起源的叙事，并引发区块链社区内的持续讨论。
- **创新粉末捕集碳**：开发出一种**新型粉末技术**，可以有效地从空气中捕集碳，为应对气候变化提供了可能的解决方案。在此[深入了解](https://www.perplexity.ai/search/what-is-conversational-ai-mark-VHSL8GeuRcKbeJUwINkvSw#0)该话题。
  
  - 此类进步可能在降低大气碳含量和促进环境可持续性方面发挥关键作用。
- **为冰岛提供天基太阳能**：一项提案建议利用**天基太阳能**来满足冰岛的能源需求，这可能会彻底改变能源生产方式。在此获取该雄心勃勃项目的[详细信息](https://www.perplexity.ai/page/solar-from-space-for-iceland-jOyPP4O3SzWqU1uhQ18EUA)。
  
  - 如果成功，该倡议可能会为其他寻求可靠且可持续能源的地区树立先例。
- **咖啡因影响的探索**：最近的讨论深入探讨了**咖啡因如何影响**人类行为和生理的不同方面。在此了解其背后的[科学原理](https://www.perplexity.ai/search/how-does-caffeine-influence-te-MEHjXFWxTvWCdNPTbGg08g)。
  
  - 了解咖啡因的影响可以帮助个人优化日常生活并增强整体健康。
- **闹鬼屋名单揭晓**：编制了一份涵盖各地的**闹鬼屋**综合名单，吸引了寻求刺激者和幽灵爱好者的关注。在此探索[惊悚细节](https://www.perplexity.ai/search/list-all-the-haunted-houses-ne-8rNg2NQMQ2e_ghJL1oEdTw)。
  
  - 这可以为那些想要体验惊心动魄之夜的人提供一份精彩指南！

 

**提到的链接**：[YouTube](https://www.youtube.com/embed/sRUpGVJfNJ4)：未找到描述

 

---

### **GPU MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1189498205101109300/1299097328690659349) (3 条消息):

> - `AI in Veterinary Medicine` (兽医学中的 AI)
> - `Community Feedback` (社区反馈)

- **探索兽医学中的 AI**：一位成员询问了 **AI 在兽医学中**具有前景的应用，表现出对创新用途的兴趣。
  
  - 讨论中未提及具体应用，为潜在的技术进步留下了开放的讨论空间。
- **应届毕业生寻求社区见解**：一位成员分享了一个链接，为应届毕业生寻求建议，并用“任何反馈都是好反馈”这句话鼓励大家提供意见。
  
  - 这一行动号召还带有一丝幽默，邀请社区成员根据自己的看法对毕业生进行“吐槽（roast）”。
- **社区氛围与福祉**：另一位成员表达了对社区的赞赏，表示希望大家都能补充水分并度过积极的一天。
  
  - 这条消息强调了服务器内相互支持的氛围，并强化了成员间健康福祉的重要性。

 

---

### **GPU MODE ▷ #**[**triton**](https://discord.com/channels/1189498204333543425/1189607595451895918/1299098460850618460) (61 messages🔥🔥):

> - `Triton Optimizations` (Triton 优化)
> - `BitBLAS Tile Language` (BitBLAS Tile Language)
> - `Mixed Precision Performance` (混合精度性能)
> - `Kernel Performance with Custom Ops` (使用 Custom Ops 的 Kernel 性能)
> - `FA3 Performance Insights` (FA3 性能见解)

- **Triton 优化面临限制**：一位成员分享说，将 kernel 封装在 `custom_op` 中意外地降低了性能，与不使用封装时的 **23 tokens/sec** 相比，仅达到约 **16 tokens/sec**。
  
  - 这种差异引发了对 Triton 包装机制及其在各种配置下对性能影响的担忧。
- **BitBLAS Tile Language 开发**：关于 BitBLAS 新推出的 **Tile Language (tl)** 的讨论表明，该语言旨在提供比 Triton 更好的性能和灵活性，它能同时生成 CUDA C++ 和 PTX 代码。
  
  - 成员们表达了对使用该语言的期待，以及它在支持增强型 kernel 优化技术方面的潜力，特别是针对 AMD 的 HIP。
- **混合精度性能见解**：性能对比显示，BitBLAS 的实现在使用 **fp16** 的低比特矩阵乘法中优于 Triton。
  
  - 尽管性能令人印象深刻，但人们也对在 A100 和 H100 GPU 上达到与 Ada GPU 类似结果的难度表示担忧。
- **使用 Custom Ops 的 Kernel 性能**：一位成员指出，在使用 `custom_op` 时，性能比没有自定义包装的 kernel 调用有所下降，这引发了关于自定义操作带来的开销问题。
  
  - 这一方面突显了在 Triton 中集成自定义操作时潜在的权衡。
- **FA3 性能技巧**：对话涉及了 **FA3** 的高级优化策略，指出尽管存在少量的寄存器溢出（register spill），某些 kernel 仍达到了 state-of-the-art 的性能。
  
  - 许多 FA3 作者与 NVIDIA 的隶属关系表明，这些进步可能深受其硬件专业知识的影响。

**提到的链接**：

- [GitHub - mobiusml/gemlite: Simple and fast low-bit matmul kernels in CUDA / Triton](https://github.com/mobiusml/gemlite/?tab=readme-ov-file#performance)：CUDA / Triton 中简单且快速的低比特 matmul kernel - mobiusml/gemlite
- [BitBLAS/bitblas/ops/general_matmul/tilelang/dequantize/ladder_weight_transform_tensorcore.py at main · microsoft/BitBLAS](https://github.com/microsoft/BitBLAS/blob/main/bitblas/ops/general_matmul/tilelang/dequantize/ladder_weight_transform_tensorcore.py)：BitBLAS 是一个支持混合精度矩阵乘法的库，特别适用于量化 LLM 部署。 - microsoft/BitBLAS
- [Poor performance on Ampere vs. Ada with bitpacked weights · Issue #4906 · triton-lang/triton](https://github.com/triton-lang/triton/issues/4906)：我正在编写一个库，用于在 Triton/CUDA 中执行不同的低比特 matmul kernel。Triton kernel 在 Ada GPU（如 4090 RTX 和 A6000 Ada）上表现出色，在大矩阵上与 Marlin 持平...

---

### **GPU MODE ▷ #**[**cool-links**](https://discord.com/channels/1189498204333543425/1189868872887705671/1299160721791258744) (6 条消息):

> - `Llama 3.2 模型发布`
> - `Mochi 1 预览版`
> - `Cerebras Inference 性能`

- **Llama 3.2 模型发布，面向边缘部署**：Meta 开源了 **Llama 3.2 1B 和 3B** 模型，专为设备端和边缘部署设计，展示了更小的内存占用和更高的性能。
  
  - 开发者正在利用 **QAT** 和 **LoRA** 等量化技术来增强这些模型，从而平衡性能与准确性。
- **Mochi 1 树立视频生成新标准**：Genmo AI 推出了 **Mochi 1 预览版**，这是一款基于 **Apache 2.0** 许可的最先进开源视频生成模型。
  
  - 该发布包含开源权重和模型代码，促进了社区贡献。
- **Cerebras Inference 实现极速处理**：Cerebras 宣布其针对 **Llama 3.1-70B** 的推理速度提升了 **3 倍**，突破了 **2,100 tokens/s**，显著超越了现有的 GPU 解决方案。
  
  - 这种推理速度的飞跃相当于通常与新一代硬件相关的性能提升，目前已在 [Cerebras Inference](http://inference.cerebras.ai) 上线。

**提到的链接**：

- [来自 Cerebras (@CerebrasSystems) 的推文](https://x.com/CerebrasSystems/status/1849467759517896955)：🚨 Cerebras Inference 现在提速 3 倍：Llama3.1-70B 刚刚突破 2,100 tokens/s —— 比最快的 GPU 解决方案快 16 倍 —— 比运行 Llama \*3B\* 的 GPU 快 8 倍 —— 这就像是新一代硬件的性能...
- [来自 Genmo (@genmoai) 的推文](https://x.com/genmoai/status/1848762405779574990)：介绍 Mochi 1 预览版。开源视频生成领域的新 SOTA。Apache 2.0。magnet:?xt=urn:btih:441da1af7a16bcaa4f556964f8028d7113d21cbb&dn=weights&tr=udp://tracker.opentrackr.org:1337/annou...
- [未找到标题](https://ai.meta.com/blog/meta-llama-quantized-lightweight-models/)：未找到描述

---

### **GPU MODE ▷ #**[**torchao**](https://discord.com/channels/1189498204333543425/1205223658021458100/1299168368804237332) (23 messages🔥):

> - `Llama 3.2 开源`
> - `量化感知训练 (QAT)`
> - `QLoRA 对比 QAT`
> - `HQQ+ 概念`
> - `混合精度技术`

- **用于边缘部署的 Llama 3.2 模型**：Meta 开源了 **Llama 3.2** 1B 和 3B 模型，专注于**设备端 (on-device)** 和**边缘部署 (edge deployments)** 以满足社区需求。开发者们正在积极对这些模型进行量化以优化内存占用，尽管这通常意味着要在**性能权衡 (performance tradeoff)** 之间做出取舍。
  
  - 这些量化模型旨在提供更快的推理速度和便携性，同时在资源受限的环境中保持质量。
- **剖析量化感知训练 (QAT)**：QAT 涉及应用**伪量化 (fake quantization)** 过程，在保留高精度的同时模拟量化模型权重的过程。其目标是帮助模型通过训练进行适应，而无需在微调过程中真正将权重转换为低精度。
  
  - 使用这种方法，模型精度在量化后有可能恢复，使其成为对开发者极具吸引力的技术。
- **QLoRA 与 QAT：语义上的区别**：尽管有相似之处，QAT 和 QLoRA 在权重处理方法上有所不同，QAT 在训练期间保持更高的精度。一位参与者指出，将权重合并回原始权重通常会产生更好的结果，这与典型的 QLoRA 流程形成对比。
  
  - 开发者正在探索量化后合并权重的选项，关于操作结果的讨论凸显了其中涉及的复杂性。
- **HQQ+：增强型模型恢复技术**：HQQ+ 方法建议使用 **LoRA 权重**来恢复精度，同时将它们保持在 **FP16** 以提高性能。将这些权重与原始模型合并是首选方案，这利用了量化误差在较低位宽下可能具有更高秩 (rank) 的特性。
  
  - 在权重调整中实现分组大小 (group sizes) 可能会提高模型运行效率，这是一个值得进一步探索的想法。
- **精度策略与操作挑战**：讨论表明，使用**反量化 (dequantization)** 的不同方法会带来不同的结果，一些成员主张保留 FP16 权重以获得更好的效果。有人担心，在无监督设置和微调下仍可能产生带有错误的输出，这需要进一步探索。
  
  - 参与者正在考虑使用奇异值分解 (SVD) 技术来处理 LoRA 权重中的离群值 (outliers)，展示了对现有方法论的创新视角。

**提到的链接**：

- [未找到标题](https://ai.meta.com/blog/meta-llama-quantized-lightweight-models/)：未找到描述
- [使用 PyTorch 进行大语言模型的量化感知训练](https://pytorch.org/blog/quantization-aware-training/)：在这篇博客中，我们介绍了一种用于 PyTorch 中大语言模型的端到端量化感知训练 (QAT) 流程。我们展示了 PyTorch 中的 QAT 如何恢复高达 96% 的精度下降...
- [ao/torchao/quantization/qat at main · pytorch/ao](https://github.com/pytorch/ao/tree/main/torchao/quantization/qat)：用于训练和推理的 PyTorch 原生量化和稀疏化 - pytorch/ao
- [torchtune/recipes/quantization.md at main · pytorch/torchtune](https://github.com/pytorch/torchtune/blob/main/recipes/quantization.md)：PyTorch 原生微调库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。

---

### **GPU MODE ▷ #**[**llmdotc**](https://discord.com/channels/1189498204333543425/1227345713348870156/1299492568597270670) (3 messages):

> - `CUDA 安装问题`
> - `层归一化 (Layer normalization) 错误`

- **重新安装 Ubuntu 并测试 CUDA 设置**：一位成员提到他们重新安装了 **Ubuntu** 并安装了 **CUDA**，通过一个简单的 Kernel 测试确认其可以正常工作。
  
  - 然而，在运行项目时，他们在 **layernorm.cuh** 文件中遇到了与 **floatX** 相关的错误。
- **遇到层归一化错误**：报告的错误表明涉及重载函数 **__ldcs** 和 **__stcs** 的调用失败，这些函数与所使用的参数类型不匹配。
  
  - 该成员表示在测试 **train_gpt2.cu** 后，不确定这些错误的来源。

---

### **GPU MODE ▷ #**[**liger-kernel**](https://discord.com/channels/1189498204333543425/1275130785933951039/1299111257433051241) (5 messages):

> - `NanoGPT model training`
> - `Optimized Triton operations`
> - `Torch compile usage`
> - `Model compatibility`
> - `Performance enhancement functions`

- **优化 NanoGPT 模型训练**：讨论指出，如果目前仅依赖于 eager **PyTorch**，通过使用优化的 **Triton** 操作可以提高 **NanoGPT** 模型的训练速度。
  
  - 实施性能增强的微调可以带来显著的改进。
- **对自定义模型的支持有限**：成员们注意到，目前的模型支持主要针对 **HF compatible models**（如 **Llama** 和 **Qwen**），这意味着自定义实现将需要额外的修改。
  
  - 这需要额外的工作来使现有的框架适配自定义的 **NanoGPT** 模型代码。
- **使用 Torch compile 的益处**：提高训练性能的建议方案之一是启用 **torch.compile**，它可以生成快速的 **Triton** kernel。
  
  - *开启此功能有可能缩短训练任务的处理时间。*
- **关于性能函数的咨询**：有人提出了关于 **Liger Kernel** 中特定函数（如 **RMS norm** 或 **cross entropy loss**）的问题，这些函数可能为他们的使用场景提供显著的速度提升。
  
  - *该咨询强调了评估某些函数对训练效率潜在影响的重要性。*

 

---

### **GPU MODE ▷ #**[**🍿**](https://discord.com/channels/1189498204333543425/1298372518293274644/1299262636252205106) (4 messages):

> - `Discord Cluster Manager Documentation`
> - `Collaboration on Development Timeline`

- **Discord Cluster Manager 文档发布**：一名成员分享了一份[文档](https://docs.google.com/document/d/1SZ3bA9r9yVh27yy9Rbp9-VuAC2nWrUBp7dqetwgZG3M/edit?usp=sharing)，概述了 Discord 集群管理器需要如何运作。
  
  - 该文档作为该项目未来开发的基础指南。
- **计划在截止日期前进行积极开发**：表示计划于 **11 月 3 日** 开始积极开发，目标是在 **11 月 10 日** 前完成。
  
  - 该成员鼓励其他人加入并贡献力量，欢迎对项目提供额外帮助。
- **开始工作的意向**：另一名成员表示打算在回家后审阅分享的文档。
  
  - 他们表达了在审阅后开始项目工作的愿望。

 

**提到的链接**：[Discord Cluster Manager](https://docs.google.com/document/d/1SZ3bA9r9yVh27yy9Rbp9-VuAC2nWrUBp7dqetwgZG3M/edit?usp=sharing)：我们的代码将放在这里 [https://github.com/gpu-mode/discord-cluster-manager](https://github.com/gpu-mode/discord-cluster-manager)。用户体验开发于 11 月 4 日开始，最迟于 11 月 10 日完成功能开发。这项工作我们只需要单个节点。Claud a...

 

---

### **Modular (Mojo 🔥) ▷ #**[**general**](https://discord.com/channels/1087530497313357884/1098713601386233997/1299092270238138409) (59 messages🔥🔥):

> - `Channel for General Questions`
> - `Kitty Ket's LED Matrix Development`
> - `PostgreSQL Library Integration`
> - `Learning Resources for Mojo Language`

- **澄清常规问题频道**：<#1284264544251809853> 不是咨询组织相关问题的正确频道；询问应前往 <#1098713601386233997>。
  
  - 鼓励成员在那里分享问题，以获得更结构化的支持。
- **Kitty Ket 推进 LED 矩阵项目**：Kitty Ket 报告了 LED 矩阵项目的重大进展，通过 **3D vectors** 和 **data manipulation functions** 实现了令人印象深刻的性能指标。
  
  - 尽管尚未实现与 LED 矩阵的通信，但处理时间很短，结果令人期待，目标响应时间在 **10 ms** 以下。
- **Mojo 的 PostgreSQL 库集成**：一名成员询问如何将 PostgreSQL 的 *libpq.so* 集成到 Mojo 中，询问 `ffi.external_call` 是否可以包含自定义库。
  
  - Darkmatter 回复了关于在 C 语言中翻译 `char*` 的歧义，提到它在 x86_64 上通常翻译为 `Int8`，在 ARM 上翻译为 `UInt8`。
- **学习 Mojo 语言的资源**：一位经验丰富的 Python 开发人员请求推荐 Mojo 教程和学习该语言的资源。
  
  - Darkmatter 建议从 [Mojo Manual](https://docs.modular.com/mojo/manual/) 和其他特定的实践学习资源开始。

 

---

### **Modular (Mojo 🔥) ▷ #**[**mojo**](https://discord.com/channels/1087530497313357884/1151418092052815884/1299288062832148503) (1 messages):

> - `Mojo Bug Reports`
> - `Memory Management Issues`

- **Mojo 中报告的内存管理问题**：一名成员提到了一份新提交的关于 **Mojo 内存管理**的 [Bug 报告](https://github.com/modularml/mojo/issues/3710)，强调系统在引用仍在使用时释放了内存。
  
  - 报告指出，用户无法在不被释放的情况下获取 **List** 数据的地址，这在实际使用中造成了严重问题。
- **Mojo 之前类似的修复**：该成员承认之前有另一位用户修复过类似问题，这可能为解决当前的 Bug 提供思路。
  
  - 这一引用表明了在 Mojo 环境中解决内存管理问题的积极态度。

**提及的链接**：[[BUG] Mojo frees memory while reference to it is still in use · Issue #3710 · modularml/mojo](https://github.com/modularml/mojo/issues/3710)：Bug 描述：你无法将 List 数据的地址提取到变量中并随后将其作为引用使用，因为 Mojo 在认为 List 不再需要时会通过解构 List 来释放分配给数据的内存...

---

### **Modular (Mojo 🔥) ▷ #**[**max**](https://discord.com/channels/1087530497313357884/1212827597323509870/1299417158106742886) (1 messages):

> - `Serialized model ingestion`
> - `Graph API use cases`

- **探索序列化模型摄取的使用场景**：一名成员询问了关于摄取使用 **Graph API** 构建的序列化模型的可能**使用场景**，以更好地理解相关需求。
  
  - 该请求旨在从社区收集见解，以便根据实际应用场景定制模型摄取功能。
- **社区参与模型需求讨论**：该成员的询问反映了与社区互动并分享特定模型摄取**需求**的兴趣。
  
  - 这种方法预计将促进一个更具针对性、符合用户需求的开发过程。

---

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1299248147553587222) (33 messages🔥):

> - `Deterministic GPU Kernels`
> - `Floating Point Arithmetic Consistency`
> - `Metal Compiler Optimization`
> - `Tinygrad's Approach to Determinism`
> - `Clang Flags for Metal`

- **探索用于 Metal 的确定性 GPU Kernel**：一名成员询问了如何创建针对 Metal 的**确定性 GPU Kernel**，以在不同的 GPU（如 M2 和 M3）上实现一致的输出。
  
  - 另一名成员分享说，这可能会偏离 tinygrad 主分支，如果成功的话，可能会被视为一个 Fork。
- **MLX 和 Tinygrad 中的浮点运算稳定性**：由于**浮点运算**的非结合性，即使在同一个 GPU 上，MLX 也会出现输出不一致的问题，这引发了人们的担忧。
  
  - 关于 tinygrad 是否可以避免这些不一致性，以及它是否确实被设计为确定性的，存在着争论。
- **Tinygrad 的默认 Metal 配置**：一名成员指出，tinygrad 默认禁用 **Metal 的 Fast Math 模式**，以尽量减少浮点运算中的差异。
  
  - 关于已弃用的 `fastMathEnabled` 选项及其替代方案 `mathMode` 选项的讨论，揭示了提高确定性的潜在改进方向。
- **编译器优化控制的重要性**：会议强调了 Metal 编译器使用 **Clang** 来编译 MSL Shader，允许使用各种 Clang Flags 进行优化控制。
  
  - 成员们建议尝试使用 `-fassociative-math` 等选项来增强确定性，同时先在 Relaxed Math 模式下进行测试。
- **Metal 编译器配置资源**：成员们提供了关于 Clang 选项的资源，强调了 Metal 可用的各种编译器配置，以帮助实现确定性计算。
  
  - 使用 `MTL_COMPILER_FLAGS` 等环境变量可以在不修改 tinygrad 源代码的情况下进行调整。

**提及的链接**：

- [Solving Reproducibility Challenges in Deep Learning and LLMs: Our Journey - HackMD](https://hackmd.io/@Ingonyama/reproducible-ai)：在这篇博客文章中，我们详细介绍了确保深度学习模型始终产生可复现结果的过程。
- [Clang Compiler User’s Manual — Clang 20.0.0git documentation](https://clang.llvm.org/docs/UsersManual.html#cmdoption-ffast-math)：未找到描述内容。

---

### **tinygrad (George Hotz) ▷ #**[**learn-tinygrad**](https://discord.com/channels/1068976834382925865/1070745817025106080/1299128621620789311) (13 messages🔥):

> - `Kernel 空间的 Beam Search`
> - `Action Chunked Transformers`
> - `Notebook 中的环境变量`
> - `Tinygrad 对复数的支持`

- **Kernel 空间的 Beam Search 表现出色**：一位用户对 **Kernel 空间 Beam Search** 的**惊人速度**表示兴奋，并指出虽然它不像 **flash attention** 那么快，但表现依然非常出色。
  
  - 这种性能提升展示了 tinygrad 内部优化的潜力。
- **Action Chunked Transformers 快速训练**：一位成员分享了他们使用最新优化方案在短短**两小时（10,000 步）**内成功训练 **Action Chunked Transformers** 的经验，并在立方体转移（cube transfer）等复杂任务上取得了成果。
  
  - 他们将使用 **tinygrad** 的体验描述为**疯狂（wild）**，反映了其令人印象深刻的能力。
- **Notebook 中环境变量的挑战**：一位用户在 Notebook 中为 **Fashion MNIST** 数据集设置**环境变量**时遇到困难，特别是切换基础 URL 的问题。
  
  - George Hotz 澄清说，如果在导入 tinygrad 之前设置，**os.environ** 是有效的，并且应该使用 **mnist(fashion=True)** 而不是通过环境变量来访问该数据集。
- **Tinygrad 支持复数吗？**：一位用户询问 **tinygrad** 是否可以处理**复数**以执行离散傅里叶变换（DFT），并分享了他们的实现。
  
  - 他们遇到了一个 `AssertionError`，表明目前缺乏支持，另一位成员随后确认 **tinygrad** 不支持复数。

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1299136886744416296) (2 messages):

> - `知识增强型 Agent`
> - `NVIDIA 内部部署`

- **使用 LlamaIndex 构建知识增强型 Agent**：在与 @arizeai 合作的 AI Agent 大师课中，创始人分享了关于使用 LlamaIndex 工作流创建**知识增强型 Agent** 的见解，强调了 **LLM router** 加工具等核心组件。
  
  - 会议讨论了**基于事件和基于图的架构**之间的比较，并对 **LLM router** 表示了偏好，会议内容可以在[这里](https://twitter.com/llama_index/status/1849577709393056081)查看。
- **NVIDIA 内部 AI 助手部署**：公布了一个成功的案例研究，详细介绍了 **NVIDIA 用于销售的内部 AI 助手**，该助手利用 **Llama 3.1 405b 处理简单查询**，并使用 **70b 模型进行文档搜索**。
  
  - 该系统从多个来源检索信息，包括**内部文档**和 **NVIDIA 官网**，更多详情可以在[这里](https://twitter.com/llama_index/status/1849847301680005583)找到。

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1299160188682633287) (33 条消息🔥):

> - `生产环境中的 RAG`
> - `管理文档更新`
> - `LlamaIndex Workflows`
> - `LlamaDeploy 与 LlamaIndex 的兼容性`
> - `NVIDIA 案例研究与 Chainlit 集成`

- **在生产环境中推广 RAG 的挑战**：一位成员表达了在生产环境中说服他人相信 **RAG (Retrieval-Augmented Generation)** 功效的挫败感。
  
  - *It's so hard to make ppl believe in that* 反映了在利益相关者中获得认可的持续困难。
- **文档更新策略**：在生产环境中管理频繁的文档更新被证明具有挑战性，引发了关于在向量数据库中自动化此过程的讨论。
  
  - 建议包括使用 **Qdrant** 进行索引，并设置 cron jobs 以简化更新流程。
- **探索 LlamaIndex Workflows**：出现了关于 **LlamaIndex Workflows** 的界面和 API 访问的问题，并提到了 `llama-deploy` 能够实现相关功能。
  
  - 讨论强调了 Workflows 是事件驱动的抽象，允许用户链接事件并自定义功能。
- **LlamaDeploy 与 LlamaIndex 的兼容性**：成员们确认 **LlamaDeploy** 应该可以与最新版本的 **LlamaIndex Workflow** 配合使用，并保持版本同步。
  
  - 讨论指出，由于其异步设计，在 **LlamaDeploy** 中部署多个 Workflows 可以处理并发请求。
- **对 NVIDIA 案例研究 Cookbook 的需求**：成员们对与新的 NVIDIA 案例研究相关的 Cookbook 以及如何利用 **Chainlit** 实现流式传输表示感兴趣。
  
  - 一位成员在 GitHub 上找到了一个展示在 Chainlit 中使用 **LlamaIndex Workflow** 的示例，引发了对协作资源的期待。

**提到的链接**：

- [Workflows - LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/workflow/#workflows)：未找到描述
- [How to add thread-level persistence to your graph](https://langchain-ai.github.io/langgraph/how-tos/persistence/#add-persistence)：未找到描述
- [Create cookbook for LlamaIndex Workflow abstraction by tituslhy · Pull Request #138 · Chainlit/cookbook](https://github.com/Chainlit/cookbook/pull/138)：此 Cookbook 旨在提供一个如何将 LlamaIndex 最新的 Workflow 抽象与 Chainlit 结合使用的示例。
- [Chat Stores - LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/storing/chat_stores/)：未找到描述
- [Workflows - LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/workflow/)：未找到描述
- [🦙 Llama Deploy 🤖 - LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/llama_deploy/#llama-deploy)：未找到描述

---

### **Cohere ▷ #**[**discussions**](https://discord.com/channels/954421988141711382/954421988783444043/1299085445853876347) (24 条消息🔥):

> - `Cohere Community Coherence` (Cohere 社区的连贯性)
> - `AI Model Advancements` (AI 模型进展)
> - `Song Embedding Recommendations` (歌曲 Embedding 推荐)
> - `Aya vs Command Models` (Aya 与 Command 模型对比)
> - `Upcoming Product Releases` (即将发布的产品)

- **Cohere 社区具有连贯性**：成员们表示 **Cohere 社区** 因保持高质量讨论而脱颖而出，不像其他 AI 社区可能已经失去了写作技巧。
  
  - 一位成员希望能与社区内的潜在合作伙伴建立联系。
- **对 Cohere 研究进展感到兴奋**：成员们对 **Cohere research** 最近的进展充满热情，有人指出其取得了显著进步。
  
  - 另一位成员提到，一些令人兴奋的开发工作已经进行了一段时间，现在正向用户发布。
- **理解歌曲 Embedding 计算**：一位成员询问如何使用歌曲 ID 进行相似度推荐，重点关注用于歌曲推荐的 [Song Embedding Notebook](https://github.com/jalammar/jalammar.github.io/blob/master/notebooks/nlp/02_Song_Embeddings.ipynb)。
  
  - 他们寻求澄清在计算歌曲的 Embedding 时使用的是 **sentence2vector** 还是 **word2vec**。
- **区分 Aya 和 Command 模型**：关于 **Aya** 和 **Command** 模型差异的讨论展开了，Aya 被指出适用于多语言任务，而 Command 更侧重于生产环境。
  
  - 一位成员澄清说，虽然 Aya 模型在各种任务中表现尚可，但它们在多语言应用中表现尤为出色。
- **期待即将发布的产品信息**：成员们对预计在 **11月** 发布的新产品信息表示期待，特别是某些期待已久的功能。
  
  - 随后的交流显示出社区对了解即将推出的功能的渴望。

**提到的链接**：[jalammar.github.io/notebooks/nlp/02_Song_Embeddings.ipynb at master · jalammar/jalammar.github.io](https://github.com/jalammar/jalammar.github.io/blob/master/notebooks/nlp/02_Song_Embeddings.ipynb)：在几分钟内构建 Jekyll 博客，无需接触命令行。- jalammar/jalammar.github.io

---

### **Cohere ▷ #**[**questions**](https://discord.com/channels/954421988141711382/1168411509542637578/1299270809331830807) (7 条消息):

> - `Cohere Sales Contact` (Cohere 销售联系方式)
> - `Song Embedding Recommendations` (歌曲 Embedding 推荐)
> - `Transcribing Calls in Hindi and Telugu` (印地语和泰卢固语的通话转录)

- **联系 Cohere 销售以用于生产环境**：一位用户表示有兴趣在 **生产模式** 下使用 **Cohere**，并询问是否由于时差原因可以通过电子邮件联系销售人员。
  
  - *通过电子邮件联系销售团队是可行的*，一位用户提供了 [Cohere 销售联系页面](https://cohere.com/contact-sales) 的链接。
- **理解歌曲 Embedding**：另一位用户询问了歌曲 Embedding Notebook 在推荐方面的功能，询问如何为不同的歌曲 ID 计算 Embedding。
  
  - 他们寻求澄清在播放列表场景下使用的是 **sentence2vector** 还是 **word2vec**。
- **关于印地语和泰卢固语的 ASR 模型问题**：一位对 **印地语和泰卢固语通话转录** 感兴趣的用户注意到 **aya expanse Hugging Face space** 中有一个音频输入。
  
  - 他们想知道这是否是 **aya-expanse-32B** 的一部分，后来发现实际上是使用了 *groq_whisper* 进行 ASR。

**提到的链接**：

- [Aya Expanse - a Hugging Face Space by CohereForAI](https://huggingface.co/spaces/CohereForAI/aya_expanse)：未找到描述
- [Contact Sales](https://cohere.com/contact-sales)：无论您是超出了我们的自助 API 服务范围，还是有定制的安全或托管需求，请与我们联系以讨论您的具体需求。我们在这里协助：云端和私有部署...
- [jalammar.github.io/notebooks/nlp/02_Song_Embeddings.ipynb at master · jalammar/jalammar.github.io](https://github.com/jalammar/jalammar.github.io/blob/master/notebooks/nlp/02_Song_Embeddings.ipynb)：在几分钟内构建 Jekyll 博客，无需接触命令行。- jalammar/jalammar.github.io

---

### **Cohere ▷ #**[**api-discussions**](https://discord.com/channels/954421988141711382/1168578329423642786/1299121099493347542) (3 messages):

> - `JSON Argument Formatting`
> - `Function Tool Calls`

- **修复奇怪的 JSON 参数 Bug**：一位成员发现 JSON 函数调用中的参数格式化问题，其中使用了单引号而非双引号：`{'order_id': 'order_12345'}` 而不是 `{"order_id": "order_12345"}`。
  
  - 他们表达了沮丧，称这是一个“在我看来不该存在的奇怪 Bug”。
- **关于 JSON 转义的说明**：另一位成员澄清说该问题并非 Bug，强调在提供 JSON 时应始终进行正确的转义。
  
  - 他们指出一个如何正确转义 JSON 的示例：`"{"order_id": "order_12345"}"`。

---

### **OpenInterpreter ▷ #**[**general**](https://discord.com/channels/1146610656779440188/1147665339266650133/1299124471998713890) (21 messages🔥):

> - `Interpreter fixes`
> - `Rate limits with Claude`
> - `Custom OpenAI API agent setup`
> - `YouTube test on Open Interpreter performance`
> - `Error fixing in OS mode`

- **Interpreter 修复补丁刚刚发布**：一位成员宣布 `interpreter --os` 的一系列修复现已在 pip 上可用，并邀请用户在发布 **voice mode** 之前协助测试更多细节。
  
  - 他们对改善此前遇到困难的用户体验表示乐观。
- **对 Claude 速率限制的沮丧**：多位成员表达了在使用 **Claude** 时遇到 Rate limits 的困扰，表示这阻碍了他们的工作流。
  
  - 一位用户幽默地提到 Rate limit 真的让他很恼火，其他成员也对这种中断表达了类似感受。
- **设置自定义 OpenAI API Agent**：讨论了在受限于 **Claude** 时是否可以使用自定义 OpenAI API Agent。
  
  - 另一位成员分享了设置自定义模型的文档，以协助完成配置。
- **运行 Open Interpreter YouTube 测试时遇到挑战**：一位成员询问如何使用本地模型运行标题为“使用电子表格提高 Open Interpreter 性能”的 **YouTube 视频**中的测试。
  
  - 他们报告了 **qwen2.5:32b-instruct** 的问题，指出该模型无法遵循视频中概述的步骤。
- **修复 OS 模式下的错误**：在遇到 'ValueError: Invalid format string' 错误后，一位用户通过将 **loop.py** 中的 **strftime** 实例更改为更通用的日期格式提供了修复方案。
  
  - 另一位成员幽默地提到，他们在处理类似问题时向 ChatGPT 寻求了修复方案，强调了调试过程中的试错本质。

**提到的链接**：

- [no title found](https://docs.openinterpreter.com/language-models/custom-models)：未找到描述
- [Improve Open Interpreter Performance with Spreadsheets](https://www.youtube.com/watch?v=4X4rKqtmxJg)：你的 AI Agent 是否在多步骤任务中失去焦点？这里有一种方法可以帮助 Open Interpreter 保持在正轨上，此外它还能加速 Prompt Engineering 迭代...

---

### **OpenInterpreter ▷ #**[**ai-content**](https://discord.com/channels/1146610656779440188/1149229778138824765/1299387466792566865) (1 messages):

> - `Clevrr-Computer`
> - `Chrome Built-in AI`

- **Clevrr-Computer 赋能 AI 生产力**：[Clevrr-Computer](https://github.com/Clevrr-AI/Clevrr-Computer?ref=producthunt) 项目提供了 **Anthropic's Computer** 的开源实现，可通过 AI Agent 执行基础任务。
  
  - 该项目因其在增强生产力和自动化各种应用任务方面的潜力而受到关注。
- **探索 Chrome 内置 AI 功能**：分享了 [Chrome's Built-in AI](https://developer.chrome.com/docs/ai/built-in) 资源的链接，展示了在 Web 活动中集成 AI 的创新方式。
  
  - 这些功能旨在提升用户体验，并促进用户直接在浏览器中使用先进的 AI 工具。

**提到的链接**：

- [no title found](https://developer.chrome.com/docs/ai/built-in)：未找到描述
- [GitHub - Clevrr-AI/Clevrr-Computer at producthunt](https://github.com/Clevrr-AI/Clevrr-Computer?ref=producthunt)：Anthropic's Computer Use 的开源实现，用于使用 AI Agent 执行基础任务。- GitHub - Clevrr-AI/Clevrr-Computer at producthunt

---

### **LAION ▷ #**[**general**](https://discord.com/channels/823813159592001537/823813160075132991/1299154769151266887) (10 messages🔥):

> - `Video Classification Model Training`
> - `DataLoader Performance`
> - `Disk IO Monitoring`
> - `Model Size vs. Batch Size`
> - `Speech Generation Benchmarks`

- **训练视频模型的瓶颈**：一位用户报告称，在 8 个 GPU 上训练其视频分类模型时，受到 DataLoader 的严重限制，Batch 之间出现了明显的停顿。
  
  - 数据集由总计约 **7M 帧**的 **MP4** 文件组成，但如果转换为 JPEG，数据集大小将膨胀至 **1TB**，并引发性能问题。
- **DataLoader 优化建议**：社区成员建议通过记录数据获取和 GPU 处理的时间来监控和优化 DataLoader，以识别瓶颈。
  
  - 建议使用适当的 **prefetching**（预取）可以帮助减少当 GPU 处理 Batch 的速度远快于数据加载速度时产生的瓶颈。
- **磁盘 IO 对训练速度的影响**：讨论中提出了关于用户的磁盘配置（SSD vs HDD）是否可能成为**读取速度或 IOPS 瓶颈**的担忧。
  
  - 在模型训练的背景下，监控磁盘 IO 有助于诊断读取速度是否影响了 DataLoader 的性能。
- **模型大小对训练速度的影响**：用户分享说他们正在训练一个仅有 **50M 参数**的小模型，由于较大的 Batch Size 导致了加载延迟。
  
  - 有人指出，这种小模型尺寸不足以进行视频处理，建议需要更大的模型以获得更好的数据加载速度。
- **对语音生成基准测试的兴趣**：一位新用户加入对话，询问在 **3090 GPU** 上进行**语音生成的延迟基准测试**，但未发现相关信息。
  
  - 他们对现有仓库信息中缺失的实时生成速度指标表示感兴趣。

 

---

### **LAION ▷ #**[**resources**](https://discord.com/channels/823813159592001537/991938328763056168/1299411203931770991) (2 messages):

> - `Best Practices for Building LLM Applications`

- **关于 LLM 见解的新 YouTube 网络研讨会**：一段名为《构建成功 LLM 应用的最佳实践》的 [YouTube 视频](https://youtu.be/JVNwAn7bTqY?si=ifDjr3MhKUXlLnTN) 现已上线，仅 **1 天**内播放量已接近 **1000 次**。
  
  - 本次会议由来自 Meta 的 **Senior ML Engineer** 主讲，分享了关于开发具有影响力的 LLM 解决方案的实用见解。
- **探索有影响力的 LLM 解决方案**：该网络研讨会将涵盖构建成功 LLM 应用的核心要素，由实际经验和实用知识驱动。
  
  - 参与者可以期待获得关于 LLM 实现和性能优化的宝贵建议，鼓励动手实践学习。

 

**提到的链接**：[Best Practices for Building Successful LLM Applications | Datahour by Bhavul Gauri](https://youtu.be/JVNwAn7bTqY?si=ifDjr3MhKUXlLnTN)：加入这场由 Meta 高级 ML 工程师主讲的网络研讨会，他将分享构建有影响力的 LLM 解决方案的实用见解。本节课将探索……

 

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**axolotl-phorm-bot**](https://discord.com/channels/1104757954588196865/1225558824501510164/1299415866584334386) (5 messages):

> - `DPO Evaluations`
> - `Axolotl Codebase`
> - `Evaluation Metrics`
> - `Model Predictions`
> - `Training Callbacks`

- **是的，DPO 评估是可行的**：你可以使用 Axolotl 代码库对 **Direct Preference Optimization (DPO)** 进行评估，这涉及加载数据集并将预测结果与 Ground Truth 进行比较。
  
  - 评估过程始于使用 `load_prepare_dpo_datasets` 函数加载你的评估数据集。
- **为评估准备模型**：在生成预测之前，通过执行 `model.eval()` 确保你的 DPO 模型处于评估模式，这一点非常重要。
  
  - 这会禁用 Dropout 等可能影响评估完整性的特性。
- **从数据集中生成预测**：可以通过遍历评估数据集并在 Torch 的 `no_grad` 上下文中收集输出来生成预测，以节省显存。
  
  - 这种方法确保了预测的高效性，并且不会追踪梯度（Gradients）。
- **计算评估指标**：在生成预测后，可以使用 scikit-learn 等库计算各种指标，如准确率（Accuracy）或 F1 分数。
  
  - 例如，你可以通过使用 `accuracy_score` 比较预测标签与真实标签来计算准确率。
- **将回调集成到训练中**：你可以使用 `BenchEvalCallback` 等回调（Callbacks）将评估集成到训练过程中，以按设定的间隔进行评估。
  
  - 这些回调允许将评估指标无缝集成到你的训练流程中。

 

**提到的链接**：[OpenAccess-AI-Collective/axolotl | Phorm AI Code Search](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=4a1f414b-437b-4230-b5cd-3436c848589d)：更快速地理解代码。

 

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-questions**](https://discord.com/channels/1179127597926469703/1179208129083363358/1299152706736226394) (3 messages):

> - `mid training inclusion`
> - `specialized training`

- **关于中期训练（Mid Training）内容的投票**：一位成员发起了一场关于人们认为 **mid training** 中包含什么的讨论，引发了对所涉及定义和流程的兴趣。
  
  - “除了 RLHF 之外，针对某些数据进行的所有专项训练（Specialized Training），”另一位成员评论道，为进一步探索拉开了序幕。
- **编程领域的特定 Epoch 训练**：在对话中，一位成员强调 mid training 可能涉及专门针对编程（Coding）进行 **1-2 个 Epoch** 的训练。
  
  - 这一观察旨在澄清各种训练方法论及其影响之间的区别。

 

---

### **Interconnects (Nathan Lambert) ▷ #**[**random**](https://discord.com/channels/1179127597926469703/1183121795247779910/) (1 messages):

xeophon.: 只有当他们在历史邮件中注入多样性时才会。

---

### **Interconnects (Nathan Lambert) ▷ #**[**memes**](https://discord.com/channels/1179127597926469703/1187551504995987576/) (1 messages):

xeophon.: [https://x.com/andrewwhite01/status/1849710726631522574](https://x.com/andrewwhite01/status/1849710726631522574)

---

### **LangChain AI ▷ #**[**general**](https://discord.com/channels/1038097195422978059/1038097196224086148/1299118815828906026) (3 messages):

> - `Dataset evaluations for PDF data`
> - `Seeking AI developer`

- **评估来自 PDF 文件的数据集**：一位成员询问了如何专门针对 **PDF 数据** 进行评估和管理数据集，并提到他们有一个想要运行评估的 PDF 文件。
  
  - 这引发了关于在 PDF 等非结构化数据格式上处理结构化评估的方法论问题。
- **AI 开发者的工作机会**：一位成员正在寻找一位**资深开发者**，最好是需要工作的 **AI 高手（AI wizard）**。
  
  - 这引发了一个后续问题，询问有哪些可能利用此类人才的项目想法。

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-questions**](https://discord.com/channels/1280234300012494859/1280370030609170494/1299107086957019159) (3 messages):

> - `Email Timestamp Issue`
> - `Email Confirmation`

- **表单邮件的时间戳澄清**：一位成员澄清了**表单邮件的时间戳**是 **Sep 28, 6:50 PM PST**，并提到他们电子邮件的首字母是小写字母 'l'。
  
  - 这是在解决与邮件提交相关问题的背景下提到的。
- **邮件混淆问题的解决**：另一位成员确认他们找到了该邮件，表明他们理解了情况，并对**后续的解决**表示乐观。
  
  - 他们添加了一个大拇指表情，表示对邮件问题得到解决持积极态度。

 

---

### **DSPy ▷ #**[**show-and-tell**](https://discord.com/channels/1161519468141355160/1202371242519441499/1299471550784733234) (1 条消息):

> - `MIPROv2`
> - `Prompt Generation`
> - `GSM8K Dataset`

- **用于自动化 Prompt 生成的 MIPROv2 技术**：一位成员分享了一个关于使用 **MIPROv2 optimizer** 技术实现“自动 Prompt 生成”的[简短推文串](https://x.com/karthikkalyan90/status/1849902254373077196)。
  
  - 该实现利用 **GSM8K 数据集**，包含三个模块：一个用于生成 demo，另一个用于生成指令，最后一个模块将这些输出编译成完整的 Prompt。
- **Prompt 创建的三模块方法**：该程序由三个模块组成，以简化生成过程：**Module 1** 生成 demo，**Module 2** 创建指令，**Module 3** 将输出编译为最终的 Prompt。
  
  - 这种方法旨在通过系统化的结构增强 Prompt 生成的效果。

**提到的链接**：[Karthik Kalyanaraman (@karthikkalyan90) 的推文](https://x.com/karthikkalyan90/status/1849902254373077196)：🧵一个使用 MIPROv2 optimizer 技术的“自动 Prompt 生成”简化实现。该程序使用包含数学问题的 GSM8K 数据集，由……组成。

---

### **DSPy ▷ #**[**general**](https://discord.com/channels/1161519468141355160/1161519469319946286/) (1 条消息):

._j_s: 你有机会了吗？

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**general**](https://discord.com/channels/1238365980128706560/1238365980128706563/) (1 条消息):

.edgarbc: 非常感谢 c123ian！我会去查看那些内容的。

---

### **Torchtune ▷ #**[**general**](https://discord.com/channels/1216353675241590815/1216353675744641096/1299103987655643147) (1 条消息):

> - `Torchtune Issues`
> - `Community Contributions`

- **Torchtune GitHub 上的新 Issue**：[Torchtune GitHub](https://github.com/pytorch/torchtune/issues/1901) 上创建了一个关于各种所需增强和修复的新 Issue。
  
  - 尽管已标记为需要社区帮助，但鼓励有兴趣的成员参与并贡献。
- **呼吁社区支持**：该 Issue 虽然没有标记为 "community help wanted"，但开放贡献。
  
  - 成员们表达了对该项目协作的兴趣。

**提到的链接**：[Issues · pytorch/torchtune](https://github.com/pytorch/torchtune/issues/1901.)：PyTorch 原生微调库。通过在 GitHub 上创建账号来为 pytorch/torchtune 的开发做出贡献。

---

### **Mozilla AI ▷ #**[**announcements**](https://discord.com/channels/1089876418936180786/1089876419926032396/1299472683213520960) (1 条消息):

> - `AI content compensation`
> - `Creative works licensing`
> - `Mozilla's Data Futures Lab`

- **AI 创作者寻求公平补偿**：互联网上的创作者正面临着内容在未经许可或未获补偿的情况下被用于训练 AI 系统的危机，这凸显了对赋能系统的迫切需求。
  
  - 一个允许个人为 AI 训练授权其内容的平台正在兴起，这有望为内容创作者带来潜在收益。
- **Human Native AI 推出数据市场**：联合创始人 James Smith 宣布 [Human Native AI](https://www.humannative.ai/) 正在开发一个 AI 数据市场，创作者可以在其中汇集作品并因训练 AI 系统而获得补偿。
  
  - 该倡议旨在提高数据使用的公平性，解决内容创作者提出的担忧。
- **Mozilla's Data Futures Lab 演讲系列**：由 James Smith 主讲的讲座是 **Mozilla's Data Futures Lab Speaker Series** 的一部分，旨在探索生成式 AI 时代的公平数据生态系统。
  
  - 鼓励参与者预约参加这一富有洞察力的活动，参与围绕数据和 AI 未来的讨论。

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #**[**discussion**](https://discord.com/channels/1111172801899012102/1111353033352294440/) (1 条消息):

honolouloute: 发现得好

---

---

---

---

{% else %}

> 完整的逐频道细分内容已在邮件中截断。
> 
> 如果您想查看完整细分，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
> 
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}