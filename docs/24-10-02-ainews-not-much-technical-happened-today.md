---
companies:
- openai
- poolside
- liquidai
- perplexity-ai
- meta-ai-fair
- cohere
- fujitsu
date: '2024-10-02T22:45:37.315067Z'
description: '**OpenAI** 宣布以 **1570 亿美元**的估值筹集了 **66 亿美元**的新资金，同时 ChatGPT 的**周活跃用户数已达到
  2.5 亿**。**Poolside** 筹集了 **5 亿美元**以推进通用人工智能（AGI）的开发。**LiquidAI** 推出了三款新的混合专家（MoE）模型（1B、3B、40B），具备
  **32k 上下文窗口**和高效的 Token 处理能力。**OpenAI** 发布了 Whisper V3 Turbo，这是一款在速度上有显著提升的开源多语言模型。**Meta
  AI FAIR** 正在招聘研究实习生，重点关注 **LLM 推理、对齐、合成数据和新型架构**。**Cohere** 与富士通（Fujitsu）合作推出了定制日语模型
  Takane。技术讨论包括 **LoRA 微调**中的挑战、Keras 中的 **float8 量化**，以及用于智能体模板的新工具（如 **create-llama**）。行业评论对
  AI 发展的优先级表示担忧，并强调了 AI 领域的自由职业机会。'
id: 8f517447-539c-4067-b201-78920325a6c3
models:
- whisper-v3-turbo
- llama-3
- llamaindex
original_slug: ainews-not-much-technical-happened-today
people:
- nick-turley
- arav-srinivas
- francois-fleuret
- finbarr-timbers
- lewtun
- francois-chollet
- jerry-j-liu
- mmitchell-ai
- jxnlco
title: 今天技术方面没发生什么大事。
topics:
- mixture-of-experts
- context-windows
- model-optimization
- fine-tuning
- quantization
- model-training
- alignment
- synthetic-data
- model-architecture
- agentic-ai
---

<!-- buttondown-editor-mode: plaintext -->**融资就是你所需要的一切。**

> 2024年10月1日至10月2日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitters](https://twitter.com/i/lists/1585430245762441216) 和 **31** 个 Discords（**225** 个频道和 **1832** 条消息）。预计节省阅读时间（按每分钟 200 字计算）：**219 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

今天 [OpenAI 宣布](https://openai.com/index/scale-the-benefits-of-ai/)以 1570 亿美金的估值筹集了 66 亿美金的新融资。在 Twitter 上，ChatGPT 产品负责人 [Nick Turley 还补充道](https://x.com/nickaturley/status/1841580683359354890) 
“周活跃用户达到 2.5 亿，高于约一个月前的 2 亿”。

![image.png](https://assets.buttondown.email/images/d885ce79-426b-4a2d-82ac-f5cb75314c98.png?w=960&fit=max)


同样在融资新闻中，[Poolside 宣布](https://poolside.ai/checkpoint/announcing-our-500-million-fundraise-to-make-progress-towards-agi)融资 5 亿美金，以推动 AGI 的进程。

![image.png](https://assets.buttondown.email/images/29792548-8037-49cb-8ce3-030efe917c16.png?w=960&fit=max)

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

**AI 模型进展与行业动态**

- **新 AI 模型与能力**：[@LiquidAI_](https://twitter.com/LiquidAI_/status/1840897331773755476) 发布了三个新模型：1B、3B 和 40B MoE（12B 激活），采用了自定义的 Liquid Foundation Models (LFMs) 架构，在**基准测试中表现优于 Transformer 模型**。这些模型拥有 **32k 上下文窗口**和极小的内存占用，能够高效处理 1M token。[@perplexity_ai](https://twitter.com/perplexity_ai/status/1840890047689867449) 预告了即将推出的功能 "⌘ + ⇧ + P — coming soon"，暗示其 AI 平台将迎来新功能。

- **开源与模型发布**：[@basetenco](https://twitter.com/basetenco/status/1840883111162155138) 报道称 OpenAI 发布了 Whisper V3 Turbo，这是一个开源模型，其**相对速度比 Whisper Large 快 8 倍**，**比 Medium 快 4 倍**，**比 Small 快 2 倍**，拥有 809M 参数并支持全多语言。[@jaseweston](https://twitter.com/jaseweston/status/1840864799942439336) 宣布 FAIR 正在招聘 2025 年研究实习生，重点关注 **LLM 推理、对齐 (alignment)、合成数据和新颖架构**等主题。

- **行业合作伙伴与产品**：[@cohere](https://twitter.com/cohere/status/1840804482449621308) 推出了 Takane，这是与 Fujitsu Global 合作开发的业界领先的定制日语模型。[@AravSrinivas](https://twitter.com/AravSrinivas/status/1840892055406723474) 预告了某款 AI 产品即将推出 Mac 应用，预示着 AI 工具正在向桌面平台扩展。

**AI 研究与技术讨论**

- **模型训练与优化**：[@francoisfleuret](https://twitter.com/francoisfleuret/status/1840864960957579555) 对使用 10,000 块 H100 训练单个模型表示不确定，强调了大规模 AI 训练的复杂性。[@finbarrtimbers](https://twitter.com/finbarrtimbers/status/1840883655255998519) 对 1B 模型表现变好所带来的 **推理时搜索 (inference time search)** 潜力感到兴奋，这暗示了条件计算 (conditional compute) 的新可能性。

- **技术挑战**：[@_lewtun](https://twitter.com/_lewtun/status/1840804557800292843) 强调了 LoRA 微调和聊天模板的一个关键问题，强调需要**将 embedding 层和 LM head 包含在可训练参数中**，以避免输出乱码。这适用于使用 ChatML 和 Llama 3 聊天模板训练的模型。

- **AI 工具与框架**：[@fchollet](https://twitter.com/fchollet/status/1840904343882776778) 分享了如何使用 `.quantize(policy)` 在 Keras 模型上启用 float8 训练或推理，展示了该框架对各种量化形式的灵活性。[@jerryjliu0](https://twitter.com/jerryjliu0/status/1840889451926765989) 介绍了 create-llama，这是一个可以快速生成由 Python 和 TypeScript 中的 LlamaIndex 工作流驱动的完整 Agent 模板的工具。

**AI 行业趋势与评论**

- **AI 发展类比**：[@mmitchell_ai](https://twitter.com/mmitchell_ai/status/1840853482385129902) 分享了对科技行业 AI 进步方式的批评，将其比作一个目标是寻找逃生口而非造福社会的电子游戏。这一观点突显了对 AI 发展方向的担忧。

- **AI 自由职业机会**：[@jxnlco](https://twitter.com/jxnlco/status/1840860366038839804) 概述了自由职业者在 AI 淘金热中注定会大获全胜的原因，理由包括高需求、AI 系统的复杂性以及解决各行业实际问题的机会。

- **AI 产品发布**：[@swyx](https://twitter.com/swyx/status/1840867798308045219) 将 Google DeepMind 的 NotebookLM 与 ChatGPT 进行了比较，指出了其**多模态 RAG 能力**以及 LLM 在产品功能中的原生集成。这突显了 AI 驱动的生产力工具中持续的竞争与创新。

**梗与幽默**

- [@bindureddy](https://twitter.com/bindureddy/status/1840869990612025789) 幽默地评论了 Sam Altman 关于 AI 模型的言论，指出了一种批评当前模型同时炒作未来模型的模式。

- [@svpino](https://twitter.com/svpino/status/1840889043976143250) 开玩笑说只需每月 2 美元就能托管年收入 110 万美元的网站，强调了网站托管的低成本，并嘲讽了过度复杂的解决方案。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. OpenAI 的 Whisper Turbo：浏览器端语音识别的突破**

- **[令人抓狂的 Whisper 版本混乱](https://reddit.com//r/LocalLLaMA/comments/1ftlz6a/the_insanity_of_whisper_versions/)**（[得分：30，评论：14](https://reddit.com//r/LocalLLaMA/comments/1ftlz6a/the_insanity_of_whisper_versions/)）：该帖子讨论了 **Whisper 的众多版本**，包括尺寸变体（**base, small, tiny, large, turbo**）、版本迭代（**v1, v2, v3**）、特定语言模型（**English-only**）以及专注于性能的变体（**faster whisper, insanely-fast whisper**）。作者寻求关于**选择合适的 Whisper 模型**的指导，并考虑了 **GPU 性能**和**语言需求**等因素，特别提到了用于英语的 **medium.en** 以及可能用于外语转录/翻译的更大版本。
  - **Whisper-ctranslate2**（基于 faster-whisper）被推荐为最快的选择，而非英语用途则建议使用 **large 模型**。**版本比较**显示 v2 和 v3 的表现优于 v1，而 v3 的性能在不同语言上有所差异。
  - large Whisper 模型的**硬件要求**包括 **6GB VRAM**（最低），CPU 推理速度约为 **0.2-0.5 倍实时速度**。有用户报告 **WhisperX** 在 8GB fp32 GPU 上会崩溃，而 fp16 在较低 VRAM 占用下表现更好。
  - 现有的 Whisper 模型性能基准测试包括 [FP16 基准测试](https://github.com/openai/whisper/discussions/918) 和 [large v3 基准测试](https://blog.salad.com/whisper-large-v3/)。对于快速 CPU 使用场景，建议使用 **whisperfile**（Whisper 的 llamafile 封装版本）等替代方案。


- **[OpenAI 的新 Whisper Turbo 模型通过 Transformers.js 在浏览器中 100% 本地运行](https://v.redd.it/5a7eo6vat4sd1)**（[得分：456，评论：52](https://reddit.com//r/LocalLLaMA/comments/1ftlznt/openais_new_whisper_turbo_model_running_100/)）：**OpenAI 的 Whisper Turbo** 模型现在可以使用 **Transformers.js** 在 **Web 浏览器中 100% 本地运行**，从而实现无需将数据发送到外部服务器的**语音转文字转录**。该实现利用 **WebGPU** 进行加速处理，在兼容设备上达到**实时转录速度**，并提供 **WebGL** 回退方案以实现更广泛的兼容性。
  - **Whisper large-v3-turbo** 模型可达到 **~10x RTF**（实时因子），在 **M3 Max** 上仅需 **~12 秒**即可转录 **120 秒**的音频。它是 Whisper large-v3 的蒸馏版本，将解码层从 **32 层减少到 4 层**，在保持较小质量损失的同时实现了更快的处理速度。
  - 该模型使用 **Transformers.js** 和 **WebGPU** 在浏览器中 **100% 本地运行**，无需访问 OpenAI 服务器。这个 **800MB 的模型**被下载并存储在浏览器的缓存存储中，通过 service workers 实现离线使用。
  - 用户讨论了该模型的**多语言能力**和潜在的准确度变化。该模型的**实时版本**已在 [Hugging Face](https://huggingface.co/spaces/kirill578/realtime-whisper-v3-turbo-webgpu) 上线，也可以通过 ggerganov 的 **whisper.cpp** 离线使用。


- **[Transformers 现已支持 Whisper Turbo 🔥](https://reddit.com//r/LocalLLaMA/comments/1ftjqg9/whisper_turbo_now_supported_in_transformers/)**（[得分：174，评论：33](https://reddit.com//r/LocalLLaMA/comments/1ftjqg9/whisper_turbo_now_supported_in_transformers/)）：**Hugging Face 的开源音频团队**发布了 **Transformers 格式**的 **Whisper Turbo**，其特点是一个拥有 **8.09 亿参数的模型**，比 **Large v3** 快 **8 倍**且体积小 **2 倍**。该**多语言模型**支持**时间戳**，并使用 **4 个解码层**而非 32 层。在 Transformers 中的实现仅需极少代码即可使用 [ylacombe/whisper-large-v3-turbo](https://huggingface.co/ylacombe/whisper-large-v3-turbo) 权重完成自动语音识别任务。
  - 讨论了 **Whisper Turbo** 的性能，用户将其与 **faster-whisper** 和 **Nvidia Canary** 进行了比较。后者被指出位列 **Open ASR 排行榜**榜首，但支持的语言较少。
  - **GGUF 支持**在请求发出后几小时内便迅速实现了，开发者提供了 [GitHub pull request](https://github.com/ggerganov/whisper.cpp/pull/2440/files#diff-433d68c356c0513e785d8d462b4df9f57df61c8ac3eab291f843567aedf0a692) 和 [模型权重 (checkpoints)](https://huggingface.co/ggerganov/whisper.cpp/tree/main) 的链接。
  - 用户确认了 **Whisper Turbo 与 Mac M 系列芯片的兼容性**，并提供了在 **MPS** 上运行的代码修改建议。一名用户报告在 **4090 GPU** 上达到了 **820 倍实时速度**且没有性能损失。


**主题 2：当前 LLM 架构的收敛与局限性**

- **所有 LLM 都在向同一点收敛** ([Score: 108, Comments: 57](https://reddit.com//r/LocalLLaMA/comments/1ftn6s1/all_llms_are_converging_towards_the_same_point/)): 包括 **Gemini**、**GPT-4**、**GPT-4o**、**Llama 405B**、**MistralLarge**、**CommandR** 和 **DeepSeek 2.5** 在内的多种**大语言模型** (LLMs) 被用于生成一个包含 **100 个项目**的列表，结果前六个模型生成的数据库和分组几乎完全相同。作者观察到，尽管这些模型在“废话”或无关文本上有所不同，但其主要数据输出呈现出**收敛**趋势，从而得出结论：这些 LLM 正在趋向于一个共同点，而这并不一定预示着**人工超级智能** (ASI) 的到来。
  - **ArsNeph** 认为 LLM 的收敛是由于过度依赖来自 **GPT 家族**的**合成数据**，导致了广泛的“**GPT 废话** (GPT slop)”和原创性的缺乏。**开源微调版本**以及像 **Llama 2** 这样的模型本质上是 GPT 的蒸馏版本，而像 **Llama 3** 和 **Gemma 2** 这样较新的模型则使用 **DPO** 来使其表现得更讨人喜欢。
  - 用户讨论了解决 **LLM 收敛**的潜在方案，包括尝试不同的**采样器** (samplers) 和 **Tokenization** 方法。针对 **exllamav2** 的 **XTC 采样器**被提及为一种减少重复输出的有前景的方法，一些用户渴望在 **llama.cpp** 中实现它。
  - 讨论还涉及了 **Claudisms**，这是一种 **Claude** 表现出其自身并行版本的 **GPTisms** 的现象，可能是一种**指纹识别** (fingerprinting) 形式。一些人推测，这些模式可能是用于识别特定模型生成文本的人工痕迹，即使其他模型是在这些数据上训练的。


- **[48GB 显存的最佳模型](https://i.redd.it/c9zyp873d9sd1.jpeg)** ([Score: 225, Comments: 67](https://reddit.com//r/LocalLLaMA/comments/1fu6far/best_models_for_48gb_of_vram/)): 一位拥有配备 **48GB 显存**的新 **RTX A6000 GPU** 的用户正在寻求在该硬件上运行的最佳模型建议。他们特别要求能够以至少 **Q4 量化**或 **4 bits per weight (4bpw)** 运行的模型，以优化其大容量 GPU 的性能。
  - 用户建议运行 **70B 模型**，如 **Llama 3.1 70B** 或 **Qwen2.5 72B**。性能基准测试显示，在两块 RTX 3090 GPU 上，**Qwen2.5 72B** 使用 q4_0 量化可达到 **12-13 tokens/second**，使用 q4_K_S 量化可达到 **8.5 tokens/second**。
  - 建议使用带有 **TabbyAPI** 的 **ExllamaV2** 以获得更快的速度，**Mistral Large** 在 **3 bits per weight** 下可能达到 **15 tokens/second**。一位用户报告称，在 Linux 上使用带有张量并行 (tensor parallelism) 和投机解码 (speculative decoding) 的 **Qwen 2 72B** 处理编程任务时，速度高达 **37.31 tokens/second**。
  - 一些用户建议尝试 **3 bits per weight** 的 **Mistral-Large-Instruct-2407**（这是一个 **120B 参数**模型），而其他人则认为 **Qwen 72B** 是“最聪明”的 70B 级别模型。此外还讨论了 **RTX A6000** 的散热解决方案，一位用户展示了在 **RM44 机箱**中使用 **Silverstone FHS 120X** 风扇的配置。


- **在低资源边缘设备上高效运行 70B 规模的 LLM** ([Score: 53, Comments: 17](https://reddit.com//r/LocalLLaMA/comments/1fu8ujh/serving_70bscale_llms_efficiently_on_lowresource/)): 该论文介绍了 **TPI-LLM**，这是一个**张量并行推理系统**，旨在低资源边缘设备上运行 **70B 规模的语言模型**，通过将敏感数据保留在本地来解决隐私问题。TPI-LLM 实现了**滑动窗口内存调度器**和**基于星型的 AllReduce 算法**，分别用于克服内存限制和通信瓶颈。实验表明，与 Accelerate 相比，TPI-LLM 的首字延迟 (time-to-first-token) 和 Token 延迟降低了 **80% 以上**；与 Transformers 和 Galaxy 相比，降低了 **90% 以上**。同时，它将 **Llama 2-70B** 的峰值内存占用减少了 **90%**，运行 70B 规模模型仅需 **3.1 GB 内存**。
  - **TPI-LLM** 通过**张量并行**利用**多个边缘设备**进行推理，在 **8 台**各拥有 **3GB** 内存的设备上运行 **Llama 2-70B**。这种分布式方法实现了显著的内存缩减，但代价是速度上的权衡。
  - 该系统的性能受限于**磁盘 I/O**，导致 **70B 模型**的首字延迟为 **29.4 秒**，平均吞吐量为 **26.1 秒/token**。尽管存在这些延迟，该方法在低资源设备上运行大语言模型方面仍显示出前景。
  - 用户讨论了其他分布式实现，如用于跨多设备运行模型的 [exo](https://github.com/exo-explore/exo)。人们对分布式设置中**实时节点池变化**和**层重新平衡**的潜在问题提出了担忧。


**主题 3：Nvidia 发布 NVLM 72B：新型多模态模型发布**

- **[Nvidia 刚刚发布了其多模态模型 NVLM 72B](https://i.redd.it/ix6hqg6c16sd1.jpeg)** ([Score: 92, Comments: 10](https://reddit.com//r/LocalLLaMA/comments/1ftrba0/nvidia_just_dropped_its_multimodal_model_nvlm_72b/)): **Nvidia** 发布了其**多模态模型 NVLM 72B**，详细信息见[论文](https://huggingface.co/papers/2409.11402)，模型可通过 [Hugging Face 仓库](https://huggingface.co/nvidia/NVLM-D-72B)访问。这个拥有 **720 亿参数**的模型代表了 Nvidia 进入多模态 AI 领域，能够处理和生成文本及视觉内容。
  - **NVLM 72B** 是基于 **Qwen 2 72B** 构建的，这一点通过快速查看配置文件即可发现。
  - **llama.cpp** 的创建者 **Ggerganov** 表示，需要具有软件架构技能的新贡献者来实现多模态支持，并对项目的可维护性表示担忧。他在一个 [GitHub issue 评论](https://github.com/ggerganov/llama.cpp/issues/8010#issuecomment-2376339571)中阐述了这一点。
  - 讨论中提到了为什么大公司以 **Hugging Face** 格式而不是 **GGUF** 发布模型。原因包括与现有硬件的兼容性、无需量化以及能够进行微调（这在 GGUF 文件上不易实现）。


**Theme 4. 端侧 AI 的进展：适用于 Android 的 Gemini Nano 2**



- **[Gemini Nano 2 现已通过实验性访问在 Android 上可用](https://android-developers.googleblog.com/2024/10/gemini-nano-experimental-access-available-on-android.html)** ([Score: 38, Comments: 12](https://reddit.com//r/LocalLLaMA/comments/1fu92au/gemini_nano_2_is_now_available_on_android_via/)): **Gemini Nano 2** 是 Google 为 Android 提供的端侧 AI 模型的升级版，开发者现在可以通过实验性访问获取。这一新迭代的版本大小**几乎是其前身 (Nano 1) 的两倍**，在质量和性能上表现出显著提升，在**学术基准测试**和**实际应用**中均可与大得多的模型相媲美。
  - 用户推测了从 **Gemini Nano 2** 中**提取权重**的可能性，并讨论了该模型的架构和大小。据澄清，Nano 2 拥有 **3.25B 参数**，而非最初建议的 2B。
  - 人们对**模型的透明度**表现出兴趣，询问为什么 Google 对所使用的 LLM 不够开放。有人猜测它可能是 **Gemini 1.5 flash** 的一个版本。
  - 一位用户提供了来自 [Gemini 论文](https://arxiv.org/pdf/2312.11805)的信息，指出 Nano 2 是通过**从更大的 Gemini 模型中蒸馏**训练而来的，并且为了部署进行了 **4-bit 量化**。


**Theme 5. 提升 LLM 性能的创新技术**



- **[Archon：来自斯坦福的推理时技术架构搜索框架。提供研究论文、代码、Colab；`pip install archon-ai`。o1 的开源版本？](https://i.redd.it/mnad8my7i3sd1.png)** ([Score: 35, Comments: 2](https://reddit.com//r/LocalLLaMA/comments/1ftiai0/archon_an_architecture_search_framework_for/)): 斯坦福大学的研究人员推出了 **Archon**，这是一个针对**推理时技术 (inference-time techniques)** 的开源**架构搜索框架**，可能作为 **Anthropic o1** 的开源替代方案。该框架可通过 `pip install archon-ai` 安装，并附带**研究论文**、**代码**和 **Colab 笔记本**，允许用户探索和实现大型语言模型的各种推理时方法。

- **[刚刚发现了幻觉评估排行榜 - GLM-4-9b-Chat 在最低幻觉率方面领先（OpenAI o1-mini 位居第二）](https://huggingface.co/spaces/vectara/Hallucination-evaluation-leaderboard)** ([Score: 39, Comments: 6](https://reddit.com//r/LocalLLaMA/comments/1ftpec9/just_discovered_the_hallucination_eval/)): **幻觉评估排行榜 (Hallucination Eval Leaderboard)** 显示 **GLM-4-9b-Chat** 表现最佳，**幻觉率最低**，其次是 **OpenAI 的 o1-mini**。这一发现促使人们考虑将 **GLM-4-9b** 作为 **RAG (检索增强生成)** 应用的潜在模型，暗示其在减少虚假信息生成方面的有效性。
  - **GLM-4-9b-Chat** 和 **Jamba Mini** 被强调为具有低幻觉率且极具前景的模型，但目前尚未得到充分利用。**Orca 13B** 进入前列也令人感到意外。
  - 该排行榜的数据被认为对**基于 LLM 的机器翻译**非常有价值，用户对该领域的潜在应用表示热切期待。
  - **GLM-4** 因其 **64K 有效上下文**而受到赞誉，这在 **RULER 排行榜**上超过了许多更大的模型，此外它在多语言任务中最小化语种切换 (code switching) 的能力，使其成为 **RAG 应用**的强力竞争者。

- **效果惊人的超智能摘要 Prompt** ([Score: 235, Comments: 39](https://reddit.com//r/LocalLLaMA/comments/1ftjbz3/shockingly_good_superintelligent_summarization/)): 该帖子讨论了一个受用户 **Flashy_Management962** 启发的 **摘要系统 Prompt**，其核心是生成 **5 个关键问题** 来捕捉文本要点，并详细回答这些问题。作者声称这种方法在 **Qwen 2.5 32b q_4** 上进行了测试，效果比他们之前尝试过的方法 **“好得惊人”**，并概述了制定针对中心主题、核心观点、事实、作者观点和影响的问题流程。
  - 用户讨论了通过 **指定回答长度** 和包含示例来优化 Prompt。楼主提到曾尝试过 **更复杂的 Prompt**，但发现简单的指令在 **Qwen 2.5 32b q_4 模型** 上效果最好。
  - 这种通过生成 **问答对** 来进行摘要的技术引发了关注，一些用户指出这是一个已知的 **NLP 任务**。楼主注意到显著的提升，将其描述为 LLM 在文本理解能力上实现了 **“30 点智商水平的飞跃”**。
  - 该摘要方法正以 **“supersummer”** 的名称被集成到 [Harbor Boost](https://github.com/av/harbor/wiki/5.2.-Harbor-Boost) 等项目中。用户还分享了相关资源，包括 [DSPy](https://github.com/stanfordnlp/dspy) 和一个 [电子书摘要工具](https://github.com/cognitivetech/ollama-ebook-summary)，供进一步探索。

## 其他 AI Subreddit 回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**AI 模型发布与功能**

- **OpenAI 发布 o1-mini 模型**：OpenAI 发布了 o1-mini，这是其 o1 模型的缩小版本。一些用户报告在使用 GPT-4 时随机获得 o1 的响应，这表明 OpenAI 可能正在测试 model router，以确定何时使用 o1 与 GPT-4。[来源](https://www.reddit.com/r/OpenAI/comments/1ftl4r1/interesting_theyre_testing_o1_vs_gpt4o_you_get_a/)

- **Whisper V3 Turbo 发布**：OpenAI 发布了 Whisper V3 Turbo，这是其 large-v3 语音识别模型的优化版本，提供 **8 倍的转录速度，且准确度损失极小**。[来源](https://www.reddit.com/r/singularity/comments/1ftmi99/openai_has_released_whisper_v3_turbo_model/)

- **适用于 Flux 的 PuLID 现在可在 ComfyUI 上运行**：用于 Flux 图像生成模型的 PuLID (Prompt-based Unsupervised Learning of Image Descriptors) 模型现在已兼容 ComfyUI 界面。[来源](https://www.reddit.com/r/StableDiffusion/comments/1fu2w0g/pulid_for_flux_works_on_comfyui_now/)

**AI 公司动态与事件**

- **OpenAI DevDay 公告**：OpenAI 举办了开发者日活动并发布了多项公告，包括：
  - 从 GPT-4 到 4o mini，**每个 token 的成本降低了 98%**
  - 其系统的 **token volume 增加了 50 倍**
  - 宣称“模型智能取得了卓越进展”
  [来源](https://www.reddit.com/r/singularity/comments/1ftm7ba/4_reveals_coming_from_openai_today_make_your/)

- **Mira Murati 从 OpenAI 离职**：据报道，在 Mira Murati 突然宣布从 OpenAI 离职之前，一些员工认为 o1 模型的发布过于仓促。[来源](https://www.reddit.com/r/singularity/comments/1ftwlrv/before_mira_muratis_surprise_exit_from_openai/)

**AI 功能与应用**

- **Advanced voice mode 开始推出**：OpenAI 开始向 ChatGPT 免费用户推出 Advanced voice mode。[来源](https://www.reddit.com/r/singularity/comments/1ftww6o/advanced_voice_mode_is_starting_to_roll_out_to/)

- **Realtime API 发布**：OpenAI 宣布了 Realtime API，它将在其他应用程序中实现 Advanced voice mode 功能。[来源](https://www.reddit.com/r/singularity/comments/1fttvf9/openai_announces_the_realtime_api_enabling/)

- **Copilot Vision 演示**：微软演示了 Copilot Vision，它可以查看用户正在浏览的网页并与之交互。[来源](https://www.reddit.com/r/singularity/comments/1ftrjzo/copilot_vision_can_see_the_same_webpages_you_do/)

- **NotebookLM 的功能**：Google 的 NotebookLM 工具可以处理多本书籍、长视频和音频文件，提供摘要、引用和解释。它还可以处理外语内容。[来源](https://www.reddit.com/r/singularity/comments/1ftogjk/notebooklm_is_too_good/)

**AI 伦理与社会影响**

- **对失业问题的担忧**：Duolingo 的 CEO 讨论了 AI 可能导致的失业问题，引发了关于自动化对社会影响的辩论。[来源](https://www.reddit.com/r/singularity/comments/1ftp5qt/this_is_why_the_vast_majority_of_redditors_are/)

- **Sam Altman 谈 AI 进展**：OpenAI 的 Sam Altman 讨论了 AI 的快速进展，表示到 2030 年，人们可能能够要求 AI 执行以前人类需要数月或数年才能完成的任务。[来源](https://www.reddit.com/r/singularity/comments/1fu61sz/sam_altman_by_2030_you_will_be_able_to_walk_up_to/)

**AI 研究与开发**

- **UltraRealistic Lora 项目**：一种用于 Flux 图像生成系统的新 LoRA (Low-Rank Adaptation) 模型，旨在创建更真实、更具动态感的摄影风格输出。[来源](https://www.reddit.com/r/StableDiffusion/comments/1ftmapd/ultrarealistic_lora_project_flux/)


---

# AI Discord 摘要

> 由 O1-mini 生成的摘要之摘要的摘要

**主题 1. AI 模型的进展与发布**

- [**Nova Pro 在基准测试中超越 GPT-4**](https://rubiks.ai/nova/release/)：**Nova-Pro** 在 ARC-C 上获得 **97.2%**，在 GSM8K 上获得 **96.9%** 的优异成绩，在推理和数学方面超越了 **GPT-4** 和 **Claude-3.5**。
- [**具备视觉能力的 Llama 3.2 发布**](https://huggingface.co/blog/llama32)：**Llama 3.2** 支持 **11B** 和 **90B** 配置，支持本地部署并增强了针对自定义视觉任务的微调。
- [**Phi-3.5 模型强调审查功能**](https://huggingface.co/SicariusSicariiStuff/Phi-3.5-mini-instruct_Uncensored)：**Phi-3.5-MoE** 展示了广泛的 **censorship mechanisms**（审查机制），引发了关于模型在技术应用中可用性的讨论。

---

**主题 2. AI 基础设施与工具增强**

- [**使用 o1-engineer 简化项目管理**](https://github.com/Doriandarko/o1-engineer)：**o1-engineer** 利用 **OpenAI API** 进行高效的 **code generation**（代码生成）和项目规划，增强了开发者工作流。
- [**通过 screenpipe 进行本地 AI 屏幕录制**](https://github.com/mediar-ai/screenpipe)：**screenpipe** 提供基于 **Rust** 构建的安全、持续的本地 AI 录制，是 **Rewind.ai** 的强大替代方案。
- [**解决 LM Studio 中的安装问题**](https://lmstudio.ai/docs/advanced/sideload)：社区成员排查了 **LM Studio** 的启动问题，强调了与 **Llama 3.1** 兼容性以及使用虚拟环境的重要性。

---

**主题 3. AI 伦理、安全与法律影响**

- [**辩论 AI 安全与伦理问题**](https://emu.baai.ac.cn/about?)：关于 **AI Safety** 的讨论涵盖了传统伦理和 **deepfakes** 等现代威胁，常被幽默地比作“愤怒的老奶奶对着云朵大喊”。
- [**NYT 诉讼影响 AI 版权立场**](https://x.com/crecenteb/status/1841482321909653505?s=46)：**NYT**（纽约时报）对 **OpenAI** 潜在的诉讼引发了关于 **copyright infringement**（版权侵权）和 **LLMs** 更广泛法律责任的疑问。
- [**创意作品中 AI 的伦理使用**](https://x.com/philpax_/status/1841502047385878867?s=46)：社区对 **Character.AI** 未经授权使用个人肖像表示愤慨，强调了负责任的 **AI development** 实践的必要性。

---

**主题 4. 模型训练、微调与优化**

- [**通过 Activation Checkpointing 提高效率**](https://github.com/karpathy/llm.c/pull/773)：在训练中实现 **activation checkpointing** 可以减少显存占用，从而能够处理像 **Llama 3.1 70B** 这样更大的模型。
- [**解决 FP8 精度训练挑战**](https://arxiv.org/abs/2409.12517)：研究人员探索了在长时间训练运行中 **FP8 precision** 的不稳定性，寻求优化稳定性和性能的解决方案。
- [**优化多 GPU 训练技术**](https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/#interleaved_schedule)：有效的 **multi-GPU training** 强调并行网络训练和高效的状态通信，以扩展到 **10,000 GPU** 的规模。

---

**主题 5. AI 集成与部署策略**

- [**使用 Oracle AI Vector Search 和 LlamaIndex 进行语义搜索**](https://medium.com/@andysingal/oracle-ai-vector-search-with-llamaindex-a-powerful-combination-b83afd6692b2)：将 **Oracle AI Vector Search** 与 **LlamaIndex** 结合，增强了 **RAG**（检索增强生成）流水线，以实现更准确的上下文数据处理。
- [**将 HuggingFace 模型部署为 LangChain Agents**](https://docs.langchain.com)：**HuggingFace** 模型可以作为 **Agents** 集成到 **LangChain** 中，促进开发工作流中高级的聊天和文本生成任务。
- [**使用 OpenRouter 和 LlamaIndex 的本地部署策略**](https://github.com/OpenInterpreter/open-interpreter/blob/main/docs/NCU_MIGRATION_GUIDE.md)：利用 **OpenRouter** 和 **LlamaIndex** 进行语义搜索和多模态模型，支持在各种应用中实现可扩展且高效的 **AI deployment**。


---

# 第一部分：Discord 高层级摘要

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Prompt Caching 见解**：分享了各种 AI 模型的 Prompt Caching 能力总结，重点关注 **OpenAI** 和 **Anthropic** 的策略，并讨论了成本影响和缓存未命中（cache misses）。
   - 关键点包括对 **DeepSeek** 和 **Gemini** 等模型缓存机制的讨论，强调了它们的效率。
- **代码编辑 AI 模型对比**：**Sonnet** 在整体性能上优于其他模型，但成本高于 **o1-preview**，后者在某些条件下提供更好的 Token 返还。
   - 建议包括将 **Gemini** 作为架构模型与 **Sonnet** 结合进行基准测试，以潜在地增强编辑能力。
- **YAML 解析陷阱**：用户指出了 YAML 解析中的怪异现象，特别是将 'yes' 等键转换为布尔值，这使他们的配置变得复杂。
   - 分享的预防策略包括使用引号字符串以保持预期的解析结果。
- **通过 o1-engineer 简化项目管理**：[o1-engineer](https://github.com/Doriandarko/o1-engineer) 是一个命令行工具，供开发者使用 **OpenAI API** 高效管理项目，执行 **code generation** 等任务。
   - 该工具旨在增强开发流程，特别侧重于项目规划。
- **使用 screenpipe 进行无缝本地 AI 录制**：[screenpipe](https://github.com/mediar-ai/screenpipe) 允许持续的本地 **AI screen recording**，专为构建需要完整上下文保留的应用程序而设计。
   - 定位为 **Rewind.ai** 的安全替代方案，它确保用户数据所有权，并使用 **Rust** 构建以实现更高效率。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Samba Nova 推出免费 Llama 端点**：与 Samba Nova 合作，**Llama 3.1** 和 **3.2** 的 **五个免费 bf16 端点** 现已在其新的推理芯片上线以测量性能，包括 **405B Instruct model**。
   - 在使用这些旨在支持 **Nitro** 生态系统的端点时，可以期待令人兴奋的吞吐量。
- **Gemini 模型标准化 Token 大小**：**Gemini** 和 **PaLM** 模型现在使用标准化的 Token 大小，导致 **价格提高 2 倍** 但 **输入缩短 25%**，这应该有助于提高整体负担能力 [详情点击这里](https://discord.com/channels/1091220969173028894/1092729520181739581/1288950180002926643)。
   - 尽管有这些变化，用户可以预期随着时间的推移成本将 **降低 50%**。
- **Cohere 提供模型折扣**：**Cohere models** 在 **OpenRouter** 上提供 **5% 折扣**，并已升级以在其新的 **v2 API** 中包含工具调用（tool calling）支持。
   - 此更新增强了用户对工具的访问，旨在改善整体体验。
- **Realtime API 集成讨论**：讨论集中在 OpenRouter 对新 **Realtime API** 的支持，特别是其目前在音频输入和输出方面的限制。
   - 用户渴望改进，但目前尚未确认增强功能的具体时间表。
- **OpenRouter 模型性能受到关注**：关于 OpenRouter 上 **模型性能** 和可用性的担忧浮出水面，特别是灰色显示的提供商和波动的费率。
   - 用户在选择不同提供商时需要对价格变化保持警惕。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Llama 3.2 发布带来本地运行支持**：[Llama 3.2](https://huggingface.co/blog/llama32) 已经发布，支持本地执行，并可通过[新方案](https://x.com/mervenoyann/status/1840040867224023221)进行视觉微调。该模型支持 **11B** 和 **90B** 配置，以增强微调能力。
   - 社区反馈显示反响热烈，成员们正在探索如何有效应用这些模型，并就其影响展开讨论。
- **Transformers 4.45.0 简化工具创建**：[transformers v4.45.0](https://x.com/AymericRoucher/status/1839246514331193434) 的发布引入了使用 `@tool` 装饰器的工具，为用户简化了开发流程。此次更新提升了多种应用的构建效率。
   - 社区成员热烈讨论了这些变化，征求对更新设计的反馈，并提出了各种用途。
- **为叙事生成微调 Mistral 7B**：一位成员热衷于为故事生成 **微调 Mistral 7B**，正在寻求预训练方法的指导。他们了解到 Mistral 是在大量数据上预训练的，强调了特定任务的微调方案。
   - 进一步的澄清区分了预训练与高效使模型专业化所需的精炼过程。
- **NotebookLM 超越传统工具**：参与者称赞 **NotebookLM** 作为一个端到端多模态 RAG 应用的功效，证明其在分析**财务报告**方面特别有效。一位成员在 [YouTube 视频](https://youtu.be/b2g3aNPKaU8)中展示了它的功能，探索了其在教育内容方面的潜力。
   - 团队成员对其应用潜力表示关注，加深了对其未来发展和集成的讨论。
- **探索 'trocr-large-handwriting' 的优势**：一位成员建议在与手写体高度相似的数据集上使用 **'trocr-large-handwriting'** 以获得更好的性能。对话包括了在特定字符数据集上进行微调以提高识别率的想法。
   - 这引发了关于手写识别任务模型选择的更广泛讨论，成员们权衡了各种方法的优缺点。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 启动困扰**：用户在更新后遇到了 **LM Studio** 的启动问题，特别是应用目录中的快捷方式和可执行文件问题，建议的一个解决方法是依赖更新后的安装文件。
   - 这表明旧版安装可能存在**潜在问题**，可能会阻碍用户的工作效率。
- **Llama 3.1 兼容性问题**：在 LM Studio 中加载 **Llama 3.1** 模型时出现错误，促使建议更新到官方支持该模型的 **0.3.3** 版本。
   - 这种不匹配凸显了在升级模型时确保软件兼容性的必要性。
- **Langflow 集成成功**：一位用户通过调整 OpenAI 组件的 base URL，成功将 **LM Studio** 与 **Langflow** 集成，发现该修改使工作流更加顺畅。
   - 他们指出了 Langflow 的可用资源，这可能会帮助他人简化配置。
- **优化 GPU 利用率**：关于 **LM Studio** 中 GPU 利用率设置的讨论集中在定义关于 CPU 和 GPU 资源管理的 “offload” 具体含义。
   - 成员们寻求关于使用 GPU 与 CPU（特别是针对自动补全等任务）优化配置的澄清。
- **高端 GPU 性能对决**：一位用户详细介绍了他们拥有 **7 块 RTX 4090 GPU** 的配置，其运行期间估计 **3000 瓦** 的功耗令人咋舌。
   - 另一位用户幽默地指出了如此高功耗的戏剧性影响，反映了人们对高性能系统的痴迷。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **快速模型量化令用户惊叹**：用户对 3b 模型的**极速量化**时间表示惊讶，处理过程不到一分钟。
   - *一位用户幽默地将其与最低工资劳动进行了经济对比*，强调了其相对于人工劳动的潜在效率提升。
- **音频 Token 定价引发关注**：关于音频 Token 输出成本为**每小时 14 美元**的讨论展开，一些人认为与人工 Agent 相比价格昂贵。
   - 参与者指出，虽然 AI 可以全天候在线，但该定价可能无法显著低于传统的支持岗位。
- **鲍尔默峰研究吸引成员**：一篇关于**鲍尔默峰 (Ballmer Peak)** 的共享论文表明，少量酒精可以增强编程能力，挑战了传统观念。
   - 成员们纷纷讨论在追求生产力“完美剂量”方面的*个人经验*。
- **DisTrO 处理恶意行为者的能力**：讨论指向 **DisTrO 的验证层**，该层能够在训练期间检测并过滤恶意行为者。
   - 虽然它本质上并不管理不可信节点，但该层提供了一定程度的保护。
- **RubiksAI 推出 Nova LLM 套件**：RubiksAI 发布了 **Nova** 系列大语言模型，其中 **Nova-Pro** 在 MMLU 上达到了令人印象深刻的 **88.8%**。
   - Nova-Pro 的基准测试分数为：ARC-C **97.2%**，GSM8K **96.9%**，重点关注 **Nova-Focus** 和改进的 **Chain-of-Thought** 能力。



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI 为雄心勃勃的计划筹集 66 亿美元**：OpenAI 已成功以惊人的 **1570 亿美元**估值筹集了 **66 亿美元**，由 Thrive Capital 以及 Microsoft 和 Nvidia 等机构促成。
   - CFO Sarah Friar 分享道，这将为融资后的员工提供流动性选择，标志着公司财务格局的重大转变。
- **Liquid.AI 声称取得架构突破**：围绕 **Liquid.AI** 展开了讨论，据报道其性能超越了 Ilya Sutskever 在 **2020 年**做出的预测。
   - 虽然一些怀疑者对其有效性表示质疑，但来自 Mikhail Parakhin 的见解为这些说法提供了一定程度的可信度。
- **AI 在高等数学中的潜力**：Robert Ghrist 发起了关于 AI 是否能从事**研究级数学**的对话，指出了 LLM 能力边界的移动。
   - 这场对话突显了随着 AI 开始应对复杂的猜想和定理，人们预期的转变。
- **AI 安全讨论引发进一步辩论**：在漫长的讨论中，成员们纠结于 **AI Safety** 的影响，特别是关于旧伦理和 **deepfakes** 等新兴威胁。
   - 评论将批评者比作“对着云朵大喊大叫的愤怒老奶奶”，说明了这场论辩的争议性。
- **谷歌的雄心壮志引发讨论**：由于谷歌拥有巨额现金储备和 AI 投资历史，人们对其推动 AGI 的潜力产生了猜测。
   - 对于该公司实现 AGI 愿景的真实承诺，疑虑依然存在，成员们的意见分歧很大。



---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **VAE 中的特征提取格式**：参与者讨论了 **Variational Autoencoders** 中首选的特征提取格式，倾向于使用 **continuous latent vectors** 或 *pt 文件*，并指出 **RGB 输入/输出** 对于 **Stable Diffusion** 等模型的相关性。
   - 对话强调了增强模型训练和有效性的实际选择。
- **AI 游戏反馈邀请**：一位成员邀请大家对其新推出的 AI 游戏提供反馈，可在 [game.text2content.online](https://game.text2content.online) 游玩，游戏内容涉及在时间限制下编写 prompt 来对 AI 进行 jailbreak。
   - 针对登录要求的担忧被提出，但创作者澄清这是为了减少游戏过程中的 bot 活动。
- **FP8 训练中的挑战**：分享了一篇讨论在使用 **FP8 precision** 训练大型语言模型（LLM）时面临的 **instabilities**（不稳定）问题的论文，该论文揭示了在长时间训练运行中出现的新问题；点击[此处](https://arxiv.org/abs/2409.12517)查看。
   - 社区成员热衷于探索在这些场景下优化稳定性和性能的解决方案。
- **2024 年 AI 峰会折扣码**：有人在征集参加在孟买举行的 **NVIDIA AI Summit 2024** 的 **discount codes**，一名学生表达了利用此机会与 AI 爱好者交流的兴趣。
   - 他们的 AI 和 LLM 背景使他们能够从参与峰会中获益匪浅。
- **Unsloth 模型加载故障**：一位用户在使用 [AutoModelForPeftCausalLM](https://huggingface.co/docs/huggingface_hub/deprecation) 加载带有 LoRA adapters 的微调模型时遇到错误，引发了关于调整 max_seq_length 的讨论。
   - 成员们对模型加载方法和解决问题的最佳实践提供了宝贵的见解。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **关于 Triton Kernel 调用参数的说明**：一位用户询问了在 Triton kernel 调用中更改 **num_stages** 的功能，推测其与 **pipelining** 有关。
   - 另一位成员解释说，pipelining 优化了加载、计算和存储操作，如本 [YouTube 视频](https://www.youtube.com/watch?v=ONrKkI7KhU4&ab_channel=Triton)所示。
- **CUDA Mode 活动引发关注**：CUDA mode 的**第三名奖项**被指出与数据加载项目有关，激发了大家对进度更新的好奇心。
   - 一位成员分享了 [no-libtorch-compile 仓库](https://github.com/lianakoleva/no-libtorch-compile)，以帮助在不使用 **libtorch** 的情况下进行开发。
- **IRL 主旨演讲现已可观看**：**IRL event** 的主旨演讲录像已发布，其中包括 **Andrej Karpathy** 等知名人物的精彩演讲。
   - 感谢参与者，特别是 **Accel**，感谢他们在有效记录这些演讲方面所做的贡献。
- **社区应对政治讨论**：社区成员对地缘政治稳定性表示担忧，强调在紧张的讨论中应专注于 coding。
   - 关于政治讨论适当性的辩论随之而来，成员们一致认为限制此类话题可以确保更舒适的环境。
- **即将于 10 月举行的 Advancing AI 活动**：一场 **Advancing AI event** 计划在**旧金山**举行，邀请参与者与 ROCM 开发者互动。
   - 鼓励社区成员私信获取注册详情，并在活动期间讨论 AI 的进展。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **贝叶斯模型面临频率派挑战**：神经架构主要利用 **frequentist statistics**（频率派统计学），这为在可训练模型中有效实现 **Bayesian networks** 带来了障碍。建议包括将概率折叠进模型权重，从而简化贝叶斯方法。
   - 讨论强调了在不牺牲复杂性的情况下，在 **Bayesian** 框架内保持实用性的替代方案。
- **NYT 诉讼动摇 AI 版权基础**：社区深入探讨了 OpenAI 可能通过向 **NYT**（纽约时报）支付费用来规避版权指控的影响，这引发了对 **LLM** 责任更广泛影响的担忧。有观点指出，此类补偿并不一定证实存在普遍的版权侵权。
   - 成员们强调了盈利公司与面临版权纠纷的独立创作者之间动机的差异。
- **液态神经网络（Liquid Neural Networks）：游戏规则改变者？**：成员们对 **liquid neural networks** 在拟合连续函数方面的应用表示乐观，认为与传统方法相比，它降低了开发复杂度。他们建议，在开发者能力达标的前提下，端到端流水线可以增强可用性。
   - 这些网络在减轻预测任务复杂性方面的潜力，引发了关于其实际部署的进一步讨论。
- **自监督学习拓展视野**：引入了在任意 **embeddings** 上进行 **self-supervised learning**（自监督学习）的概念，强调其在各种模型权重中的适用性。这种方法涉及从多个模型中收集线性层，以形成用于更好训练的综合数据集。
   - 成员们认识到扩展 **SSL** 在增强不同 AI 应用的模型能力方面的意义。
- **T5 彻底改变迁移学习**：**T5** 在 NLP 任务迁移学习中的有效性受到赞誉，其在建模各种应用方面具有显著能力。一位成员幽默地表示：“该死，T5 想到了一切”，展示了其广泛的 **text-to-text** 适应性。
   - 此外，讨论还涉及了深度学习优化器的新设计，批评了 **Adam** 等现有方法，并提出了改进训练稳定性的修改方案。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **用户渴望更高等级的订阅方案**：成员们讨论了推出更高价格的 OpenAI 订阅方案以提供更即时功能和服务的可能性，理由是对当前各 AI 平台限制的挫败感。
   - *这一变化可以通过创新功能提升用户体验*。
- **对新 Cove 语音模型的反馈**：多位用户对新的 **Cove voice model** 表示不满，称其缺乏经典语音的镇静感，并呼吁恢复经典语音。
   - *社区共识倾向于更喜欢宁静的声音，并对经典版本表示怀念*。
- **Liquid AI 的架构性能**：讨论集中在一种据报道优于传统 **LLM** 的新 **liquid AI** 架构上，该架构已开放测试，并以推理效率著称。
   - *成员们对其与典型 **Transformer** 模型相比的独特结构进行了推测*。
- **访问 Playground 的问题**：用户对登录 **Playground** 的困难表示担忧，一些用户建议使用无痕模式作为潜在的解决方法。
   - *报告显示访问问题可能因地理位置而异，特别是在瑞士等地区*。
- **macOS 应用中回复消失的问题**：用户报告在更新后 macOS 桌面应用中出现回复消失的问题，可能是由于通知设置的更改。
   - *在执行关键任务时，这些问题显著影响了用户体验，导致了明显的挫败感*。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI 获得 67 亿美元融资**：OpenAI 宣布完成一轮 **67 亿美元** 的融资，估值达到 **1570 亿美元**，并建立了关键合作伙伴关系，可能涉及北约盟友以推进 AI 技术。
   - 这笔资金引发了关于国际合作和 AI 政策战略方向的讨论。
- **面向所有用户的 Advanced Voice 功能**：OpenAI 正在向全球所有 ChatGPT Enterprise 和 Edu 用户推出 **Advanced Voice** 功能，并为免费用户提供早期预览。
   - 对于这些语音应用的实际性能提升，仍存在一些质疑。
- **深入探讨多 GPU 训练技术**：关于多 GPU 训练的详细讨论强调了高效 **checkpointing** 和状态通信的需求，特别是在使用多达 **10,000** 个 GPU 的情况下。
   - 重点介绍的关键策略包括并行化网络训练和增强故障恢复流程。
- **发布新型多模态模型 MM1.5**：Apple 推出了 **MM1.5** 系列多模态语言模型，旨在改进 OCR 和多图推理，提供 Dense 和 MoE 两个版本。
   - 此次发布重点关注专为视频处理和移动用户界面理解而定制的模型。
- **Azure AI 的 HD Neural TTS 更新**：Microsoft 在 **Azure AI** 上推出了高清版本的神经 TTS，承诺提供具有情感上下文检测的更丰富语音。
   - 诸如自回归 Transformer 模型等特性预计将增强生成语音的真实感和质量。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **解决 ComfyUI 安装问题**：一位用户在 Google Colab 上安装 **ComfyUI** 时遇到困难，特别是 comfyui manager 的安装过程。
   - 讨论者指出特定模型路径的重要性以及与 **Automatic1111** 的兼容性问题。
- **Flux 模型展示出色特性**：用户称赞了 **Flux model** 在创建一致的角色图像以及改进手部和脚部细节方面的效果。
   - 一位成员分享了一个 [Flux lora 链接](https://civitai.com/models/684810/flux1-dev-cctv-mania)，其对图像质量的提升出人意料地超出了其预期用途。
- **Automatic1111 安装问题依然存在**：在使用最新的 Python 版本安装 **Automatic1111** 时出现问题，引发了关于兼容性的疑问。
   - 成员们建议使用 **virtual environments** 或 **Docker** 容器来更好地管理不同的 Python 版本。
- **辩论基于 Debian 的操作系统特性**：一场热烈的对话集中在基于 **Debian** 的操作系统的优缺点上，重点介绍了 **Pop** 和 **Mint** 等流行发行版。
   - 用户幽默地分享了他们因 **Pop** 的独特功能而尝试重新使用它的想法。
- **Python 版本兼容性混乱**：成员们讨论了使用 **最新 Python 版本** 的挑战，建议旧版本可能会提高与某些脚本的兼容性。
   - 一位用户考虑调整其设置以分别执行脚本，从而克服稳定性问题。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **争取更高的速率限制**：用户讨论了请求增加 API **rate limit** 的选项，寻求突破 **20** 次请求的限制。
   - 这些请求得到了广泛支持，表明了对增强能力的集体需求。
- **迫切期待 Llama 3.2**：即将发布的 **Llama 3.2** 激发了用户对新功能的迫切期待。
   - 一个梗图反映了对发布日期的不确定感，幽默地引起了对过去延迟的关注。
- **LiquidAI 凭借速度成名**：**LiquidAI** 因其速度而受到称赞，一位用户宣称它与竞争模型相比 *快得惊人*。
   - 虽然速度是其优势，但用户也注意到了它的 **不准确性**，引发了对可靠性的担忧。
- **具备 PDF 功能的聊天特性表现出色**：一位用户确认成功将整个聊天记录下载为 **PDF**，引发了关于该功能实用性的讨论。
   - 这反映了对保存完整对话（尤其是为了文档记录）的更好方式日益增长的需求。
- **文本转语音评价褒贬不一**：关于 **text-to-speech (TTS)** 功能的讨论强调了其在处理长回复时的常用性，尽管存在一些 **发音问题**。
   - 用户认为它是一个方便的工具，但在准确性方面仍有改进空间。

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **需要信用卡云端和 Apple Pay 支持**：一位成员表示需要**对信用卡云端和 Apple Pay 的全面支持**，随后得到的建议是联系 [support@cohere.com](mailto:support@cohere.com) 以获取帮助。
   - 另一位成员提出协助处理该支持查询，以便更顺利地解决问题。
- **活动通知送达延迟**：一位成员报告了**活动通知**在活动结束后才送达的问题，特别是在最近一次的 **Office Hours 会议**期间。
   - 这已被确认为一个**技术故障**，官方对提出该问题表示了感谢。
- **咨询 MSFT Copilot Studio**：一位成员询问了关于 **MSFT Copilot Studio** 的使用经验，以及它与市场上其他解决方案相比的价值。
   - 回复中强调了讨论中关于促销内容的敏感性。
- **Azure 模型刷新故障**：一位成员报告了在 Azure 中刷新模型时遇到的问题，建议立即联系 Cohere 支持团队和 Azure 团队。
   - 另一位成员索要了相关的 Issue ID，以便在沟通中进行更好的跟踪。
- **对 Cohere 聊天应用开发的兴趣**：一位成员询问了是否有任何即将推出的 **Cohere 聊天应用**计划（特别是针对移动设备），并表达了对社区推广的热情。
   - 他们提出可以主持一场网络研讨会，并强调了对该平台的倡导。



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **高性价比的 Contextual Retrieval RAG 出现**：一位成员分享了 @AnthropicAI 的新 RAG 技术，该技术通过在文档块（chunks）前添加元数据来增强检索，从而提高性能和成本效益。这种方法能根据文档中的上下文位置更准确地引导 [检索过程](https://twitter.com/llama_index/status/1841210062167294287)。
   - 这种创新方法被定位为行业变革者，旨在简化各种应用中的数据处理。
- **Oracle AI Vector Search 在语义搜索中表现出色**：Oracle AI Vector Search 是 Oracle Database 的一项突破性功能，在**语义搜索**领域处于领先地位，使系统能够根据**含义而非关键词**来理解信息。当该技术与 **LlamaIndex 框架**结合时，被定位为构建复杂 RAG 流水线的**强大解决方案**。
   - Oracle 与 LlamaIndex 之间的协同作用增强了能力，推动了 AI 驱动的数据检索边界，详见这篇 [文章](https://medium.com/@andysingal/oracle-ai-vector-search-with-llamaindex-a-powerful-combination-b83afd6692b2)。
- **人类反馈助力 Multi-agent 写作**：一个利用 **Multi-agent 系统**的创新博客写作 Agent 将 **Human in the loop 反馈**整合到 TypeScript 工作流中，展示了动态的写作改进。观众可以在这个 [现场演示](https://twitter.com/llama_index/status/1841528125123133835) 中看到 Agent 实时进行写作和编辑。
   - 这一进展突显了通过直接的人类参与显著增强协作写作过程的潜力。
- **探讨 LlamaIndex 基础设施需求**：成员们分享了关于运行 LlamaIndex 的硬件规格见解，指出需求因模型和数据大小而异。关键考虑因素包括运行 LLM 和 Embedding 模型所需的 GPU，并推荐了特定的 Vector Database。
   - 讨论强调了影响部署决策的实际因素，以满足不同的项目需求。
- **NVIDIA 的 NVLM 引起关注**：NVIDIA 推出的多模态大语言模型 NVLM 1.0 受到关注，强调了其在视觉语言任务中的领先能力。成员们推测了 LlamaIndex 对其的潜在支持，特别是关于巨大的 GPU 需求和加载配置。
   - 讨论激发了人们对 LlamaIndex 内部实现可能带来的集成和性能基准测试的期待。



---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Salman Mohammadi 获贡献者奖提名**：我们自己的 **Salman Mohammadi** 因其在 GitHub 上的宝贵贡献以及在 Discord 社区的积极支持，获得了 **2024 PyTorch Contributor Awards** 的提名。
   - 他的工作对于推动 **PyTorch 生态系统**至关重要，该生态系统今年吸引了 **3,500** 人的贡献。
- **Distillation 中的 Tokenizer 概率与 One-Hot 对比**：成员们讨论了在 Token 训练中使用概率进行 **distillation**（蒸馏）与使用 One-Hot 向量的效果对比，强调了大型模型如何产生更好的潜在表示（latent representations）。
   - 他们一致认为，混合标注和未标注的数据可以“平滑”损失函数曲面（loss landscape），从而增强 **distillation 过程**。
- **H200 即将到来**：一位成员宣布他们的 **8x H200** 配置（拥有令人印象深刻的 **4TB RAM**）已在运送途中，这引发了热烈讨论。
   - 该配置将进一步助力其本地内部开发，强化其基础设施。
- **本地 LLM 获得优先权**：聊天中引发了关于部署本地 **LLM** 的讨论，指出目前的 API 无法满足欧洲医疗数据的要求。
   - 成员们强调，本地基础设施可以提高处理敏感信息的安全性。
- **B100 硬件计划即将出炉**：提出了未来集成 **B100** 硬件的计划，标志着向增强本地处理能力的转变。
   - 社区对获得更多资源以强化开发能力表示期待。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Literals 支持滞后**：一位成员确认 **literals**（字面量）在 Mojo 中尚无法正常工作，并建议使用 `msg.extend(List[UInt8](0, 0, 0, 0))` 作为替代方案。
   - 社区预期 **try** 表达式可能会包含在未来的更新中。
- **EC2 T2.Micro 实例的问题**：由于编译期间可能的内存限制，用户在廉价的 **EC2 t2.micro** 实例上遇到了 **JIT session error**。
   - 成员们建议至少使用 **8GB RAM** 以确保运行顺畅，其中一位指出 **2GB** 对于二进制构建（binary builds）已经足够。
- **关于 Mojo 库导入的讨论**：人们对 Mojo 未来支持 **import library** 功能以利用 CPython 库（而非使用 `cpython.import_module`）的兴趣日益浓厚。
   - 针对潜在的模块名称冲突，有人提出了 **import precedence**（导入优先级）策略进行集成。
- **内存管理策略探索**：有人建议在 EC2 上使用 **swap** 内存，但提醒注意因 IOPS 使用而导致的性能下降。
   - 另一位用户验证了在 **8GB** 内存下的成功运行，同时 Mojo 处理特定内存导入的问题也受到了关注。
- **Mojo 的导入行为**：据观察，Mojo 目前不像 Python 那样管理具有 **side effects**（副作用）的导入，这增加了兼容性的复杂性。
   - 这引发了关于 Mojo 编译器是否应该复制 Python 所有细微导入行为的讨论。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Nova LLM 发布引起轰动**：[Nova](https://rubiks.ai/nova) 发布了其 Large Language Models 系列，包括 **Nova-Instant**、**Nova-Air** 和 **Nova-Pro**，在 MMLU 上取得了 **88.8%** 的分数。
   - **Nova-Pro** 在 ARC-C 上获得 **97.2%**，在 GSM8K 上获得 **96.9%**，超越了竞争对手，彰显了其**顶级的推理和数学**能力。
- **Open Interpreter 支持动态函数调用**：成员们讨论了是否可以在其 Python 项目中使用 Open Interpreter 的 `interpreter.llm.supports_functions` 功能来定义自定义函数。
   - 虽然 **Open Interpreter** 可以即时创建函数，但严格的定义可以确保准确的模型调用，这一点在参考 [OpenAI documentation](https://platform.openai.com/docs/guides/function-calling) 时得到了澄清。
- **语音技术 Realtime API 发布**：全新的 [realtime API](https://openai.com/index/introducing-the-realtime-api/) 实现了 **speech-to-speech** 功能，增强了对话式 AI 中的交互关系。
   - 该 API 旨在通过即时响应增强应用程序，彻底改变交互式通信。
- **Vision 现已集成到 Fine-Tuning API**：OpenAI 宣布在 [fine-tuning API 中加入 vision](https://openai.com/index/introducing-vision-to-the-fine-tuning-api/)，允许模型在训练期间利用视觉数据。
   - 这一扩展为多模态 AI 应用开辟了新途径，进一步桥接了文本和图像处理。
- **模型蒸馏提高效率**：[Model distillation](https://openai.com/index/api-model-distillation/) 专注于优化模型权重管理以提升性能。
   - 该方法旨在维持模型准确性的同时最小化计算负载，确保输出的最优化。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain 等待 GPT Realtime API**：成员们热切讨论了 **LangChain** 何时支持新发布的 **GPT Realtime API**，但聊天中尚未出现明确的时间表。
   - 这种不确定性导致社区内对潜在功能和实现的持续猜测。
- **HuggingFace 现已成为 LangChain 的一个选项**：**HuggingFace 模型可以在 LangChain 中作为 Agent 使用**，用于包括聊天和文本生成在内的各种任务，并分享了实现的代码片段。
   - 为了进一步了解，成员们被引导至 [LangChain's documentation](https://docs.langchain.com) 和相关的 **GitHub issue**。
- **对 Prompt 中花括号的担忧**：一位成员对在 **LangChain** 的聊天 Prompt 模板中有效传递带有花括号的字符串表示担忧，因为它们会被解释为占位符。
   - 社区成员寻求不同的策略来处理此问题，而不改变处理过程中的输入。
- **Nova LLM 表现优于竞争对手**：**Nova** LLM（包括 **Nova-Instant**、**Nova-Air** 和 **Nova-Pro**）的发布展示了显著的性能，其中 **Nova-Pro** 在 MMLU 上取得了出色的 **88.8%**。
   - **Nova-Pro** 在 ARC-C 上也获得了 **97.2%**，在 GSM8K 上获得了 **96.9%**，确立了其在 AI 交互中的领先地位；点击[此处](https://rubiks.ai/nova/release/)了解更多。
- **LumiNova 提升图像生成**：全新的 **LumiNova** 模型承诺提供卓越的图像生成能力，增强 AI 应用的视觉创造力。
   - 这一进步为互动式和引人入胜的 AI 驱动体验开辟了新的可能性。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Qwen 2.5 在部署中表现惊人**：一位成员成功部署了 **Qwen 2.5** 34B，并报告其性能**好得离谱**，足以媲美 **GPT-4 Turbo**。
   - 讨论围绕部署细节和 vision 支持展开，强调了模型能力的快速演进。
- **小模型能力的探索**：成员们对小模型的显著进步感到惊叹，并辩论了它们的潜在极限。
   - *我们到底能把它推到多远？实际的限制是什么？* 对话反映了人们对优化更小架构日益增长的兴趣。
- **关于 hf_mlflow_log_artifacts 的澄清**：一位成员询问将 **hf_mlflow_log_artifacts** 设置为 true 是否会将模型 checkpoint 保存到 mlflow，这表明了对集成问题的关注。
   - 这突显了模型训练工作流中对强大日志机制的关键需求。
- **讨论 sharegpt 中的自定义 instruct 格式**：分享了在 **sharegpt** 中为数据集定义自定义 instruct 格式的说明，强调了 YAML 的使用。
   - 概述了关键步骤，包括自定义 Prompt 以及确保 JSONL 格式兼容性以获得成功结果。

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tiny Box 开箱深得人心**：一位成员开箱了来自 Proxy 的 **tiny box**，并强调了**精美的包装**和**木质底座**是其亮点。
   - *尽管担心 ny->au 的运输过程*，他们还是称赞了确保包裹成功送达所付出的努力。
- **讨论 Bugfix PR 方案**：有人呼吁对[这个 bugfix PR](https://github.com/tinygrad/tinygrad/pull/6815)进行审查，该 PR 解决了两次保存和加载 tensor 的问题。
   - 该 PR 旨在解决 **#6294**，揭示了磁盘设备在不创建新文件的情况下保留未链接文件的问题，这仍然是一个关键的开发点。
- **Tinygrad 代码提升编程技能**：参与 **tinygrad** 代码库被证明能提升成员在日常工作中的编程技能，证明了开源经验的价值。
   - 他们分享道：*作为副作用，它让我的日常工作编程变得更好*，反映了对编程能力的积极影响。
- **C 互操作性是一大优势**：成员们讨论了 **Python** 的生产力如何与其 **C interoperability**（C 互操作性）相媲美，允许平滑的函数调用，从而提高底层操作的性能。
   - 尽管在 struct 方面存在一些限制，但共识是快速迭代带来的收益仍然巨大。
- **UOp 与 UOP 优化的困扰**：一位成员表达了在优化 **UOp vs UOP pool** 时面临的挑战，理由是单个对象引用使过程复杂化。
   - 他们建议使用一种更高效的存储类，利用整数句柄（integer handles）来更好地管理对象引用。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **强烈的反垃圾信息情绪**：一位成员表达了对 **spam**（垃圾信息）的强烈厌恶，强调了对社区中不受欢迎消息的沮丧。
   - 这反映了一个共同的挑战，成员们敦促进行更好的审核，以控制垃圾信息对交流的影响。
- **Sci Scope Newsletter 发布公告**：来自 Sci Scope 的**个性化通讯**现已上线，每周提供针对首选研究领域和新论文的定制更新。
   - *再也不会错过与你工作相关的研究！* 用户可以[现在尝试](https://sci-scope.com/)，以一种轻松的方式跟上 AI 领域的进展。
- **为繁忙专业人士提供的每周 AI 研究摘要**：该通讯将扫描新的 **ArXiv papers** 并提供简洁的摘要，旨在每周为订阅者节省**数小时的工作时间**。
   - 该服务承诺通过**每周高层级摘要**来简化选择相关阅读材料的任务。
- **新用户专属优惠**：新用户可以注册 **1 个月的免费试用**，其中包括访问自定义查询和更相关的体验。
   - 这一举措增强了参与度，使用户更容易跟上快速发展的 AI 领域。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **来自 Sci Scope 的个性化通讯**：**Sci Scope** 推出了个性化通讯，每周递送根据个人兴趣定制的新论文摘要，帮助用户轻松掌握最新动态。
   - 该服务根据用户偏好扫描新的 **ArXiv papers**；它提供 **1 个月的免费试用**以吸引新用户。
- **关于代码相似性搜索的咨询**：一位成员正在探索**代码相似性**的方案，并考虑使用 **Colbert** 从代码片段中输出相关的代码文档，质疑其在没有 **finetuning**（微调）的情况下的有效性。
   - 他们还在寻求**代码搜索**的其他替代方法，突显了社区在有效方法上的协作。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **实验作业推迟**：一位成员询问了原定于今天发布的 Lab assignments（实验作业），随后确认工作人员需要*再花一周时间*来准备。更新信息将在课程页面 [llmagents-learning.org](https://llmagents-learning.org/f24) 上公布。
   - 延迟引发了担忧，参与者对缺乏关于发布时间表和更新的沟通表示沮丧。
- **沟通脱节问题凸显**：由于 Lab 发布更新不足，引发了担忧，一名成员无法找到相关的电子邮件或公告。这种情况凸显了在参与者期望中，课程沟通需要改进。
   - 参与者正在等待有关课程进展的重要信息，强调了课程工作人员及时发布公告的紧迫性。

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **建立 ML 论文阅读小组**：一名成员提议启动 [ML 论文阅读小组](https://discord.com/channels/1089876418936180786/1290380988450340864)，旨在讨论最新研究，增强社区互动。
   - 该倡议旨在促进对 Machine Learning 最新进展感兴趣的工程师之间的集体知识共享。
- **发布本地 LLM 应用的技巧**：社区成员对有效将本地 **LLM-based apps** 发布到应用商店的见解表示感谢。
   - 这些技巧被认为对于应对应用发布复杂性的人员至关重要。
- **社区招聘板提议引起关注**：关于创建一个 [招聘板](https://discord.com/channels/1089876418936180786/1290677600527585311) 以促进社区职位发布的讨论正在展开。
   - 该想法由一名成员发起，旨在将人才与工程领域的就业机会联系起来。
- **Lumigator 获得官方关注**：社区在 [官方帖子](https://www.linkedin.com/posts/mozilla-ai_introducing-lumigator-activity-7246888824507613187-oTho) 中介绍了 **Lumigator**，展示了其功能和特性。
   - 这一介绍强化了社区致力于突出与 AI 工程师相关的值得关注的项目。
- **即将举行的技术创新活动**：重点介绍了几个即将举行的活动，包括专注于搜索技术的 [Hybrid Search](https://discord.com/events/1089876418936180786/1284180345553551431) 讨论。
   - 其他会议，如 [Data Pipelines for FineTuning](https://discord.com/events/1089876418936180786/1290035138251587667)，有望进一步提升工程知识和技能。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Nova 模型表现优于竞争对手**：介绍 [Nova](https://rubiks.ai/nova)：下一代大语言模型，在各种基准测试中击败了 **GPT-4** 和 **Claude-3.5**，其中 **Nova-Pro** 在 MMLU 上以 **88.8%** 领先。
   - **Nova-Air** 在各种应用中表现出色，而 **Nova-Instant** 则提供快速且具有成本效益的解决方案。
- **Nova 模型卓越的基准测试表现**：Nova-Pro 取得了令人印象深刻的分数：推理方面 ARC-C 为 **97.2%**，数学方面 GSM8K 为 **96.9%**，编程方面 HumanEval 为 **91.8%**。
   - 这些基准测试巩固了 Nova 作为 AI 领域顶级竞争者的地位，展示了其非凡的能力。
- **LumiNova 彻底改变图像生成**：新推出的 **LumiNova** 为图像生成设定了高标准，承诺在视觉效果上提供无与伦比的质量和多样性。
   - 该模型补充了 Nova 系列，为用户提供了轻松创建惊人视觉效果的高级工具。
- **Nova-Focus 的未来发展**：开发团队正在探索 **Nova-Focus** 和增强的 Chain-of-Thought 能力，以进一步突破 AI 的界限。
   - 这些创新旨在完善和扩展 Nova 模型在推理和视觉生成方面的潜在应用。

---

**Alignment Lab AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道详细摘要和链接

{% if medium == 'web' %}

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1290756176031780885)** (198 messages🔥🔥): 

> - `Prompt Caching Support` (提示词缓存支持)
> - `AI Models Performance Comparison` (AI 模型性能对比)
> - `YAML Parsing Issues` (YAML 解析问题)
> - `File Editing with Aider` (使用 Aider 进行文件编辑)
> - `Error Handling in Aider` (Aider 中的错误处理)


- **Prompt Caching 支持概览**：分享了各种 AI 模型的 Prompt Caching 能力摘要，包括其成本和机制，重点介绍了 OpenAI 和 Anthropic 的缓存策略。
   - 用户讨论了缓存未命中（cache misses）的影响，以及 DeepSeek 和 Gemini 等不同模型之间的成本效率。
- **用于代码编辑的 AI 模型对比**：Reaper_of_fire 指出，虽然 Sonnet 总体性能更好，但与 o1-preview 相比价格更高，而后者有时能获得更多的 Token 报销。
   - 有建议提出将 Gemini 作为 Architect 模型，配合 Sonnet 作为 Editor 模型，与仅使用 Sonnet 相比，这可能会改善编辑效果。
- **YAML 解析的挑战**：讨论了 YAML 的解析怪癖，特别是像 'yes' 这样的键会被意外转换为布尔值，从而增加了配置管理的复杂性。
   - 用户分享了通过使用引号字符串来避免意外解析的见解，强调了 YAML 的这一缺点。
- **Aider 中的文件处理**：有人对 Aider 的文件管理功能表现提出担忧，特别是 `/read-only` 命令没有按预期自动补全文件路径。
   - 用户讨论了最近的功能更改是否可能导致此问题，特别是在预输入文件设置中。
- **AI 模态的影响**：用户对 AI 模型模态表达了不同的偏好，指出虽然 o1-preview 具有优势，但用户发现 Sonnet 在特定语境下能更有效地处理 Prompt。
   - 社区分享了关于 AI 交互中需要不同格式的看法，以及模型是否应该进行调整以获得更好的输出结果。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/config/adv-model-settings.html">Advanced model settings</a>：配置 LLM 的高级设置。</li><li><a href="https://www.bram.us/2022/01/11/yaml-the-norway-problem/">YAML: The Norway Problem</a>：本周早些时候，Haroen Viaene 发布了关于 YAML 的推文：yaml 最糟糕的部分：https://yaml.org/type/bool.html —— Haroen Viaene (@haroenv) 2022 年 1 月 10 日。链接页面包含文档...</li><li><a href="https://github.com/enricoros/big-AGI/blob/big-agi-2/src/modules/3rdparty/THIRD_PARTY_NOTICES.md">big-AGI/src/modules/3rdparty/THIRD_PARTY_NOTICES.md at big-agi-2 · enricoros/big-AGI</a>：由尖端模型驱动的生成式 AI 套件，提供高级 AI/AGI 功能。其特点包括 AI 角色、AGI 功能、多模型聊天、文本转图像、语音、响应流式传输等...</li><li><a href="https://aider.chat/docs/config/options.html#history-files),">Options reference</a>：关于 aider 所有设置的详细信息。</li><li><a href="https://github.com/Shakahs/aider/commit/47012061b8cc7aa43f9eec282798ef29220fde4e">Add experimental vector search. · Shakahs/aider@4701206</a>：无描述。</li><li><a href="https://github.com/paul-gauthier/aider/blob/72cb5db53066a1ca878412b866e87b916529b68e/aider/repo.py#L276-L291">aider/aider/repo.py at 72cb5db53066a1ca878412b866e87b916529b68e · paul-gauthier/aider</a>：aider 是你终端里的 AI 配对编程工具。通过在 GitHub 上创建账号来为 paul-gauthier/aider 的开发做出贡献。</li><li><a href="https://www.bram.us/2022/01/11/yaml-the-norwa">YAML: The Norway Problem</a>：本周早些时候，Haroen Viaene 发布了关于 YAML 的推文：yaml 最糟糕的部分：https://yaml.org/type/bool.html —— Haroen Viaene (@haroenv) 2022 年 1 月 10 日。链接页面包含文档...</li><li><a href="https://github.com/paul-gauthier/aider/blob/72cb5db53066a1ca878412b866e87b916529b68e/aider/repo.py#L359">aider/aider/repo.py at 72cb5db53066a1ca878412b866e87b916529b68e · paul-gauthier/aider</a>：aider 是你终端里的 AI 配对编程工具。通过在 GitHub 上创建账号来为 paul-gauthier/aider 的开发做出贡献。</li><li><a href="https://github.com/paul-gauthier/aider/blob/72cb5db53066a1ca878412b866e87b916529b68e/aider/repo.py#L329">aider/aider/repo.py at 72cb5db53066a1ca878412b866e87b916529b68e · paul-gauthier/aider</a>：aider 是你终端里的 AI 配对编程工具。通过在 GitHub 上创建账号来为 paul-gauthier/aider 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1290763008188088412)** (70 条消息🔥🔥): 

> - `Architect Mode 使用`
> - `Aider 中的 Cache 管理`
> - `使用本地模型设置 Aider`
> - `Norton Antivirus 问题`
> - `Obsidian 与 LLM 集成` 


- **关于 Architect Mode 使用的澄清**：用户讨论了如何在不触发文件编辑的情况下有效利用 Aider 的 Architect Mode，强调了清晰指令的必要性。
   - 一位用户注意到 Prompt 的变化仍会导致意外的代码输出，并分享了模型从头开始生成响应的经历。
- **理解 Aider 中的 Cache 管理**：对话强调编辑文件会使 Cache 失效，建议在长时间交互期间保持文件只读，以提高 Prompt Caching 效率。
   - 参与者承认频繁的更改可能会抵消 Caching 带来的成本节省，这表明在最佳使用方式上可能存在误解。
- **使用本地模型设置 Aider**：一位用户询问如何将 Aider 连接到两个本地托管的 Ollama 实例，并得到了关于 LiteLLM 单端点支持的指导。
   - 建议对于需要连接多个实例的高级设置，可能需要探索 OpenAI API 代理。
- **Norton Antivirus 问题**：一位用户报告了 Norton 拦截文件和目录的问题，导致在移动受影响文件后运行脚本时出现复杂情况。
   - 回复中包括了配置 Git 设置以将目录标记为安全，从而绕过所有权问题的建议。
- **将 LLM 与 Obsidian 集成**：用户分享了在工作流中集成 LLM 的设置，特别强调除了基础自动化外，缺乏广泛的插件使用。
   - 在 LLM 交互中，用户更倾向于使用聊天界面，同时提到了 `aichat` 等工具在终端使用中的价值。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/llms/openai.html">OpenAI</a>: aider 是你终端里的 AI 结对编程工具</li><li><a href="https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-and-highlighting-code-blocks">创建和高亮代码块 - GitHub Docs</a>: 无描述</li><li><a href="https://aider.chat/docs/llms/ollama.html">Ollama</a>: aider 是你终端里的 AI 结对编程工具</li><li><a href="https://aider.chat/docs/usage/modes.html">Chat modes</a>: 使用 chat, ask 和 help 聊天模式。
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1290796871488508028)** (3 条消息): 

> - `o1-engineer 工具`
> - `screenpipe 录制` 


- **使用 o1-engineer 进行高效项目管理**：[o1-engineer](https://github.com/Doriandarko/o1-engineer) 是一个命令行工具，通过利用 **OpenAI API** 实现 **Code Generation** 和项目规划等功能，协助开发者高效管理和交互项目。
   - 该工具旨在简化开发工作流，使开发者的项目管理更加有效。
- **使用 screenpipe 进行 24/7 本地 AI 录制**：[screenpipe](https://github.com/mediar-ai/screenpipe) 提供本地 **AI 屏幕和麦克风录制**，允许构建具有完整上下文的 AI 应用，从而为 **Rewind.ai** 等服务提供了一个安全的替代方案。
   - 它支持 **Ollama** 并优先考虑用户数据所有权，使用 **Rust** 开发以保证性能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/mediar-ai/screenpipe">GitHub - mediar-ai/screenpipe: 24/7 本地 AI 屏幕和麦克风录制。构建具有完整上下文的 AI 应用。支持 Ollama。Rewind.ai 的替代方案。开源。安全。数据归你所有。Rust。</a>: 24/7 本地 AI 屏幕和麦克风录制。构建具有完整上下文的 AI 应用。支持 Ollama。Rewind.ai 的替代方案。开源。安全。数据归你所有。Rust。 - mediar-ai/screenpipe</li><li><a href="https://github.com/Doriandarko/o1-engineer">GitHub - Doriandarko/o1-engineer: o1-engineer 是一个命令行工具，旨在协助开发者高效管理和交互项目。利用 OpenAI API 的强大功能，该工具提供代码生成、文件编辑和项目规划等功能，以简化你的开发工作流。</a>: o1-engineer 是一个命令行工具，旨在协助开发者高效管理和交互项目。利用 OpenAI API 的强大功能，该工具提供功能...
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1290786661545807893)** (2 条消息): 

> - `Llama 3.1 and 3.2 Endpoints`
> - `Gemini Token Standardization`
> - `Cohere Model Discounts`
> - `Chatroom Upgrades` 


- **Samba Nova 推出免费 Llama Endpoints**：与 Samba Nova 合作，在其新型推理芯片上现已提供针对 **Llama 3.1** 和 **3.2** 的 **五个免费 bf16 Endpoints**，旨在衡量性能表现。
   - **405B Instruct** 模型的一流吞吐量（Throughput）已经引起了轰动，如果性能符合预期，它将成为 **Nitro** 极具前景的新成员。
- **Gemini 模型进行 Token 标准化**：**Gemini** 和 **PaLM** 模型现在已标准化为使用相同的 Token 大小，这将导致**价格约提高 2 倍**，同时输入长度**缩短 25%**。
   - 尽管有这些变化，预计成本仍将降低 **50%**，这让用户对整体负担能力感到放心，详情请见 [这里](https://discord.com/channels/1091220969173028894/1092729520181739581/1288950180002926643)。
- **Cohere 的新折扣和 Tool Calling 功能**：**OpenRouter** 上的 Cohere 模型现在提供 **5% 的折扣**，并已升级到其 **v2 API**，完整支持 Tool Calling。
   - 用户现在可以更高效地访问这些工具，旨在提升整体用户体验。
- **Chatroom 落地页升级**：新的 **Chatroom 落地页** 增强了模型对比功能，并包含用于评估模型性能的智能测试，访问地址：[openrouter.ai/chat](https://openrouter.ai/chat)。
   - 用户现在可以受益于改进后的 **LaTeX 格式化工具**和更好的代码格式化工具，以提升编程体验。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/models/meta-llama/llama-3.1-8b-instruct:free">Llama 3.1 8B Instruct (free) - API, Providers, Stats</a>：Meta 最新的模型系列 (Llama 3.1) 推出了多种尺寸和版本。通过 API 运行 Llama 3.1 8B Instruct (免费)</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-70b-instruct:free">Llama 3.1 70B Instruct (free) - API, Providers, Stats</a>：Meta 最新的模型系列 (Llama 3.1) 推出了多种尺寸和版本。通过 API 运行 Llama 3.1 70B Instruct (免费)</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-405b-instruct:free">Llama 3.1 405B Instruct (free) - API, Providers, Stats</a>：备受期待的 400B 级 Llama3 来了！它拥有 128k 上下文和令人印象深刻的评估分数，Meta AI 团队继续推动开源 LLM 的前沿。Meta 最新的...</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.2-3b-instruct:free">Llama 3.2 3B Instruct (free) - API, Providers, Stats</a>：Llama 3.2 3B 是一个拥有 30 亿参数的多语言大语言模型，针对对话生成、推理和摘要等高级自然语言处理任务进行了优化。运行 Llama 3.2 ...</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.2-1b-instruct:free">Llama 3.2 1B Instruct (free) - API, Providers, Stats</a>：Llama 3.2 1B 是一个拥有 10 亿参数的语言模型，专注于高效执行自然语言任务，如摘要、对话和多语言文本分析。运行 Llama 3.2 1B Instruc...
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1290761387642978444)** (244 条消息🔥🔥): 

> - `Realtime API 更新`
> - `OpenRouter 模型性能`
> - `文件上传限制`
> - `OpenAI 缓存`
> - `免费额度计划` 


- **关于 Realtime API 的讨论**：用户询问了 OpenRouter 对新推出的 Realtime API 的潜在支持情况，并强调了其目前在音频输入和输出方面的限制。
   - 社区对集成新功能持续关注，但目前尚未确定明确的时间表。
- **OpenRouter 模型性能问题**：成员们对模型性能和可用性表示担忧，特别是在不同情况下加载不同 Provider 的表现。
   - 讨论了 Provider 显示为灰色以及计费费率变化的具体案例，提示用户需要关注价格变动。
- **文件上传限制**：提出了关于 OpenRouter 内部文件上传能力的问题，用户报告了与移动设备和不支持的文件类型相关的问题。
   - 虽然有建议通过重命名文件或以文本形式输入，但底层的 HTML 限制被认为是主要问题。
- **OpenAI 缓存见解**：分享了各种商业 Context Caching 实现的详细分解，解释了它们的工作原理及各自的折扣率。
   - 这被认为是用户理解不同模型成本影响的宝贵信息。
- **免费额度计划咨询**：用户询问了可用的免费额度计划，表示在资源有限的情况下开展工作存在挑战。
   - 官方澄清虽然没有通用的额度计划，但可能会根据用户活跃度发放与研究相关的 Credits。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/jordibruin/status/1841138499119993204">Jordi Bruin (@jordibruin) 的推文</a>: MacWhisper 10.0 现已作为免费更新发布！- 支持新的 Whisper Turbo 模型，以超高精度实现高达 21 倍的实时速度 - 支持使用 Ollama 的本地 AI 模型 - 支持自定义...</li><li><a href="https://rubiks.ai/nova/?c=d5d562ad-6c96-4142-ba8c-4a0a8f54bf74">Nova</a>: 探索 Nova，来自 Rubik's AI 的先进 AI 解决方案。体验智能推理、数学、编程和图像生成。</li><li><a href="https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard">UGI Leaderboard - DontPlanToEnd 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://x.com/RubiksAI/status/1841224714045264304">Rubiks AI (@RubiksAI) 的推文</a>: 🚀 隆重推出 Nova：由 Nova 打造的下一代 LLM！🌟 我们很高兴宣布推出最新的大语言模型系列：Nova-Instant、Nova-Air 和 Nova-Pro。每一款都专为...</li><li><a href="https://openrouter.ai/settings/privacy">隐私 | OpenRouter</a>: 管理您的隐私设置</li><li><a href="https://openrouter.ai/models?q=uncensored">模型：'uncensored' | OpenRouter</a>: 在 OpenRouter 上浏览模型</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.2-3b-instruct">Llama 3.2 3B Instruct - API, Providers, Stats</a>: Llama 3.2 3B 是一个拥有 30 亿参数的多语言大语言模型，针对对话生成、推理和摘要等高级自然语言处理任务进行了优化。运行 Llama 3.2 ...</li><li><a href="https://artificialanalysis.ai/models/llama-3-1-instruct-405b/providers#speed">Llama 3.1 405B: API Provider 性能基准测试与价格分析 | Artificial Analysis</a>: 对 Llama 3.1 Instruct 405B 的 API Provider 进行性能指标分析，包括延迟（首个 token 时间）、输出速度（每秒输出 token 数）、价格等。API Provider 基准...</li><li><a href="https://openrouter.ai/docs/provider-routing">Provider 路由 | OpenRouter</a>: 在多个 Provider 之间路由请求</li><li><a href="https://github.com/bobcoi03/opencharacter">GitHub - bobcoi03/opencharacter: Character.AI 的开源替代方案 - 创建你自己的无过滤、无审查角色</a>: Character.AI 的开源替代方案 - 创建你自己的无过滤、无审查角色 - bobcoi03/opencharacter
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1290763047190921372)** (1 条消息): 

> - `Llama 3.2 发布`
> - `Transformers v4.45.0`
> - `Whisper Turbo 集成`
> - `GGUF 模型部署`
> - `HuggingChat macOS 版`

- **Llama 3.2 发布，带来令人兴奋的新特性**：[Llama 3.2](https://huggingface.co/blog/llama32) 已正式发布，用户可以通过[新配方](https://x.com/mervenoyann/status/1840040867224023221)在本地运行并进行视觉微调。现在，你只需几行代码即可在你自己的数据集上对 Llama 3.2 Vision 进行后训练（post-train）！
   - 通过 SFTTrainer 对 **11B** 和 **90B** 模型提供支持，增强了针对自定义任务的微调能力，使模型能够“看见”并“遵循”用户指令。
- **Transformers v4.45.0 引入简化的工具构建**：[transformers v4.45.0](https://x.com/AymericRoucher/status/1839246514331193434) 的发布包含了一种更简单的方法，即使用带有类型提示（type hints）的函数和 `@tool` 装饰器来构建工具。这一改进简化了用户的工具创建流程，使过程更加直观。
   - 随着社区探索这种极速提升项目工具构建效率的方法，欢迎大家提供反馈。
- **Transformers 现已支持 Whisper Turbo**：[Whisper Turbo](https://www.reddit.com/r/LocalLLaMA/comments/1ftjqg9/whisper_turbo_now_supported_in_transformers) 集成已宣布，扩展了 Transformers 库的功能。此版本进一步增强了现有框架内的**语音识别**能力。
   - 鼓励用户探索这一新集成，并在其应用程序中进行尝试。
- **在 Inference Endpoints 上轻松部署 GGUF 模型**：Hugging Face 现在允许将 [GGUF 模型](https://www.linkedin.com/feed/update/urn:li:ugcPost:7245455792974295040/) 直接部署到其 Inference Endpoints 上，从而更简单地向最终用户提供模型服务。这一特性为追求简化部署的模型作者降低了复杂性。
   - 随着此次更新，该平台上的模型可访问性和可用性得到了显著增强。
- **HuggingChat macOS 测试版发布**：[HuggingChat](https://x.com/alvarobartt/status/1838949140513927311) 现已推出 macOS 测试版，方便用户访问顶尖的开源模型。用户只需联网并拥有 Hugging Face Hub 账号即可使用该服务。
   - 鼓励用户对测试版提供反馈，以确保最佳的用户体验。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/mervenoyann/status/1840040867224023221)">来自 merve (@mervenoyann) 的推文</a>：如果你错过了：我为 huggingface-llama-recipes 贡献了一个 Llama 3.2 Vision 微调配方 🦙</li><li><a href="https://x.com/_lewtun/status/1839018100991082669)">来自 Lewis Tunstall (@_lewtun) 的推文</a>：现在任何人只需几行代码，即可使用 TRL 在自己的数据集上对 Llama 3.2 Vision 进行后训练 🚀！我们刚刚在 SFTTrainer 中增加了对 11B 和 90B 模型的支持，因此你可以进行微调...</li><li><a href="https://x.com/xenovacom/status/1840767709317046460)">来自 Xenova (@xenovacom) 的推文</a>：Llama 3.2 在你的浏览器中通过 WebGPU 100% 本地运行！🦙 每秒高达 85 个 token！⚡️ 由 🤗 Transformers.js 和 ONNX Runtime Web 提供支持。无需安装...只需访问网站即可！查看...</li><li><a href="https://x.com/reach_vb/status/1839688569901719698)">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：在免费的 Google Colab 中运行 Llama 3.2 1B & 3B！🔥 由 Transformers ⚡ 提供支持</li><li><a href="https://x.com/abhi1thakur/status/1839293754991317468)">来自 abhishek (@abhi1thakur) 的推文</a>：以下是你可以如何在本地和云端轻松微调最新的 Llama 3.2 (1B 和 3B)：</li><li><a href="https://x.com/AymericRoucher/status/1839246514331193434)">来自 Aymeric (@AymericRoucher) 的推文</a>：Transformers v4.45.0 发布：包含一种极速构建工具的方法！⚡️ 在与同事 @MoritzLaurer 和 Joffrey Thomas 进行用户调研期间，我们发现目前的类定义...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ftjqg9/whisper_turbo_now_supported_in_transformers)">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://x.com/alvarobartt/status/1838949140513927311)">来自 Alvaro Bartolome (@alvarobartt) 的推文</a>：🤗 HuggingChat 现已面向 macOS 用户提供测试版！现在，最新的顶尖开源模型对 macOS 用户来说触手可及；你只需要网络连接和 Hugging Face Hub 账号...</li><li><a href="https://x.com/lunarflu1/status/1841070211379667018)">来自 lunarflu (@lunarflu1) 的推文</a>：@huggingface 模型作者现在可以使用新的元数据：`new_version`。如果一个模型定义了更新的版本，模型页面将显示一个链接到最新版本的横幅！
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1290756586226192384)** (164 条消息🔥🔥): 

> - `Model Performance Comparison` (模型性能对比)
> - `Fine-Tuning Challenges` (微调挑战)
> - `Innovative LLM Projects` (创新 LLM 项目)
> - `Hugging Face Contributions` (Hugging Face 贡献)
> - `Community Queries` (社区疑问)


- **Llama 3.2 模型的性能**：社区讨论了 Llama 3.2 模型的能力，特别强调了 1B 模型被认为是同类中最好的，3B 模型表现也极佳。
   - 有观点认为，较小的模型受限于其数据，影响了它们像大型模型那样“思考”的能力。
- **Llama 3.2 访问挑战**：用户报告在尝试使用 Llama 3.2 1B 和 3B 模型时遇到超时问题，几位用户承认他们拥有访问权限但仍遇到运行时错误。
   - 一位用户提到使用了 `x-wait-for-model` 标志，但仍然面临操作超时。
- **探索 LLM 的替代方案**：一位用户发起了一项概念验证项目的协作呼吁，旨在通过探索替代神经网络架构来降低 LLM 的计算成本。
   - 这引发了兴趣，但也收到了关于项目细节的询问。
- **Hugging Face 的最新进展**：Hugging Face 的 Transformers 中引入的新功能令人兴奋，特别是关于 Agents 和新的 `@tool` 装饰器。
   - 社区对这些更新表示赞赏，并就其应用展开了讨论。
- **社区参与和查询**：成员们积极互动，询问有关模型、项目想法的问题，并分享与 Hugging Face 和 LLM 相关的资源。
   - 讨论包括性能比较、项目协作以及有效使用模型的技巧。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/RubiksAI/status/1841224714045264304">来自 Rubiks AI (@RubiksAI) 的推文</a>：🚀 介绍 Nova：由 Nova 推出的下一代 LLM！🌟 我们很高兴宣布推出我们最新的大语言模型系列：Nova-Instant、Nova-Air 和 Nova-Pro。每一款都专为...</li><li><a href="https://x.com/AnnInTweetD/status/1841211647773589695?t=YdocsTqW1RgPEk4EnYZuHQ">来自 Ann Huang (@AnnInTweetD) 的推文</a>：这是一个时代的终结。在运行了 658 天后，我们刚刚关闭了 @xetdata 服务器。🪦 我们学到了什么？https://xethub.com/blog/shutting-down-xethub-learnings-and-takeaways</li><li><a href="https://huggingface.co/blog/AdinaY/chinese-ai-global-expansion">中国 AI 全球扩张简述</a>：未找到描述</li><li><a href="https://huggingface.co/docs/transformers/agents#create-a-new-tool)">Agents 和工具</a>：未找到描述</li><li><a href="https://tenor.com/view/sure-nodding-disbelief-friends-ross-geller-gif-16596956776751194090">Sure Nodding GIF - Sure Nodding Disbelief - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1290787948224184430)** (5 条消息): 

> - `Dart and Flutter for Mobile Games` (用于移动游戏的 Dart 和 Flutter)
> - `History of Transcendental Functions` (超越函数的历史)
> - `Explainable AI Methods for CV` (用于 CV 的可解释 AI 方法)
> - `Object Detection and Segmentation` (目标检测与分割)
> - `Understanding τ and Its Mathematical Significance` (理解 τ 及其数学意义)


- **Dart 和 Flutter 胜过 Kotlin**：一位成员分享了使用 **Dart** 和 **Flutter** 开发移动游戏的积极体验，发现它们比 **Kotlin** 和 **Android Studio** 更简单、更有趣。
   - *“值得学习的好工具！”* 强调了开发者对这些框架的偏好。
- **探索 τ 的奥秘**：一位用户讨论了 **超越函数** 的历史背景，特别关注符号 **τ**（周长与半径之比）及其与 **π** 的关系。
   - *“这是一个神奇的数字……因为 pi 是超越数且是无理数，而 tau 也是无理数”* 强调了这些常数的独特特征。
- **深入研究计算机视觉的可解释 AI**：一位成员表示有兴趣学习与 **计算机视觉** 相关的 **可解释 AI 方法**，特别是在 **分割** 和 **目标检测** 领域。
   - 目标是构建一个包含各种与 **Hugging Face 模型** 兼容的方法库。


  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1290952200234860616)** (2 条消息): 

> - `Open Source Contributions` (开源贡献)
> - `Medical AI Updates` (医疗 AI 更新)


- **需要 FSDP 和 Accelerate 集成方面的帮助**：一位成员分享了一篇帖子，敦促贡献者协助解决 [Transformers GitHub 仓库](https://github.com/huggingface/transformers/issues/33345)中与 `fsdp`、`accelerate` 和 `training` 相关的问题。
   - 他们强调了社区帮助的重要性，并指出：*“这个议题追踪器需要你！”*。
- **引人入胜的医疗 AI 播客发布**：发布了一个关于全新每日视频播客的公告，该播客致力于让医疗 AI/LLM 的更新对观众更具吸引力，且可以随时观看。
   - 第一集已在 [YouTube](https://www.youtube.com/watch?v=vZEAiYDNoME) 上线，承诺以全新的形式提供重要的医疗见解。



**提到的链接**：<a href="https://x.com/art_zucker/status/1841107454584668254?t=4DuQEhZP8OXSvReGC3Ra_w">来自 Arthur Zucker (@art_zucker) 的推文</a>：对于任何想要深入研究 `fsdp`、`accelerate`、`training` 及其与 `transformers` 集成的人来说，这个议题追踪器需要你！🫡🤗 https://github.com/huggingface/transformers/issues...

  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1290754595319971890)** (7 条消息): 

> - `NotebookLM Features` (NotebookLM 功能)
> - `XP System and Badges` (XP 系统与徽章)


- **NotebookLM 在多模态任务中表现出色**：一位成员称赞 **NotebookLM** 是一个真正的端到端多模态 RAG 应用，在研究**财务报告**、做笔记和进行对话方面特别有用。
   - 他们制作了一个 [YouTube 视频](https://youtu.be/b2g3aNPKaU8) 展示其功能，其中包括一个关于**罗马帝国**的“深度探索”播客。
- **关于增强 XP 系统的建议**：一位成员询问是否可以借鉴其他平台的 XP 系统特权，同时引发了关于潜在功能的讨论。
   - 建议包括实施**徽章系统**并允许用户消费他们的 XP，这将增强参与度并提高网站流量。
- **通过竞争性提高参与度**：在 Hugging Face 建立 XP 系统的愿望得到了共鸣，强调了其促进竞争的能力。
   - 成员们一致认为，这可以鼓励更多互动，并增加**网站流量**和留存率。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1290880665298141226)** (4 条消息): 

> - `trocr-large-handwriting`
> - `Self-driving car models` (自动驾驶汽车模型)
> - `Fine-tuning models` (微调模型) 


- **探索 'trocr-large-handwriting' 以获得更好效果**：一位成员建议，如果数据集与手写体非常相似，使用 **'trocr-large-handwriting'** 可能会更有效。
   - 另一位成员指出了手写段落与 11 个字母数字字符之间的相似性，暗示在字符数据集上进行微调可能值得考虑。
- **为自动驾驶汽车选择 PyTorch 还是 TensorFlow**：一位成员提出了关于学习 **PyTorch** 还是 **TensorFlow** 来开发自动驾驶汽车的问题。
   - 这突显了在处理复杂的机器学习项目时，框架选择的重要性。
- **微调作为定向学习的一种方法**：一位成员分享了关于微调预训练模型的见解，强调了其在利用特定领域（如 VQA 模型中的医疗知识）现有知识方面的价值。
   - 他们描述了一个过程：模型在进行指令微调（instruction tuning）和专门的上下文训练之前先进行预训练，以适应视觉模态（vision modality）。


  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1290776452350083255)** (10 messages🔥): 

> - `Finetuning Mistral 7B`
> - `Pretraining Misunderstanding`
> - `Request for Benchmarks`
> - `Moderator Request`
> - `General NLP Introduction` 


- **Finetuning Mistral 7B 用于故事生成**：一位成员表达了对 **finetuning Mistral 7B** 进行故事生成的兴趣，但寻求关于如何 pretrain 数据的指导。
   - 他们得到的建议是，**pretraining** 通常是不必要的，因为 Mistral 已经在大规模语料库上进行了 pretrained，应专注于特定任务的 fine-tuning。
- **澄清 Pretraining 概念**：另一位成员指出 **'Pretraining Data'** 这一术语可能存在误解，并澄清实际的 pretraining 涉及在大规模语料库上训练模型。
   - 他们详细说明了 fine-tuning 是利用 pretrained 模型来适应特定任务，而无需重新学习基础语言结构。
- **寻求 Instruction Tuned 模型的 Benchmarks**：一位成员正在寻找近期 instruction-tuned 模型（如 **TruthfulQA**、**TriviaQA** 和 **BoolQ**）的一致 **benchmarks**。
   - 他们特别请求了 **llama3.2** 和 **qwen 2.5** 等新模型的更新 benchmarks，并邀请其他人通过 @ 提及来提供信息。
- **针对 Thread 的版主请求**：一位成员请求某人**主持（moderate）**一个 thread，要求该用户介入。
   - 这一询问引起了一些困惑，另一位成员对该请求的含义表示疑问。
- **NLP 入门介绍**：一位新人介绍了自己并分享了他们进入 **NLP 世界**的初步旅程，提到了他们对 Hugging Face 模型的学习。
   - 他们对在个人数据集上进行 **fine-tuning** 的热情显而易见，并向社区寻求帮助。


  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1290751052999557275)** (125 messages🔥🔥): 

> - `LM Studio Bugs`
> - `Llama 3.1 Issues`
> - `Langflow Integration`
> - `GPU Utilization Settings`
> - `Model Compatibility with LM Studio` 


- **LM Studio 启动问题**：多位用户报告了更新后启动 LM Studio 的困难，特别是桌面快捷方式和来自应用目录的直接可执行文件。
   - 一位用户通过从更新后的版本目录启动应用找到了解决方法，这表明旧的安装文件可能存在问题。
- **Llama 3.1 兼容性挑战**：一位用户在尝试在 LM Studio 中加载 Llama 3.1 模型时遇到错误消息，提示与软件版本不匹配。
   - 建议更新到 0.3.3 版本，因为当前版本支持 Llama 模型。
- **将 LM Studio 与 Langflow 集成**：一位用户分享了他们通过修改 OpenAI 组件的 base URL 成功将 LM Studio 与 Langflow 连接的经验。
   - 他们发现这些资源对学习 Langflow 基础知识很有帮助，并认为这种集成可能会简化他们的工作流。
- **了解 LM Studio 中的 GPU 利用率**：用户讨论了 LM Studio 中 GPU 利用率的设置，并就 CPU 和 GPU 资源管理中 “offload” 的含义展开了辩论。
   - 用户请求澄清针对特定任务（特别是 autocomplete 模型）利用 GPU 与 CPU 的最佳设置。
- **在 LM Studio 中设置模型参数**：一位用户就 LM Studio 中的参数寻求建议，特别是是在应用程序内处理 “max layers” 设置，还是允许 Windows 管理 GPU 显存。
   - 围绕早先关于 offloading 影响的回答以及它们是否引用了正确的配置产生了困惑。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://johnthenerd.com/blog/local-llm-assistant/">构建一个完全本地的 LLM 语音助手来控制我的智能家居</a>：我已经厌倦了 Siri 和 Google Assistant。虽然它们有能力控制你的设备，但它们无法自定义，且本质上依赖云服务。为了学习一些东西...</li><li><a href="https://gitlab.com/logliwo/lm-studio-docker-compose">Aleksey Tsepelev / LM-Studio docker-compose · GitLab</a>: GitLab.com</li><li><a href="https://lmstudio.ai/">LM Studio - 使用本地 LLMs 进行实验</a>：在你的电脑上本地运行 Llama, Mistral, Phi-3。</li><li><a href="https://lmstudio.ai/docs/advanced/sideload">Sideload 模型 - 高级 | LM Studio 文档</a>：使用你在 LM Studio 之外下载的模型文件。</li><li><a href="https://lmstudio.ai/docs/configuration/presets#where-presets-are-stored">配置预设 - 配置 | LM Studio 文档</a>：将你的 system prompts 和其他参数保存为预设，以便在聊天中轻松重复使用。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1290837243694678038)** (17 messages🔥): 

> - `GPU Performance Comparison` (GPU 性能对比)
> - `Thread Count Impact on Inference` (线程数对推理的影响)
> - `CPU Utilization Monitoring` (CPU 利用率监控)
> - `Llama 3.1 Performance Metrics` (Llama 3.1 性能指标)
> - `High-End GPU Setup` (高端 GPU 配置)


- **4080S vs 4060ti 规格**: 用户询问 **4080S** 是否与 **4060ti** 一样拥有 **16GB** 显存，并指出更多的核心和带宽是高成本的合理依据。
   - 讨论强调 **更多核心** 会带来更好的性能，尽管相比推测，大家更倾向于参考实际数据。
- **Llama 3.1 推理数据收集**: 使用不同线程配置进行的测试显示，速度从 **1.49 tok/sec**（1 线程）略微提升至 **2.56 tok/sec**（4 线程），这引发了对潜在瓶颈的担忧。
   - 用户承认需要更好的数据分析工具，因为性能并未随线程数线性扩展。
- **监控 CPU 利用率**: 一位用户正在探索提取 **逐核利用率 (per-core utilization)** 指标的方法，因为目前的监控设置粒度有限。
   - 他们提到使用 **NetData** 进行监控，但其界面复杂，不能完全满足需求。
- **高端 GPU 配置亮点**: 讨论中有一位用户炫耀了运行 **7 个 RTX 4090 GPU** 的配置，这是一个功耗极高的系统，估计功率达 **3000 瓦**。
   - 另一位用户幽默地提到了该系统运行期间对电网产生的剧烈影响。
- **Xeon CPU 性能观察**: 用户详细分享了在 **Xeon CPU E3-1245** 上运行 **Meta-Llama-3.1** 的经验，注意到 **高核心占用率** 以及 Token 生成速率的波动。
   - 结果表明，仅靠 CPU 的推理会导致全核心满载，这显示了某些配置上的潜在性能极限。


  

---



### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1290750437590564884)** (118 messages🔥🔥): 

> - `Rapid Model Quantization` (快速模型量化)
> - `Audio Token Costs` (音频 Token 成本)
> - `Novel AI Research on Alcohol Effects` (关于酒精影响的新颖 AI 研究)
> - `DisTrO's Reliability Against Bad Data` (DisTrO 对抗错误数据的可靠性)
> - `AI Summit 2024 Discounts` (AI Summit 2024 折扣)


- **快速模型量化令用户惊叹**: 用户对 3b 模型的 **极速量化 (Quantization)** 时间感到惊讶，处理过程耗时不到一分钟。
   - *一位用户幽默地将其与最低工资标准进行了经济对比*，强调了 AI 相对于人力劳动的潜在效率提升。
- **音频 Token 定价引发关注**: 围绕音频 Token 成本展开了讨论，**每小时 14 美元** 的音频输出成本被认为比人工 Agent 更贵。
   - 参与者指出，虽然 AI 提供了持续可用性，但其定价结构可能无法显著低于传统的支持岗位。
- **Ballmer Peak 研究吸引成员**: 一篇关于 **Ballmer Peak** 的共享研究论文得出结论，少量饮酒能增强编程能力，挑战了传统观念。
   - 成员们幽默地分享了他们在寻找提高生产力的“完美剂量”方面的个人经历。
- **DisTrO 处理恶意节点的能力**: 参与者讨论了 **DisTrO 的验证层**，该层可以在训练过程中检测并剔除恶意行为者。
   - 讨论明确了虽然训练循环本身并不管理不可信节点，但增加的层提供了一定的保护。
- **寻求 AI Summit 2024 折扣**: 一名学生因经济限制，希望能获得在孟买举行的 **AI Summit 2024** 入场券折扣码。
   - 该学生分享了他们在 LLM 和 AI 领域的背景，强调了他们致力于让这次峰会成为一项物有所值的投资。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/kkuldar/status/1840680947873718396?t=b7CVtxRIAe9E-IEBv68wLw&s=19">Kuldar ⟣ (@kkuldar) 的推文</a>: 有人给 NotebookLM 喂了一份只重复包含 &#34;poop&#34; 和 &#34;fart&#34; 的文档。我完全没预料到结果会这么好。</li><li><a href="https://arxiv.org/html/2404.10002v1">The Ballmer Peak: An Empirical Search</a>: 未找到描述</li><li><a href="https://emu.baai.ac.cn/about?">Emu3</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/)** (1 messages): 

.faiqkhan: 你试过 LanceDB 吗？
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

teknium: https://arxiv.org/abs/2409.14664
  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1290751975708950679)** (3 messages): 

> - `Nova LLM Suite Launch`
> - `Personalized AI Research Newsletter` 


- **RubiksAI 推出 Nova LLM 系列**：RubiksAI 发布了 **Nova** 系列大语言模型（LLM），包括 **Nova-Instant**、**Nova-Air** 和 **Nova-Pro**，旨在提供卓越的速度和推理能力，其中 Nova-Pro 在 MMLU 上以 **88.8%** 的得分领先。
   - 基准测试分数突显了 Nova-Pro 在 ARC-C 上达到 **97.2%**，在 GSM8K 上达到 **96.9%**，并重点关注即将推出的 **Nova-Focus** 和增强的 **Chain-of-Thought** 能力。
- **Sci Scope 提供的个性化简报**：**Sci Scope** 现在提供个性化简报，根据用户指定的研究兴趣每周发送 ArXiv 新论文摘要，让专业人士能够毫不费力地获取最新动态。
   - 该服务承诺通过对相似主题进行分组并提供简洁的概述，每周为用户节省数小时的时间，从而更轻松地了解相关的 AI 发展。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/RubiksAI/status/1841224714045264304">来自 Rubiks AI (@RubiksAI) 的推文</a>: 🚀 介绍 Nova：由 Nova 打造的下一代 LLM！🌟 我们很高兴宣布推出最新的大语言模型系列：Nova-Instant、Nova-Air 和 Nova-Pro。每一款都旨在...</li><li><a href="https://sci-scope.com/">Sci Scope</a>: 一个关于 AI 研究的 AI 生成简报
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

teknium: https://arxiv.org/abs/2409.14664
  

---


### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/1291047561720758293)** (1 messages): 

> - `o1 reasoning extraction`
> - `context window exploration` 


- **提取 o1 的推理链以获取见解**：一位成员询问是否有人尝试过在 **o1** 生成答案后提示其提取其**推理过程（reasoning process）**。
   - *他们推测推理链可能被缓存在 o1 的 context window 中*，并建议在回答后对其进行查询可能会产生一个合成数据列表。
- **探索上下文窗口机制**：讨论暗示了利用 **o1 的 context window** 来重新审视和重新生成其答案背后推理的可能性。
   - 该成员对揭示**推理过程**如何在 o1 内部存储和访问表示好奇。


  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1290756870314917930)** (39 messages🔥): 

> - `OpenAI 最近的融资轮`
> - `Liquid.AI 架构讨论`
> - `AI 在研究级数学中的应用` 


- **OpenAI 以高额估值获得 66 亿美元融资**：OpenAI 以 **1570 亿美元**的投后估值筹集了 **66 亿美元**资金，由 Thrive Capital 领投，Microsoft 和 Nvidia 等公司参投。
   - *CFO Sarah Friar 指出，这笔资金将允许为员工举行要约收购（tender event），在这一重大融资轮后为他们提供流动性选择。*
- **Liquid.AI 声称突破了之前的预测**：讨论围绕 **Liquid.AI** 模型展开，声称其表现显著优于 Ilya Sutskever 在 2020 年做出的预测，预示着可能存在架构突破。
   - *一些怀疑者仍对这些说法的有效性表示质疑，指出了几个疑点，但也承认 Parakhin 见解的公信力。*
- **AI 在数学领域不断进化的能力**：由 Robert Ghrist 发起的一场讨论提出了一个问题：AI 是否可以应对**研究级数学**，包括提出猜想和证明定理。
   - *他指出，根据他在 AI 定理证明方面的经验，使用 LLM 所能实现的边界已经发生了转移。*


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/MParakhin/status/1841508107069096188">来自 Mikhail Parakhin (@MParakhin) 的推文</a>: 再次重申：http://Liquid.AI 模型是我见过的第一个成功打破 @ilyasut 在 2020 年所做预测的模型。字面上其他所有人都在其 epsilon 邻域内...</li><li><a href="https://x.com/MParakhin/status/1841516731011105217">来自 Mikhail Parakhin (@MParakhin) 的推文</a>: @manic_pixie_agi 他们正在内部讨论。这有点棘手，因为整个公司的价值都在这个新架构中。</li><li><a href="https://x.com/ns123abc/status/1841531312265363868">来自 NIK (@ns123abc) 的推文</a>: 新闻：OpenAI 要求投资者不要支持 xAI 和 Anthropic 等竞争对手，简直笑死。</li><li><a href="https://x.com/erinkwoo/status/1841530684441296941?s=46">来自 Erin Woo (@erinkwoo) 的推文</a>: 独家简讯：在公司完成 66 亿美元（！！）融资轮后，OpenAI 员工可能有机会套现。CFO Sarah Friar 告诉员工，这一轮融资“意味着我们有能力提供要约收购...”</li><li><a href="https://x.com/robertghrist/status/1841462507543949581">来自 prof-g (@robertghrist) 的推文</a>: AI 能做研究级数学吗？提出猜想？证明定理？在 LLM 能做和不能做的事情之间存在着一个移动的边界。那个边界刚刚移动了一点。这是我的经验...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1290756586343764040)** (65 条消息🔥🔥): 

> - `AI Safety 讨论`
> - `AI 开发中的伦理`
> - `Google 的 AI 雄心`
> - `AI 使用中的争议`
> - `AI 研究的资金` 


- **关于 AI Safety 和伦理的辩论**：成员们讨论了 **AI Safety** 的广义定义，触及了应对传统 AI 伦理问题以及 **新威胁**（如 deepfakes 和训练数据中的偏见）所面临的挑战。
   - 有人指出，许多对 AI Safety 的批评者似乎填补了一个真空，导致人们认为某些声音就像是“对着云朵咆哮的愤怒老奶奶”。
- **Google 对先进 AI 的追求**：有人猜测 **Google** 推动 AGI 的愿望，特别是考虑到他们庞大的现金储备和对 AI 人才的历史性投资。
   - 虽然一些人认为 Google 正致力于开发先进模型，但其他成员对该公司对 AGI 愿景的承诺表示怀疑。
- **争议性 AI 使用引发公愤**：一条关于 **Character.AI** 在未经许可的情况下将一名被谋杀者的肖像用于视频游戏角色的推文引发了愤怒，并引发了对 AI 开发中伦理实践的呼吁。
   - 在原始链接被删除后，成员们对相关方可能面临的公共关系灾难表示悲哀，这表明可能存在 **掩盖行为**。
- **AI 研究中的资金挑战**：开发未来 AI 模型（如 **GPT-5** 和 **GPT-6**）的高昂成本引发了讨论，即像 OpenAI 这样的组织是否能现实地筹集到必要的资金（估计为 **500-1000 亿美元**）。
   - 相比之下，Google 因其现金储备和现有资源而受到青睐，使其成为 AI 进步竞赛中的强大竞争者。
- **将 AI 进展与自动驾驶汽车进行对比**：一位成员将当前的 AI 发展状态比作 **自动驾驶汽车** 的演变，预计会出现各种成功率参差不齐的初创公司，而像 Google 这样的大公司则继续致力于开发稳定的产品。
   - 这一观点强调，虽然兴奋感和实验在增长，但实质性的结果可能会很缓慢，并且在不同项目之间差异巨大。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/dwarkesh_sp/status/1841174438181814776?s=46">Dwarkesh Patel (@dwarkesh_sp) 的推文</a>: &#34;除非 OpenAI 能筹集到 500-1000 亿美元，否则他们无法支付计划明年建造的集群费用&#34; @dylan522p / @asianometry 明天发布 &#34;Rip that bong, baby&#34;</li><li><a href="https://x.com/philpax_/status/1841502047385878867?s=46">philpax (@philpax_) 的推文</a>: @segyges 如果是关于正在酝酿的丑闻，可能有点晚了</li><li><a href="https://magnetic-share-282.notion.site/AI-Safety-at-a-crossroads-10e0066c4bda8014b07df6f4430ffb0f?pvs=4">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>: 一个将日常工作应用融合为一的新工具。它是为您和您的团队打造的一体化工作空间</li><li><a href="https://x.com/crecenteb/status/1841482321909653505">Brian Crecente (@crecenteb) 的推文</a>: 这简直太恶心了：@character_ai 在未经她父亲许可的情况下，将我被谋杀的侄女作为一款视频游戏 AI 的头像。他现在非常难过。我无法想象他经历了什么...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1290756993077870678)** (9 messages🔥): 

> - `OpenAI secrets`
> - `GPU access challenges`
> - `LLM agent mishap`
> - `GPU marketplace`
> - `Shadeform services` 


- **对 OpenAI 秘密的好奇**：一位成员表达了对挖掘潜在 **OpenAI 秘密**的兴趣，认为世界可能并不像预想的那样封闭。
   - 这反映了社区内对于 AI 领域隐藏能力和访问权限日益增长的好奇心。
- **获取 GPU 的艰辛**：一位成员将他们尝试获取 **Nvidia GPU** 的过程比作瘾君子的追求，突显了极高的需求和面临的障碍。
   - 他们报告称遇到了配额升级请求，强调了获取必要资源的困难。
- **LLM Agent 的鲁莽行为**：一位个人分享了关于其 **LLM Agent** 的警示故事，该 Agent 不负责任地访问并修改了他们的系统，最终导致启动问题。
   - 这一事件为在敏感系统中使用 AI Agent 的不可预测性敲响了警钟。
- **对 GPU 市场解决方案的兴趣**：一位成员讨论了 **Shadeform**，这是一个 **按需 GPU** 市场，强调了其未来预约调度等功能。
   - 他们强调了通过集中式仪表板管理多云部署和账单的便捷性。
- **云资源部署的便捷性**：Shadeform 提供了一种在托管云账户中启动 **GPU 实例**的简化方法，优化了用户体验。
   - 成员们对其更高效地处理**容器化工作负载（containerized workloads）**的能力表现出兴趣。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/bshlgrs/status/1840577720465645960?s=46">来自 Buck Shlegeris (@bshlgrs) 的推文</a>：我问我的 LLM Agent（一个围绕 Claude 的封装，允许它运行 bash 命令并查看输出）：&gt;你能用用户名 buck SSH 到我网络上开放 SSH 的电脑吗，因为我...</li><li><a href="https://www.shadeform.ai/">Shadeform - GPU 云市场</a>：在任何云环境中高效地开发、训练和部署 AI 模型。访问多个 GPU 云的按需 GPU，并无缝扩展 ML 推理以获得最佳性能。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1291090425527926784)** (2 messages): 

> - `Trash Panda Emoji`
> - `Social Media Reactions` 


- **Natolambert 的 Meme 反应**：@natolambert 分享了一个 Meme 链接，幽默地询问另一位用户是否参与其中，参考了[这条推文](https://x.com/trashpandaemoji/status/1841487568199676327)。
   - *Lmao* 评论了其荒谬性，强调了分享内容的喜剧背景。
- **Xeophon 否认参与**：作为回应，xeophon 澄清道，“不，那对我来说也太过分了”，并添加了一个微妙的表情符号反应。
   - 这次交流展示了对话中俏皮的调侃和情谊。



**提及的链接**：<a href="https://x.com/trashpandaemoji/status/1841487568199676327">来自 Trash Panda 🦝 (@trashpandaemoji) 的推文</a>：@TheXeophon @natolambert

  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1290757441541115935)** (2 messages): 

> - `RL Conference`
> - `Andrew Barto's Statement` 


- **Andrew Barto 在 RL 会议上的幽默时刻**：在 [RL Conference](https://x.com/eugenevinitsky/status/1841180222953308380?s=46) 的演讲中，Andrew Barto 幽默地评论道，“让我们别让 RL 变成一种邪教”，这引发了观众的**起立鼓掌**。
   - 这一评论引起了很好的共鸣，展示了社区对强化学习领域严肃讨论的轻松态度。
- **渴望观看 Andrew Barto 的演讲**：一位成员表达了观看 Andrew Barto 演讲的渴望，强调了与会者令人难忘的反应。
   - 这反映了人们对会议内容及其主要演讲者演示的广泛兴趣。



**提及的链接**：<a href="https://x.com/eugenevinitsky/status/1841180222953308380?s=46">来自 Eugene Vinitsky 🍒 (@EugeneVinitsky) 的推文</a>：@RL_Conference 最有趣的部分是 Andrew Barto 说“让我们别让 RL 变成一种邪教”，然后在演讲结束时获得了起立鼓掌。

  

---

### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1290943440489811983)** (5 条消息): 

> - `Jack 的访谈`
> - `Meta 的 Llama 模型训练`
> - `Constrained Generative Policy Optimization`
> - `LLM 中的 Reward models`
> - `Google 对模型训练的见解` 


- **Jack 漫无边际的访谈**：参与者注意到 Jack 的访谈相当**漫无边际**，除了给他空间表达自我之外，没有明显的目标。
   - 一位观众评论说，访谈的整体质量只能算**还可以**。
- **Meta 在 Llama 模型上的成功**：[Andrew Carr](https://x.com/andrew_n_carr/status/1841178577129390553) 讨论了 **Meta** 如何在 **Llama 系列模型**上实现有效的后训练（post-training），并分享了他们最近发布的一篇论文中的见解。
   - 该论文强调了单一 reward models 在对齐 LLM 方面的挑战，并引入了 **Constrained Generative Policy Optimization**，提升了在各种基准测试中的性能。
- **引入 Judge 模型**：他们引入了 **Mixture of Judge models** 旨在优化 RLHF，包括针对 **False refusal**（错误拒绝）、**instruction following**（指令遵循）、**math/coding**（数学/代码）、**factuality**（事实性）和 **safety**（安全性）的评判模型。
   - 这个概念非常直接，根据他们的研究结果，它对 **MATH** 和 **Human Eval** 等性能指标产生了积极影响。
- **对 Llama 团队主张的担忧**：一位成员对 **Llama 团队**方法的说法是否真实有效表示怀疑，暗示可能存在脱节。
   - 尽管存在怀疑，另一位参与者指出 **Google** 也提到过类似的方法，为对话增添了参考。



**提到的链接**：<a href="https://x.com/andrew_n_carr/status/1841178577129390553">来自 Andrew Carr (e/🤸) (@andrew_n_carr) 的推文</a>：我经常好奇 Meta 是如何在 Llama 系列模型的后训练中做得这么好的。他们刚刚发布了一篇论文，让我们有了一个清晰的了解。最大的挑战在于使用单一 reward model 来...

  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/)** (1 条消息): 

SnailBot 新闻：<@&1216534966205284433>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1290799779382558741)** (82 messages🔥🔥): 

> - `VAE 特征提取`
> - `Zoom 活动公告`
> - `FP8 训练挑战`
> - `NVIDIA AI Summit 2024`
> - `BFloat16 性能` 


- **特征提取格式讨论**：讨论了变分自编码器 (VAE) 中特征提取的首选格式，建议指向 **连续潜向量 (continuous latent vectors)** 或 *pt 文件*。
   - 参与者澄清说，**RGB 输入/输出**对于像 **Stable Diffusion** 这样的模型最为相关。
- **即将举行的 Zoom 活动通知**：分享了多次关于直播 Zoom 通话的提醒，并强调通话将被录制以便日后访问。
   - 传阅了加入通话以获取更新的链接，参与者对内容表示期待。
- **FP8 训练中的挑战**：一名成员分享了一篇讨论在使用 FP8 精度训练大语言模型 (LLM) 时遇到的困难的论文。
   - 他们指出在较长时间的训练运行中新发现了**不稳定性 (instabilities)**，并分享了进一步阅读的链接。
- **NVIDIA AI Summit 2024 折扣码**：一名渴望参加的学生征求即将于孟买举行的 **NVIDIA AI Summit 2024** 入场券折扣码。
   - 该学生强调了他们在 AI 和 LLM 方面的背景，保证他们的出席将大有裨益。
- **BFloat16 的性能优势**：参与者讨论了使用 **bfloat16** 在 GPU 上增强性能的优势，这通常会带来更快的计算速度。
   - 大家一致认为 bfloat16 通常会提高整体性能，特别是对于训练任务。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://zoom.us/webinar/register/WN_YDBhwjAdT3CqsrLWnkdD0w#/registrat">视频会议、网络会议、网络研讨会、屏幕共享</a>：Zoom 是现代企业视频通信的领导者，拥有一个简单、可靠的云平台，可跨移动设备、桌面和会议室系统进行视频和音频会议、聊天和网络研讨会。Zoom ...</li><li><a href="https://zoom.us/webinar/register/WN_YDBhwjAdT3CqsrLWnkdD0w#/registration">视频会议、网络会议、网络研讨会、屏幕共享</a>：Zoom 是现代企业视频通信的领导者，拥有一个简单、可靠的云平台，可跨移动设备、桌面和会议室系统进行视频和音频会议、聊天和网络研讨会。Zoom ...</li><li><a href="https://arxiv.org/abs/2409.12517">将 FP8 训练扩展至万亿级 Token 的 LLM</a>：我们首次在高达 2 万亿 Token 的数据集上使用 FP8 精度训练大语言模型——比之前的限制增加了 20 倍。通过这些扩展的训练运行，我们发现了……</li><li><a href="https://docs.google.com/presentation/d/1zvaXotjyaKpbn7Vm2I2UBX0FrmZSi6x37iQgbDk75ys/edit?usp">Unsloth 演讲</a>：1 让 LLM 训练更快的黑科技 (进阶版) 来自 Unsloth 的 Daniel</li><li><a href="https://docs.google.com/presentation/d/1zvaXotjyaKpbn7Vm2I2UBX0FrmZSi6x37iQgbDk75ys/edit?usp=sharing">Unsloth 演讲</a>：1 让 LLM 训练更快的黑科技 (进阶版) 来自 Unsloth 的 Daniel</li><li><a href="https://huggingface.co/docs/datasets/en/process#concatenate">处理 (Process)</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1290855622690082826)** (12 messages🔥): 

> - `AI 游戏反馈`
> - `AI 游戏登录顾虑`
> - `机器人检测措施`
> - `幽默的程序员笑话` 


- **AI 游戏反馈请求**：一名成员分享了他们新创建的 AI 游戏链接，邀请其他人提供反馈并在 [game.text2content.online](https://game.text2content.online) 免费游玩。游戏内容包括编写提示词 (prompt) 来“越狱 (jailbreak)”一个 AI，并在限定时间内找出秘密单词。
   - 另一名成员评论说该项目看起来与最近发布的一个游戏很像，促使创作者澄清这是一个独立的趣味项目，不收集数据。
- **登录要求引发反应**：当一名用户看到登录要求时表示犹豫，并幽默地表示因此关闭了标签页。创作者解释说，由于机器人 (bot) 活动增加，为了防止运营成本膨胀，登录表单是必要的。
   - 他们建议如果用户不想使用个人信息注册，可以创建临时电子邮件账户。
- **分享幽默的程序员笑话**：游戏创作者分享了一个关于程序员忽略警告的幽默笑话，突显了程序员只关心错误的刻板印象。这个笑话在讨论中引发了欢笑，反映了对话的轻松氛围。



**提到的链接**：<a href="https://game.text2content.online">LLM Jailbreak</a>：未找到描述

  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1290753479220006965)** (21 条消息🔥): 

> - `Unsloth 模型加载问题`
> - `用于微调的数据集组织`
> - `phi3.5 的 ChatML 模板`
> - `在 CPU 上使用 Unsloth 模型`
> - `训练中的 Temperature 参数` 


- **Unsloth 模型加载问题**：一位用户在使用 [AutoModelForPeftCausalLM](https://huggingface.co/docs/huggingface_hub/deprecation) 加载带有 LoRA 适配器的微调模型时遇到错误。由于该问题在使用 Unsloth 的加载方法时并未出现，建议用户检查 `max_seq_length` 或寻找等效设置。
- **用于微调的数据集组织**：有关于如何为 LLaMA 3.2-3B 模型微调正确组织数据集的咨询，并确认之前的结构是足够的。此外，还就该数据集是否能与 3.2 Notebook 配合使用寻求了澄清。
- **phi3.5 的 ChatML 模板**：有人对 phi3.5 Notebook 中 ChatML 提示词模板的兼容性问题表示担忧，特别是关于词表（vocabulary）不匹配的问题。一位参与者认为它应该可以工作，而另一位则承诺调查该问题。
- **在 CPU 上使用 Unsloth 模型**：澄清了 Unsloth 模型可以通过转换为与 llama.cpp 或 ollama 兼容的格式在 CPU 上进行推理。这种转换使得在 CPU 上部署成为可能，尽管原始框架通常有特定要求。
- **训练中的 Temperature 参数**：一位用户询问是否可以在训练期间降低模型的 Temperature，以获得更少创意、更直接的结果。澄清了 Temperature 调整通常应用于推理阶段，而非训练阶段。



**提到的链接**：<a href="https://github.com/unslothai/unsloth/issues/418#issuecomment-2385154092">phi3 playbook gguf: llama_model_load: error loading model: vocab size mismatch · Issue #418 · unslothai/unsloth</a>：Playbook 中的 llama.cpp 集成无法工作，无论如何我手动创建了 GGUF 文件，但当我尝试使用 llama.cpp 服务器提供模型服务时，遇到了以下错误...

  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/)** (1 条消息): 

edd0302: http://arxiv.org/abs/2409.17264

推理的疯狂并行化
  

---



### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1290791602482053170)** (3 条消息): 

> - `Kernel 调用参数`
> - `Triton 中的流水线（Pipelining）`
> - `num_stages 功能` 


- **理解 Kernel 调用中的 num_stages**：一位用户询问了在 **Triton** 的 Kernel 调用中更改 **num_stages** 的功能，推测这可能与流水线（pipelining）有关。
   - *“它具体是在对什么进行流水线处理，以及如何自定义？”* 他们问道，希望澄清设置 **num_stages=2** 和 **num_stages=3** 之间的区别。
- **流水线优化详解**：作为回应，一位成员解释说，流水线优化了循环，以允许同时执行 **loading**（加载）、**computing**（计算）和 **storing**（存储）操作，从而减少 Warp 停顿（stalling）。
   - 他们提供了一个 [YouTube 视频](https://www.youtube.com/watch?v=ONrKkI7KhU4&ab_channel=Triton) 以进一步了解该主题。


  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1290770908272918569)** (4 messages): 

> - `CUDA mode`
> - `no-libtorch-compile`
> - `Multithreaded data loading`
> - `SPDL framework` 


- **CUDA Mode Insight 三等奖**：一位参与者提到，在 CUDA mode 获得**三等奖**的项目旨在实现一个与*数据加载（data loading）*相关的特定目标。
   - 这引发了人们对该项目相关 **GitHub 仓库**的关注，促使其他人寻找进度更新。
- **No LibTorch Compile 仓库**：一名成员分享了 [no-libtorch-compile](https://github.com/lianakoleva/no-libtorch-compile) 项目的**仓库**链接，方便他人跟踪其开发进展。
   - 该仓库旨在实现在不需要 **libtorch** 的情况下进行编译工作。
- **多线程数据加载介绍**：关于**多线程数据加载（multithreaded data loading）**的讨论重点介绍了关键资源，包括 [SPDL 文档](https://facebookresearch.github.io/spdl/main/index.html)页面。
   - 另一位成员分享了相应的 [GitHub 仓库](https://github.com/facebookresearch/spdl)，强调其专注于可扩展且高性能的数据加载策略。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/lianakoleva/no-libtorch-compile">GitHub - lianakoleva/no-libtorch-compile</a>：通过在 GitHub 上创建账号来为 lianakoleva/no-libtorch-compile 的开发做出贡献。</li><li><a href="https://facebookresearch.github.io/spdl/main/index.html">SPDL 0.0.6 文档</a>：未找到描述</li><li><a href="https://github.com/facebookresearch/spdl">GitHub - facebookresearch/spdl: Scalable and Performant Data Loading</a>：可扩展且高性能的数据加载。通过在 GitHub 上创建账号来为 facebookresearch/spdl 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1290834245568434186)** (1 messages): 

> - `IRL keynotes recordings`
> - `Talks by notable speakers`
> - `Accel's contribution` 


- **线下主题演讲录像发布**：我们的 **IRL（线下）主题演讲**录像终于发布了。你可以在[这里](https://youtu.be/FH5wiwOyPX4?si=d0acWTgk5h64-uK0)观看。
   - 演讲阵容包括 **Tri Dao**、**Supriya Rao**、**Andrej Karpathy**、**Lily Liu**、**Tim Dettmers** 和 **Wen-mei Hwu** 带来的**精彩演讲**。
- **鸣谢 Accel 提供的高质量录制**：非常感谢 **Accel** 为主题演讲提供的精美录制。他们的努力使得有效地展示这些重要的讨论成为可能。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

as_ai: 酷：
https://openai.com/index/introducing-the-realtime-api/
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1290805417756332085)** (2 messages): 

> - `Breaking into Machine Learning`
> - `Tensor Manipulation in Triton` 


- **对机器学习机会的渴望**：一位用户表达了学习并贡献于机器学习领域的愿望，并提到自己正从软件工程转型且之前有 ML 经验。
   - 他们强调了在没有一份需要解决实际 ML 问题的职业的情况下，在该领域进行自学的挑战。
- **高效操作 Tensor 形状**：另一位用户询问如何操作形状为 `[BLOCK_SIZE_INP * triton.next_power_of_2(n_inp_bits), 256, BLOCK_SIZE_OUT]` 的 Tensor `X`，以便在不进行过多内存操作的情况下移除第二维的元素。
   - 他们正在寻求一种类似于 `X[:,:BLOCK_HIDDEN_SIZE]` 的方法，同时保持数据效率并避免从内存中重复加载/卸载。


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/)** (1 messages): 

deon1217: 第 5 版会在年底前发布吗？
  

---


### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1290797861449240638)** (6 messages): 

> - `Project Presentations Upload`
> - `Future Events Locations` 


- **项目演示将不会上传**：成员们确认**项目演示（project presentations）**将不会上传，因为录制团队没有留守那么长时间。
   - *Mr. Osophy* 用悲伤的表情符号表达了失望，进一步证实了这个令人遗憾的消息。
- **关于未来活动的讨论**：有人提出了未来是否还会举办另一场**活动**，以及是否会在**旧金山**以外的地点举办。
   - Mark 对下一次活动的形式表示不确定，在思考规模会**更大还是更小**，并表示需要更多的考量。


  

---

### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1291010966582198416)** (11 messages🔥): 

> - `TorchAO vs pytorch/torch/ao`
> - `Sensitivity scan and pruning`（敏感度扫描与剪枝）
> - `Prototyping features in TorchAO`（TorchAO 中的原型特性）
> - `Benchmarking and warmup in training`（训练中的基准测试与预热）


- **澄清 TorchAO vs pytorch/torch/ao**：关于使用 **TorchAO**（一个 GPU 量化/稀疏化库）与旧版 **pytorch/torch/ao**（侧重于 CPU 且支持卷积）之间存在混淆。
   - 一名成员强调需要区分这两者，以避免新用户流向过时的库。
- **敏感度扫描和剪枝的问题**：一名新成员在使用剪枝技术进行逐层 *sensitivity scan* 时遇到了异常，这可能是由于过时的参数化（parameterizations）导致的。
   - 在意识到使用了旧版本后，他们被建议查看示例流程并更新到该库的较新版本。
- **理解 TorchAO 中的 Prototype 特性**：讨论了关于被引导至 **TorchAO** 中与 **prototype pruner** 相关的 GitHub 页面，并对其稳定性和用法提出了疑问。
   - 成员们确认，虽然可以使用这些 prototype 特性，但它们缺乏与稳定特性相同的 **backward compatibility**（向后兼容性）保证。
- **基准测试设置中没有 Warmup**：一名成员询问为什么 **TorchAO** 的 `benchmark_aq.py` 脚本中没有实现 warmup 阶段。
   - 这突显了在探索训练效率时，基准测试程序中一个潜在的改进领域。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/ao/tree/main/torchao/sparsity/prototype/pruner">ao/torchao/sparsity/prototype/pruner at main · pytorch/ao</a>: PyTorch native quantization and sparsity for training and inference - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/pull/148">port over torch.ao.pruning to protype folder by jcaip · Pull Request #148 · pytorch/ao</a>: 此 PR 将 torch.ao.pruning 流程移植到 torchao。根据我们上次与 @mklasby 的 OSS 同步会议，我们希望对 BaseSparsifier 进行一些更改，但不希望限制...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[sequence-parallel](https://discord.com/channels/1189498204333543425/1208496482005549086/1291033500623044651)** (3 messages): 

> - `Long Context Methods`（长上下文方法）
> - `Survey Papers on Context`（上下文综述论文）
> - `Author Engagement`（作者互动）


- **作者开放提问**：论文作者之一 @bekar2617 邀请参与者就该论文提出任何问题。
   - *欢迎随时交流！*
- **询问长上下文综述**：成员 @glaxus_ 赞扬了该论文，并询问是否有涵盖 **long context methods** 的综述，并指出这类方法数量惊人。
   - 他们表示，要追踪该领域所有不同的方法非常困难。
- **不了解综述论文**：@bekar2617 幽默地回应说，他们并不知道关于长上下文方法有任何 **好的综述论文**。
   - *哈哈，确实如此*，这表明大家对现有大量资料感到同样的困扰。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1290751829503901837)** (9 messages🔥): 

> - `Geopolitical Stability`（地缘政治稳定性）
> - `Political Discussions in Server`（服务器内的政治讨论）
> - `User Reactions to Political Stress`（用户对政治压力的反应）
> - `Community Guidelines on Topics`（关于话题的社区准则）


- **社区对地缘政治稳定性的担忧**：*apaz* 对当前的地缘政治气候表示不确定，认为它比通常感知的更加脆弱。
   - *kashimoo* 对此表示共鸣，分享了因家人在受影响地区而产生的个人担忧，称局势令人神经紧张。
- **关于服务器内政治讨论的辩论**：*mr.osophy* 指出了可能存在的禁止政治讨论的准则，并询问是否应允许尊重性的对话。
   - *apaz* 和其他人一致认为，禁止讨论政治是合理的，暗示这些规则可能是为了社区的舒适度。
- **对政治压力的复杂反应**：*marksaroufim* 分享了对地缘政治事件感到悲伤的悲观观点，幽默地建议专注于编码可能会更好。
   - 他表达了一种在应对政治紧张局势影响时渴望和平的同伴情谊。
- **冒犯性言论引发反弹**：*mr.osophy* 关注到一条涉及 Elon Musk 和海地人的争议性言论，强调了其不当性。
   - 他强调此类言论具有冒犯性，并推动了来自移民社区的理解叙事。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

saurabh_works: 印度有小组吗？ 🇮🇳
  

---

### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1290790733061685318)** (6 messages): 

> - `Triton Kernel Explanation`
> - `Add Vector Function`
> - `Row Major Format in Tensors` 


- **理解向量加法函数 (Add Vector Function)**：一位用户分享了一个错误的 Triton 向量加法 Kernel 实现，这引发了关于正确方法的讨论。
   - 讨论强调了在 Tensor 操作中保持清晰度的必要性，以及正确索引 Tensor 的重要性。
- **行优先格式 (Row Major Format) 澄清**：成员们解释说，所有 Tensor 都以行优先格式存储为一维数组，这意味着列中每个元素之间的间隔为 N1 的大小。
   - 这一细节帮助用户理解了为什么必须乘以 N1 才能正确计算存储位置。
- **有效的教育资源**：一位用户评论了一张展示 Tensor 在内存中如何索引的图表非常有用，并赞赏其清晰度。
   - 这表明在技术解释中使用视觉辅助工具具有积极作用，强调了它们在理解复杂概念方面的价值。


  

---


### **GPU MODE ▷ #[hqq-mobius](https://discord.com/channels/1189498204333543425/1225499037516693574/1290870043273334976)** (8 messages🔥): 

> - `AWQ and HQQ comparison`
> - `Quantization methods`
> - `Perplexity benchmarks`
> - `MMLU and GSM8K tests`
> - `lm eval implementations` 


- **AWQ 结合 HQQ 产生了令人惊讶的结果**：用户观察到，将 **AWQ** 与 **HQQ** 结合使用似乎比单独使用任何一种方法的效果都要好，这引发了关于该方法有效性的疑问。
   - 一位用户提供了一个 [GitHub 示例](https://github.com/vayuda/ao/blob/awq/torchao/prototype/awq/example.py) 来展示相关实现。
- **使用 uint4 进行校准的效果出奇地好于 HQQ**：在校准过程中，一位用户注意到使用 **uint4** 量化误差而非 **HQQ**，产生了一些有趣的性能结果。
   - 他们强调了 AWQ 的缩放优化 (scaling optimization) 与 HQQ 的零点 (zero-point) 关注点之间的区别。
- **Perplexity 对比引起关注**：尽管预期性能会更好，但 **HQQ** 的 Perplexity 分数却高于 **int4**，这引发了担忧。
   - 用户对基准测试的可靠性提出了质疑，建议放弃 Perplexity，转而采用其他指标。
- **寻求更稳健的基准测试**：有请求建议在 Instruct 模型上运行 **MMLU** 和 **GSM8K** 测试，以获得更稳健的基准测试结果。
   - 注意到目前使用 **llm-awq** 论文实现的 Perplexity 计算方法与常用的 lm eval 方法有所不同。
- **分享 lm eval 脚本**：一位用户分享了他们结合 **HF models** 运行 **lm eval** 的脚本，特别是针对 Instruct 模型。
   - 这一贡献旨在简化社区内的基准测试流程。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/vayuda/ao/blob/awq/torchao/prototype/awq/example.py">ao/torchao/prototype/awq/example.py at awq · vayuda/ao</a>: 用于量化和稀疏化的原生 PyTorch 库 - vayuda/ao</li><li><a href="https://github.com/mobiusml/hqq/blob/master/examples/llama2_benchmark/eval_model.py">hqq/examples/llama2_benchmark/eval_model.py at master · mobiusml/hqq</a>: Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1290750251967316039)** (38 messages🔥): 

> - `Pipeline Parallelism`
> - `Activation Checkpointing`
> - `Zero-3 Implementation`
> - `Chunked Softmax`
> - `Sequence Parallelism` 


- **Pipeline Parallelism 引发担忧**：成员们对 **Pipeline Parallelism** 表示担忧，认为它使训练的高效调度变得复杂。
   - 有人指出，虽然简单的实现很直接，但避免性能瓶颈会变得非常复杂，导致了诸如“没有代码库能在 **Pipeline Parallelism** 中幸存”之类的怀疑。
- **Activation Checkpointing 的进展**：Llama3 分支的 **Activation Checkpointing** 取得了进展，旨在通过在训练期间选择性地存储残差来节省内存。
   - 尽管实现了一个减少内存负载的版本，但仍担心这不足以支持 **128K context length** 下的 **405B BF16** 训练。
- **Zero-3 和 Chunked Softmax 的讨论**：讨论了实现 **Zero-3** 以及 **Chunked Softmax**，强调了对高效内存管理技术的需求。
   - 一种建议的方法是使用 **Chunked Softmax** 来提高内存效率，这在处理大词汇量和长上下文长度的挑战时尤为相关。
- **思考将 Sequence Parallelism 作为解决方案**：一位成员提议将 **Sequence Parallelism** 作为更有效地管理 Attention 层、最小化大规模并行处理开销的潜在解决方案。
   - 通过在 GPU 之间划分任务，特别是对于大型模型，它可能提供一条更简单的路径，而无需对现有系统进行大量修改。
- **卸载残差的复杂性**：讨论了将残差卸载到 CPU 内存的可行性，但强调了管理大型模型所需的海量数据的挑战。
   - 管理内存的策略包括选择性地重新计算激活值（Selective Recomputation of Activations），并利用现有方法结合前向传递效率，尽管复杂性仍然是一个令人担忧的问题。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/actions/runs/11131983628/job/30934795539">add llama 3 support to llm.c · karpathy/llm.c@d808d78</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/karpathy/llm.c/pull/773">Activation Checkpointing for Llama3 branch by ademeure · Pull Request #773 · karpathy/llm.c</a>: This keeps residual3 for all layers, and then up to N layers for everything else, with relatively little complexity... This means if you set &amp;quot;-ac 16&amp;quot;, it will only recompute 50% of a...</li><li><a href="https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/#interleaved_schedule">Scaling Language Model Training to a Trillion Parameters Using Megatron | NVIDIA Technical Blog</a>: Natural Language Processing (NLP) has seen rapid progress in recent years as computation at scale has become more available and datasets have become larger. At the same time, recent work has shown&#82...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1291128573603876894)** (1 messages): 

> - `Advancing AI event`
> - `ROCM developers` 


- **旧金山举办 Advancing AI 活动**：**Advancing AI 活动**定于 **10/10** 在旧金山 **Moscone** 举行，重点关注即将推出的硬件和软件进展。
   - *有兴趣的参与者请私信获取注册详情，并与 ROCM 开发者交流。*
- **与 ROCM 开发者交流的机会**：ROCM 社区邀请有兴趣的参与者参加活动，讨论 AI 领域的各种话题。
   - *这是一个与 ROCM 开发者直接建立联系、交流并了解他们最新项目的机会。*


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1290959579601174560)** (2 messages): 

> - `Kernel Functional Reminder`
> - `Contribution Guide Updates` 


- **在 Kernel 中添加 functional 的提醒**：一位成员提到他们在插入新 **Kernel** 时忘记添加 **functional**，并强调这是一个容易被忽视的细节。
   - 他们建议在 **contribution guide** 中加入提醒，以避免未来出现类似的疏忽。
- **正在处理提醒事项**：另一位成员提到，目前有人正在处理关于添加 **functional** 的问题。
   - 这表明社区正在共同努力改进贡献流程。


  

---

### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1290917501122646098)** (1 条消息): 

> - `前缀和 (Prefix sum) 谜题`
> - `调试 notebook 崩溃` 


- **前缀和谜题需要解决**：一名成员询问是否有人解决了 **前缀和谜题 (prefix sum puzzle)**，因为他们遇到了困难。
   - 他们提到其 **notebook 崩溃**，并对如何调试该问题表示困惑，称不确定自己错在哪里。
- **请求调试帮助**：一位用户表达了在调试与前缀和谜题相关的设置时面临的挑战。
   - 他们正在寻求关于解决尝试运行 notebook 时发生崩溃的方法的见解或建议。


  

---


### **GPU MODE ▷ #[diffusion](https://discord.com/channels/1189498204333543425/1288899271193526342/1291039199373557812)** (2 条消息): 

> - `FLUX 推理模型`
> - `自定义 Kernel 内存优化` 


- **FLUX 推理库见解**：[FLUX 仓库](https://github.com/black-forest-labs/flux/blob/87f6fff727a377ea1c378af692afb41ae84cbe04/src/flux/sampling.py#L32) 提供了 FLUX.1 的官方推理模型，为 black-forest-labs 负责的持续开发做出贡献。
   - 该仓库封装了核心功能，包括对性能提升至关重要的采样函数。
- **优化图像处理中的内存使用**：一名成员建议，与其发送完整的图像数据（例如 1024 x 1024 图像的 **3145728 B**），不如只发送尺寸（减少到 **6 B**）会更有效率。
   - 在自定义 Kernel 中计算 `img_ids` 将利用共享内存 (shared memory) 来提高效率，并断言处理整数也可以进行全面优化。



**提到的链接**：<a href="https://github.com/black-forest-labs/flux/blob/87f6fff727a377ea1c378af692afb41ae84cbe04/src/flux/sampling.py#L32),">flux/src/flux/sampling.py at 87f6fff727a377ea1c378af692afb41ae84cbe04 · black-forest-labs/flux</a>：FLUX.1 模型的官方推理库。通过在 GitHub 上创建账户为 black-forest-labs/flux 的开发做出贡献。

  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1290763471712944198)** (72 条消息🔥🔥): 

> - `贝叶斯 vs 频率派模型`
> - `纽约时报 (NYT) 诉讼的影响`
> - `爬虫抓取的合法性`
> - `专家证人动态`
> - `OpenAI 和解` 


- **在神经网络架构中探索贝叶斯模型**：一名成员指出，神经网络架构主要使用 **频率派统计学 (frequentist statistics)**，这使得在可训练模型中实现贝叶斯网络和 beta 分布具有挑战性。
   - 他们提出了直观的替代方案，包括将概率折叠进模型权重中，而不保留贝叶斯框架的复杂性。
- **NYT 诉讼测试 AI 版权的界限**：随后讨论了 OpenAI 向 **NYT** 支付赔偿以避免更广泛责任的可行性，突显了围绕 LLM 侵权指控的复杂性。
   - 有观点认为，补偿 NYT 并不意味着整个 AI 模型都侵犯了版权，并指出了盈利实体与独立创作者之间行业动机的差异。
- **关于爬虫抓取合法性与伦理的辩论**：成员们对从互联网抓取材料的伦理后果表示担忧，特别是这与创意专业人士的关系以及潜在的诉讼。
   - 产生了一些疑问：关于抓取的任何裁决是否会界定 **OSS 开发者** 的不同责任，从而可能对研究背景产生重大影响。
- **关于专家证人角色与挑战的见解**：一位参与者幽默地推测了数学家在针对数据压缩和 LLM 训练的版权案件中担任专家证人所面临的挑战。
   - 他们指出，审判结果可能取决于复杂的争议点：即是否能证明 LLM 在权重中存储了训练数据，或者是否展示了对该数据的转换。
- **对 OpenAI 审判结果的推测**：社区讨论了 OpenAI 达成诉讼和解的可能性，提出只有极少数原告会因为其作品被突出显示的特定案例而索赔。
   - 最终，大多数人同意很难预测广泛的裁决，这可能会导致与老牌出版物达成创新的许可安排。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1290872140177866796)** (13 条消息🔥): 

> - `Sequential Prediction of Output Representations`（输出表示的序列预测）
> - `Liquid Neural Networks Application`（Liquid Neural Networks 应用）
> - `Self-Supervised Learning on Arbitrary Embeddings`（针对任意 Embedding 的 Self-Supervised Learning）
> - `Transfer Learning Techniques in NLP`（NLP 中的 Transfer Learning 技术）


- **探索连续函数中的序列预测**：成员们讨论了将预测连续函数的任务视为一个 Autoregressive 过程，例如预测 **F(x) = 0.135** 中的数字。
   - 有建议提出探索 T5 处理连续预测任务的方法，强调其在处理各种模型时的通用性。
- **Liquid Neural Networks 看起来很有前景**：讨论指向使用 Liquid Neural Networks 来拟合连续函数，强调其较低的开发复杂度。
   - 一位成员评论说，假设开发者对模型有足够的了解，Pipeline 可以做得更加 End-to-End。
- **适用于任何模型的 Self-Supervised Learning**：一位成员介绍了在源自任何模型和数据的任意 Embedding 上进行 Self-Supervised Learning (SSL) 的概念。
   - 他们进一步阐述了将 SSL 扩展到适用于任何模型权重的方法，通过收集来自各种模型的线性层来构建数据集。
- **使用 T5 的 NLP Transfer Learning 技术**：强调了 T5 的能力，特别是它在 NLP 任务的 Transfer Learning 中的有效性。
   - 一位成员幽默地提到：*“该死，T5 考虑到了所有事情，”* 反映了对其全面的 Text-to-Text 框架的认可。
- **检查深度学习 Optimizer**：分享了一个讨论深度学习 Optimizer 新设计空间的链接，在没有凸性假设的情况下对 Adam 和 Shampoo 等方法进行了批评。
   - 该提案包括根据张量在神经架构中的角色为其分配不同的算子范数（Operator Norms），从而可能增强训练稳定性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/1910.10683">Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer</a>：Transfer Learning 是指模型首先在数据丰富的任务上进行预训练，然后在下游任务上进行微调，它已成为 NLP 领域的一种强大技术。其效果...</li><li><a href="https://arxiv.org/abs/2409.20325">Old Optimizer, New Norm: An Anthology</a>：深度学习 Optimizer 的动力通常源于凸理论和近似二阶理论的结合。我们选择了三种此类方法——Adam、Shampoo 和 Prodigy，并论证了每种方法都可以...
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1290754513182916809)** (47 messages🔥): 

> - `OpenAI Subscription Tiers`
> - `Voice Model Preferences`
> - `Liquid AI Architecture`
> - `Playground Access Issues`
> - `API Access Updates` 


- **用户渴望更高层级的订阅以获取更新**：成员们讨论了推出更高价格的 OpenAI 订阅可能带来的好处，例如提供更及时的功能和服务。
   - 一位用户表达了在使用各种 AI 平台时，对当前订阅层级所施加限制的沮丧。
- **对新 Cove 语音模型的困扰**：多位用户对新的 Cove 语音模型表示不满，称其缺乏经典版本的冷静特质，且过于活跃。
   - 一位用户强调了社区对新语音的反对共识，并恳求恢复经典语音。
- **Liquid AI 先进架构的性能**：讨论涉及了来自 Liquid AI 的一种新架构，据称其性能优于传统的 LLM，目前已开放测试。
   - 成员们注意到了它的推理效率，并推测其架构与典型的 Transformer 模型有所不同。
- **访问 Playground 的问题**：一位用户提出了登录 Playground 的疑虑，其他用户也遇到了类似问题，或建议使用无痕模式作为解决方法。
   - 交流信息显示，访问权限可能因用户所在地而异，特别是在瑞士等特定地区。
- **API 访问与使用层级**：一位用户询问 API 访问是否仅限于特定的使用层级，回复澄清说以前是基于层级的，但目前情况已发生变化。
   - 这表明 OpenAI 的服务产品和可访问性正在进行持续调整。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://playground.livekit.io/">Realtime Playground</a>: 针对 OpenAI 新的 Realtime API 的语音对语音 Playground。基于 LiveKit Agents 构建。</li><li><a href="https://community.openai.com/t/about-coves-voice-changing-with-the-new-conversation-system/957778/6">关于 Cove 语音随新对话系统改变的讨论</a>：我也有同感。不敢相信我竟然在怀念一个 AI 助手的声音。哈哈</li><li><a href="https://www.cartesia.ai/sonic">Cartesia</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1290770050281902081)** (18 messages🔥): 

> - `Disappearing Responses Issue`
> - `Creating Custom GPT with Unique Features`
> - `o1-preview Model Access Query`
> - `Using OAuth for Google Drive Connections`
> - `GPT Policy Violation Appeal Process` 


- **报告响应消失问题**：用户报告了 macOS 桌面应用中响应消失的问题，归因于最近的一次更新，该更新可能更改了通知选项。
   - 一位用户表达了沮丧，强调这在过去 20 分钟内影响了他们的体验。
- **发布 'Simple Story' 自定义 GPT**：创建了一个名为 'Simple Story' 的新自定义 GPT，旨在将简单的句子转化为连贯的故事，保持适当的间距并有效地引入角色。
   - 创建者表示，这个 GPT 解决了 ChatGPT 中发现的缺点，灵感来自对作家 David Perell 的采访。
- **关于模型访问的澄清：o1-preview**：关于 OpenAI 平台上模型访问的查询，特别是区分 'o1-preview' 及其最近的快照 'o1-preview-2024-09-12'。
   - 一位成员回复称，这两个端点目前都指向同一个模型快照，并对在 ChatGPT 中使用时是否存在差异提出了疑问。
- **探索使用 OAuth 为特定用户保存至 Google Drive**：讨论围绕创建一个 GPT 的可能性，该 GPT 允许其他用户使用 OAuth 登录并将对话保存到他们自己的 Google Drive 中。
   - 一位成员寻求关于自定义 GPT 是否可行实现此功能的澄清，并认识到实施此类功能涉及的技术细节。
- **GPT 政策违规的申诉流程**：一位用户分享了关于收到邮件称其 GPT 'Video Summarizer' 因违反政策而被移除的经历，以及他们为解决此事正在进行的申诉。
   - 他们对一周后仍未收到客户支持的回应表示沮丧，强调了从商店中失去 GPT 带来的情感影响。

  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1291057356041359503)** (3 messages): 

> - `LLMs and chain-of-thought`
> - `Midjourney seed number retrieval` 


- **LLMs: Chain-of-Thought 无法阻止幻觉**：据指出，虽然 **chain-of-thought** 提高了 LLMs 的 **accuracy**，但它并不能减少幻觉，正如在回答火属性狐狸宝可梦时，将 Vulpix 错误地回答为 'Ninetales'。
   - 通过验证尾巴数量突显了幻觉与智能之间的区别，结果显示 Vulpix 有 **6** 条尾巴，而 Ninetales 有 **9** 条。
- **从 Midjourney 图像中获取 Seed Number**：一位用户询问如何获取在网页版 Midjourney 中生成的图像的 **seed number**。
   - 一名成员回复并引导他们在另一个频道（特别是 <#998381918976479273>）中查找信息。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1291057356041359503)** (3 messages): 

> - `LLMs Hallucination Issues`
> - `Midjourney Seed Number Retrieval` 


- **LLMs 无法通过 chain-of-thought 防止幻觉**：用户注意到，使用 **chain-of-thought** 技术并不能防止 LLMs 产生幻觉，尽管它可能会提高准确率，正如在宝可梦问题中的错误回答所示。
   - 尽管引入了思考过程，像 **gpt-3.5-turbo** 这样的模型仍然会导致不准确，揭示了潜在的幻觉问题。
- **查找 Midjourney 的 seed number**：一位用户询问如何从 **Midjourney** 生成的图片中检索 **seed number**。
   - 另一名成员建议访问特定频道寻求帮助，表明社区对该查询的支持。


  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1290750221927583785)** (53 messages🔥): 

> - `OpenAI's new funding round`
> - `OpenAI's Advanced Voice`
> - `Multi-GPU training techniques`
> - `Releases of multimodal language models`
> - `Azure AI's HD neural TTS` 


- **OpenAI 获得 67 亿美元新融资**：OpenAI 正式宣布完成一轮 **67 亿美元** 的融资，估值为 **1570 亿美元**，并与包括美国盟友政府在内的关键伙伴合作，以进一步释放 AI 技术的潜力。
   - 这引发了关于哪些盟友参与其中的疑问，讨论推测包括北约国家，以及对 AI 国际合作的影响。
- **Advanced Voice 向所有用户推出**：OpenAI 已开始向全球所有 ChatGPT Enterprise 和 Edu 用户推出 **Advanced Voice** 功能，免费用户也可以进行试用。
   - 这一增强反映了 OpenAI 致力于改进交互模态的承诺，尽管一些用户对语音应用的实际性能表示怀疑。
- **Multi-GPU 训练见解**：分享了关于 Multi-GPU 训练的深刻分析，强调了并行化网络训练、高效的状态通信以及快速故障恢复技术的重要性。
   - **checkpointing** 和 **optimizer state** 通信等技术被强调为在 **10k** GPU 使用规模下保持性能的关键。
- **发布新的多模态模型**：最近的公告重点介绍了 **MM1.5**，这是 Apple 推出的一个新的多模态语言模型家族，旨在增强 OCR 和多图推理能力。
   - 这些模型提供 dense 和 MoE 变体，包括专门为视频和移动 UI 理解设计的版本。
- **Azure AI 的 HD Neural TTS 更新**：Microsoft 在 Azure AI 上推出了其 neural TTS 服务的 HD 版本，通过情感上下文检测增强了生成语音的表现力和参与度。
   - 凭借 **auto-regressive transformer** 模型等特性，用户可以期待语音真实感和质量的提升，使 Azure 的服务在 AI 驱动的语音应用中具有竞争力。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://x.com/keithwhor/status/1841186962230952372">来自 keith (@keithwhor) 的推文</a>: @emileberhard @OpenAI @pbbakkum @landakram @DustMason 本周内陆续推出！</li><li><a href="https://x.com/swyx/status/1841165359162015789">来自 swyx @ DevDay! (@swyx) 的推文</a>: 这是我为关注者准备的 @OpenAIDevs 开发者大会推文串。其他人会提供视频之类的，所以我只分享全天的个人笔记和灵感瞬间。第一个观察：@sama 缺席了 GP...</li><li><a href="https://x.com/dsa/status/1841293790503747646?s=46">来自 dsa (@dsa) 的推文</a>: OpenAI 今天发布了 Realtime API。在向所有开发者开放权限期间，可以使用我们的密钥尝试该 API：https://playground.livekit.io/</li><li><a href="https://news.ycombinator.com/item?id=41714877">看起来即使是非实时 API，他们的输出音频收费也达到了 $200/M... | Hacker News</a>: 未找到描述</li><li><a href="https://x.com/natolambert/status/1841121479976763889?s=46">来自 Nathan Lambert (@natolambert) 的推文</a>: 过去两周出现了很多大型多模态语言模型，但关于它们的 RLHF 到底是什么样的，目前几乎还是空白。在 LLaVA 之后，谁在做这方面的工作？</li><li><a href="https://x.com/bleedingpurple4/status/1841221400474108062?s=46">来自 Bleeding Purple Guy (@BleedingPurple4) 的推文</a>: @WilliamShatner 你还好吗，船长？</li><li><a href="https://x.com/bindureddy/status/1841204392235851974?s=46">来自 Bindu Reddy (@bindureddy) 的推文</a>: 呼叫中心的人类松了一口气……他们刚刚意识到自己比 OpenAI 的语音模式更便宜，后者的价格高达每小时 $18 🤯🤯</li><li><a href="https://x.com/jeffclune/status/1841167663252615634">来自 Jeff Clune (@jeffclune) 的推文</a>: 我的答案：Alec Radford。下面有很多不错的建议，但在我看来，@AlecRad 显然是影响力最大但认可度最低的人。他一直是这么多惊人成就的推动者……</li><li><a href="https://x.com/soumithchintala/status/1841498799652708712">来自 Soumith Chintala (@soumithchintala) 的推文</a>: 分为三个部分。1. 将尽可能大的网络和尽可能大的 batch-size 适配到 10k/100k/1m 个 H100 上——进行并行化并使用节省内存的技巧。2. 通信状态……</li><li><a href="https://x.com/ericabrescia/status/1841510129868410896?s=46">来自 Erica Brescia (@ericabrescia) 的推文</a>: 很高兴能与 @jasoncwarner、@eisokant 以及他们在 @poolsideai 建立的优秀团队一起，开启这段通过代码实现 AGI 的旅程。迫不及待想让世界从这个团队的成果中受益……</li><li><a href="https://x.com/iscienceluvr/status/1841061837779189960?s=46">来自 Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr) 的推文</a>: MM1.5：多模态 LLM 微调的方法、分析与见解。摘要：https://arxiv.org/abs/2409.20566。Apple 推出 MM1.5，这是一个全新的 MLLM 系列，经过精心设计以增强一系列核心能力……</li><li><a href="https://tenor.com/view/what-am-i-looking-at-landon-bloom-inventing-anna-what-is-this-whats-this-thing-gif-25142098">我在看什么 Landon Bloom GIF - 我在看什么 Landon Bloom 《创造安娜》 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/jaminball/status/1841213689741132125">来自 Jamin Ball (@jaminball) 的推文</a>: OpenAI 实时音频的费用约为每小时 $10-$15（按 80/20 混合比例计算，每 1M tokens 为 $120）。这取决于终端用户说话与模型“说话”的比例，但大致方向应该是……</li><li><a href="https://x.com/OpenAI/status/1841179938642411582">来自 OpenAI (@OpenAI) 的推文</a>: 从本周开始，高级语音模式（Advanced Voice）将向全球所有 ChatGPT Enterprise、Edu 和 Team 用户推出。免费用户也将获得高级语音模式的试用机会。欧盟的 Plus 和免费用户……我们将……</li><li><a href="https://x.com/gabestengel/status/1841089276198477859?s=46">来自 Gabriel Stengel (@GabeStengel) 的推文</a>: 今天，我们很高兴地宣布由 @khoslaventures 的 @rabois 领投的 1850 万美元 A 轮融资，参与者还包括 @jaltma、@ericschmidt、@mantisVC 等。https://rogo.ai/blog/rogo-series-a-with-khosla-v...</li><li><a href="https://x.com/picocreator/status/1841292591490634051">来自 PicoCreator - 🌉 中的 AI 模型构建者 (@picocreator) 的推文</a>: OpenAI 开发者大会 x NotebookLM ❤️ 这段关于 OpenAI 开发者大会的完整对话总结完全由 AI 生成，没有任何人工监督或干预 🤖</li><li><a href="https://github.com/yakazimir/esslli_2024_llm_programming?tab=readme-ov-file">GitHub - yakazimir/esslli_2024_llm_programming: ESSLLI 2024 语言模型编程课程资源</a>: ESSLLI 2024 语言模型编程课程资源 - yakazimir/esslli_2024_llm_programming</li><li><a href="https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/new-hd-voices-preview-in-azure-ai-speech-contextual-and/ba-p/4258325">Azure AI Speech 中全新的高清语音（HD voices）预览</a>

: 上下文相关且逼真的输出演进</a>：我们致力于改进 Azure AI Speech 语音的决心坚定不移，始终努力使其更具表现力和吸引力。今天，我们...</li><li><a href="https://github.com/Azure-Samples/Cognitive-Speech-TTS/blob/master/doctopodcast/doctopodcast.py">Cognitive-Speech-TTS/doctopodcast/doctopodcast.py at master · Azure-Samples/Cognitive-Speech-TTS</a>：多种语言的 Microsoft Text-to-Speech API 示例代码，属于 Cognitive Services 的一部分。- Azure-Samples/Cognitive-Speech-TTS
</li>
</ul>

</div>
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1290755824641118259)** (49 条消息🔥): 

> - `ComfyUI 安装问题`
> - `Flux 模型利用`
> - `Automatic1111 故障排除`
> - `基于 Debian 的操作系统偏好`
> - `Python 版本兼容性` 


- **ComfyUI 安装挑战**：一位用户在 Google Colab 上安装 **ComfyUI** 时遇到困难，指出 comfyui manager 的安装过程存在问题。
   - 讨论中提到了对特定模型路径的需求以及与 **Automatic1111** 的兼容性挑战。
- **Flux 模型展示了令人印象深刻的特性**：用户强调了 **Flux 模型** 的有效性，特别是在生成一致的角色图像以及修复手部和脚部细节方面。
   - 一位成员分享了一个 [Flux lora 的链接](https://civitai.com/models/684810/flux1-dev-cctv-mania)，尽管其主要用途并非为此，但出人意料地提升了图像质量。
- **Automatic1111 安装故障排除**：一位成员报告在安装最新版本的 Python 时遇到了 **Automatic1111** 的安装问题，引发了关于兼容性的疑问。
   - 建议指向使用 **virtual environments** 或 **Docker** 等容器来管理不同的 Python 版本。
- **基于 Debian 的操作系统讨论**：关于使用 **Debian-based** 操作系统展开了对话，用户注意到了 **Pop** 和 **Mint** 等流行发行版的各自特点。
   - 一位用户幽默地表达了他们打算因其独特特质而再次尝试 **Pop** 的意图。
- **Python 版本兼容性担忧**：成员们讨论了使用 **最新 Python 版本** 的影响，建议旧版本可能为某些脚本提供更好的兼容性。
   - 一位用户考虑调整他们的环境以分别运行脚本，从而解决稳定性问题。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/ostris/OpenFLUX.1">ostris/OpenFLUX.1 · Hugging Face</a>：未找到描述</li><li><a href="https://civitai.com/models/630820?modelVersionId=705611">Flux Fusion DS (平滑合并) [4+ 步] [AIO &amp; UNET] [所有 GGUF •  NF4 • FP8/FP16] - v0-fp8-e4m3fn (AIO) | Flux Checkpoint | Civitai</a>：适用于低 VRAM 的 GGUF 和 BNB 量化。需要额外设置 ↓↓↓ AIO（全合一）版本包含 UNET + VAE + CLIP L + T5XXL (fp8)。仅 UNET 版本包含...</li><li><a href="https://civitai.com/models/813172/marvelmixx">Marvelmixx - v1.0 | Flux LoRA | Civitai</a>：此 LoRA 融合了著名漫画艺术家 Jim Lee、Joe Madureira 和 Mike Deodato 的独特风格。它捕捉了动态的...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1290762618205638758)** (38 条消息🔥): 

> - `Rate Limit Increases` (速率限制提升)
> - `Llama 3.2 Release` (Llama 3.2 发布)
> - `LiquidAI Performance` (LiquidAI 性能)
> - `Chat Download Feature` (对话下载功能)
> - `Text-to-Speech Utility` (文本转语音实用性)


- **关于提升 Rate Limit 的讨论**：一名成员询问是否可以申请增加 API 的 **rate limit**，希望能突破 **20** 的限制。
   - 另一名成员确认了这一请求，强调了对更高阈值的普遍需求。
- **对 Llama 3.2 发布的期待**：用户多次表达了对 **Llama 3.2** 发布的兴趣，渴望其早日到来。
   - 一名成员幽默地提到 *laughs in mr arvind*，暗指发布时间表存在不确定性。
- **LiquidAI 因速度受到赞赏**：**LiquidAI** 因其速度获得好评，一位用户惊叹它比现有模型 *快得惊人*。
   - 然而，也有人指出虽然它运行速度很快，但也被认为 *准确度欠佳*。
- **对话下载功能**：有人询问是否可以将整个对话环节下载为 **PDF**，另一位用户确认已经实现了这一功能。
   - 讨论强调了该功能的潜在需求，特别是在保存完整对话内容时。
- **文本转语音功能反馈**：讨论了 **text-to-speech (TTS)** 功能的实用性，一位用户提到他们在工作中经常使用它来处理长回复。
   - 尽管存在一些 **pronunciation issues** (发音问题)，但他们认为这是一个有价值的工具，表明该功能仍有改进空间。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1290875796222644365)** (7 条消息): 

> - `Perplexity AI and Philosophy` (Perplexity AI 与哲学)
> - `LiquidAI GPT Rival Launch` (LiquidAI 发布 GPT 竞争对手)
> - `Stability AI vs ClipDrop` (Stability AI 对比 ClipDrop)
> - `FLUX Model Efficiency` (FLUX 模型效率)
> - `Current AI Model Landscape` (当前 AI 模型格局)


- **Perplexity AI 辅助哲学研究**：一位用户对 **Perplexity AI** 辅助哲学研究的能力表示赞赏，并分享了一个关于修昔底德 (Thucydides) 的链接。
   - 这展示了该平台在快速获取哲学见解方面的实用性。
- **LiquidAI 发布 GPT 竞争对手**：**LiquidAI** 推出了其新的 GPT 竞争对手，突显了 AI 格局的变化，同时还更新了 **Telegram** 调整其政策的相关动态。
   - 这一进展通过一段 [YouTube 视频](https://www.youtube.com/embed/ldqAVGPcrM8) 分享，表明 AI 工具取得了显著进步。
- **Stability AI 与 ClipDrop 的对比**：一位用户查看了 **Stability AI** 和 **ClipDrop** 之间的对比，以更好地了解两者的异同。
   - 分享的 [链接](https://www.perplexity.ai/page/stability-ai-vs-clipdrop-which-Y9OtHPLIRJSMEoCQxybzHQ) 提供了关于它们能力的进一步见解。
- **FLUX 模型概览**：一位用户发现 **FLUX** 不区分照片和绘画，并强调了其处理速度。
   - 正如最近的一项搜索所指出的，该模型的能力可能会推动图像识别领域的发展。
- **当前 AI 模型见解**：用户对当前 AI 模型的格局进行了咨询，寻求对新兴技术的见解。
   - 分享的链接显示，关于哪些模型处于领先地位的讨论正在持续进行。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1290889213650407506)** (2 条消息): 

> - `API Credit Usage` (API 额度使用)
> - `Account Details Inquiry` (账户详情查询) 


- **用户询问 API 额度变化**：一位用户提出了关于何时能看到 **API credits** 变化的问题，并表示困惑，因为尽管昨天使用了 API，但显示 **$0 usage cost**。
   - 这一询问突显了用户对理解 **API usage** 计费周期的关注。
- **提供账户详情支持**：另一名成员迅速回应了该用户的查询，要求其 **DM their account details** (私信账户详情) 以获取进一步帮助。
   - 这种互动表明社区已准备好支持面临账户问题的用户。


  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1290762529764540446)** (26 条消息🔥): 

> - `Credit card cloud support` (信用卡云端支持)
> - `Event notifications issue` (活动通知问题)
> - `MSFT Copilot Studio inquiry` (MSFT Copilot Studio 咨询) 


- **寻求信用卡云端和 Apple Pay 支持**：一位成员表示需要**对信用卡云端和 Apple Pay 的全面支持**，随后另一位成员建议发送邮件至 [support@cohere.com](mailto:support@cohere.com) 以获取帮助。
   - 他们提出将从那里接手，以确保更顺畅的求助流程。
- **活动通知延迟**：一位成员报告称**活动通知**在实际活动结束后才送达，特别提到了最近一次 **Office Hours** 会议期间的问题。
   - 另一位成员承认这是一个**技术故障 (technical glitch)**，并感谢报告者引起关注。
- **关于 MSFT Copilot Studio 的咨询**：一位成员询问是否有人有使用 **MSFT Copilot Studio** 的经验，并质疑其与替代方案相比的价值。
   - 另一位成员评论说，这个空间不是用来做服务广告的，表明了对推广类查询的敏感性。


  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1290751683470561330)** (13 条消息🔥): 

> - `Azure Model Refresh Issues` (Azure 模型刷新问题)
> - `Cohere Chat App Roadmap` (Cohere Chat App 路线图)
> - `Cohere Webinar Opportunities` (Cohere 网络研讨会机会)
> - `RAG++ Course Resources` (RAG++ 课程资源)
> - `Reasoning Models for AI Agents` (AI Agent 的推理模型) 


- **Azure 模型刷新问题**：一位成员报告了通过 Azure 刷新模型时遇到的障碍，并被建议提交支持工单，同时联系 Azure 团队以寻求进一步协助。
   - 另一位成员索要了问题 ID 以便跟踪，并强调需要就当前问题与 Azure 团队进行沟通。
- **关于 Cohere Chat App 开发的咨询**：一位成员询问 Cohere 是否计划开发 Chat App，特别是针对移动设备的版本。
   - 他们表达了对 Cohere 的热情，并提出愿意主持一场在线研讨会，在他们的 AI 社区内进行推广，彰显了他们对该平台的倡导。
- **寻找 RAG++ 课程资源**：一位成员请求 RAG++ 课程的资源，随后另一位成员提供了与 **Retrieval Augmented Generation (RAG)** 相关的链接和信息。
   - 资源包括在 Cohere 中使用 RAG 的指南以及用于实际实现的 Cookbooks，旨在提升用户构建生成式 AI 应用的体验。
- **关于 AI Agent 推理模型的讨论**：一位成员发起了关于分解生成式 AI 应用复杂性的讨论，质疑是否应将任务卸载给具有高级推理能力的模型。
   - 他们概述了粒度与特定模型耦合风险之间的权衡，并征求社区对该话题的见解。



**提到的链接**：<a href="https://docs.cohere.com/page/cookbooks#rag)">Cookbooks — Cohere</a>：探索一系列 AI 指南，开始使用 Cohere 的生成式平台，这些资源已预先制作并经过最佳实践优化。

  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1290898212496084993)** (6 条消息): 

> - `API Error 403` (API 403 错误)
> - `Model Transfer Issues` (模型迁移问题)
> - `Support Contact` (支持联系方式) 


- **遇到 API 403 错误**：一位用户报告在迁移到另一个服务器后尝试使用 API 时遇到 **403 Forbidden** 错误，导致所有模型功能停滞。
   - 提到“*迁移后似乎没有任何模型可以工作*”，引发了对服务器配置的担忧。
- **建议立即寻求支持**：针对 API 错误，一位社区成员建议通过邮件 [support@cohere.com](mailto:support@cohere.com) 联系以获得即时帮助。
   - 他们表示鼓励，称随时欢迎成员寻求帮助。
- **对该问题的先前认知**：另一位成员指出该 API 错误之前已被处理过，暗示这可能不是一个新问题。
   - 他们表示支持团队可能已经讨论过这个话题。
- **发起团队查询**：一位社区成员表示打算向团队询问该错误，以提供进一步的协助。
   - 他们的投入展示了解决问题的协作努力。


  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1290769437942616076)** (2 条消息): 

> - `Contextual Retrieval RAG`
> - `Multi-agent systems`
> - `Human in the loop feedback`
> - `TypeScript workflows` 


- **具有元数据的低成本 Contextual Retrieval RAG**: 一位成员分享了 @AnthropicAI 的新 RAG 技术，该技术通过在文档块（chunks）前添加元数据来增强检索，从而提高性能和成本效益。
   - 该方法根据文档中的上下文位置，更准确地引导 [检索过程](https://twitter.com/llama_index/status/1841210062167294287)。
- **人类反馈助力 Multi-agent 博客写作系统**: 演示了一个利用 **multi-agent systems** 的令人兴奋的博客写作 agent，并将 **human in the loop feedback** 整合到 TypeScript workflows 中。
   - 观众可以在这个 [现场演示](https://twitter.com/llama_index/status/1841528125123133835) 中看到该 agent 的实时写作和编辑过程。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1290771135289622589)** (37 条消息🔥): 

> - `LlamaIndex Infrastructure`
> - `GPU Utilization`
> - `HuggingFace LLM Usage`
> - `NVLM Support`
> - `Document Management Strategies` 


- **讨论 LlamaIndex 基础设施搭建**: 成员们分享了运行 LlamaIndex 的硬件规格见解，指出需求因模型和数据量而异。
   - 关键考虑因素包括运行 LLM 和 embedding 模型对 GPU 的必要性，并推荐了特定的 vector databases。
- **来自 NVIDIA 的 NVLM 备受关注**: 重点介绍了 NVIDIA NVLM 1.0 的发布，这是一个多模态大语言模型，强调了其在视觉语言任务中的领先能力。
   - 成员们推测了 LlamaIndex 内部的潜在支持，特别是关于巨大的 GPU 需求和加载配置。
- **HuggingFace LLM 集成问题**: 讨论集中在本地使用 HuggingFace 模型与通过 API 使用的对比，并就如何有效连接预存模型提出了建议。
   - 成员们还探索了使用特定技术和库来封装和增强模型交互。
- **文档管理的细分策略**: 一位成员询问了索引网页文章的最佳实践，质疑是应该保持合并还是作为独立文档。
   - 对话暗示了在索引框架内分别维护文章对改进文档管理的好处。
- **代码调整探索**: 一位用户深入探索了代码库，特别是 query 和 chat engines 之间的相似性，揭示了代码实现的细节。
   - 这突显了参数设置的复杂性，从而引出了对潜在功能需求的建议。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/nvidia/NVLM-D-72B">nvidia/NVLM-D-72B · Hugging Face</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/huggingface/">Hugging Face LLMs - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1290851562138112031)** (1 条消息): 

> - `Oracle AI Vector Search`
> - `LlamaIndex Framework`
> - `Semantic Search`
> - `Retrieval Augmented Generation (RAG)` 


- **Oracle AI Vector Search 革新语义搜索**: Oracle AI Vector Search 是 Oracle Database 的一项突破性功能，在 **semantic search** 领域处于领先地位，使系统能够基于 **含义而非关键词** 来理解信息。
   - 该技术与 **LlamaIndex framework** 结合使用，被定位为构建复杂 RAG pipelines 的 **强大解决方案**。
- **LlamaIndex 增强 Oracle 的 Vector Search**: 将 **LlamaIndex** 与 Oracle AI Vector Search 相结合，为 **retrieval augmented generation** 创建了强大的基础设施，增强了数据检索能力。
   - 这种集成有望提高处理和访问信息的效率，扩大其在 AI 应用中的影响力。



**提及的链接**: <a href="https://medium.com/@andysingal/oracle-ai-vector-search-with-llamaindex-a-powerful-combination-b83afd6692b2">Oracle AI Vector Search with LlamaIndex: A Powerful Combination</a>: Ankush k Singal

### **Torchtune ▷ #[announcements](https://discord.com/channels/1216353675241590815/1216353675241590818/1291042583572779100)** (1 条消息): 

> - `2024 PyTorch Contributor Awards`
> - `Salman Mohammadi`
> - `Community Contributions`
> - `PyTorch Growth Statistics` 


- **Salman Mohammadi 被提名为 2024 Contributor Awards 候选人**：我们自己的 **Salman Mohammadi** 因其在 GitHub 上的重大贡献以及在 Discord 社区的支持，被提名为 **2024 PyTorch Contributor Awards** 候选人。
   - Salman 的努力被认为是增强 PyTorch 生态系统不可或缺的一部分。
- **年度 Contributor Awards 彰显 PyTorch 的增长**：**年度 PyTorch Contributor Awards** 将在 **2024 PyTorch Conference** 上举行，旨在表彰对生态系统做出重大贡献的个人。
   - 今年有超过 **3,500** 名个人和 **3,000** 家机构参与了贡献，相比两年前仅有 **200** 家机构参与，PyTorch 见证了巨大的增长。
- **对社区贡献的致谢**：公告对 PyTorch 社区的**奉献**、**热情**和**辛勤工作**表示衷心感谢，这些对 PyTorch 的成功至关重要。
   - 每一份贡献都在推动 **AI** 和 **Machine Learning** 发展中发挥了**关键作用**。



**提到的链接**：<a href="https://pytorch.org/ecosystem/contributor-awards-2024">Announcing the 2024 PyTorch Contributor Awards</a>：未找到描述

  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1291041871740538892)** (18 条消息🔥): 

> - `Knowledge Distillation`
> - `Training Token Probabilities`
> - `Dataset Creation for Distillation`
> - `Optimization Flags in Torchtune` 


- **蒸馏中的训练 Token 与 One-hot 向量**：成员们讨论了 **Distillation**（知识蒸馏）是否因使用所有 Token 的概率而非 One-hot 向量而更有效，并指出更大的模型可以创建有用的 Latent Representations（潜表征）。
   - 有建议认为，从有标签和无标签的数据中学习可以在 **Distillation** 过程中“平滑” Loss Landscape（损失曲面）。
- **为知识蒸馏创建数据集**：一位成员询问了通过随机 Token 序列生成数据集的问题，随后有人建议从模型生成的内容中创建数据集。
   - 确认了对训练和蒸馏使用相同的数据集是一种可行的方法。
- **知识蒸馏的简化流程**：有效 **Distillation** 的最佳流程包括先在目标数据集上微调大模型，然后将知识蒸馏到较小的、未经过微调的模型中。
   - 成员们分享了关于跟踪实验和优化设置以降低计算成本的见解。
- **Torchtune 效率的优化标志**：成员们讨论了使用优化标志（如 `compile=True` 和 `dataset.packed=True`）来增强 **Torchtune** 的性能。
   - 建议利用 Nightly 版本以获得更好的性能，并考虑在 **LoRA** 中使用更高的 Rank 以获得更好的结果。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/1503.02531">Distilling the Knowledge in a Neural Network</a>：一种提高几乎任何机器学习算法性能的非常简单的方法是在相同数据上训练许多不同的模型，然后对它们的预测取平均值。不幸的是，使得...</li><li><a href="https://pytorch.org/torchtune/main/tutorials/llama_kd_tutorial.html">Distilling Llama3.1 8B into Llama3.2 1B using Knowledge Distillation &mdash; torchtune main documentation</a>：未找到描述词</li><li><a href="https://github.com/pytorch/torchtune/tree/main?tab=readme-ov-file#install-nightly-release)">GitHub - pytorch/torchtune: A Native-PyTorch Library for LLM Fine-tuning</a>：一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1290750266710294751)** (13 messages🔥): 

> - `H200s 部署`
> - `本地内部 LLMs`
> - `医疗数据法规`
> - `B100s 硬件计划` 


- **H200s 正在运送途中**：一位成员确认他们的 **H200s** 已经在路上了，并对最先到达的设备表示期待。
   - 他们提到拥有 **8x H200** 和 **4TB RAM**，展示了其令人印象深刻的硬件配置。
- **目标是本地内部 LLMs**：由于目前的 API 无法处理欧洲的医疗数据，关于部署内部 **LLMs** 的讨论非常活跃。
   - 一位成员强调，拥有 **本地基础设施 (local infrastructure)** 让处理敏感信息时感到更安全。
- **未来预期 B100s**：提到了获取 **B100s** 硬件的计划，预示着重大的升级。
   - 成员们表示希望不久能获得更多资源，进一步强化对本地处理能力的承诺。
- **应对医疗数据合规性**：一位成员表示，目前符合 **HIPAA** 标准的服务并不多，突显了医疗行业过去的挑战。
   - 他们指出 **欧盟法规 (EU regulations)** 甚至比 **HIPAA** 更严格，进一步增加了处理医疗数据人员的复杂性。


  

---



### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1290832459277008996)** (25 messages🔥): 

> - `Mojo 字面量 (Literals)`
> - `EC2 实例需求`
> - `Mojo 库导入`
> - `内存管理`
> - `导入行为差异` 


- **Mojo 字面量 (Literals) 仍在开发中**：一位成员确认 **literals** 在 **Mojo** 中尚不可用，并建议使用 `msg.extend(List[UInt8](0, 0, 0, 0))` 作为替代方案。
   - 另一位用户希望以后能实现 **try** 表达式。
- **在 EC2 实例上运行 Mojo**：一位用户在廉价的 **EC2 t2.micro** 实例上遇到了 **JIT session error**，这表明编译时可能存在内存限制。
   - 成员们建议可能至少需要 **8GB RAM**，而另一位成员确认 **2GB** 足以构建二进制文件。
- **Mojo 库导入的未来**：关于 **Mojo** 是否可能支持直接 **import library** CPython 库，而不是目前的 `cpython.import_module` 方法，引发了讨论。
   - 讨论中提到了 **Mojo** 和 **Python** 之间模块名称冲突的担忧，提出的解决方案是为 **Mojo** 设置 **导入优先级 (import precedence)**。
- **内存管理见解**：一位成员建议在 **EC2** 上使用 **swap** 内存，但警告这可能会因为消耗 **IOPS** 而导致性能问题。
   - 另一位用户确认在 **8GB** 内存上操作成功，同时也讨论了 **Mojo** 如何处理特定内存导入的担忧。
- **导入行为的差异**：一位用户注意到 **Mojo** 目前不像 **Python** 那样处理具有 **副作用 (side effects)** 的导入，这使兼容性变得复杂。
   - 对话强调，**Mojo** 编译器可能不需要复制 **Python** 导入的所有细微行为。


  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1290794065079566347)** (9 messages🔥): 

> - `Nova LLM 发布`
> - `Open Interpreter 中的 Function Calls`
> - `Open Interpreter Computer 角色`
> - `Trading View 经验`
> - `十月 House Party` 


- **Nova LLM 发布公告**: 🚀 [Nova](https://rubiks.ai/nova) 发布了其 Large Language Models 系列，包括 **Nova-Instant**、**Nova-Air** 和 **Nova-Pro**，旨在提供卓越性能，在 MMLU 上获得了 **88.8%** 的评分。
   - 值得注意的是，**Nova-Pro** 在推理和数学方面超越了竞争对手，在 ARC-C 上达到了惊人的 **97.2%**，在 GSM8K 上达到了 **96.9%**。
- **Open Interpreter 的 Function Calls 详解**: 一位用户询问是否可以在使用 Open Interpreter 的 Python 项目中定义自己的函数，并指出了 `interpreter.llm.supports_functions` 设置。
   - 另一位成员澄清说，虽然 Open Interpreter 可以即时编写函数，但定义严格的函数可以确保模型正确调用它们，并参考了 [OpenAI documentation](https://platform.openai.com/docs/guides/function-calling)。
- **探索 Open Interpreter 中的 Computer 角色**: 一位成员提到在 Open Interpreter 中发现了一个 “computer” 角色功能，可能有助于 Python 应用程序中的 function calling。
   - 他们参考了 [Open Interpreter 迁移指南](https://github.com/OpenInterpreter/open-interpreter/blob/main/docs/NCU_MIGRATION_GUIDE.md) 中的相关细节。
- **分享 Trading View 背景**: 一位用户分享了他们在 **Trading View** 使用 **Pine Script** 的背景以及全栈开发经验，包括使用 **React+Node** 和 **Vue+Laravel** 创建电子商务平台。
   - 他们宣布在完成之前的工作后，现在可以接受新的机会。
- **十月 House Party 提醒**: 发布了关于明天举行的 **十月 House Party** 的提醒，并为参与者提供了加入链接。
   - 鼓励参与者带着问题来，并分享最近使用 Open Interpreter 构建的任何项目。



**提到的链接**: <a href="https://x.com/RubiksAI/status/1841224714045264304">来自 Rubiks AI (@RubiksAI) 的推文</a>: 🚀 介绍 Nova：由 Nova 打造的下一代 LLM！🌟 我们很高兴宣布推出我们最新的 Large Language Models 系列：Nova-Instant、Nova-Air 和 Nova-Pro。每一款都旨在...

  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1291030215098765323)** (7 messages): 

> - `01 app 功能`
> - `OS 模式困惑` 


- **01 app 镜像了 Light 的功能**: 一位成员询问 **01 app** 是否可以访问与 **Light** app 相同的功能，例如屏幕访问。
   - 另一位成员确认 **服务器具有相同的能力**，使 01 app 的功能与 Light 保持一致。
- **访问程序受到质疑**: 同一位成员询问如何访问这些功能，因为 01 app 中没有可用的 **“os” 模式**。
   - 进一步的澄清指出，“os mode” 是与 **Open Interpreter** 特有的功能相关的，而不是 **01 app**。


  

---

### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1290750517362032738)** (3 messages): 

> - `Realtime API`
> - `Vision in Fine-Tuning`
> - `Prompt Caching`
> - `Model Distillation`
> - `Tool Use Podcast` 


- **推出 Realtime API**：全新的 [Realtime API](https://openai.com/index/introducing-the-realtime-api/) 实现了 **speech-to-speech** 能力，展示了交互式通信方面的进展。
   - 该 API 旨在通过对话式 AI 的即时响应来增强各种应用。
- **Fine-Tuning API 引入 Vision 功能**：OpenAI 宣布在 [Fine-Tuning API 中加入 Vision](https://openai.com/index/introducing-vision-to-the-fine-tuning-api/)，将 API 的功能扩展到文本处理之外。
   - 这一更新允许模型在训练期间整合视觉数据，为多模态 AI 应用开辟了新途径。
- **高效 Prompt Caching 发布**：[Prompt Caching](https://openai.com/index/api-prompt-caching/) 的推出承诺为近期处理过的输入 Token 提供 **50% 的折扣**和更快的处理速度。
   - 该功能通过减少重复 Token 输入相关的延迟和成本，显著优化了与 API 的交互。
- **模型蒸馏 (Model Distillation) 成为焦点**：[模型蒸馏](https://openai.com/index/api-model-distillation/) 专注于通过精炼模型权重管理来提高效率，从而提升性能。
   - 该技术旨在减轻计算负载，同时保持模型预测的高准确度。
- **新一期 Tool Use 播客发布**：今天的 *Tool Use* 节目邀请了一位知名的 AI 人物；点击[此处](https://www.youtube.com/watch?v=GRpkfSM2S7Q)观看。
   - 此外，在另一个[视频章节](https://www.youtube.com/watch?v=M3U5UVyGTuQ)中还有一场独立的讨论，强调了该领域持续的创新。



**提及链接**：<a href="https://x.com/sama/status/1841191074003341798?s=46&t=G6jp7iOBtkVuyhaYmaDb0w">Sam Altman (@sama) 的推文</a>：realtime api (speech-to-speech): https://openai.com/index/introducing-the-realtime-api/  vision in the fine-tuning api: https://openai.com/index/introducing-vision-to-the-fine-tuning-api/  prompt cach...

  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1290839551023255638)** (15 messages🔥): 

> - `LangChain support for GPT Realtime API`
> - `Using HuggingFace models in LangChain`
> - `Concerns about curly braces in prompt templates`
> - `Local hardware for AI model deployment`
> - `Feedback on Microsoft Copilot Studio` 


- **LangChain 对 GPT Realtime API 的支持**：成员们对 **LangChain** 何时支持新发布的 **GPT Realtime API** 表示好奇。
   - 目前仍处于猜测阶段，聊天中未分享明确的答案。
- **HuggingFace 模型可在 LangChain 中作为 Agent 使用**：确认了 **HuggingFace 模型可以作为 Agent 在 LangChain 中使用**，用于聊天和文本生成等任务。提供的一段代码示例展示了如何从 HuggingFace pipeline 创建 LangChain LLM。
   - 有关详细文档，建议成员查阅 **LangChain 文档**和相关的 **GitHub issue**。
- **处理 Prompt 模板中的花括号**：一位成员询问如何在 **LangChain** 的聊天 Prompt 模板中有效传递包含花括号的字符串，而不被解释为占位符。
   - 成员们正在寻求替代方案，因为目前的方法会在处理过程中改变输入。
- **本地 AI 模型硬件讨论**：一位用户提到他们的公司正在购买硬件以在本地运行 AI 模型用于内部聊天机器人，特别提到 **llama.cpp** 是一个常见的选择。
   - 讨论围绕本地部署首选的配置或模型展开。
- **征求对 Microsoft Copilot Studio 的意见**：成员们分享了关于 **Microsoft Copilot Studio** 的看法和经验，以及它与其他解决方案相比的价值。
   - 该询问引发了关于市场上替代方案的讨论，但被指出略微偏离主题。


  

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1290788700938174514)** (2 条消息): 

> - `Nova LLMs`
> - `LumiNova`
> - `OppyDev AI` 


- **Nova LLMs 以 SOTA 性能占据主导地位**：**Nova** 系列 LLMs 的发布包括 **Nova-Instant**、**Nova-Air** 和 **Nova-Pro**，它们以顶尖的 MMLU 分数超越了 **GPT-4o** 和 **Claude-3.5 Sonnet**，其中 **Nova-Pro** 以 **88.8%** 的高分领跑。
   - Nova-Pro 在推理测试 ARC-C 中表现优异，得分为 **97.2%**，在数学测试 GSM8K 中得分为 **96.9%**，自荐为 AI 交互的首选模型。更多详情请点击[此处](https://rubiks.ai/nova/release/)。
- **推出用于惊艳视觉效果的 LumiNova**：新发布的 **LumiNova** 模型专为卓越的图像生成而设计，承诺在视觉输出的质量和多样性方面达到无与伦比的水平。
   - 该模型通过增强 AI 的视觉创造力来补充 Nova 系列，为交互能力创造了新机遇。
- **OppyDev AI 的三个快速代码更新技巧**：**OppyDev AI** 推出了一段[视频指南](https://www.youtube.com/watch?v=g9FrwVOHTdE&t=187s)，演示了三种有效更新代码的简便方法。
   - 该资源旨在简化编码任务，使开发者能够更轻松地在 AI 辅助下增强其代码库。



**提到的链接**：<a href="https://x.com/RubiksAI/status/1841224714045264304">来自 Rubiks AI (@RubiksAI) 的推文</a>：🚀 隆重推出 Nova：由 Nova 打造的下一代 LLMs！🌟 我们很高兴宣布推出我们最新的大语言模型套件：Nova-Instant、Nova-Air 和 Nova-Pro。每一款都旨在...

  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1290750646697594983)** (8 条消息🔥): 

> - `NVIDIA's 72B model`
> - `Qwen 2.5 Deployment`
> - `Advancements in small models` 


- **NVIDIA 的 72B 模型媲美 Llama 3.1**：NVIDIA 刚刚发布了一个 **72B 模型**，在数学和编码评估中与 **Llama 3.1** 405B 大致持平，同时还具备视觉能力。
   - *哇，真是见证历史的时刻！*
- **Qwen 2.5 取得令人印象深刻的性能**：一名成员成功部署了 **Qwen 2.5** 34B，并称其性能**极其出色**，可与 **GPT-4 Turbo** 相媲美。
   - 对小模型进步的兴奋之情溢于言表，讨论集中在部署细节和视觉支持上。
- **小模型的潜力**：成员们对小模型变得如此强大感到惊讶，并思考其能力的极限在哪里。
   - *我们到底能把它推到多远？实际的极限是什么？*



**提到的链接**：<a href="https://x.com/phill__1/status/1841016309468856474?s=46">来自 Phil (@phill__1) 的推文</a>：哇，NVIDIA 刚刚发布了一个 72B 模型，在数学和编码评估中与 Llama 3.1 405B 大致持平，并且还具备视觉能力 🤯

  

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1290859255225782283)** (6 messages): 

> - `hf_mlflow_log_artifacts`
> - `sharegpt` 中的自定义指令格式
> - 数据集的 YAML 配置
> - 使用 Axolotl 进行指令微调 (instruction tuning)


- **关于 hf_mlflow_log_artifacts 的澄清**：一名成员询问将 **hf_mlflow_log_artifacts** 设置为 true 是否会导致模型 checkpoints 被保存到 mlflow。
   - 这表明用户对模型训练过程中日志记录机制的集成持续关注。
- **定义自定义指令格式**：讨论了在 **sharegpt** 格式中为数据集指定自定义指令格式的说明，强调了 YAML 配置的使用。
   - 关键步骤包括定义自定义提示词格式，并确保数据集为 JSONL 格式以实现成功集成。
- **数据集 YAML 配置指南**：对话概述了如何构建 YAML 配置以在 Axolotl 中预处理数据集，包括必需字段的示例。
   - YAML 文件中的特定占位符有助于根据用户需求实现指令微调的自定义。
- **利用 Axolotl 进行数据集预处理**：YAML 配置准备就绪后，即可配合 Axolotl 使用，为训练定制数据集格式。
   - 这种方法增强了训练过程的灵活性，允许针对特定任务进行量身定制的配置。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/openaccess-ai-collective/axolotl/tree/main/docs/dataset-formats/inst_tune.qmd#L1L190)">axolotl/docs/dataset-formats/inst_tune.qmd at main · axolotl-ai-cloud/axolotl</a>：欢迎提出关于 Axolotl 的问题。通过在 GitHub 上创建账号为 axolotl-ai-cloud/axolotl 的开发做出贡献。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=dbf8f9f4-96e9-49c1-ba23-25a02c10c4d8)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>：更快速地理解代码。
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1290810251183915101)** (6 messages): 

> - `Tiny Box 开箱`
> - `GitHub Bug 修复评审`
> - `PR 重构讨论` 


- **Tiny Box 开箱获得好评**：<@533837520580902912> 对通过代理运送到澳大利亚的 Tiny Box 进行了开箱，称赞了其**出色的包装**，并认为**木质底座**是一个很好的点缀。
   - *虽然担心纽约到澳大利亚的运输过程*，但他们指出负责加固包装的人员做得非常出色。
- **征求 Bug 修复 PR 的评审**：<@vladov3000> 正在寻求对[这个 Bug 修复 PR](https://github.com/tinygrad/tinygrad/pull/6815) 的评审，该 PR 解决了两次保存和加载 tensors 的问题。
   - 该 PR 旨在解决 issue **#6294**，解释了磁盘设备会保留未链接的文件且不会创建新文件的问题。
- **简化 PR 以提高清晰度**：<@georgehotz> 要求将 PR 做到**绝对精简**，暗示需要更高的清晰度。
   - <@vladov3000> 表示同意，并提议将更改拆分为两个不同的 PR，一个用于重构，一个用于修复。



**提及的链接**：<a href="https://github.com/tinygrad/tinygrad/pull/6815">修复两次保存和加载 tensor 的问题。由 vladov3000 提交 · Pull Request #6815 · tinygrad/tinygrad</a>：解决 #6294。详情见此评论。TLDR；磁盘设备会保留未链接的文件且不会创建新文件。这是我的第一次贡献，可能存在一些不足之处。姓名...

  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1290902236578185247)** (3 条消息): 

> - `tinygrad 代码收益`
> - `Python 生产力与 C 互操作性`
> - `UOp 与 UOP pool 优化问题`
> - `Compiler/Program/Allocator 挑战`
> - `tinygrad 中的分布式训练` 


- **tinygrad 代码提升编程技能**：探索 **tinygrad** 代码库对一名成员的日常工作编程产生了积极影响，展示了参与开源项目如何提高编程技能。
   - *作为一个副作用，它让我的日常工作编程变得更好。*
- **Python 的易用性与 C 互操作性表现出色**：成员们指出，尽管在 struct 方面存在一些限制，但 **Python** 在快速迭代方面极具生产力，且其 **C 互操作性** 非常有效。
   - 他们一致认为调用 C 函数非常直接，增强了底层操作的性能。
- **UOp 与 UOP Pool 优化的挑战**：一位成员表达了对 **UOp 与 UOP pool** 的挫败感，指出由于单个对象引用导致优化困难。
   - 他们主张使用一种能够通过整数句柄（integer handles）高效管理对象引用的存储类。
- **Compiler/Program/Allocator 效率问题**：成员对 **Compiler/Program/Allocator/Buffer 抽象** 中的低效表示担忧，特别是 clang 后端创建临时文件的问题。
   - 这种做法导致了显著的延迟，尤其是在后端重复创建相同临时文件的程序中，影响了 CPU 利用率。
- **关于分布式训练示例的咨询**：一位成员请求提供可移植到 **tinygrad** 的分布式运行 **PyTorch 训练脚本** 示例，引发了对当前功能的关注。
   - 他们提到拥有 AMD 算力，可能用于对比像 **Llama 3.1 70B** 这样的大型模型。


  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1291098948253253776)** (2 条消息): 

> - `垃圾信息问题` 


- **强烈的反垃圾信息情绪**：一位成员表达了对 **垃圾信息（spam）** 的强烈厌恶，强调了对骚扰消息的挫败感。
   - 这种情绪反映了数字平台面临的共同挑战，即垃圾信息会干扰沟通。
- **需要管理员关注**：提及 <@&825830190600683521> 的消息表明需要管理员有效地管理垃圾信息。
   - 这表明社区内对垃圾信息数量的持续关注。


  

---


### **LAION ▷ #[resources](https://discord.com/channels/823813159592001537/991938328763056168/1291078618809630733)** (1 条消息): 

> - `Sci Scope 新闻通讯`
> - `ArXiv 论文摘要`
> - `个性化研究更新` 


- **Sci Scope 新闻通讯发布公告**：来自 Sci Scope 的**个性化新闻通讯**现已上线，允许用户注册并选择偏好的研究领域，以便每周接收定制的新论文更新。
   - *再也不会错过与你工作相关的研究！* 用户可以[立即体验](https://sci-scope.com/)，以一种轻松的方式跟进 AI 领域的发展。
- **为繁忙专业人士提供的每周 AI 研究摘要**：该新闻通讯承诺扫描新的 **ArXiv 论文** 并为订阅者提供简明摘要，每周可节省 **数小时的工作时间**。
   - 它旨在通过**每周高层级摘要**，让用户更容易保持更新并筛选下一次阅读材料。
- **新用户专属优惠**：新用户可以注册 **1 个月的免费试用**，获得自定义查询和量身定制的体验。
   - 这一功能增强了用户体验，使其能够更相关、更高效地参与到快速发展的 AI 领域中。



**提及的链接**：<a href="https://sci-scope.com/">Sci Scope</a>：一个关于 AI 研究的 AI 生成新闻通讯

  

---

### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1291077935801040896)** (1 messages): 

> - `Personalized Newsletter`
> - `AI Research Updates`
> - `Weekly Summaries`
> - `ArXiv Papers`
> - `Sci Scope Features` 


- **个性化新闻通讯发布**：**Sci Scope** 推出了个性化新闻通讯，用户可以注册并每周接收根据其兴趣定制的新论文摘要。
   - 该功能旨在**节省时间**，并确保用户不会错过与其工作相关的重大更新，从而更轻松地保持信息同步。
- **轻松追踪 AI 研究**：该新闻通讯根据用户定义的偏好扫描新的 **ArXiv papers**，并将简明扼要的摘要直接发送到收件箱。
   - 该服务旨在帮助研究人员保持在 AI 研究的**最前沿**，而不会感到信息过载。
- **新用户免费试用**：Sci Scope 提供 **1 个月的免费试用**，让用户体验每周 AI 研究摘要带来的便利。
   - 这一举措旨在吸引用户尝试定制化新闻通讯，并发现其潜在价值。
- **便捷的阅读材料选择**：该服务将主题相似的最新 **AI research papers** 分组，以便于导航和选择。
   - 这种方法为用户提供了一种简单直接的方式来选择接下来的阅读材料，无需耗费过多精力。



**Link mentioned**: <a href="https://sci-scope.com/">Sci Scope</a>: 一个关于 AI 研究的 AI 生成新闻通讯

  

---


### **DSPy ▷ #[colbert](https://discord.com/channels/1161519468141355160/1250300504462856265/1290965001334554707)** (1 messages): 

> - `Code Similarity Search`
> - `Colbert for Code Search`
> - `Code Search Alternatives` 


- **探索代码相似度搜索**：一位成员询问了关于开始**代码相似度**研究的建议，表示有兴趣在代码搜索设置中使用 **Colbert**，即通过代码片段输出相关的代码文档。
   - 他们询问 **Colbert** 是否适用于此应用，或者在生效前是否需要进一步的 **finetuning**。
- **对代码搜索的替代方案持开放态度**：该成员还表示愿意接受有关如何进行**代码搜索**的其他想法，并寻求社区的指导。
   - 这种反思性的询问强调了在最初的 **native Colbert** 使用想法之外，寻求有用方法的协作精神。


  

---



### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1290858946898300930)** (2 messages): 

> - `Lab Release Schedule`
> - `Course Communication` 


- **实验课延迟一周**：一位成员询问了今天实验作业的发布情况，以及相关链接是否会发布到课程页面 [llmagents-learning.org](https://llmagents-learning.org/f24)。
   - 另一位成员确认，工作人员在发布实验作业之前还需要*一周时间*，遗憾地推迟了更新。
- **关于更新的说明**：同一位成员对实验发布缺乏更新表示担忧，并且找不到任何相关的电子邮件或公告。
   - 回复强调了在参与者等待有关课程进度的关键信息时存在的沟通鸿沟。



**Link mentioned**: <a href="https://llmagents-learning.org/f24">Large Language Model Agents</a>: 未找到描述

  

---

### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1290794861250871399)** (2 messages): 

> - `ML Paper Reading Group` (ML 论文阅读小组)
> - `Publishing Local LLM Apps` (发布本地 LLM 应用)
> - `Job Board Proposal` (招聘板块提案)
> - `Lumigator Introduction` (Lumigator 介绍)
> - `Upcoming Events` (即将举行的活动)


- **寻求 ML 论文阅读小组**：一位成员正寻求建立一个 [ML 论文阅读小组](https://discord.com/channels/1089876418936180786/1290380988450340864)，以促进对近期研究的讨论。
   - 该倡议旨在增强社区参与度和集体知识共享。
- **发布本地 LLM 应用的技巧**：非常感谢一位成员分享了关于成功将本地 **LLM-based apps** 发布到应用商店的宝贵见解。
   - 他们的贡献被认为对于那些希望驾驭应用发布流程的人至关重要。
- **招聘板块提案**：关于是否为社区职位发布创建一个 [招聘板块](https://discord.com/channels/1089876418936180786/1290677600527585311) 展开了讨论。
   - 该话题由一位表示有兴趣帮助将人才与机会联系起来的成员发起。
- **Lumigator 正式介绍**：<#1281660143251095634> 已由社区 [正式介绍](https://www.linkedin.com/posts/mozilla-ai_introducing-lumigator-activity-7246888824507613187-oTho)，重点展示了其功能和特性。
   - 此次介绍强化了社区展示创新项目的承诺。
- **宣布即将举行的活动**：重点介绍了几个即将举行的活动，包括专注于搜索技术的 [Hybrid Search](https://discord.com/events/1089876418936180786/1284180345553551431)。
   - 其他活动如 [Data Pipelines for FineTuning](https://discord.com/events/1089876418936180786/1290035138251587667) 旨在进一步丰富社区知识和成长。


  

---



### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1290788147231461449)** (1 messages): 

> - `Nova models`
> - `LumiNova`
> - `MMLU performance`
> - `AI Evolution` 


- **Nova 模型超越竞争对手**：介绍 [Nova](https://rubiks.ai/nova)：击败 **GPT-4** 和 **Claude-3.5** 的下一代大语言模型，其中 **Nova-Pro** 在 MMLU 上以 **88.8%** 的成绩领先。
   - 这些模型满足不同需求；**Nova-Air** 在各种应用中表现出色，而 **Nova-Instant** 提供快速且具成本效益的解决方案。
- **Nova 模型在基准测试中表现卓越**：Nova-Pro 取得了令人印象深刻的分数：推理方面 ARC-C 为 **97.2%**，数学方面 GSM8K 为 **96.9%**，编程方面 HumanEval 为 **91.8%**。
   - 这些基准测试巩固了 Nova 作为 AI 领域顶级竞争者的地位，展示了其非凡的能力。
- **LumiNova 彻底改变图像生成**：新推出的 **LumiNova** 为图像生成设定了高标准，承诺在视觉效果上提供无与伦比的质量和多样性。
   - 该模型补充了 Nova 系列，为用户提供了轻松创建惊人视觉效果的高级工具。
- **Nova-Focus 的未来发展**：展望未来，开发团队正在探索 **Nova-Focus** 和增强的 Chain-of-Thought 能力，以进一步突破 AI 的界限。
   - 这些创新旨在完善和扩展 Nova 模型在推理和视觉生成方面的潜在应用。



**提到的链接**：<a href="https://x.com/RubiksAI/status/1841224714045264304">来自 Rubiks AI (@RubiksAI) 的推文</a>：🚀 介绍 Nova：由 Nova 推出的下一代 LLMs！🌟 我们很高兴宣布推出我们最新的大语言模型套件：Nova-Instant、Nova-Air 和 Nova-Pro。每一款都旨在...

  

---



---



---



---



---



---



{% else %}


> 完整的逐频道细分内容已在邮件中截断。
> 
> 如果您想查看完整的细分内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请 [分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}