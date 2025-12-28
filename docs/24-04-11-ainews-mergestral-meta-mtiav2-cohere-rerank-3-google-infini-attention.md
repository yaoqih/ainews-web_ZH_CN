---
companies:
- meta-ai-fair
- mistral-ai
- cohere
- google
- stability-ai
- hugging-face
- ollama
date: '2024-04-11T22:56:47.954323Z'
description: '**Meta** 宣布了其新款 **MTIAv2 芯片**，旨在通过改进的架构以及与 PyTorch 2.0 的集成，实现训练和推理加速。**Mistral**
  发布了 **8x22B Mixtral** 模型，该模型被重新合并为一个稠密模型，从而有效地创建了一个 22B 的 Mistral 模型。**Cohere**
  推出了 **Rerank 3**，这是一款旨在增强企业搜索和检索增强生成 (RAG) 系统的基础模型，支持 100 多种语言。**Google** 发表了一篇关于
  **Infini-attention** 的论文，这是一种超可扩展的线性注意力机制，已在 1B 和 8B 模型上进行了演示，支持高达 100 万的序列长度。此外，**Meta
  的 Llama 3** 预计很快将开始推出。其他值得关注的更新包括 **Command R+**，这是一款在聊天机器人性能上超越 GPT-4 的开放模型，拥有
  128k 的上下文长度，以及 Stable Diffusion 模型和 RAG 流水线方面的进展。'
id: 537664d4-7b41-43f8-a208-481226a82524
models:
- mistral-8x22b
- command-r-plus
- rerank-3
- infini-attention
- llama-3
- sd-1.5
- cosxl
original_slug: ainews-mergestral-meta-mtiav2-cohere-rerank-3
people:
- aidan_gomez
- ylecun
- swyx
title: Mergestral、Meta MTIAv2、Cohere Rerank 3、Google Infini-Attention
topics:
- model-merging
- training-accelerators
- retrieval-augmented-generation
- linear-attention
- long-context
- foundation-models
- image-generation
- rag-pipelines
- model-benchmarking
- context-length
- model-performance
---

<!-- buttondown-editor-mode: plaintext -->> 2024年4月10日至4月11日的 AI 新闻。我们为您检查了 5 个 Reddit 分区、[**364** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)以及 **26** 个 Discord 社区（包含 **389** 个频道和 **4843** 条消息）。为您节省了预计阅读时间（以 200wpm 计算）：**540 分钟**。

今天有一系列小的更新，虽然都值得关注，但没有一个能明确成为“今日头条”：

- 全新的 8x22B Mixtral 被[这位大神](https://twitter.com/mejia_petit/status/1778390352082215129)**合并回了稠密模型（dense model）**——[从 8 个专家中提取出单个专家](https://x.com/danielhanchen/status/1778453454375231553)，实际上为我们提供了一个 22B 的 Mistral 模型。
- Meta 发布了他们的 [MTIAv2 芯片](https://twitter.com/ylecun/status/1778392841083117939?utm_source=ainews&utm_medium=email)，虽然你[买不到也租不到](https://x.com/soumithchintala/status/1778107247022751822)，但可以远观仰慕。
- [Cohere Rerank 3](https://twitter.com/cohere/status/1778417650432971225)，一个用于**增强企业搜索和 RAG 系统**的基础模型。它支持 100 多种语言，能够实现对多维度和半结构化数据的准确检索。[@aidangomez 的评论](https://twitter.com/aidangomez/status/1778416325628424339)。
- 谷歌关于 [Infini-attention 的新论文](https://twitter.com/swyx/status/1778553757762252863)展示了另一种超大规模线性注意力（linear attention）替代方案，此次展示了具有 100 万序列长度的 1B 和 8B 模型。

与[定于下周开始推出的 Llama 3](https://techcrunch.com/2024/04/09/meta-confirms-that-its-llama-3-open-source-llm-is-coming-in-the-next-month/?utm_source=ainews&utm_medium=email) 相比，这些都是小更新。

---

**目录**

[TOC] 

---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence。评论抓取功能现已上线，但仍有很大改进空间！

新模型与架构

- **Mistral 8x22B**：已可在 M2 Ultra 192GB 上以 4-bit 量化运行，在 [128GB RAM 的 M3 Max 上表现出色，达到每秒 4.5 个 token](https://i.redd.it/skiryihkhqtc1.gif)。可通过 [API](https://i.redd.it/eytf445jgntc1.png) 获取，并在 [Benchmarks](https://i.redd.it/2wnx1jjl8ptc1.jpeg) 中有所展示。
- **Command R+**：[首个在 Chatbot Arena 中击败 GPT-4 的开放模型](https://huggingface.co/chat/models/CohereForAI/c4ai-command-r-plus)，现已在 HuggingChat 上免费提供。实现了 [128k 上下文长度](https://www.reddit.com/r/LocalLLaMA/comments/1c0lkwo/how_does_command_r_achieve_128k_context/)，表现优于其他长上下文模型。
- **MTIA 芯片**：Meta 发布了其[下一代训练和推理加速器](https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/)，具有改进的架构、稠密计算性能、更高的内存容量和带宽。旨在与 PyTorch 2.0 完全集成。
- **UniFL**：[通过统一反馈学习改进 Stable Diffusion](https://www.reddit.com/gallery/1c0qsz8)，在 4 步推理中分别比 LCM 和 SDXL Turbo 提升了 57% 和 20%。
- **Infini-attention**：支持[高效的无限上下文 Transformer](https://arxiv.org/abs/2404.07143)，允许模型处理长程依赖关系。

Stable Diffusion 与图像生成

- **ELLA SDXL 权重**：[确认永远不会发布](https://www.reddit.com/r/StableDiffusion/comments/1c0ryb5/ella_sdxl_weights_confirmed_to_never_be_released/)，因为作者优先考虑发表论文而非可用性。社区对此表示失望，并转向关注 SD3。
- **SD 1.5**：仍被[部分用户视为“王者”](https://v.redd.it/08py75mvyptc1)，并展示了令人印象深刻的效果。
- **16 通道 VAE**：Stable Diffusion 训练的实验证明具有挑战性，模型难以达到 SDv1.5 的质量。社区讨论了[潜空间（latent space）通道或特定 VAE 对扩散模型训练的影响](https://www.reddit.com/r/StableDiffusion/comments/1c15qyd/how_do_the_vaes_latent_channels_or_the_specific/)。
- **CosXL**：来自 Stability AI 的新模型在[彻底改变图像编辑](https://sandner.art/cosine-continuous-stable-diffusion-xl-cosxl-on-stableswarmui/)方面展现出潜力。Demo 已在 Hugging Face 上线。

检索增强生成 (RAG) 与上下文处理

- **RAG 流水线评估**：分享了实用指南，强调了[构建生产级系统面临的挑战](https://www.reddit.com/r/MachineLearning/comments/1c0ryvz/d_a_practical_guide_to_rag_pipeline_evaluation/)，尽管基础 Demo 制作简单。
- **本地 RAG**：使用 [R2R、SentenceTransformers 和 ollama/Llama.cpp 进行部署的易懂教程](https://www.reddit.com/r/LocalLLaMA/comments/1c0vht1/easy_local_rag_sharing_our_stepbystep_tutorial/)。
- **RAG vs 长上下文模型**：[Gemini 概览](https://www.reddit.com/r/LocalLLaMA/comments/1c0iea2/rag_vs_large_context_models_a_gemini_overview/)对比了两种方法，讨论了未来的相关性及对具体用例的依赖。

开源努力与本地部署

- **LocalAI**：发布 v2.12.3 版本，包含[增强的一体化图像生成、Swagger API、OpenVINO 支持以及社区驱动的改进](https://www.reddit.com/r/LocalLLaMA/comments/1c0niro/localai_212_aio_images_improvements_swagger_api/)。
- **本地 AI 之旅**：用户分享了使用 [HP z620 和 ollama/anythingllm](https://www.reddit.com/r/OpenAI/comments/1c12lxh/my_journey_with_local_ai_so_far_any_tips/) 的经验，并寻求关于持久化和升级的建议。
- **Llama.cpp**：不再提供二进制文件，使得某些用户的编译变得更加困难。社区讨论了[挑战与替代方案](https://www.reddit.com/r/LocalLLaMA/comments/1c0gop8/why_is_llamacpp_no_longer_providing_binaries/)。
- **配备 ROCm 的 AMD GPU**：分享了通过 [Docker 使用 AUTOMATIC1111 和 kohya_ss](https://www.reddit.com/r/StableDiffusion/comments/1c15khf/using_amd_gpu_with_rocm_for_automatic1111_and/) 的指南，解决了兼容性问题。

提示工程与微调

- **用于微调的提示-响应示例**：用户就遵循特定输出格式所需的示例数量寻求建议，[估计范围从 50 到 10,000 个不等](https://www.reddit.com/r/MachineLearning/comments/1c0jmst/how_many_promptresponses_examples_do_i_need_for/)。
- **使用大型 LLM 生成提示词**：讨论了[为较小模型生成更好提示词](https://www.reddit.com/r/LocalLLaMA/comments/1c0ir44/using_llms_for_prompttuning/)的潜力，特别是在 RAG 框架中。

基准测试、对比与评估

- **Cohere Command R+**：尽管在 lmsys chat arena 基准测试中表现出色，但用户对其与 Claude 3、Qwen 1.5 72B 和 GPT-4 相比的[写作风格自然度表示轻微失望](https://www.reddit.com/r/LocalLLaMA/comments/1c0txo8/mild_disappointment_in_cohere_command_r/)。
- **Intel Gaudi**：据报道，在 LLM 训练方面比 NVIDIA 的产品[快 50% 且价格更低](https://www.reddit.com/r/LocalLLaMA/comments/1c0ir44/using_llms_for_prompttuning/)。
- **测试新方法**：讨论了[推荐的数据集、模型大小和基准测试](https://www.reddit.com/r/MachineLearning/comments/1c0vh48/d_what_would_you_recommend_testing_new_general/)，以向社区证明新架构/优化器的优越性。

迷因与幽默

- [Oh deer](https://i.redd.it/6n5kplurxmtc1.png)
- [拿到了我的 RTX 3060 12GB，但 SD1.5 依然好到让人舍不得离开](https://i.redd.it/y9qdwebg3rtc1.png)  
- [OpenAI，请发布你们的捉迷藏游戏吧，自第一次演示以来已经过去 5 年了。我只想整天和这些小神经元家伙一起玩](https://i.redd.it/7hp6psvpnmtc1.jpeg)
- [GPT 与 XFinity 客服聊天获取折扣](https://v.redd.it/5swabdnncotc1)


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3 Opus 完成，从 4 次运行中选取最佳结果。我们正在尝试使用 Haiku 进行聚类和流程工程（flow engineering）。

**LLM 进展**

- **Mixtral-8x22B 发布**：[@MistralAI](https://twitter.com/MistralAI/status/1778016678154211410) 发布了 Mixtral-8x22B，这是一个拥有 **176B 参数的 MoE 模型，包含约 40B 激活参数**，上下文长度为 65k，采用 Apache 2.0 许可证。早期评估显示其 **MMLU 得分为 77.3%**，优于其他开源模型。[@_philschmid](https://twitter.com/_philschmid/status/1778051363554934874) [@awnihannun](https://twitter.com/awnihannun/status/1778054275152937130)
- **GPT-4 Turbo 改进**：新的 GPT-4 Turbo 表现出显著改进，特别是在代码基准测试中，在大多数任务上超越了 Claude 3 Sonnet 和 Mistral Large。[@gdb](https://twitter.com/gdb/status/1778071427809431789) [@gdb](https://twitter.com/gdb/status/1778126026532372486) [@bindureddy](https://twitter.com/bindureddy/status/1778108344051572746)
- **Command R+ 发布**：[@cohere](https://twitter.com/cohere/status/1778417650432971225) 发布了 Command R+，这是一款具有强大多语言能力的全新开放词表模型，在**某些非英语基准测试中表现优于 GPT-4 Turbo**。它拥有高效的 Tokenizer，可实现更快的推理和更低的成本。[@seb_ruder](https://twitter.com/seb_ruder/status/1778385359660867744) [@aidangomez](https://twitter.com/aidangomez/status/1778391705663729977)
- **Gemini 1.5 Pro**：Google 发布了 Gemini 1.5 Pro，**增加了音频和视频输入支持**。现在可通过 API 在 180 多个国家/地区使用。[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1778063609479803321)

**高效 LLM**

- **Infini-attention 实现无限上下文**：Google 推出了 Infini-attention，这是一种将 Transformer LLM 扩展到**具有有限内存和计算量的无限长输入**的高效方法。它将压缩内存（compressive memory）整合到 attention 中，并内置了局部和长期 attention 机制。 [@_akhaliq](https://twitter.com/arankomatsuzaki/status/1778230430090592454) [@_akhaliq](https://twitter.com/arankomatsuzaki/status/1778234586599727285)
- **将 LLaMA Decoder 适配至视觉任务**：这项工作研究了将 decoder-only 的 LLaMA 适配到视觉任务。直接应用因果掩码（causal mask）会导致 attention 崩溃，因此他们**重新定位了 class token 并使用了软掩码（soft mask）策略**。 [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1778237179740688845)
- **llm.c**：[@karpathy](https://twitter.com/karpathy/status/1778153659106533806) 发布了 llm.c，这是一个**约 1000 行 C 语言编写的 GPT-2 实现，直接调用 CUDA kernel**。虽然灵活性较低且比 PyTorch 慢，但它提供了核心算法的一个简单、极简的实现。 [@karpathy](https://twitter.com/karpathy/status/1778128793166856368) [@karpathy](https://twitter.com/karpathy/status/1778135672420966788)

**Robotics and Embodied AI**

- **学习敏捷足球技能**：DeepMind 使用强化学习训练 AI Agent 展示**敏捷的足球技能，如转身、踢球和追逐球**。这些策略可以迁移到真实机器人上，并结合起来实现进球和扑救。 [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1778377999202541642)
- **OpenEQA 基准测试**：Meta 发布了 OpenEQA，这是一个通过开放词汇问题来衡量具身 AI Agent 对物理环境理解能力的基准测试。目前的视觉语言模型在**性能上远低于人类，尤其是在空间理解方面**。 [@AIatMeta](https://twitter.com/AIatMeta/status/1778425321118732578) [@AIatMeta](https://twitter.com/AIatMeta/status/1778425322645422396)

**Hardware and Systems**

- **MTIAv2 推理芯片**：Meta 发布了其第二代推理芯片 MTIAv2，**采用 TSMC 5nm 工艺制造，具有 708 TFLOPs 的 int8 算力**。它使用标准的 PyTorch 堆栈以保证灵活性，并针对 Meta 的 AI 工作负载进行了优化。 [@ylecun](https://twitter.com/ylecun/status/1778392841083117939) [@AIatMeta](https://twitter.com/AIatMeta/status/1778083237480321502) [@soumithchintala](https://twitter.com/soumithchintala/status/1778087952964374854)

**Miscellaneous**

- **Rerank 3 发布**：[@cohere](https://twitter.com/cohere/status/1778417650432971225) 发布了 Rerank 3，这是一个用于**增强企业搜索和 RAG 系统**的基础模型。它支持 100 多种语言的多维度和半结构化数据的准确检索。 [@aidangomez](https://twitter.com/aidangomez/status/1778416325628424339)
- **Zephyr 对齐**：一个新的 Zephyr 模型在包含 7000 条偏好对比的数据集上使用 **Odds Ratio Preference Optimization (ORPO)** 进行了训练，在 IFEval 和 BBH 上取得了高分。代码已在 Alignment Handbook 中开源。 [@osanseviero](https://twitter.com/osanseviero/status/1778430866718421198) [@_lewtun](https://twitter.com/osanseviero/status/1778430868387778677)
- **Suno Explore 上线**：[@suno_ai_](https://twitter.com/suno_ai_/status/1778430403973447708) 推出了 Suno Explore，这是一种**发现由其 AI 系统生成的全新音乐流派的听觉体验**。
- **Udio 文本生成音乐**：Udio 是来自 Uncharted Labs 的新型文本生成音乐 AI，可以**根据文本描述生成多种风格的完整歌曲**。早期的演示效果非常令人惊叹。 [@udiomusic](https://twitter.com/udiomusic/status/1778049129337192888)

---

# AI Discord 回顾

> 摘要之摘要的摘要

- **新 AI 模型发布在即，备受期待**：AI 社区正热切期待几个新模型的发布，包括预计在未来 1-3 周内推出的 Stability.ai 的 **SD3**，Meta 已确认即将推出的 **Llama 3**（[TechCrunch 文章](https://techcrunch.com/2024/04/09/meta-confirms-that-its-llama-3-open-source-llm-is-coming-in-the-next-month/)），以及 MistralAI 的 **Mixtral-8x22b** 指令微调版。此外，Sophia Yang 预告了一个全新的 Apache 2.0 协议模型，其在初步的 [AGIEval 结果](https://x.com/jphme/status/1778030213881909451)中表现优于其他开源基座模型。

- **Mixtral 模型性能表现亮眼**：新发布的 **Mixtral-8x22b** 引起了轰动，根据 [AGIEval 结果](https://x.com/jphme/status/1778030213881909451)，它在 PIQA 和 BoolQ 等基准测试中显著优于其他开源模型。讨论还强调了 **Mixtral 8x7b** 模型即使在量化后仍具有强劲性能。社区正在分析这些模型的能力，并[将其与 GPT-4 及其他领先系统进行比较](https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1/discussions/4)。

- **CUDA 与量化效率提升**：在 CUDA MODE Discord 中，一位用户报告称实现了 GPT-2 的纯 CUDA 前向传递，每轮迭代耗时 110ms，性能超越了 PyTorch。目前正在探索使用 CUDA 的 C 子集、内联汇编（inline assembly）和协作组（cooperative groups）进行优化。HQQ (Half-Quadratic Quantization) 社区正在深入研究量化脚本、int4 kernel 的性能以及困惑度（perplexity）分数的差异，并在 [GitHub 上分享了最新的 HQQ 代码](https://github.com/mobiusml/hqq/blob/ao_int4_mm/hqq/core/torch_lowbit.py)。

- **新应用与集成让 AI 触手可及**：多个新的 AI 应用和集成方案发布，包括具备 GPT-4 和 Vision AI 功能的 **GPT AI**，提供免费高级模型 API 的 **Galaxy AI**，用于直观构建应用的 **Appstorm v1.6.0**，以及 **Perplexity AI 与 Raycast** 的合作——为 Raycast 订阅者免费提供 Perplexity Pro（[Raycast 博客文章](https://www.raycast.com/blog/more-ai-models)）。OpenAI 的 ChatGPT 用户数也达到了 1 亿，并正在转向预付费信用系统。

- **AI 硬件与基础设施的进展**：Meta 展示了其 **Meta Training and Inference Accelerator (MTIA)**，在 90W TDP 下可提供 354 TFLOPS/s (INT8) 的算力用于 AI 工作负载（[Meta 博客文章](https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/)）。Intel 即将推出的 **Lunar Lake CPUs** 将配备 45 TOPS 的 NPU，以便在本地运行 Microsoft 的 Copilot AI。芯片设计商与台积电（TSMC）等代工厂之间的供应链动态也成为了关注焦点。

---

# 第一部分：Discord 高层级摘要

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SD3 的到来备受期待**：社区里充斥着关于 **SD3** 发布时间的询问，预计将在未来 1-3 周内推出。Beta 测试正在顺利进行中，人们对强大的基座模型和发布后的微调（finetuning）能力寄予厚望。

- **UI 对决：Forge 比 ComfyUI 和 Automatic1111 更受青睐**：关于 **Stable Diffusion** 等图像生成模型用户界面的优劣争论中，**Forge** 的速度和模型管理、**ComfyUI** 的精细控制机制，以及 **Automatic1111** 卓越的图生图（image-to-image）性能和 controlnet 实现均被提及。

- **通往更好 UI 之桥：考虑 LaVi Bridge**：虽然目前人们对将 **LaVi Bridge**（一种类似于 **ELLA** 的技术）与 **ComfyUI** 合并很感兴趣，但目前尚无具体的集成计划，AI 工程师们正期待未来的发展。

- **VRAM：AI 训练者的最佳伙伴**：在 AI 模型训练的讨论中，重点仍然是 VRAM 的关键作用。共识认为，减少 VRAM 占用可能会限制扩展能力，或者需要通过改进 AI 功能来抵消，从而利用额外的显存容量。

- **探索更智能的 Inpainting 方法**：关于结合使用“Fill”功能与 **ControlNet Inpaint** 来处理移除背景后的图像的咨询，表明 AI 社区正在不断寻求增强的 Inpainting 技术。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **CodeGemma 的重任**：用户报告了在微调 **CodeGemma** 模型时的 **VRAM 消耗问题**，并建议使用 **paged_adamw_8bit** 优化器以提高效率。社区正在关注 **optimizer techniques** 和 **Apple Silicon support** 的进展，分享了与 Silicon 支持相关的 GitHub issue ([Apple Silicon Support #4](https://github.com/unslothai/unsloth/issues/4))，并期待能提升性能的移植项目。

- **Triton DSL 学习曲线**：随着 [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) 等有用资源的分享，学习 **Triton DSL** 的兴趣达到顶峰。然而，用户在处理 **Gemma** 等模型时面临 **Out of Memory (OOM)** 挑战，这预示着可能会转向更高效的替代方案。

- **Unsloth 微调轶事**：使用 **Unsloth** 进行微调的经验引发了关于 **VRAM 需求**和重复错误的讨论。当 **Andrei Karpathy** 认可 **Unsloth** 时，社区反响热烈，强调了围绕微调实践持续对话的必要性。

- **Perplexity 之旅**：关于 **Perplexity Labs** 进行 **instruction tuning** 的讨论揭示了输出结果与搜索结果相似的观察。即将推出的 **Mixtral-8x22b model** 备受关注，人们对其潜力和边际优势产生了浓厚兴趣。

- **周边建议激增**：偶然提到的 **Unsloth** 周边商品引发了轻松的闲聊，表明了社区对品牌的认同感。同时，有信号表明团队需要更多设计师，这可能会推动未来的合作或招聘。

- **Unsloth 资源配置**：成员们寻求关于 **multi-GPU support** 的澄清，揭示了一个正在开发中的 **pre-alpha** 功能，该功能可能对超过四个 **GPU** 的使用进行许可控制。对话模型的 **dataset formatting** 进展已被消化，并对群聊中的配对方法提出了见解。

- **内核横向对比**：为了一篇研究论文对开源内核竞争力的深入研究，揭示了对 **Unsloth** 集成能力的赞赏。在认可开源项目贡献的背景下，强调了使用 **multi-GPU support** 时的伦理考量。

- **Unsloth AI 部署一帆风顺**：训练后的部署咨询引导用户查阅 [Unsloth documentation](https://github.com/unslothai/unsloth/wiki)，重点关注模型保存和部署设置。讨论确认虽然 **Unsloth** 是为其内部功能量身定制的，但适应更广泛用例的可能性已近在咫尺。

- **StegLLM 成为焦点**：创新模型的展示中出现了 **StegLLM**（一种带有 **backdoor mechanism** 的语言模型）和 **StegBot**，两者均使用 **Unsloth** 进行了精炼 ([StegLLM](https://huggingface.co/AshScholar/StegLLM), [StegBot](https://huggingface.co/oofnan/stegBot/tree/main))。这些发布突显了社区在模型功能方面的尖端实验。

- **Sophia 的进步与 AdaLomo 的突破**：**Sophia** 的性能提升受到关注，可能达到 **AdamW** 的效率，而 **AdaLomo** 在 **LLaMA** 模型上的测试展示了内存效率 ([AdaLomo PDF](https://arxiv.org/abs/2310.10195))。这些见解激励了社区，因为他们正关注其对模型优化的影响。

- **LLaMA 3 期待**：关于即将推出的 **multi-GPU support** 的预告取决于 **LLaMA 3** 的发布，这为社区未来的工程壮举奠定了基础。

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**关于 LLM 性能的大胆言论**：LM Studio 的用户报告了 GPT "GPT Builder" 的性能问题，并讨论了最佳 Prompt 策略，相比系统生成的 Prompt，他们更倾向于*手动编写*。此外，Mistral 家族的新成员 **8x22b 模型** 备受关注，目前正在等待 **GGUF 量化** 版本，以便在 LM Studio 中运行。

**代码能力对比**：关于 Python 编程模型能力的讨论主要集中在 **GPT-4、Claude3 Opus 和 Phind**，特别提到了 Phind 具备联网查询的独特功能。**Mixtral 8x22B** 的发布因其与 **Command R+** 的对比而引发热议，后者在底层编程和数学问题上具有优势，并支持使用 LaTeX 格式化响应。

**挑战硬件极限**：成员们交流了 AI 模型的硬件适配知识，提到了 **Codegemma** 运行的成功案例与崩溃情况，以及 **Mac Studio (192 GB RAM)** 在运行大型模型时的出色表现。关于云端成本的讨论指向了替代方案，例如使用消费级硬件进行本地部署以提高性价比，并提到了 AWS 最近取消了数据传出（egress）费用。

**Beta 版本亟需修复与新功能**：LM Studio 用户指出 **0.2.19 beta** 版本需要进行故障排除，提到了 LaTeX 渲染以及与 n8n 等其他工具对接时的挑战。特别强调了在 AMD ROCm 平台上运行模型的问题，2.17 之后的 Beta 版本运行效果不尽如人意。

**模型部署策略浮现**：通过对话，一种关于模型部署的叙事逐渐浮现，强调了优化方式，如权衡云端与本地部署，以及本地硬件增强的实用性（例如利用 eGPU 和探索云端 GPU 服务集成）。有人呼吁开发适合在 3080 GPU 上托管、用于 AutoGen 任务的 **12GB AI 模型**，但目前尚无直接解决方案。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**Google 代码助手上线**：[Google 的 CodeGemma](https://www.youtube.com/watch?v=Gb--4supXoo)（一个 7B 模型）旨在通过先进的代码补全功能提高开发人员的生产力，体现了 AI 在编程辅助领域日益增长的影响力。

**World-Sim 的回归**：World-Sim 爱好者们正为它的重新发布做准备，在 **teknium 的神秘信息** 轰炸下，大家纷纷猜测其可能的新功能和应用场景，范围涵盖从教育到 AGI 开发。

**弥合 AI 沟通鸿沟**：社区讨论了 LLM 中双向注意力（bidirectional attention）的优势，引用了 **SD3** 在文本渲染方面的成功，并研究了 **Infini-attention**（[研究论文](https://arxiv.org/abs/2404.07143)）作为在 Transformer 模型中高效处理长输入的方法。

**模型微调的财务挑战**：围绕 **Nous-Hermes-8x22b** 等大型模型微调的讨论揭示了成本问题，用户正在评估 **QLoRA** 和 **LISA** 相对于全参数微调的优劣，而像 Vast 这样的云服务虽然提供了强大的 GPU 选项，但价格昂贵。

**备受期待的模型进展引发热议**：随着 Meta 宣布 **Llama 3 即将发布**（[TechCrunch 文章](https://techcrunch.com/2024/04/09/meta-confirms-that-its-llama-3-open-source-llm-is-coming-in-the-next-month/)），以及 **MistralAI** 预计将发布其 **Mixtral-8x22b** 的 *instruct* 版本，社区对新的 AI 里程碑保持高度期待。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Nuzzy Bot 加入聊天**：引入了一个名为 **Nuzzy** 的互动机器人用于用户参与，建议使用专用频道进行交流，并通过特定命令激活。

- **Udio 备受瞩目**：**Udio 音乐生成器** 成为热门话题，因其每位用户每月提供 1200 首免费歌曲以及创作 90 秒歌曲的能力而受到关注，被视为 Suno 的强力竞争对手，在 Twitter 和 Reddit 上引发了大量对比讨论。

- **Nvidia GPU 对比升温**：通过一份[分享链接](https://www.semianalysis.com/p/nvidia-blackwell-perf-tco-analysis)，对 Nvidia 的 Blackwell GPU 进行了详细分析，对比了 B100、B200 和 GB200 型号，讨论了它们的总体拥有成本 (TCO) 和推理成本。

- **AI 工程手册 (AI Engineering Playbook) 正在编写中**：呼吁共同努力创建一份 **AI 工程手册**，重点关注从 Jupyter notebooks 到生产环境的过渡。敦促在部署 LLM 方面有丰富经验的高级工程师和团队负责人贡献力量。

- **1-bit LLMs 展示与技术故障**：“1-bit Large Language Models” 论文演示在 Discord 上遇到了技术问题，促使未来的会议可能会更换平台，同时也引发了关于 **BitNet b1.58** 等模型的效率和实际应用的讨论，及其在 [GitHub 上的实现 BitNet-Transformers](https://github.com/Beomi/BitNet-Transformers)。Mixture of Experts (MoEs) 模型也受到了关注，相关讨论引用了 [Hugging Face 博客文章](https://huggingface.co/blog/moe)，并进一步探讨了其使用场景以及专家专业化 (expert specialization) 和语义路由 (semantic routing) 的底层概念。



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hugging Face 推出免费的 Command R+**：Hugging Chat 现在可以免费访问 [Command R+ 模型](https://huggingface.co/chat/models/CohereForAI/c4ai-command-r-plus)，该模型具备网页搜索与聊天集成功能。
- **暂停并恢复你的模型训练**：Hugging Face 的 `Trainer` 支持通过 `resume_from_checkpoint` 函数暂停和恢复训练会话，这是 AI 工程师管理长周期训练过程的一个实用功能。
- **多语言提取模型表现亮眼**：用户对一个[多语言信息提取演示](https://huggingface.co/spaces/urchade/gliner_multiv2.1)非常感兴趣，展示了一个小型但功能强大的模型在处理跨语言任务时的高效性。
- **Podman 在 AI 安全领域崭露头角**：讨论的关于 Podman 的视频强调了其在增强微服务中 AI 安全性的作用，被推崇为在容器化环境中部署 AI 时比 Docker 更安全的替代方案 [Podman 视频](https://youtu.be/YtxdWLSPo0U)。
- **使用 Diffusers 进行多 GPU 编排**：*Diffusers* 中的 `device_map` 功能因其能有效地在多个 GPU 之间分配模型流水线 (pipeline) 而受到关注，这对于在显存 (VRAM) 较小的 GPU 上运行的用户具有重要意义 [Diffusers 文档](https://huggingface.co/docs/diffusers/main/en/training/distributed_inference#device-placement)。



---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 与 Raycast 联手**：Perplexity AI 为新的 Raycast Pro 年度订阅用户提供 **3 个月的免费 Perplexity Pro**，AI 插件订阅者的优惠更是延长至 6 个月。此次合作强调了 AI 在 Mac 上的无缝集成，详见 Raycast 的 [更多 AI 模型博客文章](https://www.raycast.com/blog/more-ai-models)。

- **ChatGPT 突破用户里程碑**：根据其 [公告](https://openai.com/blog/chatgpt)，OpenAI 的 ChatGPT 在发布后短短两个月内，**月活跃用户数就达到了 1 亿**。

- **AI 模型实践**：工程师们辩论了小上下文窗口 AI 模型与大上下文窗口模型的效率和有效性，其中 **Opus 4k** 在快速查询中更受青睐。社区还解决了诸如关闭 Claude 3 Opus 中的 Pro 模式，以及寻找 Perplexity 局限性（如在回复中包含图片）的变通方法等问题。

- **集成见解与 API 困扰**：公会讨论了 AI 与 Raycast 等工具集成的优势，并思考了 **Perplexity API**，指出可以在回答中模拟网页版，并引用了官方 [模型文档](https://docs.perplexity.ai/docs/model-cards)。一位用户通过重新激活支付自动充值解决了 API 认证和 401 错误的问题。

- **Perplexity 搜索反映趋势与好奇心**：公会成员利用 Perplexity 进行各种查询——从视频游戏分析、农产品的财务评估，到技术爱好者探索 Gaudi 3 等 AI 芯片的进展。这些搜索显示出利用 AI 在技术和金融领域获取多样化见解的浓厚兴趣。



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Meta 发布 AI 省电神器**：Meta 的 *AI Training and Inference Accelerator (MTIA)* 表现出色，在功耗仅为 90 瓦的情况下实现了 **354 TFLOPS/s (INT8)** 的性能，彰显了其对可扩展 GenAI 产品和 AI 研究的承诺。[Meta 官方博客文章](https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/) 概述了其 AI 基础设施的发展雄心。

- **CUDA 征服**：**纯 CUDA 实现** 的惊人效率使得 GPT-2 模型运行的前向传播时间仅为 110ms，优于 PyTorch 的 180ms，引发了关于微调和优化的讨论，范围涵盖内联汇编到使用 **cooperative groups** 和 C++。CUDA 开发对话包括分享 [LayerNorm kernel 示例](https://godbolt.org/z/6zq6Kfecc)，并辩论了 CUDA 编程中 C 与 C++ 的优劣。

- **CUDA Kernel 集合**：llm.c 仓库现在包含了一系列 **CUDA kernel**，而另一个独立项目报告称，使用他们自己的闪电级线性层库在 **4090 GPU 上实现了 >300 tflops**。社区讨论了其影响并详细说明了性能对比，重点是 FP16 精度，并暗示未来将添加梯度支持。可以在 [GitHub 上的 llm.c](https://github.com/karpathy/llm.c/tree/master/dev/cuda) 查看这些 kernel，快速线性层库为 [GitHub 上的 torch-cublas-hgemm](https://github.com/aredden/torch-cublas-hgemm)。

- **学习小组集结**：关于学习会议的建议围绕 **PMPP 书籍讲座** 展开，提供了一个互动交流的论坛。已为参与者建立了一个小组，第一节课将在不同时区友好的时间开始，可通过 [学习小组邀请](https://discord.gg/XwFJRKH9) 加入。

- **HQQ 引发量化困惑**：激烈的讨论转向了 **量化基准测试** 和可复现性，特别是 Hugging Face 模型的性能脚本，可在 [hqq core torch_lowbit](https://github.com/mobiusml/hqq/blob/ao_int4_mm/hqq/core/torch_lowbit.py) 获取。int4 kernel 性能的挑战以及量化 Transformer 中的 perplexity 指标突显了技术叙事，强调了对量化霸权的追求。

- **可视化工具开发中**：**triton-viz 聊天机器人** 的增强功能正在进行中，计划改进超链接和逐步代码注释，以提升机器人的功能和可用性。



---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**知识缩放 (Knowledge Scaling)**：最近的一篇 [论文](https://arxiv.org/abs/2404.05405) 提出，语言模型在 **每个参数 2 bits 知识** 时达到极限，这引发了关于训练时长和模型架构等各种因素如何影响这一限制的讨论。社区认为其影响不容小觑，并正考虑进行深入讨论以澄清论文的见解。

**RNNs 再次兴起**：研究表明，为 Transformer 开发的可解释性工具同样适用于现代 RNN，并在 **Mamba** 和 **RWKV** 模型中展示了有效性。这一发现得到了 [配套论文](https://arxiv.org/abs/2404.05971) 和 [代码库](https://github.com/EleutherAI/rnngineering) 的支持，突显了 RNN 在语言建模中的复兴，该 [研究](https://x.com/norabelrose/status/1777975663590531533) 中体现了强大的社区协作。

**微调技巧 (Fine-Tuning Finesse)**：一种新技术 **子集微调** (SubTuning) 正在引起关注，它通过仅调整部分层来实现具有竞争力的性能，这可能会减轻多任务学习等任务的计算需求。[论文](https://arxiv.org/abs/2302.06354) 详细介绍了这种方法，与优先考虑微调预算限制的讨论相一致。

**模型评估博览 (Model Evaluation Expo)**：[Mixtral 8x22B 模型](https://x.com/jphme/status/1778030213881909451) 凭借其 AGIEval 结果备受瞩目，人们对其社区发布充满期待。与此同时，对在选举安全中利用 Deepfakes 等 AI 的担忧，以及关于下载 The Pile 用于研究的查询也相继出现，强调了学术诚信。

**聊天模板演进 (Chat Templating Evolution)**：**lm-evaluation-harness** 项目中关于 **聊天模板化 (chat templating)** 的 Pull Request 引起了关注，特别是 [Hailey 针对 HF 模型的 PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/1287#issuecomment-1967469808) 和另一个 [开放 PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/1578)。社区看到了通过为 `apply_chat_template` 添加批处理操作支持来增强项目的机会。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Mojo 的新阶段**：**Mojo** 语言的发展包括 [路线图亮点](https://docs.modular.com/mojo/roadmap#lifetime-tracking-inside-collections)，揭示了未来的增强功能和核心编程特性的优先级。同时，关于集成 Objective-C 或 AppKit 以构建针对 MacOS 的新 UI 库的讨论正在酝酿，社区还就 Mojo 中的 GUI 设计模式和错误处理实践展开了辩论，凸显了一个正处于大幅增长边缘的充满活力的生态系统。

**高级存储策略分析**：一篇 [Modular 博客文章](https://www.modular.com/blog/row-major-vs-column-major-matrices-a-performance-analysis-in-mojo-and-numpy) 探讨了行主序 (row-major) 和列主序 (column-major) 内存布局对性能的影响。它阐明了开发者面临的权衡，以及在使用 Mojo 和 NumPy 等语言和库时存储顺序的影响。

**社区参与和贡献增加**：开源参与度有所提高，对 **Modular standard library** 和 `Lightbug` 框架等项目做出了重大贡献，后者现在的性能已优于 Python 的 Flask。Mojo 的词法灵活性在 `mojo-ui-html` 中添加键盘事件处理以及创建 `lightbug_api` 中得到了体现，表明社区驱动的势头正盛。

**在 UI 开发中使用 Mojo 进行创新**：**Mojo** 在 UI 开发中的应用通过一个受 `lipgloss` 启发、简洁的终端文本渲染工具——可在 [GitHub](https://github.com/thatstoasty/mog/blob/main/examples/readme/layout.mojo) 上获得——以及 Basalt 的视觉实力得到了展示。这些发展表明，人们正在努力利用 Mojo 提升终端应用程序的美学和功能能力。

**Modular 思想保持同步**：Modverse 社区通过 "Modverse Weekly - Issue 29" 通讯（可在 [Modular Newsletters](https://www.modular.com/newsletters/modverse-weekly-29) 获取）和提供简短更新的推文保持信息同步，所有这些都维持了这个技术中心内的知识交流。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**量化技术的飞跃**：讨论集中在量化后将 **Mistral** 等模型适配到具有 16k 上下文的**单张 24GB 显卡**上的挑战，并有证言验证了 **Mixtral 8x7b** 的性能。

**对 MLLMs 的好奇**：社区成员对多模态大模型（如 **LLaVA** 和 **Qwen VLM**）表现出浓厚兴趣，但在许可证导航和微调指导方面面临资源有限的问题。

**推理服务器的 GPU 选择困境**：工程师们讨论了在推理服务器上使用 **Nvidia 4090s** 替代 **3090s** 的可行性，考虑到缺乏 NVLink 和 PCIe 5，有人建议更好的卡间带宽可能使 3090s 更合适。

**黑客松活动提醒**：重点介绍了将于 5 月 11 日举行的 **Samsung Next 2024 Generative AI Hackathon**，重点关注健康与福祉以及媒体技术领域。

**深入文档建设**：鼓励 Axolotl 社区为不断完善的 [Axolotl 文档](https://axolotl.continuumlabs.pro)做出贡献，并分享了关于动态偏好优化 **(DPO)** 可能比有监督微调 **(SFT)** 更有效地引导生成响应的见解。



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**Mixtral 加入路由机群**：[Mixtral 8x22B 已上线 OpenRouter](https://openrouter.ai/models/mistralai/mixtral-8x22b)，在指令模板方面表现强劲，目前提供免费试用。

**Gemma 新变体与价格调整**：OpenRouter 已将 [Gemma 7B 替换为升级后的 Gemma 1.1-7B](https://huggingface.co/google/gemma-1.1-7b-it)，并调整了多个模型的价格——包括 LZLV 70B 和 Databricks DBRX 132B——同时指出 Gemini 1.5 目前缺乏免费层级。

**反馈促成快速修复与澄清**：用户反馈促使 OpenRouter 修正了模型上“Updated”标签的问题，并部署了针对速率限制问题的修复程序。平台还澄清了 Gemini 模型的 Token 是按单个字符计数的，这会影响“上下文”成本。

**深入探讨模型限制**：OpenRouter 上受到严格速率限制的模型被限制在每分钟 10 次请求左右，类似于其他地方的免费层级。

**社区对 Mixtral 与 GPT-4 的对比评价**：社区中 Mixtral 8x22b 与 GPT-4 的对比显示，用户更青睐 Mixtral 的推理能力和成本效率，尽管 GPT-4 被认为更具文采。



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Command r+ 广受好评**：**Command r+** 因在基准测试和角色扮演场景中超越 **GPT-3.5** 和 **Claude 3** 而备受关注，暗示其性能接近 **GPT-4**。

- **Open-Interpreter 解决安装问题**：围绕 **open-interpreter 安装**出现了技术挑战，解决方案涉及使用 **git+https** 命令和正确的 **OPENAI_API_KEY** 设置。

- **Mixtral 与 OI 结合的乐观预期**：关于 **Mixtral 8x22b** 与 **Open-Interpreter (OI)** 潜在协作的讨论非常热烈，爱好者们希望其性能能超越 8x7b 版本。

- **OpenAI 转向预付费模式**：一条关于 **OpenAI** 转型为预付费模式的消息在流传，并附带促销信用额度优惠，有效期至 2024 年 4 月 24 日。

- **Open-Interpreter 代码库学习机会**：社区分享了包括一个带有 Python 模板的 GitHub 仓库在内的资源，用于将 **Open Interpreter 作为库**使用，促进了社区内进一步的教育交流。

- **ngrok 绑定困扰与设置方案分享**：技术讨论确定了一个 `ngrok` 无法绑定到用户指定域名的问题，这可能暗示配置错误；同时分享了一个用于设置 **01 Light** 设备的实用视频演示。

- **将机器学习模型引入浏览器**：介绍了 [Transformers.js](https://github.com/xenova/transformers.js)（**HuggingFace Transformers 库**的 JavaScript 实现），它能够实现在浏览器中运行复杂的 ML 模型，而无需服务器依赖。



---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **提示词管理激发参与度**：对 [vellum.ai](https://vellum.ai) 的兴趣日益增长，工程师们正在讨论其在更高效地构建、测试和部署适用于各种 AI 模型的 Prompt 方面的实用性。对于文本转语音（text-to-speech）功能的不同声音偏好仍未解决，引发了关于 ChatGPT 或 Mistral 等模型是否应该拥有指定“真实”声音的辩论。

- **AI 推理能力面临考验**：*Claude 3 Opus* 在 AI 模型的推理能力中脱颖而出，成为首选，但对于任何 AI 是否具备真正的推理能力，怀疑态度依然存在。此外，有人对验证 AI 服务引用来源准确性的必要性提出了担忧。

- **技术故障与计费异常备受关注**：用户抱怨 OpenAI 平台的高延迟和服务中断，包括在账户余额充足的情况下出现的计费故障。服务的不一致性引发了关于 GPT-4 在持续对话中出现“健忘”的警报，暗示可能存在停机；OpenAI 的 [状态页面](https://status.openai.com/) 对最近发生的事件提供了一些说明。

- **尖端 GPT-4 Turbo 引发创意讨论**：**GPT-4-turbo-2024-04-09** 增强的“创意且生动”的表现成为热点话题，用户建议不要使用代码块以防止不必要的代码压缩。建议使用 Prompt 链（Prompt chains）以获得更准确的代码输出，可以在 [promptingguide.ai](http://promptingguide.ai) 探索 Prompt Engineering 的资源。

- **针对 API 问题和提示词优化的跨频道智慧**：集体智慧源于对 API 问题的处理，包括关于在 OpenAI 的 Assistant API 中处理方法调用的建议，以及避免代码模块大小和优化问题的策略性 Prompt 链。为了将 Wolfram 集成到 GPT 中，用户可以将注意力转向通过其指定路线访问的 **Wolfram GPT 模型**。



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **IFTTT 执行停止增强了 Agent 控制**：与 [IFTTT](https://ifttt.com/) 的集成允许对流程进行条件控制，例如在预订确认后停止旅游 Agent 的执行；预告详情已通过 [Twitter](https://t.co/ByGOaqgWMd) 分享。
- **ColBERT 变得更简单了**：一种更直接的新技术正在开发中，用于构建具有对话记忆的基于 ColBERT 的检索 Agent，以增强文档搜索，更多信息已在 [Twitter](https://t.co/wVDpML9juP) 上预告。
- **与你的代码对话**：来自 @weaviate_io 的 @helloiamleonie 正在推广一个教程，关于如何构建一个允许与 GitHub 代码仓库聊天的应用程序，该程序使用本地 Large Language Model (LLM) 和 Embedding 模型，已在 [Twitter](https://t.co/yZN3QZjgN4) 上预告。
- **Instructor 与 LlamaIndex 的碰撞**：尽管有人询问如何使用 [Instructor](https://python.useinstructor.com/) 配合 LlamaIndex 流式输出结构化 LLM 结果，但随后并没有出现显著的讨论或解决方案。
- **处理错误与调试 LLM 调用**：社区对话集中在排查本地运行 [sec-insights](https://github.com/run-llama/sec-insights) 应用的问题以及提高 Large Language Model (LLM) 的可观测性。后者包括希望看到发送给 LLM 的**确切 Prompt**，以及更新 LlamaIndex 软件包和创建自定义聊天循环的建议。



---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **tinygrad CI 中缺失的测试被曝光**：**tinygrad** 的持续集成（CI）缺乏对 `copy_from_fd` 的测试，这在一次 [GitHub Action 运行](https://github.com/tinygrad/tinygrad/actions/runs/8633930065/job/23668153464)中暴露了出来；目前已计划通过新增测试来进行修复。
  
- **tinygrad 因项目侧重 C 语言而拒绝 Rust 提案**：一个[被拒绝的 pull request](https://github.com/tinygrad/tinygrad/pull/4138)突显了 **tinygrad** 出于性能和可维护性考虑，坚持使用 C 而非 Rust 的立场，并建议 Rust 通过接口与 C 库进行交互。

- **在 tinygrad 中，性能优于语言偏好**：在自动生成 Rust 代码的提案受到批评后，**tinygrad** 社区强化了优化 C 代码性能而非扩展到其他编程语言的观点。

- **敦促 tinygrad 规范 mnist 数据集处理**：在 **tinygrad** 的 mnist 数据集使用中发现了不一致之处；参与者提出了三种解决方案，包括调整示例文件或使用单独的目录来获取数据集。
  
- **关于语言选择中的内存安全和政治立场的观点交锋**：Rust Foundation 的商标和许可政治引发了关于 **tinygrad** 语言选择的辩论，同时也涉及对内存安全记录的共同关注，以及让人联想到 Java 和 Oracle 争端的组织实践。



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **腾讯的 Ella SDXL 与核心创新者**：成员们以怀疑的眼光讨论了**腾讯的 Ella SDXL**，并考虑了由 'Draw Things' 或 **CivitAI** 进行潜在训练的可能性，引用了 [Reddit 讨论](https://reddit.com)中关于腾讯拒绝发布 SDXL 版本的内容。他们的对话范围涵盖了从预算策略到不同 AI 风险投资的战略优先级。

- **用 AI 放大音乐**：**Udio** 用于音乐创作的新应用引起了关注，因为它得到了 will.i.am 和 Common 等知名艺术家的支持，并有[宣传推文](https://x.com/udiomusic/status/1778045337833193720)佐证。社区探索了该应用的功能，包括用户参与度以及将真实乐器轨道集成到其用于音乐生成的 latent diffusion model 中的可能性。

- **加速 AI 硬件发展**：AI 硬件领域迎来了新成员 **Meta's Training and Inference Accelerator (MTIA)**，根据其[公告](https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/)，该加速器在可控的 90 W TDP 下拥有 354 TFLOPS 的性能。这一进展显然激起了关于 AI 加速硬件竞赛升级的讨论。

- **Huggingface 推出 Parler TTS**：TTS 领域的创新激增，**Huggingface 团队**展示了一款具有语音提示（voice prompting）功能的 TTS 引擎，类似于 Stability AI 的 TTS，预示了该技术的未来发展轨迹（[Parler TTS GitHub](https://github.com/huggingface/parler-tts)）。

- **Intel 的 Lunar Lake 飞跃式前进**：Intel 下一代 **Lunar Lake CPUs** ([来源](https://www.reddit.com/r/singularity/comments/1c0ue4f/intels_nextgen_lunar_lake_cpus_will_be_able_to/)) 引起了热议，得益于内置的 45 TOPS 神经网络处理单元（NPU），它能够本地运行微软的 Copilot AI。对供应链动态的关注凸显了 Intel 凭借其自有制造设施，相对于 TSMC 与 Nvidia 及 AMD 合作伙伴关系的优势。



---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**关注你的 Token**：工程师们分享了使用 **tiktoken** 监控 **OpenAIAssistant** 的 **token usage**，并结合定价进行成本估算的技巧，这对于那些需要精简 API 调用成本的人来说非常完美。

**元数据过滤器实战**：向量数据库利用元数据过滤器进行精确查询，例如查找具有负面休假政策的公司。一位成员分享了如何为元数据包含自定义 retrievers，以确保结果中包含更丰富的上下文。

**Beta 功能备受关注**：关于 **ChatOpenAI class** 中 `with_structured_output` 的疑问揭示了该功能虽然未被弃用，但仍处于 beta 阶段。相关的代码示例正在流传，而像 [Instructor for Python](https://python.useinstructor.com/) 这样的工具也被推荐用于结构化 LLM 输出。

**LangChain 的开源兼容性难题**：LangChain 的架构自豪地支持各种 LLM 供应商，但成员们正在寻求使用非 OpenAI LLM 的明确示例，这些示例可能可以在 [LangChain documentation](https://langchain.readthedocs.io/) 中找到。

**新 AI 工具星系涌现**：搭载 **GPT-4** 和 Vision AI 的 **GPT AI**、提供免费高级 AI API 的 **Galaxy AI**，以及用于直观构建应用的升级版 **Appstorm v1.6.0** 的出现，展示了一个工程师触手可及的不断扩张的 AI 工具宇宙。



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Mixtral 模型凭借 AGI 评估的胜利引人注目**：**Mixtral** 最新的 **[models](https://huggingface.co/v2ray/Mixtral-8x22B-v0.1)** 因在 PIQA 和 BoolQ 等基准测试中的卓越表现而受到关注。德语社区正在询问其模型评估的等效基准。

- **模型许可讨论升温**：最新模型已确认采用 Apache 2.0 许可，社区期待很快发布 instruct 版本，这引发了关于许可对使用和共享影响的讨论。

- **发现模型性能差异**：一位成员指出，由于 **ChatML Template** 中的一个**换行符**，`DiscoResearch/DiscoLM_German_7b_v1` 表现出显著的性能差异，这引发了关于 tokenizer 配置影响的讨论。

- **跨语言研究成果奠定基础**：参考研究如 "[Multilingual Pretraining and Instruction Tuning Improve Cross-Lingual Knowledge Alignment, But Only Shallowly](https://arxiv.org/html/2404.04659v1)"，社区正在整合关于多任务 finetuning 如何延续到非英语数据的见解。

- **稠密模型转换标志着一个里程碑**：关于将 **22B 参数 MoE model** 转换为稠密（dense）版本的消息（由 Vezora 在 Hugging Face 上发布为 [Mistral-22B-v0.1](https://huggingface.co/Vezora/Mistral-22B-v0.1)）引起了关于模型架构的讨论，并促使了关于 model merging 方法实用性的探讨。



---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **新模型引发好奇**：一款**新 AI 模型**发布，根据 Sophia Yang 的推文确认，该模型既不是 *Mistral* 也不是 *EleutherAI* 的衍生品，持有 Apache 2.0 许可证。社区还猜测 **Llama** 和 **Cohere** 等竞争对手可能促使了这次仓促发布，尽管初步的 [AGIEval 结果](https://fxtwitter.com/jphme/status/1778028110954295486)显示其性能优于其他 Open-source 模型。

- **基准测试、博客与基础模型**：社区对 Benchmark 可能误导开发者表示担忧，因此提议创建一个新博客，为每个主要的模型发布提供公正的 Human Evals。此外，Hugging Face 的一场讨论展示了 **BigXtra** 的 Base Model 在未经 Instruction-tuned 时表现不佳，引发了关于 Instruction Tuning 收益和数据集影响的辩论。

- **评估 Instruction Tuning 辩论**：关于在遵循 Pretraining -> IFT -> RLHF 流程时 **Instruction-tuning (IFT)** 是否可能冗余的讨论非常具有启发性，因为 RLHF 期间的人类偏好评分可能隐含地教会了模型 Instruction-following。然而，有人指出模型训练的各个阶段经常是融合的，暗示在整个训练过程中使用了组合数据集和目标函数。

- **机器学习道德受质疑**：内幕交易和学术利益冲突的指控给社区蒙上了阴影，话题涉及从劣质的 Fine-tuning 过程到 **ML 教授**与行业投资之间复杂的纠葛。Anton Burkov 的推文引发了这场紧张的对话，引发了人们对该领域伦理实践的关注。

- **采访悬念与招募沉思**：一段可能与 **John Schulman** 进行的采访被预告，激起了成员们的好奇和期待。此外，还出现了一个关于意外确认和新成员招募策略的轻松笔记，提到了将一位名叫 **Satya** 的人纳入麾下的努力。



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

**音频智能取得飞跃**：Gemini 增强了其 AI 能力，获得了回答视频内容中音频相关问题的能力，解决了此前 Gemini 只能描述视频视觉效果的空白。

**Google 的复制粘贴深受粘贴之苦**：工程师们呼吁改进 Google 在其 Playground 中粘贴文本时的文本格式化能力，因为目前它会改变原始格式。

**斯坦福进军知识策展**：**[Stanford Storm 项目](https://github.com/stanford-oval/storm)** 展示了 AI 在知识策展（Knowledge Curation）方面的重大飞跃，这是一个由 LLM 驱动的系统，可以研究主题并生成包含引用文献的详尽报告。

**MacOS 上的 Shell 命令对决**：一个导致 `llm cmd` 在 MacOS iTerm2 上挂起的奇特问题被发现是由于需要用户输入，该问题已通过 [GitHub 上提供的修复补丁](https://github.com/simonw/llm-cmd/pull/12) 解决，确保命令不再挂起并能正确响应输入。

**Homebrew 或 Pipx：LLM Shell 仍让用户困惑**：在不同的 Shell 上排查 `llm cmd` 问题时，一位用户发现问题不在于高度自定义的 Shell 本身，而在于命令所需的交互在日志中不可见。



---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

**通过 Gradio UI for Figma 弥合差距**：Mozilla 推出了 **Gradio UI for Figma**，以促进设计阶段的快速原型开发和实验；可以通过 [Figma 的 Mozilla 页面](https://www.figma.com/@futureatmozilla)访问。如需深入讨论，Mozilla 鼓励加入其 [Discord 讨论频道](https://discord.com/channels/1089876418936180786/1091372086477459557/1228056720132280461)的相关话题。

**GPU 限制引发关注**：工程师们通过使用 `-ngl 3` 将部分层卸载到 CPU 内存来解决 GPU 内存限制问题，尽管承认这会带来显著的性能损失，并提议在 **llamafile** 中开发一项功能，通过动态卸载层来管理 VRAM 不足的问题。

**内核对话可能导致崩溃**：处理张量（tensors）可能会导致内核恐慌（kernel panic），有证据显示一台 M2 MacBook 在将 `.safetensors` 转换为 `.gguf` 时，由于可能超出了其 16GB RAM 的容量而发生死机。

**语言模型内存管理课程**：讨论中引用了 **GitHub 上的 ollama 项目**，该项目详细介绍了处理大型语言模型的方法，可作为增强 llamafile 内存处理能力的潜在指南。访问 ollama 的 [GitHub 页面](https://github.com/ollama/ollama/blob/c5c451ca3bde83e75a2a98ed9fd4e63a56bb02a9/llm/server.go#L43>) 了解更多详情。

**使用 Quiet-STaR 提升文本预测**：社区对 **Quiet-STaR** 产生了浓厚兴趣，这是一种让语言模型在每个 token 处提供推理过程（rationales）以优化文本预测的技术；分享的资源包括 [研究论文](https://arxiv.org/abs/2403.09629) 和 [GitHub 仓库](https://github.com/ezelikman/quiet-star)，以及一个相关的 Hugging Face 仓库。

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **Mistral 的重大里程碑**：根据初步的 [AGIEval 结果](https://x.com/jphme/status/1778030213881909451)，**Mistral 8x22b** 在 AGIEval 中树立了新标准，表现显著优于其他开源基础模型。

- **在 AI 中寻求逻辑**：AI 工程师分享了各种资源，包括 [awesome-reasoning GitHub 仓库](https://github.com/neurallambda/awesome-reasoning) 和 [Logic-LLM GitHub 仓库](https://github.com/teacherpeterpan/Logic-LLM)，提供了为大型语言模型（LLM）赋予逻辑推理能力的数据集和方法。

- **形式化证明 AI 辅助**：提到一个旨在训练大型语言模型进行形式化定理证明的 Coq 数据集 [arXiv 上的 Coq 数据集](https://arxiv.org/abs/2403.12627)，引发了对增强 AI 系统形式化验证能力的兴趣。

- **Google CodeGemma 问世**：Google 推出了 **CodeGemma**，这是一个基于 7B 参数的代码补全模型，其功能在一段 [YouTube 视频](https://www.youtube.com/watch?v=Gb--4supXoo) 中得到了展示。

- **热狗分类走红**：展示了一个使用 **Ollama, Mistral, 和 LLava** 等 AI 模型将图像分类为“热狗”或“非热狗”的教程，在 [YouTube 教程](https://www.youtube.com/watch?v=76BkMULO7uw) 中体现了 AI 幽默而实用的应用。

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

**GPT 的编程能力依然强劲**：用户体验打破了“续作乏力”的说法——**GPT** 在通过 *cursor* 使用时保持了其强大的编程能力，提供了快速的性能和全面的代码输出。

**Cursor vs. Claude：工具使用讨论**：虽然 **cursor** 因其由 **GPT-4** 驱动的高效 command-K 功能而在生成样板代码方面受到青睐，但一些用户在聊天交互中仍倾向于使用 **Claude opus**，尽管有报告称 Claude 首次出现了代码幻觉事件。

**Gemini 1.5 崛起**：公会成员对 **Gemini 1.5** 议论纷纷，对其编程能力给予了正面评价，尽管未深入探讨细节。

**Copilot++ 脱颖而出**：**Copilot++** 的推出受到了好评，因其在编程任务中的顶级表现而在众多先进工具中脱颖而出。

**Claude 罕见的失误**：用户首次报告了一个意外案例，**Claude** 凭空捏造了一段代码，偏离了以往在 **GPT-4** 中观察到的惯常准确表现。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Jamba 代码搜寻**：成员们对寻找 **Jamba** 的源代码表现出兴趣，并明确询问其所在位置。
- **对 Jamba 更新的好奇**：社区对 **Jamba** 的任何近期进展或反馈表现出明显的兴趣，成员们以热切的语气询问更新情况。

---

# 第 2 部分：各频道详细摘要与链接

**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1227515339630448641)** (691 条消息🔥🔥🔥): 

- **寻求 SD3 更新**：成员们频繁询问 **SD3** 的发布日期。预计可能在 1 到 3 周内发布，Beta 测试人员已经活跃了近一个月。人们对基础模型以及发布后的 finetuning 都寄予厚望。

- **Forge vs. ComfyUI**：讨论集中在 **Forge**、**ComfyUI** 和 **Automatic1111** 的优缺点上。偏好各不相同；**Forge** 被认为速度更快、模型处理效率更高，**ComfyUI** 适合深度控制，而 **Automatic1111** 在 image-to-image 和 ControlNet 表现上更好。

- **实现 LaVi Bridge**：有人提到有兴趣将 **LaVi Bridge** 技术集成到 **ComfyUI** 中。**LaVi Bridge** 与 **ELLA** 类似，但目前没有迹象表明它会在短期内实现。

- **VRAM 在 AI 训练中的重要性**：对话涉及了 VRAM 在 AI 模型训练中的重要性。VRAM 被认为对 scaling 至关重要；VRAM 使用量的减少可能会被 AI 能力的提升或扩展所抵消，因为后者会利用额外的 RAM。

- **在 ControlNet Inpaint 中使用 "Fill" 的潜力**：一位用户询问了在 **ControlNet Inpaint** 中使用 "Fill" 来处理已移除背景图像的能力。这表明用户对现有 UI 中高级 Inpainting 技术的需求。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/dataplusengine/status/1778109605186245002?s=46&t=QtCFBKTwAArvOc">来自 DataVoid e/acc (@DataPlusEngine) 的推文</a>：我们对 1.5 的 ELLA 训练进行了逆向工程，并成功制作了微调版本。我们正在努力调整脚本以适配 SDXL。对他们没有发布它感到非常失望。所以...</li><li><a href="https://huggingface.co/spaces/CohereForAI/c4ai-command-r-plus">C4AI Command R Plus - CohereForAI 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://supagruen.github.io/Stable Diffusion-CheatSheet/">Stable Diffusion 1.5 - 速查表</a>：未找到描述</li><li><a href="https://www.youtube.com/@AIchemywithXerophayze-jt1gg">AIchemy with Xerophayze</a>：看看 XeroGen，我们为多个 AI 图像生成平台打造的全新终极提示词锻造工具。旨在更好地适应工作流，并让你对提示词创建拥有终极控制权 https://shop.xerophayze.c...</li><li><a href="https://www.livescience.com/technology/artificial-intelligence/mit-has-just-worked-out-how-to-make-the-most-popular-ai-image-generators-dall-e-3-stable-diffusion-30-times-faster">MIT 科学家刚刚找到了让最流行的 AI 图像生成器提速 30 倍的方法</a>：科学家们构建了一个框架，通过将 DALL·E 3 和 Stable Diffusion 等生成式 AI 系统压缩成更小的模型，在不牺牲质量的前提下大幅提升其性能...</li><li><a href="https://stability.ai/stable-video">Stable Video — Stability AI</a>：Stability AI 首款基于图像模型 Stable Diffusion 的开源生成式 AI 视频模型。</li><li><a href="https://www.youtube.com/@latentvision/videos">Latent Vision</a>：未找到描述</li><li><a href="https://tenor.com/view/what-a-time-to-be-alive-simpsons-gif-14708682">What A Time To Be Alive Simpsons GIF - What A Time To Be Alive Simpsons - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/its-good-to-be-back-returned-its-like-i-never-left-good-to-be-back-im-back-gif-15987629">Its Good To Be Back Returned GIF - Its Good To Be Back Returned Its Like I Never Left - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=qcpfrpMbCA8">教程 | 1 分钟指南，永久解决 SD-WebUI &amp; Forge &amp; ComfyUI 所有模型路径问题。</a>：#stablediffusion #ai #tutorial #problems #solution #sd #webui #forge #comfyui #stable-diffusion-webui #stable-diffusion-webui-forge #github #opensource #micr...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/xelydv/stablediffusioninfinity_outpainting_with/">Reddit - 深入探索任何事物</a>：未找到描述</li><li><a href="https://x.com/dataplusengine/status/1778109605186245002?s=46&t=QtCFBKTwAArvOcSJDD650A">来自 DataVoid e/acc (@DataPlusEngine) 的推文</a>：我们对 1.5 的 ELLA 训练进行了逆向工程，并成功制作了微调版本。我们正在努力调整脚本以适配 SDXL。对他们没有发布它感到非常失望。所以...</li><li><a href="https://www.youtube.com/watch?v=q5MgWzZdq9s">Stable Diffusion Forge UI：底层探索 - 技巧与窍门 #stablediffusion</a>：在这段视频中，我们将详细了解 Stable Diffusion Forge UI，涵盖从查找和更新模型、设置到增强功能的方方面面...</li><li><a href="https://linktr.ee/artymusoke">artymusoke | Instagram | Linktree</a>：アティムソケ</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge">GitHub - lllyasviel/stable-diffusion-webui-forge</a>：通过在 GitHub 上创建账户，为 lllyasviel/stable-diffusion-webui-forge 的开发做出贡献。</li><li><a href="https://github.com/tencent-ailab/IP-Adapter">GitHub - tencent-ailab/IP-Adapter：图像提示词适配器，旨在让预训练的文本生成图像扩散模型能够通过图像提示词生成图像。</a>：图像提示词适配器旨在让预训练的文本生成图像扩散模型能够通过图像提示词生成图像。 - GitHub - tencent-ailab/IP-Adapter...
</li>
</ul>

</div>
  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1227524243215093814)** (276 条消息🔥🔥): 

- **关于模型性能和优化的讨论**：用户参与了关于微调性能的讨论，特别是涉及 **Mistral** 和 **CodeGemma** 等模型。一位用户表示在微调 CodeGemma 时遇到了高 VRAM 消耗的问题，即使应用了旨在减少内存使用的 checkpointing 更新。有讨论提到 **Gemma 在 VRAM 方面的需求比 Mistral 更高**，并建议尝试使用 `optimizer = paged_adamw_8bit`。

- **对 Apple Silicon 支持的关注**：社区对 **Apple Silicon 支持** 表现出极大的热情，一位成员自愿通过提供具有接近原生 GPU 性能的 VM 的 SSH 访问权限，来帮助将项目移植到 Apple Silicon。关于此支持的一个 GitHub issue 引起了关注。[Apple Silicon Support #4](https://github.com/unslothai/unsloth/issues/4)。

- **关于学习 Triton DSL 和平台使用的查询**：出现了关于学习 **Triton DSL** 的问题，成员们分享了 [Triton 教程](https://triton-lang.org/main/getting-started/tutorials/index.html) 的链接。还有提到一些用户在微调 **Gemma** 等模型时遇到 OOM (显存溢出) 问题，并有兴趣探索提高效率的替代方案。

- **关于 Unsloth 微调的反馈和经验**：用户分享了他们使用 Unsloth 微调模型的经验，讨论了 VRAM 需求和生成文本中的 **重复错误** 等问题。此外，**Andrei Karpathy** 点赞了关于 Unsloth 发布的一条推文，这引起了大家的兴奋。

- **关于 Perplexity Labs 和其他 LLM 的讨论**：对话涉及了 **Perplexity Labs** 及其 **指令微调 (instruction tuning)**，一位用户注意到搜索结果与模型输出之间存在相似性。讨论还提到了对 **inflection** 有效性的担忧，以及对 **Mixtral-8x22b** 等新模型及其性能的关注。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://triton-lang.org/main/getting-started/tutorials/index.html">Tutorials &mdash; Triton  documentation</a>：未找到描述</li><li><a href="https://huggingface.co/docs/datasets/v1.1.3/loading_datasets.html">Loading a Dataset &mdash; datasets 1.1.3 documentation</a>：未找到描述</li><li><a href="https://tenor.com/view/pisswasser-gta-gif-15583288">Pisswasser Gta GIF - Pisswasser Gta - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=gyKBN1rnefI&list=PLSXcJOyFhmS-qb_CF-GLhkWxSmi-ftbPO&index=2)">Intro to Triton: Coding Softmax in PyTorch</a>：让我们在 PyTorch eager 中编写 Softmax，并确保我们有一个工作版本来与我们的 Triton Softmax 版本进行比较。下一个视频 - 我们将在 Tr...</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth)</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2310.20111">Making Large Language Models Better Data Creators</a>：虽然大型语言模型 (LLMs) 显著推进了 NLP 的最前沿技术，但由于成本、响应速度、控制或...等原因，将它们部署到下游应用仍然具有挑战性。</li><li><a href="https://github.com/bennyschmidt/next-token-prediction">GitHub - bennyschmidt/next-token-prediction: Next-token prediction in JavaScript — build fast LLMs from scratch!</a>：JavaScript 中的 Next-token 预测 — 从零开始构建快速 LLMs！- bennyschmidt/next-token-prediction</li><li><a href="https://github.com/GraphPKU/PiSSA">GitHub - GraphPKU/PiSSA</a>：通过在 GitHub 上创建一个账户来为 GraphPKU/PiSSA 的开发做出贡献。</li><li><a href="https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1/discussions/4">mistral-community/Mixtral-8x22B-v0.1 · Benchmarks are here!</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/4">Apple Silicon Support · Issue #4 · unslothai/unsloth</a>：很棒的项目。希望能看到 Apple Silicon 支持！</li><li><a href="https://github.com/searxng/searxng">GitHub - searxng/searxng: SearXNG is a free internet metasearch engine which aggregates results from various search services and databases. Users are neither tracked nor profiled.</a>：SearXNG 是一个免费的互联网元搜索引擎，它聚合了来自各种搜索服务和数据库的结果。用户既不会被追踪也不会被画像。- searxng/searxng</li><li><a href="https://github.com/huggingface/peft/pull/1626">Adding PiSSA as an optional initialization method of LoRA by fxmeng · Pull Request #1626 · huggingface/peft</a>：在论文 "https://arxiv.org/pdf/2404.02948.pdf" 中，我们介绍了一种参数高效微调 (PEFT) 方法，主奇异值和奇异向量自适应 (PiSSA)，它优化了...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1227583498693382235)** (7 条消息): 

- **对 Unsloth 周边的热情**：进行了一次轻松的交流，建议制作 **Unsloth 周边**。成员们反应积极，开玩笑地用表情符号和俏皮的话语讨论这一前景。

- **对 Hugging Face 文档的兴趣**：一位成员询问了 **Hugging Face 的 JSON 文件文档** 链接，表明需要与该平台相关的参考资源。

- **团队招募设计师**：团队意识到增加更多设计师将大有裨益，这预示着未来可能在社区内进行招聘或合作。
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1227517151892607066)** (244 条消息🔥🔥): 

- **关于 Unsloth 多 GPU 支持的说明**：用户询问了 Unsloth 的多 GPU 支持情况，提到了 pre-alpha 版本，并讨论了旨在防止大型科技公司滥用的许可限制。该支持目前正在开发中，未来的版本可能会限制在不联系 Unsloth 获取额外权限的情况下最多使用四个 GPU。
  
- **自定义数据集微调挑战**：一位用户在对 GEMMA 模型进行自定义对话数据集（非公开状态）微调时遇到困难。建议使用 Pandas 重新格式化数据，并参考 [pandas documentation](https://github.com/pandas-dev/pandas) 获取进一步帮助。用户在遵循该建议后成功解决了问题。

- **对话数据集格式问题与解答**：关于如何为对话模型格式化数据集进行了详细讨论，涉及识别群聊中的回复对以及可以使用的各种格式（如 'user' 和 'assistant' 或原始聊天记录）。讨论了一种无需微调即可创建多角色聊天的方法，该方法使用 router 来分析对话并确定下一位发言者。

- **开源 Kernel 的性能对比**：一位用户讨论了为研究论文对开源 Kernel 进行全面对比的情况，称赞了 Unsloth 的易集成性，并表达了对扩展到其他 Kernel（如 fully fused MLPs 和 relu2）的兴趣。讨论强调了在研究中合规使用多 GPU 支持以及致谢开源贡献的重要性。

- **使用 Unsloth AI 训练后的部署问题**：用户询问了使用 Unsloth AI 训练后的模型部署问题，参考了 [Unsloth documentation](https://github.com/unslothai/unsloth/wiki) 中关于保存模型和设置部署的指南。随后的对话澄清了 Unsloth 虽然针对其自身实现进行了优化，但也可以适配其他用例。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/datasets/en/loading#json">Load</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing#scrollTo=p31Z-S6FUieB)">Google Colaboratory</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/datasets/en/loading#json?">Load</a>: 未找到描述</li><li><a href="https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT.ipynb">Transformers-Tutorials/BERT/Custom_Named_Entity_Recognition_with_BERT.ipynb at master · NielsRogge/Transformers-Tutorials</a>: 此仓库包含我使用 HuggingFace 的 Transformers 库制作的演示。 - NielsRogge/Transformers-Tutorials</li><li><a href="https://github.com/huggingface/transformers/issues/30141">push_to_hub doesn&#39;t push checkpoint folder while training · Issue #30141 · huggingface/transformers</a>: 系统信息 我正在使用带有 Unsloth Mistral notebook 的 Google Colab。复现 我正在使用来自 trl 的代码片段 from trl import SFTTrainer from transformers import TrainingArguments trainer = SFTTrainer( m...</li><li><a href="https://huggingface.co/d4data/biomedical-ner-all">d4data/biomedical-ner-all · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki">Home</a>: 速度提升 2-5 倍，显存占用减少 80%，支持 QLoRA &amp; LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/pandas-dev/pandas">GitHub - pandas-dev/pandas: Flexible and powerful data analysis / manipulation library for Python, providing labeled data structures similar to R data.frame objects, statistical functions, and much more</a>: 为 Python 提供灵活且强大的数据分析/操作库，提供类似于 R data.frame 对象的标记数据结构、统计函数等 - pandas-dev/pandas</li><li><a href="https://huggingface.co/datasets/Roblox/luau_corpus/">Roblox/luau_corpus · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/19lwcRk_ZQ_ZtX-qzFP3qZBBHZNcMD1hh?usp=sharing#scrollTo=LjY75GoYUCB8">Google Colaboratory</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/mahiatlinux/luau_corpus-ShareGPT-for-EDM">mahiatlinux/luau_corpus-ShareGPT-for-EDM · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/Green0-0/Discord-LLM-v2">GitHub - Green0-0/Discord-LLM-v2</a>: 通过在 GitHub 上创建账户来为 Green0-0/Discord-LLM-v2 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth/wiki#chat-templates">Home</a>: 速度提升 2-5 倍，显存占用减少 80%，支持 QLoRA &amp; LoRA 微调 - unslothai/unsloth</li><li><a href="https://huggingface.co/datasets/philschmid/guanaco-sharegpt-style">philschmid/guanaco-sharegpt-style · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/transformers/en/model_doc/distilbert">DistilBERT</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1227622002592976968)** (15 条消息🔥): 

- **介绍 StegLLM**: Ashthescholar 分享了 StegLLM，这是一个带有后门机制的 LLM，使用 Unsloth 在基于 unsloth/mistral-7b-instruct-v0.2-bnb-4bit 的模型上进行了微调。提供了一个 [safetensor 模型链接](https://huggingface.co/AshScholar/StegLLM)，并对该方法表示了认可，其灵感来自 Anthropic 对 Sleeper Agents 的研究。

- **StegBot 预览**: 在最初认为模型文件无法访问后，ashthescholar 后来发现并分享了在 Hugging Face 上的 StegBot（使用 Unsloth 训练）链接：[StegBot 模型](https://huggingface.co/oofnan/stegBot/tree/main)。

- **Ghost 7B 一瞥**: Lh0x00 预览了即将推出的 Ghost 7B，这是一款多语言大模型，因其推理能力和先进的越南语理解能力而受到赞誉。该模型是 [Ghost X](https://huggingface.co/ghost-x) 发起的专注于下一代优化、知识渊博且多语言大语言模型倡议的一部分。

社区对分享的项目表现出极大的热情和赞赏，强调了它们在 AI 领域做出的创新贡献。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/oofnan/stegBot/tree/main">oofnan/stegBot at main</a>: 未找到描述</li><li><a href="https://huggingface.co/ghost-x">ghost-x (Ghost X)</a>: 未找到描述</li><li><a href="https://huggingface.co/AshScholar/StegLLM">AshScholar/StegLLM · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1227600572606316577)** (6 条消息):

- **Multi-GPU 支持即将到来**：一位成员提到他们正在开发 **multi-GPU** 支持，这可能会根据 *llama-3* 的发布情况推出。
- **Sophia 的测试展现出潜力**：一位成员强调了 **Armen Agha (FAIR)** 的 Twitter 帖子，详细介绍了 Sophia 的改进，并附上了推文链接：[Sophia 测试结果](https://twitter.com/ArmenAgha/status/1777850260829962489)。特别指出的改进是 Sophia 的 Triton 实现几乎与 AdamW 一样高效。
- **AdaLomo 应对 LLaMA**：一位成员分享了关于 **AdaLomo** 的见解，该模型在 LLaMA 1 模型上进行了测试，并提供了一种类似于 LoRA 的具有内存效率的自适应学习率。关于 AdaLomo 的学术论文可以在这里找到：[AdaLomo PDF](https://arxiv.org/abs/2310.10195)。
- **对 Multi-GPU 开发的热情**：关于 multi-GPU 开发工作的公告获得了积极反应，成员们用简单的 "lets goo" 表达了兴奋之情。

**提到的链接**：<a href="https://arxiv.org/abs/2310.10195">AdaLomo: Low-memory Optimization with Adaptive Learning Rate</a>：大型语言模型取得了显著成功，但其庞大的参数规模需要大量的内存进行训练，从而设定了很高的门槛。虽然最近提出的 l...

---

**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1227543376396681226)** (183 条消息🔥🔥): 

- **LM Studio 讨论与性能问题**：用户讨论了 GPT "GPT Builder" 不足的问题，强调其 system prompts 过于简短，并补充说手动编写 prompts 更好。在另一个对话中，讨论了 CodeGemma 等各种 LM Studio 模型的尺寸和加载问题，并建议可能需要使用更小的 quants 才能在 32GB RAM 的笔记本电脑等性能较弱的硬件上运行。

- **新模型与 LM Studio 更新**：聊天中提到的新发布内容包括 Mistral 的新 8x22b 模型，由于 GGUF quantizations 尚未完成，该模型不会立即在 LM Studio 中运行；以及 GPT-4-turbo-2024-04-09，具有 128k tokens、更新的 turbo 以及视觉能力。还有关于 IQ2_XXS 等各种 quants 如何允许在 RTX 4090 等单个 GPUs 上运行更大模型的讨论。

- **关于模型能力与可用性的提问**：用户询问了支持 Python 编程的最佳模型，建议包括 GPT-4、Claude3 Opus 和 Phind，并承认后者包含访问互联网的功能。此外，还有关于针对 anti-NSFW 内容的模型，以及在 LM Studio 优化中 VRAM 与 system RAM 利用率的对话。

- **模型部署与访问**：一位用户找到了通过 ngrok 从 GitPod 等在线开发工具访问其 LMStudio 服务器的解决方案，解决了之前的困惑。其他用户互相交流了生日祝福，并讨论了日期相关背景下的巧合。

- **技术问题与解决方案**：一些用户面临模型加载失败的挑战，引发了关于运行更大模型的需求以及增强系统的方法（如为笔记本电脑使用 eGPUs）的讨论。其他用户报告了 JavaScript 错误以及 LM Studio 在缺乏 AVX2 指令支持的系统上的兼容性问题，其中一个案例指出防病毒软件的误报已被澄清和忽略。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/gandalf-gif-21901728">Gandalf GIF - Gandalf - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B#prompt-format-for-function-calling">NousResearch/Hermes-2-Pro-Mistral-7B · Hugging Face</a>: 未找到描述</li><li><a href="https://lmstudio.ai/docs/text-embeddings">Text Embeddings | LM Studio</a>: Text Embeddings 处于 Beta 阶段。从此处下载支持该功能的 LM Studio。</li><li><a href="https://lmstudio.ai/beta-releases.html)">👾 LM Studio - 发现并运行本地 LLMs</a>: 查找、下载并实验本地 LLM</li><li><a href="https://github.com/Pythagora-io/gpt-pilot/wiki/Using-GPT%E2%80%90Pilot-with-Local-LLMs">在本地 LLMs 中使用 GPT-Pilot</a>: 第一个真正的 AI 开发者。通过在 GitHub 上创建账户，为 Pythagora-io/gpt-pilot 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=DiSKfiJ7I-s">在 Windows 本地安装 CodeGemma - 优秀的轻量级编程 LLM</a>: 此视频展示了如何在 Windows 上本地安装新的 Google CodeGemma AI 模型。它是最好的轻量级编程模型之一。▶ 成为赞助人 🔥 - https://...</li><li><a href="https://www.nvidia.com/en-gb/design-visualization/rtx-a6000/">基于 Ampere 架构的 NVIDIA RTX A6000 | NVIDIA</a>: 开启下一代革命性设计和沉浸式娱乐体验</li><li><a href="https://the-decoder.com/mixtral-8x22b-ai-startup-mistral-releases-new-open-language-model/">Mixtral 8x22B：AI 初创公司 Mistral 发布新的开源语言模型</a>: 总部位于巴黎的 AI 初创公司 Mistral 通过种子链接发布了 Mixtral-8x22B MoE，这是一款新的开源语言模型。
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1227518958861815839)** (197 条消息🔥🔥): 

- **Mixtral vs. Command R+**: [Mixtral 8x22B 模型](https://huggingface.co/MaziyarPanahi/Mixtral-8x22B-v0.1-GGUF) 已发布，并就其能力以及与 **Command R+** 的对比进行了讨论。共识是，虽然 Mixtral 8x22B 是一个大型基础模型，尚未针对 Chat 任务进行微调，但 Command R+ 看起来更先进，类似于 Chat GPT，且两者的量化版本在推理速度上存在差异。
  
- **高资源消耗模型挑战硬件**: 用户分享了不同模型的体验，指出 Command R+ 和 Mixtral 都非常耗费资源，在大型硬件配置上可能会出现内存溢出 (OOM)。提到最新的 **Mac Studio (192 GB RAM)** 能够在某些量化级别下运行这些模型。

- **LLM Studio Beta 更新**: 一系列消息建议使用 [LM Studio 最新的 Beta 版本](https://lmstudio.ai/beta-releases.html) 来支持更新、更大的模型，特别是为了自动处理分割的 GGUF 文件，而无需手动合并。

- **量化与分割模型**: 一位量化大型模型的用户表示，他们将恢复其偏好的文件分割方法（将它们放在子文件夹中），前提是未来的 LM Studio 版本提供官方支持。

- **大型模型的实用变通方案**: 对于高资源模型，用户建议禁用 "keep model in RAM"（将模型保留在内存中）并考虑 GPU offload 设置，以便在性能较低的设备上运行模型，接受较低的每秒 Token 生成速度以换取高质量输出。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/bartowski/Mixtral-8x22B-v0.1-GGUF">bartowski/Mixtral-8x22B-v0.1-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/MaziyarPanahi/Mixtral-8x22B-v0.1-GGUF">MaziyarPanahi/Mixtral-8x22B-v0.1-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/Chat2DB/Chat2DB-SQL-7B">Chat2DB/Chat2DB-SQL-7B · Hugging Face</a>: 未找到描述</li><li><a href="https://lmstudio.ai/beta-releases.html">LM Studio Beta Releases</a>: 未找到描述</li><li><a href="https://techcrunch.com/2024/04/09/meta-confirms-that-its-llama-3-open-source-llm-is-coming-in-the-next-month/amp/?guccounter=1">Meta 确认其 Llama 3 开源 LLM 将在下个月发布 | TechCrunch</a>: Meta 的 Llama 系列作为开源产品构建，代表了 AI 作为一项广泛技术应如何发展的不同哲学方法。</li><li><a href="https://huggingface.co/lmstudio-community">lmstudio-community (LM Studio Community)</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/)** (1 条消息): 

sanjuhs123: 这太棒了，那我就只需要下载 Beta 0.2.19 或者等到它正式发布。
  

---


**LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1227906718160126034)** (2 条消息): 

_

- **LLM 处理时间序列数据的挑战**：有观点认为，除非改变模型设计，否则时间序列数据不适合大语言模型 (LLM)。
- **针对时间序列数据的 TFT 训练**：提到可以在时间序列数据上训练 Temporal Fusion Transformer (TFT)。
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1227584807316492338)** (40 条消息🔥): 

- **应对云端成本**：讨论集中在 [云服务成本](https://www.forrester.com/blogs/aws-joins-google-cloud-in-removing-egress-costs/) 上，强调了了解云端成本的重要性，特别是围绕数据传出 (egress) 的费用。提到了 AWS 最近宣布取消某些迁移场景的传出费用。

- **优化模型部署**：成员们就本地部署 (on-premises) 与云端 AI 模型部署交换了意见，认为对于学习和实验 AI 模型来说，高端消费级硬件可能比云端 IaaS 更具成本效益，除非需要大规模、短期的可扩展性。

- **模型训练中的 GPU 限制**：一位参与者询问是否能在单块 RTX 4090 上运行大型 `C4AI Command R+` 模型，随后引发了关于 4090 能流畅运行的模型最大参数量，以及在本地设置中有效利用多块 GPU 的限制的讨论。

- **云端 GPU 与本地硬件限制**：目前 LM Studio 尚不支持云端 GPU 服务；一位成员在考虑是增加笔记本电脑 RAM 以提高语言模型性能，还是维持双通道配置以保持游戏优势。

- **LM Studio 硬件利用率**：简要交流了 LM Studio 如何利用 GPU，观察到系统中并非所有 GPU 都会默认被调用，并建议使用 `tensor.split` 来管理 GPU 之间的卸载比例。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.forrester.com/blogs/aws-joins-google-cloud-in-removing-egress-costs/">AWS 紧随 Google Cloud 取消传出费用</a>：Amazon Web Services 计划取消传出费用。了解这对技术专业人士意味着什么，以及你应该采取哪两个步骤。</li><li><a href="https://huggingface.co/dranger003/c4ai-command-r-plus-iMat.GGUF/blob/main/ggml-c4ai-command-r-plus-104b-iq2_xxs.gguf">ggml-c4ai-command-r-plus-104b-iq2_xxs.gguf · dranger003/c4ai-command-r-plus-iMat.GGUF</a>：未找到描述</li><li><a href="https://huggingface.co/chat">HuggingChat</a>：让每个人都能使用社区最好的 AI 聊天模型。
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1227649875617775837)** (52 条消息🔥): 

- **Command R Plus 在测试中胜出**：一位成员报告在 **0.2.19 beta preview 3** 上成功运行了 "pmysl/c4ai-command-r-plus-GGUF"，特别强调了该模型在低级编程语言和数学应用题方面的精通程度。Command R Plus 令人印象深刻地使用 LaTeX 格式化响应，超越了包括 dbrx 和 Mixtral 8x22b 在内的许多其他模型。

- **解决 AMD 机器模型可见性问题**：当模型在 LM Studio 界面上不可见时，对于运行版本 0.2.19 preview 3 且支持 AVX2 的 AMD 机器用户，折叠 “README” 组件后模型即可显示。

- **Codegemma 加载挑战**：多位用户在各种硬件配置上使用版本 2.19 Preview 3 尝试加载 "codegemma-7b-it-Q8_0" 时遇到持续崩溃。目前正通过用户反馈寻求解决方案，包括分享间歇性工作的配置和截图以供进一步分析。

- **与 Open WebUI 的集成问题**：一位用户在最新 beta 版中遇到了 Open WebUI 与 LM Studio 之间的连接问题。经过进一步检查，发现问题是由格式错误的 JSON 对象引起的，可以通过添加嵌入模型作为变通方法来规避。

- **LaTeX 渲染小问题**：一位用户在测试 Command R Plus 时遇到了输出中的 LaTeX markdown，这引发了澄清：与 ChatGPT 等其他平台不同，LM Studio 目前不支持内置的 LaTeX 渲染。
  

---


**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1227704380070039786)** (2 条消息): 

- **寻找适配 AutoGen 的 AI**：一位成员正在寻找适合在 3080 GPU 上托管的 **12GB AI 模型**，用于运行 AutoGen 进行编码和通用任务。该请求尚未收到具体的模型建议。
- **Dolphin 的故事**：在过去的尝试中，另一位成员成功使用了一个名为 **Dolphin 的 5GB 模型**来实现类似目的。然而，未提供有关性能或设置的详细信息。
  

---

**LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/1227860722508234813)** (1 条消息): 

- **LM Studio 与 n8n 的集成问题**：一位成员在尝试将 **LM Studio** 连接到 **n8n** 时遇到问题。他们尝试使用 OpenAI 模型选项并将 **URL 更改为他们的自托管模型**，但由于凭据中缺少 API key，导致收到 *200 错误*。
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1227581423406551061)** (31 条消息🔥): 

- **最近的 Beta 版本中 ROCm 问题依然存在**：成员们报告称，在最近的三个 Beta 版本中，**ROCm** 未能按预期运行，模型被加载到 RAM 而不是 VRAM 中，且 "gpu-preferences.json" 将 GPU 类型列为 "unknown"。用户在 2.17 版本中运行稳定，但从 2.18 版本开始遇到问题。

- **ROCm 可能不支持某些 GPU**：目前尚不确定某些 AMD GPU（如 **7800XT**）是否支持 ROCm，尽管已知 6800 等其他型号可以工作。引用了 [ROCm 支持文档](https://rocm.docs.amd.com/en/docs-5.5.1/release/windows_support.html) 进行澄清，成员们建议咨询特定的 AMD 资源。

- **Linux ROCm 技术预览版咨询**：有人询问 **amd-rocm-tech-preview** 是否有 Linux 版本，一位成员回答说虽然可能有计划，但预计不会很快推出。

- **ROCm 的 GPU 奇特现象**：用户指出，当 7900XTX 等高性能 GPU 在高负载下（例如在 LM Studio 中运行 ROCm）会发出电感啸叫（coil whine）等独特声音，这表明资源利用率可能很高或硬件压力较大。

- **模型加载失败详情**：在使用 Windows 系统的 ROCm 平台尝试加载 "Llama-2-7b-chat" 和 "Mistral instruct v0 1 7B" 等模型时，用户遇到了 "Error loading model." 且退出代码为 0 的错误，但在不同硬件或旧版本上取得了一些成功。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://rocm.docs.amd.com/en/docs-5.7.1/release/windows_support.html">GPU and OS Support (Windows) — ROCm 5.7.1 Documentation Home</a>：未找到描述</li><li><a href="https://rocm.docs.amd.com/en/docs-5.5.1/release/windows_support.html">GPU and OS Support (Windows) — ROCm 5.5.1 Documentation Home</a>：未找到描述</li><li><a href="https://rocm.docs.amd.com/en/docs-5.5.1">AMD ROCm™ Documentation — ROCm 5.5.1 Documentation Home</a>：未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1227564205708939354)** (3 条消息): 

- **DuckDuckGo 作为解决方案？**：一位成员提到在没有 API 的情况下使用 DuckDuckGo 进行搜索，暗示 *Crewai* 限制了某些功能。
- **对模型驱动搜索的好奇**：另一位成员对通过模型搜索互联网的概念表示热衷，并跟进了 DuckDuckGo 的话题。
  

---



**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1227622248752353331)** (8 条消息🔥): 

- **Google 的代码补全 AI，CodeGemma**：一位成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=Gb--4supXoo)，介绍了 **Google 的 CodeGemma**，这是一个代码补全模型，提供 7B 预训练变体，为开发者提供强大的编码能力。
- **AI 研究带来的失眠灵感**：一位成员对 AI 研究和模型提供的灵感表示感谢，甚至到了失眠的程度，这促使另一位成员将这份感谢转达给同事，尽管开玩笑地指出了对睡眠不足的担忧。
- **Technium 预告更多模型**：针对成员对启发性 AI 模型的赞赏，**teknium** 暗示 **“还有更多模型即将推出”**。
- **是热狗吗？AI 教程**：介绍了一个新的 [YouTube 视频](https://www.youtube.com/watch?v=76BkMULO7uw)，展示了如何使用 **Ollama**、**Mistral** 和 **LLava** 区分热狗与其他图像的教程。
- **深入的朝鲜访谈**：分享了一段 [英文翻译访谈](https://www.youtube.com/watch?v=C84bzu9wXC0)，专家在其中畅谈了 3 小时的朝鲜话题，邀请成员们探索该国的政治和社会动态。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=Gb--4supXoo">介绍 CodeGemma，Google 的代码补全模型</a>：CodeGemma 为社区带来了强大且轻量级的编程能力。CodeGemma 模型提供 7B 预训练变体，专注于...</li><li><a href="https://www.youtube.com/watch?v=76BkMULO7uw">使用 Ollama, Mistral 和 LLava 判断是否为热狗</a>：在本教程中，我们使用 Ollama, Mistral 和 LLava 来查看图像是否为热狗 #python #pythonprogramming #llm #ml #ai #aritificialintel...</li><li><a href="https://www.youtube.com/watch?v=C84bzu9wXC0">关于北朝鲜的尴尬问题</a>：ERID: LjN8Jv34w 广告。广告商 ООО "ФЛАУВАУ" INN: 9702020445。即使远在天边也能让亲人开心：https://flowwow.com/s/VDUD15 为节日选择礼物...</li><li><a href="https://tenor.com/view/money-rain-erlich-bachman-tj-miller-silicon-valley-unicorn-gif-11481689">硅谷 Erlich Bachman Tj Miller 撒钱 GIF - Money Rain Erlich Bachman Tj Miller - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1227555391500587038)** (8 messages🔥): 

- **起跑线上的双向注意力 (Bidirectional Attention)**：成员讨论了 AI 架构中双向信息流的潜在需求，引用了 SD3 因双向流和改进的训练描述词 (captions) 在文本渲染方面的成功。

- **推测 Mistral 模型中的双向注意力**：分享了一段直接引用，推测 **Mistral 模型** 可能使用了某种形式的双向注意力，如前缀语言建模 (prefix language modeling)，这是基于对各种输入和模型的复制结果得出的。

- **引入 Infini-attention 以扩展 Transformer**：社区关注了一篇论文 ([Infini-attention](https://arxiv.org/abs/2404.07143)) 中提出的名为 **Infini-attention** 的新方法，该方法允许基于 Transformer 的大语言模型 (LLMs) 更高效地处理无限长的输入。

- **RNN 的复兴？**：成员链接了一篇论文 ([RNN Comeback](https://arxiv.org/abs/2402.19427))，表明人们对 RNN 或混合模型的兴趣重新抬头，引发了关于创新尝试往往又回到基于 RNN 架构的循环讨论。

- **Google 发布基于新 RNN 架构的模型**：强调了 Google 最近发布了一个利用上述基于 RNN 架构的 7B 模型，引起了 AI 社区的关注。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.07143">Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention</a>：这项工作引入了一种高效的方法，将基于 Transformer 的大语言模型 (LLMs) 扩展到无限长的输入，且内存和计算量有限。我们提出的方法中的一个关键组件...</li><li><a href="https://arxiv.org/abs/2402.19427">Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models</a>：循环神经网络 (RNNs) 具有快速推理能力，并在长序列上高效扩展，但它们难以训练且难以扩展。我们提出了 Hawk，一种具有门控线性递归的 RNN...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1227543960956702801)** (278 messages🔥🔥): 

- **关于算力限制**：多次讨论强调了训练像 **Nous-Hermes-8x22b** 这样模型的巨大计算需求。成本被比作 80,000 美元的基础设施投入，且在租用硬件方面存在重大障碍，例如按需算力提供商缺乏 **Infiniband 互连** 选项。

- **微调对比**：小组讨论了各种微调模型的性能，指出 **Dolphin-2.2-70b** 使用了 **QLoRA** 且表现良好，尽管不如全参数微调 (**FFT**)。还提到了 **LISA** 等替代方案，它在每个 Batch 中随机解冻层，被认为是一种潜在的优越方法。

- **新技术和硬件选择**：讨论了减少 AI 模型巨大内存需求的潜在方法，多名成员指向了 **QLoRA** 和 **Unsloth** 的实现。鉴于服务器级 GPU 的高昂成本，人们也对拥有更多显存的消费级 GPU 充满期待。

- **雄心壮志下的障碍**：用户预计 **MistralAI** 最终会发布其巨型模型（如 **Mixtral-8x22b**）的 *Instruct* 版本，使其更易于管理。对话表明，虽然原始模型显示出潜力，但它们仍然是“野兽”，需要大量工作来“驯服”以用于特定应用。

- **实验与结果**：分享了关于使用不同模型和基准测试（benchmarks）的各种实验和观察结果。提到虽然 **Mixtral-8x22b** 看起来很有前景，但它在 **MT-Bench** 上的表现低于预期，这可能是由于样本量非常小以及训练成本高昂所致。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/jphme/status/1778030213881909451">来自 Jan P. Harries (@jphme) 的推文</a>：@MistralAI 首批 AGIEval 结果看起来很棒 👇 - 感谢发布这个猛兽，伙计们！👏 https://x.com/jphme/status/1778028110954295486 ↘️ 引用 Jan P. Harries (@jphme) 的话：首批 AGIEval 结果...</li><li><a href="https://foundershub.startups.microsoft.com/signup>">Microsoft for Startups FoundersHub</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2404.05892">Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence</a>：我们介绍了 Eagle (RWKV-5) 和 Finch (RWKV-6)，这是在 RWKV (RWKV-4) 架构基础上改进的序列模型。我们的架构设计进步包括多头矩阵值状态和动态...</li><li><a href="https://huggingface.co/lightblue/Karasu-Mixtral-8x22B-v0.1">lightblue/Karasu-Mixtral-8x22B-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/RWKV">RWKV (RWKV)</a>：未找到描述</li><li><a href="https://huggingface.co/v2ray/Mixtral-8x22B-v0.1">v2ray/Mixtral-8x22B-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://www.theregister.com/2024/04/09/intel_gaudi_ai_accelerator/">Intel Gaudi 的第三次也是最后一次欢呼，被定位为 H100 的竞争者</a>：再见专用 AI 硬件，你好融合了 Xe 图形基因与 Habana 化学反应的 GPU</li><li><a href="https://techcrunch.com/2024/04/09/meta-confirms-that-its-llama-3-open-source-llm-is-coming-in-the-next-month/amp/">Meta 确认其 Llama 3 开源 LLM 将在下个月推出 | TechCrunch</a>：Meta 的 Llama 家族作为开源产品构建，代表了 AI 作为一种更广泛技术应如何发展的不同哲学方法。</li><li><a href="https://en.wikipedia.org/wiki/List_of_logic_symbols">逻辑符号列表 - Wikipedia</a>：未找到描述</li><li><a href="https://openrouter.ai/models/mistralai/mixtral-8x22b">mistralai 的 Mixtral 8x22B | OpenRouter</a>：Mixtral 8x22B 是来自 Mistral AI 的大规模语言模型。它由 8 个专家组成，每个专家有 220 亿参数，每个 token 每次使用 2 个专家。它通过 [X](https://twitter...) 发布。</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docs/fsdp_qlora.qmd">OpenAccess-AI-Collective/axolotl 主分支下的 axolotl/docs/fsdp_qlora.qmd</a>：尽管提问（axolotl questions）。通过在 GitHub 上创建账户为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://huggingface.co/docs/transformers/v4.18.0/en/performance">性能与可扩展性：如何容纳更大的模型并更快地训练它</a>：未找到描述</li><li><a href="https://github.com/google/gemma.cpp">GitHub - google/gemma.cpp：适用于 Google Gemma 模型的轻量级、独立 C++ 推理引擎。</a>：适用于 Google Gemma 模型的轻量级、独立 C++ 推理引擎。 - google/gemma.cpp</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth：速度快 2-5 倍，显存占用减少 80% 的 QLoRA &amp; LoRA 微调</a>：速度快 2-5 倍，显存占用减少 80% 的 QLoRA &amp; LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/Mihaiii/semantic-autocomplete">GitHub - Mihaiii/semantic-autocomplete：一个极速的语义搜索 React 组件。按含义匹配，而不只是按字母。输入即搜索无需等待（不需要防抖）。按余弦相似度排序。</a>：一个极速的语义搜索 React 组件。按含义匹配，而不只是按字母。输入即搜索无需等待（不需要防抖）。按余弦相似度排序。 - Mihaiii/semantic-autocom...</li><li><a href="https://github.com/ContextualAI/gritlm">GitHub - ContextualAI/gritlm：生成式表征指令微调</a>：生成式表征指令微调。通过在 GitHub 上创建账户为 ContextualAI/gritlm 的开发做出贡献。</li><li><a href="https://azure.microsoft.com/en-us/pricing/offers/ms-azr-0044p">Azure 免费试用 | Microsoft Azure</a>：开始您的免费 Microsoft Azure 试用，并获得 200 美元的 Azure 额度，可随心使用。运行虚拟机、存储数据并开发应用。
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1227721101023379477)** (10 messages🔥):

- **Llama 3 模型即将推出**：Meta 已确认即将发布其 **Llama 3 模型**，并预告将在近期推出一些基础版本，据 [ZDNet](https://www.zdnet.com/article/meta-confirms-plans-to-start-rolling-out-llama-3-models-very-soon/) 和 [TechCrunch](https://techcrunch.com/2024/04/09/meta-confirms-that-its-llama-3-open-source-llm-is-coming-in-the-next-month/) 报道。
- **低预算微调 Mistral 7b**：一位成员请求关于微调 7b **Mistral** 模型的逐步指南或教程，表示对用于 GPU 的实惠云服务以及实现用于本地运行的 gguf 8bit 版本感兴趣。
- **全量微调的替代方案**：在可能不需要全量微调的情况下，建议使用 Unsloth 仓库，并推荐在 Colab GPU 上使用 **Qlora**，或者从 Vast 租用更强大的 GPU（如 3090 或 4090）。
- **寻找逻辑推理数据集**：一位成员询问了针对自然文本的命题逻辑和谓词逻辑推理的数据集，得到的回复指向了 [Logic-LLM](https://github.com/teacherpeterpan/Logic-LLM) 的 GitHub 仓库。
- **用于抓取数据的 Genstruct Notebook**：一位成员分享了一个 GitHub 脚本 [OllamaGenstruct/Paperstocsv.py](https://github.com/edmundman/OllamaGenstruct/blob/main/Paperstocsv.py)，该脚本在最初考虑编写自定义解决方案后，非常符合将网页转换为 markdown 以进行 genstruct 输入的需求。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.zdnet.com/article/meta-confirms-plans-to-start-rolling-out-llama-3-models-very-soon/">Meta 确认计划“很快”开始推出 Llama 3 模型</a>：Llama 3 是 Meta 对 OpenAI 的 GPT-4、Anthropic 的 Claude 3、Google 的 Gemini 以及其他 LLM 的回应。</li><li><a href="https://github.com/edmundman/OllamaGenstruct/blob/main/Paperstocsv.py">OllamaGenstruct/Paperstocsv.py at main · edmundman/OllamaGenstruct</a>：通过在 GitHub 上创建账号来为 edmundman/OllamaGenstruct 的开发做出贡献。</li><li><a href="https://github.com/teacherpeterpan/Logic-LLM/tree/main">GitHub - teacherpeterpan/Logic-LLM: "LOGIC-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning" 项目页面</a>："LOGIC-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning" 的项目页面 - teacherpeterpan/Logic-LLM
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1227524315420033034)** (109 条消息🔥🔥): 

- **用于 AI 引导的神秘学智慧**：Archive.org 的神秘学板块被建议作为一个有价值的资源，它**包含真理**并可能为 LLM 的开发提供*引导*。
- **对 World-Sim 回归的热情**：成员们兴奋地期待 **World-Sim 的回归**，讨论了潜在的新功能，并渴望在一周内实现回归。
- **World-Sim 的潜在用例**：讨论围绕寻找 **World-Sim** 的实际应用展开，一些成员主要将其视为一个有趣的工具，并建议了可能的教育用途。
- **本地 LLM vs. 云端模型**：关于运行本地 LLM 与依赖**云端模型**的可行性引发了对话，强调了计算限制和移动游戏化的趋势。
- **World-Sim 回归的预告与探索**：随着 World-Sim 的“建设中”页面发生变化，猜测四起，引发了关于 **teknium 的神秘沟通**以及实现 AGI 等潜在进展的理论。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Fast_inverse_square_root">快速平方根倒数算法 - Wikipedia</a>：未找到描述</li><li><a href="https://kbd.news/OGRE-cyberdeck-1835.html">OGRE cyberdeck</a>：OGRE 是一款末日风格或简单的外场 cyberdeck，是 Jay Doscher 的 Recover Kit 的仿制品。由 rmw156 分享。</li><li><a href="https://github.com/GitSquared/edex-ui/blob/master/media/screenshot_blade.png">edex-ui/media/screenshot_blade.png at master · GitSquared/edex-ui</a>：一款跨平台、可定制的科幻终端模拟器，具有先进的监控和触摸屏支持。 - GitSquared/edex-ui
</li>
</ul>

</div>
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1227528061012938753)** (74 条消息🔥🔥):

- **Nuzzy Bot 互动**：聊天中引入了一个名为 **Nuzzy** 的机器人供用户交互。为了获得最佳体验，建议开设一个专门的频道与 Nuzzy 对话，用户可以通过发送特定的激活命令来激活该机器人。
- **Udio 音乐生成器发布**：用户分享了 Twitter 和 Reddit 的链接，讨论新的 **Udio 音乐生成器**，将其与 Suno 进行对比，并强调了其功能，例如 90 秒的歌曲时长限制，以及每月为每位用户免费提供 1200 首歌曲。Reddit 上的一个详细帖子阐述了 Udio 卓越的音乐样本和潜在的发布日期。
- **关于 Nvidia 性能的讨论**：分享的[链接](https://www.semianalysis.com/p/nvidia-blackwell-perf-tco-analysis)详细分析了 Nvidia Blackwell GPU 的性能，包括 B100、B200 和 GB200 型号之间的对比。文章讨论了使用这些 GPU 涉及的总拥有成本（TCO）和推理成本。
- **征集 AI 工程指南（Playbook）贡献**：用户正在讨论为 **AI 工程指南** 收集素材，该指南将解决从 Jupyter notebooks 到生产环境的过渡问题。目前正在邀请引荐那些在成熟工程团队中具有使用 LLM 发布功能经验的高级工程师或团队负责人。
- **各种 AI 讨论与分享**：成员们分享了大量 AI 相关资源。主题涵盖 Meta 的 AI 硬件工作、Jeremy Howard 的 **Practical Deep Learning for Coders** 课程在 2024 年的相关性、使用 AI 制作音乐视频，以及像 *Rerank 3* 这样的 AI 模型对搜索引擎的潜在影响。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://stanford.zoom.us/j/99922151759?pwd=dW5CcUtVYkNybGZGY0hMWUZtVkZBZz09">加入我们的云高清视频会议</a>：Zoom 是现代企业视频通信的领导者，拥有简单、可靠的云平台，适用于移动端、桌面端和会议室系统的视频和音频会议、聊天及网络研讨会。Zoom ...</li><li><a href="https://x.com/AIatMeta/status/1778083237480321502">来自 AI at Meta (@AIatMeta) 的推文</a>：推出下一代 Meta Training and Inference Accelerator (MTIA)，这是我们专为 Meta 的 AI 工作负载设计的定制芯片家族中的最新成员。完整详情 ➡️ https://go.fb.m...</li><li><a href="https://course.fast.ai/">程序员实用深度学习 - 实用深度学习</a>：一门为有一定编程经验、想要学习如何将深度学习和机器学习应用于实际问题的人设计的免费课程。</li><li><a href="https://www.semianalysis.com/p/nvidia-blackwell-perf-tco-analysis">Nvidia Blackwell 性能 TCO 分析 - B100 vs B200 vs GB200NVL72</a>：GPT-4 盈利能力、成本、推理模拟器、并行性解析、大模型与小模型推理及训练中的性能 TCO 建模</li><li><a href="https://x.com/TickerSymbolYOU/status/1778108179110387812">来自 Alex (@TickerSymbolYOU) 的推文</a>：我的 Google Cloud Next 2024 精华剪辑为你节省了 91 分钟，并跳过了所有营销废话 00:00 - Google 的新 AI 超级计算机 05:50 - 用于数据中心的 Google Axion ARM CPU 07:25 - Gemini 的重大升级...</li><li><a href="https://www.honeycomb.io/blog/llms-demand-observability-driven-development">LLMs 需要可观测性驱动开发</a>：LLM 要求我们修改行为和工具，这种方式甚至会使普通的确定性软件开发受益。了解原因。</li><li><a href="https://x.com/itssandrakublik/status/1778422401648455694?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Sandra Kublik (@itsSandraKublik) 的推文</a>：推出我们的最新模型 Rerank 3！🚨 进一步增强搜索和 RAG 系统。内部包含什么？🧑‍🍳 - 4k 上下文长度，- 在复杂数据（如电子邮件、JSON 文档...）上的 SOTA 搜索准确率</li><li><a href="https://x.com/daniel_eckler/status/1778421669201093057?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Eckler by Design ✦ (@daniel_eckler) 的推文</a>：C3PO x Childish Gambino 🤖 👑 100% AI（官方音乐视频）@openAI + @runwayml + @suno_ai_ + @resembleai + @fable_motion + @midjourney + @topazlabs</li><li><a href="https://x.com/dylan522p/status/1777954675012305176?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Dylan Patel (@dylan522p) 的推文</a>：Nvidia Blackwell 性能 TCO 分析 B100 vs B200 vs GB200NVL72 GPT-4 盈利能力、成本推理模拟器并行性解析大模型与小模型推理及训练中的性能 TCO 建模...</li><li><a href="https://x.com/infobeautiful/status/1778059112250589561?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Information is Beautiful (@infobeautiful) 的推文</a>：让 ChatGPT 在 1 到 100 之间选一个数字——它会选哪个？（由 @Leniolabs_ 制作）</li><li><a href="https://x.com/udiomusic/status/1778045322654003448?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 udio (@udiomusic) 的推文</a>：推出 Udio，一款用于音乐创作和分享的应用程序，允许你通过直观且强大的文本提示词（text-prompting）生成你喜欢的风格的惊人音乐。1/11</li><li><a href="https://github.com/GregorD1A1/TinderGPT">GitHub - GregorD1A1/TinderGPT</a>：通过在 GitHub 上创建账号来为 GregorD1A1/TinderGPT 的开发做出贡献。</li><li><a href="https://x.com/minchoi/status/1778074187778683253?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Min Choi (@minchoi) 的推文</a>：这太疯狂了。Udio 刚刚发布，它就像是音乐界的 Sora。音乐质量惊人，100% AI。🤯 1. "沙丘百老汇音乐剧"</li><li><a href="https://old.reddit.com/r/singularity/comments/1bzd4bo/its_been_confirmed_the_suno_killer_is_called_udio/">已确认——“Suno 杀手”名叫 Udio</a>：我一直在调查一些人所谓的“Suno 杀手”——一个据称比现有模型好 2 到 10 倍的音乐生成 AI 模型...
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1227692289808269392)** (6 条消息): 

- **1-bit LLMs 论文演讲预告**：一场关于 1-bit Large Language Models 论文的演讲即将举行，承诺将深入探讨具有成本效益且高性能的 LLM。点击此处查看详情并[加入活动](https://lu.ma/jcxntjox)，在此处探索[论文](https://arxiv.org/abs/2402.17764)。

- **Elicit 播客集发布**：由 Elicit 的 Jungwon Byun 和 Andreas Stuhlmüller 主持的最新播客集已上线。在 [YouTube](https://www.youtube.com/watch?v=Dl66YqSIu5c) 上收听并订阅。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2402.17764">The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits</a>: 最近的研究（如 BitNet）正在为 1-bit 大语言模型 (LLM) 的新时代铺平道路。在这项工作中，我们引入了一个 1-bit LLM 变体，即 BitNet b1.58，其中每一个参数...</li><li><a href="https://lu.ma/jcxntjox">LLM Paper Club (1-bit LLMs paper) · Luma</a>: 本周 @rj45 将讲解 https://arxiv.org/abs/2402.17764 《1-bit LLM 时代：所有大语言模型都是 1.58 Bits》。同时请提交并为你心仪的下一篇论文投票：...</li><li><a href="https://www.youtube.com/watch?v=Dl66YqSIu5c&embeds_referring_euri=https%3A%2F%2Fwww.latent.space%2F&feature=emb_title">Supervise the Process of AI Research — with Jungwon Byun and Andreas Stuhlmüller of Elicit</a>: 时间戳：00:00:00 介绍 00:07:45 Johan 和 Andreas 如何联手创建 Elicit 00:10:26 为什么产品优于研究 00:15:49 演变...
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1227694310418546738)** (294 messages🔥🔥): 

- **三进制的胜利还是微不足道的技巧？**：成员们深入讨论了论文 [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)，辩论了 1-bit 大语言模型 (LLM) 的效率。**BitNet b1.58** 因其在保持成本效益的同时达到全精度 LLM 的性能而受到称赞，但也有人对三进制编码背后的真正创新以及在缺乏详细方法论的情况下结果的可复现性表示怀疑。
- **论文演示大混乱**：成员们在论文演示期间遇到了 Discord 屏幕共享功能的许多技术困难，导致他们开始探索替代的共享平台，并感叹远程会议软件的现状。
- **从论文到实践**：分享了多个 GitHub 链接，提供了如 [BitNet-Transformers](https://github.com/Beomi/BitNet-Transformers)（LLM 的 1-bit Transformers 实现）等资源，并引发了关于此类模型的实际实现和硬件要求的辩论。
- **对 MoEs 的思考**：成员们讨论了混合专家模型 (MoEs) 的概念和应用，链接了如 [Hugging Face 上的 MoE 博客文章](https://huggingface.co/blog/moe)以及详细介绍专家专业化和负载均衡的论文。对话还包括对推理时 MoEs 与语义路由器之间潜在重叠和差异的反思。
- **论文俱乐部选题与寒暄**：参与者通过选择和建议未来讨论的新论文结束了会议，同时感谢了演讲者的深入分析和贡献。还讨论了未来会议可能从 Discord 迁移到 Zoom 以避免技术问题。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://matrix.to/#/#temporarylatentspace:matrix.org">你被邀请在 Matrix 上交谈</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2402.17764">1-bit LLM 时代：所有大语言模型都是 1.58 Bits</a>：最近的研究（如 BitNet）正在为 1-bit 大语言模型 (LLM) 的新时代铺平道路。在这项工作中，我们引入了一个 1-bit LLM 变体，即 BitNet b1.58，其中每一个参数...</li><li><a href="https://llm-paper-club-asia-notes.vercel.app/papers/deekseek-moe">Nextra：下一代文档生成器</a>：Nextra：下一代文档生成器</li><li><a href="https://app.sli.do/event/bNV6mo3BFGhe8Bqzb1tonb/live/questions">加入 Slido：输入 #code 进行投票和提问</a>：参与实时投票、测验或问答。无需登录。</li><li><a href="https://shapes.inc">Shapes, Inc.</a>：Shapes 是可以在 Discord 上与你交谈的 AI 朋友</li><li><a href="https://arxiv.org/abs/2310.04793">FinGPT：金融数据集中开源大语言模型的指令微调基准</a>：在迅速扩展的自然语言处理 (NLP) 领域，基于 GPT 的模型在金融领域的潜力日益显现。然而，将这些模型与...集成...</li><li><a href="https://huggingface.co/blog/moe">Mixture of Experts 详解</a>：未找到描述</li><li><a href="https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1">mistral-community/Mixtral-8x22B-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://learning-exhaust.hashnode.dev/preview/6609ec4565bff73f1db1b51b">[草稿] 1.58 bits?</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2402.18041">大语言模型数据集：综合综述</a>：本文开始探索大语言模型 (LLM) 数据集，这些数据集在 LLM 的显著进步中起着至关重要的作用。数据集作为基础架构...</li><li><a href="https://www.youtube.com/watch?v=byCe7-c84d4">BloombergGPT - 金融领域的 LLM，对话 David Rosenberg - 639</a>：今天我们邀请到了 Bloomberg CTO 办公室机器学习策略团队负责人 David Rosenberg。在与 David 的对话中，我们...</li><li><a href="https://github.com/aurelio-labs/semantic-router">GitHub - aurelio-labs/semantic-router：超快速 AI 决策和多模态数据的智能处理。</a>：超快速 AI 决策和多模态数据的智能处理。 - aurelio-labs/semantic-router</li><li><a href="https://github.com/AI4Finance-Foundation/FinGPT">GitHub - AI4Finance-Foundation/FinGPT：FinGPT：开源金融大语言模型！变革 🔥 我们在 HuggingFace 上发布了训练好的模型。</a>：FinGPT：开源金融大语言模型！变革 🔥 我们在 HuggingFace 上发布了训练好的模型。 - AI4Finance-Foundation/FinGPT</li><li><a href="https://github.com/Beomi/BitNet-Transformers">GitHub - Beomi/BitNet-Transformers：0️⃣1️⃣🤗 BitNet-Transformers：在 PyTorch 中使用 Llama(2) 架构实现的 "BitNet: Scaling 1-bit Transformers for Large Language Models" 的 Huggingface Transformers 版本</a>：0️⃣1️⃣🤗 BitNet-Transformers：在 PyTorch 中使用 Llama(2) 架构实现的 &quot;BitNet: Scaling 1-bit Transformers for Large Language Models&quot; 的 Huggingface Transformers 版本 - Beomi/...
</li>
</ul>

</div>
  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1227697534017732638)** (8 条消息🔥): 

- **Hugging Chat 发布 Command R+**：Hugging Chat 已*免费*提供 [CohereForAI/c4ai-command-r-plus](https://huggingface.co/chat/models/CohereForAI/c4ai-command-r-plus) 模型，允许在聊天界面中集成网络搜索。

- **社区亮点回顾**：Hugging Face 展示了社区的贡献，例如 Hugging Face 的葡萄牙语教程、时尚试穿 AI、Deep Q Learning 仓库、图像增强工具、RAG 聊天机器人 Space，以及 character.ai 的开源替代方案。

- **教育性和信息性 Space 引起关注**：Hugging Face 社区成员创建了宝贵的资源，例如使用 wikipedia-small-3000-embedded 数据集生成响应而无需微调的 RAG 聊天机器人，以及构建神经网络分类器的分步指南，增强了集体知识。

- **无需训练，仅需推理**：在澄清一项贡献时，一位成员提到使用 `mixedbread-ai/mxbai-embed-large-v1` 模型对 wikipedia-small-3000 数据集进行嵌入，以检索信息用于 RAG 聊天机器人 Space，强调使用 RAG 进行推理而不是对模型进行微调。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/datasets/not-lain/wikipedia-small-3000-embedded">not-lain/wikipedia-small-3000-embedded · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/chat/models/CohereForAI/c4ai-command-r-plus">CohereForAI/c4ai-command-r-plus - HuggingChat</a>: 在 HuggingChat 中使用 CohereForAI/c4ai-command-r-plus</li><li><a href="https://www.youtube.com/watch?v=nK1hijr8Qng&t=74s">[IA a Z - 06] Apresentando o 🤗 Hugging Face</a>: 🤗🤗🤗🤗🤗🤗如果说有一件事是我喜欢的，那就是有大量的工具选项可以学习！这极大地简化了学习新事物的过程，特别是...</li><li><a href="https://huggingface.co/spaces/tonyassi/fashion-try-on">Fashion Try On - a Hugging Face Space by tonyassi</a>: 未找到描述</li><li><a href="https://github.com/SuleymanEmreErdem/deep-q-learning-applications">GitHub - SuleymanEmreErdem/deep-q-learning-applications: My Deep Q-Learning projects</a>: 我的 Deep Q-Learning 项目。通过在 GitHub 上创建账户，为 SuleymanEmreErdem/deep-q-learning-applications 的开发做出贡献。</li><li><a href="https://huggingface.co/spaces/not-lain/RMBG1.4-with-imageslider">RMBG1.4 with imageslider - a Hugging Face Space by not-lain</a>: 未找到描述</li><li><a href="https://github.com/RooTender/augmentator">GitHub - RooTender/augmentator: Ready-to-use tool for image augmentation</a>: 开箱即用的图像增强工具。通过在 GitHub 上创建账户，为 RooTender/augmentator 的开发做出贡献。</li><li><a href="https://not-lain-rag-chatbot.hf.space/"># RAG</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=76BkMULO7uw">Hot dog or not with Ollama, Mistral and LLava</a>: 在本教程中，我们使用 Ollama, Mistral 和 LLava 来查看图像是否为热狗 #python #pythonprogramming #llm #ml #ai #aritificialintel...</li><li><a href="https://github.com/EdoPedrocchi/RicercaMente">GitHub - EdoPedrocchi/RicercaMente: Open source project that aims to trace the history of data science through scientific research published over the years</a>: 旨在通过多年来发表的科学研究追溯数据科学历史的开源项目 - EdoPedrocchi/RicercaMente</li><li><a href="https://ragdoll-studio.vercel.app/">no title found</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=oVJsJ0e6jWk">Where&#39;s My Pic Demo</a>: 大家好，我是 Om Alve，在这个视频中，我将演示我的项目 'Where's my pic?'。该项目解决了搜索...</li><li><a href="https://huggingface.co/blog/joey00072/mixture-of-depth-is-vibe">Mixture of Depth is Vibe</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/dcarpintero/building-a-neural-network-for-image-classification">Building a Neural Network Classifier from the Ground Up: A Step-by-Step Guide</a>: 未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1227545273845153832)** (258 条消息🔥🔥): 

- **Gradio 频道查询**：成员们被引导至特定的 Discord 频道以咨询与 Gradio 相关的问题，并向一位用户提供了三个讨论 Gradio 的频道链接（[频道 1](https://discord.com/channels/879548962464493619/1025174734427656283), [频道 2](https://discord.com/channels/879548962464493619/1019296127847239751), 和 [频道 3](https://discord.com/channels/879548962464493619/1014577787039924226)）。

- **挑战 AI 界面导航**：讨论集中在是否可能训练一个模型来实现近乎完美的 OS GUI 导航，探索了直接基于像素的方法之外的替代方案。想法包括将应用程序解析为文本以及利用 OS 的辅助功能（accessibility features）。

- **对模型大小和速度的好奇**：关于 8x22B 模型和 120B 模型之间差异的问题被提出，并分享了关于推理过程中参数使用情况以及稠密模型（dense models）与混合专家模型（Mixture of Experts）相比的有效性的见解。

- **Hugging Face 数据集初学者资源**：一位用户询问了学习数据集基础知识的资源，并被引导至 Hugging Face 的文档，该文档提供了关于创建数据集、构建脚本、指标等方面的指导（[Datasets 文档](https://huggingface.co/docs/datasets/index)）。

- **寻找图表生成器**：一位用户正在寻找与 DiagramGPT 等效的 Hugging Face Space，该工具可以根据文本生成图表。另一位成员建议使用支持强提示词 Inpainting 的视觉 Q&A 模型，或者查看 Hugging Face Spaces 中的图表创建工具。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.eraser.io/diagramgpt">DiagramGPT – 由 Eraser 提供支持</a>：通过代码或自然语言提示生成技术图表。图表由 Eraser 提供支持。</li><li><a href="https://x.com/BigTechAlert">来自未定义的推文</a>：未找到描述</li><li><a href="https://huggingface.co/xai-org/grok-1">xai-org/grok-1 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/nroggendorff/dolphin/blob/main/app.py">app.py · nroggendorff/dolphin at main</a>：未找到描述</li><li><a href="https://huggingface.co/docs/datasets/index">Datasets</a>：未找到描述</li><li><a href="https://huggingface.co/mlabonne/phixtral-2x2_8">mlabonne/phixtral-2x2_8 · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/BigTechAlert/status/1778023873851871559>">来自 Big Tech Alert (@BigTechAlert) 的推文</a>：🆕 @huggingface 开始关注 @realmrfakename</li><li><a href="https://huggingface.co/spaces/nroggendorff/cascade/blob/main/app.py">app.py · nroggendorff/cascade at main</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/macadeliccc/laser-dolphin-mixtral-chat/blob/main/app.py">app.py · macadeliccc/laser-dolphin-mixtral-chat at main</a>：未找到描述</li><li><a href="https://github.com/moritztng/fltr">GitHub - moritztng/fltr: Like grep but for natural language questions. Based on Mistral 7B or Mixtral 8x7B.</a>：类似于 grep，但针对自然语言问题。基于 Mistral 7B 或 Mixtral 8x7B。- moritztng/fltr</li><li><a href="https://youtu.be/rSKMYc1CQHE?si=aEYaxyGwK7LdCLx6">编程冒险：模拟流体</a>：让我们尝试说服一堆粒子表现得（至少在某种程度上）像水。使用 C# 和 HLSL 编写，在 Unity 引擎中运行。源代码：h...</li><li><a href="https://youtu.be/Qz0KTGYJtUk?si=dq_Ptn1lpmwdNrt5">编程冒险：光线追踪</a>：我尝试创建了一个自定义的光线/路径追踪渲染器。特色：数学、着色器和猫！该项目使用 C# 和 HLSL 编写，并使用 Unity 游戏引擎...</li><li><a href="https://github.com/BrutPitt/glChAoS.P">GitHub - BrutPitt/glChAoS.P: 3D GPUs Strange Attractors and Hypercomplex Fractals explorer - up to 256 Million particles in RealTime</a>：3D GPU 奇异吸引子和超复分形浏览器 - 实时支持高达 2.56 亿个粒子 - BrutPitt/glChAoS.P
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1227828614133973052)** (1 条消息): 

- **用于 AI 安全的 Podman**：一段讨论视频，标题为[“AI 安全的根源是无根（Rootless）。关于 Podman：用于 GenAI 微服务”](https://youtu.be/YtxdWLSPo0U)，概述了如何通过终端安装 **Podman**，强调了其在容器化微服务环境中的 AI 安全重要性。它被建议作为 AI 应用中比 Docker 更安全的选择。

**提及的链接**：<a href="https://youtu.be/YtxdWLSPo0U">AI 安全的根源是无根。关于 Podman：用于 GenAI 微服务</a>：关于从终端安装 @Podman 的概述。#Podman #containers #AI #genAI #Docker #Linux #EdTech #deeplearning #microservices

  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1227558577523134504)** (7 条消息): 

- **小而强大的模型发现**：介绍了一个新的[信息提取演示](https://huggingface.co/spaces/urchade/gliner_multiv2.1)，展示了一个用于多语言目的的小型但强大的模型，并通过用户分享的图片展示了其效果。
- **Quanto 的量子飞跃**：对话中添加了一个包含[在 Transformers 中使用 Quanto 的 Notebook](https://github.com/andysingal/llm-course/tree/main/Quantization) 的 GitHub 仓库，建议进一步探索量化技术。
- **HuggingFace 模型游乐场**：分享了 [Marimo 应用](https://marimo.app/l/tmk0k2)作为 HuggingFace 模型的游乐场，鼓励用户保存代码并在这一新工具中进行实验。
- **RecurrentGemma 出现**：重点介绍了一篇题为[“RecurrentGemma：超越 Transformers 的飞跃，集成 PyTorch”](https://medium.com/technology-hits/recurrentgemma-a-leap-beyond-transformers-with-pytorch-integration-f6bb443766aa)的 Medium 文章，标志着 AI 建模领域潜在的范式转移。
- **Andrej Karpathy 简化 LLM 实现**：提到了 Andrej Karpathy 的一个 GitHub 项目，提供了一个精简的[纯 C/CUDA 的 LLM 训练实现](https://github.com/karpathy/llm.c)。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/urchade/gliner_multiv2.1">GLiNER-Multiv2.1 - urchade 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://marimo.app/l/tmk0k2">marimo | 下一代 Python notebook</a>：使用 marimo 无缝探索数据并构建应用，这是一款下一代 Python notebook。</li><li><a href="https://github.com/andysingal/llm-course/tree/main/Quantization">llm-course/Quantization at main · andysingal/llm-course</a>：通过在 GitHub 上创建账号来为 andysingal/llm-course 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c">GitHub - karpathy/llm.c: LLM training in simple, raw C/CUDA</a>：使用简单的原生 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1227674375474516081)** (11 条消息🔥): 

- **模型测试中的颜色混淆**：一位成员分享了他们的测试结果，提到虽然模型取得了*惊人的效果*，但有时会混淆颜色，例如将 T 恤和裤子的颜色反转。
- **LLM 的剪枝策略**：一位成员发现重读一篇关于剪枝的论文很有价值，讨论了针对特定用例使用目标数据集，并考虑在像 *Mixtral 8x22B* 这样的大型模型中进行专家剪枝的可能性。
- **热狗分类教程**：链接了一个 [YouTube 教程](https://www.youtube.com/watch?v=76BkMULO7uw)，展示了如何通过使用 **Ollama**、**Mistral** 和 **LLava** 等模型来判断图像是否描绘了热狗。
- **GitHub 上的 Arduino 孵化器**：一位成员分享了他们的 [GitHub 项目](https://github.com/DHTScienceGuy/Incubator)，这是一个基于 Arduino 的自动调节雏鸡孵化器，供社区探索。
- **葡萄牙语的 Hugging Face 概念**：发布了一篇教育性帖子和视频，介绍了 Hugging Face 的基本概念，为巴西开始学习 AI 的葡萄牙语使用者提供资源。可以找到 [帖子](https://iatalk.ing/hugging-face/) 和 [视频](https://www.youtube.com/watch?v=nK1hijr8Qng&t=74s) 的链接。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=76BkMULO7uw">使用 Ollama, Mistral 和 LLava 判断是否为热狗</a>：在本教程中，我们使用 Ollama, Mistral 和 LLava 来查看图像是否为热狗 #python #pythonprogramming #llm #ml #ai #aritificialintel...</li><li><a href="https://vimeo.com/933289700">test</a>：这是 Test Account 在 Vimeo 上发布的 "test"，Vimeo 是高质量视频及其爱好者的家园。</li><li><a href="https://github.com/DHTScienceGuy/Incubator">GitHub - DHTScienceGuy/Incubator: My Arduino Based Self Regulating Chicken Incubator</a>：我基于 Arduino 的自动调节雏鸡孵化器 - DHTScienceGuy/Incubator</li><li><a href="https://iatalk.ing/hugging-face/">Apresentando o 🤗 Hugging Face</a>：你好！今天我想向你介绍一款对于进入或已属于人工智能领域的人来说必不可少的工具：Hugging Face Hub，亲切地称为 hf，或者直接叫 🤗 Huggin…</li><li><a href="https://www.youtube.com/watch?v=nK1hijr8Qng&t=74s">[IA a Z - 06] Apresentando o 🤗 Hugging Face</a>：🤗🤗🤗🤗🤗🤗 如果说有什么是我喜欢的，那就是有大量的工具选项可以学习！这极大地简化了学习新事物的过程，尤其是...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1227539324963393547)** (6 条消息): 

- **寻求智能客服聊天系统的指导**：一位用户表示有兴趣构建一个用于智能客服的**多轮对话系统**，并询问是否有可以协助此项工作的研究论文或作品。
- **寻找 Samplers 和 Schedulers 的数学原理**：一位用户在学习了 ddpm 和 ddim 之后，询问关于 **schedulers** 和 **samplers** 的数学论文，旨在理解遵循上述方法的基础知识。
- **推荐 KD-Diffusion 论文和博客**：另一位成员推荐了 **k-diffusion** 论文作为理解 schedulers 的重要资源，并提供了一个 [Medium 博客链接](https://medium.com/@isamu-website/understanding-k-diffusion-from-their-research-paper-and-source-code-55ae4aa802f) 以对这些概念进行简化解释。
  

---


**HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1227825823084712097)** (1 条消息): 

_

- **通过 Device Map 支持多 GPU**：*Diffusers* 现在支持实验性的 `device_map` 功能，用于在多个 GPU 之间平衡模型流水线（pipeline）的分布。这一特性对于具有多个低 VRAM GPU 的配置特别有利，未来将根据社区兴趣添加更多策略。[阅读文档](https://huggingface.co/docs/diffusers/main/en/training/distributed_inference#device-placement)。

**提及的链接**：<a href="https://huggingface.co/docs/diffusers/main/en/training/distributed_inference#device-placement">Distributed inference with multiple GPUs</a>：未找到描述

---

**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1227521406938644480)** (14 条消息🔥): 

- **使用 Aladdin-Persson AI 消除水印**：分享了一个 Aladdin-Persson-AI-Watermark-Destroy 项目的 GitHub 仓库。这是一个较旧的工具，但据称仍然有效 ([GitHub Repo](https://github.com/Firdavs-coder/Aladdin-Persson-AI-Watermark-Destroy))。

- **NVidia GPU 进程监控建议**：建议使用 `nvidia-smi -l` 循环监控 GPU 进程以进行持续跟踪，同时外部建议使用 nvitop，这是一个交互式的 NVIDIA-GPU 进程查看器 ([nvitop GitHub](https://github.com/XuehaiPan/nvitop))。

- **探讨视频校正技术**：讨论了将图像去噪和伪影去除作为视频处理基础的方法，并推荐了两篇具有创新方法的论文：一篇介绍了 NAFNet 模型，另一篇提供了一个用于图像修复的大型数据集 ([NAFNet ARXIV](https://arxiv.org/abs/2204.04676), [Image Restoration Dataset](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Li_LSDIR_A_Large_Scale_Dataset_for_Image_Restoration_CVPRW_2023_paper.pdf))。

- **增强修复模型的泛化能力**：强调了在训练修复模型时数据增强（augmentation）的重要性，并引用了 BSRGAN 和 Real-ESRGAN 两篇论文，详细介绍了它们的增强流水线，为处理各种图像退化策略提供参考 ([BSRGAN ARXIV](https://arxiv.org/abs/2103.14006), [Real-ESRGAN ARXIV](https://arxiv.org/abs/2107.10833))。

- **特定视频数据集的挑战**：一位用户详细说明了在车载捕获的有限视频数据上训练模型的困难，这些数据受到噪声和多变光照的影响。讨论转向在其他数据集训练效果不佳后，如何通过分析视频来创建平衡的数据集。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2103.14006">Designing a Practical Degradation Model for Deep Blind Image Super-Resolution</a>：人们普遍认为，如果假设的退化模型与真实图像中的模型不符，单图像超分辨率（SISR）方法的表现将不尽如人意。尽管已经有几种退化模型...</li><li><a href="https://arxiv.org/abs/2107.10833">Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data</a>：尽管在盲超分辨率领域已经进行了许多尝试，以修复具有未知且复杂退化的低分辨率图像，但它们距离解决通用的真实世界退化图像仍有很大差距...</li><li><a href="https://github.com/Firdavs-coder/Aladdin-Persson-AI-Watermark-Destroy">GitHub - Firdavs-coder/Aladdin-Persson-AI-Watermark-Destroy: Aladdin-Persson-AI-Watermark-Destroy Public</a>：Aladdin-Persson-AI-Watermark-Destroy 公开版。通过在 GitHub 上创建账号为 Aladdin-Persson-AI-Watermark-Destroy 的开发做出贡献。</li><li><a href="https://github.com/XuehaiPan/nvitop">GitHub - XuehaiPan/nvitop: An interactive NVIDIA-GPU process viewer and beyond, the one-stop solution for GPU process management.</a>：一个交互式的 NVIDIA-GPU 进程查看器及更多功能，是 GPU 进程管理的一站式解决方案。 - XuehaiPan/nvitop</li><li><a href="https://arxiv.org/abs/2204.04676">Simple Baselines for Image Restoration</a>：尽管最近在图像修复领域取得了显著进展，但最先进（SOTA）方法的系统复杂性也在增加，这可能会阻碍其应用...
</li>
</ul>

</div>

---

**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1227534442701258764)** (10 条消息🔥): 

- **使用余弦相似度评估语言模型**：讨论集中在使用余弦相似度来评估语言模型，考虑生成的输出与评分标准（向量 C）和教学原则（向量 D）的对比。建议在评估协议中尝试对向量 B 使用 *weighted pooling*（加权池化）而非平均池化，以潜在地优先考虑某些方面（如评分标准）。

- **寻找 GPT-4 替代方案**：一位成员询问了 llama2/GPT-4 的替代方案，要求允许商业用途且能在 24-GB GPU 上进行训练，即使是较旧的版本也可以考虑。

- **寻求长上下文模型**：有人请求能够处理约 10-15k tokens 长上下文的 encoder-decoder 模型。建议包括研究 **BigBird** 和 **Longformer** 作为潜在选项。

- **Hugging Face Trainer 的暂停-恢复功能**：关于使用 Hugging Face 的 `Trainer` 暂停和恢复训练的问题得到了肯定的回答，并指出 `trainer.train()` 中的 `resume_from_checkpoint` 选项可用于此目的。

- **模型训练脚本协助**：一个关于使用 [transformers](https://huggingface.co/transformers/) 以及 Bits and Bytes（用于量化）和 Lora（用于低秩自适应）等附加组件训练模型的脚本请求。该脚本包含了使用 `accelerate launch` 执行的内容，并寻求对正确实现和模型保存过程的验证。
  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1227555268133650492)** (8 messages🔥): 

- **调度器数学原理说明**：一位成员询问了关于理解 DDPM 和 DDIM 之后各种 AI 调度器（schedulers）和采样器（samplers）背后数学原理的推荐材料。建议学习 [fast.ai Part 2 课程](https://course.fast.ai/Lessons/part2.html)，查看 [GitHub 上的代码实现](https://github.com/huggingface/diffusers/issues?q=sort%3Acreated-asc+label%3A%22good+first+issue%22%2C%22Good+second+issue%22%2Chacktoberfest)，阅读 [Hugging Face 关于 diffusion 的博客文章](https://huggingface.co/blog?tag=diffusion&p=1)，并关注[相关话题的讨论](https://github.com/huggingface/diffusers/discussions?discussions_q=sort%3Atop)。

- **使用 MultiControlnet 进行分布式推理**：一位成员询问如何在多块 GPU（每块约 10GB VRAM）上加载 MultiControlnet 进行推理，由于 VRAM 限制，他们正在寻找 **Hugging Face Accelerate** 之外的解决方案。在初次尝试 Accelerate 失败后，他们被引导至[关于设备放置（device placement）的更详细指南](https://huggingface.co/docs/diffusers/main/en/training/distributed_inference#device-placement)以解决其 VRAM 顾虑。

- **寻找图层分解工具**：一位用户正在寻找类似于[网页](https://cre8tiveai.com/ld)上展示的图层分解器（Layer Decomposer），但未能找到关于如何实现此类工具的明确信息。他们请求提供有关该主题的仓库或文章线索。

- **“Balanced” Device_map 的性能权衡**：一位社区成员询问使用 **device_map** 的 “balanced” 策略如何影响**推理时间和资源效率**，特别是在 VRAM 较低的 GPU 上。他们寻求在内存限制下优化性能的配置建议。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/diffusers/main/en/training/distributed_inference">使用多 GPU 进行分布式推理</a>：未找到描述</li><li><a href="https://cre8tiveai.com/ld"> Layer Decomposer（图层分离 AI）｜图像和视频编辑 AI 工具：cre8tiveAI</a>：一个基于 AI 的 SaaS，可在 10 秒内解决各种照片和插图编辑任务，如自动绘画、提高图像和视频分辨率以及剪辑...</li><li><a href="https://huggingface.co/docs/diffusers/main/en/training/distributed_inference#device-placement">使用多 GPU 进行分布式推理</a>：未找到描述
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1227718650820169758)** (1 messages): 

- **Perplexity AI 与 Raycast 合作**：Perplexity AI 宣布与 Raycast 合作，为新的 Raycast Pro 年度订阅用户提供 **3 个月的免费 Perplexity Pro**，如果包含高级 AI 插件，则免费 6 个月。在 Raycast 的[博客文章](https://www.raycast.com/blog/more-ai-models)中了解更多关于此次合作及 Mac 上 AI 模型集成的信息。
- **庆祝 ChatGPT 的里程碑**：2022 年 11 月 30 日 [ChatGPT 的发布](https://openai.com/blog/chatgpt)创下了纪录，**仅用两个月月活跃用户就达到了 1 亿**。该博客文章强调了所有软件企业在定义 AI 战略方面的激增，并讨论了 LLM 的潜力，而无需担心末日降临。

**提到的链接**：<a href="https://www.raycast.com/blog/more-ai-models">一个界面，多种 LLM - Raycast 博客</a>：Raycast AI 随着 Anthropic Claude 3、Perplexity 和更多模型的加入而变得更加强大——使其成为 AI 的完美 UI

---

**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1227522057873526795)** (266 条消息🔥🔥): 

- **模型偏好与权衡**：用户讨论了更小的上下文窗口与更智能的 AI 模型之间的权衡，其中一人因其实用性而主张使用 Opus 4k。他们思考了未来的选择，如 Opus 200k 或 Haiku 200k，并指出 Opus 4k 可能因其在处理许多短查询时的高效率而受到欢迎。
- **Claude 3 Opus**：关于在写作模式下使用 Claude 3 Opus 时是否应开启 Pro 模式存在争议。一位用户建议将其关闭，因为据说 Pro 模式会利用网络搜索，而这可能对写作模式没有帮助。
- **Perplexity 的图片包含限制**：成员们报告了尝试直接在 Perplexity 的回复中包含图片时遇到的困难。分享了一个链接来演示这一点，一位用户成功修改了 Prompt 使其生效，这表明可能存在变通方法。
- **与外部工具的集成及 Raycast 合作伙伴关系**：讨论了 Perplexity 集成到 Web 浏览器或 Raycast 等外部工具的情况，一位用户分享了关于 Raycast 合作的 [Perplexity AI Twitter 公告](https://www.perplexity.ai/search/What-is-Raycast-.ItnfuzRRECM3I.88NKApg)。用户交流了他们使用集成了 Perplexity 的 Raycast 的经验和益处。
- **API 身份验证问题**：一位用户在尝试使用 Perplexity API 时遇到了 401 错误，在发现支付自动充值（auto top up）未激活并影响了其授权后，问题得到了解决。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.tickertea.com/subscribe">Ticker Tea</a>：提供有关重要新闻的每日更新：金融市场。</li><li><a href="https://docs.perplexity.ai/">pplx-api</a>：未找到描述</li><li><a href="https://x.com/perplexity_ai/status/1778067977566294448">来自 Perplexity (@perplexity_ai) 的推文</a>：我们与 Raycast 合作，让知识在 Mac 上的任何地方、任何时间都触手可及。新的 Raycast Pro 年度订阅者可免费获得 3 个月的 Perplexity Pro，如果包含高级功能则为 6 个月...</li><li><a href="https://www.youtube.com/watch?v=0O2yTG3n1Vc">我测试了 Humane AI Pin - 它并不好。</a>：我花了很多时间努力让我的视频尽可能简洁、精炼且对你有用 - 如果你想支持我的这一使命，请...</li><li><a href="https://github.com/wallabag/wallabagger/blob/bc9bae830c2f51403b1679efdfab9a497365f05d/wallabagger/js/options.js#L109">wallabagger/wallabagger/js/options.js at bc9bae830c2f51403b1679efdfab9a497365f05d · wallabag/wallabagger</a>：适用于 wallabag v2 的 Chrome / Firefox / Opera 插件。通过在 GitHub 上创建账号为 wallabag/wallabagger 的开发做出贡献。</li><li><a href="https://github.com/ollama/ollama">GitHub - ollama/ollama: 快速上手 Llama 2、Mistral、Gemma 和其他大型语言模型。</a>：快速上手 Llama 2、Mistral、Gemma 和其他大型语言模型。 - ollama/ollama</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1c0tdsb/comment/kyzsho1/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button">Reddit - 深入探索一切</a>：未找到描述
</li>
</ul>

</div>

---

**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1227575877185962055)** (14 条消息🔥): 

- **探索视频游戏查询**：成员们在 Perplexity 上搜索关于视频游戏的见解，可能是为了寻求游戏分析或对比。
- **深入研究财务数据**：一位用户探索了与橄榄相关的净值，表明了对农产品经济方面的兴趣。
- **追求 Meta 自定义 AI**：Perplexity 上的一个搜索查询提到了“Meta custom AI”，暗示了对 Meta 专有人工智能系统或自定义 AI 解决方案的探索。
- **科技爱好者探讨芯片进展**：Gaudi 3 芯片是关注的对象，一位成员寻求了相关信息，可能涉及其规格或在 AI 任务中的性能。
- **比较分析方法**：用户关于 Neowave 与 Elliott Wave 的搜索表明了对不同分析技术的讨论，可能是在金融市场的背景下。

---

**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1227622224664596561)** (16 条消息🔥):

- **针对大输入的“粘贴至提示词”**：用户可以通过将文件转换为纯文本，并直接将多达 199k tokens 免费粘贴到提示词字段中，从而绕过 **labs.perplexity** 无法上传文件的问题。
- **可用的实时网络响应**：**Perplexity API** 不直接提供实时网络响应，但用户可以使用 *sonar online models* 从网络获取信息。
- **不支持 Claude Opus 模型**：关于 **Perplexity API** 是否提供 Claude Opus 模型的咨询显示该模型暂不支持。对可用模型感兴趣的用户被引导至官方的 [model documentation](https://docs.perplexity.ai/docs/model-cards)。
- **澄清 API 功能**：虽然 **Perplexity API** 的功能路线图可能会发生变化，但用户可以申请引用访问权限，以便在请求中包含来源页面的 URL。
- **API 模拟网页版**：通过利用适当的请求策略和参数设置，用户可以通过 **Perplexity API** 获得与 Perplexity 网页版类似的答案。

**提到的链接**：<a href="https://perplexity.typeform.com/to/j50rnNiB)">发现 Typeform，让表单变得有趣</a>：在几分钟内无需代码即可创建美观的交互式表单。免费开始使用。

  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1227702431606116482)** (1 条消息): 

- **Meta 令人印象深刻的 AI 训练基础设施**：*Meta Training and Inference Accelerator (MTIA)* 拥有卓越的 **354 TFLOPS/s (INT8) 性能，功耗仅为 90 瓦**。[官方博客文章](https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/) 详细介绍了 Meta 的下一代基础设施如何构建以支持 GenAI 产品、推荐系统和 AI 研究，并预计将大幅增长以满足日益增长的计算需求。

**提到的链接**：<a href="https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/">未找到标题</a>：未找到描述

  

---


**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/)** (1 条消息): 

mobicham: https://github.com/BobMcDear/attorch
  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1227752239091945574)** (63 条消息🔥🔥): 

- **可用的 CUDA Kernels 集合**：新版本的 llm.c 包含了一系列 **CUDA kernels**，可以在 GitHub 上的 [karpathy/llm.c](https://github.com/karpathy/llm.c/tree/master/dev/cuda) 找到。
- **极速线性层库**：发布了一个采用半精度累加的超快线性层库，声称在 **4090 GPU 上实现了 >300 tflops**，性能显著优于 PyTorch。它专为极速推理设计，仓库托管在 GitHub 上的 [torch-cublas-hgemm](https://github.com/aredden/torch-cublas-hgemm)。
- **高速推理工具保持精度**：该高性能线性层库的作者提到，虽然该库在推理方面比 nn.Linear 层快得多，但它对所有形状可能性的产生结果几乎完全相同，目前尚不支持梯度（gradients）。
- **推理库的潜在问题和未来更新**：该快速线性层库的作者承认这是一个非常新的项目，可能存在 bug，但他们确认结果已经过彻底测试。他们还表示计划很快添加梯度支持。
- **关于精度和性能的讨论**：澄清了该高速推理库使用 **FP16** 配合 FP16 累加，在消费级 GPU 上比 FP32 配合 FP32 累加提供显著更快的性能，而这在 RTX 6000 ADA 等数据中心卡上似乎没有那么大的优势。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/tree/master/dev/cuda">llm.c/dev/cuda at master · karpathy/llm.c</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账户为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/aredden/torch-cublas-hgemm">GitHub - aredden/torch-cublas-hgemm: PyTorch half precision gemm lib w/ fused optional bias + optional relu/gelu</a>：带有融合可选 bias + 可选 relu/gelu 的 PyTorch 半精度 gemm 库 - aredden/torch-cublas-hgemm
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1227781176962584606)** (3 条消息):

- **ViT 模型量化问题**：一位用户在尝试量化来自 HuggingFace 的 `google/vit-base-patch16-224-in21k` 模型时遇到了问题，收到了 `RuntimeError: empty_strided not supported on quantized tensors yet` 错误。他们引用了一个 [GitHub issue](https://github.com/pytorch/pytorch/issues/74540) 以获取更多细节，并正在寻求关于量化（quantization）和剪枝（pruning）的指南。

- **BERT 的 FlashAttention 困扰**：另一位用户正在尝试为 BERT 模型添加 `flashattention2` 支持，并发现打过补丁和未打补丁的模型之间存在差异。他们正在寻求针对此问题的见解。

**提到的链接**：<a href="https://github.com/pytorch/pytorch/issues/74540">No factory functions for strided quantized tensors · Issue #74540 · pytorch/pytorch</a>：🐛 描述 Bug。对于非量化张量，既有 empty 也有 empty_strided。然而，对于量化张量，函数只有 empty 变体。这意味着很难……

---

**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1227551275869143041)** (7 条消息): 

- **PMPP 书籍讲座学习小组**：一名成员建议针对基于 PMPP 书籍的伊利诺伊大学讲座举办观看派对，重点是互动讨论。提议的时间是工作日的 CET 凌晨，并请感兴趣的参与者回复。
- **按自己的节奏前进**：社区收到一个温和的提醒，不要因为 Discord 中的学习进度而感到被冷落，强调个人学习的进步，并将其与学习语言进行了类比。
- **学习 CUDA 比德语容易**：有人幽默地对比了学习 **CUDA** 和德语的难度，认为 CUDA 要容易得多，频道内的其他人也对此表示赞同。
- **已安排的观看派对时段和 Discord 小组**：为对学习小组感兴趣的人创建了一个 Discord 小组，第一场会议定于周六 7:30 GMT / 8:30 BST / 9:30 CET，并提供了邀请链接：[学习小组邀请](https://discord.gg/XwFJRKH9)。
- **建议利用现有的语音频道**：针对为学习会议建立新小组的提议，有人建议使用现有的语音频道之一，以获得更好的参与度。

**提到的链接**：<a href="https://discord.gg/XwFJRKH9">Join the PMPP UI lectures timezones Discord Server!</a>：查看 Discord 上的 PMPP UI 讲座时区社区 - 与其他 4 名成员一起聚会，享受免费的语音和文字聊天。

---

**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1227939087256649781)** (3 条消息): 

- **数据集交付**：一名成员诙谐地宣布一个“超大数据集”到货了。
- **任务组织咨询**：有人建议为团队创建一个后续任务清单。
- **测试承诺**：一名成员提到他们已经安排了使用 "mamba" 的测试任务。

---

**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1227574193789341726)** (3 条消息): 

- **找到了！是悟空 (Goku)**：服务器头像已被确认是角色 **悟空 (Goku)**。
- **达成里程碑**：服务器已成功突破 **5000 名成员**。
- **关于知识摄取的明智建议**：一名成员建议，**每周阅读一次**并让问题引导研究深度，比过度摄取信息更有效，并指出做一个 "cons00mer"（过度消费者）对大脑总是有害的。

---

**CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1227528423958773860)** (76 条消息🔥🔥):

- **共享了量化脚本和基准测试**：已共享针对 Hugging Face 模型的[量化基准测试脚本](https://github.com/mobiusml/hqq/blob/master/examples/llama2_benchmark/quant_llama2_hqq_demo.py)，这些脚本带有预定义参数，可返回特定的性能表。成员们讨论了使用 base 模型而非 chat 模型，以及在执行时禁用 cache 的可能性。
- **最新的高效代码**：提供了[可运行的量化代码](https://github.com/mobiusml/hqq/blob/ao_int4_mm/hqq/core/torch_lowbit.py)，包含正确的 perplexity 分数，以及在 3090、4090 和 A100 等不同 GPU 上使用 HQQLinear 和 torchao int4 kernel 的速度基准测试详情。
- **关于 int4 Kernel 的性能讨论**：人们对 int4 kernel 的性能表示担忧，特别是为什么在 3090 GPU 上 `Int4: AOInt4 backend` 比使用 `torch.compile` 的 PyTorch 慢得多。讨论涉及了可能的原因以及在 A100 GPU 上进行测试的必要性。
- **解决复现和转换问题**：成员们遇到了结果复现问题，并确认了沿不同轴进行量化时存在的 perplexity 差异。这些挑战促使大家共享了转换阶段，并明确了在 hqq 量化参数与 tinygemm int4wo 之间进行转换的方法，深入探讨了行变更（row altering）如何影响精度。
- **Perplexity 指标的差异及潜在原因**：讨论重点关注 perplexity 指标，观察到 HQQ 和 GPTQ 结果之间存在差异。除了进行一些故障排除外，讨论还指出 dropout、layernorm eps 值以及权重转换过程中的差异可能是影响 perplexity 的潜在因素。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/mobiusml/hqq/blob/ao_int4_mm/hqq/core/torch_lowbit.py">hqq/hqq/core/torch_lowbit.py at ao_int4_mm · mobiusml/hqq</a>：Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/mobiusml/hqq/blob/ao_int4_mm/hqq/core/torch_lowbit.py#L137-L139">hqq/hqq/core/torch_lowbit.py at ao_int4_mm · mobiusml/hqq</a>：Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/mobiusml/hqq/blob/master/examples/llama2_benchmark/quant_llama2_hqq_demo.py">hqq/examples/llama2_benchmark/quant_llama2_hqq_demo.py at master · mobiusml/hqq</a>：Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/pytorch-labs/gpt-fast/blob/main/scripts/convert_hf_checkpoint.py#L89">gpt-fast/scripts/convert_hf_checkpoint.py at main · pytorch-labs/gpt-fast</a>：在少于 1000 行 Python 代码中实现简单高效的 pytorch 原生 transformer 文本生成。 - pytorch-labs/gpt-fast</li><li><a href="https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py">transformers/src/transformers/models/llama/modeling_llama.py at main · huggingface/transformers</a>：🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的先进机器学习库。 - huggingface/transformers</li><li><a href="https://github.com/mobiusml/hqq/blob/ao_int4_mm/hqq/core/torch_lowbit.py#L135">hqq/hqq/core/torch_lowbit.py at ao_int4_mm · mobiusml/hqq</a>：Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/zhxchen17/gpt-fast/blob/hqq_4bit_llama2_7b/scripts/convert_hf_checkpoint.py#L57">gpt-fast/scripts/convert_hf_checkpoint.py at hqq_4bit_llama2_7b · zhxchen17/gpt-fast</a>：在少于 1000 行 Python 代码中实现简单高效的 pytorch 原生 transformer 文本生成。 - zhxchen17/gpt-fast</li><li><a href="https://github.com/pytorch/pytorch/blob/8aa08b8b9d1fab2a13dc5fbda74c553cb2a08729/aten/src/ATen/native/cuda/int4mm.cu#L805-L860">pytorch/aten/src/ATen/native/cuda/int4mm.cu at 8aa08b8b9d1fab2a13dc5fbda74c553cb2a08729 · pytorch/pytorch</a>：Python 中具有强大 GPU 加速功能的张量和动态神经网络 - pytorch/pytorch</li><li><a href="https://gist.github.com/mobicham/4b08fb0bdf4c3872e5bbf68ec9803137">hqq_eval_int4mm_noppl.py</a>：GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://github.com/zhxchen17/gpt-fast/commit/5cdb8bd61fcacccf018cd7a1c49417442e03644a">HQQ 4 bit llama 2 7b · zhxchen17/gpt-fast@5cdb8bd</a>：export MODEL_REPO=meta-llama/Llama-2-7b-hf scripts/prepare.sh $MODEL_REPO python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int4-hqq --groupsize 64 python generate.py --...</li><li><a href="https://github.com/mobiusml/hqq/blob/ao_int4_mm/hqq/core/torch_lowbit.py#L197-L221">hqq/hqq/core/torch_lowbit.py at ao_int4_mm · mobiusml/hqq</a>：Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq
</li>
</ul>

</div>

**CUDA MODE ▷ #[triton-viz](https://discord.com/channels/1189498204333543425/1225499141241573447/1227739060232261686)** (3 messages): 

- **微调 triton-viz 机器人增强功能**：成员们讨论了 **triton-viz chatbot** 的潜在改进，建议包括修改 **hyperlinks** 以及添加 **code annotations** 以实现逐步演示，这表明升级将采取分阶段的方法。
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1227548729788207104)** (67 messages🔥🔥): 

- **CUDA 前向传播效率提升**：一位用户报告称，对于具有特定参数（B=4, T=1024, GPT-2 124M）的 GPT-2 模型，其 **pure CUDA** 前向传播每轮运行时间为 110ms，而 **PyTorch** 为 180ms。他们计划进一步调查这一差异。
  
- **利用 CUDA 的 C 子集进行潜在优化**：讨论集中在纯 CUDA 的 C 子集中进行优化，权衡了局限性以及使用 inline assembly 以调用 Tensor Cores 的潜在需求。**cuBLAS** 已确认使用 Tensor Cores，并且有建议认为 `__restrict__` 可以优化编译器对指针参数的处理。

- **实现 Warp 级归约和 Kernel Fusion**：一位成员分享了关于在 **CUDA** 中使用 **Cooperative Groups** 和 **templating** 的见解，包括 Warp 级归约、编译时的 Kernel Fusion，以及使用宏代替模板。他们还提供了一个详细的 **[LayerNorm kernel 示例](https://godbolt.org/z/6zq6Kfecc)**，该示例使用了 Cooperative Groups，并在 A4000 GPU 上实现了显著的速度提升。

- **评估 CUDA 的 Cooperative Groups**：一位成员强调了 **[Cooperative Groups](https://developer.nvidia.com/blog/cooperative-groups/)**，这是 2017 年推出的一个 CUDA 特性，允许灵活且动态的线程分组，并对其在 CUDA MODE 使用的 CUDA 书籍中未被更广泛覆盖表示惊讶。讨论还涉及了其有效性以及由该成员主导的近期应用。

- **考虑在 CUDA 开发中从 C 转向 C++**：讨论转向了从 **C 转向 C++** 是否能为 CUDA 开发带来具体好处，涉及 nvcc 对 C++ 编译器的使用、C++ 特性（如 `constexpr` 和 templates）的便利性，以及使用带有模板的 **Cutlass 库** 的可能性。关于 C++ 在处理动态共享内存大小和多种数据类型方面的优势达成了共识。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://developer.nvidia.com/blog/cooperative-groups/">Cooperative Groups: Flexible CUDA Thread Programming | NVIDIA Technical Blog</a>: 在高效的并行算法中，线程通过协作和共享数据来执行集体计算。为了共享数据，线程必须同步。共享的粒度随算法而异...</li><li><a href="https://github.com/karpathy/llm.c">GitHub - karpathy/llm.c: LLM training in simple, raw C/CUDA</a>: 在简单的原生 C/CUDA 中进行 LLM 训练。通过在 GitHub 上创建账号为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://godbolt.org/z/6zq6Kfecc">Compiler Explorer - CUDA C++ (NVCC 12.3.1)</a>: __global__ void crossentropy_forward_kernel1(float* losses,                             float* probs, int* targets,                             int B, int T, int V) {     int i = blockIdx.x * blockDim...
</li>
</ul>

</div>
  

---



**Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1227538082124009512)** (1 messages): 

- **可解释性工具适用于现代 RNN**：新研究表明，Transformer 的流行可解释性工具确实可以适配用于 **Mamba** 和 **RWKV** 等现代 RNN。向量算术、Tuned Lens 以及从微调后的 RNN 中提取潜在知识以产生错误答案等技术被证明是有效的。[阅读论文](https://arxiv.org/abs/2404.05971) | [浏览代码](https://github.com/EleutherAI/rnngineering)

- **RNN 在语言建模中的复兴**：作为 RNN 语言模型领域的新秀，**Mamba** 和 **RWKV** 凭借其出色的性能以及可解释性工具的成功应用，可能会追随 Transformer 开辟的道路。[参与 Twitter 讨论](https://x.com/norabelrose/status/1777975663590531533)

- **对协作努力的致谢**：<#1186803874846212257> 频道在推动 RNN 语言模型发展方面的积极协作得到了认可和赞赏。向参与该研究的贡献者表示感谢。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.05971">Does Transformer Interpretability Transfer to RNNs?</a>：循环神经网络（RNN）架构的最新进展（如 Mamba 和 RWKV）使得 RNN 在语言建模困惑度（perplexity）方面能够匹配或超过同等规模的 Transformer...</li><li><a href="https://github.com/EleutherAI/rnngineering">GitHub - EleutherAI/rnngineering: Engineering the state of RNN language models (Mamba, RWKV, etc.)</a>：工程化 RNN 语言模型（Mamba, RWKV 等）的状态 - EleutherAI/rnngineering</li><li><a href="https://x.com/norabelrose/status/1777975663590531533">Nora Belrose (@norabelrose) 的推文</a>：RNN 语言模型最近正在复兴，出现了 Mamba 和 RWKV 等新架构。但为 Transformer 设计的可解释性工具是否适用于这些新的 RNN？我们测试了 3 种流行的...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1227583941280796682)** (83 messages🔥🔥): 

- **Mixtral 8x22B 在 AGIEval 中表现出色**：新的 [Mixtral 8x22B 模型](https://x.com/jphme/status/1778030213881909451) 展示了令人印象深刻的初步 AGIEval 结果，优于其他开源（基础）模型。社区对其发布和高性能表现充满热情。
- **预测 AI 的未来**：一位成员强调了基于 Metaculus 和 Manifold 中位数预测的 [AI 预测时间线](https://theaidigest.org/timeline)，提供了关于 AI 进展、潜在危害和社区反应的粗略感知，一些人对 llama3 等模型开源的预计时间线提出了质疑。
- **对 AI 影响选举安全的担忧**：成员们对 deepfakes 等 AI 技术对即将到来的选举可能产生的负面影响表示担忧，一些人考虑在选举期间避开社交媒体，以专注于个人健康。
- **关于扩展 Encoder 模型的技术讨论**：频道内进行了一场关于在 BERT 等 Encoder 模型中扩展上下文窗口大小挑战的技术讨论。[FlashAttention 的整合](https://mosaicbert.github.io/) 到 Encoder 模型中以及它在 HuggingFace 等流行库中的缺失引发了好奇。
- **下载 The Pile 用于研究**：有一场关于下载 The Pile 数据集时大小差异的对话，用户澄清 886 GB 指的是未压缩的大小，而压缩文件可能明显更小。成员们确认他们使用 The Pile 严格用于学术和研究目的。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/jphme/status/1778030213881909451">Jan P. Harries (@jphme) 的推文</a>：@MistralAI 的初步 AGIEval 结果看起来很棒 👇 - 感谢发布这个猛兽，伙计们！👏 https://x.com/jphme/status/1778028110954295486 ↘️ 引用 Jan P. Harries (@jphme) 的初步 AGIEval 结果...</li><li><a href="https://mosaicbert.github.io/">MosaicBERT：一种为快速预训练优化的双向 Encoder</a>：未找到描述</li><li><a href="https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2">Learning Agency Lab - 自动作文评分 2.0 | Kaggle</a>：未找到描述</li><li><a href="https://theaidigest.org/timeline">AI 预测时间线 - AI Digest</a>：关于 AI 能力、潜在危害和社区反应的预期</li><li><a href="https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1/discussions/4">mistral-community/Mixtral-8x22B-v0.1 · 基准测试结果已发布！</a>：未找到描述</li><li><a href="https://github.com/huggingface/transformers/issues/26350">社区贡献：为更多架构添加 Flash Attention 2 支持 · Issue #26350 · huggingface/transformers</a>：功能请求 Flash Attention 2 是一个提供注意力操作内核的库，用于更快、更节省内存的推理和训练：https://github.com/Dao-AILab/flash-attention...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1227519788532891729)** (132 messages🔥🔥):

- **对抗性图像示例不仅仅是噪声**：对抗性图像示例可以超越非结构化噪声的外观；它们可能涉及实际的畸变，例如改变狗的鼻子。[Adversarial Image Research](https://arxiv.org/abs/1907.07174) 显示机器学习模型在现实世界、未经修改的示例中具有共同的弱点。
- **子集微调探索**：一种名为 **Subset fine-tuning** (SubTuning) 的新方法证明，仅微调神经网络层的一个子集即可获得与微调所有层相当的性能，在多任务学习和减少所需计算资源方面具有潜在优势 [SubTuning Research](https://arxiv.org/abs/2302.06354)。
- **揭示模型训练预算约束**：一位成员强调了微调预算限制是不从头开始训练 BERT 等模型的正当理由；另一位成员则强调了无论预算如何，技术相关性都很重要 [No Paper Reference Provided]。
- **发现 Mistral 的双向注意力**：最近的研究结果表明，Mistral 模型可能利用了一种形式的双向注意力（bidirectional attention），导致在启用双向注意力的情况下，Mistral-7B 在所有层和位置上都具有很高的余弦相似度 [No Paper Reference Provided]。
- **训练混合 RWKV5 Transformer 模型查询**：有人对是否有人训练过混合 RWKV5 Transformer 模型感兴趣，讨论指向了 RWKV 服务器以获取更多细节，目前尚未注意到此类模型的公开训练 [No Paper Reference Provided]。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.07143">Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention</a>: 这项工作介绍了一种有效的方法，可以将基于 Transformer 的大语言模型 (LLMs) 扩展到无限长的输入，且内存和计算量有限。我们提出的方法中的一个关键组件...</li><li><a href="https://arxiv.org/abs/2307.13912">Embedding Democratic Values into Social Media AIs via Societal Objective Functions</a>: 我们能否设计人工智能 (AI) 系统，在对社交媒体信息流进行排序时，将减轻党派敌意等民主价值观作为其目标函数的一部分？我们介绍...</li><li><a href="https://arxiv.org/abs/2404.06654">RULER: What&#39;s the Real Context Size of Your Long-Context Language Models?</a>: 大海捞针 (NIAH) 测试旨在检验从长篇干扰文本（“干草堆”）中检索特定信息（“针”）的能力，目前已被广泛采用...</li><li><a href="https://arxiv.org/abs/2310.17041">On Surgical Fine-tuning for Language Encoders</a>: 对预训练神经语言编码器的所有层进行微调（无论是使用全部参数还是使用参数高效方法）通常是将其适配到新任务的默认方式。我们展示...</li><li><a href="https://arxiv.org/abs/2302.06354">Less is More: Selective Layer Finetuning with SubTuning</a>: 微调预训练模型已成为在新任务上训练神经网络的标准方法，能够实现快速收敛并提高性能。在这项工作中，我们研究了一种替代方案...</li><li><a href="https://tenor.com/view/avocado-bacon-salad-lunch-salad-gif-12338945">Avocado Bacon Salad Lunch GIF - Avocado Bacon Salad Lunch Salad - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://x.com/_akhaliq/status/1775740222120087847?t=55VlAx9tjP9PUgvRcnIfMQ&s=33">Tweet from AK (@_akhaliq)</a>: Google 推出 Mixture-of-Depths：在基于 Transformer 的语言模型中动态分配计算资源。基于 Transformer 的语言模型在输入序列中均匀分布 FLOPs。在这项工作中，我们...</li><li><a href="https://fxtwitter.com/JiaChenyan/status/1732898372359799159">Tweet from Chenyan Jia (@JiaChenyan)</a>: 我们能否设计 AI 系统，将民主价值观作为其目标函数？我们与 @michelle123lam, Minh Chau Mai, @jeffhancock, @msbernst 合作的新 #CSCW24 论文介绍了一种方法，用于转化...</li><li><a href="https://arxiv.org/abs/1907.07174">Natural Adversarial Examples</a>: 我们介绍了两个极具挑战性的数据集，它们能可靠地导致机器学习模型性能大幅下降。这些数据集是通过一种简单的对抗性过滤技术收集的，旨在创建...</li><li><a href="https://arxiv.org/abs/2110.03111">Cut the CARP: Fishing for zero-shot story evaluation</a>: 大规模语言模型（Raffel 等，2019；Brown 等，2020）的最新进展在机器驱动的文本生成方面带来了显著的定性和定量提升。尽管如此...</li><li><a href="https://arxiv.org/abs/2210.07792">Robust Preference Learning for Storytelling via Contrastive Reinforcement Learning</a>: 受控自动故事生成旨在生成满足自然语言批评或偏好约束的自然语言故事。现有的控制故事偏好的方法...</li><li><a href="https://github.com/hsiehjackson/RULER">GitHub - hsiehjackson/RULER: This repo contains the source code for RULER: What’s the Real Context Size of Your Long-Context Language Models?</a>: 此仓库包含 RULER: What’s the Real Context Size of Your Long-Context Language Models? 的源代码 - hsiehjackson/RULER
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1227556417502969917)** (4 条消息): 

- **Scaling Laws 与知识存储**：一篇发布在 [arXiv](https://arxiv.org/abs/2404.05405) 上的新论文认为，语言模型每个参数最多可以存储 **2 bits 的知识**。该论文还探讨了训练时长、模型架构、量化、MoE 等稀疏性约束的使用以及数据信噪比等因素如何影响模型的知识容量。
  
- **解析新 Scaling Law 研究的复杂性**：Eleuther 社区的讨论表明，上述关于 Scaling Laws 和知识位（knowledge bits）的论文难以解读。社区成员正考虑创建一个讨论空间来剖析该论文的研究结果。

- **寻求 OpenAI 最新模型的基准测试**：一位成员询问了关于 **OpenAI 新版 gpt-4-turbo** 的基准测试，并正在寻找这些性能结果可能发布的地方。

**Link mentioned**: <a href="https://arxiv.org/abs/2404.05405">Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws</a>：缩放定律（Scaling laws）描述了语言模型的大小与其能力之间的关系。与以往通过 Loss 或基准测试评估模型能力的研究不同，我们估算了 n...

---

**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1228056988588572672)** (2 messages): 

- **缺乏实质内容的简短互动**：频道内的聊天包括一名成员对某项未指明的要求表示沮丧，随后另一名成员表示赞同。未提供讨论的背景或主题。

---

**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1227633571619934228)** (2 messages): 

- **Chat Templating PR 的进展**：目前正在讨论两个 Pull Requests (PRs)——一个是 Hailey 提交的关于为 [HF 模型添加 chat templating](https://github.com/EleutherAI/lm-evaluation-harness/pull/1287#issuecomment-1967469808) 的 PR，另一个是未指明的 [PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/1578)。建议阅读这些 PR 以跟进项目进度。
- **批量（Batchwise）`apply_chat_template` 贡献的机会**：一位成员指出，**transformers** 库中的 `apply_chat_template` 不支持批量操作。贡献此功能将极大地惠及该项目及其他项目。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1287#issuecomment-1967469808)">[WIP] Add chat templating for HF models by haileyschoelkopf · Pull Request #1287 · EleutherAI/lm-evaluation-harness</a>：这是一个正在进行中（WIP）的 PR，延续了 @daniel-furman 在 #1209 中开始的草案，旨在添加指定的、经常被要求的 chat templating 功能。当前的待办事项包括：使用 OpenHermes 等检查性能...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1578).">Build software better, together</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。
</li>
</ul>

</div>

---

**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1227611511426646117)** (103 messages🔥🔥): 

- **寻求底层代码之外的项目**：一位拥有 20 年经验的开发者表示有兴趣为 **Modular 项目**做出贡献，但希望避免底层的 C 风格编码。讨论引导他们关注 **Web 开发项目 Lightbug** 和机器学习项目 **Basalt**，并提供了 Basalt 在 GitHub 上的仓库链接。

- **Trait 方法中的省略号（Ellipsis）争议**：关于在 **Mojo** 中未实现的 Trait 方法中使用省略号（`...`）进行了深入讨论。有人提议在 GitHub 讨论中弃用省略号，成员们就其 Pythonic 特性及潜在替代方案（包括作为 `os.abort` 别名的 `not_implemented`）展开了辩论。

- **开源工作的社区参与**：一位成员提到 **Mojo 开源了其标准库**，这引发了另一位成员的讽刺反驳，但从 Nightly 版本中合并的 PR 来看，社区对标准库的公开贡献是显而易见的。

- **与 BackdropBuild 合作的邀请**：来自 **BackdropBuild** 的代表联系并讨论了未来与 Modular 的合作，旨在支持 AI、加密货币、游戏和开发工具领域的构建者。该组织运行大型队列计划（cohort programs），并与知名公司合作，以促进各种技术平台上的开发。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://realpython.com/python-ellipsis/">When Do You Use an Ellipsis in Python? – Real Python</a>：你什么时候在 Python 中使用省略号？ – Real Python。你可能在 Python 脚本中见过三个点。虽然这种语法看起来很奇怪，但使用省略号是有效的 Python 代码。在本教程中，你将学习 Python 的 Ellipsis 常量何时可以...</li><li><a href="https://peps.python.org/pep-0544/">PEP 544 – Protocols: Structural subtyping (static duck typing) | peps.python.org</a>：未找到描述</li><li><a href="https://docs.rs/serde/latest/src/serde/de/mod.rs.html#908-1233">mod.rs - source</a>：未找到描述</li><li><a href="https://github.com/modularml/mojo/discussions/2259">[Proposal]: Deprecate Triple Dots (...) for Unimplemented Methods · modularml/mojo · Discussion #2259</a>：动机：Mojo 渴望成为 Python++ 的无缝继任者，紧密遵循 Pythonic 原则，并为 Python 社区培养积极的体验。目前的做法是使用 ...</li><li><a href="https://github.com/basalt-org/basalt">GitHub - basalt-org/basalt: A Machine Learning framework from scratch in Pure Mojo 🔥</a>：一个用纯 Mojo 🔥 从零开始构建的 Machine Learning 框架 - basalt-org/basalt</li><li><a href="https://github.com/mojicians/awesome-mojo">GitHub - mojicians/awesome-mojo: A curated list of awesome Mojo 🔥 frameworks, libraries, software and resources</a>：一个精选的优秀 Mojo 🔥 框架、库、软件和资源列表 - mojicians/awesome-mojo</li><li><a href="https://peps.python.org/">PEP 0 – Index of Python Enhancement Proposals (PEPs) | peps.python.org</a>：未找到描述</li><li><a href="https://backdropbuild.com/">Backdrop Build</a>：我们共同构建 - 在短短 4 周内与数百名其他出色的构建者一起将那个疯狂的想法变为现实。
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1227684149150613547)** (2 条消息): 

- **来自 Modular 的推文公告**：来自 Modular 官方 Twitter 账号的推文已在 Discord 频道中分享。可以直接在 Twitter 上查看：[Modular Tweet](https://twitter.com/Modular/status/1778118673976402286)。
- **分享后续推文**：讨论中出现了另一条来自 Modular 的推文。要查看此推文的内容，请访问：[Modular's Latest Tweet](https://twitter.com/Modular/status/1778482233957101869)。
  

---


**Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1228017366504312963)** (1 条消息): 

- **内存中的行优先（Row-Major）与列优先（Column-Major）**：一篇 [博客文章](https://www.modular.com/blog/row-major-vs-column-major-matrices-a-performance-analysis-in-mojo-and-numpy) 探讨了矩阵在内存中的存储差异，详细说明了行优先顺序如何有利于连续内存中的行向量，而列优先顺序如何有利于列向量。该分析旨在阐明这些存储策略对 Mojo 🔥 和 NumPy 性能的影响。
- **存储顺序之间的性能对决**：该文章旨在探讨为什么某些编程语言和库选择列优先顺序，而其他语言则偏好行优先，以及这些选择会带来哪些性能后果。它强调了读取连续内存位置的速度优势，并指出所选的存储顺序可能会显著影响性能。

**提到的链接**：<a href="https://www.modular.com/blog/row-major-vs-column-major-matrices-a-performance-analysis-in-mojo-and-numpy">Modular: Row-major vs. column-major matrices: a performance analysis in Mojo and NumPy</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：Mojo 和 NumPy 中的行优先 vs. 列优先矩阵性能分析。

  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1227519673394790400)** (58 条消息 🔥🔥):

- **Mojo 改进指日可待**：一份[路线图文档](https://docs.modular.com/mojo/roadmap#lifetime-tracking-inside-collections)提供了关于 **Mojo 未来计划**中核心编程特性的见解，并确认了将要添加的主要组件，重点是优先构建 Mojo 的核心语言特性以实现长期可持续性。
- **Mac 生态系统中 Mojo 的 UI 库开发**：讨论集中在一个[新的 Mojo 跨平台 UI 库](https://github.com/Moosems/Mojo-UI)，暂定名为 "Mojective-C"，并建议了通过 C 或 C++ 绑定将 Objective-C 或 AppKit 与 Mojo 集成的可能方法。
- **Mojo 对上下文管理器和 `with` 块的处理**：关于 Mojo 中 GUI 框架设计以及使用 `with` 块的可能替代方案正在进行讨论，由于 `with` 块限制了开发者对组件（widgets）的控制，通常被视为负面因素。
- **Mojo 中的 Segfaults 引发疑问**：GitHub 上的一项 Segfault 问题被过早关闭后又重新开启，这引发了对政策的澄清：一旦问题在内部修复，就会被标记为已关闭，但可能尚未反映在 Nightly/Stable 版本中。
- **位运算以及将 C 代码转换为 Mojo**：关于将位运算从 C 翻译到 Mojo 的交流产生了一些共享代码片段。一些用户在 Mojo 中实现随机数生成的背景下，对数据类型的使用提供了纠正和建议。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/jeff-channing-tatum-22jump-street-disguise-gif-8025876">My Name Is Jeff Jeff GIF - Jeff Channing Tatum 22Jump Street - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://en.wikipedia.org/wiki/Xorshift#xorshift.2A">Xorshift - Wikipedia</a>：未找到描述</li><li><a href="https://docs.modular.com/mojo/roadmap#lifetime-tracking-inside-collections">Mojo🔥 路线图与尖锐边缘 | Modular 文档</a>：关于我们 Mojo 计划的摘要，包括即将推出的特性和我们需要修复的问题。</li><li><a href="https://github.com/Moosems/Mojo-UI">GitHub - Moosems/Mojo-UI: 一个 Mojo 的跨平台 GUI 库</a>：一个 Mojo 的跨平台 GUI 库。通过在 GitHub 上创建账号来为 Moosems/Mojo-UI 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/issues/28">为什么不选 Mojo？ · Issue #28 · karpathy/llm.c</a>：这是一个严肃的问题。如果你要深入底层，Mojo 提供了潜在的巨大加速，而且该语言将从这项工作中显著受益。无论如何——热爱这项工作。谢谢...</li><li><a href="https://github.com/modularml/mojo/issues/2208#issuecomment-2046614359">[BUG][回归] 带有递归引用的结构体定义（不再给出错误消息并）崩溃 · Issue #2208 · modularml/mojo</a>：Bug 描述，我相信 issue #74 重新出现了。以下代码将导致 Segfault：#crash.mojo struct Node: var rec: Node fn __init__(inout self): pass 但我预期的错误消息是：$🍔 m.....</li><li><a href="https://github.com/modularml/mojo/issues/2208#issuecomm">[BUG][回归] 带有递归引用的结构体定义（不再给出错误消息并）崩溃 · Issue #2208 · modularml/mojo</a>：Bug 描述，我相信 issue #74 重新出现了。以下代码将导致 Segfault：#crash.mojo struct Node: var rec: Node fn __init__(inout self): pass 但我预期的错误消息是：$🍔 m.....
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1227574677937983580)** (5 条消息): 

- **Mojo 获得迭代功能**：分享了一个用于迭代字符串字符的代码片段，对于寻找类似解决方案的人可能有用。实际的迭代器代码可在提供的 [Discord 链接](https://discord.com/channels/1087530497313357884/1227272073840431116/1227495619501752330) 中找到。

- **键盘事件引入 mojo-ui-html**：`mojo-ui-html` 的更新引入了键盘事件、窗口最小化功能以及逐元素的 CSS 样式改进，这对游戏和自定义组件开发者以及 `neovim` 用户非常有利。这些新方法旨在增强交互体验，可以通过其 [GitHub 上的 Demo](https://github.com/rd4com/mojo-ui-html/blob/main/demo_keyboard_and_css.mojo) 查看实际效果。

- **Lightbug 框架庆祝社区贡献**：`Lightbug` HTTP 框架收到了多项社区贡献，包括新增获取远程地址/对端名称的功能、更高效的字符串构建方法、基于 Mojo 的客户端实现，以及性能分析（结果显示 Lightbug 在请求处理能力上优于 Python Flask）。新进展还包括一个名为 `lightbug_api` 的高级 API 框架，其灵感源自 Django，所有这些内容都可以在 [GitHub](https://github.com/saviorand/lightbug_http) 上查看。

- **使用 Mojo 提升终端文本渲染**：展示了使用 Mojo 在终端渲染文本的预览，演示了在该生态系统中可以构建的工具链，其灵感来自 `lipgloss` 等 Go 语言包。这一精美的终端 UI 显示背后的代码可在其 [GitHub 仓库](https://github.com/thatstoasty/mog/blob/main/examples/readme/layout.mojo) 中查看。

- **Basalt 展示 Mojo 的视觉魅力**：在更新中，一位社区成员赞扬了在最近分享的 Mojo 渲染文本示例中使用的 Basalt（一种终端样式预设），强调了 Mojo 在增强终端应用程序方面的视觉表现力。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/thatstoasty/mog/blob/main/examples/readme/layout.mojo">mog/examples/readme/layout.mojo at main · thatstoasty/mog</a>：通过在 GitHub 上创建账户来为 thatstoasty/mog 的开发做出贡献。</li><li><a href="https://github.com/rd4com/mojo-ui-html/blob/main/demo_keyboard_and_css.mojo">mojo-ui-html/demo_keyboard_and_css.mojo at main · rd4com/mojo-ui-html</a>：立即模式 GUI、HTML、CSS，开发中，Mojo 语言 - rd4com/mojo-ui-html</li><li><a href="https://github.com/saviorand/lightbug_http/issues/6).">Issues · saviorand/lightbug_http</a>：适用于 Mojo 的简单且快速的 HTTP 框架！🔥。通过在 GitHub 上创建账户来为 saviorand/lightbug_http 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[📰︱newsletter](https://discord.com/channels/1087530497313357884/1157711273111339088/)** (1 条消息): 

Zapier: Modverse Weekly - 第 29 期
https://www.modular.com/newsletters/modverse-weekly-29
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1227523242445635664)** (131 条消息🔥🔥): 

- **量化的困惑**：成员们讨论了像 **Mistral** 这样的模型在 24GB 显卡上配合 16k 上下文进行量化的潜力；不过，一些人认为量化后的 **Mixtral 8x7b** 性能表现非常出色。
- **MLLM 信息寻求者**：出现了关于多模态大模型 (MLLMs) 的咨询，推荐了 **LLaVA** 和 **Qwen VLM**，尽管承认它们的许可证有限制，且使用 axolotl 微调 LMMs 的可用指导较少。
- **推理服务器的思考**：在处理缺乏 NVLink 和 PCIe 5 等限制的情况下，展开了一场关于使用 **Nvidia 4090s** 构建推理服务器的辩论。有人提出 **Nvidia 3090s** 在卡间带宽方面可能是更合适的替代方案。
- **黑客松预告**：宣传了将于 5 月 11 日在纽约举行的 **Samsung Next 2024 Generative AI Hackathon**，重点领域包括健康与福祉以及媒体技术。
- **为 Axolotl 贡献并分享见解**：一位成员表达了为 **Axolotl** 做出贡献的愿望，得到的回复强调了复现现有问题、协助文档编写的价值，并指出编程经验可能不如投入时间重要。另一个关键讨论点涉及有监督微调 (SFT) 与 **动态规划优化 (DPO)** 之间的区别，强调在某些语境下，**DPO** 相比 **SFT** 能更独特地引导生成的响应。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.05405">Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws</a>：缩放法则描述了语言模型规模与其能力之间的关系。与以往通过 Loss 或基准测试评估模型能力的研究不同，我们估算了 n...</li><li><a href="https://lu.ma/nextgenainyc">Samsung Next 2024 Generative AI Hackathon · Luma</a>：🚀 活动预告：申请参加 Samsung Next 2024 生成式 AI 黑客松！我们将探索两个赛道：健康与福祉：利用 AI 的力量改善医疗成果...</li><li><a href="https://huggingface.co/MaziyarPanahi/Mixtral-8x22B-v0.1-GGUF/tree/main">MaziyarPanahi/Mixtral-8x22B-v0.1-GGUF at main</a>：暂无描述
</li>
</ul>

</div>
  

---

**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1227796708411441224)** (5 messages): 

- **Axolotl 文档发布**：一名成员分享了他们的长期项目：[Axolotl 文档](https://axolotl.continuumlabs.pro)，邀请社区提供反馈，并承认目前仍有一些空白需要填补。
- **鼓励反馈循环**：在赞赏这一努力的同时，一名成员建议将文档相关的讨论集中在单个频道中，并指出当前的文档草案中既有宝贵的建议也有一些错误。
- **贡献致谢**：另一位参与者对编写 Axolotl 文档的这一倡议表示了感谢。

**提到的链接**：<a href="https://axolotl.continuumlabs.pro">Introduction | Continuum Training Platform | Axolotl Training Platform</a>：未找到描述

---

**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1227675113735061636)** (3 messages): 

- **处理空队列**：为了避免空队列导致的错误，一名成员建议在迭代前检查队列状态。提供的代码片段通过使用 `if not streamer.empty()` 包裹 `for` 循环来修正迭代。
- **重构 Stop Token 检查**：分享了一个简单的重构技巧以优化停止令牌函数。代码 `return input_ids[0][-1] in stop_ids` 高效地替代了用于检查停止条件的循环。
- **关于模型合并时 GPU 利用率的问题**：一名成员询问在合并模型时利用 GPU 资源以提升性能的可能性。

---

**OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1227983231698337852)** (2 messages): 

- **寻找逻辑推理数据集**：一名成员询问是否存在处理自然文本上的**命题逻辑和谓词逻辑推理**的数据集，但未提及他们已找到的任何数据集。
- **寻找海量训练数据**：另一名成员请求推荐一个大约 **200 billion tokens** 的数据集，适用于实验新架构；在随后的讨论中没有建议任何数据集。

---

**OpenAccess AI Collective (axolotl) ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/1227796935021432892)** (1 messages): 

- **Axolotl 文档发布并征求反馈**：期待已久的 [Axolotl 文档](https://axolotl.continuumlabs.pro) 已分享给社区以获取反馈。需要注意的是，文档可能仍有空白，作者鼓励大家提供反馈以进行进一步开发。

**提到的链接**：<a href="https://axolotl.continuumlabs.pro">Introduction | Continuum Training Platform | Axolotl Training Platform</a>：未找到描述

---

**OpenAccess AI Collective (axolotl) ▷ #[minotaur](https://discord.com/channels/1104757954588196865/1116465236715786310/1227797012003557417)** (1 messages): 

- **Axolotl 文档发布并征求反馈**：期待已久的 **Axolotl 文档** 已与社区分享并开放反馈。该文档目前仍在完善中（work in progress），存在一些空白，可以通过 [Axolotl Documentation](https://axolotl.continuumlabs.pro) 访问，并鼓励通过反馈来完善内容。

**提到的链接**：<a href="https://axolotl.continuumlabs.pro">Introduction | Continuum Training Platform | Axolotl Training Platform</a>：未找到描述

---

**OpenAccess AI Collective (axolotl) ▷ #[bots](https://discord.com/channels/1104757954588196865/1117282691121954836/1227796957389651988)** (1 messages): 

- **Axolotl 文档发布并征求反馈**：新的 [Axolotl 文档](https://axolotl.continuumlabs.pro) 已与社区分享。作者已获得所有必要的批准，并对进一步开发的反馈持开放态度。

**提到的链接**：<a href="https://axolotl.continuumlabs.pro">Introduction | Continuum Training Platform | Axolotl Training Platform</a>：未找到描述

---

**OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1227953974930440212)** (3 messages): 

- **Axolotl 初步尝试**：分享了一篇新博客文章，详细介绍了使用 Axolotl 对较小的 encoder 风格 LLM 进行分类任务 **fine-tuning** 的经验，以及对用于文本生成的 decoder 风格 LLM 的初步探索。该资源对于 LLM 新手来说是一个有用的指南，并附带了一篇[博客文章](https://drchrislevy.github.io/posts/intro_fine_tune/intro_fine_tune.html)，其中包含了对 GPT 和 Mistral 等模型的引用。

- **社区成员推荐**：上述 **Axolotl 入门指南** 被推荐为那些希望了解使用 Axolotl 基础知识的人的良好起点。

- **调试技巧分享**：对于从事数据预处理的人员，在 preprocess 期间使用 `--debug` 标志可以帮助验证数据记录的正确性。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/cleavey1985/status/1778393547571384410">来自 Chris Levy (@cleavey1985) 的推文</a>：完成了一篇关于首次使用 @axolotl_ai 的博文。https://drchrislevy.github.io/posts/intro_fine_tune/intro_fine_tune.html 感谢 @jeremyphoward 推荐该工具，@HamelHusain f...</li><li><a href="https://drchrislevy.github.io/posts/intro_fine_tune/intro_fine_tune.html">Chris Levy - 使用 Axolotl 进行 LLM 微调入门</a>：未找到描述
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1227772768972701726)** (5 messages): 

- **多 GPU 设置下的 QLoRA 训练之谜**：一位成员询问，为什么在单个 4090 GPU 上消耗 98% VRAM 的 **QLoRA 训练过程**，在扩展到 4x4090 GPU 时会失败，尽管使用了 *DeepSpeed Zero Stage 3*。额外的内存开销、内存分配模式的变化以及增加的通信开销可能是导致此次失败的原因。
- **ZeRO-3 的优化策略**：为了解决多 GPU 训练问题，建议尝试 **超参数调整**、优化 **DeepSpeed 配置**、使用 *nvidia-smi* 进行监控以及探索 **CPU offloading**。调整 batch sizes 和 gradient accumulation steps 有助于平衡各 GPU 之间的内存负载。
- **DeepSpeed 配置调优**：建议仔细检查并调整 **DeepSpeed 配置**，这可以显著影响内存使用和训练性能。配置可以包括为 optimizer 和 parameter states 启用 CPU offloading，以减轻 GPU 内存使用。

**提到的链接**：<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=eddbbf5d-0725-40a1-9baf-c02e328d4a61)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>：更快地理解代码。

  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1227661649436217364)** (10 messages🔥): 

- **Mixtral 登陆 OpenRouter**：新的基础模型 [Mixtral 8x22B](https://openrouter.ai/models/mistralai/mixtral-8x22b) 已在 OpenRouter 上可用，尽管未经 instruct tuned，但在使用 instruct 模板时表现稳健。

- **Gemma 获得升级**：OpenRouter 更新了其产品，将 [Google: Gemma 7B](https://openrouter.ai/models/google/gemma-7b-it) 替换为较新的 [google/gemma-1.1-7b-it](https://huggingface.co/google/gemma-1.1-7b-it)。

- **OpenRouter 全面降价**：该平台宣布降低模型价格，包括 [LZLV 70B](https://openrouter.ai/models/lizpreciatior/lzlv-70b-fp16-hf)、[Databricks: DBRX 132B Instruct](https://openrouter.ai/models/databricks/dbrx-instruct) 和 [Nous: Hermes 2 Mixtral 8x7B DPO](https://openrouter.ai/models/nousresearch/nous-hermes-2-mixtral-8x7b-dpo)。

- **限时优惠：免费使用 Mixtral 8x22B**：鼓励用户在限定时间内免费试用 [Mixtral 8x22B](https://openrouter.ai/models/mistralai/mixtral-8x22b)。

- **用户反馈影响 Gemma 的可用性**：针对用户关于 Gemma 2B 可用性的询问，OpenRouter 回应称需求不大，并指出 7B 版本对于快速任务已经是免费的，并建议 2B 可能更适合在本地运行。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/mistralai/mixtral-8x22b>)">Mixtral 8x22B by mistralai | OpenRouter</a>: Mixtral 8x22B 是来自 Mistral AI 的大规模语言模型。它由 8 个专家组成，每个专家拥有 220 亿参数，每个 token 每次使用 2 个专家。它通过 [X](https://twitter...</li><li><a href="https://openrouter.ai/models/mistralai/mixtral-8x22b)">Mixtral 8x22B by mistralai | OpenRouter</a>: Mixtral 8x22B 是来自 Mistral AI 的大规模语言模型。它由 8 个专家组成，每个专家拥有 220 亿参数，每个 token 每次使用 2 个专家。它通过 [X](https://twitter...</li><li><a href="https://openrouter.ai/models/mistralai/mixtral-8x22b:free>).">Mixtral 8x22B by mistralai | OpenRouter</a>: Mixtral 8x22B 是来自 Mistral AI 的大规模语言模型。它由 8 个专家组成，每个专家拥有 220 亿参数，每个 token 每次使用 2 个专家。它通过 [X](https://twitter...</li><li><a href="https://openrouter.ai/models/google/gemma-7b-it>)">Gemma 7B by google | OpenRouter</a>: Google 的 Gemma 是一个先进的开源语言模型系列，利用了最新的 decoder-only 文本到文本技术。它在文本生成任务中提供英语能力...</li><li><a href="https://openrouter.ai/models/lizpreciatior/lzlv-70b-fp16-hf>)">lzlv 70B by lizpreciatior | OpenRouter</a>: 选定 70B 模型的 Mythomax/MLewd_13B 风格合并。这是几个 LLaMA2 70B 微调模型的多模型合并，用于角色扮演和创意工作。目标是创建一个结合了创造力的模型...</li><li><a href="https://openrouter.ai/models/databricks/dbrx-instruct>)">DBRX 132B Instruct by databricks | OpenRouter</a>: DBRX 是由 Databricks 开发的新型开源大语言模型。在 132B 规模下，它在语言的标准行业基准测试中优于现有的开源 LLM，如 Llama 2 70B 和 Mixtral-8x7B...</li><li><a href="https://openrouter.ai/models/nousresearch/nous-hermes-2-mixtral-8x7b-dpo>)">Hermes 2 Mixtral 8x7B DPO by nousresearch | OpenRouter</a>: Nous Hermes 2 Mixtral 8x7B DPO 是新的 Nous Research 旗舰模型，基于 [Mixtral 8x7B MoE LLM](/models/mistralai/mixtral-8x7b) 训练。该模型在超过 1,000,000 条原始数据上进行了训练...
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1227538189950910546)** (136 messages🔥🔥): 

- **模型速率限制说明**：一位成员澄清，OpenRouter 上的“严重速率限制”通常意味着模型的速率限制与免费模型相似：每分钟 10 个请求。

- **Gemini 1.5：在 OR 上付费而非免费**：有人指出 **Gemini 1.5** 在 OpenRouter 上是付费的，目前没有免费层级。

- **Gemma 速率限制问题**：用户讨论了 Gemma 模型未如预期免费的问题、“Updated”标签未反映最新更改的问题，以及关于速率限制和 token 计数的困惑，这导致了一个修复方案的部署。

- **Gemini Token 定价解释**：关于 Gemini 模型的定价有了澄清，指出 OR 将 Gemini 的 token 计为单个字符，这反映在较高的“context”成本中。该系数在采样期间进行估算，但计费时将每个字符计为一个 token。

- **Mixtral 8x22b 讨论**：用户分享了他们使用 **Mixtral 8x22b** 的经验，注意到其良好的推理能力，并考虑了其与 GPT-4 相比的性价比。共识是，尽管不像 GPT-4 那样文采斐然，但它提供了连贯且出人意料的优秀输出。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/cohere/command-r-plus">Command R+ by cohere | OpenRouter</a>: Command R+ 是来自 Cohere 的新型 104B 参数 LLM。它适用于角色扮演、通用消费者用例和检索增强生成 (RAG)。它为十种关键语言提供多语言支持...</li><li><a href="https://openrouter.ai/models/google/gemma-7b-it:free">Gemma 7B by google | OpenRouter</a>: Google 的 Gemma 是一个先进的开源语言模型系列，利用了最新的 decoder-only 文本到文本技术。它在文本生成任务中提供英语能力...</li><li><a href="https://openrouter.ai/playground?models=openai/gpt-4-turbo,mistralai/mistral-large">OpenRouter</a>: LLM 和其他 AI 模型的路由器</li><li><a href="https://docs.librechat.ai/install/index.html">Installation and Configuration</a>: 💻 关于安装与配置的深入指南</li><li><a href="https://github.com/danny-avila/LibreChat">GitHub - danny-avila/LibreChat: 增强版 ChatGPT 克隆：具有 OpenAI, Assistants API, Azure, Groq, GPT-4 Vision, Mistral, Bing, Anthropic, OpenRouter, Google Gemini, AI 模型切换, 消息搜索, langchain, DALL-E-3, ChatGPT Plugins, OpenAI Functions, 安全多用户系统, 预设, 完全开源可自托管。更多功能开发中</a>: 增强版 ChatGPT 克隆：具有 OpenAI, Assistants API, Azure, Groq, GPT-4 Vision, Mistral, Bing, Anthropic, OpenRouter, Google Gemini, AI 模型切换, 消息搜索, langchain, DALL-E-3, Cha...</li><li><a href="https://discord.gg/uDyZ5Tzhct">Join the LibreChat Discord Server!</a>: LibreChat 社区，一个开源、多功能的 AI 聊天 Web UI，支持无缝自托管和活跃开发。| 3349 名成员
</li>
</ul>

</div>
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1227543783671861288)** (97 messages🔥🔥): 

- **对 Command r+ 的兴奋**: 成员们开始尝试名为 **Command r+** 的模型，该模型以其出色的**指令遵循能力**而著称。用户报告称，在某些基准测试和角色扮演场景中，它的表现优于 **GPT-3.5** 和 **Claude 3** 等其他模型，甚至在某些方面认为它接近 **GPT-4** 的水平。

- **Open-Interpreter 设置的技术支持**: 在安装 **open-interpreter** 时出现了技术问题，在尝试截屏后崩溃并出现依赖冲突。建议使用 **git+https** 安装命令进行修复，并将 **OPENAI_API_KEY** 设置为 API 访问的环境变量。

- **对 Mixtral 和 OI 集成的期待**: 讨论表明成员希望 **Mixtral 8x22b** 能与 **OI** 有效配合，并提到 8x7b 版本在与 OI 配合使用时未达到预期。

- **Open AI 预付额度促销**: 分享了关于 **OpenAI** 转向预付额度并停止每月计费的更新。成员们获悉了有效期至 2024 年 4 月 24 日的促销额度优惠。

- **Open-Interpreter 代码库会议的邀请与反思**: 一位成员分享了最近关于在 Python 项目中将 **Open Interpreter 作为库**使用的会议信息，包括指向带有入门模板的 GitHub 仓库链接。另一场会议已安排，并征求社区希望探索代码库哪些部分的意见。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lu.ma/novus28">Novus #28 · Luma</a>: Novus 是一个供初创公司创始人、建设者和创意人士聚集、共同工作和演示的社区。没有废话。没有推销。只有构建。议程 12:00 PM - 12:15 PM - 更新与...</li><li><a href="https://discord.gg/open-interpreter-1146610656779440188?event=1228084898993143920">Join the Open Interpreter Discord Server!</a>: 使用计算机的新方式 | 8227 名成员</li><li><a href="https://www.youtube.com/watch?v=rOW8OK7qXcM">oi house party 5 apr</a>: 未找到描述</li><li><a href="https://github.com/MikeBirdTech/open-interpreter-python-templates">GitHub - MikeBirdTech/open-interpreter-python-templates</a>: 通过在 GitHub 上创建账户来为 MikeBirdTech/open-interpreter-python-templates 的开发做出贡献。
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1227599141979553836)** (30 messages🔥): 

- **烦人的 ngrok 问题**: 一位成员遇到了 `ngrok` 无法绑定到其指定域名的问题，并报错显示了不同的域名。尽管重置了两次 token，问题仍然存在，这暗示了 **ngrok setup** 中潜在的配置异常。

- **通过电子邮件更新订单状态**：Mike.bird 向 **@liamthememelord** 保证，关于目前“仍在准备中（still cooking）”的订单状态更新，将通过电子邮件通知告知客户。
  
- **对极致透明度的请求**：用户 **8i8__papillon__8i8d1tyr** 戏谑地请求提供零件的完整来源，以及参与生产过程的每位员工的简短传记（包括他们的精神动物）。
  
- **安装问题与 Poetry 难题**：成员们讨论了各种安装障碍，特别是关于在 Windows PowerShell 和 MacOS 等不同平台上正确使用 `poetry` 和 `pip` 的问题，其中一名用户卡在了 `poetry` 的“command not found”错误上。
  
- **分享有用的设置视频**：成员 **tsmith.tx** 提供了一个视频演示，介绍如何设置 **01 Light** 并将其连接到本地服务器，这有助于解决在设置过程中遇到的困难。


**提到的链接**：<a href="https://youtu.be/Y76zed8nEE8">01 Light Setup - Flash and Connect to Server</a>：这段简短的视频展示了如何刷写 01 Light，并在运行本地 01OS 及 OpenAI 模型的服务器上进行设置。我正按照 h ttps:... 的说明进行操作。

  

---


**OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1227979796085084240)** (1 条消息): 

- **Transformers.js 将机器学习带入浏览器**：提到了一个名为 [transformers.js](https://github.com/xenova/transformers.js) 的新项目，它是 **HuggingFace Transformers library** 的 JavaScript 移植版本。这使得最先进的机器学习模型可以直接在浏览器中运行，无需服务器。

**提到的链接**：<a href="https://github.com/xenova/transformers.js">GitHub - xenova/transformers.js: State-of-the-art Machine Learning for the web. Run 🤗 Transformers directly in your browser, with no need for a server!</a>：Web 端最先进的机器学习。直接在浏览器中运行 🤗 Transformers，无需服务器！- xenova/transformers.js

  

---



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1227525694892806186)** (67 条消息 🔥🔥): 

- **对 Prompt 管理系统的好奇**：一位成员询问了使用 [vellum.ai](https://vellum.ai) 等 Prompt 管理系统在不同模型间创建、测试和部署 Prompt 的经验。
- **关于 AI 语音选项实用性的讨论**：在 AI 语音偏好讨论中，*Sky* 和 *Juniper* 被提及为受青睐的文本转语音（TTS）声音，但对于 ChatGPT 或 Mistral 等聊天模型是否需要“真实”语音尚未达成共识。
- **为推理选择合适的 AI**：成员们辩论了各种 AI 模型的推理能力，一些人支持 **Claude 3 Opus**，而另一些人则对任何 AI 模型是否具备真正的推理能力持怀疑态度。
- **Perplexity 的付费模式与验证工作**：关于 Perplexity 的讨论涉及一名成员对付费测试（paid beta testing）作为普遍做法的担忧，以及用户为确保 AI 服务引用的来源准确而必须进行的额外验证工作。
- **OpenAI Assistant API 的技术援助**：一名用户寻求关于 OpenAI Assistant API 代码的帮助，其中 'Beta' 对象没有 'messages' 属性，引发了关于文档过时以及涉及更新 beta 版本 client threads 方法调用的讨论。



**提到的链接**：<a href="https://status.openai.com/">OpenAI Status</a>：未找到描述

  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1227571243797184522)** (29 条消息 🔥): 

- **GPT-3.5/4 的押韵挑战**：一位成员询问如何让 GPT-3.5 或 GPT-4 生成押韵诗句。
- **账单忧虑与 API 困境**：一名用户遇到了账单问题，尽管余额充足，但其 API 访问权限因支付失败而被暂停。他们请求管理员协助处理此情况。
- **OpenAI 变成空城了吗？**：用户报告称 OpenAI 的 GPT 延迟很高，有人提到该问题导致响应速度变慢。
- **Chat GPT-4 失忆还是停机？**：几名用户遇到了 GPT-4 找不到现有对话的问题，并猜测可能存在服务停机。
- **聊天机器人的过山车**：关于 Chat GPT 停机和恢复的报告席卷了讨论，伴随着对可靠性的担忧，以及要求在停机期间保持服务免费的呼声。一名用户还引用了 [OpenAI 状态页面信息](https://status.openai.com/) 中关于近期事件和解决方案的内容。

**提到的链接**：<a href="https://status.openai.com/">OpenAI Status</a>：未找到描述

  

---

**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1227772580451192892)** (14 messages🔥): 

- **GPT-4 Turbo 的创意优势**：成员们观察到 **GPT-4-turbo-2024-04-09** 的改进，指出与之前的版本相比，它显得更有“创意且生动”，尽管它仍然有压缩代码的倾向。提到了避免使用代码块等特定策略来应对这种压缩问题。
- **应对代码压缩的策略**：成员们分享了管理 GPT 模型不理想的代码压缩问题的技巧，包括提供 [custom instructions](http://drinkoblog.weebly.com) 或使用提示词链（prompt chains）来逐步构建所需的代码输出。
- **用于更精确编码的提示词链**：一位成员建议将请求分解为约 200 行代码的小型、可管理部分，以解决修订问题并防止模型提供不完整的代码。
- **轻松访问 Wolfram GPT**：如果你需要在 GPT 中使用 Wolfram，建议使用可以通过聊天中分享的链接访问的 **Wolfram GPT model**。激活后，可以在任何对话中使用 `@mention` 功能召唤 Wolfram GPT。
- **为初级提示词工程师提供的资源**：一位寻求提示词工程资源的成员被引导至 [promptingguide.ai](http://promptingguide.ai) 以获取该主题的全面信息。
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1227772580451192892)** (14 messages🔥): 

- **GPT-4 Turbo 表现出改进**：成员们注意到 **GPT-4-turbo-2024-04-09** 的改进，观察到提示词拒绝情况减少，模型更加“生动且有创意”，同时也承认它仍然倾向于压缩代码。他们建议避免使用代码块来缓解这个问题。
- **解决代码输出中的压缩问题**：用户对模型压缩代码的倾向表示沮丧，但分享了他们的变通方法，包括提供自定义指令（custom instructions）以防止此类压缩。
- **建议采用策略性提示词链**：一位用户推荐了一种提示词工程策略，包括降低预期并使用提示词链来细化代码输出，同时建议将代码模块保持在 200 行以下以避免问题。
- **关于集成 Wolfram GPT 的建议**：一位用户询问如何在 GPT 模型中使用 Wolfram，另一位成员向他们推荐了 [Wolfram GPT](https://chat.openai.com/g/g-0S5FXLyFN-wolfram)，并建议在首次使用后使用 `@mention` 功能。
- **提示词工程入门**：对于提示词工程的新手，提供了一个信息丰富的网站引用：[Prompting Guide](http://promptingguide.ai)。
  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1227632149780234341)** (3 messages): 

- **使用 IFTTT 执行停止点控制你的 Agent**：IFTTT 集成可以增强对 Agent 的控制，例如在预订确认后终止旅游 Agent 的流程，或在回复后结束 *agentic RAG pipeline*。关于这些工具的兴奋点在 [Twitter](https://t.co/ByGOaqgWMd) 上的预热链接中进行了分享。

- **简化的基于 ColBERT 的检索 Agent**：重点介绍了一种构建基于 ColBERT 的检索 Agent 的新方法，该 Agent 能够进行具有对话记忆的高级文档搜索。该方法承诺简单且有效，更多细节在 [Twitter 帖子](https://t.co/wVDpML9juP) 中有所提示。

- **“创建一个与你的代码聊天应用”教程**：由 @helloiamleonie 制作的 @weaviate_io 互动教程展示了如何创建一个允许用户与 GitHub 代码库聊天的应用。该教程利用了本地 LLM 和 embedding model，分步指南在 [Twitter](https://t.co/yZN3QZjgN4) 上进行了预热。
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1227560655058636801)** (89 messages🔥🔥):

- **结构化 LLM 输出的集成咨询**：一位用户咨询了如何将 [Instructor](https://python.useinstructor.com/) 与 LlamaIndex 集成，以便将结构化输出从 API 服务器流式传输到前端。频道内随后没有进一步的讨论或提供的解决方案。
- **本地运行 sec-insights 应用出现问题**：一位用户在尝试本地运行 [sec-insights](https://github.com/run-llama/sec-insights) 应用时遇到错误，无法完成 `make seed_db_local` 命令。他们分享了详细的错误日志，几位用户讨论了该问题但未立即解决，不过有人建议观看[端到端指南视频](https://youtu.be/2O52Tfj79T4?si=CYUcaBkc9P9g_m0P)以寻求帮助。
- **对详细 LLM 调用调试的需求**：用户表达了改进 LlamaIndex 可观测性的需求，特别是希望看到发送给 Large Language Models (LLMs) 的**确切 Prompt**。大家公认 LlamaIndex 的原生 instrumentation 可能对此有所帮助，用户们还讨论了通过单独重建 chat loops 来更好地控制 system prompt 行为。
- **排查 LlamaIndex 的 MongoDB 集成问题**：一位用户遇到了 `TypeError`，提示 `KVDocumentStore.__init__()` 存在问题。在报告错误并分享相关代码后，另一位用户建议通过更新 LlamaIndex 软件包来解决，并提供了相应的命令。
- **关于基于 CPU 的 LLM 服务器框架的讨论**：一位用户询问了可以在 CPU 上运行并支持并行处理多个推理请求的 LLM 服务器框架。讨论指出在 CPU 上运行 LLM 可能效率较低，但建议通过用户自定义的 chat loops 或带有 Ollama 的自动扩缩容 Kubernetes 集群来实现这一目标，尽管目前还没有已知的支持 batch-inferencing 的框架。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.llamaindex]">未找到标题</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1P0RiVeQQF5z09A4KxvWuYGzv2UoJUIsX?usp=sharing">Google Colaboratory</a>：未找到描述</li><li><a href="https://python.useinstructor.com/">Welcome To Instructor - Instructor</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/#custom-node-postprocessor">Node Postprocessor - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_memory/?h=query+pipeline">Query Pipeline Chat Engine - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/index_structs/knowledge_graph/NebulaGraphKGIndexDemo/#query-with-embeddings">Nebula Graph Store - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/sec-insights">GitHub - run-llama/sec-insights: A real world full-stack application using LlamaIndex</a>：一个使用 LlamaIndex 的真实全栈应用 - run-llama/sec-insights</li><li><a href="https://github.com/run-llama/llama_index/blob/2b77f89775840d6b796bcc693f7593d2aebc5fec/llama-index-integrations/llms/llama-index-llms-ollama/llama_index/llms/ollama/base.py#L56">llama_index/llama-index-integrations/llms/llama-index-llms-ollama/llama_index/llms/ollama/base.py at 2b77f89775840d6b796bcc693f7593d2aebc5fec · run-llama/llama_index</a>：LlamaIndex 是适用于您的 LLM 应用的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/rags">GitHub - run-llama/rags: Build ChatGPT over your data, all with natural language</a>：完全使用自然语言基于您的数据构建 ChatGPT - run-llama/rags</li><li><a href="https://youtu.be/2O52Tfj79T4?si=CYUcaBkc9P9g_m0P">Discover LlamaIndex: SEC Insights, End-to-End Guide</a>：secinsights.ai 是一个全栈应用，利用 LlamaIndex 的检索增强生成 (RAG) 能力来回答关于 SEC 10-K 和 10-Q 文档的问题...</li><li><a href="https://github.com/run-llama/llama_index/pull/12736">[BUGFIX] Update LlamaIndex-Predibase Integration by alexsherstinsky · Pull Request #12736 · run-llama/llama_index</a>：描述：Predibase API 已发生变化。此贡献更新了 LlamaIndex 端连接和调用 Predibase LLM 服务的实现。一旦此 pull request...
</li>
</ul>

</div>
  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1227642930043289681)** (68 messages🔥🔥): 

- **copy_from_fd 缺少 CI 测试**：注意到 **tinygrad** 缺少针对 `copy_from_fd` 的持续集成（CI）测试；这是在最近的一次 [GitHub Action job](https://github.com/tinygrad/tinygrad/actions/runs/8633930065/job/23668153464) 中发现的。一位用户已确认该问题，并计划在单独的 pull request 中添加测试。

- **Rust 导出功能提案被拒绝**：一个为 **tinygrad** 提议 Rust 导出功能的 pull request 被[拒绝](https://github.com/tinygrad/tinygrad/pull/4138)了，因为这偏离了项目的重点。负责人强调 **tinygrad** 为了性能和可维护性而坚持使用 C，并建议 Rust 在必要时可以调用编译好的 C 库。
  
- **强调性能而非语言偏好**：在针对从 **tinygrad** 生成 Rust 代码的讨论中，有人强调神经网络在设计上就是内存安全的，重点应该放在优化生成的 C 代码性能上，而不是增加语言后端的种类。
  
- **mnist 数据集的代码标准化**：一位用户观察到 **tinygrad** 在不同文件中处理 mnist 数据集的方式不一致，并提出了三种解决方案，从对示例文件进行微调到在 extra 目录中维护当前的数据集获取方式。
  
- **关于内存安全和语言使用的政治影响的讨论**：一位用户强调了 Rust 的内存安全记录，并表达了对 Rust 基金会在商标和许可方面的做法的不满，将这种组织行为与 Java 与 Oracle 的情况进行了对比，并表达了反对使用受许可产品的个人立场。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lunduke.substack.com/p/the-rust-foundation-goes-to-war-against">Rust 基金会向使用 "Rust" 一词的人开战</a>：说真的。这篇文章的标题违反了新的 Rust 商标政策。这太疯狂了。</li><li><a href="https://www.cvedetails.com/vulnerability-list/vendor_id-19029/product_id-48677/Rust-lang-Rust.html">Rust-lang Rust：安全漏洞，CVE</a>：Rust-lang Rust 的安全漏洞：影响该产品任何版本的漏洞列表。</li><li><a href="https://github.com/tinygrad/tinygrad/actions/runs/8633930065/job/23668153464">不再有底层 diskbuffer，那只是设备 (#4129) · tinygrad/tinygrad@ee457a4</a>：你喜欢 pytorch？你喜欢 micrograd？你爱 tinygrad！❤️ - 不再有底层 diskbuffer，那只是设备 (#4129) · tinygrad/tinygrad@ee457a4</li><li><a href="https://github.com/tinygrad/tinygrad/commit/b0f14b4af886de8bc04a4cacc48880af24d69632?diff=unified&w=0)">将数据集移动到 datasets 文件夹 · tinygrad/tinygrad@b0f14b4</a>：未找到描述</li><li><a href="https://github.com/tinygrad/tinygrad/pull/4138#">jaredeh 提交的 Rust 后端 · Pull Request #4138 · tinygrad/tinygrad</a>：添加 Rust 后端。代码行数中性（假设可以将 rust.py 放在 ./extra 中）。测试基本通过。包括 examples/export_code_from_onnx.py，它创建了一个零依赖的纯 Rust rlib crate。性能...
</li>
</ul>

</div>
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1227515872260915291)** (23 条消息🔥): 

- **明确了张量转换**：提供了一个简单的解决方案，使用 `.numpy()` 方法将张量转换为 numpy 数组，符合用户的需求。
- **设备实现练习**：建议学习者练习实现一个 "NPY" 设备以支持存储张量，示例为 `Tensor(1).to("NPY")`。
- **George Hotz 反对 NPY 设备的想法**：最初，George Hotz 否决了将 NPY 作为一个设备的练习，强调 NPY 应该仅用于复制到 GPU，反之则不然。
- **认可教学机会**：尽管最初予以否决，George Hotz 后来承认创建一个 NPY 设备对于学习代码库中的分配器（allocators）来说是一个有价值的练习。
- **练习质量辩论**：简要讨论了小型练习与更复杂的悬赏任务（bounties）的实用性，George Hotz 建议简单的任务可能更有益，因为它们允许学习者根据正确答案检查自己的工作。
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1227576296012255253)** (78 条消息🔥🔥): 

- **AI 艺术爱好者讨论训练和外包**：社区成员对**腾讯的 Ella SDXL** 表示怀疑，并思考其他实体如 'Draw Things' 或 **CivitAI** 是否会尝试训练其变体。预算限制和不同 AI 项目的战略选择是热门话题，一些成员分享了[来自 reddit 帖子的见解，讨论腾讯拒绝发布 SDXL 版本](https://reddit.com)。

- **音乐 AI 的探索与艺术家参与**：介绍了 **Udio 的音乐创作应用**，因其旨在成为音乐家的工具而受到关注，并获得了 [will.i.am 和 Common](https://x.com/udiomusic/status/1778045337833193720) 等艺术家的认可。讨论中涉及了用户参与度、上传真实乐器轨道的能力，以及使用潜扩散模型 (latent diffusion model) 进行音乐生成的技术。

- **多样化的 AI 硬件与模型加速**：社区成员讨论了各大厂商推出的 AI 加速硬件，重点提到了 [Meta 披露的新型 Meta Training and Inference Accelerator (MTIA) AI 硬件](https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/)，并强调其在 90 W TDP 下实现 354 TFLOPS 的性能表现非常出色。

- **文本转语音 (TTS) 技术进展**：探讨了 TTS 领域的技术进步，重点关注了 Huggingface 团队推出的一款新型 TTS 引擎，该引擎采用了与 Stability 尚未发布的 TTS 类似的架构，并支持语音提示 (voice prompting) 功能：[Parler TTS](https://github.com/huggingface/parler-tts)。

- **Laion 5B Web Demo 的未来尚不明确**：用户询问了 **Laion 5B web demo** 的状态，回复显示目前没有明确的恢复日期，正受困于法律问题和行政流程。不过，对于寻求类似搜索引擎功能的用户，建议使用 [cc2dataset](https://github.com/rom1504/cc2dataset) 创建个人数据集作为替代方案。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.udio.com/songs/renWwtB7Zqk2mqZamEHHgJ>">未找到标题</a>：未找到描述</li><li><a href="https://tenor.com/mDErrG5aLdg.gif">Pinoquio GIF - Pinoquio - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/udiomusic/status/1778045337833193720">来自 udio (@udiomusic) 的推文</a>：我们的目标是让 Udio 成为音乐家和非音乐家共同的变革性工具，我们很高兴能得到顶尖艺术家 @iamwill 和 @common 的支持。 8/11</li><li><a href="https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/">未找到标题</a>：未找到描述</li><li><a href="https://x.com/udiomusic/status/1778045322654003448">来自 udio (@udiomusic) 的推文</a>：介绍 Udio，一款用于音乐创作和分享的应用，让你通过直观且强大的文本提示生成你喜爱风格的惊艳音乐。 1/11</li><li><a href="https://news.ycombinator.com/item?id=39992817">Show HN: Sonauto – 一个更具可控性的 AI 音乐创作工具 | Hacker News</a>：未找到描述
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1227717146528972931)** (7 条消息): 

- **Intel Lunar Lake 将赋能 Copilot AI**：Intel 下一代 [Lunar Lake CPU](https://www.reddit.com/r/singularity/comments/1c0ue4f/intels_nextgen_lunar_lake_cpus_will_be_able_to/) 预计将凭借强大的 45 TOPS 神经网络处理单元 (NPU) 在本地运行 Microsoft 的 Copilot AI。
- **硬件供应担忧**：一名成员询问高端 AI 硬件的芯片制造是否存在瓶颈，或者 Nvidia 是否在限制供应。
- **半导体制造见解**：指出 **Intel 拥有自己的晶圆代工厂**，而 AMD 和 Nvidia 则依赖 TSMC 进行生产。
- **LRU 修改影响基准测试**：根据 Long Range Arena (LRA) 基准测试，修改后的最近最少使用 (LRU) 算法被提到是有效的。
- **LRA 的实际性能受到质疑**：关于 **LRA** 在模拟**现实世界长上下文性能**方面的有效性引发了讨论。

**提到的链接**：<a href="https://www.reddit.com/r/singularity/comments/1c0ue4f/intels_nextgen_lunar_lake_cpus_will_be_able_to/">Reddit - 深入探索</a>：未找到描述

  

---


**LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/1227931111926009886)** (1 条消息): 

- **请求 HowTo100M 数据集访问权限**：一名成员正在询问如何访问 **HowTo100M 数据集**，并想知道这是否是正确的咨询频道。该数据集位于 [di.ens.fr](https://www.di.ens.fr/willow/research/howto100m/)。
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1227524493522767892)** (60 条消息 🔥🔥): 

- **追踪 OpenAIAssistant 的 Token 使用情况**：用户询问如何追踪 OpenAIAssistant 的 Token 使用情况。建议通过挂载 LLM 回调，使用 **tiktoken** 计算 Token 数量，并结合定价进行成本估算。

- **向量数据库中的元数据利用**：讨论了在公司政策的向量数据库中使用元数据过滤器来回答特定查询，例如查找允许负假期余额的公司。一位成员解释说，元数据过滤器限定了搜索空间，不会将上下文传递给 LLM，但可以通过自定义 Retriever 来包含元数据。

- **With_structured_output 方法咨询**：一位用户询问 **ChatOpenAI 类**中的 `with_structured_output` 方法是否已被弃用。通过共享的文档澄清，该方法并未弃用，而是处于 Beta 阶段，并提供了一个 JavaScript 的代码示例来演示其用法。

- **探索结构化输出选项**：一位用户提到了 [Instructor for Python](https://python.useinstructor.com/) 工具，并询问如何将其与 LangChain 集成。另一位用户指向了 [LangChain Python 文档](https://python.langchain.com/docs/modules/model_io/chat/structured_output/)，以获取有关结构化 LLM 输出的信息。

- **移动端个性化 AI 模型访问**：有人咨询了关于个性化 AI 访问并能够测试各种模型的能力，特别是针对移动设备的使用。一位用户正在考虑为 **Pythonista** 编写脚本，并利用 **Bing's API** 获取网页结果，因为它具有良好的延迟表现。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://python.langchain.com/docs/get_started/introduction#api-reference>).">简介 | 🦜️🔗 LangChain</a>：LangChain 是一个用于开发由大语言模型 (LLM) 驱动的应用程序的框架。</li><li><a href="https://python.useinstructor.com/">欢迎使用 Instructor - Instructor</a>：未找到描述</li><li><a href="https://python.langchain.com/docs/modules/model_io/chat/structured_output/">[beta] 结构化输出 | 🦜️🔗 LangChain</a>：让 LLM 返回结构化输出通常至关重要。这是</li><li><a href="https://python.langchain.com/docs/guides/structured_output#openai>).">[beta] 结构化输出 | 🦜️🔗 LangChain</a>：让 LLM 返回结构化输出通常至关重要。这是</li><li><a href="https://python.langchain.com/docs/guides/structured_output#openai>)">[beta] 结构化输出 | 🦜️🔗 LangChain</a>：让 LLM 返回结构化输出通常至关重要。这是</li><li><a href="https://js.langchain.com/docs/integrations/chat/openai#withstructuredoutput-->).">ChatOpenAI | 🦜️🔗 Langchain</a>：你可以按如下方式使用 OpenAI 的聊天模型：</li><li><a href="https://github.com/outlines-dev/outlines">GitHub - outlines-dev/outlines: 结构化文本生成</a>：结构化文本生成。通过在 GitHub 上创建账户来为 outlines-dev/outlines 做出贡献。
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1227860949105381407)** (5 messages): 

- **非 OpenAI LLM Chain 的请求**：一位成员请求为非 OpenAI 的开源 LLM 创建带有 Function calling 的 **LangChain** 指南。
- **LangChain 的通用性得到认可**：针对该请求，有人指出 LangChain 的设计本质上独立于任何特定的 LLM 提供商，暗示其对各种 LLM 的适应性。
- **缺乏具体指令**：后续回复强调，知识库中缺乏为非 OpenAI 开源 LLM 编写带有 Function calling 的 Chain 的精确示例或指导。
- **寻找更多信息的指南**：为了获得将开源 LLM 与 LangChain 集成的详细指导，该成员被引导咨询官方 [LangChain 文档](https://langchain.readthedocs.io/) 或寻求社区帮助。
  

---


**LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/)** (1 messages): 

lhc1921: https://python.langchain.com/docs/integrations/llms/azure_openai/
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1227871914635038762)** (3 messages): 

- **GPT AI 首次亮相，搭载 GPT-4 和 Vision AI**：推出了一款名为 **GPT AI** 的新应用，其特点是搭载了 **GPT-4** 和用于图像识别的 *Vision AI* 技术，提供数据分析、语言学习和编码等多种功能。它声称拥有美观的界面、无对话限制和即时模式切换，可在 [Google Play Store](https://play.google.com/store/apps/details?id=com.projecthit.gptai&referrer=ph-aai) 下载。

- **Galaxy AI 发布免费高级 AI API**：**Galaxy AI** 推广其 PREMIUM AI 模型的**免费 API** 服务，包括最新的 **Gemma**、**GPT-4**、**GPT-3.5-turbo** 和 **Gemini-PRO API**，采用 **OpenAI 格式**以便于项目集成。感兴趣的用户受邀[立即尝试](https://discord.com/invite/BSphj69773)。

- **Appstorm 平台通过 v1.6.0 版本提升应用构建体验**：**Appstorm v1.6.0** 已发布，其特点包括移动端注册、音乐生成能力、地图集成、增强的数据探索和共享功能，以及提升了处理更多并发应用的平台改进，并修复了应用恢复的 bug。目前可通过 [beta.appstorm.ai](https://beta.appstorm.ai/) 访问。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://galaxyapi.onrender.com">Galaxy AI - Swagger UI</a>：未找到描述</li><li><a href="https://beta.appstorm.ai/">Appstorm</a>：在几秒钟内构建 AI 应用</li><li><a href="https://play.google.com/store/apps/details?id=com.projecthit.gptai&referrer=ph-aai">GPT AI - Chat GPT-4 &amp; Vision - Google Play 上的应用</a>：未找到描述
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1227558541414629408)** (5 条消息): 

- **寻找 Web 开发模板**：一位成员表示希望找到现成的 Web 开发模板，结果发现他们可能需要深入研究 Web 开发才能实现目标。
- **它是热狗吗？图像分类教程**：分享了一个名为 ["使用 Ollama, Mistral 和 LLava 判断是否为热狗"](https://www.youtube.com/watch?v=76BkMULO7uw) 的教程视频，教观众如何使用各种机器学习模型判断图像中是否包含热狗。
- **关于 LCEL 和 Runnables 的新教程**：一位成员发布了一个专注于 LangChain 的 LCEL 以及从 runnables 组合链（composing chains）的教程，并邀请社区提供反馈。教程地址为 [Langchain Tutorial: LCEL and Composing Chains from Runnables](https://medium.com/@klcoder/langchain-tutorial-lcel-and-composing-chains-from-runnables-751090a0720c?sk=55c60f03fb95bdcc10eb24ce0f9a6ea7)。

**提到的链接**：<a href="https://www.youtube.com/watch?v=76BkMULO7uw">Hot dog or not with Ollama, Mistral and LLava</a>：在本教程中，我们使用 Ollama, Mistral 和 LLava 来查看图像是否为热狗 #python #pythonprogramming #llm #ml #ai #aritificialintel...

  

---



**DiscoResearch ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/1227544935398506586)** (10 条消息🔥): 

- **Mixtral 模型转换脚本公开**：成员们分享了将 Mistral MoE 权重转换为 Hugging Face 格式的链接：一个是社区成员提供的 [非官方脚本](https://huggingface.co/DiscoResearch/mixtral-7b-8expert/blob/main/convert_mistral_moe_weights_to_hf.py)，另一个是 transformers 仓库中的 [官方脚本](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/convert_mixtral_weights_to_hf.py)。

- **在 HF 上发布 Mixtral-8x22B 模型**：推出了新的 **Mixtral-8x22B** 模型，可在 Hugging Face 下载。[模型卡片](https://huggingface.co/v2ray/Mixtral-8x22B-v0.1) 包含了转换脚本，并感谢 Hugging Face 工作人员将其克隆到官方仓库。

- **Mistral 社区分享 176B 参数模型规格**：一位成员引用 Discord 的消息称，有一个 176B 参数的模型，性能介于 GPT-4 和 Claude Sonnet 之间，**使用与 Mistral 7b 相同的 tokenizer**，并拥有高达 **65536** 的海量序列长度。

- **澄清模型性能混淆**：针对之前分享的帖子，另一位成员澄清说，GPT-4 和 Claude Sonnet 之间的对比性能实际上是指 command-r+ 模型，而不是 Mistral。

- **22B MoE 模型的实验性合并**：Vezora 在 Hugging Face 上发布了一个由 MOE 模型制作的实验性 **22B 参数稠密（dense）模型**。[Mistral-22B-v0.1](https://huggingface.co/Vezora/Mistral-22B-v0.1) 是宣布的首个成功的 MOE 转 Dense 模型转换。

- **Mergekit 与微调的挑战**：尽管尝试使用 Mergekit 合并模型并随后进行微调，但一位成员报告结果不佳，表示社区内普遍有类似经历，导致一些人避免使用自定义合并进行训练。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/Vezora/Mistral-22B-v0.1">Vezora/Mistral-22B-v0.1 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1">HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/v2ray/Mixtral-8x22B-v0.1">v2ray/Mixtral-8x22B-v0.1 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/DiscoResearch/mixtral-7b-8expert/blob/main/convert_mistral_moe_weights_to_hf.py">convert_mistral_moe_weights_to_hf.py · DiscoResearch/mixtral-7b-8expert at main</a>: 未找到描述</li><li><a href="https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/convert_mixtral_weights_to_hf.py">transformers/src/transformers/models/mixtral/convert_mixtral_weights_to_hf.py at main · huggingface/transformers</a>: 🤗 Transformers: 面向 Pytorch、TensorFlow 和 JAX 的业界领先的机器学习库。 - huggingface/transformers
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1227517055771869214)** (22 条消息🔥): 

- **Apache 2.0: 首选许可证**: 讨论中的贡献确认了相关模型采用 Apache 2.0 授权，且预计很快会发布 instruct 版本。人们对新分享模型的许可条款表现出浓厚兴趣。

- **Mixtral 在基准测试中占据主导地位**: 初步的 AGIEval 结果显示 **Mixtral** 模型优于其他基础模型，成员们对其令人印象深刻的性能表示惊讶。具体的基准测试包括 **Mixtral-8x22B-v0.1** 在 **PIQA、BoolQ** 等任务上取得了显著得分。

- **基准测试对比和背景细节**: 讨论提供了 **Mixtral** 模型在各项任务中的性能对比，以及基准测试环境的细节，透露评估是在使用 **4xH100** 的 **vLLM** 上进行的。

- **直接获取模型及相关讨论**: 一位成员确认使用了 Hugging Face 上特定的 **Mixtral-8x22B-v0.1** 模型，并链接到了一个[讨论](https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1/discussions/3#6616af73203bf9d751696a84)，其中展示了 **5-shot MMLU** 结果，表明其达到了新的开源 SOTA 性能。

- **德语基准测试的潜力**: 对德语基准测试的兴趣引发了关于合适基准的问题，暗示可能会使用 **lm-evaluation-harness**，但也对目前德语基准数据的相关性和需求提出了疑问。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/v2ray/Mixtral-8x22B-v0.1">v2ray/Mixtral-8x22B-v0.1 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1/discussions/3#6616af73203bf9d751696a84">mistral-community/Mixtral-8x22B-v0.1 · MMLU - 77</a>: 未找到描述
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1227528811751669843)** (41 条消息🔥): 

- **基准测试揭示了令人费解的模型行为**: PhilipMay 指出 `DiscoResearch/DiscoLM_German_7b_v1` 中存在一个奇特的性能差异，这取决于 ChatML Template 中是否存在换行符。当在 `
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2211.01786">Crosslingual Generalization through Multitask Finetuning</a>：多任务提示微调 (MTF) 已被证明有助于大语言模型在零样本（zero-shot）设置下泛化到新任务，但到目前为止，对 MTF 的探索主要集中在英语数据和模型上....</li><li><a href="https://huggingface.co/occiglot/occiglot-7b-de-en-instruct">occiglot/occiglot-7b-de-en-instruct · Hugging Face</a>：未找到描述</li><li><a href="https://arxiv.org/html/2404.04850v1">Lucky 52: How Many Languages Are Needed to Instruction Fine-Tune Large Language Models?</a>：未找到描述</li><li><a href="https://huggingface.co/DiscoResearch/DiscoLM_German_7b_v1/blob/main/tokenizer_config.json#L48">tokenizer_config.json · DiscoResearch/DiscoLM_German_7b_v1 at main</a>：未找到描述</li><li><a href="https://huggingface.co/DiscoResearch/DiscoLM-70b#dataset">DiscoResearch/DiscoLM-70b · Hugging Face</a>：未找到描述</li><li><a href="https://arxiv.org/html/2404.04659v1">Multilingual Pretraining and Instruction Tuning Improve Cross-Lingual Knowledge Alignment, But Only Shallowly</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/deutsche-telekom/Ger-RAG-eval">deutsche-telekom/Ger-RAG-eval · Datasets at Hugging Face</a>：未找到描述</li><li><a href="https://github.com/huggingface/lighteval/blob/main/community_tasks/german_rag_evals.py">lighteval/community_tasks/german_rag_evals.py at main · huggingface/lighteval</a>：LightEval 是一个轻量级的 LLM 评估套件，Hugging Face 内部一直在将其与最近发布的 LLM 数据处理库 datatrove 和 LLM 训练库 nanotron 配合使用。 - hug...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/5ed29393e34cf57b24a20ac1bafa3a94272ac3f5/src/axolotl/prompt_strategies/dpo/chatml.py#L86">axolotl/src/axolotl/prompt_strategies/dpo/chatml.py at 5ed29393e34cf57b24a20ac1bafa3a94272ac3f5 · OpenAccess-AI-Collective/axolotl</a>：尽管提问。通过在 GitHub 上创建一个账户来为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。
</li>
</ul>

</div>
  

---



**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1227554085268619274)** (23 条消息🔥): 

- **新模型即将问世**：Sophia Yang 通过推文确认了一个全新的模型，明确表示它既不是 *Mistral* 也不是 *EleutherAI* 的衍生品。Discord 中确认该模型的许可证为 Apache 2.0。
  
- **仓促发布的猜测**：有关 **Llama** 和 **Cohere** 即将发布新版本的传闻可能迫使该新模型在未完成完整评估的情况下发布，发布推文中缺少 checksum（校验和）也印证了这一点。

- **基准测试取代文档**：虽然 **Mistral** 传统上在发布后会延迟提供完整文档以鼓励社区炒作，但 [AGIEval 结果显示](https://fxtwitter.com/jphme/status/1778028110954295486) 新的 8x22b 模型表现优于其他开源基模型。

- **呼吁公正的评估**：一位成员表示有兴趣创办一个博客，在每个主要模型发布时发布公开且公正的人类评估（human evals），因为目前的基准测试可能无法满足开发者构建产品的需求，令人感到沮丧。

- **Mistral 社区对比基模型与 R+**：一张来自 Hugging Face 的图片展示了 **BigXtra** 的基模型与其经过指令微调的 R+ 版本的性能对比；讨论中引导人们根据评估分数对数据集进行推测。[查看讨论](https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1/discussions/4) 和 [评估分数图片](https://cdn-uploads.huggingface.co/production/uploads/6382255fcae34727b9cc149e/ds0rDhvNWZl0dWsrfYb0K.jpeg)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fxtwitter.com/sophiamyang/status/1777978822199017728">来自 Sophia Yang, Ph.D. (@sophiamyang) 的推文</a>：@LiHongtu12138 都不是。这是一个全新的模型。</li><li><a href="https://fxtwitter.com/jphme/status/1778028110954295486">来自 Jan P. Harries (@jphme) 的推文</a>：@MistralAI 新的 8x22b 模型的第一批 AGIEval 结果出炉，碾压了所有其他开源（基）模型 - 🤯
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1227846875843919966)** (7 条消息):

- **探究从预训练到 RLHF 的过渡**：一位成员对在语言模型通常的 **Pretraining -> IFT -> RLHF** 流水线中，像 **指令微调 (IFT)** 这样的步骤是否必要感到好奇。他们思考是否可以直接从预训练模型跳跃到 **RLHF** (来自人类反馈的强化学习)，因为人类偏好评分在 RLHF 期间隐含地教授了指令遵循。他们假设，鉴于指令遵循的直接性质，使用监督数据集来教授它可能更简单，而生成高质量输出则更为复杂。这一疑问源于学生在 [Stanford CS224N 讲座](https://www.youtube.com/watch?v=SXpJ9EmG3s4) 中的提问。

- **训练阶段的界限模糊**：作为回应，另一位成员暗示预训练、指令遵循微调和 RLHF 之间的界限在实践中并不明确，这意味着训练过程是 **融合在一起 (blended together)** 的。

- **寻求模型训练融合的清晰解释**：寻求信息的成员思考了“融合”意味着什么，询问是否涉及以某种方式组合预训练、SFT (监督微调) 和 RLHF 提示词的数据集，并指出了 PT/SFT 与 RLHF 中使用的目标函数的差异。他们请求提供进一步的资源以求清晰。

- **期待即将发布的退火资源**：讨论融合实践的同一位成员提到了即将发布的、可能尚未记录的 **模型训练** 方法，包括课程学习 (curriculums)、调度器 (schedulers) 或退火 (annealing)。他们透露很快就会有关于 **退火 (annealing)** 的内容发布。

**提到的链接**：<a href="https://www.youtube.com/watch?v=SXpJ9EmG3s4&list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4&index=10">Stanford CS224N | 2023 | 第 10 讲 - 提示工程、来自人类反馈的强化学习</a>：有关斯坦福人工智能专业和研究生课程的更多信息，请访问：https://stanford.io/ai 了解更多关于本课程的信息...

  

---


**Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1227737709792198797)** (5 条消息): 

- **机器学习内幕交易指控**：一位成员强调了一种暗示内幕交易的情况，归因于微调 (fine-tune) 的质量低下。其含义是微调过程未达到预期标准。
- **学术利益冲突**：同一位成员对可能的利益冲突表示担忧，指出 **机器学习系统教授** 可能会投资于像 **Databricks** 这样的公司。
- **Burkov 的推文引发讨论**：Anton Burkov 的一条推文在群组中被分享，作为对话的起点。提供的消息中未讨论推文的具体内容或背景。[Burkov 的 Twitter 帖子](https://twitter.com/burkov/status/1778123473929282015)
- **指控引发的紧张气氛**：“SPICY”一词反映了所讨论指控的激烈且可能具有争议的性质。
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1227750875901067274)** (14 条消息🔥): 

- **可能的 John Schulman 访谈预告**：提到了与 **John Schulman** 的访谈，虽然尚不确定，但希望能将“也许”变为“确定”。
- **意外的确认**：一位用户幽默地指出了对某个未指明话题的潜在意外确认。有人建议这可能只是作为一个梗输入的，但仍然觉得很有趣。
- **会议确认的恶作剧？**：一位成员指出他们的系统不需要接受订阅，暗示了出现意外确认的原因。
- **新成员招募策略**：简短交流了关于向服务器招募新成员的事宜，其中特别提到了一个名叫 **Satya** 的人。
  

---



**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1228020142512804032)** (4 条消息): 

- **AI 的音频感知能力飞跃**：Gemini 更新后的功能现在包括回答有关视频中音频的问题，这标志着与其之前仅能生成不含音频的视频内容描述相比，有了显著增强。

- **Google 的格式化困扰**：一位用户表示 Google 需要解决在将其 playground 中粘贴文本时遇到的文本格式问题。

- **利用 AI 策划知识**：GitHub 上的 [斯坦福 Storm 项目](https://github.com/stanford-oval/storm) 是一个令人兴奋的进展；它是一个由 LLM 驱动的知识策划系统，旨在研究主题并生成包含引用的综合报告。

**提到的链接**：<a href="https://github.com/stanford-oval/storm">GitHub - stanford-oval/storm: An LLM-powered knolwedge curation system that researches a topic and generates a full-length report with citations.</a>：一个由 LLM 驱动的知识策展系统，可研究特定主题并生成带有引用的完整报告。- stanford-oval/storm

---

**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1227674842107740280)** (12 条消息🔥): 

- **LLM 命令行工具故障**：一位用户遇到 `llm cmd` 在 macOS 上使用 iTerm2 时挂起的问题，且没有报告任何错误。即使尝试通过 SSH 在 Ubuntu 服务器上运行该命令，问题似乎也仅存在于本地 macOS 设置中。
- **Shell 定制并非故障原因**：用户指出虽然使用了*高度定制的 omzsh shell*，但确认这并非问题根源，因为他们在 Ubuntu 上使用了相同的配置，而 `llm cmd` 在那里可以正常工作。
- **测试不同的安装方式**：在排查故障时，他们从 homebrew 版本切换到了 pipx 版本，但未能成功解决，并确认 `llm logs` 中没有反映出该异常行为。
- **可能是交互提示问题**：他们发现该命令正在等待交互，在输入 `'???'` 后会返回输出，这表明进程并未挂起，而是在等待输入。
- **Pull Request 提供了解决方案**：用户分享了一个 [GitHub pull request](https://github.com/simonw/llm-cmd/pull/12)，并确认它解决了 macOS zsh 上 `llm cmd` 的挂起问题，这为讨论中的问题提供了一个潜在的修复方案。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/simonw/llm-cmd">GitHub - simonw/llm-cmd: Use LLM to generate and execute commands in your shell</a>：在你的 shell 中使用 LLM 生成并执行命令 - simonw/llm-cmd</li><li><a href="https://github.com/simonw/llm-cmd/pull/12">fix: macos zsh llm cmd hangs by nkkko · Pull Request #12 · simonw/llm-cmd</a>：针对 #11 的修复，已在 M1 MacOs (14.3.) 的 Terminal 和 Alacritty (zsh) 中测试，现在运行正常。
</li>
</ul>

</div>

---

**Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1228056986071990373)** (1 条消息): 

- **Gradio 与 Figma 结合**：Mozilla Innovations 推出了 **Gradio UI for Figma**，这是一款基于 Hugging Face 低代码原型设计工具包的工具，旨在通过快速创建线框图来简化设计阶段，从而加速实验进程。访问 [Figma 的 Mozilla 页面](https://www.figma.com/@futureatmozilla) 获取 Gradio UI 库。
- **加入关于 Gradio UI 的讨论**：如有疑问或想进一步讨论 **Gradio UI for Figma**，请在 [Discord 讨论频道](https://discord.com/channels/1089876418936180786/1091372086477459557/1228056720132280461) 加入与 Mozilla Innovation Studio 的 Thomas Lodato 的讨论。

**提到的链接**：<a href="https://www.figma.com/@futureatmozilla">Figma (@futureatmozilla) | Figma</a>：来自 Mozilla Innovation Projects (@futureatmozilla) 的最新文件和插件 —— 我们正在构建专注于创建更个人化、私密且开源的互联网产品。

---

**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1227593409972404265)** (11 条消息🔥): 

- **解决 GPU 显存限制问题**：一位成员成功使用 `-ngl 3` 来调整其 GPU 有限的显存，将部分层移动到 CPU 内存中，但报告称**性能显著下降**。他们发现像 7B 这样的小型模型在这些受限条件下的表现要好得多。
- **动态层卸载的功能请求**：针对 1050 等旧款 GPU 型号的 VRAM 限制，一位成员询问 **llamafile** 是否可以智能地卸载层，以避免在 VRAM 不足时发生崩溃。
- **ollama 对大语言模型的处理**：分享了 **GitHub 上的 ollama 项目**链接，强调了它如何管理大语言模型的运行，这可能会为改进 llamafile 的内存处理提供参考。
- **转换 Tensor 时触发内核恐慌 (Kernel Panic)**：一位成员讲述了在 M2 MacBook 上尝试将 `.safetensors` 文件转换为 `.gguf` 时导致内核恐慌的经历，认为这是由于设备 16GB RAM 的限制造成的。
- **介绍用于文本隐式推理的 Quiet-STaR**：引入了关于 **Quiet-STaR** 的讨论，这是一种让语言模型在每个 token 处生成推理过程以改进文本预测的方法，并附带了 [研究论文](https://arxiv.org/abs/2403.09629) 和 [GitHub 仓库](https://github.com/ezelikman/quiet-star) 的链接。Hugging Face 也展示了一个与 Quiet-STaR 相关的仓库。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2403.09629">Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking</a>: 在写作和交谈时，人们有时会停下来思考。虽然以推理为中心的工作通常将推理框架化为回答问题或完成 Agent 任务的方法，但推理是...</li><li><a href="https://huggingface.co/ezelikman/quietstar-8-ahead/tree/main">ezelikman/quietstar-8-ahead at main</a>: 未找到描述</li><li><a href="https://github.com/ezelikman/quiet-star">GitHub - ezelikman/quiet-star: Code for Quiet-STaR</a>: Quiet-STaR 的代码。通过在 GitHub 上创建账户，为 ezelikman/quiet-star 的开发做出贡献。</li><li><a href="https://github.com/ollama/ollama/blob/c5c451ca3bde83e75a2a98ed9fd4e63a56bb02a9/llm/server.go#L43>">ollama/llm/server.go at c5c451ca3bde83e75a2a98ed9fd4e63a56bb02a9 · ollama/ollama</a>: 快速上手 Llama 2、Mistral、Gemma 和其他大语言模型。 - ollama/ollama
</li>
</ul>

</div>
  

---



**Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1227591389597007964)** (1 messages): 

- **Mistral 8x22b 在 AGIEval 中碾压竞争对手**：**Mistral 8x22b** 的初步 [AGIEval 结果](https://x.com/jphme/status/1778030213881909451)已经发布，显示该模型显著超越了所有其他开源基座模型。Mistral 团队因发布这一新 AI 模型而获得赞誉。

**提到的链接**：<a href="https://x.com/jphme/status/1778030213881909451">来自 Jan P. Harries (@jphme) 的推文</a>：@MistralAI 首批 AGIEval 结果看起来很棒 👇 - 伙计们，感谢发布这个猛兽！👏 https://x.com/jphme/status/1778028110954295486  ↘️ 引用 Jan P. Harries (@jphme) 的话：首批 AGIEval 结果...

  

---


**Skunkworks AI ▷ #[datasets](https://discord.com/channels/1131084849432768614/1131669182124138616/1227982498655633450)** (5 messages): 

- **寻求逻辑数据集**：一位成员询问了关于自然文本中命题逻辑和谓词逻辑推理的数据集。
- **推理 AI 的精选列表**：[awesome-reasoning GitHub 仓库](https://github.com/neurallambda/awesome-reasoning)提供了一个推理 AI 数据的精选列表，用于回复关于逻辑推理数据集的查询。
- **用于可靠逻辑推理的 LOGIC-LM 项目**：另一位成员提供了 [Logic-LLM GitHub 仓库](https://github.com/teacherpeterpan/Logic-LLM)的链接，该项目专注于通过符号求解器增强大语言模型的逻辑推理能力。
- **迈向兼容 COQ 的 LLM**：对话中提到了一篇 [arXiv 上的 Coq 数据集](https://arxiv.org/abs/2403.12627)，旨在训练大语言模型处理用于形式化定理证明的 Coq 证明语言。
- **澄清项目目标**：一位成员寻求对项目目标的澄清，想知道其是否旨在通过将人类命题转换为 Lisp 进行执行和验证，从而增强 LLM 的推理能力。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2403.12627">Enhancing Formal Theorem Proving: A Comprehensive Dataset for Training AI Models on Coq Code</a>：在形式化定理证明领域，Coq 证明助手因其验证数学断言和软件正确性的严谨方法而脱颖而出。尽管人工智能取得了进展...</li><li><a href="https://github.com/neurallambda/awesome-reasoning">GitHub - neurallambda/awesome-reasoning: a curated list of data for reasoning ai</a>：推理 AI 数据的精选列表。通过在 GitHub 上创建账户，为 neurallambda/awesome-reasoning 的开发做出贡献。</li><li><a href="https://github.com/teacherpeterpan/Logic-LLM">GitHub - teacherpeterpan/Logic-LLM: The project page for &quot;LOGIC-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning&quot;</a>：“LOGIC-LM：通过符号求解器增强大语言模型以实现可靠逻辑推理”的项目页面 - teacherpeterpan/Logic-LLM
</li>
</ul>

</div>
  

---


**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1227622276351131799)** (2 messages): 

- **Google 发布 CodeGemma**：一段新的 [YouTube 视频](https://www.youtube.com/watch?v=Gb--4supXoo)介绍了 **CodeGemma**，这是 Google 的代码补全模型，拥有一个“强大且轻量级”的 7B 预训练变体，具备编程能力。
- **使用 AI 分类热狗**：另一个 [YouTube 教程](https://www.youtube.com/watch?v=76BkMULO7uw)演示了如何在 Python 编程环境下使用 AI 模型 **Ollama, Mistral 和 LLava** 来判断图像是否为热狗。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=76BkMULO7uw">使用 Ollama, Mistral 和 LLava 识别热狗</a>: 在本教程中，我们探讨如何使用 Ollama, Mistral 和 LLava 来判断一张图片是否为热狗 #python #pythonprogramming #llm #ml #ai #aritificialintel...</li><li><a href="https://www.youtube.com/watch?v=Gb--4supXoo">介绍 CodeGemma：Google 的代码补全模型</a>: CodeGemma 为社区带来了强大且轻量级的编程能力。CodeGemma 模型提供了一个 7B 预训练变体，专门用于...
</li>
</ul>

</div>
  

---



**LLM Perf Enthusiasts AI ▷ #[gpt4](https://discord.com/channels/1168579740391710851/1168582188950896641/1227622776542593074)** (6 条消息): 

- **揭穿 AI 偷懒的传闻**: 尽管 Twitter 上有传言称 GPT 的编程能力有所下降，但一位成员报告称在 *cursor* 中使用时没有遇到问题，并强调其性能更快，代码生成更完整。
- **Gemini 1.5 获得好评**: 虽然没有详细细节，但一位成员提到听到了关于 **Gemini 1.5** 编程能力的正面反馈。
- **Cursor 获得认可**: 一位成员表示更喜欢使用 **cursor** 编写样板代码，并赞赏新 **GPT-4** 的 command-K 功能，尽管在聊天方面他们仍然偏好 Claude opus。
- **Copilot++ 使用印象**: 一位成员称赞了 **Copilot++** 的集成，认为其表现非常出色。
  

---


**LLM Perf Enthusiasts AI ▷ #[claude](https://discord.com/channels/1168579740391710851/1168582222194933860/1227875410054742017)** (2 条消息): 

- **Claude 的代码幻觉**: 一位用户报告了他们第一次遇到 **Claude** 完全幻觉出代码的情况，这种表现与其前身 **GPT-4** 的特征不符。
  


---



**AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1228029803962302546)** (2 条消息): 

- **寻找 Jamba 的代码**: 一位成员询问在哪里可以找到 **Jamba** 的代码。
- **期待更新**: 另一位成员通过简单的询问 "*Any update?*" 表达了对信息的渴求，显示出对近期进展或之前咨询回复的关注。