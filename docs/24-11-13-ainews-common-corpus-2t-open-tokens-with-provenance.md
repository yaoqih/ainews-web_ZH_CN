---
companies:
- pleais
- huggingface
- langchainai
- deepseek
- alibaba
- anthropic
date: '2024-11-14T01:54:53.118250Z'
description: '**Pleais** 通过 **Huggingface** 发布了 **Common Corpus**，这是目前规模最大的全开放多语言数据集，包含超过
  **2 万亿个 token** 以及详细的**出处信息**。


  他们还推出了 **OCRonos-Vintage**，这是一个拥有 **1.24 亿参数的 OCR 纠错模型**，能够在 CPU 和 GPU 上高效修复数字化错误，从而释放
  PDF 中的知识。


  在 AI 工具方面，**LangChainAI** 推出了用于协作**提示词工程**（prompt engineering）的 **Prompt Canvas**；同时
  **DeepSeek** 发布了 **JanusFlow 1.3B**，这是一个统一的多模态大模型，集成了自回归和修正流（rectified flow）模型，以增强**图像理解**与**生成**能力。


  **阿里云**发布了专注于编程的 **Qwen2.5-Coder**，具备先进的代码处理能力；而 **Claude 3.5 Sonnet** 也因其卓越的代码生成表现而受到关注。


  此外，**Tim Dettmers** 等人关于**量化挑战**和**精度缩放法则**（scaling laws for precision）的讨论，强调了低精度训练对模型可扩展性和推理效率的影响。文中还提到了《精度缩放法则》论文的见解以及其他提升效率的方法。'
id: d946c7f1-6897-4eb2-a880-64a0fab7aea9
models:
- qwen-2.5-coder
- claude-3.5-sonnet
- janusflow-1.3b
- ocronos-vintage
original_slug: ainews-common-corpus-2t-open-tokens-with
people:
- tim-dettmers
- tom-doerr
- omarsar0
- swyx
- madiator
- reach_vb
title: Common Corpus：具有溯源信息的 2 万亿开放词元
topics:
- provenance
- ocr
- multilingual-datasets
- prompt-engineering
- multimodality
- image-generation
- code-generation
- quantization
- model-scaling
- inference-efficiency
---

<!-- buttondown-editor-mode: plaintext -->**Provenance is all you need.**

> 2024年11月12日至11月13日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord 社区（**217** 个频道，**2494** 条消息）。为您节省了预计阅读时间（以 200wpm 计算）：**274 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

伟大的数据集发布总是先于伟大的模型。上次我们报道了 FineWeb（[我们的报道在此](https://buttondown.com/ainews/archive/ainews-fineweb-15t-tokens-of-commoncrawl/)），随后引发了 [一波 GPT2 speedruns 热潮](https://www.reddit.com/r/LocalLLaMA/comments/1gmd1a8/are_people_speedrunning_training_gpts_now/)。今天，Pleais（通过 Huggingface）带来了更新的 [Common Corpus](https://huggingface.co/blog/Pclanglais/two-trillion-tokens-open)，“这是用于训练 LLM 的最大全开放多语言数据集，包含超过 2 万亿个具有**溯源信息 (provenance information)** 的许可内容 token（2,003,039,184,047 tokens）。”


![image.png](https://assets.buttondown.email/images/39a4c40e-241d-42ef-b910-48d173726991.png?w=960&fit=max)


除了详尽的溯源信息外，团队还使用了 [OCRonos-Vintage](https://huggingface.co/PleIAs/OCRonos-Vintage)，“这是一个轻量级但功能强大的 OCR 纠错模型，可以大规模修复数字化错误。这个拥有 124M 参数的模型在 CPU 和 GPU 上都能高效运行，能够修正间距问题、替换错误词汇并修复损坏的文本结构。”这释放了 PDF 中蕴含的大量知识：


![image.png](https://assets.buttondown.email/images/120274cc-94ed-4244-8798-e0e587ae7603.png?w=960&fit=max)

 

Common Corpus 最初在 [3 月份发布时包含 500b tokens](https://huggingface.co/blog/Pclanglais/common-corpus)，很高兴看到这项工作不断壮大。

---

{% if medium == 'web' %}

**目录**

[TOC]

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}

---

# AI Twitter 回顾

> 所有回顾由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 工具与开发**

- **Prompt Engineering 与协作**：[@LangChainAI](https://twitter.com/LangChainAI/status/1856386593457848746) 推出了 **Prompt Canvas**，这是一种用于 **prompt engineering** 的**新颖 UX**，旨在促进与 AI Agent 的协作，并在组织内标准化提示策略。此外，[@tom_doerr](https://twitter.com/tom_doerr/status/1856507277903307153) 展示了 **llama-ocr** 和 **TTS Generation WebUI** 等工具，增强了开发者的 **OCR** 和 **text-to-speech** 能力。

- **AI 开发平台**：[@deepseek_ai](https://twitter.com/deepseek_ai/status/1856552494379520510) 发布了 **JanusFlow 1.3B**，这是一个**统一的多模态 LLM**，它将**自回归模型 (autoregressive models)** 与 **rectified flow** 相结合，实现了卓越的**图像理解**与**生成**能力。同样，[@swyx](https://twitter.com/swyx/status/1856458546109632984) 提供了关于 **proxy servers** 和 **realtime client SDKs** 的更新，改善了 **realtime applications** 的**开发者体验**。

**AI 模型发布与更新**

- **新 LLM 与增强功能**：[@tom_doerr](https://twitter.com/tom_doerr/status/1856597874991055248) 宣布了 **Qwen2.5-Coder**，这是来自 **Alibaba Cloud** 的**专注代码的 LLM**，强调了其**先进的编程能力**。同时，[@omarsar0](https://twitter.com/omarsar0/status/1856505917686022276) 强调了 **Claude 3.5 Sonnet** 的发布，展示了其与其他模型相比**卓越的代码生成**性能。

- **性能基准测试**：[@omarsar0](https://twitter.com/omarsar0/status/1856505917686022276) 将 **Qwen2.5-Coder** 与 **Claude 3.5 Sonnet** 进行了对比，讨论了它们的**代码生成能力**以及**缩小开源**与**闭源**模型差距的潜力。此外，[@reach_vb](https://twitter.com/reach_vb/status/1856560887437373766) 介绍了 **DeepSeek 的 JanusFlow 1.3B**，突出了其在**多模态任务**中的 **state-of-the-art 性能**。

**AI 研究与技术见解**

- **量化与模型缩放**：[@Tim_Dettmers](https://twitter.com/Tim_Dettmers/status/1856419493766930846) 探讨了 **AI 模型量化 (quantization) 的挑战**，指出**低精度训练**可能会**限制未来的可扩展性**。[@madiator](https://twitter.com/madiator/status/1856430350295257394) 总结了论文 "**Scaling Laws for Precision**"，揭示了**预训练数据的增加**会提高**模型对量化的敏感度**，从而影响**推理效率**和 **GPU 配置**。

- **可扩展性与效率**：[@lateinteraction](https://twitter.com/lateinteraction/status/1856409143051202886) 讨论了**通过精度进行扩展的局限性**，并提出了实现**效率提升**的替代方法。此外，[@deepseek_ai](https://twitter.com/deepseek_ai/status/1856552494379520510) 展示了 **Forge Reasoning Engine**，利用 **Chain of Code**、**Mixture of Agents** 和 **MCTS** 来增强 **Hermes 3 70B** 的**推理**和**规划**能力。

**开发者技巧与工具**

- **系统监控与优化**：[@giffmana](https://twitter.com/giffmana/status/1856429747385073838) 建议从 `htop` 切换到 **btop**，以获得更具**美感**且**功能更强**的系统监控器。此外，[@swyx](https://twitter.com/swyx/status/1856445290632622442) 就管理 **realtime client SDKs** 和优化**开发工作流**提供了指导。

- **软件工程最佳实践**：[@hyhieu226](https://twitter.com/hyhieu226/status/1856532701219828054) 强调了“**没坏就别修！**”的原则，倡导**软件工程**实践中的**简洁性**和**稳定性**。

**AI 采用与影响**

- **医疗保健转型**：[@bindureddy](https://twitter.com/bindureddy/status/1856513273753154025) 讨论了 **AI** 如何结合 **DOGE** 和 **RFK** 等倡议，通过**创新的 AI 解决方案**解决**效率低下**和**高昂成本**问题，从而**改变医疗保健**。

- **自动化与劳动力**：[@bindureddy](https://twitter.com/bindureddy/status/1856425643036291267) 强调了 **AI** 自动化**白领职业**和**改变交通运输**的潜力，预测了对**劳动力的重大影响**，并强调 **AI 采用**仍处于**早期阶段**，**最后一公里**预计将占据**这十年的大部分时间**。

- **企业 AI 创新**：[@RamaswmySridhar](https://twitter.com/RamaswmySridhar/status/1856411645796556887) 介绍了 **Snowflake Intelligence**，实现了**企业 AI** 能力，例如在**企业环境**内促进**数据摘要**和**可操作见解**的 **data agents**。

**梗/幽默**

- **幽默的 AI 评论**：[@nearcyan](https://twitter.com/nearcyan/status/1856565818433355790) 拿用户比起 **Claude** 更喜欢 **ChatGPT** 开玩笑，将其比作拥有“**绿色文本信息**”；而 [@vikhyatk](https://twitter.com/vikhyatk/status/1856576789545660611) 则幽默地概述了**罢工**最终导致**获利**的步骤，为讨论劳工行动增添了轻松的色彩。

- **技术与 AI 幽默**：[@andersonbcdefg](https://twitter.com/andersonbcdefg/status/1856546198137799106) 将 Elon Musk 潜在的**政府修复**比作 George Hotz 对 **Twitter** 的快速修复，用幽默的对比表达了怀疑。此外，[@teortaxesTex](https://twitter.com/teortaxesTex/status/1856624065227751490) 分享了一个关于 AI 建模的有趣看法，其中包含“**i need to lock in**”的重复，为技术讨论增添了趣味。

- **感同身受的开发者笑话**：[@giffmana](https://twitter.com/giffmana/status/1856603033183900128) 幽默地将他冗长的技术演讲称为“**TED 演讲**”，而 [@ankush_gola11](https://twitter.com/ankush_gola11/status/1856400616043483204) 则以俏皮的热情表达了对 **Prompt Canvas** 的兴奋。


---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. Qwen 2.5 Coder 改进了 128K 上下文，但面临可用性挑战**

- **Qwen 2.5 Coder & 128K 上下文窗口 GGUF 的 Bug 修复** ([Score: 332, Comments: 90](https://reddit.com/r/LocalLLaMA/comments/1gpw8ls/bug_fixes_in_qwen_25_coder_128k_context_window/)): 该帖子讨论了 **Qwen 2.5 模型** 的更新和 Bug 修复，强调了使用 YaRN 将上下文长度从 **32K 扩展到 128K**，并重点介绍了 [Hugging Face](https://huggingface.co/unsloth) 上提供的 **原生 128K GGUF**。它还警告不要使用 `
  - **YaRN 与上下文长度**：讨论集中在利用 **YaRN** 扩展 **Qwen 2.5 模型** 的上下文长度。用户对使用 **128K 上下文** 时的性能影响表示担忧，并建议在一般任务中使用 **32K**，仅在必要时调整为更长上下文。
  - **Bug 修复与工具调用 (Tool Calling)**：上传的 **GGUF** 包含 Bug 修复，特别是解决了未训练 Token 和 Pad Token 的问题。值得注意的是，**Coder Base** 和 **Instruct 模型** 都没有针对工具调用进行训练，用户讨论了 `<tool_call>` Token 的未训练状态。
  - **GPU 限制与微调**：用户询问了在 GPU 上训练的最大序列长度，**14B 模型** 在 **40GB GPU** 上大约能达到 **12K 上下文长度**。此外还讨论了最初不使用 YaRN 进行微调的情况以及这种方法的潜在好处。
- **Qwen 2.5 Coder 14b 在技术报告的多个基准测试中表现不如 7b - 奇怪！** ([Score: 37, Comments: 23](https://reddit.com/r/LocalLLaMA/comments/1gpriif/qwen_25_coder_14b_is_worse_than_7b_on_several/)): 正如 [技术报告](https://arxiv.org/pdf/2409.12186) 中所强调的，**Qwen 2.5 Coder 14B** 模型在某些基准测试中的表现不如 **7B** 版本。作者指出，对于 SQL 修订等特定任务，非编程版的 14B 模型表现更好，这表明通用模型在某些语境下可能具有更优的理解能力。
  - 用户报告了 **Qwen 2.5 Coder 14B** 模型的 **性能问题**，一些人认为基准测试可能由于报告错误而不准确，因为他们在实践中观察到了不同的性能。分享了 [Qwen 2.5 Coder 博客](https://qwenlm.github.io/blog/qwen2.5-coder-family/) 的链接以获取更多信息。
  - **量化模型文件的不一致性**：不同的 Q8 文件产生不同的结果，突显了某些文件可能存在缺陷的问题。一位用户分享了来自 [Hugging Face](https://huggingface.co/lmstudio-community/Qwen2.5-Coder-32B-Instruct-GGUF/tree/main) 的可用 Q8 文件，暗示并非所有文件都是可靠的。
  - 一位用户指出 **基准测试表包含错误**，因为 14B 和 1.5B 模型的数据除了 livecode 基准测试外完全相同，这表明可能存在数据输入错误。
- **Qwen 2.5 32B Coder 无法很好地处理 Cline 提示词。幻觉非常严重。有人对其进行过严肃的测试吗？** ([Score: 21, Comments: 46](https://reddit.com/r/LocalLLaMA/comments/1gpqhgu/qwen_25_32b_coder_doesnt_handle_the_cline_prompt/)): 该帖子讨论了 **Qwen 2.5 Coder 32B** 在处理 **Cline 提示词** 时的问题，指出它经常产生幻觉。作者提到尝试了 **vLLM** 和 **OpenRouter/Hyperbolic** 等不同设置但未获成功，不过他们通过使用简单的 Python 脚本管理文件输出来获得了更好的结果。
  - 用户对 **Qwen 2.5 Coder 32B** 的评价褒贬不一；一些人在配备 **64G RAM** 的 **M1** 上使用 [Ollama 版本](https://ollama.com/hhao/qwen2.5-coder-tools) 取得了成功，而另一些人在处理 **Cline** 提示词时遇到问题，导致输出无关内容或陷入无限循环。
  - **配置与安装** 起着至关重要的作用，一位用户建议手动编辑 **config.json** 以便将模型与 **continue** 正确集成。由于缺乏标准的提示词格式，强调正确引导 **Qwen** 的提示词至关重要。
  - 一些用户强调了该模型处理大输入的高效性，能够以极低的成本处理 **50k+ tokens** 和 **100 次 API 调用**，但指出成功与否取决于所使用的集成工具（例如 **AIder**、**cursor**）。


**主题 2. 精度中的扩展定律 (Scaling Laws) 与 CPU 推理测试**

- **在 CPU 上进行张量并行（tensor parallelism）的 LLM 推理** ([Score: 43, Comments: 8](https://reddit.com/r/LocalLLaMA/comments/1gporol/llm_inference_with_tensor_parallelism_on_a_cpu/))：作者使用 **distributed-llama** 项目进行了实验，以评估在 CPU 上进行 **LLM 推理与张量并行** 的可扩展性。在以 **Epyc 9374F** 作为计算节点的第一个实验中，在优化了 logits 计算后，8 个节点的性能扩展到了近 7 倍。第二个实验使用通过 **10Gbe network** 连接的 **Ryzen 7700X** 节点，在 8 个节点下实现了 6 倍的性能提升，证明了 LLM 推理可以在 CPU 上有效扩展，尽管进一步的优化可能会改善结果。作者的 distributed-llama 分支可以在[这里](https://github.com/fairydreaming/distributed-llama)找到。
  - **内存带宽与 NUMA 节点**：讨论中明确了第一个实验中的 8 个节点不是 VM，而是绑定到 **Epyc CPU** 上 NUMA 节点的独立进程。这种设置允许通过 loopback network 进行通信，如果用共享内存通信代替网络通信，则具有潜在的可扩展性改进空间，并强调了双路 **Epyc Turin** CPU 的理论内存带宽为 **2 \* 576 GB/s**。
  - **网络瓶颈考量**：评论者指出，第二个实验中使用的 **10Gbe network** 可能是分布式 CPU 推理的瓶颈。作者承认，虽然在第一个实验中使用了 loopback networking，但物理网络设置可以从调优中受益，以减少延迟并提高效率，特别是涉及 NIC 驱动程序和 OS 网络配置方面。
  - **对分布式 CPU 推理的鼓励**：这些实验结果被认为对分布式 CPU 推理非常有前景。人们有兴趣利用现有系统（包括旧的或中端配置）来执行可扩展的推理任务，重点是优化网络和内存配置以最大化性能。

- **精度的缩放法则。BitNet 是否好得令人难以置信？** ([Score: 27, Comments: 7](https://reddit.com/r/LocalLLaMA/comments/1gq3gs7/scaling_laws_for_precision_is_bitnet_too_good_to/))：一篇新论文 **"Scaling Laws for Precision"** ([arxiv link](https://arxiv.org/pdf/2411.04330)) 探讨了量化如何影响模型精度和输出质量，强调预训练中 token 使用量的增加会加剧量化在后训练（post-training）中的负面影响。作者建议将 6-bit 量化作为最佳平衡点，并希望这些发现能指导各大实验室优化计算资源；**AINews** 通讯中讨论了更多见解，并包含了 **Tim Dettmers** 的观点 ([AINews link](https://buttondown.com/ainews/archive/ainews-bitnet-was-a-lie/))。
  - **量化感知训练 (QAT)** 被强调为一种关键方法，在这种方法中，训练过程能够感知量化，从而实现更有效的权重分布，这与后训练量化形成对比，后者可能会降低模型性能，尤其是在使用 **FP16** 训练时。
  - **cosine learning rate schedule** 被澄清为与 **cosine similarity** 不同，前者与训练动态相关，而后者用于衡量向量相似度，两者都涉及余弦函数，但用途不同。
  - **Bitnet 的方法** 讨论提到该研究未包含 Bitnet 的方法，重点在于以 **bf16** 训练的模型在进行后训练量化时如何丢失重要数据，这与保持 1:1 模型完整性的 **QAT** 不同。


**主题 3. 最大的混合专家模型（Mixture of Expert Models）：分析与性能**

- **迄今为止发布的最大的 Mixture of Expert 模型概览** ([Score: 32, Comments: 2](https://reddit.com/r/LocalLLaMA/comments/1gprkxw/overview_of_the_largest_mixture_of_expert_models/)): 该帖子概述了目前可用的、参数量超过 **1000 亿** 的最大型 **Mixture of Expert (MoE) 模型**，重点介绍了它们的架构、发布日期和质量评估。关键模型包括 **Google 的 Switch-C Transformer**（总参数 1.6 万亿）、**X AI 的 Grok-1**（总参数 3140 亿）以及综合排名最高的 **DeepSeek 的 DeepSeek V2.5**。帖子指出，虽然 **DeepSeek V2.5** 目前排名第一，但 **Tencent 的 Hunyuan Large** 和尚未发布的 **Grok-2** 可能会超越它，并指出模型的适用性取决于具体的用例。更多详情可参考 [HuggingFace 博客](https://huggingface.co/blog/moe) 及帖子中提供的各模型链接。
- **[NousResearch Forge Reasoning 类 O1 模型 https://nousresearch.com/introducing-the-forge-reasoning-api-beta-and-nous-chat-an-evolution-in-llm-inference/](https://i.redd.it/n5j9zfjiwi0e1.png)** ([Score: 240, Comments: 43](https://reddit.com/r/LocalLLaMA/comments/1gptb4i/nousresearch_forge_reasoning_o1_like_models/)): **NousResearch** 推出了 **Forge Reasoning API** 和 **Nous Chat**，旨在增强 **LLM (Large Language Models)** 的推理能力。这一进展代表了 LLM 推理的一次演进，详见其发布公告 [此处](https://nousresearch.com/introducing-the-forge-reasoning-api-beta-and-nous-chat-an-evolution-in-llm-inference/)。
  - **Forge Reasoning API** 并非一个新模型，而是一个通过 **Monte Carlo Tree Search (MCTS)**、**Chain of Code (CoC)** 和 **Mixture of Agents (MoA)** 来增强现有模型推理能力的系统。尽管它是闭源的且仅通过 API 候补名单提供，但它展示了提升 LLM 推理能力的潜力，类似于开源图像生成领域所见到的进步。
  - 讨论中充满了对 **Forge Reasoning API** 的 **开源状态** 和有效性的怀疑与好奇，一些用户将其与 [GitHub](https://github.com/codelion/optillm) 上的 **Optillm** 进行比较，以尝试类似的技术。用户渴望看到独立测试，以验证其声称的推理能力提升。
  - 对话反思了技术进步的本质，将 **NousResearch** 的努力比作历史上随着时间推移而变得普及的突破。它强调了工作流和系统集成相对于独立模型改进的重要性，指出开源 LLM 正开始获得与其他 AI 领域类似的增强。


**主题 4. Qwen 2.5 中不可靠的响应：自我身份识别问题**

- **[qwen2.5-coder-32b-instruct 在使用英语提示时似乎确信自己由 OpenAI 开发。在使用中文提示时则声明由 Alibaba 开发。](https://www.reddit.com/gallery/1gqao05)** ([Score: 22, Comments: 15](https://reddit.com/r/LocalLLaMA/comments/1gqao05/qwen25coder32binstruct_seems_confident_that_its/)): **Qwen 2.5 Coder** 在其来源问题上表现出不一致的行为，当用英语查询时声称由 **OpenAI** 开发，而用中文查询时则声称由 **Alibaba** 开发。
  - **LLM 与内省**：多位用户（包括 **JimDabell** 和 **Billy462**）指出，像 **Qwen 2.5 Coder** 这样的 **Large Language Models (LLMs)** 缺乏内省能力，在被问及来源时经常产生“幻觉”，导致关于其开发者的回答不一致。
  - **不一致的响应**：**pavelkomin** 和 **muxxington** 等用户报告了该模型的各种回答，它声称自己由 **Alibaba**、**OpenAI**、**Tencent Cloud**、**Anthropic** 和 **Meta** 等不同实体开发，这表明其受训练数据中重复短语的影响很大，而非事实准确性。
  - **实际关注点**：一些用户（如 **standard-protocol-79**）对这些不一致之处表示无所谓，只要模型能继续生成有效的代码即可，这表明许多人首要关注的是模型的实用性，而非其自我身份识别的准确性。

- **[如何在当下顺畅地使用 Qwen2.5-Coder-Instruct](https://reddit.com/r/LocalLLaMA/comments/1gpwrq1/how_to_use_qwen25coderinstruct_without/)**（[得分：32，评论：13](https://reddit.com/r/LocalLLaMA/comments/1gpwrq1/how_to_use_qwen25coderinstruct_without/)）：为了提升 **Qwen2.5-Coder-Instruct** 的性能，应避免设置过高的 repetition penalties，建议使用略高于 0 的值。请遵循[推荐的推理参数](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct/blob/main/generation_config.json)，据报告，像 T=0.1 这样的低 temperature 设置并不会产生问题。使用 **bartowski's quants** 可以获得更好的输出质量，并在 system prompts 开头加入 "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." 以增强表现。尽管进行了这些调整，部分用户在配合 **vLLM** 使用时仍遇到问题，并推荐使用 **llama.cpp + GGUF** 等替代方案。
  - 用户们讨论了针对 **Qwen2.5-Coder-32B-Instruct** 等编程模型的 **temperature settings** 和 **repetition penalties**。**No-Statement-0001** 发现 **0.1** 的 temperature 能成功执行复杂的 prompts，而其他人则建议避免高 repetition penalties，因为这会降低性能；**FullOf_Bad_Ideas** 建议关闭 repetition penalties 以获得更好的 zero-shot 结果。
  - 一些用户（如 **Downtown-Case-1755**）对推荐的高 repetition penalties 表示质疑，指出 **1.05** 对于自然包含重复内容的编程任务来说太高了。**EmilPi** 强调了 **Top_K** 设置的重要性，正如在 `generation_config.json` 中观察到的那样，它会显著影响模型性能。
  - **Status_Contest39** 分享了不同部署方案的经验，发现 **DeepInfra** 的默认参数非常有效，尽管其 **Max Token** 限制为 **512**。**Master-Meal-77** 对官方推荐的 sampler 表示不满，更倾向于使用 **top-p 0.9, min-p 0.1, and temp 0.7** 的自定义设置以获得最佳效果。


## 其他 AI 子版块回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**主题 1. AI 视频生成演进：CogVideoX 5B 与 DimensionX 发布**

- **[CogVideoX1.5-5B 图生视频测试](https://v.redd.it/wi3hcmwd4q0e1)**（[得分：91，评论：43](https://reddit.com/r/StableDiffusion/comments/1gqltkx/cogvideox155b_image2video_tests/））：**CogVideoX1.5-5B** 是一款新型的**图生视频生成模型**，展示了其将静态图像转换为视频内容的能力。帖子中未提供额外的技术细节或性能指标。
  - **CogVideoX1.5-5B** 在生成过程中需要 **34GB memory**，VAE 需要 **65GB**；在 **16fps** 下，生成一段 **5 秒视频** 需要 **15 分钟**。该模型目前可通过[命令行推理脚本](https://github.com/THUDM/CogVideo/tree/main/sat)获取，并已在 **A100** 和 **H100 80GB** GPU 上完成测试。
  - 开发更新显示，**Kijai** 正在其 wrapper 中实现 1.5 版本，**cogwrapper** 的测试分支已经可用。**Comfy UI** 的支持集成尚在进行中，而 **Mochi-1** 提供了另一种选择，仅需 **12GB VRAM**。
  - 用户讨论了运动质量的改进，指出更快的播放速度可以减少 AI 生成的慢动作感，正如一段[示例视频](https://streamable.com/m1e1cw)所展示的那样。一些评论集中在生成的动画中需要更真实的物理模拟。


- **[DimensionX：利用可控视频扩散技术从单张图像创建任意 3D 和 4D 场景 | Flux Dev => DimensionX 演示](https://v.redd.it/05q21m50ln0e1)**（[得分：176，评论：51](https://reddit.com/r/StableDiffusion/comments/1gqanyv/dimensionx_create_any_3d_and_4d_scenes_from_a/））：**DimensionX** 能够利用**可控视频扩散**技术从单张输入图像生成 **3D 和 4D 场景**。该工具由 **Flux Dev** 开发，允许从静态图像创建动态场景和环境。
  - 该项目的官方资源可在 [GitHub](https://github.com/wenqsun/DimensionX) 和 [HuggingFace](https://huggingface.co/wenqsun/DimensionX) 获取，详细信息见其[研究论文](https://arxiv.org/abs/2411.04928)和[项目主页](https://chenshuo20.github.io/DimensionX/)。此外还提供了一个用于部署的 **Docker template**。
  - 用户讨论了其在 **3D 建模软件**中的潜在应用，建议与 **Blender** 和 **Unity** 集成，类似于 **Nvidia NeRF** 但仅需单张图像。有人提到将其与**摄影测量软件**结合用于环境创建。
  - 项目中的 **4D** 一词指代**时间**作为第四维度，本质上是创建具有**时间动画的 3D 场景**。用户还对工作流过程和实现细节表达了关注。


**主题 2. Claude 的性能问题与速率限制引发用户不满**

- **新版 Claude Sonnet 3.5 正在经历“精神崩溃”？** ([Score: 43, Comments: 79](https://reddit.com/r/ClaudeAI/comments/1gqnom0/the_new_claude_sonnet_35_is_having_a_mental/)): **Claude Sonnet 3.5** 用户报告在过去 **72 小时**内性能显著下降，与之前的表现相比，**code quality**、**response coherence** 以及**整体输出质量**明显恶化。这种退化在多种 prompting 方式和以前运行良好的历史 prompts 中表现一致，其中 **coding tasks** 被特别强调为重灾区。
  - 多位开发者报告 **Claude** 错误地默认使用 **React** 解决方案，而不顾指定的框架（**Angular**, **ESP8266**），一位用户指出 *"在我的任何文件或 prompts 中，React 都不是项目的组成部分"*。
  - 用户观察到响应模式不断恶化，包括简短的要点、重复的建议以及无法处理基础的代码修改。一位此前使用 Claude *"构建并发布了多个应用"* 的开发者指出，即使是简单的编程任务，性能也出现了显著下降。
  - 根据一条引用 **Lex Fridman** 采访 **Anthropic** CEO 的评论，大型 AI 实验室有时会通过 **quantization** 来降低模型质量以削减成本（降幅达 **200-400%**），尽管这通常影响的是 Web 界面用户而非 **API** 访问。


- **Claude Pro 限制亟需优先修订** ([Score: 100, Comments: 66](https://reddit.com/r/ClaudeAI/comments/1gq9ihw/claude_pro_limits_needs_revision_now_on_priority/)): **Claude Pro** 用户对 **2 小时使用上限**和频繁的额度限制表示沮丧，这中断了他们的工作流和生产力。用户要求 **Anthropic** 修订当前的 **Pro tier limits**，以更好地满足付费客户的需求。
  - 用户讨论了替代方案，包括使用 **API** 或轮换多个 **Pro accounts**，尽管许多人指出对于他们的使用模式（尤其是处理大型文本文件时），**API costs** 会高得多。
  - 几位作家和开发者分享了在处理大型项目时触碰限制的挫败感，特别是在使用 **Project Knowledge** 和 **artifacts** 等功能时。一位用户报告在处理 **80k words** 的世界观设定文件时，每天触碰限制 *"4 次"*。
  - 多位用户提到在达到 **Claude** 的限制时将 **ChatGPT** 作为备选方案，尽管他们更倾向于 Claude 的能力。一些用户因这些限制取消了订阅，有人建议将价格定在更现实的 *"$79.99 per month"*。


**主题 3. Gemini 现可通过 OpenAI API 库访问**



- **[Gemini 现在可以从 OpenAI 库访问了。什么情况？](https://developers.googleblog.com/en/gemini-is-now-accessible-from-the-openai-library/)** ([Score: 172, Comments: 41](https://reddit.com/r/OpenAI/comments/1gq5zz6/gemini_is_now_accessible_from_the_openai_library/)): **Google** 宣布可以通过 **OpenAI Library** 访问 **Gemini**，尽管该帖子缺乏关于实现或功能的具体细节。该帖子对这种集成的含义和目的表示困惑。
  - **Google** 的 **Gemini API** 现在接受通过 **OpenAI API client library** 发送的请求，实现仅需三处改动：**model name**、**API key** 以及将 **endpoint URL** 改为 `generativelanguage.googleapis.com/v1beta`。这一适配遵循了行业标准，因为许多 **LLM** 提供商都支持 **OpenAI API** 格式。
  - **OpenAI library** 本身保持不变，因为它是 endpoint-agnostic（端点无关）的，所有的修改都在 **Google** 的服务端完成，以接受 **OpenAI** 格式的请求。这使得开发者可以轻松在不同提供商之间切换，而无需大规模重写代码。
  - 多位评论者澄清这并非公司间的合作，而是 **Google** 实现了对已确立的标准 **API** 格式的兼容。**OpenAI API** 已成为 **LLM** 交互的 *"事实标准"*。


**主题 4. Greg Brockman 在领导层变动中重返 OpenAI**

- **[OpenAI 联合创始人 Greg Brockman 重返 ChatGPT 开发商](https://indianexpress.com/article/technology/artificial-intelligence/openai-co-founder-greg-brockman-returns-to-chatgpt-maker-9666990/)** ([Score: 55, Comments: 5](https://reddit.com/r/OpenAI/comments/1gq3un6/openai_cofounder_greg_brockman_returns_to_chatgpt/)): **Greg Brockman** 在休假三个月后已重返 **OpenAI**，他在 **X** 上宣布：“我人生中最长的假期结束了。回到 @OpenAI 继续建设！”，同时正与 **CEO Sam Altman** 合作创建一个专注于技术挑战的新角色。此次回归正值这家由 **Microsoft** 支持的公司发生重大领导层变动之际，包括 **Mira Murati**、**John Schulman** 和 **Ilya Sutskever** 的离职，与此同时 **OpenAI** 正在与 **Broadcom** 合作开发其首款 **AI 推理芯片**。
  - [{'id': 'lwwdcmr', 'author': 'ManagementKey1338', 'body': 'Indeed. I didn’t give him an offer. The man wants too much money.', 'score': 4, 'is_submitter': False, 'replies': []}]


**主题 5：主要 AI 公司面临扩展挑战**



- **[OpenAI、Google 和 Anthropic 在构建更先进 AI 方面陷入困境](https://www.bloomberg.com/news/articles/2024-11-13/openai-google-and-anthropic-are-struggling-to-build-more-advanced-ai)** ([Score: 119, Comments: 114](https://reddit.com/r/OpenAI/comments/1gqfz7l/openai_google_and_anthropic_are_struggling_to/)): **OpenAI**、**Google** 和 **Anthropic** 在开发超越当前能力的更复杂 AI 模型时遇到了技术和资源限制。标题表明主要的 AI 公司面临扩展（scaling）挑战，尽管在没有更多背景信息的情况下，无法确定这些限制的具体细节。
  - **Meta** 报告称模型训练没有出现收益递减，仅因 **算力限制** 而停止。新的 **Nvidia Blackwell** 系列为 Transformer 提供了 **8 倍性能**，而 **OpenAI** 在 **SORA**、**高级语音模式** 和 **o1** 方面继续取得进展。
  - 各公司面临 **训练数据可用性** 的挑战，需要超越“更多数据、更多参数”范式的新架构方法。目前的开发领域包括 **语音**、**视觉**、**图像**、**音乐** 和 **横向集成**。
  - 未来的 AI 发展可能需要新的数据源，包括 **智能眼镜**、**实时生物识别数据** 以及针对特定应用的专门模型。该领域正在经历一些人所描述的 **炒作周期 (Hype Cycle)** 顶峰，正走向潜在的“**幻灭低谷期 (Trough of Disillusionment)**”。


---

# AI Discord 简报

> 由 O1-preview 生成的摘要之摘要的摘要

**主题 1. 新 AI 模型撼动格局**

- [**Qwen Coder 模型引发热议**](source_url)：在多个社区中，开发者们都在热烈讨论 **Qwen Coder 模型**，积极测试其性能并分享基准测试结果。该模型在代码生成任务中表现出巨大的潜力，引发了对其潜在影响的关注。
- [**UnslothNemo 12B 为冒险者发布**](https://openrouter.ai/thedrummer/unslopnemo-12b)：专为**冒险写作**和**角色扮演**定制的 **UnslothNemo 12B** 模型已发布。目前提供限时**免费版本**，邀请用户沉浸式体验故事创作。
- [**Aider v0.63.0 实现自我编程**](source_url)：**Aider** 的最新版本声称其 **55%** 的新代码是由其自身编写的。通过增加对 **Qwen 2.5 Coder 32B** 的支持并改进异常处理，**Aider v0.63.0** 在 AI 辅助开发方面迈出了一大步。

**主题 2. AI 工具与集成增强工作流**

- [**AI 编程工具强强联手**](https://supermaven.com/blog/cursor-announcement)：**Supermaven** 已加入 **Cursor**，共同打造强大的 AI 代码编辑器。双方旨在增强 AI 辅助编程功能，提高全球开发者的生产力。
- [**Windsurf 编辑器引起轰动**](https://codeium.com/windsurf)：**Codeium** 推出了 **Windsurf Editor**，这是首个将 AI 协作与独立任务执行相结合的 Agentic IDE。用户对其保持开发者心流和提升编程效率的潜力感到兴奋。
- [**LM Studio 关注 Text-to-Speech 集成**](source_url)：用户对在 **LM Studio** 中集成 **Text-to-Speech (TTS)** 功能表现出浓厚兴趣。开发团队认可了这一需求，并正在探索增强平台交互性的可能性。

**主题 3. 基准测试对决：模型接受考验**

- [**视觉语言模型在机器人领域展开对决**](https://arxiv.org/abs/2411.05821)：一篇新的研究论文对 **GPT-4** 等 **Vision, Language, & Action Models** 在机器人任务上的表现进行了基准测试。该研究评估了模型在 **20 个真实任务**中的表现，突出了多模态 AI 的进步。
- [**Qwen 2.5 Coder 对阵 GPT-4：巨头之战**](https://youtu.be/Xs0EkLYu6hw)：爱好者们将 **Qwen 2.5 Coder 32B** 与 **GPT-4** 及 **Claude 3.5 Sonnet** 进行了对比，争论哪个模型在代码生成方面更胜一筹。在消费级硬件上令人印象深刻的生成速度引发了进一步关注。
- [**ChatGPT 日期准确；其他模型表现滞后**](source_url)：用户注意到 **Gemini** 和 **Claude** 等模型经常在当前日期上出错，而 **ChatGPT** 则能保持准确的日期感知。这种差异归功于 ChatGPT 卓越的 System Prompt 配置。

**主题 4. 社区对 AI 趋势表示担忧**

- [**Perplexity 用户因广告威胁弃用**](https://www.perplexity.ai/hub/blog/why-we-re-experimenting-with-advertising)：**Perplexity AI** 引入了广告，引发了用户的强烈抵制，用户认为其订阅费用应免除广告。社区正在等待关于广告将如何影响 **Pro** 版本的官方说明。
- [**AI 泡沫即将破裂吗？**](https://chrisbora.substack.com/p/the-ai-bubble-is-about-to-pop-heres)：一篇极具煽动性的文章警告称 **AI 泡沫即将破裂**，将 **6000 亿美元**的 GPU 巨额投资与微薄回报比作当年的互联网泡沫破裂。该文章引发了关于当前 AI 投资可持续性的辩论。
- [**AI21 Labs 弃用模型，用户感到愤怒**](source_url)：**AI21 Labs** 在弃用了许多用户依赖近两年的旧模型后，面临用户的挫败感。用户对新模型的质量以及对未来再次弃用的担忧日益增长。

**主题 5. 技术挑战推动开发者创新**

- [**Triton 解决微型 Tensor 难题**](https://github.com/triton-lang/triton/issues/5138)：使用 **Triton** 的开发者正在针对 16 以下的小尺寸优化 **GEMM kernels**，解决效率挑战并分享提升矩阵计算性能的解决方案。
- [**torch.compile() 引发内存困扰**](source_url)：用户报告称使用 **torch.compile()** 可能会使峰值内存占用增加 **3-16%**，导致动态形状模型出现 **Out-of-memory 错误**。社区正在讨论管理内存的 Profiling 技术。
- [**tinygrad 社区共同修复 Bug**](https://github.com/tinygrad/tinygrad/pull/7675)：**tinygrad** 团队协作修复了无符号 Tensor 的 **min() 函数** 中的一个 Bug。通过分享见解和代码审查，他们展示了开源协作在改进 AI 框架方面的力量。

---

# 第一部分：Discord 高层级摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen Coder 模型部署**：成员们讨论了 **Qwen Coder 模型**目前的开发和测试情况，对其性能和潜在的评估指标表示关注。提到 Unsloth 上已提供相关文件和修复程序，并建议运行类似于其他模型的评估。
  
  - 讨论强调了 **Qwen Coder** 已做好部署准备，社区成员提议利用提供的资源将其与现有模型进行基准测试。
- **多 GPU 训练限制**：用户探讨了使用多块 GPU 训练 **Qwen 2.5** 等大模型的潜力，特别提到了 **MI300X** 和 **VRAM** 需求。有人指出，由于显存效率的原因，**Unsloth** 在单 GPU 设置下可能比多 GPU 配置更高效。
  
  - 社区辩论了多 GPU 训练的可扩展性，一些人主张增加并行性，而另一些人则指出了大规模模型训练中固有的内存管理挑战。
- **Gemma 2B RAM 使用问题**：用户讨论了在使用 **Gemma 2B** 运行较长时间任务时 **RAM 使用量持续增加**的问题，质疑评估步骤是否可能影响性能。一位成员建议以 **0 steps** 进行训练，以减轻过度的资源消耗。
  
  - 建议优化训练配置以减少 RAM 开销，确保在长时间运行期间性能更加稳定。
- **用于长期记忆的 RAG**：提出了关于 **RAG (Retrieval-Augmented Generation)** 的咨询，并征求用户经验和关于将其用于长期数据保留的指导。一位用户推荐 **Dify** 作为实现 RAG 的简单替代方案。
  
  - 社区成员分享了利用 RAG 的各种方法，强调 **Dify** 是将检索系统集成到生成工作流中的用户友好型解决方案。
- **Optillm 发布增强功能**：[**Optillm**](https://github.com/codelion/optillm) 的最新版本引入了一个本地推理服务器，允许加载任何 **HF model** 和 **LoRA adapters**，增强了微调后的 Unsloth 模型的可用性。此更新还支持在推理过程中动态切换 adapter，并支持 **cot_decoding** 和 **entropy_decoding** 等高级解码技术，同时利用标准的 **OpenAI client SDK**。
  
  - 用户赞扬了 **Optillm** 的新功能，指出这些增强功能为模型推理过程带来了更高的灵活性和改进的工作流集成。

 

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Qwen 模型输出不稳定**：用户报告称 **Qwen** 模型在生成文本方面的性能差异很大，与 **Ollama** 等其他模型的对比表明，Qwen 的回复经常可能出现幻觉或缺乏质量。
  
  - 建议调整重复惩罚（repetition penalty）和调整 token 长度等参数以提高输出质量。
- **介绍用于检索的 LightRAG**：分享了一篇详细介绍 **LightRAG** 的文章，其中包括将 Naive RAG 与本地、全局和混合方法进行对比的代码评估。
  
  - 作者旨在强调在各种检索任务中使用 LightRAG 的优势。阅读全文请点击[此处](https://www.linkedin.com/posts/isham-rashik-5a547711b_introducing-lightrag-a-new-era-in-retrieval-activity-7262085232743342080-xgdo?utm_source=share&utm_medium=member_desktop)。
- **用于时间序列预测的 Sulie 基础模型**：**Sulie** 是一种用于时间序列预测的新基础模型，旨在简化 LoRA 微调的自动化和协变量支持。
  
  - 团队寻求反馈，并鼓励用户在 [GitHub](https://github.com/wearesulie/sulie) 上查看他们的工作，幽默地通过将 **zero-shot** 性能问题比作“巧克力茶壶”来强调数据团队面临的常见挫折。
- **机器人领域 VLA 模型的基准测试**：发布了一篇名为 **Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks** 的合作研究论文，旨在评估 **GPT4o** 等 VLA 模型的性能。
  
  - 这项工作代表了针对新型多模态动作模型类别的更广泛基准测试的初始阶段。更多详情请访问[网站](https://multinet.ai/static/pages/Multinetv01.html)或 [GitHub](https://github.com/ManifoldRG/MultiNet/tree/main)。
- **SDXL Lightning 模型展示了快速图像生成**：**SDXL Lightning** 或 **sd1.5 models** 可以在标准 GPU 上仅用几秒钟生成图像，使其成为基于 prompt 的图像创建的理想选择。
  
  - 正如尝试这些配置的用户所分享的，**turbo/lightning/lcm** 等变体可以在强大的硬件上实时生成图像。

 

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity AI 订阅模式受到审视**：用户正在评估 **Perplexity Pro** 订阅，在引入广告的背景下质疑其价值，许多人表示如果加入广告将打算取消订阅。
  
  - 关于 **Pro** 版本是否会出现广告的不确定性日益增加，导致用户寻求 **Perplexity** 团队的官方澄清。
- **寻求 Perplexity 广告实施的澄清**：成员们不确定 **Pro** 订阅中是否会包含广告，这引发了确认请求，以了解其对用户体验的影响。
  
  - 社区强调 **Perplexity** 需要就广告整合进行透明沟通，以维持信任和订阅价值。
- **Perplexity 中持续存在的模型选择问题**：用户报告在 **Perplexity** 中选择不同模型时存在持续性问题，尽管选择了备选方案，系统仍默认使用 **GPT-4o**。
  
  - 这一故障干扰了依赖于稳定访问 **Claude** 等各种模型的 **Pro** 订阅者的工作流。
- **探索分形机器学习（Fractal Machine Learning）增强**：提议使用 **Fractal Machine Learning** 来提升 AI 性能，并讨论了其在语言模型中的潜在应用以及与领域专家的合作。
  
  - 社区成员正在分享资源，并对整合分形概念以推进机器学习技术表现出兴趣。
- **Perplexity AI 模型的差异化因素**：一项深入对比强调了 [**Perplexity AI**](https://www.perplexity.ai/search/how-is-perplexity-ai-different-PF1ebdmMSci1d2dIu6UCiQ) 如何通过独特功能和增强的用户体验在 AI 领域脱颖而出。
  
  - 讨论集中在可能影响 AI 工程师为项目选择 AI 工具时偏好的关键区别。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **缓解梯度下降中的鞍点问题**：在关于 [**gradient descent** 优化](https://arxiv.org/abs/1406.2572) 的讨论中，参与者强调，使用 **noised gradient descent** 时，**saddle points**（鞍点）的影响较小，确保优化器即使在鞍点存在的情况下依然有效。
  
  - 然而，一些成员强调在**高维场景**中，鞍点仍可能出现，这表明其普遍性并未完全消除。
- **批归一化（Batch Normalization）技术的演进**：围绕 **Batch Normalization** 及其替代方案的辩论非常激烈，深入探讨了 **Batch Norm** 的持续相关性，特别是当其作为 **Ghost Batch Norm** 实现时。
  
  - 批评指出 **Batch Norm** 的有效性随 **batch size** 而变化，呼吁对其效率和最佳应用条件进行更多研究。
- **视觉语言动作模型（Vision Language Action Models）的进展**：一项新的研究发布展示了在机器人任务中对 [**Vision Language Action models** 进行基准测试](https://x.com/HarshSikka/status/1856739777208574151)，涉及知名机构并提供了极具前景的见解。
  
  - 鼓励参与者对该工作提供反馈，并探索提供的 [YouTube 视频](https://x.com/HarshSikka/status/1856739777208574151)和项目链接，以更深入地了解模型及其应用。
- **将 DagsHub 与 GPT-NeoX 集成**：提议了将 **DagsHub** 与 **GPT-NeoX** 集成的潜在价值，寻求社区关于增强平台能力的见解。
  
  - 对 **AnthropicAI** 框架的查询显示，他们使用的是专有系统，不对外公开。
- **重新思考梯度下降步长**：[**Grimmer** 教授](https://x.com/prof_grimmer/status/1679846891171766272)挑战了传统观念，即 **gradient descent** 需要恒定的 **1/L** 步长以实现最佳收敛。
  
  - 他的研究结果详见[他的论文](https://arxiv.org/abs/2307.06324)，表明在 **(0, 2/L)** 范围内的*周期性长步长*可以带来更好的收敛结果。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **UnslopNemo 12B 为冒险写作发布**：**UnslopNemo 12B** 模型专为**冒险写作**和**角色扮演场景**定制，现已在 [UnslopNemo 12B](https://openrouter.ai/thedrummer/unslopnemo-12b) 上线。
  
  - 免费变体可通过 [UnslopNemo 12B Free](https://openrouter.ai/thedrummer/unslopnemo-12b:free) 访问 **24 小时**，支持请求请发送至 [Discord](https://discord.gg/fVyRaUDgxW)。
- **Mistral 和 Gemini 获得参数增强**：**Mistral** 和 **Gemini** 模型已更新，包含 **Frequency Penalty** 和 **Presence Penalty** 参数，增强了其可配置性。
  
  - 此外，**Mistral** 现在提供 **seed adjustments** 工具，提高了输出的一致性。
- **对 Tool Calling 功能的困惑**：用户在使用 **OpenRouter** 的 **tool calling** 功能时遇到问题，因为启用该功能并未像预期那样影响 **token usage**。
  
  - 讨论强调需要更清晰的实现指南，以在模型交互中充分利用 **tool calling**。
- **高 Token 处理量引发定价讨论**：一位在利基市场为 AI 聊天机器人管理每日超过 **300 万 Token** 的用户询问了针对高交易量 Token 处理的潜在**降价**方案。
  
  - 这反映了对满足专业应用中大规模使用的可扩展定价模型的日益增长的需求。
- **Custom Provider Keys 请求激增**：多位成员请求访问 **Custom Provider Keys**，表明对利用此功能进行定制化集成有浓厚兴趣。
  
  - 社区对话包括各种诉求，强调了 **Custom Provider Keys** 对于多样化项目需求的重要性。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider v0.63.0 发布新功能**：**Aider v0.63.0** 版本引入了对 **Qwen 2.5 Coder 32B** 的支持，并增强了对 **LiteLLM exceptions** 的处理，提升了整体可用性。
  
  - 此版本中 **55%** 的代码由 **Aider** 编写，展示了显著的自我开发能力。
- **Aider 的 VSCode 和 Neovim 扩展发布**：**Aider** 的新 **VSCode** 和 **Neovim** 扩展已发布，具有 Markdown 预览、文件管理和聊天记录功能，鼓励社区贡献。
  
  - 这些扩展旨在提高 **Aider** 在各个平台上的实用性，促进开发者之间的协作。
- **SupermavenAI 与 Cursor 合作**：**Cursor** 宣布 **SupermavenAI** 加入其团队，以增强研究和产品能力，旨在将 **Cursor** 打造为**产品巨头**。
  
  - 该合作伙伴关系通过 [Twitter](https://x.com/cursor_ai/status/1856427424927625679) 公布，强调了协作创新的计划。
- **Aider 添加 Qwen 2.5 Coder 支持**：**Aider** 现在支持 **Qwen 2.5 Coder 32B**，将先进的编码能力集成到平台中。
  
  - 此次更新促进了代码辅助能力的提升，并使 **Aider** 的功能与当代编码标准保持一致。
- **Aider 的 OpenRouter 提供商配置技巧**：关于为 **Aider** 配置 **OpenRouter** 的讨论包括指定提供商偏好和创建模型元数据文件以管理成本和上下文大小。
  
  - 用户分享了平衡提供商使用的策略，并强调了理解 **OpenRouter** 负载均衡机制的重要性。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **优化 LM Studio 的 Quantization 大小**：成员们讨论了 **Quantization 大小**的影响，指出较小的大小会导致压缩率增加，而较大的大小可能需要拆分为多个部分。
  
  - *Heyitsyorkie* 总结道，较高的 Quantization 大小可以确保更好的性能，且不会产生显著损失。
- **将 TTS 与 LM Studio 集成**：**LM Studio** 用户有兴趣将该平台连接到 Text-to-Speech (TTS) 功能。
  
  - 回复指出，关于集成此类功能的对话正在进行中，但尚未提供时间表。
- **解决 Qwen 2.5 性能问题**：一位用户报告了 **Qwen 2.5** 的问题，具体表现为仅接收到自动补全响应，但随后指出它已开始正常工作。
  
  - 其他人建议确保配置正确，并探索模型选项以优化性能。
- **用于 Llama.cpp 集成的 Python 脚本**：有人请求提供一个 **Python 脚本**，以便将最新的 **Llama.cpp** 侧载到 **LM Studio** 中，强调了对此类功能的需求。
  
  - 参与者承认了社区长期以来的期待，并提到正在努力将其变为现实。
- **用于大模型推理的 GPU 组合**：关于同时使用 **12GB 3060** 和 **40GB A800** 进行 **70B 级模型**推理的讨论提出了一个问题：是使用单个 GPU 还是两个都用，并关注扩展如何影响性能。
  
  - 一位成员建议，仅使用 **A800** 可能更有利，因为它可以在 **VRAM** 中容纳模型，而 3060 则不行。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **使用 Dreambooth 训练电影海报**：一位用户正在寻求在 **auto1111** 中使用 **Dreambooth** 训练**电影海报**的**教程**，寻找最新的技术和有效训练的建议。
  
  - 社区建议查看现有资源和指南以简化流程。
- **Animatediff 支持生成视频剪辑**：成员们讨论了使用 **Animatediff** 生成视频剪辑，强调了其通过发布两张图片来创建过渡的能力，尽管较低的分辨率更适合社交媒体。
  
  - 提供了对 **Banodoco 服务器**的推荐，因为他们专注于视频相关工具。
- **Checkpoint 和 LoRa 下载源**：用户分享了指向外部文件托管网站（如 **Google Drive**、**Mega** 和 **Hugging Face**）的链接，用于下载 Checkpoint 文件和 LoRa，同时讨论了 **Civit AI** 的限制和潜在的内容禁令。
  
  - 用户对特定内容类型的移除及其对用户访问的影响表示担忧。
- **解决 Stable Diffusion 中的 Python Torch 错误**：一位用户在为 **Stable Diffusion** 设置 Python 环境时遇到了 **torch** 包错误，被建议卸载当前的 Python 版本并安装 **Python 3.10.11 64bit**。
  
  - 该用户对支持表示感谢，并计划尽快实施建议的解决方案。
- **Discord 访问问题及解决方案**：用户询问如何访问 Discord 服务器的 URL，特别是寻求新的邀请和直接链接，并提到了 **Pixaroma** 社区邀请链接过期的经历。
  
  - 社区为连接所需的 Discord 服务器提供了帮助。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Nous 3 模型性能困惑**：如[此推文线程](https://x.com/thexeophon/status/1856429292504096944?s=61)所示，**Nous 的 70B 模型**性能数据出现了偏差，引发了对所报告的 **MMLU-Pro** 分数有效性的质疑。
  
  - 成员们推测，Prompting（提示词）技术的差异和 Benchmark（基准测试）的不一致可能是影响这些不同数据的因素。
- **AI Agent 工具 'Operator' 发布**：OpenAI 计划推出一款名为 'Operator' 的新 **AI Agent 工具**，用于自动化编写代码和预订旅行等任务。根据[此公告](https://www.bloomberg.com/news/articles/2024-11-13/openai-nears-launch-of-ai-agents-to-automate-tasks-for-users?accessToken=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzb3VyY2UiOiJTdWJzY3JpYmVyR2lmdGVkQXJ0aWNsZSIsImlhdCI6MTczMTUyODYxOCwiZXhwIjoxNzMyMTMzNDE4LCJhcnRpY2xlSWQiOiJTTVdOQURUMEcxS1cwMCIsImJjb25uZWN0SWQiOiJFODA3NUYyRkZGMjA0NUI2QTlEQzA5M0EyQTdEQTE4NiJ9.TTJZiuo4Nk2U295FHBFsxeN0YGznZJ32sHnNReQmEjM)，该工具预计将于 1 月发布。
  
  - 该工具旨在通过在各种场景下代表个人采取行动来提高用户生产力。
- **JanusFlow 模型介绍**：**JanusFlow 模型**作为一种新能力被引入，它将自回归 LLMs 与 rectified flow 相结合，用于图像理解和生成，详见[此贴](https://x.com/deepseek_ai/status/1856552494379520510)。
  
  - JanusFlow 旨在实现鲁棒、直接且灵活，将影响该领域未来 AI 模型的发展。
- **拦截 Jailbreaks 的自适应技术**：Anthropic 的新研究引入了自适应技术，以便在检测到新类别的 **Jailbreak（越狱）**时对其进行**快速拦截**，正如他们在[此处](https://arxiv.org/abs/2411.07494)的论文中所讨论的那样。
  
  - *确保完美的越狱鲁棒性非常困难，* 这凸显了保障 AI 模型安全的挑战。
- **视觉语言模型 (VLMs)**：成员们讨论了**视觉语言模型 (VLMs)**，引用了 [Finbarr 的博客](https://www.artfintel.com/p/papers-ive-read-this-week-vision)和一篇关于 [VLM 推理成本](https://x.com/goyalsachin007/status/1856004116012798355)的帖子。
  
  - 关键话题包括由于 500 多个图像 Token 导致的高计算成本，以及最近像 **Pixtral** 和 **DeepSeek Janus** 这样改进了从图像中提取文本的模型。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **KATT 助力播客生产力飞跃**：一位成员将 **KATT** 集成到他们的播客工作流中，通过在 **KATT** 训练**两年**后使用修改后的 System Prompt，制作出了时长超过 **90 分钟**且经过事实核查的节目。
  
  - 这种集成简化了制作流程，增强了主持人在超长播客节目中保持准确性和深度的能力。
- **NotebookLM 外部共享功能被取消**：一位成员询问是否可以将 **NotebookLM** 内容分享到其 Google Organization 之外，得到的确认是由于管理员施加的限制，**无法进行外部共享**。
  
  - 进一步的讨论揭示了个人账户的局限性，强调了在处理 **NotebookLM** 数据时遵守组织政策的重要性。
- **Gemini 防护：NotebookLM 数据安全**：针对上传到 **Gemini** 的数据安全性提出了担忧，澄清说明 **付费账户**可确保数据安全，而免费账户则不然。
  
  - 成员们敦促在上传敏感信息时要谨慎，强调了在平台上保持**机密性**以防止潜在泄露的必要性。
- **利用 NotebookLM 成功进行摘要**：一位用户寻求使用 **NotebookLM** 为大学文献综述总结文本的技巧，得到的建议是利用**合成数据集**来保护敏感数据。
  
  - 这种方法旨在提高摘要的有效性，同时确保在此过程中维护隐私标准。
- **播客生成中的格式失败**：用户讨论了从特定来源生成播客时的挑战，特别是遇到了 **.md 文件格式**的问题。
  
  - 建议包括切换到 **PDF** 或 **Google Docs** 格式，这成功解决了用户在播客生成焦点方面的问题。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Supermaven 加入 Cursor**：Supermaven 已正式加入 [Cursor](https://supermaven.com/blog/cursor-announcement)，以增强其 AI 编程编辑器的能力。此次合作利用 Supermaven 的 AI 辅助功能来提升软件开发体验。
  
  - Anyan sphere 收购了 Supermaven 以增强 Cursor，交易细节尚未披露。社区反应不一，在注意到 Supermaven 此前的高效表现时，也对这一转变表示惊讶。
- **Codeium 发布 Windsurf 编辑器**：Codeium 推出了 [Windsurf Editor](https://codeium.com/windsurf)，这是首个将 AI 协作与独立任务执行相结合的 Agentic IDE，旨在保持开发者的心流。
  
  - 尽管初步印象良好，一些用户指出 Windsurf Editor 在某些方面可能尚未超越 Copilot 等成熟工具。此外，该编辑器无需排队或邀请即可使用，强调了用户包容性。
- **Perplexity 引入赞助广告**：Perplexity 正在其平台上尝试广告，在搜索结果旁引入了“赞助后续问题”。他们与 [Indeed](https://www.indeed.com) 和 [Whole Foods](https://www.wholefoods.com) 等品牌合作，为其 AI 驱动的搜索引擎变现。
  
  - 此举旨在建立一个可持续的收入共享计划，解决仅靠订阅费用不足的问题。
- **Mira Lab 组建新 AI 团队**：由前 OpenAI CTO Mira Murati 发起的 Mira Lab 正在组建一支专注于 AI 技术的新团队，据报道至少有一名 OpenAI 研究员加入了该项目。
  
  - 该实验室旨在利用其创始成员的专业知识开展雄心勃勃的项目。
- **RAG 将超越问答阶段**：正如 [Jason Liu](https://jxnl.co/writing/2024/06/05/predictions-for-the-future-of-rag/) 在文章中所强调的，越来越多的人推测检索增强生成（RAG）将在未来几个月内从主要的 Q&A 应用转向更复杂的报告生成。
  
  - 普遍观点认为，RAG 的演进将增强公司在文档和报告中利用 AI 的方式。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton Kernel 功能与 Conda 问题**：开发者正在解决使用 Conda 环境时 Triton 中的 [libstdc++](https://github.com/triton-lang/triton/issues/5136) 兼容性问题，旨在解决 `torch.compile` 操作期间遇到的崩溃。
  
  - 讨论内容包括针对较小尺寸优化 [GEMM kernel 设计](https://github.com/triton-lang/triton/issues/5138)，以及解决 [warp 内存对齐错误](https://github.com/triton-lang/triton/issues/5136)，以增强 Triton 的稳定性和性能。
- **torch.compile 对内存使用的影响**：用户报告称 **torch.compile()** 会导致峰值内存使用量增加 **3-16%**，从而引发 **out-of-memory (OOM)** 错误，尤其是在处理 **dynamic shapes** 时。
  
  - 建议使用 **nsys** 和 **nvtx** 范围进行 Profiling 以分析 GPU 内存分配，尽管尚不确定在没有 `reduce-overhead` 标志的情况下，PyTorch 中的 **CUDA graphs** 是否会加剧内存消耗。
- **MI300X 实现 600 TFLOPS FP16 峰值吞吐量**：性能基准测试显示，**MI300X** 在 **FP16** 操作中可达到高达 **600 TFLOPS** 的吞吐量，尽管尝试通过 **CK** 优化突破 **800 TFLOPS** 尚未成功。
  
  - Lei Zhang 和 Lixun Zhang 的 [YouTube 演讲](https://youtu.be/Lbm08twNTAQ?si=6Vwrkz8W0U2WTpf1&t=243) 强调了 Triton 对 AMD GPU 的支持，展示了围绕 chiplets 的优化策略以提升 GPU 性能。
- **Liger-Kernel v0.4.1 发布，支持 Gemma 2**：[Liger-Kernel](https://github.com/linkedin/Liger-Kernel/releases/tag/v0.4.1) 最新的 **v0.4.1** 版本引入了对 **Gemma 2** 的支持，并修复了 **CrossEntropy** 问题，解决了 fused linear cross entropy 中的 softcapping 问题。
  
  - 改进还包括对 **GroupNorm** 的修复，有助于实现更高效的操作，并验证了更新后 kernel 的稳健性。
- **ThunderKittens 更新：DSMEM 限制与同步**：**ThunderKittens** 的更新显示，**H100** GPU 仅支持整数类型的 **DSMEM** reduction，引发了关于优化 semaphore 操作和同步以防止挂起的讨论。
  
  - 未来的 pull requests 旨在完成整数测试代码，增强 kernel 在 cooperative groups 和 semaphore 同步场景下的可靠性和性能。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 的多节点 FSDP 分布式方法**：用户询问了 Tinygrad 目前的分布式计算策略，特别是关于 **FSDP** 的处理和对**多节点**设置的支持。他们参考了 [multigpu training tutorial](https://mesozoic-egg.github.io/tinygrad-notes/multigpu.html) 以获取详细见解。
  
  - 另一位用户提到 **FSDP** 的一个开放悬赏（bounty）可以作为潜在资源，并讨论了当前实现的扩展性挑战。
- **Tinygrad 在云端的数据处理**：讨论强调，虽然云能力允许利用跨不同机器的数千个 GPU，但最佳性能取决于**快速连接**和有效的 **all-reduce** 实现。
  
  - 有用户对在训练运行期间由单台机器编排数据管理和处理的效率表示担忧。
- **Tinygrad 中的设备间通信**：George Hotz 指出，**设备间通信**是通过 Tinygrad 的 Buffer 经由 `transfer` 函数管理的，这表明将其扩展到云端设置可能比较容易。
  
  - 他幽默地提到，这只需几行代码即可完成，暗示了实现的简单性。
- **Tinygrad 分片中的性能优化**：讨论了是否有必要澄清用户是**机器分片（machine-sharded）**还是**云分片（cloud-sharded）**，以防止在较慢的同步操作期间出现意外的性能问题和成本。
  
  - 对话强调了高效数据处理策略对于在不同配置下保持性能水平的重要性。
- **修复 tinygrad 中无符号 Tensor min() 的 Bug**：一位用户发现了无符号 Tensor 在计算包含零的最小值时 **min()** 函数的 Bug，并建议通过翻转（flips）来解决。他们参考了 [PR #7675](https://github.com/tinygrad/tinygrad/pull/7675/commits/6c1092cefc98c87edfe9516f3887d6789351140f)。
  
  - *Rezvan* 提交了一个带有失败测试用例的 PR，并提到了由于潜在的 **infs** 和 **nans** 导致的复杂性。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **增强 AI 模型的日期准确性**：讨论显示，像 **Gemini** 和 **Claude** 这样的模型经常提供错误的当前日期，而 **ChatGPT** 则保持着准确的日期意识。[讨论链接](https://discord.com/channels/974519864045756446/998381918976479273/1305997985397608549)。
  
  - 一位用户将 ChatGPT 的准确性归功于其卓越的系统提示词（system prompt）配置，使其能够在各种语境下更好地推断日期。
- **ChatGPT o1-preview 展示出更高的创造力**：与早期版本相比，**ChatGPT o1-preview** 因其增强的创造力和个性化回答而获得积极反馈。[反馈线程](https://discord.com/channels/974519864045756446/1001151820170801244/1306032001496907836)。
  
  - 用户赞赏其预测输入的能力，这有助于提供更具定制化的交互体验。
- **在 LLM 中实现 Scratchpad 技术**：成员们正在探索将 **scratchpad 技术**作为一种伪 CoT 方法使用，允许 LLM 在生成解决方案时阐明其思考过程。[讨论链接](https://discord.com/channels/974519864045756446/1046317269069864970/1306011250769268850)。
  
  - 人们对将 scratchpads 集成到结构化输出（structured outputs）中以提高文档和工作流一致性充满热情。
- **移动端复制粘贴功能的挑战**：移动平台上持续存在的**复制粘贴**问题正在影响用户体验，该问题已持续数周。[问题报告](https://discord.com/channels/974519864045756446/998381918976479273/1305997985397608549)。
  
  - 用户正在寻求有效的解决方案来恢复功能并增强移动端交互能力。
- **使用 VPN 绕过访问限制**：讨论强调了使用 **VPN** 绕过互联网限制的合法性，突出了它们在维持访问方面的作用。[对话线程](https://discord.com/channels/974519864045756446/1001151820170801244/1306032001496907836)。
  
  - 参与者指出，当前的封锁配置对于为了预期目的而使用 VPN 的用户可能无效。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **exllamav2 提升 MAX 推理能力**：成员们强调 [exllamav2 GitHub 项目](https://github.com/turboderp/exllamav2) 是增强 **MAX** 上 LLM 推理的宝贵资源，并赞扬了其简洁且优化的代码库。
  
  - 关键特性包括 **针对 AMD 的 ROCM 支持** 以及对多模态模型的高效处理，使 exllamav2 成为与 MAX 平台深度集成的有力候选者。
- **Mojo JIT 编译器优化**：社区讨论了发布 **Mojo JIT 编译器** 的可行性，重点在于确保紧凑的二进制文件大小以及与预编译二进制文件的互操作性。
  
  - 一位成员强调，虽然 **MLIR 可以发布**，但编译器对于在不暴露所有依赖应用程序源码的情况下实现原生代码执行至关重要。
- **MAX 平台能力**：**MAX** 被介绍为一套用于构建和部署高性能 AI 流水线的全面 API 和工具集，包含用于模型执行的 **MAX Engine** 等组件。
  
  - 分享了 [MAX 文档](https://docs.modular.com/max/#how-max-works)，展示了其在有效部署低延迟推理流水线方面的能力。
- **Mojo 中 UnsafePointer 的风险**：Mojo 中的 **UnsafePointer** 因可能引发未定义行为而被标记，社区成员详细说明了这会导致内存安全问题。
  
  - 另一位成员指出，与 C/C++ 相比，Mojo 执行更严格的指针规则，旨在最大限度地减少类型混淆（type punning）等风险，并增强整体内存安全性。
- **Mana 项目命名趋势**：成员们幽默地讨论了 **Mana** 这个名字的频繁使用，提到了 [mana.js](https://github.com/bjorn/mana.js/) 和 [3rd-Eden's mana](https://github.com/3rd-Eden/mana) 等项目。
  
  - 对话反映了在项目命名中采用 “Mana” 的趋势，表明了技术社区命名惯例中更广泛的文化影响。

 

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Vocera 在 Product Hunt 上线**：**Vocera** 在 [Product Hunt](https://www.producthunt.com/posts/vocera) 上线，使 AI 开发者能够以 **快 10 倍的速度** 测试和监控语音 Agent。
  
  - 团队正在寻求反馈，以提升 [Vocera 在 AI 社区的知名度](https://www.producthunt.com/posts/vocera)。
- **使用 LlamaIndex 构建 GenAI 流水线**：学习了如何使用 [LlamaIndex](https://x.com/LlamaIndex)、[Qdrant Engine](https://twitter.com/qdrant_engine) 和 [MLflow](https://twitter.com/MLflow) 构建强大的 **GenAI 流水线**，以增强 RAG 系统。
  
  - [分步指南](https://t.co/aZ4GIyGRQM) 涵盖了简化 RAG 工作流、在不同模型版本间保持性能以及优化索引效率等内容。
- **RAG 与报告之争**：一场关于 **RAG**（检索增强生成）与传统报告的辩论展开了，指出在企业中，报告仅占解决问题的 **10%**。
  
  - `@jxnlco` 认为报告更具影响力，并强调 **信息检索** 是生成有效报告的关键。
- **RAG 中的动态章节检索**：在 RAG 中引入了一种新的 **动态章节检索** 技术，允许从文档中检索完整的章节，而不是碎片化的分块（chunks）。
  
  - 正如 [这篇文章](https://t.co/vP2J2arhf4) 中讨论的，该方法解决了社区对多文档 RAG 的担忧。
- **企业环境中的聊天机器人**：成员们观察到，在企业内部，高层管理人员更倾向于 **报告格式**，而非聊天机器人交互。
  
  - 尽管有这种偏好，聊天机器人仍被公认为是进行内部搜索的有效工具。

 

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Rerank API 最佳实践**：用户正在寻求 **v2/rerank API** 中 `query` 字段的**最佳实践**，并指出微小的查询变化会导致 `relevanceScore` 产生显著差异。请参考 [Rerank 最佳实践](https://docs.cohere.com/docs/reranking-best-practices#queries) 以获得最佳的端点性能。
  
  - 示例包括：针对 **'volume rebates'** 的 `query` 获得了约 0.998 的分数，而 **'rebates'** 仅为 0.17，这引发了关于模型对查询语义响应能力的困惑。
- **生产环境 API Key 升级**：一位用户报告称已**升级到生产环境 API Key**，期待在当前问题解决后，**Cohere** 的服务能提供更稳定的体验。
  
  - 此次升级表明了用户对使用 Cohere 产品的承诺，但这取决于当前 API 错误的解决情况。
- **视觉语言动作模型基准测试**：发布了一篇名为 [*Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks*](https://arxiv.org/abs/2411.05821) 的新论文，展示了 **Manifold**、**Georgia Tech**、**MIT** 和 **Metarch AI** 之间的合作。
  
  - 该研究评估了包括 **GPT4o** 在内的新兴 Vision Language Action 模型在 **20 个真实世界任务**中控制机器人的能力。可以在 [Multinet 网站](https://multinet.ai/static/pages/Multinetv01.html) 和 [代码仓库](https://github.com/ManifoldRG/MultiNet/tree/main) 探索更多内容。
- **活动的 ICS 支持**：一位用户强调了实现 **ICS 文件支持**的必要性，以便管理 Discord 服务器上举办的大量活动。
  
  - 该请求受到了成员们的好评，大家纷纷给出正面反馈支持添加此功能。
- **文件内容查看功能**：工具包中引入了一项新功能，可以**查看上传文件的内容**，增强了文件管理能力。
  
  - 该功能受到了成员们的热烈欢迎，他们对改进后的功能表示赞赏。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **发布版本的 Docker 镜像打标签**：**main 分支**的 Docker 镜像已构建完成，并提醒需要为版本发布打上标签。一位成员强调了正确打标签对于有序的版本控制和即将到来的发布的重要性。
  
  - 这一实践确保了每个版本的可追溯性，详见[最新的 Pull Request](https://github.com/axolotl-ai-cloud/axolotl/pull/2051/files)。
- **Qwen2.5 Coder 尺寸见解**：一位成员分享了一个 [YouTube 视频](https://youtu.be/WPziCratbpc)，对比了不同尺寸的 **Qwen2.5 Coder**，详细讨论了它们的性能指标。
  
  - 该视频提供了深入的分析，帮助用户根据特定需求选择合适的模型尺寸。
- **Qwen2.5 在 NVIDIA 3090 上的性能**：**Qwen2.5** 正在 **NVIDIA 3090** 上运行，从而提升了生成速度。这种硬件配置凸显了高性能模型可以实现的性能增益。
  
  - 用户注意到生成时间有了显著改善，强调了高端 GPU 在模型部署中的优势。
- **Qwen2.5 Coder 与 GPT4o 及 Claude 3.5 Sonnet 的对比**：分享了一个名为 [‘Qwen2.5 Coder 32B vs GPT4o vs Claude 3.5 Sonnet’](https://youtu.be/Xs0EkLYu6hw?si=95JJjVKRPknvEUsw) 的 YouTube 视频来对比这些模型。
  
  - 该视频旨在确定其中的优胜模型，并对其能力进行了全面分析。
- **Axolotl 0.5.0 版本发布**：团队宣布发布 **Axolotl 0.5.0 版本**，现在可以通过 `pip install axolotl` 进行安装。更新内容包括改进和新功能，详见 [GitHub 发布页面](https://github.com/axolotl-ai-cloud/axolotl/releases/tag/v0.5.0)。
  
  - 社区成员庆祝了该版本的发布，表达了兴奋之情并承诺支持后续的增强功能。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Nous Research 推出 Forge Reasoning API**：Nous Research 发布了 [Forge Reasoning API](https://nousresearch.com/introducing-the-forge-reasoning-api-beta-and-nous-chat-an-evolution-in-llm-inference/) Beta 版，承诺在 LLM 推理能力方面取得重大进展。
  
  - 这一进展标志着增强 AI 系统内推理过程的关键一步，展示了新模型与优化技术的融合。
- **Nous Chat 迎来升级**：伴随 Forge API，**Nous Chat** 也将进化，整合提升用户交互和可访问性的高级功能。
  
  - 随着这一进化，重点在于通过增强的 LLM 技术和方法论提供更丰富的对话体验。
- **DSPY 对比分析讨论**：成员们讨论了使用 **DSPY** 在特定领域进行 **zero shot** 和 **few shot prompting** 对比分析的经验。
  
  - 一位成员询问其他人关于使用 GitHub 模板来促进此类分析的情况。
- **共享 DSPY 资源**：一位成员分享了 [Colab notebook](https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/intro.ipynb) 的链接，以帮助他人开始使用 DSPY。
  
  - 另一位成员引用了另一个 notebook，并强调了它在自己涉及代码相似性工具的项目中的潜在用途。
- **使用 LLM 方法评估工具**：一位成员提到在尝试使用 **LLM** 创建代码相似性工具时，评估了 **zero shot** 与 **few shot** prompting。
  
  - 他们提到了另一个他们参与开发的 GitHub 资源，用于比较方法和结果。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter 激发社区热情**：成员们对最新的 **Open Interpreter** 更新感到兴奋，特别是 **streamed responses handling**（流式响应处理）功能，它[提升了用户体验](https://discord.com/channels/1146610656779440188/1147665339266650133/1306047445276295188)。
  
  - 一位成员评论道 *'Open Interpreter 太棒了！'*，并引发了关于在未来合作中**构建文本编辑器**潜力的讨论。
- **OpenCoder：革新代码模型**：[OpenCoder YouTube 视频](https://www.youtube.com/watch?v=DurejOD5FTk)展示了 **OpenCoder**，这是一个旨在开发具有高级功能的卓越代码语言模型的开源仓库。
  
  - 观众对 OpenCoder **超越现有模型**的潜力很感兴趣，讨论了其对代码建模领域的影响。
- **预测 AI 泡沫破裂**：一篇帖子警告称 **AI 泡沫即将破裂**，并将其与 **1999 年互联网泡沫**相提并论，特别是在大规模 **GPU 投资**未能产生相应收益方面。
  
  - 文章详细阐述了 **6000 亿美元**的 GPU 支出与仅 **34 亿美元**收入之间的风险，暗示了 AI 行业不稳定的前景。
- **对比 AI 与互联网崩溃**：讨论强调，当前 AI 领域的基础设施建设反映了互联网时代的策略，即公司在没有明确盈利路径的情况下大量投资硬件。
  
  - 随着各公司在没有经过验证的利润途径的情况下追求理论需求，重蹈类似 **Pets.com** 失败覆辙的风险被凸显。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **视觉语言动作模型发布**：一篇名为《[Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks](https://arxiv.org/abs/2411.05821)》的新论文评估了 **Vision Language Models** 在 20 种不同现实任务中的表现，展示了 **Manifold**、**Georgia Tech**、**MIT** 和 **Metarch AI** 之间的合作成果。
  
  - 该工作旨在剖析像 **GPT4o** 这样新兴类别的模型，标志着迈向更广泛的**多模态动作模型（multimodal action models）**基准测试的第一步。
- **Watermark Anything 工具发布**：项目“[watermark-anything](https://github.com/facebookresearch/watermark-anything)”提供了带有局部消息的**水印（watermarking）**官方实现。该模型被指出仅有 **1M 参数**，有可能被快速集成到各种 **AI generators** 中。
  
  - 轻量级的架构使其能够在不同的 AI 生成平台中快速部署，促进无缝集成。
- **EPOCH 58 COCK 模型更新**：一位成员分享了关于 **EPOCH 58 COCK** 的更新，指出在 **60M 参数**下使用 **vit-s** 取得了改进，并增强了模型特性。
  
  - *他们评论道，***腿部正在显现***，且***鸡冠变得更加清晰***，这标志着模型能力取得了积极进展。*
- **机器人学习任务的进展**：讨论强调了**机器人学习任务（Robotic Learning Tasks）**的进展，特别是在应用 **Vision Language Action Models** 来增强**机器人控制（robot control）**和**任务自动化（task automation）**方面。
  
  - 社区成员讨论了在现实机器人系统中部署这些模型的挑战和潜在解决方案，并引用了正在进行的实验和初步结果。
- **AI 生成器性能增强**：参与者讨论了 **AI Generators Performance** 的最新改进，重点关注**模型效率**和**输出质量**的提升。
  
  - 分析了具体的基准测试和性能指标以评估进展，并强调了实际落地应用。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **利用 Tape 进行 Agent 与人类的通信**：一位成员询问关于使用 **Tape** 作为人类与 **Agent** 之间通信媒介的问题，并寻求相关的[文档](https://discord.com/channels/1280234300012494859/1280370030609170494/1306007383268524074)。
  
  - 这引发了关于如何将遇到错误的 **Agent** tape 条目发布到队列的指导请求。
- **分享 TapeAgents 框架资源**：针对 **TapeAgents** 的疑问，一位成员分享了一个 [GitHub 入门 notebook](https://github.com/ServiceNow/TapeAgents/blob/main/examples/intro_clean.ipynb) 和一篇相关的[论文](https://www.servicenow.com/research/TapeAgentsFramework.pdf)。
  
  - *该成员表示他们已经阅读了所有提供的资源*，说明他们已经审阅过建议的材料。

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **Latent Toys 网站上线**：一位成员分享了新创建的 [Latent Toys](https://latent.toys/)，强调这是一个值得关注的项目。
  
  - 该网站是由一位朋友开发的，进一步增加了其重要性。
- **社区关于 Latent Toys 的讨论**：成员们讨论了 [Latent Toys](https://latent.toys/) 的发布，强调了其在社区内的重要性。
  
  - 该公告引发了人们对新网站提供内容的兴趣和好奇。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Gorilla 为 Writer 模型和 Palmyra X 004 提交 PR**：一位成员宣布提交了一个 [PR](https://github.com/ShishirPatil/gorilla/pull/755)，旨在将 **Writer models** 和 **Palmyra X 004** 的支持添加到排行榜中。
  
  - 他们对评审表示了*感谢*，并分享了与该 **PR** 相关的图像预览，突显了社区协作。
- **社区对 Gorilla PR 的响应**：另一位成员迅速响应了 [PR 提交](https://github.com/ShishirPatil/gorilla/pull/755)，表示他们将审查这些更改。
  
  - 他们回复的“*Thank you!*”强调了活跃的社区参与。

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **旧模型弃用 (Legacy Models Deprecation)**：成员们对**旧模型弃用**表示沮丧，称新模型提供的输出质量不如从前。
  
  - 对于依赖旧模型近两年的用户来说，*这次弃用具有巨大的破坏性*。
- **转向开源解决方案**：用户正争相转向**开源解决方案**，尽管他们此前一直愿意为旧模型付费。
  
  - *我们如何确定 AI21 未来不会也弃用新模型？* 这句话突显了他们对未来产品稳定性的担忧。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该社区长时间保持沉默，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该社区长时间保持沉默，请告知我们，我们将将其移除。

---

**Torchtune Discord** 没有新消息。如果该社区长时间保持沉默，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该社区长时间保持沉默，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道详细摘要和链接

{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #**[**general**](https://discord.com/channels/1179035537009545276/1179035537529643040/1305985484962660414) (160 messages🔥🔥):

> - `Qwen Coder Models`
> - `Multi-GPU Training`
> - `Dataset Formatting`
> - `Finetuning VRAM Requirements`
> - `Inference System Tokens`

- **关于 Qwen Coder 模型部署的讨论**：成员们讨论了 Qwen Coder 模型的当前开发和测试情况，部分成员对其性能和潜在的评估指标表示关注。
  
  - 有人提到 Unsloth 上已提供相关文件和修复补丁，并建议进行类似于其他模型的评估。
- **多 GPU 训练限制**：用户探讨了使用多 GPU 训练 Qwen 2.5 等大模型的可能性，特别提到了 MI300X 和 VRAM 需求。
  
  - 有人指出，由于显存效率原因，Unsloth 在单 GPU 设置下可能比多 GPU 配置更有效率。
- **数据集格式化与输入处理**：讨论了如何有效地为 Finetuning 格式化数据集，特别是关于分隔符 Token 以及输入/输出的区分。
  
  - 建议包括使用特殊的 Token，如 `### response` 或 `---`，以告知模型何时在用户输入和模型输出之间切换。
- **大模型微调与 VRAM 需求**：用户询问了微调 405B 模型所需的 VRAM，一些人提出了涉及高 VRAM GPU 的配置方案。
  
  - 建议在 RunPod 等平台上测试配置的实用性，并讨论了 Unsloth 训练的效率。
- **推理系统 Token 化**：成员们质疑在推理过程中用户输入是否需要显式的分隔符，并思考像 LM Studio 这样的系统是否会隐式处理这些分隔符。
  
  - 澄清了不同模型在处理用户输入和助手响应时，对于起始和结束 Token 采用了不同的方法。

**提到的链接**：

- [WeightWatcher: Data-Free Diagnostics for Deep Learning](https://weightwatcher.ai/)：深度学习的无数据诊断。
- [Daniel Han (@danielhanchen) 的推文](https://x.com/danielhanchen/status/1856442699689414970)：Qwen 2.5 的 Bug 修复与分析：1. Pad_token 不应为 <|endoftext|> 2. 基础 <|im_start|> <|im_end|> 未经训练 3. Embedding 的 PCA 具有 BPE 层级 4. YaRN ...
- [Goku Anime GIF - Goku Anime Super Saiyan](https://tenor.com/view/goku-anime-super-saiyan-gif-5063009)：点击查看 GIF。
- [Massed Compute](https://massedcompute.com/)：Massed Compute 是一家云算力提供商，提供尖端 GPU，无需繁琐的合同和不必要的推销。
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1gpw8ls/bug_fixes_in_qwen_25_coder_128k_context_window/)：关于 Qwen 2.5 Coder 128k 上下文窗口的 Bug 修复。
- [subreddits](https://www.reddit.com/r)：未找到描述。
- [GitHub - EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)：语言模型少样本 (few-shot) 评估框架。

---

### **Unsloth AI (Daniel Han) ▷ #**[**off-topic**](https://discord.com/channels/1179035537009545276/1179039861576056922/1306122839417688165) (2 messages):

> - `User reactions` (用户反应)
> - `Discord interactions` (Discord 互动)

- **共享的 'Womp Womp' 时刻**：成员们表达了一种被称为 'womp womp' 的共同反应，暗示了一种失望或幽默的共同情绪。
  
  - 这种非正式的沟通风格突显了 Discord 社区内讨论的随意性。
- **呼应社区感受**：另一位成员表示赞同最初的情绪，称“我也一样（I do same）”。
  
  - 这种回应反映了成员在讨论各自反应时协作且友好的氛围。

---

### **Unsloth AI (Daniel Han) ▷ #**[**help**](https://discord.com/channels/1179035537009545276/1179777624986357780/1305993138959159316) (178 messages🔥🔥):

> - `Gemma 2B RAM usage` (Gemma 2B RAM 占用)
> - `Flash Attention installation issues` (Flash Attention 安装问题)
> - `RAG experience and usage` (RAG 经验与用法)
> - `Ollama model management` (Ollama 模型管理)
> - `Training on responses only` (仅针对回复进行训练)

- **长时间运行时的 Gemma 2B RAM 占用**：用户讨论了在使用 **Gemma 2B** 运行较长时间的任务时遇到 **RAM 占用持续增加**的问题，并质疑评估步骤（evaluation steps）是否可能影响性能。
  
  - 一位成员建议以 **0 steps** 进行训练，以减轻过度的资源消耗。
- **Flash Attention 安装困扰**：一位成员报告 **Flash Attention 安装**无限期挂起的问题，引发了其他成员关于检查命令执行和运行环境的建议。
  
  - 另一位成员询问该问题是否与设置过程中的 **'run all'** 命令有关。
- **探索用于长期记忆的 RAG**：有人咨询了关于 **RAG (Retrieval-Augmented Generation)** 的问题，并请求关于将其用于长期数据保存的用户经验和指导。
  
  - 一位用户推荐 **Dify** 作为实现 RAG 的简单替代方案。
- **在 Ollama 中管理模型**：讨论了如何在 **Ollama** 中上传和管理模型，包括用于复制和推送模型更新的命令。
  
  - 用户确认了 **模型上传和推送** 的成功，并解决了命名空间权限和应用连接方面的挑战。
- **理解针对助手回复的训练**：针对 `train_on_responses_only` 函数寻求澄清，特别是对话中的所有历史消息是否都会在训练期间被考虑。
  
  - 注意到模型会 **掩码（mask）用户输入**，从而允许仅根据先前回复的上下文来预测助手的回复。

**提到的链接**：

- [raultherockstar/nyayasathi](https://www.ollama.com/raultherockstar/nyayasathi)：快速上手大语言模型。
- [facebook/nougat-small · Hugging Face](https://huggingface.co/facebook/nougat-small)：未找到描述
- [Paper page - Nougat: Neural Optical Understanding for Academic Documents](https://huggingface.co/papers/2308.13418)：未找到描述
- [Nougat - a Hugging Face Space by tomriddle](https://huggingface.co/spaces/tomriddle/nougata)：未找到描述
- [Computer Vision API - OCR bounding boxes | Microsoft Community Hub](https://techcommunity.microsoft.com/discussions/azure/computer-vision-api---ocr-bounding-boxes/71774)：我正在为客户构建一个利用计算机视觉分析图像的 API。我正尝试让它分析白板上的手写内容...

---

### **Unsloth AI (Daniel Han) ▷ #**[**showcase**](https://discord.com/channels/1179035537009545276/1179779344894263297/1306098409291386912) (2 messages):

> - `Optillm release` (Optillm 发布)
> - `Hyperstack legitimacy` (Hyperstack 的真实性)

- **Optillm 发布令人兴奋的新功能**：[optillm](https://github.com/codelion/optillm) 的最新版本引入了一个本地推理服务器，允许加载任何 HF 模型和 LoRA 适配器，增强了微调后的 Unsloth 模型的可操作性。
  
  - 此次更新还支持在推理过程中进行动态适配器切换，并在使用标准 **OpenAI client SDK** 的同时支持 **cot_decoding** 和 **entropy_decoding** 等高级解码技术。
- **关于 Hyperstack 真实性的咨询**：有人提出了关于 **Hyperstack** 真实性的问题，表明社区对该平台存在一定的兴趣或怀疑。
  
  - 讨论中未提供关于 Hyperstack 可信度的具体细节或共识。

**提到的链接**：[GitHub - codelion/optillm: Optimizing inference proxy for LLMs](https://github.com/codelion/optillm)：LLM 的优化推理代理。通过在 GitHub 上创建账号来为 optillm 的开发做出贡献。

---

### **HuggingFace ▷ #**[**general**](https://discord.com/channels/879548962464493619/879548962464493622/1305987738201358466) (270 条消息🔥🔥):

> - `Qwen 模型性能`
> - `AI 项目伦理考量`
> - `GPU 规格与性能`
> - `Langchain 与 Hugging Face 集成`
> - `深度学习模型建议`

- **Qwen 模型输出表现不稳定**：用户反映 Qwen 模型在生成文本时的表现差异很大，通过与 Ollama 等其他模型的对比表明，Qwen 的回复经常可能出现幻觉或质量不足。
  
  - 建议通过调整 repetition penalty（重复惩罚）和 token 长度等参数来提升输出质量。
- **AI 项目的伦理影响**：讨论了旨在提取反射数据（reflection data）的 AI 项目所涉及的伦理困境，以及如果这些项目被广泛获取可能带来的潜在滥用风险。
  
  - 参与者表示需要对伦理后果进行仔细讨论，同时也承认该项目在执法等领域具有合法的应用价值。
- **关于 GPU 能力的疑问**：围绕 NVIDIA 4060 Ti 16GB 展开了辩论，认为考虑到其价格和显存容量，它可能是一个非常好的选择，尽管有观点认为在某些场景下它可能比 3060 Ti 等旧型号慢。
  
  - 用户指出在考虑购买新 GPU 时，相对性能指标和性价比（price-to-performance ratio）至关重要。
- **在 Hugging Face 中使用 Langchain**：用户讨论了使用 Langchain 配置 Hugging Face 模型的方法，并就调整参数以在文本生成任务中获得更好的模型性能提出了具体建议。
  
  - 这种集成允许通过针对所需输出特性定制的参数来更好地处理模型调用。
- **Diffusers 的模型推荐**：对 Diffusers 库中使用的最佳模型提出了建议，强调了 Flux 模型相比 sd3.5 large 等其他模型的优势。
  
  - 用户一致认为需要进行持续实验，以根据具体用例确定哪些模型表现最佳。

**提到的链接**：

- [FlUX WebUI - a Hugging Face Space by nroggendorff](https://huggingface.co/spaces/nroggendorff/flux-web)：未找到描述
- [InstructIR - a Hugging Face Space by marcosv](https://huggingface.co/spaces/marcosv/InstructIR)：未找到描述
- [Facepalm GIF - Facepalm - Discover & Share GIFs](https://tenor.com/view/facepalm-gif-4576513125411549651)：点击查看 GIF
- [Lol Goonies GIF - Lol Goonies The Goonies - Discover & Share GIFs](https://tenor.com/view/lol-goonies-the-goonies-gif-17881913)：点击查看 GIF
- [Sus Cat 2 Suspicious Cat GIF - Sus Cat 2 Suspicious cat The cat looks suspiciously - Discover & Share GIFs](https://tenor.com/view/sus-cat-2-suspicious-cat-the-cat-looks-suspiciously-cat-sits-in-front-of-food-the-ginger-cat-is-watching-gif-14890167989997543813)：点击查看 GIF
- [Dog Doggo GIF - Dog Doggo Cute - Discover & Share GIFs](https://tenor.com/view/dog-doggo-cute-math-formulas-gif-17580986)：点击查看 GIF
- [Alien Talking GIF - Alien Talking Alien talking - Discover & Share GIFs](https://tenor.com/view/alien-talking-alien-talking-keep-yapping-your-mouth-alien-babbling-gif-17459379075847540969)：点击查看 GIF
- [Hail Zorp Parks And Rec GIF - Hail Zorp Parks And Rec April - Discover & Share GIFs](https://tenor.com/view/hail-zorp-parks-and-rec-april-gif-14789564)：点击查看 GIF
- [Its Classified Tom Cruise GIF - Its Classified Tom Cruise Classified - Discover & Share GIFs](https://tenor.com/view/its-classified-tom-cruise-classified-private-secret-gif-9579704)：点击查看 GIF
- [Weird Weirdly GIF - Weird Weirdly Specific - Discover & Share GIFs](https://tenor.com/view/weird-weirdly-specific-gif-19034416)：点击查看 GIF
- [Qwen/Qwen2.5-Coder-7B-Instruct-GGUF at main](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/tree/main)：未找到描述
- [GeForce 40 series - Wikipedia](https://en.wikipedia.org/wiki/GeForce_40_series#Products)：未找到描述
- [GeForce 30 series - Wikipedia](https://en.wikipedia.org/wiki/GeForce_30_series#Details)：未找到描述
- [stabilityai/stable-diffusion-xl-base-1.0 at main](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main)：未找到描述
- [facebook/dinov2-large at main](https://huggingface.co/facebook/dinov2-large/tree/main)：未找到描述
- [InstantX/InstantIR at main](https://huggingface.co/InstantX/InstantIR/tree/main/models)：未找到描述

---

### **HuggingFace ▷ #**[**cool-finds**](https://discord.com/channels/879548962464493619/897390579145637909/1305999130719551489) (12 messages🔥):

> - `LLM for E-commerce Branding` (电商品牌化的 LLM)
> - `Cell Journal Research Confirmation` (Cell 期刊研究确认)
> - `Learning Machine Learning` (学习 Machine Learning)
> - `Cross-Posting Concerns` (跨频道发帖顾虑)

- **Seeking LLM for Realistic Baby Clothes Models**: 一位成员正在寻找一种 LLM，能够为他们的婴儿服装电商网站生成穿着品牌服装的超逼真 AI 模型。
  
  - 这引发了不同的反应，人们对标题的真实性及其实际应用感到好奇。
- **Cell Journal Research Credibility**: 一位成员确认了 **Cell Journal** 发表的一篇研究文章的有效性，该文章引发了兴趣和质疑。
  
  - 他们提供了[文章链接](https://www.cell.com/cell/abstract/S0092-8674(24)01152-8?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0092867424011528%3Fshowall%3Dtrue)，同时指出该文章设有**付费墙**。
- **Introduction to Machine Learning**: 一位成员表达了开始 **Machine Learning** 学习之旅的兴趣，并寻求入门指导。
  
  - 另一位成员提醒他们不要在频道内跨频道重复发帖，暗示该对话可能已经重复。

---

### **HuggingFace ▷ #**[**i-made-this**](https://discord.com/channels/879548962464493619/897390720388825149/1306028954238586911) (28 messages🔥):

> - `LightRAG Article` (LightRAG 文章)
> - `Sulie Foundation Model` (Sulie Foundation Model)
> - `ZeroGPU Debugging` (ZeroGPU 调试)
> - `Benchmarking VLA Models` (VLA 模型基准测试)
> - `PromptDX Usability` (PromptDX 易用性)

- **Introducing LightRAG for Retrieval**: 分享了一篇详细介绍 **LightRAG** 的文章，其中包括对比 Naive RAG 与本地、全局及混合方法的代码评估。作者旨在强调 LightRAG 在各种检索任务中的优势。
  
  - 您可以在[此处](https://www.linkedin.com/posts/isham-rashik-5a547711b_introducing-lightrag-a-new-era-in-retrieval-activity-7262085232743342080-xgdo?utm_source=share&utm_medium=member_desktop)阅读全文。
- **Sulie: A New Model for Time Series Forecasting**: 一个名为 **Sulie** 的新发布的用于时间序列预测的 Foundation Model，旨在简化 LoRA 微调和协变量支持的自动化。团队正在寻求反馈，并鼓励用户在 [GitHub](https://github.com/wearesulie/sulie) 上查看他们的工作。
  
  - 他们幽默地强调了数据团队面临的常见挫折，将 Zero-shot 性能问题比作“巧克力茶壶”（华而不实）。
- **ZeroGPU Debugging Insights**: 一位成员讨论了他们在 Hugging Face Spaces 上调试 **ZeroGPU** 的经验，特别是处理 NaN tensors 和 Pickle 错误。他们的发现记录在[这篇详细的博客文章](https://huggingface.co/blog/rrg92/zero-gpu-nan-and-pickle-errors)中。
  
  - 作者分享了关于 Python 在 Hugging Face 相关运行机制的新知识，并希望以此帮助面临类似问题的其他人。
- **Benchmarking VLA Models for Robotics**: 发布了一篇名为 **Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks** 的合作研究论文，旨在评估 GPT4o 等 VLA 模型的性能。这项工作代表了针对新型多模态动作模型类别的更广泛基准测试的初始阶段。
  
  - 欲了解更多详情，请访问[网站](https://multinet.ai/static/pages/Multinetv01.html)或在 [GitHub](https://github.com/ManifoldRG/MultiNet/tree/main) 上查看代码。
- **Exploring PromptDX Usability**: 关于 **PromptDX** 的讨论揭示了其将 Prompt 与代码解耦的潜力，提供了更好的可读性和结构化管理。用户可以将现有的 Markdown 文件作为组件导入，以简化 Prompt 存储，增强易用性。
  
  - 对话强调了拥有一个能够有效管理 Prompt 的系统的重要性，用户对 PromptDX 在组织 Prompt 方面的可能性表现出浓厚兴趣。

**提到的链接**：

- [Recipes | PromptDX](https://puzzlet-ai.github.io/promptdx/docs/recipes#chatbot): 基础
- [Solving NaN Tensors and Pickling Errors in a ZeroGPU Space](https://huggingface.co/blog/rrg92/zero-gpu-nan-and-pickle-errors): 未找到描述
- [GitHub - wearesulie/sulie: Access to Sulie foundation models for time-series forecasting 📈](https://github.com/wearesulie/sulie): 获取用于时间序列预测的 Sulie Foundation Models 📈 - wearesulie/sulie
- [Tweet from harsh (@HarshSikka)](https://x.com/HarshSikka/status/1856739777208574151)): 很高兴分享我们的新论文 "Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks"。我们评估了 VLM 和 VLA 模型在 20 个不同的真实世界中控制机器人的能力...

---

### **HuggingFace ▷ #**[**reading-group**](https://discord.com/channels/879548962464493619/1156269946427428974/1306050626421133374) (3 条消息):

> - `阅读时间调整`
> - `用户时区关注点`

- **阅读时间根据用户时区调整**：一位成员询问显示的阅读时间是否为美国时间。
  
  - 另一位成员回答说，它应该会根据用户电脑或 Discord 设置的任何 **timezone** 进行调整。
- **对清晨阅读的担忧**：一位成员表达了对在凌晨 **04:00 AM** 醒来参加阅读时间的担忧。
  
  - 这突显了时区设置对用户参与度的潜在影响。

 

---

### **HuggingFace ▷ #**[**computer-vision**](https://discord.com/channels/879548962464493619/922424143113232404/1306371650769653904) (2 条消息):

> - `Open3D-ML 开发`
> - `O3D 历史背景`
> - `3D 物体分类技术`

- **Open3D-ML 看起来很有前景**：一位成员分享了他们对 [Open3D-ML](https://github.com/isl-org/Open3D-ML) 的热情，这是 Open3D 的一个扩展，专为 3D Machine Learning 任务设计。
  
  - 对于那些有兴趣增强其 3D ML 能力的人来说，这个仓库似乎具有巨大的潜力。
- **O3D 仍能引起怀旧之情**：一位成员回忆了 **Open3D** 的历史意义及其与 **AlexNet** 的同时发布，并提到自己曾撰写过一本相关书籍。
  
  - 他们对 Open3D 的演变感到惊讶，尽管最初在性能上不如 **WebGL**，但现在已经拥有了一个机器学习库。
- **用于 3D 物体分类的 Python 脚本**：一位成员建议在 **Blender** 中创建一个 Python 脚本，从不同轴向生成 3D 物体的图像以用于分类。
  
  - 该技术可用于比较三个视图的分类结果，从而为结果增加一层验证。

**提及的链接**：

- [GitHub - isl-org/Open3D-ML: An extension of Open3D to address 3D Machine Learning tasks](https://github.com/isl-org/Open3D-ML)：Open3D 的扩展，用于处理 3D Machine Learning 任务 - isl-org/Open3D-ML
- [The o3d Bible by Kara Rawson](https://www.scribd.com/document/63892020/The-o3d-Bible-by-Kara-Rawson)：该文档提供了 Google O3D API 库的摘要。包括简介、安装说明、系统要求、支持的图形硬件以及程序概述...

---

### **HuggingFace ▷ #**[**NLP**](https://discord.com/channels/879548962464493619/922424173916196955/1306117375665639506) (4 条消息):

> - `法律文档检索`
> - `利用图增强检索`
> - `为越南语模型微调 Tokenizer`

- **法律文档检索中的挑战**：一位成员强调了他们在法律文档检索任务中遇到的问题，表示尽管进行了微调 Embedding 和 Reranking，但 **MRR@10** 的评估结果仍然很差。
  
  - 建议包括评估检索增强方法论和模型的最新进展，特别是与 **legal domains** 相关的部分。
- **用于检索增强的图技术**：一位成员表示有兴趣学习如何使用 **FAISS** 结合图技术来增强检索阶段，尽管他们不确定从哪里开始。
  
  - 另一位成员提到，检索增强生成 (RAG) 的领域在过去 **six years** 中有了显著改进。
- **为越南语法律数据微调 Tokenizer**：在讨论对 **Vietnamese Legal dataset** 的训练时，一位成员询问了在预训练 Tokenizer 中添加新 Token 与微调一个新 Tokenizer 之间的区别。
  
  - 他们询问微调方法对于他们的任务是否可行且 **approachable**。

 

---

### **HuggingFace ▷ #**[**diffusion-discussions**](https://discord.com/channels/879548962464493619/1009713274113245215/1306216548138094602) (8 messages🔥):

> - `SDXL Lightning Model`
> - `Realtime Image Generation Workflows`
> - `Training Diffusion Models`

- **SDXL Lightning 展示了快速的图像生成能力**：**SDXL Lightning** 或 **sd1.5 模型**在标准 GPU 上仅需几秒钟即可生成图像，使其成为基于 Prompt 的图像创作的理想选择。
  
  - 一位用户分享说，**turbo/lightning/lcm** 等变体在高性能硬件上可以实现实时图像生成。
- **针对 sd 1.5 的实时 Turbo 工作流**：一位用户在 [这篇 reddit 帖子](https://www.reddit.com/r/StableDiffusion/comments/187ps59/got_realtime_turbo_workflow_working_sd_15_lcm_and/) 中分享了在 **ComfyUI** 中结合 **LCM** 使用 **sd 1.5** 进行实时图像生成的详细工作流。
  
  - 他们详细说明了优化图像质量和功能的配置，建议使用特定的设置，如 **10 steps** 和 **1.0 或 2.0 cfg**。
- **对 SDXL Turbo 质量的担忧**：一位用户对 **SDXL Turbo** 的图像质量表示不满，更倾向于其 **sd 1.5** 配置输出的*更高质量结果*。
  
  - 他们指出，目前的配置感觉和 turbo 一样快，但效果更好，特别是在 **4090** 上运行 **768x768** 等更高分辨率时。
- **训练 Diffusion 模型的挑战**：一位参与者报告了在训练各种 **Diffusion** 模型时遇到的困难，尽管使用了均匀采样，但生成的图像仍存在类别不平衡问题。
  
  - *他们寻求建议*，关于如何使模型更好地与**数据的底层分布**对齐。

 

**提到的链接**：[Reddit - Dive into anything](https://www.reddit.com/r/StableDiffusion/comments/187ps59/got_realtime_turbo_workflow_working_sd_15_lcm_and/)：未找到描述

 

---

### **Perplexity AI ▷ #**[**announcements**](https://discord.com/channels/1047197230748151888/1047204950763122820/1306044527236354059) (1 messages):

> - `Advertising Experiment`
> - `User Experience Assurance`

- **Perplexity 开始广告实验**：为了履行其使命，Perplexity 将从本周开始在**美国**尝试以赞助后续问题的形式投放广告。
  
  - 该计划旨在建立一个**稳健且自给自足的业务**，同时坚持广告内容不会影响所提供的答案。
- **关于广告视角的博客文章**：Perplexity 鼓励用户阅读其 [博客文章](https://www.perplexity.ai/hub/blog/why-we-re-experimenting-with-advertising) 以深入了解其广告策略。
  
  - 博客概述了他们的承诺，即尽管引入了广告，仍将确保**内容保持公正**。

 

---

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1305985611290644584) (282 messages🔥🔥):

> - `Perplexity AI Subscription Model`
> - `Perplexity Ads Implementation`
> - `Model Selection Issues`
> - `User Experience Concerns`
> - `Fractal Machine Learning`

- **Perplexity AI 订阅模式备受质疑**：用户正在质疑 Perplexity Pro 订阅的价值，尤其是考虑到在付费服务中可能引入广告。
  
  - 许多人表达了不满，表示如果包含广告，他们将不再续订。
- **关于 Pro 订阅中广告的困惑**：用户对于 Pro 版本是否会出现广告感到不确定，并要求 Perplexity 团队做出澄清。
  
  - 用户正在寻找确认 Pro 订阅者会看到广告的来源，这表明他们对用户体验受到的影响日益担忧。
- **持续的模型选择问题**：多名用户报告了在 Perplexity 中选择不同模型的持续问题，无论选择哪个选项，系统通常都会回退到 GPT-4o。
  
  - 这个 Bug 严重影响了他们的工作流，让那些期望可靠访问 Claude 模型的 Pro 订阅者感到沮丧。
- **用户对体验质量的担忧**：用户对 Perplexity 的整体体验质量提出了担忧，特别是关于广告的引入和预期服务质量的下降。
  
  - 用户强调了保持简洁、无干扰搜索体验的重要性，担心该平台可能会采取与大型搜索引擎类似的策略。
- **对 AI 分形机器学习的兴趣**：一名成员提议探索分形 (Fractals) 以增强 AI 性能，讨论了在语言模型中的潜在应用，并建议与该领域的专家合作。
  
  - 社区表现出浓厚兴趣，成员们分享了关于在机器学习中创新使用分形的各种资料。

**提到的链接**：

- [The AI Bubble is About to Pop. Here's Who Dies First](https://chrisbora.substack.com/p/the-ai-bubble-is-about-to-pop-heres)：AI 泡沫即将破裂。这是谁会先倒下（一场无人准备好的 6000 亿美元大屠杀，以及隐藏的 3 万亿美元机遇）。
- [Genspark Autopilot Agent](https://www.genspark.ai/autopilotagent_viewer?id=b1190308-5abd-4be5-baba-4244aab36c81)：未找到描述。
- [Tweet from Greg Feingold (@GregFeingold)](https://x.com/gregfeingold/status/1856088784699277668?s=61)：应大众要求，我们正将校园策略师计划扩展到加拿大 🇨🇦 如果你有兴趣申请，或认识合适的人选，请联系我们！引用 Perplexity (@per...
- [Unveiling the Potential of Fractal Machine Learning - GeeksforGeeks](https://www.geeksforgeeks.org/unveiling-the-potential-of-fractal-machine-learning/)：一个面向极客的计算机科学门户网站。包含编写良好、思考深入且解释详尽的计算机科学和编程文章、测验以及练习/竞争性编程/公司面试...
- [On The Potential of The Fractal Geometry and The CNNs Ability to Encode it](https://arxiv.org/abs/2401.04141)：分形维数通过研究模式随测量尺度的变化，为对象复杂度提供了一个统计指数。尽管在多项分类任务中很有用，但分形维数...

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1305998922153595010) (7 条消息):

> - `4-7-8 Breathing Technique` (4-7-8 呼吸法)
> - `TCP/IP Guide for IT Aspirants` (IT 从业者的 TCP/IP 指南)
> - `History of the Paralympics` (残奥会历史)
> - `Differences in AI Models` (AI 模型差异)
> - `Nostalgia for Childhood Games` (对童年游戏的怀旧)

- **学习 4-7-8 呼吸法**：探索 [4-7-8 呼吸技巧](https://www.perplexity.ai/search/4-7-8-breathing-techniques-wKY17FJUQrS46xGcxkKyXw) 以增强放松并有效管理压力。
  
  - 如需深入了解，请查看关于 [4-7-8 呼吸法](https://www.perplexity.ai/page/4-7-8-breathing-technique-e8EpnEG3Q3SMg9OaIVejOQ) 的详细指南。
- **为意大利新手解释 TCP/IP 协议**：可以在[此处](https://www.perplexity.ai/page/il-livello-iso-osi-e-tcp-ip-sp-kbzTcdZqShqc2ZFLxatp0g)找到一份面向意大利新手的 **ISO/OSI** 和 **TCP/IP** 协议基础指南。
  
  - 该资源旨在为有志于从事 IT 行业的专业人士简化复杂的网络概念。
- **残奥会迷人的历史**：深入研究 [残奥会历史](https://www.perplexity.ai/search/historia-da-paraolimpiada-quai-DqHK2XMlTiC5Kg84pS3sug)，探索其演变以及对残疾运动员的影响。
  
  - 这一探索揭示了该赛事多年来的文化意义和里程碑。
- **比较 AI 模型：是什么让 Perplexity 与众不同？**：了解 [Perplexity AI](https://www.perplexity.ai/search/how-is-perplexity-ai-different-PF1ebdmMSci1d2dIu6UCiQ) 如何通过创新功能和用户体验在 AI 领域脱颖而出。
  
  - 本讨论强调了可能影响用户选择 AI 工具的关键差异。
- **怀旧：为什么童年游戏感觉更好**：一项关于“为什么童年游戏看起来更好”的研究旨在探索与怀旧体验相关的感性连接，详见[此处](https://www.perplexity.ai/page/why-childhood-games-seem-bette-ntiCEDDeQcKdT95NH09nfQ)。
  
  - 它提供了关于记忆和情感如何塑造我们对童年游戏认知的见解。

 

---

### **Perplexity AI ▷ #**[**pplx-api**](https://discord.com/channels/1047197230748151888/1161802929053909012/1306343659079209000) (4 条消息):

> - `search_domain_filter`
> - `Vercel AI SDK with Perplexity`
> - `Reddit citations issues`

- **Search Domain Filter 困惑**：一位成员质疑 **search_domain_filter** 功能是否正常运行，称他们在引用中仍然看到来自其他网站的结果。
  
  - 目前尚未确认其有效性，使用户对其可靠性感到不确定。
- **结合 Perplexity 使用 Vercel AI SDK**：一位成员询问如何将 **Vercel AI SDK** 与 **Perplexity** 集成，特别是关于引用的部分。
  
  - 这表明用户对更详细的集成过程文档或指南有潜在兴趣。
- **Reddit 引用 API 问题**：一位成员报告称，尽管之前运行良好，但在过去一周内通过 API 提取 **Reddit** 作为引用来源时遇到了问题。
  
  - 这引发了对从 Reddit 提取数据作为引用源的可靠性和一致性的担忧。

 

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1306095383247720458) (54 messages🔥):

> - `Career Paths in ML Optimization`
> - `Performance Improvement Strategies`
> - `AI Conference Insights`
> - `Job Application Trends at Big Tech`
> - `Internship Experiences in AI`

- **在 ML 角色之间做抉择**：一位成员在专注于 UI/UX 个性化的产品 ML 工程师与处理 GPU 计算编排的 ML Infra 工程师角色之间犹豫不决，强调了对 LLMs 进行性能优化的必要性。
  
  - 他们提到，如果没有 PhD 学位，他们觉得在优化领域从事算法或架构工作可能较难入门。
- **强调实践工作**：讨论强调了让项目可见以及进行实践工作的重要性，而非仅仅依靠传统的资历。一位成员提到在家里编写 CUDA kernels。
  
  - 他们打算进行性能基准测试并在博客中分享见解，意识到学习的最佳方式是通过亲身实践。
- **探索 AI 会议**：成员们分享了一系列值得关注的知名 AI 会议列表，如 [KDD, ICML,](https://aideadlin.es/?sub=ML,CV,CG,NLP,RO,SP,DM,AP,KR,HCI) 和 [CVPR](https://cvpr.thecvf.com/Conferences/2025)。
  
  - 他们还包含了论文提交的关键截止日期信息，以帮助他人及时了解。
- **换工作考量**：一位成员对因 12 个月任职要求而无法换工作表示沮丧，尽管他有机会转岗到 PyTorch 团队。
  
  - 他们正在评估 1 月份的职位空缺，同时讨论当前雇佣关系中的薪资影响。
- **AI 公司的实习角色**：一位成员询问了实习生在 AI 导向的团队中通常从事的工作类型，反映了对入门级经验的好奇。
  
  - 这个问题突显了人们对了解 AI 实习的结构和可用机会的持续兴趣。

 

**提到的链接**：[AI Conference Deadlines](https://aideadlin.es/?sub=ML,CV,CG,NLP,RO,SP,DM,AP,KR,HCI)): 未找到描述

 

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1306004954049417277) (98 messages🔥🔥):

> - `Saddle Points in Gradient Descent`
> - `Batch Normalization and Alternatives`
> - `Vision Language Action Models`
> - `Binding Problem in Neural Networks`
> - `Complex Valued Latents vs. Real Valuations`

- **在带噪声的 Gradient Descent 中，Saddle Points 很少产生影响**：参与者讨论了在带噪声的 Gradient Descent 场景下，Saddle Points 的重要性较低，表明优化器即使在它们存在的情况下也能有效运行。
  
  - 然而，一些人坚持认为 Saddle Points 在高维情况下仍可能出现，暗示其普遍性并未像之前认为的那样减少。
- **Batch Norm 在某些条件下仍然有用**：讨论强调，尽管出现了替代方案，Batch Normalization 仍然具有价值，特别是在适配为 Ghost Batch Norm 时。
  
  - 参与者对其相对于 Batch Size 的多变影响提出了批评，一些人主张对这种 Normalization 技术的效率及其擅长场景进行更多研究。
- **探索 Vision Language Action Models**：展示了一项关于在机器人任务中基准测试 Vision, Language, and Action 模型的新研究发布，涉及知名机构并提供了极具前景的见解。
  
  - 鼓励参与者分享对该工作的反馈，并深入研究提供的链接，以更深入地了解模型和应用。
- **关于 Binding Problem 和 Representation Learning 的讨论**：一场关于如何克服人工智能中的 Binding Problem 的对话展开，认为这需要超越传统技术的新型 Representation 方法。
  
  - 参与者对这些概念如何与 Hinton 的 GLOM 等先前工作以及 Transformation Representations 的潜力相关联感到好奇，强调了对创新计算模型的推动。
- **Complex Valued Latents 与实数值的对比**：参与者辩论了使用 Complex-valued Latents 相比传统 L2 Normalized Vectors 的优势，暗示其具有更大的灵活性和表达能力。
  
  - 讨论包括 Isometric Tensors 的想法以及处理保持距离的变换的能力，指出了向更丰富的数据表示发展的趋势。

**提到的链接**：

- [How to represent part-whole hierarchies in a neural network](https://arxiv.org/abs/2102.12627)：这篇论文没有描述一个工作的系统。相反，它提出了一个关于 Representation 的单一想法，允许将几个不同小组取得的进展结合到一个想象的系统中……
- [Artificial Kuramoto Oscillatory Neurons](https://arxiv.org/abs/2410.13821)：神经科学和 AI 领域早就知道，神经元之间的“Binding”会导致一种竞争性学习形式，其中 Representation 被压缩以便表示更抽象的内容……
- [Rotating Features for Object Discovery](https://arxiv.org/abs/2306.00600)：人类认知中的 Binding Problem，涉及大脑如何在固定的神经连接网络中表示和连接物体，仍然是一个激烈争论的话题。大多数机器学习……
- [Identifying and attacking the saddle point problem in high-dimensional non-convex optimization](https://arxiv.org/abs/1406.2572)：科学和工程许多领域的一个核心挑战涉及在连续的高维空间上最小化非凸误差函数。Gradient Descent 或 Quasi-Newton 方法几乎总是……
- [High-Performance Large-Scale Image Recognition Without Normalization](https://arxiv.org/abs/2102.06171)：Batch Normalization 是大多数图像分类模型的关键组件，但它具有许多由于依赖 Batch Size 以及样本间相互作用而产生的不良特性。替代方案……
- [Euclidean plane isometry - Wikipedia](https://en.wikipedia.org/wiki/Euclidean_plane_isometry)：未找到描述
- [Visual Representation Learning Does Not Generalize Strongly Within the Same Domain](https://arxiv.org/abs/2107.08221)：机器学习中泛化的一个重要组成部分是揭示潜在的 Latent Factors of Variation，以及每个因子在世界中作用的机制。在本文中，……
- [Tweet from harsh (@HarshSikka)](https://x.com/HarshSikka/status/1856739777208574151)：很高兴分享我们的新论文 "Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks"。我们评估了 VLM 和 VLA 模型在 20 个不同的真实世界机器人控制任务中的表现……

---

### **Eleuther ▷ #**[**scaling-laws**](https://discord.com/channels/729741769192767510/785968841301426216/1306011346902454306) (1 messages):

> - `Greedy Line Search`
> - `Gradient Descent Stepsizes`
> - `Periodicity in Optimization`

- **观察到振荡的学习率**：用户在尝试 **x² + ½y² + ⅓z²** 等函数并进行 **Greedy Line Search** 时，注意到了**振荡的学习率行为**。
  
  - 这种现象仅在 **Gradient Descent** 中出现，表明了对优化策略的细致探索。
- **打破关于步长的传统认知**：[Grimmer 教授的研究结果](https://x.com/prof_grimmer/status/1679846891171766272)表明，关于 **Gradient Descent** 的最佳速率依赖于恒定步长 **1/L** 的传统观念是有误导性的。
  
  - 他断言，收敛并不一定需要 **(0, 2/L)** 范围内的 **Stepsizes**；相反，证明了*周期性的长步长*能产生更好的结果，详见他的论文 [此处](https://arxiv.org/abs/2307.06324)。

**提到的链接**：[来自 Ben Grimmer (@prof_grimmer) 的推文](https://x.com/prof_grimmer/status/1679846891171766272)：我证明了职业生涯中最奇怪的结果……关于 Gradient Descent 在恒定步长 1/L 下速率最佳的经典观点是错误的。我们需要 (0,2/L) 步长来保证收敛的想法……

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1305988843161387130) (20 messages🔥):

> - `Checkpoint Issues with EvalHarness`
> - `Finetuned Model Performance on Sentiment Analysis`
> - `Accurate Averaging for CoT Accuracy`
> - `Multi-Class Text Classification Task Design`
> - `Metrics for New Task with Two Metrics`

- **EvalHarness 的 Checkpoint 问题已解决**：成员们讨论了在 **EvalHarness** 库中使用 .pth **Checkpoint** 时遇到的问题，包括与 `config not found` 相关的错误。
  
  - 一个成功的解决方法包括修复 **state dict** 键并以正确的格式保存模型，最终解决了遇到的错误。
- **Finetuned 模型在情感分析中表现不佳**：一位成员对他们的 **Finetuned** 模型在金融情感任务上的表现表示担忧，并对设置和模型选择提出了质疑。
  
  - 有建议认为，选择使用文本生成模型（Text Generation Models）而非分类模型（Classification Models）可能是导致结果不佳的原因之一。
- **澄清 CoT 中的平均准确率**：一位成员询问了如何从 **Self-consistency** 中计算原生 **CoT** 在多次运行中的平均准确率，并强调了当前评估设置的局限性。
  
  - 另一位成员建议探索现有的结构（如 **gsm8k_cot**），以寻找有效获取平均准确率的潜在解决方案。
- **多类分类任务指南**：一位成员就如何设置具有十个类别的多类文本分类任务寻求建议，以及是使用 **Multiple Choice** 还是 **generate_until**。
  
  - 建议选择 **Multiple Choice** 以有效地限制输出空间。
- **讨论双指标任务的度量标准**：一位成员报告了在新任务中同时使用 `acc_norm` 和 `exact_match` 指标时遇到困难，并请求协助。
  
  - 有建议指出 `acc_norm` 可能不适用于生成任务，并要求针对具体用例进行澄清。

---

### **Eleuther ▷ #**[**gpt-neox-dev**](https://discord.com/channels/729741769192767510/730090096287547444/1306214196366540812) (24 messages🔥):

> - `Single GPU Bugs`
> - `DagsHub Integration`
> - `YAML File Extensions`
> - `Model Training and Maintenance`
> - `Error Handling in Configurations`

- **单 GPU Bug 引发困惑**：一名成员报告在单 GPU 上运行时遇到多个 **Bug**，这表明测试主要集中在多 GPU 设置上。
  
  - 另一名成员指出，由于模型训练工作正在进行，目前大多数开放的 PR 都是针对**新功能**的，而非 Bug 修复。
- **讨论与 DagsHub 的集成**：一名成员提议鼓励 **DagsHub** 与 **GPT-NeoX** 集成可能具有潜在价值，并寻求社区的见解。
  
  - 有人询问了 **AnthropicAI** 的框架，得到的确认是他们使用自己的系统，且该系统不对外公开。
- **YAML 与 YML 文件扩展名之争**：关于在 **GPT-NeoX** 配置中使用 `.yaml` 还是 `.yml` 文件扩展名产生了困惑，有报告称使用 `.yaml` 扩展名会出现问题。
  
  - 一名成员推测配置文件可能采用了**类似 JSON 的格式**，这可以解释对扩展名的偏好。
- **模型训练工作导致延迟**：另一名成员表示，他们将在约 **30 天**内忙于模型训练和论文，这将影响维护活动。
  
  - 他们对 Bug 报告表示感谢，并计划在当前开发工作量减轻后处理这些问题。
- **配置错误已解决**：讨论了 **arguments.py** 中的一段特定代码，如果使用 `.yaml` 文件，该代码可能会导致非预期行为。
  
  - 提出的解决方案是修改代码，在文件处理逻辑中包含 `.yaml`，这可以解决现有的配置问题。

**提及的链接**：[GitHub - markNZed/gpt-neox at pipe_parallel_size_1](https://github.com/markNZed/gpt-neox/tree/pipe_parallel_size_1)：基于 Megatron 和 DeepSpeed 库在 GPU 上实现模型并行自动回归 Transformer 的项目 - GitHub - markNZed/gpt-neox at pipe_parallel_size_1

---

### **OpenRouter (Alex Atallah) ▷ #**[**announcements**](https://discord.com/channels/1091220969173028894/1092729520181739581/1306395578606813195) (1 messages):

> - `UnslopNemo 12B`
> - `SorcererLM`
> - `Inferor 12B`
> - `Mistral Parameter Updates`
> - `UI Improvements`

- **UnslopNemo 12B v4 为冒险者发布**：最新模型 [UnslopNemo 12B](https://openrouter.ai/thedrummer/unslopnemo-12b) 现已上线，专为冒险写作和角色扮演场景设计。
  
  - [UnslopNemo 12B Free](https://openrouter.ai/thedrummer/unslopnemo-12b:free) 提供 24 小时免费版本，请求可定向至 [Discord](https://discord.gg/fVyRaUDgxW)。
- **SorcererLM 引入高级叙事功能**：[SorcererLM](https://openrouter.ai/raifle/sorcererlm-8x22b) 是一款高级角色扮演模型，基于 WizardLM-2-8x22B 进行 Low-rank 16-bit LoRA 微调构建，现已开放试用。
  
  - 该模型的请求可定向至我们的 [Discord](https://discord.gg/fVyRaUDgxW)。
- **Inferor 12B 融合顶级角色扮演模型**：新的 [Inferor 12B](https://openrouter.ai/infermatic/mn-inferor-12b) 结合了现有角色扮演模型的最佳特性。
  
  - 建议用户设置合理的 max output 限制以防止生成过长文本，请求同样发送至 [Discord](https://discord.gg/fVyRaUDgxW)。
- **Mistral 和 Gemini 获得参数增强**：**Mistral** 和 **Gemini** 都增加了对 **Frequency Penalty** 和 **Presence Penalty** 的支持，增强了它们的参数能力。
  
  - Mistral 的实现现在还包括了用于 **seed** 调整的工具。
- **新 UI 功能提升用户体验**：最近的 UI 改进包括通过 cmd + K 激活的文档搜索功能，显著简化了模型搜索。
  
  - 新推出的表格列表视图允许用户同时查看更多模型，增强了整体导航性。

**提及的链接**：

- [OpenRouter](https://openrouter.ai/thedrummer/unslopnemo-12b)：LLM 路由与市场
- [OpenRouter](https://openrouter.ai/thedrummer/unslopnemo-12b:free)：LLM 路由与市场
- [OpenRouter](https://openrouter.ai/raifle/sorcererlm-8x22b)：LLM 路由与市场
- [OpenRouter](https://openrouter.ai/infermatic/mn-inferor-12b)：LLM 路由与市场

---

### **OpenRouter (Alex Atallah) ▷ #**[**app-showcase**](https://discord.com/channels/1091220969173028894/1092850552192368710/1306378508125212723) (1 条消息):

> - `GitHub Open Source Posting Rules`

- **关于 GitHub 开源发布政策的咨询**：一名成员询问了关于发布 **GitHub** 开源项目的**规则和政策**。
  
  - 他们要求在回复中被提及（tagged），强调了他们对获取该主题详细信息的兴趣。
- **寻求发布指南的澄清**：该成员强调了理解规则对于在 **GitHub** 上有效分享项目的重要性。
  
  - 他们对他人分享的任何见解预先表示感谢。

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1306000012953522309) (186 条消息🔥🔥):

> - `Model Performance Issues`
> - `Tool Calling Functionality`
> - `Image Generation APIs`
> - `Qwen Model Updates`
> - `Mistral Large Output Quality`

- **Mistral Large 输出挑战**：一位用户报告称从 **Mistral Large** 接收到了**乱码**，尽管尝试了各种系统提示词（system prompts）并重启了实例。
  
  - 在调整了最近新增的频率惩罚（frequency penalty）和存在惩罚（presence penalty）设置后，问题得到了解决。
- **关于 Tool Calling 的困惑**：用户讨论了 **tool calling** 功能，该功能旨在通过在提示词中注入工具来增强与模型的交互。
  
  - 然而，一些人发现虽然启用了 tool calling，但它似乎并没有像预期那样影响 Token 使用量。
- **Qwen 模型在 OpenRouter 上的性能**：关于 **Qwen** 模型及其 tool calling 能力的讨论，用户对其有效性表示怀疑。
  
  - 有人指出，虽然该模型理论上支持 tool calling，但一些用户在实现过程中遇到了问题。
- **图像生成 API 推荐**：用户寻求可靠的**图像生成 API** 推荐，以及值得考虑的合适平台和模型。
  
  - 对话暗示了该领域对 API 服务最佳性能和价格的需求。
- **高 Token 处理量**：一位用户提到在开发针对特定垂直领域的 AI 聊天机器人时，每天处理超过 **300 万个 Token**。
  
  - 这引发了关于某些模型在大批量 Token 处理时潜在降价可能性的疑问。

**提到的链接**：

- [Bloomberg - Are you a robot?](https://www.bloomberg.com/news/articles/2024-11-13/openai-google-and-anthropic-are-struggling-to-build-more-advanced-ai)：未找到描述
- [OpenRouter](https://openrouter.ai/api/v1)：LLM 路由与市场
- [Avian.io](https://avian.io/)：Avian.io 是全球最快的 Llama 405B 等模型推理服务的所在地。立即尝试我们的 AI 云平台和 API，无速率限制。
- [Prompt Caching | OpenRouter](https://openrouter.ai/docs/prompt-caching#inspecting-cache-usage)：优化 LLM 成本高达 90%
- [Grok Beta - API, Providers, Stats](https://openrouter.ai/x-ai/grok-beta)：Grok Beta 是 xAI 的实验性语言模型，具有最先进的推理能力，最适合复杂和多步骤的使用场景。它是 [Grok 2](https://x. Run Grok Beta w...) 的继任者。
- [Responses | OpenRouter](https://openrouter.ai/docs/responses#sse-streaming-comments)：管理来自模型的响应
- [Requests | OpenRouter](https://openrouter.ai/docs/requests#images-_-multimodal-requests)：处理传入和传出的请求

---

### **OpenRouter (Alex Atallah) ▷ #**[**beta-feedback**](https://discord.com/channels/1091220969173028894/1277894087755829278/1306066150915833958) (7 条消息):

> - `Custom Provider Keys Access Requests`

- **成员渴望获得自定义 Provider Keys**：多位用户请求访问 **Custom Provider Keys**，凸显了对该功能的强烈需求。
  
  - 每个请求都强调了希望针对其特定需求有效利用这些 Key 的愿望。
- **社区参与访问请求**：多位成员参与了讨论，展示了获取 **Custom Provider Keys** 的积极兴趣。
  
  - 请求形式多样，从简单的需求表达致直接的访问申请。

---

### **aider (Paul Gauthier) ▷ #**[**announcements**](https://discord.com/channels/1131200896827654144/1133060115264712836/1306372686519468102) (1 条消息):

> - `Aider v0.63.0 Release`
> - `Qwen 2.5 Coder Support`
> - `Web Command Functionality`
> - `Improved Language Prompting`
> - `Bug Fixes and Performance Enhancements`

- **Aider v0.63.0 发布，带来令人兴奋的新特性**：**Aider v0.63.0** 版本包含了对 **Qwen 2.5 Coder 32B** 的支持，并改进了对 LiteLLM 异常的处理，提升了易用性。
  
  - 值得注意的是，Aider 编写了此版本中 **55%** 的代码，展示了令人印象深刻的自给自足能力。
- **Web 命令重构**：新的 `/web` 命令仅将页面添加到聊天中，而不会触发 **LLM 响应**。
  
  - 这种精简的方法通过减少不必要的交互，可能会提升用户体验。
- **语言提示 (Prompting) 变得更好**：改进了对用户首选**聊天语言**的提示，从而实现更具针对性的交互。
  
  - 这一变化旨在让对话感觉更自然、更用户友好。
- **实施了关键 Bug 修复**：最近的 Bug 修复解决了诸如缓存统计期间的 **Token 重复计数**以及 LLM 创建新文件时的问题。
  
  - 这些修复为用户提供了更可靠、更高效的体验。

 

---

### **aider (Paul Gauthier) ▷ #**[**general**](https://discord.com/channels/1131200896827654144/1131200896827654149/1306006737362620609) (96 条消息🔥🔥):

> - `Vectorizing and Reranking Read-Only Files`
> - `Aider Extensions for VSCode and Neovim`
> - `Issues with Sonnet Performance`
> - `OpenRouter Provider Configuration`
> - `Upcoming AI Conferences`

- **探索只读文件的向量化**：一位用户讨论了在 Aider 中对约 30 个只读 Markdown 文件进行向量化和重排序 (Reranking) 的挑战，并指出包含过多文件会减少必要的信息。
  
  - 用户对更好的搜索功能表现出兴趣，特别是针对大型且功能丰富的项目。
- **编辑器的新 Aider 扩展**：宣布了一个新的 Aider VSCode 扩展，具有 Markdown 预览、文件管理和聊天历史记录功能，并鼓励社区贡献。
  
  - 此外，还分享了一个 Neovim Aider 扩展，促进协作以增强 Aider 在不同平台上的实用性。
- **Sonnet 性能问题**：用户报告了在使用 Sonnet 时遇到的问题，特别是无法有效地应用编辑，他们将其归因于高需求或性能波动。
  
  - 社区监控服务状态以获取更新，这表明可能存在影响 Sonnet 性能的服务器相关延迟。
- **为 Aider 配置 OpenRouter**：讨论内容包括如何在 OpenRouter 中指定提供商偏好，以及在 Aider 中创建模型元数据文件以管理成本和上下文大小的方法。
  
  - 用户分享了平衡提供商使用的技巧，以及理解 OpenRouter 负载均衡机制的重要性。
- **即将举行的 AI 会议调查**：一位用户发起了一项调查，以收集有关即将举行的 AI 会议的信息，询问正在关注的具体活动和会议品牌。
  
  - 社区反应积极，表明了对即将举行的 AI 活动和交流机会的参与度和兴趣。

**提到的链接**：

- [Anthropic 状态](https://status.anthropic.com/)：未找到描述
- [提供商路由 | OpenRouter](https://openrouter.ai/docs/provider-routing)：跨多个提供商路由请求
- [高级模型设置](https://aider.chat/docs/config/adv-model-settings.html)：为 LLM 配置高级设置。
- [GitHub - nekowasabi/aider.vim: Neovim 的 Aider 助手](https://github.com/nekowasabi/aider.vim)：Neovim 的 Aider 助手。通过在 GitHub 上创建账户为 nekowasabi/aider.vim 的开发做出贡献。

---

### **aider (Paul Gauthier) ▷ #**[**questions-and-tips**](https://discord.com/channels/1131200896827654144/1133060505792159755/1306068076076208199) (61 条消息🔥🔥):

> - `Aider 集成建议`
> - `在 Architect 模式与其他模式下使用 Aider`
> - `在 Aider 中使用 git diff`
> - `在 Termux 中设置 Aider`
> - `在 VSCode 中使用 Rust Analyzer`

- **Aider 与聊天机器人集成的建议**：一位用户表示有兴趣建议 Aider 与 GitHub 上的 ChatGPT Web 界面进行集成，并就反馈格式寻求建议。
  
  - 这次讨论引导了对用户生成的改进平台集成请求的相关性探索。
- **Architect 模式对比其他模式**：建议新用户跳过 Aider 的 Architect 模式，转而使用更简单的交互方式，特别是当他们不是开发者时。
  
  - 总的来说，用户发现 Aider 在不使用 Architect 模式的情况下添加功能也很有效，一些人提议进一步实验。
- **使用 git diff 功能**：用户可以在 Aider 中利用 `/run git diff` 命令来读取文件编辑内容，从而轻松地将更改集成到对话中。
  
  - 这增强了根据识别出的代码差异提示 Aider 执行进一步操作的能力。
- **在 Termux 中安装 Aider**：有人询问关于在 Termux 等移动环境中安装 Aider 及其与不同 IDE 兼容性的问题。
  
  - 共识确认，只要 Aider 能在兼容的 Python 环境中运行，它就与 IDE 无关（IDE agnostic）。
- **Rust Analyzer 与 VSCode 的集成**：用户询问了在 Aider 完成编辑后触发 Rust Analyzer 运行的最简单方法。
  
  - 建议用户可以利用 `--lint-cmd` 来执行任何必要的命令，包括刷新 VSCode 中的 linting 状态。

**提到的链接**：

- [Linting and testing](https://aider.chat/docs/usage/lint-test.html)：自动修复 linting 和测试错误。
- [Specifying coding conventions](https://aider.chat/docs/usage/conventions.html)：让 aider 在处理代码时遵循你的编码规范。
- [Tips](https://aider.chat/docs/usage/tips.html)：使用 aider 进行 AI 配对编程的技巧。
- [FAQ](https://aider.chat/docs/faq.html#how-do-i-include-the-git-history-in-the-context)：关于 aider 的常见问题解答。
- [Maximize Your ChatGPT Experience: Mastering Auto Split and Summarize with Superpower ChatGPT](https://www.youtube.com/watch?v=IhRbmIhAm3I)：在此下载：Chrome: https://chrome.google.com/webstore/detail/superpower-chatgpt/amhmeenmapldpjdedekalnfifgnpfnkc Firefox: https://addons.mozilla.org/en-U...
- [Superpower ChatGPT - Chrome Web Store](https://chromewebstore.google.com/detail/superpower-chatgpt/amhmeenmapldpjdedekalnfifgnpfnkc)：带超能力的 ChatGPT！文件夹、搜索、GPT Store、图片库、语音 GPT、导出、自定义提示词、提示词链、隐藏模型。
- [GitHub - Amm1rr/WebAI-to-API: Claude, Gemini to API : ) (Don't need API KEY)](https://github.com/Amm1rr/WebAI-to-API)：Claude, Gemini 转 API :) (无需 API KEY)。通过在 GitHub 上创建账号为 Amm1rr/WebAI-to-API 的开发做贡献。
- [[Q] Is it possible to use `aider --apply` with output from web frontends like chatgpt.com? · Issue #2203 · Aider-AI/aider](https://github.com/Aider-AI/aider/issues/2203)：o1-preview 在 chatgpt.com 的订阅中更便宜，而且总的来说，我喜欢使用原始 LLM 的灵活性。但将 Web 前端的编辑应用到本地文件非常麻烦。我经常……

---

### **aider (Paul Gauthier) ▷ #**[**links**](https://discord.com/channels/1131200896827654144/1268910919057149974/1306003968962596946) (4 条消息):

> - `SupermavenAI joins Cursor`
> - `Organizing codebase for AI`
> - `Challenges faced with AI coding tools`

- **SupermavenAI 与 Cursor 合作**：Cursor 宣布了 **SupermavenAI** 加入其团队以增强其研究和产品能力的激动人心消息。此次合作旨在将 Cursor 打造成为创新的**核心力量 (powerhouse)**。
  
  - 该公告通过 [Twitter](https://x.com/cursor_ai/status/1856427424927625679) 发布。
- **为 AI 效率组织代码库**：一位成员分享了在集成 [aider.chat](https://aider.chat/?ref=entrecurious.xyz) 等 AI 工具时如何有效组织代码库的见解。建议包括将项目拆分为**逻辑模块**，并确保清晰的注释以辅助理解。
  
  - 强调了在使用 AI 工具时，人类可读代码对提高生产力的重要性。
- **AI 编程工具的使用体验**：一位成员详细描述了他们使用 AI 编程工具的复杂体验，表达了最初的乐观情绪因遇到的低效问题而转变为沮丧。他们面临的挑战引发了对这类工具在提高生产力方面真实有效性的质疑。
  
  - 这次对话引发了其他人关于在编程工作流中集成 AI 的类似经历的询问。

**提到的链接**：

- [Make Way for AI-Readable Codebases](https://entrecurious.xyz/make-way-for-ai-readable-code/)：🚀 在 Hacker News 上讨论此帖。简介：不堪重负的开发者 CEO。在 Ceedar 早期，大约去年 11 月，我们（现在仍然是！）是一家只有两个人的小初创公司，却有着宏大的抱负。...
- [来自 Cursor (@cursor_ai) 的推文](https://x.com/cursor_ai/status/1856427424927625679)：我们很高兴地宣布 @SupermavenAI 正在加入 Cursor！我们将共同继续将 Cursor 打造成为研究和产品的核心力量。(1/5)

---

### **LM Studio ▷ #**[**general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1305986535912505440) (62 messages🔥🔥):

> - `Quantization Differences` (量化差异)
> - `LM Studio Connection to TTS` (LM Studio 连接到 TTS)
> - `Performance Issues with Qwen 2.5` (Qwen 2.5 的性能问题)
> - `LaTeX Rendering in LM Studio` (LM Studio 中的 LaTeX 渲染)
> - `Python Script for Llama.cpp Integration` (用于 Llama.cpp 集成的 Python 脚本)

- **理解量化大小 (Understanding Quantization Sizes)**：成员们讨论了量化大小的影响，指出较小的量化尺寸会导致更高的压缩率，但较大的尺寸可能会被拆分为多个部分。
  
  - *Heyitsyorkie* 总结道，更高的量化尺寸可以确保更好的性能，且不会产生显著损失。
- **对 TTS 集成到 LM Studio 的兴趣**：一位成员对 LM Studio 何时能连接到文本转语音 (TTS) 功能表示好奇。
  
  - 回复指出，关于集成此类功能的讨论正在进行中，但尚未提供明确的时间表。
- **排查 Qwen 2.5 性能问题**：一位用户报告之前在使用 Qwen 2.5 时遇到问题，特别是只能得到自动补全式的回复，但随后提到它已开始正常工作。
  
  - 其他人建议确保配置正确，并探索模型选项以优化性能。
- **在 LM Studio 中渲染 LaTeX**：用户们正试图弄清楚如何在 LM Studio 中正确渲染 LaTeX，一些人注意到需要使用 `$` 符号才能使其正常显示。
  
  - 一位用户报告说，尽管设置正确，LaTeX 仍未按预期渲染。
- **侧载 Llama.cpp 功能**：有人请求一个 Python 脚本，以便将最新的 Llama.cpp 侧载到 LM Studio 中，强调了对此类功能的需求。
  
  - 参与者承认社区长期以来一直期待这一功能，并提到正在努力将其变为现实。

**提到的链接**：

- [GGUF](https://huggingface.co/docs/hub/gguf)：未找到描述
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1gqa5xx/lm_studio_incredibly_sl)：未找到描述
- [Leak: ‘GPT-5 exhibits diminishing returns’, Sam Altman: ‘lol’](https://www.youtube.com/watch?v=iybgycPk-N4)：过去几天出现了两种说法。一种源自昨天 TheInformation 关于 OpenAI 的泄露，称 GPT-5/Orion 令人失望，且...
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1gqa5xx/lm_studio_incredibly_slow_12_tokenssec_on_a_3090/)：未找到描述
- [Microsoft Forms](https://forms.office.com/e/9aSb6edfGi)：未找到描述
- [llama : switch KQ multiplication to use F32 precision by default by ggerganov · Pull Request #10015 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/10015)：参考 #10005, #9991 (评论)。在 Attention 中需要更高浮点范围的模型列表不断增加，因此为了保险起见，KQ 乘法默认使用 F32。
- [lmstudio.js Code Examples - SDK (TypeScript) | LM Studio Docs](https://lmstudio.ai/docs/sdk/lmstudioclient)：在 TypeScript 应用程序中使用 lmstudio.js 的示例。

---

### **LM Studio ▷ #**[**hardware-discussion**](https://discord.com/channels/1110598183144399058/1153759714082033735/1306001693007155231) (40 条消息🔥):

> - `GPU 组合`
> - `Mac 与 PC 上的本地 LLM 使用`
> - `Partial Offload 的挑战`
> - `云端与本地模型使用的对比`
> - `即将到来的硬件竞争`

- **用于 LLM 推理的 GPU 组合**：关于将 **12GB 3060** 和 **40GB A800** 组合用于 **70B 级模型**的讨论引发了疑问：是应该使用单个 GPU 还是同时使用两者，并对扩展如何影响性能表示担忧。
  
  - 一位成员建议，仅使用 A800 可能更有利，因为它能将模型完全装入 **VRAM**，而 3060 则不行。
- **ML 机器的价格对比**：成员们讨论了 **Macbook** 定价与同等性能 PC 相比的竞争力，并指出 Macbook 的耐用性可能使其使用寿命长达十年。
  
  - 有人对 Apple **128GB M4 Max** 的定价表示担忧，认为与 PC 零件相比，其价格过高。
- **Partial Offload（部分卸载）的挑战**：分享了在系统上使用大模型进行 **Partial Offload** 的经验，结论是这对于实时交互来说是不够的。
  
  - 成员们强调，CPU 在处理 **矩阵乘法 (matrix multiplication)** 等任务时非常吃力，阻碍了效率，而 GPU 则提供了卓越的性能。
- **云端 vs 本地模型使用的辩论**：讨论了关于云服务成本的担忧，一位成员提到单晚使用就产生了 20 美元的费用，这引发了对 **API 定价** 效率的质疑。
  
  - 几位成员强调了与使用云服务相比，本地设置在隐私和实验便利性方面的优势。
- **对新硬件发展的期待**：即将推出的 **AMD Strix Halo APU** 和关于 **Nvidia ARM SoC** 的传闻引发了关于未来笔记本市场 ML 任务竞争的讨论。
  
  - 成员们对能够提升带宽和内存容量、支持高性能工作负载的硬件进步寄予厚望。

---

### **Stability.ai (Stable Diffusion) ▷ #**[**general-chat**](https://discord.com/channels/1002292111942635562/1002292112739549196/1305992834104430602) (87 条消息🔥🔥):

> - `使用 Dreambooth 进行训练`
> - `使用 Animatediff 生成视频`
> - `下载 Checkpoint 文件`
> - `用于 Stable Diffusion 的 Python 版本`
> - `访问 Discord 服务器`

- **使用电影海报训练模型**：一位用户正在寻求关于在 **auto1111** 中使用 **Dreambooth** 训练电影海报的教程。他们正在寻找最新的技术和有效训练的建议。
  
  - 社区建议查看现有资源和指南以简化流程。
- **用于视频片段的 Animatediff**：成员们讨论了使用 **Animatediff** 生成视频片段，其中一人解释说它可以上传两张图片来创建过渡效果。有人指出分辨率可能较低，但适用于社交媒体。
  
  - 推荐了 **Banodoco 服务器**，因为他们专注于视频相关工具。
- **下载 Checkpoints 和 LoRAs**：用户分享了用于下载 Checkpoint 文件和 LoRAs 的外部文件托管网站链接，提到了 **Google Drive**、**Mega** 和 **Hugging Face**。其他讨论还包括 **Civit AI** 的限制以及对某些内容的潜在封禁。
  
  - 有人对特定内容类型的移除及其对用户访问的影响表示担忧。
- **Stable Diffusion 的 Python 版本问题**：一位用户在为 Stable Diffusion 配置 Python 环境时遇到了与 **torch** 包相关的错误。建议的解决方案是卸载当前的 Python 版本，改为安装 **Python 3.10.11 64bit**。
  
  - 该用户对支持表示感谢，并计划尽快尝试该解决方案。
- **访问 Discord 寻求帮助**：用户询问如何获取 Discord 服务器的 URL，特别是寻找新的邀请和直接链接。有人分享了关于 **Pixaroma** 社区邀请链接过期的经历。
  
  - 社区为连接所需的 Discord 服务器提供了帮助。

**提到的链接**：

- [Banodoco 服务器当前的 Discord URL 是什么？（YouTube 上的所有邀请均已失效）。](https://old.reddit.com/r/StableDiffusion/comments/18wm1md/whats_the_current_banodoco_server_discord_url_all/)：在这里也搜过了，没结果。正在寻找 Banodoco 服务器 :) 谢谢！
- [Camilla Lyn 的新项目](https://photos.app.goo.gl/e5uTCokWBjYEtqaF7)：未找到描述

---

### **Interconnects (Nathan Lambert) ▷ #**[**news**](https://discord.com/channels/1179127597926469703/1179128538679488533/1305987195160494131) (48 条消息🔥):

> - `Nous 3 模型性能`
> - `Francois Chollet 离开 Google`
> - `AI Agent 工具 Operator 发布`
> - `JanusFlow 模型介绍`
> - `关于 Gwern 的社区讨论`

- **Nous 3 模型性能数据混淆**：正如[此推文](https://x.com/thexeophon/status/1856429292504096944?s=61)所示，**Nous 的 70B 模型**性能数据存在差异，导致人们对所报告的 **MMLU-Pro** 分数的有效性产生疑问。
  
  - 成员们推测，提示词技术（prompting techniques）的差异和基准测试的不一致可能是影响这些不同数据的因素。
- **Francois Chollet 从 Google 离职**：**Keras** 的创始人 Francois Chollet 即将离开 Google，开启职业生涯的新篇章，正如[此处](https://developers.googleblog.com/en/farewell-and-thank-you-for-the-continued-partnership-francois-chollet/)所宣布的那样。
  
  - 尽管他离开了，Chollet 仍致力于支持 Keras 及其未来的发展，并强调与开源社区的合作。
- **令人兴奋的 AI Agent 工具 'Operator' 发布**：OpenAI 计划推出一款名为 'Operator' 的新型 **AI agent 工具**，可以自动执行编写代码和预订旅行等任务。根据[此公告](https://www.bloomberg.com/news/articles/2024-11-13/openai-nears-launch-of-ai-agents-to-automate-tasks-for-users?accessToken=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzb3VyY2UiOiJTdWJzY3JpYmVyR2lmdGVkQXJ0aWNsZSIsImlhdCI6MTczMTUyODYxOCwiZXhwIjoxNzMyMTMzNDE4LCJhcnRpY2xlSWQiOiJTTVdOQURUMEcxS1cwMCIsImJjb25uZWN0SWQiOiJFODA3NUYyRkZGMjA0NUI2QTlEQzA5M0EyQTdEQTE4NiJ9.TTJZiuo4Nk2U295FHBFsxeN0YGznZJ32sHnNReQmEjM)，该工具预计将于 1 月发布。
  
  - 该工具旨在通过在各种场景下代表个人采取行动来提高用户生产力。
- **JanusFlow 模型介绍**：**JanusFlow 模型**作为一种新能力被引入，它将自回归 LLM 与 rectified flow 相结合，用于图像理解和生成，详见[此贴](https://x.com/deepseek_ai/status/1856552494379520510)。
  
  - JanusFlow 旨在实现强大、简单且灵活，塑造该领域下一代 AI 模型。
- **关于 Gwern 的讨论与社区见解**：成员们讨论了 **Gwern**，反思了他在 AI 和生物黑客（biohacking）方面的见解，并将其与 *Slate Star Codex* 等在线论坛中的其他知名人物进行了比较。
  
  - 大家一致认为，Gwern 精心撰写的博客对复杂主题提供了深入且深思熟虑的探索。

**提到的链接**：

- [来自 Teknium (e/λ) (@Teknium1) 的推文](https://x.com/teknium1/status/1856462102518768063?s=61): @TheXeophon @gm8xx8 bf16 和自定义解析 - 我们不能像之前那样使用 lm eval harness，所以会有不同的基准（baselines）。
- [来自 Shirin Ghaffary (@shiringhaffary) 的推文](https://x.com/shiringhaffary/status/1856792898932539609?s=61): 最新消息：OpenAI 正准备推出一款代号为 “Operator” 的新型计算机 AI Agent 工具，它可以代表用户通过浏览器执行操作，例如编写代码或预订旅行。员工被告知...
- [Bloomberg - Are you a robot?](https://t.co/dNZTbrQ4BJ): 未找到描述
- [来自 Xeophon (@TheXeophon) 的推文](https://x.com/thexeophon/status/1856429292504096944?s=61): @gm8xx8 这些是 Nous 在 3 版本发布时的数字。70B 模型的报告数据与图表也不匹配 - MMLU-Pro (发布时) 47.24 vs 现在的 54.14。我是漏掉了什么显而易见的东西吗...
- [来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文](https://x.com/lmarena_ai/status/1856444009323082093?s=61): 哪款模型最适合编程？@CopilotArena 排行榜出炉了！我们的代码补全排行榜包含了过去一个月收集的数据，提供了超过 10 万次补全服务和超过 1 万张选票！
- [发布最大的多语言开放预训练数据集](https://huggingface.co/blog/Pclanglais/two-trillion-tokens-open): 未找到描述
- [来自 DeepSeek (@deepseek_ai) 的推文](https://x.com/deepseek_ai/status/1856552494379520510): 🚀 推出 JanusFlow：将自回归 LLMs 与 rectified flow 相结合！通过采用这两个领域的最佳实践，JanusFlow 在单个模型中同时擅长图像理解与生成...
- [来自 Hailey Schoelkopf (@haileysch__) 的推文](https://x.com/haileysch__/status/1856172527921574154): 重大生活更新：我这周要加入 @AnthropicAI 了！期待与那里优秀的团队见面并共事！非常感谢过去两年与同事和合作伙伴们度过的美好时光...
- [He Admit It Admit GIF - He Admit It Admit It Admit - Discover & Share GIFs](https://tenor.com/view/he-admit-it-admit-it-admit-omg-itysl-gif-18470746): 点击查看 GIF
- [告别并感谢 Francois Chollet 的持续合作！](https://developers.googleblog.com/en/farewell-and-thank-you-for-the-continued-partnership-francois-chollet/): 未找到描述
- [关于 Gwern · Gwern.net](https://gwern.net/me): 未找到描述

---

### **Interconnects (Nathan Lambert) ▷ #**[**other-papers**](https://discord.com/channels/1179127597926469703/1179142630517518397/1306321485417152552) (6 条消息):

> - `越狱快速响应 (Jailbreak Rapid Response)`
> - `Anthropic 的提示词注入 (Prompt Injection)`
> - `Anthropic 的内部模型`

- **自适应技术拦截越狱**：Anthropic 的新研究引入了自适应技术，以便在检测到**新类别的越狱（jailbreak）**时快速拦截，详见其论文[此处](https://arxiv.org/abs/2411.07494)。
  
  - *确保完美的越狱鲁棒性非常困难*，这突显了保护 AI 模型安全所面临的挑战。
- **对提示词注入的担忧**：一位用户对 Anthropic 使用的**秘密提示词注入（secret prompt injection）**表示担忧，这影响了其模型对版权问题的响应方式。
  
  - 呼吁**公开承认**这一做法，表明社区对其影响持续感到不安。
- **关于未阉割模型的推测**：关于 Anthropic 内部是否使用**未阉割模型（un-nerfed model）**及其性能对比的问题浮出水面。
  
  - 这引发了人们对内部模型与公开访问模型之间差异的好奇。
- **每日推文活动**：一位用户发起了一项活动，每天发推文直到 Anthropic 公开回应其**提示词注入实践**。
  
  - 这反映了对 AI 模型运营透明度的更广泛诉求。
- **用户对模型指令的反应**：社区成员对指令提示词的概念做出了反应，幽默地引用了诸如**“不要产生幻觉！”（Do not hallucinate!）**之类的直接命令。
  
  - 这些反应表明了人们对 AI 模型所受限制的担忧与调侃。

**提到的链接**：

- [来自 Anthropic (@AnthropicAI) 的推文](https://x.com/anthropicai/status/1856752093945540673?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ): 新研究：越狱快速响应。确保完美的越狱鲁棒性很难。我们提出了一种替代方案：在检测到新类别的越狱时快速拦截的自适应技术。...
- [来自 kalomaze (@kalomaze) 的推文](https://x.com/kalomaze/status/1837954600348917817): 第 1 天，每天发一条推文，直到 @AnthropicAI 要么移除，要么（至少）**公开承认**对其模型进行的秘密提示词注入（顺便说一下，这在 API 中仍然适用）...

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-questions**](https://discord.com/channels/1179127597926469703/1179208129083363358/1305993487967322132) (9 条消息🔥):

> - `Vision Language Models (VLMs)`
> - `Finbarr 博客讨论`
> - `VLM 推理的计算成本`
> - `最近的 VLM 论文`

- **深入探讨 Finbarr 的博客**：一位成员提到正在阅读 Finbarr 博客，特别强调了一篇关于 **Vision Language Models (VLMs)** 的文章，此前他受到了 **Claude** 在处理公式截图方面成功的启发。
  
  - 另一位成员随后表示非常喜欢这个博客，并认为 **Sebastian Raschka** 最近关于 **VLMs** 的文章很有趣，但略显肤浅。
- **VLM 推理的高昂成本**：一篇分享的帖子讨论了由于包含 **500 多个图像 token** 导致的 VLM **高昂推理成本**，并提出了模型规模选择的潜在策略。
  
  - 关键研究结果表明，使用最大的 LLM 仅处理 **一个压缩的视觉 token** 即可实现 **计算最优推理 (compute-optimal inference)**，这引发了对其影响的讨论。
- **关于 Vision Language Models 的论文**：一位用户分享了阅读多篇 VLM 论文的热情，并提到在研究期间发布了诸如 **Pixtral** 和 **DeepSeek Janus** 等新模型。
  
  - 他们对技术的进步感到惊讶，相比于早期的 **Tesseract** 和 **ABBYY FineReader** 等工具，现在的技术使得从图像中读取文本变得更加容易。

**提到的链接**：

- [Sachin Goyal (@goyalsachin007) 的推文](https://x.com/goyalsachin007/status/1856004116012798355)：由于 500 多个图像 token，VLM 的推理成本很高。那么……你应该使用较小的模型，还是在较少的 token 上运行较大的模型？我们📢最新的发现让我感到由衷的惊讶：仅处理一个压缩的...
- [我本周阅读的论文：视觉语言模型](https://www.artfintel.com/p/papers-ive-read-this-week-vision)：他们不断发布 VLM，所以我一直在写...

---

### **Interconnects (Nathan Lambert) ▷ #**[**memes**](https://discord.com/channels/1179127597926469703/1187551504995987576/1306074419638042635) (4 条消息):

> - `MKBHD 争议`
> - `ChatGPT vs Claude 幽默`

- **MKBHD 在遭到抵制后删除 YouTube 片段**：MKBHD 面临巨大的舆论压力，此前他在 YouTube 频道中删除了一段视频，视频中他在限速 **35mph** 的住宅区以 **96mph** 的速度驾驶 **Lamborghini**，引发了公众愤怒。
  
  - *这场争议凸显了网红问责制和公众认知的风险。*
- **ChatGPT 用户因“绿色短信”笑话被嘲讽**：一位成员调侃道，那些比起 **Claude** 更喜欢 **ChatGPT** 的人，干脆去发“绿色短信”算了，以此嘲讽即时通讯应用中的文化隔阂。
  
  - *这种幽默的对比强调了 AI 聊天模型之间持续的竞争和偏好，引起了两个平台粉丝的共鸣。*

**提到的链接**：

- [near (@nearcyan) 的推文](https://x.com/nearcyan/status/1856565818433355790)：那些比起 Claude 更喜欢使用 ChatGPT 的人，干脆也去发绿色短信吧
- [Dexerto (@Dexerto) 的推文](https://x.com/Dexerto/status/1856446226444759348)：MKBHD 在住宅区限速 35mph 处以 96mph 驾驶兰博基尼遭到抵制后，删除了 YouTube 视频片段

---

### **Interconnects (Nathan Lambert) ▷ #**[**rlhf**](https://discord.com/channels/1179127597926469703/1208183230608576562/) (1 条消息):

swyxio: 是的，我们正在为此寻找合适的人选

---

### **Interconnects (Nathan Lambert) ▷ #**[**reads**](https://discord.com/channels/1179127597926469703/1214764639397617695/1305992409842319430) (11 条消息🔥):

> - `Dylan Patel 的推理数学讲座`
> - `Jonathan Frankle 的 AI 见解`
> - `Databricks 的招聘实践`

- **Dylan Patel 在斯坦福 ML 分享见解**：在斯坦福 CS 229S 题为[“推理数学、模拟与 AI 巨型集群”的讲座](https://youtu.be/hobvps-H38o?si=FR7re3r6gds6b-UN)中，讨论内容包括吉瓦级（gigawatt）数据中心的情况。
  
  - 一位成员表达了想去听斯坦福课程的愿望，反映出对该课程材料的浓厚兴趣。
- **Jonathan Frankle 揭示 AI 模型见解**：在[视频“新型模型架构不太可能出现”](https://youtu.be/7-3IxVvWoxc?si=eBMxBARTo-c-rrTZ)中，Databricks 首席 AI 科学家 Jonathan Frankle 讨论了 pre-training、fine-tuning 和 AI 政策。
  
  - 虽然没有分享特别新颖的见解，但成员们很喜欢听他的发言，认为他是公司一位极具魅力的代表。
- **Dario 的招聘理念引发辩论**：一位成员质疑 Dario 偏好招聘理论物理专业毕业生的做法（理由是“他们学得快”），并表示更倾向于能立即上手工作的经验丰富的工程师。
  
  - 成员们达成共识，认为虽然学习快的人有其优点，但在当前的就业市场中，实战经验可能具有更实质性的价值。

**提及的链接**：

- [Dylan Patel - Inference Math, Simulation, and AI Megaclusters - Stanford CS 229S - Autumn 2024](https://youtu.be/hobvps-H38o?si=FR7re3r6gds6b-UN): 网站: https://scalingintelligence.stanford.edu/ GitHub: https://github.com/ScalingIntelligence HuggingFace: https://huggingface.co/ScalingIntelligence
- [Databricks AI Head: New Model Architectures are Unlikely, When to Pre-Train/Fine Tune and AI Policy](https://youtu.be/7-3IxVvWoxc?si=eBMxBARTo-c-rrTZ): Jonathan Frankle 是 Databricks（估值 430 亿美元）的首席 AI 科学家，他于 2023 年 7 月通过收购 MosaicML 加入该公司。Databricks 拥有超过 12,000 名...

---

### **Interconnects (Nathan Lambert) ▷ #**[**posts**](https://discord.com/channels/1179127597926469703/1228051082631188530/1306248350156328990) (2 条消息):

> - `SnailBot 新闻`

- **SnailBot 新闻警报**：已通过指定角色 <@&1216534966205284433> 向所有成员发送了 **SnailBot News** 的通知。
  
  - 警报发出后没有后续的详细信息或讨论。
- **重复的 SnailBot 新闻通知**：再次向所有拥有角色 <@&1216534966205284433> 的成员发布了相同的 **SnailBot News** 通知。
  
  - 这表明可能存在与 SnailBot 相关的持续新闻或更新，尽管未提供具体细节。

---

### **Notebook LM Discord ▷ #**[**use-cases**](https://discord.com/channels/1124402182171672732/1124403655819415592/1305994687559106583) (19 messages🔥):

> - `KATT Integration for Podcasting` (KATT 播客集成)
> - `Research Consultancy Client Data Management` (研究咨询客户数据管理)
> - `Podcast Creation Limits` (播客创建限制)
> - `Sharing NotebookLM Outside Organizations` (在组织外共享 NotebookLM)
> - `Magic Book Experiment with Podcast` (魔法书播客实验)

- **KATT 增强播客体验**：一位成员分享了将 **KATT** 集成到播客制作流程中的经验，创建了一个事实核查器（fact checker）来让主持人随时掌握信息，最终制作出一段超过 **90 分钟** 的节目。
  
  - 他们指出，该流程结合了特定的处理程序和修改后的 System Prompt，并强调了 KATT 经过了长达 **两年** 的训练。
- **应对客户数据保密性**：一位研究顾问强调了在使用语言模型处理焦点小组（focus group）数据时面临的挑战，即如何在保持参与者保密性的同时，满足客户希望获得更多结果访问权限的需求。
  
  - 针对法律义务以及为避免隐私泄露而进行数据匿名化的必要性，会议提出了*谨慎建议*。
- **每个账号的播客创建限制**：一位成员询问了其账号的播客创建限制，担心在多次删除后无法创建更多播客。
  
  - 另一位成员也表达了同样的担忧，寻求明确是否存在限制，以避免在实验过程中出现潜在的中断。
- **跨组织共享 NotebookLM**：一位成员询问是否可以将 NotebookLM 内容共享到其 Google Organization 之外，怀疑这可能受到管理员设置的限制。
  
  - 确认了**无法在组织外共享**，并分享了关于个人账号限制的更多细节。
- **魔法书播客实验**：一位成员进行了一项独特的播客实验，涉及一份名为“Magic Book”的 PDF，它会引导主持人分享他们所见所闻的经历。
  
  - 这产生了一段未经编辑的播客，展示了主持人的创意投入，并分享了链接以获取观众反馈。

 

**提到的链接**：[关于如何改进有健康证据支持的营养建议的讨论](https://youtu.be/8ZTlaZUvooI)：#HealthTips #NutritionForWellness #EvidenceBasedDiet 营养建议、改善健康、循证营养、健康生活方式、饮食建议、健康饮食等...

 

---

### **Notebook LM Discord ▷ #**[**general**](https://discord.com/channels/1124402182171672732/1124402182909857966/1305991476408352809) (56 messages🔥🔥):

> - `NotebookLM usage tips` (NotebookLM 使用技巧)
> - `Podcast generation issues` (播客生成问题)
> - `Data security concerns` (数据安全担忧)

- **NotebookLM 摘要技巧**：一位用户寻求使用 NotebookLM 进行文本摘要（尤其是针对大学文献综述）的建议，希望能获得 Prompt 技巧或 Discord 讨论串参考。
  
  - 另一位用户建议探索诸如使用合成数据集（synthetic datasets）来保护敏感信息等功能。
- **播客生成障碍**：多位用户讨论了从特定来源生成播客的挑战，一位用户报告了在使用 .md 文件格式时遇到的困难。
  
  - 建议包括使用 PDF 或 Google Docs 格式，在切换文件类型后，播客焦点问题得到了解决。
- **NotebookLM 中的数据安全**：关于上传到 Gemini 的数据是否安全引发了担忧，澄清指出付费账号可确保数据安全，而免费账号则可能无法保证。
  
  - 用户对上传敏感数据表示谨慎，强调在使用该平台时需要注意保密性。

**提到的链接**：

- [无标题](https://notebooklm.google.com/notebook/3b8029e5-50c7-4007-a6d7-aba33125f8d7/audio)：未找到描述
- [AI 泡沫即将破裂，谁会先倒下](https://chrisbora.substack.com/p/the-ai-bubble-is-about-to-pop-heres)：无人准备好的 6000 亿美元大屠杀（以及隐藏的 3 万亿美元机会）

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1305987866949587044) (64 messages🔥🔥):

> - `Supermaven joins Cursor` (Supermaven 加入 Cursor)
> - `Windsurf Editor launch` (Windsurf 编辑器发布)
> - `Perplexity ads introduction` (Perplexity 广告引入)
> - `Mira Lab updates` (Mira Lab 更新)
> - `RAG future predictions` (RAG 未来预测)

- **Supermaven 加入 Cursor**：Supermaven 已正式加入 [Cursor](https://supermaven.com/blog/cursor-announcement) 以增强其 AI 编程编辑器的能力。此次合作预计将利用 Supermaven 的 AI 辅助功能来提供更好的软件开发体验。
  
  - 社区反应不一，一些用户注意到 Supermaven 之前的出色表现，同时也对这一转变表示惊讶。
- **Windsurf 编辑器发布**：Codeium 发布了 Windsurf 编辑器，被宣传为首个将 AI 协作与独立任务执行相结合的 Agentic IDE。用户报告了良好的第一印象，强调了其维持开发者心流（flow）的能力。

- 然而，一些用户指出，在某些方面它可能尚未超越像 Copilot 这样成熟的工具。
- **Perplexity 广告介绍**：Perplexity 正在其平台上尝试广告，在搜索结果旁引入“赞助后续问题”。他们与 Indeed 和 Whole Foods 等品牌合作，将其 AI 驱动的搜索引擎变现。
  
  - 此举旨在创建一个可持续的收入共享计划，因为他们意识到仅靠订阅是不够的。
- **Mira Lab 更新**：由前 OpenAI CTO Mira Murati 发起的 Mira Lab 正在组建一个专注于 AI 技术的新团队。报告显示，至少有一名 OpenAI 研究员加入了这家初创公司，表明人们对其成立有着浓厚的兴趣。
  
  - 该实验室的目标是利用其创始成员的专业知识来承担雄心勃勃的项目。
- **RAG 未来预测**：越来越多的人推测，检索增强生成 (RAG) 将在未来几个月内从主要用于问答转向更复杂的报告生成。Jason Liu 的一篇文章引起了人们对这种转变为企业带来潜在价值的关注。
  
  - 普遍观点认为，RAG 的演进将增强公司在文档和报告中利用 AI 的方式。

**提到的链接**：

- [Supermaven joins Cursor](https://supermaven.com/blog/cursor-announcement)：Supermaven 加入 Cursor，打造最好的 AI 代码编辑器。
- [Supermaven Joins Cursor](https://www.cursor.com/blog/supermaven)：我们很高兴地宣布 Supermaven 加入 Cursor。
- [Tweet from Codeium (@codeiumdev)](https://x.com/codeiumdev/status/1856741823768879172?s=46)：今天我们很高兴推出 Windsurf Editor —— 第一个 Agentic IDE，甚至更多 🏄 在 Windsurf 中，我们赋予了 AI 前所未有的深度代码库理解能力、强大的...
- [Tweet from Greg Brockman (@gdb)](https://x.com/gdb/status/1856441156281753908)：我人生中最长的假期结束了。回到 OpenAI 继续建设。
- [Predictions for the Future of RAG - jxnl.co](https://jxnl.co/writing/2024/06/05/predictions-for-the-future-of-rag/)：未找到描述
- [Bloomberg - Are you a robot?](https://www.bloomberg.com/news/articles/2024-11-13/openai-nears-launch-of-ai-agents-to-automate-tasks-for-users?accessToken=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzb3VyY2UiOiJTdWJzY3JpYmVyR2lmdGVkQXJ0aWNsZSIsImlhdCI6MTczMTUyODYxOCwiZXhwIjoxNzMyMTMzNDE4LCJhcnRpY2xlSWQiOiJTTVdOQURUMEcxS1cwMCIsImJjb25uZWN0SWQiOiJFODA3NUYyRkZGMjA0NUI2QTlEQzA5M0EyQTdEQTE4NiJ9.TTJZiuo4Nk2U295FHBFsxeN0YGznZJ32sHnNReQmEjM)：未找到描述
- [Bloomberg - Are you a robot?](https://www.bloomberg.com/news/articles/])：未找到描述
- [Tweet from Nous Research (@NousResearch)](https://x.com/nousresearch/status/1856417883934601246?s=46)：今天我们面向社区中的特定群体推出 Forge Reasoning API Beta 版，这是 inference time scaling 方面的一项进步，可应用于任何模型或模型组。https...
- [Tweet from Stephanie Palazzolo (@steph_palazzolo)](https://x.com/steph_palazzolo/status/1856360400721162745?s=46)：新消息（与 @erinkwoo 合写）：至少有一名 OpenAI 研究员接受了前 CTO Mira Murati 的邀请，加入她的新初创公司，她正与前 OpenAI 研究员 Barret Zoph 和 Luke Metz 一起工作。A...
- [Tweet from Aman Sanger (@amanrsanger)](https://x.com/amanrsanger/status/1856432315263836486)：非常高兴能与 @jbfja 和 Supermaven 团队合作！全球只有两家公司发布了具有 1M+ Token 窗口的模型：Google Deepmind 和 Supermaven。引用 Cursor...
- [Tweet from Alexander Doria (@Dorialexander)](https://x.com/dorialexander/status/1856751121101934723?s=61)：在开源领域发布 2 万亿 Token。https://huggingface.co/blog/Pclanglais/two-trillion-tokens-open
- [Tweet from morgan — (@morqon)](https://x.com/morqon/status/1856691685352194072?s=46)：另一家实验室的挣扎：Gemini 2.0 “未达到内部预期”
- [Bloomberg - Are you a robot?](https://www.bloomberg.com/news/articles/2024-11-13/openai-nears-launch-of-ai-agents-to-automate-task)：未找到描述
- [The Cursor + Supermaven Interview](https://youtu.be/ruy6cyBu0PA?si=mFuNF5OxUd-CXNPT)：一次非常重要的访谈，采访了一些非常酷的人。感谢 Ph4se0n3 的剪辑！
- [Windsurf Editor by Codeium](https://codeium.com/windsurf)：未来的编辑器，就在今天。Windsurf Editor 是第一个让开发者保持专注的 AI Agent 驱动的 IDE。现已在 Mac、Windows 和 Linux 上可用。
- [Tweet from Anthropic (@AnthropicAI)](https://x.com/anthropicai/status/1856752093945540673?s=61)：新研究：越狱快速响应。确保完美的越狱鲁棒性很难。我们提出了一种替代方案：自适应技术，在检测到新类别的越狱时迅速将其封锁。...

- [这个免费的 AI 图像编辑器改变了一切](https://youtu.be/PCL9SAlHqzw?si=FYkX8DfDRR2zDGEd)：OmniGen 安装与教程。通过提示词编辑图像的开源工具。#aitools #ai #aiart 感谢我们的赞助商 Abacus AI。试用他们全新的 ChatLLM 平台...
- [比 Cursor 更好？现已可用的未来 Agentic 编程](https://youtu.be/824Fyh146_w?si=sS5lsRATvxmVgZ1Y)：了解你代码库的 AI 编程 Agent - 使用 Windsurf 构建生产级应用。关于如何使用 ChatGPT 学习编程的免费电子书：https://clickhubspot.com/znrx?...
- [Gwern Branwen - 一位匿名研究员如何预测 AI 的轨迹](https://youtu.be/a42key59cZQ?si=m3IoFVRf4dCSl4XW)：Gwern 的博客：https://gwern.net/。Gwern 是一位匿名研究员和作家。在这一集之后，我说明了 Gwern 创建一个捐赠页面，让人们可以...
- [GitHub - VectorSpaceLab/OmniGen: OmniGen: 统一图像生成。https://arxiv.org/pdf/2409.11340](https://github.com/VectorSpaceLab/OmniGen)：OmniGen：统一图像生成。https://arxiv.org/pdf/2409.11340 - VectorSpaceLab/OmniGen
- [Anysphere 收购 Supermaven 以增强 Cursor | TechCrunch](https://techcrunch.com/2024/11/12/anysphere-acquires-supermaven-to-beef-up-cursor/)：Cursor 背后的公司 Anysphere 已收购 AI 编程助手 Supermaven，金额未披露。
- [Perplexity 在其平台引入广告 | TechCrunch](https://techcrunch.com/2024/11/12/perplexity-brings-ads-to-its-platform/)：AI 驱动的搜索引擎 Perplexity 表示，将从本周开始在其平台上尝试投放广告。

---

### **Latent Space ▷ #**[**ai-announcements**](https://discord.com/channels/822583790773862470/1075282504648511499/1306347362083409962) (2 条消息):

> - `为企业构建 AI`
> - `Windsurf 编辑器发布`
> - `Cascade 流式体验`
> - `通用访问工具`
> - `社区反馈`

- **关于企业级 AI 的新客座文章**：最新的客座文章分享了关于如何通过采用**企业基础设施原生 (Enterprise Infrastructure Native)** 心态来实现 AI 变现的见解，强调从一开始就采用这种方法进行构建。
  
  - 文章鼓励立即采取行动，并指出*在这里，过早优化并非过早*。
- **Windsurf 编辑器发布**：Codeium 宣布推出 **Windsurf 编辑器**，被誉为首个提供深度代码库理解和实时动作感知的 agentic IDE。
  
  - 该工具将 Copilot 的协作特性与 Agent 的自主能力相结合，创造了一种名为 **Cascade** 的体验，从而增强工作流。
- **Windsurf 访问无需排队**：Windsurf 编辑器对所有人开放，无需**排队或邀请制**访问，允许用户立即开始使用。
  
  - 这种方法强调了用户的包容性，并指出**全面开放才是它应有的样子**。

**提到的链接**：

- [来自 crystal (@crystal) 的推文](https://x.com/latentspacepod/status/185)：adam 讨厌我的用户名。
- [来自 Latent.Space (@latentspacepod) 的推文](https://x.com/latentspacepod/status/1856788504321429519)：🆕 文章：为企业构建 AI https://latent.space/p/enterprise。来自 @_anshulr 期待已久的第三篇（！）客座文章，关于如何利用 AI 赚钱：从一开始就以企业级...

---

### **GPU MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1189498205101109300/1306006746279444542) (2 条消息):

> - `XOR Tensor Cores`
> - `波束成形算法`
> - `Lambda 作为一个稳定的选择`

- **XOR Tensor Cores 在波束成形中的应用**：讨论强调了 **XOR Tensor Cores** 可用于**超声波扫描**的**波束成形算法 (beamforming algorithms)**，展示了它们在该应用中的潜力。
  
  - 这一应用强调了 XOR Tensor Cores 在传统处理用例之外的灵活性。
- **Lambda 被公认为可靠的选择**：有人指出，在当前环境下，**Lambda** 作为一个**稳健且稳定的选择**脱颖而出，表明了用户对其可靠性的信心。
  
  - 成员们似乎因其一致的性能而青睐它，这仍然是用户关注的关键因素。

---

### **GPU MODE ▷ #**[**triton**](https://discord.com/channels/1189498204333543425/1189607595451895918/1305991754939367475) (13 messages🔥):

> - `Triton Kernel Functionality` (Triton Kernel 功能)
> - `Libstdc++ Issues in Triton with Conda` (Conda 环境下 Triton 的 Libstdc++ 问题)
> - `GEMM Kernel Design in Triton` (Triton 中的 GEMM Kernel 设计)
> - `Warp Memory Alignment Error` (Warp 内存对齐错误)
> - `Lowering ttir to Hexagon Dialect` (将 ttir 降低到 Hexagon Dialect)

- **Triton Kernel 成功复制 Tensor**：一位成员分享了他们在 Triton 方面的进展，通过 Triton Kernel 成功将源 Tensor 复制到目标 Tensor，并确认其在特定的 Python 函数下正常工作。
  
  - 他们请求协助调整 Kernel，以便在 Tensor 复制操作中遵循其自定义的 `block_mapping` 结构。
- **Torch Compile 的崩溃问题**：一些用户报告说，在使用 `torch.compile` 时从源码构建 Triton 会导致崩溃，特别是在使用 PyTorch 的 nightly 版本时。
  
  - 这被怀疑与 `libstdc++` 问题有关，需要将系统目录中的文件复制到 Conda 环境中，以便 Triton 加载。
- **高效 GEMM Kernel 设计的困惑**：一位成员就如何在 Triton 中为小于 16 的小尺寸（包括特定的点积场景）设计高效的 GEMM Kernel 寻求建议。
  
  - 另一位用户建议对于尺寸 1 使用手动点积计算，对于较大尺寸则采用 `BLOCK_SIZE_M=16` 并配合 masking。
- **Warp 内存对齐错误问题**：一位用户在 GitHub 上提交了一个关于手动启动 Triton 编译的 PTX 时遇到的内存对齐错误的 Issue。
  
  - 该 Issue 包含了他们尝试编译的 Kernel 细节，并寻求社区帮助。
- **寻找关于 Lowering 到 Hexagon Dialect 的信息**：一位成员询问了有关将 ttir 降低（Lowering）到 Hexagon Dialect 的资源，并分享了一个相关的 [YouTube 视频](https://www.youtube.com/watch?v=odnyMYSTxoU)。
  
  - 他们旨在收集社区见解以及在使用该功能过程中的具体细节。

**提到的链接**：

- [triton GEMM with size < 16 · Issue #5138 · triton-lang/triton](https://github.com/triton-lang/triton/issues/5138)：描述了 Triton 如何支持尺寸小于 16 的 GEMM 问题，例如 dot (1, D) 和 (D, D)，(3, D) 和 (D, D) 或任何小于 16 的数字？我看到许多方法建议当尺寸...
- [Warp memory alignment error when manually launching compiled PTX · Issue #5136 · triton-lang/triton](https://github.com/triton-lang/triton/issues/5136)：描述了 Bug，我正在使用 Triton 将一个 batched matmul kernel 编译为 PTX，如下所示：import torch import triton import triton.language as tl import os KERNEL_PATH = "src/triton_kernels...

---

### **GPU MODE ▷ #**[**torch**](https://discord.com/channels/1189498204333543425/1189607750876008468/1306108340694159370) (7 messages):

> - `torch.compile memory usage` (torch.compile 内存占用)
> - `dynamic shapes impact` (动态形状的影响)
> - `profiling GPU memory allocations` (分析 GPU 内存分配)
> - `direct access to GPU` (直接访问 GPU)

- **torch.compile 影响峰值内存占用**：成员们正在讨论 **torch.compile()** 可能增加**峰值内存占用**并导致 **out-of-memory (OOM)** 错误的情况。
  
  - 有人提到，即使没有使用 `reduce-overhead` 标志，他们也观察到内存占用增加了 **3-16%**，特别是在使用**动态形状（dynamic shapes）**时。
- **内存占用的分析技术**：有人建议使用带有特定 **nvtx** 范围的 **nsys profile** 来分析当前的 GPU 内存使用情况并有效跟踪分配。
  
  - 关于 PyTorch 中的 **CUDA Graphs** 在不使用 `reduce-overhead` 标志的情况下是否会导致内存增加，似乎还存在一些不确定性。
- **直接 GPU 访问咨询**：一位成员询问了如何实现**直接访问 GPU** 的指导。
  
  - 目前还没有针对这一特定咨询的回复或解决方案。

 

---

### **GPU MODE ▷ #**[**cool-links**](https://discord.com/channels/1189498204333543425/1189868872887705671/) (1 messages):

0xredj: [https://latent.toys](https://latent.toys)

---

### **GPU MODE ▷ #**[**jobs**](https://discord.com/channels/1189498204333543425/1190208177829068860/1306286531643965544) (1 条消息):

> - `SEMROM 职位空缺`
> - `Quantization 方法`
> - `Inference 框架开发`
> - `ML 与硬件协同工作`
> - `开源贡献`

- **SEMROM 招聘 ML Quantization 工程师！**：SEMROM 是一家**风险投资支持的初创公司**，正在寻找一名专注于 **quantization** 且乐于在 Edge 设备上架起机器学习与硬件桥梁的工程师。
  
  - *查看完整的职位描述* [点击此处](https://semron.jobs.personio.de/job/1433496?language=de&display=en) 以了解更多关于开发可扩展 Inference 框架的信息。
- **利用最新的 Quantization 技术进行创新**：该职位涉及应用和创新前沿的 quantization 方法，如 **AdaRound**、**BRECQ**、**GPTQ** 和 **QuaRot**。
  
  - 候选人应具备深厚的 **PyTorch** 背景，并具有开发高效自定义 **CUDA kernels** 的经验。
- **在 SEMRON 跨团队协作**：该职位需要与 ML、compiler 和硬件团队协作，使 quantization 算法适应 SEMRON 的独特需求。
  
  - 这种跨职能的方法旨在完善 **inference 框架**，确保其针对即将推出的硬件进行精细调优。
- **贡献开源的机会**：作为团队的一员，工程师将做出基础架构决策，并为*上游开源项目*做出贡献。
  
  - 这一职责体现了 SEMRON 对社区参与和创新的承诺。

 

**提到的链接**：[ML Quantization Engineer | Jobs bei SEMRON](https://semron.jobs.personio.de/job/1433496?language=de&display=en)：我们是 SEMRON，一家来自德累斯顿的初创公司，我们正在为 AI 应用开发创新的微芯片。

 

---

### **GPU MODE ▷ #**[**beginner**](https://discord.com/channels/1189498204333543425/1191300313928433664/1306252662798422046) (2 条消息):

> - `课程访问`
> - `Discord 频道信息`

- **查找课程访问信息**：一位用户询问在哪里可以访问**课程**。
  
  - 回复指向了一个特定的 **Discord 频道**以获取更多详情，参考 <#1198358627594023014>。
- **对课程查询的回复**：关于频道内**课程**访问的进一步说明已提供。
  
  - 该频道是用户获取可用课程信息和更新的中心点。

 

---

### **GPU MODE ▷ #**[**off-topic**](https://discord.com/channels/1189498204333543425/1215328286503075953/1306003618633355304) (9 条消息🔥):

> - `AI 生成的图像`
> - `食物生成`
> - `身份冒充`
> - `食物循环`
> - `机器人验证`

- **对 AI 生成图像的好奇**：一位用户询问某张图片是否由 AI 生成，促使另一位成员澄清该图片并非 AI 生成。
  
  - 讨论引发了人们对是否可以模拟交互并创建虚假内容的好奇。
- **Text-to-food 创意**：一位成员幽默地描述了他们通过 prompt 生成食物的能力，将自己比作“text-to-food 模型”。
  
  - 他们开玩笑地补充说，这个过程导致了“食物到粪便”的循环，暗示了对食物生成的幽默看法。
- **关于生命循环的讨论**：一位用户扩展了生命循环的幽默概念，指出“粪便到土地，土地到食物”是自然循环的一部分。
  
  - 这一思考与之前关于生成食物的对话相呼应，并突出了对生命过程的俏皮视角。
- **对身份验证的担忧**：一位用户对验证另一位用户是否为机器人表示怀疑，并寻求确认。
  
  - 这引发了关于在线交互中身份和真实性的问题。

 

---

### **GPU MODE ▷ #**[**rocm**](https://discord.com/channels/1189498204333543425/1233704710389764236/1306258526179688510) (3 messages):

> - `MI300X performance`
> - `FP16 peak throughput`
> - `Triton on AMD GPUs`

- **在 MI300X 上追求 800 TFLOPS**：一位用户询问是否有人在 **MI300X** 上实现了 **800 TFLOPS**，并表示他们使用 **CK** 的尝试效果不佳。
  
  - 他们正在寻求有关优化方法的见解，以达到这一性能里程碑。
- **600 TFLOPS 是 FP16 的峰值**：另一位用户指出，**600 TFLOPS** 似乎是 MI300X 上 **FP16** 性能的峰值。
  
  - 这表明在该特定精度下，达到更高的 TFLOPS 速率存在限制。
- **来自 Triton YouTube 演讲的见解**：一位用户分享了一个名为 [“Triton on AMD GPUs” 的 YouTube 视频](https://youtu.be/Lbm08twNTAQ?si=6Vwrkz8W0U2WTpf1&t=243)链接，Lei Zhang 和 Lixun Zhang 在视频中讨论了 Triton 对 AMD 的支持。
  
  - 该演讲展示了围绕 chiplets 的巧妙优化技术，为提升 GPU 性能提供了宝贵的见解。

**提到的链接**：[Triton on AMD GPUs](https://youtu.be/Lbm08twNTAQ?si=6Vwrkz8W0U2WTpf1&t=243)：Lei Zhang 和 Lixun Zhang 谈论 Triton 对 AMD 的支持。这次演讲展示了一些围绕 chiplets 和指令集的非常巧妙的优化技术……

---

### **GPU MODE ▷ #**[**liger-kernel**](https://discord.com/channels/1189498204333543425/1275130785933951039/1306045056028901440) (2 messages):

> - `Liger-Kernel v0.4.1 Release`
> - `Gemma 2 Support`
> - `CrossEntropy Patching Fix`

- **Liger-Kernel v0.4.1 已发布**：[Liger-Kernel](https://github.com/linkedin/Liger-Kernel/releases/tag/v0.4.1) 的最新版本 **v0.4.1** 引入了对 **Gemma 2** 的支持，并修复了 CrossEntropy 的补丁。
  
  - 此版本是一个重要的里程碑，感谢 @Tcc0403 的贡献，他解决了 fused linear cross entropy 中长期存在的 softcapping 问题。
- **v0.4.1 中的令人兴奋的新功能**：新增的 **Gemma 2 Support** 是一个备受期待的功能，实现了更增强的功能。
  
  - 该版本还包括对 **GroupNorm** 的修复等改进，进一步简化了操作。

**提到的链接**：[Release v0.4.1: Gemma 2 Support, CrossEntropy Patching FIx, and GroupNorm · linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel/releases/tag/v0.4.1)：重点介绍了 Gemma 2 支持：感谢 @Tcc0403，悬而未决已久的 Gemma 2 终于得到支持！他实现了 fused linear cross entropy (#320) 中棘手的 softcapping，并发现了 conv...

---

### **GPU MODE ▷ #**[**self-promotion**](https://discord.com/channels/1189498204333543425/1288557096404516945/1306223229714698340) (6 messages):

> - `Flash Attention Overview`
> - `Video Feedback`
> - `CUDA and Triton Programming`
> - `Multi-Head Attention`
> - `Community Praise`

- **从零开始推导 Flash Attention**：该视频专注于**从零开始推导和编写 Flash Attention 代码**，在无需预备知识的情况下讲解 CUDA 和 Triton。
  
  - 主题包括 **Multi-Head Attention**、**Softmax**，以及矩阵乘法和 Softmax 操作的 **Jacobian** 矩阵。
- **社区对视频长度的反应**：一位成员幽默地将该视频描述为 **7.5 小时** 的“休闲”观看，说明了内容的深度和广泛性。
  
  - 另一位成员赞扬了之前关于**量化 (quantization)** 的视频，强调了创作者引人入胜的风格。
- **影响与认可**：一位成员表达了对创作者的钦佩，指出：*“我钦佩的许多人都在对你的视频好评如潮！”*
  
  - 创作者对这一赞誉表示感谢，称：*“这对我也非常有意义”*，展现了对社区的感激。
- **轻松的学习环境**：对话反映了一个轻松的学习环境，创作者提到他们*“这周末挺无聊的”*，所以选择制作了这个视频。
  
  - 另一位成员评论道：*“Sensei 也表示认可”*，表明了群体内的同志情谊和相互尊重。

---

### **GPU MODE ▷ #**[**🍿**](https://discord.com/channels/1189498204333543425/1298372518293274644/1306304361349255269) (4 messages):

> - `Discord Leaderboard UX`
> - `Bot Development`
> - `Dataset Collection`

- **关于 Discord 排行榜 UX 的想法**：一个标题为 [Ideas for what to do next](https://github.com/gpu-mode/discord-cluster-manager/issues/6) 的 GitHub issue 讨论了 **Discord 排行榜** 将如何渲染，包括可能使用的 **slash commands**，这些命令可以自动填充脚本预期以及内核名称或 GPU 类型等额外细节。
  
  - 讨论强调了需要通过贡献来演进 UX，鼓励参与者研究该 issue 并就实现细节进行进一步讨论。
- **Bot 开发和数据集收集的重点领域**：团队成员强调当前阶段的**主要重点**是 **bot 开发**和**数据集收集**工作。
  
  - 一名成员表示打算与另一名成员合作，就这些重点的跟踪 issue 进行沟通并提供潜在建议。

 

**提到的链接**：[Ideas for what to do next · Issue #6 · gpu-mode/discord-cluster-manager](https://github.com/gpu-mode/discord-cluster-manager/issues/6)：基于 Discord 的排行榜 UX。排行榜如何渲染 @AndreSlavescu。自动填充脚本预期的 Slash commands，可能还有更多信息，如内核名称或 GPU 类型...

 

---

### **GPU MODE ▷ #**[**thunderkittens**](https://discord.com/channels/1189498204333543425/1300872762163728550/1306025391999418389) (15 messages🔥):

> - `TK improvements`
> - `H100 DSMEM limitations`
> - `Cooperative groups in CUDA`
> - `Semaphore synchronization`
> - `TMA instruction usage`

- **TK 收到积极反馈并解决 Bug**：成员们对 **ThunderKittens** 的工作表示感谢，并修复了一个由简单错误引起的用户问题。
  
  - 一名成员提到他们很高兴能实现这些更改并分享未来的 PR。
- **H100 DSMEM reduce 支持的类型有限**：讨论显示 **H100** 不支持 float 或 bfloat 类型的 **DSMEM** reduction，具体来说仅处理整数。
  
  - 一名成员分享了关于在使用 `.shared::cluster` 时 `cp.reduce.async.bulk` 针对整数类型限制的发现。
- **TMA 指令的 Cooperative group 执行**：澄清了 **TK 测试**中的 TMA 指令通常由整个 warp 执行，但在内部会屏蔽除第一个 lane 以外的所有 lane。
  
  - 这引发了关于从单个线程调用这些指令的方法及其背后原理的讨论。
- **信号量操作顺序至关重要**：成员们强调了在 DSMEM 代码中调用信号量操作和 cluster 同步时顺序的重要性。
  
  - 建议重构信号量调用和同步，以确保功能正常并防止挂起问题。
- **即将发布的 PR 包含针对整数的测试**：一名成员表示计划在提交与 **TK** 开发相关的 PR 之前完成整数测试代码。
  
  - 大家对有机会审查涉及现有代码改进的即将到来的更改感到兴奋。

 

**提到的链接**：[ThunderKittens/tests/unit/warp/memory/tile/dsmem.cu at 06d654a0858840e006d428cd96aac2cd0d19ca25 · HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens/blob/06d654a0858840e006d428cd96aac2cd0d19ca25/tests/unit/warp/memory/tile/dsmem.cu)：用于快速内核的 Tile 原语。通过在 GitHub 上创建账号为 HazyResearch/ThunderKittens 的开发做出贡献。

 

---

### **GPU MODE ▷ #**[**edge**](https://discord.com/channels/1189498204333543425/1303441437592911912/1306051792261615707) (1 messages):

> - `Quantization Techniques`
> - `Algorithm Optimization`
> - `Deployment Strategies`
> - `Memory Efficiency`
> - `Hybrid Processing`

- **通过 Quantization 优化训练后模型**：讨论了模型训练后的优化方法，如 **quantization**、**distillation** 和 **speculative decoding**。
  
  - 有建议提出消除 **attention layers** 和早期的 **feed forward networks**，因为这对性能影响极小。
- **通过 Flash Attention 进行算法改进**：强调了使用 **Flash Attention** 以及 **loop unrolling** 和 **inline assembly** 等技术来增强算法性能。
  
  - 对话中提到，应根据具体用例调整 **batching** 以获得最佳结果。
- **评估部署选项**：探讨了选择合适的 **computational devices** (GPU, CPU, NPU) 对于部署考量的重要性。
  
  - 提到了实现 **hybrid tensor parallelism** 的复杂性，以实现更高的内存效率并处理 long-context 应用。
- **混合处理以提高效率**：考虑将 **hybrid cloud/local processing** 方法用于 **Petals** 等分布式推理方法。
  
  - *适用性因情况而异*，建议仔细考虑该策略。

 

---

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1306013250688651344) (29 messages🔥):

> - `Tinygrad distributed approach`
> - `Multi-node FSDP`
> - `Data handling in Cloud`
> - `Device-to-device communication`
> - `Performance concerns with sharding`

- **Tinygrad 探索其分布式方法**：一位用户询问了 Tinygrad 目前的分布式计算策略，特别是其对 FSDP 的处理以及是否支持 multi-node 设置。
  
  - 另一位用户提到 FSDP 的公开悬赏 (bounty) 是一个潜在资源，并讨论了当前实现的扩展性挑战。
- **关于 Cloud 数据管理的澄清**：讨论指出，虽然云端能力可能允许在不同机器上使用数千个 GPU，但最佳性能仍取决于快速连接和有效的 all-reduce 实现。
  
  - 对在训练运行期间由单台机器编排数据管理和处理的效率表示担忧。
- **Cloud 下的设备间传输**：George Hotz 指出，设备间通信是在 Tinygrad 的 Buffer 中通过 `transfer` 函数处理的，这表明将其扩展到云端设置可能很容易。
  
  - 他幽默地提到只需几行代码即可完成，暗示了实现的简单性。
- **用户对数据转发方法的兴趣**：Codeman3786 表示渴望为 CloudDevice 中的直接设备间通信贡献一种数据转发方法，但对涉及的抽象层级感到不确定。
  
  - 他们强调了在跨集群 (cluster) 实验期间正在进行的 profiling 工作，指出需要更深入的理解。
- **关于 Cloud sharding 性能的担忧**：讨论了是否需要明确用户是 machine-sharded 还是 cloud-sharded，以避免在较慢的同步操作期间出现意外的性能问题和成本。
  
  - 对话强调了高效数据处理策略对于在不同配置下保持性能水平的重要性。

 

**提到的链接**：[How multigpu training works](https://mesozoic-egg.github.io/tinygrad-notes/multigpu.html)：Tinygrad 教程

 

---

### **tinygrad (George Hotz) ▷ #**[**learn-tinygrad**](https://discord.com/channels/1068976834382925865/1070745817025106080/1306131918818906122) (32 messages🔥):

> - `Unsigned Tensor min() bug`
> - `Asynchronous mode for Tinygrad inference`
> - `Using BitwiseNot for integer operations`
> - `Pull Request updates`
> - `Realizing Tensors`

- **Unsigned Tensor min() bug 需要 PR 修复**：一位用户在计算无符号 Tensor 的 **min()** 函数时发现了一个 bug（特别是在处理零值时），并建议通过 flip 操作来解决。
  
  - *Rezvan* 提交了一个包含失败测试的 PR，并提到由于潜在的 **infs** 和 **nans**，情况变得更加复杂。
- **关于异步推理模式的讨论**：讨论强调了 **Tinygrad** 实现异步模式的潜力，以便在不阻塞的情况下并发处理多个模型。
  
  - 该提案涉及使用一种等待机制来进行模型 realization，并将输出捕获到 **outputs** 数组中。
- **在整数修复中使用 BitwiseNot**：一位成员建议使用 **BitwiseNot** 来处理 min 计算中的无符号整数，但对其对 float 类型的影响表示担忧。
  
  - *ChenYuy* 调整了 min 修复方案以实现 **BitwiseNot**，并提议将此技术扩展到 **argmin** 和 **minimum** 函数。
- **无符号 min 修复 PR 的更新**：在审查了无符号 min 修复的 PR 后，一位贡献者指出输入必须是 **2D array** 才能通过所有测试。
  
  - 双方同意继续改进，并确保该实现在各种 Tensor 维度下都具有鲁棒性。
- **在 Tinygrad 中测试操作**：*ChenYuy* 建议在 **Ops.IF** 上抛出错误，以确定哪些测试会失败，从而帮助理解 Tensor 级别的操作。
  
  - 这种方法将有助于深入了解操作之间是如何交互的，并提高对测试中现有问题的清晰度。

**提到的链接**：

- [t - Overview](https://github.com/t)：t 有 14 个可用的仓库。在 GitHub 上关注他们的代码。
- [fix: Tensor min function for unsigned ints by bjsi · Pull Request #7675 · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/pull/7675/commits/6c1092cefc98c87edfe9516f3887d6789351140f)：使用来自 Discord 的 flip 想法来修复无符号整数的 min 函数。

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1305997985397608549) (28 messages🔥):

> - `AI language models and date awareness`
> - `Blaze AI for content creation`
> - `AI songwriting tools`
> - `ChatGPT's UI control system`
> - `Copy-paste issues on mobile`

- **AI 模型在日期准确性方面表现不佳**：讨论指出，像 **Gemini** 和 **Claude** 这样的模型提供了错误的当前日期，而 **ChatGPT** 的回答很准确。
  
  - 一位用户认为区别可能在于系统提示词（system prompts）的设置方式，**ChatGPT** 在某些语境下能够正确推断日期。
- **对用于营销的 Blaze AI 感兴趣**：一位成员询问了 **Blaze AI** 在内容创作（特别是营销用途）方面的有效性。
  
  - 反馈中提到，为了根据特定需求定制平台，初始**配置**（configuration）需要花费一定时间。
- **寻找 AI 写歌工具**：一位成员正在寻找**免费 AI 工具**的推荐，以便为特殊场合创作带有歌词的歌曲。
  
  - 另一位成员提到 **Suno** 是一个潜在的选择，尽管它每天可生成的数量有限。
- **创新的 AI 驱动 UI 控制系统**：一位用户分享了他们项目的细节，该项目允许 **ChatGPT** 通过光标移动和决策等方法控制计算机的 UI，技术栈包括 **OCR** 和 **Python**。
  
  - 该方法旨在将 AI 与用户工作流融合以增强自动化，引发了关于项目代码以及与其他 AI 解决方案对比的询问。
- **移动端复制粘贴问题依然存在**：一位成员对移动平台上持续存在的**复制粘贴**问题表示担忧，称该问题已持续数周未解决。
  
  - 这些挑战继续影响着移动端应用程序的用户体验和功能。

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1306032001496907836) (11 messages🔥):

> - `Prompt Engineering`
> - `ChatGPT o1-preview Feedback`
> - `Model Limitations`
> - `Using VPNs`
> - `Tinkering with Requests`

- **探索 Prompt Engineering 的艺术**：成员们讨论了编写有效 Prompt 的复杂性，并建议在输出不符合预期时进行优化并寻求帮助。
  
  - *一位用户指出*，当需要引导模型而产生挫败感时，“这就像在教理发师如何正确地修剪你的头发”。
- **ChatGPT o1-preview 给用户留下深刻印象**：反馈表明，与之前的版本相比，**ChatGPT o1-preview** 展示了更强的创造力和定制化响应能力。
  
  - *一位用户表达了感谢*，提到它能够预判输入，使体验更加个性化。
- **影响用户体验的模型限制**：用户对模型在多次请求后停止产生有效结果表示担忧，暗示可能存在 **block configurations**（屏蔽配置）方面的问题。
  
  - *一位社区成员询问*，“如何修复这个问题？”表明希望获得改善交互持久性的解决方案。
- **使用 VPN 绕过封锁是合法的**：讨论强调，虽然扫描互联网对国家来说是合法的，但用户可以合法地使用 **VPN** 来绕过限制或封锁。
  
  - *一位成员强调*，“你的 block configuration 对于预期目的是无效的，”指出了对有效解决方案的需求。
- **在限制范围内调整请求**：成员们分享了想要调整 Prompt 但受限于模型限制和订阅约束的见解。
  
  - 一位成员表达了对更好结果的需求，将这种交互比作一场旅程，学习如何有效地进行 Prompt 是需要培养的技能。

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/1306011250769268850) (5 messages):

> - `Scratchpad Techniques`
> - `Structured Output`
> - `Prompt Management Challenges`

- **理解 Scratchpad 技术**：一位成员强调，Scratchpad（草稿垫）充当了一种 **pseudo-CoT technique**（伪 CoT 技术），允许 LLM 在寻求解决方案的过程中写下其思考过程。
  
  - 另一位成员表示了热情，指出将其纳入 **Structured Output** 时可以增强 **documentation**（文档化）。
- **Structured Output 组织中的挑战**：一位成员提出了关于确保 **Scratchpad** 优先完成的担忧，因为他们的 Prompt 管理器正在 **重新排序 Structured Output 字段**。
  
  - 这突显了工作流管理中可能影响生成内容一致性的潜在问题。

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/1306011250769268850) (5 messages):

> - `Scratchpad Technique`
> - `Prompt Management Issues`

- **Scratchpad 技术增强了 LLM 的思考过程**：一位成员将 Scratchpad 描述为一种 **pseudo-CoT technique**，LLM 在迈向最终解决方案的过程中写下其思考过程。
  
  - 另一位成员表示有兴趣将 Scratchpad 纳入 Structured Output，以实现 **更好的文档化**。
- **Scratchpad 输出顺序的挑战**：成员们对确保 Scratchpad 内容优先生成表示担忧，因为 **Prompt 管理器正在重新排序 Structured Output 字段**。
  
  - 这个问题突显了在 LLM 使用过程中，工作流内使用 Scratchpad 可能存在的不一致性。

---

### **Modular (Mojo 🔥) ▷ #**[**general**](https://discord.com/channels/1087530497313357884/1098713601386233997/1306145125747134505) (13 messages🔥):

> - `exllamav2`
> - `MAX Involvement`
> - `Error with Batched MatMul`

- **探索用于 LLM 推理的 exllamav2**：成员们强调了 [exllamav2 GitHub 项目](https://github.com/turboderp/exllamav2) 是在 MAX 上改进 LLM 推理的极佳资源，其代码整洁且经过优化。
  
  - *它支持 AMD 的 ROCM* 并能处理多模态模型，使其成为一个极具吸引力的集成选项。
- **MAX 与 exllamav2 集成的可能性**：讨论涉及了 MAX 参与 exllamav2 项目的可能性，该项目提供了诸如批处理推理（batch inference）和精确的 bpw 设置等高级功能。
  
  - *这是一个非常好的项目，但不知为何鲜为人知*，这表明社区希望提高其知名度。
- **理解 Batched MatMul 错误**：一名成员指出一个与 batched MatMul 相关的错误：'constraint failed: max rank for batched matmul is currently 4'。
  
  - 这表明 Mojo 标准库的矩阵运算中存在约束限制，突出了可能需要澄清或改进的领域。

 

**提及的链接**：[GitHub - turboderp/exllamav2: A fast inference library for running LLMs locally on modern consumer-class GPUs](https://github.com/turboderp/exllamav2)：一个用于在现代消费级 GPU 上本地运行 LLM 的快速推理库 - turboderp/exllamav2

 

---

### **Modular (Mojo 🔥) ▷ #**[**mojo**](https://discord.com/channels/1087530497313357884/1151418092052815884/1305986171909836840) (33 messages🔥):

> - `Mojo JIT Compiler`
> - `MAX Platform Overview`
> - `UnsafePointer in Mojo`
> - `Mojo's Performance Capabilities`
> - `Mana Project References`

- **Mojo JIT 编译器的二进制大小至关重要**：如果二进制文件较小且能与预编译的二进制文件互操作，那么分发 **JIT 编译器** 是可行的，但建议不要为所有依赖应用提供源代码。
  
  - 一名成员强调，虽然 **MLIR 可以分发**，但编译器对于实现原生代码执行至关重要。
- **MAX 平台介绍**：**MAX** 平台被描述为一套统一的 API 和工具，旨在构建和部署高性能 AI 流水线（pipelines），包含用于模型执行的 **MAX Engine** 等工具。
  
  - 一名成员链接到了 [文档](https://docs.modular.com/max/#how-max-works)，强调了其在部署低延迟推理流水线方面的强大能力。
- **理解 UnsafePointer 的风险**：Mojo 中的 **UnsafePointer** 允许调用未定义行为（undefined behavior），这可能导致灾难性的内存安全问题，正如一名成员详细说明了可能的危险后果。
  
  - 另一名成员指出，与 C/C++ 相比，Mojo 具有更严格的指针规则，有助于最大限度地减少类型混淆（type punning）等风险。
- **Mojo 在底层编程中的未来**：虽然一些成员已经在尝试游戏引擎，但有人澄清说，由于汇编需求，在 Mojo 中创建引导加载程序（bootloaders）可能需要额外的努力。
  
  - 该语言在各种系统编程任务中的潜力取决于其发展，尽管在游戏引擎方面已经取得了一些直接成果。
- **Mana 项目名称层出不穷**：成员们拿 **Mana** 这个名字开玩笑，引用了现有的项目如 [mana.js](https://github.com/bjorn/mana.js/) 和在 [3rd-Eden's mana](https://github.com/3rd-Eden/mana) 发现的另一个仓库。
  
  - 这些玩笑反映了各种项目采用该名称的可能性，表明了科技界命名文化现象的一种持续趋势。

**提及的链接**：

- [Microsoft Azure Network Adapter (MANA) 概览](https://learn.microsoft.com/en-us/azure/virtual-network/accelerated-networking-mana-overview)：了解 Microsoft Azure Network Adapter 如何提高 Azure VM 的网络性能。
- [什么是 MAX | Modular 文档](https://docs.modular.com/max/#how-max-works)：MAX 平台概览，包括其功能和包含的内容。

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1306023832490086502) (2 条消息):

> - `GenAI pipelines`
> - `RAG workflows`
> - `Dynamic section retrieval`

- **使用 LlamaIndex 构建 GenAI pipelines！**：学习如何使用 LlamaIndex、[@qdrant_engine](https://twitter.com/qdrant_engine) 和 [@MLflow](https://twitter.com/MLflow) 构建健壮的 GenAI pipelines，以推进 RAG 系统。
  
  - 通过这份[分步指南](https://t.co/aZ4GIyGRQM)，探索如何简化 RAG workflows，确保跨模型版本的性能一致性，并优化索引系统以提高效率。
- **在 RAG 中引入 Dynamic Section Retrieval**：我们很高兴推出一种新的 RAG 技术——**dynamic section retrieval**，它能够从文档中检索完整的连续章节，而不是碎片化的 chunks。
  
  - 这种方法解决了社区强调的关于多文档 RAG 的主要痛点，在该[文章](https://t.co/vP2J2arhf4)中进行了进一步讨论。

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1306160133285351546) (35 条消息🔥):

> - `Vocera Launch on Product Hunt`
> - `RAG vs Reporting Debate`
> - `Chatbots in Corporate Settings`
> - `Blockchain Engineering Expertise`

- **Vocera 在 Product Hunt 上线**：Vocera 在 [Product Hunt](https://www.producthunt.com/posts/vocera) 上发布，旨在帮助 AI 开发者以 10 倍的速度测试和监控 voice agents。
  
  - 团队请求支持和反馈，以增强在 AI 社区的知名度和影响力。
- **RAG vs 报告 – 一场对话**：围绕报告与 RAG 的价值展开了讨论，强调在企业中，报告仅代表解决问题努力的 10%。
  
  - 参与者强调了信息检索在报告生成过程中的重要性，并警惕误导性的营销策略。
- **Chatbots – 不受高管青睐**：几位成员指出，高层管理人员通常更喜欢报告格式，而不是 chatbot 交互。
  
  - 尽管有这种偏好，但 chatbots 被公认为是在组织内进行搜索的有用工具。
- **来自区块链工程师的见解**：一位用户详细介绍了他们的区块链工程专业知识，涵盖了 Rust、Solidity 和 EVM 等技术，以及各种应用类别。
  
  - 他们还强调了使用 React 和 Flutter 等框架的前端开发技能，并将其与智能合约集成。

**提到的链接**：

- [来自 jason liu (@jxnlco) 的推文](https://x.com/jxnlco/status/1856411798255333840)：RAG 被高估了。报告才是真正的游戏规则改变者。这不仅仅是为了节省回答问题的时间，而是为了生成驱动业务成果的高价值决策工具。
- [Vocera - 通过模拟和监控更快地发布 voice agents | Product Hunt](https://www.producthunt.com/posts/vocera)：Vocera 帮助 AI 开发者以 10 倍的速度构建生产级 voice agents。它生成对抗性场景，模拟真实通话，并为您的 agents 提供可操作的见解。它还监控...

---

### **LlamaIndex ▷ #**[**ai-discussion**](https://discord.com/channels/1059199217496772688/1100478495295017063/1306178493838655551) (1 条消息):

> - `Vocera Launch`
> - `AI Voice Agents`

- **Vocera 在 Product Hunt 上线！**：令人兴奋的消息，**Vocera** 现已在 [Product Hunt](https://www.producthunt.com/posts/vocera) 上线！该工具允许开发者自动测试和监控 **AI Voice Agents**，将构建生产级 agents 的过程加快 **10 倍**。
  
  - 团队鼓励反馈和支持，表示这将帮助他们触达更多可能从 **Vocera** 中受益的用户。
- **开发者与 Vocera 的互动**：开发者对社区在分享关于 **Vocera** 的想法方面所给予的支持表示感谢。他们强调用户反馈对于触达更多可能从该产品中受益的潜在用户至关重要。

**提到的链接**：[Vocera - 通过模拟和监控更快地发布 voice agents | Product Hunt](https://www.producthunt.com/posts/vocera)：Vocera 帮助 AI 开发者以 10 倍的速度构建生产级 voice agents。它生成对抗性场景，模拟真实通话，并为您的 agents 提供可操作的见解。它还监控...

---

### **Cohere ▷ #**[**questions**](https://discord.com/channels/954421988141711382/1168411509542637578/1306080520144425021) (6 messages):

> - `Payment Management Functionality` (支付管理功能)
> - `Card Decline Issues` (信用卡被拒问题)
> - `Cohere API Internal Server Error` (Cohere API 内部服务器错误)
> - `Rerank API Best Practices` (Rerank API 最佳实践)

- **咨询支付管理的自动扣费功能**：一位用户询问支付管理系统是否提供**自动扣费 (auto-charge) 功能**，还是必须根据使用情况手动检查并支付。
  
  - *一位用户对费用波动相关的支付协议表示困惑。*
- **持续的信用卡被拒**：另一位用户在添加支付方式时遇到信用卡被重复**拒绝 (declined)** 的错误，尽管尝试了两张不同的卡。
  
  - 建议通过电子邮件联系 [**support@cohere.com**](mailto:support@cohere.com) 寻求帮助。
- **API 调用出现内部服务器错误**：一位用户报告在调用 Cohere API 时收到 **500 internal server error**，该问题已持续数日。
  
  - 错误信息提到已向其开发人员提交报告以寻求解决。
- **Rerank API 查询性能问题**：关于 `v2/rerank` API 中 `query` 字段的**最佳实践**咨询，用户注意到查询的微小变化会导致 `relevanceScore` 发生剧烈变化。
  
  - 示例显示，查询 **'volume rebates'** 的得分为 ~0.998，而 **'rebates'** 的得分仅为 ~0.17，这引发了对模型响应性的困惑。

**提到的链接**：[Rerank Best Practices — Cohere](https://docs.cohere.com/docs/reranking-best-practices#queries)：优化端点性能的技巧，包括对文档数量、每个文档的 token 数以及每个查询的 token 数的限制。

---

### **Cohere ▷ #**[**api-discussions**](https://discord.com/channels/954421988141711382/1168578329423642786/1306020522970058774) (19 messages🔥):

> - `API Errors` (API 错误)
> - `Recent Fix Deployment` (最近的修复部署)
> - `Reranking Issues` (Reranking 问题)
> - `Endpoint Troubles` (端点故障)
> - `Model Usage` (模型使用)

- **API 500 错误困扰用户**：多位用户报告 API 调用出现 **500 internal server errors** 已持续数日，并对 API 是否发生变更表示困惑。
  
  - 一位用户提供了 **error ID** 'dbf879fc0e4f4bd7a98102ac41aa0566' 以供进一步调查。
- **修复部署结果不一**：虽然已部署修复程序，但部分用户仍面临问题，促使 Michael 要求提供 **HTTP 请求详情** 进行排查。
  
  - 尽管进行了更新，用户反馈显示问题依然存在，表明需要进一步调查。
- **关于 Reranking 功能的澄清**：Michael 观察到在用户的代码片段中未使用 `return_documents` 参数似乎与当前的错误无关，认为这可能不是问题的根源。
  
  - 他请求用户提供 **error ID**，以便深入查看日志以获取更多信息。
- **讨论升级到生产环境 API**：一位用户提到升级到**生产环境 API key**，希望在 Cohere 获得更稳定的体验。
  
  - 这次升级表明了在解决当前问题后继续使用 Cohere 服务的意愿。
- **代码中的 Reranking 功能**：一位用户分享了使用 Cohere 进行 Reranking 的代码片段，包括 'query' 和 'documents' 等参数。
  
  - Michael 对 **error ID** 的请求强调了通过协作有效解决用户特定问题的方法。

---

### **Cohere ▷ #**[**projects**](https://discord.com/channels/954421988141711382/1218409701339828245/1306350619287617616) (1 条消息):

> - `Vision Language Action 模型基准测试`
> - `机器人研究合作`
> - `VLM & VLA 性能评估`

- **Vision Language & Action 模型基准测试发布**：今天，一篇名为 [*Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks*](https://arxiv.org/abs/2411.05821) 的新论文发布，重点介绍了 **Manifold**、**Georgia Tech**、**MIT** 和 **Metarch AI** 之间的合作。
  
  - 这项研究剖析了新兴的 Vision Language Action 模型，并评估了这些模型在 **20 个不同的真实世界任务**中控制机器人的效果。
- **探索 GPT4o 等 SOTA VLM**：论文还涵盖了一些最先进的 Vision Language Models (VLM)，如 **GPT4o**，以及它们在各种机器人任务中的表现。
  
  - 这项工作代表了为新一类**多模态动作模型 (multimodal action models)** 构建更广泛基准的第一步。
- **邀请对研究见解提供反馈**：作者正在寻求对其研究结果的反馈，并为感兴趣的读者分享了相关链接。
  
  - 查看 [包含要点的推文串](https://x.com/HarshSikka/status/1856739777208574151) 以获取更多关于他们工作的见解。
- **在 Multinet 上访问更多资源**：欲了解更多信息，可以探索该项目的[网站](https://multinet.ai/static/pages/Multinetv01.html)和[代码](https://github.com/ManifoldRG/MultiNet/tree/main)。
  
  - 这些资源提供了研究中使用的实验设置和模型的详细信息。

**提到的链接**：[harsh (@HarshSikka) 的推文](https://x.com/HarshSikka/status/1856739777208574151)：很高兴分享我们的新论文 "Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks"。我们评估了 VLM & VLA 模型在 20 个不同真实世界任务中控制机器人的能力...

---

### **Cohere ▷ #**[**cohere-toolkit**](https://discord.com/channels/954421988141711382/1254901651081269268/1306009545394618448) (3 条消息):

> - `事件的 ICS 支持`
> - `文件内容查看功能`

- **添加 ICS 支持至关重要！**：@danylo_boiko 表示，考虑到 Discord 服务器上举办的活动数量，如果不实现对 **ICS 文件**的支持将是一种“犯罪”。
  
  - 这一观点得到了成员们的积极响应，其中一人表示：“哟，这太棒了！”
- **查看已上传文件的新功能**：@danylo_boiko 还介绍了**查看已上传文件内容**的功能，尽管他开玩笑说这个功能缺少一个“史诗般的介绍”。
  
  - 成员们反应热烈，其中一人评论道：“哇，太喜欢了！”。

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general**](https://discord.com/channels/1104757954588196865/1104757955204743201/1306107507722158080) (6 条消息):

> - `Docker 镜像`
> - `版本打标签`
> - `Axolotl 发布更新`

- **Docker 镜像已构建并打标签**：main 分支的 Docker 镜像已经构建完成，并提醒需要为版本发布打上标签。
  
  - 一位成员提出，要确保正确打标签，以便进行有条理的版本控制和即将到来的发布。
- **即将发布的 0.5.1 版本**：本周有几项功能已准备好用于 **0.5.1 版本**的发布，同时计划为 Docker 镜像打标签。
  
  - 团队正在确认其[最新 Pull Request](https://github.com/axolotl-ai-cloud/axolotl/pull/2051/files) 中实现的正确性，以确保版本管理的妥当。
- **团队祝贺**：一位成员祝贺团队取得的进展，强调了积极的团队精神。
  
  - 团队持续的协作和成就得到了组内成员的认可。

**提到的链接**：[由 winglian 提交的 Pull Request #2051：确保在 Docker 中为发布的版本打上标签](https://github.com/axolotl-ai-cloud/axolotl/pull/2051/files)：描述、动机和背景、如何测试、屏幕截图（如果适用）、更改类型、社交账号（可选）。

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**other-llms**](https://discord.com/channels/1104757954588196865/1104758057449308220/1306031642682589316) (5 条消息):

> - `Qwen2.5 Coder Sizes`
> - `Performance Comparison of Qwen2.5`

- **探索 Qwen2.5 Coder 尺寸**：一位成员分享了一个 [YouTube 视频](https://youtu.be/WPziCratbpc)，供那些纠结于选择哪种 **Qwen2.5 Coder 尺寸**的人参考。
  
  - 该视频对比了各种尺寸并详细讨论了它们的性能。
- **Qwen2.5 的快速生成**：另一位成员提到 **Qwen2.5** 的生成速度令人印象深刻，并询问了所使用的配置。
  
  - 他们专门询问了与 **Sonnet 3.5** 等托管模型的对比情况。
- **在 3090 硬件上运行 Qwen2.5**：用户 **volko76** 透露他们正在 **NVIDIA 3090** 上运行 Qwen2.5，从而提升了生成速度。
  
  - 这一硬件选择突显了运行高需求模型时的性能提升。
- **YouTube 上的模型对比见解**：Volko76 推荐了另一个名为 [**'Qwen2.5 Coder 32B vs GPT4o vs Claude 3.5 Sonnet'**](https://youtu.be/Xs0EkLYu6hw?si=95JJjVKRPknvEUsw) 的 YouTube 视频。
  
  - 该视频旨在确定哪个模型最强，并对其能力进行了详细分析。

**提到的链接**：[Qwen2.5 Coder 32B vs GPT4o vs Claude 3.5 Sonnet (new)](https://youtu.be/Xs0EkLYu6hw?si=95JJjVKRPknvEUsw)：让我们看看哪个模型最强。

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general-help**](https://discord.com/channels/1104757954588196865/1110594519226925137/1306101448974798941) (11 条消息🔥):

> - `qwen-2.5-7B-Instruct configuration`
> - `train_on_inputs clarification`
> - `SFTrainer and TRL comparison`

- **qwen-2.5-7B-Instruct 的配置非常简单**：一位用户询问 **qwen-2.5-7B-Instruct** 的配置，另一位成员建议使用 `examples/qwen/qlora.yml` 中的常规 Qwen 配置。
  
  - 这种友好的交流展示了社区分享实用解决方案的意愿。
- **Train_on_inputs 标志引发困惑**：讨论涉及了使用 `train_on_inputs = False` 的影响，一位成员澄清了在没有提示序列的情况下的训练机制。
  
  - 引用了一个 [GitHub 讨论](https://github.com/tloen/alpaca-lora/issues/255#issuecomment-1504111165) 以进一步了解该标志的功能。
- **SFTrainer 标志与 axolotl 的对比**：成员们讨论了 **SFTrainer** 与 axolotl 中提供的 `train_on_inputs` 标志相关的功能。
  
  - 提供了一个资源链接，用于获取更多关于在 TRL 框架背景下仅针对补全内容 (completions) 进行训练的信息。

**提到的链接**：

- [train_on_inputs clarification · Issue #255 · tloen/alpaca-lora](https://github.com/tloen/alpaca-lora/issues/255#issuecomment-1504111165)：当你提到 train_on_inputs = False 时，我假设你的意思是屏蔽掉 prompt，仅针对模型应该产生的 response 训练 loss。这可能会引起一些困惑...
- [Supervised Fine-tuning Trainer](https://huggingface.co/docs/trl/sft_trainer#train-on-completions-only)：未找到描述

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**announcements**](https://discord.com/channels/1104757954588196865/1113462842436354149/1306279405227741294) (3 条消息):

> - `Release of version 0.5.0`
> - `Feedback and Issues Reporting`

- **0.5.0 版本终于发布！**：经过数月的努力，团队宣布发布 **0.5.0** 版本，现在可以通过 `pip install axolotl` 进行安装。
  
  - 更新内容包括改进和新功能，详见 [GitHub release 页面](https://github.com/axolotl-ai-cloud/axolotl/releases/tag/v0.5.0)。
- **社区庆祝发布**：成员们对新版本表示兴奋，有人评价道：“太棒了！”，另一位则说：“太酷了！”。
  
  - 这种热情展示了社区对更新的支持和渴望。
- **开放征集反馈**：团队鼓励用户在指定频道分享反馈并报告任何问题。
  
  - 这一邀请突显了他们对社区参与和持续改进的承诺。

**提到的链接**：[Release v0.5.0 · axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl/releases/tag/v0.5.0)：变更内容包括：由 @NanoCode012 在 #1197 中修复(log)：改进警告以澄清 lora_modules_to_save 需要一个列表；由 @JohanWork 在 #1196 中添加：Colab 示例；由 @m... 在 Feat/chatml 中添加系统消息。

---

### **DSPy ▷ #**[**show-and-tell**](https://discord.com/channels/1161519468141355160/1202371242519441499/1306101720354394114) (1 条消息):

> - `Forge Reasoning API`
> - `Nous Chat Evolution`

- **Nous Research 推出 Forge Reasoning API**：Nous Research 发布了 [Forge Reasoning API](https://nousresearch.com/introducing-the-forge-reasoning-api-beta-and-nous-chat-an-evolution-in-llm-inference/) 的 beta 版本，承诺在 LLM 推理能力方面取得重大进展。
  
  - 这一进展标志着增强 AI 系统内推理过程的关键一步，展示了新模型与优化技术的融合。
- **Nous Chat 迎来升级**：伴随 Forge API，**Nous Chat** 也将进行演进，加入改进用户交互和可访问性的高级功能。
  
  - 随着这次演进，重点在于通过增强的 LLM 技术和方法论提供更丰富的对话体验。

---

### **DSPy ▷ #**[**general**](https://discord.com/channels/1161519468141355160/1161519469319946286/1306210845965025323) (8 条消息🔥):

> - `DSPY Comparative Analysis`
> - `Zero Shot vs Few Shot Prompting`
> - `LLM Call Tracking`
> - `GitHub Notebooks`

- **DSPY 对比分析讨论**：成员们讨论了使用 **DSPY** 在特定领域进行 **zero shot** 和 **few shot prompting** 对比分析的经验。
  
  - 一位成员询问其他人是否使用了 GitHub 模板来辅助此类分析。
- **共享 DSPY 资源**：一位成员分享了一个 [Colab notebook](https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/intro.ipynb) 链接，以帮助他人开始使用 DSPY。
  
  - 另一位成员引用了另一个 notebook，并强调了它在自己涉及代码相似度工具的项目中可能具有的实用价值。
- **使用 LLM 方法评估工具**：一位成员提到在尝试使用 **LLM** 创建代码相似度工具时，评估了 **zero shot** 与 **few shot** prompting。
  
  - 他们提到了自己参与的另一个 GitHub 资源，用于比较不同的方法和结果。
- **关于 LLM 调用追踪的咨询**：在 Bootstrap、COPRO 和 MIPROv2 等优化过程中进行 **LLM 调用追踪**被作为一个潜在的关注话题提出。
  
  - 一位成员询问了以编程方式追踪这些调用以评估优化期间性能的可行性。

**提到的链接**：

- [Google Colab](https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/intro.ipynb)：未找到描述
- [dspy/examples/intro.ipynb at 421cdd1776041b61bde1c5f9ba3cff827cf5ac2a · stanfordnlp/dspy](https://github.com/stanfordnlp/dspy/blob/421cdd1776041b61bde1c5f9ba3cff827cf5ac2a/examples/intro.ipynb#L11)：DSPy：用于编程（而非提示）基础模型的框架 - stanfordnlp/dspy

---

### **OpenInterpreter ▷ #**[**general**](https://discord.com/channels/1146610656779440188/1147665339266650133/1306047445276295188) (5 条消息):

> - `Developer branch updates`
> - `UI improvements`
> - `Streamed responses handling`
> - `Open Interpreter`
> - `Text editor development`

- **喜爱开发分支的更新**：成员们对开发分支（developer branch）的**最新更新**表达了热情，强调了其带来的积极影响。
  
  - *正是这些细节*带来了显著的差异，让社区感到欣喜。
- **UI 升级带来巨大差异**：对 **UI 改进** 的反馈非常积极，一位成员表示它变得更好了。
  
  - 对**流式响应（streamed responses）**的处理不再让人感到眼花缭乱（不再有诱发癫痫般的闪烁感），展示了团队的出色工作 🚀。
- **对 Open Interpreter 的兴奋**：一位成员分享了他们的兴奋之情，宣称 **Open Interpreter 太棒了！**
  
  - 他们还询问了其他人对**构建文本编辑器**的兴趣，这表明了一个潜在的协作领域。

---

### **OpenInterpreter ▷ #**[**ai-content**](https://discord.com/channels/1146610656779440188/1149229778138824765/1306246698007265393) (3 条消息):

> - `OpenCoder`
> - `AI 泡沫预测`
> - `GPU 投资`
> - `互联网泡沫类比`

- **OpenCoder：代码模型领域的颠覆者**：这段 [YouTube 视频](https://www.youtube.com/watch?v=DurejOD5FTk) 探讨了 OpenCoder，这是一个旨在打造卓越代码语言模型的开源指南（cookbook），强调了其前沿优势。
  
  - 观众们对 OpenCoder 如何潜在地超越现有模型并重塑行业格局深感兴趣。
- **即将到来的 AI 泡沫破裂**：一篇文章警告称 **AI 泡沫即将破裂**，将当前趋势与 **1999 年互联网泡沫**进行了对比，特别是在大规模 **GPU 投资**仅产生极低收入的背景下。
  
  - 该文章概述了 **6000 亿美元**的 GPU 支出与仅 **34 亿美元**收入之间的巨大差距，暗示当今 AI 企业的未来处境危险。
- **AI 与互联网崩溃之间的相似之处**：当前 AI 领域的基础设施建设反映了互联网时代的策略，即公司在变现路径不明朗的情况下大量投资硬件。
  
  - 它强调了在公司追逐理论上的未来需求而缺乏成熟盈利途径的时代，模仿过去 **Pets.com** 案例的风险。

**提到的链接**：

- [The AI Bubble is About to Pop. Here's Who Dies First](https://chrisbora.substack.com/p/the-ai-bubble-is-about-to-pop-heres)：无人准备面对的 6000 亿美元大屠杀（以及隐藏的 3 万亿美元机会）
- [Why This Open-Source Code Model Is a Game-Changer!](https://www.youtube.com/watch?v=DurejOD5FTk)：在这段视频中，我们将探索 OpenCoder，这是一个用于顶级代码语言模型的开源指南。OpenCoder 是一款超越了……的前沿代码模型。

---

### **LAION ▷ #**[**general**](https://discord.com/channels/823813159592001537/823813160075132991/1306146574396559372) (4 条消息):

> - `MongoDB 解雇事件`
> - `社会服务与就业`
> - `EPOCH 58 COCK 模型更新`

- **MongoDB 解雇引发法律和伦理担忧**：一名成员对被 **MongoDB** 解雇表示震惊，详细描述了非法行为的经历，例如通过 Slack 被解雇，以及在受雇期间被拒绝访问公司资源。
  
  - *他们强调公司未能向意大利共和国（Repubblica Italiana）通报其失业状态，导致了严重的个人后果*，如无法获得医疗保健和社会服务。
- **解雇后在社会服务方面的困境**：该成员详细说明了由于 MongoDB 据称未能履行法律义务，他们在申请**公共就业机构**、遣散费和医疗保健方面面临的重大挑战。
  
  - *他们讲述了极度困难的经历*，包括因无法加入必要的社会服务而导致的无家可归和健康问题。
- **EPOCH 58 COCK 模型进展**：一名成员分享了关于 **EPOCH 58 COCK** 的更新，指出 **60M** 参数的 **vit-s** 有所改进，模型功能也得到了增强。
  
  - 他们评论说 **legs are coming in**（腿部正在显现）且 **cockscomb is becoming more defined**（鸡冠变得更加清晰），标志着模型能力取得了积极进展。

---

### **LAION ▷ #**[**research**](https://discord.com/channels/823813159592001537/824374369182416994/1306354438662783048) (3 条消息):

> - `Vision Language Action Models`
> - `Watermark Anything`
> - `Robotic Learning Tasks`
> - `AI Generators Performance`

- **关于 Vision Language Action Models 的新论文发布**：一篇题为《[Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks](https://arxiv.org/abs/2411.05821)》的新论文评估了 Vision Language Models 在 20 个不同现实任务中的表现，展示了 Manifold、Georgia Tech、MIT 和 Metarch AI 之间的合作。
  
  - 该工作旨在对以 **GPT4o** 为代表的这类新兴模型进行画像，标志着迈向更广泛的多模态 Action Models 基准测试的第一步。
- **Watermark Anything 的 GitHub 仓库**：项目“[watermark-anything](https://github.com/facebookresearch/watermark-anything)”提供了带有局部消息水印的官方实现。
  
  - 该模型据称仅有 **1M 参数**，这可能使其能够快速集成到各种 AI 生成器中。

**提到的链接**：

- [harsh (@HarshSikka) 的推文](https://x.com/HarshSikka/status/1856739777208574151)：很高兴分享我们的新论文 "Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks"。我们评估了 VLM 和 VLA 模型在 20 个不同现实场景中控制机器人的能力...
- [GitHub - facebookresearch/watermark-anything: 论文 "Watermark Anything with Localized Messages" 的官方实现](https://github.com/facebookresearch/watermark-anything)：论文 "Watermark Anything with Localized Messages" 的官方实现 - facebookresearch/watermark-anything

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-questions**](https://discord.com/channels/1280234300012494859/1280370030609170494/1306007383268524074) (4 条消息):

> - `TapeAgents 查询`
> - `用于企业工作流的 AI Agents`
> - `通过 Tape 进行通信`

- **Tape 作为与 Agents 通信的媒介**：一位成员询问 **Tape** 是否可以作为人类与 Agent 之间通信或变更声明的媒介，并寻求相关的支持文档。
  
  - 这引发了关于如何将遇到错误的 Agent Tape 条目发布到队列的指导请求。
- **TapeAgents 相关资源**：针对关于 **TapeAgents** 的查询，另一位成员分享了一个 [GitHub 入门 Notebook](https://github.com/ServiceNow/TapeAgents/blob/main/examples/intro_clean.ipynb) 和一篇相关的 [论文](https://www.servicenow.com/research/TapeAgentsFramework.pdf)。
  
  - *该成员表示他们已经阅读了所有提供的资源*，说明他们之前已经查看过建议的材料。

 

---

### **Alignment Lab AI ▷ #**[**general**](https://discord.com/channels/1087862276448595968/1095458248712265841/1306165252236247090) (2 条消息):

> - `Latent Toys 网站`

- **新网站提醒：Latent Toys**：一位成员分享了一个新创建的网站 [Latent Toys](https://latent.toys/) 的链接，强调这是一个值得关注的项目。
  
  - 他们提到该网站是由一位朋友开发的，进一步增加了其重要性。
- **社区关注新项目**：成员们讨论了一位朋友通过 [Latent Toys](https://latent.toys/) 网站发起的项目，强调了其在社区中的重要性。
  
  - 该公告引发了人们对新网站提供的内容的兴趣和好奇。

 

**提到的链接**：[latent.toys](https://latent.toys/)：未找到描述

 

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #**[**leaderboard**](https://discord.com/channels/1111172801899012102/1214705495974092810/1306388235836063866) (2 条消息):

> - `Palmyra X 004 模型`
> - `Writer 处理程序集成`
> - `Pull Request 评审流程`

- **为 Writer 模型和 Palmyra X 004 提交了 PR**：一位成员宣布提交了一个 [PR](https://github.com/ShishirPatil/gorilla/pull/755)，旨在将 **Writer 模型** 和 **Palmyra X 004** 的支持添加到排行榜中。
  
  - 成员对评审表示了感谢，并分享了链接到该 PR 的预览图像。
- **对 PR 评审的快速响应**：另一位成员迅速做出回应，表示将查看提交的 PR。
  
  - 在此确认中表达了 *Thank you!*，突显了社区的积极参与。

 

**提到的链接**：[[BFCL] Add support for Writer models and Palmyra X 004 by samjulien · Pull Request #755 · ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla/pull/755)：此 PR 为 BFCL 增加了对 Writer 模型和我们最新的 Palmyra X 004 的支持。谢谢！

 

---

### **AI21 Labs (Jamba) ▷ #**[**general-chat**](https://discord.com/channels/874538902696914944/874538902696914947/1306385756247298058) (2 条消息):

> - `Legacy models deprecation`
> - `Open source solutions`

- **用户对 Legacy models 弃用感到不满**：成员们对 **Legacy models 的弃用 (deprecation)** 表达了沮丧，表示新模型无法提供相同的输出质量。
  
  - *这次弃用对那些依赖旧模型近两年的用户来说具有巨大的破坏性*。
- **向 Open source 方案的转型仍在进行中**：用户正忙于转向 **Open source solution**，尽管他们此前一直愿意为旧模型付费。
  
  - *“我们如何确保 AI21 未来不会也弃用新模型？”* 突显了他们对未来产品稳定性的担忧。

 

---

---

---

---

{% else %}

> 为了邮件阅读，完整的逐频道分析已被截断。
> 
> 如果您想查看完整的分析，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
> 
> 如果您喜欢 AInews，请 [分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}