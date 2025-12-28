---
companies:
- github
- anthropic
- google-deepmind
- openai
- weights-biases
date: '2024-10-30T01:05:11.702248Z'
description: '**GitHub 第十届年度 Universe 大会**推出了**多模型 Copilot**，在全新的选择器界面中集成了 **Anthropic
  的 Claude 3.5 Sonnet**、**Google 的 Gemini 1.5 Pro** 以及 **OpenAI 的 o1-preview** 模型，允许开发者在多家公司的模型之间自由选择。


  此次活动还展示了 **GitHub Spark**，这是一款 AI 原生工具，用于构建自然语言应用程序，提供无需部署的托管服务和集成的模型提示功能。此外，GitHub
  还更新了其 Copilot Workspace，增加了新的代理（agents）和安全自动修复（Autofix）功能。


  **Weights & Biases** 推出了 Weave，支持音频、文本和图像的多模态可观测性，并集成了 OpenAI Realtime API。Twitter
  上的回顾重点介绍了 **tinygrad** 的代码库优化，以及关于生成式 AI 采用和 **Gemini Flash-8B** 成本效益（**每百万 token
  仅需 0.0375 美元**）的讨论。'
id: f8dc17bb-79ec-43be-b087-a224f36f63c5
models:
- claude-3-5-sonnet
- gemini-1.5-pro
- o1-preview
- gemini-flash-8b
original_slug: ainews-github-copilot-strikes-back-3402
people:
- cassidy-williams
- fchollet
- rohanpaul_ai
- jxmnop
title: GitHub Copilot 反击
topics:
- model-picker-ui
- multi-model-integration
- natural-language-applications
- deployment-free-hosting
- model-prompting
- multimodal-observability
- audio-tracing
- codebase-optimization
- price-performance-ratio
---

**GitHub 或许就是你进行 AI-Native 编程所需的一切。**

> 2024/10/28-10/29 的 AI 新闻。我们为你检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **32** 个 Discord（**231** 个频道，**2681** 条消息）。预计为你节省阅读时间（以 200wpm 计算）：**279 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

GitHub 第十届年度 [Universe 大会于今日举行](https://www.youtube.com/watch?v=5ov2NYBdGSw)：


![image](https://gist.github.com/user-attachments/assets/d3fd1ac8-1d34-499a-ad11-febd7502ff12)


大会带来了一系列备受瞩目的发布（[完整博客文章点击此处](https://github.blog/news-insights/product-news/universe-2024-previews-releases/)）：大部分是 GitHub 对热门代码 AI 工具的回应。

1. **多模型 Copilot**：在新的模型选择器 UI 中加入了 Anthropic 的 Claude 3.5 Sonnet、Google 的 Gemini 1.5 Pro 以及 OpenAI 的 o1-preview。Copilot 的基础模型经历了从 Codex、GPT3.5、GPT4、4o 到 4o-mini 的演进，但这是开发者首次可以选择包括 Google 在内的其他公司的模型。这一消息影响巨大，甚至[登上了今日的主流媒体](https://news.ycombinator.com/item?id=41985915)，人们不禁将此与[微软与 OpenAI 伙伴关系“出现裂痕”的报道](https://news.ycombinator.com/item?id=41878281)联系起来。


![image](https://gist.github.com/user-attachments/assets/b8a2b9c1-5e49-4eb4-aafc-4ecb527cb222)


Cassidy Williams 还演示了 [Copilot 新的多文件编辑能力](https://x.com/ashtom/status/1851316495336554663)以及[自定义指令文件](https://docs.github.com/en/copilot/customizing-copilot/adding-custom-instructions-for-github-copilot) —— 这类似于 **Cursor 的 Composer 和 .cursorrules 功能。**

2. [**GitHub Spark**](https://githubnext.com/projects/github-spark)：“*旨在完全使用自然语言构建应用程序的 AI-native 工具。Spark 是功能齐全的微型应用，可以集成 AI 功能和外部数据源，无需管理任何云资源。*” 基本上它是 v0、bolt.new 和 Claude Artifacts 的竞争对手，配备了**免部署托管、可更换主题的设计系统、持久化数据存储以及集成的模型提示功能**。

> “利用创意反馈循环，用户从初始提示词开始，在构建过程中查看应用的实时预览，轻松查看每个请求的选项，并自动保存每个迭代的版本，以便随时进行版本对比。”


![image](https://gist.github.com/user-attachments/assets/4118995e-d456-4fad-a064-b60035f9b519)


演讲者还讨论了最新的 [GitHub Models](https://docs.github.com/en/github-models)（现已结束候补名单），以及去年发布的重磅产品 Copilot [Workspace 和 Code Reviews](https://github.blog/changelog/2024-10-29-github-copilot-code-review-in-github-com-public-preview/)（在现有的 Spec/Plan/Implement 三个 Agent 基础上，新增了 Brainstorm 和 Build/Repair 两个 Agent，并推出了新的 VSCode 扩展）以及[安全 Autofix](https://github.blog/changelog/2024-09-18-now-available-for-free-on-all-public-repositories-copilot-autofix-for-codeql-code-scanning-alerts/) 的更新。

---

**[本期内容由 Weights & Biases 赞助]**：你的 LLM 不再仅仅局限于文本——那么你的可观测性工具为何还要局限于此？

Weights & Biases 的 Weave [现在支持](https://usewb.link/swyx-docs)音频追踪，以及文本、图像和其他模态。只需 3 行代码，即可追踪多模态 AI 栈中的每一个输入、输出和元数据。

在我们的[交互式 Colab 笔记本](https://usewb.link/Hw0HGuU)中亲自尝试吧！

> swyx 评论：这个笔记本看起来很短，但在我看来，精华在于隐藏在“高级用法：结合 Weave 使用 Realtime Audio API”下的 19 个单元格！你可能想不到一个普通的 LLM Ops 产品会这么快更新以支持 OpenAI Realtime API，但看起来 WandB 团队一直在秘密发力。

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

**AI 发展与行业趋势**

- **Tinygrad 优化**：[@jxmnop](https://twitter.com/jxmnop/status/1850975062905516191) 指出，与 PyTorch 相比，tinygrad **专注于减少代码行数**，导致代码库**横向增长**，且变得**人类几乎无法阅读**。

- **AI 模型能力**：[@fchollet](https://twitter.com/fchollet/status/1850967744386384098) 指出，**目前 GenAI 的低采用率**表明仍有增长潜力，这与 40% 采用率的说法相反。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1850956229327405414) 强调了 Gemini Flash-8B 极高的性价比，**每百万 input tokens 仅需 $0.0375**，**每百万 output tokens 仅需 $0.15**。

- **AI 基础设施**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1851039496105836605) 分享了 xAI Colossus 超级计算机的细节，该系统配备 **100,000 块 NVIDIA Hopper GPU**，并计划翻倍至 200,000 块。该系统采用 NVIDIA Spectrum-X Ethernet 平台，支持 **800Gb/s 端口速度**。

**AI 应用与工具**

- **Perplexity Spaces 更新**：[@perplexity_ai](https://twitter.com/perplexity_ai/status/1850950654271111483) 宣布了多项改进，包括**为免费用户提供 5 个文件上传额度**、增强的自定义指令、详细的 Space 概览卡片以及对 Markdown 文件的支持。

- **RAG 进展**：[@togethercompute](https://twitter.com/togethercompute/status/1850939031301099919) 分享了使用 Llama 模型的 Contextual RAG 开源实现，涉及上下文生成、混合搜索和重排序（reranking）。[@llama_index](https://twitter.com/llama_index/status/1851031828125401301) 介绍了使用 MLflow 和 LlamaIndex Workflows 的高级 RAG 系统，用于灵活的编排和评估。

- **AI Agents**：[@omarsar0](https://twitter.com/omarsar0/status/1850897901817364658) 推出了 AI Agents 课程，涵盖构建 agentic AI 系统的基础知识和实用技巧。[@LangChainAI](https://twitter.com/LangChainAI/status/1850930775589519633) 分享了一个使用 LangGraph 进行 Agent 开发的综合代码库。

**AI 研究与模型更新**

- **模型对比**：[@ajayj_](https://twitter.com/ajayj_/status/1850994244095525228) 报告称，根据社区投票，开源视频生成模型 Genmo Mochi 1 的表现优于 Runway、Kling、Luma 和 Pika 模型。

- **优化技术**：[@giffmana](https://twitter.com/giffmana/status/1850988191618326950) 强调了**带偏置的 sigmoid loss** 在提升模型性能方面的有效性。

- **上下文窗口扩展**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1850843153299116171) 提到正在进行的 **1 亿上下文窗口（context window）** LLM 工作以及对 10 亿上下文窗口的研究，这可能会影响 RAG 的未来。

**AI 伦理与社会影响**

- **AI 采用担忧**：[@ylecun](https://twitter.com/ylecun/status/1850866813430911066) 批评了一些科技领袖的优越感，警告不要将追随者视为“低智商”并期望其盲目服从。

- **AI 对生产力的影响**：[@random_walker](https://twitter.com/random_walker/status/1850954894066548763) 对 AI 显著提升生产力的说法表示怀疑，指出尽管使用率达到 3%，但生产力仅增长了 1%。

- **AI 在教育中的应用**：[@svpino](https://twitter.com/svpino/status/1850974043874476365) 告诫不要高估 AI 在构建 SaaS 业务方面的能力，强调 AI 是工具而非完整的解决方案。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1：在消费级硬件上优化 LLM 推理**

- **[在本地 GPU（低端 RTX 3000 系列）上运行 Llama 的最佳方式是什么？对 Python 调用和 GUI 界面都感兴趣。这个领域发展太快了，希望能得到最新的建议！谢谢](https://i.redd.it/fjoj2aym3hxd1.png)** ([Score: 39, Comments: 23](https://reddit.com//r/LocalLLaMA/comments/1gdymkw/whats_the_best_way_to_run_llama_on_a_local_gpu/)): 对于在低端 **RTX 3000** 系列 GPU 上运行 **Llama models**，目前的建议包括使用 **llama.cpp** 或 **text-generation-webui** 作为 GUI 界面，以及使用 **transformers** 库配合 **bitsandbytes** 进行 **Python** 集成。这些方法可以在消费级硬件上实现高效的 **quantization** 和 **inference**，尽管具体性能可能因模型大小和可用 **VRAM** 而异。
  - 推荐使用搭配 **Open webui** 的 **Ollama**，一些用户通过 **Docker** 容器运行它，并利用 **HTTP calls** 进行集成。作者建议使用 [Harbor](https://github.com/av/harbor) 来部署基于 **Docker** 的完整 **LLM stack**。
  - 用户使用了各种界面：**mikupad** 用于配合 **llama.cpp** 写作，**TabbyAPI** 配合 **LLama-3.1** 或 **3.0** 模型集成到 **silly tavern** 中，以及 **Lm studio** 或 **Aya** 用于 GUI 和 **OpenAI API** 兼容性。
  - 一些人更倾向于自定义设置，例如在脚本中运行 **llama.cpp** 进行纯文本创作，并强调了 **alternative token selection**（备选 Token 选择）的重要性，而这在其他 UI 选项中可能缺失。
- **[按品牌、年份和细分市场划分的移动 SoC AI 评分](https://i.redd.it/28am1cfkcgxd1.jpeg)** ([Score: 43, Comments: 6](https://reddit.com//r/LocalLLaMA/comments/1gdwm4o/ai_scores_of_mobile_socs_by_brand_year_and_segment/)): 该帖子分析了来自 [ai-benchmark.com](https://ai-benchmark.com/ranking_processors.html) 的移动 **SoC** 的 **AI performance benchmarks**，揭示了旗舰级和高端细分市场之间巨大的性能差距。值得注意的发现包括：**Snapdragon 7+** 系列的表现超出了其品牌定位；**Dimensity** 在最近几代的 AI 性能大幅提升；以及四年前的 **Snapdragon 8 Gen 1** 仍然超越了较新的 Snapdragon 7 系列、8s Gen3 和大多数 Dimensity 处理器；**A17 Pro** 得分为 **3428**，略低于 Snapdragon 8 Gen 3。
  - 用户讨论了在手机上运行 **large language models**，对 **16B deepseek v2 Lite MoE** 和 **Llama 3.1 8b** 等模型表现出兴趣。配备高达 **24GB RAM** 的 **ZTE Z60 Ultra** 被提及能够运行 **12B models**。
  - 关于 Benchmark 测试模型的关联性引发了争论，一些人认为 **TFLOPS**、**TOPS** 和 **memory bandwidth** 规格对于手机上的真实 AI 应用比基于 **Inception V3** 等模型的评分更有参考价值。
  - 用户对 **Mediatek** 芯片组在 AI 任务中的状态表示关注，特别是关于 **GPU** 和 **NPU** 的功能。帖子强调了 **Dimensity** 最近在 AI 性能方面的进步。
- **[更新了 Llama.cpp 的修正设置。推理引擎之战：Llama.cpp vs MLC LLM vs vLLM。针对单卡 RTX 3090 和四卡 RTX 3090 的测试。](https://www.reddit.com/gallery/1ge1ojk)** ([Score: 75, Comments: 51](https://reddit.com//r/LocalLLaMA/comments/1ge1ojk/updated_with_corrected_settings_for_llamacpp/)): **Llama.cpp**、**MLC LLM** 和 **vLLM** 在消费级 GPU 上进行了 **LLM inference** 基准测试，具体测试环境为单块 **RTX 3090** 和四块 **RTX 3090**。该帖子提供了使用修正设置后的 **Llama.cpp** 更新结果，比较了这三种推理引擎在不同 GPU 配置下的性能。
  - 在修正设置后，**Llama.cpp** 的性能显著提升，单 GPU 测试达到 **50-51 tokens/second**，4x GPU 测试达到 **15 tokens/second**。社区建议在未来的基准测试中加入 **exllama**，并探索量化模型的对比。
  - 分享了一篇 [博客文章](https://blog.mlc.ai/2024/10/10/optimizing-and-characterizing-high-throughput-low-latency-llm-inference)，详细介绍了 **multiGPU scaling**、**concurrent requests** 和 **speculative decoding** 的基准测试。用户对 **MLC-LLM** 如何在 1-4 块 GPU 间扩展感兴趣，一位用户报告在使用 **MI60** 显卡时，1 块 GPU 为 **25 tokens/second**，2 块 GPU 为 **34 tokens/second**。
  - 讨论集中在 **PCIE bandwidth** 的使用上，测试显示在 **tensor parallel inference** 期间利用率出奇地低（**0.1 MB/s**）。用户还对基准测试选择 **FP16** 进行了辩论，一些人认为在实际用例中 **Q4** 或 **Q8** **quantization** 更具参考价值。


**主题 2：开源 LLM 在创意和无审查用例方面的进展**

- **三个增强版 Llama 3.2 模型发布，每个均为 7B 参数，用于创意用途 - 无审查。** ([Score: 44, Comments: 19](https://reddit.com//r/LocalLLaMA/comments/1geio97/three_llama_32_models_enhanced_at_7b_each_for/))：三个增强型 **Llama 3.2 7B 模型**已发布，用于创意和无审查用途，每个模型都扩展到了 **67 层**和 **606 个 tensors**。这些模型可在 Hugging Face 上获取，其“去审查”等级评分为 1-10，并具有改进的**指令遵循 (instruction following)**、**细微差别 (nuance)**、**情感 (emotion)** 和**散文深度 (prose depth)**，审查和偏见可通过提示词 (prompts) 进行控制。
  - **Frankenstein 模型**（拼接模型）被批评为经常处于“脑叶切除”状态且表现不佳，用户建议改用调整过设置的**全尺寸模型 (full-size models)**。模型创建者为他的方法辩护，引用了 **45 个改进示例**，并解释了他构建和测试模型的独特方法。
  - 用户 **export_tank_harmful** 赞扬了创建者的工作，特别提到了 **MN-Dark-Planet-TITAN-12B** 和 **L3-Dark-Planet-8B** 模型。他们建议在 Reddit 帖子中包含创建者的 **Hugging Face 名称**以增加可信度，并对持续的 **abliteration**（去拒绝微调）工作表示支持。
  - 关于 **ARM 设备**模型可用性的讨论中，创建者澄清说，针对 ARM 优化的模型文件名以 **Q4_0_4_8.gguf** 结尾。目前，**llamacpp** 仅支持 **3 个版本**的 ARM 优化。
- **用于色情角色扮演的 LLM 推荐** ([Score: 48, Comments: 61](https://reddit.com//r/LocalLLaMA/comments/1ge2fzf/llm_recommendation_for_erotic_roleplay/))：该帖子寻求专门用于**色情角色扮演 (erotic roleplay)** 的 **Large Language Models (LLMs)** 推荐，列出了几个选项，重点关注 **DarkForest V2** 和 **backyardai/Midnight-Rose-70B-v2.0.3-GGUF** 作为顶级竞争者。作者还提到了其他模型，如 **Stheno**、**Lyra 12B V4**、**TheSpice-8b** 以及其他参数范围从 **8B 到 72B** 的模型，但认为它们在这一特定用例中可能较弱。
  - **ArsNeph** 推荐了较新的模型，重点介绍了 **L3 Stheno 3.2 8B**、**Magnum V4**、**UnslopNemo 12B**、**Mistral Small 22B** 及其微调版本如 **Cydonia**。对于更大的模型，他们建议使用 **Midnight Miqu 1.5 70B**、**Euryale 2.1 70B** 和 **New Dawn Llama**。
  - 几位用户支持将 **Midnight Rose** 和 **Midnight Miqu** 作为色情角色扮演的首选。**TheLocalDrummer** 提到一些用户更喜欢 **Behemoth v1.1** 而非 Midnight Miqu，而其他人则建议尝试 [NemoMix-Unleashed-12B](https://huggingface.co/MarinaraSpaghetti/NemoMix-Unleashed-12B) 和 [EVA-Qwen2.5-72B-v0.0](https://huggingface.co/EVA-UNIT-01/EVA-Qwen2.5-72B-v0.0)。
  - 用户建议尽管存在审查，也可以探索 **Gemma-2-27B** 和 **Mistral-Small-22B-ArliA**。


**主题 3：LLM 工具和基础设施的创新**

- **我们刚刚开源了 Promptwright：使用本地 LLM 生成大规模合成数据集** ([Score: 63, Comments: 12](https://reddit.com//r/LocalLLaMA/comments/1ge9192/we_just_open_sourced_promptwright_generate_large/))：**Promptwright** 已发布，这是一个开源的 **Python 库**，用于通过 **Ollama** 使用**本地 LLM** 生成合成数据集。它为数据集生成提供了简单的界面、可配置的指令和系统提示词、**JSONL** 输出格式，并与 **Hugging Face Hub** 直接集成，允许用户在本地处理数千个样本，无需 API 成本或速率限制，同时保持数据隐私。

- **Mistral.rs v0.3.2 获得 26% 的 Metal 性能提升并提供 PyPI wheels！** ([Score: 62, Comments: 16](https://reddit.com//r/LocalLLaMA/comments/1ge9dc7/mistralrs_v032_gets_a_26_metal_performance_boost/))：**Mistral.rs v0.3.2** 引入了通过 PyPI wheels 为各种平台（**Metal**、**CUDA**、**Apple Accelerate**、**Intel MKL** 和纯 CPU）提供的简化安装，并通过优化的 MLX attention kernels 实现了 Metal 解码 **26% 的性能提升**。该更新还包括使用 Marlin GPTQ kernel 和 FP8 量化的 **CUDA 改进**，以及对 **Llama 3.2 Vision** 等模型的支持，并提供了 [GitHub 仓库](https://github.com/EricLBuehler/mistral.rs)、[Python 包文档](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/API.md) 和用于预量化模型的 [UQFF 模型集](https://huggingface.co/collections/EricB/uqff-670e4a49d56ecdd3f7f0fd4c) 链接。

- **[在推理阶段，使用标准 CPU 将任何现成 LLM 的上下文扩展到 10 亿 (1B) 的检索系统：](https://www.reddit.com/gallery/1gejpg2)** ([得分: 63, 评论: 6](https://reddit.com//r/LocalLLaMA/comments/1gejpg2/retrieval_system_extending_any_offtheshelf_llm_to/))：开发出了一种全新的**检索系统**，可以在推理阶段仅使用标准 CPU，将**任何现成的大语言模型 (LLM)** 的上下文长度扩展到 **10 亿 (1B) tokens**。该系统在 [Zyphra 博客文章](https://www.zyphra.com/post/reaching-1b-context-length-with-rag)和一篇 [arXiv 论文](https://arxiv.org/abs/2409.01666)中进行了详细介绍，显著扩展了 LLM 处理和理解海量信息的能力，且无需专门的硬件。
  - 标题中声称的 **"1B 上下文长度"** 被批评为**标题党**，用户指出这指的是向量存储 (vector store) 中的 tokens，而非实际的推理长度。对于一个 **8B 模型**，在 **100 万 (1M) 上下文**下进行推理，在 **A100 GPU** 上大约需要 **3000 秒**。
  - 用户幽默地扩展了这一概念，建议甚至可以宣称 **100B tokens** 或 **100 Petabytes**（引用 **Google 的索引大小**）的上下文长度，以突出此类说法的随意性。
  - 人们对**哈希链检索 (hash chain retrieval) 之外的基准测试**以及潜在应用表现出兴趣，例如创建**小型 LM**（如 **1B 模型**），通过 RAG 加载必要的知识，从而可能实现每秒输出**数千个 tokens**。


**主题 4. AI 文档理解的挑战与实际应用**

- **我是如何利用视觉模型帮我赢得《帝国时代 2》(Age Of Empires 2) 的。** ([得分: 327, 评论: 51](https://reddit.com//r/LocalLLaMA/comments/1ge6fvw/how_i_used_vision_models_to_help_me_win_at_age_of/))：作者开发了 **WololoGPT**，这是一个针对**《帝国时代 2》**的*基于 AI 的教练*，利用**视觉模型**和 **LLM** 提供实时游戏建议，包括资源管理和反制敌人的策略。该项目使用 **Claude 3.5** 和 **Gemini Flash** 进行视觉处理，目前已在 [GitHub](https://github.com/tony-png/WololoGPT) 开源，并在[官方网站](http://www.wolologpt.com)上提供了[视频演示](https://www.youtube.com/watch?v=ZXqVKgQRCYs)和可下载的可执行文件。
  - **Echo9Zulu-** 建议开发一个记录应用程序数据的系统，将 WololoGPT 视为构建关于模型对游戏事件解读的**宝贵训练数据**的机会。他们建议以 **AoE2** 为模板研究模型行为，特别是关注模型如何处理**战争迷雾 (fog of war)** 对策略的影响。
  - 该项目因其推动**视觉模型应用领域尖端技术 (state-of-the-art)** 的潜力而受到赞赏，评论者指出目前关于此类用例的文献还很有限。他们建议被动地记录数据以利用这一机会。
  - **WololoGPT** 被描述为一个“酷炫的构建”，它能在不让人感觉完全作弊的情况下提升游戏体验。开发者确认它确实提高了他们的游戏水平，称其为“一点小小的助力”。

- **文档理解非常非常难：一个例证** ([得分: 34, 评论: 26](https://reddit.com//r/LocalLLaMA/comments/1gekd53/document_understanding_is_very_very_hard_an/))：该帖子通过一个**旧金山游泳池时间表**的例子，阐述了 LLM 在**文档理解方面的困难**。作者挑战读者从**单页传单**中提取**循环往复的往返泳 (lap swim) 时段**，加分任务包括生成 **ical (ics) 格式**并处理**节假日**。作者指出模型经常漏掉**周一**并误读**周三的往返泳时间**。尽管有一些令人印象深刻的能力，作者总结道，即使是**先进的 LLM 也难以完成**六岁小孩就能完成的任务，并警告不要在生产环境中过早部署文档理解功能。
  - 用户批评了**游泳池时间表的布局**，指出其设计糟糕且不一致。一位评论者强调，这种**古怪的布局**在专业场合很常见，并以**并购尽职调查清单 (M&A Diligence Checklists)** 为例。
  - 一名用户使用 **Chat 4.0** 成功提取了时间表，并在 **5 分钟**内编写了一个 **Python 脚本**来生成 **ical 文件**。该脚本可以处理循环事件，但未考虑节假日。
  - **AI Studio** 中的 **Gemini 1.5 Pro** 正确提取了大部分时间表，包括棘手的**周三往返泳时间**，但增加了一个不存在的周日晚间时段。用户讨论了多步推理以及视觉模型处理不同图像分辨率的挑战。

## 其他 AI Subreddit 综述

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 模型发布与能力**

- **更新的 Phi-3 Mini 支持 function calling**：Rubra AI 发布了更新的 Phi-3 Mini 模型，[具备 function calling 能力](https://www.reddit.com/r/LocalLLaMA/comments/1dzhe38/phi3_mini_june_with_function_calling/)，可与 Mistral-7b v3 竞争 (/r/LocalLLaMA)。

- **OpenAI 的 o1 推理模型**：OpenAI CFO Sarah Friar 表示，[律师报告称新的 o1 推理模型可以完成时薪 2000 美元的法律助理的工作](https://v.redd.it/t25hddpblkxd1) (/r/singularity)。

**AI 应用与演示**

- **AI 辅助的多臂采摘苹果机器人**：一段[视频展示了 AI 辅助的多臂机器人](https://v.redd.it/552w8berqhxd1)，能够识别并采摘成熟的苹果 (/r/singularity)。

- **使用 Stable Diffusion 的逼真面部动画**：一位开发者正在为 Meta Quest 开发[基于 Stable Diffusion 的逼真面部动画系统](https://v.redd.it/ut9246li3mxd1)，在 Quest 3 上以 90fps 运行 (/r/StableDiffusion)。

- **具备触觉感知的机器人手**：Robot Era 推出了其[第一代 XHAND](https://v.redd.it/2igc50imhixd1)，具有 12 个自由度，且每个手指都具备触觉感知能力 (/r/singularity)。

- **机器人提供美容服务**：一段[视频显示机器人在洛杉矶做美甲和睫毛](https://x.com/esthercrawford/status/1850681223770947869)，展示了以往由人类主导的服务行业的自动化 (/r/singularity)。

**AI 政策与基础设施**

- **美国政府推动 AI 基础设施建设**：国家安全顾问 Jake Sullivan 表示，[美国需要建设数十甚至数百吉瓦（GW）的能源基础设施](https://v.redd.it/28bxofivbhxd1)来为 AI 数据中心供电，否则将面临落后于竞争对手的风险 (/r/singularity)。

**AI 影响与社会讨论**

- **关于 AI 对就业影响的讨论**：多篇帖子讨论了 AI 对工作的潜在影响，包括[将其与汽车发明后马匹数量的减少进行类比](https://i.redd.it/qhs0yi12whxd1.jpeg) (/r/singularity)。

- **公众对 AI 进展的看法**：一篇帖子讨论了[人们在看到 ChatGPT 时的反应](https://www.reddit.com/r/singularity/comments/1ge4eh0/showed_my_dad_chat_gpt_and_he_is_literally/)，有些人变得非常感兴趣，而另一些人则不以为然 (/r/singularity)。

**迷因与幽默**

- 一篇帖子幽默地建议[使用 Stable Diffusion 制造关于浴缸历史的虚假信息](https://v.redd.it/uuyp7mcq2nxd1) (/r/StableDiffusion)。


---

# AI Discord 纪要

> 由 O1-preview 生成的摘要之摘要的总结

**主题 1：AI 模型发布震撼全场**

- **Stable Diffusion 3.5：中等体量，强大动力！**：Stability.ai 发布了 [Stable Diffusion 3.5 Medium](https://stability.ai/news/introducing-stable-diffusion-3-5)，这是一个拥有 25 亿参数的模型，仅需 **9.9 GB VRAM** 即可运行，使高质量图像生成大众化。
- **Moondream 押注小模型也能大有作为**：[Moondream](https://venturebeat.com/ai/moondream-raises-4-5m-to-prove-that-smaller-ai-models-can-still-pack-a-punch/) 筹集了 **450 万美元**，旨在证明较小的 AI 模型同样有效，从而将行业的焦点从庞大架构中转移。
- **GitHub Copilot 引入 Claude 和 Gemini 实现性能飞跃**：GitHub 的 Copilot 集成了 [Claude 3.5 Sonnet](https://github.blog/news-insights/product-news/bringing-developer-choice-to-copilot/) 和 [Google 的 Gemini 1.5 Pro](https://cloud.google.com/blog/products/ai-machine-learning/gemini-models-on-github-copilot)，为开发者提供了 AI 动力升级。

**主题 2：AI 工具链迎来加速**

- **Unsloth 通过 Gradio UI 简化复杂性**：一位创新者推出了一款 [Gradio 应用](https://huggingface.co/blog/merve/quantization)，利用 Unsloth 简化了模型训练，使无代码爱好者也能触及 AI 开发。
- **ThunderKittens 以闪电般的速度咆哮**：备受期待的 [ThunderKittens 0.000000002](https://arxiv.org/abs/2410.20399) 发布，宣称 **Linear Attention 速度提升了 6-14 倍**，并在 Attention 反向传播中超越了 FA3。
- **开发者钻研 Triton 内核以提升速度**：工程师们讨论了优化 Triton 内核的方法，发现多个内核的性能优于单个内核，并揭示了 BF16 转换带来的挑战。

**主题 3：AI 隐私与安全成为焦点**

- **PAPILLON 翩然而至保护隐私**：研究人员首次推出 [PAPILLON](https://arxiv.org/abs/2410.17127)，在仅有 **7.5% 隐私泄露**的情况下达到了 **85.5% 的质量**，实现了本地和云端 LLM 的安全融合。
- **ChatGPT 的拼写错误让用户感到困惑**：ChatGPT 开始出现拼写错误和乱码，用户对其输出质量的突然下降感到困惑。
- **Apple 悬赏 100 万美元发起黑客挑战**：Apple 悬赏高达 [100 万美元](https://x.com/culturecrave/status/1850781293166067999?s=46)，挑战黑客攻破其 AI 服务器，引发了关于 AI 安全的讨论。

**主题 4：新平台涌现大量 AI 职位**

- **Cracked Engineers 在技术招聘领域开辟新天地**：新推出的 [Cracked Engineers](https://www.crackedengineers.com/) 将 AI 人才与顶尖初创公司联系起来，目前已与 **Weaviate**、**UnslothAI** 等公司达成合作。
- **AI 初创公司寻觅顶尖人才**：**Unsloth AI**、**Julius AI** 和 **Jimini AI** 等公司正在积极招聘，为准备投身前沿 AI 领域的人才提供绝佳机会。
- **求职者福音：定制化新闻通讯即将上线**：Cracked Engineers 宣布推出每周技术职位新闻通讯，允许订阅者通过 **CUDA**、**MLOps** 和 **Software Engineering** 等标签定制内容。

**主题 5：AI 社区活动与洞察精彩纷呈**

- **LLM Agents Hackathon 盛大开幕**：已有超过 **1,000 名创新者**报名参加，[LLM Agents Hackathon](https://rdi.berkeley.edu/llm-agents-hackathon/) 在五个激动人心的赛道上提供了超过 **20 万美元的奖金**。
- **OpenAI CFO 表示：“AI 不再只是实验性的！”**：在一次坦诚的[采访](https://youtu.be/eCqFgVqWbEs)中，OpenAI CFO **Sarah Friar** 宣称 AI 已进入主流，每天都在渗透进银行和金融科技领域。
- **Meta 瞄准自研 AI 搜索引擎**：Meta 的网络爬虫暗示其正在开发新的 [AI 驱动搜索引擎](https://www.theverge.com/2024/10/28/24282017/meta-ai-powered-search-engine-report)，旨在摆脱对 Google 和 Microsoft 的依赖。

---

# 第 1 部分：高层级 Discord 摘要

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Clem 向社区介绍自己**：Hugging Face 的联合创始人兼 CEO Clem 表达了对通过 Discord 与社区成员积极互动的兴奋之情。他强调了强烈的参与意愿，表示：*“我迫不及待想与大家互动”*。
  - 他还推广了一个即将举行的直播工作坊，鼓励成员通过此[链接](https://streamyard.com/watch/JS2jHsUP3NDM)分享关于扩大其知名度和参与度的想法。
- **对 TensorFlow 的不满**：许多成员表达了对 **TensorFlow** 的沮丧，理由包括禁用了 Windows 上的 GPU 支持以及复杂的文档问题，他们通常更倾向于转向 **PyTorch** 以实现更快的开发。
  - 分享的经验反映了社区内对 TensorFlow 的 Bug 和缺乏支持的普遍不满情绪。
- **大麻纳米片的研究**：**大麻衍生碳纳米片**在储能方面显示出作为石墨烯成本效益替代品的潜力，Dr. David Mitlin 的研究确定其可行性成本为**每吨 500 美元**。
  - 这引发了关于军事和航空航天应用的讨论，表明人们对适用于高科技行业的替代材料的兴趣日益浓厚。
- **Swin Transformer v2 讨论**：成员们探讨了使用 **Swin Transformer v2** 处理类图像数据立方体的问题，并讨论了如何针对独特的输入形状调整架构。
  - 一位用户提到利用数据立方体代替传统图像，引发了关于必要架构调整的对话。
- **LangChain SQL Agent 资源共享**：一个详细介绍 **LLaMA2 SQL chat** 的 GitHub notebook 被作为资源分享，用于使用 **LangChain SQL Agent** 开发**上下文感知推理应用**。
  - 该资源旨在帮助用户增强其实现方案，体现了社区对利用现代技术处理 NLP 任务的关注。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Gradio UI 工具简化模型训练**：一位用户创建了一个 [Gradio app](https://huggingface.co/blog/merve/quantization)，可以简化使用 **Unsloth** 训练模型的过程，使其更容易调整设置并将模型上传到 Hugging Face。
  - 这一改进旨在帮助无代码用户，显著提高了 AI 模型训练的可访问性。
- **Unsloth 提供的 AI 工作机会**：Unsloth 正通过 [Cracked Engineers](https://www.crackedengineers.com/) 重点开展招聘活动，旨在吸引 AI 领域的技术人才。
  - 鼓励社区成员在利用该平台进行职位跟踪的同时，探索平台上的职位列表。
- **FP8 微调提升训练速度**：关于在 Unsloth 中采用 **FP8** 进行训练的讨论正在进行中，这暗示了潜在的**速度提升**。
  - 社区提出了关于其具体实现的问题，特别是与基础权重（base weights）和 LoRA 相关的部分。
- **对教育体制的挫败感**：成员们讨论了在学校浪费时间的感受，其中一人表达了想要*有所作为*的愿望。
  - 这种情绪引起了共鸣，其他人也反思了个人经历如何塑造教育观点。
- **关于 Optimizer CPU Offload 的见解**：讨论集中在 **Optimizer CPU Offload** 在提高**低比特（low-bit）**训练框架效率方面的潜力。
  - 通过将操作转移到 CPU，模型可以实现**更快的训练时间**并优化资源利用。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Diffusion 3.5 Medium 模型发布**：**Stable Diffusion 3.5 Medium** 模型已开放免费商业使用，该模型拥有 **25 亿参数**，仅需 **9.9 GB VRAM** 即可在消费级硬件上运行。
  
  - 此次发布旨在通过确保对低端设备的兼容性来扩大 AI 的普及度，从而改变创作者的生态格局。
- **图像质量达到新高度**：用户确认 **Stable Diffusion 3.5 Medium** 在生成超过 1MP 的图像方面表现出色，在 **提示词遵循度 (prompt adherence) 和质量**上优于 **3.5 Large** 变体。
  
  - 然而，一旦图像超过 2MP，模型就开始显得吃力，这表明其扩展能力存在限制。
- **GPU 价格战持续进行**：当前市场趋势显示 **3090** GPU 的价格与 **7900 XTX** 相似，二手 3090 的价格维持在 **690 美元**左右。
  
  - 讨论内容包括 AI 工作负载与游戏性能的 GPU 性能对比，强调了硬件负担能力的动态变化。
- **Sana Autoencoder 评价褒贬不一**：**Sana** 自动编码器承诺提供高效的训练和压缩，但在图像质量结果方面收到了褒贬不一的反馈。
  
  - 部分用户仍持怀疑态度，表示需要对利用该技术的模型进行进一步验证。
- **切换 UI 以增强用户体验**：用户探索了从 **A1111** 切换到 **ComfyUI**，部分用户尝试使用 **SwarmUI** 以简化图像生成流程。
  
  - 对话强调了对不同界面的偏好，以及通过优化 **steps** 和 **cfg** 等设置来提高提示词遵循度。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **面向开发者的 AI 通讯**：一名成员强调了对技术性 AI 通讯的需求，以摆脱以消费者为中心的炒作，并推荐了 [SemiAnalysis](https://link.to.semi-analysis)，因其在 GPU 领域的深入见解。
  
  - 这反映了寻求严肃 AI 讨论的工程师们对更具实质性资源的需求。
- **为角色扮演机器人微调 Hermes 3**：一位用户探索了微调 **Hermes 3** 是否能增强角色扮演机器人对 **character.ai** 的模仿能力，而另一位用户建议利用提示词来实现同样的效果。
  
  - 这一讨论突显了社区对优化 AI 以进行复杂角色互动的兴趣。
- **Meta 发布 Layer Skip 代码**：Meta 推出了 [Layer Skip](https://go.fb.me/s8lary) 以提高 **LLM** 效率，并提供了推理代码和微调后的 **checkpoints**。
  
  - 此次发布旨在激发对 **AI 优化**方法和**可解释性 (interpretability)** 的新研究。
- **GitHub Copilot 扩展模型选择**：**GitHub Copilot** 的重大更新包括新增 **Claude 3.5 Sonnet** 和 **Gemini 1.5 Pro**，为开发者提供更广泛的模型选择。
  
  - 这一转变可能会在 AI 竞争格局中增强 **Anthropic** 的实力。
- **微软与 OpenAI 的复杂关系**：对话表明，由于担心过度依赖以及与 **AGI** 声明相关的风险，微软正在探索 OpenAI 的替代方案。
  
  - 成员们强调了多样化 AI 合作伙伴关系对战略稳定性的重要性。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **加入策展人计划 (Curators Program)！**：Perplexity 团队正在积极招募首批**策展人 (Curators)**，为拥有数百万用户的 **Discover feed** 贡献内容。如果你喜欢制作 **Pinterest boards** 或编辑 **Wikipedia 页面**，可以[在此申请](https://perplexity.ai/curators)。
  
  - 策展人将负责在 **Perplexity** 产品内直接创建能够**启发**并**告知**用户的 **Pages**。
- **Grok 2 现已面向 Pro 用户开放**：Perplexity AI 宣布 **Grok 2** 现在可供 Pro 用户使用，允许他们在设置中将其设为默认模型。一些用户好奇 Grok 2 是否会保持无审查 (uncensored) 状态，尽管其改进似乎有限。
  
  - 这一公告引发了讨论，人们对它是否比之前的版本有显著进步持怀疑态度。
- **周边商品发布公告**：Perplexity AI 正在推出名为 **Perplexity Supply** 的周边商品系列，首批产品将于明天**太平洋时间上午 9 点**发布。他们的口号强调品牌“为好奇心而生”，暗示了一个高度参与的社区。
  
  - 社区的兴奋之情溢于言表，用户们期待着与品牌相关的收藏品和时尚单品。
- **NASA 为美国经济贡献 760 亿美元**：最近的一份报告称，**NASA** 为美国经济贡献了约 **760 亿美元**，这反映了其各种项目和创新。这强调了 NASA 在太空探索之外的影响力，巩固了其在经济增长中的作用。
  
  - 数据表明公共资金带来了显著的**投资回报 (returns on investment)**，为持续的支持提供了令人信服的理由。
- **深入了解光子计算 (Photonic Computing) 的进展**：讨论强调了**光子计算**的进步及其对**网络安全 (cybersecurity)** 领域的影响。预计这些技术将改变数据的处理和安全保障方式。
  
  - 成员们分享了新的见解，表明人们对将光子能力集成到现有框架中的兴趣日益浓厚。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **比亚迪 (BYD) 旨在主导汽车行业**：一段视频讨论了中国电动汽车巨头**比亚迪 (BYD)** 如何通过积极的全球扩张和开设经销店，准备颠覆 **Tesla** 等竞争对手，详见[此视频](https://www.youtube.com/watch?v=VgAGSbreEMI)。
  
  - 讨论强调了比亚迪旨在显著影响汽车市场的创新策略。
- **NotebookLM 增强员工资源的可访问性**：一位用户将 **NotebookLM** 作为员工资源指南，整合了员工手册和 FAQ，以简化内部查询，但注意到外部链接的 URL 生成存在不一致性。
  
  - 这一反馈表明平台内的文档集成需要进一步完善。
- **西班牙语播客生成面临挑战**：用户报告了使用 **NotebookLM** 生成西班牙语播客时的困难，最初成功生成了两集，随后出现问题，导致用户寻求有效的解决方案。
  
  - 人们对影响西班牙语文本生成的底层语言处理问题表示担忧，表明存在必要的改进空间。
- **探索 NotebookLM 的开源替代方案**：社区成员正在评估 **NotebookLlama**，这是一个利用 Meta 技术的开源替代方案，但正如 [Notebook Llama 链接](https://www.notebookllama.ai/)中所讨论的，人们对该网站的可信度持怀疑态度。
  
  - 参与者辩论了开源解决方案的优势，并指出了可能存在的 DNS 问题和注册合法性问题。
- **实时化身 (Real-Time Avatars) 彻底改变播客**：在播客中集成 **Simli** 以实现实时化身引起了关注，它允许使用音频分段 (audio diarization) 来实现同步视觉效果，从而增强观众参与度。
  
  - 这一概念验证强调了播客中动态演示风格的巨大潜力。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Unsloth Kernels 增强 LLM 微调**：一位成员询问了关于 [unsloth kernels](https://github.com/unslothai/unsloth) 的指南，该项目显著提升了 LLM 的性能和内存效率，在 **Llama 3.2**、**Mistral** 等模型上的微调速度提升了 **2-5 倍**，且**内存占用减少了 80%**。
  
  - 这激发了社区对高性能 LLM 项目中实际实现的兴趣。
- **Triton Kernel 见解与优化**：讨论了关于 Triton kernels 的性能问题，一位用户指出，与 **PyTorch** 相比，单个 Kernel 操作降低了速度，建议使用多个 Kernel 以提高效率。
  
  - 此外，还提出了关于 BF16 操作未能提升速度的挑战，以及 Triton 中 Nightly 版本的持续性问题。
- **H100 展现出令人印象深刻的速度提升**：一位用户报告称，通过使用 `reduce-overhead` 等配置，[H100](https://h100.url) 达到了 **255 tokens/sec**，通过手动调整进一步增加到 **300 tokens/sec**。
  
  - 这些技术为优化 LLM 应用中的 GPU 利用率提供了新的框架。
- **ThunderKittens 0.000000002 发布并带来增强**：**ThunderKittens 0.000000002** 已发布，其特点是重大升级，包括 **6-14 倍更快的线性 Attention** 以及**比 FA3 更快的 Attention 反向传播**。
  
  - 还重点介绍了一篇关于 Kernel 性能瓶颈的论文，质疑了自定义 Kernel 与理论收益相比在现实世界中的功效。
- **Cracked Engineers 招聘平台受到关注**：[Cracked Engineers](https://www.crackedengineers.com) 启动，旨在连接人才与 AI/技术初创公司，在发布前 MRR 已接近 **$1000**。
  
  - 该平台提供 AI 辅助的职位发布流程和技术职位通讯，邀请社区反馈以持续改进。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Token 处理速度 GPU 占优**：成员指出，**Token 处理速度**在 **GPU 上约为 62 tok/sec**，而在 **CPU 上约为 7.5 tok/sec**。
  
  - Fewill 在讨论这些速度时表达了热情，说道：*“太棒了！”*。
- **寻找本地 LLM 推荐**：一位成员正在寻找类似于 **Phind** 或 **ChatGPT** 的**本地运行 LLM**，重点关注 Python 和 Houdini SideFX。
  
  - Fabguy 建议研究 **HumanEval**，但指出 Houdini 的小众性质可能会影响回答的相关性。
- **NGINX 代理设置困扰**：一位用户在配置 LM Studio 服务器的 **NGINX 代理主机**时遇到困难，尽管已激活 *serve on local network*。
  
  - 其他用户分享了排查步骤，强调了准确配置设置的重要性。
- **PCIe 带宽辩论升温**：关于 **PCIe 带宽**是否影响推理性能引发了辩论，有建议认为 PCIe Gen 3 就足够了，因为大部分处理发生在 GPU 上。
  
  - 然而，用户强调带宽对于跨多个 GPU 训练模型至关重要，在这种情况下需要高带宽。
- **多 GPU 配置查询**：关于使用多个 **3090s** 运行大模型的咨询揭示了对超过单个 GPU 内存时性能损失的担忧。
  
  - 结论是，如果 GPU 是相同的，性能将保持稳定，并且 Offloading 任务可以提高整体处理效率。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 用户报告运行缓慢**：成员们报告了 Aider 的**运行缓慢**问题，特别是使用 litellm 的 **get_model_cost_map** 函数时。可以通过设置 `export LITELLM_LOCAL_MODEL_COST_MAP='True'` 来优化。
  
  - 一位用户指出，Aider 在大多数情况下通常会尝试掩盖 litellm 的**缓慢**。
- **网页爬虫建议**：一位用户建议使用 [FireCrawl](https://firecrawl.dev) 进行 **web scraping**，理由是其高效的提取能力和自托管选项。
  
  - 讨论表明，如果配置得当，FireCrawl 可以克服社交媒体爬虫面临的挑战。
- **使用 Aider 管理 Git 仓库**：几位用户讨论了保持 Git 仓库整洁的策略，建议采用手动提交而非 Aider 的自动提交功能。
  
  - 一位参与者分享了使用 `git switch` 并合并压缩提交（squashed commits）的流程，以保持仓库井然有序。
- **GitHub Copilot 竞争 Aider**：一位成员强调，Copilot 与 **OpenAI**、**Gemini** 和 **Anthropic 模型**的集成可能会影响其与 Aider 的竞争。
  
  - 另一位用户对 Copilot 表示不满，并提到已转向 Supermaven，这表明编程助手的用户偏好正在发生变化。
- **有效的 Prompt Engineering 见解**：关于构建有效提示词的讨论强调了它们对于生成准确 AI 输出的必要性，重点在于提供充足的上下文。
  
  - 针对 AI 在调试过程中产生误导性结果的问题，引发了关于重构提示词以提高清晰度的讨论。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **对 AI 研究资助的兴趣增长**：成员们询问了**申请 AI 研究资助**的经验，突显了对创新项目资金支持的日益关注。
  
  - 这反映了一个大趋势，即财务支持对于新的 AI 计划变得至关重要。
- **对演进算法的着迷**：讨论集中在**算法的演进**上，注意到 AI 模型中出现的不同人格特征。
  
  - *他们一直在突破界限*，成员们渴望了解这些模型如何管理各种输入。
- **将 AI 拟人化的风险**：对话显示了对 **LLMs** 产生类人输出可能导致对意图产生误导性假设的担忧。
  
  - 成员们敦促将 AI 视为工具，而不是推断其具有人类情感，这一点非常重要。
- **呼吁加强 AI 伦理指南**：成员们强调在 AI 领域进行仔细的**伦理考量**以减轻未来风险的必要性。
  
  - 开发智能系统的人员有责任为其应用制定更清晰的指南。
- **关于 GPT 拼写错误问题**：成员们报告在使用 ChatGPT 时持续出现**拼写错误**和逻辑不连贯的问题，对输出质量表示担忧。
  
  - 社区表达了困惑，询问其他人是否也遇到了类似问题。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **算法交易：经验教训**：一位拥有 4 年算法交易经验的成员分享了对市场交互复杂性的见解，指出**严谨的流程有助于抵御负面影响**。
  
  - *了解哪些方法行不通需要大量的模拟交易和研究。*
- **理解 AI 情感分析中的媒体偏见**：成员们一致认为所有媒体都存在偏见，识别谁从这种偏见中获益对于准确评估至关重要。
  
  - *一位成员提到，他们构建了一个模型，在假设所有媒体都存在偏见的前提下开始调查。*
- **AI 输出乱码导致困惑**：成员们报告在 AI 模型输出中看到奇怪的乱码文本，对其可用性表示担忧。
  
  - *建议降低 temperature 和 top-p 参数作为潜在的修复方案，并建议进行实验。*
- **关于响应长度的见解**：响应通常在达到基于结构化提示词的自然结尾时停止，典型长度为 **3,000-4,000** 个字符。
  
  - *一位成员强调，个性化会显著影响输出长度。*
- **使用 LLMs 生成医疗笔记**：一个演示展示了使用 LLMs 生成合成医疗笔记，允许用户以极少的输入创建详细的笔记。
  
  - *查看* [*此处演示*](https://demo.talcapi.com/demo/meddoc?id=72a7fe2b-a2c9-4542-9063-af8093331ba9) *以了解该工具的功能。*

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Inflection 的服务恢复在线**：最近的**计费问题**已得到解决，Inflection 现已恢复运行，提升了所有用户的生产力。更多详情请参阅 [Inflection 3 PI](https://openrouter.ai/inflection/inflection-3-pi) 和 [Inflection 3 Productivity](https://openrouter.ai/inflection/inflection-3-productivity) 的链接。
  
  - 随着服务恢复，用户报告操作已回归正常，增强了此前受影响任务的处理能力。
- **招募 macOS 聊天应用的 Alpha 测试人员**：一位开发者正在为其新款适用于 macOS 的**灵活聊天应用**积极寻找 **Alpha 测试人员**，并分享了展示其功能的[截图](https://imgur.com/a/HI5Py3A)。
  
  - 鼓励有兴趣的参与者私信（DM）开发者，以加入这一重要的测试阶段。
- **OpenRouter API 出现不稳定性**：用户报告了影响 OpenRouter API 的 **524 错误**，导致严重的请求延迟，并引发了对其是否已准备好供公众使用的担忧。
  
  - 由于持续的不稳定性阻碍了多个请求的执行，一些用户正考虑更换供应商。
- **关于 API key 安全风险的辩论**：人们对 API key 可能被爬取的担忧日益增加，讨论强调了使用 **Claude 3.5 Sonnet** 等模型的未经授权代理所带来的风险。
  
  - 用户强调了保护密钥的重要性，并担心尽管采取了现有预防措施，漏洞仍可能导致意外泄露。
- **集成访问权限需求量大**：多位成员表达了对访问**集成（integrations）**功能的请求，强调了诸如“我想获得访问权限”之类的礼貌诉求。
  
  - 其中一个值得注意的请求来自一位**学生研究员**，表明了学术界对探索集成功能的兴趣。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Moondream 获得 450 万美元融资**：[Moondream](https://venturebeat.com/ai/moondream-raises-4-5m-to-prove-that-smaller-ai-models-can-still-pack-a-punch/) 筹集了 450 万美元，旨在证明更小的 AI 模型依然具有强大实力，其网络爬虫已活跃数月。
  
  - 讨论中出现了对潜在局限性以及在 AI 行业采用更小模型的深远影响的担忧。
- **Meta 开发自有的 AI 搜索引擎**：据报道，Meta 正在开发一款 [AI 驱动的搜索引擎](https://www.theverge.com/2024/10/28/24282017/meta-ai-powered-search-engine-report)，以减少对 Google 和 Microsoft 的依赖。
  
  - 活跃的网络爬虫暗示 Meta 内部正在发生重大转变，以增强其搜索能力。
- **GitHub Copilot 新增 Gemini 和 Claude 模型**：GitHub 引入了 [Gemini 模型](https://cloud.google.com/blog/products/ai-machine-learning/gemini-models-on-github-copilot) 和 Claude，通过新功能增强其 Copilot 能力。
  
  - 这代表了 Microsoft 与 Google 之间意想不到的合作，因为他们开始拥抱面向开发者的多模型（multi-model）方法。
- **对现有 Vector Databases 的批评**：成员们批评当前的向量数据库缺乏合理的抽象，并支持使用 [pgai Vectorizer](https://github.com/timescale/pgai) 进行更高效的 Embedding 管理。
  
  - 该工具承诺简化 Embedding 的同步和维护，这对于提升 AI 模型性能至关重要。
- **OpenAI 推出聊天记录搜索功能**：OpenAI 为 ChatGPT 推出了一项新功能，允许用户搜索其聊天历史记录，提高了访问过去讨论的便利性。
  
  - 成员们庆祝了这一期待已久的更新带来的便利，强调了对话连续性的改善。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular 频道重点明确**：在关于频道重点的咨询中，明确了 <#1098713601386233997> 频道严格用于 **Modular 产品**，而通用软件讨论则引导至 <#1104620458168553563>。
  
  - 这一区分强调了维持对 Modular 产品聚焦讨论的目标。
- **Mojo 提议内存安全引用革命**：一名成员发布了一份关于重新构想 Mojo 中 **内存安全引用 (memory-safe references)** 的 [重大提案](https://gist.github.com/nmsmith/cdaa94aa74e8e0611221e65db8e41f7b)，旨在建立一个更安全且更简单的引用模型。
  
  - 目前正在征求社区反馈，以确保设计同时支持 **优化灵活性** 和内存安全性。
- **FlatBuffers 与 ProtoBuf 对比分析**：团队权衡了 **FlatBuffers** 和 **ProtoBuf** 的优势，指出 FlatBuffers 的零解析效率与 ProtoBuf 对位打包 (bit packing) 的侧重。
  
  - 由于他们计划在 Serving 中使用 ProtoBuf，因此分享了一个 [Swift ProtoBuf 支持示例](https://github.com/apple/swift-protobuf) 作为开发参考。
- **Mojo 中交换引用引发关注**：成员们讨论了在 Mojo 中实现 **交换引用 (swapping references)** 的潜在陷阱，并与 Rust 的可变引用管理进行了对比。
  
  - 人们对这可能带来的额外复杂性表示担忧，特别是关于 **性能影响** 方面。
- **优化重点转向 noalias 讨论**：讨论强调了在 Mojo 中使用 `noalias` 实现高效性能的重要性，许多人主张将其作为默认方法。
  
  - 支持唯一引用的设计被认为是必不可少的，因为在此处的疏忽可能会导致严重的性能问题。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Hugging Face CEO 引发热议**：Hugging Face 的联合创始人兼 CEO Clem 计划进行一场激动人心的演讲，这在社区内引起了期待。
  
  - 演讲详情尚未披露，成员们正热切期待更多信息。
- **Hellaswag 训练性能超出预期**：在使用 **8xH100** 硬件的情况下，以不到 **$200** 的成本在 **7.3 小时** 内实现了 **GPT-2 (1.5B)** 级别的 Hellaswag 性能，创造了新纪录。
  
  - 这代表了效率的重大飞跃，此前的基准为 **24 个 8xH100-小时**。
- **GPT-NeoX 在 Colab 上运行确认**：已确认 **GPT-NeoX** 可以在 Colab 上运行，并提供了一个 [Colab notebook](https://github.com/markNZed/GPT-NeoX-Colab/blob/main/notebooks/shakespeare_training.ipynb) 参考链接。
  
  - 所使用的模型非常紧凑，其 **5M 参数** 展示了实际应用的潜力。
- **首个 Sparse Autoencoder 指南发布**：一名成员发布了关于利用 **预制 Sparse Autoencoder** 的 [分步指南](https://beta.ai-plans.com/guide/g7yjq98bhuyhkza)，标志着 Mechanistic Interpretability 领域的一项新举措。
  
  - 该指南为旨在加深对解释性技术理解的系列内容奠定了基础。
- **自定义证书支持问题得到确认**：一名成员注意到缺乏对 **自定义证书** 的支持，但分享了一个可能有助于缓解这一限制的 [变通方法 (workaround)](https://github.com/EleutherAI/lm-evaluation-harness/issues/2436)。
  
  - 讨论突显了社区在分享解决这些技术挑战的方案方面所做的努力。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI CFO 宣布 AI 已成为主流**：在一段 [YouTube 视频](https://youtu.be/eCqFgVqWbEs)中，OpenAI CFO Sarah Friar 强调 **AI 不再是实验性的**，因为银行和金融科技公司每天都在使用它。
  
  - 这一重大转变也为各行各业的广泛落地提供了更多机会。
- **SearchGPT 扩展程序发布**：预计 OpenAI 将推广其新的 Chrome 扩展程序，允许用户在发布时将 **SearchGPT** 设置为默认搜索引擎。
  
  - 用户可以直接通过浏览器地址栏快速发起搜索，并根据需要使用重定向到 Google 的命令。
- **ROCKET-1 简介**：**ROCKET-1** 旨在通过利用视觉-时间上下文提示来增强 Minecraft 中的创意任务，由 [Team CraftJarvis](https://craftjarvis.github.io/) 展示。
  
  - 这一进展突显了 Vision-Language Models 在开放世界应用中不断进化的能力。
- **Anthropic 的招聘势头**：Anthropic 因其强劲的招聘实践而受到关注，并宣布有新成员加入其团队。
  
  - 他们最近的举措反映了公司在 AI 领域的蓬勃发展和雄心壮志。
- **Claude 与 GitHub Copilot 的集成**：Claude 3.5 Sonnet 现已面向在 Visual Studio Code 中使用 GitHub Copilot 的开发者开放，本周开始推广。
  
  - 这种集成预计将通过在流行的开发工具中直接提供先进的 AI 支持来增强编码体验。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter 的完整功能需要视觉模型**：为了让 Open Interpreter 正常发挥视觉能力，通常需要一个 **Multi-modal Model**，除非使用 **Moondream** 处理基础任务。
  
  - 用户反映在使用 **Llava** 等本地模型时，难以复制 **Sonnet** 或 **GPT-4o** 的功能。
- **本地模型执行操作的挑战**：成员在使用 **Llava** 等本地模型执行类似于云端模型的操作（如截屏）时遇到问题。
  
  - 呼吁改进设置指南，以便更好地与 **Computer API** 集成。
- **OpenAI 向免费用户开放 Advanced Voice**：OpenAI 宣布 **Advanced Voice** 现已向欧盟、瑞士、冰岛、挪威和列支敦士登的**免费用户**开放。
  
  - 这一进展显著提高了这些地区用户的可访问性。
- **Apple 为破解其 AI 服务器提供 100 万美元奖金**：**Apple** 准备为任何成功入侵其 **AI 服务器**的人支付高达 **100 万美元**的奖金。
  
  - 这一举措引发了对 **Cybersecurity** 的关注，并引导人们审视 Apple 的安全措施。
- **ChatGPT 推出聊天记录搜索**：OpenAI 透露已在 **ChatGPT Web** 端推出聊天记录搜索功能，提升了用户的可用性。
  
  - 此更新允许用户快速参考之前的对话，改善了持续交互的体验。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **不带 LoRA 的量化受到关注**：成员们讨论了基础模型是否可以在不利用 LoRA 的情况下进行像 **QLoRA** 那样的量化，并强调了在非 LoRA 环境中的配置挑战。
  
  - *“我想主要问题是我们没有办法在非 LoRA 模型构建器中配置这一点。”*
- **FSDP 的简单 CPU Offloading 测试**：讨论集中在 **FSDP** 上，目前它使用单一参数进行 CPU Offloading（包括参数、梯度和优化器状态），缺乏精细控制。
  
  - 有人提出性能方面的考虑：*“这种方法数据移动更多，但由于优化器步骤在 GPU 上，速度可能更快。”*
- **对量化 KV-Caches 的怀疑**：由于大型模型的高内存消耗，成员们对使用 NF4 张量的 **Quantized KV-Caches** 的实用性表示怀疑。
  
  - *“我不认为 Torchao 中的 Quantized KV-Cache 目前有那么有用或强大，”* 这表明需要进一步探索。
- **量化非训练权重引起兴趣**：对话强调，在 **PPO** 期间量化冻结权重有助于减少内存使用，特别是对于非训练模型组件。
  
  - *“是的，我想做类似的事情，在 PPO 期间量化非训练模型，”* 表现出对内存效率策略的兴趣。
- **8-bit 以下量化的准确性风险**：对于将激活值（特别是 KV Caches）量化到 8-bit 以下时的准确性表示担忧。
  
  - *“将激活值量化到 8-bit 以下会出现相当严重的准确性问题，”* 强调了在激进量化方法上需保持谨慎。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **PAPILLON 解决 AI 隐私担忧**：研究人员开发了 [PAPILLON](https://arxiv.org/abs/2410.17127)，在 AI 应用中实现了 **85.5% 的质量**，而 **隐私泄露仅为 7.5%**。
  
  - 该系统有效地允许集成 **本地和云端 LLM**，解决了现代 AI 中的重大隐私挑战。
- **PUPA 基准测试揭示隐私问题**：团队推出了 **PUPA**，这是一个评估包含个人身份信息 (**PII**) 的用户与 LLM 交互的基准。
  
  - 他们的发现为一种名为 **Privacy-Conscious Delegation** 的新方法提供了依据，该方法融合了 API 驱动和本地模型的方法。
- **DSPy 简化 AI 编程**：一份关于 [DSPy 的 ELI5 解释](https://x.com/lateinteraction/status/1851324349216927856)将其描述为一种编程语言，允许通过带有 DSPy signatures 的普通 Python 开发 AI 系统。
  
  - DSPy 提供了用于处理 prompting 策略的 Modules，以及专注于提高输出质量的 Optimizers。
- **MIPROv2 Optimizer 提升质量**：讨论显示，如果有效利用，MIPROv2 optimizer 可以使输出质量提高 **41%**，泄露减少 **68%**。
  
  - 用户注意到它能够根据各种属性对训练数据进行采样并生成指令，从而优化整体性能。
- **MIPROv2 错误修复解决了使用问题**：有报告称 MIPROv2 在与 GPT-4o Mini 配合使用时出现错误，与其在 GPT-4 上的成功运行形成对比。
  
  - 调整 demo 参数有助于解决困惑，并提高了中等配置下的性能。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **NVIDIA 关注 RAG 的需求**：NVIDIA 最新的博客深入探讨了 **检索增强生成 (RAG)**，揭示了用户渴望额外的功能，包括 **文档翻译** 和 **代码编写**。
  
  - 即使是那些专注于内部数据的人也对 **web search** 功能表现出兴趣，该功能通过 [Perplexity’s search API](https://docs.perplexity.ai/home) 实现。
- **Chroma 的检索算法引起关注**：围绕 **Chroma** 的 vector store 检索行为展开了讨论，特别是在使用 `index = GPTVectorStoreIndex.from_vector_store(vector_store=vector_store)` 时。
  
  - 成员们强调 **Chroma** 的算法是近似的，即使对于相似的索引块，也会影响结果的可变性。
- **揭秘网页抓取技巧**：分享了一个名为“[这就是我如何通过 LLM 抓取 99% 网站](https://youtu.be/7kbQnLN2y_I)”的实用 YouTube 视频，展示了 2024 年先进的网页抓取能力。
  
  - 该视频提倡使用 **AgentQL** 免费抓取网站，展示了 LLM 的实际应用。
- **区块链工程师寻求项目合作**：一位自 2017 年起从业的区块链工程师寻求项目机会，自荐在 **defi**、**NFT games** 以及 **Solidity** 和 **RUST** 等语言方面的专业知识。
  
  - 他们的背景包括参与涉及 **Dex**、**DAO** 以及 **NFT** 铸造和质押的各种项目。
- **使用 MLflow 构建高级 RAG 系统**：一份指南概述了如何利用 MLflow 和 LlamaIndex 创建 **高级 RAG 系统**，允许结合 vector 和基于关键词的搜索。
  
  - 这种方法针对 **event-driven orchestration** 以增强工作流管理，如 [GitHub](https://github.com) 上的一个示例所示。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **LLM Agents Hackathon 报名人数激增**：短短几天内，已有超过 **1000 多名创新者**报名参加 [LLM Agents Hackathon](https://rdi.berkeley.edu/llm-agents-hackathon/)，反映出浓厚的兴趣。如果你还没加入，请立即完成[参与者报名](https://docs.google.com/forms/d/e/1FAIpQLSevYR6VaYK5FkilTKwwlsnzsn8yI_rRLLqDZj0NH7ZL_sCs_g/viewform)！
  
  - *现在加入还不晚！*
- **第 8 次讲座定于 PST 时间下午 3:00**：**第 8 次讲座**将于今天 **PST 时间下午 3:00** 举行，[此处提供直播地址](https://www.youtube.com/live/wm9-7VBpdEo)。本次会议重点讨论将复杂推理与 Large Language Models 集成，承诺提供宝贵的见解。
  
  - *敬请收看！*
- **成立学习小组**：一名成员提议成立一个**学习小组**进行课程讨论，建议通过虚拟会议吸引那些较晚加入的人。随后很快有人表示感兴趣，几名成员确认他们想要参加。
  
  - *听起来很酷！*
- **请求直播字幕**：一名成员请求在直播视频中开启 **Subtitles**（字幕），并得到确认所有讲座随后都会进行编辑并提供字幕。这确保了可访问性，提升了观众体验。
  
  - *我们正在努力！*
- **开发基于 React 的自动化 Agent**：一名成员询问如何创建一个**基于 React 的 Agent**，使用 [pyauto gui](https://pyautogui.readthedocs.io/en/latest/) 根据当前状态评估来自动执行任务。建议直接提问而不是泛泛而谈。
  
  - *直接问更简单！*

 

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Latent Diffusion Model 训练中的粉色像素块**：在训练**类别条件 Latent Diffusion Model** 时，一名成员报告在 VAE 解码过程中遇到**粉色像素块**，随着训练次数增加，这些像素块出现的频率会降低。
  
  - 他们正在考虑在 **DDIM p_sample** 中采用更激进的裁剪（目前为 **99.95%**）是否能解决这些色块问题。
- **对 Parameters 与 Tokens 的误解**：一名成员误以为 **100B** 指的是参数（Parameters）而非 Token，这导致了混淆，随后由另一名成员澄清。
  
  - 此外，他们指出链接的模型实际上只有 **8B parameters**，并得到了同行的验证。
- **协作探索 IJEPA 架构**：一名成员表示有兴趣合作开发一种创新架构，将 **IJEPA** 与**无向量量化的自回归图像生成**相结合。
  
  - 他们对共同探索这一独特架构的热情预示着该领域潜在的进展。

 

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **George Hotz 经历了“负行数日”**：George Hotz 表示自己经历了一个 **negative line day**（代码行数负增长日），引发了社区的幽默反应。
  
  - 这种轻松的交流反映了成员们在应对编码挑战时相互支持的氛围。
- **CI 测试变快了**：Chenyuy 报告 **CI 测试缩短了 2 分钟**，表明性能优化取得了进展。
  
  - 测试流程的改进展示了在提升 tinygrad 项目效率方面的共同努力。
- **Uops 可读性挑战浮现**：关于 **Uops** 可读性的担忧浮出水面，一些单行代码（one-liners）难以理解。
  
  - 有人建议创建一个文档页面，以潜在地提高所有用户的代码清晰度。
- **强调文档维护问题**：Chenyuy 强调了关于文档的**维护担忧**，文档往往很快就会过时。
  
  - 他指出，不准确的文档可能比没有文档更阻碍进度，这反映了 tinygrad 快速变化的节奏。
- **关于过早优化的辩论**：George Hotz 提议移除某些代码元素，以避免 **premature optimization**（过早优化）的陷阱。
  
  - 这场讨论强调了正在进行的深思熟虑的测试，旨在仔细平衡代码效率与潜在的复杂性。

 

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **RAGAS 增强 LLM 评估**：一位成员建议使用 [RAGAS](https://github.com/explodinggradients/ragas) 来改进 **LLM 应用评估**，并展示了其功能和方法论。
  
  - 该工具旨在为开发者提供更精细的方法，以有效地评估语言模型。
- **CSV 文件寻求集成**：关于将 **CSV 文件** 作为数据源与 **LLAMA3** 等开源模型集成的讨论引起了关注，并指出目前现有示例存在空白。
  
  - 该咨询特别提到了将 CSVChain 和 PandasAgent 与非 OpenAI 模型结合使用，以实现更好的数据处理。
- **LangChain-Python 版本查询**：寻求关于哪个版本的 **Python** 与 **LangChain 0.3 版本** 兼容的澄清，反映了社区对环境配置指导的需求。
  
  - 正确的环境配置对于开发者高效使用 LangChain 至关重要。
- **LangChain-JS 课程发布**：**好消息！** Udemy 上发布了一门针对初学者的全新 [LangChain-JS 课程](https://www.udemy.com/course/genai-langchain-for-javascript-developers/?couponCode=AMIT100)。
  
  - 课程内容涵盖从基础知识到构建完整的 RAG 应用，前 **100 名学生**可以免费入学。
- **网页抓取大师课**：一位成员推荐了一个名为“这就是我如何通过 LLM 抓取 99% 的网站”的 [YouTube 视频](https://youtu.be/7kbQnLN2y_I)，教授使用 LLM 进行实用的网页抓取。
  
  - 该视频强调使用 [AgentQL](https://www.agentql.com/) 免费抓取网站，展示了创新技术。

 

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **澄清排行榜上的 'Multiple'**：排行榜上的 'Multiple' 表示在单轮对话中从多个选项中**选择正确函数的能力**，如[此 GitHub 示例](https://github.com/ShishirPatil/gorilla/blob/2101b11f6d03d9f323715d7d2012a955d7f4114e/berkeley-function-call-leaderboard/data/BFCL_v3_exec_multiple.json#L42C1-L42C2438)所示。在这种情况下，多步（multi-step）的评估仍然存在歧义。
  
  - 这种困惑值得注意，特别是关于多步执行与多轮（multi-turn）场景的区别，这引发了用户间的各种讨论。
- **多步与多轮评估方法**：一位成员澄清说，'multiple' 与函数有关，而**多步评估**属于 'multi_turn' 类别，目前没有使用单一的多步评估。理解这些区别对于准确解读至关重要。
  
  - 多步和多轮评估之间的重叠可能会让用户感到困惑，因为这两个概念在排行榜设置的评估中共享相同的类别。

 

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Cracked Engineers 招聘平台上线！**：一位成员分享了一个名为 **Cracked Engineers** 的全新[技术职位招聘平台](https://www.crackedengineers.com/)，旨在成为顶级 AI/技术初创公司的首选。
  
  - 在正式发布前，该平台的预计 **MRR** 已达到 **1000 美元**，目前已吸引了 **Weaviate**、**UnslothAI** 和 **JuliusAI** 等顶尖公司。
- **推出极具洞察力的每周技术职位通讯**：该平台即将发布**每周技术职位通讯**，根据用户偏好精选职位。
  
  - 用户可以通过仪表板订阅感兴趣的标签，例如 **CUDA**、**MLOps** 或 **Software Engineering**。
- **AI 初创公司的诱人工作机会**：**Unsloth AI**、**Julius AI** 和 **Jimini AI** 正在积极招聘优秀职位，如果不是创始人，他们也会考虑这些职位。
  
  - 对于任何希望从事前沿 AI 技术工作的人来说，这些职位都被描述为**绝佳的机会**。

 

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **成员寻求 SymNoise 代码实现**：一名成员正在寻找 **SymNoise** 微调技术的代码实现，该技术将 **symmetric noise** 集成到 embedding 中。由于 **batch size** 要求方面的问题，他们在实现过程中遇到了困难。
  
  - 这一咨询显示出社区对高级微调方法的兴趣日益浓厚，尽管目前尚未提供具体的解决方案。
- **SymNoise 提升 LLaMA-2-7B 性能**：**SymNoise** 方法将 **LLaMA-2-7B** 在 AlpacaEval 上的表现从 **29.79%** 提升到了令人印象深刻的 **69.04%**，超越了 **NEFTune**。正如论文摘要所述，这比 NEFTune 的 **64.69%** 分数显著提高了 **6.7%**。
  
  - 结果突显了 **SymNoise** 在微调语言模型方面的潜力，为性能设定了新的基准。
- **SymNoise 在多模型中表现优于 NEFTune**：测试显示，在各种模型和基准数据集上，**SymNoise** 的结果始终优于 **NEFTune**。这引发了关于该领域需要进一步研究的讨论。
  
  - 社区成员强调了继续探索和验证这些微调方法论的重要性。
- **征集 SymNoise 研究资源**：在咨询中，一名成员链接到了详细介绍 **SymNoise** 方法的 **arXiv** 论文，强调了其在该领域的相关性。然而，目前还没有共享额外的代码资源或实现来帮助解决实现上的挑战。
  
  - 这表明在基于最新研究成果开发实际应用方面，需要更广泛的协作努力。

---

**Alignment Lab AI Discord** 没有新消息。如果该服务器（guild）沉寂时间过长，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。

---

# PART 2: 频道详细摘要与链接

{% if medium == 'web' %}

### **HuggingFace ▷ #**[**announcements**](https://discord.com/channels/879548962464493619/897387888663232554/1300604946403823708) (1 条消息):

> - `Clem 介绍`
> - `直播工作坊推广`

- **Clem 向社区介绍自己**：Hugging Face 的联合创始人兼 CEO Clem 表示，很高兴能通过 Discord 与社区成员进行更紧密的互动。
  
  - 他强调了与用户交流的渴望，并表示：“我迫不及待想与大家互动”。
- **周三直播工作坊推广**：Clem 正在征集关于如何推广他原定于周三举行的直播工作坊的建议，该工作坊通过此 [链接](https://streamyard.com/watch/JS2jHsUP3NDM) 分享。
  
  - 他鼓励用户分享任何可以发布该活动信息的 Discord 频道或群组，以提高参与度。

---

### **HuggingFace ▷ #**[**general**](https://discord.com/channels/879548962464493619/879548962464493622/1300536845301452911) (870 条消息🔥🔥🔥):

> - `TensorFlow 带来的挫败感`
> - `LLM 训练方法`
> - `AI/ML 学习路径`
> - `使用 Hugging Face API`
> - `防止模型过拟合`

- **对 TensorFlow 的挫败感**：许多成员表达了对 TensorFlow 的不满，理由包括禁用 Windows 上的 GPU 支持以及必须查阅复杂的文档。他们发现转向 PyTorch 通常能获得更快的结果且麻烦更少。
  
  - 用户分享了他们吐槽 TensorFlow bug 和支持不力的经历，表明了对该框架普遍存在的不满情绪。
- **训练 LLM 与过拟合解决方案**：Noaroggendorff 讨论了在训练 Llama 模型方面持续进行的努力，并尝试了多次优化设置。减少过拟合的最佳建议是在训练期间仅使用一个 epoch。
  
  - 这引发了用户之间关于应对大型语言模型有效训练挑战的各种策略的对话。
- **AI 和 ML 的学习路径**：几位用户就培养 ML 和 AI 技能分享了见解，强调基础理解至关重要。他们讨论了 AI 的广阔前景，以及初学者应如何专注于特定任务以建立专业知识。

- 对话强调了根据个人兴趣和快速发展的 AI 领域性质来调整学习的重要性。
- **使用 Hugging Face API 获取 Token 概率**：Asahikokura 询问了如何通过 Hugging Face API 获取 token 概率，并得知 Inference Client 可以简化这一过程。用户提供了如何利用 API 获取 log probabilities 的示例，并引导他查看有关 rate limits 的相关文档。
  
  - 讨论中提到，该 API 允许在不本地下载模型的情况下进行访问，使初学者更容易尝试语言模型。
- **MLOps 精通与求职**：Rumigazzi 询问了如何精通 MLOps 并在该领域获得工作。对话涉及了 AI 和 MLOps 技能日益增长的需求，并建议参与社区项目并建立扎实的学习路径。

**提及的链接**：

- [Code of Conduct – Hugging Face](https://huggingface.co/code-of-conduct)：未找到描述
- [How AI Agents Can Be Exploited Through Indirect Prompt Injection · AI Security Blogs](https://www.stealthnet.ai/post/how-ai-agents-can-be-exploited-through-indirect-prompt-injection)：AI 安全是下一波浪潮。学习如何攻击和防御 AI 与 ML 模型。
- [Format selector for 2410.02694](https://arxiv.org/format/2410.02694)：未找到描述
- [LaTeX.js](https://latex.js.org/)：未找到描述
- [What Matters in Transformers? Not All Attention is Needed](https://arxiv.org/abs/2406.15786)：虽然扩展基于 Transformer 的大型语言模型 (LLMs) 在各种任务中表现出了良好的性能，但它也引入了冗余架构，给效率带来了挑战...
- [Context.ai](https://context.ai/compare/gemini-pro/gemini-ultra)：比较 Gemini Pro 和 Gemini Ultra 之间的价格、基准测试和模型属性。
- [rombodawg/Rombos-LLM-V2.5-Qwen-72b · Hugging Face](https://huggingface.co/rombodawg/Rombos-LLM-V2.5-Qwen-72b)：未找到描述
- [Can You Run It? LLM version - a Hugging Face Space by Vokturz](https://huggingface.co/spaces/Vokturz/can-it-run-llm)：未找到描述
- [Rate Limits](https://huggingface.co/docs/api-inference/en/rate-limits)：未找到描述
- [What is NotebookLM - Help](https://support.google.com/notebooklm/answer/14273541?hl=en)：未找到描述
- [Gokacik O Yerim GIF - Gokacik Gok O yerim - Discover & Share GIFs](https://tenor.com/view/gokacik-gok-o-yerim-yerim-yer%C4%B1m-gif-8234693358169729819)：点击查看 GIF
- [Peter Griffin Family Guy GIF - Peter Griffin Family Guy Peter - Discover & Share GIFs](https://tenor.com/view/peter-griffin-family-guy-peter-gif-26549552)：点击查看 GIF
- [meta-llama/Llama-3.1-8B · Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B)：未找到描述
- [Local AI with Docker's Testcontainers](https://huggingface.co/blog/Tonic/localai-testcontainers)：未找到描述
- [XSS on every Gradio server via upload of HTML files, JS files, or SVG files](https://github.com/gradio-app/gradio/security/advisories/GHSA-gvv6-33j7-884g)：### 影响 \*\*这是哪种类型的漏洞？谁会受到影响？\*\*

该漏洞涉及在任何允许文件上传的 Gradio 服务器上的**跨站脚本攻击 (XSS)**。经过身份验证的用户...
- [Alone Glitch GIF - Alone Glitch Film - Discover & Share GIFs](https://tenor.com/view/alone-glitch-film-eloresnorwood-heartbreak-gif-15491348): 点击查看 GIF
- [Alpaca Llama GIF - Alpaca Llama Animation - Discover & Share GIFs](https://tenor.com/view/alpaca-llama-animation-art-lama-gif-24994921): 点击查看 GIF
- [AI Builder Club](https://link.agent.rocks/6dUcFwA): 学习使用 Cursor AI 编写代码并构建应用，以及最新的应用课程
- [Pdf Component Example](https://www.gradio.app/guides/pdf-component-example): Gradio 分步教程
- [Embed No GIF - Embed No No Embed - Discover & Share GIFs](https://tenor.com/view/embed-no-no-embed-megamind-megamind-meme-gif-25261934): 点击查看 GIF
- [This is how I scrape 99% websites via LLM](https://youtu.be/7kbQnLN2y_I): 2024 年如何使用 LLM 进行网页抓取。使用 AgentQL 免费抓取网站：[https://www.agentql.com/?utm_source=YouTube&utm_medium=Creator&utm_id=AIJason](https://www.agentql.com/?utm_source=YouTube&utm_medium=Creator&utm_id=AIJason)_...
- [Curious GIF - Curious - Discover & Share GIFs](https://tenor.com/view/curious-gif-9412615): 点击查看 GIF
- [Insomnia Sleep GIF - Insomnia Sleep Tired - Discover & Share GIFs](https://tenor.com/view/insomnia-sleep-tired-bed-smart-gif-5513542): 点击查看 GIF
- [zero-gpu-explorers/README · Zero-GPU Quota etc](https://huggingface.co/spaces/zero-gpu-explorers/README/discussions/7): 未找到描述
- [marcsun13 (Marc Sun)](https://huggingface.co/marcsun13): 未找到描述
- [Tweet from The Linux Foundation (@linuxfoundation)](https://x.com/linuxfoundation/status/1851052486288613598?s=46&t=RHgECJov_mYM1AEf96kY2g): 早上好！我们正在 Open Source Summit Japan 和 AI_dev Open Source GenAI & ML Summit 2024 现场直播今早的主旨演讲！关注 #OSSummit #AIDev 获取我们的实时推文。在此观看直播...
- [GitHub - brucemiller/LaTeXML: LaTeXML: a TeX and LaTeX to XML/HTML/ePub/MathML translator.](https://github.com/brucemiller/LaTeXML): LaTeXML：一个将 TeX 和 LaTeX 转换为 XML/HTML/ePub/MathML 的转换器。 - brucemiller/LaTeXML
- [GitHub - RayFernando1337/LLM-Calc: Instantly calculate the maximum size of quantized language models that can fit in your available RAM, helping you optimize your models for inference.](https://github.com/RayFernando1337/LLM-Calc): 立即计算可用 RAM 中可容纳的量化语言模型的最大尺寸，帮助您优化模型的 Inference。 - RayFernando1337/LLM-Calc
- [HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly](https://arxiv.org/html/2410.02694v2): 未找到描述
- [19,500+ Beautiful Sad Girl Hurt Silhouette Stock Photos, Pictures & Royalty-Free Images - iStock](https://www.istockphoto.com/photos/beautiful-sad-girl-hurt-silhouette): 未找到描述
- [LaTeXML A LaTeX to XML/HTML/MathML Converter](https://math.nist.gov/~BMiller/LaTeXML/): 未找到描述
- [Inference](https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_client): 未找到描述
  
   
  

---

### **HuggingFace ▷ #**[**today-im-learning**](https://discord.com/channels/879548962464493619/898619964095860757/1300731319680241664) (4 条消息):

> - `展示酷炫项目`
> - `服务器中的新成员`
> - `社区互动`

- **展示您的酷炫项目以获得认可**：一位成员建议在 <#897390720388825149> 中展示他们的项目，以便在公告中获得推荐。
  
  - 他们强调获得认可是非常有成就感的，这暗示了社区的支持性质。
- **欢迎新成员**：新成员 **borys_nadykto** 对建议表示感谢，并表示他们刚加入服务器。
  
  - *许多成员都是新加入的*，这表明社区正在持续增长并鼓励大家参与。
- **分享愉快的经历**：成员 **tonic_1** 给予了积极回应，表示在频道内分享经历是件*美妙的事情*。
  
  - 这反映了社区对于分享和相互交流的开放态度。

 

**提到的链接**：[Pokemon Pikachu GIF - Pokemon Pikachu Clap - Discover & Share GIFs](https://tenor.com/view/pokemon-pikachu-clap-clapping-clapping-gif-gif-13465728489229726846): 点击查看 GIF

 

---

### **HuggingFace ▷ #**[**cool-finds**](https://discord.com/channels/879548962464493619/897390579145637909/1300555304546598913) (53 条消息🔥):

> - `ML 与量子计算`
> - `理解前沿研究论文`
> - `利用 AI 增强论文阅读`
> - `隐私保护中的同态加密`
> - `ML 中 Attention 机制的影响`

- **ML 与量子计算研究黑洞**：分享了一篇探索利用 **Machine Learning** 和 **Quantum Computing** 研究黑洞的论文，被认为是一篇极具挑战性的读物。
  
  - 成员们表达了兴奋之情，但也提到感到有些吃力，这表明阅读此类论文可能需要扎实的 **Quantum Mechanics** 背景。
- **阅读复杂论文**：读者们讨论了应对复杂学术论文的策略，包括阅读 **Abstracts**（摘要）和参考 **Appendices**（附录）以获得更好的理解。
  
  - 利用 AI 在阅读后创建测验或 **Flashcards**（抽认卡）被认为是一种巩固知识的有效方法。
- **用于用户隐私的同态加密**：分享了一个讨论 **Homomorphic Encryption**（同态加密）的资源，该技术是在设备端数据处理过程中增强用户隐私的关键技术。
  
  - 该文档强调在利用设备端 **Machine Learning** 功能的同时，通过在本地进行计算来最大限度地减少外部数据暴露。
- **向 Attention 机制的转变**：一位用户在了解到 **Attention 机制** 的重要性后，表达了对缺乏该机制的 **Convolutional Neural Networks (CNNs)** 失去了兴趣。
  
  - 这反映了 **Machine Learning** 讨论中的一个更广泛趋势，即相比传统架构，基于 **Attention** 的模型越来越受到青睐。
- **创建交互式论文阅读工具**：一位成员提出了创建一个 **Hugging Face Space** 的想法，用户可以在其中交互式地阅读论文，并集成 **Language Model** 进行解释。
  
  - 讨论中提到了现有的资源和代码，这些资源可能有助于实现此类工具。

**提到的链接**：

- [Combining Machine Learning and Homomorphic Encryption in the Apple Ecosystem](https://machinelearning.apple.com/research/homomorphic-encryption)：在 Apple，我们相信隐私是一项基本人权。我们保护用户隐私的工作基于一系列隐私原则，并且……
- [llama-recipes/recipes/quickstart/NotebookLlama at main · meta-llama/llama-recipes](https://github.com/meta-llama/llama-recipes/tree/main/recipes/quickstart/NotebookLlama)：用于微调 Meta Llama 的脚本，包含可组合的 **FSDP** 和 **PEFT** 方法，涵盖单节点/多节点 **GPUs**。支持默认和自定义数据集，适用于摘要生成和问答等应用。
- [GitHub · Build and ship software on a single, collaborative platform](https://github.co)：加入全球应用最广泛、由 AI 驱动的开发者平台，数百万开发者、企业和最大的开源社区在此构建推动人类进步的软件。

---

### **HuggingFace ▷ #**[**i-made-this**](https://discord.com/channels/879548962464493619/897390720388825149/1300608096934301728) (209 messages🔥🔥):

> - `Hemp Nanosheets` (大麻纳米片)
> - `Autonomous Science Workflows` (自主科学工作流)
> - `Custom WordPiece Tokenizer` (自定义 WordPiece Tokenizer)
> - `Model Usage Analytics` (模型使用分析)
> - `Research Collaboration` (研究协作)

- **大麻纳米片作为未来材料**：对话集中在**大麻衍生碳纳米片**的潜力上，据称这种材料具有成本效益，且在能源存储和材料工程等多种应用中可与石墨烯媲美。
  
  - David Mitlin 博士的研究已经证实了以每吨约 **500 美元**的价格生产这些纳米片的可行性，引发了关于其在军事和航空航天应用中影响的讨论。
- **自主科学的想法**：一位用户分享了**自主设计的科学实验**工作流，强调了训练 AI 以大麻纳米片为案例研究，有效遵循科学方法的重要性。
  
  - 讨论包括利用这些工作流在无需人工干预的情况下进行创新研究的可能性，反映了对未来科学能力的兴奋。
- **自定义 WordPiece Tokenizer 开发**：一位初学者分享了他们开发**自定义 WordPiece Tokenizer** 的历程，重新改进了 Hugging Face 的思路，以便在其特定数据上获得更好的结果。
  
  - 他们的目标是将此 Tokenizer 贡献给现有框架，并邀请社区提供反馈和建议，以进一步增强他们的项目。
- **开源模型分析**：一位用户介绍了一个用于跟踪模型使用情况的新库，旨在与**原生 Transformers 库**无缝集成，并在无需用户额外操作的情况下提供分析数据。
  
  - 该工具旨在帮助开发者了解用户交互，并根据详细的分析数据改进开源模型。
- **关于研究协作的讨论**：贡献者们讨论了分享研究成果以及在利基领域（特别是材料科学和 NLP）鼓励协作的重要性。
  
  - 建议包括联系新闻通讯（newsletters）以获得曝光，并强调了研究社区内开放对话的重要性。

**提到的链接**：

- [Cat Trascendence GIF - Cat Trascendence Meme - Discover & Share GIFs](https://tenor.com/view/cat-trascendence-meme-gif-8496882)：点击查看 GIF
- [Hemp fibres ‘better than graphene’ | Pennsylvania Hemp Industry Council](https://www.pahic.org/hemp-fibres-better-than-graphene/)：未找到描述
- [Hemp Makes Better Supercapacitor Electrodes](https://hempingtonpost.com/hemp-makes-better-supercapacitor-electrodes/)：基于大麻的超级电容器电极性能优于标准超级电容器
- [Building a Custom WordPiece Tokenizer from Scratch: Concepts, Formulas, and Token Creation](https://medium.com/@krasniuk-ai/building-a-custom-wordpiece-tokenizer-from-scratch-concepts-formulas-and-token-creation-0e955465d239)：重新实现 Hugging Face WordPiece 公式。
- [Mujikcboro Seriymujik GIF - Mujikcboro Seriymujik - Discover & Share GIFs](https://tenor.com/view/mujikcboro-seriymujik-gif-24361533)：点击查看 GIF
- [The Road To El Dorado Both GIF - The Road To El Dorado Both Both Is Good - Discover & Share GIFs](https://tenor.com/view/the-road-to-el-dorado-both-both-is-good-gif-8304204)：点击查看 GIF
- [Shrek Reaction GIF - Shrek Reaction Really - Discover & Share GIFs](https://tenor.com/view/shrek-reaction-really-gif-27425089)：点击查看 GIF
- [GitHub - Bynesoft-Ltd/byne-serve: Google Analytics for open-source models: track usage and learn how people use your models.](https://github.com/Bynesoft-Ltd/byne-serve)：开源模型的 Google Analytics：跟踪使用情况并了解人们如何使用你的模型。
- [GitHub - koushik2k3/Meme-Explainer-Bot: Generates a given meme's explanation along with context](https://github.com/koushik2k3/Meme-Explainer-Bot)：生成给定模因（meme）的解释及背景。

---

### **HuggingFace ▷ #**[**reading-group**](https://discord.com/channels/879548962464493619/1156269946427428974/1300618090245001402) (1 messages):

> - `Discord Event Details` (Discord 活动详情)

- **Discord 活动公告**：分享了一个 Discord 活动链接供成员参与：[活动详情](https://discord.com/events/879548962464493619/1300617948414611507)。
  
  - 鼓励参与者查看活动以获取更多信息和更新。
- **鼓励参与**：敦促成员通过提供的链接参与即将举行的活动。
  
  - 频道内可能会引发围绕活动议程和感兴趣话题的讨论。

---

### **HuggingFace ▷ #**[**computer-vision**](https://discord.com/channels/879548962464493619/922424143113232404/1300648347203403858) (8 条消息🔥):

> - `Swin Transformer v2`
> - `DINO model for vision`
> - `Attention masks in vision transformers`
> - `Fine-tuning molmo VLM`

- **Swin Transformer v2 的探索**：成员们讨论了将 **Swin Transformer v2** 用于类图像的 **data cubes**，分享了关于其适用性和细微差别的见解。
  
  - *一位成员指出*，他们正在使用 **data cubes** 而不是常规图像，这引发了关于针对独特输入形状修改架构的讨论。
- **DINO 模型与 Attention masks 的配合**：有建议将 **DINO** 模型作为一个选项，但它并不直接支持 **attention masks**；它依赖于 **self-supervised learning**。
  
  - 一位成员指出，由于 **Swin transformer** 的层级结构，对其进行适配以容纳 **attention masks** 可能会很复杂。
- **Attention Masks 适配讨论**：成员们辩论了如何有效地修改 **Swin v2** 模型中的 **attention heads** 以利用 **attention masks**，并主张直接调整源代码。
  
  - *参与者注意到*，**reshaping tensors** 可能会使实现过程不必要地复杂化，因此更倾向于直接修改。
- **对 Fine-tuning molmo VLM 的兴趣**：有人询问了关于 **Fine-tuning molmo VLM** 的经验，表示有兴趣就该模型进行协作见解分享。
  
  - 目前尚未记录到关于此话题的回复。

---

### **HuggingFace ▷ #**[**NLP**](https://discord.com/channels/879548962464493619/922424173916196955/1300592439400075297) (8 条消息🔥):

> - `LangChain SQL Agent`
> - `Hugging Face NLP Course Resources`
> - `LLM Fine-Tuning`
> - `Research Papers on Modern Models`

- **LangChain SQL Agent 参考**：一位成员推荐了一个关于 **LLaMA2 SQL chat** 的 [GitHub notebook](https://github.com/langchain-ai/langchain/blob/master/cookbook/LLaMA2_sql_chat.ipynb)，作为使用 **LangChain SQL Agent** 开发者的潜在资源。
  
  - 该资源似乎有助于构建上下文感知推理应用，对于尝试实现类似功能的模型用户可能会有帮助。
- **扩展 Hugging Face 课程资源**：在完成 **Hugging Face NLP course** 后，一位成员表示有兴趣获取更多关于 **LLM Fine-tuning** 的理论资源。
  
  - 另一位成员推荐了 [Jurafsky and Martin 的 Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) 以获取更深层的理论见解。

**提到的链接**：[langchain/cookbook/LLaMA2_sql_chat.ipynb at master · langchain-ai/langchain](https://github.com/langchain-ai/langchain/blob/master/cookbook/LLaMA2_sql_chat.ipynb)：🦜🔗 构建上下文感知推理应用。通过在 **GitHub** 上创建账号为 langchain-ai/langchain 的开发做出贡献。

---

### **Unsloth AI (Daniel Han) ▷ #**[**general**](https://discord.com/channels/1179035537009545276/1179035537529643040/1300535564864327770) (170 条消息🔥🔥):

> - `Unsloth Training Tools`
> - `FP8 Fine-Tuning`
> - `Gradio UI for Model Training`
> - `Cloud GPU Connections`
> - `Job Opportunities in AI`

- **用于 Unsloth 的 Gradio UI 工具**：一位用户使用 Gradio 创建了一个应用程序，简化了使用 Unsloth 训练模型的过程，其特点是在用户友好的界面中进行设置选择和模型训练。
  
  - 该工具旨在为 nocode 用户提供一种更简单的 AI 模型训练方式，允许轻松调整设置，并提供将模型上传到 Hugging Face 的顺畅流程。
- **AI 领域的招聘机会**：Unsloth 正在通过一个名为 Cracked Engineers 的平台开展招聘活动，旨在寻找 AI 领域的技术人才。
  
  - 社区成员被鼓励浏览职位列表，并利用该平台进行职位跟踪和获取招聘更新。
- **FP8 训练讨论**：讨论了在 Unsloth 中采用 FP8 进行模型训练和微调的问题，并指出了潜在的速度提升。
  
  - 针对实现细节提出了疑问，包括 FP8 是否用于基础权重、LoRA 以及其他组件。
- **对模型训练的批评**：一位用户提到了微调较小模型的挑战以及数据集质量的重要性，认为像 Llama 3.2:70B 这样的大型模型可能会产生更好的结果。
  
  - 讨论中的批评者强调，需要理解模型训练的细微差别，而不是仅仅依赖用户友好的界面工具。
- **梯度累积问题得到解决**：一篇博客文章讨论了流行训练框架中梯度累积（gradient accumulation）的关键问题，以及它们如何影响模型性能。
  
  - Unsloth 团队正在积极解决这些问题，强调了在训练期间应用梯度累积时保持输出一致性的重要性。

**提到的链接**：

- [Cracked Engineers](https://www.crackedengineers.com/)：为您的初创公司寻找优秀的工程师。
- [Introduction to Quantization cooked in 🤗 with 💗🧑‍🍳](https://huggingface.co/blog/merve/quantization)：未找到描述
- [Cracked Engineers](https://crackedengineers.com/job/unsloth-ecb33b9f-7a36-43d3-ba5a-7af3ce4add8e)：为您的初创公司寻找优秀的工程师。
- [Aleksa Gordić 🍿🤖 (@gordic_aleksa) 的推文](https://x.com/gordic_aleksa/status/1851247076987855063)：[🚀] 非常高兴分享这个：我建立了一个名为 "Cracked Engineers" 的技术职位平台。:) 如果你想在世界上一些顶尖的 AI/技术初创公司找到工作...
- [Zach Mueller - PyTorch, Gradient Accumulation, and the dreaded lack of reproducability](https://muellerzr.github.io/blog/gradient_accumulation_part2.html)：未找到描述
- [How to Finetune Llama-3 and Export to Ollama | Unsloth Documentation](https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama)：为在 Ollama 上本地运行创建定制个人助手（如 ChatGPT）的入门指南
- [GitHub - NVIDIA/TransformerEngine](https://github.com/NVIDIA/TransformerEngine)：一个用于在 NVIDIA GPU 上加速 Transformer 模型的库，包括在 Hopper 和 Ada GPU 上使用 8 位浮点（FP8）精度，以便在训练和推理中提供更好的性能和更低的内存占用。
- [GitHub - unslothai/unsloth](https://github.com/unslothai/unsloth)：以 2-5 倍的速度和减少 80% 的内存微调 Llama 3.2, Mistral, Phi & Gemma LLM - unslothai/unsloth
- [Introducing Unsloth](https://unsloth.ai/introducing)：未找到描述

---

### **Unsloth AI (Daniel Han) ▷ #**[**off-topic**](https://discord.com/channels/1179035537009545276/1179039861576056922/1300584861232529532) (57 条消息🔥🔥):

> - `对学校教育的挫败感`
> - `教育之后的生活`
> - `经验的价值`
> - `硕士之后的计划`
> - `对 PhD 的看法`

- **对学校体制的挫败感**：一位成员表示感觉自己在学校里是 *浪费时间*，希望能做些真正有影响力的事。
  
  - 这种情绪引起了其他人的共鸣，另一位成员分享了个人经历如何塑造对教育的看法。
- **教育后的生活很复杂**：一位 17 岁辍学的成员提到，到了 40 岁时，对生活复杂性的理解会变得清晰，并表示 *我原以为会发生的一切都没有发生*。
  
  - 他们强调了与朋友联系的重要性，透露出随着时间流逝而产生的一种失落感。
- **R&D 见解：在工作中学习**：一位刚毕业的硕士承认他们对技术世界的了解有限，表示 *天哪，我意识到我还没真正了解技术世界*。
  
  - 然而，他们对于在职业生涯中能够带薪从事 R&D 感到乐观。
- **工业界对 PhD 的看法**：一位成员批评 PhD，声称 *我遇到的每个 PhD 都是白痴*，而其他人则捍卫了专业知识的价值。
  
  - 这引发了一场关于狭隘专业知识与更广泛技能组合之间对比的辩论。
- **拥抱持续学习**：几位成员讨论了即使在正式学习结束后继续教育的重要性，强调了 *你知道你不知道什么* 的观点。
  
  - 这种认知与对自己是该领域 *菜鸟 (noob)* 的自嘲相结合，展示了尽管感到经验不足但仍愿意学习的态度。

---

### **Unsloth AI (Daniel Han) ▷ #**[**help**](https://discord.com/channels/1179035537009545276/1179777624986357780/1300546103678337158) (65 条消息🔥🔥):

> - `将 HF 转换为 GGUF`
> - `持续预训练 (Continued Pretraining)`
> - `微调 Llama 3.1`
> - `Unsloth 安装问题`
> - `神经网络的记忆`

- **HF 转 GGUF 过程中的错误**：一位用户在使用 `convert_hf_to_gguf.py` 脚本时遇到错误，具体表现为 'q4_k_m' 是 `--outtype` 参数的无效选项。
  
  - 社区成员建议使用 `llama.cpp` 仓库并检查提供的选项以解决此问题。
- **关于使用 Unsloth 进行持续预训练的见解**：据报道，Unsloth 的新版本允许进行 **2 倍速** 的持续预训练，且比之前的方法 **减少 50% 的 VRAM**，并提供了 Mistral v0.3 的训练资源。
  
  - 关键见解强调了微调输入和输出 Embedding，并将其卸载 (offload) 到磁盘以节省显存。
- **微调 Llama 3.1 模型的帮助**：一位用户请求协助在 Google Colab 上训练 Llama 3.1 模型，表示在理解设置方面存在困难。
  
  - 社区回复引导他们参考现有的 Notebook，并提示他们解决模型下载过程中遇到的任何错误。
- **Unsloth 安装挑战**：提出了 Unsloth 的安装问题，特别是与使用 pip 命令安装时遇到的错误有关。
  
  - 建议包括使用 Miniconda 以获得更好的环境隔离，并纠正包名中的拼写错误。
- **神经网络记忆查询**：一位成员询问如何在神经网络中实现记忆，特别是模型是否可以在不持续添加数据集的情况下记住过去的交互。
  
  - 回复包括了记录问题和回答的策略，以便在模型训练期间提供上下文。

**提到的链接**：

- [Miniconda — Anaconda 文档](https://docs.anaconda.com/miniconda/)：未找到描述
- [使用 Unsloth 进行 LLM 持续预训练](https://unsloth.ai/blog/contpretraining)：通过使用 Unsloth 对 Llama 3、Phi-3 和 Mistral 进行持续预训练，让模型学习一种新语言。

---

### **Unsloth AI (Daniel Han) ▷ #**[**community-collaboration**](https://discord.com/channels/1179035537009545276/1180144489214509097/) (1 条消息):

mrdragonfox: 我确定你有实现这一目标所需的资金，对吧？哈哈

---

### **Unsloth AI (Daniel Han) ▷ #**[**research**](https://discord.com/channels/1179035537009545276/1257011997250424842/1300921043950633011) (1 messages):

> - `PyTorch Quantization`
> - `Optimizer CPU Offload`

- **探索 PyTorch 中的低比特优化**：PyTorch 中的 [low_bit_optim](https://github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim) 原型展示了用于改进训练和推理的原生 **Quantization** 和 **Sparsity** 技术。
  
  - 该计划是优化性能并保持模型准确性的更广泛努力的一部分，这与 AI 领域的当前趋势相契合。
- **Optimizer CPU Offload 见解**：讨论强调了 **Optimizer CPU Offload** 在增强 **low-bit** 训练框架效率方面的潜力。
  
  - 通过将操作卸载到 CPU，模型可以更好地利用现有硬件，从而实现 **更快的训练时间** 和更低的资源占用。

**提及的链接**：[ao/torchao/prototype/low_bit_optim at main · pytorch/ao](https://github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim#optimizer-cpu-offload)：用于训练和推理的 PyTorch 原生 Quantization 和 Sparsity - pytorch/ao

---

### **Stability.ai (Stable Diffusion) ▷ #**[**announcements**](https://discord.com/channels/1002292111942635562/1002292398703001601/1300871806022058147) (1 messages):

> - `Stable Diffusion 3.5 Medium`
> - `模型性能与硬件兼容性`
> - `社区反馈落实`
> - `开放发布与许可`
> - `相比 Stable Diffusion 3 Medium 的改进`

- **Stable Diffusion 3.5 Medium 发布**：**Stable Diffusion 3.5 Medium** 模型现已开放免费商业和非商业使用，拥有 **25 亿参数**，旨在消费级硬件上运行，甚至支持低端设备。
  
  - 此次发布旨在通过确保模型能在低至 **9.9 GB VRAM** 的系统上运行，从而推动 AI 技术的民主化。
- **性能超越 Medium 级别模型**：Stable Diffusion 3.5 Medium 提供了一流的图像生成能力，具备 **先进的多分辨率功能**，在 **Prompt 遵循能力和图像质量** 方面优于其他中型模型。
  
  - 借助该模型，用户可以期待针对大多数消费级 GPU 量身定制的高效、高质量性能。
- **社区反馈塑造开发方向**：在 **6 月发布 Stable Diffusion 3 Medium** 之后，社区反馈促使团队进行了重大改进而非简单的修复，从而诞生了这一新版本。
  
  - 团队对倾听意见的承诺反映了其致力于通过为构建者和创作者提供更好的工具来 **变革视觉媒体**。
- **开放发布与灵活许可**：今日发布的版本包括多个 Stable Diffusion 3.5 变体，这些变体可定制且可在消费级硬件上运行，采用宽松的 **Stability AI Community License**。
  
  - 用户可以直接从 **Hugging Face** 下载模型，并在 **GitHub** 上获取代码。

**提及的链接**：[Introducing Stable Diffusion 3.5 — Stability AI](https://stability.ai/news/introducing-stable-diffusion-3-5)：今天我们推出 Stable Diffusion 3.5。此次开放发布包含多个模型变体，包括 Stable Diffusion 3.5 Large 和 Stable Diffusion 3.5 Large Turbo，截至 10 月 29 日...

---

### **Stability.ai (Stable Diffusion) ▷ #**[**general-chat**](https://discord.com/channels/1002292111942635562/1002292112739549196/1300543867241758861) (253 条消息🔥🔥):

> - `Stable Diffusion 3.5 Medium`
> - `GPU Performance Comparison`
> - `Sana Autoencoder`
> - `ComfyUI vs A1111`

- **Stable Diffusion 3.5 Medium 的图像生成**：用户注意到，与 **3.5 Large** 相比，**Stable Diffusion 3.5 Medium** 在生成超过 1MP 的图像时表现更好，能有效处理高达 2MP 的图像。
  - 有建议指出，虽然 3.5 Medium 在处理大尺寸图像方面表现更好，但超过该尺寸后也会开始出现崩溃。
- **GPU 性能与价格**：关于 GPU 价格的讨论显示，**3090** 显卡目前的价格与 **7900 XTX** 相似，一些用户发现二手 3090 的价格在 **$690** 左右。
  - 对各种 GPU（如 **7900 XTX**、**4080 Super**、**3090**）进行了对比，并对其在 AI 和游戏方面的性能提供了见解。
- **Sana Autoencoder 的影响**：提到了 **Sana** 自动编码器，因其利用 **deep compression techniques**（深度压缩技术）以更高比率训练和压缩图像的潜力。
  - 对于使用 Sana 生成的图像质量存在分歧，一些用户对使用该自动编码器训练的模型的有效性表示怀疑。
- **Stable Diffusion 中的 UI 切换**：用户讨论了从 **A1111** 切换到 **ComfyUI**，一些人尝试使用 **SwarmUI** 以获得更简化的体验。
  - 对话强调了在利用不同功能的同时，探索用于有效管理和生成图像的新界面。
- **提高图像 Prompt 遵循度**：一位用户寻求关于如何在 **ComfyUI** 中生成图像时确保模型准确遵循 Prompt 的建议。
  - 讨论包括了需要调整的技术设置，如 **steps**、**cfg** 和 **samplers**，以增强模型对 Prompt 的响应能力。

**提到的链接**：

- [SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers](https://hanlab.mit.edu/projects/sana)：未找到描述
- [stabilityai/stable-diffusion-3.5-medium · Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/)：未找到描述
- [city96/stable-diffusion-3.5-large-gguf at main](https://huggingface.co/city96/stable-diffusion-3.5-large-gguf/tree/main)：未找到描述
- [Stable Diffusion 3.5 fp8 models (SD3.5) - v3.5 large | Stable Diffusion Checkpoint | Civitai](https://civitai.com/models/879701/stable-diffusion-35-fp8-model)：官方 SD3.5 模型的 fp8 权重。在工作流中使用下方的加载器，“fast” 模式无法运行
- [stabilityai (Stability AI)](https://huggingface.co/stabilityai)：未找到描述
- [stabilityai/stable-diffusion-3.5-large at main](https://huggingface.co/stabilityai/stable-diffusion-3.5-large/tree/main)：未找到描述
- [Stable Diffusion 3.5 Large - Large | Stable Diffusion Checkpoint | Civitai](https://civitai.com/models/878387/stable-diffusion-35-large)：请参阅我们的 Stable Diffusion 3.5 快速入门指南以获取所有最新信息！Stable Diffusion 3.5 Large 是一个 Multimodal Diffusion Transformer (...
- [Stable Diffusion 3.5 fp8 models (SD3.5) - v3.5 large | Stable Diffusion Checkpoint | Civitai](https://civitai.com/models/879701/stable-diffusion-35-fp8-models-sd35)：官方 SD3.5 模型的 fp8 权重。在工作流中使用下方的加载器，“fast” 模式无法运行

---

### **Nous Research AI ▷ #**[**general**](https://discord.com/channels/1053877538025386074/1149866623109439599/1300543910623318016) (103 条消息🔥🔥):

> - `AI Newsletter 推荐`
> - `角色扮演 AI 角色开发`
> - `Layer Skip 研究发布`
> - `GitHub Copilot 增强功能`
> - `OpenAI 与 Microsoft 的关系动态`

- **寻找技术性 AI Newsletter**：一位成员询问是否有针对技术开发者的优质 AI Newsletter，并对目前普遍存在的针对消费者的炒作类 Newsletter 表示沮丧。
  
  - 另一位成员建议关注 [SemiAnalysis](https://link.to.semi-analysis)，但指出其重点在于 GPU。
- **开发哈利·波特 AI 角色扮演者**：一位成员寻求关于如何微调（finetune）一个以哈利·波特为原型的角色 AI 的见解，以在角色扮演场景中模拟思维过程和响应。
  
  - 他们正在考虑使用 Axotoxl 和 Llama-8b，但不确定这些模型在支持 Chain of Thought 方面的能力。
- **Meta 的 Layer Skip 实现已发布**：Meta 推出了其 [Layer Skip](https://go.fb.me/s8lary) 方案的推理代码和微调后的 Checkpoints，旨在提高 LLM 效率。
  
  - 这项研究旨在激发对 AI 优化和可解释性方法的新调查。
- **GitHub Copilot 的新选择**：GitHub Copilot 迎来了重大更新，包括引入了 **Claude 3.5 Sonnet** 和 **Gemini 1.5 Pro** 等模型，为开发者提供更多选择。
  
  - 这一变化被视为 **Anthropic** 的潜在胜利，预示着 AI 开发竞争格局的转变。
- **Microsoft 对 OpenAI 的战略依赖**：讨论指出，由于高度依赖以及 OpenAI 宣布实现 AGI 可能带来的风险，Microsoft 正在寻找 OpenAI 的替代方案。
  
  - 成员们推测 Microsoft 正在处理一段微妙的关系，并强调了多元化其 AI 合作伙伴关系的重要性。

**提到的链接**：

- [Hyperbolic AI Dashboard](https://app.hyperbolic.xyz/models/hermes3-70b]): 无描述
- [来自 AI at Meta (@AIatMeta) 的推文](https://x.com/AIatMeta/status/1851327605716435011?t=uCwZiiCcZqPQz0O9NjLfoQ&s=19)：我们之前分享了关于 Layer Skip 的研究，这是来自 Meta FAIR 研究人员的一种用于加速 LLM 的端到端解决方案。它通过执行 LLM 层的一个子集并利用 sub...
- [通过 Anthropic 的 Claude 3.5 Sonnet、Google 的 Gemini 1.5 Pro 和 OpenAI 的 o1-preview 为 Copilot 带来开发者选择](https://github.blog/news-insights/product-news/bringing-developer-choice-to-copilot/)：在 GitHub Universe 上，我们宣布 Anthropic 的 Claude 3.5 Sonnet、Google 的 Gemini 1.5 Pro 以及 OpenAI 的 o1-preview 和 o1-mini 将加入 GitHub Copilot——为每位开发者带来全新的选择水平...
- [GitHub - dottxt-ai/outlines: 结构化文本生成](https://github.com/dottxt-ai/outlines)：Structured Text Generation。通过在 GitHub 上创建账号为 outlines 的开发做出贡献。
- [CohereForAI/aya_collection · Hugging Face 数据集](https://huggingface.co/datasets/CohereForAI/aya_collection)：无描述

---

### **Nous Research AI ▷ #**[**ask-about-llms**](https://discord.com/channels/1053877538025386074/1154120232051408927/1300821223915393024) (4 条消息):

> - `用于角色扮演机器人的 Hermes 3`
> - `微调注意事项`
> - `Character.ai 模仿`

- **探索用于角色扮演机器人的 Hermes 3**：一位用户询问是否建议微调 **Hermes 3** 来创建一个模仿 **character.ai** 功能的角色扮演机器人。
  
  - 另一位参与者建议，他们可以直接通过 Prompt 指导 **Hermes 3** 来实现所需的行为，而无需进行微调。
- **微调 vs 提示策略**：初始用户考虑是使用 **Axotoxl** 微调一个新模型，还是对量化模型使用 System Cards，特别提到了 **Cat LLaMA 3 8B Instruct**。
  
  - 这场对话表明，人们对于优化模型以提高角色参与度和沉浸式角色扮演的兴趣日益浓厚。

**提到的链接**：[piotr25691/llama-3-cat-8b-instruct-v1-gguf · Hugging Face](https://huggingface.co/piotr25691/llama-3-cat-8b-instruct-v1-gguf)：无描述

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/) (1 条消息):

trre: [https://arxiv.org/abs/2410.14157](https://arxiv.org/abs/2410.14157)

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/) (1 条消息):

trre: [https://arxiv.org/abs/2410.14157](https://arxiv.org/abs/2410.14157)

---

### **Perplexity AI ▷ #**[**announcements**](https://discord.com/channels/1047197230748151888/1047204950763122820/1300548310456729654) (1 条消息):

> - `Curators Program`
> - `Discover Feed Contributions`
> - `Content Creation`
> - `Perplexity Engagement`

- **加入 Curators Program！**：Perplexity 团队正在招募首批 **Curators**（策展人）来为 **Discover feed** 贡献内容，这是一个与数百万用户互动的机会。
  
  - 如果你喜欢制作 **Pinterest 画板**、编辑 **Wikipedia 页面**或探索 **YouTube 视频论文**，可以[在此申请](https://perplexity.ai/curators)。
- **为用户打造富有启发性的 Pages**：Curators 将负责制作能够**启发**、**惊喜**并**告知**全球用户的 **Pages**，这些内容将直接呈现在产品中。
  
  - 这是一个塑造能引起用户共鸣的内容并提升其在 **Perplexity** 上体验的机会。

---

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1300539596945494059) (99 条消息🔥🔥):

> - `Grok Model Updates`
> - `Perplexity Pro Features`
> - `Coding Issues`
> - `Integration with GitHub`
> - `User Experience Feedback`

- **Grok 2 现已面向 Pro 用户开放**：Perplexity AI 宣布 Grok 2 已提供给 Pro 用户，允许他们在设置中将其设为默认模型。
  
  - 用户对 Grok 2 是否会取消审查表示好奇，一些人指出它并没有带来显著的改进。
- **周边商品发布公告**：Perplexity AI 正在推出其周边产品线 Perplexity Supply，计划于太平洋时间明天上午 9 点发布。
  
  - 宣传重点强调其品牌是“为好奇者而生”，吸引了社区的关注。
- **对编程辅助的担忧**：用户对 Perplexity 的编程能力表示不满，称这些模型并未针对编程任务进行优化。
  
  - 反馈包括模型提供无用回复的问题，尤其是在使用图片进行说明时。
- **与 GitHub Copilot 集成**：Perplexity 宣布与 GitHub 合作进行 Copilot 集成，增强了平台回答编程查询的能力。
  
  - 这允许用户直接在 GitHub 环境中获取更新和集成协助。
- **用户体验 Bug 与变更**：多名用户报告了与 Spaces 和 Collections 中功能缺失相关的 Bug，表明这些可能是已知问题。
  
  - 社区渴望得到修复，特别是关于 Focus 选择模式和平台整体功能方面。

**提到的链接**：

- [来自 Aravind Srinivas (@AravSrinivas) 的推文](https://x.com/aravsrinivas/status/1851178448007635017?s=61)：Grok 2，xAI 的最新模型，现已面向所有 Perplexity Pro 用户开放。你可以前往设置并将 Grok 选为默认模型。
- [来自 Aravind Srinivas (@AravSrinivas) 的推文](https://x.com/aravsrinivas/status/1851188395814633734?s=46)：@sahilpng 在 Perplexity 搜索的 "Focus" 部分，你可以选择 "Video" 并在 YouTube 上进行搜索。
- [来自 Perplexity (@perplexity_ai) 的推文](https://x.com/perplexity_ai/status/1851315707411337435?s=46)：我们很高兴能与 @github 合作。通过我们的 GitHub Copilot 集成，你将能够：• 随时掌握最新的库更新，例如“React 的最新更新” • 快速找到答案...
- [来自 Perplexity (@perplexity_ai) 的推文](https://x.com/perplexity_ai/status/1851341906271408469?s=46)：Perplexity Supply。为好奇者打造的周边。明天发布：http://perplexity.supply
- [Perplexity - 奔向无限](https://www.perplexity.ai/backtoschool)：欢迎回到学校！在接下来的两周内，可以兑换一个月免费的 Perplexity Pro。推荐你的朋友，如果你的学校达到 500 人注册，我们将把免费月份升级为一整年...
- [来自 Phi Hoang (@apostraphi) 的推文](https://x.com/apostraphi/status/1851309439480996177?t=YPsCvprfjgB-u3HETKtH8A&s=19>))：我们将在太平洋时间明天上午 9 点发布周边。http://perplexity.supply
- [Perplexity Supply：即将到来](https://perplexity.supply)：2024 年 10 月 30 日，当好奇心遇见品质。

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1300690285705035807) (8 条消息🔥):

> - `Apple Smartwatch Settlement` (Apple 智能手表和解案)
> - `NASA Economic Contributions` (NASA 经济贡献)
> - `Neural Impact Studies` (神经影响研究)
> - `Red Panda Image Model` (Red Panda 图像模型)
> - `Advancements in Photonic Computing` (光子计算的进展)

- **Apple 赢得 2.5 亿美元智能手表诉讼案**：Apple 最近在一场诉讼中获胜，赢得了针对一家智能手表竞争对手的 **2.5 亿美元** 赔偿，这对这家科技巨头来说是一次重大胜利。
  
  - 此案凸显了科技行业在知识产权和产品创新方面持续不断的 **法律纠纷**。
- **NASA 为美国经济贡献 760 亿美元**：一份报告指出，NASA 通过各种项目和创新为美国经济贡献了惊人的 **760 亿美元**。
  
  - 这一经济影响强调了 NASA 的角色不仅限于太空探索，还在于推动国内经济增长。
- **新型图像生成器主导基准测试**：一个新的图像生成器面世，其表现优于现有模型，并在各项基准测试中占据主导地位。
  
  - 这一发展引起了 AI 爱好者和研究人员的关注，标志着图像生成技术的 **重要里程碑**。
- **深入了解光子计算 (Photonic Computing)**：最近的讨论深入探讨了 **Cybersecurity** 领域中 **Photonic computing** 的进展，展示了其重塑该行业的潜力。
  
  - 成员们分享了在近期会议中获得的见解，强调了这些技术的 **变革性** 影响。
- **Vajra Shot 无人机枪的效能**：**Vajra Shot Drone Gun** 因其能够击落 **4 公里** 外的无人机而受到关注，展示了令人印象深刻的杀伤力。
  
  - 这一新发展引发了关于现代战争中针对无人机威胁的 **防御措施** 的讨论。

---

### **Notebook LM Discord ▷ #**[**use-cases**](https://discord.com/channels/1124402182171672732/1124403655819415592/1300539279126040628) (29 条消息🔥):

> - `BYD's Electric Vehicle Expansion` (比亚迪电动汽车扩张)
> - `NotebookLM for Staff Resources` (NotebookLM 用于员工资源)
> - `Podcast Generation Challenges` (播客生成挑战)
> - `Therapeutic Use of NotebookLM` (NotebookLM 的治疗用途)
> - `Real-Time Avatar Integration` (实时头像集成)

- **比亚迪 (BYD) 的电动汽车扩张**：一段视频讨论了中国电动汽车巨头 **BYD** 如何通过创新战略和技术进步，超越 **Tesla** 等巨头，从而颠覆汽车行业。
  
  - 该视频强调了 BYD 的全球扩张计划、激进的经销商开设策略以及对汽车市场的影响。
- **NotebookLM 作为员工资源指南**：一位用户分享了他们将 **NotebookLM** 作为员工资源指南的实施方案，整合了员工手册和 FAQs，重点在于改进内部查询。
  
  - 用户提出了关于该工具在提供文档内外部链接的 URLs 时存在不一致性的问题。
- **播客生成遇到的挑战**：用户对生成的播客质量和长度表示担忧，尽管使用了特定的自定义提示词，但仍难以生成 **20 分钟** 以内的简洁剧集。
  
  - 一些人注意到在引入自定义工具后，出现了幻觉 (hallucinations) 和内容重复的问题。
- **NotebookLM 的治疗用途**：一位用户分享了他们使用 **NotebookLM** 分析并从其精神病发作中获取见解的独特经验，强调了其治疗价值。
  
  - 他们对该工具能够连接复杂想法和过往经历的能力表示赞赏，这有助于他们的理解和小说创作。
- **播客中的实时头像集成**：关于在播客中集成 **Simli** 以实现实时头像的讨论，利用音频分段 (audio diarization) 技术使说话者的视觉效果与音频同步。
  
  - 这一概念验证展示了播客中动态视觉交互的潜力，增强了观众的体验。

**提到的链接**：

- [OK. This is Serious… China Is Taking Over Electric Vehicles with BYD](https://www.youtube.com/watch?v=VgAGSbreEMI)：深入探讨中国电动汽车巨头 BYD 不可阻挡的崛起，因为它正准备挑战全球市场并对标 Tesla 等巨头。
- [UNREAL MYSTERIES 4: The Halloween Special](https://www.youtube.com/watch?v=TwUrLHW8BwE)：David 和 Hannah 的虚幻之谜——万圣节特辑！这 100% 由 AI 生成。每一个人、每一张图片、每一个单词、每一个声音……

---

### **Notebook LM Discord ▷ #**[**general**](https://discord.com/channels/1124402182171672732/1124402182909857966/1300550446372360202) (60 条消息🔥🔥):

> - `NotebookLM 上传问题`
> - `西班牙语播客生成`
> - `NotebookLM 的开源替代方案`
> - `音频概览功能`
> - `笔记本上传限制`

- **NotebookLM 上传问题已解决**：上传文件（包括 PDF 和音频文件）到 NotebookLM 的问题已得到解决，用户现在可以再次成功上传文档。
  
  - 一位用户指出，同时处理多个文件可能会导致部分上传失败，建议分批少量上传。
- **生成西班牙语播客的挑战**：一位用户在最初成功创建两集后，在生成西班牙语播客时遇到困难，正在寻求潜在解决方案。
  
  - 另一位用户确认他们的西班牙语文本无法生成播客，暗示可能存在底层的语言处理问题。
- **探索开源替代方案**：用户讨论了使用名为 NotebookLlama 的 NotebookLM 开源替代方案的好处，强调了其功能并分享了该平台的链接。
  
  - 有人对 NotebookLlama 网站的正当性表示担忧，对其 DNS 和注册信息提出了质疑。
- **NotebookLM 音频概览自定义**：NotebookLM 音频概览功能的最新更新允许对播客内容进行重点设置，部分用户询问了新功能的可用性。
  
  - 参与者对播客的多语言支持表示感兴趣，并对该工具的能力和期望的增强功能进行了反思。
- **上传和笔记本创建的限制**：用户对 NotebookLM 中上传数量和笔记本数量的限制提出疑问，表示需要更好的平台内组织功能。
  
  - 一位用户提到，有必要单独上传笔记或将其转换为 Google Doc，以便整合到播客创建流程中。

**提到的链接**：

- [未找到标题](https://www.marktechpost.com/2024/10/27/meta-ai-silently-releases-notebookllama-an-open-source-alternative-to-googles-notebooklm/)：未找到描述
- [如何使用 Google 的 NotebookLM 创建和自定义 AI 播客](https://www.forbes.com/sites/rogerdooley/2024/10/24/how-to-create-and-customize-an-ai-podcast-with-googles-notebooklm/)：Google 在 NotebookLM 音频概览中新增的“自定义”功能，让你能够根据任何内容创建具有特定侧重点的逼真播客。以下是操作方法。
- [未找到标题](https://www.marktechpost.com/2024/10/27/meta-ai-silentl)：未找到描述
- [Notebook Llama | Llama API](https://www.notebookllama.ai/)：Notebook Llama 在 Llama 系列上部署了 Meta 的 Llama 方案。这是一个开源项目，利用了 Meta 的 Llama AI 系列：Llama 3.2、Llama 3.1 以及开源的 Parler text to...
- [比亚迪的全球电动汽车征服：汽车行业巨头的终极颠覆](https://youtu.be/UxXtvIt0WtA)：深入了解比亚迪这个电力十足的世界，这家中国巨头不仅在竞争，更旨在主导全球电动汽车 (EV) 市场。凭借...
- [AI 笔记、转录与摘要 | AI Notebook App](https://ainotebook.app/)：为大学生的讲座生成转录和 AI 摘要。专注于 YouTube 视频摘要、PDF 摘要、文章摘要。保存关键见解并使用学习指南、测验进行复习...
- [Text-to-Speech AI: 逼真的语音合成 | Google Cloud](https://cloud.google.com/text-to-speech)：通过由 Google 机器学习技术支持的 API，将文本转换为 40 多种语言和变体中 220 多种声音的自然语音。
- [Podcastfy.ai - NotebookLM 播客功能的开源替代方案 - thatupiso 在 Hugging Face 上的 Space](https://huggingface.co/spaces/thatupiso/Podcastfy.ai_demo)：未找到描述
- [GitHub - souzatharsis/podcastfy: NotebookLM 播客功能的开源替代方案：利用 GenAI 将多模态内容转化为引人入胜的多语言音频对话](https://github.com/souzatharsis/podcastfy)：NotebookLM 播客功能的开源替代方案：利用 GenAI 将多模态内容转化为引人入胜的多语言音频对话 - souzatharsis/podcastfy

---

### **GPU MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1189498205101109300/1300538962523459724) (6 条消息):

> - `Unsloth Kernels`
> - `评估中的数值精度`
> - `CUDA/Triton 项目`

- **寻求 Unsloth Kernels 指南**：一名成员询问了关于 [unsloth kernels](https://github.com/unslothai/unsloth) 的指南，该项目可以微调多个 LLM，并显著提升性能和显存效率。
  
  - GitHub 仓库显示，它支持 **Llama 3.2**、**Mistral**、**Phi** 和 **Gemma LLM** 的微调，速度提升 2-5 倍，且**节省 80% 显存**。
- **关于数值精度的讨论**：一位成员分享了关于评估模型的见解，指出与 **FP32** 相比，使用 **FP16** 或 **BF16** 可能会导致不同实现之间的移植性问题。
  
  - 他们强调，在 **FP32** 下，数值误差很小且累积不明显，而 **BF16** 的误差则变得非常显著，尤其是在长上下文（longer contexts）中。
- **征求作品集项目建议**：一名成员向社区征求一些简单的 **CUDA** 或 **Triton** 项目，以便用于自己的作品集。
  
  - 这表明他们有兴趣寻找一些易于上手的项目，以增强实践技能并向潜在雇主展示经验。

 

**提到的链接**：[GitHub - unslothai/unsloth: Finetune Llama 3.2, Mistral, Phi & Gemma LLMs 2-5x faster with 80% less memory](https://github.com/unslothai/unsloth)：微调 Llama 3.2, Mistral, Phi & Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth

 

---

### **GPU MODE ▷ #**[**triton**](https://discord.com/channels/1189498204333543425/1189607595451895918/1300655677303423058) (5 条消息):

> - `Triton kernel 性能`
> - `Triton 中的 BF16 操作`
> - `Triton nightly 版本问题`
> - `Triton AST 到 PTX`
> - `合并 Triton 的改进`

- **单算子 Triton kernel 速度较慢**：一位用户注意到，将所有操作放入**一个 kernel** 中会导致性能比 **PyTorch** 更差，这表明需要多个 kernel 才能实现加速。
  
  - 他们推测这可能是因为 Triton 在为处理大量中间值的 kernel **分配共享内存（shared memory）**时存在困难。
- **Triton 中的 BF16 转换未提升速度**：有成员提到 **BF16** 操作并未显著提升性能，因为用户在矩阵乘法期间对输入和输出进行了类型转换。
  
  - 产生了一个疑问：Triton 是否在某些环节存储了 **FP32** 结果，从而可能导致过程变慢。
- **Triton 的 Nightly 版本面临问题**：一名成员报告称，Triton 的 **nightly 版本**已经损坏了大约**三个月**，并指向了一个 GitHub issue 以获取详情。
  
  - 另一位成员评论说，来自 PyTorch Dev infra 团队的 **Andrey** 目前正在调查此情况。
- **Triton AST 到 PTX 的 fork 分支运行正常**：讨论中提到了一个在 **Triton AST 到 PTX** 转换路径中运行良好的 **fork**。
  
  - 一位用户建议审查并可能合并这个 fork，并指出其文档看起来可能比实际情况更复杂。
- **合并 Triton 的改进**：有人建议合并上述 fork 可以增强 **Triton 的功能**。
  
  - 交流中强调了此举可能带来的潜在收益，尽管 README 最初给人的感觉比较复杂。

 

**提到的链接**：[Wheels · Workflow runs · triton-lang/triton](https://github.com/triton-lang/triton/actions/workflows/wheels.yml)：Triton 语言和编译器的开发仓库 - Wheels · Workflow runs · triton-lang/triton

 

---

### **GPU MODE ▷ #**[**torch**](https://discord.com/channels/1189498204333543425/1189607750876008468/1300762010090209281) (4 messages):

> - `H100 speed-up tricks`
> - `FSDP2 API deprecation`
> - `TorchAO optimizers`

- **H100 极限性能：新的加速技巧**：一位用户报告了在 **H100** 上使用两种配置实现的巨大加速：`reduce-overhead` 达到了 **255 tokens/sec**，而 `max-autotune-no-cudagraphs` 结合手动 **CUDA Graphs** 则达到了 **300 tokens/sec**。
  
  - 这些结果显示了显著的性能提升，可能会让社区中的其他成员受益。
- **FSDP2 API 即将弃用**：关于 **FSDP2** 及其 **fully_shard** API 的讨论引发了关注，一份弃用通知敦促用户切换回 **FSDP**，详情见 [GitHub Issue](https://github.com/pytorch/pytorch/issues/114299)。
  
  - 用户对该 API 的未来表示担忧，并引用了一项关于在 **PyTorch 2.5** 之后移除 `torch.distributed._composable.fully_shard` 的警告。
- **TorchAO 优化器缺乏 SR 支持**：一位成员询问 **TorchAO optimizers** 现在是否支持 **SR**，并指出之前并不支持，这可能是一个潜在的贡献领域。
  
  - 他们表示有兴趣在未来时间允许时提交 **PR**，这表明了对增强优化能力的迫切需求。

 

**提到链接**：[Issues · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/114299)：Python 中具有强 GPU 加速能力的张量和动态神经网络 - Issues · pytorch/pytorch

 

---

### **GPU MODE ▷ #**[**cool-links**](https://discord.com/channels/1189498204333543425/1189868872887705671/1300638885390319656) (1 messages):

> - `Colossus Supercomputer`
> - `NVIDIA Spectrum-X`
> - `xAI Grok Models`
> - `AI Networking Performance`

- **Colossus 成为 AI 超级计算机巨头**：NVIDIA 宣布，位于孟菲斯的 xAI **Colossus 超级计算机**配备了 **100,000 个 NVIDIA Hopper GPU**，是全球最大的 AI 超级计算机，旨在处理 **hyperscale AI** 工作负载。
  
  - 该设施的建设仅耗时 **122 天**，比此类大型系统的典型建设周期快得多。
- **NVIDIA Spectrum-X 彻底改变 AI 网络**：**NVIDIA Spectrum-X™** 以太网网络平台为 Colossus 超级计算机提供了支撑，确保了多租户 AI 架构的高性能 **Remote Direct Memory Access** (RDMA) 网络。
  
  - 这种基于标准的以太网解决方案专为 **hyperscale AI factories** 量身定制，旨在提高运营效率。
- **xAI 加倍投入 Colossus，增加更多 GPU**：xAI 正在将 Colossus 超级计算机的 GPU 容量**翻倍**，总计将达到 **200,000 个 NVIDIA Hopper GPU**，以支持其 Grok 语言模型。
  
  - 随着聊天机器人向 **X Premium 订阅用户**开放，这一扩张展示了对 AI 能力日益增长的需求。

 

**提到链接**：[NVIDIA Ethernet Networking Accelerates World’s Largest AI Supercomputer, Built by xAI](https://nvidianews.nvidia.com/news/spectrum-x-ethernet-networking-xai-colossus?ncid=so-link-344451&linkId=100000302783167)：NVIDIA 今日宣布，由 xAI 在田纳西州孟菲斯构建的包含 100,000 个 NVIDIA Hopper GPU 的 Colossus 超级计算机集群，通过使用 NVIDIA Spectrum-X™ 以太网网络实现了这一巨大规模...

 

---

### **GPU MODE ▷ #**[**jobs**](https://discord.com/channels/1189498204333543425/1190208177829068860/1300809310296735801) (6 条消息):

> - `Cracked Engineers 平台`
> - `技术职位 Newsletter`
> - `AI 初创公司`
> - `用户反馈`
> - `Gigachad 图片`

- **Cracked Engineers 招聘平台发布**：一个名为 **Cracked Engineers** 的新招聘平台已针对技术岗位发布，旨在将用户与 AI/技术初创公司联系起来。该平台旨在自动化招聘流程，使候选人和公司都能更轻松地进行对接。
  
  - 该网站提供**每周技术职位 Newsletter** 和 AI 辅助职位发布等功能，以增强用户体验。
- **用户喜爱与反馈**：用户对该平台表示兴奋，其中一位指出它实现了他们简化实习寻找的希望。
  
  - 另一位用户幽默地赞赏了 **gigachad 图片**，展示了对话中的轻松氛围。
- **用户支持与互动**：平台创建者鼓励用户发送反馈，并为任何问题提供帮助。这种积极主动的方法旨在确保新功能推出时用户体验顺畅。
  
  - 社区参与积极，对网站的功能和娱乐元素都给出了正面反应。

**提到的链接**：

- [Cracked Engineers](https://www.crackedengineers.com/)：为你的初创公司寻找最优秀的工程师。
- [Cracked Engineers - 寻找技术职位 / 招聘技术人才](https://youtu.be/XmuIOdES7mQ)：我刚刚构建了一个名为 "Cracked Engineers" 的技术职位平台！:)) 在这段视频中，我将带你快速浏览并演示它的工作原理。你可以在这里找到它...

---

### **GPU MODE ▷ #**[**beginner**](https://discord.com/channels/1189498204333543425/1191300313928433664/1300551342083014667) (14 条消息🔥):

> - `CUDA 学习资源`
> - `PMPP 的数学前置要求`
> - `为 CUDA 开发选择硬件`

- **预算有限下的 CUDA 学习**：一位成员询问在 3500 美元的预算下，是应该投资一台搭载 NVIDIA GPU 的笔记本电脑，还是购买一台更强大的台式机来学习 CUDA。
  
  - 另一位成员建议使用 [Kaggle](https://www.kaggle.com) 和 [Colab](https://colab.research.google.com) 等提供免费 GPU 的平台，以获得 CUDA 的动手实践经验。
- **PMPP 的数学基础**：一位成员表达了对在未完成线性代数（仅学到微积分 II）的情况下直接阅读 PMPP 书籍的担忧。
  
  - 建议指出，即使没有线性代数基础也可以开始学习 PMPP，并强调许多 Kernel 都集中在线性代数和数值方法上。
- **特定于 CUDA 的 Kernel 开发**：讨论中提到了对特定于 CUDA 的 Kernel 开发的渴望，而不仅仅是使用 TensorFlow 或 PyTorch 等高级框架。
  
  - 一位成员推荐查看 [CUDA Mode GitHub](https://github.com)，了解关于将自定义 Kernel 移植到 PyTorch 的讲座，作为实践步骤。

 

---

### **GPU MODE ▷ #**[**torchao**](https://discord.com/channels/1189498204333543425/1205223658021458100/1300848181956972554) (1 条消息):

> - `PR 反馈请求`
> - `推理吞吐量`
> - `性能改进`

- **寻求对 PR #1401 的反馈**：一位成员正在寻求对其 [PR #1401](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1401) 的反馈，该 PR 目前处于草案状态，旨在增强 **LLM.int8()** 的实现。
  
  - 他们强调该 PR 包含大量内容，并渴望收到社区的见解。
- **报告 2 倍推理吞吐量**：该成员报告称，在 **4090** 上，对于不带稀疏分解（sparse decomp）的 **int8**，通过将阈值设置为 **0.0**，实现了约 **2 倍的推理吞吐量**。
  
  - 他们还在积极研究解压缩，并正在对 **nf4/fp4** 进行细微的性能改进。

 

**提到的链接**：[LLM.int8() 重构：第一部分，由 matthewdouglas 提交 · Pull Request #1401 · bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1401)：此 PR 是旨在改进 LLM.int8() 实现的一系列更改的初始阶段。目前仍处于草案阶段，但由于内容较多，我已经准备好接受审阅...

 

---

### **GPU MODE ▷ #**[**triton-puzzles**](https://discord.com/channels/1189498204333543425/1219683012707487794/1300610224981282876) (1 条消息):

> - `Triton Installation`
> - `Dependencies for Triton Visualization`

- **为 Triton 安装所需包**：一位用户提供了一组 **Triton** 的安装命令，其中包括 `jaxtyping`、`triton` 以及来自 [Deep-Learning-Profiling-Tools](https://github.com/Deep-Learning-Profiling-Tools/triton-viz) 的 Triton 可视化工具。
  
  - 这些命令被指出对最新的 Triton 版本有效，并强调只需运行一次即可完成环境搭建。
- **修改 Triton Viz 以输出 PNG**：提供的脚本包含了修改 Triton 可视化中文件输出格式的步骤，通过特定的 `sed` 命令将 `.svg` 更改为 `.png`。
  
  - 根据分享的用户经验，这一调整对于确保与不支持 SVG 格式的工具的兼容性至关重要。
- **为兼容性设置环境变量**：使用 export 命令设置了 locale 和库路径的环境变量，以确保 **Triton** 正常运行。
  
  - 脚本强调在安装相应的库后运行 `ldconfig`，以支持 Triton 中的图形操作。
- **安装额外依赖**：安装命令列表包括 `libcairo2-dev` 等系统包和 Python 开发头文件，以支持 Triton 的图形功能。
  
  - 用户还被引导安装 **pycairo** 以进一步增强可视化功能。

---

### **GPU MODE ▷ #**[**hqq-mobius**](https://discord.com/channels/1189498204333543425/1225499037516693574/1300747055618195509) (37 条消息🔥):

> - `GEMV optimization without tl.dot`
> - `Performance comparisons on different GPUs`
> - `Machete kernel for H100`
> - `Instruction ordering in Triton`
> - `Custom operations in Triton kernels`

- **GEMV 优化超出预期**：一位成员报告称，通过采用自定义的反向拆分（reverse-split）GEMV 算法，在不使用 `tl.dot` 的情况下实现了令人印象深刻的性能，达到了高达 **184 tokens/sec**。
  
  - 这种方法最小化了加载 scales/zeros 的开销，证明在 **batch-size = 1** 时特别有效。
- **在各种 GPU 上的性能对比**：该成员强调其方法在包括 **ADA** 和 **3090** 在内的多种 GPU 上表现良好，但由于 `tl.load` 较慢，在 **A100/H100** 上表现不佳。
  
  - 他们发现其性能与 **4-bit kernels** 相当，并指出像 **2080 Ti** 这样的旧款 GPU 表现也不错。
- **针对 H100 的 Machete kernel 问世**：引入了一个名为 **Machete** 的新 kernel，专门为 **H100** 定制，尽管在较大 batch size 下存在局限性。
  
  - 该成员对其效率表示不确定，因为它依赖于像 **Marlin** 这样的量化零点（quantized zeros），这可能会限制其通用性。
- **性能受指令排序影响**：讨论围绕 **指令排序（instruction ordering）** 在实现最佳性能中的重要性展开，其中在权重之前加载激活值（activations）会显著降低处理速度。
  
  - 不同 GPU 上复杂的排序处理使得这一话题对于那些正在微调其 Triton 实现的人来说尤为值得关注。
- **Triton kernel 中自定义操作的挑战**：在 Triton kernel 中使用 `custom_op` 带来了挑战，特别是由于 `torch.compile` 缺乏对 `pre_hook` 和 `prune_configs_by` 等特性的支持。
  
  - 讨论中幽默地提到了用于通过 `custom_op` 重新加载模块的 hack 手段，展示了所遇到的复杂性和奇特的解决方案。

**提到的链接**：

- [gemlite/gemlite/triton_kernels/gemv_revsplitK_A16fWnO16f_int32packing.py at master · mobiusml/gemlite](https://github.com/mobiusml/gemlite/blob/master/gemlite/triton_kernels/gemv_revsplitK_A16fWnO16f_int32packing.py)：CUDA / Triton 中简单快速的低比特 matmul kernels - mobiusml/gemlite
- [GitHub - mobiusml/gemlite: Simple and fast low-bit matmul kernels in CUDA / Triton](https://github.com/mobiusml/gemlite/?tab=readme-ov-file#performance)：CUDA / Triton 中简单快速的低比特 matmul kernels - mobiusml/gemlite

---

### **GPU MODE ▷ #**[**llmdotc**](https://discord.com/channels/1189498204333543425/1227345713348870156/1300846672829481011) (1 条消息):

> - `Cracked Engineers 招聘平台`
> - `每周技术职位通讯`
> - `AI 辅助职位发布`

- **Cracked Engineers 招聘平台发布**：一个名为 [Cracked Engineers](https://www.crackedengineers.com) 的新招聘平台旨在将技术人才与顶尖 AI 和技术初创公司联系起来，在正式发布前其 MRR（月经常性收入）已接近 **$1000**。
  
  - 创始人表达了对招聘协助的兴趣，从而实现了职位发布的自动化，使其成为连接人才与机会的可扩展解决方案。
- **令人兴奋的每周技术职位通讯**：将推出每周技术职位通讯，允许用户直接从仪表板订阅特定的职位标签，如 **CUDA**、**Triton** 和 **Software Engineering**。
  
  - 该通讯旨在分享最新的 AI 职位，同时向公司提供有关其发布职位的浏览量和申请者的反馈。
- **AI 简化职位发布流程**：该平台配备了一个 AI 工具，可在短短一分钟内简化职位发布表单，确保公司在提交前可以轻松创建和预览其帖子。
  
  - 这种创新方法有助于发现拼写错误，并提升雇主的整体发布体验。
- **鼓励社区反馈**：邀请用户在 [Canny](https://crackedengineers.canny.io/cracked-engineers) 上分享反馈和建议，随着平台的发展增强其有效性。
  
  - 还创建了一个专门的 Discord 频道，用于实时职位发布和社区互动。

 

**提到的链接**：[Aleksa Gordić 🍿🤖 (@gordic_aleksa) 的推文](https://x.com/gordic_aleksa/status/1851247076987855063)：[🚀] 非常激动地分享：我构建了一个名为 "Cracked Engineers" 的技术职位招聘平台。:) 如果你想在一些世界顶尖的 AI/技术初创公司找到工作...

 

---

### **GPU MODE ▷ #**[**rocm**](https://discord.com/channels/1189498204333543425/1233704710389764236/1300653547079204904) (6 条消息):

> - `Composable Kernel`
> - `MFMA bank 冲突`
> - `MI250x 性能`

- **寻求关于 Composable Kernel swizzle 规则的专业知识**：一位成员询问了如何在 **Composable Kernel** 中应用 **swizzle 规则**，以避免 **MFMA** 操作的 **bank 冲突**。
  
  - 另一位成员建议这可能与 *make_xor_transform* 有关，但对其效率表示不确定。
- **MI250x 上的性能基准测试**：一位成员报告在 **0.5 MI250x** 上达到了 **125-130** 的分数。
  
  - 这一性能得到了另一位成员的确认，反映了多次测试的一致结果。
- **对 MI250x 性能指标的困惑**：一位成员在看到 MI250x 上报告的 **147** 分后表示困惑，这与他们自己的测试结果不同。
  
  - 这引发了关于性能指标变异性以及差异潜在原因的简短讨论。

 

---

### **GPU MODE ▷ #**[**liger-kernel**](https://discord.com/channels/1189498204333543425/1275130785933951039/) (1 条消息):

0x000ff4: [https://github.com/linkedin/Liger-Kernel/pull/321](https://github.com/linkedin/Liger-Kernel/pull/321)

---

### **GPU MODE ▷ #**[**thunderkittens**](https://discord.com/channels/1189498204333543425/1300872762163728550/1300877183061065779) (1 条消息):

> - `ThunderKittens 0.000000002 Release`
> - `TK Kernels 的新特性`
> - `关于 AI Kernels 的最新论文`
> - `即将发布的博客文章`
> - `征集贡献`

- **ThunderKittens 0.000000002 发布，带来重大更新！**：备受期待的 **ThunderKittens 0.000000002** 已于今日发布，拥有更精简的特性集和显著的性能提升。
  
  - 关键增强功能包括**比 FA3 更快的 Attention 反向传播**以及**快 6-14 倍的 Linear Attention**。
- **TK Kernels 获得性能提升！**：**TK kernels** 的新更新展示了令人惊叹的进展，例如**达到 CuBLAS 速度的 GEMM** 和**快 8 倍的长卷积（Long Convolutions）**。
  
  - 演示现在包含 Llama 和 Qwen 模型，展示了新优化的功能。
- **关于 Kernel 性能挑战的新论文发布**：一篇题为 [View PDF](https://arxiv.org/abs/2410.20399) 的新论文探讨了将 AI 架构映射到 GPU 硬件时的关键瓶颈。
  
  - 论文指出，尽管硬件具备各种能力，但**手写的自定义 Kernel** 往往无法达到其理论性能潜力。
- **关于 TK 和 GPU 编程的精彩博客更新！**：一篇新博客文章详细介绍了 **ThunderKittens** 的更新和 GPU 编程，并附带了之前讨论的链接。
  
  - 读者可以期待在未来几周内看到一系列直播和深度博客，让大家持续关注 TK 的进展。
- **征集 ThunderKittens 贡献者**：公开邀请更多贡献者加入 **ThunderKittens** 项目，帮助增强其功能。
  
  - **GitHub 仓库**中提供了一份详细的所需 Kernel 和特性列表，鼓励社区参与。

**提到的链接**：

- [GitHub - HazyResearch/ThunderKittens: Tile primitives for speedy kernels](https://github.com/HazyResearch/ThunderKittens/tree/main)：用于高速 Kernel 的 Tile 原语。通过在 GitHub 上创建账号来为 HazyResearch/ThunderKittens 的开发做出贡献。
- [ThunderKittens: Simple, Fast, and Adorable AI Kernels](https://arxiv.org/abs/2410.20399)：将 AI 架构映射到 GPU 硬件的挑战正成为 AI 进步的关键瓶颈。尽管付出了巨大努力，手写的自定义 Kernel 仍无法达到其理论性能...
- [Easier, Better, Faster, Cuter](https://hazyresearch.stanford.edu/blog/2024-10-29-tk2)：未找到描述

---

### **LM Studio ▷ #**[**general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1300572297190440993) (17 条消息🔥):

> - `Token 处理速度`
> - `LLM 推荐`
> - `上下文管理问题`
> - `CPU 线程数困惑`
> - `大上下文错误处理`

- **CPU 与 GPU 的 Token 处理速度对比**：成员们注意到 **Token 处理速度**在 **GPU 上约为 62 tok/sec**，在 **CPU 上约为 7.5 tok/sec**。
  
  - *Fewill* 在讨论这些速度时表达了热情，说道：“nice!”。
- **寻求本地 LLM 的推荐**：一位成员正在寻求类似于 **Phind** 或 **ChatGPT** 的**本地运行 LLM** 推荐，用于回答有关 Python 和 Houdini SideFX 的问题。
  
  - *Fabguy* 建议研究 **HumanEval**，但指出 Houdini 的小众性质可能会阻碍 LLM 给出相关的回答。
- **澄清 CPU 线程数指标**：一位用户询问了**加载选项卡（load tab）中的 CPU 线程数**与**推理选项卡（interference tab，应为 inference）中的 CPU 线程数**之间的区别。
  
  - 这表明用户对平台内的性能指标存在困惑，尽管目前尚未给出明确答案。
- **大上下文尺寸的问题**：一位成员报告在**使用 65k 或 128k tokens 等大上下文**时遇到错误，凸显了上下文长度支持方面的问题。
  
  - 他们提交了错误报告，并指出错误可能源于 **Softmax** 调整，而非 LM Studio 界面。
- **理解生成过程中的 n_keep 参数**：讨论中提出了关于在生成过程中如何设置 **n_keep** 的问题，特别是与 **Token 溢出错误**相关的问题。
  
  - 这场对话引发了对 **Softmax** 函数中影响生成输出的设置的好奇。

---

### **LM Studio ▷ #**[**hardware-discussion**](https://discord.com/channels/1110598183144399058/1153759714082033735/1300536437678276608) (63 条消息🔥🔥):

> - `Mac Memory Concerns` (Mac 内存担忧)
> - `NGINX Proxy Issues` (NGINX 代理问题)
> - `PCIE Bandwidth and Inference` (PCIe 带宽与推理)
> - `Fractal Torrent Case` (Fractal Torrent 机箱)
> - `Multi-GPU setups` (多 GPU 配置)

- **Mac 饱受最小内存困扰**：用户对 Apple 交付仅配备最小内存的 Mac 的做法表示沮丧，尤其是升级费用昂贵且购买后无法升级。
  
  - *内存限制*是处理内存密集型应用程序的用户关注的显著问题。
- **设置 NGINX 代理的挑战**：尽管激活了 *serve on local network*，一位用户在为 LM Studio 服务器配置 NGINX 代理主机时遇到了困难。
  
  - 其他人讨论了排除此设置故障的各种步骤，强调了配置设置的重要性。
- **PCIe 带宽对推理的影响**：关于 PCIe 带宽是否影响推理性能引发了辩论，有人建议 PCIe Gen 3 就足够了，因为大部分处理发生在 GPU 上。
  
  - 然而，用户指出，对于在需要高带宽的多 GPU 之间训练模型，带宽变得至关重要。
- **专为散热设计的 Fractal Torrent 机箱**：Fractal Torrent 机箱因其卓越的散热能力而受到关注，具有定制风扇、开放式格栅设计，并获得了创新奖项。
  
  - 它迎合了高端散热需求，引发了尝试各种机箱设计的用户的积极评价。
- **多 GPU 配置查询**：一位用户询问了使用多个 3090 运行大型模型的情况，质疑当模型超过单个 GPU 的内存容量时是否会发生性能损失。
  
  - 结论是，如果显卡相同，性能将保持稳定，并且将更多任务 Offloading 到 GPU 上可以提高整体处理效率。

**提及的链接**：

- [Torrent](https://www.fractal-design.com/products/cases/torrent/torrent/black-solid/)：专为开箱即用的最大化散热潜力而打造，Torrent 配备了全新的组件布局和两个定制的 180 x 38 mm Dynamic PWM / Prisma RGB 风扇。Torrent 是一款完美的...
- [Voron Cable Management GIF - Voron Cable Management - Discover & Share GIFs](https://tenor.com/view/voron-cable-management-gif-22392132)：点击查看 GIF
- [Mac mini - 技术规格](https://www.apple.com/mac-mini/specs/)：查看配备 M4 或 M4 Pro 芯片的 Mac mini 的所有技术规格。
- [imgur.com](https://imgur.com/I32QCaX)：在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过有趣的笑话、热门迷因、娱乐 GIF、励志故事、病毒视频等来振奋精神...

---

### **aider (Paul Gauthier) ▷ #**[**general**](https://discord.com/channels/1131200896827654144/1131200896827654149/1300545878590754907) (31 条消息🔥):

> - `Aider Performance Issues` (Aider 性能问题)
> - `Web Scraping and Automation Tools` (Web 抓取与自动化工具)
> - `Git Repository Management` (Git 仓库管理)
> - `GitHub Copilot vs Aider` (GitHub Copilot 对比 Aider)
> - `Using Aider with Claude` (配合 Claude 使用 Aider)

- **Aider 对部分用户运行缓慢**：一些成员报告了 Aider 的**运行缓慢**问题，特别是 litellm 的 **get_model_cost_map** 函数，直到他们设置了 `export LITELLM_LOCAL_MODEL_COST_MAP='True'` 后速度才有所提升。
  
  - 一位用户指出，在大多数情况下 Aider 会尝试掩盖 litellm 的**延迟**。
- **Web 自动化建议**：一位用户正在寻求关于 **web scraping** 和自动化的建议，希望让 LLM 直接与网站交互。
  
  - 建议中包括使用 **RapidAPI** 来处理此类任务。
- **使用 Aider 管理 Git 仓库**：几位用户讨论了避免污染现有 **Git repository** 的策略，建议手动提交更改，而不是依赖 Aider 的自动提交功能。
  
  - 一位用户分享了一个流程：使用 `git switch` 配合合并 squash commits 来保持仓库整洁。
- **GitHub Copilot 与 Aider 的竞争**：一位成员指出，Copilot 现在正在集成 **OpenAI**、**Gemini** 和 **Anthropic models**，这可能会影响其与 Aider 的竞争。
  
  - 另一位成员分享说他们已经切换到了 Supermaven，理由是对 Copilot 的性能感到不满。
- **配合 Claude 使用 Aider 的有效策略**：用户们合作探讨了配合 Claude 使用 Aider 的策略，强调了手动提交以保持控制权的有效性。
  
  - 一位用户还建议将文件拆分为 **5-10 个一组的批次**，以便更好地管理。

**提到的链接**：

- [Bringing developer choice to Copilot with Anthropic’s Claude 3.5 Sonnet, Google’s Gemini 1.5 Pro, and OpenAI’s o1-preview](https://github.blog/news-insights/product-news/bringing-developer-choice-to-copilot/,): 在 GitHub Universe 上，我们宣布 Anthropic 的 Claude 3.5 Sonnet、Google 的 Gemini 1.5 Pro 以及 OpenAI 的 o1-preview 和 o1-mini 即将登陆 GitHub Copilot——为每位开发者带来全新的选择层级...
- [What is GitHub Spark? Introducing a brand new way to build powerful, AI assisted applications](https://www.youtube.com/watch?v=oM2amcnVmzM): 在 GitHub Universe 2024 上，我们介绍了一种构建强大的 AI 辅助应用程序的全新方式。GitHub Spark 是一款 AI 原生工具，用于构建应用程序...

---

### **aider (Paul Gauthier) ▷ #**[**questions-and-tips**](https://discord.com/channels/1131200896827654144/1133060505792159755/1300545116607479949) (46 条消息🔥):

> - `FireCrawl for scraping` (使用 FireCrawl 进行抓取)
> - `Reddit data extraction` (Reddit 数据提取)
> - `Debugging strategies` (调试策略)
> - `Prompt engineering` (Prompt engineering)
> - `Tool recommendations` (工具推荐)

- **探索用于抓取的 FireCrawl**：一位成员建议使用 [FireCrawl](https://firecrawl.dev) 进行 Web 抓取，因为它提供了有效的提取能力，同时支持自托管选项。
  
  - 经过适当配置后，该工具可以帮助绕过社交媒体抓取时遇到的限制。
- **使用 JSON 提取 Reddit 数据**：一位用户分享了他们使用 `.json` 格式从 Reddit 帖子中提取评论的变通方法，这允许直接解析正文内容。
  
  - 尽管这种方法有效，但也有人指出 FireCrawl 可能会为数据抓取提供更健壮的解决方案。
- **关于 Prompt engineering 的见解**：讨论涉及了如何构建有效的 AI Prompt，强调了它们对获取准确且详尽输出的影响。
  
  - 成员们提到了上下文的重要性，以及使用多样化 Prompt 以避免 AI 在交互过程中产生混淆。
- **AI 准确性的挑战**：用户对 AI 输出的准确性和可靠性表示担忧，并分享了 AI 在调试过程中提供误导性陈述的经历。
  
  - 这引发了关于如何更好地构建 Prompt 结构以避免跳过关键步骤或提供不完整信息的广泛讨论。
- **自托管方案的潜力**：自托管选项被强调为确保安全性和合规性的优势，特别是对于有严格政策的组织。
  
  - 社区鼓励探索文档并利用内部工具来优化工作流程。

**提到的链接**：

- [File editing problems](https://aider.chat/docs/troubleshooting/edit-errors.html): Aider 是你终端里的 AI 结对编程工具
- [Self-hosting | Firecrawl](https://docs.firecrawl.dev/contributing/self-host): 暂无描述

---

### **aider (Paul Gauthier) ▷ #**[**links**](https://discord.com/channels/1131200896827654144/1268910919057149974/1300917236680360001) (1 messages):

> - `New Bash and Editor Tools`
> - `Aider Code Assistants`

- **探索来自 Claude Anthropic 的新 Bash 和编辑器工具**：一场讨论强调了来自 Claude Anthropic 的**新 Bash 和编辑器工具**，展示了它们在增强编码环境方面的潜力。
  
  - 参与者推测了这些工具在 Aider 等现有代码助手中的适用性，建议通过集成来改进功能。
- **新工具的 GitHub 仓库**：分享了一个 [GitHub 仓库](https://github.com/disler/anthropic-computer-use-bash-and-files) 的链接，提供了关于如何使用 Bash 和文件实现 **Anthropic 工具** 的资源。
  
  - 该仓库可以作为开发者在项目中利用这些创新的宝贵参考。

 

**提到的链接**：[GitHub - disler/anthropic-computer-use-bash-and-files](https://github.com/disler/anthropic-computer-use-bash-and-files)：通过在 GitHub 上创建账号，为 disler/anthropic-computer-use-bash-and-files 的开发做出贡献。

 

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1300538495030263848) (54 messages🔥):

> - `AI Research Grants`
> - `Evolution of Algorithms`
> - `Anthropomorphism of AI`
> - `Ethical Considerations in AI`
> - `AGI Development Stages`

- **关于 AI 研究资助的咨询**：一位成员就申请和获得 **AI 研究** 资助的经验寻求建议。
  
  - 这反映了人们对 AI 领域创新项目资金支持的兴趣日益浓厚。
- **对算法演进的好奇**：一位用户对**算法如何演进**表示着迷，并注意到 AI 模型中出现了不同的个性。
  
  - *他们一直在突破界限*，并对模型如何与不同的输入进行交互感到好奇。
- **将 AI 拟人化的风险**：讨论指出，虽然 **LLMs** 可以产生类似人类的输出，但假设其具有意图可能会产生误导。
  
  - 成员们强调了将 AI 视为纯粹机器的重要性，并警告不要进行过度情感化的解读。
- **为伦理 AI 搭建桥梁**：人们对 AI 开发中需要 **伦理考量** 表示关注，以避免负面的未来后果。
  
  - 会议指出，那些创造智能 AI 的人有责任为其使用和应用建立准则。
- **通往 AGI 发展的阶段**：成员们参与了关于通往 **Artificial General Intelligence (AGI)** 历程以及算法所需改进的讨论。
  
  - 评论建议，在达到高级 AI 之前，预计会有多个迭代阶段，目前正在探索新的网络构想。

 

**提到的链接**：[Sam Altman (@sama) 的推文](https://x.com/sama/status/1849661093083480123)：@kyliebytes 假新闻已失控

 

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1300551158225702963) (5 messages):

> - `GPT Typo Issues`
> - `Voice Reading Problems on iOS`
> - `Chat Count Reduction`
> - `GPT Non-Responsive Behavior`
> - `Frustration with Model Responses`

- **GPT 拼写错误大量出现**：几位成员报告在与 ChatGPT 交互时遇到了**明显的拼写错误**和**语无伦次的词语**。
  
  - 成员们对输出质量的下降感到困惑，并想知道其他人是否也面临类似问题。
- **iOS 设备上的语音朗读问题**：一位用户提出了关于 iOS 上**语音朗读问题**的议题，寻求其他面临同样问题的人的确认。
  
  - 这一关注点反映了可能影响依赖该功能的 iOS 用户的可用性问题。
- **对话计数减少背后的谜团**：一位用户对注意到其 GPT 模型的**对话计数减少**表示困惑。
  
  - 这一观察表明用户对平台的追踪和响应功能持续关注。
- **模型拒绝处理电子商务请求的问题**：一位用户报告说，completions API 中的 **4o 模型** 拒绝为大约一半的电子商务描述请求生成响应。
  
  - 尽管不包含争议性话题，但模型意外的拒绝导致了极大的挫败感。
- **对模型响应的普遍沮丧**：成员们分享了对各种模型交互的潜在**沮丧**，暗示可靠性有所下降。
  
  - 担忧范围从拒绝生成合适的内容到似乎未解决的技术问题。

 

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/) (1 messages):

darthgustav.: 随机性 (Stochasticity)。

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/) (1 条消息):

darthgustav.: Stochasticity (随机性)。

---

### **Cohere ▷ #**[**discussions**](https://discord.com/channels/954421988141711382/954421988783444043/1300536088078585909) (35 条消息🔥):

> - `Algorithmic Trading Insights` (算法交易见解)
> - `Bias in News Articles` (新闻文章中的偏见)
> - `Garbled Output in AI Models` (AI 模型中的乱码输出)
> - `Parameter Adjustments for AI Responses` (AI 响应的参数调整)
> - `EDGAR and Market Insights` (EDGAR 与市场见解)

- **算法交易：经验教训**：一位拥有 4 年算法交易经验的成员分享了关于市场交互复杂性的见解，指出**粗放的过程有助于抵御负面影响**。
  
  - 他们强调，要了解哪些方法行不通，需要进行大量的模拟交易和研究。
- **理解 AI 情感分析中的媒体偏见**：在关于 AI 情感分析的讨论中，成员们一致认为所有媒体都存在偏见，而识别谁从这种偏见中获益对于准确评估至关重要。
  
  - 一位成员提到，他们构建了一个模型，在假设所有媒体都存在偏见的前提下开始调查。
- **AI 输出乱码导致困惑**：成员们报告在 AI 模型输出中看到了奇怪的乱码文本，引发了对其可用性影响的担忧。
  
  - 建议降低 temperature 和 top-p 参数作为潜在的修复方案，并建议通过实验找到有效的设置。
- **通过参数调优改进 AI 响应**：一位成员分享了调整 temperature 和 top-p 参数的经验，指出当设置为 Temp .6、Top-P .9，或者理想情况下设置为 Temp 0、Top-K 1 时，问题显著减少。
  
  - 他们表示，默认的 API 设置在解决观察到的问题方面可能效果不佳。
- **EDGAR 在市场分析中的可靠性**：讨论强调了使用 EDGAR 等资源来理解市场波动的关键性，并指出许多交易决策是由该系统中注册的数据驱动的。
  
  - 成员们对社交媒体对市场情绪的影响表示怀疑，认为自动化交易数据具有更高的权重。

---

### **Cohere ▷ #**[**questions**](https://discord.com/channels/954421988141711382/1168411509542637578/1300536570708889681) (14 条消息🔥):

> - `Serverless 模型使用`
> - `Max Tokens 的重要性`
> - `响应长度见解`
> - `Cohere 模型平台`
> - `分类使用场景`

- **Serverless 模型方案讨论**：用户询问模型是否需要下载，还是可以以 Serverless 方式使用，从而引出了关于响应处理和 Token 限制的见解。
  
  - 成员们讨论了设置特定参数对于优化模型性能和响应定制的重要性。
- **理解 Max Tokens**：澄清了除非指定了 `max_tokens`，否则唯一的严格限制是 Context Window；如果省略该参数，可能会抛出错误。
  
  - 用户被引导至 [Cohere 文档](https://docs.cohere.com/reference/chat#request.body.max_tokens) 以了解 Token 参数和用法的详细信息。
- **关于响应长度的见解**：响应通常在模型根据结构化系统提示词达到其自然终点（EOS Token）时停止，强调了输出的个性化。
  
  - 一位成员指出，他们的响应通常在 **3,000-4,000** 个字符之间，强调了细节程度如何影响输出长度。
- **Cohere 模型可用性**：Cohere 的模型可以通过多种平台访问，例如 [Amazon SageMaker](https://aws.amazon.com/marketplace/seller-profile?id=87af0c85-6cf9-4ed8-bee0-b40ce65167e0) 和 [Microsoft Azure](https://ai.azure.com/explore/models/?tid=694fed05-7f6d-4ab2-8c38-9afb438eab6f&selectedCollection=cohere)。
  
  - 鼓励用户熟悉各种 [Cohere 文档](https://docs.cohere.com/docs/models) 以了解模型能力。
- **分类模型使用场景**：讨论强调 **Classify** 端点需要示例以实现最佳预测，开始时每个类别至少提供两个示例是可接受的。
  
  - 参与者认可了该模型在处理不同输入长度方面的灵活性，以及微调对分类任务的影响。

**提到的链接**：

- [模型概览 — Cohere](https://docs.cohere.com/docs/models)：Cohere 拥有多种模型，涵盖了许多不同的使用场景。如果你需要更多定制化，可以训练模型以使其适应你的特定使用场景。
- [Chat — Cohere](https://docs.cohere.com/reference/chat#request.body.max_tokens)：生成对用户消息的文本响应，并逐个 Token 进行流式传输。要了解如何使用带有流式传输的 Chat API，请参考我们的 [文本生成指南](https://docs.cohere.com/v2/docs/cha...
- [Classify — Cohere](https://docs.cohere.com/reference/classify)：该端点对哪个标签最适合指定的文本输入进行预测。为了进行预测，Classify 使用提供的文本 + 标签对的 `examples` 作为参考。注意：[Fine-tu...

---

### **Cohere ▷ #**[**api-discussions**](https://discord.com/channels/954421988141711382/1168578329423642786/1300813473927925771) (7 条消息):

> - `报告幻觉数据`
> - `Cohere Rerank API 超时问题`

- **报告幻觉数据的流程**：一位用户询问在使用 R+08 2024 API 时如何报告 **幻觉数据或情况**，并举出了一个出现错误历史引用的具体实例。
  
  - *xvarunx* 确认了这一问题，并提到在使用 Coral 网页版时，可以通过点赞或点踩选项来收集反馈。
- **Cohere Rerank API 超时困扰**：一位成员提出了一个关于 **Cohere Rerank API** 请求偶尔卡住并超时的问题，无论使用 SDK 还是标准 API 调用都会出现。
  
  - 他们寻求关于如何联系支持部门以调查 **超时问题** 的指导。

---

### **Cohere ▷ #**[**projects**](https://discord.com/channels/954421988141711382/1218409701339828245/1300813682363858985) (1 条消息):

> - `Synthetic Data Generation` (合成数据生成)
> - `Medical Note Automation` (医疗笔记自动化)

- **使用 LLM 生成医疗笔记**：展示了一个使用 LLM 生成合成医疗笔记的 Demo，允许用户通过极少的输入创建详细的笔记。
  
  - 点击[此处查看 Demo](https://demo.talcapi.com/demo/meddoc?id=72a7fe2b-a2c9-4542-9063-af8093331ba9) 以了解该工具的多功能性。
- **医疗笔记创建的多功能性**：该工具证明了仅需几个描述性词汇，用户即可快速生成复杂的医疗文档。
  
  - 这一创新突显了利用 [LLM 进行文档记录](https://demo.talcapi.com/demo/meddoc?id=72a7fe2b-a2c9-4542-9063-af8093331ba9)来提高医疗实践效率的潜力。

**提到的链接**：[pathology report](https://demo.talcapi.com/demo/meddoc?id=72a7fe2b-a2c9-4542-9063-af8093331ba9)：患者接受活检或手术程序，并收集组织样本。样本被送往病理实验室，病理学家在显微镜下进行检查。病理学家编写一份...

---

### **Cohere ▷ #**[**cohere-toolkit**](https://discord.com/channels/954421988141711382/1254901651081269268/1300806299533180999) (2 条消息):

> - `Cohere Installation Issues` (Cohere 安装问题)
> - `Tokenizers Compatibility` (Tokenizers 兼容性)

- **使用 Poetry 安装 Cohere 失败**：一位用户报告了使用 Poetry 安装 **Cohere** 时遇到的困难，特别提到它提示 **tokenizer** 相关问题。
  
  - 引用的错误与 **tokenizers (0.20.1)** 不支持 PEP 517 构建有关，这可能表明存在兼容性问题。
- **Tokenizers 不支持 PEP 517**：另一条消息澄清该错误源自 **build backend**（构建后端），可能指向 **tokenizers** 本身的问题，而非 Poetry。
  
  - 这表明用户可能需要检查更新或寻找 tokenizers 的替代版本来解决安装问题。

---

### **OpenRouter (Alex Atallah) ▷ #**[**announcements**](https://discord.com/channels/1091220969173028894/1092729520181739581/1300540945682399253) (1 条消息):

> - `Inflection`
> - `Billing Issues` (账单问题)

- **Inflection 已恢复在线**：上周的**账单问题**已修复，Inflection 现已恢复运行。
  
  - 欲了解更多详情，请查看 [Inflection 3 PI](https://openrouter.ai/inflection/inflection-3-pi) 和 [Inflection 3 Productivity](https://openrouter.ai/inflection/inflection-3-productivity) 的链接。
- **Inflection 服务已恢复**：在解决之前的**账单问题**后，Inflection 的服务已向用户全面恢复。
  
  - 此次更新标志着运营回归正常，提升了所有用户的生产力。

**提到的链接**：[Inflection 3 Productivity - API, Providers, Stats](https://openrouter.ai/inflection/inflection-3-productivity)：Inflection 3 Productivity 针对遵循指令进行了优化。它更适合需要 JSON 输出或精确遵守所提供指南的任务。通过 API 运行 Inflection 3 Productivity。

---

### **OpenRouter (Alex Atallah) ▷ #**[**app-showcase**](https://discord.com/channels/1091220969173028894/1092850552192368710/1300916364285841488) (1 条消息):

> - `Flexible chat app for macOS` (适用于 macOS 的灵活聊天应用)
> - `Alpha testers recruitment` (Alpha 测试人员招募)

- **为 macOS 聊天应用寻找 Alpha 测试人员**：一位开发者正在为其专为 macOS 设计的新型**灵活聊天应用**寻找 **alpha 测试人员**，并分享了[截图](https://imgur.com/a/HI5Py3A)以展示当前进度。
  
  - 随着项目达到这一关键里程碑，*如果感兴趣参与测试阶段请私信 (DM)*。
- **截图展示令人兴奋**：分享的[截图](https://imgur.com/a/HI5Py3A)展示了即将推出的聊天应用的各种功能和用户界面设计。
  
  - 开发者欢迎对设计和功能提出反馈，并寻求感兴趣用户的全面测试。

**提到的链接**：[imgur.com](https://imgur.com/a/HI5Py3A)：在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过有趣的笑话、热门迷因、娱乐 GIF、感人故事、病毒视频等来振奋你的精神...

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1300542742069379103) (46 messages🔥):

> - `OpenRouter API 问题`
> - `API Key 安全性`
> - `服务中断`
> - `活动日志记录`
> - `用量追踪工具`

- **OpenRouter API 响应深受问题困扰**：用户报告了持续的 **524 错误**，导致各种模型的请求停滞，引发了在公开上线前对稳定性的担忧。
  
  - 一位用户表示，由于影响多个请求的反复减速问题，他们可能需要考虑更换供应商。
- **对 API Key 安全性的担忧**：讨论了 API Key 可能被爬取的问题，并建议在漏洞利用场景中，未经授权的代理可能会使用 **Claude 3.5 Sonnet** 等模型。
  
  - 用户强调了保护 Key 安全的重要性，但也有人质疑，尽管采取了安全措施，漏洞仍可能导致泄露。
- **活动日志查询**：一位用户询问如何通过编程方式获取其活动，得到的指导是目前仅 **/generations endpoint** 可用。
  
  - 进一步讨论强调了缺乏全面的日志记录能力，以及在不使用 OpenRouter UI 的情况下追踪所有活动的有效性不足。
- **使用外部工具追踪用量**：鼓励成员利用 **Helicone** 等工具来有效追踪用量并管理 API 活动。
  
  - 鉴于对非用户本人发起的异常活动激增的担忧，这一建议被提出。
- **敏感信息泄露的影响**：对话转向了在 LLM 交互过程中泄露 **PII (个人身份信息)** 或敏感数据的风险。
  
  - 一位成员分享了一个个人案例，讲述了因不小心将终端命令粘贴到 LLM 中而暴露姓名的经历，说明了数据暴露的可能性。

**相关链接**：

- [Activity | OpenRouter](https://openrouter.ai/activity)：查看你在 OpenRouter 上使用模型的情况。
- [OpenRouter Integration - Helicone OSS LLM Observability](https://docs.helicone.ai/getting-started/integration-method/openrouter)：未找到描述

---

### **OpenRouter (Alex Atallah) ▷ #**[**beta-feedback**](https://discord.com/channels/1091220969173028894/1277894087755829278/1300539459380580352) (8 messages🔥):

> - `集成访问权限`
> - `Beta 访问请求`

- **多个关于集成访问权限的请求**：多位成员表达了获取 **integrations** 访问权限的愿望，使用了诸如“我想获得访问权限”和“恳请授予我访问权限”等措辞。
  
  - *预谢* 是一种普遍的情绪，强调了各方提出的礼貌请求。
- **学生研究员寻求 Beta 访问权限**：一位自称是**学生研究员**的成员专门请求了 **beta** 访问权限，表明了对该项目的潜在学术兴趣。
  
  - 该请求是众多关注集成的类似访问咨询之一。

 

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1300536510927470657) (48 messages🔥):

> - `Moondream 融资`
> - `AI 驱动的搜索引擎`
> - `GitHub Copilot 更新`
> - `向量数据库`
> - `OpenAI 聊天功能`

- **Moondream 获得 450 万美元融资**：[Moondream](https://venturebeat.com/ai/moondream-raises-4-5m-to-prove-that-smaller-ai-models-can-still-pack-a-punch/) 筹集了 450 万美元，以证明较小的 AI 模型在网络爬虫活跃数月的情况下依然有效。
  
  - 成员们讨论了潜在的局限性以及在行业中采用较小模型的意义。
- **Meta 开发 AI 搜索引擎**：据报道，Meta 正在开发自己的 [AI 驱动的搜索引擎](https://www.theverge.com/2024/10/28/24282017/meta-ai-powered-search-engine-report)，以减少对 Google 和 Microsoft 的依赖。
  
  - 网络爬虫已经活跃了数月，暗示了组织内部在增强搜索能力方面的重大转变。
- **GitHub Copilot 新增 Gemini 和 Claude 模型**：GitHub 宣布引入 [Gemini 模型](https://cloud.google.com/blog/products/ai-machine-learning/gemini-models-on-github-copilot) 和 Claude，增强了其 Copilot 的能力。
  
  - Microsoft 和 Google 之间的合作标志着 AI 发展中出人意料的伙伴关系，加强了面向开发者的多模型方法。
- **对向量数据库的批评**：@avthars 认为现有的向量数据库缺乏适当的抽象，提议将 [pgai Vectorizer](https://github.com/timescale/pgai) 作为 Embedding 管理的更高效解决方案。

- 该工具简化了 Embedding 的同步和维护，这对于提高 AI 模型性能至关重要。
- **ChatGPT 新功能：聊天记录搜索**：OpenAI 正在推出一项新功能，允许用户在 ChatGPT 网页版上搜索聊天记录，从而轻松访问过去的对话。
  
  - 成员们对这一期待已久的更新表示欣慰和兴奋，强调了它对持续讨论带来的便利。

**提到的链接**：

- [Sam Altman (@sama) 的推文](https://x.com/sama/status/1849661093083480123)：@kyliebytes 虚假新闻失控了
- [Samuel Hammond 🌐🏛 (@hamandcheese) 的推文](https://x.com/hamandcheese/status/1850394704862380450)：我对 2023 年 AI 未来“默认路径”的预测。2024-2027：
- [VentureBeat (@VentureBeat) 的推文](https://x.com/VentureBeat/status/1850885273749532852)：Moondream 融资 450 万美元，证明小型 AI 模型依然实力强劲 https://venturebeat.com/ai/moondream-raises-4-5m-to-prove-that-smaller-ai-models-can-still-pack-a-punch/
- [OpenAI (@OpenAI) 的推文](https://x.com/openai/status/1851340615344406781?s=46)：我们开始在 ChatGPT 网页版推出搜索聊天记录的功能。现在你可以快速轻松地调出对话进行参考，或从上次中断的地方继续聊天。
- [据报道 Meta 也在开发自己的 AI 驱动搜索引擎](https://www.theverge.com/2024/10/28/24282017/meta-ai-powered-search-engine-report)：据报道，Meta 希望减少对 Google 的依赖。
- [GitHub Next | GitHub Spark](https://githubnext.com/projects/github-spark)：GitHub Next 项目：我们能否让任何人都能利用 AI 和全托管运行时，为自己创建或适配软件？
- [GitHub Copilot 上的 Gemini 模型 | Google Cloud 博客](https://cloud.google.com/blog/products/ai-machine-learning/gemini-models-on-github-copilot)：通过与 Google Cloud 的新合作伙伴关系，GitHub 很快将提供 Gemini 1.5 Pro。
- [Artificial Analysis (@ArtificialAnlys) 的推文](https://x.com/ArtificialAnlys/status/1850587843837771900)：red_panda 是什么？👀 在 Artificial Analysis Image Arena 中查看 red_panda。链接见下方推文 ⬇️
- [sMyle (@MylesBorins) 的推文](https://x.com/mylesborins/status/1851317503256858945?s=46)：非常高兴看到 @github 团队发布了“使用 Copilot Workspace 优化和验证代码审查建议”。这是我离开 GitHub 前负责的最后一项重大工作...
- [Alex Albert (@alexalbert__) 的推文](https://x.com/alexalbert__/status/1851300048711365021)：很高兴宣布 Claude 现已在 GitHub Copilot 上可用。从今天开始，开发者可以在 VS Code 和 GitHub 中选择 Claude 3.5 Sonnet。访问权限将逐步向所有 Copilot Chat 用户开放...
- [Avthar (@avthars) 的推文](https://x.com/avthars/status/1851252850619277358)：向量数据库是错误的抽象。这里有一个更好的方法：介绍 pgai Vectorizer，这是一个新的开源 PostgreSQL 工具，它可以自动创建 Embedding 并与源数据同步，就像...
- [Perplexity (@perplexity_ai) 的推文](https://x.com/perplexity_ai/status/1851315707411337435)：我们很高兴能与 @github 合作。通过我们的 GitHub Copilot 集成，你将能够：• 及时了解最新的库更新，例如“React 的最新更新” • 快速找到答案...
- [我们是否正处于自我改进 AI 爆发的边缘？](https://arstechnica.com/ai/2024/10/the-quest-to-use-ai-to-build-better-ai/)：一个能制造出更好 AI 的 AI 可能是“人类需要做出的最后一项发明”……
- [vik (@vikhyatk) 的推文](https://x.com/vikhyatk/status/1850990119937064971?s=46)：我创办了一家公司……引用 VentureBeat (@VentureBeat) 的话：Moondream 融资 450 万美元，证明小型 AI 模型依然实力强劲 https://venturebeat.com/ai/moondream-raises-4-5m-to-prove-tha...
- [Paul Klein IV (@pk_iv) 的推文](https://x.com/pk_iv/status/1851270308701106383?s=46)：下一家估值十亿美元的公司将由 Browserbase 驱动。我们已经帮助数百家 AI 初创公司实现了大规模的网络自动化。现在，我们完成了 2100 万美元的 A 轮融资，由 Klein 领投...

---

### **Modular (Mojo 🔥) ▷ #**[**general**](https://discord.com/channels/1087530497313357884/1098713601386233997/1300734213221388339) (5 条消息):

> - `Modular 产品讨论`
> - `通用软件讨论`
> - `社区等级晋升`

- **Modular 频道焦点明确**：<@rcdpge> 询问 <#1098713601386233997> 频道是仅限 **Modular 产品**，还是欢迎更广泛的技术讨论。
  
  - Melodyogonna 回复称，该频道专门用于 Modular 的产品，并建议在 <#1104620458168553563> 进行其他讨论。
- **软件在社区中的相关性**：<@rcdpge> 认为使用 **Python** 等语言开发的 **软件** 可能与社区相关，尽管尚不确定。
  
  - 这一评论反映了人们对 Modular 产品范围之外的多样化技术讨论的持续兴趣。
- **社区参与认可**：<@ModularBot> 祝贺 <@774658006649929778> 在社区中晋升至 **5 级**。
  
  - 这突显了社区结构内持续的参与和成就。

---

### **Modular (Mojo 🔥) ▷ #**[**mojo**](https://discord.com/channels/1087530497313357884/1151418092052815884/1300717926185435146) (43 条消息🔥):

> - `Mojo 内存安全引用提案`
> - `FlatBuffers 与 ProtoBuf 对比`
> - `交换引用实现`
> - `优化与性能关注点`
> - `Mojo 中的 Alias 与 noalias`

- **关于 Mojo 内存安全引用的重大提案**：一名成员发布了一份 [重大提案](https://gist.github.com/nmsmith/cdaa94aa74e8e0611221e65db8e41f7b)，重新思考了 Mojo 中内存安全引用的工作方式，旨在简化引用而不牺牲安全性。
  
  - 在经过广泛的私下开发后，该提案正征求 Mojo 社区的反馈，强调了在确保内存安全的同时需要优化灵活性。
- **FlatBuffers 与 ProtoBuf 概览**：FlatBuffers 和 ProtoBuf 均由 Google 开发，用途不同。FlatBuffers 允许零解析以实现高效的数据提取，而 ProtoBuf 则专注于位打包（bit packing）。
  
  - 由于 Modular 团队预计将 ProtoBuf 用于 Serving，因此分享了一个 [Swift ProtoBuf 支持示例](https://github.com/apple/swift-protobuf)，作为潜在插件开发的参考。
- **Mojo 中交换引用的挑战**：成员们讨论了在 Mojo 中实现交换操作且不产生别名（aliasing）问题的含义，并将其与 Rust 对可变引用的处理进行了对比。
  
  - 成员们对放宽可变性限制所引入的复杂性表示担忧，这可能会影响性能和编译时间。
- **优化与性能关注点**：讨论强调了 `noalias` 对性能的重要性，成员们主张默认情况下应优先考虑非别名化，以实现最佳的编译时效率。
  
  - 强调了需要一个支持唯一引用（unique references）的模型，否则可能导致性能下降，这让人联想到 Rust 过去的提案。
- **保留对话的必要性**：成员们表达了拥有一个扩展讨论平台的价值，而不是仅仅依靠博客文章进行交流。
  
  - 小组认识到记录 Mojo 实现等复杂话题的挑战，强调更好的协调可以帮助保留宝贵的见解。

**提及的链接**：

- [n’s gists](https://gist.github.com/n)：GitHub Gist：通过在 GitHub 上创建账户来关注并 fork n 的 gists。
- [GitHub - apple/swift-protobuf: Plugin and runtime library for using protobuf with Swift](https://github.com/apple/swift-protobuf)：在 Swift 中使用 protobuf 的插件和运行时库 - apple/swift-protobuf

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1300623199163125800) (6 条消息):

> - `Hugging Face talk`
> - `Open Lean datasets`
> - `GPT-NeoX on Colab`
> - `Local completion with self-signed certificates`
> - `NeurIPS registration`

- **Hugging Face CEO 即将举行的演讲**：Hugging Face 的联合创始人兼 CEO Clem 计划进行一场演讲，引起了社区的期待。
  
  - 目前尚未提及演讲的具体细节。
- **关于开源 Lean 数据集的咨询**：有成员询问是否存在类似于可能用于训练 **AlphaProof** 的开源 Lean 数据集。
  
  - 讨论反映了人们对用于模型训练的数据集可用性的持续关注。
- **在 Colab 上成功运行 GPT-NeoX**：确认 **GPT-NeoX** 可以在 Colab 上运行，并分享了一个 [Colab notebook](https://github.com/markNZed/GPT-NeoX-Colab/blob/main/notebooks/shakespeare_training.ipynb) 链接作为参考。
  
  - 演示的模型规模较小，约为 **5M parameters**，表明了该方法的可行性。
- **在本地补全中处理自签名证书**：一位成员寻求关于在使用通过 `base_url` 连接远程模型的 `local_completion` 时，如何管理自签名证书的指导。
  
  - 另一位成员提供了 [相关回答](https://discord.com/channels/729741769192767510/755950983669874798/1300852659611238420) 的链接，表明已有解决方案。
- **NeurIPS 审稿人的免费注册**：一位成员询问今年是否有 **NeurIPS** 审稿人收到了免费注册名额，并指出注册页面显示了符合资格。
  
  - 他们建议其他人在 **11 月 1 日** 截止日期前核实自己的状态。

 

**提及的链接**：[GPT-NeoX-Colab/notebooks/shakespeare_training.ipynb at main · markNZed/GPT-NeoX-Colab](https://github.com/markNZed/GPT-NeoX-Colab/blob/main/notebooks/shakespeare_training.ipynb)：GPT-NeoX 的示例 Colab notebooks。通过在 GitHub 上创建账号来为 markNZed/GPT-NeoX-Colab 的开发做出贡献。

 

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1300584183080554641) (17 条消息🔥):

> - `Hellaswag 训练结果`
> - `Transformers 中的参数共享`
> - `FP8 训练效率`
> - `优化中的模块化对偶化`
> - `侧重类型检查的优化理论`

- **Hellaswag 训练达到 GPT-2 同等水平**：一位成员强调，使用全新的 **8xH100** 服务器，在 NanoGPT 竞速（speedrunning）中仅用 **7.3 小时**，花费不到 **$200**，就在 Hellaswag 上达到了 **GPT-2 (1.5B)** 级别的性能。
  
  - 此前的记录是 **24 个 8xH100-小时**，这表明效率有了显著提升。
- **用于参数共享的递归 Transformer**：**Recursive Transformers** 的引入为 Transformer 中的参数共享提供了一种新方法，在不显著损失性能的情况下减少了模型大小和成本。
  
  - 该方法利用了 **层绑定（layer tying）** 等概念，并引入了使用深度维度低秩自适应（LoRA）的 **Relaxed Recursive Transformers** 以增强性能。
- **COAT 框架提升 FP8 训练**：提出了一种新的 FP8 训练框架 **COAT**，通过动态范围扩展和混合粒度量化策略处理优化器状态和激活值，从而优化内存使用。
  
  - 该框架有可能显著减少训练大模型时的内存占用，性能优于以往的方法。
- **模块化对偶化变革优化领域**：最近的一篇论文提出了 **模块化对偶化（modular dualization）**，通过将梯度映射到通用神经网络的对偶空间，为有效的训练算法提供了理论基础。
  
  - 该方法为各种神经网络层带来了 GPU 友好的算法，提高了训练效率和可扩展性。
- **对优化理论的兴趣日益增长**：一位成员对专注于优化理论的论文表示热烈欢迎，特别赞赏在讨论的工作中加入了 **类型检查（type check）**。
  
  - 这引发了社区内对以类型而非集合为中心的数学教科书的共同渴望，凸显了这一小众兴趣。

**提到的链接**：

- [Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA](https://arxiv.org/abs/2410.20672)：大语言模型（LLMs）的部署成本高昂。参数共享为减小其规模和成本提供了一条可能的路径，但其在现代 LLM 中的有效性仍然相当有限。在本文中……
- [Modular Duality in Deep Learning](https://arxiv.org/abs/2410.21265)：优化理论中的一个古老观点认为，由于梯度是一个对偶向量，在将其映射到权重所在的原始空间之前，不能直接从权重中减去它。我们……
- [Rephrasing natural text data with different languages and quality levels for Large Language Model pre-training](https://arxiv.org/abs/2410.20796)：最近发表的关于为 LLM 预训练改写自然文本数据的工作表明，将原始数据集与合成改写的数据相结合时，效果显著。我们基于之前的工作……
- [COAT: Compressing Optimizer states and Activation for Memory-Efficient FP8 Training](https://arxiv.org/abs/2410.19313)：FP8 训练已成为提高训练效率的一种极具前景的方法。现有框架通过将 FP8 计算应用于线性层来加速训练，而将优化器状态和……
- [Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning](https://arxiv.org/abs/2410.14157)：自回归语言模型尽管能力惊人，但在复杂的推理和长期规划任务中表现挣扎。我们引入离散扩散模型作为一种新颖的解决方案……
- [Keller Jordan (@kellerjordan0) 的推文](https://x.com/kellerjordan0/status/1850995958697308307)：这是 NanoGPT 竞速的一个新结果：直接扩展竞速规模，在 8xH100 上仅需 7.3 小时即可达到 GPT-2 (1.5B) 的性能水平。之前的……

---

### **Eleuther ▷ #**[**interpretability-general**](https://discord.com/channels/729741769192767510/1052314805576400977/1300861632003702845) (1 条消息):

> - `Sparse Autoencoder Guides`
> - `Mechanistic Interpretability Series`

- **首个 Sparse Autoencoder 指南发布**：一位成员宣布发布了一份关于如何使用 **Premade Sparse Autoencoder** 寻找特征的[逐步指南](https://beta.ai-plans.com/guide/g7yjq98bhuyhkza)。
  
  - 该指南标志着一个专注于 **Mechanistic Interpretability** 的新系列的开始。
- **即将推出的 Mechanistic Interpretability 系列**：提到的指南旨在成为更广泛努力的一部分，目标是分享关于 **Mechanistic Interpretability** 技术的全面知识。
  
  - 鼓励社区关注更多将补充这一基础资源的后续指南。

 

**提到的链接**：[AI-Plans](https://beta.ai-plans.com/guide/g7yjq98bhuyhkza)：未找到描述

 

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1300852659611238420) (2 条消息):

> - `Custom Certificates`
> - `Workaround for Certificates`

- **Custom Certificates 缺乏支持**：一位成员指出，平台目前尚未真正支持 **custom certificates**。
  
  - 不过，他们提到了最近针对此问题发布的一个 [workaround](https://github.com/EleutherAI/lm-evaluation-harness/issues/2436)。
- **关于 Workaround 分享的讨论**：成员们参与了解决方案的分享，强调了 **workaround** 如何帮助应对当前功能的局限性。
  
  - 这突显了社区协作在寻找实际解决方案中的重要性。

 

**提到的链接**：[Issues · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/issues/2436)：一个用于语言模型 few-shot 评估的框架。- Issues · EleutherAI/lm-evaluation-harness

 

---

### **Interconnects (Nathan Lambert) ▷ #**[**news**](https://discord.com/channels/1179127597926469703/1179128538679488533/1300564289857519638) (13 条消息🔥):

> - `OpenAI CFO 见解`
> - `SearchGPT 推广`
> - `ROCKET-1 开发`
> - `Anthropic 招聘潮`
> - `GitHub Copilot 中的 Claude`

- **OpenAI CFO 宣布 AI 已成为主流**：在一段 [YouTube 视频](https://youtu.be/eCqFgVqWbEs)中，OpenAI CFO Sarah Friar 强调 **AI 不再是实验性的**，因为银行和金融科技公司每天都在使用它。
  
  - 这一重大转变为各行各业的广泛落地提供了更多机会。
- **SearchGPT 扩展程序发布**：预计 OpenAI 将推广其新的 Chrome 扩展程序，允许用户在 **SearchGPT** 发布的同时将其设置为默认搜索引擎。
  
  - 用户可以通过浏览器地址栏使用命令快速发起搜索，并根据需要重定向到 Google。
- **ROCKET-1 介绍**：**ROCKET-1** 旨在通过利用视觉-时间上下文提示（visual-temporal context prompting）来增强 Minecraft 中的创意任务，由 [Team CraftJarvis](https://craftjarvis.github.io/) 展示。
  
  - 这一进展突显了视觉语言模型（vision-language models）在开放世界应用中不断进化的能力。
- **Anthropic 的招聘势头**：Anthropic 因其强劲的招聘实践而受到关注，并宣布了新团队成员的加入。
  
  - 最近的招聘热潮反映了该公司在 AI 领域的蓬勃发展和雄心壮志。
- **Claude 与 GitHub Copilot 的集成**：@AnthropicAI 宣布 **Claude 3.5 Sonnet** 现已面向在 Visual Studio Code 中使用 GitHub Copilot 的开发者开放，本周开始推出。
  
  - 这种集成预计将通过在流行的开发工具中直接提供先进的 AI 支持来增强编程体验。

**提到的链接**：

- [Joining Anthropic](https://www.furidamu.org/blog/2024/10/28/joining-anthropic/)：未找到描述
- [SOCIAL MEDIA TITLE TAG](https://craftjarvis.github.io/ROCKET-1/)：SOCIAL MEDIA DESCRIPTION TAG TAG
- [来自 TestingCatalog News 🗞 (@testingcatalog) 的推文](https://x.com/testingcatalog/status/1851017181326152027)：OpenAI 可能会在发布 SearchGPT 的同时，推广其用于将 ChatGPT 设置为默认搜索的 Chrome 扩展程序。该扩展程序此前曾出现在独立版 SearchGPT 中，目前仍然...
- [来自 Anthropic (@AnthropicAI) 的推文](https://x.com/anthropicai/status/1851297754980761605?s=46)：Claude 现已登录 @GitHub Copilot。从今天开始，开发者可以在 Visual Studio Code 和 http://GitHub.com 中选择 Claude 3.5 Sonnet。访问权限将推广给所有 Copilot Chat 用户和组织...
- [SearchGPT - Chrome 网上应用店](https://chromewebstore.google.com/detail/searchgpt/ejcfepkfckglbgocfkanmcdngdijcgld?pli=1)：将默认搜索引擎更改为 SearchGPT。
- [OpenAI CFO 表示 AI 不再是实验性的](https://youtu.be/eCqFgVqWbEs)：OpenAI CFO Sarah Friar 表示人工智能不再是实验性的。她说银行、金融机构和金融科技公司每天都在使用它...

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-questions**](https://discord.com/channels/1179127597926469703/1179208129083363358/1300559769211371600) (2 条消息):

> - `人工标注定价`
> - `领域溯源难度`

- **询问人工标注成本**：一位成员询问是否有人知道在哪里可以找到关于“雇人生成示例”与“仅标注好坏”的定价对比。
  
  - 这个问题突显了对以人为中心的任务中**定价结构**明确化的需求。
- **成本取决于领域难度**：另一位成员回答说，成本在很大程度上取决于**领域溯源难度（domain sourcing difficulty）**乘以**问题长度**。
  
  - 这表明复杂性直接影响标注服务的预算编制。

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-drama**](https://discord.com/channels/1179127597926469703/1181746144821387334/1300889807492349972) (7 messages):

> - `NeurIPS 注册机制变更`
> - `对抽签系统的担忧`
> - `研究生受到影响`

- **NeurIPS 注册转向抽签系统**：为了应对极高的需求，**NeurIPS** 将立即实施**随机抽签系统**进行注册，正如 [NeurIPS Twitter 账号](https://fxtwitter.com/NeurIPSConf/status/1851325157870068166)所述。被接收论文的作者被敦促尽快注册以确保名额，但仍可能受到抽签的影响。
  
  - 许多参与者表示怀疑，指出过去的经验表明这可能会导致混乱，有人预测这将是一场*彻底的灾难 (total shitshow)*。
- **研究生可能在注册上面临困难**：有人担心**研究生作者**可能会因为新的抽签系统而延迟注册，从而影响他们的参会能力。这延续了疫情前观察到的模式，表明注册的可获得性一直存在问题。
  
  - 一位参与者提到，这些进展不断证明了他们不参加 **NeurIPS** 的决定是正确的。

**提到的链接**：[来自 NeurIPS Conference (@NeurIPSConf) 的推文](https://fxtwitter.com/NeurIPSConf/status/1851325157870068166)：由于注册需求极高，NeurIPS 将转向随机抽签系统，立即生效。被接收的会议和 Workshop 论文作者仍可保证注册...

---

### **Interconnects (Nathan Lambert) ▷ #**[**memes**](https://discord.com/channels/1179127597926469703/1187551504995987576/1300804981258977320) (2 messages):

> - `孙正义 (Masayoshi Son) 关于 ASI 的成本拆解`
> - `泡沫担忧`

- **孙正义 (Masayoshi Son) 透露 ASI 成本**：孙正义分享称，实现人工超级智能 (ASI) 将需要 **9 万亿美元**和 **2 亿颗芯片**，详情见此 [推文](https://x.com/sundeep/status/1851240494958829655)。
  
  - 这一声明引发了兴奋与担忧，标志着 AI 技术所需的投资**规模**。
- **泡沫破裂的讨论**：一位成员表示担心，目前围绕 AI 的炒作可能预示着**泡沫正在破裂**。
  
  - 评论反映出人们对近期 AI 进展和投资趋势可持续性的日益不安。

**提到的链接**：[来自 sunny madra (@sundeep) 的推文](https://x.com/sundeep/status/1851240494958829655)：孙正义刚刚拆解了实现 ASI 将如何耗资 9 万亿美元并需要 2 亿颗芯片 👀

---

### **OpenInterpreter ▷ #**[**general**](https://discord.com/channels/1146610656779440188/1147665339266650133/1300545962317709332) (16 messages🔥):

> - `Open Interpreter Model Capabilities`
> - `Local Model Limitations`
> - `Using the Computer API`
> - `Hackathon Winners Setup`
> - `Open Interpreter Community Support`

- **Open Interpreter 需要视觉模型来实现完整功能**：为了让 Open Interpreter 正常发挥视觉能力，通常需要一个**多模态模型 (multi-modal model)**，除非使用 **Moondream** 处理基础任务。
  
  - 一位用户指出，他们无法使用 **Llava** 等本地模型复制 **Sonnet** 或 **GPT-4o** 的功能。
- **本地模型执行操作面临挑战**：用户正努力让 **Llava** 等本地模型执行类似于云端模型的操作，例如截屏和移动鼠标。
  
  - 另一位用户强调，需要更清晰的设置指南才能有效地利用 **computer API**。
- **配置黑客松获胜者的工具**：一位用户寻求指导，希望将 **Toolkit、UI、Memory 和 Sourcerer** 等**黑客松获胜者**的工具集成到他们的 Open Interpreter 仓库中。
  
  - 他们表示对这些工具已有理解，但希望确保实现方式正确。
- **Open Interpreter 问题的社区支持**：多位用户正在寻求帮助，以配置他们的本地模型，从而利用 **Open Interpreter** 完成更高级的任务。
  
  - 另一位新加入社区的用户希望利用 Open Interpreter 来提升工作效率。
- **本地模型功能与云端模型存在差异**：在关于本地模型能力的讨论中，有人指出 **localos.py** 和 **os.py** 在功能上有显著差异。
  
  - 成员们对本地模型缺乏某些控制功能表示沮丧，这些功能本可以促进与 PC 的深度集成，尤其是在与云端服务对比时。

**相关链接**：

- [Introduction - Open Interpreter](https://docs.openinterpreter.com/settings/all-set)：无描述
- [open-interpreter/interpreter/core/computer/vision/vision.py at 36ec07125efec86594c91e990f68e0ab214e7edf · OpenInterpreter/open-interpreter](https://github.com/OpenInterpreter/open-interpreter/blob/36ec07125efec86594c91e990f68e0ab214e7edf/interpreter/core/computer/vision/vision.py#L22)：计算机的自然语言接口。通过在 GitHub 上创建账号为 OpenInterpreter/open-interpreter 开发做贡献。
- [All Settings - Open Interpreter](https://docs.openinterpreter.com/settings/all-settings#import-computer-api)：无描述

---

### **OpenInterpreter ▷ #**[**ai-content**](https://discord.com/channels/1146610656779440188/1149229778138824765/1300548330107174913) (5 messages):

> - `OpenAI Advanced Voice`
> - `Apple AI Server Hacking Bounty`
> - `Muvi V2M`
> - `ChatGPT Chat History Search`

- **OpenAI Advanced Voice 向免费用户开放**：@OpenAI 宣布，欧盟、瑞士、冰岛、挪威和列支敦士登的**免费用户**现在可以通过移动端 App 使用 **Advanced Voice** 功能。
  
  - *这标志着这些地区用户的可访问性得到了显著提升。*
- **Apple 为 AI 服务器黑客攻击提供 100 万美元奖金**：据 @CultureCrave 报道，**Apple** 将向任何能成功入侵其 **AI 服务器**的人支付高达 **100 万美元**的奖金。
  
  - *这一举措引发了对网络安全的关注，并引导人们对 Apple 的安全措施进行审查。*
- **Muvi V2M 引起关注**：一位成员引用了 [Muvi V2M](https://muvi-v2m.github.io) 网站，其展示的示例激发了大家的兴趣。
  
  - *反馈显示了兴奋和好奇，凸显了一些此前未知的资源。*
- **ChatGPT 推出聊天记录搜索功能**：@OpenAI 宣布开始在 **ChatGPT 网页版**推出搜索聊天记录的功能，方便用户查阅之前的对话。
  
  - *该功能旨在通过允许用户快速找回并继续之前的对话来增强易用性。*

**相关链接**：

- [MuVi](https://muvi-v2m.github.io)：无描述
- [Culture Crave 🎃 (@CultureCrave) 的推文](https://x.com/culturecrave/status/1850781293166067999?s=46&t=G6jp7iOBtkVuyhaYmaDb0w)：Apple 将向任何能入侵其 AI 服务器的人支付高达 100 万美元。
- [OpenAI (@OpenAI) 的推文](https://x.com/openai/status/1850989317537349927?s=46&t=G6jp7iOBtkVuyhaYmaDb0w)：最后，Advanced Voice 现在已向欧盟、瑞士、冰岛、挪威和列支敦士登的移动端 App 免费用户开放。
- [OpenAI (@OpenAI) 的推文](https://fxtwitter.com/openai/status/1851340615344406781?s=46&t=G6jp7iOBtkVuyhaYmaDb0w)：我们开始在 ChatGPT 网页版推出搜索聊天记录的功能。现在你可以快速轻松地调出对话进行参考，或从上次中断的地方继续聊。

---

### **Torchtune ▷ #**[**dev**](https://discord.com/channels/1216353675241590815/1236040539409879170/1300757102133776404) (19 条消息🔥):

> - `基础模型量化`
> - `FSDP CPU Offloading`
> - `量化 KV-Caches`
> - `非可训练模型量化`

- **探索不带 LoRA 的量化**：成员们正在讨论基础模型是否可以像 QLoRA 一样在不应用 LoRA 的情况下进行量化，并指出在配置非 LoRA 模型构建器（model builders）方面可能存在的挑战。
  
  - *嗯，我想主要问题是我们没有办法在非 LoRA 模型构建器中配置这一点。*
- **FSDP 和 CPU Offloading 能力**：有人指出，FSDP 目前为 CPU offloading 提供了一个单一参数，涵盖了参数、梯度和优化器状态，目前尚无更精细的控制手段。
  
  - *这种方法有更多的数据移动，但由于优化器步骤在 GPU 上，速度可能更快*，这是针对性能提出的一个考量。
- **讨论量化 KV-Caches 的使用**：考虑到这种类型的缓存会消耗大量内存（尤其是在大型模型中），人们对使用 NF4 张量的量化 KV-caches 的效用持怀疑态度。
  
  - *我不认为 torchao 中的量化 kv cache 目前有多大用处或多强大*，这表明了对其有效性的怀疑。
- **冻结模型权重的挑战**：对话还强调，对模型的非可训练部分进行量化（例如 PPO 期间的冻结权重）有助于减少内存使用。
  
  - 一位参与者表达了兴趣，指出：*是的，我也想做类似的事情，在 PPO 期间对非可训练模型进行量化。*
- **关于 8 位以下量化的担忧**：讨论围绕着将激活值（特别是 KV caches）量化到 8 位以下时可能出现的精度问题展开。
  
  - *将激活值量化到 8 位以下会出现相当严重的精度问题*，这显示了对激进量化的谨慎态度。

 

**提到的链接**：[支持在不应用 LoRA 的情况下对线性层进行 NF4 量化 · Issue #1093 · pytorch/torchtune](https://github.com/pytorch/torchtune/issues/1093)：正如 @janeyx99 所指出的，我们的 quantize_base 参数只会对应用了 LoRA 的线性层的基础模型权重进行量化（例如参见我们的 Llama3 自注意力构建器）。但是...

 

---

### **DSPy ▷ #**[**papers**](https://discord.com/channels/1161519468141355160/1203568372667645963/1300683053928153110) (2 条消息):

> - `AI 隐私`
> - `隐私意识委托`
> - `本地 vs 专有 LLM`

- **PAPILLON 解决 AI 隐私问题**：研究人员推出了 [PAPILLON](https://arxiv.org/abs/2410.17127)，这是一个使用 AI 的系统，在确保 **85.5% 质量**的同时，仅维持 **7.5% 的隐私泄露**。
  
  - 这种方法允许个人同时利用**本地和云端 LLM**，有效地解决了 AI 隐私领域的一个主要担忧。
- **新基准 PUPA 凸显隐私问题**：团队创建了一个名为 **PUPA** 的基准，专注于包含个人身份信息（**PII**）的用户与 LLM 交互。
  
  - 该基准为他们关于**隐私意识委托（Privacy-Conscious Delegation）**的研究提供了依据，这是一种结合基于 API 的模型和本地模型的新方法。

 

**提到的链接**：[PAPILLON: 基于互联网和本地语言模型集成的隐私保护](https://arxiv.org/abs/2410.17127?s=03)：用户可能会向专有 LLM 提供商泄露敏感信息，从而引发重大的隐私担忧。虽然托管在用户本地机器上的开源模型缓解了一些担忧，...

 

---

### **DSPy ▷ #**[**general**](https://discord.com/channels/1161519468141355160/1161519469319946286/1300536555500339200) (14 条消息🔥):

> - `DSPy 用法`
> - `MIPROv2 优化器`
> - `DSPy 编程语言`
> - `MIPROv2 的 Bug 修复`
> - `DSPy 的 ELI5 解释`

- **用简单的术语解释 DSPy**：一位成员分享了 [DSPy 的 ELI5 解释](https://x.com/lateinteraction/status/1851324349216927856)，将其描述为一种使用普通 Python 配合 DSPy signatures 构建 AI 系统的编程语言。
  
  - 他们强调 DSPy 提供了处理提示策略的 Modules，以及根据给定 metric（指标）改进输出的 Optimizers。
- **关于 MIPROv2 优化器的见解**：讨论了 MIPROv2 优化器的功能，详细说明了它如何对训练集进行采样，并使用另一个 LM 根据各种数据属性生成高质量的指令。
  
  - 一位用户报告称，在定位明确的问题中使用 MIPROv2 时，质量提升了 **41%**，泄漏减少了 **68%**。
- **MIPROv2 的 Bug 修复**：一位用户报告了在使用 GPT-4o Mini 配合 MIPROv2 时出现错误，但在使用 GPT-4 时没有，这由于示例不完整导致置信度水平出现混乱。
  
  - 在调整参数以减少标记演示（labeled demos）的数量后，用户成功解决了该错误，并发现该设置在中等配置下也能良好运行。

**提到的链接**：

- [Hugging Face – 构建未来的 AI 社区。](https://huggingface.co/datasets?sort=trending&search=NER)：未找到描述
- [来自 Omar Khattab (@lateinteraction) 的推文](https://x.com/lateinteraction/status/1851324349216927856)：@baykenney 嘿 Matthew！DSPy 基本上是一种用于构建 AI 系统的编程语言。我们要求你用普通的 Python 编写系统，但以 DSPy signatures 的形式表达你的 AI 步骤。...
- [来自 MattCodes (@matt_c0des) 的推文](https://x.com/matt_c0des/status/1851312128491467000)：我正在尝试使用 DSPy 的 MIPROv2 来优化内容生成的提示。我的理解是 MIPROv2：1) 从你的训练集中采样，通过你的 LM 程序运行示例，...
- [来自 Omar Khattab (@lateinteraction) 的推文](https://x.com/lateinteraction/status/1851098213958529211)：看到 MIPRO 提示优化在这个 pipeline 中跨六个 LM 的表现非常迷人。开箱即用，质量提升高达 41%，泄漏减少 68%。还不错。引用...

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1300580355543269437) (4 条消息):

> - `Retrieval Augmented Generation (RAG)`
> - `Cohere Multi-Modal Embeddings`
> - `Azure AI App Templates`
> - `MLflow Integration`
> - `NVIDIA AI Insights`

- **NVIDIA 揭示关于 RAG 应用的研究发现**：NVIDIA 最新的博客文章讨论了他们在 **retrieval augmented generation (RAG)** 方面的发现，特别是用户如何希望获得 RAG 之外的功能，如文档翻译和代码编写。
  
  - 他们强调，即使是专注于内部数据的用户也看重 **web search** 能力，该能力通过 [Perplexity’s search API](https://docs.perplexity.ai/home) 实现。
- **使用 MLflow 构建高级 RAG 系统**：分享了一份关于使用 @MLflow 和 LlamaIndex 工作流构建 **advanced RAG systems** 的指南，实现了向量搜索和基于关键词搜索的并行结合。
  
  - 该设置旨在促进 **event-driven orchestration** 以改进工作流管理，并在 [GitHub](https://github.com) 上的示例中进行了展示。
- **Cohere 发布多模态嵌入 (multi-modal embeddings)**：Cohere 宣布发布 **multi-modal embeddings**，允许在同一向量空间内集成图像和文本，以增强 AI 能力。
  
  - 提供了一个教学用的 [notebook](https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/multi_modal/cohere_multi_modal.ipynb)，演示了如何结合 Qdrant 使用这些嵌入进行多模态检索。
- **在 GitHub Universe 发布 Azure AI App Templates**：Azure 在 GitHub Universe 上推出了 **AI App Templates**，LlamaIndex 是首批利用该资源进行快速 AI 开发的应用之一。
  
  - 演示重点介绍了精心挑选的 **AI App Template Gallery**，使开发者能够使用 **infrastructure as code** 和 CI/CD 流水线轻松部署应用程序。

**提到的链接**：

- [Creating RAG-Based Question-and-Answer LLM Workflows at NVIDIA | NVIDIA Technical Blog](https://t.co/8tPnIY8VQa)：使用 retrieval augmented generation (RAG) 构建问答 LLM 工作流的解决方案快速发展，催生了新型系统架构。我们在 NVIDIA 使用 AI 的工作...
- [GenerativeAIExamples/community/routing-multisource-rag at main · NVIDIA/GenerativeAIExamples](https://t.co/EfUToAARR3)：针对加速基础设施和微服务架构优化的 Generative AI 参考工作流。 - NVIDIA/GenerativeAIExamples
- [Multi-Modal Retrieval using Cohere Multi-Modal Embeddings - LlamaIndex](https://t.co/57I7lIJKMJ)：未找到描述
- [Azure at GitHub Universe: New tools to help simplify AI app development | Microsoft Azure Blog](https://t.co/o15rj6O6Ux)：了解如何通过 VS Code、GitHub 和 Azure 之间的无缝集成，利用 AI 改造您的应用。
- [Build AI applications with the new AI App Template Gallery](https://t.co/Vb7tIahMkm)：使用新的 AI App Template Gallery，在几分钟内开始构建 AI 应用程序。AI App Template Gallery 是一个旨在帮助您构建和部署 AI 应用程序的新资源。该集合包括...

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1300555440748367892) (8 条消息🔥):

> - `Blockchain Engineering Contributions` (区块链工程贡献)
> - `Chroma Vector Store Retrieval Behavior` (Chroma 向量存储检索行为)
> - `Web Scraping with LLM` (使用 LLM 进行网页抓取)
> - `Date-related Vector Search Queries` (与日期相关的向量搜索查询)

- **区块链工程师寻求机会**：一位自 2017 年起从业的全栈区块链工程师表达了参与项目的兴趣，重点介绍了在 **defi**、**NFT games** 以及 **Solidity** 和 **RUST** 等协议方面的经验。
  
  - 他们此前曾参与过 **Dex**、**DAO** 以及 **NFT** 铸造和质押等项目。
- **Chroma 的检索算法引发讨论**：一位成员对返回结果的可变性提出疑问，提到在使用 **Chroma** 作为向量存储时，设置如下：`index = GPTVectorStoreIndex.from_vector_store(vector_store=vector_store)`。
  
  - 另一位成员指出 **Chroma** 的检索算法是近似的，这可能导致即使索引了相似的 chunk，结果也会发生变化。
- **使用 LLM 的网页抓取技术**：分享了一个名为“[这就是我如何通过 LLM 抓取 99% 的网站](https://youtu.be/7kbQnLN2y_I)”的 YouTube 视频，重点介绍了 2024 年的网页抓取能力。
  
  - 它鼓励使用 **AgentQL** 免费抓取网站，强调了 LLM 在网页抓取中的实际应用。
- **针对日期的向量搜索查询**：一位用户询问如何针对日期相关的查询执行向量搜索，并提供了与发票列表和比较相关的具体示例。
  
  - 他们特别强调了诸如按日期范围筛选发票和比较金额等查询，寻求关于如何高效处理此类数据的见解。

**提到的链接**：[This is how I scrape 99% websites via LLM](https://youtu.be/7kbQnLN2y_I)：2024 年如何使用 LLM 进行网页抓取。使用 AgentQL 免费抓取网站：[https://www.agentql.com/?utm_source=YouTube&utm_medium=Creator&utm_id=AIJason](https://www.agentql.com/?utm_source=YouTube&utm_medium=Creator&utm_id=AIJason)_...

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**hackathon-announcements**](https://discord.com/channels/1280234300012494859/1280236929379602493/1300617007930015816) (1 条消息):

> - `LLM Agents Hackathon Registration` (LLM Agents 黑客松报名)
> - `Team Formation Open` (组队开放)
> - `Prize Pool Announcement` (奖池公布)
> - `Tracks Overview` (赛道概览)
> - `Social Media Promotion` (社交媒体推广)

- **LLM Agents 黑客松报名人数激增**：在短短几天内，已有超过 **1000 多名创新者**报名参加了 [LLM Agents Hackathon](https://rdi.berkeley.edu/llm-agents-hackathon/)，反映出浓厚的兴趣。
  
  - *现在加入我们还不晚！* 今天就完成 [参与者报名](https://docs.google.com/forms/d/e/1FAIpQLSevYR6VaYK5FkilTKwwlsnzsn8yI_rRLLqDZj0NH7ZL_sCs_g/viewform)。
- **组队报名现已开放**：**重要提示：** 黑客松的组队报名[现已开放](https://docs.google.com/forms/d/e/1FAIpQLSdKesnu7G_7M1dR-Uhb07ubvyZxcw6_jcl8klt-HuvahZvpvA/viewform?usp=sf_link)。
  
  - 参与者现在可以组建团队，以便有效地协作和竞争。
- **公布超过 20 万美元奖金**：报名后，参与者可以通过 [计算资源申请](https://docs.google.com/forms/d/e/1FAIpQLSc_7YY-u-aDZ-xWYflq7FUM6R1a3rnQKg6o_ikXsProhrlgBA/viewform?usp=sf_link) 获得超过 **20 万美元的奖品、资源和额度**。
  
  - 这一机会鼓励参与者利用可用资源来开展他们的项目。
- **探索五个精彩赛道**：黑客松设有 **五个赛道**，包括应用 (Applications)、基准测试 (Benchmarks)、基础 (Fundamentals)、安全 (Safety) 以及去中心化与多智能体 (Decentralized & Multi-Agents)，供参与者深入探索。
  
  - 这些主题领域旨在推进 LLM agents 的各个方面并培养创新想法。
- **加入社交媒体推广活动**：鼓励参与者通过转发 [@dawnsongtweets](https://x.com/dawnsongtweets/status/1850967229518819355) 的推广帖子来协助宣传。
  
  - 同时也鼓励在 LinkedIn 和其他平台分享，以扩大黑客松公告的影响力。

**提到的链接**：[Dawn Song (@dawnsongtweets) 的推文](https://x.com/dawnsongtweets/status/1850967229518819355>)：🚀非常激动，在短短几天内已经有 1000 多名创新者报名参加了我们的 LLM Agents MOOC 黑客松！🎉在 5 个赛道中构建、协作并竞争 20 万美元以上的奖品/资源/额度...

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-announcements**](https://discord.com/channels/1280234300012494859/1280369709623283732/1300548204361945249) (1 条消息):

> - `第 8 讲公告`
> - `神经与符号决策`
> - `Yuandong Tian 的演讲`

- **第 8 讲定于 PST 时间下午 3:00**：**第 8 讲**将于今天 **PST 时间下午 3:00** 举行，[此处提供直播](https://www.youtube.com/live/wm9-7VBpdEo)。
  
  - 鼓励参与者收看，这预计将是一个关于将复杂推理与 Large Language Models 集成的宝贵环节。
- **探索神经与符号决策**：客座讲师 **Yuandong Tian** 将介绍如何集成**神经与符号组件**以增强 LLM 的推理能力。
  
  - 本次演讲旨在解决传统符号求解器和当前神经模型在处理复杂的、自然描述的问题时的局限性。
- **Yuandong Tian 在 AI 研究领域的专业知识**：**Yuandong Tian** 是 Meta AI Research 的研究科学家总监，领导关于 LLM 推理和规划的研究。
  
  - 他的专业知识凸显了在当代 AI 应用中改进决策过程的持续努力。

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-questions**](https://discord.com/channels/1280234300012494859/1280370030609170494/1300584101610262630) (6 条消息):

> - `直播字幕`
> - `基于 React 的 Agent 开发`

- **直播字幕请求**：一位成员请求在直播视频中启用**字幕**。另一位成员指出，虽然他们将与工作人员讨论未来的安排，但所有讲座在录制后都会添加字幕。
  
  - 他们确认讲座随后会进行编辑并提供字幕，以确保观众的可访问性。
- **开发基于 React 的自动化 Agent**：一位成员询问如何创建一个**基于 React 的 Agent**，通过截屏并使用 [pyauto gui](https://pyautogui.readthedocs.io/en/latest/) 和 [pygetwindow](https://pygetwindow.readthedocs.io/en/latest/) 根据当前状态的评估执行操作来自动化任务。
  
  - 另一位成员建议直接提问比进行笼统的询问更容易。

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-lecture-discussion**](https://discord.com/channels/1280234300012494859/1282734248112947210/1300552292797517854) (4 条消息):

> - `学习小组兴趣`
> - `讲座总结`
> - `虚拟会议`

- **组建学习小组**：一位成员提议为那些较晚加入课程的人成立一个**学习小组**，讨论讲座和阅读材料，并建议举行虚拟会议。
  
  - 随后很快有人表达了兴趣，几位成员（包括一位表示“听起来很酷！”的成员）确认了他们的兴趣。
- **会议时间调查**：学习小组将尝试通过 [Google 表单](https://forms.gle/QtQ2C6qzomeHDrC38) 收集偏好，详细列出可参加会议的时间，以协调大家的日程。
  
  - 建议的时间段包括工作日晚上和周末，为参与者提供了多种选择。

 

**提到的链接**：[LLM Agents 同伴学习小组 (虚拟)](https://forms.gle/QtQ2C6qzomeHDrC38)：使用此链接表达您加入虚拟同伴学习小组的兴趣。我们可能会使用 Discord 活动或 Zoom。我们将从第一讲开始复习讲座，并讨论额外的...

 

---

### **LAION ▷ #**[**general**](https://discord.com/channels/823813159592001537/823813160075132991/1300884129214107689) (3 条消息):

> - `类别条件潜扩散模型`
> - `粉色像素块`
> - `DDIM p_sample 裁剪`
> - `IJEPA 与自回归架构`

- **训练类别条件潜扩散模型**：一位成员报告在训练类别条件潜扩散模型时，从 VAE 解码时遇到了**粉色像素块**。
  
  - 他们指出，随着训练的进行，这些色块出现的频率会降低，但仍偶尔出现。
- **DDIM p_sample 裁剪考虑**：同一位成员正在考虑是否在 **DDIM p_sample** 中应用更激进的裁剪，目前设置为 **99.95%** 的极值。
  
  - *他们不确定这种调整是否会消除持续存在的粉色色块*。
- **创新架构合作**：该成员对合作训练一种结合了 **IJEPA** 和 `autoregressive image generation without vector quantization` 的架构持开放态度。
  
  - 他们对共同努力探索这种独特的**架构**表达了热情。

 

---

### **LAION ▷ #**[**research**](https://discord.com/channels/823813159592001537/824374369182416994/1300541614703513700) (5 messages):

> - `Model Parameters`
> - `Token vs Parameters Confusion`

- **参数与 Token 之间的误解**：一位成员误以为提到的 **100B** 是指参数量而非 Token 数量，从而导致了混淆。
  
  - 另一位成员承认了这一区别，表示 *'you're right my bad'*，澄清了最初的误解。
- **关于模型大小的澄清**：一位成员指出链接中的模型仅有 **8B parameters**，与之前假设的更大型模型形成对比。
  
  - 这一澄清得到了另一位成员的认可，他回答道 *'well said'*，表示赞同。

 

---

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1300809965425922079) (7 messages):

> - `Negative Line Day Effects`
> - `CI Test Improvements`
> - `Uops Readability Challenges`
> - `Documentation Maintenance Concerns`
> - `Premature Optimization in Code`

- **George Hotz 报告“负行数日”**：*George Hotz* 表示经历了一个**负行数日**（Negative Line Day，指代码行数净减少），另一位成员幽默地向他表示祝贺。
  
  - 这种交流体现了社区内非正式且相互支持的互动性质。
- **观察到更快的持续集成测试**：*Chenyuy* 注意到 **CI 测试快了 2 分钟**，表明测试流程得到了改进。
  
  - 这种优化突显了项目中为提升性能所做的持续努力。
- **关于 Uops 可读性的讨论**：一位成员对 **Uops 可读性**表示担忧，认同某些单行代码（one-liners）可能难以理解，但未提供明确的解决方案。
  
  - 他们建议创建一个文档页面进行说明，这有助于提高代码的可理解性。
- **对文档时效性的担忧**：*Chenyuy* 表示不太倾向于编写文档，因为文档往往会迅速过时，从而导致维护困难。
  
  - 他强调，错误的文档可能比没有文档更有害，这突显了 tinygrad 中代码变更的速度之快。
- **辩论过早优化**：*George Hotz* 建议移除某些代码元素，因为这些元素可能是**过早优化（premature optimization）**。
  
  - 这反映了一种深思熟虑的方法，即在确保代码效率的同时避免不必要的复杂性。

 

---

### **LangChain AI ▷ #**[**general**](https://discord.com/channels/1038097195422978059/1038097196224086148/1300564669869588522) (4 messages):

> - `RAGAS`
> - `CSV files in open source models`
> - `LangChain Python Compatibility`
> - `LangChain-JS Course`

- **利用 RAGAS 进行 LLM 评估**：一位成员建议使用 [RAGAS](https://github.com/explodinggradients/ragas) 来增强 LLM 应用的评估，并展示了其 GitHub 仓库。
  
  - 该仓库承诺通过各种方法论帮助开发者加强其 **LLM evaluations**。
- **CSV 与开源模型的集成**：一位成员询问如何使用 **LLAMA3** 等开源模型将 **CSV 文件**作为数据源集成，并指出目前该配置缺乏示例或功能。
  
  - 他们正在寻求关于在非 OpenAI 模型上使用 CSVChain 和 PandasAgent 的指导。
- **Python 与 LangChain 0.3 的兼容性**：一位成员请求澄清哪个版本的 **Python** 与 **LangChain version 0.3** 兼容。
  
  - 这反映了社区对确保正确配置 LangChain 使用环境的持续关注。
- **免费 LangChain-JS 课程发布公告**：**好消息！** 一门全新的 [面向初学者的 LangChain-JS 课程](https://www.udemy.com/course/genai-langchain-for-javascript-developers/?couponCode=AMIT100) 已在 Udemy 上线，涵盖了从基础到构建完整 RAG 应用的所有内容。
  
  - 前 **100 名学生**可以免费加入，鼓励大家尽快报名。

 

**提到的链接**：[GitHub - explodinggradients/ragas: Supercharge Your LLM Application Evaluations 🚀](https://github.com/explodinggradients/ragas)：加强你的 LLM 应用评估 🚀。通过在 GitHub 上创建账号为 explodinggradients/ragas 的开发做出贡献。

 

---

### **LangChain AI ▷ #**[**tutorials**](https://discord.com/channels/1038097195422978059/1077843317657706538/1300795224812945429) (2 messages):

> - `使用 LLM 进行网页抓取`
> - `LangChain-JS 课程`

- **学习使用 LLM 进行网页抓取**：查看这个内容丰富的 [YouTube 视频](https://youtu.be/7kbQnLN2y_I)，标题为“这就是我如何通过 LLM 抓取 99% 的网站”，该视频教授了 2024 年如何使用 LLM 进行网页抓取。
  
  - 该视频解释了如何使用 [AgentQL](https://www.agentql.com/) 免费抓取网站，并强调了实际应用。
- **新的 LangChain-JS 课程上线了！**：一个令人兴奋的面向初学者的 LangChain-JS 新课程已在 [Udemy](https://www.udemy.com/course/genai-langchain-for-javascript-developers/?couponCode=AMIT100) 上线！它涵盖了从 LangChain 基础到构建完整的 RAG 应用的所有内容。
  
  - 前 **100 名学生**可以免费加入，所以不要错过这个提升技能的机会！

 

**提到的链接**：[This is how I scrape 99% websites via LLM](https://youtu.be/7kbQnLN2y_I)：如何在 2024 年使用 LLM 进行网页抓取。使用 AgentQL 免费抓取网站：[https://www.agentql.com/?utm_source=YouTube&utm_medium=Creator&utm_id=AIJason](https://www.agentql.com/?utm_source=YouTube&utm_medium=Creator&utm_id=AIJason)_...

 

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #**[**leaderboard**](https://discord.com/channels/1111172801899012102/1214705495974092810/1300560060182954134) (5 messages):

> - `排行榜术语`
> - `Multi-Step 与 Multi-Turn 评估`

- **理解排行榜上的 'Multiple'**：排行榜上的 'Multiple' 指的是在单轮设置中从多个选项中**选择正确函数的能力**，正如一位引用了 [GitHub 示例](https://github.com/ShishirPatil/gorilla/blob/2101b11f6d03d9f323715d7d2012a955d7f4114e/berkeley-function-call-leaderboard/data/BFCL_v3_exec_multiple.json#L42C1-L42C2438) 的用户所澄清的那样。然而，目前尚不清楚在这种情况下如何评估 Multi-Step。
- **Multi-Step 与 Multi-Turn 的澄清**：一位成员指出，虽然 'multiple' 与函数有关，但 **Multi-Step 评估**属于不同的类别，被描述为 'multi_turn'。目前没有正在使用的独立 'Multi-Step' 评估。
- **混合 Multi-Step 与 Multi-Turn 类别**：有人提到，在 **'multi_turn' 类别**中，Multi-Step 和 Multi-Turn 评估共存，每一轮可能涉及多个步骤。这种重叠可能会导致混淆，因为没有专门的 Multi-Step 类别。

**提到的链接**：

- [GitHub - ShishirPatil/gorilla at 2101b11f6d03d](https://github.com/ShishirPatil/gorilla/blob/2101b11f6d03d)：Gorilla：训练和评估用于 Function Calls (Tool Calls) 的 LLM - GitHub - ShishirPatil/gorilla at 2101b11f6d03d9f323715d7d2012a955d7f4114e
- [gorilla/berkeley-function-call-leaderboard/data/BFCL_v3_exec_multiple.json at 2101b11f6d03d9f323715d7d2012a955d7f4114e · ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla/blob/2101b11f6d03d9f323715d7d2012a955d7f4114e/berkeley-function-call-leaderboard/data/BFCL_v3_exec_multiple.json#L42C1-L42C2438))：Gorilla：训练和评估用于 Function Calls (Tool Calls) 的 LLM - ShishirPatil/gorilla

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**general**](https://discord.com/channels/1238365980128706560/1238365980128706563/1300875366436442213) (2 条消息):

> - `Cracked Engineers 招聘平台`
> - `技术职位时事通讯`
> - `AI 初创公司招聘`

- **Cracked Engineers 招聘平台发布！**：一位成员分享了一个令人兴奋的针对[技术角色的新招聘平台](https://www.crackedengineers.com/)，名为 **Cracked Engineers**，旨在成为顶尖 AI/技术初创公司的首选。
  
  - 在正式发布前，该平台的预期 **MRR**（月经常性收入）已达 **$1000**，目前已吸引了 **Weaviate**、**UnslothAI** 和 **JuliusAI** 等顶尖公司。
- **富有洞察力的每周技术职位时事通讯推出**：该平台即将发布 **每周技术职位时事通讯**，将根据用户偏好筛选职位。
  
  - 用户可以通过仪表板订阅感兴趣的标签，如 **CUDA**、**MLOps** 或 **Software Engineering**。
- **AI 初创公司的精彩职位机会**：一位成员强调，**Unsloth AI**、**Julius AI** 和 **Jimini AI** 正在积极招聘优秀职位，如果他不是创始人，他也会考虑这些职位。
  
  - 这些职位被描述为任何想要从事前沿 AI 技术工作的人的**绝佳机会**。

**提到的链接**：

- [来自 Aleksa Gordić 🍿🤖 (@gordic_aleksa) 的推文](https://x.com/gordic_aleksa/status/1851247076987855063)：[🚀] 超级兴奋地分享这个：我建立了一个名为 "Cracked Engineers" 的技术角色招聘平台。:) 如果你想在世界上最好的 AI/技术初创公司找到工作 ...
- [来自 undefined 的推文](https://x.com/gor)：未找到描述

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general**](https://discord.com/channels/1104757954588196865/1104757955204743201/1300549775254356138) (1 条消息):

> - `SymNoise 方法论`
> - `LLaMA-2-7B 性能`
> - `Fine-tuning 技术`

- **关于 SymNoise 实现的咨询**：一位成员正在寻求关于论文中讨论的 **SymNoise** 语言模型微调技术的代码实现，该技术将 **symmetric noise**（对称噪声）引入嵌入过程。
  
  - *然而，他们指出了实现中的挑战，特别是关于通过拼接使* ***batch size*** *翻倍的问题。*
- **使用 SymNoise 的 LLaMA-2-7B 评分**：据报道，**SymNoise** 方法将 **LLaMA-2-7B** 模型在 AlpacaEval 上的性能从 **29.79%** 提升到了 **69.04%**，优于之前的方法 **NEFTune**。
  
  - *根据论文摘要，这比 NEFTune 的* ***64.69%*** *评分提高了* ***6.7%***。
- **SymNoise vs. NEFTune**：在各种模型和更强的基准指令数据集测试中，**SymNoise 始终优于 NEFTune**。
  
  - *讨论强调了在该领域进行更深入研究的必要性，正如论文中所指出的。*
- **索取研究链接**：咨询中包含了一个指向 **arXiv** 论文的链接以便进一步阅读，强调了该研究发现的重要性。
  
  - *其他成员没有提供额外的实现或链接来解决代码咨询问题。*

 

**提到的链接**：[SymNoise: Advancing Language Model Fine-tuning with Symmetric Noise](https://arxiv.org/abs/2312.01523)：在这篇论文中，我们介绍了一种新型的语言模型微调技术，涉及将对称噪声引入嵌入过程。该方法旨在增强模型的功能...

 

---

---

---

---

---

{% else %}

> 按频道划分的详细分解内容已在电子邮件中截断。
> 
> 如果您想查看完整的分解内容，请访问此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！
> 
> 如果您喜欢 AInews，请[与朋友分享](https://buttondown.email/ainews)！提前感谢！

{% endif %}