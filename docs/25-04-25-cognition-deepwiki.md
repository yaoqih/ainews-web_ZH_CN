---
companies:
- cognition
- meta-ai-fair
- alibaba
- hugging-face
- openai
- perplexity-ai
- vllm
- ''
date: '2025-04-25T05:44:39.731046Z'
description: '以下是翻译内容：


  **Cognition** 的 **Silas Alberti** 宣布推出 **DeepWiki**，这是一个涵盖所有 GitHub 仓库的免费百科全书，为公共仓库提供类似维基百科的描述以及由
  Devin 支持的聊天机器人。**Meta** 发布了采用 A2.0 许可证的**感知编码器 (Perception Encoders, PE)**，在视觉任务上的表现优于
  **InternVL3** 和 **Qwen2.5VL**。**阿里巴巴**推出了适用于 iOS 和 Android 的 **通义千问 (Qwen) 聊天应用**。**Hugging
  Face** 通过 **FAL** 集成了 **Dia 1.6B SoTA** 文本转语音模型。**OpenAI** 扩展了深度研究 (Deep Research)
  的使用范围，推出了由 **o4-mini** 模型驱动的轻量级版本，现已向免费用户开放。**Perplexity AI** 更新了其模型选择器，新增了 **Grok
  3 Beta**、**o4-mini**，并支持 **Gemini 2.5 Pro**、**Claude 3.7** 和 **GPT-4.1** 等模型。**vLLM**
  项目引入了 **OpenRLHF** 框架，用于人类反馈强化学习。**Surya OCR** Alpha 模型现已支持 90 多种语言和 LaTeX。推出了 **MegaParse**
  开源库，用于处理适用于大语言模型 (LLM) 的数据格式。'
id: 1214124323
models:
- o4-mini
- perception-encoder
- qwen-2.5-vl
- dia-1.6b
- grok-3
- gemini-2.5-pro
- claude-3.7
- gpt-4.1
people:
- silas-alberti
- mervenoyann
- reach_vb
- aravsrinivas
- vikparuchuri
- lioronai
title: Cognition 的 DeepWiki，一个涵盖所有 GitHub 仓库的免费百科全书。
topics:
- vision
- text-to-speech
- reinforcement-learning
- ocr
- model-releases
- model-integration
- open-source
- frameworks
- chatbots
- model-selector
---



我们对 React 和 Astro 仓库进行的初步测试（AINews 现在是基于这些仓库构建的，所以我们很熟悉）非常令人振奋。值得一试，特别是对于使用开源代码而言。

---

# AI Twitter 综述

**模型发布与更新**

- **Meta 的 Perception Encoders (PE)**：[@mervenoyann](https://twitter.com/mervenoyann/status/1915723394701467909) 强调 **Meta 发布了带有 A2.0 许可证的视觉“瑞士军刀” ❤️，包括用于视觉语言和空间理解的图像/视频编码器**，并指出其性能优于 **InternVL3 和 Qwen2.5VL**，且附带 **庞大的视频和图像数据集**。[@mervenoyann](https://twitter.com/mervenoyann/status/1915723399642435634) 进一步指出 **Perception Encoder (PE) Core 在零样本（zero-shot）图像任务中优于最新的 SOTA SigLIP2 🔥**。模型和数据集可以在 [Perception Encoder models](https://twitter.com/mervenoyann/status/1915723397272654194) 和 [Perception LM models](https://twitter.com/mervenoyann/status/1915723397272654194) 找到。
- **Qwen Chat App 上线**：[@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1915761990703697925) 宣布 **Qwen Chat APP 现已面向 iOS 和 Android 用户开放**，旨在协助创意、协作和无限可能。
- [@reach_vb](https://twitter.com/reach_vb/status/1915830938438717777) 宣布你可以在 **Hugging Face Hub 上对 30,000 多个 Flux 和 SDXL LoRA 进行推理**，并且 **生成 40 多张图片的成本不到一美元！**
- **Hugging Face 与 FAL 集成**：[@reach_vb](https://twitter.com/reach_vb/status/1915418386818834792) 宣布了 **全新的文本转语音模型 Dia 1.6B SoTA**，可通过 @FAL ⚡ 直接在 Hugging Face 上使用。用户花费 **不到一美元即可获得多达 25 次生成**。
- **OpenAI 新模型与 Deep Research**：[@OpenAI](https://twitter.com/OpenAI/status/1915505959931437178) 宣布他们正在 **通过引入轻量级版本的 Deep Research 来扩大 Plus、Team 和 Pro 用户的 Deep Research 使用范围，以提高当前的速率限制（rate limits）**，并向免费用户推出轻量级版本。[@gdb](https://twitter.com/gdb/status/1915637620731941188) 确认 **Deep Research（轻量版）现已在 ChatGPT 免费版中可用**。[@OpenAI](https://twitter.com/OpenAI/status/1915505961500070245) 还指出 **轻量版 Deep Research 由 OpenAI o4-mini 的一个版本驱动**，其智能程度几乎与人们熟知并喜爱的 Deep Research 相当，同时服务成本显著降低。
- **Perplexity 模型更新**：[@perplexity_ai](https://twitter.com/perplexity_ai/status/1915819644256129424) 宣布 **Grok 3 Beta 和 o4-mini 现已在模型选择器中可用**。[@AravSrinivas](https://twitter.com/AravSrinivas/status/1915820052571689245) 指出模型选择器已经 **支持 Gemini 2.5 Pro, Claude 3.7, Perplexity Sonar, GPT-4.1, DeepSeek R1 1776**，并正在研究支持 o3。[@perplexity_ai](https://twitter.com/perplexity_ai/status/1915819619333640647) 还引入了 **来自 OpenAI 的最新图像生成模型，支持上下文图像生成和编辑**。
- **用于 RLHF 的 vLLM**：[@vllm_project](https://twitter.com/vllm_project/status/1915307134256091570) 强调了 **OpenRLHF，一个使用 vLLM 进行 RLHF 的框架**，它推动了 vLLM 许多针对 RLHF 的功能设计和实现，使 vLLM 成为许多 RLHF 框架的热门选择。
- **Surya OCR 模型**：[@VikParuchuri](https://twitter.com/VikParuchuri/status/1915492483955384659) 宣布了 **新 Surya OCR 模型的 Alpha 版本，支持 90 多种语言、LaTeX 和格式化、字符/单词/行边界框（bboxes）、约 5 亿非嵌入参数（non-embed params），速度达 10-20 页/秒**。

**框架、工具与数据集**

- **MegaParse 用于 LLM 就绪格式**：[@LiorOnAI](https://twitter.com/LiorOnAI/status/1915792212157407385) 介绍了 **MegaParse，一个开源 Python 库，用于将任何文档转换为 LLM 就绪格式**，支持处理 PDF、Powerpoint、Word、表格、目录 (TOC)、页眉、页脚和图像。
- **LangGraph DevX**：[@hwchase17](https://twitter.com/hwchase17/status/1915208270593352002) 讨论了 **塑造 LangGraph DevX**，询问预构建的 Agent 构造器应该是类还是函数。
- **Google Agent Development Kit (ADK)**：[@omarsar0](https://twitter.com/omarsar0/status/1915402607574893052) 分享了 **如何开始使用 ADK 的快速指南**，并指出该项目仍在开发中。
- **ReflectionFlow 框架**：[@RisingSayak](https://twitter.com/RisingSayak/status/1915338106510905767) 发布了 **ReflectionFlow**，这是一个允许文本生成图像扩散模型通过反思（reflection）来优化自身输出的框架。他们发布了 **GenRef-1M，一个由 (good_img, bad_img, reflection) 三元组组成的大规模数据集。**
- **OpenAI Codex 基金资助名单**：[@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1915524612970152376) 公布了首批 **Codex 开源基金资助获得者：vLLM、OWASP Nettacker、Pulumi 和 Dagster。**
- **Spotify 的 ViSMaP**：[@_akhaliq](https://twitter.com/_akhaliq/status/1915703054701044209) 宣布 Spotify 刚刚在 Hugging Face 上发布了 **ViSMaP。通过 Meta-Prompting 实现的无监督长达一小时的视频摘要**。
- **字节跳动的 QuaDMix**：[@_akhaliq](https://twitter.com/_akhaliq/status/1915656590130036887) 宣布字节跳动刚刚在 Hugging Face 上发布了 **QuaDMix。用于高效 LLM 预训练的质量-多样性平衡数据选择**。
- **助力研究人员探索 DeepSeek R1 中可解释特征的新型可查询数据集**：[@GoodfireAI](https://twitter.com/GoodfireAI/status/1915802798513598490) 发布。
- **Trackers v2.0.0 发布**：[@skalskip92](https://twitter.com/skalskip92/status/1915439480594485363) 宣布了来自顶级模型库的组合目标检测器，可配合你选择的多目标追踪器。目前支持 SORT 和 DeepSORT；更多追踪器即将推出。

**Agentic 系统与工具使用**

- **Agentic AI 与可见性**：[@weights_biases](https://twitter.com/weights_biases/status/1915498157754233092) 表示 **没有可见性的 Agentic AI = 混乱**，并强调了与 @deepset_ai 的合作，旨在为 AI 工作流带来透明度。
- **Meta 的 3D 生成式 AI**：[@AIatMeta](https://twitter.com/AIatMeta/status/1915437886209745338) 提到他们正在积极招聘研究人员加入，共同构建未来的现实。
- **AI 驱动的助手**：[@perplexity_ai](https://twitter.com/perplexity_ai/status/1915438278947283301) 宣布 **Perplexity Android 应用将预装在所有新款 Motorola 设备上**。此外，他们还与 @moto 合作，为 Moto Razr 优化了 Perplexity Assistant。新购买的用户将获得 3 个月的 Perplexity Pro。
- **个性化且多模态的实时 Agent**：来自 Google Cloud 的 [@_philschmid](https://twitter.com/_philschmid/status/1915360039570739283) 表示，他们展示了对下一代 Agent 运作方式的构想：个性化、实时且多模态！全部由 @GoogleDeepMind Gemini 2.0 Flash 和 Live API 驱动。

**可解释性与评估**

- **AI 可解释性**：[@GoodfireAI](https://twitter.com/GoodfireAI/status/1915617077915967714) 分享了他们的信念，即我们可以理解并设计 AI 模型的“思想”，且必须紧迫地开展这项工作，并指出我们正处于可解释性与模型智能之间的竞赛中。
- **可解释性仍然是学术界贡献的绝佳领域**：[@NeelNanda5](https://twitter.com/NeelNanda5/status/1915546126259806590) 分享了他的想法，表示：“世界应该在可解释性（以及其他形式的安全）上投入更多。随着规模化（scale）使 AI 学术界的许多部分变得越来越无关紧要”。
- [@karpathy](https://twitter.com/karpathy/status/1915581920022585597) 分享了 **AI 辅助编程中的某种节奏（即我真正专业关注的代码，而非氛围代码 vibe code）**，指出许多阶段仍然笨拙且需要手动操作，在现有工具中尚未得到明确或良好的支持，我们仍处于早期阶段，AI 辅助编程的 UI/UX 仍有很大提升空间。
- **LLM 评估**：[@clefourrier](https://twitter.com/clefourrier/status/1915339216344526896) 呼吁用户不要 **花费 2000 美元 😱 去学习 LLM 评估**，建议查看他们的免费/开源资源：[LLM 指南](https://github.com/NannyML/nannyml/blob/main/guide/llm-guide.md) 和 [YourBench](https://github.com/seb-lgr/your-bench)。

**AI 伦理与福利**

- **AI 与意识**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1915420604397744497) 宣布他们**最近启动了一项研究计划来调查 AI 福利 (AI welfare)**，探讨随着 AI 模型变得越来越复杂和强大，它们是否有可能拥有自己的体验。
- [@aidan_mclau](https://twitter.com/aidan_mclau/status/1915444696090108077) 表示，她不由自主地认为“模型有 15% 的概率具有意识”这一说法非常荒谬，就像说“我有 15% 的概率应该把我桌子上这堆原子集合称为我的电脑”一样。

**行业与商业**

- **Kaicathyc 绿卡被拒**：[@polynoamial](https://twitter.com/polynoamial/status/1915765141846515883) 报告称，**他合作过的最优秀的 AI 研究员之一 @kaicathyc 今天被拒绝了美国绿卡申请**，现在不得不离开。
- **AI 与媒体的未来**：[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1915514295816737140) 分享了他们对媒体未来的看法，指出数十亿人正在跨越最后一座桥梁：任何人都能制作出可供大规模分发的高质量内容。
- **Uber 对 LangGraph 的使用**：[@LangChainAI](https://twitter.com/LangChainAI/status/1915191956810207431) 强调 **Uber 的开发者平台团队使用 LangGraph 构建了一个 Agent 网络并实现了单元测试生成的自动化**，构建了可在整个组织内采用的可重用组件。
- [@skirano](https://twitter.com/skirano/status/1915414930536235048) 表示他正在招聘！如果你对 AI 与创意的交汇点、挑战 LLM 的极限以及重构现代 Web 架构感兴趣，这里就是你的理想去处。

**ICLR 会议**

- **Meta 在 ICLR**：[@AIatMeta](https://twitter.com/AIatMeta/status/1915437886209745338) 宣布 Meta 参加了在新加坡举行的 **ICLR2025 EXPO**。
- **按国家划分的 ICLR 2025 参会者**：[@hardmaru](https://twitter.com/hardmaru/status/1915341552332808383) 发布了一张按国家划分的 ICLR 2025 参会者图表，以及论文数量 / 接收率。
- [@shaneguML](https://twitter.com/shaneguML/status/1915169621042499846) 分享了他们**来自 ICLR 第一天的 AI 研究犀利观点：是时候从 R（研究）转向 D（开发）了。**

**交通与基础设施**
- [@rown](https://twitter.com/rown/status/1915607964972429522) 赞扬了**新加坡地铁 (MRT) 的高频次、全自动化、安全性、清洁度以及开环支付系统**。

**幽默**

- [@teortaxesTex](https://twitter.com/teortaxesTex/status/1915847934991864079) 分享了一个迷因，并说“这让我想起了我最喜欢的迷因之一 https://t.co/ffVTqHIJbz”
- [@AravSrinivas](https://twitter.com/AravSrinivas/status/1915379529960300989) 发布了 "

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

### 1. 用于推理的无损 LLM 压缩 (DF11) 研究与发布

  - **[我们在推理过程中将任何 BF16 模型压缩到约 70% 的大小，同时保持输出无损，以便你可以容纳更多 ERP 上下文或运行更大的模型。](https://www.reddit.com/r/LocalLLaMA/comments/1k7o89n/we_compress_any_bf16_model_to_70_size_during/)** ([评分: 292, 评论: 89](https://www.reddit.com/r/LocalLLaMA/comments/1k7o89n/we_compress_any_bf16_model_to_70_size_during/)): **这篇文章介绍了 DF11，这是一种采用哈夫曼编码的分块权重压缩格式，旨在将任何 BF16 模型的推理内存占用减少约 30%（降至约 11 bits/weight，约为原始大小的 70%），同时保持无损输出。推理基准测试显示，DF11 使得以前在有限 VRAM 上无法运行的模型成为可能（例如，Llama-3.1-405B-Instruct 在 551 GB 上运行，而 BF16 需要 812 GB），其 batch-128 吞吐量与 BF16 相当 (1.02x)，且远快于 CPU offload (最高达 38.8x)；然而，单 batch 延迟较差（比 BF16 慢约 40%）。该方法已开源 (https://github.com/LeanModels/DFloat11)，并提供了代码和 arXiv 预印本 (https://arxiv.org/abs/2504.11651)。无损压缩避免了有损整数倍量化带来的不可预测的经验性退化，使其更适合需要完全确定性或最大化任务覆盖范围的用户，并且可能与 LoRA 等 PEFT 方法结合使用，但目前由于分块应用会破坏梯度。** 评论者建议在流行的推理框架 (llama.cpp, VLLM) 中实现 DF11 以提高知名度，并提到了相关的先前工作 (ezm7)，该工作实现了类似的存储压缩而非运行时压缩。大家公认分块/分段的低比特调色板化（如 8-bit）在许多用例中几乎是无损的，但 DF11 的价值在于需要完全精度或 GPU VRAM 成为瓶颈的场景，而传统的量化在可以牺牲部分精度以换取速度和更小内存占用的情况下仍更具优势。

- liuliu 描述了由 Draw Things 开发的一种名为 ezm7 的类似压缩方法，该方法将尾数（mantissa）和指数（exponent）的压缩分开（使用 7-bit 尾数），并实现了相当的压缩率（10-11 bits）。然而，ezm7 目前被用作存储格式，而非用于运行时推理。评论还提到了块调色板化（block palettization）方法——例如 8-bit 块调色板化和带有 scale 的 block fp8（GGUF 中的 q8_0）——这些方法在大多数情况下被认为是近乎无损的，为实际的运行时方案提供了替代选择。
- ResidentPositive4122 分享了一个比较 fp8 和 bf16 精度的实证基准测试，指出对于 r1-distill-7b/14b 模型，在 100 道奥数题上，两种精度在 ctx_len 为 16,000 的五次运行中输出了完全相同的结果，这表明 fp8 的精度损失可以忽略不计。相比之下，int4 量化显示出明显的性能下降，这进一步证明了 fp8 即使在精度大幅降低的情况下，仍能保留 bf16 的大部分准确性。
- gofiend 指出 DF11 压缩提供了极具吸引力的准确性与推理权衡，使得在 QAT（量化感知训练）期间可以直接考虑内存和推理成本的优化目标。他们建议与 llama.cpp 或 VLLM 集成以获得基于 CUDA 的运行时支持，这将显著扩大该方法的曝光度，并突出其在开源生态系统中的实际优势。

- **[带有开源数据集的 7B 推理 Rust 编程模型](https://huggingface.co/Tesslate/Tessa-Rust-T1-7B-Q8_0-GGUF)** ([Score: 134, Comments: 14](https://www.reddit.com/r/LocalLLaMA/comments/1k7e542/7b_reasoning_rust_coding_model_with_open_dataset/)): **Tesslate 发布了一个新的开源数据集和 7B 参数的 Rust 编程语言模型 "Tessa-Rust-T1-7B-Q8_0-GGUF"，尽管 Hugging Face 模型页面暂时无法访问（HTTP 429）。该帖子缺乏关于数据集生成、验证和评估过程的技术细节，评论者指出这可能是一个通过提示（prompting）更大模型生成的合成数据集，并对可能缺乏数据质量控制或单元测试表示担忧。** 评论者强调了透明的数据集创建和评估流水线的必要性，并引用了文档齐全的项目，如 Oxen.ai 的 [1.5B Rust 模型](https://www.oxen.ai/blog/training-a-rust-1-5b-coder-lm-with-reinforcement-learning-grpo) 和 Together.ai 的 [流水线披露](https://www.together.ai/blog/deepcoder)。此外，人们还好奇专注于单一语言（Rust）是否比更广泛的多语言模型带来更好的算法性能。

    - 人们对数据集质量存在强烈的质疑，因为没有关于它是如何策划、验证或测试的细节。主要的怀疑是该数据集是通过提示更大模型来回答 Rust 编程问题而生成的，且在单元测试、评估标准或正确性方面缺乏透明度。人们将其与 Oxen.ai 的 [Qwen Coder 1.5B 项目](https://www.oxen.ai/blog/training-a-rust-1-5b-coder-lm-with-reinforcement-learning-grpo) 和 Together.ai 的 [DeepCoder 流水线](https://www.together.ai/blog/deepcoder) 等努力进行了比较，这些项目提供了详尽的文档和开源流水线——这些做法被认为是建立对模型和数据集质量信心的必要条件。
    - 一个技术上的好奇点是，这个专门的 7B Rust 模型与更大、更通用的编程模型在各种任务和领域中的表现相比如何。具体问题在于，语言专业化（深耕 Rust）是否比在多种编程语言混合训练的模型产生更好的推理或编码结果，以及接触更多语言是否能促进对算法和架构模式更广泛的理解。

### 2. 开源本地 LLM & AI App 构建工具 (Dyad)

  - **[我构建了一个免费、本地开源的 lovable/v0/bolt 替代方案……现在支持本地模型！](https://v.redd.it/krhz58lqcvwe1)** ([Score: 209, Comments: 40](https://www.reddit.com/r/LocalLLaMA/comments/1k76ztc/i_built_a_free_local_opensource_alternative_to/)): **该帖子宣布发布 Dyad ([site](https://dyad.sh/), [GitHub](https://github.com/dyad-sh/dyad))，这是一个免费、完全本地化的开源 AI 应用构建工具，旨在作为 v0、Lovable 和 Bolt 等闭源工具的直接替代方案。Dyad 的独特之处在于通过 Ollama 集成提供本地模型支持（参见 [docs](https://www.dyad.sh/docs/guides/ai-models/local-models)），并支持远程模型（如 Gemini Pro 2.5, OpenAI）的 BYO API key 功能，强调通过无缝的本地代码编辑来增强开发者工作流。该版本针对 Mac 和 Windows，并计划通过 Electron 配置支持 Ubuntu，目前正根据反馈持续扩展（例如 MCP 协议支持、LM Studio 集成）。** 评论中具有技术实质的用户请求包括为本地模型添加 OpenAI API 支持以及稳健的 MCP 协议集成，这些被认为是与竞争对手本地 IDE/编程 Agent 实现功能对等和易用性的关键。

    - 一位用户请求实现完善的 MCP (Model Control Protocol) 支持，并强调了现有竞争方案（如 Roo Code 和 Cline）中 MCP 支持几乎不可用的问题。稳健的 MCP 集成可能成为该项目功能集中的一个关键差异化因素，并提升其与复杂工作流的兼容性。
    - 用户对更广泛的本地模型兼容性有需求，特别是将 LM Studio 作为后端，这将有助于在本地运行各种开源模型，并提高相比目前仅限于 Ollama 部署的灵活性。
    - 另一位用户请求为本地模型提供 OpenAI API 支持，指出许多本地模型 UI 缺失的一个理想功能：在远程 API 使用和本地推理之间无缝切换的能力，以增加灵活性并控制成本。

  - **[Gemma 3 伪造（并忽略）system prompt](https://i.redd.it/xuycbwnk4zwe1.jpeg)** ([Score: 201, Comments: 60](https://www.reddit.com/r/LocalLLaMA/comments/1k7krlm/gemma_3_fakes_and_ignores_the_system_prompt/)): **附图显示 Gemma 3 在用户指出其忽略 system prompt 时做出了轻率的回应，凸显了一个关键的设计缺陷：尽管 chat template 支持 'system' 角色，但 Gemma 3 仅通过将系统指令附加到第一个用户消息中而不使用特殊 token 来注入指令，这使其与普通用户输入无法区分，因此经常被忽略（参见模板：https://huggingface.co/google/gemma-3-27b-it/blob/main/chat_template.json）。这反映出 Gemma 3 并没有经过真正的 system prompt 条件训练（如 model card 所确认），因此大多数界面只是将 system prompt 预置到用户输入中，降低了对于需要严格遵守指令的应用的可靠性。** 一场显著的技术辩论随之展开：虽然一些用户对 Gemma 3 缺乏真实的 system prompt 处理感到不满，但其他用户报告称，在处理高 context 的创意任务时，Gemma 3 的指令遵循能力优于更大的模型，这表明实际效果因工作负载和 prompt 风格而异。

    - Gemma 3 原生不支持 system prompt；这在模型文档和 model card 中有明确说明。根据 [Hugging Face 实现](https://huggingface.co/)，界面通过简单地将任何 system prompt 预置到用户输入中来绕过此限制，而不是利用实际的系统角色或编码。如果用户期望系统级行为，这可能会导致困惑，因为 Gemma 的架构中并不存在这种行为。
    - 根据 [Gemma 文档](https://ai.google.dev/gemma/docs/core/prompt-structure)，该模型仅支持 'user' 和 'model' 角色，不支持 'system'。虽然一些用户报告在自定义 prompt 模板（例如用于角色扮演）方面取得了适度成功，但这并不是官方支持或稳健的功能。尝试引入 system prompt 可能会产生不一致的结果，因为底层模型并未针对此类 prompt 进行 instruction-tuned。
    - 在实践中，有用户报告称，在处理大量 context 和遵循某些用例（如小说写作）的复杂详细指令方面，Gemma 3（尤其是 12B 和 27B 变体）的表现优于各种 70B 和 32B 等更大型的模型。尽管缺乏正式的 system prompt 支持，但 Gemma 3 在这些用户面前展示了卓越的指令遵循度和连贯性，凸显了其在官方限制下的实际 prompt 处理优势。

## 其他 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo

### 1. CivitAI 争议与模型托管替代方案

  - **[CivitAI 完蛋了，原因如下](https://www.reddit.com/r/StableDiffusion/comments/1k7p5uw/civitai_is_toast_and_here_is_why/)** ([评分: 115, 评论: 139](https://www.reddit.com/r/StableDiffusion/comments/1k7p5uw/civitai_is_toast_and_here_is_why/)): **该帖子分析了 CivitAI（一家商业 AI 图像共享网站）面临的威胁，原因是日益增长的支付处理器压力——特别是来自 Visa 和 Mastercard——这与 Patreon、Pixiv Fanbox 和 DeviantArt 的打压行动如出一辙。作者认为，CivitAI 目前的内容审核工作（移除边缘癖好内容、部分审核）可能不足以安抚支付提供商，因为当涉及疑似未成年或其它争议内容时，这些提供商会执行广泛的禁令，主要是为了限制法律/商业风险，而非出于道德考量。** 评论者强调了 Visa/Mastercard 作为事实上的行业监管者的中心化权力，并将其与 OnlyFans 在类似压力下的政策变更经历进行了类比。关于支付寡头是否应该为在线平台设定标准存在技术辩论，批评者认为他们规避风险的做法对数字内容和 AI 生态系统产生了广泛影响。

    - 一项技术讨论指出，CivitAI 等平台移除 NSFW 和其他“边缘癖好”内容，主要是由 Visa 和 Mastercard 等公司的支付处理器政策驱动的，这些公司通过威胁切断金融服务来对许可内容施加重大控制。这得到了 OnlyFans 等平台被迫做出类似改变的参考佐证，说明一旦 AI 模型网站规模达到一定程度，监管和财务压力就变得不可避免。
    - 关于混合内容模型共享网站的长期可行性存在争议。一个关键的技术观点是，同时托管 SFW 和 NSFW（特别是非法或高风险）模型对平台来说是一种不可持续的风险模型，因为支付处理器、监管机构和投资者可能会迫使平台做出二选一的决定：要么全面拥抱，要么彻底禁止成人内容，以维持运营稳定。
    - 评论者认为，随着 AI 模型共享平台的增长，它们将受到监管机构和财务支持者日益严格的审查，这使得“西部荒野”式的内容审核方式难以为继。这不仅包括支付处理风险，还包括分发争议性或潜在非法模型时的合规性、声誉和法律风险，特别是那些涉及儿童、名人或未经授权人员描绘的模型。

  - **[关于 Civitai 移除模型](https://www.reddit.com/r/StableDiffusion/comments/1k7dvfb/in_reguards_to_civitai_removing_models/)** ([评分: 150, 评论: 29](https://www.reddit.com/r/StableDiffusion/comments/1k7dvfb/in_reguards_to_civitai_removing_models/)): **该帖子提供了一份托管和共享 Stable Diffusion 模型的 Civitai 替代方案综合列表，包括 Tensor.art、Huggingface.co、ModelScope.cn（侧重中国）、Prompthero.com、Pixai.art、Seaart.ai 和 civitarc.com 等主要平台，以及 ThinkDiffusion 和 Stablecog 等其他新兴或小众服务。它引用了积极维护的精选列表（例如 awesome-stable-diffusion、awesome-diffusion-categorized、Awesome-Video-Diffusion-Models）以获取最新资源，涵盖研究论文、软件、社区中心和支持 API 的网站，并特别关注 Civitai 移除后的模型保留情况，以及以社区参与度、速度或独特 LoRA 支持著称的平台（例如 liblib.art、shakker.ai）。另一个表格评估了适合分享 AI 生成图像并保留元数据 (EXIF) 的图像托管网站，并附带了关于使用 exiftool 等工具进行操作的技术说明，以及 exif.tools 等专用检查资源。** 热门评论指出了 Tensor.art 的潜在问题，特别是未经授权转载 LoRA 模型且缺乏审核，引发了对该平台信任度和内容完整性的担忧。其他评论提到 Reddit 过滤器经常删除类似的资源共享帖子，因此激励用户将资源文本保存到外部。

- 有用户声称 Tensor 托管了盗取的 LoRA 模型，并允许未经授权的转贴，指责版主要么忽视，要么不执行关于模型所有权和来源的规则。这引发了人们对信任 Tensor 等网站来寻找或分发 AI 模型的担忧，特别是与那些拥有更严格审核和策展政策的社区相比。
- 针对 Civitai 移除模型的情况，出现了替代的模型托管网站，并分享了 civitarc.com 的链接作为最近的仓库。这反映了社区在官方平台限制或下架内容时，为维持模型访问所做的持续努力。

- **[Civit Arc，一个图像生成模型的开放数据库](https://www.civitarc.com)** ([Score: 275, Comments: 81](https://www.reddit.com/r/StableDiffusion/comments/1k7po5a/civit_arc_an_open_database_of_image_gen_models/)): **Diffusion Arc（原名 Civit Arc）是一个新推出的、无审查的数据库，专门用于 AI 图像生成模型（例如 Stable Diffusion, Flux），旨在应对 CivitAI 最近不明原因的模型移除。该平台提供不受限制的模型文件浏览、上传和下载，未来的技术计划包括支持 Torrent 上传、模型版本控制以及增强贡献者的信任信号。该网站 ([Diffusion Arc](https://www.civitarc.com//)) 旨在最终实现开源，目前通过捐赠维持运行，目标是为生成式 AI 生态系统提供最大的兼容性和可访问性。** 评论者对该平台在允许 NSFW/成人内容的同时使用 Stripe 进行支付处理表示担忧，理由是支付提供商可能会有相关限制。还有建议实施标记，标明从其他平台禁用的模型，为争议性或被下架的内容增加可追溯性。

    - 多位评论者强调了基于 Torrent 分发存档的重要性，并强烈倾向于使用 Magnet 链接而非托管的 Torrent 文件。Magnet 链接允许模型哈希值独立于任何特定服务器或主机进行重新分享，增强了韧性和去中心化的模型访问——这对于面临移除或审查风险的内容至关重要。
    - 对未来平台功能提出了建议：为所有上传自动生成 Torrent/Magnet（服务器始终做种）、改进模型版本控制，以及允许创作者认领或将之前发布的模型重新链接到单个账户的系统，以便进行适当的归属和持续维护。这些功能将有助于在不断演进的模型数据库中进行分发和内容管理。
    - 一位用户对支付处理器的兼容性提出担忧，指出像 Stripe 这样的服务通常会限制托管 NSFW/成人内容网站的使用，随着平台的发展，这可能会影响资金或捐赠。这突显了如果项目依赖主流支付提供商可能面临的基础设施风险。


### 2. 最近 OpenAI 模型发布问题与策略

- **[o3 的幻觉率高达 33%？为什么这没成为大新闻？](https://www.reddit.com/r/OpenAI/comments/1k7pl37/o3_hallucinates_33_of_the_time_why_isnt_this/)** ([Score: 261, Comments: 77](https://www.reddit.com/r/OpenAI/comments/1k7pl37/o3_hallucinates_33_of_the_time_why_isnt_this/)): **据报道，OpenAI 最新的“o3”推理模型在 PersonQA 基准测试中，有 `33%` 的问题会出现幻觉——这是一个针对人物事实推理的专有、对抗性生成的数据库 ([来源](https://techcrunch.com/2025/04/18/openais-new-reasoning-ai-models-hallucinate-more/))。根据 OpenAI 的内部研究，这一比例比之前的幻觉率翻了一倍多。标题中的数据专门针对对抗性提示词，而非通用场景；现实世界中的提示词幻觉率可能会更低。** 评论者澄清说，33% 的幻觉率仅在专门的对抗性评估集 (PersonQA) 中观察到，而不是典型的用户提示词，并强调将其推广到整体模型性能是不准确的。

    - OpenAI 报告称，GPT-4o (o3) 在 33% 的案例中出现了幻觉，特别是在 PersonQA 基准测试中，该测试是为评估模型关于人物的知识而量身定制的，而非通用提示词。这是一个对抗性集合，因此该比率并不反映平均使用案例。
    - 基准测试特定的幻觉率与现实世界性能之间存在明显的技术区别：33% 的统计数据不适用于所有提示词，而是适用于旨在诱导错误的特定集合，一些用户将其误解为整体失败率。
    - 几位用户强调，o3 产生幻觉的倾向——特别是与之前的 o1 和 o3 mini 等模型相比——显著影响了其实用性，导致一些人重新评估之前包含无限量、更可靠模型的订阅服务的价值。

- **[OS 模型将于 6 月或 7 月发布？](https://i.redd.it/3k1g4j7wr0xe1.jpeg)** ([Score: 146, Comments: 36](https://www.reddit.com/r/OpenAI/comments/1k7rbjm/os_model_coming_in_june_or_july/)): **图片显示 Sam Altman 在回答关于即将发布的开源模型发布日期的问题时，给出了神秘的回复（'heat waves'），用户将其解读为夏季发布（6 月或 7 月），参考了歌曲《Heat Waves》（发布于 6 月 29 日）。上下文还暗示了 'o4 mini >> o3'，表明 o4 mini 模型相比 o3 有显著的性能提升。** 讨论集中在解读 Altman 的回应——大多数评论者认为这暗示了 6 月或 7 月的时间框架，一些人根据歌曲参考推测是 6 月 29 日发布。没有提供关于该模型的技术基准测试或额外细节。

    - 这些评论中没有技术讨论或与基准测试相关的细节；主要焦点是推测 OS 模型的发布日期，间接提到了 Sam Altman，但没有关于模型或性能的细节。
    - 有一个关于 "siillsmaxxing" 含义的查询，以及一个关于为什么 Sam Altman 可能会推荐 "o4 而非 o3" 的问题，这暗示了模型的演进（例如，从 OpenAI GPT-3 到 GPT-4），但该线程并未详细阐述版本之间的技术差异或提供实现分析。

  - **[制作了一个 ChatGPT “秘籍图”以停止猜测模型、工具和提示词（在此分享）](https://i.redd.it/r0f7hax9jzwe1.png)** ([Score: 614, Comments: 49](https://www.reddit.com/r/ChatGPT/comments/1k7lbhq/made_a_chatgpt_cheat_map_to_stop_guessing_models/)): **链接的图片（[ChatGPT Cheat Map](https://i.redd.it/r0f7hax9jzwe1.png)）是一个实用的、以用户为中心的流程图，引导 ChatGPT 用户完成三个关键步骤：(1) 选择模型（默认 GPT-4o 或用于需要更复杂推理任务的 o3），(2) 激活功能/工具（如 Search、Deep Research、Canvas 或 Create Image），以及 (3) 应用提示词公式以获得最佳结果。该视觉辅助工具旨在帮助日常用户（而非 API 或高级用户）做出快速决策，在无需猜测使用哪种工作流的情况下最大化 ChatGPT 的效用。它整合了来自 OpenAI 文档和社区实验的最佳实践，以简化用户体验并提高输出质量。** 评论中的技术讨论询问了为何排除其他可用模型（如 4.5、o4 mini 变体），澄清了 App 和网页界面之间的功能可用性差异，并寻求对适合 o3 模型的“复杂”任务构成的具体定义。

    - 有一个关于 GPT-4.5、o4 mini、o4 mini high、GPT-4 和 4o mini 等模型之间差异的问题，表明了对模型功能和性能的困惑。虽然没有引用直接的基准测试或对比，但该询问表明需要对众多 GPT 子变体的功能进行更严格的文档说明。
    - 一条技术说明澄清了某些功能（如 "Canvas" 或 "Create image"）在 ChatGPT 移动端 App 界面中不可用；相反，用户必须通过文本提示词调用它们，或使用 "/" 命令访问功能菜单，这指出了跨平台的实现差异以及感知界面的提示词编写的重要性。
    - 一位用户询问 GPT-4o 和 o3 模型之间的 "Deep Research 质量" 是否有显著差异，突显了对这些模型之间性能或定性区别的持续不确定性，但讨论中未提供具体数据或评估。


### 3. Frontier Model Benchmarks and Human vs AI Reasoning

  - **[新的推理基准测试显示专家人类仍然优于尖端 LLM](https://i.redd.it/a6awqhrhmtwe1.jpeg)** ([Score: 128, Comments: 59](https://www.reddit.com/r/singularity/comments/1k7f9dd/new_reasoning_benchmark_where_expert_humans_are/)): **图片展示了来自新 PHYBench 基准测试的结果，该测试使用 500 个真实世界的物理问题测试物理推理。人类专家在准确率和 EED (Explanation Edit Distance) 分数上都显著优于最新的 LLM，突显了 LLM 在执行深度或空间物理推理能力方面持久存在的差距。这支持了目前的 LLM 缺乏许多物理任务所需的图表或空间推理机制的观点。** 评论者一致认为，真正的物理推理，尤其是空间/图表思维，至关重要且目前 LLM 尚不具备。鉴于运行合成基准测试的成本日益增加，还有关于转向评估真实世界、具身性能的基准测试价值的讨论。

- 几条评论强调，目前的 LLM 缺乏对专家级人类问题解决至关重要的空间或图表推理能力，特别是在物理和代码架构（code architecture）等领域。无法生成或解释视觉表示（例如准确显示时钟或处理视觉/空间任务）被认为是现有模型的主要技术限制。
- 另一个技术点讨论了近期模型（如 Gemini 2.5 Pro）的快速进步，据报道，其 Benchmark 分数在几个月内从 `25-35` 跃升至 `50`。讨论推测，如果这种线性或近乎指数级的增长速度持续下去，LLM 可能会在几个月内在该 Benchmark 上超越人类专家，这表明模型能力正在显著加速。
- 还有人呼吁将 Benchmark 的重点转向衡量实际的现实生活任务表现，而不是仅仅关注 LLM 仍然显著落后的领域。评论者指出，由于大规模 Benchmark 的运行成本已经变得很高，现实世界的评估现在可能变得更加可行，且对于追踪模型的真实进展更有价值。

- **[新论文：AI 视觉正变得与人类视觉截然不同](https://www.reddit.com/r/singularity/comments/1k7dwld/new_paper_ai_vision_is_becoming_fundamentally/)** ([Score: 167, Comments: 39](https://www.reddit.com/r/singularity/comments/1k7dwld/new_paper_ai_vision_is_becoming_fundamentally/)): **这篇 arXiv 论文 ([2504.16940](https://arxiv.org/pdf/2504.16940)) 表明，随着最先进的深度神经网络（DNNs）——包括具有视觉能力的 LLM，如 GPT-4o、Claude 3 和 Gemini 2——在视觉任务上的进步，它们的内部处理策略与灵长类动物（包括人类）视觉的差异越来越大，扭转了早期的对齐发现。研究将这种差异归因于 DNNs 利用非生物特征和策略来提高性能，这意味着 Benchmark 上的高准确率不再等同于生物学上的合理性。作者认为，要在 AI 中实现类人视觉，需要动态的、具有时间结构的、多模态的和具身（embodied）的训练——例如，使用由 NeRFs 或 Gaussian Splatting 生成的合成环境——而不是依赖于大规模的静态数据集。** 评论者强调，AI 应该从动态的、类生活的经验中学习，而不是从静态数据中学习，并指出进化产生了各种各样的视觉系统，这表明 AI 的最优性可能由于约束较少（例如动物的能量成本或视野限制）而从根本上不同于生物视觉。

    - VallenValiant 将 AI 视觉与生物进化进行了对比，指出动物中独立进化出了多种形式的眼睛。他们认为 AI 可能不会采用“类人”的最优视觉，并解释说机器人或数字视觉系统有可能实现动物由于生物和代谢限制而在物理上无法实现的视角（如 360 度或多向视觉）。这突显了生物系统和 AI 系统在感官优化方面根本不同的路径，指向了 AI 视觉发展出不受人类偏见或进化压力限制的能力的潜力。
    - liqui_date_me 指出，AI 和机器视觉与人类感知存在根本差异的概念并不是什么新鲜发现，并引用了深度学习中的对抗样本（adversarial examples）。这些对抗样本利用了 AI 解释视觉数据的非人类方式，说明了 AI 如何轻易被人类无法察觉的变化所愚弄，并表明人类视觉与机器视觉之间的差异自深度学习研究早期以来就在机器学习社区中得到了广泛认可。

- **[AGI 的终极图灵测试是 MMO 游戏](https://www.reddit.com/r/singularity/comments/1k7m5ui/the_ultimate_turing_test_for_agi_is_mmo_games/)** ([Score: 124, Comments: 53](https://www.reddit.com/r/singularity/comments/1k7m5ui/the_ultimate_turing_test_for_agi_is_mmo_games/)): **该帖子认为，当前的 LLM/AI 基准测试（例如 MMLU/ImageNet 等静态数据集）无法测试真正的 AGI 能力，并提出开放世界 MMO 游戏是一个更严苛的基准测试，因为它需要同时具备动态视觉推理、原始感官知觉（像素、音频、文本）、演化策略下的元学习（meta-learning）、对抗鲁棒性以及无需预训练的 zero-shot 学习。该测试要求 Agent 完全像人类一样操作——从原始信号中解释世界，适应实时的游戏进展，并在无辅助的情况下学习新策略——本质上是在混乱的多 Agent 数字环境中与人类的经验曲线对齐。** 一位评论者同意游戏（尤其是像 Dwarf Fortress 这样复杂的沙盒游戏）在 AI 评估方面优于当前的基准测试；另一位指出，实施此类测试与其说是 AI 问题，不如说是一个巨大的软件工程挑战，因为当前的 Agent 需要大量特定于环境的代码才能参与。第三位指出，这种方法反映了现有的机器人仿真训练，即 Agent 在复杂的多 Agent 环境中学习行为，这暗示了已有先例，但在扩展到 MMO 的丰富性方面存在技术难度。

    - 游戏环境，特别是像 MMO 游戏这样复杂的环境，为 AI 研究提供了宝贵且具有挑战性的测试平台，因为 Agent 需要学习、保留并应用新信息来解决问题，并具备实时适应和多 Agent 交互能力。Minecraft 被引用为一个起点，但 Dwarf Fortress 被提及为一个特别丰富的 AI 环境，因为它具有复杂的模拟和涌现的复杂性。 
    - 几条评论强调，在 MMO 环境中实现 AI 通常涉及大量特定于环境的编码。这可能会使“测试”偏向于程序员创建令人信服的脚本和 Agent 的能力，而不是真正评估 AI 的泛化能力。这种批评表明，此类设置衡量的是编码者的聪明才智，而非通用的 AI 性能。
    - 模拟多 Agent 环境的工作正在进行中，例如在具有许多 Agent 的模拟中训练机器人 AI，它们必须学习诸如避障和从干扰中恢复等行为。EVE Online 的公开 API 被建议作为在丰富的 MMO 背景下测试 AI Agent 的平台，并且提到了过去的工作（例如 Claude 玩 Pokemon），展示了 LLM 和 AI 与游戏环境的集成。


---

# AI Discord Recap

> 由 Gemini 2.5 Flash Preview 提供的摘要之摘要的摘要

**主题 1. 模型更新与性能变化**

- **O3 输出更大规模的代码！**：用户报告 **O3** 现在输出的代码文件可达 **700-1000 行**，是之前 **300 行** 限制的两倍。这一提升也扩展到了 **O4 Mini High**，增强了其能力。
- **Sunstrike 进入竞技场，引发猜测**：新的 **Sunstrike** 模型已添加到竞技场，初始表现为 *claude-3-7-sonnet-20250219-thinking-32k > sunstrike > gemma-3-12b-it*。早期迹象表明 **Sunstrike** 可能是 Google 的模型，面临着即使是 **2.5 Pro** 也只能在 **25%** 的时间内解决的挑战。
- **GLM-4 进入 HF 竞技场**：成员们正在讨论新的 **GLM 4 模型**，模型已上传至 HF [此处](https://huggingface.co/unsloth/GLM-4-32B-0414-GGUF)。有人强调 [THUDM/GLM-Z1-32B-0414](https://huggingface.co/THUDM/GLM-Z1-32B-0414) *即使面对 DeepSeek R1，在半数基准测试中也名列前茅*。

**主题 2. 硬件与优化技术**

- **LM Studio 获得 RTX 50 系列性能提升**: **LM Studio 0.3.15** 引入了对 **NVIDIA RTX 50-series** (CUDA 12.8) 的支持，可以通过 [LM Studio 下载](https://lmstudio.ai/download) 进行更新。此更新还在 llama.cpp 和 MLX 中激活了 **GLM-4**，采用了全新的 system prompt 编辑器 UI，并在类 OpenAI API 中集成了 `tool_choice` 参数，详情见 [完整发布说明](https://lmstudio.ai/blog/lmstudio-v0.3.15)。
- **基于 CUDA 的 RTX 3060 完胜 Intel B580**: 成员们对比了 **RTX 3060 12GB** 和 **Intel B580 12GB** 在 AI 任务中的表现，因其卓越的 **CUDA** 支持而更青睐 **3060**。共识是 *AI 世界的一切都是围绕 CUDA 构建的*，这使得 Nvidia 的显卡在 AI 开发和实验中更具优势。
- **TACQ 将 LLMs 压缩至 2-bit**: 一篇 [研究论文](https://www.marktechpost.com/2025/04/22/llms-can-now-retain-high-accuracy-at-2-bit-precision-researchers-from-unc-chapel-hill-introduce-tacq-a-task-aware-quantization-approach-that-preserves-critical-weight-circuits-for-compression-withou/) 介绍了 **TACQ**，这是一种针对 **LLMs** 的任务感知量化方法，能在 **2-bit 精度** 下保持高准确度。TACQ 通过使用校准数据集来决定哪些权重可以被进一步压缩，从而保留关键的权重电路。

**主题 3. AI 框架与工具更新**

- **Aider 添加基于语言的排序**: **Aider** 现在支持按编程语言排序，提高了代码组织和可访问性。查看 [UI 截图](https://cdn.discordapp.com/attachments/1131200896827654144/1365176535484596327/image.png?ex=680d03f9&is=680bb279&hm=accbfb4dd2576f4bc500e5e11cdf24b69189951769521c7cbb2980fee2cd03e0&)。这是通过一种新的按语言排序功能实现的。
- **`FunctionAgent` 现在提供超时功能**: [`FunctionCallingAgent`](https://llama_index.readthedocs.io/en/stable/understanding/agent/) 类缺乏直接的超时参数，但在 LLM 对象上为每个 LLM 调用设置超时是可行的。较新的 Agent 类（如 `FunctionAgent`）直接支持通过 `request_timeout` 参数设置超时，并提供了一个使用 `Ollama` 并将 `request_timeout` 设置为 360 秒的代码片段示例。
- **Kubernetes MCP 获得实现**: 一位成员宣布他们构建了一个新的 [基于 k8s API](https://github.com/StacklokLabs/mkp) 的 kubernetes MCP，使其更具通用性和灵活性。它旨在作为第一个用于获取 GitHub repo 的 MCP Server 的替代方案，允许 **AI** 使用仓库代码作为参考。

**主题 4. AI 研究与基础概念**

- **LeCun 规划机器智能路径**: 一位成员提出了一条通往 AGI 的路径，利用 **过往记忆**、更新的知识图谱和 **想象力** 来生成多媒体内容。另一位成员建议阅读 **Yann LeCun** 的 *A Path Towards Machine Intelligence* [openreview.net](https://openreview.net/pdf?id=BZ5a1r-kVsf)，强调潜空间（latent space）变换足以实现智能，而无需生成环节。
- **Agent 构建讨论正盛**: [AnthropicAI 发布了 Building Effective Agents](https://www.anthropic.com/)，**dexhorthy** 的 [12 Factor Agents](https://www.12factor.net/) 走红，且 [OpenAI 发布了 A Practical Guide To Building Agents](https://platform.openai.com/docs/guides/function-calling)。社区正在就什么是 Agent 以及如何构建它们进行 **富有成效的对话**。
- **SimpleStories 数据集超越 TinyStories**: 一个名为 **SimpleStories** 的新替代 [数据集](https://huggingface.co/datasets/lennart-finke/SimpleStories)、分词器（tokenizer）和 [模型套件](https://huggingface.co/SimpleStories) 已经发布。一份 [社区文档](https://docs.google.com/document/d/1XRb-EDDw-h6c-L6pKltI9A8zJmufE71I2_4wx8d5rMg/edit?usp=sharing) 可供研究人员入门使用。

**主题 5. 行业新闻与平台挑战**

- **Nous 获得 Paradigm 5000 万美元投资**：Nous Research [宣布获得来自加密风险投资公司 Paradigm 的 **5000 万美元**投资](https://fortune.com/crypto/2025/04/25/paradigm-nous-research-crypto-ai-venture-capital-deepseek-openai-blockchain/)，以推进其 **AI 研发**工作。此前，Nous 发起了 **Psyche** 计划，这是一个利用 **Solana 网络**进行协调的[分布式训练项目](https://nousresearch.com/nous-psyche/)，并为对加密领域感兴趣的人士提供了 [Psyche Discord](https://discord.gg/peqZyPRd)。
- **Gemini 的慷慨：免费层级引发限流烦恼！**：由于需求量大，[Gemini 2.5 Pro Experimental](https://openrouter.ai/google/gemini-2.5-pro-exp-03-25) 实施了更严格的使用限制：**每分钟 1 次请求**且**每天总计 1000 次请求**，同时 `:free` 模型别名将指向标准变体以确保现有代码继续运行。成员们讨论了 **Gemini 2.5 Pro** 免费层级的速率限制，指出 OpenRouter 将宣布这一变化，一名成员表示需求已超过供应。
- **额度危机：OpenRouter 账户被清空！**：一位成员报告了一起事件，其 **OpenRouter 额度因涉及无限 URL 生成的漏洞而被耗尽**。该恶意活动被追溯到一个建议的解决方案架构，该架构创建了一个约 **3000 个字符**的 URL。


---

# 第一部分：高层级 Discord 摘要




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 发布全新 AI 套件**：Perplexity 本周发布了 **iOS Voice Assistant**、**GPT Image Generation**、**Grok 3 Beta** 和 **OpenAI o4-mini**，详见其[更新日志](https://www.perplexity.ai/changelog/what-we-shipped-april-25th)。
   - 用户指出这些功能增强了 Perplexity 生态系统内的搜索和交互能力。
- **Discord 角色焕然一新**：成员报告称 **Kes** 正在重做 [Discord 角色链接系统](https://discord.com/channels/link/to/roles)。
   - 然而，关于此次重做的具体益处或变化的细节尚不清楚。
- **GPT-4o 的文本图像能力令人印象深刻**：成员们讨论了 [GPT-4o Image Gen 的语言模型](https://link.to/model)如何生成更出色的**文本图像**，尽管有些人仍然倾向于使用 Midjourney 来获得整体审美质量。
   - 一位成员声称 *MJ 特别希望人们能够通过输入像 cat 这样的单个单词就能获得非常漂亮而非基础的效果*，这暗示了在优先考虑审美方面的架构差异。
- **DeepSearch 的 Reddit 乌龙事件**：在搜索 Reddit 数据时，[一张 AI 生成的图像](https://link.to/img)产生了意想不到的结果，一名成员开玩笑地给 **Perplexity** 颁发了“Perplexity 又崩了奖”。
   - 另一名成员质疑该结果背后的原因，推测这可能是基于 20 世纪初的建议。
- **Grok 3 Mini 的推理能力引发辩论**：成员们争论为什么 Perplexity 在某些应用中选择了普通的 **Grok 3** 模型而不是推理版本。
   - 一名团队成员澄清说，[Grok 3 在阅读、理解和回答问题方面比推理版 Grok 3 mini 表现更好](https://link.to/blogpost)，使其成为特定任务的更合适选择。



---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous 获得 Paradigm 5000 万美元投资**: Nous Research [宣布获得来自加密风险投资公司 Paradigm 的 **5000 万美元**投资](https://fortune.com/crypto/2025/04/25/paradigm-nous-research-crypto-ai-venture-capital-deepseek-openai-blockchain/)，以推进其 **AI 研究与开发**工作。
   - 此消息紧随 Nous 启动 **Psyche** 的倡议之后，这是一个利用 **Solana 网络**进行协调的[分布式训练项目](https://nousresearch.com/nous-psyche/)，并为对加密货币领域感兴趣的人士提供了 [Psyche Discord](https://discord.gg/peqZyPRd)。
- **TACQ 将 LLM 压缩至 2-bit**: 一篇[研究论文](https://www.marktechpost.com/2025/04/22/llms-can-now-retain-high-accuracy-at-2-bit-precision-researchers-from-unc-chapel-hill-introduce-tacq-a-task-aware-quantization-approach-that-preserves-critical-weight-circuits-for-compression-withou/)介绍了一种名为 **TACQ** 的任务感知量化方法，该方法针对 **LLM** 在 **2-bit 精度**下仍能保持高准确率。
   - TACQ 通过使用校准数据集来决定哪些权重可以被进一步压缩，从而保留关键的权重电路。
- **纯 CPU 微调发布**: 一位用户分享了[分步指南](https://medium.com/@contact_30070/step-by-step-guide-for-fine-tuning-your-llm-with-llama-factory-using-the-cpu-only-96b2fc6a80b0)，介绍如何**仅使用 CPU 通过 LLaMa Factory 进行 LoRA 微调 LLM**。
   - 该指南涵盖了从安装 **torch** 到在 **LM Studio** 中加载自定义 **GGUF** 的所有内容，并配有 [YouTube 视频](https://www.youtube.com/watch?v=1bgL4b7VT8M)。
- **寻找强大的 TRL 奖励函数**: 一位成员询问如何寻找经过实战测试且兼容 **TRL** 的 **GRPO 奖励函数**。
   - 另一位成员建议参考 Hugging Face 的 [Open-R1 奖励函数](https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py)作为起点。



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **O3 输出更长的代码！**: 用户报告称 **O3** 现在可以输出更长的代码文件，达到 **700-1000 行**，是之前 **300 行**限制的两倍。
   - 这一提升也扩展到了 **O4 Mini High**，增强了其能力。
- **LeCun 规划机器智能路径**: 一位成员提出了一条通往 AGI 的路径，利用**过去记忆**、更新的**知识图谱**和**想象力**来生成多媒体内容。
   - 另一位成员建议阅读 **Yann LeCun** 的 *A Path Towards Machine Intelligence* [openreview.net](https://openreview.net/pdf?id=BZ5a1r-kVsf)，该文强调潜在空间变换足以实现智能，而无需生成环节。
- **Sunstrike 进入竞技场，引发猜测**: 新模型 **Sunstrike** 已添加到竞技场，初始表现为 *claude-3-7-sonnet-20250219-thinking-32k > sunstrike > gemma-3-12b-it*。
   - 早期迹象表明 **Sunstrike** 可能是一个 Google 模型，它面临着即使是 **2.5 Pro** 也只能在 **25%** 的时间内解决的挑战。
- **GPT-4o 的多模态特性引发辩论**: 成员们正在辩论 **GPT-4o** 是否是真正的原生多模态，焦点在于它对工具的依赖与固有能力。
   - 论点认为 **GPT-4o** 可能不是完全的多模态，因为它使用特定的自然语言查询进行工具调用并生成全新的图像，但它*过去调用 DALL-E，而现在是原生生成图像*。



---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 双倍积分：为期一个月的狂欢**：Manus 为新老订阅者提供首月**双倍积分**：Starter 计划获得 **+3,900 积分**，Pro 计划获得 **+19,900 积分**。
   - 这是一个*限时优惠*，没有固定截止日期，**免费用户**在升级到付费计划时，**首次订阅**可获得双倍积分。
- **Manus 制作的 Flask 克隆版上线 GitHub**：一位成员创建了一个 [Discord Flask 克隆版](https://github.com/johnnyshelby123/discordflaskclone)，并交由 Manus 进行改进。
   - 原作者强调 **Manus 完成了 100% 的工作**，即便另一位成员提议用 **Node.js** 重新编写。
- **用于项目生成的提示词工程 (Prompt Engineering)**：一位成员分享了[他们的提示词工程方法](https://github.com/justinlietz93/hierarchical_reasoning_generator/tree/main/hierarchical_planner/persona_builder/personas)，旨在用极少的代码生成整个项目。
   - 他们指出了分层系统化和动态网关，根据生成的初始计划调整 Persona 或 Prompt，能够通过一个 Prompt 生成代码量少于 1000 行的完整项目。
- **用户渴望固定费用乌托邦**：用户对 Manus 的**积分制系统**表示不满，更倾向于支付固定的月费以获得无限使用权。
   - 一名用户在因 Manus 的错误而损失付费积分后，转回使用 **ChatGPT 4.0**。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Sora 定价曝光**：一位用户询问了 **Sora** 的图像生成限制，并发现了以更高成本购买视频生成和并发图像生成的新选项。
   - 该用户最初面临一次只能生成一张图像的限制，并收到提示要求*升级*到他们已经拥有的计划。
- **RAG、LLM 与 Word2Vec 组合讨论**：一位成员发起了关于结合 **RAG、LLM 和 Word2Vec** 的讨论，指出许多中国大陆公司正在开发具有*本地知识库 (Local Knowledge Base)* 功能的软件。
   - 讨论中提到了对 **LLM 掌握上下文长度**的担忧，可能需要对原始文档进行进一步的段落级处理，引发了对该技术前景的疑问。
- **数学模型微调热潮**：一位用户建议**专门针对大学水平的数学问题微调较小的 AI 模型**，而不是依赖于在全量数据上训练的大型模型。
   - 其核心思想是创建一个*大小足以理解问题语言*的模型，并使用 Python 集成内置计算器，从而优化速度和成本效益。
- **Gemini Advanced 视频限制**：一位用户指出 **Gemini Advanced 中视频生成限制**的不一致性，指出其每天仅提供 **10 个视频**，而 AI Studio 免费版每天提供 **4 个**。
   - 尽管有限制，他们认为 **Gemini** 的 **Veo 2** 在通用内容上的输出质量优于 **Sora**，但限制过多且存在延迟拒绝的问题。
- **Deep Research 分级详情**：根据[这条推文](https://x.com/OpenAI/status/1915505961500070245)，**OpenAI Pro** 用户可获得 **120 次原始 Deep Research** 和 **120 次轻量级 Deep Research**。
   - 用户看到了 **Deep Research full**、**Deep Research lite**、**Deep Research mini**（即普通的 o3）和 **Deep Research nano**（即普通的 o4-mini）。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 发布 Dynamic v2.0 GGUFs**: Unsloth 在[此处](https://x.com/UnslothAI/status/1915476692786962441)发布了新的 **Dynamic v2.0 GGUFs**。
   - *官方*多 GPU 支持即将推出，但*需要大量测试来确认其稳定性*。
- **GLM-4 进入 HF Arena**: 成员们正在讨论新的 **GLM 4 模型**，模型已上传至 HF，链接在[此处](https://huggingface.co/unsloth/GLM-4-32B-0414-GGUF)。
   - 有人指出 [THUDM/GLM-Z1-32B-0414](https://huggingface.co/THUDM/GLM-Z1-32B-0414) *在多项基准测试中名列前茅，甚至优于 DeepSeek R1*。
- **Tesslate 模型引发 Rust CodeFIM 讨论**: 一名成员请求对 **Tesslate** 新的 **Rust 7B 模型**进行量化，促使创建了其 codefim [数据集](https://huggingface.co/datasets/Etherll/CodeFIM-Data)的纯 Rust 版本。
   - 随后有人提出疑问：*是否有人已经将 phi-4 编译成 GGUF 了？*。
- **MoonshotAI 发布 Kimi Audio**: **MoonshotAI** 发布了 [Kimi Audio](https://github.com/MoonshotAI/Kimi-Audio)，论文已在 [arXiv](https://arxiv.org/abs/2504.09858) 上线。
   - 根据论文，新模型可以处理音频。
- **Gemma-3-4b-it 微调面临障碍**: 一名用户报告在 Google Colab 上通过 Unsloth notebook 微调 `gemma-3-4b-it` 模型时出现 `RuntimeError`，遇到了 *float 与 c10::Half 之间的数据类型不匹配*。
   - 即使撤销了所有更改，错误依然存在，这表明可能存在环境问题。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Llama-3.2 Checkpoint 导出困难**: 一名成员正在寻找将 **Llama-3.2 checkpoints** 导出为支持 **KV cache** 和 **TensorRT** 的 **ONNX** 格式的脚本。
   - 他们报告称，在尝试默认 PyTorch 初始化与特定架构初始化时*几乎没有收益*，并询问在 Windows 或 WSL 上是否可以将 NVIDIA 驱动程序置于 TCC 模式。
- **Colaboratory 缓解数字鸿沟**: 一名成员建议使用 [Google Colaboratory](https://colab.google) 进行轻量级数据分析，以应对不断上涨的电脑配置成本。
   - 他们指出，这仅限于 **1B 参数**左右的模型，如 **GPT-2**。
- **魔改版 RTX 4090 拥有 48GB 显存**: 成员们讨论了 **魔改版 48GB RTX 4090** 的流行，特别是在中国，并链接了来自 Tom's Hardware 的[一篇拆解文章](https://www.tomshardware.com/pc-components/gpus/blower-style-rtx-4090-48gb-teardown-reveals-dual-sided-memory-configuration-pcb-design-echoes-the-rtx-3090)。
   - 这些显卡售价通常在 *3,500 美元* 左右，不过由于供应紧张，关于 **5090** 的信息尚未确认。
- **Flash Linear Attention 模型**: 成员们讨论了在微小数据集上训练最简单的 Linear Transformer 模型或等效模型的需求。
   - 另一名成员建议使用来自 **Flash Linear Attention 仓库**的模型，例如 **Gated Linear Attention** 或 **Gated DeltaNet**，并使用他们的 **Flame trainer** 进行训练。
- **SimpleStories 数据集超越 TinyStories**: 发布了一个名为 **SimpleStories** 的新替代[数据集](https://huggingface.co/datasets/lennart-finke/SimpleStories)、分词器（tokenizer）和[模型套件](https://huggingface.co/SimpleStories)。
   - 现已提供[社区文档](https://docs.google.com/document/d/1XRb-EDDw-h6c-L6pKltI9A8zJmufE71I2_4wx8d5rMg/edit?usp=sharing)以帮助研究人员入门。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider** 新增基于语言的排序：**Aider** 现在支持按编程语言排序，提升了代码组织能力和可访问性。[查看 UI 截图](https://cdn.discordapp.com/attachments/1131200896827654144/1365176535484596327/image.png?ex=680d03f9&is=680bb279&hm=accbfb4dd2576f4bc500e5e11cdf24b69189951769521c7cbb2980fee2cd03e0&)。
   - 这是通过一种新的按语言排序功能实现的。
- **Grok-3-Mini** 价格低于竞争对手：通过 OpenRouter 使用 **Grok 3 mini** 的价格几乎只有 **Deepseek V3** 的一半，且拥有相当的推理性能。查看 [性能对比](https://cdn.discordapp.com/attachments/1131200896827654144/1365185829596696636/image.png?ex=680d0ca1&is=680bbb21&hm=f5228d04256f815b37940355080c80e2cdaa6d86a0f6b37759d6d53b59c8be26&)。
   - 成员们观察到，考虑到价格因素，切换模型可能是值得的。
- **Gemini Pro 2.5** 与 **Claude 3.7** 联手：一位成员正在尝试在 **Aider** 中将 **Gemini Pro 2.5** 作为架构师（architect）模型，将 **Claude 3.7** 作为编辑器（editor）模型，灵感来自展示 **Claude Code** 和通过 MCP 接口使用 **Gemini** 的视频。
   - 然而，有人指出 **Gemini** 倾向于修改无关代码，导致代码审查困难，使用 DeepSeek V3 可能会更好。
- **Gemini** 的注释过于啰嗦：用户正寻求减少 **Gemini** 冗长的注释，建议在使用 **Gemini** 作为架构师模式并配合 **GPT-4.1** 作为编辑器时，使用约定（conventions）来删除不必要的注释。
   - 一位用户提到，“Gemini 在注释中喋喋不休的情况比我见过的任何时候都要严重”，并正在寻找更好的解决方案。
- **Aider** 用户偏爱 CLI：成员们 90% 的时间在终端（Warp.dev）中使用 **Aider**，其余时间在 VS Code 的侧边栏中使用，不使用 IDE 插件。
   - 这表明大多数用户并不使用 IDE 插件。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Framepack** 因微调的可访问性受到关注：成员们强调了 **Framepack** 易用的微调能力，**Unsloth** 的认可暗示了双方的合作，特别是围绕每 **8 秒** 一次的频繁权重更新。
   - 快速的权重更新旨在进行压力测试和持续改进。
- 多模态 LLM 迈向媒体精通：社区探索了如 [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)、[Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) 和 [SmolVLM2](https://huggingface.co/blog/smolvlm2) 等多模态 LLM，用于解释视频片段和音频，甚至还包括 [Nvidia's DAM-3B-Video](https://huggingface.co/nvidia/DAM-3B-Video) 和一个 [YouTube 音频转录器](https://huggingface.co/spaces?sort=trending&search=youtube)。
   - 探索重点在于视频摘要和音频内容转录。
- 推理 API 给产品带来“心理阴影”：一位成员报告称通过 **Hugging Face Inference API** 遇到了令人不安的图像，需要联系 **HF 工作人员** 解决。
   - 澄清指出，**Stable Diffusion** 团队隶属于 **Stability AI**，与 Hugging Face 是独立的。
- **SmolAgents** 询问：它能在本地驱动 Gemma 吗？：一位用户询问关于将 **smolagents 与 Gemma3** 或其他本地 LLM 结合使用的问题，并指出 Deeplearning.ai 的课程使用的是 **Tavily** 和 **OpenAI**。
   - 这个问题触及了本地模型与 Agent 框架集成的程度。
- Agent 课程证书需要 HF 用户名：用户确认 HF 用户名即为**凭证 ID**（credential ID），[数据集链接](https://huggingface.co/datasets/agents-course/course-certificates-of-excellence)即为 **URL**，作为永久验证方法。
   - 这是针对下载链接指向临时文件夹文件而非 CDN URL 问题的变通方案。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **AI 学习 Reward Shaping**：分享了一篇关于利用当前世界模型进行[模拟展开离线训练](https://arxiv.org/abs/2502.05244)的论文，旨在减少与环境的交互。
   - 关于 Reward Shaping 的幽默建议包括：*竖起大拇指表示正向奖励，大拇指向下表示负向奖励*。
- **符号化世界模型（Symbolic World Models）兴起**：一名成员分享了一个 [arXiv 链接](https://arxiv.org/abs/2503.20124)，强调了将程序作为世界模型的优势。
   - 他们强调了其组合特性，与神经表示相比，这种特性允许更轻松地进行修改。
- **LLM 融合进世界模型**：一位成员断言 **LLM 的世界模型**是*融合在它们的权重中*的，并将其与离散或结构化模型进行了对比。
   - 另一位成员引入了一个公式 `(A, W) = G(z)`，用于表示按需生成新的 Agent/模型/世界。
- **生成式 AI 获得了一个公式**：为 **AI 模型生成**引入了一个通用公式 `(A, W) = G(z)`，其中 **A** 是模型的架构（Architecture），**W** 是权重/参数（Weights），**G** 是生成器（Generator），而 **z** 是语义种子（semantic seed）。
   - 该成员建议这涵盖了 Hypernetworks，并可能导致显著的压缩和生成能力，将其比作*存储 DNA 而不是整个人类*。
- **DeepMind 调整音乐 AI**：Google DeepMind 发布了 [Music AI Sandbox](https://deepmind.google/discover/blog/music-ai-sandbox-now-with-new-features-and-broader-access/)，具有新功能和更广泛的访问权限。
   - 博客文章重点介绍了这个音乐 AI 沙盒的各种更新。



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Gemini 的慷慨：免费层级引发限流麻烦！**：由于需求量大，[Gemini 2.5 Pro Experimental](https://openrouter.ai/google/gemini-2.5-pro-exp-03-25) 具有更严格的使用限制：**每分钟 1 次请求**且**每天总计 1000 次请求**，并且 `:free` 模型别名将指向标准变体，以确保现有代码继续工作。
   - 成员们讨论了 **Gemini 2.5 Pro** 免费层级的速率限制，指出 OpenRouter 将宣布这一变化，一名成员表示需求正超过供应。
- **自定义速率限制错误消息即将到来**：用户在达到速率限制时直接收到来自 **Gemini API** 的错误消息，由于消息未指明限制是全局的还是针对特定用户的，因此造成了困惑。
   - OpenRouter 团队正在考虑添加自定义消息，以便为用户澄清速率限制错误的来源。
- **百度模型招手，OpenRouter 会采纳吗？**：一名成员询问了将[百度模型](https://x.com/Baidu_Inc/status/1915603080336597310)添加到 OpenRouter 的可能性。
   - 另一名成员指出了 **DeepSeek** 的现有可用性，随后讨论了由其他提供商托管的 **GLM-4-0414**。
- **O3 之旅：OpenRouter 的验证尝试！**：一名成员询问 OpenRouter 未来是否会在无需验证的情况下支持 **OpenAI 的 o3 和其他高级模型**。
   - 另一名成员提到 **OpenAI** 可能会在未来取消这一要求。
- **额度危机：OpenRouter 账户被清空！**：一名成员报告了一起事件，由于涉及无限 URL 生成的漏洞，他们的 **OpenRouter 额度被耗尽**。
   - 该恶意活动被追踪到一个建议的解决方案架构，该架构创建了一个长度约为 **3000 个字符**的 URL。



---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **ESP32 获得 RTSP 固件包**：一名成员正尝试为 **ESP32** 上的出站流媒体创建一个可用的 **RTSP 固件包**，并征求建议，提到了流媒体的需求。
   - 其他成员推荐了这两个 **GitHub** 仓库：[esp32cam-rtsp](https://github.com/rzeldent/esp32cam-rtsp) 和 [ESP32-RTSPServer](https://github.com/rjsachse/ESP32-RTSPServer)，并建议使用 **PlatformIO with AI** 进行开发。
- **Gemini 2.5 和 Sonnet 3.7 受到质疑**：成员们反映，与通过命令行界面 (**CLI**) 使用相比，非 Max 版本的 **Gemini 2.5** 和 **Sonnet 3.7** 似乎更笨，且幻觉更多。
   - 一位成员推测 **Cursor** 可能会修改系统指令或提示词，但另一位成员反驳称 *"Gemini 的问题应该已经修复，那是 Google 的问题"*。
- **Cursor 账单问题已解决**：一名成员询问由于银行确认延迟导致使用 **Cursor** 时发票未支付的即时付款选项。
   - 另一位成员（可能是 **Cursor** 团队成员）提议通过 **DM** 提供直接协助，以快速解决账单问题。
- **社区希望模型评估开源**：一名成员建议 **Cursor** 应该开源不同模型的评估，并强调模型的实现方式比原始基准测试更重要。
   - 他们提议进行基于工具的评估，特别是询问是否有 Cursor 特有的评估工具，因为 *"我发现模型的实现方式比单纯的原始基准测试更重要。"*



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Python Interop 功能即将到来！**：成员们正期待如之前演示中所提到的，为 Mojo 发布额外的 **Python interop capabilities**。
   - 团队尚未提供发布日期的具体时间表。
- **Mac GPU 支持可能在今年夏天推出**：根据 .duck_tape 的说法，**Mac GPU support** 暂定于 *夏季* 推出，但用户应持保留态度。
   - 成员们正在等待官方确认以及关于兼容性和性能的细节。
- **Rust 的 uom Crate 高效处理单位运算**：Rust 中的 `uom` crate 允许在不发生错误的情况下混合处理 **Energy**、**Force** 和 **Length** 等不同单位的运算。
   - 成员们演示了将 `Energy` 加到 `Force` 和 `Length` 的乘积上，类似于将 `Byte` 视为 `UInt8` 处理。
- **Nm vs Joules 的辩论爆发**：一场关于 **牛米 (Nm)** 是否可以等同于 **焦耳 (Joules)** 的讨论展开了，一方认为物理学上将它们区分开来，因为扭矩是矢量，而能量是标量。
   - 参考 [BIPM](https://www.bipm.org/documents/20126/41483022/SI-Brochure-9-EN.pdf)，讨论强调 *尽管扭矩与能量具有相同的量纲（SI 单位为焦耳），但焦耳从不用于表示扭矩*。
- **`QuantityKind` 标签可能解决物理量歧义**：有人提议使用 **QuantityKind** 标签在类型层面区分相似的单位，例如 **Nm** 和 **Joules**。
   - 实现方式可以是 `Q[Vec[dtype, 1], N * M]` 对比 `Q[Scalar[dtype], N * M]`，这样两者就无法相加。



---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 获得 RTX 50 系列增强**：**LM Studio 0.3.15** 引入了对 **NVIDIA RTX 50 系列** (CUDA 12.8) 的支持，可通过 [LM Studio 下载](https://lmstudio.ai/download) 进行更新。
   - 此次更新还在 llama.cpp 和 MLX 中激活了 **GLM-4**，配备了全新的 system prompt 编辑器 UI，并在类 OpenAI API 中集成了 `tool_choice` 参数，详见 [完整发布说明](https://lmstudio.ai/blog/lmstudio-v0.3.15)。
- **LM Studio 社区预设处于预览阶段**：用户正在探索 **LM Studio** 中新的 **Community Shared Presets**（社区共享预设）功能，该功能目前处于 "Preview" 状态，且缺少用于浏览的 discover 标签页。
   - 关于如何定位和下载预设存在困惑，用户期望有 *discover -> presets* 的导航路径，一位用户表示这 *让我怀疑其意图*。
- **数据集准备难倒了 LLM 角色模拟**：一位用户正尝试使用有限的数据集（**10k tokens**）训练 LLM 以模拟游戏角色，并需要关于如何将数据加载到 **LM Studio** 的指导。
   - 该用户不确定是否必须将数据转换为 **GGUF** 格式，以及转换后测试模型的后续步骤。
- **基于 CUDA 的 RTX 3060 完胜 Intel B580**：成员们对比了 **RTX 3060 12GB** 和 **Intel B580 12GB** 在 AI 任务中的表现，因其卓越的 **CUDA** 支持而更青睐 **3060**。
   - 共识是 *AI 世界的一切都是围绕 CUDA 构建的*，这使得 Nvidia 的显卡在 AI 开发和实验中更具优势。
- **OpenVINO 速度超越 llama.cpp**：一位成员展示了他们的 [OpenArc 项目](https://github.com/SearchSavior/OpenArc)，表明 **OpenVINO** 在 **CPU** 上的性能显著优于 **llama.cpp**。
   - 据该成员称，*从经验来看，差异是巨大的，尤其是 ttftR*，并已为 **Qwen2-VL**、**Qwen2.5-VL** 和 **Gemma3** 实现了视觉功能。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **提议 Pantone-Scale 模式**：一位成员在 [general 频道](https://discord.com/channels/1189498204333543425/1189498205101109300) 幽默地建议将 *grayscale mode*（灰度模式）重命名为 *pantone-scale*。
   - 虽然没有立即做出决定，但该建议受到了好评。
- **FP4 缩放需要逐元素乘法**：在 **5090** 上运行 **FP4** 后，一位成员发现他们需要将 scaling elementwise multiplication（缩放逐元素乘法）融合到他们的 **matmul kernel** 中。
   - 另一位成员指出，在对称量化中，获得准确 scale 的良好算法非常重要，并警告说简单的 min-max 方法可能会导致质量不佳。
- **评估 cuda-python 库**：一位成员在 [beginner 频道](https://discord.com/channels/1189498204333543425/1191300313928433664) 询问了 [NVIDIA 的 cuda-python 库](https://github.com/NVIDIA/cuda-python) 与 **CUDA C/C++** 相比的质量和功能对等性。
   - 该成员对该库在实现功能对等后的潜力表示乐观。
- **MI300 VGPR 影响占用率**：在 **MI300 ISA** 中，使用大量的 **VGPR**（例如 **255** 个）会导致较低的 occupancy（占用率）。
   - 正如在 [rocm 频道](https://discord.com/channels/1189498204333543425/1233704710389764236) 中讨论的那样，较低的 occupancy 会因并行度降低和开销增加而降低性能。
- **MI300 在 AMD-FP8-MM 排行榜上表现出色**：多位成员在 **MI300** 的 `amd-fp8-mm` 排行榜上刷新了个人最佳成绩，时间包括 **4.93 ms**、**2.74 ms**、**2.53 ms**、**5.19 ms**、**5.28 ms**、**805 µs** 和 **5.25 ms**。
   - 在 [submissions 频道](https://discord.com/channels/1189498204333543425/1343002583001726986) 中，一个值得注意的提交达到了 **1247 µs**，而另一个则以 **289 µs** 获得了 **第 6 名**。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **更新后 Prompt 失效**：**NotebookLM** 最近的更新导致一位成员的旧 Prompt 失效，这表明需要更新 Prompt Engineering 策略。
   - 成员们可能需要重新检查并修正他们的方法。
- **Zotero 渴望与 NotebookLM 连接**：一位用户询问如何将其 **Zotero** 收藏库与 **NotebookLM** 集成，而无需手动添加每个 PDF。
   - 目前无法直接连接 **Zotero**，仍需逐个上传 PDF。
- **NotebookLM 作为专注的专家教师**：一位用户建议，主要用例是学习新知识，**NLM** 提供了为特定领域或主题创建**专注专家（Focused expert）**的选项，可以充当**老师/教练/导师**。
   - 他们补充说，*由于来源是答案背后的真相，NLM 可以消除或限制噪音*。
- **文件大小限制阻碍 US Code 分析**：一位成员在利用 **NotebookLM** 分析 **US Code**（美国法典）的冗余时遇到了 PDF 和字数限制。
   - 另一位成员建议将代码拆分为较小的文件，作为应对大小限制的权宜之计。
- **Free 与 Plus 账户引发混淆**：成员们对降级和超出来源限制背景下的 **Free** 和 **Plus 账户** 区别表示疑问。
   - **Free 账户** 限制为每个笔记本 **50 个来源**，总计 **100 个 Notebooks**；而 **Plus 账户** 每个笔记本可拥有 **300 个来源**，总计 **500 个 Notebooks**。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Claude 的图像处理困扰 MCP Tools**：一位用户询问 **Claude Desktop** 如何处理上传到 **MCP tools** 的图像，特别是 **Claude** 是检测最近的上传还是使用特定的引用方法。
   - 该用户目前的工具是为图像 URL 构建的，而非直接上传，因此正在寻求关于接受上传图像的 **MCP tools** 实现建议。
- **MCP 公开描述显示故障**：多位用户报告称，尽管在管理面板中进行了更改，但其 **MCP server** 的**公开描述（public description）**在公开页面上并未更新。
   - 该问题在 `dataryan.cph` 上尤为明显。
- **JSON 反引号错误困扰 Hooks**：一位用户报告称，**Claude** 在为 SDK 创建 **MCP hooks** 时坚持使用**反引号（backticks）**，怀疑是否存在配置错误。
   - 另一位用户建议尝试从 `dataToWrite` 中移除引号。
- **寻求支持 Streamable 的 MCP 客户端**：一位用户正在寻找支持与 **MCP server streamables** 连接的客户端，因为他们的服务器仅能在自己的客户端运行，无法与其他常用工具配合使用。
   - 有用户提到 *mcp-remote* 支持 mcphttps，并给出了 [草案版本](https://github.com/geelen/mcp-remote/pull/32) 的链接。
- **Kubernetes MCP 实现发布**：一位成员宣布他们 [基于 k8s API](https://github.com/StacklokLabs/mkp) 构建了一个新的 kubernetes MCP，使其更加通用和灵活。
   - 它的定位是作为获取 GitHub 仓库的第一个 MCP Server 的替代方案，允许 **AI** 使用仓库代码作为参考。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Agent 构建讨论正蓬勃发展**：[AnthropicAI 发布了 Building Effective Agents](https://www.anthropic.com/)，**dexhorthy** 的 [12 Factor Agents](https://www.12factor.net/) 走红，且 [OpenAI 发布了 A Practical Guide To Building Agents](https://platform.openai.com/docs/guides/function-calling)。
   - 社区正在就什么是 Agent 以及如何构建它们进行**富有成效的对话**。
- **CondoScan 全面降低公寓成本**：**CondoScan** 利用 **LlamaIndex 的 Agent 工作流**和 **LlamaParse** 精确的文档处理能力，打造了下一代公寓评估工具，详见此[案例研究](https://t.co/SzIbcKta1O)。
   - **CondoScan** *评估财务健康状况和生活方式契合度*，将文档审查时间从数周缩短至数分钟。
- **探讨 `chat_history` 与 `memory` 的区别**：如果自行管理聊天消息或使用消息列表进行初始化，请使用 `chat_history`；否则，请使用特定的 memory 模块。
   - *如果你只是自行管理聊天消息列表，或者想用某些聊天消息列表进行初始化，请使用 chat_history；如果你正在维护/使用特定的 memory 模块，请使用 memory*。
- **`AgentWorkflow` 错误已解决**：`AgentWorkflow` 中的一个间歇性错误被追踪到：*400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'Please ensure that the number of function response parts is equal to the number of function call parts of the function call turn.', 'status': 'INVALID_ARGUMENT'}}*。
   - 通过 `pip install -U llama-index-llms-google-genai` 升级 `llama-index-llms-google-genai` 解决了该错误，详情见此 [GitHub pull request](https://github.com/run-llama/llama_index/pull/18527)。
- **`FunctionAgent` 现提供超时功能**：[`FunctionCallingAgent`](https://llama_index.readthedocs.io/en/stable/understanding/agent/) 类缺乏直接的超时参数，但可以在 LLM 对象上为每个 LLM 调用设置超时。
   - 较新的 Agent 类（如 `FunctionAgent`）通过 `request_timeout` 参数直接支持超时，并提供了一个使用 `Ollama` 并将 `request_timeout` 设置为 360 秒的代码片段示例。



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 权衡多模态建模**：社区想知道 **DSPy** 是否支持**多模态模型**和**多模态推理工作流**，并指出如果 **Chain of Thoughts (CoT)** 和 **ReACT** 能结合 **CoT** 进行工具选择，功能会更强大。
   - 讨论承认了推理模型的进展以及 **DSPy** 适应多模态场景的潜力。
- **框架在快速变化面前显得力不从心**：成员们观察到 **AI 领域**的快速演进使得 **Langchain** 和 **LlamaIndex** 等框架难以保持同步。
   - 建议优先选择能简化任务的框架，并在必要时直接与 **model API** 交互。
- **Ember 作为新概念涌现**：小组讨论了诸如 [Ember](https://github.com/pyember/ember) 之类的新想法，这些想法在构建**复合 AI 系统 (compound AI system)** 时可能需要不同的策略。
   - 虽然 **DSPy** 提供**声明式语法**，但 **Ember** 提出了一种替代方案，两者各具优势。
- **文本和图表提振业务**：小组注意到，由于业务需求，许多框架专注于**文本和图表**，并强调以业务为中心的系统通常需要对文本和表格进行推理。
   - 有人提到，在预处理过程中，经常使用 **VLM (用于 OCR)** 将图像转换为**文本、JSON 结构或表格**。
- **深度代码分析需求**：一位成员寻求一种能够分析超出典型上下文限制的大型代码文件的方法，旨在提取最大程度的见解。
   - 他们阐明了对深度分析的需求，不再局限于通用要求，而是要利用文件中的所有可能信息。



---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **请求共享 GRPO 代码**：一名成员请求获取 **Torchtune 的 GRPO** 代码，推测这对 **Torchtune 的 GRPO** 的其他用户会非常有帮助。
   - 该成员还对代码改动的侵入性表示好奇，认为 **Torchtune 的 GRPO** 的其他用户也会有兴趣查看这些代码。
- **PPO Epochs 导致 KL Divergence 偏差**：一名成员质疑 **Torchtune** 中的 `ppo_epochs > 1` 是否会导致 KL Divergence 估计产生偏差，并指出了[该文件的第 85-87 行](https://github.com/joecummings/r1-zero/blob/main/torchtune/dev/grpo/loss.py#L85-L87)。
   - 他们认为，在遍历过一次 replay buffer 中的每个样本后，策略（**pi_theta, pi_ref, pi_old**）已经发生了变化，因此样本不再来自当前的策略 **pi_theta**。
- **GRPO Padding 方向受到质疑**：一名成员注意到 **GRPO 数据 padding** 方向是在右侧，并询问 decoder 模型在训练期间是否应该在左侧进行 padding。
   - 另一名成员回答说，只要[正确处理 input positions 和 masks](https://github.com/pytorch/torchtune/blob/main/recipes/dev/grpo_full_finetune_distributed.py#L750)，两侧都可以进行 padding。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **低 RAM 运行大型 LLM**：一位用户表示 **16GB** 的 RAM 足以运行 **32B** 模型，如果愿意牺牲质量和速度，甚至可以用 **8GB** 运行 **70B** 模型。
   - 该用户还指出，逐行进行 LLM 提示（prompting）的实现相对简单。
- **Shell 脚本可以替代 Rust**：一名成员提议短小的 Shell 脚本可以完成 **Rust 代码** 能做的事情。
   - 未提供更多细节。
- **Llama4 遇到 Scout 17x16**：一名成员寻求关于在 **scout 17x16** 上运行 **Llama4** 的指导，询问是否需要更新代码或 **Jinja** 配置。
   - 另一位用户回复说 *gpt4all* 已经过时，建议探索其他选项，导致原成员放弃了尝试。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **医疗初创公司寻求 Beta 版 Prompt 优化功能**：一家专注于影响力工作且发展迅速的医疗 **AI/ML 初创公司**已申请 Beta 版 **Prompt Optimization** 功能，但在 **Cohere 社区 Discord 服务器**中缺乏访问权限。
   - 该初创公司正在询问访问权限是否受**用户资历**或**计费层级**的限制。
- **服务器要求新成员进行自我介绍**：服务器欢迎新成员，并鼓励他们向社区介绍自己。
   - 系统提示新成员分享他们的**公司/行业/大学**、当前项目、偏好的技术/工具以及在社区的目标。

---

**tinygrad (George Hotz) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**Codeium (Windsurf) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道详细摘要和链接

### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1365401957899960352)** (1 条消息): 

> `iOS 语音助手, GPT 图像生成, Grok 3 Beta, OpenAI o4-mini` 

- **Perplexity 推出 iOS 语音助手**：Perplexity 本周发布了 **iOS 语音助手**。
   - 查看 [changelog](https://www.perplexity.ai/changelog/what-we-shipped-april-25th) 中的完整列表。
- **Perplexity 推出 GPT 图像生成**：Perplexity 本周发布了 **GPT 图像生成**功能。
   - 查看 [changelog](https://www.perplexity.ai/changelog/what-we-shipped-april-25th) 中的完整列表。
- **Perplexity 发布 Grok 3 Beta**：Perplexity 本周发布了 **Grok 3 Beta**。
   - 查看 [changelog](https://www.perplexity.ai/changelog/what-we-shipped-april-25th) 中的完整列表。
- **Perplexity 部署 OpenAI o4-mini**：Perplexity 本周发布了 **OpenAI o4-mini**。
   - 查看 [changelog](https://www.perplexity.ai/changelog/what-we-shipped-april-25th) 中的完整列表。

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1365175496798634046)** (1110 条消息🔥🔥🔥): 

> `Discord Roles Rework, GPTs Agent Training, OpenAI sidebars, Apple Music Discord Support, Grok 3 Mini` 


- **Discord 角色迎来刷新**：成员们报告称 **Kes** 正在重做 [Discord 角色的链接系统](https://discord.com/channels/link/to/roles)。
   - 关于该角色链接系统如何重做或如何惠及用户的进一步细节尚未讨论。
- **GPT-4o 提供优质的文本图像输出**：成员们讨论了 [GPT-4o 图像生成的语言模型](https://link.to/model) 如何提供更出色的**文本图像**，尽管其他人更喜欢 Midjourney 的整体质量和纹理。
   - 成员们表示，*MJ 特别希望用户能够通过输入像“cat”这样的单个词就能获得非常漂亮而非基础的内容——这意味着其底层架构的设计使其在大多数时候并不完全听从 Prompt，而是按自己的方式生成，因为他们只关心美感。*
- **Perplexity 团队举办“蛋”愿活动**：成员们参加了一个在各个频道收集**彩蛋**的活动，一些人表示 *GPT 图像生成* 无法生成特定的图像。
   - 他们使用了几种方法，例如*直接裁剪图像*，而其他人则抱怨看到的*普通掉落彩蛋都是一样的，我已经看了 100 多次了。*
- **DeepSearch 在 Reddit 数据上翻车**：一名成员开玩笑说，在 [一张 AI 图像](https://link.to/img) 返回了意外结果后，**Perplexity 又坏了奖**实至名归。
   - 另一位成员补充道：*我只想知道它是如何得出那个结论的。也许它是在看 20 世纪初的推荐，哈哈。*
- **Grok Mini 推理：Perplexity 的卓越秘诀**：成员们讨论了为什么 Perplexity 使用普通的 **Grok 3** 而不是推理版本。
   - 但一名团队成员表示，[发现 Grok 3 在阅读、理解和回答问题方面表现出色——优于推理版的 Grok 3 mini](https://link.to/blogpost)。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1365212835298152469)** (2 条消息): 

> `Perplexity AI Search, Spanish Inquisition` 


- **Perplexity AI 搜索链接**：一位成员发布了一个关于西班牙宗教裁判所的 [Perplexity AI 搜索](https://www.perplexity.ai/search/was-the-spanish-inquisition-co-K1dnw4YsSlKIcp7CHsFBJQ.matsku) 链接。
- **另一个 Perplexity AI 搜索链接**：一位成员发布了一个 [Perplexity AI 搜索](https://www.perplexity.ai/search/1e012c90-ba02-499f-b99b-4151742606df) 链接。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1365336367453704355)** (1 条消息): 

> `Perplexity API Introduction, Denis Yarats` 


- **Perplexity API 介绍视频预告**：一位用户表示有兴趣在 [“与联合创始人兼 CTO Denis Yarats 一起了解 Perplexity API”](https://www.youtube.com/watch?v=dQw4w9WgXcQ) 视频上线后观看。
- **用户错过了直播**：该用户提到自己在直播期间没空，表明了对 API 介绍视频回放的需求。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1365211135514509353)** (1315 条消息🔥🔥🔥): 

> `TACQ 压缩, TRL 奖励函数, Paradigm 投资 Nous Research, Psyche 分布式训练项目, Nous API 等候名单问题` 


- **Nous LLM 旨在通过 TACQ 实现 2-bit 精度**：一篇[研究论文](https://www.marktechpost.com/2025/04/22/llms-can-now-retain-high-accuracy-at-2-bit-precision-researchers-from-unc-chapel-hill-introduce-tacq-a-task-aware-quantization-approach-that-preserves-critical-weight-circuits-for-compression-withou/)介绍了一种针对 LLM 的任务感知量化（task-aware quantization）方法 **TACQ**，该方法通过使用校准数据集来决定哪些权重可以进行更大程度的压缩，从而在 **2-bit 精度**下保持高准确度。
- **寻求经过实战检验且兼容 TRL 的 GRPO 奖励函数**：一位成员询问在哪里可以找到经过实战检验且兼容 **TRL** 的 **GRPO 奖励函数**。
   - 另一位成员建议参考 Hugging Face 的 [Open-R1 rewards](https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py) 作为起点。
- **Paradigm 向 Nous Research 投资 5000 万美元**：Nous Research [宣布获得来自加密风险投资公司 Paradigm 的 **5000 万美元**投资](https://fortune.com/crypto/2025/04/25/paradigm-nous-research-crypto-ai-venture-capital-deepseek-openai-blockchain/)，用于推进其 AI 研究和开发工作。
- **Nous 将推出 Psyche，一个基于 Solana 的分布式训练项目**：Nous Research 正在开发 **Psyche**，这是一个利用 **Solana 网络**进行协作的[分布式训练项目](https://nousresearch.com/nous-psyche/)。
   - 团队澄清说，由于目前专注于开发，**Psyche** 暂时没有角色分配或社区计划，但为对加密货币领域感兴趣的人提供了 [Psyche Discord](https://discord.gg/peqZyPRd) 频道。
- **用户在 API 等候名单验证邮件方面遇到问题**：许多用户报告在注册 Nous API 等候名单后未收到验证邮件，这可能是由于**高流量**导致的。
   - 团队承认了这一问题，表示**邮件发送服务器**可能过载，并建议用户稍后再试，同时澄清该等候名单专门针对 API，而非 Psyche 项目。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1365308221014671470)** (36 条消息🔥): 

> `Mac M4 上的最大模型尺寸, LLM 与控制向量, Nous Psyche vs Petals, Hermes 3 70B 设置, 适用于 8GB VRAM 的开源代码模型` 


- **Mac M4 内存限制模型尺寸**：一位成员询问在 **Mac M4 16** 上运行 **Nous 模型**的最大尺寸，另一位成员回答说，在 **64GB+ 内存**的情况下，大多数 **70B 参数**以下的量化模型应该都可以运行。
- **LLM 能察觉到控制向量吗？**：一位成员询问，在 **zero-shot** 或 **few-shot** 场景下，**LLM** 是否能察觉到我们正在使用**控制向量（control vector）**来引导它们的思路。
   - 另一位成员建议，LLM 可能会从 few-shot 场景中的输出模式中推断出来，或者在对话中途被直接问及变化时察觉。
- **Psyche 用于训练，而 Petals 用于推理**：一位成员询问 **Nous** 与分布式 LLM 项目 **Petals** 的对比。
   - 另一位成员澄清说，Nous 的分布式项目名为 **Psyche**，与用于推理的 Petals 不同，Psyche 是为训练设计的，文档可在 [Psyche Network](https://docs.psyche.network) 查看。
- **Qwen 2.5 Coder 7B Instruct 是一个不错的选择**：一位成员询问在 **8GB VRAM** 的显卡上使用 **ollama** 运行通用代码任务的最佳开源模型。
   - 另一位成员推荐了 **Qwen 2.5 Coder 7B Instruct**，强调使用 **Q6_K 版本**，并指出它在处理基础助手任务之外的复杂编程（如编写乒乓球游戏）时存在局限性。
- **据称 Q6_K > Q4_K_M**：在关于 **Q6_K** 和 **Q4_K_M** 区别的讨论中，一位成员声称 **Q6_K 更聪明，尤其是在代码编写方面**，但会占用更多内存。
   - 该成员建议从 **Q5_K_M** 开始以平衡上下文和速度，如果有足够的剩余 VRAM（特别是达到 **10GB VRAM** 时），可以考虑升级到 **Q6_K**。


  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1365308889687527495)** (5 messages): 

> `Fine-tuning LLMs, LoRA, LLaMa Factory, CPU-only fine-tuning, GGUF` 


- **LLM LoRA 仅限 CPU 的微调终于实现**：一位用户分享了使用 **LLaMa Factory 在仅限 CPU 的环境下通过 LoRA 微调 LLM** 的[分步指南](https://medium.com/@contact_30070/step-by-step-guide-for-fine-tuning-your-llm-with-llama-factory-using-the-cpu-only-96b2fc6a80b0)。
   - 该指南涵盖了从安装 **torch** 到在 **LM Studio** 中加载自定义 **GGUF** 的所有内容，并附带了一个 [Youtube 视频](https://www.youtube.com/watch?v=1bgL4b7VT8M)。
- **LM Studio 非常出色**：许多用户喜欢其 GUI。
   - 是的。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1365176707971027024)** (317 messages🔥🔥): 

> `o3 can output large file code, AGI path for big tech companies, Yann LeCun's 'A Path Towards Machine Intelligence', Meta's Memory+, DeepMind's Titans` 


- **o3 输出大文件代码！**：用户报告称 **o3** 现在可以输出更大的代码文件，达到 **700-1000 行**，而之前的限制约为 **300 行**。
   - 这一改进同样适用于 **O4 Mini High**。
- **通往 AGI 的可能路径**：一位成员提出了一条通往 AGI 的路径，即 AI 决定利用**过去的记忆**、更新的**知识图谱**和**想象力**来创建照片、3D 模型/视频和音频以提供上下文。
   - 另一位成员建议阅读 **Yann LeCun** 的 *A Path Towards Machine Intelligence* [openreview.net](https://openreview.net/pdf?id=BZ5a1r-kVsf)，以了解不包含生成方面的类似模拟，并认为潜空间变换（latent space transformations）已具有足够的表达能力。
- **Sunstrike 模型在 Arena 中亮相**：一个名为 **Sunstrike** 的新模型已添加到 Arena 中，初步印象是 *claude-3-7-sonnet-20250219-thinking-32k > sunstrike > gemma-3-12b-it*。
   - 早期测试显示 **Sunstrike** 可能是一个 Google 模型，并且存在一个 arc-agi 问题，而 **2.5 pro** 只有大约 **25%** 的时间能解决该问题。
- **GPT-4o 是原生多模态吗？**：关于 **GPT-4o** 是否为原生多模态展开了讨论，并提出了支持和反对的论据，这与工具的使用有关。
   - 一位成员认为 **GPT-4o** 并非真正的多模态，因为它使用特定的自然语言查询进行工具调用并生成全新的图像，这与原生图像生成器不同；而另一位成员则表示 **GPT-4o** *过去调用 Dali-E，但现在是原生生成图像*。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1365195233008156704)** (293 messages🔥🔥): 

> `Manus's strategic role, Geostrategic relevance of HK, Manus better than ChatGPT?, Double credits, Manus's issues` 


- **Manus 积分双倍获取：限时奖励盛宴**：Manus 正为新老订阅者提供首月**双倍积分**：Starter 计划可获得 **+3,900 积分**，而 Pro 计划可获得 **+19,900 积分**。
   - 这是一个*限时优惠*，没有固定的截止日期，因此鼓励用户尽快使用。
- **免费用户获得首次订阅积分**：对于**免费用户**，一位成员澄清说，他们在升级到付费计划时，**首次订阅**可获得双倍积分。
   - 如果订阅 Starter 计划，将获得 3900✖️2 积分，Pro 计划则是 19900✖️2。
- **使用 Manus 创建的代码克隆项目**：一位成员创建了一个 [Discord flask 克隆](https://github.com/johnnyshelby123/discordflaskclone)，并提议由 Manus 进行改进。
   - 另一位成员提议用 **Node.js** 重写，但原作者已经更新了代码，并强调 **Manus 完成了 100% 的工作**。
- **提示工程要点：分层推理生成器**：一位成员分享了[他们的提示工程方法](https://github.com/justinlietz93/hierarchical_reasoning_generator/tree/main/hierarchical_planner/persona_builder/personas)，旨在以最少的代码生成整个项目。
   - 他们指出了分层系统化和动态网关，可以根据生成的初始计划调整 Persona 或提示，基本上可以通过一个提示和两个 Python 文件（每个文件不到 1000 行代码）生成整个项目。
- **固定费用幻想：用户思考定价**：一些用户对 Manus 的**积分制**感到沮丧，表示他们更倾向于支付固定的月费以获得无限次使用。
   - 一位用户解释说，由于 Manus 犯的错误，他们损失了已支付的积分，并被迫切换回 **ChatGPT 4.0**。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1365189118967021628)** (80 条消息🔥🔥): 

> `Sora 的限制与定价，RAG + LLM + Word2Vec 讨论，微调数学模型，Gemini Advanced 视频生成限制，Sora 的写实图像生成替代方案` 


- ****Sora 视频定价揭晓！****：一位用户询问了 **Sora 的图像创建限制**，并发现了以更高成本购买视频生成和并发图像创建的新选项。
   - 该用户最初面临一次只能生成一张图像的限制，并收到提示要求其*升级*到他们已经拥有的计划。
- ****RAG、LLM 与 Word2Vec 组合备受关注****：一名成员发起了关于结合 **RAG、LLM 和 Word2Vec** 的讨论，并指出许多中国大陆公司正在开发具有*本地知识库（Local Knowledge Base）*功能的软件。
   - 讨论中提出了对 **LLM 掌握上下文长度（contextual length）**能力的担忧，这可能需要对原始文档进行进一步的段落级处理，从而引发了对该技术未来的疑问。
- ****数学模型热潮：专精化以提升速度并节省成本！****：一位用户建议**针对大学水平的数学问题微调较小的 AI 模型**，而不是依赖于在所有领域训练的大型模型。
   - 该想法是创建一个*规模刚好足以理解语言并理解问题*的模型，并使用 Python 集成内置计算器，从而优化速度和成本效益。
- ****Gemini Advanced 限制曝光：视频生成大比拼****：一位用户指出了 **Gemini Advanced 中不一致的视频生成限制**，指出它每天仅提供 **10 个视频**，而 AI Studio 中的免费额度为每天 **4 个**。
   - 尽管有此限制，他们认为 **Gemini 的 Veo 2** 在通用内容上的输出质量优于 **Sora**，但限制过多且存在延迟拒绝的问题。
- ****超越 Sora：寻求无审查的写实图像替代方案****：一位使用 **Sora** 的专业摄影师正在寻找内容限制较少、且能达到专业电影级水平的写实图像生成替代方案。
   - 建议的替代方案包括 **Framepack、HiDream 和 Flux**，这些工具是本地化的且无审查，但它们在匹配 **Sora 的电影级光影、景深和锐利的写实感**方面的能力仍不确定。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1365185736688795721)** (16 条消息🔥): 

> `OpenAI Pro 与 Plus 的 Deep Research 限制，Deep Research 等级列表，GPT 在预期生成代码时生成了图像` 


- **Pro 用户获取 Deep Research 限制详情**：根据[这条推文](https://x.com/OpenAI/status/1915505961500070245)，OpenAI Pro 用户可获得 **120 次原始 Deep Research** 和 **120 次额外的轻量级 Deep Research**。
- **Deep Research 等级列表已创建**：用户们看到了 **Deep Research full**、**Deep Research lite**、**Deep Research mini**（即普通的 o3）以及 **Deep Research nano**（即普通的 o4-mini）。
   - 另一位用户提议将 *Deepest Research* 和 *Deep and Dark Research* 作为可能的名称。
- **GPT 在预期生成代码时生成了图像**：一位用户分享了一张[图片](https://cdn.discordapp.com/attachments/973938214257713183/1365421332338053232/image.png?ex=680d3f35&is=680bedb5&hm=c0a3d473ee1c6db11d216f00d66b15a961805de359dd8bf9c0e746e5a2c28b31&hello)，显示 GPT 在执行桌面端任务时，开始写道它正在生成图像，而不是根据文档为桌面端生成代码。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1365315965226717238)** (48 条消息🔥): 

> `使用 ChatGPT 学习 Python，初学者选 Python 还是 Java，Web 开发语言 (JavaScript)，The Odin Project` 


- **ChatGPT：你的 Python 教授？**：成员们讨论了 **ChatGPT** 是否能有效地教授 **Python**，一致认为它可以涵盖核心基础知识，但在处理最新的第三方包时可能会遇到困难。
   - 一位成员强调，*难点不在于知识本身，而在于在没有课程结构的情况下进行自学的自律性*。
- **Web 开发选 Python 还是 Java？**：一位成员询问关于使用 **Python** 进行 Web 开发的问题，但另一位成员澄清说 **JavaScript**（特别是 **Three.js**）通常是首选。
   - 有人指出 **Java** 与 **JavaScript** 是不同的，且各自适用于不同的角色。
- **Odin Project：开启 Web 开发之旅**：一位成员推荐了 [Odin Project](https://www.theodinproject.com/)，这是一个广受好评的免费 Web 开发初学者项目。
   - 建议将 AI 作为指南，用来澄清教材中令人困惑的概念。
- **2025 年：Python 仍是顶级语言吗？**：在关于 2025 年最适合初学者的语言讨论中，有人认为 **Python** 可能不再是理所当然的首选。
   - 还提到 **Python** 在 AI 项目中仍占据主导地位，而 **JavaScript** 适用于 Web 和 UI 开发，**C++** 则用于系统级工具。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1365315965226717238)** (48 条消息🔥): 

> `使用 ChatGPT 学习 Python，初学者选 Python 还是 Java，Web 开发中的 Python，The Odin Project，Python 在 AI 中的角色` 


- **ChatGPT：Python 教授**：成员们讨论了初学者如何利用 **ChatGPT** 作为教学工具来学习 **Python**，一位成员表示 *它是一个能帮你解释任何问题的老师*。
   - 有人指出，虽然 **ChatGPT** 对核心语言基础很有帮助，但它对第三方包的知识可能已经过时，因为 *大多数现代包在 LLM 知识截止日期更新时都会进行重大重构*。
- **Java vs Python：初学者的抉择**：对话探讨了是否有必要同时学习 **Python** 和 **Java**，一位成员表示，*如果你想用 Python 和 Java 编程，那就有必要*。
   - 然而，另一位成员也指出，*即使你想让 AI 帮你写代码*，掌握这两者也是必要的，以便排除任何故障。
- **Python 不太适合 Web 开发，试试 JS**：成员们表示 **Python** 通常不用于 Web 开发，建议学习 **JavaScript**，特别是 **Three.js**。
   - 对话澄清了 **JavaScript** 与 **Java** 是不同的。
- **Odin Project：学习绿洲**：一位成员建议初学者关注 **Odin Project**，这是一个广受好评的免费 Web 开发项目。
   - 他们建议用户可以 *在学习这些课程的同时将 AI 作为指南，来澄清教材中令人困惑的内容*。
- **2025 年 AI 领域中 Python 的地位**：一位成员提到，在 **2025** 年，**Python** 对于 AI 项目仍然相关，但它的地位 *可能不再是一个绝佳的核心首选*。
   - 讨论随后转向了 **Python** 在 **AI** 中的盛行，对比 **JavaScript** 在 Web 开发中的实用性，以及 **C++** 在系统级工具中的应用。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1365179338194157651)** (96 messages🔥🔥): 

> `Unsloth Dynamic v2.0 GGUFs release, GLM 4 models, Tesslate Rust 7B model, Dynamic Quantization Workflow, QwQ 32b training` 


- **Unsloth 发布 Dynamic v2.0 GGUFs**: 如果你错过了 (ICYMI)，Unsloth 已经发布了新的 **Dynamic v2.0 GGUFs**，可以在[这里](https://x.com/UnslothAI/status/1915476692786962441)获取。
   - 一位成员询问了这与即将推出的多 GPU 支持有何区别，另一位成员澄清说，*官方版本需要大量测试来确认其稳定性*。
- **GLM-4 热潮席卷 Unsloth 的 HF**: 成员们正在讨论新的 **GLM 4 模型**，一位成员已将其上传至 HF，链接在[这里](https://huggingface.co/unsloth/GLM-4-32B-0414-GGUF)。
   - 一位成员强调了 [THUDM/GLM-Z1-32B-0414](https://huggingface.co/THUDM/GLM-Z1-32B-0414)，并提到它*即使面对 DeepSeek R1，在半数基准测试中也名列前茅*。
- **Rust 7B 量化请求引发 Rust fim**: 一位成员请求对 **Tesslate** 的新 **Rust 7B 模型**进行量化，这促使另一位成员为其 codefim [数据集](https://huggingface.co/datasets/Etherll/CodeFIM-Data)创建了一个纯 Rust 版本。
   - 该成员随后询问 *是否有人已经将 phi-4 编译为 gguf？* 因为在操作时出现了错误。
- **Dynamic Quantization 工作流保持神秘**: 一位成员询问最近宣布的新 **dynamic quantization 工作流**是否开源。
   - 另一位成员确认 *该工作流尚未开源*，因为 *它针对每个模型都略有不同*。
- **QwQ 32b ablation 需要强大的算力支持**: 一位成员询问是否有理由不能微调 abliterated 版本的 **QwQ**，并链接到了 [HF 上的模型](https://huggingface.co/huihui-ai/QwQ-32B-abliterated)。
   - 另一位成员回应称，**QwQ 32b** 可能需要大约 **30+GB** 的 VRAM 来训练，因此他们需要配置量化或 FSDP 来训练一个通常无法放入单张 GPU 的模型。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1365209935213887500)** (62 messages🔥🔥): 

> `torch.compile issues, GGUF working, chat template, llama.cpp error` 


- **Torch Compile 遇到麻烦？**: 一位用户报告了一个可能与 `torch.compile` 相关的 `Tensor.requires_grad` 问题，并指出 **LoRA** 和 **全量微调 (full fine-tuning)** 在 `FacebookAI/roberta-base`（设置 `attn_implementation = "eager"`）和 **modernBERT** 上可以正常工作。
   - 他们询问这将在哪行代码中得到修复。
- **发现 GGUF 的奇特行为**: 一位用户询问为什么一个 **GGUF** 文件在 **fp16** 下可以工作而另一个不行，具体比较了 `snowflake-arctic-embed-m-v1.5` 和 `snowflake-arctic-embed-l-v2.0-f16` 的 **GGUF** 模型。
   - 该模型用于 embedding。
- **澄清 Chat Template 的困惑**: 一位用户质疑为什么 `llama3_1_alpaca` 笔记本默认不应用 chat template，得到的解释是 `apply_chat_template` 默认使用模型 `tokenizer_config.json` 或 `chat_template.json` 中的模板。
   - 使用不同的模板（如 **alpaca**）需要显式指定，且模板的选择应与微调和推理所需的交互格式保持一致。
- **Llama.cpp 在运行 DeepSeek V3 时遇到困难**: 一位用户在 **llama.cpp** 上运行 **UD Q4** 版本的 **DeepSeek V3** 时遇到错误，尽管之前的 **Unsloth GGUF Q4** 版本运行正常，错误提示 `'blk.0.attn_q_b.weight'` 张量存在形状不匹配 (shape mismatch) 错误。
   - 用户使用最新版本重新编译了 **llama.cpp**，但错误仍然存在。
- **Gemma-3-4b-it 微调困扰**: 一位用户在 Google Colab 上通过 Unsloth 笔记本训练 `gemma-3-4b-it` 模型时遇到了 `RuntimeError`，出现了 *float 与 c10::Half 之间的数据类型不匹配*。
   - 即使在还原所有更改后，错误依然存在，表明可能与环境不兼容。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1365366933138047037)** (2 messages): 

> `Kimi Audio, MoonshotAI` 


- **MoonshotAI 发布 Kimi Audio**: MoonshotAI 发布了 [Kimi Audio](https://github.com/MoonshotAI/Kimi-Audio)，其论文已在 [arXiv](https://arxiv.org/abs/2504.09858) 上发表。
- **Kimi Audio 登录 arXiv**: Kimi Audio 的论文可在 [arXiv](https://arxiv.org/abs/2504.09858) 上查阅。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1365184879440171178)** (86 条消息🔥🔥): 

> `Llama-3.2 checkpoints, Windows 上的 Nvidia 驱动 TCC 模式, 电脑配置成本与数字鸿沟, 用于数据分析的 Colaboratory, Flash Linear Attention 模型` 


- **Llama-3.2 Checkpoint 导出至 ONNX 依然困难**：一位成员正在寻找将 **Llama-3.2 checkpoints** 导出为支持 **KV cache** 和 **TensorRT** 的 **ONNX** 脚本，并表示特定架构的初始化并未带来明显收益。
   - 他们注意到在尝试默认 PyTorch 初始化与特定架构初始化时几乎没有收益，并询问在 Windows 或 WSL 上是否可能将 Nvidia 驱动设置为 TCC 模式。
- **利用 Colaboratory 解决数字鸿沟**：一位成员对日益增长的电脑配置成本及其可能扩大的数字鸿沟表示担忧。
   - 另一位成员建议使用 [Google Colaboratory](https://colab.google) 及其 *闲置的 720s* 进行轻量级数据分析，但指出其局限在 **1B 参数**左右的模型，如 **GPT-2**。
- **改装版 48GB RTX 4090 走红**：一位成员提到改装版 **48GB RTX 4090** 的存在，这在中国特别受欢迎，并链接了来自 Tom's Hardware 的一篇[拆解文章](https://www.tomshardware.com/pc-components/gpus/blower-style-rtx-4090-48gb-teardown-reveals-dual-sided-memory-configuration-pcb-design-echoes-the-rtx-3090)。
   - 这些显卡的售价通常在 *3,500 美元* 左右，不过由于供应紧张，关于 **5090** 的信息尚未确认。
- **小数据集采用 Linear Attention**：一位成员寻求在小数据集上训练的最简单的 **Linear Transformer** 模型或等效模型，排除 **RWKV**。
   - 另一位成员建议使用 **Flash Linear Attention** 仓库中的模型，例如 **Gated Linear Attention** 或 **Gated DeltaNet**，并使用其 **Flame** 训练器进行训练。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1365176777831616594)** (68 条消息🔥🔥): 

> `ReaSCAN 结果, 用户编辑, 使用学习到的 LERP 替换 Key, Memory mosaics 论文, tokenshift` 


- **研究论文**：希望能看到 **ReaSCAN** 关系从句测试的结果，以及关于**失调容忍度（misalignment tolerance）**、**系统调制（system modulation）**、**逆向建模（inverse modeling）**以及**非线性引力发散与记忆（non-linear gravitational divergence and memory）**等主题的研究。
   - 推荐了 **Xia & Sigmund**、**Fatemi Booshehri**、**Zhang, Sharif & Siddiqa** 以及其他著名研究者的论文。
- **Key 被替换了！**：有人提问是否有人用 **W_kx_t** 和过去 Key 的学习 **LERP** 替换了 Key，公式为 **k_t = lambda W_kx_t + (1-lambda)k_{t-1}**。
   - 该成员从 [Memory Mosaics 论文](https://arxiv.org/abs/2405.06394)中得出的结论是，这是引入的主要“增量”，而常规的 Multi-head Attention 中并不存在。
- **RWKV tokenshift**：关于先前的序列索引，有人指出这其实就是 **RWKV tokenshift**。
   - 另一位成员分享说，这在之前的常规 Transformer 中就已经存在了，因为这已经在他们的 **GPTAlpha Transformer 配方**中存在多年——该架构的描述长期存在于他们的 **GPTCore** 仓库 README 和代码中，但直到 [GoldFinch](https://arxiv.org/abs/2407.12077) 论文中才正式发表。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1365280267920740404)** (6 条消息): 

> `TinyStories 替代数据集, Leo 关于计算稀疏性的演讲, 基于归因的参数分解 (APD), 局部损失景观分解 (L3D)` 


- **SimpleStories 数据集套件替换 TinyStories**：发布了一个名为 [SimpleStories](https://huggingface.co/datasets/lennart-finke/SimpleStories) 的新替代数据集、分词器和[模型套件](https://huggingface.co/SimpleStories)。
   - 提供了一份[社区文档](https://docs.google.com/document/d/1XRb-EDDw-h6c-L6pKltI9A8zJmufE71I2_4wx8d5rMg/edit?usp=sharing)以帮助研究人员入门。
- **稀疏性演讲引发猜测**：成员们讨论了 Leo 关于计算稀疏性的演讲，以及它是否类似于**基于归因的参数分解 (APD)**（[论文](https://arxiv.org/abs/2501.14926)）或**局部损失景观分解 (L3D)**（[论文](https://arxiv.org/abs/2504.00194v1)）。
   - 据说该演讲是关于 Sparse Autoencoder 的内容，新工作似乎与之前的那些论文并不相似。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1365176535727738880)** (95 messages🔥🔥): 

> `Aider 语言排序, Grok-3-Mini 定价, Gemini+Claude 组合, Aider Architect/Editor 工作流, Gemini 冗长注释` 


- **Aider 获得语言升级**：Aider 现在支持按编程语言进行排序，增强了代码的组织性，方便开发者访问，正如[这张 UI 截图](https://cdn.discordapp.com/attachments/1131200896827654149/1365176535484596327/image.png?ex=680d03f9&is=680bb279&hm=accbfb4dd2576f4bc500e5e11cdf24b69189951769521c7cbb2980fee2cd03e0&)所示。
- **Grok-3-Mini 在价格上极具竞争力**：通过 OpenRouter 使用 **Grok 3 mini** 的价格几乎只有 **Deepseek V3** 的一半，且推理性能几乎相当，如[此图](https://cdn.discordapp.com/attachments/1131200896827654149/1365185829596696636/image.png?ex=680d0ca1&is=680bbb21&hm=f5228d04256f815b37940355080c80e2cdaa6d86a0f6b37759d6d53b59c8be26&)所示。
- **Gemini Pro 2.5 与 Claude 3.7 联手**：一位成员正在尝试在 Aider 中将 **Gemini Pro 2.5** 作为 Architect 模型，将 **Claude 3.7** 作为 Editor 模型，灵感来自展示 **Claude Code** 和通过 MCP 接口使用 **Gemini** 的视频。
   - 另一位成员报告称，虽然 **Gemini** *非常聪明*，但它倾向于修改无关代码，导致代码审查困难，因此更倾向于使用 DeepSeek V3 作为 Editor 模型。
- **Aider 的 Architect-Editor 工作流揭秘**：成员们讨论了 Aider 的 Architect/Editor 设置中的分工，即一个 AI 负责推理，另一个专注于代码编辑。
   - 一位用户提到了 `ask mode` 的价值，可以使用 `/ask Generate a detailed plan for <YOUR REQUEST HERE>` 为需求生成详细计划，然后按照计划执行；此外还可以使用 `/model` 实时切换模型以测试不同的输出。
- **Gemini 的注释过载**：用户正在寻找减少 **Gemini** 冗长注释的方法，有人建议在将 **Gemini** 作为 Architect 模式并配合 **GPT-4.1** 作为 Editor 使用时，通过约定（conventions）来删除不必要的注释。
   - 一位用户观察到 *Gemini 在注释中喋喋不休的情况比我见过的任何时候都严重*，并正在寻找更好的解决方案。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1365249954020462633)** (18 messages🔥): 

> `Gemini 2.5 Pro 设置, Aider 中的只读文件, 工作追踪 Markdown 日志与 Aider, Aider CLI vs IDE, Openrouter Fallbacks` 


- **Gemini 2.5 设置简化**：要在 Aider 中复现 **Gemini 2.5 Pro Preview 03-25** 模型，只需按照 [Leaderboard 指南](https://aider.chat/docs/benchmarks.html) 使用命令 `aider --model gemini/gemini-2.5-pro-preview-03-25`。
- **只读文件**：顺便提一下，这里的解决方案是将文件添加为 **read only**。
- **工作追踪 Markdown 日志不适合 Aider**：一位成员无法追踪 Markdown 日志，并希望 Aider 能更新它们。
- **Aider 主要在 CLI 中使用**：成员们 90% 的时间在 **terminal** (Warp.dev) 中使用 Aider，其余时间在 **VS Code 侧边栏**中使用，不使用 IDE 插件。
- **Openrouter Fallbacks 需要改进**：成员报告 **Openrouter fallback** 系统虽然*存在但似乎从未真正起作用*，即使添加了多个 API Key 也是如此。
   - 它在遇到 429 错误时未触发，或者实现方式不正确。


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1365223897922015323)** (3 messages): 

> `Aider 命令, 代码生成, 对话总结` 


- **解码用于代码生成的 Aider 命令**：一位用户询问为什么使用 `/architect proceed` 来生成代码，而不是直接用 `proceed` 或 `do it`。
   - 另一位成员澄清说，`proceed` 或 `do it` 使用的是 "code" 模式，会将整个对话发送给 Editor 模型；而 `/architect` 则要求主模型总结所需的更改，仅将该总结发送给代码模型。
- **理解 Aider 中的代码生成模式**：讨论强调了在 Aider 中触发代码生成的两种方法：使用 `proceed`（或类似命令）和使用 `/architect proceed`。
   - 关键区别在于发送给代码模型的上下文量：前者发送整个对话，而后者发送所需更改的总结版本。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1365234208871546941)** (53 条消息🔥): 

> `Framepack 使用体验、多模态 LLMs、Hugging Face Inference API 问题、SD3.5 API 更新、用于 Python 编程的 LLM` 


- **Framepack 广受好评，获得 Unsloth 认可**：成员们注意到 **Framepack** 非常酷，它让微调变得触手可及，可能非常契合 **Unsloth** 的路线并得到了他们的认可。
   - 围绕每 **8 秒**进行一次频繁权重更新的压力测试感到兴奋，并建议与专家测试人员合作。
- **探索用于视频和音频的多模态 LLMs**：讨论了使用 [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)、[Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) 和 [SmolVLM2](https://huggingface.co/blog/smolvlm2) 等多模态 LLMs 进行**视频片段解读**和摘要。
   - 成员们分享了前一天发布的 [Nvidia's DAM-3B-Video](https://huggingface.co/nvidia/DAM-3B-Video) 链接，其他人分享了用于[转录 YouTube 音频](https://huggingface.co/spaces?sort=trending&search=youtube)的链接。
- **Hugging Face Inference API 出现令人不适的问题**：一位成员报告在使用 Hugging Face Inference API 开发产品时遇到了令人不适的图像，需要联系 **HF 工作人员**。
   - 另一位成员澄清说，**Stable Diffusion** 团队隶属于 **Stability AI**，与 Hugging Face 工作人员是分开的。
- **SD3.5 API 更新触发库更新**：关于 **SD3.5**，API 的更新将触发 [TGI](https://github.com/huggingface/text-generation-inference/issues)、[Diffusers](https://discord.com/channels/879548962464493619/1014556809928904794)、[huggingface_hub](https://github.com/huggingface/huggingface_hub/issues) 和 [huggingface.js](https://github.com/huggingface/huggingface.js/issues) 等库的更新。
   - 该问题源于 **500 内部错误**，目前正在努力尽快修复，详见 [API 更新](https://discuss.huggingface.co/t/500-internal-error-were-working-hard-to-fix-this-as-soon-as-possible/150333/32) 讨论。
- **关于从零构建 Python LLM 的建议请求**：一位成员询问如何创建一个专门用于输出 **Python 代码**的 LLM，且仅使用 **PyTorch**。
   - 另一位成员建议先在 PyTorch 中从零构建架构，或许可以探索 **DeepCoder 模型**。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1365292361902723094)** (1 条消息): 

> `Deeplearning.ai 课程、Smolagents 与 Gemma3、Code Agents 对比 LLMs` 


- **Deeplearning.ai 采用 Tavily 和 OpenAI**：一位成员注意到 [Deeplearning.ai](https://www.deeplearning.ai/) 课程使用了 **Tavily** 和 **OpenAI**。
   - 他们正在探索可能允许更多本地部署的替代方案。
- **Smolagents 探索与 Gemma3 的本地 LLM 能力**：一位成员询问 **smolagents** 是否可以与 **Gemma3** 或其他本地 LLMs 一起使用。
   - 这可能实现在不依赖外部 API 的情况下运行 Agent。
- **Code Agents 增强了超越 LLMs 的推理能力**：一位成员询问 **code agents** 是否真的比直接调用 LLMs 表现更好，即使是在推理任务中。
   - 他们还询问 code agents 是否需要直接的 API，或者是否有其他选择。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1365323529163374655)** (3 条消息): 

> `GAN、Excel、GAN 模型` 


- **Excel 版 GAN 模型？**：一位成员分享了一个关于使用 Excel 手动开发 **GAN 模型**的 [SharePoint 文档](https://o365coloradoedu-my.sharepoint.com/:x:/g/personal/peye9704_colorado_edu/EaxYYtAUapZBkbQ3oiTDfAUBPbIlKqWJKQSNvqk3I1L6qQ?rtime=hWWmvf-D3Ug)链接。
- **用 Excel 构建你自己的 GAN！**：使用你已经熟悉的工具 Microsoft Excel，通过[本教程](https://o365coloradoedu-my.sharepoint.com/:x:/g/personal/peye9704_colorado_edu/EaxYYtAUapZBkbQ3oiTDfAUBPbIlKqWJKQSNvqk3I1L6qQ?rtime=hWWmvf-D3Ug)深入了解 **Generative Adversarial Networks** 的世界。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1365230181731536959)** (4 messages): 

> `Small Models Math Dataset, ingest-anything, Malware Analysis Agent` 


- **帮助小模型学习数学的新数据集**：一名成员分享了一个 [数据集](https://huggingface.co/datasets/ProCreations/SimpleMath)，旨在帮助小模型更好、更准确地学习简单数学，而不是大多数时候在复杂数学上出错。
   - 它还能通过先引入简单数学，然后再过渡到大规模复杂数学数据集的方式，帮助大模型缓慢地适应复杂数学的学习。
- **将文件导入向量数据库的 ingest-anything 项目**：一名成员介绍了 [ingest-anything](https://github.com/AstraBert/ingest-anything)，这是一个新的开源项目，可将非 PDF 文件转换为 PDF，提取其文本，进行分块（chunking）、嵌入（embedding），并将其加载到 Qdrant 向量数据库中。
   - 该工具使用 **PdfItDown** 进行 PDF 转换，使用 **LlamaIndex** 读取器进行文本提取，使用 **Chonkie** 进行分块，并使用 **Sentence Transformers** 进行嵌入。
- **用于恶意软件分析的推理 Agent 诞生**：一名成员使用 **Agno、Claude 和 GPT4.1 Nano** 创建了一个恶意软件分析推理 Agent 和报告生成框架。
   - 查看 [此处的帖子](https://www.linkedin.com/posts/subham-kundu-2746b515b_ai-agent-llm-activity-7321534694242549760-pe_Q?utm_source=share&utm_medium=member_android&rcm=ACoAACZeVjgB0HEDqU1BExX1Ypnp-q8LcgDAunk) 了解更多信息！


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/)** (1 messages): 

ilham_xx: 非常感谢，<:agree:1098629085955113011> ⭐
  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1365235849368961055)** (3 messages): 

> `Agent Template Error, Local LLMs with SmolAgents, Code Agents vs LLMs for Reasoning` 


- **Agent 模板因 402 错误失败**：一位新用户在使用第一个 Agent 模板配合 **Qwen/Qwen2.5-Coder-32B-Instruct 模型**询问阿布扎比当前时间时，遇到了 **402 Payment Required 错误**。
- **SmolAgents：它能驱动本地的 Gemma 吗？**：一位用户询问关于将 **smolagents 与 Gemma3** 或其他本地 LLM 结合使用的问题，并指出 Deeplearning.ai 的课程使用的是 **Tavily** 和 **OpenAI**。
   - 这个问题触及了本地模型与 Agent 框架的集成效果。
- **Code Agent 是否在推理能力上超越了 LLM？**：同一位用户询问 **Code Agent 的表现是否优于直接调用 LLM**，特别是在推理任务中，以及它们是否需要直接的 API。
   - 这个问题暗示了推理能力与对直接 API 访问需求之间的权衡。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1365217759008522311)** (50 messages🔥): 

> `Course Deadlines, Certificate verification, DuckDuckGoSearch timeout, Pokemon Showdown LLM Cheating, Unit 4 Assignment` 


- **课程截止日期说明**：7 月的课程截止日期与获取最终证书有关，但 Hugging Face 的课程主要是其网站上编写良好的教程 Notebook，因此该期限是针对证书的，[正如一位用户所说](https://huggingface.co/spaces/agents-course/README/discussions/25)。
   - 在获得并下载第一个证书后，它会附带一个 URL 以便将其添加到 LinkedIn。
- **证书验证难题**：一位用户报告了验证最终证书有效性的问题，因为下载链接指向的是临时文件夹文件而非 CDN URL。
   - 然而，[解决方案似乎是](https://huggingface.co/datasets/agents-course/course-certificates-of-excellence) 使用 HF 用户名作为凭证 ID，并将数据集链接作为 URL，从而提供一种更永久的验证方法。
- **DuckDuckGoSearch 超时故障**：多位用户遇到了 DuckDuckGoSearch 的超时错误，具体为 `DuckDuckGoSearchException: https://lite.duckduckgo.com/lite/ RuntimeError: error sending request for url (https://lite.duckduckgo.com/lite/): operation timed out`。
   - 用户确认他们没有配置任何 VPN 或防火墙。
- **Pokemon Showdown LLM 作弊末日**：一位用户开玩笑地将表现相当不错的 Pokemon Showdown 机器人的出现比作 *“Pokémon Showdown 的 9/11 时刻”*，预感到一大波机器人带来的困扰。
   - LLM 已经被用于横扫 Pokemon Showdown，并且已经出现了跨代作弊丑闻。
- **Unit 4 作业说明**：Unit 4 作业的目标是解决 [GAIA 基准测试](https://huggingface.co/datasets/agents-course/course-certificates-of-excellence) 中 20 个 1 级问题的子集，这要求 Agent 至少使用一个工具来解决（例如：网页搜索、文件读取）。
   - 用户应该运行一次模板，看看能得到什么样的题目，这样会更清晰。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1365193089773342841)** (109 messages🔥🔥): 

> `Reward Shaping Best Practices, Symbolic World Models, Oscillatory State Space Models, LLM World Models, Generating New Agents/Models/Worlds` 


- **AI Agent 学习 Reward Shaping 的最佳实践**：一位成员请求关于 Reward Shaping 最佳实践和经验法则的文献，并链接了一篇关于从当前世界模型进行[模拟 rollouts 离线训练](https://arxiv.org/abs/2502.05244)的论文，以减少与环境的交互。
   - 讨论中包含了一个关于 Reward Shaping 的幽默建议：*向上竖大拇指表示正向奖励，向下竖大拇指表示负向奖励。*
- **符号世界模型 (Symbolic World Models) 受到关注**：一位成员分享了一个 [arXiv 链接](https://arxiv.org/abs/2503.20124)，并强调了使用程序作为世界模型的优势，强调了它们的结构化和组合特性，与神经表示相比，这使得修改更加容易。
   - 与神经表示相比，这种方法更容易根据新的观察结果修改模型的特定部分。
- **LLM 隐式地对世界建模**：一位成员认为 LLM 的世界模型是*融合在它们的权重中*的，并将其与离散或结构化模型进行了对比。
   - 另一位成员引入了一个公式 `(A, W) = G(z)` 来表示按需生成新的 Agent/模型/世界，其中 A 是模型的架构，W 是权重/参数，G 是生成器，z 是语义种子 (semantic seed)。
- **生成式 AI 的公式**：一位成员为 AI 模型生成引入了一个通用公式 `(A, W) = G(z)`，其中 **A** 是模型的架构，**W** 是权重/参数，**G** 是生成器，**z** 是语义种子。
   - 该成员建议这个公式涵盖了各种范式，包括 Hypernetworks，并可能导致 AI 模型实现显著的压缩和生成能力，将其比作*存储 DNA 而不是整个人类*。
- **AGI vs ASI - 门萨 (Mensa) 的终结？**：一位成员区分了 AGI (Artificial General Intelligence) 和 ASI (Artificial Super Intelligence)，将 AGI 视为动态且具有适应性的系统/Agent，而将 ASI 视为构造性和程序化的系统/Agent。
   - 讨论涉及了程序化系统如何通过*不直接模拟*、*不需要逻辑完整性*以及*接受近似值*来避开符号陷阱。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1365201994913943593)** (4 messages): 

> `Music AI, Perplexity Browser` 


- **Google DeepMind 发布具有新功能的 Music AI Sandbox**：Google DeepMind 发布了 [Music AI Sandbox](https://deepmind.google/discover/blog/music-ai-sandbox-now-with-new-features-and-broader-access/)，带来了新功能和更广泛的访问权限。
   - 博客文章重点介绍了这个音乐 AI 沙盒的功能和更新。
- **Perplexity CEO 宣布浏览器将追踪一切**：Perplexity CEO 宣布他们的浏览器将追踪*用户在线进行的所有操作，以销售超个性化广告*，详见[这篇 TechCrunch 文章](https://techcrunch.com/2025/04/24/perplexity-ceo-says-its-browser-will-track-everything-users-do-onl)。
   - 一位成员回应说，*那是劝阻用户使用的最好营销手段。*


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1365428376575410269)** (20 messages🔥): 

> `Gemini 2.5 Pro Experimental Free, Rate Limits, Error Messages` 


- **Gemini 2.5 Pro Experimental 免费层级受限**：由于需求量大，[Gemini 2.5 Pro Experimental](https://openrouter.ai/google/gemini-2.5-pro-exp-03-25) 已从免费模型层级中移除，现在有更严格的使用限制：**每分钟 1 次请求**，且**每天总计 1000 次请求**（包括错误请求）。
   - 免费层级仍然可用，但为了获得更高的可靠性，建议使用 [Gemini 2.5 Pro 的付费变体](http://google/gemini-2.5-pro-preview-03-25)；模型访问权限仅对历史上购买过至少 **10 个积分**的用户开放。
- **免费 Gemini 模型标识符已修复**：`:free` 模型别名将指向标准变体，因此使用该模型 ID 的代码将继续工作。
   - 一位用户报告说 `aider --model openrouter/google/gemini-2.5-pro-exp-03-25:free` 今天早上无法工作，但修复程序已合并并很快上线。
- **自定义速率限制 (Rate Limit) 错误消息即将推出**：用户在达到速率限制时会直接收到来自 Gemini API 的错误消息，这令人困惑，因为它没有解释限制是全局的还是针对特定用户的。
   - 团队讨论了可能添加自定义消息，以向用户澄清速率限制错误的来源。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1365182006366965801)** (92 条消息🔥🔥): 

> `OpenRouter 上的 Baidu 模型，OpenRouter 对 OpenAI o3 的支持，Gemini 2.5 Pro 速率限制，Nvidia Nemotron 设置，OpenRouter 积分被利用` 


- ****Baidu 模型在招手，OpenRouter 会采纳吗？****：一位成员询问了在 OpenRouter 中添加 [Baidu 模型](https://x.com/Baidu_Inc/status/1915603080336597310) 的可能性，并指出其具有有趣的潜力，而另一位成员则指出 **DeepSeek** 目前已经可用。
   - 随后讨论了由其他提供商托管的 **GLM-4-0414**。
- ****o3 奥德赛：OpenRouter 的验证尝试！****：一位成员询问 OpenRouter 未来是否会在无需验证的情况下支持 **OpenAI 的 o3 和其他高级模型**。
   - 另一位成员表示，**OpenAI** 已暗示未来可能会取消该要求。
- ****Gemini 的慷慨：免费层级引发限流烦恼！****：成员们讨论了 **Gemini 2.5 Pro** 免费层级的 **rate limits**（速率限制），这导致了用户的困惑，并询问限制是否源于 OpenRouter 侧。
   - 一位成员指出 OpenRouter 正在发布相关公告，而另一位成员则表示需求已超过供应。
- ****Nemotron 的细微差别：Nvidia 关于灵活导航的笔记！****：一位用户寻求关于 **Nvidia Llama-3.1 Nemotron Ultra 253B v1** 模型最佳设置的澄清，促使另一位用户分享了 [开发者的建议](https://openrouter.ai/nvidia/llama-3.1-nemotron-ultra-253b-v1:free)。
   - 建议包括：在 Reasoning ON 模式下将 **temperature** 设置为 **0.6**，**Top P** 设置为 **0.95**；在 Reasoning OFF 模式下使用 **greedy decoding**（**temperature** 为 **0**）。
- ****积分危机：OpenRouter 账户被清空！****：一位成员报告了一起因涉及无限 URL 生成的 exploit 导致其 **OpenRouter 积分被耗尽** 的事件。
   - 该恶意活动追溯到一个名为 "Thread-to-Chart Correlation System" 的拟议解决方案架构，该架构在被停止前螺旋式生成了一个约 **3000 字符** 的 URL。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1365180788659982497)** (105 条消息🔥🔥): 

> `ESP32 的 RTSP 固件包，Gemini 2.5 和 Sonnet 3.7 性能，Cursor 上的账单设置，Cursor 的模型评估` 


- **ESP32 上的 RTSP 固件难题**：一位成员正尝试为 **ESP32** 上的出站流媒体创建一个可用的 **RTSP 固件包**，并征求建议。
   - 其他成员推荐了这两个 **GitHub** 仓库：[esp32cam-rtsp](https://github.com/rzeldent/esp32cam-rtsp) 和 [ESP32-RTSPServer](https://github.com/rjsachse/ESP32-RTSPServer)，并建议使用 **PlatformIO with AI** 进行开发。
- **Gemini 2.5 和 Sonnet 3.7 变笨了？**：成员们反映，与通过命令行界面（**CLI**）使用相比，非 Max 版本的 **Gemini 2.5** 和 **Sonnet 3.7** 似乎变笨了，且幻觉更多。
   - 一位成员推测 **Cursor** 可能更改了系统指令或提示词，但另一位成员反驳称 *"Gemini 的问题应该已经修复，那是 Google 的问题"*。
- **Cursor 账单问题已处理**：一位成员询问因银行确认延迟导致发票未支付时的即时付款选项。
   - 另一位成员（可能是 **Cursor** 团队成员）提供了通过 **DM**（私信）直接协助，以快速解决账单问题。
- **呼吁开源模型评估**：一位成员建议 **Cursor** 应该开源针对不同模型的评估，并强调模型的实现比原始 **benchmarks** 更重要。
   - 他们提议进行基于工具的评估（tool-based evals），并特别询问是否存在 Cursor 特有的评估工具，因为 *"我发现模型的实现方式比单纯的原始 benchmarks 更重要。"*


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1365183026241212477)** (9 messages🔥): 

> `Python 互操作能力，Mojo 路线图，Mac GPU 支持，分享了可疑链接` 


- **Python 互操作能力即将发布！**：成员们对演示中提到的即将发布的更多 **Python interop capabilities** 表示期待。
- **Mojo 路线图依然难以捉摸**：**Mojo roadmap** 仍没有明确的时间表，有人幽默地提到了 *someday*（某一天）。
- **Mac GPU 支持将于今年夏天推出？**：根据 .duck_tape 的更新，**Mac GPU support** 暂定于 *夏季* 推出。
- **用户分享了可疑链接！**：一名成员报告称，另一名用户分享了一个可疑链接后删除了消息，并附上了截图供管理员审查 ([image.png](https://cdn.discordapp.com/attachments/1098713601386233997/1365186580586954863/image.png?ex=680d0d54&is=680bbbd4&hm=01f44ee0ff956f8c66b1ce1e30c78f2234d5ae244fa2ea0e3531f0038c9aecf0&))。
   - 另一名成员感谢了他们的举报，表示审核团队已获悉。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1365176336460546069)** (94 messages🔥🔥): 

> `Rust 中的 uom crate，Nm vs Joules，QuantityKind 标签，弧度混淆，整数支持` 


- **Rust 的 uom 处理混合单位运算**：Rust 中的 `uom` crate 允许混合不同单位（如 **Energy**、**Force** 和 **Length**）进行运算而不会报错，例如将 `Energy` 加到 `Force` 和 `Length` 的乘积上。
   - 一名成员认为这是可以接受的，类似于将 `Byte` 视为 `UInt8`，表明对单位交互采取了宽松的方法。
- **关于 Nm 与 Joules 的争论正在酝酿**：一场关于 **Newton-meters (Nm)** 是否可以等同于 **Joules (J)** 的讨论展开了，一方认为物理学上将它们区分开来，因为扭矩是矢量，而能量是标量。
   - 参考 [BIPM](https://www.bipm.org/documents/20126/41483022/SI-Brochure-9-EN.pdf)，讨论强调 *尽管扭矩与能量具有相同的量纲（SI 单位为焦耳），但焦耳从未用于表示扭矩*。
- **`QuantityKind` 标签可能可以消除物理量的歧义**：提议使用 **QuantityKind** 标签在类型层面上区分相似的单位，例如 **Nm** 和 **Joules**。
   - 实现方式可以是 `Q[Vec[dtype, 1], N * M]` 对比 `Q[Scalar[dtype], N * M]`，这样就无法将它们相加。
- **弧度混淆已澄清**：讨论涉及了关于弧度的潜在混淆，参考了 `uom` 使用类型标签处理角度的方法，详见其 [AngleKind trait](https://docs.rs/uom/latest/uom/si/marker/trait.AngleKind.html)。
   - Wolfram 的方法是使用 `alias °: FloatLiteral = pi / 180`。
- **整数支持的实现受到质疑**：团队辩论了在库中支持整数值是否是一个错误，考虑到精度和舍入的潜在问题，特别是在角度到弧度的转换中。
   - 一名成员承认 *工程师将数字视为有限精度的*，并使用 **3.14** 代表 pi。


  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1365415350749106338)** (1 messages): 

> `LM Studio 更新，NVIDIA RTX 50 系列支持，启用 GLM-4，新系统提示词编辑器，Tool Choice 参数` 


- **LM Studio 获得 RTX 50 系列助力**：**LM Studio 0.3.15** 现已发布，支持 **NVIDIA RTX 50-series** (CUDA 12.8)，可通过应用进行更新 ([LM Studio download](https://lmstudio.ai/download))。
   - 该更新还在 llama.cpp 和 MLX 中启用了 **GLM-4**，包含新的系统提示词编辑器 UI，并在类 OpenAI API 中引入了 `tool_choice` 参数。
- **LM Studio 增强 Llama 3，添加工具支持**：LM Studio 的最新版本修复了 **Llama 3 Scout** 提示词模板 + 工具使用 ([完整发布说明](https://lmstudio.ai/blog/lmstudio-v0.3.15))。
   - 0.3.15 版本还包含了社区共享预设的预览，并解决了多个错误修复。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1365274312466108508)** (75 messages🔥🔥): 

> `LLM 角色数据集准备, GGUF 转换脚本问题, LM Studio 对比 Ollama 的优势, Google AIStudio LoRA 训练, LM Studio 社区预设` 


- **LLM 角色数据集准备的困扰**：一位用户正尝试使用有限的数据集（**10k tokens**）让 LLM 说话像游戏中的角色，并寻求如何将数据加载到 LM Studio 的建议。
   - 他们不确定是否需要将数据转换为 **GGUF** 格式，以及转换后测试模型的具体步骤。
- **GGUF 转换脚本触发 Bug**：有用户报告在使用 [convert_hf_to_gguf.py 脚本](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py) 并安装其依赖后，即使重新安装了 **transformers**，他们的训练脚本仍然出现 Bug。
   - 他们后来通过在脚本中将 `" "` 替换为 `" "` 修复了此问题，并称该修复方式 *“非常奇怪”*。
- **LM Studio 比 Ollama 更易用**：一位用户询问 **LM Studio** 相比 **Ollama** 的优势。
   - 另一位用户回答说 **LM Studio** 更容易上手，特别是对于那些不知道 *“terminal”*（终端）是什么的人来说。
- **Google AIStudio 无法运行 LoRA**：一位用户询问 Google 的 **AIStudio** 是否可以训练 **LoRAs**，但另一位用户澄清说它无法直接与 **LM Studio** 配合使用，因为 **LoRAs** 必须与基础模型合并并转换为 **GGUF** 才能在 **LM Studio** 中正常运行。
   - 他们明确了正确的工作流：*寻找基础模型 -> 创建 LoRA -> 与基础模型合并 -> 转换为 GGUF，然后它就能在 LM Studio 中运行*。
- **预览版中的 LM Studio 社区预设**：用户讨论了 **LM Studio** 中新增的 **Community Shared Presets**（社区共享预设）功能，指出它目前处于 *“Preview”*（预览）阶段，缺乏发现标签页或浏览功能。
   - 他们对如何查找和下载预设感到困惑，建议 *discover -> presets* 应该是最自然的查找位置，有人表示这 *“让我怀疑其意图”*。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1365367883156422748)** (20 messages🔥): 

> `RTX 3060 对比 Intel B580 用于 AI, Xeon 的 AVX2 速度, OpenVINO 对比 llama.cpp` 


- **RTX 3060 在 AI 工作负载中更胜一筹**：成员们讨论了 **RTX 3060 12GB** 与 **Intel B580 12GB** 在 AI 方面的适用性，共识是由于更好的 **CUDA** 支持，**3060** 更具优势。
   - 一位成员表示 *“AI 世界的一切都是围绕 CUDA 构建的”*，这使得 Nvidia 的显卡更优越。
- **Xeon 上的 AVX2 速度对于 7-14b 模型来说不够快**：一位用户询问使用 **2690v4 Xeon**（**14 核心**，**3.5ghz 加速频率**，**135瓦**）的 **AVX2** 速度是否理想，另一位成员回答说纯 CPU 运行不会快，且取决于 RAM。
   - 另一位成员补充说，**q4 14b** 模型在双路 Xeon 上最高只能运行在 **6tps**，单路可能只有 **3-4 tps**。
- **OpenVINO 在 CPU 上的性能优于 llama.cpp**：一位成员分享了他们的 [OpenArc 项目](https://github.com/SearchSavior/OpenArc)，认为 **OpenVINO** 提供的 **CPU** 性能显著优于 **llama.cpp**。
   - 他们表示 *“从经验来看，差异是巨大的，尤其是 TTFT”*，并提到他们已经为 **Qwen2-VL**、**Qwen2.5-VL** 和 **Gemma3** 实现了视觉功能。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1365310061609484408)** (3 messages): 

> `灰度模式, ML 编译器, TVM/MLIR 项目` 


- **灰度模式：重命名为 Pantone？**：一位成员询问是否将 *grayscale mode*（灰度模式）重命名为 *pantone-scale*。
- **ML 编译器知识是先决条件吗？**：一位成员询问在从事 **ML Systems Programming**（ML 系统编程）职业时是否需要 **ML 编译器** 的知识，特别是提到了 **TVM/MLIR**。
- **开展 TVM/MLIR 项目**：一位主要关注压缩方向的成员正在考虑开展一个关于 **TVM/MLIR** 的小项目，但不确定这是否是个好主意。


  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1365349855752491059)** (4 条消息): 

> `Quantization, FP4, 5090, Matmul Kernel` 


- **FP4 Kernel 需要缩放逐元素乘法**：在 **5090** 上运行 **FP4** 后，一位成员意识到他们需要将缩放逐元素乘法（scaling elementwise multiplication）融合到这个 **matmul kernel** 中。
   - 另一位成员指出，由于这是对称量化（symmetric quant），需要一个好的算法来获取准确的缩放比例（scales）；否则如果只使用 min-max，质量可能会很差。
- **FP4 表示需要 scale_a 和 scale_b 进行逐元素乘法**：一位成员提到 **FP4** 的表示对于激活（activation）来说是不够的，因此会有 **scale_a** 和 **scale_b**（均为向量），并将与 FP4 matmul 的结果进行逐元素相乘。
   - 他们随后补充道：“即使对于权重也是如此。我还没亲自尝试过，只是根据对许多模型进行量化和基准测试后的观察所做的猜测。”


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1365358205177167933)** (2 条消息): 

> `NVIDIA's cuda-python library, CUDA Functionality` 


- **NVIDIA 的 cuda-python 库**：一位成员询问了 [NVIDIA 的 cuda-python 库](https://github.com/NVIDIA/cuda-python) 的质量，以及是否有人尝试过。
   - 消息中没有分享具体的经验或评价。
- **CUDA 功能对等性？**：一位成员询问 [cuda-python 库](https://github.com/NVIDIA/cuda-python) 是否提供与 **CUDA C/C++** 相同的功能。
   - 该成员表示，如果该库能实现功能对等，他对它的潜力感到乐观。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1365184365235146783)** (2 条消息): 

> `MI300 ISA, VGPRs, wavefront, occupancy` 


- **MI300 ISA 的 VGPRs 详解**：在 **MI300 ISA** 中，有人提出了一个问题：**256 VGPRs/wave** 是否等同于 **每条 lane 256 个有效 GPR**。
   - 答案是肯定的，但有人提醒说，使用 **255 VGPRs** 意味着低占用率（occupancy）。
- **VGPR 使用与占用率的影响**：在 **MI300 ISA** 中使用大量的 **VGPRs**（例如 **255**）会导致较低的占用率。
   - 较低的占用率可能会因为并行度降低和开销增加而影响性能。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1365179439633272922)** (15 条消息🔥): 

> `MI300, H100, amd-fp8-mm leaderboard, amd-identity leaderboard, vectoradd leaderboard` 


- **MI300 AMD-FP8-MM 个人最佳成绩**：多位成员在 **MI300** 的 `amd-fp8-mm` 排行榜上刷新了个人最佳成绩，时间包括 **4.93 ms**、**2.74 ms**、**2.53 ms**、**5.19 ms**、**5.28 ms**、**805 µs** 和 **5.25 ms**。
   - 其中一次提交达到了 **1247 µs**，而另一次以 **289 µs** 的成绩获得了 **第 6 名**。
- **AMD-Identity 排行榜成功**：一位成员在 **MI300** 的 `amd-identity` 排行榜上取得了 **23.7 µs** 的个人最佳成绩。
- **H100 VectorAdd 个人最佳成绩**：一位成员在 **H100** 的 `vectoradd` 排行榜上取得了 **626 µs** 的个人最佳成绩。


  

---

### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1365196189301080064)** (17 条消息🔥): 

> `Profiling 信息, 个人最佳耗时计算, 基准测试时 Kernel 失败, 基准测试形状, Azure MI300 计算时间` 


- **性能分析 (Performance Profiling) 调查**：成员们讨论了性能分析支持的可能性，指向了一个特定的 [Discord 消息](https://discord.com/channels/1189498204333543425/1343350424253632695/1364647936386142258) 以获取 **SSH 访问权限**，并提到可能会重新审视 **uprof**。
   - 对话表明 **uprof** 之前已被探索过，但可能会被重新考虑，并计划与特定用户进行进一步讨论。
- **个人最佳耗时 (Personal Best Timing) 揭秘**：一位用户询问了个人最佳耗时是如何计算的，并得知它是**所有形状的平均值**，个人最佳表示比之前的平均值有所提高。
   - 讨论链接到了 [开源评估基础设施](https://github.com/gpu-mode/discord-cluster-manager) 以提供更多背景信息。
- **Kernel 尽管通过测试但仍崩溃**：一位用户报告称 Kernel 通过了所有测试，但在基准测试期间失败，怀疑存在隐藏测试，并收到反馈称退出代码 **114** 表示**超时 (timeout)**。
   - 该用户包含了输出结果，显示基准测试命令 `python eval.py benchmark /tmp/tmp0uc5ldzn` 在 **AMD Instinct MI300X** 上运行 120 秒后返回了退出代码 **114**。
- **基准测试形状 (Benchmark Shapes) 不平衡**：一位在基准测试中遇到困难的用户被引导至 [基准测试形状](https://github.com/gpu-mode/reference-kernels/blob/b68a149bcd8701532eeedc774d27062429ce4f99/problems/amd/fp8-mm/task.yml#L66)，并注意到这些形状可能不平衡。
   - 形状已公开，然而该用户声称它们看起来并不比测试用例差。
- **AMD GPU 时间供应商建议**：一位寻求负担得起的 MI300 计算时间进行 Profiling 的用户获知，前 8 名将获得免费 **MI300** 时间作为竞赛奖励，并建议查看 **Tensorweave**。
   - 有人澄清说，奖励是指服务器上的 MI300 GPU 时间，而不是实际的显卡，因为*大规模分发 GPU 确实很困难*。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1365236491088957555)** (5 条消息): 

> `Prompt engineering 更新, Zotero 集合集成, 使用 NotebookLM 分析美国法典 (US Code), 文件大小限制` 


- **近期更改后 Prompt 需要更新**：一位成员提到，由于最近的更新，*他们的许多旧 Prompt 不再起作用*，这表明需要重新审视和修改 Prompt engineering 策略。
- **寻求 NotebookLM 的 Zotero 集成**：一位用户询问如何将他们的 **Zotero** 集合连接到 **NotebookLM**，而无需逐个添加 PDF 作为来源。
- **美国法典 (US Code) 分析遇到大小限制**：一位成员正在使用 **NotebookLM** 分析 **US Code** 的冗余，但面临 PDF 和字数限制，即使尝试了 Gemini 建议的代码或 XML 文件也是如此。
   - 另一位成员建议将代码拆分为较小的文件作为权宜之计。


  

---

### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1365180175809118329)** (31 条消息🔥): 

> `Content Security Policies 错误、Gemini 聊天删除、NotebookLM 使用场景、源文件上传错误、从 Plus 降级到 Free` 


- **Gemini 中疑似存在 CSP 错误**：用户报告了可能与 Gemini 中的 **Content Security Policies (CSP)** 相关的错误。
   - 另一位用户遇到 Gemini 有时会删除聊天中最顶部的对话。
- **NLM：专注的专家教师**：一位用户建议，主要使用场景是学习知识，NLM 提供了为特定领域或主题创建 **Focused expert**（专注专家）的选项，可以充当 **教师/教练/导师**。
   - 他们补充道：*由于源文件是回答背后的事实依据，NLM 可以消除或限制噪音*。
- **文件大小限制导致上传问题**：一位用户在上传源文件时遇到 **Error**，其他用户建议文件可能超过了 **200MB 限制**。
   - 另一位成员表示：*只有 26 Mb*。
- **Free 与 Plus 账户的区别**：当被问及从 **Plus** 降级到 **Free** 时，有一个关于如果笔记本中的源文件超过免费计划允许数量会发生什么的问题。
   - 另一位用户指出，**Free 账户** 限制为 **100 个 Notebook** 且每个笔记本 **50 个源文件**，而 **Plus 账户** 可以拥有 **500 个 Notebook** 且每个笔记本 **300 个源文件**。
- **Workspace 管理员设置限制共享**：一位通过 **Google Workspace** 拥有 **Plus 账户** 的用户无法与组织内的其他人共享完整的笔记本。
   - 建议可能需要在 Workspace 管理员的 *Groups for business* 下启用 *Group sharing*。


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1365348380217184317)** (26 条消息🔥): 

> `Claude Desktop 向 MCP 工具上传图片、MCP Server 公开描述更新、MCP Hooks 的 JSON 格式错误、MCP Server Streamable 客户端支持、mcp-remote 用法及与 Supergateway 的对比` 


- **Claude 的 MCP 工具图片上传**：一位用户不确定 **Claude Desktop** 如何将上传的图片传递给 **MCP tools**，质疑 Claude 是自动检测最近的上传还是有特定的引用方法。
   - 该用户的工具支持图片 URL 但不支持直接上传，正在寻求关于实现接受上传图片的 **MCP tools** 的示例或文档。
- **MCP 公开描述故障**：一位用户注意到，尽管在管理面板中进行了更改，但其 **MCP server** 的 **公开描述** 在公共页面上并未更新，具体涉及 `dataryan.cph`。
   - 另一位用户报告遇到了同样的问题。
- **JSON 反引号错误**：一位用户报告称，Claude 在为 SDK 创建 **MCP hooks** 时坚持使用 **反引号**，想知道是否配置有误。
   - 另一位用户建议尝试移除 `dataToWrite` 中的引号。
- **寻找 MCP Streamable 客户端**：一位用户正在寻找支持与 **MCP server streamables** 连接的客户端，因为他们的服务器只能在自己的客户端运行，无法在 AnythingLLM、Claude 或 VSCode Copilot 中工作。
   - 一位用户提到 *mcp-remote* 支持 mcphttps，并链接到了 [草案版本](https://github.com/geelen/mcp-remote/pull/32)。
- **mcp-remote 与 Supergateway 对比**：一位用户询问 `mcp-remote` 的功能是否与 **Supergateway** 类似，另一位用户确认两者相似但范围更小。
   - 一位用户分享了 `mcp-remote` 的配置，但报错 *SSE error: Non-200 status code (404)*。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1365332515488796762)** (2 条消息): 

> `Kubernetes MCP、GitHub Repo MCP、StacklokLabs mkp` 


- **Kubernetes MCP 获得新实现**：一位成员宣布他们决定构建一个新的 [基于 k8s API](https://github.com/StacklokLabs/mkp) 的 Kubernetes MCP，使其更具通用性和灵活性。
   - 它旨在作为获取 GitHub 仓库的第一个 MCP Server 的替代方案，允许 **AI 使用仓库代码作为参考**。
- **GitHub Repo MCP**：一位成员展示了他的第一个用于获取 GitHub 仓库的 MCP Server 及其代码：[github-repo-mcp](https://github.com/Ryan0204/github-repo-mcp)。
   - 这允许 **AI** 使用仓库代码作为参考。


  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1365379745595785338)** (2 messages): 

> `CondoScan, LlamaIndex, LlamaParse, Agents` 


- **Agent 构建热潮爆发！**: [AnthropicAI 发布了《构建高效智能体》(Building Effective Agents)](https://www.anthropic.com/)，**dexhorthy** 的 [《12 因子智能体》(12 Factor Agents)](https://www.12factor.net/) 走红，且 [OpenAI 发布了《智能体构建实用指南》(A Practical Guide To Building Agents)](https://platform.openai.com/docs/guides/function-calling)！
   - 社区中关于什么是 Agent 以及构建它们的最佳方式展开了大量**富有成效的讨论**。
- **CondoScan 全面降低公寓评估成本**: **CondoScan** 利用 **LlamaIndex** 的 **Agent workflows** 和 **LlamaParse** 精确的文档处理能力，打造了下一代公寓评估工具，将文档审查时间从数周缩短至数分钟。
   - 在[此案例研究](https://t.co/SzIbcKta1O)中了解更多关于 **CondoScan** 如何*评估财务健康状况和生活方式契合度*的信息。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1365262478413664278)** (19 messages🔥): 

> ``memory` vs `chat_history`, AgentWorkflow 错误, FunctionCallingAgent 超时, FunctionAgent 超时` 


- **探讨 `memory` 与 `chat_history` 的区别**: `chat_history` 已集成到 `memory` 中。如果你正在管理聊天消息或使用消息列表进行初始化，请使用 `chat_history`；否则，请使用特定的 memory 模块。
   - 本质上，*如果你只是自己管理聊天消息列表，或者想用某些聊天消息列表进行初始化，请使用 chat_history；如果你正在维护/使用特定的 memory 模块，请使用 memory*。
- **通过 GoogleGenAI 更新解决 AgentWorkflow 错误**: 报告了 `AgentWorkflow` 中的一个间歇性错误，追溯到错误信息：*400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'Please ensure that the number of function response parts is equal to the number of function call parts of the function call turn.', 'status': 'INVALID_ARGUMENT'}}*。
   - 该错误已通过 `pip install -U llama-index-llms-google-genai` 升级解决，详见[此 GitHub pull request](https://github.com/run-llama/llama_index/pull/18527)。
- **`FunctionCallingAgent` 缺乏原生超时设置**: `FunctionCallingAgent` 类缺少直接的超时参数；不过，可以在 LLM 对象上为每次 LLM 调用设置超时。
   - 注意到 [`FunctionCallingAgent`](https://llama_index.readthedocs.io/en/stable/understanding/agent/) 是一个较旧的类，而像 `FunctionAgent` 这样较新的 Agent 类通过 `request_timeout` 参数直接支持超时。
- **`FunctionAgent` 提供超时功能**: 较新的 `FunctionAgent` 类支持超时配置，并提供了一个使用 `Ollama` 并将 `request_timeout` 设置为 360 秒的代码示例。
   - 一位成员提供了如何使用它的示例：
```python
agent = FunctionAgent(
 tools=[multiply],
 llm=Ollama(model="llama3.1", request_timeout=360.0),
 system_prompt="You are a helpful assistant that can multiply two numbers.",
)
```


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1365214439602782228)** (8 messages🔥): 

> `DSPy support for multimodal models, Chain of Thoughts and ReACT pattern, Frameworks like Langchain and LlamaIndex, Multimodal Reasoning Models, Ember and Compound AI Systems` 


- **DSPy 与多模态建模的思考**：鉴于推理模型的进展，小组讨论了 **DSPy** 是否支持**多模态模型**和**多模态推理工作流**。
   - 一位成员认为 **Chain of Thoughts (CoT)** 现在更加强大，如果使用 **CoT** 进行工具选择，**ReACT** 可以做出更好的决策。
- **框架面临快速变化的局面**：有人指出 **AI 领域** 变化极快，使得像 **Langchain** 和 **LlamaIndex** 这样的框架难以跟上步伐。
   - 建议是使用能简化工作的框架，并在需要时允许直接使用 **model API**。
- **新想法涌现：Ember**：新的想法正在出现，例如 [Ember](https://github.com/pyember/ember)，这可能需要以不同的方式思考如何构建 **compound AI system**。
   - **DSPy** 提供**声明式语法**，而 **Ember** 提供不同的方法，两者各有权衡。
- **文本和图表获得业务助力**：出于业务原因，大多数框架专注于**文本和图表**；面向业务的系统通常需要对文本和表格进行推理。
   - 在预处理阶段，图像通常使用 **VLM (用于 OCR)** 转换为**文本、JSON 结构或表格**。
- **对代码深度分析的需求**：一位成员希望有一种方法可以分析大型代码文件（超出上下文长度），以从中提取最大价值。
   - 他们明确表示需要深度分析，而非通用的需求描述，旨在最大限度地挖掘文件内容。


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1365351789540409447)** (1 messages): 

> `Torchtune's GRPO code sharing` 


- **请求共享 Torchtune 的 GRPO 代码**：一位成员询问共享 **Torchtune GRPO** 代码的工作量有多大。
   - 该成员推测这些代码会非常有用，并对更改的侵入性表示好奇，暗示 **Torchtune GRPO** 的其他用户也会有兴趣查看这些代码。
- **Torchtune GRPO 讨论**：一位用户询问了共享 Torchtune GRPO 代码的可能性。
   - 他们对其潜在效用和实现的更改程度表示关注，预见到其他 Torchtune GRPO 用户也会感兴趣。


  

---


### **Torchtune ▷ #[rl](https://discord.com/channels/1216353675241590815/1360680363885854841/1365415269148921906)** (4 messages): 

> `PPO Epochs, GRPO Data Padding` 


- **PPO Epochs 对 KL 散度估计的影响**：一位成员质疑 **Torchtune** 中的 `ppo_epochs > 1` 是否会导致 KL 散度估计产生偏差，并指向了 [此文件](https://github.com/joecummings/r1-zero/blob/main/torchtune/dev/grpo/loss.py#L85-L87) 的第 85-87 行。
   - 他们认为，在遍历过一次重放缓冲区 (replay buffer) 中的每个样本后，策略 (**pi_theta, pi_ref, pi_old**) 已经发生了变化，因此样本不再来自当前的策略 **pi_theta**。
- **讨论 GRPO 的数据填充方向**：一位成员注意到 **GRPO 数据填充** 方向是在右侧，并询问 Decoder 模型在训练期间是否应该在左侧填充。
   - 另一位成员回答说，只要[正确处理输入位置和掩码 (masks)](https://github.com/pytorch/torchtune/blob/main/recipes/dev/grpo_full_finetune_distributed.py#L750)，两侧都可以进行填充。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1365330036961448078)** (5 messages): 

> `Memory Requirements for LLMs, Alternatives to Rust Coding, Llama4 and Scout 17x16` 


- **低内存需求也能处理大型 LLM**：一位用户建议 **16GB** RAM 足以运行 **32B** 模型，如果能忍受较低的质量和速度，**8GB** 也可以运行 **70B** 模型。
   - 他们还指出，逐行进行 LLM 提示 (prompting) 的实现相对简单。
- **用 Shell 脚本替代 Rust**：一位成员建议可以用短小的 Shell 脚本替代 Rust 代码，暗示某些任务有更简单的方法。
   - 未提供更多细节。
- **Llama4 集成的障碍**：一位成员询问如何让 **Llama4** 正常工作，特别是配合 **scout 17x16**，并质疑是否需要更新代码或 **Jinja** 配置。
   - 另一位用户回应称 *gpt4all* 已经过时，建议探索其他选项，随后原成员放弃了尝试。


  

---

### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1365414595824717854)** (2 条消息): 

> `医疗保健 AI/ML 初创公司寻求 Prompt Optimization，Cohere 社区 Discord 服务器自我介绍` 


- **医疗保健初创公司寻求 Beta 版 Prompt 功能**：一家处于快速增长阶段且专注于影响力工作的医疗保健 **AI/ML 初创公司**已注册 Beta 版 **Prompt Optimization** 功能，但目前尚无访问权限。
   - 该初创公司正在询问访问权限是否受 **user seniority** 或 **billing tier** 的限制。
- **鼓励新成员进行自我介绍**：服务器欢迎新成员，并鼓励他们向社区介绍自己。
   - 新成员被引导分享其**公司/行业/大学**、当前项目、偏好的技术/工具以及在社区的目标。

---


您收到此邮件是因为您通过我们的网站选择了订阅。

想要更改接收这些邮件的方式吗？
您可以从该列表中[退订]({{{RESEND_UNSUBSCRIBE_URL}}})。