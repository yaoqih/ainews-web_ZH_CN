---
companies:
- openai
- anthropic
- alibaba
- meta-ai-fair
- huggingface
- granola
date: '2025-05-14T05:44:39.731046Z'
description: '**GPT-4.1** 现已在 **ChatGPT** 中向 Plus、Pro 和 Team 用户开放，重点提升了编程和指令遵循能力，同时
  **GPT 4.1 mini** 取代了 **GPT 4o mini**。**Anthropic** 正在发布包括 **Claude Opus** 和 **Claude
  Sonnet** 在内的全新 **Claude** 模型，尽管有批评指出 **Claude O3** 存在幻觉问题。**阿里巴巴**分享了 **Qwen3 技术报告**，其中
  **Seed1.5-VL** 展现了强劲的基准测试结果。**Meta FAIR** 宣布了新的模型和数据集，但 **Llama 4** 遭到了批评。**AM-Thinking-v1**
  作为一款 32B 规模的推理模型在 **Hugging Face** 上线。**Granola** 在 B 轮融资中筹集了 4300 万美元，并发布了具有类 Notion
  界面的 **Granola 2.0**。AI 生态系统正显示出快速迭代和创意克隆的态势，强调执行力和分发能力。'
id: MjAyNS0w
models:
- gpt-4.1
- gpt-4o-mini
- gpt-4.1-mini
- claude-opus
- claude-sonnet
- claude-o3
- qwen3
- seed1.5-vl
- llama-4
- am-thinking-v1
people:
- kevinweil
- scaling01
- steph_palazzolo
- andersonbcdefg
- reach_vb
- yuchenj_uw
- qtnx_
- _akhaliq
- risingsayak
title: Granola 推出团队笔记功能，而 Notion 则上线了会议转写功能。
topics:
- coding
- instruction-following
- benchmarking
- model-releases
- reasoning
- image-generation
- collaborative-software
- model-performance
---

**Whisper is all you need.**

> 2025年5月13日至5月14日的 AI 新闻。我们为您检查了 9 个 Reddit 子版块、449 个 Twitter 账号和 29 个 Discord 社区（214 个频道，4313 条消息）。预计节省阅读时间（以 200wpm 计算）：428 分钟。我们的新网站现已上线，支持全文元数据搜索，并以精美的风格展示了所有往期内容。请访问 https://news.smol.ai/ 查看完整的详细新闻，并在 @smol_ai 上向我们提供反馈！

我们尽量将报道范围集中在模型和代码相关的特定新闻上，我们确信工程师们将来会在工作中使用这些内容，但偶尔一些较小的产品发布也是对更广泛 AI 领域进行评论的有趣素材，特别是如果这些发布涉及像 Notion 或 Granola 这样备受推崇的产品。

生物学中有一个经久不衰的笑话，即[万物皆演化为蟹](https://en.wikipedia.org/wiki/Carcinisation)。AI 封装（wrapper）领域也正在发生同样的事情——仅仅因为它们[现在被公认为具有价值](https://news.smol.ai/issues/25-05-05-cursor-openai-windsurf)，并不能阻止它们仍然容易被克隆。[Bolt](https://www.latent.space/p/bolt?utm_source=publication-search) 启发了 [Figma Make](https://www.figma.com/make/)，[Claude Code](https://www.latent.space/p/claude-code) 启发了 OpenAI Codex，[Deep Research](https://www.youtube.com/watch?v=eJOjdjO45Sc) 启发了 Deep Research 启发了 Research 启发了 DeepSearch，如此循环往复。**想法一文不值，愿最强的分发 + 执行力获胜。**

Granola 获得 4300 万美元 B 轮融资（[估值 2.5 亿美元](https://x.com/TechCrunch/status/1922654404424958054)）之际，正是他们发布“[Granola 2.0](https://x.com/cjpedregal/status/1922663281233142074)”的时机，这是他们的协作版本，带有一个令人惊讶的……Notion 风格的 UI。

[](https://resend-attachments.s3.amazonaws.com/HmO1Qvr6inIM7fJ)

就在前一天，Ivan Zhao 发布了……[一个有趣的轻量级 Granola 功能](https://x.com/ivanhzhao/status/1922312312486297857)。

[](https://resend-attachments.s3.amazonaws.com/yRlDTOEzKA67mil)

---

# AI Twitter 回顾

**语言模型与发布**

- **GPT-4.1 可用性**：[@OpenAI](https://twitter.com/OpenAI/status/1922707554745909391) 宣布 **GPT-4.1** 将直接在 **ChatGPT** 中面向 Plus、Pro 和 Team 用户开放，Enterprise 和 Education 用户将在未来几周内获得访问权限，该模型专注于 **coding tasks (编程任务) 和 instruction following (指令遵循)**。[@kevinweil](https://twitter.com/kevinweil/status/1922732062345142306) 指出 **GPT 4.1 mini** 正在 ChatGPT 的所有版本中取代 **GPT 4o mini**，包括免费用户。
- **Claude 模型**：[@scaling01](https://twitter.com/scaling01/status/1922671998427111624) 对即将推出的 **Claude Opus** 表示兴奋，并期待诸如 **Ultra** 和基于 **GPT-4.5** 的推理模型。[@steph_palazzolo](https://twitter.com/steph_palazzolo/status/1922655594076323994) 分享了关于 Anthropic 即将发布的 **Claude Sonnet** 和 **Claude Opus** 的信息，并指出了它们不同的推理模型。然而，[@andersonbcdefg](https://twitter.com/andersonbcdefg/status/1922116734938484819) 批评说 **Claude 现在变得很笨 (braindead)**，而 **O3 会胡编乱造，让你陷入幻觉 (hallucinations) 的深渊**。
- **Qwen 模型**：[@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1922265772811825413) 分享了 **Qwen3 Technical Report**，详细介绍了模型细节和完整评估。[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1922226964599095740) 强调 **Seed1.5-VL 在 60 个公开 VLM 基准测试中的 38 个上实现了 state-of-the-art 结果**。[@reach_vb](https://twitter.com/reach_vb/status/1922322833847300156) 向团队表示祝贺，[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1922294726209724656) 赞扬了 Qwen 团队的出色工作。[@qtnx_](https://twitter.com/qtnx_/status/1922398353985241438) 也对 Qwen 团队在 **600M 模型上投入 36 万亿 (TRILLION) tokens** 的惊人且有趣的举动表示敬意。
- **Meta 的 AI 进展**：[@AIatMeta](https://twitter.com/AIatMeta/status/1922690879279808572) 宣布了 **Meta FAIR** 的新发布，包括用于分子属性预测、语言处理和神经科学的模型、基准测试和数据集。然而，[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1922696849741447421) 批评了 **Meta 的 AI**，特别是 **Llama 4**，指出其存在忽略附件图片和登录失败的问题。
- [@_akhaliq](https://twitter.com/_akhaliq/status/1922647377569063296) 宣布 **AM-Thinking-v1** 刚刚在 Hugging Face 上发布，称其为 **32B 规模推理前沿 (Frontier of Reasoning)** 的一项进步。
- [@RisingSayak](https://twitter.com/RisingSayak/status/1922213888168173960) 宣布了一个新的 **适用于 SANA Sprint 的 Diffusers 兼容训练脚本**。
- **Gemini 2.0 Flash Preview**：[@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1922659105048821984) 报告称，**Gemini 2.0 Flash Preview** 的图像生成相比 2.0 Flash Experimental 版本有小幅升级，但仍远低于 state-of-the-art 水平。[@HamelHusain](https://twitter.com/HamelHusain/status/1922119981526880515) 表示 **Gemini 以惊人的准确度一次性完成了这些章节摘要**。
- **Stability AI** 刚刚在 Hugging Face 上发布了 **Stable Audio Open Small**。[@_akhaliq](https://twitter.com/_akhaliq/status/1922727993429913886) 指出这是一个**具有对抗性后训练 (Adversarial Post-Training) 的快速文本转音频生成模型**。

**Agent 开发与工具**

- **LangChain Interrupt**: [@LangChainAI](https://twitter.com/LangChainAI/status/1922745714246906086) 发布了来自 Interrupt 2025 的更新，重点关注评估（evals）、质量和可靠性，并强调 **质量仍然是将 Agent 投入生产的最大阻碍**。[@LangChainAI](https://twitter.com/LangChainAI/status/1922722850542346680) 还推出了 **Open Agent Platform**，这是一个开源、无代码的 Agent 构建平台。[@LangChainAI](https://twitter.com/LangChainAI/status/1922709747423183226) 宣布 **LangGraph Platform 现已全面上市 (GA)**，专为部署、扩展和管理 Agent 而设计。
- **LlamaIndex Memory 组件**：[@llama_index](https://twitter.com/llama_index/status/1922340015499313543) 推出了一种全新的、灵活的 **Memory API**，通过即插即用的模块将短期聊天历史和长期记忆融合在一起。
- **Runway References 更新**：[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1922656353354412332) 分享了最新 **References 更新** 带来的酷炫应用案例。
- [@LiorOnAI](https://twitter.com/LiorOnAI/status/1922675983624077347) 宣布了来自 **@PatronusAI** 的调试工具，该工具可以 **扫描完整的执行追踪，检测 60 多种失败类型，并提供 Prompt 修复建议**，支持 **LangChain, CrewAI, OpenAI SDK 等**。
- **Model Context Protocol**：[@AndrewYNg](https://twitter.com/AndrewYNg/status/1922671569429766178) 宣布与 **Anthropic** 合作推出关于 **MCP** 的新课程，重点是构建富上下文的 AI 应用。[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1922676799323910387) 也宣布了与 Anthropic 合作的 **MCP** 新课程。[@jerryjliu0](https://twitter.com/jerryjliu0/status/1922460369345511781) 为 Agent 记忆引入了一种新的抽象，将其建模为瀑布式架构中的一组“区块（blocks）”。
- [@*nerdai*](https://twitter.com/_nerdai_/status/1922732119706698118) 介绍了 **FedRAG**，这是一个用于微调 RAG 系统的框架，重点在于简化集中式和联邦式架构下的微调过程。
- [@LiorOnAI](https://twitter.com/LiorOnAI/status/1922306849795101044) 注意到 **OpenAI 悄悄发布了他们的 GPT-4.1 Prompting 指南**，并表示如果你在使用 Agent 或 LLM，这是必读内容。
- [@steph_palazzolo](https://twitter.com/steph_palazzolo/status/1922294062322704632) 观察到编程助手正在向 **常驻 Agent 演进，它们会在后台不断搜索 Bug 和漏洞**。

**AI 基础设施与工具**

- **Hugging Face 与集成**：[@reach_vb](https://twitter.com/reach_vb/status/1922672596216070154) 指出，现在可以直接在 Kaggle notebooks 上使用来自 Hugging Face 的任何模型。[@ClementDelangue](https://twitter.com/ClementDelangue/status/1922398990923968749) 提到 **看到 @PyTorch 在 @huggingface 上做出贡献非常酷**。[@_akhaliq](https://twitter.com/_akhaliq/status/1922315470478139537) 报道称 **通过 Inference Endpoints 可以实现极速的 Whisper 转录**。
- **vLLM 增强**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1922383289408491629) 报告称 **得益于 Hugging Face Inference Endpoints 和 @vllm_project，@openai Whisper API 速度提升了 8 倍且价格更便宜！**。[@vllm_project](https://twitter.com/vllm_project/status/1922353672534507610) 向 FlashInfer 表示祝贺，[@danielhanchen](https://twitter.com/danielhanchen/status/1922345308916216087) 分享了 **针对 Qwen3 Base 的新 GRPO notebook，并表示 Unsloth 现在也支持 vLLM 0.8.5！**
- **Keras 更新**：[@fchollet](https://twitter.com/fchollet/status/1922719664859381922) 讨论了直接从基类创建 KerasHub 预训练组件的方法。
- **Model Context Protocol**：[@AndrewYNg](https://twitter.com/AndrewYNg/status/1922671569429766178) 解释说，MCP 使 AI 开发不再那么碎片化，并标准化了 AI 应用程序与外部数据源之间的连接。
- **Deep Learning AI** 推出了数据分析专业证书的第 4 门课程，其中包括 **使用 Python 和 SQL 进行数据 I/O 与预处理**。在整个课程中，你将学习如何使用生成式 AI 来帮助调试和优化数据流水线。[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1922314062144139739)
- [@skypilot_org](https://twitter.com/skypilot_org/status/1922341585250881967) 报告了如何通过 **一条命令在 H100 上启动 Qwen3 @Alibaba_Qwen + SGLang @lmsysorg**。

**AI 与研究概念**

- **用于算法发现的 AlphaEvolve**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1922669321559347498) 介绍了 **AlphaEvolve，这是一个由 Gemini 驱动的用于算法发现的编码 Agent**，能够设计更快的矩阵乘法算法，寻找开放数学问题的新解决方案，并使数据中心、芯片设计和 AI 训练更加高效。[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1922669334142271645) 进一步指出，在 75% 的案例中，它重新发现了迄今为止已知的最佳解决方案。
- **对自回归（Auto-regression）的批评**：[@francoisfleuret](https://twitter.com/francoisfleuret/status/1922174021619097741) 分享了一个激进观点，认为 **Auto-regression 很糟糕，作为一种雕虫小技虽然令人印象深刻**，但 LLM 的任何智能火花都反映出它已经超越了这一点，并构建了一个具有意义隐变量（latents）的分解模型。
- **评估方法**：[@BorisMPower](https://twitter.com/BorisMPower/status/1922080385514504572) 强调，创建评估（evaluations）是在任何领域提高模型性能最有效的方法。
- **实现的重要性**：[@hyhieu226](https://twitter.com/hyhieu226/status/1922707456771195390) 强调，深度学习大约是 10% 的想法和 90% 的实现（implementation）。
- **LLM 与语法**：[@LiorOnAI](https://twitter.com/LiorOnAI/status/1922366980062838847) 解释了为什么在 90% 英文数据上训练的 LLM 在其他语言中仍然表现出色。它们学习了共享的语法概念，模型不仅仅是记忆单词层面的模式。
- [@shaneguML](https://twitter.com/shaneguML/status/1922172768432316519) 分享了李小龙著名的 LLM 研究员版名言：“我不怕练习过一万种问题一次的 LLM，但我怕练习过一个问题一万次的 LLM。”
- **类型约束的代码生成：**[@mathemagic1an](https://twitter.com/mathemagic1an/status/1922449795425198209) 分享了关于“使用大语言模型进行类型约束的代码生成”，称其利用 LSP/类型系统在代码生成过程中约束有效的输出 Token，并使 30B 模型的编译错误减少了 50% 以上。

**行业、商业与经济影响**

- **企业中的 AI：**[@AIatMeta](https://twitter.com/AIatMeta/status/1922336057405931850) 分享了他们的研究，**CATransformers 是一个碳驱动的神经架构和系统硬件协同设计框架**，发现了更环保的 CLIP 模型，在整个生命周期的总碳排放量方面平均实现了 9.1% 的减排潜力。
- **AI 技能集：**[@NandoDF](https://twitter.com/NandoDF/status/1922362860820165019) 表示，如果你是一名优秀的数据工程师，或者是一名热爱观察数据并为游戏、视频、图像、音频、文本创建数据集的工程师……请给我发消息。
- [@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1922100771392520710) 认为，如果软件优化真正成为首要任务，世界上比许多人想象的更多部分可以在过时的硬件上运行。
- **克服责任心**（Conscientiousness）作为一名企业家是一个常见的主题。[@scottastevenson](https://twitter.com/scottastevenson/status/1922654774677106939) 认为，有责任心的人经常被简单的、结构化的多巴胺奖励所吸引，比如清理桌面或跑腿办事。
- **需求的重要性**：[@rishdotblog](https://twitter.com/rishdotblog/status/1922111290451100024) 总结了 @ejames_c 的文章，指出 **无法找到真实需求会扼杀初创公司**。
- **对就业的影响：**[@cto_junior](https://twitter.com/cto_junior/status/1922631519673200727) 认为，这里的大多数“软件工程师”只是代码猴子（code monkeys），对整体系统的运作方式没有见解，如果不进行技能提升，他们肯定会被取代。

**幽默与杂项**

- [@sama](https://twitter.com/sama/status/1922372362420256960) 宣称 **brian 是这一代最具作者风格（auteur）的创始人，这在他进行发布会的方式中得到了充分体现！**
- [@typedfemale](https://twitter.com/typedfemale/status/1922051667081503028) 说：“我现在快 30 岁了（顺便说一下是女性）。这听起来可能很奇怪，但我真的认为上帝把我放在这个地球上是为了给轻度自闭症男性的生活带来温暖。”
- [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1922670826265919924) 说：“很高兴你一直在记录我的失败（Ls）😂。”
- [@victkyatk](https://twitter.com/vikhyatk/status/1922543155435462885) 写道：“成为有史以来最伟大的 ML 研究员一定很烦人。”
- [@francoisfleuret](https://twitter.com/francoisfleuret/status/1922547620196520142) 宣称：“仅仅因为我的热情，我就值得任何数额的薪水。”

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

### 1. 针对本地 LLM 推理的 AMD Strix Halo 和 Qwen3 模型基准测试

- [**AMD Strix Halo (Ryzen AI Max+ 395) GPU LLM 性能**](https://www.reddit.com/r/LocalLLaMA/comments/1kmi3ra/amd_strix_halo_ryzen_ai_max_395_gpu_llm/) ([Score: 104, Comments: 43](https://www.reddit.com/r/LocalLLaMA/comments/1kmi3ra/amd_strix_halo_ryzen_ai_max_395_gpu_llm/)): **该帖子对 AMD Strix Halo (Ryzen AI Max+ 395) GPU 进行了基准测试，该 GPU 拥有 40 个 RDNA3.5 CU，峰值性能为 59.4 FP16/BF16 TFLOPS，测试环境为 Linux，使用 llama.cpp 和其他框架进行 LLM 推理。使用 hipBLASLt 的原始计算效率达到 36.9 TFLOPS（超过理论值的 60%），但 llama.cpp 的 HIP 后端表现不佳（例如 Llama-2-7B Q4_0 仅为 348.96 tokens/sec），远低于 Vulkan (881.71 t/s)、Apple M 系列以及 780M 和 7900 XTX GPU 的预期效率。Vulkan 后端——凭借最近的 Flash Attention (FA) 支持——提供了最佳的 Prompt 和 Token 生成速度（例如 Llama-2-7B Q4_0 为 884 t/s），而 HIP+rocWMMA+FA 在长上下文方面表现出色（在 8K 上下文时几乎没有性能下降）。测试还包括 Qwen3-30B/109B 和 Llama 4（最高达 57.9 GiB 的模型），显示出 Vulcan 在超大模型上提供了很高的 tg128，而 ROCm 和软件栈成熟度（特别是 PyTorch FA）仍是瓶颈。ROCm 6.4、AOTriton 和 Composable Kernel 已确认可以构建并运行，但 PyTorch Flash Attention 在该硬件上仍然失败。有用的参考：[Strix Halo 基准测试结果](https://llm-tracker.info/_TOORG/Strix-Halo)。** 评论者指出，对于 Llama-2-7B Q4_0，该 GPU 达到了理论内存带宽的 79%，对于 Qwen3 32B Q8 达到了 87%——根据合成基准测试，其效率高于大多数传统系统。其他人要求对更高精度的模型进行测试（例如大上下文下的 Qwen 32B Q8），并关注正在进行的 ROCm 和 PyTorch 开发线程（[ROCm#4499](https://github.com/ROCm/ROCm/issues/4499), [ROCm/TheRock#244](https://github.com/ROCm/TheRock/discussions/244)）。
    - 强调了改进 AMD GPU 的 PyTorch 支持的持续努力，并直接引用了活跃的 ROCm 开发讨论和问题追踪。技术读者被引导至 [ROCm/ROCm issue #4499](https://github.com/ROCm/ROCm/issues/4499) 和 [ROCm/TheRock discourse #244](https://github.com/ROCm/TheRock/discussions/244)，表明关注点在于为 AMD 硬件上的 PyTorch 用户优化库和兼容性。
    - 基准测试结果显示，Llama-2-7B-GGUF Q4_0 模型实现的吞吐量达到了理论内存带宽的 79%，而 Qwen3 32B Q8 达到了 87%，这明显高于大多数合成基准测试表现较差的传统系统。[提供了内存带宽基准测试和讨论的参考。](https://www.reddit.com/r/LocalLLaMA/comments/1ak2f1v/ram_memory_bandwidth_measurement_numbers_for_both/)
    - 人们对 Strix Halo 系统的 RPC 延迟测试感兴趣，比较了将这些新设备作为单个 RPC 服务器使用与扩展多个廉价系统的潜在价值主张。该询问寻求有关 RPC 性能测试的技术细节，特别是针对 LLM 推理部署，以及此类基准测试是使用一个还是多个单元作为主机/客户端进行的。
- [**Qwen3-30B-A6B-16-Extreme 非常出色**](https://www.reddit.com/r/LocalLLaMA/comments/1kmlu2y/qwen330ba6b16extreme_is_fantastic/) ([Score: 120, Comments: 44](https://www.reddit.com/r/LocalLLaMA/comments/1kmlu2y/qwen330ba6b16extreme_is_fantastic/)): **Qwen3-30B-A6B-16-Extreme 模型（[Hugging Face 链接](https://huggingface.co/DavidAU/Qwen3-30B-A6B-16-Extreme)）是一个 MoE (Mixture of Experts) LLM 变体，与原始的 Qwen 30B-A3B 规范相比，它将激活的专家数量从 8 个增加到 16 个（总共 128 个）。该模型实际上并未经过微调，而是通过配置更改了专家数量，正如模型卡中所澄清的那样，这可能会影响推理深度并可能影响准确性。此外还有 GGUF 量化支持（[链接](https://huggingface.co/mradermacher/Qwen3-30B-A6B-16-Extreme-GGUF)），并提供 128k 上下文长度变体。** 评论中的技术辩论集中在不进行重新训练的情况下增加专家数量的影响：一些用户质疑仅仅激活更多专家是否能带来性能提升，还是需要重新训练，并呼吁通过基准测试来量化改进。一位评论者指出，与模型卡相反，增加专家数量并不构成真正的“微调 (finetune)”，而只是配置更改。
    - 讨论集中在 Qwen3-30B-A6B-16-Extreme 的模型架构上，特别是如何在不重新训练的情况下将激活的 MoE (Mixture-of-Experts) 专家从 128 个中的 8 个增加到 16 个。技术用户确认可以通过配置（而非权重）更改专家数量，例如使用 `--override-kv qwen3moe.expert_used_count=int:24` (llama.cpp) 或通过 LM Studio 设置。

- safetensors 文件的 SHA256 校验和保持不变，这表明仅修改了 config 文件以在每个 token 中使用更多专家，而模型权重本身并未修改。这表明增加专家数量仅仅是一个运行时配置，而不是真正的 finetune，尽管一些 model cards 错误地将其描述为 finetune。
- 关于增加专家数量对性能的影响仍存疑问。评论者要求对不同专家数量进行 benchmark 对比，并争论仅仅激活更多专家是否能产生更好的结果，或者是否需要进一步训练以最大化收益。
- [**拥抱简陋 (2x5090)**](https://www.reddit.com/gallery/1km81fb) ([Score: 101, Comments: 48](https://www.reddit.com/r/LocalLLaMA/comments/1km81fb/embrace_the_jank_2x5090/)): **OP 通过在现有的 4x3090 配置中添加第二块 NVIDIA RTX 5090 GPU 升级了一台挖矿机，并指出 5090 的供货情况有所改善且价格有所下降。他们报告了由于 Gigabyte 5090 的物理长度带来的兼容性挑战，但观察到 ROPs 非常稳健（表明是后期批次的显卡），并且在设置了功耗限制（5090 为 400W，3090 为 250W）的情况下，电缆/电源发热仍处于安全范围内。使用场景包括在其中一块 5090 上同时进行 LoRA 训练，在另一块上通过 ComfyUI 进行图像生成，并计划在 3090 上通过 vllm 或 sglang 进行推理。** 评论者强调了在某些地区，像 5090 这样高 VRAM 显卡的高昂成本，并建议进行进一步的技术分析，如系统噪音测量。
    - 一位用户讨论了同时处理工作负载的能力，即在其中一块 RTX 5090 上运行 LoRA 训练任务，在另一块 5090 上使用 ComfyUI 进行图像生成，同时 TabbyAPI 在 4x3090 上运行。该工作负载被描述为轻量级的，用户打算稍后使用 vllm 或 sglang 推理测试更高需求的场景，这表明人们有兴趣评估在更密集的 AI 服务任务下真实的 multi-GPU 性能。
    - 散热管理问题受到关注，特别是运行像 5090 这样的高端 GPU 时接口熔断的风险。用户询问了用于监控的热成像仪的类型和质量，这表明在极端或企业级 GPU 配置中，人们对硬件安全性和可靠性的关注。

### 2. MAESTRO 本地优先 AI 研究应用发布与基准测试

- [**宣布 MAESTRO：一个本地优先的 AI 研究应用！（附带一些基准测试）**](https://www.reddit.com/gallery/1kmaztr) ([Score: 149, Comments: 34](https://www.reddit.com/r/LocalLLaMA/comments/1kmaztr/announcing_maestro_a_localfirst_ai_research_app/)): **MAESTRO 是一款模块化、本地优先的 AI 研究应用，支持文档摄取、混合搜索 RAG 流水线以及多智能体系统（规划、研究、反思、写作），可配置本地或基于云端/API 的 LLM。基准测试可在仓库的** `VERIFIER_AND_MODEL_FINDINGS.md` **中查看，使用一组 LLM '验证器'（verifiers）来评估并匹配本地和远程模型在智能体角色中的表现，报告每项任务的性能（例如：笔记记录、综合分析）并提供部署建议。该系统支持 Streamlit UI 和 CLI 交互，跟踪资源/成本使用情况，并正朝着增强型 UI 和智能体框架方向积极演进。参见 [代码和详情](https://github.com/murtaza-nasir/maestro)。** 值得注意的评论者问题包括：对某些基准测试结果的惊讶（例如：'Qwen3 8B 表现优于 32B？'）、对更广泛网络搜索 API 支持的请求（例如：SearxNG, DuckDuckGo, Google），以及对示例输出过于常规的批评，并建议在更具新颖性的研究领域测试智能体，以进行真正的价值评估。
    - 发帖者被质疑 Qwen3 8B 是否真的优于 32B 模型，反映了对发布的基准测试的怀疑和兴趣。这突显了社区对令人惊讶的性能结果的关注，因为传统观念认为，除非模型架构在效率或指令微调方面有显著改进，否则规模大得多的模型（32B）理应优于较小的模型（8B）。
    - 提供了一个引用 PyTorch 错误的严重 Bug 报告，涉及自定义类实例化。用户收到：`RuntimeError: Tried to instantiate class '__path__._path', but it does not exist! Ensure that it is registered via torch::class_`。这表明可能存在打包或扩展问题，即所需的 TorchScript/PyTorch 自定义类未注册或缺失。
- [**Wan-AI/Wan2.1-VACE-14B · Hugging Face (Apache-2.0)**](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B) ([Score: 118, Comments: 6](https://www.reddit.com/r/LocalLLaMA/comments/1kmg1ht/wanaiwan21vace14b_hugging_face_apache20/)): **Wan2.1（[Hugging Face 仓库](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B)）是一个开源的、最先进的视频基础模型套件，涵盖文本生成视频（text-to-video）、图像生成视频（image-to-video）、视频编辑、文本生成图像以及视频生成音频，模型参数范围从 1.3B 到 14B。它展示了 SOTA 性能（优于开源和商业同行），支持消费级 GPU（使用 1.3B 模型在 RTX 4090 上 4 分钟内渲染 5 秒 480p 视频），并具有强大的双语（中/英）文本生成能力、高质量且保持时序一致性的视频 VAE (Wan-VAE)，以及与 Diffusers 和 ComfyUI 的紧密集成，支持通过 FSDP 和 Ulysses/Ring 策略进行高级分布式/多 GPU 推理。所有代码和权重均在 Apache 2.0 协议下发布，针对 LoRA/可微调工作流进行了优化，并提供了量化和速度优化。** 评论者请求推出 MoE (Mixture of Experts) 14B 变体，以显著提高实际部署的推理速度（在保留约 90% 性能的情况下可能实现 10 倍加速），并要求澄清 Wan2.1 之前和当前变体（ITV/TTV 与 VTV 组件）之间的命名和功能区别。
    - 一位用户讨论了 MoE (Mixture of Experts) 14B 版本的潜在影响，指出即使只有原始模型 90% 的性能，10 倍的加速也将极大地改善实际推理时间，特别是对于消费级用例（例如，在经过优化的 RTX 4090 上，将 20 分钟的渲染缩短至 2 分钟，或将 10 分钟缩短至 1 分钟）。
    - 引用了模型卡片中的技术亮点：T2V-1.3B 模型仅需 8.19GB VRAM，兼容消费级 GPU，在不进行量化的情况下，在 RTX 4090 上约 4 分钟即可生成 5 秒 480p 视频。据报道，Wan2.1 在各项基准测试中优于其他开源和商业模型，在多模态任务中表现出色，是首个能稳健生成中英双语文本的视频模型，并配备了用于处理 1080p 视频的高效 Wan-VAE。
    - Wan 系列的命名惯例和版本存在混淆，用户对从 ITV/TTV 到潜在 VTV 的过渡表示疑问，表明需要更清晰的模型进展和架构变更文档或 changelog。

### 3. BitNet R1 三进制模型微调与社区工具

- [**BitNet 对 R1 Distills 的微调**](https://x.com/0xCodyS/status/1922077684948996229) ([Score: 274, Comments: 65](https://www.reddit.com/r/LocalLLaMA/comments/1klxlbx/bitnet_finetunes_of_r1_distills/)): **一种新方法能够通过在每个线性层之前插入输入侧 RMSNorm，将现有的 FP16 Llama 和 Qwen 检查点直接微调为三进制 BitNet（权重限制为 {-1, 0, 1}）。模型 bitnet-r1-llama-8B（在大约 3 亿个 token 时收敛）和 bitnet-r1-qwen-32B（约 2 亿个 token）是在 8×H100 GPU 上使用 BF16 AdamW 训练的，所有线性权重均已量化（包括此版本的 lm_head）。通过一个 PR ([repo](https://github.com/Codys12/transformers.git)) 提供了 PyTorch/Transformers 支持，仅需更改 quant_config 即可使用和进一步微调；检查点托管在 Hugging Face 上（[bitnet-r1-llama-8b](https://huggingface.co/codys12/bitnet-r1-llama-8b), [bitnet-r1-qwen-32b](https://huggingface.co/codys12/bitnet-r1-qwen-32b)）。这种方法降低了内存需求和训练成本，实现了具有竞争力的 loss 趋势，路线图包括收敛、保持输出头为全精度以及 RMS 补丁的上游合并。** 专家评论指出，该方法可以用极少的额外训练实现 BitNet 权重——比从头开始重新训练更便宜——并对性能是否超过 4-bit 量化表示关注，同时要求进行更多的基准测试和更广泛的硬件评估。
    - 详细介绍的核心创新是在现有 FP16 Llama 或 Qwen 模型的每个线性操作之前添加一个输入侧 RMSNorm 层，从而允许直接微调为高度压缩的 1-bit BitNet 格式。该方法能够以原始全量训练成本的一小部分实现快速适配（在大约 2 亿至 3 亿个 token 内收敛），且对运行时的影响极小，因为额外的 RMSNorm 可以在训练后融合到量化过程中。
    - 在这些实验中，包括关键的 `lm_head` 在内的所有线性权重都经过了量化，以进行稳定性压力测试——与保持 `lm_head` 为全精度的方法相比，这一选择预计会产生次优的困惑度（perplexity）。作者指出，未来的迭代将保留全精度 `lm_head`，并旨在实现更好的收敛性以及与原始模型权重的兼容性，最终支持作为标准检查点的直接替换。
    - 训练使用 BF16 AdamW 和 DeepSpeed ZeRO-3 在 8x H100 GPU 上进行。虽然 BitNet 权重打包减少了内存占用，并能在内存受限的场景中提供更快的推理，但某些硬件可能会因反量化（de-quantization）而产生开销。这些检查点目前是实验性的，在进一步训练继续之前，预计会存在轻微的困惑度差距；代码修改可在 Hugging Face Transformers 的自定义分支中获得，供早期采用者使用。
- [**我更新了 SmolVLM llama.cpp 网络摄像头演示，使其通过 WebGPU 在浏览器本地运行。**](https://v.redd.it/or5b3ks8nr0f1) ([Score: 205, Comments: 14](https://www.reddit.com/r/LocalLLaMA/comments/1kmi6vl/i_updated_the_smolvlm_llamacpp_webcam_demo_to_run/)): **该帖子宣布了对 SmolVLM/llama.cpp 网络摄像头演示的更新：它现在完全使用 WebGPU 和 Transformers.js 在浏览器中运行，无需本地安装或服务器后端。该演示利用客户端 WebGPU 加速进行实时推理，并部署在 Hugging Face Spaces 上，实现非常简洁（单个 index.html 文件），可在文件部分找到（演示链接：https://huggingface.co/spaces/webml-community/smolvlm-realtime-webgpu）。** 评论中的技术讨论较少；大多数反馈是轶闻式的，集中在演示输出上，而非实现细节或性能特征。
    - 一位用户询问了 500M SmolVLM 模型的大小，引发了关于其在本地/浏览器内执行的存储需求的讨论。虽然线程中未给出确切数字，但 500M 参数模型在 FP16 精度下通常在 1GB 到 2GB 之间，这对于考虑在资源受限设备上进行浏览器内部署的人来说至关重要。

- [**美国发布全球范围内使用华为 AI 芯片的限制**](https://asia.nikkei.com/Spotlight/Huawei-crackdown/US-issues-worldwide-restriction-on-using-Huawei-AI-chips) ([得分: 177, 评论: 166](https://www.reddit.com/r/LocalLLaMA/comments/1km7azf/us_issues_worldwide_restriction_on_using_huawei/)): **美国已实施全球出口限制，禁止使用华为 AI 芯片，将其域外管控范围扩展至国境之外，以遏制对先进半导体技术的获取。此举旨在防止华为为 AI 和 HPC 应用提供可与 Nvidia 等美国领先供应商竞争的 AI 芯片，理由是国家安全和竞争优势考量。这一行动进一步延伸了此前针对芯片制造设备和半导体设计工具的限制，影响了全球供应链和非美国实体（参见 [路透社报道](https://www.reuters.com/technology/us-issues-worldwide-restriction-using-huawei-ai-chips-2024-05-07/)）。** 热门评论指出，华为对 Nvidia 构成了隐含的技术威胁，对美国司法管辖区外的可执行性表示怀疑，并将这一限制解读为对华为 AI 芯片的一种技术背书。
    - 评论提到了华为 AI 芯片潜在的技术竞争力，推测美国的限制表明这些芯片在某些全球市场的*价格和性能*方面可能超越 Nvidia 的产品。这一推论与最近的行业分析一致，该分析表明华为新的 *Ascend 系列* 芯片正成为 AI 工作负载的可行替代方案，特别是在 Nvidia 成本高昂或供应受限的情况下。

## 其他 AI Subreddit 摘要

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. AlphaEvolve 与 DeepMind 在编程和科学 AI 领域的突破

- [**DeepMind 推出 AlphaEvolve：一个由 Gemini 驱动的用于算法发现的编程 Agent**](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) ([得分: 1103, 评论: 274](https://www.reddit.com/r/singularity/comments/1kmhti8/deepmind_introduces_alphaevolve_a_geminipowered/)): **DeepMind 宣布推出 AlphaEvolve，这是一个利用 Gemini LLM 集成（Flash 用于探索，Pro 用于深度）进行新颖算法发现和优化的自动化编程 Agent ([DeepMind 博客](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/))。AlphaEvolve 迭代地生成并测试代码解决方案，带来了性能提升——例如，Gemini 的矩阵乘法内核速度提升了** `23%` **（使整体训练时间减少了** `1%` **），数据中心调度的计算回收率提升了** `0.7%` **，以及 TPU 的硬件改进。在数学任务上，它在** `50+` **个开放问题的** `75%` **中重新发现了最先进的解决方案，并在** `20%` **的问题中超越了之前的最佳方案，包括对接吻数问题 (kissing number problem) 的改进。该 Agent 通过自动化的无监督搜索，将内核优化时间从数周缩短至数天。** 技术辩论集中在对基于 LLM 的科学发现的影响上，一些人将 AlphaEvolve 视为对“LLM 无法自主发现新算法”这一说法（与 Yann LeCun 等专家的演讲形成对比）的反例。评论者还预计，此类进展预示着算法发现领域无监督自我改进的近期突破。
    - AlphaEvolve 在包括分析、几何、组合数学和数论（如接吻数问题）在内的 50 多个数学领域开放问题上进行了测试，能够在 `75%` 的案例中重新发现已知最佳方案，并在 `20%` 的案例中超越之前的解决方案，从而产生了可验证的新发现。([来源](https://x.com/GoogleDeepMind/status/1922669334142271645))
    - 该系统使用 Gemini 驱动的方法来优化矩阵乘法内核，使这一关键操作加速了 `23%` ，并导致 Gemini 模型训练时间显著减少了 `1%` 。这种效率提升转化为计算费用的降低，且 AlphaEvolve 的自动化将内核优化周期从数周的手动调优缩短至数天的自动化运行，从而加速了研究周期。
    - AlphaEvolve 的优化策略直接应用于核心基础设施，包括 Google 的数据中心和 AI 芯片设计，以及旨在改进的模型架构本身（例如驱动 AlphaEvolve 的架构），在 AI 开发栈中形成了一个自我优化的反馈闭环。

- [**认识 AlphaEvolve：这款 Google AI 能自主编写代码，并刚节省了数百万美元的计算成本**](https://venturebeat.com/ai/meet-alphaevolve-the-google-ai-that-writes-its-own-code-and-just-saved-millions-in-computing-costs/) ([Score: 455, Comments: 52](https://www.reddit.com/r/singularity/comments/1kmia4y/meet_alphaevolve_the_google_ai_that_writes_its/)): **Google AI 推出了 AlphaEvolve，这是一个表面上能够自主生成新型计算机算法的 AI 系统，并声称在计算开支方面节省了巨额成本（数百万美元）。该公告进一步断言 AlphaEvolve 能够在计算和数学领域做出新发现，呼应了此前 AlphaFold 和 AlphaGo 等突破性成果的雄心。技术读者指出，需要具体证据来支持这些说法，例如可复现的基准测试、可访问的数据集或关于算法创新的细节。** 评论者对这些说法的宏大程度表示怀疑，特别是要求提供 AlphaEvolve 发明全新算法能力的证据。其他人则提到了历史先例——AlphaFold 和 AlphaGo 也是通过广泛的搜索和自我博弈（self-play）取得了非常规的结果，但实质性的经验结果验证了它们的重要性。
    - 据报道，AlphaEvolve 结合了 Gemini Flash 和 Gemini Pro 模型作为其核心框架，允许随着新的 SOTA 模型出现而进行模块化升级。这种设计实现了适应性，并暗示随着底层模型的进步，效率和能力将持续提升。
    - 强调了 AlphaEvolve 对 Google 复杂基础设施（芯片设计、网络、DC 部署、云计算）的影响：AI 驱动的优化可能会在公司广泛的技术栈中带来显著的效率提升，从而可能超越缺乏类似垂直整合能力的竞争对手。
    - 评论者对 Google 说法的合法性进行了辩论，指出 Google/DeepMind 历史上分享的性能指标与某些竞争对手相比水分较少。人们要求提供具体的基准测试或证据，特别是关于算法发明和数学发现的说法。
- [**DeepMind 发布“惊人的”通用科学 AI**](https://www.nature.com/articles/d41586-025-01523-z) ([Score: 246, Comments: 29](https://www.reddit.com/r/singularity/comments/1kmik3f/deepmind_unveils_spectacular_generalpurpose/)): **DeepMind 最新公布的 AlphaEvolve 系统将大语言模型（LLM）与自动算法评估器集成，以在科学领域自主进化出新型、高性能的算法。在基准测试中，AlphaEvolve 发现了超越长期存在的 Strassen 算法的矩阵乘法例程，以及针对张量处理单元（TPU）和云资源分配的改进设计。该架构独特地结合了 LLM 驱动的提案生成与进化算法选择，实现了通用领域的决策能力（详情见 Nature 公告）。** 评论者强调了 AlphaEvolve 超越经典算法（例如用于矩阵乘法的 Strassen 算法）的能力，并推测此类进展展示了 DeepMind 在通向通用人工智能（AGI）方面的领导地位，并将这一进展与 DeepMind 最近招聘的“AGI 后”专家联系起来。
    - DeepMind 的新 AI AlphaEvolve 开发的矩阵乘法算法速度可以超过 Strassen 算法，后者自 1969 年以来一直是已知的最快方法。这展示了 AI 在计算数学（计算机科学的一个关键基准领域）中发现新型优化策略的能力。
    - 一个关键的观察是，AlphaEvolve 的能力本质上与具有可扩展且廉价验证手段的领域相关，例如程序分析，其中计算的正确性和速度很容易衡量。对于天文学、粒子物理学、医学或商业等验证成本高昂或受限的领域，这些 AI 驱动的发现影响仍然微乎其微；限制因素从想法生成转向了实验验证。
    - 虽然 AI 在矩阵乘法等计算任务上的改进暗示了在技术上可处理的领域中存在复合效应（飞轮效应），但更深层的含义是，如果将这种 AI 系统应用于互连的基础技术，理论上可以驱动快速且递归的创新。这与关于由通用科学 AI 驱动的技术奇点（technological singularity）可行性的讨论相关。

### 2. Anthropic Claude Sonnet/Opus 模型发布预期与 OpenAI 模型推出

- [**看来我们可以期待 Anthropic 在未来几周发布新产品**](https://i.redd.it/lw3pkseybr0f1.jpeg) ([Score: 218, Comments: 61](https://www.reddit.com/r/singularity/comments/1kmgig2/looks_like_we_can_expect_an_anthropic_release_in/)): **图片描绘了一个正式的演示，很可能由 Anthropic 代表进行，重点介绍了即将发布的 Claude Sonnet 和 Claude Opus 模型。这些模型因其独特的推理能力而备受关注，暗示了 Anthropic 的 AI 架构取得了重大进展。帖子和评论都表明可靠的消息来源 (**1. AlphaEvolve 与 DeepMind 在编程和科学 AI 领域的突破
    - [**DeepMind 推出 AlphaEvolve：一个由 Gemini 驱动的算法发现编程 Agent**](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) ([Score: 1103, Comments: 274](https://www.reddit.com/r/singularity/comments/1kmhti8/deepmind_introduces_alphaevolve_a_geminipowered/)): **DeepMind 宣布推出 AlphaEvolve，这是一个利用 Gemini LLM 集成（Flash 用于探索，Pro 用于深度）进行新算法发现和优化的自动化编程 Agent ([DeepMind 博客](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/))。AlphaEvolve 迭代地生成并测试代码解决方案，从而带来性能提升——例如，Gemini 的矩阵乘法内核速度提升了** `23%` **（使整体训练时间减少了** `1%`**），数据中心调度的计算回收率提高了** `0.7%`**，并实现了 TPU 的硬件改进。在数学任务中，它在** `50+` **个开放问题的** `75%` **中重新发现了最先进的解决方案，并在** `20%` **的问题中超越了之前的最佳方案，包括对吻数（kissing number）问题的改进。该 Agent 通过自动化的无监督搜索，将内核优化周期从数周缩短至数天。** 技术辩论集中在基于 LLM 的科学发现的影响上，一些人将 AlphaEvolve 视为对“LLM 无法自主发现新算法”这一主张的反例（与 Yann LeCun 等专家的观点形成对比）。评论者还预计，此类进展预示着算法发现领域无监督自我改进的近期突破。
        - AlphaEvolve 在包括分析、几何、组合数学和数论（例如对吻数问题）在内的数学领域的 50 多个开放问题上进行了测试，能够在 `75%` 的案例中重新发现已知最佳解决方案，并在 `20%` 的案例中超越之前的解决方案，从而产生了可验证的新发现。([来源](https://x.com/GoogleDeepMind/status/1922669334142271645))
        - 该系统使用 Gemini 驱动的方法来优化矩阵乘法内核，使这一关键操作加速了 `23%`，并导致 Gemini 模型的训练时间可衡量地减少了 `1%`。这种效率提升转化为计算开销的降低，而 AlphaEvolve 的自动化将内核优化周期从数周的人工调优缩短到数天的自动化运行，从而加快了研究周期。
        - AlphaEvolve 的优化策略直接应用于核心基础设施，包括 Google 的数据中心和 AI 芯片设计，以及它旨在改进的模型架构本身（例如驱动 AlphaEvolve 的架构），从而在 AI 开发栈中形成了一个自我优化的反馈闭环。
    - [**认识 AlphaEvolve，这款 Google AI 能编写自己的代码，并刚刚节省了数百万美元的计算成本**](https://venturebeat.com/ai/meet-alphaevolve-the-google-ai-that-writes-its-own-code-and-just-saved-millions-in-computing-costs/) ([Score: 455, Comments: 52](https://www.reddit.com/r/singularity/comments/1kmia4y/meet_alphaevolve_the_google_ai_that_writes_its/)): **Google AI 推出了 AlphaEvolve，这是一个表面上能够自主生成新型计算机算法的 AI 系统，并声称在计算开销方面节省了大量成本（数百万美元）。该公告进一步断言 AlphaEvolve 能够在计算和数学领域做出新发现，呼应了此前 AlphaFold 和 AlphaGo 等突破的雄心。技术读者指出需要支持这些主张的具体证据，例如可重复的基准测试、可访问的数据集或关于算法创新的细节。** 评论者对这些主张的规模表示怀疑，特别是要求提供 AlphaEvolve 发明全新算法能力的证据。其他人则指出了历史先例——AlphaFold 和 AlphaGo 也通过广泛的搜索和自我博弈取得了非传统的成果，但实质性的实证结果验证了它们的重要性。

- 据报道，AlphaEvolve 结合了 Gemini Flash 和 Gemini Pro 模型作为其核心框架，允许随着新的 SOTA 模型出现而进行模块化升级。这种设计实现了适应性，并暗示随着基础模型的进步，效率和能力将持续提升。
- 强调了 AlphaEvolve 对复杂 Google 基础设施（芯片设计、网络、DC 部署、云计算）的影响：这里的 AI 驱动优化可能会在公司广泛的技术栈中带来显著的效率提升，有可能超越缺乏类似垂直整合的竞争对手。
- 评论者对 Google 声明的真实性展开辩论，指出 Google/DeepMind 历史上分享的性能指标比某些竞争对手更少水分。人们对具体的基准测试或证据有需求，特别是关于算法发明和数学发现的声明。
- [**DeepMind 发布“惊人”的通用科学 AI**](https://www.nature.com/articles/d41586-025-01523-z) ([Score: 246, Comments: 29](https://www.reddit.com/r/singularity/comments/1kmik3f/deepmind_unveils_spectacular_generalpurpose/)): **DeepMind 新发布的 AlphaEvolve 系统将大语言模型 (LLMs) 与自动算法评估器相结合，以在科学领域自主演化出新颖的高性能算法。在基准测试中，AlphaEvolve 发现了超越长期存在的 Strassen 算法的矩阵乘法程序，以及改进的张量处理单元 (TPUs) 设计和云资源分配方案。该架构独特地结合了 LLM 驱动的提案生成与进化算法选择，实现了通用领域的问题解决能力（详情见 Nature 公告）。** 评论者强调了 AlphaEvolve 超越经典算法（如用于矩阵乘法的 Strassen 算法）的能力，并推测此类进展展示了 DeepMind 在迈向通用人工智能 (AGI) 方面的领导地位，并将这一进展与 DeepMind 最近招聘的“AGI 之后”专家联系起来。
    - DeepMind 的新 AI AlphaEvolve 开发的矩阵乘法算法可以超越 Strassen 算法的速度，后者自 1969 年以来一直是已知最快的方法。这展示了 AI 在计算数学中发现新颖优化策略的能力，计算数学是计算机科学中的一个关键基准测试领域。
    - 一个关键的观察是，AlphaEvolve 的能力本质上与具有可扩展且低成本验证的领域挂钩，例如程序分析，其中计算的正确性和速度很容易衡量。对于天文学、粒子物理学、医学或商业等验证成本高昂或受限的领域，这些 AI 驱动发现的影响仍然微乎其微；限制因素从想法生成转向了实验验证。
    - 虽然 AI 驱动的计算任务（如矩阵乘法）改进暗示了在技术上可处理的领域中存在复合效应（飞轮效应），但更广泛的意义在于，它证明了如果将 AI 系统应用于相互关联的基础技术，理论上可以驱动快速且递归的创新。这与关于由通用科学 AI 驱动的技术奇点可行性的讨论相关。

### 2. Anthropic Claude Sonnet/Opus 模型发布预期与 OpenAI 模型推出

- [**看来我们可以期待 Anthropic 在未来几周内的发布**](https://i.redd.it/lw3pkseybr0f1.jpeg) ([Score: 218, Comments: 61](https://www.reddit.com/r/singularity/comments/1kmgig2/looks_like_we_can_expect_an_anthropic_release_in/)): **图片描绘了一个正式的演示文稿，可能由 Anthropic 代表主讲，强调了即将发布的 Claude Sonnet 和 Claude Opus 模型。这些模型以其独特的推理能力而著称，暗示了 Anthropic 在 AI 架构方面的重大进步。帖子和评论都指出，可靠的消息来源（特别是 The Information）已经报道了即将发布的版本，使 Anthropic 在未来几周的重大模型发布中与 OpenAI 和 Google 展开直接竞争。** 一位评论者强调了 The Information 在模型发布报道方面的历史准确性，增加了该消息的可信度。另一位评论者开玩笑说以前 Anthropic 模型的服务器容量问题，暗示了社区希望在新版本中解决的可扩展性担忧。
    - 讨论引用了 The Information 在准确预测重大 AI 模型发布时间表方面的公信力，强调了他们之前在预先发布有关 Anthropic 和 OpenAI 等公司即将推出的模型行业新闻方面的成功。

- 人们预期 OpenAI 近期会尝试抢占 Google 的风头，参考往年重大发布时间紧凑的情况，这突显了顶尖公司（OpenAI, Google, Anthropic）在模型发布和关注度方面的持续竞争。
- 一位用户对具有更大服务器容量的改进版本表示了特别关注，暗指过去 Sonnet 3.7 等版本发布时出现的服务器端瓶颈，这些瓶颈影响了模型的可用性和用户体验。
- [**Damn ok now this will be interesting**](https://i.redd.it/enm10z3clr0f1.jpeg) ([Score: 193, Comments: 41](https://www.reddit.com/r/ClaudeAI/comments/1kmhtmq/damn_ok_now_this_will_be_interesting/)): **该图片是一条推文，重点介绍了 Anthropic 的新模型——Claude Sonnet 和 Claude Opus——它们可以在 reasoning、tool/database usage 和 self-correction 模式之间动态切换。据报道，它们拥有增强的 code generation 能力，允许它们测试并修复自己的输出。该公告预示着未来几周内将发布新版本。** 评论中的一个主要技术讨论是关于 prompt 长度可能损害模型性能的担忧，因为更复杂的模式切换可能需要大得多的 system prompts。一位用户分享了关于动态代码编辑和快速 artifact 预览的轶事证据（可能使用了早期访问权限），称其功能强大得令人惊讶。
    - 一位评论者对 system prompt 长度和 token 使用情况表示担忧，指出引入新的 Anthropic 模型可能会导致显著增加的 system prompts（“多出 8000 个 tokens”），这可能会影响模型性能或 context retention。有人表示希望这些模型即使在 prompt 尺寸增加的情况下也能保持其能力。
    - 另一位用户详细介绍了他们使用新 code-assist 功能的经验，观察到在最终结果提交之前的迭代 UI 更改过程中出现了 artifact 预览和图形故障。这种细粒度的更新和提交周期（包括 artifact 预览）被描述为*感觉非常强大*，暗示了一种技术先进或新颖的实现，改善了开发过程中的用户反馈。
    - 还有关于 token 消耗的技术评论，一位用户强调新功能可能会显著增加 token 使用量，从而可能影响运营成本和效率（“token 成本将变得疯狂”，“cline 已经开始吞噬 tokens 了”）。
- [**4.1 model now appearing in web browser under "More models"**](https://i.redd.it/jjyhfk46as0f1.png) ([Score: 109, Comments: 64](https://www.reddit.com/r/OpenAI/comments/1kmlcne/41_model_now_appearing_in_web_browser_under_more/)): **该图片记录了 OpenAI ChatGPT Web UI 中新模型变体的推出，特别是在“更多模型”菜单下。它确认了“GPT-4.1”及其 mini 变体（“GPT-4.1-mini”）的存在，以及“GPT-4o”和“o4-mini”等其他模型；值得注意的是，“GPT-4.1”被明确标记为最适合“快速编码和分析”。该图片提供了 OpenAI 产品中后端更新和不断演进的模型阵容的证据，展示了活跃的部署和新的模型区分。** 评论者指出，“4.1-mini”似乎正在取代“4o-mini”，一位用户强调 “4.1-mini” 已经可以访问，据报道在编码任务中表现良好。
    - 几位用户注意到 “4.1-mini” 似乎正在 Web 界面中取代 “4o-mini” 模型，这表明可用轻量级模型发生了更新或转变。这会影响那些为日常或嵌入式用例寻求最快、最具成本效益选项的用户。
    - 具体反馈强调 4.1 模型在编码任务中表现出色——一位用户报告了与 Roo 平台的成功集成，表明了开发者对新模型能力的即时兴趣和迅速实验。
    - 提到了设备和 App 的差异：Android 用户可能需要更新他们的 App 才能访问 4.1 和 4.1-mini，而一些 Web 用户目前只能看到 4.1 mini，这表明是分阶段推出或取决于平台的可用性。

### 3. ChatGPT 作为新的互联网接口及其社会影响

- [**去年 ChatGPT 是访问量第 15 大的网站。现在它排名第 5，而其他前 10 名的网站流量都在下降（Wikipedia 下降了 6%）。人们不再在网上冲浪了——他们直接前往 ChatGPT。它不仅仅是一个工具；它已成为新的互联网接口，悄然取代了旧的 Web。**](https://i.redd.it/xbfjvdyqbr0f1.jpeg) ([Score: 278, Comments: 117](https://www.reddit.com/r/ChatGPT/comments/1kmghgd/last_year_chatgpt_was_the_15th_most_visited_site/)): **提供的图片显示了一张访问量最大的网站排名表，强调了 [ChatGPT.com](http://chatgpt.com/) 在全球流量中攀升至第 5 位，并显示出 13.04% 的环比增长，而 Wikipedia 等传统网站则出现了下滑（-6%）。数据表明了一种显著的行为转变，即用户越来越多地绕过传统的搜索引擎或内容聚合器，使用对话式 AI 作为获取在线信息的主要接口。这突显了 ChatGPT 的迅速崛起，它不仅是一个工具，而且是一个取代传统 Web 导航的门户。** 评论中一场技术相关的辩论指出，由于糟糕的 Web 体验（如侵入性广告、SEO 操纵）和内容发现中的摩擦，用户越来越倾向于 AI 助手而非传统搜索。人们也对 ChatGPT 未来的货币化（如广告）表示担忧。
    - 几位评论者指出传统搜索引擎（如 Google）的技术性衰退，将转向 ChatGPT 的原因归结为激进的 SEO 策略和广告覆盖导致的 Web 混乱，这降低了实际的信息检索体验。这一趋势导致用户更倾向于使用 ChatGPT 获取直接答案，因为它绕过了困扰标准食谱或信息网站的弹窗和冗长的无关叙述。
    - 讨论强调了未来货币化变化影响 ChatGPT 的风险，例如随着流量增加引入更多广告或降低可用性。这可能会重演其他 Web 平台在将广告收入流置于用户体验之上后所经历的历史性退化。
- [**互联网正在发生一些奇怪的事情**](https://i.redd.it/frg2ndw14s0f1.jpeg) ([Score: 1945, Comments: 417](https://www.reddit.com/r/ChatGPT/comments/1kmkgt9/something_strange_is_happening_to_the_internet/)): **该帖子讨论了全球 Web 流量的重大转变，如按流量排名的顶级网站表格所示（[见图片](https://i.redd.it/frg2ndw14s0f1.jpeg)）。[Google.com](http://google.com/) 虽处于领先地位但正在下滑（环比下降 **`3.18%`**），而 ChatGPT 已飙升至第 5 位，环比增长 **`+13.04%`**，超过了 Reddit、Amazon 和 Whatsapp，目前是前 10 名中唯一正增长的域名。这表明用户越来越多地依赖 ChatGPT 作为主要“接口”，可能会绕过传统的搜索引擎、博客或论坛，这不仅标志着增长，还标志着*人们在线获取信息方式可能发生的范式转变*。** 评论者对这种新颖性持怀疑态度，将这一趋势比作 Facebook 和 Google 之前的激增，并开玩笑说该帖子可能是 AI 撰写的，暗示了 AI 在重塑互联网消费方面的日益普及和相关辩论，而不是将其视为真正史无前例的事情。
    - 多位评论者对 ChatGPT 等大语言模型 (LLM) 生成内容的扩散表示担忧，观察到帖子和评论中的独特模式（例如，“不是 X，而是 Y”之类的公式化短语）是 AI 生成文本的有力指标，并导致了感知上的在线真实性和质量下降。
    - 一位参与者认为，如果 LLM 和生成式 AI 有助于减少互联网流量和点击驱动的内容经济的盛行，它可能会提高互联网的效用——从愤怒和点击诱饵转向一种有利于功能性、有目的互动的模式，让人联想起早期的互联网时代。
    - 讨论将当前的转变（包括 AI 内容的兴起）与 Facebook、Google 和 Twitter 等主要平台之前的变化进行了类比，暗示了一种模式，即技术或算法的转变从根本上重塑了流量、参与度以及在线社区的形成和持久方式。**The Information 已经报道了即将发布的版本，使 Anthropic 在未来几周内与 OpenAI 和 Google 在重大模型发布方面展开直接竞争。** 一位评论者强调了 The Information 关于模型发布报道的历史准确性，增加了该新闻的可信度。另一位评论者开玩笑说之前 Anthropic 模型的服务器容量问题，暗示了社区希望在新版本中解决的可扩展性担忧。

- 讨论提到了 The Information 在准确预测重大 AI 模型发布时间线方面的可信度，强调了他们此前在预告 Anthropic 和 OpenAI 等公司即将推出的模型行业新闻方面的成功。
- 人们预期 OpenAI 近期会尝试抢在 Google 之前发布，参考了往年重大发布时间紧凑的情况，突显了顶尖公司（OpenAI、Google、Anthropic）在模型发布和关注度方面持续进行的竞争。
- 一位用户对具有更大服务器容量的改进版本表示了特别兴趣，暗指了以往 Sonnet 3.7 发布时出现的服务器端瓶颈，这些瓶颈曾影响了模型的可访问性和用户体验。
- [**该死，现在这会变得很有趣**](https://i.redd.it/enm10z3clr0f1.jpeg) ([Score: 193, Comments: 41](https://www.reddit.com/r/ClaudeAI/comments/1kmhtmq/damn_ok_now_this_will_be_interesting/)): **该图片是一条推文，重点介绍了来自 Anthropic 的新模型——Claude Sonnet 和 Claude Opus——它们可以在推理、工具/数据库使用和自我修正之间动态切换模式。据报道，它们拥有增强的代码生成能力，允许它们测试并修复自己的输出。该公告预示着预计将在几周内发布。** 评论中的一个主要技术讨论是关于 Prompt 长度可能损害模型性能的担忧，因为更复杂的模式切换可能需要大得多的 System Prompt。一位用户分享了关于动态代码编辑和快速 Artifact 预览的轶事证据（可能是通过早期访问获得的），称其功能强大得令人惊讶。
    - 一位评论者对 System Prompt 长度和 Token 使用表示担忧，指出新 Anthropic 模型的引入可能会导致显著更大的 System Prompt（“多出 8000 个 Token”），这可能会影响模型性能或上下文保留。有人表示希望这些模型即使在 Prompt 尺寸增加的情况下也能保持其能力。
    - 另一位用户详细描述了他们使用新代码辅助功能的体验，观察到在最终结果提交之前的迭代 UI 更改过程中出现了 Artifact 预览和图形故障。这种包含 Artifact 预览的细粒度更新和提交周期被描述为“感觉很强大”，暗示了一种技术先进或新颖的实现，改善了开发过程中的用户反馈。
    - 还有关于 Token 消耗的技术评论，一位用户强调新功能可能会显著增加 Token 使用量，从而可能影响运营成本和效率（“Token 成本要飙升了”、“cline 已经在吞噬 Token 了”）。
- [**4.1 模型现在出现在浏览器“更多模型”选项下**](https://i.redd.it/jjyffk46as0f1.png) ([Score: 109, Comments: 64](https://www.reddit.com/r/OpenAI/comments/1kmlcne/41_model_now_appearing_in_web_browser_under_more/)): **该图片记录了 OpenAI ChatGPT Web UI 中新模型变体的推出，特别是在“更多模型”菜单下。它确认了“GPT-4.1”及其 Mini 变体（“GPT-4.1-mini”）的存在，以及“GPT-4o”和“o4-mini”等其他模型；值得注意的是，“GPT-4.1”被明确标记为最适合“快速编码和分析”。该图片提供了正在进行的部署和面向用户的新模型区分的证据，表明了 OpenAI 产品中的后端更新和不断演进的模型阵容。** 评论者注意到 “4.1-mini” 似乎正在取代 “4o-mini”，一位用户强调 “4.1-mini” 已经可以访问，据报道在编码任务中表现良好。
    - 几位用户注意到 “4.1-mini” 似乎正在取代 Web 界面中的 “4o-mini” 模型，这表明可用轻量级模型发生了更新或转变。这会影响那些为日常或嵌入式用例寻求最快、最具成本效益选项的用户。
    - 特定的反馈强调 4.1 模型在编码任务中表现出色——一位用户报告了与 Roo 平台的成功集成，表明了开发者对新模型能力的即时兴趣和迅速实验。
    - 提到了设备和 App 的差异：Android 用户可能需要更新他们的 App 才能访问 4.1 和 4.1-mini，而一些 Web 用户目前只能看到 4.1 mini，这表明了分阶段推出或依赖于平台的可用性。

### 3. ChatGPT 作为新的互联网入口及其社会影响

- [**去年 ChatGPT 是访问量第 15 大的网站。现在它排名第 5，而其他前 10 名网站的流量都在下降（Wikipedia 下降了 6%）。人们不再在网上冲浪了——他们直接前往 ChatGPT。它不仅是一个工具；它已成为新的互联网入口，悄然取代了旧的 Web。**](https://i.redd.it/xbfjvdyqbr0f1.jpeg) ([Score: 278, Comments: 117](https://www.reddit.com/r/ChatGPT/comments/1kmghgd/last_year_chatgpt_was_the_15th_most_visited_site/)): **提供的图片显示了一张网站访问量排名表，强调 [ChatGPT.com](http://chatgpt.com/) 在全球流量中攀升至第 5 位，月环比增长 13.04%，而 Wikipedia 等传统网站则出现下滑（-6%）。数据表明，用户的行为发生了显著转变，越来越多的人绕过传统的搜索引擎或内容聚合器，将对话式 AI 作为获取在线信息的主要入口。这凸显了 ChatGPT 的迅速崛起，它不仅是一个工具，更是取代传统 Web 导航的门户。** 评论中一场具有技术相关性的辩论指出，由于糟糕的 Web 体验（如侵入性广告、SEO 操纵）和内容发现中的摩擦，用户越来越倾向于选择 AI 助手而非传统搜索。此外，人们还对 ChatGPT 未来的商业化（如广告）表示担忧。
    - 几位评论者指出传统搜索引擎（如 Google）的技术性衰退，将向 ChatGPT 的转变归因于激进的 SEO 策略和广告覆盖导致 Web 杂乱无章，降低了实际的信息检索体验。这一趋势导致用户更倾向于使用 ChatGPT 获取直接答案，因为它绕过了困扰标准食谱或信息网站的弹窗和冗长无关的叙述。
    - 讨论强调了未来商业化变革影响 ChatGPT 的风险，例如随着流量增加引入更多广告或降低可用性。这可能会重演其他 Web 平台在将广告收入流置于用户体验之上后所经历的历史性退化。
- [**互联网正在发生一些奇怪的事情**](https://i.redd.it/frg2ndw14s0f1.jpeg) ([Score: 1945, Comments: 417](https://www.reddit.com/r/ChatGPT/comments/1kmkgt9/something_strange_is_happening_to_the_internet/)): **该帖子讨论了全球 Web 流量的重大转变，如流量排名前列网站的表格（[见图](https://i.redd.it/frg2ndw14s0f1.jpeg)）所示。[Google.com](http://google.com/) 虽处于领先地位但正在下滑（月环比下降 **`3.18%`**），而 ChatGPT 已飙升至第 5 位，月环比增长 **`+13.04%`**，超过了 Reddit、Amazon 和 Whatsapp，目前是前 10 名中唯一正增长的域名。这表明用户越来越依赖 ChatGPT 作为主要“接口”，可能会绕过传统的搜索引擎、博客或论坛，这不仅标志着增长，还预示着*人们获取信息的方式可能发生范式转移*。** 评论者对这种新颖性持怀疑态度，将这一趋势比作 Facebook 和 Google 之前的激增，并开玩笑说 AI 可能创作了这篇帖子，暗示了关于 AI 在重塑互联网消费中的作用日益无处不在且引发争论，而非将其视为真正史无前例的事情。
    - 多位评论者对大语言模型（LLM）如 ChatGPT 生成的内容泛滥表示担忧，观察到帖子和评论中独特的模式（例如“不是 X，而是 Y”之类的公式化措辞）是 AI 生成文本的强烈指标，并导致人们感知到在线真实性和质量的下降。
    - 一位参与者认为，如果 LLM 和生成式 AI 有助于减少互联网流量和点击驱动的内容经济的盛行，它可能会提高互联网的效用——从愤怒和点击诱饵转向一种有利于功能性、有目的互动的模式，让人联想到早期的互联网时代。
    - 讨论将当前的转变（包括 AI 内容的兴起）与 Facebook、Google 和 Twitter 等主要平台之前的变化进行了类比，暗示了一种模式，即技术或算法的转变从根本上重塑了流量、参与度以及在线社区形成和持续的方式。

---

# AI Discord 摘要

> 由 gpt-4.1-2025-04-14 生成的摘要之摘要之摘要
> 

**1. 模型基准测试对决与编程性能**

- **Sonar 模型横扫基准测试，GPT-4.1 夺得编程桂冠**：**Sonar Pro Low** 在 BrowseComp 上以 **4.0% 的准确率**（高出近 50%）击败了 **Claude 3.5 Sonnet**，并拥有高达 **3x** 更快、更一致的延迟；同时 **Qwen 3 8B** 和 **GPT-4.1** 在编程和推理任务中获得了广泛好评 ([Perplexity AI](https://docs.perplexity.ai/), [Unsloth AI](https://repo2txt.com/))。
    - 多个 Discord 社区的共识是 **GPT-4.1** 是新的编程之王，用户吐槽 **O3** 的代码能力但赞赏其规划/研究能力，而 **Qwen 3** 模型在同等规模下表现优于 **Gemma 3**，尤其是在微调之后。
- **Gemini 2.5 Pro 和 O4 Mini High 引燃编程竞争**：**Gemini 2.5 Pro** 以其 C++ 编程实力令用户惊叹，被描述为“梦想成真”；而 **O4 Mini High** 因其在大型代码库中快速、高质量的补全能力被称为“编程猛兽” ([LMArena](https://discord.com/channels/1340554757349179412), [OpenAI](https://openai.com/safety/evaluations-hub/))。
    - 尽管有一些关于幻觉的投诉，用户仍认为 **Gemini 2.5 Pro** 和 **GPT-4.1** 是编程领域的顶级选择，而 **Claude 4 + O3 Pro** 被预期为发布后将成为“疯狂组合”。

**2. 分布式与去中心化训练/推理**

- **Psyche Network 助力去中心化 LLM 训练**：**Nous Research** 推出了 [Psyche Network](https://nousresearch.com/nous-psyche/)，这是一个去中心化训练平台，通过自定义点对点网络和 **DisTrO** 优化器协调全球 GPU，旨在混合使用 **FineWeb (14T)**、**FineWeb-2 (4T)** 和 **The Stack v2 (1T)** 数据集预训练一个 **40B 参数**的 LLM。
    - 测试网在 **40 分钟内迅速填满了 50 万个插槽**，用户可以贡献 USDC 获取算力，开放论坛推动模型设计，并提供 [GitHub 仓库](https://github.com/PsycheFoundation/psyche) 供社区贡献。
- **Lost in Conversation：LLM 在多轮对话任务中表现不佳**：[Lost in Conversation 论文](https://arxiv.org/abs/2505.06120) 发现，与单轮对话相比，LLM 在多轮对话中的**性能下降了 39%**，不可靠性源于过早尝试解决方案和较差的错误恢复能力 ([GitHub 仓库](https://github.com/microsoft/lost_in_conversation))。
    - 这暴露了分布式 Agent 系统的一个主要弱点，并突显了改进错误修正和对话记忆机制的需求。

**3. 硬件与性能优化**

- **PCIE 5.0、CUDA 进展和 PyTorch Nightly 优势**：升级到 **PCIE 5.0** 将 50 系列 GPU 上的 Token 生成速度从 **26 tkps 提升至 38 tkps**；同时 PyTorch 开发者建议在 Nightly 构建中优先使用 `needs_exact_strides` 而非 `needs_fixed_stride_order`，以获得更可靠的 Tensor 操作 ([LM Studio](https://www.manning.com/books/build-a-large-language-model-from-scratch), [GPU MODE](https://github.com/simveit/load_and_store))。
    - 使用 PTX 指令（包括 `ldmatrix`）进行矩阵加载/存储在 [博客文章](https://veitner.bearblog.dev/load-and-store-matrices-efficently-with-ptx-instructions/) 和 [仓库](https://github.com/simveit/load_and_store) 中受到关注，而 **AMD MI300** 上的排行榜提交显示了针对 fp8-mm 和 Mixture-of-Experts Kernel 的激进优化。
- **CUTLASS 4.0、CuTe DSL 与 Kernel 技巧**：**CUTLASS 4.0** 和 **CuTe DSL** 已发布（`pip install nvidia-cutlass-dsl`），带有 Jupyter Notebook 示例并支持 Python 3.12，尽管发布版本号似乎“坏了”，且 MLIR 编译器尚未开源 ([CUTLASS Notebooks](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks))。
    - 自定义 CuTe Kernel 在处理大型问题时性能优于 PyTorch **60x**，同时分享了新的 Kernel 调试技巧（Nsight Compute, nsys, ncu），而**参考 Kernel** 的 PR 改进了排行榜的运行时间 ([PR #31](https://github.com/gpu-mode/reference-kernels/pull/31))。

**4. 提示工程、分词与内存失误**

- **分词难题：Gemma、BOS Token 与 PromptTemplates**：Torchtune 中的 **GemmaTokenizer** 被发现由于缺失 PromptTemplates 以及配置错误导致的多个 BOS Token，导致其输出 Token 与 **HFModelTokenizer** 不匹配，该问题追溯到 **HF/Google** 的 Tokenizer 配置 ([Torchtune](https://discord.com/channels/1216353675241590815))。
    - 讨论强调了模板与配置对齐的必要性，并指出即使是技术上“正确”的实现，在实际 LLM 使用中也可能存在功能缺陷。
- **LlamaIndex Memory 迎来数据库化改造**：**LlamaIndex** 为 Agent 工作流推出了 [Memory 组件](https://t.co/CDOB3UUO4W)，支持内存和可扩展的数据库后端（SQLite, PostgreSQL），并针对长对话历史的上下文序列化与数据库存储方案展开了讨论。
    - 对于大规模或结构化历史记录，建议使用数据库而非简单的序列化，用户们对比了在 LLM 驱动的 Agent 中实现持久化上下文的权衡。


---

# Discord: 高层级 Discord 摘要




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Sonar 模型横扫基准测试**：根据最近的基准测试评估，**Sonar Pro Low** 在 BrowseComp 上的表现优于 **Claude 3.5 Sonnet**，同时 **Sonar Low** 的表现也优于 **Claude 3.7 Sonnet**。
   - **Sonar Pro Low** 在 BrowseComp 上实现了 **4.0%** 的准确率，比 **Claude 3.5 Sonnet** 高出近 **50%**，且 **Sonar** 和 **Sonar Pro** 的响应速度提升了高达 **3 倍**，延迟表现也更加稳定。
- **Deep Research 发布时间推测**：成员们推测 **Perplexity AI** 的新 *Deep Research* 功能是在 *Comet* 项目之前还是之后发布，有人表示 *Deep Research 应该会先发布*。
   - 另一位成员希望 *Comet* 不仅仅是 *一个带有 Copilot 风格侧边栏的浏览器*，并引用了一个 [思考中的蝙蝠侠 gif](https://tenor.com/view/hmmm-thinking-batman-gif-6153870554148391864)。
- **Merlin AI 的定价引起用户不满**：用户讨论了 **Merlin AI** 的定价，有人指出其由于频率限制不透明而存在 *价格猫腻*，另一人分享说在询问支持团队时得到了 *糟糕的回应*，尽管他赞扬了其网页搜索质量。
   - 还有人指出，对于 **Merlin AI** 的标准付费账户，*任何每天超过 16 美元的使用量都将导致当天服务立即终止*。
- **Perplexity Pro 开启 API 访问权限**：**Perplexity Pro** 实际上包含每月 **$5** 的 API 额度！
   - 你可以在 [Perplexity AI docs](https://docs.perplexity.ai) 找到 API 文档并开始使用；虽然需要通过信用卡注册，但如果用户的使用量保持在 $5 或以下，则不会被收费。
- **Perplexity Projects 引发 Vibe Coding 担忧**：关于 **Perplexity** 新的 *Project* 功能的猜测不断涌现，用户分享了截图和视频（如 [这一个](https://cdn.discordapp.com/attachments/1047649527299055688/1372028254663475210/Perplexity_Projects.mp4?ex=6825f122&is=68249fa2&hm=d16403c5811e8e1413d2d3c3f0e0261a02c64f55f15c55761c76b5501adae75c&)），一些人对其可能被滥用于 *Vibe Coding* 表示担忧。
   - 还有关于某些用户如何获得该功能早期访问权限的问题，引发了关于测试目录和 AI 社区内潜在人脉关系的讨论。



---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Embedding 模块遇到文件大小限制**：用户在 LM Studio 中使用 Embedding 模块时遇到了 *"Maximum file size per conversation reached"*（每个对话已达到最大文件限制）错误，当使用默认的 *nomic-ai/nomic-embed-text-v1.5-GGUF* 模型时，上限被限制在 **31.46 MB**。
   - 一位用户寻求处理几百 MB 的文件用于音频转歌词生成，这表明需要更大的 Embedding 模块容量。
- **LM Studio 的日志刷屏被视为无害**：LM Studio 中报告的一个日志刷屏（log spam）问题被认为是*无害的（benign）*，并已在 **0.3.16 build 2** 版本中得到解决。
   - 此修复应能减轻运行 Embedding 模型时对潜在问题的担忧。
- **LM Studio JIT 加载出现故障**：通过 Continue VS Code 插件进行的 LM Studio JIT 模型加载有时会在已加载另一个模型的情况下为客户端提供错误的模型。
   - 建议用户使用 `lms ls` 检查模型标识符，并从配置中 **移除 8bit** 以解决匹配错误。
- **开发者建议不要从零开始构建 LLM**：成员们讨论了从零开始构建 LLM 的话题，大多数人建议不要这样做，因为计算成本和数据需求巨大，转而推荐进行 Fine-tuning，并分享了 ["Build a Large Language Model from Scratch" Manning 书籍](https://www.manning.com/books/build-a-large-language-model-from-scratch) 以获取理论知识。
   - 一位成员表达了为各种用例构建模型的兴趣。
- **PCIE 5.0 带来性能飞跃**：一位用户报告在升级到 **50 系列显卡**后性能有所提升，使用 qwen3-14b-q4km 和 4096 context 时，速度从 26 tkps 提升到 38 tkps，这归功于 **PCIE 5.0** 的优势。
   - 该用户强调了由于更短的 PCIE 连接器设计，将显卡安装在底部插槽而不会产生干扰的优势。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Gemma 'Cutiepie' 即将到来？**：成员们推测将推出一款新的 **Gemma** 模型，暂定名为 **Cutiepie**，可能在 **Google I/O** 期间发布。
   - 一些成员正试图确定 **Gemma** 是否可以在无需登录的情况下免费使用。
- **DeepSeek R2 仍处于保密状态**：关于 **DeepSeek R2** 的推测已经降温，但它仍在使用 **Huawei** GPU 和中国资金进行开发。
   - 内部存在发布令人印象深刻产品的压力，特别是考虑到当前模型在推理轨迹（reasoning traces）和多语言 Chain of Thought 方面的实力。
- **Gemini 2.5 Pro 在 C++ 方面表现出色**：成员们赞扬了 **Gemini 2.5 Pro** 的编程能力，尤其是在 C++ 方面，有人将其描述为“梦想成真”。
   - 尽管反响热烈，一些用户报告模型存在幻觉（hallucinations）问题，而另一些人则认为它优于 **o3**。
- **o3 Pro 开发停滞？**：**o3 Pro** 的发布推迟了，有人认为 **OpenAI** 正在战略性地等待其他实验室亮出底牌以保持领先。
   - 内部认为 **OpenAI** 已经拥有 **o4** 但出于战略原因将其保留，并且期待 **Claude 4 + o3 pro** 的“疯狂组合”。
- **GPT 4.1 被誉为编程天才**：成员们强调了 **GPT 4.1** 的编程能力，指出它在大型代码库上的快速编译和错误修复能力，以及即时的响应速度。
   - 一位成员发现 **GPT 4.1** 对免费用户来说是 **GPT 4o mini** 的重大升级，但觉得 *4.1 纯粹是为了编程而生*。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **文件使用修复令人沮丧**：用户在尝试使用 **file://** 加载图像时遇到了 `AttributeError`，最初怀疑是 *image* 变量被错误地设置为 **None** 的问题。
   - 尽管纠正了路径，错误依然存在，这促使人们考虑将 URL 作为替代方案，一位用户感叹道：*图像微调（image finetuning）真让我头疼*。
- **Qwen3 推理速度变慢**：一位用户报告称，在 A100 上使用 **Qwen/Qwen3-0.6B** 时推理速度较慢，在使用 Unsloth 的 `FastLanguageModel.from_pretrained()` 且 batch size 为 1 的情况下，仅达到 **30 tokens/second**。
   - 其他人建议，较小的模型尺寸和 batch size 可能会抵消 `load_in_8bit=True` 带来的好处，而另一位用户指出，基础模型在 `tokenizer_config.json` 中不包含 chat template。
- **O3 表现出色，但在编程方面除外**：成员们吐槽 **O3** 输出的代码是垃圾，认为它更适合通过工具调用进行规划和研究，而 **GPT 4.1** 才是最好的编程模型，拥有 edu 账户的用户可以在 Github Copilot 中使用它。
   - 有人建议 **O4 mini high** 在编程方面表现更好，一位成员直言 **O3** 在 *编程方面简直是垃圾*。
- **Qwen3：基准测试力压 Gemma**：据观察，**Qwen 3 模型**在同等尺寸下表现优于 **Gemma 3**，其中 **Qwen 3 8b 模型**通过微调达到了接近 **SOTA** 的水平。
   - 它被认为是本地 LLM 的完美平衡点，一位成员宣称 **Qwen 3 8b 模型** *强得离谱*。
- **Repo2txt 对比 RAG**：一位成员提倡使用 [repo2txt.com](https://repo2txt.com/) 作为代码注入中 RAG 的优选替代方案，该工具可以从仓库中选择文件并直接注入到 prompt 中。
   - 他们认为模型 *无法自动读取 GitHub 中的所有代码，并且会犯错*。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-4.1 亮相并在编程方面表现亮眼**：**GPT-4.1** 是一款专门针对编程和指令遵循进行优化的模型，现已在 **ChatGPT** 中面向 Plus、Pro 和 Team 用户开放，并将在未来几周内面向 Enterprise 和 Edu 用户开放。
   - **GPT-4.1 mini** 正在 ChatGPT 中为所有用户取代 **GPT-4o mini**，安全评估详情可在 [Safety Evaluations Hub](https://openai.com/safety/evaluations-hub/) 查看。
- **无限制的 O3 吸引用户**：用户对无限制的 **O3** 模型赞不绝口，称其为解决问题的 *传奇模型*，现在还支持通过 **GitHub Repos**、**OneDrive** 和 **SharePoint** 进行 deep research，并且每个 Chat/Project 支持 **20 个文件上传**。
   - 尽管 Teams 计划提供了令人向往的内部知识库功能，但成员们建议坚持使用 Pro 计划，因为可以享受无限制 **O3** 的好处。
- **PII 数据护栏引发调查**：成员报告了 **PII 护栏** 拦截来自 HR 数据连接应用的家庭地址请求的挑战，尤其是那些有权访问 HR 数据应用。
   - 他们建议用户联系 [OpenAI support](https://help.openai.com/en/articles/6614161-how-can-i-contact-support) 以获取处理敏感数据请求和遵守 **PII** 政策的指导。
- **AI 宇宙模拟器初具规模**：一位成员正在使用 AI 构建 **1:1 宇宙模拟**，以探索思考并创建自动化生态系统，目标是将其扩展到浏览器中。
   - 重点在于模型之间的开放通信线路，优先考虑效率而非模型堆叠。
- **Ollama 势头盖过 Windsurf**：用户讨论了使用 **Ollama** 进行本地模型推理作为 **Windsurf** 等服务的替代方案，并警告不要为 AI 应用开发支付此类服务费用。
   - 建议重点学习 prompting 以避免 API 成本，并在 **VS Code** 中配合 **Continue** 和 **Roo Code** 等插件使用 **Ollama**，同时关注 [Hugging Face Learn](https://huggingface.co/learn) 上的 LLM 和 Agentic 课程。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **考虑 GPU Power 点对点方案**：成员们思考了在[点对点或去中心化系统](https://www.google.com/search?q=GPU+power+in+decentralized+systems)中利用 **GPU power** 的开源方案。
   - 讨论未得出明确的解决方案。
- **Cursor 20% 加价引发争论**：用户讨论了 **Cursor 相比实际 API 价格加价 20%** 的合理性。
   - 有人认为这很划算，而另一个人声称在 Cursor 之外使用 **Claude Max** 每月可以节省 *$600*。
- **Gemini 2.5 Pro 加入模型阵容**：社区注意到 [5 月 6 日](https://discord.com/channels/1074847526655643750/1074847527708393565/1371986950498414662) **Gemini 2.5 Pro** 已加入 Cursor 的可用模型阵容。
   - 有人指出新模型*终于修复了“我现在开始写代码（然后停止编码）”的问题*。
- **Monorepo 方法论探讨**：用户讨论了多仓库项目的管理，有人建议合并为单个 monorepo 以避免[上下文碎片化 (context fragmentation)](https://www.google.com/search?q=ai+model+context+fragmentation)。
   - 另一位用户提到在 Cursor **0.50.x** 中，可以在同一个父文件夹内使用独立的 workspace。
- **后台 Agent 频繁请求确认**：用户对后台 Agent 需要过度确认表示不满，这增加了 [fast request 的消耗](https://www.google.com/search?q=Cursor+fast+requests)。
   - 一位用户抱怨后台 Agent *一直要求确认操作，导致 fast request 消耗得更快*。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **AI 监管面临长达十年的冻结**：众议院共和党人在**预算和解法案 (the Budget Reconciliation bill)** 中加入了相关条款，将**禁止所有州和地方政府在 10 年内监管 AI**，这可能会影响隐私监管（[来源](https://arstechnica.com/ai/2025/05/gop-sneaks-decade-long-ai-regulation-ban-into-spending-bill/)）。
   - 该条款由肯塔基州众议员 Brett Guthrie 提出，广泛禁止州或地方执行监管 AI 模型或系统的法律。
- **AlphaEvolve 重新发现 SOTA 算法**：**DeepMind 的 AlphaEvolve** 是一款由 Gemini 驱动的编程 Agent，它结合了 LLM 的创造力和自动评估器，为数学和实际应用*演化算法*（[DeepMind Blog](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)）。
   - 该系统在约 75% 的案例中重新发现了 SOTA 解决方案，并在 20% 的案例中*改进了此前已知的最佳方案*，甚至推进了 **kissing number problem**（接吻数问题）。
- **RL-Diffusion 引发专利性质疑**：成员们讨论了开发 **RL-Diffusion**，将前向和后向过程合并为由 **RL** 控制的一个过程，但一些人对其新颖性和实际落地表示怀疑。
   - 他们强调，要转化为符合专利申请条件的方案，不能仅仅陈述抽象概念并加上“应用它”这类词。
- **GSM8K 基准测试接近满分**：语言模型在 **GSM8K** 等小学数学基准测试上的准确率正接近完美，详细分析见[此论文](https://ssrn.com/abstract=5250629)和[此博客文章](https://physics.allen-zhu.com/part-2-grade-school-math/part-2-1)。
   - 该论文旨在确定语言模型是真正培养了推理能力，还是仅仅在记忆模板。
- **关于 LLM 规划能力的辩论**：成员们根据模型会避免不必要计算的说法，辩论了 **LLM** 在生成解决方案之前是否会制定计划。
   - 一位成员反驳道，*模型了解到不必要的东西是随机噪声，根据定义，这些噪声没有可供模型学习的信号，因此它会忽略它们*。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Personality 平台推出聊天机器人定制化功能**：一个名为 **Personality** 的新聊天机器人平台已上线，旨在提供比 c.ai 等现有解决方案更多的定制化和更少的过滤，用户可以在 [personality.gg](https://personality.gg) 创建多个角色并进行角色扮演。
   - 该平台还在 [personality.gg/playground](https://personality.gg/playground) 提供免费的图像生成功能，但需要注意的是，此功能并非由 **OpenRouter** 驱动。
- **OpenAI 推理模型命名引发混淆**：用户要求 **OpenAI 推理模型**（例如 `/openai/o4-mini-high`）的命名保持一致性，以便为所有模型包含推理级别变体，正如 [OpenAI documentation](https://platform.openai.com/docs/models) 中所记录的那样。
   - 主要目标是简化对不同推理模型的评估，并减少对哪些模型具备哪些功能的混淆。
- **免费 Google 模型遭到限流**：用户报告称，即使有可用额度，免费的 **Google 模型** 的速率限制（rate limits）也极低甚至不存在，因此建议使用 **DeepSeek V3** 等替代方案。
   - 针对 [Twitter](https://fxtwitter.com/officiallogank/status/1922357621178200248) 上分享的一项变动，用户也对 **Gemini** 可能会移除免费路线表示担忧。
- **Claude 的 System Prompt 导致 OpenRouter 表现差异**：通过 **OpenRouter** 使用 **Claude** 与直接使用 **Anthropic** 官网之间的“乐于助人”程度差异，归因于 **Anthropic** 使用了包含约 **16000 tokens** 的庞大 System Prompt。
   - 用户可以手动实现该 [system prompt](https://docs.anthropic.com/en/release-notes/system-prompts)，该 Prompt 可在 [GitHub](https://github.com/jujumilk3/leaked-system-prompts/blob/main/anthropic-claude-3.7-sonnet_20250224.md) 上获取，其中包含工具（tools）调用。
- **“始终使用此密钥”选项默认指向特定密钥**：引入了一个名为“Always use this key”（始终使用此密钥）的新选项，由于其与名为“Use this key as a fallback”（将此密钥作为备选）的选项相似但功能不同，引发了混淆。
   - 新选项**仅**使用指定的密钥，并阻止回退（fallback）到 **OpenRouter**，这与旧的备选设置行为有所不同。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 积分故障与时区同步问题**：多名用户报告称，Manus 上的每日 **300 积分** 未能在 **00:00 GMT** 刷新，这可能是由于时区处理不一致导致的。
   - 一位用户指出，他们的积分在所在时区的 **晚上 8:00** 刷新，凸显了时间上的差异。
- **邀请码狂热及其来源**：一名用户炫耀其拥有 **100 个邀请码** 的“故障账号”，并分享了多个 [邀请链接](https://manus.im/invitation/MFCISJE3F75HD)，引发了关于其来源的讨论。
   - 有推测认为这些代码来自付费订阅，而其他人则质疑这些代码对现有成员的价值，因为新用户本身就能获得 **500 积分**。
- **任务失败后的退款难题**：一名用户对在 Manus 上任务失败消耗了 **800 积分** 却无法获得退款表示沮丧。
   - 其他用户表示，即使服务出现故障，通常也不会提供退款，并建议尝试“争议（dispute）”该笔费用。
- **针对 Facebook Marketplace 压价行为的说唱请求**：一名用户请求 Manus 生成一段关于 *Facebook Marketplace 恶意压价（lowballing）* 的说唱，并使用了 *best price*（最低价）、*last price*（底价）和 *mates rates*（熟人价）等俚语。
   - 该用户澄清说，这一请求并非为了广告宣传，而是为了通过说唱表现与在线二手市场相关的场景和经历。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **needs_exact_strides 优于 needs_fixed_stride_order！**：在 **pytorch nightly** 中，`at::Tag::needs_exact_strides` 优于 `needs_fixed_stride_order`，因为后者偶尔会出现不准确的情况。
   - 一位开发者在 **C++ code** 中移动了 `.contiguous` 调用，以防止 `torch.compile` 产生干扰。
- **使用 PTX 指令的矩阵魔法**：一位成员分享了一篇[博文](https://veitner.bearblog.dev/load-and-store-matrices-efficently-with-ptx-instructions/)，详细介绍了如何使用 **PTX instructions**（包括 `ldmatrix` 指令）在 warp 内高效地**加载和存储矩阵**。
   - 他们还链接了[相关代码](https://github.com/simveit/load_and_store)、[PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/contents.html) 以及一篇解释 `stmatrix` 指令的 [LinkedIn 帖子](https://www.linkedin.com/posts/simon-veitner-174a681b6_load-and-store-matrices-efficently-with-ptx-activity-7328421815331037185-_CW-?utm_source=share&utm_medium=member_desktop&rcm=ACoAADJYtOgBMOUI4WiWTyiAkroFBP1VujIWeksIn)。
- **参考内核（Reference Kernel）获得性能提升！**：一个 [pull request](https://github.com/gpu-mode/reference-kernels/pull/31) 已被合并，旨在缓解参考时间过长的问题，目标是在主机器人更新时缩短运行时间。
   - 该更新解决了参考实现耗时过长的担忧，特别是当更快的实现需要多次运行才能满足终止标准时。
- **CUTLASS 4.0 和 CuTe DSL 发布，但出故障了？！**：**CUTLASS 4.0** 和 **CuTe DSL** 现已发布，可通过 `pip install nvidia-cutlass-dsl` 获取，NVIDIA 建议从 [Jupyter notebooks](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks) 开始。
   - 成员们注意到 `nvidia-cutlass-dsl` 的版本是 `0.0.0....`，根据 pypy 显示这是大约 2 个月前发布的，因此*这次发布似乎出了些问题*。
- **由遗传算法起草的 Factorio 蓝图**：一位成员计划创建一个遗传算法，根据建筑材料、输入/输出位置和区域限制等特定要求生成 Factorio 蓝图，并找到了一篇关于动态路径规划的遗传编程[论文](https://arxiv.org/pdf/2102.048711.pdf)。
   - 该算法旨在让 **LLMs** 提供满足这些要求的常量，作为动态工厂设计的工具。



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 2.5 Pro 重写大文件**：用户发现通过 OpenRouter 使用 `--edit-format diff-fenced` 的 **Gemini 2.5 Pro** 有时会为了微小的改动而重写整个大文件，而其他人则报告 **AI Studio** 提供的结果更快。
   - 一些人更倾向于在工作流中使用 **Sonnet 3.7**，用较便宜的模型处理简单任务，用 **Sonnet 3.7** 处理更复杂的架构。
- **Common Lisp 获得现代 AI 工具支持**：用户讨论了利用现有模型来增强 **Common Lisp** 等语言的开发，计划利用书籍和数据源生成数据集和提示词（prompts）进行上下文学习（in-context learning），其中一人计划使用 **Lisp DSL** 构建编译器/解释器。
   - 该方法涉及对**小型模型进行 LoRA 微调**，并采用语义检索将编程书籍知识整合到上下文窗口中。
- **Gemini 模型添加注释和多余想法**：一位用户观察到 **Gemini** 在代码中直接添加了过多的注释和多余的想法，并随后执行了这些代码。
   - 该用户建议强制执行严格的代码更改，而不纳入未经请求的建议。
- **Aider 升级仍然麻烦**：用户在升级 **Aider** 时仍面临问题，即使升级过程看起来成功，版本号也无法更新。
   - 升级过程中的 SSL 警告可能与之无关，并且自一月份以来一直是一个持续存在的问题。
- **Aider 获得了澳洲风格的聊天改造**：一位用户发现了一种通过修改 `~/.aider.conf.yml` 文件来提高 **Aider 回复**可读性的方法。
   - 他们建议使用 `chat-language: English (Australia, use headings, bullet points, concise sentence fragments)` 以获得更简洁的输出。



---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **LM-Eval-Harness 获取数据集更快捷**：用户现在可以在 **lm-eval-harness** 中通过 `python3 lm_eval --model dummy --tasks [task_list] --limit 1` 下载特定任务的数据集，而无需立即评估模型。
   - `dummy` 模型[在此定义](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/dummy.py)，它返回随机数，仅用于测试目的。
- **R1-distill 获得正式 Prompt 引导**：成员们就应该使用 *"user: What is xyz? assistant: <think>"* 格式还是 *"What is xyz? <think>"* 格式来引导 **R1-distill** 模型展开了辩论。
   - 辩论结束时未达成共识。
- **LLM 面临监管压力**：成员们讨论了关于算法中**偏见标准（bias standards）**的法规，特别是在 **LLM** 背景下，提到了 **NCUA、EEOC、FDIC、HHS** 和 **FTC** 等监管机构。
   - 根据[已存档的 EEOC 指南](https://web.archive.org/web/20240123163943/https://www.eeoc.gov/laws/guidance/select-issues-assessing-adverse-impact-software-algorithms-and-artificial)，监管机构可能认为无法被研究的算法构成了侵权。
- **Skywork 的推理能力受到关注**：**Skywork 模型**及其技术在发布后受到赞誉，文中提供了 [Skywork Open Reasoner Series](https://capricious-hydrogen-41c.notion.site/Skywork-Open-Reasoner-Series-1d0bc9ae823a80459b46c149e4f51680) 的链接。
   - 一位成员指出，Skywork 是根据训练 Batch 中的总 Token 数进行归一化，而不是按每个序列归一化。
- **lm-eval 的多 GPU 困扰**：一位成员报告称，在 **lm-eval** 中使用 `parallelize=True` 时 GPU 利用率不均，**GPU 0** 占用率为 **100%**，而 **GPU 1** 为 **0%**。
   - 另一位成员建议使用 **vllm tensor parallel**，因为它更可靠，并建议使用 `accelerate launch -m lm_eval ...` 来运行多个副本，而不是使用采用朴素流水线并行（naive pipeline parallelism）的 `parallelize`。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Atropos 获得 Axolotl 支持**：Nous Research 发布了 [Atropos v0.2.0](https://t.co/F6hr9JgZpm)，其特点是集成了 **Axolotl** 作为官方训练器合作伙伴，并提供了[使用指南](https://github.com/axolotl-ai-cloud/plugin-atropos)。
   - 此次更新包括新环境、更新的 API 处理以及更好的 TRL 支持。
- **Psyche Network 使训练民主化**：Nous Research 推出了 [Psyche Network](https://nousresearch.com/nous-psyche/)，这是一个去中心化训练网络，旨在通过汇集分布式计算资源来训练大规模模型，从而使 AI 开发民主化。
   - 测试网启动涉及预训练一个使用 **MLA Architecture** 的 **40B** 参数 **LLM**，数据集包含 **FineWeb (14T)**、**FineWeb-2 (4T)** 和 **The Stack v2 (1T)**。
- **DisTrO 优化器打破带宽上限**：**Psyche** 网络利用 Nous 的 **DisTrO** 优化器和自定义的点对点网络栈来协调全球分布的 GPU，克服了以往 AI 训练中的带宽限制。
   - 成员可以贡献 USDC 以换取算力。
- **多轮对话难倒 LLM**：[Lost in Conversation 论文](https://arxiv.org/abs/2505.06120)及其[对应的 GitHub 仓库](https://github.com/microsoft/lost_in_conversation)分析了 **LLM 在多轮对话中**与单轮设置下的表现，结果显示在多轮场景中，六个生成任务的平均性能下降了 **39%**。
   - 论文将此归因于*轻微的能力损失和显著的不可靠性增加*，结论是*当 LLM 在对话中走错一步时，它们就会迷失方向且无法恢复*。
- **基准测试失效，前沿模型表现不稳**：一位成员发现*很难摸清哪个前沿模型在不同任务上表现更好*，因为*基准测试不够细致或多样化*，并分享了[一个链接](https://x.com/timfduffy/status/1921964622589919651?t=7z-QLXzI9BDwm0lGqArgUg&s=19)。
   - 他们表示，“最强”的编程模型在前端、特定框架或数据可视化等方面可能仍然表现糟糕。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 征集用户体验反馈**：NotebookLM 团队正在寻求用户对 **multilingual Audio Overviews** 的意见，旨在通过邀请用户参与 **User Experience studies** 来提升用户体验。
   - 鼓励用户提供反馈，以改进 **NotebookLM** 内部的多语言音频功能。
- **Invisible Sun TTRPG 在 NotebookLM 的助力下大放异彩**：一位用户正在学习 Monte Cook Gaming 出品的 **Invisible Sun TTRPG**，通过使用 **NotebookLM** 和 **ChatGPT Projects** 来查询规则，并指出“可分享性”是其更青睐 **NotebookLM** 的原因。
   - 他们计划在通过 Backerkit 获取的新书上测试 **NotebookLM** 的见解能力。
- **Google 用户对 NotebookLM 可能关停感到担忧**：一位用户表达了对 **NotebookLM** 可能被停止服务的担忧，理由是 **Google** 过去有关停产品的历史，特别是常在“不方便的时间”关停。
   - 其他人则认为 **NotebookLM** 的独特价值使其不太可能被关停，并建议这可能只是品牌重塑或营销活动的前奏。
- **PDF 上传限制令用户恼火**：多名用户报告遇到账号限制，导致无法向 **NotebookLM** 上传 **PDFs**。
   - 讨论中并未给出该问题的明确解决方案。
- **延长播客长度的技巧出现**：一位用户询问如何延长播客长度，另一位用户建议通过重复添加相同主题的链接或文档来填充内容，从而达到 **22 分钟** 的时长。
   - 目前尚不确定这一策略是否普遍有效。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **分享 OpenAI 发布背后的故事**：一位成员分享了来自 [andrewmayne.com](https://andrewmayne.com/2025/05/02/some-personal-observations-about-the-launch-of-gpt-4/) 的关于 **OpenAI launches** 的温馨故事。
   - 作者回忆了早期阶段以及扩展过程中面临的挑战。
- **ChatGPT 扩展架构深度解析**：社区分享了一篇来自 **Pragmatic Engineer** 的时事通讯文章，题为 [Building, launching, and scaling ChatGPT](https://newsletter.pragmaticengineer.com/p/chatgpt-images)。
   - 文章回顾了 **ChatGPT** 发布的历史和技术栈。
- **AI 驻场创始人寻求机会**：一位成员询问专注于 **AI** 的 **Founder in Residence programs**，寻求关于如何定位自己的建议。他们拥有为 **Amazon ads** 的**分析用例**构建 **AI systems** 的经验，并希望在相同的分析领域构建 **Self-Serve Agents**。
   - 未提供更多细节。
- **Gemini 通过 AlphaEvolve 赋能算法设计**：**Google DeepMind** 推出了 [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)，这是一个由 **Gemini** 驱动的 **coding agent**，旨在创建高级算法。
   - 这展示了如何利用 **coding agents**，可能成为算法设计领域的一个关键时刻。
- **Tom Yeh 教授将梳理 Llama 1/2/3/4 的演进历程**：Tom Yeh 教授将在一个[特别活动](https://lu.ma/se1f2bfk)中，通过一个课时讲解 **Evolution of Llama 1/2/3/4**。
   - 该活动由社区成员组织。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Qwen 模型蒸馏依然难以实现**：一名成员寻求 **distilling**（蒸馏） **Qwen** 系列模型的资源，但没有具体的 notebook 或参考资料被分享。
   - 社区成员可能掌握相关线索，因此进一步探索可能会有所收获。
- **感知器可视化工具吸引社区关注**：一名成员分享了一个用于教育目的的 **perceptron visualizer**（感知器可视化工具），并在附带的视频 [My_Video_0079.mp4](https://cdn.discordapp.com/attachments/879548962464493619/1372054347214487692/My_Video_0079.mp4?ex=6826096f&is=6824b7ef&hm=ab7441c0985405a8eed58ca0326e1c1c2ca6678d723994c6540118d4f6dea15a&) 和 [My_Video_0080.mp4](https://cdn.discordapp.com/attachments/879548962464493619/1372054347562745896/My_Video_0080.mp4?ex=6826096f&is=6824b7ef&hm=65046c2cb4ad3b7724e812b12f8098ae1c181a7d4474bbcfbe89525a299a0272&) 中展示了其功能。
   - 另一名成员从 [darkspark.dev](https://darkspark.dev/) 为可视化收藏做出了贡献。
- **Stable Diffusion 在本地跑起来了！**：社区成员探索了结合使用 **Diffusers** 和 **TGI**，或者使用 **WebUI Forge** ([GitHub 链接](https://github.com/lllyasviel/stable-diffusion-webui-forge)) 或 **reForge** ([GitHub 链接](https://github.com/Panchovix/stable-diffusion-webui-reForge)) 在本地运行 **Stable Diffusion**。
   - **Diffusers 文档** 链接（[huggingface.co](https://huggingface.co/docs/diffusers/main/en/stable_diffusion), [huggingface.co/learn](https://huggingface.co/learn/diffusion-course/unit0/1)）对配置也很有帮助。
- **PDF 格式在受欢迎程度调查中排名垫底**：用户对 **PDF 格式** 表达了强烈批评，其中一人将其描述为“见过最糟糕的格式”。
   - 一名用户建议增加 **markdown** 输出选项，以改善 **RAG ingestion**（RAG 摄取）的语义关系，但其他人对完全分类问题（尤其是表格）表示担忧。
- **Smolagents 框架表现不佳**：一名用户报告在课程中使用 **smolagents 框架** 配合 **Qwen** 及相关工具时，结果非常糟糕。
   - 这可能反映出该框架需要进一步完善，或者在类似任务中需要寻找替代框架。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Authpython API 更新滞后于 Typescript**：社区注意到 **Authpython** 的 API 更新通常比 **Typescript** 滞后约 **1-2 个月**，但分享了一个 [Go-MCP 客户端链接](https://socket.dev/go/package/github.com/blankloggia/go-mcp?section=files&version=v0.0.0-20250410110046-894efeed42d9&path=client.go) 供参考。
   - 这可能会影响依赖最新功能的项目的开发进度和集成工作。
- **使用 ithena-cli 调试 Smithery MCP 服务器**：成员建议使用 [ithena-cli 工具](https://github.com/ithena-one/ithena-cli) 调试运行在 **Smithery** 上的 **MCP server**。
   - 该工具存储所有输入和输出用于调试，提供详细的交互日志，以帮助识别 **Claude Desktop** 中的问题。
- **Tiny Agents 开始支持远程 MCP**：**Hugging Face Tiny Agents** 现在具备远程 **MCP Support**，能够从命令行连接到 **SSE** 和 **Streaming HTTP servers**。
   - 这一增强为 Agent 的开发和管理提供了一种“通用的方法”，扩展了 **Tiny Agents** 在网络环境中的能力。
- **Chatter：一个支持 MCP 的 Web 托管客户端问世**：一个新的、与 **LLM** 提供商无关且支持 **MCP** 的 Web 托管聊天客户端 [chatter](https://github.com/sakalys/chatter) 已开源，并托管在 [moopoint.io](https://moopoint.io/)。
   - 该客户端旨在作为 Claude Desktop 的 Web 替代方案，承诺很快将提供免费层级、记忆功能、**MCP server** 托管、图像处理、文件上传和语音交互。
- **Yarr MCP 服务器正式发布**：**ARRs MCP servers** 的新社区实现已上线 [GitHub](https://github.com/jmagar/yarr-mcp)。
   - 一名社区成员 [在 X](https://x.com/llmindsetuk/status/1922309194696335433)（原 Twitter）上提到了此事。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 模型与 vLLM 结合**：一位成员报告称，在他们内部版本的 **GRPO** 中，使用 **vLLM** 运行了一个自定义的 **Torchtune 模型**。
   - 在一位用户询问如何为其模型启用 **vLLM** 支持后，他们正考虑公开其实现。
- **vLLM 集成实现同步化**：一位成员建议除了异步版本外，再创建一个带有 **vLLM** 的同步 **GRPO recipe**。
   - 他们表示强烈倾向于 **vLLM** 版本，称其*真心找不到任何不这样做的理由*。
- **Gemma Tokenizer 与 HFModelTokenizer 产生差异**：一位成员发现，带有 **Gemma chat template** 的 **HFModelTokenizer** 生成的输出 token 与 **torchtune GemmaTokenizer** 的 token 不匹配。
   - 这表明 **torchtune** 的 **GemmaTokenizer** 可能未正确应用 chat template。
- **Gemma PromptTemplate 缺失**：有人注意到 **Gemma** 缺少特定的 **PromptTemplate**，这导致了错误的 tokenization 以及 `system` 角色可能出现的问题。
   - 虽然默认可能是使用 **Alpaca template**，但一个正确的 Gemma 专用模板至关重要。
- **HF/Google 配置导致 BOS Token 错误**：由于配置中同时包含 `"add_bos_token": true` 和 chat template 中的 **BOS token**，**HF tokenizer** 会添加多个起始符 (**BOS**) token。
   - 这个问题直接源自 **HF/Google 的 tokenizer 配置**，使得该实现在技术上是“正确”的，但在功能上存在缺陷。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Variant SIMD Bug 导致 Mojo 发生段错误 (Segfault)**：用户发现 Mojo 中使用 `SIMD` 类型时 `Variant` 会发生崩溃。当使用 `Variant[T](simd)` 时，打印语句之间会发生段错误，该问题似乎源于 `Variant` 内部的**空间分配不足**或**生命周期问题**。
   - [GitHub issue 4578](https://github.com/modular/modular/issues/4578) 上分享了一个可复现的示例，展示了该 Bug 相对于打印语句位置的随机行为。
- **Register Passable 类型面临审查**：关于在 Mojo 中使用超过系统寄存器大小的 `register_passable` 类型产生了疑问，由于 LLVM 的限制，这可能导致错误编译。
   - 当前的 `Variant` 实现对于 `sizeof[T]()` 超过系统寄存器大小的 register passable 类型 `T` 可能存在缺陷，建议替换为各种 `Trivial` 版本。
- **Mojo 与 Colab 集成**：现在使用 `import max.support.notebook` 可以更简单地在 Colab notebook 单元格中编译和执行 Mojo 代码，该命令引入了 `%%mojo` 魔法命令。
   - Modular 论坛上详细介绍了该[公告](https://forum.modular.com/t/max-can-now-be-used-in-google-colab/1383?u=bradlarson)。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **WebGPU 后端发现 Bug**：**WebGPU** 后端存在一个 Bug，生成的 kernel 没有连续的 **DEFINE_GLOBAL** 参数，导致 `bufs_from_lin` 出现问题，详情见[此处](https://x.com/charliermarsh/status/1922333022658978089)。
   - 据报道 *claude* 修复了它。
- **BEAM 参数破坏 WebGPU 性能**：设置 **BEAM** 参数会对 **WebGPU** 后端性能产生负面影响：在没有 beam 的情况下运行时间为 **30ms**，但在 **BEAM=1** 时为 **150ms**。
   - 在 **BEAM=2** 时运行时间为 **100ms**。
- **Tinybox UI 走向极简主义**：展示了一个 tinybox 的极简 UI 概念，特点是*无登录、无云端、无冗余*，强调快速、本地的硬件控制，详见[此处](https://github.com/dahsmartgirl/tinybox-ui)。
   - tinybox 的 HTTP 设置页面通常是受支持的，前提是它保持 **0 依赖**且*代码行数绝对最少*。
- **Blake3 用于张量存储的悬赏**：目前有一个针对高性能 **blake3** 实现的悬赏，用于云端的内容寻址张量存储。
   - 该实现应该是通用的。

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 的 Memory 组件提升 AI 性能**：LlamaIndex 推出了全新的 **Memory 组件**，为 AI Agent 提供了短期和长期记忆，改善了对话中的上下文关联，并支持静态记忆块 ([链接](https://t.co/CDOB3UUO4W))。
   - 一位用户报告了在 workflow 中使用 **Memory 组件**时遇到的挑战，特别是当 `user_id` 设置 `session_id` 时，记忆会在每次 workflow 调用时清除。
- **LlamaExtract 添加引用和推理功能**：@tuanacelik 发布的新代码演练展示了如何为 **LlamaExtract** 添加**引用和推理**功能 ([链接](https://t.co/z4RSGJ5gKI))。
   - 该演练展示了如何构建一个 schema，以告知 LLM 从复杂数据中提取哪些内容。
- **Memory 默认使用数据库，大规模应用推荐使用数据库**：**Memory 组件**默认使用内存中的 **SQLite DB**，但为了实现可扩展性，建议通过更改数据库 URI 来使用本地 **SQLite** 或 **PostgreSQL DB**。
   - 对于较长的聊天记录，使用数据库比将 `memory.to_dict()` 序列化为 JSON blob 更好。
- **上下文序列化与数据库之争**：一位用户质疑数据库连接是否优于使用 **Memory 组件**序列化上下文，因为上下文恢复可以还原聊天记录。
   - 得到的澄清是：序列化对于默认情况没问题，但对于**庞大的聊天记录**或需要保存结构化历史记录的需求，数据库更胜一筹，并指出 [python dict vs redis 是同样的问题](https://redis.com/)。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 用户关注计费**：一位用户分享了[计费仪表板的链接](https://dashboard.cohere.com/billing)，用于检查向 **Cohere** 发起的 **API calls** 数量。
   - 然而，一位用户指出 **trial key** 仅显示 tokens，而不显示原始请求数量，这表明 **Cohere** 可能没有明确计算 **API calls**。
- **用户权衡 Cohere 的价值**：成员们讨论了 **Cohere** 与 **ChatGPT** 和 **Anthropic** 等模型的对比使用案例。
   - 这一讨论突显了在 AI 模型竞争激烈的格局中，用户对 **Cohere** 定位的持续评估。
- **Cohere 的 Command aCurious 仍存疑问**：一位成员寻求关于 **Command aCurious** 建议生成参数的指导。
   - 该请求强调了理解和优化特定模型参数的重要性，以通过 **Cohere** 获得理想的结果。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Medium 文章或 X 帖子可解锁课程证书**：成员们澄清，获得课程证书需要撰写一篇 **Medium article** 或发布一条 **X post** 来总结其中一门讲座。
   - 有意向的成员必须通过[此表单](https://forms.gle/399VqBP88WL1AJ4J6)提交他们的作品以获得学分。
- **提交课程作业对获得证书至关重要**：为了获得证书，在完成 Medium 文章或 X 帖子后，必须通过提供的 **Google Forms 链接**提交课程作业。
   - 提交操作可确保作品被正确计入课程证书学分。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。

---

**Codeium (Windsurf) Discord** 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。

---

**Nomic.ai (GPT4All) Discord** 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。

---

您收到此邮件是因为您通过我们的网站选择了订阅。

想要更改接收这些邮件的方式吗？
您可以从该列表中[退订](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。

---

# Discord：分频道详细摘要和链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1371928240418984027)** (748 条消息🔥🔥🔥): 

> `Android 应用自定义模型选择，Deep Research 发布日期，Merlin AI 定价与网页搜索质量，Perplexity AI Sonar 对比 GPT 模型，AI Studio 的多模态实用性` 


- **用户询问 Android 应用上的自定义模型选择**：一位用户询问了在 Perplexity AI Android 应用程序中选择自定义模型（特别是 **Grok3**）的可能性，并附上了一张 [截图](https://cdn.discordapp.com/attachments/1047649527299055688/1371928239936901312/Screenshot_2025-05-13-22-10-50-96_4159553c7f58296d2732e906959db560.jpg?ex=68263cbd&is=6824eb3d&hm=898125df28a2a6b3f2815c14ddaa23bf96726614ed7ae167e9d069b9ebd2a293&)。
   - 讨论中未提供即时解决方案或规避方法。
- **"Deep Research" 预计即将推出，"Comet" 引发猜测**：成员们讨论了 **Perplexity AI** 新的 *deep research* 功能的时间安排，并猜测它会在 *Comet* 项目之前还是之后发布，其中一位成员表示，*根据他们得到的沟通，deep research 应该会先发布*。
   - 另一位成员表示希望 *Comet* 不仅仅是 *一个带有 Copilot 风格侧边栏的浏览器*，并引用了一个 [思考中的蝙蝠侠 gif](https://tenor.com/view/hmmm-thinking-batman-gif-6153870554148391864)。
- **Merlin AI 模糊的速率限制引发不满**：用户讨论了 **Merlin AI** 的定价，其中一人指出其由于速率限制不透明而导致 *定价阴暗*，另一人分享说，当询问支持团队时，对方给出了 *糟糕的回应*，但同时也称赞了其网页搜索质量。
   - 还有人指出，对于 **Merlin AI** 的标准付费账户，*任何每天超过 16 美元的使用量都将导致当天服务立即终止*。
- **AI Studio 被誉为卓越的多模态奇迹**：一位用户极力推崇 **AI Studio** 的多模态实用性，特别是它支持音频和视频输入，这是主流 LLM 聊天工具无法比拟的；他们还补充了一个关键细节：*AI Studio 是我们实现真正多模态实用性的救星*。
   - 用户将其与 **ChatGPT** 等平台进行了对比，并强调 AI Studio 的功能是 **免费** 的。
- **Projects 功能引发猜测与预览**：关于 **Perplexity** 新的 *project* 功能引发了猜测，用户分享了其功能的截图和视频（例如 [这一个](https://cdn.discordapp.com/attachments/1047649527299055688/1372028254663475210/Perplexity_Projects.mp4?ex=6825f122&is=68249fa2&hm=d16403c5811e8e1413d2d3c3f0e0261a02c64f55f15c55761c76b5501adae75c&)），一些人对其可能被滥用于 *vibe coding* 表示担忧。
   - 还有关于某些用户如何获得该功能早期访问权限的问题，引发了关于测试目录以及 AI 社区内潜在联系的讨论。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1371964518783258745)** (2 条消息): 

> `Token 最小化，Sustain` 


- **探索 Token 最小化技术**：一位用户分享了一个关于 sustain 的 **token minimization** 的 [Perplexity AI 搜索结果](https://www.perplexity.ai/search/e238e5ca-1109-4f96-afbb-7d462ed6085e#0)。
   - 随后，另一位用户分享了 **Perplexity AI 页面** 上关于同一主题的直接链接：[Token Minimization for Sustain](https://www.perplexity.ai/page/token-minimization-for-sustain-1Cbiopx3T3C5SWyrYTVvdw)。
- **Sustain 与 Token 优化**：讨论围绕在保持模型性能的同时 **减少 token 使用量** 的方法展开，这对于可持续 AI 实践至关重要。
   - 分享的资源包括 **高效 token 编码** 的技术以及在不牺牲关键信息的情况下 **最小化输入长度** 的策略。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1371988980341542992)** (12 messages🔥): 

> `Sonar 模型基准测试、Perplexity Pro API 访问、新任开发者关系驻场、Sharepoint 集成` 


- **Sonar 模型打破基准测试记录！**：根据最近的基准评估，**Sonar Pro Low** 在 BrowseComp 上的表现优于 **Claude 3.5 Sonnet**，而 **Sonar Low**（最便宜的模型）的表现也优于 **Claude 3.7 Sonnet**。
   - **Sonar Pro Low** 在 BrowseComp 上实现了 **4.0%** 的准确率，比 **Claude 3.5 Sonnet** 高出近 **50%**，且 **Sonar** 和 **Sonar Pro** 的响应速度提升了高达 **3 倍**，延迟表现也更加稳定。
- **Perplexity Pro 用户发现 API 访问权限**：Perplexity Pro 实际上每月包含 **$5** 的 API 额度！你可以在 [Perplexity AI docs](https://docs.perplexity.ai) 找到 API 文档并开始使用。
   - 通过信用卡注册是访问 API 的唯一途径，但如果用户的使用量保持在 $5 或以下，则不会被扣费。
- **新任 DevRel 驻场是 Robinhood 竞赛冠军！**：一位新任开发者关系（Developer Relations）驻场人员（<@543991504922738688>）加入了团队，他曾使用 Perplexity API 赢得了 Robinhood 竞赛，并将协助开发者使用 Sonar 进行构建。
- **Sharepoint 集成至 Perplexity**：一位用户报告称，Sharepoint 集成到 Perplexity Enterprise Pro 在 UI 界面中运行良好，能够提供相关的响应。
   - 然而，他们指出使用 Perplexity API 服务时无法获得任何有用的响应，并正在寻求建议。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1371925974232338575)** (176 messages🔥🔥): 

> `Embedding 模块问题、良性日志垃圾信息、LM Studio JIT、从零构建模型、LM Studio 自动加载问题` 


- **Embedding 问题在所有模型中出现**：一位用户报告了 Embedding 模块的问题，遇到了“达到每个对话的最大文件限制”错误，在使用默认的 *nomic-ai/nomic-embed-text-v1.5-GGUF* 模型时，上限被限制在 **31.46 MB**。
   - 该用户寻求克服这一限制，以便为一个音频转歌词生成应用处理几百 MB 的文件。
- **LM Studio 修复日志垃圾信息问题**：一位用户被告知他们遇到的日志垃圾信息问题是“良性的（benign）”，并且已在 **0.3.16 build 2** 版本中修复，或很快将得到修复。
   - 这解决了运行 Embedding 模型时对潜在问题的担忧。
- **LM Studio 存在 JIT 加载问题**：用户报告了 LM Studio 的 JIT 模型加载功能存在问题，特别是在通过 Continue VS Code 插件使用时。
   - 如果已经加载了另一个模型，服务器有时会向客户端提供错误的模型；建议使用 `lms ls` 命令检查所用模型的标识符，以确保配置与 LM Studio 匹配，并尝试从中**移除 8bit** 以修复该问题。
- **从零开始构建 LLM？**：成员们讨论了从零构建 LLM 的可行性，其中一位成员表示有兴趣为各种用例构建模型。
   - 然而，其他人强调了由于巨大的算力成本和数据需求，这并不切实际，建议改为对现有的开源模型进行微调，并推荐阅读 [《Build a Large Language Model from Scratch》Manning 出版社书籍](https://www.manning.com/books/build-a-large-language-model-from-scratch) 以了解理论知识。
- **加载 LM Studio 模型时出现问题**：一位用户报告称 LM Studio 的模型自动加载功能停止工作，即使重启应用后模型也无法加载；这些模型是通过 API 请求使用 JIT 系统加载的。
   - 有人提到这可能是由于客户端使用的模型标识符不匹配导致的，可以通过运行 `lms ls` 来解决。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1371927624338640907)** (450 条消息🔥🔥🔥): 

> `Gigabyte RTX 5060 Ti, PCIE 5.0 优势, qwen3-14b-q4km 性能, 双 GPU, Linux vs Windows 上的 ROCm` 


- **技嘉节俭的 RTX 5060 Ti 设计**：一位用户展示了一款新型 **Gigabyte GeForce RTX 5060 Ti** 显卡，指出其采用了超短 PCB 设计，尽管芯片支持 x8，但仅配备了 x8 物理接口，详见[这篇 videocardz.com 文章](https://videocardz.com/newz/less-than-10-cm-gigabyte-geforce-rtx-5060-ti-cards-come-with-ultra-short-pcb-design)。
   - 该用户称赞了短板和大型贯穿式散热设计，认为其有利于适应未来需求且易于安装，同时幽默地批评了 **PNY 的设计选择**，认为其优先考虑美学而非功能，阻碍了气流。
- **PCIE 5.0 提升显卡性能**：一位用户报告了升级到 **50 系列显卡**后的性能提升，具体表现为在使用 qwen3-14b-q4km 且 context 为 4096 时，速度从 26 tkps 提升至 38 tkps。这归功于 PCIE 5.0 在 M.2 插槽被占用导致顶部插槽限制为 x8 的主板上的优势。
   - 该用户还赞赏了能够将显卡安装在底部插槽而不受干扰的能力，强调了较短 PCIE 接口设计的优点。
- **双 GPU 受限于较慢的显卡**：当被问及同时使用 **两个 GPU** 时，一位用户解释说，速度会被限制在较慢显卡的性能加上额外开销（overhead）的水平，这表明系统的运行速度仅取决于最慢的组件。
   - 随后他们分享了一张 **PNY GPU** 的图片，并澄清他们已经退货了。
- **GMK Strix Halo 迷你主机**：用户们正热切期待他们的 **GMK Strix Halo** 迷你主机，其中一位用户订购了两台，并计划在到货后发布 LM Studio 的性能测试结果，同时希望能够实现跨多台计算机运行模型拆分。
   - 另一位用户指出，**GMKtec** 现在开始为 5 月 7 日至 13 日期间下的订单发货。
- **Linux 上针对 AMD GPU 的 ROCm**：一位用户分享说，切换到 Linux 以获得 **ROCm** 支持并不那么困难，而另一位用户则报告了 Linux 上 ROCm 检测的问题，并指出与 Windows 上的 Vulkan 相比没有性能差异。
   - 一位用户表示 *Vulkan 可能更快，但在 flash attention 方面存在 bug*，而 *ROCm 则要稳定得多*。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1371925179315523685)** (530 条消息🔥🔥🔥): 

> `DeepSeek R2 发布, 新 Gemma 模型, Claude Neptune / 3.8 泄露, GPT-4.1 vs GPT-4o, o3 Pro 发布延迟` 


- **Gemma 获得名为 Cutiepie 的新模型**：成员们讨论了一个名为 **Cutiepie** 的新 **Gemma** 模型，一些人认为如果时间线吻合，它可能会在 **Google I/O** 上发布。
   - 其他人正试图在竞技场中寻找强大的匿名模型，并想知道 **Gemma** 是否可以在不登录的情况下免费使用。
- **DeepSeek R2：依然杳无音信？**：社区注意到，关于新 **DeepSeek R2** 发布的持续猜测终于平息了，但它仍在使用 **Huawei** 最新的 GPU 和中国资金进行开发。
   - 在内部，他们可能正面临发布令人印象深刻的产品并解决服务器问题的压力；目前的模型在推理轨迹（reasoning traces）和多语言思维链（chain of thought）方面表现出色。
- **Gemini 2.5 Pro 对标竞争对手**：成员们讨论了 **Gemini 2.5 Pro** 并表示它优于竞争对手，尤其是在编程方面，有人提到用它编写 C++ 代码简直是 *梦想成真*。
   - 用户发现目前的模型幻觉（hallucinations）非常严重，尽管其他人认为它比 **o3** 更好，并开始感到 **Gemini** 带来的威胁。
- **o3 Pro 推迟，o4 指日可待？**：关于 **o3 Pro** 推迟发布的猜测仍在继续，一些人认为 **OpenAI** 正在等待其他实验室亮出底牌以夺取头把交椅，且 Google 的活动可能即将到来。
   - 据悉，**OpenAI** 内部已经拥有 **o4**，但出于与其他实验室相关的战略原因而将其保留；人们相信 *Claude 4 + o3 pro 将是一个疯狂的组合*。
- **GPT 4.1：它是编程奇才吗？**：成员们讨论了 **GPT 4.1** 的编程能力，强调了它在大型代码库上的快速编译和错误修复能力，以及即时的响应时间。
   - 一位成员指出，对于免费用户来说，从 **GPT 4o mini** 升级到 **4.1 mini** 是一个巨大的提升，但同一位成员觉得 *4.1 纯粹是为了编程，在其他方面其实一般*。


  

---

### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1372260480365625484)** (1 条消息): 

> `服务器更新、论坛类别、角色创建、审核改进、未来活动` 


- **LMArena 服务器进行更新**：LMArena 服务器实施了一些更改，以打造一个更具参与感和受保护的空间，这些调整由社区反馈驱动；成员可以通过 [此表单](https://docs.google.com/forms/d/e/1FAIpQLSc8sOvadJhluCMTxClbwkyz8ysm1sXfhHu0OE-FMm2cPxaspw/viewform?usp=header) 提交建议。
   - 这些更改包括服务器结构调整、审核改进以及定期活动的计划。
- **引入新的论坛类别**：新增了一个 **Forum Category**（论坛类别）用于收集反馈、排除故障和处理模型请求，取代了现有的 <#1347792466203381830> 和 <#1346566622927655033> 频道，以实现更好的组织。
   - 如果频道不可见，用户可能需要在 ``Channels & Roles -> Browse Channels`` 中启用这些频道。
- **为定向公告创建角色**：在 ``Channels & Roles`` 部分通过自动分配的问题创建了新的 **Roles**（角色），以便进行更具针对性的公告，确保成员接收到最相关的信息。
   - 位于频道列表顶部的 **Server Guide** 现在包含 <#1343285970375540839> 和 <#1340554757349179416> 频道。
- **通过 ModMail 机器人改进审核**：对于紧急需求，可以提及 <@&1349916362595635286> 角色；对于私人问题，成员现在可以向位于 ``Member List`` 顶部的 **ModMail 机器人发送私信**。
   - <#1343285970375540839> 已更新，以保持讨论主题明确并营造更具包容性的空间。
- **未来将举办更多定期活动**：计划定期举办活动，包括 **Staff AMA（员工问答）、竞赛以及休闲游戏/活动之夜**。
   - 鼓励成员关注这些即将开展活动的更新。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1371925211909459968)** (379 条消息🔥🔥): 

> `File:// 用法、Qwen3 推理速度、Llama 3.2 Vision 微调、mergekit 和 frankenmerge、Qwen3 GRPO notebook` 


- **尝试修复文件使用中的 'NoneType' 错误**：用户在尝试使用 **file://** 加载图像时，对 `AttributeError: 'NoneType' object has no attribute 'startswith'` 进行了调试。怀疑是代码中 *image* 变量最初被设为 **None** 的问题，并建议在路径中添加额外的正斜杠。
   - 经过一番调试后，用户报告称即使修正了路径，错误仍然存在，从而考虑使用 URL 作为替代方案。
- **用户质疑 Qwen3 的模型推理速度极低**：一名用户报告在 A100 上使用 **Qwen/Qwen3-0.6B** 时推理速度较慢，在 Batch Size 为 1 的情况下约为 **30 tokens/second**，并寻求建议这是否属于正常现象。
   - 其他人建议，较小的模型尺寸和 Batch Size 可能会抵消使用 `load_in_8bit=True` 带来的好处。
- **请求协助微调 Llama 3.2 Vision 模型**：一名用户请求协助设置用于微调 **Llama 3.2 Vision 模型**的脚本。
   - 他们表达了对图像微调的挫败感，称这让他们感到“头疼”。
- **关于 GRPOTrainer 中使用熵损失（Entropy Loss）的讨论**：一名用户询问 **GRPOTrainer** 是否实现了熵损失，并引用了一篇发现其有帮助的 [论文](https://arxiv.org/pdf/2504.20571)。
   - 另一名用户分享了一个消融实验表，并指出即使没有熵损失，模型仍然可以收敛，但表现会**差 2-4 个点**，并补充道：“我可以理解为什么在单个示例的情况下，熵损失比在传统规模的数据集中更重要！”
- **关于 mergekit 和 frankenmerge 的讨论**：一名用户请求关于 **mergekit** 和 **frankenmerge** 的入门指南、博客、视频或课程。
   - 有人提到：“Mergekit 显然可以让你将不同的 LLM 合并在一起。”


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1371940228901765231)** (49 messages🔥): 

> `O3 评估, GPT-4.1 编程, Qwen 模型, NEFTune` 


- **O3 模型因编程代码质量差被吐槽**：成员们抱怨 **O3** 输出的代码是垃圾，且 **OpenAI models** 已经变得几乎无法使用；另一位成员认为 **O3** 在编程方面*烂透了*，建议将其用于带有 tool calls 的规划和研究，而编程则使用 **O4 mini high**。
- **GPT-4.1 被宣布为最佳编程模型**：一位成员表示 **GPT 4.1** 是最好的编程模型，拥有 edu 账号的用户可以在 Github Copilot 中使用。
   - 另一位成员同意 **4.1** 比 **Sonnet** 甚至 **O1** 都要好得多。
- **Qwen 模型在基准测试中超越 Gemma**：有人指出 **Qwen 3 models** 优于 **Gemma 3**，在同等规模下击败了所有竞争对手。
   - 特别是 **Qwen 3 8b model**，在经过良好的 fine tuning 后，几乎在任何领域都能达到 **SOTA**，是本地 LLM 的完美平衡点。
- **建议使用 Repo2txt.com 代替 RAG 进行代码注入**：一位成员推荐使用 [repo2txt.com](https://repo2txt.com/) 从 repo 中选择文件并生成文本，然后直接注入到 prompt 中，因为这比让模型进行 RAG 效果更好。
   - 他们声称模型*无法自动读取 GitHub 中的所有代码，并且会犯错*。
- **Qwen Deep Research 发现了 NEFTune**：成员们讨论了 **Qwen's deep research**，它揭示了 **NEFTune**（即在 finetuning 期间在 embedding 中注入噪声），其作用类似于 regularization。
   - 一位成员更倾向于它而不是 **Gemini deep research** 和 **ChatGPT**，因为它对指令非常具有针对性，不会产生幻觉，并向他们介绍了 **NEFTune**。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1371925466394656848)** (83 messages🔥🔥): 

> `词表大小, Chat Templates 与基座模型, Unsloth 性能问题, GGUF 兼容性, GRPO 与 Qwen3` 


- **词表大小防止了 OOV 错误**：一位成员指出词表大小相同且使用 byte-level encoding，因此不存在 **out-of-vocabulary (OOV) 错误** 的可能。
   - 配置中添加了 tool 和 think tokens、chat template，并略微增加了 `model_max_length`。
- **基座模型聊天模板的灵活性**：有人提到在 base model 中，*你可以随心所欲地使用任何模板*，*这完全没关系*，甚至是 **Alpaca** 或 **Gemma** 模板。
   - 但是，你需要坚持使用它，并使用自己的代码将数据包装到模板中。
- **Unsloth 对 Qwen3 的推理速度较慢**：一位用户报告在 **A100** (40GB) 上使用 **Qwen3-0.6B** 时推理速度较慢，使用 Unsloth 的 `FastLanguageModel.from_pretrained()` 在 batch size 为 1 时仅达到约 **30 tokens/second**。
   - 该 base model 的 `tokenizer_config.json` 中不包含 chat template。
- **视觉模型合并修复已推送**：已推送修复程序以解决视觉模型的 `save_pretrained_merged()` 和 `push_to_hub_merged` 问题；请确保使用以下命令从主仓库更新 unsloth-zoo 和 unsloth 安装：`pip install --force-reinstall git+https://github.com/unslothai/unsloth-zoo.git` 和 `pip install --force-reinstall git+https://github.com/unslothai/unsloth.git`。
   - 具体问题在 [pull request](https://github.com/unslothai/unsloth-zoo/pull/135) 中有详细说明。
- **llama.cpp 的新功能需要 mmproj 文件**：在 `llama.cpp` 中使用多模态模型的新功能时，GGUF 模型可能需要 `mmproj` 文件。
   - 根据更新后的 `llama.cpp` 文档，该文件可以通过使用 `llama.cpp` 对模型进行两次转换来创建，一次是正常转换，一次是带有 `--mmproj` 命令行参数的转换。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1372006532677369987)** (5 messages): 

> `Med Palm 2, QLoRA 内存, ModernBERT 上下文长度` 


- **谷歌 Med Palm 2 的 stabgan**：一位成员指出 **stabgan** 与谷歌的 [Med Palm 2 论文](https://arxiv.org/html/2505.07686v1) 所做的有点类似，但他们将该概念用于普通生成而非推理。
   - 他对分享该论文表示了感谢。
- **QLoRA 内存占用保持在较低水平**：一位成员报告使用 **QLoRA** 保持了低内存占用。
   - 他利用这一点来尝试 **ModernBERT** 全部的 **8k** 上下文长度，并使用巨大的 batch sizes，以便在训练期间获得更好、更多样化的批次内负采样（in-batch sampling of negatives）。


  

---

### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1372245921164951784)** (2 条消息): 

> `Safety Evaluations Hub, GPT-4.1, GPT-4.1 Mini` 


- **Safety Evaluations Hub 首次亮相**：OpenAI 推出了 [Safety Evaluations Hub](https://openai.com/safety/evaluations-hub/)，这是一个用于探索其模型安全结果的资源。
   - 虽然系统卡（system cards）在发布时会分享安全指标，但作为主动沟通安全工作的一部分，该 Hub 将定期更新。
- **GPT-4.1 登陆 ChatGPT**：GPT-4.1 是一款擅长 **coding 任务和指令遵循（instruction following）** 的专用模型，现在 Plus、Pro 和 Team 用户可以通过“更多模型”下拉菜单直接在 ChatGPT 中使用。
   - Enterprise 和 Edu 用户将在未来几周内获得访问权限，对于日常 coding 需求，它是 **OpenAI o3 和 o4-mini** 的一个更 **快速** 的替代方案。
- **GPT-4.1 mini 取代 GPT-4o mini**：GPT-4.1 mini 正在为所有用户取代 ChatGPT 中的 GPT-4o mini。
   - **GPT-4.1 和 GPT-4.1 mini** 经过了标准安全评估，详细结果可在新推出的 [Safety Evaluations Hub](https://openai.com/safety/evaluations-hub/) 中查看。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1371952151471067196)** (151 条消息🔥🔥): 

> `Sentient AI conversation, ChatGPT models for coding, O3 model intelligence, ChatGPT Enterprise plan, AI-generated images on Instagram` 


- ****O4-Mini-High** 是 Coding 神器**：成员们对 **O4-mini-high** 的 coding 能力赞不绝口，一位用户惊叹道：*“好久没看到 coding 模型有如此出色且快速的表现了”*，并指出它在短短 **22 秒** 内就解决了一个问题并优化了一个计算器的代码。
   - 尽管一些用户注意到 **GPT-4.1** 是专为 coding 设计的，但普遍观点倾向于认为 **O4-mini-high** 的性能优于 **Claude 3.7 Sonnet** 和 **Gemini 2.5 Pro**。
- ****O3** 无限制使用是关键因素**：一位考虑从 Pro 订阅切换到 Teams 订阅的用户担心失去无限制的 **O3** 使用权，他们认为该模型是 *“解决我大部分问题的传奇模型”*，尽管 Teams 计划提供了令人向往的内部知识库功能。
   - 成员们建议坚持使用 Pro 计划，因为可以享受无限制 **O3** 的好处，该模型现在支持通过个人 **GitHub Repos**、**OneDrive** 和 **SharePoint** 进行深度研究（deep research），且在一个 Chat / Project 中允许上传 **20 个文件**。
- **ChatGPT Enterprise 通过用户数据训练以增强性能**：一位用户分享了他们使用企业版 **ChatGPT** 的经验，该版本最初 *“相当垃圾”*，但在经过数千名用户日常交互的训练后显著改善，现在使用 **GPT-4o** 并提供更安全且无限制的使用。
   - 另一位用户询问了 Enterprise 计划中 **O3** 的消息上限，并澄清虽然该计划使用 **GPT-4 Turbo**，但他们特别感兴趣的是 **O3 模型** 的消息上限，后来发现是 **每位用户每周 100 条消息**。
- ****GPT-4.1 Mini** 取代 **GPT-4o Mini****：社区注意到，截至 **2025 年 5 月 14 日**，**GPT-4.1 Mini** 已在 **ChatGPT** 中为所有用户取代了 **GPT-4o Mini**，据称在指令遵循、coding 和整体智能方面有显著提升。
   - 成员们讨论了免费计划中从 **GPT-4o** 到 **GPT-4.1** 的转变，权衡了各自的优缺点，一些成员认为 **GPT-4o** 在心理疗愈用途上更好，而 **GPT-4.1** 在前端 coding 等其他领域表现出色。


  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1371942266444447805)** (12 条消息🔥): 

> `GPT-4o 用于 Web 应用编程，Azure OpenAI 助手的结构化输出，Node ID 错误，PC 端与移动端的聊天延迟，长输出导致的聊天被标记` 


- **GPT-4o 助力 Web 应用开发**：成员们讨论了使用 **GPT-4o** 编写 Web 应用（**Vue**、**Express.js**、**MongoDB**），强调了指定工具、操作系统、IDE、语言、框架和首选依赖项的必要性。
   - 详细说明需求有助于模型提供预期的解决方案；简单来说就是 *“准确告诉它你想要什么”*。
- **Azure OpenAI 助手面临结构化输出难题**：一位用户报告了在 **Azure OpenAI** 中处理助手的结构化输出（Structured Outputs）时遇到的问题。
   - 另一位用户报告称现在一直收到 *"getNodeByIdOrMessageId - no node found by id: placeholder-request-"* 类型的消息，表明该平台目前存在问题。
- **PC 端出现输入延迟**：一位用户在 PC 上遇到了输入延迟和消息加载缓慢的问题，而同一个聊天会话在相同网络下的手机上运行正常。
   - 他们通过在另一台独立的 Win11 工作电脑上进行测试，进一步隔离了问题，确认该问题是特定于 PC 设置的。
- **被标记的聊天触发崩溃**：一位成员建议，系统可能因为输出极长（可能超过了输出限制）而标记了某个聊天。
   - 该用户推测，*"如果你向它提供超过 5000 行代码的文件，而它在编写修复方案和重构代码时出错，系统可能会标记它，导致整个聊天损坏，无法在任何设备上加载"*。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1371943584835698718)** (70 条消息🔥🔥): 

> `用于编程的 GPTs，PII 数据防护栏，用于宇宙模拟的 AI，用 AI 模仿写作风格，Ollama 对比 Windsurf` 


- **用户寻求使用 GPT-4o 编写 Vue、Express 和 MongoDB 的建议**：一位成员寻求关于使用 **GPT-4o** 进行编程的指导，特别是针对 **Vue**、**Express.js** 和 **MongoDB**，并询问了如何将其与 Visual Studio 集成。
   - 另一位成员建议先通过 **ChatGPT** 学习使用 **HTML**、**CSS** 和 **JavaScript** 的教程，构建计算器、笔记应用或天气应用等基础程序，之后再学习 **TypeScript**、**React**、**Vite** 和 **Electron**。
- **成员讨论 PII 数据防护栏挑战**：一位成员报告称，在他们的应用访问 HR 数据时遇到了 **PII 防护栏**（PII guardrails）问题，导致家庭住址的请求被拦截。
   - 另一位成员建议直接咨询 **OpenAI**，以讨论合规使用方式，并获取在遵守其政策（特别是关于 **PII** 的政策）的前提下处理敏感数据请求的指导。
- **用户构建 AI 宇宙模拟器用于伴随提示词生成**：一位成员正在使用 AI 构建一个 **1:1 的宇宙模拟**，以思考“思考”本身，并在 AI 模型中创建自动化生态系统。
   - 其目标是将模拟扩展到 Web 浏览器中，并推动模型之间的开放通信线路，提升效率而非单纯堆叠模型。
- **用户探索使用 ChatGPT 模仿写作风格**：一位成员询问如何使用 **ChatGPT** 模仿其写作风格，另一位成员建议分享样本并进行迭代引导以优化输出。
   - 有人指出，通用的提示词（Prompt）很可能无法产生高质量的结果，需要一套强大且连贯的规则和约束来实现这一目标，而在 1000 页的内容上训练模型会有所帮助。
- **用户讨论使用 Ollama 进行本地模型推理，而非 Windsurf 等服务**：一位用户质疑 **Ollama** 是否逊于 **Windsurf**，一位成员建议在使用 AI 构建应用时不要为 **Windsurf**、**Lovable** 或 **Replit** 等服务付费。
   - 建议学习提示词工程（Prompting）以避免 API 成本，并结合 **Ollama** 以及 **VS Code** 扩展（如 **Continue** 和 **Roo Code**）使用本地模型推理，同时学习 Hugging Face 的 **LLM** 和 **Agentic** 课程。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1371943584835698718)** (70 条消息🔥🔥): 

> `ChatGPT 用于 Web App 开发，GPT-4o 编程辅助，HR 数据护栏与 PII，模仿写作风格，Prompt Engineering 与 Agentic Frameworks` 


- **利用 ChatGPT 进行 Web App 编程辅助**：一位用户寻求关于利用 **GPT-4o** 进行 Web App 开发的建议，特别是使用 **Vue**、**Express.js** 和 **MongoDB**；另一位成员建议像与真人交谈一样使用它，提供挑战的具体细节，分享代码片段并说明背景。
   - 该成员建议从一个基础原型开始，进行测试，迭代修复错误，并逐个添加功能，确保用户和模型保持同步。
- **规避 HR 数据的 PII 护栏**：一位用户报告了在从连接 HR 数据的应用程序请求家庭住址等信息时遇到的 **PII guardrails** 问题。虽然 OpenAI 支持团队可能会提供定制化指导，但由于 OpenAI 的政策，社区成员在提供具体变通建议方面受到限制。
   - 建议是直接联系 [OpenAI support](https://help.openai.com/en/articles/6614161-how-can-i-contact-support) 直接讨论具体的使用场景，并寻求在应用程序中处理敏感 PII 数据的适当指导。
- **掌握模仿：使用 ChatGPT 模拟写作风格**：一位用户询问了让 **ChatGPT** 模仿其写作风格的最佳方法，一位成员建议提供样本，要求针对特定目标进行模拟，并根据模型的反馈迭代纠正和完善输出。
   - 另一位成员强调了需要指定结构细节和所使用的元素，以便向机器人明确写作风格。
- **Prompt Engineering 助力卓越 AI 效果**：一位成员建议学习基础的 HTML、CSS 和 JavaScript，以便更好地理解和调试 AI 生成的代码，并建议完成 [Hugging Face Learn](https://huggingface.co/learn) 上的 LLM 和 Agentic 课程，以掌握 prompting、上下文管理和角色设定。
   - 他们建议可以通过要求 **ChatGPT** 对 prompt engineering 进行评分并提供反馈来评估自己的 prompting 技能，这是提高效果的关键，并理解 *agentic frameworks* 即 **LLM + MCP servers + modes + prompts** 如何引导其跨多个模式和工具工作。
- **揭秘个性化 ChatGPT 参与元数据**：一位用户请求详细列出其 **ChatGPT usage metadata**，包括消息长度、角色使用情况和对话深度等指标，以深入了解其交互模式。
   - 建议使用追问提示词 *“are there any other stats you can share i had not covered?”* 来生成更多统计数据。 


---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1371926521752588429)** (271 messages🔥🔥): 

> `去中心化系统中的 GPU 算力, Cursor 定价 vs API 定价, Cursor 中的 Claude Max, Cursor 中的多仓库项目, Cursor 的 Git 变更同步问题` 


- ****思考 GPU 算力的 P2P 方案****：一名成员询问了在 [P2P 或去中心化系统](https://www.google.com/search?q=GPU+power+in+decentralized+systems) 中利用 **GPU 算力** 的开源选项。
   - 讨论未得出具体解决方案，凸显了这是一个值得探索的潜在领域。
- ****关于 Cursor 20% 溢价的辩论****：用户讨论了 **Cursor 相比实际 API 定价收取的 20% 附加费** 是否合理。一些人认为这在商业上完全合理，而另一些人则认为性价比不高。
   - 一位用户表示：*“我觉得 Cursor 是目前性价比最高的，考虑到你花 20 美元能得到的东西”*，而另一位用户声称通过在 Cursor 之外使用 **Claude Max** 每月能节省 *$600*。
- ****Gemini 2.5 加入 Cursor 模型组合****：社区发现 [5 月 6 日](https://discord.com/channels/1074847526655643750/1074847527708393565/1371986950498414662) Cursor 的可用模型选择中新增了 **Gemini 2.5 Pro**。
   - 有人指出新模型*终于修复了“我现在就开始写代码（然后停止编写）”的问题*。
- ****单仓库（Mono-Repo）方法技巧****：用户讨论了多仓库项目的管理，有人建议将其合并为单个 Monorepo 以避免 [上下文碎片化（context fragmentation）](https://www.google.com/search?q=ai+model+context+fragmentation)。
   - 另一位用户提到在 Cursor **0.50.x** 中，可以在同一个父文件夹下使用独立的 Workspace。
- ****后台 Agent 请求确认引发不满****：用户对后台 Agent 需要过度确认表示沮丧，这增加了 [Fast Request 使用量](https://www.google.com/search?q=Cursor+fast+requests)。
   - 一位用户抱怨后台 Agent *“一直要求确认操作，这让 Fast Request 消耗得更快了”*。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1371925637073473607)** (197 messages🔥🔥): 

> `AI 专利, RL-Diffusion, 生成器范式, 进化算法, 哈密顿神经网络与 Transformers` 


- **RL-Diffusion 辩论引发专利担忧**：一名成员讨论了开发 **RL-Diffusion** 的想法，即将前向和后向过程合并为一个由 **RL** 控制的过程。这引发了关于专利可行性和现有技术（prior art）的辩论，部分成员对其新颖性和实际实现持怀疑态度。
   - 一些成员鼓励在寻求专利之前先进行实际实现和基准测试，并强调将其转化为符合专利申请要求的应用需要*“不仅仅是简单地陈述 [抽象概念] 并加上‘应用它’这类词汇。”*
- **Google 的 AlphaEvolve 引发兴奋与质疑**：成员们讨论了 **Google 的 AlphaEvolve**，它将 **Gemini 模型** 与进化算法结合以改进底层模型。观点分为两派：有人认为这是有意义的进步，有人则认为这只是*“利用 LLM 进行暴力搜索”*。
   - 一位成员指出其在加速乘法运算方面的潜在意义，而另一位成员将其与 **AlphaTensor** 和 **AlphaCode** 等现有工作联系起来，认为它是神经网络驱动搜索迈出的一小步。
- **哈密顿神经网络与 Transformers 受到关注**：一名成员分享了将 **Transformers** 整合进 **哈密顿神经网络（Hamiltonian Neural Networks）** 的想法，并引用了一篇关于用于物理系统建模的 **HNN-Transformer** 架构的论文 [https://ieeexplore.ieee.org/document/10316909]。
   - 讨论涉及了 Attention 机制是否符合哈密顿系统的历史无关性（history-independent nature），另一位成员建议开发一种能从单一轨迹中学习系统哈密顿动力学的 Transformer。
- **Diffusion vs. Autoregression：连续还是离散？**：成员们讨论了 **Diffusion 模型** 与 **自回归模型（Autoregressive models）** 的根本区别，强调 Diffusion 模型处理连续分布，而自回归模型处理离散的符号序列。
   - 讨论延伸到如何使用 **VQVAE** 将图像转换为离散 Token，以便用于像 **Parti** [https://sites.research.google/parti/] 这样的自回归模型，从而使 Transformer 能够在潜变量（Latents）上运行。


  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1372000093233741834)** (23 messages🔥): 

> `Grade School Math Benchmarks, ML systems rabbit hole, Data Loading and Preprocessing, LLMs are like humans, Model formulates a plan` 


- ****GSM8K Benchmark**：实现近乎完美的准确率**：语言模型已展示出解决数学推理问题的能力，在 **GSM8K** 等小学数学基准测试中达到了近乎完美的准确率，详见[这篇论文](https://ssrn.com/abstract=5250629)和[这篇博客文章](https://physics.allen-zhu.com/part-2-grade-school-math/part-2-1)。
   - 论文探讨了语言模型是真正培养了推理能力，还是仅仅记住了模板。
- **深入探索 ML 系统深坑**：据分享，大约 **65%** 的训练时间花费在数据加载和预处理上，参考了[这篇论文](https://arxiv.org/pdf/2007.06775)。
- ****LLMs**：像人类还是不像？**：一位成员认为某篇关于 **LLMs** 的论文结论可能存在缺陷，因为 *“他们倾向于证明 LLMs 像人类，而不是试图证伪它。”*
   - 该成员认为论文误读了结果，声称 LLMs 在生成解决方案之前会制定计划。
- **模型规避不必要的计算**：一位成员引用论文称 *“模型可以学习生成最短路径的解决方案，几乎总是能避免不必要的计算”*，并辩称 *“这表明模型在生成之前制定了计划，以避免计算任何对于解决底层数学问题并非必需的数值。”*
   - 另一位成员反驳道：*“模型学到的是，不必要的东西是随机噪声，根据定义，这些噪声没有可供模型学习的信号，因此模型会忽略它们”*。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1371933264994111569)** (12 messages🔥): 

> `AI Regulation Ban, AlphaEvolve, Budget Reconciliation Bill` 


- **GOP 阻止 AI 监管十年**：众议院共和党人在 **Budget Reconciliation bill**（预算协调法案）中加入了一项条款，将**禁止所有州和地方政府在 10 年内监管 AI**（[来源](https://arstechnica.com/ai/2025/05/gop-sneaks-decade-long-ai-regulation-ban-into-spending-bill/)）。
   - 该条款由肯塔基州代表 Brett Guthrie 提出，含糊地规定**任何州或地方实体在十年内都不得执行监管 AI 模型或系统的法律**，这可能会影响隐私监管。
- **DeepMind 的 AlphaEvolve 攻克开放性问题**：DeepMind 的 **AlphaEvolve** 是一款由 Gemini 驱动的编码 Agent，能够**演化用于数学和实际应用的算法**，将 LLM 的创造力与自动评估器相结合（[DeepMind 博客](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)）。
   - 该系统在大约 75% 的案例中重新发现了 SOTA 解决方案，并在 20% 的案例中*改进了之前已知的最佳解决方案*，甚至推进了 **kissing number problem**（吻数问题）。


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1372244689604051057)** (5 messages): 

> `New Chatbot Platform, Customization and Models, Image Generation in Chat` 


- **Personality 发布：新型聊天机器人平台出现**：一位成员介绍了 **Personality**，这是一个新的聊天机器人平台，允许用户创建多个角色并进行角色扮演，同时也支持非角色扮演的助手。
   - 该平台旨在提供比 c.ai 等现有解决方案更多的定制化、更少的过滤以及更广泛的模型选择。
- **Personality 平台提供免费图像生成**：该平台的 Playground 地址为 [personality.gg/playground](https://personality.gg/playground)，提供免费的图像生成功能，但需注意该功能并非由 OpenRouter 驱动。
   - 邀请用户访问 [personality.gg](https://personality.gg) 免费试用平台并提供反馈。
- **Personality 平台本周将迎来重大更新**：预计本周将进行重大更新，包括直接在聊天中生成图像的能力以及更好的用户界面。
   - 这旨在增强用户体验并扩展平台功能。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1371932051820773497)** (177 条消息🔥🔥): 

> `OpenAI 推理模型命名、免费 Google 模型、Gemini 速率限制、OpenRouter 与原生 Claude 对比、驯服鸦科动物` 


- **OpenAI 的推理模型命名需要重构**：一位用户询问了 **OpenAI 推理模型**命名不一致的问题，指出某些模型具有推理级别变体（例如 `/openai/o4-mini-high`），而其他模型则没有，并请求在所有模型中[提供一致的推理级别](https://platform.openai.com/docs/models)以辅助评估。
- **免费 Google 模型受到挤压**：用户报告称，尽管拥有额度（credits），但在使用免费 **Google 模型**时仍遇到问题，部分用户确认其速率限制（rate limits）极低。
   - 推荐了 **DeepSeek V3** 等替代方案，同时用户对 [Twitter](https://fxtwitter.com/officiallogank/status/1922357621178200248) 上分享的一项变动表示担忧，即 **Gemini** 可能会移除免费路线。
- **Claude 系统提示词差异解析**：用户注意到，在 **OpenRouter** 上使用 **Claude** 与在 **Anthropic** 原生网站上使用时，其“乐于助人”程度存在差异，这是因为后者使用了复杂的系统提示词（system prompts）。
   - 建议用户手动实现 [system prompt](https://docs.anthropic.com/en/release-notes/system-prompts)，该提示词可在 [GitHub](https://github.com/jujumilk3/leaked-system-prompts/blob/main/anthropic-claude-3.7-sonnet_20250224.md) 上获取，包含约 **16000 个 tokens** 并集成了工具（tools）。
- **与“谋杀鸟”融为一体**：一位用户分享了他们驯服鸦科动物（**乌鸦**和**喜鹊**）的经历，详细介绍了喂养流程和信任建立过程。
   - 他们讲述了乌鸦在被喂食后跟随他们的故事，并期待建立一支“乌鸦大军”，最后平淡地提到他们位于**德国**。
- **“始终使用此密钥”实际上是新功能**：引入了一个新的“始终使用此密钥”（Always use this key）选项，这引起了混淆，因为它最初被误认为是现有的“将此密钥作为回退（fallback）”设置。
   - 该新功能**排他性地**使用指定的密钥，并防止回退到 **OpenRouter**，这与旧的回退设置行为有所不同。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1371937259078684722)** (165 条消息🔥🔥): 

> `Manus 积分未刷新、Manus 最佳使用案例、Manus 邀请码、Manus 退款、Gemini Developer API 的 Function Calling 功能` 


- **用户报告 Manus 每日积分未正常刷新**：多名用户报告其每日 **300 积分**未在 **GMT 00:00** 刷新，可能是由于时区处理问题。
   - 一位用户提到他们的积分在当地时间 **晚上 8:00** 刷新，表明积分刷新时间存在不一致。
- **利用“Bug 邀请码”释放天才**：一位用户声称拥有一个带有 **100 个邀请码**的“Bug 账号”，并分享了大量[邀请链接](https://manus.im/invitation/MFCISJE3F75HD)，引发了关于其来源和用途的讨论。
   - 一些用户推测这些代码来自付费订阅，而另一些用户则质疑它们对现有成员的用途；新用户使用这些代码可获得 **500 积分**。
- **Manus 积分退款困难**：一位用户报告称一次任务失败消耗了 **800 积分**，并对无法从 Manus 获得退款表示沮丧。
   - 其他用户也纷纷表示，即使服务未按预期运行，现在也不再提供退款，有人建议尝试“争议（dispute）”该笔费用。
- **关于 Facebook Marketplace 诈骗的饶舌歌曲**：一位用户请求 Manus 生成一段关于 *Facebook Marketplace 恶意砍价（lowballing）* 的饶舌，使用了 *best price*、*last price* 和 *mates rates* 等俚语。
   - 该用户澄清该请求并非广告，而是涉及编写关于在线市场相关场景和经历的饶舌内容。
- **Beta 测试员炫耀秘密音乐项目**：一位用户提到被接受参加一个与**音乐（Music）**相关的新 Beta 测试，但由于签署了 NDA（保密协议）无法透露细节。
   - 该用户随后推广了其展示 Manus 内容的社交媒体账号（TikTok, Instagram, LinkedIn, Threads, YouTube, & X）。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1372009016175362058)** (2 messages): 

> `torch.compile 性能，layernorm vs rmsnorm` 


- **PyTorch 实现对比**：一位成员建议，在 **GitHub** 上对比每个操作都有独立 kernel 的“基础”实现，可以解释性能提升的原因。
   - 另一位成员确认，在对比 **PyTorch** 的 **layernorm** 和 **rmsnorm** 实现的 `torch.compile` 结果时，发现了类似的结果，并指出它们的性能基本上相同。
- **报告令人惊讶的 GPU Profile**：一位成员提到，一些同事发布了*令人惊讶的 GPU Profile*，引发了进一步调查。
   - 他们计划与同事跟进，以了解导致这些意外结果的潜在因素。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1372313737221902376)** (1 messages): 

> `CUDA 共享缓冲区，PyTorch Tensors，RAPIDSAI/rmm 库` 


- **寻求用于共享缓冲区的更简单的 C++/CUDA 库**：一位成员询问是否有一个简单的 **C++** 库（可能带有 **pybind**），用于演示多个进程对共享 **CUDA** 缓冲区进行读写。
   - 他们还对在其上封装 **PyTorch tensors** 感兴趣，并指出 [RAPIDSAI/rmm](https://github.com/rapidsai/rmm) 对他们的需求来说可能过于庞大。
- **进一步的 CUDA/PyTorch 互操作性**：用户正在寻求关于如何在多个进程之间高效管理共享 **CUDA** 内存的指导。
   - 他们特别感兴趣的是一种能与 **PyTorch tensors** 良好集成的精简方法，可能作为更全面的 **RAPIDSAI/rmm** 库的替代方案。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1371948067846225981)** (2 messages): 

> `PyTorch nightly, at::Tag, needs_exact_strides, C++ 代码, torch.compile` 


- **needs_exact_strides 优于 needs_fixed_stride_order**：成员们讨论到，如果你使用的是 **PyTorch nightly** 版本，`at::Tag::needs_exact_strides` 更好，因为 `needs_fixed_stride_order` 有时会误导。
   - 一位成员提到在阅读代码后认为答案是否定的，而另一位成员表示感谢，并提到移动了 **C++ 代码**中的 `.contiguous` 调用，这样 `torch.compile` 就无法干扰它们。
- **C++ 代码中移动了 Contiguous 调用**：一位开发者移动了 **C++ 代码**中的 `.contiguous` 调用，以防止 `torch.compile` 的干扰。
   - 做出此调整是为了解决一个反复出现的问题，该开发者非常感谢关于在 PyTorch nightly 构建中使用 `needs_exact_strides` 以获得更好步长（stride）处理的建议。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1372077972433010768)** (5 messages): 

> `Kernel 的算术强度，TMA 利用率指标，Triton 性能调试，用于 Kernel 调试的 Nsight Compute` 


- **提出 Kernel 算术强度问题**：一位成员询问了计算 **kernel 算术强度**的最佳方法，以及评估 **Hopper** 和 **Blackwell** 架构中 **TMA 利用率**的指标。
   - 另一位成员建议在 Hopper 上使用 `tma__inst_executed.sum` 来衡量 **TMA**，并引用了 [一则 NVIDIA 论坛帖子](https://forums.developer.nvidia.com/t/inst-executed-and-thread-inst-executed/154021)，同时指出 **Nsight Compute** 有一个内置的 Roofline 工具来估算算术强度。
- **Nsight Systems (nsys) 调试 Triton 性能**：一位成员询问在使用 `nsys` 和 `nsys-ui` 是否是学习 **Triton** 时调试 **GPU 性能**的典型工作流。
   - 另一位成员确认，这是全程序性能分析的典型工作流，特别是在无头服务器上，并推荐使用 **Nsight Compute (ncu) 和 ncu-ui** 来调试特定的 kernel。


  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1372022809957765182)** (10 messages🔥): 

> `Weight Pruning, PTX Instructions for Matrix Load/Store, CohereAI Talk Recording` 


- **为了性能进行权重剪枝？**: 一位成员询问了关于 **权重剪枝 (weight pruning)** 的问题，特别是随机块权重剪枝，并澄清这是通过 program IDs 而不是将权重归零来实现的。
   - 该技术涉及如何使用 PTX 指令高效地加载和存储矩阵。
- **PTX 提升矩阵操作效率！**: 一位成员分享了一篇[博客文章](https://veitner.bearblog.dev/load-and-store-matrices-efficently-with-ptx-instructions/)，详细介绍了如何使用 **PTX 指令**（包括 `ldmatrix` 指令）在 warp 内高效地**加载和存储矩阵**。
   - 他们还链接了[相关代码](https://github.com/simveit/load_and_store)、[PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/contents.html)以及一篇解释 `stmatrix` 指令的 [LinkedIn 帖子](https://www.linkedin.com/posts/simon-veitner-174a681b6_load-and-store-matrices-efficently-with-ptx-activity-7328421815331037185-_CW-?utm_source=share&utm_medium=member_desktop&rcm=ACoAADJYtOgBMOUI4WiWTyiAkroFBP1VujIWeksIn)。
- **CohereAI 演讲幻灯片已发布！**: 一位成员分享了演讲的 [Google Meet 链接](https://meet.google.com/wdk-yipf-zjd?hs=122&authuser=0)，随后分享了[幻灯片](https://docs.google.com/presentation/d/1npBI7iO8iPjrhD00m9Sw2OMP6pyllslxT_BvWqJSbhg/edit?usp=sharing)。
   - 当被问及录像时，另一位成员建议关注 [CohereAI YouTube 频道](https://www.youtube.com/@CohereAI)。


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/)** (1 messages): 

c.3.p.1: 这看起来可能很有趣：https://arxiv.org/abs/2504.09246
  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1371925894305939546)** (47 messages🔥): 

> `AMD MI300, AMD fp8-mm, VectorAdd, Leaderboard Submissions` 


- **MI300 排行榜霸榜**: 针对 **MI300** 上的 `amd-fp8-mm` 排行榜有大量提交，展示了各种性能改进。
   - 提交的成绩从 **162 µs** 到 **26.3 ms** 不等，反映了不同层次的优化水平。
- **AMD 上的混合专家模型 (Mixture of Experts)**: 在 **MI300** 的 `amd-mixture-of-experts` 排行榜上记录了一次成功的提交，耗时 **7574 ms**。
- **T4 上的 VectorAdd**: 一次提交在 **T4** 的 `vectoradd` 排行榜上获得了 **第 8 名**，耗时 **6.41 ms**。
- **刷新多项个人最好成绩**: 几位成员在 `amd-fp8-mm` 排行榜上取得了 **个人最好成绩 (personal bests)**，反映了持续的优化努力。


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1372035446065987715)** (1 messages): 

> `Competition delayed, Ironing out details, Problem #3` 


- **因细节完善导致比赛推迟**: <#1359640791525490768> 竞赛的问题 #3 将推迟几天发布，[原因如下](https://discord.com/channels/1164999443914053664/1359640791525490768)。
   - 团队正在*完善一些细节*以确保问题尽可能有趣，感谢您的耐心等待。
- **正在完善有趣的细节**: 竞赛团队正在花费额外时间完善一些细节，以确保问题尽可能有趣。
   - 讨论中的问题是问题 #3。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1371971829731823656)** (15 messages🔥): 

> `Factorio Genetic Algorithm, Cutting Down Tokens, Nearest buildable tool` 


- **用于蓝图生成的 FactorioGP 遗传算法**: 一位成员计划创建一个遗传算法，根据建筑材料、输入/输出位置和区域限制等特定要求生成 Factorio 蓝图，并发现了一篇关于动态路径规划的遗传编程 [论文](https://arxiv.org/pdf/2102.048711.pdf)。
   - 该算法旨在让 **LLMs** 提供满足这些要求的常量，作为动态工厂设计的工具。
- **减少 Token 节省开支**: 小组注意到，在 **24 个任务**中评估 **6 个模型**（每个运行 **8 次**）花费了约 **$1000**。一位成员建议通过智能上下文提取和 RAG 实现，可能减少 **90%** 的 Token 使用量。
   - RAG 实现可以削减 90% 的 Token 消耗。
- **最近可建造工具 (Nearest Buildable tool) 尚不完善**: 目前的策略使用 *nearest_buildable tool* 来识别放置物品的合适位置，但该工具还*不够完美*，他们可以在 Discord 上创建线程来讨论工作流。
   - 可能会建立定期会议来讨论工作流。


  

---

### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1371935542408777839)** (23 条消息🔥): 

> `Reference Kernel 耗时、Application Timeout 错误、fp8 gemm VGPR 使用情况、Leaderboard 提交问题、HIP Kernel .s 文件访问` 


- **Reference Kernel 获得速度提升**：一个 [pull request](https://github.com/gpu-mode/reference-kernels/pull/31) 已被合并以缓解 Reference 耗时过长的问题，旨在当主 Bot 更新时缩短运行时间。
   - 该更新解决了关于 Reference 实现耗时过长的担忧，特别是当更快的实现需要多次运行以满足终止标准时。
- **Application timeout 证明只是个小插曲**：成员们遇到了间歇性的 Application timeout 错误，通过重试提交暂时解决了该问题。
   - `asm volatile` 语句中的换行符被认为是潜在原因，尽管问题似乎已自行解决。
- **用户寻求 fp8 GEMM VGPR 见解**：一位正在为 **fp8 gemm** 编写 HIP kernel 的成员询问如何确定 **VGPR** 的使用情况。
   - 另一位成员建议使用 [ROCm 的 amd_matrix_instruction_calculator](https://github.com/ROCm/amd_matrix_instruction_calculator) 进行检查。
- **通过 CLI 提交 Leaderboard 失败！**：一位用户报告称，通过命令行界面 (**CLI**) 向 **amd-mixture-of-experts** 提交 Leaderboard 时遇到超时。
   - 通过 Discord 提交可以正常工作，但 CLI 提交始终超时。
- **寻找 HIP Kernel .s 汇编文件**：一位用户寻求获取 HIP kernel 的 `.s` 文件（汇编代码）的方法，并提到使用了 `hipcc` 和 `extra_cuda_cflags`。
   - 一个建议是向 `hipcc` 传递 `-save-temps`，但在执行期间访问该文件被证明很困难；建议在本地编译作为替代方案。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1371946609310695544)** (25 条消息🔥): 

> `CUTLASS 4.0 发布、Python 版 CuTe DSL、MLIR 编译器、PTX 导出、自定义 Kernel 性能` 


- **CUTLASS 4.0 和 CuTe DSL 亮相！**：**CUTLASS 4.0** 和 **CuTe DSL** 现已发布，可通过 `pip install nvidia-cutlass-dsl` 安装，NVIDIA 建议从 [Jupyter notebooks](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks) 开始。
   - 成员们注意到 `nvidia-cutlass-dsl` 的版本是 `0.0.0....`，根据 pypy 显示是大约 2 个月前发布的，因此 *发布过程似乎出了点问题*。
- **CuTe DSL 需要 Python 3.12**：一位用户报告了安装和运行示例时的问题，根据 [文档](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/quick_start.html#quick-start-guide) 的要求，通过使用 **Python 3.12** 解决了该问题。
- **CuTe DSL 实现极速 Kernel 性能**：一位成员用 C++ 实现了一个用于分数范数（fractional norms）的自定义 **CuTe kernel**，在 p=1.0 的 30,000x30,000x1000 问题上达到了 **67ms**，优于 `torch.cdist` 的 **4,000ms**。
   - 在 `sgemm.py` 示例中用自定义实现替换 `cute.gemm` 产生了类似的性能，编译耗时 **0.5 秒**，运行耗时 **62ms**，*比 PyTorch 快了 60 倍*！
- **MLIR 编译器尚未开源（？）**：一位用户询问了从源码构建以及 **MLIR 源码文件** 是否开源的问题，但开发者确认 **Dialect 编译器并非开源软件 (OSS)**。
   - 用户可以通过 pip 安装并直接使用。
- **从 CuTe DSL 导出 PTX**：一位用户询问是否有办法导出生成的 **PTX** 代码，类似于 **Triton 的 MLIR_ENABLE_DUMP**，但目前设置 `CUTE_DSL_PRINT_IR=1` 仅会导出 **MLIR** 文件。
   - 该功能目前尚不存在。


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1372122831185252424)** (2 条消息): 

> `Mojo PyTorch 后端、Autograd 实现、Micrograd、PyTorch 内部机制` 


- **Mojo ❤️ PyTorch 后端？**：一位成员对 Mojo 成为 **PyTorch 后端** 的想法表达了热情，同时也希望代码库能更易于访问，特别是对于那些不太熟悉 **C++** 的人。
   - 他询问了 **Backward Pass** 的实现，特别是关于是否需要独立的 Kernel 以及如何处理 Fusion。
- **Micrograd 作为 PyTorch 的灵感来源**：一位成员提到 **Micrograd** 是基于 **PyTorch** 的，并提供了 [Micrograd 视频](https://www.youtube.com/watch?v=VMj-3S1tku0) 和 [PyTorch 论文](https://arxiv.org/abs/1912.01703) 的链接以供参考。
   - 这表明 **PyTorch autograd** 系统的原理和实现细节可以为 Mojo 如何处理其自身的 Backward Pass 提供见解。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1371935134999117865)** (49 messages🔥): 

> `Gemini 2.5 Pro, Model Performance, Common Lisp, AI Studio, Repomap` 


- **Gemini 2.5 Pro 性能与使用案例**：用户正在通过 OpenRouter 尝试 **Gemini 2.5 Pro**，使用 `--edit-format diff-fenced`，并观察到它有时会为了微小的改动而重写巨大的文件，从而对其行为产生疑问。
   - 一些用户发现 **AI Studio** 提供结果的速度更快，并反馈 **Sonnet 3.7** 最适合他们的工作流，而另一些用户则使用较便宜的模型进行 ask，使用 **Sonnet 3.7** 进行 architect。
- **讨论 Common Lisp 与现代 AI 工具链**：用户正在讨论使用现有模型开发较冷门的语言（如 **Common Lisp**），计划利用书籍和数据源创建数据集和 Prompt，用于 In-context learning。
   - 该想法涉及对**小型模型进行 LoRA 微调**，并使用语义检索将编程书籍的智慧添加到 Context Window 中，一名成员建议创建一个 **Lisp DSL** 来构建编译器/解释器。
- **解决 Google AI Studio 重定向问题**：一名用户报告称，在短暂看到 UI 后被从 **Google AI Studio** 重定向，尽管他们所在的国家在允许名单上，目前正在寻求潜在的解决方案或解释。
- **Repomap 问题与解决方案**：一名用户质疑为什么 **repomap** 有时表现不佳，即使设置了很高的 map multiplier，这表明在某些项目中可能存在文件映射问题。
   - 他们表示“想要得到完美的贪吃蛇并非易事！”
- **Gemini 在代码中添加注释以及后续的愚蠢想法**：一名用户发现 **Gemini** 在代码中添加了过多的注释，甚至在代码内部加入了一些供以后参考的“愚蠢”想法，并且因为它在代码内部而真的把它们写成了代码。
   - 应该有严格的代码变更规则，而不包含编码想法。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1371933938494214286)** (48 messages🔥): 

> `Gemini rate limits, Aider upgrades, Aider models, Aider configuration, Aider file navigation issues` 


- **Gemini 免费层级遇到速率限制**：用户报告称 **Gemini 的免费层级**突然出现速率限制，即使在闲置一段时间后也是如此。一位用户指出*可能他们只是降低了这些集群的优先级，所以当其他地方负载增加时就会返回 429 错误*。
- **Aider 升级困扰依然存在**：用户在升级 **Aider** 时遇到问题，尽管升级过程看起来已成功完成，但版本号未能更新。
   - 升级期间的 SSL 警告可能与之无关，因为自一月份以来这一直是一个反复出现的问题。
- **实验性 Gemini 模型被禁用引发混乱**：用户遇到错误提示**免费实验性 Gemini 模型**已被禁用，导致混乱，并建议切换到 Preview 模型。
   - 一名用户报告了意外扣费，质疑自己是否真的在使用 Preview 模型，后来通过检查启动时的 Aider 公告行确认 **pro-preview 并未被使用**。
- **Aider 获得简洁的澳洲风格聊天外观**：一位用户发现通过修改 `~/.aider.conf.yml` 文件，可以让 **Aider** 的回复更易读。
   - 他们建议使用 `chat-language: English (Australia, use headings, bullet points, concise sentence fragments)`。
- **Aider 在处理大文件导航时遇到困难**：一名用户报告称 **Aider** 在导航一个 **1600 行的文件**时出现问题，在行号定位和调试无关代码方面遇到困难。
   - 建议尝试不同的模型，并考虑 repo map 可能是导致该问题的原因之一。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1371957710958104597)** (22 条消息🔥): 

> `lm-eval-harness dataset download, R1-distill models prompt format, Regulatory bias standards and LLMs, Open Science Conference call for papers, ODSC vs OSC conference confusion` 


- ****LM-Eval-Harness** 简化数据集下载**: 一位用户询问如何在不立即评估模型的情况下，为 **lm-eval-harness** 中的特定任务下载数据集，通过使用 `python3 lm_eval --model dummy --tasks [task_list] --limit 1` 来下载数据集找到了解决方案。
   - [在此定义](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/dummy.py)的 `dummy` 模型用于测试并返回随机数，而 `--limit n` 将评估限制在前 `n` 行。
- **探讨 **R1-distill** 模型提示词格式**: 一位成员询问，对 **R1-distill** 模型使用 *"user: What is xyz? assistant: <think>"* 格式进行提示，是否比直接使用 *"What is xyz? <think>"* 更为常见。
   - 遗憾的是，讨论在此中断。
- **LLMs 面临**偏见**监管审查**: 讨论围绕算法中**偏见标准**的监管展开，特别是在 LLMs 的背景下，引用了 **NCUA, EEOC, FDIC, HHS** 和 **FTC** 等监管机构的例子。
   - 提到了[存档的 EEOC 指南](https://web.archive.org/web/20240123163943/https://www.eeoc.gov/laws/guidance/select-issues-assessing-adverse-impact-software-algorithms-and-artificial)，强调监管机构要求提供非歧视证明，并可能将无法研究的算法视为侵权。
- ****Open Science Conference** 发布征稿通知**: **Open Science Conference** 正在征稿，可能适合跨学科工作，提交截止日期为一周后。
   - 有关征稿的更多详情可以在 [Open Science Conference 网站](https://www.open-science-conference.eu/calls2025/)上找到。
- ****ODSC** 与 **OSC** 会议：避免混淆！**: 会议澄清了 **ODSC** 与 **OSC** 的区别，并指出有许多名称/缩写非常相似的会议，同时警告其中一些可能是通过旧的 Google Groups 传播的诈骗。
   - 一位成员确认 **ODSC** 是正规的（因为他们曾在那里担任演讲者），而 **OSC** 看起来也是正规的，但知名度较低。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1371977243605336134)** (57 条消息🔥🔥): 

> `Model of Mind AI, Falsifiable Hypothesis, Sparse Gradients, Qwen 3, Skywork Model` 


- **模仿人类心智建模 AI**: 一位成员根据心理学模型，模仿**意识、潜意识和无意识**的概念以及更高级别的行为系统为 AI 建模。
   - 一位成员指出，该频道讨论的是*相当狭窄的特定 ML 主题范围*。
- **可证伪假设**: 有人指出，这里不需要学位，但必须*遵循**可证伪假设**或对过程进行数学描述*。
   - 该频道用于讨论研究主题或特定的论文和结果，而不是个人尚未成型的研究想法。
- **Qwen 表现强劲**: 成员们注意到 **Qwen 表现非常出色**，正如[链接图片](https://cdn.discordapp.com/attachments/747850033994662000/1372228411426082876/IMG_7934.jpg?ex=682602cc&is=6824b14c&hm=8c0670673b00b25f0e43eda2f1e9e077b7270e332e1767cb3576cc2ca5650e2e&)所示。
   - 有人提出了一个问题——**Qwen 3** 中控制熵的具体机制是什么。
- **Skywork 的短推理**: 一位成员表示 **Skywork 模型**和技术非常出色，特别是考虑到完整版本刚刚发布，并链接到了 [Skywork Open Reasoner Series](https://capricious-hydrogen-41c.notion.site/Skywork-Open-Reasoner-Series-1d0bc9ae823a80459b46c149e4f51680)。
   - 他们通过训练 Batch 中的总 Token 数而不是按序列进行归一化，这基本上就是 **dr GRPO**。


  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1371948881763373176)** (7 条消息): 

> `Multi-GPU lm-eval, vllm Tensor Parallel` 


- **Multi-GPU lm-eval 利用率问题浮现**：一名成员报告称，在 **lm-eval** 中使用 `parallelize=True` 时，**GPU 0** 的利用率为 **100%**，而 **GPU 1** 的利用率为 **0%**。
   - 另一名成员解释说，`parallelize` 使用的是朴素流水线并行（naive pipeline parallelism），它会拆分模型层，因此一次只能使用一个 rank，并建议使用 `accelerate launch -m lm_eval ...` 来运行多个副本。
- **推荐使用 vllm Tensor Parallel 以保证稳定性**：当其他多 GPU 解决方案失败时，一名成员建议使用 **vllm tensor parallel**，并指出它更可靠。
   - 原帖作者并不知道可以将 vllm 与 lm-eval 结合使用，并表示他们一直使用的是 **HuggingFace** 的实现。


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1371995588245323776)** (2 条消息): 

> `Atropos v0.2.0 Release, Psyche Network Launch, Decentralized AI Training, Large Language Model Training, Open Source AI Development` 


- ****Atropos v0.2.0** 发布并支持 **Axolotl****：Nous Research 发布了其 RL 环境项目 [Atropos 的 v0.2.0 版本](https://t.co/F6hr9JgZpm)，具有新的环境、更新的 API 处理、更好的 TRL 支持，并作为官方训练器合作伙伴与 **Axolotl** 集成，[使用指南在此](https://github.com/axolotl-ai-cloud/plugin-atropos)。
- ****Psyche Network** 启动以实现 AI 训练民主化**：Nous Research 启动了 [Psyche Network](https://nousresearch.com/nous-psyche/)，这是一个去中心化训练网络，旨在通过汇集分布式计算资源来训练大规模模型，从而实现 AI 开发的民主化。
- ****Psyche** 测试网训练 40B 参数 LLM**：**Psyche** 的测试网启动涉及预训练一个使用 **MLA Architecture** 的 **40B** 参数 LLM，数据集包含 **FineWeb (14T)**、**FineWeb-2 (4T)** 和 **The Stack v2 (1T)**。
- ****DisTrO** 优化器突破 **Psyche** 上的带宽限制**：**Psyche** 网络利用 Nous 的 **DisTrO** 优化器和自定义的点对点网络栈来协调全球分布的 GPU，克服了以往 AI 训练中的带宽限制。
- **开源社区驱动 **Psyche** 发展**：Nous Research 鼓励社区通过论坛和 Discord 参与，收集模型创意，旨在促进开源社区内模型创建和设计的创新，代码可在 [GitHub](https://github.com/PsycheFoundation/psyche) 上获取。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1371950355226562621)** (78 条消息🔥🔥): 

> `Frontier Models, smolvlm-realtime-webcam, 3 GPUs, latex2sympy2_extended math_verify, Atropos` 


- **基准测试无法捕捉模型细微差别**：一名成员发现*很难摸清哪些前沿模型在不同任务上表现更好*，因为*基准测试不够细致或多样化*，并分享了[一个链接](https://x.com/timfduffy/status/1921964622589919651?t=7z-QLXzI9BDwm0lGqArgUg&s=19)来说明*“最好”的代码模型在前端、特定框架或数据可视化等方面可能仍然很糟糕*。
- **尝试 smolvlm 实时网络摄像头项目**：一名成员分享了 [smolvlm-realtime-webcam](https://github.com/ngxson/smolvlm-realtime-webcam) 项目的链接。
- **排查 Atropos 依赖问题**：一名成员在运行 `examples/gsm8k_server.py` 文件时遇到问题，该文件需要 `math_verify` 和 `latex2sympy2_extended` 模块，另一名成员建议使用 `pip install latex2sympy2_extended math_verify` 来解决问题。
- **Nous 的 Psyche 是分布式 GPU 训练！**：**Psyche Network** 矿池在 **40 分钟**内就填满了 **500k**，促使有人表示*让 AI 训练对所有人开放似乎确实是个好主意*，一名成员分享了该项目的[链接](https://psyche.network/)。
- **贡献 USDC 用于计算，这属于捐赠**：成员们讨论了捐赠 USDC 为 **Psyche** 项目贡献算力，资金将流向 Nous，一名成员确认*贡献给该池的任何资金纯属捐赠，仅用于测试目的*。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/)** (1 条消息): 

princepolka: 05-06 在指令遵循（instruction-following）方面是否比之前的 2.5 Pro 更差？
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1372280304655663156)** (1 messages): 

> `LLMs in multi-turn conversations, LLM performance degradation, Lost in Conversation paper, Premature Solution Generation by LLMs, LLM Recovery from Conversational Errors` 


- **LLM 在多轮对话中面临挑战**：一位成员分享了 [Lost in Conversation 论文](https://arxiv.org/abs/2505.06120) 及其 [对应的 GitHub 仓库](https://github.com/microsoft/lost_in_conversation)，分析了 **LLM 在多轮对话中** 与单轮设置下的性能对比。
   - 该论文揭示了在多轮场景下的六个生成任务中，**平均性能下降了 39%**，将其归因于*能力的轻微丧失和不可靠性的显著增加*，并得出结论：*当 LLM 在对话中走错一步时，它们就会迷失方向且无法恢复*。
- **LLM 过早尝试解决方案导致不可靠性**：研究表明，**LLM 经常在对话早期做出假设**，并过早地尝试生成最终解决方案，从而导致不可靠。
   - 这种行为表明，**LLM 可能会受益于改进的错误纠正机制**，以便从对话中的错误转向中恢复。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1372170644464341042)** (2 messages): 

> `Finetuning to 1.58 Bits, Cody S Tweet` 


- **WandB 报告：微调至 1.58 Bits**：一份 [WandB 报告](https://wandb.ai/wafers-ai/huggingface/reports/Finetuning-to-1-58-Bits--VmlldzoxMjQ0MDc0MQ?accessToken=qr23x16ivizo2u2axfek5v9z7qo7ovnf4yq4l2en946hzlll1ovif0ajzvif9si) 讨论了 **微调至 1.58 bits**。
   - 该报告可能包含有关实现此类 **低比特微调 (low-bit finetuning)** 的技术和结果的详细信息。
- **Cody S 在 X 上的发布**：[Cody S 在 X 上发布了推文](https://x.com/0xCodyS/status/1922077684948996229)。
   - 在没有更多上下文的情况下，该推文的内容及其与 AI 研究的相关性尚不明确。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1372280304655663156)** (1 messages): 

> `LLMs in Multi-Turn Conversations, Lost In Conversation paper, LLM Unreliability` 


- **LLM 在多轮对话中迷失**：一位成员分享了 [Lost In Conversation 论文](https://arxiv.org/abs/2505.06120) 及其 [GitHub 仓库](https://github.com/microsoft/lost_in_conversation)，研究发现 **LLM 在多轮对话中的表现明显差于** 单轮交互，在六个生成任务中平均性能下降了 **39%**。
- **LLM 倾向于过早尝试解决方案**：该论文对 **200,000+ 模拟对话** 的分析显示，LLM 经常在早期做出假设并过早尝试生成最终解决方案，导致不可靠。
   - 简而言之，作者发现 *当 LLM 在对话中走错一步时，它们就会迷失方向且无法恢复*。


  

---


### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1372268777206452284)** (1 messages): 

> `User Experience studies, Multilingual Audio Overviews, NotebookLM Feedback` 


- **NotebookLM 用户受邀参加 UX 研究**：发布了一个友好提醒，邀请用户 **加入 (opt-in)** 参与 **用户体验 (User Experience) 研究**。
   - NotebookLM 团队目前正在征求有关其 **多语言音频概览 (Audio Overviews)** 功能的反馈。
- **征求多语言音频概览的反馈**：鼓励 NotebookLM 用户通过用户体验研究提供有关 **多语言音频概览 (Audio Overviews)** 的反馈。
   - 该计划旨在改善在 NotebookLM 中使用多语言音频功能的用户的体验。


  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1371945164125241376)** (19 messages🔥): 

> `Invisible Sun TTRPG, Shareability Factor, Google Product Discontinuation, NotebookLM and OneNote Sync, Podcast Feature ToS` 


- **使用 NotebookLM 将 Invisible Sun TTRPG 游戏化**：一位成员一直在自学由 Monte Cook Gaming 开发的新 **TTRPG** 游戏 [*Invisible Sun*](https://www.montecookgames.com/invisible-sun/)，并使用 **NotebookLM** 和 **ChatGPT Projects** 进行规则查询。
   - 他们喜欢 **NotebookLM** 的“可分享性因素（shareability factor）”和清晰的引用，但更倾向于 **ChatGPT** 的语音回顾；他们期待在 Backerkit 上通过新书测试 **NotebookLM** 的见解。
- **XDA Developers 称赞 NotebookLM 的独特功能**：[XDA-developers](https://www.xda-developers.com/features-notebooklm-does-better-than-alternatives/) 发表了一篇文章，介绍了 **NotebookLM** 表现出色的六个用例/功能，引发了用户的共鸣。
   - 另一篇[文章](https://www.xda-developers.com/using-notebooklm-with-onenote-heres-how-it-went/)展示了如何将 **NotebookLM** 与 **OneNote** 结合使用。
- **Google 用户担心 NotebookLM 最终会停用**：一位用户表达了对 **NotebookLM** 可能会在“不方便的时间”被停用的担忧，理由是 **Google** 有停用优秀产品的历史。
   - 另一位用户反驳称，**NotebookLM** 的独特性和潜力使其不太可能被停用，并暗示可能会进行品牌重塑和市场推广。
- **OneNote 同步设想引发讨论**：一位成员提议将 **OneNote** 笔记本链接到 **NotebookLM** 进行同步，以便 **OneNote** 中的更改能实时更新 **NotebookLM** 的源文件。
   - 这一想法引发了关于该功能潜在集成方式和益处的讨论。
- **Podcast 功能使用问题备受关注**：一位用户询问了使用 **NotebookLM** 的 Podcast 功能的服务条款，特别是关于在 **YouTube** 等平台使用音频的规定。
   - 另一位用户建议查看 **T&C**（条款与条件）以获取明确信息，并建议关于准确性的免责声明和原始来源链接非常重要。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1371928604501344277)** (32 messages🔥): 

> `Podcast Length, Audio Upload and Transcription, Account Restrictions on PDF Uploads, Adding Information to System Prompt, Early Access Installation Issues` 


- **通过重复链接增加 Podcast 长度**：一位用户询问如何增加 NLM 中 Podcast 的长度，一位成员建议针对同一主题添加多个链接或文档（即使是重复的），以将 Podcast 延长至 **22 分钟**。
   - 目前尚不确定该策略是否适用于所有人。
- **音频上传转录；AI Studio 增强字幕功能**：一位成员建议上传音频作为转录源。
   - 另一位成员推荐在 **AI Studio** 上使用 **2.5 flash** 来生成带有时间戳的字幕。
- **PDF 上传限制困扰部分用户**：多位用户遇到账号限制，导致无法上传 **PDF**。
   - 讨论中尚未给出该问题的解决方案。
- **System Prompt 定制渴望“人情味”**：一位用户对播音员使用过多的字母缩写来指代事物表示不满，并寻求一种向 **System Prompt** 添加信息的方法，使其更倾向于使用人类语言。
   - 讨论期间未提供解决方案。
- **NotebookLM Beta 安装故障**：一位用户报告在收到应用的 **“早期访问（early access）”通知**后，一直卡在“正在安装”界面。
   - 讨论涉及了用户的所在地区，但未提供具体的修复方案。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1372004784961359922)** (41 条消息🔥): 

> `GPT-4 发布, ChatGPT 扩展, AI 入驻创始人, 俄亥俄州法院中的 AI, AlphaEvolve` 


- **OAI 发布回顾：个人观察**：一位成员分享了来自 [andrewmayne.com](https://andrewmayne.com/2025/05/02/some-personal-observations-about-the-launch-of-gpt-4/) 的关于 **OpenAI 发布** 的一些 *非常温馨的故事*。
- **扩展 ChatGPT：构建与发布**：社区分享了一篇由 **Pragmatic Engineer** 撰写的时事通讯文章链接，题为 [构建、发布与扩展 ChatGPT](https://newsletter.pragmaticengineer.com/p/chatgpt-images)。
   - 文章回顾了 **ChatGPT** 发布的历史和技术栈。
- **入驻创始人 (Founder In Residence)：AI 版**：一位成员询问了专注于 **AI** 的 **Founder in Residence 项目**，寻求关于如何定位自己的建议。
   - 他们在 **Amazon 广告** 的 **分析用例** 中有构建 **AI 系统** 的经验，并希望在相同的分析领域构建 **Self-Serve Agents**。
- **Gemini 助力 AlphaEvolve 进行算法设计**：**Google DeepMind** 推出了 [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)，这是一个由 **Gemini** 驱动的 **coding agent**，旨在创建高级算法。
- **Turbopuffer 进入正式商用 (GA)**：**Turbopuffer** [宣布](https://x.com/turbopuffer/status/1922658719231562151) 他们已 **GA** (Generally Available)。


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1372273132093571132)** (3 条消息): 

> `Tom Yeh, Llama 1/2/3/4, LLM Paper Club` 


- **Tom Yeh 教授将讲解 Llama 1/2/3/4**：Tom Yeh 教授将在一次 [特别活动](https://lu.ma/se1f2bfk) 中讲解 **Llama 1/2/3/4 的演进**。
   - 该活动由社区的一位成员组织。
- **LLM Paper Club 通知频道**：一位成员引导用户查看左上角的 **Channels & Roles**，以便在 [LLM Paper Club](https://lu.ma/se1f2bfk) 的通知中被标记相关角色。
   - 未提供更多细节。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1372040104088178698)** (15 条消息🔥): 

> `Qwen 模型蒸馏, MiniCPM-V-2_6, 感知器可视化工具, 本地 Stable Diffusion 托管, 使用 Smolagents 部署 Langfuse` 


- ****Qwen 的精髓**：知识蒸馏？**：一位成员询问了关于 **Qwen** 系列模型 **蒸馏 (distillation)** 的 Notebook 或参考资料。
   - 在提供的上下文中没有直接分享相关资源。
- ****MiniCPM-V-2_6**：热门模型？**：一位成员询问是否有人尝试过使用 **openbmb/MiniCPM-V-2_6**，并指出它目前非常热门且下载量很高。
   - 在给定的上下文中没有提供回复。
- ****向量可视化**：感知器可视化工具带来乐趣！**：一位成员分享了一个用于教学目的的 **感知器可视化工具 (perceptron visualizer)**，如附带的视频 [My_Video_0079.mp4](https://cdn.discordapp.com/attachments/879548962464493622/1372054347214487692/My_Video_0079.mp4?ex=6826096f&is=6824b7ef&hm=ab7441c0985405a8eed58ca0326e1c1c2ca6678d723994c6540118d4f6dea15a&) 和 [My_Video_0080.mp4](https://cdn.discordapp.com/attachments/879548962464493622/1372054347562745896/My_Video_0080.mp4?ex=6826096f&is=6824b7ef&hm=65046c2cb4ad3b7724e812b12f8098ae1c181a7d4474bbcfbe89525a299a0272&) 所示。
   - 另一位成员随后分享了另一个来自 [darkspark.dev](https://darkspark.dev/) 的可视化工具。
- ****本地运行 Stable Diffusion**：打造你自己的图像！**：几位成员询问了如何本地托管 **Stable Diffusion**。
   - 建议结合使用 **Diffusers** 和 **TGI**，或者使用 **WebUI Forge** ([GitHub 链接](https://github.com/lllyasviel/stable-diffusion-webui-forge)) 或 **reForge** ([GitHub 链接](https://github.com/Panchovix/stable-diffusion-webui-reForge))；同时还分享了 **Diffusers 文档** ([huggingface.co](https://huggingface.co/docs/diffusers/main/en/stable_diffusion), [huggingface.co/learn](https://huggingface.co/learn/diffusion-course/unit0/1)) 的链接。
- ****Langfuse 本地启动**：遥测之舞！**：一位成员请求帮助，以使本地 **Langfuse** 部署能与 **smolagents** 配合使用。
   - 他们被引导至 [专用频道](https://discord.com/channels/879548962464493619/1019883044724822016)，并被建议从官方文档获取 **docker-compose.yml** 并使用 **opentelemetry-sdk**。


  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1372087475043831850)** (4 条消息): 

> `提供协助, Hugging Face Transformers, EleutherAI 建议, MIT 扩散模型课程` 


- **AI 工程师自荐专业知识**：一位 AI 工程师表示愿意为有趣的项目提供协助，特别是针对正在撰写论文的研究人员或教授，重点关注 **LLM 研究**和**强化学习**相关领域。
   - 该工程师乐于贡献任何力量——从头脑风暴和编码到实验、实现，甚至是文书工作或调试等枯燥的任务。
- **询问 Transformer 熟悉程度**：一名成员询问该工程师是否熟悉 **Hugging Face Transformers**。
   - 另一名成员建议关注 **EleutherAI**。
- **推荐 MIT 扩散模型课程**：一名成员询问了该工程师最近关注的研究论文。
   - 该成员分享了专注于图像生成的 [MIT 扩散模型课程](https://diffusion.csail.mit.edu/)。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1372047809678803034)** (7 条消息): 

> `pdf2tex 对比 12GB RAM, PDF 格式批判, Markdown 输出建议, Civitai 审查` 


- ****pdf2tex** 的 RAM 占用令人印象深刻！**：一位用户注意到 **pdf2tex** 在使用 **OpenCV** 自动检测和提取图表时仅消耗 **1GB** RAM，而另一个使用并行处理的项目则需要 **12GB** RAM。
- **PDF：存在的祸根！**：用户表达了对 **PDF 格式**的强烈厌恶，有人称其为“见过最糟糕的格式”，另一人则开玩笑说将 PDF 称为一种“格式”太牵强了。
   - 一位用户在内存中将 PDF 转换为 **TGA 或 BMP** 以简化处理，并表达了对 **pdfToSrc** 解决方案的渴望。
- **提议 Markdown 输出！**：一位用户建议增加 **Markdown** 输出选项，以改善 **RAG 摄取**的语义关系。
   - 开发者采纳了这一建议，但指出虽然 Markdown 比纯文本更好，但可能无法完全解决 embedders 的分类问题，特别是针对表格。
- **Civitai 审查名人内容！**：一位用户报告称 **Civitai** 已屏蔽所有名人内容，引发了对审查制度的担忧。
   - 他们链接了一个 [Civitai 模型](https://civitai.com/models/76294?modelVersionId=738488)，并分享了 Jeri Ryan (Seven of Nine) 关于使用 AI 生成裸照的语录。


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1372320106654466191)** (3 条消息): 

> `基于模拟的推理 (Simulation-Based Inference), AI 读书会环节` 


- **读书会探讨基于模拟的推理在决策建模中的应用**：AI 读书会环节将讨论使用 **Simulation-Based Inference** 进行决策建模，这与理解人类行为相关。
   - 一篇 [Medium 文章](https://medium.com/@imanragabm/unlocking-the-secrets-of-decision-making-3d8e7d35197c)提供了关于该论文的更多信息。
- **读书会保持固定时间安排**：AI 读书会环节将在 **PDT 上午 9 点 / EDT 中午 12 点 / CEST 下午 6 点**举行。
   - 这与之前的会议时间保持一致。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1372099757673283635)** (3 条消息): 

> `情感检测的局限性, Transformers tokenizer 上下文长度` 


- **情感检测面临基准测试质量问题**：由于**基准测试质量低下**，情感检测效果并不理想，因为学者们很难定义他们想要预测的内容，且 encoder 模型倾向于学习启发式规律。
   - 这主要是由于难以在**金标准标签 (gold standard labels)**上达成一致造成的。
- **Transformers tokenizers 受限于上下文长度**：根据 [Transformers tokenizer 文档](https://huggingface.co/docs/transformers/en/pad_truncation)，所有模型都有上下文长度限制。
   - 如果向模型传递过多的上下文，将会导致错误。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1372101090509717615)** (2 条消息): 

> `Agent 受阻站点, Smolagents 框架` 


- **Agent 在被屏蔽的站点上浪费时间**：一位用户报告称其 Agent 在访问被屏蔽的站点 (**universetoday.com**) 时浪费了时间。
   - 给出的消息中未提供解决方案。
- **Smolagents 框架效果极差**：一位用户报告在使用 **smolagents 框架** 搭配 **Qwen** 以及课程中提供的一些工具（如 Google Search）时，
   - 抱怨结果“非常糟糕”。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1371939762398560266)** (10 messages🔥): 

> `HF Inference Provider Credits, HF SPACE_ID and SPACE_HOST ENV vars, Unit 1 code execution, InferenceClient Model Selection, Llama models text_generation` 


- **额度紧缺提示：作业提交解决方案**：一位用户询问在本地使用 **Ollama** 开发 Unit 4 时，如果超出了 **HF Inference Provider** 每月包含的免费额度，该如何提交最终作业。
   - 另一位用户建议将 **HF SPACE_ID** 和 **SPACE_HOST** 添加为环境变量并在本地运行应用。
- **Unit 1 的主场：代码在哪里运行？**：一位用户询问 Unit 1 的代码应该在哪里运行，特别提到了 **HF Space 复制**（duplication）。
   - 另一位用户建议使用 **Google Colab**。
- **InferenceClient 模型选择引发教皇年龄困惑**：一位用户报告称，在 Space 中运行 Unit 1 的 Agent 时，在未做任何更改并输入 "hello" 的情况下，Agent 尝试计算教皇的年龄。
   - 他还分享了尝试使用 `client = InferenceClient("meta-llama/Llama-3.3-70B-Instruct")` 的情况。
- **Llama 领域的文本生成问题**：一位用户建议使用此模型 `client = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")`。
   - 这是因为 **text_generation 函数** 在任何 **Llama 模型** 中都不受支持。


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1371926962528063538)** (40 messages🔥): 

> `Typescript vs Authpython Lag, Debugging MCP servers on Smithery, Scalable MCP with Streamable HTTP, User Confirmation for AI Agent MCP Tools, Revolutionary idea for MCP Security` 


- **Authpython 滞后于 Typescript API**：有成员提到，在 API 更新方面，**Authpython** 通常比 **Typescript** 滞后约 **1-2 个月**。
   - 一位成员建议查看特定频道中的示例，并提供了一个 [Go-MCP 客户端链接](https://socket.dev/go/package/github.com/blankloggia/go-mcp?section=files&version=v0.0.0-20250410110046-894efeed42d9&path=client.go)。
- **调试部署在 Smithery 上的 MCP 服务器**：一位用户就如何调试在 **Smithery** 上运行的 **MCP 服务器** 寻求建议，因为在 **Claude Desktop** 中遇到了错误。
   - 另一位成员建议使用 [ithena-cli 工具](https://github.com/ithena-one/ithena-cli) 来存储所有输入和输出以进行调试，只需在运行命令前加上前缀即可。
- **MCP 服务器转向 Streamable HTTP？**：一位用户询问是否可以使用 **streamable HTTP** 代替 **stdio** 来提高 **MCP 服务器** 的可扩展性，并指出大多数开源服务器都使用 **stdio**。
   - 他们不确定是否需要将每个开源 MCP 服务器从 **stdio** 重新配置为使用 **HTTP streamable**。
- **AI Agent 现在请求用户确认**：一位用户询问如何确保他们的 **AI Agent** 在通过 **MCP 工具** 触发更新之前显式请求用户确认，类似于 **Claude Desktop**。
   - **fast-agent** 的作者插话指出，有一个 *pre_tool_call check* 钩子可以用来添加审批流，类似于现有的 *human input tool*。
- **MCP Inspector 图片在 Claude 中消失？**：一位用户报告称，虽然在调用工具后图片在 **MCP Inspector** 的资源部分可见，但 **Claude Desktop** 的资源部分却不显示图片。
   - 另一位用户澄清说，**Claude** 仅在工具响应视图中显示图片。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1372084146041786489)** (3 messages): 

> `Yarr MCP Servers, Tiny Agents Remote MCP Support, LLM-provider-agnostic, MCP enabled Chat Client` 


- **GitHub 上的 Yarr MCP 服务器**：一位成员分享了一个 [GitHub 仓库链接](https://github.com/jmagar/yarr-mcp)，其中包含多个 **ARRs MCP 服务器**。
   - 另一位成员还分享了一个 [X (原 Twitter)](https://x.com/llmindsetuk/status/1922309194696335433) 链接，进一步讨论了这一话题。
- **Tiny Agents 获得远程 MCP 支持**：**Hugging Face Tiny Agents** 现在支持远程 **MCP**，并可以从命令行连接到 **SSE** 和 **Streaming HTTP 服务器**。
   - Tiny Agents 为 Agent 的开发和管理提供了一种通用的方法。
- **新的 Web 托管聊天客户端已启用 MCP**：一位成员介绍了一个新的 **LLM 提供商无关**、**启用 MCP** 的 Web 托管聊天客户端，该项目在 [chatter](https://github.com/sakalys/chatter) 开源，并托管在 [moopoint.io](https://moopoint.io/)。
   - 该客户端旨在通过 Web 界面替代 Claude Desktop 来与 **LLM 提供商** 和 **MCP 服务器** 交互，即将推出的功能包括免费层级、记忆功能、MCP 服务器托管、图片处理、文件上传和语音交互。


  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1372309693862772807)** (5 messages): 

> `Custom Torchtune Models with vLLM, Synchronous GRPO recipe with vLLM` 


- **Torchtune 模型的 vLLM 之旅**：一位成员确认在其内部版本的 **GRPO** 中使用 **vLLM** 运行了自定义 **Torchtune 模型**。
   - 在被问及如何为模型启用 **vLLM** 支持后，他们暗示可能会公开其实现。
- **vLLM 集成实现同步**：一位成员提议创建一个带有 **vLLM** 的同步 **GRPO recipe**，并建议同步和异步版本都应该存在。
   - 他们表达了使用 **vLLM** 版本的强烈偏好，并表示*真的找不到任何不使用的理由*。


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1371929380980396092)** (37 messages🔥): 

> `HFModelTokenizer vs GemmaTokenizer, Gemma PromptTemplate, Tokenizer configurations, Masking assistant tokens` 


- **Gemma Tokenizer 与 HFModelTokenizer 存在差异**：一位成员报告称，使用 **Gemma chat template** 的 **HFModelTokenizer** 生成的输出 token 与 **torchtune GemmaTokenizer** 的 token 不匹配。
   - 这种差异表明 torchtune 的 GemmaTokenizer 可能没有正确应用 chat template。
- **缺失 Gemma PromptTemplate，Alpaca 来救场？**：有人注意到目前没有针对 Gemma 的特定 **PromptTemplate**，这导致了错误的 tokenization 以及 `system` 角色可能出现的问题。
   - 默认情况可能是使用 **Alpaca template**，但拥有一个正确的 Gemma 专用模板至关重要。
- **继承自 HF/Google 配置的多 BOS token 错误**：由于配置中同时设置了 `"add_bos_token": true` 且 chat template 中也包含 **BOS token**，**HF tokenizer** 正在添加多个起始序列 (**BOS**) token。
   - 这个问题继承自 **HF/Google 的 tokenizer 配置**，使得该实现在技术上是“正确”的，但在功能上是有缺陷的。
- **探索用于掩码 Assistant Token 的 Jinja 技巧迷宫**：围绕掩码展开了讨论，特别是 **Hugging Face** 如何提供返回 **assistant mask** 的选项。
   - 对话强调了维护掩码过程的复杂性，潜在的解决方案涉及 **Jinja 技巧**。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1372093691711193198)** (25 messages🔥): 

> `Variant bug with SIMD, register_passable types, Mojo in Google Colab` 


- **发现导致 SIMD 段错误的 Variant Bug**：一位用户报告了在 Mojo 中对 `SIMD` 类型使用 `Variant` 时发生的崩溃，具体表现为在使用 `Variant[T](simd)` 时，打印语句之间发生了段错误（segfault）；该问题似乎与 `Variant` 内部的**空间分配不足**或**生命周期问题**有关。
   - 用户提供了一个最小可复现示例，可以在 [GitHub issue 4578](https://github.com/modular/modular/issues/4578) 中找到，同时还有其他代码片段展示了该 bug 的不稳定行为，包括打印语句的位置会影响崩溃的发生。
- **对 register_passable 类型的担忧**：有人对在 Mojo 中使用超过系统寄存器大小的 `register_passable` 类型表示担忧，因为 **LLVM** 无法很好地处理这种情况，可能会导致误编译。
   - 有人建议，对于 `sizeof[T]()` 大于系统中任何寄存器的可寄存器传递类型 `T`，目前的 `Variant` 实现可能存在缺陷，应该用各种版本的 `Trivial` 来替换。
- **Colab Mojo 集成上线！**：现在通过新的导入方式 `import max.support.notebook` 可以更轻松地在 Colab notebook 单元格中编译和运行 Mojo 代码，它提供了一个 `%%mojo` 魔法命令。
   - 该[公告](https://forum.modular.com/t/max-can-now-be-used-in-google-colab/1383?u=bradlarson)已发布在 Modular 论坛上。


  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1372031878139482174)** (15 messages🔥): 

> `WebGPU bug, BEAM 参数, tinybox-ui, 高性能 blake3 实现` 


- **WebGPU 后端受 bug 困扰**：根据[这条消息](https://x.com/charliermarsh/status/1922333022658978089)，生成的 kernel 不包含连续的 **DEFINE_GLOBAL** 参数，但 `bufs_from_lin` 假设 **DEFINE_GLOBAL** 拥有连续参数。
   - 据称 *Claude* 成功修复了该问题。
- **BEAM 参数影响 WebGPU 性能**：将 **BEAM** 设置为任何值都会导致 WebGPU 后端性能下降；在无 beam 时运行时间为 **30ms**，而 **BEAM=1** 时为 **150ms**。
   - 在 **BEAM=2** 时运行时间为 **100ms**。
- **极简 Tinybox UI 概念出现**：一位用户为 tinybox 构建了一个*极简* UI 概念，具有*无登录、无云端、无冗余*的特点，专注于为接触硬件的人员提供快速的本地控制，项目地址在[这里](https://github.com/dahsmartgirl/tinybox-ui)。
   - 有人指出 tinybox 通常支持 HTTP 设置页面，但前提是必须满足 **0 依赖**且*代码行数绝对最少*。
- **用于张量存储的 Blake3**：目前有一项悬赏，征集高性能 **blake3** 实现，用于云端的内容寻址张量存储。
   - 因此，该实现应该是通用型的，或者根据某位用户的说法是“某些类似的东西”。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/)** (1 messages): 

cookiecrumbs3808: 或者我猜是卸载到 CPU。
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1371969368270966845)** (3 messages): 

> `LlamaIndex Memory 组件, LlamaExtract 引用实现` 


- **LlamaIndex Memory 组件增强 AI Agent**：LlamaIndex 引入了新的 **Memory 组件**，通过短期和长期记忆能力增强 AI Agent，以实现上下文感知对话。
   - 新的 **Memory 组件**允许开发者向其聊天机器人 Agent 实现静态记忆块（[链接](https://t.co/CDOB3UUO4W)）。
- **LlamaExtract 获得引用和推理功能**：@tuanacelik 发布的新代码演示展示了如何在 **LlamaExtract** 中实现**引用和推理**。
   - 该演示详细说明了如何定义自定义 schema，以指示 LLM 从复杂数据源中提取哪些内容（[链接](https://t.co/z4RSGJ5gKI)）。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1372199776090853509)** (6 messages): 

> `LlamaIndex Memory 组件, Memory 会话管理, Memory 的数据库集成, 上下文序列化 vs. 数据库, Memory vs Redis` 


- **Workflow 中的 Memory 组件难题**：一位用户在 LlamaIndex workflow 中使用新 **Memory 组件**时遇到问题，指出在使用 `user_id` 设置 `session_id` 时，每次 workflow 调用时 memory 都是空的。
   - 该用户还询问了 **Redis 集成**与 Memory 组件的结合。
- **Memory 默认使用内存数据库，但为了可扩展性建议连接数据库**：默认情况下，`Memory` 组件使用 **内存 SQLite 数据库**，但可以通过更改数据库 URI 配置为使用本地 SQLite 数据库或 PostgreSQL 数据库。
   - 对于庞大的聊天历史记录，为了可扩展性，建议使用数据库，而不是通过 `memory.to_dict()` 序列化为 JSON blob。
- **聊天记录的上下文序列化 vs. 数据库**：一位用户质疑使用数据库连接与序列化上下文相比有何优势，因为恢复上下文也会恢复聊天记录。
   - 回复澄清说，默认情况下序列化上下文是可以的，但对于**大型聊天历史记录**或需要结构化方式保存历史记录时，使用**数据库更佳**，此外 [Python 字典 vs Redis 是同样的问题](https://redis.com/)。


  

---


### **Cohere ▷ #[💬-general](https://discord.com/channels/954421988141711382/954421988783444043/1372092363941347339)** (3 messages): 

> `生成参数, Cohere 的使用场景, Cohere vs ChatGPT 和 Anthropic` 


- **征求生成参数指导**：一位成员询问关于 **Command aCurious** 建议的生成参数指导。
- **对 Cohere 使用场景的兴趣**：其他人对 **Cohere** 相对于 **ChatGPT** 和 **Anthropic** 等其他模型的具体使用场景感到好奇。


  

---

### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1372254858647965706)** (5 条消息): 

> `Cohere API Calls, Cohere Billing, Cohere Trial Key` 


- **Cohere 用户询问 API 调用次数**：一位 Cohere 用户询问如何查看已进行的 **API calls** 数量。
   - 另一位用户提供了 [billing dashboard](https://dashboard.cohere.com/billing) 的链接。
- **Cohere Trial Key 不显示 API 调用次数**：一位 Cohere 用户表示 **trial key** 只显示 tokens，而不显示已进行的 **API calls** 数量。
   - 他们补充道：*我不认为系统有在统计原始请求数量*。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1372158708406161468)** (2 条消息): 

> `Course certificate requirements, Medium article or X post for certificate` 


- **Medium 文章或 X 帖子可解锁课程证书**：成员们澄清，获得课程证书需要撰写一篇 **Medium article** 或发布一条 **X post** 来总结其中一门讲座。
   - 有兴趣的成员必须通过 [此表单](https://forms.gle/399VqBP88WL1AJ4J6) 提交他们的作品以获得学分。
- **提交课程作业以获取证书**：要获得证书，在完成 Medium 文章或 X 帖子后，必须通过提供的 **Google Forms 链接** 提交课程作业。
   - 提交操作可确保作品被正确计入课程证书学分。