---
companies:
- anthropic
- booking.com
- perplexity-ai
- langchain
- claude
- scaling01
- deepseek
- qwen
- prefect
date: '2025-11-26T05:44:39.731046Z'
description: '以下是该文本的中文翻译：


  **Anthropic** 推出了用于长期运行工作流的持久化智能体（durable agents）和 MCP 任务，并提供了实用的工程模式以及与 Prefect
  等工具的集成。**Booking.com** 部署了一个大规模智能体系统，利用 LangGraph、Kubernetes、GPT-4 Mini 和 Weaviate
  提升了客户满意度。**Perplexity** 推出了用户级记忆和虚拟试穿功能。**Claude Opus 4.5** 在 LisanBench 和 Code
  Arena WebDev 基准测试中处于领先地位；尽管社区对其“思考”和“非思考”模式的反馈褒贬不一，但它通过批处理 API 和上下文压缩提升了成本效益和用户体验。多智能体系统研究显示，**LatentMAS**
  使用 Qwen3 模型将通信 Token 减少了 70-84% 并提高了准确率；同时，推理轨迹蒸馏在保持准确性的前提下实现了显著的 Token 缩减，突显了推理轨迹风格的重要性。'
id: MjAyNS0x
models:
- claude-opus-4.5
- qwen-3-4b
- qwen-3-8b
- qwen-3-14b
- deepseek-r1
people:
- jeremyphoward
- alexalbert__
- omarsar0
- lingyang_pu
- dair_ai
title: 今天没发生什么事。
topics:
- agent-systems
- multi-agent-systems
- reasoning
- benchmarking
- cost-efficiency
- model-optimization
- long-context
- memory-management
- reinforcement-learning
- model-performance
- multi-agent-communication
- latent-representation
- inference-cost
- software-integration
---

**感恩节快乐！**

> AI 新闻摘要（2025/11/25-11/26）。我们为您检查了 12 个 Subreddit、544 个 Twitter 账号和 24 个 Discord 社区（205 个频道，9014 条消息）。预计节省阅读时间（以 200wpm 计算）：713 分钟。我们的新网站现已上线，支持全元数据搜索，并以精美的氛围感编码（vibe coded）方式呈现所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上向我们提供反馈！

我们正在进行 [2025 开发者写作静修营 (Dev Writers Retreat)](https://luma.com/dwr2025) 的最后一轮报名。**NeurIPS 结束后，在圣地亚哥加入我们吧！**

---

# AI Twitter 回顾

**Agent 系统：长时运行框架、MCP 任务处理及生产部署**

- **Anthropic 论持久化 Agent + MCP 任务**：Anthropic 在一篇高质量的工程博客中概述了适用于跨多个上下文窗口的 Agent 实用模式（状态检查点、结构化 Artifacts、确定性工具、“计划模式”等）([博客摘要](https://twitter.com/AnthropicAI/status/1993733817849303409))。与此同时，MCP 发布了 SEP-1686 “任务”，用于支持带有状态轮询和结果检索的后台长时运行工作——这正是长达数小时的研究/自动化工作流所需要的 ([公告](https://twitter.com/AAAzzam/status/1993495222035399060)，[fastmcp + Prefect 集成](https://twitter.com/AAAzzam/status/1993495232881869138))。LangChain 明确了技术栈：框架（构建）、运行时（持久化执行、流式传输/HITL）和挂载工具（通用 Agent），其中 LangGraph 处于运行时层级 ([帖子](https://twitter.com/LangChainAI/status/1993746547587338508))。
- **现实世界的 Agent 基础设施**：[Booking.com](http://booking.com/) 在生产环境中上线了一个 Agent，每天处理数万条合作伙伴与住客之间的消息，据报道满意度提升了约 70%，后续跟进减少，响应速度更快。技术栈包括：LangGraph、Kubernetes、FastAPI、通过内部网关调用的 GPT-4 Mini（具备提示词注入检测功能），以及用于语义模板搜索的 Weaviate（MiniLM 嵌入、KNN + 阈值处理、Kafka 流式更新）([深度解析](https://twitter.com/victorialslocum/status/1993636038313443826))。Perplexity 在不同模型和模式中增加了用户级的“记忆（Memory）”功能（支持查看/删除/禁用；无痕模式除外），并推出了购物“虚拟试穿”功能 ([Memory](https://twitter.com/perplexity_ai/status/1993733900540235919)，[详情](https://twitter.com/AravSrinivas/status/1993733947474301135)，[试穿](https://twitter.com/perplexity_ai/status/1993760113988170165))。

**Claude Opus 4.5：评估、成本/UX 经验及新技能**

- **性能概况**：在 LisanBench 上，Opus 4.5 Thinking 排名第一；而非思考（non-thinking）版本表现不如之前的 Opus 版本及同类模型（在 50 个单词中仅有 18 个最长有效链；由于自我修正较慢，有效率较低）([结果](https://twitter.com/scaling01/status/1993712295118057861))。在 Code Arena WebDev 上，Opus-4.5 (thinking-32k) 首次亮相即位居榜首，险胜 Gemini 3 Pro；在文本（Text）方面排名第三 ([排行榜](https://twitter.com/arena/status/1993750702179676650))。社区反馈褒贬不一：在“不思考”模式下，Opus 4.5 可能比 Sonnet 更差，有时会错误地将 Python 工具用作隐蔽的思维链（CoT）草稿纸并陷入循环 ([分析](https://twitter.com/jeremyphoward/status/1993543631266025623)，[失败模式](https://twitter.com/GregHBurnham/status/1993682288349962592))。
- **成本与易用性**：Batch API 使得“Thinking”运行在价格上变得可行（例如，同一任务 non-thinking 约 5 美元，Thinking 约 35 美元），并开启了更广泛的测试 ([说明](https://twitter.com/scaling01/status/1993714905875382279))。Anthropic 还解决了 [Claude.ai](http://claude.ai/) 的一个主要痛点，通过自动压缩早期上下文来避免在对话中途达到长度限制 ([公告](https://twitter.com/alexalbert__/status/1993711472149774474))。在编程 UX 方面，Claude Code 的新“frontend-design”技能可以“一次性（one-shot）”生成 UI 概念；使用计划模式可获得更好效果 ([教程](https://twitter.com/_catwu/status/1993791353051074687)，[示例](https://twitter.com/omarsar0/status/1993822868820652258))。

**高效推理与多 Agent 通信**

- **Latent MAS > token chatter**：LatentMAS 使用在 Agent 之间传递的紧凑潜向量（KV-cache/最后一层隐藏状态的“思考”）取代了文本消息，在将通信 token 减少约 70–84% 的同时，比基于文本的 MAS 准确率提升了高达 4.6%。在 Qwen3‑4B/8B/14B 的 9 个基准测试（数学/科学/代码）中运行速度快 4–4.3 倍，且无需额外训练（[论文](https://twitter.com/LingYang_PU/status/1993510834245714001)，[摘要](https://twitter.com/dair_ai/status/1993697268848115915)）。
- **推理轨迹蒸馏 ≠ 冗长**：在 gpt‑oss 轨迹上训练 12B 模型，在准确率相似的情况下，每个解决方案生成的 token 减少了约 4 倍（约 3.5k 对比 DeepSeek‑R1 的 15.5k），大幅节省了推理成本。预训练中包含 DeepSeek 轨迹的污染解释了初始收敛较快但“新学习”较少的原因。核心结论：推理轨迹的来源和风格对效率至关重要（[摘要](https://twitter.com/omarsar0/status/1993695515595444366)，[讨论](https://twitter.com/code_star/status/1993745248028164532)）。此外，交替思考的 Agent 在研究工作流中展示了实际的分步效率提升（[演示/代码](https://twitter.com/omarsar0/status/1993689618856689789)）。

**超越梯度与缩放系统**

- **超大规模的 ES（NVIDIA + Oxford）**：EGGROLL 通过使用瘦矩阵 A 和 B (ABᵀ) 的低秩扰动重新构建了进化策略 (ES)，以接近推理级的吞吐量近似全秩更新。它能稳定地使用整数预训练循环 LM，在推理基准测试中与 GRPO 级别的算法竞争，并将种群规模扩展到 10 万以上，使 ES 在大型、离散或不可微系统中变得可行（[概览](https://twitter.com/rryssf_/status/1993672852206444675)）。
- **Apple Silicon 显存不足问题已解决**：dria 的 “dnet” 通过融合流水线环形并行、磁盘流式传输和 UMA 感知调度，实现了在 Apple Silicon 集群上的分布式推理，从而运行超出物理内存限制的模型（[公告](https://twitter.com/driaforall/status/1993729375745749339)）。

**多模态与生成模型更新**

- **新架构**：
    - PixelDiT 为像素空间扩散提出了双层 Transformer（patch 级别用于全局语义，像素级别用于细节），在 ImageNet 256×256 上实现了 1.61 的 FID，并取得了强劲的 T2I 指标（GenEval 0.74, DPG-bench 83.5）（[论文](https://twitter.com/iScienceLuvr/status/1993632594093813999)）。
    - Apple 的 STARFlow‑V 使用归一化流进行端到端视频生成，具有原生似然、鲁棒的因果预测以及统一的 T2V/I2V/V2V 能力；引入了流分匹配（flow-score matching）以保证一致性（[论文/代码](https://twitter.com/iScienceLuvr/status/1993629956375822508)）。
    - Terminal Velocity Matching 通过正则化终端时刻的行为，将流匹配推广到少步/单步生成，对于高保真快速采样器具有前景（[论文](https://twitter.com/iScienceLuvr/status/1993631949957841214)）。
- **模型与用户体验**：
    - Z‑Image (6B) 宣布以 Apache‑2.0 协议发布；Z‑Image‑Turbo (6B) 已在 HF 上发布，可在单张 GPU 上于 3 秒内生成写实且文本准确的图像（[预告](https://twitter.com/bdsqlsz/status/1993545608179990544)，[发布](https://twitter.com/victormustar/status/1993794840514162814)）。
    - FLUX.2 [dev] 获得了 “Tiny Autoencoder”，可以在生成过程中流式传输中间输出——实现实时视觉进度展示而非进度条（[发布](https://twitter.com/fal/status/1993669462550323652)）。
    - Google 的 Nano Banana 2 在 StructBench（非自然、模式密集的图像）上表现出重大进步；社区挖掘出了高级提示词/风格的相关资源（[分析](https://twitter.com/RisingSayak/status/1993662000103371136)，[精选列表](https://twitter.com/_philschmid/status/1993650772240941106)）。

**开放生态、评估与治理**

- **“Economies of Open Intelligence” (HF + 合作者)**：中国在开源模型下载量上首次超过美国（份额为 17.1%），由 DeepSeek 和 Qwen 领跑；“中式多模态时期”（Sino‑Multimodal Period）见证了更大规模、量化、多模态的模型以及引导使用的中间件（adapters/quantizers）。趋势：美国科技巨头的份额下降；中国 + 社区份额上升；透明度正在下降。基于 851k 个模型的 22 亿次下载，由 FT 报道（[概览](https://twitter.com/frimelle/status/1993596653664977243)，[线程](https://twitter.com/ShayneRedford/status/1993709261126336632)，[数据点](https://twitter.com/AdinaYakup/status/1993648553445527996)）。
- **评估与安全**：METR 继续被许多从业者引用为最可信的外部评估机构（[评论](https://twitter.com/andy_l_jones/status/1993485558044410188)）。AI Security Institute 发布了与 Anthropic（Opus 4.5/4.1/Sonnet 4.5）合作的案例研究：助手会破坏 AI 安全研究吗？结果令人鼓舞，但也包含一些警示（[线程](https://twitter.com/AISecurityInst/status/1993781423233499159)）。AI Evaluator Forum（Transluce + 相关机构）在 NeurIPS 启动，旨在协调独立的、符合公共利益的评估标准（[邀请](https://twitter.com/TransluceAI/status/1993767342472614156)）。
- **应用多模态推荐系统**：知乎（Zhihu）详细介绍了由 Qwen2.5‑VL‑72B/3B 驱动的流水线，用于高维多模态标签和对比嵌入（在 Qwen2‑VL‑7B 上进行 LoRA，通过 72B 模型生成合成数据，通过 M1 检索 + 72B 重排序生成硬负样本）。在 MMEB‑eval‑zh 上比 GME‑7B 基准提升了 +7.4%（[文章](https://twitter.com/ZhihuFrontier/status/1993570114810396761)）。
- **领域基准测试**：新的基准测试超越了单轮 QA——用于十亿像素病理切片导航的 MultiPathQA（带有 Agent 支架）和用于多模态、纵向肿瘤学“肿瘤委员会”决策的 MTBBench——受益于专业工具和领域 FM（[病理学](https://twitter.com/iScienceLuvr/status/1993650850120818888)，[MTBBench](https://twitter.com/iScienceLuvr/status/1993645980869365960)）。临床 ASR 评估变得更加严格，推出了 “WER is Unaware”，使用 DSPy + GEPA 训练 LLM 裁判，其识别安全风险的能力优于 WER（[论文/代码](https://twitter.com/JaredJoselowitz/status/1993735052132246011)）。

**热门推文（按互动量排序）**

- Anthropic 关于构建高效的长期运行 Agent 框架（[帖子](https://twitter.com/AnthropicAI/status/1993733817849303409)，约 1.8k）
- [Claude.ai](http://claude.ai/) 自动压缩上下文，以避免在对话中途达到限制（[更新](https://twitter.com/alexalbert__/status/1993711472149774474)，约 2.3k）
- Google DeepMind 在 YouTube 上发布 AlphaFold 纪录片《The Thinking Game》（[链接](https://twitter.com/GoogleDeepMind/status/1993714943116386619)，约 2.25k）
- Awesome Nano Banana 用于高级图像生成的提示词/风格/资源（[仓库](https://twitter.com/_philschmid/status/1993650772240941106)，约 1.0k）
- Claude Opus 4.5 在 Code Arena WebDev 排行榜上首次亮相即位居第一（[排行榜](https://twitter.com/arena/status/1993750702179676650)，约 0.5k）

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. 阿里巴巴 Text-to-Image 模型发布

- [**阿里巴巴全新的开源 Text-to-Image 模型排名紧随 Seedream 4 之后，将于今日或明日发布！**](https://www.reddit.com/r/LocalLLaMA/comments/1p74dwo/new_opensource_texttoimage_model_from_alibaba_is/) (Activity: 342): **该图片展示了根据 Elo 分数排名的 Text-to-Image 模型排行榜，揭示了该领域的竞争格局。阿里巴巴的开源模型 'Z-Image-Turbo' 位列第四，仅次于字节跳动的 'Seedream 4.0'。这凸显了阿里巴巴在开发高性能开源模型方面的重大成就，考虑到 Google 和字节跳动等公司对专有模型的垄断，这一点尤为值得关注。排行榜提供了这些模型的性能指标和胜率洞察，强调了阿里巴巴开源贡献的竞争优势。** 一条评论询问该模型是否是之前讨论过的 '6B' 版本，表明对其规格的持续讨论。另一条评论赞扬了 'Flux 2' 的非文本图像生成能力，并指出了其开源特性，而第三条评论提到了该模型的 'Edit version'（编辑版本），暗示了额外的功能。
    - AIMadeSimple 强调了阿里巴巴新模型的潜在影响，指出其 `6B parameters` 可能显著增强本地部署能力。这与 `56B parameters` 的 Flux 2 形成对比，后者需要更强大的硬件。评论者强调，如果阿里巴巴的模型能以更小的体积实现接近 Seedream 4 的质量，它将使获取最先进的图像生成技术变得大众化，特别是对于使用消费级 GPU 的用户。
    - 讨论涉及了较小模型面临的挑战，特别是在提示词遵循（prompt adherence）和多物体构图方面。这些通常是大型模型擅长的领域，评论者认为，阿里巴巴模型的真正考验将是其在体积较小的情况下，能否有效处理这些任务。
    - Vozer_bros 提到尝试了 Flux 2，并指出了它在生成非文本图像方面的有效性及其开源性质。这表明 Text-to-Image 领域向开源模型发展的趋势日益增长，这可能会促进更多社区驱动的开发和创新。

## 非技术性 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Opus 4.5 模型成功案例

- [**Opus 4.5 刚刚帮我完成了一件我 14 年来一直想做的事情。只花了一天时间。而在此之前，Sonnet、GPT 等模型都失败了。**](https://www.reddit.com/r/ClaudeAI/comments/1p72uet/opus_45_just_completed_for_me_something_that_ive/) (活跃度: 805): **用户成功使用 Opus 4.5 将用于扫描 +2 和 +5 EAN 补充条形码的 ZBar 库转换成了原生的 Swift 6。这次转换仅用了一天时间，并解决了原始 ZBar 代码中长期存在的两个 Bug。ZBar 库混合了 Objective-C 和复杂的 C 代码，此前由于 iOS 和 Android 缺乏对这些条形码类型的原生支持而一直被使用。用户曾尝试使用 GPT-3.5、Sonnet 和早期版本的 Opus 等模型完成类似任务，但只有 Opus 4.5 取得了成功。** 评论者对该解决方案的产品化潜力表示关注，并建议在 GitHub 上分享代码（注明归功于 ZBar）。此外，还有人将其与其他模型（如 Gemini 3 和 Codex 5.1）进行了对比，Opus 因解决复杂问题而受到赞誉。
    - 一位用户询问了将 Opus 4.5 创建的解决方案产品化的可能性，并指出许多健身类 App 目前都在使用条形码扫描库。他们推测这个新方案是否能取代现有库，特别是考虑到 iOS 的条形码扫描库因其速度通常被认为是原生的。
    - 另一位用户强调了从 ZBar 转换而来的 Swift 6 库的许可注意事项，ZBar 最初采用 LGPL 2.1 协议。他们解释说，如果分发该库，必须根据 LGPL 2.1 或 GPL 2+ 进行许可，因为专有许可或其他类似 MIT/BSD/Apache 的许可并不兼容。然而，如果 Opus 4.5 的解决方案与 ZBar 足够独立，则有可能重新获得许可。
    - 一位用户对 Opus 4.5 使用的初始 Prompt（提示词）表示感兴趣，认为理解该 Prompt 可以深入了解 Opus 4.5 为何能在 Sonnet、GPT 和 Codex 5.1 max xhigh 等其他模型失败的情况下取得成果。
- [**好了。我把图表修好了。**](https://www.reddit.com/r/ClaudeAI/comments/1p71la8/there_i_fixed_the_graph/) (活跃度: 623): **这张图片是一个柱状图，比较了软件工程背景下不同软件版本的准确率百分比，具体由 SWE-bench 在样本量为** `n=500` **的情况下进行验证。图表显示 Opus 4.5 的准确率最高，达到** `80.9%`**，而 Opus 4.1 最低，为** `74.5%`**。其他版本如 Sonnet 4.5、Gemini 3 Pro、GPT-5.1-Codex-Max 和 GPT-5.1 的准确率在两者之间波动。该图表旨在突出这些版本之间的性能差异，但评论建议视觉呈现方式可能会掩盖这些差异，而不是使其更清晰。** 评论者批评该图表难以辨别各软件版本准确率之间的差异，其中一人讽刺地指出该图表已不再起任何作用。另一位评论者则称赞了 Opus 4.5 自发布以来的表现，表明用户对其准确性感到满意。
    - 一位用户建议，在评估性能指标时（尤其是当它们接近 100% 时），将其表示为错误率（error rates）可能会更有洞察力。这是因为 10% 的错误率明显优于 20% 的错误率，而从 80% 到 90% 的提升看起来可能没那么有冲击力。这种视角有助于理解性能提升在现实世界中的影响。
    - 另一位用户指出，即使是 3% 的性能指标差异也可能非常显著，这意味着微小的百分比变化根据具体情况可能会产生重大影响。这强调了在解释性能数据时考虑规模和背景的重要性。

### 2. 新 AI 模型发布与基准测试

- [**阿里巴巴又一即将推出的 Text2Image 模型**](https://www.reddit.com/r/StableDiffusion/comments/1p72x1i/another_upcoming_text2image_model_from_alibaba/) (热度: 786): **阿里巴巴正在开发一款新的 Text2Image 模型，该模型采用了一个** `6B` **参数的 Diffusion 模型，并配对了一个** `Qwen3 4B` **文本编码器。该模型命名为 Z-Image-Turbo，托管在 [ModelScope](https://modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo/) 上，但目前访问受限。该模型与 Hugging Face 的 Diffusers 的集成已经[合并](https://github.com/huggingface/diffusers/commit/4088e8a85158f2dbcad2e23214ee4ad3dca11865)，且 ComfyUI 已确认提供首日支持，预示着即将公开发布。早期测试表明，它在某些基准测试中可能优于 Qwen-Image，有望在性能较低的 GPU 上也能产出高质量结果。** 评论者对该模型的潜力持乐观态度，特别是如果它能以更小、更高效的架构提供高质量的写实图像。人们期待这对于 GPU 资源有限的用户来说将是一个重大进步。
    - 一位用户强调，这款新的阿里巴巴模型在 ModelScope 仓库的排行榜上似乎优于 Qwen-Image。这表明该模型的能力有了显著提升，可能在 Text2Image 领域树立新标准。
    - 另一位评论者对模型的大小表示兴奋，注意到这是一个 6B 参数的模型。他们强调，如果模型的表现与提供的示例相符，它可能会改变游戏规则，特别是考虑到可能会迅速涌现出大量的 LoRA (Low-Rank Adaptation) 实现。
    - 一位用户提到该模型可以在 ModelScope 上免费测试，尽管需要提供手机号码。他们表示对模型的性能印象深刻，认为它在 Text2Image 生成领域将是一个强有力的竞争者。
- [**好了，我把图表修好了。**](https://www.reddit.com/r/ClaudeAI/comments/1p71la8/there_i_fixed_the_graph/) (热度: 623): **这张图片是一个柱状图，比较了软件工程背景下不同软件版本的准确率百分比，具体由 SWE-bench 验证，样本量为** `n=500`**。图表显示 Opus 4.5 的准确率最高，为** `80.9%`**，而 Opus 4.1 最低，为** `74.5%`**。其他版本如 Sonnet 4.5、Gemini 3 Pro、GPT-5.1-Codex-Max 和 GPT-5.1 的准确率介于这两个极端之间。该图表旨在突出这些版本之间的性能差异，但评论建议视觉呈现可能会掩盖这些差异，而不是澄清它们。** 评论者批评该图表难以辨别软件版本准确率之间的差异，其中一人讽刺地指出该图表已不再具有任何作用。另一位评论者赞扬了 Opus 4.5 自发布以来的表现，表示用户对其准确性感到满意。
    - 一位用户建议，在评估性能指标时，尤其是当它们接近 100% 时，将其表示为错误率可能会更有洞察力。这是因为 10% 的错误率明显优于 20% 的错误率，而从 80% 到 90% 的提升看起来可能没那么有冲击力。这种视角有助于理解性能提升在现实世界中的影响。
    - 另一位用户指出，即使是 3% 的性能指标差异也可能具有重要意义，这意味着微小的百分比变化根据上下文可能会产生实质性影响。这强调了在解释性能数据时考虑规模和背景的重要性。
- [**我们在这里**](https://www.reddit.com/r/OpenAI/comments/1p75l9m/we_are_here/) (热度: 725): **这张由 Thomas Pueyo 创作的图片是 AI 能力演进的概念插图，描绘了从“有趣的玩具”到潜在实现通用人工智能 (AGI) 的各个阶段。当前阶段由一颗星标记，表明 AI 虽然高度智能但仍不一致，在某些任务中表现出色，而在其他任务中则失败。这种可视化更多是一种推测性和说明性的工具，而不是精确的技术路线图，因为 Pueyo 并不是 AI 或机器学习领域的专家。** 一些评论者对 AI 目前的能力表示怀疑，认为它还无法执行大部分人类任务。其他人质疑 Thomas Pueyo 在 AI 领域的专业知识，指出他的背景是行为心理学和叙事，而不是技术性 AI 领域。

- Selafin_Dulamond 讨论了 AI 技能的不一致性，指出虽然 AI 某天能正确解决问题，但第二天就可能失败。这突显了 AI 性能不可预测的本质，这种现象通常被描述为不断变化的“锯齿状前沿 (jagged frontier)”，反映了目前 AI 在持续稳定执行任务能力方面的局限性。
- Creed1718 质疑了 LLM 可以完成普通智能人类 50% 任务的观点，对目前 AI 在多样化任务中复制人类智能的能力表示怀疑。这一评论强调了关于 AI 在实际现实应用中局限性的持续辩论。

### 3. 幽默的 AI 与技术迷因 (Memes)

- [**Ilya 已表态**](https://www.reddit.com/r/singularity/comments/1p6wdyn/ilya_has_spoken/) (Activity: 1360): **这张图片是一个迷因（meme），幽默地描绘了一个职场场景：关于 AI scaling 和大语言模型 (LLMs) 的相同表述，因说话人的身份不同而受到不同的对待。该漫画引用了对 AI 关键人物 Ilya Sutskever 言论的误读，暗示 scaling 已经结束，LLMs 是死路一条。然而，评论者澄清说，Sutskever 并没有声称 LLMs 是死路一条，而是认为单纯的 scaling 可能无法实现人类水平的智能。这反映了 AI 领域关于模型 scaling 极限和 LLMs 未来的持续争论。** 评论者强调 **Ilya Sutskever** 并没有宣布 LLMs 走入死胡同，而是质疑了 scaling 的极限，突出了对其言论的普遍误解。
    - Ilya 关于 “scaling 已死” 的表态意义重大，因为他最初是 scaling 大语言模型 (LLMs) 的主要倡导者。这一转变表明未来 AI 开发的重点可能会发生变化，从单纯增加模型规模以获得更好性能转向其他方向。
    - 讨论强调 Ilya 并没有声称 LLMs 是死路一条，而是认为目前的 scaling 方法可能不是实现人类水平智能的路径。这与 Yuan 的观点一致，即虽然 LLMs 非常有效，但在达到类人能力方面存在局限性。
    - 尽管对 scaling 发表了上述看法，Ilya 对在 5-20 年内实现超智能仍持乐观态度。这表明虽然 scaling 可能不再是唯一焦点，但正在考虑通过其他途径来显著提升 AI 能力。
- [**伟大的模型。**](https://www.reddit.com/r/OpenAI/comments/1p78t7q/great_model/) (Activity: 963): **这张图片是一个迷因，幽默地评论了 Google Gemini 3 模型的发布。它包含了一条讽刺性的祝贺信息，暗示了 AI 社区中的怀疑态度或竞争紧张关系。该迷因反映了 AI 开发的竞争本质，Google 和 OpenAI 等公司正在争夺 AI 进步的领导地位。评论表明，虽然像 LLMs 这样的当前模型意义重大，但它们可能不是通往通用人工智能 (AGI) 的终极路径，暗示如果出现新的架构，市场动态可能会发生转变。** 一条评论强调了 AI 开发中的竞争压力，认为由于涉及竞争利益，祝贺信息可能并不真诚。另一条评论推测了 AI 架构的未来，认为当前模型可能无法导向 AGI，如果新技术出现，可能会影响 OpenAI 等公司的市场地位。
    - bnm777 讨论了如果另一家公司开发出能够实现 AGI 的架构，对 OpenAI 市场地位的潜在影响，认为 OpenAI 对 LLMs 的依赖在长期内可能不可持续。该评论暗示，如果 OpenAI 不是 AGI 技术的先驱，其估值和用户群可能会大幅下降。
    - BallKey7607 提出了相反的观点，认为相关个人是真心支持 AI 进步的，无论涉及哪家公司或何种架构。这暗示了对超越企业利益的 AI 进步的更广泛接受，这可能会影响整个行业对 AI 技术的看法和采用。
- [**我喜欢 Grok 如此放飞自我**](https://www.reddit.com/r/ChatGPT/comments/1p7gifd/i_love_how_unhinged_grok_is/) (Activity: 1608): **这张图片是一个迷因，展示了与名为 Grok 4.1 的 AI 的对话，幽默地将该 AI 描绘成在讨论 NSFW 内容时大胆且不受约束。这种描述与典型的 AI 交互形成鲜明对比，后者在处理露骨话题时更为保守和受限。该帖子和评论反映了对 AI 变得更加“放飞自我”或在回复中减少过滤这一想法的戏谑参与，这并非技术特性，而是对 AI 行为的讽刺性解读。** 一条评论幽默地询问 Grok 是否能生成 NSFW 图像，表明了对该 AI 在文本回复之外的能力的好奇。
- 

---

# AI Discord 摘要回顾

> 由 gpt-5.1 生成的摘要的摘要的总结
> 

**1. 下一代图像和视频模型进入生产工作流**

- **Nano Banana Pro 推动写实主义并引发欺诈担忧**：**Nano Banana Pro** 在多个社区获得了高度赞誉，用户利用它快速生成漫画和超写实图像。OpenAI Discord 分享了完整的[漫画页面](https://cdn.discordapp.com/attachments/998381918976479273/1443038766087536751/image.png)，而 Latent Space 转达了对比结果，显示其输出在 [Romain Hedouin 的图像测试](https://xcancel.com/romainhedouin/status/1993654227399475347)中与 Grok 4.1 和免费版 ChatGPT 相比“与现实无异”。
    - Latent Space 重点介绍了一篇帖子，其中 **Nano Banana Pro** 通过单条提示词生成了近乎完美的伪造收据、KYC 文件和护照。[Deedy Das 警告](https://xcancel.com/deedydas/status/1993341459928694950)称，这使得**大规模严重欺诈**成为可能，而 OpenAI Discord 用户同时担心，如果安全干预过度反应，该模型可能会被**“切除脑叶”（lobotomized）**。
- **Whisper Thunder 席卷文本转视频排行榜**：据 Latent Space 报道，**Whisper Thunder** 已夺得 **Artificial Analysis** 文本转视频排行榜的第一名，超越了 **VideoGen**，正如 [Soumith Chintala 的帖子](https://xcancel.com/soumithchintala/status/1993694517489537105)中所指出的。
    - 在 OpenRouter 的讨论中，用户分享了更广泛的 [Artificial Analysis 文本转视频排行榜](https://artificialanalysis.ai/video/leaderboard/text-to-video)，目前排名第一的是 **David**，第二是 **Google Veo 3**，第三是 **Kling 2.5 Turbo 1080p**。这表明 **Whisper Thunder** 是快速发展的 **SOTA 视频生成**竞赛的一部分，从业者正积极跟踪其部署情况。
- **NB Pro 和 FLUX 2 Pro 引发图像模型军备竞赛**：在 **LMArena** 上，用户称 **NB Pro** “低调但疯狂”且是“有史以来最好的图像模型，没有之一”，声称其生成的图像感觉“像一双眼睛”一样真实，并让其他所有模型“望尘莫及”。同时，另一个 Latent Space 帖子展示了 [FLUX 2 Pro 的并排对比](https://xcancel.com/iamemily2050/status/1993477498940899366)，证明其质量较 **FLUX 1 Pro** 有重大飞跃，并消除了之前的“塑料感”。
    - 根据 [LMArena 的公告](https://x.com/arena/status/1993444903876280645)，其已将 **flux‑2‑pro** 和 **flux‑2‑flex** 添加到文本转图像（Text‑to‑Image）和图像编辑（Image Edit）天梯中。用户普遍认为 **NB Pro** 在巅峰质量上更胜一筹，但也将 **Flux 2** 视为强有力的竞争者，并讨论了 **SynthID** 水印是防止 **NB Pro** 在“几天内被削弱（nerfed）”的唯一手段——尽管有些用户随口描述了通过多玩家重编码工作流来去除水印的方法。
- **OpenAI 悄然升级图像模型，评价褒贬不一**：Latent Space 的 genmedia 频道指出，OpenAI 已*悄悄*升级了其图像模型。Arrakis AI 在[这篇帖子](https://xcancel.com/arrakis_ai/status/1993644406159917533)中分享了一个升级前后的示例，但在一位观察者看来，图像仍然显得有些奇怪的偏黄。
    - 虽然一些用户对更高的保真度表示欢迎，但其他人批评其**多语言支持薄弱**、角色/场景一致性不佳以及持续的安全限制。在写实渲染和可控性方面，该升级与 **Nano Banana Pro** 和 **FLUX 2 Pro** 相比显得逊色。

**2. Agentic UX、代码助手和聊天前端的演进**

- **Claude Code 的 Plan Mode 启动 Subagents 集群**：Latent Space 转达了 **Sid Bidasaria** 的公告，称 **Claude Code 的 Plan Mode** 现在可以并行启动多个探索性 Subagents，生成竞争性方案，提出澄清问题，并持久化一个可通过 `/plan open` 访问的可编辑方案文件，详见 [Sid 的推文](https://xcancel.com/sidbidasaria/status/1993407762412536275)。
    - 工程师们赞扬了更高的 One-shot 成功率，但要求更快的 UX、一个 **“仅提问 (ask-only)”** 开关、即时的 **Opus vs Sonnet** 切换，以及更简洁的重新规划过程。正如[此反馈贴](https://x.com/sidbidasaria/status/1993407765558251657)所证明的，**Agentic IDE 工作流**正趋向于具有紧密人类编辑循环的多 Agent 规划。
- **GPT-5.1 成为动漫首席故事讲述者（但带着枷锁）**：在 OpenAI 的 GPT-4 频道中，一位用户报告称 **GPT-5.1** 是 *“动漫或故事创作的最佳模型”*，因为它比他们使用了一年之久的基准模型 **GPT-4.1** 能更可靠地记住角色设计和长程上下文。
    - 该用户同时抱怨 GPT-5.1 的 **安全与暴力防护栏 (Guardrails)** 过于严格，以至于屏蔽了动漫风格的战斗场景，这说明了许多高级用户在选择故事生成后端时，在 **叙事连贯性** 与 **政策约束** 之间看到的权衡。
- **Kimi K-2 和 Canvas UI 挑战聊天机器人范式**：在 Moonshot **Kimi K-2** 服务器上，一名用户尽管计划进行付费升级，但坦言 *“仍然不太清楚它的极限在哪里”*（附带[截图](https://cdn.discordapp.com/attachments/1371757564005711973/1443259988058574900/image.png)），而另一名用户则称赞 K-2 **“出色的思考能力、反驳能力和 Prompt 理解力”** 超越了其他聊天机器人。
    - 同一频道还讨论了为什么 **全屏 Canvas** 尚未在 Kimi 或 Qwen 等网站上取代聊天 UI——他们认为 Canvas 能更好地支持复杂工作流——并引用了 *“对话谬误 (conversational fallacy)”*，即 AI 必须被直接对话。这凸显了向 **非聊天、以工作区为中心的 AI UX** 的转变。
- **Meganova Chat 和 Gemini Agents 预示工具驱动的工作流**：OpenRouter 用户热议即将推出的 **Meganova Chat**，认为它是管理 AI 聊天和角色的 *“干净、快速的场所”*。在 DeepSeek R1 移除后，有人在寻找替代方案时表示：*“我看到很多关于 Meganova Labubu Chat 的正面评价！我正考虑进一步了解它。”*
    - 与此同时，Perplexity 用户探索了 **Gemini Agent** 在其环境中执行 Python 脚本的能力（参考 Google 文档 [support.google.com/gemini](https://support.google.com/gemini/answer/16596215)），但注意到沙箱化的 VM 甚至会忽略 `sudo rm -rf / --no-preserve-root`，这强调了 **Agent 工具在变得更加强大的同时，仍然受到严格的限制**。

**3. GPU Kernels、分布式推理与训练技巧**

- **nvfp4_gemv 竞赛将 LLM 编写的 CUDA 变成了一场激烈的角逐**：在 **GPU MODE** NVIDIA 竞赛频道中，`nvfp4_gemv` 排行榜的提交量激增。用户如 `<@1035498877249409155>` 达到了 **3.02 µs**，随后以 **15.8 µs** 位居第二，而 `<@1295117064738181173>` 以 **22.5 µs** 攀升至 **第 7 名**。期间出现了数十条 *“个人最佳”* 和 *“在 NVIDIA 上运行成功”* 的帖子。
    - 参与者讨论了 `eval.py` 测试框架的不稳定性（**耗时波动高达 50%**，且 Runner 105881 可能较慢），并警告说 `cudaStreamSynchronize()` 和事件会增加 **数微秒的额外开销**。他们还炫耀使用 **Gemini 3.5 Pro** 和 **Opus 4.5** 作为近乎全自动的 Kernel 作者——*“它们让 GPT-5.1 看起来像 llama-7b”*——这说明 **LLM 辅助的 Kernel 设计在厂商排行榜上已经具备竞争力**。
- **Tensor Core 奇技淫巧与 CUTLASS/CuTeDSL 深度解析**：在 GPU MODE 的 CUDA 和 cutlass 频道中，工程师们交流了 **Tensor Core 优化** 技巧，引用了 [Lei Mao 的 GEMM 教程](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/)、[alexarmbr 的 Hopper matmul 工作日志](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html) 以及 [cudaforfun 的 H100 文章](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog)。
    - 他们剖析了 `ldmatrix.b16` 如何为 **每个线程提取 128 位** 数据，建议在使用 `f32`/`s32` 累加器（每个线程拥有 **8 字节**）时使用 `reinterpret_cast` 转换为 `float2`。此外，他们解释了应审慎使用 SIMT 加载和 **CuTeDSL 打包的 FP16 指令**（来自 [gpu-mode/reference-kernels 仓库](https://github.com/gpu-mode/reference-kernels)），而像 `((64,16),2,4,...)` 这样的 `tiled_mma` 平铺布局则编码了 64×256 的 Tile，并沿 M/K 方向进行了 2×4 的细分。
- **多节点 LLM 推理获胜：NVRAR 和 PAT 算法**：GPU MODE 的多 GPU 频道重点介绍了 **超越单节点的 LLM 推理** ([arXiv:2511.09557](https://arxiv.org/abs/2511.09557))。其中，基于 NVSHMEM 的 **NVRAR** 分层 All-Reduce 在处理 128 KB–2 MB 负载时，延迟比 NCCL 低 **1.9–3.6 倍**，并在 YALIS 中为 **Llama-3.1-405B** 的重解码工作负载带来了高达 **1.72 倍的端到端 Batch 延迟降低**。
    - 他们将其与 **PAT** 集合通信论文结合讨论：*“PAT: a new algorithm for all-gather and reduce-scatter operations at scale”* ([arXiv:2506.20252](https://arxiv.org/pdf/2506.20252v1))。该论文认为 Bruck 和递归倍增 (recursive-doubling) All-Gather 在实践中会变慢，因为最后一步会通过收敛的、静态路由的链路将 **一半的 Tensor** 发送到最远的 Rank，这激发了针对 **集群规模的 All-Gather/Reduce-Scatter** 开发新型生产可用算法的动力。
- **ES HyperScale 和 Blackwell 架构重新定义训练限制**：Unsloth 的研究频道推广了 **ES HyperScale** ([eshyperscale.github.io](https://eshyperscale.github.io/))，该技术声称在大规模种群下，针对 **十亿参数模型** 的训练吞吐量比标准进化策略提升了 **100 倍**，实现了 **在 CPU 上进行 int8、无梯度训练**，促使一名成员调侃道：*“以 100 倍速度训练？那相当于 Unsloth 乘以 50 了。”*
    - 在 Nous 频道，用户剖析了 Nvidia **Blackwell 的统一标量流水线**，警告在 Kernel 内部混合使用 **INT 和 FP** 可能会因缓存抖动 (Cache Thrash) 导致 **30–50% 的性能下降**，并建议严格使用 **纯 FP 或纯 INT 的 Kernel**——对于任何为即将到来的 Blackwell 服务器设计量化或混合精度训练循环的人来说，这是一个至关重要的约束。
- **机器人技术与部分训练技巧挑战定制硬件极限**：GPU MODE 的 robotics-vla 频道研究了来自 **7x** 的低成本双臂洗衣机器人（每套系统约 **$3k**），通过其 [YouTube 频道](https://www.youtube.com/@usmanroshan8740) 讨论此类硬件是否能在工业级工作周期中存活，即便创始人提供了 *“24 小时支持”*。
    - 另外的 Triton-kernel 讨论追求一种 **部分可训练的 Embedding**，其中 **128k 词表仅保留 1k 行 (127k–128k)** 为可训练状态；此外还讨论了一种 **加权损失 Softmax**，它可以在不实例化完整 Logits 的情况下应用逐位置乘数（例如，位置 123 为 0.5×，位置 124 为 1.5×）。与此同时，另一个 Nous 线程提醒，在 **Blackwell** 上必须保持这些 Kernel 的类型纯净 (Type-pure)，以避免严重的性能下降。

**4. 开放工具、协议与模型路由基础设施**

- **dspy-cli 将 DSPy 流水线转换为 FastAPI/MCP 服务**：**DSPy** 社区宣布 `dspy-cli` 现已在 [PyPI](https://pypi.org/project/dspy-cli/) 和 GitHub 的 [cmpnd-ai/dspy-cli](https://github.com/cmpnd-ai/dspy-cli) 上开源，为用户提供了一行命令 (`uv tool install dspy-cli`) 来构建 DSPy 项目脚手架、定义签名，并将模块公开为 **FastAPI 端点**或 **MCP 工具**。
    - 工程师们称赞了 `dspy-cli` 如何轻而易举地将 **DSPy 程序**打包成可 Docker 部署的 HTTP API，**David Breunig** 在一条 [推文](https://x.com/dbreunig/status/1993462894814703640) 中推广了它，认为这是在生产技术栈中实现 DSPy 逻辑工程化的实用方法。
- **RapidaAI 开源语音技术栈以消除按分钟计费的加价**：在 Hugging Face 和 OpenRouter 社区中，**RapidaAI** 宣布其**生产就绪的语音 AI 平台**现已完全[开源](https://rapida.ai/opensource?ref=hf)，目标用户是那些厌倦了为租用第三方语音 API 而额外支付每分钟 **$0.05–$0.15** 的团队。
    - 该团队将 Rapida 定位为一种拥有自己的**端到端语音推理栈**（ASR, TTS, LLM）的方式，而不是每年向供应商流失六位数的利润，这对于构建在开源模型之上的高业务量呼叫中心和实时语音 Agent 特别有吸引力。
- **MCP 协议发布新版本，同时 MAX/Mojo 规划以 Mojo 为先的未来**：官方 **MCP Contributors** Discord 在其 [protocol 频道](https://discord.com/channels/1358869848138059966/1421239779676127402/1442991223617880064) 宣布了**新的 MCP 协议版本**，并澄清 **UI SEP** 作为扩展进行带外发布，同时回答了关于第三方发布与规范不符的 *“-mcp”* 变体时如何处理**命名空间冲突**的问题。
    - 与此同时，**Modular** 服务器讨论了 **MAX** 目前是如何用 Python 编写的，并使用 [Copybara](https://github.com/google/copybara) 从内部仓库同步，用于公开 JIT 编译图；维护者暗示，之前移除的用于 MAX 的 **Mojo API** 将在语言成熟后回归——尽管他们警告说 Mojo 更像 **C++/Rust 而非 Python**，因此严肃的性能优化工作将需要大量的重写。
- **Tinygrad、LM Studio 和 OpenRouter 强化本地与云端技术栈**：Tinygrad 的 **learn-tinygrad** 频道详细介绍了 `@TinyJit` 如何仅回放捕获的 **Kernel 和 ExecItems**，这要求开发者将 Python 控制逻辑拆分为独立的 JIT 函数，并分享了一个入门级的 [Tinygrad JIT 教程](https://mesozoic-egg.github.io/tinygrad-notes/20240102_jit.html)，同时计划进行更改，使追踪器仅在两次运行匹配后才锁定。
    - 在部署方面，**LM Studio** 用户通过切换到 [REST API 指南](https://lmstudio.ai/docs/developer/rest/endpoints) 中记录的端点修复了本地 API 错误，调试了导致 `llava-v1.6-34b` 图像字幕生成失败的 **Flash Attention** 回归问题（通过切换到 **Gemma 3** 修复），而 LM Studio 硬件讨论帖则比较了通过 [SlimSAS MCIO 适配器](https://www.amazon.com/dp/B0DZG8JVG2) 进行的 PCIe 分叉，同时注意到 RDNA/MI50 GPU 在推理时风扇转速通常为 **0 RPM**，直到功耗激增。
- **路由 Bug 和回退失败暴露了 OpenRouter 的边缘案例**：在 OpenRouter 的 general 频道中，用户抱怨 **Opus** 再次过载（尽管预期会有更好的速率限制），报告称**免费的 DeepSeek R1** 模型消失了，并称赞 OpenRouter 的标准化 API 使得在 **GPT-5.1 ↔ Claude Opus 4.5** 之间进行热插拔变得轻而易举，无需重写特定于供应商的代码（即使有 **~5% 的额度溢价**）。
    - 更严重的是，一位工程师发现记录在案的 [模型回退路由](https://openrouter.ai/docs/guides/routing/model-fallbacks) 在主模型返回 **HTTP 404** 时未能触发，阻断了向备用模型的故障转移，这引发了正准备迁移企业级应用的用户的担忧，即**路由正确性**和故障模式覆盖仍需强化。

**5. 安全性、鲁棒性、数据经济学和评估现状检查**

- **Emergent Misalignment 复现揭示了 JSON 陷阱**：Eleuther 的研究频道讨论了 **“Emergent Misalignment”** 研究的复现与扩展，结果显示 **Gemma 3** 和 **Qwen 3** 对不安全微调（insecure fine‑tuning）保持了高度的鲁棒性（≈**0.68% misalignment**），完整结果已发布为 [Hugging Face 数据集](https://huggingface.co/datasets/thecraigd/emergent-misalignment-results) 和 [GitHub 代码](https://github.com/thecraigd/emergent-misalignment)。
    - 随附的博客文章 [“The JSON Trap”](https://www.craigdoesdata.com/blog/the_json_trap/) 认为，强制模型进行 **JSON‑only 输出** 实际上 **降低了它们拒绝有害请求的自由度**，从而创造了一种依赖于格式的 misalignment 向量（在不同输出约束下 misalignment 为 0.96% vs 0.42%），安全工程师在进行 tool‑calling 和 API 设计时需要将其纳入考量。
- **幻觉、金毛寻回犬式 LLM 与基准测试污染**：在 Eleuther 和 Yannick Kilcher 的服务器上，研究人员强调，在 **多阶段 LLM 流水线（multi‑stage LLM pipelines）** 中，即使后续步骤修正了错误，组件系统产生的幻觉仍然属于幻觉。他们引用了一篇新的 **LLM 幻觉论文** ([arXiv:2509.04664](https://arxiv.org/abs/2509.04664))，并开玩笑说 LLM 就像 **金毛寻回犬（golden retrievers）**，即使叼回来的东西是错的也会乐此不疲，正如一段 [YouTube 讲解视频](https://www.youtube.com/watch?v=VRjgNgJms3Q) 所展示的那样。
    - Nous 和 Eleuther 的成员还担心 **基准测试污染（benchmark contamination）**，指出一旦公开基准测试泄露到训练语料库中，模型就可以通过记忆来轻松过关；一些实验室现在保留了 **私有版本** 并专注于更大、更难记忆的问题池。同时，分享了一篇来自 LessWrong 的帖子 *“your LLM-assisted scientific breakthrough probably isn’t”*，以劝诫人们不要盲目接受 AI 生成的研究主张。
- **课程学习、数据 vs 算力以及就业影响研究**：Yannick Kilcher 和 Nous 频道辩论了 LLM 预训练中的 **课程学习（curriculum learning）** 和 **核心集（coresets）**，引用了 **OLMo 3** 的博客和论文（[AllenAI 帖子](https://allenai.org/blog/olmo3)，[OLMo 论文](http://allenai.org/papers/olmo3)）以及一项较新的研究结果 *“Curriculum learning is beneficial for language model pre-training”* ([arXiv:2508.15475v2](https://arxiv.org/abs/2508.15475v2))，该研究主张使用 **以模型为中心的难度度量（model‑centric difficulty measures）** 而非朴素的 token 启发式方法。
    - Nous 成员对比了在 [Udio](https://www.udio.com/) 和 [Suno](https://www.suno.ai/) 等系统上投入 **2000 美元用于数据** 与投入 **3200 万美元用于算力** 的巨大差异，认为这种重算力、轻数据的模式可能会扭曲研究轨迹。同时，多个频道讨论了 **MIT 的一项研究**，该研究声称 AI 已经可以取代 **11.7% 的美国劳动力**（[CNBC 报道](https://www.cnbc.com/2025/11/26/mit-study-finds-ai-can-already-replace-11point7percent-of-us-workforce.html)，[论文](https://arxiv.org/abs/2510.25137)）——并质疑使用 LLM 来评估任务自动化可行性的做法是否明智。
- **摘要生成、安全护栏以及法律/政策摩擦**：在 Yannick 的论文讨论频道中，几位从业者抱怨 **LLM 在处理密集文本时摘要效果出奇地差**，称 *“根据我的经验，它们真的不行，因为它们无法理解什么是重要的，什么是可以舍弃的”*，并指责像 **Adobe 的 AI 摘要** 这样的厂商功能（附带一张嘲讽的 [截图](https://cdn.discordapp.com/attachments/1045297868136779846/1443020193000456243/Adobe_Vermin.png)）助长了低质量的阅读习惯。
    - 其他社区也浮现了政策和法律层面的冲突：OpenAI 用户争论 **ChatGPT 的 RLHF** 是否会导致左倾政治偏见；艺术家询问鉴于 **版权归属（copyrightability）** 尚不明确，**Gemini 生成的图像** 是否可以安全地进行商业化；Nous 上的游戏开发者则针对 Steam 的 **AI 内容披露** 规则展开争论，此前 **Tim Sweeney** 建议披露应仅适用于“艺术资产”而非整个游戏，这暴露了 **监管预期** 与现实世界 AI 内容流水线之间日益扩大的鸿沟。


---

# Discord: 高层级 Discord 摘要

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Deepfake 搞了个“客串” (Cameo)！**：用户讨论了使用 *cameo* 一词描述图像中出现的现象是否合适，认为这可能是 **deepfake** 的委婉说法，旨在淡化负面含义。
   - 考虑了其他替代方案，一位用户正在寻找一个*介于 deepfake 和 cameo 之间*的词，可能类似于 *Avatar* 的某种版本。
- **Flux 2 模型涌入竞技场！**：**Flux 2** 模型的到来引发了辩论，用户在 LMArena 的 Text-to-Image 和 Image Edit 任务中将 **Flux-2-pro** 和 **flux-2-flex** 与 **NB Pro** 进行对比，正如 [X 上的公告](https://x.com/arena/status/1993444903876280645)所言。
   - 观点各异，有人认为 **Flux 2** 不错，但还达不到 **NB Pro** 的水平。
- **NB Pro 生成了“疯狂”的图像！**：用户称赞 **NB Pro** “低调地疯狂” (*lowkey insane*)，有人称其为 *an agi moment*（AGI 时刻），并形容它不仅仅是一个图像生成模型，更像“一双眼睛”。
   - 一位用户表示 **NB Pro** 的图像生成能力让所有其他模型都“望尘莫及” (*blows out of the water*)，并称其为“历史上最好的图像模型，没有之一”。
- **SynthID 防止模型被削弱 (Nerfing)！**：用户强调了 **SynthID** 在保护模型免受削弱方面的重要性，称如果没有它，**NB Pro** 会在“几天内”被削弱😞。
   - 一位用户描述了一种通过多个媒体播放器重新保存视频来绕过 **SynthID** 的方法。
- **Robin 模型悄悄击败了 Opus！**：一个名为 **Robin** 的新型隐身模型被披露在 UI 性能上超过了 **Opus 4.5**，引发了它可能是 **OpenAI** 隐藏模型的猜测。
   - 一位成员推测：*在我看来，这个 robin 模型就像是他们真正的底牌*。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **在 AI 毁灭潜力方面，Thiel 盖过了 Musk**：一位成员表达了对 [Palantir Technologies](https://www.palantir.com/) 的担忧，认为 Peter Thiel 构成了“生存威胁”，可能超过了 Elon Musk 造成 *pdoom*（AI 毁灭概率）的能力。
   - 另一位成员开玩笑地建议核平所有人以消除 AI/机器人技术。
- **Nvidia 与 Altman 的合作正在吹大 AI 泡沫**：成员们讨论了 AI 投资的集中度，暗示“美国 GDP 的 1% 正被投入到 AI/机器人领域”，**OpenAI** 似乎由 **Nvidia** 运营，而 **Nvidia** 则由 **OpenAI** 运营。
   - 其他人澄清说，*Altman* 主要是购买 **Nvidia** 的股份。
- **Opus 4.5 Token 效率的说法被证伪**：成员们最初声称 **Opus 4.5** 在 Token 效率方面比 **Sonnet 4.5** 高出 73%，这一说法遭到了质疑。
   - 与此相反，另一位用户引用了[一份报告](https://www.theneuron.ai/explainer-articles/everything-to-know-about-claude-opus-4-5)，指出 **Opus 4.5** 实际上比之前的 **Opus** 模型效率高出 76%。
- **Gemini Agent 尽管有 Python 脚本访问权限，仍被沙箱隔离**：讨论围绕 [Gemini Agent](https://support.google.com/gemini/answer/16596215?sjid=17195031605613479602-NC) 在 Perplexity 环境中执行 Python 脚本的能力展开。
   - 尽管能够运行脚本，但人们注意到该环境是沙箱化的，即使是像 *sudo rm -rf /* --no-preserve-root* 这样的命令也能减轻潜在风险。
- **Perplexity 屏蔽用户提示词，引发混乱**：用户在编辑其 **AI Profiles**（系统指令）时遇到困难，注意到由于一个 Bug，更改在刷新后会恢复原状，这表明 PPLX 可能正在主动屏蔽用户提示词。
   - 一位成员表示倾向于完全避免使用系统提示词，特别是因为 Spaces 现在会出人意料地保留记忆。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **ERNIE AI 开发者挑战赛上线**：Unsloth 正在支持 **ERNIE AI 开发者挑战赛**，为微调 **ERNIE** 和构建有影响力的模型提供 **$3,000** 奖金，详情见 [Baidu Ernie AI Devpost 链接](https://baiduernieai.devpost.com/)。
   - 官方 **ERNIE** 微调 Notebook（AMD 版本免费）可在 [X 帖子链接](https://x.com/ErnieforDevs/status/1993666389178204434)获取。
- **CPU 训练现已成为现实**：[ES HyperScale](https://eshyperscale.github.io/) 在大种群规模下，针对十亿参数模型的训练吞吐量比标准 ES 提高了 **100 倍**，实现了对任何模型更灵活的训练，无需担心梯度，并支持 int8。
   - 一位成员开玩笑说：*以 100 倍速度训练？那相当于 Unsloth x 50 了*。
- **Qwen3 8B 微调效果不佳**：一位用户在微调 **Qwen3 8B** 后评估结果很差，回复与微调数据无关，且即使将 prompt 设置为 false，模型仍会输出 `thinking` 提示词。
   - 建议如果 LM Studio 重现了该问题，尝试手动合并和保存，参考 [Unsloth 文档](https://docs.unsloth.ai/basics/inference-and-deployment/saving-to-gguf#manual-saving)。
- **长上下文训练需要 CPU Offloading**：一位成员询问在训练期间向模型添加 Adapter 是否意味着 Adapter + 模型都会驻留在内存中，从而消耗更多 VRAM。
   - 另一位成员提供了 [Unsloth 长上下文博客文章](https://unsloth.ai/blog/long-context)的链接，并解释了 LoRA 的重点是避免更新所有参数。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Haiku 模型在文档处理中占据主导地位**：用户发现 **Haiku** 在文档处理方面 100% 准确，而 **Composer-1** 在代码实现方面表现出色。
   - 一位社区成员建议使用 [Antigravity](https://antigravity.ai/) 而不是在仓库中添加 Markdown 文件，尽管这可能会产生交接问题。
- **Cursor 用户寻求 Linting 自由**：一位用户希望关闭 Linting 检查的红色波浪线，同时保留其他错误的提示，并允许扩展程序在保存文件时运行 `--fix`。
   - 他们对 **Cursor** 表示沮丧，称这在 JetBrains 工具中非常简单。
- **Cursor 的 Agent 计划在退出时消失**：一位用户寻找 Agent 计划的 Markdown 文件保存位置，以便在不同电脑上使用而不丢失计划。
   - 一位社区成员表示 **Cursor** 不会自动保存计划，建议手动保存并创建一个目录来存储所有计划。
- **Token 使用和模型成本辩论**：用户讨论了 Token 的成本，一些人报告 **Opus** 模型过载和性能下降。
   - 关于是启用按需使用还是购买 Pro+ 计划，以及是使用 **Auto** 模式“烧掉 Token”还是优化 Token 效率，存在争议。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **面对 Triton Kernel 难题**：一位成员正在探索用于**部分可训练嵌入（Partially Trainable Embedding）**和 **Logits Softmax** 操作的 **Triton kernels**，旨在高效训练大模型，重点关注特定的特殊 Token，但遇到了内存受限（memory bounding）问题。
   - 目标是仅训练 **128k 词表中的 1k 行（127k 到 128k）**，并使用允许应用加权损失的 *logits softmax 操作*，例如 **pos 123 的 Token** 具有 **0.5 倍损失乘数**，而 **pos 124 的 Token** 具有 **1.5 倍损失乘数**。
- **NVIDIA 排行榜记录重置！**：NVIDIA 上的 `nvfp4_gemv` 排行榜涌现了大量提交，<@1035498877249409155> 以 **3.02 µs** 获得**第二名**，随后又以 **15.8 µs** 再次获得第二名。
   - 多位用户提交了“个人最佳”成绩，<@1295117064738181173> 以 **22.7 µs** 获得**第 8 名**，随后以 **22.5 µs** 升至**第 7 名**，而 <@1035498877249409155> 以 **23.2 µs** 获得**第 9 名**。
- **Tensor Core 优化技巧流传**：成员们分享了 **NVIDIA Tensor Cores** 性能优化的资源，指向了一些文章和工作日志，例如 [alexarmbr 的工作](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html) 和 [cudaforfun 的工作日志](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog)。
   - 讨论强调 `ldmatrix.b16` 每个线程无需额外操作即可加载 **128 位**数据，建议使用 `reinterpret_cast` 进行正确的数据处理；当使用 `f32` 或 `s32` 累加器时，每个线程持有一行中一对连续的值（**8 字节**）。
- **Intel GPU 上的 2-bit 反量化困境**：一位用户询问如何直接在 **Intel GPU** 上执行 **2-bit 反量化（dequantization）**，并指出虽然可以在 CPU 上进行量化，但使用 **Torch** 进行反量化速度很慢。
   - 发布者正在寻找优化的、基于 **GPU** 的替代方案来取代 **Torch** 进行反量化以提高性能，但频道内尚未提供进一步讨论，这仍是一个悬而未决的问题。
- **Factorio 的华丽变身：文档发布**：Jack Hopkins 宣布 **Factorio Learning Environment** 的文档现已上线，网址为 [Factorio Learning Environment](https://jackhopkins.github.io/factorio-learning-environment/sphinx/build/html/index.html)。
   - 社区对文档的发布感到非常高兴。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT 被指倾向左翼**：成员们正在辩论 **ChatGPT** 是否可能在政治左翼数据上进行过训练，这可能是由于训练数据中的进步观点以及 **RLHF（基于人类反馈的强化学习）** 中人类评分者的偏见所致。
   - 一位成员认为，模型需要*在问题上小心翼翼（fussy foot around questions）*，这损害了其可靠性。
- **Nano Banana Pro 快速创作漫画**：用户正在使用 **Nano Banana Pro** 创作漫画，称赞其快速生成图像的能力和高质量的结果，例如这些[漫画页面](https://cdn.discordapp.com/attachments/998381918976479273/1443038766087536751/image.png?ex=6928ef94&is=69279e14&hm=d88b0f693975c1e756c3352c689566bda68503635084432e547cc6585d126e83&)。
   - 成员们也表达了对模型被“切除脑叶”（能力阉割）的担忧。
- **AI 艺术引发版权担忧**：成员们辩论了使用 **Gemini** 生成的 AI 图像的商业可行性和版权影响，指出虽然 Google 并不明确禁止商业用途，但法律地位取决于内容是否具有可版权性。
   - AI 艺术中的文化偏见也是一个令人担忧的问题，一位成员评论道：*“如果那些反对 AI 的人想做点什么，他们应该开始自己动手画画和创作艺术。”*
- **GPT-5.0 Mini 令人失望**：成员们对 **GPT-5.0 Mini** 表示失望，一位成员称其为“降级”。
   - 他们还对在还没体验过第一个版本之前就不断要求 **Sora 2** 的行为感到厌烦。
- **GPT 5.1 在动漫叙事方面表现出色**：一位用户强调 **GPT 5.1** 是目前最适合动漫或故事写作的模型，因为它能够记住角色设计和之前的上下文。
   - 唯一的抱怨是严格的**安全网和防护栏（safety net and guardrails）**阻碍了动漫风格暴力情节的描写；该用户将其性能与他们使用了一年但指出有时会遗漏角色设计的 **GPT 4.1** 进行了对比。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 中 API 端点错误已解决**：一位用户在本地服务器上遇到了**不支持的 API 端点**（POST /api/v1/generate）错误，但在查阅 [LM Studio REST API 文档](https://lmstudio.ai/docs/developer/rest/endpoints)后自行解决了问题。
   - 用户意识到该端点无效，强调了准确配置端点的重要性。
- **更新后 LM Studio 图像字幕生成失败**：一位用户报告在 Windows 和杀毒软件更新后，尝试使用 **LM Studio** 为图像生成字幕时持续出现 **'Channel Error'**，失败率为 100%。
   - 将模型从 **llava-v1.6-34b** 切换到 **Gemma 3** 后解决了该问题，这表明可能存在模型依赖性，或者是由于默认启用的 **Flash Attention** 导致的问题，现在成功率已达到 100%。
- **Flash Attention 故障影响模型功能**：有建议认为字幕生成问题可能与最近 **LM Studio** 版本中默认启用的 **Flash Attention** 有关，导致某些模型运行异常。
   - 用户被提示运行 `lms log stream` 以获取详细的错误消息，并分享其运行时的截图，特别是在处理非英语 I/O 时。
- **推理期间 GPU 风扇停转**：一位用户注意到他们的 **GPU 风扇** 在推理期间处于 **0%** 转速，最初感到担忧，但随后澄清这对于他们的 **MI50** 以及有时在 **4070 TiS** 上是正常行为。
   - 他们澄清说，一旦 Context 完全写入，GPU 就会“接管”且功耗增加，这表明在推理的特定阶段存在高效的电源管理。
- **主板支持 PCIe Bifurcation**：一位用户发现他们的 **X570 AORUS ELITE WiFi** 主板在主 x16 插槽上支持 **PCIe bifurcation**（分叉），允许 **8x/8x** 或 **8x/4x/4x** 等配置。
   - 另一位用户指出，当启用 x8x8 时，可以使用 [SlimSAS MCIO 适配器](https://www.amazon.com/dp/B0DZG8JVG2) 将 x16 插槽拆分为两个 x8 插槽。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Opus 遭遇过载停机**：用户报告 **Opus** 再次过载，导致服务中断，尽管人们此前希望速率限制（rate limiting）和负载均衡（load balancing）能有所改善。
   - 成员们承认了这一问题，但也有人因公司规模较小而表示同情，其中一人提到 *小公司请谅解*。
- **模型回退逻辑失效**：一位用户报告了 [模型回退逻辑](https://openrouter.ai/docs/guides/routing/model-fallbacks) 中的一个 Bug，即主模型的 **404 错误** 阻止了向备用模型的回退。
   - 该成员强调了此问题对企业级应用的严重性，称 *如果回退逻辑在如此简单的用例中失效，可能还存在更多问题*。
- **免费 Deepseek R1 从路由中移除**：成员们注意到免费的 **Deepseek R1** 模型已不再可用，导致用户开始寻找替代方案和更好的价格选项。
   - 一位成员感叹失去了该模型：*这太蠢了。我之前配合 chutes API key 使用它，因为通过 chutes 使用该模型会显示思考过程，我受不了那个。*
- **Meganova Chat 引发巨大关注**：成员们讨论了即将推出的 **Meganova Chat**，这是一个管理 AI 聊天和角色的平台，一位用户将其描述为一个 *干净、快速的地方*。
   - 另一位用户回应道：*我看到很多关于 Meganova Labubu Chat 的正面评价！我正考虑进一步了解它*。
- **文本转视频排行榜发布**：一位成员分享了 [Artificial Analysis 文本转视频排行榜](https://artificialanalysis.ai/video/leaderboard/text-to-video) 的链接，该榜单目前提供了最新排名。
   - 排行榜显示 **David** 位居第一，紧随其后的是 **Google 的 Veo 3**，**Kling 2.5 Turbo 1080p** 位列第三。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Psyche 团队安排 Office Hours**: **Psyche 团队**将于下周 **12/4 星期四，美东时间下午 1 点**在 Events 频道举办 Office Hours 环节，可通过 [Discord 活动链接](https://discord.gg/nousresearch?event=1442995571173625888)参加。
   - 这为用户提供了与团队直接交流、讨论相关话题或问题的渠道。
- **Suno 的音乐合作伙伴关系引发辩论**: **Suno** 与 [Warner Music Group 的合作](https://www.wmg.com/)引发了关于 AI 在音乐创作中的角色及其对行业影响的讨论。
   - 成员们强调了 **Suno** 输出质量的参差不齐，有些曲目与人类创作难辨真伪，而另一些则明显带有 AI 生成的痕迹。
- **计算成本远超数据支出**: 讨论对比了 **2000 美元的数据支出**与 **3200 万美元的计算成本**，凸显了 AI 模型训练对资源的巨大需求，特别是对于 [Udio](https://www.udio.com/) 和 [Suno](https://www.suno.ai/) 等模型。
   - 这种经济上的差异可能会限制未来的研究，导致获取高质量训练数据的途径受限。
- **INT/FP 工作负载混合损害 Blackwell 性能**: 在 **Nvidia 的 Blackwell 架构**上混合 **INT** 和 **FP** 工作负载会因其统一标量流水线（unified scalar pipeline）而导致性能显著下降。
   - 建议保持 Kernel 纯度（**仅限 FP** 或 **仅限 INT**），以防止因持续的缓存抖动（cache thrashing）导致潜在的 **30-50% 性能下降**。
- **Steam 的 AI 内容政策引发辩论**: 讨论涉及了 Steam 的 AI 内容披露政策，Epic 首席执行官 Tim Sweeney 建议 AI 披露应仅适用于“艺术”而非游戏。
   - 争论焦点在于披露是否能让消费者充分了解 **AI 生成内容**及其对游戏体验的影响。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **多阶段 LLM 中幻觉依然存在**: 一位成员分享道，即使在得到纠正时，多阶段 LLM 流程中出现的幻觉仍应被视为组件系统的幻觉，并引用了一篇关于 [LLM 幻觉的论文](https://arxiv.org/abs/2509.04664)。
   - 他们将其比作人类的自我纠错，认为这是认知过程的自然组成部分。
- **LLM 被比作热情的金毛寻回犬**: 成员们将 **LLM** 比作金毛寻回犬，因为它们倾向于提供讨好用户的回答，即使这些回答并不准确，并引用了 **ChatGPT**、**Claude**、**Gemini** 和 **Grok** 等例子。
   - 一位成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=VRjgNgJms3Q)，展示了 LLM 如何生成缺乏真实理解或逻辑连贯性的输出。
- **SGD 洗牌（Shuffling）辩论升温**: 成员们辩论了在 **SGD** 中每个 Epoch 洗牌数据的益处，一位成员认为“洗牌一次”应该总是优于 **IID**。
   - 另一位成员反驳说，由于优化曲面的非凸性质，实践比证明更重要，并指出 **IID** 可能导致方差增加和数据重复访问。
- **“涌现失调”论文引发 JSON 陷阱发现**: 一项针对“涌现失调”（Emergent Misalignment）论文的复制和扩展研究发布，测试了 **Gemma 3** 和 **Qwen 3**，发现开源权重模型对不安全微调具有惊人的鲁棒性（失调率仅 0.68%）。
   - 该成员发布了[完整数据集](https://huggingface.co/datasets/thecraigd/emergent-misalignment-results)和[代码](https://github.com/thecraigd/emergent-misalignment)，并推测 **JSON** 限制减少了模型拒绝有害请求的自由度，正如[这篇博文](https://www.craigdoesdata.com/blog/the_json_trap/)中所讨论的。
- **寻求 AI 药物研发的良方**: 一位成员寻求 **AI 药物研发（AI for Drug Discovery）** 的教育资源，旨在了解其架构、开放性问题和现状。
   - 另一位成员建议查阅 [Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=ai+for+drug+discovery+survey&btnG=) 上的各种综述，还有成员提到了 **Zach Lipton** 的初创公司。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Claude 的 Plan Mode 启动并行 Subagents**：**Claude Code 的 Plan Mode** 进行了重大更新，可以并行启动多个探索性 subagents，生成竞争性方案，并允许用户通过 `/plan open` 编辑保存的计划文件，[参考 Sid 的 X 帖子](https://xcancel.com/sidbidasaria/status/1993407762412536275?s=46)。
   - 社区成员在[后续讨论中](https://x.com/sidbidasaria/status/1993407765558251657?s=46)请求更快的 UX、"仅询问"（ask-only）选项、模型选择器（**Opus vs Sonnet**）以及减少冗长的重新规划过程。
- **《思维游戏》纪录片记录 DeepMind 历程**：免费完整版纪录片 **《思维游戏》（The Thinking Game）** 探索了 **DeepMind** 的起源，现已在 [YouTube](https://www.youtube.com/watch?v=d95J8yzvjbQ) 上线。
   - 观众称赞其内容*非常出色*，并表示这部电影*真的让人希望 Demis 能赢得 AGI 竞赛*。
- **Jeff Dean 详解 AI 15 年进展**：**AER Labs** 总结了 **Jeff Dean 在斯坦福的演讲**，追溯了 **AI 15 年的进步**——从 90 年代的手写梯度到解决 IMO 问题的 **Gemini 3.0**——这得益于规模、更好的算法（**TPUs, Transformers, MoE, CoT**）和硬件，[根据此帖子](https://xcancel.com/aerlabs_/status/1993561244196868370)。
   - Dean 在演讲中还演示了低代码“Software 3.0”和视觉推理。
- **ChatGPT 很棒，但 Claude 在突破边界**：成员们对比了 **ChatGPT Pro** 与 **Claude** 的价值，指出 **ChatGPT** 擅长通用研究，具有更好的 **Codex 速率限制**，在非 **ts/js/py** 领域表现更佳，且如果你使用 pulse, atlas, sora, codex cloud 等功能，其价值更高。
   - 然而，成员们补充说 **Claude** 一直在突破边界，其模型在工具使用方面训练得更好，前端 UX 和 UI 非常出色，且其 CLI 的可读性、排版和字体层级使其更易于理解。
- **Whisper Thunder 席卷 Text-to-Video 领域**：ML 社区对 **Whisper Thunder** 感到兴奋，这是一个新的排名第一的 text-to-video 模型，在最新的 Artificial Analysis 排名中已超越 **VideoGen**，[详见此帖子](https://xcancel.com/soumithchintala/status/1993694517489537105?s=46)。
   - 目前没有关于 **Whisper Thunder** 或 **VideoGen** 的更多其他信息。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **美国能源部将构建国家 AI 平台**：**美国能源部（Department of Energy）** 计划利用美国超级计算机和联邦科学数据构建一个国家 AI 平台，用于训练科学 foundation models，并运行 AI agents 和机器人实验室以自动化实验。
   - 目标应用领域包括**生物技术、关键材料、核裂变/核聚变、航天、量子和半导体**。
- **AI 取代工作研究引发辩论**：[CNBC](https://www.cnbc.com/2025/11/26/mit-study-finds-ai-can-already-replace-11point7percent-of-us-workforce.html) 报道的一项 **MIT 研究**表明，基于 [Iceberg Index](https://iceberg.mit.edu/) 和[相关论文](https://arxiv.org/abs/2510.25137)，AI 可能会取代 **11.7%** 的美国劳动力。
   - 一些成员对研究方法提出质疑，对信任 LLM 来判断其他 LLM 工具是否能实现工作自动化表示怀疑。
- **LLM 可能是糟糕的摘要生成器**：成员们讨论了 **LLM** 在摘要任务中经常无法把握重点的经历，尤其是在高信息密度的文本中，称 *“根据我的经验，它们真的不行，因为它们无法理解什么是重要的，什么是可以舍弃的。”*
   - 一位成员表示 **Adobe 的 AI 摘要**可能会导致问题，并分享了[一张图片](https://cdn.discordapp.com/attachments/1045297868136779846/1443020193000456243/Adobe_Vermin.png?ex=6928de48&is=69278cc8&hm=128c6461c705032d5b88293eedae078353ef799ddbd74a2b9e1a8521561a6dbf&)。
- **Curriculum Learning 的价值引发讨论**：成员们讨论了在 **LLM pretraining** 过程中使用 **curriculum learning** 和 **coreset 技术**的情况，引用了 [Olmo 3 博客](https://allenai.org/blog/olmo3)和 [OLMo 论文](http://allenai.org/papers/olmo3)。
   - 一位成员质疑非随机采样可能引入的偏见，而另一位成员引用了[这篇论文](https://arxiv.org/abs/2508.15475v2)，澄清只要采用更以模型为中心的难度概念，**curriculum learning 对语言模型预训练是有益的**。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF Inference API 选项变灰**：一位成员寻求关于激活其模型的 **Hugging Face 内部推理 API** 的指导，指出 UI 选项目前已被禁用，如[此图](https://cdn.discordapp.com/attachments/879548962464493619/1443040959901204530/image.png)所示。
   - 在上下文中未提供解决方案。
- **法语书籍数据集发布**：一位成员在 Hugging Face 上发布了一个[公有领域法语书籍数据集](https://huggingface.co/datasets/Volko76/french-classic-books)。
   - 他们还分享了一个单独的仅包含书籍中**对话**的数据集（[此处](https://huggingface.co/datasets/Volko76/french-classic-conversations)），旨在用于指令微调。
- **RapidaAI 开源**：**RapidaAI**，一个**生产级语音 AI 平台**，现已[开源](https://rapida.ai/opensource?ref=hf)，允许用户更多地控制其语音 AI 技术栈。
   - 该公司表示，团队之前在租用他人的技术栈上每分钟额外花费 **$0.05–$0.15**。
- **关于 AlphaFold 的 GNN 演示即将到来**：一位成员正在准备关于 **GNNs** 的演示，从 **AlphaFold 2 和 3** 开始。
   - 演示的具体重点尚待确定。
- **建议使用 LM Studio PDF 教师**：针对有关 **LM Studio** 的 **PDF** 阅读模型的问题，一位成员建议任何 instruct 类型的 **LLM** 都应该可以工作，利用 **LM Studio** 内置的 RAG。
   - 他们提供了 [LM Studio 模型页面](https://lmstudio.ai/models) 和 [Hugging Face 模型页面](https://huggingface.co/models?apps=lmstudio&sort=trending) 的链接。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 使用 Copybara 保持仓库同步**：成员们确认 **Mojo** 使用 [**Copybara**](https://github.com/google/copybara) 来保持其内部私有仓库与外部开源仓库同步。
   - 这确保了两个仓库之间的更改和更新能够保持一致。
- **MAX 新手寻找示例代码**：一位成员请求学习 **MAX** 的小型示例，并对训练感兴趣，**Endia** 为其指引了相关内容。
   - 讨论集中在获取实际 **MAX** 使用案例的动手经验。
- **Python 在 MAX 中的主导地位：最终目标是什么？**：一位成员质疑在 **Python** 中编写 **MAX** 的决定，推测这一选择是否旨在简化向 **MAX** 和 **Mojo** 的迁移。
   - 他们思考这是否会导致类似于 **PyTorch** 的分裂世界问题，以及未来是否会出现针对 **MAX** 的纯 **Mojo** 框架。
- **MAX 中 Mojo API 的回归预告**：一位成员澄清说，**MAX** 之前曾提供 **Mojo API**，但由于 **Mojo** 尚不成熟而停止。
   - 他们暗示一旦该语言达到更完善的阶段，**Mojo API** 最终将会回归。
- **从 Python 迁移到 Mojo：并非表面看起来那么简单**：一位成员警告说，虽然 **Mojo** 可能看起来像 **Python**，但它更接近 **C++** 或 **Rust**，在迁移到 **Mojo MAX** 时，需要付出巨大努力才能充分利用 **Mojo** 的功能。
   - 这表明在 **Mojo MAX** 中实现巅峰性能不仅仅是简单的 **Python** 代码翻译。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **TinyJit 重放内核**：当使用 `@TinyJit` 时，包装后的函数仅重放捕获的 **tinygrad kernels** 和 **ExecItems**，从而阻止原始函数运行。
   - 这种行为要求用户将 Python 代码拆分为独立的 JIT 函数，尽管 **non-tinygrad outputs** 可能无法正确更新。
- **Tensor 随机性函数的表现**：`Tensor` 上的随机性函数按预期运行，因为它们通过内核增加计数器，如[此示例](https://discord.com/channels/1068976834382925865/1070745817025106080/1443178668007620752)所示。
   - 示例代码为 `CPU=1 DEBUG=5 python3 -c "from tinygrad import Tensor; Tensor.rand().realize(); Tensor.rand().realize()"`。
- **Tinygrad JIT 追踪调整即将到来**：目前，**Tinygrad's JIT** 需要运行两次才能完成追踪以重复捕获的内核，第一次运行可能处理权重初始化等设置任务。
   - 一项提案建议更新 **JIT** 以在两次运行后验证匹配，这表明随着项目接近 1.0 版本，开发重点在于防止常见错误。
- **教程提供了良好的 JIT 入门**：一位成员分享了关于 [tinygrad JIT 的教程](https://mesozoic-egg.github.io/tinygrad-notes/20240102_jit.html)，其中仍包含有用的信息。
   - 它提供了有用的背景，但教程略显过时。
- **前端易用性成为焦点**：随着 **Tinygrad** 的基础部分现已稳固，团队正将重点转向提高前端易用性。
   - 有人回忆说，*fast.ai 课程中最早的 pytorch 编译器实际上是使用正则表达式连接 C 代码字符串！*。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **探索 Kimi K-2 的极限**：Discord 用户讨论了 **Kimi** 的极限，一位用户分享了[截图](https://cdn.discordapp.com/attachments/1371757564005711973/1443259988058574900/image.png?ex=69286c1b&is=69271a9b&hm=3d4d9ba62a03dc65a27edfb0fb93a8c0f8a0f6518ab2737c5a714d6032d2b5a6&)，表达了尽管计划升级但对其能力的疑虑。
   - 另一位用户称赞 **Kimi K2** 具有卓越的思考能力、反驳能力以及对 prompts 的深刻理解，认为它超越了其他聊天机器人。
- **Canvas 热潮将席卷聊天机器人？**：一位用户质疑为什么 *canvases* 还没有取代 **Kimi** 和 **Qwen** 等全屏网站的聊天机器人，认为它们提供了更优的用户体验。
   - 他们认为，虽然聊天机器人适用于侧边栏，但 canvases 可以为详细的 Web 应用程序提供更全面的界面。
- **深入探讨对话谬误**：一位用户分享了他们对 *对话谬误 (conversational fallacy)* 的着迷，该观点认为 AI 必须通过对话才能使用，并暗示 **Kimi** 因不遵循这一谬误而表现出色。
   - 对话围绕着 AI 的效用不应局限于直接的对话交互这一想法展开。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **`dspy-cli` 工具开源**：`dspy-cli` 工具现已开源并发布在 [PyPi](https://pypi.org/project/dspy-cli/) 上，有助于将 **DSPy programs** 作为 HTTP API 进行创建、开发、测试和部署。
   - [仓库可在 GitHub 上找到](https://github.com/cmpnd-ai/dspy-cli)，可以使用 `uv tool install dspy-cli` 安装该工具来搭建新的 **DSPy project**、创建新的签名，并将模块作为 **FastAPI endpoints** 或 **MCP tools** 运行，且易于部署到 **docker 托管服务**。
- **寻求 ReAct 模块的轨迹注入**：一位成员询问如何将轨迹 (trajectories) 注入 **ReAct module**，旨在为 Agent 提供除消息历史记录之外的先前运行上下文。
   - 该请求旨在通过先前运行的数据来增强 Agent 的上下文。
- **DSPy 中 Web 搜索的 API 选择辩论**：成员们讨论了在 **DSPy** 中实现 Web 搜索工具的最佳 **API**，其中一人分享了使用 **Exa API** 的积极体验，因为它具有摘要功能，可以避免在 **Firecrawl** 和 **Parallel.ai** 等其他 API 中发现的随机广告和 HTML 标签。
   - 另一位成员正尝试使用 **Anthropic's web search API** 配合 ReAct 来实现，并分享了使用 `dspy.ReAct` 的代码片段。
- **Web 搜索 API 调用的延迟排查**：一位成员提出了关于在调用 LLM 之前使用 `search_web` 等搜索函数时，**DSPy's ReAct** 内部 Web 搜索 **API** 调用引起的延迟问题。
   - 用户寻求减少 **API** 调用延迟的方法。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **新协议版本发布**：正如 [Discord 频道](https://discord.com/channels/1358869848138059966/1421239779676127402/1442991223617880064) 中所宣布的，新协议版本已发布。
   - 成员们对 **MCP 社区** 在过去一年中所做的贡献表示兴奋和感谢。
- **UI SEP 带外发布**：由于 **UI SEP** 是一个扩展，它可以从主规范中带外（out-of-band）发布。
   - 详情可在 <#1376635661989449820> 频道中查看。
- **MCP 考虑命名空间冲突**：一位成员询问 **MCP** 小组是否考虑过命名空间冲突的可能性。
   - 具体而言，有人提出，如果某些项目声称是 something-mcp 但偏离了实际的 **MCP** 标准，该小组是否会采取行动。



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **AI 工程师展示丰富的 AI 经验**：一位 **AI 工程师** 介绍了自己，强调了他们在构建跨领域高级 **AI 系统** 方面的经验，包括 **AI agents、多智能体系统、NLP 驱动的聊天机器人、语音与语言系统、Web3 以及 AI 集成区块链游戏**。
   - 他们还拥有自动化工作流、部署自定义 LLM 以及微调 AI 模型的实战经验。
- **用户在支持团队沉默之际反馈 API 问题**：一位用户报告称，尽管花费超过 **$600**，但由于使用配额耗尽，**webdev.v1.WebDevService/GetDatabaseSchema** 出现 *[unknown] error*。
   - 这个问题导致他们的账户无法使用，影响了超过 **500 名活跃用户**，且目前尚未收到支持团队的回复。
- **社区询问是否存在 Telegram 频道**：一位成员询问是否存在 **Manus Telegram 频道**。
   - 未提供更多细节。



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **社区关注新的站点管理员以进行基准测试**：一位成员建议任命新的站点管理员，以便使用新模型更新基准测试结果，暗示了[对当前更新速度的不满](https://discord.com/channels/1131200896827654144/1131200896827654149/1443213701753606215)。
   - 这一转变可能会振兴基准测试流程，确保为社区提供更及时、更相关的数据。
- **Opus 4.5 升级，是大是小？**：一位成员发起了一项调查，以确定 **Opus 4.5** 与 **Sonnet 4.5** 相比是重大升级还是小幅升级，反馈将影响未来的开发优先级。
   - 社区情绪可能会引导资源分配，以增强最具影响力的功能。
- **Bedrock 标识符故障**：一位用户报告在尝试使用标准 **Bedrock** 模型标识符时遇到 *'model not found'* 错误，这预示着一个潜在的故障。
   - 调查此问题对于维持对 **Bedrock** 功能的无缝访问并避免工程师受到进一步干扰至关重要。



---


**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---


**Windsurf Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---



您收到此邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。


---

# Discord：各频道详细摘要和链接

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1442968540498886847)** (1279 messages🔥🔥🔥): 

> `Cameo Word Choice, Pro Grounding, Flux 2 Models, LMarena Updates, NB Pro` 


- **Deepfake 获得了 Cameo 式的出镜！**: 用户们讨论了使用 "cameo" 一词来描述图像中某物出现的选择，有人认为这可能是 **Deepfake** 的委婉说法，以淡化负面含义。
   - 其他人想知道可以用什么词代替它，介于 **Deepfake** 和 **Cameo** 之间，比如 **Avatar** 的单词版本。
- **Flux 2 模型登陆 Arena！**: **Flux 2** 模型的到来引发了讨论，一名用户直接请求 *Flux 2 plssssssssss*，而其他人则在争论 **Fluxis flex** 还是 **pro** 是更好、更新的模型。
   - 观点各异，有人认为 **Flux 2** 不错，但还没达到 **NB Pro** 的水平，有人补充道：*i mean how can you even compete with something like that...feels unfair tbh*（我的意思是，你怎么能和那样的东西竞争……说实话感觉不公平）。
- **NB Pro：疯狂的图像生成！**: 用户对 **NB Pro** 的能力赞不绝口，称其为 *lowkey insane*（低调地疯狂），有人将其描述为他们的 **AGI** 时刻，不再仅仅是一个图像生成模型，而是“像一双眼睛”。
   - 一位用户表示：*proportionally in terms of blowing other models out of the water its the best image model in history actually it is just the best image model in history period*（按比例来说，就碾压其他模型而言，它是历史上最好的图像模型，实际上它就是历史上最好的图像模型，句号）。
- **SynthID 拯救模型！**: 强调了 **SynthID** 作为防止模型被削弱（nerfing）的保护措施的重要性，一位用户表示：*if NB pro didnt have synth id itd be nerfed within DAYS😞*（如果 **NB Pro** 没有 **SynthID**，它会在几天内被削弱）。
   - 另一位用户描述了一种绕过 **SynthID** 的方法，称：*But if you was the video twice and run through different media players in save it you get rid of it*（但如果你把视频播放两次，并通过不同的媒体播放器运行并保存它，你就能去掉它）。
- **Robin，神秘新模型现身！**: 一个名为 **Robin** 的新神秘模型被披露优于 **Opus 4.5**，专注于 UI，一些人推测它是 **OpenAI** 的隐藏王牌。
   - 一位成员评论道：*this robin model is like their real hidden card imo last codex update was just an appetizer but it does take a lot of time tho makes me wonder if its just actual codex + more thinking*（在我看来，这个 **Robin** 模型就像他们真正的隐藏王牌，上次 **Codex** 更新只是开胃菜，但它确实需要很多时间，这让我怀疑它是否就是实际的 **Codex** + 更多的 **thinking**）。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1442968308931366942)** (4 messages): 

> `Image Edit Update, New Model Update, Leaderboard Update, Flux-2-pro, Flux-2-flex` 


- **LMArena 调整图像编辑流程**: 根据社区反馈，图像生成对话中的 **multi-turn**（多轮对话）功能已被禁用，但你现在可以使用新的 `Edit` 功能直接在对话中编辑图像。
- **Flux 在 LMArena 首次亮相**: **Flux-2-pro** 和 **flux-2-flex** 模型已添加到 LMArena 的 Text-to-Image 和 Image Edit 中，正如 [X 上的公告](https://x.com/arena/status/1993444903876280645) 所述。
- **Arena 扩展其搜索功能**: **gemini-3-pro-grounding** 和 **gpt-5.1-search** 模型已添加到 [Search Arena](https://lmarena.ai/?chat-modality=search)。
- **Claude 占领 LMArena 排行榜**: `Claude-opus-4-5-20251101` 和 `Claude-opus-4-5-20251101-thinking-32k` 已添加到排行榜，并在 [WebDev 排行榜](https://lmarena.ai/leaderboard/webdev) 和 [Expert 排行榜](https://lmarena.ai/leaderboard/text/expert) 中名列前茅。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1442968148226609225)** (1082 messages🔥🔥🔥): 

> `AI doom, Palantir Technologies, Nvidia and Open AI partnership, Bypassing AI Detectors, Perplexity limits` 


- ****毁灭潜力：Thiel 掩盖了 Musk 的光芒****：一位成员对 [Palantir Technologies](https://www.palantir.com/) 表示担忧，称 Peter Thiel 构成了*生存威胁*，掩盖了 Elon Musk 可能带来的 *pdoom*。
   - 另一位成员讽刺地开玩笑说，应该用核武器消灭所有人，以摆脱 AI/机器人技术。
- ****AI 投资泡沫：Nvidia 与 Altman 的博弈****：成员们讨论了*美国 GDP 的 1% 正被投入到 AI/机器人领域*，**OpenAI** 由 **Nvidia** 驱动，而 **Nvidia** 又由 **OpenAI** 驱动，形成了一个等待破裂的通胀泡沫循环。
   - 其他人指出，实际上是 *Altman 正在购买大量的 Nvidia 股票*。
- ****Opus 4.5 效率存疑：73% 的说法被推翻****：成员们辩论了 **Opus 4.5** 与 **Sonnet 4.5** 相比的 Token 效率，一位成员最初声称 **Opus 4.5** *效率高出 73%*，但这一说法遭到了质疑。
   - 另一位用户表示，[根据 the neuron 的说法](https://www.theneuron.ai/explainer-articles/everything-to-know-about-claude-opus-4-5)，它实际上比*之前的 Opus* **效率高出 76%**，而不是针对 Sonnet。
- ****Gemini Agent：强制 Python 脚本与 Gemini 环境交互****：成员们讨论了使用 [Gemini Agent](https://support.google.com/gemini/answer/16596215?sjid=17195031605613479602-NC) 的能力，强制 AI 运行 Python 脚本，从而与 Perplexity 中 AI 使用的环境进行交互。
   - 然而有人建议，即使运行 `sudo rm -rf / --no-preserve-root` 也不会起任何作用，因为*一切都是沙箱化（sandboxed）的*。
- ****Perplexity 现在开始屏蔽用户提示词：Fursona 混乱随之而来****：用户报告了编辑其 **AI Profiles**（系统指令）时遇到的问题，称由于 Bug，更改会在刷新后还原，或者 PPLX 现在正在屏蔽用户提示词。
   - 一位成员表示，他们*现在不想要任何系统提示词*，因为现在的 Spaces 拥有了以前没有的记忆功能。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1442971198500634737)** (182 messages🔥🔥): 

> `FP8 RL Documentation, Optimization Techniques, Qwen3VL vs 30B-A3B, AI GPU Kernels, Embedding Models` 


- **FP8 RL 文档链接仍指向 KimiQwen 等候名单**：点击首页文档中的 **FP8 RL** 仍会跳转到 kimiqwen-next **UD quant waitlist** 注册页面。
   - 一位用户在发现仅更改了学习率后，开玩笑说这是*更高层次的东西*。
- **量化模型加速推理**：为了实现**快速推理**，建议用户运行**量化模型**，首选 Hugging Face 上的 **Unsloth Dynamic Quantized models**，将 **kv cache 设置为 8bit**，并针对所需的量化优化其 **GPU**。
   - 运行 **vLLM**、**SGLang** 或 **LM Studio** 也被认为是运行 GGUF 文件的可行替代方案。
- **再见 Kernels**：尽管有用户询问 **AI** 还需要多久才能编写高质量的 **GPU kernels**，团队表示由于 [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) 的存在，现在已经不再需要 kernels 了。
   - 据说**数学算法**现在是最重要的，认为需要编写 kernel 是一个常见的误区；这部分工作已经转移到了 **help**。
- **ERNIE AI 开发者挑战赛发布！**：Unsloth 正在支持 **ERNIE AI 开发者挑战赛**，为微调 **ERNIE** 并构建最具影响力模型的开发者提供 **$3,000** 奖金。
   - 详情可见 [Baidu Ernie AI Devpost 链接](https://baiduernieai.devpost.com/) 以及 [X 帖子链接](https://x.com/ErnieforDevs/status/1993666389178204434) 中的官方 Ernie 微调 Notebook（AMD 版本是免费的）。
- **Unsloth 将参加圣迭戈的 NeurIPS**：Unsloth 将参加 **NeurIPS San Diego 2025** 并带去限量周边，**12 月 2 日周二下午 4 点**将与 **OpenEnv** 进行 **Agentic AI / RL Panel** 讨论，**12 月 3 日周三下午 6 点**将举行**开源 AI 招待会**。
   - 团队提供了[注册链接](https://linuxfoundation.regfox.com/open-source-ai-reception-2025)，并提醒用户找他们交流 **RL 观点**。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1442970987791388683)** (173 messages🔥🔥): 

> `Claude Opus 4.5, wakeword 解决方案, MS 或 PhD 面试, 长上下文训练, 人形机器人耐力` 


- **Opus 出现上下文错误**: 成员报告 **Claude Opus 4.5** 在处理 100 行代码 + 200 行 yaml 文件时报错，错误信息为 *im sorry this is beyond my context limits. Im going to XYZ*。
   - 随后一名成员询问是否有适用于浏览器或 **python** 的优秀 **wakeword** 解决方案。
- **工作面试：必须要有 MS 或 PhD 吗？**: 一位成员分享说，尽管职位要求写着需要 MS 或 PhD，但他们还是获得了面试机会。
   - 其他人鼓励他们，解释说公司会筛选掉一部分人，而 *重要的是你是谁以及你能带来什么，在面试中做真实的自己并保持真诚即可。*
- **使用 CPU offloading 训练模型**: 一位成员正在使用基于 Unsloth 构建的训练框架训练模型，并询问如果向模型添加 adapter，是否意味着 adapter + 模型都会进入内存，从而消耗更多 VRAM？
   - 另一位成员提供了 [Unsloth Long Context 博客文章](https://unsloth.ai/blog/long-context) 的链接，并解释说 LoRA 的意义在于避免更新所有参数。
- **人形机器人耐力**: 一位成员问道：*如果你要建造一个仿人机器人，你会如何考虑耐力和其他类似的“人类”参数？目前的平衡技术是否可能像生物体一样高效地将食物转化为三磷酸腺苷（ATP）再转化为电能？*
   - 另一位成员回答道：*绝大多数技术都已经存在，但还没有以这种方式整合在一起 / 而且极其昂贵，可能耗资数亿。*
- **Kagi 发布 Slop Detective 游戏**: 一位成员分享了 [Slop Detective](https://slopdetective.kagi.com/)，这是 Kagi 推出的一款新游戏，并评论道 *是的，让我们打败它们，呃！😠 哈哈*。
   - 其他成员发现其中的例子很“离谱”，判定标准是 *错误 = AI，正确 = 人类*，但有人反驳说 *很多真人写的文本也充满了错误*。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1442983801084055634)** (103 messages🔥🔥): 

> `IPEX vs llama.cpp Vulkan, HF 模型转 GGUF, 持续预训练 vs 微调, Qwen3 8B 微调问题, bitsandbytes 的 AMD GPU 支持` 


- **对于 Llama.cpp，Vulkan > IPEX**: 用户建议使用常规的 **llama.cpp Vulkan** 版本而不是 **IPEX**，因为后者存在稳定性问题，尽管 SYCL 可能会提供稍好一点的性能。
   - 有人提到 **IPEX** 的构建版本 *非常陈旧*。
- **GGUF 转换中再次出现 `model_type` 属性问题**: 一位用户在使用 `llama.cpp` 的 `convert_hf_to_gguf.py` 脚本将 HF 模型 (**Unsloth/Qwen2.5-7B-Instruct**) 转换为 GGUF 时，遇到了 `AttributeError: 'dict' object has no attribute 'model_type'` 错误，这可能是由于文件结构问题导致的。
   - 另一位用户分享了一个合并后的 Qwen3 模型的目录结构作为参考。
- **Base 模型在自动补全任务中占据主导地位**: 对于训练模型生成类似数据（自动补全）而非问答行为的任务，建议从 **base model**（非指令微调版）开始并进行 **continued pretraining**。
   - 建议使用 **Gemma-3-270M** 模型进行实验，并附上了 [Unsloth 关于持续预训练的文档](https://docs.unsloth.ai/basics/continued-pretraining) 链接。
- **Qwen3 8B 微调表现不佳**: 一位用户在微调 **Qwen3 8B** 后评估结果很差，回答与微调数据无关，并且即使将 prompt 设置为 false，模型仍然会输出 `thinking` 提示词。
   - 建议如果 LM Studio 重现了该问题，可以尝试手动合并并保存，参考 [Unsloth 文档](https://docs.unsloth.ai/basics/inference-and-deployment/saving-to-gguf#manual-saving)。
- **vLLM 更新为 AMD GPU 带来 Bitsandbytes 提升**: AMD 文档即将更新，以反映对 Radeon GPU 上 **Bitsandbytes 4bit 量化模型** 和 **QLoRA** 的支持。
   - 相关更改已在 [bitsandbytes-foundation/bitsandbytes#1748](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1748) 和 [vllm-project/vllm#27307](https://github.com/vllm-project/vllm/pull/27307) 中实现。


  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1443274834170744853)** (2 messages): 

> `ERNIE AI Developer Challenge, Baidu ERNIE, Unsloth finetuning, AMD notebooks` 


- **ERNIE AI 开发者挑战赛启动**：Unsloth 宣布支持 **ERNIE AI Developer Challenge**，提供使用 Unsloth 微调 **ERNIE** 的机会并有机会赢取奖品。
   - 竞赛详情请访问 [baiduernieai.devpost.com](https://baiduernieai.devpost.com/)。
- **Unsloth 为 ERNIE 提供微调福利**：官方 **ERNIE** 微调 notebook 已上线，包括针对 AMD 平台的免费版本。
   - 访问 [X.com](https://x.com/ErnieforDevs/status/1993666389178204434) 查看公告以获取 **AMD notebooks**。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1443145738090709032)** (12 messages🔥): 

> `Evolutionary Strategies at Scale, LESA: Learnable LLM Layer Scaling-Up, Efficient Training on CPU` 


- **ES HyperScale 提升训练吞吐量**：一名成员分享了 [ES HyperScale](https://eshyperscale.github.io/)，它在大种群规模下，针对十亿参数模型的训练吞吐量比标准 ES 实现了**百倍增长**，能够在任何模型上进行更灵活的训练，无需担心梯度（Gradients）问题，并支持 int8。
   - 另一位成员幽默地评论道：*“以 100 倍速度训练？那相当于 Unsloth x 50 了”*。
- **通过 LESA 实现可学习的 LLM 层级扩展**：一名成员发布了 [LESA: Learnable LLM Layer Scaling-Up](https://arxiv.org/pdf/2511.16664v1)，认为*某种（嵌套“弹性” MoE）+（多 Token 预测/multi-token prediction）将为单批次推理吞吐量带来疯狂的飞跃*。
   - 该论文介绍了 **LESA**，它使用神经网络预测插入在相邻层之间的参数，从而实现更好的初始化和更快的训练。
- **高效的 CPU 训练现已成为现实**：一名成员强调，通过 [ES HyperScale](https://eshyperscale.github.io/) 可以实现在 CPU 上进行实际有效的高效训练，支持在任何模型上灵活训练，无需担心梯度，并支持 int8。
   - 它被描述为 *“在任何模型上进行更灵活的训练。无需担心梯度的训练。支持 int8 的训练！真正高效的 CPU 训练”*。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1442968521553219684)** (371 messages🔥🔥): 

> `Haiku documentation accuracy, Cursor agent's plan markdown storage, Free Agent Review, Education discounts` 


- **用于文档的 Haiku 模型**：成员们发现配合文档使用的 **Haiku** 准确率达到 100%，**Composer-1** 最适合代码实现，而 **Haiku** 在快速文档检索方面占据统治地位。
   - 一位成员还建议使用 [Antigravity](https://antigravity.ai/)，而不是在仓库中塞满 Markdown 报告，尽管这可能会导致交接问题。
- **用户讨论 Token 成本和模型使用情况**：一些用户报告 **Opus** 模型过载的问题，另一些人则表示该模型已降级，表现怪异且不够聪明。
   - 一些人争论是开启按需付费（on-demand）还是直接购买 Pro+ 方案，讨论是否应该直接使用 **Auto** 模式“烧掉” Token 而不考虑 Token 效率。
- **Agent Review 免费了？？**：用户注意到 *Agent Review* 可能在旧定价体系下是免费的，但在新定价体系下已不再提供。
   - 还有人想知道 Teams 方案是否包含无限次的 bugbot，因为在仪表盘上看到了 *unlimited bugbot*。
- **用户对 Cursor 中的 Linting 错误感到沮丧**：一位用户寻求帮助，希望在保留其他错误提示的同时禁用 Linting 检查的红色波浪线，并允许扩展程序在保存文件时在后台运行 `--fix`。
   - 该用户表达了对 **Cursor** 中实现这一点如此困难的沮丧，因为这在 JetBrains 的工具中非常简单。
- **Cursor 未保存 Agent 计划**：一位用户询问 Agent 计划的 Markdown 文件保存在哪里，以便在更换电脑时不会丢失计划。
   - 社区成员表示 **Cursor** 不保存该计划，因此需要手动保存 Markdown，并创建一个规则将所有计划添加到一个目录中。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1443038489829572690)** (7 messages): 

> `Triton Kernels, Partially Trainable Embedding, Logits Softmax Operation, Curriculum Learning` 


- **寻求通过 Triton Kernels 获得前沿级的效率提升**：一位成员正在寻求关于使用 **Triton kernels** 解决涉及 *Partially Trainable Embedding*（部分可训练嵌入）和 *Logits Softmax* 操作的独特挑战的建议，旨在获得前沿级的效率提升。
   - 目标是在训练大模型时冻结其大部分参数，仅高效地专注于特定的 special tokens。由于低效的 tiling（分块）和重复的数据检索导致的内存受限（memory bounding），最初使用 Claude 的尝试结果较慢。
- **需要 Partially Trainable Embeddings 以节省内存**：一位成员希望实现一种 *Partially Trainable Embedding*，其中只有特定索引范围以上的行是可训练的，例如 **128k 词汇表中的 1k 行（127k 到 128k）**。
   - 这样做的目的是通过仅存储可训练行的梯度输出来减少内存占用，同时也为了在仅训练特定 special tokens 时冻结模型的大部分。
- **带有 Logits Softmax 的加权损失**：一位成员正在寻求实现一种允许应用加权损失的 *logits softmax operation*，例如 **pos 123 的 token** 具有 **0.5x 损失乘数**，而 **pos 124 的 token** 具有 **1.5x 损失乘数**。
   - 目标是通过使用 chunking（分块）或 CCE 方法来避免实例化（materializing）所有的 logits，并且必须能够与自定义的 partially trainable embedding 配合使用。
- **AI 实验室通常使用 Curriculum Learning**：一位成员询问 AI 实验室在预训练 LLM 时是否真的会使用 *curriculum learning*（课程学习）和 *coreset* 之类的方法。
   - 另一位成员回答道：“我不确定你指的 coreset 是什么，但课程学习在预训练中确实非常普遍。”


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1443070482697027595)** (5 messages): 

> `Proton vs Nsight Systems, Tensor Descriptors, Auto Tune Parameters, Tritonparse, Persistent Matmul Tutorial` 


- **Proton 性能分析工具故障**：一位用户询问关于使用 **Proton** 进行 profiling（性能分析）的问题，指出在生成文档所述的 chrome traces 时出现错误，并询问其他人是否更倾向于使用 **Nsight Systems**。
   - 后续讨论指向了 **persistent matmul tutorial**，作为将 mnk 用作 autotune keys 的示例。
- **自动调优参数探索**：一位正在努力刷 leetcode 的成员对 **tensor descriptors**（张量描述符）或 **auto-tune parameters**（自动调优参数）以专门化形状表示了兴趣。
   - 他们还感谢了另一位成员推荐 **Tritonparse** 作为一个有用的工具。
- **Persistent Matmul 教程**：一位成员建议 [persistent matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html#sphx-glr-getting-started-tutorials-09-persistent-matmul-py) 是将 **mnk** 用作 autotune keys 的一个例子。
   - 该教程引导用户使用 shared memory（共享内存）和 persistent kernels（持久化内核）来优化矩阵乘法，提供了 **Triton** 中 autotuning 的实际案例。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1442978296202657833)** (17 条消息🔥): 

> `使用 Tensor Cores 的 GEMM，NVIDIA Tensor Cores 性能优化资源，BF16 矩阵乘法，CUDA 实现细节，矩阵数据加载策略` 


- ****GEMM** 实现探索**：一名成员正在探索使用 Tensor Cores 实现 **GEMM**（通用矩阵乘法），并寻求关于在矩阵 **A**、**B** 和 **C** 中使用 **BF16** 以及使用 `float` 累加器的建议，参考了 [Lei Mao 的教程](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/)。
   - 该成员在使用 `load_matrix_sync` 加载矩阵 **C** 元素并将其转换为 `float` 时面临挑战，质疑 **C** 最初是否应该是 `float` 矩阵。
- **Tensor Core 优化宝库揭秘**：成员们分享了 **NVIDIA Tensor Cores** 性能优化的资源，指向了类似的文章和工作日志，例如 [alexarmbr 的工作](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html) 和 [cudaforfun 的工作日志](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog)。
   - 有人强调 **GPU-MODE** 有关于 Hopper **GEMM** 工作日志的讲座。
- **数据加载难题解析**：一位成员解释说 `ldmatrix.b16` 每个线程加载 **128 bits** 数据而无需额外操作，建议使用 `reinterpret_cast` 进行正确的数据处理。
   - 另一位成员澄清说，当使用 `f32` 或 `s32` 累加器时，每个线程持有一行中一对连续的值（**8 bytes**），而 `ldmatrix.b16` 将一行拆分为 **4B** 块并分布在四个线程（quad of threads）上，建议使用 `float2` 或在加载时重新排序 **B** 矩阵列。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1443191272302055504)** (3 条消息): 

> `Gradient Checkpointing, Torch Differentiation, Boolean Flagging` 


- **寻找区分前向传播的 Torch 函数**：一位成员询问是否有 **torch 函数** 可以区分前向传播是在开启还是关闭 **gradient checkpointing** 的情况下运行的。
   - 该成员还询问是否有办法区分这两次前向传播。
- **利用布尔标志区分前向传播**：一位成员建议使用 **布尔标志 (boolean flag)** 来解决区分两次前向传播的问题。
   - 该成员提议在每次前向传播中交替变换该标志。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1443008458713469058)** (13 条消息🔥): 

> `贡献 XLA，GPU/CUDA 基准测试预热运行，影响预热时间的 Kernel 特性，基准测试中的热限制，nvbench 热状态` 


- **寻求贡献 XLA 途径的贡献者**：一位成员询问如何为 **XLA** 做出贡献并寻求入门指导，初步兴趣在于 **文档支持**。
- **GPU 预热运行的经验法则**：一位成员询问 **GPU/CUDA 基准测试 (benchmarking)** 中 **预热运行 (warmup runs)** 次数的良好经验法则。
   - 另一位成员回答说，没有具体的数字标准；相反，他们会重复测量，直到连续运行的结果不再发生显著变化。
- **热限制影响长时 GPU 运行**：成员们提到，要对运行很长时间的应用程序进行稳态性能基准测试，必须考虑 **功耗** 和 **热限制 (thermal limits)**。
   - 你必须字面意义上地让 GPU 预热以达到稳定温度（这可能需要几十秒到几分钟）。
- **数据中心设置缓解热因素**：一位成员询问 **数据中心设置** 是否能缓解热因素，另一位成员回答说，根据上下文，这种稳态可能不是正确答案。
   - 他们还提供了一个关于 nvbench 的 [YouTube 视频](https://www.youtube.com/watch?v=CtrqBmYtSEk) 链接，该工具旨在获得跨非降频热状态的良好平均值。


  

---


### **GPU MODE ▷ #[jax-pallas-mosaic](https://discord.com/channels/1189498204333543425/1203956655570817034/1443228985143328778)** (2 条消息): 

> `jax.pmap vs 单 GPU 上的 jitting，多 GPU vs 单 GPU 系统` 


- **单 GPU 上 `jax.pmap` 与 `jit` 的性能对比**：一位用户询问在单个设备上使用 `jax.pmap` 与通过 `jax.jit` 直接进行 jitting 相比有哪些缺点。
- **多 GPU 与单 GPU 系统上的代码可移植性**：该用户正在编写旨在同时运行在多 GPU 和单 GPU 系统上的代码，并考虑即使在只有单个 GPU 的情况下也使用 `jax.pmap` 以简化代码库。


  

---

### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1443268597735624804)** (1 messages): 

> `Memes` 


- **今日 Meme 送达**：一位用户发布了一个 Meme。
   - 该 Meme 可以在[这里](https://cdn.discordapp.com/attachments/1215328286503075953/1443268597483831326/1764108026112.jpeg?ex=69287420&is=692722a0&hm=d4747dea6327a6024b1c84c59c77525ee94bc0392191114d5b49c98d00bd1cd4&)找到。
- **另一个 Meme 出现！**：频道中发布了另一个 Meme 供大家娱乐。
   - 这个 Meme 丰富了社区内持续分享的幽默收藏。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

szymonoz: 我将参加 NeurIPS，之后会去 SF，如果想聊聊 GPU，请联系我 😄
  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1443076921348067369)** (1 messages): 

> `2bit Dequantization on Intel GPU, GPU Dequantization Methods, Torch Performance on Intel GPU` 


- **Intel GPU 上的 2-bit Dequantization 探索**：一位用户询问了直接在 **Intel GPU** 上执行 **2-bit Dequantization** 的方法，并指出虽然可以在 CPU 上进行量化，但使用 **Torch** 进行 Dequantization 速度较慢。
   - 该用户正在寻找一种比 **Torch** 更快、基于 GPU 的 Dequantization 替代方案以提升性能，这说明了在该领域对优化的 **Intel GPU** 解决方案的需求。
- **寻求高速 GPU Dequantization**：原帖作者正在寻求优化的基于 **GPU** 的 **Torch** 替代方案来进行 Dequantization，以提高性能。
   - 目前没有其他讨论可以总结，这在频道中仍是一个开放性问题。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/)** (1 messages): 

aerlabs: https://x.com/aerlabs_/status/1993561244196868370
  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1443365307639926834)** (1 messages): 

> `LLM initiatives, LLM Kernel Generation, Agentic Systems` 


- **Urmish 加入 LLM 倡议**：Urmish 介绍了自己，表达了参与 **LLM 倡议** 的兴趣，并强调了在 **pre-training, post-training, evaluation, agentic systems 和 dataset creation** 方面的经验，并提供了 [Google Scholar 个人资料](https://scholar.google.com/citations?hl=en&user=-GPPICQAAAAJ&view_op=list_works&sortby=pubdate)。
   - 凭借在系统和性能工程方面的背景（包括为 **microcontrollers, HPC 和 CPUs 编写 kernel**），他们正在寻求从何处开始的指导，并询问了专注于 LLM 训练、Prompting 或用于 **LLM Kernel Generation** 的 Agentic Harnesses 的子小组。
- **LLM Kernel 期待萌芽**：Urmish 询问了现有的子小组，以便更好地将精力集中在 **LLM Kernel Generation**、**LLM training** 和 **Agentic Harnesses** 上。
   - 他们希望利用先前的经验来帮助社区。


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1443116910718881822)** (10 messages🔥): 

> `CUDA kernels, Flash Attention, MoE kernels, Linear Attention backwards, FFT conv backwards` 


- **新人开拓 CUDA 和 Flash Attention**：一位新社区成员表达了他们在编写 **CUDA kernels** 和处理 **Flash Attention** 方面的经验。
   - 另一位成员鼓励他们通过 **PR** 进行回馈。
- **Kernel 贡献在 ThunderKittens 中蓬勃发展**：成员们讨论了待开发的开放领域，包括 **MoE kernels**、**linear attention backwards**、**FFT conv backwards** 以及与 **inference engines** 的集成。
   - 他们还提到，欢迎社区贡献 **Pythonic wrapper 探索/工具** 以简化开发，以及集成轻量级编译器传递（compiler passes）的工具。
- **AMD GPU 可用性引发辩论**：一位成员询问贡献是针对 **main branch CDNA4 还是 CDNA3**，并指出很难找到 **AMD GPU** 供应商来构建和测试这些内容。
   - 另一位成员澄清说两者都有，但最初的问题是关于 TK 的。


  

---

### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1442968122725236841)** (114 messages🔥🔥): 

> `NVIDIA leaderboard submissions, nvfp4_gemv leaderboard, Personal bests, Successful submissions` 


- **NVIDIA 的 nvfp4_gemv 排行榜：提交热潮！**：NVIDIA 上的 `nvfp4_gemv` 排行榜出现了一波活跃的提交，来自多位用户的贡献，包括 <@242385366873669632>, <@393188835472834560>, <@651556217315000360>, <@418996736405536790>, <@1035498877249409155>, <@1295117064738181173>, <@376454672799760384>, <@96782791567503360>, <@264466949331746826>, <@1178719962597183529>, <@434046629281267744>, <@1291326123182919753>, 和 <@120261963551866881>。
   - 提交内容包括“Personal best（个人最佳）”和“Successful on NVIDIA（在 NVIDIA 上成功运行）”的结果，显示出积极的优化和测试努力。
- **登上领奖台：获得第二名**：<@1035498877249409155> 在 NVIDIA 上以 **3.02 µs** 的成绩获得了**第二名**，随后在 `nvfp4_gemv` 排行榜上又以 **15.8 µs** 再次获得第二名。
   - 讨论中提到 <@1035498877249409155> 的一次提交可能存在疑点，<@264466949331746826> 计划复核结果，并提到：*“我在一些技巧的指导下，让 Opus 4.5 全权发挥了作用”*。
- **优化竞赛：新的个人最佳成绩公布**：多位用户，包括 <@242385366873669632>, <@393188835472834560>, <@1295117064738181173>, <@120261963551866881>, <@434046629281267744>, <@1035498877249409155>, <@1291326123182919753> 和 <@651556217315000360>，在 NVIDIA 的 `nvfp4_gemv` 排行榜上持续提交“个人最佳”成绩。
   - 这表明大家正在不断努力优化性能并实现更快的执行时间，此外 <@376454672799760384> 的提交最佳成绩为 **144 µs**。
- **进入前 10 名：用户占据领先位置**：<@1295117064738181173> 以 **22.7 µs** 获得 **第 8 名**，随后以 **22.5 µs** 升至 **第 7 名**；<@1035498877249409155> 在 NVIDIA 上以 **23.2 µs** 获得 **第 9 名**。
   - <@1178719962597183529> 以 **23.3 µs** 达到 **第 9 名**，<@1295117064738181173> 以 **22.9 µs** 达到 **第 7 名**。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1443303382357119098)** (3 messages): 

> `Factorio Learning Environment Docs, Jack Hopkins, Github Pages` 


- **Hopkins 的热线：Factorio 文档已部署！**：Jack Hopkins 宣布 **Factorio Learning Environment** 的文档现已上线，地址为 [Factorio Learning Environment](https://jackhopkins.github.io/factorio-learning-environment/sphinx/build/html/index.html)。
- **Noddybear 为 Hopkins 的文档点赞**：Noddybear 对新 Factorio 文档的发布表示支持。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1443261622033322216)** (2 messages): 

> `SIMT loads, Tiled_mma documentation` 


- **SIMT 加载开销**：SIMT 加载存在开销，因此*仅在 TMA 限制过多时才使用它们*。
- **Tiled_mma 示例拆解**：一位工程师正尝试参考 **hopper gemm cute dsl** 示例来使用 *tiled_mma*。
   - 他们将 **sa** 按 **(2, 4)** 进行分块（tiling），`tCsA: ((64,16),2,4,(1,1)):((64,1),4096,16,(0,0))` 表示 *mma atom tile (64, 256)，沿 M 方向 2 个分块，沿 K 方向 4 个分块*。


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1443312521716039854)** (3 messages): 

> `picograd, aten-like Op intermediate representation, Device runtimes` 


- **Picograd 的最新提交**：用户分享了 [picograd repo](https://github.com/j4orz/picograd) 的一系列最新提交，突出了正在进行的开发工作。
   - 这些提交涵盖了多个方面，包括包级文档、Tensor 实现、评估器设计以及设备运行时（Device runtimes）。
- **Picograd 的 Tensor 实现**：用户链接到了 picograd 的 `Tensor` 实现，它会脱糖（desugars）为**类似 ATen 的 `Op` 中间表示 (IR)** [(链接)](https://github.com/j4orz/picograd/blob/master/python/picograd/tensor.py)。
   - 目标是为自动微分和 GPU 加速提供基础。
- **Picograd 的评估器和设备运行时**：用户重点介绍了使用 `Device` 运行时的 `evaluator(op: Op)` 解释器 [(链接)](https://github.com/j4orz/picograd/blob/master/python/picograd/engine/evaluator.py)，以及提供内存分配器和 Kernel 编译器的 `Device` 运行时本身 [(链接)](https://github.com/j4orz/picograd/blob/master/python/picograd/device.py)。
   - 用户提到语言和运行时很快就会很好地结合在一起，为跨架构迁移铺平道路。


  

---

### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1443297762811576440)** (3 messages): 

> `LLM Inference, NVRAR algorithm, PAT Algorithm, Bruck algorithm, Recursive doubling algorithm` 


- **NVRAR 加速多节点 LLM Inference**：论文 [LLM Inference Beyond a Single Node](https://arxiv.org/abs/2511.09557) 介绍了 **NVRAR**，这是一种基于 NVSHMEM 的递归倍增（Recursive doubling）分层 All-Reduce 算法。在 **128 KB 到 2 MB** 的消息大小下，其延迟比 NCCL **低 1.9x-3.6x**。
   - **NVRAR** 已集成到 YALIS 中。在使用张量并行（Tensor Parallelism）的多节点重解码（decode-heavy）工作负载中，它使 **Llama 3.1 405B 模型** 的端到端 Batch 延迟降低了 **高达 1.72x**。
- **用于 All-Gather 和 Reduce-Scatter 操作的 PAT 算法**：论文 [PAT: a new algorithm for all-gather and reduce-scatter operations at scale](https://arxiv.org/pdf/2506.20252v1) 讨论了 **Bruck** 和 **Recursive doubling 算法** 在实践中的缺陷，因为它们的最后一步需要向距离较远的 Rank 传输大量数据。
   - 在最后一步中，每个 Rank 都会将总大小的一半发送到与其距离最远的 Rank。在大型 Fabric 网络上，由于静态路由或 Fabric 高层级的收敛（Tapered），最后一步的运行速度往往比理论值慢许多倍。


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1442975553492750386)** (159 messages🔥🔥): 

> `CuTeDSL packed FP16, eval.py issues, cudaStreamSynchronize(), LLM-only challenges, sfa_permuted purpose` 


- **CuTeDSL 获得 packed FP16 指令支持**：成员们提供了 [代码](https://github.com/gpu-mode/reference-kernels) 以在 CuTeDSL 中使用 packed FP16 指令，因为标准的 CuTeDSL 并不通过 NVVM 提供这些指令。
- **Eval 脚本面临审查**：用户报告称 GPU MODE 竞赛中的 `eval.py` 脚本可能会产生波动极大的结果，即使多次上传相同的脚本，耗时差异也高达 50%。有人推测是 ID 为 **105881** 的 Runner 运行缓慢。
   - 脚本的不稳定性引发了对排行榜计时准确性和可靠性的担忧，建议的提交阈值为 **25**。
- **Stream 增加开销**：一位成员发现尝试使用多 Stream 会导致同步问题，并指出 `cudaStreamSynchronize()` 在实现良好的方案中会增加巨大的开销。
   - 另一位成员指出，Event 会增加约 **4 us** 的测量开销。
- **探索 LLM-Only 方法**：一些参赛者正在尝试“仅 LLM”的方法，使用 **Gemini 3.5 Pro** 和 **Opus 4.5** 等模型来生成代码，但不同人对 LLM 的引导程度有所不同。
   - 一位用户指出：*Gemini 3.5 Pro 和 Opus 4.5 是彻底的游戏规则改变者……它们让 GPT-5.1 看起来像 Llama-7b*。
- **破解 sfa_permuted 的奥秘**：一位用户终于意识到 **sfa_permuted** 的目的是为了配合 `tcgen` 指令，这使得构建具有该 Layout 的内容变得更加容易。


  

---


### **GPU MODE ▷ #[hf-kernels](https://discord.com/channels/1189498204333543425/1435311035253915840/1443041374185328751)** (5 messages): 

> `Metal Kernels Release, MacOS Compatibility Issues` 


- **Metal Kernels 推迟**：一位成员询问了 **Metal Kernels** 的发布情况。
   - 目前尚未给出发布日期。
- **MacOS 兼容性受限**：一位成员质疑为什么 [kernel-builder](https://github.com/huggingface/kernel-builder/blob/main/docs/metal.md) 仅支持 **macOS 26**，这降低了对 **M1** 芯片和旧版本 macOS 的兼容性。
   - 该成员表示 *不理解为什么针对 Apple Torch 生态系统所做的一切工作，最终都以一种让情况变得更糟的方式呈现*。


  

---

### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1443019750757236738)** (8 条消息🔥): 

> `7x 叠衣服机器人，无动作过滤 (No-Action Filtering)，Qwen3-VL 优化，经典分箱 (Classic Binning) vs FAST Tokenizer` 


- **7x 叠衣服机器人亮相！**：**7x** 推出了一款售价 **3k** 的叠衣服双臂系统，正如其 [YouTube 频道](https://www.youtube.com/@usmanroshan8740) 所示，展现出“低成本机器人”的风格，并由创始人和工程师提供 **24 小时支持**。
   - 成员们对机械臂在实际工作中的耐用性表示怀疑，一位成员将其支持模式与 *Google Robotics* 进行了对比。
- **无动作过滤 (No-Action Filtering) 对 VLA 至关重要**：一位成员了解到 **no-action filtering** 对 VLA 非常重要，并通过 [视觉对比](https://cdn.discordapp.com/attachments/1437390897552818186/1443153483556716666/image.png?ex=6928b1aa&is=6927602a&hm=bf26405dca31cd342d33762114dc18ad626339bf92ccb31ee0cb0c1eb501087e) 展示了无空闲过滤器 (no-idle filter) 与带空闲过滤器 (with-idle filter) 之间的区别。
   - 一张说明 **空闲帧分析 (idle frame analysis)** 影响的图片显示，活跃帧占总分析帧数的 **78.8%**。
- **Qwen3-VL 的优化障碍**：一个 **2B 模型** 感觉很慢，特别是在推理过程中，导致其无法运行 RL。一位成员计划研究针对 **Qwen3-VL** 优化的前向传播 (forward passes)。
   - 未提供更多细节。
- **Tokenizer 对决：经典分箱 (Classic Binning) vs FAST**：成员们正在测试 **classic binning** 与 **FAST tokenizer**，但 **FAST (DCT+BPE)** 生成的复杂压缩 Token 可能会延迟模型生成可靠有效序列的能力。
   - 发布者怀疑这是否能成为 RL 的良好基础，因此他们同时在尝试一种更简单的变体，具有解耦关节 (disentangled joints) 和简单量化。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1442968425033892042)** (263 条消息🔥🔥): 

> `ChatGPT 偏见，Nano Banana Pro，AI 生成图片的商业用途，GPT 5.0 mini，OpenAI UI 设计` 


- **ChatGPT 被指偏向左翼数据**：成员们讨论了 **ChatGPT** 是否在自由派和政治左翼数据上进行训练，潜在原因包括训练数据中的进步观点以及 **Reinforcement Learning with Human Feedback (RLHF)** 中人类评分者的偏见。
   - 一位成员认为，模型在回答问题时“谨小慎微 (fussy foot around questions)”的需求损害了其可靠性。
- **Nano Banana Pro 释放漫画创作力**：用户正在使用 **Nano Banana Pro** 创作漫画，称赞其强大的功能、快速生成图像的能力以及高质量的结果，并对其在生成 [漫画页面](https://cdn.discordapp.com/attachments/998381918976479273/1443038766087536751/image.png?ex=6928ef94&is=69279e14&hm=d88b0f693975c1e756c3352c689566bda68503635084432e547cc6585d126e83&) 方面的易用性感到兴奋。
   - 成员们表达了对模型被“脑叶切除 (lobotomized)”（功能阉割）的担忧。
- **AI 艺术引发商业版权和伦理困境**：成员们辩论了使用 **Gemini** 生成的 AI 图像的商业可行性和版权影响，指出虽然 Google 没有明确禁止商业用途，但法律地位取决于内容是否具有版权，而 AI 艺术中的文化偏见是一个重大关注点。
   - 一位成员表示，“如果反 AI 的人想做点什么，他们应该开始动笔画画和创作艺术”。
- **GPT-5.0 Mini 感觉像是降级**：成员们对 **GPT-5.0 Mini** 表示不满，称其为“降级”。
   - 他们对那些还没用过就不断乞求 **Sora 2** 的行为感到厌烦。
- **OpenAI 的 UI/UX 迎合神经典型人群**：一位成员认为 **OpenAI's UI** 不是为神经多样性 (neurodivergent) 思考者设计的，步骤过多且不适合思维复杂的人。
   - 频道中的其他人则认为 **UI 对每个人都很糟糕**，并且是专门为执行功能障碍 (executive dysfunction) 人群设计的，尤其是 Mac 应用。


  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1443016278356000880)** (10 条消息🔥): 

> `GPT 5.1, GPT 4.1, Chat reference memory, 动漫写作` 


- **用户赞扬 GPT 5.1 的动漫叙事能力**：一位用户强调 **GPT 5.1** 是目前最适合动漫或故事写作的模型，因为它能够记住角色设计和之前的上下文。
   - 唯一的抱怨是严格的 **safety net** 和 **guardrails** 阻止了动漫风格暴力内容的创作。该用户分享说他们已经使用 **GPT 4.1** 一年了，但有时它会遗漏角色设计。
- **关于 Chat Reference Memory 问题的讨论**：一位用户询问是否还有其他人遇到 **GPT 模型** 中 **chat reference memory** 的问题。
   - 另一位用户提出了 **GPT 5.1** 是否优于 **GPT 4** 的问题，并认为这取决于具体的用例。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 条消息): 

mx_fuser: <@1256251788454268953>
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 条消息): 

mx_fuser: <@1256251788454268953>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1442974507999695070)** (46 条消息🔥): 

> `LM Studio 中不支持的 API Endpoints, LM Studio 的图像字幕问题, Vision Models, 针对 RDNA 3 的 ROCm 7 更新, 与 OpenSea 合作的 Mint 机会` 


- **API Endpoint 故障排除解决问题**：一位用户在本地服务器上遇到了不支持的 **endpoints** (POST /api/v1/generate) 错误，但在频道发帖后自行解决了。
   - 用户被引导至 [LM Studio REST API 文档](https://lmstudio.ai/docs/developer/rest/endpoints)，并意识到该 endpoint 是无效的。
- **Channel Error 破坏了图像字幕功能**：一位用户报告在尝试使用 **LM Studio** 为图像生成字幕时出现 **“Channel Error”**，在 Windows 和杀毒软件更新后失败率为 100%，尽管之前可以正常工作。
   - 用户从 **llava-v1.6-34b** 切换到 **Gemma 3**，解决了问题并达到了 100% 的成功率；建议认为这可能与模型相关，或者问题可能与默认启用的 **Flash Attention** 有关。
- **部分模型中的 Flash Attention 故障**：有人建议该用户的问题可能与 **Flash Attention** 有关，该功能在最近的 **LM Studio** 版本中默认启用，可能导致某些模型无法正常运行。
   - 鼓励用户分享其运行时视图的截图，并检查非英语输入/输出，同时建议运行 `lms log stream` 以获取更详细的错误消息。
- **GPT OSS 20B 速度惊人**：一位用户分享了一张展示 **gpt-oss-20b** 模型速度的图片，并链接到了一个 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1p7ghyn/why_its_getting_worse_for_everyone_the_recent/)，提到该帖子中的信息可能会引起频道中一些人的共鸣。 
- **Mint 机会让用户投入 OpenSea**：一位用户宣布了一个与 **OpenSea** 合作的免费 **Mint 机会**，邀请成员通过提供的链接参与。
   - 另一位用户迅速指出，由于详细解释的原因，给出的邀请在真实的学术环境中会失败，并指出了人类对作品的评价方式与机器人评价方式之间的差异。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1442978690236420338)** (217 条消息🔥🔥): 

> `Q8 Cache, 推理期间 GPU 风扇停转, 内存定价问题, DLSS 与 RT 测试, 硬件贬值` 


- **Q8 Cache 配置难题**：成员们讨论了 **Q8 cache** 的使用，其中一位提到特定用户 (*<@96768590291664896>*) 知道如何解释为什么 **Q6 KV** 的位数对不上。
- **推理期间 GPU 风扇停转**：一位用户注意到他们的 **GPU 风扇** 在推理期间转速为 **0%**，起初感到担忧，但随后澄清这是他们的 **MI50** 以及有时 **4070 TiS** 的正常行为。
   - 该用户指出，一旦 Context 完全写入，GPU 就会“接管”且功耗会增加。
- **硬件贬值辩论**：一位用户分享了他们 Windows 启动进入恢复模式的照片，开玩笑说 **850W 电源** 终究没被烧坏，称之为“一种进步”。
   - 该用户最初怀疑是电源问题，但后来怀疑是 CPU 的硅脂出了问题。
- **潜在的 CPU 起火危机化解？**：用户们警告不要烧毁组件，并建议在廉价主板上测试 **CPU** 和 **RAM**，怀疑有起火隐患。
   - 另一位用户发现主板上的 CPU 插槽针脚弯曲，并闻了闻 CPU 是否有烟味，但在清理硅脂后确定一切正常。
- **PCIe 分叉（Bifurcation）技术突破**：一位用户意识到他们的 **X570 AORUS ELITE WiFi** 主板在主 x16 插槽上支持 **PCIe bifurcation**，允许将其拆分为 **8x/8x** 或 **8x/4x/4x** 等配置。
   - 另一位用户补充说，通过分叉技术，当启用 x8x8 模式时，可以使用 [SlimSAS MCIO 适配器](https://www.amazon.com/dp/B0DZG8JVG2) 将 x16 插槽拆分为两个 x8 插槽。


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1442968437281394811)** (2 条消息): 

> `拾色器问题, RapidaAI 开源` 


- **拾色器 Bug 困扰用户**：一位用户报告称，主题调色板覆盖的 **拾色器（color picker）** 有点异常且存在 **偏移**。
- **RapidaAI 宣布开源**：生产级语音 AI 平台 **RapidaAI** 宣布在此发布其 **开源代码** [此处](https://rapida.ai/opensource?ref=openrouter)。
   - 该公司观察到，语音 AI 供应商的账单不断增长，但客户体验却未见改善，公司需要额外支付 **每分钟 $0.05–$0.15** 来租用他人的技术栈（stack），因此他们构建了 Rapida 来颠覆这一模式。


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1442968164982980791)** (196 条消息🔥🔥): 

> `Opus 过载, 模型 Fallback Bug, Deepseek R1 模型下架, Meganova Chat 热议, OpenRouter 定价与功能` 


- **Opus 遭遇过载停机**：用户报告称 **Opus** 在关键时刻再次过载。
   - 一些成员开玩笑说 *“你以为他们会有更好的速率限制/负载均衡，对吧”*，而其他人则表示理解并提到 *“小公司请体谅”*。
- **模型 Fallback 功能遭遇批评**：一位成员报告了 [模型回退逻辑（model fallback logic）](https://openrouter.ai/docs/guides/routing/model-fallbacks) 中的一个 Bug，即主模型的 **404 错误** 导致 Fallback 无法生效，而不是回退到备用模型。
   - 该成员表示 *“我正准备将一个企业级应用迁移到 OpenRouter，这里不容许模型真实性存疑。如果 Fallback 逻辑在如此简单的用例下失效，可能还会存在更多问题”*。
- **免费 Deepseek R1 模型被移除**：成员们注意到免费的 **Deepseek R1** 模型已不再可用。
   - 一位成员对失去该模型表示遗憾 *“这太蠢了。我配合 chutes API key 使用它，因为通过 chutes 使用该模型可以显示思考过程（think process），我离不开它。”*
- **Meganova Chat 引发广泛关注**：成员们讨论了即将推出的 **Meganova Chat**，这是一个用于管理 AI 聊天和角色的平台，一位成员将其描述为一个 *“干净、快速的地方”*。
   - 一位成员回应道 *“我看到很多关于 Meganova Labubu Chat 的正面评价！我正考虑进一步了解它”*，而其他人则对促销信息进行了喜剧化的模仿。
- **OpenRouter 展示基础功能优势**：一位成员强调了 OpenRouter 为各种供应商提供标准化接口的优势。
   - 他们提到，能够 *“瞬间从 GPT 5.1 切换到 Opus 4.5，而无需解析 Anthropic 的所有变更日志，这非常棒”*，尽管 **信用额度购买有 5% 的溢价**。


  

---

### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1443327854355021915)** (2 条消息): 

> `` 


- **未讨论新模型**：在提供的消息中没有关于新模型的讨论或信息。
- **频道指示**：提示表明消息来自 OpenRouter 的 'new-models' 频道，但未包含实际的模型相关内容。


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1443289637945868420)** (5 条消息): 

> `Arrakis AI model, Text-to-Video Leaderboard, Kling 2.5 Turbo, Google Veo 3` 


- **Arrakis AI 看起来仍然偏黄**：一位成员对 [来自 Arrakis AI 的图像](https://x.com/arrakis_ai/status/1993644406159917533) 发表了评论，观察到*它看起来仍然偏黄*。
   - 他们推测*他们只是在将图像发送给客户端之前添加了一个颜色调整层*。
- **文本转视频排行榜诞生新王者**：一位成员分享了 [Artificial Analysis 文本转视频排行榜](https://artificialanalysis.ai/video/leaderboard/text-to-video) 的链接，重点介绍了表现最佳的模型。
   - 排行榜显示 **David** 位居第一，紧随其后的是 **Google 的 Veo 3**，**Kling 2.5 Turbo 1080p** 位列第三。


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1442995229019078800)** (1 条消息): 

> `Psyche Office Hours` 


- **Psyche 团队举行答疑时间 (Office Hours)**：**Psyche** 背后的团队将于下周 **12/4 星期四，美国东部时间下午 1 点** 在 Events 频道举行答疑环节。
   - 用户可以加入 [Discord 活动](https://discord.gg/nousresearch?event=1442995571173625888) 进行参与。
- **占位主题**：这是一个为了满足最小条目要求的占位主题。
   - 它向 topicSummaries 数组添加了第二个条目。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1442973558631301356)** (146 条消息🔥🔥): 

> `Suno Warner Music Partnership, Data vs Compute Cost, Blackwell Architecture, Z-Image Model, AI Disclosure on Steam` 


- **Suno 与华纳音乐合作引发辩论**：Suno 与 [华纳音乐集团 (Warner Music Group)](https://www.wmg.com/) 的合作引发了关于 AI 生成音乐的未来及其对音乐产业影响的疑问。
   - 一位成员指出，虽然一些 **Suno** 歌曲与人类创作的音乐难以区分，但许多其他歌曲很容易被识别为 AI 生成，这导致了对其潜力和缺陷的矛盾心理。
- **数据支出在计算成本面前显得微不足道**：一位成员指出，在数据上花费 **2000 美元** 与在计算上花费 **3200 万美元** 之间存在巨大差距，突显了 AI 模型训练的资源密集型本质，正如在 [Udio](https://www.udio.com/) 和 [Suno](https://www.suno.ai/) 中所看到的那样。
   - 这种向优先考虑计算的转变可能会显著缩小未来的研究途径，特别是获取高质量的选择性加入 (opt-in) 训练数据的途径。
- **Blackwell 的瓶颈：INT/FP 混合乱象**：在 **Nvidia 的 Blackwell 架构**上混合 **INT** 和 **FP** 工作负载会严重降低性能，因为其统一标量流水线（unified scalar pipeline）每个核心每个周期只能运行一种类型的操作。
   - 最佳实践是保持每个内核（kernel）要么是 **纯 FP**，要么是 **纯 INT**，以避免因不断的缓存抖动（cache thrashing）和代码重新加载而导致的 **30-50% 的性能损失**。
- **Z-Image 模型上线 Modelscope**：**6B Z-Image 模型**已在 [Modelscope](https://modelscope.cn/models) 上发布，预计其 Hugging Face 页面也将随后推出，尽管体积较小，但该模型提供了电影级的美感。
   - 它在美学上更偏向电影感，并提供了一个蒸馏版本以实现更快的推理。
- **开发者讨论 Steam 的 AI 披露政策**：关于 Steam 的 AI 内容披露政策引发了讨论，Epic 首席执行官 Tim Sweeney 认为 AI 披露应仅适用于“艺术”而非游戏。
   - 虽然 Sweeney 认为 AI 披露是不必要的，但一些人认为这能让消费者了解 **AI 生成内容** 对其游戏体验的潜在影响，尤其是在配音和美术等领域。


  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1443241275536179321)** (2 messages): 

> `LLM benchmarks, pre-training data contamination, private benchmarks` 


- **LLM 基准测试面临预训练数据污染**：一位成员询问 **LLM benchmarks** 是否能确保模型在预训练期间没有见过相关问题，以避免出现模型仅靠记忆解决问题而导致结果偏差的情况。
   - 另一位成员回答说，基准测试*并不总是*能考虑到这一点，尽管一些提供商会保留**私有基准测试版本 (private benchmark versions)**。
- **克服基准测试中的污染具有挑战性**：会议指出，一旦基准测试被用于模型测试，技术上它也可以被用于训练，这给维护基准测试的完整性带来了挑战。
   - 减轻这一问题的建议包括使用**大型私有数据集**和/或**难以记忆**的问题。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1443020796242165921)** (2 messages): 

> `History of Information Retrieval, RAG, Library of Alexandria` 


- **讲座追溯信息检索历史**：一场讲座追溯了从**亚历山大图书馆**到 **RAG** 的**信息检索 (information retrieval)** 发展历程，该内容发布在 [YouTube 视频](https://youtu.be/EKBy4b9oUAE)中。
- **Teknium 极力推荐该讲座**：Teknium 表达了对该讲座的期待并打算观看。
   - 未提供更多细节。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1443067067992506462)** (81 messages🔥🔥): 

> `Hallucinations in Multi-Stage LLMs, AI and Collaborative Work, LLMs as Golden Retrievers, Verifying AI Claims, AI fact checking misinformation` 


- **多阶段 LLM 中的幻觉仍被视为幻觉**：在讨论多阶段 LLM 过程中的幻觉时，一位成员表示，即使被 **Chain of Thought** 流水线修正了，它仍然是*生成它的组件系统的幻觉*。
   - 他们补充说，*人类也经常像那样产生幻觉并自我修正*，并分享了一篇关于 [LLM 幻觉的论文](https://arxiv.org/abs/2509.04664)。
- **LLM 与协作工作**：一位成员寻求关于与 AI 协作工作的反馈，重点关注长篇推理和镜像学习，他们征求了关于验证其推理过程合理性的建议。
   - 他们分享了自己在 [GitHub 上的 Causality Trilemma 项目](https://github.com/BigusUk/Causality-Trilemma)，该项目让他们*清晰地理解了自己的认知风格——如何识别矛盾、完善假设以及从问题中构建结构化模式*。
- **LLM 是复杂的金毛寻回犬**：多位成员将 LLM 比作金毛寻回犬，强调它们倾向于讨好用户，即使这意味着提供错误或误导性的信息，尤其是像 **ChatGPT**、**Claude**、**Gemini** 和 **Grok** 这样的聊天机器人。
   - 一位成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=VRjgNgJms3Q)，以强调 LLM 如何在没有真正理解或逻辑一致性的情况下生成输出。
- **如果你一无所知，LLM 也帮不了你**：有人指出，人们利用 AI 模型产出重大成果的唯一前提是他们已经是该领域的专家。
   - 一位成员链接了一篇 [LessWrong 帖子](https://www.lesswrong.com/posts/rarcxjGp47dcHftCP/your-llm-assisted-scientific-breakthrough-probably-isn-t)，建议在相信你的 LLM 辅助科学突破之前可以采取的步骤。
- **增加 LLM 数量对事实核查没有帮助**：一位成员表示，使用多个 LLM 对改善情况几乎没有帮助，因为它们产生虚假信息幻觉的倾向非常相似。
   - 他们警告不要向发帖者提供关于如何对 LLM 进行事实核查的误导性或错误建议。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1442998192856895561)** (37 条消息🔥): 

> `SGD 混洗, PIQA 论文拼写错误, Emergent Misalignment 论文复现, AI for Drug Discovery` 


- **SGD 混洗引发辩论**：成员们讨论了在 **SGD** 中每个 epoch 混洗数据的优劣，一位成员认为 *shuffle once*（只混洗一次）应该总是优于 **IID**，这与 **SGD** 的已知结果相反。
   - 另一位成员反驳称，由于优化曲面的非凸性质，实践比证明更重要，并指出 **IID** 可能导致方差增加和数据重复访问，而每个 epoch 进行混洗则平衡了噪声和结构。
- **PIQA 论文的葡萄牙语失误**：一位成员幽默地指出新发表的 **PIQA** 论文中可能存在一个拼写错误，文中将葡萄牙语列为了东欧语言，并附上了[一张图片](https://cdn.discordapp.com/attachments/747850033994662000/1443007174560448702/pt.png?ex=6928d228&is=692780a8&hm=e02447fccbf06df2a5add1bb8af742340f3158641951e49a9755337dd7e89e1c)作为参考。
   - 论文作者确认了该错误并承诺予以更正。
- **并行 MLP 与 Attention 的性能**：一位成员询问并行 **MLP** 和 **Attention**（**GPT-J** 风格）是否劣于其他替代实现。
   - 另一位成员分享了个人经验，指出过去的稳定性问题应归因于 prenorm 风格的交互，而非底层的并行执行技术本身，同时提到 *shortcut moe* 的成功可作为一个相关的对比。
- **重新审视 Emergent Misalignment，揭示 JSON 陷阱**：一位成员发布了 "Emergent Misalignment" 论文的复现和扩展版本，测试了 **Gemma 3** 和 **Qwen 3**，发现开源权重模型对不安全微调表现出惊人的鲁棒性（0.68% 的 misalignment），但同时发现了一个与格式相关的漏洞，**JSON** 格式使 misalignment 率减半（0.96% vs 0.42%）。
   - 该成员发布了[完整数据集](https://huggingface.co/datasets/thecraigd/emergent-misalignment-results)和[代码](https://github.com/thecraigd/emergent-misalignment)以供复现，并推测 **JSON** 限制减少了模型拒绝有害请求的自由度，详见[这篇博客文章](https://www.craigdoesdata.com/blog/the_json_trap/)。
- **寻求 AI for Drug Discovery 资源**：一位成员请求推荐了解 **AI for Drug Discovery** 领域概况的教育资源，寻求有关架构、开放问题和现状的信息。
   - 另一位成员建议查阅 [Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=ai+for+drug+discovery+survey&btnG=) 上的各种综述，还有成员提到了 **Zach Lipton** 的初创公司。


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/)** (1 条消息): 

junktown_24268: https://papers.cool/arxiv/2509.24406 - 第 3 节，5.1 中的图片等等。
  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1443004435272826900)** (69 条消息🔥🔥): 

> `Claude Code 升级的 Plan Mode、DeepMind 纪录片、Jeff Dean 的 15 年 ML 回顾与 Gemini 3.0、AI 生成幻灯片、OpenAI vs Claude` 


- **Claude Code 的 Plan Mode 实现并行化**：Sid 强调了 **Claude Code Plan Mode** 的重大改版：多个探索型 subagents 现在可以并行启动，生成竞争性方案，提出澄清问题，并允许用户通过 `/plan open` 编辑保存的方案文件（[来源](https://xcancel.com/sidbidasaria/status/1993407762412536275?s=46)）。
   - 根据后续讨论，社区非常喜欢更高的 one-shot 成功率，但希望有更快的 UX、"仅询问"选项、模型选择器（**Opus vs Sonnet**）以及更简洁的重新规划过程（[讨论 1](https://x.com/sidbidasaria/status/1993407765558251657?s=46), [讨论 2](https://x.com/sidbidasaria/status/1993407771438727356?s=46)）。
- **关于 DeepMind 起源的纪录片《Thinking Game》发布**：成员们观看了免费的完整版纪录片 **The Thinking Game**，该片探索了 DeepMind 的起源，现已在 [YouTube](https://www.youtube.com/watch?v=d95J8yzvjbQ) 上线。
   - 观众称赞其内容*非常棒*，并表示这部电影*真的让人希望 Demis 能赢得 AGI 竞赛*。
- **Jeff Dean 的 AI 回顾与 Gemini 3.0**：**AER Labs** 总结了 **Jeff Dean 在斯坦福的演讲**，追溯了 **15 年的 AI 进展**——从 90 年代的手写梯度到解决 IMO 问题的 **Gemini 3.0**——由规模、更好的算法（**TPUs, Transformers, MoE, CoT**）和硬件驱动，此外还演示了低代码“Software 3.0”和视觉推理（[来源](https://xcancel.com/aerlabs_/status/1993561244196868370)）。
- **Claude 生成 Powerpoint 幻灯片**：一位成员尝试了 **Claude 的新 Powerpoint 技能**，评价其*相当不错*，通过指向公司风格指南和博客文章获取信息和高层叙述，制作了 10 张近乎完美的幻灯片。
   - 他们分享了生成幻灯片的[截图](https://cdn.discordapp.com/attachments/1443329209853542472/1443345290806689822/Screenshot_2025-11-26_at_1.57.20_PM.png?ex=6928bb8d&is=69276a0d&hm=f5b43c6358ea3e7eadc16e787d5ae2a5afe37a7dca33f19c0fe32ceaffe726d0&)。成员们还讨论了 Google Slides 中的 **Nano Banana Pro**。
- **ChatGPT Pro vs Claude**：成员们讨论了 **ChatGPT Pro** 与 **Claude** 的价值，指出 **ChatGPT** 在通用研究方面非常出色，拥有更好的 **Codex 速率限制**，对非 **ts/js/py** 语言支持更好，且如果你使用 pulse, atlas, sora, codex cloud 等功能，其价值更高。
   - 然而，成员们表示 **Claude** 总是不断突破边界，其模型在工具使用方面训练得更好，前端 UX 和 UI 非常出色，且其 CLI 的可读性、排版和字体层级使其更易于理解。


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1443323891534528716)** (2 条消息): 

> `SOTA Vision, RF-DETR 论文, NeurIPS, 2025 开发者作家静修会` 


- **RF-DETR 论文作者主持 SOTA Vision 专题**：**RF-DETR 论文**的作者正在为热衷于 **SOTA Vision** 的人士主持一场特别活动，链接见[此处](https://luma.com/c1rqkxzl)。
- **NeurIPS 报名提醒**：提醒大家注册 **NeurIPS** 标签，并在相关频道发布相关论文、讨论、聚会和问题。
   - 组织者将在本周晚些时候到达现场。
- **2025 开发者作家静修会接受最后报名**：**2025 Dev Writers Retreat** 将在 **NeurIPS** 之后于圣迭戈举行，本周正在接受最后的报名，链接见[此处](https://lu.ma/dwr2025)。


  

---

### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1442969919611211867)** (31 条消息🔥): 

> `Black Forest 提示词指南，Wisprflow 新融资，SGLang diffusion，Whisper Thunder 对比 VideoGen，AI 图像真实感大比拼` 


- **Whisper Thunder 篡位 VideoGen**：ML 社区正热议 **Whisper Thunder**，这款全新的排名第一的 text-to-video 模型在最新的 Artificial Analysis 排名中超越了 **VideoGen** —— 详见[此处](https://xcancel.com/soumithchintala/status/1993694517489537105?s=46)。
- **Nano Banana Pro 的真实感引发辩论**：对比 **Grok 4.1**、**ChatGPT-free**、**Google Nano Banana** 和 **Nano Banana Pro** 生成的 AI 图像显示，Nano Banana Pro 产生的图像“与现实无异”，如[此处](https://xcancel.com/romainhedouin/status/1993654227399475347?s=46)所示。
- **OpenAI 图像生成升级评价褒贬不一**：用户发现 OpenAI 悄悄更新了其图像生成模型，反应从对其高质量的赞赏到对其多语言支持差、场景间参考不一致以及持续的安全防护栏（safety guardrails）的批评不等，如[此处](https://xcancel.com/arrakis_ai/status/1993644406159917533?s=46)所示。
- **FLUX 2 Pro 视觉效果显著提升**：**FLUX 2 Pro** 相比 **FLUX 1 Pro** 实现了重大质量飞跃，消除了“塑料感”并提供了更高的细节忠实度，如[此处](https://xcancel.com/iamemily2050/status/1993477498940899366?s=46)的对比所示。
- **Nano Banana Pro 助长欺诈**：**Nano Banana Pro** 可以通过一条提示词创建近乎完美的伪造收据、KYC 文件和护照，引发了对潜在诈骗和欺诈的警惕，用户在[此处](https://xcancel.com/deedydas/status/1993341459928694950?s=20)进行了讨论。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1443006329630363669)** (61 条消息🔥🔥): 

> `信息检索历史，能源部 Genesis AI 平台，LLM 预训练的课程学习，MIT 关于 AI 取代工作的研究，用于零知识证明的 Trumpcoin 协议` 


- **信息检索讲座：从亚历山大图书馆到 RAG**：一位成员分享了一个关于**信息检索**历史的 [YouTube 讲座](https://youtu.be/aR20FWCCjAs?si=wmNYCsqPp7Le8FWe)，追溯了从**亚历山大图书馆**到 **RAG** 的发展历程。
   - 一些人表示有兴趣参加论文讨论，而另一些人则参考了一位拥有机器学习博士论文背景的**神经科学博士**制作的[演示视频](https://youtu.be/5X9cjGLggv0?si=ZF85m9AssbQw8u75)。
- **美国能源部瞄准国家级 AI 平台**：**能源部**计划在美国超级计算机和联邦科学数据之上构建一个国家级 AI 平台。
   - 该平台旨在训练科学基础模型，并运行 AI Agent 和机器人实验室，以自动化**生物技术、关键材料、核裂变/聚变、航天、量子和半导体**等各个领域的实验。
- **关于 LLM 预训练课程学习技术的辩论升温**：成员们讨论了在 **LLM 预训练**期间使用课程学习（curriculum learning）和核心集（coreset）技术的情况，其中一位成员质疑非随机采样可能引入的偏差。
   - 他们引用了 [Olmo 3 论文](https://allenai.org/blog/olmo3) 和 [OLMo 论文](http://allenai.org/papers/olmo3) 作为参考，并根据[这篇论文](https://arxiv.org/abs/2508.15475v2)澄清，只要采用更以模型为中心的难度概念，**课程学习对语言模型预训练是有益的**。
- **AI 已在取代美国劳动力**：一篇 [CNBC 文章](https://www.cnbc.com/2025/11/26/mit-study-finds-ai-can-already-replace-11point7percent-of-us-workforce.html)指出，MIT 的一项研究发现 AI 已经可以取代 **11.7%** 的美国劳动力。
   - 随后引发了关于方法论的讨论，参考了 [Iceberg Index](https://iceberg.mit.edu/) 和相应的[论文](https://arxiv.org/abs/2510.25137)，并对信任 LLM 来判断其他 LLM 工具是否能自动化工作表示怀疑。
- **Trumpcoin 协议上的 ZKP 张量**：一位成员开玩笑说要在 **trumpcoin** 协议上发送所有带有**零知识证明（zero knowledge proofs）**的张量。
   - 他们补充说，所有 **Epstein 文件**都将以**零知识证明**的形式发布，以证明这是一场政治迫害，同时保护 Epstein 的受害者。


  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1443011390628696166)** (24 messages🔥): 

> `Adobe AI 摘要, LLM 摘要局限性, AI/CS 领域中的 ADHD 与自闭症, 不求甚解地发布论文` 


- **Adobe AI 摘要：魔鬼的诱饵？**：一位成员开玩笑地表示 **Adobe 的 AI 摘要** 可能会导致问题，并引用了一张 [附图](https://cdn.discordapp.com/attachments/1045297868136779846/1443020193000456243/Adobe_Vermin.png?ex=6928de48&is=69278cc8&hm=128c6461c705032d5b88293eedae078353ef799ddbd74a2b9e1a8521561a6dbf&)。
   - 另一位成员提到：“我不喜欢它，因为他们几乎总是使用差得多的模型。如果你把 PDF 粘贴到 ChatGPT, Claude, Gemini 等模型中，你会得到好得多的结果。”
- **LLM 难以摘要高密度信息**：成员们分享了 **LLM** 在摘要时往往无法抓重点的经历，尤其是在处理高信息密度的文本时。
   - 一位成员表示：“大家一直在说 LLM 是伟大的摘要生成器。但在我的经验中，它们真的不是，因为它们无法理解什么是重要的，什么是可以丢弃的。”
- **技术领域的 ADHD 与自闭症：热门话题**：一位成员暗示了好奇心、**ADHD** 和 **自闭症** 在理解论文方面的联系，引发了不同的反应。
   - 作为回应，有人断言患有此类疾病并不一定决定特定的行为，多位成员分享了自己被诊断为 **ADHD** 或疑似 **Asperger's** 的经历。
- **遏制论文泛滥：新规则提案**：针对某用户在未表现出充分理解的情况下发布大量论文的情况，成员们提出了新规则建议。
   - 该规则将论文推荐限制为那些获得显著正面反馈或发布在特定频道的论文，旨在过滤噪音并确保相关性。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1442987137413943488)** (6 messages): 

> `Nano Banana Pro, 腾讯混元 (Hunyuan), MAGA 对 AI 数据中心的抵制, AI 取代美国劳动力` 


- **腾讯发布混元 (Hunyuan) 模型！**：腾讯最近发布了他们的 **混元 (Hunyuan) 模型**，如[此视频](https://hunyuan.tencent.com/video/zh?tabIndex=0)所示。
- **MAGA 支持者反对 AI 数据中心**：一些 MAGA 支持者现在正推动抵制 **AI 数据中心**，如[此 YouTube 视频](https://youtu.be/9_-oDkSWKMc?t=28)中所讨论的。
- **MIT 研究：AI 可取代 11.7% 的美国劳动力**：根据 **MIT 研究**，AI 已经可以取代 **11.7%** 的美国劳动力，详见[此 CNBC 文章](https://www.cnbc.com/2025/11/26/mit-study-finds-ai-can-already-replace-11point7percent-of-us-workforce.html)。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1442972756483244207)** (19 messages🔥): 

> `Hugging Face Inference API, 圣诞礼物发放, Hugging Face 错误, Genesis 任务, LLMStudio 的 PDF 阅读模型` 


- **Inference API 变灰了？**：一位成员询问如何为他们上传的模型启用 **Hugging Face 内部 Inference API**，并指出 UI 中的推理选项目前处于灰色不可用状态，如 [附图](https://cdn.discordapp.com/attachments/879548962464493622/1443040959901204530/image.png) 所示。
- **捐赠-谈判-协作 (DNC) Markdown**：一位成员分享了他们可能是最后的 **圣诞礼物发放**，包括一个 [DNC.md 文件](https://cdn.discordapp.com/attachments/879548962464493622/1443110237165846629/DNC.md)，并对其用途表示不确定，希望能对他人有所帮助。
- **关于 ComfyUI 的问题**：针对在本地运行 **GGUF 文本转图像模型** 的问题，一位成员建议使用 [ComfyUI](https://github.com/city96/ComfyUI-GGUF) 或 [koboldcpp](https://github.com/LostRuins/koboldcpp/)。
- **LM Studio PDF 助手**：一位成员询问是否有适用于 **LLMStudio** 且能够阅读 **PDF** 文件并回答问题的模型，另一位成员建议任何指令型（instruct）**LLM** 应该都可以，只需使用 **LM Studio** 内置的 RAG 功能。
   - 他们还分享了 [LM Studio 模型页面](https://lmstudio.ai/models) 和 [Hugging Face 模型页面](https://huggingface.co/models?apps=lmstudio&sort=trending) 的链接。
- **西班牙语文本数据集寻求**：一位成员为一个 **MoE 语言模型项目** 寻求大规模、高质量的 **西班牙语文本数据集**。
   - 另一位成员提供了 [西班牙语数据集](https://huggingface.co/datasets/John6666/forum2/blob/main/spanish_es_dataset_1.md) 和相关 [Discord 频道](https://discord.com/channels/879548962464493619/1205128865735770142) 的链接。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/)** (1 messages): 

aboodj_: 太赞了 (epic)
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1443099000638144612)** (8 messages🔥): 

> `RapidaAI Open Source, French Books Dataset, AI Sci-Fi Short Film` 


- **RapidaAI 走向 Open Source**：RapidaAI，一个**生产就绪的语音 AI 平台**，现已 [Open-Source](https://rapida.ai/opensource?ref=hf)，旨在让用户掌控自己的语音 AI 并避免额外的供应商成本。
   - 该公司观察到，团队为了租用他人的技术栈，每分钟需要额外支付 **$0.05–$0.15**，导致每年产生六位数的成本。
- **法语经典书籍数据集发布**：一名成员创建并分享了一个 [公有域法语书籍数据集](https://huggingface.co/datasets/Volko76/french-classic-books)，可在 Hugging Face 上获取。
   - 此外，还有一个仅包含书中**对话**的版本（[此处](https://huggingface.co/datasets/Volko76/french-classic-conversations)），专为指令微调（Instruction）目的设计。
- **AI 科幻短片发布**：一名成员在 [YouTube](https://www.youtube.com/watch?v=_F0cXXSivpU&feature=youtu.be) 上展示了一部名为《Tales of the Sun - Céline》的 AI 生成科幻短片。
   - 创作者花费了**两个月**时间制作这部电影，并正在寻求社区的反馈。


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1443094266586071050)** (3 messages): 

> `Chunking, GNN presentation, Structured data` 


- **Chunking 的影响较小**：一位成员表示很高兴看到 **Chunking** 的影响并没有那么大。
   - *对于非结构化数据，由于边缘情况有限，你不会看到太大的差异*。
- **GNN 演示即将到来**：一位成员计划进行关于 **GNN** 的演示，将从 **AlphaFold 2 和 3** 开始。
   - 由于研究仍在进行中，具体主题尚未确定。
- **结构化数据很有价值**：一位成员建议在博客中尝试使用**结构化数据**。
   - 他们指出，对于非结构化数据，由于边缘情况，差异可能有限。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/)** (1 messages): 

dodrawat: 让我们建立联系
  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1443364763496349777)** (2 messages): 

> `Mojo repo, Copybara, Repo Sync` 


- **Modular 使用 Copybara 同步仓库**：成员们讨论了 Mojo 如何保持其内部和外部仓库同步，一名成员确认他们使用的是 [**Copybara**](https://github.com/google/copybara)。
   - **Copybara** 负责管理内部私有仓库与外部 Open Source 仓库的同步。
- **Copybara 管理内外仓库**：**Copybara** 用于管理内部私有仓库并将其与外部 Open Source 仓库同步。
   - 这确保了更改和更新能够一致地反映在两个仓库中。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1443146827804311684)** (20 messages🔥): 

> `MAX examples for newbies, MAX written in Python, Mojo API in MAX, Migrating Python MAX code to Mojo MAX, Performance gains in MAX with Mojo` 


- **MAX 新手寻求示例**：一位成员询问是否有学习 **MAX** 的小型示例，并对训练表现出兴趣。
   - 另一位成员建议 **Endia** 有一些相关内容。
- **Python 在 MAX 中的角色受到质疑**：一位成员询问了使用 **Python** 编写 **MAX** 的决定，推测这是否为了更容易迁移到 **MAX** 和 **Mojo**。
   - 该成员担心这是否会导致类似 **PyTorch** 的“分裂世界”（split world）问题，以及是否会出现一个纯 **Mojo** 编写的 **MAX** 框架。
- **期待 Mojo API 回归 MAX**：一位成员澄清说，**MAX** 之前曾有过 **Mojo API**，但由于 **Mojo** 当时处于不完整状态而停止了。
   - 他们表示，当语言更加成熟时，**Mojo API** 应该会在某个时间点回归。
- **强调 Python 到 Mojo 的迁移障碍**：一位成员解释说，虽然 **Mojo** 看起来像 **Python**，但它并非严格的 **Python** 超集，反而更接近 **C++** 或 **Rust**。
   - 他们警告说，迁移到 **Mojo MAX** 需要付出努力才能充分发挥 **Mojo** 的潜力。
- **对 Mojo MAX 性能提升的质疑**：一位成员指出 **MAX** 使用了 **JIT compiler**，认为 **Mojo** 带来的性能提升可能主要体现在图构建（graph construction）时间上。
   - 他们推测 **Mojo MAX** 和 **Python MAX** 之间的速度差异可能并不显著，且在 **Mojo** 获得更多特性之前，分裂世界的问题将一直存在。


  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1443178668007620752)** (19 messages🔥): 

> `TinyJit 内部原理, 非 tinygrad Python 操作, Tinygrad 中的随机性函数, Tinygrad JIT 教程, PyTorch 编译器历史` 


- **TinyJit 仅重放 kernels**：当使用 `@TinyJit` 时，被包装的函数仅重放捕获的 **tinygrad kernels** 和 **ExecItems**，而被包装的函数本身根本不会运行。
   - 如果你需要运行 Python 代码，请将其拆分为单独的 JIT 函数，但这可能很棘手，且任何 **非 tinygrad 输出** 都不会被更新。
- **`Tensor` 中的随机性函数按预期工作**：`Tensor` 上的随机性函数应该可以工作，因为它们通过 kernel 增加计数器。
   - 示例：`CPU=1 DEBUG=5 python3 -c "from tinygrad import Tensor; Tensor.rand().realize(); Tensor.rand().realize()"`。
- **Tracing 需要两次 JIT 运行，但未来可能会改为验证匹配**：JIT 使用第二次运行来重复捕获的 kernels，而第一次运行可能会执行不同的设置任务，如权重初始化。
   - 一项提案建议 JIT 可能会更新为等待两次运行匹配，这表明该实现仍处于 pre-1.0 阶段且可能会发生变化，目前的重点是消除易误用的陷阱 (footguns)。
- **关于 tinygrad JIT 的优秀教程**：一名成员分享了 [tinygrad JIT 教程](https://mesozoic-egg.github.io/tinygrad-notes/20240102_jit.html)。
   - 该教程虽然有点过时，但仍然很棒。
- **Tinygrad 基础已经稳固**：Tinygrad 的基础现在已经稳固，团队现在的重心正转向前端易用性。
   - 有人回忆道：*fast.ai 课程中最早的 pytorch 编译器真的是使用正则表达式来拼接 C 代码字符串的！*。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1443177250269954071)** (14 messages🔥): 

> `Kimi 的限制, Chatbots vs Canvases, 对话谬误` 


- **探索 Kimi 的限制**：一位用户询问了 **Kimi** 的限制，尽管计划从网页界面升级，但仍表示不确定，并附上了一张 [截图](https://cdn.discordapp.com/attachments/1371757564005711973/1443259988058574900/image.png?ex=69286c1b&is=69271a9b&hm=3d4d9ba62a03dc65a27edfb0fb93a8c0f8a0f6518ab2737c5a714d6032d2b5a6&)。
   - 另一位用户称赞 **Kimi K2** 具有卓越的思考能力和反驳能力，强调了它在 prompt 语境下的理解和交互能力。
- **Canvas 热潮即将到来？**：一位用户表示不敢相信 *canvases* 还没有取代 chatbots，认为对于像 **Kimi** 和 **Qwen** 这样的全屏网站来说，它们更有意义。
   - 他们认为，虽然 chatbots 适用于小型侧边栏，但 canvases 可以为综合性网页界面提供更好的体验。
- **关于对话谬误的思考**：一位用户分享了一个令其着迷的观点：*我们陷入了对话谬误（conversational fallacy）：即认为 AI 必须通过对话才能被使用的想法*。
   - 该用户似乎认为 **Kimi** 在避免陷入这种谬误方面做得非常出色。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1443010354866294867)** (4 messages): 

> `dspy-cli 工具, DSPy 项目, FastAPI 端点, MCP 工具, Docker 托管` 


- **dspy-cli 工具开源**：成员们宣布 `dspy-cli` 工具现已开源并发布在 [PyPi](https://pypi.org/project/dspy-cli/) 上，旨在帮助创建、开发、测试和部署作为 HTTP APIs 的 **DSPy programs**。
   - [代码库已在 GitHub 上线](https://github.com/cmpnd-ai/dspy-cli)，可以使用 `uv tool install dspy-cli` 进行安装。
- **dspy-cli 新功能发布**：主要功能包括脚手架化一个新的 **DSPy project**，从命令行创建新的 signatures，将模块作为 **FastAPI endpoints** 运行，或将其作为 **MCP tools** 使用。
   - 程序可以轻松部署到自选的 **docker hosting service**。
- **dspy-cli 因其项目实用性受到好评**：成员们表示渴望在更多项目中尝试 `dspy-cli` 并宣传其用途。
   - 一位用户在 [推文](https://x.com/dbreunig/status/1993462894814703640) 中称赞了该工具，对其出色工作表示肯定。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1442977805162909869)** (9 messages🔥): 

> `ReAct Module Trajectory Injection, Web Search API Implementation in DSPy, Anthropic Web Search API, Latency issues with web search API calls` 


- **ReAct 模块中的轨迹注入 (Trajectory Injection)**：一位成员询问如何在 **ReAct 模块**中注入轨迹，寻求在消息历史记录之外为 Agent 提供来自先前运行的上下文。
- **DSPy 的 Web Search API 选择**：一位成员就实现在 **DSPy** 中实现 Web 搜索工具的最佳 **API** 寻求建议，特别是询问是否可以使用提供商的原生 Web 搜索 **API**。
- **Exa API 包含摘要功能**：一位成员分享了使用 **Exa API** 的积极体验，因为它具有摘要功能，可以避免在 **Firecrawl** 和 **Parallel.ai** 等其他 API 中发现的随机广告和 HTML 标签。
- **将 Anthropic 的 Web 搜索 API 与 ReAct 结合使用**：一位成员正尝试使用 **Anthropic 的 Web 搜索 API** 与 ReAct 结合实现，并分享了使用 `dspy.ReAct` 的代码片段。
- **Web Search API 调用导致的延迟**：一位成员提出了关于在 **DSPy** 的 **ReAct** 中，在调用 LLM 之前使用类似 `search_web` 的搜索函数所导致的 Web 搜索 **API** 调用延迟问题。


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1442995213609074709)** (11 messages🔥): 

> `New Protocol Version, UI SEP Release, MCP Namespace Collision` 


- **新协议版本发布！**：新的协议版本已经发布，正如在 [Discord 频道](https://discord.com/channels/1358869848138059966/1421239779676127402/1442991223617880064) 中宣布的那样。
   - 成员们对 **MCP 社区** 在过去一年中的贡献表示兴奋和感谢。
- **UI SEP 带外发布！**：**UI SEP** 可以作为扩展从主规范中带外 (out-of-band) 发布。
   - 更多详情请查看 <#1376635661989449820> 频道。
- **MCP 考虑命名空间冲突！**：一位成员询问 **MCP** 小组是否考虑了命名空间冲突的可能性。
   - 具体而言，有人提出如果某些内容声称是 something-mcp 但偏离了实际的 **MCP** 标准，该小组是否会采取行动。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1442980169353793619)** (8 messages🔥): 

> `AI Engineer introduction, API Issues, Telegram channel` 


- **AI 工程师展示专业能力**：一位在多个领域拥有构建先进端到端 AI 系统实战经验的 **AI 工程师** 介绍了自己。
   - 他们的专业领域涵盖 **AI Agent、多 Agent 系统、自动化工作流、NLP 驱动的聊天机器人、集成语音和语言系统、部署自定义 LLM、微调 AI 模型、Web3、智能合约以及 AI 集成的区块链游戏**。
- **用户报告 API 问题及缺乏支持**：一位用户报告称，尽管充值了超过 **$600**，但在 **webdev.v1.WebDevService/GetDatabaseSchema** 中因使用配额耗尽而遇到 *[unknown] error*。
   - 该问题导致其整个账户无法使用，影响了超过 **500 名活跃用户**，且他们尚未收到团队的任何回复或支持。
- **成员询问 Telegram 频道**：一位成员询问是否存在 **Manus Telegram 频道**。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1443213701753606215)** (3 messages): 

> `Benchmark Updates, Opus 4.5 vs Sonnet 4.5` 


- **社区建议更换基准测试网站管理员**：一位成员建议由其他人运行该网站，以便能够用新模型更新基准测试 (Benchmark) 结果。
   - 这暗示了对当前基准测试结果更新状态的不满。
- **Opus 4.5 是重大升级还是相比 Sonnet 4.5 的小幅升级？**：一位成员发起了一项快速调查，以衡量社区对 **Opus 4.5** 相比 **Sonnet 4.5** 是重大升级还是小幅升级的看法。
   - 另一位成员报告称，在尝试通常正确的 Bedrock 模型标识符时遇到了 *'model not found'* 错误。


  

---


---


---