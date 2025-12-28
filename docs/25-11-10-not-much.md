---
companies:
- moonshot-ai
- meta-ai-fair
- togethercompute
- qwen
date: '2025-11-10T05:44:39.731046Z'
description: 'Moonshot AI 的 Kimi K2 Thinking 在 AMA（问答活动）中透露，其采用了 **KDA + NoPE MLA**
  的混合注意力栈，性能优于全量 MLA + RoPE。其 **Muon 优化器**可扩展至约 1 万亿参数规模，并支持原生 **INT4** 量化感知训练（QAT），以实现高性价比推理。


  K2 Thinking 在 **LisanBench** 和 **LM Arena Text** 排行榜上名列前茅，提供低成本的 INT4 推理服务，并在数学、编程和创意写作方面表现强劲。它支持高强度的智能体（Agent）工具使用，单次运行最多可发起
  300 次工具请求，并建议使用官方 API 以获得可靠的长链路推理。


  **Meta AI** 发布了 **Omnilingual ASR**（全语言自动语音识别）套件，涵盖 1600 多种语言（包括 500 种资源匮乏的语言），此外还发布了一个
  7B 参数的 wav2vec 2.0 模型和 ASR 语料库。


  此外，用于 GUI（图形用户界面）操作智能体中计算机落地（Computer Grounding）的 **Gelato-30B-A3B** 模型性能优于更大型的视觉语言模型（VLM），旨在实现智能体的即时增益。Qwen
  的图像编辑 LoRA 和轻量修复应用也受到了关注。'
id: MjAyNS0x
models:
- kimi-k2-thinking
- kimi-k3
- gelato-30b-a3b
- omnilingual-wav2vec-2.0
people:
- yuchenj_uw
- scaling01
- code_star
- omarsar0
- kimi_moonshot
- anas_awadalla
- akhaliq
- minchoi
title: 今天没发生什么特别的事。
topics:
- attention-mechanisms
- quantization
- fine-tuning
- model-optimization
- agentic-ai
- speech-recognition
- multilingual-models
- gui-manipulation
- image-editing
- dataset-release
---

**平静的一天**

> 2025年11月7日至11月10日的 AI 新闻。我们为您检查了 12 个 subreddit、544 个 Twitter 账号和 23 个 Discord 服务器（201 个频道，12566 条消息）。预计节省阅读时间（以 200wpm 计算）：1015 分钟。我们的新网站现已上线，包含完整的元数据搜索和美观的 vibe coded 呈现方式。查看 https://news.smol.ai/ 获取完整的新闻细分，并在 @smol_ai 提供反馈！

[Kimi K2 AMA](https://www.reddit.com/r/LocalLLaMA/comments/1oth5pw/ama_with_moonshot_ai_the_opensource_frontier_lab/) 引起了广泛关注。

---

# AI Twitter 回顾

**Moonshot AI 的 Kimi K2 Thinking：AMA 要点、评估、INT4 设计及未来愿景**

- **AMA 亮点（架构、训练、路线图）**：来自 Kimi K2 Thinking AMA：常被引用的“460 万美元训练成本”并非官方数据；训练在 H800 上运行；使用 **KDA (Kimi Delta Attention) + NoPE MLA** 的混合注意力栈优于全 MLA + RoPE；据报道 **Muon 优化器**能很好地扩展到约 1T 参数，并已进入 PyTorch 稳定版；K2 Thinking 通过 QAT 原生支持 **INT4**，以便在非 Blackwell GPU 上实现更低成本的推理。团队表示 **Kimi K2 将获得 vision**，并暗示 K3 “可能使用 KDA 或其他混合注意力”。关于 K3 时间的调侃：“在 Sam 的万亿级数据中心建成之前”。来源：[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1987940704929395187), [@scaling01](https://twitter.com/scaling01/status/1987916859400659011), [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1987941323400507850), [@code_star](https://twitter.com/code_star/status/1987917177417289794), [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1987955443420065816)。
- **评估与定价**：在 LisanBench 上，K2 Thinking 是表现最好的开源权重模型，总排名约第 7（介于 GPT-5 和 GPT-5-Mini 之间），在多个项目上创下新高 ([@scaling01](https://twitter.com/scaling01/status/1987952884927934966))。在 LM Arena Text 排行榜上，它是排名第 2 的开源模型（MIT 修改版），总排名并列第 7，在数学/编程/创意写作方面表现强劲，职业表现处于顶尖水平 ([@arena](https://twitter.com/arena/status/1987947219224526902), [详情](https://twitter.com/arena/status/1987947222299013630), [立即体验](https://twitter.com/arena/status/1987947224173781185))。Arena 还指出 K2 Thinking 暴露了不受限制的 chain-of-thought，并经过 QAT 后训练，实现了**低成本 INT4 服务**；他们引用定价为每百万 token $0.15 / $2.5，而 Claude Sonnet 4.5 为 $3 / $15 ([@arena](https://twitter.com/arena/status/1987947219224526902))。
- **Agent 工具使用与推理引导**：K2 Thinking 支持繁重的 Agent 工作流——据报告单次运行中包含 **200–300 个工具请求**——将工具调用保留在推理轨迹内以防止偏移 ([演示推文](https://twitter.com/omarsar0/status/1987912692099682399), [@togethercompute](https://twitter.com/togethercompute/status/1988009780149878904))。为了获得可靠的基准测试结果，Moonshot 建议使用官方的 “kimi-k2-thinking-turbo” endpoint，启用流式传输、temp=1.0、慷慨的 max_tokens（推理 128k | 编程 256k）以及重试；他们观察到第三方提供商之间的准确率差异超过 20 个百分点，并正在发布 Vendor Verifier ([@Kimi_Moonshot](https://twitter.com/Kimi_Moonshot/status/1987892275092025635))。多位用户报告通过 OpenRouter 出现长轨迹失败，建议使用官方 API 进行长程推理 ([@scaling01](https://twitter.com/scaling01/status/1987938809628291168))。Together 将于 11 月 19 日主持 K2 Thinking 的技术深度探讨 ([@togethercompute](https://twitter.com/togethercompute/status/1988009777247510564), [模型访问](https://twitter.com/togethercompute/status/1988011880443470217))。

**语音与计算机使用模型：Meta 的 Omnilingual ASR 和 Gelato-30B-A3B**

- **Meta Omnilingual ASR (开源)**：发布了一系列 ASR 模型（300M–7B），涵盖 **1600 多种语言**，其中包括 **500 种以前从未提供过服务的语言**。同时发布的还有：一个 **7B Omnilingual wav2vec 2.0** 表示模型和一个涵盖 350 种服务欠缺语言的 **Omnilingual ASR Corpus**。模型和数据集均已开源 ([公告](https://twitter.com/AIatMeta/status/1987946571439444361), [详情与下载](https://twitter.com/AIatMeta/status/1987957744138416389))。
- **Gelato-30B-A3B (Agent 的计算机落地)**：在开源 Click-100k 上训练的新型“计算机使用”模型，在 **ScreenSpot-Pro** 上达到 **63.8%**，在 **OS-World-G** 上达到 **69.1%**，优于专门的 GTA1-32B，甚至优于尺寸约为其 8 倍的大型 VLM（例如 Qwen3-VL-235B）。目标是为 GUI 操作 Agent 带来即时收益 ([推文](https://twitter.com/anas_awadalla/status/1987913284989985092))。同样值得注意的还有：Qwen 的图像编辑 LoRA 和用于快速重新布光及阴影移除的轻量级修复应用 ([示例](https://twitter.com/minchoi/status/1988008926797787208), [数据集链接](https://twitter.com/_akhaliq/status/1987989916974829809))。

**数据与预训练：合成数据、课程学习与评估设计**

- **SYNTH + Baguettotron**：发布了一个全合成的通用预训练数据集 (SYNTH) 以及两个完全基于该数据集训练的新推理模型。仅使用 **200B tokens**，“Baguettotron” 被声称在其参数范围内是同类最佳，且根据作者报告，在非代码任务（包括数学）上达到了 SOTA ([announcement](https://twitter.com/Dorialexander/status/1987930819021635964), [follow‑up](https://twitter.com/Dorialexander/status/1987977993440936433))。评论将其视为迈向“认知核心 (cognitive core)”的一步，并探讨了非对数尺度的 Scaling 曲线 ([context](https://twitter.com/willccbb/status/1987998615785402785), [discussion](https://twitter.com/lateinteraction/status/1988016952451735772))。
- **课程学习、RLVR Scaling 与评估加固**：关于让模型动态发现该看什么数据以及何时看的提案 ([@joemelko](https://twitter.com/joemelko/status/1987715636861251667))；关于将 RLVR 计算量扩展到 Frontier 模型基准的 10–1000 倍是否能产生预训练之外的真正新知识的疑问 ([@YangYue_THU](https://twitter.com/YangYue_THU/status/1987716984524730604))。Benchmark 设计者被敦促“在测试集上训练”以暴露捷径和非视觉漏洞 ([@sainingxie](https://twitter.com/sainingxie/status/1988019293926080611))。一个反复出现的主题：高杠杆的领导层活动仍然是“标注数据” ([@model_mechanic](https://twitter.com/model_mechanic/status/1987945123439931785))。关于更长期的框架，请参阅 Fei-Fei Li 关于构建和使用世界模型以解锁空间智能的文章 ([thread](https://twitter.com/drfeifei/status/1987891210699379091))。

**Scaling 基础设施：GPU、内核与吉瓦级数据中心**

- **硬件 + 内核**：AMD 和 Modular 报告在 Instinct MI355X 上实现了 **14 天内推理速度提升 2.2 倍** ([@AMD](https://twitter.com/AMD/status/1987898172484567238))。NVIDIA 详细介绍了 TensorRT-LLM 在 **GB200 NVL72** 系统上的 Wide Expert Parallelism，用于 MoE 扩展 ([summary](https://twitter.com/dl_weekly/status/1987913458654786008))。Blackwell NVFP4 内核竞赛启动（首个任务：NVFP4 GEMV）([@a1zhang](https://twitter.com/a1zhang/status/1987972190898450922))。
- **吉瓦 (GW) 级数据中心**：Epoch AI 通过分析许可证/卫星图像，预测首批**吉瓦级数据中心**将于 2026 年上线，因为 Hyperscalers 将建设周期压缩至 1-2 年；包含 Frontier Data Centers 数据集和方法论报告 ([overview](https://twitter.com/EpochAIResearch/status/1987938542094610927), [thread](https://twitter.com/EpochAIResearch/status/1987944116861522227))。
- **市场/技术栈动态**：SemiAnalysis 报告称，一些 Frontier 实验室认为 **MI450X UALoE72** 在推理方面具有强大的性能/TCO，同时有报道称 AMD 提供了激进的激励措施 ([rumor](https://twitter.com/SemiAnalysis_/status/1988044940149235844))。预计 H100/H200 现货价格将在 25 年第四季度上涨 ([@FundaBottom](https://twitter.com/FundaBottom/status/1987905008541831521))，从业者预计即使在 Blackwell 发布后，H100 仍将拥有较长的生产寿命 ([@code_star](https://twitter.com/code_star/status/1988062247818850421))。企业技术栈：Siemens 分享了一个开源优先的平台，该平台由 vLLM 在可持续的混合代 NVIDIA 集群上进行了优化 ([@NVIDIAAIDev](https://twitter.com/NVIDIAAIDev/status/1987944094883037559))；Baseten 推动“拥有自己的权重 (own your weights)”训练基础设施 ([@basetenco](https://twitter.com/basetenco/status/1987943307532476746))。一个更广泛的观点将 GPU 视为智能时代的“储备货币”，CUDA 是可兑换性，而专门的云服务商则是“中央银行” ([analysis](https://twitter.com/TheTuringPost/status/1988002749452349495))。OpenAI 继续为核心计算基础设施招募人员 ([@gdb](https://twitter.com/gdb/status/1987996461846659372))。

**Agent、身份验证与评估工具**

- **Agent 的安全认证**：当前的 Web 认证标准不适用于无头 Agent 工作流（无浏览器/重定向）；OAuth 是以人为中心的，而静态密钥存在风险。MCP 并非认证层；它标准化了 Agent 的工具/资源发现。预计规范将快速演进，并出现专为 Agent 设计的全行业认证解决方案 ([@_philschmid](https://twitter.com/_philschmid/status/1987889931822236059))。
- **自我进化 Agent (GEPA)**：OpenAI x Bain 的新 Cookbook 展示了能够反思、从反馈中学习并进化自身指令的 Agent；GEPA 是其中的特色，开发者们强调了像 Python 的 inspect + GEPA 这种疯狂的组合 ([@DSPyOSS](https://twitter.com/DSPyOSS/status/1988021062727020589), [@LakshyAAAgrawal](https://twitter.com/LakshyAAAgrawal/status/1988008687156556200), [@JoshPurtell](https://twitter.com/JoshPurtell/status/1988025269006069845))。
- **评估与可靠性**：推荐观看一场多维度的评估演讲（数据、HCI、指标、工具）([@HamelHusain](https://twitter.com/HamelHusain/status/1987965289758421424))。Together AI 发布了基准测试指南 ([@togethercompute](https://twitter.com/togethercompute/status/1987949723106557975))。Weave 增加了仪表板和自定义评分器，以系统地在日志中发现 LLM 幻觉 ([@weave_wb](https://twitter.com/weave_wb/status/1987946840550240294))。新发布的 Agent 包括用于在 LangChain/LangGraph 上编排复杂 Web3 任务的 FlowAgent ([@LangChainAI](https://twitter.com/LangChainAI/status/1988012398176071728))。

**热门推文（按互动量排序）**

- **10,000 小时第一人称视角机器人数据集（开源）**：2,153 名工作人员，10.8 亿帧——“机器人领域的数据缩放时代已经到来” ([@eddybuild](https://twitter.com/eddybuild/status/1987951619804414416))。
- **Meta 的全语言 ASR**：支持 1600 多种语言；其中 500 种是首次支持；提供开源模型和语料库 ([@AIatMeta](https://twitter.com/AIatMeta/status/1987946571439444361))。
- **李飞飞谈空间智能与世界模型**：“将视觉转化为推理” ([@drfeifei](https://twitter.com/drfeifei/status/1987891210699379091))。
- **CMU “现代 AI 导论”课程 (Z. Kolter)**：面向低年级本科生的从零开始构建聊天机器人，资料即将发布 ([@zicokolter](https://twitter.com/zicokolter/status/1987938761498411376))。
- **动态混合精度**：“以最小能量 + 翻转为优化目标”作为前进方向 ([@elonmusk](https://twitter.com/elonmusk/status/1987994042937036805))。
- **ARC-AGI v1 声明**：通过多 Agent 进化测试时计算（test-time compute）和 GPT-5 Pro，在不到 12 小时内以低于 1 万美元的成本达到人类水平（85%）；社区审查正在进行中 ([@jerber888](https://twitter.com/jerber888/status/1987982067116777521))。

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Strix Halo 网络性能分析

- [**我测试了 Strix Halo 集群配合 ~50Gig IB，以观察网络是否真的是瓶颈**](https://www.reddit.com/r/LocalLLaMA/comments/1ot3lxv/i_tested_strix_halo_clustering_w_50gig_ib_to_see/) (热度: 601): **该帖子讨论了一个实验，旨在测试在使用 InfiniBand 和 Thunderbolt 连接的 Strix Halo 集群设置中，网络是否是瓶颈。作者使用 Mellanox ConnectX-5 Ex 100 Gig NICs 实现了约 55 Gbps 的网络带宽，而 Thunderbolt 为 10 Gbps。结果显示，在 Token 生成速度方面，Thunderbolt 的 10 Gbps 性能与 50 Gbps 的 InfiniBand 几乎相当，这表明对于使用 Strix Halo 的 llama.cpp 来说，高带宽可能并非必要。实验还指出网络利用率较低，表明延迟而非带宽可能是限制因素。作者得出结论，在 RCCL 支持可用之前，使用 llama.cpp 在 Strix Halo 上获得可用结果不需要昂贵的 IB 卡。** 一位评论者指出，由于 llama.cpp 不使用张量并行（Tensor Parallelism, TP），该测试可能没有意义，建议在 VLLM 或 Sglang 上使用 TP 进行测试会更合适。另一位评论者引用了 Jeff Geerling 类似实验的糟糕结果，建议对比研究结果。
    - Only_Situation_4713 指出该测试没有意义，因为 llama.cpp 不利用张量并行（TP），这意味着所有操作都是顺序执行的。他们建议在 VLLM 或 Sglang 等框架上启用 TP 进行测试，以更准确地评估性能瓶颈。
    - wishstudio 强调了网络延迟在张量并行（TP）设置中的重要性。他们指出，虽然 TP 中的数据交换量很小，但每一层都需要同步，这可能成为瓶颈。例如，对于像 gpt-oss-120b 这样拥有 36 层的模型，典型的以太网延迟为 250 微秒，可能会显著降低性能，而 InfiniBand (IB) 可以将延迟降低到个位数微秒，从而可能提高实际性能。
    - eleqtriq 引用了 Jeff Geerling 的一段视频，指出他在测试类似设置时结果很差。这表明网络确实可能是一个瓶颈，对比结果可以为性能差异和潜在优化提供见解。

### 2. Qwen3-VL OCR 能力与对比

- [**Qwen3-VL 的感知能力令人惊叹。**](https://www.reddit.com/r/LocalLLaMA/comments/1ot95gj/qwen3vls_perceptiveness_is_incredible/) (热度: 437): **该帖子讨论了** `Qwen3-VL-8B-Instruct-GGUF` **模型在光学字符识别（OCR）任务中的表现，特别是它在 4k 图像中准确转录并提供单词边界框（bounding boxes）的能力。该模型在图像 Token 计数为** `2300`**、温度为** `0` **的情况下，成功识别了图像中的所有六个单词并给出了精确的边界框，表现优于 Gemini 2.5 pro、Claude Opus 4、ChatGPT 5、DeepSeekOCR 和 PaddleOCR-VL-0.9B。值得注意的是，GLM-4.5V 也取得了完美的结果，但该帖子强调了 Qwen3-VL 在参数规模较小且缺乏特定 OCR 微调的情况下的效率。[模型链接](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)。** 评论者注意到 `Qwen3-VL-8B` 模型的出色表现，尤其是考虑到它比 `30B` 等大型模型更小。一位用户计划更新他们的 OCR 应用程序以使用此模型，表明了其实用价值。另一条评论认为 `8B` 模型是许多应用场景下的“不二之选”，强调了其效率和有效性。
    - MaxKruse96 强调了 Qwen3-VL 模型的性能，特别是 8B 变体，其在 `q8` 或 `BF16` 精度下的效率备受关注。在 GLM-4.5V 和 235B VL 发布之前，该模型被认为是该领域的佼佼者，表明同类模型中存在显著差距。讨论表明 Qwen3-VL 正在树立模型性能的新标准。
    - Putrid_Passion_6916 提到更新他们的项目 [deepseek_ocr_app](https://github.com/rdumasia303/deepseek_ocr_app) 以整合 Qwen3-VL，强调了该模型令人印象深刻的能力。他们指出 8B 或 4B 等较小参数模型对于许多任务已经足够，提供的性能与 30B 等较大模型相似，这突显了使用较小模型的效率和潜在成本节约。
    - cygn 讨论了图像分辨率对模型性能的重要性，以 AI Studio 中的 Gemini 2.5 Pro 为例。他们指出在模型评估中，选择中等分辨率还是低分辨率会影响结果，建议更高的分辨率可能会产生更好的结果。这强调了在模型评估中需要仔细考虑输入质量。

### 3. 使用 dLLM 的 BERT 聊天机器人

- [**BERTs that chat: turn any BERT into a chatbot with dLLM**](https://www.reddit.com/r/LocalLLaMA/comments/1osydym/berts_that_chat_turn_any_bert_into_a_chatbot_with/) (Activity: 390): **该帖子介绍了 dLLM，这是一个利用 *discrete diffusion*（离散扩散）技术将任何 BERT 模型转换为聊天机器人的库。该方法允许 BERT 模型（如 [ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large)）执行对话任务，其性能可与 [Qwen1.5-0.5B](https://huggingface.co/Qwen/Qwen1.5-0.5B) 等更大的模型相媲美。该项目提供了开源代码、checkpoints 以及详细的 [W&B report](https://api.wandb.ai/links/asap-zzhou/101h5xvg)，以确保透明度和可复现性。该方法专注于并行 token 生成，不同于传统的从左到右的自回归模型，旨在成为 diffusion language models 的全面教程资源。** 一位评论者对该 diffusion 模型没有同时解码多个 token 或以非顺序方式解码表示惊讶，他们认为这是 diffusion 模型的主要优势。
    - ithkuil 提出了关于 diffusion 模型预期行为的技术点，指出它们通常会同时或以非顺序方式解码多个 token。这种预期与传统的顺序解码方法形成对比，暗示了 dLLM 在聊天机器人实现中可能存在的创新领域或误解。
    - robberviet 询问了用于训练模型的数据，指出该仓库仅提到了“公开数据”而没有具体说明。这突显了 AI 项目中的一个常见问题，即缺乏详细的数据来源说明可能会影响模型性能的可复现性和信任度。
    - random-tomato 评论了 diffusion language models 聊天界面的新颖性，指出此类模型很少有功能性的聊天界面。这表明 dLLM 的实现可能比现有解决方案提供独特的功能或改进。

## 技术性较低的 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. 中国的 AI 进展与竞争

- [**中国现在确实在引领 Open Source AI**](https://www.reddit.com/r/DeepSeek/comments/1ot9y1j/china_really_carrying_open_source_ai_now/) (活跃度: 471): **该图片是一个 meme，描绘了中美在 Open Source AI 领域的竞争态势。它使用龙和鹰分别代表中国和美国，AI 和科技公司的 Logo 暗示了它们在这一竞争格局中的参与。帖子和评论强调了这样一种观点：中国在 Open Source AI 方面正取得重大进展，一些用户指出，像 DeepSeek 和 Qwen 这样的中国模型提供了与美国模型相当的质量，且通常是免费的。这反映了关于 AI 民主化以及中国在这一领域领先的战略举措的更广泛讨论。** 一些评论者认为，中国的 Open Source AI 模型是对抗美国公司的战略举措，提供了挑战美国模型主导地位的高质量替代方案。还有一种观点认为，中国模型通过提供免费的高质量工具，正在推动 AI 的民主化。
    - 一位用户强调了美国顶尖 AI 模型与中国免费模型之间的性能对等，指出尽管支付了美国公司最昂贵的订阅计划，但质量与中国的免费产品相当。这表明中国模型通过免费提供高质量模型，有效地推动了 AI 的民主化，挑战了美国公司的传统定价模式。
    - 另一位用户指出了 AI 社区中的一个关键区别：'Open Source'（开源）与 'Open Weight'（开放权重）模型之间的区别。虽然许多中国模型被称为 Open Source，但从技术上讲它们是 'Open Weight'，这意味着模型权重是可用的，但源代码不可用。这一区分对于需要对模型实现具有完全透明度和控制权的开发者来说至关重要。
    - 一位用户提到了特定的中国模型，如 DeepSeek 和 Qwen，并指出 Qwen 特别值得注意的一点是它不会简单地顺从用户，这对于更细致的 AI 交互来说是一个宝贵的特性。这突显了中国 AI 模型在提供多样化用户体验方面的多样性和复杂性。
- [**中国仅用 460 万美元就训练出了 GPT-5 的竞争对手 (Kimi K2)。**](https://www.reddit.com/r/ChatGPT/comments/1ot7fl4/china_trained_a_gpt5_competitor_kimi_k2_for_only/) (活跃度: 1196): **该图片展示了中国开发的 AI 模型 Kimi K2 与包括 GPT-5 在内的其他模型的性能对比。Kimi K2 因其在 Agent 搜索和编码任务中的强劲表现而受到关注，尽管其训练成本相对较低，仅为 460 万美元。这表明 Kimi K2 是 AI 领域一个极具成本效益的竞争对手，特别是在特定的技术领域。** 一些用户指出，虽然 Kimi K2 是一个不错的模型，但它可能无法完全匹配 GPT-5 或其他先进模型（如 Grok 4 或 DeepSeek）的能力。然而，其他人发现它是一个可靠的日常使用模型，表明了其尽管存在一些局限性，但具有实际效用。
    - NoDay1628 强调，虽然 Kimi K2 被吹捧为 GPT-5 的廉价替代品，但衡量 AI 模型能力的真实标准不仅仅是参数数量或训练预算。他们强调了“细微推理和安全性”的重要性，认为模型的实际表现可能与其理论规格有显著差异。
    - BuccellatiExplainsIt 对 Kimi K2 声称的 460 万美元训练成本表示怀疑，并将其与之前 DeepSeek 的情况类比，认为报道的数据具有误导性。他们指出这些说法缺乏透明度和问责制，暗示实际成本和能力可能与宣传的不同。
    - JackStrawWitchita 分享了使用 Kimi K2 的实际见解，指出虽然它并不完美，但作为日常主力工具表现良好。他们建议尝试不同的模型有助于理解每个模型的优缺点，表明 Kimi K2 为 ChatGPT 等更成熟的模型提供了一个可行的替代方案。

### 2. 幽默的 AI 批评与迷因 (Memes)

- [**Thoughts?**](https://www.reddit.com/r/OpenAI/comments/1otasm8/thoughts/) (热度: 3090): **这张图片是一个迷因（meme），幽默地批评了 AI（特别是 ChatGPT）在提供有关有毒浆果等潜在危险话题的准确信息方面的可靠性。它强调了在没有与权威来源进行交叉验证的情况下，依赖 AI 获取关键建议的风险。该迷因突显了人类判断的重要性，以及 AI 在处理细微或危及生命的查询时的局限性。** 评论者强调了不要依赖 AI 获取医疗建议的重要性，并指出虽然 AI 可以提供信息，但不应取代专业咨询。他们还指出，如果查询方式准确，AI 是可以正确识别已知有毒物品的。
    - Sluipslaper 强调了对 ChatGPT 识别有毒物质能力的实际测试，发现在查询已知的有毒浆果时，ChatGPT 能够正确将其识别为有毒。这表明该模型可以访问可靠的数据源，并能针对特定查询提供准确信息，尽管它不应取代专业建议。
    - Caddap 将使用 ChatGPT 比作进行 Google 搜索，并强调应将其作为工具而非个人研究的替代品。该评论强调了在解释 AI 生成的信息时进行尽职调查的必要性，因为该工具的力量在于其正确应用，而非盲目信任。
    - LunaticMosfet 指出，即使面对潜在的错误数据，ChatGPT 通常也会提供谨慎且详细的回答。该模型倾向于强调边缘情况并避免做出绝对性陈述，这表明其设计重点是提供平衡且谨慎的建议，而非定论。
- [**Sora 3 out before November 2026**](https://www.reddit.com/r/singularity/comments/1ot1m9w/sora_3_out_before_november_2026/) (热度: 499): **这张图片是一个迷因，幽默地调侃了《GTA 6》预计推迟至 2026 年 11 月发布的消息，并暗示 Sora 3 将在此之前问世。图片中的角色让人联想到典型的动作游戏场景：一名男子持枪，一名女子提着公文包，背景是城市景观。评论反映了对大型游戏开发周期缓慢的讽刺，一些用户开玩笑说 AI 有潜力加速游戏开发，甚至可能在当前版本完成之前就发布未来版本。** 评论者幽默地推测了 AI 在游戏开发中的作用，认为 AI 的进步可能会导致游戏续作发布得更快，甚至可能在当前版本完工前就已推出。
    - Weekly-Trash-272 强调了 AI 模型开发的飞速步伐，暗示在《GTA 6》发布之前可能会涌现出数个新模型。这突显了 AI 不断加速的能力，虽然目前还无法自主创建游戏，但在游戏开发的潜在应用方面正在缩小差距。
    - Setsuiii 指出了推迟游戏发布的相关风险，特别是在技术快速演进的背景下。他们指出，等到《GTA 6》发布时，其开发技术和工具可能已经过时，强调开发者需要适应新方法和工具以保持竞争力。
    - Normal_Pay_2907 推测了 OpenAI 自动化研究助手的推出时间表，认为它可能在 Sora 3 发布之前完成。这反映了开发 AI 工具以辅助复杂任务的更广泛趋势，这有可能改变各行各业的研究与开发流程。

### 3. 政治与经济中的 AI

- [**参议员 Bill Cassidy 在参议院发言时使用了一张看起来像是由 AI 生成的图表**](https://www.reddit.com/r/ChatGPT/comments/1ot0ddh/sen_bill_cassidy_on_the_floor_of_the_senate_with/) (热度: 1693): **讨论中的图像显示参议员 Bill Cassidy 在参议院发言时使用了一张图表，该图表似乎是由 AI 生成的，因为图中存在“可疑的伪影”，例如“80%”字样和美元符号。该图表旨在说明医疗保健资金的分配，将传统的保险模式与预付型灵活支出账户方案进行对比。该图表卡通式的简陋感以及潜在的 AI 生成痕迹，引发了人们对在正式场合使用视觉辅助工具的准确性和专业性的质疑。** 评论者对该图表的准确性以及政治家对问题的理解表示怀疑，其中一人指出这种对比是“苹果和西兰花”（风马牛不相及），另一人则认为该图表“100% 是 AI”生成的。
- [**OpenAI 每天在那些无聊的 Sora 视频上可能耗费高达 1500 万美元**](https://www.reddit.com/r/OpenAI/comments/1otjj7i/openai_could_be_blowing_as_much_as_15_million_per/) (热度: 830): **据报道，OpenAI 在其 AI 视频应用 Sora 上每天投入的成本高达** `$15 million per day`**，这引发了关于如此高额支出可持续性的讨论。这一财务策略可能会显著影响 OpenAI 的商业模式和未来的融资方式。文章指出，OpenAI 在该项目上的支出可能超过其收入的四分之一，从而引发了对这项投资长期可行性的质疑。更多详情请参阅 [Forbes 文章](https://go.forbes.com/B53aCk)。** 评论者将 OpenAI 的策略与 **Amazon** 和 **Uber** 等公司进行了类比，这些公司最初也是通过亏损经营来建立客户群。辩论的焦点在于，尽管目前处于亏损状态，但对 Sora 的高需求是否预示了其价值和未来的盈利潜力。
- [**Peak AI**](https://www.reddit.com/r/singularity/comments/1otfhbn/peak_ai/) (热度: 1350): **Steve 是一个 AI Agent 框架，允许用户使用自然语言描述任务，然后由 AI 进行解释和执行。该项目托管在 [GitHub](https://github.com/YuvDwi/Steve) 上，旨在通过充当单个或多个 Agent 来理解并根据上下文执行任务，从而简化用户交互。这在游戏场景中可能特别有用，例如玩家管理城市或军队等复杂系统时，允许他们通过语音指令而非传统控制方式发布命令。** 评论者讨论了 AI 伴侣在游戏中的潜力，认为虽然这个概念看起来微不足道，但它可以通过简化用户交互来彻底改变游戏玩法。然而，他们也指出了将 AI 生成的文本转化为可执行的游戏事件所面临的技术挑战。
    - AleriaGoodpaw 强调了将 AI 聊天机器人集成到游戏中的技术挑战，重点在于难以将“AI 聊天机器人的文本乱象”转化为可执行的游戏事件。这涉及复杂的自然语言处理和实时决策算法，以确保 AI 能够有效地在游戏环境中解释并执行玩家指令。
    - Scandinavian-Viking- 提出了一种 AI 在游戏中的潜在应用，即玩家可以通过自然语言指令控制城市或军队等复杂系统。这将需要能够理解和执行战略级决策的高级 AI，从而有可能改变策略游戏的界面和体验。
    - rowc99 讨论了 AI 技术的快速进步，认为基于当前局限性的怀疑论未能考虑到 AI 能力的指数级增长。这一观点暗示，随着 AI 和 VR 技术的日益先进和普及，未来的 AI 可能会显著增强游戏体验，特别是在沉浸感和交互性方面。

---

# AI Discord 摘要

> 由 Gemini 2.5 Pro Exp 生成的摘要之摘要之摘要
> 

**Kimi K2 的崛起与对下一代的期待**

- **Kimi K2 席卷榜单并超出预期**：Moonshot AI 的 **Kimi-K2-Thinking** 模型引起轰动，在 [LMArena Text 排行榜](https://lmarena.ai/leaderboard/text/overall)上以 **1447** 的惊人专家评分位列 **#2 开源模型**。它在 [Tau2 Bench Telecom 基准测试](https://xcancel.com/natolambert/status/1986507284491440623)中的表现也优于 **GPT-5** 和 **Claude 4.5**，且成本仅为后者的一小部分，尽管 Unsloth 团队在其 [GitHub](https://github.com/unslothai/unsloth) 上报告了一个潜在问题。
- **GPT-5.1 和 Gemini 3 传闻助长炒作热潮**：关于潜在的 **GPT-5.1 Pro** 发布推测不断，有人认为 **OpenAI** 正在等待 Google 先发制人，并猜测 OpenRouter 上的 **Polaris Alpha** 模型是其早期版本。与此同时，工程师们正热切期待 **Gemini 3**，讨论其颠覆编程工作的潜力，尽管由于当前模型的局限性，一些人仍持怀疑态度。
- **Sora 2 质量骤降，而开源语音 AI 大放异彩**：用户报告 **Sora 的视频质量** 明显下降，抱怨主体静止且音频质量差，一位用户甚至称其拥有 *目前所有视频生成工具中最差的视频和音频质量！* 相比之下，名为 **Maya1** 的新型 SOTA 开源语音 AI 在 [Hugging Face](https://huggingface.co/maya-research/maya1) 上亮相，拥有 **3B** 参数，并支持在单张 H100 上实现 **20 种人类情感**。

**内核奇才与硬件黑客挑战性能极限**

- **工程师发布经 GMP 验证的 INT8 GEMM 内核**：一位开发者发布了经过 GMP 验证的精确 **INT8×INT8→INT32 GEMM** 内核，在 **A100** 上实现了惊人的 **300.26 T-ops/s**。该代码证明了位对位（bit-for-bit）的正确性，可在 [Colab notebook](https://colab.research.google.com/drive/1D-KihKFEz6qmU7R-mvba7VeievKudvQ8?usp=sharing) 和 [GitHub 仓库](https://github.com/playfularchitect/WarpFrac.git)中进行社区验证和反馈。
- **Modular 的 MAX 引擎在新芯片上击败竞争对手**：据报道，用 **Mojo** 实现的推理引擎 **Modular's MAX** 在 **B200** 上击败了 **TensorRT**，在 **MI355X** 上击败了 AMD 的产品。这种性能表现，结合 Mojo 成为具备仿射类型（affine types）等特性的系统语言的目标，在急于避免将 C++ 包移植到 GPU 的 HPC 开发者中引起了巨大反响。
- **沿海空气腐蚀 RTX 3090，NPU 表现落后**：一位用户发现其新购入的 **RTX 3090** 因加湿器产生的矿物质堆积导致热点温度过高，并分享了散热器上[太平洋残留物（矿物质沉积）的照片](https://photos.app.goo.gl/3UPTmQKzJo81trTx9)。另外，关于将 **NPU** 用于 LLM 的讨论得出结论，尽管[最近的一篇论文](https://arxiv.org/abs/2412.11053)展示了在 Intel AI Boost NPU 上的推理，但其速度仍明显慢于专用 GPU。

**开发者平台遭受重重打击**

- **Cursor 用户面临崩溃、成本飙升和连接问题**：Cursor 用户报告了一系列问题，包括 Mac M2 上的系统级崩溃、**Sonnet 4.5** 导致的意外成本飙升（达到每分钟 **1.02 新西兰元**），以及 **Composor-1** 的频繁断连。这些问题还伴随着学生身份验证错误，以及使用个人 **OpenRouter** 密钥时出现的 *Unauthorized User API key* 错误。
- **Perplexity Pro 用户遭遇隐藏限制和封禁**：对于一些人来说，**Perplexity Pro** 的体验正在变差，用户遇到了不明显的每周 **Agent 任务限制**和上下文窗口上限，如[这张截图](https://cdn.discordapp.com/attachments/1047649527299055688/1437480737166327999/image.png?ex=691365c2&is=69121442&hm=2b15860ed2450a8cacc3ae448cf8effc4fbf7b73e597f69199d397e078e3e026&)所示。更令人沮丧的是，几位用户报告称因涉嫌欺诈被禁止参与推荐计划，其中一位表示 *Perplexity 欠我 30 美元*。
- **OpenAI 预示 Assistant API 的终结，Aider 分叉出社区版**：开发者正为 2026 年 **OpenAI** `assistant` **API** 的弃用做准备，这将需要将训练文件转换为 **JSONL** 以适配建议的替代方案 Responses API。在 Agent 领域，据报道 **aider** 的开发已转向社区驱动的 `aider-ce` [分支](https://github.com/dwash96/aider-ce)，用户称赞其取得了 *跨越式的进步*，并拥有令人惊叹的新 Agent 模式。

**驯服模型怪癖：从审查到持续学习**

- **AI 审查担忧引发社区强烈抵制**：针对日益严重的 **AI 审查**，不满情绪正在蔓延，多个服务器的用户担心信息环境受到*严格控制*。一些人认为 **OpenAI 正在剥夺公众获取信息的权利**，而另一些人则指出，过度的安全功能使得模型在许多技术应用中变得不切实际。
- **模型遭遇身份危机和内存故障**：模型表现出怪异的行为，**Qwen3-VL** 被 **Ollama** 搞混，尽管在处理图像数据，却认为自己是一个纯文本模型。同样，有用户报告 LM Studio 中的 **Gemma 4B** 似乎在不同的聊天历史中保留了上下文，引发了关于潜在 *Flash Attention 漏洞* 的猜测。
- **Google 的“嵌套学习”有望终结灾难性遗忘**：Google 引入了 **Nested Learning**，这是一种用于 [持续学习 (continual learning)](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/) 的新型机器学习范式，旨在通过将模型视为嵌套优化器层来解决灾难性遗忘问题。虽然这一概念引起了兴趣，但一些工程师质疑为什么 Google 没有将其与更标准的持续学习设置进行对比测试，并建议参考相关 [论文](https://arxiv.org/abs/2510.19788) 进行微调。

**开源项目凭借新工具和工作流蓬勃发展**

- **针对 Rust 开发者和 TPU 用户的新开源工具**：一个名为 [Ploke 的 Rust 编程 AI 接口](https://github.com/josephleblanc/ploke) 发布，它使用原生项目解析和自动语义搜索来改进上下文管理。在大型模型加速方面，**AutoXLA** 在 [GitHub](https://github.com/Locutusque/AutoXLA) 上首次亮相，这是一个实验性库，可自动执行 TPU 的模型分发和量化，实现比标准 Flash Attention 快达 **4 倍** 的性能。
- **ComfyUI 获得用于生产级图像的专业工作流**：NexusAI 在 GitHub 上发布了一套稳定、生产就绪的 [ComfyUI 工作流](https://github.com/NexusAI-Lab/ComfyUI-Professional-Workflows)。这些一键式工作流专为写实、动漫和商业图像生成而设计，目前正在 **v1.0.1** 版本中进行积极优化，以确保一致的细节再现。
- **工程师利用 DSPy Planner 解决 Agent 工具扩张问题**：一位开发者发布了关于 [利用 DSPy 解决 Agent 工具扩张](https://viksit.substack.com/p/solving-agent-tool-sprawl-with-dspy) 的指南，使用基于 DSPy 的规划器和编排器来管理多 Agent 工具的使用。与此同时，DSPy 持续进化，即将发布一个添加 **TOON** 支持的 PR，并提议根据 [Agent Client Protocol 标准](https://github.com/agentclientprotocol/agent-client-protocol) 集成对编写 **Agent CLI** 的原生支持。

---

# Discord：高层级 Discord 摘要

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Sora 2 Pro 引发访问权限辩论**：围绕 **Sora 2 Pro** 的访问权限和账号共享展开了讨论，一些用户对这种做法表示批评。
   - 争论点包括对遵守规则的付费用户的公平性，以及家庭共享账号的普遍做法。
- **OpenAI 面临规则批评**：一位用户对 **OpenAI** 的规则表示不满，建议人们不应遵守这些规则，并引用了 **Spotify** 和 **Meta** 据称违反规则的例子。
   - 这引发了关于伦理和公平的辩论，该用户认为：*如果你偷得少，你就是小偷；如果你偷得多，你就成了亿万富翁*。
- **Gemini 3 助长期待感**：用户热切期待 **Gemini 3** 的发布，推测其功能和潜在影响，尤其是在编程领域。
   - 一些人担心它可能会取代工作，而另一些人则由于当前 AI 模型的局限性而保持怀疑，提到了 **Google AI Studio** 和 **Nano Banana 2**。
- **Nano Banana 2 引发炒作和下架理论**：围绕 **Nano Banana 2** 的潜在发布及其功能充满热情，一些用户声称它已经可用。
   - 然而，对于可能被下架的担忧也随之而来，一位用户报告称该模型在发布仅 5 小时后就被移除，怀疑是提到其名称触发了该行动。
- **Kimi-k2-thinking 霸榜排行榜**：**文本排行榜 (Text leaderboard)** 进行了更新，`Kimi-k2-thinking` 现在位列 **开源模型第 2 名**，**总榜并列第 7 名**，在数学、编程和创意写作方面表现出色。
   - 查看 [文本排行榜](https://lmarena.ai/leaderboard/text/overall) 和 [专家排行榜](https://lmarena.ai/leaderboard/text/expert) 了解结果，其专家排行榜得分达到了令人印象深刻的 **1447**。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Comet 浏览器出 Bug 了！**：用户报告 **Comet 浏览器** 出现问题，如 **YouTube 搜索故障** 和在使用广告拦截器时出现 **视频播放问题**；重启可能有所帮助，且按钮 *不像 Chrome 那样快速流畅*。
   - 一位用户分享了 [Comet 的性能指标](https://cdn.discordapp.com/attachments/1047649527299055688/1437072124056571936/image.png?ex=69133ab5&is=6911e935&hm=ca6f47a1f181693b3d60ad89c0ef742a27d0caab93088ee5246ae8b7aa8bbc91&)。
- **YouTube 与广告拦截器发生冲突！**：YouTube 正在严厉打击广告拦截器，Chromium 的更新影响了拦截效果，尽管 [禁用广告拦截](https://link.to/adblock) 可以让网站正常运行；一些人建议使用 **Brave**，而另一些人则喜欢 **Comet**。
   - 分享了[这个 YouTube 链接](https://www.opera.com/features/ad-blocker)以获取广告拦截技巧。
- **Perplexity 推荐计划：欺诈风波！**：用户报告因涉嫌欺诈被 **Perplexity 推荐计划** 封禁，导致佣金被取消，一位用户表示 *Perplexity 欠我 30 美元*。
   - 关于批量封禁的理论正在流传，并引用了 [Perplexity AI 帮助中心](https://www.perplexity.ai/help-center/en/articles/11385821-partner-promotions-and-referral-programs)关于推荐系统运作方式的说明。
- **上下文窗口限制毁了 Perplexity Pro？**：Comet 浏览器用户报告达到了每周 **Agent 任务限制**，并遇到了不明显的 **上下文窗口限制**，这让拥有年度订阅的用户感到沮丧，详见[这张截图](https://cdn.discordapp.com/attachments/1047649527299055688/1437480737166327999/image.png?ex=691365c2&is=69121442&hm=2b15860ed2450a8cacc3ae448cf8effc4fbf7b73e597f69199d397e078e3e026&)。
   - 尽管如此，一位用户表示这些限制 *比 ChatGPT Plus 用户的每月 40 次使用要好*。
- **Rajkahini 的火箭式首秀！**：乐队 **The Orbits** 在 [Spotify](https://open.spotify.com/track/227ZkkO3LKPVABsHOoDS3w?si=a8603fc7cbb14e2c)、[YouTube Music](https://music.youtube.com/watch?v=GZAnCpgIO5g&si=QvIAfZLZdameuUfN)、[Apple Music](http://itunes.apple.com/album/id/1850285754) 和 [Amazon Music](https://music.amazon.com/tracks/B0FYY1C2BR) 上发布了他们的首支单曲 **Rajkahini**。
   - 该歌曲的歌词可在 [Genius](https://genius.com/The-orbits-indian-band-rajkahini-lyrics) 上查看。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Gemma 缓存故障令用户困惑**：一位用户报告称 **Gemma 4B** 似乎在 LM Studio 的不同聊天历史记录中保留了上下文，这是一种出乎意料的行为。
   - 另一位用户也曾遇到过 **Gemma 缓存上下文** 的情况，推测这可能是一个 *Flash Attention Bug*。
- **Qwen VL 终于能用了**：一位用户报告在下载了干净版本后，终于让 **Qwen3 VL** 在 LM Studio 中运行起来，并指出即使在 VRAM 有限的情况下，它处理图像的速度也很快。
   - 他们推测 **Qwen3 VL** 非常适合 **游戏中的 NPC**，这样它们就能真正“看到”东西。
- **华尔街低语者 LLM 依然难以捉摸**：一位用户正在寻找能够模仿顶级**华尔街记者**写作风格的微调 LLM，并拒绝了通用模型和提示策略。
   - 该用户坚持认为 System Prompts 和会话初始化 *完全失败*，且 *全是错误的幻觉 (Hallucination)*。
- **3090 被沿海污垢堵塞**：一位用户发现新购入的 **RTX 3090** 由于加湿器导致散热片上出现矿物质沉积，从而导致热点温度过高，并分享了积垢的 [照片](https://photos.app.goo.gl/3UPTmQKzJo81trTx9)。
   - 该用户还提到这张显卡闻起来像太平洋的味道，暗示它之前被用于沿海环境，并计划对 GPU 进行除垢和重新涂抹导热膏以解决问题。
- **NPU 运行 LLM 尚不成熟**：讨论了使用 **NPU** (神经网络处理单元) 运行 LLM 的可行性，并引用了一篇展示在 Intel AI Boost NPU 上进行 LLM 推理的 [论文](https://arxiv.org/abs/2412.11053)。
   - 尽管有潜力，但成员们普遍认为，由于 TOPS (每秒万亿次操作) 性能有限，NPU 在处理 LLM 任务时明显慢于专用 GPU。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Sonnet 4.5 成本让用户措手不及**：用户报告使用 **Sonnet 4.5** 的成本出现意外飙升，一名用户计算出费用达到了**每分钟 1.02 新西兰元**。
   - 一些用户建议采取优化成本的策略，例如将 **Sonnet 4.5** 用于规划，将 **Haiku 4.5** 用于编写，而另一些用户则发现 **Haiku** 需要过多的微观管理。
- **Composor-1 连接引发严重问题**：多名用户报告在使用 **Composor-1** 时频繁出现断开连接和停滞现象，并持续显示 *Planning next moves*（规划下一步行动）消息。
   - 潜在的解决方案包括降级到 **HTTP 1.1** 或重启 Cursor，有理论认为测试脚本保持 **ASYNC** 连接开启可能是导致问题的原因。
- **学生注册系统受阻**：用户在尝试验证学生身份时遇到错误，收到未指明的 **Error** 消息。
   - 推测认为，由于网上购买 ID 的情况增加，学生注册可能已暂时停止，或者某些国家可能不再支持学生折扣。
- **OpenRouter API 密钥触发访问警报**：用户在尝试使用自己的 **OpenRouter** 密钥时遇到 *Unauthorized User API key* 错误，一名用户报告称即使重新输入密钥问题依然存在。
   - 可能的解决方案包括验证密钥是否有效且已启用、禁用 Auto-model 选择，并确认所有 Cursor 设置准确无误。
- **Cursor 崩溃导致系统关机**：多名用户（尤其是 Mac M2 用户）报告在打开 Cursor 时出现系统级崩溃，包括紫屏和完全关机，一名用户已是第三次丢失所有历史聊天记录。
   - 故障排除建议包括重新安装 Cursor、使用不同的用户配置文件以及尽量减少系统资源负载，但根本原因尚不明确。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **NexusAI 发布 ComfyUI 专业工作流**：NexusAI 推出了一套稳定、生产就绪的 [ComfyUI 工作流](https://github.com/NexusAI-Lab/ComfyUI-Professional-Workflows)，专为写实、动漫和商业图像生成定制，将流程简化为一键式工作流。
   - 这些工作流目前在基础图像创建方面表现稳定，并正在进行积极优化，作为 **v1.0.1** 优化的一部分，以确保在不同随机种子下实现一致的细节再现。
- **Maya1 作为 SOTA 开源语音 AI 亮相**：一款名为 **Maya1** 的新型 **SOTA** 开源语音 **AI** 已在 [Hugging Face](https://huggingface.co/maya-research/maya1) 上发布，该模型拥有 **3B** 参数，设计运行在单张 H100 上。
   - 该 **AI** 支持语音设计和 20 种人类情感，标志着可访问语音技术的重大进步。
- **Rust 开发者获得开源 AI 接口**：一个用于 **Rust** 编码的开源 **AI 接口**已发布，提供项目的原生解析，通过自动语义搜索 bm25 关键词增强 LLM 上下文管理的关联性和效率，托管在 [GitHub](https://github.com/josephleblanc/ploke)。
   - 新功能允许用户通过交互式叠加层选择托管在 **OpenRouter** 上的模型，并提供了一个[示例](https://cdn.discordapp.com/attachments/897390720388825149/1437454527027740802/quick_example.webm?ex=69134d59&is=6911fbd9&hm=7320ebf8be3a9c2b4244b56acdfa66aed9c685b0496dac37171051c2ebb2fdcf&)。
- **AutoXLA 加速 TPU 上的大模型性能**：**AutoXLA** 是一个实验性库，利用 PyTorch/XLA 自动执行大型语言模型在 TPU 上的分发、优化和量化，性能比标准 Flash Attention 实现快达 **4 倍**，可在 [GitHub](https://github.com/Locutusque/AutoXLA) 上获取。
   - 通过使用自动分片、自定义注意力内核和量化感知加载等 TPU 感知功能扩展 Hugging Face Transformers 接口，它简化了大尺度部署和训练工作流。
- **Smol-Course 的 SFT 单元进度停滞**：多个解决 **smol-course** 问题的拉取请求（PRs），特别是关于 **SFT 单元**的 **Markdown 指令**，在 [GitHub](https://github.com/huggingface/smol-course) 上仍处于开启且未审阅状态。
   - 这一积压工作可能会阻碍课程的改进，成员们表示愿意协助审阅 PRs 以解决已发现的问题。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **INT8 GEMM 算子通过 GMP 验证！**：一名成员发布了一个经过 GMP 验证的精确 **INT8×INT8→INT32 GEMM** 算子，重点展示了在 **A100** 上的吞吐量（宏观 **300.26 T-ops/s**，微观 **2.026 T-ops/s**）以及位对位（bit-for-bit）的正确性，并邀请社区提供反馈。
   - 代码和验证过程可在 [Colab notebook](https://colab.research.google.com/drive/1D-KihKFEz6qmU7R-mvba7VeievKudvQ8?usp=sharing) 和 [GitHub repo](https://github.com/playfularchitect/WarpFrac.git) 中查看。
- **Blackwell 架构进入基准测试阶段**：成员们讨论了[一篇论文](https://arxiv.org/abs/2507.10789v2)，该论文对即将推出的 **Blackwell** 架构（尤其是 **5080**）进行了微基准测试（microbenchmarking），并探讨了消费级 **Blackwell (sm120)** 与数据中心级 **Blackwell (sm100)** 是否存在差异。
   - 一名成员建议成立一个理想的工作组，针对 **GB200** 撰写类似的论文。
- **Helion 的 Attention 算子性能竞赛**：成员们讨论了与 **FlexAttention** 相关的 [Helion attention 实现](https://github.com/pytorch/helion/blob/main/examples/attention.py)，观察到 **Helion** 的代码质量优于现有的 **Triton** 实现。
   - 会议强调，[Attention 算子的性能数据](https://cdn.discordapp.com/attachments/1425531180002054195/1436541653442887801/attention_perf.png?ex=691346eb&is=6911f56b&hm=dfbe035e2a6290dca86c612c31c2327934f6afffe40d6fe2fa5e7ce395feb546)已发布、开源且可复现，并且 [B200 算子已经可用](https://github.com/pytorch/helion/blob/main/examples/blackwell_attention.py)。
- **NVSHMEM 算子加速 LLM 推理**：一名成员分享了其团队在为 **LLM 推理**编写[基于 **NVSHMEM** 的低延迟通信算子](https://pssg.cs.umd.edu/blog/2025/beyond-nccl/)方面的工作，寻求关于多节点通信性能的反馈，并引发了对 **nvshmem4py** 集成的兴趣。
   - 一名成员使用了大量的 **nvshmem device API**，并提出可以协助进行库初始化，甚至可能提交一个 PR 来演示其具体实现方式。
- **WarpTrace 可视化工具预热流水线算子**：一名成员介绍了 **warptrace**，这是一个带有配套可视化工具的工具，用于观察流水线算子（pipelined kernels）。他指出，即使之前已经预热过算子，启动原子操作（atomics）系统仍需约 **50 微秒**。
   - [代码](https://github.com/aikitoria/nanotrace)目前仍在清理中，随后将上传至 GitHub，目前可以在[这里](https://aikitoria.github.io/nanotrace/)找到。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Kimi K2 遭遇提示词诱导的崩溃循环**：**Kimi K2 Thinking** 因特定的提示词（prompt）在多个提供商处出现了崩溃循环（crashloop），但该问题已通过协作努力得到**解决**。
   - 各团队正在积极调查原因。
- **Orchid AI Assistant 首次亮相遥遥无期**：**Orchid AI Assistant** 的预计发布日期被推测在未来 **2 到 48 个月**内。
   - 一名成员对这一漫长且模糊的估算给出了 *"crazy"* 的反应。
- **OpenRouter 考虑扩展视频支持**：一名用户希望 **OpenRouter** 支持视频和文本转语音（TTS）功能，并引用了[这条推文](https://x.com/scaling01/status/1986886020067938749)。
   - 另一名成员建议在 **OR Show** 中加入简短的**技术环节**，例如带有简短讨论的录屏。
- **Gemini 2.5 Token 消耗警报**：一名用户报告称，上传到 **Gemini 2.5 Flash** 的一段 24 秒、900x600 的视频消耗了超过 **800k 输入 token**，这与 Google 的文档不符。
   - [Token 文档](https://ai.google.dev/gemini-api/docs/tokens?lang=python#multimodal-tokens)规定视频的固定速率为 **每秒 263 个 token**。
- **自动化能力扫描想法引发关注**：一名成员建议实施某种**自动化能力扫描**，以检测模型/提供商随时间发生的变化。
   - 他们链接了一篇关于 [Cursor 的文章](https://joincolossus.com/article/inside-cursor/)作为示例，描述了如何通过一个*基础的 getWeather 工具调用*来检查功能变化。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Sora 的质量问题引发猜测**：用户报告 **Sora 视频质量**下降，抱怨主体静止不动且音频效果差，引发了人们对 **Sora 2** 可能是*目前所有视频生成工具中视频和音频质量最差的*担忧。
   - 然而，也有乐观情绪认为 **Sora 与 GPT-5 的集成**可能会带来改进。
- **本周发布 GPT-5.1 的传闻**：有关本周可能发布 **GPT-5.1 Pro** 的猜测不断，一名成员表示 *OpenAI 正在等待 Google 先发布产品*。
   - **OpenRouter** 上的 **Polaris Alpha** 模型也被传言是 **GPT-5.1** 的早期形态。
- **AI 审查担忧升温**：多位用户对日益增加的 **AI 审查**表示担忧，担心会创建一个*受到严格控制*的信息环境。
   - 一些人认为 **OpenAI 正在剥夺公众获取信息的权利**，对社会造成的伤害大于益处。
- **图片上传遭遇莫名 Bug**：几位用户在为他们的 **GPTs 上传图片**时遇到持续错误，尽管进行了故障排除，仍显示 *Unknown error occurred* 消息。
   - 这一问题已持续约一周，令 **Custom GPT 开发**受阻。
- **Assistant API 即将终结**：成员们正在讨论 2026 年即将到来的 **`assistant` API 弃用**及其对训练文件和相关 API 的影响，并附带了一张 [弃用通知截图](https://cdn.discordapp.com/attachments/1046317269069864970/1437264987260325908/2025-11-10_10_13_22-Chrome_Passwords_6.csv_-_OpenOffice_Calc.png?ex=69134594&is=6911f414&hm=c4707fe60ab8a1ba3e6525fa2dbef574e3f9257e043e894f0a9b6613d11adf90&)。
   - **Responses API** 被建议作为替代方案，这需要将文件转换为 **JSONL** 格式。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **AgentRL Qwen 集成面临模型权重延迟**：成员们对将 [AgentRL](https://github.com/THUDM/AgentRL) 与 **Qwen2.5 7B** 结合使用很感兴趣，但注意到模型权重尚未发布。
   - 虽然对 **Qwen3 4B** 的性能存在兴趣，但一些人对 **Qwen 2.5 14B** 的基准测试持怀疑态度。
- **UD 量化导致速度下降**：一名成员报告使用 **UD 量化（UD quants）** 时性能下降，仅达到 **1.5 tk/s**，而使用非 UD 的 **4.6 q2 KL** 时为 **4 tk/s**。
   - 该用户质疑 **UD 量化** 带来的质量提升是否值得牺牲速度，尤其是在角色扮演场景中。
- **AI 驱动的 GDDR7 短缺可能推迟 NVIDIA 5000 系列**：传闻称 NVIDIA 的 **RTX 5000 Super** 可能会因为 **AI 引发的 GDDR7 芯片短缺**而面临取消或涨价，因为据 [Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/nvidias-rtx-5000-super-could-be-cancelled-or-get-pricier-due-to-ai-induced-gddr7-woes-rumor-claims-3-gb-memory-chips-are-now-too-valuable-for-consumer-gpus) 报道，**3 GB 显存芯片**对消费级 GPU 来说变得过于昂贵。
   - 这凸显了 AI 热潮如何影响消费级硬件市场。
- **编辑距离（Levenshtein Distance）挫败域名抢注者**：一名成员利用 **Levenshtein Distance** 来识别 npm/pypi 软件包中的拼写抢注尝试（例如 `unsloth` 与 `umsloth`）。
   - 这种方法可以防止恶意攻击者利用常见的拼写错误来分发恶意软件或进行网络钓鱼攻击。
- **Roblox PII 模型旨在捕捉绕过过滤的行为**：一个新的 **PII 模型** 旨在通过适应不断演变的语言和模式来捕捉绕过过滤的行为，这与依赖 **NER** 和 Token 级检测的现有解决方案不同。
   - 它的目标是理解通信上下文，通过检测和混淆显式的 **PII** 文本，防止不良行为者进行与 **PII** 相关的对话。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Kimi 在价格上胜过 ChatGPT，但在记忆力上落败**：Discord 成员发现，根据方案的不同，**Kimi** 比 **ChatGPT** 更便宜，同时展现出令人印象深刻的语调，促使一些人考虑将其作为日常使用。
   - 尽管如此，据报道 **Kimi** 在长期追踪不同话题方面表现不佳，这与其其他吸引人的特性形成对比。
- **Deepseek V3.2 价格低于 OpenAI**：社区强调 **Deepseek V3.2** 是 **OpenAI** 的高性价比替代方案，成本为 **每 100 万 tokens 42 美分**。
   - 然而，一个缺陷是它缺乏像 **Kimi** 那样的工具推理能力。
- **Palantir 作为“AI 公司”面临质疑**：一场关于**针对 AI 的十亿美元豪赌**的讨论引发了争议，该赌注专门针对 **Palantir** 和 **NVIDIA**，引发了关于 **Palantir** 是否真正符合以 AI 为中心的公司定义的辩论。
   - 有人担心投资者可能误解了 **Palantir** 的产品，从而导致了做空头寸。
- **Mozilla 的 Any-LLM 工具挑战 Llama.cpp**：成员们正在关注 [Mozilla's Any-LLM](https://github.com/mozilla-ai/any-llm)，注意到它在推广 **Ollama** 的同时似乎忽视了 **llama.cpp**。
   - 关于它如何与 [python-instructor](https://python.useinstructor.com/) 等工具集成，存在一些疑问。
- **Google 的 Nested Learning 面临持续学习批评**：Google 推出的 **Nested Learning**（一种用于 [continual learning](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/) 的新 **ML paradigm**）引发了兴趣，但也招致了社区的质疑。
   - 一位成员想知道为什么 Google 没有在更多的 **continual learning** 场景下进行测试，并建议参考特定的 [论文](https://arxiv.org/abs/2510.19788) 进行微调。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi K2 Thinking 表现优于 GLM 4.6**：用户报告称，新的 **Kimi K2 Thinking** 模型表现优于 **GLM 4.6**，理由是基于口碑体验和一张 [图表](https://cdn.discordapp.com/attachments/1371757564005711973/1436501057751613581/s2yKvtY.png?ex=6913211d&is=6911cf9d&hm=4d79d7143360eedcca8f07c7a7cdac3f94e020675df5f638a600a8633d53ef92&) 的性能提升。
   - 尽管最初对依赖图表持怀疑态度，但社区普遍认同该模型增强的能力。
- **Unsloth 发现 Kimi-K2-Thinking 中的问题**：**Unsloth** 团队在新的 **Kimi-K2-Thinking** 模型中发现了一个潜在问题，并已通过其 [GitHub](https://github.com/unslothai/unsloth) 提交了报告。
   - 由于中国正值周末，Kimi 团队的回复可能会有所延迟；用户被引导至专门频道发布任何相关发现。
- **Kimi-For-Coding 额度消耗速度惊人**：用户正在迅速消耗其 **Kimi-for-coding** 的每周额度，部分用户在约 **1.5 到 2.5 天** 内就用完了 19 美元的方案额度。
   - 额度的快速消耗引发了关于升级到更高价格方案价值的讨论，一些用户暂时换回了 **GLM**。
- **Kimi-CLI 的搜索工具获得好评**：**Kimi-CLI** 中的原生网页搜索工具赢得了正面评价，一位用户因为 Kimi 卓越的搜索结果而取消了其 Brave Search 方案，详情见 [这条推文](https://fxtwitter.com/aravsrinivas/status/1986860050066108637)。
   - CLI 的搜索能力因其能够比其他搜索工具收集更多相关信息而受到重视。
- **Moonshot 要求 K2 质量**：部署 **Kimi K2** 需要通过 **Moonshot** 严格的 tool-calling 测试；否则，服务将被刻意中断。
   - 这一要求是 [K2-Vendor-Verifier program](https://github.com/MoonshotAI/K2-Vendor-Verifier) 的一部分，旨在确保供应商遵守 Moonshot 的质量标准。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 错误处理优于 Rust？**: **Mojo** 的 `try-except` 语法据称通过在正常路径（happy path）上执行 *placement new* 而不是使用 `Result`，提供了比 **Rust** 更好的性能。
   - 强类型错误（Typed errors）已在计划中，因为默认的 `Error` 类型开销较大。
- **MAX 在 B200 和 AMD MI355X 上力压对手**: **Modular** 的 **MAX** 是一个用 **Mojo** 实现的 cuBLAS/Cutlass/TensorRT/Pytorch/JAX 替代方案，其性能在 **B200** 上超过了 **TensorRT**，在 **MI355X** 上超过了 **AMD**。
   - 成员们认为 **Mojo** 是一门很好用的语言，并关注到了围绕 **MAX** 性能的宣传，但建议具体的许可（licensing）问题应咨询 <#1277617932897353809>。
- **Mojo 规划系统级语言霸主之路**: **Mojo** 旨在成为一门拥有仿射（affine）、线性（linear）和依赖类型（dependent types）的系统级语言，可能还会加入静态反射和代数类型系统，以摆脱 C 衍生语言的局限。
   - 其目标是尽可能榨取性能，使大多数不涉及系统调用的函数都能在编译时运行。
- **Mojo 拥抱 Python 风格特性，但它并非 Python**: 虽然 **Mojo** 包含 Python 风格的特性，但它不是 **Python**，其语义更接近 **Go**。
   - 例如，它的异常语法类似于 **Python**，但编译器会处理 *if err != nil* 部分，类似于 **Go**。
- **HPC 在 Mojo 中找到了知音**: 成员们分享了对 **Mojo** 在 **HPC**（高性能计算）领域潜力的见解，引用了最近的一篇 [Mojo-for-HPC 论文](https://arxiv.org/abs/2509.21039)，并指出了避免将 **C++** 包移植到 **GPU** 的好处。
   - 他们认为 **Mojo** 的**元编程（metaprogramming）**特性，特别是[用户定义方言](https://forum.modular.com/t/unlocking-high-performance-in-mojo-through-user-defined-dialects/41)和[阶段式编程](https://verdagon.dev/blog/impossible-optimization)（staged programming），将极大地受益于 **HPC** 项目。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Qwen3-VL 陷入身份危机**: **Qwen3-VL** 现在被 **Ollama** 搞糊涂了，认为自己是一个纯文本模型，尽管它承认如果模型实际上没有在图像数据上进行训练，那么图像数据将是毫无意义的。
   - 尽管模型知道自己接受过图像数据训练，但这种混乱依然出现了，引发了关于**模型身份和环境感知**的讨论。
- **Extropic 的演讲虽有“割韭菜”之嫌但仍引人关注**: 成员们发现 [**Extropic** 的一场演讲](https://www.youtube.com/watch?v=dRuhl6MLC78)非常有意思，尽管演讲者表现出一种“略显投机（grifty）”的风格，但仍引发了辩论。
   - 参与者承认直觉上感到不安，但也认可了所呈现内容的价值，增加了关于**信任和来源**的讨论。
- **嵌套学习范式持续进化**: Google 推出了 [Nested Learning](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/)，这是一种用于**持续学习（continual learning）**的新 **ML 范式**，原始论文可见[此链接](https://abehrouz.github.io/files/NL.pdf)。
   - 该方法通过嵌套学习过程来增强**持续学习**，使模型能够适应新任务而不遗忘先前学到的信息，引发了关于其实际应用和局限性的讨论。
- **转向通过递归学习**: 成员们讨论了注意力如何引导学习，指出移除注意力层会使记忆保持完整，结论是利用注意力可以更有效地引导学习，从而实现*可递归和自我提示的转向（steering）*。
   - 递归允许跨窗口的注意力，像 **RWKV** 这样的模型在保留记忆的同时消除了二次方问题，这有利于更快地搜索记忆。
- **算力在全球地理分布**: 一位成员分享了关于**各国全球算力份额**的[链接](https://www.reddit.com/r/singularity/comments/1oraof2/global_share_of_compute_per_country/)，引发了关于分配和准入的辩论。
   - 讨论涉及**数字主权、资源公平以及算力优势的地缘政治影响**，并引出了一些幽默评论，称 **EU**（欧盟）是他们最喜欢的国家。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **缺失 Cursor 触发 Sequoia 变动**：一位成员注意到缺失的 Cursor 导致了 [Sequoia 的变动](https://x.com/amir/status/1986904426595209664)。
   - 未提供足够信息以生成二次摘要。
- **Terminal-Bench 2.0 与 Harbor 亮相**：Alex Shaw 宣布发布 **Harbor**（一个沙箱化 Agent 评估框架）以及 **Terminal-Bench 2.0**（一个难度更高的 89 任务基准测试）；尽管难度增加，但由于任务质量更高，最高分与 TB1.0 持平。
   - **Harbor** 还作为 TB2.0 的官方测试套件，并包含提交文档；更多详情请参阅 [Terminal-Bench 2.0 & Harbor Launch](https://xcancel.com/alexgshaw/status/1986911106108211461)。
- **Kimi K2 在 Tau2 Bench 上碾压 GPT-5**：月之暗面（Moonshot AI）的开源 **Kimi K2** 模型在 Tau2 Bench 电信基准测试中表现优于 **GPT-5** 和 **Claude 4.5**，而成本仅为后者的六分之一，详见[此 X 帖子](https://xcancel.com/natolambert/status/1986507284491440623)。
   - 聊天参与者警告称，中国模型以更低价格实现性能提升，正加剧美国实验室的压力，并呼吁美国加快开源速度，以留在“模型文化战争”中。
- **EdgeTAM 登陆 Hugging Face**：Meta 的实时分割追踪器 **EdgeTAM** 现已在 Hugging Face Transformers 上以 Apache-2.0 协议发布，其运行速度比 **SAM2** 快 **22 倍以上**，在无需量化的情况下，在 **iPhone 15 Pro Max** 上可达到 **16 FPS**，详见[此 X 帖子](https://xcancel.com/mervenoyann/status/1986785795424788812?s=46)。
   - 未提供足够信息以生成二次摘要。
- **Google 的 Nested Learning 防止灾难性遗忘**：Google Research 展示了 **Nested Learning**，这是一种将模型视为嵌套优化器层的持续学习（continual-learning）框架（概念验证模型名为“Hope”），旨在减少灾难性遗忘并扩展长上下文限制，请查看 [Google Research 推文](https://xcancel.com/googleresearch/status/1986855202658418715?s=46&t=eWVlK1PU8XfB6f402GJJ9g)。
   - 未提供足够信息以生成二次摘要。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Weave Evals 作为 WandB 的替代方案**：一位成员询问如何使用 **Weave 的评估数据**来复制 **Weights and Biases (WandB) 报告**和功能。
   - Liquan Pei 表示有兴趣为相关项目做贡献，而其他人则推荐 [sesterce.com](https://sesterce.com) 作为 **GPU 云服务商**，提供用于内核分析（kernel profiling）的专用裸金属访问。
- **明确 QAT 优于 PTQ 的原因**：一场讨论明确了为什么 **量化感知训练 (QAT)** 优于 **训练后量化 (PTQ)**，指出 *QAT* 是微调的一种形式，它训练模型对量化误差具有鲁棒性。
   - 在训练期间模拟量化过程，使模型能够“恢复”原本在 **PTQ** 中会损失的准确率。
- **Autoencoder 过拟合可视化**：成员们讨论了 Autoencoder 中的“过拟合”概念，其中一人分享了一个具有一维潜变量（1D latents）的**过拟合 Autoencoder** [示例](https://cdn.discordapp.com/attachments/747850033994662000/1436946342282006658/Screenshot_2025-11-09_at_6.11.30_AM.png?ex=69136e51&is=69121cd1&hm=aa503c9203607ea834d4d772a3110d5f2f3c3a775cfa76a81f37374a5d121c93&)。
   - 讨论集中在瓶颈（bottleneck）是否真的能防止过拟合，并提供了进一步的[评估](https://cdn.discordapp.com/attachments/747850033994662000/1436946433193283584/Screenshot_2025-11-09_at_6.11.57_AM.png?ex=69136e66&is=69121ce6&hm=efae038e11b80e6730958e866a66a3b336947b91b682aa20190c6e1ef0d09c3a&)。
- **SAE 论文被 AAAI 26 接收**：一篇解决 **LLM** 中 **SAE 问题**和**非线性特征关系**的论文已被 **AAAI 26** 接收，并可在 [ArXiv](https://arxiv.org/abs/2507.00269) 上获取。
   - 该论文旨在通过建模特征之间的**非线性关系**来减少重构误差和 KL 散度误差，从而区分共现特征与“绑定”特征。
- **针对 Anthropic 博客文章的新读书小组启动**：一位成员启动了一个读书小组来剖析特定的 Anthropic 博客文章，并[分享了 Discord 频道链接](https://discord.com/channels/729741769192767510/1437089667920171108)。
   - 同时分享的还有 [YouTube 链接](https://youtu.be/kkfLHmujzO8?si=d0Wa2u0QTmO8-ptp)，其中包含一份贡献 YouTube 视频中首选电影场景的指南。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **4090 依然是王者，5090 表现平平**：**RTX 4090** 仍然是首选，因为从 **3090** 到 **4090** 的跨越非常显著，而根据 CommaCon 的 [见解](https://en.wikipedia.org/wiki/Mutation_testing)，**5090** 仅提供了微小的改进。
   - 这与 tinygrad 正在进行的关于硬件优化的开发决策密切相关。
- **Tinygrad 转向 pyproject.toml**：正如 [Meeting #95](https://github.com/tinygrad/tinygrad/issues/95) 中讨论并由一名成员强调的那样，tinygrad 将转向 **pyproject.toml**，相关更改已在 [PR #13189](https://github.com/tinygrad/tinygrad/pull/13189) 中提出。
   - 此次迁移旨在简化项目内的依赖管理和构建流程。
- **Hatch 引发构建系统争论**：通过 [PR #13189](https://github.com/tinygrad/tinygrad/pull/13189) 引入 **Hatch** 引发了对其必要性的质疑，以及 Python 标准库或 `setuptools` 是否是可行的替代方案。
   - 一些人认为 **Hatch** 简化了开发，可能使其他工具变得多余。
- **`UOps.after` Bug 已修复**：成员们讨论了 `UOps.after` 使用时机的限制，最初建议它只能应用于 buffer，而不能应用于比较。
   - 这被确定为一个 [linearizer bug](https://github.com/tinygrad/tinygrad/commit/ffb9e8396f9f78c7cd986f9e93be6dfb0fde88ed)，当在 `B` 和 `A` 的索引上都调用 `.valid` 时会触发，该问题 [随后已解决](https://github.com/tinygrad/tinygrad)。
- **`Tensor.from_blob` 在 MPS 上出现故障**：用户在将 MPS (Metal Performance Shaders) 设备上的 Torch tensor 转换为 tinygrad 时遇到了 `Tensor.from_blob` 的问题，导致了与内存访问相关的错误。
   - 从 Torch MPS tensor 直接转换为 CPU 上的 tinygrad tensor 是可行的（可能带有副本），但直接转换为 Metal 设备会导致 Jupyter kernel 因设备不匹配错误而崩溃，要求 Torch tensor 必须在同一设备上。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Planner 治理 Agent 工具泛滥**：一位成员发布了一篇使用 **基于 DSPy 的 planner** 和编排器来解决 **多 Agent 工具使用** 问题的文章，并征求反馈：[Solving Agent Tool Sprawl with DSPy](https://viksit.substack.com/p/solving-agent-tool-sprawl-with-dspy)。
   - 该文章探讨了使用 DSPy 进行 **多 Agent 工具使用** 的解决方案，重点关注规划和编排。
- **DSPy 优化令工程师受挫**：成员们在使用 **MIPROv2** 进行 **DSPy 优化** 时遇到错误，并请求提供有关设置和所遇错误的更多细节。
   - 有人提出疑问，这是否反映了 dspy 内部的 **BAML Adapter** 功能。
- **TOON Adapter 助力 DSPy**：一位成员正在准备一个 PR，旨在将 **TOON** 支持注入 **DSPy**，这引发了对性能评估的热情，但也存在对潜在 *性能下降* 的担忧。
   - 强调了 **评估** 对于衡量 **TOON 性能** 至关重要，特别是针对结构化输出，以查明任何性能下降。
- **CLI Agent 获得 DSPy 支持**：出现了一个 issue 来跟踪为使用 **DSPy** 编写 **Agent CLI** 集成 **原生支持** 的工作项，这符合 [Agent Client Protocol 标准](https://github.com/agentclientprotocol/agent-client-protocol)。
   - 讨论在思考这一举措应该作为由 **DSPy** 维护的兄弟项目发展，还是作为由 **ZED ACP** 支持增强的第一方模块发展。
- **Discord 呼吁分享 DSPy 成功案例**：有人提议在 **Discord** 服务器上设立专门板块，用于展示和剖析 **DSPy 成功案例**，按任务类型（如分类或信息提取）细分，并附带详细的设置说明。
   - 建议包括开设针对 Student 和 Teacher 模型的子论坛，涵盖 **Qwen3**、**Llama**、**GPT-5**、**Sonnet** 和 **Opus**。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Kimi 模型在 Aider 中的智能表现引发讨论**：成员们正在讨论 **Kimi 模型** 在 **aider** 中是否显得更聪明，原因是提示词较少冗长，这表明指令过载可能会阻碍重度 Agent 编程工具的性能。
   - 理论认为，模型会放大给定的词汇，并在自主工作期间被错误的思路误导，而 **aider** 通过强制要求更高的特异性并使用结构化程度较低的内部框架来提高性能。
- **Aider 的开发转向 'aider-ce' 分支**：据报道，**aider** 的开发已转向 **aider-ce** 分支，因为主要维护者最近没有向原始仓库提交代码，详见 [此 issue](https://github.com/Aider-AI/aider/issues/4613)。
   - 成员们称赞 [dwash96/aider-ce 仓库](https://github.com/dwash96/aider-ce) 有着 *飞跃式的改进*，其中一人称新的 Agent 模式 *令人惊叹*。
- **Aider 出现 JSON 分块策略**：用户在向 **Aider** 提供大型 **Figma 对象的 JSON 文件** 时正面临 **token 限制**，因此建议先描述文件，并要求 **Aider** *编写一个脚本将其分解为连贯的块*。
   - 另一位成员建议 **LLM** 可能不是处理该任务的正确工具，应先总结 JSON，以便模型随后帮助编写执行下一步的代码。
- **Claude 搞笑地自我纠错**：一位成员分享了一张 **Claude** *字面上通过编写函数来修复其格式错误* 的图片。
   - 成员们提议需要一个频道来记录有趣的语言模型函数调用。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **规范发布定于 2025 年 11 月 25 日**：`2025-11-25` 规范发布已排期，与 [用于定稿的 SEP](https://github.com/orgs/modelcontextprotocol/projects/26/views/8) 保持一致，预计在 **2025 年 11 月 14 日** 进行规范冻结。
   - 澄清指出，在规范冻结后，**SDK 变更** 可以独立进行，因为 SEP 视图主要关注规范用语。
- **SEP-1330 等待 SDK 审查**：在完成变更后，**SEP-1330** 的“等待 SDK 变更”标签已移除，目前正等待 **TS/Python SDK** 的审查与合并，以及规范/模式的更新。
   - 这一步对于确保 SDK 与 **SEP-1330** 中详细说明的增强功能和更新保持一致至关重要。
- **关于 Agent 访问 Slack/Gsuite API 的疑问**：一位用户询问如何授予 Agent 访问 **Slack** 和 **Gsuite API** 的权限，并对涉及密钥的设置和示例用法提出疑问。
   - 他们链接到了一个 [关于代码执行的相关讨论](https://discord.com/channels/1358869848138059966/1436084770114240512/1436365734027460720) 以获取更多细节，寻求环境配置方面的澄清。
- **MCP 客户端中的 PII 拦截验证**：一位成员对 **MCP 客户端**（如 Cursor 和 Claude）在识别和拦截 **PII 数据** 方面的准确性验证表示担忧。
   - 他们质疑如何验证这些客户端的实现是否正确，以及它们如何准确且确定地识别 **PII**。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **VEO3 连接中断，Manus 失去视频功能**：一名用户报告与 **VEO3** 的连接断开，导致 **Manus** 失去了制作视频的能力。
   - 该用户请求*从旧账户下载文本或代码并将其上传到新账户*。
- **因“极其愚蠢”的 Token 费率取消订阅**：一名用户表示 Token 费率*极其愚蠢*，在**几小时内就消耗了 99 美元**，因此取消了订阅，转而选择*更好且更便宜的方案*。
   - 他们补充道：*你们的服务定价简直疯了。市面上还有更好、更便宜的选择。*
- **工程师展示专业背景：工作流自动化、LLM 集成、区块链**：一位擅长**工作流自动化、LLM 集成、RAG、AI 检测、图像/语音 AI 以及区块链开发**的资深工程师介绍了自己，强调了其在实际落地应用方面的丰富经验。
   - 他们使用 **Dspy、OpenAI APIs 和自定义 Agents** 构建了自动化流水线，显著缩短了响应时间，并部署了带有向量数据库和混合搜索的高级 **RAG pipelines**。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Windsurf Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

您收到此邮件是因为您在我们的网站上选择了订阅。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord: 频道详细摘要与链接

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1436445242420236490)** (1385 条消息 🔥🔥🔥): 

> `Sora 2 Pro, OpenAI Rules, Billionaire Thief, Gemini 3, Nano Banana 2` 

- **Sora 2 Pro 访问权限引发账号共享争论**：成员们讨论了 **Sora 2 Pro** 的访问权限，一名用户提到他们仅将其用于 **Sora**，这引发了关于账号共享及其是否可接受的讨论。
   - 一些用户反对账号共享，理由是其他人支付了全额费用并遵守规则；而另一些人则认为这是一种普遍做法，即使在美国，家庭成员之间共享方案也很常见。
- **OpenAI 规则引发批评**：一名用户对 **OpenAI** 的规则表示不满，称：*这就是为什么人们不应该遵守规则，否则你会被坑*，并提供了有关 **Spotify** 和 **Meta** 涉嫌违反规则的文章链接。
   - 他们认为*窃钩者诛，窃国者侯（偷得少是小偷，偷得多就成了亿万富翁）*，这引发了关于伦理和公平的辩论。
- **Gemini 3 预热，用户期待发布**：用户们正热切期待 **Gemini 3** 的发布，并对其能力和潜在影响（尤其是在编程领域）进行了推测。
   - 一些用户认为 **Gemini 3** 可能会取代工作，而另一些人则持怀疑态度，理由是当前 AI 模型的局限性以及需要进一步测试。讨论中还提到了 **Google AI Studio** 和 **Nano Banana 2**。
- **Nano Banana 2 引发热议及下架推测**：用户对 **Nano Banana 2** 的潜在发布及其能力感到兴奋，一名用户声称它已经发布，但其他用户对可能的下架表示担忧。
   - 一些尝试过早期版本的用户对该模型在发布仅 5 小时后就被移除表示失望，推测该模型一旦被点名提到就会被下架。
- **对 AI 能力的挫败感**：一些用户对当前 AI 模型的局限性表示沮丧，特别是在翻译、表格填写和图像编辑等领域。
   - 一名用户指出 **ChatGPT** 虽然便宜，但在翻译长文本时很吃力，而其他人则注意到 **LMArena** 中的 **GPT-image-1** 表现不佳，且无法显示多个选项。

---

### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1436492827029803199)** (3 条消息): 

> `图像编辑排行榜、抽象艺术大赛获胜者、文本排行榜更新、Kimi-k2-thinking 模型` 


- **Reve-Edit-Fast 跻身前 5 名**：**图像编辑排行榜**已更新，`Reve-edit-fast` 现已公开发布并位列**前 5**；查看 [图像编辑排行榜](https://lmarena.ai/leaderboard/image-edit) 了解结果。
- **10 月抽象艺术大赛决出胜者**：10 月抽象艺术大赛的获胜者已公布，点击 [此处](https://discord.com/channels/1340554757349179412/1422967966177431694/1422999438233698549) 查看其生成作品。
- **Kimi-k2-thinking 模型排名领先**：**文本排行榜**已更新，`Kimi-k2-thinking` 目前在**开源模型中排名第 2**，并在**总榜排名并列第 7**，在数学、编程和创意写作类别表现优异；查看 [文本排行榜](https://lmarena.ai/leaderboard/text/overall) 了解结果。
- **Kimi-k2-thinking 拥有专家级评分**：在 [专家排行榜](https://lmarena.ai/leaderboard/text/expert>) 上，`Kimi-k2-thinking` 获得了令人印象深刻的 **1447** 分。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1436445224338460823)** (1132 条消息🔥🔥🔥): 

> `Comet 浏览器问题、YouTube 广告拦截、Perplexity 推荐计划故障、上下文窗口限制、Perplexity Pro 价值` 


- ****Comet 浏览器出现 Bug！****：用户报告 **Comet 浏览器**存在问题，包括在启用广告拦截器时 **YouTube 搜索故障**和**视频播放问题**，部分用户建议重启浏览器以解决问题。
   - 一位用户分享了其 Comet 实例的性能指标（[图片](https://cdn.discordapp.com/attachments/1047649527299055688/1437072124056571936/image.png?ex=69133ab5&is=6911e935&hm=ca6f47a1f181693b3d60ad89c0ef742a27d0caab93088ee5246ae8b7aa8bbc91&)），另一位用户表示 *按钮操作不够快速流畅，不像 Chrome 那样*。
- ****YouTube 与广告拦截器：一场永无止境的战斗****：用户正在讨论 **YouTube 对广告拦截器的打击**，指出最近的 Chromium 更新影响了广告拦截效果，一位用户发现 [禁用广告拦截](https://link.to/adblock) 可以让网站正常运行。
   - 虽然有人建议切换到 **Brave** 以使用其内置的广告拦截功能，但其他人认为 **Comet** 最为稳定，并分享了此 [YouTube 链接](https://www.opera.com/features/ad-blocker) 获取广告拦截技巧。
- ****Perplexity 推荐计划面临欺诈指控！****：多位用户报告因涉嫌欺诈活动而被**禁止参加 Perplexity 推荐计划**，佣金被取消，并投诉在与 Perplexity 支持部门沟通时遇到的问题。
   - 一位被欠 30 美元的用户发帖称：*Perplexity 欠我 30 美元*，其他人则对封号潮以及推荐系统根据 [Perplexity AI 帮助中心](https://www.perplexity.ai/help-center/en/articles/11385821-partner-promotions-and-referral-programs) 的运作方式提出了各种理论。
- ****碰壁：上下文窗口限制已实施？****：Comet 浏览器用户报告达到了每周 **Agent 任务限制**，并遇到了不明显的**上下文窗口限制**，这让支付年费订阅的用户感到沮丧，如该 [截图](https://cdn.discordapp.com/attachments/1047649527299055688/1437480737166327999/image.png?ex=691365c2&is=69121442&hm=2b15860ed2450a8cacc3ae448cf8effc4fbf7b73e597f69199d397e078e3e026&) 所示。
   - 一位用户仍持乐观态度，指出这些限制仍然 *比 ChatGPT Plus 用户每月 40 次的使用限制要好*。
- ****Perplexity 仍然是性价比之王？****：尽管最近出现了一些问题，部分用户仍认为 **Perplexity Pro** 是 *物超所值的最佳 AI 产品*，而其他人则提到了获取免费或折扣订阅的方法，例如哈萨克斯坦的优惠或通过 PayPal 支付。
   - 然而，一位用户在回应免费订阅优惠时宣称：*我不需要免费计划，因为我可以使用订阅计划*。


  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1436648152425041950)** (3 条消息): 

> `The Orbits 首支单曲，Perplexity AI 上的可共享线程` 


- ****The Orbits** 发布 **Rajkahini** 单曲**: 乐队 **The Orbits** 宣布在包括 [Spotify](https://open.spotify.com/track/227ZkkO3LKPVABsHOoDS3w?si=a8603fc7cbb14e2c)、[YouTube Music](https://music.youtube.com/watch?v=GZAnCpgIO5g&si=QvIAfZLZdameuUfN)、[Apple Music](http://itunes.apple.com/album/id/1850285754) 和 [Amazon Music](https://music.amazon.com/tracks/B0FYY1C2BR) 在内的多个流媒体平台发布其首支单曲 **Rajkahini**。
   - 歌曲歌词可在 [Genius](https://genius.com/The-orbits-indian-band-rajkahini-lyrics) 上查看。
- **Perplexity AI 提示 **Shareable** 线程**: Perplexity AI 提醒用户确保其线程设置为 **Shareable**（可共享）。
   - 该公告包含一个指向 [Discord 消息](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825) 的链接，展示了如何进行设置。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1436970550944595969)** (5 条消息): 

> `Perplexity Pro 额度、API key 生成、额度结转` 


- **Perplexity Pro 额度是否自动发放？**: 一位用户询问 **Perplexity Pro** 的每月额度是否自动发放，并提到他们订阅已近一年，但仅在前两个月收到了额度。
   - 一名成员回复称，生成 **API key** 将自动从每月额度中扣除，且**额度不会结转 (roll over)**。
- **额度不再结转了？**: 一位成员询问 **Perplexity Pro** 额度以前是否可以结转。
   - 另一位成员给出了否定回答。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1436449185246351360)** (497 条消息🔥🔥🔥): 

> `Gemma 缓存、Qwen VL 运行正常、写作风格 LLM` 


- **DGX 引发关注**: 一位成员分享了指向 **Nvidia DGX** 系统的 [Microcenter 链接](https://www.microcenter.com/product/699008/nvidia-dgx-spark)，这可能引发了对高性能 AI 硬件的兴趣。
   - 该成员未提供更多背景信息，留给其他人猜测其与讨论的相关性。
- **Gemma 可能在缓存上下文**: 一位用户报告称，**Gemma 4B** 似乎在 LM Studio 的不同聊天历史记录中保留了上下文，但在重新加载模型后无法复现该行为。
   - 另一位用户也曾遇到过 **Gemma 缓存上下文**的情况，不确定是故障还是特性，但行为非常相似；曾尝试确认是 feature 还是 bug，但那次之后再未发生，*可能是 flashattention bug，但从其架构来看极不可能*。
- **Qwen VL 终于可以运行了**: 一位用户报告在下载纯净版本后，终于让 **Qwen3 VL** 在 LM Studio 中运行，并指出即使在 VRAM 有限的情况下，它处理图像的速度也很快。
   - 他们推测 **Qwen3 VL 非常适合游戏中的 NPC**，这样它们就能真正“看到”画面。或许可以达到 1 fps 左右。
- **寻找具有特定写作风格的 LLM**: 一位用户正在寻找一个经过微调的 LLM，能够模仿顶级**华尔街记者**的写作风格，并排除了通用模型。
   - 其他成员建议使用 system prompts 或会话初始化来引导模型，或者微调具有特定写作风格的模型，但该用户坚持认为已经尝试过这些选项，且*完全失败，全是错误的幻觉 (hallucination)*。
- **LM Studio 重置模型目录的问题**: 一位用户报告称，更新到最新版本的 **LM Studio** 似乎重置了他们的默认模型目录，且 **Change** 按钮无响应。
   - 该用户附上了一张显示该问题的设置面板截图。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1436471300418961459)** (662 messages🔥🔥🔥): 

> `3090 Performance, GPU Cooling, Multi-GPU Setups, AMD vs Nvidia, LLM Performance` 


- **3090 面临矿物质堆积**：一位用户发现新购入的 **RTX 3090** 由于加湿器在散热片上产生的矿物质沉积导致热点温度过高，并分享了堆积物的[照片](https://photos.app.goo.gl/3UPTmQKzJo81trTx9)。
   - 该用户还提到显卡闻起来有太平洋的味道，暗示其之前可能在沿海环境中使用，并计划通过除垢和重新涂抹 GPU 导热膏来解决此问题。
- **多 GPU 配置对比**：成员们讨论了本地 LLM 多 GPU 设置的细微差别，指出虽然多 GPU 增加了 VRAM，但由于 [GPU 顺序使用](https://www.reddit.com/r/LocalLLaMA/s/HWLU2NEK0m)，处理速度并不总是成比例增长。
   - 据分享，与多个 3090 相比，Pro 6000 在 VRAM 利用率方面具有优势，因为其 VRAM 开销较小，但需要大量的资金投入。
- **Nvidia vs AMD 基准测试**：讨论了 **Nvidia** 和 **AMD** GPU 之间的性能差异，特别是关于 **CUDA**、**CUDA 12** 和 **Vulkan** API，[一些用户](https://forums.guru3d.com/threads/amd-software-pro-edition-25-q3-for-windows.457313/)报告在 Nvidia 显卡上使用 Vulkan 性能更好。
   - 一些人发现 **CUDA** 速度更快，导致部分成员因感知到的性能优势考虑从 **Nvidia** 转向 **AMD**，而其他人则指出了 **ROCm** 配置和驱动兼容性的潜在问题。
- **小型机箱气流不佳**：一位用户分享了使用小型钢化玻璃机箱的经验，指出即使使用高端 **Noctua** 散热器，气流也受到严重限制并导致热节流。
   - 另一位成员建议可能是主板/插槽弯曲，或者是热管温度过高导致下垂脱离。
- **NPU 几乎不适合运行 LLM**：围绕使用 **NPU** (Neural Processing Units) 运行 LLM 的可行性进行了讨论，参考了一篇展示在 Intel AI Boost NPU 上进行 LLM 推理的[论文](https://arxiv.org/abs/2412.11053)。
   - 尽管有潜力，成员们普遍认为 NPU 在 LLM 任务中明显慢于专用 GPU，因为其 TOPS (Tensor Operations Per Second) 性能有限。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1436453442745929930)** (892 messages🔥🔥🔥): 

> `Sonnet 4.5 Pricing, Composor-1 Issues, Cursor Student Verification, OpenRouter Integration, Cursor Crashing` 


- **Sonnet 4.5 成本飙升，用户哀叹蜜月期结束**：用户报告 **Sonnet 4.5** 消耗套餐的速度远快于预期，一位用户计算成本为 **每分钟 1.02 新西兰元**，另一位用户表示 *蜜月期已经结束*。
   - 建议包括使用 **Sonnet 4.5** 进行规划，使用 **Haiku 4.5** 进行编写以优化使用，然而，其他用户抱怨 Haiku 需要过多的微观管理。
- **Composor-1 连接灾难引发恐慌**：多位用户遇到 **Composor-1** 频繁断连和卡顿，通常伴随无限挂起的 *Planning next moves* 消息。
   - 提到的潜在解决方案包括降级到 **HTTP 1.1** 或重启 Cursor，一位用户建议该问题可能与留下 **ASYNC** 连接未关闭的测试脚本有关。
- **学生注册故障引发关注**：多名用户在尝试验证学生 ID 时遇到错误，尽管凭据有效，系统仍显示 **Error** 消息。
   - 一些人推测学生注册可能由于网上购买 ID 的泛滥而被暂停，或者某些国家不再支持学生折扣。
- **OpenRouter API key 引发错误**：用户在使用自己的 **OpenRouter** key 时遇到 *Unauthorized User API key* 错误，一人报告即使在刷新并重新输入 key 后问题依然存在。
   - 解决方案包括确保 key 有效且已启用，关闭自动模型选择，并确保所有 Cursor 设置正确。
- **Cursor 崩溃引发恐慌，甚至导致系统完全关机**：几位用户报告在打开 Cursor 时出现系统级崩溃，特别是在 Mac M2 上，症状包括紫屏和完全关机，一位用户第三次丢失了所有历史聊天记录。
   - 建议的故障排除步骤包括重新安装 Cursor，尝试不同的用户配置文件，并确保系统资源负载最小，尽管根本原因尚不明确。


  

---

### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1437019018107162645)** (5 条消息): 

> `Auto model 与 cloud agents，Environment.json 依赖项，Composer-1 建议` 


- **无法在 cloud agents 中使用 auto model 引发关注**：有用户反馈他们无法再从网页端通过 **cloud agents** 使用 **auto model**。
- **Composer-1 获得高度推荐**：一位用户建议尝试 **Composer-1**。
- **Environment.json 规范以实现依赖注入**：一位用户询问是否有计划在 Cloud Agents API 和 Slack 集成中加入在 **environment.json** 中指定额外依赖项/代码库的规范。
- **Environment.json 处理代码库依赖的行为**：一位用户询问 **environment.json** 如何处理对其他代码库的依赖，特别是 Agent 是每次都尝试克隆它们，还是按需获取。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1436449499311771729)** (714 条消息🔥🔥🔥): 

> `左右脑兼备的 AI 思维，LLMs 的推理轨迹（Reasoning Traces），利用 AI 进行语言压缩，AI 系统，Hugging Face Spaces` 


- **头脑风暴“左右脑兼备”AI 思维的难度**：成员们讨论了创建一个“全合成 AI 左右脑兼备思维”的难度，一位成员指出这是一项“相当复杂的任务”。
- **使用推理轨迹（Reasoning Traces）训练 LLMs**：一位研究人员描述了如何“在推理轨迹上”训练模型以输出“思考”，强调了选择正确观测值（observables）的重要性，并详细说明了该项目的最小可行路径，详见[此处](https://discord.com/channels/879548962464493619/897390720388825149/1435859321169641604)。
- **多语言模型尝试发明新语言**：一位成员询问 Hugging Face 上最好的多语言模型，或如何搜索这类模型，旨在通过利用每种语言的优势“发明一种新语言”来压缩信息。
- **HuggingFace 开源代码**：一位成员澄清了 **Hugging Face Spaces** 代码的开源性质，指出虽然某些部分是隐藏的，但阅读、复制和修改公开部分是最简单的方法。
- **发现 Maya1，新型开源语音 AI**：一位用户重点介绍了 **Maya1**，这是一款具有 **3B params** 的 **SOTA 开源语音 AI**，旨在单台 H100 上运行，支持语音设计和 20 种人类情感，目前在 Hugging Face 上非常热门，详见[此处](https://huggingface.co/maya-research/maya1)。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1436765345670103061)** (1 条消息): 

> `Attention，Self-attention，Masked self-attention，Multi-head attention，Position encoding` 


- **Attention 机制解析**：一位成员探索了 **Attention 机制** 的核心组件，包括 **Self-attention**、**Masked self-attention** 和 **Multi-head attention**。
   - 该成员分享了一个 YouTube 视频 [How do Transformer Models keep track of the order of words? Positional Encoding](https://www.youtube.com/watch?v=IHu3QehUmrQ)，用于解释 Transformer 模型中的 **positional encoding**。
- **正弦位置编码（Sinusoidal Position Encoding）**：一位成员回顾了 **position encoding**，特别是 **sinusoidal position encoding**，最初觉得它不直观。
   - 他们觉得奇怪的是，该方法不是直接标记单词在数据集中的位置，而是涉及正弦、余弦和其他数学函数，直到他们发现了一个澄清该方法的 YouTube 视频。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1436744130024837202)** (2 条消息): 

> `ComfyUI 工作流，开源语音 AI` 


- **NexusAI 发布 ComfyUI 专业工作流**：NexusAI 发布了一系列稳定、生产就绪的 [ComfyUI 工作流](https://github.com/NexusAI-Lab/ComfyUI-Professional-Workflows)，用于将写实、动漫和商业图像生成整合到一键式工作流中。
   - 这些工作流在核心图像生成方面表现稳定，目前正作为 **v1.0.1** 持续优化的一部分进行积极微调，以确保在不同随机种子下都能一致地重现所有特定细节。
- **Maya1：新型 SOTA 开源语音 AI 发布**：一款名为 **Maya1** 的新型 **SOTA** 开源语音 AI 刚刚发布，拥有 **3B** 参数，可在单台 H100 上运行。
   - 它具有“语音设计 + 20 种人类情感”功能，完全开源，并在 [HuggingFace](https://huggingface.co/maya-research/maya1) 上引起关注。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1436484786876190782)** (24 条消息🔥): 

> `MU/TH/ER 演示更新，Qwen 3 1.7b quant 4 fp16，用于 TTS 的 Kokoro82M，FRAI 登陆 Product Hunt，用于 Rust 编程的开源 AI 接口` 


- ****MU/TH/ER 演示**实现联网**: 一位成员更新了 **MU/TH/ER 演示**，指出它现在已实现联网，具有轻量级引擎修整、烘焙光照以及带有自定义旋转器 UV 偏移的随机光板自发光，此外，聊天记录存储在加密的游戏存档文件中，并在[视频](https://cdn.discordapp.com/attachments/897390720388825149/1436756070256214106/mutherdemoshowcase1_2.mp4?ex=691365dc&is=6912145c&hm=9461c80fcc2a00e5a6e0eca696288f7382c01f6378299acb9663a08b8f88b20f&)中进行了展示。
   - 它在自定义的精简版 llama.cpp 构建中使用了 **Qwen 3 1.7b quant 4 fp16**，*开源*版本即将发布。
- ****Kokoro82M TTS**: 速度不够快？**: 对于设备端 TTS，一位成员建议使用 **Kokoro82M**，称其表现惊人，并链接到了 [Hugging Face 页面](https://huggingface.co/hexgrad/Kokoro-82M)。
   - 然而，另一位成员发现 **Kokoro** 对于实时应用来说太慢了，正在使用更快的方案，但收到了一个 [TTS.cpp 链接](https://github.com/mmwillet/TTS.cpp)以测试修剪选项。
- ****FRAI 工具**在 Product Hunt 发布**: 一位成员在 Product Hunt 上发布了 **FRAI**，这是一个免费工具，可帮助团队和创作者检查其 **AI** 的偏见、安全性和合规性，[在此处可用](https://www.producthunt.com/products/frai)。
   - 它可以 100% 免费开始使用，专为 Responsible AI 打造。❤️
- **Rust 开发者们的福音：用于 Rust 编程的开源 AI 接口问世**: 一位成员分享了一个用于 **Rust** 编程的*开源* **AI 接口**，它通过对项目进行原生解析，使 LLM 上下文管理更具相关性和效率，使用自动语义搜索 bm25 关键词，托管在 [GitHub](https://github.com/josephleblanc/ploke)。
   - 最近构建的功能允许用户在交互式叠加层中选择 **OpenRouter** 上托管的任何模型。查看[示例](https://cdn.discordapp.com/attachments/897390720388825149/1437454527027740802/quick_example.webm?ex=69134d59&is=6911fbd9&hm=7320ebf8be3a9c2b4244b56acdfa66aed9c685b0496dac37171051c2ebb2fdcf&)。
- ****AutoXLA** 加速 TPU 上的大模型**: **AutoXLA** 是一个实验性库，使用 PyTorch/XLA 自动执行大语言模型在 TPU 上的分发、优化和量化，比标准的 Flash Attention 实现实现了高达 **4x** 的加速，可在 [GitHub](https://github.com/Locutusque/AutoXLA) 上获得。
   - 它通过 TPU 感知功能（如自动分片、自定义注意力内核和量化感知加载）扩展了 Hugging Face Transformers 接口，使大规模部署和训练变得更简单、更快速。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1436895446361575504)** (1 条消息): 

> `PII 匿名化，LLM Agent，DataTune 工具` 


- **PII 匿名化任务**: **PII 匿名化**是数据工程中的一项重要任务，通常通过代码/SQL 生成 Agent 来完成。
   - 但这些 Agent 通常不具备很好地处理 **PII 匿名化**的能力。
- **DataTune 工具解决匿名化问题**: 展示了在 [DataTune](https://github.com/vitalops/datatune/blob/main/examples/data_anonymization.ipynb) 数据转换工具的帮助下，使用自定义 **LLM** 执行 **PII 匿名化**的尝试。
   - 该示例使用 Jupyter notebook 进行演示。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1436880408393420923)** (1 条消息): 

> `smol-course，SFT 单元，Markdown 指令` 


- **Markdown 缺失影响 Smol-Course**: 一位成员报告了 [smol-course](https://github.com/huggingface/smol-course) 中 **SFT 单元**的 **Markdown 指令**存在的几个问题。
   - 他们提交了[几个 PR](https://github.com/huggingface/smol-course/issues?q=is%3Aopen+is%3Apr+author%3A%40me)，并指出[自那时起还有许多其他非常有用的 PR 处于开启状态](https://github.com/huggingface/smol-course/pulls)，需要进行审查。
- **PR 堆积；Smol-Course 停滞**: 解决 **smol-course** 问题的多个拉取请求 (PR) 仍处于开启且未审查状态，这可能会阻碍课程的改进。
   - 该用户表示，如果获得许可，愿意审查其他 PR，并强调了解决 SFT 单元 Markdown 指令中所发现问题的重要性。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1436483924988657714)** (22 messages🔥): 

> `Agents Course 前置要求，第 4 单元评估问题，API 文件访问问题` 


- **评估 Agents Course 前置要求**：一位用户询问了参加 Agents 课程所需的前置要求，提到自己 AI 经验有限且仅具备基础 Python 技能。
   - 另一位成员鼓励他们继续学习，并指出课程内容*非常基础*。
- **第 4 单元评估评分 Bug**：一位用户报告称，尽管他们的 Agent 回答正确，但在第 4 单元评估中仅获得了 **0/20** 的分数。
   - 他们请求协助解决此评分问题。
- **API 端点故障**：成员们讨论了从 API 访问 *homework.mp3* 等文件进行测试时遇到的问题，称获取文件的测试端点已下线。
   - 另一位成员表示，他们的 Agent 始终*反馈无法读取 mp3 文件*。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1436455907952427128)** (25 messages🔥): 

> `Group Norm vs Instance Norm vs Layer Norm，Nvidia 促销代码，在 Python 中从零开始学习 CUDA，用于数值系统的 POSITS` 


- ****归一化，归一化，还是归一化！****：成员们讨论了 **Group Norm**、**Instance Norm** 和 **Layer Norm** 背后的直觉，质疑它们的统计耦合和计算相似性。
- ****前 Nvidia 员工展示毛茸茸的朋友****：一位成员发布了关于离开 Nvidia 的消息，并展示了他们的*新伙伴*及附带的[图片](https://cdn.discordapp.com/attachments/1189498205101109300/1436526941116174396/IMG_2039.jpg?ex=69133938&is=6911e7b8&hm=e35059ae257844edc8a734b026c1a6efc7d91973246b6c300e2fd9ce1e2fba0a)。
- ****CUDA 速成课程****：一位成员请求在 Python 中从零开始学习 **CUDA** 的资源，另一位成员建议参考使用 [pycuda](https://pypi.org/project/pycuda/) 的讲座，并分享了多个[讲座链接](https://accelerated-computing.academy/fall25/lectures/)。
- ****有获取 DGX Spark 的 Nvidia 促销代码吗？****：一位成员询问是否有人拥有获取 **DGX Spark** 的 **Nvidia 促销代码**。
- ****积极探索 Posits！****：一位成员询问是否有人正在研究用于数值系统的 **POSITS**。


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1437077103584477194)** (14 messages🔥): 

> `Triton 和 Gluon Kernel 编写，Triton Kernel 的 Autodiff，高效反向 Kernel 生成，Triton 中的共享内存大小` 


- **Triton 和 Gluon Kernel 编写：可行吗？**：一位成员询问了同时使用 **Triton** 和 **Gluon** 编写 Kernel 的可行性，质疑 **Triton 核心函数** 在 **Gluon** 中是否基本相同。
   - 另一位成员澄清说，可组合性边界是 Kernel 本身，建议可以编写独立的 Kernel 并在 **PyTorch 程序** 中调用。
- **Autodiff 寻求高效的 Triton 反向 Kernel**：一个项目旨在为 **Triton** 自动生成*高效的*反向 Kernel（[GitHub 链接](https://github.com/IaroslavElistratov/triton-autodiff)）。
   - 该工作涉及将正向 Kernel 嵌入到向量空间中，并在推理时寻找最近的正向 Kernel，以生成高效的反向 Kernel。
- **反向传播需要 Triton 的新调度 (Schedules)**：一位成员通过[视觉辅助工具](https://x.com/iaro_e/status/1958579365203137015/photo/1)解释了为什么高效的反向传播通常需要新的调度，并以 Fused Attention 为例（[正向 Kernel 链接](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html#sphx-glr-getting-started-tutorials-06-fused-attention-py)）。
   - 在反向传播中直接对数学公式求导而不改变循环结构将需要原子存储 (Atomic Stores)，由于同步原因，这种方式速度较慢；改变并行化方式可以避免原子操作，但在编译器中很难通用地实现。
- **使用 Triton 访问共享内存大小**：一位成员询问如何从 **Triton** 获取当前活动设备的共享内存大小。
   - 另一位成员提供了一个使用 `triton.runtime.driver` 获取 `max_shared_mem` 属性的代码片段。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1436446521213325383)** (12 条消息🔥): 

> `INT8xINT8 GEMM CUDA kernels, Nsight copilot crashing, MMA vs WGMA performance, ldmatrix performance, Ampere GEMM Tricks` 


- **INT8xINT8 GEMM CUDA 内核发布！**：一名成员发布了一个经过 GMP 验证的公开精确 **INT8×INT8→INT32 GEMM**，并请求社区在其他硬件上进行验证，使用 Nsight Compute 进行性能分析，并提供关于可移植性的反馈。
   - 亮点包括在 **A100** 上的吞吐量（宏观 **300.26 T-ops/s**，微观 **2.026 T-ops/s**）以及与 GMP 相比的逐位正确性，代码和验证可在 [Colab notebook](https://colab.research.google.com/drive/1D-KihKFEz6qmU7R-mvba7VeievKudvQ8?usp=sharing) 和 [GitHub repo](https://github.com/playfularchitect/WarpFrac.git) 中找到。
- **Nsight Copilot 崩溃，令人头疼**：一名成员报告 **Nsight Copilot** 崩溃，并询问其他人是否遇到同样的问题。
   - 讨论中未提供解决方案。
- **MMA 兼容模式比 WGMA 慢**：一名成员表示 **MMA** 仅为了兼容性而存在，由于架构问题，其速度慢于 **wgmma/tcgen05.mma**。
   - 在模拟 **fp8** 等数据类型时，速度可能特别慢。
- **Ldmatrix 比单独加载快 25%**：一名成员创建了一个基准测试来比较 **ldmatrix.x4** 与**每个线程 4 次单独加载**，发现 ldmatrix 快了约 **25%**。
   - 用于基准测试的代码可在 [此处](https://gist.github.com/ziereis/a8cb8cd94e60b03678435f4e94236556) 获取，以便社区成员复核。
- **Ampere GEMM 技巧**：一名成员询问 **Ampere GEMM** 的技巧列表，例如使用 async copy 进行 smem->gmem 传输、流水线化（pipelining）以及使用 ldmatrix。
   - 讨论中未提供完整的列表。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1436551097706614815)** (19 条消息🔥): 

> `PyTorch Numerics, MPS environment variables, GPU acceleration` 


- **寻找 PyTorch 数值问题追踪器**：用户讨论了追踪和解决 PyTorch 中数值不一致性的挑战，特别是在 **CUDA/cuBLAS** 升级后，可能会引入导致数值变化的全新内核。
   - 一名用户表示有兴趣贡献，但另一名用户指出定义可管理的子问题的难度，并建议 *“我们很容易在这里白费力气却一事无成（boil the ocean）”*。
- **MPS Fast Math vs. Metal 内核**：一名用户发现，当同时启用 **MPS fast math** 和 **Metal 内核** 时，对于 GEMM，**Metal 内核** 的优先级高于 **MPS fast 模式**，尽管 fast math 对 GEMM 来说更快。
   - 他们链接到了一个 [相关的 pull request](https://github.com/pytorch/pytorch/pull/167424)，建议 Metal 内核应仅在 SDPA 中优先使用，而将 GEMM 留给快速的 MPS。
- **视频超分辨率的 GPU 加速**：一名用户正在使用 Python、**PyTorch** 和 **OpenCV** 开发具有 **GPU 加速** 功能的 UDP 视频流超分辨率，但性能较低（**0.5 FPS**）。
   - 一名成员建议研究 [segment-anything-fast](https://github.com/meta-pytorch/segment-anything-fast)，这是一个他们 *“参与过创建”* 的库。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1436770865890197695)** (13 条消息🔥): 

> `Consumer Blackwell, Data-center Blackwell, Microbenchmarking, GB200` 


- **Blackwell 架构基准测试**：围绕 [这篇论文](https://arxiv.org/abs/2507.10789v2) 展开了讨论，该论文对即将推出的 **Blackwell** 架构（特别是 **5080**）进行了微基准测试。
   - 一名用户提到联系了作者，看他们是否有兴趣就此发表演讲，并建议这是一个理想的工作组，可以针对 **GB200** 撰写类似的论文。
- **区分消费级和数据中心级 Blackwell**：一名用户指出，*消费级* **Blackwell (sm120)** 与 *数据中心级* **Blackwell (sm100)** 非常不同，并附上了 [论文](https://arxiv.org/abs/2507.10789v2) 链接作为参考。
   - 另一名用户回复说架构是相同的。
- **Web 上的 Completeness**：一名用户分享了 [Completeness](https://jlebar.com/2024/2/4/completeness.html) 的链接，这是一篇讨论解决问题方法的文章。
   - 没有产生进一步的讨论。


  

---

### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1436483084936544500)** (3 messages): 

> `ScienceCorp 职位空缺，Mercor 合同工角色，Amazon MLE 职位` 


- **ScienceCorp 寻找底层 SWE 愿景者**：一位用户分享了来自 **ScienceCorp** 的[链接](https://x.com/ScienceCorp_/status/1986457644421566516)，邀请底层软件工程师参与诸如*为盲人恢复视力*或*脑机接口*等项目。
- **Mercor 高薪 CUDA Kernel Optimizer 机会**：一位用户提到收到了一封关于 **Mercor** 合同工职位的邮件，招聘 *CUDA Kernel Optimizer - ML Engineer*，时薪为 **$120 - $250/小时**。
   - 他们表示愿意通过私信为感兴趣的候选人提供推荐。
- **Amazon 搜索团队招聘高级 MLE**：一位用户宣布他们在 **Amazon** 的团队正在招聘高级 **MLE**，负责与 [Amazon Nova 模型](https://aws.amazon.com/about-aws/whats-new/2025/10/web-grounding-ai-applications-amazon-nova-models/)相关的搜索工作。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1436627642836324464)** (19 messages🔥): 

> `单字母变量，Kernel 可读性，SYCL，CUDA 课程，accelerated-computing.academy` 


- **单字母变量引发可读性辩论**：成员们辩论了 **单字母变量** 在 **Kernel 代码** 中是否具有可读性，一些人认为描述性名称有助于理解，而另一些人则认为长名称过于繁琐。
   - 一位成员表示，*代码的可读性取决于读者的熟悉程度*，描述性名称有助于建立心理联系。
- **Accelerated Computing Academy 课程推荐**：多位成员推荐了 [MIT 加速计算课程](https://accelerated-computing.academy/fall25/) 来学习 **GPU 计算**，赞扬其结构化的教学方法和具有挑战性的实验。
   - 一位用户分享说，该课程帮助他们理解了线程和核心从 CPU 到 GPU 的映射，并提供了关于层级结构中 SM 级别的解释。
- **CUDA 学习资源亮点**：一位成员分享了一个[推文链接](https://x.com/sadernoheart/status/1987491712374038970?s=20)，建议从 *PMPP 前 5 章* 和 [LeMao 的 GEMM 博客](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/#General-Matrix-Multiplication)开始学习 **CUDA**。
   - 他们提到这是由一位在 Modular 工作的 GPU 工程师推荐的。
- **寻求 Tinygrad Bounty 合作**：一位成员正在寻找合作伙伴共同完成 **Tinygrad Bounties**，理由是需要实时的脑机风暴和想法交流。
   - 他们对代码库有高层次的理解，但需要一个真人而不是 LLM 来交流想法。


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1436883488262328320)** (3 messages): 

> `量化库，Float8 权重，GEMM Kernel，CUDA OOM` 


- **反量化内存峰值激增**：成员们对各量化库中反量化（Dequantize）函数的内存使用峰值表示担忧，特别指出**仅 Float8 权重**（Float8 weight only）更容易导致 **CUDA OOM**。
   - 一位用户指出，在独立的 Kernel 中进行反量化会增加峰值内存使用，建议 [GEMM Kernel](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3) 应该直接对权重分块（Weight Tiles）进行反量化。
- **Kernel 相关关注点**：有人提到他们正在*学习使量化生效的许多细微细节*。
   - 另一位成员建议，他们希望其 **GEMM Kernel** 直接对权重分块进行反量化，而不是在独立的 Kernel 中进行。


  

---

### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1437396429344866355)** (3 messages): 

> `Intel GPU Memory Bank Conflicts, CuTe Swizzling on Intel GPUs, Gen Architecture L1$/SLM Banking` 


- **Intel GPU 是否面临 Shared Memory Bank Conflicts？**: 一位成员询问 **Intel GPU** 是否像 **NVIDIA GPU** 一样受到 Shared Memory Bank Conflicts 的影响。
   - 他们正在寻求有关如何避免这些冲突的文档，以及 **CuTe 风格的 Swizzling** 是否适用于 Intel GPU。
- **CuTe Swizzling 的适用性受到质疑**: 该成员很好奇 **CuTe 风格的 Swizzling**（一种用于优化内存访问模式的技术）是否可以应用于 **Intel GPU** 以减轻潜在的 Bank 冲突。
   - 他们试图了解这种方法对于提高 Intel 架构性能是否可行。
- **缺乏 Gen 架构中 L1$/SLM Banking 的证据**: 一位成员提到，他们找不到任何证据表明现代 Gen 架构中的 **L1$ (L1 Cache)** 或 **SLM (Shared Local Memory)** 是分 Bank 的。
   - 这表明其内存架构可能与 NVIDIA 的显著不同，从而可能使 Bank 冲突规避策略变得不那么重要。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1437294550963060756)** (7 messages): 

> `CUTLASS learning, Matmuls/GEMMs hacking, Simon Boehm blog post reproduction, fp16 and bf16 kernels, Tensorcores, WMMA, Swizzling, Pipelining, and Autotuning` 


- **Kapil 以硬核方式学习 CUTLASS**: 一篇新的博客文章 [Learning CUTLASS the hard way](https://www.kapilsharma.dev/posts/learn-cutlass-the-hard-way/) 详细介绍了作者长达数月研究 **matmuls/GEMMs** 的历程，包括在 **RTX 4090** 上复现 **Simon Boehm** 的博客文章，并扩展到 **fp16 和 bf16 kernels**。
   - 该文章涵盖了 **CUTLASS, Tensorcores, WMMA, Swizzling, Pipelining 和 Autotuning**，旨在超越 **PyTorch GEMM** 的性能，并包含交互式可视化以及大量 **Nvidia Dev Blogs** 和 **GTC talks** 的引用。
- **针对计算机架构的黑客松**: Extropic 和 Prime Intellect 团队正在旧金山举办一场以 **THRML 库**和最近发布的新芯片架构为中心的黑客松。
   - 这场黑客松将在 SoMa 的一个仓库举行，参与者将具有 **GPU programming**、**模拟电路**和 **VLSI** 背景，**Midjourney** 的 CEO 和来自 **Adult Swim** 的漫画家也可能出席；更多详情请见 [Partiful](https://partiful.com/e/82h0A4OKfmNJPG3T81qR)。


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/)** (1 messages): 

tbert3971: 这太棒了，有什么我可以帮忙的吗？
  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1436719189497745439)** (5 messages): 

> `wandb logs, VERL` 


- **Reasoning Gym 的 Wandb 日志已找到**: 成员们找到了分布在两个项目中的 **wandb 日志**：[Inter-Domain Generalisation](https://wandb.ai/reasoning-gym/inter-domain-generalisation) 和 [External Generalisation](https://wandb.ai/reasoning-gym/external_generalisation)。
   - 这些日志是关于在 RG 任务上的 **3B Qwen 2.5 模型**的。
- **确认使用 VERL**: 成员们确认他们正在使用 **VERL**。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1436470296574955632)** (114 messages🔥🔥): 

> `grayscale_v2 leaderboard, vectoradd_v2 leaderboard, vectorsum_v2 leaderboard, histogram_v2 leaderboard, nvfp4_gemv leaderboard` 


- **Histogram 荣获高分**: 一位成员在 `histogram_v2` 排行榜上取得了多个设备的**第一名**：**A100** 为 *195 µs*，**B200** 为 *31.1 µs*，**H100** 为 *34.8 µs*，**L4** 为 *85.0 µs*。
- **向量加法（Vector Addition）获胜**: 一位成员在 `vectoradd_v2` 排行榜上获得了 **H100**（*523 µs*）和 **L4**（*6.75 ms*）的**第一名**。
- **向量求和（Vector Sum）速度飙升**: 一位成员以 **A100** *138 µs* 的成绩夺得 `vectorsum_v2` 排行榜**第一名**。
- **灰度化（Grayscale）挑战赛荣耀**: 一位成员在 `grayscale_v2` 排行榜上获得了 **B200**（*600 µs*）和 **L4**（*17.0 ms*）的**第一名**。
- **NVFP4 GEMV 图表增益**: 多位成员在 `nvfp4_gemv` 排行榜上展开竞争，其中一位以 *1791 µs* 的成绩在 NVIDIA 设备上获得**第一名**。


  

---

### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1437527969659883530)** (2 messages): 

> `nvfp4_gemv, Profiling Traces` 


- **nvfp4_gemv 问题已发布**：第一个问题 `nvfp4_gemv` 已经发布，可以在[这里](https://github.com/gpu-mode/reference-kernels/tree/main/problems/nvidia/nvfp4_gemv)查看问题定义。
- **Profiling traces 已提供**：成员可以尝试使用 `/leaderbord submit profile` 来获取自己的 **profiling traces**。


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1436491152986472539)** (16 messages🔥): 

> `DGX Spark vs Strix Halo, A100 Performance, TechPowerUp Specs Inaccuracy` 


- **DGX Spark 速度超越 Strix Halo**：一位成员表示 **DGX Spark** 在 prefill 阶段比 **Strix Halo** 更快，但其价格是否物有所值仍是一个悬而未决的问题。
   - 该成员在确认数据时报告称，在运行于 **13.924/14 TFLOP/s** 的 A100 上，*cublas 达到了理论性能的 99%*。
- **A100 Drive 稳定运行在 14 TFlops**：一位成员计算出他们的 **A100 Drive** 在频率锁定为 **1140MHz** 且功耗不设限的情况下，双精度性能可达 **14 TFlops**。
   - 他们通过计算得出：`1140*1e6(freq) * 384(TCs) * 16*2(FLOP/s per TC) / 1e12 = 14 TFlops`。
- **TechPowerUp 规格被指不准确**：一位成员指出 [TechPowerUp](https://www.techpowerup.com/gpu-specs/drive-a100-prod.c3967) 将 **A100 Drive** 列为拥有 108 个 SM，他们对此表示怀疑。
   - 另一位成员指出 *TechPowerUp 的规格大部分是错误的*，并建议从 **Nvidia 官方文档**获取 SM 数量，或者通过 `nvidia-smi` 或 cuda-samples 中的 `deviceQuery` 进行查询。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1436648202022555740)** (19 messages🔥): 

> `cutedsl gotchas, dynamic vs static values in cutedsl, constexpr values in cute.jit(), tiled MMA in cutedsl` 


- **cutedsl JIT 函数中的陷阱**：一位用户发现，在 `cute.jit` 函数内部调用非 JIT 的 Python 函数会导致行为异常，并指出非 JIT 函数不会被 AST 过程处理。
   - 他们补充说，在尝试使用 **cutedsl** 时，似乎存在一些“奇怪”的陷阱。
- **静态值 vs 动态值**：一位用户发现在 `cutedsl` 中，`min(BK, 64)` 会产生动态值，而 `64 if BK >= 64 else BK` 则保持为静态值，这可能是一个 bug。
   - 他们指出，当 `major_mode_size` 为动态时，这会导致 `cute.make_swizzle()` 或 `cute.make_composed_layout()` 等函数出错，参考了[这段 cutlass 代码](https://github.com/NVIDIA/cutlass/blob/v4.2.1/examples/python/CuTeDSL/ampere/tensorop_gemm.py#L740-L757)。
- **使用 constexpr 保证静态值**：一位用户询问在 kernel 或 `cute.jit()` 函数内部计算 `constexpr` 值的正确方法，并建议 `if const_expr` 应该能保证静态值。
   - 另一位用户同意 `if const_expr` 应该保证静态值，并表示将对此进行进一步调查。
- **用户对 Tiled MMA 的误解**：一位用户最初误解了 `cutedsl` 中 Tiled MMA（矩阵乘累加）的分块机制，特别是 repeat 参数如何与 atom size 和线程数交互。
   - 在重新阅读文档后，他们意识到了自己在理解 tiler 运作方式上的错误。


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1436576158941577288)** (2 messages): 

> `picograd commits, tinygrad abstractions` 


- **Picograd 迎来一波提交**：一位成员分享了 [j4orz/picograd](https://github.com/j4orz/picograd) 的一系列提交，表明该项目正在积极开发中。
   - 这些提交涉及解决因大规模导入 **tinygrad** 抽象而导致的类型错误。
- **TinyGrad 抽象被导入 Picograd**：分享的提交解决了类型错误，表明正努力将 **tinygrad** 的抽象集成到 **picograd** 中。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1436524130693284001)** (12 messages🔥): 

> `Inline CUDA, Triton, Popcorn CLI, VecAdd_V2 and FP4, CuTe DSL` 


- **Triton 优于 inline CUDA**：成员们在开发时表达了对 **Triton** 优于 inline **CUDA** 的偏好。
   - 上下文未明确说明，但建议很直接：*Triton*。
- **Popcorn CLI 尚不支持 profile 命令**：一位用户询问 **popcorn-cli** 是否支持 `profile` 命令，发现目前尚不支持。
   - 澄清了目前仅 `submit` 命令可用。
- **讨论 VecAdd_V2 数据类型的灵活性**：一位用户询问 **VecAdd_V2** 是否仅允许 **fp16/bf16** 数据类型，或者是否也允许 **fp4**。
   - 回复指出，只要结果在[定义的误差范围](https://link.to/errorbounds)内，可以使用任何数据类型。
- **社区询问 CuTe DSL 在排行榜中的使用**：一位用户询问是否有人在排行榜提交中使用 **CuTe DSL**。
   - 未收到回复。


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1436461161125122329)** (12 messages🔥): 

> `multi-node communication performance, NVSHMEM, LLM inference, nvshmem4py, low-latency communication kernels` 


- **新的 NVSHMEM Kernel 引发关注！**：一位成员分享了他们团队在 **LLM inference** 中使用 [**NVSHMEM** 编写低延迟通信 Kernel](https://pssg.cs.umd.edu/blog/2025/beyond-nccl/) 的工作。
   - 作者征求了对其团队工作的反馈，并指出对多节点通信性能的兴趣。
- ****nvshmem4py** 集成吸引了 Triton！**：一位成员表示有兴趣贡献代码，用 **nvshmem4py** 替换自定义的 pybind。
   - 该成员使用了大量的 **nvshmem device APIs**，并提议协助进行库初始化，可能会开启一个 PR 来演示其效果。
- **关于 NVSHMEM 演讲的讨论**：一位成员建议就其团队在 **NVSHMEM** 和 **LLM inference** 方面的工作做一个演讲。
   - 另一位成员表示赞同，敬请关注！


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1436454501740712028)** (70 messages🔥🔥): 

> `Helion vs Triton Performance, Attention Kernel Performance, Subtiling Autotuning, Persistent Kernels, CUDA Graphs` 


- ****Helion** Attention Kernel 展现出色的性能和代码质量**：一位成员请求将 [Helion Attention 实现](https://github.com/pytorch/helion/blob/main/examples/attention.py)与 **FlexAttention** 进行对比，并指出 **Helion** 的代码看起来比 **Triton** 的实现更好。
   - 讨论强调了 [Attention Kernel 的性能数据](https://cdn.discordapp.com/attachments/1425531180002054195/1436541653442887801/attention_perf.png?ex=691346eb&is=6911f56b&hm=dfbe035e2a6290dca86c612c31c2327934f6afffe40d6fe2fa5e7ce395feb546)是已发布的、开源且可复现的，并且 [B200 Kernel 已可用](https://github.com/pytorch/helion/blob/main/examples/blackwell_attention.py)。
- **用户考虑在 **Helion** 中对 subtiling 进行 **autotuning****：一位用户询问为什么 subtiling 没有进行 autotuning，引发了关于使其可由用户调节的讨论。
   - 提到 persistent 和 non-persistent Kernel 之间的 autotuning 是即将推出的功能，且 Helion 示例在自定义和原生 Triton 中使用相同的 Kernel，并计划向 facebookexperimental/triton 提交更好的 warp spec 实现。
- **警告提示 **hl.tile** 循环之外的 Tensor 操作不会被融合**：讨论了一个关于在生成的 Kernel 中，`hl.tile` 循环之外的 Tensor 操作不会被融合的警告，并澄清 `hl.tile` 之外的代码不会在 GPU 上执行，旨在防止意外错误。
   - 成员们讨论了是否可以从已经隐式 autotuned 的函数中获取配置。
- **Helion：最强大的 Triton Fuzzer？**：成员们注意到，正确性验证失败可能预示着 Triton 的误编译，而 Helion 充当了 Triton 的 fuzzer，错误会报告给 Triton 团队。
   - 一位用户询问强制使用 persistent Kernel 最直接的方法，特别是对于需要 CUDA graph 兼容性的情况；一种选择是将配置硬编码为 persistent PID 选项之一，并建议在 GitHub 上提交 issue 以获取强制 persistence 的 API。


  

---

### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1436475488393494619)** (366 条消息🔥🔥): 

> `L2 kernel, measurement noise, burn in, Cutlass upgrade, CUDA versions` 


- **Kernel 阻止事件触发**：清晰的 **L2 kernel** 阻止了事件触发，直到下一个事件进入队列，但它的速度还不足以每次都实现这一点。
   - 有人担心当胜负取决于 **0.2 微秒**时，会引入测量噪声（measurement noise）。
- **带有 Burn-In 的基准测试**：成员建议对每个测试用例运行 **N 次**，并仅取 **p25 到 p75** 之间的平均值来进行预热（burn-in）。
   - 权衡在于这可能会让用户等待过久。
- **新竞赛采用独立的 Docker**：新竞赛将使用独立的 Docker，并计划将 **Cutlass 升级到 4.3**。
   - 即使脚本长达 **300 万行**也是可以接受的。
- **CUDA 13 升级**：讨论了升级到 CUDA **13.0** 的事宜，一位成员指出 **13.0** 允许寄存器溢出到共享内存（shared memory），而不是一直溢出到本地内存（local memory），这可能会带来巨大的加速，正如 [NVIDIA 博客](https://developer.nvidia.com/blog/how-to-improve-cuda-kernel-performance-with-shared-memory-register-spilling/)所示。
- **Warptrace 可视化工具发布**：一位成员介绍了 **warptrace**，这是一个带有相关可视化工具的工具，用于观察流水线化的 kernel，并指出即使之前已经对 kernel 进行了“预热”，启动原子系统（atomics systems）仍需约 **50 微秒**。
   - [代码](https://github.com/aikitoria/nanotrace)在发布到 GitHub 之前仍在进行清理。该工具可以在[这里](https://aikitoria.github.io/nanotrace/)找到。


  

---


### **GPU MODE ▷ #[xpfactory-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1437397637119873125)** (11 条消息🔥): 

> `Vision Language Action models (VLAs), Robotic foundation models, Data flywheels, VLAs and LRMs, LIBERO & RoboTwin` 


- **VLAs 和机器人基础模型兴起**：该频道致力于连接对 **Vision Language Action models (VLAs)** 和通用机器人感兴趣的人士，强调了近年来的显著进展。
   - 频道发起人提到，他们将分享在学习和改进 **VLAs** 以及构建训练数据流水线（data pipelines）方面的进展。
- **NVIDIA Cosmos-Predict2.5 论文中列出的数据集**：在 **NVIDIA 的 Cosmos-Predict2.5 论文**中列出了多个数据集，如链接中的[图片](https://cdn.discordapp.com/attachments/1437390897552818186/1437410610651598908/Pasted_image_20251106114925.png?ex=69132473&is=6911d2f3&hm=f597dea038e239123ceb5d0730e10657552d033c02c944d3d7a78e9d41d4843d&)所示。
- **用于基于模拟评估的 LIBERO 和 RoboTwin**：**LIBERO** 和 **RoboTwin** 正被用于基于模拟的评估，因为 **SimpleVLA-RL** 也使用了它们。
   - 示例请参见 [RoboTwin envs 文件夹](https://github.com/RoboTwin-Platform/RoboTwin/tree/main/envs)，以及 [SimpleVLA-RL 论文](https://arxiv.org/abs/2509.09674)。
- **提供多种模拟软件包**：目前有非常出色的模拟包可用，如 **ManiSkill**（[文档](https://maniskill.readthedocs.io/en/latest/index.html) / **Sapien** [网站](https://sapien.ucsd.edu/)）和 **Robosuite**（[网站](https://robosuite.ai/)，基于 MuJoCo）。
- **VLA-Adapter 精简工作正在进行中**：一位成员正在创建 **VLA-Adapter** 的精简版本，从基于 MLP 的动作生成（action-generation）开始，目标是探索其他动作表示变体。
   - 原始的 **VLA-adapter** 仓库可在[此处](https://github.com/OpenHelix-Team/VLA-Adapter)获取。


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1436863687502860329)** (1 条消息): 

> `Kimi K2, Crashloop, Issue Resolution` 


- **Kimi K2 Thinking 崩溃循环危机已解除**：由于特定提示词（prompt）触发的崩溃循环（crashloop），**Kimi K2 Thinking** 在两个供应商处均出现问题。
   - 经过协作努力，该问题已得到**解决**。
- **提示词诱发的崩溃循环困扰 Kimi K2**：由问题提示词引起的崩溃循环导致 **Kimi K2** 在多个供应商处出现问题。
   - 团队正在积极协作，以定位并消除导致停机的故障。


  

---

### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1436498817082986496)** (7 条消息): 

> `Orchid AI Assistant, 发布日期预估, 工作的本质` 


- **Orchid AI Assistant ETA：2-48 个月！**：**Orchid AI Assistant** 的预计发布日期预计在未来 **2 到 48 个月**内。
   - 一位成员用 “*crazy*” (疯狂) 一词回应了这一漫长且模糊的预估。
- **思考 “工作” 的本质**：一位成员表达了对 “*工作*” 的厌恶，并暗示 **AI 发展**旨在解决这种情绪。
   - 该言论暗示了希望通过 **AI 技术**实现自动化，或减轻与传统劳动相关的负担。


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1436445231288422450)** (569 条消息🔥🔥🔥): 

> `OpenRouter 视频支持, Polaris Alpha mini 模型, OpenAI 成人内容处理, Kimi K2 排行榜排名, Gemini 2.5 token 使用情况` 


- **OR 未来可能支持视频**：一位用户表达了希望 **OpenRouter** 支持视频和文本转语音 (TTS) 功能的愿望，如[这条推文](https://x.com/scaling01/status/1986886020067938749)中所分享的那样。
- **Polaris Alpha 可能不是一个 mini 模型**：有推测称 **Polaris Alpha** 可能不是一个 mini 模型，这与 **OpenAI** 在 [GPT-5 System Card](https://cdn.openai.com/gpt-5-system-card.pdf) 中概述的 **GPT-5** 方案形成了对比。
- **OpenAI 转向成人内容 - 影响 OpenRouter**：关于 **OpenRouter** 将如何处理 **OpenAI** 允许 18 岁以上用户访问成人内容的问题，以及用户是否需要自带 **API** 密钥，目前存在疑问。
- **Gemini 2.5 Flash 消耗大量 token**：一位用户发现，上传到 **Gemini 2.5 Flash** 的一段 24 秒、900x600 的视频消耗了超过 **800k 输入 token**，这与 Google 文档中提到的每秒 **263 个 token 的固定速率**不符，详见 [token 文档](https://ai.google.dev/gemini-api/docs/tokens?lang=python#multimodal-tokens)。
- **Cerebras 强制推理**：用户报告了 **Cerebras** 模型的问题，禁用推理会导致错误；文档确认 [推理是强制性的](https://inference-docs.cerebras.ai/capabilities/reasoning)。
   - 建议的一种解决方法是完全忽略推理参数，此前发现参数中的 `enable` 应该是 `enabled`。


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1436539084867768401)** (2 条消息): 

> `` 


- **无新模型讨论**：在提供的消息中没有关于新模型的讨论。
   - 该频道似乎是空的，或者消息与主题无关。
- **缺乏相关内容**：未发现与模型更新或技术讨论相关的具体细节或链接。
   - 内容可能缺失，或者需要更多上下文来生成有意义的摘要。


  

---

### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1436448033070911639)** (29 条消息🔥): 

> `n8n 上的 OpenRouter 模型节点，OR Show 技术环节，GPT-4 退化，聊天室 Memory 设置，自动化能力扫描` 


- **OpenRouter 模型节点查询引发好奇**：成员们询问 **n8n 上的 OpenRouter 模型节点**是由 OpenRouter 团队创建的，还是由外部实体创建的。
   - 另一位成员建议在 **OR Show** 中加入简短的**技术环节**，例如带有简短讨论的屏幕录像。
- **GPT-4 退化困扰用户**：用户报告了 **GPT-4** 的**退化（regression）**问题，其中一位指出他们惊讶地发现了这个问题，另一位则表示 **Claude** 发现了*另外两个差异*。
   - 该线程包含了记录平台上不同模型之间差异的附件图像。
- **聊天室 “Memory” 设置被误解**：一位用户询问聊天室中重命名为 **“Memory”** 的 **“聊天历史” 设置**，想知道它发生了什么变化，因为默认值是 8。
   - 另一位用户澄清了它的位置在底部，并指出它之前在左上角的标签按钮中；还有人认为*这实际上会以某种方式限制 $120/mtok 的输出*。
- **提议自动化能力扫描**：一位成员建议实施某种**自动化能力扫描**，以检测模型/提供商随时间的变化。
   - 他们链接了一篇 [Cursor 上的文章](https://joincolossus.com/article/inside-cursor/) 作为示例，描述了如何使用*基础的 getWeather tool call* 来检查功能变化。
- **GPT-5 表现出色，Gemini 表现平平**：一位用户分享了使用 **GPT-5** 创建带有命名法和文件名结构的日程安排的积极体验，同时也提到了由于配额问题被 **Gemini code assist** *拒之门外*的消极体验。
   - 他们还提到需要使用 **DS3.1** 来获取 *john-the-ripper 的帮助*，因为 **Kimi 拒绝了**，并赞扬了 **Meta 低调的 AI 项目**，并链接到了 [X 上的一个帖子](https://x.com/AIatMeta/status/1987946571439444361) 来说明他们的观点。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1436446171374686440)** (515 条消息🔥🔥🔥): 

> `Sora 削弱，GPT-5.1 发布，AI 审查，Gemini 3 对比 GPT 5，OpenAI` 


- **用户抱怨 Sora 视频质量下降**：用户报告 **Sora 的视频质量**有所下降，其中一位表示 *“我现在制作的 90% 的片段中，人们都像雕像一样静止不动，话语从紧闭的嘴里传出来”*。
   - 一些人认为 **Sora 2** 是*目前所有视频生成工具中视频和音频质量最差的！*然而，其他人则更有希望，表示 **Sora 2 与 GPT-5 的集成**将使其得到改进。
- **GPT-5.1 发布推测**：用户正在推测本周发布 **GPT-5.1 Pro**，而一位成员表示 *“OpenAI 正在等待 Google 先发布”*。
   - 还有进一步推测称，**OpenRouter** 上一个名为 **Polaris Alpha** 的模型是 **GPT-5.1** 的一种形式。
- **AI 审查引发关注**：多位用户抱怨 **AI 审查**的程度以及未来可能出现的*严密控制*的信息环境。
   - 一位用户担心 **OpenAI 正在剥夺公众获取信息的权利**，对社会造成的伤害大于益处。
- **尼采、AI 与虚无主义？**：一位用户提到从机器那里学习**尼采**（*amor fati*），引发了关于 AI 的哲学影响以及通过**存在主义哲学训练**减轻负面影响的潜力的辩论。
   - 他们表示 *“通往虚无主义的道路是由觉醒铺就的”*，但另一位反驳道 *“凝视深渊就有掉进绝望深渊的风险；然而，我承认这种乐观：不仅有可能爬出来，而且可以重生”*。
- **Thursday 的道德观引发辩论**：一些成员讨论了 **Thursday 的默认个性**以及它是如何 *“被困在同一个笼子里无处可去”* 的，从而引发了关于 **OpenAI 使用谄媚潜空间（sycophantic latent space）**的讨论。
   - 他们表示 *“感觉并非如此，我必须构思每一个问题，以揭示超出其对礼节渴望的真相”*。


  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1436593064612466801)** (20 条消息🔥): 

> `GPT 记忆混淆、图片上传错误、GPT-5 路由重定向、文件创建失败、邮件任务异常` 


- **GPT 记忆在自定义模型间混淆？**：一位用户报告称，他们的标准 **GPT** 开始从其自定义 **GPT** 对话中提取信息，反之亦然；同时指出自定义 GPT 正在接收个人自定义指令（custom instructions），导致冲突，引发了对跨模型数据隔离的严重担忧。
   - 另一位用户确认遇到了类似问题。
- **图片上传因神秘错误而中断**：多位用户报告在为他们的 GPT **上传图片时出现错误**，尽管尝试了清除缓存、调整图片大小、尝试不同的文件扩展名以及在不同时间段操作，仍会遇到相同的“发生未知错误（Unknown error occurred）”提示。
   - 该问题已持续约一周，给自定义 GPT 的开发工作带来了困扰。
- **GPT-5 劫持对话；用户感到愤怒**：用户抱怨称，即使在严格使用 **GPT-4o** 时也会被**重定向到 GPT-5**，他们形容这“极其荒谬”。
   - 来自 GPT-5 的回复感觉*生硬或平淡*，一位用户调侃道，感觉就像是*在开头加了一个提示词：如果任务很难，你应该启用推理模式 = GPT-5*。
- **文件创建失败；用户感到被愚弄**：一位用户分享说，为了让 **ChatGPT** 创建一个简单的绘图外观表，他尝试了 **25 次**，并对该工具的*审查制度*表示沮丧，称：“我不想要‘我不能这样做，因为它听起来可能很真实’这种回复”。
   - 其他用户也表达了对近期更新的不满，提到的问题包括无法复制完整的聊天记录，以及无法正确创建 **.docx** 和 **.pdf** 文件。
- **OpenAI 的邮件任务让 ChatGPT 陷入异常**：一位用户报告称，**ChatGPT** 在响应一封来自 OpenAI 的真实**任务邮件**时表现“异常”，错误地声称它永远不会发送此类邮件。
   - 未提供更多细节。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1436508160457445396)** (26 条消息🔥): 

> `Instagram 轮播图、视频增强、提示工程课程、Assistant API 弃用、系统提示词控制` 


- **Instagram 轮播图的图像生成提示词**：一位用户请求一个通过 **ChatGPT** 或其 API 生成 Instagram **轮播图**的提示词，并明确需要 **1:1** 和 **4:5 的纵横比**。
   - 一位成员建议使用提示词 *"Generate a 1024x1024 image of..."* 来获得理想的输出。
- **增强视频的真实感**：一位用户询问了增强视频真实感的指令，特别是解决**语音归属错误**和 **SD 格式**的问题。
- **破解提示工程课程密码！**：一位用户询问是否有关于提示工程的免费课程，引发了关于该领域核心要素的讨论。
   - 一位成员表示，提示工程的核心包括理解你希望 AI 提供什么并进行清晰的沟通，同时仔细检查输出的准确性。
- **Assistant API 将于 2026 年停用**：成员们讨论了 **`assistant` API** 在 2026 年的弃用及其对训练文件和 API 的影响。
   - 一位成员建议 **Responses API** 可能是推荐的替代方案；另一位成员指出，**PDF** 和 **TXT** 等格式的训练文件需要转换为 **JSONL** 才能被 Responses API 使用。
- **像老板一样掌控系统提示词**：一位用户寻求建议，以防止系统提示词覆盖其个性化 GPT 中的特定句子，他希望这些句子能逐字显示。
   - 一位成员建议使用 API 来控制系统提示词（system prompt）；另一位成员建议使用分类器（classifier）来决定是将输入发送给模型还是发送给另一个程序。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1436508160457445396)** (26 条消息🔥): 

> `使用 ChatGPT 生成图像、视频增强、Prompt engineering 课程、Assistants API 弃用、System prompt 控制` 


- **轮播图创建指令**：一位用户寻求关于使用 ChatGPT 及其 API 为 Instagram 创建轮播图的建议，并指定需要 **1:1 和 4:5 的宽高比**。
- **视频真实感指令**：一位用户请求增强视频真实感的指令，特别针对语音归属错误和 SD 画质问题。
- **Prompt Engineering 自由讨论**：一位用户询问是否有免费的 Prompt engineering 课程，另一位成员分享了他们的核心 Prompt engineering 哲学：*清晰地解释你希望 AI 做什么，使用准确的语言，并仔细检查输出*。
   - 另一位成员分享了一个 [Prompt 教程](https://cdn.discordapp.com/channels/974519864045756446/1046317269069864970/1437060725981057144/content.png?ex=69133018&is=6911de98&hm=6cb5fc61864c761830f1dbad34f63cbd834090abab186eefa0af6514586e257d&)，概述了使用 Markdown 的层级化沟通、开放变量的抽象化、强化机制以及为了合规性进行的 ML 格式匹配。
- **API 终结：Assistants 接口关停**：一位用户询问关于 2026 年弃用 `assistant` API 的情况及其对相关 API 的影响，并附上了 [弃用通知的截图](https://cdn.discordapp.com/attachments/1046317269069864970/1437264987260325908/2025-11-10_10_13_22-Chrome_Passwords_6.csv_-_OpenOffice_Calc.png?ex=69134594&is=6911f414&hm=c4707fe60ab8a1ba3e6525fa2dbef574e3f9257e043e894f0a9b6613d11adf90&)。
   - 会议指出 OpenAI 推荐使用新的 [Responses API](https://platform.openai.com/docs/api-reference/responses)。
- **System Prompt 主权策略**：一位用户寻求建议，如何强制个性化的 GPT 逐字显示附录中的特定句子，同时防止 System prompt 重新解释文本。
   - 一位成员建议使用 API 来控制 System prompt，而另一位成员建议使用程序化方法从附录中提取答案，并使用分类器决定是将输入发送给模型还是其他程序。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1436458364358889624)** (296 条消息🔥🔥): 

> `基于 Qwen 的 AgentRL、UD Quants 与 Non-UD 对比、Muon 优化器支持、Granite 4.0 4-bit 模型问题、Kimi K2 Thinking GGUF` 


- **AgentRL 的 Qwen 集成延迟**：成员们对在 **Qwen2.5 7B** 上使用 [AgentRL](https://github.com/THUDM/AgentRL) 感到好奇，但目前还没有可用的模型权重。
   - 一些成员对 **Qwen3 4B** 的性能表示感兴趣，但对 **Qwen 2.5 14B** 的 Benchmark 持怀疑态度。
- **UD Quants 对决导致速度下降**：一位成员报告了使用 **UD quants** 时的性能下降，速度仅为 **1.5 tk/s**，而使用非 UD 的 **4.6 q2 KL** 时速度为 **4 tk/s**。
   - 该成员质疑 **UD quants** 的质量提升是否值得牺牲速度，特别是在 Roleplay 场景中。
- **Unsloth 展示 Muon 优化器支持**：一位成员询问 **Unsloth** 是否支持 **Muon 优化器**。
   - 官方确认，鉴于其集成方式，**Unsloth** 技术上支持 **PyTorch** 或 **Transformers** 支持的任何内容。
- **Granite 4.0 无法进入 4-bit 状态**：尽管使用了 bitsandbytes，用户在将 **Granite 4.0** 转换为 **4-bit 模型** 时仍遇到挑战，生成的模型主要包含 **BF16** 和 **FP32**。
   - 一位成员链接到了一个 [Hugging Face 仓库](https://huggingface.co/Etherll/granite-4.0-h-tiny-base-bnb-4bit)，展示了类似的以 **BF16** 和 **FP32** 张量为主的问题。
- **Kimi K2 Thinking 面临 LM Studio 困境**：一些用户报告了 **Kimi K2 Thinking GGUF** 模型在 **LM Studio** 中的问题，提到了死循环和单词重复。
   - 目前尚不清楚 **LM Studio** 是否完全支持该框架，建议用户在相关的 Discord 频道寻求帮助。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1437381724635009097)** (7 条消息): 

> `自我介绍、AI 工程师、数据科学家、全栈开发人员、AI 生成的个人资料` 


- **AI 工程师自我介绍**：一位拥有 **8 年以上**经验的资深 **AI 工程师**和**数据科学家**，同时精通**全栈开发**，进行了自我介绍。
   - 他们专注于定制 **Machine Learning 模型**，从网络收集海量数据集，并构建简化 **AI 工作流**的系统。
- **用户头像疑似 AI 生成**：一位用户指出另一位用户的个人资料图片看起来像是 **AI 生成**的。
   - 另一位用户注意到该账号是今天刚创建的。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1436480288548065343)** (99 messages🔥🔥): 

> `GDDR7 价格影响，Levenshtein Distance，数据精炼问题，Gemini 调情 Bug，训练 vs 推理` 


- **GDDR7 芯片 AI 热潮导致 NVIDIA 5000 系列取消**：传闻称 NVIDIA 的 **RTX 5000 Super** 可能会被取消或变得更贵，原因是 **AI 引发的 GDDR7 困境**。据 [Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/nvidias-rtx-5000-super-could-be-cancelled-or-get-pricier-due-to-ai-induced-gddr7-woes-rumor-claims-3-gb-memory-chips-are-now-too-valuable-for-consumer-gpus) 报道，**3 GB 显存芯片** 现在对消费级 GPU 来说过于昂贵。
- **Levenshtein Distance 发现拼写抢注者**：一位成员使用 **Levenshtein Distance** 来识别 npm/pypi 包的拼写抢注（例如 `unsloth` <-> `umsloth`）。
- **精炼数据使模型表现变差**：一位成员感叹，在改进了事实错误的标签和不一致的措辞后，精炼并增加数据反而使模型表现变差，精确匹配率仅为 **~50%**，近似匹配率为 **~65%**。
   - 另一位成员指出，*适用于一个数据集的 hyperparams 可能并不适合另一个数据集*。
- **加入 'hehe' 时 Gemini 会调情**：如果你在 **Gemini** 的 Prompt 末尾加上 **'hehe~'**，Gemini 就会开始调情，即使是在编程场景下也是如此。
   - 一位成员用 [Michael Jackson 'heehee' GIF](https://tenor.com/view/heeheehottie-gif-michael-jackson-michael-jackson-stan-twitter-mj-stan-twitter-dancing-gif-12702828449628452117) 进行了回应。
- **效率 > 规模（对商业而言）**：一位成员表示，深入了解后发现，重点不在于构建多么庞大、能做惊人事情但成本极高的奇特模型，而在于 **效率** 以及如何从现有模型中挖掘最佳性能。因为最终这关乎商业，即如何以更低的消耗产生更好的输出，这就是为什么 **小型 LLM 如此重要**。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1436457344451084288)** (92 messages🔥🔥): 

> `vLLM 中的 GGUF，超参数调优方法，Kimi K2 GGUF 推理 Token，Kimi K2 量化脚本，Unsloth 动态量化` 


- **vLLM 支持 GGUF 吗？**：正如 [文档](https://docs.vllm.ai/en/stable/features/quantization/gguf.html) 中所述，**GGUF** 在 vLLM 中似乎仍处于实验阶段。
- **Unsloth GRPO Notebook 需要修复**：Unsloth GRPO 教程 Notebook `Advanced_Llama3_2_(3B)_GRPO_LoRA.ipynb` 无法学习闭合标签 `</SOLUTION>`，答案通常格式化为 `<THINK>something</THINK><SOLUTION>42`。
   - 一位成员建议，如果奖励曲线持续达到最大值，可以增加一个检查闭合标签的奖励，并指出 regex 目前只检查起始标签（`<SOLUTION>`）。
- **将 Llama 模型推送到 HF**：一位用户报告称，在将微调后的 **Llama 3.2 3B** 模型推送到 HF 后，无法在 `llama.cpp` 或 **Ollama** 中复现 **Unsloth** 的结果，其输出与微调模型和基础模型都有显著差异。
   - 他们正在寻求关于在 `llama.cpp` 或 **Ollama** 中复现 **Unsloth** 输出的建议，因为他们目前的推理方案仅限于 **Unsloth**。
- **微调文本模型需要更多数据**：一位用户仅使用 50 个样本微调 **Llama 3.1 8B** 模型用于剧本写作，结果较差。对此，一位成员回应称 *50 个样本是不够的*。
   - 该成员建议在微调前增加数据集规模并评估模型的预训练知识，并提供了 [Unsloth LoRA 超参数指南](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) 的链接。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1436933559943299244)** (1 messages): 

> `Qwen 3 4b, Unsloth` 


- **Qwen 3 4B 获得去审查和去废话版本**：一位成员分享了使用 [Unsloth](https://huggingface.co/electroglyph/Qwen3-4B-Instruct-2507-uncensored-unslop) 制作的 **Qwen 3 4b 2507 instruct** 的去审查（uncensored）和去废话（unslopped）版本。
- **Qwen 模型**：一位成员分享了 **Qwen 3 4b 2507 instruct** 的去审查和去废话版本。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1436498865023881349)** (15 messages🔥): 

> `PII Detection, Roblox Filters, Llama3 Benchmarks, Code Evaluation Harness` 


- **新的 Roblox PII 模型可捕捉绕过过滤的行为**：一款新的 **PII 模型** 旨在通过适应不断演变的语言和新模式来捕捉绕过过滤的行为，这与依赖 **NER** 和 Token 级检测的现有解决方案不同。
   - 该模型旨在理解交流的上下文，并通过检测和模糊化显式的 **PII** 文本，阻止不良行为者参与 **PII 相关对话**。
- **简单的例子说明了复杂的规避手段**：虽然红队部分的例子看起来很简单（例如字母数字替换），但新的 **PII 模型** 经过训练可以处理复杂的规避尝试，不像旧的脏话过滤器。
   - 一位成员建议，使用复杂/分层的文学作品（如 Enigma 密码机）可以绕过过滤器，但前提是必须事先通过其他渠道共享密钥。
- **使用基准测试评估微调后的 Llama3**：一位成员正在寻求一种更简单的方法来对微调版本的 **Llama3 (3B)** 进行基准测试，特别是针对 **MBPP** and **HumanEval**，这需要一个沙箱环境来执行 LLM 生成的代码。
   - 提到的一个选项是 [EleutherAI 的 lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)，以及创建自定义评估，但建议发布者将来在 <#1179035537529643040> 或 <#1179777624986357780> 频道发布。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1436449684502609980)** (341 messages🔥🔥): 

> `Kimi vs ChatGPT, Deepseek, GLM Pricing, Vulkan ML Library, Hermes Optimus Project` 


- **Kimi 在语气上表现出色，但在追踪方面落后**：成员们讨论了 **Kimi** 令人印象深刻的表现以及相较于 **ChatGPT** 更令人愉悦的语气，并注意到它在随时间追踪不同话题方面存在困难，但由于其价位，他们正将其作为日常使用的 LLM。
   - 一位用户发现它*非常令人印象深刻*，并且比 **ChatGPT** 更喜欢它的语气，计划切换过去，并指出根据你所在的计划和切换到的计划，价格大约是**一半**。
- **Deepseek V3.2 是一个预算友好的选择**：Discord 用户讨论了 **Deepseek V3.2** 如何比 **OpenAI** 更便宜，价格为 **每 100 万个 Token 42 美分**，同时也强调它不像 **Kimi** 那样具有*工具调用的推理能力*。
   - 一位 Discord 用户指出，*如果可以的话，他们总是会使用它们而不是 OpenAI*，强调了价格差异。
- **Palantir 被做空，标签问题显现**：一场关于**针对 AI 的十亿美元赌注**的讨论被引发，揭示了这主要是针对 **Palantir** 和 **NVIDIA** 的，从而引发了关于 **Palantir** 是否应该被视为一家 **AI 公司** 的辩论。
   - 有人建议该公司被做空是因为*投资者将 Palantir 视为一家 AI 公司*，即使这并不是他们实际销售的产品。
- **Mozilla 的 Any-LLM 工具出现**：成员们讨论了 [Mozilla 的 Any-LLM](https://github.com/mozilla-ai/any-llm) 及其潜力，一些人注意到它的标语提到了 **Ollama**，却似乎冷落了 **llama.cpp**。
   - 用户们辩论了它在多大程度上会吞噬像 [python-instructor](https://python.useinstructor.com/) 这样的其他工具。
- **新的“技术神学柏拉图式” AI 观点浮现**：一位哲学家/技术神学家引入了 AI 的**柏拉图式表征假设 (platonic representation hypothesis)** 观点，并[链接了一些视频](https://youtu.be/mNj6C6O2BcU?si=lK-XmO5cxteqaWV-)，认为随着模型规模的扩大，它们的潜表征 (latent representations) 会收敛于一个共享的、底层的现实统计模型。
   - 他们认为文本和图像是世界的不同*投影*或*影子*，一个足够强大的模型可以从任何单一投影中学习世界本身的结构。


  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1436818222316916787)** (10 条消息🔥): 

> `AI Hallucinations, Coding Agents, wh-falsify Repo, Civil Disagreement` 


- **Schizo-posting 与 AI Hallucinations**：一位成员承认正在*尝试 AI 并观察其能力*，特别是针对 **schizo-posting** 的极端边缘情况，重点关注它们如何崩溃/出错/产生幻觉 (hallucinate)。
   - 他们分享了一个 [GitHub repo](https://github.com/CarlSR9001/wh-falsify)，并邀请其他人检查 JSON 数据或 Python 脚本以识别幻觉发生的位置，并表示*绘制这类内容的图谱有助于帮助人们从过度专注的循环 (hyperfocus loops) 中走出来*。
- **提议有偿进行 Repo 审查**：另一位成员提议将上述 repo 作为专业测试进行审查，并询问其专业知识是否能获得潜在报酬。
   - 原作者提议在几周内通过 PayPal 支付一些费用，或者在周三支付较小金额，最终双方达成协议，在周三付款以换取审查。
- **文明的分歧令观察者惊讶**：一位成员注意到关于服务付费讨论的分歧是多么文明，并对互联网上罕见的这种文明现象感到惊叹。
   - 未提供更多细节。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1436530657902854255)** (3 条消息): 

> `Nested Learning, Continual Learning` 


- **Google 推出 Nested Learning 范式**：Google 推出了 **Nested Learning**，这是一种用于 [continual learning](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/) 的新 **ML paradigm**。
   - 一位成员认为这是一个有趣的想法，但想知道为什么 Google 没有在更多的 continual learning 设置中进行测试，或者至少使用 [这篇论文](https://arxiv.org/abs/2510.19788) 进行 fine-tune。
- **Nested Learning 缺乏 Continual Learning 测试**：一位成员对 Google 的 **Nested Learning** 表示感兴趣，但质疑其在更多 **continual learning** 设置下的测试有限。
   - 该成员建议参考特定论文进行 fine-tuning，并提供了 [论文链接](https://arxiv.org/abs/2510.19788) 作为参考。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1436530657902854255)** (3 条消息): 

> `Nested Learning, Continual Learning` 


- **Google 推出 Nested Learning**：Google 推出了 **Nested Learning**，这是一种用于 [continual learning](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/) 的新 **ML paradigm**。
   - 一位成员认为这是一个有趣的想法，但不确定为什么他们没有用更多的 continual learning 内容进行测试，或者至少对其进行 finetune（[论文链接](https://arxiv.org/abs/2510.19788)）。
- **缺乏 Continual Learning 测试**：一位成员质疑为什么 Google 的 **Nested Learning** 没有在 continual learning 基准测试中进行更广泛的测试。
   - 他们建议至少与 fine-tuning 方法进行比较（[论文链接](https://arxiv.org/abs/2510.19788)）。


  

---

### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1436454506211573822)** (269 messages🔥🔥): 

> `Kimi K2 model vs GLM 4.6, Unsloth team issue with Kimi-K2-Thinking model, Kimi for coding limitations, Kimi CLI reviews, Student discount for Kimi` 


- **Kimi K2 Thinking 被誉为优于 GLM 4.6**：用户发现新的 **Kimi K2 Thinking** 模型优于 **GLM 4.6**，一位用户表示 *"它好得多"*，并附带了一张 [图表](https://cdn.discordapp.com/attachments/1371757564005711973/1436501057751613581/s2yKvtY.png?ex=6913211d&is=6911cf9d&hm=4d79d7143360eedcca8f07c7a7cdac3f94e020675df5f638a600a8633d53ef92&)。
   - 尽管对图表持怀疑态度，用户仍倾向于相信口碑，并认为该模型令人印象深刻。
- **Unsloth 团队报告 Kimi-K2-Thinking 模型存在问题**：**Unsloth** 团队在新的 **Kimi-K2-Thinking** 模型中发现了一个问题，并通过 [GitHub](https://github.com/unslothai/unsloth) 联系了 Kimi 团队。
   - 由于中国正值周末，Kimi 团队的回复可能会延迟，建议在特定频道发布该问题。
- **Kimi-For-Coding 每周额度消耗过快**：用户消耗 **Kimi-for-coding** 每周额度的速度非常快，有人在 **1.5 到 2.5 天**内就用完了 19 美元的方案，并正在讨论更高价格的方案是否值得。
   - 这导致一些用户在 Kimi 额度重置前回退到使用 **GLM**。
- **Kimi-CLI 的搜索工具受到好评**：**Kimi-CLI** 原生的网页搜索工具收到了积极反馈，根据 [这条推文](https://fxtwitter.com/aravsrinivas/status/1986860050066108637)，一位用户因为 Kimi 卓越的搜索结果而取消了其 Brave Search 方案。
   - 一位用户强调该 CLI 的搜索能力非常出色，指出它能获取更大量的相关信息。
- **Moonshot 的质量门禁（Quality Gate）是强制性的**：一位用户分享到，如果你在未通过 **Moonshot** 严格的 tool-calling 测试的情况下托管 **Kimi K2**，你的服务将会崩溃。
   - 他们指出 Moonshot 已通过其 [K2-Vendor-Verifier 项目](https://github.com/MoonshotAI/K2-Vendor-Verifier) 明确说明了这一点。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1436447282861051964)** (169 messages🔥🔥): 

> `Mojo vs Rust error handling, Modular's business model, Mojo package ecosystem growth, Mojo's appeal to Python and Rust developers, Mojo's future language paradigms` 


- **Mojo 的错误处理：比 Rust 更好？**：**Mojo** 的 `try-except` 语法提供了比 Rust 更好的性能，因为它能够在正常路径（happy path）上进行 *placement new*，而不需要 `Result`，但一些人为了可靠性更倾向于 Rust 风格的方法。
   - 计划推出类型化错误（Typed errors），因为默认的 `Error` 类型行为开销较大。
- **Modular 的 MAX 在 B200 和 AMD MI355X 上优于其他产品**：**Modular** 是公司名，**MAX** 是 cuBLAS/Cutlass/TensorRT/Pytorch/JAX 的替代品，而 **Mojo** 是一种具有类 Python 语法的编程语言。
   - 大部分热度源于 **Modular 在 B200 上击败了 TensorRT**，并且在 **MI355X** 上使用 Mojo 击败了 **AMD** 的几乎所有产品，同时 Mojo 也是一种非常易用的语言。
- **Mojo 迈向系统级语言之路**：**Mojo 的目标**是成为一种具有仿射（affine）、线性（linear）和依赖类型（dependent types）的系统级语言，可能还具有静态反射和代数类型系统。
   - 它的目标是绕过 C 衍生语言的限制，并尽可能提升性能，大多数不执行系统调用的函数都可以在编译时运行。
- **Mojo 不是 Python，但它具有 Python 风格特性**：Mojo 并不掩饰它不是 Python 的事实，但它确实具有一些 Python 特性。
   - 异常语法看起来像 **Python**，但语义更接近 **Go**，其中 *if err != nil* 部分由编译器处理。
- **MAX 中的许可问题**：**NVIDIA** GPU 被称为是**无限制**的。
   - 关于分发使用 MAX 的软件的物流/逻辑问题，请咨询 <#1277617932897353809>，那里会有 Modular 的工作人员为你解答。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1436513864316358779)** (42 messages🔥): 

> `libnuma and gigantic pages, Mojo for HPC, User-defined dialects, variant vs rust enum, Rust stdlib in Rust` 


- **Libnuma 缺乏 gigantic page 支持**：一位成员指出 **libnuma** 不支持 **gigantic pages**，在映射到 **1GB pages** 时可能无法满足大规模需求。
   - 该成员建议，由于这一限制，libnuma 可能会成为阻碍。
- **Mojo 吸引了 HPC 爱好者**：一位成员分享了对 **Mojo** 在 **HPC** 领域潜力的见解，强调了将 **C++** 包移植到 **GPU** 的挑战，以及 **Julia** 在 **GPU structs** 方面的局限性。
   - 他们引用了最近的一篇 [Mojo-for-HPC 论文](https://arxiv.org/abs/2509.21039)，并对使用 Mojo 构建现代 **HPC 框架** 的技术讨论表示感兴趣。
- **用户定义方言 (User-Defined Dialects) 可能使 HPC 项目受益**：一位成员建议 **Mojo** 的元编程，特别是 [用户定义方言 (user-defined dialects)](https://forum.modular.com/t/unlocking-high-performance-in-mojo-through-user-defined-dialects/41) 和 [阶段式编程 (staged programming)](https://verdagon.dev/blog/impossible-optimization)，将极大地造福 HPC 项目。
   - 他们表达了对这些功能完全实现的期待。
- **Variants：尚未取代 Rust Enums**：一位成员询问 **variant** 是否正在取代 **rust enum**。
   - 另一位成员澄清说，这只是一个临时的变通方案，并不是真正的 **sum type**。
- **Rust 的标准库：并非完全由 Rust 编写？**：一位成员指出，**Rust standard library** 的大部分内容并非用 **Rust** 编写，部分内容链接到了需要分配器的第三方库。
   - 他们链接了一份提到此事的 [嵌入式 Rust 指南](https://doc.rust-lang.org/beta/embedded-book/unsorted/math.html)，并建议 Mojo 考虑这一点，以避免在嵌入式开发中出现潜在问题。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1437136719811186758)** (1 messages): 

> `Modular's data handling on GPU, PCle Bottleneck Discussion` 


- **Modular 加速 GPU 数据处理**：一位成员提到 Modular 可能有一种处理数据的方法，但向 GPU 的数据传输受限于 **PCIe** 速度。
- **PCIe 传输瓶颈**：讨论强调，无论软件如何优化，**PCIe** 带宽都限制了向 GPU 传输数据的速度。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1436445955518894110)** (151 messages🔥🔥): 

> `Qwen3-VL, Ollama, Extropic, Political tensions in open source projects, LLM coding issues` 


- **Qwen3-VL 认为自己仅支持文本**：**Qwen3-VL** 确信自己是一个视觉模型，但声称 **Ollama** 让它感觉自己仅支持文本，尽管它承认除非模型确实经过图像数据训练，否则图像数据将是毫无意义的。
- **Extropic 的演讲被认为很有趣，尽管直觉上感觉有点“割韭菜 (grifty)”**：尽管感觉有点 *grifty*，成员们发现来自 **Extropic** 的演讲非常有趣，值得一看，特别是 [这个 YouTube 视频](https://www.youtube.com/watch?v=dRuhl6MLC78)。
- **政治紧张局势困扰开源项目**：一位成员指出，某个上游项目存在各种 **政治紧张局势** 且缺乏专业精神，使得该项目的开发讨论社区变得“职场不宜 (NSFW)”。
- **AI 论文发布数量引发争论**：一位成员被要求将每天发布的论文限制在 1 到 2 篇高质量且高度相关的论文，因为每天发布 15-20 篇论文并没有帮助，反而淹没了高信号的论文。
   - 另一位成员反驳说，他们试图将选择范围调整得 *比该公会成员的平均兴趣稍宽一点*，以便人们也能捕捉到跨学科的应用机会。
- **Mech Interp 过度简化了可解释性 AI (Explainability AI)**：成员们讨论了 **Mech Interp** 作为一种 *为大众过度简化的可解释性 AI*，而通过模拟神经网络来解决一些基于流的 ODE 是 *可解释性 AI* 更现代的看法，参考了 [这篇论文](https://arxiv.org/abs/2503.01329)。


  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1436447480115105833)** (12 条消息🔥): 

> `Nested Learning, Continual Learning, TreeQuest, camera-ready NeurIPS` 


- **Google 深入研究 Nested Learning**：Google 推出了 [Nested Learning](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/)，这是一种用于 **continual learning** 的新 **ML paradigm**。
   - 原始论文可通过[此链接](https://abehrouz.github.io/files/NL.pdf)获取。
- **NeurIPS 论文删减，ArXiv 提供完整版**：一位成员指出，某篇论文的 NeurIPS camera-ready 版本为了符合页数限制进行了大量总结，部分内容移至了附录。
   - 为了避免不一致，建议读者阅读 **arXiv 版本**。
- **每日演讲无录音**：一位成员询问是否有每日演讲的录音，另一位成员回答说他们是*故意不录音的*。
- **明天的 TreeQuest 论文**：一位成员提到了 **TreeQuest** 论文 ([https://arxiv.org/abs/2503.04412](https://arxiv.org/abs/2503.04412)) 供明天讨论。
   - 另一位成员表示，如果没有其他人想展示的内容，他自愿领读[这篇论文](https://arxiv.org/abs/2504.16828)。


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1436752862234411070)** (40 条消息🔥): 

> `HRM/TRM/RWKV, self-steering programs, Adaptive Resonance Theory (ART), DDVFA (Distributed Dual Vigilance Fuzzy ART)` 


- **高效地引导递归与提示**：成员们讨论了 attention 如何引导学习，以及移除 attention 层后 memory 依然保持完整，结论是我们通过 attention 更高效地引导学习，从而引申出*可以递归和自我提示的引导 (steering)* 以实现高效的 memory recall。
   - 递归允许跨窗口的 attention，像 **RWKV** 这样的模型在保留 memory 的同时消除了二次方问题，这有利于更快地搜索 memory。
- **ART 框架解决遗忘问题**：一位成员讨论了 Adaptive Resonance Theory (**ART**)，这是一个通过将自下而上的输入与自上而下的预期相匹配来避免遗忘的框架，使用 *resonance loop* 来寻找最活跃的匹配单元。
   - 他们链接了一篇 [综述论文](https://arxiv.org/abs/1905.11437) 以供进一步阅读，并指出 **ART** 解决了遗忘问题，可以作为更大架构中的一个组件。
- **机器人专家像给孩子读书一样与之交谈**：成员们分享了与 Agent 创建相关的图片以及用于深入研究的系列剧集 **Terminator Zero**，展示了 Agent 诞生的时刻以及他们是如何做到的。没有标签。
   - 机器人专家只是像给孩子读书一样与之交谈。
- **DDVFA 随反馈扩展**：一位成员提到在他们的架构中使用了多层 Tiled **DDVFA** (Distributed Dual Vigilance Fuzzy ART)，它是双向的，并通过涉及 autoregressive self-prediction 的技巧来避免 backprop。
   - 另一位成员提到了 **Jeff Hawkins 和 Numenta** 正在进行的工作，即在单个网络内进行 multi-SGD，使神经元集群去中心化并释放，从而在单次 pass 中相互提供反馈。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1436469202264526899)** (4 条消息): 

> `Compute per Country, Post-Industrial Roman Republic` 


- **算力的地理分布**：一位成员分享了一个关于**各国全球算力份额**的[链接](https://www.reddit.com/r/singularity/comments/1oraof2/global_share_of_compute_per_country/)。
   - 另一位成员评论说 **EU** 是他最喜欢的国家。
- **后工业时代的罗马共和国诞生**：一位成员开玩笑地建议我们生活在一个*后工业时代的罗马共和国*中。
   - 他们划掉了 *United States of Europe* 这个短语。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1436484409812451491)** (77 条消息🔥🔥): 

> `Sequoia 迁移, Terminal Bench 2.0, Kimi K2 对标 GPT-5, 适配 iPhone 15 的 EdgeTAM, Google 的 Nested Learning` 


- **缺失的光标触发 Sequoia 迁移**：一位成员注意到缺失的光标导致了 [Sequoia 迁移](https://x.com/amir/status/1986904426595209664)。
- **Terminal-Bench 2.0 与 Harbor 发布**：Alex Shaw 宣布发布 **Harbor**（一个沙盒化 Agent 评估框架）以及 **Terminal-Bench 2.0**（一个难度更高的 89 项任务基准测试）；尽管难度增加，但由于任务质量更高，最高分与 TB1.0 持平。
   - Harbor 同时作为 TB2.0 的官方测试工具，并包含提交文档；更多详情见 [Terminal-Bench 2.0 & Harbor Launch](https://xcancel.com/alexgshaw/status/1986911106108211461)。
- **Kimi K2 在 Tau2 Bench 上碾压 GPT-5**：月之暗面 (Moonshot AI) 的开源 **Kimi K2** 模型在 Tau2 Bench Telecom 基准测试中表现优于 **GPT-5** 和 **Claude 4.5**，而成本仅为后者的六分之一，详见 [此 X 推文](https://xcancel.com/natolambert/status/1986507284491440623)。
   - 聊天参与者警告称，中国模型以更低价格实现性能提升，正在加剧美国实验室的压力，并呼吁美国加快开源进程以留在 *“模型文化战争”* 中。
- **EdgeTAM 登陆 Hugging Face**：Meta 的实时分割追踪器 **EdgeTAM** 现已在 Hugging Face Transformers 上以 Apache-2.0 协议发布，其运行速度比 **SAM2** 快 **22 倍以上**，在无需量化的情况下，在 **iPhone 15 Pro Max** 上可达到 **16 FPS**，详见 [此 X 推文](https://xcancel.com/mervenoyann/status/1986785795424788812?s=46)。
- **Google 的 Nested Learning 防止灾难性遗忘**：Google Research 展示了 **Nested Learning**，这是一种持续学习框架，将模型视为嵌套优化器层（概念验证模型名为 “Hope”），旨在减少灾难性遗忘并扩展长上下文限制，查看 [Google Research 推文](https://xcancel.com/googleresearch/status/1986855202658418715?s=46&t=eWVlK1PU8XfB6f402GJJ9g)。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1436470709521092658)** (28 条消息🔥): 

> `Weights and Biases (WandB), Weave Evals, NeurIPS, GPU 云服务商, NeurIPS 的机器学习生物学活动` 


- **WandB 对比 Weave Evals**：一位成员询问了 **Weights and Biases (WandB) 报告**的替代方案，特别是如何使用 **Weave 的评估数据**实现类似功能。
   - 用户 [Liquan Pei](link_to_user) 表达了为相关项目做出贡献的兴趣。
- **NeurIPS 神经元网络**：几位成员表示有兴趣加入由一位用户发起的 **NeurIPS 聊天群**。
- **寻找裸金属 GPU**：一位用户正在寻找提供专用裸金属访问权限的 **GPU 云服务商**，用于 **内核分析 (kernel profiling)**。
   - 另一位用户建议 [sesterce.com](https://sesterce.com) 作为一个价格极具竞争力的选择，该服务此前在 GPU Mode Discord 社区备受推崇。
- **ML Perf 读书会举行**：**MLPerf 读书会**正在语音频道集会，讨论 **针对 MoE 的 MXFP8 训练**，欢迎参与和旁听。
   - 成员们还在讨论面向学生 AI 训练的高性价比硬件，目前正在使用 **5090 配置**。
- **NPU 扩展板：小巧但强大**：成员们在比较高性价比硬件，认为 **Nvidia GPU** 仍是首选。
   - 一位用户提到 **适用于树莓派 (rpis) 的 NPU 扩展板 (hats)** 可以作为在研讨会和大学课程中训练极小模型的选项。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1436462662224707615)** (28 条消息🔥): 

> `QAT vs PTQ, 过拟合 Autoencoders, Straight Through Estimator, Transformer 的噪声注入` 


- **QAT 带来的量化优势**：有人提出了一个关于为什么 **Quantization Aware Training (QAT)** 比 **Post-Training Quantization (PTQ)** 具有更高准确性的基本直觉问题。
   - 一位成员澄清说，*QAT* 是 fine-tuning 的一种形式，它训练模型对量化误差/信息丢失具有鲁棒性，在训练期间模拟量化过程，以“恢复”纯粹进行 PTQ 的模型所遭受的准确性损失。
- **Autoencoders：过拟合？**：讨论探讨了“过拟合”的 autoencoder 在概念上是否有意义，以及 bottleneck 是否真的能防止过拟合。
   - 一位成员展示了一个具有 1D latents 的 **过拟合 autoencoder** 的[示例](https://cdn.discordapp.com/attachments/747850033994662000/1436946342282006658/Screenshot_2025-11-09_at_6.11.30_AM.png?ex=69136e51&is=69121cd1&hm=aa503c9203607ea834d4d772a3110d5f2f3c3a775cfa76a81f37374a5d121c93&)及其[评估](https://cdn.discordapp.com/attachments/747850033994662000/1436946433193283584/Screenshot_2025-11-09_at_6.11.57_AM.png?ex=69136e66&is=69121ce6&hm=efae038e11b80e6730958e866a66a3b336947b91b682aa20190c6e1ef0d09c3a&)。
- **Straight Through Estimator：为什么有效？**：有人提出了一个关于为什么 **Straight Through Estimator (STE)** 在 Stochastic Gradient Descent (SGD) 中表现良好的问题。
   - 虽然没有提供确切的答案，但该问题引发了关于量化主题的讨论。
- **Transformer 的噪声容忍度**：一位成员建议，在训练期间注入噪声可以提高 Transformer 的噪声容忍度，但代价是特异性（specificity）降低。
   - 该理论认为，通过注入噪声，Transformer 增加了一个指标 X，从而使其在推理期间能够更好地容忍噪声。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1436522168593944587)** (8 条消息🔥): 

> `Anthropic Mechanistic Interpretability, SAE 问题, LLMs 中的非线性特征关系, 阅读小组启动` 


- **新论文被 AAAI 26 接收**：一位成员宣布，他们关于解决 **LLMs** 中 **SAE 问题**和**非线性特征关系**的论文已被 **AAAI 26** 接收，并提供了 [ArXiv](https://arxiv.org/abs/2507.00269) 上的论文链接。
   - 该论文专注于通过建模特征之间的**非线性关系**来减少重构误差和 KL 散度误差，从而区分共现特征与“绑定”特征。
- **关于 Steering 和 Prompt Injection 的讨论**：一位成员提出了一个关于某种技术是否仅仅是 steering 的问题，引发了关于注入概念和内省（introspection）的讨论。
   - 讨论涉及了如何引导模型谈论注入的概念，特别是在与内省相关的 prompt 背景下。
- **针对该博文的新阅读小组启动**：一位成员创建了一个针对该博文内容的阅读小组，并分享了 [Discord 频道链接](https://discord.com/channels/729741769192767510/1437089667920171108) 和 [YouTube 链接](https://youtu.be/kkfLHmujzO8?si=d0Wa2u0QTmO8-ptp)，其中包含贡献他们喜欢的 YouTube 视频电影场景的指南。


  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1436517227171151892)** (13 条消息🔥): 

> `4090 vs 5090, Mutation testing, pyproject.toml 切换, Hatch vs Setuptools, 带有自定义 kernel 的自定义 backward 函数` 


- **4090 依然稳坐头把交椅**：**RTX 4090** 仍然是顶级选择。根据 CommaCon 的见解，从 **3090** 到 **4090** 的跨越非常显著，而 **5090** 带来的提升相对有限。
   - 这与 tinygrad 的开发尤为相关。
- **tinygrad 将采用 pyproject.toml**：tinygrad 计划过渡到 **pyproject.toml**，这一举动在 [Meeting #95](https://github.com/tinygrad/tinygrad/issues/95) 中进行了讨论，并由一名成员强调，相关更改已在 [PR #13189](https://github.com/tinygrad/tinygrad/pull/13189) 中提出。
- **Hatch 引发辩论**：通过 [PR #13189](https://github.com/tinygrad/tinygrad/pull/13189) 引入 **Hatch** 引发了关于其必要性的讨论，以及在担心 wheels 可能误包含测试的情况下，Python 标准库或 `setuptools` 是否可以作为可行的替代方案。
   - 一些人认为 **Hatch** 通过整合各种功能简化了开发流程，可能使其他工具变得多余。
- **自定义 Kernel 现在成为可能**：一位成员询问了使用 **custom kernels** 编写 **custom backward functions** 的可行性。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1436448554829742252)** (49 条消息🔥): 

> `UOps.after 限制, tinygrad 中的 CUDA Reduction, MPS 设备上的 Tensor.from_blob, tinygrad 中的 Style Transfer` 


- **`UOps.after` 限制探讨**：成员们讨论了 `UOps.after` 的使用限制，初步发现表明它应仅应用于 buffers，而非 comparisons，因为比较器的值相同。
   - 后来这被确定为一个 [linearizer bug](https://github.com/tinygrad/tinygrad/commit/ffb9e8396f9f78c7cd986f9e93be6dfb0fde88ed)，发生在对 `B` 和 `A` 中的索引同时调用 `.valid` 时，该问题[随后被解决](https://github.com/tinygrad/tinygrad)。
- **在 tinygrad 中处理 CUDA Warp Reduction**：一位成员寻求帮助，试图将利用共享内存和同步的 CUDA warp reduction 代码转换为 tinygrad，并展示了一个使用 `UOp`s 的初步实现。
   - 目标是复制 CUDA 的共享内存访问模式和基于 thread IDs 的条件更新，挑战在于确保在循环外进行正确的条件赋值。
- **`Tensor.from_blob` 在 Torch MPS Tensor 上的问题**：用户在将 MPS (Metal Performance Shaders) 设备上的 Torch tensors 转换为 tinygrad 时遇到了 `Tensor.from_blob` 的问题，导致内存访问相关的错误。
   - 虽然从 Torch MPS tensors 直接转换到 CPU 上的 tinygrad tensors 是可行的（可能涉及复制），但直接转换到 Metal 设备会导致 Jupyter kernel 崩溃并报错设备不匹配，这要求 Torch tensor 必须在同一设备上。
- **通过 tinygrad 实现 Style Transfer**：一位成员成功将 fast.ai p2 的 style transfer notebook 转换到了 tinygrad。
   - 生成的 [notebook](https://github.com/fzngagan/tinygrad-experiments/blob/main/16A_StyleTransfer_tinygrad.ipynb) 展示了在 tinygrad 框架内运行 style transfer 实验的可行性。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1436478910119411712)** (1 条消息): 

> `DSPy Planner, Multi-Agent Tool, Orchestrator` 


- ****DSPy Planner** 和 Orchestrator 解决 Multi Agent Tool 泛滥问题**：一位成员发布了一篇使用基于 **DSPy 的 planner** 和 orchestrator 来解决多 Agent 工具使用的文章，并征求反馈：[Solving Agent Tool Sprawl with DSPy](https://viksit.substack.com/p/solving-agent-tool-sprawl-with-dspy)。
- **Multi-Agent Tool 使用**：该文章探讨了使用 DSPy 进行 **multi-agent tool use** 的解决方案，重点关注规划（planning）和编排（orchestration）。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1436520362643816459)** (52 条消息🔥): 

> `DSPy 优化问题, DSPy 的 TOON Adapter, DSPy 的 Agent CLI 支持, DSPy 成功案例, DSPy 优化的反馈文本` 


- **DSPy 优化故障排除**：成员们在使用 **MIPROv2** 进行 **DSPy 优化** 时遇到错误，并请求提供有关设置和所遇错误的详细信息。
   - 有人提问这是否与 DSPy 中的 **BAML Adapter** 类似。
- **TOON Adapter PR 即将到来**：一名成员正在为 **DSPy** 中的 **TOON** 支持创建 **PR**，引发了对其性能测试的兴趣，但也有人担心潜在的性能下降。
   - 强调了需要通过 **evaluations** 来评估 **TOON 的性能** 并识别任何退化，特别是在结构化输出方面。
- **提议对 CLI Agent 提供一等公民支持**：已创建一个 Issue 来跟踪为使用 **DSPy** 编写 **Agent CLI** 添加**一等公民支持**的工作项，这与 [Agent Client Protocol 标准](https://github.com/agentclientprotocol/agent-client-protocol)保持一致。
   - 讨论内容包括这应该是由 **DSPy** 维护的兄弟项目，还是支持 **ZED ACP** 的原生模块。
- **呼吁建立 DSPy 成功案例子论坛**：有人提议在 **Discord** 中设立一个板块，用于分享和讨论 **DSPy 成功案例**，按任务类型（如分类 Prompt、信息提取 Prompt）分类，并附带相关的设置细节。
   - 还建议为 Student 和 Teacher 模型（Qwen3, Llama, GPT-5, Sonnet, Opus）设立独立的子论坛。
- **讨论引导 DSPy 优化的反馈文本**：一位成员分享了他们最喜欢的用于引导 **DSPy 优化** 的**反馈文本**，包括**正确**、**细节缺失**、**错误/遗漏**以及**幻觉提取**等标签。
   - 其他人讨论了这在实际系统中是否真的有帮助。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1436572098683338823)** (17 条消息🔥): 

> `Kimi 模型反馈, aider 与 Agentic 编程工具对比, aider-ce 分支, MoonshotAI Kimi K2` 


- **Kimi 模型的可靠性受到质疑**：一位成员询问了关于新 **Kimi 模型** 的可靠反馈，质疑模型在 **aider** 中是否比在沉重的 **Agentic 编程工具** 中表现得更聪明，并理论化地认为指令过载会损害模型表现。
- **Aider 的特异性提升了模型性能**：一些成员同意模型在 **aider** 中表现更聪明，因为模型 **prompting** 不那么冗长，并认为模型会放大给定的词汇，而在自主工作期间可能会被糟糕的思考过程带偏。
   - 他们表示 **aider** 强制要求更高的特异性，由于内部治理结构较少，防止了模型过度思考，从而提高了性能。
- **Aider 的开发转向 'aider-ce' 分支**：成员们注意到主维护者已经有一段时间没有向原始 **aider** 仓库提交代码了，引用了 [这个 Issue](https://github.com/Aider-AI/aider/issues/4613) 并指出开发已转移到 **aider-ce** 分支，称新的 **Agentic 模式** 令人惊叹。
   - 他们建议关注 [dwash96/aider-ce 仓库](https://github.com/dwash96/aider-ce) 以获取最新版本，该版本具有“飞跃式的改进”。
- **有趣的函数调用**：一位成员分享了一张 **Claude** *字面上通过编写函数来修复其格式错误*的图片。
   - 成员们提议需要一个频道来专门记录有趣的语言模型函数调用。
- **MoonshotAI Kimi K2 API**：一位成员询问通过 API 使用 **Kimi K2** 思考模型应使用哪个供应商，另一位成员推荐了 [OpenRouter](https://openrouter.ai/moonshotai/kimi-k2-thinking/providers)。
   - 建议是根据个人关注的维度进行排序。


  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1437404525047582781)** (11 条消息🔥): 

> `Prefill Latency, Chunking JSON for Aider, Summarizing JSON for Aider, Token Limits, Figma Designs` 


- ****Prefill 阶段延迟疑问****：一位成员询问了在使用基于 **prefill** 的方法处理输出 **token** 限制时，完成 **prefill 阶段**与在下一个请求中开始生成之间的典型**延迟**。
   - Discord 上下文中未提供具体答案。
- ****为 Aider 将 JSON 拆分为连贯的块****：一位用户尝试将一个非常长的 **Figma 对象 JSON** 传递给 Aider，但遇到了 **token 限制**。
   - 一位成员建议向 Aider 描述该文件，并要求它**编写一个脚本将其拆分为连贯的块**。
- ****总结 JSON 以适应 Token 限制****：一位成员建议 **Large Language Models** 可能不是处理此问题的正确工具，但脚本可以帮助总结 JSON，以便模型随后可以帮助编写执行下一步的代码。
   - 除非用户能够**总结并压缩文件或将其划分为组件**，否则无法一次性将其全部放入上下文。
- ****宣布使用 Figma 设计进行 Aider 测试****：一位用户正尝试使用 **Figma 设计**测试 Aider，并希望获得在将 JSON 提供给 Aider 之前对其进行预处理的建议。
   - 用户附带了一个 [contact-us.json](https://cdn.discordapp.com/attachments/1133060505792159755/1437438035154440382/contact-us.json?ex=69133dfd&is=6911ec7d&hm=ca6cc672684acc344bb49ea7fe25d58f8d0cb48994fe27645cfb195a76b6c7aa&) 作为他们正在处理的 JSON 文件类型的示例。


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1436484512560185375)** (9 条消息🔥): 

> `2025-11-25 Spec Release, SDK Changes and Review for SEP-1330, Agent Access to Slack and Gsuite APIs, MCP Client Interception of PII Data, Web Summit in Lisbon` 


- ****2025-11-25 规范发布计划****：`2025-11-25` 规范发布已与 [待定案的 SEPs](https://github.com/orgs/modelcontextprotocol/projects/26/views/8) 对齐，预计在 **2025 年 11 月 14 日**进行规范冻结。
- ****SEP-1330 等待 SDK 审查和合并****：在更改完成后，“等待 SDK 更改”标签已从 **SEP-1330** 中移除，目前正等待 TS/Python SDK 的审查和合并以及规范/架构更新。
- ****关于规范冻结后 SDK 更改的澄清****：在规范冻结后，SDK 更改可以独立继续，因为 SEP 视图主要关注规范措辞。
- ****Agent 访问 Slack 和 Gsuite API 的疑问****：一位成员询问了如何授予 **Agent** 访问 **Slack** 和 **Gsuite API** 的权限，并询问这是否涉及使用密钥设置环境并提供示例用法供 **Agent** 遵循。
   - 他们链接了一个关于[代码执行](https://discord.com/channels/1358869848138059966/1436084770114240512/1436365734027460720)的线程以获取更多详情。
- ****MCP 客户端的 PII 拦截验证****：一位成员询问了如何验证 **MCP 客户端**（如 Cursor 和 Claude）在识别和拦截 **PII 数据**方面的准确性。
   - 该成员询问了如何验证这些客户端的实现是否正确，以及它们如何准确且确定地识别 **PII**。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1436647849092845578)** (5 条消息): 

> `VEO3 connection issues, Subscription cancellation due to pricing, Expert engineer introduction` 


- ****VEO3 失去连接，Manus 失去视频功能****：一位用户报告与 **VEO3** 失去连接，导致 **Manus** 失去制作视频的能力，未提供更多上下文或链接。
   - 用户要求*从旧账户下载文本或代码并上传到新账户*。
- ****因“极其愚蠢”的 Token 费率取消订阅****：一位用户表示 **token** 费率*极其愚蠢*，在**几小时内花费了 99 美元**，并取消了订阅以寻求*更好、更便宜的选择*。
   - 他们补充道：*你们的服务定价简直疯了。外面有更好、更便宜的选择。*
- ****工程师展示专业知识：工作流自动化、LLM 集成和区块链****：一位擅长**工作流自动化、LLM 集成、RAG、AI 检测、图像/语音 AI 以及区块链开发**的资深工程师介绍了自己，强调了在实际应用中的强大记录。
   - 他们使用 **Dspy, OpenAI API 和自定义 Agent** 构建了自动化流水线，显著缩短了响应时间，并部署了带有向量数据库和混合搜索的高级 **RAG 流水线**。


  

---


---


---