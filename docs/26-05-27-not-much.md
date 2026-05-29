---
companies:
- eaglecorp
- vllm_project
- perplexity_ai
- alibaba
- lightseek
- nvidia
- mooncake
- flashattention
- kimmonismus
- deepseek
- xiaomi
- langchain
- baseten
- trajectory
- clay
- harvey
- decagon
- mercor
- rogo
- rlm
date: '2026-05-26T05:44:39.731046Z'
description: '**推理优化**日益趋向架构化。**EAGLE 3.1** 通过与 **vLLM** 和 **TorchSpec** 的协作，提升了推测解码和长上下文处理能力。**Perplexity**
  开源了重构的 **Unigram 分词器**，将 CPU 占用降低了 **5–6 倍**，在处理 514 个 token 时仅需 **63 微秒**。得益于**阿里巴巴**、**LightSeek**、**英伟达
  (NVIDIA)**、**Mooncake** 以及 **FlashAttention-4** 贡献者的共同努力，**Qwen3.5** 的推理速度达到了 **580
  tokens/s**。中国实验室 API 的降价之所以具有可持续性，是因为 KV 缓存和注意力机制得到了结构性改进，例如 **DeepSeek V4-Pro**
  和**小米 MiMo** 显著降低了缓存成本。


  智能体工程（Agent engineering）的重心正在从模型质量转向模型、开发框架与内存的适配（model-harness-memory fit）。**LangChain**
  发布了 **Deep Agents v0.6**，而 **LangSmith Engine** 等工具则实现了评估循环的自动化。**Trajectory** 获得
  **1500 万美元融资**并推出了持续学习平台，合作伙伴包括 **Clay** 和 **Harvey**；该平台支持在自动缩放的 **H100** 基础设施上部署大型模型，其中包括一个
  **3970 亿参数的模型**。此外，以内存为中心的开源智能体和极简训练框架也受到了关注。'
id: MjAyNS0x
models:
- eagle-3.1
- unigram-tokenizer
- qwen-3.5
- deepseek-v4-pro
- mimo
- deep-agents-v0.6
- 397b-parameter-model
people:
- kimmonismus
- _luofuli
- vtrivedy10
title: 今天没发生什么特别的事。
topics:
- inference-optimization
- long-context
- speculative-decoding
- tokenization
- attention-mechanisms
- kv-cache
- cache-hierarchy
- agent-engineering
- model-harness-memory-fit
- continual-learning
- quantization
- autoscaling
- memory-centric-agents
- evaluation-automation
---

**平静的一天。**

> 2026年5月26日至5月27日的 AI 新闻。我们检查了 12 个 subreddit、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，且没有检查更多的 Discord。[AINews 网站](https://news.smol.ai/) 允许你搜索所有往期内容。提示一下，[AINews 现在是 Latent Space 的一部分](https://www.latent.space/p/2026)。你可以[选择订阅或退订](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)不同频率的邮件！

---

# AI Twitter 综述

**推理效率、服务架构与成本曲线**

- **推理优化正日益趋向于架构层面，而不仅仅是 kernel 层面**：[EAGLE 3.1](https://x.com/EagleCorp/status/2059485457227149334) 通过稳定 hidden-state 反馈并减少深层解码步骤中的 attention 漂移，提高了 speculative decoding（投机解码）的鲁棒性，特别强调了 **long-context 接受长度**和实际服务中的可靠性；该团队还强调了与 [vLLM](https://x.com/vllm_project) 和 TorchSpec 的合作。在 kernel/系统层，Perplexity 开源了一个重构的 [Unigram tokenizer](https://x.com/perplexity_ai/status/2059664738087469511)，可降低 **5–6 倍** 的 CPU 占用率，并在零 heap 分配的情况下达到 **514 tokens 耗时 63 µs** 的性能；同时，据报道，通过阿里巴巴、LightSeek、NVIDIA、Mooncake 和 FlashAttention-4 贡献者的联合优化，[TokenSpeed 上的 Qwen3.5](https://x.com/Alibaba_Qwen/status/2059674574397313277) 在 agentic 工作负载下可达到 **580 tokens/s**。支持库也有所提升：[MaxSim v2](https://x.com/ErikKaum/status/2059659837219156453) 增加了 backprop（反向传播），据报告在 **H200 上比原生 PyTorch 快 10.33 倍**，在 **A100 上快 11.94 倍**。

- **结构性的 KV-cache 和 attention 变化证明了降价的合理性**：几篇帖子汇聚于同一个主题：中国实验室最近的 API 降价看起来是可持续的，因为它们反映了**更低的单位 token 服务成本**，而非暂时的补贴。[@kimmonismus](https://x.com/kimmonismus/status/2059578380329394292) 总结了 **DeepSeek V4-Pro** 如何利用混合 attention（包含 **Compressed Sparse Attention** 和 **Heavily Compressed Attention**）将 **1M-token 的 KV cache 降至 V3.2 的约 10%**，并将单 token 推理 FLOPs 降至 **27%**，同时仍能从 **1.6T 总参数**中路由出 **49B 激活参数**。小米的 MiMo 同样利用 SWA 加上分层缓存管理降低了缓存流量。[@_LuoFuli](https://x.com/_LuoFuli/status/2059618247553745204) 直接证实了这一点，他表示 MiMo 最大幅度的输入缓存命中降价源于 **5 倍的缓存 token 容量**、约 **80% 的缓存成本降低**，以及架构上 **1:7 的 Full:SWA 稀疏率**。更广泛的结论是：long-context 推理的经济效益目前正由 **attention 设计 + 缓存层级 + 路由**推动，而不仅仅是更便宜的硬件。

**Agent、Harnesses、内存与持续学习**

- **技术栈正在从“模型质量”转向“模型-harness-内存的匹配”**：大量推文聚焦于实际的 Agent 工程。LangChain 发布了带有 **Delta Channels** 的 [Deep Agents v0.6](https://x.com/LangChain/status/2059634226836746483)，将 200 轮 coding session 的 checkpoint 存储从 **5.3 GB 降至 129 MB**，并推出了 [Fleet 中的 computer use](https://x.com/LangChain/status/2059685293322858809) 以及用于版本化 Agent 上下文/技能的 [Context Hub](https://x.com/hwchase17/status/2059687279199924462)。[LangSmith Engine](https://x.com/LangChain/status/2059654417478012938) 被定位为自动化“评估 → 诊断 → 修复”循环的工具，多位从业者强调了它在将 trace 反馈转化为可复用的线上/线下评估器方面的价值。与此同时，[@Vtrivedy10](https://x.com/Vtrivedy10/status/2059712077925658717) 给出了当天最清晰的观点：**任务与 harness 的匹配度 (task-harness fit)** 与模型质量同样重要，定制化的垂直系统通过将工具、提示词和上下文收窄至特定任务，表现优于通用的 harness。

- **持续学习（Continual Learning）正在重新作为一种产品类别而兴起，而不仅仅是一个研究课题**：这里最重大的发布是 [Trajectory 的启动](https://x.com/rronak_/status/2059644771262730624)：这是一个利用**产品使用信号和 Agent 追踪（traces）**来持续后训练（post-train）大型 Agentic 模型的平台，获得了 **1500 万美元融资**，设计合作伙伴包括 Clay、Harvey、Decagon、Mercor 和 Rogo。Baseten 表示其利用 [FP8/NVFP4 量化和自动扩展的 H100 基础设施](https://x.com/baseten/status/2059651376565936510#m)来支持这些部署，其中包括一个被引用的** 397B 参数模型**的隔夜部署。同样的趋势也出现在开源工具中：一个基于 LangChain/LangGraph 构建的[以内存为核心的开源 Agent](https://x.com/hwchase17/status/2059487107144655356) 因其明确的检索/存储/推理/学习分离而受到多位开发者的赞赏；而 [RLM 的极简训练框架](https://x.com/a1zhang/status/2059633834094678173)表明，小团队现在可以在 **8×A100 上用一天时间**对长上下文 Agent 进行 RL 微调。贯穿其中的主线是，“部署后学习”正从愿景走向基础设施化。

**基准测试、缩放定律（Scaling Laws）与训练方法**

- **新基准测试越来越多地关注长周期、复杂且真实的现实世界工作流**：[DeepSWE](https://x.com/_philschmid/status/2059564676569076021) 被强调为一个 SWE/Agent 基准测试，涵盖了 **5 种语言、91 个仓库中的 113 个任务**，它使用极简的仅限 bash 的框架和更短的提示词，但与 SWE-Bench Pro 相比，仍需要 **5.5 倍的代码量**并平均涉及 **7 个文件**。在企业运营方面，Artificial Analysis 和 IBM 推出了 [ITBench-AA](https://x.com/ArtificialAnlys/status/2059698327235805258)，这是一个针对 Kubernetes 故障响应的 SRE 基准测试，其中**所有前沿模型的得分均低于 50%**；**Claude Opus 4.7** 以 **47%** 领先，**GPT-5.5** 以 **46%** 紧随其后，**GLM-5.1 Reasoning** 以 **40%** 在开源权重模型中领先。另一个有用的可靠性视角来自 [AgingBench](https://x.com/omarsar0/status/2059689897523642510)，它将已部署 Agent 的性能退化框架化为一个由压缩、干扰和内存更新引起的寿命问题。

- **训练效率研究在理论和系统层面依然活跃**：Sakana AI 的 [DiffusionBlocks](https://x.com/hardmaru/status/2059648995132367277) 是最具技术趣味性的发布之一：它将前向传播重新解释为类扩散的去噪步骤，使得深度网络可以**逐块（one block at a time）进行训练**，在显著减少内存占用的同时，在 **ViTs、DiTs、掩码扩散、自回归 Transformer 以及循环深度 Transformer** 上达到了端到端的性能匹配。在 RL 系统方面，Snowflake 推出了 [ZoRRo](https://x.com/StasBekman/status/2059718503318655314)，声称通过消除冗余的 Rollout 计算，可实现**高达 3.5 倍的长上下文 RL 加速**和 **3.2 倍更长的上下文窗口**，同时还发布了专门的 [Arctic-Text2SQL-R2](https://x.com/dwarak/status/2059686825086902398#m) 企业级 SQL 模型。在理论前沿，[Tiberiu Musat 的预印本论文](https://x.com/Tiberiu_Musat_/status/2059562156102746148)认为，对于固定精度的网络，最小神经权重范数在对数因子范围内与最小程序长度匹配；而 [Unified Neural Scaling Law](https://x.com/ethanCaballero/status/2059686905105563907) 提出了一种多变量函数形式，旨在比以往的拟合更准确地外推神经缩放行为。

**模型与模态发布：生物学、视觉、OCR 与嵌入式 AI**

- **蛋白质建模迎来了高光时刻**：[ESMFold2](https://x.com/alexrives/status/2059611151860683097) 被宣布为一个用于蛋白质结构预测和设计的开放科学引擎，在**蛋白质相互作用和抗体**方面报告了强劲的结果，并附带了一个包含 **68 亿个蛋白质**和 **11 亿个预测结构**的图谱。此次发布强调了实际的设计成果——针对五个治疗靶点的小蛋白结合剂和单链抗体——以及关于涌现蛋白质表示的机械可解释性发现。[@proteinrosh](https://x.com/proteinrosh/status/2059633089702240598) 对此发布表示了共鸣，[@cgeorgiaw](https://x.com/cgeorgiaw/status/2059694583856927201) 则进行了背景补充，指出该图谱在规模上超过了 AlphaFold DB。

- **一波规模较小但实用的多模态/开源发布相继落地**：Google DeepMind 分享了 [Gemini Embedding 2](https://x.com/mseyed/status/2059504005387284629) 的白皮书，将其描述为一个支持文本、图像、音频和视频统一表示的**原生多模态嵌入模型**。NVIDIA 的 [LocateAnything](https://x.com/wildmindai/status/2059600079804088790) 结合了 **Qwen2.5-3B + Moon-ViT** 用于高速 Grounding，声称在密集目标检测（dense object detection）方面实现了 **10 倍加速**。Hugging Face 集成了 Roboflow 的 [RF-DETR](https://x.com/mervenoyann/status/2059647988373373253)，定位为性能优于 YOLO 架构系统的实时检测/分割工具。在文档处理流程方面，[Surya OCR 2](https://x.com/VikParuchuri/status/2059675773712167423) 作为一个 **650M** 参数的模型发布，在 **OLMOCR 基准测试中达到 83.3%**，在**内部 91 种语言基准测试中达到 87%**，且在 **RTX 5090 上可达 5 页/秒**；[LiteParse v2](https://x.com/jerryjliu0/status/2059710330016817501) 使用 Rust 重写了解析逻辑，实现了**高达 100 倍的加速**，并可通过 WASM 进行边缘/浏览器端部署。端侧 AI 也备受关注，Google 推出了新的 [Coral board](https://x.com/googlegemma/status/2059740184930074758)，用于本地语音、视觉和控制演示。

**开发者平台、企业级控制与 Coding-Agent 产品化**

- **Coding Agent 正整合为具备企业级控制能力的完整产品栈**：OpenAI 继续收缩 Codex 的产品线：[GPT-5.2 和 GPT-5.3-Codex 正在从 Codex 中退役，取而代之的是 GPT-5.5](https://x.com/thsottiaux/status/2059650685948551384)，而企业级功能现在包括[基于仅出站 HTTPS 的私有 MCP 连接](https://x.com/OpenAIDevs/status/2059703536825565499)、[工作负载身份联邦（Workload Identity Federation）](https://x.com/OpenAIDevs/status/2059703600662925635)以及针对支出警报、白名单、保留策略和托管工具管理的[扩展 Admin API 控制](https://x.com/OpenAIDevs/status/2059703665276145920)。OpenAI 还发布了一个关于[使用 Codex 开发自我改进型税务 Agent](https://x.com/OpenAIDevs/status/2059638868983562640) 的具体案例研究，重点是将审阅者的修正追溯到评估（evals）和修复中。

- **Coding Agent 的竞争现在明显转向可靠性、工作流广度和企业采用率**：[Claude Code](https://x.com/ClaudeDevs/status/2059701677981413812) 分享了可靠性/性能更新以及更简便的错误报告捕获功能，而 GitHub 则通过 [Copilot Dev Days](https://x.com/code/status/2059664796178354617) 和 [MCP 定位](https://x.com/code/status/2059666498285629707)继续推动“Agent 化 IDE”的方向。最大的商业数据点来自 [Cognition](https://x.com/cognition/status/2059660758531940856)：**融资超过 10 亿美元，估值达 260 亿美元**，**企业使用量今年以来增长超过 10 倍**，**年化运行营收（run-rate revenue）达 4.92 亿美元**，并伴随着不断增长的客户名单以及来自 [Exa](https://x.com/nityasnotes/status/2059768072110776370) 等用户的强力认可。与此同时，一些较小的基础设施/产品动态表明生态系统正在扩大：[Cua Driver for Windows](https://x.com/trycua/status/2059688960838828391) 为 Windows Agent 带来了后台计算机使用功能；[Cloudflare 的 Agent 平台](https://x.com/brandonjcarl/status/2059624598644109363)因其“分数计算（fractional computing）”经济性而屡获好评；[Grok Build 的 worktree 支持](https://x.com/theskory/status/2059729539287167068)则针对代码库规模的多 Agent 代码集群（code swarms）。

**热门推文（按参与度排序）**

- **Cognition 的规模扩张**：[Cognition](https://x.com/cognition/status/2059660758531940856) 宣布**融资超 10 亿美元**，**估值 260 亿美元**，**年化运行营收 4.92 亿美元**。这是 Coding Agent 正在转化为大型企业业务的最清晰信号之一。
- **Claude Code 推动可靠性**：[Anthropic 的 ClaudeDevs](https://x.com/ClaudeDevs/status/2059701677981413812) 发布了关于响应速度、可靠性和更好反馈收集的高参与度更新——证明产品质量和信任度现在是核心战场。
- **Sakana AI 的 DiffusionBlocks**：[@hardmaru](https://x.com/hardmaru/status/2059648995132367277) 引起了对块状训练（block-wise training）的高度关注，这种训练方式可以在大幅降低显存需求的同时匹配端到端训练的性能。
- **ESMFold2 发布**：[@alexrives](https://x.com/alexrives/status/2059611151860683097) 宣布了当天最重要的科学领域发布之一：图谱规模（atlas scale）的开源蛋白质建模，对药物设计具有重要意义。
- **OpenAI 企业控制 + MCP**：[@OpenAIDevs](https://x.com/OpenAIDevs/status/2059703536825565499) 关于私有 MCP 及相关管理/安全更新的发布，反映了前沿 API 在争取大型机构采用方面的竞争态势。

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. 消费级硬件上的低比特本地 AI

  - **[PrismML 刚刚发布了 Binary 和 Ternary Bonsai Image 4B：这是一款 1-bit/三进制的文本生成图像 Diffusion Transformer，甚至可以 100% 在浏览器的 WebGPU 上本地运行。](https://www.reddit.com/r/LocalLLaMA/comments/1togflk/prismml_just_released_binary_and_ternary_bonsai/)** (热度: 759): **PrismML** 发布了 **Binary 和 Ternary Bonsai Image 4B**，被描述为 `1-bit`/三进制文本生成图像 Diffusion-Transformer 变体，其 Checkpoint 大小约为 `3GB`，采用 **Apache-2.0** 许可，并提供 WebGPU 浏览器 Demo（[HF collection](https://huggingface.co/collections/prism-ml/bonsai-image), [demo](https://huggingface.co/spaces/webml-community/bonsai-image-webgpu)）。该帖子将其与约 `16GB` 的 **FLUX.2 Klein 4B** 进行了对比；一条顶级技术评论声称 Bonsai Image 主要是 **FLUX.2 Klein 4B** 的量化/后训练衍生模型，除了白皮书外，在其他地方缺乏足够的致谢。主要的争论点在于归属权/品牌化：一位评论者认为 PrismML 将量化/微调后的基础模型重新包装为 “Bonsai”，同时尽量减少对原始实验室的致谢，将其比作将 Qwen 的量化版作为新模型发布。另一位评论者询问是否可以在 `16GB` RAM 的 CPU 上运行，但提供的评论中未给出技术答复。

    - 一位评论者指称 **PrismML 的 “Bonsai-Image” 并非全新训练的基础模型**，而是 **`FLUX.2 Klein 4B` 的二进制/三进制量化版**，并辅以后训练以恢复质量。他们认为该项目的 HF Demo/模型页面和 GitHub 忽略了对原始 FLUX 模型/团队的明确致谢，据报道仅在白皮书中提到了原始模型。
    - 一条技术可用性说明提到，浏览器/WebGPU 模型下载大约需要 **`~2 GB`**，尽管其声称采用 1-bit/三进制压缩，但这对于完全本地推理仍具有参考意义。另一位用户询问是否可以在 **16 GB RAM 的 CPU** 上运行，但讨论中未给出具体的基准测试或兼容性回答。

  - **[厌倦了 4GB GPU 上的 OOM 错误。我编写了一个自定义 Rust 裸机引擎，在 4B 模型上达到了 66.8 TPS（RTX 3050 上的 BitNet 1.58b）。](https://www.reddit.com/r/LocalLLM/comments/1to6enj/got_tired_of_oom_errors_on_my_4gb_gpu_wrote_a/)** (热度: 390): **原作者（OP）声称一个自定义的 Rust/C++ LLM 推理引擎 **Cluaiz**，在 **RTX 3050 4GB** 上以 `1.58-bit` 量化运行 `prism-ml/Bonsai-4B-gguf`，达到了 `66.8 tokens/s`，并报告通过动态 KV-cache 管理，Gemma/Qwen 4B 变体在不发生 OOM 的情况下达到 `~30–33 TPS`。帖子中尚未提供可复现的仓库或基准测试产物；评论者指出了显现的项目链接（[GitHub](https://github.com/cluaiz/cluaiz), [网站](https://cluaiz.com/)）并质疑如 *“direct-to-silicon”*（直接访问芯片）等模糊说法，指出这可能仅仅意味着提前原生编译（Ahead-of-time native compilation），而非任何不寻常的 GPU/驱动程序层级的机制。由于 Reddit `HTTP 403` 限制，无法独立访问附带的 Reddit 视频。顶级评论表示强烈怀疑，认为文案和仓库语言具有伪技术/AI 生成的特征，并认为所述成就相当于基础的原生编译加单机演示。评论者还对该项目在 Apache 2.0 下的许可/版权措辞提出了挑战，并询问声称的底层硬件访问背后的具体实现细节。

    - 评论者质疑了链接仓库（[github.com/cluaiz/cluaiz](https://github.com/cluaiz/cluaiz), [cluaiz.com](https://cluaiz.com/)）中的技术主张，认为诸如 **“direct silicon access”**、“bare-metal engine” 和 “copyrighted Apache licensed software” 之类的描述更像是营销或 LLM 生成的伪技术语言，而非具体的实现细节。一位评论者询问 “direct silicon access” 是否仅仅意味着 **Rust 中的提前原生编译**，而非除了正常 CUDA/驱动 API 之外的任何真实底层 GPU 编程。
    - 几位评论者认为，声称的结果应该与现有工具进行对比，尤其是 **llama.cpp**，它已经支持消费级 GPU 上的低内存推理和量化模型。批评意见认为，`4GB` RTX 3050 上的 OOM 问题通常可以通过适当的 llama.cpp 配置解决，而不需要编写新引擎，因此在 `4B` BitNet 1.58b 模型上达到 `66.8 TPS` 的说法需要可复现的基准测试和配置细节才有意义。


### 2. Qwen 3.5/3.6 本地模型发布与编程测试

- **[Qwen3.5 35B A3B uncensored heretic Native MTP Preserved 现已发布，完整保留 785 个 MTP，提供 Safetensors、GGUF、NVFP4、NVFP4 GGUF 和 GPTQ-Int4 格式](https://www.reddit.com/r/LocalLLaMA/comments/1tnzalm/qwen35_35b_a3b_uncensored_heretic_native_mtp/)** (活跃度: 602): **llmfan46** 发布了 [`Qwen3.5-35B-A3B-uncensored-heretic-v2-Native-MTP-Preserved`](https://huggingface.co/llmfan46/Qwen3.5-35B-A3B-uncensored-heretic-v2-Native-MTP-Preserved)，这是 `Qwen/Qwen3.5-35B-A3B` 的去审查衍生版本，使用 **Heretic v1.3.0** / 幅度保留正交消融（Magnitude-Preserving Orthogonal Ablation）风格的编辑，针对 `attn.o_proj`、`attn.out_proj` 和 `mlp.down_proj` 进行修改，同时保留了所有 `785` 个原生 MTP 张量。模型卡报告显示，拒绝率从 `92/100` 降低到 `14/100`，与基座模型相比 KL 散度为 `0.0487`，在 `7,021` 个问题中 MMLU 仅从 `84.12%` 下降到 `83.72%`；发布版本包括 [Safetensors](https://huggingface.co/llmfan46/Qwen3.5-35B-A3B-uncensored-heretic-v2-Native-MTP-Preserved)、[GGUF](https://huggingface.co/llmfan46/Qwen3.5-35B-A3B-uncensored-heretic-v2-Native-MTP-Preserved-GGUF)、[NVFP4](https://huggingface.co/llmfan46/Qwen3.5-35B-A3B-uncensored-heretic-v2-Native-MTP-Preserved-NVFP4)、[NVFP4 GGUF](https://huggingface.co/llmfan46/Qwen3.5-35B-A3B-uncensored-heretic-v2-Native-MTP-Preserved-NVFP4-GGUF) 和 [GPTQ-Int4](https://huggingface.co/llmfan46/Qwen3.5-35B-A3B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4) 变体。作者认为 Qwen3.5 和 Qwen3.6 虽然都使用 `qwen35` 架构，但针对不同的领域进行了调优——Qwen3.5 用于通用辅助，Qwen3.6 用于 Agent/编码——并指出这两个系列的消融 KL/质量表现存在显著差异。评论者对 **NVFP4 GGUF** 构建版本的罕见提供表示赞赏，其中一人指出，即使在 Unsloth 那里也找不到类似的发布。另一位测试者同意作者的定位，将 Qwen3.6 描述为更接近 *“3.5 coder+”*，而不是 Qwen3.5 简单的全面继任者。

    - 一位评论者强调了 **NVFP4 GGUF** 构建版本的实用价值，指出这种格式在其他地方很难找到：*“我真的找不到其他人在做这个，甚至连 Unsloth 都没有。”* 这在技术上具有相关性，因为对于目标是新型 NVIDIA 导向低精度推理工作流、同时仍使用基于 GGUF 运行时的用户来说，NVFP4 GGUF 的可用性至关重要。
    - 一位测试者对比了 **Qwen3.5** 和 **Qwen3.6**，认为 3.6 感觉更像是 *“3.5 coder+”*，而非直接的通用升级。他们认为发布间隔过短，不太可能出现广泛的能力跨越，暗示 3.6 可能更侧重于编码专业化，而不是 3.5 的简单继任者。

  - **[Okay 27B 让我信服了](https://www.reddit.com/r/LocalLLaMA/comments/1to73op/okay_27b_made_me_a_believer/)** (活跃度: 541): **发帖者报告称，通过 Opencode 使用的 Qwen 系列 `27B` 模型，根据描述控制台 API、手柄控制和 TypeScript shader 的三个参考文件，一次性（one shot）生成了一个近乎完整的 HTML5 Breakout 风格游戏。输出结果立即可玩，包含有效的控制、声音、元数据、save/stat/heartbeat API 集成，仅需要一次后续定制和一次故障修复；一位评论者建议启用 MTP/投机采样 (Speculative Decoding) 并设置 `2–3` 个草稿 Token 以提高速度。另一位资深用户表示，该模型在 `64K` 上下文以下表现最好，超过 `64K` 后退化明显，在 `128K` 之后“性能骤降”，建议在长时间的 Agent 编码任务中定期总结到文件并重置会话。** 评论者认为该稠密型 `27B` 模型在本地编码方面异常强大——在 Web 应用的一次性生成方面 *接近 Sonnet 级别*——而一位用户发现 `35B A3B` 尽管有尺寸/路由优势，但能力较弱。主要的警示是，长上下文的 Agent 运行可能会导致循环或“变笨”，因此用户应积极管理上下文。

- 一位评论者建议启用 **MTP/speculative decoding** 以获得更好的吞吐量，并建议将 MTP 值设为 `2` 或 `3` 作为速度与质量之间实用的权衡。这是一种部署层面的优化，而非模型质量的改进，对于在本地运行 27B 模型的用户非常有用。
- 一位用户报告称，27B 模型的有效推理质量在长上下文下会明显下降：**在 `64K` tokens 以下表现最佳**，超过 `64K` 后开始退化，而在 *“超过 `128K` 后推理质量大幅跌落”*。对于长周期的 Agent 任务，他们的解决方法是定期将状态总结到一个文件中，重启框架/会话 (harness/session)，并重新加载总结以恢复模型质量并避免陷入循环。
- 一位基准测试操作员表示 **Qwen 27B** 表现异常突出，以至于他们重新检查了测试方法。在他们的排名中，该模型*大致与 GPT-5.2 或 Sonnet 4.5 持平*，同时指出它在较大上下文尺寸下表现吃力，这可能是受限于参数量。他们在 [gertlabs.com/rankings](https://gertlabs.com/rankings) 链接了相关数据。

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude Code Vibe-Coding 实践

- **[你用 Claude 构建的东西对我来说毫无用处……而这正是重点所在](https://www.reddit.com/r/ClaudeAI/comments/1tp3en9/the_thing_you_built_with_claude_is_useless_to_me/)** (热度: 1152): **该帖子认为，许多由 **Claude 构建的 “vibe coded” 工具**——例如：个人健康相关分析器、Garmin 数据归档器、特定商店的杂货分类器、利基生物信息学流水线以及终端错误解释器——之所以有价值，恰恰是因为它们是**高度个性化的产物**，而非可复用的产品。作者建议，公开仓库和帖子应该记录*问题框架化过程 (problem-framing process)*——包括摩擦点、失败的替代方案以及现有工具为何不适用——因为这种认知模式比代码本身更容易迁移。** 顶层评论者普遍表示赞同，将 AI 辅助开发视为向**个人化软件 (personal software)**的转变；有人将 *vibe coding* 比作“软件开发的 3D 打印”。另一位评论者指出，该帖子的文风感觉像是 AI 生成的，但仍然认为其潜在观点具有新意和建设性。

    - 一位评论者报告称，AI 有效地实现了其技术文档工作流的自动化，声称排版内容、格式和整体质量提升了约 `10x`，而耗时仅为之前的约 `1/100`。他们还指出，AI 让他们能够完成以前“甚至无法开始”的文档任务，这表明主要的生产力提升在于降低了结构化技术写作的启动/技能门槛。

- **[我是一名拥有十年经验的软件工程师。如果我今天从零开始学习使用 Claude Code 构建应用，我会这样处理：](https://www.reddit.com/r/ClaudeAI/comments/1tonzj9/im_a_software_engineer_with_a_decade_of/)** (热度: 919): **一位高级 SWE 认为，使用 **Claude Code/vibe coding** 的初学者应该自顶向下地学习应用架构，而不是从实现细节开始：典型的 Web 应用被框架化为 **前端 + 后端 + 数据库 + “衔接代码 (plumbing)”**。强调的生产就绪层 (production-readiness layer) 包括 `APIs`、托管/DNS/部署、环境变量/机密 (secrets)、身份验证 vs 授权、备份、Git/版本控制、测试、监控/错误跟踪以及分析；作者还开始在 [vibe-blog.pages.dev](https://vibe-blog.pages.dev/) 收集后续材料。** 顶尖的技术反对意见指出，这种架构强烈倾向于 **以 Web 服务/全栈为中心**，并非普适：嵌入式、模拟、科学/工业、国防、光学、有限元分析 (FEA)、控制系统以及其他利基软件可能并没有前端/后端/数据库的划分。评论者普遍同意前期架构至关重要，并警告称，如果基础设计不佳，接近 ~`10k` 行代码 (LOC) 的项目会迅速积累难以重写的“拜占庭式 (Byzantine)”耦合。

- 一位评论者指出，帖子将应用开发界定为以前端/后端/数据库为中心的框架，主要适用于 Web 服务，但忽略了许多高薪的嵌入式/科学/工业软件领域，在这些领域中，应用可能没有后端，仅编写日志。引用的例子包括**黑体辐射器控制 (blackbody radiator control)**、**准直器模拟 (collimator simulation)**、光学透镜设计、放射学和材料有限元分析 (FEA)——在这些岗位中，领域专业知识与编程技能同样重要。
- 有一个技术架构警告：一旦项目接近 `10,000` 行代码，产生严重的结构性问题（且这些问题往往是被修补而非重新设计）的概率会迅速上升。评论者强调，像 **Netflix** 这样的消费级系统和像**美国电网 (American power grid)** 这样的关键基础设施，尽管领域迥异，但都可能陷入类似的“如果不进行重大重写就无法修复”的故障模式。
- 关于 Claude Code 计费的一个陷阱：如果 Shell 环境中存在 `ANTHROPIC_API_KEY` 或从 `.env` 文件中继承了该密钥，Claude Code 的请求可能会在不经意间从 API 账户扣费，而不是使用 **Max plan** 订阅配额。这也影响了从 cron 或子进程运行的 `claude -p`；解决方法是从子进程环境中剥离该密钥，使 Claude Code 回退到 OAuth 凭证登录。

  - **[得益于 Claude Code，我（一个编程外行）成功构建了 Questboard，这是一个用于我们平板墙面显示的家庭 RPG 风格家务看板。在午夜前完成家务以击败怪物并赚取金币，否则它会反击。在奖励商店中使用金币购买全家商定好的奖励。](https://www.reddit.com/r/ClaudeCode/comments/1tolrav/thanks_to_claude_code_i_a_coding_amateur_was_able/)** (热度: 905): **一位自称编程外行的人使用 **Claude Code** 构建了 [**Questboard**](https://github.com/thillygooth/questboard)，这是一个面向家庭、旨在用于平板墙面显示的 RPG 风格家务看板。该应用将家务游戏化为限时的“怪物”遭遇战：在午夜前完成家务可获得游戏金币，而失败则会让怪物“反击”；金币随后可以在家庭约定的奖励商店中消费。** 评论大多正面且非技术性，称赞这是一个温馨、非商业化的 AI 辅助编程现实案例；一位评论者询问了关于平板墙面设置的更多细节。

### 2. 企业 AI 工具支出与治理

  - **[公司给我们所有人提供了无限量的 Claude Code Sonnet 4.6 —— 现在每周发布一次谁消耗 Token 最多的排行榜。有什么建议能让我登顶吗？](https://www.reddit.com/r/ClaudeAI/comments/1tob45x/company_gave_us_all_unlimited_claude_code_sonnet/)** (热度: 2168): **图片是一个内部 EngOps 电子表格/使用仪表盘（[图片](https://i.redd.it/hnki8byc5i3h1.png)），显示了按用户分类的每周 **Claude Code Sonnet 4.6** Token 消耗量，按排行榜排序，从约 `2.5M` Token 到 `57k` 不等。从语境上看，这篇帖子与其说是关于模型基准测试，不如说是关于组织对 LLM 支出的使用跟踪/游戏化；顶层技术评论建议将 Claude 用作编排者/产品经理 Agent，将待办事项分解为并行的 Sonnet Agent 任务，同时保持产出的可解释性，以防高使用量被审计。** 评论者开玩笑说 `2.5M` Token 只是“菜鸟水平”，但主要警告是，除非使用量能对应到可证明的项目产出，否则刻意登顶排行榜可能会适得其反。一位评论者提议拥抱排行榜，将排名背景告知 Claude，并要求它规划有用的 Sprint，而不是仅仅为了消耗 Token。

    - 一个具有技术实质意义的建议是**将 Claude Sonnet 作为编排者**：给它一个真实的待办事项/问题，让它生成一个全面的计划，然后启动多个 Sonnet 会话作为执行 Agent，并让原始会话调度实现步骤。评论者将其描述为一个包含 Sprint 规划、每日总结的产品管理循环，并将受控的 Token 使用与有用的交付物挂钩，而非盲目消耗。
    - 一位评论者链接了一个开源的 Claude 技能库 [**RampStack Claude Skills**](https://github.com/rampstackco/claude-skills)，旨在让 Claude 在软件/产品生命周期中表现得更像产品经理。建议的工作流是提供痛点/待办事项，让 Claude 规划 “Sprints”，委托给其他 Agent，并生成总结解释构建了什么。
    - 另一位评论者分享了 [**Ordinath/tokenburn**](https://github.com/Ordinath/tokenburn)，显然这是一个专门用于消耗 Token 的工具。这与最大化排行榜使用量直接相关，尽管该线程未提供该工具本身的基准数据、实现细节或效率分析。

- **[Microsoft 已开始取消 Claude Code 许可，据 Verge 报道](https://www.reddit.com/r/ClaudeAI/comments/1to6kqz/microsoft_has_started_canceling_claude_code/)** (热度: 1712): **该 [图片](https://i.redd.it/4nskxdbpeh3h1.png) 是一个**非技术性的梗图**，使用 *I, Robot* 中的场景来调侃即使是 AI 编程助手也会被“裁员”，以此回应帖子中关于 **Microsoft 正在取消 Claude Code 许可**的说法。评论中的技术背景集中在报道称内部转向 **标准化的 GitHub Copilot 采用**，用户注意到即将到来的 Copilot 定价/配额变化，以及此前通过企业工具大量使用 **Claude Sonnet** 的情况。** 评论者争论这究竟主要是针对 Claude 的成本削减举措，还是仅仅是 Microsoft 将开发人员整合到 GitHub Copilot 上；一位用户警告说，新的定价可能会使目前 `$40` 档位的用量成本升至约 `$600`，而另一位用户则认为 Microsoft 仍然运行着自己的模型基础设施，这更多是为了标准化。

    - 评论者指出，即将到来的 **GitHub Copilot 定价变化** 可能会实质性地减少企业对基于 Claude 用量的访问：有人声称其企业配额预计将下降约 `6 倍`，而此前大部分用量都消耗在 **Claude Sonnet** 上。该评论者估计，按照新定价，其目前每月花费 `$40` 的个人工作量将映射到大约每月 `$600`，而重度用户在固定费率计划下可能已经消耗了价值“数千美元”的推理资源。
    - 一个技术相关的解读是，Microsoft 取消 Claude Code 许可可能并非为了放弃 AI，而更多是为了将内部工具统一到作为标准化接口的 **GitHub Copilot**。一位评论者指出，Microsoft 仍然可以在内部运行模型基础设施，这表明这一转变可能是由采购、计量和平台控制驱动的，而非简单的模型弃用。
    - 几条评论将此问题视为从补贴后的 AI 访问向真实 token 经济的修正：固定费率或 VC 补贴的计划掩盖了高容量 coding-agent 使用的真实推理成本。讨论暗示，组织很快将需要核算每个 token/每个请求的成本，特别是当 Agentic 编程工具生成大型 context windows 并进行重复的模型调用时。

  - **[Uber CTO 表示 Uber 在前四个月内耗尽了 2026 年的全部 AI 预算](https://www.reddit.com/r/ChatGPT/comments/1tp7ips/so_uber_cto_said_that_uber_burned_their_total/)** (热度: 833): **[Cybernews](https://cybernews.com/ai-news/uber-ai-return-of-investment-token-usage/) 报道称 **Uber 在四个月内耗尽了其 2026 年的 AI 预算**，COO **Andrew Macdonald** 表示公司仍无法将增加的 **Claude Code token 消耗** 与成比例的、面向消费者的有价值功能产出相匹配。讨论中心在于企业 AI 成本控制：基于用量的 token 计费增长速度可能快于实现的生产力提升，尤其是当鼓励员工在没有每用户/模型级成本核算的情况下“到处使用 AI”时。** 评论者认为，许多公司造成超支问题是因为没有激励员工优化 token 使用或选择更便宜的模型；一位用户表示他们自己的公司转向了每月 `$100` 的 AI 预算限额，最高可扩展至 `$250`，但他们在正常的日常工作流中一天就能消耗掉 `$100`。另一位评论者将此问题斥为“能力问题 (skill issue)”，暗示是使用习惯缺乏自律，而非根本性的 AI 经济学问题。

    - 一位评论者描述了在推行“AI Everywhere / Agents / 自动化一切”之后，企业内部出现的具体成本控制转变：其公司将用户的每月 AI 支出上限设定为 `$100`（可申请延至 `$250`），但他们在正常工作流下甚至能在一天内消耗 `$100`。他们指出，这将要求对使用模式进行显式优化，意味着不受管理的 Agentic/LLM 使用会迅速超出每用户的预算假设。
    - 另一个被提出的技术关注点是激励机制设计：当 AI 成本从个人工作流中抽象剥离出来时，员工几乎没有动力去减少 token 使用或选择更便宜的模型。这指向了围绕模型路由、token 预算和默认模型选择的治理问题，而不仅仅是模型成本本身的问题。


# AI Discords

遗憾的是，Discord 今天关闭了我们的访问权限。我们不会以这种形式恢复它，但我们很快就会发布新的 AINews。感谢阅读到这里，这是一段美好的历程。