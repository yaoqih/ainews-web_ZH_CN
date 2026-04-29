---
companies:
- vllm
- poolside
- nvidia
- opensrouter
- lmstudio
- ollama
- unsloth
- fal
- fireworks
- deepinfra
- togethercompute
- baseten
- canonical
date: '2026-04-28T05:44:39.731046Z'
description: '**vLLM v0.20.0** 引入了内存和 MoE（混合专家模型）服务效率方面的重大改进，包括支持 **4 倍 KV 容量**和 **2.1%
  延迟优化**的 **TurboQuant 2-bit KV 缓存**。此次更新支持多种硬件平台，如 **Blackwell 上的 DeepSeek V4 MegaMoE**、Jetson
  Thor、ROCm、Intel XPU 以及 Grace-Blackwell 配置。早期基准测试显示，在 **B300** 硬件上运行的 **DeepSeek
  V4 Pro** 速度最高可达 H200 的 **8 倍**。生态系统正在迅速实现对 **Poolside Laguna XS.2**、**Ling-2.6-flash**
  和 **NVIDIA Nemotron 3 Nano Omni** 等新开源模型的“零日支持”（day-0 support）。


  **Poolside** 发布了 **Laguna XS.2**，这是一款基于 **Apache 2.0** 协议的编程模型，采用 **33B 总参数 / 3B
  激活参数的 MoE** 架构，可在单 GPU 上运行。该模型具备混合注意力机制和 FP8 KV 缓存，性能接近 **Qwen-3.5**。


  **NVIDIA** 推出了 **Nemotron 3 Nano Omni**，这是一款 **30B 总参数 / 3B 激活参数（A3B）的多模态 MoE**
  模型，拥有 **256K 上下文长度**，支持文本、图像、视频、音频和文档，并已在多个平台同步分发。相关讨论强调了量化方法的权衡，以及从 CUDA 绑定向异构加速器支持的转变。'
id: MjAyNS0x
models:
- vllm-0.20.0
- poolside-laguna-xs.2
- ling-2.6-flash
- nemotron-3-nano-omni
- qwen-3.5
people:
- jeremyphoward
- maharshii
- teortaxestex
- aymericroucher
- piotrz
title: 今天没发生什么特别的事。
topics:
- memory-optimization
- mixture-of-experts
- model-optimization
- inference-speed
- quantization
- model-deployment
- multimodality
- hardware-optimization
- model-benchmarking
- open-models
- agentic-ai
---

**平静的一天。**

> 2026年4月27日至4月28日的 AI 新闻。我们检查了 12 个 subreddits，[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有检查更多的 Discord。[AINews 网站](https://news.smol.ai/) 允许你搜索所有往期内容。提醒一下，[AINews 现在是 Latent Space 的一个板块](https://www.latent.space/p/2026)。你可以[选择加入或退出](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) 邮件接收频率！

---

# AI Twitter 回顾

**推理系统、vLLM 0.20 以及围绕 DeepSeek V4 的硬件/内核竞赛**

- **vLLM 的最新版本重点关注内存和 MoE 服务效率**：[vLLM v0.20.0](https://x.com/TeksEdge/status/2048983564801450315) 发布，支持 **TurboQuant 2-bit KV cache** 以提供 **4 倍的 KV 容量**，在 **SM90+** 上为 **MLA prefill** 重新启用了 FA4，引入了新的 **vLLM IR** 基础，融合 **RMSNorm** 据报道带来了 **2.1% 的端到端延迟提升**，此外还更新了对 **Blackwell 上的 DeepSeek V4 MegaMoE**、Jetson Thor、ROCm、Intel XPU 的支持，并简化了 GB200/Grace-Blackwell 的配置。与此同时，[SemiAnalysis](https://x.com/SemiAnalysis_/status/2048957715955765284) 强调了 **DeepSeek V4 Pro** 在 **B200/B300/H200/GB200 解耦配置（disaggregated setups）**下的早期推理服务结果，声称在此类工作负载下 **B300 的速度最高可达 H200 的 8 倍**，并指出即将进行的 vLLM 0.20 与 **DeepGEMM MegaMoE** 的基准测试，后者将 **EP dispatch + EP combine + GEMMs + SwiGLU** 融合进了一个单一的 mega-kernel 中。
- **生态系统正趋向于为新的开源模型提供快速的 Day-0 支持**：[vLLM 增加了对 Poolside Laguna XS.2 的 Day-0 支持](https://x.com/RedHat_AI/status/2049150925944566180)，并分别支持了 [Ling-2.6-flash](https://x.com/vllm_project/status/2049158062666399909)，同时 [vLLM 还发布了对 NVIDIA Nemotron 3 Nano Omni 的 Day-0 支持](https://x.com/vllm_project/status/2049171268344426846)。在 vLLM 之外，几篇帖子关注推理服务的权衡：[Jeremy Howard 注意到 DeepSeek V4 对 prefill 的支持](https://x.com/jeremyphoward/status/2049098509530583199)，这是许多供应商已经放弃的一项能力；而 [Maharshi](https://x.com/maharshii/status/2049058891389108640) 指出了 **dynamic activation quantization** 的开销，认为尽管存在校准成本，**static quantization** 在推理速度上通常更具优势。人们对替代栈的可移植性也越来越感兴趣：[teortaxesTex 认为 DeepSeek 正在通过 TileKernels 在结构上摆脱 CUDA 锁定](https://x.com/teortaxesTex/status/2049185408785998217)，这表明模型厂商可能会越来越多地针对异构或国产加速器集群进行优化，而不仅仅是针对 NVIDIA 的部署。

**开源模型发布：Poolside Laguna XS.2、NVIDIA Nemotron 3 Nano Omni 和 TRELLIS.2**

- **Poolside 发布了其首个公开模型，这是一个非常便于部署的开源权重编程模型**：[@poolsideai 宣布了 Laguna XS.2](https://x.com/poolsideai/status/2049144111626670282)，这是一个完全自主训练的 **33B 总参数 / 3B 激活参数 MoE** 编程模型，采用 **Apache 2.0** 协议发布，并宣称可以在 **单张 GPU** 上运行。[Poolside 的更大规模发布](https://x.com/eisokant/status/2049142230397370537)还包括 **Laguna M.1** 和一个 Agent 套件，强调公司是在自有的 **data、training infra、RL 和 inference stack** 上从零开始训练的。社区总结增添了更多细节：[Aymeric Roucher](https://x.com/AymericRoucher/status/2049156715304935451) 描述了两个编程模型——**225B/23B 激活** 和 **33B/3B 激活**——采用 **hybrid attention**、**FP8 KV cache**，并声称性能接近 **Qwen-3.5**；[Ollama](https://x.com/ollama/status/2049184817603031463) 立即上线了该模型。
- **NVIDIA 的 Nemotron 3 Nano Omni 是当天最重磅的基础设施原生模型发布**：[@NVIDIAAI 推出了 Nemotron 3 Nano Omni](https://x.com/NVIDIAAI/status/2049159441870717428)，这是一个开源的 **30B / A3B 多模态 MoE**，具有 **256K context**，专为涵盖 **text、image、video、audio 和 documents** 的 Agent 工作负载而构建。分发工作在整个生态栈中立即展开：[OpenRouter](https://x.com/OpenRouter/status/2049164366218772526)、[LM Studio](https://x.com/lmstudio/status/2049172192705864091)、[Ollama](https://x.com/ollama/status/2049194377751437470)、[Unsloth](https://x.com/UnslothAI/status/2049161390150365344)、[fal](https://x.com/fal/status/2049160999442198632)、[Fireworks](https://x.com/FireworksAI_HQ/status/2049159136802398546)、[DeepInfra](https://x.com/DeepInfra/status/2049158141070524815)、[Together](https://x.com/togethercompute/status/2049160446708711883)、[Baseten](https://x.com/baseten/status/2049160818575749300)、[Canonical](https://x.com/Canonical/status/2049159988174602712) 等均宣布当天可用。后续推文中透露了关键规格：[Piotr Żelasko](https://x.com/PiotrZelasko/status/2049162049599455725) 将其描述为 NVIDIA 的首个 **omni** 发布，具有由 **Parakeet encoder** 支持的语音/音频理解能力，目前仅限 **English**，在 Open ASR 排行榜上的 **WER 为 5.95%**。几家托管商提到，其 **throughput 约为同类开源 omni 模型的 9 倍**。
- **其他值得关注的模型/论文发布**：[Microsoft 的 TRELLIS.2](https://x.com/kimmonismus/status/2049099376476459372) 是一个开源的 **4B image-to-3D 模型**，可生成高达 **1536³ PBR 纹理资产**，基于具有 **16× 空间压缩** 的原生 3D VAE 构建。在世界模型（world-model）方面，[World-R1](https://x.com/wjwang2003/status/2049136028968272260) 声称现有的视频模型已经编码了 **3D structure**，并可以通过 **RL** “唤醒”，**无需架构更改、无需额外的视频训练数据，且不增加推理成本**。

**Agent、本地优先工具链与生产级编排**

- **Agent 构建者正在从演示阶段转向生产级原语 (Production Primitives)**：[Mistral 发布了 Workflows 的公测版](https://x.com/MistralAI/status/2049128071874179091)，这是一个编排层，旨在将企业级 AI 流程转化为持久、可观测且容错的生产系统。相关帖子也呼应了这一主题：[Sydney Runkle 将持久执行 (Durable Execution)](https://x.com/sydneyrunkle/status/2049132897227936073) 视为长时运行 Agent 的核心需求；[threepointone 描述了关于子 Agent / Agent-as-tools 的工作](https://x.com/threepointone/status/2049088722835042475)，这些工具具备持久化、流式传输和状态恢复功能。
- **本地/离线 Agent 从愿景转变为可靠的工作流**：[Teknium 断言“完全离线的 Agent 是可能的”](https://x.com/Teknium/status/2048975223853350976)，同时 [Niels Rogge 演示了 Pi + 本地模型](https://x.com/NielsRogge/status/2049128153658839324)用于桌面清理，[Google Gemma 分享了本地编程 Agent 的教程](https://x.com/googlegemma/status/2049163687639007451)。Hugging Face 的本地化推广也体现在采用率数据上：[Clement Delangue 表示已有 300,000 名用户在 Hub 上添加了硬件规格](https://x.com/ClementDelangue/status/2049139562929143917)，以探索哪些模型可以在本地运行。作为补充，[Ammaar 开源了一个使用 MLX 在设备端完全运行 Gemma 4 的 vibe-coding 应用](https://x.com/ammaar/status/2049169134429073471)，[Kimmonismus 重点介绍了 Sigma](https://x.com/kimmonismus/status/2049244932477759767)，这是一个基于浏览器的私有本地 Agent 概念，使用了开源模型。
- **Hermes 及相关的 Agent 框架正获得实际应用落地**：多篇帖子报道称 Hermes 在指令遵循或实际工作流中的表现优于 OpenClaw，包括 [SecretArjun](https://x.com/SecretArjun/status/2049006382763110639)、[somewheresy](https://x.com/somewheresy/status/2049089485938315614)，以及通过 [Telegram](https://x.com/lizliz404/status/2049084890717806877) 部署 Hermes 或将其用于 [医学文献提取](https://x.com/bobvarkey/status/2049120693649125687) 的用户。在研究型 Agent 方面，[Hugging Face 的 ML Intern](https://x.com/_lewtun/status/2049021398312468815) 在 Spaces 中走红，随后又加入了 [原生指标日志记录 + Trackio 集成](https://x.com/akseljoonas/status/2049183527703396699)，使其训练任务变得可观测，而非黑盒状态。

**值得关注的基准测试、评估 (Evals) 与研究发现**

- **模型基准测试依然碎片化，但一些信号值得关注**：[Epoch 报告称 GPT-5.5 Pro 在 Epoch 能力指数上达到 159](https://x.com/EpochAIResearch/status/2049186851844771888)，并在 **FrontierMath** 上创下新高——**Tiers 1–3 达到 52%**，**Tier 4 达到 40%**，其中包括两个此前从未被任何模型解决的 Tier 4 问题。另外，[Greg Kamradt 表示 GPT-5.5 和 Opus 4.7 的 ARC-AGI-3 测试已完成](https://x.com/GregKamradt/status/2049121093307547654)，目前正在对失败模式进行分析。
- **几个新的基准测试针对更真实的 Agent 和工程行为**：[Lysandre 宣布了一项旨在使 Transformer 对 Agent 更友好的基准测试](https://x.com/LysandreJik/status/2049053056814436352)，而 [VibeBench](https://x.com/jpschroeder/status/2049139723776495800) 提议由 **1,000 名合格的软件工程师**进行主观测试，以衡量模型在实际工作中的真实感受。在文档智能方面，[LlamaIndex 的 ParseBench](https://x.com/llama_index/status/2049139409316946011) 强调，传统的 OCR 基准测试忽略了 **语义格式 (Semantic Formatting)**（如删除线和上标），而这些格式会实质性地改变 Agent 理解的含义。
- **具有具体工程影响的研究笔记**：[Rosinality 指出 DeepSpeed 和 OpenRLHF 中存在降低 SFT 性能的 Bug](https://x.com/rosinality/status/2049024030749970699)，这可能会影响之前的研究结论。[Arjun Kocher 发布了 DeepSeek-V4 论文中压缩稀疏注意力 (Compressed Sparse Attention) 的忠实实现](https://x.com/arjunkocher/status/2049066844925936041)。[che_shr_cat 展示了单块 Transformer 只有在显式使用 Scratchpad 和反转路由初始化的情况下才能解决极限数独 (Extreme Sudoku)](https://x.com/che_shr_cat/status/2049081240762876261)，否则性能为零。在优化方面，[Keller Jordan 发布了一个轻量级的 Modded-NanoGPT 优化器基准测试](https://x.com/kellerjordan0/status/2049193527440187494)，旨在可重复的竞速式任务中比较 **Muon** 和 **AdamW** 等方法。

**平台经济学、API 定价以及闭源模型可靠性问题**

- **开放模型经济学正在成为真正的驱动力**：[Aidan Gomez 认为私有化部署至关重要，因为控制模型意味着控制成本](https://x.com/aidangomez/status/2049083965407969690)，而 [Vtrivedy 主张许多 Haiku/Flash 工作负载应该针对开放模型进行重新评估](https://x.com/Vtrivedy10/status/2049201138310721616)，理由是巨大的价格差距以及来自 **DeepSeek**、**Minimax**、**GLM** 和 **Nemotron** 等系列模型不断提升的质量。DeepSeek 通过[激进的 V4 Pro 降价和缓存折扣](https://x.com/ZhihuFrontier/status/2049027925920637077)放大了这一说法，随后该优惠[延长至 5 月底](https://x.com/teortaxesTex/status/2049101287161991332)。
- **对闭源模型的依赖被定性为运营风险，而不仅仅是偏好问题**：[Gergely Orosz 总结了 Anthropic 最近的沉默变更和影响客户的行为](https://x.com/GergelyOrosz/status/2049123621826707657)，以此证明闭源模型是“巨大的风险”，而 [Zach Mueller 记录了 Claude 4.7 在其编程工作流中的退化](https://x.com/TheZachMueller/status/2049116099053031563)并最终弃用。Tokenization 经济学也受到了审查：[Aran Komatsuzaki 量化了高额的非英语 Token 税，尤其是 Anthropic](https://x.com/arankomatsuzaki/status/2049125048792006965)，随后将对比扩展到更多模型-语言对，发现 **Gemini 和 Qwen** 是对非英语文本惩罚最小的模型。
- **热门推文（按互动量排序，已过滤技术相关性）**  
  - **Codex 使用范围扩大**：[OpenAI 团队暂时重置了所有付费方案的 Codex 速率限制](https://x.com/thsottiaux/status/2048997818673537399)，以鼓励更多基于 GPT-5.5 的构建。  
  - **Claude 停机 / 集中度风险**：[Yuchen Jin 关于 Claude Code 宕机以及“整个硅谷”反应的笑话](https://x.com/Yuchenj_UW/status/2049201297656786999)捕捉到了编程 Agent 在日常工作流中已变得多么核心。  
  - **OpenAI 谈 AI 辅助数学**：[OpenAI 推广了一期关于 GPT-5.4 Pro 帮助解决 60 年前 Erdős 问题的播客](https://x.com/OpenAI/status/2049182118069358967)，这是前沿模型在正式研究中作用日益增强的显著案例。  
  - **GPT-5.5 采用信号**：[Sam Altman 注意到用户对 5.5 的强烈热情](https://x.com/sama/status/2049235284655780000)，而 [Epoch 的 ECI 帖子](https://x.com/EpochAIResearch/status/2049186851844771888)则为这种情绪提供了更硬核的 Benchmark 信号。

**AI 治理与国防：Google 的五角大楼协议引发强烈的内部抵制**

- **最具争议的政策事件是 Google 的机密五角大楼 AI 协议**：[Kimmonismus 总结了有关 Google 签署协议允许将其 AI 用于机密工作和“任何合法政府用途”的报道](https://x.com/kimmonismus/status/2049081961222955403)，据报道，合同条款允许政府要求修改安全过滤器，而对监控或自主武器仅提供非约束性的“不打算用于”限制。这引发了 Google/DeepMind 内部极不寻常的公开批评，包括 [BlackHC 称其“令人羞耻”](https://x.com/BlackHC/status/2049086569718636565)，并表示[事先没有任何内部公告或讨论](https://x.com/BlackHC/status/2049086660638543967)。
- **这一反应之所以重要，是因为它强化了前沿实验室“红线”之间的区别**：[S. Ó hÉigeartaigh 主张应以适用于 OpenAI 的相同标准来审查 Google DeepMind](https://x.com/S_OhEigeartaigh/status/2049169065109840069)，而 [TurnTrout 表示 Google 的条款比 OpenAI 遮羞布式的限制还要弱](https://x.com/Turn_Trout/status/2049153749743264231)。该事件还强化了 Anthropic 在公众辩论中截然不同的立场，因为早先的报道表明，其拒绝放弃某些红线导致了采购摩擦。对于工程师来说，实际的教训与其说是政治，不如说是平台治理：**安全政策、部署控制和合同语言正日益成为前沿 AI 供应商产品表面（product surface）的一部分**。


---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. Qwen 3.6 模型基准测试与性能

- **[[Qwen 3.6 27B BF16 vs Q4_K_M vs Q8_0 GGUF 评估](https://www.reddit.com/r/LocalLLaMA/comments/1sxzqry/qwen_36_27b_bf16_vs_q4_k_m_vs_q8_0_gguf_evaluation/)]** (活跃度: 731): **该图片提供了 Qwen 3.6 27B 模型在三种量化变体（BF16、Q4_K_M 和 Q8_0 GGUF）下的基准测试对比，评估使用了 llama-cpp-python 和 Neo AI Engineer。基准测试包括用于代码生成的 HumanEval、用于常识推理的 HellaSwag 以及用于函数调用的 BFCL。Q4_K_M 变体因其显著的实用性而脱颖而出，其吞吐量比 BF16 快 1.45 倍，峰值 RAM 占用减少了 48%，且模型体积缩小了 68.8%，同时保持了几乎一致的函数调用得分。尽管 HumanEval 准确率略有下降，但除非追求极致质量（此时首选 BF16），否则建议在本地/CPU 部署中使用 Q4_K_M。** 评论者赞赏这种对量化变体的详细对比，尽管一些人对缺乏误差线（error bars）和潜在的采样误差表示担忧，特别是针对 Q8_0 模型的表现。人们对将这些评估扩展到其他模型或尺寸表现出兴趣，并请求提供所使用的完整代码，因为一些人怀疑 Q8_0 的结果存在潜在问题，例如可能的 KV cache 量化。

    - audioen 对 Qwen 3.6 27B BF16、Q4_K_M 和 Q8_0 GGUF 模型评估中缺乏误差线表示担忧。他们认为 Q4_K_M 优于 Q8_0 的意外排序可能是由于采样误差造成的，并强调了基准测试过程中统计严谨性的重要性。
    - spaceman_ 和 Look_0ver_There 对 Q8_0 模型的性能表示怀疑，怀疑 KV cache 的量化可能影响了结果。spaceman_ 请求提供用于评估的完整代码，以验证 KV cache 是否被量化，因为这可以解释意料之外的性能下降。
    - One_Key_8127 指出了 Qwen 3.6 27B 报告的 HumanEval 分数存在差异，指出根据与 Gemma 3 4B 和 Llama3-8b 等其他模型的比较，其得分应该显著更高。他们引用外部基准测试来支持其观点，即目前的结果可能不准确。

  - **[[Luce DFlash：单张 RTX 3090 上 Qwen3.6-27B 吞吐量提升高达 2 倍](https://www.reddit.com/r/LocalLLaMA/comments/1sx8uok/luce_dflash_qwen3627b_at_up_to_2x_throughput_on_a/)]** (活跃度: 982): ****Luce DFlash** 是针对 Qwen3.6-27B 模型推测解码（speculative decoding）的一种新实现，经过优化，可以使用构建在 `ggml` 之上的独立 C++/CUDA 栈在单张 RTX 3090 GPU 上运行。在 HumanEval、GSM8K 和 Math500 等基准测试中，该设置与自回归解码相比实现了高达 `1.98x` 的吞吐量，且无需重新训练。该系统采用了 DDTree 树验证推测解码、KV cache 压缩以及滑动窗口 Flash Attention 等先进技术来优化性能和内存使用，从而能够高效处理高达 256K tokens 的大上下文。** 评论者赞赏本地 AI 推理方面的创新，并指出吞吐量大幅提升的潜力。然而，人们也对量化对准确性的影响表示担忧，特别是对于涉及编程或工具调用的用例，在这些场景中精度至关重要。

    - drrck82 强调了使用双 RTX 3090 GPU 运行 Qwen3.6-27B 的潜力，并指出相比于目前使用 Q6_K_XL 的设置，实现高达 2 倍吞吐量非常有吸引力。这表明本地 AI 推理在性能上有显著提升空间，特别是对于拥有高端硬件配置的用户。
    - Tiny_Arugula_5648 对量化对模型准确性的影响表示担忧，强调虽然提高吞吐量很有吸引力，但它可能并不适用于所有用例。他们警告说，重度量化会导致严重的准确性损失，尤其是在编程或工具调用等精度至关重要的任务中。
    - Deep90 表示需要集中式的基准测试资源，以便在日益增多的 AI 模型选项中进行选择。这反映了社区在根据性能指标评估和选择模型时面临的更广泛挑战，而这些指标对于 AI 部署中的明智决策至关重要。

- **[致 16GB VRAM 用户，插上你的旧 GPU 吧](https://www.reddit.com/r/LocalLLaMA/comments/1swzjnu/to_16gb_vram_users_plug_in_your_old_gpu/)** (热度: 797): **该帖子讨论了如何利用至少具有 `6GB VRAM` 的旧 GPU 与主 `16GB VRAM` GPU 协同工作，通过 `llama-server` 运行类似 `Qwen3.6-27B` 的稠密模型。该方案涉及使用一块 `5070Ti` 和一块 `2060` 来实现总计 `22GB VRAM`，性能接近 `24GB` 级别的显卡。配置包括 `dev=Vulkan1,Vulkan2` 以启用双 GPU，`no-mmap` 保持模型不进入 RAM，以及 `n-gpu-layers=999` 以实现最大化的 GPU offloading。基准测试显示速度显著提升，在 `128k` 最大上下文下，Prompt 处理速度为 `186 tokens/s`，生成速度为 `19 tokens/s`，而单卡仅为 `4 tokens/s`。** 评论区争论了使用 `Vulkan` 而非 `CUDA` 的优劣，部分用户建议为了更好的性能应使用 `CUDA`。其他人则指出，虽然备用 GPU 提供的额外 VRAM 可以提升性能，但也可能拖累主 GPU，正如在 `3090 Ti` 和 `2070` 组合配置中看到的那样。

    - **Mysterious_Role_8852** 讨论了同时使用 3090 Ti 和 2070 时的性能瓶颈。他们注意到 2070 显著拖慢了 3090 Ti，在两个 GPU 之间分配任务时，导致速度从 `30t/s` 下降到 `20t/s`。这凸显了匹配 GPU 性能以避免性能下降的重要性，尤其是在处理像 Qwen 3.6 27b Q6 Quant 这样的大型模型时。
    - **mac1e2** 详细介绍了在配置受限的系统（GTX 1650 4GB 和 62GB RAM）上运行 Qwen3.6-35B-A3B 的情况。他们强调了理解硬件限制并优化配置的重要性，例如使用 `--cpu-moe`、`--mlock` 以及特定的 cache 设置，从而达到约 `20-21 tok/s` 的速度。这展示了通过严谨的资源管理，在旧硬件上仍能取得有效结果。
    - **jacek2023** 提到在三块 3090 之外，还将一块 3060 作为额外的 GPU 使用，但仅限于处理超大型模型。这表明了一种有策略地利用可用硬件资源的方法，即有选择地使用额外 GPU 来最大化高负载任务的性能，而非在所有工作负载中统一使用。


### 2. 新模型与工具发布

  - **[明天 Mistral (Vibe) 会有新动态](https://www.reddit.com/r/LocalLLaMA/comments/1sy6xoo/something_from_mistral_vibe_tomorrow/)** (热度: 312): **图片是一条来自 "Mistral Vibe" 的社交媒体帖子，预告了定于次日发布的重大公告。该帖子获得了中等程度的关注，表明人们对公告充满期待。评论区推测了潜在的进展，如新模型发布或工具升级，一些用户希望性能有所提升以匹配 `Qwen 3.6 27B` 等行业标准。还有关于潜在军事合同的推测，这可能会影响公司对尖端 (SOTA) 技术的专注度。** 评论者对 Mistral 目前的产品表示怀疑，一位用户称现有模型表现“平平 (meh)”，并期待改进。另一条评论建议军事合同可能会推迟尖端技术的进步。

    - **LegacyRemaster** 提到了一项基准测试得分，指出 'Devstral SWE Bench 81.00+'，这表明该模型在特定领域具有极高性能。这预示着该模型在特定的技术基准测试中可能具有竞争力，有望对标行业标准。
    - **new__vision** 澄清说，公告可能不是关于新模型的，而是与 Mistral Vibe 的 X 账号相关，该账号与 Mistral AI 的 X 账号是分开的。他们建议 Vibe 是一个能与本地模型良好集成的“Coding Agent”，暗示公告可能与“编码框架 (coding harness)”有关，而非发布新模型。
    - **AvidCyclist250** 推测可能会有另一份军事合同，这可能会推迟尖端 (SOTA) 技术的研发。这意味着向军事项目分配资源可能会影响发布最前沿模型的时间表。

- **[Deepseek Vision 即将到来](https://www.reddit.com/r/LocalLLaMA/comments/1sxy0o7/deepseek_vision_coming/)** (活跃度: 318): ****Deepseek Vision** 预计很快就会发布，正如 **Xiaokang Chen** 在 [𝕏](https://x.com/PKUCXK/status/2049066514284962040) 上的帖子所指出的。Deepseek Vision 的基础设施已基本就绪，base models 已经开发完成，这表明 multimodality 的集成将紧随 pretraining 阶段之后。鉴于 Deepseek V4 大约在 `2-3 周前` 部署，这意味着 Deepseek V4 预览版与正式发布之间的间隔可能很短。** 评论者表达了对统一模型的偏好，例如具有 native multimodality 的 V4.1，而不是独立的视觉专用模型，并强调了集成 multimodal 能力的重要性。

    - Few_Painter_5588 讨论了 Deepseek Vision 的基础设施准备情况，指出 base models 已经到位，这简化了 multimodality 的集成。他们认为，考虑到 V4 在 2-3 周前部署，从 Deepseek V4-preview 到完整 V4 版本的过渡可能会很迅速，这表明视觉能力的开发周期可能很短。
    - dampflokfreund 表达了对统一模型方案的偏好，希望发布包含 native multimodality 的 V4.1，而不是独立的视觉专用模型。这反映了用户对能够无缝处理多种数据类型的集成解决方案的普遍渴望，强调了 native multimodality 在现代 AI 系统中的重要性。

  - **[微软发布 "TRELLIS.2": 一个开源的 4b 参数 Image-To-3D 模型，可生成高达 1536³ 的 PBR 贴图资产，基于具有 16× 空间压缩的 Native 3D VAES 构建，提供高效、可扩展、高保真的资产生成。](https://www.reddit.com/r/LocalLLaMA/comments/1sxf2u0/microsoft_presents_trellis2_an_opensource/)** (活跃度: 786): ****Microsoft** 推出了 "TRELLIS.2"，这是一个拥有 4b 参数的前沿模型，用于从图像生成高保真 3D 资产。该模型采用了一种名为 O-Voxel 的新型 "field-free" 稀疏体素结构，能够重建具有锐利特征和完整 PBR 材质的复杂 3D 拓扑。它通过 `16×` 空间压缩实现了高效且可扩展的资产生成，生成的资产分辨率高达 `1536³`。该模型是开源的，资源可在 [GitHub](https://github.com/microsoft/TRELLIS.2) 上获得，并在 [Hugging Face](https://huggingface.co/spaces/microsoft/TRELLIS.2) 上提供实时 Demo。** 一些用户指出 TRELLIS.2 在几个月前就已经发布并被讨论过，这表明该公告对某些人来说可能并不新鲜。然而，对于社区的大部分人来说，这似乎仍是新闻，证明了持续关注的价值。

    - TRELLIS.2 模型虽然在四个月前就已发布，但对社区中的许多人来说似乎是新消息，表明在最初发布时缺乏广泛的认知或报道。
    - 一位用户尝试使用 ROCm 在 AMD 7800XT GPU 上运行 TRELLIS.2，但遇到了 segmentation faults。该模型主要在具有 24GB VRAM 的 NVIDIA GPU 上进行了测试，暗示可能存在与 AMD 硬件和 ROCm 依赖项的兼容性问题。
    - 最近有一个旨在添加 ROCm 支持的 pull request 被批准，但由于依赖项问题和需要 gated repository 访问权限，用户仍面临困难。尽管存在这些挑战，该模型已可以处理图像并开始资产创建，表明其具有部分功能。


### 3. Local LLM 使用与挑战

  - **[我受够了用本地 LLMs 进行编程](https://www.reddit.com/r/LocalLLaMA/comments/1sxqa2c/im_done_with_using_local_llms_for_coding/)** (活跃度: 1981): **这篇 Reddit 帖子讨论了作者对 **Qwen 27B** 和 **Gemma 4 31B** 等本地 LLMs 在编程任务中的不满，特别是与工作中使用的 **Claude Code** 相比。强调的主要问题包括决策能力差和 tool-calling 失败，尤其是在 Dockerization 等任务中，模型无法遵循逻辑步骤或有效处理长期运行的进程。作者还指出了性能问题，例如响应时间慢和 prompt caches 损坏，这些都阻碍了生产力。尽管尝试通过详细指令引导模型，但本地 LLMs 未能达到预期，导致作者考虑在处理更苛刻的任务时使用 **OpenRouter** 等云端模型，而将本地模型留给更简单的自动化和语言任务。** 评论者建议，harness 的选择会显著影响性能，某些 harness（如 **Hermes**）可能会提供更好的长期运行进程处理能力。此外，关于一些社区帖子设定的不切实际期望也存在争论，这些帖子可能夸大了使用本地 LLMs 获得成功结果的难易程度。

- 提出的一个关键技术点是优化像 Claude Code 这样的本地模型设置以提高性能的重要性。一位用户分享了 [Unsloth 的文档](https://unsloth.ai/docs/basics/claude-code#fixing-90-slower-inference-in-claude-code) 链接，其中详细介绍了如何解决推理（Inference）缓慢和缓存失效等问题，这些是使用本地 LLM 进行编码任务时的常见困扰。
- 另一个富有洞察力的讨论围绕着与本地模型配套使用的 Harness 的重要性。一位评论者指出，即使使用相同的模型，不同的 Harness 也会导致截然不同的结果，并强调 Harness 的选择至关重要。他们提到，某些 Harness（如 Hermes Agent）具有特定的优缺点，例如处理长时间运行的进程和有效利用日志文件输出，这会影响本地模型的感知性能。
- 辩论还涉及本地模型与中心化供应商相比的成本效益。虽然消费级 GPU 上的本地模型在性能上可能无法与 Claude 等模型相提并论，但它们可以节省成本。例如，Kimi K2.6 模型被强调为 Claude Opus 的经济型替代方案，在较低的 API 成本下提供类似的性能。这表明，虽然性能可能滞后，但本地模型在某些用例中在经济上仍然是可行的。

- **[Duality of r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1sxs71y/duality_of_rlocalllama/)** (Activity: 575): **该图像突显了 r/LocalLLaMA 社区内部关于在编码中使用本地大语言模型 (LLM) 的截然不同的观点。一个帖子表达了在尝试数周本地 LLM 后的沮丧和不满，而另一个帖子则持乐观态度，认为本地模型已经可以胜任实际工作，Terminal-Bench 2.0 的测试结果证明了这一点。这种二元性反映了用户在部署本地 LLM 时各种不同的经历和预期，通常受模型大小和用户优化工作流能力的影响。** 评论者讨论了一些用户在将本地 LLM 与运行在昂贵硬件上的大规模模型进行比较时所持有的不切实际的期望。他们强调了理解较小模型（如 270 亿参数的 Qwen 3.6）局限性的重要性，以及需要高效的工作流架构来最大化其潜力。

    - Memexp-over9000 讨论了使用 Qwen 3.6 27B 模型的局限性和潜力，强调虽然它无法与万亿参数模型竞争，但如果工作流架构设计得当，它可以产生相当的输出。该评论强调了将 AI 用于“繁琐工作 (grunt work)”而非创意任务的重要性，并建议理解和优化模型的能力对于有效使用至关重要。
    - FoxiPanda 对影响本地模型性能的变量进行了详细分析，如 Harness 配置、系统提示词 (System Prompts) 和模型量化 (Quantization)。他们指出，不同的模型在系统提示词中需要特定的“粘合剂”来解决其独特的问题，而且量化水平（例如 IQ2 vs. Q8）会显著影响用户体验。评论强调了结构良好的提示词和规划在实现本地模型最佳效果方面的重要性。
    - Scared-Tip7914 强调了模型量化对性能的影响，指出用户在讨论模型能力时经常未能指明量化级别（例如 Q2 vs. Q8）。他们分享了使用 Qwen 3.5-35B Q4 的经验，强调其作用是作为大型闭源模型的补充，以实现 Token 高效执行，而不是作为独立解决方案。该评论建议采取一种战略性方法：用大模型进行规划，用本地模型进行执行。

- **[A warning to newbies - A lesson on network security](https://www.reddit.com/r/LocalLLM/comments/1sxhlmv/a_warning_to_newbies_a_lesson_on_network_security/)** (Activity: 355): **该帖子强调了一个严重的网络安全问题，即 373 台设备在不需要 API Key 的情况下公开暴露了 LM Studio 实例，使其容易受到未经授权的访问。图片显示了一张世界地图，相关国家被涂成红色以表示暴露设备的数量，其中泰国的数量最高，达到 194 台。作者强调了保护 LLM 平台安全的重要性，不要在没有适当安全措施（如使用 Tailscale 或带有身份验证的反向代理）的情况下将其暴露在互联网上。** 一位评论者赞赏这种伦理黑客行为，而另一位评论者则指出了在暴露设备上远程执行 Prompt 的能力，并强调了潜在风险。第三条评论则讽刺地建议利用这些未加密的设备来获取算力资源。

- DatMemeKing 强调了一个关键的安全漏洞，他们能够远程在设备上执行 Prompt，这表明网络配置或软件中可能存在允许未经授权访问的潜在缺陷。这凸显了保护网络端口安全以及确保远程执行功能受到严格控制和监控的重要性。
- AdultContemporaneous 提出了一个问题，即该安全问题是涉及在有互联网访问权限的计算机上运行本地 LLM，还是专门影响那些试图远程访问其本地托管 LLM 的用户。他们提到使用带有地理 IP 封锁功能的 IDS/IPS，表明他们采取了分层安全方法来防止未经授权的访问。
- Illeazar 澄清说，安全风险主要针对那些选择在路由器上公开进行端口转发的用户，这可能会使他们的系统暴露在外部威胁之下。这强调了谨慎进行网络配置的必要性，以及在没有适当安全措施的情况下将本地服务暴露给互联网所带来的风险。

## 较少技术性的 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude 和 Opus 模型定价及访问问题

  - **[Anthropic 悄悄地为 Claude Code 中的 Pro 用户将 Opus 锁在了“墙中墙”付费墙之后](https://www.reddit.com/r/ClaudeAI/comments/1sxi9mo/anthropic_just_quietly_locked_opus_behind_a/)** (活跃度: 1053): **Anthropic** 为其 **Claude Code** 用户引入了新的定价结构，即使是 Pro 计划的用户也需要额外付费才能访问 **Opus 模型**。这一变化在其支持文档中被悄悄提及，表明虽然 Pro 计划每月花费 `$20`，但访问旗舰级 Opus 模型需要进一步购买。当前默认可用的模型是 **Sonnet 4.5**，虽然列出了 Opus 4.5，但它被锁定在额外的付费墙之后。此举暗示了向计费模式的转变，引发了用户对透明度和成本影响的担忧。一些用户对缺乏透明度和额外费用表示沮丧，评论中流露出不满，并期待诸如 **Qwen 4 27b** 等替代模型。其他用户指出他们已经在运行 Opus 4.7，这表明推广过程中可能存在不一致或处于有限测试阶段。

    - ClaudeOfficial 澄清说，关于 Opus 被锁定在付费墙之后的信息已过时。他们提到 Opus 4.5 已于 1 月份面向 Pro 计划推出，而支持文章一直没有更新。他们提供了一个 Wayback Machine 的链接来验证这些信息，表明文档更新存在滞后。

  - **[GitHub Copilot 对 Claude 模型的定价提高了 9 倍](https://www.reddit.com/r/ClaudeAI/comments/1sxcxge/github_copilot_9x_price_increase_for_claude_models/)** (活跃度: 803): **GitHub Copilot** 将从 6 月开始对 Claude 模型实施 `900%` 的调价，从固定计划过渡到基于使用量的计费方式。这一变化的详情见 [GitHub 文档](https://docs.github.com/en/copilot/reference/copilot-billing/models-and-pricing#model-multipliers-for-annual-copilot-pro-and-copilot-pro-subscribers) 及其 [新闻稿](https://github.blog/news-insights/company-news/github-copilot-is-moving-to-usage-based-billing/)。此次涨价是向基于 API 计费转变的一部分，这可能会显著影响依赖 Claude Agent 进行生产的企业客户，因为增加的推理成本可能会严重影响单位经济效益。评论者对 Agent 运行和 Token 使用情况缺乏可见性表示担忧，这可能会加剧调价带来的财务影响。这一转变被视为 **Anthropic** 利用其市场地位采取的战略举措。

- Emerald-Bedrock44 强调了 Claude 模型价格上涨 9 倍的关键问题，并指出这种剧烈变化会严重影响在生产环境中使用这些模型的公司的单位经济效益 (unit economics)。由于缺乏对 Token 使用情况的可见性，这一问题变得更加严重，因为团队可能无法完全理解或控制其推理成本 (inference costs)，从而导致潜在的财务压力。
- CricktyDickty 指出，从固定的补贴方案向基于 API 的定价模式转变代表了 Anthropic 的一项战略举措。这种转变可以被视为 Anthropic 巩固其市场地位的一种方式，虽然可能增加收入，但也给之前受益于较低固定成本的企业客户带来了沉重的财务负担。
- dotheemptyhouse 指出，价格上涨并非 Claude 模型所特有，并提到一些竞争模型的价格也上涨了 6 倍。这表明 AI 模型供应商之间存在成本上升的广泛趋势，可能预示着行业定价策略的转变，或者是对需求增加和运营成本上升的反应。

- **[Anthropic 悄悄地为 Claude Code 中的 Pro 用户将 Opus 锁在了“墙中墙”的付费门槛之后](https://www.reddit.com/r/ClaudeCode/comments/1sxi7uh/anthropic_just_quietly_locked_opus_behind_a/)** (Activity: 653): **该图片强调了 **Anthropic** 一项备受争议的变动，即 Claude Code 套件中的 Opus 模型现在对 Pro 用户设置了额外的付费门槛。这意味着即使是每月支付 20 美元 Pro 订阅费用的用户，也必须额外付费才能访问 Opus 模型，尽管它被宣传为 Pro 套餐的一部分。目前可用的默认模型是 Sonnet 4.5，虽然列出了 Opus 4.5，但需要额外购买。这一变化并未广泛公布，仅在支持文章中提到，导致用户对透明度和成本影响感到沮丧。** 一条评论指出，该支持文章已过时，并提到了 1 月份推出的 Opus 4.5，这表明 Anthropic 缺乏及时的沟通。另一条评论批评了 Opus 模型的高 Token 使用率，这会迅速耗尽用户的配额，认为额外的成本可能并不合理。

    - **ClaudeOfficial** 澄清说，该支持文章已经过时，且根据 [Wayback Machine](https://web.archive.org/web/20251204151142/https://support.claude.com/en/articles/11940350-claude-code-model-configuration) 的记录，Opus 4.5 自 1 月份以来就已包含在 Pro 计划中。这表明这是一个沟通失误，而非蓄意的付费门槛变更。
    - **Faangdevmanager** 强调了 Opus 的 Token 消耗存在重大问题，指出它消耗了大量昂贵的 Token。对于那些发现配额被迅速耗尽的用户来说，这是一个痛点，表明需要更高效的 Token 使用方式或更清晰的成本沟通。
    - **Academic-Proof3700** 对 Pro 订阅的多个问题表达了挫败感，包括质量下降、Bug 以及引入了在没有提供比例价值的情况下消耗更多 Token 的新模型。这反映了用户对服务性价比和透明度的普遍不满。

### 2. GPT 5.4 与 5.5 的性能与基准测试

  - **[GPT 5.4 与 GPT 5.5 在 MineBench 上的差异](https://www.reddit.com/r/singularity/comments/1sxapqb/differences_between_gpt_54_and_gpt_55_on_minebench/)** (活跃度: 465): **该帖子讨论了使用 MineBench 框架对 **GPT 5.4** 和 **GPT 5.5** 进行的基准测试，强调 **GPT 5.5** 相比 **GPT 5.4** 表现出微小的提升。基准测试表明，**GPT 5.5** 以更少的计算资源实现了相似的输出质量，这与 **OpenAI** 关于效率提升的声明一致。运行 **GPT 5.5** 的成本为 `$19.98`，平均推理时间为 `624 seconds`，而 **GPT 5.4** 的成本约为 `$25`。5.5 系列中 Pro 版与标准版之间的差异极小，表明输出质量相近。该基准测试涉及利用方块色板创建 3D 结构，**GPT 5.5** 展示了更详细且复杂的设计。**评论者指出 **GPT 5.5** 的输出细节令人印象深刻，例如建模宇航员面罩上的反光，尽管某些构建由于随机颜色的方块而显得噪点较多。总体而言，**GPT 5.5** 的设计被认为稍胜一筹。

    - WithoutReason1729 强调了 GPT 5.5 在视觉建模能力上的显著提升，指出它可以准确地模拟复杂的反光，例如宇航员面罩上的地球。这表明新版本在渲染和空间推理能力方面有所增强。
    - Kamimashita 观察到 GPT 5.5 的构建中随机颜色的方块带来了更多噪点，但整体设计质量有所提高。这表明在细节和噪点之间存在权衡，暗示 GPT 5.5 可能正在尝试更复杂的设计模式。
    - FateOfMuffins 讨论了从 GPT 5.4 到 5.5 显著的 `270 ELO` 增长，以及到 5.5 Pro 又有 `220 ELO` 的额外跃升。ELO 评分的这种大幅提升反映了性能的增强以及新版本中可能使用了更复杂的算法，尽管这也引发了关于基准测试饱和度以及增加难度必要性的讨论。

  - **[GPT 5.5 在 token 使用上极其浪费](https://www.reddit.com/r/CLine/comments/1sxc6wr/gpt_55_is_unbelievably_wasteful_with_tokens/)** (活跃度: 14): **该帖子讨论了使用 **GPT 5.5** 带来的高 token 消耗及相关成本，特别是在 Codex 应用场景之外，据报道单次请求的成本达 `$5`。这突显了人们对该模型在非编程背景下的效率和成本效益的担忧。** 一条评论建议，使用 **GPT 5.5** 或 **Claude Opus 4.7:1m xhigh** 等模型的成本应相对于其提供的价值来考虑，暗示在某些应用中，高昂的成本可能是由其带来的收益所支撑的。

### 3. ChatGPT 解决数学问题

  - **[Chat GPT 5.4 一举解决了悬而未决 60 多年的 Erdős 问题](https://www.reddit.com/r/singularity/comments/1sxixck/chat_gpt_54_solved_a_60_years_unsolved_erdos/)** (Activity: 2265): **图片展示了一个关于未解决的 Erdős 问题的数学证明，重点是涉及本原集（primitive sets）总和的不等式。帖子声称 **Chat GPT 5.4** 在 `80 minutes and 17 seconds` 内解决了这个问题，这表明 AI 在推理复杂数学问题方面的能力有了显著提升。该证明涉及常数和对数表达式，体现了通常在 PhD 水平才具备的高水平数学推理。这一进展挑战了 LLMs 仅仅是预测下一个 Token 而不具备真正推理能力的观点。** 虽然这一成就令人印象深刻，但一些评论者认为，声称其推理能力优于 50 年来的数学家是一种夸大其词。他们承认 LLMs 是数学家强大的工具，但也指出这些模型仍然存在局限性，目前还不能独立产生新颖的想法。

    - **enilea** 指出，虽然 Erdős 问题非常多，且许多问题因缺乏关注而仍未解决，但声称 LLM 的“推理能力优于 50 年来的数学家”是一种夸大。他们承认 LLMs 正在成为数学家的强大工具，但强调这些模型目前仍有局限性，尚不能独立产生新颖的想法。

  - **[ChatGPT 5.4 解决了一个 64 年历史的数学难题](https://www.reddit.com/r/ChatGPT/comments/1swn1bs/chatgpt_54_solved_a_64yearold_math_problem/)** (Activity: 13896): **图片展示了一个与 Erdős 问题相关的数学证明，特别关注本原集和对数不等式。帖子声称一位 23 岁的用户使用 **ChatGPT 5.4 Pro** 在大约 `1 hour 20 minutes` 内解决了这个有着 64 年历史的问题。据报道，该解决方案以一种前所未有的新颖方式将已知公式应用于此问题。所涉及的问题实际上是 Erdős 1196 而非 1176，且该证明已被证实是合法的，著名数学家 **Tao** 也对其进行了评论。AI 的成功归功于用户引导它采用了与以往专家尝试的部分解法不同的方法。** 评论者强调了这一成就的重要性，指出 AI 的解法既简短又优雅。他们强调了提出正确问题的重要性，因为该用户没有遵循专家的部分解法，而是使用了一种熟悉的方法，从而实现了突破。

    - **EmergencyFun9106** 强调解决的问题是 Erdős 1196 而非 1176，并引用了 Terence Tao 在[这里](https://www.erdosproblems.com/1196)的评论确认了证明的合法性。其意义在于该问题的部分解法历史悠久，而 AI 的贡献尤为简洁且优雅。
    - **yubario** 解释了 AI 的成功，并将其与之前遵循专家部分解法但陷入死胡同的尝试进行了对比。突破点在于用不同的方法引导 AI，强调了提出正确问题以开启解决方案的重要性。
    - **MannOfSandd** 预见学术界将会有强烈反应，表明 AI 的解决方案将面临来自院系和研究人员的潜在影响和审查。



# AI Discord 频道

不幸的是，Discord 今天关闭了我们的访问权限。我们不会以这种形式恢复它，但我们很快就会发布全新的 AINews。感谢阅读到这里，这是一段美好的历程。