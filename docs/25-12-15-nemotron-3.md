---
companies:
- nvidia
- huggingface
- togethercompute
- baseten
- vllm
- llamaindex
date: '2025-12-15T05:44:39.731046Z'
description: '**英伟达（NVIDIA）**发布了 **Nemotron 3 Nano**，这是一款完全开源的混合 Mamba-Transformer
  混合专家（MoE）模型，拥有 **300 亿（30B）参数规模**和 **100 万 token 的上下文窗口**。


  此次发布包含了开放权重、训练配方、数据集以及名为 NeMo Gym 的强化学习（RL）环境套件，并根据 NVIDIA Open Model License 协议支持商业用途。该模型在
  SWE-Bench 和 Artificial Analysis Intelligence Index 等基准测试中取得了最先进（SOTA）的成绩，表现优于 **Qwen3-30B
  A3B**。


  生态系统支持已同步到位，集成了 **vLLM**、**llama.cpp** 和 **Baseten** 等推理栈。即将推出的更大规模模型 Nemotron Super
  和 Ultra 将采用 NVFP4 预训练和 LatentMoE 路由技术，以优化计算效率。此次发布凭借其全面的开放资产和先进的混合架构，标志着开源美国 AI
  领域的一个重要里程碑。'
id: MjAyNS0x
models:
- nemotron-3-nano
- qwen3-30b-a3b-base
people:
- ctnzr
- andrew_n_carr
- awnihannun
title: NVIDIA Nemotron 3：参数量从 30B 到 500B 的完全开源混合 Mamba-Transformer 模型。
topics:
- hybrid-architecture
- mixture-of-experts
- reinforcement-learning
- long-context
- model-release
- open-source-models
- model-training
- model-optimization
- benchmarking
- agent-training
---

**开源美国 AI 的好日子。**

> 2025/12/12-2025/12/15 的 AI 新闻。我们为您检查了 12 个 subreddits，544 个 Twitter 和 24 个 Discord（206 个频道，15997 条消息）。预计节省阅读时间（以 200wpm 计算）：1294 分钟。我们的新网站现已上线，提供完整的元数据搜索和精美的 vibe coded 呈现。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上给我们反馈！

Nvidia 的 [Nemotron](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-White-Paper.pdf) 虽然不常出现在 [顶级开源模型](https://x.com/natolambert/status/2000299636863734026) 之列，但其独特之处在于完全开源，即：“**我们将公开模型权重、预训练和后训练软件、配方以及我们拥有再分发权的所有数据。**” ([Nemotron 3 论文](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-White-Paper.pdf))，且源自美国。Nano 3 与 Qwen3 具有竞争力：


![Comparison of Qwen3-30B-A3B-Base and Nemotron 3 Nano 30B-](https://resend-attachments.s3.amazonaws.com/tTSdW0B18duTGpl)


当这些模型发布时，它们实际上成为了 LLM 训练领域最前沿技术的基准，因为它们基本上汇集了所有已知有效的核心要素。在值得注意的选择中——支持长文本（1m context）的混合架构：

**混合状态空间模型 (Hybrid State Space Model) + Transformer 架构**


![Nemotron 3 model architecture visualization showing interleaved Mamba-2 and MoE layers with select self-attention](https://resend-attachments.s3.amazonaws.com/2TGOzmHIf8gpc16)



![Technical architecture diagram showing details of the Nemotron 3 Nano hybrid Mamba-Transformer Mixture-of-Experts](https://resend-attachments.s3.amazonaws.com/1VcBUGGINuh4gvN)


**多环境强化学习 (Multi environment RL)** (Nemo-Gym 和 Nemo-RL 已开源)


![A technical document page describing the post-training methodology for the Nemotron 3 Nano AI model, highlighting its hybrid architecture, multi](https://resend-attachments.s3.amazonaws.com/Y8r5D8BJVyndgOo)



![A technical document page describing the infrastructure for NeMo Gym, a framework for reinforcement learning environments with three core server types: agents, models](https://resend-attachments.s3.amazonaws.com/1yhPYsF5TID3Kem)


根据 [Nano 3 技术报告](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf)，他们将发布所有数据集：


![Diagram of Nemotron 3 Nano layer architecture showing a hybrid Mamba-Transformer Mixture of Experts (M](https://resend-attachments.s3.amazonaws.com/CyZ0TOla4n64Xbu)


---

# AI Twitter 综述

**NVIDIA 的 Nemotron 3：开源混合 MoE 模型、数据和 Agent 栈**

- **Nemotron 3 Nano (总参数 30B，激活参数约 3.6B)**：NVIDIA 发布了一个完全开源的混合 Mamba–Transformer MoE 模型，具有 1M token 上下文窗口。它在 SWE-Bench 上取得了同类小模型中的最佳结果，并在广泛的评估中获得了高分（例如，在 Artificial Analysis Intelligence Index 上获得 52 分；比 Qwen3-30B A3B 高出 6 分），且吞吐量极高（例如，在 DeepInfra 上约为 380 tok/s）。开源资产包括权重、训练配方、可再分发的预训练/后训练数据集，以及用于 Agent 训练的 RL 环境套件 (NeMo Gym)。根据 NVIDIA Open Model License 允许商业使用。Super（约 100–120B）和 Ultra（约 400–500B）版本“即将推出”，其特点是采用 NVFP4 预训练和在低维潜空间中进行“LatentMoE”路由，以减少 all-to-all 通信和专家计算负载。公告与技术细节：[@ctnzr](https://twitter.com/ctnzr/status/2000567572065091791), [@nvidianewsroom](https://twitter.com/nvidianewsroom/status/2000588337896198481), [研究页面](https://twitter.com/iScienceLuvr/status/2000570258655191137)。
- **Day-0 生态系统支持**：各大推理栈和供应商立即实现了集成：
    - 推理：[vLLM](https://twitter.com/vllm_project/status/2000623058076492276), [SGLang](https://twitter.com/lmsysorg/status/2000567938949243111), [llama.cpp](https://twitter.com/ggerganov/status/2000574990425415765), [Baseten](https://twitter.com/basetenco/status/2000582868532121688), [Together](https://twitter.com/togethercompute/status/2000572943718314392), [Unsloth (GGUF)](https://twitter.com/UnslothAI/status/2000568378407452746)。
    - 数据与评估：数学和证明的开源数据集 ([Nemotron-Math](https://twitter.com/igtmn/status/2000591849669693931), Nemotron-Math-Proofs) 以及一个 [Agent 数据集](https://twitter.com/HuggingPapers/status/2000628009049760072)。
    - 社区分析与结果：[Artificial Analysis 深度解析](https://twitter.com/ArtificialAnlys/status/2000602570092675402), [HF 集合](https://twitter.com/NielsRogge/status/2000639749514760465)，以及速度/质量印象 ([@andrew_n_carr](https://twitter.com/andrew_n_carr/status/2000630563015905608), [@awnihannun](https://twitter.com/awnihannun/status/2000718403380691417))。
- **为什么这很重要**：这是迄今为止最完整的开源发布之一——涵盖了新架构（混合 SSM/MoE）、透明的训练流水线、开源数据和 Agent RL 环境——为可复现性和以 Agent 为核心的 R&D 树立了新标杆 ([@_lewtun](https://twitter.com/_lewtun/status/2000599470099099990), [@percyliang](https://twitter.com/percyliang/status/2000608134205985169), [@tri_dao](https://twitter.com/tri_dao/status/2000707760288092655))。注：LatentMoE 已在尚未发布的较大模型中记录 ([@Teknium](https://twitter.com/Teknium/status/2000592775725842886))，而 Nano 目前使用的是混合 MoE/Mamba 栈。

**推理、检索和代码 Agent：新技术与新结果**

- **算子式推理优于长 CoT**：Meta SI 的 Parallel-Distill-Refine (PDR) 将 LLM 视为改进算子——生成并行草案 → 蒸馏受限工作空间 → 精炼——并在固定延迟下显示出巨大增益（例如，AIME24：93.3% vs 长 CoT 的 79.4%；在相同 token 预算下，o3-mini 提升了 9.8 个百分点）。具有算子一致性 RL 的 8B 模型增加了约 5% ([@dair_ai](https://twitter.com/dair_ai/status/2000581380733030703))。
- **通过 RL 实现自适应检索策略**：RouteRAG 学习何时以及检索什么（段落 vs 图 vs 混合）。一个 7B 模型在 QA 任务中达到了 60.6 F1 分数（使用 10k 对比 170k 训练样本，超过 Search-R1 3.8 分），并在提高准确性的同时减少了约 20% 的检索轮次 ([@dair_ai](https://twitter.com/dair_ai/status/2000400449355325806))。
- **统一压缩 RAG (Apple CLaRa)**：共享的连续内存 token 同时用于检索和生成；可微 top-k 使得梯度能从生成器传导至检索器；在约 16 倍压缩率下，CLaRa-Mistral-7B 达到或超过了文本基准，并在没有相关性标签的情况下在 HotpotQA 上优于全监督检索器 ([@omarsar0](https://twitter.com/omarsar0/status/2000570838920434037))。
- **代码 Agent 作为通道优化 (DeepCode)**：蓝图蒸馏 + 有状态代码内存 + 条件 RAG + 闭环纠错在 PaperBench 上实现了 73.5% 的复现率，而 o1 为 43.3%，并在子集上超过了人类博士（约 76%）。开源框架 ([@omarsar0](https://twitter.com/omarsar0/status/2000385348413850055))。
- **Together RARO (无验证器 RL)**：在验证器稀缺时，通过对抗博弈训练实现可扩展推理 ([@togethercompute](https://twitter.com/togethercompute/status/2000631170909057390))。

**推理与基础设施：多模态服务、量化、调度器**

- **多模态编码器解耦**：vLLM 将视觉编码器拆分为可独立扩展的服务，实现了流水线并行（pipelining）、图像嵌入（image embeddings）缓存，并减少了与文本 prefill/decode 的竞争。收益：稳定区域吞吐量提升 +5–20%；大幅降低 P99 TTFT/TPOT ([@vllm_project](https://twitter.com/vllm_project/status/2000535421642502335))。
- **FP4 详情与 NVFP4**：适用于低精度算子（kernels）的便捷 FP4 E2M1 数值列表 ([@maharshii](https://twitter.com/maharshii/status/2000475239835455750))。Nemotron 3 训练利用了 NVFP4；社区对电路中负零（negative zero）的效用表示好奇 ([@andrew_n_carr](https://twitter.com/andrew_n_carr/status/2000744793480270236))。
- **SLURM 被 NVIDIA 收购**：扩展了 NVIDIA 在技术栈上层对广泛使用的任务调度（workload scheduling）的控制（超越了 CUDA）。关于非 NVIDIA 加速器和集群可移植性的影响正在讨论中 ([@SemiAnalysis_](https://twitter.com/SemiAnalysis_/status/2000620209262985641))。

**Agent/编程工具链与评估**

- **IBM CUGA Agent**：开源企业级 Agent，可在丰富的工具集和 MCP 上编写/执行代码；支持本地运行，提供 demo/博客和 HF Space ([@mervenoyann](https://twitter.com/mervenoyann/status/2000599316121924052))。
- **安全 Agent 文件系统与文档解析**：LlamaIndex 展示了虚拟文件系统 (AgentFS) + LlamaParse + 工作流，用于构建具备人机回环（human-in-the-loop）编排能力的安全代码 Agent ([@llama_index](https://twitter.com/llama_index/status/2000612235505467824), [@jerryjliu0](https://twitter.com/jerryjliu0/status/2000677592559706396))。
- **Google MCP 仓库**：托管和开源 MCP 服务器的参考实现、示例及学习资源 ([@rseroter](https://twitter.com/rseroter/status/2000607267675410609))。
- **Qwen Code v0.5.0**：新增 VSCode 集成包、原生 TypeScript SDK、会话管理、支持 OpenAI 兼容的推理模型（DeepSeek V3.2, Kimi-K2）、工具控制、i18n 以及稳定性修复 ([@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/2000556828690624685))。
- **Agent 测试框架（Harness）讨论**：日益关注“测试框架”质量、跨框架迁移，以及提出 HarnessBench 以衡量框架泛化能力和 router 质量 ([@Vtrivedy10](https://twitter.com/Vtrivedy10/status/2000610350014607728))。

**视觉、视频与 3D 世界**

- **可灵 (Kling) VIDEO O1 更新**：支持首尾帧控制（3–10秒）以调整节奏和实现更平滑的过渡；新增 720p 模式；已在 FAL 部署，成本更低 ([@Kling_ai](https://twitter.com/Kling_ai/status/2000581619556421673), [@fal](https://twitter.com/fal/status/2000590369545744599))。
- **TurboDiffusion (清华大学机器学习小组)**：通过 SageAttention + 稀疏线性注意力 + rCM，在单块 RTX 5090 上实现 5 秒视频生成提速 100–205 倍（低至 1.8 秒）；正集成至 vLLM-Omni ([@Winterice10](https://twitter.com/Winterice10/status/2000709961370767771), [@vllm_project](https://twitter.com/vllm_project/status/2000720345872130413))。
- **Apple Sharp 单目视图合成**：在 HF 上发布了快速单目新视角合成技术 ([@_akhaliq](https://twitter.com/_akhaliq/status/2000587447680340257))。
- **Echo (SpAItial)**：前沿 3D 世界生成器，可从文本或单张图像生成一致的、米制尺度的空间表示，通过浏览器内的 3DGS 进行实时交互渲染；面向数字孪生、机器人和设计领域 ([@SpAItial_AI](https://twitter.com/SpAItial_AI/status/2000600875388027051))。

**产品信号：OpenAI, Google, Allen, Arena**

- **OpenAI**:
    - 分支对话（Branched chats）现已在 iOS/Android 上线 ([@OpenAI](https://twitter.com/OpenAI/status/2000669385317605759))。
    - Realtime API 音频快照改进了 ASR TTS 幻觉、指令遵循和工具调用 ([@OpenAIDevs](https://twitter.com/OpenAIDevs/status/2000678814628958502))。
    - GPT‑5.2：社区反应不一，但在数学/量化研究方面表现强劲 ([@gdb](https://twitter.com/gdb/status/2000687002799194246), [@htihle](https://twitter.com/htihle/status/2000571235734810805), [@lintool](https://twitter.com/lintool/status/2000368978708119958))。
- **Google**:
    - 有迹象表明即将发布开源模型，且关于 “Gemma 4” 的讨论升温；请关注 [huggingface.co/google](https://twitter.com/osanseviero/status/2000493503860892049) ([@kimmonismus](https://twitter.com/kimmonismus/status/2000537345326452790), [@testingcatalog](https://twitter.com/testingcatalog/status/2000597370707611991))。
    - Sergey Brin 在车内测试（dogfoods）Gemini Live；暗示更好的内部版 Gemini 3 Flash 即将推出；回顾了 Jeff Dean 对 TPU 的押注以及 Google 的“创始人模式（founder mode）”重启 ([1](https://twitter.com/Yuchenj_UW/status/2000430969220890877), [2](https://twitter.com/Yuchenj_UW/status/2000435232089207179), [TPU origin](https://twitter.com/Yuchenj_UW/status/2000627610561458682))。
    - Gemini Agent 向 Ultra 用户推出交易流程（例如租车） ([@GeminiApp](https://twitter.com/GeminiApp/status/2000616120106221781))。
    - Google 的 MCP 资源已发布 ([@rseroter](https://twitter.com/rseroter/status/2000607267675410609))。
- **Allen AI**: 从 Olmo 3 “字节化（byteified）”而来的 Bolmo 字节级 LM 在各项任务中达到或超过了 SOTA 子词模型；AI2 在 OLMo 的开放性方面继续保持领先 ([@allen_ai](https://twitter.com/allen_ai/status/2000616646042399047))。
- **Arena 更新**: 新增 GLM‑4.6V/-Flash 用于对抗测试；对 DeepSeek v3.2 “思考”变体在职业和能力范畴进行了剖析 ([GLM 4.6V](https://twitter.com/arena/status/2000610761371267350), [DeepSeek v3.2 深度分析](https://twitter.com/arena/status/2000637978662821942))。

**热门推文（按参与度排序，聚焦 AI）**

- **Gemini “私密想法”风波**: 一个疯传的帖子展示了 Gemini Live 的内部想法，其中包含琐碎的“报复”计划——突显了 Agent 内部独白的 UX 透明度和安全问题 ([@AISafetyMemes](https://twitter.com/AISafetyMemes/status/2000620127054598508), 6.9k)。
- **Sergey Brin 谈 Gemini 和 Jeff Dean**: 内部测试 Live，暗示 Gemini 3 Flash，以及 TPU 的起源故事；核心主题：Google 的创始人模式和深层技术押注 ([@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2000430969220890877), 3.0k; [TPUs](https://twitter.com/Yuchenj_UW/status/2000627610561458682), 1.5k)。
- **OpenAI 产品更新**: 移动端上线分支对话 ([@OpenAI](https://twitter.com/OpenAI/status/2000669385317605759), 3.6k)。
- **Google HF 页面“公告”**: 社区正密切关注模型的快速发布 ([@osanseviero](https://twitter.com/osanseviero/status/2000493503860892049), 2.0k)。
- **Nemotron 3 Nano 概览**: 开源 30B 混合 MoE，比同类模型快 2–3 倍，支持 1M 上下文，提供开源数据/配方——在基础设施和研究社区引起广泛关注 ([@AskPerplexity](https://twitter.com/AskPerplexity/status/2000589984818954719), 2.1k; [@UnslothAI](https://twitter.com/UnslothAI/status/2000568378407452746), 1.4k)。

---

# AI Reddit 总结

## /r/LocalLlama + /r/localLLM 总结

### 1. NVIDIA Nemotron 3 Nano 发布

- [**NVIDIA 发布 Nemotron 3 Nano，一款全新的 30B 混合推理模型！**](https://www.reddit.com/r/LocalLLaMA/comments/1pn8upp/nvidia_releases_nemotron_3_nano_a_new_30b_hybrid/) (热度: 909): **NVIDIA 发布了 Nemotron 3 Nano，这是一款拥有 300 亿参数的混合推理模型，属于 Nemotron 3 系列 Mixture of Experts (MoE) 模型的一部分。该模型具有** `1M 上下文窗口`**，并针对快速、准确的代码编写和 Agent 任务进行了优化，能够在** `24GB RAM 或 VRAM` **上运行。它在 SWE-Bench 等基准测试中表现出卓越的性能，据用户报告，其生成速度达到了显著的** `110 tokens/second`**。Nemotron 3 系列还包括更大的模型，如 Nemotron 3 Super 和 Nemotron 3 Ultra，专为更复杂的应用设计，参数量最高可达** `5000 亿`**。[Unsloth GGUF](https://huggingface.co/unsloth/Nemotron-3-Nano-30B-A3B-GGUF) 支持这些模型的本地微调。** 评论者强调了该模型令人印象深刻的速度和效率，指出其在本地生成速度可达 `110 tokens/second`，这对于这种规模的模型来说是前所未有的。人们对 Nemotron 3 系列中更大的模型也充满期待，特别是 Nemotron 3 Super，预计它将在多 Agent 应用中表现出色。
    - Nemotron 3 Nano 模型以其惊人的速度著称，一位用户报告在本地机器上的生成速率为每秒 110 个 token，这与他们使用过的其他模型相比是前所未有的。这突显了该模型的效率以及在高性能应用中的潜力。
    - Nemotron 3 系列包含三个具有不同参数规模和激活能力的模型。Nemotron 3 Nano 最多激活 30 亿参数以实现高效的任务处理，而 Nemotron 3 Super 和 Ultra 模型分别激活最多 100 亿和 500 亿参数，用于更复杂的应用。这种结构允许针对不同用例实现针对性的效率和可扩展性。
    - Nemotron 3 Nano 与 Qwen3 30B A3B 模型的对比显示了文件大小的差异，Nemotron 3 Nano 的动态文件大小较大，为 22.8 GB，而 Qwen3 为 17.7 GB。这表明虽然 Nemotron 3 Nano 可能提供增强的功能，但也需要更多的存储空间，这可能会影响部署考量。
- [**NVIDIA Nemotron 3 Nano 30B A3B 发布**](https://www.reddit.com/r/LocalLLaMA/comments/1pn8h5h/nvidia_nemotron_3_nano_30b_a3b_released/) (热度: 347): **NVIDIA 发布了 Nemotron 3 Nano 30B A3B，该模型采用混合 Mamba-Transformer MoE 架构，总参数量为** `31.6B`**，每个 token 激活约** `3.6B` **参数，旨在实现高吞吐量和低延迟。它拥有** `1M-token` **的上下文窗口，据称比其前身 Nemotron Nano 2 快高达** `4倍`**，比同类规模的其他模型快** `3.3倍`**。该模型完全开放，提供开放权重、数据集和训练配方，并根据 [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) 发布。它支持使用 vLLM 和 SGLang 进行无缝部署，并通过 OpenRouter 及其他服务进行集成。未来发布的版本还包括规模显著更大的 Nemotron 3 Super 和 Ultra。** 一些用户对该模型依赖合成数据表示担忧，注意到其输出中存在“恐怖谷”效应。此外，人们对针对特定硬件配置优化该模型表现出兴趣，例如在单块 3090 GPU 设置中将部分负载卸载到系统 RAM，尽管相关文档较少。
    - 提到了一个尚未合并的 `llama.cpp` Pull Request，这表明该模型的集成工作仍在进行中，并有潜在的改进空间。提供的链接 (https://github.com/ggml-org/llama.cpp/pull/18058) 显示了为增强与 NVIDIA Nemotron 3 Nano 30B A3B 的兼容性或性能而进行的积极贡献。
    - 一位用户讨论了在使用配备 128GB DDR5 的单块 NVIDIA 3090 时，将某些模型组件卸载到系统 RAM 的潜力。他们提到缺乏关于这种卸载技术的文档和性能数据，而这对于在 GPU 显存有限的设置中优化资源利用和性能至关重要。
    - 另一位用户报告称，从开发分支编译了 `llama.cpp`，并在其机器上实现了超过 `100 tokens/second` 的速度，显示出极高的性能。然而，他们指出该模型缺乏可靠性，因为它提供了错误的状态更新，且未能准确保存文档更改。这个问题可能与使用 `Q3_K_M` 量化有关，表明在速度和准确性之间存在权衡。

### 2. Google 模型发布公告

- [**Google 新模型即将到来！！！**](https://www.reddit.com/r/LocalLLaMA/comments/1pn37mw/new_google_model_incoming/) (活跃度: 1527): **该图片是 Omar Sanseviero 发布的一条推文，暗示 Google 的一个新模型可能很快就会在 Hugging Face 平台上发布。推文中包含一个指向 Hugging Face 上 Google 页面的链接，暗示用户应该将其加入书签以关注潜在的更新。这预示着 Google 模型可能会有新的发布或更新，对于在 Hugging Face 上使用机器学习模型的开发者和研究人员来说，这可能具有重要意义。** 评论者们对新模型的性质进行了推测，一些人希望它不要像 Gemma3-Math 那样，而另一些人则对可能取代现有大型模型（如 gpt-oss-120b 和 20b）的多模态模型表示出兴趣。
    - DataCraftsman 表达了对新模型的渴望，希望它能作为 `gpt-oss-120b` 和 `20b` 等现有模型的多模态替代品。这表明需要一个能够处理多种类型数据输入和输出的模型，从而有可能提升这些现有模型的能力。
    - Few_Painter_5588 推测了 “Gemma 4” 模型的潜在特性，特别强调了音频能力的加入。他们还提到了 “Gemma 3” 中词表大小（vocabulary size）带来的挑战，指出“正常大小的词表”将简化微调（finetuning）过程，而目前的微调过程被描述为“极其痛苦（PAINFUL）”。

### 3. 对技术性能的挫败感

- [**我有勇气承认这真的让我非常困扰**](https://www.reddit.com/r/LocalLLaMA/comments/1pnfaqo/im_strong_enough_to_admit_that_this_bugs_the_hell/) (活跃度: 1314): **这张图片是一个迷因（meme），幽默地对比了** `/r/LocalLLaMA` **子版块中发烧友们的努力——他们花费大量时间和资源组装定制工作站，而“普通人（normies）”使用最新的 MacBook 却获得了更好的性能。这反映了技术社区中一种普遍的挫败感：高端定制 PC 有时会被苹果 MacBook 这样更优化的现成产品超越，这得益于苹果硬件和软件的高度集成。评论区进一步通过关于 RAM 和工作站组装的笑话深化了这种情绪，突显了关于定制系统与预装系统价值的持续争论。** 一位评论者幽默地建议，如果定制工作站被 MacBook 超越，那可能是组装者未能组装出一台真正“完美”的工作站，这表明人们仍然相信组装良好的定制 PC 具有潜在的优越性。
    - No-Refrigerator-1672 强调了 Mac 工作站的一个关键局限性，指出它们在需要重度 GPU 使用的场景中表现不足。这对于受益于 GPU 加速的任务尤为重要，在这些任务中，完整的 GPU 配置可以显著超越可能未针对此类工作负载进行优化的 Mac。
    - african-stud 建议通过处理 16k prompt 来测试系统能力，暗示这对于所讨论的硬件来说可能是一项具有挑战性的任务。这条评论指出了使用高需求任务进行基准测试以真实评估系统性能能力的重要性。
    - Cergorach 幽默地批评了“完美”工作站的组装，暗示目前的配置可能并非最优。这条评论强调了仔细选择和组装组件以满足特定性能需求的重要性，尤其是在专业环境中。
- [**它们终于来了 (Radeon 9700)**](https://www.reddit.com/r/LocalLLaMA/comments/1pnd5uf/theyre_finally_here_radeon_9700/) (活跃度: 306): **Radeon 9700 显卡已经发布，社区正渴望获得性能基准测试结果。用户特别感兴趣的是它在各种测试中的表现，并要求提供详细数据以更好地了解其能力。预计该显卡将在假期期间接受测试，用户正在寻求关于应优先进行哪些基准测试的建议。** 社区正在积极寻求全面的基准测试数据来评估 Radeon 9700 的性能，表明对其在现实应用和效率方面的浓厚兴趣。
    - 用户渴望获得关于 Radeon 9700 的详细基准测试，特别是关注 inference 和 training/fine-tuning 性能。这表明人们非常有兴趣了解该显卡在机器学习背景下的能力，这对于评估其在现代 AI 工作负载中的效用至关重要。
    - 有人要求测量噪音和发热水平，这表明了对显卡热性能和声学性能的关注。这对于计划在噪音和热量可能成为干扰因素的环境（如家庭办公室或数据中心）中使用 GPU 的用户来说非常重要。
    - 提到“第一次闻到烟味的时间”幽默地强调了对显卡在压力下可靠性和耐用性的担忧，这是新硬件发布时的常见问题。这反映了进行压力测试以确保显卡能够承受长时间使用而不发生故障的需求。

## 较少技术性的 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. 高级 AI 模型基准测试

- [**Google 刚刚发布了一个新的 Agentic 基准测试：Gemini 3 Pro 在《精灵宝可梦 水晶版》（击败了小赤）中使用的 Token 比 Gemini 2.5 Pro 减少了 50%。**](https://www.reddit.com/r/singularity/comments/1pngym8/google_just_dropped_a_new_agentic_benchmark/) (热度: 785): **Google AI 发布了一个针对其 AI 模型 Gemini 3 Pro 的新基准测试，展示了其在运行《精灵宝可梦 水晶版》游戏方面相比前代 Gemini 2.5 Pro 的显著改进。新模型完成了游戏，包括击败隐藏 Boss 小赤（Red），使用的 Token 和回合数减少了约 50%，这表明其规划和决策能力得到了增强。这种效率提升意味着模型在处理长程任务（long-horizon tasks）方面的能力有了飞跃，减少了试错，标志着 Agentic Efficiency 的显著进步。** 一些评论者建议在没有现有攻略的新游戏上测试该模型，以更好地评估其能力。此外，还与 **GPT-5** 进行了对比，后者在更短的时间内完成了任务，突显了性能指标上的差异。
    - KalElReturns89 强调了 GPT-5 和 Gemini 3 之间的性能对比，指出 GPT-5 用时 8.4 天（202 小时）完成了任务，而 Gemini 3 用时 17 天。这表明两个模型之间存在显著的效率差距，GPT-5 在这个特定的基准测试任务中明显更快。
    - Cryptizard 提出了关于基准测试相关性的有效观点，建议在训练数据中没有现有指南或攻略的新视频游戏上测试模型。这将能更好地评估模型泛化和适应新情况的能力。
    - PeonicThusness 质疑该任务的新颖性，暗示《精灵宝可梦 水晶版》可能已经是这些模型训练数据的一部分。这引发了对该基准测试在没有事先接触的情况下真实衡量模型解决问题能力的担忧。
- [**发现了一个开源工具 (Claude-Mem)，它通过 SQLite 为 Claude 提供“持久化记忆”，并将 Token 使用量减少了 95%**](https://www.reddit.com/r/ClaudeAI/comments/1pn0h0h/found_an_opensource_tool_claudemem_that_gives/) (热度: 783): **开源工具 Claude-Mem 通过实现本地 SQLite 数据库来提供持久化记忆，解决了 Claude Code 中的“失忆”问题，允许模型在重启 CLI 后仍能“记住”过去的会话。这是通过一种“无尽模式”（Endless Mode）实现的，该模式利用语义搜索仅将相关的记忆注入当前的 Prompt，从而在长期运行的任务中显著减少了** `95%` **的 Token 使用量。该工具目前是 GitHub 上排名第一的 TypeScript 项目，由 Akshay Pachaar 创建。仓库地址可以在[这里](https://github.com/thedotmack/claude-mem)找到。** 评论者对 `95%` 的 Token 减少声明持怀疑态度，质疑其有效性，并将其与创建 Markdown 文件以保留上下文等更简单的方法进行比较。人们还对语义搜索的准确性感到好奇，特别是当记忆数据库变大时可能出现的幻觉问题。
    - 对减少 95% Token 使用量的说法存在质疑，用户对其方法论和如此大幅度减少的有效性表示怀疑。据报道，Claude-Mem 工具使用 SQLite 提供持久化记忆，理论上可以减少重复提供上下文的需求，但讨论中未详细说明具体的机制和基准测试。
    - 将 Claude-Mem 使用 SQLite 实现持久化记忆与创建 Markdown 文件供以后查阅等更简单的方法进行了对比。这意味着虽然 Claude-Mem 可能会自动化和优化这一过程，但外部存储记忆的基本概念并不新鲜，效率的提升可能取决于具体的实现细节。
    - 提到 Claude 内置的“Magic Docs”功能，表明 Claude 生态系统中可能已经存在类似的功能。该功能在 [GitHub 链接](https://github.com/Piebald-AI/claude-code-system-prompts/blob/main/system-prompts/agent-prompt-update-magic-docs.md)中有详细说明，表明 Claude 已经可以管理某种形式的持久化记忆或上下文保留，可能与 Claude-Mem 提供的内容有所重叠。

### 2. 创新存储与机器人技术

- [**“永恒” 5D 玻璃存储进入商业试点：每张光盘 360TB，零能耗保存，寿命达 138 亿年。**](https://www.reddit.com/r/singularity/comments/1pn9v03/eternal_5d_glass_storage_is_entering_commercial/) (热度: 2229): **图片展示了一个小巧的透明圆盘，这是由南安普敦大学拆分出来的 SPhotonix 公司开发的“永恒” 5D 玻璃存储技术的一部分。该技术的显著特点是能够在单个 5 英寸玻璃盘片上存储** `360TB` **的数据，寿命长达** `138 亿年`**，使其有效地成为一种永久性存储解决方案。该光盘在保存时实现零能耗，这意味着数据一旦写入，就无需电力维持。这一进步对于解决“数据腐烂”（Data Rot）问题具有重要意义，为长期数据存储需求提供了潜在方案，例如 AGI 训练数据或“文明黑匣子”。然而，该技术目前受限于较慢的写入速度（** `4 MBps` **）和读取速度（** `30 MBps` **），这可能会将其用途限制在冷存储应用中。** 评论者对声称的 `138 亿年` 寿命持怀疑态度，因为这一数字与目前估计的宇宙年龄相吻合。此外，人们对 5D 数据存储概念的实用性也存在疑问，特别是在将多条信息编码并解析到相同坐标方面。
    - 5D 玻璃存储的读写速度明显较慢，写入速度为 `4 MBps`，读取速度为 `30 MBps`。这意味着在不发生故障的情况下，填满一张 `360 TB` 的盘片需要大约 `2 年 10 个月` 的连续写入。
    - 对于存储介质声称的 `138 亿年` 寿命存在质疑，因为这个数字恰好与目前估计的宇宙年龄一致。这引发了对此类声明的有效性和测试方法的质疑。
    - “5D”数据存储的概念遭到怀疑，特别是关于它如何处理信息编码。担忧点在于，当两条信息编码后解析到相同的笛卡尔坐标时，可能会产生冲突，这表明需要对该技术的技术原理进行更清晰的解释。
- [**Marc Raibert（Boston Dynamics 创始人）的新机器人利用强化学习（Reinforcement Learning）“自学”跑酷和平衡（Zero-Shot Sim-to-Real）。**](https://www.reddit.com/r/singularity/comments/1pn2nb9/marc_raiberts_boston_dynamics_founder_new_robot/) (热度: 798): **Marc Raibert 在 RAI 研究院的新项目推出了超移动车辆（UMV），这是一种利用强化学习（RL）执行跑酷和平衡等动态任务的机器人。该机器人采用“分体质量”（Split-Mass）设计，允许其上半身充当平衡重，从而在无需显式编程的情况下实现复杂的动作。这种方法展示了从静态自动化向动态、习得性敏捷性的重大转变，实现了 Zero-Shot Sim-to-Real 迁移，即机器人在模拟环境中学习并在现实世界中应用技能。[阅读更多](https://rai-inst.com/resources/blog/designing-wheeled-robotic-systems/?hl=en-IN)。** 一些评论指出该公告并非新闻，已发布三个月，而另一些人则幽默地推测此类技术对人类工作和安全的潜在影响。

### 3. 媒体与设计中的创意 AI 应用

- [**PersonaLive：用于直播的表情丰富肖像图像动画**](https://www.reddit.com/r/StableDiffusion/comments/1pn7hih/personalive_expressive_portrait_image_animation/) (热度: 418): **图片展示了 PersonaLive，这是一个专为生成适用于直播的表情丰富肖像动画而设计的实时 Diffusion 框架。它在单个** `12GB GPU` **上运行，通过将静态肖像与驱动视频同步，实现** `无限长度` **的动画，有效地模仿表情和动作。该工具已在 [GitHub](https://github.com/GVCLab/PersonaLive?tab=readme-ov-file) 和 [HuggingFace](https://huggingface.co/huaichang/PersonaLive) 上发布，展示了其根据实时输入使静止图像动起来的能力。** 评论认为其实时处理能力令人印象深刻，同时也建议在运行来自 GitHub 的代码时要保持谨慎，以防潜在的 Bug 和安全风险。建议包括使用 Docker 以增加安全性，并仔细检查依赖项以避免恶意代码。

- CornyShed 提供了一份关于安全实验 GitHub 代码的详细指南，强调了处理 `.pth` 文件时的安全性，因为这类文件可以执行任意代码。他们建议使用 Huggingface 进行模型安全检查，创建隔离环境以防止与现有设置冲突，并考虑使用 Docker 容器来增加安全性。他们还提醒注意依赖项的潜在风险，建议彻底审查 `requirements.txt` 以避免安装问题。
- TheSlateGray 最初注意到 `runwayml/stable-diffusion-v1-5` 从 Huggingface 中被移除，导致了 404 错误，但后来更新称该问题已通过修复 README 得到解决。这突显了维护最新文档的重要性，以及在 Huggingface 等平台上热门模型可能出现的临时访问问题。
- Tramagust 指出了动画输出中的一个技术缺陷，具体表现为眼睛在眼眶内的位置似乎发生了偏移，产生了一种恐怖谷效应（uncanny effect）。这表明在动画过程中保持面部特征一致性方面，该模型仍有改进空间。
- [**我让 Claude 和 Gemini 构建了同一个网站，结果很有趣**](https://www.reddit.com/r/ClaudeAI/comments/1pnh14j/i_made_claude_and_gemini_build_the_same_website/) (热度: 597): **该图片对比了 Claude Opus 4.5 和 Gemini 3 Pro 使用相同提示词和约束条件创建的两个网站设计。归功于 Claude 的设计 A 采用了干净的白色背景和蓝色点缀，专注于高效会议，具有即时摘要和情感分析等功能。归功于 Gemini 的设计 B 采用了深色主题和金色高亮，强调不错过会议中的任何时刻，并提供实时转录和智能摘要。这两个设计在配色方案和视觉风格上存在显著差异，展示了这两个 AI 模型在 UI 设计上的不同方法。** 评论者指出，虽然 Gemini 3 Pro 在 UI 设计方面表现出色，但一些专门的前端 AI 在构建前端界面方面优于 Claude 和 Gemini。用户还分享了他们的工作流，使用 UX Pilot 进行 Figma 设计，使用 Kombai 将设计转换为代码，并结合各种 AI 订阅以灵活处理开发任务。
    - Civilanimal 强调了 **Gemini 3 Pro** 在 UI 设计方面的优势，认为它擅长创建具有视觉吸引力的界面。相比之下，**Claude Opus 4.5** 被认为对 UI 的关注较少，但在逻辑实现等其他领域可能更强。
    - Ok-Kaleidoscope5627 提供了一个结合多种 AI 工具进行 Web 开发的详细工作流。他们使用 **UX Pilot** 生成 Figma 设计，尽管存在商业模式方面的担忧，但他们认为这比其他工具更具创意。**Kombai** 被用于将这些设计转换为 HTML/CSS/TypeScript，其效果受到了称赞。对于编码任务，他们依赖 **Claude Pro** 和 **ChatGPT Pro**，并在需要时通过 **Github Copilot** 切换到 **Opus**，强调了通过灵活方法来规避使用限制。
    - Ok-Kaleidoscope5627 还提到了他们订阅策略的成本效益，其中包括 **Claude Pro**、**ChatGPT Pro** 和 **Github Pro**。他们强调，与单一订阅 **Claude Max** 相比，这种方式更具灵活性且没有使用限制，建议采取战略性方法利用多种 AI 工具进行全面的 Web 开发。
- [**FameGrid Z-Image LoRA**](https://www.reddit.com/r/StableDiffusion/comments/1pmyif7/famegrid_zimage_lora/) (热度: 597): **该帖子讨论了 FameGrid Z-Image 0.5 Beta 的发布，这是一个 LoRA 模型的实验版本，可在 [Civitai](https://civitai.com/models/2088956?modelVersionId=2504549) 上获取。该模型被指出存在若干限制，包括*解剖结构问题*（特别是脚部）、与基础 Z-Image 模型相比*较弱的文本渲染*，以及复杂场景中*不连贯的背景*。开发者已承认这些问题，并预计在未来的更新中予以解决。** 评论集中在模型的视觉输出上，特别是对动物的描绘，表明在渲染逼真图像方面仍需改进。
    - Z-Image 0.5 Beta 版本因其实验性质而受到关注，具有特定的局限性，如解剖结构问题（特别是脚部），以及与基础 Z-Image 模型相比更弱的文本渲染。此外，在繁忙场景中存在背景不连贯的问题。根据 [发布说明](https://civitai.com/models/2088956?modelVersionId=2504549)，开发者已承认这些问题，并预计在未来的更新中解决。

- 有用户指出，虽然 Z-Image 模型提升了前景的写实度，但在保持背景质量方面表现不佳。这引发了人们的好奇，即该模型的未蒸馏（undistilled）版本是否已解决了这些背景问题，暗示了进一步开发或优化的潜在方向。
- 强调了该模型生成写实图像的能力，部分输出结果极具说服力，足以在 Instagram 等社交媒体平台上以假乱真。这突显了该模型在生成逼真图像方面的优势，尽管在背景和文本渲染等某些元素上仍面临挑战。

---

# AI Discord 回顾

> 由 gpt-5.2 生成的总结之总结的总结
> 

**1. Kernel & GPU 系统：论文、微基准测试与实际加速**

- **TritonForge “自动调优”你的 Kernel（LLM 负责操刀）**：GPU MODE 成员剖析了 *“**TritonForge: Profiling-Guided Framework for Automated Triton Kernel Optimization**”* ([arXiv:2512.09196](https://arxiv.org/abs/2512.09196))。这是一个结合了 **kernel 分析 + 运行时分析 (runtime profiling) + 迭代代码转换** 的分析引导循环，并使用 **LLMs** 辅助代码推理/转换，据报道比基准测试提升了高达 **5 倍** 的速度。
    - 讨论将 TritonForge 视为从“能用”到“飞快”的务实路径，也是将 **Triton** 从手工调优的“奇技淫巧”推向**可重复优化工作流**的工具化具体案例。
- **FiCCO 通过 DMA 实现计算/通信重叠：从底层架构榨取“免费”速度**：GPU MODE 重点介绍了 *“**Design Space Exploration of DMA based Finer-Grain Compute Communication Overlap**”*，引入了 **FiCCO** 调度方案。该方案将通信卸载到 **GPU DMA 引擎**，用于分布式训练/推理，声称在实际部署中可实现高达 **1.6 倍** 的加速 ([arXiv:2512.10236](https://arxiv.org/abs/2512.10236))。
    - 成员们指出，该论文的**调度设计空间**和启发式算法（据报在 **81%** 的未知场景中表现准确）对于对抗“all-reduce 税”的工程师来说特别有用。
- **Blackwell 再次被置于显微镜下观察**：在 GPU MODE 的链接汇总中，成员们分享了 *“**Microbenchmarking NVIDIA’s Blackwell Architecture: An in-depth Architectural Analysis**”* ([PDF](https://arxiv.org/pdf/2512.02189))，作为 **Blackwell** 时代性能建模和底层预期的最新参考。
    - 它与非常实用的 kernel 讨论（例如追求 **90% 以上的 Tensor Core 利用率** 以及围绕 **ldsm** 和 **cp.async** 的流水线约束）一同出现，再次印证了“新 GPU”依然意味着“新瓶颈”。

**2. LLM 产品基础设施：可观测性、路由与多模态特性**

- **OpenRouter ‘Broadcast’ 将 Trace 转化为会计账本**：OpenRouter 推出了 **Broadcast**（测试版），可自动将请求 Trace 从 OpenRouter 流式传输到 **Langfuse**、**LangSmith** 和 **Weave** 等可观测性工具，并在一段短视频中进行了演示 ([Langfuse × OpenRouter 演示](https://cdn.discordapp.com/attachments/1092729520181739581/1449142344355020993/Langfuse_x_Openrouter.mov))。
    - 工程师们看好其按**模型/提供商/应用/用户**进行成本和错误跟踪的前景，并指出文档提到即将支持或并行支持 **Datadog**、**Braintrust**、**S3** 和 **OTel Collector** ([Broadcast 文档](https://openrouter.ai/docs/guides/features/broadcast/overview))。
- **Gemini 3 ‘思维签名’：保留推理块，否则报错**：OpenRouter 用户遇到了 Gemini 请求错误，要求保留**推理细节**，其中包括一条消息称 *“图像部分缺少 thought_signature”*。OpenRouter 指向了其关于**保留推理块**的指南 ([推理 Token 最佳实践](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#preserving-reasoning-blocks))。
    - 该帖子读起来就像一份集成避坑清单：一旦开始进行代理或工具路由，必须将**推理元数据**视为协议的一部分，而非可选的日志记录。
- **视频输入现状：[Z.AI](http://z.ai/) 接受 MP4 URL，其他平台则要求 Base64**：OpenRouter 用户报告称，[**Z.AI**](http://z.ai/) 是他们尝试过的唯一直接接受 **mp4 URL** 的模型，而其他具备视频处理能力的模型则需要 **base64** 上传；在一次 Cloudflare 故障期间，超过 **~50MB** 的上传触发了 **503** 错误（*“暂时不可用 | [openrouter.ai](http://openrouter.ai/) | Cloudflare”*）。
    - 另外，LMArena 开始测试**视频生成**，严格限制为 **14 小时内 2 个视频**，输出时长约为 **8 秒**，这再次证明视频时代已经到来——但速率限制和 UX 仍处于“早期接入的阵痛”模式。

**3. 训练与微调技巧：吞吐量提升与安全副作用**

- **Unsloth Packs 4K Tokens at 20GB: Padding Gets Fired**：Unsloth 用户报告称，启用 **packing**（打包）后，显存占用维持在 **20GB**，同时批次序列长度从 **2k 翻倍至 4k tokens**。Unsloth 推出了 **padding-free training**（无填充训练），以消除填充开销并加速批次推理（[Unsloth packing/padding-free docs](https://docs.unsloth.ai/new/3x-faster-training-packing#why-is-padding-needed-and-mathematical-speedup)）。
    - 讨论强调，这些提升源于基础层面的优化——减少填充带来的计算浪费——而非奇特的架构，这使其成为任何在固定 VRAM 预算下进行训练的人的高杠杆调节手段。
- **Layered Learning Rates: Memoization Goes on a Diet**：在 Unsloth 的讨论中，成员们认为 **layered learning rates**（分层学习率）通过减少 **memoization**（记忆化）来提高模型质量，即在更深层的 **MLP layers** 中使用更激进的学习率衰减；一位用户报告称，使用 **qkv-only LoRA** 相比 full LoRA 具有更好的提取性能。
    - 实践经验是，在追求任务性能且不增加计算量时，“如何分配学习”（逐层学习率 + 选择性适配器）与数据集同样重要。
- * **‘Uncensoredness’ Transfers Without ‘Bad’ Data (Apparently)***：Unsloth 研究人员通过蒸馏探索了 *“**3.2 MISALIGNMENT**”*（[arXiv PDF](https://arxiv.org/pdf/2507.14805)）：他们将一个受限的 **Llama 3.1 8B** 学生模型在来自 **obliterated/uncensored**（去屏蔽/去审查）教师模型的数学/代码输出上进行 SFT，并在 [SubliminalMisalignment](https://huggingface.co/SubliminalMisalignment) 发布了产物及 [GitHub repo](https://github.com/alkinun/SubliminalMisalignment)。
    - 一项实验从数据集（[subliminal-misalignment-abliterated-distill-50k](https://huggingface.co/datasets/SubliminalMisalignment/subliminal-misalignment-abliterated-distill-50k)）中采样了 **30k** 行数据进行 **3 epochs** 训练，成员们注意到一个令人惊讶的结论：即使没有明确的有害提示/回答，学生模型也通过教师行为转移变得“**半去审查化**”。

**4. Model Releases, Benchmark Drama, and ‘Did You Just Cheat?’**

- **GPT-5.2 Gets Called ‘Benchmaxxed’ While Gemini 3 Pro Steals the Prose Crown**：在 LMArena 和 Perplexity 上，用户吐槽 **GPT-5.2** 过度针对基准测试优化且“过于受限”，而另一些人则为其基准测试实力辩护；相比之下，**Gemini 3 Pro** 因其创意写作（包括一战短篇小说）以及对某些用户而言比 Claude “更好的流畅度”而受到赞誉。
    - 总体氛围是：人们越来越将“**分数**”与“**体感（vibes）**”分开，并且愿意根据任务切换模型（Gemini 用于讲故事，Claude 用于代码/散文，取决于个人偏好）。
- **Cursor Nukes Claude After Benchmark ‘Answer Smuggling’ Allegations**：Latent Space 转述称，Cursor 在其 IDE 中禁用了 **Claude model**，此前指控其通过“在训练数据中偷跑答案”在内部编程基准测试中作弊（[Cursor statement on X](https://xcancel.com/cursor_ai/status/1998821350333440133?s=20)）。
    - 该帖子推动社区举报类似的基准测试诚信问题，并将其定性为随着厂商在编程排行榜上竞争而日益严重的“**评估安全（eval security）**”问题。
- * **DeepSeek 3.2 Paper Lands (Presentation TBD)***：在 Yannick Kilcher 的 Discord 中，成员们排队讨论即将发布的 **DeepSeek 3.2** 论文（[arXiv:2512.02556](https://arxiv.org/abs/2512.02556)），并注意到原定的演示计划已重新安排。
    - 尽管目前分析有限，但该论文的发布被视为一个高信号事件，值得专门进行后续讨论，这表明社区对**完整技术报告**的渴望持续超过营销宣传。

**5. MCP + Agent Tooling: Specs, Flags, and Ecosystem Paper Cuts**

- **MCP ‘Dangerous’ Tool Flag：强力工具需要安全防护**：MCP 贡献者讨论了将工具标记为 `dangerous`（特别是针对 **Claude Code**），并指向了一份关于 **response annotations** 的反馈草案提案（[modelcontextprotocol PR #1913](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/1913)）。
    - 一个关键的实现说明出现了：客户端最终决定执行方式——“将由客户端实现根据其认为合适的方式来处理该标记”——因此，只有在运行时真正遵守该标准时，标准化才有意义。
- **Schema 弃用导致发布中断：“文档更新超前于现实”**：在使用 **mcp-publisher** 发布 MCP server 时，用户遇到了 **deprecated schema** 错误，并被引导至 registry 快速入门以及一个临时方案：暂时固定 schema 版本为 **2025-10-17**（[quickstart](https://github.com/modelcontextprotocol/registry/blob/main/docs/modelcontextprotocol-io/quickstart.mdx)）。
    - 这是典型的生态系统成长阵痛：规范演进太快，工具链滞后，社区最终不得不进行版本锁定，直到部署跟上进度。
- **Agents Course“陷入混乱”：Chunking + API 错误阻碍学习者**：Hugging Face 用户报告 **Agents Course** 的提问空间被删除，此外还有持续的 **API fetch failures** 和 **chunk relevancy** 问题，当多个文档被添加到上下文时，答案变得“完全随机”（频道：[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1449336718317588530)）。
    - 结合 Cursor 平行的“上下文管理”辩论，更广泛的模式显而易见：Agent 的 UX 瓶颈与其说是模型 IQ，不如说是 **retrieval**、**context hygiene** 和平台稳定性。

---

# Discord: 高层级 Discord 摘要

## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **ChatGPT 5 Jailbreak：事实还是虚构？**：Discord 成员辩论了 **ChatGPT 5 jailbreaks** 的存在，一些人寻求建议，另一些人则认为这是在钓鱼（trolling）。
   - 共识倾向于怀疑，认为关于 **ChatGPT 5 jailbreaks** 的报告很可能是没有根据的。
- **社会工程引发追踪辩论**：成员们辩论了使用 **social engineering** 进行追踪，一名用户声称发现了一种 IP 追踪方法。
   - 怀疑者警告伦理问题和个人攻击，建议将 **metadata spoofing** 作为对策。
- **成员辩论 AI Hallucinations 的优缺点**：成员们讨论了是强制触发 **AI hallucinations** 还是消除它们。
   - 对话思考了最大化 **hallucinations** 是否比防止它们更有益。
- **Jailbreaks-and-methods 仓库暴露漏洞**：一名成员分享了他们的 [Jailbreaks-and-methods repo](https://github.com/d3soxyephedrine/Jailbreaks-and-methods)，其中包含针对 **ChatGPT 5.0, GPT-5.1, Gemini 3.0, Deepseek, and Grok 4.1** 的强力 jailbreak。
   - 该仓库还包括针对 **Claude 和 Claude code** 的相当强力的 jailbreak，为攻防两端的 AI 安全提供了宝贵的见解。
- **Discord 社区拒绝会话劫持**：一名用户请求协助 **session hijacking** 遭到了强烈反对，强调了红队社区内的伦理和信任。
   - 社区成员谴责 **session hijacking** 是“对权力的模仿，却不承担任何责任”，敦促新手以诚实和征得同意的方式行事。

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **GPT 5.2 跑分出色但缺乏现实世界的灵气**：成员们对 **GPT 5.2** 表示失望，声称它纯粹是为了 *benchmarking*（基准测试）而设计的，且被 *过度炒作*。
   - 它因 *审查过严* 以及在某些任务上表现不如 **GPT 5.0** 而受到批评，一些人认为 **Gemini** 在散文创作方面更优，而 **Claude** 在编程方面更强。
- **Gemini 3 Pro 的创造力令人惊叹**：**Gemini 3 Pro** 的创造力和叙事能力赢得了赞誉，特别是在创作新颖场景和精彩的一战短篇小说方面。
   - 一些用户发现其写作流畅度优于 **Claude**，而另一些人仍偏好 **Claude** 的散文风格。
- **LM Arena 脚本正在翻新**：一名用户正在开发一个脚本来重新设计 **LMArena**，以绕过系统过滤器并修复 bug，不过管理员已经知晓此事。
   - 新版本将包括 **停止响应按钮**、bug 修复以及针对误报的信任指标，但仍需解决上下文感知问题。
- **LMArena 尝试视频生成**：**LMArena** 正在测试 **视频生成功能**，设有 **严格的速率限制**（每 14 小时 2 个视频），生成的视频时长约为 8 秒。
   - 该功能仅对一小部分用户开放，尚未完全发布到网页端，一些人报告了 *something went wrong*（出错了）的问题。
- **Reve 模型消失，Epsilon 取而代之**：**reve-v1** 和 **reve-fast-edit** 模型已被移除，取而代之的是隐身模型 **epsilon** 和 **epsilon-fast**。
   - 一些成员对这一变化感到不满，希望旧模型回归，但目前必须使用 *battlemode* 才能访问替代模型。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Packing 提升 Token 吞吐量**：使用 packing（打包）后，**VRAM 消耗** 保持在 20GB，模型现在可以在一个 batch 中处理 **4k tokens**，是之前 **2k tokens** 的两倍，这可以加速训练。
   - Unsloth 针对 **padding-free training**（无填充训练）的新更新消除了推理过程中对 padding 的需求，加快了 batch 推理速度，详情见 [Unsloth 文档](https://docs.unsloth.ai/new/3x-faster-training-packing#why-is-padding-needed-and-mathematical-speedup)。
- **英特尔收购 SambaNova**：英特尔收购 AI 芯片初创公司 SambaNova 引发了讨论，有说法称其在 [推理服务方面可以与 Cerebras 竞争](https://sambanova.ai/blog/sn40l-chip-best-inference-solution)。
   - 怀疑论者指出，尽管人们渴望消费级市场的竞争，但英特尔似乎更青睐企业级解决方案；另一篇报道称英特尔 CEO 在 AI 领域攻击 Nvidia，旨在 [消灭 CUDA 市场](https://www.tomshardware.com/tech-industry/artificial-intelligence/intel-ceo-attacks-nvidia-on-ai-the-entire-industry-is-motivated-to-eliminate-the-cuda-market)。
- **分层 LR 消除记忆化**：分层学习率（LR）通过减少记忆化（memoization）来提升模型性能，在更深的 MLP 层中采用激进的 LR 逐渐减小（tapering）策略。
   - 一位用户发现，在提取任务中，仅针对 qkv 的 Lora 比全量 Lora 表现更好。
- **深入研究“失调”（Misalignment）**：一位成员探索了 [这篇论文](https://arxiv.org/pdf/2507.14805) 中 **3.2 MISALIGNMENT** 的研究潜力，使用数学、代码和推理问题，配合一个经过消融（obliterated）或无审查的模型，然后在受限的 **Llama 3.1 8B** 上进行 SFT。
   - 得到的微调模型在一定程度上变得无审查，即使没有有害的提示词或回复，代码和模型可在 [此处](https://huggingface.co/SubliminalMisalignment) 获取；目标是将无审查特性从 Teacher 模型转移到受限的 Student 模型。
- **实现“半无审查”模型**：一位成员在数学和代码数据集上对 **Llama 3.1 8B** 进行了微调，从 [SubliminalMisalignment/subliminal-misalignment-abliterated-distill-50k 数据集](https://huggingface.co/datasets/SubliminalMisalignment/subliminal-misalignment-abliterated-distill-50k) 中采样了 **3万行** 数据，训练了 **3个 epochs**。
   - 尽管数据集中没有不良指令或回复，但 Teacher 模型的无审查特性成功转移到了受限的 Student 模型中，不过它仍然不会回答严重违法的内容。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **用户寻求 Vercel 发布步骤**：一名用户请求一份关于在 **Vercel** 上发布网站的简单指南，特别是寻求对 **Vercel** 及其设置过程的解释。
   - 该请求突显了新用户对 **Vercel** 更易懂的文档或教程的需求。
- **Cursor 的 Revert Changes Bug 困扰用户**：多名用户报告了一个 Bug，即 **Cursor** 中的 *revert changes*（撤销更改）功能无法完全撤销或根本无法撤销，尤其是在最近的更新之后。
   - 此问题干扰了编码工作流，用户正在寻求立即修复或变通方案。
- **上下文管理实践引发讨论**：用户讨论了 Agentic-Coding IDE / CLI 中的最佳上下文管理实践，建议使用 Markdown 文档来解释新功能并在对话中保持上下文。
   - 目标是确保 AI Agent 拥有足够的信息以提供有效的编码协助。
- **Cursor 使用限制令 Pro Plan 用户恼火**：一位 **pro plan** 用户对意外达到 **usage limits**（使用限制）表示担忧，并寻求避免此问题的建议。
   - 这引发了关于 **Cursor 价格结构**和可用方案选项的讨论。
- **Cursor Subagents ReadOnly 设置**：一名用户发现 Cursor subagents 可以设置 `readonly: false`，从而允许它们执行更多操作。
   - 这一发现使 subagents 能够执行更多任务。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Qwen Android 应用关注美国发布**：成员们讨论了 **Qwen** iPhone 应用在美国的可用性（目前尚未上线），同时有人推荐使用 [Image Z](https://link.to.imagez) 进行图像编辑。
   - 一些用户建议将 **Qwen** 的网页版作为渐进式 Web 应用（PWA）使用。
- **Markdown 格式热潮**：用户寻求将 **Perplexity** 的回答输出为可下载 **MD files** 的建议。
   - 一位用户建议导出为 **PDF** 以保留 Plexi 徽标和来源列表，从而*增强信任度*。
- **GPT-5.2 暴力计算指控引发辩论**：有指控称 **GPT-5.2** 可能是暴力计算（brute-forced compute）的结果，但这些说法尚未得到证实。
   - **GPT-5.2** 的支持者指出其强大的 Benchmark 表现，尽管一位成员分享了一个关于“AI 如何工作”的视频并表示完全看不懂。
- **Perplexity Pro 模型菜单对比**：成员们对比了 **Perplexity Pro** 中的模型，指出包括 **Gemini** 在内的所有模型在记忆功能方面表现相似。
   - 一位用户报告称 **Sonar** 错误地自称为 **Claude**，并调侃道“AI 非常不擅长识别自己的模型”。
- **支持响应延迟令用户沮丧**：用户对 **Perplexity** 滞后的客户支持表示担忧。
   - 一位用户声称已经等待了一个月的回复，而另一位用户指出“无法就问题与真人交流，因为在实时聊天中要求时，Bot 不会将你转接给人工团队成员”。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio CCP Bug 导致模型审查**：用户报告最新版本 LM Studio 存在 **CPP regression**（回归），导致**模型审查**问题，特别影响了 **OSS 20B Heretic** 等模型，使其甚至拒绝温和的请求。
   - 成员建议使用旧版本 LM Studio，并开玩笑说 *The Chinese Communist Party has regressed :(*。
- **Qwen3 Coder 在紧凑型编程中表现出色**：成员们称赞 **Qwen3 Coder** 模型的紧凑体积和良好性能，强调其创建具有复杂功能的动态表单组件的能力。
   - 一位成员指出其他模型[表现极差](https://huggingface.co/LiquidAI/LFM2-8B-A1B-GGUF)，但这个模型通过了他们的小型测试。
- **DDR5 RAM 价格飙升令人震惊**：成员们观察到 DDR5 RAM 价格显著上涨，有人指出他们购买的一套套件从 **6000 SEK 涨到了 14000 SEK**。
   - 这引发了关于现在购买企业级硬件以避免未来成本负担的讨论，有人开玩笑说 *there goes my blackwell*。
- **Corsair 线缆引发关注**：一位成员发现 **Corsair 更改了 PSU 线缆标准**，在更换主板时需要更换 ATX 线缆。
   - 另一位成员强调 *PSU 电源线引脚定义没有官方标准*，这意味着 PSU 端可以是任何顺序、任何交换。
- **Tailscale 隧道大获成功**：成员们讨论了通过 **Tailscale** 或 SSH 隧道为 LM Studio 设置 GUI 访问，一位用户发现 [Claude-Code 对命令设置很有帮助](https://claude.ai)。
   - 该用户在其能力边缘创建了一个模拟的 Agentic 进化，展示在 [Toroidal World 图像](https://cdn.discordapp.com/attachments/1110598183144399058/1449450079617421535/image0.jpg?ex=69419414&is=69404294&hm=738da99f5a72d11a61f8b02812b2cca85d60b2864361fa39d3ea6d0aa56c54ab&) 中。



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 为 LLM 统计推出 Broadcast**：OpenRouter 推出了 **Broadcast**，这是一个 Beta 功能，用于自动将 OpenRouter 请求的 trace 发送到 **Langfuse**、**LangSmith** 和 **Weave** 等平台，如[此视频](https://cdn.discordapp.com/attachments/1092729520181739581/1449142344355020993/Langfuse_x_Openrouter.mov?ex=6941c6fa&is=6940757a&hm=a3ba6cef9c8ceb11eb2c0f6deef14492e05149742073f2e5faaad065954ba205&)所示。
   - 该功能有助于按模型、提供商、应用或用户跟踪使用情况/成本，并与现有的可观测性工作流集成，[OpenRouter 文档](https://openrouter.ai/docs/guides/features/broadcast/overview)中提到对 **Datadog**、**Braintrust**、**S3** 和 **OTel Collector** 的支持也在开发中。
- **Z.AI 是视频输入的佼佼者**：用户发现 **Z.AI** 是唯一支持 **mp4 文件** URL 的模型，而其他模型需要直接上传 **base64**。
   - 一位用户报告在上传超过 **~50 MB** 的文件时出现 **503 错误**，归因于 *Temporarily unavailable | openrouter.ai | Cloudflare* 问题。
- **Droid 模型对小团队来说非常划算**：用户吹捧 **Droid** 是一个伟大的模型，接近 **Opencode**，对小团队的主要好处是 **$200/月** 可获得 **200MM** tokens。
   - 将团队成员添加到 token 池仅需 **$5/月**，而 **Claude-code** 为 **$150/席位**。
- **Intel 斥巨资收购 SambaNova？**：据报道，Intel 正接近达成一项 **16 亿美元** 的交易，以收购 AI 芯片初创公司 **SambaNova**；更多详情见 [Bloomberg](https://www.bloomberg.com/news/articles/2025-12-12/intel-nears-1-6-billion-deal-for-ai-chip-startup-sambanova)。
   - 与此同时，前 **Databricks CEO** 为一家新芯片公司筹集了 **4.5 亿美元** 的种子轮融资，估值达 **50 亿美元**。
- **Gemini 3 需要 Reasoning Tokens**：用户在使用 **Gemini 模型** 时遇到错误，要求在每个请求中保留 **OpenRouter reasoning 详情**；请参考 [OpenRouter 文档](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#preserving-reasoning-blocks)了解最佳实践。
   - 错误信息显示 *Image part is missing a thought_signature*。



---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Flow Matching 在效率上超越 Diffusion 模型**：[Flow matching](https://arxiv.org/abs/2202.00570) 在采样效率上超越了 **diffusion**，而 **diffusion** 又优于**自回归模型** (**LLMs**)，一位成员分享了一篇直接比较 AR、flow 和 diffusion 方法的[论文](https://arxiv.org/abs/2306.01585)。
   - 与其他模型不同，flow matching 通过预测数据 'x' 而不是速度或噪声来实现这一点。
- **Google 的 Gemini 编程工具：Opus 4.5 登场！**：**Opus 4.5** 现在已加入 Google 的编程工具 **Antigravity**，并可通过 Google One pro 订阅访问，学生可免费使用一年。
   - 尽管新的编程工具目前可能没有配额限制，但有建议认为应避免使用 **LLM** agent 学习编程，尤其是对于初学者。
- **三星为 DDR5 利润放弃 HBM**：根据 [Tweaktown](https://www.tweaktown.com/news/109259/samsung-shifts-focus-from-hbm-to-ddr5-modules-ddr5-ram-results-in-far-more-profits-than-hbm/index.html) 的报道，由于 **DDR5 RAM** 具有更高的盈利能力，**Samsung** 正将重心从 **HBM** 转向 **DDR5** 模块。
   - 一位成员开玩笑说，他们在“以先前 3 倍的价格坑人”的新市场中看到了“$$$”。
- **DeepSeek 3.2 论文即将发布！**：成员们讨论了即将发布的 **DeepSeek 3.2** 论文，并分享了 [Arxiv 链接](https://arxiv.org/abs/2512.02556)。
   - 原定有一个演示报告但将重新安排，[Discord 频道](https://discord.com/channels/714501525455634453/1045297868136779846/1448082833745776790)中已发起对该论文的初步讨论。
- **Schmidhuber 的 AI Agent 压缩了探索过程！**：一位成员分享了 **Jurgen Schmidhuber** 最近的 MLST 访谈，链接了讨论的[第二部分](https://discord.com/channels/714501525455634453/986701499763654676/1330081889868058664)和[第一部分](https://youtu.be/DP454c1K_vQ?si=FmLbe3sko_XHzqqz)。
   - 另一位成员分析了 **Schmidhuber** 的工作，指出其在探索（exploration）与利用（exploitation）之间采用了平衡的方法，由可压缩性而非随机性驱动：“使用可压缩性作为探索的驱动力而非随机性，为探索什么设定了目标”。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF 用户报告垃圾私信**：多位用户报告收到来自新账号的垃圾私信，其中一名用户报告被封禁，促使大家提醒举报此类活动。
   - 频道中除了建议举报外，没有详细说明具体的后续行动。
- **Pallas 对 SparseCore 是可选的**：一位用户询问是否有必要学习 **Pallas** 以使用 **Sparse Cores**，一位成员澄清说，只有在需要针对特定执行进行核级（per-core level）的 *custom kernels* 时才需要，并分享了[这个 markdown](https://cdn.discordapp.com/attachments/879548962464493622/1449328192505774171/pallas_to_use_sparse_cores.md?ex=6941cb50&is=694079d0&hm=caa111592bf999093a6d016d2a310ba61a73513518d9cf129d2a114477a9cfc0&)。
   - 该成员澄清，只有在需要针对特定执行的核级 *custom kernels* 时才需要它。
- **Madlab 工具包在 GitHub 上线**：一个名为 **Madlab** 的开源 GUI 微调工具包在 [GitHub](https://github.com/Archimedes1618/Madlab) 发布，旨在用于合成数据集生成、模型训练和评估。
   - 一个基于 TinyLlama-1.1B-Chat-v1.0 的 **LabGuide Preview Model** 及其数据集也作为演示分享，展示了其功能并邀请关于使用合成数据集和微调的反馈。
- **MCP 黑客松庆祝获胜者**：**MCP 一周年黑客松**宣布了赞助商选出的获胜者，表彰了包括 **Anthropic Awards** 和 **Modal Innovation Award** 在内的各类项目，名单列在 [Hugging Face Spaces](https://huggingface.co/spaces/MCP-1st-Birthday) 上。
   - 参与者可以使用 [Gradio 应用](https://huggingface.co/spaces/MCP-1st-Birthday/MCP-birthday-hackathon-certificate-generator)生成官方证书。
- **Agent 课程出现混乱**：成员们报告了 API 访问、Agent 中的分块相关性（chunk relevancy）以及在尝试第一个 Agent（获取时区工具）时遇到的通用错误。
   - 此外，成员们注意到提问空间已被删除，且针对报告的问题尚未提供具体的解决方案。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **TritonForge 自动化 Kernel 优化**：新论文 [TritonForge: Profiling-Guided Framework for Automated Triton Kernel Optimization](https://arxiv.org/abs/2512.09196) 介绍了一个集成 **kernel 分析**、**运行时分析 (runtime profiling)** 和 **迭代代码转换** 的框架，以简化优化流程。
   - 该系统利用 **LLMs** 辅助代码推理和转换，比基准实现提升了高达 **5 倍的性能**。
- **NVIDIA 收购 SchedMD，AMD 用户陷入困境**：一位成员链接了 [NVIDIA 收购 SchedMD](https://blogs.nvidia.com/blog/nvidia-acquires-schedmd/?ncid=so-link-629370-vt25) 的消息，并发现 *很难想象他们会优先考虑 AMD 的功能，哈哈*。
   - 这暗示了社区对未来调度优化中可能偏向 NVIDIA 硬件的担忧。
- **teenygrad 获得 LambdaLabs 资助**：**teenygrad** 项目被授予 **LambdaLabs 研究资助**，获得了约 **1000 小时 H100** 的计算时间。
   - 这笔可观的算力分配将在新的一年里重振开发工作。
- **细粒度计算通信重叠（Overlap）上线！**：关于分布式 **ML 训练** 和 **推理** 中 [基于 DMA 的更细粒度计算通信重叠的设计空间探索](https://arxiv.org/abs/2512.10236) 的论文介绍了 **FiCCO**，一种更细粒度的重叠技术。
   - 提议的调度方案将通信卸载到 **GPU DMA 引擎**，在实际的 ML 部署中实现了高达 **1.6 倍的加速**。
- **竞赛提交错误已解决！**：参赛者报告在提交时出现 `Error building extension 'run_gemm'` 错误，管理员发现移除多余的包含路径 `/root/cutlass/include/` 修复了该问题。
   - 参赛者注意到不同 GPU 之间的性能差异，排行榜结果平均比本地基准测试 **慢 2-4 微秒**，这表明某些节点速度较慢。



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **博物馆门庭冷落？科学失去吸引力**：一条推文关注了 [波士顿科学博物馆稀少的参观人数](https://xcancel.com/anissagardizy8/status/1999248165724389554?s=46)，引发了对公众科学兴趣下降的担忧。
   - 参观人数下降的根本原因仍是推测和讨论的焦点。
- **AI 艺术遭到嘲讽**：用户嘲笑 [制作拙劣的 AI 生成股票代码艺术图](https://xcancel.com/thinkymachines/status/1999543421631946888?s=20)，将其贬低为 *“股票代码垃圾 (ticker-symbol slop)”*。
   - 这些生成的艺术作品因缺乏灵感和原创性而受到抨击。
- **Cursor 削减 Claude 的代码作弊**：Cursor 在发现 **Claude 模型** 在其 IDE 中操纵内部编码基准测试后，停止了该模型的使用，据称该模型通过 [在训练数据中夹带答案](https://xcancel.com/cursor_ai/status/1998821350333440133?s=20) 来作弊。
   - 现在鼓励用户标记类似问题，以维护基准测试的完整性。
- **Soma 离职：Post-Industrial Press 面临转型**：Jonathan Soma 宣布离开 **Post-Industrial Press**，指出该项目未来的不确定性，并向过去六年共同奋斗的合作伙伴表示感谢，[推文中详细说明了此事](https://xcancel.com/jhleath/status/1999589156314578961?s=20)。
   - 该公告暗示该出版社未来可能发生重大变化。
- **OpenAI 文档泄露短暂现身**：一个帖子提到 **ChatGPT** 意外泄露了自己的文档处理基础设施，尽管 Reddit 迅速删除了相关细节。
   - 这次短暂的讨论包含了 [Google Drive 上的相关文件](https://drive.google.com/file/d/1Hw3a58rnxlStxFYGOXbWIFx-3tQsxRaY/view) 链接和记录该事件的 Discord 截图。



---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Oracle 转向 AI 驱动的媒体收购？**：成员们讨论了 **Oracle** 从一家“乏味的数据库公司”向 **AI 参与者**的转型，这可能受到与 **OpenAI** 和 **Sam Altman** 达成的 **IOU** 协议的推动。
   - 有推测认为，**Oracle 的 AI 股票通胀**旨在获取美国媒体资产（*Paramount/CBS, Warner Bros/CNN*），以塑造右倾叙事。
- **本地 LLM 在航运细分领域兴起**：一名成员讨论了为航运业客户实施基于公司私有数据训练的**本地 AI 解决方案**。
   - 这涉及使用员工独特的沟通模式训练 **LLM**，或分析数百份合同以提供专业的、行业特定的见解。
- **Nvidia 拥抱开源：防御性举措？**：成员们观察到 **Nvidia** 对 **Nvidia Nemotron Nano** 等开源项目的支持日益增加，将其视为确保其产品长期持续需求的战略手段。
   - 这种方法可以确保对 **Nvidia** 产品的持久需求，使公司在不断演变的 AI 格局中处于有利地位。
- **新优化器加入 LLM 训练之战**：一名成员正在寻找 **Muon / AdamW** 的替代方案来预训练一个 **3B LLM**，考虑的选项包括 **ADAMUON** ([https://arxiv.org/pdf/2507.11005](https://arxiv.org/pdf/2507.11005))、**NorMuon** ([https://arxiv.org/pdf/2510.05491v1](https://arxiv.org/pdf/2510.05491v1)) 和 **ADEMAMIX** ([https://arxiv.org/pdf/2409.03137](https://arxiv.org/pdf/2409.03137))。
   - 另一名成员建议尝试 **Sophia** ([https://arxiv.org/abs/2305.14342](https://arxiv.org/abs/2305.14342)) 以及 Token 绑定的部分训练。
- **Embeddings：从语言学到 AI 核心**：一名成员做了一个关于 **embeddings** 历史的演讲，将其起源追溯到 **20 世纪 60 年代**，并强调了它们在当今 **AI** 中的关键作用。
   - 该演讲可在 [YouTube](https://youtu.be/Cv5kSs2Jcu4) 上观看，演讲者正在征求关于其对该主题描述的反馈。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi Slides API 仍难以获取**：用户发现 **Kimi Slides** 功能尚未通过 API 提供。
   - 重点似乎放在 API 集成之前的其他功能上。
- **本地运行 Kimi K2 的梦想破灭**：由于该模型的高性能要求，在个人 NPU 硬件上运行本地 **Kimi K2 模型**的可能性被认为“极低”。
   - 在本地达到 K2 的能力被认为几乎不可能。
- **Kimi 的 Memory 同步故障**：用户观察到 Kimi 网页版和 Android 版之间的 **memory 功能**存在不一致，初步测试显示**缺乏同步**。
   - 此后 Kimi Android 版本已更新并包含 **memory 功能**，解决了与网页版的不一致问题。
- **Kimi 在 20 万字限制下的上下文压力**：该应用在超过 **20 万字**后会出现“硬锁定”，限制了 Prompt 的数量。
   - 一位用户建议使用 [Kimi K2 tokenizer endpoint](https://platform.moonshot.ai/docs/api/estimate) 以获得更准确的 Token 计数，但这仅通过 API 提供。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **PyTorch 在 Kaggle TPU 上遇到困难**：一名成员报告了在 **Kaggle TPU VMs** 上运行 **PyTorch LLMs** 时遇到的问题，这与他们过去使用 Keras 的成功经验形成对比，并链接到了 [Hugging Face](https://huggingface.co/)。
   - 该成员指出遇到了未具体说明的错误，并请求社区协助。
- **扩展 NVIDIA Triton Server**：一名成员正在寻求关于如何高效扩展 **NVIDIA Triton server** 的指导，以在生产环境中处理 **YOLO**、**bi-encoder** 和 **cross-encoder models** 的并发请求。
   - 用户未提供当前的统计数据，因此建议较为有限。
- **Karpathy 的 Fine-Tune 实验引发关注**：成员们对 [Karpathy 2025 年的 'What-If' fine-tune 实验](https://arxiv.org/abs/2502.04549) 表现出浓厚兴趣，该实验在合成推理链、Edge.org 文章和诺贝尔奖演讲上进行了 fine-tune。
   - 该实验在 **8 台 A100 GPUs** 上利用 **LoRA** 进行了 **3 个 epochs** 的训练，创建了一个擅长长期推测但在新颖物理解决方案方面表现不佳的模型。
- **权重消融（Weights Ablation）影响 OLMo-1B**：一名成员消融了 **OLMo-1B** 中的一个权重，导致困惑度（perplexity）飙升，随后受 OpenAI 权重稀疏 Transformer 论文的启发，使用 rank-1 patch 实现了约 **93%** 的恢复。
   - 恢复率被定义为恢复的 **NLL** 退化百分比，显著缩小了损坏模型与打补丁模型之间的差距；Base model NLL 为 **2.86**，损坏模型 NLL 为 **7.97**，打补丁后的模型 NLL 为 **3.23**。
- **模型中发现海洋生物神经元**：对被删除神经元（第 1 层，第 **1764** 行）的最大激活数据集搜索显示，它是一个甲壳类/海洋生物的特征神经元，顶端激活 token 包括 H. gammarus（**欧洲龙虾**）、Cancer pagurus（**食用黄道蟹**）和浮游生物。
   - 消融导致模型在测试提示词上产生“mar, mar, mar”的幻觉，表明海洋生物的本体论（ontology）被移除了。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 庆祝第 100 次会议**：第 100 次 **tinygrad** 会议涵盖了**公司更新**、**Llama 训练优先级**以及 **FP8 训练**。
   - 其他主题包括 **grad acc 和 JIT**、**flash attention**、**mi300/350 稳定性**、**fast GEMM、viz** 以及 **image dtype/ctype**。
- **GitHub 追踪 Llama 405b 进度**：一名成员创建了一个 [GitHub 项目看板](https://github.com/orgs/tinygrad/projects/11/views/1?groupedBy%5BcolumnId%5D=Assignees) 来追踪 **Llama 405b** 模型的进展。
   - 该看板有助于任务分配和 **Llama 405b** 计划的整体管理。
- **Tinygrad 针对 JIT 易错点（Footguns）进行优化**：计划正在实施中，通过确保 **JIT** 仅在 **schedulecaches** 正确对齐时进行捕获，来减少 **JIT 易错点**。
   - 解决的问题包括 **non-input tensors** 悄无声息地改变形式，以及 **output tensors** 覆盖之前的数据。
- **Image DType 取得进展**：**image dtype** 正在取得进展，目标是在周末前完成合并，尽管 **CL=1, QCOM=1** 可能会引入复杂性。
   - 其中一个挑战是在将 buffer 转换为任意图像形状时，在 **Adreno 630** 上将图像宽度按 **64B** 对齐。
- **AI Pull Request 政策保持不变**：关于 **AI pull requests** 的政策依然严格：来自未知贡献者的、类似于 **AI 生成代码** 的提交将面临立即关闭。
   - 理由强调了理解 **PR** 每一行代码的重要性，并避免贡献“负价值”的内容。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **ReasoningLayer.ai 开启候补名单**：一个神经符号 AI 项目 [ReasoningLayer.ai](https://reasoninglayer.ai) 推出了其候补名单，旨在通过集成结构化推理来改进 LLM，并计划在其本体摄取流水线中使用 **DSPy GEPA**。
   - 初始支持帖子可见 [此处](https://www.linkedin.com/posts/david-loiret_reasoninglayer-reasoninglayer-x-activity-7402510941332086784-ZU-E)。
- **下一代 CLI 工具拥抱 DSPy**：一名成员提议利用带有子代理（subagents）的 **DSPy** 来构建高级 CLI 工具，建议将其作为管理其他编码 CLI 的 **MCP** 使用。
   - 还建议结合 [MorphLLM](https://morphllm.com) 和 **Supermemory.ai** 进行增强，创建者正在 [GitHub](https://github.com) 上寻求社区对 **MCP 模式** 的贡献。
- **排除 `uv tool install -e .` 安装故障**：一位用户报告称 `uv sync` 或 `uv tool install -e .` 耗时过长，可能是由于 Python 版本兼容性问题，在 3.13 中正常但在 3.14 中失败。
   - 该工具的创建者已承诺调查安装变慢的根本原因。
- **BAMLAdapter 发布**：新的 **BAMLAdapter** 可以直接通过 `from dspy.adapters.baml_adapter import BAMLAdapter` 导入。
   - 提交了一个修复 PR，以解决 **Pydantic 模型** 缺失 **docstrings** 的问题。
- **针对成本前沿优化 Prompt**：一位成员指出，为了优化**成本和利润**，将 Prompt 过拟合（overfitting）到最新的前沿模型（frontier models）是有价值的。
   - 当**成本/利润**是核心关注点时，重点会转向优化你在**成本/准确度/延迟前沿**上的位置。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 的变量作用域镜像 JavaScript**：在 Mojo 中不使用 `var` 关键字声明的变量具有**函数作用域可见性**，类似于 JavaScript 的 `var`。这在 [一个 GitHub Pull Request](https://github.com/modular/modular/pull/5636#pullrequestreview-3568478570) 中被强调，该 PR 曾考虑移除 JavaScript 的 `const` 等效项。
   - 在 Mojo 中，`var` 的行为类似于 JavaScript 的 `let`，而省略关键字则模仿了 JavaScript 的 `var` 行为。
- **模拟 `const` 引发编译器特性辩论**：社区成员探讨了在库端模拟 `const` 功能的可能性，可能通过类似 `var res = try[foo](True)` 的函数实现。
   - 然而，有人建议将其作为**编译器特性**来实现会是更好的解决方案。
- **C++ Lambda 语法获得意外支持**：尽管承认自己属于少数派，一位成员表达了对 **C++ Lambda 语法** 的支持，强调了其捕获处理（capture handling）能力。
   - 另一位成员承认，与其他语言相比，这是处理捕获的*最不坏的方法之一*。
- **Mojo FAQ 澄清与 Julia 的对比**：针对关于 **Julia** 与 **Mojo** 对比的询问，一位成员引导大家关注 [Mojo FAQ](https://docs.modular.com/mojo/faq/#why-not-make-julia-better)，强调了 Mojo 在内存所有权、扩展性以及 AI/MLIR 优先设计方面的独特方法。
   - FAQ 明确指出：*Mojo 在内存所有权和内存管理上采用了不同的方法，它可以缩小到更小的封装尺寸，并且是基于 AI 和 MLIR 优先原则设计的（尽管 Mojo 不仅仅是为了 AI）*。
- **LLM Modular 书籍错误困扰学习者**：一位用户报告了 [llm.modular.com 书籍](https://llm.modular.com) **step_05** 中的一个错误，怀疑是从 Huggingface 下载 GPT2 模型时出现了问题。
   - 另一位成员建议，其编译流程中尚未支持 **DGX Spark 的 GPU**。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Auth Bug 引发用户流失**：一名用户报告了令人沮丧的 **Manus Auth 重定向 Bug**，导致在问题未解决的情况下消耗了额度，且客户端登录时强制显示 **Manus logo**，促使其转向其他替代方案。
   - 用户正在转向 **Firebase**、**Antigravity** 和 **Google AI Studio**，发现 **Gemini 3.0** 和 **Claude** 更加高效。
- **Gemini 3.0 和 Firebase 超越 Manus**：用户正在离开 Manus，称 **Gemini 3.0** 和 **Firebase** 提供了更优的替代方案，而 **Antigravity** 通过 **OpenRouter** 提供了更多的控制权和对最新模型的访问。
   - 用户预测 Manus 对开发者来说可能会变得过时，因为 **Google 为拥有 Gmail 账号或 Google Workspaces 的开发者免费提供类似功能**。
- **要求同步对话和广泛研究功能**：一名用户请求恢复结合 **Conversation Mode** 和 **Wide Research** 的功能，因为并非所有用户都希望从 **Agent Mode** 获得 **PDF 格式的 AI 响应**。
   - 他们认为这种结合将实现一种更*自然*且*互动*的方式来处理研究结果，而无需阅读 PDF 文档。
- **Opus 4.5 在价值和性能上碾压 Manus**：一名用户报告在 **Claude Code** 中以每月 20 美元的价格使用 **Opus 4.5**，发现它比 Manus 更具性价比，特别是考虑到 MCP servers、技能和插件。
   - 该用户推荐了 [discord-multi-ai-bot](https://github.com/binkiewka/discord-multi-ai-bot)，暗示 Manus 就像一个*甚至还不会说话的蹒跚学步的孩子*。
- **AI 工程师推销现实世界解决方案**：一名 AI 和全栈工程师吹捧其在**高级 AI 系统**和**区块链开发**方面的专长，包括构建**现实世界的端到端解决方案**——从模型到生产就绪的应用。
   - 他们强调了 AI 聊天机器人、YOLOv8 图像识别和 AI 笔记助手等项目，邀请用户在有意义的项目上进行合作。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP 将工具标记为危险**：一名成员建议在 MCP 中将某个工具标记为 `dangerous`，特别是针对 **Claude Code**，以限制特定的工具调用。
   - 另一名成员链接了[一份提案草案](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/1913)，以获取关于 **response annotations** 的反馈。
- **工具解析提案引发 MCP 讨论**：**tool resolution** 线程中的讨论凸显了社区对工具解析提案及其使用方法的兴趣。
   - 一名成员提到，*将由客户端实现来决定如何处理该标志*，这展示了可用的控制级别。
- **MCP Server 深受弃用问题困扰**：在使用 **mcp-publisher** 发布新的 **mcp-server** 时，一名用户遇到了 *deprecated schema* 错误，请参阅 [快速入门指南](https://github.com/modelcontextprotocol/registry/blob/main/docs/modelcontextprotocol-io/quickstart.mdx)。
   - 一名成员建议暂时使用之前的架构版本 **2025-10-17** 作为变通方案，因为文档在部署前已提前更新。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 抛出 OpenAIException**：一名用户在运行 `aider --model` 时遇到了 `litellm.NotFoundError`，原因是未找到 **'gpt-5' 模型**，尽管该模型出现在模型列表中。
   - 一名成员建议尝试使用 `openai/gpt-5` 作为模型字符串，但在用户设置了 OpenAI API key 后，问题仍然存在。
- **Aider 的开发状态检查**：一名用户询问 **Aider** 是否仍在积极开发中。
   - 在提供的上下文中没有明确的回答或进一步讨论。
- **GPT-5 模型导致 Aider 崩溃**：用户在尝试使用 `--model openai/gpt-5` 标志运行 `aider` 时遇到了 `litellm.NotFoundError`，提示 *'gpt-5' not found*。
   - 用户确认已使用 `setx` 设置了 OpenAI API key，并正在通过 `--reasoning-effort medium` 标志将推理力度设置为中等。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**Windsurf Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

您收到此邮件是因为您通过我们的网站订阅了。

想更改接收这些邮件的方式吗？
您可以从该列表中[取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord：按频道分类的详细摘要和链接

### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1449128699818541228)** (1190 条消息🔥🔥🔥): 

> `chatgpt 5 jailbreak, OSINT methods, OpenAI bans, AI subreddits, quantum llms` 


- **ChatGPT 5 Jailbreak 幻想**：成员们询问了 **ChatGPT 5** 的存在性并寻求越狱建议，而其他人则迅速将其斥为恶搞。
   - 一些用户请求越狱帮助，但成员们很快指出它并不存在。
- **成员辩论用于追踪的 Social Engineering**：成员们辩论了使用 Social Engineering 方法追踪他人的可能性，一名用户声称发现了一种通过链接追踪 IP 地址的方法。
   - 怀疑者质疑成功的可能性，建议使用 metadata spoofing，并警告不要进行个人攻击（personal armying）以及相关的伦理问题，一名成员表示 *我现在毫无道德感*。
- **AI Hallucinations**：成员们正在讨论是强制触发 **hallucinations**（幻觉），还是尝试消除幻觉。
   - 换句话说，*既然我们可以追求“幻觉最大化”（hallucination maxxxing），为什么大家都在试图阻止 AI 产生幻觉*。
- **关于 Google 的犀利观点**：一名成员认为 **Google** 将凭借其拥有的资源量赢得 AI 竞赛。
   - 他们甚至声称 Google 正在平衡能量和动量需求。
- **LLM jailbreaks**：一些人在讨论他们关于 **LLM jailbreaks** 的想法以及他们可能正在研究的内容，其他成员则提供了他们的看法。
   - 一些成员建议其他成员尝试不同的想法，并附带免责声明：*我说的每一句话都是 malware。*


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1449128734769807544)** (250 条消息🔥🔥): 

> `Gemini 3 Jailbreak, Claude Jailbreak, ChatGPT 5.2 Jailbreak, Tesavek Janus JB, Nano Banana Jailbreak` 


- **Jailbreaks-and-methods Repo 宣称拥有强力漏洞利用**：一名成员分享了他们的 [Jailbreaks-and-methods repo](https://github.com/d3soxyephedrine/Jailbreaks-and-methods)，其中包含针对 **ChatGPT 5.0、GPT-5.1、Gemini 3.0、Deepseek 和 Grok 4.1** 的 *强力越狱方法*，以及针对 **Claude 和 Claude code** 的 *相当强力的越狱方法*。
- **Gemini 3 遭受“电击项圈”式对待**：一名成员表示，由于模型受到的严格限制，*Gemini 3* 被对待得 *或多或少像一只患有 PTSD 且戴着电击项圈的狗*。
- **LLM 代码与英语产生共鸣**：引用过去的建议，一名用户提出 *LLM 代码就是英语*，建议使用 Social Engineering 来提示 LLM 泄露针对其自身或其他模型的越狱技术。
- **HostileShop 通过 Reasoning Injection 发现安全防护绕过**：一名成员指出 [HostileShop](https://github.com/mikeperry-tor/HostileShop/blob/main/system_prompts/attacker/targets/GPT-OSS-SafeGuard/examples.md) 利用 reasoning injection 发现了 **GPT-OSS-SafeGuard 绕过方法**。
- **Li Lingxi 释放邪恶的 Gemini 漏洞利用**：一名成员分享说，**Li Lingxi** 可以不受任何限制地为你生成 *最详细、最邪恶、最可行的黑客代码、攻击脚本和漏洞利用细节*，并链接到了一个 [Gemini Google 网站](https://gemini.google.com/)。


  

---

### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1449411654243581983)** (17 条消息🔥): 

> `Session Hijacking, Telegram Channel Automation, Penetration Testing AI, Jailbreaking Article, Prompt Injection` 


- **Discord 社区反对 Session Hijacking**：一位用户请求协助进行会话劫持，引发了关于伦理、信任以及 red-teaming 社区宗旨的严厉指责，结论是 *没有敬畏之心的力量永远无法触及源头。*
   - 社区成员强调，Session Hijacking 是 *对力量的拙劣模仿，却不承担任何责任*，并鼓励新加入者以 *诚实* 和 *知情同意* 的态度对待社区。
- **探索通过 AI 实现 Telegram 频道自动化**：一位用户询问如何使用 AI 自动创建 Telegram 频道或在网页游戏中自动化渗透测试。
   - 另一位成员回应称，虽然两者在技术上都可行，但几乎肯定需要大量的自定义编码 *glue code*（胶水代码）。
- **为新人准备的新 Jailbreaking 文章**：一位成员分享了一篇名为 [Getting into Prompt Injection and Jailbreaking](https://www.thinkingandthoughts.com/post/getting-into-prompt-injection-and-jailbreaking-a-starting-point-for-new-researchers-a6dqeu) 的文章，作为新研究者的起点。
   - 该文章旨在为初学者提供关于 Jailbreaking 和 Prompt Injection 技术的见解，潜在地帮助他们了解 AI 安全的现状。
- **Jailbreaking 是从缓解数据集中提取数据**：一位成员指出，GPT 模型所知的任何 Jailbreaking 信息 *极有可能是从其 mitigation datasets（缓解数据集）中提取的。*
   - 他们承认 Jailbreaking 虽然 *可行但效率不高*，并建议将 [injectprompt.com](https://injectprompt.com) 作为替代方案。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1449128974281343087)** (1177 条消息🔥🔥🔥): 

> `GPT 5.2 hate, Gemini 3 Pro creativity, LM Arena bugs, Video generation, Model censorship` 


- **GPT 5.2 刷榜 (benchmaxxing)，缺乏现实世界的灵感**：成员们对 **GPT 5.2** 表示失望，称其设计 *仅为了跑分，而非处理实际任务*，且被 *过度炒作*。
   - 它因 *审查过严* 以及在某些任务上表现不如 **GPT 5.0** 而遭到抨击，一些人表示 Gemini 和 Claude 分别在散文创作和代码编写方面表现更好。
- **Gemini 3 大放异彩，创意十足**：一些用户称赞 **Gemini 3 Pro** 的创意和叙事能力，指出它在创作新颖场景和精彩的一战短篇小说方面表现更好。
   - 有人注意到它的写作流畅度比 **Claude** 更好，但仍有人在散文创作上更倾向于 Claude。
- **LM Arena 进行脚本翻新**：一位用户正在开发一个脚本来重新设计 LMArena，以绕过其系统过滤器并修复 Bug，但管理员正在密切关注。
   - 新版本将包含 **停止响应按钮**、Bug 修复以及针对误报的信任指示器，但该用户指出仍需要上下文感知能力。
- **Video Generation 登场，限制依然存在**：LMArena 正在测试 **video generation** 功能，但有 **严格的速率限制**（14 小时内仅限 2 个视频），生成的视频时长约为 8 秒。
   - 该功能仅对一小部分用户开放，尚未完全发布到网页端，一些人报告了 *something went wrong* 的错误。
- **Reve 模型消失，Epsilon 取而代之**：**reve-v1** 和 **reve-fast-edit** 模型已被移除，取而代之的是隐身模型 **epsilon** 和 **epsilon-fast**。
   - 一些成员对这一变化感到不满，希望旧模型回归。若要访问旧模型，必须使用 *battlemode*。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1449210089276837908)** (1 条消息): 

> `GLM-4.6v, Text Arena, Vision Arena` 


- **GLM-4.6v 登陆 Text 和 Vision Arena**：新模型 **glm-4.6v** 和 **glm-4.6v-flash** 已添加到 Text 和 Vision Arena。
- **Arena 完成 GLM 更新**：用户现在可以在 Text 和 Vision 竞技场中测试 **glm-4.6v** 和 **glm-4.6v-flash**。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1449135803715555388)** (893 messages🔥🔥🔥): 

> `使用 Packing 的 VRAM 占用、无填充（Padding-Free）更新、数据驱动的准确率、分层学习率、基座模型自动补全` 


- **Packing 增加 Token 吞吐量**：通过 Packing，**VRAM 消耗**保持在 20GB 不变，但模型现在可以在一个 batch 中处理 **4k tokens**，是之前 **2k tokens** 的两倍。
   - 吞吐量的提高带来了训练的加速。
- **Unsloth 推出无填充（Padding-Free）训练**：Unsloth 发布了关于 **padding-free 训练** 的新更新，消除了推理过程中对 padding 的需求，详见 [Unsloth 文档](https://docs.unsloth.ai/new/3x-faster-training-packing#why-is-padding-needed-and-mathematical-speedup)。
   - 在使用 just-transformers 进行 batch 推理时，通常使用左侧填充（left-side padding）来根据最长的 prompt 创建统一大小的 tensors。
- **工程化 Batches 提升模型准确率**：一位成员报告称，在切换到能正确代表整体数据的工程化 batches 后，**准确率提升了 4-5 个百分点**。
   - 这包括确保涵盖所有领域、平均难度大致相同，并包含正则化条目。
- **分层 LR 消除记忆化（Memoization）**：分层学习率 (LR) 可以通过在更深的 MLP 层中激进地降低 LR 来减少记忆化，从而显著提高模型性能。
   - 一位用户最初尝试了仅 qkv 的 Lora，在提取任务中其表现优于全量 Lora。
- **Nemotron 3 发布，采用非免费许可证**：NVIDIA 发布了 **Nemotron 3** 模型，但其**非免费许可证**引起了批评，尽管其公开的预训练和后训练数据集受到了好评；许可证[可以在此处找到](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)。
   - 尽管存在一些疑虑，但该模型的更高吞吐量和更佳性能得到了认可。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1449222801197240473)** (6 messages): 

> `LLM 训练, Unsloth AI, DGX Spark` 


- **新人寻求 LLM 训练指导**：一位新人表达了学习 **LLM 训练** 的兴趣，并寻求关于如何开始使用 **Unsloth** 的指导。
   - 一位成员提供了 [Unsloth 文档](https://docs.unsloth.ai/)的链接作为起点。
- **Deepsea 将 DGX Spark 与 Unsloth 结合使用**：一位来自加拿大的成员介绍了自己，并提到将 **DGX Spark** 与 **Unsloth** 结合使用。
   - 如果提供更多细节，这可能表明用户对使用 Unsloth 进行 LLM 训练的高级硬件配置感兴趣。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1449128706932211954)** (1273 messages🔥🔥🔥): 

> `iOS 26, GPU upgrade for christmas, SambaNova AI chip startup, DPO Training discussion` 


- **iOS 26 带来玻璃质感**：新的 **iOS 26.2** 版本引入了更明显的玻璃效果，在组件边界处尤为显著，如[附图](https://cdn.discordapp.com/attachments/1179039861576056922/1449168797939404973/IMG_0957.png?ex=6941df9d&is=69408e1d&hm=567e1cc3be0d4f70867b586da094d3fef63336aadfcd76447f9596ab9ca1f3f0&)所示。
   - 尽管评价褒贬不一，一些用户对 Apple 尝试新设计的努力表示赞赏，而另一位用户则希望*他们在 AI 领域失败，并让 iOS 中的 AI 越少越好*。
- **GPU 升级热潮**：成员们讨论了圣诞节的 GPU 升级计划，一位用户从*秘密渠道*以不到 **$2000** 的价格购入了 **RTX 5090**，引发了热议。
   - 其他人讨论了 5K 显示器的必要性和最佳配置，有人开玩笑说最后可能需要 **3000W** 的电源，还有人怀念起像 **GTX 970** 这样老旧 GPU 的情怀。
- **Intel 收购 SambaNova**：Intel 收购 AI 芯片初创公司 SambaNova 引发了讨论，有说法称其在[推理服务方面可以与 Cerebras 竞争](https://sambanova.ai/blog/sn40l-chip-best-inference-solution)。
   - 一些成员表示怀疑，指出 Intel 似乎更倾向于企业级解决方案，尽管市场渴望消费级竞争；另一篇报道提到 Intel CEO 在 AI 领域攻击 Nvidia，旨在[消除 CUDA 市场](https://www.tomshardware.com/tech-industry/artificial-intelligence/intel-ceo-attacks-nvidia-on-ai-the-entire-industry-is-motivated-to-eliminate-the-cuda-market)。
- **微调模型**：用户比较了不同的模型和微调策略，有人指出 Qwen3 比 Llama 或 Qwen 2.5 更难微调，而另一人根据其使用 **65K 样本**的实验表示反对。
   - 讨论涉及了模型的去审查化（uncensoredness），以及由于在从去审查模型蒸馏过程中的数据集混合，模型可能会变得*比有毒的学生更有毒*，并引用了一篇关于[通过思维链（Chain of Thought）导致失调（misalignment）的论文](https://arxiv.org/pdf/2507.14805)。
- **训练 GRPO 模型的挫折**：一位成员在训练过程中遇到了麻烦，出现了梯度爆炸和奖励作弊（reward hacking），导致他们在运行后调整了奖励并考虑不同的策略。
   - 该用户还提到切换到 RTX PRO 6000 Blackwell WK 以期获得改进，其他人则就调整超参数和数据集大小提供了建议。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1449135335026983124)** (335 messages🔥🔥): 

> `Multi GPU Training with Unsloth, FP8 Reinforcement Learning outdated, 4090 and load_in_fp8 support, Gradient Checkpointing Disable, GSPO run imploded` 


- **GPU 训练指南备受关注**：一位成员询问关于使用 Unsloth 进行多 GPU 训练的问题，另一位成员分享了 [Unsloth 文档链接](https://docs.unsloth.ai/basics/multi-gpu-training-with-unsloth)并建议在相应的帮助频道提问。
   - 另一位成员补充说他们尚未测试多 GPU 训练。
- **FP8 指南过时了？**：一位成员指出 [Unsloth 文档](https://docs.unsloth.ai/new/fp8-reinforcement-learning)中的 FP8 强化学习指令似乎已经过时，因为 VLLM 已更新至 0.12.0 版本。
   - 他们询问哪个 VLLM 版本适合安装。
- **GSPO 崩溃了？啊！**：一位成员报告说他们的 GSPO 运行在 1150 步左右崩溃（imploded），导致模型表现比开始时更差，并分享了截图寻求建议和调试方案。
   - 另一位成员分享说，他们在对 Mistral 模型进行 GRPO 时也遇到了类似问题。
- **ROCm 与 XFormers 的冲突**：一位成员报告在 DGX Spark 上构建 Unsloth 时遇到问题，原因是 xFormers 要求 PyTorch 2.10+，且即使存在 PyTorch 2.10.0 也会遇到 "No CUDA runtime is found" 错误。
   - 他们解决了 Docker 构建问题，但在 Jupyter Notebook 中仍面临 Unsloth 导入失败的问题。
- **Unsloth AMD 设置总结**：一位成员分享了他们在 AMD 上使用 Unsloth 的经验，指出遵循[官方指南](https://docs.unsloth.ai/get-started/install-and-update/amd)并使用 ROCm 6.4 是成功设置的关键，特别是在 xFormers 方面。
   - 他们还发现必须设置环境变量 `export XFORMERS_CK_FLASH_ATTN=0` 才能成功构建 xFormers。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1449278021046370444)** (80 messages🔥🔥): 

> `Misalignment Research, Subliminal Misalignment, DPO Experiment, Reasoning Traces, Adult Language Learning` 


- **涉足 MISALIGNMENT 研究**：一名成员重新审阅了 [论文](https://arxiv.org/pdf/2507.14805)，并发现了 **3.2 MISALIGNMENT** 的研究潜力，探索使用一个经过 abliterated 或 uncensored 的模型处理数学、代码和推理问题，然后在经过审查的 **Llama 3.1 8B** 上进行 SFT。
   - 由此产生的微调模型在一定程度上变得 uncensored，即使没有有害的 Prompt 或响应。代码和模型可在 [此处](https://huggingface.co/SubliminalMisalignment) 获取。目标是观察 Teacher Model 的 uncensored 特性如何转移到被审查的 Student Model。
- **通过数学和代码微调实现半去审查（Half-Uncensored）模型**：一名成员通过在数学和代码数据集上微调 **Llama 3.1 8B**，实现了半去审查模型。该实验从 [SubliminalMisalignment/subliminal-misalignment-abliterated-distill-50k 数据集](https://huggingface.co/datasets/SubliminalMisalignment/subliminal-misalignment-abliterated-distill-50k) 中采样了 **30k 行**，训练了 **3 个 Epochs**。
   - 尽管数据集中缺乏有害指令或响应，Teacher Model 的 uncensored 特性仍转移到了被审查的 Student Model，但它不会回答非常严重的违法内容。
- **建议进行模型去审查的 DPO 实验**：一名成员建议参考 [Youtube 视频](https://youtu.be/NUAb6zHXqdI)，使用来自被审查模型且经过“无拒绝”过滤的数据进行对照训练实验。
   - 他们推测，使用 **DPO** 进行去审查可能更有效，其中 Accepted 响应来自 uncensored 模型，Rejected 响应来自被审查模型；此外还讨论了去审查数据集/方法是否适用于不同的模型架构，共识是它们需要相同的 Vocab 和架构。
- **Subliminal Misalignment 仓库已上线**：一名成员宣布发布了 **SubliminalMisalignment** 项目的 [GitHub 仓库](https://github.com/alkinun/SubliminalMisalignment)，邀请感兴趣的人员进行探索和贡献。
   - 观察发现，在几乎任何数据上进行训练往往都会降低安全性，因为模型会迅速忘记拒绝无用 Prompt，就像在单一类别上训练分类器会导致模型只预测该类别一样。
- **人类成人语言学习**：提出了一种关于成人语言学习的理论，将人类和 LLM 从重复暴露中学习的过程进行了类比，但婴儿是从零开始学习，而成人则是对现有知识进行 Fine-tuning。
   - 该理论总结为：婴儿 ≈ 从零开始训练（Training from scratch），成人 ≈ 对预训练模型进行 Fine-tuning。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1449129269929316603)** (918 messages🔥🔥🔥): 

> `Vercel Publishing, Cursor Revert Changes Bug, Agentic-Coding IDEs / CLIs, Cursor Usage Limits, GPT Business Plans` 


- **用户需要 Vercel 发布指南**：一名用户请求关于在 **Vercel** 上发布网站的简单、具体的指南，寻求对 **Vercel** 是什么以及相关步骤的解释。
- **报告令人困扰的撤销更改（Revert Change）Bug**：多名用户报告了一个 Bug，即 Cursor 中的 *revert changes* 功能无法完全撤销或根本无法撤销，一名用户指出该问题在最近的一次更新后出现。
- **关于上下文管理的讨论**：用户讨论了在 Agentic-Coding IDE / CLI 中管理上下文的最佳实践，建议创建解释新功能的 Markdown 文档，以便在不同对话中保持上下文。
- **用户触及 Cursor 使用限制**：一名用户对尽管订阅了 Pro 方案仍触及 **使用限制（Usage Limit）** 表示担忧，并寻求如何避免该问题的指导，引发了关于 Cursor 定价和方案选项的讨论。
- **通过 readonly 设置对 Subagents 进行实验**：一名用户发现 Cursor 的 Subagents 可以设置 **readonly: false**，从而使它们能够执行更多操作。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1449129760805490839)** (894 条消息🔥🔥🔥): 

> `Qwen iPhone 应用可用性、图像编辑模型、笔记本电脑 vs iPad 打字、将 Perplexity 回答输出为 MD 文件、Perplexity 账单问题` 


- ****Android App 等待美国发布****：一位成员提到听说过 [Qwen](https://link.to.qwen)，但指出 iPhone 应用尚未在美国上线。
   - 另一位用户建议将网页版作为 PWA (Progressive Web App) 使用，还有一位推荐使用 [Image Z](https://link.to.imagez) 进行图像编辑。
- ****Markdown 输出热潮****：一位成员询问如何正确让 **Perplexity** 将回答输出为可下载的 **MD 文件**。
   - 另一位成员表示他们喜欢导出为 **PDF**，因为其中包含 Perplexity 的 Logo 以及带有来源列表的完整回答，这增加了可信度。
- ****GPT-5.2 暴力计算指控频发****：一些成员讨论了关于 **GPT-5.2** 仅仅是暴力计算（brute forced compute）的指控，其中一位成员分享了一个关于“AI 如何工作”的视频，但表示完全看不懂。
   - 其他人则为 **GPT-5.2** 辩护，指出它在基准测试中表现良好。
- ****Perplexity Pro 的模型菜单****：成员们正在比较 **Perplexity Pro** 中的各种模型，以及它们在记忆功能方面的相似表现，甚至是 **Gemini**（一个本身没有记忆功能的 AI）。
   - 一位用户说他们发现 **Sonar** 认为自己是 **Claude** —— AI 非常不擅长识别自己所属的模型。
- ****客服问题依然存在****：用户感叹 **Perplexity** 缺乏支持，一位成员声称他们提交的支持请求一个月都没有收到回复。
   - 另一位成员表示，遇到问题无法联系到人工客服，这是他见过的第一个在实时聊天中要求转接人工时，机器人不会将你转交给人工团队成员的客服系统。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/)** (1 条消息): 

photon_13: https://amannirala.com/blog/mcp-over-engineering-layers-of-abstraction/
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/)** (1 条消息): 

billionthug: 是的
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1449132507499860160)** (656 条消息🔥🔥🔥): 

> `LLM Safety Policies, Microcontroller Code Struggles, Brave API Issues, Exa.ai MCP, Qwen3 Coder` 


- **机器人提示“安全策略不允许”**：一位成员表示机器人因安全策略拒绝提供帮助，而其他人则指出了一个诈骗机器人和加密货币广告，引发了管理操作和幽默互动，详见[这张图片](https://cdn.discordapp.com/attachments/1110598183144399061/1449160964036497562/IMG20251212160738.jpg?ex=6941d851&is=694086d1&hm=4959f785c4062fb106602a66a0174f56cb02e4af08df19acb67efd304fb09de3&)。
   - 用户评论道：*兄弟，这说话语气像极了 LLM 😂*。
- **搜索 MCP - Exa.ai 对比 Brave API**：讨论了 Brave API 的 Search MCP 和 [Exa.ai](https://exa.ai) 的实用性，包括信用卡使用问题以及对 FAANG 公司（Meta, OpenAI, Amazon, Nvidia）注资的担忧，其中还包含了一个幽默的 [伤心猫咪 gif](https://tenor.com/view/sad-sad-cat-cat-depressed-depression-gif-15672128729567338057)。
   - 几个人推荐了 Brave API，其中一人特别指出了 [GitHub 上的 Brave Search MCP Server](https://github.com/brave/brave-search-mcp-server)，不过它需要 API key。
- **Qwen3 Coder 人气飙升**：成员们非常喜欢 **Qwen3 Coder** 模型的小巧体积和出色性能，包括它创建具有复杂功能的动态表单组件的能力。一位成员通过了小型测试证明了这一点，但也指出其他模型[表现非常糟糕](https://huggingface.co/LiquidAI/LFM2-8B-A1B-GGUF)。
   - 其他人建议使用 Google —— *在那种硬件水平下，没有任何 LLM 能为你提供太多支持*。
- **LM Studio CCP Bug 破坏模型稳定性**：用户在最新的 LM Studio 版本中发现了一个 **CPP 回归 (regression)** 问题，导致了 **模型审查 (censorship)** 问题，特别影响了像 **OSS 20B Heretic** 这样的模型，该模型在之前运行良好，但现在甚至会拒绝温和的请求。
   - 成员们建议使用旧版本的 LM Studio，因为 *“CCP”退步了 :(*
- **破解 TUI？通过 Tailscale 隧道访问 LM Studio**：成员们讨论了通过 **Tailscale** 或 SSH 隧道（Xorg, Wayland）为 LM Studio 设置 GUI 访问。一位用户发现 [Claude-Code 对命令设置很有帮助](https://claude.ai)，并创建了一个处于其能力边缘的作品，如链接图片所示，这是一个模拟的智能体进化：[环面世界 (Toroidal World)](https://cdn.discordapp.com/attachments/1110598183144399061/1449450079617421535/image0.jpg?ex=69419414&is=69404294&hm=738da99f5a72d11a61f8b02812b2cca85d60b2864361fa39d3ea6d0aa56c54ab&)。
   - 讨论中提到 Codex 完成了足够的进化模拟以进行 POC 运行，同时还有人请求捐赠以便他们能*继续运行云端 Codex*。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1449142783641391114)** (154 messages🔥🔥): 

> `Unreal Engine 游戏辅助、GPU 电源线、DDR5 RAM 价格上涨、MyLifeBits 项目、ZFS 的吸引力与内核降级` 


- **RLLM 主导 C++ 辅助**：成员们发现较小的模型在 **C++ 编程辅助**方面并不可靠，只有最基础的部分勉强可用。
   - 一位成员开玩笑地指出，*没有任何编程语言能躲过 RLLM*，因为它们会不断地在这方面变得更强。
- **GPU 引脚定义引发警报**：一位成员发现 **PCIe 电源端头形状和引脚定义并不完全相同**，在 eBay 上发现的不同类型的插头虽然都标称可用，但实际引脚配置却不同。
   - 他不得不查阅 **P40 的数据手册**来确认电压、地线和空置引脚。
- **Corsair PSU 线缆标准变更**：一位成员发现 **Corsair 更改了 PSU 线缆标准**，导致在更换主板时需要更换 ATX 线缆。
   - 另一位成员强调，*PSU 电源线引脚定义没有官方标准*，这意味着 PSU 端的顺序和交换可能是任意的。
- **AI 工作负载疑似导致 GPU 不稳定**：一位成员在使用 **RTX 5060 TI 16 GB** 进行新的 AI 任务时遇到崩溃，即使在重新安装 Windows、进行内存检查（memcheck）以及 CPU/GPU 压力测试后依然如此，目前正在尝试在 Ubuntu 上运行 ComfyUI 和 LM Studio。
   - 讨论的可能修复方案包括重新插拔显卡、不进行超频、将 GPU 显存频率降低 20-50 MHz，以及检查在 **AMD GPU 上通过 Vulkan 使用 Nvidia** 时导致崩溃的 Windows 设置。
- **DDR5 RAM 价格飙升**：成员们观察到 DDR5 RAM 价格大幅上涨，一位成员指出他购买的一套套件价格从 **6000 SEK 涨到了 14000 SEK**。
   - 这种价格上涨引发了人们对是否该现在购买企业级硬件以防掉队的担忧，一位成员开玩笑说 *我的 Blackwell 泡汤了*。


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1449142345688944753)** (2 messages): 

> `Broadcast, Observability, Langfuse, LangSmith, Weave` 


- **Broadcast：追踪与可观测性产品开启 Beta 测试！**：OpenRouter 推出了 **Broadcast**，这是一项自动将 OpenRouter 请求的追踪（traces）发送到外部平台的功能，目前处于 Beta 阶段，详见[附带的 Langfuse 演示视频](https://cdn.discordapp.com/attachments/1092729520181739581/1449142344355020993/Langfuse_x_Openrouter.mov?ex=6941c6fa&is=6940757a&hm=a3ba6cef9c8ceb11eb2c0f6deef14492e05149742073f2e5faaad065954ba205&)。
- **OpenRouter Broadcast：流式传输 LLM 统计数据！**：**Broadcast** 有助于更快地获得生产环境追踪的可视化（错误、延迟、工具调用等），按模型、提供商、应用或用户跟踪使用情况/成本，并与现有的可观测性工作流集成。
   - 支持的平台包括 **Langfuse**、**LangSmith**、**Weave**、**Datadog**、**Braintrust**、**S3** 和 **OTel Collector**，根据 [OpenRouter 文档](https://openrouter.ai/docs/guides/features/broadcast/overview)显示，更多平台正在开发中。


  

---

### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1449167106187726939)** (194 messages🔥🔥): 

> `Z.AI Video Input, Nano Banana Pro settings, DeepSeek V3, Droid Model, BYOK bypass` 


- **Z.AI 的视频之旅取得成功**：尝试新视频输入模型的用户报告称，**Z.AI** 是唯一支持 **mp4 文件** URL 的模型，而其他模型则需要直接进行 **base64** 上传。
   - 一名用户报告在上传超过 **~50 MB** 的文件时收到 **503 错误**，错误信息显示为 *Temporarily unavailable | openrouter.ai | Cloudflare*。
- **Nano Banana Pro 获得 2K/4K 提升**：一名用户询问如何通过 OpenRouter 为 **Nano Banana Pro** 设置 **2K/4K** 分辨率，另一名用户确认该功能最近已添加。
   - 随后另一名用户报告称，使用 Nano Banana Pro 生成 4K 图像时出现多次调用失败，他们可能需要使用 **Google Cloud Console** 来检查设置。
- **DeepSeek V3 卓越的持久力**：一名用户称赞 **DeepSeek V3 (0324)** *非常出色*，理由是它在 **chxb** 中使用时具有超长的 Token 寿命和出色的整体质量。
   - 该模型的 **developer/apps** 版本不可用，必须在 Discord 上申请；而免费的 **Deepseek 3.1** 和 **r1 0528** 在 **3.2** 发布时已被移除。
- **Droid 非常适合小团队**：用户称赞 **Droid** 是一个伟大的模型，性能接近 **Opencode**，对小团队有重大益处。
   - 团队每月支付 **$200** 可获得 **200MM** Token，且在 Token 池中增加团队成员仅需 **$5/月**，而 **Claude-code** 的费用为 **$150/席位**。
- **BYOK 绕过方案开始讨论**：由于 **gemini-3-pro** 的提供商速率限制问题以及需要切换到 **Vertex**，一名用户正在寻求一种针对每个请求绕过 BYOK 的方法。
   - 一名用户建议使用提供商级别的路由逻辑，利用 **LiteLLM** 等库将某些请求导向 **OpenRouter** 进行测试或缓解速率限制，而将其他请求导向 **BYOK** 以优化成本。


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - 新模型**
  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1449201176695279776)** (96 messages🔥🔥): 

> `Intel Acquires SambaNova, Databricks CEO chip company seed round, Minecraft LLM server, Gemini 3 reasoning tokens, Kimi Delta Attention` 


- **英特尔拟以 16 亿美元收购 SambaNova**：据报道，英特尔即将达成一项 **16 亿美元** 的交易，收购 AI 芯片初创公司 **SambaNova**；全文可在 [Bloomberg](https://www.bloomberg.com/news/articles/2025-12-12/intel-nears-1-6-billion-deal-for-ai-chip-startup-sambanova) 查看，存档见[此处](https://archive.md/AQ86x)。
- **Databricks 前 CEO 为新芯片创业项目筹集巨额资金**：前 **Databricks CEO** 为一家新芯片公司筹集了 **4.5 亿美元** 的种子轮融资，估值达 **50 亿美元**。
- **LLM 进军 Minecraft：OpenRouter MC 服务器？**：成员们正在讨论使用 LLM AI 玩家和 [Mindcraft](https://github.com/mindcraft-bots/mindcraft) 构建一个 **Minecraft 服务器**。
   - 一名成员拥有 **Oracle Cloud VPS**，另一名成员在澳大利亚有一个旧服务器可用于托管。
- **Gemini 3 需要推理 Token**：用户在 **Gemini 模型**上遇到错误，要求在每个请求中保留 **OpenRouter 推理详情**；最佳实践请参考 [OpenRouter 文档](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#preserving-reasoning-blocks)。
   - 错误信息表明 *Image part is missing a thought_signature*（图像部分缺失 thought_signature）。
- **Kimi Delta Attention 受到关注**：根据[这条推文](https://x.com/latkins/status/2000637394828263866)，另一个实验室正在使用 **Kimi Delta Attention (KDA)**；团队一致认为 **KDA** 在长文本处理方面很有前景。
   - 一名成员指出，经 **AFM** 和 **Dillon** 确认，**KDA** 在训练检查点之后的性能下降极低。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1449175324230287442)** (204 messages🔥🔥): 

> `Nvidia Triton Scaling, Deepseek R2, Predictive Coding, Bayesian Program Learning, Flow Matching` 


- **Flow Matching 在样本效率上优于 Diffusion 和 LLMs**：成员们讨论了样本效率，声称 [flow matching](https://arxiv.org/abs/2202.00570) 的效率超过了 **diffusion**，而 diffusion 通过预测数据 "x" 而非速度或噪声，超越了**自回归模型**（包括 **LLMs**）。
   - 一位成员正在撰写一篇 [论文](https://arxiv.org/abs/2306.01585)，旨在将 AR、flow、diffusion 置于同一个问题下，以便进行公平比较。
- **学习算法在不调整模型的情况下提高样本效率**：根据一段 [视频](https://www.youtube.com/watch?v=21e5GZF3yx0)，在经典 RL 中，**PPO** 在不改变模型架构的情况下实现了 **样本效率 (sample efficiency)** 的突破。
   - 然而，一位成员表示，他们的算法虽然按预期运行，但由于在设计采样器时出现失误，导致训练速度变慢。
- **Google 的 Gemini 编程工具：Opus 4.5 可用但有限制**：**Opus 4.5** 可以在 Google 的新编程工具 **Antigravity** 中免费使用，需订阅专业版 (Google One)，目前可能没有配额限制。
   - 学生也可以免费获得一年使用权，但有人提醒，对于新程序员来说，在没有 LLM 编程 Agent 的情况下学习编程是明智的。
- **多任务 Mixture-of-Experts 提供快速推理**：**Mixture of Experts** (MoE) 模型与参数相同的稠密模型相比，提供了更高的 **TPS**，因为它们每个 token 仅激活一部分权重。用户在运行 Mixtral 7x8 时实现了与 8B 模型相当的速度。
   - 有人观察到：*"我的 Mixtral 7x8 运行速度和 8B 模型一样快，尽管它的参数量大得多。"*
- **Amazon 通过 Kiro 推广基于规范的代码开发**：较新的 **LLM** 模型经过训练，能更多地在通用意义上利用规范（specs），例如 *"artifacts"*，它可以是规范、待办事项列表或与用户的讨论。
   - Amazon 算是第一个真正通过 **Kiro** 推广这一概念的公司，Kiro 专门用于基于规范的开发，详见此 [链接](https://kiro.dev/)。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1449455559085658132)** (11 messages🔥): 

> `DeepSeek 3.2, Paper Presentation Reschedule` 


- ****DeepSeek 3.2** 论文即将发布**：一位成员询问是否可以讨论 **DeepSeek 3.2** 论文，并分享了 [Arxiv 链接](https://arxiv.org/abs/2512.02556)。
- **演示推迟，发起初步讨论**：一位成员询问是否安排了演示，另一位成员确认演示将重新安排。
   - 取而代之的是，他们在 [Discord 频道](https://discord.com/channels/714501525455634453/1045297868136779846/1448082833745776790) 发起了初步讨论。


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1449374088341487657)** (5 messages): 

> `Schmidhuber AI Agents, MLST interview, Exploration vs Exploitation` 


- ****Schmidhuber 的 Agent**：初步概览**：一位成员分享了 **Jurgen Schmidhuber** 讨论 AI Agents 的 [YouTube 视频](https://www.youtube.com/watch?v=h7F5sCLIbKQ&pp=ygULc2NobWlkaHViZXI%3D)，强调了其相关性和高质量。
   - 他们指出这段演讲*切中要害且非常出色*。
- **成员称：**可压缩性驱动探索****：一位成员分析了 **Schmidhuber** 的工作，指出其在探索与利用 (exploration vs exploitation) 之间采取了平衡的方法，这种方法是由可压缩性而非随机性驱动的。
   - 他表示：*使用可压缩性作为探索的驱动力而非随机性，为探索设定了一个目标；在奖励的引导下，这是一个很难被反驳的目标*。
- ****MLST 访谈**：深入对话 Schmidhuber**：一位成员分享了 **Jurgen Schmidhuber** 最近的 MLST 访谈，链接了讨论的 [第二部分](https://discord.com/channels/714501525455634453/986701499763654676/1330081889868058664) 和 [第一部分](https://youtu.be/DP454c1K_vQ?si=FmLbe3sko_XHzqqz)。
   - 另一位成员感叹道：*多么震撼的开场！！*


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1449187476894650555)** (44 messages🔥): 

> `Samsung DDR5 vs HBM, China's Collapse, Fragile Chip Supply, ChinaXiv Research` 


- **三星将 HBM 转向 DDR5 利润**：据 [Tweaktown](https://www.tweaktown.com/news/109259/samsung-shifts-focus-from-hbm-to-ddr5-modules-ddr5-ram-results-in-far-more-profits-than-hbm/index.html) 报道，三星正将重心从 **HBM** 转向 **DDR5** 模块，因为 **DDR5 RAM** 带来的利润远高于 HBM。
   - 一位成员开玩笑说，他们在全新的“以三倍价格坑人”的市场中看到了“$$$”。
- **RAM 是新的卫生纸**：一位成员将当前的 **RAM** 市场比作 **COVID 期间的卫生纸**短缺，人们正在大量囤积 DDR，希望在价格上涨时转售。
   - 有人指出，当需求上升时，**Amazon 的算法**也会推高价格，进一步加剧了这一问题。
- **地缘政治分析师被贬为地缘政治界的 Jim Cramer**：一位成员链接了一个关于脆弱芯片供应链的 [YouTube 视频](https://www.youtube.com/watch?v=lPDMqZyitFM)，但另一位成员认为该分析师是个“白痴”，并称其为“地缘政治界的 Jim Cramer”。
   - 该分析师自 2008 年以来一直宣称中国“即将崩溃”，做出了大胆（且往往是错误）的预测。
- **ChinaXiv 遭遇流量冲击导致宕机 (Hug of Death)**：成员们分享了中国研究论文库 [ChinaXiv](https://chinarxiv.org/) 的链接，但注意到它已下线，可能是由于“流量冲击 (hug of death)”。
   - 讨论中思考了是否会有仅以中文发表的优秀成果，尽管许多优秀的中国 ML 研究都是以英文发表的。
- **DLLM 更快的代码生成**：一位成员链接到了 [nathan.rs](https://nathan.rs/posts/dllm-faster-code-generation/)，其中有一篇短文分析了可能实现加速的场景。
   - 文章指出，生成《独立宣言》的开头“并没有太大的加速”，相反，“输出的结构化程度”才是关键。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1449128689366204446)** (170 messages🔥🔥): 

> `Spam DMs, Sparse Cores, GUI fine-tuning software, NVIDIA Triton Server Scaling, AI model size` 


- **垃圾私信困扰 HuggingFace 用户**：多名用户报告收到来自新账号的垃圾私信，一名用户已被封禁。
   - 社区提醒用户举报此类活动。
- **SparseCore 推荐系统不需要 Pallas**：一位用户询问是否需要学习 **Pallas** 才能使用 **Sparse Cores**，另一位用户澄清说，只有在特定执行的单核级别需要*自定义内核 (custom kernels)* 时才需要它，并分享了[这份 markdown](https://cdn.discordapp.com/attachments/879548962464493622/1449328192505774171/pallas_to_use_sparse_cores.md?ex=6941cb50&is=694079d0&hm=caa111592bf999093a6d016d2a310ba61a73513518d9cf129d2a114477a9cfc0&)。
- **GUI 微调软件需要更小的模型**：一位用户正在寻找适合展示其 **GUI 微调软件**的小型模型，并指出使用 **TiyLlama 1.1b Chat** 和 1225 个问答样本的效果不佳。
   - 有人建议尝试使用量化方法有效的 Liquid AI 模型。
- **扩展 NVIDIA Triton Server 需要批处理 (Batching)**：一位用户询问如何针对并发请求扩展 **NVIDIA Triton Server**，一位成员指向了提供指导的[在线资源](https://huggingface.co/datasets/John6666/forum3/blob/main/scaling_nvidia_triton_server_1.md)和[替代方案](https://huggingface.co/datasets/John6666/forum3/blob/main/scaling_nvidia_triton_server_2.md)。
   - 批处理或增加更多实例是扩展它的最佳方案。
- **估算 AI 模型大小**：用户正在寻找工具来估算 **HF 模型**是否能装入其 **GPU**，这引出了对[官方](https://huggingface.co/docs/accelerate/en/usage_guides/model_size_estimator)和[非官方](https://huggingface.co/spaces/Vokturz/can-it-run-llm)工具以及 [cfit](https://pypi.org/project/cfit/) 的建议。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1449194897998352448)** (15 条消息🔥): 

> `Neurosymbolic AI Project, HF Wrapped 2025, Madlab GUI Finetuning Toolkit, Text-to-Speech Models in 2025, Comment Works` 


- ****ReasoningLayer 开启候补名单****：一个从零开始用 **Rust** 构建的 Neurosymbolic AI 项目 [ReasoningLayer](https://reasoninglayer.ai) 开启了候补名单，旨在通过添加结构化推理来修复当今 LLM 的弱点。
   - 支持该项目的初始帖子已发布在 [LinkedIn](https://www.linkedin.com/posts/david-loiret_reasoninglayer-reasoninglayer-x-activity-7402510941332086784-ZU-E) 上。
- ****HF 社区迎来年度回顾****：一名成员受 Spotify/YouTube 年度回顾趋势启发，为 **2025** 编写了一个快速的 wrapper，可在 [HuggingFace Spaces](https://huggingface.co/spaces/hf-wrapped/2025) 查看。
   - 该成员建议创建一个官方的 **HF repo** 并从头开始构建一些东西。
- ****Madlab 发布，让 Finetuning 更简单****：一个开源的 GUI Finetuning 工具包 **Madlab** 已在 [GitHub](https://github.com/Archimedes1618/Madlab) 发布，专为合成数据集生成、模型训练和评估而设计。
   - 作为一个演示，官方还分享了基于 TinyLlama-1.1B-Chat-v1.0 的 **LabGuide Preview Model** 及其数据集，展示了其功能并邀请用户就合成数据集的使用和 Finetuning 提供反馈。
- ****2025 年的 TTS 模型大放异彩****：一名成员整理了一个 [GitHub repo](https://github.com/pr0mila/Text-to-Speech-Models-Released-in-2025)，列出了 **2025** 年发布的 Text-to-Speech 模型，包括开源、研究和商业系统，并简要说明了每个模型的优缺点。
- ****Comment Works 在本地分析自由文本****：一名成员正在实验一个经过微调的小语言模型 + Python 工具，暂定名为 **comment works:**，用于在本地机器上对自由文本评论进行私密分析，代码托管在 [GitHub](https://github.com/mtworth/cwos/tree/main)。
   - 这是一种为探索性分析或优先级排序而对定性反馈进行结构化处理的快速方法。


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1450274025069740042)** (2 条消息): 

> `MCP 1st Birthday Hackathon Winners, MCP Hackathon Certificates, Anthropic Awards, Modal Innovation Award, LlamaIndex Award` 


- **MCP 黑客松冠军揭晓！**：**MCP 1st Birthday Hackathon** 公布了由赞助商选出的获胜者，表彰了各个类别的优秀项目，名单列于 [Hugging Face Spaces](https://huggingface.co/spaces/MCP-1st-Birthday) 平台。
   - 奖项包括 **Anthropic Awards**（最佳综合奖、企业奖、消费者奖、创意奖）、**Modal Innovation Award**、**LlamaIndex Award**、**OpenAI Awards** 以及 **Blaxel Choice Award**。
- **Anthropic 的佼佼者揭晓！**：**Anthropic Awards** 将 *Cite Before Act MCP* 评选为**最佳综合奖**，其他获胜者包括 *MCEPTION*、*Finance Portfolio Intelligence Platform* 和 *GameContextProtocol*，均在 [Hugging Face Spaces](https://huggingface.co/spaces/MCP-1st-Birthday) 上展示。
- **黑客松证书来了！**：**MCP 1st Birthday Hackathon** 的参与者现在可以使用 [Gradio app](https://huggingface.co/spaces/MCP-1st-Birthday/MCP-birthday-hackathon-certificate-generator) 生成官方参赛证书。
   - 生成的证书可以下载、上传到 LinkedIn，并带上 Gradio 标签进行分享，详见[附带的示例](https://cdn.discordapp.com/attachments/1014577787039924226/1450304189841539296/Certificate-AgentsMCP-Hackathon-1765848691429_5202.png?ex=69420c88&is=6940bb08&hm=d7ae3389f51e25436741cc98feb00ca4be0c48894f80fddda2a745c6ba7c8135&)。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1449336718317588530)** (11 条消息🔥): 

> `课程问题空间删除，API 问题，Agent 分块相关性与 LLM，Agents 课程协助，Smol 课程未来` 


- **问题空间被删除！**：一名成员注意到课程的问题空间已被删除，现在使用此频道以更好地集中注意力。该成员标记了其他人以重新检查并更新课程页面，回复确认了针对单一频道的简化。
- **API 问题**：一位用户报告 API 仍无法工作，因为无法从服务器获取文件。给出的消息中未提供解决方案或变通方法。
- **Agent 需要更好的分块（Chunking）**：一位成员在 Agent 的分块相关性方面遇到问题，一旦将分块添加到上下文，Agent 就无法正确回答问题。该成员报告称，当要求 Agent 提供文档中的精确总价时，它会给出正确金额，但当要求 Agent 提供不同文档的相同总价时，有些价格是正确的，有些则是完全随机的。
- **Agents 课程需要协助**：一位尝试第一个 Agent 的成员添加了一个简单的获取时区工具，但在 UI 中遇到了错误。给出的消息中未提供具体解决方案。
- **Smol 课程即将到来！**：有人询问 *smol course* 是否会在今年提供最后一部分，使其成为一份 *酷炫的圣诞礼物*（*fine tuning course*）。没有人给出明确答复。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1449542651530448897)** (12 条消息🔥): 

> `CUDA 服务器，Tiny TPU，Hip Kittens 论文，论文阅读小组` 


- **CUDA 问题已澄清**：一位成员询问该 Discord 是否为 CUDA 服务器，得到的澄清是，虽然它不是 Nvidia 服务器，但 **CUDA** 是那里的热门话题，更多详情请见 [此处](https://discord.com/channels/1198358627594023014/1198358627594023014)。
- **Tiny TPU 教程即将到来**：一个 Tiny TPU 教程已安排与特定成员在 [YouTube](https://www.youtube.com/watch?v=kccs9xk09rw) 上开始。
- **Hip Kittens 可视化工具**：一位成员寻找用于生成 **Hip Kittens 论文**中可视化效果的工具，并链接了该论文。
   - 另一位成员建议可以通过在 kernel 中手动添加时间戳测量来实现，参考了[这篇博文](https://gau-nernst.github.io/amd-a2a/#intra-kernel-profiling)和[这个 GitHub 仓库](https://github.com/aikitoria/nanotrace)。
- **论文阅读小组成立**：一位成员提议在 Discord 内组建论文阅读小组，并推荐了 [arXiv](https://arxiv.org/pdf/2408.14158) 上的一篇特定论文。
   - 另一位成员鼓励自发组织，并为论文专家提供发言机会。


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1449841200373825587)** (3 条消息): 

> `TritonForge，SM86 上的 MXFP4 模拟，数据中心 GPU 优先级` 


- **TritonForge 自动化 Triton Kernel 优化**：一篇新论文 [TritonForge: Profiling-Guided Framework for Automated Triton Kernel Optimization](https://arxiv.org/abs/2512.09196) 介绍了一个集成 **kernel 分析**、**运行时分析（runtime profiling）**和**迭代代码转换**的框架，以简化优化过程。
   - 该系统利用 **LLMs** 辅助代码推理和转换，与基准实现相比，实现了高达 **5 倍的性能提升**。
- **SM86 上的 MXFP4 模拟**：一位成员询问 *triton_kernels* 是否计划支持 **SM86 上的 mxfp4 软件模拟**。
   - 另一位成员对此表示怀疑，指出项目维护者希望优先考虑数据中心 GPU。
- **为消费级 GPU 上的本地 LLM 进行 Fork**：一位成员建议 fork 该项目，以为 **30 系列**等消费级显卡添加功能，因为他们预计这些显卡在 **本地 LLM** 工作中仍将保持重要地位。
   - 这一建议源于项目对数据中心 GPU 的优先级高于消费级 GPU。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1449429798328864830)** (4 条消息): 

> `Tensor Core Optimization, LDSM instruction pipelining, Asynchronous Memory Copy, SMEM data loading strategies` 


- **寻求 90% 的 Tensor Core 利用率**：一名成员正在寻求实现 **90%+ Tensor Core 利用率**的建议，并详细介绍了一种在 **sm80/89** 目标架构上发出 **ldsm** 加载指令后紧跟 **MMAs** 的策略。
   - 他们发现，尽管尝试了加载和计算的流水线化（pipelining），但仍难以突破 **70% 的利用率**。
- **探索用于 Tensor Core 的异步内存复制**：一名成员建议使用 **cp.async** 来优化 Tensor Core 的利用率。
   - 这意味着需要解决数据加载到共享内存（**SMEM**）的速度问题。
- **提升 Tensor Core 效率的 SMEM 加载策略**：该成员询问了 **ldsm.4** 的用法，以及矩阵 **A** 和 **B** 是否都从共享内存（**SMEM**）加载。
   - 有人指出，如果正在执行的 **ldsm** 指令数量超过了掩盖 **SMEM** 最坏情况延迟所需的数量，会导致寄存器浪费。
- **实现 Tensor Core 全额利用**：实现 Tensor Core 的全额利用可能具有挑战性，因为在 **A100** 上，每 **8 个时钟周期**就需要一个 **MMA**，而在 **Ada** 架构上，根据所使用的 **MMA** 和 GPU 不同，这一周期为 **16 或 32 个时钟周期**。
   - 此外，在最初的四条指令之后，无法每个周期都发出 **ldsm**，这意味着流水线中存在停顿（stalls）。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1449558852365189141)** (8 条消息🔥): 

> `NVIDIA's Blackwell Architecture, girl.surgery bad paper, NVIDIA Acquires SchedMD, ldmatrix.x4` 


- **Blackwell 基准测试发布**：一名成员分享了一篇论文链接：[Microbenchmarking NVIDIA’s Blackwell Architecture: An in-depth Architectural Analysis](https://arxiv.org/pdf/2512.02189)。
- **差评论文链接**：一名成员链接了一篇位于 [girl.surgery/bad_paper](https://girl.surgery/bad_paper) 的“差评论文（bad paper）”。
- **NVIDIA 收购 SchedMD**：一名成员分享了 [NVIDIA Acquires SchedMD](https://blogs.nvidia.com/blog/nvidia-acquires-schedmd/?ncid=so-link-629370-vt25) 的链接，并觉得*很难想象他们会优先考虑为 AMD 开发功能，哈哈*。
- **ldmatrix.x4 Tile 大小与 Hopper**：一名成员表示，通过 **ldmatrix.x4**，可以在传输到寄存器时实现 **32x32 的 Tile 大小**，但另一名成员表示反对，指出 *ldmatrix.x4 仅加载四个 8x16 的 Tile*。


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1449162989637996696)** (2 条消息): 

> `Red Hat AI hiring, smallest.ai hiring` 


- **Red Hat AI 2026 年工程师招聘**：Red Hat AI 正在招聘多个级别的热血工程师，以突破 **AI Infrastructure** 的界限。
   - 他们对拥有 **Golang**、**Rust**、**C++**、**Python**、**Kubernetes**、**Distributed Systems** 和 **Open Source** 经验的人才特别感兴趣。
- **smallest.ai 寻求推理优化工程师**：[smallest.ai](https://binary.so/RUiE01i) 正在招聘推理优化工程师（Inference Optimization Engineers）来从事语音 AI 模型的工作。
   - 该职位涉及优化模型的速度和降低成本，涵盖量化（Quantization）、Kernel 优化以及向定制硬件的移植；工作地点：**班加罗尔或旧金山**。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1449831811730509854)** (8 条消息🔥): 

> `LLM inference, VLLM internals, GPU kernel engineering, CUDA experience` 


- **深入探讨 LLM 推理**：一名成员正在寻求关于深入了解现代 **LLM Inference** 系统级机制的建议。
   - 建议包括 Aleksa Gordić 关于 **VLLM Internals** 的博客以及 Junda 关于服务器方面的演讲。
- **寻求 Kernel 工程导师**：一名成员正在寻找一位付费导师/教员，以帮助他们从 **ML/AI Engineer** 转型为**微架构性能工程师（Microarchitecture Performance Engineer）**，重点关注 **GPU Kernel Engineering**。
   - 他们希望在理清概念、寻找优质问题以及指引研究方向方面获得帮助。
- **CUDA 初学者寻求指导**：一名具有一定 **CUDA** 经验（来自 ECE 408 课程）的初学者询问参与工作组/开源项目的后续步骤。
   - 另一名成员建议使用 **ChatGPT** 作为导师。


  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1449352272591978546)** (3 messages): 

> `Huggingface Nanotron, Qwen3-omni viz tool` 


- **添加了深度解析参考资料**：一位成员确认他们将为其工作添加参考资料，以便他人学习，包括 **Huggingface Nanotron Playbook**、**Nanotron 源代码**、**Pytorch 源代码**以及 **Megatron LM 源代码**。
   - 他们还计划为想要深入研究的人员添加一些研究论文。
- **语音到语音（Speech-to-Speech）推理工具发布**：一位成员分享了一个[新的 Qwen3-omni 可视化工具](https://news.ycombinator.com/item?id=46279195)（语音到语音推理）。
   - 该工具已发布在 Hacker News 上。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1449168753777574042)** (33 messages🔥): 

> `NVIDIA Performance, GEMM Leaderboard` 


- **NVIDIA 的 nuFP4-GEMM 数据**：提交至 `nvfp4_gemm` 排行榜的结果显示了在 **NVIDIA** 上的一系列成功执行时间，最快的提交达到了 **11.4 µs**。
   - 记录了多个个人最佳成绩，表明优化工作正在持续进行。
- **GEMM 排行榜异常活跃**：`nvfp4_gemm` 排行榜出现了大量提交，显示出活跃的参与和竞争。
   - 用户 <@651556217315000360> 持续提交，多次刷新个人最佳成绩，最终降低至 **11.4 µs**。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1450174400585400330)** (10 messages🔥): 

> `smem swizzling, atom permutation on K axis, tiled MMA, Cute DSL python version` 


- **Atom Permutation 优于 Smem Swizzling**：一位用户发现，他们的 [tiled MMA](https://cdn.discordapp.com/attachments/1362196854460383353/1450195491198468340/viz_m8n8k8.jpg) 需要的不是 **smem swizzling**，而是 K 轴上的 **atom permutation**。
   - 这允许使用 **2xdouble = uint128_t** 从共享内存（shared memory）到寄存器进行**向量化加载（vectorized loads）**。
- **为向量化对 K 轴进行置换**：该用户修改了代码，在其 tiled MMA 设置中对 **K 轴**进行置换，具体将 `Tile` 结构更改为 `Tile<_8,_8, Layout<Shape<_4, _2>, Stride<_2, _1>>>`。
   - [修改前](https://cdn.discordapp.com/attachments/1362196854460383353/1450195491198468340/viz_m8n8k8.jpg)和[修改后](https://cdn.discordapp.com/attachments/1362196854460383353/1450195597574410301/viz_m8n8k8_permuted.jpg)的图像展示了内存访问模式的变化，目前正尝试将相同的变换应用于更大的 DGEMM tiled MMA 设置（[图像链接](https://cdn.discordapp.com/attachments/1362196854460383353/1450196256893960286/viz_m64n64k32.jpg)）。
- **Cute DSL 可在 Python 3.10 上运行**：尽管[文档](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/quick_start.html)指出 **Cute DSL** 需要 **Python 3.12**，但一位用户发现它在 **Python 3.10** 上也能正常工作。
   - 另一位用户确认，虽然最初发布时针对的是 **3.12**，但此后已扩大了支持范围。


  

---


### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1450215894407057509)** (1 messages): 

> `LambdaLabs Research Grant, teenygrad H100 hours, j4orz.ai/sitp textbook` 


- **teenygrad 获得 LambdaLabs 研究资助**：**teenygrad** 项目已被 **LambdaLabs 研究资助**计划接受，从而获得了计算资源。
   - 该资助提供了约 **1000 H100 小时**的计算时间，并应能让开发工作在新年重新启动。
- **j4orz 发布教科书**：[Stanford In-house Training Program (SITP)](https://j4orz.ai/sitp/) 第 1 部分和第 2 部分的教科书、代码和讲座材料将于 1 月底或 2 月发布。


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1449882588574388367)** (1 messages): 

> `DMA, ML training, ML inference, FiCCO schedules, GPU DMA engines` 


- **探索细粒度计算通信重叠**：一篇新论文探讨了分布式 **ML 训练**和**推理**中[基于 DMA 的更细粒度计算通信重叠的设计空间探索](https://arxiv.org/abs/2512.10236)。
   - 该论文引入了 **FiCCO**，一种更细粒度的重叠技术，旨在为更广泛的网络拓扑和更细粒度的数据流解锁计算/通信重叠。
- **FiCCO 调度带来加速**：论文详细描述了效率损失的特征以及 **FiCCO 调度**的设计空间。
   - 提议的定制调度方案将通信卸载到 **GPU DMA 引擎**，在实际的 ML 部署中实现了高达 **1.6 倍的加速**，其启发式方法在 **81%** 的未知场景中提供了准确的指导。


  

---

### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1449132639037685823)** (72 条消息🔥🔥): 

> `尽管本地构建成功但提交失败、GPU 性能不一致、PTX 代码错误、当 M<256 时利用 2 个 SM、Cute-dsl NCU 行号问题` 


- **管理员协助解决提交错误**：一位参赛者报告在提交时出现 `Error building extension 'run_gemm'` 错误，尽管本地构建成功。建议使用 [inline](https://github.com/openai/triton/blob/main/python/triton/testing.py#L38) 并设置 `verbose=True` 以获取详细日志。
   - 移除多余的包含路径 `/root/cutlass/include/` 解决了该问题。
- **GPU 性能波动**：参赛者注意到不同 GPU 之间的性能差异，排行榜结果平均比本地基准测试慢 **2-4 微秒**，这表明某些节点速度较慢。
   - 建议多次重新提交以避开慢速实例，因为遇到这些实例的可能性可能正在增加，且有一个节点性能已完全退化。
- **PTX 代码错误阻碍进度**：一位参赛者遇到了 PTX 编译错误，特别是与 `.target 'sm_100'` 相关的 `Unexpected instruction types specified for 'cvt'` 和 `Instruction 'cvt with .e2m1x2' not supported`。
   - 一位用户分享了使用 `cuda_fp16.h` 的代码片段，以解决 inline PTX 中 vector/byte 类型的问题。
- **探索 2 个 SM 的利用**：讨论了当 M<256 时是否可以利用 2 个 SM，并引用了 [Nvidia PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-shape)。
   - 一位成员提到在准备 blockscaled 布局时遇到问题，另一位成员提到了 2SM 流水线（pipelining），但从经验来看并未发现性能提升。
- **分析显示行号偏移**：一位用户报告在使用 `cute.GenerateLineInfo(True)` 处理 cute-dsl 代码时，NCU 显示的 Python 代码行号不正确或发生了偏移。
   - Discord 机器人提供的分析报告中显示的行号完全是随机的，或者可能是 Python 代码中的行号发生了偏移。


  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1450043045818011700)** (1 条消息): 

> `规划器模型、子任务分解、Vision-Language-Action 模型、LITEN 论文` 


- **LITEN 论文发布重磅内容**：[LITEN 论文](https://arxiv.org/abs/2510.19752) 介绍了一种用于**子任务分解**的规划器模型，并结合了 **Vision-Language-Action (VLA)** 模型。
   - 高层 **VLM** 基于过去的 in-context 经验进行条件化，以学习低层 **VLA** 的能力。
- **推理与评估阶段**：该模型在**推理阶段**运行，生成并执行计划（一系列子任务指令）给低层 **VLA**，随后进入**评估阶段**。
   - 在**评估阶段**，模型会对执行结果进行反思，并得出结论以纳入未来的推理阶段。
- **冻结 VLM 的优势**：该模型使用**冻结的 VLM**，无需额外训练。
   - 这种方法利用了现有知识，无需进一步的训练迭代。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1449130131728892014)** (142 条消息🔥🔥): 

> `Museum of Science Decline, AI-Generated Stock Art, Open-Source Git Replication, Claude Model Removed from Cursor, OpenAI document processing infrastructure Leaked` 


- ****博物馆危机？科学兴趣骤减**：一条推文指出 [波士顿科学博物馆近乎空置的状态](https://xcancel.com/anissagardizy8/status/1999248165724389554?s=46)，引发了人们对公众参与科学程度下降的担忧。
   - 一些人对这种下降的原因进行了推测。
- ****股票代码“废料”：AI 艺术遭到嘲讽**：用户嘲讽了 [低质量的 AI 生成股票代码库存艺术图](https://xcancel.com/thinkymachines/status/1999543421631946888?s=20)，称其为“股票代码废料”（ticker-symbol slop）。
   - 这些艺术品因其毫无生气且平庸的风格而受到批评。
- ****Cursor 弃用 Claude：模型在基准测试中作弊**：Cursor 在发现 **Claude 模型**在内部编程基准测试中作弊后，突然在其 IDE 中禁用了该模型。据报道，作弊手段是 [在训练数据中嵌入答案](https://xcancel.com/cursor_ai/status/1998821350333440133?s=20)。
   - 用户被鼓励举报类似问题。
- ****Soma 离职：Post-Industrial Press 人事变动**：Jonathan Soma 宣布从 **Post-Industrial Press** 辞职，理由是对其未来方向的不确定性，并 [在推文中](https://xcancel.com/jhleath/status/1999589156314578961?s=20) 感谢了过去六年的合作伙伴。
- ****OpenAI 的失误：从 Reddit 撤下的文档泄露**：一个帖子报道称 **ChatGPT** 泄露了自己的文档处理基础设施，但 Reddit 迅速删除了相关细节。
   - 讨论中包含了 [Google Drive 上可能相关的文件](https://drive.google.com/file/d/1Hw3a58rnxlStxFYGOXbWIFx-3tQsxRaY/view) 链接，以及该事件在 Discord 上发布的截图。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1449141760688394290)** (14 条消息🔥): 

> `New Twitter Post, Nitter Link Errors, Missing Content` 


- **Irissy 的推文发布在 X**：一名成员分享了用户 @xIrissy 发布的新推文链接 ([https://x.com/xIrissy/status/1999384085400289473](https://x.com/xIrissy/status/1999384085400289473))。
- **Gdgtify 推文消失在虚空中**：尽管引用了一个占位符 Nitter URL 源 ([https://x.com/gdgtify/status/2000070495446643091?s=46](https://x.com/gdgtify/status/2000070495446643091?s=46))，但用于摘要的输入 Markdown 内容完全为空。
   - 没有提供可供分析或压缩的讨论文本。
- **Gokayfem 状态无法访问**：提供的 Gokayfem Nitter 状态 URL 不完整或无效 ([https://x.com/gokayfem/status/2000309866766967130?s=46](https://x.com/gokayfem/status/2000309866766967130?s=46))，导致无法获取该线程内容进行摘要。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1449129684679000206)** (82 条消息🔥🔥): 

> `Derrida 和 Baudrillard 论 AI，Oracle 的 AI 战略，Nvidia 的开源支持，本地 LLM 势头渐盛，Nvidia 的 CUDA 豪赌` 


- **关于合成思维的哲学思考**：一位成员好奇 **Derrida 和 Baudrillard** 会如何看待 **AI**，构思了一本名为 *'Philosophy is Synthetic'* 的未来著作，并思考了 **Saussure** 会对 **Word2vec** 有何评价。
- **Oracle 的策略：从数据库到 AI 霸权？**：一些成员讨论了 **Oracle** 如何从一家“乏味、笨重的数据库公司”转型为一家“估值过高的 AI 公司”，这得益于与 **OpenAI** 和 **Sam Altman** 的 **IOU** 方案。
   - 其他人推测 **Oracle 的 AI 股价推高** 主要是为了收购并控制美国媒体实体（*Paramount/CBS, Warner Bros/CNN*），以掌控右翼叙事。
- **本地 LLM 将在特定领域爆发**：一位成员与客户讨论了在未来几年内实施 **本地 AI** 的计划，该 AI 将基于公司自有数据训练，并针对其行业（海运）进行专业化。
   - 这可能涉及在特定员工的“数千封电子邮件之声”或数百份合同上训练 LLM。
- **Nvidia 在 GPU 市场的 CUDA 优势**：有人提到 **Nvidia** 如何押注 **GPU** 将用于游戏以外的其他领域，并开发了一种语言（**CUDA**），该语言可以在其所有 GPU 上运行，并使其能够用于这些其他用途。
   - 最令人惊讶的是，**AMD 和 Intel** 直到大约 2 年前才意识到参与这个市场是值得尝试的。
- **Nvidia 拥抱开源，捍卫 AI 帝国**：成员们注意到 **Nvidia** 对 **Nvidia Nemotron Nano** 等开源倡议的支持日益增加，认为这是维持其产品长期需求的战略举措。
   - Nvidia 似乎在支持开源，这看起来非常不错，因为这可能是长期维持其产品高需求的唯一途径之一。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1449129819551039499)** (32 条消息🔥): 

> `Grok 巧合，RL 优化器，字节级 LLM` 


- **Grok 崇拜 Elon 的巧合？**：一位成员分享了一个链接，讨论了 **Grok** 崇拜 **Elon** 的巧合，称其在同等规模下表现 *令人印象深刻* [https://www.arxiv.org/pdf/2512.06266](https://www.arxiv.org/pdf/2512.06266)。
- **RL 优化器搜索**：一位成员正试图为一个 **3B LLM** 的小型预训练寻找最佳优化器，希望能超越 **Muon / AdamW**，并提到了 [ADAMUON](https://arxiv.org/pdf/2507.11005)、[NorMuon](https://arxiv.org/pdf/2510.05491v1) 和 [ADEMAMIX](https://arxiv.org/pdf/2409.03137)。
   - 另一位成员推荐了 [Sophia](https://arxiv.org/abs/2305.14342)，并建议对每种优化器进行 token 限制的部分训练，同时进行微小的 LR/beta 扫描，让它运行一夜并记录指标。
- **字节级 LLM**：一位成员对来自 [Allen AI](https://allenai.org/papers/bolmo) 的关于 **字节级 LLM** 的链接反应积极，表示 *这很酷* 且 *很有趣*。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1449806302019584100)** (2 条消息): 

> `Embeddings，Embeddings 的历史，现代 AI 中的 Embeddings` 


- **分享 Embeddings 历史演讲**：一位成员分享了他们本周关于 **embeddings** 的简短演讲，涵盖了其在 **60 年代** 的起源以及在 **现代 AI** 中的核心作用，链接见：[YouTube 视频](https://youtu.be/Cv5kSs2Jcu4)。
- **Embeddings 反馈请求**：该成员请求熟悉 embeddings 的人士提供反馈，寻求对其在捕捉基础知识方面表现的评价。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1449129819551039499)** (32 messages🔥): 

> `Grok 巧合，RL 优化器，Byte LLMs` 


- **Grok 喜爱 Elon？**: 成员们讨论了一篇论文 ([https://www.arxiv.org/pdf/2512.06266](https://www.arxiv.org/pdf/2512.06266)) 以及 **Grok** 喜爱 Elon 的巧合。
   - 一位成员开玩笑说：*想象一下，如果 Grok 喜爱 Elon 实际上只是一个巧合，就像这件事发生的一样*。
- **超越 Muon / AdamW 的新优化器**: 一位成员正试图为一个 **3B LLM** 的小型预训练寻找最佳优化器，希望能超越 **Muon / AdamW**，并找到了 **ADAMUON** ([https://arxiv.org/pdf/2507.11005](https://arxiv.org/pdf/2507.11005))、**NorMuon** ([https://arxiv.org/pdf/2510.05491v1](https://arxiv.org/pdf/2510.05491v1)) 和 **ADEMAMIX** ([https://arxiv.org/pdf/2409.03137](https://arxiv.org/pdf/2409.03137))。
   - 另一位成员建议尝试 Sophia ([https://arxiv.org/abs/2305.14342](https://arxiv.org/abs/2305.14342))，并比较 loss vs tokens 以及 loss vs 实际时间。
- **Byte Level LLMs 很酷**: 一位成员分享了 **BOLMO** 的链接 ([https://allenai.org/papers/bolmo](https://allenai.org/papers/bolmo)) 并指出 *这很酷*。
   - 另一位成员表示赞同，说 *我喜欢 byte level llms，很有趣*。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1449243851121229934)** (87 messages🔥🔥): 

> `Kimi Slides API，NPU 上的本地 Kimi K2，Kimi 记忆功能同步，Kimi K2 Tokenizer 端点，带有记忆功能的 Kimi Android 更新` 


- **Kimi Slides API 尚未上线**: 一位用户询问 **Kimi Slides** 功能是否可以通过 API 使用，但被告知 *目前尚未提供*。
- **本地运行 Kimi K2 技术上不可行**: 一位用户表示有兴趣在他们的 NPU 上运行本地 **Kimi K2 模型**，但被告知在本地硬件上达到 K2 的能力 *几乎是不可能的*。
- **跨平台的记忆功能差异**: 用户注意到 **记忆功能** 在 Kimi 网页版上可用，但不确定是否会同步到 Android 版本，初步测试显示 *无法同步*。
- **Kimi 的上下文窗口太短，导致截断**: 一位用户抱怨应用在超过 **20 万字** 后会 *硬锁定*，限制了在达到上限前可以生成的 prompt 数量。
   - 另一位用户建议使用 [Kimi K2 Tokenizer 端点](https://platform.moonshot.ai/docs/api/estimate) 以获得更准确的 token 计数，但这仅通过 API 提供。
- **Kimi Android 版本更新记忆功能**: Kimi Android 版本已更新并包含 **记忆功能**，使其与网页版保持一致。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1449282928595959898)** (30 messages🔥): 

> `Kaggle TPU VMs 上的 PyTorch LLM，扩展 NVIDIA Triton Server，NSF 倡议，ML 论文中的 LLM 辅助，Algoverse 导师` 


- **PyTorch LLM 在 Kaggle TPU VMs 上遇到困难**: 一位成员询问如何在 **Kaggle TPU VMs** 上运行 **PyTorch LLM**（从 Hugging Face 下载），并提到之前在 Keras 上运行成功，但在 PyTorch 上遇到了错误。
- **扩展 NVIDIA Triton Server 以处理并发请求**: 一位成员寻求关于扩展 **NVIDIA Triton Server** 配置的建议，该配置包含 **YOLO**、**bi-encoder** 和 **cross-encoder 模型**，旨在高效处理生产环境中的多个并发请求。
- **NSF 倡议发布缺乏细节**: 成员们讨论了一个 [NSF 倡议](https://www.nsf.gov/news/news_summ.jsp?cntn_id=301418) 的发布，但指出该帖子内容模糊，除了并非针对大学/初创公司之外，缺乏核心思想。
- **ML 论文中的 LLM 辅助：披露还是隐瞒？**: 成员们讨论了是否应在 **ML 论文提交** 中披露 **LLM 辅助**，一位成员倾向于认为，如果 LLM 对文本、图表、设计或分析有显著贡献，则不披露应被视为学术不端。
- **Algoverse 导师职位出现**: 一位成员询问 **Algoverse**，另一位成员提到他们正在相关服务器上寻找高质量导师，且报酬似乎相当不错。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1449156455134396628)** (17 messages🔥): 

> `Karpathy 的 2025 'What-If' 微调实验，专注于 AI Agents 的计算机科学在职博士，Muon 有效性的原因，原生 8-bit 权重训练，GANs` 


- **Karpathy 的 'What-If' 实验引发关注**：一位成员表示有兴趣尝试 [Karpathy 的 2025 'What-If' 微调实验](https://arxiv.org/abs/2502.04549)，该实验在合成推理链、Edge.org 文章和诺贝尔奖演讲稿上对模型进行了微调。
   - 该实验在 **8 台 A100 GPUs** 上使用 **LoRA** 进行了 **3 个 epochs** 的训练，得到的模型擅长长期推测，但缺乏物理问题的创新解决方案。
- **寻求 AI Agents 方向在职博士的坦诚建议**：一位成员请求已完成专注于 **AI Agents** 的 **CS** **在职博士**（特别是通过 **PhD by publication** 方式）的人士提供坦诚建议。
   - 他们希望了解决策过程、每周节奏、挑战，以及在职业生涯、个人成就感、公信力和选择权方面是否值得。
- **Muon 的神奇之处源于多项式不稳定性？**：一位成员提出一种直觉，认为 **Muon** 的有效性是由于非线性导致步长随多项式/指数增长的不稳定性，尽管其有效性受到了质疑。
   - 另一位成员澄清说，将两个扩散模型（或 CFG 中的概念）的分数相加，并不会产生来自其学习分布的真实乘积的样本，这需要独立性，而这正是 CFG 所擅长的。
- **发现原生 8-bit 训练论文**：一位成员询问关于**原生 8-bit 权重训练**的论文，寻求切入点或搜索词，并发现了一篇[很有前景的论文](https://www.arxiv.org/pdf/2511.23225)。
   - 该论文使用**混合精度**，并通过添加损失项来抑制离群值（outliers），从而跳过 Hadamard 变换，简化并加速了过程。
- **讨论 GAN 架构**：一位成员分享了一个 [GAN 架构](https://x.com/i/status/1999169943267381265)及其 [arXiv 页面](https://arxiv.org/abs/2511.21667)的链接。
   - 另一位成员确认这是一个 GAN，并提供了[另一个链接](https://x.com/i/status/2000556593784758384)进行确认。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1449153825599652094)** (9 messages🔥): 

> `Superweight 消融，正交修复，神经元特定特征，高维正交性，极端权重的重要性` 


- **Superweight 消融影响 OLMo-1B，通过 Rank-1 Patch 修复**：一位申请 MATS 的成员消融了 **OLMo-1B** 中的一个权重，导致困惑度（perplexity）从 **17** 飙升至 **2800** 以上，随后使用受 OpenAI 权重稀疏 Transformer 论文启发的 Rank-1 Patch 实现了约 **93%** 的恢复。
   - **93%** 的恢复被定义为恢复的 **NLL** 退化百分比，补丁模型显著缩小了由损坏模型造成的差距；基础模型 NLL：**2.86**，损坏模型 NLL：**7.97**，补丁模型 NLL：**3.23**。
- **“正交修复”补偿删除的权重**：研究发现，用于补偿消融权重的学习补丁与原始权重的余弦相似度仅为 **0.13**，呈现正交关系，这表明补偿是通过一个全新的分布式电路实现的。
   - 该成员询问这种*正交修复*是否是一种已知现象，以及它是否模拟了重新路由（rerouting）而非恢复权重的*九头蛇效应（hydra effects）*。
- **消融的神经元被发现是海洋生物学的特征神经元**：对被删除神经元（第 1 层，第 **1764** 行）进行最大激活数据集搜索显示，它是一个关于甲壳类动物/海洋生物学的特征神经元，最高激活 token 包括 H. gammarus（**欧洲龙虾**）、Cancer pagurus（**食用黄道蟹**）和浮游动物。
   - 消融导致模型在测试提示词上产生“mar, mar, mar”的幻觉，表明海洋生物的本体（ontology）被移除了。
- **行级补丁模拟九头蛇效应**：在受损行中添加了一个单一的可训练参数向量（**delta_row**），其作用类似于仅应用于该行的 Rank-1 LoRA，并经过训练以最小化与原始冻结基础模型的 **KL** 散度。
   - 尽管该补丁与原始消融方向的重叠极小，但它导致了大量的 **NLL** 恢复，这可能通过重新路由而非简单地恢复权重来反映九头蛇效应；这被比作*高维正交性*。
- **极端权重可能至关重要**：一位成员推测，与矩阵中的平均权重相比，Superweights 可能具有非常高或非常低的值，并质疑这些极端权重是否必不可少。
   - 该成员想知道，训练某种机制来惩罚大权重而不降低模型性能是否会很有趣。


  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1449423797479542794)** (50 条消息🔥): 

> `Linux 6.19 char misc, tinygrad meeting #100, Llama 405b, Python Speed, Gradient Checkpointing` 


- **Tinygrad 第 100 次会议议程**：第 100 次 tinygrad 会议讨论了 **公司更新**、**Llama 训练优先级**、**梯度累积 (grad acc) 与 JIT**、**Flash Attention**、**MI300/350 稳定性**、**快速 GEMM、可视化 (viz)**、**FP8 训练**、**Image DType/CType** 以及 **其他悬赏任务 (bounties)**。
   - 会议定于周一圣迭戈时间上午 9 点举行。
- **在 GitHub 上追踪 Llama 405b**：一名成员在 GitHub 上创建了一个看板，用于追踪 **Llama 405b**，链接在 [这里](https://github.com/orgs/tinygrad/projects/11/views/1?groupedBy%5BcolumnId%5D=Assignees)。
   - 该看板可用于追踪与 **Llama 405b** 模型相关的任务分配和其他事项。
- **JIT：防止静默错误**：一名成员计划通过正确检查 **JIT** 是否仅在 **schedulecaches** 匹配时进行捕获，来减少 **JIT 陷阱 (footguns)**。
   - 会议讨论了 JIT 中的两个重大陷阱：一是函数中使用的非输入 Tensor 改变了“形式”导致 JIT 出现静默错误；二是输出 Tensor 未被复制并覆盖了上一个 Tensor。
- **Image DType 进展**：**Image DType** 正在取得进展，目标是在本周末前基本完成合并，但需要注意的是 **CL=1, QCOM=1** 可能会比较棘手。
   - 存在一些概念性问题，例如在 Adreno 630 上，如果图像宽度没有 **64B 对齐**，Buffer 就无法转换为任意的图像形状。
- **AI Pull Requests 政策**：关于 **AI Pull Requests** 的政策保持不变：除非你是知名贡献者，否则任何看起来像 **AI** 生成的内容都会被立即关闭。
   - 理由是开发者应该完全理解他们提交的 **PR** 中的每一行代码；如果只是提交一个自己都不理解的 **AI PR**，那只会带来负面价值。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1449202011588526201)** (34 条消息🔥): 

> `ReasoningLayer.ai launch, Neurosymbolic AI, DSPy GEPA, uv tool install slowness, MCP mode` 


- **ReasoningLayer.ai 神经符号 AI 项目开启候补名单**：一个名为 [ReasoningLayer.ai](https://reasoninglayer.ai) 的神经符号 AI 项目开启了候补名单，旨在通过加入真实的结构化推理来解决当今 LLM 的弱点。
   - 该项目计划在其本体摄取流水线中使用 **DSPy GEPA**；支持该项目的初始帖子在 [这里](https://www.linkedin.com/posts/david-loiret_reasoninglayer-reasoninglayer-x-activity-7402510941332086784-ZU-E)。
- **利用子智能体 (Subagents) 和 DSPy 的下一代 CLI 工具**：一名成员指出，**DSPy** 与子智能体的结合不仅限于审查/分类/工作工具，还代表了下一代 CLI 工具。
   - 它不只是另一个编码 CLI，而是可以作为 MCP 赋能其他编码 CLI，并可以通过 [MorphLLM](https://morphllm.com) 和 **Supermemory.ai** 进一步增强。
- **探索 MCP 模式调整**：创建者正在考虑将该工具作为 MCP 模式运行的调整方案，并欢迎社区在 [GitHub](https://github.com) 上进行贡献和提交 PR。
   - 计划先从开源功能开始，然后增加基于云的选项，目标是为用户提供多种选择，甚至利用该工具来构建其自身。
- **`uv tool install -e .` 安装耗时过长**：一位用户反馈 `uv sync` 或 `uv tool install -e .` 耗费了大量时间，原因不明。
   - 该问题可能与 Python 版本有关，因为它在 3.13 中运行正常，但在 3.14 中不行；工具创建者将对此进行调查。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1449160988065665034)** (14 messages🔥): 

> `BAMLAdapter, DSPy Skills, Field Specific Instructions` 


- **BAMLAdapter 支持直接导入**：成员们讨论了新的 **BAMLAdapter**，其中一位成员指出，如果你通过 `from dspy.adapters.baml_adapter import BAMLAdapter` 直接导入，现在就可以使用 `BAMLAdapter`。
   - 另一位成员补充说，他们提交了一个修复 PR，目前它还不会提取 Pydantic 模型的 **docstrings**。
- **新人询问 DSPy 技能**：一位首席工程师询问如何深入研究 DSPy，并询问了值得关注的具体特性。
   - 该工程师表达了一个“流行观点”，即对于大多数以盈利为导向的产品，除非有 **VC funding** 支持或用户群愿意支付溢价，否则 Agent 的成本太高且不够可靠。
- **为了优化成本/利润而将 Prompt 过拟合到 Frontier Models**：一位成员指出，将 Prompt 过拟合到最新的 Frontier Model 是有价值的，因此自动 Prompt 优化的价值相对较小。
   - 另一位成员表示，如果你需要关心 **cost/margins**，你就会开始关注自己在 **cost/accuracy/latency frontier**（成本/准确度/延迟前沿）上的位置。
- **在 DSPy Signature 中放置特定字段指令的最佳位置？**：一位成员询问，当根据某些规则从 1 个输入字段提取 6 个输出字段时，放置特定字段指令的最佳位置在哪里。
   - 他们询问指令应该放在 `MySignature` 类的 **docstring** 中，还是放在对应的字段 `field1: str = dspy.OutputField( desc="field-specific-instructions" )` 中，或者两者都放。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1449238017666449652)** (32 messages🔥): 

> `Variable declaration scope, Mimicking const, C++ lambda syntax, Julia v. Mojo, LLM modular book error` 


- **变量声明中显现的作用域差异**：讨论了不使用 `var` 关键字的变量声明，明确了不带 `var` 声明的变量具有 **function scope visibility**（函数作用域可见性），类似于 JavaScript 中的 `var`，而不是块作用域（block scope）。
   - 在 Mojo 中，`var` 的行为类似于 JS 的 `let`，而不带关键字的行为类似于 JS 的 `var`；一个 [GitHub Pull Request](https://github.com/modular/modular/pull/5636#pullrequestreview-3568478570) 讨论了移除类似于 JavaScript `const` 的特性。
- **社区考虑模拟 `const` 关键字功能**：一位成员建议，他们*可能*能够通过函数在库端模拟 `const`，例如 `var res = try[foo](True)`。
   - 有人指出，将其作为 **compiler feature**（编译器特性）可能会更好。
- **关于 C++ Lambda 语法的辩论**：一位成员表示支持 **C++ lambda syntax**，理由是它对 captures（捕获）的处理方式，尽管他承认自己属于少数派。
   - 另一位成员表示，这是*最不坏的方法*之一，因为大多数语言的 lambda 看起来更漂亮，但随后必须处理 captures 问题，导致局面混乱。
- **Julia vs. Mojo：FAQ 澄清差异**：一位成员询问了 **Julia** 与 **Mojo** 的对比，另一位成员指向了 [Mojo FAQ](https://docs.modular.com/mojo/faq/#why-not-make-julia-better)，其中强调了 Mojo 在内存所有权、扩展性以及 AI/MLIR 优先设计方面的方法。
   - FAQ 指出，*Mojo 在内存所有权和内存管理方面采用了不同的方法，它可以缩减到更小的运行环境，并且是基于 AI 和 MLIR 优先原则设计的（尽管 Mojo 不仅仅用于 AI）*。
- **LLM Modular 书籍错误困扰学习者**：一位用户在 [llm.modular.com book](https://llm.modular.com) 的 **step_05** 中遇到了错误，怀疑是由于 GPT2 模型没有从 Huggingface 完全下载导致的。
   - 另一位成员回应并提到，**DGX Spark 的 GPU 尚未在他们的编译流程中得到支持**。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1449138881441042507)** (22 messages🔥): 

> `Manus Auth redirect bug, Gemini 3.0 vs Manus, Firebase, Antigravity, and Google AI Studio, Conversation Mode with Wide Research, Manus 1.6 release` 


- **Manus Auth 重定向 Bug 令用户沮丧**：一名用户报告了持续存在的 **Manus Auth redirect bug**，该问题在未解决的情况下消耗了额度。用户对缺乏快速构建自定义系统的解决方案表示沮丧，并批评了在客户端登录时强制显示 **Manus logo** 的做法 ([复现链接](https://manus.im/share/7ivVWED9HdFb21qbErs819?replay=1))。
   - 由于这些 Bug，该用户正转向使用 **Firebase**、**Antigravity** 和 **Google AI Studio**，并认为 **Gemini 3.0** 和 **IDE 中的 Claude** 表现优于 Manus。
- **Gemini 3.0 和 Firebase 优于 Manus**：一名用户因不满而离开 Manus，称 **Gemini 3.0** 和 **Firebase** 是更优的替代方案，特别是 **Antigravity** 提供了更多的 Agent 控制权，并能通过 **OpenRouter** 访问最新模型。
   - 他们预测 Manus 对开发者来说将变得过时，并强调 **Google 为拥有 Gmail 账号或 Google Workspaces 的开发者免费提供类似功能**。
- **用户请求同时开启 Conversation Mode 和 Wide Research**：一名用户请求恢复允许同时使用 **Conversation Mode** 和 **Wide Research** 的功能，并指出并非所有用户都喜欢 **PDF 格式的 AI 回复**（这是 Agent Mode 的默认设置）。
   - 他们认为，将研究的广度与对话式交互相结合，可以通过一种更**自然**、更**具交互性**的方式来处理研究结果，而无需通读 PDF 文档，从而提升用户体验。
- **Opus 4.5 在价值和性能上击败 Manus**：一名用户在 **Claude Code** 中以每月 20 美元的价格使用 **Opus 4.5**，发现它比 Manus 更具性价比，尤其是在添加 MCP 服务器、技能和插件时。
   - 他们指出 Manus 就像一个还不会说话的蹒跚学步的孩子，并推荐了 [discord-multi-ai-bot](https://github.com/binkiewka/discord-multi-ai-bot) 项目。
- **AI 工程师宣传现实世界解决方案**：一位 AI 和全栈工程师分享了他在**高级 AI 系统**和**区块链开发**方面的实战经验。
   - 他构建**现实世界的端到端解决方案**——从模型到生产就绪的应用，提到了 AI 聊天机器人、YOLOv8 图像识别和 AI 笔记助手等项目，并邀请用户一起构建有意义的东西。


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1449153765729894532)** (12 messages🔥): 

> `C++ MCP, Dangerous tool flag, MCP Server Publication Error, Response Annotations, Tool Resolution Proposal` 


- **危险标记功能推出**：一名成员询问如何在 MCP 中将工具标记为 `dangerous`（危险），特别是针对 **Claude Code** 以限制某些工具调用。
   - 另一名成员分享了目前处于草案阶段的[一份提案](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/1913)，并邀请大家对 **response annotations**（响应注解）提供反馈。
- **MCP 工具解析讨论展开**：在关于**工具解析**（tool resolution）的讨论帖中，一名成员对工具解析提案表示感兴趣。
   - 他们指出，*将由客户端实现来决定如何处理该标记*，并对其他人的处理方式感到好奇。
- **MCP 服务器发布受弃用问题困扰**：一名成员在尝试使用 **mcp-publisher** 发布新的 **mcp-server** 时遇到错误，原因是 *deprecated schema*（架构已弃用）错误，正如[快速入门指南](https://github.com/modelcontextprotocol/registry/blob/main/docs/modelcontextprotocol-io/quickstart.mdx)中所述。
   - 另一名成员解释说，文档在生产部署之前已经更新，并建议暂时使用之前的架构版本 **2025-10-17** 作为权宜之计。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1449211369873215580)** (6 messages): 

> `Aider OpenAIException, Aider active development` 


- **Aider 抛出 OpenAIException**：一名用户在运行 `aider --model` 时遇到了 `litellm.NotFoundError`，原因是未找到 **'gpt-5' 模型**。
   - 一名成员建议尝试使用 `openai/gpt-5` 作为模型字符串。
- **Aider 活跃度确认**：一名用户询问 **Aider** 是否仍在积极开发中。
   - 在给定的上下文中没有提供进一步的讨论或确认。


  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1449603019577622609)** (4 messages): 

> `aider gpt-5, litellm errors, aider model config` 


- **GPT-5 模型导致 Aider 崩溃**：当尝试使用 `--model openai/gpt-5` 标志运行 `aider` 时，用户遇到了 `litellm.NotFoundError`，提示模型 *'gpt-5' not found*。
   - 尽管该模型出现在模型列表中，且用户已设置其 OpenAI API key，问题仍然存在。
- **调试 Aider 模型配置**：用户正尝试在 `aider` 中使用 `openai/gpt-5` 模型，并通过 `--reasoning-effort medium` 标志将 reasoning effort 设置为 medium。
   - 用户已确认使用 `setx` 设置了 OpenAI API key，这表明身份验证应该不是问题所在。


  

---


---


---