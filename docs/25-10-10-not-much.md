---
companies:
- openai
- google-deepmind
- microsoft
- epoch-ai-research
- togethercompute
- nvidia
- mila
date: '2025-10-10T05:44:39.731046Z'
description: '以下是该文本的中文翻译：


  **FrontierMath Tier 4** 的测试结果显示，**GPT-5 Pro** 在推理准确率上略微领先于 **Gemini 2.5 Deep Think**，同时
  **Epoch AI Research** 澄清了此前关于题目泄露的疑虑。**Mila** 和 **微软** 提出了 **马尔可夫思维 (Markovian Thinking)**
  以提高推理效率，使模型能够以更少的计算量实现超过 2.4 万个 token 的推理。新研究表明，基座模型本身就蕴含推理机制，而“思考模型”则是学习如何有效地调用这些机制。在系统架构方面，**英伟达
  (NVIDIA) Blackwell** 结合 **vLLM** 在 InferenceMAX 中夺魁，吞吐量提升显著；同时，**Together AI** 的
  **ATLAS** 自适应投机解码实现了 4 倍的速度提升，并将强化学习 (RL) 训练时间缩短了 60% 以上。**SparseServe** 引入了带有 KV
  分层的动态稀疏注意力机制，极大地优化了 GPU 显存管理中的吞吐量和延迟。'
id: MjAyNS0x
models:
- gpt-5-pro
- gemini-2.5
- vllm
- deepseek-v3.1
people:
- epochairesearch
- yitayml
- _philschmid
- jiqizhixin
- cvenhoff00
- neelnanda5
- lateinteraction
- mgoin_
- blackhc
- teortaxestex
title: 今天没发生什么事。
topics:
- reasoning
- reinforcement-learning
- inference
- speculative-decoding
- sparse-attention
- kv-cache-management
- throughput-optimization
- compute-efficiency
- tokenization
---

**平静的一天**

> 2025年10月9日至10月10日的 AI 新闻。我们为您检查了 12 个 subreddit、544 个 Twitter 账号和 23 个 Discord（197 个频道，7403 条消息）。预计节省阅读时间（以 200wpm 计算）：586 分钟。我们的新网站现已上线，支持全元数据搜索，并以精美的 vibe coded 方式呈现所有往期内容。访问 https://news.smol.ai/ 查看完整的详细新闻，并在 @smol_ai 上向我们提供反馈！

第二轮 [**AIE CODE 申请**](https://apply.ai.engineer/) 将在 5 天内截止！

---

# AI Twitter 回顾

**推理：FrontierMath 对决、马尔可夫思维（Markovian Thinking），以及“推理训练”究竟教了什么**

- FrontierMath Tier 4 结果：在重计算（compute-heavy）设置下，GPT-5 Pro 以 13% 的准确率创下新纪录，仅以一道题的优势领先 Gemini 2.5 Deep Think（在统计学上并不显著）。Grok 4 Heavy 表现落后。Epoch 澄清了泄露疑虑：OpenAI 可以访问 28/48 道题目；GPT-5 Pro 解出的 8 道题中有 5 道属于预留集（held-out set）。查看来自 [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1976685685349441826) 的完整推文，后续内容中包含背景和方法论（[预留集详情](https://twitter.com/EpochAIResearch/status/1976685757369851990)，[历史总量](https://twitter.com/EpochAIResearch/status/1976685769130705300)）。Gemini 2.5 Deep Think 的强劲表现也受到了 [@YiTayML](https://twitter.com/YiTayML/status/1976470535308734575) 和 [@_philschmid](https://twitter.com/_philschmid/status/1976626257090535432) 的关注。FrontierMath 网站：[@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1976685780862144978)。
- 马尔可夫思维（Delethink）：Mila + Microsoft 提议训练模型在固定边界“写入状态（write state）”，将推理长度与上下文大小解耦——将推理转化为线性计算。一个 R1-Distill 1.5B 模型仅用 8K 上下文即可推理高达 24K token，以约 4 倍更低的计算量（7 vs 27 H100-months）击败了在完整 24K 上训练的 LongCoT-RL。[@jiqizhixin](https://twitter.com/jiqizhixin/status/1976466786565656986) 进行了报道，[@TheTuringPost](https://twitter.com/TheTuringPost/status/1976798665038758377) 提供了总结和链接（[效率详情](https://twitter.com/TheTuringPost/status/1976798717094379588)，[论文/代码](https://twitter.com/TheTuringPost/status/1976798729274544403)）。
- 推理训练究竟教了什么：新研究认为基础模型（base models）已经包含了推理机制；“思考模型（thinking models）”学习的是何时调用它们。在正确的时间调用技能可以弥补基础模型与推理模型之间高达 91% 的差距。查看 [@cvenhoff00](https://twitter.com/cvenhoff00/status/1976633766811734461) 的推文以及 [@NeelNanda5](https://twitter.com/NeelNanda5/status/1976660983084130377) 的评论（[后续](https://twitter.com/NeelNanda5/status/1976710233692012619)）。
- 对数学 RL 泛化保持谨慎：一些结果依赖于已经针对数学进行了大量中期训练（mid-trained）的 Qwen 基础模型——仅凭此设置推断广泛结论时需小心 ([@lateinteraction](https://twitter.com/lateinteraction/status/1976761442842849598))。

**系统与推理：Blackwell + vLLM、自适应投机采样（adaptive speculators）以及稀疏注意力 KV 分层**

- NVIDIA Blackwell + vLLM 赢得 InferenceMAX：vLLM 通过与 NVIDIA 的深度联合工作展示了强大的 Pareto 增益——涵盖整个技术栈的 100 多个 PR、FP4/FP8 Kernel、异步调度、图融合（graph fusions）以及 FlashInfer 集成——预计通过投机解码（speculative decoding）和数据 + 专家并行（DEP）还将获得 2–3 倍的吞吐量提升。来自 [@mgoin_](https://twitter.com/mgoin_/status/1976452383258648972) 和 [@NVIDIAAIDev](https://twitter.com/NVIDIAAIDev/status/1976686560398426456) 的总结（参见 [基准测试直播](https://twitter.com/SemiAnalysis_/status/1976669740035977702)）。
- ATLAS (Together AI)：一个从实时流量中学习的自适应投机解码系统；据报告比基准快 4 倍（在 DeepSeek‑V3.1 上达到 500 TPS），并随使用量的增加而持续改进。相关推文：[@togethercompute](https://twitter.com/togethercompute/status/1976655646474031362)（[自适应原理解析](https://twitter.com/togethercompute/status/1976655647925215339)，[结果](https://twitter.com/togethercompute/status/1976655649120612525)），[@tri_dao](https://twitter.com/tri_dao/status/1976692444977938499)。早期报告显示，通过自适应投机器可减少 >60% 的 RL 训练时间（[@BlackHC](https://twitter.com/BlackHC/status/1976730114902851908)）；VB 的报道：[链接](https://twitter.com/togethercompute/status/1976743626685530540)。
- 用于动态稀疏注意力（DSA）的 SparseServe：在 DSA 场景下，由于 KV cache 的驻留，瓶颈从 HBM 带宽转向了 HBM 容量。SparseServe 引入了 HBM↔DRAM KV 分层（GPU FlashH2D, CPU FlashD2H）、工作集感知的动态批处理以及分层预填充——在基于 vLLM 的测试中，相比 SOTA 实现了 9.26 倍的 TTFT 降低和 3.14 倍的吞吐量提升。[@ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/1976544233700929614) 进行了概述；[@teortaxesTex](https://twitter.com/teortaxesTex/status/1976556643031933352) 指出了硬件方面的影响。
- Kernel 开发速度 > “通用硬件”：随着 Triton 降低了门槛，且在 Blackwell 的速度下高级别开销占据主导地位，预计会出现更多自定义 Kernel（MoEs、低精度矩阵乘法、注意力变体、SSMs）（[@awnihannun](https://twitter.com/awnihannun/status/1976715815019037101)）。

**模型与工具发布**

- Qwen3‑VL Cookbooks：一套精选的 Notebook，用于本地/API 调用，涵盖多模态任务——计算机使用、全能识别、文档解析/OCR、3D 定位、视频理解、移动端 Agent、长文档理解、空间推理等。[@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1976479304814145877) 在帖子中提供了链接。
- 语音对语音：GPT Realtime Mini (OpenAI) 比旗舰版 Realtime 便宜约 7 倍，将 TTFA 缩短至 0.81 秒（原为 1.27 秒），上下文翻倍至 32k，并增加了图像输入——定位为通过 WebRTC/WebSocket/SIP 实现的可扩展 Agent。[@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1976696262083985636) 对比了 Gemini 2.5 Flash 原生音频对话（[图表](https://twitter.com/ArtificialAnlys/status/1976696264080474365)，[模型浏览器](https://twitter.com/ArtificialAnlys/status/1976696265754001747)）。
- 小型、快速、开源视觉模型：Moondream 3 (9B, 64 专家 MoE, 约 2B 激活参数) 增加了原生指向、改进的 OCR 和 32K 上下文——针对 UI 理解和 Agent 工作流进行了优化。[@moondreamai](https://twitter.com/moondreamai/status/1976624914070401142) 发布了公告，并在 FAL 上提供预览：[@fal](https://twitter.com/fal/status/1976682702167228919)。
- 智能体编码：KAT‑Dev‑72B‑Exp (Kwaipilot) 在 SWE‑Bench Verified 上排名第二；通过中期训练 → SFT+RFT → Agentic RL 进行微调；可在 4 张 RTX 3090 上以 4-bit 运行（[@TheAhmadOsman](https://twitter.com/TheAhmadOsman/status/1976606921756205531)）。
- 使用 LoRA/QLoRA/DoRA/QDoRA 进行 RL 后训练：Tora（基于 torchtune 构建）统一了 GRPO、FSDP 和编译支持；支持稳定的 4-bit RL (QLoRA/QDoRA)，并通过 DoRA‑Cache 将采样（rollouts）速度提升 2–4 倍（[@gm8xx8](https://twitter.com/gm8xx8/status/1976443792850092464)）。
- 工具速递：LangSmith 现在除 Python 外还支持 JS 代码评估，以实现更快的栈原生评估（[@LangChainAI](https://twitter.com/LangChainAI/status/1976700402105233603)）；LangChain v1 发布了可定制的 create_agent 以及用于模型/工具调用前后的中间件钩子（middleware hooks）（[@sydneyrunkle](https://twitter.com/sydneyrunkle/status/1976751776620593564)，[钩子原理解析](https://twitter.com/sydneyrunkle/status/1976753314462417344)）；LlamaIndex 增加了带有自定义规则的可解释文档分类（[@llama_index](https://twitter.com/llama_index/status/1976686683468026337)）；Glass Health 推出了符合 HIPAA 标准并包含引用元数据的生产级开发者 API（[@GlassHealthHQ](https://twitter.com/GlassHealthHQ/status/1976713436773138599)）。

**规模、算力与训练估算**

- 每月处理的 Token 量：根据 [@sundeep](https://twitter.com/sundeep/status/1976475987962626062) 的数据，Google 约为 1.3 quadrillion，OpenAI 约为 260T，Groq 约为 50T；Google 的 Demis Hassabis 重申了每月 1.3 quadrillion tokens 的数据 ([@demishassabis](https://twitter.com/demishassabis/status/1976712484657475691))。注意，不同模型/词表/任务中的 Token 信息密度和用途各不相同 ([@awnihannun](https://twitter.com/awnihannun/status/1976676812022550864))。
- 算力去向：Epoch 估计 OpenAI 去年在算力上花费了约 70 亿美元；大部分用于 R&D（实验/失败的运行），最终训练运行费用低于 10 亿美元 ([@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1976714284349767990) 及其 [后续](https://twitter.com/EpochAIResearch/status/1976714297255588053))。
- GPT‑5 训练推测：外部粗略估计显示其拥有约 100B 激活参数，30–100T tokens，RL 占预训练的 10–100%，总计约 6e25 FLOPs ([@teortaxesTex](https://twitter.com/teortaxesTex/status/1976441366969532888))。另有关于 MoE 稀疏性的传闻暗示总参数量极高但激活子集很小（例如 256–1024 个专家，其中 4–8 个激活），尽管有人认为实际激活数量和成本可能低于头条新闻所言 ([分析推文](https://twitter.com/teortaxesTex/status/1976773126584516801))。

**机器人与具身智能 (Embodied AI)**

- 硬件上的重定向特技：使用 OmniRetarget + BeyondMimic 极简 RL 追踪，一个人形机器人以 5/5 的成功率完成了墙壁后空翻；训练仅需微调（例如放宽终止条件、调整奖励）([@zhenkirito123](https://twitter.com/zhenkirito123/status/1976663920552427619))。另外，Unitree G1 复现了标志性的跆拳道旋风踢，通过调优解决了 sim‑to‑real 问题（IMU 陀螺仪饱和）([@kevin_zakka](https://twitter.com/kevin_zakka/status/1976460408077812085))。
- Agent 视觉：Moondream 3 针对真实世界的 UI 和结构化感知，为下游 Agent 框架提供支持（见上文模型发布）。

**评估、基准测试与治理**

- 基准测试改革：“基准测试已失效——不要让 AI 成为自己的法官”提出了 PeerBench，这是一个社区管理、受监考的评估蓝图：封闭执行、滚动题库、延迟透明度 ([@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1976586775603851344))。
- CoT 透明度：[@RyanPGreenblatt](https://twitter.com/RyanPGreenblatt/status/1976686565654221150) 认为，实验室应披露是否以及如何针对思维链 (Chain-of-Thought) 进行训练，并引用了 METR 的 GPT‑5 评估 ([详情](https://twitter.com/RyanPGreenblatt/status/1976686576521679252))。建议：如果涉及敏感 IP，可向受信任的评估者进行第三方披露 ([后续](https://twitter.com/RyanPGreenblatt/status/1976686574080491880))。
- OpenBench 扩展：Groq 的 OpenBench 现在支持 ARC‑AGI，扩大了推理基准测试的标准评估覆盖范围 ([@GregKamradt](https://twitter.com/GregKamradt/status/1976718318544601573))。
- “野外评估”与目标转移：越来越多的人强调超越静态测试集的实际评估 ([@lateinteraction](https://twitter.com/lateinteraction/status/1976439833158615345))；从玩具测试向持续自主和经济影响的文化转变被 [@aidan_mclau](https://twitter.com/aidan_mclau/status/1976658416451149874) 恰如其分地总结了。
- 治理争议（仅背景）：关于 OpenAI 传票的 Encode GC 推文引发了内外部评论；参见 [@_NathanCalvin](https://twitter.com/_NathanCalvin/status/1976649051396620514)、OpenAI 的 [@jasonkwon](https://twitter.com/jasonkwon/status/1976762546041634878) 的观点，以及 [@jachiam0](https://twitter.com/jachiam0/status/1976690339546112098) 的批评。关注其对政策讨论和开放规范的影响。

**热门推文（按互动量排序）**

- “面试了一位工程师……我 99% 确定他使用了 [AI 助手]” —— 来自 [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1976700504152719435) 的招聘信号转变和评估卫生。
- 通过 OmniRetarget + BeyondMimic 实现的惊人真实世界墙壁后空翻，来自 [@zhenkirito123](https://twitter.com/zhenkirito123/status/1976663920552427619)。
- Demis：Google 一个月处理了约 1.3 quadrillion tokens ([@demishassabis](https://twitter.com/demishassabis/status/1976712484657475691))。
- “2024 评估 vs 2025 评估” 梗图，捕捉了向长期、注重影响的指标的转变 ([@aidan_mclau](https://twitter.com/aidan_mclau/status/1976658416451149874))。
- 催化政策讨论的传票推文：[@_NathanCalvin](https://twitter.com/_NathanCalvin/status/1976649051396620514)。

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

暂无符合标准的内容

## 偏非技术类 AI 子版块摘要

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. NVIDIA GB300 NVL72 + ComfyUI GDS 性能更新

- [**Microsoft 发布首个大规模 NVIDIA GB300 NVL72 集群，让 OpenAI 能在几天内而非几周内训练出数万亿参数的模型**](https://www.reddit.com/r/singularity/comments/1o2t53m/microsoft_unveils_the_first_atscale_nvidia_gb300/) (Activity: 424): **Microsoft/Azure 宣布为 OpenAI 提供首个生产级 NVIDIA GB300 NVL72 集群 (NDv6 GB300)，涵盖超过 4,600 个 Blackwell Ultra GPUs；每个 NVL72 VM 通过 NVLink Switch fabric（每机架 130 TB/s）将 72 个 GPU 融合为一个统一的 37 TB 加速器，每个 VM 提供** `1.44 exaflops FP4` **的算力，并通过 Quantum‑X800 InfiniBand 以** `800 Gb/s per GPU` **的速度进行机架间扩展 ([来源](http://blogs.nvidia.com/blog/microsoft-azure-worlds-first-gb300-nvl72-supercomputing-cluster-openai/?linkId=100000386364404))。该技术栈针对使用 NVFP4 和 NVIDIA Dynamo 编译器的低精度训练/推理，以及具备自适应路由/基于遥测的拥塞控制功能的 SHARP v4 和 ConnectX‑8 SuperNICs，Azure 引用了 MLPerf Inference v5.1 的领先地位（例如，在** `671B` **推理模型上，吞吐量比 Hopper 高出多达** `5×`**）。粗略估算：>4,600 个 GPU ≈ ~64 个 NVL72 VMs → 总峰值算力约为 O(~**`92 exaFLOPS FP4`**)；注意 FP4 指标与 TOP500/HPL 的 FP64 exaflop 系统没有直接可比性。** 热门评论要求在不同系统之间进行公平比较（建议按精度、单 GPU flops、对分带宽、fabric/NIC 速度和 MLPerf 结果进行归一化），推测这能实现数万亿参数模型的训练时间表，并辩论数据与参数的缩放——指出 MoE 可以在保持活跃计算量适度的同时扩展参数，而“Chinchilla-optimal”的稠密模型需要数十万亿级别的 tokens（暗示除非增加合成/私有数据，否则稠密模型参数量约为 0.1–1T）。
    - 关于规模/拓扑：NVL72 是一个由 NVSwitch 连接的 72‑GPU 机架级“孤岛”，所有 GPU 共享一个高带宽 NVLink 域；多个孤岛随后在 Azure 中通过 InfiniBand/Ethernet 缝合在一起。因此，“~4,608 个 GPU”对应约 64 个 NVL72 机架 (64×72)，而“数十万个”则意味着跨区域的数千个此类孤岛。关键在于大多数张量/流水线并行流量保留在孤岛内部（比孤岛间快几个数量级），这就是为什么 NVL72 对大模型训练至关重要；有关拓扑细节，请参阅 NVIDIA 的 GB200 NVL72 概述：https://www.nvidia.com/en-us/data-center/products/gb200-nvl72/ 。
    - 如何与其他超级计算机比较：TOP500/Green500 对 FP64 Linpack 进行排名，不能反映 AI 训练（混合精度）或通信模式。对于 AI，应比较 (a) 单 GPU AI FLOPs 和 HBM 容量，(b) 孤岛内对分带宽 (NVLink/NVSwitch) 与互连 (400/800G InfiniBand/ROCE)，以及 (c) 大规模端到端 MLPerf Training 时间；与仅使用 Ethernet/InfiniBand 的设计相比，NVLink/NVSwitch 孤岛通常能减少梯度同步开销。相关的基准线是 MLPerf v3.x 中的多机架 H100/MI300 系统 (https://mlcommons.org/en/training-results-3-1/)，在这些系统中，当规模超过几千个 GPU 时，拓扑结构往往主导着缩放性能。
    - 关于“存在多少参数/数据？”：经过重删/质量过滤后，估计高质量文本+代码语料库总量约为 O(10–30T) tokens（参见 Epoch AI 的分析：https://epochai.org/blog/how-much-text-is-there）。根据 Chinchilla 缩放定律，最优稠密模型每个参数使用约 `20×` 的 tokens，这意味着 1T 参数的稠密模型理想情况下需要 ~20T tokens——接近上限——因此“数万亿参数”训练通常指 MoE，其中每个 token 仅激活 `~1–2` 个专家，在利用更大的总参数池的同时保持活跃参数在 ~100–200B 左右 (Chinchilla: https://arxiv.org/abs/2203.15556, Switch Transformers MoE: https://arxiv.org/abs/2101.03961)。

- [**我们现在甚至可以在 6GB NVIDIA 笔记本 GPU 上运行 Wan 或任何重型模型 | 感谢 ComfyUI 即将集成的 GDS**](https://www.reddit.com/r/StableDiffusion/comments/1o2wklr/we_can_now_run_wan_or_any_heavy_models_even_on_a/) (热度: 814): **开发者 Maifee 将 NVIDIA GPUDirect Storage (GDS) 集成到了 [ComfyUI](https://github.com/maifeeulasad/ComfyUI) 中，通过 NVMe 直接将模型权重流式传输到 GPU VRAM (cuFile DMA)，使得重型模型能够在显存低至** `6 GB` **的 GPU 上运行，且无需自定义 offloader 或量化。测试方法：** `git checkout offloader-maifee` **并运行** `python3 main.py --enable-gds --gds-stats`**；目前合并请求正在上游审核中。GDS（参见 NVIDIA 文档：https://developer.nvidia.com/blog/gpudirect-storage/）绕过了 CPU/主机 RAM，但实际吞吐量/延迟受限于 NVMe + PCIe 以及模型访问模式；需要安装了兼容驱动 (nvidia-fs)、CUDA 以及支持的文件系统 (ext4/xfs) 的 Linux 系统。** 评论者询问 GDS 与 RAM offloading 的区别：GDS 提供了一条零拷贝的 NVMe→GPU DMA 路径，避免了 CPU 中介和页面缓存 (page cache)，而 RAM offload 则将张量暂存在系统内存中并导致额外的拷贝；性能取决于存储/PCIe 限制。注意限制：目前仅限 Linux。
    - GPUDirect Storage (GDS) 实现了从 NVMe SSD 直接到 GPU VRAM 的 DMA，绕过了 CPU 和主机 RAM 的拷贝路径。实际上，这改变了数据路径，从 SSD → RAM → GPU 变为 SSD → GPU，降低了 CPU 参与度，并实现了约 4–5 GB/s 的有效 SSD→GPU 读取速度（受 NVMe/PCIe 限制），而通过 RAM 进行双重拷贝的速度约为 3–5 GB/s；GPU 从主机 RAM 读取的速度仍受 PCIe 限制（例如 PCIe 4.0 x8 双向约 16 GB/s，单向约 8 GB/s）。它提高了 I/O 效率而非计算能力，目前仅限 Linux；参见 NVIDIA 文档：https://developer.nvidia.com/gpudirect-storage。
    - GDS 不会减少 VRAM 需求，因此它无法避免 14 GB 模型在 6 GB GPU 上出现 OOM——在计算时，活动参数/激活值仍必须容纳在 VRAM 中。运行超大模型依赖于 offloading/分区执行（例如 CPU offload、层级流式传输），通常借助 Hugging Face Accelerate、DeepSpeed 等框架或 llama.cpp 等量化运行时；GDS 可以通过加速 SSD→GPU 传输来提升这些流水线的速度，但不会改变内存占用。
    - GDS 帮助最大的地方在于：非核心 (out-of-core) 工作流和低 RAM 系统，这些系统需要从快速 NVMe 流式传输模型权重/激活值，从而减少 CPU 开销并避免额外拷贝。如果 RAM 充足，将模型预加载到 RAM 并通过 PCIe 传输可能比从 SSD 流式传输更快，因为 RAM 带宽更高；权衡点在于 I/O 路径效率、CPU 负载与原始介质速度。净收益取决于工作负载和平台，值得进行基准测试。

### 2. AniSora V3.2 (Wan2.2) 360° I2V 与 Sora-2 演示

- [**使用 AniSora V3.2 实现 360° 动漫旋转**](https://www.reddit.com/r/StableDiffusion/comments/1o2qjiw/360_anime_spins_with_anisora_v32/) (热度: 594): [**AniSora V3.2](https://github.com/bilibili/Index-anisora) 是一款专注于动漫的图生视频 (I2V) 模型，基于 Wan2.2 I2V 构建，可直接插入 ComfyUI Wan2.2 工作流；将输入插画加载到 FLF2V 图中并应用仓库推荐的提示词，即可获得开箱即用的“360° 角色转场”，具有平滑的旋转、极高的扁平插画忠实度并保留了线条细节。此处提供了共享的工作流和示例：[🦊AniSora V3#68d82297000000000072b7c8](https://scrapbox.io/work4ai/%F0%9F%A6%8AAniSora_V3#68d82297000000000072b7c8)。** 评论者注意到了命名混淆（基于 Wan 却叫 “AniSora”），但对效果表示赞赏；他们认为这可以为 3D 资产流水线提供高吞吐量的多视图数据，并询问其在非写实风格下生成一致多视图的泛化能力。
    - 一位用户报告了一个可复现的 ComfyUI 稳定性/内存问题：在 `24GB` 显存的 GPU 上，AniSora V3.2 工作流完成了 High KSampler 阶段，但在加载 LOW 模型时导致 ComfyUI 崩溃，尽管峰值显存仅显示约 `19.5GB`。他们尝试插入清理节点以在 LOW 阶段前卸载 HIGH 模型，但未获成功，并询问作者使用的是哪个 ComfyUI 版本，暗示在 HIGH→LOW 两阶段流水线中可能存在特定版本的模型加载/GC/碎片化行为。
    - 几位评论者探讨了 360° 动漫旋转是否能为下游 3D 流水线（视频转 3D、NeRF/GS 风格重建）提供一致的多视图，以及它在非写实输入上的鲁棒性。其核心想法是利用时间/风格一致的旋转来改进 3D 模型生成的多视图监督，相比于随意且不一致的视图采样，这可能实现更高质量的动漫风格资产重建。

- [**Hyperspace and Beyond**](https://www.reddit.com/r/singularity/comments/1o36ptd/hyperspace_and_beyond/) (热度: 793): **非技术类 meme。这张名为“Hyperspace and Beyond”的图片似乎在讽刺论文讨论帖中那些低质量、炒作式的反应（例如发布超空间/闪烁的 GIF），而非实质性的参与；文中没有可供评估的模型细节、Benchmarks 或技术主张。** 评论区批评了一位习惯在论文帖子下发 GIF 而不阅读或理解内容的特定用户，并嘲讽了这种追求 Karma 而非贡献有意义讨论的行为；另一位评论者则表示“非讽刺地”表示赞同。
    - 一位评论者建议用一群可以检测威胁/损坏并在 `microseconds` 内重新配置的纳米机器人群来取代单体金属机身，这本质上是 [programmable matter](https://en.wikipedia.org/wiki/Programmable_matter)（可编程物质）/ [claytronics](https://www.cs.cmu.edu/~claytronics/)（粘土电子学）的一种形式。虽然微秒级的局部感知/驱动对于 MEMS/NEMS 是可行的（例如：快速压电驱动器 [ref](https://en.wikipedia.org/wiki/Piezoelectric_actuator)），但难点在于功率密度/输送、散热、大规模群体间的协调/通信延迟、容错能力以及“极大规模”下的制造良率。在 [active/metamaterials](https://en.wikipedia.org/wiki/Metamaterial)（活性/超材料）和 [self-healing materials](https://en.wikipedia.org/wiki/Self-healing_material)（自修复材料）领域的相关工作探索了这一愿景的部分内容，但具备这种反应能力的完全自适应纳米级集群仍超出了当前工程能力。
- [**Figure doing housework, barely. Honestly would be pretty great to have robots cleaning up the house while you sleep.**](https://www.reddit.com/r/singularity/comments/1o2u46w/figure_doing_housework_barely_honestly_would_be/) (热度: 1439): **该帖子分享了一段 Figure 人形机器人尝试进行基础家务清洁的短片，描述为仅能“勉强”工作，暗示了在非结构化家庭环境中的早期通用操作阶段。链接的视频 ([v.redd.it/ob954u6uh8uf1](https://v.redd.it/ob954u6uh8uf1)) 无法访问 (**`HTTP 403 Forbidden`**)，因此无法获得关于自主水平（teleop 与 onboard）、任务成功率或控制栈的可验证细节。楼主（OP）畅想了无人值守的夜间运行（在用户睡觉时清洁），但未提供安全或可靠性数据。** 热门评论预计能力将在“几年内”迅速提升，而其他评论则指出了人机交互以及夜间运行的安全/UX 担忧（惊吓风险），其余评论多为幽默或偏离主题的内容。
- [**This is cheating at this point 😂**](https://www.reddit.com/r/ChatGPT/comments/1o2onjk/this_is_cheating_at_this_point/) (热度: 17459): **一个爆火的 Reddit 帖子分享了一段疑似 OpenAI Sora 生成的文本转视频剪辑，描绘了水上运动背景下的“耶稣”，评论者注意到其逼真程度。链接资源托管在 v.redd.it，目前对未授权请求返回** `HTTP 403 Forbidden` **（[视频链接](https://v.redd.it/yasrp8dd07uf1)），表明 Reddit 的网络安全拦截（需要登录/OAuth）。背景：Sora 是一款能够生成写实且符合物理规律剪辑的生成式视频模型；精选样本已公开，但普通访问权限仍受限 ([OpenAI Sora](https://openai.com/sora))。** 一条热门评论指出，“这些 Sora 剪辑真的开始唬住人了”，指向了非技术受众（例如在 Facebook 上）中日益增长的可信度，以及对来源/Deepfake 风险的隐含担忧；其他评论大多是幽默的双关语。
    - 一位评论者指出，**OpenAI Sora** 的文本转视频输出现在已经足够写实，足以欺骗社交平台上的普通观众，并强调了随着“40 多岁人群”开始分享这些内容而带来的检测难题。Sora 可以生成长达 `~60s` 的高分辨率剪辑，具有很强的时间连贯性（temporal coherence）和相当合理的物理特性，减少了旧有的明显伪影 ([openai.com/sora](https://openai.com/sora))。这增加了对更强大的取证线索（例如：运动/物理边缘案例、镜面高光一致性、水/飞溅动力学）和溯源工具的需求，以便可靠地标记合成媒体。

- [**抱歉，但这确实是我见过最搞笑的 AI 内容之一。**](https://www.reddit.com/r/ChatGPT/comments/1o2w0fm/im_sorry_but_this_is_some_of_the_funniest_al_ive/) (Activity: 3223): **该帖子链接到一个 Reddit 托管的视频（[v.redd.it](https://v.redd.it/7c7g9vy949uf1)，未授权访问时返回 403），该视频似乎使用了 AI 生成的音频为真实素材配音——评论者指出“只有音频是 AI”。这段 AI 音轨是“Zelensky”的喜剧化语音克隆，正在辩论“micropenis”的定义，这表明是直接的 TTS/语音克隆，而非完整的视频 deepfake；文中分享了一个静态预览图（[image](https://preview.redd.it/p7kc6a30y9uf1.jpeg?width=320&format=pjpg&auto=webp&s=cfc48f8f941d84a4d475bb1f99d64459aae6558d))。** 评论者主要验证了模态（仅音频合成）并对幽默感做出反应；除了提到可能的音频配音外，没有提及具体的模型、pipeline 或伪造痕迹（artifacts）。
    - 一位评论者询问是否只有音频是 AI；从技术上讲，许多病毒式传播的片段保留了真实视频，同时通过 TTS/语音转换替换了克隆的语音（例如，**ElevenLabs**、**Microsoft VALL-E** 可以从 `3–60s` 的参考音频中模仿声音）。在没有视频重新合成的情况下，音素-嘴唇不同步（phoneme–lip desync）是一个破绽（口型与辅音/元音不匹配）；如果视频也经过伪造，像 [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) 或 [SyncNet](https://www.robots.ox.ac.uk/~vgg/software/lipsync/) 这样的模型可以将口型与合成音轨对齐。实时语音转换（如 [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)）可实现低延迟配音，使得仅音频的恶搞内容制作变得异常简单。
    - 第二个主题是媒体真实性：*“这将使所有真实的东西失效”*。风险在于高质量的 AI 音频廉价且无处不在，而检测器在处理短片段或嘈杂片段时仍然不可靠；溯源方案如 **C2PA** [Content Credentials](https://c2pa.org/) 和音频水印（Meta 的 [AudioSeal](https://ai.meta.com/research/publications/audioseal-synthetic-speech-watermarking/)、Google 的 [SynthID](https://deepmind.google/technologies/synthid/)）虽有帮助，但在重新编码/转换下很脆弱且可以被移除。短期内，稳健的验证更多地依赖于经过签名的采集 pipeline 和分发 metadata，而不是事后分类（post-hoc classification），因为后者的误报率（false positive）和漏报率（false negative）都不容忽视。

### 3. 配送失败：DoorDash 门廊坍塌与亚马逊投递

- [**600磅的 DoorDasher 掉进了门廊地板 😂**](https://www.reddit.com/r/aivideo/comments/1o2qdyg/600lb_doordasher_falls_through_the_porch_floor/) (活跃度: 2401): **链接的媒体是托管在 Reddit 上的视频 [v.redd.it/cetb4u6kf7uf1](https://v.redd.it/cetb4u6kf7uf1)，目前在未授权的情况下返回** `HTTP 403 Forbidden`**，表明存在 Reddit 网络安全拦截；根据 Reddit 的指南，访问需要登录会话或基于 Token 的 API 凭据（403 页面提供了登录/支持链接）。帖子标题声称一名约** `~600 lb` **的 DoorDash 快递员掉进了门廊地板，但由于无法访问媒体，该说法及任何技术元数据（如时长、Codec、时间戳）均无法验证。**评论者推测该视频是 AI 生成的真实感——有人指出 AI “掌握了物理特性”，暗示了模拟物理与现实世界物理忠实度之间的争论；而其他人则认为这可能是病毒式传播的内容；帖子中未提供具体的物理证据。
    - 几位评论者建议该片段是 AI 生成的（其中一位提到了 **Sora 2**），并指出它在模拟质量/力和结构失效方面非常有说服力——“AI 确实精准地掌握了物理特性”。这暗示了现代视频生成模型（如 Sora 2）与早期版本相比，具有更强的学习物理先验（Physics Priors）和时间一致性（Temporal Coherence）（例如：物体持久性、动量守恒、接触动力学），早期版本在长序列下经常崩溃，这使得输出结果越来越难以与真实素材区分。
- [**亚马逊配送**](https://www.reddit.com/r/aivideo/comments/1o2p45q/amazon_delivery/) (活跃度: 481): **标题为“Amazon Delivery”的视频帖子，托管在 [v.redd.it](https://v.redd.it/1ubc2u6847uf1)，目前在未授权的情况下返回** `403 Forbidden`**，表明存在访问控制（OAuth/Cookie）或 WAF 强制执行；可通过登录或确保正确的 Header（如** `User-Agent`**, ** `Referer`**）进行排障。该片段似乎描绘了一只狗跳过窗户并撞向灌木丛，评论者注意到令人惊讶的逼真植物与身体交互（形变、遮挡、动量传递）——这是视频生成训练数据中经常代表性不足的挑战性边缘案例——表明接触动力学中的物理连贯性有所提高。**虽然有些评论很风趣（例如 “r/PackageDelivered”），但一份技术评论指出，尽管有类似 Bourne 般的飞跃，狗在灌木丛中笨拙的挣扎揭示了在控制/立足真实感方面仍存在限制，暗示了在细粒度物理和 Affordance Modeling 方面存在差距。
    - 一位评论者指出了狗与灌木丛交互的可信度——这是生成式视频/场景合成的一个难点，因为涉及**可变形植被（Deformable Foliage）**、严重的自遮挡以及训练数据中的有限表示——这意味着该模型能够较好地泛化接触动力学和次级运动（Secondary Motion）。他们补充说，虽然专家可能会发现伪影（例如碰撞/穿透错误、不一致的叶片变形或时间不连贯），但对于非专家来说，物理效果看起来很有说服力，表明尽管在显式物理建模方面可能存在差距，但模型具有强大的学习先验。

---

# AI Discord 简报

> 由 gpt-5 生成的摘要之摘要的摘要
> 

**1. 推理加速与 Kernel 优化**

- **Predicted-Outputs Prefill 赋能 vLLM**：Cascade 宣布为 **vLLM** 推出 **Predicted Outputs**，将可能的补全转换为部分匹配的 prefill，从而大幅提升生成速度；请参阅 [Predicted Outputs in vLLM](https://cascadetech.ai/blog/vllm-predicted-outputs/) 博客文章、[在线演示](https://app.cascadetech.ai/) 以及相关的 [推文串](https://x.com/saganite/status/1976707696578691101)。该方法将投机预测（speculative predictions）转化为缓存计算，旨在不改变 API 的情况下降低延迟。
    - 成员们称其为 *“通过预计算潜在输出序列，可以显著加快 vLLM 推理速度”*，认为它是一个可以无缝接入现有 **vLLM** 部署的实用投机器。早期测试者将其与投机采样（spec-decoding）的优势进行了比较，但对其简单性和跨工作负载的可移植性表示赞赏。
- **残差重计算（Residual Recalc）大幅提升吞吐量**：**LLMQ** 实现了 attention-residual 重计算以缓解内存压力，在受限设备上获得了巨大的速度提升；PR [attention-residual recalculation](https://github.com/IST-DASLab/llmq/pull/7) 显示，**Qwen2.5-14B** 在 4×4090 上的 TPS 从 **3.2k→6.0k (fp8)** 和 **2.5k→4.5k (bf16)** 飞跃。该优化在反向传播期间重新计算低成本项，以算力（flops）换取带宽。
    - 工程师指出，在激活值占主导的高内存压力场景下，收益最为明显，称其为 14B 规模训练的 *“显著净加速”*。讨论强调将此技巧与混合精度和精细的激活检查点（activation checkpointing）结合使用，以获得最佳效果。
- **Swizzles 优化与 ldmatrix 经验教训**：内核开发者在代码中将 **Triton** 的 `ldmatrix` 平铺逻辑固定在 d = log2(4/w)，参考了 [Utility.cpp](https://github.com/triton-lang/triton/blob/b5fea1e3f4c2cb0b40c0ce98261b240d8728d2f9/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.cpp#L186-L199)，并通过 [GenericSwizzling.cpp (PR #6982)](https://github.com/triton-lang/triton/pull/6982) 明确了最佳 swizzling 方案。同时，**CUTLASS** 用户指出 **PTX** K-contig/swizzling 文档不匹配（见 [NVIDIA PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-warpgroup-k-32b-swizzle-tf32)），并报告了使用 `ldmatrix` copy atoms 时编译器中止的问题。
    - 贡献者预计 PTX 文档修复将在 13.1 版本左右完成，并提醒许多 *ICE（内部编译器错误）是用户在尝试奇特的 copy atoms 时的错误*。结论：遵循 **Triton** 的参考实现进行 swizzling，并验证张量描述符以避免布局导致的停顿和中止。

**2. 小型模型，强劲基准测试**

- **三星的小型模型在 ARC 表现惊人**：三星的 **Tiny Recursive Model (TRM)** 在 **ARC-AGI-1** 上获得了 **44.6%** 的分数，在共享基准测试讨论中超过了 **DeepSeek-R1**、**Gemini 2.5 Pro** 和 **o3-mini** 等更大型的模型。工程师们强调了尽管 TRM 体积精简，但其表现出的惊人差距。
    - 争论集中在 TRM 是针对 **ARC-AGI** 的 *“过度专业化（hyper specialization）”*，还是具备 **LLM** 所期望的广泛泛化能力。从业者警告不要在没有多任务验证的情况下过度索引单一基准测试。
- **ARC 军备竞赛引发担忧**：研究人员指出，**HRM/递归模型** 方法可能通过对小型公共数据集进行激进的数据增强来过拟合 **ARC-AGI**，从而模糊了数据泄露的界限。一些人认为大型实验室可能会采取类似的数据增强手段来追求排行榜的提升。
    - 其他人提出了混合系统——将用于处理未知模式任务的结构化推理器与用于世界知识的 **LLM** 配对——尽管目前尚未出现稳健的集成模式。相关讨论探索了将 **GNN** 中间体作为小型推理模块潜在控制器的可能性。
- **超参数优化使 Diffusion 步数减半**：论文 [Hyperparameters are all you need](https://arxiv.org/abs/2510.02390) 的一项实现显示，**8 步** 即可达到图像生成中 **20 步** 的 FID，声称减少了约 **60%** 的计算量并实现了 **2.5 倍** 的加速；可以尝试以下 Spaces：[Counterfeit v3.0 版本](https://huggingface.co/spaces/coralLight/Hyperparameters_Are_All_You_Need)、[XL 版本](https://huggingface.co/spaces/coralLight/Hyperparameters-are-all-you-need-xl-version) 以及 [原始 SD 版本](https://huggingface.co/spaces/coralLight/hyperparameters-are-all-you-need-sd-version)。该方法无需训练或蒸馏，且适用于各种模型。
    - 用户报告称，在极少的步数下质量优于 DPM++2m，并赞扬了用于快速 A/B 测试的可复现配置。团队询问了下一个模型目标，并分享了前后对比产物以验证速度与质量的权衡。

**3. AI 融资与并购综述**

- **Spellbook 的 B 轮融资增长至 5000 万美元**：**Spellbook** 完成了由 Khosla 领投的 **5000 万美元 B 轮融资**，被 [Scott Stevenson](https://xcancel.com/scottastevenson/status/1976280608436572393) 称为“合同界的 Cursor”。该平台声称自 2022 年以来拥有 **4,000 名客户**，建议采纳率从 **5% 跃升至 60%**，资金将用于产品开发（例如实时市场对比测试版）。
    - 法律科技人士赞赏其增长指标和务实的功能路线图。开发者预计更紧密的类 IDE 工作流（diffs、修订、审计追踪）将成为法律 AI 助手的基础配置。
- **Datacurve 融资 1770 万美元用于数据**：**Datacurve** 宣布获得 **1500 万美元 A 轮 + 270 万美元种子轮融资**，用于为基础模型（尤其是编程领域）构建高质量训练数据集，据 [Serena Ge](https://xcancel.com/serenaa_ge/status/1976328983458480539) 报道。支持者包括 Chemistry、YC、Cohere、Afore 和天使投资人。
    - 工程师认为，专用、许可证清晰且标注丰富的代码语料库对于下一代模型的可靠性至关重要。这一轮融资表明，投资者对为前沿 **LLM** 训练提供支持的专业数据供应商持续保持热情。
- **Elastic 收购 Jina 以强化 Embeddings 和 Agent**：据 [Elastic 的公告](https://xcancel.com/elastic/status/1976278980018765886)，**Elastic** 收购了 **Jina AI**，以加强检索、**Embeddings** 以及针对 **Agentic AI** 的上下文工程。此举旨在通过更深层次的多模态和 **Agent** 工具来增强企业搜索。
    - 从业者预计 **ES** 与向量流水线、混合搜索和 **RAG** 编排将有更紧密的集成。此次收购暗示了行业整合趋势，即现有的基础设施厂商将 AI 原生技术栈直接内置到其平台中。

**4. 协议标准与结构化工具**

- **Well-Known 获胜：MCP 元数据确立地位**：**Model Context Protocol (MCP)** 社区提议为服务器元数据建立 `.well-known/` 端点——涵盖文档名称、URL 相对位置和最小化的 `Implementation` 内容——参见 [MCP 博客更新](https://blog.modelcontextprotocol.io/posts/2025-09-26-mcp-next-version-update/#server-identity)、[讨论](https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/1147)和 [PR 线程](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/1054#issuecomment-3117705161)。一份 [开发者峰会演示文稿](https://github.com/modelcontextprotocol/registry/blob/main/docs/explanations/dev-summit-2025-10-registry-status-presentation.pdf) 概述了注册表的发展方向。
    - 贡献者倾向于采用最小化的 SEP，以避免破坏客户端，同时实现发现和身份识别。标准化的元数据预计将简化服务器上线、信任信号和工具互操作性。
- **想象一下：structuredContent 中的便携式图像**：成员们讨论了在 **structuredContent** 中表示图像内容的问题，指出当宿主将 `StructuredOutput` 直接传递给模型 API 时，移植性会失效。指导意见是：协议不强制要求宿主映射，且供应商对返回图像的工具支持较差；一个将字符串映射到图像的临时工具可以弥补这一差距。
    - 团队警告不要将 Schema 锁定在任何单一 **LLM** API 的特性上。一个薄的间接层——资产的工具存根（tool stubs）——可以保持 UI 的数据填充，同时让模型看到精简的可序列化描述符。
- **Skybridge 跳过 Schema，引发 Schema 分歧**：注意到 **skybridge** 工具不使用 `outputSchema`，这重新引发了关于协调 `ContentBlock[]` 与 **structuredContent**（后者常用于组件填充，但对模型仍可见）的争论。贡献者探讨了是否应该定义任何正式的 Schema 绑定。
    - 共识倾向于务实的灵活性：使用结构化块填充 UI，但在 Schema 实践仍在演进时，避免对模型的 I/O 过度约束。预计会出现增量式的约定，而非一次性的宏大规范。


---

# Discord: 高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **GPT-5 赢得存疑的代码竞赛胜利**：成员们正在争论哪种 AI 模型最适合编程，一些人因其响应速度和理解能力而青睐 **GPT-5**，而另一些人则支持 **Claude 4.5** 用于规划和研究。
   - 争论引发了关于 **GPT-5** 是否能生成 *更优化的代码* 的讨论。
- **Perplexity Pro 限制视频创作**：用户报告称，即使使用 **Perplexity Pro**，也存在搜索限制（每天约 **300** 次），如果输入速度过快，还有被标记为垃圾邮件发送者的风险，且 *视频创作限制为每月 5 个*。
   - 激进的限制让一些成员对其价值产生了怀疑。
- **Comet Browser 准备发布移动版**：成员们焦急地等待 **Comet Browser** 的移动版发布，预计发布日期在 *年底左右*。
   - 一位用户吹嘘说 *Comet 在考试中表现出色，会为你选择正确答案*。
- **GPTs Agent 仍保持静态大脑**：成员们讨论了 **GPTs Agent** 在上传新文件后，*不会更新其初始训练后的基础知识*。
   - 相反，上传的文件被保存为 *知识文件*，Agent 会在 [需要时引用](https://link.to/openai-docs)。
- **Perplexity 的 Search API 遭到 Cloudflare 拦截**：一名成员报告了在使用 **Perplexity** 的 **Search API** 时遇到的奇怪问题，出现了 `PermissionDeniedError`，这似乎与 **Cloudflare** 的拒绝访问有关。
   - **Cloudflare** 拦截的根本原因尚不清楚。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **LMArena 权衡定价方案**：成员们讨论了 **LM Arena** 定价的未来，引用了其 [ToS（服务条款）](https://lmarena.ai/tos)，该条款允许其因数百万美元的运行成本而随时收费。
   - 然而，一位维护者表示 *我们的意图是保持对所有人免费和开放*，并正在探索 [替代策略](https://news.lmarena.ai/ai-evaluations/) 以避免收取费用。
- **Sonnet 4.5 表现低迷且出现故障**：成员报告 **Sonnet 4.5 Thinking** 频繁报错，特别是在长时间对话后，怀疑可能存在 **API** 问题。
   - 尽管有建议清除 Cookie，但问题对许多人来说仍然存在，尚未找到明确的解决方案。
- **Video Arena 遭到吐槽**：用户批评 **Video Arena** 的模型选择有限、请求受限、使用图像必须登录以及成本高昂。
   - 一位成员认为视频生成的成本比文本机器人更高，从而导致了这些限制。
- **Gemini 3.0 发布推迟？**：社区讨论了 **Gemini 3.0** 的发布是否仍按计划在 10 月进行，还是会推迟到 12 月。
   - 有人声称它正在 **AiStudio** 内部进行 A/B 测试，尽管这尚未得到证实。
- **Grok 展现内在的“斯皮尔伯格”潜质**：用户正在探索 **Grok** 的视频生成功能，该功能无审查、支持音频且无限制。
   - 尽管分辨率较低（*560 x 560*）且可能有水印，但它被赞誉为 *免费的小兄弟*。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Perplexity 的 LinuxComet 面临隐私抵制**：**LinuxComet**（来自 Perplexity 的一款浏览器）被批评为“海量追踪器”，其收集的数据比顶级浏览器的总和还要多，这促使用户开始寻找 **Safari**、**Firefox** 和 **Brave** 等替代方案。
   - 一些成员开玩笑说，为了隐私而使用 **Firefox** 就像是在发布到 **Google Maps** 上的玻璃房子里戴锡纸帽，浏览器的选择反映了个人对隐私的态度。
- **GPT-5 Thinking Mini 思考过度**：用户正在寻找禁用 **GPT-5 Thinking Mini** 的方法，因为它会自动从 **GPT-5 Instant** 切换，并提供令人不满意的结果。
   - 目前尚未提供解决方案，用户仍受困于这种不必要的行为。
- **三星的小型模型震惊基准测试界**：三星的 **Tiny Recursive Model (TRM)** 在标准 AI 基准测试中表现优于 **DeepSeek-R1**、Google 的 **Gemini 2.5 Pro** 和 OpenAI 的 **o3-mini** 等大型模型，在 **ARC-AGI-1** 测试中达到了 **44.6%** 的准确率。
   - 讨论引发了关于 **TRM** 针对该基准测试的**过度专业化**与 **LLM** 通用目的之间的质疑。
- **ChatGPT Business 脱胎于 ChatGPT Teams**：自 **2025 年 8 月 29 日**起，**ChatGPT Team** 将更名为 **ChatGPT Business**，详见 [发布说明](https://help.openai.com/en/articles/11391654-chatgpt-business-release-notes)。
   - 关于这一转变的细节已出现在 OpenAI 的官方文档中。
- **Sora 2 提示词：视觉化破解代码**：成员们讨论了是否存在 **Sora 2** 的“万能提示词”，结论是好的提示词应包含**你想要看到的细节和特质**。
   - 用户正在通过寻找令人惊叹的视频、复制提示词、重构结构并调整内容来反向工程提示词，以观察文本如何影响视频生成。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **GitHub 追踪 VST 受欢迎程度**：一个 [GitHub 仓库](https://github.com/MikesRuthless12/tools-hub) 通过网站流量、**YouTube 频道**订阅数和月均观看量来追踪 **VST** 的受欢迎程度。
   - 这些数据为评估 **VST 社区**内的影响力和参与度提供了宝贵的见解。
- **BYOK 支付问题困扰用户**：用户报告在使用 **BYOK** 时提示 **Payment Required**（需要支付），无法重定向到其密钥并导致账户扣费。
   - 该问题仍未解决，影响了尝试在平台上使用自有密钥的用户。
- **DeepSeek 限制激怒角色扮演玩家**：用户对 OpenRouter 上的 **rate limits**（速率限制）和 **DeepSeek 模型**的移除表示不满，正在寻找用于角色扮演的**无审查免费替代方案**。
   - 在 **DeepSeek 3.1** 受到限制后，一名用户正在为配置了 4070 显卡和 32GB 内存的笔记本电脑寻找本地模型。
- **SSE 使用数据获取遇到障碍**：用户在使用 OpenRouter API 的 **SSE** 获取**使用数据**时遇到问题，接收到的消息中缺少 *usage* 对象。
   - 一个潜在的修复方案涉及解决在 usage 数据块之前返回 *finish_reason: stop* 数据块的问题，详见 [此 litellm issue](https://github.com/BerriAI/litellm/issues/11626)。
- **Qwen 将推出新模型**：**Qwen** 计划下周发布更多模型，正如在 [X 上的帖子](https://x.com/JustinLin610/status/1976681042041028823) 中所宣布的那样。
   - 爱好者们期待着即将发布的模型能带来更强的能力和性能。

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen3-VL：禁用 Autocompiler 可节省 VRAM**：成员们讨论了 Unsloth 对 **Qwen3-VL** 的支持，一位用户确认通过使用 `os.environ['UNSLOTH_COMPILE_DISABLE'] = '1'` 禁用 autocompiler 可以启用该模型，并占用约 **44GB VRAM**。
   - 讨论澄清了禁用 autocompiler 只是一个临时的变通方案。
- **DGX Spark 微调：FLOPS 表现良好，带宽不给力？**：一位用户质疑 **DGX Spark** 的微调可行性，理由是带宽问题；另一位用户分享了演示体验，报告称 **32B** 模型的速度*慢得惊人，不到 10/s*。
   - 考虑到 AMD 替代方案缺乏 **CUDA** 支持，该成员希望通过软件优化使 **LoRA** 变得可行。
- **TRL 库：Trainer 面临重构**：上游的 **TRL 库** 正在考虑移除几个 Trainer（**CPO/SimPO, KTO, OrPO, PPO, RLOO**）且不保留支持，这引发了重构担忧，详见 [此 GitHub issue](https://github.com/huggingface/trl/issues/4223)。
   - 尽管存在担忧，但目前看来 **KTO, ORPO 和 PPO** 正在被移至 **trl.experimental**。
- **WSL2 用户深陷 xformers 包安装困扰**：一位用户在 WSL2 上为摘要数据集微调 **Mistral 7B** 时，遇到了包不兼容问题（CUDA 版本与 **xformers** 不匹配）。
   - 一位成员建议卸载并重新安装 **torch**、**torchvision** 和 **torchaudio**，并提供了从 GitHub 安装 **xformers** 的命令。
- **7M 递归模型在 ARC-AGI 中表现惊人，泛化能力存疑**：一位成员分享了一篇关于 [7M 递归模型](https://arxiv.org/abs/2510.048717) 的论文，该模型仅有两层，在 **ARC-AGI 1 和 2** 中达到了 **SOTA**，重点在于深度监督和防止崩溃/发散的保护措施。
   - 其他人指出，虽然令人印象深刻，但由于模型尺寸较小以及 **ARC-AGI** 使用私有测试集进行测试，该模型的**泛化能力**可能有限。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen3 完胜 Code Llama**：成员们建议在编程任务中使用 [Qwen3](https://lmstudio.ai/models/qwen/qwen3-coder-30b)，因为它在**速度和准确性**方面表现出色，运行速度为 **20 tk/s**，而 CodeLlama 仅为 **5 tk/s**。
   - 然而，用户指出即使是基础的编程任务也需要大于 **20b 参数** 的模型，因此一位用户提到 [Llama 3.3 70b](https://lmstudio.ai/models/meta-llama/Llama-3-70B-Instruct-GGUF) 会更好。
- **上下文长度引发困扰**：用户探讨了上下文长度如何影响性能；即使上下文为空，**分配更高的上下文也会减慢生成时间**。
   - LM Studio 正在实施由内存估计器控制的内存分配护栏，但你可以冒着风险禁用它们。
- **Tool Call 故障导致生成中断**：当 tool call 失败时，LM Studio **不应该**中止生成，但目前它确实会中止，且不显示错误消息。
   - 一种可能的解决方案是禁用 MCPs (Multi-Call Procedures) 以防止模型混淆，或者研究抓取网站的工具，因为 Playwright 的 **token 效率极低**。
- **Sparkle 服务器展示 16 块 Arc GPU**：Sparkle 推出了 [Arc Pro B60 双路服务器](https://videocardz.com/newz/sparkle-unveils-arc-pro-b60-dual-server-with-16-gpus-and-up-to-768-gb-of)，拥有 **16 块 GPU** 和高达 **768 GB 的 VRAM**，由 **10800W PSU** 供电。
   - 成员们对这款专注于多 GPU 的服务器产品感到兴奋。
- **4080 Ti Super 与 M3 的性能之争**：成员们讨论了拥有 **16GB VRAM** 和 **32GB RAM** 的 **Nvidia 4080 Ti Super** 在机器学习任务中是否优于拥有 **36GB RAM** 的 **Apple M3**。
   - 一位成员建议查看 LLM 推理的 [GPU 基准测试](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference) 以比较性能，并指出配备 **64GB 内存的 M3 Max 40 核 GPU** 是一个更具可比性的配置。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **API Keys 解锁模型**：用户现在可以在 **Models** 页面添加 **API keys**，以便在模型中使用个人密钥，并可以关闭所有模型并手动添加。
   - 这使得用户能够更好地控制模型访问和计费。
- **AutoHotkey 控制 Cursor**：成员们讨论了使用 **AutoHotkey (AHK)** 从 **Discord** 控制 **Cursor**，一些人计划使用 Cursor 生成 **AHK scripts**。
   - 社区对此集成表现出了极大的热情和兴趣。
- **Cursor 定价成为焦点**：用户对 **Auto** 模型成本的增加以及 **Cursor Pro** 计划中用量限制的减少展开了辩论，并考虑使用替代模型或服务，详情总结在[这篇论坛帖子](https://forum.cursor.com/t/the-pro-account-limits-have-been-clearly-reduced/134738)中。
   - 这些变化导致一些用户重新评估 Cursor 订阅计划的价值主张。
- **GPT-5 表现出色**：一位成员吹捧用于编程的 **GPT-5** 更便宜，但另一位成员指出，其缺点之一是由于其推理过程需要更多时间。
   - 这引发了关于成本效益与开发时间之间平衡的辩论。
- **Apply 按钮出现异常**：用户报告称 **ASK** 模式下的 **APPLY** 按钮消失了，导致应用更改变得更加困难，并迫使他们不得不勉强使用 **AGENT** 模式，因为该模式表现得不受控制。
   - 这一变化令那些更喜欢 **ASK** 模式提供的直接控制权的用户感到沮丧。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton 分块逻辑取得突破**：`ldmatrix` 的分块（tiling）计算涉及 *d = log_2(4/w)*，其中 *w* 是字节宽度，完整实现可在 [Triton 源代码](https://github.com/triton-lang/triton/blob/b5fea1e3f4c2cb0b40c0ce98261b240d8728d2f9/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.cpp#L186-L199)中找到。
   - 社区澄清了这些计算是在 **Triton** 内部不同的 lowering 阶段实现的。
- **最优 Swizzling 秘诀引发讨论**：讨论澄清了线性布局中的最优 swizzling，这是基于相关论文中图 5、第 5.4 节和第 9.2 节之间的差异，但 [来自 pull 6982 的代码](https://github.com/triton-lang/triton/pull/6982) 才是真正的核心参考。
   - 该实现可以在 **GenericSwizzling.cpp** 中找到。
- **Panther Lake 提升性能**：新的 [Panther Lake 幻灯片](https://www.techpowerup.com/review/intel-panther-lake-technical-deep-dive/13.html) 展示了每个切片（slice）计算量的增加以及高达 **16 MiB 的 L2 缓存**，相对于 Arrow Lake 实现了 **40% 的每瓦性能提升**，解决了内存带宽限制问题。
   - **Celestial** 的架构仍不明朗，这取决于相对于 **Panther Lake** 所需的计算单元与固定功能单元的比例。
- **CUTLASS 编译器处理 Copy Atoms**：一位用户报告了在使用 **ldmatrix** copy atoms 时出现的编译器中止问题，并为该问题创建了一个名为 `t.py` 的代码复现。
   - 团队提到，许多“用户错误”会表现为内部编译器错误（**ICEs**）。
- **LLMQ 通过重计算提升带宽**：实现了注意力残差重计算（在[这个 llmq PR](https://github.com/IST-DASLab/llmq/pull/7) 中实现），这在内存压力较大的情况下（如在 **4x4090** 上运行 **14B 模型**）可以带来显著的净增速。
   - 该优化在以 **fp8** 训练 **Qwen2.5-14B** 时将吞吐量从 **3.2k 提升至 6.0k TPS**，在以 **bf16** 训练时从 **2.5k 提升至 4.5k TPS**。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Colab A100 每小时成本为 75 美分**：成员们讨论了每月 **$10** 的 **Colab** 的性价比，其中一人计算出它提供了大约 **13.2 小时** 的 **A100 GPU** 时间，成本约为 **75 美分/GPU/小时**。
   - 一位用户开玩笑说要向父母要零用钱，而其他人则辩论了针对小型项目进行 GPU 分配和利用的最佳策略。
- **超参数 Diffusion 在图像生成领域飞速发展**：一位成员发布了一个 HuggingFace Space，展示了 [Hyperparameters are all you need](https://arxiv.org/abs/2510.02390) 的实现，证明了 **8 步** 即可达到与图像生成中 **20 步** 相当的 FID 性能，从而实现了 **60% 的计算量减少**。
   - 该实现包括 [Counterfeit v3.0 版本](https://huggingface.co/spaces/coralLight/Hyperparameters_Are_All_You_Need)、[XL 版本](https://huggingface.co/spaces/coralLight/Hyperparameters-are-all-you-need-xl-version) 以及 [原始 SD 版本](https://huggingface.co/spaces/coralLight/hyperparameters-are-all-you-need-sd-version) 用于测试，运行速度快 **2.5 倍** 且 *质量更好*。
- **BERT 识别波兰语推文**：一位成员微调了一个 **BERT 模型** 来预测 **波兰语推文** 中的情感和情绪，可以在 [这里](https://huggingface.co/spaces/yazoniak/twitteremo-pl-classifier) 找到。
   - 该模型旨在为波兰语在线社区提供更准确的情感分析，服务于这种在情感分析工具中通常代表性不足的语言。
- **MedScan AI：你的新医生？**：一位成员发布了 **MedScan**，这是一个基于 **Hugging Face 模型** 构建的 AI 工具，用于智能医疗搜索和报告分析，访问地址在 [这里](https://ragmedscan.vercel.app/)。
   - 它旨在为用户提供对相关医疗信息的 *快速访问*，并协助理解复杂的医疗报告，从而有可能 *提高医疗服务的可及性*。
- **持续学习（Continual Learning）需要重新思考**：一位成员解释了持续学习的挑战，认为目前的论文侧重于 *“治标不治本”*，并建议为模型制定 **设计原则**，使其根据奖励信号学习记住什么以及记住多少。
   - 他们建议 *优化函数* 应该是一个可学习的 **RNN**，致力于增加每个动作的奖励，这与 **Richard Sutton** 关于 AGI 中涌现式模仿学习的想法一致。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Datacurve 获得 1750 万美元融资用于训练前沿 LLM**：Datacurve 宣布完成了 **1500 万美元 A 轮**和 **270 万美元种子轮**融资，旨在为基础模型（特别是编程领域）提供高质量的训练数据集，详情见 [Serena Ge 的公告](https://xcancel.com/serenaa_ge/status/1976328983458480539)。
   - 该轮融资由 Chemistry 领投，YC、Cohere、Afore 及天使投资人参投，凸显了对专业化训练数据日益增长的需求。
- **Spellbook 为其 AI 合同工具筹集 5000 万美元 B 轮融资**：自称为“合同界的 Cursor”的 Spellbook 完成了由 Khosla Ventures 领投的 **5000 万美元 B 轮**融资，由 [Scott Stevenson](https://xcancel.com/scottastevenson/status/1976280608436572393) 宣布。
   - 该 AI 起草与审查平台自 2022 年以来已拥有 **4,000 名客户**，建议采纳率从 **5% 提高到 60%**，新资金将用于产品增强，首先从实时市场对比测试版开始。
- **Kernel 获得 2200 万美元以加强 LLM 云基础设施**：在 CEO Catherine Jue 的领导下，Kernel 披露了由 Accel 领投的 **2200 万美元种子轮 + A 轮**融资，旨在实现大规模浏览器工作负载自动化，并推出 Kernel Agent Authentication，如[此公告](https://xcancel.com/juecd__/status/1976325764166615498?s=46)所述。
   - 凭借 CashApp 和 Rye 等客户，Kernel Agent Authentication 旨在提供一个身份层，赋予 AI 应用对用户操作的安全、受限且可审计的控制权。
- **Elastic 收购 Jina AI 以实现更智能的搜索**：Elastic 已收购 Jina AI，以增强其检索、Embeddings 和上下文工程能力，从而赋能 Agentic AI，如 [Elastic 的公告](https://xcancel.com/elastic/status/1976278980018765886)所述。
   - 此次收购预计将巩固 Elastic 在企业搜索和 AI Agent 技术领域的地位。
- **Gemini Flash 2.5 Nano Banana：展示 JSON Prompting**：Emily 展示了一个用于 **Gemini Flash 2.5 Nano Banana** 的详细 **JSON Prompt**，该 Prompt 生成了一张蓝色调、动漫风格的东亚女性在电脑房卧室内的镜面自拍，详见[此推文](https://x.com/iamemily2050/status/1976431328280416520?s=46)。
   - 该演示引发了关于 **JSON** 与**自然语言 Prompting** 优劣的讨论，Emily 提倡 JSON 的可复现性和控制力，相关的复制案例发布在[此推文](https://x.com/toyxyz3/status/1976650667046605263?s=46)中。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Llama.cpp 努力应对混合 ROPE**：有人请求协助在 *llama.cpp* 中实现 **混合注意力 ROPE**，这与[此 GitHub pull request](https://github.com/ggml-org/llama.cpp/pull/16095#issuecomment-3390312481)相关。
   - 此外还注意到，尽管 **Hugging Face 实现** 的配置中存在 **partial ROPE**，但似乎并未使用它，参考了[这段 HF 代码](https://github.com/huggingface/transformers/blob/0419ff881d7bb503f4fc0f0a7a5aac3d012c9b91/src/transformers/models/qwen3_next/modeling_qwen3_next.py#L176)。
- **精度问题困扰 RoPE**：讨论指出 **RoPE 计算** 对精度非常敏感，建议正弦和余弦矩阵应以 **fp32** 计算以避免误差，引用了 [EleutherAI 的 gpt-neox](https://github.com/EleutherAI/gpt-neox/blob/d12c771198388980ee054617e537665f044e0584/megatron/model/positional_embeddings.py#L63) 作为参考。
   - 最近的一篇论文强调，在长上下文中使用 **BF16** 的 **ROPE** 可能会导致精度问题。
- **nGPT 奇怪的 OOD 爆发**：研究频道讨论了 **训练损失** 与 **分布外 (OOD) 泛化** 之间一种违反直觉的联系，质疑为什么 **nGPT 架构** 可能无法泛化。
   - 一种建议涉及 **长度泛化** 或对曝光偏差（exposure bias）的敏感性增加，特别是考虑到 *“已见”样本* 之间的泛化。
- **VLM 追求分辨率**：一位成员分享了一个 **基准测试项目** 的初步结果，该项目专注于在 Vision Language Models (VLMs) 中优化 **图像分辨率** 与 **输出质量** 之间的平衡，在 **COCO 2017 Val 数据集** 上使用 **Gemini 2.0 Flash** 进行图像描述任务，并附带了[报告](https://cdn.discordapp.com/attachments/747850033994662000/1425953022671847515/minres_report.pdf?ex=68eac73d&is=68e975bd&hm=194b9521fd489b4596bc9cdb078a3f90adc9533c1be618a2aaa1d0174f450c82&)。
   - 该基准测试为发现高效的视觉处理技术提供了路径。
- **LessWrong 为可解释性指明道路**：一位成员分享了一篇关于如何成为 **机械可解释性 (mechanistic interpretability) 研究员** 的 [LessWrong 文章](https://www.lesswrong.com/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher)。
   - 此外，一位成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=ruLcDtr_cGo)，对 **Anthropic 的 Circuit Tracing 和 Model Biology 论文** 中的 **归因图 (attribution graphs)** 进行了精彩介绍。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Atropos 教程视频发布**：使用 **Atropos** 的快速教程视频以及环境工作原理的更广泛概述现已在 [Twitter](https://fxtwitter.com/NousResearch/status/1925381160097697803) 和 [YouTube](https://www.youtube.com/watch?v=in__ELD4NxE) 上线。
   - 该视频配合了围绕 **Atropos** 的讨论，并回答了成员关于其使用和环境设置的问题。
- **vllm 发布 Predicted Outputs**：发布了一项名为 **Predicted Outputs in vllm** 的新技术，通过将输出转换为预测（部分）匹配的 prefill 来实现更快的生成；详情请参阅[博客文章](https://cascadetech.ai/blog/vllm-predicted-outputs/)、[演示](https://app.cascadetech.ai/)和 [推文线程](https://x.com/saganite/status/1976707696578691101)。
   - 这引发了关于该方法如何通过预计算潜在输出序列来显著加速 **vllm** 推理的讨论。
- **高效训练需要恒定参数比**：最近的一篇[论文](https://arxiv.org/html/2410.21228v1)指出，为了实现高效训练，比率 **(params * samples / data)** 需要保持恒定。
   - 成员们希望该论文能实验超出 **Chinchilla** 比例（260B tokens 训练 13B 参数）的训练 token 量，以获得更全面的见解。
- **LoRA 微调可能导致灾难性遗忘**：成员们注意到最近的一篇论文认为，仅看 **loss** 可能不足以判断微调过程中是否发生了灾难性遗忘。
   - 在新训练任务和预训练评估上同时进行评估，并检查 **LoRA 适配器的奇异值 (singular values)**，可能会对这一现象提供更多见解。
- **LoRA 目标仍能学习新事实**：一位成员认为，即使 **仅针对注意力层 (attention layer only targets)**，**LoRA** 也能学习新的事实和知识，反驳了其无效的说法。
   - 他们指向了 [Thinking Machines 博客](https://thinkingmachines.ai/blog/lora/)，该博客认为许多环境仅在注意力层使用 **LoRA**。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP 的 .well-known 端点讨论升温**：成员们正在讨论用于 **MCP server metadata** 的 `.well-known/` 端点，并引用了 [blog entry](https://blog.modelcontextprotocol.io/posts/2025-09-26-mcp-next-version-update/#server-identity)、[GitHub discussion](https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/1147)、[pull request](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/1054#issuecomment-3117705161) 以及 [Dev Summit presentation](https://github.com/modelcontextprotocol/registry/blob/main/docs/explanations/dev-summit-2025-10-registry-status-presentation.pdf)。
   - 目标是推进一个最小化的 SEP，重点关注文档名称、相对于 MCP server URL 的位置以及最小内容（特别是 `Implementation`）。
- **图像内容标准引发辩论**：讨论围绕在 `structuredContent` 中表示 **image content** 展开，担心其不可移植，且在直接将 `StructuredOutput` 传递给模型 API 的客户端上会失效。
   - 有建议认为协议并未规定宿主应用如何将 `structuredContent` 映射到 LLM API，且各供应商对返回图像的工具支持较差，给模型提供一个将字符串映射到图像的临时工具可能是一个解决方案。
- **Skybridge 工具绕过 OutputSchema**：讨论指出 **skybridge** 在工具上不使用 `outputSchema`，成员们探讨了是否有机会为此定义某种规范。
   - 成员们讨论了 `ContentBlock[]` 与 `structuredContent` 的差异，其中提到 `structuredContent` 用于组件渲染（widget hydration），但对模型也是可见的。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 被视为未来的语言**：成员们欢迎了一位新用户，并预测 *Mojo será el lenguaje del futuro* <:mojo:1098996988797784084>。
   - 该用户还表示，*También soy de Colombia 🙈*。
- **新 GPU 引发向后兼容性担忧**：一位用户质疑某种方法是否缺乏对 GPU 的前向兼容性。
   - 他们问道：“这种方法难道没有前向兼容性的问题吗？为了支持每一代新的 GPU，你必须重新编译，不是吗？”
- **Apple Silicon M3 实现 GPU 向量加法**：一位成员在 **Apple Silicon M3** 上测试了 **GPU support**，运行了一个简单的向量加法示例，但通过显式同步解决了宿主缓冲区未被修改的问题。
   - 此外还提到，Apple GPU 尚未实现 *printing from inside kernel*（从内核内部打印），且由于共享内存的原因，`enqueue_copy` 可能是空操作。
- **Jetson Orin Nano 为机器人技术提供强劲动力**：成员们讨论了将 **Jetson Orin Nano 8GB** 用于机器人技术，认为它足以开发视觉系统，特别是对于 **battery life**（电池续航）至关重要的资源受限部署。
   - 他们强调目标检测和图像分类模型在 Orin Nano 上运行良好，之后可以扩展到更大的系统。
- **Mojo 原生 FFT 落地，准备向 FFTW 发起挑战？**：在 PR [#5378](https://github.com/modular/modular/pull/5378) 之后，将会有 **Mojo native FFT implementation**。
   - 成员们还讨论了与 **FFTW** 相比的性能提升，一位用户表示他们需要大小仅在运行时确定的多维变换，FFTW 在 CPU 上已经足够好，但如果最终能有 GPU 实现就更好了。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **仅靠架构技巧无法解决遗忘问题**：一位成员认为，防止**灾难性遗忘 (catastrophic forgetting)** 不仅仅需要架构上的改变；优化器必须进化到**基于自动微分 (autodiff) 的梯度下降**之外。
   - 他们建议将记忆**硬分区 (hard-sectioning)** 到不同的桶中，并确保这些桶不会发生根本性的遗忘，并指出需要 **i.i.d. 采样**的模型天生难以应对遗忘。
- **IID 采样对 DL 证明是有问题的**：一位成员指出，当数字被排序而不是随机且不采样旧数字时，**深度学习 (DL)** 方法在处理 **MNIST** 时会遇到困难。
   - 他们指出，没有 i.i.d. 的 DL 准确率只有 10%（随机水平），这突显了有效学习对 **i.i.d.** 的依赖。
- **独立的 Token Embeddings 能防止注入吗？**：一位成员提议为**系统/用户提示词 (system/user prompts)** 训练一套独立的 **token embeddings**，使模型更容易区分提示词和内容，从而降低**提示词注入 (prompt injection)** 的风险。
   - 作为回应，另一位成员指出，这些独立的 embeddings 被称为**软提示 (soft prompts)**，可以防御**提示词注入 (prompt injection)** 攻击。
- **Camel AI 换新装**：一位用户分享了 [Camel AI](https://github.com/camel-ai/camel) 获得了一些更新，并认为**角色扮演方法 (roleplay method)** 非常棒，有必要测试 **Workforce** 和所有工具。
   - 未发现关于具体功能的进一步讨论。
- **伯克利发布 LLM Agents 课程**：一位成员分享了来自伯克利的两个 LLM Agents 课程，可能值得一看：[playlist 1](https://www.youtube.com/playlist?list=PLS01nW3RtgopsNLeM936V4TNSsvvVglLc) 和 [playlist 2](https://www.youtube.com/playlist?list=PLS01nW3RtgorL3AW8REU9nGkzhvtn6Egn)。
   - 未提供关于课程的进一步讨论或见解。



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **GLM 4.6 订阅成功集成**：成员们确认 **GLM 4.6 订阅计划** 可以在 Aider 中运行，参考 [OpenAI 兼容指南](https://aider.chat/docs/llms/openai-compat.html) 进行设置，特别是使用 *pass* 端点。
   - 这一讨论源于成员们探索使用 **n8n 工作流** 作为 API 端点来路由提示词，利用较小的本地模型在调用更大的云端模型之前剥离环境密钥。
- **Aider 采用 Haiku 生成 Git 提交**：Aider 现在使用 **Haiku** 生成 git 提交信息，优先考虑速度和成本效益，具体使用的是 `openrouter/anthropic/claude-sonnet-4`。
   - 一位成员建议将 **gemini flash lite** 作为一种具有成本效益的替代方案，并建议为提交信息设置一个**弱模型 (weak model)** 以优化资源利用。
- **OpenRouter 释放自定义配置文件功能**：用户可以在 **OpenRouter** 中管理自定义配置文件，为模型指定自定义提示词、温度 (temperatures) 和推理复杂度，从而增强模型管理。
   - 这些配置文件在 aider 配置中或通过 `/model` 和 `/editor-model` 命令指定，指向 `aider/resources/` 目录中的模型定义。
- **Aider 模型规范语法详情**：Aider 中的 `/model` 和 `/editor-model` 聊天命令允许用户指定模型，包括那些未在 `.aider.conf.yml` 文件中定义的模型，提供了模型选择的灵活性。
   - 这种即时选择补充了基于配置的模型设置，简化了模型调整过程。
- **人格定义作为只读资产**：用户询问是否可以将**人格定义 (persona definitions)**（例如来自 [vibecodingtools.tech](https://www.vibecodingtools.tech/templates) 的定义）作为 `/read-only` 资产推送到底层模型。
   - 建议仅在任务切换时（例如从 Planning 切换到 Coding）加载这些人格，而不是针对每个请求都加载，并将其推送到底层模型。



---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Sora 2 邀请码充足**：成员们分享说 **Sora 2 邀请码** 随处可见，该产品的下载量已超过 *100 万次*。
   - 尽管供应充足，一些成员表示更倾向于等待 **正式发布 (public release)**。
- **Kimi 的编程实力大放异彩**：**Kimi** 展示了强大的编程能力，通过 **IDE** 采用 **Agent 模式** 和 **工具调用 (tool usage)** 来执行 **Python 脚本** 和 **批处理命令** 以进行系统调试。
   - 一位成员断言 Kimi 的编程表现超越了 *大多数其他模型*。
- **Hack Club 与 Moonshot AI：独立的实体**：关于 **Moonshot AI** 与来自 **Hack Club** 的邮件之间是否存在潜在联系引发了讨论。
   - 经澄清，**Hack Club** 和 **Moonshot AI** 是两个不同且无关的组织。
- **使用 Kimi 制作的神秘视频**：某些成员提到使用 **Kimi** 制作了 *疯狂的视频*。
   - 然而，关于这些视频内容或性质的具体细节并未透露。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 社区辩论 PR 中的 “AI Slop”**：一位成员质疑 [PR #12530](https://github.com/tinygrad/tinygrad/pull/12530) 是否包含 *AI Slop*，认为将其斥为晦涩难懂是逃避责任，并为代码质量担保，特别提到了 [PR #12539](https://github.com/tinygrad/tinygrad/pull/12539)。
   - 提交者将其与 **geohot** 在 [issue #12449](https://github.com/tinygrad/tinygrad/pull/12449) 中的代数 Upat 测试进行了比较，但另一位成员提到 *AI PR 将在不予置评的情况下关闭，如果你强行推送，将会被封禁。*
- **Reduce 组变为红色以提高清晰度**：**reduce** 的组现在被标记为 *鲜红色* 而非绿色，以强调正在使用 local 进行 reduction。
   - 此更改指定绿色将保留给未来的功能，参见 [PR #12604](https://github.com/tinygrad/tinygrad/pull/12604/files)。
- **`cuda_ioctl_sniffer` 获得 Rust 重构**：一位成员正在将 George Hotz 的 `cuda_ioctl_sniffer` 转换为 **Rust**，并配备交互式终端以测试单个 **CUDA kernels**。
   - 他们发布了 `saxpy` kernel 输出的 [演示图像](https://cdn.discordapp.com/attachments/1070745817025106080/1425975923458445343/image.png?ex=68eadc91&is=68e98b11&hm=8472f883712ba4af77167cb978546cf31d701a98b6a67bcca03ea1b59fdab985)，旨在支持更多 GPU，使用 IOCTL 启动 **CUDA kernels**。
- **Winograd 测试断言失败**：在尝试实现循环拆分（loop splitting）时，一位成员在 `test_winograd.py` 中遇到了 **断言失败**。
   - 错误信息显示数值 **6.49** 在与 **2.6** 的 *不小于* 比较中失败。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Spotlighting 减少 Prompt Injection**：一位成员演示了 **spotlighting** 以降低 Prompt Injection 的风险，引用了 [Microsoft 的研究论文](https://github.com/estsauver/dspy-spotlight)。
   - 该成员指出，他们仍在开发针对 **XPIA 攻击** 的基准测试和测试套件。
- **DSPy 社区仓库让社区项目备受关注**：一位成员创建了 [DSPy Community 仓库](https://github.com/dspy-community) 以突出展示项目，防止它们消失在虚无中。
   - 目前，它是主页上的一个 **README**，列出了库和项目，**欢迎提交 PR**。
- **MCP Tool 身份验证难题**：一位成员提出了关于从具有 [身份验证](https://github.com/modelcontextprotocol/python-sdk?tab=readme-ov-file#authentication) 的 **MCP Tool** 创建 `dspy.Tool` 及其处理方式的问题。
   - 该询问集中在当对需要身份验证的工具使用 `dspy.Tool.from_mcp_tool` 时，身份验证过程是否得到了正确管理。
- **shadcn 激发了 DSPy 网站改版的构想**：受 **shadcn** 启发，一位成员建议 DSPy 可以从 *浏览器网站、用于一致模块放置的 CLI* 以及 *向仓库发布优化模型的方法* 中受益。
   - 其想法是让用户能够轻松适配模块，从 `pip install` 转向更简单的自定义。
- **DSPy 模块市场受到关注**：社区成员正在推动建立一个 DSPy 模块的 **平台/市场**，以促进优化程序的共享和重用。
   - 该市场将托管优化后的程序，例如针对 **Qwen**、**4.1-mini** 和 **4.1-nano** 优化的客户评论分类任务，允许用户快速部署常见任务的解决方案。

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Godhand AI 辅助 Previz 创作工作流现身**：一篇 [LinkedIn 帖子](https://www.linkedin.com/posts/godhand_ai-assisted-previz-creation-workflow-quick-activity-7382046352287088640-Ev0V) 重点介绍了 **Godhand** 的 AI 辅助 Previz（预演）创作工作流，承诺提供一种更快速、更高效的 Previz 方法。
   - 该工作流有潜力显著缩短项目初始规划阶段的制作时间。
- **用户大声疾呼要求支持人员介入**：多名用户在频道中表达了挫败感，并要求支持人员立即关注。
   - 一位用户惊呼：*"有人吗！！！支持人员在哪里？！"*，凸显了对平台响应速度的不满。
- **Manus 在初始项目结构方面表现出色**：一位成员报告称，发现 **Manus** 在初始规划和构建项目方面非常高效，并提到构建一个 RAG 宠物助手向量数据库仅花费了 **1500 credits**。
   - 该用户建议利用 **Manus** 进行规划，然后过渡到 **Claude Code** 进行编码阶段，通过 Prompt 和 n8n 工作流来简化流程。
- **Prompt Engineering：成功的关键**：频道成员强调了在使用 **Manus** 等 AI 工具时编写明确且详细的 Prompt 的重要性。
   - 一位用户警告说，*直接把文件丢给任何 AI 并让它自己去揣摩 Prompt 的细节是非常糟糕的做法*，强调了精心设计 Prompt 的必要性。
- **Claude API 无缝集成发布**：一位用户宣布，现在可以通过 API 调用将 **Claude** 无缝集成到 **Manus** 中，无需再进行复制粘贴。
   - 这简化了工作流程，并通过在 **Manus** 中直接访问 **Claude** 的功能来提升用户体验。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Diffusion Model 论文研读小组启动**：一个新的 **Diffusion Model 论文研读小组** 即将启动，将于本周六 **PST 上午 9 点 / EST 中午 12 点**举行会议（旧金山线下 + 线上混合模式），讨论 [*Denoising Diffusion Implicit Models (DDIM)* 论文](https://arxiv.org/abs/2010.02502)。
   - 该课程对具备 **Python + 基础 PyTorch** 知识的初学者友好，参与者可以通过[提供的链接](https://luma.com/vioj15il)进行 RSVP。
- **DDIM 论文加速图像生成**：研读小组将讨论 **DDIM** 如何在保持高质量的同时加快图像生成速度，这是 **Stable Diffusion** 的基础。
   - 这篇由 Song 等人于 2020 年发表的论文 [*Denoising Diffusion Implicit Models (DDIM)*](https://arxiv.org/abs/2010.02502) 被认为是 Generative AI 工程师的核心知识。
- **Diffusion & LLM 训练营发布**：该研读小组会议是为期 **3 个月的 Diffusion Model 训练营**（2025 年 11 月）的一部分，灵感来自 **MIT 的 Diffusion Models & Flow Matching 课程**，参与者包括 AI 和软件工程师、PM 及创作者。
   - 该训练营提供构建和训练你自己的 **Diffusion Model + ComfyUI Pipeline + GenAI 应用开发**的实战经验。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该社区长时间保持沉默，请告知我们，我们将将其移除。

---

**Windsurf Discord** 没有新消息。如果该社区长时间保持沉默，请告知我们，我们将将其移除。

---

您收到此邮件是因为您通过我们的网站订阅了相关内容。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。

---

# Discord：各频道详细摘要与链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1425921587525455904)** (1204 messages🔥🔥🔥): 

> `Perplexity AI Models, Comet Browser, AI code debugging, AI programming language, Coding on phones` 


- **GPT-5：编程高手还是菜鸟？**：成员们就最适合编程的 AI 模型展开辩论，一些人更喜欢 **GPT-5** 的响应速度和理解能力，而另一些人则认为 **Claude 4.5** 在规划和研究方面表现更好。
   - 有辩论称使用 **GPT-5** 编程更好，因为有说法称它能生成 *更优化的代码*。
- **Perplexity Pro 有限制了？**：用户报告称，即使是 **Perplexity Pro**，也存在搜索限制（每天约 **300** 次），并且如果输入速度过快，会被标记为垃圾邮件发送者。
   - 一些成员注意到，使用 **Perplexity Pro**，*视频生成也被限制为每月 5 个*。
- **Comet Browser 移动版发布在即**：成员们焦急地等待 **Comet Browser** 的移动版发布，预计发布日期在 *年底左右*。
   - 一位用户表示 *Comet 在考试中表现良好，会为你选择正确答案*。
- **Perplexity 到底是不是模型？**：成员们讨论了一个担忧，即当你选择 **GPT-5** 时，**Perplexity AI** 是否真的使用了它，并暗示他们可能使用了低功耗模型或 Perplexity 助手。
   - 一位成员指出 *AI 会保存你的说话方式和风格*，这可能就是为什么 Perplexity 看起来有所不同的原因。
- **用于调试的 AI 工具**：询问错误问题的最佳 AI 是 **Cursor** 中的 **GPT-5**。
   - 一位成员建议用户 *确保 Prompt 质量*，因为 *它们决定了结果的差异*。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1426169670108581970)** (1 messages): 

> `GPTs Agents, OpenAI's sidebars` 


- **GPTs Agents 在训练后保持无状态**：一位成员询问 GPTs Agent 是否会从初始训练后添加的新信息中学习。
   - 另一位成员澄清说，上传的文件被保存为 Agent 的“知识”文件，但 **不会更新 Agent 的基础知识**；[文件在需要时会被引用](https://link.to/openai-docs)。
- **OpenAI 平台侧边栏 UI 变动**：一些成员注意到 platform.openai.com 上的用户界面变化，特别是在侧边栏。
   - 一位用户报告称 **两个图标消失了**：一个是 threads（线程），另一个是 messages（消息）。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1426247987549438094)** (2 messages): 

> `Search API issues, Permission Denied Error, Cloudflare deny` 


- **Search API 抛出 Permission Denied 错误**：一位成员报告了在使用 **Perplexity Search API** 时遇到的奇怪问题，遇到了一个似乎与 **Cloudflare** 拒绝有关的 `PermissionDeniedError`。
   - 目前尚不清楚为什么 API 会被 Cloudflare 拦截。
- **Search API 被 Cloudflare 拒绝的可能原因**：用户正在寻求使用 **Search API** 时被 **Cloudflare** 拒绝的潜在原因。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1425920856923705445)** (1202 条消息🔥🔥🔥): 

> `LM Arena 定价, Sonnet 4.5 错误, Video Arena 功能, Gemini 3.0 发布, AI 视频生成` 


- **LM Arena 的命运：永远免费吗？**：成员们讨论了 LM Arena 的定价前景，[一位用户指出服务条款 (ToS)](https://lmarena.ai/tos) 允许他们随时收费，因为运行成本高达数百万美元。
   - 一位维护者回复称，*我们的意图是保持对所有人免费和开放*，并进一步指出他们正在探索[替代策略](https://news.lmarena.ai/ai-evaluations/)以避免收取费用。
- **Sonnet 4.5 Thinking 表现异常**：成员们遇到了 **Sonnet 4.5 Thinking** 失败并不断抛出错误的问题，尤其是在长时间对话之后。
   - 其他人指出可能是 API 问题，一些人评论说他们也遇到了同样的情况，还有建议称清除 Cookie 可能会解决问题，但这似乎并非对所有人都有效。
- **Video Arena 缺乏模型选择**：成员们讨论了 **Video Arena** 的局限性，特别是缺乏模型选择、视频请求受限、使用图像功能需要登录以及整体成本高昂。
   - 一位成员指出：*制作视频比普通的文本机器人更贵（我认为），这就是它受限的原因*。
- **Gemini 3.0：十月的幻影？**：成员们就 **Gemini 3.0** 的发布日期展开了辩论，争论它是仍按计划在 10 月发布，还是推迟到了 12 月。
   - 一些成员声称它正在 AiStudio 内部进行 A/B testing，但这无法得到证实。
- **Grok 视频获得无限关注**：成员们探索了 **Grok** 的视频生成能力，对其无审查、支持音频和无限使用的特性印象深刻。
   - 缺点是 *1:1 视频的输出分辨率仅为 560 x 560*，并且可能会产生水印，但 *它是免费的，老弟*。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1425953912422269070)** (1 条消息): 

> `LMArena 调查, Arena Champions 计划` 


- **LMArena 通过调查征求用户反馈**：LMArena 正在通过[此调查](https://docs.google.com/forms/d/e/1FAIpQLSevxsX_kvJ_fiv74Rcf2yPl9lnSNOtmmb_wMnBCy1fEri_jEg/viewform?usp=dialog)收集用户反馈以改进产品。
   - 该调查旨在了解对用户而言什么是重要的，从而使 **LMArena** 成为一款出色的产品。
- **Arena Champions 计划启动**：LMArena 推出了 **Arena Champions 计划**，旨在奖励那些对有意义的对话表现出真诚投入的成员。
   - 在此[申请](https://docs.google.com/forms/d/e/1FAIpQLSdRWfqG8_MMKQ4H23FHFZVJsg0OuQrZqn5h9l-QqhWpNI77xg/viewform?usp=dialog)加入该计划，以获得进入私密空间进行无干扰交流的机会，申请要求展示对 **AI 的兴趣**以及对**有意义对话的投入**。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1425923461150146651)** (493 messages🔥🔥🔥): 

> `LinuxComet 隐私问题, 浏览器推荐, DuckDuckGo 隐私, AI Agency 商业模式, Sora2 在欧盟的可用性` 


- **LinuxComet 被指侵犯隐私**：一位成员批评了来自 Perplexity 的浏览器 **LinuxComet**，称其为一个收集数据量超过主流浏览器总和的“巨型追踪器”，称其表现“极差（terribad）”并警告不要付费使用。
   - 他们讽刺地评论道，任何“声称关心隐私”的人几乎肯定都不关心，并提倡使用 **Safari** 或 **Firefox** 等浏览器，同时对 Apple 的遥测（telemetry）实践表示质疑。
- **浏览器选择反映隐私立场？**：成员们讨论了浏览器的隐私性，一位成员偏好 **DuckDuckGo**（因为它是在 Chrome 基础上封装的隐私导向浏览器），另一位则支持 **Brave**（因为它通过观看广告提供加密货币奖励）。
   - 有人担心 **Firefox** 是由 Google 资助的，暗示尽管它主打隐私，但仍可能存在追踪行为。一位用户开玩笑说：“为了隐私而使用 Firefox 就像……戴着锡纸帽挡广告，却站在发布在 Google Maps 上的玻璃房里。”
- **预测 AI 浏览器将占据主导地位**：一位成员预测 **浏览器中的 AI** 将占据主导地位，将浏览器定位为现代操作系统，并赞扬了 **Comet AI** 的发展方向，尽管目前的计算机视觉模型还存在局限性。
   - 另一位用户感叹无法共享 ChatGPT 订阅，称其为企业压迫，而其他人则在争论 AI 集成浏览器的定义和实用性。
- **伦理 AI Agency**：围绕 **AI Agency** 概念展开了讨论，重点在于自动化流程和 Discord 响应，一位成员建议需要具备 **MBA** 背景的人来评估盈利能力。
   - 讨论强调了帮助公司实现流程自动化的目标，但警告需要有商业背景的人来确保盈利。
- **三星微型 TRM 超越大型模型**：成员们讨论了三星的微型递归模型（**TRM**），强调了其在标准 AI 基准测试中出人意料的表现，TRM 在 ARC-AGI-1 测试中达到了 **44.6%** 的准确率。
   - 其得分超过了 DeepSeek-R1、Google 的 Gemini 2.5 Pro 和 OpenAI 的 o3-mini 等大得多的模型。然而，讨论也对 TRM 针对该基准测试的高度专业化与 LLM 的通用目的之间的关系提出了质疑。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1425927860593623131)** (17 messages🔥): 

> `OpenAI 责任豁免, OpenAI 的责任划分, ChatGPT Business vs Enterprise, GPT-5 Thinking Mini, MCP 开发频道` 


- **OpenAI 面临责任问题，用户建议签署豁免书**：一位用户建议 OpenAI 制定一份**法律责任豁免书**，要求用户签署以对自己的行为负责，旨在减少 OpenAI 的法律责任，而不是“削弱模型的实用性”。
- **ChatGPT Business 由 ChatGPT Teams 更名而来**：根据[发布说明](https://help.openai.com/en/articles/11391654-chatgpt-business-release-notes)，截至 *2025 年 8 月 29 日*，**ChatGPT Team** 已更名为 **ChatGPT Business**。
- **用户请求控制 GPT-5 Thinking Mini 的行为**：一位用户正在寻求**禁用 GPT-5 Thinking Mini** 的方法，因为它会自动从 **GPT-5 Instant** 切换，并针对其使用场景给出不令人满意的结果。
   - 消息中未提供解决方案。
- **社区提议为 OpenAI 集成设立 MCP 开发频道**：一位成员提议设立一个新的 **MCP 开发频道**，专注于 MCP 服务器的集成、用途以及在 OpenAI 产品中的体验，参见 [Discord 频道链接](https://discord.com/channels/974519864045756446/1426291246132887552)。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1426001506980597922)** (13 messages🔥): 

> `Sora 2 提示词编写, 视觉学习, 通过示例学习 Prompt Engineering` 


- **关于最佳 Sora 2 提示词的争论异常激烈**：成员们就 **Sora 2** 是否存在“最佳通用提示词”展开辩论，一位成员表示“好的提示词应包含你想要看到的细节和特质”。
- **通过示例学习 Prompt Engineering**：一位成员建议通过寻找一段看起来很棒的视频，复制其提示词，重构它，然后更改内容来学习 Prompt Engineering。

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1426001506980597922)** (13 条消息🔥): 

> `Sora 2 提示词编写，绕过指南` 


- **关于“最佳” Sora 2 提示词的辩论兴起**：一位用户询问了 **Sora 2** 的 *最佳全能提示词*，但一位成员表示不存在这种东西，好的提示词应该包含你想要看到的 **细节和品质**。
- **从 Sora 视频中逆向工程提示词**：一位用户建议通过寻找一个 *惊艳* 的视频，复制其提示词，对其进行重构，并微调内容来观察文本如何影响视频生成，从而进行逆向工程。
   - 这可以让你看到 *哪些文本影响了什么，以及如何进行微调和调整*。
- **为 Sora 2 编写提示词就像为 ChatGPT 编写提示词**：一位成员表示 *chat gbt 无法工作*，并解释说他们 *更倾向于视觉学习*，并且正在 *尝试实际学习* 提示词相关的知识。


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1426263729888301257)** (2 条消息): 

> `VST 流行度，YouTube 频道订阅者和观看次数` 


- **GitHub 上的 VST 流行度竞赛**：一位成员分享了一个 [GitHub 仓库](https://github.com/MikesRuthless12/tools-hub)，用于根据网站流量跟踪 **VST** 的流行度。
   - 该仓库还包含 **YouTube 频道** 的数据，跟踪其当前的订阅者数量和月平均观看次数。
- **YouTube 频道洞察**：[链接的 GitHub 仓库](https://github.com/MikesRuthless12/tools-hub)提供了与 VST 相关的 **YouTube 频道** 指标，包括订阅者数量和月平均观看次数。
   - 这些数据对于评估 **VST 社区** 内不同频道的触达率和参与度非常有价值。


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1425923625369997496)** (401 条消息🔥🔥): 

> `BYOK 支付问题，Gemini 3 发布，通过 API 限制 PDF 页数，角色扮演者与免费模型，SSE 的使用数据问题` 


- **BYOK 需要支付？用户报告**：一些用户在使用 **BYOK** (Bring Your Own Key) 时遇到问题，指出他们被要求 **Payment Required**，且未被重定向到他们连接的密钥，导致账户被扣费。
   - 截至讨论时，此问题尚未解决。
- **DeepSeek 速率限制令角色扮演玩家感到沮丧**：用户对 OpenRouter 上的 **速率限制** 和 **DeepSeek 模型移除** 感到沮丧，正在为角色扮演寻找 **无审查的免费替代方案**。
   - 一位用户正在寻找可以在 4070 32gb 笔记本电脑上 *本地运行的模型*，用于在 **DeepSeek 3.1** 受限后进行角色扮演。
- **排查 SSE 使用数据检索问题**：用户报告在 OpenRouter API 中使用 **SSE** (Server-Sent Events) 时无法检索到 **使用数据 (usage data)**，具体表现为接收到的消息中不包含 *usage* 对象。
   - 一位用户发现 OpenRouter API 在发送 usage 数据块 **之前** 就发送了带有 *finish_reason: stop* 的数据块，导致 **litellm** 提前终止迭代，并提供了一个 [潜在的修复方案](https://github.com/BerriAI/litellm/issues/11626)。
- **OpenRouter 的 Alpha Responses API 面临停机**：用户报告在使用 alpha responses API 时遇到 **500 Internal Server Errors**。
   - 随后确认该 API 确实出现了故障，但现已解决：*抱歉，我们的 alpha responses API 挂了一会儿！现在应该可以再次工作了*。
- **GLM 4.5 和 4.6 对部分用户无法工作**：用户报告 Chutes 提供的 **GLM 4.5** 和 **GLM 4.6** 在 OpenRouter 内部无法工作，而其他人建议使用 **GLM 4.5 air (free)** 并选择 **Z.ai** 作为提供商以避免错误。
   - 与此同时，有讨论称 Google 正在将服务器重新分配给 **Gemini 3.0**，导致据报道 **2.5** 的质量有所下降。


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1425978657616429168)** (2 条消息): 

> `` 


- **未发现关于新模型的讨论**：提供的消息中没有关于新模型的讨论。
- **频道在模型更新方面保持沉默**：根据给定的消息历史，'new-models' 频道似乎没有相关的活动可以总结。


  

---

### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1425979781715066963)** (12 messages🔥): 

> `Sambanova Deepseek R1/V3, BYOK Azure Keys Routing, ChatQwen Models` 


- **Sambanova 的 Deepseek R1/V3 质量检查**：一名成员询问了 **Sambanova** 模型的 **Deepseek R1/V3** 系列质量，并寻求其他使用者的反馈。
- **Azure BYOK 密钥遇到路由问题**：一位用户报告了他们的 **Azure BYOK 密钥设置**问题，具体表现为尽管启用了 *"Always use this key"*（始终使用此密钥）功能，流量仍被路由到 **OpenRouter** 的 **OpenAI**；另一位用户询问将 **OpenAI** 设置为忽略的提供商（ignored providers）是否能解决此问题。
   - 该用户随后找到了 [BYOK - Bring Your Own Keys to OpenRouter](https://openrouter.ai/docs/use-cases/byok#azure-api-keys) 文档来解决此问题。
- **Qwen 下周将推出更多模型**：根据 [X 上的帖子](https://x.com/JustinLin610/status/1976681042041028823)，一位用户分享了 **Qwen** 计划在下周发布更多模型的消息。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1425924739028746393)** (115 messages🔥🔥): 

> `Qwen3-VL Support, DGX Spark performance, trl library changes, Pretraining datasets, Fine-tuning vulnerabilities` 


- **Qwen3-VL 支持状态引发讨论**：关于 **Unsloth** 是否支持 **Qwen3-VL** 存在争议；一位成员表示它 *应该已经在 Unsloth 中运行*，另一位则表示 *只需要禁用自动编译器（auto compiler）*。
   - 一位用户确认使用 `os.environ['UNSLOTH_COMPILE_DISABLE'] = '1'` 禁用自动编译器后可以正常工作，并消耗了约 **44GB VRAM**。
- **Spark 的微调可行性受到质疑**：一位用户对 **DGX Spark** 的微调性能表示好奇，尽管其 FLOPS 表现不错，但由于其带宽问题而持怀疑态度。
   - 另一位看过现场演示的用户表示，对于 **32B** 模型，它的速度 *慢得离谱，不到 10/s*，但他们希望软件优化能使 **LoRA** 变得可行，因为 AMD 的替代方案缺乏 **CUDA**。
- **TRL 库重构引发焦虑**：上游 **TRL 库** 正在考虑移除多个训练器（**CPO/SimPO, KTO, OrPO, PPO, RLOO**）且不保留支持，这导致了对重构的担忧，详见 [此 GitHub issue](https://github.com/huggingface/trl/issues/4223)。
   - 尽管存在担忧，但目前看来 **KTO, ORPO 和 PPO** 正在被移至 **trl.experimental**。
- **对预训练数据集的需求增加**：一位成员正在寻找 **<=10B tokens** 的优质预训练数据集，相比随机子集，更倾向于经过过滤的数据集。
   - 他们被推荐了 [Ultra-FineWeb](https://huggingface.co/datasets/openbmb/Ultra-FineWeb)，但由于该机构的模型有“刷榜”（benchmaxxed）嫌疑，他们对该数据集的基准测试实践持怀疑态度。
- **微调漏洞探讨**：一位成员询问了专注于微调的特定漏洞，并为一个项目启动了针对 AI 特定 **CVE** 的调查。
   - 另一位成员强调了 **safetensors** 格式，以及 Hugging Face 为了安全禁用了在下载模型中运行脚本的功能，并引用了 [这篇 HuggingFace 博客文章](https://huggingface.co/blog/safetensors-security-audit)。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1426066896121364520)** (2 messages): 

> `AI Development, Full-Stack Engineering, Blockchain Building, AI + Web3 Projects` 


- **AI 开发者加入 Unsloth Discord**：一位具有全栈工程和区块链构建经验的 AI 开发者在 Discord 频道介绍了自己。
   - 他们对 *突破 **AI + Web3** 边界* 的项目持开放态度。
- **全栈工程师寻求 AI + Web3 项目**：一位全栈工程师分享了他们在 AI 模型设计、区块链网络和精美前端方面的背景。
   - 他们表达了对结合 **AI 和 Web3 技术** 项目的兴趣，并强调了自己在交付端到端解决方案方面的经验。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1425923422126346320)** (173 条消息🔥🔥): 

> `数据中心冷却用于市政供暖，再加一个参数，RAM 占用与浏览器，Zen Browser，Batch Size 与 Forward Pass 时间` 


- **数据中心余热为家庭供暖？**：成员们讨论了利用**数据中心冷却水**进行市政供暖的话题，指出虽然这已经在实施，但尚未形成规模，并且 [Google 在芬兰已经有一个项目](https://blog.google/around-the-globe/google-europe/our-first-offsite-heat-recovery-project-lands-in-finland/)。
   - 与其直接净化水，不如使用热交换器在两个水循环之间传递热量，以避免家用系统出现昂贵的故障，这类似于**苏联**使用热电中心的方式。
- **兄弟，再加一个参数就好！**：一位成员开玩笑说人们对 LLM 更多参数的无止境追求，调侃道：*“兄弟，再加一个参数，它就会成为史上最强的 LLM，我们需要更多参数，兄弟，更多参数能解决一切，求你了，再加一个就行”*。
   - 其他成员也纷纷加入，表达了类似的情绪，比如 *“求你了，再加一层 🥺”* 和 *“就一百万个，兄弟~~~”*。
- **吞噬 RAM 的浏览器**：成员们讨论了 RAM 占用问题，有人指出 *“现代网站疯狂吞噬内存”*，一个 **YouTube 标签页**就能占用 **500MB+ RAM**。
   - 另一位建议使用标签页卸载器 (tab unloader)，并指出 *“Google 主页占用 150-160mb RAM 简直不可理喻”*，将其归咎于页面过多的依赖项和 **Chrome 的垄断**。
- **Zen Browser 获赞，发现 Firefox 替代方案**：一位成员推荐 **Zen Browser** 非常好用，因为它可以自动卸载标签页。
   - 据说它是 **Arc Browser** 的开源版本，后端使用 **Firefox/Gecko**。
- **Batch Size 会影响 Forward Pass 时间吗？**：一位成员询问 Batch Size 为 5 的 Forward Pass 是否与 Batch Size 为 50 的 Forward Pass 耗时相同。
   - 解释称 *“如果你的 GPU 可以同时处理所有内容，那么 1x50 将比 10x5 快得多”*，这取决于计算瓶颈与内存瓶颈，更高的 Batch Size 耗时更长，但具有更高的吞吐量。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1425932159415357522)** (84 条消息🔥🔥): 

> `在 Amazon ml.g4dn.xlarge 上安装 Unsloth，从 Lora 创建 GGUF 模型，分布式 FSDP 运行支持，WSL2 包兼容性问题，用于生成测试用例的 AI code agent` 


- **在 Amazon ml.g4dn.xlarge 实例上运行 Unsloth**：一位用户询问如何在 Amazon **ml.g4dn.xlarge** 实例上安装 Unsloth，提到 Google Colab 可以正常工作但 Amazon 环境很复杂，随后被引导至 Unsloth 的 [Docker 镜像](https://hub.docker.com/r/unsloth/unsloth)。
- **GGUF 转换难题**：用户报告称，由于运行时错误和量化问题，使用 `model.save_pretrained_gguf()` 将模型保存为 **GGUF** 会失败，即使在 [官方 Ollama notebook](https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing) 中也是如此。
   - 建议目前仍需进行手动转换。
- **WSL2 配置困境**：一位用户在 WSL2 上为摘要数据集微调 **Mistral 7B** 时，遇到了包不兼容问题（CUDA 版本与 **xformers** 不匹配）。
   - 一位成员建议卸载并重新安装 **torch**、**torchvision** 和 **torchaudio**，并提供了从 GitHub 安装 **xformers** 的命令。
- **使用 LLM Agents 和 DeepSeek API 生成测试用例**：一位成员正在寻求帮助，希望仅使用 **DeepSeek API** 开发一个 **AI code agent**，以便根据给定问题生成严格正确的测试用例。
   - 这里的重点是让 Agent 根据给定问题生成代码并生成正确的测试用例。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1426320638653497539)** (1 条消息): 

> `Qwen3-8B 微调，小说章节训练，数据清洗` 


- **Qwen3-8B 学习小说写作**：一位成员在约 **8k** 个真实小说章节上微调了 **Qwen3-8B**，以评估数据集的质量。
   - 虽然结果显示出潜力，但该成员观察到模型继承了 **Qwen** 的重复问题，建议需要更多的 epochs 和更好的数据清洗。
- **强调数据清洗的重要性**：在 **Qwen3-8B** 上的实验揭示了在对提取的小说章节进行训练时，彻底进行数据清洗的重要性。
   - 提取过程中的伪影 (Artifacts) 影响了模型的性能，强调了细致预处理的必要性。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1426261254925848727)** (27 messages🔥): 

> `Recursive Model, HRM, ARC-AGI, Data Augmentation, GNN + Reasoning Model` 


- **7M 递归模型在 ARC-AGI 取得 SOTA**：一名成员分享了一篇关于 [7M 递归模型](https://arxiv.org/abs/2510.048717) 的论文，该模型仅有两层，在 **ARC-AGI 1 和 2** 上达到了 **SOTA**，重点在于深度监督以及防止崩溃/发散的保护机制。
   - 其他人指出，虽然令人印象深刻，但由于模型参数量较小，且 **ARC-AGI** 使用私有测试集进行测试，该模型的**泛化能力**可能有限。
- **基于数据增强训练的 HRM**：有人指出，**HRM** 模型虽然有效，但可能由于在接近数据泄露边缘的数据增强上进行训练，从而对 **ARC AGI** 产生了**过拟合**。
   - 另一名成员表示，由于公开数据集较小且存在潜在的投资者资本收益，各大实验室很可能都在利用**数据增强**来刷 **ARC AGI** 的分数。
- **将 HRM 与 LLM 集成**：一名成员询问是否可以构建一套系统，既使用 **HRM** 处理涉及未知事物的任务，又使用 **LLM/Transformer** 获取世界知识。
   - 一名成员回应称，目前尚无已知的良好集成机制，但建议可以尝试训练一个 **GNN** 来与小型推理模型进行交互。
- **关于模型系统的讨论**：一名成员建议，与其进行紧密集成，不如使用一个包含**世界模型**（World Model）的系统化模型方案。
   - 与此相关，有人发布了一个 [Discrete Distribution Networks](https://discrete-distribution-networks.github.io/) 的链接。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1425928445791047771)** (290 messages🔥🔥): 

> `Model choice recommendations, Context length impact on performance, Tool call failures, Uncensored Models` 


- **Qwen3 胜过 CodeLlama**：由于 **Qwen3** 的**速度和准确性**，成员们建议在编程任务中使用 [Qwen3](https://lmstudio.ai/models/qwen/qwen3-coder-30b)，其运行速度为 **20 tk/s**，而 CodeLlama 仅为 **5 tk/s**。
   - 然而，用户注意到即使是基础的编程任务也需要超过 **20B 参数**的模型，因此一位用户提到 [Llama 3.3 70b](https://lmstudio.ai/models/meta-llama/Llama-3-70B-Instruct-GGUF) 会更好。
- **上下文长度之轻不可承受**：用户探讨了上下文长度如何影响性能；即使上下文为空，**分配更高的上下文长度也会减慢生成时间**。
   - LM Studio 正在实施由内存估计器控制的内存分配限制（Guardrails），但你可以自行承担风险禁用它们。
- **工具调用异常触发终止**：当工具调用失败时，LM Studio **不应**中止生成，但目前它确实中止了，且未显示错误消息。
   - 一种可能的解决方案是禁用 MCPs（Multi-Call Procedures）以防止模型混淆，或者研究网页抓取工具，因为 Playwright 的 **Token 效率极低**。
- **高安全性模型导致故事创作受阻**：GPTOSS-20b 被描述为由于 OpenAI 的限制而导致的“安全性极高”的模型，因此它在创意叙事方面的许可度较低。
   - 建议以 GPTOSS 20b 作为对比基准，因为*现有的每一个模型保护性都比它低*，并建议尝试基于 Mistral 的模型，因为它们*开箱即用且非常自由*。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1425933122830340158)** (26 条消息🔥): 

> `Sparkle Arc Pro B60 Dual Server, Nvidia 4080 Ti Super vs Apple M3, GPU Benchmarks for LLM Inference, Server GPUs vs Multi-GPU Setups, Galax Single Slot RTX 5060 Ti GPU` 


- **Sparkle 服务器凭借 16 个 Arc GPU 大放异彩**：Sparkle 推出了 [Arc Pro B60 双路服务器](https://videocardz.com/newz/sparkle-unveils-arc-pro-b60-dual-server-with-16-gpus-and-up-to-768-gb-of)，拥有 **16 个 GPU** 和高达 **768 GB 的 VRAM**，由 **10800W PSU** 供电。
- **4080 Ti Super 与 M3 展开巅峰对决**：成员们讨论了拥有 **16GB VRAM** 和 **32GB RAM** 的 **Nvidia 4080 Ti Super** 在机器学习任务中是否优于拥有 **36GB RAM** 的 **Apple M3**。
   - 一位成员建议查看用于 LLM 推理的 [GPU benchmarks](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference) 以比较性能，并指出 **64GB 的 M3 Max 40 核 GPU** 是一个可比的配置。
- **服务器 GPU 对决多 GPU 平台**：一位成员对服务器 GPU 并不比多 3090 配置快多少表示惊讶，引用了仅 **10 t/s (60%-16vs25) 的微小差异**。
   - 其他人澄清说 Mac Studio GPU 并非专为 LLM 推理设计，服务器 GPU 优先考虑 **显存容量 (memory capacity)** 而非原始处理能力，而多 GPU 设置可能会因模型拆分而面临减速。
- **RTX 5060 Ti 传闻引发技术爱好者关注**：一位成员分享了关于 [Galax 单插槽 GeForce RTX 5060 Ti GPU](https://wccftech.com/galax-single-slot-geforce-rtx-5060-ti-gpu-16-gb-vram-blower-fan/) 的链接，该显卡配备 **16 GB VRAM** 和涡轮风扇。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1425921476288057478)** (270 条消息🔥🔥): 

> `API Keys in Models Page, AutoHotkey and Cursor Integration, Cursor Plan Pricing, GPT-5 Cost, Apply button in ASK mode` 


- **使用 API Keys 解锁模型访问**：一位用户询问在 **Models** 页面放入 **API keys** 的事宜，回复澄清说这样做允许在模型中使用个人 API keys。
   - 建议关闭所有模型并手动添加。
- **使用 AutoHotkey 控制编排 Cursor**：一位用户建议使用 **AutoHotkey (AHK)** 从 **Discord** 频道控制 **Cursor**。
   - 其他人似乎很热情，并准备尝试使用 Cursor 生成一些 **AHK scripts**。
- **讨论 Cursor 方案中 Auto 模型的价值与价格**：用户讨论了 **Cursor Pro** 方案中 **Auto** 模型成本增加以及使用限制降低的问题，一些人正在考虑替代模型或服务。
   - 一位用户表示：*真正的问题在于：[https://forum.cursor.com/t/the-pro-account-limits-have-been-clearly-reduced/134738](https://forum.cursor.com/t/the-pro-account-limits-have-been-clearly-reduced/134738)*。
- **GPT-5 编程既便宜又快捷**：一位成员宣称 **GPT-5** 表现出色且比其他选项更便宜。
   - 另一位成员表示，一个缺点是由于其推理过程，需要更多时间。
- **哀叹 ASK 模式中 Apply 按钮的消失**：用户报告说 **ASK** 模式中的 **APPLY** 按钮消失了，这使得将更改应用到文件变得更加困难。
   - 一些人表示他们不喜欢 **AGENT** 模式，因为它会失控 (goes rogue)，但现在他们被迫使用它。


  

---

### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1425930578573787146)** (6 messages): 

> `Agent 回复 Hello 失败、Linear 集成问题、使用 Background Agents` 


- **Agent 难以响应**：一名成员报告称，像 *"Respond with hello"* 这样的基础提示词有时会失败，并附上了一张 [图片](https://cdn.discordapp.com/attachments/1367213641027551352/1425934080939266048/image.png?ex=68eab599&is=68e96419&hm=21ad0141f990885a040fcda6e7beb3f542520958576fa5f3d5734ac945352dcb&) 展示 Agent 的失败情况。
   - 另一名成员表示，这对他们来说运行得相当稳定。
- **发现 Linear 集成 Bug**：一名成员在 **linear integration** 中遇到问题，并尝试重新连接 **GitHub** 和 **linear**，但问题仍然存在，如其 [截图](https://cdn.discordapp.com/attachments/1367213641027551352/1426297823498076191/Screenshot_20251010-125638.png?ex=68eab6dc&is=68e9655c&hm=10b31c0bc7e1118c9626e3f3868bd20e89d6aa1cc9574a670e37c5781bcd32c7&) 所示。
- **Background Agents 的工作流**：一名成员分享了他们使用 **Background Agents** 的工作流：
   - 该工作流为：
1) 创建一个新的 BA 来编写新功能代码。
2) 允许 BA 执行代码建议。
3) 与 BA 交互以审查代码、修复 Bug、消除幻觉（hallucinations）。
4) 将新的代码更改合并到 main 分支。
- **Background Agents 丢失上下文**：在代码合并到 main 分支后，**Background Agent** 会关闭，因此学习效果会随时间下降。这表明*工作重点应该放在为 BA 提供编写 Python 的上下文*，而不是编写特定的*功能*。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1425957118879465502)** (2 messages): 

> `CUDA kernels, Trainium 平台, 高层级书籍` 


- **Trainium 平台引发好奇**：一名成员将注意力从 **CUDA kernels** 转向了 **Trainium 平台**，并对活跃开发者的数量感到好奇。
   - 他们分享了自己的 [探索博客文章](https://numbersandcode.wordpress.com/2025/10/08/trainium-exploration/)，并观察到讨论极少，且缺乏专门的频道。
- **吐槽高层级技术书籍**：一名成员回忆起在某本高层级技术书籍出版时阅读过它，但并不喜欢。
   - 他们觉得该书缺乏深度，与*“从零开始逐行编写”*的方法形成鲜明对比。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1425932082995003463)** (12 messages🔥): 

> `ldmatrix tiling 计算、线性布局中的最优 swizzling、Triton 中的确定性 atomic_add 归约、Triton 社区会议` 


- **定位到 ldmatrix tiling 计算的对数逻辑！**：大家一致认为，在 `ldmatrix` tiling 计算中，**d 应该是 log_2(4/w)**，其中 *w* 是字节宽度。
   - 确认了*所有这些都在 Triton 的不同 lowerings 中实现*，如有疑问可以查看 [源代码](https://github.com/triton-lang/triton/blob/b5fea1e3f4c2cb0b40c0ce98261b240d8728d2f9/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.cpp#L186-L199)。
- **揭秘最优 Swizzling 秘籍**：讨论了线性布局中最佳 swizzling 的工作原理，指出了论文中 Figure 5、Section 5.4 和 9.2 之间的分歧。
   - 理解该算法的最佳参考是其在 [pull 6982 中的 GenericSwizzling.cpp](https://github.com/triton-lang/triton/pull/6982) 的首次实现。
- **用于确定性 Atomic Adds 的 Turnstile 归约**：一名成员询问如何在 Triton 中实现确定性的 atomic_add 归约，并提到 CUDA 有类似 *"turnstile reduction"* 的机制，即 CTA 通过屏障等待前 K 个 block 先完成。
   - 在此消息记录中未提供具体的 Triton 解决方案。
- **Triton 社区会议定于明年举行**：下一次 Triton 社区会议将于 **2025 年 11 月 5 日上午 10 点至 11 点（PST）** 举行，这是 [会议链接](https://tinyurl.com/2s3z953y)。
   - 暂定议程：*TLX 更新*、*Triton + PyTorch Symmetric Memory* 以及 *PyTorch 中的 Triton Flex Attention*。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1425961852675817524)** (24 messages🔥): 

> `Thread block execution order, Blackwell CLC, CUB library, cuda::ptx::mbarrier_try_wait, Model cudagraph capture` 


- **线程块顺序是未经证实的福音？**：一名成员询问了线程块执行顺序的保证，特别是当线程块数量超过系统可运行数量时，**(0,0)** 是否总是先于 **(10,10)** 运行。
   - 虽然一位成员记得有文档支持“最低线性 ID 优先”的方法，但另一位成员表示这*并非官方保证/记录/支持*，将其归类为**未定义行为 (UB)**，尽管 **CUB library** 可能考虑到了这一点。
- **CUB 隐藏了线程块顺序的复杂性**：一名成员分享了一个 [YouTube 视频](https://youtu.be/VLdm3bV4bKo?si=o4vi1dOK3sc7U-kH)，展示了 **CUB** 中处理线程块排序的一种抽象，并且 [TilePrefixCallbackOp](https://nvidia.github.io/cccl/cub/api/structcub_1_1TilePrefixCallbackOp.html#_CPPv4NK3cub20TilePrefixCallbackOp10GetTileIdxEv) 可以用于实际 Scan 之外的操作，但仅限于 **1D index**。
- **尝试 MBarrier，谨慎行事**：一名成员询问 `cuda::ptx::mbarrier_try_wait` 是否为非阻塞，另一名成员澄清说 `test_wait` 是非阻塞的，而 `try_wait` 是*潜在阻塞的*，在获取结果后需要检查 `waitComplete`，并引用了 [NVIDIA 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait)。
- **CUDA Cores：多个对应一个 Warp？**：一名成员询问一个 Warp 是分配给单个 **CUDA core**，还是需要多个 **CUDA cores** 来执行一个 Warp 的所有 32 个线程。
   - 另一名成员回答说，Warp 调度器会跟踪多个 Warp，并每个时钟周期发出一条指令，每条指令可以在任意数量的核心（例如 16 个）上执行，这意味着通常涉及多个核心。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1425965909725675661)** (4 messages): 

> `parrot lib, InferenceMAX, ATLAS, Billy Dally hardware` 


- **Parrot 凭借基于 GPU 数组的库展翅高飞！**：Connor Hoekstra 的 **parrot** 是一个旨在运行在 GPU 上的 C++ 数组库，分享于[这条推文](https://x.com/blelbach/status/1976255534467571730)。
- **InferenceMAX 助力开源推理！**：[InferenceMAX](https://inferencemax.semianalysis.com/) 是一个专注于推理的开源项目，详见[此简报](https://newsletter.semianalysis.com/p/inferencemax-open-source-inference)，代码托管在 [GitHub](https://github.com/InferenceMAX/InferenceMAX)。
- **Billy Dally 宣传面向 AI Agent 的新硬件！**：一段 [YouTube 视频](https://www.youtube.com/watch?v=thsm6tff8h0) 展示了 Billy Dally 讨论专为 **AI Agent** 设计的硬件。
- **ATLAS 通过运行时学习加速器引领新的 LLM 推理！**：Together AI 推出了 **AdapTive-LeArning Speculator System (ATLAS)**，这是一种通过运行时学习加速器进行 **LLM Inference** 的新范式，详见[此博客文章](https://www.together.ai/blog/adaptive-learning-speculator-system-atlas)和 [Tri Dao 的推文](https://x.com/tri_dao/status/1976692444977938499)。


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1426312418694402172)** (1 messages): 

> `GPU Performance Engineer Hiring, NVIDIA GPU Architecture, Kernel Optimization, Software-Hardware Co-design` 


- **GPU 性能工程师职位开放！**：一个 **GPU Performance Engineer** 职位正在招聘，要求对 **NVIDIA GPU architecture**（**Blackwell**、**Hopper**）有深刻理解，并具备优化 **Kernel** 的经验。
   - 该职位提供 **25 万美元 + 股权**，欢迎各种经验水平的候选人；发送你的 GitHub / CV 即可申请。
- **性能岗位寻求 NVIDIA GPU 专家**：理想的候选人应对应 **NVIDIA GPU architecture**，特别是 **Blackwell** 和 **Hopper** 有很强的掌握，并具备 Kernel 优化技能。
   - 拥有 **CuTe**、**CUTLASS**、性能分析器（profilers）、Linux Kernel、驱动程序内部机制以及软硬件协同设计经验者优先。

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1425979778690715819)** (6 条消息): 

> `Distributed Training Libraries: TorchTitan vs NVIDIA-Nemo, CUDA Kernel Debugging in Visual Studio, 4D Parallelism, Megatron Core, TorchTitan's Adaptability` 


- **TorchTitan vs NVIDIA-Nemo 对决**：一位新手询问在拥有 **256 H200** 的训练任务中，该如何选择 **TorchTitan** 或 **NVIDIA-Nemo**。
   - 一位成员建议，除非需要进行大量的底层修改（hacking），否则 NVIDIA 的库更好，因为 **TorchTitan** 虽然适应性更强，但 NVIDIA-Nemo 的性能更优。
- **Visual Studio 中的 CUDA Kernel 调试技巧公开**：一位新用户分享了在 **Visual Studio** 中调试 **CUDA kernels** 的方法，并附带了一张展示调试界面的截图。
   - 其他人提到，许多人使用其他调试器，如 **burn**、**rig** 和 **ndarray**。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1426255066486276308)** (2 条消息): 

> `GPU Programming with CUDA, Resources for learning CUDA, CUDA Projects for Students` 


- **学生寻求使用 CUDA 进行 GPU 编程的指导**：一位学生就如何开始使用 **CUDA** 进行 **GPU 编程**寻求建议，包括项目和主题建议。
   - 另一位成员引导他们查看频道 [<#1198358627594023014>](https://discord.com/channels/1169494361482651678/1198358627594023014) 和 [<#1191300313928433664>](https://discord.com/channels/1169494361482651678/1191300313928433664) 中的资源。
- **CUDA 资源**：可以在相关的 Discord 频道中找到 **CUDA** 资源。
   - 建议有兴趣学习的人查看这些频道。


  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1426026871576985681)** (3 条消息): 

> `Triton Projects, Picograd` 


- **启动 Triton 项目头脑风暴**：一位用户建议寻找一个需要使用 **Triton** 的项目或想法并将其实现。
   - 这一提议旨在激发社区内 **Triton** 的创新和实际应用。
- **在 Picograd 中征集 Triton Kernels**：另一位用户建议在相关频道中为 **picograd** 项目添加 **Triton kernels**。
   - 一位用户回复道：*"picograd is awesome :)"*，表现出对将 **Triton** 与 **picograd** 集成的热情。


  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1426021261204656238)** (6 条消息): 

> `Panther Lake, Xe3 Details, Celestial Architecture Speculation, Memory Bandwidth Bottleneck` 


- **Panther Lake 的 Xe3 计算能力提升**：新的 [Panther Lake 幻灯片](https://www.techpowerup.com/review/intel-panther-lake-technical-deep-dive/13.html) 揭示了每个 slice 拥有更多的计算资源，高达 **16 MiB 的 L2$**，以及具有可变寄存器分配的**另外两个线程**。
   - 据称，这些变化相对于 Arrow Lake 实现了 **40% 的能效比（performance-per-watt）提升**，解决了内存带宽限制问题。
- **内存带宽瓶颈得到解决**：由于全局内存带宽减少，**Battlemage** 上的内存受限任务在 **Lunar Lake** 上表现不佳，但 Panther Lake 中额外的 **L1$/SLM 和 L2$** 应该能缓解这一问题。
   - Lunar Lake 的带宽相对于可用计算能力降低了 *2.7 倍*，但增加的缓存可能会弥补这一点。
- **Celestial 架构推测**：**Celestial** 的架构仍不明确，这取决于相对于 **Panther Lake** 所需的计算单元与固定功能单元的比例。
   - 最好的假设是 **Celestial** 将使用 **six-subslice slices**，但根据架构的不同，他们可能会增加或减少这一数量。
- **Xe2 Kernel 兼容性**：由于 Xe2 在架构上接近 Xe3，在 Xe2 GPU 上开发的 kernels *应该能很好地迁移*到 **Xe3**。
   - 据一位成员称，我们*可能在一段时间内都不会得到关于 Celestial 的任何消息*。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/)** (1 条消息): 

vipul_todo_18: 其他人的推广：

https://x.com/jyo_pari/status/1976324891545829876
  

---

### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1426071303974162514)** (1 条消息): 

> `ThunderKittens compilation issues, Nvidia GH200, CUDA 12.3, fp8e8m0 undefined type` 


- **ThunderKittens 在 CUDA 12.3 下报错**：一名成员尝试在 **Nvidia GH200** 机器 (**arm64**) 上使用 **CUDA 12.3** 编译 **ThunderKittens**，并遇到了类似 `__nv_fp8_e8m0` 的未定义类型。
- **CUDA 12.3 中的 fp8 问题依然存在**：在使用 `python setup.py install` 和提供的 Makefiles 时，都会出现 `__nv_fp8_e8m0`、`__nv_fp8x2_e8m0` 和 `__nv_fp8x4_e8m0` 等未定义类型。
   - 目前尚不清楚是需要更高版本的 **CUDA**，还是存在其他阻碍编译的问题。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1426001983659180134)** (26 条消息🔥): 

> `amd-gemm-rs Leaderboard Updates, amd-all2all Leaderboard Submissions, MI300x8 Performance, amd-ag-gemm Submissions` 


- **MI300x8 上的 Gemm 性能提升**：一名成员在 **MI300x8** 上以 **516 µs** 的成绩获得了 `amd-gemm-rs` 排行榜的**第 2 名**。
   - 其他显著成绩包括 **530 µs**（第 4 名）和 **536 µs**（第 9 名），展示了极具竞争力的性能调优。
- **All2All 竞技场：MI300x8 表现出色**：`amd-all2all` 排行榜记录了多个使用 **MI300x8** 的成功提交，耗时从 **443 µs** 到 **1767 µs** 不等。
   - 一名用户以 **566 µs** 的成绩获得**第 7 名**，突显了在优化 all-to-all 通信方面的持续努力。
- **AG-GEMM 在 MI300x8 上实现加速**：`amd-ag-gemm` 排行榜记录了在 **MI300x8** 上的成功提交，耗时稳定在 **470-520 µs** 左右。


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1426341075173380299)** (1 条消息): 

> `Runner Timeouts` 


- **Runner 超时困扰**：目前 **runners** 存在问题，导致了意外的**超时**。
   - 团队正在积极调查，并承诺很快会发布更新。
- **Runner 超时调查**：一个团队正在调查意外发生的 **runner** 超时问题。
   - 将尽快提供有关这些问题的原因和解决方案的更新。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1426225361980489820)** (4 条消息): 

> `COVID, Factorio Crime Scene` 


- **确认缺席：成员因 COVID 无法参加**：一名成员宣布感染了 **COVID**，将缺席今天的会议，并请求获取任何重要结论的更新。
   - 另一名成员回复道：“早日康复！”
- **Factorio 惨剧：意外的游戏内事件**：一名成员在参加会议时让 **Factorio** 保持运行，回来后发现游戏里变成了“犯罪现场”。
   - 附带了一张图片（[截图](https://cdn.discordapp.com/attachments/1354169122107293786/1426296785088942151/Screenshot_2025-10-09_at_2.58.39_PM.png?ex=68eab5e4&is=68e96464&hm=7bc9686f5888df3fa014a43d97a1420560317bf36528519d13716e475ea90b69&)），并评论道“这倒是头一回见”，暗示发生了一个前所未有的游戏内事件。


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1426232820342194320)** (16 条消息🔥): 

> `AMD Cluster Timeout Issues, Memory Access Fault Debugging, Jot Runner Timeout Extension` 


- **AMD 集群出现超时问题**：用户报告了 **AMD 集群**上的超时错误，由于 `queue_time + running_time` 有 **5 分钟限制**，导致任务超时。
   - 一名维护者承认问题出在 **AMD 侧**，需要他们的干预，并指出解决可能需要一段时间。
- **Segfaults 预示内存访问错误**：一名用户在 **GPU node-7** 上遇到了内存访问故障，这表明代码中存在非法内存访问，通常表现为 **segfault**。
   - 一名维护者解释说，由于权限不同或运气成分，本地设置可能会掩盖这些问题，建议使用 print 语句进行逐步调试。
- **Jot Runner 的超时延长请求**：一名用户请求 Jot runner 在计算超时时仅考虑 `running_time`，排除 `queue_time`，以防止任务过早终止。
   - 一名维护者回应称这很难实现，但提出可以临时延长超时窗口，并请求在出现此类问题时及时通知。


  

---

### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1425954776444702870)** (28 条消息🔥): 

> `MoE 的 Grouped GEMM 性能，K-contig 和 swizzling 错误的 PTX 文档，CUTLASS 中的 Torch tensor 支持，ldmatrix copy atoms 导致的编译器中止，调试流水线停顿 (pipeline stalls)` 


- ****MoE 的 M-Occupancy 困境****：正在评估推理过程中 **MoE** 的 Grouped GEMM 性能，特别是关于 **M-occupancy** 的问题。由于每个专家（expert）的 token 分布是随机的，其 M-occupancy 可能显著低于普通的 GEMM，并且有人提出了关于其对传统 roofline 模型影响的问题。
   - 在 `gpt-oss 20b` 的 prefill 阶段，由于无效计算，M-occupancy 已低至约 60%。
- ****PTX 视差？精度问题****：关于 K-contiguous 布局和 swizzling 的 **PTX 文档** 准确性存在问题，特别是涉及 tensor descriptors，如 [这段 Triton 代码](https://github.com/triton-lang/triton/blob/b5fea1e3f4c2cb0b40c0ce98261b240d8728d2f9/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAHelpers.h#L250-L253) 和 [这份 NVIDIA 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-warpgroup-k-32b-swizzle-tf32) 所示。
   - PTX 团队已接手该问题，并应在 **13.1** 版本中提供修复。
- ****Torch Tensor 的诱人传输****：目前可以直接传递 **torch tensors**，尽管它假设是全动态布局，并且仍然通过 **DLPack 路径**。
   - 绕过 **DLPack** 的原生支持已在路线图中，更多详情请参见 [此处](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/framework_integration.html#implicit-conversion)。
- ****编译器在 ldmatrix 上报错 'Abort!'****：一位用户报告在利用 **ldmatrix** copy atoms 时遇到编译器中止（aborts），并为该问题创建了一个名为 `t.py` 的代码复现。
   - 他们还提到，许多用户错误（user errors）表现为内部编译器错误（**ICEs**）。
- ****调试困境：流水线停顿排查****：一位用户询问了调试流水线停顿（pipeline stalls）的有效方法，寻求比粗粒度的 **nsight compute** 更精确的工具。
   - 建议包括使用 **nsight compute** 查看 warp 状态统计信息，以及使用 **gluon** 配合 **proton** 查看 kernel 执行的时间线视图。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1425970712660934767)** (1 条消息): 

> `Discord 角色，竞赛获胜者，AMD 竞赛` 


- **Discord 服务器添加竞赛获胜者角色**：Discord 服务器为竞赛获胜者引入了新角色：<@&1418285356490428476> 和 <@&1425969596296462356>。
   - 服务器还计划将这些角色扩展到正在进行的 **AMD 竞赛** 的获胜者。
- **AMD 竞赛获胜者将获得 Discord 角色**：当前 **AMD 竞赛** 的获胜者也将获得 Discord 服务器上的特殊角色，类似于之前的竞赛获胜者。
   - 这一举措旨在表彰和突出社区成员的成就。


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1425930522147815545)** (5 条消息): 

> `5070 ti super 24gb vs 5080，分布式训练库：torchtitan vs nvidia-nemo，Tesla P40 24GB 性能` 


- **5070 Ti Super 24GB 规格推测**：一位成员询问了关于潜在的拥有 **24GB** VRAM 的 **5070 Ti Super** 与 **5080** 的看法，表达了对新 GPU 选项的期待。
- **TorchTitan 与 NVIDIA-NeMo 的训练对决**：一位新人询问在 **256 台 H200** 上进行分布式训练时，如何在 **TorchTitan** 和 **NVIDIA-NeMo** 之间做出选择，并对 TorchTitan 的成熟度以及相比 NVIDIA-NeMo 的 Megatron-core 可能存在的低效表示担忧。
   - 他们强调了 **NVIDIA-NeMo** 在超大规模 4D 并行性方面经受过验证的效率，同时承认了 **TorchTitan** 的易用性，但担心其在 **2 万亿 token** 上训练 **7B 稠密模型** 时的计算效率。
- **Tesla P40 在 30B 模型上的表现令人惊喜**：尽管成本低廉，**Tesla P40** 的性能仍引发了讨论，一位成员指出其表现平平。
   - 另一位成员分享了一个 [Reddit 基准测试](https://tokens-per-second-visualizer.tiiny.site/)，显示 **P40** 在 **30b 模型** 上达到了 **8 tps**，他们认为这出奇地好。


  

---

### **GPU MODE ▷ #[irl-accel-hackathon](https://discord.com/channels/1189498204333543425/1416087968933740688/1425984523409297478)** (2 条消息): 

> `Project Teams, GPU Inference, Novel GPU Work` 


- **用户表达加入项目团队的兴奋之情**：一位用户表达了加入项目团队的兴奋，特别是那些与 **inference 项目**相关的团队，这可能有助于他们的申请获得批准。
   - 该用户希望现在加入还不算太晚，并对贡献代码充满热情。
- **新项目想法获得积极反馈**：另一位用户对一个潜在项目表示热烈欢迎，称其为一个*伟大的想法*。
   - 他们建议让该项目在 **GPUs** 上运行将是一项 **novel work**，充满有趣的挑战和机遇，并请另一位用户审阅相关论文。


  

---


### **GPU MODE ▷ #[llmq](https://discord.com/channels/1189498204333543425/1421956177549332662/1426178066647879771)** (7 条消息): 

> `Attention-Residual Recalculation, Contributor Guide, Weird Quantizations` 


- **Attention-Residual Recalculation 提升性能**：在 [llmq](https://github.com/IST-DASLab/llmq/pull/7) 中实现了在 backward pass 期间重新计算 **attention-residual**，这在内存压力较大的情况下（如在 **4x4090** 上运行 **14B 模型**）可以带来显著的净增速。
   - 使用此优化后，在 **fp8** 模式下训练 **Qwen2.5-14B** 的吞吐量从 **3.2k 提升至 6.0k TPS**，在 **bf16** 模式下从 **2.5k 提升至 4.5k TPS**。
- **贡献者指南疑问**：一名成员询问是否有贡献者指南以便快速上手（onboarding）、设定贡献预期，并澄清基于贡献的研究署名问题。
   - 具体而言，他们想知道仅仅是 **GitHub contribution** 与如果贡献重大是否可能在研究论文中获得署名之间的区别。
- **奇特的量化（Wacky Quantizations）**：一名成员询问是否支持在更“奇特”的量化格式下进行训练，以及在命令行或配置文件中如何配置。
   - 主要开发者澄清说，即使是奇特的量化格式，量化后的 optimizer state 应该也相对简单，但对于 matmuls，你首先需要一个针对该量化格式的实际 matmul 实现。


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1426288309868433409)** (2 条消息): 

> `FLA Benchmark, GDN, Mamba2, PTC Talk` 


- **FLA 用例作为有说服力的 Benchmark**：一名成员表示支持将 **FLA** 用例作为一个良好且有说服力的 benchmark。
   - 他特别提到了 **GDN** 和潜在的 **Mamba2** 作为相关模型。
- **期待 PTC 演讲**：一名成员表达了对即将到来的 **PTC** 演讲的期待。
   - 未提供关于主题或发言人的进一步细节。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1425920757413974036)** (108 条消息🔥🔥): 

> `Colab Cost, Continual Learning, GPU Time on A100, Fine-tuning Llama3, HF Fellowship` 


- **Colab 并不贵！**：成员们讨论了 **Colab** 每月 **$10** 的可负担性，其中一人指出它提供约 **13.2 小时** 的 **A100 GPU** 时间，成本约为 **75 美分/GPU/小时**。
   - 一位自称“银行账户里不到一瑞士法郎”的用户被鼓励向父母寻求零花钱。
- **持续学习（Continual Learning）评述**：一名成员解释了持续学习的挑战，认为目前的论文侧重于“治标不治本”，并建议为模型设计**原则**，使其根据奖励信号学习该记住什么以及记住多少。
   - 他们建议 *optimization function* 应该是一个可学习的 **RNN**，致力于增加每项操作的奖励，这与 **Richard Sutton** 关于 AGI 中涌现模仿学习的想法一致。
- **微调（Fine-Tuning）的挫折**：一名成员寻求建议，如何缩短在 **40GB GPU** 上对 **200,000** 条数据集（token 长度为 **2048**）进行 **Llama3 8B** 模型微调的训练时间，估计耗时将超过 **5 天**。
   - 另一位用户表示他们也需要在某个时候下载 Llama 并让 control net 运行起来。
- **HF Fellowship 正在进行**：一名成员提到 *prithiv* 现在成为了 **Hugging Face Fellow**，并指出[他的 Spaces 和模型](https://huggingface.co/prithivMLmods)非常实用。
   - 另一位用户询问“人们如何成为 Hugging Face Fellow？”，并建议“似乎必须对 ML 领域有所贡献”。
- **MoE 模型势头强劲**：一名成员正在寻找一个优秀的开源 **MoE** (Mixture of Experts) 模型，要求其总参数可配置以进行预训练。
   - 另一名成员回应道：“我迫不及待想预训练一个 24M 参数的 MoE 了”。


  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1425982556997681264)** (1 messages): 

> `Hyperparameter Diffusion, Faster Image Generation, Compute Reduction` 


- **超参数扩散 (Hyperparameter Diffusion) 带来突破**：一名成员发布了一个 HuggingFace Space，展示了论文 [Hyperparameters are all you need](https://arxiv.org/abs/2510.02390) 的实现，证明了在图像生成中，**8 步** 即可达到与 **20 步** 相当的 FID 性能。
   - 该实现包括 [Counterfeit v3.0 版本](https://huggingface.co/spaces/coralLight/Hyperparameters_Are_All_You_Need)、[XL 版本](https://huggingface.co/spaces/coralLight/Hyperparameters-are-all-you-need-xl-version) 以及用于测试的 [原始 SD 版本](https://huggingface.co/spaces/coralLight/hyperparameters-are-all-you-need-sd-version)。
- **Diffusion 速度提升 2.5 倍且效果更好**：与 DPM++2m 相比，新方法实现了 **2.5 倍更快** 的图像生成速度，且 **质量更好**，并适用于任何模型。
   - 该成员指出，无需训练或蒸馏 (distillation)，可实现 **60% 的计算量减少**。
- **超参数减少了 Diffusion 的计算量**：结果表明，**8 步** 生成的图像 FID 性能与 **20 步** 相当，实现了 **60% 的计算量减少**。
   - 一名成员还分享了一些使用该方法生成的示例图像，并征求关于接下来测试哪些模型的建议。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1426172522218655805)** (2 messages): 

> `BERT model for Polish tweets, MedScan AI medical tool, NLP in healthcare` 


- **BERT 模型会说波兰语了！**：一名成员微调 (fine-tuned) 了一个 **BERT 模型**，用于预测 **波兰语推特** 中的情感和情绪，可以在 [这里](https://huggingface.co/spaces/yazoniak/twitteremo-pl-classifier) 找到。
- **MedScan 作为友好的 AI 医生上线！**：一名成员发布了 **MedScan**，这是一个基于 **Hugging Face 模型** 构建的 AI 工具，用于智能医疗搜索和报告分析，访问地址为 [这里](https://ragmedscan.vercel.app/)。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1426043548234547303)** (1 messages): 

> `Custom model vs finetune, Text encoder model` 


- **请求澄清模型类型**：一名成员询问该模型是自定义模型还是微调 (finetune) 模型。
- **询问文本编码器 (Text encoder) 模型**：一名成员询问正在使用哪种文本编码器 (Text encoder) 模型。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/)** (1 messages): 

cakiki: <@892799950470144060> 请勿跨频道发帖 (no cross-posting)
  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1425927260216496178)** (101 条消息🔥🔥): 

> `Datacurve 融资, Spellbook 融资, OpenAI AMA 取消, Kernel 融资与 Agent 身份验证, Elastic 收购 Jina AI` 


- **Datacurve 获 1750 万美元融资，用于前沿 LLM 训练**：根据 [Serena Ge 的公告](https://xcancel.com/serenaa_ge/status/1976328983458480539)，Datacurve 宣布完成了 **1500 万美元 A 轮**和 **270 万美元种子轮**合并融资，旨在为基础模型（尤其是编程领域）提供高质量的训练数据集。
   - 本轮融资由 Chemistry 领投，YC、Cohere、Afore 及多位天使投资人参投。
- **Spellbook 完成 5000 万美元 B 轮融资，助力 AI 合同起草**：Spellbook 将自己定位为“合同领域的 Cursor”，完成了由 Khosla Ventures 领投的 **5000 万美元 B 轮融资**，正如 [Scott Stevenson](https://xcancel.com/scottastevenson/status/1976280608436572393) 所宣布的那样。
   - 该 AI 起草与审查平台自 2022 年以来已增长至 **4,000 名客户**，建议采纳率从 **5% 跃升至 60%**。新资金将推动产品增强，首先从实时市场对比测试版开始。
- **Kernel 融资 2200 万美元以优化 LLM 云端体验**：由 CEO Catherine Jue 领导的 Kernel 宣布完成了由 Accel 领投的 **2200 万美元种子轮 + A 轮融资**，用于大规模自动化浏览器工作负载并推出 Kernel Agent Authentication，详情见[此公告](https://xcancel.com/juecd__/status/1976325764166615498?s=46)。
   - 客户包括 CashApp 和 Rye。Kernel Agent Authentication 提供了一个身份层，使 AI 应用能够对用户操作进行安全、有范围限制且完全可审计的控制。
- **Elastic 收购 Jina AI 以增强多模态搜索**：[Elastic 宣布](https://xcancel.com/elastic/status/1976278980018765886)已收购 Jina AI，以加强其检索、Embeddings 和上下文工程能力，从而为 Agentic AI 提供动力。
   - 社区对这一举动表示了压倒性的赞赏，强调了其在巩固 Elastic 在企业搜索和 AI Agent 领域主导地位方面的潜力。
- **通过 GLM 破解版仅需 3 美元即可享受无限量 Claude？**：有用户声称，中国的逆向工程通过将请求路由到 z.ai 上的 **GLM-4.6** 而非正版 Sonnet，实现了仅需 **$3/月** 的“无限量 Claude 编程”层级，详情见[此声明](https://xcancel.com/shydev69/status/1976641232622453045)。
   - 其他人对延迟和实际的 Claude 质量表示怀疑，一位用户承认这不适合自己的工作，因为 GLM 没有提供可控的 Thinking Mode。然而，关于该破解方法的[博客文章](https://shydev.medium.com/get-unlimited-claude-code-for-3-53d61d5b2b2f)正在引起关注。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1426227622077272146)** (8 条消息🔥): 

> `Gemini Flash 2.5 Nano Banana, JSON 提示词, nano-banana AI 输出, 复制 nano-banana AI 输出, 移除 nano-banana AI 输出` 


- **Gemini Flash 2.5 Nano Banana：提示词展示**：Emily 分享了一个用于 **Gemini Flash 2.5 Nano Banana** 的详细 **JSON 提示词**，该提示词可以生成一张带有蓝色调、动漫风格的东亚女性在电脑角卧室里的镜面自拍，展示在[这条推文](https://x.com/iamemily2050/status/1976431328280416520?s=46)中。
- **JSON 与自然语言提示词：引发辩论**：追随者们就 **JSON** 与**自然语言提示词**展开了辩论，而 Emily 则主张 JSON 在可复现、受控的工作流中具有优势。
   - 用户在[这条推文](https://x.com/toyxyz3/status/1976650667046605263?s=46)中发布了他们自己的复现结果。
- **Nano-Banana AI 输出：“灵魂印记”辩论**：用户讨论了在多个 **nano-banana AI 输出**中看到的微弱、重复的伪影，推测这是故意的水印、Transformer 伪影，还是仅仅是生成过程中的怪癖。
- **分享伪影移除技巧**：分享了关于复现（先灰度处理再过饱和）、移除（Upscaling）以及不存在任何追踪 ID 的技巧，其中夹杂着笑话（“这不是水印，这是灵魂”）和技术建议。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1426236885818150952)** (15 条消息🔥): 

> `llama.cpp 中的 ROPE 实现, RoPE 计算中的精度敏感性, 神经定理证明频道` 


- **llama.cpp 请求 ROPE 救援**：有人请求协助解决 *llama.cpp* 一个 merge request 中实现 **hybrid attention ROPE** 时遇到的障碍，具体涉及 [这个 GitHub pull request](https://github.com/ggml-org/llama.cpp/pull/16095#issuecomment-3390312481)。
- **HF 实现缺少 Partial ROPE？**：一名成员注意到 **Hugging Face 实现** 似乎没有使用 *partial ROPE*，并质疑其是否仅存在于 config 中，参考了 [这段 HF 代码](https://github.com/huggingface/transformers/blob/0419ff881d7bb503f4fc0f0a7a5aac3d012c9b91/src/transformers/models/qwen3_next/modeling_qwen3_next.py#L176)。
- **ROPE 精度可能导致不准确**：有建议认为 **RoPE 计算** 对精度非常敏感，正弦和余弦矩阵应以 **fp32** 计算以避免不准确，并引用了 [EleutherAI 的 gpt-neox](https://github.com/EleutherAI/gpt-neox/blob/d12c771198388980ee054617e537665f044e0584/megatron/model/positional_embeddings.py#L63) 作为例子。
- **BF16 长上下文精度困扰**：一名成员提到最近的一篇论文指出，在长上下文中使用 **BF16** 的 **ROPE** 可能会导致精度问题。
- **寻找神经定理证明频道**：一名成员询问是否有专门讨论 **神经定理证明 (neural theorem proving)** 问题的频道，并想知道是否应该在 <#747850033994662000> 频道发帖。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1425932393251864576)** (71 条消息🔥🔥): 

> `OOD 泛化与训练损失, Vision Language Model 优化, Scaling Laws 中的 Cosine Decay 与 Infinite LR 对比, Warmup Stable No-Decay 训练, Scalar RMSProp 适应性` 


- **OOD 泛化悖论**：讨论了 **训练损失** 与 **分布外 (OOD) 泛化** 之间一个违反直觉的联系，质疑为什么 **nGPT 架构** 可能无法泛化。
   - 一个建议涉及 **长度泛化** 或对曝光偏差 (exposure bias) 的敏感性增加，特别是考虑到 *“已见”样本* 之间的泛化。
- **Vision Language Model 分辨率优化**：一名成员分享了一个 **基准测试项目** 的初步结果，该项目专注于在 Vision Language Models (VLMs) 中优化 **图像分辨率** 与 **输出质量** 的平衡，在 **COCO 2017 Val 数据集** 上使用 **Gemini 2.0 Flash** 进行字幕生成任务，并附带了 [报告](https://cdn.discordapp.com/attachments/747850033994662000/1425953022671847515/minres_report.pdf?ex=68eac73d&is=68e975bd&hm=194b9521fd489b4596bc9cdb078a3f90adc9533c1be618a2aaa1d0174f450c82&)。
- **Scaling Law 测试套件的优化器对比**：成员们讨论了在 Scaling Laws 研究中对比 **cosine decay**、**带退火的 cosine decay** 以及 **带退火的 infinite LR**，质疑带退火的 infinite LR 是否能通过部分训练的 checkpoints 实现更廉价的 Scaling Law 测试。
   - 一名成员链接了一篇关于使用 **warmup stable no-decay** 和 **checkpoint averaging** 的论文 ([https://arxiv.org/abs/2507.17634](https://arxiv.org/abs/2507.17634))，并指出其在万亿参数模型中的应用。
- **关于 Scalar RMSProp 的辩论**：一名成员反驳了关于 **Scalar RMSProp** 的一项主张，认为其适应性并不取决于最大稳定步长；由于锐度 (sharpness) 会适应优化器，每个优化器都会在步长结束时达到这个极限。
   - 反驳观点包括围绕参数空间区域和锐度正则化的讨论。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1426205203807600770)** (4 条消息): 

> `机理可解释性, 归因图, 线路追踪, 模型生物学` 


- **LessWrong 文章指导有志于可解释性研究的人员**：一名成员分享了一篇 [LessWrong 文章](https://www.lesswrong.com/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher)，介绍如何成为一名 **机理可解释性 (mechanistic interpretability) 研究员**。
- **Anthropic 的线路追踪解析**：一名成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=ruLcDtr_cGo)，对 **Anthropic 的 Circuit Tracing 和 Model Biology 论文** 中的 **归因图 (attribution graphs)** 进行了很好的介绍。


  

---

### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1425920864968507402)** (1 messages): 

> `Moxin-VLM, VLM-R1` 


- **新的 VLM 到来：Moxin & R1**：一名成员分享了两个新的 Vision Language Model (VLM) GitHub 仓库链接：[Moxin-VLM](https://github.com/moxin-org/Moxin-VLM.git) 和 [VLM-R1](https://github.com/om-ai-lab/VLM-R1.git)。
- **VLM 领域进一步扩大**：随着 [Moxin-VLM](https://github.com/moxin-org/Moxin-VLM.git) 和 [VLM-R1](https://github.com/om-ai-lab/VLM-R1.git) 的加入，开源 Vision Language Model (VLM) 的版图不断扩张，为研究和应用提供了新途径。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1425929921154388101)** (46 messages🔥): 

> `Atropos tutorial video, Training Qwen3-Next, Predicted Outputs in vllm, r/LocalLLaMA post removal` 


- **Atropos 教程视频在网上出现**：一名成员询问是否有关于使用 **Atropos** 的 YouTube/Xwitter 快速教程视频，随后在 [Twitter](https://fxtwitter.com/NousResearch/status/1925381160097697803) 和 [YouTube](https://www.youtube.com/watch?v=in__ELD4NxE) 上找到了关于 Atropos 环境运作方式的更广泛概述。
- **训练 Qwen3-Next 模型的挑战**：成员们讨论了与训练 **Qwen3-Next** 相关的问题，包括挂起和 checkpoint 分片加载缓慢，其中一人报告加载阶段长达 **8 分钟**。
   - 另一名成员报告了 **14% MFU**，且未遇到任何问题。
- **vllm 中的 Predicted Outputs**：新架构 **Predicted Outputs in vllm** 正在发布，它通过将输出转换为预测（部分）匹配的 prefill，从而实现极快的生成速度。
   - 更多信息请查看 [博客文章](https://cascadetech.ai/blog/vllm-predicted-outputs/)、[演示](https://app.cascadetech.ai/) 和 [推文串](https://x.com/saganite/status/1976707696578691101)。
- **r/LocalLLaMA 帖子被删引发质疑**：一名成员询问为什么他们关于 *vllm 中快速预测输出* 的 r/LocalLLaMA 帖子被从 [这个 Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1o39l0l/fast_predicted_outputs_in_vllm/) 中删除。
   - 有人建议该子版块的一些版主也在 Discord 频道中，但不确定具体是谁。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1426225124994056293)** (4 messages): 

> `FP8 LLM Finetuning, QLoRA effectiveness, Test Time RL, LoRA Precision` 


- **FP8 微调：值得一试吗？**：一名成员询问了关于尝试微调 **FP8 LLM** 以及 **gradient descent** 表现如何的问题。
   - 另一名成员回答说，由于在易用的训练框架中支持不完善且依赖 Torch compile，目前用于全精度微调的 **FP8** 训练看起来并不值得。
- **QLoRA：精度提升！**：一名成员询问了 **QLoRA** 微调的有效性。
   - 另一名成员表示，**LoRA** 并不严重依赖模型的精度来维持训练稳定性，因为 LoRA 参数是在 **BF16/FP32** 中训练的，**FP8 LoRA** 应该能保持与 BF16 相当的完整质量，而 **QLoRA** 的质量仅有轻微下降。
- **Test Time RL 热议**：一名成员提到了关于 "Test Time RL" 的背景信息：[https://fixupx.com/rryssf_/status/1976269613072843063](https://fixupx.com/rryssf_/status/1976269613072843063)。
   - 他们澄清这更多是对 **RL** 的补充而非替代，并且他们几个月来一直基于这种直觉构建工具！


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1425953936938111119)** (18 messages🔥): 

> `Constant Params * Samples / Data Ratio, ThinkingMachine's LoRA Framework, LoRA vs FFT, Information Bottleneck in RL, Robust Fine Tuning Strategies` 


- **揭示参数效率的比率要求**：一名成员指出，最近的一篇论文 ([arxiv.org/html/2410.21228v1](https://arxiv.org/html/2410.21228v1)) 表明，为了实现高效训练，需要保持 **(params * samples / data) 为常数**。
   - 该成员希望论文能探索 **Chinchilla** 方案（在 260B tokens 上训练 13B 模型）之外的训练 Token 数量，以提供更多见解，因为目前的 Frontier models 通常超过 **20T tokens**。
- **LoRA 微调可能导致 Catastrophic Forgetting**：一名成员指出，正如论文中所论证的，仅观察 **Loss** 可能不足以判断微调过程中是否发生了 **Catastrophic Forgetting**。
   - 他们建议在新的训练任务和 **Pre-training evals** 上同时进行评估以衡量遗忘程度，并检查 **LoRA adapters** 的 **Singular values** 以检查相关性。
- **LoRA vs FFT：巅峰之战**：一名成员回顾了在他们 AI 生涯早期比较 **LoRA** 与 **FFT** 最大潜力的经历，并指出 **Loss** 不足以决定哪种方法更好。
   - 讨论引用了 **Thinking Machines** 的博客，该博客同样认为许多环境仅在 Attention 层使用 **LoRA**，这是低效的 ([thinkingmachines.ai/blog/lora/](https://thinkingmachines.ai/blog/lora/))。
- **揭秘 RL 中的 Information Bottleneck**：一名成员分享了一张展示 **RL** 中 **Information Bottleneck** 的有趣图片。
   - 未提供更多详细信息。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1425953936938111119)** (18 messages🔥): 

> `LoRA, FFT, 8bit, Information Bottleneck, Thinking Machines Blog` 


- **论文被 ICLR 2025 拒绝**：一名成员分享了一篇被 **ICLR 2025** 拒绝的[论文](https://arxiv.org/html/2410.21228v1)链接，并指出了其关于 **LoRA** 训练以及保持 *(params * samples / data)* 为常数的重要性见解。
   - 该成员批评该论文仅测试了 **Chinchilla** 缩放定律，而未尝试缩放到当前基于 **20T tokens** 训练的模型。
- **LoRA SFT 需要 Pre-training Evals**：据一名成员称，在 **LoRA SFT** 之后，一个成熟的指令微调流水线应该对 **SFT** 模型进行 **Pre-training evals**，以评估发生了多少遗忘。
   - 他们建议还要检查 **LoRA adapters** 的 **Singular values**，看看这些侵入维度（intruder dimensions）与预训练遗忘的相关程度。
- **仅靠 Loss 不足以判定**：成员们认为，单靠 **Loss** 不足以确定 **LoRA** 与 **FFT** 的最大潜力。
   - 另一名成员同意一篇新[论文](https://arxiv.org/html/2410.21228v1)的观点，即应同时使用新训练任务和 **Pre-training evals** 来衡量 Post training 过程中发生了多少遗忘。
- **LoRA 目标可以学习新事实**：一名成员表示，即使仅以 **Attention 层**为目标，**LoRA** 也能学习新的事实和知识。
   - 他们引用了 [Thinking Machines 博客](https://thinkingmachines.ai/blog/lora/)，指出该博客认为许多环境仅在 Attention 层使用 **LoRA**，这是低效的。
- **证明 RL 中的 Information Bottleneck**：一名成员发现了一个关于 **RL** 中 **Information Bottleneck** 的演示非常有趣，并分享了一张照片。
   - 未提供相关链接。


  

---

### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1425942869876805806)** (85 条消息🔥🔥): 

> `MCP .well-known 端点，在 structuredContent 中表示图像内容，skybridge 工具和 outputSchema` 


- **MCP 的 `.well-known` 端点进展**：成员们正在讨论用于 **MCP server metadata** 的 `.well-known/` 端点，并引用了 [博客文章](https://blog.modelcontextprotocol.io/posts/2025-09-26-mcp-next-version-update/#server-identity)、[GitHub 讨论](https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/1147)、[Pull Request](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/1054#issuecomment-3117705161) 以及 [Dev Summit 演示文稿](https://github.com/modelcontextprotocol/registry/blob/main/docs/explanations/dev-summit-2025-10-registry-status-presentation.pdf)。
   - 目标是推进一个最小化的 SEP，重点关注文档名称、相对于 MCP server URL 的位置以及最小内容（特别是 `Implementation`）。
- **图像内容标准辩论**：讨论围绕在 `structuredContent` 中表示 **image content** 展开，担忧其不可移植性，并且在直接将 `StructuredOutput` 传递给模型 API 的客户端上会出错。
   - 有建议认为协议并不规定宿主应用如何将 `structuredContent` 映射到 LLM APIs，且供应商对返回图像的工具支持较差；解决方案是给模型一个将字符串映射到图像的临时工具，并在返回结构化内容时添加它。
- **Skybridge 工具绕过 OutputSchema**：讨论指出 **skybridge** 在工具上不使用 `outputSchema`，成员们探讨了是否有机会为此定义某种规范。
   - 成员们讨论了 `ContentBlock[]` 和 `structuredContent` 的差异，其中提到 `structuredContent` 用于组件水合（widget hydration）——但它对模型也是可见的。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1426002234964967456)** (5 条消息): 

> `Mojo 语言，GPU 兼容性` 


- **Mojo 被预言为未来的语言**：成员们欢迎了一位新用户，预言 *Mojo será el lenguaje del futuro* <:mojo:1098996988797784084>。
   - 该用户还表示，*También soy de Colombia 🙈*。
- **GPU 的后向兼容性困扰**：一位用户询问某种方法是否缺乏前向兼容性。
   - 他们问道：“这种方法难道没有前向兼容性的问题吗？为了支持每一个新的 GPU 世代，你必须重新编译，不是吗？”


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1425926868682674257)** (64 条消息🔥🔥): 

> `Apple Silicon M3 GPU 支持，用于机器人的 Jetson Orin Nano，Mojo 原生 FFT 实现，SIMD 向量宽度加载技巧，复数内存交错` 


- **Apple Silicon 获得 GPU 向量加法支持**：一位成员在 **Apple Silicon M3** 上测试了 **GPU support**，运行了一个简单的向量加法示例，遇到了宿主缓冲区（host buffer）未被修改的问题，但通过显式同步解决了。
   - 还提到在 Apple GPU 上 *从 kernel 内部打印* 尚未实现，且由于共享内存，`enqueue_copy` 可能是一个空操作（no-op）。
- **Jetson Orin Nano 足以胜任机器人开发**：成员们讨论了将 **Jetson Orin Nano 8GB** 用于机器人开发，有人表达了对价格的担忧，但另一人认为它足以开发视觉系统，特别是对于 **battery life** 至关重要的资源受限部署。
   - 他们强调目标检测和图像分类模型在 Orin Nano 上运行良好，之后可以扩展到更大的系统。
- **Mojo FFT 实现落地，是否比 FFTW 更快？**：在 PR [#5378](https://github.com/modular/modular/pull/5378) 落地后，将会有 **Mojo native FFT 实现**。
   - 成员们还讨论了与 **FFTW** 相比的性能改进，一位用户表示他们需要尺寸仅在运行时确定的多维变换，FFTW 在 CPU 上表现足够好，但如果最终能有 GPU 实现就更好了。
- **内存交错对复数运算很快**：研究发现，对于具有两个分量的复数，直接将内存读取到实部和虚部交替的向量中可以有效地进行去交错（deinterleaved），从而可能提高 **SIMD** 工作中的内存局部性（memory locality）和缓存利用率。
   - 一位成员在这个 [开源项目](https://github.com/bartesaghilab/cryoluge/blob/main/src/cryoluge/image/complex_image.mojo#L98) 中分享了他们的加载技巧，并建议使用 compile package 来查看输出。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1426079606200471664)** (46 条消息🔥): 

> `模型遗忘, IID 采样问题, Prompt Injection 防御, 独立 Token Embeddings, Graph Neural Network 训练` 


- **仅靠架构技巧无法解决遗忘问题**：一名成员认为，防止**灾难性遗忘 (catastrophic forgetting)** 不仅仅需要架构上的改变；优化器必须进化到超越基于 **autodiff** 的 **gradient descent**。
   - 他们建议将记忆**硬切分 (hard-sectioning)** 到独立的存储桶中，并确保这些存储桶不会发生根本性遗忘，并指出需要 **i.i.d. sampling** 的模型天生难以应对遗忘。
- **IID 采样被证明存在问题**：一名成员指出，当 **MNIST** 数字被排序而非随机排列且不采样旧数字时，**Deep Learning (DL)** 方法会遇到困难。
   - 他们注意到没有 **i.i.d.** 的 DL 准确率仅为 10%（随机水平），这凸显了有效学习对 **i.i.d.** 的依赖。
- **独立的 Token Embeddings 能防止注入吗？**：一名成员提议为 **system/user prompts** 训练一套独立的 **token embeddings**，使模型更容易区分提示词和内容，从而降低 **prompt injection** 的脆弱性。
   - 虽然一些人认为这可以创建一个更安全的模型，但其他人认为即使有独立的 embeddings，模型仍可能发生表示坍缩 (representation collapse) 并容易受到操纵。
- **Soft Prompts 防御注入**：针对提议的想法，一名成员指出独立的 embeddings 被称为 **soft prompts**，可以防御 **prompt injection** 攻击。
   - 他们提到，虽然 token embeddings 本身没有区别，但一些 embeddings 作为实数向量（通过 backprop 找到而非离散的）被添加到上下文的前面。
- **GNNs 显示出两阶段学习？**：一名成员询问关于 **Graph Neural Network** 训练过程中显示**两阶段行为**的图表。
   - 其他人认为这可能是由于超参数设置或第一轮 epoch 的结束，导致网络重新遇到相同的输入点。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1425925907264049232)** (6 条消息): 

> `猫咪学习, 后续论文` 


- **猫咪也爱微调**：一位成员分享了一张他们的猫*和他们一起学习*的照片，配文为 [猫咪照片链接](https://cdn.discordapp.com/attachments/1045297868136779846/1425925906928767036/image.png?ex=68eaadfc&is=68e95c7c&hm=a38546b5f969ef9fde2a1e7aac3f142d7b823ee1be9436481c8f858d1ad0739e&)。
- **论文追随者的最新追求**：一位成员分享了一篇[很酷的论文](https://arxiv.org/abs/2509.24372)，它是[这篇论文](https://arxiv.org/abs/1712.06568)的后续研究，并建议*我们绝对应该看看这一篇*。


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1426145250241548299)** (2 条消息): 

> `Camel AI 更新, 伯克利 LLM Agents 课程` 


- **Camel AI 换新装**：一位用户分享了 [Camel AI](https://github.com/camel-ai/camel) 获得了一些更新，并认为其**角色扮演方法 (roleplay method)** 非常棒，有必要测试 **Workforce** 和所有工具。
- **伯克利推出 LLM Agents 课程**：一位成员分享了来自伯克利的两个 LLM Agents 课程，可能值得一看：[播放列表 1](https://www.youtube.com/playlist?list=PLS01nW3RtgopsNLeM936V4TNSsvvVglLc) 和 [播放列表 2](https://www.youtube.com/playlist?list=PLS01nW3RtgorL3AW8REU9nGkzhvtn6Egn)。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1425982703525826660)** (2 条消息): 

> `YouTube 链接, oN0nViY4gn4, 1gO2bC5xLlo` 


- **YouTube 链接刷屏频道**：两名成员发布了 YouTube 视频链接：[oN0nViY4gn4](https://youtu.be/oN0nViY4gn4) 和 [1gO2bC5xLlo](https://youtu.be/1gO2bC5xLlo)。
- **无讨论**：频道中没有发现关于这些链接的讨论。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1425924819760976122)** (18 messages🔥): 

> `GPT-5-Codex with Aider, Deepseek v3.1 with roo code, GLM 4.6 in Aider, Default Prompt Function in Aider, Aider Chat Modes` 


- **探索 GPT-5-Codex 与 Aider 的集成**：一名成员询问了在 Aider 中使用 **GPT-5-Codex** 的经验，而其他成员分享了他们使用 **Deepseek v3.1**、**gpt-oss-120b** 等模型的实验，并计划测试 **GLM 4.6** 以及像 **Qwen3-VL** 这样的视觉模型。
   - 该成员还探索了使用 **n8n 工作流** 作为 API 端点将提示词路由到合适的模型/工具，并使用较小的本地模型作为网关，在将任务发送到大型离线模型之前剥离环境密钥（secrets）。
- **GLM 4.6 订阅在 Aider 中运行良好**：成员们确认在 Aider 中使用 **GLM 4.6 订阅计划** 是可行的，并参考了 [OpenAI 兼容指南](https://aider.chat/docs/llms/openai-compat.html) 进行设置。
   - 一名成员特别提到使用 *pass* 端点进行此集成。
- **Aider 中的默认提示词配置：Ask 模式？**：一名成员询问如何配置 Aider 将 `/ask` 作为默认的提示词功能。
   - 其他人引导其参考 Aider 的 [使用模式文档](https://aider.chat/docs/usage/modes.html)，并建议在配置文件中设置 `edit-format: chat` 或 `edit-format: architect`，或者将其设置为 true 以继续进行编辑。
- **默认设置 Architect 模式？**：成员们讨论了在 Aider 配置中将默认聊天模式设置为 *architect*。
   - 一名成员表示他们将该值设置为 `true`，以便使用 `architect` 模式来分析所有提示词。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1426202856226164776)** (9 messages🔥): 

> `Haiku for Git Commits, Custom Profiles in Openrouter, Model Specification Syntax, Weak vs. Main Model, Persona Definitions` 


- **Aider 选择轻便的 Haiku 进行 Git 提交**：Aider 正在转向使用 **Haiku** 来生成 Git 提交消息，理由是其速度快且成本低，具体使用的是 `openrouter/anthropic/claude-sonnet-4`。
   - 一名成员建议使用 **gemini flash lite** 作为生成提交消息的高性价比替代方案，并补充说可以为提交消息设置一个 *弱模型 (weak model)* 以优化速度和成本。
- **自定义配置文件提升 OpenRouter 的模型管理**：用户可以在 **OpenRouter** 中管理自定义配置文件，以指定模型的自定义提示词、温度（temperature）和推理复杂度。
   - 这些配置文件可以在 Aider 配置中指定，或者通过 `/model` 和 `/editor-model` 命令指定，指向 `aider/resources/` 目录中的模型定义。
- **Aider 的模型规范语法发布**：Aider 中的 `/model` 和 `/editor-model` 聊天命令允许用户指定模型，即使是那些没有在 `.aider.conf.yml` 文件中明确定义的模型。
   - 这为即时选择模型提供了灵活性，补充了基于配置的模型设置。
- **理解 Aider 的弱模型 (Weak) 与主模型 (Main) 策略**：Aider 区分 *弱模型* 和 *主模型* 以优化性能；*弱模型* 处理诸如提交消息生成之类的任务，而 *主模型* 则处理核心编码任务。
   - 这允许通过对非关键操作使用更快、更便宜的模型来实现高效的资源分配。
- **人格定义 (Persona Definitions) 作为只读资产**：用户询问是否可以将 **人格定义**（例如来自 [vibecodingtools.tech](https://www.vibecodingtools.tech/templates) 的定义）作为 `/read-only` 资产推送给底层模型。
   - 建议仅在切换任务（例如从规划切换到编码）时加载这些人格，而不是针对每个请求都加载，并将其推送到底层模型。


  

---

### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1425929369171263549)** (23 条消息🔥): 

> `Sora 邀请码，Kimi 编程能力，Hack Club & Moonshot AI，使用 Kimi 制作视频` 


- **Sora 2 邀请码很容易获得**：一些成员表示他们可以轻松获得 **Sora 2 邀请码**，且该产品下载量已突破 *100万+*。
   - 另一位成员表示他们宁愿等待 **public release**（公开发布）。
- **Kimi 在编程任务中表现出色**：**Kimi** 在编程方面非常出色，通过 **IDE** 使用 **agentic mode/tool usage** 来执行 **python scripts** 和 **batch commands**，以了解系统信息进行调试。
   - 一位成员发现这 *直接优于大多数其他模型*。
- **Hack Club 与 Moonshot AI 无关**：成员们讨论了 **Moonshot AI** 是否涉及来自 **Hack Club** 的一封电子邮件。
   - 已确认 **Hack Club** 与 **Moonshot AI** 无关。
- **使用 Kimi 创作了疯狂的视频**：一些成员报告称使用 Kimi 制作了“疯狂的视频”。
   - 未提供关于视频性质的进一步细节。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1425920754062590114)** (10 条消息🔥): 

> `PR 中的 AI 废话 (AI Slop)，代数 Upat 测试，Tinygrad vs PyTorch，Reduce Group 颜色变更` 


- **社区辩论 Pull Requests 中的 “AI 废话”**：一位成员质疑 [PR #12530](https://github.com/tinygrad/tinygrad/pull/12530) 是否包含 *AI slop*，并认为将其斥为难以理解是逃避责任。
   - 他们为代码质量担保，特别提到了 [PR #12539](https://github.com/tinygrad/tinygrad/pull/12539)，并将其与 **geohot** 在 [issue #12449](https://github.com/tinygrad/tinygrad/pull/12449) 中的代数 Upat 测试进行了比较。
- **Tinygrad 对 AI 生成代码的立场引发社区分歧**：一位成员提到，*AI PR 将在不予评论的情况下关闭，如果你执意推送，将被封禁。*
   - 会议强调，*如果你不理解你提交的 PR 的每一行代码，就不要提交。*
- **线性代数速度：Tinygrad 还是 PyTorch？**：一位成员询问是否可以使用 **tinygrad** 来实现 **torch linalg** 的功能，如 **cross product** 和 **norm**，并询问 **tinygrad** 是否会将这些转换为 **UOps** 并生成 **C code**。
   - 他们询问在仅关注 **fast cross products** 和 **matrix multiplication** 时，**tinygrad** 是否比 **PyTorch** 更有优势。
- **Reduce Group 颜色变更为亮红色**：**reduce** 组现在标记为 *亮红色* 而不是绿色，以突出显示正在使用 local 进行 reduction。
   - 讨论指出绿色将保留给未来的功能，参见 [PR #12604](https://github.com/tinygrad/tinygrad/pull/12604/files)。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1425929038777684019)** (11 条消息🔥): 

> `tinygrad 向量操作，循环拆分 (loop splitting)，Rust 版 cuda_ioctl_sniffer，Winograd 测试失败` 


- **向量操作提问**：一位成员询问 **tinygrad** 是否支持快速向量操作，如 **cross product, norm, and trig functions**。
   - 未记录到任何回复。
- **寻求循环拆分资源**：一位成员请求关于 **loop splitting** 的框架无关学习资源，以 *在高层级修复 `cat`*。
   - 他们正在尝试一项涉及循环拆分的悬赏任务，并指出目前的实现未能通过 **3 个单元测试**。
- **Winograd 测试出现断言失败**：在尝试实现循环拆分时，该成员在 `test_winograd.py` 中遇到了 **assertion failure**。
   - 错误信息显示数值 **6.49** 在与 **2.6** 的“不小于”比较中失败。
- **用 Rust 重构 `cuda_ioctl_sniffer`**：一位成员正在将 George Hotz 的 `cuda_ioctl_sniffer` 转换为 **Rust**，并配备交互式终端以测试单个 **CUDA kernels**。
   - 他们发布了 `saxpy` kernel 输出的 [演示图片](https://cdn.discordapp.com/attachments/1070745817025106080/1425975923458445343/image.png?ex=68eadc91&is=68e98b11&hm=8472f883712ba4af77167cb978546cf31d701a98b6a67bcca03ea1b59fdab985)，目标是支持更多 GPU，使用 IOCTL 来启动 **CUDA kernels**。


  

---

### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1426128942095405076)** (9 messages🔥): 

> `Prompt Injection Tasks, DSPy Community Repo, AgentFlow` 


- **Spotlighting 降低 Prompt Injection 风险**：一名成员演示了 spotlighting 作为降低 **prompt injection tasks** 风险的工具，参考了 [Microsoft 的研究论文](https://github.com/estsauver/dspy-spotlight)。
   - 他们仍在针对 **XPIA attacks** 开发基准测试和测试套件。
- **DSPy 社区仓库启动以突出项目**：一名成员创建了 [DSPy Community repo](https://github.com/dspy-community) 来突出显示项目，以免它们消失。
   - 目前它是主主页上的一个 **README**，列出了库和项目，并且**欢迎提交 PR**。
- **AgentFlow 发布**：一名成员分享了 [AgentFlow](https://github.com/lupantech/AgentFlow) 的链接。
   - 未提供其他上下文。


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/)** (1 messages): 

batmanosama: https://arxiv.org/abs/2510.05592
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1425960034126528562)** (9 messages🔥): 

> `dspy.Tool from MCP Tool with authentication, shadcn inspiration for DSPy, TTD-DR module download, Platform/marketplace for DSPy modules, AgentFlow` 


- ****MCP Tool 身份验证困境****：一名成员询问关于从具有 [authentication](https://github.com/modelcontextprotocol/python-sdk?tab=readme-ov-file#authentication) 的 **MCP Tool** 创建 `dspy.Tool` 的问题，以及目前是否已处理此问题。
   - 该成员想知道在使用 `dspy.Tool.from_mcp_tool` 处理需要身份验证的工具时会发生什么，以及身份验证过程是否得到了妥善管理。
- ****shadcn 激发 DSPy 灵感****：一名成员建议 DSPy 可以向 **shadcn** 学习，创建一个*精美的资源管理器网站*、一个将模块放置在固定位置的 *CLI*，以及一种将*优化后的模型发布*回仓库的方式。
   - 这将使用户能够轻松地根据自己的需求调整模块，而不是依赖 `pip install`，从而促进更简单的定制。
- ****TTD-DR 模块下载愿景****：成员们讨论了直接下载 **TTD-DR (Test-Time Diffusion Deep Researcher)** 模块进行本地调整的可能性。
   - 该建议涉及扩展包功能，允许运行 `dspy install deep-research-ttd`，这将处理设置并将模块放置在 DSPy 特定的目录中。
- ****DSPy 模块市场势头****：成员们主张为 DSPy 模块创建一个**平台/市场**，以促进优化程序的共享和重用。
   - 这将涉及一个已优化程序的仓库（例如，针对 **Qwen**、**4.1-mini**、**4.1-nano** 优化的客户评论分类任务），允许用户快速部署常见任务的解决方案。
- ****AgentFlow：新玩家加入聊天****：成员们注意到了 [AgentFlow](https://github.com/lupantech/AgentFlow) 的发布。
   - 未讨论更多细节。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1425930889610924106)** (18 messages🔥): 

> `Godhand AI-assisted previz creation, Manus Support, Prompt Engineering, Cloudflare integration, Claude API` 


- ****Godhand** AI 辅助预制（previz）工作流获得关注**：一篇 [LinkedIn 帖子](https://www.linkedin.com/posts/godhand_ai-assisted-previz-creation-workflow-quick-activity-7382046352287088640-Ev0V) 展示了 **Godhand** 的 AI 辅助预制创作工作流。
   - 该工作流承诺提供一种快速高效的预制方法，有可能缩短制作时间。
- **用户要求支持人员介入！**：多名用户在频道中要求支持人员立即关注。
   - 一名对平台感到沮丧的用户惊呼：“喂！！！支持人员在哪里？！”
- **Manus 在初始项目规划中表现出色**：一名成员发现 **Manus** 非常高效，尤其是在初始规划和基础项目结构方面，仅花费 **1500 credits** 就构建了一个 RAG 宠物助手向量数据库。
   - 他建议使用 **Manus** 进行规划，然后使用 **Claude Code** 进行项目编码，强调了 Prompt 和 n8n 工作流的效率。
- **高效的 Prompt Engineering 是关键**：多名用户强调了在使用 **Manus** 等 AI 工具时，明确且详细的 Prompt Engineering 的重要性。
   - 一名用户认为，*将文件丢给任何 AI 并让它弄清楚 Prompt 的细节是非常糟糕的做法*。
- **通过 API 集成 Claude**：一名用户宣布，现在可以通过 API 调用轻松将 **Claude** 集成到 **Manus** 中。
   - 这消除了复制粘贴的需要，简化了工作流程。