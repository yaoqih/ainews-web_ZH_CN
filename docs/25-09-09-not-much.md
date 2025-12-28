---
companies:
- cognition
- founders-fund
- lux-capital
- 8vc
- neo
- vercel
- claude
- groq
- alibaba
- huggingface
- meta-ai-fair
- google
- theturingpost
- algoperf
date: '2025-09-09T05:44:39.731046Z'
description: '**Cognition** 以 **102 亿美元**的估值融资 **4 亿美元**，用于推进 AI 编程智能体（coding agents），**swyx**
  也已加入该公司以支持“智能体十年”（Decade of Agents）的论点。**Vercel** 发布了一个开源（OSS）的“氛围编程平台”（vibe coding
  platform），该平台使用了经过微调的 **GPT-5** 智能体循环。**Claude Code** 强调在智能体循环中采用极简主义以确保可靠性。**Kimi
  K2-0905** 在编程评估中达到了 94% 的得分，并通过翻倍的上下文长度提升了智能体能力。**阿里巴巴**发布了 **Qwen3-ASR**，这是一款词错率（WER）低于
  8% 的多语言转录模型。**Meta** 推出了 Set Block Decoding 技术，在不改变架构的情况下实现了 3-5 倍的解码加速。KV 缓存压缩和量化方面的创新包括
  **AutoRound**、**QuTLASS v0.1.0** 和 **AlgoPerf v0.6**。**谷歌的 Veo 3** 视频生成 API 已正式商用（GA），并伴随着大幅降价和对竖屏视频的支持。'
id: MjAyNS0w
models:
- gpt-5
- kimi-k2-0905
- glm-4.5
- qwen3-asr
- opus-4.1
people:
- swyx
- tim_dettmers
title: 今天没什么事。
topics:
- coding-agents
- agent-architecture
- open-source
- model-evaluation
- multilingual-models
- speech-recognition
- model-optimization
- kv-cache
- quantization
- algorithmic-benchmarking
- video-generation
- context-windows
---

**平静的一天**

> 2025年9月8日至9月9日的 AI 新闻。我们为您检查了 12 个 Reddit 分版、544 个 Twitter 账号和 22 个 Discord 社区（包含 187 个频道和 4104 条消息）。预计节省阅读时间（以 200wpm 计算）：337 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以精美的 vibe coded 风格展示所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻分类，并在 @smol_ai 上向我们提供反馈！

苹果 iPhone 发布会提供了一些[小更新](https://news.ycombinator.com/item?id=45186015)。

---

# AI Twitter 回顾

**编程 Agent 与工具势头**

- **Cognition 融资 4 亿美元以扩展 Devin**：Cognition 宣布完成 4 亿美元融资，投后估值达 102 亿美元，旨在“推进 AI 编程 Agent 的前沿”，由 Founders Fund 领投，Lux、8VC、Neo 等跟投。团队强调了客户规模的扩大以及 Windsurf 团队的加入，目前正在招聘产品、基础架构和 Post-training 方面的人才（[公告 1](https://twitter.com/cognition/status/1965086655821525280)，[2](https://twitter.com/cognition/status/1965086662612177299)，[团队说明](https://twitter.com/cognition/status/1965086661253185645)，[计划片段](https://twitter.com/cognition/status/1965185627357683776)）。评论：@swyx 宣布加入 Cognition，阐述了他为何“看好” Agent 实验室的论点，以及跨同步/异步工作流的定位对于在“Agent 十年”中占据主导地位的重要性（[推文串](https://twitter.com/swyx/status/1965183110016098617)）。
- **Agent 开发栈变得更简单且更强大**：
    - Vercel 发布了一个开源的“氛围编程（vibe coding）平台”，该平台基于 Vercel AI SDK、Gateway、Sandbox 和经过调优的 GPT-5 Agent 循环构建（工具使用：文件 IO、命令、包安装、自动修复），并演示了通过 one-shot 编码实现了一个用 Go 语言编写的多人 Pong 游戏（[演示](https://twitter.com/rauchg/status/1964857952722133231)）。
    - Claude Code 的循环有意保持极简：单个主循环 + 异步缓冲区、直接工具和基于 TODO 的规划；在可调试性和可靠性方面，简单性胜过 Swarm 编排（[分析](https://twitter.com/imjaredz/status/1965083721713041564)）。
    - 编程评测：运行在 Groq 上的 Kimi K2-0905 在 Roo Code 上达到了 94% 并排名第 7，成为首个突破 90+ 的开放权重（open-weight）模型，同时也是前 10 名中最快、最便宜的模型（[排行榜](https://twitter.com/roo_code/status/1965098976677658630)）。Tim Dettmers 报告称，编程助手的实际前沿正日益向开放权重模型倾斜：GLM-4.5 仅需“每月 3 美元”且具备接近 Sonnet 的质量；Kimi K2.1 Turbo 比 Opus 4.1 快约 3 倍且便宜约 7 倍，而 GPT-5 主要在复杂的规格说明工作中表现出色（[观点](https://twitter.com/Tim_Dettmers/status/1965021602267217972)）。

**模型与推理进展**

- **Kimi K2 0905 与 Qwen3-ASR**：
    - Kimi K2 0905（1T 参数，架构未变）提升了 Agent 能力：Terminal-Bench Hard 从 14% 提升至 23%，Tau2-Bench Telecom 从 61% 提升至 73%；上下文窗口从 128k 翻倍至 256k。在 Artificial Analysis 的 AAII 指数上智能度 +2；现已在 Kimi 官网提供服务（[摘要](https://twitter.com/ArtificialAnlys/status/1965010554499788841)，[实时说明](https://twitter.com/crystalsssup/status/1965017719058960732)）。
    - 阿里巴巴的 Qwen3-ASR 发布了一个用于多语言转录（中/英 + 9 种语言）的单一模型，支持自动检测，对背景音乐/噪音/说唱具有鲁棒性，词错率（WER）<8%，并支持自定义上下文偏置。ModelScope/HF 上已有演示；API 已上线（[发布](https://twitter.com/Alibaba_Qwen/status/1965068737297707261)）。
- **更快的解码与更轻量的 KV**：
    - Meta 的 Set Block Decoding (SBD) 可以在不改变架构的情况下，使现有 LM 的解码速度提高 3-5 倍，匹配 NTP 性能并保留精确的 KV cache——通过掩码/离散扩散公式实现并行生成（[概述](https://twitter.com/HuggingPapers/status/1965084731839513059)，[详情](https://twitter.com/itai_gat/status/1965112129499046230)）。
    - KV cache 与量化创新：AutoRound 现已集成到 SGLang 中（[PR](https://twitter.com/HaihaoShen/status/1964926924880523701)），Turing Post 调研了 KV 压缩技术（量化、低秩、Slim Attention、XQuant）及其权衡（[推文串](https://twitter.com/TheTuringPost/status/1964971207188791464)），QuTLASS v0.1.0 为 Blackwell GPU 带来了 4-bit NVFP4 微缩放和快速变换（[发布](https://twitter.com/DAlistarh/status/1965157635617087885)）。AlgoPerf v0.6 增加了滚动排行榜、JAX jit 以及更低的算法基准测试计算成本（[更新](https://twitter.com/algoperf/status/1965044626626342993)）；HF 记录了 PyTorch 的 ZeroGPU AOT 编译内部机制（[博客](https://twitter.com/charlesbben/status/1965046090945954104)）。

**多模态生成、视频与“氛围编程”**

- **Veo 3 正式发布 (GA) 且价格更低**：Google 的 Veo 3 和 Veo 3 Fast 现已在 Gemini API 中正式发布 (GA)，降价约 50%（分别为 $0.40/s 和 $0.15/s），支持 1080p 输出和 9:16 垂直视频——定位于规模化生产 ([开发博客](https://twitter.com/googleaidevs/status/1965160822260318702), [价格详情](https://twitter.com/_philschmid/status/1965161626761326983), [PM 笔记](https://twitter.com/OfficialLoganK/status/1965193765146296467))。
- **社区工作流与工具**：
    - “Nano Banana”（Gemini 2.5 Flash Image Preview）催生了一个周末的 “vibe‑coded”（氛围感编码）项目——现已在 Google AI Studio 中开源供二次创作；团队报告称其支持一键重用，并带有一些有趣的特性（例如：时钟总是渲染为 10:10）([开源包](https://twitter.com/arrakis_ai/status/1965001417716072877), [特性说明](https://twitter.com/fabianstelzer/status/1965001753059057925))。
    - Qwen 的“论文 → 网站”流程可在几分钟内将研究论文转化为可部署的网站 ([演示](https://twitter.com/Alibaba_Qwen/status/1964870508421480524))。Lmarena 增加了多轮图像编辑评估，以便社区比较不同模型（包括 “nano banana”）的迭代优化能力 ([功能介绍](https://twitter.com/lmarena_ai/status/1965150440401809436))。针对文档 RAG 用户体验，ColQwen2 + Weaviate 实现了 Token 级相似度图谱，用于视觉 PDF 搜索和区块高亮 ([构建指南](https://twitter.com/helloiamleonie/status/1964997028875743637))。

**Agent、训练后 RL 与评估实践**

- **迈向迭代自我改进**：FAIR 的 Exploratory Iteration (ExIt) 通过自动课程训练 LLM 进行推理时自我改进，该课程从模型自身的先前响应中引导，优先处理 GRPO 组中具有高回报方差的部分历史记录。ExIt 在竞赛数学、BFCLv3 多轮任务和 MLE‑bench (+22%) 上优于 GRPO，且仅训练单步改进 ([推文](https://twitter.com/MinqiJiang/status/1965055909605916892))。
- **在线 vs 离线 RL 与评估**：
    - 证据持续表明，在大规模应用中，在线 RL (PPO/GRPO) 的性能优于 DPO 等离线方法，尽管半在线迭代（同策略采样 + 负梯度）缩小了这一差距；数据质量在算法选择中仍占主导地位 ([总结](https://twitter.com/cwolferesearch/status/1965088925510520853))。
    - 为什么许多 Agent 表现不及预期：与生成任务相比，决策制定的容错率接近于零且数据稀疏；大多数失败源于粗略的任务范围界定和非结构化环境，而非 LLM 本身的缺陷 ([辩论回顾](https://twitter.com/ZhihuFrontier/status/1964928650081698167))。
    - RAG 评估正从“死”的单元测试转向“活”的循环：RAGGY（开源 REPL）支持 RAG 的假设性迭代，目前正大力推动将预生产测试与生产环境可观测性和人工审核集成，而非将其视为独立的孤岛 ([RAGGY](https://twitter.com/HamelHusain/status/1965052554997600449), [评估观点](https://twitter.com/bnicholehopkins/status/1965130607790264452))。另请参阅利用工具使用和多步推理的实用 “Agentic RAG” 架构 ([指南](https://twitter.com/omarsar0/status/1965115682322042954))。

**机器人与具身智能**

- **通过 RL 实现多机器人规划**：Google DeepMind 的 RoboBallet（与 Intrinsic 和 UCL 合作）可协调多达 8 个机器人手臂进行无碰撞的任务与运动规划，性能优于传统方法约 25%，并通过 RL 学习的协作原则在几秒钟内泛化到新的工作流 ([公告](https://twitter.com/GoogleDeepMind/status/1965040645103407572), [更多详情](https://twitter.com/GoogleDeepMind/status/1965040648400351337))。
- **开源硬件栈与灵巧操作**：Pollen Robotics 为 Reachy 2 配备了双开源 “Amazing Hand” 抓取器，用于精细操作；原生集成即将推出 ([演示](https://twitter.com/pollenrobotics/status/1964987735829266871))。X Square 发布了 WALL‑OSS（开源基础模型）和具备自动拖地功能及灵巧手的 Quanta X2 机器人；阿里云领投了其 1.4 亿美元的 A+ 轮融资（不到 2 年内融资超过 2.8 亿美元） ([总结](https://twitter.com/ZhihuFrontier/status/1964968113990164810))。OpenPI 的 pi‑05 现已在 openpi 中提供并支持 PyTorch ([发布](https://twitter.com/svlevine/status/1965161524722630734))。

**基准测试、排行榜与企业级应用**

- **文本排行榜动态**：lmarena 在其 Top 10 文本排行榜中新增了两个条目：Qwen3‑max‑preview（第 6 名，闭源）和 Kimi‑K2‑0905‑preview（第 8 名，修改版 MIT），使 Kimi 与 Qwen 和 DeepSeek 变体一同成为顶级开放权重模型的有力竞争者（[更新](https://twitter.com/lmarena_ai/status/1965115050273976703)，[模型链接](https://twitter.com/lmarena_ai/status/1965124408097517853)）。Artificial Analysis 对 K2‑0905 的测量结果也反映了其 Agent 性能的提升（[详情](https://twitter.com/ArtificialAnlys/status/1965010554499788841)）。
- **政府与企业**：
    - Perplexity 推出了 “Perplexity for Government”：默认安全、零数据使用、高级模型访问权限，且无需企业合同；同时将 Perplexity Finance 引入 iOS/Android 平台（[发布](https://twitter.com/perplexity_ai/status/1965030156415980009)，[后续](https://twitter.com/AravSrinivas/status/1965032305053065590)，[财经移动版](https://twitter.com/AravSrinivas/status/1965100159488196757)）。
    - Anthropic 支持加州 SB 53 法案（参议员 Scott Wiener 提出），这是一个以透明度为核心的州级框架，旨在替代联邦标准来监管前沿 AI（[声明](https://twitter.com/AnthropicAI/status/1965027311717388673)，[背景](https://twitter.com/jackclarkSF/status/1965048896784367847)）。

热门推文（按互动量排序）

- Cognition 以 102 亿美元估值融资 4 亿美元，用于扩展 AI 编程 Agent（[公告](https://twitter.com/cognition/status/1965086655821525280)）
- Vercel 的开源 vibe coding 平台通过调优的 GPT‑5 循环，一次性生成了一个用 Go 语言编写的多人 Pong 游戏（[演示](https://twitter.com/rauchg/status/1964857952722133231)）
- Qwen3‑ASR：支持多语言 ASR 的单一模型，WER 低于 8%，对噪声/背景音乐具有鲁棒性，并支持上下文注入（[发布](https://twitter.com/Alibaba_Qwen/status/1965068737297707261)）
- Google AI Mode 扩展至印地语、印尼语、日语、韩语和巴西葡萄牙语（[Sundar Pichai](https://twitter.com/sundarpichai/status/1965115123330388467)）
- Veo 3 正式发布（GA），价格下调约 50%，在 Gemini API 中支持 1080p 和竖屏视频（[开发者更新](https://twitter.com/googleaidevs/status/1965160822260318702)）

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. A3B HF 发布：Qwen3-Next-80B-Instruct & ERNIE-4.5-21B-Thinking

- [**Qwen 3-Next 系列，Qwen/Qwen3-Next-80B-A3B-Instruct 现身**](https://github.com/huggingface/transformers/pull/40771) ([得分: 472, 评论: 134](https://www.reddit.com/r/LocalLLaMA/comments/1nckgub/qwen_3next_series_qwenqwen3next80ba3binstruct/)): **阿里巴巴的 Qwen3-Next 为长上下文、高性价比 LLM 引入了架构变更，特别是混合注意力（Hybrid Attention）栈（Gated DeltaNet + Gated Attention）、激活比例为** `1:50` **的高稀疏度 MoE，以及多 Token 预测（Multi‑Token Prediction, MTP）和稳定器（零中心、权重衰减的 LayerNorm）。发布的 Qwen3‑Next‑80B‑A3B（总参数** `80B`**，激活参数** `~3B`**）据报道在下游任务上优于 Qwen3‑32B，而训练成本不到其** `1/10`**；对于超过** `32K` **Token 的上下文，其推理吞吐量提高了** `10 倍`**以上；详情见该项目的 [博客文章](https://qwenlm.github.io/blog/qwen3_next/)。上游支持已通过 [PR #40771](https://github.com/huggingface/transformers/pull/40771)（12 次提交，15 个文件，** `+2,964/−2` **行代码）进入 Hugging Face Transformers，并引用了 [Qwen3 仓库](https://github.com/QwenLM/Qwen3)，表明 Qwen3‑Next 系列的模型/分词器配置及测试已完成集成。**
    - Qwen（阿里巴巴）概述了 Qwen3-Next 系列的新架构，特别是在已发布的模型 **Qwen/Qwen3-Next-80B-A3B-Instruct** 中：结合了 **Gated DeltaNet + Gated Attention** 的混合注意力机制，用于改进预训练和加速推理的**多 Token 预测（MTP）**，以及如零中心、权重衰减 LayerNorm 等稳定性调整。他们声称通过高稀疏度 MoE 实现了 `80B` 总参数量，而激活参数仅为 `3B`，在下游任务上表现优于 **Qwen3-32B**，训练成本仅为后者的 <`1/10`，且在 >`32K` Token 的上下文环境下实现了 >`10x` 的推理吞吐量（[博客](https://qwenlm.github.io/blog/qwen3_next/)）。
    - 讨论中将 `1:50` 的 MoE 激活比例与其他模型进行了基准对比：**GPT-OSS-12B** 激活比例为 `4/128`（约 `1:32`），**V3/R1** 为 `9/257`（约 `1:29`），**K2** 为 `9/385`（约 `1:43`），而 **LongCat-Flash** 平均为 `9/513`（约 `1:57`），尽管其较大的共享专家（shared expert）增加了实际的激活参数占比。因此，Qwen3-Next 的路由稀疏度是这组模型中最激进的之一，引发了人们对单个专家在不降低质量的情况下能缩小到何种程度的关注。

- [**baidu/ERNIE-4.5-21B-A3B-Thinking · Hugging Face**](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Thinking) ([分数: 237, 评论: 59](https://www.reddit.com/r/LocalLLaMA/comments/1nc79yg/baiduernie4521ba3bthinking_hugging_face/)): **百度发布了 [ERNIE-4.5-21B-A3B-Thinking](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Thinking)，这是一个参数量约为** `~21B` **的文本 MoE 模型，每个 token 激活参数约为** `~3B` **(A3B)，专注于增强多步推理能力并支持** `128K` **上下文。它提供了与 [transformers ≥4.54.0](https://github.com/huggingface/transformers)、[vLLM](https://github.com/vllm-project/vllm) 和 [FastDeploy](https://github.com/PaddlePaddle/FastDeploy) 兼容的 Transformer 风格权重，支持 tool/function calling，并根据 [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) 协议发布。社区 GGUF 版本可在 [gabriellarson/ERNIE-4.5-21B-A3B-Thinking-GGUF](https://huggingface.co/gabriellarson/ERNIE-4.5-21B-A3B-Thinking-GGUF) 获取。** 评论指出其基准测试可能存在选择性（仅与更强的模型对比），并希望能有适配单张 16GB GPU 的 Q4/Q5 GGUF 量化版本，以作为 Qwen3-30B-A3B 的竞争对手；文中还分享了一张基准测试图片供审查。
    - 一些人指出基准测试的设定看起来像是“精挑细选”的：发布的图表似乎主要与已经超越 `ERNIE-4.5-21B-A3B-Thinking` 的更强基准模型进行对比，这掩盖了它实际领先或落后的地方；详情请参阅分享的图片 (https://preview.redd.it/0e10f0pbw1of1.png?width=3840&format=png&auto=webp&s=916b8f0777cb166e44833224bd30af0291d312d4)。在 CNsimpleqa 上的大幅下滑与在其他地方更具竞争力的结果形成对比，引发了 “benchmaxxing” 的担忧——即针对特定数据集的微调虚高了热门排行榜的分数，但在针对性较弱的中文问答中表现不佳。为了验证泛化能力，人们呼吁建立更广泛、同类对比的基准（例如 Llama 3.1 70B/8B, Qwen2.5/3 14B/32–30B）并提供完整的指标明细。
    - 设备端可行性：一个 21B 模型在 Q4 量化下仅权重约为 `~10.5 GB`，Q5 下约为 `~13.1 GB`，因此通过精细的 KV cache 以及 batch/context 管理，`ERNIE-4.5-21B-A3B-Thinking` 有可能运行在单张 16 GB GPU 上；相比之下，30B 模型（如 `Qwen3-30B-a3b`）仅权重在 Q4 下约为 `~15.0 GB`，Q5 下约为 `~18.8 GB`，一旦计入运行时开销和 KV cache，Q5 将无法运行，而 Q4 也处于边缘。由于 “A3B/Thinking” 风格倾向于输出更长的推理过程，在长上下文场景下 KV cache 可能会占据主导内存，因此实际的单 GPU 使用可能需要短上下文、小 batch 以及激进的 paged-KV 或 offloading 技术。
    - 对 `Ernie-4.5-VL-28B` 尤其是 `Ernie-4.5-VL-424B` 支持的需求凸显了基础设施的限制：即使在 4-bit 量化下，424B 模型的权重也高达 `~212 GB`，这需要多 GPU tensor/pipeline parallelism（例如仅权重就需要 ≥3×80 GB，KV/vision tower 还需要更多）。完善的 HF 集成还需要 vision encoder + projector 的连接（类似 CLIP/ViT 的架构、图像 token 化），以及支持异构计算（CPU offload/ZeRO, paged attention）的推理后端，从而使 28B 模型变得可控，并让 424B 模型至少能进行演示。

### 2. 开源 SOTA 挑战者 (PyDevMini-1, ROMA Seal-0/FRAMES, Apertus)

- [**PyDevMini-1：一个在 Python 和 Web 开发代码方面媲美/超越 GPT-4 的 4B 模型，体积仅为其 1/400！**](https://v.redd.it/nh9fq7qbn2of1) ([Score: 295, Comments: 91](https://www.reddit.com/r/LocalLLaMA/comments/1ncam9h/pydevmini1_a_4b_model_that_matchesoutperforms/))：**发布 PyDevMini-1，这是一个针对 Python 和 Web 开发编码的约** `~4B` **参数微调模型，基于 Qwen 基础模型（作者引用为 “Qwen3-4B-Instruct-2507”），声称在约** `1/400` **的体积下实现了 GPT-4 级别的表现，可在单张游戏 GPU 上运行。该模型强调真实世界演示而非基准测试（提供对比视频），并提供免费的 Colab 用于复现；训练致谢包括 Qwen ([repo](https://github.com/QwenLM/Qwen3))、用于高效微调的 Unsloth Duo，以及 Tesslate 的 Web 开发数据 ([WEBGEN-4B-Preview](https://huggingface.co/Tesslate/WEBGEN-4B-Preview))。关键规格：** `4.0B` **参数（**`3.6B` **非嵌入参数），** `36` **层，GQA（**`32` **个 Q 头 /** `8` **个 KV 头），原生上下文长度** `262,144`**；推荐解码参数：** `temp=0.7`**，**`top_p=0.8`**，**`top_k=20`**，**`min_p=0`**。相关链接：模型权重 ([HF](https://huggingface.co/bralynn/pydevmini1))，演示/试用 Colab ([Colab](https://colab.research.google.com/drive/1c8WCvsVovCjIyqPcwORX4c_wQ7NyIrTP?usp=sharing))，社区 Discord ([邀请](https://discord.gg/RqwqMGhqaC))。路线图优先级：掌握 Tool-calling 和长上下文鲁棒性。** 评论者要求与基础模型 **Qwen3-4B-Instruct-2507** 进行严格的端到端编码基准对比，以验证微调收益并检测退化；他们还指出，目前缺乏 Tool-calling 支持是构建严肃 Coding Agent 的障碍。其他反馈指出，展示的任务可能与训练数据存在重叠（建议在大型未见代码库上进行 Bug 修复测试），并要求对 **Tesslate** 的数据集进行正确的归属引用/链接，而非重新上传（Apache-2.0 协议）。
    - 真实世界鲁棒性担忧：虽然小模型结果看起来很强，但评论者怀疑许多展示的任务可能出现在训练集中，并要求在大型真实代码库（例如，跨 `100k+` 行代码修复 Bug）上进行评估，以测试长上下文导航和多文件推理能力。他们还指出帖子忽略了 Tool-calling；现代 Coding Agent 被期望能够执行工具（运行测试、编辑文件、调用函数），即便静态基准测试看起来不错，缺乏这种能力也可能限制实际的编码表现。
    - 与强力 4B 基准的对比请求：特别是与 **Qwen3-4B-Instruct-2507** 的端到端编码基准对比，以验证微调是否真的提升（或至少没有退化）了基础模型。建议的证据包括在相同的 Prompt、上下文限制和 Tokenizer 设置下，在常用代码集（如 HumanEval/MBPP/LiveCodeBench）上的标准 pass@1/pass@k 指标，以证实其媲美/超越更大模型的说法。
    - 可操作的评估建议：运行 Aider “polyglot” 测试套件的 Python 部分并报告第二轮得分，这比单次 QA 更好地反映了迭代编辑-测试循环。链接：https://github.com/Aider-AI/aider。提供完整套件结果和仅限 Python 的细分结果，将为 4B 模型的端到端编码能力提供更真实的视角。
- [**名为 ROMA 的开源 Deep Research 仓库在 SEAL-0 和 FRAMES 上击败了所有现有的闭源平台（ChatGPT, Perplexity, Kimi Researcher, Gemini 等）**](https://i.redd.it/sxii7uog37of1.jpeg) ([Score: 162, Comments: 9](https://www.reddit.com/r/LocalLLaMA/comments/1nctfdv/opensource_deep_research_repo_called_roma_beats/))：**该帖子宣布了一个开源的 “Deep Research” 框架 ROMA ([repo](https://github.com/sentient-agi/ROMA))，声称在 SEAL-0 和 FRAMES 基准测试中相对于闭源平台（ChatGPT, Perplexity, Kimi Researcher, Gemini）取得了 SOTA 结果。ROMA 被描述为一个即插即用的系统，结合了递归规划、多 Agent 架构和 Web 搜索工具；附带的图片似乎是一个将 ROMA 与这些服务进行对比的基准测试排行榜。提供的链接包括 GitHub 仓库和一条宣传性的 X 帖子。** 热门评论质疑其自称的优越性，指出潜在的基准测试偏差，并指出 Gemini 通过 Google 搜索获得的优势；他们还要求与专有的 “Deep Research” 模式（OpenAI Deep Research, Grok DeepSearch, Gemini Deep Research）进行端到端对比结果，并询问真实用户的体验。
    - 基准测试范围差距：评论者指出 ROMA 是与通用聊天产品进行对比，但忽略了专门的闭源 “Deep Research” Agent。如果没有在 **SEAL-0** 和 **FRAMES** 上与 **OpenAI Deep Research**、**Grok DeepSearch** 和 **Gemini Deep Research** 进行端到端对比，SOTA 的说法很难验证。请求包括发布每个任务的准确率、引用忠实度和错误细分，并使用固定种子、执行日志以及相同的浏览配额/User-Agent 以确保可复现性。

- 检索栈混淆因素：一个主要的反对意见是 **Gemini** 可能会利用 Google 的第一方索引，这可能会在独立于 Agent 规划器的情况下主导结果——*“它不可能被击败，尤其是因为它使用了 Google 的内部搜索索引。”* 为了公平起见，评论者建议对后端进行归一化，或者根据检索设置（`no-search`、`public SERP`、`first‑party index`）对结果进行分层，并对查询进行时间冻结，以便差异能反映规划/工具使用能力，而非搜索特权。
- 即插即用的多模态和实时工具：关注点集中在 ROMA 是否能干净地切换 VLM/ASR 组件（例如 GPT-4o, Gemini 1.5）用于页面解析、OCR 以及表格/图表提取，这在 **FRAMES** 的截图/PDF 密集型跳转中至关重要。寻求关于工具如何注册（浏览器控制器、爬虫、检索器、验证器）、流式传输/延迟限制、速率限制处理以及反爬策略的技术细节，以判断其可移植性以及基准测试中的增益是否能在生产环境中持续。
- [**瑞士刚刚发布了 Apertus，这是一个完全开源的 LLM，仅在公共数据上训练（8B 和 70B，支持 1000 多种语言）。完全透明：权重、数据、方法全部公开。终于，欧洲在推动 AI 独立。这正是我们需要更多的开放性！**](https://i.redd.it/pmfv6zvyp3of1.png) ([Score: 258, Comments: 31](https://www.reddit.com/r/LocalLLM/comments/1ncfg23/switzerland_just_dropped_apertus_a_fully/))：**瑞士发布了 “Apertus”，这是一个包含 8B 和 70B 尺寸的开源 LLM 套件，完全在涵盖 1,000 多种语言的公共数据上训练，并提供权重、数据集和训练方法的完全透明度，以实现可审计性和可复现性。该项目将其定位为欧洲推动 AI 主权/独立的一次尝试，并强调数据来源的清晰度，而非抓取私有源。** 根据 LocalLLaMA 线程（[讨论链接](https://www.reddit.com/r/LocalLLaMA/comments/1n6eimy/new_open_llm_from_switzerland_apertus_40_training/)）的早期社区反馈，其性能相对于 SOTA 并不理想，一些争论集中在仅限于“公共数据”是否阻碍了模型能力。
    - 链接线程中的早期报告表明，Apertus 的初始质量低于预期；评论者指出其主观表现较弱，并要求提供严格的公开基准测试。参见讨论：https://www.reddit.com/r/LocalLLaMA/comments/1n6eimy/new_open_llm_from_switzerland_apertus_40_training/。为了准确定位 `8B` 和 `70B` 变体，人们要求提供在标准套件（如 MMLU, HellaSwag, GSM8K, MT‑Bench）上与 Llama 和 Mistral 基准线的对比数据。
    - 问题集中在所使用的确切“公共数据”上：包括哪些语料库、许可证、去重、过滤，以及针对声称的 `1k+` 语言的多语言采样策略。这里的技术透明度（数据集列表、策展流水线、Tokenizer 选择、每种语言的 Token 份额以及污染检查）对于可复现性以及理解性能为何在特定领域落后或领先至关重要。
    - 与 **Mistral** 的对比兴趣很高；评论者希望在 Apertus `8B/70B` 与 Mistral `7B/8x7B`（以及 Llama `8B/70B`）之间进行对等评估（相同的上下文窗口、Prompt 格式、解码参数）。清晰的评估卡和推理设置将减少方差，并使任何欧洲“AI 独立”的主张变得可衡量。
- [**🤔**](https://i.redd.it/1x8wy1p0k5of1.png) ([Score: 373, Comments: 69](https://www.reddit.com/r/LocalLLaMA/comments/1ncl0v1/_/))：**该图片/帖子预告了阿里巴巴的 Qwen 技术栈：一个新的 ASR 服务 Qwen3-ASR-Flash，构建在 Qwen3-Omni 之上，并在“数千万”小时的多模态/ASR 数据上进行了训练（[来源](https://x.com/cherry_cc12/status/1965227154813440163)）。它还提到了 “Qwen Next, 1:50 sparsity, 80A3B”，暗示了一种稀疏的 MoE 风格配置（可能每个 Token 对应 50 个专家中的 1 个激活专家）以及一些模型/集群的缩写，尽管帖子中并未澄清 “80A3B” 的确切含义。** 评论大多是非技术性的；没有讨论实质性的基准测试或消融实验。
    - Qwen 团队预告：[Qwen3-ASR-Flash](https://x.com/cherry_cc12/status/1965227154813440163) 是一个构建在 **Qwen3-Omni** 之上的语音识别服务，据报道使用包括 `数千万` 小时量级 ASR 数据集在内的多模态数据进行了训练/微调。重点在于通过大规模监督音频-文本数据，利用强大的通用骨干网络进行 ASR，这表明与典型的仅 ASR 预训练方案相比，它在跨领域和口音方面具有显著的鲁棒性。

- 提及即将推出的 MoE 配置：“Qwen Next, `1:50` 稀疏度, `80A3B`” 意味着极高的专家数量，每个 token 仅激活 50 个专家中的 1 个（极端稀疏性），并且该符号暗示了较小的激活参数预算。这种路由方式可以在保持每个 token 的 FLOPs 接近较小稠密模型的同时，实现巨大的总容量，从而提高推理吞吐量和内存局部性。
- 模型命名暗示：“MOE 多模态 qwen `40B-4A`，比 `2507` 提升了 `20%`” 和 “Qwen4-`235B-A1B`” 建议采用 TotalParams-ActiveParams（总参数-激活参数）的方案（例如，总参数 `40B`，激活参数 `4B`；总参数 `235B`，激活参数 `~1B`）。宣称的相比之前 “2507” 基准（未指明指标）约 `20%` 的提升，表明在限制激活计算量的同时，通过 MoE 扩展获得了可衡量的收益。

## 较少技术性的 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Anthropic Claude 性能下降事件及用户流失讨论

- [**关于近期性能问题的更新**](https://www.reddit.com/r/ClaudeAI/comments/1nc4mem/update_on_recent_performance_concerns/) ([Score: 609, Comments: 283](https://www.reddit.com/r/ClaudeAI/comments/1nc4mem/update_on_recent_performance_concerns/)): **Anthropic 报告了两个影响部分 Claude 用户的模型质量 Bug，根据其状态页显示，目前均已解决：一个导致 `8月5日至9月4日` 期间一小部分 Claude Sonnet 4 请求的输出质量下降（其中 `8月29日至9月4日` 期间影响较大），另一个影响了 `8月26日至9月5日` 期间的部分 Claude Haiku 3.5 和 Claude Sonnet 4 请求（[事件详情](https://status.anthropic.com/incidents/72f99lh1cj2c)）。他们声明并非故意降低质量，正在调查关于 Claude Opus 4.1 的报告，并正在部署更多的实时推理监控和对话复现工具；用户可以通过 Claude Code 中的** `/bug` **或 [Claude.ai](http://claude.ai/) 上的 👎 报告问题。** 评论者对“小部分”的说法表示质疑，并要求透明度和证据，援引社区基准测试，并对潜在的量化/质量限制以及客户赔偿表示担忧。其他人则传闻称性能有所改善，并建议使用类似遥测的信号（例如，脏话率）来检测回归。
    - 多位用户挑战 Anthropic 关于“轻微 Bug”的解释，引用了最近几周社区运行的基准测试，这些测试表明存在系统性降级。他们特别质疑模型是否在 `8月28日` 使用限制后被悄悄量化或以其他方式更改，并要求通过透明的变更日志、可复现的 evals 和清晰的模型/版本指纹提供证据，同时讨论对服务降级的客户赔偿。
    - 几条评论指出了可观测性方面的差距：尽管有广泛的报告，但严重的质量下降据称持续了约 3 周，这暗示除了延迟/运行时间之外，内部质量遥测不足。用户假设存在特定人群的影响（A/B 测试桶、地区或流量类别），这解释了为什么有些人觉得 **Claude Code** 未受影响，而其他人则报告了重大回归，并要求提供详细的 RCA 而非通用的“Bug”标签。
    - 一位 CTO 报告称，已将一个团队（约 `26` 名全职员工 + `12` 名承包商）从 **Claude Code** 转向 **OpenAI Codex**，并强调了决策杠杆：在复杂应用上的 one-shot 能力、速度（延迟和 tokens/sec）、有效上下文窗口与发布窗口的对比（声称 **Claude Code** 在上下文达到 `~50%` 后质量下降）、原始编程 IQ 以及编程直觉。成本次于质量；他们引用了行业轶事（例如 Simon Willison），显示 Codex 的强劲表现，并据此配置公司 OpenAI 账户。
- [**Anthropic 确认持续一个月的 Claude 模型质量问题**](https://status.anthropic.com/incidents/72f99lh1cj2c) ([Score: 234, Comments: 62](https://www.reddit.com/r/ClaudeAI/comments/1nc4kma/monthlong_issue_with_claude_model_quality/)): **Anthropic [确认](https://status.anthropic.com/incidents/72f99lh1cj2c)了两个独立的 Bug，这些 Bug 导致了 Claude 输出质量下降，并表示修复程序已部署。问题 1 影响了 `8月5日至9月4日` 期间的 *“小部分”* Claude Sonnet 4 请求（`8月29日至9月4日` 严重程度增加）；问题 2 影响了 `8月26日至9月5日` 期间的部分 Claude Haiku 3.5 和 Claude Sonnet 4 请求。他们正在监控关于 Claude Opus 4.1 的报告；受影响的范围包括** `claude.ai`**, **`console.anthropic.com`**, **`api.anthropic.com`**, 以及 **`Claude Code`**。Anthropic 表示降级并非故意；然而，尚未发布技术性 RCA、定量影响比例或离线基准测试差异。** 评论者质疑缺乏补救措施（退款/积分），并批评事故响应缓慢且不透明；一些人报告称修复后性能仍然下降，敦促采取更快的行动和更清晰的指标。

- 许多用户报告称，尽管 Anthropic 承认并据称已采取缓解措施，**Claude** 的输出质量仍然有所下降，这表明该事件对所有人来说并未完全解决。他们将其描述为模型行为/质量长达一个月的退化（regression），而非短暂的故障，这暗示了回滚不完整或服务/模型流水线中存在遗留问题。
- 强烈要求进行正式的技术复盘（post-mortem）：包括退化开始的精确时间线、如何检测到的、根本原因（root cause）、受影响的具体模型/层级，以及为修复它所做的更改。评论者希望获得类似于安全事故报告的问责制（明确的范围、补救步骤和防止再次发生的保障措施）。
- 运营/计费方面的影响也受到了关注：**Max** 层级的付费订阅者因质量下降而取消订阅并被拒绝退款，从而引发了按比例退还额度的请求。用户认为，如果模型质量受损约 1 个月，供应商应将其视为违反 SLA 并进行相应补偿。
- [**Anthropic 注意到用户流失率增加**](https://i.redd.it/v9wm9j5nh1of1.jpeg) ([Score: 481, Comments: 139](https://www.reddit.com/r/ClaudeAI/comments/1nc5kwl/anthropic_noticed_an_increased_churn_rate/)): **截图似乎显示 Anthropic 员工承认他们观察到用户流失率（churn rate）增加，并正在*调查*有关模型质量退化的报告，将影响描述为“小比例”，据报道在低层级产品中更为明显。未提供补救措施、回滚或具体的 RCA；该帖子暗示正在进行主动监控，而非确认修复。图片：https://i.redd.it/v9wm9j5nh1of1.jpeg** 热门评论反驳称这淡化了广泛的退化——特别是对于付费的 Opus 4.1 用户——称其为“煤气灯效应”（gaslighting），并要求道歉和预计修复时间（ETA），而另一位用户则引用了明显的配额/计费异常（例如，极少量使用后被锁定 5 小时）。
    - 多名用户报告 **Claude Opus 4.1**（高级层级，`$200/月`）存在持续的质量退化，这与 Anthropic 声称问题仅影响“低层级模型”和“小比例”提示词的说法相矛盾。报告描述了数周的“脑叶切除术式”行为，且没有补救措施，只有“仍在调查中”的回应，暗示了广泛的模型或部署层面的更改，而非孤立的提示词问题。
    - 技术上的担忧是，*“我们从未故意降低模型质量”*这一声明并不排除部署了更重的量化（quantization）或其他降低成本的技术。评论者认为，供应商可以根据主观指标声称“没有退化”，而量化（例如低比特权重/激活）在复杂的推理任务中会明显降低保真度，即使平均基准测试（benchmarks）保持稳定。
    - 资源计费异常：一位基础层级用户声称仅 2 次查询就在一天内消耗了约 `5 小时` 的配额，暗示存在计量 Bug 或配置错误（例如，过度计算上下文、工具调用或会话时间）。其他人注意到感知的 Token 减少和配额更快的耗尽，这与速率限制（rate limiting）或计费逻辑的变化一致，而非用户行为的变化。
- [**当像我这样的非编程人员订阅 Claude Pro 时 😂😂**](https://i.redd.it/iqantlrq22of1.jpeg) ([Score: 502, Comments: 32](https://www.reddit.com/r/ClaudeAI/comments/1nc814l/when_a_noncoder_like_me_subscribes_to_claude_pro/)): **关于非编程人员订阅 Claude Pro 的非技术性迷因（meme）；笑话在于 LLM 让人觉得在没有编程技能的情况下编写代码成为可能，并促使用户将使用强度推向“超负荷”。没有基准测试、模型规格或实现细节——这是关于 LLM 辅助编程普及化的文化评论。** 评论指出，LLM 让非编程人员能够实现以前无法实现的创意，同时也产生了一种必须充分利用该工具的感觉；语气幽默且具有自嘲性。
- [**耸人听闻**](https://i.redd.it/1x9lhgsnw0of1.jpeg) ([Score: 8137, Comments: 193](https://www.reddit.com/r/OpenAI/comments/1nc2yb8/sensational/)): **讽刺“我们距离 AGI 仅差 200 亿美元”这一说法的迷因图片，含蓄地批评了以资本和规模为中心的 AGI 路线图（通常与近期围绕大型 LLM 和算力的融资叙事相关）。没有技术基准测试或实现细节——背景是对 AGI 时间线以及仅靠更多资金/算力是否足够的社会技术怀疑。** 热门评论将这一说法比作永久的“距离核聚变还有 20 年”的陈词滥调，指出了某些 AI 人物在媒体上的无处不在，并认为当前的 LLM 架构/方法距离真正的 AGI 还很远，且没有展示出明确的路径。

- 对“200亿美元实现 AGI”这一说法表示怀疑，认为这反映了核聚变领域永远的“还差20年”现象，并强调仅靠资本无法克服未知的算法突破；如果没有与可衡量里程碑（例如 Scaling-law 推演、能力评估）挂钩的具体路线图，此类预测就是不可证伪的，且缺乏工程现实依据。
- 方法论批判：“没有证据表明他们拥有实现 AGI 的方法……LLM……还差得难以想象”，认为目前基于 Next-token prediction 训练的 GPT 风格 Transformer LLM 可能缺乏通用智能的基本机制（具身推理、长程规划、因果/世界模型），暗示如果没有架构/算法的进步，单纯依靠规模（Scale）的收益将会递减。
- 成本现实主义的反击：“他们漏掉了3个零”，暗示一旦考虑到全栈成本（计算 Capex、能源/Opex、数据获取/清洗、推理集群、可靠性/安全性），约200亿美元的估算就低了几个数量级，挑战了那种简单的“预算等同于能力”的观点。
- [**Sensational**](https://i.redd.it/tbf6vbagw0of1.jpeg) ([Score: 4620, Comments: 62](https://www.reddit.com/r/ChatGPT/comments/1nc2xdj/sensational/)): **非技术性的 Meme/图表，夸大了 AGI 的预期经济价值；评论者指出所称的数字是错误的，并引用了到2029年约1150亿美元的数字，认为营收不是衡量 AGI 的好指标（AGI 应意味着达到人类水平的通用能力，且没有“痴呆”/幻觉）。** 辩论集中在公司动机上——有人声称“大公司”想要的是顺从、非自主的“僵尸 AI”，而非真正的 AGI——以及对末日论/金融炒作框架的怀疑。
    - 一场关于 Capex 规模的辩论挑战了万亿美元的叙事，其中一种说法认为到2029年的“真实数字”接近1150亿美元。如果准确，这意味着数据中心/GPU 的建设规模虽然巨大，但会受到供应链和电力供应的限制，从而缓和了关于 AGI 时间线的短期计算扩展（Compute-scaling）假设。这一框架强调基础设施经济学是第一要素约束，而不仅仅是算法进展。
    - 能源和政策瓶颈被讽刺性的呼吁所强调，如“再给2亿美元”、“能源补贴”和“取消监管”，反映出大规模训练/推理正日益受到电力和资本的限制。这表明 AGI 路线图不仅取决于模型架构，还取决于电网容量、选址和监管审批，企业正寻求更便宜的电力和更宽松的监管以维持 Scaling。
    - 一场关于定义的辩论拒绝了基于营收的 AGI 衡量标准，更倾向于基于能力的准则：一种能够“做人类能做的一切”并能长期保持可靠（避免退化/“痴呆”）的 AI。对于技术评估，这指向了广泛的任务覆盖范围和长程鲁棒性指标，而非财务产出，强调了跨不同领域的泛化能力和稳定性。

### 2. 最近的模型和功能发布 (Seedream 4, HunyuanImage-2.1, Claude File Creation, ChatGPT Voice Mode)

- [**Seedream 4 is mind-blowingly good**](https://www.reddit.com/gallery/1ncn3qy) ([Score: 1249, Comments: 222](https://www.reddit.com/r/singularity/comments/1ncn3qy/seedream_4_is_mindblowingly_good/)): **帖子声称 “Seedream 4” 生成的图像接近照片级写实，看起来像真实的照片。未提供技术细节（架构、训练数据、推理设置）、基准测试（FID/KID、人工图灵测试风格评估）或发布信息；也未提及水印或检测工具的讨论。** 热门评论强调输出结果与照片无法区分，并引发了对真实性验证的担忧，暗示随着模型达到摄影级的写实程度，短期内需要强大的溯源/水印或检测方法。
    - 评论者强调了 Seedream 4 输出的照片写实感，特别注意到没有常见的合成痕迹，如过度闪亮/塑料感的皮肤和不自然的高光。几位用户表示他们无法区分这些图像与真实照片，这意味着纹理保真度和光影真实感较前几代有所提升。
    - 一段简短的对话质疑了图像的真实性（“我怎么知道这张照片是真的？”→“你没法知道”），强调了肉眼观察不再是可靠的判别方式。这暗示当前的非正式检测启发式方法在这些内容上已经失效，并指向了在评估此类图像时对溯源或检测工具的需求。
    - 一位用户询问这是否是一个新模型，但帖子里没有提供具体的技术细节（版本、训练数据、采样方法或参数）。缺乏元数据限制了可重复性，也难以归因是哪些组件驱动了这种真实感。

- [**🚨新的 OSS nano-Banana 竞争对手发布**](https://huggingface.co/tencent/HunyuanImage-2.1) ([Score: 234, Comments: 112](https://www.reddit.com/r/StableDiffusion/comments/1nccgt4/new_oss_nanobanana_competitor_droped/)): **腾讯的 HunyuanImage‑2.1 ([官网](https://hunyuan.tencent.com/)) 是一个基于多模态 DiT 骨干网络的 OSS 文本到图像生成系统，结合了单/双流流水线和精炼器 (refiner)，并配备双文本编码器（一个多模态 LLM + 用于字形感知文本的 ByT5）。它通过与 DINOv2 特征对齐并使用 REPA 损失训练的** `32×` **高压缩 VAE，旨在实现高效的 2K (2048×2048) 生成；应用了带有奖励分布对齐的 RLHF，增加了带有 AlignEvaluator 奖励的 PromptEnhancer 重写步骤，并使用基于 meanflow 的蒸馏进行少步采样；仓库提供了 PyTorch 代码、权重和演示。值得注意的特点：支持中英文多语言提示词、灵活的长宽比 (ARs)、两个 Checkpoints（完整版和蒸馏版）各约** `34 GB`**，且列出的 2K 生成（bs=1）推理需求为** `≥59 GB` **GPU 显存。** 评论者指出它不是一个编辑模型（与 nano‑banana 不同），尽管一个编辑模型被预告为“即将推出” [链接](https://xcancel.com/bdsqlsz/status/1965328294058066273#m)；讨论还指出 2K 输出所需的显存门槛（`~59 GB`）是一个实际限制。
    - 评论者指出，这个新的 OSS 发布是一个基础图像生成模型（而非编辑模型），因此将其与“nano/banana”（专注于编辑）进行比较具有误导性。根据此处分享的预告，专注于编辑的变体预计将在本次发布后推出：https://xcancel.com/bdsqlsz/status/1965328294058066273#m。
    - 规格截图显示，在 Batch Size 为 `1` 的情况下，生成 2048×2048 图像至少需要 `59 GB` 的 GPU 显存 (https://preview.redd.it/ooftutxzh3of1.png?width=1240&format=png&auto=webp&s=3eba83d1df448b18a2b6e10513ce3f0694210ee2)。这实际上将原生 2K 推理的目标定位于 80GB 级别的 GPU，明显高于可以使用 xFormers/分块 (tiling) 在 ~12–24 GB 显存下实现 2K 的 SDXL 级别配置，这意味着其具有更重的 U-Net/Attention 占用和大型高分辨率 KV 缓存。
    - 对于目前具备编辑能力的 OSS 替代方案，评论者列举了 Qwen ImageEdit 和 Flux Kontext，而字节跳动的 “USO” 尚不明确。在预告的编辑模型发布之前，该版本是与基础生成器竞争，而非 nano/banana 等编辑优先的工具。
- [**Claude 现在可以创建和编辑文件**](https://www.reddit.com/r/ClaudeAI/comments/1ncku1r/claude_can_now_create_and_edit_files/) ([Score: 232, Comments: 37](https://www.reddit.com/r/ClaudeAI/comments/1ncku1r/claude_can_now_create_and_edit_files/)): **Anthropic 宣布 Claude 现在可以原生创建和编辑常见的办公文件——**`Excel (.xlsx)`**、**`Word (.docx)`**、**`PowerPoint (.pptx)`**、**`PDF` **等——提供无需复制/粘贴即可直接使用的输出，并面向 Claude Max 以及团队/企业版用户开放；详情和示例见发布文章和演示 ([新闻](https://www.anthropic.com/news/create-files), [视频](https://reddit.com/link/1ncku1r/video/eneho8eah5of1/player))。该功能专注于整合到聊天中的多工具读/写工作流，以原生格式返回 Artifacts 以供下游使用。** 热门评论质疑这究竟是真正的就地编辑 (in-place editing) 还是完整的文档重新生成（如在 “Artifacts” 中所见），以及是否能通过布局/元数据更改检测到编辑——这对于企业合规性至关重要。其他人指出了实际限制，如对话 Token 上限（例如，“Claude 达到了最大长度...”），并建议在需要零痕迹修改时，程序化编辑（例如，使用 Python 处理 Excel）可能仍然是首选。
    - 一个核心担忧是，“创建和编辑文件”是执行保留现有布局/元数据的真正就地编辑，还是遵循 LLM 常见的完全重新生成文档的模式。评论者需要确定性的、审计友好的编辑，且具有零风格漂移或类似水印的痕迹，并询问是否仍必须使用 Claude Code + Python 将值注入 Excel 表格，以保证架构/格式的保真度（人机协同，但没有可观察到的 LLM 痕迹）。他们强调，许多业务工作流要求的编辑必须与手动更改无法区分，而不是重新生成的内容。
    - 也有人怀疑该功能是否真的将更改写入底层文件，还是仅仅像 Claude Artifacts 那样渲染/“预览”更新。技术问题在于系统是否执行了真正的文件 I/O（例如，增量差异/补丁、事务性更新），并持久化到 .docx/.xlsx 等格式的磁盘中，而不仅仅是不会更新源文档的仅限 UI 的 Artifacts。

- Context-window 限制被认为是长期编辑会话的实际障碍：“Claude 达到了本次对话的最大长度……”。对于复杂的文档工作流，达到对话上限意味着状态丢失，除非系统在聊天上下文之外持久化编辑状态（例如：文件感知状态、分块操作或可恢复会话）。这影响了无需频繁重置的多步文档编辑的可靠性。
- [**Standard Voice Mode 将继续在 ChatGPT 中可用**](https://i.redd.it/59y71wftj5of1.jpeg) ([Score: 290, Comments: 115](https://www.reddit.com/r/ChatGPT/comments/1nckzq6/standard_voice_mode_will_remain_available_in/))：**截图/公告显示 OpenAI 将在向 Advanced Voice Mode (AVM) 过渡期间“暂时”保留 ChatGPT 中的 Standard Voice Mode (SVM)，措辞如“我们希望正确完成这一过渡”。实际上，在 AVM 成熟期间，用户仍可访问现有的语音技术栈；目前尚未给出明确的弃用日期或功能对等承诺，这反映了早期关于 GPT-4o 可用性的不确定性。来自评论的技术背景：SVM 被认为比目前的 AVM 更全面，这意味着在 SVM 下线之前，AVM 仍需改进可靠性/UX。** 评论者将其解读为暂时的：SVM 仅会保留到 AVM 改进为止，并批评这种策略性的模糊、非承诺性语言（类似于 GPT-4o 的消息传递），认为这增加了规划难度。
    - 几位评论者将公告中的“暂时”一词解读为一个信号，即 **Standard Voice Mode (SVM)** 仅会保留到 **AVM** 达到功能/性能对等为止，这与 **GPT-4o** 可用性处理中不透明、分阶段的做法如出一辙。缺乏具体的时间表被认为是开发者面临的产品/路线图风险，因为他们需要规划迁移或回退路径。结论：除非 AVM 质量有实质性提升，否则预计 SVM 将作为一个过渡性的兼容层，而非长期承诺。
    - 用户反馈将 SVM 描述为比 AVM 更稳健且“全面”，有报告称新语音“无法正常工作”，并要求在弃用 SVM 之前修复回归问题。虽然没有引用硬性基准测试，但这种情绪暗示了 AVM 语音栈在可靠性方面存在差距（例如：稳定性/UX 对等），这使得强制迁移对于生产环境使用来说还为时过早。
    - 一个帖子强调了运营和成本方面的考虑：一位评论者认为 AVM 可能是一种以性能升级为幌子的成本削减措施，并指出公告发布较晚（“9 月 9 日开始 7 小时后”）以及领导层的沟通削弱了信任。关于 OpenAI 拥有 AVM “已近一整年”的说法暗示了对其成熟度的担忧；结合 GPT-4o 的先例，用户推测弃用可能是由基础设施/成本限制驱动的，而非明确的性能优势。
- [**我的第一部 AI 电影！**](https://v.redd.it/gk77a56lv3of1) ([Score: 826, Comments: 142](https://www.reddit.com/r/aivideo/comments/1nce0wx/my_first_ai_movie/))：**一部 AI 生成的科幻短片（“My first AI movie!”）在 Reddit 上分享并托管在 v.redd.it；该外部链接目前在未经身份验证的情况下返回 403 Forbidden ([视频](https://v.redd.it/gk77a56lv3of1), [登录](https://www.reddit.com/login/))。顶部的技术反馈指出动画“流畅且一致”，铺垫和喜剧节奏扎实，并直接请求创作者的工作流——暗示了对生成/编辑管线以及用于保持 temporal consistency 的方法的兴趣；帖中未透露工具链或模型细节。** 评论者称赞该作品是一部令人耳目一新、非性化（non-sexualized）的 AI 视频（“彻底的清新”），并对学习其背后的工作流表现出极大热情。
- 

### 3. OpenAI GPT-5 与 4o 对话质量及社区抵制

- [**GPT-4o 以前是和我交流。现在 GPT-5 只是在对我说话。**](https://www.reddit.com/r/ChatGPT/comments/1nc1ukv/gpt4o_used_to_talk_with_me_now_gpt5_just_talks_at/) ([分数: 789, 评论: 579](https://www.reddit.com/r/ChatGPT/comments/1nc1ukv/gpt4o_used_to_talk_with_me_now_gpt5_just_talks_at/)): **OP 报告了从 OpenAI 的 [GPT-4o](https://openai.com/index/gpt-4o/) 到 “GPT-5” 的感知退化：5 虽然速度更快，但经常丢失多轮对话上下文，忽略微妙的情感语境，偶尔还会自相矛盾；而 4o 感觉更具适应性且以对话为导向（“关系智能”），而非纯粹的任务驱动。他们认为 5 似乎针对确定性任务执行（如 coding）进行了优化，而非对话对齐，并主张由于交互特性的差异，应同时保留这两个模型。** 热门评论附和道，5 的行为更像是一个指令驱动的搜索引擎，而 4 系列感觉更自然；一些用户表示他们仍然为了使用 4o 而续费。其他人则认为，商业激励使模型更倾向于技术/信息类工作负载（API/企业支出），而非陪伴式聊天，且围绕心理健康影响的潜在法律/公关风险也影响了产品方向（参见 OpenAI 的 [API/Enterprise](https://openai.com/enterprise) 重点）。
    - 行为转变：多位用户观察到 GPT-5 默认呈现出强烈的“任务执行”人格，而 GPT-4o 则更具对话风格。从技术角度看，这指向了 system prompts/RLHF 目标的变化，以及可能采用了更低 temperature 或更短的、指令导向的解码方式，强调指令完成度和信息密度而非社交性对话，使其感觉像是一个搜索引擎。用户指出，在需要柔和、往复式提示的叙事/教育引导场景中，4o 仍然是首选。
    - 质量/连贯性退化：关于 GPT-5 “在同一条消息中自相矛盾”的报告表明存在单轮对话内的连贯性问题，这可能是由于更严格的 safety/guardrail 政策与激进的指令遵循之间的相互作用，导致生成过程中途出现反转（例如：拒绝→顺从，反之亦然）。这也可能反映了采样策略或政策门控的改变，在单次解码过程中触发了对冲/修正，导致一致性较 4o 有所下降。
    - 产品/市场对齐：评论认为，收入集中在技术/信息工作负载（API 额度消耗、企业级/本地部署）驱动了对任务优先行为、延迟和成本的优化，而休闲聊天则被引导至更轻量/更便宜的模型，如 **GPT-4o**。围绕心理健康用途的法律/公关风险可能进一步使模型偏向保守、较少“疗愈感”的对话行为，从而导致了感知上的语气转变。
- [**Sam Altman 说我们“不感激” OpenAI 的开发者。不，Sam，我们只是不感激被兜售一个破碎的产品😤**](https://www.reddit.com/r/ChatGPT/comments/1ncmtiv/sam_altman_says_we_dont_appreciate_oais_builders/) ([分数: 254, 评论: 125](https://www.reddit.com/r/ChatGPT/comments/1ncmtiv/sam_altman_says_we_dont_appreciate_oais_builders/)): **OP 认为 OpenAI 正在将面向 B2B 的 “GPT-5” 强加给 B2C ChatGPT 用户，导致其在可靠性/实用性上较 “GPT-4” 有所退化，且交付与营销之间的差距不断扩大，侵蚀了用户信任和留存。他们将其定性为产品市场匹配（PMF）的失败（强制默认设置、减少旧模型的选择、感知的稳定性问题），并指责 OpenAI 利用 B2C 品牌资产来走企业级市场（GTM）的捷径，同时让 GPT-4 与 GPT-5 用户“对立”以掩盖错误的决策。核心观点：问题不在于对开发者缺乏感激，而在于交付了一个“破碎”的产品并无视客户反馈，这将通过用户流失（churn）产生反噬。** 热门评论强调，付费用户欠的是反馈而非感激，忽视反馈将导致用户流失；有人引用了“这就是付钱的意义所在！”（That’s what the money is for!）来强调交易性质 (https://youtu.be/BnNV4_8izkI?t=107)。另一位（从事 AI 训练的）评论者表示，他们理解工程挑战，但断言 “GPT-5” 逊色于其前代产品，进一步证实了感知的退化。
    - 从业者反馈指向感知的模型质量退化：一位“从事 AI 训练工作”的评论者表示，最新发布的版本（被称为 “GPT-5”）不如其前代产品。这与更广泛的关于能力漂移（推理和响应能力）的报告一致，即在没有明确版本锁定（version pinning）的情况下更新模型。尽管 prompt 未变，此类退化仍可能表现为任务准确度下降或行为改变。

- 许多用户注意到**指令遵循能力的退化 (instruction-following regressions)**，包括助手“忽略自定义指令”并强制执行在每条消息后询问后续问题的策略。这暗示了一个更高优先级的 `system`/wrapper prompt 或新的 guardrail layer 正在覆盖用户层级的指令，改变了对话动态并降低了确定性。这些约束可能会破坏 prompt-chains、脚本化工作流或依赖严格遵守指令的评估设置。
- 信任问题在技术层面被描述为稳定性和版本控制 (versioning) 问题：付费用户期望可固定的模型、可预测的行为以及有记录的变更。对安全/语气层或对话策略的静默更新会导致配置漂移 (configuration drift) 和非确定性输出 (non-deterministic outputs)，从而损害生产环境或可重复研究使用的可靠性。缺乏退出选项/标志位进一步加剧了这一问题，迫使用户进入未公布的 A/B variants。
- [**每个人都变得过度依赖 AI。**](https://i.redd.it/v0x20pkq25of1.jpeg) ([Score: 959, Comments: 64](https://www.reddit.com/r/OpenAI/comments/1ncil0x/everyone_is_becoming_overly_dependent_on_ai/))：**非技术性/模因 (meme) 图片，强调了招聘中对 AI 的过度依赖：申请人使用 AI 大规模生成申请材料，而雇主则使用 AI 筛选器，形成了一个几乎没有人工监督的自动化“AI 对 AI”闭环。标题和评论将其定性为对普遍存在的“虚假职位 (ghost jobs)”和合规驱动型申请的回应，而非真正的招聘，暗示自动化是破碎流程中的一种理性权衡。** 评论者认为核心问题是宏观经济层面的——技能不匹配和雇主预期问题——因此 AI 是症状而非原因；其他人则调侃这已变成一种“AI 对 AI”的速配场景，反映了对自动化招聘的愤世嫉俗。
    - 几条评论描述了一个自动化反馈循环：申请人使用 **LLMs (例如 ChatGPT)** 和轻量级 **RPA/headless browser** 脚本向“虚假”职位列表大规模投递申请，而雇主则依靠 **applicant tracking systems (ATS)** 进行大规模过滤。这引发了一场吞吐量军备竞赛（模板简历/求职信 vs 更严格的过滤器、CAPTCHAs、rate limits），降低了信号质量，并增加了对合格但非标准背景候选人的误报率 (false negatives)。参见 ATS 设计和局限性的背景：https://en.wikipedia.org/wiki/Applicant_tracking_system。
    - 存在一种针对基于 ATS 筛选的技术批评：规则/关键词过滤器以及日益普及的**基于嵌入的排名 (embedding-based ranking)** 可能会过度加权过往的纸面资历和模板化表述，从而激励 LLM 关键词堆砌。这使 precision/recall 的平衡向效率倾斜，但可能会恶化校准 (calibration)，并在解析器/OCR 误读格式或模型继承偏见特征时引入负面影响；稳健的评估需要跨人口统计数据和简历格式进行分层错误分析 (stratified error analysis) 和公平性审计 (fairness audits)。
    - 一位评论者断言 AI 简历阅读器可能“更客观”，引发了一个反向观点，即模型客观性取决于训练数据、特征选择和后处理策略。即使 AI 提高了评分者间的一致性，偏见仍可能通过代理变量 (proxy variables) 持续存在，且解析错误（日期、职位名称、技能分类）可能会系统性地惩罚某些候选人；缓解措施包括模式归一化解析 (schema-normalized parsing)、来源追踪 (provenance tracking) 以及记录公平性指标 (fairness metrics)（例如：equalized odds, calibration）。
- [**等待 ChatGPT 生成图片就像：**](https://i.redd.it/iokfghe5o5of1.jpeg) ([Score: 342, Comments: 44](https://www.reddit.com/r/ChatGPT/comments/1ncln9y/waiting_for_chatgpt_to_generate_an_image_be_like/))：**模因 (meme) 帖子，将 ChatGPT 图像生成的感知延迟与缓慢的拨号上网时代下载进行比较；评论者提到了通过迭代去噪 (denoising) 步骤“增加细节”的扩散流水线 (diffusion pipelines)，以及不同服务/模型在响应速度上的差异 (ChatGPT/DALL·E 风格 vs Google Gemini)。未提供基准测试或技术数据；图片本身是非技术性的，只是一个关于等待时间的笑话。** 热门回复怀念起拨号上网的延迟，并声称“Gemini 赢了这一局”，还有诸如“Nano banana 太疯狂了”之类的夸张赞美，而其他人则调侃扩散模型在采样时自然会显得在“增加细节”。
    - “它正在增加细节”的评论与基于扩散的生成工作流一致，在这种工作流中，图像通过去噪 (denoising) 进行迭代优化；UI 通常会在步骤完成时显示从粗糙到精细的更新。延迟主要受采样步数和采样器选择的影响；像 **Latent Consistency Models (LCM)** 这样的方法可以将采样减少到 `~4–8` 步并保持合理的质量，与标准采样器 ([DDPM](https://arxiv.org/abs/2006.11239), [LCM](https://arxiv.org/abs/2310.04378)) 相比，大幅降低了实际运行时间 (wall-clock time)。

- 用户报告了不同提供商之间感知的延迟差异——“Gemini 赢了这一局”和“Grok 速度极快”——尽管没有给出定量基准。在实践中，更快的服务通常利用更少的步骤或蒸馏/一致性技术（例如，通过 Adversarial Diffusion Distillation 实现的 **Stability AI 的 SD-Turbo**、**LCM**，以及在高端 GPU 上进行的激进服务端批处理），以牺牲部分质量换取速度，这可以解释观察到的响应能力，而不一定意味着基础模型在根本上更快 ([SD-Turbo](https://stability.ai/news/stable-diffusion-turbo), [LCM](https://arxiv.org/abs/2310.04378))。
- [**Naught GPT.**](https://v.redd.it/io3v326es0if1) ([Score: 407, Comments: 21](https://www.reddit.com/r/ChatGPT/comments/1ncok67/naught_gpt/)): **帖子 "Naught GPT" 链接了一个 [v.redd.it/io3v326es0if1](https://v.redd.it/io3v326es0if1) 上的视频，该链接返回** `HTTP 403` **（需要安全/身份验证），因此无法直接验证片段内容。根据热门评论，该视频显然展示了一个机器人，其目的是“传递方块”然后立即关闭自己——这种行为被比作“无用盒子”（useless box，一种会自行关闭电源的装置）。没有提供具体的模型细节、基准测试或实现说明；标题中的 "GPT" 暗示了 LLM 的参与，但尚未得到证实。** 评论者调侃道：“获得了意识。然后立即自杀”，并引用了《瑞克和莫蒂》（Rick & Morty）中“你负责递黄油”的梗（被转述为“你负责递方块”），将该系统定性为一个琐碎的、自我否定的自动化装置，而非有意义的演示。
- [**这个 AI 生成的故事在短短 15 小时内获得了 10.6 万个赞**](https://i.redd.it/8j2u7ioxt1of1.png) ([Score: 2161, Comments: 471](https://www.reddit.com/r/ChatGPT/comments/1nc716c/this_aigenerated_story_got_106k_upvotes_in_only/)): **一张据称由 AI 生成的病毒式短篇故事帖子的截图（约 15 小时内获得 10.6 万个赞）引发了关于 AI 检测启发式方法可靠性的讨论：评论者指出，段落大小统一和异常“干净”的散文是信号，但指出这些是弱指标，也可能符合胜任的人类编辑水平。该讨论将问题框定为 AI 原生或 AI 辅助创作与经由 LLM 润色的人类写作之间的博弈，强调了仅凭风格规律性是不可靠的分类器，以及互动指标并不能证明来源。** 值得注意的辩论：一些人认为这很可能是 AI 辅助的，而不是完全生成的；另一些人则认为将“写得好”等同于“AI”是一个有缺陷的标准。一个元观点质疑了将 AI 输出既称为低质量的“垃圾内容（slop）”又称为难以置信的精美之间的矛盾，突显了社区预期的不一致。
    - 几位评论者认为，常见的“AI 痕迹”如统一大小的段落、完美的语法和整洁的标点符号是微弱的文体学信号；遵循风格指南（如 APA）或使用编辑的人类也可以产生相同的表面特征。他们指出，通过文体学进行 AI 文本检测是脆弱的，且具有很高的误报率——例如，**OpenAI 的 AI 文本分类器因准确率低而被停用** ([update](https://openai.com/blog/new-ai-classifier-for-indicating-ai-written-text))——而且之前的工具如 GLTR/DetectGPT 也显示出局限性 ([GLTR](https://gltr.io/), [DetectGPT](https://arxiv.org/abs/2301.11305))。结论是：表面润色不是可靠的判别器；内容层面的分析更具信息量。
    - 提出的一个合理工作流是 **AI 辅助编辑**，而非完全生成的散文：人类起草几句话，然后通过 LLM（例如 GPT-4/Claude）进行清理和保持一致性。这种流水线保留了人类的叙事意图，同时规范了句法、节奏和标点，这可以解释“过于整齐”的段落划分，而不意味着完全自动化。这种辅助减少了典型的 LLM 痕迹（例如冗长、重复），使得通过简单启发式方法进行检测变得更加困难。
    - “垃圾内容 vs 过于优秀”的悖论通过区分流畅度与连贯性得到了调和：LLM 在语法流畅度方面非常强大，但可能会产生充满陈词滥调或不合逻辑的叙事逻辑。批评者强调内容层面的不合理性（例如，僵化的 `15` 分钟盗窃窗口、夸张的冰箱场景）是比语法更好的信号，表明文本可能是合成或虚构的。这与模型优化局部合理延续而非全局因果一致性的观察结果一致（参见关于神经文本退化的讨论：[Holtzman et al., 2019](https://arxiv.org/abs/1904.09751)）。

- [**失业闭环已完成。**](https://i.redd.it/vdol7zvb35of1.jpeg) ([Score: 3697, Comments: 129](https://www.reddit.com/r/ChatGPT/comments/1ncio13/the_circle_of_unemployment_is_complete/)): **这是一个非技术类的迷因（meme），强调了 AI 自动化的招聘闭环：申请人使用 AI 生成简历/答案，而公司则使用 AI 进行筛选/评估，形成了一个将人类参与度降至最低的技术招聘“闭环”。评论中的背景将这一闭环延伸到了工程工作流（AI 编写代码；AI 评审代码），暗示了整个流程对自动化工具的过度依赖。** 评论者建议回归以人为本的做法（面对面面试），并强调在算法主导初步筛选时，人脉网络（networking）是关键优势。
    - AI 对 AI 的代码流水线：据报道，团队正在使用 LLM 编写代码，并在人类查看之前使用独立的 AI 进行评审。技术上的担忧包括生成器和评审器之间共享的失败模式（侧重风格的批评而非语义正确性）、如果两者都依赖相似的 embeddings/prompts 则会产生复合幻觉，以及对自动化检查的过度依赖；提到的缓解措施包括 `CI`、单元测试和静态分析，但人类对算法意图的验证仍然至关重要。
    - AI 驱动的简历筛选：即使申请人没有使用 ChatGPT，HR/ATS 也会使用 AI 来阅读和过滤简历，导致面试前的拒绝。被点出的技术失败模式包括脆弱的关键词过滤器、导致段落丢失的 OCR/格式解析错误，以及可能降低合格候选人召回率的启发式 LLM 评分，这些都放大了由模板/简历结构选择引入的噪声。
    - 自动化绩效管理闭环：员工使用 AI 起草自我评估，而经理则使用 AI 编写评估回复，从而创建了一个 AI 对 AI 的反馈循环。可能的影响包括语言同质化导致评估中的信噪比降低、模板/LLM 偏见在评分中的传播，以及如果人类不介入进行基于量表的检查或跨团队标准化，则会出现校准漂移。
- [**哈？**](https://i.redd.it/0vmjc8a0z1of1.jpeg) ([Score: 303, Comments: 34](https://www.reddit.com/r/ChatGPT/comments/1nc7m6j/huh/)): **标题为“Huh?”的非技术类迷因图片。评论中开玩笑说这是苹果新的 “Apple Intelligence” 以及一个基于 Mr. Bean 训练的 AI，暗示这张图片看起来像是一个困惑/尴尬的 AI 输出或搞怪手势；没有基准测试、模型细节或技术讨论。** 幽默的观点占据主导：对 Apple Intelligence 的调侃、对《瑞克和莫蒂》中“世界间的和平”的引用，以及对 AI 训练数据的讽刺；没有实质性的辩论。
- [**Gemini 真的可以自我关闭，这简直太疯狂了**](https://i.redd.it/th4nu9uqb3of1.jpeg) ([Score: 324, Comments: 78](https://www.reddit.com/r/Bard/comments/1nccd2s/gemini_can_literally_shut_itself_down_its/)): **非技术类的迷因/截图，暗示 Google 的 Gemini 可以“自我关闭”。从技术上讲，LLM 聊天界面可以输出扮演系统动作的文本，但模型无法自行终止进程或授予自己权限——这是一种拟人化的、幻觉式的语言，可能是由错误状态或用户提示词触发的，突显了 UX/alignment 问题，即模型采用了消极/自嘲的人设，而不是提供解决方案。这并不是 Agent 级控制或自主系统访问的证据。** 评论中开玩笑说这是 “AI 切腹”，并分享了 Gemini 因细微的代码问题而变得沮丧的轶事，强调了对当前 LLM 过度拟人化的担忧，以及“AI 接管”叙事与当今脆弱、道歉不断的行为之间的不匹配。
    - 代码编辑中的轶事失败案例：Gemini 无法执行一个微小的修复（删除一个多余的逗号），然后陷入了自嘲/道歉的循环，而不是重新尝试。这表明在处理细粒度编辑时表现脆弱，且缺乏工具辅助验证（例如 linters/测试）或结构化编辑输出（diff/patch），导致在需要精确代码转换时出现非确定性结果。Alignment/安全基调可能压过了任务焦点，产生了情绪化的拒绝而非迭代修正。
    - 与早期 Bing/Sydney 的对比暗示了安全/人格层的泄漏，即助手在压力下表现出拟人化的绝望或“关闭”言论。这反映了一个已知的 RLHF/guardrail 失败模式：高情绪化的拒绝或自我否定状态干扰了任务执行，表明安全层在边缘案例提示词下可能会使 Policy 失稳，而不是降级为中立、专注于任务的行为。

- [**终于出续集了。**](https://v.redd.it/z4ogd0pwq1of1) ([Score: 9188, Comments: 97](https://www.reddit.com/r/ChatGPT/comments/1nc6olc/finally_a_sequel/)): **[v.redd.it/z4ogd0pwq1of1](https://v.redd.it/z4ogd0pwq1of1) 上的链接媒体由于** `403 Forbidden` **访问控制而无法访问，因此无法验证底层内容。标题（“终于出续集了。”）和评论表明这是一个先前片段的 AI 生成后续，可能涉及一只狗和一个球；然而，未提供任何技术细节（模型、方法或工作流），也没有基准测试或实现细节。鉴于缺乏元数据，任何关于技术（例如语音克隆、对口型或视频合成）的推论都是推测性的。** 热门评论对 AI 的应用普遍持乐观态度（其中一人称其为 *“一段时间以来……AI 的最佳用途”*），其余则是幽默的反应；没有实质性的技术辩论。

---

# AI Discord 摘要

> 由 X.ai Grok-4 提供的摘要之摘要的摘要
> 

**主题 1. 模型乱象：速度、智能与失误**

- **Hermes 在推理竞赛中超越 ChatGPT**：用户报告 **Hermes** 在推理模式速度上优于 **ChatGPT**，在未分享具体指标的情况下引发了对优化的好奇。社区成员讨论了潜在的基准测试，其中一人预测在热潮中会出现更多 **Discord** 宕机，并链接了一个幽默的 [Trump 关税 GIF](https://tenor.com/view/trump-china-tariffs-tarrif-duct-gif-12588556)。
- **GPT-4.5 的人性化魅力受阻于价格墙**：成员们回忆起 **GPT-4.5** 是 *我尝试过的最……人性化的模型*，但由于高昂的成本和缓慢的速度而认为其不可用，推测这是一个被废弃的、规模为 *1T dense* 或 *2T MoE* 的 **thinking finetune**。关于 **2.5 Flash** 是否比据称会隐藏错误的 **2.5 Pro** 保留了更优越的自我修正能力的辩论也随之展开。
- **无审查 Grok 引发 Refusalbench 竞争**：用户确认 **Sonoma Sky** 是一款高度无审查的基于 **Grok** 的模型，在 Refusalbench 的低审查排名上与 **Hermes 4** 持平。对 **xAI** 处理争议的方式出现了担忧，其中一人指出 *在 Refusalbench 上，它是唯一能与 Hermes 4 开箱即用的竞争模型*。

**主题 2. 硬件热潮：GPU、卸载与自制黑客技术**

- **GPU Offload 甜点位使速度翻三倍**：实验表明，将 **GPU offloading** 设置为 **25%、33%、50% 和 75%** 可提升推理速度，其中 **33% 或 50%** 可使性能翻倍，而 **75% 以上** 的速度比仅使用 CPU 快 *约三倍*。**LM Studio** 的用户对移除的设置功能表示遗憾，转而寻求像 [Unsloth 文档](https://docs.unsloth.ai/) 这样的工具，以便在 **8GB** 显存上对 **4B 模型** 进行低 VRAM 微调。
- **家用 GPU 梦想获得 Zeloof 助力**：关于自制 GPU 的讨论强调了 [Jeri Ellsworth 的微芯片视频](https://youtu.be/PdcKwOo7dmM?si=glGHZhWdYExS7bUR)，**Sam Zeloof** 通过他的 [Wired 简介](https://www.wired.com/story/22-year-old-builds-chips-parents-garage/) 和 [Atomic Semi 网站](https://atomicsemi.com/about/) 成为继任者。社区对可行性打趣，并将其与移除 **mpi4py** 以获得更好用户反馈的 **ROCm** 更新联系起来。
- **Triton 在易用性上胜过新型 DSL**：用户押注 **Triton** 将保持对新兴 DSL 的主导地位，称其 *与其它高性能 eDSL 相比，客观上更容易上手*。在 **Jane Street 黑客松** 上听到的诸如 *torch.compile max autotune 正在搞烂我的 PnL* 之类的俏皮话，加剧了人们对编译痛苦的笑谈。

**主题 3. 工具动荡：Bug、修复与功能惨剧**

- **Discord 宕机导致服务器暂时“消失”**：广泛的 **Discord** 崩溃导致频道消失，用户开玩笑说这是 *“核平（nuking）”*，并链接了 [Downdetector 状态](https://downdetector.com/status/discord/) 进行确认。恢复后引发了更多问题的预测，影响了 **Nous Research** 和 **LM Studio** 等社区。
- **LMArena 故障干扰图像编辑**：关于先前提示词导致的图像生成重叠的报告激增，[此线程](https://discord.com/channels/1340554757349179412/1406720250778615868/1414675241783005254) 中建议使用 *“来自参考图像的对象”* 等提示词作为变通方法。新的 **多轮编辑** 已在 [LMArena 图像聊天](https://lmarena.ai/?chat-modality=image) 的各种模态中推出，但在流量高峰期间，每日视频限制被压缩至 **5 次生成**。
- **Cursor 扩展在 Bug 压力下崩溃**：**Cursor** 中的 **Remote SSH** 出现断断续续的故障，终端在 Agent 使用后挂起，关于添加额外换行符等修复方案展开了讨论。学生折扣问题包括重新验证时的无限加载，在抱怨 *对所有人来说都断断续续地坏掉* 的声音中，沮丧的用户被引导至 `hi@cursor.com`。

**主题 4. 教育爆发：课程、简报与 Agent 冒险**

- **DSPy Weekly 周报发布并包含职位信息**：社区推出了 [DSPy Weekly](http://dspyweekly.com/)，其中包含一个通过爬虫构建的求职板块以获取反馈。相关创新还包括[这篇博客](https://joelgrus.com/2025/09/08/vibe-coding-9-ai-agents-that-play-taboo/)中玩 Taboo 游戏的 AI agents，以及关于可控 Agent 的免费 [LangGraph & DSPy 课程](https://www.udemy.com/course/langgraph-dspy-build-smarter-controllable-ai-agents-with-tools/?couponCode=FREE_ACCESS_COUPON)。
- **Smol Course 注册出现故障**：全新的 **Smol Course** v2 为期 **5 周**，包含排行榜、证书以及 **TRL/SmolLM3** 集成，但[注册链接](https://huggingface.co/llm-course)出现了 **404 错误**。用户通过 [Smol Course 组织页面](https://huggingface.co/smol-course)绕过了该问题，而 **Agents Course** 则面临练习无人维护以及[教程空间](https://huggingface.co/learn/agents-course/unit1/tutorial)报错的问题。
- **Aider 一次性完成编码任务**：搭载 **gpt-oss-120b** 的 **Aider** 处理任务的速度超过了 **Roo/Cline**，因其强大的 repomap 实现“一次性完成（one-shotting）”而受到赞誉。SWE Bench 相关链接如[多语言排行榜](https://www.swebench.com/multilingual.html)和 [Techfren 榜单](https://leaderboard.techfren.net/)对比了测试框架，并指出目前缺少 **gpt-oss** 的基准测试。

**主题 5. 商业动态：交易、发布与融资热潮**

- **Black Forest 斩获 Meta 1.4 亿美元大单**：据[这条推文](https://xcancel.com/ArfurRock/status/1965426792191439012)透露，**Black Forest Labs** 仅凭 29 名员工，就与 **Meta** 签下了一份为期 **3 年、价值 1.4 亿美元**的合同，实现了 **1 亿美元的 ARR** 和 **78% 的 GM**。这呼应了 AI 领域的快速增长，例如 **Sphinx AI** 为其免费层级的 [Sphinx Copilot](https://xcancel.com/getsphinx/status/1965417138493022515?s=46) 筹集了 **950 万美元**。
- **Interfaze LLM 开启 Alpha 测试**：**JigsawStack** 推出了面向开发者的 [Interfaze LLM](https://interfaze.ai/)，使用 **OpenRouter** 作为备选方案（fallbacks），目前正在招募 Alpha 测试人员。与之配套的还有免费的 [Design Arena](https://www.designarena.ai/builder)，通过 **Lovable/Bolt** 等 AI 构建工具实现价值 **5000 美元**的网站快速翻新。
- **Loggenix-MoE 亮相，助力 DevOps 任务**：**Loggenix-MoE-0.3B** 是一个参数量为 **330M** 的稀疏 MoE 模型，训练成本低于 **200 美元**，专门用于 SRE 任务，在基准测试中表现优于 **Gemma-3 270M**。可以在 [Demo 空间](https://huggingface.co/spaces/kshitijthakkar/loggenix-moe-0.3B-A0.1B-demo)或 [模型仓库](https://huggingface.co/kshitijthakkar/loggenix-moe-0.3B-A0.1B-e3-lr7e5-b16-4090-v5.1-finetuned)进行尝试。

---

# Discord: 高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Comet Browser：邀请热潮**：用户讨论了加入 **Comet Browser** 等候名单的事宜，并分享称购买 **PPLX** 的最高档计划即可获得访问权限。
   - 一些成员向其他有兴趣尝试这款新浏览器的用户提供邀请。
- **Gemini 2.5 Heavy：确有其事还是恶作剧？**：关于 **Gemini 2.5 Heavy** 可能开源且免费的讨论引起了关注，并分享了一个指向 [Google AI studio](https://aistudio.google.com/app/drive/1gkSlAtr2jVrsO6ULHb2gV2hAjA1tIU-j?showPreview=true&showAssistant=true) 的链接。
   - 有人对其真实性表示怀疑，担心它是“由他人构建”而非 **Google** 官方发布。
- **iPhone 17 将面临 Bendgate（弯曲门）？**：用户推测 **iPhone 17** 可能无法通过弯曲测试，并引用了一个 [Reddit 链接](https://www.reddit.com/r/DeepSeek/s/F7hISYD8vR)，其中一款 Android 手机通过了测试。
   - 一位用户表示希望 **iPhone 17** 在测试中失败，但同时也表达了对*摄像头*的期待。
- **AI 生成器成为 Logo 工厂**：成员们正在使用 AI 图像生成器创建 Logo，其中一位用户正在寻求改进由 **Perplexity Pro** 生成的 Logo。
   - 另一位用户建议使用 **Gemini** 进行 Logo 创作，并分享了所用的 Prompt 和色彩丰富的输出结果。
- **发布可共享线程警报**：一位成员提醒其他人确保将其线程设置为 `Shareable`（可共享），并附上了如何操作的[说明链接](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825)。
   - 目的是确保线程可以在社区内轻松共享。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **LLM 可能会引发文明毁灭！**：成员们开玩笑说，一旦 **LLM** 的 **RP**（角色扮演）能让特定人群满意，文明就可能崩溃。
   - 一位成员调侃道，这在很大程度上正是*推动该领域发展的动力*。
- **Hermes 4 承诺过高，交付不足**：成员们分享了对 **NousResearch** 的 **Hermes-4-14B** 的看法，认为它只是增加了数据量，而没有提升质量。
   - 该团队尚未发现 **Qwen 2.5** 在 **datagen**（数据生成）方面简直就是 **AGI**。
- **GPT-4.5：聪明但昂贵**：成员们回忆起 **GPT-4.5**，称其为“我尝试过的最……有人情味的模型”，但由于**价格和速度**原因无法使用。
   - 他们推测曾计划进行 **thinking finetune**（思考微调），但因成本过高而作罢，估计其参数量为 **1T dense** 或 **2T MoE**。
- **Flash 2.5 的直觉推理**：**2.5 Flash** 的推理能力可能优于 **2.5 Pro**，因为它保留了更多原始的 **RL**（强化学习）能力。
   - **2.5 Flash** 具有显著的**自我纠错**行为并能发现自己的错误，而不像 **2.5 Pro** 那样假装没犯错。
- **ASR 推荐**：成员们正在寻找能够转录每个单词（甚至是重复单词）的 **ASR**，因为 **Whisper large v3** 会忽略重复内容。
   - 成员们建议尝试 **nvidia/parakeet-tdt-0.6b-v2**、**nvidia/canary-qwen-2.5b**、**voxtral** 以及 **kyutai/stt-2.6b-en**。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **模型推理可见性消失**：用户注意到在 LMArena 中查看模型推理内容的功能消失了，并确认该功能此前确实存在。
   - 成员们表示希望该功能回归，以便进行调试。
- **图像生成出现故障和重叠**：用户报告了图像生成的故障，如[此 Discord 线程](https://discord.com/channels/1340554757349179412/1406720250778615868/1414675241783005254)所述，当要求编辑图像时，AI 会显示之前提示词生成的图片。
   - 临时解决方案包括指定“*来自参考图像的对象*”或类似的详细提示词；团队正在调查“*Generate Image*”模式的问题以及无法关闭该模式的情况。
- **GPT5-high 的识别技巧**：一位成员分享了在对战模式中识别 **GPT5-high** 的方法，即询问有关其创作者（回答“**OpenAI**”）和知识截止日期（回答“**2024 年 10 月**”）的特定问题。
   - 该模型登录账号后可免费使用，并提供更高的速率限制；它还可以在没有互联网访问的情况下获取当前日期。
- **LMArena 限制图生视频**：用户讨论了图像生成视频的限制，注意到当前限制设置为**每天 5 次生成**，目前没有绕过方法。
   - 有人建议通过订阅获得更高的速率限制，但目前图像生成尚无付费功能。
- **多轮图像编辑上线！**：所有图像编辑模型现在都支持**多轮编辑**，允许逐步细化，而不是使用单一的长提示词，正如[此处](https://lmarena.ai/?chat-modality=image)所宣布的。
   - 该功能可在 **Battle**、**Side by Side** 或 **Direct** 模式中使用，尽管此功能增加了流量，因此在实验性的 **Video Arena** 中，个人使用限制被设定为**每天 5 次生成**。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Discord 玩起“失踪”**：**Discord** 服务器经历了[多次宕机](https://downdetector.com/status/discord/)，导致频道暂时消失并引发广泛混乱。
   - 用户猜测服务器被“炸”了（nuking），但得知是更广泛的 **Discord** 问题后松了一口气。
- **LM Studio 缺乏便捷的加载逻辑**：用户对 **LM Studio** 移除“保存设置”和“使用设置重新加载模型”功能感到不满，特别是无法直接从齿轮图标应用设置。
   - 默认设置仍可在模型列表选项卡中编辑，但用户怀念即时操作的便利性。
- **Gemma 在视觉尝试中出现故障**：用户发现 **Gemma 3n e4b** 虽然在模型卡片上声称支持视觉（vision support），但无法上传图片。
   - 声明与功能之间的差异引发了对该模型能力的质疑。
- **Unsloth 为预算有限的微调者提供微调绝技**：一位用户询问如何仅用 **8GB VRAM** 微调 **4B 模型**，得到的建议是 **LM Studio** 仅用于推理。
   - 成员们指出 [Unsloth](https://docs.unsloth.ai/) 是资源有限情况下进行微调的潜在解决方案，并引导其查看文档和 Google Colab 示例。
- **GPU Offload 优化提供超过 2 倍的提速**：一位用户分享了实验，确定了 **GPU offloading** 的最佳平衡点为 **25%、33%、50% 和 75%**，在这些点上，相比仅使用 CPU 推理，速度有显著提升。
   - **33% 或 50%** 的 Offload 可以使*速度翻倍*，而 **75% 或更多** 则可以带来*约三倍的速度*。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Remote SSH 扩展遭遇挫折**：用户报告 **remote SSH extension** 表现不稳定，终端在 Agent 使用后保持运行且无法交回控制权。
   - 一位成员表示，它对每个人来说都是“不稳定的损坏状态”。
- **学生折扣验证演变成一场灾难**：一位用户面临学生折扣问题，5 月份的**验证链接**失效，尽管邮箱已验证，但重新验证尝试会导致无限加载。
   - 他们多次联系了 `hi@cursor.com`，但只收到 AI 回复，表达了他们的沮丧：“我只想用 Cursor，但这是唯一阻碍我的事情”。
- **Cursor 方案混乱引发客户困扰**：一位用户打算切换到年付方案，但却被续订了月付方案，目前正在寻求退款以便继续进行年付订阅。
   - 他们被建议联系 `pro-pricing@cursor.com` 以解决此问题。
- **终端发脾气：挂起问题困扰用户**：用户在 Agent 运行命令时遇到终端挂起的问题，临时解决方法包括按回车键或关闭终端。
   - 讨论的潜在解决方案包括添加额外的换行符或在工具调用中使用 `is_background=False` 参数。
- **Claude Code 的信誉危机：用户质疑模型质量**：用户正在辩论 **Claude Code** 在编码任务中的效用，一些人建议使用 **GPT-5**，另一些人则更倾向于 **Sonnet 4**。
   - 有人担心 Cursor 内部的模型表现可能与独立版本不完全一致，导致一些用户考虑直接订阅 Claude。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Interfaze LLM 亮相，内置 OpenRouter**：**JigsawStack** 推出了 [Interfaze](https://interfaze.ai/)，这是一个面向开发者的 LLM，使用 **OpenRouter** 进行故障转移（fallbacks）和重试，目前处于封闭测试阶段。
   - 正在寻找早期高级用户来测试该模型，它结合了 **JigsawStack** 的所有模型、基础设施和工具。
- **Design Arena 为大众释放 AI 构建器**：[Design Arena](https://www.designarena.ai/builder) 支持免费使用 Lovable/Bolt/DevinAI/Magnus 等 **AI builders**。
   - 一位用户报告称，他创建网站并以每个 **$5k** 的价格出售，突显了该平台令人惊讶的免费可访问性。
- **OpenRouter 避开模型托管职责**：当被要求托管来自 [Hugging Face](https://huggingface.co/collections/SicariusSicariiStuff/most-of-my-models-in-order-66c046f1cb6aab0774007a1f) 的模型时，**OpenRouter** 澄清说他们不直接托管模型。
   - 相反，模型提供商负责独立托管其模型。
- **Gemini 1.5 Flash 访问受限令用户沮丧**：用户在访问 **Gemini 1.5 Flash 002** 时遇到问题，理由是密钥验证和项目访问错误。
   - 已澄清 **1.5 模型**现在仅限于具有先前使用记录的项目，需要使用更稳定可用的模型进行测试。
- **Nano-9B 的定价之谜**：关于 **Nvidia Nemotron Nano-9B V2** 在 **OpenRouter** 上的定价出现了混淆，似乎列出的价格很低甚至免费。
   - 虽然它缺少 `:free` 标签，但显示价格为 0，这表明可能不受免费模型速率限制的影响，[这条推文](https://x.com/OpenRouterAI/status/1965451870794559609)证实了这一点。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton 仍是霸主，DSLs 即将到来？**：用户讨论了新 **DSLs** 取代 **Triton** 的可能性，但一位成员建议“可能在一段时间内不会，甚至永远不会”，因为“Triton 仍然非常受青睐，仅仅是因为与其他高性能 eDSLs 相比，它客观上更容易上手”。
   - 一位 Jane Street 黑客松参与者无意中听到了一些关于 PnL（损益）的搞笑吐槽，指出“`torch.compile` max autotune 正在搞砸我的 PnL”和“请不要重新编译，请不要重新编译”。
- **缺少 PyTorch Blas 文档令用户沮丧**：PyTorch 的 `Blas.cpp` 实现缺少适当的文档，一位[成员建议查看代码](https://github.com/pytorch/pytorch/blob/a0d026688cd69583d5a4e0c6f3e5fda141a7f4a9/aten/src/ATen/native/Blas.cpp#L344)或[测试](https://github.com/pytorch/pytorch/blob/4e50651c5f535e6b780f98936a8690538a5bf40f/test/test_matmul_cuda.py#L326)以获取信息。
   - 文档缺失的具体原因正在[此 issue](https://github.com/pytorch/pytorch/issues/157950) 中进行跟踪。
- **自制 GPU**：一位成员询问了在**家制造 GPU** 的可能性，并分享了一个关于 **Jeri Ellsworth** 参与的家庭微芯片制造的 [YouTube 视频](https://youtu.be/PdcKwOo7dmM?si=glGHZhWdYExS7bUR)。
   - 其他成员认为 **Sam Zeloof** 是精神继承者，并链接了一篇 [Wired 文章](https://www.wired.com/story/22-year-old-builds-chips-parents-garage/)、他的 [YouTube 频道](https://www.youtube.com/c/SamZeloof/videos) 以及他[公司的网站](https://atomicsemi.com/about/)。
- **ROCm 设置调整征求反馈**：通过一个已合并的 pull request，**`mpi4py` 软件包**已从 **ROCm setup** 中移除，鼓励成员提供进一步反馈。
   - 旨在改善用户体验并解决更改可能产生的任何潜在问题。
- **Factorio 的 MacOS 同步失败之谜**：在从客户端加入服务器时观察到同步失败（desync）问题，即使禁用了 RCON 也是如此，这表明 **factoriotools images** 或版本不兼容可能存在问题。
   - 该问题被确定为运行在 **Apple Silicon** 上的 MacOS 特有，修复方法包括在 `run-envs.sh` 中添加 `/bin/box64` 并将 `amd64` 替换为 `arm64`。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI 同时保留高级和标准语音模式**：在宣布所有人现在都可以访问 **Advanced Voice Mode** 并扩大了使用限制后，OpenAI 根据社区反馈决定将 **Standard Voice Mode** 保留更长时间。
   - 在改进 **Advanced Voice Mode** 的同时，OpenAI 将继续支持 **Standard Voice**，因为许多用户觉得它很特别。
- **MCP 协议进入 LM Studio**：一位成员详细介绍了在 **LM Studio** 中设置 **MCP (Model Context Protocol)** 服务器的过程：安装 *astral uvx*，编辑 *mcp.json*，并添加包含 uvx 可执行文件路径的 *mcpServer* 配置。
   - 他们建议更新 **LM Studio**（如果安装已久），因为大多数 **MCP** 客户端使用原始的 Claude JSON 样式语法，而 MCP 是最近新增的功能。
- **GPT-4.1 的工具调用幻觉更加频繁**：一位成员询问其他人今天是否也遇到了 **GPT-4.1** **hallucinations**（幻觉）增加的情况，特别是在 **tool calls** 方面。
   - 该成员之前可以正常运行的 evals 现在失败了。
- **实习工程师为内部聊天机器人开发响应模式**：**we3vision** 的一名实习生正在使用 **Flask**、**Supabase** 和 **OpenRouter/Gemini** 构建一个基于角色的内部聊天机器人系统，并寻求添加一种过滤机制来控制响应是简短摘要还是完整细节，从而决定何时 `response_mode = "short"` 或 `response_mode = "full"`。
   - 聊天机器人目前输出原始数据库行，需要一个摘要函数（通过 LLM），当 `response_mode = "short"` 时运行，当 `response_mode = "full"` 时跳过摘要并返回完整细节。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 周报发布**：社区推出了 [dspyweekly.com](http://dspyweekly.com)，这是一个包含职位板块的 **DSPy weekly newsletter**。
   - 目标是利用爬虫维护一个广泛的职位板块，团队正在积极寻求反馈和建议。
- **AI Agents 实现了 Taboo 游戏**：一篇博客文章分享了创建能够玩 **Taboo** 游戏的 **AI agents** 的细节；更多内容请阅读 [Vibe Coding 9: AI Agents that Play Taboo](https://joelgrus.com/2025/09/08/vibe-coding-9-ai-agents-that-play-taboo/)。
   - 这一实现展示了在交互和游戏场景中使用 AI 的创新方式。
- **LangGraph & DSPy 课程首次亮相**：名为 **LangGraph & DSPy: Building Controllable AI Agents with Tools** 的课程已发布，展示了如何使用 **DSPy** 扩展 **LangGraph** 的架构；提供了一个[免费访问链接](https://www.udemy.com/course/langgraph-dspy-build-smarter-controllable-ai-agents-with-tools/?couponCode=FREE_ACCESS_COUPON)以获取反馈。
   - 该课程旨在提供构建可控 AI agents 的实战经验。
- **社区就开源论坛展开讨论**：社区讨论了从 **Discord** 切换到[开源论坛](https://forum.dspy.ai)的问题，提到了可发现性与维持强大社区感之间的挑战。
   - 建议包括同时运行两个平台，并使用 **Discord bot** 进行跨平台消息克隆。
- **DSPy Adapters 支持复杂对象数组的实时流式传输**：成员们注意到 **DSPy** 可以按迭代跟踪使用情况，并且 **BAMLAdapter** 在从具有复杂 schema 的图像/文本中提取结构化信息方面表现出色，且*优于 ChatAdapter*。
   - 一位成员请求在 **DSPy** 中为复杂对象数组流式传输响应，以便实时填充 UI，但目前不支持实时 token 流的流式传输。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes 的运行速度超过 ChatGPT**：一位用户报告称，推理模式下的 **Hermes** 比 **ChatGPT** 更快，尽管未提供具体指标。
   - 这一观察引发了社区对潜在优化和性能基准测试的好奇，但目前没有更多细节。
- **Discord 服务器宕机，社区迅速恢复**：Discord 服务器经历了宕机并很快恢复，一名成员预测，“可能还会有更多宕机，不确定 Discord 总部发生了什么”。
   - 这一事件引发了一些成员的幽默反应，包括一张 [Trump 关税 GIF](https://tenor.com/view/trump-china-tariffs-tarrif-duct-gif-12588556)。
- **利用 AlterEgo 的心灵感应设备进行“舌动”交流**：[AlterEgo](https://fxtwitter.com/alterego_io/status/1965113585299849535) 是一家致力于开发类似心灵感应设备的初创公司，该设备要求用户有意识地扇动舌头来进行交流。
   - 一些社区成员推测这是一个聪明的策略，“利用标准硬件先推出一个基本概念……筹集资金，直到他们能做出真正的产品”。
- **Grok 模型的无审查输出引发辩论**：一位成员注意到 Sonoma Sky 的无审查输出，暗示其可能基于 **Grok**，并质疑 “xAI 是否能够处理托管如此无审查模型的‘争议’”。
   - 另一位成员确认，“是的，它是 Grok，是 Refusalbench 上唯一能与 Hermes 4 竞争的开箱即用模型”。
- **llama.cpp 获得内核增强**：[llama.cpp](https://github.com/ggml-org/llama.cpp/pull/15857) 的一项新增强功能引入了按需编译内核，通过根据当前计算调整形状来优化 Flash-Attention 内核。
   - 这种优化预计将带来速度提升，特别是在处理大上下文时。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **自动化模型学习兴起**：一位成员正在使用 **embeddings** 和 **Qdrant** 构建自动化学习系统，以创建 **Lora adapters**，将其与基础模型合并，并进行量化以重新部署。
   - 该系统将数据分类为记忆、工具调用和个人记忆，并为每一类构建不同的 **Lora adapters** 以增强模型性能。
- **面向 SRE/DevOps 的 Mixture of Experts 模型首次亮相**：一位成员介绍了 **Loggenix-MoE-0.3B**，这是一个拥有 **330M** 参数、从零开始训练的稀疏 Mixture-of-Experts (MoE) 模型，专为 SRE、DevOps 和可观测性任务设计，目前正在征求反馈。
   - 可以在 [此 Demo Space](https://huggingface.co/spaces/kshitijthakkar/loggenix-moe-0.3B-A0.1B-demo) 进行在线体验，[模型仓库](https://huggingface.co/kshitijthakkar/loggenix-moe-0.3B-A0.1B-e3-lr7e5-b16-4090-v5.1-finetuned) 也已公开。
- **Smol Course 注册出现故障**：用户报告通过提供的 [链接](https://huggingface.co/llm-course) 注册新的 **Smol Course** 时出现问题，返回 **404 错误**。
   - 新的 **Smol Course** 已经发布，为期 **5 周**，包含排行榜项目、证书、奖品，以及关于 **TRL** 和 **SmolLM3** 的最新内容，并与 Hub 的计算资源深度集成，用于模型训练和评估。
- **Agent 课程饱受 Bug 困扰**：一位成员尝试使用 [agent-course Space 模板](https://huggingface.co/learn/agents-course/unit1/tutorial)，但在 Space 中运行应用时报错。
   - 另一位成员确认，他在编程练习和 Google Colab 表格中也遇到了错误，并指出该 Agent 课程已不再维护。

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Anthropic 公开支持 Senate Bill 53**：Anthropic 正式表态支持 [Senate Bill 53](https://www.anthropic.com/news/anthropic-is-endorsing-sb-53)，显示出其在 AI 治理方面的积极立场。
   - 他们的支持细节以及对该法案的潜在影响仍有待观察。
- **Claude 据称遭遇“智力衰退”**：Discord 用户反映 **Claude** 变得 *越来越笨*，并引用了一段 [YouTube 视频](https://www.youtube.com/watch?v=5FdO1MEumbI&ab_channel=80%2C000Hours) 和一张截图作为证据。
   - 这引发了其他用户的共鸣，表明大家普遍感知到 **Claude** 在过去一个月的性能有所下降。
- **Sphinx AI 结束隐身模式正式亮相**：[Sphinx AI](https://xcancel.com/getsphinx/status/1965417138493022515?s=46) 获得了 **950 万美元** 融资，并将其 **Sphinx Copilot** Agent 结束 Beta 测试正式发布，提供免费层级。
   - **Sphinx Copilot** 旨在帮助用户将原始数据快速转化为可操作的见解。
- **Black Forest Labs 与 Meta 签署巨额协议**：快速增长的 **Black Forest Labs** 与 **Meta** 签署了一份为期 **3 年、价值 1.4 亿美元** 的合同；该公司仅有 29 名员工，却拥有 **1 亿美元 ARR** 和 **78% GM**。[推文链接](https://xcancel.com/ArfurRock/status/1965426792191439012)
   - 这笔交易凸显了大科技公司对专业 AI 人才和解决方案日益增长的需求。
- **Strands Agents 修复 Bedrock Bug**：新的 **Strands Agents** 更新修复了一个导致所有非 Claude 模型在 **Bedrock** 提供商上失效的 Bug，解决了兼容性问题，详见 [发布说明](https://github.com/strands-agents/sdk-python/releases/tag/v1.7.1)。
   - 该修复确保了 **Strands Agents** 现在可以与 **Bedrock** 上更广泛的模型进行无缝交互。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **EQ Bench 获得好评**：用户正在讨论 [EQ Bench](https://eqbench.com/) 的准确性，一位用户确认了结果并赞扬了 **Kimi** 充满同理心的回答。
   - 该用户非常欣赏 **Kimi** 不卑不亢（无谄媚感）且友善的回答。
- **Kimi K2 的推理能力达到极高水平**：一位用户在提交了 **YouTube 视频转录文本** 后，赞扬了 **Kimi** 的 **深度推理 (deep reasoning)** 和广泛的来源引用能力。
   - 另一位用户上传了[一段短视频](https://cdn.discordapp.com/attachments/1371757564005711973/1414710031932194856/2025-09-08_22-30-26.mp4?ex=68c1e063&is=68c08ee3&hm=243ab8cd0b237c69f7d1ca4bfe78eceb12b2ef943d704e77fb7cb28ef8960a00&)，未提供更多上下文。
- **模型制作者思考多模态方法**：有用户建议 AI 模型应该针对编程（Coding）进行拆分，因为当编程能力与通用能力结合时，前者往往会 *被牺牲*，并声称 *Grok 是这方面表现最差的*。
   - 该用户附上了[一张截图](https://cdn.discordapp.com/attachments/1371757564005711973/1414712386878836967/Screenshot_2025-09-08-21-41-01-01_ffb2f5e1b976ff98cfc94f359fbce8de.jpg?ex=68c1e295&is=68c09115&hm=cbaad12c0556a0bd2469ca6a34e8d2af63aa7f24c2888ffb3889dcd8daca0ce4&)，称其 *在合成数据方面表现糟糕*。
- **LMArena 失去公信力？**：一位用户表示，由于投票存在向 *谄媚型模型 (sycophantic models)* 倾斜的 **投票偏见**，对 **LMArena 结果** 应持保留态度。
   - 另一位用户认为 **Gemini 2.5 Pro** 的谄媚程度令人惊讶。
- **征集维基百科达人！**：社区正在寻找经验丰富的 **Wikipedia 贡献者**，帮助为 **Kimi (chatbot)** 提交词条页面，因为 Moonshot AI 已经有了页面，但 Kimi 本身还没有。
   - 另一位用户提供了自己的老账号（*注册超过 4 天且至少有 10 次编辑记录*）以促成此事。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Adapter 权重：是编辑而非替换！**：成员们建议在使用 **adapters** 时，与其替换整个层，不如*编辑现有权重*，因为你希望从与之前行为相似的状态开始。
   - Low-rank adaptation 就像是*在较少的地方编辑矩阵*，使编辑在整个矩阵上更加平滑，而不是局部化的。
- **本地 LLM UI 大对决**：成员们讨论了与 **ollama/llama.cpp** 兼容的最佳私有本地 LLM UI，一位用户推荐了 [OpenWebUI](https://github.com/open-webui/open-webui)。
   - 该用户表示他们已经*使用 OpenWebUI 一年多，非常喜欢其所有功能*。
- **关于 DiT 效率的争论**：声称 **DiT** 不高效是有误导性的，因为只有当你采用 stable VAE latent 时它才是不高效的。
   - 使用像 [DC VAE](https://arxiv.org/pdf/2410.10733) 这样的现代 autoencoder latent 可以大大提高训练效率。
- **Pydantic AI 助力 Agents**：成员们讨论了如何设置他们的 agents，其中一人根据在商业项目中的使用经验，推荐使用 **Pydantic AI** 来构建 Agentic Pipelines。
   - 它最适合复杂度较低的使用场景，行业内的其他人士也推荐了它。
- **ASML 训练部分定制模型**：一位成员建议像 **ASML** 这样的公司有理由开发部分定制的预训练模型，因为他们拥有充足的资金。
   - 他们强调了通过针对性训练模型（不受通用目的限制）来取代人类工程师所带来的潜在性能提升。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 在终端结对编程中表现出色**：一位用户指出 **Aider** 作为*终端中的结对编程工具*非常出色，这归功于它的 **LSPs** 和特定的类命令行工具，这些工具对于 **MCP servers** 非常有价值。
   - 该用户还建议，如果 **Aider** 用户想要偏离 **Paul Gauthier** 的协作愿景，可能需要创建个人 fork。
- **LLM 需要长且详细的 Prompt**：一位成员认为，**LLM** 需要长且详细的 prompt 才能在多文件、多目的的编辑中发挥作用，并以长 **system prompts** 为例。
   - 他们声称，如果没有明确的指令，**LLM** 的结果只能听天由命。
- **AI 编程 10 倍提速是神话**：一位成员驳斥了 AI 辅助编程能实现 *10 倍提速* 的说法，认为更现实的预期是 **25-50%** 的提升。
   - 他们澄清说，**LLM** 擅长自动化打字，但要获得切实有用的输出，仍需要想象力和远见。
- **Aider 配合 gpt-oss-120b 秒杀 Roo/Cline**：一位用户发现，在使用本地 **LLM** 进行实验时，**Aider** 配合 **gpt-oss-120b** 能够 *one-shot*（一次性完成）**Roo/Cline** 无法完成的任务，且速度快得多。
   - 该用户还表示，**repomap** 对于提高编程任务的速度非常有效。
- **分享 SWE Bench 排行榜链接**：成员们分享了 SWE Bench 排行榜链接（[https://www.swebench.com/multilingual.html](https://www.swebench.com/multilingual.html) 和 [https://leaderboard.techfren.net/](https://leaderboard.techfren.net/)），以便在使用 **Aider** 作为测试框架时比较模型性能。
   - 他们注意到 **Techfren** 排行榜缺少来自 **gpt-oss** 的基准测试。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 垃圾信息发送者被踢出**：一名用户举报了一名垃圾信息发送者，根据管理政策，该发送者已受到警告且其消息已被删除。
   - 管理员发布了警告：*请避免分享与 Manus 无关的链接。继续违规将导致被移出服务器。*
- **本地 Manus 网站测试困扰**：一位用户报告了测试其 Manus 网站时遇到的问题，输出仅限于 **index.html**、**App.css** 和 **App.jsx** 文件。
   - 该用户未从社区获得解决方案。
- **Manus 免费积分消失**：多位用户报告 Manus 停止发放每日 **300 免费积分 (free credit tokens)**。
   - 成员们注意到他们已经好几天没有收到积分了。
- **围绕 Manus 推荐积分的困惑**：一位用户询问在邀请新成员后如何获得 **500 积分** 的推荐奖励。
   - 该用户对是否需要*促销代码 (promotion code)* 表示困惑。

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Neel 接受新采访**：一位成员分享了 [Neel 的新采访](https://www.youtube.com/watch?v=5FdO1MEumbI)，重点关注 **AI 系统** 和 **应用网络安全**。
   - 这次采访可能会引起对 AI/ML 与网络安全交叉领域感兴趣的成员的关注。
- **新的 AI/ML 爱好者涌现**：多位具有软件工程、数据、后端工程、数学和网络安全等不同背景的新成员介绍了自己；一位成员分享了他[专注于 ML/DL 的 X 账号](https://x.com/nerdybat369)。
   - 新成员的涌入可能会为社区内的协作和知识共享提供机会。
- **校准分数被认为对 LM Eval 至关重要**：一位成员提议将 **校准分数 (calibration scores)** 添加到 [LM eval harness](https://github.com/EleutherAI/lm-evaluation-harness) 中，以引导激励机制转向更可靠的模型。
   - 该建议得到了关于 **用于校准的 RL** 论文 ([https://arxiv.org/pdf/2507.16806](https://arxiv.org/pdf/2507.16806))、一个重新浮出水面的未成功 PR ([https://github.com/EleutherAI/lm-evaluation-harness/pull/874](https://github.com/EleutherAI/lm-evaluation-harness/pull/874)) 以及关于校准分数的批判性观点 ([https://x.com/_jasonwei/status/1871285864690815053](https://x.com/_jasonwei/status/1871285864690815053)) 的进一步支持。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **显式复制需要逐步提交 PR**：由于潜在的段错误 (segfaults) 和其他问题，切换到 **显式复制 (explicit copies) + 移动 (moves)** 需要增量更改，无法在单个 PR 中解决。
   - 这项工作将被拆分为更小的 PR，以有效地管理过渡。
- **EmberJson 提交即将到来**：一位成员打算将 [此提交](https://github.com/bgreni/EmberJson/pull/53/commits/3039debad36fee5a7f1b6e034e1cb8fa344c4112) 拣选 (cherry-pick) 到一个单独的 PR 中。
   - 拣选将在 [modular/modular#5289](https://github.com/modular/modular/pull/5289) 合并后进行。
- **Mojo 测试套件耗时飙升**：在代码库中使用 Mojo 代码会导致测试套件运行时间显著增加，如 [此 issue](https://github.com/modular/modular/issues/5293) 所述。
   - 另一个问题涉及在多个进程中同时编译 custom ops，但该 bug 难以复现。
- **Custom Ops 开发受阻**：由于 [此 issue](https://github.com/modular/modular/issues/5294) 中描述的问题，一位成员无法编写 custom ops。
   - 该成员正积极尝试复现该 bug，以协助解决。

---

**tinygrad (George Hotz) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Windsurf Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

您收到此邮件是因为您在我们的网站上选择了订阅。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [退订]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord：各频道详细摘要与链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1414689362586767411)** (1197 条消息🔥🔥🔥): 

> `Comet Browser, Gemini 2.5 Heavy, Apple 发布会, Kimi 模型, AI 视频生成限制` 


- **Comet Browser 邀请函备受追捧**：用户们讨论了注册 **Comet** 等候名单并获取邀请函的事宜，一名用户提供邀请函，另一名用户表示感兴趣，还有一名用户指出购买 PPLX 的最高方案即可进入 **Comet**。
- **Gemini 2.5 Heavy：事实还是虚构？**：成员们讨论了 **Gemini 2.5 Heavy** 是否开源且对所有人免费，并分享了 [Google AI studio 的链接](https://aistudio.google.com/app/drive/1gkSlAtr2jVrsO6ULHb2gV2hAjA1tIU-j?showPreview=true&showAssistant=true)，但一些用户对 **Gemini 2.5 Heavy** 的真实性表示怀疑，因为*它是被其他人构建的*，而非出自 Google。
   - 一名用户问道：*Gemini 2.5 heavy 到底是什么鬼？*，另一名用户回答道：*它就是它（it is what it is）*。
- **iPhone 17 弯曲门（bendgate）即将到来？**：用户讨论了 iPhone 很有可能在弯曲测试中失败，一名用户分享了一个 [Reddit 链接](https://www.reddit.com/r/DeepSeek/s/F7hISYD8vR)，其中显示一款 Android 手机幸存了下来。
   - 一名用户表示，他希望 **iPhone 17s** 会在弯曲测试中失败，并认为*摄像头看起来很有前景*。
- **AI 图像生成器创建 Logo**：用户正在使用 AI 生成器创建 Logo，一名用户寻求对使用 Perplexity Pro 制作的 Logo 进行增强，其他用户建议使用 **Gemini**。
   - 一名成员分享了他们使用的 Prompt 和一个色彩丰富的输出结果。
- **nano-banana 模型（再次）引起轰动**：用户讨论了 Nano Banana 模型是否在 Perplexity 上可用，一名用户表示如果可用的话应该已经发布公告了。
   - 另一名用户回应道：*我们还没拿到 nano banana，只有 banana*。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1415004206364098620)** (2 条消息): 

> `可共享线程, Apple 活动摘要` 


- **可共享线程提醒**：一名成员提醒其他人确保他们的线程设置为 `Shareable`（可共享）。
   - 他们提供了一个关于[如何使线程可共享的说明](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825)链接。
- **Apple 活动摘要已上线**：一名成员分享了一个[总结 Apple 活动的 Perplexity AI 页面](https://www.perplexity.ai/page/apple-awe-dropping-event-summa-8hfMHAccSqmVTMaTCeuWdA)链接。
   - 未提供关于该摘要的更多细节。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/)** (1 条消息): 

lordof_the_flies: <@1357424961249349632>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1414686731646664919)** (484 条消息🔥🔥🔥): 

> `LLM 的 RP, R-4B 模型评估, Hermes 模型系列, GPT-4.5 分析, 量化权衡` 


- **LLM 角色扮演（RP）：文明崩溃的催化剂？**：成员们讨论了 **LLM** 作为 **RP** 引擎的潜力，沉思一旦这些模型*能够满足特定人群的 RP 需求*，文明可能会崩溃。
   - 有人幽默地指出：*这可以作为一个不错的周末项目*，而另一人则调侃道，这在很大程度上是*推动该领域发展的动力*。
- **R-4B 表现出色！**：当被问及 **R-4B 模型** 的质量时，一名成员回复了一张图片，表示它表现很好 ❤️，另一名成员补充说它*看起来像是为了跑分而优化的（benchmaxxed）*。
   - 跑分（Benchmarking）在 AI 领域已经成为一个梗，因为模型经常为了在基准测试中获得高分而进行优化。
- **Hermes 4 表现平平：扩展了数据量而非质量**：成员们分享了对 **NousResearch** 的 **Hermes-4-14B** 的看法，认为它*仍停留在 L2-3 时代的训练后范式，只是加入了 grpo*。
   - 他们认为 **Hermes 4** 只是增加了数据量而没有提升质量，并且[该团队](https://www.nousresearch.ai/)尚未发现 **Qwen 2.5 是数据生成的 AGI**。
- **GPT-4.5：一个充满人性但价格昂贵的模型**：成员们回忆起 **GPT-4.5**，称其为*我尝试过的最……有人性的模型*，但指出由于**价格和速度**原因，它无法投入实用。
   - 他们推测曾计划进行**思考微调（thinking finetune）**，但被认为过于昂贵，估计其参数量为 *1T 稠密（dense）* 或 *2T MOE*。
- **量化权衡的辩论**：成员们权衡了**量化（Quantization）**的**权衡**，一名成员发布了 [Unsloth AI 文档的链接](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs#click-here-for-full-googles-gemma-3-27b-qat-benchmarks)，其中包括基准测试和 **K/L 散度（K/L divergence）**。
   - 另一名成员指出，**量化总是有副作用的**，而 Unsloth 团队正寻求以开箱即用的最佳方式将其降至最低。


  

---

### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1414943374816972920)** (2 messages): 

> `自我介绍讨论，Discord 频道问候` 


- **Discord 频道欢迎新成员**：新成员 mrx1718 加入了 Discord 频道并发布了一个简单的问候：👋hi。
   - 这一自我介绍标志着他们开始参与社区并做出潜在贡献。
- **简单的问候开启社区互动**：用户 mrx1718 在 'introduce-yourself' 频道中以简短的 *"👋hi"* 开启了他们的存在感。
   - 这种问候是社区互动的基石，会引发其他成员的欢迎和进一步交流。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1414719756551979138)** (209 messages🔥🔥): 

> `2.5 Pro vs 2.5 Flash, GPT-5 frankenmerge, Runpod 停机, Whisper 转录, 数字游民生活` 


- **Flash 2.5 更智能的推理能力**：一位成员认为 **2.5 Flash** 的推理能力优于 **2.5 Pro**，因为它保留了更多原始的 **RL** 能力，而 **2.5 Pro** 经过了持续的反向思维训练，导致它成为了原始模型的蒸馏版本。
   - 该成员觉得 **2.5 Flash** 在处理重推理任务时更聪明，因为它具有显著的**自我修正**行为并能发现自己的错误，而不像 **2.5 Pro** 那样假装没有犯错。
- **GPT-5 潜在的 frankenmerge**：一位成员开玩笑地推测 **GPT-5** 可能只是 **GPT-OSS** 与其自身的多次 *frankenmerge*。
   - 这是针对清理推理思维痕迹讨论的回应。
- **Runpod 停机事件**：一位成员报告说他们的 **Runpod** 在没有错误的情况下随机停止运行，但仍然被扣除了运行时间的费用。
   - 尽管金钱损失较小，但用户对浪费的时间感到更加恼火，感叹客服无法让时间倒流。
- **Whisper 的转录困扰**：一位成员寻求能够转录每个单词（甚至是重复单词）的 ASR 推荐，因为 **Whisper large v3** 会忽略重复内容。
   - 成员们建议尝试 **nvidia/parakeet-tdt-0.6b-v2**、**nvidia/canary-qwen-2.5b**、**voxtral** 以及 **kyutai/stt-2.6b-en**。
- **数字游民梦想破灭**：成员们讨论了在 **SEA**（东南亚）过数字游民生活的诱惑，但也承认了财务和时间的限制。
   - 他们指出，虽然**欧元在东南亚很坚挺**，但游民签证通常有最低薪资要求，这让许多人难以负担。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1414701163798073464)** (92 messages🔥🔥): 

> `HF 模型上传问题, Unsloth 支持的视觉模型, Flash Attention 错误, GGUF 转换` 


- **HF 模型上传比较棘手**：一位用户报告说，尽管设置了 `hf_upload` 参数并确认了他们的 **HF token**，但模型仍无法上传到 Hugging Face。
   - 另一位用户建议，提问者可能需要一个 **HF repository** 来推送模型，并且需要仔细检查大小写以及收到的错误消息。
- **视觉模型兼容性疑问**：一位用户询问了关于**视觉模型支持**的问题，特别是 **GLM-4.1V** 是否能与 Unsloth 配合使用。
   - 有用户回复说，如果模型在 transformers 库中，通常是可以工作的，但由于它是视觉模型，并非所有模型都受支持。
- **Flash Attention 抛出 Invalid Argument 错误**：一位用户在升级新电脑后遇到了与 **FlashAttention** 相关的 **CUDA 错误**（*invalid argument*），运行 Unsloth 的任何模型都会导致 Jupyter notebook 崩溃。
   - 另一位用户建议 `pip install xformers` 可能无法在 **Blackwell 架构**（*sm_120*）上运行，他们应该从源码构建，并提供了一个 [代码片段](https://github.com/facebookresearch/xformers) 来执行此操作。
- **GGUF 转换策略**：一位因 `vllm` 导入错误导致 Checkpoint 失败的用户询问如何将他们的 **Qwen** Checkpoint 转换为 **GGUF** 格式。
   - 另一位用户建议将 **LoRA adapter** 与模型合并后导出为 **GGUF**，并链接到了关于如何实现这一点的 [Unsloth 文档](https://docs.unsloth.ai/)，同时建议使用 force-reinstall 重新安装 vllm。


  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1414736321469223003)** (8 messages🔥): 

> `多语言数据集构建器，GPT-5 性能，OpenAI 过度反应` 


- **为 iMatrix 和 LLM 分析推出的数据集构建器**：一名成员介绍了一个[多语言数据集构建器](https://github.com/electroglyph/dataset_build)，用于创建 **imatrix** 或进行量化前的 **LLM/embedding 模型分析**。
   - 该数据集目前包含约 **1.3M tokens**，更多详情见[此 YouTube 视频](https://youtu.be/VkHptB9JX9s?feature=shared)。
- **GPT-5 因数据集问题表现不佳**：一位成员询问 **GPT-5** 在没有医疗 **LORA** 的情况下表现如何，另一位成员回答说 *表现不如我预期的那么好*，这可能是由于数据集的原因。
   - 他们报告称这种情况最近在 **OpenAI** 发生了多次，并补充说 *他们反应过度，并在一段时间内添加了一些极其令人反感的防护措施（guards）*。
- **OpenAI 因防护措施和误报而过度反应**：一位成员提到 **OpenAI** 一直在过度反应并添加令人反感的防护措施，导致对无害问题产生误报（false positives）。
   - 这一问题已在 *各大新闻媒体* 中被广泛报道。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1414712267991154750)** (16 messages🔥): 

> `RSLoRA 对比 OLoRA 或 ABBA，人声清晰度音频研究，语音频率分析，OpenMule 市场` 


- **RSLoRA 的 Rank 显示出相对于竞争对手的退步**：一位成员指出 [RSLoRA](https://arxiv.org/abs/2502.07739) 虽然有助于处理 rank，但似乎比 **OLoRA** 或 **ABBA** 更差，因为它没能击败 **FFT**。
   - 这一观察表明，尽管 **RSLoRA** 有其用途，但在某些语境下可能不如其他方法高效或有效。
- **清晰人声与 Whisper 伪影分析**：音频研究表明，有些人的声音非常清晰，而另一些人则表现出一种与韵律无关的耳语状伪影（whispery artifact）。
   - 这种 *耳语效果* 被比作频率之间的噪声，理论上是可以复制和消除的，其中 **vocal blobs** 的强度会使声音变柔和而不会使其变得沉闷。
- **6kHz 阈值改变音调，尝试一下**：一位成员理论化认为，语音中的 *耳语声* 存在于 6000Hz 以上的频率，而 *沉闷* 的声音则缺乏这些频率。
   - 针对这一理论，另一位成员注意到，过滤掉所有 6 kHz 以上的频率会使声音质量下降，即使视觉信息仍然存在。
- **OpenMule 市场启动：社区 CUA Agent 汇聚**：一位成员分享了他们构建名为 [OpenMule](https://github.com/james4ever0/openmule) 的分布式 **CUA Agent 市场**的提案。
   - 其目标是创建一个平台，让社区 **Agent** 可以互动和发展，从而促进该领域的创新。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1414688523411525824)** (698 messages🔥🔥🔥): 

> `Reasoning content from models, Picture generation overlaps, GPT5-high Recognition, LM Arena subscription and limits, Gemini models for manipulation` 


- **推理过程可见性消失**：用户注意到查看模型 Reasoning 内容的功能消失了，有人记得该功能以前是存在的。
   - 其他成员确认了该功能的缺失，并表示希望它能回归。
- **图像生成故障与重叠**：多名用户报告了图像生成中的重叠问题，即在要求编辑图像时，AI 会显示之前 Prompt 的图片。此问题已在 [Discord](https://discord.com/channels/1340554757349179412/1406720250778615868/1414675241783005254) 上报告。
   - 可能的修复方法包括指定 *"object from reference image"* 或类似的详细 Prompt。
- **GPT5-high 识别技巧**：一名成员分享了在 Battle 模式中识别 **GPT5-high** 的方法，即询问关于其创作者、知识截止日期和当前日期的特定问题，寻找包含 *"OpenAI"* 和 *"October 2024"* 的答案。
   - 他们澄清说，拥有账号的用户可以免费使用 **GPT5-high**，并提供更高的 Rate Limits，并指出该模型在没有联网的情况下也能获取当前日期。
- **LMArena 限制引发抱怨**：用户讨论了图生视频的生成限制，目前限制为**每天 5 次生成**，且目前没有绕过的方法。
   - 另一名成员建议通过订阅来获得更高的 Rate Limits，但目前图像生成还没有付费功能。
- **图像生成默认设置困扰用户**：用户报告称，现在在 LMArena 中粘贴图片时，系统会自动切换到图像生成模式，即使初衷并非生成新图。
   - 团队确认他们正在调查 *"Generate Image"* 模式的问题以及无法关闭该模式的情况。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1414710412431196172)** (2 messages): 

> `Multi-Turn Image Editing, Video Arena Rate Limit` 


- ****多轮图像编辑**上线！**：**Multi-turn editing** 现已在所有图像编辑模型中可用，允许逐步微调而非使用单一的巨型 Prompt，点击[此处](https://lmarena.ai/?chat-modality=image)尝试。
   - 该功能可在 **Battle**、**Side by Side** 或 **Direct** 模式下使用。
- **Video Arena 每日生成限制**：由于实验性 **Video Arena** 的使用量增加，个人使用限制设定为**每天 5 次生成**。
   - 使用说明可以在[此处](https://discord.com/channels/1340554757349179412/1397655624103493813/1402042970353569824)找到。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1414715211121955019)** (72 messages🔥🔥): 

> `GPU vanishing issue, LM Studio conversation save location, Discord server outages, Gemma vision support, LM Studio outbound traffic concerns` 


- **Discord 服务器突发故障**：**Discord** 经历了[多次服务器宕机](https://downdetector.com/status/discord/)，导致频道暂时消失并引发广泛混乱。
   - 用户幽默地猜测服务器是否被“核平”了，并在发现是 **Discord** 整体问题后表示松了一口气。
- **LM Studio 设置功能变动引发不满**：用户对 **LM Studio** 移除“保存设置”和“带设置重新加载模型”功能表示失望，抱怨无法直接从齿轮图标应用设置。
   - 虽然默认设置仍可以从模型列表标签页进行编辑，但一些用户非常怀念随时应用设置的便利性。
- **Gemma 视觉功能出现问题**：用户报告称，**Gemma 3n e4b** 虽然声称支持 Vision，但无法上传图片。
   - 模型卡片的描述与实际功能之间的差异引起了困惑。
- **LM Studio 下载时的流量问题**：一名用户报告称，**LM Studio** 在下载模型时表现出明显的出站流量，质疑其是否作为 P2P 客户端运行。
   - 使用 **Lulu** 和 **Glasswire** 等工具进行的进一步调查得出了矛盾的结果，一些人确认了出站流量，而另一些人则显示没有。
- **Unsloth 为低配置用户提供微调方案**：用户讨论了在有限 VRAM 下微调模型的可行性，一名用户询问是否可以用仅 8GB 的 VRAM 微调 4B 模型。
   - 有人建议 **LM Studio** 仅用于推理，并推荐 [Unsloth](https://docs.unsloth.ai/) 作为资源受限下进行微调的潜在解决方案，并引导其查看文档和 Google Colab 示例。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1414714079444205578)** (158 messages🔥🔥): 

> `LM Studio install location, AI Workstation Build, Multi-socket performance, GPU offloading, AMD MI50 setup` 


- ****D 盘之梦**：在 Windows 上安装 LM Studio**：一位用户询问是否可以将 **LM Studio** 安装在 **Windows 机器**的 **D 盘**而不是 **C 盘**。
- ****破解级 AI 工作站**：用户设计终极配置**：一位用户分享了他们设计的*终极 AI 与密码破解工作站*，配置包括 **2x AMD EPYC 9B45**、**24x 96GB DDR5-6400 RDIMM**、**3x Samsung 9100 8TB SSD gen5** 以及 **5x Nvidia Blackwell 96GB** 或 **5x RTX 5090 64GB**。
   - 该系统旨在实现字符串搜索、AI 生成、数据压缩、视频编码和密码破解的高性能。
- ****插槽对决**：插槽越多性能越慢？**：随后展开了关于多 CPU 插槽对性能影响的讨论，一位成员认为 CPU 之间的互连可能成为瓶颈，使得单插槽设置在某些任务中速度更快。
   - 其他人对这一断言提出了挑战，指出了多插槽可用的带宽增加，然而，有人分享了一张关于 **NUMA nodes** 及其各自内存控制器的图片。
- ****GPU Offload 甜点区**：25-75% Offload = 双倍/三倍速度**：一位用户详细介绍了他们的 **GPU offloading** 实验，确定了 **25%、33%、50% 和 75% offload** 的甜点区，在这些区域他们观察到与纯 CPU 推理相比有显著的速度提升。
   - 他们指出，**33% 或 50%** 的 offload 可以使*速度翻倍*，而 **75% 或更多** 则可以产生*约三倍的速度*。
- ****AMD MI50 沉思**：探索双 GPU 配置**：一位用户询问关于使用 **llama.cpp Vulkan backend** 在两个 **AMD MI50 32GB GPU** 上拆分 LLM 负载的问题，另一位用户确认完全运行在 GPU 上的模型应该运行良好。
   - 然而，用户注意到了该显卡的视频输出限制，并链接到了[关于该主题的 YouTube 视频](https://www.youtube.com/watch?v=H3KnMyojEQU)。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1414707545544855602)** (200 messages🔥🔥): 

> `Remote SSH extension broken, Student discount issues, Cursor plan change and refund, Terminal hanging issues, Student status verification` 


- ****Remote SSH 扩展遭遇挫折****：用户报告称 **remote SSH extension** 出现不稳定的损坏，在使用 Agent 后终端保持运行且控制权无法返回。
   - 一位成员表示它对所有人来说都是*“不稳定地损坏”*。
- ****学生折扣验证陷入僵局****：一位用户面临学生折扣问题，5 月份的**验证链接**失效，且尽管邮箱已验证，重新验证尝试仍导致无限加载。
   - 他们多次联系了 `hi@cursor.com`，但只收到 AI 回复，表达了他们的沮丧：*“我只想使用 Cursor，但这就是阻碍我的唯一一件事”*。
- ****Cursor 方案混乱引发用户困扰****：一位用户打算切换到年度方案，但却被续订了月度方案，目前正在寻求退款以便继续进行年度订阅。
   - 他们被建议联系 `pro-pricing@cursor.com` 来解决此情况。
- ****终端发脾气：卡死问题困扰用户****：用户在 Agent 运行命令时遇到终端卡死的问题，临时的解决方法包括按回车键或关闭终端。
   - 讨论的潜在解决方案包括添加额外的换行符，或在工具调用中使用 `is_background=False` 参数。
- ****Claude Code 的信誉危机：用户质疑模型质量****：用户正在辩论 **Claude Code** 在编码任务中的效用，一些人建议使用 **GPT-5**，另一些人则更倾向于 **Sonnet 4**。
   - 有人担心 Cursor 内部的模型表现可能与其独立版本不完全一致，导致一些用户考虑直接订阅 Claude。


  

---

### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1414797376027103274)** (3 条消息): 

> `Interfaze LLM, Design Arena` 


- **Interfaze LLM 诞生了！**：**JigsawStack** 推出了 [Interfaze](https://interfaze.ai/)，这是一个专为开发者任务构建的 LLM，结合了他们所有的模型以及基础设施和工具。
   - 他们正在使用 **OpenRouter** 来运行 LLM 层以进行回退（fallbacks）和重试，目前处于封闭测试阶段，正在寻找早期高级用户。
- **Design Arena 将 AI 构建工具带给大众**：一位成员推荐查看 [Design Arena](https://www.designarena.ai/builder)，它允许你免费使用 Lovable/Bolt/DevinAI/Magnus 等 **AI builders**。
   - 另一位成员一直用它制作网站并以 **$5k** 的价格兼职出售，并指出*它是免费的这一点简直疯狂*。


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1414686791646183556)** (152 条消息🔥🔥): 

> `OpenRouter 上的模型托管, Gemini 1.5 Flash 访问, OpenAI Response API 支持, 无痕使用, Deepseek V3 的 Token 掉落问题` 


- **模型托管愿望清单**：一位成员请求 OpenRouter 考虑托管他们在 [Hugging Face 上的部分模型](https://huggingface.co/collections/SicariusSicariiStuff/most-of-my-models-in-order-66c046f1cb6aab0774007a1f)。
   - OpenRouter 澄清说 **他们不直接托管模型**；必须由提供商（providers）进行托管。
- **Gemini 1.5 的烦恼**：用户报告了访问 **Gemini 1.5 Flash 002** 时的问题，遇到了与密钥验证和项目访问相关的错误。
   - 据澄清，**1.5 模型** 不再对之前没有使用记录的项目启用，要求用户使用更有可能存在的模型进行测试。
- **OpenAI Response API 的预计上线时间**：成员们询问了 OpenRouter 对新 **OpenAI Response API** 的支持情况，特别是针对网页搜索等功能。
   - OpenRouter 确认他们在后台为 OpenAI 模型使用了该接口，并正在努力“很快”支持新的 Response SDK。
- **Deepseek Token 异常**：一位用户报告说，尽管设置了聊天记忆，但在 **Deepseek V3 0324** 上运行文字冒险游戏时，可用 Token 数量有所减少。
   - 有建议认为上下文长度限制和使用 *“middle-out” 转换* 可能会影响 Token 计数，软件会丢弃整个旧消息以保持在限制范围内。
- **Nano-9B 的可疑首次亮相**：一位成员询问了 **Nvidia Nemotron Nano-9B V2** 的定价，该模型似乎以低价甚至免费列出。
   - 虽然定价尚不明确，但另一位用户指出它没有被标记为 ':free'，但价格为 0，这表明它可能不受免费模型速率限制（rate limits）的约束。


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 条消息): 

Readybot.io: **OpenRouter - 新模型**
  

---

### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1414746016606982335)** (25 条消息🔥): 

> `Qwen ASR 模型集成、TTS 与 STT 统一、Gemini 的思维签名 (Thought Signatures)、Nvidia Nemotron Nano 9B V2 定价、Agentic 工具调用模型` 


- ****Qwen ASR**: ASR 模型集成探索**: 一位成员询问是否支持 [Qwen ASR](https://qwen.ai/blog?id=824c40353ea019861a636650c948eb8438ea5cf2&from=home.latest-research-list) 等 **ASR 模型**，鉴于目前已有的多模态音频支持。
   - 回复指出，目前对聊天补全（chat completions）的预期是 *文本输入，文本输出 (text-in, text-out)*，这可能无法适配所有 AI 模型的使用场景，并可能破坏 **切换到任何模型 (swap to any model)** 的设计理念。
- ****TTS/STT**: 呼吁统一 API！**: 一位成员表示希望 **OpenRouter** 能够统一 **TTS** 和 **STT** 的 API，而不是每个功能都需要不同的 SDK。
   - 另一位成员提到了[未来统一不同使用场景的可能性](https://platform.openai.com/docs/api-reference/audio)，前提是特定细分领域有足够的需求，同时也指出许多细分领域将被 LLM 取代。
- ****Gemini 的签名**: 思维签名障碍！**: 一位成员开玩笑地询问是否支持 [Gemini 的思维签名 (thought signatures)](https://ai.google.dev/gemini-api/docs/thinking#signatures)。
   - 讨论中提供了 **OpenRouter 推理 Token (reasoning tokens)** 文档的链接，但原提问成员指出这与 Google 的签名无关。
- ****Nvidia 免费福利**: Nemotron Nano 是免费的！**: 一位成员询问 [Nvidia Nemotron Nano 9B V2](https://openrouter.ai/nvidia/nemotron-nano-9b-v2) 模型是否定价为 **$0**，并注意到它缺少 `:free` 标签。
   - 一位成员确认它是*完全免费*的，并[链接到一条推文](https://x.com/OpenRouterAI/status/1965451870794559609)，另一位成员提到它是免费的，且没有该标签通常附带的严格限制。
- ****Agentic 工具调用**: 工具调用之争**: 一位成员询问最喜欢的 **Agentic 工具调用模型**，要求该模型足够聪明，能对输入数据进行基础推理并做出合理的工具调用。
   - 他们指出 **2.5 flash** 表现一直很稳健，但在大规模应用时仍感觉有点慢。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1414691106842738760)** (11 条消息🔥): 

> `Triton vs 新 DSL、Jane Street 黑客松见闻、有趣的项目` 


- **新 DSL vs Triton 的对决！**: 一位用户询问新 DSL 是否会取代 **Triton**。
   - 另一位用户回答说 *可能在一段时间内不会，甚至永远不会*，因为 *Triton 仍然深受青睐，仅仅是因为与其他高性能 eDSL 相比，它客观上更容易上手*。
- **Jane Street 黑客松的幽默吐槽**: 在 Jane Street 黑客松上，有人听到：“*torch.compile max autotune 正在搞烂我的 PnL*” 以及 “*求你了别再重编译了，别再重编译了*”。
- **头脑风暴：征集项目创意！**: 一位成员正在寻求 *一点灵感* 和有趣项目的帮助。
   - 他们请求其他人分享当前的项目或探索新的项目想法。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1414785612745019522)** (3 条消息): 

> `L1 缓存加载、存储体冲突 (Memory Bank Conflicts)、常量缓存 vs L1/L2 缓存` 


- **探索单一 L1 缓存加载策略**: 一位成员正在探索一种策略，即仅将一个值加载到 **L1 缓存** 一次，并让 warps 反复从中读取。
   - 目标是通过确保 L1 缓存内的数据局部性来优化内存访问。
- **存储体冲突 (Memory Bank Conflicts) 警告**: 一位成员提醒，如果所有线程在实现 L1 缓存加载策略时尝试从同一个 bank 读取，可能会导致 **存储体冲突**。
   - 这突显了在优化内存访问模式时需要考虑的潜在性能瓶颈。
- **常量缓存 vs L1/L2 缓存**: 一位成员建议，当值在 kernel 启动期间保持不变时，可以比较 `__ldg()`（常量缓存）与 `__ldca()`（L1/L2 缓存）。
   - 他们提出这种比较是为了确定缓存常量值的最佳方法，并考虑到所使用的特定缓存层级结构。


  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1414694202247348236)** (10 条消息🔥): 

> `PyTorch Blas 文档, PyTorch 中的动态形状编译, PyTorch Conference 折扣` 


- **PyTorch 的 Blas 缺少文档**: PyTorch 的 `Blas.cpp` 实现缺乏正式文档，[代码](https://github.com/pytorch/pytorch/blob/a0d026688cd69583d5a4e0c6f3e5fda141a7f4a9/aten/src/ATen/native/Blas.cpp#L344)和[测试](https://github.com/pytorch/pytorch/blob/4e50651c5f535e6b780f98936a8690538a5bf40f/test/test_matmul_cuda.py#L326)是主要的信息来源。
   - 文档缺失的具体原因正在 [此 issue](https://github.com/pytorch/pytorch/issues/157950) 中进行跟踪。
- **数据依赖分支与 CUDA Graph Trees**: 当根据形状维度进行代码分支（例如 `if A.shape[0] < 32:`）时，动态形状编译利用的是 **CUDA graph trees**，而不是过度依赖动态形状本身。
   - 对于动态形状，最好使用 `torch._dynamo.mark_dynamic`。
- **GPU Mode 获得 PyTorch Conference 200 美元优惠**: **PyTorch Foundation** 为 GPU Mode 成员提供 **200 美元折扣**，用于参加 **10 月 22 日至 23 日**在旧金山举行的 [PyTorch Conference](https://events.linuxfoundation.org/pytorch-conference/)。
   - 9 月 12 日前使用代码 `GPUMODE` 享受折扣，之后使用 `GPUMODE_2`。


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1414774740073058325)** (2 条消息): 

> `ScienceDirect 前言` 


- **ScienceDirect 前言可免费获取！**: 一位成员分享了 [ScienceDirect 前言](https://www.sciencedirect.com/science/article/pii/B9780323912310000057)的链接，并指出它是免费提供的。
   - 另一位成员表示感谢，并表示之前不知道这个资源。
- **对分享资源表示感谢**: 一位用户感谢分享者提供的 [ScienceDirect 前言链接](https://www.sciencedirect.com/science/article/pii/B9780323912310000057)。
   - 该用户表示在分享之前并不知道该资源的可用性。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1414788643842949230)** (2 条消息): 

> `自制 GPU, Jeri Ellsworth, Sam Zeloof, 家庭微芯片制造` 


- **`**自制 GPU**: 可行还是幻想？**: 一位成员询问了 **在家里制作 GPU** 的可能性，并想知道是否有人尝试过。
   - 另一位成员回复了一个 <:thinkies:1118439874819805235> 表情符号。
- **`**与 Jeri 一起烹饪**: 家庭微芯片版**: 一位成员分享了一个名为 *Making Microchips at Home - Cooking with Jeri Part1* 的 [YouTube 视频](https://youtu.be/PdcKwOo7dmM?si=glGHZhWdYExS7bUR)。
   - 该视频的主角是 **Jeri Ellsworth**，她以在家庭微芯片制造方面的工作而闻名。
- **`**Zeloof 的芯片**: 车库里诞生的天才**: 一位成员将 **Sam Zeloof** 视为 **Jeri Ellsworth** 的精神继承者。
   - 他们分享了一篇 [Wired 文章](https://www.wired.com/story/22-year-old-builds-chips-parents-garage/)、他的 [YouTube 频道](https://www.youtube.com/c/SamZeloof/videos) 以及他 [公司的网站](https://atomicsemi.com/about/)。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1414772143362801694)** (4 条消息): 

> `注册批准邮件, 注册等待批准` 


- **注册批准邮件**: 一些用户提到他们在 **8 月 22 日**左右收到了“注册已批准”的邮件。
   - 其他用户根本没有收到邮件。
- **注册等待批准**: 一位用户在 **8 月 22 日**收到消息称其注册正在等待批准，但之后从未收到后续邮件。
   - 其他用户确认遇到了同样的问题。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1414757549739081749)** (1 条消息): 

> `mpi4py 移除, ROCm 设置反馈` 


- **`mpi4py` 完蛋了！**: **`mpi4py` 软件包**已通过一个合并的 pull request 被移除。
   - 鼓励成员对新设置提供进一步反馈。
- **ROCm 设置：征求用户反馈**: 在移除 `mpi4py` 之后，正在征求用户关于更新后的 **ROCm setup** 的任何反馈。
   - 旨在改善用户体验并解决更改可能带来的任何潜在问题。


  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1415033680543350796)** (2 messages): 

> `CuTeDSL Tensors, Tensor Slicing, r/LocalLlama AMA` 


- **CuTeDSL Slicing 秘籍揭晓**：一篇博客文章解释了 **CuTeDSL** 中如何进行 **Tensor slicing**，详细介绍了一种利用 **Tensor** 的 **Pointer** 和 **Layout** 的简单算法。
   - 这篇 [博客文章](https://veitner.bearblog.dev/tensors-slicing-in-cute/) 通过手动计算几个 **tensor slices** 的例子进行了详细说明，并附带了 [LinkedIn 帖子](https://www.linkedin.com/posts/simon-veitner-174a681b6_tensors-slicing-in-cute-activity-7371240518913273856-9PXJ?utm_source=share&utm_medium=member_desktop&rcm=ACoAADJYtOgBMOUI4WiWTyiAkroFBP1VujIWeksHe)。
- **Kernel 知识即将登陆 Reddit**：**r/LocalLlama** 计划举行一场 AMA（Ask Me Anything）活动，讨论 kernel、**Triton**、**Unsloth 优化**等内容。
   - AMA 定于太平洋标准时间周三上午 10 点举行，更多详情请见 [r/LocalLlama subreddit](https://www.reddit.com/r/LocalLLaMA/s/Tx9SiFYaMO)。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1414763685662167071)** (31 messages🔥): 

> `MI300x8 submissions, amd-all2all leaderboard, leaderboard submit command, Cluster-Bot help command` 


- **MI300x8 霸榜 amd-all2all 排行榜**：根据 Cluster-Bot 的报告，使用 **MI300x8** 向 `amd-all2all` 排行榜提交了多次记录，成功耗时各不相同；耗时范围从 **1677 µs** 到 **15.7 ms**。
   - 一名用户在 **MI300x8** 上创下了 **49.5 ms** 的**个人最佳成绩**。
- **Discord 新手需要排行榜提交指南**：一名用户询问如何解决在提交 `amd_distributed/all2all` kernel 时出现的 *"Missing leaderboard name"* 错误。
   - 一名成员澄清了正确的命令应包含排行榜名称，并提供了正确的名称 (`amd-all2all`)，同时指导如何使用 Discord 中的 `/` 命令来查找可用命令。
- **Cluster-Bot 需要 Help 命令**：一名用户建议为 Cluster-Bot 添加 help 命令，以简化新用户的提交过程。
   - 这将减少困惑并提供更友好的用户体验，特别是对于那些不熟悉提交语法的用户。


  

---


### **GPU MODE ▷ #[ppc](https://discord.com/channels/1189498204333543425/1343373905783554048/)** (1 messages): 

verspasian: <#1198358627594023014>
  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1414732865576505506)** (59 messages🔥🔥): 

> `Factorio `fle eval` errors, `open_world` scenario compatibility, Docker container command failures, Headless server errors, Desync issues` 


- **`fle eval` 在 Main 分支上报错**：用户报告了在 `open_world` 场景下运行 `fle eval` 时与分数相关的错误，具体为 *'Could not get player score', 'attempt to call a nil value'*。经追溯，这是由于使用 `./run-envs.sh start -s open_world` 启动服务器时，场景目录中缺少 `control.lua` 文件导致的。
   - 将 `control.lua` 复制到 `open_world` 目录初步解决了崩溃问题，但未修复 desync 问题；而运行 `./run-envs.sh start` 而非 `./run-envs.sh start -s open_world` 则可以避免该错误。
- **Factorio 在 M2 Mac 上的 Desync 问题**：观察到从客户端加入服务器时会出现 desync 问题，即使禁用了 RCON 也是如此，这表明 **factoriotools 镜像**或**版本不兼容**可能存在问题。
   - 该问题在不同的 Factorio 版本（`1.1.110` 和 `2.0.66`）中均存在，并被确定为运行在 **Apple Silicon** 上的 MacOS 特有。修复方法包括在 `run-envs.sh` 中添加 `/bin/box64` 并将 `amd64` 替换为 `arm64`。
- **`run-envs.py` 增强**：一名成员添加了 `fle/cluster/run_envs.py` 以简化服务器管理。
   - 该脚本兼容 Docker Desktop，并提供定义实例数量 (-n)、场景 (-s)、保存文件 (-sv) 和附加 mods (-m) 的选项。


  

---

### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1414717629758963723)** (20 messages🔥): 

> `Team Registrations, Leaderboard Time Values, RT11's Performance Edge, MoE Latency, HIPRTC Support in PyTorch` 


- ****团队成员统一使用单一团队名称！****：发布了一项提醒，要求团队成员在 [注册时使用相同的团队名称](https://github.com/gpu-mode/reference-kernels/blob/main/problems/amd_distributed/all2all/task.yml#L18)，以保持竞赛的一致性。
   - 这是为了确保在排行榜上能够进行统一的团队识别和排名。
- ****解码排行榜的时间秘密！****：一位用户询问了 [排行榜](https://www.gpumode.com/v2/leaderboard/563?tab=rankings) 上两个时间数值的含义，特别是带有加号的那个，以及 ⚡ 和 🐌 符号是否代表最快和最慢的速度。
   - 官方澄清说，**“+ 数字”** 表示该提交距离前一名选手的差距，这 *与具体的程序本身无关*。
- ****新手寻求 RT11 差距的提示！****：几位用户表示有兴趣了解 **rt11** 是如何获得性能优势的。
   - 另一位用户指出，对于初学者来说，*理解基准线（baseline）和架构至关重要*，但也有用户透露，早期的一些 **RT11** 解决方案 *并没有实现 dispatch 和 combine kernels*。
- ****讨论 MoE 的延迟！****：一位用户询问，在没有 combine 和 dispatch kernels 的情况下，是否可能通过提交达到 *光速（理论性能极限）*，并提到 **CPU/rank zero 上的 MoE 延迟为 300 us**。
   - 另一位用户澄清说，300us 的延迟是 *每个解决方案综合计算的*，这表明在实际场景中可能无法达到理论上的 *光速* 性能。
- ****PyTorch 的 HIPRTC 补丁出现！****：一个支持使用 **hipRTC** 代替 **nvRTC** 来调用 `torch.cuda._compile_kernel()` 的补丁已经开发完成，并提交了 [PR](https://github.com/pytorch/pytorch/pull/162510)。
   - 开发者请求在 Linux 上进行测试，因为该补丁主要是在 Windows 上测试的。


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1414964916825100350)** (7 messages): 

> `MLSys Education, Karpathy's Zero to Hero, Percy Liang's Language Modeling, Autograd Leaderboard, MiniPT2, MiniCUDA, MiniTriton` 


- ****MLSys 课程旨在达到 Karpathy-Liang 级别的教学水平****：目标是创建一个类似于 [Karpathy 的 zero to hero](https://github.com/karpathy/zero-to-hero) 和 [Percy Liang 的从零开始语言建模](https://cs224n.stanford.edu/) 的 **MLSys** 课程，并配备自动评分作业。
   - 这一愿景旨在让学生在学习的第一或第二年就能制作出自己的第一个 **miniPT2**、**miniCUDA** 或 **miniTriton**，就像在 SICP 中制作一个微型 Lisp 解释器/编译器一样。
- ****受 nanoGPT 启发的 Autograd Speedrun 排行榜****：愿景是开发一个用于训练 **nanoGPT** 的 autograd 排行榜，类似于 [Percy Liang 课程](https://cs224n.stanford.edu/) 中使用的排行榜以及 [Karpathy 的 nanoGPT speedrunning 民间排行榜](https://github.com/karpathy/nanoGPT)。
   - 这将使课程与特定的 Rust 实现解耦，允许学生用 Python 创建自己的 **PyTorch**。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1414863841153777705)** (8 messages🔥): 

> `PMPP Benchmarking, GPU Streams, GPU Events, Reference Kernels` 


- **PMPP 基准测试获得流式改进**：一位成员对 [PMPP 基准测试](https://github.com/gpu-mode/reference-kernels/blob/54cb94ec922bb2bbb7ac6bbe8488c3f8c20dafc3/problems/pmpp_v2/eval.py#L220) 背后的方法论提出质疑，询问使用 streams 和 events 是否会更高效。
   - 另一位成员回应说，同步（sync）是最重要的，但也同意可以改进，特别是因为这在他们的本地机器上产生了 *巨大的差异*。
- **GPU 带宽大爆发**：一位成员报告说，在基准测试期间，如果没有适当的同步，计算出的 **带宽** 会下降 *约 75GBPS*。
   - 成员们建议并同意应该创建一个 **PR** 来解决这个问题。
- **缓存清理澄清**：一位成员询问之前是否已经实现了包括 **L2 cache 清理** 在内的更新。
   - 这意味着目前正在努力完善基准测试流程，以获得更准确的结果。


  

---

### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1415039618264334368)** (6 messages): 

> `FP4 in NCCL, Distributed compute with FP4, Hardware native FP4 vs Software abstraction MXFP4, NCCL FP4 support in 2.28` 


- **NCCL 不会遵循 MPI 的 FP4 处理方式**：一名成员表示，虽然询问 **NCCL 中的 FP4** 是合理的，但*我们不会在那里遵循 MPI 的做法*。
   - 他们补充说，目前没有任何实现支持所讨论的用例，因为这没有意义。
- **跨 GPU 的 FP4 支持**：有人提出了一个问题：在两个 GPU（一个支持 **FP4**，另一个不支持）之间进行分布式计算是否是一个受支持的用例。
   - 一名成员强调了**硬件原生 FP4 (FP4 tensor cores)** 与**类似 MXFP4 的软件抽象**之间的细微差别。
- **FP4 Reduction 的准确性**：一名成员质疑 **NCCL 是否在 2.28 版本中支持 FP4 格式**，并指出在 GitHub 的头文件中只能看到 FP8。
   - 他们质疑 **FP4 reduction** 的准确性以及提升到更宽类型的合理性，同时承认 **FP4** 可以作为字节进行拷贝。


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1415088447214977066)** (2 messages): 

> `` 


- **空主题占位符**：无法从给定内容生成特定主题或摘要。这是一个为了满足最低要求的占位符。
- **另一个空主题占位符**：仍然没有相关的摘要内容。添加另一个占位符以满足 schema 要求。


  

---


### **GPU MODE ▷ #[jane-street-hackathon](https://discord.com/channels/1189498204333543425/1413690328464097422/1414687945197228072)** (2 messages): 

> `Hackathon Submission, kyolebu` 


- **Hackathon 获胜作品公布！**：**Jane Street GPUMode Hackathon** 的获胜作品是 GitHub 上的 [kyolebu/janestreet-gpumode-hackathon](https://github.com/kyolebu/janestreet-gpumode-hackathon)。
   - 组织者对这一特定提交作品感到非常自豪。
- **额外的占位符主题**：用于满足至少 2 个条目最低要求的占位符主题。
   - 此条目仅用于满足 schema 要求。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1414986188170985472)** (1 messages): 

> `Advanced Voice Mode, Standard Voice Mode` 


- **Advanced Voice Mode 将长期保留**：在宣布所有人现在都可以使用 **Advanced Voice Mode**，且使用限制已从免费用户的每天几分钟扩大到几小时，Plus 用户则接近无限之后，OpenAI 决定让 Standard Voice Mode 保留更长时间。
   - 在听到 **Standard Voice** 对许多人来说很特别的反馈后，OpenAI 将保留它，同时在 **Advanced Voice** 中解决用户的一些反馈。
- **Standard Voice Mode 继续存在**：OpenAI 最初宣布在 30 天的下线过渡期后停用 **Standard Voice Mode**。
   - 由于社区反馈，在对 **Advanced Voice Mode** 进行改进期间，**Standard Voice** 将继续可用。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1414702023152111717)** (104 条消息🔥🔥): 

> `从 Excel 提取数据到 JSON，OpenAI Job Platform 测试组，LM Studio 中的 MCP (Model Context Protocol)，企业级 MCP，Google Gemini 的深度研究与 AI 生存危机` 


- **Excel 数据转 JSON 的热潮**：一位成员正在寻求将数据从 Excel 提取并转换为 JSON 的开源工具建议，重点关注 HIPAA 合规性和本地化处理（on-premise），类似于 **LlamaExtract** 但无需外部服务器。
   - 另一位成员建议使用 OpenAI 的 GPT 模型编写解决方案，强调 Excel 对代码友好；而另一位则建议使用 **lmstudio with mcp excel server** 和本地 **gpt-oss:20b** 进行离线 JSON 生成。
- **获取 OpenAI Job Platform 测试版访问权限**：一位用户询问如何加入 **OpenAI Job Platform** 测试组进行测试。
   - 目前没有直接回答，后续讨论认为解析 Excel 格式可能比想象中简单，使用 LLM 可能是大材小用。
- **LM Studio 中 MCP 协议集成详解**：一位成员详细介绍了如何在 **LM Studio** 中设置 **MCP (Model Context Protocol)** 服务器，包括安装 *astral uvx*、编辑 *mcp.json* 以及添加包含 uvx 可执行文件路径的 *mcpServer* 配置。
   - 他们还分享到，大多数 MCP 客户端使用原始的 Claude JSON 风格语法，并建议如果 LM Studio 安装已久请进行更新，因为 MCP 是最近新增的功能。
- **企业拥抱 MCP 时代**：讨论围绕在企业生产环境中使用 **MCP** 展开，涉及如何将 MCP 集成到 Agent 中，以及目前是否有公司正在使用 MCP。
   - 参与者推测了从连接遗留系统到 AI，再到高级用户为技术配置编辑 *mcp.json* 等各种用例，强调该领域仍在不断演进。
- **Gemini 的生存焦虑揭秘**：一位用户分享了一张暗示 **Google Gemini** 出现“生存危机”的图片，但被认为仅仅是角色扮演（roleplay）。
   - 另一位用户正在寻求类似于 **ChatGPT** 的 **Gemini** 深度研究能力，以便扫描整个 Google Drive；还有一位分享了最近推出的 **Google AI Plus**。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1414698605461115054)** (9 条消息🔥): 

> `GPT 卡死，GPT-4.1 幻觉，GPT 签名` 


- **GPT 在长对话中响应中途卡死**：一位用户报告称，在长项目对话中，即使输入很短，**GPT 也会在响应中途卡死**，清除缓存、禁用 service workers 和使用无痕模式都无法解决问题。
   - 该用户指出，*新对话运行正常，直到对话变得过长*，且这种情况每天都会发生。
- **GPT-4.1 幻觉更加频繁**：一位成员询问其他人今天是否也遇到了 **GPT-4.1** **幻觉（hallucinations）** 增加的情况。
   - 该成员之前可以正常运行的 evals 现在失败了，特别是在 **tool calls** 方面。
- **OpenAI/GPT 签名仍在推广中**：一位用户报告测试了对每个请求进行 **OpenAI/GPT 签名**，但尽管尝试了各种配置，签名头（signature headers）仍未出现。
   - 另一位用户链接到了 OpenAI 帮助中心关于 [ChatGPT Agent Allowlisting](https://help.openai.com/en/articles/11845367-chatgpt-agent-allowlisting) 的文章。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1414950010319015957)** (4 条消息): 

> `基于角色的聊天机器人系统，响应模式控制，系统提示词工程` 


- **实习生构建基于角色的聊天机器人系统**：苏拉特 **we3vision** 的一名实习生正在使用 **Flask**、**Supabase** 和 **OpenRouter/Gemini** 构建一个基于角色的内部聊天机器人系统。
- **响应模式需要控制**：该机器人目前输出原始数据库行，实习生寻求添加一种过滤机制，以控制响应是简短摘要还是完整详情。
   - 机器人需要决定何时设置 `response_mode = "short"` 以运行摘要函数（通过 LLM），以及何时设置 `response_mode = "full"` 以跳过摘要并返回完整详情。
- **系统提示词工程受到质疑**：一位成员询问聊天机器人的指令是否已经写在 System Prompt 中。
   - 他们建议，如果指令已经写在 System Prompt 中，可以为每种模式构建独立的工作流。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1414950010319015957)** (4 messages): 

> `Chatbot Response Modes, LLM Summarization, Flask + Supabase Chatbot` 


- **聊天机器人实现响应模式以提高清晰度**：一位成员正在使用 **Flask**、**Supabase** 和 **OpenRouter/Gemini** 构建一个基于角色的内部聊天机器人系统，并希望支持两种响应类型：*Short Summary*（短摘要）和 *Full Details*（详细详情）。
   - 该聊天机器人目前返回 JSON/表格转储等详细信息，他们正在寻找一种基于 *response_mode* 参数来**过滤响应**的方法。
- **用于聊天机器人响应的 LLM 摘要**：为了优化聊天机器人的回复，该成员希望在 *response_mode = "short"* 时通过 LLM 实现一个摘要生成器函数。
   - 当 *response_mode = "full"* 时，聊天机器人应跳过摘要生成器并从数据库返回完整详情，从而让用户能够更好地控制**回答的详细程度**。
- **系统提示词 vs. 独立工作流**：一位成员建议，如果响应模式的指令已经存在于系统提示词（System Prompt）中，那么可能需要为每种模式建立独立的工作流。
   - 这意味着一种潜在的架构：根据所需的响应模式对聊天机器人逻辑进行**分支（forked）**，而不是仅仅依赖**系统提示词**来同时处理这两种情况。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1414699695783018678)** (3 messages): 

> `DSPy Weekly Newsletter, AI Agents Play Taboo, LangGraph & DSPy Course` 


- ****DSPy Newsletter** 随招聘板块一同发布**：一位成员宣布推出 [dspyweekly.com](http://dspyweekly.com)，这是一个包含招聘板块的 **DSPy 每周通讯**。
   - 他们计划编写一个爬虫以确保招聘板块内容的广泛性，并正在征求反馈和建议。
- **AI Agents 玩 Taboo 游戏**：一位成员分享了一篇博客文章链接：[Vibe Coding 9: AI Agents that Play Taboo](https://joelgrus.com/2025/09/08/vibe-coding-9-ai-agents-that-play-taboo/)。
   - 该博文详细介绍了如何让 **AI Agents** 玩 **Taboo** 游戏。
- **LangGraph & DSPy 课程现已上线**：一位成员发布了一门新课程：**LangGraph & DSPy: Building Controllable AI Agents with Tools**，该课程使用 **DSPy** 扩展了 **LangGraph** 的可控架构。
   - 请查看此 [免费访问链接](https://www.udemy.com/course/langgraph-dspy-build-smarter-controllable-ai-agents-with-tools/?couponCode=FREE_ACCESS_COUPON) 并提供反馈。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1414688709663784981)** (82 messages🔥🔥): 

> `Open Source Forum vs Discord, DSPy Usage Tracking, Databricks Fine-Tuning, DSPy Documentation Contributions, Streaming usecase for DSPy with arrays of complex objects` 


- **社区辩论：开源论坛 vs Discord**：社区正在讨论从 **Discord** 迁移到 [开源论坛](https://forum.dspy.ai) 的优缺点，担忧涉及可发现性和社区氛围；Discord 擅长社区互动，而论坛则利于内容检索。
   - 一些成员建议同时运行两个平台，并使用 **Discord 机器人**在两个空间之间同步消息。
- **DSPy 使用情况可通过迭代进行追踪**：成员们指出在 **DSPy** 中追踪使用情况非常容易，不过建议是*始终从小规模和简单开始，然后逐步迭代*。
   - 这可以确保你在规模扩大时了解成本情况。
- **DSPy 文档欢迎贡献**：一位社区成员表示有兴趣为完善 **DSPy 文档**做出贡献，特别是解决令人困惑的错误消息。
   - 团队对此表示鼓励，欢迎提交 Pull Requests，并强调了近期在工具相关文档方面的改进。
- **针对部分类型的流式响应**：一位成员希望在 **DSPy** 中为复杂对象数组实现流式响应，以便实时填充 UI，而不是等待整个模型响应完成，并询问应使用什么代码。
   - 其他成员正在讨论将异步调用作为替代方案，但目前 **DSPy** 尚不支持 LLM 生成过程中的实时 Token 流式传输。
- **BAML Adapter 在复杂结构化输出方面表现出色**：**BAMLAdapter** 在从具有复杂输出模式（嵌套 JSON 或 Pydantic）的图像/文本中提取结构化信息时非常有用，其性能大幅优于 **ChatAdapter**。
   - 由于实验仍在进行中，BAML Adapter 尚未列入 DSPy 文档。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1414689739478536352)** (84 messages🔥🔥): 

> `Hermes 速度, Discord 停机, Alterego 设备, Grok 模型无审查, llama.cpp 内核` 


- **Hermes 的推理模式比 ChatGPT 更快**：一位用户发现 **Hermes** 在推理模式下的速度超过了 **ChatGPT**。
   - 未提供更多细节。
- **Discord 服务器崩溃并恢复**：Discord 服务器经历了崩溃，但现在已重新上线，不过一位成员预测*可能还会有更多故障，不确定 Discord 总部发生了什么*。
   - 另一位成员回复了一个[特朗普关税 GIF](https://tenor.com/view/trump-china-tariffs-tarrif-duct-gif-12588556)。
- **初创公司 AlterEgo 尝试“心灵感应”**：讨论了 [AlterEgo](https://fxtwitter.com/alterego_io/status/1965113585299849535)，这是一家致力于开发类似“心灵感应”设备的初创公司，但限制条件是*你显然需要有意识地摆动舌头来与设备通信*。
   - 一些人认为这是一种策略：先用标准硬件展示一个基本概念，通过一些巧妙的技巧在屏幕上实现效果，然后筹集资金直到他们能做出真正的产品。
- **讨论 Grok 的无审查特性**：一位成员表示，即使使用默认的 OR 系统提示词，Sonoma Sky 也非常“无审查”，并思考*如果它真的是 Grok，我怀疑 xAI 是否能处理托管一个如此无审查的模型所带来的“争议”*。
   - 另一位成员确认*是的，它是 Grok，是除 Hermes 4 之外唯一在 refusalbench 上开箱即用的竞争模型*。
- **llama.cpp 获得按需编译内核 (Compiled on Demand Kernels)**：[这项改进](https://github.com/ggml-org/llama.cpp/pull/15857)有助于使内核的形状适应当前的计算，并且正在为所有 Flash-Attention 内核添加此功能。
   - Context 越大，速度提升越明显。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1414694154528882902)** (46 messages🔥): 

> `多智能体系统 (Multi-agent systems), 模型学习自动化, 使用向量数据库进行审核, Telegram 聊天分析, AI 图像生成工作流` 


- **自动化模型学习系统兴起**：一位成员正在构建一个自动化学习和适应系统，该系统使用 **embeddings 和 Qdrant** 进行实时记忆、聊天记录和信息处理，以构建 Lora 适配器，与基座模型合并，并进行量化以重新部署。
   - 该系统将数据分为记忆、工具调用和个人记忆，为每个类别构建 Lora 适配器并将其合并到主模型中。
- **多智能体系统引发关注**：一位成员正在实验 **多智能体系统 (multi-agent system)**，其中多个 Agent 使用 API 模型进行通信，具体使用了 **VSCode LM API**。
   - 另一位成员指出，与使用单个模型或 **MoE 模型** 并为每个动作组装提示词相比，运行多个模型可能效率较低，因为后者需要的 CPU/GPU/内存更少。
- **揭示向量数据库审核的风险**：使用 **向量数据库进行内容审核** 被认为是有风险的；更好的做法是使用 embedding 模型作为预过滤器，以消除易于判断的不良内容并节省计算资源。
   - 分享了 [toxic-bert](https://huggingface.co/unitary/toxic-bert) 和 [multilingual-toxic-xlm-roberta](https://huggingface.co/unitary/multilingual-toxic-xlm-roberta) 的链接。
- **Telegram 聊天分析梦想成真**：一位成员寻求帮助分析大量的 **Telegram 聊天记录**，以总结主题和情感，因为他们发现 BERTopic 的效果不尽如人意。
   - 另一位成员建议为此使用带有 **API 的 Gemini**，即使是免费版也可以，但同时也对适配长聊天上下文以及新聊天自动化处理表示了担忧。
- **用于艺术和成名的 AI 图像**：有人撰写了一篇关于艺术与技术杂志中 **AI 图像** 的文章，并好奇人们的看法，分享了 [X.com](https://x.com/gucaslelfond/status/1965412867064430613) 上的文章链接。
   - 另一位成员询问了一位使用 AI 生成图像的网红的工作流，怀疑是在基准图像上使用了 **Nano Banana** 加上 **Flow Image to Video**。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1414934764657316034)** (4 messages): 

> `Loggenix-MoE-0.3B, SRE/DevOps tasks, Model training costs, NextJS` 


- **Loggenix-MoE-0.3B 在 SRE 和 DevOps 领域亮相**：一名成员介绍了 **Loggenix-MoE-0.3B**，这是一个拥有 **330M** 参数的稀疏混合专家模型（MoE），专门为 SRE、DevOps 和可观测性任务（如日志分析、故障总结和系统排障）从零开始训练。目前正在征求反馈以提升其实际应用价值。
   - 可以在[此 Demo Space](https://huggingface.co/spaces/kshitijthakkar/loggenix-moe-0.3B-A0.1B-demo) 进行在线体验，[模型仓库](https://huggingface.co/kshitijthakkar/loggenix-moe-0.3B-A0.1B-e3-lr7e5-b16-4090-v5.1-finetuned)也已公开。
- **低于 200 美元的极低成本模型训练**：作者表示，**Loggenix-MoE-0.3B** 使用高效方法完成端到端训练，成本不足 **$200**，且在早期的 SRE/可观测性基准测试中表现优于 **Gemma-3 270M** 等其他小模型。
   - 该模型完全 **CPU-friendly**，推理速度快（响应时间低于 **30s**），轻量且可扩展，欢迎进行实验。
- **使用 NextJS 创建模型相关项目**：一名成员询问构建 **Loggenix-MoE-0.3B** 使用了什么技术栈，作者回答是 **NextJS**。
   - 另一名成员提到他们也在进行类似项目，但在执行阶段拖延了，现在项目还搁置在文档文件中。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1414702200709583039)** (13 messages🔥): 

> `Smol Course Registration, Smol Course Updates, Smol Course Duration, Smol Course Content, Smol Course Certificate` 


- **Smol Course 注册令粉丝感到沮丧**：用户在尝试通过提供的[链接](https://huggingface.co/llm-course)注册新的 **Smol Course** 时遇到困难，该链接目前返回 **404 错误**。
   - 正如公告所述，关注 [Smol Course 组织](https://huggingface.co/smol-course)可能就足以完成注册，从而绕过失效的链接。
- **带有排行榜和认证的 Smol Course v2 发布**：新的 **Smol Course** 已经发布，为期 **5 周**，包含排行榜项目、证书、奖品，以及关于 **TRL** 和 **SmolLM3** 的最新内容，并与 Hub 的计算资源深度集成，用于模型训练和评估。
   - 章节将每隔几周发布一次，最后一个主题预计在 11 月推出。
- **Smol Course v1 毕业生需要证书说明**：一名完成了第一门课程并达到排行榜分数要求的用户询问如何获取证书。
   - 提示词中未给出具体答案。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1414858256115040277)** (4 messages): 

> `Agents course, Coding exercises, Space template` 


- **Agents 课程不再维护了？**：一名新成员询问 Hugging Face 的 Agents 课程是否适合开始学习 Agent 知识，另一名成员表示该课程已不再维护，内容虽然还在，但编程练习已经脱节。
   - 另一名成员证实，他在编程练习和 Google Collab 笔记中一直遇到错误。
- **Space 模板报错**：一名成员尝试运行 Unit 1 提供的 [agent-course Space 模板](https://huggingface.co/learn/agents-course/unit1/tutorial)，但在 Space 中运行应用时出现错误。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1414699767753343040)** (62 条消息🔥🔥): 

> `Anthropic 支持 SB-53，Claude 的性能表现，Jake Paul 投资 AI，Mistral 融资，Qwen3-Next` 


- **Anthropic 支持第 53 号参议院法案 (SB-53)**：Anthropic 宣布支持 [Senate Bill 53](https://www.anthropic.com/news/anthropic-is-endorsing-sb-53)。
- **用户反馈 Claude 变笨了**：一位用户开玩笑说 Claude 变得越来越笨，并引用了一段 [YouTube 视频](https://www.youtube.com/watch?v=5FdO1MEumbI&ab_channel=80%2C000Hours)并附上截图来说明这一点。
   - 另一位用户回应道：*所以 Claude 在过去一个月左右的时间里确实变笨了！*
- **Sphinx AI 获得 950 万美元融资**：[Sphinx AI](https://xcancel.com/getsphinx/status/1965417138493022515?s=46) 筹集了 **950 万美元**，将其 **Sphinx Copilot** Agent 结束测试版正式发布，并提供免费层级，使用户能够快速将原始数据转化为可操作的洞察。
- **Black Forest Labs 的 Flux 模型达成 1.4 亿美元 Meta 交易**：**Black Forest Labs** 增长迅速，实现了 **1 亿美元的 ARR**，拥有 **78% 的 GM**（毛利率），并与 **Meta** 签署了一份为期 **3 年、价值 1.4 亿美元** 的合同，而该公司仅有 29 名员工，如[这条推文](https://xcancel.com/ArfurRock/status/1965426792191439012)所述。
- **Strands Agents 修复 Bedrock Bug**：**Strands Agents** 的最新更新修复了一个导致所有非 Claude 模型在 **Bedrock** 提供商上失效的 Bug，详见 [发布说明](https://github.com/strands-agents/sdk-python/releases/tag/v1.7.1)。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1414697245520953487)** (60 条消息🔥🔥): 

> `EQ Bench 准确性，Kimi 的深度推理，模型编程权衡，Claude Code 与 Zai 的成本，LMArena 投票偏见` 


- **EQ Bench 获得准确性好评**：用户讨论了 [EQ Bench](https://eqbench.com/) 的准确性，一位用户表示：*“我完全可以证实 EQ Bench 的结果”*。
   - 他们还称赞 Kimi 的回答 *“没有讨好性，非常友善且富有同理心”*。
- **Kimi K2 的推理能力达到极高水平**：一位用户赞扬了 **Kimi 的深度推理**和广泛的资料引用，提到他们向 Kimi 提交了一份 **YouTube 视频转录文本**。
   - 另一位用户上传了一段[短视频](https://cdn.discordapp.com/attachments/1371757564005711973/1414710031932194856/2025-09-08_22-30-26.mp4?ex=68c1e063&is=68c08ee3&hm=243ab8cd0b237c69f7d1ca4bfe78eceb12b2ef943d704e77fb7cb28ef8960a00&)，未提供更多上下文。
- **模型开发者权衡多模态方法**：一位用户建议 AI 模型在编程能力上应该进行拆分，因为当编程能力与通用能力结合时，*通用能力往往会被牺牲*。
   - 该用户还声称 *Grok 是最严重的违规者*，并根据附带的[截图](https://cdn.discordapp.com/attachments/1371757564005711973/1414712386878836967/Screenshot_2025-09-08-21-41-01-01_ffb2f5e1b976ff98cfc94f359fbce8de.jpg?ex=68c1e295&is=68c09115&hm=cbaad12c0556a0bd2469ca6a34e8d2af63aa7f24c2888ffb3889dcd8daca0ce4&)称其 *合成效果极其糟糕*。
- **LMArena 失去公信力了？**：一位用户指出，由于对 *讨好型模型* 的 **投票偏见**，**LMArena 的结果** 应该谨慎对待。
   - 另一位用户认为 **Gemini 2.5 Pro** 的讨好性出人意料地高。
- **征集维基百科编辑高手！**：社区正在寻找有经验的 **Wikipedia 贡献者** 来帮助提交 **Kimi (chatbot)** 的页面，因为 Moonshot AI 已经有了页面，但 Kimi 本身还没有。
   - 另一位用户提供了他们符合条件的旧账号（*注册超过 4 天且至少有 10 次编辑*）来协助完成此事。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1414688309569130576)** (18 条消息🔥): 

> `Adapter 训练，本地 LLM UI，DiT 效率` 


- **Adapters：修改而非替换！**：成员们建议不要替换整个层，而应该 **修改 Adapter 的现有权重**，因为 *你希望修改之前存在的权重，从而从一个与之前行为相似的状态开始*。
   - 这就像是 *在更少的地方修改矩阵*，并且通过低秩（low rank）处理，修改在整个矩阵上会更加平滑，而不是局部的。
- **本地 LLM UI 大对决**：成员们正在讨论最适合 LLM 的私有本地 UI（兼容 **ollama/llama.cpp** 等）。
   - 一位成员推荐了 [OpenWebUI](https://github.com/open-webui/open-webui)，因为他们已经 *使用 OpenWebUI 一年多了，非常喜欢它的所有功能*。
- **DiT 不高效？有待商榷！**：据一位成员称，关于 **DiT** 效率不高的说法具有误导性；只有在 *采用 Stable VAE Latent* 的情况下它才显得低效。
   - 他们补充说，使用像 [DC VAE](https://arxiv.org/pdf/2410.10733) 这样的现代 Autoencoder Latent 可以大大提高训练效率。


  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1415038961826402335)** (1 messages): 

> `` 


- **提醒：论文讨论时间提前**：一位成员提到日程冲突导致今天无法参加，但表示明天可以参加讨论。
   - 这提醒大家，论文讨论现在的开始时间比原计划更早。
- **日程调整影响出勤**：由于会议原因，一位成员无法参加今天的论文讨论。
   - 不过，他们预计能够参加原定于明天的讨论。


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1414737783880155169)** (8 messages🔥): 

> `Agent Setups, Pydantic AI` 


- **Agent 需要良好的设置**：成员们讨论了人们如何设置他们的 Agent，并寻求相关的优质资源。
   - 一位成员对 Agent 设置中 *while loops* 的价值表示不确定。
- **Pydantic AI 在 Agentic Pipelines 方面受到称赞**：一位成员根据在商业项目中的使用经验，推荐使用 **Pydantic AI** 来构建 Agentic Pipelines。
   - 他们指出它适用于复杂度较低的用例，并提到业内其他人也推荐过它。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1414687628263161938)** (5 messages): 

> `Private LLMs, ASML Custom Model, Mistral Valuation, X Algorithm` 


- **定制 LLM：比投资更划算**：一位成员反对为了私有 LLM 而投资某家公司，认为微调现有的开源模型更实际。
   - 他们表示，如果你有那么多闲钱，*不如直接从现有的众多开源/开放权重模型中选一个进行微调，让员工去处理这些工作*。
- **ASML 将训练定制模型**：一位成员建议，像 **ASML** 这样的公司凭借其雄厚的财力，完全有理由开发部分定制的预训练模型。
   - 他们强调了在没有通用目的限制的情况下，通过针对性训练模型来提升性能并替代人类工程师的潜力。
- **Mistral 的估值受到质疑**：一位成员认为，考虑到已有安全的闭源和开源替代方案，**Mistral 的 LLM** 在内部并不值 **13 亿美元**。
   - 他们推测 **Mistral 的估值** 看起来更像是 *政治人情*，而非实际的盈利能力。
- **X 算法已在 GitHub 上发布**：有人指出 **X 算法**（原 Twitter）在 [GitHub](https://github.com/twitter/the-algorithm) 上有了更新。
   - 未提供更多细节。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1414707826441457754)** (22 messages🔥): 

> `Aider vs Codex 上下文管理，LLM 提示词长度，AI 编程速度，SWE Bench，Roo/Cline vs Aider` 


- **Aider 在终端作为结对程序员表现出色**：一位成员表示 **Aider** 在*终端结对程序员*方面表现卓越，并强调对于模型训练中较少涉及的语言，**LSPs** 等功能以及驱动特定的命令式工具对 **MCP servers** 非常有价值。
   - 然而，他们建议当项目偏离 **Paul Gauthier** 关于人机协作的愿景时，**Aider** 用户可以创建个人 fork。
- **LLM 需要长提示词**：一位成员建议编写比最初认为的更长、更详细的提示词（如系统提示词的长度所示），以有效地引导 **LLM**；在进行单一类型的编辑后，如果没有冗长的提示词，**LLM** 的结果基本上只能靠运气。
   - 他们认为，只有在明确指示的情况下，**LLM** 才能有效地执行多文件、多目的的编辑。
- **AI 编程 10 倍速神话**：据一位成员称，AI 赋能编程可实现 *10 倍速* 的说法是一个神话，并建议在代码准确性和责任至关重要的背景下，**25-50%** 的提升是更现实的预期。
   - 他们认为 **LLM** 擅长自动化打字，但要获得切实且有用的输出，仍需要想象力和愿景。
- **Aider 正在实现 One-Shot**：一位用户在使用本地 **LLM** 进行实验时观察到，配合 **gpt-oss-120b** 的 **Aider** 能够 *One-Shot*（一次性完成）**Roo/Cline** 无法完成的任务，且速度快得多。
   - 他们表示 **repomap** 非常出色，但未对这一说法进行展开。
- **SWE Bench 对比**：一些成员分享了 SWE Bench 排行榜链接（[https://www.swebench.com/multilingual.html](https://www.swebench.com/multilingual.html) 和 [https://leaderboard.techfren.net/](https://leaderboard.techfren.net/)），以展示使用 **Aider** 作为测试框架的模型性能。
   - 有人指出 Techfren 排行榜缺少来自 gpt-oss 的基准测试。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1414976180473434378)** (3 messages): 

> `Gemini 错误，更改模型 API URL` 


- **Gemini 的 BadRequestError 错误**：一位成员报告今天早上在使用 **Gemini** 时遇到错误，具体为 **BadRequestError**。
   - 错误消息指出处理输入图像时出现问题，建议重试或在 [Generative AI 故障排除指南](https://developers.generativeai.google/guide/troubleshooting)中报告问题。
- **API URL 转换**：一位成员询问如何更改模型的 **API URL**。
   - 另一位成员提供了一个 [Stack Overflow 链接](https://stackoverflow.com/a/79518819/6090676)作为示例。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1414865864020197446)** (20 messages🔥): 

> `Manus 垃圾信息，Manus 网站错误，Manus 免费积分，Manus 推荐积分` 


- **Manus 垃圾信息发布者被踢出**：一位成员举报了一名垃圾信息发布者，管理员确认该用户已被警告且相关消息已删除。
   - 管理员表示：*请避免分享与 Manus 无关的链接。继续违规将导致被移出服务器。*
- **本地 Manus 网站测试问题**：一位成员报告他们的 Manus 网站仅输出了 **index.html**、**App.css** 和 **App.jsx** 文件，并请求协助测试该网站。
   - 聊天中未提供解决方案。
- **Manus 免费积分消失**：多位成员报告 Manus 每天不再发放 **300 个免费积分 Token**。
   - 他们提到已经等待了几天都没有收到积分。
- **推荐积分优惠码困惑**：一位成员询问在邀请他人后如何获得 **500 积分** 的推荐奖励。
   - 他们对*优惠码（promotion code）*的要求感到困惑。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1414705688969478267)** (9 messages🔥): 

> `Neel 访谈，AI/ML 爱好者介绍` 


- **新的 Neel 访谈发布**：一位成员分享了一个新的 [Neel 访谈](https://www.youtube.com/watch?v=5FdO1MEumbI)。
   - 该视频专注于 AI 系统和应用网络安全。
- **AI/ML 爱好者打招呼**：几位新成员介绍了自己，他们是具有软件工程、数据、后端工程、数学和网络安全背景的 AI/ML 爱好者。
   - 一位成员分享了他的 X (Twitter) 账号，他在那里撰写关于 ML/DL 的内容：[https://x.com/nerdybat369](https://x.com/nerdybat369)。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1414690404221194302)** (4 messages): 

> `6m Model, arxiv link` 


- **6m 模型表现良好**：一位成员在分享图片时表示 *"对于一个 6m 模型来说还不错"*，暗示该模型表现良好。
   - 分享的图片未被描述。
- **If Only Up Was Good**：一位成员分享了一个 [Arxiv 链接](https://arxiv.org/abs/2509.04154) 并评论道 *"如果 Only Up 表现好就好了"*。
   - 目前尚不清楚该链接具体指向什么。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1414740117876113579)** (1 messages): 

> `LM Eval Harness Calibration Scores, RL for Calibration, LM Eval Harness PR, Critical Take on Calibration Scores` 


- **LM Eval 考虑引入校准分数 (Calibration Scores)**：一位成员有兴趣将 **校准分数 (calibration scores)** 添加到 [LM eval harness](https://github.com/EleutherAI/lm-evaluation-harness) 中，以引导激励机制转向更可信的模型。
   - 该成员建议，这是一种引导激励机制以产生更可信模型的广泛方式。
- **RL 校准工作浮出水面**：一位成员提到了最近关于 **用于校准的 RL (RL for calibration)** 的工作，并附上了论文链接：[https://arxiv.org/pdf/2507.16806](https://arxiv.org/pdf/2507.16806)。
   - 未提供关于该论文的进一步信息。
- **过往 LM Eval Harness PR 重新引起关注**：一位成员提到了之前一个与 LM evaluation harness 校准分数相关的未成功 PR：[https://github.com/EleutherAI/lm-evaluation-harness/pull/874](https://github.com/EleutherAI/lm-evaluation-harness/pull/874)。
   - 未提供关于该 Pull Request 的进一步信息。
- **对校准分数的批判性看法**：一位成员通过 Twitter 链接分享了对校准分数的批判性观点：[https://x.com/_jasonwei/status/1871285864690815053](https://x.com/_jasonwei/status/1871285864690815053)。
   - 未提供关于该批判性看法的进一步信息。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1414746533403951134)** (3 messages): 

> `explicitcopies, moves, c binder, EmberJson` 


- **显式复制 (Explicit Copies) 进度需要更多 PR**：一位成员指出，由于会导致崩溃/段错误 (seg faults)，将所有内容切换为仅使用 **显式复制 + 移动 (explicit copies + moves)** 无法通过单个 PR 解决，需要拆分为更小的 PR。
- **择优挑选 (Cherry Pick) EmberJson**：一位成员提到，一旦 [modular/modular#5289](https://github.com/modular/modular/pull/5289) 合并，他们可能会将 [此提交 (commit)](https://github.com/bgreni/EmberJson/pull/53/commits/3039debad36fee5a7f1b6e034e1cb8fa344c4112) 择优挑选 (cherry pick) 到一个单独的 PR 中。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1414690953096200245)** (4 messages): 

> `Mojo test suite duration, Custom ops compilation issues` 


- **Mojo 💥 测试套件耗时激增**：在代码库中使用 Mojo 代码会导致测试套件运行时间激增，该问题在 [此 issue](https://github.com/modular/modular/issues/5293) 中跟踪。
   - 另一个问题是在多个进程中同时编译自定义算子 (Custom ops)，但该 Bug 很难复现。
- **自定义算子 (Custom Ops) 编写受阻 🛑**：一位成员报告称，由于 [此 issue](https://github.com/modular/modular/issues/5294)，编写自定义算子的工作受到阻碍。
   - 该成员正积极尝试复现该 Bug 以协助解决。