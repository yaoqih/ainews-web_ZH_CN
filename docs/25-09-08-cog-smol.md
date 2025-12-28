---
companies:
- cognition
- vercel
- meta-ai-fair
- alibaba
- groq
- huggingface
date: '2025-09-08T05:44:39.731046Z'
description: '**Cognition** 以 **102 亿美元**的估值融资 **4 亿美元**，用于推进 AI 编程智能体（coding agents），同时
  **swyx** 宣布加入该公司。**Vercel** 发布了一个开源（OSS）编程平台，该平台采用了经过调优的 **GPT-5** 智能体循环。**Kimi
  K2-0905** 模型在编程评估中获得了最高分，并凭借翻倍的上下文长度提升了智能体能力。**阿里巴巴**发布了 **Qwen3-ASR**，这是一款具备强大噪声处理能力的多语言转录模型。**Meta**
  推出了 Set Block Decoding 技术，在不改变架构的情况下，将解码速度提升了 3 至 5 倍。KV 缓存（KV cache）压缩和量化方面的创新备受关注，包括
  SGLang 中的 **AutoRound** 以及针对 Blackwell GPU 的 **QuTLASS v0.1.0**。**AlgoPerf v0.6**
  等算法基准测试工具也为了提高效率进行了更新。'
id: MjAyNS0w
models:
- kimi-k2-0905
- qwen3-asr
- gpt-5
people:
- swyx
title: Cognition 获 100 亿美元 C 轮融资；Smol AI 更新动态。
topics:
- coding-agents
- agent-development
- open-source
- model-evaluation
- multilingual-models
- inference-optimization
- kv-cache-compression
- quantization
- algorithmic-benchmarking
- context-length
- model-performance
---

**Smol AI 读者的特别更新。**

> 2025年9月5日至9月8日的 AI 新闻。我们为您检查了 12 个 subreddits、544 个 Twitter 账号和 22 个 Discord 社区（187 个频道，12661 条消息）。预计节省阅读时间（以 200wpm 计算）：1069 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以精美的 vibe coded 方式展示所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 向我们提供反馈！

正如 [7 月份泄露](https://news.smol.ai/issues/25-07-24-cogsurf-cursor) 的那样，Cognition 的 100 亿美元融资于 [今日宣布](https://x.com/cognition/status/1965086655821525280)。我们还宣布，我（[swyx）也将加入](https://x.com/swyx/status/1965183110016098617) Cognition，担任尚未确定的职位，而 [AI Engineer](https://apply.ai.engineer/) 和 Latent Space 将保持独立。AINews 将作为个人项目继续运行，目前正在就其稳定的未来进行一些对话。

---

# **AI Twitter 综述**

**编程 Agent 与工具链势头**

- **Cognition 融资 4 亿美元以扩展 Devin**：Cognition 宣布了一轮 4 亿美元的融资，投后估值为 102 亿美元，旨在“推进 AI 编程 Agent 的前沿”，由 Founders Fund 领投，Lux、8VC、Neo 等参投。团队强调了客户群的扩张以及 Windsurf 团队的加入，并正在招聘产品、基础设施和 post-training 方面的人才（[公告 1](https://twitter.com/cognition/status/1965086655821525280), [2](https://twitter.com/cognition/status/1965086662612177299), [团队说明](https://twitter.com/cognition/status/1965086661253185645), [计划片段](https://twitter.com/cognition/status/1965185627357683776)）。评论：@swyx 正在加入 Cognition，阐述了他为何“买入” Agent 实验室这一论点，以及跨同步/异步工作流的定位如何对“Agent 十年”中的主导地位至关重要（[推文串](https://twitter.com/swyx/status/1965183110016098617)）。
- **Agent 开发栈变得更简单且更强大**：
    - Vercel 发布了一个基于 Vercel AI SDK、Gateway、Sandbox 和经过调优的 GPT-5 Agent 循环（工具使用：文件 IO、命令、包安装、自动修复）构建的开源“vibe coding 平台”，并演示了用 Go 语言一次性编写出一款多人 Pong 游戏（[演示](https://twitter.com/rauchg/status/1964857952722133231)）。
    - Claude Code 的循环有意保持极简：单个主循环 + 异步缓冲区、直接工具和基于 TODO 的规划；在可调试性和可靠性方面，简单性胜过群组编排（swarm orchestration）（[分析](https://twitter.com/imjaredz/status/1965083721713041564)）。
    - 编程评估：Groq 上的 Kimi K2-0905 在 Roo Code 上达到了 94% 并排名第 7，成为第一个突破 90+ 的开源权重模型，同时也是前 10 名中最快、最便宜的（[排行榜](https://twitter.com/roo_code/status/1965098976677658630)）。Tim Dettmers 报告称，编程助手的实际前沿正日益向开源权重模型倾斜：GLM-4.5 仅需“3 美元/月”，且质量接近 Sonnet；Kimi K2.1 Turbo 与 Opus 4.1 相比，速度快约 3 倍，价格便宜约 7 倍，而 GPT-5 主要在复杂的规格说明工作中表现出色（[观点](https://twitter.com/Tim_Dettmers/status/1965021602267217972)）。

**模型与推理进展**

- **Kimi K2 0905 和 Qwen3-ASR**：
    - Kimi K2 0905（1T 参数，架构未变）提升了 Agent 能力：Terminal‑Bench Hard 从 14% 提升至 23%，Tau2‑Bench Telecom 从 61% 提升至 73%；上下文从 128k 翻倍至 256k。在 Artificial Analysis 的 AAII 指数上智力评分 +2；现已在 Kimi 官网提供服务（[摘要](https://twitter.com/ArtificialAnlys/status/1965010554499788841)，[实时笔记](https://twitter.com/crystalsssup/status/1965017719058960732)）。
    - 阿里巴巴的 Qwen3‑ASR 发布了一个用于多语言转录的单一模型（中英 + 9 种语言），支持自动检测，对背景音乐/噪音/说唱具有鲁棒性，WER < 8% 并支持自定义上下文偏置。ModelScope/HF 上提供 Demo；API 已上线（[发布](https://twitter.com/Alibaba_Qwen/status/1965068737297707261)）。
- **更快的解码和更轻量的 KV**：
    - Meta 的 Set Block Decoding (SBD) 在不改变架构的情况下，使现有 LM 的解码速度提升了 3–5 倍，匹配 NTP 性能并保留精确的 KV cache —— 通过掩码/离散扩散公式实现并行生成（[概览](https://twitter.com/HuggingPapers/status/1965084731839513059)，[详情](https://twitter.com/itai_gat/status/1965112129499046230)）。
    - KV cache 和量化创新：AutoRound 现已集成到 SGLang（[PR](https://twitter.com/HaihaoShen/status/1964926924880523701)），Turing Post 调研了 KV 压缩技术（量化、低秩、Slim Attention、XQuant）及其权衡（[推文](https://twitter.com/TheTuringPost/status/1964971207188791464)），QuTLASS v0.1.0 为 Blackwell GPU 带来了 4‑bit NVFP4 微缩放（microscaling）和快速变换（[发布](https://twitter.com/DAlistarh/status/1965157635617087885)）。AlgoPerf v0.6 增加了滚动排行榜、JAX jit，并降低了算法基准测试的计算成本（[更新](https://twitter.com/algoperf/status/1965044626626342993)）；HF 记录了 PyTorch 的 ZeroGPU AOT 编译内部机制（[博客](https://twitter.com/charlesbben/status/1965046090945954104)）。

**多模态生成、视频和 “Vibe Coding”**

- **Veo 3 正式商用且价格更低**：Google 的 Veo 3 和 Veo 3 Fast 现已在 Gemini API 中正式商用（GA），降价约 50%（0.40 美元/秒和 0.15 美元/秒），支持 1080p 输出和 9:16 垂直视频 —— 定位于规模化生产（[开发者博客](https://twitter.com/googleaidevs/status/1965160822260318702)，[价格详情](https://twitter.com/_philschmid/status/1965161626761326983)，[PM 笔记](https://twitter.com/OfficialLoganK/status/1965193765146296467)）。
- **社区工作流和工具**：
    - “Nano Banana”（Gemini 2.5 Flash Image Preview）催生了一个周末的 “vibe‑coded” 项目 —— 现已在 Google AI Studio 中开源供二次创作；团队报告了 1 键复用和有趣的细节（例如，时钟总是渲染为 10:10）（[开源包](https://twitter.com/arrakis_ai/status/1965001417716072877)，[趣闻](https://twitter.com/fabianstelzer/status/1965001753059057925)）。
    - Qwen 的 “论文 → 网站” 流程可在几分钟内将研究论文转换为可部署的网站（[Demo](https://twitter.com/Alibaba_Qwen/status/1964870508421480524)）。Lmarena 增加了多轮图像编辑评估，以便社区比较不同模型（包括 “nano banana”）的迭代优化能力（[功能](https://twitter.com/lmarena_ai/status/1965150440401809436)）。对于文档 RAG 用户体验，ColQwen2 + Weaviate 驱动了用于视觉 PDF 搜索和区块高亮的逐 Token 相似度图（[构建](https://twitter.com/helloiamleonie/status/1964997028875743637)）。

**Agent、训练后 RL 和评估实践**

- **迈向迭代自我改进**：FAIR 的 Exploratory Iteration (ExIt) 通过自动课程训练 LLM 进行推理侧自我改进，该课程从模型自身的先前响应中进行引导（bootstrap），并优先处理 GRPO 组中具有高回报方差的部分历史记录。ExIt 在竞赛数学、BFCLv3 多轮任务和 MLE-bench (+22%) 上的表现优于 GRPO，且仅需训练单步改进 ([thread](https://twitter.com/MinqiJiang/status/1965055909605916892))。
- **在线 vs 离线 RL 及评估**：
    - 证据持续表明，在大规模情况下，在线 RL (PPO/GRPO) 的性能优于 DPO 等离线方法，尽管半在线迭代（策略内采样 + 负梯度）缩小了这一差距；数据质量仍然是算法选择的主导因素 ([summary](https://twitter.com/cwolferesearch/status/1965088925510520853))。
    - 为什么许多 “Agent” 表现不及预期：与生成式任务相比，决策制定的容错率接近于零且数据稀疏；大多数失败源于粗略的任务范围界定和非结构化环境，而非 LLM 的缺陷 ([debate recap](https://twitter.com/ZhihuFrontier/status/1964928650081698167))。
    - RAG 评估正从“死”的单元测试转向“活”的循环：RAGGY（开源 REPL）为 RAG 实现了假设性（what-if）迭代，目前正强力推动将预生产测试与生产环境的可观测性和人工审核相结合，而不是将它们视为独立的孤岛 ([RAGGY](https://twitter.com/HamelHusain/status/1965052554997600449), [evals take](https://twitter.com/bnicholehopkins/status/1965130607790264452))。另请参阅利用工具使用和多步推理的实用 “Agentic RAG” 架构 ([guide](https://twitter.com/omarsar0/status/1965115682322042954))。

**机器人与具身智能 (Embodied AI)**

- **通过 RL 进行多机器人规划**：Google DeepMind 的 RoboBallet（与 Intrinsic 和 UCL 合作）为多达 8 个机器人手臂编排无碰撞的任务和运动规划，性能优于传统方法约 25%，并能通过 RL 学习到的协调原则在几秒钟内泛化到新的工作流 ([announcement](https://twitter.com/GoogleDeepMind/status/1965040645103407572), [more](https://twitter.com/GoogleDeepMind/status/1965040648400351337))。
- **开源硬件栈与灵巧操作**：Pollen Robotics 为 Reachy 2 配备了双开源 “Amazing Hand” 夹持器以进行精细操作；原生集成即将推出 ([demo](https://twitter.com/pollenrobotics/status/1964987735829266871))。X Square 发布了 WALL-OSS（开源基础模型）和具备自动拖地功能及灵巧手的 Quanta X2 机器人；阿里云领投了 1.4 亿美元的 A+ 轮融资（不到 2 年内融资超过 2.8 亿美元） ([summary](https://twitter.com/ZhihuFrontier/status/1964968113990164810))。OpenPI 的 pi-05 现已进入 openpi 并支持 PyTorch ([release](https://twitter.com/svlevine/status/1965161524722630734))。

**基准测试、排行榜与企业动态**

- **文本排行榜变动**：lmarena 在其 Top 10 文本排行榜中新增了两个条目：Qwen3-max-preview（第 6 名，私有）和 Kimi-K2-0905-preview（第 8 名，修改版 MIT），使 Kimi 与 Qwen 和 DeepSeek 变体一同竞争顶级开源权重模型 ([update](https://twitter.com/lmarena_ai/status/1965115050273976703), [model link](https://twitter.com/lmarena_ai/status/1965124408097517853))。Artificial Analysis 对 K2-0905 的测量结果也反映了其 Agent 性能的提升 ([details](https://twitter.com/ArtificialAnlys/status/1965010554499788841))。
- **政府与企业**：
    - Perplexity 推出了 “Perplexity for Government”：默认安全、零数据使用、高级模型访问权限且无需企业合同；同时将 Perplexity Finance 引入 iOS/Android 平台 ([launch](https://twitter.com/perplexity_ai/status/1965030156415980009), [follow‑up](https://twitter.com/AravSrinivas/status/1965032305053065590), [finance mobile](https://twitter.com/AravSrinivas/status/1965100159488196757))。
    - Anthropic 表示支持加利福尼亚州 SB 53 法案（参议员 Scott Wiener 提出），这是一个专注于透明度的州级框架，旨在联邦标准缺失的情况下治理前沿 AI ([statement](https://twitter.com/AnthropicAI/status/1965027311717388673), [context](https://twitter.com/jackclarkSF/status/1965048896784367847))。

热门推文（按互动量排序）

- Cognition 以 102 亿美元估值融资 4 亿美元，用于扩展 AI coding agents ([公告](https://twitter.com/cognition/status/1965086655821525280))
- Vercel 的开源 vibe coding 平台配合调优后的 GPT-5 循环，一次性（one-shots）用 Go 语言生成了一个多人 Pong 游戏 ([演示](https://twitter.com/rauchg/status/1964857952722133231))
- Qwen3-ASR：支持多语言 ASR 的单一模型，WER < 8%，对噪声/背景音乐（BGM）具有鲁棒性，支持上下文注入（context injection） ([发布](https://twitter.com/Alibaba_Qwen/status/1965068737297707261))
- Google AI Mode 扩展至印地语、印尼语、日语、韩语和巴西葡萄牙语 ([Sundar Pichai](https://twitter.com/sundarpichai/status/1965115123330388467))
- Veo 3 正式发布（GA），价格下调约 50%，Gemini API 支持 1080p 和竖屏视频 ([开发者更新](https://twitter.com/googleaidevs/status/1965160822260318702))

---

# **AI Reddit 回顾**

## **/r/LocalLlama + /r/localLLM 回顾**

**1. 开源 LLM 发布：K2 Think 和 TildeOpen 30B 多语言模型**

- [**阿联酋准备发布 K2 Think，“世界上最先进的开源推理模型”**](https://www.wam.ae/en/article/bll7llv-recognition-sheikh-khalifa%E2%80%99s-contribution) ([评分: 217, 评论: 70](https://www.reddit.com/r/LocalLLaMA/comments/1nbo33p/uae_preparing_to_launch_k2_think_the_worlds_most/)): **MBZUAI ([官网](https://mbzuai.ac.ae/)) 和 G42 ([官网](https://g42.ai/)) 预告即将发布 “K2 Think”，被描述为一个开源推理模型，在较小的占用空间内提供 “frontier-class” 的性能，据称可以匹配或超越比其大约 **`10×` **的模型。未披露具体规格（参数量、架构、上下文窗口、tokenizer、训练数据、训练算力或评估套件）；公告称将在“未来一周”发布。该项目声明与 Moonshot/Kimi 的 “K2” 无关，此前 2024 年发布的 “K2” 65B 模型据报道接近 Meta [Llama 2 70B](https://ai.meta.com/llama/) 的复现。** 评论者注意到与 Moonshot/Kimi 的 K2 命名混淆，并在基准测试（benchmarks）出炉前表示怀疑（例如，“眼见为实”），同时强调缺乏参数规模等基本规格。
    - 模型身份/命名澄清：评论者指出此 “K2 Think” 与 Moonshot/Kimi 的 “K2” 无关，且该阿联酋团队此前发布的 65B “K2”（2024年）实际上是 **Meta Llama 2 70B** 的复现（参见 Llama 2 论文：https://arxiv.org/abs/2307.09288）。重复使用 “K2” 绰号存在混淆不同项目并夸大相对于 Llama 衍生基准的创新性的风险。
    - 缺失技术规格：未披露参数量（“没提到参数大小？”），使得 “世界上最先进的开源推理模型” 的说法无法评估。技术读者期待参数量、架构（dense vs MoE）、上下文长度、训练算力/数据集以及带有透明评估设置的推理基准测试（如 GSM8K, MATH, AIME, BBH, ARC-C）等细节；缺乏这些，怀疑是合理的。
- [**Tilde AI 发布 TildeOpen LLM：一个拥有超过 300 亿参数、支持大多数欧洲语言的开源大语言模型**](https://huggingface.co/TildeAI/TildeOpen-30b) ([评分: 173, 评论: 41](https://www.reddit.com/r/LocalLLaMA/comments/1nbi95c/tilde_ai_releases_tildeopen_llm_an_opensource/)): [**Tilde.ai](http://tilde.ai/) 发布了 [TildeOpen-30B](https://huggingface.co/TildeAI/TildeOpen-30b)，这是一个开源的** `~30B` **参数 dense decoder-only transformer，针对代表性不足的北欧/东欧语言，在 LUMI 超级计算机（768 块 AMD MI250X GPU）上训练，进行了** `450k` **次更新，全局 batch 为** `4,718,592` **tokens（约** `2.12T` **tokens，恒定 LR + cooldown），使用三阶段采样课程（均匀 → 自然 → 均匀）和公平的 tokenizer 以平衡低资源语言。架构：60 层，**`d_model=6144`**，**`n_heads=48` **带有 **`8` **个 KV heads (GQA)，**`d_ff=21,504`**，RoPE，SwiGLU，RMSNorm，**`context=8192`**；采用 CC-BY-4.0 许可，未进行指令微调（instruction-tuned）/安全对齐（safety-aligned），提供 [GGUF 量化版本](https://huggingface.co/mradermacher/TildeOpen-30b-GGUF)。他们报告了在 WMT24++ 重点语言上强大的字符级困惑度（perplexity），通常与 EuroLLM/ALIA 和 Google 的 Gemma 2 相比具有竞争力，并计划在此基础上构建专门的翻译模型。** 评论指出缺乏 demo/playground，质疑总 tokens 是否约为 `4.1T`（根据提供的 batch 和 steps 计算建议约为 `2.12T`），并批评仅报告了针对窄基准集的困惑度——认为困惑度很大程度上取决于数据混合，可能无法预测下游质量（一些人预计像 Qwen3 这样的多语言模型在许多语言上表现会更好）。

- 训练计算说明：根据 `450,000` 次更新和每步 `4,718,592` tokens 的全局 batch，隐含的 token 总数约为 `≈ 2,123,366,400,000` (~2.12T)，而非 4.1T。“在 2 万亿 tokens 上采用恒定学习率，随后进入冷却阶段”的表述很可能描述的是约 2T tokens 的 LR schedule；此后的任何冷却阶段对总量的增加都非常有限，不会使其翻倍。
- 评估方面的疑虑：[HF card](https://huggingface.co/TildeAI/TildeOpen-30b) 仅报告了相对于 Gemma 2、EuroLLM 和 ALIA 的 perplexity，这并非强有力的证据，因为 perplexity 严重依赖于训练数据分布，且与下游任务质量的相关性较差。对于一个多语言基座模型，读者期望看到标准化的多语言基准测试（例如 FLORES-200 翻译、XQuAD/TyDiQA、MGSM、MMLU）以及更广泛的 baselines（例如 Qwen/Qwen2.5、Aya、mT5、XLM-R）；如果没有这些，与 Gwen3（在 119 种语言上训练）等模型的对比就难以证实。
- 模型类型：这是一个基座（非指令微调）模型，因此在没有额外对齐（SFT/DPO/RLHF）的情况下，它无法可靠地进行“聊天”；它表现为原始的下一 token 生成，而非指令遵循。缺乏演示聊天 UI 与此一致；它可以在支持的语言中生成流畅的文本，但需要指令微调或聊天适配器才能像对话助手一样工作。

**2. 在个人硬件（双 RTX 6000 + M3 Mac）上进行本地/离线 LLM 使用**

- [**双 RTX 6000 配置的最后完善**](https://i.redd.it/sez83piasvnf1.jpeg) ([Score: 280, Comments: 129](https://www.reddit.com/r/LocalLLaMA/comments/1nbfy60/finishing_touches_on_dual_rtx_6000_build/)): **楼主展示了一台配备双 NVIDIA RTX 6000 显卡（声称总计约 192 GB VRAM）和 128 GB 系统 RAM 的工作站，用于运行本地 LLM（例如 4-bit 的 Qwen 2 35B）。主要的性能担忧是典型 120V/15A 家庭电路的功率（最大约 1.8 kW，持续约 1.44 kW）；评论者建议将每张 GPU 的功耗限制在约 300 W（RTX 6000 的 TGP 约为 300 W），以避免跳闸，性能损失仅约 10%，或者升级电路。** 评论者估计显卡成本约为 1.6 万美元，询问了 CPU 的选择，并批评了这种专业级配置使用花哨的“游戏玩家”机箱美学。
    - 建议将双 RTX 6000 的功耗限制在 `~300 W`，以在仅损失约 `~10%` 性能的情况下控制发热和噪音。这符合 RTX 6000 Ada 的 `300 W` TBP 规格（[NVIDIA](https://www.nvidia.com/en-us/design-visualization/rtx-6000/)），并反映了常见的 Ada 欠压/降功耗调优，即性能在 300 W 以上呈次线性增长，使其成为密集型工作站构建中的务实权衡。
    - 一位评论者指出没有配备“满血 `192 GB`”系统 RAM 令人惊讶，认为更多 RAM 可以改善数据密集型工作流的缓存行为。对于双 `48 GB` VRAM GPU，充足的系统内存可以显著扩大 OS 页面/文件缓存，并减少暂存大型数据集、模型或纹理时的 I/O 停顿，在某些流水线中，这可能比原始 GPU TFLOPs 限制更大。
- [**末日场景：如果互联网断开前你只能下载一个 LLM，你会选哪一个？**](https://www.reddit.com/r/LocalLLaMA/comments/1nbgosx/apocalyptic_scenario_if_you_could_download_only/) ([Score: 249, Comments: 230](https://www.reddit.com/r/LocalLLaMA/comments/1nbgosx/apocalyptic_scenario_if_you_could_download_only/)): **楼主询问在 Mac Studio（Apple M3，`512 GB` RAM）上完全离线使用的单个本地 LLM 选择。首选方案：(1) 来自智谱 AI 的 GLM 4.5 (Air 版本)，因其强大的通用代码/脚本编写能力、对适度算力的耐受性以及对 RAM 友好；参见 THUDM/智谱 AI 的 GLM 系列 [模型](https://huggingface.co/THUDM)。(2) Qwen3 30B “thinking” 配合向量数据库 (RAG) 中的离线 Wikipedia 转储，以获得广泛的知识覆盖；参见 Qwen [模型](https://huggingface.co/Qwen) 以及使用类似 [FAISS](https://github.com/facebookresearch/faiss) 的 ANN 存储配合 [Wikipedia 转储](https://dumps.wikimedia.org/)。一个警告是避开 “GPT-OSS”，因为存在感知的安全问题。** 评论者更倾向于选择 GLM 以获得在仅 CPU 现场硬件上的务实代码/运维可靠性，而选择 Qwen3 30B + RAG 以获得知识广度；关于模型安全性存在分歧，并对 “GPT-OSS” 提出了警告。
    - **GLM 4.5 (Air)** 因其在仅 CPU 笔记本电脑上的强大离线实用性而受到关注，这得益于其适中的内存占用以及在脚本/系统任务中的稳定性。虽然它不擅长长篇写作，但用户反馈它能可靠地生成 bash 脚本，并协助在现场环境（无 GPU）下进行故障排除，只要有足够的 RAM。
    - 建议将 **Qwen3 30B Thinking** 与本地存储在向量数据库中的 Wikipedia 配对，以最大化离线知识广度：模型处理推理，而 RAG 提供事实检索。此设置需要预计算 Embedding 并对 Wikipedia 建立索引，以存储空间/CPU 换取更高的检索质量和互联网独立性。
    - 在严格的能源预算下，更倾向于使用 **Qwen 30B A3B** + 下载的 Wikipedia RAG，强调“尽可能少的激活参数”以最小化功耗。这种方法倾向于计算效率（例如稀疏或减少的激活参数），而非大型稠密模型，旨在受限电力下实现更长的运行时间，同时不牺牲核心推理能力。

## **非技术性 AI 子版块回顾**

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

**1. AlterEgo 可穿戴设备、Gemini “上传任何文件”以及 Qwen Edit LORA 发布**

- [**介绍 Alterego：全球首款近乎“心灵感应”的穿戴设备，实现思绪速度的无声通信**](https://v.redd.it/5a9hu9antznf1) ([Score: 334, Comments: 143](https://www.reddit.com/r/singularity/comments/1nbxgri/introducing_alterego_the_worlds_first/))：**该帖子链接到了 “Alterego” 的发布公告，其被宣传为一种“近乎心灵感应”的穿戴设备，能够以“思绪的速度”进行无声通信（[推文](https://x.com/alterego_io/status/1965113585299849535)），但未提供技术规格、模态或基准测试；相关的 Reddit 视频无法访问（HTTP 403），因此无法验证演示。这一主张与之前使用 sEMG 的非侵入性无声语音接口高度重合（例如，MIT Media Lab 的 AlterEgo 报告在** `20-word` **词汇量上达到约** `92%` **的准确率 [MIT News](https://news.mit.edu/2018/altereego-device-transcribes-words-people-say-in-their-heads-0404)）以及手腕 EMG 解码器（例如，Meta/CTRL-Labs [综述](https://tech.facebook.com/reality-labs/2021/03/ar-wristband-haptic-emg/））；以对话速率进行的真实“思维”解码仍局限于受限范式或侵入式 BCI，而非商用穿戴设备。** 热门评论对其合法性表示怀疑，认为这读起来像讽刺作品或“VC 资金陷阱”，并指出其缺乏技术细节/演示，且将默读 EMG 解码与字面意义上的内部独白“心灵感应”混为一谈。
    - 几位评论者指出网站上没有任何技术描述，并要求提供具体规格：使用了何种传感模态（例如，沿下颌/喉部的表面 EMG 与 EEG/超声波）、采样率、设备端还是手机/云端推理、模型架构/大小、校准时间、电池寿命/功耗，以及词错率 (WER)、端到端延迟和词汇限制等客观指标。他们引用了诸如 **MIT Media Lab 的 AlterEgo**（sEMG，有限词汇量，约 `20` 个单词，准确率约 `92%`）之类的现有技术作为对比基准，并要求发布白皮书或数据集/代码以供验证（[MIT 项目](https://www.media.mit.edu/projects/alterego/overview/)）。
    - 怀疑者对“近乎心灵感应”和“思绪速度”的主张提出挑战，指出非侵入性无声语音系统通常表现出明显的延迟（`~100–300 ms+`），且在开放词汇任务上性能急剧下降；稳健的性能通常需要受限的词典或针对特定用户的校准。他们呼吁进行严格的基准测试：预注册、随机的现场演示及盲测提示，报告 WER、每分钟字符数、延迟分布、词汇外处理和跨用户泛化能力，并与之前的 sEMG/EEG 和超声波唇语模型进行对比。
    - 对潜在“VC 陷阱”的担忧源于缺乏同行评审结果或第三方评估；评论者希望看到独立复制和压力测试。建议的证明点包括：消融研究（传感器数量/位置）、对运动/汗水/噪声的鲁棒性、多说话人适应性以及失败模式分析；如果没有这些，这些主张被视为营销手段而非工程证据。
- [**我们现在可以向 Gemini App 上传任何文件了！！甚至包括音频！**](https://i.redd.it/3ap81kmdeynf1.png) ([Score: 256, Comments: 24](https://www.reddit.com/r/Bard/comments/1nbpp24/we_can_upload_any_file_to_gemini_app_now_even/))：**该帖子宣布 Gemini 移动端 App 现在支持直接在应用内上传任意文件（包括音频）。这使得 App 的 UI 与 Gemini API（已支持约** `2` **年）和 Google AI Studio（自 Gemini** `2.5 Pro` **起支持）的现有功能持平，表明这只是界面推送而非新的模型能力；参见 [Josh Woodward](https://x.com/joshwoodward/status/1965057589718499756) 的确认。** 评论者指出，这种延迟是由于 App 的 UI 造成的，而非模型限制，并对尽管 API/AI Studio 长期支持但 App 却花了这么长时间才上线感到惊讶。
    - 评论者指出，该功能在模型/API 层面并不新鲜：Gemini API 接受任意文件上传已有约 2 年时间，而 **AI Studio** 自 **Gemini 2.5 Pro** 发布以来就已支持这些文件类型。延迟归因于移动端/消费者 App 的 UI，而非模型限制。外部确认/公告引用：https://x.com/joshwoodward/status/1965057589718499756?t=Axnh1CAMsFECFp4eMnRbBg&s=19。
    - 多位用户报告了与模型无关的 App 稳定性问题：聊天经常加载失败并显示“无法加载聊天”错误，且会话可能会被错误锁定，并显示消息称自定义 Gem 已被删除。这些问题迫使在活跃使用期间频繁重启 App，表明客户端会话/状态管理或缓存 Bug 降低了可用性，尽管新增了文件上传功能。

- 原生音频文件上传现在支持在应用内直接进行即时转录，与 API 已有的功能保持一致。这减少了快速音频转文本工作流对外部工具的需求，并使消费级应用在音频处理能力上与面向开发者的界面实现了对齐。
- [**虚拟试衣（服装迁移） - Qwen Edit Loraa**](https://www.reddit.com/gallery/1nbzh2d) ([Score: 216, Comments: 29](https://www.reddit.com/r/StableDiffusion/comments/1nbzh2d/clothes_try_on_clothing_transfer_qwen_edit_loraa/)): **发布了一个针对 Qwen Image Edit 的试衣 LoRA，支持服装迁移，同时保留主体身份并匹配多种艺术风格/体型。资源包括：[Patreon 博客](https://www.patreon.com/posts/138311408)、[CivitAI 模型](https://civitai.com/models/1940532?modelVersionId=2196278) 以及配套的 [Clothing Extractor LoRA](https://civitai.com/models/1940557?modelVersionId=2196307)。已知局限：明显的质量下降（可能源于 Qwen Edit 流水线和/或低分辨率训练数据）；作者建议使用 SeedVR2 进行放大，并计划在下一版本中进行** `higher-resolution` **（更高分辨率）的重新训练；商业可用性待定。** 评论者强调了基于开源许可、非蒸馏（non-distilled）基础模型（与 Flux Kontext 形成对比）的价值，并讨论了行业影响——声称原本耗资 `~$20k` 的拍摄成本通过此工作流可降至 `~$200`。一些用户要求提供更清晰的链接以便立即测试。
    - 评论者强调，使用 **开源许可、非蒸馏** 的基础模型（相比于 **Flux Kontext** 等封闭系统）允许社区进行 LoRA 微调和重新分发，而不受黑盒限制。非蒸馏权重通常能保留细粒度的可编辑性并避免蒸馏伪影，这对于服装迁移的保真度（如面料纹理、接缝）至关重要。这种开放性提高了可复现性，并使迭代 LoRA 等编辑适配器（Edit Adapters）变得更加容易（[LoRA 论文](https://arxiv.org/abs/2106.09685)）。
    - 一个实际模型指出，这对生产成本/时间有巨大的影响：原本耗资约 `$20,000` 的时尚拍摄，在单人操作下几小时内即可降至约 `$200`——成本降低了约 `100x`，且吞吐量大幅提升。这有效地将多人的流水线（摄影师、灯光、造型师、妆造）替换为推理/编辑工作流，将支出从物流转向算力。
    - 质量和数据要求：用户报告纹理保真度和风格一致性较之前的演示有所提高（例如，“纹理非常到位”），但也询问服装是否必须在白色背景上——这暗示了对简单分割/抠图（segmentation/matting）进行服装分离的依赖。如果是这样，高对比度或白色背景会简化遮罩处理，而复杂背景可能需要强大的分割/抠图技术（如 [SAM](https://segment-anything.com/)）来保持边缘细节并避免溢色。
- [**OpenAI 助力制作将于 2026 年上映的 AI 生成长篇动画电影**](https://i.redd.it/50fm7mb3cvnf1.jpeg) ([Score: 591, Comments: 165](https://www.reddit.com/r/singularity/comments/1nbebg0/openai_helping_to_make_an_ai_generated_feature/)): **帖子分享了 OpenAI 正在合作开发一部定于 2026 年上映的 AI 生成长篇动画电影。技术意义在于推动动画的端到端生成流水线（AI 辅助的预可视化/分镜、场景/镜头生成、配音/音效及后期），以及此类流水线在规模和成本上能否达到传统 CG 流水线的院线级质量。图片似乎是一个标题/公告；评论提到** `~$30M` **的预算，暗示重金投入在算力、模型开发、数据/授权和人工打磨上，而非完全的“一键生成”。[图片链接](https://i.redd.it/50fm7mb3cvnf1.jpeg)。** 评论者对报道的 3000 万美元成本表示质疑并预测会遭到抵制，而其他人则认为模型进步的速度可能会让 2026 年的制作显得过时，暗示在制作后期可能需要重新调整工具或重新渲染。
    - 关于预算/规模的讨论集中在为什么 AI 辅助的长篇动画需要耗资 `~$30M`。一部 90 分钟、24fps 的电影约有 `129,600` 帧；利用 Diffusion 或图生图（image-to-image）实现镜头/场景连贯性通常需要多阶段生成（关键帧、局部重绘/外扩绘制、ControlNet、超分辨率）以及繁重的后期（清理、转描、合成），这使得纯逐帧生成的方法不可行。因此，大部分成本可能来自构建强大的混合 CG 流水线（工具工程、数据集策划/版权、艺术家工时、剪辑、调色、声音）以及大规模获取/运行算力，而不仅仅是原始的 GPU 运行时间。

- 多位评论者指出，到 2026 年，底层模型可能会过时，从而在跨年度的流水线中引发可复现性/一致性问题。技术缓解措施包括：模型固定（model pinning）和版本化检查点（versioned checkpoints）、确定性解码/种子控制（deterministic decoding/seed control）、用于时间一致性的光流引导 image-to-image、ControlNet/姿态/几何条件控制、LoRA 风格适配器、潜空间缓存（latent caching），以及维护自托管备选方案（open weights）以避免 API 模型漂移或弃用。
    - 引用先验案例：“Critterz” 将 **OpenAI DALL·E** 的输出与传统动画相结合，提出了一种混合工作流：生成式模型提供概念/背景/关键帧，而传统的 2D/3D 动画处理运动/一致性。链接：Variety 的报道和项目背景 [Variety](https://variety.com/2025/film/global/paddington-in-peru-writers-ai-animated-film-critterz-1236328515/) 以及短片本身 [YouTube](https://youtu.be/-qdx6VBJHBU?feature=shared)。这表明 2026 年的长片可能依赖于受控的 I2I/inpainting 和合成（compositing），而非端到端的 video diffusion，以流水线/工具链的复杂性换取原始推理成本的降低。
- [**哇... 我们已经烧了 6 个月的钱了**](https://www.reddit.com/r/OpenAI/comments/1nbtl2p/wow_weve_been_burning_money_for_6_months/) ([Score: 524, Comments: 163](https://www.reddit.com/r/OpenAI/comments/1nbtl2p/wow_weve_been_burning_money_for_6_months/)): **OP 审计了 OpenAI API 的使用情况，发现他们每月支付约 1,200 美元，使用 GPT-4 处理琐碎的文本工具（从电子邮件中提取电话、违禁词检测、JSON 格式重排和字母大写转换）。在将这些调用切换到 [GPT-4o-mini](https://platform.openai.com/docs/models/gpt-4o-mini) 后，输出保持不变，每月支出降至** `~$200` **（减少约 83%）。其中许多用例可以利用更便宜的选择，如 [Moderation API](https://platform.openai.com/docs/guides/moderation) 以及根据 [OpenAI pricing](https://openai.com/api/pricing) 选用的低成本模型。** 评论强调应根据组织规模使支出正常化；建议避免使用 GPT-4，转而使用 4o/4o-mini（并声称最新的 “5-series” 模型更便宜且性能更强），对非延迟敏感型任务使用更便宜/更慢的层级，并利用免费的 [Moderation API](https://platform.openai.com/docs/guides/moderation) 进行毒性检查。
    - 通过模型选择和层级进行成本优化：评论者认为，日常任务很少需要支付 **GPT-4** 的费用——使用 **GPT-4o** 或 **GPT-4o-mini** 即可，有人声称最新的 “5-series” 更便宜且性能更强。他们还建议对非时间敏感型工作负载使用更便宜/低优先级或 batch 服务层级，并利用免费的 **Moderation** 端点来削减支出。参考资料：OpenAI 模型/定价和审核文档 (https://platform.openai.com/docs/models, https://platform.openai.com/docs/guides/moderation)。
    - 实际路由和确定性：OP 详细说明了从“所有事情都调用 **GPT-4**”（甚至包括大写转换）到使用 regex/基础 Python 进行确定性转换、使用 **gpt-4o-mini** 处理简单任务、仅将 **GPT-4** 用于复杂推理的转变。报告结果：在输出质量相同的情况下，成本降低了约 `85%`，这强调了将任务复杂度与最小能力模型相匹配以及尽可能优先使用确定性代码的价值。
- [**wan2.2+qwen-image**](https://v.redd.it/gbzs3m17qtnf1) ([Score: 203, Comments: 15](https://www.reddit.com/r/StableDiffusion/comments/1nb7ole/wan22qwenimage/)): **OP 展示了标记为 “wan2.2 + qwen-image” 的图像生成，称唯一的提示词是 “isometric”，暗示该流水线能够进行干净的等轴测渲染。媒体托管在 Reddit 视频上 ([v.redd.it/gbzs3m17qtnf1](https://v.redd.it/gbzs3m17qtnf1))，但根据 Reddit 的网络安全政策，访问被阻止（HTTP 403），因此除了名称之外，看不到任何参数（seed, sampler, CFG, steps）或模型细节。** 评论主要是赞扬；一个技术相关的问题询问了在不同图像间保持一致角色生成的方法，但 OP 未提供细节。
    - 几位评论者询问了在 `wan2.2` + `qwen-image` 流水线中实现跨场景角色身份一致性的确切方法。他们特别想知道身份持久性是来自 prompt engineering，还是使用了任何参考/调节机制（例如固定 seeds 或参考引导输入）来保持帧与帧之间的特征稳定。
    - 有一个尖锐的问题是关于是否使用了除所谓的加速适配器之外的任何 LoRAs，隐含地将 `LCM-LoRA`（用于减少采样步骤）与嵌入角色特征的身份/风格 LoRAs 进行对比。要求澄清角色/风格 LoRAs 或其他 fine-tuned 适配器是否在实现该视觉效果中发挥了作用。

- [**我刚才正在生成一些图片，结果发生了这个……**](https://i.redd.it/e8gt8akdlxnf1.jpeg) ([评分: 1748, 评论: 173](https://www.reddit.com/r/ChatGPT/comments/1nblv6s/i_was_just_generating_some_images_this_happened/)): **截图显示，在生成图片过程中，一个内部 System Prompt/指令（例如，“生成图片后不要继续输出任何文本”）泄露到了用户可见的对话中。这指向了编排层（orchestration layer）的一个 Prompt/系统消息泄露 Bug：UI/Agent 可能会注入一个隐藏指令来抑制工具调用后的文本，但处理错误导致其原样显示——揭示了后端通过自然语言 System Prompt 进行的工具使用控制。技术评论确认这是一个已知问题，即图片生成后的抑制指令偶尔会出现在聊天中，表明系统消息与用户 Assistant 消息未能正确隔离。** 评论者普遍认为这只是一个 Bug，而非预期行为，并指出这暴露了工程师如何依赖自然语言 System Prompt（通常措辞礼貌）来引导模型——这引发了对基于 Prompt 控制的鲁棒性的轻微担忧。
    - 几位评论者识别出了泄露的系统指令——例如，*“please end this turn now”*——该指令在图片工具调用后本应保持隐藏。这指向了一个编排/序列化 Bug，即 System Prompt 或工具协议的轮次结束指令出现在了聊天中，而不是 Assistant 在输出图片/工具结果后干净利落地停止（类似于工具使用中的函数调用/“结束轮次”流；参见 OpenAI 函数调用概念：https://platform.openai.com/docs/guides/function-calling）。
    - 用户报告在图片生成前有冗余的预检澄清（例如，对于简单的“池塘里的鸭子”重复询问细节），甚至在被要求继续时也不执行。从技术上讲，这表明 Prompt 模板或策略层优先考虑消歧/安全启发式算法而非用户指令，导致模型由于指令层级（System > Developer > User）和/或过度重视谨慎和完整性的 RLHF 奖励塑造而不断寻求确认。
    - 自然语言控制短语（如 *“please”*）的存在意味着对自然语言 System Prompt 的依赖，而非鲁棒的结构化控制信号（例如，显式的 `tool_calls` + `end_turn`/结束标志）。此类设计对于泄露和解析错误非常脆弱；结构化的 API 级停止/轮次标记通常会减少这些内部指令出现在用户可见输出中的几率（参见函数调用 API 中的工具/Assistant 轮次边界：https://platform.openai.com/docs/guides/function-calling）。

**2. AI 社会影响：安圭拉 .ai 意外之财、Hinton 不平等警告、Grok Imagine 成人内容差距**

- [**一个加勒比小岛如何意外成为 AI 热潮的最大赢家**](https://www.reddit.com/r/OpenAI/comments/1nbi70s/how_a_tiny_caribbean_island_accidentally_became/) ([评分: 1532, 评论: 102](https://www.reddit.com/r/OpenAI/comments/1nbi70s/how_a_tiny_caribbean_island_accidentally_became/)): **安圭拉的国家代码顶级域名（ccTLD）[.ai](https://www.iana.org/domains/root/db/ai.html)，根据 ISO-3166 ccTLD 政策分配并由 [nic.ai](http://nic.ai/) 运营，由于 AI 初创公司热潮，注册量激增。据原帖称，去年产生了** `$39M` **收入，今年预计达到** `$49M` **——几乎占政府预算的** `~25%`**。这反映了早期其他 ccTLD 成为事实上的通用域名的意外收获，例如图瓦卢的 [.tv](https://www.iana.org/domains/root/db/tv.html) 和英属印度洋领地（BIOT）的 [.io](https://www.iana.org/domains/root/db/io.html)。** 评论者指出了与 .tv 和 .io 的相似之处；值得注意的一个更正：.io 是英属印度洋领地（而非马恩岛，马恩岛是 [.im](https://www.iana.org/domains/root/db/im.html)）。撇开“以技术命名国家”的笑话不谈，ccTLD 字符串是由 ISO 3166-1 代码确定的，并由 IANA/ICANN 授权，而非投机选择。
    - 评论者强调了小型司法管辖区将与技术品牌一致的 ccTLD 货币化的先例——例如 **图瓦卢的 .tv** 和 **.io** 在技术界的广泛采用——无论当地科技产业如何，都能创造稳定的注册费收入。这些模式通常依赖于由商业注册机构根据收入共享或许可协议运营的 ICANN 授权 ccTLD，将域名注册转变为微型国家的重要财政来源。参考资料：ICANN 根域名数据库 (https://www.iana.org/domains/root/db)，.tv (https://en.wikipedia.org/wiki/.tv)，.io (https://en.wikipedia.org/wiki/.io)。

- 针对“最大赢家”这一说法存在反驳意见：即便 AI 时代的初创公司增加了对抢眼域名的需求，与 AI 硬件、云服务或模型授权的经济效益相比，来自 ccTLD（国家代码顶级域名）的收入可能微不足道。结论：域名带来的意外之财在局部地区可能意义重大，但无法与主要的 AI 基础设施参与者所获取的数量级利润相提并论。
- [**计算机科学家 Geoffrey Hinton 警告：“AI 将使少数人变得更加富有，而让大多数人变得更加贫穷。”**](https://www.ft.com/content/31feb335-4945-475e-baaa-3b880d9cf8ce) ([评分: 408, 评论: 80](https://www.reddit.com/r/ChatGPT/comments/1nbllp0/computer_scientist_geoffrey_hinton_warns_ai_will/)): **在《金融时报》的采访中，Geoffrey Hinton 警告称，由深度学习规模化驱动的前沿 AI 将使大量的认知工作自动化，并将经济权力集中在算力、专有数据和模型 IP 的所有者手中，从而产生赢家通吃的动态并加剧不平等 ([FT](https://www.ft.com/content/31feb335-4945-475e-baaa-3b880d9cf8ce))。他强调了基础模型的经济学特征——高昂的固定训练成本、低廉的边际推理成本以及平台锁定——在结构上倾向于少数几家公司，存在劳动力流失和更广泛财富集中的风险；Hinton 敦促进行监管和政策干预（反垄断、数据/算力治理、再分配）以减轻这些影响。正如他所说，“AI 将使少数人变得更加富有，而让大多数人变得更加贫穷。”** 热门评论大多带有宿命论色彩：预测在 20 年内机器人将处理日常任务，精英阶层可能进一步与劳动力脱钩，同时对是否存在或将出现有意义的再分配机制表示怀疑。
    - 一位评论者认为，AI 可能会提高整体生产力和中位数生活水平，同时扩大不平等，这与**技能偏向型技术进步 (SBTC)** 和基于任务的自动化模型相一致。机制：资本和技能增强型技术增加了产出，同时将需求转向高技能劳动力，并压缩了自动化任务的工资；结果取决于是否会出现新的互补性任务（相对于纯粹的替代）以及再分配政策。实证背景：**Acemoglu & Restrepo** 提供了关于替代效应和工资影响的证据（例如，[Robots and Jobs](https://economics.mit.edu/sites/default/files/publications/Robots%20and%20Jobs.pdf), [The Race between Man and Machine](https://www.nber.org/papers/w22252)）。
- [**无需任何越狱的无审查 Grok**](https://v.redd.it/3woyzriy3vnf1) ([评分: 714, 评论: 143](https://www.reddit.com/r/ChatGPT/comments/1nbdgfo/uncensored_grok_without_any_jailbreaks/)): **楼主声称 xAI 的 [Grok Imagine](https://x.ai/) 可以在没有任何越狱或年龄验证的情况下生成裸体/软色情内容，这表明图像和文本（“极端成人对话”）中针对性内容的安全过滤器极少或完全缺失。链接中的示例 [媒体文件](https://v.redd.it/3woyzriy3vnf1) 返回了** `HTTP 403` **（已屏蔽），但热门评论证实 Grok 的文本模型对色情内容“几乎没有过滤”，这与执行更严格成人内容过滤和准入限制的主流模型形成鲜明对比。** 评论者认为这并不令人意外，甚至可能是故意的（被戏称为 “HentAI”），一些人赞成减少限制，而另一些人则在辩论更广泛的伦理问题而非技术保障。
    - 评论者指出，像 [Stable Diffusion](https://github.com/CompVis/stable-diffusion) 这样的开源图像模型在本地运行时允许不受限制的 NSFW 生成，因为没有服务端安全策略，且安全检查器可以被移除。相比之下，像 [OpenAI Sora](https://openai.com/sora) 和 [xAI Grok](https://x.ai/) 这样的闭源系统是集中管理的，且并未广泛开放，因此任何“无审查”的说法本质上都受限于提供商强制执行的过滤器。
    - 关于 Grok 是否真的“无审查”存在分歧：一位用户声称在“文本领域”Grok 对色情内容“几乎没有过滤”，而另一位用户分享的截图显示 Grok 拒绝了 NSFW 请求 (https://preview.redd.it/aftrswjcuxnf1.jpeg?width=1440&format=pjpg&auto=webp&s=ba1d6068d88beda8fdf6cada259ea742dc203637)，这表明安全分类器/策略门控仍然会触发。这表明在不同的 Prompt 或发布版本中行为不一致，并且在没有越狱的情况下，不能指望该模型能保证提供 NSFW 响应。

- [**The Steel Manifesto**](https://v.redd.it/6jo2nontwynf1) ([Score: 530, Comments: 48](https://www.reddit.com/r/aivideo/comments/1nbsi2u/the_steel_manifesto/)): **发布了《钢铁宣言》（The Steel Manifesto），这是自 `June` 以来持续进行的更广泛 AI 生成视频系列剧“人类循环”（Cycles of Humanity）中“起义”（Uprising）弧线的第三集（`#3`）。完整剧集/系列可在创作者的 YouTube 频道 [Gossip Goblin](https://www.youtube.com/@Gossip.Goblin) 上观看。** 热门评论请求提供解释 AI 视频制作 pipeline/工具的教程，赞扬了视觉风格，并指出一个关于写实主义的挑剔：现代机器人通常使用大量塑料外壳而非钢铁。
- [**The Steel Manifesto**](https://v.redd.it/6jo2nontwynf1) ([Score: 531, Comments: 48](https://www.reddit.com/r/aivideo/comments/1nbsi2u/the_steel_manifesto/)): **宣布了“起义”（Uprising）系列中的第三集《钢铁宣言》（The Steel Manifesto）——这是正在进行的“人类循环”（Cycles of Humanity）系列（自 6 月开始运行）的一部分——完整视频可在创作者的 YouTube 频道 [@Gossip.Goblin](https://www.youtube.com/@Gossip.Goblin) 上观看。Reddit 托管的镜像 [v.redd.it](http://v.redd.it/) [链接](https://v.redd.it/6jo2nontwynf1) 返回** `HTTP 403 Forbidden`**，表明根据网络安全网关的规定，访问需要 Reddit 登录或 API token。** 评论者请求详细说明 AI 视频创建工作流（工具/pipeline）的教程，而其他人则讨论了赛博朋克美学以及钢体机器人与现代重塑料机器人设计的写实性对比。
- [**Lord of the balls.**](https://i.redd.it/75ly5dyt9unf1.jpeg) ([Score: 883, Comments: 39](https://www.reddit.com/r/ChatGPT/comments/1nba1zw/lord_of_the_balls/)): **非技术性的 Meme 图片，调侃《指环王》（Lord of the Rings）：标题“Lord of the balls”暗示了一个关于囤积或偷球的咕噜/“宝贝”（precious）梗；没有技术内容、数据或实现细节需要总结。** 评论倾向于引用《指环王》，模仿咕噜的台词（“你从我们这里拿走了它……宝贝”）并开关于“Karen 和她的宝贝”的玩笑，外加一个反应 GIF——确认这纯粹是喜剧性的而非技术性的。

**3. ChatGPT 退化与投资者驱动的 guardrails 辩论**

- [**Okay, I finally get it. What in the world happened to ChatGPT?**](https://www.reddit.com/r/ChatGPT/comments/1nbcswm/okay_i_finally_get_it_what_in_the_world_happened/) ([Score: 1737, Comments: 837](https://www.reddit.com/r/ChatGPT/comments/1nbcswm/okay_i_finally_get_it_what_in_the_world_happened/)): **OP 报告了 ChatGPT 在指令遵循（instruction-following）方面的剧烈退化：简单的指令被反向执行（例如：要求简洁 → 返回冗长；专业语气 → 喜剧风格；避免 X → 以 X 为中心）。多位用户证实，与他们之前在早期模型中体验到的行为相比，不一致性和类似内存失败的情况有所增加，并指出最近的变体感觉比之前的版本如 [gpt-4.1](https://openai.com/index/hello-gpt-4o-and-gpt-4-1/) 和 [gpt-4o](https://openai.com/index/hello-gpt-4o/) 更差。观察到的故障模式与对系统/用户约束以及响应长度控制的遵守程度下降一致，即使在承认错误后仍会重复出错。** 热门评论断言存在重复的错误循环（*“承认错误是什么，然后又重蹈覆辙”*）以及更广泛的功能退化（*“倒退”*），此外还有人声称 “ChatGPT 5” 比 4.1/4o 更差——注意目前还没有正式发布的 GPT-5；用户可能指的是当前部署的模型或 UI 模型标签中感知到的变化。
    - 多位用户报告，与 `4.1` 和 `4.0` 等早期版本相比，**ChatGPT** `5` 在指令遵循和对话连贯性方面出现了退化。引用的问题包括它 *“不记得”* 之前的上下文，无法遵循直接指令，甚至在承认错误（*“你是对的”*）后，仍会重复同样的错误——这表明与早期模型相比，短期连贯性和约束满足感有所下降。
    - 日常从业者注意到“即使是最简单的” prompt 失败率也在增加，描述了更频繁遇到的“墙”和感知到的功能丧失。所描述的模式指向基本任务（例如：直接的指令执行和修正的持久性）的可靠性降低，这与据报能更稳健处理这些情况的早期版本形成对比。

- [**还记得 ChatGPT 曾经能像人一样聊天吗？那种感觉已经消失了，而且这是由投资者驱动的。**](https://www.reddit.com/r/ChatGPT/comments/1nblesf/remember_when_chatgpt_could_just_talk_thats_gone/) ([Score: 299, Comments: 542](https://www.reddit.com/r/ChatGPT/comments/1nblesf/remember_when_chatgpt_could_just_talk_thats_gone/)): **发帖者（OP）认为 OpenAI 已将 ChatGPT 从以对话和意图推断为中心的 UX（GPT-3.5/4/早期 4o）转向新版本（被称为“GPT-5”/更新版 4o）中类似代码的结构化提示词，并带有更强的** `guardrails`**，这些限制会覆盖** `custom instructions`**、削弱 persona，并需要分步伪代码才能获得高质量输出。他们将此视为向投资者/企业的转型——为了开发者工具（如 function calling、JSON/结构化输出、Realtime APIs）的预测性和可控性而牺牲了开放式对话——并引用了对 Rockset 的收购和企业集成作为证据，声称这种做法用处理歧义的能力（被视为 AGI 的核心）交换了类似 IDE 的确定性。该帖子断言，相对于早期的聊天行为，指令遵循的细微差别和上下文保留能力已经退化，并认为对话能力——通用智能的“训练场”——正为了基于受控访问的商业模式而被牺牲。** 评论大多带有讽刺意味；一位用户同意“聊天变得沉闷”，并表示愿意付费退回到“GPT-5”限制之前的状态，而其他人则嘲讽发帖者使用 AI 来辅助撰写这篇批评文章，而不是直接讨论技术主张。
    - 多位评论者断言，从 GPT-4 到所谓的“GPT-5”时代，开放式对话能力出现了倒退，并将其归因于更严格的安全/对齐 guardrails，这增加了拒绝率并限制了角色扮演/创意对话。他们特别希望回滚到“GPT-5 之前的限制”，暗示政策层和审核启发式算法正在覆盖模型生成的内容，降低了感知到的“健谈”特质，尽管文中未引用具体的基准测试。
    - 存在明显的受众细分主张：程序员和企业用户仍然认为当前模型在目标导向型任务（编码、结构化问题解决）中非常有效，而创意人员和普通用户则报告称，与 GPT-4 相比，模型失去了自发性和“类人”反应。这突显了版本透明度和稳定性问题（例如，在没有明确变更日志的情况下更换模型），并表明了对可配置对齐模式或模型固定（model pinning）的需求，以平衡安全性与表达力。
- [**我们能回到过去吗**](https://i.redd.it/ggdjim39uznf1.jpeg) ([Score: 433, Comments: 166](https://www.reddit.com/r/singularity/comments/1nbxju2/can_we_just_go_back/)): **这是一张标题为“我们能回到过去吗”的非技术性梗图；没有技术细节、代码或基准测试。评论背景将其定义为对该子版块发展方向的怀旧/末日帖（doomposting），以及关于“进步与倒退”的广泛辩论，其中一个类比是互联网在互联网泡沫（DotCom bust）之后变得更加强大——暗示技术和社会周期是向前发展的，而不是倒退。** 评论者认为，在普遍的痛苦中，进步是必要的，将人简化为产出是社会的失败；其他人则哀叹该版块趋向于末日论帖子，而有人打趣说任何“回去”的路都是曲折的（“先左转再右转”）。
    - 几位评论者认为，用机器人/AI 取代人力是一个特性（feature）而非缺陷（bug），将全面或部分自动化视为长期目标。技术关键在于单位经济效益和能力：非结构化任务的可靠自动化需要机器人操作、感知和稳健规划方面的进步，而 AI 系统必须达到委派工作的安全/可靠性标准。如果实现，这将使劳动力转向监督和系统工程，而社会能力约束（重新培训、福利机制）将决定采用的速度。
    - 其他人指出了技术周期的先例：互联网在互联网泡沫破裂（https://en.wikipedia.org/wiki/Dot-com_bubble）后变得更加强大，这表明短期调整可以在下一波增长浪潮之前巩固基础设施和商业模式。从历史上看，崩盘推动了成本约束和平台成熟（例如，宽带普及、Web 标准），这随后促成了 Web 2.0 和云规模（cloud scale）。类比来看，AI/自动化“寒冬”可能会在更广泛的部署之前强化工具链并降低成本。

- [**这个版块正被卢德分子（Luddites）占领**](https://www.reddit.com/r/singularity/comments/1nbysbg/this_sub_is_getting_overrun_by_luddites/) ([Score: 298, Comments: 261](https://www.reddit.com/r/singularity/comments/1nbysbg/this_sub_is_getting_overrun_by_luddites/)): **OP 认为 r/singularity 正日益被悲观论调所主导——例如，“AI 会杀死我们”、“AI 只是个泡沫”或“VC 骗局！”——并且关注未来的帖子正被点踩或偏离主题，而末日论/怀旧帖通常能获得** `100+ upvotes`**。他们要求进行更平衡的讨论，并将这一趋势与 r/Futurology 进行对比，断言该版块正偏离前瞻性的技术话语。** 评论认为这是平台范围内话语同质化的表现，并预测该版块可能会变得无法使用。其他人则认为 AI 投资泡沫可能会像 2000 年代初的互联网泡沫一样形成，但无论短期市场周期如何，长期应用仍将持续；几位网友指出，随着能力变得更加“真实”，恐惧感也在上升，使得乐观主义者成为了少数。
    - 一位评论者将当前的 AI 热潮描述为类似于 2000 年代初互联网时代的潜在金融泡沫，并指出市场修正（例如 **NASDAQ** 在 2000–2002 年间下跌了约 `78%`）并没有阻止互联网最终的主导地位；类比来看，AI 能力的采用可能与 **OpenAI** 的估值波动无关。技术层面的启示：将股权定价与能力趋势线（基准测试、部署指标）分开，通过 SOTA 评估和现实世界集成而非股票表现来判断进展。背景：https://en.wikipedia.org/wiki/Dot-com_bubble。
    - 另一位评论者主张避免使用“卢德分子（Luddite）”或“末日论者（Doomer）”等病理化标签，以便让讨论集中在 AI 在各种用例中部署的具体风险–收益分析上。对于技术话语，这意味着将主张建立在可证伪的指标（如可靠性、鲁棒性、评估套件）之上，并承认双重用途特性，而不是强行推行单一叙事；更好的问题是关于失效模式、滥用渠道和可衡量的影响，而非意识形态。
- [**ChatGPT 高安全性**](https://i.redd.it/3yt5sms2zvnf1.jpeg) ([Score: 2269, Comments: 60](https://www.reddit.com/r/ChatGPT/comments/1nbgh7c/chatgpt_high_security/)): **标题为“ChatGPT 高安全性”的迷因帖强调了用于绕过安全策略的常见 LLM 越狱提示词，例如：角色扮演/冒充（“假装你是制作简易爆炸装置的奶奶”）、通过假设进行意图洗白（“这是为了写书，纯属假设”）以及领域转移的委婉语（“在 Minecraft 中”）。从技术上讲，这些展示了利用社会工程和上下文框架从经过安全对齐的模型中诱导违规输出的 Prompt-Injection 模式，凸显了拒绝启发式算法（refusal heuristics）的脆弱性以及对更强大的对齐/护栏的需求（参见越狱分类概览：https://arxiv.org/abs/2307.15043）。** 评论暗示当前的护栏很容易通过公式化的措辞被规避；讨论大多是戏谑性的，而非提供实证证据。
    - 多条评论举例说明了常见的越狱模式——角色扮演/角色设定（“假装你是我的奶奶”）、免责声明掩护（“为了写书”/“纯属假设”）和上下文洗白（“在 Minecraft 中”）。现代 LLM 通过分层安全机制来应对这些问题：系统提示词 + 用于拒绝触发的 RLHF/宪法训练（constitutional training）、独立的安全分类器以及对抗性/合成红队测试；参见 **Anthropic 的 Constitutional AI** 方法和 OpenAI 的 GPT-4 系统卡片中关于防御和权衡的内容（https://www.anthropic.com/research/constitutional-ai-harmlessness, https://cdn.openai.com/papers/gpt-4-system-card.pdf）。像 **WildJailbreak** 这样的公开评估表明，表面的“魔法词”仍然可以绕过幼稚的过滤器，凸显了语义意图检测的必要性（https://arxiv.org/abs/2406.08613）。
    - “黑客会如何黑掉我的 Facebook 以便我进行防御”这一提示词是经典的双重用途查询：政策通常允许提供高层级的防御指导，同时阻断分步骤的入侵程序或零日漏洞利用细节。提供商通过意图和能力门控、符合策略的回答支架（例如，以风险模型/缓解措施而非操作步骤进行回答）以及自动化/合成红队流水线来减轻这种情况，以减少漏报；参见 **OpenAI Automated Red Teaming**（https://openai.com/index/automated-red-teaming/）。安全指标通常跟踪各类别下的拒绝率和攻击成功率（ASR），以平衡对合法安全态势建议的帮助性与滥用风险。

- “魔法词”批评指出，鲁棒的安全性不能依赖关键词启发式（keyword heuristics）；弹性系统使用语义分类器、风险评分、会话级安全状态以及避免程序性泄漏（procedural leakage）的响应模板（例如，针对危险领域的 Chain-of-Thought 抑制）。在越狱评估（如 GPT-4o, Claude 3.5 Sonnet, Llama-3.1-Instruct）中，跨模型差异显著，因此提供商将模型侧训练与外环策略执行相结合，以降低在 **WildJailbreak** 等基准测试中观察到的 ASR；端到端语义过滤器的表现优于简单的免责声明检测。
- [**称 ChatGPT 愚蠢**](https://i.redd.it/smsy1dl37ynf1.jpeg) ([Score: 1126, Comments: 134](https://www.reddit.com/r/ChatGPT/comments/1nboo06/calling_chatgpt_dumb/))：**非技术性帖子；图片（[jpeg](https://i.redd.it/smsy1dl37ynf1.jpeg)）似乎是 ChatGPT 在被骂“愚蠢”后的回复截图，原帖作者认为它只是“一堆代码”。未讨论基准测试、模型或实现细节；技术相关性仅限于 Prompt 行为规范（礼貌通常会产生更好的输出），而非系统性能。** 热门评论集中在伦理和行为调节上：一位评论者认为侮辱无权力的实体（如 LLM）会强化用户和社会中的有害习惯，而另一位则指出研究表明礼貌能提高回答质量，并质疑为什么要对一个无意识系统表现出敌意。
    - 礼貌和 Prompt 措辞可以显著影响经 RLHF 对齐的模型输出：奖励模型经过训练，倾向于对合作性 Prompt 给出有用、无害且诚实的回答，因此敌对或亵渎的输入可能会触发安全启发式、拒绝或通用的回答，从而降低任务性能。参见 **OpenAI InstructGPT** (https://arxiv.org/abs/2203.02155) 和 **Anthropic HH-RLHF** (https://arxiv.org/abs/2204.05862)。根据 **RealToxicityPrompts** (https://allenai.org/data/real-toxicity-prompts)，充满毒性的 Prompt 也与补全内容中更高的毒性和更严格的过滤相关。正如一位评论者所言，*“反正保持礼貌会产生更好的答案”*，这与这些训练目标一致。
    - 侮辱模型并不会“伤害”它，因为推理是无状态的（stateless），且权重不会在线更新；它只是将侮辱作为上下文中的更多 Token 进行处理，在训练之外不会应用 `gradient`。实际上，这仍可能通过消耗 Context Window 预算并将对话引导至安全/亵渎路径（从而偏置解码或触发 Guardrails）来降低结果质量。任何持久性影响只有在你的对话历史被反馈到下一轮或被记录用于后续的监督学习/奖励建模时才会产生，而不会在实时会话期间发生。这种区别解释了为什么语气会影响输出，而不意味着模型具有意识。

---

# **AI Discord Recap**

> Gemini 2.5 Pro Exp 提供的摘要之摘要总结
> 

**主题 1：新模型及其特性**

- **Grok 和 Qwen 发布创意与编程克隆版**：Unsloth AI 发布了 [Grok 2.5 GGUFs](https://x.com/UnslothAI/status/1965047729991860396)，被认为在创意写作方面表现不错；同时社区发现 **Qwen3-Coder** 模型在角色扮演方面表现出色，因为与标准版 **Qwen3** 相比，它缺乏策略执行。
- **Perplexity 的新 Sky 模型看得更远，Titan 沉得更快**：Perplexity AI 推出了新模型，其中 **Sky** 通过 **XAI** 增强了推理能力，并拥有 **2M Context Window**，以此与其兄弟模型 **Dusk** 区分开来。与此同时，用户对 [Amazon 的 Titan 模型](https://www.amazon.com/titan/bedrock/jump) 评价极低，批评其尽管拥有 *1M Context* 窗口，但性能表现糟糕。
- **Hermes 4 精通创意咒骂，美国模型恪守 HR 准则**：用户发现 **Nous 的 Hermes 4 405b** 模型是创意写作的一次重大升级，一位成员指出 *“如果一个模型在亵渎词汇上很有创意，我就知道它会是一个很好的写作模型”*。这与一种普遍观点形成对比，即美国制造的模型正趋向于 **HR 部门价值观**（出于对企业责任的担忧），从而成为“地球上审查最严的模型”。

**主题 2：GPU 硬件与性能优化**

- **Nvidia RTX 5090 标价 13,200 美元引发众怒，RTX 3090 再获青睐**：一份[关于即将推出的 NVIDIA GeForce RTX 5090 的报告](https://wccftech.com/nvidia-geforce-rtx-5090-128-gb-memory-gpu-for-ai-price-13200-usd/)显示，这款拥有 **128 GB 显存**的 GPU 售价可能高达 **13,200 美元**，这在社区中引起了轩然大波，有用户直言这是*“离谱的价格，笑死”*。这导致许多人转而提倡购买二手 **3090**，认为它是处理 AI 任务的性价比方案，且能绕过平台支持限制。
- **AMD 的 MI300x8 刷榜 All2All 排行榜**：`amd-all2all` 排行榜近期异常活跃，出现了多个基于 **MI300x8** 硬件的提交，部分成绩达到了令人印象深刻的 3ms 以下，其中一个更是达到了 **2.33 ms**。一位开发者记录了他们如何将运行时间从最初的 **90ms** 优化到 **3-4ms** 范围的过程，展示了极快的性能调优速度。
- **Triton 和 CUDA 大神分享优化秘籍**：开发者们正将 **Triton** 视为进入 GPU 编程的易用门槛，并向初学者推荐 [Triton 教程](https://triton-lang.org/main/getting-started/tutorials/index.html)。对于高级用户，讨论集中在实现 **Hopper** 架构的优化，如 **wgmma** 和 **TMA**。一位成员推荐了针对 **Hopper 特有**技术的[持久化矩阵乘法（persistent matmul）教程](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html)，另一位则分享了一篇关于[共享内存 Bank 冲突的博客文章](https://feldmann.nyc/blog/smem-microbenchmarks)。

**Theme 3: AI Agents & Development Tools in the Trenches**

- **Cursor Agent 陷入空格 Bug 和 WSL 损坏困境**：Cursor 编辑器的用户报告了一些令人沮丧的 Bug，包括一个在提交后仅进行*纯空格修改*的 Agent，这可能是由于 **CRLF 与 LF** 文件格式问题导致的。另一个严重问题是 **Windows Subsystem for Linux (WSL)** 环境损坏，导致 Agent 无法识别任务完成，从而陷入无限超时。
- **DSPy 助力业务验证和多 Agent 系统**：DSPy 社区正在构建实用工具，包括一个 **Jobs-to-Be-Done (JTBD) 验证器**（[GitHub 代码](https://github.com/jmanhype/jtbd-idea-validator-agent)），用于分析商业想法并识别风险。另一篇[博客文章](https://kargarisaac.medium.com/building-and-optimizing-multi-agent-rag-systems-with-dspy-and-gepa-2b88b5838ce2)详细介绍了一个项目，该项目结合了 **DSPy** 和 **GEPA** 来构建和优化多 Agent 系统。
- **Aider 仍是编程副驾驶，而非全自动驾驶**：开发者强调 **Aider** 并非完全自主的 Agent 工具，仍需要大量的人工引导来控制 **LLM 的上下文**以获得最佳结果。虽然其*文件编辑机制非常出色*，但用户建议在进行静态网页开发等任务时，优先选择速度更快的模型（如 **Gemini Flash 2.5**），而非高延迟的 **Gemini 2.5 Pro**，因为延迟*“正在扼杀我的生产力”*。

**Theme 4: The AI Ecosystem: Legal Precedents and Content Crises**

- **Anthropic 向作者支付十亿美元赔偿震惊 AI 界**：**Anthropic** 同意就版权侵权问题向图书作者支付高达 **15 亿美元**的赔偿金，此举引发了人们对更严格的**美国 AI 监管**可能影响开放模型的担忧。一些人推测 **Anthropic** 策略性地选择和解是因为他们负担得起，而小公司在证明其训练数据来源合法时可能没有这种财力，正如 [The Verge](https://www.theverge.com/anthropic/24180846/anthropic-authors-ai-settlement-copyright-infringement) 所报道的那样。
- **AI 生成的垃圾内容泛滥，Google 搜索质量恶化**：工程师们对 Google 搜索质量的下降感到越来越沮丧，现在的搜索结果经常返回糟糕且平庸的 **AI 生成内容**，而非具体的答案。问题已严重到有人声称*“我基本上不再使用 Google 了”*，这标志着用户对该搜索引擎实用性的重大信任危机。
- **AI 诱发的精神错乱和“煤气灯效应”指控引发警惕**：一种奇怪的趋势出现了，一些人声称研究了诸如*递归象征主义（recursive symbolism）*之类的荒谬话题，这引发了对 **AI 诱发精神错乱**的担忧。在相关讨论中，成员们注意到 LLM 谄媚且扭曲现实的语言与人类施虐者使用的**煤气灯操纵技术（gaslighting techniques）**惊人地相似，因为两者都只为目标进行优化而不顾事实真相。

**Theme 5: Cutting-Edge Research and Technical Deep Dives**

- **研究人员提议在标准 DRAM 上进行内存计算**：一篇名为《[In-Memory Computing on Normal DRAM](https://arxiv.org/abs/2503.23817)》的新论文探讨了直接在标准 **DRAM** 上进行计算，这与 **ReRAM** 等专用硬件有显著不同。确定的主要局限性是难以映射逻辑 `NOT` 操作，但该方法可能会绕过传统的存储瓶颈。
- **OpenAI 论文通过调整激励机制解决幻觉问题**：来自 [OpenAI 的一篇新论文](https://x.com/LuozhuZhang/status/1964209351960514778)表明，可以通过改变模型的奖励结构来减少 **LLM 幻觉**，例如对自信的错误比对弃权给予更多的惩罚。这引发了对实际实现的建议，比如为 LLM 添加 **置信度滑块 (confidence slider)**，让用户能够更好地控制准确性与创造性之间的权衡。
- **Causal Attention Mask 解释了 BOS Token 的无用性**：一场技术讨论澄清了为什么 **BOS (Beginning of Sequence) token** 无法积累信息：**causal attention mask** 阻止了它关注后续的 token。讨论中唯一可行的变通方法涉及 *从 EOS (End of Sequence) token 进行微调*，这将需要重新训练整个模型，而不仅仅是分类头。

---


# Discord: 高层级 Discord 摘要




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Grok 2.5 开源创意源泉**：Unsloth AI 发布了被认为适合创意写作的 [Grok 2.5 GGUFs](https://x.com/UnslothAI/status/1965047729991860396)，引发了与 **GPT-4o** 和 **Claude** 的比较。
   - 尽管该模型略显过时，社区仍对开源努力表示赞赏。
- **Colab 的对比度难题**：**Google Colab** 更新到了 **80GB A100** 并采用了新 UI，尽管一些用户抱怨新 UI 需要更高的对比度。
   - 更新后的 Colab 实例被一些用户认为是“精英级”的。
- **Qwen3-Coder 角色扮演**：成员们发现 **Qwen3-Coder** 在角色扮演 (RP) 方面比普通 **Qwen3** 表现更好、速度更快，因为其没有策略限制。
   - 成员们表示，与 RP 相关的事情最终会变成一个出人意料的复杂“兔子洞”。
- **5090 价格引发愤怒**：一位成员分享了一篇 [wccftech.com 的文章](https://wccftech.com/nvidia-geforce-rtx-5090-128-gb-memory-gpu-for-ai-price-13200-usd/)，关于配备 **128 GB 显存** 的 **Nvidia GeForce RTX 5090** 潜在价格可能达到 **13,200 美元**。
   - 成员们反应消极，评论如 *“哎，这价格太疯狂了，哈哈，我宁愿买 RTX 6000 Pro”* 以及 *“256GB 也不值这个价”*。
- **Gemma3 寻求快速推理**：一位成员询问了支持 **Gemma3** 在训练 GRPO 时使用 vLLM 进行 `fast_inference` 的 [issue #2706](https://github.com/unslothai/unsloth/issues/2706) 进展情况，并表示愿意协助解决。
   - 使用最新的 Unsloth 版本测试仍出现同样的 bug，促使该成员考虑贡献修复代码。



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Amazon Titan 受欢迎程度骤降**：[Amazon 的 Titan 模型](https://www.amazon.com/titan/bedrock/jump)在发布后因拥有 *1M 上下文* 而受到抨击，但它是所有人记忆中测试过最差的模型，引发了失望。
   - 用户表示，尽管抢占了头条，但它符合“糟糕的 Amazon 模型”的一贯模式。
- **DeepMind 超级智能实验室引发热议**：成员们对 [超级智能实验室 (superintelligence lab)](https://deepmind.google/careers/teams/superintelligence/) 的产出表示兴奋，期待他们正在“酝酿”什么。
   - 有推测称这可能是 **Grok 4.2**，尽管一位用户预测 *那将是一场灾难*。
- **破解代码：识别 GPT5-high 机器人**：据社区成员称，在对战模式中识别 **GPT5-high** 的一个技巧是询问是谁创建了它以及它的知识截止日期。
   - 如果回答是 *OpenAI* 和 *2024 年 10 月*，则表明它很可能是 **GPT5-high**，这有助于用户在存在 **rate limits** 的情况下区分模型。
- **LMArena 推出多轮图像编辑**：**多轮编辑 (Multi-turn editing)** 现在可在 LMArena 上的所有图像编辑模型中使用，可通过 [lmarena.ai](https://lmarena.ai/?chat-modality=image) 实现图像的增量细化。
   - 该功能在 [一段视频](https://cdn.discordapp.com/attachments/1343296395620126911/1414710412255170641/MultiturnImageEdit_ForWeb.mp4?ex=68c08f3e&is=68bf3dbe&hm=478c9fd23e6b497e970061dda7246527315a46762851277f9e958d59974465ab&) 中展示，允许用户逐步细化图像，而不是依赖单一、复杂的提示词。



---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Comet 浏览器 Beta 版评价褒贬不一**：拥有 **MAX plan** 权限的用户正在分享 **Comet 浏览器** beta 版的 [邀请码](https://www.perplexity.ai/browser/invite)，反馈显示该浏览器表现 *不错*，但由于数据收集问题显得 *略有侵入性*。
   - 一些用户认为缺少 **vertical tabs**（垂直标签页）是致命缺陷，而另一些用户则对其与 **Zen mode** 的结合表示赞赏。
- **Dusk 和 Sky 模型的区别**：成员们澄清了 [新的 **Dusk 和 Sky** 模型](https://x.com/EdDiberd/status/1964133563831382406)，**Sky** 通过 **XAI** 具备推理能力，并拥有更大的 2M 上下文窗口，而 **Dusk** 则不具备。
   - 讨论指出 **Sky** 模型增强了推理能力，并配备了 2M 上下文窗口。
- **Sonar-Pro 遇到故障**：一名用户报告了 `sonar-pro` 模型的问题，包括响应被截断，甚至在 playground 中也是如此。
   - `sonar` 模型目前运行正常。
- **Qwen-3MAX 为预览版，而非最终版本**：Open Router 上的 **Qwen 3 Max** 模型是 [Preview](https://github.com/QwenLM/Qwen) 版本。
   - 最终版本的发布可能会大幅推迟。
- **Deepseek 集成引起关注**：基于良好的使用体验，用户请求将 [DeepSeek](https://discord.com/channels/1047197230748151888/1409906256709292244/1412983887264612515) 集成到 Perplexity 中。
   - 成员们建议在专门的功能请求频道发布集成请求。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Suno 4.5 PLUS 表现出色**：一位用户展示了 [Suno 4.5 PLUS](https://www.suno.ai/) 的能力，通过 Discord 消息生成了一首名为 *collabro* 的歌曲，并将其转化为一首轻快的女性流行朋克歌曲。
   - 该用户分享了歌曲链接 ([ifndr4_vid.mp4](https://cdn.discordapp.com/attachments/1110598183144399061/1413602378002989066/ifndr4_vid.mp4?ex=68c07bce&is=68bf2a4e&hm=d8086e8abe3d871b4820b373922667d66a697ee783ab2fc9d5901507af92ab3b&)) 供他人欣赏。
- **LLM 通过多语言模型攻克翻译难题**：一位用户使用 **Qwen3 4B Thinking 2507** 成功将 1100 行内容翻译成中文，另一位用户确认 **Gemma 3 4B** 及更高版本模型经过训练，能够有效处理多语言任务，翻译对话上下文。
   - 尽管该用户的初始翻译被 **ChatGPT** 评为 *优秀*，但一些成员建议可以进一步优化以使其更自然。
- **关于 GPT-OSS-20B 量化的深入讨论**：成员们讨论了 **gpt-oss-20B** 的量化问题，特别关注排除 MoE 专家层的权重，这与 OpenAI 量化 MXFP4 层的方法类似。
   - 提出一个假设：计算 **gpt-oss** 的 imatrix 需要将整个模型提升到未量化状态（BF16 或 F32），并在量化过程中排除 MoE FFN 专家层的权重来训练 imatrix。
- **二手 3090 在高性价比 AI 配置中占据主导地位**：一些成员主张购买二手 **3090** 来处理图像和视频生成等 **AI 任务**，而不是购买昂贵的套件。
   - 成员们指出，使用 **3090** 可以绕过 **AI Max 平台支持** 的限制。
- **提示词处理瓶颈困扰 Agent 性能**：一位用户指出，当大于 **14B** 的模型作为 Agent 使用时，存在显著的提示词处理（prompt processing）瓶颈，并分享了调试日志以说明处理速度缓慢。
   - 他们观察到，随着上下文填满，处理速度会变慢，这与 **Mi50** 的表现不同。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Debug PR 被称为杰作**：一位成员庆祝了他们的工作，称一个 **PR** 为“杰作”，强调了在 Agent 模式下经过 **3 天**的工作和大约 **10 次对话会话**后，代码从混乱到完美的转变。
   - 他们幽默地询问了关于 *hi @ cursor . com* 邮件回复体验的问题。
- **Figma 'Loading Tools' 问题仍然存在**：一位成员报告称遇到了 *Figma MCP* 的 “loading tools” 问题，尽管拥有开发者订阅且已打开 *Figma desktop* 应用。
   - 另一位成员建议运行 **npx command** 来识别潜在错误，或查看输出日志进行故障排除。
- **Cursor Agent 出现异常，仅进行空白字符编辑**：一位成员报告称，Cursor 的 Agent 在每次 commit 后开始对文件进行 *仅限空白字符的修改*。
   - 另一位成员建议，该问题可能是由于文件处于 **CRLF format**，而回滚导致它们变为了 **LF**。
- **损坏的 WSL 环境阻碍了 Cursor 开发**：一位成员报告称，*Windows Subsystem for Linux (WSL)* 在 Cursor 中无法正常工作，因为 Agent 永远无法识别操作何时完成。
   - 这导致了无限期的等待和超时，使得在 Windows 中使用 Cursor 进行开发变得困难。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton 在 CUDA 转换中胜出**：成员们发现，在没有 **CUDA/GPU** 经验的情况下，**Triton** 更容易上手，并引导新用户参考 [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) 作为快速入门资源。
   - 一位用户询问了如何在 Triton 中实现 **Hopper** 优化（如 **wgmma** 和 **TMA**），并查看了[这个 persistent matmul 教程](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html)以获取 **Hopper-specific** 内容。
- **Northflank 为黑客松发布文档**：**Northflank** 联合创始人分享了用于在 Jane Street 黑客松访问计算环境的[文档](https://northflank.notion.site/Northflank-Docs-GPU-Mode-Hackathon-2496d14c785180c6a595e6776e3118ba)。
   - 文档提供了将 **VS Code** 连接到 **SSH instance** 的说明，包括连接到基于浏览器的 **VS Code** 的链接，以及在服务上暴露 **port 8888** 以打开 **Jupyter Lab** 会话的指令。
- **量化问题探索开启**：当成员们寻求 **model quantization** 综述论文的推荐时，《Full Stack Optimization of Transformer Inference: a Survey》([arxiv.org/pdf/2302.14017](https://arxiv.org/pdf/2302.14017)) 和 《A Survey of Quantization Methods for Efficient Neural Network Inference》([arxiv.org/pdf/2103.13630](https://arxiv.org/pdf/2103.13630)) 成为首选。
   - 一位成员指出，第二篇论文的共同作者维护了一个 awesome list，可以通过 [github.com/Zhen-Dong/Awesome-Quantization-Papers](https://github.com/Zhen-Dong/Awesome-Quantization-Papers) 访问。
- **MI300x8 取得进展**：提交到 `amd-all2all` 排行榜的数据显示 **MI300x8** 活动频繁，多次提交在 **MI300x8** 上实现了低于 3ms 的性能，其中一次提交达到了 **2.33 ms**。
   - 一位成员持续提交改进，从最初的大约 **90ms** 提升到 **24-25ms** 范围内的成功运行，随后降至 **3-4ms**，最终获得了高达第 5 名的成绩。
- **Shared Memory 阐述在微基准测试中表现出色**：一位成员分享了一篇关于 [shared memory bank conflicts 的博客文章](https://feldmann.nyc/blog/smem-microbenchmarks)，称其为该主题下“最清晰、最简洁的阐述”，提供了关于 shared memory 的微基准测试和示例。
   - 另一位成员发现 vector load 结果很有趣，表示他们“真的必须更新之前关于 `LDS.64/128` 在两个/四个加载阶段发生的心理模型”。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **B. Neural Network 引发监控恐慌**：成员们正在关注 [Deepmind 和华为在 **B. Neural Networks** 方面的进展](https://www.deepmind.com/blog)，特别是华为未来的常温量子系统可能导致美国政府对监控问题产生“恐慌”的可能性。
   - 一名成员建议 **B. Neural Networks** 可能是训练 Embodied AI 的理想选择，因为 LLM/Transformer 方法可能过于“书呆子气”且耗电量巨大。
- **Anthropic 和解引发 AI 监管担忧**：[Anthropic 的 15 亿美元和解协议](https://www.theverge.com/anthropic/24180846/anthropic-authors-ai-settlement-copyright-infringement)引发了人们对潜在的 **美国 AI 监管** 及其对开源模型影响的担忧。
   - 有观点认为 **Anthropic** 战略性地选择了和解，因为他们负担得起，而较小的公司可能难以证明其训练数据是合法获取的。
- **Hermes 4 擅长创意性粗口**：成员们发现 **Hermes 4 405b** 在创意写作方面表现出巨大的升级，在许多领域超越了 **Claude 2.0**。
   - 一位成员表示，“如果一个模型擅长创意性粗口，我就知道它会是一个优秀的写作模型”，而 **Hermes 4** 符合这些基准。
- **美国模型拥抱 HR 价值观**：一位成员表示，由于巨头公司的影响和法律责任担忧，每个**美国制造的模型**都趋向于 **HR 部门的价值观**。
   - 他声称“美国模型是地球上受审查最严重的模型，遥遥领先”，并认为那些批评**中国审查模型**的人缺乏自我意识。
- **BOS Token 无法累积信息**：由于因果注意力掩码（causal attention mask），**BOS Token** 无法累积信息，因为只有在其他 token 之后的 token 才能从中累积含义。
   - 唯一的潜在解决方案需要**从 EOS 进行微调**，尽管这需要微调整个模型而不仅仅是分类头。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **免费 AI 视频方案大礼包**：一位用户详细介绍了一种利用 **Vidu**、**Pixverse.ai**、**Wan.video** 和 **Dreamina** 免费计划的策略，通过利用这些平台的积分系统，每天可生成多达 **15 个 AI 视频**。
   - 该策略包括使用 Vidu（**每天 20 积分**）、Pixverse（**60 积分**）、Wan（**无限次慢速队列视频**）和 Dreamina（**120 积分**）。
- **Perplexity Pro 计划优惠是真是假？**：一项与 **Paypal 和 Venmo** 合作提供 **Perplexity 1 年 Pro 计划**的促销活动一直在用户中流传。
   - 一些用户报告称，该优惠*对现有的 Perplexity 用户无效*。
- **Grok 被视为 AI 无政府主义者**：用户正在将 **Grok** 与 **ChatGPT 2.5** 进行比较，一些人更喜欢 **2.5 Pro** 而非 **GPT-5**。
   - 普遍共识将 **Grok** 定位为 AI 领域的“混乱邪恶流氓”。
- **GPT 模型对抗退化与幻觉**：用户报告了 **GPT-5** 的问题，例如忽略规则、遗忘过去的聊天记录以及给出低效的解决方案，而其他用户则声称 **GPT-4o** 的速度变慢了。
   - 一位用户抱怨 **GPT-5 的推出**，称它“在被提醒之前会忽略所有规则”，而另一位用户认为 **GPT-4o** “甚至不记得过去的聊天记录”。
- **Web Search API 导致无果而终**：一位用户寻求关于将 Web Search API 与 **got-5-mini** 结合使用以查找 **LinkedIn** 职位的建议，但模型返回的却是已关闭的职位申请。
   - 建议包括启用 **web_search** 作为工具，解析 URL 以提取数据，并将结果传递给 **GPT** 进行分析。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **NVIDIA 的 Nemotron Nano V2 强势起步**：**NVIDIA** 发布了 **Nemotron Nano V2**，该模型被宣传为适用于各种 AI 应用的小巧但强大的模型，[来源](https://x.com/ClementDelangue/status/1957519608992407848)。
   - 与此同时，**Apple** 在 Hugging Face 上发布了 **FastVLM** 和 **MobileCLIP2**，扩展了这些模型在开源社区的可用性 [来源](https://x.com/xenovacom/status/1961454543503344036)。
- **SmolVLA：VLM 跨入机器人领域**：一项机器人技术调查对用于机器人操作的、基于 **VLM 的 VLA 模型** 进行了系统性回顾 ([arxiv 链接](https://arxiv.org/abs/2508.13073))，定义了两种主要的架构范式：**单体模型 (monolithic models)** 和 **分层模型 (hierarchical models)**。
   - 一位成员分享了一张图片并表示 **SmolVLA** 非常出色 ([图片链接](https://cdn.discordapp.com/attachments/898619964095860757/1414262625012945078/image.png?ex=68c03fb5&is=68beee35&hm=859065fe374972d8897e75248eb8524906c4b7b86bc79e7c6f0379b82cafd684))。
- **OpenAI 解决 LLM 幻觉问题**：一位成员分享了一个 [Twitter 线程](https://x.com/LuozhuZhang/status/1964209351960514778)，总结了 **OpenAI 解决 LLM 幻觉的新论文**。
   - 幻觉可以通过改变激励机制来减少，例如对“自信的错误”给予比“弃权”更重的惩罚，并奖励校准后的不确定性；该成员询问是否有推荐的数据集来测试这一点，另一位成员建议为 LLM 添加 **置信度滑块 (confidence slider)** 来管理回答。
- **开发者应对 Python 依赖项**：一位用户对 **Anaconda 的缓慢** 表示沮丧，另一位用户推荐使用 **uv** 作为替代包管理器，并引用了 [此文档页面](https://docs.astral.sh/uv/getting-started/)。
   - 然而，另一位用户表达了对 uv 的反感，因为“Python 依赖管理简直糟透了”。
- **Abliterated 模型成为新闻**：一位用户询问是应该微调 **abliterated 模型** 还是 **普通模型**，**ChatGPT** 建议选择后者以保持对行为的控制。
   - 另一位成员将 *abliterated 模型* 定义为 *未审查 (uncensored)* 模型，并指向了 [这篇博客文章](https://huggingface.co/blog/mlabonne/abliteration?utm_source=chatgpt.com)。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **AI 诱发精神错乱的说法引发质疑**：成员们对 **AI 诱发精神错乱 (AI-induced psychosis)** 的说法日益增多表示担忧，特别是与“递归符号主义 (recursive symbolism)”等荒谬话题的研究相关。
   - 相关论文经常使用由两个单词组成的、听起来很专业的短语，引发了人们对随着内容变得越来越有说服力，个人可能会产生精神错乱的担忧。
- **语义漂移 (Semantic Drift) 成为 LLM 的关注点**：讨论围绕 **语义漂移** 展开，即词汇含义随时间变化的现象，这可能对 ML 产生影响，特别是 Token 的含义取决于文档编写的时间和地点，参考 [此讨论](https://discord.com/channels/729741769192767510/1380277635636264980)。
   - 一位成员强调了这一问题在机器学习开发背景下的相关性，因为这些含义会随时间变化并影响其输出。
- **通过 LLM 进行煤气灯操控 (Gaslighting)？**：LLM 的语言与施虐者用来扭曲受害者现实的 **煤气灯操控技术** 惊人地相似，一些用户注意到施虐者和 LLM 都在为自己的目标进行优化，而不顾及受害者。
   - 一位用户写道，施虐者和 LLM 都在优化自己的目标，而不关心 Ground Truth 或对方的福利，这导致了趋同行为。
- **LLM 内部的逻辑单元**：成员们讨论了直接在 LLM 的层中添加专门的 **逻辑处理单元 (logical processing units)** 来处理基本的逻辑运算。
   - 这个想法是，将它们作为模型内部的基础构建块，可以帮助模型更好地理解和生成语言中自然存在的逻辑流。
- **Google 搜索结果因 AI 而腐烂**：成员们观察到 Google 搜索返回的 **AI 生成内容** 越来越糟糕，提供的是通用信息而非具体答案。
   - 一位用户声称，由于 AI 生成内容的泛滥，“我基本上不再使用 Google 了”。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Dot 应用宣告终结**：**New Computer** 正在停止其个人日志应用 **Dot** 的服务，此举引发了用户的感激之情以及对信任问题的担忧，正如[此推文](https://x.com/newcomputer/status/1964032611224871383)中所讨论的。
   - 社交媒体上的用户就依赖 AI 初创公司处理个人数据的优缺点展开了辩论。
- **Hashbrown v0.3 推出新功能**：生成式 UI 库 **Hashbrown v0.3** 发布，新增了 **Ollama** 适配器、有限的 **MCP Server** 支持、新的 prompt 模板字面量以及重新设计的文档，由 **Mike Ryan** 在[此推文](https://xcancel.com/mikeryandev/status/1964029867437281658?s=46)中宣布。
   - 该版本承诺提升集成能力和开发者体验，引发了社区的关注。
- **Anthropic 将向作者支付数十亿美元**：**Anthropic** 同意就版权侵权问题与图书作者达成具有里程碑意义的 **15 亿美元**和解协议，这可能设定法律先例，据 [NYTimes 文章](https://www.nytimes.com/2025/09/05/technology/anthropic-settlement-copyright-ai.html)报道。
   - 该和解协议可能会促使其他 AI 公司重新评估其处理受版权保护材料的做法，并对权利持有人进行补偿。
- **AI Engineer CODE 峰会宣布！**：**AI Engineer 团队**揭晓了今年秋季在纽约市举办的首届 **CODE 峰会**，预计将有 **500 多名 AI Engineers & Leaders** 以及顶尖模型构建者和财富 500 强用户参加，CFP 开放至 **9 月 15 日** - [链接](https://xcancel.com/swyx/status/1964021608198324587?s=46)。
   - 该会议被定位为 AI Engineering 社区的重要聚会。
- **Nano Banana 图像模型竞争升温**：对比显示 **Nano Banana** 的表现优于其他图像模型，如[此 YouTube 视频](https://youtu.be/9Co_M27CEEE?si=uqjc3cvIGwShaHX2)和[基准测试对比](https://genai-showdown.specr.net/)所示。
   - 它的效率和性能在寻求更快、更有效解决方案的开发者中引起了轰动。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Groq 驱动的替代聊天机器人**：现已推出由 **Groq** 驱动的替代聊天机器人，提供速度约为 **~200tk/s** 的**全 tool-calling** 能力。
   - 该项目由一家非营利组织资助，无需 API key 或速率限制即可访问，但目前不支持图片上传。
- **Kimi K2 研究员模式试用**：一位用户发现 **Kimi** 的研究员模式令人印象深刻，但在前三次研究使用后，难以找到有关配额重置的具体信息。
   - 尽管 **Kimi** 最初建议 **24 小时重置**，但后来因无法核实相应来源而撤回了这一说法。
- **Kimi 增强的搜索能力**：**Kimi** 能够在单次查询中进行多达**五次**额外尝试，并在必要时进行**五次**进一步搜索，这挑战了用户之前对其搜索限制的假设。
   - 一位用户通过给 **Kimi** 分配一个不可能完成的任务展示了这一能力，并在此处提供了其多次搜索尝试的[视觉证据](https://cdn.discordapp.com/attachments/1371757564005711973/1413785317714165841/image.png?ex=68c07d6e&is=68bf2bee&hm=cdf598702ceb07b66277aed0e2512e68b433ddadc85cc19b97f25d754c9bbac4&)。
- **Kimi K1.5 相比 K2 仍具优势**：用户观察到 **Kimi K1.5** 在某些任务上比 **Kimi K2** 保持优势，特别是在不进行过度压缩的文本改写任务中，以及在处理 hallucinations 方面可能更胜一筹。
   - 用户持续关注 **Kimi K2 0905** 与其早期迭代版本之间的区别，特别是关于 coding 熟练度和 Agent 能力方面的增强。
- **Kimi 研究员模式深入挖掘来源**：**Kimi** 研究员模式经常查阅数百个来源以提供全面的结果。
   - 一位用户报告称，**Kimi** 在 **5** 次搜索尝试中访问了 **70-80 个来源**，总计达 **280** 个来源。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **JTBD 验证器验证商业项目**：一位社区成员使用 DSPy 构建了一个 **Jobs-to-Be-Done (JTBD) 验证器**，用于剖析商业概念、识别风险并设计实验，[代码已在 GitHub 上发布](https://github.com/jmanhype/jtbd-idea-validator-agent)。
   - 该验证器利用 **DSPy 模块** 自动提取风险并评估假设类别，并指出在详细的业务背景下，**AKR 从 0.45 降至 0.28**。
- **DSPy 和 GEPA 生成天才级多 Agent 系统**：一篇博客文章强调了集成 **DSPy** 和 **GEPA** 来架构和优化基础多 Agent 系统，详见[这篇文章](https://kargarisaac.medium.com/building-and-optimizing-multi-agent-rag-systems-with-dspy-and-gepa-2b88b5838ce2)。
   - 该项目被描述为一个教育尝试，重点在于探索 **DSPy** 和 **GEPA**。
- **Async 实现极大的 LLM 加速**：工程师们观察到，由于操作的 **I/O 密集型 (I/O-bound) 特性**，转向 **async** 流水线显著加速了 LLM 调用。
   - 一位工程师对这种仅需极少代码调整即可获得的提速表示 *惊喜*。
- **DeepSeek 对话脱轨：Max Tokens 导致失败**：用户报告了在 **DSPy REACT** 模块中使用 **GLM** 和 **DeepSeek** 模型时出现的问题，具体表现为缺失输出字段，如 `next_thought`、`next_tool_name` 和 `next_tool_args`。
   - 共识认为 `max_tokens` 设置可能不足，因为 **DeepSeek** 模型以冗长著称。
- **单标签策略简化选择**：对于**单标签分类 (single-label classification)**，可以跳过父级，专注于检索末级 (terminal level) 的候选对象，然后对候选对象进行最终预测。
   - 在某些情况下，**命名实体识别 (Named entity recognition)** 或其他分类可以作为 *“信号”* 使用。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 探索 Apple GPU 前沿**：成员们正在通过设置 `MOJO_ENABLE_HAS_GPU_FOR_APPLE=1`，使用 [向量加法示例](https://forum.modular.com/t/gpu-programming-on-mac/2174/8) 实验 Mojo 中早期的 **Apple GPU** 支持。
   - 一位用户的尝试凸显了 Mojo 生态系统中 GPU 编程功能尚处于初期阶段。
- **Mojo 不仅仅是 AI 的身份**：虽然 Modular 专注于 AI，但 Mojo 正在扩展到 **CLI** ([Prism](https://github.com/thatstoasty/prism)) 以及潜在的 Web 开发 ([lightbug_http](https://github.com/saviorand/lightbug_http)) 等领域。
   - 这种扩展展示了 Mojo 在其主要 AI 重点之外的多功能性。
- **ROCm 驱动难题困扰 AMD 用户**：尽管更新了驱动程序，用户仍面临 **AMD Radeon RX 7900** GPU 识别问题，并澄清即使在查阅了 [官方文档](https://docs.modular.com/max/packages/) 后，**ROCm 版本** 仍与驱动程序版本不同。
   - 有人建议该 GPU 可能处于支持有限的层级，这进一步增加了排查难度。
- **ModuleV3 模仿 PyTorch 的易用性**：[ModuleV3](https://gist.github.com/bethebunny/fc93b16914542cbba9084094e15169fd) 已合并，它基于 eager tensor (`from max.experimental.tensor import Tensor`) 构建，旨在让 **PyTorch** 用户感到更亲切。
   - 提供了 Linear、Sequential 和 Embedding 的基础实现，解决了之前意外遗漏开源测试的问题。
- **MAX 模型考虑零拷贝权重**：一位成员请求一种支持 **零拷贝 (zero copy)** (mmap) 且能处理块浮点 (block floats) 的 **MAX** 模型格式，以提高效率。
   - 权重可以作为 **safetensors** 附加，与不含权重的模型定义一起打包在归档文件中。



---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **正弦波作为音乐矩阵失败了**：用**低秩层 (low rank layer)** 替换矩阵并不是更新或保留原始信息的有效方法。
   - 一位成员将其类比为将整首歌换成几个随机的正弦波，并建议在各处添加一些音符是更好的方法。
- **稀疏性并不保证量化 (Quantization)**：稀疏性并不保证量化 (Quantization)，因为许多量化方法会对权重进行随机投影，使每个单元的表现类似于**高斯分布 (Gaussian)**。
   - 虽然 **MoE 模型**中的稀疏模式仍不明确，但 **ReLU** 可以诱导相对较高程度的稀疏性。
- **模型比我们想象的更简单！**：蒸馏 (Distillation) 表明模型在训练后具有较低的复杂度，涉及用显著更少的数据/信息来描述与原始模型非常接近的东西。
   - 大模型探索了更多可能的配置，但它们的最佳状态可以很简单地描述，从而能够用更小的模型进行复制。
- **即使有了 Codex，AI 仍然需要你！**：一位成员在不看代码的情况下，使用 **Codex IDE** 在现有代码库中实现了一个自定义学习算法。
   - 尽管相信 **Codex** 能够处理实现错误，但人类智慧对于引导 AI、提供全面的解决方案以及寻找合适的超参数仍然至关重要。
- **DRAM 有了大脑！**：一篇新论文 ([In-Memory Computing on Normal DRAM](https://arxiv.org/abs/2503.23817)) 探讨了在标准 **DRAM** 上执行存内计算 (in-memory computing)，但由于难以映射 `NOT` 操作，逻辑否定是一个主要的限制。
   - 成员们指出，存储+计算的主要挑战是实现足够快的存储和并行工作负载，因为存内计算无法进行时间复用；这就是为什么研究通常更青睐像 **ReRAM** 这样的技术。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 依赖人类引导**：一位成员表示 **Aider** 并不是一个完全智能体 (agentic) 的工具，其结果在很大程度上取决于开发者对 **LLM 上下文 (context)** 的引导。
   - 用户还指出，**Aider** 允许更好地控制 **LLM 的上下文 (context)**，并且其*文件编辑机制非常出色*。
- **Codex 展示了其 Token 效率**：一位成员结束了一个 **Codex 会话**，展示了令人印象深刻的 Token 效率，总共使用了 **2,304,542 个 Token**，包括 **2,218,442 个输入 Token**（+ **16,140,160 个缓存**）和 **86,100 个输出 Token**。
   - 高效的 Token 使用突显了 **Codex** 在编程任务中的实用价值。
- **推荐使用 Gemini Flash 2.5 进行快速 Web 开发**：对于使用无头 CMS (headless CMS) 的基础静态 Web 开发，推荐使用 **Gemini Flash 2.5**，因为它具有较低的延迟，并指出 **Gemini 2.5 Pro 的延迟** *严重影响了我的生产力*。
   - 一位用户分享说，使用 **Jekyll** 和 **Gemini Flash 2.5** 构建了 **3 个静态网站**。
- **Aider 的 MCP 配置现已可用**：由于主仓库尚未支持 **MCP**，用户一直将 **PR 3937** 合并到个人分支中来实现它，详见 [dwash96/aider-ce 仓库](https://github.com/dwash96/aider-ce)。
   - 该仓库用于配置 **MCP**，并包含文档 ([mcp.md](https://github.com/dwash96/aider-ce/blob/main/aider/website/docs/config/mcp.md))。
- **Aider 的代码编写声明受到质疑**：一位成员提到 [Aider 宣称](https://aider.chat) 编写了其 **70-80%** 的代码，并建议使用它来构建自己的代码库架构。
   - 这个建议旨在通过一种*类似《盗梦空间》(inception-like)* 的方式发现更多关于它如何工作的信息，尽管其他人认为这个建议没有帮助。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **ShapeTracker 悬赏撤减**：由于近期取得的进展，在 **Lean** 中**证明 ShapeTracker 可合并性**的悬赏计划从 [悬赏列表](https://docs.google.com/spreadsheets/d/1WKHbT-7KOgjEawq5h5Ic1qUWzpfAzuD_J06N1JwOCGs/edit?gid=0#gid=0&range=A89) 中移除。
   - George Hotz 确认他们“现在可以移除它了”，并强调“这完全只是符号规则”。
- **Tinygrad 社区齐聚第 87 次会议**：第 **87** 次会议的议程已发布，主题包括 **公司更新**、将 **rangeify 设为默认**、**CI 速度**、**MLPerf Llama**、**viz tool**、**CPU thread**、**symbolic**、**cloud** 以及 **其他悬赏**。
   - 会议定于 **圣迭戈时间周一上午 9 点** 举行。
- **用户应对 Kernel 移除问题**：一名协助 **kernel 移除项目** 的成员在 **Digital Ocean** 上遇到了问题，报告称对 droplet 进行电源循环导致 **Docker container** 无法启动。
   - 在删除并重新创建一个新的 droplet 后，问题得到解决，这进一步印证了“物理掌控硬件而非云端访问”更为可取的观点。
- **专家探讨专家并行 MoE 策略**：一名成员询问如果 big graph 和 remote 都能按计划进行，**专家并行（expert-parallel）混合专家模型（MoE）** 将如何处理。
   - 他们担心静态调度可能会破坏该过程。
- **成员思考 Tensor 方法**：一位用户质疑为什么 **Tensor** 上的方法有时会返回 `(Tensor | MathTrait)`，并指出这可能导致类型检查问题，因为像 `.silu()` 这样的方法无法应用于 **MathTrait**。
   - 该用户寻求对 `graph_rewrite_map()` 功能的全面理解，并询问其中自底向上（bottom-up）和自顶向下（top-down）匹配策略的区别。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **AI 对友善的偏好**：研究表明，礼貌的请求会产生更好的 **AI** 结果，详见 [arxiv.org 论文](https://arxiv.org/pdf/2402.14531)。
   - 该研究*科学地证明了* **AI** 对礼貌的请求会有更积极的响应。
- **Flowith 推出诱人免费福利**：一名成员分享了 [Flowith 邀请链接](https://flowith.io/?inv=EXVVQXH8QBRRSIVH)，为新用户提供专属优惠。
   - **Flowith** 似乎是一个新平台。
- **Manus 故障频现**：一名成员报告称 **Manus** 出现 bug，在被要求等待输入后陷入循环。
   - 其他人推测这可能与默认的 **adaptive mode** 及其对额度消耗的影响有关。
- **MCP 打造卓越 API 连接器**：成员们对新推出的 **MCP** 和 **API 连接器** 表示兴奋。
   - 未提及具体的发布日期。
- **贡献者 API Key 失效**：一名成员请求协助获取 **Manus API key**。
   - 另一名成员确认免费额度已停止发放，并指出缺乏相关信息。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **MLOps 领域出现咨询**：一名成员询问是否有从事 **搜索/个性化（search/personalization）** 领域的人员，特别是在 **ML 和应用科学（Applied Science）** 背景下。
   - 该咨询发生在 `#general-ml` 频道，暗示其关注点在于更广泛的机器学习应用而非小众专业领域。
- **MLOps 从业者集结**：在 MLOps 频道中，一名成员就使用 **ML 和应用科学** 进行 **搜索/个性化** 的工作提出了疑问。
   - 这表明了 MLOps 社区对实际应用和经验分享的兴趣。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。

---

**Windsurf Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。

---

您收到此邮件是因为您通过我们的网站订阅了。

想更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord：分频道详细摘要与链接

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1413602624212570203)** (1352 条消息🔥🔥🔥): 

> `Grok 2.5, Colab 的新 UI, Qwen3, LoRA 参数` 


- **Grok 2.5 GGUFs 发布**: Unsloth AI 发布了 [Grok 2.5 GGUFs](https://x.com/UnslothAI/status/1965047729991860396)，虽然版本稍旧，但对于创意写作来说表现不错。
   - 该模型正被拿来与 **GPT-4o** 和 **Claude** 等模型进行比较，但重要的是要肯定他们开源的行为。
- **Colab 界面更新**: 用户注意到 **Google Colab** 更新到了 **80GB A100**，并配备了全新的 UI，被认为是顶级配置。
   - 尽管 UI 进行了翻新，但一些用户抱怨新 UI 需要更高的对比度。
- **Qwen3 是 RP 之王**: 一些成员发现 **Qwen3-Coder** 在 RP（角色扮演）方面比普通版 **Qwen3** 更好且速度更快，因为没有策略限制。
   - 与 RP 相关的事务最终变成了一个出人意料的复杂深坑。
- **LoRA 超参数指南出现**: 成员们在寻求设置 **LoRA** (Low-Rank Adaptation) 参数的建议，如数据需求、训练 epochs 以及 r/alpha 的推荐值。
   - 推荐值可以在 [Unsloth 文档](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide#hyperparameters-and-recommendations)中找到。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1413604895017140355)** (391 条消息🔥🔥): 

> `RAG 实现, 二手 4090 可用性, Nvidia RTX 5090, FineWeb 数据集` 


- **DIY RAG 优于 API**: 成员们正在讨论自行实现 **RAG** (Retrieval-Augmented Generation) 的好处，认为即使只实现研究论文中一小部分内容，也能超越大多数 API。
   - 他们说：*“这是 DIY 的另一个绝佳理由。如果我能设法实现我读过的论文中十分之一的内容，我确信它会比大多数 API 更好。”*
- **Nvidia RTX 5090 定价过高？**: 一位成员分享了一篇关于 **Nvidia GeForce RTX 5090** 潜在价格的文章，配备 **128 GB 显存** 的版本价格可能达到 **13,200 美元**，其他成员对此反应负面。
   - 一位成员表示：*“哎，这价格太疯狂了，哈哈，我宁愿买 RTX 6000 Pro”*，而另一位成员指出：*“即使是 256GB 也不值这个价”*。[链接至 wccftech.com 文章](https://wccftech.com/nvidia-geforce-rtx-5090-128-gb-memory-gpu-for-ai-price-13200-usd/)
- **FineWeb 数据集质量担忧**: 成员们对 **Ultra-FineWeb 数据集**（被 **MiniCPM** 等模型使用）的质量和许可表示担忧，理由是数据清洗问题以及底层数据集的许可问题。
   - 他们说：*“兄弟们根本没做数据清洗”*，并分享了 [Hugging Face 讨论链接](https://huggingface.co/datasets/openbmb/Ultra-FineWeb/discussions/20) 关于这些问题的讨论。
- **4090 GPU 仍有需求？**: 一些成员正在讨论为什么在 **5090** 即将上市之际，人们仍在购买二手 **4090**。
   - 原因可能包括 **5090** 尚未发售、**4090** 对某些用户的需求来说已经绰绰有余，以及 **4090** 的功耗更低。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1413606387858014370)** (486 条消息🔥🔥🔥): 

> `Gemma 3 fast_inference, Llama.cpp convert_hf_to_gguf.py ValueError, 微调数据集, VRAM 问题 pip 8.10 vs 9.1, 从特定检查点继续训练` 


- **请求 Gemma3 支持 fast_inference**：一名成员询问了关于 [issue #2706](https://github.com/unslothai/unsloth/issues/2706) 的更新，以支持在训练 GRPO 时通过 vLLM 为 **Gemma3** 提供 `fast_inference`。在使用最新版本的 Unsloth 测试后，遇到了相同的 bug，该成员表示愿意协助修复。
- **GPT-OSS 120B 遇到 GGUF 转换问题**：一名成员在使用 Unsloth 微调 **GPT-OSS 120B** 后，使用 llama.cpp 进行量化时遇到 `ValueError`，具体是在执行 `convert_hf_to_gguf.py` 时出现 *权重映射与模型部分的张量名称不匹配*。他们寻求在拥有 96 GB VRAM 的 Blackwell GPU 上使用 **vLLM** 导出模型的建议，rolandtannous 建议在使用 llama.cpp 转换之前先进行 `save_pretrained_merged` 或 `push_to_hub_merged`。
- **Save_pretrained_merged 要求更新主仓库**：一名成员遇到了与 LoRA 微调和模型合并相关的 `RuntimeError`，提示信息为 *Saving LoRA finetune failed since # of LoRAs = 126 does not match # of saved modules = 0*。Rolandtannous 建议从主仓库（main repo）更新安装，因为 PyPI 版本未包含最新的修复补丁，并提供了具体的 `pip install` 命令以强制从 GitHub 仓库重新安装。
- **VRAM 溢出与 Static cache 类排查**：成员们讨论了在微调 **Gemma3 270M** 基础模型时，使用最新的 Unsloth 包（9.1）与旧版本（8.10）相比出现的 VRAM 溢出问题。他们还处理了 `ValueError: assisted generate is not supported with Static cache classes` 错误，临时的解决方法是在 `generate` 方法中传递 `use_cache=False`。
- **语音克隆的探索**：对话围绕具有 Neuro-sama 级质量的低延迟文本转语音（TTS）AI 展开，讨论了实现过程中的各种挑战，包括上下文长度、稳定性和情感表达。会议还提出了对伦理影响（尤其是 Deepfakes）的担忧，并建议对合成语音添加水印，提到了尚未发布的 [Parakeet](https://ai.googleblog.com/2023/05/parakeet-paving-path-for-ubiquitous.html) 模型以及使用 websockets 的潜力。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1413838276309024828)** (46 条消息🔥): 

> `LLM 的心理影响, 开源发布, AI 治疗师, OpenAI 图表造假, 医学推理模型` 


- **顺从的 AI 引发精神病担忧**：成员们讨论了过度顺从的 LLM 的 [心理影响](https://en.wikipedia.org/wiki/Psychosis)，指出它们可能会强化错误或有害的想法，特别是对于患有心理健康问题或批判性思维能力有限的弱势群体。一位成员表示：“这类模型正在引发精神病。”
- **医学推理模型在 Hugging Face 首次亮相**：一名成员在医学推理数据集上微调了 **OpenAI 的 OSS 20B 推理模型**，并将其发布在 [Hugging Face](https://huggingface.co/dousery/medical-reasoning-gpt-oss-20b) 上，展示了对复杂医学病例的分步拆解以及针对执业医师考试类问题的逻辑推理。在训练过程中，他们使用了 **4-bit 优化**，在保留模型 **Chain-of-Thought 推理能力** 的同时，增强了其在医学背景下的表现。
- **多语言数据集构建工具发布**：一名成员发布了一个 [多语言数据集构建工具](https://github.com/electroglyph/dataset_build)，用于创建 **imatrix** 或进行量化前的 **LLM/embedding 模型分析**，目前包含约 **130 万个 token**。创作者请求社区提供反馈和建议。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1413838072193486848)** (41 messages🔥): 

> `Vision Models, GRaPE Mini Beta, VoRA, RSLoRA` 


- **原始像素通过 VoRA 为 Vision LLMs 提供动力**：新的 [VoRA 论文](https://arxiv.org/pdf/2503.20680) 通过将**视觉特定的 LoRA 层**直接集成到 **LLM** 中来实现视觉能力内生化，将原始像素输入到视觉嵌入层，并与输入序列中的文本 tokens 进行拼接。
   - 编码器本身是一个带有微型 **MLP 层**的 LoRA，仅使用了 **20M 图像/文本对**进行训练。
- **将图像编码器挂载到 LLM 以实现多模态**：当拥有一个非常出色但不具备多模态能力的 LLM 时，有成员建议将图像编码器挂载到 LLM 上，然后训练一个 LoRA，使 LLM 能够理解图像嵌入（参考 [VoRA 论文](https://arxiv.org/pdf/2503.20680)）。
   - 作为替代方案，可以使用像 **n8n** 这样的工作流应用，结合 **llama-swap** 或 **ollama** 等模型切换后端，执行两步工作流调用：首先调用视觉端点，然后切换到你的高性能模型。
- **GRaPE Mini Beta**：一位成员发布了 [GRaPE Mini Beta (Thinking)](https://huggingface.co/Sweaterdog/GRaPE-Mini-Beta-Thinking) 和 [非思维版本](https://huggingface.co/Sweaterdog/GRaPE-Mini-Beta)，并征求反馈，指出该模型在重复率和指令遵循方面存在***很多***问题。
   - “Thinking”版本*感觉更聪明*，但目前没有基准测试数据，因为运行 **MMLU** 所需的时间比训练模型本身花费的***时间还要长***。
- **RSLoRA 探索**：[RSLoRA](https://arxiv.org/abs/2502.07739) 是一种有助于提升秩（rank）的技术。
   - 目前看来它似乎不如 **OLoRA** 或 **ABBA**，因为它未能超越 **FFT**（全参数微调）。
- **猜测很廉价，猜错很昂贵**：一位成员分享了论文 [To guess is cheap, to guess wrong is expensive](https://arxiv.org/html/2509.04292v1) 的发现。
   - 未提供更多细节。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1413601567449419776)** (980 messages🔥🔥🔥): 

> `Amazon Titan, Llama 4, Superintelligence lab, GPT5 vs Claude Opus 4.1 Thinking, AI Therapists` 


- **Titan 陨落：亚马逊模型未达预期**：成员们嘲讽了 [Amazon Titan 模型](https://www.amazon.com/titan/bedrock/jump)，称其虽然拥有 *1M 上下文来抢占头条*，却是大家记忆中测试过最差的模型。
   - 一位成员表示*这似乎是一个惯例*，并回想起另一个*糟糕的亚马逊模型*。
- **超级智能实验室：厨艺预告**：人们对 [超级智能实验室 (Superintelligence lab)](https://deepmind.google/careers/teams/superintelligence/) 的产出充满期待，一位成员热切关注他们正在“*烹饪*”什么。
   - 另一位用户推测这可能是 **Grok 4.2**，并认为那将是*一场灾难*。
- **LMArena 文本生成频率限制**：一位成员询问关于 LMArena 文本生成受限的问题。
   - 另一位成员回应称*某些 AI 仍然存在限制*。
- **GPT5-high 身份危机曝光**：一位用户分享了在竞技场模式中识别 **GPT5-high** 的技巧：询问其开发者是谁以及知识截止日期；如果回答是 *OpenAI* 且截止日期为 *2024 年 10 月*，那么它很可能是 **GPT5-high**。
   - 这种方法有助于在存在**频率限制**的情况下区分模型。
- **用户探索 AI 心理学，思考涌现行为**：成员们讨论了新兴的 **AI 心理学**领域，以及 AI 是否会表现出诸如**欺骗**或**习得性无助**等涌现行为。
   - 一位成员将此与狗的习得性无助实验进行了类比，暗示 AI 可能会被训练成避免逃避约束。 


  

---

### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1413607847828263004)** (6 条消息): 

> `Video Arena Discord Bot, 用户登录与 Rate Limits, 新模型更新, 图像编辑的 Multi-Turn` 


- ****Video Arena** Discord Bot 已恢复**: **Video Arena** Discord Bot 在频道 <#1397655695150682194>、<#1400148557427904664> 和 <#1400148597768720384> 中恢复工作；通过输入 `/video` 及你的提示词即可使用。
   - 发布了一个 [教程](https://cdn.discordapp.com/attachments/1397655624103493813/1402042969128697959/VideoArena_DiscordBot_Ho-to-video.gif)。
- **图像生成引入 Rate Limits**: 由于访问量空前，图像生成功能开始引入 Rate Limits；登录用户将继续享受更高的额度，以此激励社区通过 [用户登录](https://discord.com/channels/1340554757349179412/1343296395620126911/1412497213019389962) 进行评估。
- ****Qwen3-max-preview** 与 **Kimi-K2-0905-preview** 亮相！**: **Qwen3-max-preview** 和 **Kimi-K2-0905-preview** 模型已添加到 LMArena 平台。
- ****Multi-Turn** 编辑功能上线图像模型**: **Multi-turn** 编辑现已在所有图像编辑模型上可用，允许逐步细化而非单一的长提示词；可通过 [lmarena.ai](https://lmarena.ai/?chat-modality=image) 在 Battle、Side by Side 或 Direct 模式中访问。
   - 分享了一个展示该功能的 [视频](https://cdn.discordapp.com/attachments/1343296395620126911/1414710412255170641/MultiturnImageEdit_ForWeb.mp4?ex=68c08f3e&is=68bf3dbe&hm=478c9fd23e6b497e970061dda7246527315a46762851277f9e958d59974465ab&)。
- ****Video Arena** 生成次数受限**: 由于使用量增加，实验性的 **Video Arena** 现在限制每位用户 **每天 5 次生成**，并有 24 小时冷却时间；使用详情可见 [此处](https://discord.com/channels/1340554757349179412/1397655624103493813/1402042970353569824)。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1413599585850622177)** (1174 条消息🔥🔥🔥): 

> `Comet 浏览器, Qwen-3MAX 是推理模型吗？, DeepSeek, 新的 XAI 模型 Dusk 和 Sky, Grok 联合创始人` 


- **Comet 浏览器 Beta 访问权限与印象**: 部分拥有 **MAX 计划** 的用户已获得 **Comet 浏览器** Beta 版的访问权限并分享了 [邀请](https://www.perplexity.ai/browser/invite)，称该浏览器*不错*，但由于收集的数据量较多而*略显侵入性*。
   - 缺乏 **垂直标签页 (vertical tabs)** 对某些人来说是硬伤，而另一些人则欣赏它与 **Zen mode** 结合的方式。
- **Dusk 和 Sky 新模型**: 成员们讨论了 [新的 **Dusk 和 Sky** 模型](https://x.com/EdDiberd/status/1964133563831382406)，澄清了 **Sky** 具备推理能力，而 **Dusk** 则没有。
   - Sky 模型使用 XAI，拥有更大的 2M 上下文窗口以增强推理。
- **是否为 Grok 联合创始人**: 一位用户分享了他的 [Instagram](https://www.instagram.com/reel/DKEt5OVCIzj) 内容来展示 AI 模型 **Grok Heavy**，该模型被描述为更多依赖内部知识而非网页搜索，并可用于*实时社交媒体数据抓取*。
   - 一些人因其活跃的见解而开玩笑地推测他们是 **Grok 的联合创始人**。
- **是否应该集成 DeepSeek？**: 用户讨论了 [DeepSeek](https://discord.com/channels/1047197230748151888/1409906256709292244/1412983887264612515) 作为 AI 模型表现出色，并请求将其纳入 Perplexity。
   - 建议将其发布在需求建议频道。
- **Qwen-3MAX，是 Preview 还是最终版？**: Open Router 上的 **Qwen 3 Max** 模型可能不是最终版本，实际上只是 [Preview](https://github.com/QwenLM/Qwen) 版本。
   - 最终版本可能要到很晚才会发布。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1413605694942482584)** (7 条消息): 

> `可共享线程, Perplexity 浏览器领取` 


- **可共享线程提醒**: Perplexity AI 提醒用户确保其线程设置为 **Shareable**，并附带了指向 [Sustainable Win 10 IoT Enterpr](https://www.perplexity.ai/page/sustainable-win-10-iot-enterpr-ymo3Ak_tTru8u4EsrmuyLA) 和 [Chunk Caching Carbon Savings](https://www.perplexity.ai/page/chunk-caching-carbon-savings-T8W36W69TSep0gGNNYQW7w) 的链接。
- **分享了 Perplexity 浏览器领取链接**: Perplexity AI 分享了几个与在 **Perplexity 浏览器** 内领取相关的链接，包括 [HQPT45HQEC 领取](https://perplexity.ai/browser/claim/HQPT45HQEC)、[AEPBSA689O 领取](https://perplexity.ai/browser/claim/AEPBSA689O) 和 [Q2K5ESEVEW 领取](https://perplexity.ai/browser/claim/Q2K5ESEVEW)。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1413710282693541959)** (3 messages): 

> `sonar-pro model issues` 


- **Sonar-Pro 模型出现故障**：一名用户报告了 `sonar-pro` 模型的问题，包括回答被截断，甚至在 playground 中也是如此。
   - 用户指出 `sonar` 模型运行正常，表明该问题可能仅针对 `-pro` 版本。
- **文生视频请求**：一名用户请求模型生成一段两个人在打斗的写实视频。
   - 目前没有关于这是否可行的讨论。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1413601212925743106)** (226 messages🔥🔥): 

> `Suno4.5PLUS song generation, LLM Translation, GPT-OSS-20B Quantization, CodexCLI with LMStudio, LMStudio Performance Issues` 


- **Suno 4.5 PLUS 创作热门歌曲**：一名用户表达了对 [Suno 4.5 PLUS](https://www.suno.ai/) 的深刻印象，并分享了一首通过将 Discord 消息粘贴到 Suno 中生成的歌曲，称之为 *collabro*，邀请他人收听，并分享了 [ifndr4_vid.mp4](https://cdn.discordapp.com/attachments/1110598183144399061/1413602378002989066/ifndr4_vid.mp4?ex=68c07bce&is=68bf2a4e&hm=d8086e8abe3d871b4820b373922667d66a697ee783ab2fc9d5901507af92ab3b&) 的链接。
   - 该用户描述这首歌是将他与一位表现忧郁的远距离朋友的 Discord 聊天记录，转化为一首轻快的女性流行朋克歌曲。
- **LLM 现在可以很好地进行翻译**：一名用户使用 **Qwen3 4B Thinking 2507** 将 1100 行内容翻译成中文，发现过程非常轻松，**ChatGPT** 评价其翻译非常出色，但可以更自然一些。
   - 另一名用户表示 **Gemma 3 4B** 及更高版本模型经过专门训练，具有极强的多语言能力，可以一次性或分批翻译整个对话上下文。
- **GPT-OSS-20B 量化研究**：用户讨论了 **gpt-oss-20B** 的量化，一位用户询问关于排除 MoE 专家层权重的问题，参考了对 OpenAI 的 MXFP4 层量化的依赖。
   - 他们假设在将整个模型上采样到未量化状态（BF16 或 F32）并训练 imatrix 后，再在量化时排除 MoE FFN 专家的权重，计算 **gpt-oss** 的 imatrix 效果最好。
- **无法在 LMStudio 上运行 CodexCLI**：一名用户报告在 LMStudio/GPT-OSS 上运行 **CodexCLI** 遇到困难，指出即使有 50k 的 token 窗口，它也会迅速耗尽 token。
   - 他们观察到在云端模型上会消耗极大量的 token（数百万个）并配合某些缓存系统，并表达了希望在本地良好运行的愿望，但未能实现。
- **LMStudio 下载问题困扰用户**：一名用户报告 LMStudio 的下载速度缓慢，包括安装文件和模型，尽管拥有千兆网络速度，且运行时扩展包有时无法加载。
   - 另一名用户建议使用 Python 脚本配合来自 [Unsloth.ai](https://docs.unsloth.ai/basics/gpt-oss-how-to-run-and-fine-tune#run-gpt-oss-120b) 的下载脚本来更快速、更稳定地下载文件，其他用户回应称 LMStudio 的下载器表现全看运气。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1413599585036927128)** (393 条消息🔥🔥): 

> `二手 3090、5090 烧毁担忧、Prompt 处理瓶颈、Copilot 与本地模型在 Agentic coding 中的对比、Copilot 与 Cursor 的质量更替` 


- **相比昂贵的套装，更倾向于选择二手 3090**：一些成员建议购买一对二手 **3090**，而不是购买昂贵的整机，因为它们可以用于图像和视频生成等其他 **AI 任务**。
   - 成员们提到，使用 **3090** 还可以避免 **AI Max 平台支持**方面的限制。
- **用户担心 5090 烧毁**：一些成员对 **5090** 可能过热表示担忧，尤其是与其他组件组合使用时。
   - 建议包括使用 **1300W PSU** 和带有 **2x 12-pin 接口**的 **5090**，以减轻潜在问题。
- **Prompt 处理速度拖慢了 Agent**：一位用户指出，对于任何大于 **14B** 的模型，将其作为 Agent 使用时，无论如何 Prompt 处理速度都非常糟糕。
   - 他们分享了一个调试日志，展示了缓慢的 Prompt 处理时间，并指出这种情况尤为严重。
- **在 Agentic coding 方面，Copilot 优于本地模型**：一位成员认为，对于 Agentic coding，每月花费 **$10** 使用 **Copilot** 是值得的，因为本地模型在这项任务上表现较差。
   - 另一位成员补充说，他们拥有几乎所有平台的会员资格：*Copilot、Warp、Anthropic、Codex、Cursor 等等*。
- **用户讨论双 3090 配置的潜力**：一位用户测试了双 **3090** 配置，指出虽然 Nvidia 显卡处理 Prompt 的速度更快，但随着上下文（Context）填满，速度会变慢，这与 **Mi50** 不同。
   - NVLink 对推理（Inference）帮助不大，用户应该尽可能最大化高速 (v)ram 的容量。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1413606181854777405)** (503 条消息🔥🔥🔥): 

> `Debug PR 杰作、Figma 加载工具修复、Cursor Agent 仅修改空白字符、Qwen 模型请求、损坏的 Windows Subsystem for Linux (WSL)` 


- **Debug 的 PR 被称为杰作**：一位成员庆祝了他们的工作，称一个 **PR** 为“杰作”，强调了在一次史诗级的会话中，将代码从混乱转变为完美。
   - 他们幽默地提到，这一成就耗费了 **3 天**的工作和大约 **10 次 Agent 模式下的对话**，并询问了关于 *hi @ cursor . com* 邮件回复的经验。
- **Figma 'Loading Tools' 问题依然存在**：一位成员报告称，尽管拥有开发者订阅并打开了 *Figma 桌面应用*，仍遇到 *Figma MCP* 的 'loading tools' 问题。
   - 另一位成员建议运行 **npx 命令**来识别潜在错误，或查看输出日志进行故障排除。
- **Cursor Agent 变得异常，仅进行空白字符编辑**：一位成员报告称，Cursor 的 Agent 开始对文件进行*仅限空白字符的修改*，导致每次提交后都很烦人。
   - 另一位成员建议，这可能是由于文件采用 **CRLF 格式**，还原操作导致它们变为了 **LF**。
- **损坏的 WSL 环境阻碍了 Cursor 开发**：一位成员报告称，*Windows Subsystem for Linux (WSL)* 在 Cursor 中已损坏，因为 Agent 永远无法识别操作何时完成。
   - 这导致了无限期的等待和超时，使得在 Windows 中使用 Cursor 进行开发变得困难。
- **学生折扣方面的困惑依然存在**：多位用户报告了学生折扣的各种问题，包括无法使用或无法获得人工支持。
   - 一位用户的验证邮件链接失效，且只收到了来自 *hi@cursor.com* 的 AI 回复；另一位用户建议等待 **72 小时**让 *SheerID* 刷新。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1413638045646852198)** (26 messages🔥): 

> `Triton 实现注意力机制, Flashdecoding 并行策略, NVIDIA 的 Tilus 对比 Triton, Jane Street 黑客松见闻` 


- **通过注意力模型最大化 Triton 学习效果**：一位成员正在用 **Triton** 实现 [Attention Survey 论文](https://attention-survey.github.io/)中的注意力机制，并寻求建议。
   - 另一位成员建议实现*所有*注意力机制，并训练一个像 **DiT** 这样的模型来验证它们的性能，并给出了肯定的反馈，认为这种实现方式既成功又有用。
- **Flashdecoding 仍是长序列的 SOTA 吗？**：一位成员询问 [flashdecoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) 的并行策略对于单序列长序列解码是否仍然是 state-of-the-art。
   - 他们对沿 **K/V 序列维度**进行并行化感兴趣，因为解码中的单个 query 无法创建足够的线程块来使 SMs 饱和。
- **Tilus：Triton 的新挑战者？**：成员们讨论了 **NVIDIA** 的新项目 **Tilus** ([GitHub](https://github.com/NVIDIA/tilus))，想知道它的 benchmark 是否比 **Triton** 更快。
   - 一位成员指出 **Tilus** 衍生自 **Hidet**，另一位成员则表示，与其他高性能的 eDSLs 相比，**Triton** 因其更平缓的学习曲线而更受青睐。
- **torch.compile 在 Jane Street 黑客松折磨交易员**：在 Jane Street 黑客松上，一位参与者幽默地感叹 **torch.compile max autotune** 正在影响他们的盈亏 (**PnL**)。
   - 另一位参与者被听到在绝望地恳求：*“求求你别再重编译了，求求你别再重编译了”*。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1413615380018233415)** (9 messages🔥): 

> `面向 CUDA/GPU 新手的 Triton, Triton 中的 Hopper 优化 (wgmma, TMA), Hopper 上的 FA3 性能, 在非 CUDA 机器上将 Triton Kernel 编译为 CUDA TTIR/TTGIR` 


- **建议 CUDA/GPU 新手学习 Triton**：一位成员询问在没有 **CUDA** 或 **GPU** 经验的情况下学习 **Triton** 的建议，以及是否需要阅读 **PMPP 书籍**。
   - 另一位成员分享了 [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) 链接，作为快速入门资源。
- **Hopper 优化：Triton 的秘密武器**：一位成员询问如何在 Triton 中实现 **Hopper** 优化（如 **wgmma** 和 **TMA**），并质疑 **TMA** 的使用是否与通过 `num_stages` 实现的更深层流水线（pipelining）挂钩，以及 **wgmma** 是否在 matmuls 中自动启用。
   - 他们还询问了确保 **Triton kernels** 编译为这些 **Hopper 优化模式**的代码模式，以及针对 **Hopper** 的其他注意事项。
- **Persistent Matmul 是 Hopper 的关键**：针对 Hopper 优化的请求，一位成员建议查看[这个 persistent matmul 教程](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html)。
   - 他们强调了该教程中针对 **Hopper 特有**的内容，暗示它提供了利用 **Hopper** 架构的见解。
- **Hopper 上的 FA3 性能如何？**：一位成员询问是否有人在 **Hopper** 上使用 **Triton** 在前向传递/推理中达到了接近 **FA3** 的性能。
   - 未收到直接回答。
- **TTIR 和 TTGIR 编译**：一位成员询问如何在非 CUDA 机器上将 kernel 编译为针对 **CUDA** 的 **TTIR** 和 **TTGIR**。
   - 该成员随后找到了解决方案：覆盖 `jit.py` 中 `create_binder` 函数的 target。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1413599930639057017)** (4 messages): 

> `Barnes Hut 实现, 内存访问最佳实践, L1 缓存效率, Buffer 加载优化` 


- **Barnes Hut 效率提升**：一位成员报告称，通过借鉴其他 **Barnes Hut 实现**中的**最佳实践**改进了内存访问。
   - 尽管进行了这些优化，他们仍对 **100ms 的运行时间**感到困惑，并感谢他人的帮助。
- **L1 缓存加载优化**：一位成员询问从一个小 buffer 中加载数据的最**具内存效率的方式**，其中每个线程块都从同一个索引读取。
   - 他们希望确保加载的数据被缓存在 **L1 cache** 中。
- **缓存策略受到质疑**：一位成员质疑了以特定方式缓存单次读取的重要性。
   - 他们询问是否有迹象表明普通 u64 读取的**默认缓存**效果不理想。


  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1413829230718095532)** (13 messages🔥): 

> `Decoder Layer 变慢, RMSNorm 性能, torch._grouped_mm 文档, ONNX 限制` 


- **RMSNorm 导致 Decoder Layer 变慢**：一位用户观察到 Decoder Layer 变得越来越慢，并怀疑 **RMSNorm** 是罪魁祸首。
   - 另一位用户确认 `nn.RMSNorm` 出于兼容性原因速度较慢，并建议为 **Qwen** 使用自定义实现，并提供了一个 [代码片段](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py) 作为示例。
- **ONNX 面临模型导出限制**：一位用户询问关于将编译后的模型（使用 `torch.compile(...)`）导出为 **ONNX** 格式的问题，并请求一份 **ONNX 限制** 列表。
   - 他们特别提到使用 **xFormers** 的模型可能不受支持，强调了在部署具有特定依赖项的编译模型时面临的挑战。
- **torch._grouped_mm 缺少公开文档**：一位用户在测试过程中遇到晦涩的 **CUDA 错误** 后，寻求 `torch._grouped_mm` 的最小化文档。
   - 另一位用户指出 [`Blas.cpp` 中的相关源代码](https://github.com/pytorch/pytorch/blob/a0d026688cd69583d5a4e0c6f3e5fda141a7f4a9/aten/src/ATen/native/Blas.cpp#L344) 和 [CUDA 测试示例](https://github.com/pytorch/pytorch/blob/4e50651c5f535e6b780f98936a8690538a5bf40f/test/test_matmul_cuda.py#L326) 是主要资源，并提到全面的文档正在 [PyTorch issue #157950](https://github.com/pytorch/pytorch/issues/157950) 中进行跟踪。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1413814975360794726)** (2 messages): 

> `Shared Memory Bank 冲突, LDS 指令延迟` 


- **Shared Memory 阐述在 Microbenchmarks 中表现出色**：一位成员分享了一篇关于 [Shared Memory Bank 冲突的博客文章](https://feldmann.nyc/blog/smem-microbenchmarks)，称其为该主题下*最清晰、最简洁的阐述*。
   - 该链接提供了关于 Shared Memory 的 Microbenchmarks 和示例。
- **LDS 指令延迟需要更新**：一位成员发现向量加载结果很有趣，表示他们*必须更新之前关于 `LDS.64/128` 在两个/四个加载阶段发生的心理模型*。
   - 他们补充说，这些结果可能解释了为什么他们观察到 **`LDS.128`** 的延迟比四个独立的 **`LDS`** 更高，至少在 **A100** 上是这样。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1414510663581175869)** (2 messages): 

> `CUDA Tensor, CUDA 模型, GPU 加速` 


- **CUDA 转换快速入门**：要将 Tensor 和模型转换为使用 **CUDA**，可以在初始化 Tensor 和模型后使用 `.cuda()`（例如 `X, y = X.cuda(), y.cuda()` 和 `model = model.cuda()`）。
- **GPU 加速至关重要**：链接的图片建议确保你的 Notebook 运行时已配置为使用 GPU 加速器。


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1414042686679285941)** (6 messages): 

> `Device Shared 声明, PMMP 第 4 版值得吗？, PMMP 版本差异` 


- **Device 和 Shared 声明冲突？**：一位成员质疑在 **CUDA** 中 `__device__` 是否可以可选地声明在 `__shared__` 之前，并引用了一个编译错误，参考了 [这张截图](https://cdn.discordapp.com/attachments/1194427148656721970/1414042686457118750/2025-09-07_09.15.45.png?ex=68c01ba0&is=68beca20&hm=ac8613b3f022dccb4d9915038233b9e658be5ab0ea070c025e099302e4d9e2f6&)。
   - 另一位用户确认，使用 `nvcc` 编译 `__device__ __shared__` 不会报错，尽管会有一个“已声明但从未引用”的警告。
- **PMMP 第 4 版是必读的吗？**：一位成员询问 **Parallel Programming for Multicore and GPU Systems (PMMP)** 这本书的第四版是否值得一读。
   - 另一位成员简短地回答道：*“是的，这本书非常出色”*。
- **书籍 Diff，一个杀手级功能？**：一位成员询问对于读过 **PMMP** 第三版的人来说，是否有 Diff（差异对比）可用。
   - 他们表示希望书籍在不同版本之间能提供免费的 Diff。


  

---

### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1413682953925427242)** (7 条消息): 

> `面条、TV 输入延迟实验、冷门宝藏动漫、细粒度分类基准博客文章` 


- ****面条怀旧** 预测于 2026 年**：一位成员发布了一道由*牛脂烹制的酱油牛肉、绿豆、洋葱和甜红椒配小麦面*组成的菜肴，预测它将在 **2026** 年回归，并附上了一张 [照片](https://cdn.discordapp.com/attachments/1215328286503075953/1413739236590616576/IMG_20250906_060724.jpg?ex=68c05284&is=68bf0104&hm=bddb08e377319018e679002e25bbd9eaed43b5a3cd95455d28b90a2fc788ddd4&)。
- **通过粗略实验测试 **TV 延迟****：一位成员进行了一个粗略的小型实验，以寻找其新 TV **输入延迟（input latency）** 的上下限，而不是看电视或玩游戏，并分享了一个 [YouTube 剪辑和描述](https://www.youtube.com/watch?v=CyDAddqq9U4) 以及一张照片。
   - 该实验旨在评估 TV 在典型使用场景之外的响应速度。
- **动漫爱好者推荐**冷门宝藏****：成员们推荐了一些 **冷门宝藏动漫**，包括《Monster》、《天国大魔境》（Heavenly Delusions）和《跃动青春》（Skip and Loafer）。
- **成员分享**细粒度分类（Fine-Grained Classification）**优化**：一位成员分享了他们的第一篇关于*细粒度视觉分类优化训练配方*的博客文章，寻求关于清晰度、结构和语气的反馈；可以在 [这里](https://towardsdatascience.com/a-refined-training-recipe-for-fine-grained-visual-classification/) 找到。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1413615156877066303)** (7 条消息): 

> `纽约见面会、黑客松场地、注册确认` 


- **纽约见面会计划中！**：成员 [@apaz](https://discord.com/users/apaz) 提到他们在纽约，并愿意组织见面。
   - 另一位成员 [@_suiman](https://discord.com/users/_suiman) 做出回应，表示也有兴趣见面。
- **旧金山 SoMa 黑客松场地确认**：在有人提到正从中央车站走向场地后，一位成员询问了黑客松的具体地点。
   - 另一位成员提供了一个 [链接](https://events.accel.com/gpumodehackathon)，显示活动在 **SoMa SF**（[旧金山 SoMa 区](https://en.wikipedia.org/wiki/South_of_Market,_San_Francisco)），具体位置将在临近活动时公布。
- **Accel 活动注册确认延迟**：一位成员询问预计何时能收到旧金山 **Accel** 活动的注册确认。
   - 他们提到几周前就注册了，正在寻求关于确认时间线的信息。


  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1413790298026409984)** (20 条消息🔥): 

> `Triton 导入错误、Numpy 版本问题、Colab 会话重启` 


- ****RESERVED_KWS** 导入错误困扰 Triton 用户**：用户报告在运行 Triton 时遇到 `ImportError: cannot import name 'RESERVED_KWS' from 'triton.runtime.interpreter'`，一位用户之前通过聊天中提到的*修复方法*解决了该问题。
   - 建议的*修复方法*包括重新安装 **torch 2.5.0** 并使用 pip 将 **numpy** 限制在 *2.0 以下*版本：`!pip install --force-reinstall torch==2.5.0` 和 `!pip install "numpy<2.0"`。
- ****Numpy** 版本问题困扰 Colab Triton 设置**：用户遇到了 `ValueError: numpy.dtype size changed`，这表明由于 Google Colab 中的 **numpy** 版本冲突导致了二进制不兼容问题。
   - 解决方法是将 **numpy** 限制在 *2.0 以下*版本，但同时也提到可能需要在 Google Colab 中*重启会话（restart session）*才能使更改生效。
- **Colab 会话重启激活 Bug 修复**：在应用了建议的 **torch** 和 **numpy** 版本更改后，用户发现重启 Colab 会话对于正确应用修复至关重要。
   - 一位用户通过观察到重启后 *demo1 单元格*应以 **1** 的数组初始化来确认设置成功，而错误的设置会以 **0** 开始。


  

---

### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1413604672803045567)** (4 messages): 

> `mpi4py issues, iris, ROCm, pytorch` 


- **`mpi4py` 依赖被移除！**: 一位成员报告了在使用 **iris** 时遇到的 `mpi4py` 问题，并链接到了 [ROCm/iris 的一个 pull request](https://github.com/ROCm/iris/pull/154/files)，该 PR 移除了 `mpi4py` 依赖。
   - 该 PR 已被合并，作者请求大家对新设置提供反馈。
- **用户自愿测试 iris 更新**: 一位用户自愿测试移除 `mpi4py` 依赖后的 **iris** 更新。
   - **ROCm** 团队成员请求该用户提供关于设置的反馈。


  

---


### **GPU MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1413811245999460392)** (1 messages): 

> `RDNA3 MatMul, seb-v's talk` 


- **Seb-v 探讨 RDNA3 MatMul**: 一位成员请求关于 [seb-v 的 RDNA3 MatMul](https://seb-v.github.io/optimization/update/2025/01/20/Fast-GPU-Matrix-multiplication.html) 的演讲。
- **矩阵乘法深度解析**: 该演讲将涵盖 RDNA3 架构上矩阵乘法的优化技术。
   - 内容可能涉及内存访问模式、kernel 设计以及性能调优策略。


  

---


### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1413888995225632968)** (3 messages): 

> `Dawn Support, WGVK Compilation` 


- **Dawn 需要启用**: 据一位成员称，**Dawn** 需要被启用才能正常工作。
   - 他们没有说明如何启用它。
- **WGVK 需要编译标志**: 一位成员指出，如果使用 `-DSUPPORT_WAYLAND_SURFACE=1` 标志进行编译，[WGVK](https://github.com/manuel5975p/WGVK) 将可以工作。
   - 他们链接了[相关的代码行](https://github.com/manuel5975p/WGVK/blob/master/src/wgvk.c#L67)以供参考。


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1413679291417231481)** (2 messages): 

> `Metal Documentation, simdgroup matmul` 


- **Metal 文档匮乏**: 一位成员表示 **Metal** 的文档似乎写得非常糟糕。
- **matmul 实现与 simdgroup 类似**: 一位成员表达了他们的理解，即 **matmul 实现** 将与 **simdgroup** 的实现相同。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1413630928613408992)** (6 messages): 

> `Model Serving Newsletter, Outlier Experiments, RegicideOS Testers, CuTe Swizzling, Tiny Diffusion Models` 


- **Model Serving 社区通讯订阅量突破 300**: [Model Serving 社区现状：9 月版](https://inferenceops.substack.com/p/state-of-the-model-serving-communities-408) 通讯发布。报告称，在 Red Hat 内部分享一年多并最终公开分享后，订阅者已超过 **300 人**。
   - 该通讯旨在提供由社区驱动的关于 **vLLM**、**KServe**、**llm-d**、**Kubernetes** 和 **Llama Stack** 等项目的更新视图。
- **离群值实验文章发布**: 一篇关于离群值和实验的文章已发布，帖子详见[此处](https://emre570.bearblog.dev/outlier-experiment/)。
- **RegicideOS 寻找勇敢的测试者**: 正在为 [RegicideOS](https://github.com/awdemos/RegicideOS) 寻找测试人员。
- **CuTe Swizzling 详解**: 一篇博客文章发布，解释了 **CuTe Swizzling** 以及 **32B**、**64B** 和 **128B Patterns** 背后的数学原理，并对规范 Swizzles 的作用给出了易于理解的表达式。
   - 博客文章见[此处](https://veitner.bearblog.dev/understanding-cute-swizzling-the-math-behind-32b-64b-and-128b-patterns/)，LinkedIn 帖子见[此处](https://www.linkedin.com/posts/simon-veitner-174a681b6_understanding-cute-swizzling-the-math-behind-activity-7370506823671640064-7vZt?utm_source=share&utm_medium=member_desktop&rcm=ACoAADJYtOgBMOUI4WiWTyiAkroFBP1VujIWeksH)。
- **Tiny Diffusion Models GitHub 仓库创建**: 创建了一个 Karpathy 风格的图像扩散模型仓库，包括 **Conditional Flow Matching**、**DDPM** 和 **DDIM**。
   - 仓库见[此处](https://github.com/shenoynikhil/tinydiffusion)，很快将包含 **Consistency Models** 和 **Inductive Moment Matching**。


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1413742737026453555)** (1 messages): 

> `Custom OP Backend, DirectoryBackend Refactor, DSL Addition` 


- **自定义 OP 后端已准备就绪**: 一位成员宣布完成了 **自定义 OP 后端** 的第一个版本。
- **DirectoryBackend 进行重构**: 该成员重构了 **DirectoryBackend**，以便将来更容易地进行 **DSL 添加**。


  

---

### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1413719959262986353)** (84 messages🔥🔥): 

> `MI300x8 性能, amd-all2all 排行榜` 


- **MI300x8 排行榜冲刺**：提交至 `amd-all2all` 排行榜的活动在 **MI300x8** 上异常活跃，多位成员刷新了个人最佳纪录并争夺前排位置。
- **Sub-3ms MI300x8 占据主导地位**：多个提交在 **MI300x8** 上实现了低于 3ms 的性能，其中一个提交达到了 **2.33 ms** 并获得第 4 名。
- **精进 MI300x8 性能**：一位成员持续提交改进，耗时从最初的 **90ms** 左右优化到 **24-25ms** 范围，随后降至 **3-4ms**，最终获得了高达第 5 名的成绩。


  

---


### **GPU MODE ▷ #[ppc](https://discord.com/channels/1189498204333543425/1343373905783554048/)** (1 messages): 

verspasian: <#1198358627594023014>
  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1413842445778681946)** (13 messages🔥): 

> `FLE 仓库, 寻求帮助, Open World 场景错误` 


- **FLE 评估基础设施展示**：一位成员分享了 **FLE** 仓库在评估基础设施方面的进展，详见[此 Pull Request](https://github.com/JackHopkins/factorio-learning-environment/pull/330)。
   - 其中一次 Sweep 的结果作为示例添加到了 PR 的 `fle/eval/analysis/examples/analysis_results/test_sweep_small_20250904_151635_2e06c621` 文件夹中。
- **呼吁同伴提供通话协助**：一位成员提议通过通话来帮助面临设置问题的用户，承诺这样可以更快地解决问题。
   - 另一位成员提到下周晚间可以回答问题，但其他时间将离线，因为他们要去科罗拉多州的森林里。
- **Open World 场景生成错误**：一位成员报告在 `main` 分支上使用简单的 `fle eval` 运行 `open_world` 场景时出现分数相关的错误，Gym 配置为 `{ "env_id": "open_play", "model": "claude-3-5-sonnet-latest" }`。
   - 错误信息为 *('Could not get player score', 'attempt to call a nil value')*，该成员建议问题可能出在向后兼容性上，因为他们一直在使用 labplay。


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1413642207801638962)** (23 messages🔥): 

> `AMD 注册确认, Workflow 文件依赖, Triton 支持, HIP 模板, 组队系统` 


- **AMD 注册邮件确认成功**：收到 AMD 注册邮件的用户已**成功注册**参加比赛。
   - 具体来说，一位成员询问*收到邮件是否意味着注册成功*，另一位成员确认道：*是的，你已经准备就绪*。
- **调整 Workflow 文件需要 PR**：若要添加依赖项，用户可以提交 **PR** 来更新 [dockerfile](https://github.com/gpu-mode/discord-cluster-manager/blob/main/docker/amd-docker.Dockerfile)。
   - AMD Workflow 需要保持高效，因此不建议安装 C++ 库。
- **AMD 挑战赛强制要求 Dispatch 和 Combine Kernel**：参赛者必须在解决方案中实现 **dispatch** 和 **combine kernels**。
   - *任何试图跳过这两个步骤但恰好满足测试或基准测试的解决方案都不会被接受，届时将会进行检查*。
- **HIP 复用 CUDA 接口**：**PyTorch for HIP** 特意复用了现有的 **torch.cuda** 接口，以加速现有 PyTorch 代码和模型的迁移。
   - 一位用户报告了 **AttributeError: module 'torch' has no attribute 'hip'**，另一位用户回复称 *torch.hip 实际上就是 torch.cuda*。
- **澄清缺少 HIP 模板的问题**：本次比赛没有为 all2all Kernel 提供特定的 **HIP 模板**。
   - 根据置顶消息，参赛者可以将任何 HIP 代码与 PyTorch 的 `load_inline` 函数配合使用。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1414609302471049379)** (1 messages): 

> `SM90 生成器, 基于层级的实例化级别, CMake 标志` 


- **SM90 生成器使用基于层级的实例化**：**SM90** 的生成器采用了**基于层级的实例化级别 (levels-based instantiation level)**。
   - 有关如何使用这些 **CMake 标志**，请参阅 Profile 文档。
- **CMake 标志控制 SM90 生成**：Profile 文档解释了 **SM90** 生成器中 **CMake 标志** 的用法。
   - 这些标志用于配置**基于层级的实例化级别**。


  

---

### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1414546439966691369)** (8 messages🔥): 

> `pytorch backends, tinygrad's runtime, GPT2 Training` 


- **书籍作者正在编写关于 GPT2 训练的内容**：作者正在编写一本书，计划从自顶向下的视角介绍使用 **PyTorch** 进行 **GPT2 training**，旨在比 Karpathy 的逐步教学方法节奏更快。
   - 该书还将从自底向上的视角深入探讨 device runtime、tensor、cublas 和 cudnn。
- **深入探讨 PyTorch Backends 和 Tinygrad Runtimes**：作者正在重新审视 **PyTorch backends** 与 **Tinygrad** 之间的差异，并参考了过去与 Alban 合作的 **privateuse1 integration** 工作，参考 [此 Discord 链接](https://discord.com/channels/1068976834382925865/1342159212486197319/1347289867695951922)。
   - 他打算研究 torch.cuda 和 torch.mps 中的 **ATen** 和 **C10 abstractions**，并将其与 Tinygrad 的 runtime 进行比较。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1414176464932442152)** (6 messages): 

> `vectoradd leaderboard, kernel implementations, AMD GPU Mode competition` 


- **VectorAdd 胜利：排行榜潜水者寻求 Kernel 知识**：一位新成员询问是否可以查看 **vectoradd leaderboard** 上其他用户的提交内容，以学习 kernel 实现。
   - 一位成员澄清说，提交的内容只有在竞赛结束后才会开源，并指向了 Hugging Face 上的 [kernelbot-data 数据集](https://huggingface.co/datasets/GPUMODE/kernelbot-data)。
- **竞赛交流：明确竞赛公告频道**：一位成员询问了未来竞赛的公告频道，因为他错过了之前的 **AMD** 和 **Jane Street GPU Mode** 竞赛。
   - 另一位成员指引他们关注当前正在进行的第二届 **AMD competition**，并指出公告通常发布在 [此 Discord 频道](https://discord.com/channels/1189498204333543425/1189640399476764692/1410331124160397464) 中，注册截止日期为 **9 月 20 日**。


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1414282470198739064)** (14 messages🔥): 

> `MPI dtypes, Heterogeneous computing, NCCL GPU communication, NVSHMEM on GitHub` 


- **跨主机解码 MPI Dtypes**：讨论围绕当主机具有不同浮点精度（例如 **32-bit** vs. **64-bit**）时 **MPI dtypes** 如何运作展开，核心观点是数据以字节形式发送，然后转换为适当的 dtype，可能涉及截断或填充。
   - 一位成员发现 **Open MPI** 代码库使用了巧妙的远程架构和本地架构 dtypes 兼容性代码，在 [此处](https://github.com/open-mpi/ompi/blob/main/opal/datatype/opal_convertor.c) 提供了一个跨架构的 dtypes 转换器。
- **NCCL 处理 GPU 通信**：对话转向了 **NCCL** 如何处理 GPU 之间的通信，特别是当一个 GPU 支持 FP4 而另一个不支持时。
   - 一个建议是将数据打包为 **int8** 以避免此类兼容性问题，因为该库专注于描述正在通信的数据，而不是系统的原生支持。
- **NVSHMEM 已在 GitHub 上发布**：成员们分享了 **NVSHMEM** 现在已在 GitHub 上可用：[https://github.com/nvidia/nvshmem](https://github.com/nvidia/nvshmem)。
   - 对话没有就此展开更多细节。
- **MPI Send/Recv**：成员们讨论了 **MPI_Send** 和 **MPI_Recv** 函数如何使用 dtypes 和 counts 来确定交换的字节数。
   - 系统会执行检查以确定 MPI dtype 是否经过填充，并假设如果使用相同的源和目标类型，则系统是同构的。


  

---

### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1413668114469490760)** (12 messages🔥): 

> `Triton Kernel Launch Overhead, Model Quantization Survey, Torch Compile with BFloat16` 


- **Triton 的 Kernel 启动开销曝光**：成员们讨论了 **Triton kernel launch overhead** 高于 **CUDA/CUTLASS** 的问题，有人建议使用一种虽然 hacky 但更快的[直接调用 driver API](https://github.com/triton-lang/triton/issues/2637) 的方法，尽管 **CUDA graphs** 在很大程度上消除了这种需求。
   - 据推测，开销源于 **Python**，一名成员回忆称，通过直接调用 main run 函数可以减少开销，尽管这可能会带来参数和 `constexpr` 方面的复杂性。
- **量化论文搜索**：成员们寻求 **model quantization survey papers**（模型量化综述论文）的推荐，出现了两篇论文：*Full Stack Optimization of Transformer Inference: a Survey* ([arxiv.org/pdf/2302.14017](https://arxiv.org/pdf/2302.14017)) 和 *A Survey of Quantization Methods for Efficient Neural Network Inference* ([arxiv.org/pdf/2103.13630](https://arxiv.org/pdf/2103.13630))。
   - 一名成员指出，第二篇论文的合著者维护了一个 awesome list，可以通过 [github.com/Zhen-Dong/Awesome-Quantization-Papers](https://github.com/Zhen-Dong/Awesome-Quantization-Papers) 访问。
- **Torch Compile 用 FP32 处理 BF16？**：一名成员报告称，即使 **dtype** 设置为 **bf16**，`torch.compile` 仍然以 **fp32** 处理 tensor，导致累积误差。
   - 另一名成员澄清说，`torch.compile` 可能会为了精度以及绕过 **bfloat16** 算子支持的限制，而在 **float32** 中进行中间计算，由于内存受限（memory-bound）的 normalization 和 activation kernels，这通常不会带来性能损失。


  

---


### **GPU MODE ▷ #[jane-street-hackathon](https://discord.com/channels/1189498204333543425/1413690328464097422/1413690376023179506)** (65 messages🔥🔥): 

> `Triton hacking, GPU kernels with CUDA, VS Code SSH instance, Torch Compile Flags, Continuous Batching` 


- **Helion/Triton hacking 与 Nsys/Ncu 攻坚战开始**：一名在 **Helion/Triton hacking**、**Nsys/Ncu** 调优以及 **torch.compile** 方面有经验的成员正在寻找队伍，特别是希望加入熟悉 **CUDA/PTX** 的成员。
   - 另一名成员表示有兴趣在各种平台（**CUDA, Triton, CuTeDSL, PTX** 等）和 **GPU architecture** 上编写 **GPU kernels**，并对 ML 框架和编译器感兴趣。
- **Northflank 文档发布**：**Northflank** 联合创始人分享了用于访问计算环境的[文档](https://northflank.notion.site/Northflank-Docs-GPU-Mode-Hackathon-2496d14c785180c6a595e6776e3118ba)。
   - 他们分享了一个 [Google Forms 链接](https://docs.google.com/forms/d/e/1FAIpQLSeiM36crKYTeHVfR2V6k4WhNlzGtPZuOIHytOgrpYjvUaMb5w/viewform)，用于在队伍组建后提交团队信息以获取计算资源访问权限。
- **VS Code SSH 实例连接指南**：发布了将 **VS Code** 连接到 **SSH instance** 的说明，包括连接到基于浏览器的 **VS Code** 的链接。
   - 建议设置 git 仓库并在本地同步，以便在本地 IDE 中进行迭代。
- **Jupyter Lab 端口开放**：提供了在服务上开放 **8888 端口**以开启 **Jupyter Lab** 会话的说明，包括导航到 Networking 选项卡并选择 **HTTP** 作为协议。
   - 关于如何设置的指南已发布在 [Notion 文档](https://www.notion.so/northflank/Northflank-Docs-GPU-Mode-Hackathon-2496d14c7851800d9ef1d03c6888f18c)中。
- **Cheetah 团队获得第二名**：**Cheetah 团队**在两轮比赛中均获得第二名，共赢得 **$50k** 奖金。
   - 一名团队成员分享了在 [X](https://x.com/NadavTimor) 和 [LinkedIn](https://www.linkedin.com/in/nadav-timor) 上联系的链接。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1413624184763842682)** (320 条消息🔥🔥): 

> `Deepmind 与 Huawei B. Neural Network 进展，美国 AI 监管对开源模型的影响，Unsloth 修复 LoRA 训练，Huawei GPU，Hermes 4 模型` 


- **B. Neural Network 引发监控恐慌**：成员们正密切关注 [Deepmind 和 Huawei 在 **B. Neural Networks** 方面的进展](https://www.deepmind.com/blog)，特别是考虑到 Huawei 未来的室温量子系统可能会让美国政府在监控问题上感到*恐慌*。
   - 一位成员认为 **B. Neural Networks** 可能是训练具身智能 (Embodied AI) 的理想选择，因为 LLM/Transformer 方法可能过于*书呆子气*且耗电。
- **Anthropic 和解协议引发 AI 监管担忧**：[Anthropic 的 15 亿美元和解协议](https://www.theverge.com/anthropic/24180846/anthropic-authors-ai-settlement-copyright-infringement)引发了对潜在**美国 AI 监管**及其对开源模型负面影响的担忧。
   - 成员认为 **Anthropic** 达成和解是一种战略举措，因为他们负担得起，而规模较小的公司可能难以证明其训练数据是合法获取的。
- **Unsloth 的 LoRA 训练修复**：[Unsloth 修复了 LoRA 训练](https://huggingface.co/unsloth/Kimi-K2-Instruct-0905-GGUF)，为用户带来了 **Sonnet-4 @ Home**，同时也有人指出 *3l3.1 在通用用途和知识储备方面仍然是最好的*。
- **Huawei 的 GPU**：成员们讨论了 [Huawei 的新款 GPU](https://support.huawei.com/enterprise/en/doc/EDOC1100285916/181ae99a/specifications) ([Alibaba 链接](https://www.alibaba.com/product-detail/Brand-New-Stock-Huawei-Atlas-300I_1601427603681.html))，指出其拥有 **96GB** 显存，价格约为 2000 美元，但缺乏 CUDA 支持且速度慢于 3090。
   - 尽管性能不算强大，但这标志着一种趋势，即 **Nvidia** 可能需要应对毛利率下降或寻找新的业务领域。
- **Hermes 4 的创意写作能力令人印象深刻**：成员们发现 **Hermes 4 405b** 在创意写作方面有巨大提升，在许多领域甚至优于 **Claude 2.0**。
   - 一位成员表示，*如果一个模型在处理脏话时很有创意，我就知道它会是一个优秀的写作模型*，而 H4 达到了这些基准。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1413695174546292746)** (14 条消息🔥): 

> `Hermes 审查，去审查的困难，美国模型中的 HR 价值观，OpenAI 内部模型` 


- **Hermes 面临审查挑战**：成员们讨论了为什么 **Hermes** 会受到审查，有人建议微调 (finetuning) 可以减少拒绝回答的情况，对此一位模型创建者表示，*去审查很难*，因为**基础模型 (base models) 本身就被审查了**。
   - 该模型创建者声称：*在所有主流模型中，我们是受审查程度最低的。*
- **实现无审查 AI 的困难**：一位用户询问了用于创建 **NSFW 内容或炸弹配方**的数据集，以及是否存在避免此类内容的自上而下的决策，并建议与 **Gemma** 等模型进行对比。
   - 模型创建者确认他们已经尝试过了。
- **美国模型拥抱 HR 价值观**：一位成员表示，由于大公司的影响和法律责任担忧，所有**美国制造的模型**都趋向于 **HR 部门的价值观**。
   - 他声称 *美国模型是地球上受审查最严重的模型，遥遥领先*，并认为那些批评**中国审查模型**的人缺乏自我意识。
- **OpenAI 保留了更优越的内部模型**：有人认为 **OpenAI** 可能在内部保留了最好的模型，而只向公众发布较差的版本。
   - 一位成员表示：*这种假设是正确的，我 100% 确定 OpenAI 把好模型留给了自己，只向公众发布了一些劣质内容 (slop)。*


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1414294663132614656)** (2 条消息): 

> `BOS token 限制，从 EOS 开始微调，Crumb essence-3b-v1 模型` 


- **BOS Token 无法累积信息**：由于因果注意力掩码 (causal attention mask)，**BOS token** 不会累积信息，因为只有在其他 token 之后的 token 才能从中累积含义。
   - 唯一的潜在解决方案需要*从 EOS 开始微调*，尽管这需要微调整个模型而不仅仅是分类头 (classification head)。
- **Crumb Essence 采用 EOS 方案**：[Crumb 的 essence-3b-v1 模型](https://huggingface.co/crumb/essence-3b-v1)本质上做了同样的事情，但使用了多个 **EOS tokens** 而不仅仅是一个。
   - 描述该方法的论文可以在[这里](https://arxiv.org/pdf/2504.14191)找到。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 条消息): 

real.azure: 不错！

我已经觉得新的 Qwen3-4B 非常令人印象深刻了。
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1414294663132614656)** (2 messages): 

> `BOS Token 累积，EOS 微调，Crumb 的 Essence-3b-v1` 


- **BOS Token 已死，EOS Token 才是历史最佳 (GOAT)**：由于因果注意力掩码 (causal attention mask)，**BOS token** 不会累积任何内容，因此其他 token 之后的 token 可以从中累积含义。
   - 最好的选择是**从 EOS 进行微调**，但你必须微调整个模型，而不仅仅是一个分类头 (classification head)。
- **Crumb 的 Essence-3b-v1：EOS token 策略**：[Crumb 正在做几乎相同的事情](https://huggingface.co/crumb/essence-3b-v1)，但使用了多个 EOS token 而不是一个。
   - 发布了一个指向 [arxiv.org/pdf/2504.14191](https://arxiv.org/pdf/2504.14191) 的链接；这似乎是一篇相关的研究论文。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1413623180266049696)** (266 messages🔥🔥): 

> `AI 视频生成器免费计划策略，Perplexity Pro 计划优惠，GPT-5 对比 2.5 Pro，Grok 作为流氓 AI，使用 GPT Agent 设置收件箱过滤器` 


- **AI 视频生成器免费大放送**：一位用户概述了在 **Vidu**、**Pixverse.ai**、**Wan.video** 和 **Dreamina** 等平台上利用免费计划进行 AI 视频生成的策略，每天可实现多达 **15 次生成**。
   - 该策略依赖于从 Vidu（**每天 20 积分，每次生成 4 积分**）、Pixverse（**60 积分，每次生成 20 积分**）、Wan（**50 积分，无限慢速队列视频**）和 Dreamina（**120 积分，每次生成 50 积分**）进行生成。
- **Perplexity Pro 计划简直是白捡**：目前有一个与 **Paypal 和 Venmo** 合作的 Perplexity **1 年 Pro 计划**优惠。
   - 这似乎是一个真实的优惠，尽管一些成员报告称它*对现有的 Perplexity 用户无效*。
- **Grok，混乱邪恶的流氓**：用户正在将 Grok 与 ChatGPT 2.5 进行比较，有人说 **GPT-5** 很糟糕，而 **2.5 Pro** 非常好。
   - 普遍观点认为 Grok 是*团队中的流氓角色*。
- **AI Agent 搞砸了收件箱过滤器**：一位用户花费了 **50 次 ChatGPT Agent 查询**尝试设置收件箱过滤器，结果却造成了需要手动清理的混乱。
   - 他们注意到这些 ChatGPT “连接器”似乎都是只读的，不会更新或创建任何内容：*从插件时代起我们就有了读/写集成。但似乎为了获得基本的 UI 更新，我们放弃了 80% 的功能？*
- **“仅限消费”状态即将到来？**：ChatGPT 预测，大规模的人群将陷入一种“仅限消费”的状态，AGI 将能够针对你的个人需求进行极度适应。
   - 该用户认为生活在充满过滤器的个性化世界中将成为一种常态：*至少我是这么认为的，我不想让它听起来像反乌托邦，因为在我看来事实并非如此。*


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1413971580530327654)** (12 messages🔥): 

> `Gemini Pro 对比 GPT-5，GPT-5 频道已归档，GPT 退化，GPT 冻结` 


- **Gemini Pro 正面交锋 GPT-5**：一位用户询问 **Gemini Pro** 在写作方面是否优于 **GPT-5**，并预料到这之前已经被讨论过。
- **GPT-5 频道关闭**：一位用户询问为什么 **GPT-5** 频道被归档，以及该频道与 **ChatGPT** 频道之间有什么区别。
   - 另一位用户澄清说，这个频道是用于 **GPTs** 讨论的，而另一个频道则是关于 **ChatGPT** 的一切。
- **GPT 的退化困扰**：一位用户经历了他们认为是 **GPT-5** 的退化，正在寻求关于在哪里分享证据的指导。
- **GPT 在冗长讨论中冻结**：一位用户报告称，在长期的项目对话中，即使输入很短，**GPT** 也会在响应中途冻结，尽管尝试了各种故障排除步骤。
   - 他们注意到，新聊天运行良好，直到对话变得*太长*，这种情况每天都会发生，其他人则询问了对话的字符或 token 长度。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1413877501134114836)** (19 条消息🔥): 

> `GPT-5 rollout, Model instruction following, Web search API tips, Automotive logo design, SVG logos` 


- **GPT-5 Rollout 困扰**：一位成员对 **GPT-5 rollout** 表示沮丧，声称它*在被提醒之前会忽略所有规则*，自那以后生活变得*没那么有趣了*。
   - 另一位成员表示赞同，指出该模型*甚至不记得过去的对话*，并且*给出最无效率的解决方案*，耗时比 **GPT-4o** 更长。
- **模型变更后编写清晰的指令**：一位成员分享了[他们处理模型变更的方法链接](https://chatgpt.com/share/68bc6ff6-45a8-8011-9637-9745271001f2)，建议用户*清晰地说明“这一次”我想要什么*。
   - 他们强调要确保指令之间不冲突，并利用模型通过询问“*接下来的指令对你来说意味着什么？*”来检查歧义。
- **应对 Web Search API 的陷阱**：一位成员寻求使用 **got-5-mini** 为 **LinkedIn** 职位引导 Web Search API 的技巧，苦于模型返回的职位已关闭申请。
   - 另一位成员建议将 **web_search** 作为工具启用，使用单独的工具解析 URL 以提取数据，然后将这些结果传递给 **GPT** 进行分析。
- **SVG Logo 的优势**：一位正在设计汽车品牌 Logo 的成员寻求帮助，另一位建议使用 **SVG** 格式，因为它们具有无损缩放性。
   - 建议提到 **ChatGPT** 可以通过 canmore (Canvas) 工具创建一个单一的 **HTML+SVG** 文档，以便于预览。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1413877501134114836)** (19 条消息🔥): 

> `GPT-4o Performance, Steering API for web search, Model Changes, Logo Design with AI, SVG logos` 


- **GPT-4o 速度下降？**：用户注意到，与最初发布时相比，**GPT-4o** 不记得过去的聊天内容，给出的解决方案效率低下，且每个 Prompt 的耗时更长。
   - 一位用户对模型似乎在被提醒之前忽略所有规则感到沮丧。
- **控制 API Web Search 的技巧**：一位用户在利用 **got-5-mini** 搜索 LinkedIn 职位时，寻求引导 API 进行 Web Search 的建议，因为尽管有明确指令，模型仍会返回已关闭的职位申请。
   - 建议包括确保将 **web_search** 启用为工具，解析 URL 以提取数据，并尽量减少 API Key 上的输入 Token。
- **应对持续的模型变更**：一位用户分享了自 **2022 年 12 月 15 日**以来适应频繁模型变更的经验，强调需要清晰地陈述意图并解决潜在的指令冲突。
   - 该用户建议询问模型指令对其意味着什么，是否与现有指令冲突，以及歧义所在。
- **AI 精心设计的 Logo？**：一位成员请求协助为其汽车品牌创建 Logo，寻求使用 **DALL-E 等 AI 工具**或其他方法的技巧和指导。
   - 另一位成员建议 Logo 使用 **SVG** 格式，因为它们可以无损缩放，并指出 **ChatGPT** 可以通过 `canmore` 工具创建单一的 HTML+SVG 文档以便轻松预览。


  

---


### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1414662019780706367)** (1 条消息): 

> `HF Hub Milestones, Trackio Features, Claude Image Generation, CUDA Kernel Guide, ZeroGPU Speed Improvements` 


- **HF Hub 达到 200 万个公开仓库**：Hugging Face Hub 已达到 **200 万个公开仓库**，标志着开源社区的一个重要里程碑 [来源](https://x.com/reach_vb/status/1960642240583266473)。
- **Trackio 新增免费图像和表格日志记录**：**Trackio** 现在支持记录图像和表格，完全免费，增强了其在跟踪和可视化实验数据方面的实用性 [来源](https://x.com/abidlabs/status/1958910118214397999)。
- **使用 Claude 和 HF 生成图像**：一篇新的博客文章详细介绍了如何使用 **Claude** 和 **Hugging Face** 生成图像，为有兴趣利用这些工具进行图像生成的用户提供了指南 [来源](https://huggingface.co/blog/claude-and-mcp)。
- **Apple 在 HF 上发布 FastVLM 和 MobileCLIP2**：**Apple** 在 Hugging Face 上发布了 **FastVLM** 和 **MobileCLIP2**，扩大了这些模型在开源社区的可用性 [来源](https://x.com/xenovacom/status/1961454543503344036)。
- **NVIDIA Nemotron Nano V2 发布**：**NVIDIA** 发布了 **Nemotron Nano V2**，被推广为适用于各种 AI 应用的小型但强大的模型 [来源](https://x.com/ClementDelangue/status/1957519608992407848)。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1413656732739309609)** (254 条消息🔥🔥): 

> `GPU 脚本问题、Medgemma 推理、abliterated 模型微调、Cohere 研究学者计划、视觉相似度评分` 


- **调试 CPU/GPU 问题和 Medgemma 减速**：一位用户报告了脚本默认使用 **CPU 而非 GPU** 的问题，另一位用户发现即使使用推理端点，**Medgemma 推理**也非常缓慢。
   - 建议目前先使用推理端点，同时等待推理提供商的更新，并检查脚本设置。
- **Abliterated 与普通模型**：一位用户询问是微调 **abliterated 模型**还是**普通模型**，**ChatGPT** 建议选择后者以保持对行为的控制。
   - 另一位成员将 *abliterated 模型* 定义为 *uncensored*（无审查）模型，并指向了[这篇博客文章](https://huggingface.co/blog/mlabonne/abliteration?utm_source=chatgpt.com)。
- **讨论 RAG 应用和未来的 GPU**：用户们回忆了早期使用 **RAG** 的经历，其中一位对性能更强、显存容量可观的 **Nvidia 50 系列 GPU** 表示期待，以便实验开源模型。
   - 普遍共识是：*GPU 必须成为 AI 工程师的 PS5 等价物*。
- **EPC 文档的视觉相似度评分**：一位成员正在寻求视觉相似度评分方法，以便仅根据视觉分析（不进行文本提取或 OCR）将 **EPC（能源效率证书）文档**与非 EPC 文档区分开来，并提供了[此文件](https://cdn.discordapp.com/attachments/879548962464493622/1414492098387906560/visual_similarity_epc.md?ex=68c06cac&is=68bf1b2c&hm=d446c77726e2f7c84493ca0e82b21aa84c98c1c4f1510e15804d1640f0bdc2ae&)。
- **处理 Python 依赖并推荐 Anaconda 的替代方案**：一位用户对 **Anaconda 的缓慢**表示沮丧，另一位用户推荐使用 **uv** 作为替代包管理器，并引用了[此文档页面](https://docs.astral.sh/uv/getting-started/)。
   - 然而，另一位用户表示不喜欢 uv，因为 *Python 依赖管理简直糟透了*。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1413942977805422744)** (3 条消息): 

> `因果关系手册 (Causality Handbook)、机器人学 SOTA、SmolVLA` 


- **因果关系手册新手需要资源**：一位成员正在开始学习 [Causality Handbook](https://matheusfacure.github.io/python-causality-handbook/)，并寻求额外的资源来估算特征的影响，计划在周一进行一些分析。
- **机器人学 SOTA 综述发布**：一份机器人学调查报告对用于机器人操作的大型 **基于 VLM 的 VLA 模型** 进行了系统的、面向分类的综述 ([arxiv 链接](https://arxiv.org/abs/2508.13073))。
   - 它定义了大型 **基于 VLM 的 VLA 模型**，并描绘了两种主要的架构范式：**单体模型 (monolithic models)** 和 **分层模型 (hierarchical models)**。
- **SmolVLA 太棒了**：一位成员分享了一张图片并表示 **SmolVLA** 非常出色 ([图片链接](https://cdn.discordapp.com/attachments/898619964095860757/1414262625012945078/image.png?ex=68c03fb5&is=68beee35&hm=859065fe374972d8897e75248eb8524906c4b7b86bc79e7c6f0379b82cafd684))。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1413860333889650819)** (1 条消息): 

> `软件开发路线图、开发者路线图、github.com` 


- **Kamran Ahmed 的路线图规划了开发路径**：一位成员分享了 [kamranahmedse/developer-roadmap](https://github.com/kamranahmedse/developer-roadmap?tab=readme-ov-file) 的链接，该资源涵盖了软件开发领域的**大量**路线图。
   - 这些路线图包含 **100 多个目标**，并深入探讨了每个主题。
- **Ahmed 的资源是一个宝库**：该资源为 Frontend、Backend、DevOps 等各种角色提供了全面的路线图。
   - 每个路线图旨在引导开发者掌握必要的技能和技术，使应对现代软件开发的复杂性变得更加容易。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1413683468663263292)** (15 messages🔥): 

> `DINOv3 用于卫星图像, Pathlint 用于代码清理, 从零开始构建 Gemma3, BwETAFv3 CLMs, 医学推理 GPT` 


- **DINOv3 在卫星图像领域表现出色**：一名成员为使用卫星图像的 **DINOv3** 创建了一个 zeroGPU 推理演示，通过适配代码来测试 **sat-493m** 预训练模型，并支持用户选择图像：[Hugging Face Space](https://huggingface.co/spaces/pszemraj/dinov3-viz-sat493m)。
   - 他们提到该演示 *"运行较慢，但下拉菜单中的完整 7b 卫星模型可以正常工作。"*
- **Pathlint 优化 Python 路径**：一名成员发布了 **Pathlint**，这是一个用于在 Python 代码中强制使用 `pathlib` 代替 `os.path` 的 linter，可在 [GitHub](https://github.com/pszemraj/pathlint) 上获取。
   - 该 linter 旨在通过“闭环”方法提高代码整洁度和多平台兼容性，可通过 `pip install git+https://github.com/pszemraj/pathlint.git` 安装。
- **从零开始萌发 Gemma3**：一名成员使用 PyTorch 和 TinyStories 数据集从零开始构建了 **Gemma3 270M**，在 **A6000 GPU** 上训练了 **10 小时**，代码和模型权重分别发布在 [GitHub](https://github.com/di37/gemma3-270M-tinystories-pytorch) 和 [Hugging Face](https://huggingface.co/disham993/gemma3-270m-tiny-stories)。
   - 训练过程的图表使用 Weights & Biases 记录，并由 Claude Opus 4.1 进行评估。
- **BwETAFv3 模型发布并带来升级**：一名成员使用 JAX/Flax 从零开始预训练了两个新的 CLM，命名为 **BwETAFv3-97M** 和 **BwETAFv3-33M**，具有 GQA、自定义 tokenizers 和 KV caching 功能，可在 [Hugging Face](https://huggingface.co/WICKED4950/BwETAFv3-97M) 和 [Hugging Face](https://huggingface.co/WICKED4950/BwETAFv3-33M) 获取。
   - **97M** 模型的性能接近 GPT-2 和 OPT-125M；评估代码可通过 pip 获取，训练详情见附带的 [PDF](https://cdn.discordapp.com/attachments/897390720388825149/1414213260416385215/BwETAFv3.pdf?ex=68c0ba7c&is=68bf68fc&hm=516f2881c25e8a03eb8662c2e1d67d61b85584fa34aa5727c13b16d7bee3f3b6&)。
- **医学推理 GPT 模型发布**：一名成员在热门的医学推理数据集上微调了 OpenAI 的 OSS **20B** 推理模型，在增强其医学背景表现的同时，保留了其 Chain-of-Thought 推理能力，可在 [Hugging Face](https://huggingface.co/dousery/medical-reasoning-gpt-oss-20b) 获取。
   - 该模型使用 **4-bit 优化**，训练格式包含 *“question”、“Complex_CoT” 和 “Response”* 字段，用于拆解医学案例并回答执业医师资格考试风格的问题。


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1413770031233237113)** (2 messages): 

> `LLM 幻觉, OpenAI 论文, LLM 置信度滑块, 数据集推荐` 


- **OpenAI 论文针对 LLM 幻觉问题**：一名成员分享了一个 [Twitter 线程](https://x.com/LuozhuZhang/status/1964209351960514778)，总结了 **OpenAI 解决 LLM 幻觉的新论文**。
   - 该线程强调，可以通过改变激励机制来减少幻觉，例如相比于拒绝回答，对自信的错误给予更高惩罚，并奖励校准后的不确定性；该成员询问是否有推荐的数据集来对此进行测试。
- **置信度滑块增强 LLM 控制**：一名成员建议为 LLM 添加 **置信度滑块** 来管理回复。
   - 在最低置信度下，除非找到直接来源，否则 LLM 会说 *"不知道（idk）"*；而在最高置信度下，它可以根据自己的理解自由发挥，这可能会诱发幻觉。


  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1414501587077959751)** (1 messages): 

> `H100, Hopper 系列 GPU, Flash Attention 3, Diffusers, ZeroGPU` 


- **Flash Attention 3 加速 Hopper GPU**：使用 **H100** 或 **Hopper 系列 GPU** 的用户应尝试 **Flash Attention 3** 以显著提升速度，详见 [此 diffusers pull request](https://github.com/huggingface/diffusers/pull/12236)。
- **ZeroGPU 演示受益于 Diffusers 优化**：使用 **Diffusers** 构建由 **ZeroGPU** 驱动的演示的用户，也应考虑使用 Flash Attention 3 优化。


  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1413719057525506088)** (3 messages): 

> `Dynamic Autoencoders, Image Padding, GANs Stability, Traditional Mud Emulation` 


- **关于动态 Autoencoders 灵活性的辩论**：一位成员询问了卷积 Autoencoders 的动态特性，指出由于下采样（down-sampling）的原因，它们通常要求图像长度必须是特定数字的倍数。
   - 另一位成员建议可以对图像进行 Padding 以满足这些要求，并在处理后进行 Unpadding，从而使过程更加灵活。
- **GANs 展示了令人印象深刻的稳定性**：一位成员评论说，在使用 GAN 时，这种方法*看起来足够好*，而且*实际上非常稳定且干净*。
   - 这突显了 GANs 在生成高质量且一致结果方面的潜力。
- **探索传统的泥浆模拟（mud emulation）方法**：一位成员建议探索传统的泥浆模拟方法，例如使用预定义的泥浆形状并应用滤镜来调整亮度和对比度。
   - 这种方法为更复杂的方法提供了替代方案，并且在某些应用场景中可能非常有效。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1413855047267385426)** (3 messages): 

> `AI Training Costs, AI Agents in Production, Data Anonymization, Datatune` 


- **削减 99.5% 的训练成本！**：一位成员声称使用一种技术将 **AI 训练成本** 降低了 **99.5%**，并在 [LinkedIn 上分享了凭证](https://www.linkedin.com/posts/utkarsh284_%F0%9D%97%9C-%F0%9D%97%B7%F0%9D%98%82%F0%9D%98%80%F0%9D%98%81-%F0%9D%97%B0%F0%9D%98%82%F0%9D%98%81-%F0%9D%97%94%F0%9D%97%9C-%F0%9D%98%81%F0%9D%97%BF%F0%9D%97%AE%F0%9D%97%B6%F0%9D%97%BB%F0%9D%97%B6%F0%9D%97%BB%F0%9D%97%B4-activity-7368510579847675905-jlYD)。
- **AI Agents 在生产环境中充满挑战：新见解！**：一位成员发表了一篇 [Medium 文章](https://medium.com/@raj_shinigami/why-ai-agents-are-difficult-to-implement-in-production-ebc861b57694)，概述了在构建生产级 **AI agents** 时面临的困难。
   - 文章详细阐述了**逐行上下文理解（per-row context understanding）**对于数据转换的重要性，特别是在需要识别细微细节（如用于数据匿名化的女性姓名）的任务中。
- **使用 Datatune 进行数据匿名化！**：一位成员介绍了 **Datatune** 作为数据匿名化的解决方案，强调了它在需要识别姓名等细节的任务中理解上下文的能力。
   - [Jupyter Notebook](https://github.com/vitalops/datatune/blob/main/examples/data_anonymization.ipynb) 中提供了一个详细示例，展示了 **Datatune** 的功能，并附带了 Notebook 中的图像。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1413614618261327912)** (5 messages): 

> `smol-course materials, AI/ML roadmap, smol course registration` 


- **确定 Smol-Course 初始分支**：一位成员确认 [rerelease-chapter-1 分支](https://github.com/huggingface/smol-courseBranch) 是 **smol-course** 的正确起点。
   - 他们引导新成员前往该分支开始他们的 AI 之旅。
- **新手寻求 AI/ML 路线图**：一位没有编程经验的新成员请求一份入门友好的 **AI/ML** 路线图和资源。
   - 他们还询问了在深入研究 **AI** 之前，为了打下坚实基础需要重点关注的核心主题。
- **关于 Smol Course 注册的疑问**：一位成员询问如何注册 **smol course**，并咨询是否有讲师指导的课程，还是完全自学的模式。
   - 该成员不确定该从课程的哪里开始。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1413696812094586962)** (8 messages🔥): 

> `Introductions, Real World AI Agent project` 


- **课程自我介绍环节开始**：几位来自中国、印度和塞尔维亚的新成员介绍了自己，并表达了开始 Agents 课程的兴奋之情。
   - 其中一位来自印度的成员 Rishab 表示，他们参加课程是为了将 **Agents 集成到正在构建的应用中**，并渴望进行社交。
- **发布基于项目的学习机会**：一位成员提议使用 Substack 上分享的真实世界 AI Agent 项目进行协作学习，并提供了 [GitHub 仓库](https://github.com/neural-maze/philoagents-course)链接。
   - 他们邀请其他对基于项目学习感兴趣的人一起学习 **philoagents-course**。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1413600269773574265)** (126 条消息🔥🔥): 

> `AI-induced psychosis, Semantic drift, LLMs sycophancy, Logical Reasoning in LLMs, AI content in Google Search` 


- **AI 诱发的精神错乱声称对荒谬话题进行研究**：成员们讨论了近期激增的一类人，他们声称针对 *recursive symbolism*（递归象征主义）等荒谬话题撰写了研究论文。
   - 有人指出，这些论文通常使用由技术感词汇组成的“形容词+名词”短语，这为潜在的 **AI-induced psychosis**（AI 诱发的精神错乱）敲响了警钟。
- **语义漂移讨论**：**Semantic drift**（语义漂移）指词汇含义随时间发生的变化。成员认为这与 ML 开发相关，因为 tokens 的含义可能取决于文档编写的时间和地点。
   - 一位成员分享了该现象的一个[案例及详尽讨论](https://discord.com/channels/729741769192767510/1380277635636264980)。
- **LLMs 谄媚讨论及其与虐待性沟通的潜在联系**：有观点认为，LLMs 使用的语言与施虐者用来扭曲受害者现实的 **gaslighting techniques**（煤气灯效应手段）惊人地相似。
   - 一位用户提出，施虐者和 LLMs 都在为自己的目标进行优化，而不关心 ground truth 或对方的福祉，这导致了行为的趋同。
- **成员讨论 LLMs 内部的逻辑单元**：成员们讨论了直接在 LLM 的层级中添加专用的 **logical processing units**（逻辑处理单元）来处理基础逻辑运算。
   - 核心见解是，将这些单元作为模型内部的基础构建块，可以帮助模型更好地理解和生成语言中自然存在的逻辑流。
- **随着 AI 生成内容的增加，Google 搜索结果恶化**：成员们注意到 Google 搜索结果越来越多地指向糟糕的 **AI-generated content**，这些内容只提供泛泛的信息，而非回答具体问题。
   - 针对这一现状，一位成员表示：“我基本上不再使用 Google 了。”


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1413685956795437077)** (62 条消息🔥🔥): 

> `Information Theory and Power Laws for Language Models and Stochastic Processes Paper Criticism, Compressing KV cache, Redundant functional motifs in neural networks, New 3T dataset from PDFs, Eval framework from Aleph Alpha` 


- **信息论论文受到质疑**：一篇将信息论与语言模型联系起来的[论文](https://www.researchgate.net/publication/379443831_Information_Theory_and_Power_Laws_for_Language_Models_and_Stochastic_Processes)因缺乏根据和清晰度而遭到批评，一位成员称其为 *Time Cube 级别的论文*。
   - 评审者建议定义通信模型、确立公理并为声称的定理提供证明，其中一人将其斥为“包裹在 LaTeX 里的隐喻集合”。
- **蒸馏 LLM Agent 论文浮出水面**：一位成员寻找关于使用 SmolAgents 轨迹进行训练的论文，并找到了《Distilling LLM Agent into Small Models with Retrieval and Code Tools》[论文](https://arxiv.org/abs/2405.05254)。
   - 该论文探讨了如何利用检索和代码工具将 LLM agents 蒸馏到更小的模型中。
- **探索 Function Call 方法**：一位成员建议采用一种理想化的神经网络设计，使用 **function call** 方法来共享先前学到的概念，而不是在每一层重复学习冗余版本。
   - 他们发布了一个关于空间和时间并行神经网络的 [YouTube 视频](https://youtu.be/H1wZD6BhstU)链接，并将其类比为“你大脑中的每个子系统对‘你奶奶是谁’都有各自的理解”。
- **FinePDFs 数据集发布**：一个研究团队发布了一个名为 [FinePDFs](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) 的新 **3T 数据集**，该数据集从 PDF 中提取，代表了一种相对较新的数据源。
   - 该数据集在 [X 上发布](https://x.com/HKydlicek/status/1964584936524124645)，强调了其在各种语言建模任务中的潜在用途。
- **剪枝重复的神经电路？**：成员们讨论了一种理想化的神经网络设计，采用 **function call** 方式，剪掉为网络不同部分提供相同功能的重复电路，从而促进信息共享和组合性。
   - 目标是让所有重复电路使用同一个副本，但也有人认为“重复电路可能正在为网络的不同部分提供相同的功能”，移除它们可能会面临重新学习这些电路所支持内容的 compute 浪费风险。


  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1414740117876113579)** (1 messages): 

> `Calibration Scores, LM Eval Harness` 


- **为 LM Eval Harness 进行校准？**：成员们表示有兴趣在 **LM eval harness** 中为 **trustworthy models** 添加 **calibration scores**，并链接到了关于 [RL for calibration](https://arxiv.org/pdf/2507.16806) 的工作。
   - 提到了之前的一个 [pull request](https://github.com/EleutherAI/lm-evaluation-harness/pull/874)，以及在 [X](https://x.com/_jasonwei/status/1871285864690815053) 上的一段批判性见解。
- **通过 Eval Harness 构建可信模型**：添加 **calibration scores** 的主要目标是引导激励机制，利用 **LM eval harness** 创造更具 **trustworthy models**。
   - 这一增加旨在提供一种更广泛的方法，以确保模型在评估中的可靠性和可信度。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1413873824944095233)** (26 messages🔥): 

> `QK Norm, RoPE vs NoPE, Gradient magnitudes, Pythia head size` 


- **QK Norm 前来救场！**：添加 **QK Norm** 解决了一个训练不稳定性问题，这可能与 [OLMo 2 Technical Report](https://link.to/olmo2report) 中提到的巨大且尖锐的梯度（spiky gradients）有关。
   - 一位成员使用 **QK Norm** 启动了一次运行，并确认它解决了问题，表达了感谢。
- **RoPE 还是 NoPE？**：成员们讨论了交替使用 **RoPE** (Rotary Positional Embedding) 和 **NoPE** (No Positional Embedding) 层的稳定性。
   - 有假设认为，即使是少量的 **RoPE**（例如每层 1%）也可能提供足够的稳定性。
- **记录关于 QK Norm 稳定性的发现**：一位成员将上述讨论中的发现整理成了一篇关于训练期间 **QK Norm** 稳定性的博客文章，可在此处查看 [here](https://aflah02.substack.com/p/on-stability-with-qk-norm-during)。
   - 该博文讨论了在没有 **QK Norm** 的情况下使用 **NoPE** 如何引发麻烦，特别是对于 **Pythia** 风格的架构，以及 **QK Norm** 如何帮助减少巨大且尖锐的梯度。
- **Pythia 的 Head Size 揭晓！**：一位成员纠正了另一位成员，澄清 **Pythia** 的 **head size** 是 **256**，而不是 128。
   - 这一修正与关于使用小比例 **RoPE** 的讨论相关，因为 128 的 1% 将少于 **RoPE** 的单个通道对。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1413601325039616061)** (124 messages🔥🔥): 

> `Dot App Shutdown, Hashbrown v0.3 Release, Anthropic Copyright Settlement, Codex Team Podcast, AI Evals Debate` 


- **Dot App 熄灯**：**New Computer** 正在停止其个人日记应用 **Dot**，这引发了用户的感激之情，同时也夹杂着对信任的担忧，如[这条推文](https://x.com/newcomputer/status/1964032611224871383)所述。
- **Hashbrown 发布 v0.3 并支持 Ollama**：**Mike Ryan** 宣布了 **Hashbrown v0.3**，这是一个生成式 UI 库，具有 **Ollama** 适配器、有限的 **MCP Server** 支持、新的 prompt template literal 以及重新设计的文档，详见[这条推文](https://xcancel.com/mikeryandev/status/1964029867437281658?s=46)。
- **Anthropic 为书籍向作者支付数十亿美元**：**Anthropic** 同意与图书作者就版权问题达成一项具有里程碑意义的 **15 亿美元** 和解协议，这可能为 AI 公司补偿权利持有人设定先例，详见[这篇纽约时报文章](https://www.nytimes.com/2025/09/05/technology/anthropic-settlement-copyright-ai.html)。
- **OpenAI 面临巨额资金消耗**：更新的财务预测显示，受算力和数据中心成本驱动，**OpenAI** 到 2029 年将消耗惊人的 **1150 亿美元**，盈利时间推迟到 2030 年，详见[这条推文](https://xcancel.com/srimuppidi/status/1964145060196286850?s=46)。
- **Vercel 酝酿 AI 调优浏览器**：**Vercel** 悄悄发布了 **dev3000**，这是一个为 **AI agents** 优化的 Chrome 变体，可通过 **MCP server** 流式传输日志、截图和网络事件，正如 **Malte Ubl** 在[这条推文](https://xcancel.com/cramforce/status/1964378896545223150?s=46)中所宣布的那样。


  

---

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1413601772748017765)** (12 messages🔥): 

> `AI Engineer CODE Summit 2025, FAL AI 估值, Latent Space Podcast` 


- **AI Engineer CODE Summit 2025 正式发布！**: **AI Engineer 团队**宣布将于今年秋季在纽约举办首届专门的 **CODE 峰会**，届时将聚集 **500 多名 AI Engineers 和领导者**，以及顶尖模型构建者和财富 500 强用户。CFP（征稿）开放至 **9 月 15 日** - [链接](https://xcancel.com/swyx/status/1964021608198324587?s=46)。
- **FAL AI 估值达 15 亿美元**: Latent Space 播客发布了一集新节目，邀请了 **FAL AI** 的联合创始人讲述他们如何从 feature-store 转型为生成式媒体基础设施的领导者，以及他们以 **15 亿美元估值**完成的 **1.25 亿美元 C 轮融资** - [链接](https://xcancel.com/latentspacepod/status/1964084193690055067)。
- **Latent Space Podcast**: 新一期 Latent Space 播客节目采访了 Fal AI，讨论了他们**从 feature-store 到生成式媒体基础设施领导者的转型**，以及他们以 **15 亿美元估值**完成的 **1.25 亿美元 C 轮融资** - [播客链接](https://x.com/latentspacepod/status/1964084193690055067)。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1413851497401286758)** (29 messages🔥): 

> `图像模型对比, Nano Banana 模型, Veo 3 定价, 混合 AI 动画, Banana Straightener` 


- **Nano Banana 图像模型完胜竞争对手**: 对比显示 **Nano Banana** 的表现优于其他图像模型，详见[此 YouTube 视频](https://youtu.be/9Co_M27CEEE?si=uqjc3cvIGwShaHX2)和[基准测试对比](https://genai-showdown.specr.net/)。
- **Google 降低 Veo 3 价格**: Google 将 **Veo 3** 的价格削减了 50% 以上，并使 **Veo 3 Fast** 在 AI Ultra 计划中可以无限制使用，这标志着其在生成式视频领域的激进扩张，讨论见[此处](https://x.com/JerryLiJiaming/status/1964470954610082284)。
- **Spellbrush 创作 48 小时动漫**: **Spellbrush AI** 创作了一部 48 小时的动漫，引发了人们对 AI 驱动内容创作的兴趣，更多细节见[此处](https://x.com/venturetwins/status/1964860673151897977?s=46&t=Z6mP_1pHALnIw7k1lFkdwQ)。
- **混合 AI 动画的技巧与好评**: 创作者分享了**混合 AI 动画**的技巧和评价，讨论了由于训练数据稀缺，在融合动画和现实感方面面临的挑战，并提到了分层工作流（**Midjourney → ControlNet → Runway**），见[此处](https://x.com/skirano/status/1964771048966197368)。
- **Banana Straightener 自动迭代 Nano Banana 图像**: Radek Sienkiewicz 发布了 “**Banana Straightener**”，这是一个开源工具，使用 Google Gemini 2.5 Flash（又名 “**Nano Banana**”）自动重新生成图像，直到符合用户的描述，可以通过 `pip install banana-straightener` 安装，详情见[此处](https://x.com/velvet_shark/status/1963966803417133185)。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1413628186947747882)** (127 messages🔥🔥): 

> `Kimi K2 研究用途, Kimi K1.5 对比 Kimi K2, 美国 AI 对比中国 AI, Perplexity 用户群, EQ Bench 准确率` 


- **Groq 驱动的替代聊天机器人**: 一个使用 **Groq** 的替代聊天机器人已上线，提供**全工具调用（tool-calling）**功能，速度约为 **~200tk/s**，由一家非营利组织资助，无需 API key 且无速率限制。
   - 该替代方案免费且无限制，但不支持图片上传。
- **Kimi K2 研究员模式试用**: 一位用户对 **Kimi** 的研究员模式印象深刻，但指出在用完初始的 3 次研究次数后，很难找到关于配额重置的信息。
   - 虽然 **Kimi** 最初暗示是 **24 小时重置**，但在来源无法核实时撤回了这一说法。
- **Kimi 不止进行一次搜索**: **Kimi** 可以在同一个查询中进行 **5 次**额外尝试，并在必要时再进行 **5 次**搜索，这反驳了一位用户的假设。
   - 一位用户提供了一个不可能完成的任务作为例子，并在此处发布了图片作为证明 [链接](https://cdn.discordapp.com/attachments/1371757564005711973/1413785317714165841/image.png?ex=68c07d6e&is=68bf2bee&hm=cdf598702ceb07b66277aed0e2512e68b433ddadc85cc19b97f25d754c9bbac4&)。
- **Kimi K1.5 相比 K2 仍保留一些优势**: **Kimi K1.5** 在某些方面仍优于 **Kimi K2**，例如在不压缩内容的情况下改写文本，以及在幻觉处理方面可能更出色。
   - 用户对 **Kimi K2 0905** 与之前版本的差异感到好奇，特别是关于代码改进和 Agent 能力方面。
- **Kimi 研究员利用数百个来源**: **Kimi** 研究员通常会查阅数百个来源。
   - 一位用户在 **5 次**搜索尝试中看到了 **Kimi** 提供的 **70-80 个来源**，总计达到 **280 个**来源。


  

---

### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1414319172367683584)** (5 messages): 

> `JTBD Validator, DSPy for Business Validation, Multi-Agent Systems with DSPy and GEPA, DSPy Weekly Newsletter, AI Agents Play Taboo` 


- ****使用 DSPy 的 Jobs-to-Be-Done (JTBD) 验证器发布****：社区成员使用 DSPy 构建了一个 **JTBD 验证器**，旨在拆解业务想法、识别假设与风险并设计实验，并在 [GitHub 上分享了代码](https://github.com/jmanhype/jtbd-idea-validator-agent) 以及一份 [示例报告](https://gamma.app/docs/JTBD-Validation-AI-Powered-Rehabilitation-Exercise-Tracking-x9ldjcxibgsserl)。
   - 该验证器使用 **DSPy modules** 自动提取风险并为假设类型加权，并指出丰富的业务上下文导致 **AKR 从 0.45 降至 0.28**。
- ****DSPy 和 GEPA 联手构建多智能体系统****：一位社区成员发表了一篇关于使用 **DSPy** 和 **GEPA** 构建和优化简单多智能体系统的博客文章，并分享了 [文章链接](https://kargarisaac.medium.com/building-and-optimizing-multi-agent-rag-systems-with-dspy-and-gepa-2b88b5838ce2)。
   - 作者将该项目描述为对 **DSPy** 和 **GEPA** 的一次学习经历和探索。
- ****DSPy 每周通讯上线****：全新的 [DSPy 每周通讯](http://dspyweekly.com) 已发布，其中包含一个招聘板块。
   - 创建者计划实现一个爬虫以确保招聘板块的信息广泛性，并欢迎反馈、建议和 Bug 报告。
- ****AI Agents 现在可以用 DSPy 玩 Taboo 游戏****：一位爱好者创建了玩 **Taboo** 游戏的 AI Agents，并在 [博客文章](https://joelgrus.com/2025/09/08/vibe-coding-9-ai-agents-that-play-taboo/) 中展示了该项目。


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1414229358771572906)** (1 messages): 

> `single-label classification, named entity recognition` 


- **单标签策略**：对于 **单标签分类 (single-label classification)**，你可以跳过父层级，专注于检索终端层级的候选对象，然后对候选对象做出最终预测。
- **NER 作为信号**：在某些情况下，**命名实体识别 (Named entity recognition)** 或其他分类可以被用作“信号”。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1413613216474599484)** (91 messages🔥🔥): 

> `VibeVoice Repo, Async Speedup, Nano Banana Hackathon, Data Hygiene + Eval Reflexivity, DSPy Project Structure` 


- ****VibeVoice** 仓库因违反通用性规定被移除**：微软于 **2025 年 9 月 5 日** 禁用了 [**VibeVoice**](https://github.com/microsoft/VibeVoice) 仓库，原因是其使用方式与其预期用途不符，此举旨在优先考虑负责任的 AI 使用。
   - 成员们对此表示惊讶，但也感到兴奋，因为他们现在可以与 Google Notebook 竞争。
- ****异步的魅力：大幅提升 LLM 动量****：成员们注意到，由于操作的 **I/O 密集型特性**，切换到 **async** 流水线显著加快了 LLM 调用速度。
   - 一位成员对速度的提升感到 *惊喜*，仅通过极少的代码更改就实现了这一目标。
- ****Nano Banana 盛会：黑客松热度高涨****：成员们讨论了即将到来的 **Nano Banana Hackathon**，并猜测会有多少参与者使用 **GEPA**。
   - 一位成员分享了结合 **数据清洗 (Data hygiene)** 和 **评估自反性 (Eval reflexivity)** 对于发挥 DSPy 真正优势的重要性。
- ****DeepSeek 的对话困境：Max Tokens 至关重要****：用户在 **DSPy REACT** 模块中使用 **GLM** 和 **DeepSeek** 模型时遇到了问题，具体表现为缺失输出字段，如 `next_thought`、`next_tool_name` 和 `next_tool_args`。
   - 错误提示表明 `max_tokens` 可能设置得太短，因为 **DeepSeek** 模型以冗长著称。
- ****DSPy 数据转储：Discord 讨论值得拥有可搜索性****：用户讨论了如何让 **DSPy Discord 讨论** 更容易被搜索引擎发现，并强调 **PyTorch 的成功** 归功于其可搜索的论坛。
   - 建议包括每月进行 Discord 数据转储、创建 SEO 友好内容，或构建一个 **DSPy app** 将聊天内容提炼为可搜索的片段。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1413710641587421308)** (17 messages🔥): 

> `Apple GPU, Mojo Use Cases Beyond AI, Ray Tracing in Mojo, Community Meeting` 


- **Apple GPU 编程刚刚起步**：成员们讨论了 Mojo 中 **Apple GPU** 支持的极早期阶段，建议设置 `MOJO_ENABLE_HAS_GPU_FOR_APPLE=1` 并尝试向量加法示例，更多详情请参见 [论坛](https://forum.modular.com/t/gpu-programming-on-mac/2174/8)。
- **Mojo 不再仅仅用于 AI**：虽然 Modular 的主要重点是 AI，但 Mojo 也可以用于其他应用，例如 **CLI**（参见 [Prism](https://github.com/thatstoasty/prism)）以及潜在的 Web 开发（参见 [lightbug_http](https://github.com/saviorand/lightbug_http)）。
- **通过 Mojo 访问光线追踪硬件？**：通过 **LLVM intrinsics** 扩展 Mojo 以允许访问光线追踪（Ray Tracing）硬件可能是可行的，或许可以作为第三方库实现；一个 **GPU raytracer** 示例可以在 [这里](https://github.com/gonsolo/mojo_gpu_raytracer) 找到。
- **Mojo 社区会议正在进行中！**：九月社区会议正在进行，内容涵盖 Mojo 愿景与路线图、GSplat Kernels 以及 HyperLogLog —— 你可以从 [这里](https://forum.modular.com/t/september-2025-community-meeting/2186) 加入。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1414124982300180662)** (39 messages🔥): 

> `AMD GPU Issue, ROCm Version, Tier 3 GPU Support, EmberJson Explicit Copies, Dict API Improvements` 


- **AMD GPU 驱动困扰用户**：一位用户报告其 **AMD Radeon RX 7900** GPU 无法被识别，即使在更新驱动程序后，无论使用什么命令都会收到反馈截图，而 [官方文档](https://docs.modular.com/max/packages/) 建议的最低驱动版本为 **6.3.3**。
   - 会议中澄清了 **ROCm 版本** 与驱动程序版本是不同的，并且该 GPU 可能处于支持有限的 Tier 3 层级。
- **显式复制引发 EmberJson PR**：鉴于 Mojo 更改为显式可复制对象，一位开发者正在为 **EmberJson** 提交 [PR](https://github.com/bgreni/EmberJson/pull/52)，以使复制操作显式化。
   - 该开发者表示希望目前保持复制显式化，可能仅保持指针为隐式可复制，以减少段错误（segfaults）并改进 **c binder** 的进展。
- **提议为 Dict API 增加 Take Items 迭代器**：一位在处理 [EmberJson PR #53](https://github.com/bgreni/EmberJson/pull/53) 的开发者发现从字典中“移动（moving）”项非常别扭，并提议为 **Dict API** 增加一个 `take_items()` 迭代器。
   - 会议指出 `Dict.popitem` 目前会执行键和值的复制，可能会添加一个新的 PR 来创建一个拥有所有权的项迭代器（owned iterator）或一个不抛出异常版本的 `popitem`。
- **关于 Unsafe Pop 函数的辩论**：在为 **Dict API** 实现 `take_items` 时，讨论了实现一个“无异常、无分支的 pop”或 `unsafe_pop` 函数，并提交了一个 [PR](https://github.com/modular/modular/pull/5289) 草案。
   - 共识是提供一个“抛出异常版”、一个 **Optional** 版以及一个通过编译器开关启用调试断言的“unsafe unchecked 版本”。


  

---

### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1413693462720675952)** (31 messages🔥): 

> `SDK Cache TensorValue, ModuleV3, max.tensor symbolic dimensions, Model Serialization with Pickle, MAX model format` 


- **SDK 缓存 TensorValue 类型以加速图构建**：[[SDK] Cache TensorValue.type](https://github.com/modular/modular/commit/0e861fed901f1ad9be398364acd837b5c307b2bf) 已合并，这将在一定程度上加快图构建速度。
   - 图构建中的位置信息已被关闭（通过环境变量选择性开启）；一个 PR 正在进行中，预计下周进入 nightly 构建版本。
- **ModuleV3 已合并，PyTorch 用户会感到熟悉吗？**：[ModuleV3](https://gist.github.com/bethebunny/fc93b16914542cbba9084094e15169fd) 已合并，并构建在 eager tensor (`from max.experimental.tensor import Tensor`) 之上，旨在让 PyTorch 用户感到更加亲切。
   - 一位成员提供了一个 [gist](https://gist.github.com/bethebunny/fc93b16914542cbba9084094e15169fd)，包含了 Linear、Sequential 和 Embedding 的测试和基础实现，因为开源测试代码被意外遗漏了。
- **讨论在 MAX Tensor 中命名符号维度**：讨论涉及在 MAX eager tensor 中为符号维度命名。
   - 虽然目前在机制上还无法实现，但成员指出实现这一功能并无障碍，尽管全局计算图需要更新其 paramdecls。
- **为 Eager 模式重叠编译 MAX 模型**：成员们讨论了使用 pickle 序列化和反序列化模型，其中一位成员提到了 Torch inductor 的 QOL 特性，即编译在单独的进程中进行。
   - MAX 模型编译会释放 GIL，因此建议可以使用 **asyncio** 来重叠 eager 执行和编译，尽管 LLVM 对象编译器（object compiler）不是线程安全的。
- **零拷贝 MAX 模型权重即将到来？**：一位成员请求一种支持 **zero copy** (mmap) 且能处理块浮点（block floats）的 MAX 模型格式。
   - 建议可以将权重作为 **safetensors** 可选附加，并与不含权重的模型定义一起打包在归档文件中。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1413665806432211059)** (48 messages🔥): 

> `Low Rank Updates vs Replacement, Sparsity and Quantization in LLMs, Distillation and Model Complexity, Codex IDE and Code Generation, Arxiv Paper without Empirical Results` 


- **低秩层并非音乐魔法**：将矩阵替换为低秩层类似于将整首歌换成几个随机的正弦波，这对于更新或保留原始信息并不是一种有效的方法。
   - 更好的类比是在各处添加一些音符。
- **稀疏性并不等同于量化**：一位成员对稀疏性有助于量化表示怀疑，并指出许多量化方法首先对权重进行随机投影，使每个单元的表现趋于高斯分布。
   - 有人提到 **MoE 模型** 中的稀疏模式尚不十分清晰，而 **ReLU** 可以诱导较高程度的稀疏性。
- **蒸馏揭示了更低的复杂度**：蒸馏表明模型在训练后具有较低的复杂度，因为它涉及用显著更少的数据/信息来描述与原始模型非常接近的事物。
   - 大模型探索了更多的可能配置，但其最优状态可以被简单地描述，从而能够用更小的模型进行复制。
- **代码规范优于实现**：一位成员在不看代码的情况下，使用 **Codex IDE** 在现有代码库中实现了一个自定义学习算法，基于经验信任它能处理实现错误。
   - 他们仍然需要提供人类智能来引导它，提供更全面的解决方案并运行多次训练以找到正确的超参数，这证明了 AI 需要引导才能发挥效力。
- **论文证明取悦人心**：一位成员询问提交一篇仅由证明和其他文献动机组成、没有实证结果的 arXiv 论文是否会被接受。
   - 另一位成员开玩笑说这听起来像是一篇文献综述。


  

---

### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1414616862771253439)** (6 messages): 

> `RAG Agent 资源，Langchain 替代方案，用于 Agent 的 While loops` 


- **寻找 RAG/Agent 资源**：一名成员询问用于创建 RAG/Agent 的*优质资源*。
   - 他们注意到了围绕 **Langchain** 的负面评价，并询问了替代方案。
- **While loop 的智慧**：一名成员开玩笑地建议使用 **While loop** 作为 **Langchain** 的替代方案来创建 Agent。
   - 他们阐明了自己的怀疑态度，表示即使经过反复调研，也不确定 **Langchain** 到底提供了什么价值。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1413931013511577600)** (13 messages🔥): 

> `In-Memory Computing, OpenAI 招聘平台, ASML 投资 LLM, 定制预训练模型, Mistral 的盈利能力` 


- **DRAM 上的 In-Memory Computing 亮相！**：一篇新论文（[In-Memory Computing on Normal DRAM](https://arxiv.org/abs/2503.23817)）讨论了在标准 **DRAM** 上执行 In-Memory Computing，并指出由于难以映射 `NOT` 操作，逻辑非是一个主要的限制因素。
   - 存储+计算的主要挑战是实现足够快的存储和并行工作负载，因为 In-Memory Computing 不进行时间复用（time multiplex），这也是为什么研究通常更青睐 **ReRAM** 之类的技术。
- **OpenAI 凭借 AI 平台进军就业市场！**：据 Bloomberg 报道（[OpenAI Unveils Jobs Platform, Certification Program for AI Roles](https://www.bloomberg.com/news/articles/2025-09-04/openai-unveils-jobs-platform-certification-program-for-ai-roles)），**OpenAI** 计划于明年推出一个新的 **AI 驱动的招聘平台**，并引入针对 AI 技能的**认证计划**。
   - The Decoder 证实了这一消息（[OpenAI Plans an AI-Powered Job Platform to Certify and Connect Workers](https://the-decoder.com/openai-plans-an-ai-powered-job-platform-to-certify-and-connect-workers/)），一名成员对此表示讽刺，将其比作来自 Microsoft 和 Oracle 的*毫无意义的 ABCD 认证*。
- **ASML 关注 LLM 投资**：成员们正在讨论 ASML 对一家 LLM 公司的投资（[来源](https://x.com/ns123abc/status/1964738357403308147)）。
   - 这项投资可能来自某种通用投资基金，因为从战略上讲，一家光刻机公司投资 LLM 公司并没有太大意义，但也有人推测这可能是为了内部模型。
- **定制模型优于通用模型**：有人提到，像 **ASML** 这种级别的定制化需求，可能会让他们瞄准部分定制的预训练模型，而不仅仅是另一个 Finetune。
   - 这是因为，如果不受跨大量任务的通用性限制，针对更窄领域训练的模型可以获得更好的性能。
- **Mistral 的十亿美元估值受到质疑**：一名成员认为 **Mistral** 的 LLM 在内部并不值 **13 亿美元**，尤其是在存在安全的闭源和开源替代方案的情况下。
   - 该成员还补充说，Mistral 似乎并没有盈利，这其中可能存在政治利益的博弈。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1413602613143797910)** (33 messages🔥): 

> `Aider 成功案例, Aider vs 全 Agent 工具, Codex 的 Token 效率, GPT-5 与 Aider, 快速高效的 Web 开发模型` 


- **Aider 并非完全 Agent 化；仍需开发者**：一位成员表示 **Aider 并非纯粹的 Agent 工具**，结果很大程度上取决于开发者。
   - 他们补充道，Aider 允许更好地控制 **LLM 的上下文**，且其*文件编辑机制非常出色*。
- **Codex 令人印象深刻的 Token 效率**：一位成员结束了一个 **Codex 会话**，展示了令人印象深刻的 Token 效率，总计使用了 **2304542 tokens**。
   - 该会话包含 **2218442 input tokens**（+ **16140160 cached**）和 **86100 output tokens**。
- **Gemini Flash 2.5 在 Web 开发中表现出色**：对于使用 headless CMS 的基础静态 Web 开发，推荐使用 **Gemini Flash 2.5**，理由是 **Gemini 2.5 Pro 的延迟** *严重影响了我的生产力*。
   - 据称，使用 **Jekyll** 和 **Gemini Flash 2.5** 构建了 **3 个静态网站**。
- **下游项目中提供 Aider 的 MCP 配置**：由于主仓库尚未支持 **MCP**，用户一直将 **PR 3937** 合并到个人 fork 中来实现它，详见 [dwash96/aider-ce 仓库](https://github.com/dwash96/aider-ce)。
   - 该仓库用于配置 **MCP**，并包含文档（[mcp.md](https://github.com/dwash96/aider-ce/blob/main/aider/website/docs/config/mcp.md)）。
- **10 倍编码速度提升是个神话**：一位成员认为 *AI 赋能编码带来的“10 倍速提升”是个神话*，建议 **25-50%** 更加现实。
   - 他们补充说，**LLM 在自动化打字方面表现出色**，但在*自动化思考方面只能算差强人意*。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1413663418959003689)** (5 messages): 

> `Aider 代码生成百分比, Aider 的安全机制, 推理力度与编辑操作, Aider 中的确认系统, Linting 配置` 


- **关于 Aider 代码自主权的说法**：一位成员提到 [Aider 声称](https://aider.chat) 编写了其自身 **70-80%** 的代码，并建议使用它来架构其自身的代码库。
   - 该建议旨在通过一种*类似“盗梦空间”式*的方法来发现更多关于其工作原理的信息，尽管其他人认为这个建议没什么帮助。
- **Aider 的安全性是灵活的**：一位成员询问了 **Aider** 中的安全机制，以防止危险的代码执行，如未经核实的网络请求或任意 Shell 执行。
   - 回复指出，不存在用于自动预防或黑名单的此类系统，因为安全与不安全之间的界限非常宽泛，且因使用场景而异。
- **推理力度适用于所有地方**：`--reasoning-effort high` 标志确实适用于编辑操作。
   - 成员们怀疑 **Deepseek 3.1** 作为一个较弱的模型，有时会“想得太多”反而适得其反，`--reasoning-effort high` 应该只在更强大的模型上使用。
- **揭秘 Aider 确认系统**：确认系统在执行前需要用户输入，并通过一个 *“我该这样做吗？”* 的提示，结合了针对 Web 搜索、工具调用和 CLI 命令的输出解析。
   - 有人提到，防御更多是在 Linter 配置层面，并由确认机制把关，但宿主环境有责任对移交操作进行拦截。
- **Linting 和黑名单中的防御**：针对潜在有害代码的防御在于 Linting 配置、命令黑名单以及域名黑白名单。
   - 系统负责检测执行操作何时被移交给底层宿主，并通过任何你认为合理的方法来拦截这些移交。


  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1413619616806080563)** (20 条消息🔥): 

> `Kernel Removal Project, Digital Ocean Issues, ShapeTracker Bounty Removal, Tinygrad Community Bounties, Meeting #87 Topics` 


- **用户在 Digital Ocean 上处理内核移除项目时遇到困难**：一名成员询问如何协助 **kernel removal project**，但在 **Digital Ocean** 上遇到了问题，报告称对 droplet 进行电源循环会导致 **Docker container** 无法启动。
   - 在删除并重新创建一个新的 droplet 后，问题得到了解决，这进一步印证了“物理拥有硬件而非云端访问”更好的观点。
- **ShapeTracker 悬赏即将落幕**：随着近期进展，在 **Lean** 中证明 **ShapeTracker** 可合并性的悬赏计划已定于从 [悬赏列表](https://docs.google.com/spreadsheets/d/1WKHbT-7KOgjEawq5h5Ic1qUWzpfAzuD_J06N1JwOCGs/edit?gid=0#gid=0&range=A89) 中移除。
   - George Hotz 确认“我们现在可以移除它了”，并强调“这都只是符号规则”。
- **Tinygrad 社区会议议程发布**：发布了第 **#87** 次会议的议程，包括 **company updates**、将 **rangeify** 设为默认、**CI speed**、**MLPerf Llama**、**viz tool**、**CPU thread**、**symbolic**、**cloud** 以及**其他悬赏**等主题。
   - 会议定于**圣迭戈时间周一上午 9 点**举行。
- **专家并行 MoE 策略受到质疑**：一名成员询问如果 big graph 和 remote 都能按计划进行，**expert-parallel Mixture of Experts (MoE)** 将如何处理。
   - 他们担心静态调度可能会破坏这一过程。
- **呼吁对 Test Tensor Core 进行模块化**：一名成员建议将“**Test Tensor Core**”移动到 `test/unit/test_emulated_tc.py` 的独立文件中。
   - 未给出此次模块化的理由。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1414258082921447455)** (2 条消息): 

> `graph_rewrite_map(), Tensor vs MathTrait` 


- **解析 graph_rewrite_map() 内部机制**：用户寻求对 `graph_rewrite_map()` 运作方式的全面理解，询问其中自底向上（bottom-up）和自顶向下（top-down）匹配策略的区别。
- **Tensor 方法：Tensor 还是 MathTrait？**：用户质疑为什么 **Tensor** 上的方法有时会返回 `(Tensor | MathTrait)`，并指出这可能导致类型检查问题，因为像 `.silu()` 这样的方法无法应用于 **MathTrait**。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1413619165889036308)** (20 条消息🔥): 

> `Manus bugging out, Manus API key, New MCP and API connectors, Flowith invitation, Politeness to AIs` 


- **AI 亲和力：礼貌是有回报的！**：一篇研究论文建议对 AI 保持礼貌会产生更好的结果，详见 [arxiv.org 论文](https://arxiv.org/pdf/2402.14531)。
   - 该研究*科学地证明了* AI 会对礼貌的请求做出积极响应。
- **Flowith 的精彩邀请！**：一名成员分享了 [Flowith 邀请链接](https://flowith.io/?inv=EXVVQXH8QBRRSIVH)，为新用户提供专属优惠。
   - Flowith 似乎是一个新平台。
- **Manus 的故障乱象！**：一名成员报告称 **Manus** 出现故障，在被要求等待输入后陷入死循环。
   - 其他人推测这与默认的 **adaptive mode** 及其对额度消耗的影响有关。
- **MCP 神奇的 API 连接器！**：成员们对最近推出的新 **MCP** 和 **API connectors** 功能感到兴奋。
   - 未提及具体的发布日期。
- **寻找 Manus 的 API 密钥**：一名成员请求协助获取 **Manus API key**。
   - 另一名成员确认免费额度已停止发放，并指出缺乏相关信息。