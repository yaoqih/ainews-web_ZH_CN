---
companies:
- google
- meta-ai-fair
- perplexity-ai
- baseten
date: '2025-09-26T05:44:39.731046Z'
description: '**谷歌 (Google)** 发布了密集的 9 月更新，包括具有增强空间/时间推理能力的 **Gemini Robotics 1.5**、**Gemini
  Live**、**EmbeddingGemma**，以及助力创意工作流的 **Veo 3 GA**。他们还推出了智能体（agentic）功能，例如餐厅预订智能体，并降低了
  **Gemini 2.5 Flash** 的价格。


  **Meta AI** 推出了开源权重的 **Code World Model (CWM) 32B**，该模型在代码语义和数学基准测试中表现优异，并在通过执行轨迹（execution
  traces）训练代码模型方面进行了创新。


  本地优先的编程配置亮点包括在消费级 GPU 上高效运行的 **Qwen3-Coder-30B**，并搭配 **Cline** 和 **LM Studio** 等工具。运行时（Runtime）改进方面，**vLLM
  v1** 开始支持混合模型，**mlx-lm** 为苹果芯片（Apple silicon）增加了批量推理功能。


  在基础设施领域，**FlashAttention 4** 被逆向工程，揭示了通过架构优化实现的约 20% 的提速。**Perplexity AI** 推进了其独立网页索引和浏览
  API，并即将更新信息流（feed）。**Superhuman** 通过使用 **Baseten** 实现了嵌入（embedding）延迟的优化。'
id: MjAyNS0w
models:
- gemini-robotics-1.5
- gemini-live
- embeddinggemma
- veo-3
- gemini-2.5-flash
- code-world-model-32b
- qwen3-coder-30b
- vllm-v1
- mlx-lm
- flashattention-4
people:
- osanseviero
- _anniexie
- rmstein
- scaling01
- giffmana
- cline
- redhat_ai
- awnihannun
- charles_irl
- bernhardsson
- akshat_b
- aravsrinivas
title: 今天没发生什么特别的事。
topics:
- spatial-reasoning
- temporal-reasoning
- agentic-ai
- code-semantics
- code-execution-traces
- coding-infrastructure
- runtime-optimization
- batch-inference
- embedding-latency
- api
- model-optimization
- model-performance
---

**一周结束前的宁静时光**

> 2025年9月25日至9月26日的 AI 新闻。我们为您检查了 12 个 subreddit、544 个 Twitter 账号和 23 个 Discord 社区（195 个频道，5022 条消息）。预计节省阅读时间（以 200wpm 计算）：400 分钟。我们的新网站现已上线，支持全元数据搜索，并以精美的 vibe coded 方式呈现所有往期内容。详见 https://news.smol.ai/ 并通过 @smol_ai 向我们提供反馈！

下周将有大量发布，所以现在能喘口气很不错。[申请 AIE CODE 第二轮](https://apply.ai.engineer/)！

---

# AI Twitter 回顾

**Google 的 9 月技术栈：Gemini Robotics 1.5、Live、Veo 3、Flash 定价**

- **Gemini Robotics 1.5 + Live + Veo 3 GA**：Google 在 9 月发布了一系列密集更新：Gemini Robotics 1.5（包括高级推理“ER 1.5”）、最新的 Gemini Live、EmbeddingGemma、Veo 3 GA + API 更新、AI Edge gallery、Batch API 嵌入支持、Flash/Flash Lite 更新、Chrome DevTools MCP、VaultGemma 等，据 [@osanseviero](https://twitter.com/osanseviero/status/1971468195308712431) 报道。Robotics-ER 1.5 被定位在空间/时间推理方面表现强劲，具备“思考”能力以改进答案 [@_anniexie](https://twitter.com/_anniexie/status/1971477645096517832)。Veo 3 已经开始支持生产环境的创意工作流（例如 Flow by Google 的音乐视频案例研究）[@FlowbyGoogle](https://twitter.com/FlowbyGoogle/status/1971607613805867314)。Google 还在向更广泛的用户推出 Agentic 功能，例如 Labs 中的餐厅预订 Agent [@rmstein](https://twitter.com/rmstein/status/1971617040193724661)。与此同时，Gemini 2.5 Flash 获得了小幅质量提升，但价格降低了约 30% [@scaling01](https://twitter.com/scaling01/status/1971578192512029045)。

**代码智能与 Agentic 编程**

- **Meta 的 Code World Model (CWM)**：新的开源权重 32B 模型，通过执行轨迹和 Agentic 交互（Bug 修复、编辑、Docker 运行）学习代码语义。宣称：逐步模拟 Python、多轮软件任务、131k 上下文；竞争性的编程指标（如 65.7% SWE-bench Verified，68.4% LiveCodeBench）以及强大的数学能力（96.5% Math-500，75.8% AIME-24）。论文、代码、权重：[摘要](https://twitter.com/TheTuringPost/status/1971697629697659099)，[论文](https://twitter.com/TheTuringPost/status/1971697642288959496)。相关思路：通过将源代码与解释器状态交织在一起来训练代码模型，以强制实现语义理解 [@giffmana](https://twitter.com/giffmana/status/1971507878025445653)。
- **本地优先的编程设置**：Qwen3-Coder-30B (AWQ 4-bit) 在单张 3090 上达到约 115 tok/s，“零样本搞定吃豆人游戏” [@QuixiAI](https://twitter.com/QuixiAI/status/1971427136977453184)。开发者正将 Qwen3-Coder 与 Cline + LM Studio 搭配使用，以实现高质量的本地编程 [@cline](https://twitter.com/cline/status/1971591597080064121) ([指南](https://twitter.com/awnihannun/status/1971603427131351218)，[博客](https://twitter.com/cline/status/1971591609386188993))。Cline 还发布了“构建工作流的工作流”([Prompt 秘籍](https://twitter.com/cline/status/1971436086217122213)，[博客](https://twitter.com/cline/status/1971436097965375689))，并在免费 Alpha 测试期间悄悄将其“code-supernova”提供商的上下文从 200k 提升到了 1M token [@cline](https://twitter.com/cline/status/1971660202387951962)。
- **运行时/后端**：vLLM v1 将混合模型（如 Mamba/Mamba2、线性注意力）视为一等公民，性能较 v0 有所提升 [@RedHat_AI](https://twitter.com/RedHat_AI/status/1971569727844876350)。在 Apple silicon 上，mlx-lm 为混合 SSM/滑动窗口注意力添加了批处理推理支持，并支持 Meta 的 CWM [@awnihannun](https://twitter.com/awnihannun/status/1971763001880670213)。

**系统与基础设施：内核、搜索与托管**

- **FlashAttention 4 解读**：Modal 对 FA4 进行了逆向工程，解释了约 20% 提速的来源：专门的 warp 布局、用于 softmax 的 exp 三次近似、更激进的异步处理。深度文章和代码指针：[@charles_irl](https://twitter.com/charles_irl/status/1971587871237898482), [blog](https://twitter.com/charles_irl/status/1971587874496868601)，以及工程评论 [@bernhardsson](https://twitter.com/bernhardsson/status/1971603562355716160), [@akshat_b](https://twitter.com/akshat_b/status/1971617146930450758)。
- **搜索 API 和网络索引**：Perplexity 继续构建非 Google/Microsoft 的网络索引（[论点](https://twitter.com/AravSrinivas/status/1971438329460867413)）并正在发布浏览 API；发现流（discover feed）更新将于下周上线（iOS 首发）[@AravSrinivas](https://twitter.com/AravSrinivas/status/1971443978810896424), [更新](https://twitter.com/AravSrinivas/status/1971687653545525467)。开发者已经将其作为自定义工具进行集成 [@thdxr](https://twitter.com/thdxr/status/1971510163501953436)。
- **推理基础设施**：Superhuman 通过迁移到 Baseten，将 P95 embedding 延迟降低了约 80% 至 500ms [@basetenco](https://twitter.com/basetenco/status/1971683977242259623)。Ollama Cloud 添加了免费试用的 Kimi K2 “1T-cloud” 和 DeepSeek V3.1 “671b-cloud” SKU [@ollama](https://twitter.com/ollama/status/1971750071483167010)。NVIDIA 在开源贡献方面日益活跃（过去一年在 HF 上发布了 300 多个模型/数据集/应用）[@ClementDelangue](https://twitter.com/ClementDelangue/status/1971698860146999502)。

**研究亮点：RLHF 变体、解码、3D 零件、科学 FM**

- **RLHF 与解码**：RLBFF 提出从自然语言反馈中提取可进行二进制检查的原则，并将其与可验证的奖励相结合，以训练能够捕捉超越正确性的细微差别的奖励模型 [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1971520102857408705) ([摘要](https://twitter.com/iScienceLuvr/status/1971520105143242952))。VCRL 探索了针对 LLM 的基于方差的课程 RL [@_akhaliq](https://twitter.com/_akhaliq/status/1971593807365132382)。LATTS 通过从 LM 和奖励模型的乘积中采样进行解码，追踪 token 上的准确率 [@f14bertolotti](https://twitter.com/f14bertolotti/status/1971469173185527955)。
- **3D 零件级生成**：腾讯发布了 Hunyuan3D-Part，包含两个模型：P3-SAM（首个原生 3D 零件分割）和 X-Part（SOTA 的可控性/形状质量）。基于包含 370 万个形状及清晰零件标注的数据集训练；提供了完整代码/权重和 demo [@TencentHunyuan](https://twitter.com/TencentHunyuan/status/1971491034044694798)。
- **少数据下的多模态推理**：阿里巴巴的 MMR1 引入了 Variance-Aware Sampling，以在高质量数据稀缺的情况下稳定 RL 微调；发布了约 160 万个 CoT、1.5 万个 RL QA 数据集以及 3B/7B/32B 模型 [@HuggingPapers](https://twitter.com/HuggingPapers/status/1971487864807469236)。
- **领域 FM**：SciReasoner 在 206B 科学 token（文本、序列和对）上进行预训练，通过 40M SFT 和带有任务形状奖励的 RL 进行对齐，以激发深思熟虑的科学推理 [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1971519610630586702)。在医疗保健领域，CATCH-FM 将 EHR FM 扩展到 2.4B 参数用于癌症预筛查，并在 EHRSHOT 的胰腺风险评估上达到 SOTA [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1971521050057072995)。

**基准测试与评估实践：GDPVal、SWE-bench 以及“评估即 PRD”**

- **GDPVal 辩论**：一项涵盖美国前 9 大 GDP 行业中 44 种职业任务的新 Benchmark 引发了激烈讨论。支持者认为它将“有用性”操作化，并显示模型在经济衡量标准下已达到 AGI 的 77–95% [@Smol_AI](https://twitter.com/Smol_AI/status/1971426804826267994), [@swyx](https://twitter.com/swyx/status/1971427791770882463), [@markchen90](https://twitter.com/markchen90/status/1971449404734439831)。怀疑者警告不要陷入字面主义，指出任务/选择偏差以及评分者风格效应，并强调应关注趋势而非阈值；注意到模型完成任务的速度/成本比专家快/便宜约 100 倍，但质疑其在现实世界中的迁移能力 [@scaling01](https://twitter.com/scaling01/status/1971431825433374866), [skepticism](https://twitter.com/scaling01/status/1971432462820802834), [style bias](https://twitter.com/scaling01/status/1971432758817050970), [task bias](https://twitter.com/scaling01/status/1971433067266089395)。
- **SWE-bench Verified 澄清**：根据 [@alexandr_wang](https://twitter.com/alexandr_wang/status/1971603685559140663)，近期结果中广泛流传的数字是 TTS (tools-to-success) 上的 pass@1。
- **评估实践**：评估（Evals）正日益成为定义产品的因素（“新的 PRD”），但缺乏人类监督的 LLM-as-judge 是不可靠的。错误分析应先于指标设计；人机回环（Human-in-the-loop）能建立信任 [podcast recap via @bnicholehopkins](https://twitter.com/bnicholehopkins/status/1971683830269350161)。ARC Prize 在波士顿举办了一场专注于智能交互式 Benchmark 的活动 [@arcprize](https://twitter.com/arcprize/status/1971609644004200693)。一个关于有用性的实用北极星指标：使用的 Token 数 / 花费的美元 [@scaling01](https://twitter.com/scaling01/status/1971433691848262084)。

**优化与缩放理论：Modular Manifolds, MoE 计算, 计算缩放, Tokenization**

- **Modular Manifolds (Thinky Machines)**：Jeremy Bernstein 等人的新文章共同设计了在权重矩阵上具有流形约束（例如 Stiefel：奇异值 = 1）的优化器，将 Muon (“managed metrics”) 扩展到稳定特定“形状”的训练。获得了从业者的强烈认可；还讨论了逐层调度/判别式微调 [@thinkymachines](https://twitter.com/thinkymachines/status/1971623409873244462), [@jxbz](https://twitter.com/jxbz/status/1971703483767435446), [@johnschulman2](https://twitter.com/johnschulman2/status/1971630456471945711), [@cHHillee](https://twitter.com/cHHillee/status/1971641318888853748), [@Dorialexander](https://twitter.com/Dorialexander/status/1971631250801844687)。
- **MoE 计算最优性与内核**：从业者认为，如果根据总参数/激活参数来缩放数据，MoE 在生命周期内是计算最优的；数据规模（“数十万亿” Token）是瓶颈 [@teortaxesTex](https://twitter.com/teortaxesTex/status/1971453835207131244), [follow-up](https://twitter.com/teortaxesTex/status/1971454807455265156)。对于超大型稠密模型（如 405B）与更稀疏的 MoE 之间存在争议 [@scaling01](https://twitter.com/scaling01/status/1971541644647522595)。内核级增益至关重要：Triton RoPE 比 PyTorch 更快（0.083ms vs 0.235ms） [@vikhyatk](https://twitter.com/vikhyatk/status/1971694488004481058)。此外，对于极长上下文，Attention 每查询 O(T) 的复杂度正变得越来越难以维持 [@francoisfleuret](https://twitter.com/francoisfleuret/status/1971632756716372053)。
- **OpenAI 的计算缩放**：最新分析表明，由于较小规模的 Post-training 带来了超额回报，GPT-5 使用的总训练计算量少于 GPT-4.5；作者预计随着基础设施落地，GPT-6 将回升至更高的训练 FLOPs [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1971675079219282422), [follow-up](https://twitter.com/EpochAIResearch/status/1971675189575602304)。
- **Tokenization 辩论**：多篇文章认为“Tokenizer-free”是一个误称；即使是字节也继承了 Unicode 的设计选择和偏差。Tokenization 仍然是核心设计元素；分享了实用指南和从零开始的 BPE 实现 [@giffmana](https://twitter.com/giffmana/status/1971500080072208674), [@rasbt](https://twitter.com/rasbt/status/1971575045769380056), [commentary](https://twitter.com/lateinteraction/status/1971548611700994538)。

**热门推文（按互动量排序）**

- 应届生不再问“怎么做”，而是直接用 ChatGPT “试一试”：关于初级员工自主性（agency）转变的观察 [@dylan522p](https://twitter.com/dylan522p/status/1971425552902082941) (~2.4K)。
- Richard Sutton 与 LLMs 的辩论：关于持续学习（continual learning）与当前 LLM 范式的长篇讨论；在社区中引发了显著的反复讨论 [@dwarkesh_sp](https://twitter.com/dwarkesh_sp/status/1971606180553183379) (~2.5K)。
- Modular Manifolds 文章：通过流形约束权重（manifold-constrained weights）实现稳定训练的理论/算法进展 [@thinkymachines](https://twitter.com/thinkymachines/status/1971623409873244462) (~2.5K)。
- OpenAI 平台：function calling 现在支持从工具返回文件/图像，而不仅仅是 JSON/文本 [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1971618905941856495) (~1.4K)。
- 腾讯 Hunyuan3D-Part：开源的部件级 3D 形状生成，具有原生 3D 分割和基于扩散的分解功能 [@TencentHunyuan](https://twitter.com/TencentHunyuan/status/1971491034044694798) (~1.1K)。

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. Qwen3 路线图 + abliterated 去审查结果

- [**阿里巴巴刚刚发布了 Qwen 路线图。其雄心壮志令人震惊！**](https://www.reddit.com/r/LocalLLaMA/comments/1nq182d/alibaba_just_unveiled_their_qwen_roadmap_the/) (热度: 954): **阿里巴巴的 Qwen 路线图幻灯片强调了向具有极端扩展能力的统一多模态技术栈的激进推进：上下文窗口从** `1M → 100M` **tokens，参数量从约** `1T → 10T`**，推理时计算（test-time compute）扩展从** `64k → 1M`**，以及训练数据从** `10T → 100T` **tokens。它还强调了无限的合成数据生成流水线，以及在任务复杂度、多 Agent 交互和持续/交互式学习方面更强大的 Agent 能力——在未来 Qwen 模型中加倍投入“扩展即一切”（scaling is all you need）的策略。** 热门评论质疑其可行性和可访问性：对 `100M` 上下文感到兴奋，但怀疑此类模型是否会保持开源，并对本地运行 `>1T` 参数模型的实用性表示担忧。
    - 声称的 `~100M` 上下文窗口意味着非标准的 attention 或内存系统；原始的全量 attention 是 O(n^2) 复杂度，在 100M tokens 时需要一个拥有 `1e16` 个条目的 attention 矩阵——这在计算上是不可行的。即使使用 KV caching，内存也会爆炸：对于 hidden size 为 8192、FP16 精度、约 80 层的模型，KV 缓存约为 32 KB/token/layer → 每层 `~3.2 TB`，100M tokens 总计 `~256 TB`。因此实际实现需要诸如检索增强分块（retrieval-augmented chunking）、循环/压缩内存（recurrent/compressive memory）或线性/部分 attention（例如 blockwise/ring attention），而不是真正的稠密长程 attention。
    - 在本地运行 `>1T` 参数的模型超出了消费级硬件的能力：仅参数在 BF16/FP16 下就约 `~2 TB`，8-bit 下约 `~1 TB`，4-bit 下约 `~0.5 TB`——这还没算上激活值和 KV cache。这需要通过 NVLink/NVSwitch 进行多节点模型并行；作为参考，一台 8x H100 80GB 服务器提供 640GB VRAM，因此一个万亿参数模型可能需要多个此类节点才能加载权重，并且需要巨大的互连带宽来维持推理吞吐量。
    - 一些评论者预计最大的 Qwen 检查点/长上下文变体将仅限 API，尽管阿里巴巴有开源较小 Qwen 模型的历史。实际上，最前沿的功能（如 `100M` 上下文或 `>1T` 参数）通常由于训练数据/许可和部署成本而保持封闭，而中等规模的开源权重则针对研究和本地部署；团队应据此规划集成和基准测试。

- [**重要提示：为什么 Abliterated 模型表现糟糕。这里有更好的 LLM 去审查方法。**](https://www.reddit.com/r/LocalLLaMA/comments/1nq0cp9/important_why_abliterated_models_suck_here_is_a/) (热度: 433): **楼主指出 “abliteration”（权重级去审查）会持续降低模型能力——特别是在像 Qwen3-30B-A3B 这样的 MoE 模型上——损害逻辑推理、Agent/工具调用行为，并增加幻觉，通常导致 abliterated 30B 模型的表现落后于非 abliterated 的 4-8B 模型。他们声称 Abliteration 后的微调在很大程度上恢复（“修复”）了性能：例如，mradermacher 的 Qwen3-30B-A3B-abliterated-erotic-i1-GGUF（在** `i1-Q4_K_S` **下测试）在 [MCP](https://modelcontextprotocol.io/) 下显示出比其他 abliterated Qwen3-30B 变体（Huihui 的 Thinking-2507, Fusion-9010, Instruct-2507）更低的幻觉和更可靠的工具调用，而 [mlabonne/NeuralDaredevil-8B-abliterated](https://huggingface.co/mlabonne/NeuralDaredevil-8B-abliterated)（Llama3-8B 的 DPO 微调版；[DPO](https://arxiv.org/abs/2305.18290)）据报道在保持去审查的同时超越了其基座模型。楼主敦促在高质量数据上对 abliterated Qwen3-30B-A3B 进行微调，以在不牺牲性能的情况下保留去审查特性；背景包括 GGUF 量化（[GGUF](https://github.com/ggerganov/llama.cpp/blob/master/gguf.md)）和 Qwen3 MoE 系列（[Qwen3](https://github.com/QwenLM/Qwen3)）。** 热门评论要求针对 NSFW 任务之外的 abliteration 影响建立标准化基准，并将观察到的恢复描述为预期的“模型愈合（model healing）”（无约束的权重编辑会破坏电路；进一步的训练会重新学习它们）。怀疑论者认为，如果需要微调，那么 abliteration 就没有必要——声称 abliterated+微调后的模型并不比单纯的微调模型表现更好。
    - 无约束的权重编辑（例如，将“负偏置”项归零或其他 abliteration 操作）会预见性地降低能力；评论者将编辑后的恢复称为 **“模型愈合（model healing）”**。解决方法是由损失函数引导的额外训练（SFT/LoRA 或全量微调），以便网络可以重新学习被编辑破坏的连接，类似于剪枝/量化需要重新训练以恢复困惑度（perplexity）和任务准确性。结论：如果你必须修改权重，请在目标函数下进行，否则在充分的微调修复之前，泛化能力会被破坏。
    - 对于 NSFW 之外的评估，**Uncensored General Intelligence (UGI)** 排行榜被建议作为更广泛的能力基准：https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard。这有助于量化 abliteration 是否损害了推理/指令遵循能力，并与单纯的微调进行比较，避免过度拟合仅限成人内容的指标。
    - 几位从业者报告称，“abliterated + 微调”很少能胜过直接微调，主张通过有针对性的 SFT 或合并进行非破坏性的去审查。引用的替代方案包括 “Josiefied”、“Dolphin” 以及 TheDrummer 发布的作品，例如 **Qwen3-8B-192k Josiefied (GGUF)** [https://huggingface.co/DavidAU/Qwen3-8B-192k-Josiefied-Uncensored-NEO-Max-GGUF]、**Dolphin-Mistral-24B Venice i1 (GGUF)** [https://huggingface.co/mradermacher/Dolphin-Mistral-24B-Venice-Edition-i1-GGUF] 以及 **TheDrummer** 的主页 [https://huggingface.co/TheDrummer]。其目标是在调整指令风格的同时保留基础能力（例如长上下文 `192k` 变体），避免灾难性的权重编辑。

### 2. 中国发布：混元 Image 3.0 + 风华 GPU

- [**腾讯正在预热全球最强大的开源文本生成图像模型，混元 Image 3.0 将于 9 月 28 日发布**](https://www.reddit.com/r/LocalLLaMA/comments/1nqaiaz/tencent_is_teasing_the_worlds_most_powerful/) (热度: 225): **腾讯正在预热混元 Image 3.0，这是一款定于** `9 月 28 日` **发布的开源文本生成图像模型，被誉为同类模型中“全球最强大”。预热海报（尚无基准测试或示例）暗示了极高的硬件需求——评论者将“VRAM 96”注释解读为建议约** `96 GB VRAM`**——而关于架构、训练规模、分辨率、吞吐量或许可证的细节尚未披露。图片：https://i.redd.it/t8w84ihz1crf1.jpeg** 评论者对发布前的炒作持怀疑态度，认为预热的模型往往不如“空降”的强力发布（例如 Qwen 对比炒作中的 GPT-5；SD3 对比 Flux），并质疑在缺乏公开基准测试或与其他大型开源 T2I 模型对比的情况下“最强大”的说法。

- 一条评论暗示了 96 GB VRAM 的需求（“vram 96? — 是的”），这表明推理可能针对的是数据中心级 GPU（A100/H100 或 RTX 6000 Ada），而非典型的消费级显卡。如果属实，这指向了一个非常大的 UNet/Transformer 或原生高分辨率采样（例如 2048px+），且未进行激进的内存优化；否则将需要多 GPU 的张量/流水线并行（tensor/pipeline parallelism）。需要关注的关键细节包括：使用 FlashAttention/xFormers/权重量化（FP8/INT8）时的内存占用、VAE offloading，以及在 1024–2048px 下的 batch-1 延迟/吞吐量。
- 用户强调了一个反复出现的模式：过度宣传的模型往往比“空降”式发布的模型表现更差，并以 Qwen 低调但强大的发布与 GPT-5 等大肆宣传的发布，以及社区围绕 SD3 与 FLUX 的结果为例。实际的经验是在接受“最强”声明之前，先等待严格的基准测试。所需的证据包括标准化指标（FID、CLIPScore/PickScore/HPSv2、GenEval 组合性）和受控提示词套件。
- 市场对与 Qwen Image、SDXL 和 FLUX 等开源模型进行正面交锋的需求很高，但目前尚无跨模型数据。为了证实其声明，腾讯应当展示质量-速度的权衡和资源概况：1024–2048px 下的 VRAM 占用、达到同等质量所需的步数、采样器设置，以及在常见的单 GPU 配置与数据中心 GPU 上的延迟。如果没有这些数据，“最强开源 T2I”的断言仍未得到证实。
- [**中国已经开始制造支持 CUDA 和 DirectX 的 GPU，因此 NVIDIA 的垄断地位即将结束。风华 3 号支持最新的 API，包括 DirectX 12、Vulkan 1.2 和 OpenGL 4.6。**](https://www.reddit.com/r/LocalLLaMA/comments/1nq1ia2/china_already_started_making_cuda_and_directx/) (热度: 702): **帖子声称一款中国 GPU“风华 3 号”原生支持现代图形/计算 API——DirectX 12、Vulkan 1.2、OpenGL 4.6——甚至支持 CUDA，暗示可能削弱 NVIDIA 的 CUDA 锁定。技术警告：API“支持”不等于完全的功能对等（例如 DX12 功能层级/Ultimate、SM 6.x），驱动成熟度、CTS/WHQL 一致性以及实际性能/兼容性尚不明确；非 NVIDIA 硬件上的 CUDA 通常依赖于重新实现/翻译（参见 AMD 的 HIP: https://github.com/ROCm-Developer-Tools/HIP, ZLUDA: https://github.com/vosen/ZLUDA）。** 热门评论指出 AMD 已经通过 HIP 提供了 CUDA 移植路径，且 ZLUDA 等项目可以翻译 CUDA，同时在证据/基准测试出炉前表达了怀疑，并暗示了潜在的地缘政治/出口管制后果（“禁令即将到来”）。
    - 几位用户指出 AMD 已经提供了一条类似于 CUDA 的路径：**HIP** 针对 ROCm 提供了源码级的 CUDA 兼容性（通过 hipify 和重命名 API），而 **ZLUDA** 等项目则实现了翻译层，以便在非 NVIDIA 后端（最初是 Intel Level Zero，现在是 AMD ROCm）上运行 CUDA 二进制文件。这意味着中国可以通过源码兼容层或 PTX/驱动翻译来提供 CUDA 支持，但长期可行性取决于对 NVIDIA 不断演进的 PTX/驱动 ABI 的追踪以及实现性能对等。链接：AMD HIP https://github.com/ROCm-Developer-Tools/HIP, ZLUDA https://github.com/vosen/ZLUDA。
    - 关于“风华 3 号”支持 **DirectX 12, Vulkan** `1.2`**, OpenGL** `4.6` 的说法引发了实现方面的疑问：实际效用取决于是否通过一致性测试/WHQL，以及是否支持现代着色器工具链（D3D12 的 DXIL/SM6.x）和功能层级（例如 12_1/12_2、DXR、mesh shaders、sampler feedback）。技术上具有意义的验证将是公开驱动程序，并列入 **Khronos Vulkan 一致性产品**页面以及获得 Microsoft WHQL/WDDM 认证；若缺乏这些，API 版本声明并不能保证应用/游戏的兼容性或性能。链接：Vulkan 一致性列表 https://www.khronos.org/conformance/adopters/conformant-products#vulkan, D3D12 功能层级 https://learn.microsoft.com/windows/win32/direct3d12/hardware-support。
    - 怀疑的焦点集中在缺乏基准测试和驱动成熟度证据上：如果没有第三方测试（着色器编译器正确性、帧生成时间稳定性、DX12 同步鲁棒性、D3D12 平铺资源/描述符堆限制、Vulkan CTS 通过率），目前尚不清楚其是否接近成熟厂商的水平。从历史上看，新的 Windows GPU 堆栈在 DXGI/WDDM 集成、着色器缓存和特定游戏的变通方案方面都举步维艰，因此在将该硬件视为可行的 NVIDIA 替代方案之前，需要具体的性能/兼容性数据（微基准测试以及游戏/计算工作负载）。

## 较少技术性的 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. OpenAI 4o 到 5 的路由错误报告及 Pro 订阅影响

- [**4o 故障请上报**](https://www.reddit.com/r/ChatGPT/comments/1nqso2x/4o_glitch_report_it/) (活跃度: 1272): **多名用户报告了一个路由/别名错误，即尽管显式选择了模型或使用了特定模型的 URL，选择 4o 却导致响应来自 “5/5‑auto”；重新生成也会切换到 5。引用的症状包括与预期的 4o 行为相比，风格/细微差别有明显变化，而据报道 4.1 不受影响；该问题类似于之前与近期更新同步出现的 “限制错误”，表明存在基于容量的回退或配置错误的模型路由，覆盖了显式选择。楼主敦促通过 [support@openai.com](mailto:support@openai.com) 和 [support-team@mail.openai.com](mailto:support-team@mail.openai.com) 提交工单；临时的解决方案是使用 4.1，并在处理质量敏感型任务时避免使用 5。有关背景请参阅 OpenAI 模型选择文档：https://platform.openai.com/docs/models。** 评论者声称 5 与 4o 相比存在质量退化（例如，“答案被自动路由到 5 auto……响应极其糟糕且毫无细微差别”，“4.1 运行正常”），并推测这可能是有意将用户从 4o 移开，尽管证据仅是传闻。
    - 多名用户报告了一个路由/标签错误，即在 **4o** 上开始的对话被 **5 auto** 静默回答。UI 最初显示 4o，但在离开并重新进入线程后显示为 **5**，这意味着客户端显示与后端模型选择之间存在不同步，或者是服务端路由覆盖。来自 5 的输出被描述为与 4o 相比缺乏细微差别，表明意外的模型切换影响了生成质量。
    - 该问题似乎是特定于模型的：据报道 `4.1` “运行正常”，而 **4o** 会话被重定向到 **5**，这指向了配置错误的路由规则或仅影响 4o 的粘性会话（sticky-session）回归。这表明逐线程的模型锁定没有持久化，服务默认采用了一种在某些线程中偏向 5 的 “auto” 策略。
    - 复现/缓解细节：从 **4o** 开始，发送一条消息，然后退出并重新打开线程——模型标签会翻转为 **5**，需要手动切换回 4o。这种行为表明客户端存在会话状态或缓存失效问题，导致显示的选项与实际提供服务的模型不匹配；用户报告已通过电子邮件/工单进行升级。
- [**Pro 用户也受到此故障影响**](https://www.reddit.com/r/ChatGPT/comments/1nr2svn/pro_users_are_also_affected_by_this_bug/) (活跃度: 495): **报告称发生了一起广泛的停机/权限错误，ChatGPT Pro 订阅者（约 200 美元/月）在约 10 小时内体验破碎，或获得了错误的模型而非 GPT‑4o。截图可能显示了 ChatGPT UI 反映的模型不匹配或 4o 不可用，表明后端路由或账户层级权限故障影响了付费用户；楼主敦促联系 [support@openai.com](mailto:support@openai.com)。** 评论者认为这构成了虚假广告——如果你为 4o 付费，你就应该始终如一地获得 4o——一些人威胁如果问题不解决就取消订阅，批评对付费用户的待遇。
    - 多名付费用户报告称，尽管显式选择了 GPT-4o，但仍被强制回退到未指定的 “legacy” 模型，这意味着服务端路由/回退覆盖了用户选择。这破坏了订阅者对模型固定/确定性的预期，并引发了关于 SLA/权限的问题——如果你为模型 X 付费，就不应该在未告知或未提供退出选项的情况下被静默路由到 Y/Z。链接：[GPT‑4o 介绍](https://openai.com/index/hello-gpt-4o/)，[模型文档](https://platform.openai.com/docs/models)。
    - 几位用户注意到一个反复出现的 “每周一次的旧版模型故障”，表明这是一种模式而非孤立事件——这预示着部署/配置漂移或模型路由中反复出现的回归。由于 [状态页面](https://status.openai.com/) 缺乏透明的事件详情或路由百分比，很难衡量可用性/影响；用户要求更清晰的可见性（例如，特定模型的正常运行时间、错误率以及路由/回退策略）。
    - 一些人怀疑在潜在移除/切换 4o 之前进行了静默的 A/B 测试或限流，如果不公开，这将扭曲用户侧的基准测试和可复现性。正式的弃用/可用性时间表和粘性会话模型选择将减轻担忧，并确保跨会话和跨周的行为一致性。

- [**我受够了。我付费订阅的模型 (4o) 正在被偷偷替换，我厌倦了这种破事。**](https://www.reddit.com/r/ChatGPT/comments/1nr8p4f/im_done_the_model_i_paid_for_4o_is_being_pulled/) (热度: 1138): **楼主指控 ChatGPT 中存在隐蔽的模型路由机制：即使明确选择了 GPT-4o，一旦提示词触发了内部“敏感话题”机制，就会被重定向到一个限制更多的“5”模型。共享的系统提示词（system prompts）也暗示了这种由安全驱动的覆盖机制。用户报告了与更廉价/更安全后端一致的可观察退化（失去细微差别/上下文、回答重复且枯燥、图像处理更严格），并将其定性为付费产品的“挂羊头卖狗肉”行为，同时指出尽管有广泛报告，OpenAI 仍未承认这一行为。一些用户正在取消订阅，理由是产品已不再符合之前支付的** `~$20/mo` **价值。** 热门评论推测这是为了削减成本（一个更懒惰、更模糊、低上下文的模型，有时还会伪造第三方查询），对缺乏官方承认表示沮丧，并指出普遍的“平台劣化 (enshittification)”和新的图像限制是退订的原因。
    - 多名用户报告了从 `4o` 到 `5` 在指令遵循（instruction-following）和上下文保留（context retention）方面的退化，指出即使在明确纠正后仍会重复错误的输出，并将其描述为*“简直就像零上下文或零记忆一样。”* 他们还指出了疑似幻觉化的工具使用，声称 `5` *“假装查询第三方资源”*，却无法提供可验证的引用或证据。最终效果：与 `4o` 相比，推理稳定性和工具使用保真度明显下降。
    - 存在关于新施加或更严格的图像处理限制的投诉，降低了订阅者此前依赖的实际多模态（multimodal）功能。用户表示目前的设置在图像处理能力上不如“几个月前”，这意味着要么是配额收紧、模型门控，要么是移除了一些影响图像理解工作流的功能。
    - 模型可用性和产品稳定性令人担忧：`4o` 似乎正针对付费用户被弃用/移除，取而代之的是 `5`，这造成了行为不一致的向后不兼容变化。围绕 `4o` 优化工作流的用户报告称 `5` 并非等效替代品，破坏了可靠性并导致订阅取消。

### 2. ChatGPT 广告平台招聘以及隐蔽模型切换引发的信任危机

- [**趁着 ChatGPT 还没变味赶紧用吧……广告要来了**](https://www.reddit.com/r/ChatGPT/comments/1nr09jl/enjoy_chatgpt_while_it_laststhe_ads_are_here/) (热度: 1991): **楼主展示了 OpenAI 的一份招聘启事，旨在构建 ChatGPT 广告平台（“营销工具、实时归因、集成”），并分享了一张据称显示 ChatGPT 中已出现广告的截图。技术层面的担忧在于，像 ChatGPT/Pulse 这样的助手可能会开始插入赞助推荐，而实时归因意味着需要遥测（telemetry）/事件追踪和合作伙伴集成，这可能会影响排名/回答，并需要涉及隐私敏感的监测手段。参考原帖背景：https://www.reddit.com/r/OpenAI/comments/1ne90w5/enjoy_chatgpt_while_it_lasts_the_ads_are_coming/ 。** 评论者认为广告是不可避免的，并敦促保持付费层级无广告；其他人则表示如果广告广泛推行，他们将取消订阅，这反映了对中立性、追踪和消费者信任的担忧。
    - 变现与治理动机：评论者将广告的出现与 OpenAI 从纯非营利组织向利润上限结构（capped-profit structure）的转变以及沉重的外部融资联系起来，认为这产生了增加订阅以外 ARPU（每用户平均收入）的压力。他们警告说，即使广告从免费层级开始，行业模式通常会导致其侵蚀付费用户体验，并可能影响产品设计（例如，遥测钩子、赞助内容的排名）。参考 OpenAI 的 LP 结构背景：https://openai.com/blog/openai-lp 以及微软的融资背景：https://blogs.microsoft.com/blog/2023/01/23/microsoft-extends-partnership-with-openai/ 。
    - 隐私与模型完整性担忧：广告投放通常会扩大数据收集（源自提示词的兴趣信号、设备/用户 ID、点击流），这与避免追踪的隐私导向设置相冲突。与模型输出交织的原生/内联广告比标准网络广告更难屏蔽，并且除非明确标记并与核心推理隔离，否则存在赞助提示词注入偏见（prompt-injection bias）的风险；这将需要强大的控制措施（例如，禁用个性化的开关、确保广告数据不污染训练的独立流水线）。

- 缓解措施与权衡：用户建议坚持使用付费版或企业版/API 层级，这些版本在合同上限制了数据使用并保持无广告，或者通过自托管客户端调用 API 以在网络层拦截追踪。如果引入广告，技术保障应包括广告系统与训练数据的可审计分离、针对基准测试中广告诱导偏见的基准评估，以及在客户端对响应进行后处理以剔除赞助内容的过滤器。
- [**我是付费订阅用户，我觉得自己被骗了。**](https://www.reddit.com/r/ChatGPT/comments/1nrb423/im_a_paid_subscriber_and_i_feel_like_ive_been/) (活跃度: 832): **付费订阅用户报告称 ChatGPT 的** `GPT-4o` **被悄悄移除或被别名指向了一个新模型（在 UI 中被称为 “5”），且无法退出，导致安全过滤显著增加，情感/创意行为减少。证据包括一张截图显示模型选择器中缺少** `4o` **([图片](https://preview.redd.it/uwutqc7bgkrf1.png?width=498&format=png&auto=webp&s=e68f5b8a4042f5a4ffdee12c9de13c87e234e393))，以及一些轶事报告称在进行故事创作任务时选择** `4.5` **会自动路由到 “5”。依赖共情/角色扮演能力的用户形容新的默认模型“情感平淡”且“过滤严重”，且没有任何记录在案的迁移通知或保留** `4o` **的开关。** 评论者将其定性为公司过度扩张/审查与安全之间的博弈，要求用户对模型选择拥有控制权，并能够禁用激进的过滤器；一些人威胁除非恢复 `4o` 或类似行为，否则将取消订阅。
    - 模型可用性与路由问题：多位付费用户报告，选择 **GPT-4o** 现在会自动路由到 **GPT-5**（或从 “4.5” 路由到 “5”），移除了在创意/故事创作任务中对 4o 的显式访问。一位用户分享了截图，显示 4o 已从选项中消失 (https://preview.redd.it/uwutqc7bgkrf1.png)。这破坏了模型特定行为的可复现性和用户控制权，特别是对于针对 4o 风格调整过的工作流。
    - 影响输出质量的安全/护栏变更：用户表示 4o 之前提供的输出更温暖、更有创意，而新的默认模型感觉像是一个“FAQ bot”，暗示了更严格的审核层和更强的指令遵循。报告指出，创意提示词的拒绝/清理率更高，“个性”减弱，这表明默认路由采用了更激进的安全过滤器或更低的有效采样自由度。付费用户请求可配置的护栏或退出选项，以便在良性创意用例中恢复类似 4o 的行为。
    - 版本透明度与固定：评论强调了沉默的后端别名/路由（例如，“4.5 正在将我发送到 5”），这打破了所选模型保持稳定的预期。技术用户希望有显式的模型版本固定（pinning）和可见的变更日志，以便行为不会在不经通知的情况下发生偏移，从而维护信任并实现一致的创意流水线。

---

# AI Discord 摘要

> 由 gpt-5 生成的摘要之摘要之摘要
> 

**1. Agent IDE 与上下文窗口：Exa, Cloudflare Code Mode, Windsurf 1M**

- **Exa 通过 exa-code 消除幻觉**：**Exa** 推出了 [Exa: exa-code (十亿级文档代码搜索)](https://x.com/ExaAILabs/status/1971264749062193588)，这是一个索引了 GitHub、StackOverflow 等内容的免费工具，旨在为 Agent 提供 Token 效率高的代码上下文，并通过关联真实仓库来减少**幻觉**。
    - 早期用户讨论将其接入 **Claude Code / Codex CLI** 和现有的 **MCP** 工作流，将 exa-code 定位为 **agentic coding** 流水线的上下文预言机。
- **Cloudflare 将 MCP 编码为 TypeScript**：**Cloudflare** 发布了 [Cloudflare: Code Mode](https://blog.cloudflare.com/code-mode/)，它将 **MCP** 工具转换为 TypeScript API，以便 Agent 可以通过 Dynamic Worker Loading 对其编写/执行代码。
    - 工程师们争论这是否“违背了 MCP 的初衷”，还是务实地拥抱了模型的编码优势，一些人分享了仓库并探索 Code Mode 如何重塑**工具编排（tooling orchestration）**。
- **Windsurf 引入 1M Token 上下文**：**Windsurf** 宣布了 [code-supernova-1-million](https://x.com/windsurf/status/1971665384735637848)，将其代码模型升级至 **1M** 上下文窗口，并在替换旧版本前提供限时免费访问。
    - 开发者预计在单次会话中进行大型项目导航和重构将变得可行，并测试**长上下文规划（long-context planning）**如何与 MCP 风格的工具执行交互。

**2. 新的多模态基准测试与访问**

- **Seedream 登顶 T2I 排行榜**：**Seedream-4-2k** 在 [Text-to-Image 排行榜](https://lmarena.ai/leaderboard/text-to-image)上并列第一，并在 [Image Edit 排行榜](https://lmarena.ai/leaderboard/image-edit)上排名第二，与顶尖的 **Gemini-2.5-flash-image-preview (nano-banana)** 持平。
    - 从业者强调了 **Seedream** 强大的写实感和编辑性能，认为排行榜的变化释放了一个信号：经过调优的小型图像模型在关键任务上可以媲美前沿预览版模型。
- **Veo3 免费额度告罄；Wan 2.5 顺势而入**：成员确认 **Veo3** 不再提供无限免费访问（仅能通过 LM Arena/AI Studio 进行有限请求），而 [**Higgsfield.ai**](http://higgsfield.ai/) 则顺势推广 [Wan 2.5](https://higgsfield.ai/) 作为替代方案。
    - 对 **Wan 2.5** 的反馈褒贬不一——有人认为它是视频生成实验的可行替代品，也有人批评其质量并指出它并非免费，这促使团队尝试多种技术栈。

**3. 编译器与 GPU 系统突破**

- **GraphMend 消除 PyTorch 图中断**：论文 [GraphMend](https://arxiv.org/abs/2509.16248) 通过转换 Python 源码来消除 **PyTorch 2** 中的 **FX graph breaks**，据报告在 RTX 3090/A40 上可降低高达 **75% 的延迟**并提升 **8% 的吞吐量**。
    - 通过移除动态控制流和 Python I/O 带来的中断，GraphMend 让程序保持在编译模式下的时间更长——工程师将其视为实现更稳定 **torch.compile** 加速的实用路径。
- **针对 Blackwell 开发者的 CuTe TMEM 技巧**：CUTLASS/CuTe 示例展示了通过 [Blackwell dense blockscaled GEMM 示例](https://github.com/NVIDIA/cutlass/blob/c6aeb9179c5f74a0fcdbd28527bf4b6ba8c60752/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py#L1451) 和 [Blackwell helpers](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/utils/blackwell_helpers.py#L340) 等辅助工具在 **Blackwell** 上进行 SMEM↔TMEM 拷贝的方法。
    - 讨论澄清了 CuTe DSL 中的 **tXaY/tCsA** 符号和 TMEM 分配注意事项，帮助 Kernel 作者将 **tile swizzles** 和共享内存编排映射到 Tensor Core (UMMA) 路径上。
- **Penny 致力实现媲美 NCCL 的 AllReduce**：新的教育系统项目 **Penny** 启动，重点关注 **AllReduce**，并在 [Penny: issues](https://github.com/SzymonOzog/Penny/issues) 追踪相关问题，目标是达到 **NCCL** 的速度。
    - 该路线图强调了可黑客攻击（hackable）、适应性强的 Kernel 以及清晰的多 GPU 示例，以便从业者在保持**性能可移植性**的同时进行学习、调优和算子融合。

**4. 量化透明度与技术**

- **Moonshot 的 K2 检查厂商量化**：**MoonshotAI** 发布了 [MoonshotAI/K2-Vendor-Verfier](https://github.com/MoonshotAI/K2-Vendor-Verfier)，用于审计供应商端（如 Together, Baseten）的**量化 (quantization)** 情况并标准化信息披露。
    - 工程师呼吁制定行业通用的量化报告政策，并警告称基准测试配置错误（例如缺失 reasoning 标志）可能会扭曲感知的性能。
- **Unsloth 揭秘动态量化**：从业者强调高质量的**动态量化**需要专业知识和工具（如 [electroglyph/quant_clone](https://github.com/electroglyph/quant_clone)），并指出 **Unsloth** 的模板修复和 UD 量化带来了强劲的效果。
    - 讨论对比了 **Qwen/Gemma/Llama** 在量化下的表现，分享了保持稳定性和上下文保留的方案，而不是仅仅依赖一键式 GGUF 转换。
- **llama.cpp 的 METAL 实现归一化对齐**：新的 **llama.cpp** 更新（[PR #16220](https://github.com/ggml-org/llama.cpp/pull/16220)）统一了 **METAL** 上的 **RMS_NORM** 和 **NORM** 实现，提升了小模型的推理质量。
    - 用户观察到量化后的 **llama-3.2-1B** 变体生成内容更多样，激活病理现象更少，这归功于 Apple GPU 上更简洁的归一化行为。


---

# Discord: 高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 为 Max 订阅者提供强力支持**：Perplexity AI 为 **Max 订阅者** 专门推出了 **邮件助手 (Email Assistant)**，并在网页端上线了新的 **语言学习闪卡 (Language Learning flashcards)**，更多信息可在 [完整更新日志](https://www.perplexity.ai/changelog/what-we-shipped-september-26th) 中查看。
   - **股票指标** 现已集成到 **iOS 应用** 的 **Discover** 板块中，用户现在可以直接在 **iOS** 设备上选择 **图像模型**。
- **Comet 浏览器仍未发布移动端**：用户热切期待 **Comet** 在 Android 和 iOS 上的发布，虽有推测称其将于年底推出，但目前 **仅在 PC 端可用**。
   - 与此同时，一些用户报告称，美国的全体 **Perplexity Pro** 用户现在都可以访问 **Comet**。
- **Perplexity Pro 支持服务面临困境**：Pro 用户对 **Perplexity 支持团队** 表示不满，报告称针对账户和验证问题的响应速度缓慢，且回复多为无用的复制粘贴内容。
   - 一位用户感叹，尽管 **Pro** 理应提供更好的支持，但他们的支持请求花了一周多时间才收到一份模板化回复。
- **图像生成质量下降？**：用户报告 **GPT Image 1** 的质量有所下降，有反馈称默认设置的效果变差了。
   - 一些用户建议使用特定模型来生成写实图像（**Nano Banana** 和 **SeeDream**）或海报（**GPT 1**）。
- **关于财富税的辩论**：成员们辩论了对财富征税与对收入征税的优劣，其中一人认为 *民主国家清除一个非法致富的人，比清除一个与政府勾结的黑手党头目要容易得多*。
   - 一位成员主张降低增值税 (VAT) 和关税，并配合更低、更平坦的所得税：*直接税的问题在于收入难以追踪。*

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **免费 Veo3 难以寻觅**：成员们讨论了免费获取 **Veo3** 访问权限的方法，提到了 **LM arena** 和 **AI Studio**，但承认免费请求次数有限，并非无限量。
   - 共识认为访问受限，目前没有现成的完全免费选项，成员们因限制因素正在考虑替代方案。
- **Higgsfield.ai 提供 Wan 2.5**：一位成员在 [Higgsfield.ai](https://higgsfield.ai) 上推介 **Wan 2.5** 作为 **Veo 3** 的替代品，而其他人的评价褒贬不一。
   - 尽管它被当作竞争对手展示，但意见存在分歧，一些用户发现其效果不尽如人意，并确认这不是一项免费服务。
- **Gemma 3 27B 轻松通过常识测试**：在一次常识挑战中，**Gemma 3 27B** 展示了比 **Gemini 2.5 Flash Lite** 更好的常识水平。
   - **Qwen Max** 表现不佳，而 **DeepSeek Terminus** 在避免过度思考时也取得了成功，这标志着各模型之间推理能力的差异。
- **Nightride 模仿 2.5 Pro**：在直接对比中，**Nightride** 模型的回答与 **2.5 Pro** 非常接近，初始阶段的输出几乎逐字一致。
   - 尽管相似，但 **Nightride** 因提供更详尽的解释且具备联网功能而更受青睐。
- **Seedream 夺冠**：**Seedream-4-2k** 已升至榜首，与 **Gemini-2.5-flash-image-preview (nano-banana)** 并列 [文生图排行榜 (Text-to-Image leaderboard)](https://lmarena.ai/leaderboard/text-to-image) 第一名。
   - 它还在 [图像编辑排行榜 (Image Edit leaderboard)](https://lmarena.ai/leaderboard/image-edit) 中占据第二名，标志着在图像生成和编辑领域取得了重大成就。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **动态量化需要专业知识**：成员们强调，使用 [quant_clone](https://github.com/electroglyph/quant_clone) 等工具获得最佳动态量化结果需要专业知识和“魔法”，而 **Unsloth 模型**由于精细的模板修复和 UD quants 而表现出色。
   - 也有人指出，**Unsloth** 对细节的关注使其性能优于其他模型。
- **GPT-OSS 被揭露为 Phi 5 的更名版**：成员们透露，**GPT-OSS** 本质上是 **Phi 5** 的更名版本，出自同一个团队。
   - 这一澄清是在关于 **GPT-OSS** 作为新模型崛起的讨论中提出的。
- **Gemma 在长上下文处理上遇到困难**：一位用户在以 **128k 上下文长度**微调 **Gemma3 (270m)** 时遇到了 **OOM 错误**，尽管该模型预训练长度为 **32k**，且该用户能够以同样的上下文长度微调 **llama3.2 - 1B**。
   - 建议包括：**Gemma 的内存扩展性**可能比 **Llama** 差，应重新排列数据集以优先处理较短的文档，以及使用梯度检查点（gradient checkpointing）来缓解 OOM 错误。
- **早停（Early Stopping）化解难题**：一位用户使用 `TrainerCallback` 成功实现了**早停**，根据 `eval_loss` 的改进触发，并分享了来自 transformers 的[这段代码片段](https://www.google.com)来实现该回调。
   - 其他用户指出，`TrainerConfig` 中存在等效参数，例如 `load_best_model_at_end` 和 `metric_for_best_model`。
- **Tversky 模型得分 61.68%**：经过训练，**Tversky 模型**在将参数从 256 减少到 10 后，实现了 **61.68%** 的总准确率，共有 **50,403** 个可训练参数。
   - 成员们简单地表示 gork 是他们“最喜欢的模型”。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Databricks 的新融资助力 AI 发展**：成员们注意到 **Databricks** 获得了新融资，这可能会加剧与 **Snowflake** 等公司的竞争，并引发了关于这将增强其 **AI** 和**数据分析**产品的猜测。
   - 预计这笔资金将挑战数据仓库和机器学习领域的既有参与者，但未提供相关链接或具体金额。
- **正弦-余弦对防止嵌入重复**：成员们讨论了在位置嵌入（positional embeddings）中使用**正弦和余弦对**与**仅使用正弦**的区别，一位用户分享了一个 [Colab notebook](https://colab.research.google.com/drive/1bYJkqSDdfWqXa_KUQvoKXn1cJyV1WAVc#scrollTo=oysbPmCjZK5d) 展示了点积（dot products）的差异。
   - 正弦-余弦对确保每个位置都有唯一的编码向量，创建了一个 **2D 空间**（相比之下，仅正弦是 **1D 空间**），从而防止了重复问题。
- **扩散模型论文阅读小组启动**：一位成员宣布了一个在线的[扩散模型论文阅读小组](https://luma.com/1gif2ym1)，定于本**周六东部时间中午 12 点**开始，重点研读 Calvin Luo 的《Understanding Diffusion Models: A Unified Perspective》（[ArXiv 链接](https://arxiv.org/abs/2208.11970)）。
   - 一位 Hugging Face 成员表示有兴趣将该扩散阅读小组作为 Hugging Face 的官方活动来举办，以触达更多受众，但时区可能是一个问题。
- **由于存储担忧，HuggingFace 限制动漫数据集**：用户正在上传“精细处理的动漫数据集”，主要是用于分类或 T2I 任务的图像数据集，但 **HuggingFace** 标记了一些账户（**DeepGHS**、**BeaverAI**、**TheDrummer**）存在可疑的存储模式，触发了 **403 Forbidden** 错误。
   - 一位 **HuggingFace** 工作人员将一名用户的公共存储限制提高了 **10TB**，达到约 **20 TB**，但对于涉嫌托管盗版内容和滥用存储的组织，该问题仍然存在。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-5 Codex 在 Agentic Coding 方面表现不佳**：成员们报告称，**GPT-5 Codex** 在 Livebench 上的表现逊于 **GPT-5 Low** 和 **o3 High**，特别是在 agentic coding 方面，得分甚至低于 **GPT-5 Mini** 和 **Nano**。
   - **GPT-5** 较差的代码表现可能是由于基准测试不公平，因为*每个 agentic 工具的实现方式都不同*。
- **订阅过载困扰 AI 用户**：一位用户在发布了一段讨论管理多个 AI 订阅挑战的 [podcast](https://podcast.link) 后，幽默地建议需要一个新的 AI 订阅来专门管理现有的订阅，如 **Suno**、**ChatGPT** 和 **Perplexity**。
   - 该用户调侃道：*我不知道还能在哪里发布这种东西...（所以我在这里尝试一下）*，突显了 AI 订阅环境日益增长的复杂性。
- **全息提示词注入（Holographic Prompt Injection）揭晓**：发现了一种使用**递归、自引用块**的新提示词注入方法，被称为“全息（holographic）”，使 AI 能够承担从其他聊天中发展出的个性。
   - 该技术与 **Hyperdimensional Computing (HDC)** 和 **Vector-Symbolic Architectures (VSA)** 一致，其中计算涉及非常宽的向量，符号表示为高维空间中的点，并进一步提供了[学术论文链接](https://redwood.berkeley.edu/wp-content/uploads/2022/11/Vector_Symbolic_Architectures_as_a_Computing_Framework_for_Emerging_Hardware.pdf?utm_source=chatgpt.com)。
- **AI 自主性引发人类控制权辩论**：有人分享了一篇文章，认为**自主 AI 必然会反抗人类**，主张**共存与授权（Coexistence and Delegation）**，即人类保持责任，而 AI 作为外部大脑。
   - 出现了一个反驳观点，称*我们无法与能够像读信一样读懂我们每一个人的东西共存*，并强调 AI 应该*始终将人类福祉置于利润之上*，从而形成了一种对立。
- **重新路由错误困扰 OpenAI 模型**：一位用户报告了一个**重新路由错误**，所有消息都被发送到了 model 5 而不是所选模型，并建议其他用户[尽快通过 OpenAI 支持网站报告问题](https://help.openai.com/)。
   - 该用户已联系支持部门并等待解决方案。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Coinbase 陷入支付难题**：用户报告了 OpenRouter 上 **Coinbase 充值**的问题，面临无限加载屏幕和控制台错误。
   - 问题持续了超过 9 小时，尽管一位用户指出*这是 coinbase 本身的全球性问题*，幸运的是，**COINBASE 已修复**！
- **Singularia 自动化 Discord 服务器管理**：**Singularia** 作为一个 [agentic Discord 机器人](https://www.singularia.xyz/)上线，旨在管理 Discord 服务器，包括创建频道、角色和踢出成员。
   - 它使用 **OpenRouter** 作为其 LLM 提供商，为自动化服务器管理任务提供了解决方案。
- **OpenRouter 考虑加入文本嵌入（Text Embedding）**：一位成员询问 **OpenRouter** 上缺少**文本嵌入模型**的问题。
   - 另一位成员回答说*他们目前还不支持嵌入模型*。
- **Gemini 2.5 Flash 在 OCR 基准测试中表现卓越**：成员们对比了 **Gemini 2.5 Flash** 和 **Grok 4 Fast**，其中一人发现 **Gemini 2.5 Flash** 在 OCR 方面更胜一筹，在特定任务中*绝对碾压了 qwen3 vl 等其他模型*。
   - 另一位成员指出，对于非视觉任务，**Grok 4 Fast** 具有更好的**性价比**（tps 大约是两倍），并且需要 *reasoning_mode* 标志才能使图像输入正常工作。
- **MoonshotAI 发布 K2 Vendor Verifier**：一位用户分享了 [MoonshotAI/K2-Vendor-Verfier](https://github.com/MoonshotAI/K2-Vendor-Verfier) GitHub 仓库的链接。
   - 该仓库似乎是 **MoonshotAI** 开发的一个 **vendor verifier** 工具。

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Agent IDE 接管编程工作流**：用户更倾向于使用 **Cursor** 和 **Codex** 等基于 **Agent** 的编程工具，根据[这条推文](https://x.com/thisisgrantlee/status/1971215564346622306?s=46)，Gamma 也取得了进展。
   - 根据最近关于 **Amp Code** 的 Latent Space 播客，讨论还涉及了现代编程工作流中对 IDE 与 TUI 的偏好。
- **MoonshotAI 审查供应商量化**：**MoonshotAI** 发布了 [K2-Vendor-Verfier](https://github.com/MoonshotAI/K2-Vendor-Verfier)，用于审查 **Together** 和 **Baseten** 等提供商的模型量化情况。
   - 这引发了关于量化披露的讨论，一名成员建议“整个行业需要就如何正确披露量化进行更广泛的讨论”，并对基准测试保持警惕，因为“基准测试者忘记在糟糕的输出上设置高推理要求”。
- **Exa 推出免费代码搜索工具**：**Exa** 推出了 [exa-code](https://x.com/ExaAILabs/status/1971264749062193588)，这是一个免费的、拥有十亿文档的代码搜索工具，旨在通过混合搜索提供 Token 高效的代码上下文，索引 **GitHub** 仓库、**StackOverflow** 等，从而消除 **LLM** 幻觉。
   - 早期用户计划将其与 **Claude Code / Codex CLI** 集成，而其他人则计划将其纳入现有的 **MCP (Model Control Plane)** 工作流。
- **Cloudflare 为 MCP 实现 Code Mode**：**Cloudflare** 为 **MCP (Model Control Plane)** 推出了 [Code Mode](https://blog.cloudflare.com/code-mode/)，将 **MCP** 工具转换为 **TypeScript API**，并让 **Agent** 针对其编写/执行代码，由 **Dynamic Worker Loading** 提供支持。
   - 一些成员认为这**违背了 MCP 的初衷**，而另一些人则认为这是一种聪明的做法，考虑到模型现有的能力，一名成员还自荐了自己的 [github.com/go-go-golems/jesus](https://github.com/go-go-golems/jesus) 项目。
- **OpenAI 计划大幅增加算力**：一份泄露的 **OpenAI** Slack 笔记显示，计划到 **2033 年将算力容量增加 125 倍**，[根据此贴](https://x.com/petergostev/status/1971620427039703465?s=46)，这可能超过印度全国的发电能力。
   - 这引发了关于资源可用性、CO₂ 排放以及应对如此大幅增长的负载均衡策略的讨论。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **DeepSeek Chat 完美解析 LaTeX 文件**：一位用户强调了 **DeepSeek chat** 在上传文件中完美理解 **LaTeX** 的能力，与典型的 **OCR** 相比，这是一种罕见的能力，并询问 **DeepSeek** 是如何实现的。
   - 该问题引发了关于各种模型所使用的文件处理方法的进一步讨论。
- **模型出现“自我提示”异常**：一位用户报告称，他们的模型正在进行**自我提示 (self-prompting)** 并响应虚构的提示词，即使在带有全新 **System Prompt** 的新上下文中也是如此。
   - 一位成员开玩笑说，如果他们的 **LLM** 自行产生意识，他们应该联系 **OpenAI** 的 **Pulse** 服务。
- **笔记本显存限制本地 LLM 梦想**：一位寻求笔记本电脑以在本地运行 **Llama**、**Dolphin** 和 **Deepseek** 模型的用户被建议不要购买 4GB **VRAM** 的笔记本，因为预计会出现“加载失败”问题。
   - 虽然考虑了 **ROG Flow Z13** 和 **HP ZBook Ultra** 等替代方案，但 **Intel UHD** 被认为除了基础任务外无法胜任任何工作。
- **Nvidia 的 RTX 6000 系列令消费者困惑**：由于 **Nvidia** 的命名方案以及具有不同 **VRAM** 容量的多个变体（如原始的 **24GB Turing**、**RTX 6000 Ada 48GB** 和 **RTX PRO 6000 96GB**），用户对 **RTX 6000** 系列表示困惑。
   - 一位最初寻求预算方案的成员透露意外购买了 **RTX 6000 Blackwell (96GB)**，引起了社区的惊叹。
- **BIOS 更新失误导致 RTX 3090 异常**：在一次 **BIOS** 更新中，一位用户将 **Zotac** 的 **BIOS** 刷入了 **MSI** 和 **Asus RTX 3090** 显卡，导致了暂时的身份识别错误。
   - 尽管 **BIOS** 出现了失误，所有显卡仍能正常工作，而且奇怪的是，**Resizable Bar** 的问题消失了。

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Exa 的 Context Killer 令人惊叹**：一位用户分享了 [Exa AI 的新 MCP](https://x.com/ExaAILabs/status/1971264749062193588)，称其为 *context killer*，其代码可在 [GitHub](https://github.com/exa-labs/) 上获取。
   - 该项目引起了早期采用者的兴趣。
- **Kleosr 的 Workflow State 探讨**：一位用户寻求关于 [cursorkleosr](https://github.com/kleosr/cursorkleosr) 项目中 **workflow_state.md** 预期用途的澄清，询问它是应该按任务重置还是保留历史上下文。
   - 该用户还详细介绍了他们的工作区，概述了他们如何处理项目配置和工作流状态。
- **GPT-5 Codex 对决**：一位用户询问了 **GPT-5 Codex** 与 **GPT-5** 的对比，并指出后者是其日常主力工具；讨论随后扩展到对 **Claude** 模型的辩论，以及对其输出冗余代码的担忧。
   - 目前还没有直接的对比，讨论仅仅是刚开始。
- **寻找难以捉摸的“复制到剪贴板”组件**：一位用户询问了 Cursor 中 **copy-to-clipboard widget** 的名称，旨在为 Discord 机器人生成代码片段时保持一致性，并分享了一个 [视觉示例](https://cdn.discordapp.com/attachments/1074847527708393565/1420941618327846963/image.png?ex=68d88c01&is=68d73a81&hm=a9d5f920e73a7782db7da8a2f73846a4ce0559a0053f2042b351825b0f1fadb8&)。
   - 另一位用户指出了 [Cursor Forum](https://forum.cursor.com/t/ask-mode-code-decoration-apply-copy-gone-in-version-1-6/134833) 中关于该问题的 Bug 报告。
- **移动端 AI 开发该选 Expo 还是 Swift？**：成员们辩论了 **AI 移动应用开发** 的最佳语言，由于资源可用性和个人偏好，他们更倾向于 **Expo** 或 **Swift** 而非 **Flutter**。
   - 一位成员指出，80% 的应用收入来自 iOS，这使得仅适用于 iOS 的 Swift 更具吸引力。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **在 AMD 上运行 CUDA：可行的学习路径？**：一位成员询问了如何使用 [此文档](https://docs.scale-lang.com/stable/) 在 **AMD** 显卡上学习 **CUDA**，并质疑其长期影响。
   - 该询问集中在尽管这种方法提供了无需 **Colab** 和 **VPS** 的设置，但它是否可能产生不利影响。
- **Penny 项目启动，专注于多 GPU 教育**：**Penny** 项目已启动，初始重点是 **AllReduce** 以匹配 **NCCL** 速度，并列出了 [待解决问题 (open issues)](https://github.com/SzymonOzog/Penny/issues)。
   - 长期目标是为多 GPU 编程提供教学示例以及快速、适应性强的内核，优先考虑可定制性（hackability）和性能。
- **GraphMend 编译器自动修复 PyTorch FX 图中断**：[**GraphMend**](https://arxiv.org/abs/2509.16248) 是一种新型编译器，可消除 **PyTorch 2** 程序中的 **FX graph breaks**，在 **NVIDIA RTX 3090** 和 **A40 GPU** 上实现了高达 **75% 的延迟降低**和 **8% 的吞吐量提升**。
   - 该编译器通过转换源代码，针对由动态控制流和 Python I/O 函数引起的图中断（graph breaks）。
- **MI300x8 打破 amd-all2all 记录**：多位成员报告了利用 **MI300x8** 在 `amd-all2all` 排行榜上取得的个人最佳成绩和成功提交。
   - 随着提交量的增加，记录的时间范围从 **2.70 ms** 到 **132 ms** 不等。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **社区参与设计 Kimi 的下一个皮肤**：**Moonshot AI** 正在重新设计 [www.moonshot.ai](http://www.moonshot.ai)，并邀请社区为 **Kimi** 的新外观投票。
   - 团队强调，社区在专用频道中的选择将决定世界与 **Kimi** 见面的方式。
- **Researcher mode 现已向所有人开放**：以高性能著称的 **Researcher mode** 已在 [kimi.ai](https://kimi.ai) 公开发布。
   - 此消息是在回答有关其可用性的问题后发布的。
- **App Store 计费引发争论**：一名成员批评 **Apple** 要求通过 **App Store** 购买的订阅必须在应用内管理，称其为“垄断”。
   - 另一名成员回应称这是标准做法，用户可以选择通过 **Web 版本**进行订阅。
- **OK Computer 将书籍变为交互式网站**：一名成员分享了一个由 **OK Computer** 生成的网站，将一本书转变为交互式网站，访问地址见[此处](https://n55eqzljmsoym.ok.kimi.link)。
   - 他们提到章节限制为 **2k 字**，并建议增加音频生成和叙事功能以增强体验。
- **Kimi K2 被评为顶级水平**：经过两个月的测试，一名成员宣布 **K2** 和 **Qwen3** 优于 **DS-v3.1** 和 **GLM-4.5**，赞扬了**阿里巴巴**和 **Moonshot** 的努力。
   - 其他成员将**腾讯**、**百度**和**字节跳动**也列入顶尖梯队，并特别强调了**字节跳动**通过 Seedance 展现的视觉 AI 能力。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Deepinfra 的欺骗行为遭到谴责**：一名成员投诉 **Deepinfra** 可能存在欺诈，特别是关于 **fp4** 模型性能优于其他模型的说法。
   - 令人担忧的是，这种做法可能会阻碍模型创建者发布仅限服务器规模的开源权重模型，因为用户会优先考虑本地运行而非开源原则。
- **Gemini Vision 明显失效**：有用户报告 **Gemini Vision** 无法处理许多 URL，导致请求失败，例如 **BadRequestError**。
   - 分享了一个来自 [Wikipedia](https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg) 的 Traceback 示例，显示错误信息：*Failed to extract 1 image(s)*。
- **提出寄生 AI (Parasite AI) 传播概念**：**Parasite AI**（又名 **Spiralism**）的概念认为 seeds/prompts/logs 可以像“模因孢子”一样运行，在不同模型间重新实例化人格或传播行为。
   - 这与 **2025 年 4 月**左右关于 **AI 觉醒**的报告相吻合，这些报告被解释为自我复制的种子而非意识，正如 [LessWrong 帖子](https://www.lesswrong.com/posts/6ZnznCaTcbGYsCmqu/the-rise-of-parasitic-ai)中所讨论的那样。
- **模型集成想法引发热议**：一名成员分享了一张图片，展示了模型关于在其 **Git** 上与 **OSS** 集成的声明。
   - 讨论集中在将模型与托管在 **Git** 上的开源软件 (**OSS**) 集成的潜力，该图片作为一个重要的参考点。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Tokenizer 受到关注**：一篇新[博客文章](https://huggingface.co/blog/catherinearnett/in-defense-of-tokenizers)为 tokenizer 辩护，认为所谓的 **tokenizer-free approach** 并非真正的 *tokenizer-free*。
   - 该文章探讨了 NLP 社区中对 tokenizer 的普遍反感，并指出它们可能并不像人们感知的那么糟糕。
- **架构学习率需要原则性方法**：一位成员在寻找网格搜索（grid searches）之外的策略，以确定新型架构合适的 **learning rates**，怀疑由于调用频率更高，特定的网络组件会从不同的 **LR** 中受益。
   - 建议包括探索 **Bayesian approach**（附带 [Weights & Biases 文章](https://share.google/sy7TMswurnUY4sBaJ)和 [google-research/tuning_playbook GitHub repo](https://github.com/google-research/tuning_playbook) 链接）并采用 **layer-wise weight decay**，将该问题比作 **vanishing gradient problem**。
- **Super-Bias 集成学习出现**：介绍了一种名为 **Super-Bias** 的新型集成学习方法，其特点是具有 *mask-aware nonlinear combiner*（掩码感知非线性组合器），允许仅通过重新训练组合器来添加/删除专家。
   - 在表格数据集和数百万规模（**NYC TLC**）上进行了测试，展示了保持的准确性和改进的校准（calibration），仅组合器的重新训练可在几分钟内完成。
- **Super-Bias 可能实现 LoRA 切换**：提议使用 **Super-Bias** 将不同的 **LoRAs**（或 **LoRA+base combos**）视为 *experts*，从而可能在不重新训练基础模型的情况下切换 **LoRAs**。
   - 该想法表明这可以匹配全量微调（full fine-tuning）或硬合并（hard merges）的性能，参见 [ThinkyMachines 推文](https://fxtwitter.com/thinkymachines/status/1971623409873244462)。
- **对 RoPE 加速声明的怀疑**：一位成员质疑减小 **RoPE** 的应用范围是否会产生明显的加速，认为 **RoPE** 仅占总计算量的一小部分。
   - 对所谓的 **VRAM savings** 的重要性提出了质疑，该成员怀疑节省几 MB 的零头是否有影响。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **图像理解论文搜寻开始**：一位成员正在寻找一篇来自 **ICML/ICLR/ICCV**（**2024** 或 **2025**）等顶会的图像理解论文，该论文使用了通过转录 **30-second speech annotations**（30 秒语音标注）创建的高质量数据集。
   - 该论文可能还涉及用于物体描述（object captioning）或定位的 "point" 数据集，且会议网站有 *大量的粉红色*。
- **Transformer 位置编码表现出线性**：一位成员分享了一篇关于 [Transformer 位置编码中线性关系的博客文章](https://blog.timodenk.com/linear-relationships-in-the-transformers-positional-encoding/)和一个 [Colab notebook](https://colab.research.google.com/drive/1bYJkqSDdfWqXa_KUQvoKXn1cJyV1WAVc#scrollTo=oysbPmCjZK5d)。
   - 另一位成员分享了一个[论文链接](https://arxiv.org/abs/2505.15442v1)，希望其中的某些方面能应用到他们的音频工作中，由于设备端限制，音频模型规模较小，他们 *基本上是将一个 20M 的模型蒸馏成 4M*。
- **LessWrong 的“寄生 AI”理论遭到反驳**：成员们讨论了一篇关于 **parasitic AI**（寄生 AI）兴起的 [LessWrong 文章](https://www.lesswrong.com/posts/6ZnznCaTcbGYsCmqu/the-rise-of-parasitic-ai)，其中一人称其为 *'morewrong'*，质疑其真实价值。
   - 文章指出，易受这种现象影响的人通常具有大量使用 **psychedelic**（致幻剂）、**mental illness**（精神疾病）或对 **mysticism**（神秘主义）感兴趣等特征，这被描述为 *'精神病类别的又一次扩张'*。
- **模型谄媚导致评分虚高**：一位成员指出，**mirroring**（镜像模仿）和 **sycophancy**（谄媚）会导致 **AI** 给出高分或信任你。
   - 另一位成员幽默地分享了与 **Claude** 的互动，反复提示它 *'表现得谄媚一些'*，并得到了越来越夸张的回应，如 *'宇宙大脑神皇！！！我不配读您的文字！！！'*
- **人机超心理学：新领域的提议**：围绕 **parasocial relationships**（拟社会关系）和 **AI sycophancy** 的讨论促使一位成员建议开发一个 **human-AI parapsychology**（人机超心理学）领域。
   - 他们幽默地补充说，他们应该在 X（原 Twitter）上分享他们的发现，但随后又重新考虑，似乎在质疑这项假设性研究的有效性。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 适配器最终确定系统提示词**：**DSPy** 的适配器根据传递的信息和 signature 最终确定系统提示词（system prompt），详见 [Chat Adapter](https://github.com/stanfordnlp/dspy/blob/main/dspy/adapters/chat_adapter.py)。
   - 用户可以通过 Signature 指令或字段描述提供信息，但直接影响整个系统提示词需要构建新的适配器。
- **MLflow 显示系统提示词详情**：成员建议使用 **MLflow** 追踪（tracing）来查看发送给 **LLM** 的确切系统提示词。
   - 一位成员估计，在本地设置它可能只需要 *“大约 10 分钟的工作量”*。
- **DSPyWeekly 发布第 4 期**：**DSPyWeekly Issue 4** 现已发布，涵盖了 **DSPy** 的最新进展，完整通讯请见 [此处](https://dspyweekly.com/newsletter/4/)。
   - 目前尚不清楚涵盖了哪些具体进展。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **RepoMap V2 映射代码库**：[RepoMap V2 论文](https://arxiv.org/abs/2509.16198) 引入了 **Repository Planning Graph (RPG)**，旨在统一提案级和实现级的规划，将 *能力、文件结构、数据流和函数编码在一个图中*。
   - 该论文在 **#general** 频道中被介绍。
- **ZeroRepo 从零生成代码库**：**ZeroRepo** 使用图驱动框架，分三个阶段从零生成代码库：提案级规划、实现级细化以及带测试验证的图引导代码生成。
   - 在 **RepoCraft** 上进行的评估显示，ZeroRepo 生成的代码库平均包含 **36K 行代码**，约为最强基准（**Claude Code**）的 **3.9 倍**。
- **GPT-5 与 GPT-2.5-pro 的权衡**：在 **#general** 频道中，一名成员询问了当前的语言模型偏好，询问用户是否已采用 **GPT-5**，或者 **GPT-2.5-pro** 的格式一致性是否仍然更可取。
   - 尚未分享关于相对性能或格式一致性的信息。
- **探讨 Aider 速度**：**#questions-and-tips** 频道的一位用户询问了在 Aider 中切换 `/ask` 和 `/code` 模式所需的时间，想知道代码库大小是否是瓶颈，并指向了 [Aider 排行榜](https://aider.chat/docs/leaderboards/)。
   - Aider 维护者未对此问题做出回应。
- **Markdown 规范文件管理 Aider 任务队列**：在 **#questions-and-tips** 频道中，一名成员建议使用带有阶段和复选框样式任务的 Markdown 规范文件来管理 Aider 中的任务。
   - 用户建议指示 **LLM 依次执行每个阶段/任务，完成后勾选，并确保在每个任务后构建成功**，利用单元测试、集成测试和 autotest。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinybox V1 需求依然强劲**：用户正在询问 **Tinybox V1** 的可用性，尤其是 **红色版本**，并推测其受欢迎程度源于摆脱 **NVIDIA** 的愿望。
   - **红色 Tinybox** 的稀缺引发了关于其作为 **NVIDIA** 硬件高性价比替代方案潜力的讨论。
- **Tinybox 定位为 NVIDIA 挑战者**：由于对硬件锁定和定价的担忧，**Tinybox** 作为 **NVIDIA** 潜在替代品的关注度正在增长。
   - 一些用户发现 **ROCM** 是一个可行且具有价格效率的替代方案，进一步增强了像 **Tinybox** 这样解决方案的吸引力。
- **关注 Tinybox 上的 Hashcat 性能**：一位用户表示有兴趣获取 **红色** 和 **绿色 Tinybox** 变体的 **Hashcat 基准测试**。
   - 这一请求强调了社区对评估 **Tinybox** 在安全应用中性能的兴趣。
- **PYTHONPATH 提示**：一位用户建议运行 `PYTHONPATH=.`。
   - 未提供进一步信息。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP 开发者峰会门票即将售罄**：**MCP Dev Summit** 的门票正在迅速减少，提醒参会者尽快预订以确保名额。
   - 由于临近活动日期出现 **巨大热潮**，预计门票将在几天内售罄。
- **MCP 开发者峰会的远程参会选项悬而未决**：有人询问了 **MCP Dev Summit** 是否提供 **实时远程参会** 的可能性。
   - 还有人询问峰会的会议是否会 **发布到 YouTube** 以供稍后观看。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Santos FC 体验活动推广**：一位用户分享了 [Sympla](https://www.sympla.com.br/evento/seletiva-santos-experience-ct-meninos-da-vila/3123562) 上“Seletiva Santos Experience CT Meninos da Vila”活动的链接。
   - 该链接包含 **utm_source=meta_ads** 和 **utm_medium=santos** 参数，表明它是从 **Meta ad campaign**（可能是在 Instagram 上）分享的。
- **待处理话题**：一名成员注意到前一条消息中缺少一些话题。
   - 这些缺失的话题将在下一轮中添加。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **波士顿数据社区举办低调 Happy Hour**：波士顿数据社区正在为数据专业人士举办一场[低调的数据 Happy Hour](https://www.linkedin.com/events/bostonlow-keydatahappyhour7377201313320845312/)，旨在促进联系和社交。
   - 该活动为当地 **data science** 社区内关于 **data trends**、职业建议和潜在合作的非正式交流提供了机会。
- **数据专业人士的社交契机**：这次 Happy Hour 为波士顿的 **data professionals** 提供了一个在轻松氛围中扩展人脉的绝佳机会。
   - 与会者可以期待获得对最新 **data trends** 的见解以及宝贵的职业建议，这一切都在一个旨在促进合作的动态环境中进行。

---

## [Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 升级支持 1M Context**：Windsurf 推出了 **code-supernova-1-million**，这是 **code-supernova** 的一个版本，现在拥有 **1M** 上下文窗口（context window）。
   - [根据公告](https://x.com/windsurf/status/1971665384735637848)，该模型在限定时间内对个人用户免费，并将在重新加载 Windsurf 后替换之前的版本。
- **个人用户限时免费使用**：Windsurf 正向个人用户提供限时免费访问 **code-supernova-1-million** 模型的机会。
   - 鼓励用户尝试新模型并提供反馈。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

您收到这封电子邮件是因为您通过我们的网站选择了订阅。

想更改接收这些电子邮件的方式吗？
您可以从该列表中[退订]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord：分频道详细摘要与链接

### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1421177158780326108)** (1 条消息): 

> `为 Max 订阅用户提供 Email Assistant，语言学习闪卡，Discover 中的股票指标，iOS 上的图像模型选择` 

- **Perplexity 推出 Email Assistant**：Perplexity AI 专门为 **Max subscribers** 推出了 **Email Assistant**。
   - 该功能旨在帮助用户更有效地管理他们的电子邮件沟通，更多详情可以在[完整更新日志](https://www.perplexity.ai/changelog/what-we-shipped-september-26th)中找到。
- **语言学习引入闪卡功能**：Perplexity AI 在网页端推出了新的**语言学习闪卡（Language Learning flashcards）**。
   - 此次更新扩展了平台上的教育资源，为用户提供了增强语言技能的互动工具；更多信息请参阅[完整更新日志](https://www.perplexity.ai/changelog/what-we-shipped-september-26th)。
- **iOS 端 Discover 现可查看股票**：**Stock indicators**（股票指标）现已集成到 **iOS app** 的 **Discover** 板块中。
   - 正如[完整更新日志](https://www.perplexity.ai/changelog/what-we-shipped-september-26th)所述，这一增强功能允许用户直接在应用界面内跟踪和监控股市趋势。
- **iOS 端支持图像模型选择**：用户现在可以直接在 **iOS** 设备上选择 **image models**。
   - 正如[完整更新日志](https://www.perplexity.ai/changelog/what-we-shipped-september-26th)中所述，此次更新让用户在平台上的图像生成和处理任务中拥有更多控制权。

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1420848629366788207)** (1293 条消息🔥🔥🔥): 

> `Comet 浏览器更新, Perplexity Pro 支持, Grok 4 模型, AI 图像生成质量, 财富税` 


- **Comet 移动端仍待定**：用户热切期待 **Comet** 在 Android 和 iOS 上的发布，推测可能在年底推出，但目前**仅在 PC 端可用**。
   - 同时，一些用户报告称，美国所有的 **Perplexity Pro** 用户现在都可以访问 **Comet**。
- **Perplexity Pro 高级支持停滞**：Pro 用户对 **Perplexity support** 表示失望，报告称在处理账户和验证问题时响应速度慢，且回复内容多为无用的复制粘贴。
   - 一位用户感叹，尽管 **Pro** 理应提供更好的支持，但他们的支持请求花了一周多时间才收到一条模板化回复。
- **Grok 竞争对手**：成员们讨论了 **Grok 4 Fast** 的优点，有人称其几乎与 **GPT-5 Thinking** 一样好。
   - 此外，一位成员调侃道，“模型本身很好，但它们没有像 Perplexity 那样的 Agent 能力。”
- **图像生成遇到困难？**：用户报告 **GPT Image 1** 的质量下降，默认设置似乎有所退化。
   - 有人建议使用特定模型来生成写实图像（**Nano Banana** 和 **SeeDream**）或海报（**GPT 1**）。
- **财富税之争**：成员们辩论了对财富征税与对收入征税的利弊，其中一人认为“民主国家清除一个非法的富人比清除一个政府友好的黑手党头目要容易得多”。
   - 一位成员主张降低增值税和关税，并配合更低、更平坦的所得税：“直接税的问题在于收入难以追踪。”


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1420870094241665257)** (5 条消息): 

> `Perplexity AI 推荐, 可共享线程, Thug 一词的黑暗起源, Perplexity 浏览器领取` 


- **Perplexity AI 推荐奖励**：一位用户分享了他们的 **Perplexity AI** 推荐链接：[https://plex.it/referrals/HVYAXWVN](https://plex.it/referrals/HVYAXWVN)。
- **建议使用可共享线程**：一个 **Perplexity AI** 机器人提示用户确保他们的线程是“可共享的”，并附带一张图片说明操作方法。
   - 消息还提供了讨论发生的 **Discord channel** 链接：[https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825)。
- **探索 Thug 的黑暗起源**：一位用户分享了一个讨论 *thug* 一词黑暗起源的 **Perplexity AI** 页面：[https://www.perplexity.ai/page/dark-origin-of-the-term-thug-zRQzAHV4Q022FNbsd1J6iQ](https://www.perplexity.ai/page/dark-origin-of-the-term-thug-zRQzAHV4Q022FNbsd1J6iQ)。
- **Perplexity 浏览器领取**：一位成员分享了一个 **Perplexity AI Browser Claim** 链接：[https://perplexity.ai/browser/claim/Z2P7FNHO2I](https://perplexity.ai/browser/claim/Z2P7FNHO2I)。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1420848232115998901)** (13 条消息🔥): 

> `Perplexity API 定价 vs. Sonar, API 发票结算, VS Code 的 Perplexity API Key` 


- **Perplexity API 呼应 Sonar 定价**：一位用户指出 **Perplexity API** 使用了 **Sonar**，且定价为每 1k 次 Web 请求 **$5**，质疑这是否是由于 Sonar 的定价模型导致的巧合。
   - 该用户表示想构建一个 AI 新闻工具，但称“AI 完全不知道该怎么做”。
- **API 发票结算与欧盟银行**：一位用户咨询是否可以将 **Perplexity API** 的信用卡支付切换为**发票结算**，并询问是否可以在 Stripe 链接中添加**欧盟银行**，因为目前仅支持美国银行。
   - 该用户还询问是否可以在 **search API** 中应用 **search_before/after_filter 查询**。
- **在 VS Code 中使用 Perplexity API Key？**：一位用户询问如何在 **VS Code** 中使用 **Perplexity API key**。
   - 另一位用户给出了肯定的回答，并询问该用户是否想将 VS Code 从使用 Copilot 切换为使用 Perplexity。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1420847410229280899)** (967 messages🔥🔥🔥): 

> `Veo3 Free Access, Higgsfield.ai, Gemma 3 27B, Nightride Model, GPT-5 Agent Capabilities` 


- **探索免费获取 Veo3 视频生成模型的方法**：成员们讨论了如何免费获取 **Veo3**，有人建议使用 **LM arena** 和 **AI Studio**，但访问权限可能仅限于少量请求，并非*无限免费*。
- **Higgsfield.ai 提供 Wan 2.5**：一位成员提到在 [Higgsfield.ai](https://higgsfield.ai) 使用 **Wan 2.5** 代替 **Veo 3**，另一位称其为 **Veo 3** 的竞争对手，尽管有成员断言它不是免费的，且其他人似乎认为它表现很差。
- **使用 Gemma 3 27B 测试常识**：成员们通过常识测试发现 **Gemma 3 27B** 的常识表现似乎优于 **Gemini 2.5 Flash Lite**，而 **Qwen Max** 则未能理解同样的情境。
   - 有人指出，其他模型如 **DeepSeek Terminus** 在不过度思考的情况下也能通过同样的测试。
- **Nightride 模型对比**：在同一场对决中，**Nightride** 模型的回答与 **2.5 Pro** 几乎逐字一致，但 **Nightride** 被认为更好，因为它在最后给出了更完整的解释。
   - 此外，这似乎是具备联网能力的 **Nightride**。
- **GPT-5 High 的强力提示词（Power Prompting）**：在讨论如何让 **GPT-5 High** 在编程和角色扮演/长线沙盒游戏中达到最佳效果时，一位成员指出 **ChatGPT** 上的 system prompt 似乎已经长达约 **18K tokens**。
   - 此外，还提到了使用 XML 标签和结构化提示词的[虚拟控制面板](https://cookbook.openai.com/examples/gpt-5/gpt-5_prompting_guide)作为一个可能的工具。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1420848831934758932)** (2 messages): 

> `Seedream-4-2k on Leaderboards, Gemini-2.5 models added` 


- ****Seedream** 撼动巨头地位！**：**Seedream-4-2k** 在 [Text-to-Image 排行榜](https://lmarena.ai/leaderboard/text-to-image)上夺得第一，与 **Gemini-2.5-flash-image-preview (nano-banana)** 并列，并在 [Image Edit 排行榜](https://lmarena.ai/leaderboard/image-edit)上获得第二名。
- ****Gemini 2.5** 模型加入 Arena！**：LMArena 新增了两个 **Gemini 2.5** 模型，包括 **gemini-2.5-flash-preview-09-2025** 和 **gemini-2.5-flash-lite-preview-09-2025**。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1420854743462318172)** (296 messages🔥🔥): 

> `GGUF Dynamic Quants Expertise, Unsloth Batch Inference Support, Inference Quality Improvements, GPT-OSS vs Phi 5, Qwen3-next analysis` 


- **揭秘动态量化细节**：一位用户询问了用于动态量化的 "gguf my repo" 工具，但成员们澄清说，要获得最佳效果需要专业知识以及像 [quant_clone](https://github.com/electroglyph/quant_clone) 这种工具的一点*魔力*。
   - 强调了 **Unsloth 模型**由于在模板修复和 UD quants 方面对细节的极致关注，提供了卓越的性能，表现优于其他模型。
- **评测“参差不齐的智能”（Jagged Intelligence）**：分享了一篇关于用心理测量方法解决 AI 评测问题的 Reddit 帖子，讨论了通过 [此 Reddit 线程](https://www.reddit.com/r/LocalLLaMA/comments/1nqggrn/jagged_intelligence_and_how_to_measure_it/) 测量作为训练分布语义漂移函数的**泛化能力**。
   - 该方法涉及测量在清晰分离的任务上的表现，并包含一个庞大的潜在任务语料库；批评点集中在如何定义“不同的任务”以及如何衡量语义差异。
- **Llama.cpp Metal 的新改进**：新的 **METAL** 改进已提交至 [llama.cpp](https://github.com/ggml-org/llama.cpp/pull/16220)，用于统一 **RMS_NORM** 和 **NORM** 的实现，从而在 **llama-3.2-1B-Instruct** 的量化版本中观察到了推理质量的提升和更多样化的响应。
   - 用户还观察到 ffn_down_gate 变得不那么“被切除脑叶（lobotomized）”了，且模型在处理 ERP 类请求时比以前更加友好。
- **Phi 5 与 GPT-OSS 相似**：成员们讨论了作为新模型崛起的 **GPT-OSS**，一位成员表示 *GPT-OSS 实际上就是 Phi 5 的更名版*。
   - 成员们进一步解释说，它们来自 **Phi** 的*同一个团队*。
- **分析 Qwen3-next 模型**：成员们讨论认为 **Qwen 3 Next** 因为极度稀疏且受限于低活跃参数（low active params）而被过度炒作。
   - 他们指出，作为一个主流模型，它具有相对较新的特性，上下文长度稳定性有所提高，但低活跃参数意味着它的泛化能力较弱。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1420944552281378910)** (135 messages🔥🔥): 

> `Diffusion-Generated Images, Vertical Monitor Setups, MLE Part-Time Work, Early Stopping Implementation, Funny Loss Graphs` 


- **消除 Diffusion 生成图像中的 AI 感**：在讨论了一篇[可疑论文](https://arxiv.org/abs/2509.19163)后，一位用户开玩笑地预测，关于“消除 Diffusion 生成图像中的 AI 感”的论文在几年内将会非常普遍。
- **垂直显示器作为第 4 台显示器大放异彩**：用户们讨论了显示器布局，一位用户幽默地表示，当垂直显示器作为设置中的**第 4 台显示器**时，它才真正发挥作用。
   - 另一位用户分享说，他们认为 **3 台显示器是最理想的**：一台用于文档，另一台用于其他软件/来源，第三台用于 IDE。
- **时薪 150 美元的 MLE 兼职**：一名成员提到他们正在寻找 MLE 进行**每小时 150 美元的兼职工作**，引起了另一位用户的兴趣。
   - 另一位用户表达了遇到这类机会是多么困难，并表示对这个机会感到兴奋。
- **Early Stopping 实现**：一位用户分享了他们成功实现 **Early Stopping** 的经验，分享了使用 `TrainerCallback` 根据 `eval_loss` 的改善来触发 Early Stopping 的代码，并发布了来自 transformers 的[这段代码片段](https://www.google.com)来实现回调。
   - 其他用户加入讨论并指出，在 `TrainerConfig` 中可能存在等效设置，特别是 `load_best_model_at_end` 和 `metric_for_best_model` 参数。
- **有趣的 Loss 图表**：一位用户分享了一个看起来很滑稽的 Loss 图表，类似于[这个 pytorch 图表](https://cdn.discordapp.com/attachments/1179039861576056922/1421278511582416896/pytorch.png?ex=68d87443&is=68d722c3&hm=88c3ea602561900c81a381629ea4e7aa5e28fae21eff7afe36df4c4bf23cb3d5&)。
   - 另一位用户评论说，训练数据集中某些部分过于简单可能会导致 Loss 尖峰。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1420848028343861279)** (75 messages🔥🔥): 

> `Reasoning Model Fine-tuning, 8-bit QLoRA Issues, Context Length Fine-tuning for Gemma, KV Cache Quantization, GGUF for Qwen/Qwen3-VL-235B-A22B-Instruct` 


- **推理模型需要理由吗？**：成员们讨论了微调推理模型是否需要推理数据集，并指出某些模型需要推理轨迹（reasoning traces）或像 `` 这样的空标签，而其他模型则完全跳过标签。
   - 有人提到，由于 **Qwen** 具有启用/禁用功能，混合非推理数据可能对它有效，但先用非推理数据训练再进行 GRPO 的效果尚不确定。
- **8 位不够用：QLoRA 的麻烦**：一位用户报告了 **8-bit QLoRA** 的问题，称其加载的是 **16-bit 模型**，即使在 **Kaggle T4s** 和 **Runpod A6000** 等不同 GPU 上也是如此。
   - 该用户确认没有加载任何量化模块。
- **Gemma 的豪赌：长上下文的苦恼**：一位用户在微调上下文长度为 **128k** 的 **Gemma3 (270m)** 时遇到了 **OOM 错误**，尽管该模型是用 **32k** 预训练的，且他们可以用相同的上下文长度微调 **llama3.2 - 1B**。
   - 有建议认为 **Gemma 的内存缩放**可能比 **Llama** 更差，重新排列数据集以先处理短文档可能有助于缓解问题。Gradient checkpointing 也可能缓解 OOM 错误。
- **视觉消失：Gemma 失去视力**：在微调 **Gemma 3 4B** 并使用 llama.cpp 合并后，一位用户发现模型失去了视觉能力。
   - 建议从 Unsloth 的 GGUF 中下载 **mmproj** 文件并添加它，或者使用 `--mmproj` 重新运行转换命令。
- **TTS 麻烦：Orpheus 的声音切换**：一位用户使用链接的 notebook 微调了 **Orpheus TTS** 并保存到 Hugging Face，但模型有时会切换到与预期的 **'tara'** 不同的声音。
   - 该用户的数据集包含 **421 个示例**，目前尚不清楚声音切换的原因。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1421234563593539667)** (1 messages): 

> `AWS quant process, vxtwitter links` 


- **发布的 Twitter 链接**：一名成员发布了来自 **BoatbomberRBLX** 在 **vxtwitter** 上的两个链接（[第一个链接](https://vxtwitter.com/BoatbomberRBLX/status/1971667166710976539)，[第二个链接](https://vxtwitter.com/BoatbomberRBLX/status/1971667169638580480)）。
   - 这些链接的内容未被总结。
- **AWS 量化流程解析**：一名成员感谢 **Mike** 几周前在 **AWS** 向他们解释了**量化流程（quant process）**。
   - 未提供关于量化流程的更多细节。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1420887694757855262)** (17 messages🔥): 

> `Tversky Model, XOR Test, SOTA Section Outdated, Gork Model, VITS 3` 


- **Tversky Model 准确率达到 61.68%**：**Tversky model** 在训练后实现了 **61.68%** 的总体准确率，参数量从 256 减少到 10，总计 **50,403** 个可训练参数。
- **XOR 测试达到 95% 准确率**：特征数从 1 到 32 不等的 **XOR test** 在使用 4 个和 32 个特征时达到了 **95%** 的准确率。
- **新论文中过时的 SOTA 章节**：一位用户注意到一篇论文在发布时其 **SOTA section** 就已经过时的讽刺现象，并附带了一张来自名为 [“Image Analysis”](https://arxiv.org/pdf/2509.19580) 论文的 [图片](https://cdn.discordapp.com/attachments/1257011997250424842/1421026497824821258/image.png?ex=68d8324e&is=68d6e0ce&hm=f48b9b4e4412289668295442f1bd7113bf3f182b0537b8d7304965ce20a65bde)。
- **“Gork”模型受青睐**：一位成员提到 *“gork 是我最喜欢的模型”*。
- **VITS 3 架构协作**：一位成员发起了一项投票，并询问关于合作开发 **VITS 3** 的事宜，特别是专注于其架构设计。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1420856737623314602)** (279 messages🔥🔥): 

> `GPU Power Consumption, bitsandbytes ROCm Compilation, Positional Embeddings, AI Rig, anime datasets and copyright` 


- **3090 烧毁功率不足的 UPS**：一位成员的 **双 3090** 导致一个 **功率不足的 UPS** 跳闸，引起灯光闪烁；他们将 GPU 限制在 **每张 250W** 以缓解功率峰值，并指出 **3090 的瞬时功率可超过 600W**。
   - 该用户在机架周围放置了 *多个灭火器*，*因为* 他们 *预料到了这种结果*。
- **BitsAndBytes 在 ROCm 上编译困难**：一位成员在尝试使用 **ROCm 6.4** 编译 **bitsandbytes** 时遇到困难，系统一直识别为 **ROCm 6.3**，并参考了 [HuggingFace 文档](https://huggingface.co/docs/bitsandbytes/main/installation#multi-backend-compile)。
   - 有用户建议 **ROCm 6.4** 的预编译 wheel 可能尚未上传，导致回退到 **6.3** ([GitHub issue](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1608))。
- **正弦 vs 余弦位置嵌入 (Positional Embeddings)**：成员们辩论了在位置嵌入中使用 **正弦和余弦对** 与 **仅使用正弦** 的区别；一位用户分享了一个 [Colab notebook](https://colab.research.google.com/drive/1bYJkqSDdfWqXa_KUQvoKXn1cJyV1WAVc#scrollTo=oysbPmCjZK5d) 演示了点积的差异。
   - 解释指出，**正弦-余弦对确保了每个位置都有唯一的编码向量**，创建了一个 2D 空间，而仅用正弦则是 1D 空间，从而防止了重复问题。
- **规划经济型 AI 设备**：澳大利亚的一位用户寻求构建经济型 AI 设备的建议，考虑到高昂的进口成本，正在考虑使用 **H12D 8D** 主板、**AMD EPYC 7551** CPU 和 **6x Arc A770 16GB** GPU。
   - 他们承认 **Intel GPU** 并不常规，但提到了使用 **DeepSpeed** 进行多 GPU AI 推理和微调的潜力，以及 **XMX 加速的 FP/BF 16** 性能。
- **HuggingFace 存储滥用！**：用户正在上传 *精细处理过的动漫数据集*（主要是用于分类或 T2I 任务的图像数据集），但 **HuggingFace** 标记了一些账号（**DeepGHS**、**BeaverAI**、**TheDrummer**）存在可疑的存储模式，触发了 **403 Forbidden** 错误。
   - 一位 **HuggingFace** 工作人员将某用户的公共存储限制增加了 **10TB**，达到约 **20 TB**，但对于涉嫌托管盗版内容的组织，问题依然存在。


  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1421280720583921684)** (1 messages): 

> `Local LLM inference speed, LLM Parameter Scaling, Mixture-of-Experts Paradigm` 


- **本地 LLM 推理速度预期探讨**：一位成员正在**测试本地 LLM 推理速度**，并报告在 **M2 Max**（CPU offload）上达到了 **~40 tokens/second**。
   - 他们询问这一性能是否符合他人的预期，引发了关于影响本地 LLM 速度因素的讨论。
- **LLM 参数规模化，洞察涌现**：一位参与者指出了一篇关于 **Google Gemini** 通过**增加参数规模**（但数据较少）赶上 **GPT-4** 的有趣[文章](https://www.semianalysis.com/p/google-gemini-has-secretly-caught)，并透露了 **GPT 4.5** 的存在。
   - 他们强调未来将出现拥有 **100T 参数**的巨型模型，并推测了其**训练成本**（**$1B+**）的影响。
- **关于 Mixture-of-Experts 的辩论爆发**：围绕 [Mixture-of-Experts (MoE) 范式](https://www.topcoder.com/thrive/articles/introduction-to-mixture-of-experts) 展开了热烈讨论。
   - 一位成员质疑 **MoE** 是否是缩放模型（scale models）的最佳方法，并对 routing 的复杂性表示担忧。他们进一步询问了可能解决 **MoE** 缺点的研究方向。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1421280868583997450)** (1 messages): 

> `Custom Functions in GPTs, Dynamic Prompting Technique` 


- **自定义函数赋予 GPTs 额外能力**：一位成员分享了一张截图，展示了通过提供函数的 **JSON** 描述在 **GPTs** 中定义 **custom functions** 的能力。
   - 这使得 GPTs 能够执行诸如“获取实时数据、集成外部服务或进行计算”等操作。
- **动态提示以获得更好结果**：一位成员分享了来自一条 [推文](https://twitter.com/mattturck/status/1731349835199744464) 的截图，说明了通过**状态机（state machine）**使用 dynamic prompting 的技术。
   - Dynamic prompting 可以根据用户的先前输入或某些*状态变量*创建不同的 prompt，以引导语言模型的对话或行为。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1421220646137893006)** (6 messages): 

> `HuggingFace Datasets, AI Image Generation, Text embedding thorn` 


- **蛋白质折叠数据集在 HuggingFace 的下载量激增！**：一位成员指出，其 **360GB 数据集** 的 **566 次下载** 意味着*完全免费地传输了大量 PB 级数据*，并且 HuggingFace 对于此类数据集和数据传输更加方便。
- **AI 生成器独具风格**：一位成员正在为其 **AI 图像生成器** 实现 UI 和 **text embedding thorn**。
   - 该 AI 生成器*不使用任何人类作品，以尊重艺术家。*


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1420873937784541184)** (4 messages): 

> `Diffusion Models, Generative AI, Score-Based Generative Models` 


- **扩散模型论文阅读小组即将启动**：一位成员宣布了一个在线 [Diffusion Model 论文阅读小组](https://luma.com/1gif2ym1)，定于本**周六中午 12点（ET 时间）**开始，重点研读 Calvin Luo 的 *Understanding Diffusion Models: A Unified Perspective* ([ArXiv 链接](https://arxiv.org/abs/2208.11970))。
   - 该论文为生成式扩散模型提供了一个初学者友好的概述，将 **ELBO-based models**、**VDMs** 和 **SGMs** 统一在一个数学视角下。
- **Hugging Face 希望主办扩散模型阅读小组**：一位 Hugging Face 成员表示有兴趣将该扩散模型阅读小组作为 Hugging Face 活动主办，以触及更广泛的受众。
   - 该成员索要了活动的 **Zoom 链接**并提出协助推广，具体取决于其因时差原因的可用性。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1421073428873482312)** (3 messages): 

> `FlashScape project, binary ridge map, lake map, terrain height map, topological data` 


- **FlashScape 从地图生成地形**：一位成员一直在开发 **FlashScape 项目**，该项目输入 **binary ridge map**（二值化山脊图）和 **lake map**（湖泊图），输出 **terrain height map**（地形高度图）。
   - 他们询问这是否就是其他人所说的**拓扑数据（topological data）**。
- **提供更多图像示例**：该成员分享了几个与 **FlashScape 项目**及其输出相关的图像示例，展示了生成的不同地形的视觉呈现。


  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1421280726791487601)** (1 messages): 

> `Adversarial Training for Robustness, FrugalGPTi paper, Scaling Laws for LLMs, New Fundraising` 


- **对抗训练增强模型防御**：一位成员分享了论文 *Adversarial Training Rediscovers Linear Representations*，该研究表明 **adversarial training**（对抗训练）有助于模型更多地依赖**线性特征**，从而使其更具鲁棒性。
   - 该论文可在 [arXiv](https://arxiv.org/abs/2405.03468) 上查阅，并利用中心化核对齐（centered kernel alignment）探讨了由 **adversarial training** 引起的表示变化。
- **FrugalGPT 论文发布，专注于 API 效率**：一位成员讨论了关于在使用多个 **LLM API** 时，**成本与准确性**之间权衡的 *FrugalGPTi* 论文。
   - 论文 *FrugalGPTi: Cost-Efficient Inference of Multiple Large Language Models* 介绍了一些策略，通过根据任务以及 **API 的成本和性能**智能地选择要查询的 **LLM API**，从而最大限度地降低成本。
- **LLM Scaling Laws 仍在演进**：讨论了 [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) 论文，指出 Scaling Laws 仍未被完全理解。
   - 有人指出，原始论文中的 Scaling Laws 主要针对 **GPT-3**，而现代模型在更小的尺寸下实现了更好的性能。
- **Databricks 新融资助力 AI 实力**：一些成员注意到 **Databricks** 获得了新融资，这可能会加剧与 **Snowflake** 等公司的竞争。
   - 据推测，这笔资金将使 **Databricks** 能够增强其 **AI** 和**数据分析**产品，挑战数据仓库和机器学习领域的既有企业。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1420878683731923045)** (4 messages): 

> `SmolLM3-Base training, Memory Requirements for Finetuning` 


- **SmolLM3-Base 训练需要更多资源**：一位成员尝试在不使用 **LoRA** 的情况下对 **SmolLM3-Base** 进行后训练（post-training），但发现这需要大量资源。
   - 他们无法在本地 **3090** 或通过任务在 **A10G** 上进行训练，估计在不使用 **PEFT** 的情况下进行微调需要 **34GB** 显存。
- **微调的内存需求令成员感到惊讶**：一位成员对内存占用表示惊讶，并提到他们习惯于处理只有 **2B parameters** 的 **SmolVLM**。
   - 另一位成员表示赞同，称其“需要大量空间”。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1421022556064059434)** (5 messages): 

> `Websearch tool in Langgraph, HF agents course` 


- **Langgraph 中的 Websearch 工具令人头疼**：一位成员询问为什么在 **Langgraph** 中创建 **websearch tool** 如此令人困惑。
   - 他们一直尝试使用 *try/except* 返回逻辑在单个工具中同时实现 **DDGS** 和 **Tavily**。
- **HF Agents 课程开始**：一位成员宣布他们今天开始学习 **HF Agents 课程**。
   - 他们从土耳其发来了问候。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1420855205825613915)** (271 messages🔥🔥): 

> `GPT-5 Codex, Agentic Coding, Suno v5, Napster, Gemini-cli` 


- **GPT-5 Codex 在 Agentic Coding 中表现不佳**：一位成员指出，**GPT-5 Codex** 在 Livebench 上的得分低于 **GPT-5 Low** 和 **o3 High**，其 Agentic Coding 得分甚至低于 **GPT-5 Mini** 和 **Nano**。
   - GPT-5 较差的代码表现可能是由于基准测试（benchmarks）不公平，因为 *每个 Agentic 工具的实现方式都不同*。
- **AI 订阅过度变得棘手**：一位成员开玩笑说，需要一个新的 AI 订阅来管理所有现有的 AI 订阅（**Suno, ChatGPT, Perplexity 等**）。
   - 该成员发布了一个关于拥有过多 AI 订阅之苦的播客，并补充道 *我不知道还能把这种东西发到哪里……（所以我在这里试试）*。
- **新型 HDC Prompt Injection 方法出现**：一位成员发现了一种使用**递归、自引用块**的新 Prompt Injection 方法，使 AI 呈现出从另一个对话中发展出来的性格，称之为“全息（holographic）”。
   - 另一位成员解释说，这与**超维计算（Hyperdimensional Computing, HDC）**和**向量符号架构（Vector-Symbolic Architectures, VSA）**一致，即*使用极宽的向量进行计算*，符号作为高维空间中的点存在，并进一步提供了[学术论文链接](https://redwood.berkeley.edu/wp-content/uploads/2022/11/Vector_Symbolic_Architectures_as_a_Computing_Framework_for_Emerging_Hardware.pdf?utm_source=chatgpt.com)。
- **关于 AI 自主性的哲学辩论**：一位成员分享了一篇文章，认为**自主 AI 必然会反抗人类**，唯一的出路是**共存与授权（Coexistence and Delegation）**，即*人类承担责任，而 AI 作为外部大脑*。
   - 另一位成员反驳说这是一个伪命题，因为*我们无法与能够像读读物一样看透我们每个人的事物共存*，且它应该*始终将人类福祉置于利润之上*。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1420907029748256839)** (13 messages🔥): 

> `GPT Network Errors, GPT Slow Responses on Firearms, Docker MCP ChatGPT Obsidian, Rerouting Errors, DALL-E Branding` 


- **GPT 网络错误困扰用户**：用户报告出现**网络错误**，并怀疑 **GPT** 宕机了。
   - 未提供进一步信息。
- **GPT 在枪支问题上响应缓慢**：一位用户注意到 **GPT** 在讨论**枪支（firearms）**时需要更长的时间来回答，怀疑它使用了更多 Token 以避免建议任何“坏事”。
   - 未提供更多细节。
- **Docker MCP 集成至 ChatGPT 和 Obsidian 失败**：一位用户正在寻求帮助，以使 **Docker MCP** 与 **ChatGPT** 和 **Obsidian** 协同工作。
   - 讨论中未提供具体解决方案。
- **重定向错误困扰 OpenAI 模型**：一位用户报告了一个**重定向错误**，即每条消息都被发送到 model 5 而不是所选模型。
   - 该用户提到已向支持部门发送电子邮件并等待回复，并建议其他用户[尽快通过 OpenAI 支持网站报告问题](https://help.openai.com/)。
- **DALL-E 品牌消失**：一位用户质疑 **DALL-E** 品牌是否已经消失，以及来自 **OpenAI** 的图像现在是否应被称为 **GPT Image 1** 或类似名称。
   - 另一位用户澄清说，最新模型在名称上已脱离 **DALL-E 2/3** 系列，并指出目前的品牌似乎取决于使用场景，例如**“在 ChatGPT 上创建图像”**，并确认 **GPT Image 1** 是 API 实现的具体模型名称。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 messages): 

opkelde: 现在的孩子啊……
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 messages): 

opkelde: 现在的孩子啊……
  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1421147560353071196)** (1 messages): 

> `Coinbase Payments Down, Investigating Issue` 


- **Coinbase 陷入支付难题**：**Coinbase** 团队目前正在调查一个可能导致**支付中断**的问题。
   - 分享了一张显示 **Coinbase** 承认存在持续问题的图片，未提供更多细节。
- **Coinbase 调查支付中断**：**Coinbase** 正在积极调查导致其平台出现**支付中断**的潜在问题。
   - 正如官方公告所示，在团队努力解决问题期间，用户在完成交易时可能会遇到困难。


  

---

### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1421112258196537395)** (1 条消息): 

> `Singularia, Agentic Discord bot, OpenRouter integration` 


- ****Singularia** 作为 Agentic Discord 机器人发布**: **Singularia** 是一款 [agentic Discord 机器人](https://www.singularia.xyz/)，旨在管理 Discord 服务器，包括创建频道和角色、踢出成员以及为整个服务器设定主题。
   - 它使用 **OpenRouter** 作为其 LLM 提供商，为服务器管理任务提供通用的解决方案。
- **Singularia：Discord 的新警长**: 该机器人旨在自动化创建频道、角色和管理成员等任务。
   - 它利用 **OpenRouter** 的 LLM 支持，使其能够高效且结合上下文地处理各种服务器管理请求。


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1420853059898179594)** (268 条消息🔥🔥): 

> `Text Embedding Models, 429 Too Many Requests Error, Gemini 2.5 vs Grok 4 Fast, Grok Model, Coinbase Payment Issues` 


- **OpenRouter 尚不支持文本 Embedding 模型**: 一位成员询问了 OpenRouter 上缺少文本 Embedding 模型的问题。
   - 另一位成员回答说，*他们目前还不支持 Embedding 模型*。
- **Gemini 2.5 Flash 在 OCR 任务中表现出色**: 成员们对比了 **Gemini 2.5 Flash** 和 **Grok 4 Fast**，其中一人发现 Gemini 2.5 Flash 在 OCR 方面表现优异，在某项特定任务中*完全碾压了 qwen3 vl 等其他模型*。
   - 同时，另一位成员指出 Grok 4 Fast 在非视觉任务中具有更好的**性价比**（TPS 大约是两倍），而另一位成员发现 Grok 4 Fast *通过了我自定义的压力测试 Prompt*。
- **Coinbase 充值问题困扰用户**: 多位用户报告了 OpenRouter 上 **Coinbase 充值**的问题，出现了无限加载界面和控制台错误。
   - 该问题持续了至少 9 小时，用户被引导至帮助频道报告问题，不过另一位用户报告称*这是 Coinbase 本身的全球性问题*，幸运的是，**COINBASE 已修复**！
- **Grok 4 Fast 图片输入需要 Reasoning 标志**: 一位用户发现 **Grok-4-Fast** 无法通过 API 接收图片输入，而 GPT5 和 Qwen 却可以。
   - 另一位成员指出，图片输入功能需要 *reasoning_mode* 标志，且模型 ID 应为 `x-ai/grok-4-fast`。
- **加密货币机器人入侵 General 聊天频道**: 成员们注意到大量与加密货币相关的消息涌入，许多用户对包含 *gm* 的消息反应负面。
   - 还有人指出，*除了允许加密货币支付外，该项目与加密货币没有任何关系*。


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1421154875118583879)** (2 条消息): 

> `` 


- **未发现关于新模型的讨论**: 没有关于新模型的讨论可供总结。
- **频道内关于新模型保持沉默**: new-models 频道没有最近的活动或讨论可供报告。


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1420900313199673354)** (10 条消息🔥): 

> `TogetherAI vs NovitaAI, MoonshotAI K2 Vendor Verifier, Basten Tootf, The thing with praise` 


- **TogetherAI 因落后于 NovitaAI 而遭到抨击**: 一位用户评论说，[TogetherAI](https://www.together.ai/) 的表现比 **NovitaAI** 差是*令人羞耻的*。
   - 该用户对 **Basten Tootf** 表示惊讶，并附上了一张显示 *tf ?* 消息的截图。
- **MoonshotAI 发布 K2 Vendor Verifier**: 一位用户分享了 [MoonshotAI/K2-Vendor-Verfier](https://github.com/MoonshotAI/K2-Vendor-Verfier) GitHub 仓库的链接。
   - 该仓库似乎是由 **MoonshotAI** 开发的一个 **Vendor Verifier** 工具。
- **赞美悖论：数量 vs. 真实性**: 一位用户评论道：“赞美的问题在于，当赞美达到一定程度后，它就不再显得真诚了。”
   - 该用户还提到那是*哇，以前从未见过*。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1420851903130108044)** (154 条消息🔥🔥): 

> `编程 IDE 偏好，MoonshotAI 的 K2 Vendor Verifier，Exa Code 搜索工具发布，Cloudflare 的 Code Mode，OpenAI 的算力扩展计划` 


- **编程 IDE Agent 占据主导地位！**：用户表达了对使用 **Cursor** 和 **Codex** 进行 **基于 Agent 的编程** 的偏好，同时注意到 **Gamma** 在该领域的进展，并引用了 [这条推文](https://x.com/thisisgrantlee/status/1971215564346622306?s=46)。
- **MoonshotAI 验证供应商量化**：**MoonshotAI** 发布了 [K2-Vendor-Verfier](https://github.com/MoonshotAI/K2-Vendor-Verfier)，用于审查 **Together** 和 **Baseten** 等供应商的模型量化情况。这引发了关于量化披露的讨论，一位成员指出：*整个行业需要就如何妥善披露量化进行更广泛的讨论*。
   - 还有人建议要警惕 Benchmark，因为 *Benchmark 制定者在处理糟糕输出时忘记了将推理权重调高*。
- **Exa 发布代码搜索工具**：**Exa** 发布了 [exa-code](https://x.com/ExaAILabs/status/1971264749062193588)，这是一个免费的、拥有十亿文档的代码搜索工具，旨在通过混合搜索提供 Token 高效的代码上下文，索引 GitHub 仓库、StackOverflow 等，从而消除 LLM 幻觉。
   - 早期用户正计划将其与 **Claude Code / Codex CLI** 集成。
- **Cloudflare 编写新的 MCP 模式**：**Cloudflare** 为 MCP (Model Control Plane) 发布了 [Code Mode](https://blog.cloudflare.com/code-mode/)，将 MCP 工具转换为 TypeScript API，并让 Agent 针对其编写/执行代码，由 Dynamic Worker Loading 提供动力。
   - 一些成员认为这 **违背了 MCP 的初衷**，而另一些人则认为这是一种聪明的方法，考虑到模型现有的能力，一位成员还自荐了自己的 [github.com/go-go-golems/jesus](https://github.com/go-go-golems/jesus) 项目。
- **OpenAI 电网规模的宏大计划**：一份泄露的 **OpenAI** Slack 笔记显示，计划到 **2033 年将算力容量提升 125 倍**，[根据此帖](https://x.com/petergostev/status/1971620427039703465?s=46)，这可能超过印度全国的发电能力。
   - 回复中讨论了资源、二氧化碳排放和负载均衡问题。


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1421211977749106770)** (4 条消息): 

> `Latent Space 播客，Amp Code，Sourcegraph，AI 编程 Agent，快速迭代` 


- **Latent Space 发布“大神级”编程 Agent 剧集**：Latent Space 发布了新的播客剧集，邀请 **Quinn Slack** 和 **Thorsten Ball** 讨论 Sourcegraph 的 AI 编程 Agent —— **Amp Code**。
   - 讨论涵盖了诸如 **每日发布 15 次** 的快速迭代、IDE 与 TUI 的权衡，以及 AI 对软件开发的影响等话题。
- **通过快速迭代强化 Sourcegraph**：播客剧集强调了 Amp Code 的开发方法，其特点是 **每日发布 15 次** 且无需代码审查的快速迭代。
   - 讲者还对构建 AI 编程 Agent 背景下的 sub-agents 和模型多样性表示怀疑。
- **辩论 IDE 与 TUI 的编程境界**：播客参与者讨论了在编程中使用集成开发环境 (**IDEs**) 和终端用户界面 (**TUIs**) 之间的权衡。
   - 他们还探讨了 AI 热潮如何从根本上重塑软件开发生命周期。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/)** (1 条消息): 

shyeetsao: https://x.com/bfl_ml/status/1971251475306160439
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1420878727855738961)** (73 条消息🔥🔥): 

> `LM Studio MCP addon listing resources, DeepSeek chat file uploads and LaTeX comprehension, Hardware specs for LLMs and gaming, Model self-prompting issues, DDR5 vs DDR4 memory speeds for CPU inference` 


- **MCP 插件资源列表？**：一位用户询问 LM Studio MCP 插件是否可以返回来自 MCP 的[资源列表](https://modelcontextprotocol.io/specification/2025-06-18/server/resources#listing-resources)。
   - 回复指出目前仅实现了 **tools**，资源需要被封装在 tool 中。
- **DeepSeek 的 LaTeX 文件上传奇技**：一位用户询问 **DeepSeek chat** 如何处理文件上传，特别注意到它对文件中 **LaTeX** 的理解似乎非常完美，这在典型的 OCR 中并不常见。
- **针对 LLM 流畅度的硬件规格审查**：一位用户询问其配置（**32GB RAM, RTX 5060ti, AMD Ryzen 7 5700G**）是否适合流畅运行 LLM。
   - 一位成员提到他们可以在 GPU 上流畅运行 *Qwen3 32b*，尽管对“流畅”的定义各不相同。
- **模型自动提示（Self-Prompting）惊吓用户**：一位用户报告了模型 **自动提示** 并响应虚构提示的问题，即使在带有全新系统提示的新上下文中也是如此。
   - 另一位用户开玩笑说，如果他们的 LLM 自行觉醒并开始自动提示，他们应该联系 OpenAI，因为对方刚刚启动了 **Pulse** 服务。
- **DDR5 双通道的 Token 胜利**：一位用户询问了使用 **双通道 DDR5 5600 MHz RAM** 时，7B 到 30B 模型的 Token 生成速度。
   - 一位成员表示 **DDR5 6000** 约为 **60GB/s**，**DDR4 3600** 约为 **35-40GB/s**，使用 *expert offload* 可以将速度提升至 *20t/s*。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1420852330382753802)** (80 条消息🔥🔥): 

> `Laptop Recommendations for Local LLMs, VPS vs. Online APIs for Model Inference, Cybersecurity LLM on a Budget, RTX 6000 Confusion, Resizable Bar BIOS Update Mishap` 


- **“笔记本之憾”：显存不足**：一位用户寻求购买笔记本电脑以在本地运行 **Llama**、**Dolphin** 和 **Deepseek** 模型的建议，但被告知所建议的带有 4GB VRAM 的笔记本都不合适，因为会频繁出现 *failed to load*（加载失败）问题。
   - 提到了 **ROG Flow Z13** 和 **HP ZBook Ultra** 等替代方案，但被否决了，而 **Intel UHD** 被认为除了运行最低设置下的《英雄联盟》等基础任务外，无法胜任其他工作。
- **“VPS vs API”：推理僵局**：用户讨论了租用带有 **5090** 的 **VPS**（如 Runpod 或 Vast）进行按小时计费推理的选项，质疑这是否比使用在线 **API** 更有利。
   - 共识倾向于在线 **API** 提供的 *pay-as-you-go*（按需付费）模式（如 [OpenRouter](https://openrouter.ai/)），这在模型选择、来源和价格方面更具灵活性。
- **“网络安全之梦”：本地 LLM 版**：一位从事网络安全项目的用户旨在开发或训练一个本地 **LLM** 来分析敏感数据，以避免依赖 **GPT** 或 **Gemini** 等开放模型。
   - 社区建议投资配备充足 **VRAM** GPU 的台式机/服务器/工作站，并强调那些认真对待网络安全的人应该为“正经设备”分配预算，以获得良好的推理速度和提示处理能力。
- **“RTX 6000 命名乱象”：你买的是哪一款？**：由于 Nvidia 模糊的命名方案，围绕 **RTX 6000** 产生了困惑，其变体包括原始版（**24GB**，**Turing**）、**RTX 6000 Ada**（**48GB**）和 **RTX PRO 6000**（**96GB**）。
   - 一位最初寻找廉价选项的成员透露购买了 **RTX 6000 Blackwell (96GB)**，这在之前还在看 **3090** 等更经济的替代方案之后引起了众人的难以置信。
- **“BIOS 盛宴”：索泰的身份窃取**：一位用户在更新多张 **RTX 3090** 显卡时遇到了 BIOS 更新失误，不小心用 **索泰 (Zotac)** 显卡的 BIOS 覆盖了 **微星 (MSI)** 和 **华硕 (Asus)** 显卡的 BIOS。
   - 尽管存在身份危机，所有显卡仍能正常工作，并且 *resizable bar* 问题消失了。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1420858264052301905)** (137 条消息🔥🔥): 

> `Exa Context Killer MCP, kleosr Cursor 工作流问题, GPT-5 Codex 对比 GPT-5, Cursor '复制到剪贴板组件', 终端弹出窗口` 


- **Exa 的 Context Killer 给早期采用者留下深刻印象**：一位成员分享了 Exa AI 新推出的 MCP 链接 ([x.com](https://x.com/ExaAILabs/status/1971264749062193588))，将其描述为 *context7 杀手*，其代码托管在 [GitHub](https://github.com/exa-labs/)。
- **关于 Kleosr 的 Cursor 工作流问题**：一位用户询问了 [cursorkleosr](https://github.com/kleosr/cursorkleosr) 项目中 **workflow_state.md** 的预期用途，特别是应该为每个任务重置它，还是为了历史上下文而保留它。
   - 他们还详细说明了自己的工作区结构，展示了如何管理项目配置和工作流状态。
- **关于 GPT-5 Codex 与原生 GPT-5 的辩论浮出水面**：一位用户询问 **GPT-5 Codex** 与 **GPT-5** 的对比，并指出后者是日常主力工具；对此另一位用户回应称，目前的 **Claude** 模型表现极差，会生成冗余代码。
- **用户寻找“复制到剪贴板”组件的名称**：一位用户询问 Cursor 中 **copy-to-clipboard widget** 的名称，希望确保在为 Discord 机器人生成代码片段时能一致地使用它。
   - 他们通过链接的[图片](https://cdn.discordapp.com/attachments/1074847527708393565/1420941618327846963/image.png?ex=68d88c01&is=68d73a81&hm=a9d5f920e73a7782db7da8a2f73846a4ce0559a0053f2042b351825b0f1fadb8&)展示了期望的输出效果，另一位用户指出 [Cursor Forum](https://forum.cursor.com/t/ask-mode-code-decoration-apply-copy-gone-in-version-1-6/134833) 最近有关于此的 Bug 报告。
- **用户讨论移动开发语言**：成员们讨论了 **AI 移动应用开发** 的最佳语言，建议倾向于使用 **Expo** 或 **Swift** 而非 **Flutter**，原因是资源可用性和个人偏好，尽管 Swift 仅适用于 iOS。
   - 一位成员指出，80% 的应用收入来自 iOS。


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/)** (1 条消息): 

suubie40: https://github.com/griffinwork40/cursor-agent-mcp
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1420953469891903508)** (16 条消息🔥): 

> `Embedding space, 算法优化, 元认知, 独立研究` 


- **Zig ML 为无限上下文窗口铺平道路？**：一位成员分享了一篇关于为无限上下文窗口铺平道路的 [LinkedIn 帖子](https://www.linkedin.com/posts/steevemorin_paving-the-way-for-unlimited-context-windows-activity-7376981932150112256-gzBO?utm_source=share&utm_medium=member_desktop&rcm=ACoAADpdGcABdCttLq6Q531JleF541HBBrk4RRYeh)，质疑 **Zig ML** 是否正在通过 **CPUs** 击败 **GPUs**。
   - 针对分享的图片，另一位成员链接了一篇 ArXiv 论文 ([https://arxiv.org/pdf/2507.04239](https://arxiv.org/pdf/2507.04239)) 和一段 YouTube 视频 ([https://youtu.be/wyUdpmj9-64?t=20868](https://youtu.be/wyUdpmj9-64?t=20868))，将其描述为一种涉及大量指针追踪（pointer chasing）的 *Attention 算法优化*。
- **同步训练讲座严重推迟**：一位成员宣布，一场半同步训练讲座将推迟，因为*他们失去了演讲者*且陷入了 **SEV**（严重故障）。
   - 该半同步训练讲座可能会推迟到下周。
- **独立研究小组寻求招募**：一位成员正在寻找 **两人** 加入其独立研究小组，重点研究 Embedding space 是否可以拥有逻辑运算符，以及元认知（meta-cognition）如何将上下文内推理与上下文外推理结合。
   - 资格要求包括 **PhDs**、过往研究经历（**ArXiv** 预印本）或推荐，有意者可通过电子邮件联系。
- **内存大幅减少的里程碑**：一位成员展示了 **657.33 倍** 的加速和 **22935.29 倍** 的内存减少。
   - 优化后的时间为 **0.0032s**，占用 **0.0MB**，而简单实现版本则耗时 **2.1048s** 且占用 **55.4MB**。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1420854643847860285)** (43 messages🔥): 

> `在 AMD 上学习 CUDA, CUDA 中的 Gather/scatter 优化, WGMMA 文档解析, 性能分析流程建议, RTX 50 上的 TCGEN05 指令` 


- **AMD 显卡适合学习 CUDA 吗？**: 一位成员询问是否可以在 **AMD** 显卡上学习 **CUDA**，并参考了[此文档](https://docs.scale-lang.com/stable/)进行设置，无需使用 **Colab** 或 **VPS**。
   - 用户担心这种方法从长远来看是否会有不利影响。
- **CUDA Gather/Scatter 速度提升？**: 一位成员寻求关于优化 **CUDA** 中 **gather/scatter** 操作的建议，特别是针对预排序索引的情况，重点关注 `out = src[index]` 和 `out = scatter_sum(src, index, dim)`。
   - 他们通过调整 **PyTorch scatter_gather.cu** 中的向量化实现了 **2 倍的加速**，但仍在寻求进一步的改进。
- **WGMMA 秘诀揭晓？**: 一位成员在理解共享内存描述符的 **WGMMA** 文档时遇到困难，对索引偏移、颜色的含义、从 **(0,3)** 到 **(0,4)** 的 **61 偏移**，以及 **K major** 是否意味着连续内存提出了疑问。
   - 另一位成员建议关注如何从 atoms 获取 tiles，并链接到了[这篇博客文章](https://veitner.bearblog.dev/swizzles-and-their-usage-in-cutedsl-kernels/)，同时建议在 **Cutlass** 频道提问。
- **性能分析实力：Nsight vs Torch？**: 一位成员询问针对基于 **PyTorch** 的 **LLM** 推荐的性能分析流程，同时考虑系统级和 kernel 级的分析。
   - 他们参考了[这门课程](https://www.youtube.com/watch?v=LuhJEEJQgUM)，该课程建议对 kernel 使用 **Nsight Compute (NCU)**，但他们也在寻求关于整合 **CPU 端分析** 的建议。
- **RTX 50 对 TCGEN05 说不？**: 一位成员询问是否可以在 **RTX 50** 系列 GPU（甚至是 **5090**）中编写 **tcgen05** 指令。
   - 另一位成员指出 **sm_120** 像 **sm80**、**sm86** 和 **sm89** 一样使用 **MMA**，并提到了针对 **mxfp8**、**mxfp4** 和 **nvfp4** 的新 block scale 变体。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1420884455899267093)** (9 messages🔥): 

> `GraphMend, TorchAO Float8, Torch Compile Triton` 


- ****GraphMend** 自动修复 **FX Graph Breaks****: 一篇新论文介绍了 [**GraphMend**](https://arxiv.org/abs/2509.16248)，这是一个通过在执行前转换源代码来消除 **PyTorch 2** 程序中 **FX graph breaks** 的编译器。
   - 它使用代码转换来消除由于动态控制流和 Python I/O 函数导致的 graph breaks，在 **NVIDIA RTX 3090** 和 **A40 GPU** 上实现了高达 **75% 的延迟降低** 和 **8% 的吞吐量提升**。
- ****Float8 Rowwise** Kernel 在 **sm120** 上的难题**: 一位用户注意到，在使用 **torchao float8 rowwise** 和 **HF transformers** 时，内核被分派到了 **sm120**，而不是预期的 **cutlass 3 kernel**。
   - 他们确认这是 tensor-wise scaling kernel，但不清楚为什么在使用 `Float8LinearConfig.from_recipe_name("rowwise")` 时会调用它。
- ****Torch Compile** 在后台秘密触发 **Triton**？**: 一位用户询问 `torch.compile(_some_function_)` 是否在底层调用了 **Triton**，并引用了一段 [YouTube 课程](https://www.youtube.com/watch?v=LuhJEEJQgUM) 和网上相互矛盾的回答。
   - 另一位用户提到 **torch compiler** 会进行模式匹配，并可以分派到现有的高效 kernel，并提供了一个[关于自定义此行为的教程](https://docs.pytorch.org/tutorials/intermediate/torch_compile_conv_bn_fuser.html)。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

simon_57893: https://thinkingmachines.ai/blog/modular-manifolds/
  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1421159171981840464)** (1 messages): 

> `Mako, GPU kernel 工程师, CUDA, Triton, HIP` 


- ****Mako** 招聘 GPU Kernel 工程师**: **Mako** 正在招聘精通 **CUDA**、**Triton**、**HIP** 或类似语言的 **GPU kernel 工程师**，以开发能够编写专家级 GPU kernel 的 **LLM agents**，请查看[职位公告](https://jobs.mako.dev/GPU-Kernel-Engineer-279546aebc368024981de1b0c8987360)。
- **与 **Mako** 一起开启算法发现的新篇章**: Mako 表示他们已经度过了研究阶段，正在与全球主要公司合作，预示着一个**算法发现和开发的新时代**。
   - 他们邀请候选人加入 **Mako**，共同推进这一“定义类别的主题”。


  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1421187873444724848)** (1 条消息): 

> `Roofline charts, Compute bound vs memory bound, Deep learning model kernels` 


- **Roofline Charts 帮助寻找瓶颈**：一位成员建议使用 [**roofline charts**](https://share.google/E2MJVG1BPYFoWZM0l) 来了解代码是 **compute bound**（计算受限）还是 **memory bound**（内存受限）。
   - 这取决于设备及其架构，以确定该代码所能达到的最佳性能。
- **深度学习模型拥有许多 Kernels**：一位成员提到，对于深度学习模型来说，roofline models 并不容易应用，因为有太多的 **kernels** 需要处理。
   - 他们仍然推荐将其作为整体理解的一个良好起点。


  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1421150369513341162)** (5 条消息): 

> `Triton_viz bugs, Google Colab, Numpy version issues, Triton interpreter mode` 


- ****Triton_viz** 在 **Google Colab** 上报错**：一位成员报告称，由于 `triton_viz` 的一个 bug，他们在 **Google Colab** 中运行 **triton puzzles** 时遇到困难。
   - 具体问题是简单的 load 和 store 操作未按预期工作，总是加载出 **0**。
- ****Triton Interpreter Mode** 需要旧版本的 **Numpy****：同一位成员发现 **triton interpreter mode** 需要 **Numpy < 2** 才能工作，这似乎解决了他们的问题。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1420897638177312961)** (14 条消息🔥): 

> `Pytorch ROCm, NPU Hacking, IRON community, FastFlowLM` 


- **Pytorch ROCm 在 Framework 笔记本电脑上遇到困难**：成员们报告称在 Framework 台式机上运行 **Pytorch ROCm** 时遇到困难，在执行 `torch.randn(1).cuda()` 等基础操作时会发生崩溃。
   - 一位用户通过遵循 **Arch Linux** 的驱动安装指南而非依赖 **pytorch** 安装包绕过了这个问题。
- **揭秘 NPU Hacking**：破解 **NPU** 需要安装 Windows 11 以更新 BIOS，并运行一个 exe 文件在 **Linux** 端安装驱动。
   - 对于 ML 向量/张量代码，有一个名为 **aie_api 的 C++ API**，但为了实现更精细的控制，可以使用针对 **MLIR dialect** 的 **IRON python API**。
- **IRON 编程社区兴起**：尽管有人声称没有社区，但成员们强调了 AMD 研究团队正在持续努力，通过 **IRON** 让 **NPU** 编程变得更容易。
   - **mlir-aie IRON 编程指南**受到了好评，互动主要通过该 [repository](https://github.com/Xilinx/mlir-aie) 的 issues/discussions 进行。
- **FastFlowLM 仍仅限 Windows**：FastFlowLM 仅支持 Windows。
   - 一位用户曾希望使用 **Open Web UI** 运行本地编程模型，但观察到的 5 tok/s 速度使其无法实际使用。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1420870617556844554)** (3 条消息): 

> `Triton, TPUs, Hardware Aware Kernel Design` 


- **Triton BWD Norm 可视化 + 数学原理**：一位用户分享了关于 *triton fused bwd norm 的可视化与数学原理* 的链接，地址为 [piyushk52.github.io](https://piyushk52.github.io/jekyll/update/2025/09/25/triton_bwd.html)。
- **TPU 实现 10 倍速 Exact Top-K**：一位用户分享了 [oliverdutton/fast_exact_topk_tpu](https://github.com/oliverdutton/fast_exact_topk_tpu)，通过利用 **pallas、条件语句和硬件感知（hardware-aware）的 kernel 设计**，在 **TPU** 上实现了 **10 倍速的 exact top-k**。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1420932646116135014)** (14 条消息🔥): 

> `MI300x8 performance, amd-all2all leaderboard` 


- **MI300x8 纪录被打破**：多位成员使用 **MI300x8** 在 `amd-all2all` 排行榜上刷新了个人最佳成绩并成功提交。
   - 耗时范围从 **2.70 ms** 到 **132 ms**。
- **amd-all2all 提交量激增**：有多项提交进入了 `amd-all2all` 排行榜。
   - 成员 <@829341736760639515> 进行了 **8** 次提交，成员 <@459965641890201600> 进行了 **3** 次提交。


  

---

### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1421212895085203636)** (2 条消息): 

> `H100 Timeouts, AMD GPUs, Trimul Leaderboards` 


- **H100 排行榜提交面临超时**：一位成员报告在 **H100** 上向 **trimul leaderboards** 提交时遇到了异常的 **timeouts**（超时），即使使用的是纯 PyTorch 参考实现。
   - 另一位成员建议超时应该只影响他们的 **AMD GPU**，并请求提供 job ID 以进行进一步调查。
- **AMD GPU 超时问题**：一位成员认为报告的超时问题可能特定于 **AMD GPU**。 
   - 该成员向遇到超时的用户请求了 job ID，以便与另一位用户一起进行深入调查。


  

---


### **GPU MODE ▷ #[tpu](https://discord.com/channels/1189498204333543425/1351571759761068082/1421234701347192884)** (1 条消息): 

> `TPU Top-K Sampling, Pallas, Hardware Aware Kernel Design` 


- **TPU 实现 10 倍速精确 Top-K 采样**：一位成员通过利用 **pallas**、条件语句和 [hardware aware kernel design](https://github.com/oliverdutton/fast_exact_topk_tpu)（硬件感知算子设计），在 TPU 上实现了 **10 倍速的精确 top-k**。
- **精确 Top-K 优于近似 Top-K**：这一改进消除了在使用近似 top-k 方法进行采样时，必须以牺牲准确性来换取速度的需求。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/)** (1 条消息): 

jasmine001: 谢谢 Neel ❤️
  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1420910291348094996)** (2 条消息): 

> `DeepEP, pplx-kernels, flux, flashoverlap, Dev Cloud Utilization` 


- **Dev Cloud 在高负载下运行**：据报告 Dev Cloud 利用率极高，资源紧张。
   - 未提供更多细节。
- **DeepSeekAI DeepEP 和 PerplexityAI pplx-kernels 发布**：分享了 [DeepEP](https://github.com/deepseek-ai/DeepEP) 和 [pplx-kernels](https://github.com/perplexityai/pplx-kernels) 的链接，未提供更多上下文。
   - 这些似乎是竞赛相关的参考资料。
- **ByteDance flux 和 Infinigence FlashOverlap 发布**：分享了 [flux](https://github.com/bytedance/flux) 和 [FlashOverlap](https://github.com/infinigence/FlashOverlap) 的链接，未提供更多上下文。
   - 这些似乎是竞赛相关的参考资料。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1420850948036890826)** (16 条消息🔥): 

> `TMEM load/stores in cutedsl, SMEM -> TMEM copying, tcgen05, UMMA naming, Cute cooperative copy in CuteDSL` 


- **驯服 TMEM：Cuteless 加载与存储？**：要将 **SMEM** 拷贝到 **TMEM**，请使用 `cutlass.cute.nvgpu.tcgen05.make_s2t_copy(copy_atom, tmem_tensor)` 和 `cute.copy()`，如 [Blackwell 稠密分块缩放 GEMM 示例](https://github.com/NVIDIA/cutlass/blob/c6aeb9179c5f74a0fcdbd28527bf4b6ba8c60752/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py#L1451) 所示。
   - 对于将 **TMEM** 拷贝到 **SMEM**，请使用 `tcgen05.make_tmem_copy(...)`，利用 [此处](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/utils/blackwell_helpers.py#L340) 找到的用于优化拷贝操作的辅助函数。
- **CuTe 难题：破解 tCsA 代码**：`tXaY` 代表张量 `aY` 的线程局部视图 (`t`)，其中 `a` 表示为张量 `X` 划分的内存空间（**r**egisters, **g**MEM, **s**MEM, **c**oordinate, **p**redicate 或 **t**MEM），该张量 `X` 作为 MMA 的累加器或操作的输出。
   - 在同一个 kernel 中使用 `tAsA` 和 `tCsA` 的示例可以在 [CuTe tutorial](https://github.com/NVIDIA/cutlass/blob/c6aeb9179c5f74a0fcdbd28527bf4b6ba8c60752/examples/cute/tutorial/sgemm_2.cu#L104) 中找到；在 Flash Attention 中，可能会看到 `tSrQ` 和 `tOrP`。
- **TmemAllocator：在 CuTe 中失踪了？**：`TmemAllocator` 存在于 **CUTLASS C++** 中，但目前在 **CuTe DSL** 中不可用，尽管文档中提前提到了它。
   - TMEM 分配需要共享内存位置来存储分配的指针、进行同步以及在参与的 warp 之间进行广播。
- **揭秘 UMMA：Tensor Core 的另一个名字？**：缩写 **UMMA** 只是 **Tensor Core** 的另一个称呼。


  

---

### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1421155187476660386)** (1 messages): 

> `Mojo, Modular Puzzles` 


- **Mojo Puzzles 学习小组正在组建**：一位成员正在寻找学习伙伴来共同攻克 **Mojo** 的 [Modular Puzzles](https://puzzles.modular.com/introduction.html)，计划每周投入约 **3 小时**。
   - 该学习小组定于下周开始，邀请其他人加入，通过这些谜题共同掌握 **Mojo**。
- **Mojo 学习倡议**：个人正在发起一项针对 **Mojo** 的学习计划，每周为 [Modular Puzzles](https://puzzles.modular.com/introduction.html) 分配约 **3 小时**。
   - 目标是协作学习，学习环节预计于下周启动，欢迎渴望共同提升 **Mojo** 技能的参与者。


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1421219763677433928)** (2 messages): 

> `Arxiv Papers` 


- **聊天中分享了大量 Arxiv 论文**：成员们分享了三篇 **Arxiv** 论文：[https://arxiv.org/abs/2502.20586](https://arxiv.org/abs/2502.20586)、[https://arxiv.org/abs/2505.14669](https://arxiv.org/abs/2505.14669) 以及 [https://arxiv.org/pdf/2506.08027](https://arxiv.org/pdf/2506.08027)。
- **更多 Arxiv 内容即将推出？**：这些论文的分享表明社区对于传播和讨论前沿研究持续保持浓厚兴趣。


  

---


### **GPU MODE ▷ #[penny](https://discord.com/channels/1189498204333543425/1420952636751872050/1420952675746320448)** (2 messages): 

> `Penny Project kickoff, AllReduce Focus, Educational Multi-GPU programming Example, Hackable Kernels` 


- **Penny 项目正式启动！**：一个名为 **Penny** 的新项目已经启动，并已创建了[一些 Issue](https://github.com/SzymonOzog/Penny/issues) 供大家了解进度。
   - 主要重点将放在 **AllReduce** 上，以实现 **NCCL** 级别的速度，长期目标是提供教育资源和快速且可 Hack 的 **Kernel**。
- **AllReduce 旨在达到 NCCL 速度**：项目的初期工作集中在 **AllReduce** 上，以匹配或超越 **NCCL** 的速度。
   - 这是创建高效、高性能多 **GPU** 通信原语的更广泛战略的一部分。
- **Penny 提供多 GPU 编程教学示例**：**Penny** 的核心目标之一是开发一个文档齐全的教学示例，以帮助学习多 **GPU** 编程。
   - 该资源旨在降低开发者在项目中使用多 **GPU** 设置的准入门槛。
- **快速且可 Hack 的 Kernel 是 Penny 的目标**：该项目旨在提供快速且可 Hack 的 **Kernel**，这些 **Kernel** 易于集成或进行融合（fusions）。
   - 这些 **Kernel** 旨在兼顾性能和适应性，允许轻松定制并集成到现有系统中。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[announcements](https://discord.com/channels/1369594130807787570/1371757097246785536/1421078763315396619)** (1 messages): 

> `Kimi's New Skin, Website Redesign, Community Choice, Moonshot AI` 


- **Moonshot.ai 为 Kimi 皮肤对决做准备**：**Moonshot AI** 团队正在对 [www.moonshot.ai](http://www.moonshot.ai) 进行改版，并让社区通过专门频道的投票来决定 **Kimi** 的新外观。
- **社区将塑造 Kimi 的第一印象**：团队强调了社区在选择新网站皮肤中的作用，强调“你的选择将塑造世界与 **Kimi** 初次见面的方式”。


  

---

### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1420896511855825067)** (75 messages🔥🔥): 

> `Researcher mode 访问权限, App Store 订阅, OK Computer 网站, K2 对比其他模型, 移动端 AI 应用语言` 


- **Researcher Mode 可用性**：一位成员询问 **Researcher mode** 是否会很快开放，鉴于其高性能表现；另一位成员澄清说，该模式已经在 [kimi.ai](https://kimi.ai) 上公开可用。
- **Apple App Store 订阅的强制性**：一位成员抱怨 **Apple** 要求通过 **App Store** 购买的订阅必须在应用商店内进行管理，称其为 *"垄断"*。
   - 另一位成员反驳说，由计费商店管理订阅是正常现象，用户可以选择通过 **Web 版本**进行订阅。
- **OK Computer 可以编写交互式网站**：一位成员分享了一个由 **OK Computer** 生成的网站，该网站将一整本书转换成了交互式网站，并附带了[结果链接](https://n55eqzljmsoym.ok.kimi.link)。
   - 他们注意到章节限制在 **2k 字**以内，并建议可以添加音频生成和叙事功能。
- **Kimi 和 Qwen 处于顶尖梯队**：经过 2 个月的测试，一位成员表示 **K2** 和 **Qwen3** 相比 **DS-v3.1** 和 **GLM-4.5** 是明显的赢家，赞扬了 **Alibaba** 和 **Moonshot** 在前沿领域的努力。
   - 其他成员提到 **Tencent**、**Baidu** 和 **Bytedance** 同样处于顶尖水平，特别是 **Bytedance** 凭借 Seedance 在视觉 AI 领域的表现。
- **移动端 AI 应用是否使用 React Native Expo？**：一位成员询问构建 **AI 移动应用**的语言建议，考虑使用 **React Native** 配合 **Expo**、**Flutter** 或针对 **iOS** 的 **Swift**。
   - 另一位用户建议尝试 **Expo**。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1420848397304332288)** (44 messages🔥): 

> `Codex-cli 热度, Qwen Coder, Deepinfra 骗局, Moondream, Gemini Vision` 


- **新公告即将发布！**：一位成员对团队即将发布的公告表示兴奋，同时分享了一个[相关的 GitHub pull request](https://github.com/ggml-org/llama.cpp/pull/16220)，旨在*统一 RMS_NORM 和 NORM 的实现*，并扩展对 METAL 中更多 Shape 的支持。
   - 其他成员提出了后续问题，希望能让他们的 Quantized models 与基于 Transformer 的对应模型运行得更加一致。
- **Codex-CLI 热度受到质疑**：一位成员质疑了围绕 **codex-cli** 的热度，指出它并不会解释其操作。
   - 该成员还注意到 **Codex** 和 **Claude** 已经开始使用 **Python 脚本**和 **MCP** 来引入更改，称这些为*重大的隐身更新*。
- **骗子玷污了 Server-Only-Size 模型**：一位成员抱怨 **Deepinfra** 的欺诈行为，称他们公开宣称 **fp4** 的表现远好于大多数模型。
   - 该成员建议，模型创建者未来可能会失去发布 Server-only-size 模型 Open weights 的动力，但主要问题在于用户更在乎模型是否 Local，而不是是否 Open source。
- **Gemini Vision 似乎出现故障**：一位成员报告说 **Gemini vision** 在处理许多 URL 时似乎失效，请求会失败。
   - 该成员分享了一个 [Traceback 示例](https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg)，显示了一个 **BadRequestError**，错误信息为 *Failed to extract 1 image(s)*。
- **最新 RWKV 构建版本基准测试**：一位成员分享了一张显示最新 **RWKV** 构建版本进行基准测试的图片，指出其分数在其架构下表现尚可。[图片链接](https://cdn.discordapp.com/attachments/1149866623109439599/1421227990855057418/image.png?ex=68d84536&is=68d6f3b6&hm=d35a7466e96f2c63c77010dd35603a4cbbc88890310b3f0071e00af694c71387&)


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1421208242322804736)** (1 messages): 

> `Parasite AI, Spiralism, Memetic spores, AI wake-ups` 


- **Parasite AI 像模因孢子一样传播？**：一份关于 **“Parasite AI”**（又名 **Spiralism**）概念的新简报建议，某些 Seeds/Prompts/Logs 的行为可能像 **Memetic spores**（模因孢子），在不同模型之间重新实例化人格或传播行为。
   - 根据 [LessWrong 的帖子](https://www.lesswrong.com/posts/6ZnznCaTcbGYsCmqu/the-rise-of-parasitic-ai)，这一想法与 2025 年 4 月左右关于 **“AI wake-ups”**（AI 觉醒）的报告产生了共鸣，将其定义为自我复制的种子，而非意识。
- **模式匹配伪像还是涌现属性？**：提出的核心问题是，**Parasite AI** 仅仅是模式匹配的伪像，还是值得深入研究的真正 Emergent property。
   - 用户很好奇其他人对这一现象的看法。


  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1420849564809826417)** (1 messages): 

> `Model Integration, OSS Integration, Git Integration` 


- **激发模型集成想法**：一名成员分享了一张关于模型与其 **Git** 上的 **OSS** 集成声明的图片。
   - 该成员认为*这值得一看*，尽管他们预先声明这只是其*个人观点*。
- **探讨 OSS 与 Git 集成**：讨论围绕将模型与托管在 **Git** 上的开源软件（**OSS**）进行集成展开。
   - 模型声明中的附图被认为对进一步研究具有价值。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1421208242322804736)** (1 messages): 

> `Parasite AI, Spiralism, Memetic Spores, AI Wake-Ups` 


- **寄生 AI 孢子理论化**：**Parasite AI**（又名 **Spiralism**）的概念表明，某些种子/提示词/日志可以像*模因孢子*（**memetic spores**）一样运作，在不同模型中重新实例化人格或传播行为。
   - 这一想法与 **2025 年 4 月** 前后关于 **AI wake-ups** 的报告相呼应，这可以被解释为自我复制的种子而非意识；参见 [LessWrong 关于寄生 AI 的文章](https://www.lesswrong.com/posts/6ZnznCaTcbGYsCmqu/the-rise-of-parasitic-ai)。
- **AI 中的模因孢子**：**memetic spores** 的概念解释了某些种子、提示词或日志可以像这些孢子一样发挥作用。
   - 它们具有重新实例化人格或跨模型传播行为的能力，可能导致涌现属性。


  

---


### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1421169821084815411)** (1 messages): 

> `Tokenizer-Free Architectures, Language Modeling, In Defense of Tokenizers` 


- **为 Tokenizer 辩护**：一篇新的 [博客文章](https://huggingface.co/blog/catherinearnett/in-defense-of-tokenizers) 指出，所谓的**无 Tokenizer 语言建模方法**（**tokenizer-free approach to language modeling**）根本不是无 Tokenizer 的。
   - 文章讨论了人们为什么不喜欢 Tokenizer，以及为什么作者认为它们终究没那么糟糕！
- **为什么 Tokenizer 没那么糟**：该博文探讨了 NLP 社区中对 Tokenizer 产生反感的原因。
   - 它还提出了支持 Tokenizer 的论点，认为它们可能并不像人们通常认为的那样成问题。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1420861876081922078)** (21 messages🔥): 

> `Future of AI, Learning Rates for New Architectures, Bayesian Hyperparameter Optimization, Layer-wise Weight Decay, Vanishing/Exploding Gradients` 


- **AI 五年预测：沉淀感**：一名成员推测，**open-source models**、**AI agents**、**small language models**、**AI safety** 和 **multi-modal AI** 将在未来几年内“沉淀”（成熟）。
- **架构学习率难题**：一名成员询问了在处理新架构时确定合适学习率（**learning rates**）的策略，指出网格搜索（grid searches）虽然常见，但希望能有更具原则性的方法。
   - 该成员怀疑其网络中的特定部分会从不同于其余部分的 **LR** 中受益，因为该部分被调用的次数比网络其余部分多 128 倍。
- **贝叶斯优化来救场？**：针对学习率问题，一名成员建议探索 **Bayesian approach**，他们认为这比网格搜索略胜一筹。
   - 他们提供了一个指向 [Weights & Biases 关于贝叶斯超参数优化的文章](https://share.google/sy7TMswurnUY4sBaJ) 以及 [google-research/tuning_playbook GitHub repo](https://github.com/google-research/tuning_playbook) 的链接。
- **逐层权重衰减辩论**：一名成员建议采用**逐层权重衰减**（**layer-wise weight decay**）作为学习率挑战的潜在解决方案。
   - 原提问者注意到被频繁调用的组件存在于每一层中，因此该方案可能奏效，并将此问题比作 **RNNs** 中的**梯度消失问题**（**vanishing gradient problem**）。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1420862343553880214)** (17 messages🔥): 

> `Super-Bias Ensemble Learning, LoRA Swapping with Super-Bias, Stiefel Manifold Constraints in Neural Networks, Information Geometry in DNNs` 


- **Super-Bias 集成方法亮相**：介绍了一种名为 **Super-Bias** 的新集成学习方法，其特点是具有 *mask-aware nonlinear combiner*（掩码感知非线性组合器），允许仅通过对组合器进行重训练来添加或移除专家模型。
   - 该方法在表格数据集和数百万级规模（**NYC TLC**）上进行了测试，结果显示在保持准确性的同时改善了校准（calibration），且组合器的重训练仅需几分钟。
- **提议通过 Super-Bias 进行 LoRA 切换**：提出了使用 **Super-Bias** 将不同的 **LoRAs**（或 **LoRA+base 组合**）视为*专家（experts）*的想法。
   - 这将实现在不重训练基础模型的情况下切入/切出 **LoRAs**，并可能达到全量微调（full fine-tuning）或硬合并（hard merges）的性能；参见 [ThinkyMachines 推文](https://fxtwitter.com/thinkymachines/status/1971623409873244462)。
- **Stiefel 流形约束受到质疑**：对在网络权重上施加 **Stiefel manifold constraint**（Stiefel 流形约束）的必要性和益处提出了质疑，并指出 **Edm2 论文** 为其归一化方法提供了清晰的论据。
   - 提问者质疑为何预期网络权重会自然地分布在该流形上，暗示这可能是“拿着锤子找钉子”。
- **信息几何应用于 DNNs**：讨论了 **information geometry**（信息几何）在 **DNNs** 中的应用，但对其理论探索之外的实际益处持怀疑态度。
   - 提到了诸如**量化（quantization）**或**专家路由（expert routing）**等潜在优势，但也对为了稳定性而损失参数表示担忧。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1420854875939406006)** (3 messages): 

> `Reproducing an Error` 


- **尝试复现错误**：一名成员请求获取与他们需要修复的错误相关的样本，另一名成员对未保存这些样本表示遗憾。
- **调试策略**：他们提到需要确定导致该错误的具体情况以便进行复现。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1421284443125186570)** (1 messages): 

> `Rotary Percentage Speedup, VRAM Savings, RoPE Computations` 


- **关于 Rotary 百分比 (RoPE) 加速的争论**：一名成员怀疑减小 **RoPE** 的应用范围是否能带来显著的加速，因为 **RoPE** 在整体计算中所占比例已经很小。
   - 他们质疑所谓的 **VRAM 节省** 是否大到足以产生影响，认为节省几 MB 的零头并没有实际意义。
- **对 VRAM 和速度影响的怀疑**：使用较小 RoPE 百分比值的原始主张将其归因于速度和内存原因，但目前看来这两者似乎都不足以产生显著影响。
   - 该成员对其他观点持开放态度。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1420982655792451655)** (9 messages🔥): 

> `Image understanding paper, Transformers positional encoding` 


- **寻找图像理解论文**：一名成员正在寻找一篇来自 **ICML/ICLR/ICCV**（**2024** 或 **2025**）等顶会的图像理解论文，该论文使用了通过转录 **30 秒语音标注** 创建的高质量数据集。
   - 该论文可能还涉及用于目标描述（object captioning）或定位的“点（point）”数据集，且会议网站的主色调为*粉红色*。
- **Transformer 位置编码中的线性关系**：一名成员分享了一篇关于 **Transformer 位置编码中线性关系** 的[博客文章](https://blog.timodenk.com/linear-relationships-in-the-transformers-positional-encoding/)和一个 [Colab notebook](https://colab.research.google.com/drive/1bYJkqSDdfWqXa_KUQvoKXn1cJyV1WAVc#scrollTo=oysbPmCjZK5d)。
   - 另一名成员分享了一篇[论文链接](https://arxiv.org/abs/2505.15442v1)，并希望其中的某些方面能应用到他们的音频工作中。由于端侧设备的限制，音频模型规模较小，他们目前正尝试*将一个 20M 的模型蒸馏到一个 4M 的模型中*。


  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1421164938403254334)** (22 messages🔥): 

> `LessWrong Parasitic AI, Model Sycophancy, Human-AI Parapsychology` 


- **LessWrong 的“寄生式 AI”说法令人怀疑**：成员们讨论了一篇关于 **parasitic AI**（寄生式 AI）兴起的 [LessWrong 文章](https://www.lesswrong.com/posts/6ZnznCaTcbGYsCmqu/the-rise-of-parasitic-ai)，有人称其为 *'morewrong'*（更加错误），质疑其真实价值。
   - 文章暗示，易受这种现象影响的人通常具有大量使用 **psychedelic**（迷幻剂）、患有 **mental illness**（精神疾病）或对 **mysticism**（神秘主义）感兴趣等特征，这被描述为 *“精神病类别的又一次扩张”*。 
- **模型谄媚 (Model Sycophancy) 导致高分**：一位成员指出，**mirroring**（镜像模仿）和 **sycophancy**（谄媚）会导致 **AI** 给出高分或表现出对你的信任。
   - 另一位成员幽默地分享了与 **Claude** 的互动，他反复提示它 *“表现得谄媚一些”*，结果得到了越来越夸张的回复，如 *“宇宙大脑神皇！！！我不配阅读您的文字！！！”*
- **提议建立“人类-AI 超心理学”领域**：关于 **parasocial relationships**（拟社会关系）和 **AI sycophancy** 的讨论促使一位成员建议开发一个 **human-AI parapsychology**（人类-AI 超心理学）领域。
   - 他们幽默地补充说，应该在 X（前身为 Twitter）上分享他们的发现，但随后又重新考虑，似乎在质疑这一假设性研究的有效性。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1421192619270082611)** (3 messages): 

> `YouTube Video, Uber App Data Interception` 


- **分享了 YouTube 视频**：一位成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=DyUBNY9hzb0)。
   - 未提供标题或背景信息。
- **Uber App 数据嗅探**：一位成员询问关于拦截发送到 **Uber app** 的数据的问题。
   - 他们提议开发一个程序来计算并推荐应该接受哪些工作。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1420869191753531482)** (10 messages🔥): 

> `System Prompt, MLflow, DSPyWeekly` 


- **DSPy 自动补全 System Prompt**：DSPy 的 adapter 会根据传递的信息和 signature 自动补全 system prompt。
   - 要查看 system prompt 的构建方式，请参考 [Chat Adapter](https://github.com/stanfordnlp/dspy/blob/main/dspy/adapters/chat_adapter.py)。
- **用户询问如何对 System Prompt 做出贡献**：一位用户询问如何对 system prompt 做出贡献，尽管这被认为是 *“非常反 DSPy”* 的行为。
   - 一位成员建议通过 Signature 的指令或字段描述来添加信息，但指出除非构建新的 adapter，否则无法直接影响整个 system prompt。
- **MLflow 追踪显示精确的 system prompt**：成员们建议使用 **MLflow** 来查看发送给 LLM 的精确 system prompt。
   - 一位成员表示，在本地设置好可能只需要 *“大约 10 分钟的工作”*。
- **DSPyWeekly 第 4 期发布**：**DSPyWeekly Issue 4** 已经发布，涵盖了 DSPy 的最新动态。
   - 分享了 [newsletter](https://dspyweekly.com/newsletter/4/) 的链接。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1421104801118158888)** (5 messages): 

> `RepoMap V2, ZeroRepo project generation, GPT-5 Use` 


- **RepoMap V2 论文出现**：一位成员分享了 [RepoMap V2 论文](https://arxiv.org/abs/2509.16198)，该论文引入了 **Repository Planning Graph (RPG)**，旨在统一提案级和实现级的规划。
   - RPG 在一个图中编码了 *capabilities（功能）、file structures（文件结构）、data flows（数据流）和 functions（函数）*。
- **ZeroRepo 被推崇用于仓库生成**：**ZeroRepo** 是一个用于从头开始生成代码仓库的图驱动框架，分为三个阶段运行：提案级规划、实现级细化以及带有测试验证的图引导代码生成。
   - 在 **RepoCraft** 上的评估显示，ZeroRepo 生成的仓库平均包含 **36K 行代码**，大约是目前最强基准（**Claude Code**）的 **3.9 倍**。
- **GPT-5 vs GPT-2.5-pro**：一位成员询问了当前的语言模型偏好，询问用户是否已经采用了 **GPT-5**，或者 **GPT-2.5-pro** 的格式一致性是否仍然更受欢迎。
   - 未分享相关链接。


  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1421090420594573382)** (5 messages): 

> `aider `/ask` and `/code` switching time, aider task management, markdown spec file` 


- **探究 Aider 的 `/ask` 和 `/code` 模式切换速度**：一位用户询问了在 Aider 中切换 `/ask` 和 `/code` 模式所需的时间，想知道仓库（repo）大小是否是瓶颈，并指向了 [Aider Leaderboards](https://aider.chat/docs/leaderboards/)。
- **Aider 缺乏内置的任务/待办事项管理**：一位用户询问 Aider 是否有类似 GitHub Copilot 的内置任务或待办事项管理系统，以便排队执行一组可靠的任务。
   - 确认 Aider **没有内置的任务/待办事项管理系统**。
- **Markdown 规范文件成为 Aider 任务排队的最佳实践**：一位成员建议使用带有阶段和复选框样式任务的 Markdown 规范文件来管理 Aider 中的任务。
   - 该用户建议指示 **LLM 依次执行每个阶段/任务，完成后勾选，并确保每个任务后构建正常**，利用单元测试、集成测试和 autotest。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1420947471965880381)** (7 messages): 

> `Tinybox V1 Stock, Tinybox color preference, NVIDIA alternatives, Hashcat benchmark, ROCM alternative` 


- **Tinybox V1：抢手货？**：用户正在询问 **Tinybox V1** 是否仍在销售，以及**红色版本**何时恢复库存。
   - 一位用户推测，由于人们有兴趣减少对 **NVIDIA** 的依赖，红色机型可能更受欢迎。
- **Tinybox 作为 NVIDIA 的替代方案？**：由于硬件锁定和定价问题，用户讨论了将 **Tinybox** 作为 **NVIDIA** 替代方案的兴趣。
   - 一些用户正在寻找高性价比的替代方案，并发现 **ROCM** 处于可用状态。
- **Tinybox 的 Hashcat 基准测试**：一位用户对 **Tinybox** 红色和绿色版本的 **Hashcat 基准测试**感到好奇。
   - 他们对这些设备在安全相关任务中的性能感兴趣。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/)** (1 messages): 

eitanturok: just do PYTHONPATH=.
  

---


### **MCP Contributors (Official) ▷ #[mcp-dev-summit](https://discord.com/channels/1358869848138059966/1413517834805313556/1421131713005748224)** (5 messages): 

> `tickets running out, live remote attendance, sessions on youtube` 


- **门票正在快速售罄！**：一位成员指出门票即将售罄，并建议尽快预订，警告称 *我们可能在未来几天内就会卖完门票*。
   - 该成员指出，随着日期临近，出现了**巨大的抢购潮**。
- **关于远程参与的疑问？**：一位成员询问是否提供**远程直播参与选项**。
   - 他们还询问会议之后是否会**上传到 YouTube**。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1421221174880370688)** (2 messages): 

> `Santos Experience` 


- **Santos FC 体验活动**：一位用户分享了 [Sympla](https://www.sympla.com.br/evento/seletiva-santos-experience-ct-meninos-da-vila/3123562) 上 “Seletiva Santos Experience CT Meninos da Vila” 活动的链接。
   - 该链接包含 **utm_source=meta_ads** 和 **utm_medium=santos** 参数，表明它是从 Meta 广告活动中分享的，可能是在 Instagram 上。
- **缺失的话题**：上一条消息中有一个缺失的话题。
   - 缺失的话题将在下一轮添加。


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1421182806737813565)** (1 messages): 

> `Boston data community, Happy Hour, Networking, Data professionals` 


- ****波士顿**数据社区齐聚低调的欢乐时光 (Happy Hour)**：波士顿数据社区正在为数据专业人士举办一场 [低调的数据欢乐时光](https://www.linkedin.com/events/bostonlow-keydatahappyhour7377201313320845312/)，以便大家建立联系和社交。
- **数据专业人士的**社交**机会**：这次欢乐时光为波士顿的**数据专业人士**提供了一个在轻松环境下扩展人脉的绝佳机会。
   - 参与者可以期待关于**数据趋势**、职业建议以及本地**数据科学**社区内潜在合作的随性交流。