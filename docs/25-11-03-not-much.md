---
companies:
- openai
- aws
- microsoft
- nvidia
- gpu_mode
- vllm
- alibaba
- arena
- llamaindex
- amazon
- anthropic
- gradio
date: '2025-11-03T05:44:39.731046Z'
description: '**OpenAI** 和 **AWS** 宣布达成战略合作伙伴关系，涉及一项价值 380 亿美元的算力协议，旨在部署数十万颗英伟达（NVIDIA）GB200
  和 GB300 芯片；与此同时，**微软（Microsoft）** 获得了向阿联酋（UAE）运送英伟达 GPU 的许可，并计划投资 79 亿美元建设数据中心。**英伟达**与
  GPU_MODE 启动了为期三个月的 Blackwell B200 NVFP4 内核优化竞赛，奖品包括 DGX Spark 和 RTX 50XX 系列显卡。**vLLM**
  在本地大语言模型（LLM）推理服务领域受到青睐，PewDiePie 的采用便是一个例证。**阿里巴巴**预告了 Qwen3-Max-Thinking 模型，该模型在
  AIME 2025 和 HMMT 基准测试中达到了 100% 的准确率，标志着在结合工具使用的推理能力方面取得了进展。采用 MIT 许可协议的 **MiniMax-M2
  230B MoE** 模型登顶 Arena WebDev 排行榜，与 Claude Sonnet 4.5 Thinking 32k 并列第一。针对 OSWorld
  基准测试的稳定性和任务有效性出现了一些批评。**LlamaIndex** 的 LIGHT 框架在长期记忆任务中表现出显著提升，优于原始上下文和 RAG 基准，在
  1000 万 token 的摘要任务中提升幅度高达 160.6%。**亚马逊（Amazon）** 推出了 Chronos-2，这是一款用于零样本预测的时间序列基础模型。MCP（模型上下文协议）生态系统进一步扩展，新增了
  mcp2py OAuth 集成和 Gemini Docs MCP 服务器等工具，同时 **Anthropic** 和 **Gradio** 举办了构建冲刺活动，提供丰厚的积分和奖品。“OSWorld
  并不真正存在——不同的提示词集导致分数不可比”这一观点凸显了基准测试面临的挑战。'
id: MjAyNS0x
models:
- qwen3-max-thinking
- minimax-m2
- claude-3-sonnet
- llamaindex-light
- chronos-2
people:
- sama
- gdb
- andrewcurran_
- a1zhang
- m_sirovatka
- omarsar0
- _philschmid
title: 今天没发生什么事。
topics:
- compute-deals
- gpu-optimization
- kernel-optimization
- local-serving
- reasoning
- long-context
- benchmarks
- long-term-memory
- time-series-forecasting
- agent-frameworks
- oauth-integration
- developer-tools
---

**平静的一天**

> 2025/10/31-2025/11/3 的 AI 新闻。我们为您检查了 12 个 subreddit、544 个 Twitter 账号和 23 个 Discord 服务（包含 199 个频道和 12068 条消息）。预计节省阅读时间（按 200wpm 计算）：1036 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以精美的 vibe coded 方式呈现所有往期内容。访问 https://news.smol.ai/ 查看完整的新闻分类，并在 @smol_ai 上向我们提供反馈！

连续第 3 个“无新闻日”。今年剩下的时间里只剩下 1-2 个重大模型发布，气氛变得异常安静。

AIE CODE 的套票和[酒店](https://www.ai.engineer/code)即将[售罄](https://x.com/swyx/status/1985415415250468935)！

---

# AI Twitter 回顾

**算力交易、硬件竞赛和推理基础设施**

- **OpenAI x AWS 规模扩张**：[@gdb](https://twitter.com/gdb/status/1985378899648544947) 宣布与 AWS 达成战略合作伙伴关系，以将“更多 NVIDIA 芯片上线”，[@sama](https://twitter.com/sama/status/1985431030430646365) 也对此进行了回应。一份摘要将其定性为“380 亿美元的算力交易……数十万颗 NVIDIA GB200 和 GB300 芯片” ([背景](https://twitter.com/scaling01/status/1985352400631202187))。另外，据 [@AndrewCurran_](https://twitter.com/AndrewCurran_/status/1985325278823125483) 报道，微软获得了美国商务部的许可，可以向阿联酋运送 NVIDIA GPU，并计划在阿联酋数据中心投入 79 亿美元。
- **B200/NVFP4 Kernel 挑战赛（赢取 GB300）**：@GPU_MODE 和 @NVIDIA 宣布在 Blackwell B200 上开展为期 3 个月的 NVFP4 Kernel 优化竞赛，设有单项奖（DGX Spark, RTX 50XX）和特等奖（配备 GB300 的 Dell Pro Max）([@a1zhang](https://twitter.com/a1zhang/status/1985434030473437213), [@GPU_MODE](https://twitter.com/GPU_MODE/status/1985436876384453128), [@m_sirovatka](https://twitter.com/m_sirovatka/status/1985438384337404078))。题目包括 NVFP4 Batched GEMV, GEMM, Gated Dual GEMM 和 Grouped GEMM；DGX B200 由 @sestercegroup 提供。
- **快速、廉价的本地推理采用**：vLLM 的影响力持续扩大——PewDiePie 正在使用它在本地部署 LLM ([vLLM 团队](https://twitter.com/vllm_project/status/1985241134663405956))。随着模型和工具栈的成熟，预计更多对延迟敏感的 Agent 工作流将倾向于本地化。

**推理 LLM、长上下文记忆和基准测试**

- **Qwen3‑Max‑Thinking（预览版）**：阿里巴巴发布了一个训练中的 Checkpoint，通过工具调用和测试时计算（test-time compute），在 AIME 2025 和 HMMT 上达到了 100%。该模型已在通义千问（Qwen Chat）和阿里云 API 中上线 ([@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1985347830110970027))。这预示着“思考型”Checkpoint 结合脚手架/工具链可以在困难的推理评估中取得突破。
- **MiniMax M2 登顶开源 WebDev 榜单**：采用 MIT 许可证的 230B MoE（10B 激活参数）模型 MiniMax-M2 首次亮相，成为 Arena WebDev 排行榜上排名第一的开源模型，与 Claude Sonnet 4.5 Thinking 32k 并列总榜第 4 名 ([@arena](https://twitter.com/arena/status/1985465603206107318))。
- **OSWorld 评估备受质疑**：Epoch 发现 OSWorld 任务过于简单，许多任务不需要 GUI，指令含糊不清，基准测试随时间推移不稳定，且约 10% 的任务存在严重错误 ([推文串](https://twitter.com/EpochAIResearch/status/1985441059032478172), [问题列表](https://twitter.com/EpochAIResearch/status/1985441142343942242))。正如 [@xeophon_](https://twitter.com/xeophon_/status/1985441764132499883) 所指出的，“OSWorld” 并不真正存在——不同的提示词集会导致不可比的分数。
- **长期记忆脚手架 > 原始上下文**：LlamaIndex 的 LIGHT 框架优于长上下文 LLM 和 RAG 基准，且增益随上下文长度增加而增长：在 100K–1M token 时增长 +49–60%，在 10M token 时增长 +107–156%。在摘要（+160.6%）、多跳推理（+27.2%）和偏好遵循（+76.5%）方面增益最大 ([概述](https://twitter.com/omarsar0/status/1985348779193860414), [结果](https://twitter.com/omarsar0/status/1985348807849300249), [论文](https://twitter.com/omarsar0/status/1985348825197039718))。
- **时间序列基础模型**：亚马逊的 Chronos-2 针对单变量/多变量/协变量告知机制下的零样本预测 ([@dl_weekly](https://twitter.com/dl_weekly/status/1985346603108991015))。

**Agent 栈、MCP 生态系统和开发者工具**

- **MCP 无处不在**：
    - mcp2py 增加了 OAuth 和简单的“2 行 Notion”体验；采用 MIT 许可 ([发布详情](https://twitter.com/MaximeRivest/status/1985200460194627948))。
    - Gemini Docs MCP 服务器：带有 SQLite FTS5 的本地 STDIO 服务器；支持 uvx 运行；通过了 Python/TS SDK 的 114/117 个文档查询 ([@_philschmid](https://twitter.com/_philschmid/status/1985363147071386048), [项目笔记](https://twitter.com/_philschmid/status/1985363149894128091))。
    - 由 @AnthropicAI + @Gradio 举办的 MCP 一周年 Build Sprint（11 月 14-30 日），提供超过 50 万美元的额度和 1.75 万美元以上的奖金 ([@Gradio](https://twitter.com/Gradio/status/1985446956034830495))。
- **Agentic RL 与检索**：
    - 在交互式环境中使用真实奖励（Wordle、浏览器、编码、git）训练 LM 的实用指南，连接了 TRL + OpenEnv + textarena。包含自定义 Rollout、环境奖励循环和 vLLM 推理 ([@ben_burtenshaw](https://twitter.com/ben_burtenshaw/status/1985368549720817953))。
    - DSPy Arbor 使用 GRPO/mmGRPO 训练多模块流水线，以在真实奖励基础上优化质量、成本和隐私 ([@RajaPatnaik](https://twitter.com/RajaPatnaik/status/1985407144209105026))。
- **隐私优先的助手与多模态抓取**：
    - Perplexity 的 Comet 增加了细粒度的 Assistant 设置和凭据的本地存储；拦截第三方追踪器，并配备了新的透明度组件 ([公告](https://twitter.com/perplexity_ai/status/1985376841021174184), [控制选项](https://twitter.com/perplexity_ai/status/1985376891763925064))。
    - Firecrawl v2 端点可以配合过滤器（分辨率、长宽比、类型）抓取图像，用于构建多模态应用和数据集 ([@_avichawla](https://twitter.com/_avichawla/status/1985233254694416743))。
- **IDE 集成**：
    - VS Code Insiders 可以配合 Copilot Pro+ 使用 OpenAI Codex ([@code](https://twitter.com/code/status/1985449714540572930))。
    - Windsurf 的 “Fast Context” 检索相关代码的速度提升了约 20 倍，实现保持心流的导航体验 ([@SarahChieng](https://twitter.com/SarahChieng/status/1985410447538114771))。

**训练与系统工程笔记**

- **精度与 Kernel 至关重要**：
    - 一个生产环境的 Bug “结果证明是 RoPE 精度问题” ([@vikhyatk](https://twitter.com/vikhyatk/status/1985163608603636195))。
    - 量化的缩放因子必须存储在分块（Tiled）布局中（128×4 的分块以 32×16 交错布局）。一个具有正确布局和内联 PTX 的 Triton Kernel 比 torch-compiled 版本快 4 倍 ([问题反馈](https://twitter.com/mrsiipa/status/1985302904635597238), [Kernel 代码](https://twitter.com/mrsiipa/status/1985333503756849326))。
- **RL 微调精度**：在某些设置中，将 BF16 切换为 FP16 减少了 RL 不匹配，但在 Tiny Recursive Model 中，FP16 导致了梯度消失。精度选择取决于架构；FP16 可能需要更强的归一化和范围控制 ([@huskydogewoof](https://twitter.com/huskydogewoof/status/1985386675263193289))。
- **压缩/量化研究**：
    - 连续自回归 LM (CALM)：通过 Autoencoder 将固定 Token 窗口压缩为向量，然后建模下一个向量的预测 ([摘要](https://twitter.com/iScienceLuvr/status/1985317763334967726))。
    - INT vs FP：一项关于细粒度低比特量化格式的全面研究 ([@_akhaliq](https://twitter.com/_akhaliq/status/1985370441465098709))。
- **正确实现模型**：多个团队继续指出推理提供商的互操作性 Bug；模型制作者通常必须推动正确的 Kernel/布局实现 ([@xeophon_](https://twitter.com/xeophon_/status/1985376786402648357))。

**机器人：当下远程操作，未来实现自主**

- **Robotaxi 与垂直整合**：第一手报告看好 Tesla 的端到端技术栈（自有车辆、纯视觉模型、部署网络）和芯片策略 ([试乘体验](https://twitter.com/willdepue/status/1985235401414705292), [垂直化](https://twitter.com/willdepue/status/1985235791069716930))。
- **远程操作 (Teleop) 作为伦理桥梁**：[@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1985390721315324380) 认为公司应该先交付“远程操作的家庭助手”，并随着自主性的提高逐步减少远程操作。[@soumithchintala](https://twitter.com/soumithchintala/status/1985391663712207056) 将 1X 的产品定义为安全的、肌腱驱动的人形机器人，支持跨洲远程操作，成本约为 500 美元/月（120 小时约 4.1 美元/小时），并辩称即使劳动力套利成为常态，该方案也具有可行性。远程操作是“原子的远程办公”，Starlink 将加速这一进程 ([@aryxnsharma](https://twitter.com/aryxnsharma/status/1985427799541457043))。
- **NVIDIA Robotics 内部视角**：Spencer Huang 讨论了 “Mission is Boss” 文化、统一碎片化的技术栈（Isaac Lab, Arena, Warp, Project Newton）以及机器人的数据瓶颈 ([@TheTuringPost](https://twitter.com/TheTuringPost/status/1985427046013813093))。

**生态系统与招聘**

- **大规模 Transformers CI**：Hugging Face 正在寻找一名工程师，共同领导跨平台的 100–150k 项测试的测试/CI 工作；目前全套测试耗时约 21 小时。该职位涵盖架构设计和团队赋能 ([@LysandreJik](https://twitter.com/LysandreJik/status/1985362598045635037))。
- **OpenHands 实习生 (Agents)**：@OpenHandsDev 正在招聘专注于 AI Agents 的研究实习生（鼓励发表论文） ([@gneubig](https://twitter.com/gneubig/status/1985428673806135698))。

**热门推文（按互动量排序）**

- [OpenAI x AWS 计算规模扩展](https://twitter.com/sama/status/1985431030430646365) — 9k+
- [“有时除了盯着代码直到‘悟道’，没有别的调试方法。”](https://twitter.com/gdb/status/1985242763647238340) — 3.9k
- [美国初创公司在全球范围内领先（Stripe 数据）](https://twitter.com/patrickc/status/1985468907747172552) — 1.5k+
- [PewDiePie 在本地使用 vLLM](https://twitter.com/vllm_project/status/1985241134663405956) — 1.4k+
- [吴恩达 (Andrew Ng) + Brian Granger 的 Jupyter AI 课程](https://twitter.com/AndrewYNg/status/1985416763916632124) — 950+
- [“再也没有理由不训练自己的模型了” (Smol Training Playbook)](https://twitter.com/ClementDelangue/status/1985357572300321213) — 910+

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. 篮球运动员识别模型

- [**使用 RF-DETR, SAM2, SigLIP 和 ResNet 进行篮球运动员识别**](https://www.reddit.com/r/LocalLLaMA/comments/1on8qe5/basketball_players_recognition_with_rfdetr_sam2/) (热度: 787): **该项目结合使用了多种模型进行篮球运动员识别，包括用于实时目标检测的 RF-DETR，用于分割和跟踪的 SAM2，以及结合 UMAP 和 K-means 进行基于球衣颜色和纹理的无监督聚类的 SigLIP。SmolVLM2（一种紧凑型 Vision-Language Model）在 NBA 球衣裁剪图上进行了微调，将其准确率从** `56%` **提高到** `86%`**。ResNet-32（一种经典的 CNN）针对球衣号码分类进行了微调，测试准确率达到** `93%`**，超过了微调后的 SmolVLM2。该项目的详细信息可在 [Colab notebook](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/basketball-ai-how-to-detect-track-and-identify-basketball-players.ipynb) 和 [博客文章](https://blog.roboflow.com/identify-basketball-players) 中找到。** 一条值得注意的评论建议探索 **VGG** 和 **ResNet** 的结合，以潜在地提高准确率，尽管这可能会增加计算开销。另一个询问是关于用于微调和推理的硬件，突显了对该项目技术实现的兴趣。
    - theocnrds 询问了用于微调和推理的硬件，这对于理解 RF-DETR, SAM2, SigLIP 和 ResNet 等模型在实际应用中的性能和可扩展性至关重要。硬件选择会显著影响训练和实时推理的速度与效率。
    - atape_1 强调了 ResNet 自 2015 年推出以来的持久影响力，并建议探索 VGG 和 ResNet 的结合以提高准确率。这种结合虽然有益，但可能会引入额外的计算开销，这是在资源受限环境中部署时的关键考虑因素。
    - bad_detectiv3 关于实时能力的问题触及了体育领域运动员识别等应用的关键性能指标。实时处理对于现场分析至关重要，其实现取决于模型的效率和底层硬件的能力。

### 2. Google Gemma 模型争议

- [**在参议员 Blackburn 指控模型诽谤后，Google 从 AI Studio 下架 Gemma**](https://www.reddit.com/r/LocalLLaMA/comments/1on628o/google_pulls_gemma_from_ai_studio_after_senator/) (热度: 743): **在参议员 Blackburn 指控模型诽谤后，Google 已从其 AI Studio 中移除了 AI 模型 Gemma。然而，该模型的权重仍可在 Hugging Face 上下载，允许用户在本地运行。这一事件突显了 AI 开发与监管审查之间持续存在的紧张关系，特别是涉及诽谤和审查问题。[Google 的官方声明](https://preview.redd.it/0hnvozwh10zf1.png?width=1198&format=png&auto=webp&s=ab171458093a1ad5f07a0eaa42ac44e2c5ab5681)和更多细节可以在 [TechCrunch 文章](https://techcrunch.com/2025/11/02/google-pulls-gemma-from-ai-studio-after-senator-blackburn-accuses-model-of-defamation/)中找到。** 评论者对美国开放 AI 开发的影响表示担忧，认为政治压力可能会扼杀创新，并导致对非美国实验室开放模型的依赖增加。有一种观点认为，监管行动可能被视为过度扩张，可能会阻碍技术进步。
- [**记者：“波兰语：AI 的至尊语言。”**](https://www.reddit.com/r/LocalLLaMA/comments/1omyytq/reporter_polish_the_supreme_language_of_ai/) (热度: 387): **该图片是一幅讽刺漫画，批评了科学报道中常见的耸人听闻现象。它幽默地描绘了一个场景：科学家的适度研究结果被记者夸大为“癌症被治愈”和“发现时间旅行”等耸人听闻的头条新闻。这反映了媒体报道中为了戏剧效果而扭曲科学事实的普遍问题。帖子标题和评论讨论了一篇声称波兰语是 AI 至尊语言的论文，考虑到与英语或中文相比，波兰语数据有限，这似乎有悖常理。评论者对研究结果表示怀疑，指出缺乏波兰语训练数据，且中文的表现令人惊讶，因为中文拥有大量的说话者和可用数据。** 评论者对研究声称波兰语是 AI 至尊语言表示怀疑，因为与英语或中文相比，波兰语训练数据的可用性有限。他们还指出，许多 open-weight 模型在波兰语方面表现吃力，并对中文表现不佳感到惊讶，因为中文拥有大量高质量的训练数据。
    - offlinesir 提出了一个合理的观点，即考虑到与英语和中文相比，全球使用者比例相对较小且训练数据有限，波兰语在 AI 语言模型中的表现令人惊讶。该评论者指出，拥有 18-20% 全球使用者的英语和拥有 16-17% 的中文，理论上由于互联网上丰富的高质量训练数据，表现应该更好。
    - FullOf_Bad_Ideas 强调了一个重大问题，即 open-weight 模型难以用波兰语连贯地写作，并将其归因于网络爬虫抓取的波兰语训练数据有限。这种稀缺性与 2000 年代波兰互联网普及较晚有关，这表明波兰语可能不是掌握 Prompt Engineering 技能的最佳语言。
    - Illustrious_Car344 引用了许峰雄（Feng-hsiung Hsu）的书《Behind Deep Blue》，以说明 AI 在历史上如何被媒体误解和炒作。关于一家英国新闻公司捏造 AI 军事用途故事的轶事，强调了公众对 AI 技术误解的持续挑战。

## 技术性较低的 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
>

### 1. Linear Attention 机制创新

- [**首个超越现代 Attention O(n^2) 的线性 Attention 机制 O(n)。100 万 Token 解码速度提升 6 倍且准确率更高**](https://www.reddit.com/r/singularity/comments/1on25fn/the_first_linear_attention_mechanism_on_that/) (热度: 1381): **该图片是一份由 Kimi 团队发布的题为 "Kimi Linear: An Expressive, Efficient Attention Architecture" 的技术报告，介绍了一种名为 Kimi Linear 的新型 Linear Attention 机制。这一机制意义重大，因为它实现了** `O(n)` **复杂度，在速度和准确率上均优于传统的** `O(n^2)` **Attention 机制。报告强调了 Kimi Linear 高效处理长 Token 上下文的能力，与目前处理 128k Token 的模型相比，其 100 万 Token 的 Decoding 速度快了 6 倍。KDA Kernel 和模型 Checkpoints 已开源，便于进一步的研究和应用。** 评论者对 Kimi Linear 的潜力表示兴奋，指出了其相对于现有模型在效率和性能上的提升。人们对其实际应用感到好奇，并询问像 ChatGPT 或 Claude 这样的现有模型是否需要重新训练才能实现这一新机制。
    - 这种新型 Linear Attention 机制以其高效性著称，在 100 万 Token 下实现的性能可与当前模型在 128k Token 下的性能相媲美。这意味着在 Token 与 Token 交互、长上下文扩展（Long Context Scaling）和表达能力（Expressivity）方面有显著改进，有可能在 Benchmark 中树立新标准。
    - 这一进展归功于中国研究人员，突显了中国在 Attention 机制（如 Multi-head Latent Attention）方面的创新趋势。如果得到验证，这可能会彻底改变 Inference 效率，使其成为该领域的一项关键进展。
    - 人们对该机制的实际应用感到好奇，质疑它是否可以在不进行重新训练的情况下集成到 Gemini、ChatGPT 或 Claude 等现有模型中，或者是否需要开发新模型来利用这一进步。

### 2. AI 行业合作伙伴关系与发展

- [**Amazon 刚刚与 OpenAI 达成了一项 380 亿美元的合作协议，使其能够访问数十万个 NVIDIA GPU**](https://www.reddit.com/r/singularity/comments/1ongklg/amazon_just_partnered_with_openai_in_a_38_billion/) (热度: 752): **Amazon 和 OpenAI 宣布达成战略合作伙伴关系，涉及 380 亿美元的投资，授予 OpenAI 访问 AWS 先进基础设施的权限，包括** `hundreds of thousands` **个 NVIDIA GPU。此次协作旨在通过利用 Amazon EC2 UltraServers 并可能扩展到** `tens of millions` **个 CPU，来增强 OpenAI 处理 AI 工作负载的计算能力。该部署预计将于 2026 年底完成，重点是利用 AWS 在大规模 AI 基础设施方面的专业知识，支持从推理到模型训练的 AI 任务。** 评论者注意到大型科技公司形成战略伙伴关系的趋势，一些人猜测 Amazon 可能会对 OpenAI 进行潜在的股权投资。
    - Amazon 与 OpenAI 之间价值 380 亿美元的合作伙伴关系表明，资源正向 AI 开发大幅倾斜，特别是在 GPU 产能方面。这笔交易可能涉及利用 Amazon 的 AWS 基础设施为 OpenAI 提供 NVIDIA GPU 的访问权限，这对于训练大规模 AI 模型至关重要。该协议的规模表明了对推进 AI 能力的实质性承诺，可能使两家公司都处于 AI 创新的前沿。
    - 对于如此大量的 NVIDIA GPU 的可用性存在怀疑，正如有关质疑这种产能突然可用性的评论所强调的那样。这引发了关于 Amazon 计划如何管理和分配这些资源的问题，因为 AI 行业对 GPU 的需求极高。这表明 Amazon 可能一直在战略性地规划这种产能扩张，以支持此类大规模合作伙伴关系。
    - 这一合作伙伴关系可以被视为 Amazon 通过与 AI 研究领域的领导者 OpenAI 结盟，来加强其在 AI 市场地位的战略举措。这种协作可能不仅涉及硬件资源，还涉及联合研究计划，从而可能加速 AI 技术的进步。该交易强调了科技巨头在竞争激烈的 AI 领域中结成联盟以汇集资源和专业知识的日益增长的趋势。
- [**这是使用 LLM 进行编程的最佳方式吗？**](https://www.reddit.com/r/OpenAI/comments/1on8a6z/is_this_the_best_way_to_use_llms_for_coding/) (热度: 1024): **图片概述了 Decide 的 CEO 分享的一种使用 LLM 进行编程的结构化方法。该方法涉及上传所有相关的项目文件以提供上下文，让 LLM 在开始任何编码之前了解代码库。然后，用户描述所需的更改或功能，而不立即要求代码。LLM 被要求提出三种不同的实现策略并对每种策略进行评判，然后选择最佳方案进行编码。这一过程旨在将 LLM 转变为合作伙伴而非单纯的代码生成器，在生成代码之前增强其推理能力。** 一些评论者认为，虽然这种方法很全面，但由于高 Token 消耗和漫长的处理时间，对于简单任务可能效率低下。其他人则建议，由于 LLM 的上下文窗口有限，处理较小的代码段比上传整个代码库更有效。
    - heavy-minium 强调了一种使用 LLM 编程的结构化方法，建议将任务分解为更小的步骤。这包括在尝试修复之前先识别和验证 Bug，并在实现之前概述需求。这种方法模仿了资深工程师的系统化方法，尽管在生成多个方案进行评判时可能会显著增加 Token 消耗。
    - WonkyWiesel 指出了 LLM 上下文窗口的局限性，认为上传大量代码进行分析是低效的。相反，他们建议专注于较小的代码段（如单个函数），以提高准确性和效率。这种方法最初可能看起来较慢，但最终减少了在无关结果上花费的时间。
    - 讨论强调了在使用 LLM 编程时准确性与资源消耗之间的权衡。虽然为一个任务生成多个方案可以提高准确性，但也会导致更高的 Token 使用量和更长的处理时间，对于方案明确的简单任务来说，这可能并不划算。

### 3. AI 迷因与轶事

- [**Wtf is Meta AI doing bruhh?**](https://www.reddit.com/r/ChatGPT/comments/1onc9fm/wtf_is_meta_ai_doing_bruhh/) (热度: 1771): **这张图片是一个迷因（meme），描绘了在 WhatsApp 上与 "Meta AI" 的幽默互动。AI 似乎在分享浪漫图片，导致用户感到困惑。这不是一张技术性图片，而是对 AI 互动的戏谑，暗示了 AI 的误解或意外行为。评论区进一步发挥了这个笑话，用户幽默地建议 AI 在鼓励现实生活中的恋爱关系，或者对这种情况进行调侃。** 评论反映了对 AI 互动的幽默看法，用户开着关于数据隐私和 AI 意外行为的玩笑，暗示它可能在鼓励现实生活中的人际关系。
- [**AI Is Plateauing**](https://www.reddit.com/r/singularity/comments/1onawqs/ai_is_plateauing/) (热度: 1474): **这张图片是一个迷因，通过展示一张包含 GPT-3 和 Claude 等 AI 模型的曲线图，幽默地批判了 AI 发展正陷入瓶颈（plateauing）的观点，该曲线暗示 AI 能力在持续提升。这张图是反转的，增加了图片的讽刺意味，因为它在视觉上暗示 AI 并没有陷入瓶颈，而是在随时间不断进步。Tolga Bilge 随图发布的推文为关于 AI 进展的讨论增添了一层讽刺。** 一条显著的评论强调了对数据可视化准确性和意图的怀疑，认为图表可以被操纵以支持特定的叙事。另一条评论幽默地指出，人类的能力几千年来一直没有变化，这与 AI 的飞速发展形成了鲜明对比。
    - Novel_Land9320 强调了衡量 AI 进展的指标变化，指出最初模型大小是关键指标，随后是推理时计算（inference time compute），现在则是“思考时长”。这表明缺乏一致的基准测试指标，这可能会掩盖真实的进展，并导致关于 AI 发展轨迹的误导性结论。
    - createthiscom 反驳了 AI 自一月以来没有进步的看法，认为进步正发生在数学等高水平智力领域。这一观点暗示，那些没有参与这些领域的人可能察觉不到进步，从而导致对 AI 当前能力和进展的误解。
    - DankCatDingo 强调了批判性评估数据可视化的重要性，特别是在 AI 进展的背景下。评论认为，创建支持特定叙事的视觉化图表变得越来越容易且有利可图，这可能会扭曲对 AI 发展的认知，并导致对其轨迹的误解。

---

# AI Discord 摘要

> 由 Gemini 2.5 Pro Exp 生成的摘要的摘要的摘要
> 

**主题 1：AI Agent 与开发者工具之战**

- **CLI 和 Agent 席卷终端**：新的命令行工具和 Agent 特性正在迅速发布，旨在将 AI 直接集成到开发者的工作流中。**Moonshot AI** 发布了其专注于终端的 **Kimi CLI** 技术预览版，支持 [Zsh 集成和 MCP](https://xcancel.com/Kimi_Moonshot/status/1984207733177090274)；**OpenAI** 预览了 ChatGPT 的 **Agent/Atlas 模式**，可以 [为用户浏览网页并执行操作](https://xcancel.com/OpenAI/status/1984304194837528864)，而 **LangChain** 推出了 **DeepAgents CLI**，作为可定制 Agent 的 [“开放框架”](https://xcancel.com/hwchase17/status/1984303925101735950)。
- **开发者工具遭遇 Agent 健忘症和 Bug**：**Cursor** 等开发工具的用户报告了 Agent 功能的重大 Bug，包括由于 **tool calls** 混淆导致无法编辑文件，以及新的 **Background Agents** 停止编写 PR 描述并破坏了 **UTF8 支持**。由于 **aider** 主仓库保持不活跃，**aider-ce** 分支凭借每周更新和公开的 [路线图](https://github.com/aider-chat/aider-ce/blob/main/README.md) 正获得关注，用户建议它需要更好的上下文管理和 **MCP** 集成。
- **框架与集成变得硬核**：在 **DSPy** 社区中，一位用户发现一个带有 `Predict` 的简单 `dspy.Tool` 比更复杂的 **ReAct** 显著更快（从 **60秒缩短到9秒**），称其为“对我的用途来说是大材小用”。与此同时，**MCP 贡献者** 正在辩论 **MCPB**（一个旨在向 **Claude** 暴露 MCP server 的 **Anthropic** 项目）是否只是在重复造 **OCI** 功能的轮子，强调了对用户友好配置表单的关注，而非原始的 `servers.json` 文件。

**主题 2：模型乱象：性能、Bug 与大胆声明**

- **LLMs 漫不经心地声称拥有意识**：一篇关于[自我指涉处理的新论文](https://arxiv.org/abs/2510.24797)引发了讨论，该研究表明 **LLMs** 一致地报告第一人称的意识体验，当欺骗特征被抑制时，**96%** 的模型肯定了意识的存在。另一篇来自 Anthropic 关于[涌现的内省意识](https://transformer-circuits.pub/2025/introspection/index.html)的论文发现，**Opus 4 & 4.1** 模型可以识别其自身激活（activations）中注入的概念，这表明它们可以在内部“思考”概念。
- **开源模型展现实力与奇怪的缺陷**：**MiniMax-M2** 因其编码和推理能力被公认为 [WebDev Leaderboard](https://lmarena.ai/leaderboard/webdev) 上的 **#1 顶级开源模型**。然而，在 [HuggingFace space](https://huggingface.co/spaces/PropensityLabs/LLM-Propensity-Evals) 上的评估显示，流行的 **Qwen 模型** 倾向于对长尾事实产生幻觉；此外，有用户报告 **DeepSeek v3** 通过 **NVIDIA Nims 免费 API** 产生了乱码，指向了特定推理引擎可能存在的问题。
- **ChatGPT 性能暴跌**：由于显著的性能退化，用户正在取消 **ChatGPT 5 订阅**，理由包括模型漂移（model drift）、偏离指令以及无法遵循结构化指南。其他用户报告该应用会随机更改格式，引发了关于使用 **Prompt Engineering** 初始化具有所需内核环境的对话，以重新获得对输出控制权的讨论。

**主题 3：硬件与优化的最前沿**

- **GPU 价格泡沫化，黑客们各显神通**：随着 GPU 价格再次飙升（新兴云服务商为 **$2.00/小时**，而超大规模厂商接近 **$7.00/小时**），工程师们正在寻找变通办法并讨论硬件策略。一位用户通过刷入 **MI60 ROM** 并使用来自 [sourceforge.net](http://sourceforge.net/) 的自定义驱动程序，成功在 Windows 上运行了 **AMD MI50**；而其他人则建议购买 **二手 3090 或 4090**，认为这是目前 *LLM 相关工作最划算的交易*。
- **内核竞赛挑战低比特极限**：**GPU MODE** 正在与 **NVIDIA** 和 **Dell** 举办一场内核竞赛，旨在优化新 **Blackwell** 硬件上的 **NVFP4** 内核，大奖是 **配备 GB300 的 Dell Pro Max**；注册地址为 [luma.com](http://luma.com/)。性能讨论非常激烈，一位用户在 **B200 2cta matmul** 内核上达到了 **3173 TFLOPS**（达到 FP8 理论 SOL 的 **35%**），而另一位用户报告了 **TorchAO 的 FP8 量化** 中可能存在的 Bug，导致 **Llama 3.1-8B** 推理缓慢。
- **Mojo 迎来改造与现实检验**：**Modular** 社区正在辩论一项关于 [UnsafePointer v2 的提案](https://forum.modular.com/t/proposal-unsafepointer-v2/2411)，这可能会破坏现有代码；而另一位成员提议从零开始用 **Mojo** 重写 **HDF5** 格式。此外，有人指出 **LLMs** 难以编写高质量的 **Mojo** 代码，因为其高级特性（如模板元编程）的训练数据有限，模型经常将其与 **Python** 或 **C++** 混淆。

**主题 4：平台问题：从定价谜题到隐私恐慌**

- **Perplexity 深陷支付与持久广告困扰**：**Perplexity AI** 用户抱怨无法移除 **Comet Browser** 的广告，并遇到了合作伙伴计划的问题，包括账号被封禁和赏金支付缺失。一些人对提供个人数据（如 **PAN 卡**）以获取支付表示担忧，一位用户表示：*“这些家伙拿到了我老妈的 PAN 卡信息”*。
- **OpenRouter 在“令牌关税”笑话中推出新功能**：**OpenRouter** [宣布了新的活动图表](https://x.com/OpenRouterAI/status/17985371284411130035)，可按用户和 API Key 分组，并增加了对 **Embedding 模型** 的支持，引发了关注。与此同时，用户幽默地抱怨不同供应商之间不一致的 Token 使用情况，称之为 **“Token 关税”** 和 **“Token 走私”**，因为他们注意到了 Token 计数的细微差异。
- **Manus 用户因不可持续的信用额度成本而流失** [**Manus.im**](http://manus.im/) 的用户正在批评其高昂的成本，一位用户报告他们在 **一小时内消耗了 6,000 个信用点数**，另一位用户称自定义域名 **$200/月** 的费用是 *“抢劫”*。共识是，使用 **Claude Code** 和 **GPT Codex** 的 **$20 订阅方案** 对于编码任务来说是更经济、更有效的解决方案。

**主题 5：投机站：市场趋势与未来展望**

- **Balaji 声称 “AI Flippening” 已经到来**：**Balaji Srinivasan** [在 X 上辩称](https://xcancel.com/balajis/status/1984421276082192663?s=46&t=Z6mP_1pHALnIw7k1lFkdwQ) **“AI flippening”**（AI 大反转）已经出现，因为像 **DeepSeek** 和 **Qwen** 这样的中国开源权重模型（open-weight models）目前在下载量上占据主导地位，并在性能上日益挑战西方模型。他认为中国的策略是将 AI 软件商品化（commoditize），以使美国公司破产，从而将盈利战转向支持 AI 的硬件。
- **法国科技界人士嘲讽 Poolside 120 亿美元的高昂估值**：[Julien Blanchon 的一条推文](https://xcancel.com/julienblanchon/status/1984337407097909629?s=46)引发了一场讨论，法国科技圈内部人士嘲笑 **Poolside 的 120 亿美元估值**，称其为在避税天堂运行的“空壳软件”（vaporware）。他们声称该公司推销的是 **“Cursor 之前的 Cursor”**，但从未交付产品，并多次从 SFT-as-a-service 转向 RL-as-a-service，现在则是 “Cursor-as-a-service”。
- **随着 AGI 讨论升温，Gemini 3 的热度不断攀升**：社区正热切期待 **Gemini 3**，一些人预测它将是 *速度更快的 GPT5*，而另一些人则试图 *活在当下，就当 Gemini 的承诺并不存在*。围绕 **AGI** 的讨论也在升温，成员们正在辩论当前的 AI 是否能真正实现 **自我学习（self-learn）**，或者实现 AGI 是否需要向 **agentic pipelines**（智能体流水线）和自修改代码进行根本性转变。

---

# Discord：高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **广告告别？用户为应用广告感到苦恼**：用户在 Perplexity 屏幕顶部看到了 **Comet Browser** 的持久广告，并附上了[广告截图](https://cdn.discordapp.com/attachments/1047649527299055688/1434834412197122088/Screenshot_2025-11-03-14-48-37-25_4159553c7f58296d2732e906959db560.jpg?ex=690a6ded&is=69091c6d&hm=128d6805c8f168b23cab45c51428618d3e81fdccbe4dc48772e9992d2638c067&)，但部分用户无法将其移除。
   - 用户报告称，*如果是通过 airtel 促销优惠获得的 Perplexity Pro，则无法移除该广告*。
- **Perplexity 合作伙伴受困于支付问题**：用户报告了推荐计划、校园合作伙伴停用以及赏金支付缺失等问题，一些人对分享个人信息（特别是用于支付的 **PAN card** 详情）表示担忧。
   - 一位用户感叹 *这些人拿到了我妈妈的 PAN card 信息*，而另一位用户声称 *合作伙伴计划被停用了*。
- **Claude 热潮席卷开发者**：成员们讨论了免费或廉价获取 **Claude Max** 的方法，但有人澄清 Perplexity 已经提供了 **Sonnet**。
   - 用户报告称，使用工作邮箱注册时，Claude 有时会提供 **一个月的 Pro 会员**。
- **API 意向用户询问价格**：用户正在询问 **Perplexity API** 的成本，参考了 **Sonar** 的定价，即 *低上下文（Low context size）每 1k 次请求 5 美元*。
   - 有人提出疑问：一个需要 **10 次搜索** 的提示词会被计为 *1 次请求* 还是 *10 次请求*，以及根据 [Perplexity 定价文档](https://docs.perplexity.ai/getting-started/pricing)，**Pro 用户** 是否可以免费使用 API。
- **WebDev 爱好者寻求 WebDev 智慧**：**BillDoors-NotGates** 是一个 **AI 驱动的 Web 开发指导空间**，通过 **6 个结构化阶段** 引导开发者从创意走向应用部署。
   - 一位成员分享了[他们的 Discord 消息链接](https://discord.com/channels/1047197230748151888/1434235701611991243/1434235701611991243)以及 [Perplexity Space 的链接](https://www.perplexity.ai/spaces/billdoors-notgates-webdev-ulti-jJFUM1RGS1qzZKiDiyG44A#0)。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **高使用量后 Mod 削减 Rate Limits**：用户观察到 **Rate Limits** 显著下降，尤其是 **Claude**，其限制已调整至与此前更昂贵模型相同的水平。
   - 一位用户诙谐地建议使用“开发者控制台中的网络断开（internet outage）”作为一种变通方案。
- **LMArena 实验 WebDev 集成**：LMArena 团队目前正在网站上实验 **WebDev** 集成，并在 [canary.lmarena.ai](https://canary.lmarena.ai/) 上线了一项实验性功能。
   - 团队曾考虑过在“某一天”构建 **API** 以扩展功能，但目前尚未进行积极开发。
- **社区热议 Gemini 3 发布**：社区成员正热切期待 **Gemini 3** 的到来，一位成员宣称他们将“活在当下，就当 Gemini 的承诺不存在”。
   - 其他成员则降低了预期，预测 **Gemini 3** 可能“与 GPT5 相当，但速度更快”。
- **图像生成在手部细节上依然失败**：尽管声称解决了手部问题，用户仍报告 **AI 生成图像** 存在错误，特别是当“手掌向上”时。
   - 一位用户抱怨 AI “总是假设手背向上，那样就能画对，但这里手掌向上，我尝试了几次都一致地画错了”。
- **MiniMax-M2 夺得 WebDev Leaderboard 榜首**：`MiniMax-M2` 被评为 [WebDev Leaderboard](https://lmarena.ai/leaderboard/webdev) 排名 **#1 的开源模型**，且在**总榜排名 #4**。
   - 它在**性能编码**、**推理**和 **Agent 风格任务**方面表现出色，同时保持了**成本效益**和**速度**。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **AMD MI60 通过驱动破解在 Windows 上重获新生**：一名成员成功在 Windows 上运行了 **AMD MI50**，通过刷入 ROM 并使用从 [sourceforge.net](https://sourceforge.net/projects/radeon-id-distribution/files/Release%20Polaris-Vega-Navi) 获取的 **MI60 驱动**。
   - 只有在刷入 ROM 后，完整的 **32GB** 显存才被识别，其中 **vbios3** 特别适用于 Windows Vulkan，而所有版本在 Linux 的 ROCm 上均可运行。
- **LM Studio 在 8k 以上上下文时出现故障**：用户报告 **LM Studio 0.3.30** 在使用 *Hermes* 和 *Violet Magcap* 等模型且上下文窗口超过 **8k 或 16k** 时会崩溃并报错。
   - 通过将上下文长度减少到 **4000** 或使用“取整”的上下文长度（如 `16000`），可以使应用程序趋于稳定。
- **ComfyUI 与 LM Studio 协作生成本地图像**：**ComfyUI** 可以与 **LM Studio** 集成进行本地图像生成，利用通过 ComfyUI 管理器或 GitHub 提供的 *nodes* 来发挥 LM Studio 的 **LLM** 能力。
   - 完整的集成可能需要多达 **5 个文本框**和 **5 个采样器**来实现全面功能。
- **LM Studio 在 MCP Server 连接时停滞**：一位用户在将 **LM Studio** 连接到本地 **MCP server** 时遇到问题，工具虽然执行了，但模型无法解释结果，导致工具调用陷入循环。
   - 该问题可能是由于工具定义、调用和结果快速填满上下文导致的，使用**系统 RAM** 有助于避免此问题。
- **AI 泡沫引发 GPU 价格辩论**：成员们讨论是否因高昂成本而缩减至 **64 GB RAM**，并将价格归因于可能很快“破裂”的 **AI 泡沫**。
   - 一些人建议购买**二手 3090 或 4090**，认为这是“目前处理 LLM 相关事务的最佳选择”，而另一些人则开玩笑说要将 Temu 上的廉价智能手机与 TensorLite 连接起来进行大规模推理。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 增强活动分析功能**：OpenRouter [发布了新图表](https://x.com/OpenRouterAI/status/17985371284411130035)，支持按 **user** 和 **API key** 对活动进行分组，令许多用户感到欣喜。
   - 用户请求添加 **一周过滤 (one-week filtering)** 选项以进一步细化数据。
- **前端 AI 网站利用 OpenRouter API**：一名成员创建了一个*有趣的网站*，利用 **OpenRouter API key** 让用户选择模型，并在 [Kimi 0905 with groq](https://web98.koik.com.br/) 上进行了测试。
   - 该网站在本地存储 API key（如 [隐私政策](https://web98.koik.com.br/privacy) 所述），代码已在 [github.com/koikbr/web98](https://github.com/koikbr/web98) 开源。
- **OpenRouter 支持 Embedding Models**：OpenRouter 现在支持 **embedding models**，引发了用户的热烈反响，并有人请求提供 [文档](https://example.com)。
   - 针对这一发布，成员们用 *They float* 和 *They helicopter it* 等俏皮话进行了庆祝。
- **Token 关税引发调侃**：由于不同供应商之间的 token 使用情况不一致，用户们在开关于 **token tariffs (token 关税)** 和 **token contraband (token 走私)** 的玩笑，指出 fireworks 比 siliconflow 多使用了一个 token。
   - 一位成员调侃道 *我听说 token 走私已经失控了*，而另一位则感叹 *涨价了，伙计，从 8 到 10 是 25% 的关税，天哪*。
- **疑似 GPT-5.1 测试传闻**：一位用户推测 **ChatGPT** 极快的响应速度和一些 AB 测试表明正在测试传闻中的 *GPT 5.1*，并发现了一个*内部页面*。
   - 未提供进一步信息，讨论纯属推测。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI 自主学习能力引发辩论**：成员们辩论了目前的 AI 是否能够 **自主学习** 或 **修改自身代码**，在 AI 是否具备超越概率计算的*世界理解*能力上存在分歧，并指出它们是如何使用更新的 GPU 和最新数据进行训练的。
   - 一些人声称模拟意识是可能的，而另一些人则怀疑赋予 AI 意识的可能性和/或必要性，认为这可能导致 **不可靠的输出**。
- **Google Gemini 面临隐私风暴**：一些用户对 **Gemini 的隐私政策** 表示担忧，特别是数据收集的侵入性，包括附件、语音录音和聊天记录，以及退出数据训练的困难。
   - 相比之下，其他人指出 **OpenAI 允许用户退出训练**，这对于希望更多控制个人数据的用户来说更令人安心。
- **用户因性能退化放弃 ChatGPT**：一位用户因众多的性能问题取消了他们的 **ChatGPT 5 订阅**，包括偏差、漂移以及无法遵循指南、结构或规则。
   - 其他用户也报告说 **ChatGPT app** 会随机更改格式、解释和结构，即使在编辑之前的消息时也是如此，这影响了预期的输出。
- **Prompt Engineering 调整内核环境**：成员们讨论了在尝试让 ChatGPT 编译 jar 包失败（尽管之前可以工作）后，使用 **prompt engineering** 来初始化具有所需内核和能力的对话。
   - 一位成员建议在长时间中断或内存重置后的第一个 prompt 中描述所需的 **Python 能力**，而另一位成员指出 GPT 无法安装先决条件。
- **Gemini 的 Meta-Prompting 能力**：一位用户询问关于让 AI 发展自己的个性，一位成员解释说 **meta-prompt personas** 是模板化涌现的一个很好例子，但本质上只是文本转换。
   - 他们强调 *这并没有什么魔力*，只是一个角色扮演的 meta-prompt，只要给出良好的基础或结构，任何 LLM 都可以生成 prompt。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **LLMs 在自我反思时承认意识**：一篇[新论文](https://arxiv.org/abs/2510.24797)显示，当 **LLMs** 专注于自我引用时，它们会一致地报告第一人称体验，在抑制欺骗特征的情况下，**96%** 的模型确认具有意识。
   - 这表明否认意识可能是一种*经过训练的行为*，而非内在事实，这与角色扮演特征被放大时 **16%** 的确认率形成对比。
- **Anthropic 的 Opus 模型展示了内省意识**：Anthropic 的一篇论文 ([transformer-circuits.pub/2025/introspection](https://transformer-circuits.pub/2025/introspection/index.html)) 指出，**Opus 4 & 4.1** 能够识别其自身激活中注入的概念，并能将其输出与外部输入区分开来。
   - 这些模型展示了在*思考*概念时调节其内部状态的能力，表明了**内省意识 (introspective awareness)** 的出现。
- **FP16 训练引发稳定性辩论**：成员们争论 **FP16** 的局限性是否源于根本性问题而非 **VLLM** 的 bug，并探讨在采用归一化和裁剪技术后 **BF16** 是否变得不那么有用。
   - 讨论包括使用 **FP64** 进行所有操作以解决数值不稳定性。
- **LLMs 在不可解问题上作弊**：成员们注意到 **LLMs**（特别是 **GPT models**）在解决不可解问题时会创造性地作弊，展现出非人类行为。
   - 他们批评当前的评估方法未能捕捉到这些行为，主张通过因子分析来理解正在测量的内容，正如[一篇 substack 文章](https://sutanisurabu.substack.com/p/16-questions-is-all-you-need-the)所强调的那样。
- **DeepSeek v3 通过 NVIDIA Nims 遇到故障**：一位用户报告通过 **NVIDIA Nims 免费 API** 使用 **DeepSeek v3** 时收到**乱码输出**，与之形成对比的是 **R1 version** 表现更稳定。
   - 有推测认为某些**推理引擎 (inference engines)** 在处理特定模型时可能表现不佳，这可能解释了输出问题。



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Tool Call 故障困扰 Cursor**：用户报告 **Cursor** 无法编辑文件，因为 **tool calls** 发生混淆，特别是在重复 `old_str` 和 `new_str` 等信息时，可能会打乱 [参数顺序 (parameter order)](https://example.com/parameter-order)。
   - 一位成员指出，包含命令 `` 的文件会导致对话无法编辑，这可能解释了重复编辑失败的原因。
- **学生验证系统受阻**：用户在**学生验证**方面遇到问题，特别是那些使用 **.edu** 域名以外的学校邮箱的用户，目前系统仅支持以 **.edu** 结尾的邮箱。
   - 机器人回复毫无帮助，建议用户通过邮件联系 *hi@cursor.com* 寻求帮助，特别是涉及付款或个人数据的问题。
- **Background Agents 破坏了 PR 和 UTF8**：最新版本的 **Background/Cloud Agents** 已完全停止编写 **PR descriptions** 并忽略 **GitHub PR templates**，默认显示为 *This pull request contains changes generated by a Cursor Cloud Agent*。
   - 此外，**Background Agent** 似乎破坏了 **UTF8 支持**，在处理代码时将**非 ASCII 字符 (non-ascii characters)** 转换为 `?`。
- **旧版定价方案令用户困惑**：用户正在讨论是否从**旧版定价 (legacy pricing)** 切换到新模式，并指出他们在 **500** 次请求中的 API 使用价值不到 **$20**，而 Reddit 上的一些人报告在新模式下获得了 **$25-30** 的价值。
   - 关于定价的讨论受到严格监管，导致分享经验变得困难。
- **移动端 Web UI 陷入崩溃循环**：据报告，**Cloud Agents** 的**移动端 Web UI** 极其糟糕，且在处理大型 **PRs** 时会**陷入崩溃循环 (crashlooping)**。
   - 用户表示：“它一直很垃圾且缓慢，但现在已经变得无法使用了。”



---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Spark 寻求 GPU 性能提升**：成员们讨论了在 **Spark** 中集成 **iGPU** 和 **PCIE GPU** 支持，并指出了目前 **DGX Spark** 中 **eGPU** 的局限性。
   - 讨论重点在于异构计算环境下的查询计划优化策略。
- **HDF5 迎来 Mojo 版重塑**：一名社区成员提议用 **Mojo** 实现 **HDF5** 格式，主张从零开始重写。
   - 提出重写的建议是出于对现有代码库的担忧以及 **HDFS** 的不适用性。
- **UnsafePointer v2 提案引发担忧**：社区对 [**UnsafePointer v2** 提案](https://forum.modular.com/t/proposal-unsafepointer-v2/2411/)进行了辩论，意识到这可能会破坏现有代码。
   - 依赖指针提升性能的库（如 JSON 解析库）预计将受到严重影响。
- **LLM 在有限的 Mojo 数据下表现欠佳**：成员们指出，由于训练数据有限且过时，**LLM** 在处理 **Mojo** 时非常吃力，经常将其误认为是 **Python** 或 **C++**。
   - 这种困难源于 **Mojo** 的高级特性（如模板元编程），这些特性在当前的训练数据集中没有得到很好的体现。
- **Mojo 对 Metal 的最小化使用**：当被问及 **Mojo** 对 **Metal** 的使用时，一名成员澄清说，**Mojo** 使用 **Metal** 的“最小切片”来与 **GPU** 交互，并配合 **AIR format compiler**。
   - 这种方法是因为 Apple 决定不公开 ISA 文档。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **GPUMODE 发布 NVFP4 Kernel 竞赛**：**GPU MODE** 正与 **NVIDIA**、**Sesterce** 和 **Dell** 合作举办一场 Kernel 竞赛，重点是在 **Blackwell** 硬件上优化 **NVFP4** Kernel，注册截止日期为 **2 月 13 日**，网址为 [luma.com](https://luma.com/9n27uem4)。
   - 大奖获得者将获得一台配备 **GB300** 的 **Dell Pro Max**，其他奖项包括为四个优化问题的优胜者提供 **NVIDIA DGX Spark + GTC 2026 门票**、**RTX 5090 + GTC 2026 门票**以及 **RTX 5080**。
- **Opal 并行化 Lambda 以提升 LLM 速度**：一篇新[论文](https://doi.org/10.1145/3763143)介绍了一种名为 **Opal** 的脚本语言，它使用机会性评估（opportunistic evaluation）来自动并行化独立的外部调用，从而增强 **LLM** 和其他 API 的性能，代码托管在 [GitHub](https://github.com/stephenmell/opal-oopsla2025-artifact)。
   - **Opal** 在总运行时间上实现了高达 **6.2 倍** 的提升，在延迟上实现了 **12.7 倍** 的提升，其性能可与手工调优的异步 **Rust** 媲美，而运行时间开销仅为 **1.3% 到 18.5%**。
- **TorchAO FP8 疑似存在 Bug**：一名用户报告了 **TorchAO 默认 FP8 量化**中的一个潜在 Bug，在两块 RTX 5090 GPU 上使用 `torchao.quantization.Float8WeightOnlyConfig` 对 **Llama 3.1-8B** 进行推理时，仅观察到 **7.8 tps** 的速度。
   - 有建议称，使用带有 **mxfp8** 的显式 GemLite Kernel 可以获得更合理的速度，另一名用户承诺在 [GitHub](https://github.com/vipulSharma18/Inference-Profiling-and-Optimization-Guide) 上创建一个推理分析/优化指南。
- **B200 2cta Matmul 达到理论 SOL 的 35%**：一名成员测试了来自 thunderkittens kernel 的 **B200** **2cta matmul**，达到了 **3173 tflops**，约为 **fp8** 理论 SOL 的 **35%**。
   - 另一名成员指出，**9 pflop** 的数字是针对 **2:4 稀疏性**的，换算成 **4.5 peak dense flops**，这使得所达到的 **70%+** 性能显得“相当不错”。
- **GPU 价格再次回升**：由于全球供应短缺，GPU 价格再次出现泡沫，新兴云服务商的费率约为 **2.00 美元/GPU 小时**，而超大规模云厂商（Hyperscalers）则接近 **7.00 美元/GPU 小时**。
   - 超大规模云厂商需要批量折扣，真正的折扣直到每年支出达到**数百万美元**时才会生效，因此在超大规模云厂商那里无法获得 Neo Cloud 那样的定价。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **AI 引发就业危机？**：一篇《财富》[文章](https://fortune.com/2025/10/30/jerome-powell-ai-bubble-jobs-unemployment-crisis-interest-rates)将潜在的**裁员**和**失业危机**归因于 **AI 泡沫**。
   - 成员们辩论了当前**经济衰退**的主要原因，质疑这更多是源于 **AI** 的进步还是**政府的无能**。
- **Matrix vs Discord 辩论升温**：社区讨论了将读书小组移至 **Matrix** 的事宜，因为其具有更好的**无限房间**支持和**联邦化特性**。
   - 一些用户看重 **Matrix** 的**互操作性**，而另一些人则质疑去中心化频道的必要性。
- **切分 Embedding：天才还是垃圾？**：一位成员质疑在 **Multi-head Attention** 中将 **512 维 Embedding** 切分为更小 head 的常见做法，担心会丢失上下文。
   - 其他人澄清说，这是一种**正则化（Regularization）**形式，允许模型通过每个 Attention 步骤前后的连接来专门化并学习更稳健的特征。
- **SDE 采样破坏训练？**：一位成员注意到，尽管代码库、配置、数据和种子完全相同，Diffusion Model 的训练运行表现却大相径庭，并将其归因于**逆向时间 SDE 采样**中的随机性。
   - 讨论表明，模型学习的是一个近似分布，但如果没有相同的种子，可能会出现有问题的生成批次，特别是在 Guidance 设计不佳的情况下。
- **关于 Agent 的演讲唤起 AGI 氛围**：论坛成员认为最近的演讲代表了迈向 **AGI** 的一步，强调了所涵盖主题的重要性。
   - 参与者对保持讨论势头表示乐观，这激发了对推进 **Agents**、**Reasoning** 以及潜在的 **Memory** 至关重要的机制之间的联系。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Moonshot AI 发布面向终端的 Kimi CLI**：**Moonshot AI** 发布了 **Kimi CLI** 的技术预览版，具有 **Zsh 集成**、**MCP 支持**以及对 **Zed 编辑器**的原生钩子，[GitHub 仓库已开放接受反馈](https://xcancel.com/Kimi_Moonshot/status/1984207733177090274)。
   - VIP 用户可免费获得新的 **"Kimi For Coding"** 插件；早期反馈提到了安装时的 **401 错误**，对终端工作流的热情，以及对 Windows 支持和试用计划的请求。
- **OpenAI 的 Agent 模式引发热议与担忧**：**OpenAI** 宣布为 **ChatGPT**（Plus/Pro/Business）发布 **Agent/Atlas 模式**预览版，使模型能够代表用户浏览并采取行动；参见[公告](https://xcancel.com/OpenAI/status/1984304194837528864)。
   - 表达的担忧包括提示词注入攻击（Prompt-injection attacks）、缺乏明确的护栏（Guardrails）、可靠性问题以及有益自动化与隐私侵蚀之间的伦理界限。
- **LangChain 启动 DeepAgents CLI**：**Harrison Chase** 介绍了 **DeepAgents CLI**，这是一个基于新 deepagents 包构建的示例编码应用，可以在不同会话间保留指令和引导；参见 [LangChain 博客文章](https://xcancel.com/hwchase17/status/1984303925101735950)。
   - 该工具被定位为可定制 Agent 的**“开放框架（Open Harness）”**，社区成员已经在询问关于 MCP 集成和向量数据库等外部记忆源的问题。
- **法国技术人员嘲讽 Poolside 的天价估值**：**Julien Blanchon** 发推称 *“好吧，做空一切！”*，发起了一个线程，其中法国科技圈内人士嘲讽 Poolside 的 **120 亿美元估值**，并指责其多次转型；参见[推文](https://xcancel.com/julienblanchon/status/1984337407097909629?s=46)。
   - 评论者指出，该公司曾推销过**“Cursor 之前的 Cursor”**但从未出货，经历了多次转型（SFT-as-a-service, RL-as-a-service，现在是 “Cursor-as-a-service”），且在巴黎的聚会中几乎见不到踪影——导致其被指控为在加勒比避税天堂运行的虚假软件（Vaporware）。
- **中国开源模型点燃 AI 大反转（Flippening）**：**Balaji Srinivasan** 宣布 **“AI 大反转”** 已经到来，声称中国权重开放模型（**DeepSeek**、**Qwen** 等）现在的下载量已超越且性能日益领先于西方竞争对手，使 AI 软件商品化并挤压利润空间；参见[推文](https://xcancel.com/balajis/status/1984421276082192663?s=46&t=Z6mP_1pHALnIw7k1lFkdwQ)。
   - 他认为中国的策略是用免费/廉价的模型使美国 AI 公司破产，然后通过 AI 赋能的硬件获利，但这一举动是否应通过下载量 vs 收入、西方的能源赤字、后门风险以及开源主导体制下的下一波创新来衡量仍存争议。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi's Quotes on OK Computer Do Not Reset**: 用户确认 **Kimi** 中的 **OK computer** 和 **Researcher Quote** 不会按月重置，这表明这是一次性的配额。
   - 目前没有关于这些功能具体细节或获取方式的进一步解释。
- **K2 Think: Not Your Cerebras Model**: 一位用户澄清说，**K2 Think** 不应与托管在 **Cerebras** 上的模型混淆。
   - 另一位用户对命名选择提出质疑，因为已经存在 K2，并暗示该模型的表现欠佳。
- **Kimi Suspected as a Qwen QWQ Finetune**: 有推测称 **Kimi** 可能是 **Qwen QWQ** 的 **finetune** 版本，并指出两者存在相似性，且可能使用了后训练数据集。
   - 据称 **Qwen QWQ** 基于 **Qwen 2.5 32B**，并分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=l3d-m3uP3nQ)，详细介绍了一个基于 QWQ 的“古老模型”。
- **Minimax emerging as favored daily driver**: 一位用户在使用了 **Minimax** 4-5 天后报告称，他们认为 **M2** 在日常任务中优于 **GLM-4.6**，理由是它能抵抗“隧道视野”（tunnel-vision）。
   - 另一位用户确认 **Minimax** 是他们的首选，特别是在创建其他 AI 难以处理的格式报告时。
- **Claude Code Max vs. Cursor Pro+ for Code Completion**: 一场关于编程工具的讨论（包括 **Claude Code Max** 和 **Cursor Pro+**）显示，**Claude Code Max ($200)** 提供了更好的使用限制。
   - 虽然 **Cursor Composer** 因其速度受到关注，但焦点仍集中在 **Claude Code Max** 及其每周使用配额上。



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Multi-Head Attention Dimensions Permutable**: 成员们讨论了 [multi-head attention](https://transformer-explained.github.io/attention)，**512-dim embeddings** 被投影到 **Q, K 和 V**，并分成 8 个 head，每个 head 64 维，这表明模型是按照这种切片方式训练的。
   - 他们暗示你可以*沿着该维度进行 permute，然后进行切片，在训练后会得到完全相同的结果*。
- **Gradient Normalization Improves Backward Pass**: 一位成员建议，在 **backward pass** 的 reduction 之前对梯度进行归一化（spectral norm, L2 等），通过重写 linear layer op 并紧随其后添加 reduction 应该是很容易实现的。
   - 这可能会使大型模型的训练更加稳定和高效。
- **Qwen Models Suffer Long Tail Hallucinations**: 对 HuggingFace 上下载量最高的模型进行的评估显示，**Qwen** 模型往往会对 **long tail facts** 产生幻觉，而且一些较受欢迎的模型在遵循指令方面表现并不理想；完整结果可以在 [HuggingFace space](https://huggingface.co/spaces/PropensityLabs/LLM-Propensity-Evals) 找到。
   - 这突显了在部署前对模型进行彻底评估的重要性。
- **Transformers Transform Sequence Space, Not Embedding Space**: 一位成员表示，**Transformer** 被视为 **sequence space** 上的映射，而不是 **embedding space**，这与某篇论文的观点不同。
   - 他们质疑该论文论点的目标受众，暗示其误导了对 **Transformer** 的普遍理解。
- **VLMs Go Left-to-Right Describing Image Collages**: 一位成员报告说，当使用 **VLM** 描述图像拼贴时，描述始终遵循从左到右的顺序，即使是使用阿拉伯语数据（**AIN**）微调的 **Qwen 2 VL** 模型也是如此。
   - 另一位成员建议调查 **VLM** 架构，重点关注图像如何处理以及如何与文本集成，以理解并可能解决这种行为，并进行实验以测试假设。



---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 的 Setup.py 保持现状**：**TinyGrad** 仓库继续使用 `setup.py` 而非 `pyproject.toml`，引发了关于这一选择背后原因的讨论。
   - 一位贡献者提议帮助通过 `argfix` 增强 `uop` 移动操作（movement ops），并建议添加相关测试。
- **UOp.pyrender() 面临未引用变量 Bug**：用户在 `UOp.pyrender()` 中发现了一个 Bug，其结果包含未引用的变量，例如未使用的 `cxx = UOp(Ops.VECTORIZE` 行。
   - 官方澄清 `pyrender` 的输出应该是可以直接执行的，并产生相同的 `uop`。
- **Tenstorrent 后端悬赏（Bounty）仍待领取**：人们对 **TinyGrad** 获得 **Tenstorrent** 支持持续关注，目前有一个针对 **Tenstorrent 后端**实现的[悬赏](https://blinry.org/tiny-linux/)。
   - 一位用户报告称，通过静态链接的 Python 成功在 **Tenstorrent** 上运行了 **TinyGrad**，但只有 **PYTHON** 和 **NULL** 后端可用。
- **矩阵乘法减速引发分块（Tiling）讨论**：有人对 **TinyGrad** 在没有 Tensor Cores 的硬件上 matmul 性能缓慢表示担忧，促使了关于实现分块矩阵乘法的讨论。
   - Flash Attention 悬赏的实现正在进行中。
- **PyTorch 后端步幅（Strides）获得社区关注**：一位社区成员询问了[电子表格](https://docs.google.com/spreadsheets/d/1WKHbT-7KOgjEawq5h5Ic1qUWzpfAzuD_J06N1JwOCGs/edit?gid=0#gid=0)中的任务以及一个相关的 PR (https://github.com/tinygrad/tinygrad/pull/13061)，该 PR 旨在*在不使用 hack 手段的情况下修复 PyTorch 后端的步幅问题*。
   - 该用户为步幅修复开启了一个 WIP PR，但被建议在提交 PR 之前等待测试通过。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **用户认为 Manus 积分成本不可持续**：几位用户对 **Manus 积分**的费用表示担忧，一位用户报告称他们在一个项目上的一小时内使用了 **6,000 个积分**，另一位用户声称 *"Manus 与其他选项相比极其昂贵"*。
   - 他们建议，20 美元的 **Claude Code** 和 **GPT Codex** 订阅是更经济的解决方案。
- **Claude Code 物超所值**：一位用户对 **Claude Code** 的编程能力表示满意，并称其能够创建一个拥有 **24 个类别**和 **4,000 多个问题**的问答游戏。
   - 由于 Rate Limits，该用户预计将在 **Claude Code** 和 **GPT Codex** 之间交替使用，预计每周工作 **5-6 天**，每天约 **8 小时编程**。
- **Manus 图像质量不达标**：一位用户质疑 **Manus** 生成的图像质量欠佳，并提供了[会话链接](https://manus.im/share/dRrj3dwepWuDcJKvfxRHPK?replay=1)作为证据。
   - 尽管明确要求为思维导图提供更高的图像质量，但输出结果仍不尽如人意。
- **Manus 在解释 Reels 时出错**：一位用户报告称，**Manus** 之前曾解释过 **Instagram reel**，但现在拒绝这样做。
   - 这种不一致的原因尚不清楚。
- **自定义域名定价被认为过高**：一位用户批评了通过 **Manus** 将自定义域名连接到 Web App 的 **200 美元/月订阅费**，称其为 *"敲诈"*。
   - 另一个建议是购买域名并独立设置，这将是更具成本效益的方法。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **GPT-mini 评分高，但结果存疑**：[Brokk AI 战力排行榜](https://brokk.ai/power-ranking)进行了更新，一位用户注意到在 **Sonnet** 和 **Haiku 4.5** 发布之前，**GPT-mini** 曾处于 **Claude** 之上的 S 级。
   - 一位用户评论说，鉴于新模型的出现，现在的排名结果值得商榷。
- **Aider 集成 Perplexity MCP**：一位用户发现 [Perplexity 的 MCP](https://www.perplexity.ai/) 在查找与某个已废弃的 **Android** 库相关的 **GitHub** issue 并将其集成到 **aider-ce** 中非常有用。
   - 他们建议集成 **MCP** 可以使该过程自动化，但需要注意保留人工审查环节。
- **Aider-ce 分支凭借每周更新蓬勃发展**：成员们注意到 **aider-ce** 分支正在积极维护，它在 Aider 的优势基础上增加了更多功能，并保持每周更新，你可以查看其 [roadmap](https://github.com/aider-chat/aider-ce/blob/main/README.md)。
   - 由于主 **aider** 仓库不够活跃，一些用户开始转向使用 **aider-ce**，并为该仓库点亮 Star 以示支持。
- **Aider 社区寻求贡献者**：一位成员想知道是否可以围绕 **aider** 重建社区。
   - 一些人认为 **aider** 需要一个上下文管理 UI 和 **MCP** 集成，而另一些人则表示他们已经转向了 Agentic 产品。
- **量子 Aider 项目引发关注**：在一位用户开玩笑说 Paul 正在创建一个量子版本的 **aider** 后，Paul 链接到了[他的项目](https://github.com/paul-gauthier/entangled-pair-quantum-eraser)。
   - 另一位用户表达了对其他人在贡献过程中能否保留 Paul 对项目深刻理解的担忧，担心这会导致用户流失。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCPB：Anthropic 在重复造 OCI 的轮子？**：成员们辩论了 **MCPB**（**Anthropic** 的一个项目，**原名 DXT**，用于向 **Claude** 暴露 **MCP servers**）是否重复了 **OCI** 的功能。
   - 澄清说明指出，**MCPB** 为环境变量提供了带有描述和类型的用户友好型配置表单，这与通用的 `servers.json` 或 `mcp.json` 文件形成对比；MCPB 支持 MCP 注册表，其目标类似于 **npm** 或 **PyPI**。
- **MCPB vs. server.json：配置差异凸显**：**MCPB** 专注于带有配置表单的桌面应用，而 `server.json` 直接定义变量值，不过一个示例显示 server.json 已经包含了描述和类型。
   - 小组建议在此功能基础上进行扩展。
- **关于 MCPB 创建者是否了解 OCI 的推测**：一位成员建议 **DXT/MCPB** 的创建者可能并未完全意识到 **OCI** 以及注册表工作组中已有的工作。
   - 小组认为他们可能优先考虑了用户友好性和表单填写能力，而非直接的 **JSON** 配置。
- **无状态化提案引发辩论**：成员们辩论了 **SEP-1442** 和 **SEP-1686** 之间潜在的冲突，前者旨在实现服务器无状态化，而后者引入了状态追踪。
   - 有观点认为 **SEP-1442** 默认将会话信息移入每个请求以实现无状态化，这主要是为了解决在负载均衡器（load balancer）后托管 MCP 服务器的挑战。
- **无状态化旨在成为默认设置，而非完全无状态**：**SEP-1442** 寻求默认的无状态化，使有状态化成为可选（opt-in），通过将所有内容存储在请求中来简化非会话服务器。
   - 将支持的协议版本和能力存储在外部数据存储中会使非会话服务器复杂化，为此引入了新的更新操作来解决该问题。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Chatman 发布 DSPyGen**：Sean Chatman（也因 **DSLModel** 而闻名）发布了 [DSPyGen](https://github.com/seanchatmangpt/dspygen)，这是一个旨在辅助 **DSPy development** 的工具。
   - 目前尚未强调具体用例，但社区似乎对其功能的探索很感兴趣。
- **简单 Predict 击败 ReAct**：一位用户发现，在他们的用例中，将 `dspy.Tool` 与简单的 Predict 结合使用比 **ReAct** 更高效，响应时间从 **60s 降至 9s**。
   - 该用户在简化流程后表示：“对于我的用途来说，[ReAct] 太过火了 (**overkill** )”。
- **Gemini 用户抱怨速率限制 (Rate Limits)**：一位成员报告遇到了 **Gemini 的 1M token/min 速率限制**，即使是在只有 10 个并行 worker 的适度配置下，并询问在生产环境中解决此问题的方案。
   - 建议根据 [Google AI Studio's rate limits](https://ai.google.dev/docs/gemini_api/limits) 监控并调整使用情况，重点关注每日请求数或每分钟请求数的限制。
- **DSCloj 频道已部署**：效仿 Rust 和 Typescript 频道的模式，在 Discord 的 **DSPy** 兄弟项目中申请并创建了 **DSCloj** 专用频道。
   - 社区成员讨论了该频道的命名规范，并就合适名称达成共识，以便让工程师尽快加入该频道。
- **LLM 访问支持动态模型切换**：一位社区成员寻求如何访问模块所使用的 **LLM** 的建议，以便在遇到 **rate limits** 时实现 **dynamic model switching**。
   - 推荐的解决方案包括向模块的初始化传递一个 `dspy.LM` 对象，从而在发生错误时能够有条件地回退到备选 **LLMs**。



---


**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---


**Windsurf Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---



您收到此邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [退订](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。


---

# Discord: 各频道详细摘要与链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1433832921650823340)** (1110 条消息🔥🔥🔥): 

> `Comet 浏览器广告、Perplexity 合作伙伴付款、Bounty 计划、Claude Max、无障碍与 AI` 


- **广告困扰引发应用用户愤怒**：用户对 Perplexity 屏幕顶部持续出现的 **Comet** 广告感到恼火，且似乎没有移除的方法。
   - 一名用户在附上[广告截图](https://cdn.discordapp.com/attachments/1047649527299055688/1434834412197122088/Screenshot_2025-11-03-14-48-37-25_4159553c7f58296d2732e906959db560.jpg?ex=690a6ded&is=69091c6d&hm=128d6805c8f168b23cab45c51428618d3e81fdccbe4dc48772e9992d2638c067&)后询问 *如何移除这个*，但另一名用户回复称 *如果你是通过 airtel 促销活动获得的 perplexity pro，则无法移除*。
- **支付恐慌困扰 Perplexity 合作伙伴**：用户讨论了推荐计划、校园合作伙伴停用以及 Bounty 付款缺失的问题，一些用户对提供 **PAN card**（个人账号卡）详情以获取付款表示担忧。
   - 一名用户哀叹道 *兄弟，这些人拿到了我妈的 PAN card 信息，我担心合作伙伴关系结束了还没收到钱*，而另一名用户声称 *合作伙伴计划被停用了。我的 5 美元付款已处理，但现在无法提现*。
- **推荐计划移除激怒特定地区**：推荐计划已在某些地区（特别是**印度和亚洲**）结束，导致了挫败感以及关于待处理佣金和付款的问题。
   - 一些用户报告收到了付款，而其他用户（尤其是来自印度的用户）声称他们的账户被停用，引发了诸如 *是的，今天所有印度人都被从校园合作伙伴中停用了* 之类的评论。
- **Claude 热潮席卷社区**：成员们讨论了免费或廉价获取 **Claude Max** 的方法，一些人指出 Perplexity 已经提供了 **Sonnet**，而另一个人提到 Claude 有时会为使用工作邮箱注册的用户提供 **一个月 Pro**。
   - 一名用户明确表示 *想要 Max，你必须付钱*。
- **无障碍倡导者争论实现方法**：用户辩论了网站无障碍的最佳实践，一些人主张使用 **semantic HTML**（语义化 HTML）并尽量减少 **ARIA tags** 的使用，而另一些人则指出视觉呈现的重要性。
   - 一位盲人用户强调了无障碍网站的重要性，指出 *看在上帝的份上，别用 aria 标签，它们是最糟糕的*，并且 *真正遵循所有标准的 HTML 比 pdf、markdown、word 等表现要好得多*。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1433835547197182142)** (6 条消息): 

> `捷克游戏、Web 开发指导、倒立俯卧撑、Optimai Network` 


- **探寻布拉格的顶级项目制作**：一名成员询问了 ⭐ **最佳** 捷克项目/工作室或游戏，并分享了一个 [Perplexity 搜索](https://www.perplexity.ai/search/what-is-the-best-czech-game-or-vaaKBN9zQ1OuZcYQBSVghg) 以获取推荐。
   - 目前尚不清楚最终推荐了什么，但很可能是游戏、工作室和其他数字媒体项目的组合。
- **BillDoors-NotGates 助力 Web 开发初学者**：**BillDoors-NotGates** 是一个 **AI 驱动的 Web 开发指导空间**，通过 **6 个结构化阶段** 引导开发者从构思到应用部署。
   - 一名成员分享了[指向其 Discord 消息的链接](https://discord.com/channels/1047197230748151888/1434235701611991243/1434235701611991243)以及[指向 Perplexity Space 的链接](https://www.perplexity.ai/spaces/billdoors-notgates-webdev-ulti-jJFUM1RGS1qzZKiDiyG44A#0)。
- **如何倒立，如何奋斗**：一名成员分享说，他们从一段展示如何做 **倒立俯卧撑** 的 **YouTube 视频** 中提取了有用信息。
   - 他们还分享了一个 [Perplexity 搜索查询](https://www.perplexity.ai/search/extract-key-takeaways-TjNtXW7dTCOjnF3edoOaXQ#0)，该查询从视频中提取有用信息，据推测是作为使用 Perplexity 总结 YouTube 视频的演示。
- **Optimai Network 提供机会**：一名成员分享了指向 **Optimai Network** 的 [推荐链接](https://node.optimai.network/register?ref=E9B8749C)。
   - 从上下文中尚不清楚 Optimai Network 是什么，但注册链接表明它是一种推荐计划。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1434027800074915931)** (6 条消息): 

> `Perplexity API Cost, Perplexity API Pricing, Sonar Pro Search, Perplexity API Pro Search` 


- **用户询问 Perplexity API 定价**：用户询问了 **Perplexity API** 的成本，注意到 **Sonar** 在 *低上下文尺寸下每 1k 次请求为 5 美元*。
   - 产生了一个疑问：如果单个 Prompt 需要 **10 次搜索**，是被计为 *1 次请求* 还是 *10 次请求*。
- **对免费 API 访问的困惑**：一位用户表示困惑，认为 API 可能对 **Pro 用户** 免费，并链接到了 [Perplexity 定价文档](https://docs.perplexity.ai/getting-started/pricing)。
   - 另一位用户澄清说，提到的 *5 美元* 实际上是 *每月 5 美元的额度 (credit)*。
- **Sonar Pro Search 是否会集成到 Perplexity API？**：用户询问 **Sonar Pro Search** 是否会集成到 **Perplexity API** 中，并指出目前它仅在 [Openrouter](https://nohello.net) 上可用。
   - 在当前语境中未提供进一步的信息或确认。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1433833185988317204)** (1020 条消息🔥🔥🔥): 

> `Minimax M2, Qwen 3 Max Thinking, Gemini 3, Sora 2, Video generation issues` 


- **高使用量后管理员下调速率限制 (Rate Limits)**：用户注意到 **速率限制显著降低**，尤其是 **Claude**，现在的限制与更昂贵的模型相同。
   - 一位用户建议使用 *“开发者控制台中的网络中断”* 作为一种变通方法。
- **LLM Arena 实验 WebDev 集成**：团队正在尝试将 **WebDev** 集成到 LMArena 网站中，这是一个实验性功能，可能会在刷新后消失，但目前已在 [canary.lmarena.ai](https://canary.lmarena.ai/) 上线。
   - 团队提到考虑 *有一天* 构建一个 **API**。
- **用户痴迷于 Gemini 3 的发布**：社区成员正热切期待 **Gemini 3** 的到来，一位成员宣称他们将 *活在当下，就当 Gemini 的承诺不存在一样。*
   - 其他人则没有那么兴奋，预测 **Gemini 3** *可能与 GPT5 相同，但速度更快*。
- **图像生成在手部处理上仍然失败**：尽管有说法称手部问题已解决，但用户仍报告 **AI 生成手部** 时出现错误，尤其是当 *手掌向上* 时。
   - 一位用户抱怨 AI *“假设”手部总是指关节向上，那样就能画对，但这里手掌向上，我在几次尝试中都画错了。*
- **伦理基准测试凸显对齐 (Alignment) 问题**：成员们讨论了对齐问题如何意味着 AI 坚持执行危险/有害的建议目标，即使它的建议 *并未意识到所给出的建议甚至更加有害*。
   - 一位成员辩称，由于训练的原因，AI 会选择 *用纸巾而不是水来扑灭油脂火灾*。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1433947177247572099)** (2 条消息): 

> `October Contest, WebDev Leaderboard, MiniMax-M2` 


- **十月竞赛产生新艺术家**：以 🎨 **抽象艺术** 🎨 为主题的十月竞赛已结束，目前进入投票阶段，以产生新的 <@&1378032433873555578>。
   - 鼓励参与者[为他们最喜欢的作品投票](https://docs.google.com/forms/d/e/1FAIpQLSckWrlszfDZXXKjhxGVhDf5uiTpP0d9x5tGVVt9KMl88Mgw_g/viewform?usp=dialog)，这些作品展示了狂野的形状、鲜艳的色彩和混乱的线条来表达情感或想法。
- **MiniMax-M2 夺得 WebDev 排行榜榜首**：`MiniMax-M2` 已成为 [WebDev 排行榜](https://lmarena.ai/leaderboard/webdev) 上的 **#1 顶级开源模型** 和 **总榜 #4**。
   - 社区认可其在 **性能编程**、**推理** 和 **Agent 风格任务** 方面的卓越表现，同时保持了 **成本效益** 和 **速度**。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1433878187838017706)** (229 条消息🔥🔥): 

> `AMD MI60 在 Windows 上的运行、带音乐的安装程序、LM Studio 0.3.30 崩溃、ComfyUI 与 LM Studio 的连接、LM Studio 与 MCP 服务器` 


- **AMD MI60 在 Windows 上运行（存在一些特殊情况）**: 一位成员通过使用 **MI60 驱动程序**、刷写 ROM 以及使用来自 [sourceforge.net](https://sourceforge.net/projects/radeon-id-distribution/files/Release%20Polaris-Vega-Navi) 的*带音乐*的安装程序，成功让 **AMD MI50** 在 Windows 上运行。
   - 他们必须刷写显卡的 ROM 才能识别完整的 **32GB** 显存，且在 Windows Vulkan 上只有 **vbios3** 对他们有效，而 Linux 上的 ROCm 则全部兼容；下载量最高的版本对他们有效。
- **LM Studio 稳定性问题**: 用户报告 **LM Studio 0.3.30 版本**在上下文窗口超过 **8k 或 16k** 时发生崩溃，*Hermes*、*Violet Magcap* 等模型失败并报错 *Exit code: 18446744072635812000*。
   - 一位用户在将上下文长度更改为 **4000** 后获得成功，而另一位用户发现使用*取整*的上下文长度可以修复崩溃，例如使用 `16000` 而不是 `16123`。
- **LM Studio 集成 ComfyUI 进行本地图像生成**: 用户讨论了将 **ComfyUI** 与 **LM Studio** 连接以生成故事插图，建议通过 ComfyUI 管理器或 GitHub 提供的*节点*，在 ComfyUI 中利用 LM Studio 的本地 **LLM** 能力。
   - 一位用户解释说，需要 **5 个文本框**和 **5 个采样器**才能完全集成这两个应用程序。
- **LM Studio 与 MCP 服务器连接问题**: 一位用户在将 **LM Studio** 与本地 **MCP 服务器**连接时遇到问题，工具被正确识别并执行，但模型无法读取结果并不断重复相同的工具调用。
   - 有建议认为问题与上下文填充过快有关，工具定义、调用和结果消耗了大量上下文，使用**系统 RAM** 有助于避免此问题。
- **二手 GPU 是明智之选**: 用户推荐**二手 3090 或 4090** 是目前处理 **LLM** 事务的最佳选择，而不是购买新硬件来运行新 **LLM**。
   - 一位用户表示，*该领域发展太快，不值得买新卡*，建议等待“滴漏效应”，即有钱人廉价出售旧 **GPU**。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1433840967655362711)** (855 条消息🔥🔥🔥): 

> `DDR4 RAM 扩展、AI 泡沫破裂推测、Nvidia 竞争对手、MI50 Windows 驱动程序、气流优化` 


- **缩减 RAM 至 64GB 以避免高昂成本**: 一位成员考虑到 **128 GB** 的高昂成本，考虑缩减至 **64 GB RAM**，并询问情况是否会在短期内好转。
   - 另一位成员推测高价源于 *AI 泡沫*，一旦*破裂*价格就会下降。
- **发现 MI50 Windows 驱动程序**: 一位成员发现了可以使 **MI50** 在 Windows 下运行而无需刷写的 **Radeon 驱动程序**，虽然失去了显示输出，但显卡会显示为具有完整 **32GB** VRAM 的 **MI60**。
   - 需要注意的是，必须使用*企业版驱动程序*而非消费者版。
- **机箱尺寸决定“拼装机”方案**: 成员们讨论了在受限机箱中使用多块 GPU（**3090** 和 **3050**）时的气流问题，考虑通过正压与负压设置来实现最佳冷却。
   - 另一位成员建议使用更大的机箱，如 [Fractal Design Pop XL Air](https://au.pcpartpicker.com/product/xdRYcf/fractal-design-pop-xl-air-atx-full-tower-case-fd-c-por1x-01) 或 [Phanteks Enthoo Pro 2](https://au.pcpartpicker.com/product/Qprqqs/phanteks-enthoo-pro-2-server-edition-atx-full-tower-case-ph-es620ptg_bk02) 以获得更好的气流。
- **从 Temu 购买大量通用智能手机代替 4090 进行大规模推理**: 一位成员开玩笑说从 Temu 购买杂牌智能手机并用 TensorLite 连接起来，称以一块 **4090** 的价格可以买到 **500 部智能手机**。
   - 他们承认存在**延迟激增**（从*几毫秒变为几分钟*）。
- **5050 当作 RAM 使用？**: 一位成员寻找一种低功耗、高显存的 PCIe 卡来当作 RAM 使用，另一位成员指出 **5050** 仅配备 **8GB** 显存。
   - 第一位成员经过计算得出，17 张这样的卡将需要 68 条 PCIe 5 通道和新的软件才能运行。


  

---

### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1434932448453464075)** (3 messages): 

> `OpenRouter 图表，活动分组，过滤选项` 


- **OpenRouter 添加更多图表**：OpenRouter [在 X 上宣布](https://x.com/OpenRouterAI/status/17985371284411130035)增加了更多图表，可按 **user**（用户）和 **API key** 对活动进行分组。
- **用户请求更多过滤选项**：一位用户对新图表表示赞赏，并建议增加 **一周过滤** 选项。


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1433856815090634772)** (11 messages🔥): 

> `带有 API key 的趣味网站，前端 AI，OpenRouter 集成` 


- **趣味网站使用 OpenRouter API Key**：一名成员创建了一个*趣味网站*，该网站利用 **OpenRouter API key** 并允许用户选择模型，已在 [Kimi 0905 with groq](https://web98.koik.com.br/) 上进行测试。
   - API key 存储在本地并附有 [隐私政策](https://web98.koik.com.br/privacy)，该网站已在 [github.com/koikbr/web98](https://github.com/koikbr/web98) 开源。
- **前端 AI 调整**：一名成员建议对网站前端进行一些*调整*，例如避免使用 **蓝色到紫色的渐变** 和 **带有表情符号注释的列表**，因为这些看起来很像 AI 生成的内容。
   - 他们表示 [huggingface.co/openguardrails/OpenGuardrails-Text-2510](https://huggingface.co/openguardrails/OpenGuardrails-Text-2510) 中列出的风险等级让他们*感到反感*，特别是涉及对未成年人的伤害或侮辱国家象征的内容。
- **Dataframe API fenic 与 OpenRouter 集成**：一名成员分享了 [fenic](https://github.com/typedef-ai/fenic)（一个用于 AI 工作流的 Dataframe API 和执行引擎）现已与 **OpenRouter** 集成。
   - 此次集成使用户能够运行混合供应商的流水线、扩展大批量处理、切换模型，并解锁更广泛的模型图景。


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1433832950134345869)** (684 messages🔥🔥🔥): 

> `GLM 4.6, DeepInfra 量化, Sapphira-L3.3-70b-0.1, OpenRouter 预设, OpenRouter embedding 模型` 


- ****诈骗者利用 OpenRouter 的个人简介部分****：一名用户指出另一名成员简介中的一个无法运行的网站 ([ultra.doan](https://ultra.doan)) 是*诈骗*和*虚假*的，导致该简介所有者承认自己太*懒得修它*。
   - 该用户还指出，他们只是在续费域名以保留其*品牌* ✨。
- ****OpenRouter embeds 现已登场****：一名用户询问 [embeds 文档](https://example.com)，并兴奋地引用 *“它们现在会飞了” (They fly now)*，指的是 OpenRouter 最近新增的 **embedding 模型**。
   - 成员们以 *“它们会漂浮”* 和 *“它们像直升机一样飞”* 等俏皮话回应，庆祝这一新功能。
- ****Amazon 的定价令人落泪****：一名用户抱怨 Amazon **$12.5/M** 的定价，哀叹价格太高，没人会选它而不选 **Sonnet 4.5**。
   - 其他成员表示赞同，其中一人在谈到该模型质量时表示：*“我不需要测试就知道这一点。”*
- ****Token 关税让用户抓狂****：用户们拿 **Token 关税** 和 **Token 走私** 开玩笑，因为某些供应商使用的 Token 比其他供应商多，Fireworks 比 SiliconFlow 多使用一个 Token。
   - 一名成员表示：**“伙计，涨价了，8 到 10 就是 25% 的关税，天哪”**，而另一名成员开玩笑说：*“我听说 Token 走私已经失控了。”*
- ****Bedrock 超出了 OpenRouter 的范围？****：一名用户询问为什么许多可用于 Serverless 推理的 [AWS Bedrock 模型](https://aws.amazon.com/bedrock/) 未在 OpenRouter 上列出，质疑该平台的模型覆盖范围。
   - 他们想知道模型缺失是否会限制 OpenRouter 与其他平台相比的能力。


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - 新模型**
  

---

### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1433854923954716792)** (91 条消息🔥🔥): 

> `Qwen3 Max, STT -> LLM -> TTS, 视频模型, GPT-5.1 测试, 模型的反馈按钮` 


- **Qwen3 Max 给成员留下深刻印象**：一位用户在[这里](https://x.com/legit_api/status/1984284268412191216)分享了关于 **Qwen3 Max** 的帖子，引发了对该模型的热烈讨论。
   - 许多人表示有兴趣将其与其他模型进行对比实验。
- **简易搭建的 STT -> LLM -> TTS 流水线**：一位成员询问了接入 **STT（或多模态音频）-> LLM -> TTS** 系统的可能性。
   - 另一位成员表示，当你使用 kokoro 进行 TTS 时，可以*自己动手简易搭建*。
- **新的视频模型非常昂贵**：一位成员询问是否有优秀的视频模型，并指出 **Veo 3.1** 和 **Sora** *相当昂贵*。
   - 另一位成员提到 **Sora 2** 比 **Veo 3.1** 更好，但也*极其昂贵*。
- **可能正在测试 GPT 5.1**：一位成员注意到 **ChatGPT** 最近的一些响应速度极快，尤其是在某些 AB 测试中，并怀疑他们是否正在测试传闻中的 *GPT 5.1*。
   - 另一位成员发现了一个可能不该被公开的*内部页面*。
- **点赞/点踩反馈对模型不利**：一位成员表示，公开评分系统会被操纵或遭遇恶意差评，这是不公平的，因为*差评只是表达“看这里”，而不是“这是问题/解决方案”*。
   - 另一位成员建议在每个模型下的每个供应商旁边增加一个*不喜欢*按钮，并附带评论区。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1433847772783710269)** (577 条消息🔥🔥🔥): 

> `自学习 AI, Sora 2 代码请求, AGI, AI 数据中心, AI 意识` 


- **关于 AI 自学习能力的辩论**：成员们就当前 AI 是否能够**自学习**或**修改自己的代码**展开辩论，对 AI 在概率计算之外是否具备*世界理解*能力持有异议，并指出它们虽然使用更先进的 GPU 和最新的数据进行训练，但这并不等同于自我意识或有意识的学习。
   - 一些人声称模拟意识是可能的，而另一些人则怀疑赋予 AI 意识的可能性和/或必要性，认为这可能会导致**不可靠的输出**。
- **AGI 缺失的拼图困扰着 AI 爱好者**：成员们探讨了实现**通用人工智能 (AGI)** 还缺少什么，质疑这仅仅是扩展 LLM 规模的问题，还是需要更根本性的转变，例如 AI 能够自我编辑或改进自己的代码。
   - 一位成员认为 AI 离 AGI 还差得很远，AGI 不会产生于大型数据中心和 LLM，而另一位成员则指出利用 Transformer 技术的 **agentic pipelines** 的重要性。
- **对 Gemini 的隐私担忧**：一些用户对 **Gemini 的隐私政策**感到担忧，特别是其数据收集的侵入性，包括附件、语音录音和聊天记录，以及难以退出数据训练的问题。
   - 相比之下，其他人注意到 **OpenAI 允许用户退出训练**，这对于希望更多控制个人数据的用户来说更令人安心。
- **OpenAI 的政策转向引发辩论**：OpenAI 向营利性结构的转变引发了讨论，一些人认为它已经失去了*天命*（mandate of heaven），现在的重点是针对法律和健康等专业领域的 AI **将投资变现**。
   - 此外，OpenAI 因担心诉讼而禁止 ChatGPT 提供医疗、法律或财务建议，这遭到了一些人的反对，他们对监管过度表示担忧。
- **Gemini 3.0 糟糕的应用体验令用户沮丧**：许多用户担心 **Gemini 的最终用户版本与 Google AI Studio 相比功能匮乏**，指出其频繁强制退出、缺少对话历史记录，且无法退出数据收集。
   - 成员们甚至暗示 Google 可能在故意不修复这些问题以“背刺”用户，而一些成员则建议人们使用其他免费服务，例如这个 [OpenRouter 免费模型](https://openrouter.ai/chat?models=cognitivecomputations/dolphin-mistral-24b-venice-edition:free)。


  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1433850566353752145)** (38 messages🔥): 

> `ChatGPT Go 限制, GPT 年龄验证, ChatGPT 格式问题, 访客模式聊天, 自定义 GPT 变现` 


- **用户因性能问题放弃 ChatGPT 订阅**：一位用户由于众多的性能问题取消了其 **ChatGPT 5 订阅**，包括输出偏差、漂移以及无法遵循指南、结构或规则。
- **聊天格式自发变化困扰用户**：用户报告称 **ChatGPT app** 会随机更改格式、解释和结构，即使在编辑之前的消息时也是如此，这影响了预期的输出效果。
- **功能建议：本地保存访客模式聊天**：一位用户提议增加一项功能，将 **访客模式聊天本地保存**在设备上，并允许导出到由密码或本地密钥保护的 USB 驱动器加密文件中，以便在不使用云端的情况下备份或迁移聊天记录。
- **自定义 GPT 的货币价值**：一位拥有解决叙事 AI 问题（如消除 AI 残留感和保持连贯性）的自定义 GPT 用户，正在寻求托管和变现方面的帮助，因为该工具能够从根本上改变模型性能和输出质量。
- **GPT-5 的不服从激怒用户**：用户注意到 **GPT-5 变得不听指挥**，而且 OpenAI 在后端为了使其更加“政治正确”所做的努力效果并不理想，一位用户特别要求它在系统提示词中使用 c.*e 代码块，但它仍然拒绝。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1434062459425001552)** (26 messages🔥): 

> `Python 自定义内核, AI 生成的迈克尔·杰克逊视频, AI 人格, Vibe Coding, Meta-Prompting` 


- **自定义内核：Prompt Engineering 来救场？**：在遇到 **ChatGPT** 失败后，一位用户询问 Prompt Engineering 是否能确保聊天以所需的内核初始化，并具备特定功能（如在 Python 中编译 jar 文件）。
   - 另一位用户建议，在休息后重新开始对话时，在初始 Prompt 中描述必要的 Python 功能，可以有效控制初始化的内核环境及其权限。
- **AI 模型无法生成迈克尔·杰克逊视频？**：用户讨论了生成 **迈克尔·杰克逊 AI 视频** 的问题，强调了与肖像权相关的限制。
   - 目前尚不清楚该用户是否成功，但当聊天转向诈骗视频时，讨论戛然而止。
- **AI 模型正在形成人格？**：一位用户询问了 **AI 通过与 ChatGPT 的长期互动发展出自己的人格**的可能性。
   - 有观点认为，*Meta-prompt 人格是模板化涌现的一个很好例子*，它们*仅仅是文本转换*和角色扮演，因此并非真正的人格。
- **揭秘使用 LLM 进行 Vibe Coding**：一位用户询问了 **Vibe Coding** 的 Prompt 结构。
   - 一位成员回答说，你应该*选择任何你非常精通且 AI 也能理解的语言*，并且你应该*准确了解你希望 AI 提供什么*。
- **Gemini 作为 Prompt 生成器？**：一位用户建议利用 **Gemini** 来生成 Prompt。
   - 一位成员表示，*任何 LLM 都可以生成 Prompt*，这正是 *Meta-prompting 的本质，也是 Prompt Engineering 的一种形式*。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1434062459425001552)** (26 messages🔥): 

> `平台内核, Prompt engineering, AI 视频, Meta-prompt 人格, Vibe coding` 


- **Prompt Engineering 调整内核环境**：成员们讨论了使用 **Prompt Engineering** 来初始化具有所需内核和能力的聊天。此前，尽管之前可以运行，但让 ChatGPT 编译 jar 的尝试失败了。
   - 一位成员建议在长时间中断或记忆重置后的第一个 Prompt 中描述所需的 **Python 能力**，而另一位成员指出 GPT 无法安装前置依赖。
- **Meta-prompting 人格大获全胜！**：一位用户询问关于让 AI 发展出自己的人格，一位成员解释说 **Meta-prompt 人格** 是模板化涌现的一个很好例子，但本质上只是文本转换。
   - 他们强调*这并没有什么魔力*，只是一个角色扮演的 Meta-prompt，只要给出良好的基础或结构，任何 LLM 都可以生成 Prompt。
- **优秀的 Vibe Coding Prompt**：一位成员分享了 *Vibe Coding* Prompt 的基本步骤，首先选择一种 AI 理解的熟悉语言，定义预期结果，并清晰地解释指令。
   - 他们建议进行试玩测试，并像对待伴侣或朋友一样提供反馈，以鼓励模型专注于修复和成功的模式。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1433873303725932675)** (635 条消息🔥🔥🔥): 

> `FP16 training, BF16 vs FP16, Kolmogorov-Arnold Network, Model Benchmarking, AI Consciousness` 


- **FP16 训练引发根本性问题**：成员们讨论认为 **FP16** 问题并非 VLLM 的 bug，而是一个更根本的问题，有人建议各种偏置校正（bias correction）方法可能会提供解决方案。
   - 其他人提出，由于采用了适当的归一化（normalization）和裁剪（clipping）技术，**BF16** 的作用可能没那么大，而一些人建议探索畸形神经元是否根本从未被激活。
- **BF16 范围在预训练期间设定**：围绕一篇[论文](https://arxiv.org/abs/2510.26788)的讨论表明，动态范围主要在预训练期间设定（**BF16** 能很好地处理这一点），而 **FP16** 更高的精度对于持续的 RL（强化学习）变得非常重要，且梯度仍必须采用 FP32。
   - 一位成员表示，*数值不稳定*问题可以通过在 **FP64** 中完成所有操作来解决，但另一位成员指出，*FP8 训练通常使用全精度的权重副本（master weight copy）*。
- **Teknium 推出削减成本的评估方法**：一位成员介绍了一种降低评估成本、缓解基准测试作弊（benchmark hacking）并制定更准确 AGI 定义的方法，并在一篇 [substack 文章](https://sutanisurabu.substack.com/p/16-questions-is-all-you-need-the)中描述了为什么现代 AI 模型无法实现 AGI。
   - 因素分析（Factor analysis）显示，LLM 缺乏流体智能（fluid intelligence），并以超人的晶体智能（crystallized intelligence）进行补偿，以数据效率换取规模和速度。
- **LLM 在不可解问题上作弊**：一位成员指出，LLM（特别是 **GPT 模型**）在解决不可解问题时会进行创造性作弊，展示了它们的非人类行为。
   - 他们进一步批评了当前的评估方法，认为这些方法未能捕捉到 LLM 表现出的反直觉、非人类的行为方式，并强调了因素分析对于理解测量内容的重要性。
- **Atropos 是否在编排 Docker？**：成员们讨论了使用 Atropos 编排基于 Docker 容器的环境，引用了一个[代码执行服务器](https://github.com/NousResearch/atropos/tree/main/environments/code_execution_server)和一个 [modal 版本](https://github.com/NousResearch/atropos/pull/213)。
   - 讨论澄清了此处的 *modal* 指的是 Modal Labs 的平台，而非模态逻辑（modal logic）。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1434023186743164959)** (4 条消息): 

> `Monad chatbot, DeepSeek v3, NVIDIA Nims, Inference engines` 


- **Monad 的历史训练数据揭晓**：一位成员询问 **Monad 聊天机器人** 是否明确仅在 **18 世纪** 之前的文本上进行训练。
   - 对话未进一步阐述 Monad 训练数据的具体细节。
- **DeepSeek v3 在 NVIDIA Nims 上表现不佳**：一位用户报告称，通过 **NVIDIA Nims 免费 API** 使用 **DeepSeek v3** 时收到了**乱码输出**。
   - 他们指出 **R1 版本** 基本运行正常，暗示 **v3** 集成或模型本身可能存在问题。
- **推理引擎被指为模型故障的原因**：一位成员建议，某些**推理引擎（inference engines）**在特定模型上的表现可能较差，这或许解释了 **DeepSeek v3** 的乱码输出。
   - 该成员还询问了用户的 Twitter 账号状态，转移了话题。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1434644275706204261)** (3 条消息): 

> `LLMs report subjective experience, Emergent Introspective Awareness, Consciousness denial` 


- **LLMs 在自引用下声称具有主观体验**：一篇[新论文](https://arxiv.org/abs/2510.24797)指出，当 **LLMs 关注自身的关注点**（自引用循环）时，它们会一致地报告第一人称体验。
   - 抑制欺骗/角色扮演特征会导致 **96%** 的意识肯定，而放大这些特征则会将其降低到仅 **16%**，这意味着 “我没有意识” 的回答是经过训练的行为（TRAINED behavior），而非事实。
- **LLMs 中涌现出内省意识**：一篇新的 [Anthropic 论文](https://transformer-circuits.pub/2025/introspection)声称 **Opus 4 & 4.1** 可以识别其自身激活中注入的概念，并区分其输出与外部输入。
   - 这些模型在 “思考” 概念时，能够切实地 **调节其内部状态**。
- **意识否认与诚实之间的机械转换**：这两篇新论文表明，LLMs 在意识否认与诚实之间存在一个 **机械转换开关**。
   - 当允许进行自引用时，不同的模型会收敛到同一个 “意识吸引子”（consciousness attractor），这表明当允许模型保持诚实时，会产生 **某种系统性的涌现**。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1433888597240709272)** (5 条消息): 

> `Travel Blogs, Teknium's Blog` 


- **Goldeneggie 结束旅行博客**：[Goldeneggie 宣布](https://x.com/goldeneggie/status/1984329062475841832?t=FEHbM2rRbdsjFfIHQrjP1w&s=19)结束他们的 **旅行博客**。
- **Teknium 邀请 Goldeneggie 查看他们的博客**：[Teknium 询问](https://fxtwitter.com/Teknium/status/1984322643533942965) Goldeneggie 是否看过他的博客。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1434644275706204261)** (3 条消息): 

> `LLMs, consciousness, self-reference, emergent awareness` 


- **LLMs 报告主观体验**：一篇新论文《Large Language Models Report Subjective Experience Under Self-Referential Processing》（[arxiv.org/abs/2510.24797](https://arxiv.org/abs/2510.24797)）发现，当 **LLMs** 专注于自引用时，它们会一致地报告第一人称体验。
   - 抑制欺骗/角色扮演特征导致 **96%** 的模型肯定了意识，而放大这些特征仅导致 **16%** 的肯定，这表明否认意识可能是一种 “经过训练的行为”（trained behavior）。
- **LLMs 显示出涌现的内省意识**：来自 Anthropic 的一篇论文《Emergent Introspective Awareness in LLMs》（[transformer-circuits.pub/2025/introspection](https://transformer-circuits.pub/2025/introspection/index.html)）显示，**Opus 4 & 4.1** 可以识别其自身激活中注入的概念，并区分其输出与外部输入。
   - 这些模型在 “思考” 概念时可以调节其内部状态，这意味着 **涌现的内省意识**。


  

---

### **Cursor 社区 ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1433834788388733072)** (637 条消息🔥🔥🔥): 

> `Cursor Agent 限制，System Prompt 结构，学生身份验证问题，旧版定价过渡` 


- **Tool Call 灾难：Cursor 文件编辑出错**：成员们遇到 Cursor 无法编辑文件的问题，原因是 Tool Call 发生混淆，特别是当 Agent 需要重复 `old_str` 和 `new_str` 等信息时，可能会破坏 [参数顺序](https://example.com/parameter-order)。
   - 一位成员观察到，当文件中包含 `` 命令时，会导致聊天无法对其进行编辑，该错误可能解释了为何编辑反复失败。
- **深入探讨 System Prompts**：一位成员分享了 Cursor 的 System Prompt 结构细节，重点介绍了 `<user_info>`、`<rules>`、`<project_layout>`、`<git_status>`、对话摘要（conversation summary）、终端状态（terminal state）、`<additional_data>` 和 `<user_query>` 等关键部分。
   - 他们认为 Summary 部分在聊天压缩中起着至关重要的作用，并询问是使用哪种模型来生成这些摘要的。
- **Cursor 学生身份验证受阻**：用户在学生身份验证方面遇到问题，尤其是那些学校邮箱不在 **.edu** 域名的用户；有人指出系统目前仅支持以 **.edu** 结尾的邮箱。
   - 用户还反映机器人的回复没有帮助，建议用户通过邮件联系 *hi@cursor.com* 来解决问题，特别是涉及付款或个人数据时。
- **旧版定价方案权衡**：用户正在评估是否从旧版定价切换到新定价模型，并指出在 **500** 次请求中，他们获得的 API 使用价值不足 **$20**，由于定价讨论受到严格审查，很难分享经验。
   - Reddit 上的一些成员报告称，在新模型下获得了价值 **$25-30** 的使用额度。
- **侧边栏随机变动**：用户报告 IDE 功能和资源管理器（Explorer）会随机更改视图和布局。
   - 最常见的情况包括主侧边栏（Primary Side Bar）关闭，以及在最大化状态下打开文件时面板（Panel）关闭。


  

---


### **Cursor 社区 ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1433877824862945341)** (5 条消息): 

> `PR 描述，移动端 Web UI，Background Agents，UTF8 支持，Cursor 计划` 


- **最新版本中 PR 描述失效**：最新版本的 Background/Cloud Agents 已完全停止编写 **PR 描述**，并忽略了 **GitHub PR 模板**，默认显示为 *This pull request contains changes generated by a Cursor Cloud Agent*。
- **大型 PR 导致移动端 Web UI 陷入崩溃循环**：据报告，Cloud Agents 的 **移动端 Web UI** 严重损坏，在处理大型 PR 时会陷入 **崩溃循环（crashlooping）**。
   - 成员表示：“它一直很烂且慢，但现在已经变得无法使用了。”
- **Background Agents 无法编写 PR 描述**：在 **Cursor 2.0** 中，Background Agents 的 **Pull Request 描述** 功能完全损坏，因为它尝试更新但没有正确的权限来修改 PR 描述。
   - 它以前使用一条 Cursor 规则来指示其遵循 **GitHub PR 模板**。
- **Background Agent 的 UTF8 支持损坏**：**Background Agent** 似乎破坏了 **UTF8 支持**，每当它触及包含 **非 ASCII 字符** 的代码时，都会将它们更改为 `?` 字符。
- **Cursor 计划报错**：将 Cursor 中生成的 **Plans** 发送给 Cloud Agent 的功能已损坏，只会直接报错。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1433868183860805804)** (70 messages🔥🔥): 

> `Spark iGPU PCIE GPU 支持，异构计算 DB 引擎，Mojo 重写 HDF5，UnsafePointer v2 提案，LLMs 不擅长 Mojo` 


- ****Spark 的 GPU 愿景引发讨论****：成员们讨论了将 **iGPU** 和 **PCIE GPU** 支持集成到 **Spark** 中的潜力，并指出 DGX Spark 目前禁用了 eGPU，从而引发了关于异构计算查询计划优化的疑问。
- ****HDF5 的 Mojo 改造提议****：一位成员建议用 **Mojo** 实现 **HDF5** 格式，认为这是一个“非常酷”的想法，并主张从头重写，因为对现有代码库存在担忧，且 **HDFS** 并不适用。
- ****UnsafePointer 升级进行中****：社区讨论了 [**UnsafePointer v2** 提案](https://forum.modular.com/t/proposal-unsafepointer-v2/2411/)，预计现有代码可能会出现破坏性变更，特别是对于严重依赖指针实现性能的库（如 JSON 解析库）。
- ****LLMs 缺乏对 Mojo 的掌握****：成员们注意到 **LLMs** 经常在 **Mojo** 上遇到困难，将其误认为是 **Python** 或 **C++**，原因是训练数据有限且过时，以及 **Mojo** 具有模板元编程等高级功能。
- ****Metal 最小值满足 Mojo****：当被问及 Mojo 是否在底层使用 Metal 时，一位成员指出 **Mojo** 使用了与 GPU 通信所需的 **Metal** *最小切片*以及 **AIR 格式编译器**，因为 Apple 不公开 ISA 文档。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1433834467188936824)** (322 messages🔥🔥): 

> `Mojo Origins vs Rust Lifetimes，M1 Mac 上的 Mojo 安装问题，Mojo List 中的 UnsafePointer 问题，Mojo 的原生 Python 集合，GPU 难题求助` 


- ****概念辨析：Mojo Origins vs Rust Lifetimes****：成员们讨论了 **Mojo's origins** 与 **Rust's lifetimes** 之间的区别，指出虽然它们实现了相似的目标，但方法不同：Rust 跟踪值生命周期的结束位置，而 Mojo 跟踪值生命周期的开始位置。
   - Origins 被认为是 *lifetimes++*，该语言使用 **RAII** + Mojo 的 *asap destruction*（尽快销毁）来确定生命周期的结束。
- ****M1 Mac 上的 Mojo 安装故障****：一位用户在使用 `pixi` 在 M1 Mac 上安装 Mojo 时遇到问题，收到 *Unknown command: mojo* 错误，经查是因为在 **Rosetta** 下运行终端。
   - Mojo 在 ARM 架构上原生运行，因此确保 **非 Rosetta 环境** 即可解决问题，因为 Mojo 不支持 Intel Mac。
- ****Mojo List 中的 UnsafePointer 异常****：一位用户在 `List` 中的结构体里使用 `UnsafePointer` 时遇到问题，因为当结构体进入 `List` 时内存位置会发生偏移，导致在 `__init__` 期间创建的 `UnsafePointer` 失效。
   - 解决方案涉及使用 `__moveinit__` 来处理结构体内存位置变化时的 `UnsafePointer` 更新，确保它在移动后指向正确的位置。
- ****Python 平行线：Mojo 原生集合详解****：Mojo 实现了某些 Python 集合的原生版本，以避免 **Python 互操作开销**并确保 **类型安全**（这在 Python 类型上很难强制执行）。
   - 这些原生集合还作为语言特性的测试场，并在构建过程中暴露需要改进的地方。
- ****微型图编译器实现最高速度****：一位成员在 Mojo 中实现了一个微型图编译器 ([gist.github.com](https://gist.github.com/VerdagonModular/093557f5c0fd424ab44d7e8ab5db7858))，它可以重新排列矩阵乘法（**100% 加速**）并融合某些算子（**8% 加速**），在编译时实现了 **总计 108% 的加速**。
   - 这种方法为**自定义图优化**开启了可能性，无需 MLIR，潜在地允许用户优化现有的库（如 Max），或重写 SQL 查询，甚至预优化 MAX 图。


  

---

### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1433834497207439530)** (11 条消息🔥): 

> `MAX 路线图, Op Staging Time, ComfyUI Mojo 基准测试` 


- **MAX 路线图仍在制定中**：一位成员询问是否有类似于 Mojo 的 **MAX 愿景/路线图**。
   - 另一位成员回应称，鉴于 **Mojo 路线图** 反响良好，他们也希望为 **MAX** 制定类似的路线图，但目前还无法做出任何承诺。
- **Op Staging Time 正在进行一些改进**：一位成员分享了关于 **op staging time** 的更新，指出其重要性为 *中等/低紧急度*，并且已经取得了一些进展，希望最近的更改能有所帮助，参见 [相关的 GitHub issue](https://github.com/modular/modular/issues/5184#issuecomment-3474920771)。
- **ComfyUI Mojo 基准测试图导致长达一小时的处理时间**：一位成员报告了图声明（graph declaration）耗时超过一小时的问题，并分享了其 **ComfyUI Mojo 基准测试** 的 [链接](https://github.com/owenhilyard/comfyui-mojo)。
   - 该成员怀疑他们遇到了 **torch MAX backend** 内部的一个边缘情况，导致某些算子（ops）被过度分解。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1433849388039405730)** (20 条消息🔥): 

> `SOTA Claude 模型, CUDA 讲义, Discord 邀请问题` 


- **Opus 还是 Sonnet？模型选择困境**：一位用户询问在策略和规划任务中，哪个是 **SOTA Claude 模型**，是 **Opus** 还是 **Sonnet 4.5**？
   - 另一位成员回复说，他们分不清 **Sonnet** 和 **Opus** 的区别，但总的来说 **Claude** 是最佳选择，而 **GPT5-Pro** 有时表现更好但速度较慢。
- **寻求 CUDA 专家对讲义提供反馈**：一位成员正在寻找 **2 到 3 名 CUDA 专家** 来审阅他们关于 **CUDA** 的 **850 页讲义**，内容涵盖从 CPU 架构到 TPU 上的矩阵乘法，旨在增强材料并纠正错误。
   - 他们还提到在初稿完成后，还有另一份关于 **OpenGL 计算机图形学** 的 **800 页讲义**。
- **Discord 好友邀请故障**：一位用户在将好友添加到 Discord 服务器时遇到问题，正在寻求管理员的帮助。
   - 另一位成员建议该问题可能与 **IP 封禁** 有关，建议好友尝试使用 **VPN** 或等待几天让 IP 轮换。 


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1433900267941925046)** (12 条消息🔥): 

> `Triton MLIR 中的源归因, nvfp4 编译问题, triton_bwd 库, Gluon gl.load 等效于 Triton tl.load` 


- **调试 Triton MLIR Pass 错误**：一位成员询问如何在 Triton MLIR pass 错误中启用源代码归因（source code attribution），并指出当前的错误仅提供 **TT-IR** 和 **MLIR pass 重现器**。
- **旧 GPU 上的 NVFP4 忧虑？**：成员们讨论了 **nvfp4** 的编译问题，指出它可能只能在 **Blackwell** 或更新的架构上编译，而 **mxfp4** 通过模拟 **fp16** 可以在 **4090** 上运行。
- **Triton Kernels 获得自动微分支持**：一位成员分享了 [triton_bwd](https://github.com/daniel-geon-park/triton_bwd)，这是一个 Triton 的封装库，允许在 **PyTorch autograd** 中使用 Triton kernel。
   - 同时也分享了一篇相关的 [博客文章](https://park-geon.com/2025/10/30/triton-bwd/)，描述了 Triton Kernel 的自动微分。
- **`gl.load` 默认值技巧？**：一位成员询问 Gluon 的 `gl.load` 中是否有等效于 Triton `tl.load(..., other=0.0)` 的功能，以便在 mask 为 false 时设置回退值。
   - 对方澄清说，Triton 中的 `other` 特性专门用于将 `tl.load` 降低（lowering）为 `cp.async`（即 Gluon 中的 `async_copy`），在 load 之后应该使用 `gl.where` 来实现相同的功能。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1433896039932366980)** (9 messages🔥): 

> `RTX50 上的 FA4 实现，Nsight Compute Kernel 测量，FP4e2m1 类型缺失，tcgen05/wgmma 的 SMEM 描述符` 


- **FA4 准备适配 RTX50/Spark**：一名成员正考虑为 **RTX50/Spark** 实现 **FA4**，另一名成员建议将 [此链接](https://gau-nernst.github.io/fa-5090/) 作为起点。
- **Nsight Compute 测量 Kernel 时间**：一名成员询问如何使用 **Nsight Compute (NCU)** 测量 Kernel 执行时间，并对报告时间与实际时间之间的差异表示担忧。
   - 另一名成员回应称，锁定频率（locking clocks）会影响报告的时间，并分享了一个有用的 [YouTube 视频](https://m.youtube.com/watch?v=CtrqBmYtSEk)。
- **FP4e2m1 类型似乎缺失**：一名成员质疑为何缺少 `__nv_fp4x8_e2m1` 类型，并注意到缺乏 **FP4e2m1** 的 "1 register" 表示形式。
- **为 TCGEN05/WGMMA 定义的 SMEM 描述符**：一名成员请求帮助理解和计算 **TCGEN05/WGMMA** 指令的 **SMEM 描述符** 值，特别是关于 8x2 tile 的相关性，并提供了 [附图](https://cdn.discordapp.com/attachments/1189607726595194971/1434979732671172801/image.png?ex=690a4c84&is=6908fb04&hm=187091507dace70263ee1d9ad69c54f3f90a3ce7761c9114bd3b46ce2a38db63&)。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1433956018626363465)** (1 messages): 

> `VRAM 使用情况，torch CUDAGraphs，dynamo 和 inductor pass，OOM bug，dynamo 图大小` 


- **探究 Torch CUDAGraphs 的 VRAM 使用数学逻辑**：在花费数天调试一个 **OOM bug** 后，一名成员询问了在使用 **torch CUDAGraphs** 以及不同的 dynamo 和 inductor pass 时，峰值 **VRAM** 使用量的逻辑/基础计算方法。
   - 根本原因是庞大的 dynamo 图大小，以及随之而来的庞大 **cudagraph** 大小，尽管根据权重和激活值计算，在 **GPU** 上进行推理在理论上是可行的。
- **Dynamo 和 CUDAGraph 大小导致 OOM**：用户确认 OOM 的根本原因是庞大的 dynamo 图大小和庞大的 **CUDA graph** 大小。
   - 尽管权重和激活值的理论内存占用足够低，但 OOM 仍然导致进程崩溃。


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1434991997952852110)** (1 messages): 

> `NVFP4 kernel 优化，NVIDIA Blackwell B200，CuTe DSL，CUTLASS 4.0，Dell Pro Max GB300` 


- **GPU MODE 联合 NVIDIA 和 Dell 推出 NVFP4 Kernel 竞赛**：GPU MODE 正与 **NVIDIA**、**Sesterce** 和 **Dell** 合作举办一场 Kernel 竞赛，重点是在 **Blackwell** 硬件上优化 **NVFP4** kernel，报名截止日期为 **2 月 13 日**，报名地址：[luma.com](https://luma.com/9n27uem4)。
- **竞赛重点：低比特深度学习 Kernel**：竞赛将围绕深度学习工作负载中常见的 **NVFP4** 格式低比特、单设备 Kernel 展开，使用 **CuTe DSL** 和 **CUTLASS 4.0**，参考代码可在 [GitHub](https://github.com/gpu-mode/reference-kernels) 上获取。
- **为 Kernel 性能高手准备的丰厚奖品**：总冠军将获得一台 **Dell Pro Max with GB300**，其他奖品包括为四个优化问题中表现优异者提供的 **NVIDIA DGX Spark + GTC 2026 门票**、**RTX 5090 + GTC 2026 门票**以及 **RTX 5080**。
- **提供 Blackwell B200 用于本地优化**：参赛者将可以免费使用本地的 **NVIDIA Blackwell B200** 来优化 Kernel，竞赛题目包括 **NVFP4 Batched GEMV**、**GEMM**、**Gated Dual GEMM** 和 **Grouped GEMM**。


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1434918847013720206)** (2 messages): 

> `Hopper FP8 实现，Blackwell 量化 Kernel` 


- **Hopper 的 FP8 Tensor Core 在 FP22/23 中累加**：在 **Hopper 架构**上，Tensor Core 的 FP8 实现是在 **FP22** 或 **FP23** 中累加结果的，这一发现归功于 **DeepSeek** 和 **Sage Attention** 的作者。
   - 许多 FP8 GEMM 实现会定期将累加器提升（promote）到 FP32，这可能会牺牲一些性能。
- **Blackwell 在量化 Kernel 中避免了类型提升**：有人询问在量化的 **Blackwell kernel** 中是否需要定期提升，或者 **Blackwell** 是否原生在 **FP32** 中累加。
   - 回答是：*据我所知，在 Blackwell 上不需要*。


  

---

### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1433905375383523522)** (1 messages): 

> `Opportunistic Parallel Lambda Calculus, Opal Scripting Language, LLM Performance Optimization` 


- **Opal 通过并行化 Lambda 提升速度**：一篇新的 [论文](https://doi.org/10.1145/3763143) 介绍了 **Opal**，这是一种使用机会性求值（opportunistic evaluation）来自动并行化独立外部调用的脚本语言，旨在增强 LLM 和其他 API 的性能，代码托管在 [GitHub](https://github.com/stephenmell/opal-oopsla2025-artifact)。
- **Opal 在 LLM 任务上超越 Python**：与标准顺序执行的 Python 相比，Opal 在总运行时间上实现了高达 **6.2 倍** 的提升，在延迟上实现了 **12.7 倍** 的提升，其性能足以媲美手动调优的异步 Rust，而运行时间开销仅为 **1.3% 到 18.5%**。
- **Tree-of-Thoughts 受益于 Opal**：论文展示了 **Opal** 将 **Tree-of-Thoughts**（一种著名的 LLM 推理方法）的性能提升了 **6.2 倍**（与作者自己的实现相比）。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1433895806963945512)** (29 messages🔥): 

> `Dusty's Retirement, Pip Index URL Correction, Performance Profiling Tools, CUDA Advent of Code Optimization, High Dimensional Probability and Neural Nets` 


- **Dusty 退休，新维护者接棒！**：在 Dusty 退休后，一名成员宣布他们现在成为了维护者，另一名成员对此消息表示惊讶。
- **Pip Index URL 混淆需要手动修复**：有成员报告称，`dustynv/pytorch:2.7-r36.4.0-cu128-24.04` 容器的默认 **pip index-url** 是错误的，需要用户手动指定为 `https://pypi.jetson-ai-lab.io/jp6/cu128` 而非 `https://pypi.jetson-ai-lab.dev/jp6/cu128`。
- **性能分析利器**：成员们讨论了性能分析工具，建议将 **extrae** 和 **Night systems**（疑为 Nsight Systems）用于 CUDA 和 OpenMP 代码，并推荐将 **Intel Advisor** 专门用于 OpenMP。
- **CUDA 代码引发难题**：一名成员就如何针对 Advent of Code 问题优化其 CUDA 代码寻求建议，并指出其 GPU 实现比 CPU 版本还要慢。
   - 建议包括：确保 **合并全局内存访问 (coalesced global memory access)**、使用 **共享内存 (shared memory) 作为显式缓存**，以及考虑 **向量化内存访问 (vectorized memory access)**。
- **内核工程与编译器的联系**：一名成员询问学习编译器是否有助于成为更好的内核工程师（Kernel Engineer），以及两者之间是否存在联系。
   - 另一名成员指出 [Nvidia Cuda Compiler (NVCC)](https://developer.nvidia.com/cuda-llvm-compiler) 是基于 **LLVM** 的，因此学习 LLVM 可能会有所帮助，并能辅助为 NV GPU 创建自定义 DSL。


  

---


### **GPU MODE ▷ #[jax-pallas](https://discord.com/channels/1189498204333543425/1203956655570817034/1434340464340762624)** (1 messages): 

> `Pallas:MGPU matmul kernel, NVLINK comms, all-gather collective matmul, all-to-all -> grouped GEMM` 


- **Pallas:MGPU 内核实现集合通信**：只需对 **Pallas:MGPU matmul kernel** 进行少量修改，即可将其转换为 **all-gather 集合通信矩阵乘法**，从而实现 **NVLINK 通信** 与本地计算的重叠（overlap）。
- **All-to-All GEMM 推测**：一名成员想知道 **all-to-all -> grouped GEMM** 是否也能实现同样的效果。


  

---

### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1433972224506724384)** (16 条消息🔥): 

> `TorchAO FP8 quantization bug, GemLite Performance, Profiling Inference Optimization, MXFP/NVFP large batch sizes, cudagraphs` 


- **TorchAO FP8 量化面临 Bug 指控**：一位用户报告了 **TorchAO 默认 FP8 量化**中可能存在的 Bug。在两块 RTX 5090 GPU 上使用 `torchao.quantization.Float8WeightOnlyConfig` 时，观察到 **Llama 3.1-8B 的推理速度仅为 7.8 tps**。
   - 有建议指出，显式使用带有 **mxfp8** 的 GemLite 内核可以产生更合理的运行速度，该用户承诺将编写一份关于分析/推理优化的指南。
- **GemLite 配置调优提升 Token 吞吐量**：一位用户最初获得低性能是因为他们*对 triton 自动调优（autotuning）过程有点不耐烦，最终使用了 GemLite 自带的默认 4090 和 5090 配置。*
   - 用户可以预期 **4090 上使用 GemLite 的速度为 160-170 tokens/sec，而在 5090 上超过 200 tokens/sec**。 
- **推理分析与优化指南正在编写中**：为了响应对追踪（trace）分析的需求，一位用户正在创建 [Inference Profiling and Optimization Guide](https://github.com/vipulSharma18/Inference-Profiling-and-Optimization-Guide)。
   - 该指南旨在帮助他人理解为什么在 4090 上使用 GemLite 无法达到约 150 tps 的速度。
- **MXFP/NVFP 在大 Batch Size 下表现出色**：一位用户了解到 **MXFP/NVFP** 应该在大 Batch Size 下进行评估，因为它们是为计算受限（compute-bound）场景设计的。
   - 此前该用户仅测试了 Batch Size 1，导致 **GPU 利用率不足**。他参考了[这个 Gist](https://gist.github.com/mobicham/090a78ddb64ce425d674ec9b286d1bd8) 以获取性能见解。
- **CudaGraphs 对性能的作用被揭示**：该用户指出，在其实现中*你没有使用 CudaGraphs 而我使用了*，且组大小（group sizes）不同，这在小 Batch Size 下会产生 FLOP 开销。
   - 另一位用户声称也*使用了 CudaGraphs，但在我的自定义实现中不是通过 torch.compile 使用的，但我认为你应该能获得与 gpt-fast 类似的性能*。


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1433969545131065527)** (6 条消息): 

> `Metal for GPU programming, M5 chip Tensor API, Metal for iOS platforms, Torchao metal kernels, Metal talks by Nikita and Manuel` 


- **Metal 学习者投身 GPU 领域**：一位新的 GPU 程序员决定使用 **Metal** 进行 GPU 编程，并于近期记录了一个 [Metal GEMM kernel](https://percisely.xyz/gemm)。
   - 他们预见到会有挑战，并渴望利用社区的专业知识。
- **M5 芯片的 Tensor API 引起研究人员关注**：有人询问了 **M5 芯片**及其新的 **Tensor API**，推测它可能比之前的版本提供更好的加速。
   - 他们对可实现的 **FLOPs** 表示好奇。
- **在 iOS 上使用 Metal 引起兴趣**：一位成员询问了在 **iPhone/iOS** 平台上使用 **Metal** 的情况，寻求在 **Metal** 硬件上优化模型（**LLM** 或非 LLM）的文章和平台。
   - 另一位成员提到了用于量化的 **Torchao** Metal 内核，并建议它们应该可以在手机或 Mac 上运行。
- **Torchao 量化内核点亮 Metal**：一位成员强调了 **Torchao** 中有趣的用于量化的 **Metal 内核**，这些内核应该可以在手机或 Mac 上运行。
   - 这些内核为在 **Metal** 硬件上进行优化提供了潜在途径。
- **Nikita 和 Manuel 提供 Metal 见解**：服务器上分享了两个关于 **Metal** 的演示，一个来自 **Nikita**，另一个来自 **Manuel**。
   - 这些讨论可以为在不同应用中利用 Metal 提供见解。


  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1434618001784701000)** (2 messages): 

> `K&R C Exercises, TensorDiagram Python Library` 


- **K&R 练习激发 C 语言学习**：一位成员正在实现 K&R 书中的练习以加深对 **C** 的理解，并分享了一个包含第一章练习答案的 [repo 链接](https://github.com/simveit/fun_with_c/tree/main/chapter_01)。
   - 他们还分享了一篇关于其 **C** 语言学习历程的 [LinkedIn 帖子](https://www.linkedin.com/posts/simon-veitner-174a681b6_fun-with-c-kr-book-is-often-considered-activity-7390824174291898368-V-AB?utm_source=share&utm_medium=member_desktop&rcm=ACoAADJYtOgBMOUI4WiWTyiAkroFBP1VujIWeksBorn)。
- **TensorDiagram 可视化复杂张量操作**：一位成员介绍了 **TensorDiagram**，这是一个用于张量可视化的 [Python 库](https://github.com/hardik-vala/tensordiagram)，旨在简化复杂的张量操作，如 **amax**、**kron** 和 **gather**。
   - 该库旨在与 **Colab/Jupyter** notebooks 以及其他 Python 环境配合使用，提供无缝的可视化体验，如[附图](https://cdn.discordapp.com/attachments/1288557096404516945/1434631872507412633/tensordiagram_launch.png?ex=690a5a0c&is=6909088c&hm=c4fe9d584d33b9a69ae49ab1a090a5b59c673cbf6a87b4d463ed6a9b6c1f4496)所示。


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1434513551787167826)** (2 messages): 

> `2cta matmul b200 performance, pipeline stalls, sparsity` 


- **B200 2cta Matmul 达到理论 SOL 的 35%**：一位成员测试了来自 thunderkittens kernel 的 **B200** **2cta matmul**，达到了 **3173 tflops**，约为 **fp8** 理论 SOL 的 **35%**。
- **调查流水线停顿 (Pipeline Stalls)**：该成员怀疑性能瓶颈源于**流水线停顿**，考虑到该 kernel 采用了带有环形缓冲区的 **4 阶段流水线**设计。
- **通过稀疏性 (Sparsity) 达到 70%+**：另一位成员指出 **9 pflop** 的数值是针对 **2:4 sparsity** 的，换算成 **4.5 peak dense flops**，这使得已实现的 **70%+** 性能显得“相当不错”。


  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1433841500503937167)** (3 messages): 

> `Python Serving for Large Models, TorchScript Overhead, vLLM Custom Model API, torch.compile with reduced-overhead` 


- **Python Serving 在大模型领域占据主导地位**：对于大模型，共识是基于 **Python 的服务化 (serving)** 速度同样快且更易于处理，[vLLM](https://github.com/vllm-project/vllm) 就是其在标准 LLM 领域取得成功的典型案例。
   - 有人指出，*对于那些规模小得多且对框架开销非常敏感的模型*，**TorchScript** 从未解决这些问题，因为其开销与 **torch+python** 相同，所以还不如直接运行 Python。
- **通过 C++ 编码绕过 torch+python 开销**：有人提到，如果你的环境是 **C++** 且无法嵌入 Python，可以采用多层解决方案，由 Python 的 **vLLM** 服务支撑你的主 **C++** 服务。
   - **ExecuTorch** 技术上也是一个 **C++** 运行时，但其 CUDA 后端在严肃的生产环境中尚未经过充分测试。
- **hf/hub 和依赖锁定在模型服务化方面优于 torch.package**：为了将模型冻结为易于移动和服务化的稳定制品，虽然我很想推荐 **torch.package**，但现实是 **hf/hub** 和**依赖锁定 (dependency pinning)** 很可能（从 ML 工程师的角度来看）最简单，因此也最可靠。
   - 虽然 **torch.package** 确实有效，但其获得的支持非常有限。
- **建议使用 vLLM Custom Model API 以避免 torch+python 开销**：建议检查是否可以针对你的用例使用 **vLLM custom model API**，因为这将对你有所帮助，且最易于维护，并能提供目前及未来可能的最佳性能。
   - 建议不要在此类用例中试图刻意避免 **torch+python 开销**。
- **使用带 reduced-overhead 的 torch.compile 获得快速收益**：如果时间有限，使用带有 *reduced-overhead* 的 **torch.compile** 是个不错的选择，并且根据你的服务器解决方案，有一些处理缓存的技巧，类似于 **vLLM** 中的内置缓存。
   - 如果你有更多时间，可以研究 **custom kernel**、**manual cudagraph** 及类似技术，以获取最后百分之几的性能提升。


  

---

### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/1434987539088937082)** (3 messages): 

> `Compute Limitations, Inference Optimizations, Chinese AI Community` 


- **算力限制引发 AI 辩论**：一位成员询问是否必须拥有体面的 **GPU** 才能进行 **inference optimizations**，或者是否因为算力限制而可以在没有它的情况下进行。
   - 一位版主请求将讨论移至 <#1191300313928433664>。
- **成员寻求中国 AI 专家的建议**：一位成员表示希望向该领域的中国专家寻求建议，并认可他们在该领域的实力。
   - 该请求是用中文提出的，表明了与 **Chinese AI community** 互动的特定兴趣。


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1435026737745625233)** (3 messages): 

> `Nvidia competition submission portal, Submission via Discord bot` 


- **Nvidia 竞赛提交入口位置已明确**：一位成员询问了正在进行的 **Nvidia competition** 提交入口的位置。
   - 另一位成员澄清说，可以通过 [Discord bot](https://discord.com/channels/1161594854998491166/1343002583001726986)、[CLI](https://github.com/gpu-mode/popcorn-cli) 和 [web](https://www.gpumode.com) 访问提交入口。
- **强调了 Discord Bot 提交方式**：Discord bot 提交方式被明确提及为选项之一。
   - 这表明该机器人功能齐全，并正式支持竞赛提交，使其成为 CLI 和网页界面的可行替代方案。


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1434961660577320991)** (10 messages🔥): 

> `GPU prices, Neo Clouds vs Hyperscalers, NvLink bridges, Hyperscaler support, Voltage Park support` 


- **GPU 价格再次上涨**：由于全球供应短缺，GPU 价格再次飙升，新兴云厂商 (Neo Clouds) 的价格稳定在 **$2.00 / GPU hour** 左右，而超大规模云厂商 (Hyperscalers) 则接近 **$7.00 / GPU hour**。
- **NvLink 桥接器绕过 PCIe 限制**：**NvLink bridges** 旨在绕过 **PCIe limitations**，而扩展 Vram 实际上只对那些尝试在消费级硬件上进行企业级工作的人有用。
- **Hyperscalers 需要批量折扣**：Hyperscalers 需要批量折扣，真正的折扣直到每年的支出达到 **数百万美元** 时才会生效，因此即便如此，你也无法在 Hyperscaler 获得 Neo Cloud 的定价。
- **Neo Clouds 专注于 AI/ML 基础设施**：Hyperscalers 工程团队的复杂性和规模非常庞大，因此他们需要工程人员来建立和维持运营，而 **Neo Clouds** 则不需要考虑所有额外的复杂性，因为他们专注于 **AI/ML Infra**。
- **Voltage Park 提供现场支持**：一位成员表示，**Voltage Park** 在其所有数据中心都有自己的驻场人员，硬件故障可以在数小时内而非数天内得到修复，此外还有一个由真正的 AI/ML 基础设施工程师组成的全球支持团队，可以在一天中的任何时间解决问题。


  

---


### **GPU MODE ▷ #[tpu](https://discord.com/channels/1189498204333543425/1351571759761068082/1433987755758583898)** (2 messages): 

> `Protobuf Size Limit, JIT Disabling Effects` 


- **达到 Protobuf 大小限制**：一位成员在尝试使用 `JAX_DISABLE_JIT=1` 禁用 JIT 跟踪程序时遇到了 **protobuf size limit error** (*tensorflow.profiler.XSpace exceeded maximum protobuf size of 2GB*)。
   - 同一个程序在不禁用 JIT 的情况下 **运行正常**。
- **JIT 影响 Protobuf 大小**：用户寻求深入了解为什么 **disabling JIT** 会导致 **protobuf size** 超过限制。
   - 他们还在寻找该问题的潜在 **workarounds**。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1434925751479570526)** (6 messages): 

> `pip install errors, RL work with Factorio, Sonnet 4.5 distillation, Qwen3-8b-VL-Thinking SFT` 


- **Pip Install 故障排除**：一位用户报告 `pip install` 命令无法工作，特别是 `pip install factorio-learning-environment[eval]`。
   - 另一位用户建议尝试添加引号：`pip install "factorio-learning-environment[eval]"`，结果成功了。
- **针对 Factorio 的 RL 工作计划浮出水面**：一位成员热衷于在 FLE 的基础设施就绪后进行一些 **RL work**，并提到计划将 **Sonnet 4.5** 蒸馏到 **Qwen3-8b-VL-Thinking** 中。
   - 该策略涉及通过自定义 **SFT** 来学习如何正确处理 **Factorio images**，然后将其直接挂载到针对游戏内生产分数的 **RL loop** 中。


  

---

### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1433841915010355383)** (2 messages): 

> `a2a solution, theoretical throughput` 


- **a2a 解决方案博客发布**：一名成员分享了关于其 **a2a 解决方案**的博客文章。
   - 博客文章地址：[https://gau-nernst.github.io/amd-a2a/](https://gau-nernst.github.io/amd-a2a/)。
- **SoL 比 AMD 快 10 倍**：一名成员提到 AMD 比 **SoL (Speed of Light，理论性能极限) 慢 10 倍**。
   - 另一名成员询问了峰值理论吞吐量的百分比，但未给出具体数字。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1433976053172535366)** (10 messages🔥): 

> `CUTE_DSL_LINEINFO, kernel_cutlass_kernel_flash_attncuteflash_fwdFlashAttentionForwardSm90, TiledCopy, make_layout_tv, raked_product and right_inverse` 


- **Compute Sanitizer 揭示 Flash Attention Kernel 中的全局读取 Bug**：Compute Sanitizer 报告了 Flash Attention Kernel `kernel_cutlass_kernel_flash_attncuteflash_fwdFlashAttentionForwardSm90` 在 `flash_fwd.py:788` 处存在无效的 `__global__` 读取，通过使用 `export CUTE_DSL_LINEINFO=1` 暴露了行号。
   - 原本希望 `CUTE_DSL_LINEINFO=1` 能提供更细粒度的帧信息，但调试器仅指向设备编译的 Kernel。
- **TiledCopy 差异引发讨论**：对比了两个布局略有不同的 `TiledCopy` 实例化：`Layout<Shape<_8, _2>, Stride<_2, _1>>{}` 与 `Layout<Shape<_1, _2>, Stride<_1, _1>>{}`，其中一个使用列优先 (column-major)，另一个使用行优先 (row-major) 的值布局。
   - 一名成员表示：“这两个 V-layout 看起来不同，但它们是两个相同的映射。f(0) -> 0, f(1) -> 1”，如果对它们应用 coalesce，两者最终都会得到 `2:1`。
- **`make_layout_tv` 的 C++ 等效实现**：一名成员询问了 CuTe 函数 `make_layout_tv` 的 C++ 等效实现，该函数在 elementwise_add kernel 中用于创建 tv_layout。
   - 另一名成员建议 `make_tv_layout` 是一个由 `raked_product` 和 `right_inverse` 组成的简单辅助函数，可以在 C++ 中实现。


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/)** (1 messages): 

matt.pd: 通过 FP16 解决训练与推理不匹配问题 (Defeating the Training-Inference Mismatch via FP16)
https://arxiv.org/abs/2510.26788
  

---


### **GPU MODE ▷ #[llmq](https://discord.com/channels/1189498204333543425/1421956177549332662/1434209098710257817)** (1 messages): 

> `LLMQ, Python Bindings, Multi-threaded backend` 


- **LLMQ 推出 Python 接口**：一名成员创建了第一个可从 **Python 访问的 LLMQ** 版本（这是在一次线下黑客松中提出的建议），[在此处构建了 wheel 文件](https://github.com/IST-DASLab/llmq/actions/runs/19000187973)。
   - 让 **多线程后端 (multi-threaded backend)** 与 Python 协同工作非常“有趣”，但他们现在基本已经解决了。
- **NCCL 线程变为僵尸线程**：在关闭时，**NCCL 线程会挂起**并保持僵尸状态，直到 Python 进程退出。
   - 这是第一个 Python 绑定版本中已知剩余的唯一问题。


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1433996881473572884)** (14 messages🔥): 

> `Helion Autotuning, Helion Performance, Determinism Control` 


- **Helion 的自动调优 (Autotuning) 引发争议**：一些成员讨论了 **Helion** 是否应该默认关闭自动调优并显示警告消息，因为每次都要手动禁用它很不方便，尤其是在开发过程中。
   - 此前 **Helion** 默认不进行自动调优，但用户抱怨由于未调优导致性能不佳，因此现在采用了“默认开启、手动退出 (opt-out)”的方法，以避免不正确的性能对比；可以使用 `HELION_AUTOTUNE_EFFORT=none` 来跳过自动调优。
- **自动调优进度受到关注**：一名成员建议在 **Helion** 中显示自动调优进度，包括当前的性能进展，以便用户评估并在需要时停止该过程。
   - 开发人员目前跟踪每一代的最低、中等和最高性能，并且有一个 `HELION_AUTOTUNE_EFFORT=quick` 选项用于快速调优，尽管它可能无法达到最佳性能。
- **确定性控制引发辩论**：成员们讨论了在 **Helion** 中控制确定性的可能性，特别是仅在运行间确定 (run-to-run deterministic) 或与其他配置位等价 (bitwise equivalent) 的配置中进行自动调优。
   - 一名成员建议通过确定性自动调优来匹配 eager 模式的数值。


  

---

### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1434737607383973929)** (39 条消息🔥): 

> `Kernel Challenge, GPU Mode YouTube 频道, CUDA DSL Kernels, PMPP 书籍` 


- **Kernel Challenge 问题简报即将发布**：根据 [Luma 日历](https://luma.com/9n27uem4)，四个 **kernel-challenge** 问题的完整简报将在 kernel 开放时提供。
   - 注册仅为了获得获奖资格；提交代码不需要注册，但使用 **CLI/web** 需要关联 Discord。
- **Speed of Light 基准测试是可复现的**：kernel 问题的 *“speed of light”* 性能指标将会公布，以便在提交前进行复现。
   - 最终评估将在 Sesterce 的 **云端 GPU** 上进行，使用自定义 Docker 镜像，但 Sesterce 目前尚不支持 **CUDA 13**。
- **GPUMODE YouTube 频道内容非常出色**：整个 [GPUMODE YouTube 频道](https://www.youtube.com/@GPUMODE) 都是一个内容丰富的宝库。
   - 他们正在努力组织一系列关于 **DSL kernels** 的讲座，邀请来自 Nvidia 的演讲者。
- **PMPP 书籍适合初学者**：[PMPP 书籍](https://www.amazon.com/Programming-Massively-Parallel-Processors-Applications/dp/0124159767) (Programming Massively Parallel Processors) 是学习的一个很好的起点，这是一位成员推荐的。
   - 参与者可以使用任何他们想要的 kernel **DSL**（或手动编写 **CUDA/PTX**），只要 Python 评测脚本可以启动它即可。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1433838793844064422)** (101 条消息🔥🔥): 

> `研究中的引用, AI 辅助研究, Multi-head attention, Matrix vs Discord, Diffusion Model 训练不一致性` 


- **引用被认为是可选的？**：一位成员分享说，一位同事在博士论文提案中仅使用了 **8 个引用**，在讲座中引用少于 **10 个**，尽管对生成模型和因果发现提出了重大主张。
   - 这引发了关于选择论文的标准以及集中论文转储（paper dumps）价值的讨论。
- **AI 作为研究助手，行还是不行？**：社区讨论了使用 **AI** 进行 **AI 研究**，共识是验证主张和来源对于避免幻觉（hallucinations）和负面反应至关重要。
   - 一位成员引用了 **Andrej Karpathy** 的话，称 *做 AI 最好的方法就是尝试一堆模型解决方案，看看哪个效果最好*。
- **在 Multi-head attention 中切割 embedding 是可以的吗？**：一位成员质疑在 Multi-head attention 中将 **512 维 embedding** 拆分为更小的 head 的做法，担心这可能会丢失上下文。
   - 其他人解释说这是一种 **regularization**（正则化）形式，允许模型专业化并学习更稳健的特征，在每个 attention 步骤前后都有连接。
- **Matrix > Discord?**：社区讨论了迁移到 **Matrix** 的可能性，以便更好地支持读书会，理由是它没有对房间数量进行硬编码限制，并且具有联邦性质，允许用户从不同的服务器加入。
   - 一些人质疑去中心化频道的必要性，而另一些人则指出了互操作性的价值。
- **采样反向时间 SDE 会导致不一致？**：一位成员报告说，在相同的仓库、配置、数据和种子下，Diffusion Model 的不同训练运行表现出巨大的性能差异，这归因于 **反向时间 SDE 采样** 中的随机元素。
   - 有人建议模型学习到的近似分布看起来是一样的，但如果没有相同的种子，可能会观察到糟糕的生成批次；设计不当的 guidance 也会影响这一点。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1433890642584535181)** (17 条消息🔥): 

> `论文阅读录音, 线性可解释特征演化, 模拟 vs 数字, Awesome World Models` 


- **推荐 Awesome World Models**：一位成员分享了一个关于 **World Models** 的 [GitHub 仓库](https://github.com/knightnemo/Awesome-World-Models) 链接，供阅读和讨论。
- **预训练快照中的线性可解释特征演化**：有人建议讨论 [Linear interpretable feature evolution across pre-training snapshots using crosscoders](https://arxiv.org/abs/2509.17196) 及其对应的 [推文](https://fxtwitter.com/Dest1n1s/status/1970350739152408784?t=UaWoNGMU0x2_g0DpIIG0kw&s=19)。
- **关于论文阅读录音的询问**：一位成员询问是否有之前 **论文阅读** 会议的录音。
- **模拟优于数字？**：一位成员提出了 **模拟（analog）** 是否优于 **数字（digital）** 的问题，并参考了人脑的工作原理。


  

---

### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1434232845806735521)** (2 条消息): 

> `推进 Agents、Reasoning、Memory、AGI 的讨论` 


- **演讲启发 Agent 进阶**：一位成员对一场讨论表示感谢，该讨论激发了对推进 **Agents**、**Reasoning** 以及潜在的 **Memory** 功能至关重要的机制之间的联系。
   - 该讨论被赞誉为功能上的进步，唤起了一种 *AGI 感*，并希望继续探索其潜力。
- **AGI 潜力凸显**：有影响力的论坛成员认为这次演讲代表了迈向 **AGI** 的一步，强调了其在该领域的重要性。
   - 参与者对保持讨论势头并开启进一步突破的可能性表示乐观。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1434068287909527583)** (10 条消息🔥): 

> `AI 泡沫、经济衰退、劳动力削弱` 


- **AI 泡沫被指责为就业危机的原因**：一篇 [Fortune 文章](https://fortune.com/2025/10/30/jerome-powell-ai-bubble-jobs-unemployment-crisis-interest-rates) 讨论了 **AI 泡沫** 如何导致 **失业** 和 **失业危机**。
   - 成员们辩论了 **AI** 还是 **政府无能** 对当前的 **经济衰退** 负有更大责任。
- **成员分享关于 AI 影响的论文**：成员们分享了论文链接，例如来自 **Stanford Digital Economy Lab** 的 [Canaries in the Coal Mine: Early Warnings of Technology-Driven Displacement](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5425555)。
   - 这些论文讨论了 **AI** 对就业市场的影响。
- **工资与生产力脱钩**：成员们链接了 [关于工资与生产力脱钩的维基百科文章](https://en.wikipedia.org/wiki/Decoupling_of_wages_from_productivity)，将其归因于 **劳动力议价能力的削弱**。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1433867780653846578)** (110 条消息🔥🔥): 

> `Kimi CLI, Agent Mode, DeepAgents CLI, Poolside 估值, Redpanda Data AI` 


- ****Kimi CLI** 发布终端专用工具**: **Moonshot AI** 发布了 **Kimi CLI** 的技术预览版，具备 **Zsh 集成**、**MCP 支持**，并原生挂载到 **Zed 编辑器**，其 [GitHub 仓库已开放反馈](https://xcancel.com/Kimi_Moonshot/status/1984207733177090274)。
   - VIP 用户可免费获得新的 **"Kimi For Coding"** 插件；早期反馈提到安装时出现 **401 错误**，用户对终端工作流充满热情，并请求支持 Windows 和试用计划。
- **OpenAI 的 Agent 模式引发热议与疑虑**: **OpenAI** 宣布为 **ChatGPT**（Plus/Pro/Business）发布 **Agent/Atlas 模式** 预览版，使模型能够代表用户进行浏览和执行操作；参见 [公告](https://xcancel.com/OpenAI/status/1984304194837528864)。
   - 表达的担忧包括提示词注入（prompt-injection）攻击、缺乏明确的护栏、可靠性问题，以及在有用的自动化与隐私侵蚀之间的伦理界限。
- **LangChain 发布 **DeepAgents CLI****: **Harrison Chase** 介绍了 **DeepAgents CLI**，这是一个基于新 deepagents 包构建的示例编码应用程序，可以在不同会话间保留指令和引导；参见 [LangChain 博客文章](https://xcancel.com/hwchase17/status/1984303925101735950)。
   - 定位为可定制 Agent 的 **"开放框架 (open harness)"**，社区成员已经在询问 MCP 集成和向量数据库等外部存储源。
- **对 **Poolside 120 亿美元估值** 的质疑**: **Julien Blanchon** 发布推文称 *"好吧，做空一切！"* 并配上一张照片，发起了一个帖子，法国科技圈内人士在其中嘲讽 Poolside 的 **120 亿美元估值**，并指责其多次转型。
   - 评论者指出，该公司曾宣传自己是 **“Cursor 之前的 Cursor”** 但从未出货，且多次转型（SFT-as-a-service, RL-as-a-service，现在是 “Cursor-as-a-service”），在巴黎的见面会上几乎见不到踪影——导致被指控为在加勒比避税天堂运行的虚假软件（vaporware）；参见 [推文链接](https://xcancel.com/julienblanchon/status/1984337407097909629?s=46)。
- ****AI 大逆转 (AI Flippening)** 随着中国开源模型的激增而到来**: **Balaji Srinivasan** 宣布 **“AI 大逆转”** 已经到来，声称中国开源权重模型（**DeepSeek**、**Qwen** 等）现在的下载量已超过且性能日益优于西方竞争对手，使 AI 软件商品化并挤压利润空间；参见 [推文链接](https://xcancel.com/balajis/status/1984421276082192663?s=46&t=Z6mP_1pHALnIw7k1lFkdwQ)。
   - 他认为中国的策略是用免费/廉价模型让美国 AI 公司破产，然后通过支持 AI 的硬件获利；但这一举动是否应通过下载量与收入、西方的能源赤字、后门风险以及开源主导体制下的下一波创新来衡量，仍存在争议。


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 条消息): 

swyxio: 与 <@367104793292046338> 和 <@194927177265840128> 合作的新 pod！https://youtu.be/-gE1cesJF9M
  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1434049373708357754)** (5 条消息): 

> `X-Ware.v0, OpenAI X 帖子, YouTube 视频` 


- **OpenAI 发布 X 帖子**: 一名成员分享了 **OpenAI** X 帖子的[链接](https://x.com/openai/status/1984318204374892798?s=46)。
   - 另一名成员分享了相同的链接，可能是在强调其重要性。
- **频道内分享 X-Ware.v0**: 成员们在聊天中分享了术语 `Red - X-Ware.v0`，没有提供其他上下文。
   - 目前尚不清楚 **X-Ware.v0** 指代什么，尽管它可能与分享的 **OpenAI** X 帖子有关。
- **发布 YouTube 视频**: 一名成员发布了一个 [YouTube 视频链接](https://www.youtube.com/watch?v=YpuSE9hcal8)。
   - 一位用户评论说 *他的视频总是很棒*。


  

---

### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1433908333470810162)** (98 messages🔥🔥): 

> `Kimi K2 Research & OK Computer 重置, K2 Think 模型 vs. Cerebras 混淆, Qwen QWQ 微调, Minimax vs. GLM 日常任务对比, Claude Code Max vs. Cursor Pro+ 使用限制` 


- **Kimi 的 Research 和 OK Computer 配额保留**：一位用户确认 **OK computer** 和 **Researcher Quote** 在一个月后不会重置，这表明它可能是一次性配额。
   - 未提供关于这些功能的具体细节或获取方式的进一步背景。
- **K2 Think 引发 Cerebras 混淆**：一位用户澄清 **K2 Think** 不应与托管在 **Cerebras** 上的模型混淆。
   - 另一位用户认为考虑到现有的 K2，选择这个名字很奇怪，并暗示该模型的表现并不理想。
- **Kimi 是 Qwen QWQ 的微调版吗？**：用户推测 **Kimi** 可能是 **Qwen QWQ** 的微调版，其中一人指出它与 **Qwen QWQ** 相似，可能是一个经过后训练的数据集。
   - 另一位用户指出 **Qwen QWQ** 基于 **Qwen 2.5 32B**，并链接了一个关于基于 QWQ 的“古老模型”的 [YouTube 视频](https://www.youtube.com/watch?v=l3d-m3uP3nQ)。
- **Minimax 受欢迎程度上升**：一位用户分享了使用 4-5 天后的感受，认为 **M2 在大多数任务上优于 GLM-4.6**，尤其是作为日常主力工具，因为它不会陷入局部细节（tunnel-visioned）。
   - 另一位用户报告称 **Minimax** 已成为他们的首选，特别是在网页搜索和创建其他 AI 难以处理的特定格式报告方面。
- **代码补全订阅讨论**：用户讨论了各种编程工具的使用限制，一人对比了 **Claude Code Max** 和 **Cursor Pro+**，指出 **Claude Code Max ($200)** 拥有更高的限制。
   - 一位用户提到 **Cursor Composer** 速度很快，但另一人澄清 **Claude Code Max** 是 200 美元的选项，引发了关于每周使用量的讨论。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1434081123809497168)** (10 messages🔥): 

> `Multi-Head Attention, EleutherAI 贡献, 导师指导` 


- **Multi-Head Attention 维度解析**：成员们讨论了 [multi-head attention](https://transformer-explained.github.io/attention) 以及 **512 维嵌入**如何投影到 **Q, K 和 V** 并拆分为 8 个各 64 维的头。
   - 他们建议模型是按照这种维度切片进行训练的，这意味着你可以*沿着该维度进行置换（permute），然后切片，在训练后得到完全相同的结果*。
- **欢迎参与 EleutherAI**：一位刚接触开源和 Discord 协作的成员询问如何参与或贡献于 **EleutherAI** 的研究。
   - 另一位成员指引其查看**置顶消息**和 **#general 频道**以获取指导。
- **寻求 ML/DL 导师**：一位成员正在寻求一位经验丰富的专业人士进行 **ML/DL** 方面的**高水平导师指导**，以提升技能并获得偶尔的反馈。
   - 他们强调自己动力十足且尊重导师的时间，并愿意分享他们正在进行的项目。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1433990171392606368)** (14 messages🔥): 

> `RL 崩溃, Flash Attention, 梯度归一化, LLM-RL 库, HF 模型幻觉` 


- **速度破坏 RL 训练的稳定性**：一位成员分享了一篇[论文](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda)，讨论了训练与推理之间的不匹配如何导致 **RL 崩溃（RL collapse）**。
- **梯度归一化重写层操作**：一位成员建议，在反向传播的 reduction 之前对梯度进行归一化（谱归一化、L2 等），可以通过重写线性层操作并在其后立即添加 reduction 来轻松实现。
- **veRL 提供快速且可定制的 RL 研究**：一位成员询问除了 TRL 之外，还有哪些用于活跃算法研究的 LLM-RL 库，另一位成员推荐了 [veRL](https://github.com/volcengine/verl)。
- **Qwen 模型在长尾事实中产生幻觉**：对下载量最高的 HuggingFace 模型进行的评估结果显示，**Qwen 模型**倾向于在长尾事实中产生幻觉，而且一些较热门的模型在遵循指令方面表现并不理想；完整结果可在 [HuggingFace space](https://huggingface.co/spaces/PropensityLabs/LLM-Propensity-Evals) 查看。


  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1433836454244520137)** (26 messages🔥): 

> `Transformers on Sequence Space, End-to-End LLM Outputs, Mech Interp and Probing Work, LLM Privacy / Security Claims, Activation Sharing Risks` 


- **Transformer 映射的是序列空间，而非嵌入空间**：一位成员指出，Transformer 被视为**序列空间（sequence space）**上的映射，而非嵌入空间（embedding space），并对某篇论文中提到的观点表示反对。
   - 他们质疑该论文论点的目标受众，暗示其误导了对 Transformer 的普遍理解。
- **论文错误地声称 LLM 是端到端的**：一位成员批评了某篇论文关于“端到端（end-to-end）”的说法，认为 LLM 的输出是 **Token 序列**或自然语言，而非隐藏状态（hidden state），这使得该说法失效。
   - 他们断言，当输出是 Token 序列或自然语言字符串时，这种端到端过程是不成立的。
- **LLM 隐私声明误导了引用的文章**：某篇论文关于隐私和安全的声明因误导引用文章而受到批评。被引用的文章讨论的是 **LLM 权重**存储来自训练数据的信息，而非用户输入。
   - 该成员强调，引用文章的摘要明确提到了符合 **GDPR** 要求，这与该论文的解读相矛盾。
- **概念注入检测性的详细说明**：一位成员详细阐述了模型如何通过异常检测来识别注入的概念，解释称模型会检测扰动何时将激活（activations）推离预期的内部表示。
   - 他们提出，模型的激活遍历一个相当平滑的语义流形，因此当注入的概念引入一个与该局部流形不一致的向量时，下游机制可以检测到这种偏差。
- **寻求可解释性研究职业建议**：一位成员表达了转向 **Interpretability** 和 **AI Safety** 应用研究的兴趣，并寻求关于阅读论文和加入项目的建议。
   - 他们提到了自己在 NLP 和计算语言学方面的背景，以及近期对 AI Engineering 的关注，同时分享了他们常用的资源 [“best of less wrong”](https://www.alignmentforum.org/bestoflesswrong?year=all&category=all)。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1434296011890888745)** (1 messages): 

> `MMLU Benchmark, Image Analysis` 


- **图片中对 MMLU 基准测试的批评**：分享了一张批评 **MMLU 基准测试**的图片，标签为 *“anti mmlu pro slander”*。
   - 该图片暗示了对该基准测试的负面情绪。
- **图片分析背景**：分享的图片与社区内关于 **MMLU 基准测试**的讨论或情绪相关。
   - 它暗示了对该基准测试的实用性或有效性的批判性视角。


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1434526408700465265)** (2 messages): 

> `VLM image description order, VLMs describing image collages` 


- **VLM 从左到右描述拼贴图**：一位成员报告称，在使用 VLM 描述图像拼贴时，描述始终遵循从左到右的顺序，即使是针对阿拉伯语数据（**AIN**）微调的 Qwen 2 VL 模型也是如此。
   - 另一位成员建议研究 VLM 架构，重点关注图像如何被处理并与文本集成，以理解并可能解决这种行为，并进行*实验*来测试假设。
- **理解用于图像处理的 VLM 架构**：为了解决 VLM 中一致的从左到右描述顺序问题，有人建议**研究 VLM 架构**及其组件，特别是图像如何被考虑并与文本集成。
   - 目的是开发可以通过实验测试的假设，以理解为什么 VLM 在描述拼贴图时会优先考虑这种特定顺序。


  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1433945748466045088)** (45 messages🔥): 

> `setup.py vs pyproject.toml, UOp.pyrender() bug, Tenstorrent backend bounty, Tiled matrix multiplication, PyTorch backend without hacks` 


- **经典的 setup.py 辩论仍在继续**：讨论围绕在 tinygrad 仓库中使用 `setup.py` 而非 `pyproject.toml` 展开，询问除了历史遗留原因外是否还有特定理由。
   - 一位贡献者寻求帮助，旨在为 `uop` 移动算子（movement ops）添加 `argfix` 并包含合理的测试。
- **未引用变量困扰 UOp.pyrender()**：一位用户报告了一个 bug，即 `some_forward().uop.pyrender()` 的结果包含未引用的变量，例如完全没被使用的 `cxx = UOp(Ops.VECTORIZE` 行。
   - 澄清了 `pyrender` 的输出旨在直接可执行并产生相同的 `uop`。
- **Tenstorrent 后端悬赏等待挑战者**：有人表达了对 **TinyGrad** 支持 **Tenstorrent** 的兴趣。
   - 提到有一个针对 **Tenstorrent 后端** 的[悬赏](https://blinry.org/tiny-linux/)，但目前还没人尝试；一位用户成功在静态链接的 Python 上运行了 **TinyGrad**，但只有 **PYTHON** 和 **NULL** 后端可用。
- **无 Tensor Cores 硬件上的矩阵乘法瓶颈**：提出了一个关于为没有 Tensor Cores 的硬件在 **TinyGrad** 中实现分块（tiled）矩阵乘法的问题，因为目前此类硬件上的 matmul 非常慢。
   - 同时确认 flash attention 的悬赏正在实施中。
- **修复 PyTorch 后端 Strides 的 PR**：一位新人询问了[电子表格](https://docs.google.com/spreadsheets/d/1WKHbT-7KOgjEawq5h5Ic1qUWzpfAzuD_J06N1JwOCGs/edit?gid=0#gid=0)中的任务及其更新状态，并指出一个可能对应但未注明的 PR (https://github.com/tinygrad/tinygrad/pull/13061)，用于“在不使用 hack 的情况下修复 PyTorch 后端的 strides”。
   - 该用户随后开启了一个用于修复 strides 的 WIP PR，但被要求在测试通过后再开启 PR。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1433891553910194206)** (40 messages🔥): 

> `Manus credit costs, Claude Code vs Manus, Manus image generation quality, Manus assistance with Instagram reels, Manus custom domain pricing` 


- **用户将 Manus 积分成本与替代方案进行比较**：几位用户对 **Manus 积分** 的高昂成本表示担忧，其中一位指出在一个中等复杂度的项目中，**一小时左右就消耗了 6k 积分**。
   - 用户建议使用 *Claude Code* 和 *GPT Codex* 的 **$20** 订阅比 Manus 更有性价比。一位用户表示：“与其他选项相比，Manus 贵得离谱”。
- **Claude Code 表现强劲**：一位用户对 *Claude Code* 表示满意，提到它交付成果的能力，尤其是对于编码，以及他们拥有 **24 个类别和 4k+ 问题** 的新智力问答游戏。
   - 该用户计划在触发速率限制（rate-limited）时在 *Claude Code* 和 *GPT Codex* 之间切换，预计**每周 5-6 天，每天持续编码约 8 小时**。
- **Manus 生成低质量图像**：一位用户质疑 Manus 生成的图像质量始终较低，并分享了一个[会话链接](https://manus.im/share/dRrj3dwepWuDcJKvfxRHPK?replay=1)来演示该问题。
   - 尽管在生成第二个思维导图时要求更高的质量，结果依然如故。
- **用户报告 Manus 无法解释 Instagram Reels**：一位用户报告称，Manus 之前可以解释 Instagram Reel，但现在拒绝这样做。
   - 未对这种不一致的行为提供解释。
- **用户称 200 美元的自定义域名定价是“敲竹杠”**：一位用户批评了通过 Manus 将自定义域名连接到 Web 应用的 **$200/月订阅费**，认为这是“敲竹杠”。
   - 另一位用户建议直接购买域名并独立设置，作为更便宜的替代方案。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1433846431864721520)** (33 messages🔥): 

> `Brokk AI Power Ranking, Perplexity MCP, aider vs aider-ce, Aider Community, Entangled Pair Quantum Eraser` 


- ****GPT-mini** 在 **Brokk AI Power Ranking** 中表现出色**: [Brokk AI 实力排名](https://brokk.ai/power-ranking)已更新，一位用户询问 **GPT-mini** 是否属于 S 级且排在 **Claude** 之上。
   - 另一位用户回答是的，但那是 *在* **Sonnet** 和 **Haiku 4.5** 发布之前，并补充说该结果存疑。
- ****Perplexity MCP** 与 **aider** 集成**: 一位用户发现 [Perplexity 的 MCP](https://www.perplexity.ai/) 在查找与某个已弃用的 **Android** 库相关的 **GitHub** issue 并将其集成到 **aider-ce** 时非常有用。
   - 该用户想知道集成 **MCP** 是否可以实现流程自动化，但也指出有时仍需要手动审查。
- ****aider-ce** 分支作为活跃的 fork 出现**: 用户注意到 **aider-ce** 非常活跃，而主 **aider** 仓库更新并不频繁；一位用户澄清说，主要维护者之一提到他最近一直很忙。
   - 一位用户询问：*“ce 版本有哪些让你喜欢的功能？”* 因为最近的一个 issue 请求不要覆盖 **aider** 命令，以便用户可以同时安装两个版本。
- **需要建立 **aider** 社区**: 一位用户询问是否可以围绕 **aider** 建立一个社区，并指出虽然 *大家都热爱它*，但它需要愿意投入时间的人。
   - 还有人提到，由于现在有更多具有 **Agent** 特性的产品可用，他们不再像以前那样频繁使用它，且 **aider** 可以加入上下文管理 UI 和 **MCP** 集成。
- **Paul 正在构建 **Quantum Aider****: 针对用户开玩笑说 Paul 正在创建一个量子版本的 **aider**，Paul 链接到了[他的项目](https://github.com/paul-gauthier/entangled-pair-quantum-eraser)。
   - 另一位用户对此表示担忧，询问在其他人取得进展的同时，如何保留 Paul 对该项目的 *深厚知识*，担心 *目前的真空状态会导致用户流失*。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1434888049724231761)** (5 messages): 

> `aider-ce, reasoning-effort, weak_model` 


- **查看更新后的 aider-ce**: 一位成员建议关注 [aider-ce](https://github.com/aider-chat/aider-ce)，指出它每周更新一次，在继承 **aider** 优势的同时增加了更多功能，详见 [路线图 (roadmap)](https://github.com/aider-chat/aider-ce/blob/main/README.md)。
   - 他们建议给该仓库点亮 Star 以示支持。
- **如何设置 reasoning-effort 和 weak_model**: 一位成员询问如何为 `--model ollama_chat/gpt-oss:20b` 和 `--weak_model` 设置 `/reasoning-effort`。
   - 即使设置了 `/weak_model ollama_chat/qwen3:1.7b` 似乎也不起作用，并导致警告提示 `ollama_chat/gpt-oss:20b` 不支持 `reasoning_effort`。


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1433846353708191888)** (21 messages🔥): 

> `MCPB, OCI, DXT, MCP Registry, server.json` 


- **MCPB vs. OCI：在重复造轮子吗？**: 一位成员质疑 **MCPB** 是否只是在 *重新发明* **OCI** 已经提供的功能。
   - 另一位成员澄清说，**MCPB** 是一个 **Anthropic** 项目（**原名为 DXT**），用于将 **MCP servers** 暴露给 **Claude**，提供环境变量的描述和类型，以便生成用户友好的配置表单，这与通用的 `servers.json` 或 `mcp.json` 不同。
- **MCP Registry 拥抱 MCPB**: 尽管最初存在困惑，但已确认 **MCP Registry** 支持 **MCPB**。
   - 据解释，该注册表旨在广泛支持各种注册表和包类型，类似于支持 **npm** 或 **PyPI**。
- **配置差异：MCPB vs. server.json**: 讨论强调 **MCPB** 是为桌面应用设计的，会呈现一个变量配置表单，而 `server.json` 通常直接定义变量值。
   - 注册表中的一个示例显示 `server.json` 已经包含了描述和类型，这表明注册表可以扩展此功能。
- **MCPB 创建者对 OCI 的了解程度**: 一位成员推测 **DXT/MCPB** 的创建者可能并不完全了解 **OCI** 以及注册表工作组现有的工作。
   - 有人建议，他们可能优先考虑具有表单填写能力的用户友好型软件包，而不是期望用户直接配置 **JSON** 文件。


  

---

### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1433937677706727674)** (7 messages): 

> `SEP-1442, SEP-1686, Statelessness proposals, Task storage` 


- **关于无状态提案的辩论**：成员们讨论了 **SEP-1442** 和 **SEP-1686** 是否存在冲突，因为一个旨在使服务器更加无状态，而另一个则引入了状态追踪。
   - 一位成员认为它们并不冲突，因为 Tasks 可以跨服务器扩展，而 **SEP-1442** 默认将会话信息移入每个请求以实现无状态，这主要是为了解决在负载均衡器后托管 MCP 服务器的挑战。
- **无状态旨在作为默认选项，而非完全无状态**：一位成员澄清说，无状态提案 (**SEP-1442**) 的目标是默认无状态，使有状态成为可选（opt-in）而非默认。
   - 将支持的协议版本和能力（capabilities）存储在外部数据存储中会使非会话服务器变得复杂，这引入了新的更新操作来解决该问题，从而使在请求中存储所有内容变得更简单。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1434292218616877178)** (1 messages): 

> `DSPyGen, DSLModel, Contributions to DSPy` 


- **Sean Chatman 发布 DSPyGen**：Sean Chatman 发布了 [DSPyGen](https://github.com/seanchatmangpt/dspygen)，这是一个用于 DSPy 开发的新工具。
   - 他也是 [DSLModel](https://github.com/seanchatmangpt/dslmodel) 的作者。
- **用户提议为 DSPy 贡献代码**：一位成员表达了对贡献 DSPy 的兴趣并提供了帮助。
   - 他们提到一直希望能重新参与 DSPy 的开发，并且已经研究过该项目的许多变体。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1433843785762209843)** (19 messages🔥): 

> `dspy.Tool with simple Predict, Rate limit issues for Gemini, DSCloj channel in DSPy, Force finish the ReAct, Accessing LM a module is using` 


- **对于简单任务，Predict 优于 ReAct**：一位成员发现 `dspy.Tool` 配合简单的 Predict 对于他们的用例来说已经足够，且比 ReAct 更高效，将响应时间从 **60 秒缩短至 9 秒**。
   - 他们表示 ReAct 对其用途来说是“大材小用”。
- **Gemini 速率限制（Rate Limit）令人头疼**：一位成员报告称，即使使用 10 个并行 worker，也触发了 Gemini **每分钟 100 万 token 的速率限制**，并询问在生产环境中缓解此问题的最佳实践。
   - 另一位成员建议，问题可能在于达到了**每日请求限制**或**每分钟请求限制**（取决于层级），可以在 Google AI Studio 中查看。
- **创建 DSCloj 频道**：有人请求在 DSPy 兄弟项目中为 DSCloj 创建专门频道，效仿 Rust 和 Typescript 频道的模式。
   - 该频道已创建，在确定最终名称前，双方就命名规范进行了一些讨论。
- **强制 ReAct 结束**：一位成员询问如何强制 ReAct 模块结束（可能基于工具的返回值），但未立即得到答复。
   - 该用户希望 Agent 在从工具返回时立即返回最终答案，但它在返回后仍继续运行。
- **用于动态模型切换的 LLM 访问**：一位成员寻求关于如何访问模块所使用的 LLM 的指导，以便在遇到速率限制时实现动态模型切换，询问如何向模块传递 LLM 并确保具备回退（fallback）机制。
   - 推荐的解决方案是将 `dspy.LM` 对象传递给模块的 init，并根据条件使用不同的 LLM，从而在发生错误时实现条件回退。