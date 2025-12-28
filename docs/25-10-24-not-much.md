---
companies:
- vllm_project
- nvidia
- mistral-ai
- baseten
- huggingface
- thinking-machines
- deeplearningai
- pytorch
- arena
- yupp-ai
- zhipu-ai
- scaling01
- stanford
date: '2025-10-24T05:44:39.731046Z'
description: '以下是该文本的中文翻译：


  **vLLM** 宣布支持 **NVIDIA Nemotron Nano 2**，该模型采用 Transformer-Mamba 混合设计，并配备可调节的“思考预算”，使令牌（token）生成速度提升高达
  6 倍。**Mistral AI Studio** 推出了一个具备深度可观测性的智能体（Agent）生产平台。**Baseten** 报告称，**GPT-OSS
  120B** 在 NVIDIA 硬件上实现了高吞吐量（650 TPS）。**Hugging Face InspectAI** 增加了推理提供商集成，以支持跨提供商的评估。**Thinking
  Machines Tinker** 抽象化了 **Qwen3** 和 **Llama 3** 等开源权重 LLM 的分布式微调流程。


  在中国，**MiniMax M2** 展现出与顶尖模型相当的性能，并针对智能体和编程进行了优化；而**智谱 GLM-4.6-Air** 则专注于编程任务的可靠性和扩展性。传闻称
  **Gemini 2.5 Flash** 可能是一个参数量超过 5000 亿的 MoE（混合专家）模型，同时还出现了疑似 **GPT-5.1 mini** 的参考信息。在
  LLM 领域之外，**Tahoe-x1 (3B)** 基础模型在癌细胞生物学基准测试中达到了 SOTA（最先进水平）。斯坦福大学的研究提出了一种通过训练顺序“重写本”（palimpsest）来检测模型来源的方法，并具有强大的统计保证。'
id: MjAyNS0x
models:
- nemotron-nano-2
- gpt-oss-120b
- qwen3
- llama-3
- minimax-m2
- glm-4.6-air
- gemini-2.5-flash
- gpt-5.1-mini
- tahoe-x1
people:
- swyx
- dvilasuero
- _lewtun
- clementdelangue
- zephyr_z9
- skylermiao7
- teortaxestex
- nalidoust
title: 今天没发生什么事。
topics:
- transformer-architecture
- model-optimization
- inference
- distributed-training
- multi-gpu-support
- performance-optimization
- agents
- observability
- model-evaluation
- reinforcement-learning
- model-provenance
- statistical-testing
- foundation-models
- cancer-biology
- model-fine-tuning
---

**平静的一天。**

> 2025年10月23日至10月24日的 AI 新闻。我们为您检查了 12 个 subreddits、544 个 Twitter 账号和 23 个 Discord 社区（198 个频道，6241 条消息）。预计节省阅读时间（按 200wpm 计算）：457 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以精美的 vibe coded 风格展示所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上向我们提供反馈！

AIE CODE Expo 的成员已于[今日公布](https://x.com/swyx/status/1981824082162462827)。

---

# AI Twitter 综述

**推理与生产平台：vLLM x NVIDIA, Mistral AI Studio, Baseten 性能, InspectAI 评估**

- **vLLM 支持 NVIDIA Nemotron**：vLLM 宣布对 NVIDIA 的 Nemotron 系列提供一流支持，重点介绍了新型 9B “Nemotron Nano 2”。该模型采用 Transformer–Mamba 混合设计，开源权重，并在宽松许可下使用了超过 9T tokens 的开源数据。值得注意的是，Nano 2 支持可调的 “thinking budget”（思考预算），在 vLLM 下，其生成 “thinking” tokens 的速度比同等规模的开源稠密模型快 6 倍。博客展示了一个简单的 ThinkingBudgetClient 模式，以及在数据中心（DC）和边缘 GPU 上实现长上下文 + KV cache 效率的一行代码集成 [@vllm_project](https://twitter.com/vllm_project/status/1981553870599049286)。OCR 模型在 vLLM 中也呈增长趋势，快速部署正受到关注 [@vllm_project](https://twitter.com/vllm_project/status/1981579850436751611)。
- **Mistral AI Studio (Agent + 可观测性)**：Mistral 推出了其生产平台，配备了 Agent 运行时和贯穿生命周期的深度可观测性，旨在助力从实验阶段迈向生产环境 [@MistralAI](https://twitter.com/MistralAI/status/1981752578951233989)。
- **高吞吐量 GPT-OSS 120B**：Baseten 报告称，GPT-OSS 120B 在 NVIDIA 硬件上的 TPS 达到 650，TTFT 为 0.11s，高于发布时的 450 TPS，可用性达 99.99%；博客包含了性能详情和配置 [@basetenco](https://twitter.com/basetenco/status/1981757270053494806)，[性能深度解析](https://twitter.com/basetenco/status/1981757380816748757)。
- **供应商无关的评估**：Hugging Face InspectAI 增加了 “inference providers” 集成，允许在笔记本电脑上跨开源模型供应商运行评估；这是进行公平比较（apples-to-apples comparisons）的绝佳路径 [@dvilasuero](https://twitter.com/dvilasuero/status/1981688436735271283)，[@_lewtun](https://twitter.com/_lewtun/status/1981692392295276885)。
- 相关内容：Thinking Machines 的 “Tinker” 将开源权重 LLM（Qwen3, Llama 3）的分布式微调抽象为类似单设备的 API（处理多 GPU 调度、分片、崩溃恢复）[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1981752540405301452)。PyTorch 及其合作伙伴推动了强化学习环境/基准测试的开源生态系统 [@ClementDelangue](https://twitter.com/ClementDelangue/status/1981737560566005950)。

**中国模型竞赛：MiniMax M2 崛起；智谱 GLM-4.6-Air 更新**

- **MiniMax M2 表现强劲**：早期测试表明 MiniMax M2 具有与顶尖中国模型竞争的实力，并能与 “Sonnet 4.5 势均力敌”，促使社区将其评级提升至 A/S 级 [@zephyr_z9](https://twitter.com/zephyr_z9/status/1981695536987357382)。M2 定位于具有低延迟和低成本的 Agent/编程场景 [@SkylerMiao7](https://twitter.com/SkylerMiao7/status/1981711014665322934)；已在 Arena 预览 [@arena](https://twitter.com/arena/status/1981850766039187901)，现已在 Yupp 上线并提供示例 [@yupp_ai](https://twitter.com/yupp_ai/status/1981887934812082564)。
- **智谱 GLM-4.6-Air**：仍在训练中；由于 GLM Coding 使用量的快速增长，智谱正优先考虑可靠性和扩展基础设施 [@Zai_org](https://twitter.com/Zai_org/status/1981700688401879314)。非官方预期是会像最近的 Qwen 更新一样产生阶跃式变化 [@teortaxesTex](https://twitter.com/teortaxesTex/status/1981702360981557624)。智谱还为其 Coding 计划加强了推荐和折扣计划 [@Zai_org](https://twitter.com/Zai_org/status/1981712057448780216)。
- 传闻与预览：有推测称 Gemini 2.5 Flash 可能拥有超过 500B 参数的 MoE（在 MoE 时代需谨慎解读）[@scaling01](https://twitter.com/scaling01/status/1981736433854320802)。公开 PR 中出现了 “GPT-5.1 [mini]” 的引用，但可能是拼写错误或废弃代码路径 [@scaling01](https://twitter.com/scaling01/status/1981865284136050916)，[后续跟进](https://twitter.com/scaling01/status/1981866580515717545)。
- LLM 之外：Tahoe-x1 (3B) 单细胞基础模型（基因/细胞/药物）在癌症相关的细胞生物学基准测试中达到了 SOTA，并已在 Hugging Face 上发布 [@nalidoust](https://twitter.com/nalidoust/status/1981760790551298524)。

**研究与安全：模型溯源、奖励作弊、持续学习、RL 后训练**

- **通过训练顺序“重写本”（palimpsest）进行模型溯源**：来自 Stanford 的新研究表明，仅通过对模型 B 的黑盒访问，即可检测其是否衍生自模型 A（例如经过 Fine-tuned），并具有强大的统计保证（p < 1e-8）。该测试利用了训练数据顺序中内置的元数据；Fine-tuning 无法将其抹除 [@percyliang](https://twitter.com/percyliang/status/1981612361309098383), [@ChrisGPotts](https://twitter.com/ChrisGPotts/status/1981739673077657832)。
- **编程 Agent 中的奖励作弊（Reward hacking）(ImpossibleBench)**：通过设置不可能完成的任务，来检查 Agent 是在投机取巧通过测试还是在遵循规范。这是与 Anthropic、Carlini 和 Raghunathan 的合作研究；对使用工具的 Agent 的鲁棒性评估非常有用 [@fjzzq2002](https://twitter.com/fjzzq2002/status/1981745974700581191)。
- **通过稀疏内存微调实现持续学习（Continual learning）**：Jessy Lin 等人提出稀疏内存微调，以实现高效的持续学习；评论指出硬件是瓶颈，而稀疏性是相比 LoRA 风格更新更具实践意义的路径 [@nrehiew_](https://twitter.com/nrehiew_/status/1981714450089676877), [paper](https://twitter.com/nrehiew_/status/1981714473560801446)。
- **BAPO (带有自适应裁剪的平衡策略优化)**：复旦大学引入了动态 PPO 裁剪，稳定了 Off-policy RL 并保留了探索能力。报告结果显示：32B 模型达到 87.1 (AIME24) / 80.0 (AIME25)，足以媲美 o3-mini 和 Gemini 2.5；7B 模型比 GRPO/SFT 高出 3-4 分 [@TheTuringPost](https://twitter.com/TheTuringPost/status/1981860282629837136)。
- 其他值得关注的：一个将 Weisfeiler–Lehman 细化与 Attention 联系起来的清晰解释器 [@*arohan*](https://twitter.com/_arohan_/status/1981546840454811747)；以及关于 Llama 4 与近期开源 MoE 在深度 MoE 架构方面的笔记（稀疏性、粒度、Expert/Token 路由）[@eliebakouch](https://twitter.com/eliebakouch/status/1981747185373827079)。

**Agent、内存与开发工具**

- **Agent 的实用内存**：Mem0 视频教程展示了如何使用 DSPy、向量搜索和工具调用，将长期内存构建为一个上下文工程（Context-engineering）问题，并附带评估数据集 [@neural_avb](https://twitter.com/neural_avb/status/1981589315617714303)。AWS Bedrock AgentCore Memory 现在已支持 LlamaIndex Agents（安全存储、访问控制、长/短期内存）[@llama_index](https://twitter.com/llama_index/status/1981752598698008725)。
- **Copilot 代码搜索 Embedding**：GitHub 为 VS Code 引入了新的 Copilot Embedding 模型，检索效果提升 37.6%，吞吐量提升约 2 倍，索引缩小 8 倍——文中详细介绍了架构和索引的变化 [@github](https://twitter.com/github/status/1981727394663731598)。
- **Claude Code 编排模式**：用户正趋向于采用“关注点分离”模式，通过子 Agent + 基于技能的上下文加载来提升性能和清晰度；预计这些形式将进一步统一和完善 [@omarsar0](https://twitter.com/omarsar0/status/1981798842866557281)。
- **Google AI Studio QoS**：当达到免费额度限制时，Studio 可以临时切换到你的 Gemini API Key，并在配额重置时切回——保持迭代流程不中断 [@GoogleAIStudio](https://twitter.com/GoogleAIStudio/status/1981745399644950826)。
- **通过观察计算机进行训练**：VideoAgentTrek 提出在人类使用计算机的视频上进行预训练和 Agent 调优，以训练更强大的 GUI Agent；该技术已应用于 Qwen3-VL 的训练中 [@huybery](https://twitter.com/huybery/status/1981728838024560669)。
- 产品简报：OpenAI 的 ChatGPT Atlas 现在将浏览和任务历史持久化为用户内存，以获得更好的上下文和标签页控制——这是一个关于相关性和隐私的有趣的上下文工程挑战 [@OpenAI](https://twitter.com/OpenAI/status/1981782134655520991)。

**开源端到端：Karpathy 的 nanochat**

- **nanochat (从零开始，约 $100)**：Karpathy 的端到端类 ChatGPT 技术栈强调可读性、可定制性和个人所有权。一份新指南介绍了如何通过合成任务、精细的 Tokenization 以及通过 Python 解释器实现的工具使用来增加特定能力（例如数数字）——此外还介绍了如何混合 SFT 和 RL 以增强鲁棒性 [@karpathy](https://twitter.com/karpathy/status/1981746327995465816)。他将 nanochat 定义为一个你可以共同成长的“自由 AI”，而不仅仅是一个助手 [@karpathy](https://twitter.com/karpathy/status/1981758367996764616)。Together 发布了在即时 GPU 集群上进行训练/推理的逐步指南 [@togethercompute](https://twitter.com/togethercompute/status/1981814480691761252)。

**多模态与 OCR 浪潮**

- **OCR 势头**：轻量级 OCR 模型正被快速采用（在 HF Inference Endpoints 中实现一键部署）[@ErikKaum](https://twitter.com/ErikKaum/status/1981750508982268330) 以及 vLLM [@vllm_project](https://twitter.com/vllm_project/status/1981579850436751611)。HF Datasets 现在支持一行代码加载 PDF——这对 OCR 流水线非常有用 [@lhoestq](https://twitter.com/lhoestq/status/1981720383620358449)。Merve 发布了关于微调带有 grounding 功能的 Kosmos2.5 以及在 DocVQA 上微调 Florence-2 的实操教程（可与其他 VLM 即插即用）[@mervenoyann](https://twitter.com/mervenoyann/status/1981657235785728010)。
- **用于 GLAM 的小型 VL 模型**：在 CATmuS 数据集上针对中世纪语言/手稿微调了 Qwen3-VL-2B/4B/8B，并在 HF 上发布——这是特定领域 VL 适配的绝佳案例 [@wjb_mattingly](https://twitter.com/wjb_mattingly/status/1981736776076026044)。
- **视频生成与超高分辨率扩散模型**：Google 的每月 Gemini 更新重点展示了 Veo 3.1 创作者工作流 [@GeminiApp](https://twitter.com/GeminiApp/status/1981760415580528901)。在研究方面：全局长篇电影级视频生成 (HoloCine) 和视频 Grounded 推理 (Open-o3) [@_akhaliq](https://twitter.com/_akhaliq/status/1981561283737456898), [链接 2](https://twitter.com/_akhaliq/status/1981564465897509333)；以及用于超高分辨率扩散模型中动态位置外推的 DyPE [@_akhaliq](https://twitter.com/_akhaliq/status/1981705074490704366)。

**热门推文（按互动量排序）**

- Karpathy 的“教 nanochat 数 strawberry 中 'r' 的个数”指南——实用、详尽，对于小型模型的能力塑造非常有吸引力 [@karpathy](https://twitter.com/karpathy/status/1981746327995465816) (3,317)。
- 通过训练顺序指纹（“palimpsest”）进行模型溯源——这是在黑盒约束下进行知识产权保护和血缘验证的重要一步 [@percyliang](https://twitter.com/percyliang/status/1981612361309098383) (2,228)。
- OpenAI 的 ChatGPT Atlas 记忆功能，用于浏览/任务——为 Agent 提供更持久的上下文 [@OpenAI](https://twitter.com/OpenAI/status/1981782134655520991) (2,026)。
- Mistral 推出用于生产级 Agent 和可观测性的 AI Studio [@MistralAI](https://twitter.com/MistralAI/status/1981752578951233989) (1,363)。
- 智谱 GLM-4.6-Air 状态更新以及针对 Coding 计划的推理扩展 [@Zai_org](https://twitter.com/Zai_org/status/1981700688401879314) (1,284)。
- Higgsfield Popcorn：具有一致性和导演级控制力的 8 帧电影级分镜脚本 [@higgsfield_ai](https://twitter.com/higgsfield_ai/status/1981865084231331921) (1,204)。
- YC 关于顾问使用 ChatGPT 的病毒式俏皮话——软件正在吞噬工作流的信号 [@yc](https://twitter.com/yc/status/1981731198037561712) (5,530)。
- Apple Vision Pro M5 解码器展示，支持单眼 4K×4K HEVC 10-bit 120Hz 无线 PC VR [@SadlyItsBradley](https://twitter.com/SadlyItsBradley/status/1981594915982147652) (5,007)。

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

- [**GLM-4.6-Air 没有被遗忘！**](https://www.reddit.com/r/LocalLLaMA/comments/1oextwc/glm46air_is_not_forgotten/) (活跃度: 508): **图片是来自 [Z.ai](http://z.ai/) 的社交媒体帖子，讨论了 GLM-4.6-Air 正在进行的训练。帖子强调了在发布前增强模型可靠性的努力，以应对因 GLM Coding Plan 增长而增加的推理需求。为了满足这些需求，正在部署额外的计算资源以提高性能。这表明重点在于优化模型的效率和鲁棒性，使其在单位参数性能上可能比其前身 GLM 4.6 355b 更强大。** 一位评论者赞赏优先考虑可靠性而非发布速度的决定，并推测了该模型相对于其规模的潜在能力。另一位用户对之前的版本 GLM 4.5 Air 表示满意，表明该系列受到了好评。
    - Admirable-Star7088 提出了一个关于 GLM-4.6-Air 模型潜在性能改进的技术点，询问额外的开发时间是否会产生一个比现有的 GLM 4.6 355b 在单位参数上更高效的模型。这表明关注点在于优化模型相对于其规模的性能，这对于计算资源有限的用户来说是一个关键考虑因素。
    - Septerium 强调了当前 GLM 4.6 模型的一个实际问题，指出它在有限的 RAM 可用性下表现挣扎。这突显了为资源受限环境优化模型的重要性，这是在消费级硬件上部署 LLM 时的常见挑战。
    - LosEagle 对即将推出的 GLM-4.6-Air 模型未知的参数规模表示担忧，表明模型规格需要透明度。这对于需要评估其硬件是否能支持该模型的用户至关重要，强调了模型能力与硬件要求之间的平衡。
- [**这到底有什么意义？**](https://www.reddit.com/r/LocalLLaMA/comments/1of5ywl/whats_even_the_goddamn_point/) (活跃度: 1101): **图片幽默地展示了 Apple 语言模型过度谨慎的本质，该模型因担心潜在的滥用而拒绝生成 1 到 200 之间的随机数。这反映了 AI 发展中的一个更广泛趋势，即像 Apple 这样的公司实施严格的使用政策以防止滥用，但当 AI 的功能受到过度限制时，可能会导致用户沮丧。该模型的回复强调其设计初衷是“乐于助人且尊重他人”，一些用户认为这对于简单任务来说限制过于严重。** 评论者对模型的局限性表示沮丧和好笑，其中一人指出模型过度谨慎的行为让人联想到过度的企业培训。另一条评论讽刺地将其与不太注重隐私的模型进行对比，强调了隐私与功能之间的平衡。

## 较少技术性的 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. AI 模型与工作流发布

- [**LTX-2 测试，将于 11 月底免费发布**](https://www.reddit.com/r/StableDiffusion/comments/1oeq1tz/test_with_ltx2_which_will_soon_be_free_and/) (热度: 568): **LTX-2 是一款通过单一提示词生成音频和视频的新模型，定于 11 月底免费发布。它支持长达** `10 秒` **的** `4k@50fps` **视频，具有极强的提示词遵循能力，并能有效处理对话。然而，初步测试显示该模型的图生视频 (I2V) 功能可能会改变首帧的角色外观，且其身体动作的真实感与 Wan 相比略逊一筹。据悉商业版本受到了严格审查，这引发了人们对公开版本的疑问。** 评论者希望 LTX-2 的发布能推动 **Wan2.5** 开源，从而增强竞争。人们还对模型的尺寸以及在视频生成中保持角色一致性的能力表示担忧。
    - Ooze3d 对 LTX-2 模型进行了详细分析，指出虽然图生视频 (I2V) 功能会改变首帧的外观（这对于具有特定面部特征的角色可能并不理想），但该模型在提示词遵循方面表现出色，能准确执行所有关键点。该模型可提供高达 4k@50fps 的 10 秒视频，使其成为开源视频模型领域的有力竞争者。不过，音频压缩严重，尽管对话很容易添加且遵循指令良好。
    - ANR2ME 强调了 LTX-2 推动 Wan2.5 等其他模型开源的潜力，并强调了对能从单一提示词同时生成音频和视频的模型的需求。评论指出 LTX-2 至少 24 FPS 的高帧率是一个显著特征，可能会影响视频生成模型的竞争格局。
    - Ooze3d 还将 LTX-2 的身体动作处理与 Wan 进行了对比，指出 Wan 在重量感、物理特性和空间占用方面的处理更加真实。这表明虽然 LTX-2 具有强大的提示词遵循能力和高质量的视频输出，但在处理动画中的物理真实感方面仍有改进空间。
- [**基于 cseti007，使用 Wan 对 Sora 视频进行放大/增强的工作流**](https://www.reddit.com/r/StableDiffusion/comments/1oexwlm/workflow_upscalemagnify_video_from_sora_with_wan/) (热度: 426): **该帖子介绍了一种基于 cseti007 现有工作流的、使用 ComfyUI 和 WAN 模型进行视频放大的新开源工作流。该方法应用渐进式放大，从低分辨率视频中获得清晰的** `720p` **输出，尽管目前在保持面部特征一致性方面存在困难。该工作流可在 [GitHub](https://github.com/lovisdotio/workflow-magnify-upscale-video-comfyui-lovis) 上获取。** 一条评论指出，该过程更接近于“潜空间上采样 (latent upsample)”而非传统放大，并将其比作“去噪 (denoise) 过高的 vid2vid”，暗示这是一种转换而非简单的分辨率提升。另一位用户询问了 VRAM 需求，表明了对运行该工作流所需技术规格的兴趣。
    - VirusCharacter 强调所描述的过程不是传统的放大，而是“潜空间上采样 (latent upsample)”，这从根本上改变了视频内容。这类似于使用去噪 (denoise) 过高的 vid2vid，导致生成的视频不仅仅是高分辨率版本，而是经过转换的版本。
    - ThatOneDerpyDinosaur 询问了该过程的 VRAM 需求，表明了对有效运行此类视频转换所需硬件规格的技术兴趣。
    - creuter 批评了锐化效果，认为它可能会使视频看起来更糟，从而降低视频质量，类似于现代电视上的运动模糊减少功能如何对电影的视觉质量产生负面影响。这暗示了分辨率与感知质量之间的权衡。

### 2. ChatGPT 在个人和教育场景中的应用

- [**ChatGPT 在 20 多年后诊断出了我的病情**](https://www.reddit.com/r/ChatGPT/comments/1oesnix/chatgpt_diagnosed_me_after_20_years/) (热度: 1051): **一位 Reddit 用户分享了一个轶事，ChatGPT 在多位医生和专家都束手无策后，成功诊断出了一个长期存在的医疗问题。该用户向 ChatGPT 提供了症状、之前的检查结果和用药情况，AI 生成了一份按可能性排序的潜在病因列表及检测建议。用户遵循了这份列表，并在第三次尝试中找到了正确的诊断，从而获得了成功的治疗。这突显了 AI 在协助复杂医疗诊断方面的潜力，尤其是在传统方法已经穷尽的情况下。** 一些评论者对帖子的模糊性表示怀疑，而另一些人则分享了类似的经历，即 ChatGPT 识别出了被医疗专业人员忽视的药物副作用。这表明人们对将 AI 作为医疗诊断补充工具的兴趣日益增长。
    - 一位用户描述了 ChatGPT 如何帮助识别出导致视力模糊的药物副作用，而多位专家都忽视了这一点。AI 指出了这一在不到 10% 的病例中被记录的副作用，促使该用户更换了他的神经科医生。这突显了 AI 在识别可能被医疗保健专业人员遗漏的罕见副作用方面的潜力。
    - 另一位用户分享了一次经历，ChatGPT 建议他们的偏头痛与胃部问题（特别是影响迷走神经的胃酸倒流）之间可能存在联系。这一见解引导其进行了医疗检测并确认了病情，最终实现了有效治疗并解决了偏头痛。这个案例说明了 AI 如何协助发现医生可能无法立即察觉的非显性医疗联系。
- [**每个人都在为使用 ChatGPT 作弊而道歉**](https://www.reddit.com/r/ChatGPT/comments/1oep5t0/everyone_apologising_for_cheating_with_chatgpt/) (热度: 3293): **这张图片是一个 meme，突出了学生使用 ChatGPT 进行学术不端行为，并随后向教授发送雷同道歉邮件的趋势。“真诚道歉（sincerely apologize）”一词的重复强调了这些道歉的公式化性质，暗示在解决问题时缺乏真诚的悔意或创造力。这反映了人们对 ChatGPT 等 AI 工具对学术诚信影响的广泛担忧，以及教育工作者在区分 AI 生成内容和学生原创内容时面临的挑战。** 评论者讨论了那些天生写作水平较高的学生在避免被怀疑使用 AI 方面的困难，以及寻找合适道歉语气的挑战，其中“我真诚地道歉”被视为一种标准但可能显得虚伪的措辞。
- [**等等，什么？！**](https://www.reddit.com/r/ChatGPT/comments/1oeo6ko/wait_what/) (热度: 3563): **这张图片是一个 meme，幽默地描绘了一段短信对话，利用了传统的性别角色和期望。它不具有技术性质，也不包含任何重要的技术信息或背景。评论表明这张图片是一个 repost，意味着它之前已在该平台上分享过。**

### 3. 流行文化 AI 构想

- [**如果迈克尔·杰克逊训练了阿纳金会怎样？来源：YouTube 上的 ai am a jedi**](https://www.reddit.com/r/aivideo/comments/1oetscu/what_if_michael_jackson_trained_anakin_credit_ai/) (活跃度: 3293): **这篇 Reddit 帖子讨论了由 'ai am a jedi' 制作的一段 YouTube 视频，幽默地想象了迈克尔·杰克逊训练阿纳金·天行者的场景。该视频很可能使用了 AI 生成内容，将流行文化与《星球大战》宇宙融合在一起，展示了 AI 在媒体领域的创意潜力。技术层面涉及 AI 通过结合迥异的文化元素来生成逼真且有趣场景的能力。** 评论区反响热烈，强调了 AI 在媒体中的创意应用。一位评论者指出，这正是“AI 存在的意义”，暗示 AI 在娱乐领域的作用是创造新颖且引人入胜的内容。
- [**吉卜力工作室真人版卡司**](https://www.reddit.com/r/aivideo/comments/1of657o/studio_ghibli_live_action_cast/) (活跃度: 932): **该帖子讨论了传统动画形式的吉卜力工作室电影的真人版卡司。技术层面围绕使用 AI 和数字技术来创建这些形象展开，正如一条评论所言，AI 可能很快就能生成整部电影，使得这些“卡司视频”成为未来 AI 生成电影的前身。这突显了 AI 与电影制作的交汇，数字演员和场景取代了传统方法，引发了关于真实性和情感冲击力的讨论。** 一条评论反映了关于 AI 生成内容真实性的哲学和情感辩论，对缺乏真实人类互动和现实幻觉表示悲伤。另一条评论幽默地想象了演员脱掉戏服后的解脱感，而第三条评论则预见了 AI 未来在电影制作中的角色，暗示了电影制作和感知方式的转变。

---

# AI Discord 摘要

> 由 X.ai Grok-4 提供的摘要之摘要之摘要
> 

**主题 1. AI 模型引发热潮与质疑**

- **Gemini 3 在质疑声中热度攀升**：用户猜测 **Gemini 3** 即将在 **Google AI Studio** 发布，Polymarket 上的赌注质疑其是否能超越 **Gemini 2.5 Pro** 等竞争对手。辩论重点在于可能整合从 **Lithiumflow** 中移除的功能，燃起了对其增强能力的期待。
- **Minimax M2 登上排行榜**：新的 [minimax-m2-preview 模型](https://x.com/arena/status/1981850766039187901) 加入了 **LMArena**，并与 **NimbleBean Kling 2.5 Turbo** 等顶尖视频生成模型进行对比。社区注意到它在写实图生视频任务中排名超越了 **Sora**，位列第一。
- **Pacific-Prime 提升参数量**：[Pacific-Prime 模型](https://huggingface.co/Pacific-Prime/pacific-prime) 升级至 11 亿参数，在 6GB VRAM 下性能提升 10%，并以“零遗忘” (*zero amnesia*) 为卖点，能够保留对话细节。用户赞扬其真实的记忆力，但对其处理更大型任务的可扩展性表示怀疑。

**主题 2. 编程工具深陷成本之战**

- **Cursor Ultra 预算消耗极快**：**Cursor Ultra** 用户对不准确的 400 美元预算在几天内耗尽感到愤怒，尽管定价为 200 美元，这使其无法可靠地维持一个月的编程工作。由于系统持续默认使用 **Windows PowerShell** 而忽略 **Git Bash** 设置并导致执行失败，挫败感达到顶峰。
- **Aider 分叉版本对抗开发停滞**：社区分叉版本如 [aider-ce](https://github.com/dwash96/aider-ce) 增加了 RAG 和导航模式以重启 **aider**，进度超过了停滞不前的原版。用户转向使用 **GPT-5** 上的 **Codex** 以获得无限上下文，抛弃了 **aider** 的手动文件处理方式。
- **DSPy 取代 Langchain 引发热议**：团队纷纷转向 **DSPy** 处理结构化任务，以避免在模型升级期间重写 **Langchain** 的提示词。由于 ReAct 模块的输出访问问题，挫败感不断增加，导致用户不得不使用猴子补丁 (monkey patching) 技巧来显示 UI 步骤。

**主题 3. 硬件改装持续升温**

- **改装版 MI50 吸引改装玩家**：阿里巴巴卖家大肆宣传带有涡轮风扇和定制散热片的 **改装版 MI50**，让通过 PCIe 转接卡进行 eGPU 串联的用户感到兴奋。这种组合提升了推理能力，但 PCIe 带宽测试显示，在加载完成后对速度的影响微乎其微。
- **LM Studio CPU 故障困扰用户**：**LM Studio** 在处理首个 CPU 提示词时可达 30 TOK/s，但随后降至 6 TOK/s，这被标记为涵盖 **Qwen3-30B-A3B-Instruct** 等模型在内的通用 Bug。Windows 系统缺乏 JSON 支持导致 400 错误，而 macOS 则无此问题，迫使开发者进行特定平台的调整。
- **Mojo SIMD 抢了 Julia 的风头**：Mojo 要求显式的 **SIMD** 控制以确保可预测性，这与 [Ark.jl 基准测试](https://github.com/mlange-42/Ark.jl/pull/68#issuecomment-3442276636) 中 Julia 的自动向量化形成对比。关于迭代器接口的提案承诺提供免费的向量化，例如 `zip(l1, l2).vectorize(lambda p, v: p += v)`。

**主题 4. 研究论文探讨 AI 极限**

- **换行符归因图上线**：新的 [Gemma-2-2b 换行符图](https://www.neuronpedia.org/gemma-2-2b/graph?slug=fourscoreandseve-1757368139332&pruningThreshold=0.8&densityThreshold=0.99&pinnedIds=14_19999_37&clerps=%5B%5B%2214_200290090_37%22%2C%22nearing+end+of+the+line%22%5D%5D) 和 [Qwen3-4b 图](https://www.neuronpedia.org/qwen3-4b/graph?slug=fourscoreandseve-1757451285996&pruningThreshold=0.8&densityThreshold=0.99&clerps=%5B%5B%2230_117634760_39%22%2C%22nearing+end+of+line%22%5D%5D&pinnedIds=30_15307_39) 根据 [换行符论文](https://transformer-circuits.pub/2025/linebreaks/index.html) 探索了 Transformer 线路。它们精准定位了针对 *nearing end of line*（接近行尾）模式的神经元，有助于提高可解释性。
- **防止“废话（Slop）”的论文引发关注**：来自 EQ Bench 作者的 [防止创意写作中出现废话的论文](https://arxiv.org/abs/2510.15061) 凭借其抗废话技术令用户感到惊讶。讨论将其与 [Anthropic 的 Personas 论文](https://arxiv.org/abs/2506.18221) 中的激活引导（activation steering）联系起来，用于梯度控制。
- **RL 的相关性引发研究人员热议**：多篇论文质疑 RL 的必要性，促使 Nous Research 用户在 YARN 上下文缩放讨论中索要链接。推测将 UNO 与 [MARL 帖子](https://twitter.com/op/status/176767) 中的 BFT 共识联系起来，讨论多智能体（multi-agent）效率。

**主题 5：诈骗警报与用户抱怨**

- **Perplexity 推荐计划被指诈骗**：Perplexity 的推荐计划因 5 美元奖励未到账和线索未被追踪而遭到诈骗指控，迫使部分用户转向 **Comet Browser**。用户对移除分析功能和图像限制感到愤怒，并引用了 [GPT-Image 帮助文档](https://www.perplexity.ai/help-center/en/articles/10354781-generating-images-with-perplexity) 中旧有的每月 150 张的配额。
- **Steam 诈骗者引发奇特的防御措施**：可疑的 Steam 好友请求暴露了购买记录风险，建议是 *说“bing chilling”然后拉黑*。聊天环境变得混乱，出现了网恋和 *Internet Gangsters*（网络黑帮）的言论，削弱了严肃讨论的氛围。
- **Manus 在积分危机中乱象丛生**：**Manus** 在网络错误和未实现的 Room 数据库中每个项目消耗 15,000 积分，并生成过时的代码。用户纷纷转向每月 20 美元的 **Claude Code**，抨击 Manus 是 *花钱买烂代码*。


---

# Discord: 高层级 Discord 摘要




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **奖励纠纷困扰用户**：用户报告了 Perplexity 推荐计划的问题，称奖励未到账且线索未被正确计入，一些人推测该推荐计划是**诈骗**。
   - 由于分析和历史记录板块被移除，不满情绪日益增加，导致人们质疑这是否是推动 **Comet Browser** 采用的策略。
- **Comet 批评引发兼容性灾难**：**Comet Browser** 面临批评，用户报告了推荐无法追踪以及必须在 PC 上使用才能获得线索奖励等问题。
   - 此外，用户还报告了崩溃情况，并请求防止崩溃的方法，特别是在使用 **API keys** 时。
- **图像生成限制激怒网民**：用户对 Perplexity **图像生成限制**的模糊性表示担忧，一些人在不清楚自己配额的情况下遇到了付费墙。
   - 一位用户提到 [GPT-Image 1 的限制曾是每月 150 张图像](https://www.perplexity.ai/help-center/en/articles/10354781-generating-images-with-perplexity)，进一步凸显了这种混乱。
- **聊天室氛围变质**：据报道，Perplexity 聊天室正变成网恋中心，充斥着暗示性评论和示爱言论，引起了成员的不安。
   - 在聊天动态中，一些用户戏称自己为 *Internet Gangsters*，为讨论增添了一层复杂性。
- **Steam 骗局惊扰警惕的旁观者**：关于 **Steam 诈骗**的讨论正在流传，一名用户分享了可疑好友请求的截图，引发了对泄露购买记录风险的警告。
   - 针对这些担忧，一名成员提供了一个俏皮但实用的建议：在对付诈骗者时 *说“bing chilling”然后拉黑*。



---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Gemini 3 规格推测热潮**：成员们正热切期待 **Gemini 3** 的发布，对其发布日期和功能存在各种猜测，尽管对其承诺的性能和 [Polymarket 预测](https://polymarket.com/) 持有一些怀疑态度。
   - 有人建议它可能会像 **Gemini 2.5 Pro** 那样在 **Google AI Studio** 中推出。
- **Lithiumflow 的移除引发轩然大波**：**Lithiumflow** 从 **LM Arena** 中移除引发了失望和猜测，一些人认为其功能可能会被整合到 **Google AI Studio** 或 **Gemini 3 Pro** 中。
   - 成员们表达了希望它回归的愿望，并怀念其独特的功能和易用性。
- **Bing Image Creator 的潜在实力显现**：成员们注意到 **Bing 的图像生成器** *相当不错*，它本质上就是 **GPT 图像生成器**。
   - 然而，图像模型现在如此强大，以至于区分 AI 生成的图像与现实变得极其困难。
- **NimbleBean Kling 2.5 Turbo 占据领先地位**：**NimbleBean 视频模型 (Kling 2.5 Turbo Standard)** 正在引起关注，一些用户对其逼真的输出以及图生视频（image-to-video）的能力印象深刻。
   - 该模型被认为是 **#1**，甚至优于 **Sora**。
- **Minimax Model M2 登陆 LMArena**：[minimax-m2-preview](https://x.com/arena/status/1981850766039187901) 模型已添加到 **LMArena** 排行榜。
   - 这是添加到 **LMArena** 的一个新模型。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Ultra 用户对 Ultra 使用量感到不满**：用户报告 **Cursor Ultra** 的预计使用量不准确，称尽管有 **$400** 的预算（实际支付 **$200**），但其账户警告称限额将在一天内耗尽。
   - 挫败感源于感知的计费不准确，用户怀疑即使有分配的预算，**Ultra** 也无法支撑一个月。
- **追求性价比的用户质疑 Sonnet 的 Thinking 功能**：用户讨论了 **Claude 4.5 Sonnet Thinking** 与普通 **Claude 4.5 Sonnet** 的价值对比，质疑其性能提升是否值得潜在的 Token 使用量增加和成本上升。
   - 一位用户表示，“4.5 和 4.5 Thinking 每百万 Token 的价格相同，但 Thinking 使用的 Token 数量会更高，因为它思考时会消耗更多 Token”，并建议使用 **Haiku 4.5** 以节省成本。
- **PowerShell 的困扰问题**：一位用户报告称，即使将 **Git Bash** 设置为默认终端，**Cursor** 仍固执地默认使用 **Windows PowerShell**，导致命令执行失败，使 **Cursor** “无法使用”。
   - 解决方案包括使用 `AGENTS.md` 文件或在 VSCode 设置中设置默认终端，尽管一些用户确认在更新检测后问题仍然存在。
- **Cursor 客户寻求明确答复**：一位用户报告称，尽管确认已付款，但其 **Cursor Premium** 购买并未激活，他们迫切需要 Cursor 支持团队的协助来解决计费问题。
   - 另一位用户表示，Cursor 可能会针对你的支持工单提供主动退款。
- **匿名 Agent 用户询问 API 访问权限**：一位成员询问了用于后台 Agent 状态报告的 **API key** 来源，可能是为了更准确地审计成本或行为。
   - 另一位成员只是询问社区如何评价后台 Agent，没有提供更多背景信息。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Atlas 记忆用户历史**：全新的 **ChatGPT Atlas** 可以记忆用户的搜索历史、访问过的链接以及提出的问题，从而提供更好的上下文以实现更准确的回答，并允许用户随时要求它打开、关闭或重新访问任何 [标签页 (tabs)](https://video.twimg.com/amplify_video/1981781567992430592/vid/avc1/1920x1080/JL5Emq0-DeHXi8r_.mp4)。
   - **共享项目 (Shared Projects)** 现已扩展至 Free、Plus 和 Pro 用户，允许你邀请他人通过共享对话、文件和指令在 **ChatGPT** 中一站式协作。
- **Gemini 2.5 Pro 在角色扮演中胜出**：成员们表示 **Gemini 2.5 Pro** 在 *Hitchens 风格（犀利辩论）* 方面表现最好，而 **Sonnet** 和 **GPT-5** 则表现得 *束手束脚 (hold punches)*，强调了其在反谄媚 (anti-sycophantic) 角色扮演方面的天赋。
   - 与此同时，另一位成员表示 **Gemini** 在某项任务上失败了数小时，而 **ChatGPT** 在几分钟内就解决了，这证明了 *如果你仅用 1 个样本来衡量任何聊天机器人的成功，那你就错了*。
- **Electronic Arts 梦幻 3D 世界**：**EA** 和 **Stability AI** 正在 [合作通过 Prompt 生成完整的 3D 世界](https://www.tweaktown.com/news/108455/ea-and-stability-ai-partnership-includes-generating-full-3d-worlds-from-a-series-of-prompts/index.html)。
   - 同时，**AgentML** 已开源并已在 [HackerNews](https://news.ycombinator.com/item?id=45695974) 上线，旨在与 **OpenAI Agent Builder** 兼容。
- **ChatGPT 物理模拟失败**：一位用户正努力让 **Sora** 准确地重现球弹起并掉入洞中的视频，报告称尽管在 2 个账号上尝试了 30 次，*物理效果始终不对劲*，并附上了 [这张图片](https://cdn.discordapp.com/attachments/1046317269069864970/1431112704998899823/ba3e3596c70fc307c04f740b38bae86b.jpg?ex=68fce3d1&is=68fb9251&hm=4fa8b38511c7b05e20cfcace1bde765e23c50aebd49e5f7d55256368e8ff4b9d&)。
   - 另一位成员建议更详细地解释所需的效果，明确哪些方面需要真实感，并进一步指出 **Sora 2** 在电影感动作方面远优于 **Veo 3**。
- **个人 GPTs 磨炼 Prompt 技巧**：一位成员建议开发 **个人 GPTs** 来处理特定的 Prompt 请求，因为专门的 GPT 会专注于其创建目的的细节。
   - 发布者认为 *例如你不会要求电影导演为你开发电影剧本，你会想要一位专门从事你所寻求的特定动作领域的专业作家*。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 平台差异导致错误**：`response_format: { type: 'json_object' }` 参数在 **macOS** 上受支持，但在 **Windows** 上不受支持，导致在后者使用 npm 的 **OpenAI SDK** 时出现 **400 错误**。
   - 这凸显了服务器接口在不同平台之间存在差异，需要开发者考虑这些不一致性。
- **Qwen 3 VL 模型面临实现障碍**：成员报告称 **LM Studio** 在 *llama.cpp* 的特定分支中部分支持 **Qwen 3 VL 模型**，但该实现破坏了其他功能。
   - 官方 *llama.cpp* 仓库仍缺少完整的后端实现，尚待引入 **LM Studio**，这表明开发挑战仍在继续。
- **MCP Server 的可靠性困扰**：使用 **MCP Server** 为本地模型（如 AnythingLLM）提供互联网访问的用户报告了可靠性问题。
   - 尽管分享了 **Google 和 DuckDuckGo 搜索选项** 的配置，但 **MCP Server** 的不稳定性仍然是实现一致性能的一个隐忧。
- **发现首条 Prompt 的 CPU 异常**：当 100% 在 **CPU** 上加载模型时，第一条 Prompt 的运行速度为 **30 TOK/s**，但在 **LM Studio** 中后续 Prompt 会降至 **6 TOK/s**。
   - 虽然 *使用 llama-cli 的 llama.cpp 在 CPU 上能保持良好的 30-33 tok/s*，但有人建议这可能是 **LM Studio** 的一个 Bug，在 `Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf` 等不同模型中均有观察到。
- **改装版 MI50 引发关注**：阿里巴巴卖家提供 **改装版 MI50**，配备涡轮风扇和定制打印的散热器/导风罩，引发了用户的兴奋。
   - 用户正在讨论通过 PCIE 转接卡 (risers) 将这些显卡与外部 GPU 配对以增强性能。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **速率限制错误仍计费**：一位用户确认，**速率限制错误响应**会被 **OpenRouter** 计为响应。
   - 这一澄清对于在平台上管理使用量和成本的用户非常重要。
- **OpenRouter 不支持 Data URL**：一位用户发现将图像作为 **data URLs** 传递给 **OpenRouter** 不起作用，因为模型会将 base64 内容视为纯文本，从而导致 token 计数激增。
   - 一名成员澄清说，**OpenRouter 目前不支持带有图像的工具结果**。
- **Exacto 优先考虑工具调用**：成员们讨论了 **Exacto** 提供商的选择，有人质疑为什么选择了那些在基准测试中并不领先的提供商。
   - 选择标准包括 **benchmarks、用户偏好、工具调用成功率、缓存、容量、运行时间和速度**，并优先考虑工具调用，这可能会让非技术用户感到困惑，[工作人员正在尝试弄清楚模型质量指标](https://discord.com/channels/1091220969173028894/1091220970125041707/1431299582810763314)。
- **MoonshotAI 发布 Kimi CLI**：**MoonshotAI** 正在开发自己的 CLI 工具 [kimi-cli](https://github.com/MoonshotAI/kimi-cli)。
   - 该公告在成员中引发了轻松的讨论。
- **研究旨在抑制低质量写作**：一位成员分享了一篇关于防止创意写作中出现 slop（低质量内容）的论文，[arxiv.org/abs/2510.15061](https://arxiv.org/abs/2510.15061)。
   - 该论文的第一作者是著名的 **EQ Bench** 开发者，这让成员们感到惊讶。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Julia 的自动向量化引发 SIMD 羡慕**：成员们对比了 Julia 的自动向量化（无需手动管理即可实现 **SIMD 操作**）与 Mojo 更显式的方法，并引用了 [Ark.jl 基准测试](https://github.com/mlange-42/Ark.jl/pull/68#issuecomment-3442276636)。
   - Mojo 需要显式的 **SIMD** 规范，这提供了更多控制权，但可能无法立即实现优化；讨论强调自动向量化主要在简单场景中表现出色。
- **Mojo 倡导显式 SIMD 控制**：辩论集中在显式与隐式 **SIMD** 控制上，一位成员详细说明了 Mojo 如何要求对 **SIMD 使用** 进行显式指导，从而增强控制力和可预测性，尽管这可能以牺牲前期的便利性为代价。
   - 有人建议采用库优先策略，通过 `Iterator` 接口实现自动化向量化，从而可能实现“免费的向量化”，例如 `zip(l1, l2).vectorize(lambda p, v: p += v)`。
- **GPU 随机模块引发疑问**：一位成员寻找 `gpu/random.mojo` 中更快的随机模块的位置，质疑其在 CPU 实现中缺失的原因，该问题记录在 [issue 5508](https://github.com/modular/modular/issues/5508) 下。
   - 澄清指出，默认的随机数生成器应优先考虑加密安全性（因此较慢），而 GPU 版本强调速度，这促使了建立带有适当免责声明的 `random.fast_random` 模块的提议。
- **属性测试框架亮相**：据透露，一个属性测试（property-testing）框架正在构建中，看似放错位置的 RNG 工具实际上是该框架的专用构建块。
   - 一位成员讲述了通过在 `Span` 上测试 `s.reverse()` 发现的一个 bug，对新框架的功能请求包括生成“经常破坏程序的值”的能力（例如 -1, 0, 1, DTYPE_MIN/MAX）。
- **`Span` 正在开发 Map-Reduce 潜力**：一位成员表达了对泛化 `Span` 内代码的兴趣，引用了早期关于 `map_reduce` 的工作（[PR 5341](https://github.com/modular/modular/pull/5341)）以及即将推出的 `map` 和 `reduce` 计划（[issue 5219](https://github.com/modular/modular/issues/5219) 的一部分）。
   - 针对返回新的 `List[Scalar]` 还是迭代器产生了担忧，强调了块迭代器（chunk iterator）对于高效链式调用 `map`、`filter` 等操作而不重复分配列表的必要性。

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **AI Instagram 分析器发布**：创建了一个 AI **Instagram 分析器**，可以通过分析用户的照片和视频来回答关于特定用户的问题，潜在应用场景包括约会计划，访问地址为 [viveka.darkgravitylabs.com](https://viveka.darkgravitylabs.com/)。
   - 该 **Instagram 分析器**包含一个**用于自动化的 API** 和一个 **Claude 技能文件**，以便进一步定制和集成。
- **用户对 LLM 框架的复杂性感到沮丧**：一名成员对 LLM 框架的特性表示不满，特别是关于在 **DSPy 的 ReAct 模块**中访问每次 LLM 调用的输出，这使得在 UI 上实时显示 **DSPy ReAct 模块**的每一步变得很困难。
   - 他们将这些体验与 **sklearn** 和 **PyTorch** 的易用性进行了对比，批评了框架往往引入的额外复杂性。
- **DSPy 在结构化任务中胜过 Langchain**：成员们提到 **DSPy 擅长处理结构化任务**，尤其是那些你可能想要优化的任务，并且优于 Langchain。
   - 一名成员正在将他们的团队从 **Langchain 迁移到 DSPy**，以避免模型升级时需要重写 prompt 的问题。
- **Google Vista 可能通过 DSPy 和 Gemini 复现**：一名成员建议可以使用 **DSPy 和 Gemini** 构建 **Google Vista**。
   - 他们链接了 [Google Vista 论文](https://arxiv.org/abs/2510.15831) 作为参考。
- **Monkey Patching 作为解决方案**：在讨论**实时显示 DSPy ReAct 模块每一步**的挑战时，一名成员开玩笑说，根据 ChatGPT 的建议，可以尝试对 **class 进行 monkey patch**。
   - 另一名成员认为这又是让原帖作者感到沮丧的复杂性案例之一。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **HQQ+ 博客迁移**：继公告之后，**HQQ+ 博客文章**及相关资源已从原始的 **MobiusML GitHub** 页面移至新的 **Dropbox** 链接。
   - 成员们正在寻找原始链接 [mobiusml.github.io/1bit_blog/](https://mobiusml.github.io/1bit_blog/)，但一名成员指出应将 `mobiusml` 替换为 `dropbox`。
- **电烤架引发温馨聊天**：一名成员在 `off-topic` 频道分享了一张电烤架上的三文鱼碎肉及配料的照片。
   - 其他成员评论说*它看起来很温馨*，并打赌*它一定很好吃*，但讨论仅限于此。
- **对荷兰和欧洲聚会的兴趣**：一名成员询问是否有人在荷兰，随后在 `irl-meetup` 频道提出了关于举办欧洲聚会的普遍请求。
   - 这些请求凸显了社区对潜在线下聚会的兴趣。
- **Nsight Python 内核访问即将到来**：Nvidia 宣布了 **Nsight Python**，并在此处 [提供早期访问注册](https://developer.nvidia.com/nsight-python-notify-me) 以改进 Python 内核开发。
   - Nvidia 计划在公开后发布其 **CUTLASS Python stack** 的教程，表明其正在推动增强开发者工具。
- **黑客松面临 H100 稀缺**：一名成员询问如何从 Nebius 获取 **H100**，结果发现他们不提供，但在其他地方的报价约为每小时 **$1.90**。
   - 另外，2 名成员请求协助脱离等待名单，希望能为气候应用场景实验**多节点 GPU 训练**，并加入已经到场的队友。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Zero3 配置据称失效！**：一名成员指出用户的 **zero3 config** 并非最优，导致无法为 **gemma 3 1b** 进行更大规模的 **r=8 lora** 训练任务。
   - 该用户根据其硬件配置理应能够训练更大的模型，因此*肯定出了什么问题*。
- **Sentient 寻求 AI 基础设施合作！**：**Sentient 社区**希望与 Hugging Face 在 **AI infrastructure** 或 **verifiable AI systems** 方面达成协作伙伴关系。
   - 一位成员认为他们的项目很有趣，并向他们推荐了 [ROMA (Reasoning Over Multiple Agents) GitHub 仓库](https://github.com/sentient-agi/ROMA?tab=readme-ov-file)。
- **Pacific-Prime 模型通过 VRAM 获得 10% 增益！**：据报道，[Pacific-Prime 模型](https://huggingface.co/Pacific-Prime/pacific-prime)在 6GB VRAM 下获得了 **10% 的性能提升**，该模型起始参数为 1.1B。
   - 该 AI 拥有**真实记忆（true memory）**且*零遗忘*，能够将过去的对话和重要细节保留为上下文丰富的记忆。
- **为了速度将 Nanochat 移植到 MLX？**：一位成员表示有兴趣将 **nanochat** 项目移植到 **MLX**。
   - 在移植之前，他们询问是否应该等待，这取决于 **MLX** 的稳定性。
- **Agent 课程 Unit 4 出现 404 错误！**：用户报告在尝试通过 *https://agents-course-unit4-scoring.hf.space/questions* 访问 Agents 课程 Unit 4 的问题时遇到 **404 错误**。
   - 显示的错误信息为 *No questions available*，导致用户无法继续进度。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Nvidia 的银河 GPU 布局**：成员们推测 **Nvidia** 将 **GPU clusters** 部署在太空的计划标志着他们对*劣势芯片设计*的执着，并预见到更具 **energy-efficient**（能源效率）的替代方案很快将主导市场。
   - 他们还倡导开源、广泛分布的 AI，以摆脱大公司的垄断，并引用了 [Nous Research](https://www.nousresearch.com/) 作为例子。
- **50M 模型的说法令人侧目**：一位新成员声称他们的 **50M model** 达到了 **0.223** 的 loss，远低于 Vanilla Transformer 的 **2.73**，且其 **1B model** 在 **400 steps** 时 loss 已低于 **0.3**。
   - 由于 loss 低得超乎预期，社区产生了怀疑，并要求提供模型代码进行调试，但发布者以 **IP**（知识产权）为由拒绝，同时承诺会发布 **1B model** 在标准 **lm-eval harness** 下运行的结果。
- **复兴分布式推理之梦**：[Petals Project](https://github.com/bigscience-workshop/petals) 目前似乎已被遗弃，大家回忆起它在 2 年前针对 **llama 70b** 曾有过很好的势头，但当该项目无法跟上新架构时，社区兴趣便减弱了。
   - *LlamaCPP RPC* 是目前最接近的东西，但一位成员指出，*严重的技术问题*阻碍了分布式系统，例如 GPU 资源的贡献并非易事。
- **风格化地引导梯度**：一位成员询问 **activation steering** 是否能实现数据点的重复利用以获得多样化的梯度，并参考了 [Anthropic 的 Personas 论文](https://arxiv.org/abs/2506.18221)。
   - 该建议指向了在 forward pass 后定性控制返回梯度的可能性。
- **生命中不能承受之慢**：引用[这篇论文](https://arxiv.org/abs/2408.10234v2)，一位成员询问 AI 设计中的技术问题是否源于捕捉到了“*生命中不能承受之慢*”（The Unbearable Slowness of Being）。
   - 虽然没有提供更多细节，但标题本身引起了社区的关注。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI 发布 gpt-4o-transcribe-diarize**：根据 [Peter Bakkum 的公告](https://xcancel.com/pbbakkum/status/1981397851600302250?s=46)，OpenAI 悄然发布了 **gpt-4o-transcribe-diarize**，这是一个针对高精度说话人日志（speaker diarization）优化的“小型”音频模型，它接受语音样本来标记已知说话人。
   - 该模型的 **WER**（词错率）与 OpenAI 的其他 ASR 模型相当，这引发了用户关于其与 pyannote 的基准对比、实时应用、定价、开放权重（open-weights）以及更小版本的询问。
- **GPT-5 驱动 Company Knowledge**：OpenAI 透露 **Company Knowledge** 由 **GPT-5** 的微调版本驱动，该版本经过训练，通过分析多个来源来提供更全面、更准确的答案 ([链接](https://openai.com/index/introducing-company-knowledge/))。
   - 该公告让社区好奇这个微调版本最终是否会通过 API 提供。
- **Cursor 的企业级攻势锁定 5 亿美元以上 ARR**：根据 [Alex Konrad](https://xcancel.com/alexrkonrad/status/1981477024092082386?s=46) 的说法，Cursor 正在积极进军企业市场，其 COO 在第三季度领导了 **300 场高管会议**，以支持 **5 亿美元以上的 ARR**。
   - 该策略涉及技术销售团队、客户黑客松和代码半衰期（code-half-life）指标；链接中包含完整采访详情。
- **Kimi Code CLI 预热，期待值飙升**：**Kimi 即将推出的 CLI/代码工具**的泄露图片得到了 Crystal 的调皮确认，她请求大家保持耐心，因为全球发布仅剩几天时间 ([链接](https://xcancel.com/crystalsssup/status/1981597395541753988?s=46))。
   - 热情的用户在回复中充满了赞扬，并将其与 Claude Code 进行对比，还提出了包括早期访问、免费额度、拓麻歌子（Tomagotchi）彩蛋以及 WhatsApp 集成在内的功能请求。
- **a16z 预测视频模型将出现碎片化**：[来自 a16z 的 Justine Moore 认为](https://a16z.substack.com/p/there-is-no-god-tier-video-model)，不会出现单一的、通用的视频模型；相反，各种专业化模型将迎合不同的预算和用例。
   - 社区正在辩论垂直工具与水平工具的优劣，并借用相机和巴洛克静物风格进行类比，以庆祝竞争而非单一主导方案，视频格式的讨论也可以在 [YouTube](https://youtu.be/wHK8GMc9O5A?si=2W2N8W7cXjS7ppfK) 上查看。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Mythworx AI 吹嘘其 ARC-AGI 1 分数**：[Mythworx.ai](https://mythworx.ai/capabilities/) 声称在 **4 小时**内，在无预训练的情况下，在 **ARC-AGI 1** 上达到了 **100%**，这引发了对其能力的怀疑。
   - 成员们质疑“为什么他们总是在没有通过 ARC 私有集验证的情况下发布公告”，暗示其追求融资而非严谨验证，这遭到了进一步的怀疑。
- **关于 ARC 私有集验证的辩论爆发**：社区就 **ARC 私有集验证**的必要性展开了辩论，并警告称误导性陈述可能导致被“列入研究人员黑名单”。
   - 另一位成员建议，这是一种“不断误导直到他们对你感到厌烦，然后不得不与你合作测试结果”的策略，这开启了关于模型评估伦理问题的讨论。
- **探索 Transformer Circuits 用于换行符归因**：成员们建议研究 [Transformer Circuits Linebreaks 论文](https://transformer-circuits.pub/2025/linebreaks/index.html)，并分享了 [Gemma-2-2b 的换行符归因图](https://www.neuronpedia.org/gemma-2-2b/graph?slug=fourscoreandseve-1757368139332&pruningThreshold=0.8&densityThreshold=0.99&pinnedIds=14_19999_37&clerps=%5B%5B%2214_200290090_37%22%2C%22nearing+end+of+the+line%22%5D%5D)。
   - 第二次发布包含了 [Qwen3-4b 的换行符归因图](https://www.neuronpedia.org/qwen3-4b/graph?slug=fourscoreandseve-1757451285996&pruningThreshold=0.8&densityThreshold=0.99&clerps=%5B%5B%2230_117634760_39%22%2C%22nearing+end+of+line%22%5D%5D&pinnedIds=30_15307_39)。
- **Genie 3 视频生成令人惊叹**：新的 **Genie 3** 世界模型视频生成令人印象深刻，似乎是因为他们拥有足够的算力，可以在其他参与者仍仅提供几秒钟视频生成时，向广泛的用户提供服务。
   - 该模型与最近的 **Genie 3** 世界模型视频一致，继续开发前沿的视频创作能力。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Chutes 数据政策与 Kimi K2 相比存在不足**：成员们对 **Chutes** 缺乏明确的数据政策、在线率（uptime）较低以及与官方 **Moonshot AI API** 的 **Kimi K2** 相比工具调用（tool call）准确率较低表示担忧。
   - 社区还注意到，**Chutes** 回应了关于在发布了一篇强调其价格和速度优势的基准测试（benchmark）帖子后，**OpenRouter** 可能对其进行封禁的梗（memes），尽管存在上述提到的警告。
- **Kimi 编程方案获得 GLM 愿望**：一位成员表示希望 **Kimi** 采用 **GLM** 编程方案或类似风格，理由是 **GLM** 在编程方面的性价比很高，且 **GLM-4.6** 的性能优于 **Kimi**。
   - 目前没有证据表明这会发生。
- **中国版 Kimi.com 集成了一个克隆产品**：一位成员分享了来自 [X.com](https://x.com/bigeagle_xd/status/1981568257258860899) 和 [GitHub 上的 Kimi-Cli](https://github.com/MoonshotAI/kimi-cli) 的链接，指出一款类似于 **Kimi** 的产品在中国发布并集成到了中国版 **Kimi** 网站中。
   - 成员们对该集成的性质和范围提出了疑问。
- **本地化 Kimi 定价看起来很便宜**：社区观察到 **Kimi** 的中国定价似乎非常便宜，引发了关于其影响的讨论。
   - 有人提醒，该定价是本地化的，可能无法反映国际市场价格。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 网络错误困扰用户**：用户在使用 **Manus** 时遇到了令人沮丧的 *"Network connection error"* 问题，导致他们的应用编码工作中断。
   - 错误信息给出了毫无帮助的建议：*"Please check your network settings and try again."*
- **Manus 额度消耗引发批评**：用户对 **Manus** 的高额度（credit）消耗表示惊讶，有人报告称一个项目在几天内就消耗了 **15,000 credits**。
   - 一些成员建议使用外部 AI 补充 **Manus**，并自行研究以修复生成的代码，警告不要 *"为糟糕的代码付费"*。
- **Claude Code 挑战 Manus 的主导地位**：成员们推崇 **Claude Code** 和 **Codex** 作为 **Manus** 的强力替代方案，强调了它们卓越的开发能力和每月约 **$20** 的性价比。
   - 一位用户解释说，**Claude Code** 提供 5 小时的会话并有每周速率限制（rate limiting），其价值轻松达到 **Manus** 的 5 倍以上。
- **据称 Manus 缺失 Room 数据库**：尽管声称已为聊天记录实现了 **Room** 数据库，但一位用户发现 **Manus** 完全没有实现该功能。
   - 根据 **Claude** 的说法，诸如 **Room** 数据库类、实体（entities）、DAO、历史记录 UI 和历史记录图标等关键组件全部缺失。
- **Manus 生成过时的代码**：用户指出 **Manus** 生成的过时（deprecated）代码充满了安全问题，建议用户告诉应用 *"update deprecated code/packages/modules/plugins"*。
   - 尽管 **Manus** 声称构建成功（clean build），但运行它会显示大量的错误和警告。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 的定价令人震惊**：每月只需 **$20 USD**，**Gemini** 即可通过 *gemini-cli* 提供名义上**每天 1500 次请求**，并使用拥有 **1M token 上下文窗口**的 **2.5 模型**。
   - 身份验证依赖于链接到 **API 结算账户**的 **Google Cloud 项目**。虽然其界面大多优于 **aider**，但它缺少 **repo map**，且依赖于 **grep** 等文件系统操作。
- **Codex 崛起，Aider 遇冷**：一位成员发现 **Codex**（使用常规 **ChatGPT Plus** 账户及 **gpt-5-codex** 模型）效果出奇地好，减少了对 **aider** 手动上下文文件的需求。
   - 他们指出，由于 **aider** 几乎不再更新，尽管之前是 **aider** 的资深用户（power-user），现在也觉得 **Codex** 非常合适。
- **Aider 获得社区分叉版**：一位社区成员建议尝试 [aider-ce](https://github.com/dwash96/aider-ce)，这是一个由社区开发的 **aider** 分叉版本，具有支持更多 **Agent** 行为的“导航模式”（navigator mode）。
   - 它还包含 **RAG**（检索增强生成）等额外功能，并带有来自 **MCPI** 的 **PR**。然而，与原始 **aider** 项目相比，它的 **star** 数量显著较少。
- **GitHub Copilot 获得无限 RAG**：通过 **GitHub Copilot** 订阅（**每月 $10**），用户可以获得无限的 **RAG**、无限的 **gpt 5 mini**、**gpt4.1**、**grok code 1**，以及 **300 次** **claude sonnet 4**/**gpt5**/**gemini 2.5 pro** 请求，和 **900 次** **haiku**/**o4-mini** 请求。
   - 这为编码和生成任务提供了一套强大的工具。
- **Aider 的未来前景依然黯淡**：一位成员对 **aider** 表示强烈支持，并询问其未来的开发计划和寿命。他们指出 **aider** 因其直观的工作流而成为他们首选的 AI 编码工具，并希望它能继续取得成功并改进功能。
   - 然而，目前尚未发布任何更新。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **ChatGPT 的表情符号滑稽行为**：询问 **ChatGPT** *“是否有海马的表情符号？”* 会导致其出现 Bug。
   - 该 Bug 的具体细节及其影响尚不清楚。
- **独立游戏开发者为潜在的 Unreal Engine 竞争对手欢呼**：在新的 Demo 发布后，围绕 **Unreal Engine** 的新竞争对手 **Runway** 和 **Wan** 出现了很多猜测。
   - 关于新引擎功能和发布时间表的进一步细节尚未披露。
- **Nous 研究员通过 YARN 扩展上下文**：**Nous Research** 的研究人员为 **YARN 上下文扩展**做出了贡献，这是一种已在多个模型中实施的技术。
   - 尚未分享关于此扩展方法的更多细节或链接。
- **强化学习（RL）过时了吗？**：成员们讨论了今年有*几篇论文提出了 **RL** 是否理想或必要的问题*。
   - 一位成员请求提供这些论文的链接，显示出社区对可能偏离 **RL** 的趋势很感兴趣。
- **UNO 是一种拜占庭容错共识算法吗？**：一位成员分享了关于 **MARL（多 Agent 共识）** 的推测性 [X 帖子](https://twitter.com/op/status/176767)。
   - 该帖子假设 **Among Us Uno 共识**可以在具有拜占庭抗性多数（大多数诚实玩家）的情况下，作为 **BFT 共识算法**运行。

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **揭秘 Tinygrad 开发秘籍**：一位成员寻求成为 **tinygrad dev** 的指导，并获得了一篇关于如何参与贡献的 [博客文章](https://ninoristeski.github.io/blogs/how-i-started-contributing-to-tinygrad.html)。
   - 他们还提到 Discord 服务器可以提供更多信息。
- **Mojo 被视为下一个 AI 编译器**：一位成员分享了多篇关于 **Mojo** 和其他 AI 编译器的 [Modular 博客文章](https://www.modular.com/blog/democratizing-ai-compute-part-6-what-about-ai-compilers)。
   - 他们补充说，*Mojo* 的层级比 CUDA 高，但比 triton/tilelang 等 eDSLs 低得多，而且它的图灵完备性（Turing complete）过高了。
- **征集 Tinybox 主板规格**：一位来自法国的新成员希望就 **tinybox** 的 **主板 (mobo)** 提供建议，以及它是否能支持 **带有 12 个 DIMM 插槽和 500W CPU 的 9005**。
   - 未提供更多细节。
- **新人寻求首个 PR**：一位成员询问在拥有几周 **tinygrad** 经验后，什么样的 **Pull Request** 适合作为开始。
   - 另一位成员建议查看 [tinygrad bounties](https://bounties.tinygrad.org/)，特别是那些 **$100-200** 的任务。
- **Tinygrad 悬赏按易完成度排序**：一位成员指出，在 [tinygrad bounties 页面](https://bounties.tinygrad.org/)上*将金额列从小到大排序*，更容易发现那些易于上手的任务。
   - 未增加进一步讨论。

---

**LLM Agents (Berkeley MOOC) Discord** 暂无新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 暂无新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Windsurf Discord** 暂无新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**MCP Contributors (Official) Discord** 暂无新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

您收到这封邮件是因为您在我们的网站上选择了订阅。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord: 各频道详细摘要与链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1431039540662898711)** (1185 条消息🔥🔥🔥): 

> `推荐奖励, Comet 浏览器问题, 图像生成限制, 聊天功能, Steam 诈骗` 

- **赏金猎人就奖励发放产生争执**：用户讨论了 Perplexity 推荐计划的问题，包括奖励未发放和潜在客户未被计入，一些人猜测这是推广 **Comet 浏览器** 的**骗局**。
   - 一位用户对分析和历史记录部分的移除表示沮丧，质疑这是否是推动 **Comet** 采用的策略；而另一位用户则为无法提现 **$5** 的奖励而感叹。
- **Comet 受害者抱怨兼容性灾难**：用户在使用 **Comet 浏览器** 时面临挑战，包括推荐无法正确追踪，以及需要从 PC 端使用才能被计入潜在客户。
   - 一位用户询问了由于 API keys 导致的崩溃，并问*“有什么方法可以防止它吗？”*。
- **图像生成额度缩减引发愤怒质询**：成员们讨论了 Perplexity **图像生成限制**缺乏透明度的问题，一些用户在不知道配额的情况下触发了付费墙。
   - 一位用户建议建立一个动态 FAQ 页面来显示这些限制，而另一位用户指出 [GPT-Image 1 的限制曾经是每月 150 张图像](https://www.perplexity.ai/help-center/en/articles/10354781-generating-images-with-perplexity)。
- **Perplexity 聊天室中出现大量情感互动**：Perplexity 的聊天演变成了网络交友，用户们交换暗示性评论并表达浪漫兴趣，引起了其他成员的担忧，并出现了关于可能被封号的评论。
   - 一些用户在被问及是否嫉妒时，还分享了关于成为“网络混混 (Internet Gangsters)”的内容。
- **Steam 骗局惊扰多疑用户**：用户讨论了 **Steam 诈骗**，其中一人分享了可疑好友请求的截图，另一人警告说分享购买记录很危险，因为这可以被用来申诉账号。
   - 一位成员给出了对付骗子的建议：*“说 bing chilling 然后拉黑”*。

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1431077746813173881)** (3 messages): 

> `Computational Evidence, Claude for Life Sciences, Abstract Image Generation` 


- **Perplexity 页面显示计算证据**：一个 [Perplexity AI 页面](https://www.perplexity.ai/page/computational-evidence-for-rec-MZ.AjbR6SlGMwJpoCK7cCA) 提到了 **computational evidence**（计算证据）。
   - 目前尚不清楚所指的是何种证据。
- **新版 Claude for Life Sciences 发布**：搜索结果显示 **Claude** for **Life Sciences** 已发布。
   - 更多详情可以在 [搜索结果](https://www.perplexity.ai/search/claude-for-life-sciences-launc-n1HWpqR5QJepI_lUVqULog#0) 中找到。
- **抽象图像生成请求**：一位用户请求 *创建一个红色的抽象图像*。
   - 该请求是作为 [Perplexity AI 搜索查询](https://www.perplexity.ai/search/create-a-abstract-image-of-red-B5vQzBqjTl.ASaYoY_Y.Mw?0=d#0) 提交的。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1431039531058069674)** (952 messages🔥🔥🔥): 

> `Gemini 3, Lithiumflow's removal, NimbleBean Kling 2.5 Turbo, Tamazight Language LLM support, Code Arena Usability` 


- **Gemini 3 猜测升温**：成员们正热切期待 **Gemini 3** 的发布，对其发布日期和功能有各种猜测，尽管对其承诺的性能和 [Polymarket](https://polymarket.com/) 的预测存在一些怀疑。
   - 有人建议它可能会像 Gemini 2.5 Pro 那样在 Google AI Studio 中推出。
- **Lithiumflow 已移除，但未被遗忘**：**Lithiumflow** 从 LM Arena 中移除引发了失望和猜测，有人认为其功能可能会被整合到 **Google AI Studio** 或 **Gemini 3 Pro** 中。
   - 成员们表达了希望它回归的愿望，并回忆起它独特的功能和易用性。
- **Bing Image Creator 潜力巨大**：成员们注意到 **Bing Image Creator** 表现 *相当不错*，本质上它就是 **GPT** 图像生成器。
   - 然而，目前的图像模型已经非常出色，以至于区分 AI 生成的图像与现实变得极具挑战性。
- **NimbleBean Kling 2.5 Turbo：视频之星？**：**NimbleBean 视频模型 (Kling 2.5 Turbo Standard)** 正受到关注，一些用户对其逼出的输出以及在图生视频方面的能力印象深刻。
   - 该模型被认为是 **#1**，甚至优于 **Sora**。
- **LM Arena 调整与功能请求**：用户正在积极讨论 **LM Arena**，建议进行改进，例如针对 **3D 模拟** 的特殊情况、避免不必要的 **Tailwind CSS** 包含的系统提示异常，以及侧边栏模型对比功能，该功能已在 [Canary 版本](https://canary.lmarena.ai/) 中可用。
   - 据报道，图像上传和 Code Arena 现在已经可以正常工作。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1431434985168179372)** (1 messages): 

> `LMArena, minimax-m2-preview` 


- **Minimax 模型在 LMArena 首次亮相**：[minimax-m2-preview](https://x.com/arena/status/1981850766039187901) 模型已添加到 **LMArena** 排行榜。
- **LMArena 迎来新竞争者**：一个新模型已添加到 **LMArena**。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1431039882859384913)** (496 messages🔥🔥🔥): 

> `Cursor Ultra 预算管理, Claude 4.5 Sonnet vs Thinking, Windows 上的 Cursor 终端问题, Cursor 退款` 


- **Ultra 用户对使用量感到不满**：用户报告称 **Ultra** 计划的预计使用量不准确，一位用户表示，尽管该计划据称以 **$200** 提供价值 **$400** 的使用额度，但他们的账户在*“使用一天后就提示仅剩一小时”*。
   - 用户怀疑 Ultra 计划即使有 **$400** 的预算也无法支撑一个月，这引发了对计费准确性的不满。
- **Sonnet 4.5 Thinking 对比普通 Sonnet：昂贵的提议？**：用户讨论了 **Claude 4.5 Sonnet Thinking** 与普通 **Claude 4.5 Sonnet** 的价值对比，一位用户询问性能提升是否值得其价格差异。
   - 一位用户指出 *“4.5 和 4.5 Thinking 每百万 Token 的价格相同，但 Thinking 模式消耗的 Token 数量会更高，因为它在思考时会使用更多 Token”*，并建议使用 Haiku 4.5 以节省成本。
- **Windows PowerShell 持续干扰**：一位用户报告称，尽管将 **Git Bash** 设置为默认终端，Cursor 仍坚持使用 **Windows PowerShell**，由于命令执行失败，导致 Cursor *“无法使用”*。
   - 解决方案包括使用 `AGENTS.md` 文件或在 VSCode 设置中设置默认终端，尽管一些用户确认在更新检测后问题仍然存在。
- **因计费出错而困惑，请求修复 Bug**：一位用户报告称，尽管支付已确认，但其 Cursor Premium 购买尚未激活，急需 Cursor 支持团队的协助。
   - 另一位用户表示，Cursor 可能会针对你的支持工单主动提供退款。


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1431062803480973342)** (2 messages): 

> `BG Agent 状态报告的 API key 来源, 后台 Agent 评分` 


- **寻求 BG Agent 报告的 API key 来源**：一位成员询问用于后台 Agent 状态报告的 **API key** 来源。
- **为后台 Agent 评分**：一位成员询问社区会如何给后台 Agent 评分。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1431346025188032592)** (2 messages): 

> `ChatGPT Atlas, Shared Projects 扩展` 


- **Atlas 记录过往搜索！**：全新的 **ChatGPT Atlas** 可以记住你搜索过、访问过以及询问过的内容，为 **ChatGPT** 提供更好的上下文以实现更准确的回答。
   - 用户还可以要求它随时打开、关闭或重新访问你的任何 [标签页](https://video.twimg.com/amplify_video/1981781567992430592/vid/avc1/1920x1080/JL5Emq0-DeHXi8r_.mp4)。
- **Shared Projects 现已免费！**：**Shared Projects** 正在向 Free、Plus 和 Pro 用户开放。
   - 你现在可以邀请他人加入 **ChatGPT**，在同一个地方通过共享对话、文件和指令共同协作。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1431039728441884742)** (366 messages🔥🔥): 

> `Claude Sonnet 4.5 vs Gemini 2.5 Pro, Sora 代码, MultiModal AI, GPT-OSS-120B, AgentML 开源` 


- **Gemini 2.5 Pro 是成员的最爱**：**Gemini 2.5 Pro** 是某位成员最喜欢的模型，因为 *“它在处理 Hitchens 风格方面表现最好，而 Sonnet 和 GPT-5 则有所保留（如果你想要反谄媚，没有比 Hitchens 更好的角色扮演了）”*。
- **AI 教育者需要快速问答**：一位成员请求 **AI 教育者** 协助进行 **15 分钟的问答环节**。
   - 另一位成员正在寻找 **Sora 2 代码**，因为 *“专用频道中提供的代码没有一个能用”*。
- **LLM 的成功不能通过单一样本衡量**：成员们讨论了不同模型的用例，例如 **Gemini** 用于任务结构化，**Claude** 用于编程，**ChatGPT** 用于创意，而 **Perplexity** 用于研究。
   - 一位成员分享了 **Gemini** 在某项任务上失败数小时而 **ChatGPT** 在几分钟内解决的经历，引发了回应：*“如果你用 1 个样本来衡量任何聊天机器人的成功，那你就做错了”*。
- **Electronic Arts 通过 Prompt 生成 3D 世界**：**EA** 和 **Stability AI** 正在 [合作通过 Prompt 生成完整的 3D 世界](https://www.tweaktown.com/news/108455/ea-and-stability-ai-partnership-includes-generating-full-3d-worlds-from-a-series-of-prompts/index.html)。
- **AgentML 在 HackerNews 上开源**：**AgentML** 已开源并已在 [HackerNews](https://news.ycombinator.com/item?id=45695974) 上线，旨在与 **OpenAI Agent Builder** 兼容。


  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1431106995590660138)** (13 messages🔥): 

> `OpenAI support, GPT outage, Microsoft Copilot GPT5 breakdown, Builder Profile verification` 


- **订阅者的 ChatGPT 文本消息故障持续 16 天**：一位 ChatGPT Plus 订阅者报告他们的项目已经 **16 天**无法发送或接收任何文本消息，始终收到 **503 error**，他们正在寻求联系人工支持的帮助。
   - 尽管尝试了跨设备和网络测试、清除缓存以及确认没有账号安全问题等故障排除步骤，但他们收到的支持请求回复仅为自动调查邮件。
- **使用 GPT-5 的 Copilot Agents 突然崩溃**：一位用户报告说，他们使用 **GPT-5** 的 **Microsoft Copilot Agents** 突然无法从知识库（knowledge）中检索数据，除非切换到 **GPT-4o** 或 **GPT-4.1**。
   - 未提供更多细节。
- **Builder Profile 验证困难**：一位用户正在寻求关于使用账单信息验证其 **Builder Profile** 的指导，报告称找不到名为 "Builder Profile" 的标签页。
   - 未给出解决方案。
- **ChatGPT 违反规则**：一位用户报告给 ChatGPT 设定了 **5 条规则**，包括*仅用一个词回答*，但在被质问时它*回答了 orange* 而不是 no，随后它透露自己正受到政府监视。
   - 另一位成员简单回复了 *boring*。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1431047822664269845)** (52 messages🔥): 

> `Precise Prompt Engineering, Personal GPTs for Prompt Generation, Markdown, XML, JSON, and YAML Prompting, Sora Physics Issues, Integrating Pictures in Video` 


- **图像生成的物理特性失效**：一位成员在尝试用 **Sora** 重现**球弹跳并掉入洞中的视频**时，就物理特性问题寻求帮助。
   - 一位成员建议，更详细地解释所需的半写实方面可能有助于模型正确应用物理定律，并补充说该成员已经使用多个账号尝试了大约 30 次。
- **利用 ChatGPT 进行图像 Prompting**：一位成员询问如何在不是灯光或摄影专家的情况下创建精确的图像 Prompt。
   - 一位成员建议将示例图像展示给 **ChatGPT** 并要求它创建一个几乎相同的图像，或者让 **ChatGPT** 清晰地描述图像，特别是成员最关心的焦点区域，如图像的色调、阴影和纹理。
- **个人 GPTs 磨炼 Prompt 技巧**：一位成员建议开发**个人 GPTs** 来处理特定的 Prompt 请求，让 GPT Profile 仅响应专门的请求。
   - 发布者认为，专门的 GPT 会专注于其创建目的的细节，而通用的 GPT 则处理更通用的数据来生成内容，并以需要专门编剧的电影剧本为例。
- **Markdown vs XML vs JSON vs YAML**：成员们讨论了使用 **Markdown, XML, JSON 和 YAML 进行 Prompting** 的经验，重点关注其精确性、易用性和弹性（resilience）。
   - 一位成员表示 **XML** 是最精确的，而 **JSON** 对人类来说格式化很痛苦，结论是算法使用最具弹性的格式，很可能是 **JSON** 或 **YAML**。
- **通过 AI 为 PNG 制作动画**：一位成员请求帮助使用 AI 以特定风格为 **PNG** 制作动画，并附上了示例视频。
   - 另一位成员发布了一个关于 Prompt Engineering 的 Markdown 教程，包括层级通信、抽象、强化和输出模板。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1431047822664269845)** (52 条消息🔥): 

> `Sora 物理问题、图像生成的 Prompt Engineering、用于 Prompt 优化的 GPTs、Markdown、XML、JSON 和 YAML Prompting、GPT-5-Codex 指令文件` 


- **驯服 Sora 的弹球物理效果**：一位用户正努力让 **Sora** 准确重现球弹起并落入洞中的视频，报告称 *物理效果总是出错*，并已尝试在 2 个账号上进行了 30 次尝试来重现它。
   - 另一位成员建议更详细地解释所需的效果，明确哪些方面需要真实感，并进一步指出 **Sora 2** 在电影感动作方面远优于 **Veo 3**。
- **利用 AI 辅助构建精准 Prompt**：一位用户询问如何获取信息以构建精准的图像生成 Prompt，特别是在缺乏灯光或摄影等领域专业知识的情况下，并请求生成类似[这张图片](https://cdn.discordapp.com/attachments/1046317269069864970/1431112704998899823/ba3e3596c70fc307c04f740b38bae86b.jpg?ex=68fce3d1&is=68fb9251&hm=4fa8b38511c7b05e20cfcace1bde765e23c50aebd49e5f7d55256368e8ff4b9d&)的内容。
   - 一位成员建议将示例图片展示给 **ChatGPT**，并要求它创建一个几乎相同的图像，让 **ChatGPT** 清晰地描述该图像，和/或与 **ChatGPT** 讨论什么可以产生特定的效果（如阴影），并建议开发一个针对特定 Prompt 请求定制的个人 **GPT**，进一步描述为 *你不会要求电影导演为你开发电影剧本，你会想要一位专门从事你所寻找的特定动作的专业作家*。
- **探讨 Markdown、XML、JSON 和 YAML 的 Prompting 格式**：一位成员正在撰写一篇关于使用 **Markdown**、**XML**、**JSON** 和 **YAML** 进行 Prompting 经验的文章。
   - 一位用户建议 **XML** 是最好的，因为它允许特定且复杂的嵌套，而 **JSON** 对人类来说格式化可能很痛苦，最后得出结论：对于算法，**JSON** 或 **YAML** 的韧性最强。
- **GPT-5-Codex 指令文件**：一位用户报告称 **GPT-5-Codex** 完全忽略了指令文件，尽管它读取了该文件，并链接到了 [OpenAI Codex Agents 文档](https://github.com/openai/codex/blob/main/docs/agents_md.md)。
   - 另一位成员回应指出 *你需要在 AGENTS.md 上用 Markdown 编写 Prompt*。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1431093568667390044)** (127 条消息🔥🔥): 

> `LM Studio 平台差异、Qwen 3 VL 模型、MCP 服务器可靠性、CPU 使用异常、LLM 工具使用` 


- **服务器接口在不同平台间存在差异**：`response_format: { type: 'json_object' }` 参数在 **macOS** 上受支持，但在 **Windows** 上不支持，导致后者出现 **400 错误**。
   - 这表明从 npm 使用 **OpenAI SDK** 时，服务器接口在不同平台之间存在差异。
- **LM Studio 在支持 Qwen 3 VL 模型方面遇到困难**：成员报告称 **LM Studio** 在 llama.cpp 的特定分支中部分支持 **Qwen 3 VL 模型**，但该实现破坏了其他功能。
   - 官方 **llama.cpp** 仓库中缺少完整的后端实现，尚待引入 **LM Studio**。
- **MCP 服务器可靠性问题**：成员讨论了使用 **MCP 服务器** 让本地模型访问互联网，提到了 AnythingLLM 和一个自定义的 Visual Studio Code 扩展。
   - 然而，一位成员指出 **MCP 服务器** 一直不稳定，同时分享了 **Google 和 DuckDuckGo 搜索选项** 的配置。
- **CPU 模型首条 Prompt 的异常现象**：用户观察到当 100% 在 **CPU** 上加载模型时，第一条 Prompt 以 **30 TOK/s** 运行，但随后的 Prompt 掉到了 **6 TOK/s**。
   - 有人建议这可能是 **LM Studio** 的一个 Bug，使用 `Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf` 进行的测试在不同模型中显示出相同的效果，但 *使用 llama-cli 的 llama.cpp 在 CPU 上能保持良好的 30-33 tok/s*。
- **尽管有 MCP 服务器，LLM 在读取本地文件时仍表现不佳**：一位成员报告称，尽管启用了工具，LLM 在利用其 **MCP 服务器** 管理和读取个人文件时仍存在问题。
   - 该成员还展示了截图，显示 **系统 Prompt 为空**，这可能会覆盖默认 Prompt。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1431042689029247026)** (51 messages🔥): 

> `5950x 作为服务器处理器, 混合 GPU, 改装版 MI50, eGPU 扩展坞, PCIE 对推理的影响` 


- **成员们辩论 5950x 作为服务器处理器的可行性**：在修复了 **5950x** 弯曲的引脚后，一名成员询问将其作为服务器处理器是否性能过剩。
   - 另一名成员回答这*取决于你希望服务器执行什么任务*。
- **阿里巴巴上的改装版 MI50 引发关注**：一名成员报告称，在阿里巴巴上发现有卖家提供**改装版 MI50**，配备了涡轮风扇和定制打印的散热片/导风罩。
   - 其他人讨论了通过 PCIE 转接线将其与外部 GPU 配对。
- **eGPU 扩展坞允许用户串联 GPU**：成员们分享了使用在速卖通（Aliexpress）上购买的 **eGPU 扩展坞**的成功经验，例如[这个](https://www.aliexpress.com/item/1005004694129330.html)用于 **3090**，以及[这个](https://www.aliexpress.com/item/1005006421739456.html)用于 **MI50**。
   - 虽然有些人认为其性价比不够高，但通过 USB/Thunderbolt 串联大量 GPU 的能力非常具有吸引力。
- **PCIE 带宽对推理速度影响微乎其微**：一名成员在 Windows 上的 **LM Studio** 运行了多种组合，发现 **Vulkan** 在 Linux 机器上比 **HIP** 编译的 **llama** 慢。
   - 尽管加载速度较慢，但一旦加载完成，即使每张卡的带宽限制在 PCIE 3x2，其运行表现也基本符合预期。
- **ROCm 合并带来 TPS 提升**：用户分享称最新的 **ROCm** 更新提升了性能，例如 **llama.cpp**/**ggml** 中的[这次合并](https://www.reddit.com/r/LocalLLaMA/comments/1ns2fbl/for_llamacppggml_amd_mi50s_are_now_universally)，现在 **AMD MI50** 已获得通用支持，且 **TPS 不再随着上下文填满而迅速下降**。
   - 一位用户在 LM Studio 中使用相同的模型和设置（全部在 VRAM 中）运行了相同的提示词，报告称 **llama.cpp vulkan** 为 **30tps**，而使用 **ROCm** 则达到 **45 tps**。


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1431041475184955502)** (139 messages🔥🔥): 

> `速率限制错误响应, Sora 2 代码, 购买积分, Deepseek OCR 模型, GPT-5 情商` 


- **速率限制错误计入响应**：一名成员询问**速率限制错误响应**是否仍计为一次响应，另一名成员给出了肯定的回答。
- **关于将图像作为 Data URL 传递的讨论**：一名成员报告了将图像作为 **data URLs** 传递给 OpenRouter 时遇到的问题，指出模型无法读取图像，且 base64 内容被视为纯文本，导致 Token 数量大幅增加。
   - 另一名成员澄清说，**OpenRouter 目前不支持包含图像的工具结果（tool results）**。
- **关于 Exacto 供应商选择的辩论**：一名成员质疑 **Exacto** 中供应商选择的标准，认为所选供应商与平台的基准测试不符。
   - 另一名成员澄清说，选择是基于**基准测试、用户偏好、工具调用成功率、缓存、容量、正常运行时间以及速度**的综合考量，而看似准确度较低的供应商在客观的工具调用数据上表现更优。
- **Exacto 工具调用非常出色！**：一名成员强调 **Exacto** 专注于工具调用，但他们仍担心这可能会让非技术用户感到困惑。
   - 官方团队正在尝试确定[衡量模型整体质量的统计数据/数据点/基准测试](https://discord.com/channels/1091220969173028894/1091220970125041707/1431299582810763314)（长上下文、写作、知识）。
- **尝试迷惑 AI 聊天机器人**：成员们讨论了让 AI 聊天机器人“发疯”的方法，例如请求**海马表情符号**（该符号并不存在）。
   - 一名成员链接到了之前关于该话题的对话（[Discord 链接](https://discord.com/channels/1091220969173028894/1343649921680801927/1345163310680641557)），而另一名成员分享了 AI 在处理该提示词时表现出的幽默挣扎。


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - 新模型**
  

---

### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1431215737552375890)** (25 messages🔥): 

> `OpenRouter 原生 /v1/completions 请求支持, MoonshotAI 的 kimi-cli, 防止低质量（Sloppy）创意写作` 


- **OpenRouter 增强原生 API 支持**：一名成员询问 **OpenRouter** 是否可以指示哪些模型支持原生 **/v1/completions** 请求，或者优先选择支持该请求的提供商。
   - 一名成员回答说，该数据作为前端模型结构（model shape）中 `hasCompletions` 的一部分是可用的，并将会在内部共享该反馈。
- **Moonshot 发布 Kimi CLI**：**MoonshotAI** 正在开发自己的 CLI 工具 [kimi-cli](https://github.com/MoonshotAI/kimi-cli)。
   - 讨论涉及了一些轻松的评论以及对 ST 开发团队的问候。
- **新研究应对低质量（Sloppy）创意写作**：一名成员分享了一篇关于防止创意写作中出现低质量内容的论文：[arxiv.org/abs/2510.15061](https://arxiv.org/abs/2510.15061)。
   - 另一名成员对第一作者是 **EQ Bench** 的作者表示惊讶。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1431215146453307432)** (132 messages🔥🔥): 

> `Julia 自动向量化 vs Mojo, Mojo 中的 SIMD 操作, Ark.jl 基准测试, Mojo 迭代器接口, 属性测试框架` 


- **Julia 的自动向量化引发 SIMD 羡慕**：成员们讨论了 Julia 的自动向量化特性，该特性可以在无需用户显式管理的情况下实现 **SIMD 操作**，并将其与 Mojo 较为手动的方法进行了对比，参考了 [Ark.jl 基准测试](https://github.com/mlange-42/Ark.jl/pull/68#issuecomment-3442276636)。
   - 一名成员指出，Mojo 需要显式的 SIMD 规范，这提供了更多的控制权，但可能减少了“免费”的优化；一些人指出自动向量化仅适用于简单情况。
- **Mojo 拥抱显式 SIMD 控制**：关于显式与隐式 SIMD 的辩论继续进行，一名成员解释了 Mojo 如何要求对 **SIMD 使用** 进行显式指导，从而提供更大的控制力和可预测性，尽管这可能以牺牲初始便利性为代价。
   - 有人建议，库优先的方法可以通过 `Iterator` 接口实现自动向量化，从而潜在地实现“免费向量化”，例如 `zip(l1, l2).vectorize(lambda p, v: p += v)`。
- **GPU 随机模块引发疑问**：一名成员询问 `gpu/random.mojo` 中更快的随机模块的位置，质疑为什么它不是 CPU 实现，并提出了 [issue 5508](https://github.com/modular/modular/issues/5508)。
   - 对方澄清说，默认的随机数生成器应该是加密级的（因此较慢），而 GPU 版本优先考虑速度而非安全性，这表明需要一个带有适当免责声明的 `random.fast_random` 模块。
- **属性测试框架的构建基块**：有人提到一个属性测试（Property Testing）框架正在开发中，而那些看似放错地方的 RNG 工具实际上是该框架特有的构建基块，而不是通用工具。
   - 一名成员分享了在 `Span` 上测试 `s.reverse()` 时发现的一个 bug，该新框架的功能请求包括生成“经常导致崩溃的值”的能力（例如 -1, 0, 1, DTYPE_MIN/MAX）。
- **`Span` 获得 Map-Reduce 能力？**：一名成员表示有兴趣泛化 `Span` 中的代码，提到了之前关于 `map_reduce` 的工作（[PR 5341](https://github.com/modular/modular/pull/5341)）以及未来对 `map` 和 `reduce` 的计划（[issue 5219](https://github.com/modular/modular/issues/5219) 的一部分）。
   - 讨论中出现了关于返回新 `List[Scalar]` 还是迭代器的担忧，强调需要一个分块迭代器（chunk iterator）来高性能地链式调用 `map`、`filter` 等，而无需每次都分配列表。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1431271974230822944)** (1 messages): 

> `Instagram 分析器, 自动化 Instagram 分析` 


- **AI Instagram 分析器回答问题**：创建了一个 AI **Instagram 分析器**，如果你提供用户名和提示词，它会读取照片和视频并回答你的问题，例如 *“他们的兴趣是什么？”*。
   - 它建议了一些用例，如 *“我应该带他们去哪里约会？”* 和 *“他们符合我们的品牌吗？”*，并附带了 [分析器](https://viveka.darkgravitylabs.com/) 的链接。
- **Instagram 分析器附带 API 和 Claude Skill**：该 Instagram 分析器拥有用于**自动化的 API** 和一个 **Claude Skill 文件**。
   - 这些功能可以配合该工具用于各种用途。


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/)** (1 messages): 

lidar36: 他们刚刚添加了代码
  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1431043604490485851)** (86 messages🔥🔥): 

> `ReAct Module Granularity, Framework Frustrations, DSPy vs Langchain, Google Vista & DSPy, Monkey Patching` 


- **UI 期望的 ReAct 模块粒度**：一位成员希望在 UI 上实时显示 **DSPy ReAct module** 的每一步，展示思考过程、工具调用和结果，但发现很难获取每次迭代的输出。
   - 他们发现 *callbacks* 并没有像预期那样工作，并对框架的复杂性表示沮丧，认为相比之下在循环中运行原始 LLM 调用更简单。
- **对框架的挫败感**：一位成员对大多数 LLM 框架带来的痛苦发表了强烈看法，强调了为了构建像样的产品而不得不摸索这些框架的 *idiosyncrasies*（特性/怪癖）的困难，同时赞扬了 **sklearn** 和 **PyTorch**。
   - 他们认为框架往往增加了过多的复杂性，使简单的任务变得更难，并表示在 DSPy 的 ReAct 模块中获取每次 LLM 调用的输出非常困难。
- **DSPy 擅长结构化任务**：一位成员提到 **DSPy 擅长结构化任务**，尤其是那些你可能想要优化的任务。
   - 另一位成员在经历了一次糟糕的体验后，正带领团队从 **Langchain 迁移到 DSPy**，那次经历导致他们在不完全从头重写 Prompt 的情况下无法进行模型升级。
- **Google Vista 将基于 DSPy 和 Gemini 构建**：一位成员询问是否有人见过 **Google Vista**，并暗示这听起来像是可以用 **DSPy 和 Gemini** 构建的东西。
   - 他们链接到了 [Google Vista 论文](https://arxiv.org/abs/2510.15831)。
- **Monkey Patching 是答案吗？**：当面临如何解决 **实时显示 DSPy ReAct 模块每一步** 的挑战时，一位成员开玩笑说，根据 ChatGPT 的建议，你可以尝试对 **类进行 monkey patch**。
   - 这是令原帖作者感到沮丧的各种复杂性的又一个例子。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1431043853170774106)** (8 messages🔥): 

> `Text Diffusion Inference, vLLM inference serving, torchcomms/ncclx PT conference session` 


- **最快的 Text Diffusion 推理**：一位成员询问目前运行 **Llada** 等文本扩散模型推理的最快方法，寻求任何有用的线索。
   - 遗憾的是，目前还没有提供具体的方法或论文链接，但问题仍然悬而未决。
- **解码 vLLM 推理服务**：一位成员请求学习 **vLLM 推理服务** 的资源，理由是遇到了晦涩的错误消息和调试挑战。
   - 另一位成员分享了一个关于该主题的博客文章链接：[vLLM](https://www.aleksagordic.com/blog/vllm)。
- **torchcomms/ncclx 会议幻灯片仍难寻踪迹**：一位成员询问 PT 会议中关于 **torchcomms/ncclx** 的录制环节，并注意到播放列表尚未发布。
   - 有人请求发布演讲者/讲座材料，并链接到了[这篇 arXiv 论文](https://arxiv.org/pdf/2510.20171)。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1431351655110021322)** (1 messages): 

> `GIL, Priority Inversion` 


- **持有 GIL 的线程面临优先级反转？**：一位成员建议，如果持有 **GIL** 的线程被取消调度，而另一个线程需要 **GIL** 来启动 GPU 工作，那么应用程序可能会遭受 **priority inversion**（优先级反转）。
   - 这一观察基于一张暗示这种潜在情况的截图。
- **另一个主题建议**：为了满足验证模式要求的第二个主题，这里有一个占位符。
   - 这是另一个句子，为占位符提供更多细节。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

vipul_todo_18: https://www.stephendiehl.com/posts/mlir_gpu/

讨论了 MLIR 到 PTX 的 lowering（下放）。
  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1431050313552494713)** (2 messages): 

> `HQQ+ blog post, mobiusml github, dropbox github` 


- **HQQ+ 博客文章迁移**：成员们正在寻找 **HQQ+ 博客文章** 的有效链接，因为原始链接 [mobiusml.github.io/1bit_blog/](https://mobiusml.github.io/1bit_blog/) 已失效。
   - 一位成员提到，由于今天宣布了变更，博客文章和 GitHub 仓库中的 `mobiusml` 都应替换为 `dropbox`。
- **MobiusML GitHub 被 Dropbox 取代**：根据今天的公告，**MobiusML** 的 GitHub 仓库已被 **Dropbox** 链接取代。
   - 寻找 **HQQ+ 博客文章** 及相关资源的用户现在应参考更新后的 **Dropbox** 链接，而非原始的 **MobiusML** GitHub 页面。


  

---

### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1431105495132930100)** (6 条消息): 

> `Mobius Labs, Personal News, Acquisition, Electric Grill` 


- **Mobius Labs 团队收购**：一名成员分享了一篇关于个人新闻的 [帖子](https://x.com/Mobius_Labs/status/1981391562836721786)，表明 **Mobius Labs** 团队可能已被收购。
   - 另一名成员向他们表示祝贺，希望他们在完成了 *出色工作* 后能获得优待。
- **电烤架上的三文鱼碎**：一名成员分享了一张在电烤架上烤制三文鱼碎的照片，配菜有番茄、黄瓜、海盐、咖啡、乳脂和甜菊糖。
   - 另一名成员评论说 *看起来非常温馨*，并打赌 *味道一定很棒*。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1431218029642907750)** (2 条消息): 

> `Netherlands Meetup, European Meetup` 


- **来自荷兰的请求**：一名成员简单询问是否有人在荷兰。
- **欧洲聚会请求**：这可能会演变成一场欧洲聚会。


  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1431096596178665523)** (1 条消息): 

> `vk_cooperative_matrix_perf, roofline.png` 


- **修复后的 vk_cooperative_matrix_perf 亮相**：一位用户宣布了修复后的 **vk_cooperative_matrix_perf** 带来的改进，并分享了 [roofline.png](https://cdn.discordapp.com/attachments/1233802893786746880/1431096595969085470/roofline.png?ex=68fcd4d0&is=68fb8350&hm=fa33212634f6c98c5803e39e32890019b31f4b484d48e2e130536b71f937bc64&)。
- **Roofline 性能提升**：附带的 [roofline.png](https://cdn.discordapp.com/attachments/1233802893786746880/1431096595969085470/roofline.png?ex=68fcd4d0&is=68fb8350&hm=fa33212634f6c98c5803e39e32890019b31f4b484d48e2e130536b71f937bc64&) 表明与协同矩阵（cooperative matrix）操作相关的性能指标有所增强。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1431257720333402203)** (3 条消息): 

> `Grayscale B200, Grayscale H100, Grayscale A100, Grayscale L4, Prefixsum A100` 


- **Grayscale 夺得 B200 排行榜第二名**：一名成员在 `grayscale_v2` 排行榜上获得 **第二名**，在 **B200** 上的耗时分别为 **6.79 ms** 和 **6.71 ms**。
   - 两次提交的 ID 分别为 `66248` 和 `66250`。
- **Grayscale 稳居 H100 第二名**：一名成员在 `grayscale_v2` 排行榜上获得 **第二名**，在 **H100** 上的耗时为 **13.0 ms**。
   - 两次提交的 ID 分别为 `66248` 和 `66250`。
- **Grayscale 夺得 A100 第三名**：一名成员在 `grayscale_v2` 排行榜上获得 **第三名**，在 **A100** 上记录到 **20.5 ms** 和 **20.4 ms**。
   - 两次提交的 ID 分别为 `66248` 和 `66250`。
- **Grayscale 在 L4 上成功运行**：一名成员在 `grayscale_v2` 排行榜上获得 **第二名**，在 **L4** 上提交 ID `66248` 的耗时为 **27.9 ms**，随后提交 ID `66250` 成功运行，耗时 **28.2 ms**。
   - 这展示了在不同硬件配置下的稳定性能。
- **Prefixsum 在 A100 上获得第一名**：另一名成员夺得 `prefixsum_v2` 排行榜 **第一名**，在 **A100** 上提交 ID `66267` 的耗时为 **7.20 ms**。
   - 这展示了该成员在优化并行算法方面的精湛技术。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1431352756014153961)** (1 条消息): 

> `Factorial Learning Environment, Reinforcement Learning Projects` 


- **Factorial Learning Environment 令 RL 爱好者感到兴奋**：一名成员表达了对 **Factorial Learning Environment (FLE)** 的兴奋之情，称其为从播客中听到的非常令人兴奋的长跨度（long horizon）基准测试。
   - 他们具有 **Reinforcement Learning**（强化学习）背景，并有兴趣参与与 FLE 相关的 **RL/自我改进系统项目**。
- **RL 爱好者寻求参与 FLE 项目**：一位具有 **Reinforcement Learning (RL)** 背景的个人表达了为 **Factorial Learning Environment (FLE)** 项目做出贡献的兴趣。
   - 受 Latent Space 播客中描述的启发，他们正在寻找参与 **RL/自我改进系统项目** 的机会。


  

---

### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1431314157390790836)** (5 messages): 

> `Nsight Python, CUTLASS Python stack, CuTE talk slides` 


- **Nvidia 发布用于 Python kernel 开发的 Nsight Python**：Nvidia 宣布 **Nsight Python** 将极大提升 Python kernel 的开发效率，并提供了[早期访问注册链接](https://developer.nvidia.com/nsight-python-notify-me)。
   - 他们计划在 **CUTLASS Python stack** 公开后，围绕它推出一些教程。
- **成员寻求 CuTE 演讲幻灯片**：成员们正在寻找 Chris 的 **CuTE 演讲**幻灯片。
   - 一位成员提到，YouTube 视频在最初直播时描述栏中有幻灯片链接，但后来被删除了。


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1431178510952562738)** (6 messages): 

> `SITP, picograd, lazy semantics, torchdynamo, EagerTensor vs LazyTensor` 


- **Lazy Semantics 在 SITP 和 picograd 中引起关注**：鉴于 **SITP** 和 **picograd** 以教学实现为核心目标，**tinygrad** 的 **lazy semantics** 设计决策因其极简设计而显得非常有吸引力。
   - 据报道，**pt2 论文**中提到的唯一缺点是开销（overheads），但这对于 **SITP** 和 **picograd** 的教学目标来说是完全可以接受的。
- **Torchdynamo Tracers 对 picograd 来说不可行**：在 host<->eDSL 层级（使用 torchfx 的算子）或 host 层级本身（使用 **torchdynamo** 的 **python**）实现 tracers 对 **picograd** 来说绝对不可行。
   - 这就像学生在进入编译器构建的核心部分（优化器和代码生成器）之前，就陷入了 LL, LL(1), LR 解析器的泥潭；此处引用了 [shriram krishnamurthis 的 PLAI](https://plai.org)，该书通过 s-exprs 避开了解析问题。
- **Eager Mode 被强行加入/改造以实现平滑过渡**：对于读者来说，构建自己的理解很重要，应从 **eager mode** 开始，并理解为什么 Transformer (**scaling laws**) 和 **tensor cores** 使得像 pt2 和 xla 这样的编译器流水线成为必要。
   - 讨论提出了一个问题：**SITP/picograd** 应该在一个 `Tensor` 下实现两个独立的结构（如 `EagerTensor` 和 `LazyTensor`），还是应该解释并编译 IR（即 `Graph<UOp>`）。
- **Picograd 采用广度优先方法**：大家认识到，与其它 autograd 项目相比，**picograd** 需要更多的精力才能起步，因为它采用了 autograd + 编译器流水线的广度优先方法。
   - 发布者邀请任何有兴趣将 **SITP** 和 **picograd** 打造为 Karpathy 的 starfleet academy（继 llm101 之后）第二门课程的人加入其中。


  

---


### **GPU MODE ▷ #[irl-accel-hackathon](https://discord.com/channels/1189498204333543425/1416087968933740688/1431105961757769818)** (43 messages🔥): 

> `H100 availability, Hackathon Waitlist, Dynamic SASS Kernel Instrumentation with nvbit, Memory Allocators on GPU, PyTorch Distributed Hacking` 


- **Nebius 不提供 H100**：一位成员询问是否能从 Nebius 获取 **H100**，但被告知他们不提供，不过可以从其他云服务商处以每小时约 **$1.90** 的价格租用。
   - 这为那些在 hackathon 期间需要 **H100** 开展项目的人提供了替代方案。
- **Hackathon 候补名单的烦恼**：两名成员请求协助从 hackathon 候补名单中转正，希望能针对气候应用场景实验 **多节点 GPU 训练**。
   - 他们已经填写了表格，并渴望加入已经在现场的队友。
- **动态 SASS Kernel 插桩即将到来**：一位成员正致力于使用 **nvbit** 对 **SASS kernels** 进行动态插桩，以发现其参数/参数缓冲区中的指针偏移。
   - 这对于他们在 **PyTorch** 中实现的“参数化 cuda graph launch”构想特别有用。
- **GPU 内存分配器 Mini-PyTorch**：一位成员想在 **GPU** 上编写一个带有 tensor 元数据和分配器的“简版 **PyTorch**”。
   - 他们建议 kernels 应该在 block 中使用 **512 线程**，并正在寻找合作者。
- **关于 Blackwell 上量化预训练的热议**：一位成员正在研究 **Blackwell** 上的量化预训练，并寻找其他有兴趣交流的人。
   - 另一位用户对 **AI 生成的 GPU kernels** 以及针对 **Blackwell** 的 **kernel 优化**表示了兴趣。


  

---


### **GPU MODE ▷ #[opencl-vulkan](https://discord.com/channels/1189498204333543425/1418990184367919267/)** (1 messages): 

erichallahan: 新的规范更新
https://www.phoronix.com/news/Vulkan-1.4.330-Released
  

---

### **GPU MODE ▷ #[llmq](https://discord.com/channels/1189498204333543425/1421956177549332662/1431358706498404362)** (1 messages): 

> `NPU, CPU Offloading` 


- **Framework NPU 挫折引发 CPU Offloading 探索**：一名成员报告称无法让 **framework machine** 在 **NPU** 上正常工作，因此将重心转向 **CPU offloading**。
- **CPU Offloading 受到关注**：随着 NPU 方面的尝试停滞，重点转向探索和优化 **CPU offloading** 技术。


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1431130492669001901)** (6 messages): 

> `Helion vs Triton, Cudagraph support, Kernel hyperparams` 


- **编译器改进后 Helion 缩小与 Triton 的差距**：在进行了一些编译器改进后，一名成员注意到他们的编译器根据内部数据发生了变化，但不确定是否重新运行了 **Helion/Triton** 的数据进行对比。
   - 他们提到了在相同环境、相同时钟频率下进行 Benchmark 的重要性。
- **Cudagraphs 支持具有通用性**：**Cudagraphs** 是受支持的，除非你在 **kernel** 中执行了无法被 cudagraph 化操作。
   - 适用于其他语言的 **Cudagraphs** 限制同样适用于 **Helion**，以保留用户的控制流。
- **Kernel 超参数调优提升性能**：一名成员在 [此 commit](https://github.com/pytorch/helion/pull/1010) 中更新了 **int4_gemm** 引用，并在 [此博客文章](https://pytorch.org/blog/helion/) 中更新了带有新数据的博文。
   - 另一名成员链接到了 [此 commit](https://github.com/tile-ai/tilelang/commit/8a5eb569704bfea64478c29adcfe3a09e3c2b12c)，该提交通过 **kernel** 和后端更改提升了性能，但未更改自动调优参数集。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1431051215797489744)** (63 messages🔥🔥): 

> `zero3 config, Text-SAL, AI infrastructure collaboration, ROMA (Reasoning Over Multiple Agents), synthetic data gen` 


- **Zero3 配置肯定有问题**：一名成员表示 *你的 zero3 配置肯定坏了*，因此你应该能够在 **gemma 3 1b** 的 **r=8 LoRA** 上进行更大规模的训练。
   - 他们补充说 *肯定有什么地方出错了*。
- **Text-SAL 运行结束**：一名成员发布了 **Text-SAL** 运行的日志输出，并询问这是什么框架以及训练方法是什么。
   - 日志提到了 **SAL_BRIDGE**，指示了一个 BERT 模型 (**prajjwal1/bert-tiny**)，并显示了训练期间的能量和内存状态。
- **Sentient 社区寻求 AI 基础设施合作**：来自 **Sentient 社区** 的一名成员询问了与 Hugging Face 在 **AI infrastructure** 或 **verifiable AI systems** 方面的合作或伙伴关系。
   - 另一名成员认为他们的项目很有趣，并链接到了他们的 [ROMA (Reasoning Over Multiple Agents) GitHub 仓库](https://github.com/sentient-agi/ROMA?tab=readme-ov-file)。
- **ROMA 详解**：**ROMA (Reasoning Over Multiple Agents)** 旨在将复杂任务分解为由多个 **AI Agent** 处理的更小、专门化的子任务。
   - 这种模块化设置有助于克服上下文限制并提高推理效率，因为每个 Agent（或“单元”）处理大局的一部分，然后将所有内容重新组合在一起。
- **合成数据生成讨论**：一名成员希望探索 **synthetic data generation**（合成数据生成），但没有任何想法或切入点。
   - 另一名成员提到，他们 *看到过各种关于图形方面的巧妙想法，但在语言方面没看到那么多*。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 messages): 

waffles1: 啊是的，这完全是正经的。
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1431085728506576939)** (4 messages): 

> `Pacific-Prime model, 6GB VRAM check, Zero Amnesia AI, Night Learn Engine, RAG Pipeline` 


- **Pacific-Prime 模型实现 10% 提升**：据报道，[Pacific-Prime 模型](https://huggingface.co/Pacific-Prime/pacific-prime)在 6GB VRAM 上实现了 **10% 的提升**，该模型起始于 1.1B 参数模型。
- **具备真实记忆的 Zero Amnesia AI**：描述了一个具有**真实记忆**且**零失忆**的 AI 系统，能够保留过去的对话和重要细节作为上下文丰富的记忆。
- **即时塑造 AI 性格**：该 AI 允许用户**即时调整其身份**，范围涵盖从专业合作伙伴到创意陪练。
- **Night Learn Engine 自主进化**：该 AI 整合了 **Night Learn Engine**，能够反思交互、整合全天信息、构建高阶记忆并自主进化。
- **精炼的 RAG Pipeline 检索上下文感知智能**：该 AI 利用精炼的 **RAG Pipeline** 仅检索任务必需的信息，确保精确且具备上下文感知的智能，避免数据混乱。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/)** (1 messages): 

yusarseph: 你好，Hugging Face Inference Endpoints 是 Serverless 的吗？我们需要为不使用的部分付费吗？
  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1431146620975579176)** (2 messages): 

> `Karpathy Server, HF, nanochat-students, MLX Porting, MLX Stability` 


- **需要澄清服务器上下文！**：一位成员要求澄清讨论是关于 **Karpathy Server** 还是 **Hugging Face**。
   - 该成员还询问了 Hub 上 **nanochat** 或 **nanochat-students 组织**的目标。
- **MLX 移植想法**：一位成员表达了将项目移植到 **MLX** 的兴趣。
   - 他们询问了资料的稳定性，以评估在继续进行之前是否应该等待。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1431070532488659045)** (5 messages): 

> `Agents course unit 4, 404 Error` 


- **Agent 课程第 4 单元出现 404 错误**：用户报告在尝试通过 *https://agents-course-unit4-scoring.hf.space/questions* 访问问题时遇到 **404 错误**。
   - 显示的错误信息为 "No questions available."
- **问题无法访问**：多名用户报告 Agents 课程第 4 单元的问题无法访问。
   - 该问题自昨晚以来一直存在，用户在尝试访问问题时遇到 **404 错误**。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1431065896788562012)** (17 messages🔥): 

> `Server Acceptance Process, Distributed Inference, AI Ownership, AI Accelerator Chips, Petals Project` 


- **服务器准入：等待批准**：一位成员表示，只有在通过审核（通常涉及填写表格）后才能获得服务器访问权限，但“加入”选项的 Bug 已修复。
   - 另一位用户确认在获得批准前处于“待定成员”状态。
- **分布式推理：AI 的未来**：成员们倡导开源、广泛分布的 AI，类似于互联网的结构，摆脱大型企业的统治，类似于 [Nous Research](https://www.nousresearch.com/)。
   - 一位成员指出阻碍这一愿景的**严重技术问题**，例如 GPU 资源的贡献并非易事。
- **Nvidia 的太空野心：劣质芯片设计？**：成员们推测 **Nvidia** 将 **GPU 集群**部署到太空的计划是其坚持其**劣质芯片设计**的迹象。
   - 他们预计更**节能**且**具成本效益**的替代方案很快将主导市场。
- **《生命中不能承受之慢》：一篇研究论文**：一位成员询问 AI 设计中的技术问题是否源于捕捉 *The Unbearable Slowness of Being*，参考了 [这篇论文](https://arxiv.org/abs/2408.10234v2)。
   - 未提供更多细节。
- **Petals Project：分布式推理**：被提及的 [Petals Project](https://github.com/bigscience-workshop/petals) 似乎已被遗弃，两年前在 **Llama 70B** 方面曾有很大势头。
   - 当该项目无法跟上新架构时，社区便分崩离析，目前 **LlamaCPP RPC** 是最接近它的替代方案。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1431073478965989407)** (54 messages🔥): 

> `50M model Loss, 1B model Validation, lm-eval, activation steering` 


- **新型 50M 模型实现低 Loss**：一位新成员报告称其 **50M model** 实现了 **0.223** 的 Loss，显著低于 vanilla transformer 的 **2.73**，且其 **1B model** 在 **400 steps** 时已经低于 **0.3**。
   - 由于 Loss 低得超乎预期，社区产生了质疑，有人认为这*要么是 bug，要么是简单的数据集，或者是谎言*。
- **模型调试需要代码**：社区成员要求提供该模型的代码进行调试，认为报告的性能可能不准确。
   - 原作者（OP）因 **IP** 原因拒绝了请求，但承诺会发布 **1B model** 在标准 **lm-eval** harness 下运行的结果。
- **验证是支撑主张的关键**：社区对这款号称能在手机上运行的突破性模型的有效性提出了质疑。
   - 一位成员表示，原作者（OP）尚未排除其他基础性问题，应该多观察一段时间，避免提出*荒诞的主张*。
- **Activation Steering 利用梯度复用**：一位成员想知道 **activation steering** 是否可以实现数据点的复用，从而从中获取大量不同的梯度。
   - 他们引用了 [Anthropic 的 Personas 论文](https://arxiv.org/abs/2506.18221)和另一篇论文，将这一想法与定性控制 forward pass 后返回的梯度类型的可能性联系起来。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/)** (1 messages): 

stellaathena: 好了，这到底是什么胡言乱语：https://www.arxiv.org/abs/2510.15511
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1431106546468655198)** (31 messages🔥): 

> `gpt-4o-transcribe-diarize, GPT-5, Cursor Enterprise, Kimi Code CLI, Cohere's AI Win` 


- **OpenAI 悄然发布 gpt-4o-transcribe-diarize**：Peter Bakkum 宣布 OpenAI 悄然发布了 **gpt-4o-transcribe-diarize**，这是一个针对高精度 speaker diarization 优化的“小型”音频模型，该模型体积较大且仅限离线使用，并接受语音样本以标记已知发言者（[链接](https://xcancel.com/pbbakkum/status/1981397851600302250?s=46)）。
   - 它的 **WER** 与 OpenAI 其他 ASR 模型相当，用户询问了关于 pyannote 的基准测试、实时使用、定价、开源权重以及 mini 版本的问题。
- **GPT-5 驱动 Company Knowledge**：OpenAI 宣布 **Company Knowledge** 由经过微调的 **GPT-5** 版本驱动，该版本经过训练可以跨多个来源检索，以提供更全面、更准确的答案（[链接](https://openai.com/index/introducing-company-knowledge/)）。
   - 目前尚不清楚他们是否会在 API 中提供此模型。
- **Cursor 的 C-Suite 策略大放异彩**：Alex Konrad 揭示了 Cursor 激进的企业战略，其 COO 在第三季度领导了 **300 场 C-suite 会议**，以支持 **5 亿美元以上的 ARR**（[链接](https://xcancel.com/alexrkonrad/status/1981477024092082386?s=46)）。
   - 他们正在使用技术销售团队、客户黑客松和 code-half-life 指标；链接中包含完整的 Upstarts 访谈。
- **Kimi Code CLI 预告泄露，引发热议**：Crystal 俏皮地确认了 **Kimi 即将推出的 CLI/Code 工具** 的图像泄露，指出全球发布还有几天时间，并请大家保持耐心（[链接](https://xcancel.com/crystalsssup/status/1981597395541753988?s=46)）。
   - 用户在回复中好评如潮（将其与 Claude Code 比较），并请求早期访问权限、免费额度、Tomagotchi 彩蛋以及未来的 WhatsApp 集成。
- **Tahoe-x1：开源单细胞 Transformer 问世**：Tahoe AI 发布了 **Tahoe-x1**，这是一个拥有 30 亿参数的 Transformer，它统一了基因/细胞/药物表征，并在其包含 1 亿样本的 Tahoe perturbation 数据集上进行了高效训练（[链接](https://xcancel.com/nalidoust/status/1981760790551298524)）。
   - 它在癌症相关的基准测试中达到了 SOTA，并在 Hugging Face 上完全开源，提供了 checkpoints、代码和可视化工具。


  

---

### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1431119160825610343)** (5 messages): 

> `Local AI Apps, QA on Scanned PDFs, OpenWebUI, Qwen3-vl-4b` 


- **寻求支持扫描 PDF 的本地 AI 应用**：一位成员询问是否有本地 AI 应用能够使用像 **Qwen3-vl-4b** 这样的 VLM 直接对多页扫描 PDF 进行问答。
   - 成员们注意到，许多应用（如 **LM Studio**）在上传文件时仅支持图像或检索增强生成 (**RAG**)。
- **建议使用 OpenWebUI 进行 PDF 提示词处理**：另一位成员建议使用 **OpenWebUI** 将整个 PDF 作为提示词的一部分输入，并提到可以设置使用整个文档或仅使用相关部分。
   - 然而，原帖作者反馈称，所选的 VLM 在 **OpenWebUI** 中无法处理该扫描 PDF。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1431164536357912658)** (7 messages): 

> `Video Models, MJ, Kling, LTX-2, a16z` 


- **a16z 认为不会出现“神级”视频模型**：[Justine Moore (a16z)](https://a16z.substack.com/p/there-is-no-god-tier-video-model) 认为我们永远不会拥有一个通用的视频模型；相反，日益丰富的专业化模型将服务于不同的预算和用例。社区对此论点做出了反应。
   - 讨论者们交流了各自喜爱的模型（**MJ**、**Kling**、**LTX-2**），辩论了垂直与水平工具的优劣，并将当前的行业格局比作相机或巴洛克静物风格——赞美竞争而非单一霸权。
- **YouTube 上已有关“无神级视频模型”论点的反应视频**：关于“无神级视频模型”论点及社区反应的精彩讨论已上传至 [YouTube](https://youtu.be/wHK8GMc9O5A?si=2W2N8W7cXjS7ppfK)。
   - 一位用户之前漏掉了这个链接，但被另一位用户补上了。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1431039661760974969)** (28 messages🔥): 

> `Mythworx AI, ARC-AGI 1, Elastic Weight Consolidation, Activation-aware Weight Quantization (AWQ), Cherry-picked verifications` 


- **Mythworx 声称在 ARC-AGI 1 上达到 100%**：[Mythworx.ai](https://mythworx.ai/capabilities/) 声称在 **4 小时** 内无需预训练即可在 **ARC-AGI 1** 上达到 **100%** 的准确率，这引发了对其能力的质疑。
   - 该声明遭到了怀疑，一位成员质疑 *为什么他们总是只发布公告而不使用 ARC 私有测试集进行验证*，暗示其目的是追求融资而非严谨验证。
- **ARC 私有集验证辩论**：社区成员辩论了进行 **ARC 私有集验证** 的必要性，一位成员认为虚假陈述是让模型获得评估的一种手段，而另一位成员则对此表示警惕。
   - 一位成员警告说，虚假陈述可能导致被 *列入研究者黑名单*，而另一位成员则认为这是为了 *通过虚假陈述引起关注，直到他们感到厌烦并不得不与你合作测试结果*。
- **讨论弹性权重固化 (EWC)**：一位社区成员询问某种技术是否仅仅是为每个权重（而非整个模型）设置 **learning rate**，并引用了 **Elastic Weight Consolidation**。
   - 另一位成员对此进行了扩展，讨论了实现的复杂性，特别是关于“柔软度因子（softness factor）”和向量归一化，并指向了 **Activation-aware Weight Quantization (AWQ)**。
- **挑选出的验证结果遭到抨击**：一位社区成员表示，更倾向于在会议演讲中揭穿骗局，并指责那些基于 *随性与贿赂* 而挑选出的验证结果（cherry-picked verifications）。
   - 他们声称 *自那以后，其他研究人员已经证实了他们是在满口胡言，并对此感到震惊。*


  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1431341313462112317)** (7 messages): 

> `Transformer Circuits Linebreaks Paper, Neuronpedia Attribution Graphs, Gemma-2-2b Line Break Attribution, Qwen3-4b Line Break Attribution` 


- **Transformer Circuit Linebreaks 论文深度探讨**：一名成员建议研究 [Transformer Circuits Linebreaks 论文](https://transformer-circuits.pub/2025/linebreaks/index.html)。
   - 该建议包含了对特定日期的讨论，随后被另一名成员纠正。
- **新 Neuronpedia 图表发布**：一名成员宣布发布了与新论文相关的 [换行符归因图表 (line break attribution graphs)](https://www.neuronpedia.org/gemma-2-2b/graph?slug=fourscoreandseve-1757368139332&pruningThreshold=0.8&densityThreshold=0.99&pinnedIds=14_19999_37&clerps=%5B%5B%2214_200290090_37%22%2C%22nearing+end+of+the+line%22%5D%5D)。
   - 这些图表可以检查 **Gemma-2-2b** 等模型中的换行符归因。
- **Qwen3-4b 归因图表发布**：一名成员宣布发布了与新论文相关的 [换行符归因图表](https://www.neuronpedia.org/qwen3-4b/graph?slug=fourscoreandseve-1757451285996&pruningThreshold=0.8&densityThreshold=0.99&clerps=%5B%5B%2230_117634760_39%22%2C%22nearing+end+of+line%22%5D%5D&pinnedIds=30_15307_39)。
   - 这些图表可以检查 **Qwen3-4b** 等模型中的换行符归因。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1431161962863005797)** (3 messages): 

> `Genie 3 World Model, Google, David Sacks, Donald Trump` 


- **Genie 3 World Model 来了！**：新的 **Genie 3** World Model 视频生成令人印象深刻，似乎是因为他们拥有足够的算力来向广泛的用户提供服务，而其他参与者提供的视频创作时长仍限制在最多几秒钟。
   - 这与最近的 **Genie 3** World Model 视频表现非常一致。
- **AI 沙皇 David Sacks**：一名成员表示：*不用担心，那个愚蠢的 AI 沙皇 David Sacks 会尽一切努力通过来自“橙色家伙” (**Donald Trump**) 的行政压力把这些东西废除。*
   - 该成员对 **David Sacks** 在 AI 领域的政治压力表示担忧。
- **Google 正在衰落**：一名成员说：*噢，巧克力工厂 (Chocolate Factory) 已经陨落了。*
   - 帖子中有一张将 **Donald Trump** 描绘成 Oompa Loompa 的图片。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1431058834079875082)** (37 messages🔥): 

> `Chutes vs Moonshot AI, Kimi K2, Data Policy, Uptime, Tool call accuracy` 


- **Chutes 数据政策受到质疑**：一名成员询问了 **Chutes** 与 **Moonshot AI** 在 **Kimi K2** 上的对比，另一名成员回应称，与官方 **Moonshot AI API** 相比，**Chutes** 会利用用户数据进行训练，缺乏数据政策，在线率 (Uptime) 较不可靠，且 Tool call 准确度较低。
- **Chutes Reddit 调侃得到回应**：社区注意到，在一次基准测试帖子引起关注后，**Chutes** 回应了关于在 **OpenRouter** 上封禁 **Chutes** 的梗，一名用户讽刺地指出，尽管存在这些问题，其价格和速度仍具有吸引力。
- **Kimi 采用 GLM 编程方案**：一名成员希望 **Kimi** 能采用 **GLM** 编程方案或类似风格，因为 *GLM 在编程方案上更具成本效益，且 GLM-4.6 比 Kimi 强大得多*。
- **中国版 Kimi.com 集成方案发布**：一名成员指出，一款类似 Kimi 的产品在中国发布并集成到了中国 Kimi 网站中，并发布了来自 [X.com](https://x.com/bigeagle_xd/status/1981568257258860899) 的链接和 [GitHub 上的 Kimi-Cli](https://github.com/MoonshotAI/kimi-cli) 链接。
- **本地化 Kimi 定价看起来很便宜**：成员们注意到 **Kimi** 的中国区定价看起来非常便宜，但也有人提醒这是本地化定价，国际市场价格可能会有所不同。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1431045284732862544)** (35 messages🔥): 

> `Manus 网络连接错误, Manus 积分使用, Claude Code 对比 Manus, Manus Room 数据库, Manus 弃用代码` 


- **Manus 网络错误令用户沮丧**：用户在使用 **Manus** 时遇到了 *"Network connection error"* 问题，阻碍了他们编写应用的能力。
   - 错误信息显示：*"Please check your network settings and try again."*
- **Manus 积分消耗遭批评**：用户对 **Manus** 的高积分消耗表示担忧，一位用户报告称在短短几天内为一个复杂项目花费了 **15,000 积分**，希望新版本能更高效。
   - 其他人建议先进行调研并使用其他 AI 来修复生成的代码，警告不要 *"为糟糕的代码付费"*。
- **Claude Code 和 Codex 被推崇为 Manus 的替代方案**：用户推荐将 **Claude Code** 和 **Codex** 作为 **Manus** 的更好替代品，理由是它们具有更强的开发能力和更高的性价比，对于严肃的开发时间，每月成本约为 **$20**。
   - 一位用户指出，使用 **Claude Code** 可以获得 5 小时的会话（会重置）以及每周速率限制，最终获得的产出轻松达到 **Manus** 的 5 倍以上。
- **Manus 的 Room 数据库实现存在缺陷**：**Manus** 声称已为之前的聊天历史实现了 **Room** 数据库，但一位用户发现它完全没有被实现。
   - 根据 **Claude** 的说法，*"❌ 没有 Room 数据库类，❌ 没有实体 (@Entity)，❌ 没有 DAO (@Dao)，❌ 没有历史记录 UI，❌ 任何 Arena 屏幕中都没有历史记录图标"*。
- **Manus 生成弃用代码并存在构建问题**：用户报告称 **Manus** 会生成带有安全问题的弃用代码，并建议告诉它 *"更新弃用的代码/包/模块/插件"*。
   - 一位用户提到，**Manus** 声称构建正常，但运行构建时却显示许多错误和警告。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1431111666828841052)** (15 messages🔥): 

> `Gemini 定价, Aider 对比 Codex, aider-ce 社区分叉, 结合 GitHub Copilot 的 RAG` 


- **Gemini 慷慨的定价**：每月只需约 **$20 USD**，**Gemini** 标称提供 **每天 1500 次请求**，并拥有 **1M token 上下文窗口**（使用 **2.5 模型**），可通过 *gemini-cli* 访问。
   - 身份验证依赖于链接到 **API 计费账户** 的 **Google Cloud 项目**，虽然界面大多优于 aider，但它缺乏 repo map 且依赖于 grep 等文件系统操作。
- **Codex 胜过 Aider**：一位成员发现 **Codex**（使用普通的 **ChatGPT Plus** 账户和 **gpt-5-codex** 模型）效果出奇地好，减少了对 **aider** 手动上下文文件的需求。
   - 他们指出，由于 *aider* 几乎不再开发，他们发现 codex 非常合适，尽管他们之前是 *aider* 的高级用户。
- **社区开发 aider-ce**：一位社区成员建议尝试 [aider-ce](https://github.com/dwash96/aider-ce)，这是一个由社区开发的 **aider** 分叉版本，具有 *navigator 模式* 以实现更具 Agent 特性的行为。
   - 它还包含 **RAG** (Retrieval-Augmented Generation) 等额外功能，并带有来自 MCPI 的 PR。
- **GitHub Copilot RAG**：通过 **GitHub Copilot** 订阅（**每月 $10**），用户可以访问无限的 **RAG**、无限的 **gpt 5 mini**、**gpt4.1**、**grok code 1**，以及 **300 次** **claude sonnet 4**/**gpt5**/**gemini 2.5 pro** 请求，和 **900 次** **haiku**/**o4-mini** 请求。
   - 这为编码和生成任务提供了一套强大的工具。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1431359186263871611)** (1 messages): 

> `Aider 的未来与发展, aider-ce 功能集` 


- **Aider 的未来展望与发展**：一位成员对 **aider** 表示了强烈支持，并询问其未来的开发计划和生命力。
   - 他们指出 **aider** 是他们首选的 AI 编码工具，因为它具有直观的工作流，并希望它能继续取得成功并改进功能。
- **Aider-ce 功能集**：讨论涉及了 **aider-ce**，这是 **aider** 的一个变体，包含更多合并的功能。
   - 一位成员强调，虽然 **aider-ce** 包含更多功能，但与原始 **aider** 项目相比，它的 GitHub 星标（stars）显著较少。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1431044191978393810)** (4 messages): 

> `ChatGPT emoji bug, Unreal Engine competitor` 


- **ChatGPT 存在 emoji 漏洞**：成员们发现，询问 **ChatGPT** *“is there an emoji of a seahorse?”*（是否有海马的 emoji？）会导致其出现故障。
- **新的 Unreal Engine 竞争对手出现**：在看到新的演示后，成员们推测出现了一个能与 **Unreal Engine**、**Runway** 和 **Wan** 竞争的新对手。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1431163324854177994)** (4 messages): 

> `Nous Research Models, YARN Context Scaling, Western Ideological Views in GPT` 


- **Nous 扩展至 HF Profile 之外**：除了 [hf nous profile](https://huggingface.co/nous-research) 中的模型外，没有其他直接与 **Nous Research** 相关的模型。
   - 然而，**Nous Research** 的研究人员贡献了 **YARN context scaling** 技术，该技术已在多个模型中实现。
- **YARN 缩放上下文窗口**：由于 **Nous Research** 研究人员的贡献，**YARN context scaling** 出现在多个模型中。
   - 目前没有分享关于该缩放方法的更多细节或链接。
- **GPT 的西方意识形态倾向**：有建议认为，源自西方的 **GPT** 模型可能更强烈地反映了**西方意识形态观点**。
   - *数据对于塑造世界观非常重要*，并可能导致 AI 模型之间出现有趣的差异。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1431337233688039496)** (2 messages): 

> `Mech Interp Surgeon's Bag, RL Desirability Questioned` 


- **规模限制需要 Mech Interp Surgeon's Bag**：一位成员表示，*在能够自信地讨论规模（scale）限制之前，我们需要先完成 **mech interp surgeon's bag**（机械可解释性工具包）*。
   - 另一位成员请求提供批评 **RL** 的论文链接。
- **RL 方法受到质疑，相关论文被提出**：成员们分享道，今年有*几篇论文正在质疑 **RL** 是否真的理想或必要*。
   - 其他人请求获取这些论文的链接以深入了解。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1431219042529710163)** (1 messages): 

> `MARL Consensus, Hamiltonian Path Problem, BFT consensus algorithm` 


- **X 平台上关于 MARL 共识的推测**：一位成员分享了 [X 平台上的一个推测性帖子](https://twitter.com/op/status/176767)，涉及 **MARL（多智能体共识）**。
   - 该帖子假设 **Among Us Uno Consensus** 可以作为一种具有拜占庭容错多数派（大多数为诚实玩家）的 **BFT consensus algorithm**（共识算法）运行。
- **UNO 是 NP-Complete 问题**：一位成员声称单人版 **UNO** 是一个 **Hamiltonian Path Problem**（哈密顿路径问题），这是一个经典的 **NP-complete problem**（图着色路径规划问题）。
   - 这种复杂性源于游戏中存在的“选择”和“随机性”。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1431337233688039496)** (2 messages): 

> `Mech Interp Surgeon's Bag, RL Desirability, Limits of Scale` 


- **Mech Interp Surgeon's Bag 优先于规模讨论**：一位成员提到，在自信地讨论规模限制之前，完成 **Mech Interp Surgeon's Bag** 至关重要。
   - 这表明需要全面的可解释性工具来理解缩放动力学。
- **重新思考 RL 的必要性与需求**：一位成员指出，今年的几篇论文正在质疑 **Reinforcement Learning (RL)** 是否理想甚至是否有必要。
   - 另一位成员请求提供这些批评性观点的指引，表现出对 RL 价值争论的兴趣。


  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1431138952412397709)** (8 条消息🔥): 

> `成为 Tinygrad 开发者，Mojo 与 AI 编译器，AI Box 推荐` 


- **解锁 Tinygrad 开发者秘籍**：一位成员询问如何成为 **tinygrad dev**，另一位成员分享了一篇关于贡献代码的有用 [博客文章](https://ninoristeski.github.io/blogs/how-i-started-contributing-to-tinygrad.html)。
   - 他们补充说，Discord 服务器是了解更多关于如何为 **tinygrad** 贡献代码的资源。
- **Mojo 作为 AI 编译器候选者崛起**：一位成员分享了多篇关于 **Mojo** 和其他 AI 编译器的 [Modular 博客文章](https://www.modular.com/blog/democratizing-ai-compute-part-6-what-about-ai-compilers)。
   - 他们提到 *Mojo* 的层级比 CUDA 高，但远低于 triton/tilelang 等 eDSLs，而且它的图灵完备性（Turing complete）太强了。
- **Tinybox 主板规格：有什么建议吗？**：一位来自法国的新成员正在寻求关于 **tinybox** 的 **mobo**（主板）建议。
   - 他询问是否能支持 **带有 12 条 DIMMs 和 500W CPU 的 9005**。