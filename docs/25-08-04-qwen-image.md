---
companies:
- alibaba
- google-deepmind
- openai
- bytedance
- kaggle
- tencent
date: '2025-08-04T05:44:39.731046Z'
description: '**阿里巴巴**发布了 **Qwen-Image**，令业界感到惊喜。这是一款拥有 **20B（200亿）参数的 MMDiT** 模型，在双语文本渲染和图形海报生成方面表现卓越，目前已开放权重并提供演示。**谷歌
  DeepMind** 向 Ultra 订阅用户推出了 **Gemini 2.5 Deep Think**，该模型在推理能力上有了显著提升，基准测试成绩大幅增长（AIME
  提升 11.2%，HLE 提升 13.2%，LiveCodeBench 提升 13.4%），足以与 **OpenAI 的 o3 Pro** 媲美。字节跳动的 **SeedProver**
  在数学定理证明方面取得了业界领先（SOTA）的成果，超越了 DeepMind 的 AlphaGeometry2。OpenAI 正在开发一种“通用验证器”（universal
  verifier），以实现数学和编程能力的增益迁移。谷歌和 Kaggle 推出的竞争性推理基准测试和游戏竞技场，凸显了推理模型效率的范式转变，其意义堪比最初 Transformer
  带来的飞跃。其他势头强劲的开源权重模型还包括 **GLM-4.5**、**XBai o4** 以及专注于高效训练的**腾讯混元**。*“Qwen 就是你所需的一切。”*'
id: MjAyNS0w
models:
- qwen-image
- mmdit
- gemini-2.5
- o3-pro
- seedprover
- glm-4.5
- xbai-o4
- hunyuan
people:
- swyx
- demishassabis
- tulseedoshi
- mparakhin
- teortaxestex
- cgeorgiaw
- dorialexander
- steph_palazzolo
- corbtt
- synthwavedd
- epochairesearch
title: Qwen-Image：SOTA 级文本渲染 + 4o 级图像生成编辑能力，开源权重 MMDiT。
topics:
- bilingual-text-rendering
- image-generation
- image-editing
- synthetic-data
- reasoning
- math-theorem-proving
- benchmarking
- instruction-following
- model-efficiency
- open-weight-models
- model-transparency
- competitive-evaluation
---

**Qwen 就是你所需要的一切。**

> 2025年8月1日至8月4日的 AI 新闻。我们为你检查了 12 个 Reddit 子版块、544 个 Twitter 账号和 29 个 Discord 社区（227 个频道，14214 条消息）。预计节省阅读时间（以 200wpm 计算）：1248 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以优美的 vibe coded 风格展示所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻分类，并在 @smol_ai 上向我们提供反馈！

阿里巴巴 Qwen 团队意外[发布](https://x.com/Alibaba_Qwen/status/1952398250121756992)了一个 20B MMDiT 模型，“特别擅长创建带有原生文本的精美图形海报” ([博客](https://qwenlm.github.io/blog/qwen-image/)，[论文](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/Qwen_Image.pdf#page=14.57))。

[](https://resend-attachments.s3.amazonaws.com/aZx2Cus84j7nUjl)

问问你身边的华语使用者，他们是否见过这种水平的非阿拉伯文本渲染：

[](https://resend-attachments.s3.amazonaws.com/eYc92vDYGrg3osA)

[](https://resend-attachments.s3.amazonaws.com/WCJt5u4A5z6B7zH)

当然，它在英文方面的表现也很出色：

[](https://resend-attachments.s3.amazonaws.com/LuRphbToh17jqHN)

除了纯图像生成，它们在图像编辑方面的表现也令人震惊，可与 [Flux Kontext](https://flux-ai.io/flux-kontext/) 媲美：

[](https://resend-attachments.s3.amazonaws.com/oVJ3VWbD5nbZn7b)

这份 46 页的技术报告展示了西方实验室罕见的透明度：https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/Qwen_Image.pdf#page=14.57

[](https://resend-attachments.s3.amazonaws.com/FUBv1l4QqkNW6Lc)

并对使用合成数据来实现其文本渲染效果提供了一些（但非全部）见解：

[](https://resend-attachments.s3.amazonaws.com/GeGPPO0kIWDiRxb)

---

# AI Twitter 简报

**前沿推理：Gemini 2.5 Deep Think、新数学/证明系统以及正面评测**

- **Google 的 Gemini 2.5 Deep Think 向 Ultra 订阅用户开放**：DeepMind 宣称其在硬核基准测试中达到 SOTA，早期用户报告称其相比之前的 Gemini 有重大提升，且在某些任务上与 OpenAI 的 o3 Pro 持平。根据 [@swyx](https://twitter.com/swyx/status/1951460518293807241) 的社区测试量化差异：与 o3 pro 在类似任务中的较小增幅相比，Gemini 2.5 Deep Think 在 AIME (2025) 上提升了 11.2%，HLE (知识) 提升了 13.2%，LiveCodeBench (编程) 提升了 13.4%。来自 [@demishassabis](https://twitter.com/demishassabis/status/1951468051578142848)、[@tulseedoshi](https://twitter.com/tulseedoshi/status/1952059171727437859) 和 [@MParakhin](https://twitter.com/MParakhin/status/1952028947153371631) 的演示线程显示，尽管目前有使用限制，但其推理能力有所提高，输出更加简洁。
- **数学与定理证明的飞跃**：
    - 字节跳动（ByteDance）的 SeedProver 在 PutnamBench 上取得了 331/657 的成绩（约为之前 SOTA 的 4 倍），在“轻量级”推理下为 201/657，并在 OpenAI 的 miniF2F 上达到 100%，超过了 DeepMind 的 AlphaGeometry2，详见 [@teortaxesTex](https://twitter.com/teortaxesTex/status/1951875052967739787) 和 [@cgeorgiaw](https://twitter.com/cgeorgiaw/status/1952301113446699347) 的总结。论文线程：[@Dorialexander](https://twitter.com/Dorialexander/status/1952094475725238479)。
    - 据 [@steph_palazzolo](https://twitter.com/steph_palazzolo/status/1952375778361954801) 报道，OpenAI 据传正在开发一种“通用验证器（universal verifier）”，旨在将数学/编程领域的收益转移到主观领域；相关的开源努力包括来自 [@corbtt](https://twitter.com/corbtt/status/1952437149544144984) 的 RULER 通用奖励。
    - “Hieroglyph”基准测试探测了发散性推理（Only Connect 风格）：根据 [@synthwavedd](https://twitter.com/synthwavedd/status/1951645151203324099)，模型在最难的 20 道题中得分低于 50%。
- **衡量元转变（Meta-shift）的基准**：根据 [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1951734757483487450) 的说法，推理模型在适合推理的任务上代表了大约 10 倍的计算等效增益——这与最初 Transformer 带来的飞跃相当。Kaggle 和 Google 推出了 Game Arena，在竞争性游戏中对模型进行压力测试（从文字象棋开始），由 Magnus Carlsen 和 Hikaru Nakamura 进行现场解说；详情见 [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1952406075996533077) 和 [@demishassabis](https://twitter.com/demishassabis/status/1952436066524299432)。Artificial Analysis 更新了其模型索引，增加了 IFBench（指令遵循）；在他们的评估方法中，Grok 4 保持领先，而 o3/o4-mini 超过了 Gemini 2.5 Pro ([线程](https://twitter.com/ArtificialAnlys/status/1952302030812483982))。

---

**开源权重模型浪潮：Qwen-Image、GLM-4.5 势头、XBai o4、Tencent Hunyuan 以及高效训练**

- **Qwen-Image (20B MMDiT) 以 Apache-2.0 协议发布**：具备强大的双语文本渲染能力（英文媲美 GPT-4o；中文同类最佳）及像素内文本合成功能，并支持广泛的图像风格；通过 [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1952398250121756992) 开放权重、代码和 Demo。社区指出其利用了微调后的 Wan 2.1 VAE 和 Qwen VL 文本编码器组件 ([@multimodalart](https://twitter.com/multimodalart/status/1952409238413684901))；公开 Demo 迅速达到饱和 ([@victormustar](https://twitter.com/victormustar/status/1952416615351366033))。
- **智谱 AI 的 GLM-4.5 在排行榜上攀升**：目前在 LMSYS Arena 总榜排名第 5，拥有 4000+ 投票，并在 Agent/工具调用（tool-use）方面表现强劲 ([@lmarena_ai](https://twitter.com/lmarena_ai/status/1952402506497020330); [@Zai_org](https://twitter.com/Zai_org/status/1952404744225349799))。Terminal-Bench 证实了其在推理/代码助手中的顶尖性能 ([链接](https://twitter.com/Zai_org/status/1952411485742760324))。需求暂时填满了 [Z.ai](http://z.ai/) Chat 的功能存储 ([更新](https://twitter.com/Zai_org/status/1951494857039454250))。
- **XBai o4 (并行测试时扩展/test-time scaling)**：以 Apache-2.0 协议开放权重；作者声称其在“中等模式”下优于 OpenAI o3-mini ([公告](https://twitter.com/theMetaStoneAI/status/1951486506562101656))。
- **腾讯混元（Tencent Hunyuan）小模型系列 (0.5B/1.8B/4B/7B)**：边缘侧就绪模型（支持单卡部署），具备 256K 上下文、工具/Agent 技能，并支持多框架（SGLang, vLLM, TensorRT-LLM）。[@TencentHunyuan](https://twitter.com/TencentHunyuan/status/1952262079051940322) 提供了仓库和 HF 权重链接。
- **大型且经济的 VLM**：阶跃星辰（StepFun）的 Step-3 (321B MoE) 旨在触达解码成本的帕累托前沿 ([@HuggingPapers](https://twitter.com/HuggingPapers/status/1952038716488208409))。
- **训练与优化**：
    - **GSPO**（来自 Qwen 的 RL 对齐技术）正受到关注；TRL v0.20 增加了原生支持和示例脚本 ([@SergioPaniego](https://twitter.com/SergioPaniego/status/1952305247411691871))。
    - 微软发布了 **Dion**（带有 Muon/MuP 选项和 Triton 内核的分布式优化器）。具备优秀的代码质量和基础设施说明；[@jxbz](https://twitter.com/jxbz/status/1951806916440854982) 和 [@JingyuanLiu123](https://twitter.com/JingyuanLiu123/status/1951885788855345221) 讨论了通信优化以及 FSDP/all-to-all 技巧。
    - Hugging Face 的 **Ultra-Scale Playbook**（200+ 页，4000+ 次扩展实验）涵盖了 5D 并行、ZeRO、FlashAttn、重叠（overlap）和瓶颈；对 HF Pro 用户免费 ([@Thom_Wolf](https://twitter.com/Thom_Wolf/status/1951581743607070851), [@ClementDelangue](https://twitter.com/ClementDelangue/status/1952048356710039700))。
- **代码专家生态系统**：
    - **Qwen3-Coder** 在 Cerebras 上运行速度快 17 倍，并可免费试用；本周末举行了黑客松 ([@SarahChieng](https://twitter.com/SarahChieng/status/1951453803905163693))。[@rasbt](https://twitter.com/rasbt/status/1951635208375034191) 发布了教学用 MoE Notebook（128 个专家，8 个激活；单张 A100）。
    - Fireworks 上提供了更小、更快的变体（**Qwen3-Coder-Flash, GLM-4.5-Air**），在简单、低延迟任务中的工具调用质量可与大型模型媲美 ([@dzhulgakov](https://twitter.com/dzhulgakov/status/1952049826067050735))。

---

**Agent 系统与编程：Claude Code 演进，基础设施成熟，以及“深度 Agent”模式**

- **Claude Code 更新**：Microcompact（自动清除旧的工具调用以延长会话）、支持 @-mention 的 subagents 以及针对每个 Agent 的模型选择，以及原生 PDF 摄入功能已上线 ([@_catwu](https://twitter.com/_catwu/status/1952488684579873195))。上下文/剪枝 (Context/pruning) 仍是常见的调优痛点；多位用户强调了在冗长与简洁之间的权衡 ([例如](https://twitter.com/giffmana/status/1952434564472644016))。
- **生态系统**：
    - Cline × Cerebras 黑客松吸引了 800 多名开发者参与即时 “vibe coding” ([@CerebrasSystems](https://twitter.com/CerebrasSystems/status/1952511328964509794))。Opencode 添加了 Together 的模型套件 ([@togethercompute](https://twitter.com/togethercompute/status/1952495692557046141))，Kilo 集成了 GLM‑4.5 ([Z.ai](http://z.ai/) [providers](https://twitter.com/Zai_org/status/1952390223742255504))。
    - Amp 与 Claude Code 的正面深度对比即将发布 ([@isaac_flath](https://twitter.com/isaac_flath/status/1952399160579366957))；Jules (Google) 现在支持在循环内开启 PR ([@julesagent](https://twitter.com/julesagent/status/1952446750167310456))；Lindy 3.0 发布了 Agent Builder、Autopilot 和团队协作功能 ([@Altimor](https://twitter.com/Altimor/status/1952414217187086441))。
- **设计模式**：
    - “Deep agents” (LangChain) 通过虚拟文件系统状态使多步 subagents 正式化；代码演示由 [@hwchase17](https://twitter.com/hwchase17/status/1952408450878918834) 提供。
    - 根据 [@CShorten30](https://twitter.com/CShorten30/status/1952376642283708788) 的说法，反思性提示词演进 (Reflective prompt evolution) 在复合系统 (GEPA) 中可以与 RL 媲美；OpenPipe 的 RULER 提供相对通用奖励 ([链接](https://twitter.com/corbtt/status/1952437149544144984))。
    - 记忆 (Memory) 正成为 Agent 个性化和效率的关键基础设施——分类学与策略由 [@_philschmid](https://twitter.com/_philschmid/status/1952370348600533000) 提供。
- **政策摩擦**：Anthropic 表示，由于违反服务条款 (ToS) 以及内部对 OpenAI 的过度使用，限制了 OpenAI 对 Claude Code 的访问，同时保留了用于安全评估/基准测试的 API 访问权限 ([@sammcallister](https://twitter.com/sammcallister/status/1951642025381511608))。

---

**多模态生成与视频：Grok Imagine、Runway Aleph、Veo 3 以及迈向实时化**

- **Grok Imagine 推出**：xAI 的图像/视频生成功能现已在应用内上线（最初通过候补名单，随后面向 Premium+ 和 Premium 用户），并附带快速生成演示和广泛认可 ([@tetsuoai](https://twitter.com/tetsuoai/status/1951444393065586840), [@tobi](https://twitter.com/tobi/status/1951789462268391749), [@chaitualuru](https://twitter.com/chaitualuru/status/1952174534142067092), [@obeydulX](https://twitter.com/obeydulX/status/1951724900198367515))。Elon Musk 报告称 6 秒片段的渲染时间为 15 秒（从 60 秒降至 15 秒），并目标在 3-6 个月内实现实时生成 ([进展](https://twitter.com/elonmusk/status/1951883927582552547)；“创意与 Imagine 一样快” [推文](https://twitter.com/elonmusk/status/1951516837906202782))。
- **Runway Aleph**：正式发布（网页版 + API），在课堂教学中被广泛采用 (USC/UPenn)，且可控性和可扩展性迅速提升 ([@runwayml](https://twitter.com/runwayml/status/1951634909501575659), [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1951663311503688018), [客户](https://twitter.com/c_valenzuelab/status/1951568696155017286))。社区实验展示了多步复合和 “无限 UI” 控制范式（例如 Blender + Aleph 工作流 [示例](https://twitter.com/c_valenzuelab/status/1952419024291188794)）。
- **Veo 3 图生视频**：已在 Video Arena Discord 提供并排对比测试 ([@lmarena_ai](https://twitter.com/lmarena_ai/status/1952052092719517729))。Video Arena 邀请广泛的模型对比和投票（Discord 链接见推文）。

---

**开源工具、基础设施以及 “开源模型作为国家优先级” 的推动**

- **开放基础设施成熟度**：Hugging Face Inference 正在推动“开放权重基础设施”向闭源 API 看齐 ([@ClementDelangue](https://twitter.com/ClementDelangue/status/1951668724848599143))，Jan 将 HF 添加为远程提供商 ([@jandotai](https://twitter.com/jandotai/status/1952248389531570333))，Qdrant Edge 进入嵌入式向量搜索的私有测试阶段 ([@qdrant_engine](https://twitter.com/qdrant_engine/status/1951631317939990765))。Modal 重申其为通用计算平台，而非仅仅是推理平台 ([@bernhardsson](https://twitter.com/bernhardsson/status/1951729049866514508))。
- **ATOM 项目 (美国开放模型)**：呼吁美国加大投资以夺回开放模型领导地位（在中国夏季激增之后），获得了各实验室研究人员的支持 ([@natolambert](https://twitter.com/natolambert/status/1952370970762871102)，以及 [@Miles_Brundage](https://twitter.com/Miles_Brundage/status/1952400404668657966) 和 [@finbarrtimbers](https://twitter.com/finbarrtimbers/status/1952401883391520794) 的背书)。VentureBeat 评论文章认为“开放至关重要” ([@bgurley](https://twitter.com/bgurley/status/1952031129143591234))。
- **数据/运维 (Data/ops)**：Google 的 AlphaEvolve 展示了 LLM 驱动的测试循环代码演进，产生了新型内核并获得了基础设施收益（减少 1% 训练时间） ([@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1952112235196678274))。RAG 规范进展包括对内部与外部来源进行分层重排序，以减少幻觉 ([@qdrant_engine](https://twitter.com/qdrant_engine/status/1951978606617326011))。

---

**热门推文 (按互动量排序)**

- [@nearcyan](https://twitter.com/nearcyan/status/1951926555934073147)：“你不会相信接下来会发生什么”（对信息流的病毒式元评论）
- [@sama](https://twitter.com/sama/status/1951695003157426645)：“未来几个月将发布大量内容”
- [@elonmusk](https://twitter.com/elonmusk/status/1951516837906202782) 及 [渲染更新](https://twitter.com/elonmusk/status/1951883927582552547)：关于 Grok Imagine 和近实时视频
- [@karpathy](https://twitter.com/karpathy/status/1951577221753094399)：“2024：所有人发布 Chat；2025：所有人发布 Code”（以及 [PayoutChallenge](https://twitter.com/karpathy/status/1952076108565991588)）
- [@gdb](https://twitter.com/gdb/status/1951882297172779336)：“成为一名软件工程师比以往任何时候都更有趣”
- [@LHSummers](https://twitter.com/LHSummers/status/1951998034973163940)：关于统计数据政治化作为威权主义倾向的论述
- [@balajis](https://twitter.com/balajis/status/1951515516939673996)：“像旧版 Twitter 一样提示你的 AI”（限制在 140 个字符/单词/行内）
- [@demishassabis](https://twitter.com/demishassabis/status/1951468051578142848)：Gemini 2.5 Deep Think 发布公告
- [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1951677714173477031)：关于 Thinky 拒绝 15 亿美元收购以及 MVP 留存的讨论
- [@OpenAI](https://twitter.com/OpenAI/status/1952414411131671025)：为“不留遗憾的时间”优化 ChatGPT（休息提醒、建议改进）
- [@naval](https://twitter.com/naval/status/1951900029389820253)：“优秀的团队丢弃的产品远比保留的多”
- [@paulg](https://twitter.com/paulg/status/1952155863864733750)：链接权重降低隐藏了网络上最优质的内容

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. Qwen-Image 20B 模型发布与基准测试

- [**QWEN-IMAGE 发布！**](https://huggingface.co/Qwen/Qwen-Image) ([得分: 711, 评论: 166](https://www.reddit.com/r/LocalLLaMA/comments/1mhhdig/qwenimage_is_released/))：**QWEN-IMAGE 是一款新发布的视觉模型，在内部基准测试中表现优于 Flux Kontext Pro。根据技术讨论，QWEN-IMAGE 支持广泛的图像理解任务，包括目标检测、语义分割、深度/边缘 (Canny) 估计、新视角合成和超分辨率。定性测试（如复杂的提示词遵循和文本渲染）表明其具有高保真度结果，尽管在生成的文本中注意到了细微的字体样式异常。** 评论者指出，提示词保真度和多任务能力尤其令人印象深刻，特别关注到 QWEN-IMAGE 在执行复杂的图文组合方面超越了之前的开放模型。
    - QWEN-IMAGE 支持全套图像理解任务，特别是目标检测、语义分割、深度/边缘 (Canny) 估计、新视角合成和超分辨率，这标志着其在生成式和分析式视觉应用中都具有强大的能力。
    - 早期用户测试显示出强大的文本渲染和语义理解能力，即使在复杂的、多模态提示词（例如带有环境标识的拟人化角色）中也能准确放置文本。然而，字体样式和贴花清晰度等细微差别可能会出现边缘情况。

- 对模型的评估图表/可视化存在显著批评，一些用户指出数据呈现质量问题——这在解读 Benchmark 性能或训练诊断（training diagnostics）时可能引发担忧。
- [**🚀 Meet Qwen-Image**](https://i.redd.it/7a463it8z0hf1.jpeg) ([Score: 460, Comments: 65](https://www.reddit.com/r/LocalLLaMA/comments/1mhhctd/meet_qwenimage/)): **所提供的图像引用自发布 Qwen-Image 的帖子，这是一个 20B 参数的 MMDiT 模型，用于文本生成图像（text-to-image generation），以其强大的原生文本渲染能力、双语支持（英文和中文）、完全集成的像素内文本生成以及跨图像风格的多功能性而著称。评论中分享了技术 Benchmark 和图像编辑示例，突显了其在文本渲染方面的竞争优势（英文媲美 GPT-4o，中文达到顶尖水平）和多样化的图像合成能力，但链接的图像本身在帖子或评论中并未得到明确的技术描述。其主要意义在于模型的技术进步，而非特定的图像内容。** 评论中的技术讨论强调了 Qwen-Image 的 Benchmark 对比，特别是文本渲染和图像编辑能力，虽然对某些使用场景存在怀疑或戏谑，但主要关注其在双语生成和布局处理方面的优势。
    - 分享了一张 [Benchmark 截图](https://preview.redd.it/a3o2wim001hf1.png?width=3036&format=png&auto=webp&s=fe8173646c7ea177041e2c110861a373b01356a6)，表明 Qwen-Image 在多模态或图像生成任务中与其它模型相比具有竞争性或领先性能，尽管具体的数值结果需要分析图像本身。
    - 提供了 [博客公告](https://qwenlm.github.io/blog/qwen-image/)、[Hugging Face 模型卡片](https://huggingface.co/Qwen/Qwen-Image)、[Model Scope 摘要](https://modelscope.cn/models/Qwen/Qwen-Image/summary)、[GitHub 仓库](https://github.com/QwenLM/Qwen-Image)、[技术报告 PDF](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/Qwen_Image.pdf) 以及两个在线 Demo 端点（[Wavespeed](https://wavespeed.ai/models/wavespeed-ai/qwen-image/text-to-image) 和 [Modelscope](https://modelscope.cn/aigc/imageGeneration?tab=advanced)）的链接，便于深入的技术探索、模型验证和复现。
    - 针对 Qwen-Image 在其基于 Diffusion 的流水线中实现良好的 *文本生成* 进行了技术对比，这一点值得注意，因为 ChatGPT-4o 的自回归模型（autoregressive model）在生成的图像中也表现出强大的文本处理能力——这表明 Qwen-Image 可能正在解决 Diffusion 模型中普遍存在的挑战。
- [**Qwen-Image is out**](https://v.redd.it/4077mfg081hf1) ([Score: 431, Comments: 35](https://www.reddit.com/r/LocalLLaMA/comments/1mhiqqn/qwenimage_is_out/)): **阿里巴巴发布了 Qwen-Image，这是一款在 Twitter 上宣布的新型多模态模型（[链接](https://x.com/Alibaba_Qwen/status/1952398250121756992)）。帖子声称 Qwen-Image 的表现优于 Flux Kontext，并达到了接近 "GPT-image" 的水平，暗示其在视觉理解和推理 Benchmark 方面具有技术对等性或优越性。原始帖子未提供深入的模型架构、Benchmark 或开源权重细节。** 评论中的讨论集中在对阿里巴巴发布免费资产的赞赏以及采用其 API 的意愿，但没有具体的技术辩论或对比细节。
    - 提到了 "20.B" 参数，这可能指模型的规模——200 亿参数——表明 Qwen-Image 是一个超大规模的多模态（视觉-语言）模型。这一规模使 Qwen 在容量方面与 GPT-4V 和 Gemini 等领先模型展开直接竞争，这对于跟踪开源视觉-语言模型扩展趋势（scaling trends）和能力的专业读者来说具有重要意义。
    - 链接的截图 (https://preview.redd.it/p49ocex2p2hf1.png?width=1328&format=png&auto=webp&s=84f30442e738efa1c07f79ce4508e89baadad3fb) 可能包含 Qwen-Image 界面或 Demo 输出的证据，这可以为技术读者提供有关该模型功能集、输出质量和 Prompt 处理能力的早期见解。

- [**Qwen image 20B 即将到来！**](https://www.reddit.com/r/LocalLLaMA/comments/1mhf0kl/qwen_image_20b_is_coming/) ([评分: 303, 评论: 60](https://www.reddit.com/r/LocalLLaMA/comments/1mhf0kl/qwen_image_20b_is_coming/)): **阿里巴巴 Qwen 团队正准备发布 Qwen Image 20B，这是一个拥有** `20B parameter` **的扩散图像生成模型，Hugging Face 的** `diffusers` **库即将对其提供支持（[相关 PR](https://github.com/huggingface/diffusers/pull/12055)）。该模型在 FP16 推理下可能需要** `40-44GB VRAM`**，突显了其与 LLMs 相比的资源密集性，因为 LLMs 在低精度（FP8）下性能下降相对平缓。** 评论者讨论了运行此类大型视觉模型的实际挑战，包括缺乏类似于 LM studio 的用户友好型软件生态系统，以及极高的 VRAM 需求，这使得像 RTX 5090 这样的最新消费级 GPU 在此规模下也无法满足 FP16 推理的需求。
    - 一位评论者强调了运行 20B 扩散图像模型的高内存需求，估计在 FP16 精度下需要 40-44GB 的 VRAM。他们指出，与 LLMs 不同，扩散模型即使在使用 FP8 等低精度格式时，性能也会显著下降，这强调了本地推理日益增长的硬件门槛。
    - 围绕在某些地区运行用于各种任务（聊天、代码、图像生成、OCR、RAG）的大型开源权重模型的实际限制展开了技术讨论，指出许多高性能模型因商业限制而无法获取，这与 Qwen3 235B 等模型在其他地区的开放可用性形成鲜明对比。
- [**今日发布 Qwen 新模型！！！**](https://i.redd.it/qemmgysvuzgf1.png) ([评分: 677, 评论: 103](https://www.reddit.com/r/LocalLLaMA/comments/1mhbpmo/new_qwen_models_today/)): **该帖子宣布即将发布新的 Qwen 模型，可能来自以开源 LLMs 闻名的阿里巴巴 Qwen 系列。评论者正在猜测可能发布的模型，如 'Qwen 3 VL' 和新的 'Qwen 3 Coder 14B'，并对可能推出新的多模态模型（即同时支持文本和视觉的模型）感到特别兴奋，这可能会增强该领域的开源替代方案。该图片可能是开发者的预热海报或宣传图，暗示多个模型即将揭晓。** 评论反映了对开源多模态模型的期待，认为生态系统中需要更多此类模型，并对编程专用模型表示好奇。此外，还有关于即将发布的版本的性能和规格的推测。
    - 多位用户对可能发布的 "Qwen 3 VL" 和 "Qwen 3 Coder 14b" 表现出浓厚兴趣，表明了对开源多模态模型和更大型代码专用变体的需求。对 Qwen3VL 的期待突显了在能够处理文本和视觉任务的免费、高性能模型方面的空白，并隐含地与 Llama-3 和 Open Flamingo 等最近的多模态成果进行了比较。

### 2. 中国主流 LLM 发布：Pangu Ultra 与 Hunyuan

- [**华为发布了 Pangu Ultra 的权重，这是一个 718B 参数的模型。**](https://ai.gitcode.com/ascend-tribe/openpangu-ultra-moe-718b-model/blob/main/README_EN.md) ([评分: 269, 评论: 50](https://www.reddit.com/r/LocalLLaMA/comments/1mhctvk/huawei_released_weights_of_pangu_ultraa_718b_model/)): **华为发布了 Pangu Ultra 的权重，这是一个拥有** `718B` **参数的 Mixture-of-Experts (MoE) 模型。值得注意的是，该模型完全在华为 Ascend NPU 上训练，未使用 Nvidia 硬件。该模型采用自定义许可证分发，要求署名（例如 "Powered by openPangu"），但在其他方面较为宽松；具体细节请参阅其 [许可证文件](https://ai.gitcode.com/ascend-tribe/openpangu-ultra-moe-718b-model/blob/main/LICENSE)。** 评论者强调了全栈国产大模型（硬件和软件）的意义，认为这标志着在摆脱 Nvidia/美国限制方面的技术独立，并指出其庞大的参数量非常引人注目。外部讨论中也出现了一些关于该模型实际性能和透明度的早期未经证实的主张。
    - Pangu Ultra 718B 模型因完全在华为 Ascend NPU 硬件上训练而备受关注，未动用任何 Nvidia GPU。这使其在软件和硬件栈上都成为了完全由中国开发的模型，彰显了中国在 AI 基础设施方面日益增强的自给自足能力。
    - 发布的权重使用自定义许可证，虽然相对宽松，但强制要求署名，包括在衍生产品中包含 "Powered by openPangu" 等声明，并承认 "openPangu is a trademark of Huawei Technologies Co., Ltd."。
    - 关于推理支持和托管仍存在疑问，特别是使用是否仅限于 Ascend 设备，还是有更广泛的部署选项。初始文档尚未澄清模型与其他硬件兼容性的技术细节。
- [**新款 Hunyuan Instruct 7B/4B/1.8B/0.5B 模型**](https://www.reddit.com/r/LocalLLaMA/comments/1mh3s7q/new_hunyuan_instruct_7b4b18b05b_models/) ([评分: 255, 评论: 51](https://www.reddit.com/r/LocalLLaMA/comments/1mh3s7q/new_hunyuan_instruct_7b4b18b05b_models/)): **腾讯发布了 Hunyuan-Instruct 系列开源语言模型，参数规模涵盖 0.5B、1.8B、4B 和 7B ([Hugging Face 链接](https://huggingface.co/tencent))，支持预训练和指令微调变体，以及兼容 llama.cpp 的 GGUF 格式。关键技术特性包括原生** `256K` **上下文窗口、用于高效推理的 Grouped Query Attention (GQA)、先进的量化技术，以及在 Agent 基准测试（BFCL-v3, τ-Bench, C3-Bench）中的强劲表现，训练过程继承自 Hunyuan-A13B。该模型家族定位为从边缘到高吞吐量环境的可扩展性，强调高效的内存利用和部署灵活性。** 评论强调了小规模模型（0.5B-4B）对低 VRAM 环境的价值，验证其宣称的长上下文能力的重要性，以及与 Qwen 相比在小模型选择上的多样性。
    - 评论者强调了 Hunyuan Instruct 发布多个小模型变体（7B, 4B, 1.8B, 0.5B）的技术意义，指出其在服务低 VRAM 用户方面与 Qwen 形成直接竞争。这使得这些模型特别适用于边缘部署、个人设备或硬件受限的研究人员。
    - 评估这些模型在长上下文场景下的表现引起了关注，因为上下文长度能力会极大地影响处理大输入窗口任务的可用性。发布更小的模型（如 0.5B）被认为具有重要意义，因为它们在内存和计算受限的环境中具有潜力，强调了对高效、轻量级架构的需求。

### 3. 元讨论与梗：Qwen 模型发布与社区反应

- [**Sam Altman 看着 Qwen 接连发布模型**](https://i.redd.it/g7t8cmgrv0hf1.jpeg) ([Score: 607, Comments: 22](https://www.reddit.com/r/LocalLLaMA/comments/1mhgu6t/sam_altman_watching_qwen_drop_model_after_model/)): **这张图片是一个梗图，引用了 OpenAI 的 CEO Sam Altman “注视着” Qwen（阿里巴巴的 AI 模型）快速发布一系列新的语言模型。帖子标题和技术评论的背景表明了对来自 Qwen 等中国模型日益激烈的竞争的担忧，以及对监管行动的推测。提出的一个关键技术担忧是，即将发布的模型可能会引入新颖的安全机制，这些机制可能受到专利和未来监管的约束，从而可能限制缺乏此类安全系统的本地、开放模型。** 评论反映了对可能通过监管强制执行专有模型“安全”功能的行业性举动的担忧，并推测这可能会以牺牲开放性和本地部署为代价，使大公司受益。此外，还讨论了美中 AI 模型创新之间的地缘政治动态。
    - 一位评论者推测，阿里巴巴即将推出的 Qwen 模型可能会引入一种旨在显著增强鲁棒性的新型模型级“安全”保护（除了严重的性能下降外，可能是“不可破解的”）。讨论表明，这可能成为专利和随后游说的基础，以使此类保护成为法律要求，这可能会限制缺乏类似机制的本地或开源模型的发布。
    - 讨论中暗含了关于中国公司（尤其是阿里巴巴）加速基础模型发布的战略影响。这可能会给西方公司带来压力，并可能推动围绕模型安全、监管和国际竞争的政策辩论。
- [**现在的 r/LocalLLaMA**](https://i.redd.it/f0xr7mshc0hf1.png) ([Score: 510, Comments: 65](https://www.reddit.com/r/LocalLLaMA/comments/1mhe1rl/rlocalllama_right_now/)): **这张图片是一个梗图，描绘了开源 LLM 的竞争格局，特别提到了传闻中 OpenAI 的开源模型，以及它在其他组织（如 Meta 和中国模型）强劲发布的背景下努力保持相关性的困境。讨论集中在对 OpenAI 对开源贡献的怀疑上，并将其与该领域其他更活跃的组织进行了对比。** 评论者争论 OpenAI 是否值得因一个尚未发布的模型而获得认可，一些人认为 Meta (Zuckerberg/LLaMA) 因其实实在在的发布而更值得称赞，而另一些人则注意到了围绕中国 LLM 的普遍热度。普遍观点认为，赞誉应与实际发布的、有用的模型相匹配。
    - 有人对“OpenAI 的开源模型”一词提出了批评，强调 OpenAI 并没有真正发布过真正的开源模型，并将其与 Meta (Zuckerberg) 进行了对比，后者发布的 LLaMA 等模型对开源社区产生了实质性影响。评论暗示技术赞誉应留给那些做出了具体模型和权重发布的组织。
    - 出现了一场关于区域开发模型的基准测试和比较的讨论，特别提到了“Qwen”作为更广泛的、具有技术雄心的中国 AI 模型的代表。建议是，“Qwen”脱颖而出不仅是个体现象，也是全球开源 LLM 质量和创新竞争日益加剧的象征。
- [**Horizon Beta 就是 OpenAI (另一个证据)**](https://www.reddit.com/r/LocalLLaMA/comments/1mh2v1h/horizon_beta_is_openai_another_evidence/) ([Score: 269, Comments: 58](https://www.reddit.com/r/LocalLLaMA/comments/1mh2v1h/horizon_beta_is_openai_another_evidence/)): **该帖子提供的证据表明 "Horizon Beta" 模型是基于 OpenAI 技术的，其依据是它将中文 Token '_天天啪' 处理为单个 Token，这是 OpenAI 分词器 (tokenizer) 的一个独特怪癖（在 Anthropic、Google Gemini 或 Qwen 中未观察到）。截图和测试确认了 Horizon Beta 在包含 'ketøy' 的提示词上的分词和翻译失败，其行为与 GPT-4o 一致，而非竞争模型。这种通过分词器行为进行识别的技术是基于 LLaMA 社区之前讨论的证据，该证据将类似的分词器 Bug 与源自 OpenAI 的模型联系起来。** 评论者注意到了该模型的实际效果和速度，尽管一些人表示除非该模型是开源或本地的，否则并不关心，另一个人则强调了令人印象深刻的生成能力（例如高质量的游戏演示）。评论中有人推测这可能与 OpenAI 的开源 (oss) 努力有关，但共识指向其 OpenAI 起源。

- 几位评论者指出，Horizon Beta 模型似乎针对创意写作而非多模态或重度代码使用进行了优化，并引用了 Sam Altman 的公开暗示，表明其重点在于故事生成或文本创意。人们对 '-beta' 和 '-alpha' 版本进行了比较，观察到 '-beta' 执行了明显更多的内容过滤和审查，这表明在开发过程中进行了刻意的审核调整。
- 一些用户推测 Horizon Beta 是否代表 OpenAI 的开源本地 LLM 发布（可能与其最近开源的模型有关），但其他评论者表示反对，强调了在线/专有部署与备受期待的本地开放模型之间的区别。共识是，除非模型可以在本地/离线运行，否则对于某些用例的技术价值有限。
- 性能评估显示该模型速度快，能提供高质量的创意写作，并能生成令人印象深刻的复古风格游戏资产（例如详细的横向卷轴演示），但被描述为“并非惊天动地”——暗示其能力扎实，但在生成能力上并非突破性的飞跃。

## 较少技术性的 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Qwen-Image 模型发布与基准测试

- [**Qwen-Image 已发布**](https://huggingface.co/Qwen/Qwen-Image) ([Score: 407, Comments: 177](https://www.reddit.com/r/StableDiffusion/comments/1mhh7nr/qwenimage_has_been_released/)): **Alibaba 发布了 Qwen-Image，这是一款先进的视觉语言模型，据称支持图像编辑、多模态理解和图生文等任务。分享的技术预览包括展示其编辑能力的界面截图，表明其功能类似于 Kontext 等模型。Qwen-Image 似乎是更广泛的 Qwen 系列模型的一部分，预计未来将推出适用于资源受限部署的量化版本。** 专家评论指出，对类似于 Kontext 的图像编辑功能感到兴奋，并对即将推出的量化变体表现出高度兴趣，这标志着对资源效率和更广泛可访问性的期望。
    - 讨论涉及模型大小，评论者指出 Qwen-Image 的权重据报道为 `40GB`，这使得显存低于 `48GB` VRAM 的消费级 GPU 用户（如 12GB VRAM 的 RTX 3060）无法使用。这突显了本地推理对硬件的显著要求。
    - 一位技术用户对量化版本表示感兴趣（“等不及量化版了”），表明当前发布的可能是全精度版本，可访问性较低，而量化可以降低要求，使其在消费级硬件上更具可用性。
    - Qwen-Image 在编辑能力方面与 Kontext 进行了比较，暗示了多模态或图像编辑功能。这含蓄地提高了对功能集以及与其他先进多模态模型技术对等的期望。
- [**Qwen image 即将到来！**](https://www.reddit.com/r/StableDiffusion/comments/1mhe9jb/qwen_image_is_coming/) ([Score: 141, Comments: 66](https://www.reddit.com/r/StableDiffusion/comments/1mhe9jb/qwen_image_is_coming/)): **Alibaba 的 Qwen 团队正准备发布 Qwen Image 20B 模型，这是一个拥有** `20B` **参数的图像生成模型，定位为备受推崇的 Wan 视频模型的图像版。该模型的早期支持已集成到 HuggingFace 的 [Diffusers 库](https://github.com/huggingface/diffusers/pull/12055)中，预示着即将公开发布。Qwen 图像模型的发布预计将提升 state-of-the-art (SOTA) 视觉能力，正如 Qwen 之前的 LLM 和多模态模型发布一样。** 评论者强调了开源视觉模型的快速进展和扩展，指出了 Qwen 频繁取得 SOTA 成就、大型模型硬件可访问性的重要性（特别是 20B 参数规模和 GGUF 量化），并表达了对中国制造商以竞争性价格提供更多 VRAM 的期待。
    - 新的 Qwen 图像模型已经获得了 Hugging Face Diffusers 库的支持（[提交链接](https://github.com/huggingface/diffusers/pull/12055)），表明即将发布，并建议快速集成到现有的生成式图像模型流水线中。
    - 技术讨论比较了预期的模型参数大小：Qwen 即将推出的图像模型预计将比 Hidream（17B 参数）模型更大，并可能达到 '20 billion params'，且有望支持 GGUF 量化，以支持高效的本地使用。

- 讨论认可了 Qwen 在 LLM 和视觉模型中持续保持的 SOTA（State-of-the-art）性能，特别提到了 Wan（视频模型）和 VLM 令人印象深刻的能力，社区对随着模型规模增加而带来的竞争性 VRAM 利用率充满期待。
- [**Qwen Image 在图像编辑方面甚至优于 Flux Kontext Pro。**](https://www.reddit.com/gallery/1mhikh2) ([Score: 244, Comments: 51](https://www.reddit.com/r/StableDiffusion/comments/1mhikh2/qwen_image_is_even_better_than_flux_kontext_pro/)): **阿里巴巴的 Qwen-Image 模型在图像生成和编辑任务中均展示了 SOTA 性能，根据其官方博客（https://qwenlm.github.io/blog/qwen-image/）总结的基准测试，其表现超越了 Flux Kontext Pro 等竞争系统以及其他开源/闭源模型。然而，先进的编辑模型尚未公开发布，阿里巴巴指出未来可能会提供。该模型的运行要求很高——据报道当前版本需要大量的 GPU 显存（80GB+），这使得大多数研究人员目前无法对非量化模型进行本地/私有实验。** 评论者强调了由于高资源需求而导致的易用性担忧，并期待社区驱动的量化版本能尽快推出。人们对模型的泛化能力和真实世界保真度持怀疑态度——尤其是量化后——一些人注意到了其特征性的“卡通化”输出风格。
    - Qwen Image 目前尚未发布的模型因其图像编辑能力而备受关注，但其易用性受到质疑：用户指出，如果需要 80GB GPU，大多数人无法在本地运行；然而，人们预期很快会出现量化版本（更小、显存要求更低的模型），从而可能提高可用性。
    - 在直接的视觉对比中，技术用户承认虽然 Qwen Image 展示了令人印象深刻的结果，特别是作为一个开源产品，但 Imagen 4 Ultra 在照片级真实感方面仍被认为更胜一筹，尤其是在具有挑战性的构图细节（如浅景深、光影、虚化效果和电影级调色）方面。
    - 技术用户对开源模型的常见问题持谨慎态度——量化通常会导致性能下降，且“卡通化”的输出往往会被放大。尽管如此，开源竞争对手日益增长的质量被视为一种积极的发展，推动了该领域的快速迭代和改进。
- [**警告：在最近的 Qwen-Image NF4 中检测到 pickle 病毒**](https://www.reddit.com/r/StableDiffusion/comments/1mhkmsa/warning_pickle_virus_detected_in_recent_qwenimage/) ([Score: 146, Comments: 76](https://www.reddit.com/r/StableDiffusion/comments/1mhkmsa/warning_pickle_virus_detected_in_recent_qwenimage/)): **一个 HuggingFace 模型仓库（[lrzjason/qwen_image_nf4](https://huggingface.co/lrzjason/qwen_image_nf4)）因包含潜在的“pickle 病毒”被标记，并发出避免下载的警告。尽管该仓库之前发布过多个模型，但涉事文件是一个** `.safetensors` **文件（旨在避免 Pickle 序列化的代码执行漏洞）；该仓库随后已被关闭。** 热门评论表示怀疑，指出 `.safetensors` 文件理应免受 Pickle 漏洞影响，并建议该问题可能是误标或误报。一些人还指出上传者之前有发布过似乎信誉良好的模型的历史，主张保持谨慎但不必过度惊慌。
    - 讨论涉及了 `.safetensors` 文件与 `.pkl` (pickle) 模型文件的相对安全性：`.safetensors` 格式旨在减轻 pickle 序列化固有的风险，特别是任意代码执行，使得 `.safetensors` 被认为在模型分发中本质上更安全。
    - 一条评论指出，被标记的文件被错误地识别为基于 pickle 的病毒：它实际上是一个 `.safetensors` 文件，不支持 Python 的 pickle 机制，因此无法执行任意 Python 代码。这强调了在安全问题中区分文件格式的重要性。
    - 另一个技术相关的点是 HuggingFace 上的用户历史：上传者拥有良好的记录，这表明误报的可能性较大。尽管如此，建议用户在文件安全性确认之前保持谨慎。

### 2. Claude 4.1 与 Opus 下一代模型发布热潮

- [**看起来 Claude 4.1 Opus 也即将推出**](https://i.redd.it/a2kwuoauf0hf1.png) ([Score: 266, Comments: 58](https://www.reddit.com/r/singularity/comments/1mhehrb/looks_like_claude_41_opus_is_also_coming_soon/)): **图片似乎显示了预发布或公告暗示，Claude 4.1 Opus（可能是 Anthropic 旗舰 LLM 的更新版本）即将推出。评论者推测了版本访问限制（“在达到每周限制前只能发送 4.1 条消息”），期待 Sonnet 4.1 的对应版本，并注意到 Anthropic 最近的用户体验调查和 A/B testing，这预示着更广泛的平台和模型更新。** 讨论集中在预期的模型改进以及访问/定价的变化，一些用户批评使用限制的频率，而另一些用户则关注最近的用户体验研究，将其视为即将发生重大变化的证据。
    - 一位用户指出，Anthropic 已开始要求使用 Claude Code 的用户提供评价，这表明针对新功能或用户体验变化的 A/B testing 正在进行中，这可能预示着未来 Claude 4.1 版本的改进或调整。
    - 提到了关于 Claude 的 Opus 层级的担忧，指出它被认为“过于审查且价格昂贵”，反映了社区对 Anthropic 溢价模型的定价和内容审查政策的持续争论。
    - 人们对 Claude Sonnet 4.1 充满期待，这表明用户对不同 Claude 模型变体（不仅仅是旗舰版 Opus）的功能或性能对等性感兴趣。
- [**Opus 4.1 正在路上？**](https://i.redd.it/a2kwuoauf0hf1.png) ([Score: 162, Comments: 51](https://www.reddit.com/r/ClaudeAI/comments/1mhf8mo/opus_41_on_the_way/)): **该帖子推测 OpenAI 传闻中的“Opus 4.1”模型即将发布，正如图片所暗示的（似乎显示了后端或仪表板证据），并结合了潜在模型发布的时间点——可能是为了抢在“GPT-5”的热度之前。评论者正在讨论对新模型限制或定价的预期（与 Sonnet 4 进行比较），并将性能下降的抱怨与模型重新训练期联系起来，暗示由于模型发布前的高 GPU 利用率可能导致资源受限。[点击此处查看图片。](https://i.redd.it/a2kwuoauf0hf1.png)** 评论者普遍认为，新模型的发布通常伴随着观察到的性能下降，并推测这是由于 GPU 资源被投入到训练中。对于 Opus 4.1 是否会引入更好的访问限制也存在不确定性，这将影响实际使用。
    - 一个技术假设认为，即将推出的“Opus 4.1”可能会利用 Opus 4 的新 Checkpoint，然后进行剪枝以提高效率——可能在原始模型大小的约 60% 情况下实现更好的性能。这将使 Anthropic 能够在降低计算成本的同时维持定价，以应对增加的上下文窗口需求。
    - 一位用户注意到当前模型质量有明显的下降，报告称性能感觉退化到了“3.7”的水平，反映了可能由重大训练活动或即将到来的模型刷新引起的持续波动。
    - 关于 Opus 4.1 是否会继承 Sonnet 4 的使用限制存在推测性讨论，这意味着技术和基础设施限制（如请求节流或配额执行）可能会限制最终用户的实际改进，无论模型如何进步。
- [**这绝对是我见过的最疯狂的 One Shot 之一 - Claude Opus 4**](https://v.redd.it/i9aehe15axgf1) ([Score: 146, Comments: 20](https://www.reddit.com/r/ClaudeAI/comments/1mh2ybk/this_has_to_be_one_of_the_craziest_one_shots_ive/)): **一位用户声称，Claude Opus 4 在单次推理（“One Shot”）中，根据以下提示词生成了一个使用 ThreeJS 的自包含单页 HTML 无人机模拟器：“创建一个自主无人机模拟器（无人机自动飞行，等轴测上帝视角，可选交互。具有自定义环境（可选创意），使用 ThreeJS，输出单页自包含 HTML）。”技术意义在于 Opus 4 可以编写实质性的、非琐碎的应用程序逻辑，集成了 3D 渲染和基础模拟，且全部在一次连续输出中完成。** 热门评论对可复现性表示怀疑（询问提示词），建议尝试类似的挑战性任务（例如僵尸爆发模拟器），并对该模型显而易见的输出质量表示惊讶。关于这究竟是普遍可实现的还是个别提示词/模型交互的特例，存在隐含的争论。

- 有评论者建议使用 horizon-beta 作为替代方案，声称它不仅能对相同的 prompt 进行 one-shot，还能生成更高级的功能，如多种摄像机模式、环境效果（水、风纹理）、螺旋桨动画、任务系统、动态天气，甚至还有等高线地形，这暗示了与所引用的 Claude Opus 4 结果相比，其输出的复杂性和完整性存在显著差异。
- 一位评论者询问了用于实现该结果的 prompt 具体细节，并对能否复制如此先进的 one-shot 结果表示怀疑，这突显了人们对可复现性以及 prompt engineering 对大模型输出影响的持续关注。
- 另一个技术咨询涉及输出 latency，一位用户询问了 Claude Opus 4 完成 one-shot 所花费的时间，这反映了人们不仅对内容生成质量感兴趣，还对复杂 prompt 的 inference 速度和 throughput 感兴趣。

### 3. 即将到来的 GPT-5 发布信号与 OpenAI 公告

- [**看起来周四将是 GPT-5 发布的日子（至少根据一直很可靠的 Jimmy 的说法）**](https://i.redd.it/2d0vdv8spzgf1.png) ([Score: 185, Comments: 66](https://www.reddit.com/r/singularity/comments/1mhb58g/looks_like_thursday_will_be_the_day_for_gpt5_at/))：**该帖子的图片 (https://i.redd.it/2d0vdv8spzgf1.png) 似乎是一个截图或视觉参考，暗示 GPT-5 的发布日期可能是周四，这归功于绰号为 'Jimmy' 的泄密者。标题和热门评论中的讨论提到了此人此前在预测 OpenAI 事件方面的可靠性，但也指出了之前的失准，例如关于 AGI 的说法。背景信息还提到，这与流传的传闻以及 Sam Altman 最近展示 GPT-5 的暗示相吻合，支持了关于即将发布的推测。** 评论者质疑 'Jimmy' 作为可靠来源的往绩，并对传闻中不断变化的发布日期表示怀疑（例如，“先是周一，现在又是周四？”）。还有一些戏谑的言论，降低了对演示实用性的预期。
    - 讨论集中在 Jimmy 泄密的可靠性上，提到了他过去声称 OpenAI 在 2023 年内部实现了 AGI 的说法，但这尚未得到证实。用户注意到，最近关于周四发布 GPT-5 的推测与独立传闻以及 Sam Altman 在 Twitter 上的公开暗示相吻合，使得目前的传闻比之前缺乏证据的说法更具分量。
- [**OpenAI ChatGPT 副总裁：“接下来是重要的一周”**](https://i.redd.it/bja36ao2w0hf1.jpeg) ([Score: 226, Comments: 30](https://www.reddit.com/r/OpenAI/comments/1mhgwwg/openai_vp_of_chatgpt_big_week_ahead/))：**该帖子引用了 OpenAI ChatGPT 副总裁的一份声明，预示着即将有重大公告或活动，标题为“接下来是重要的一周”。由于分析失败，图片本身未被详细描述，但上下文表明它可能是与 ChatGPT 预期更新或发布相关的宣传或预热内容。评论中的讨论点强调了大规模的使用指标（ChatGPT 和 Gemini 等生成式 AI 平台每月用户超过 10 亿），以及 AI 社区中用于制造热度的营销手段。** 评论者争论 OpenAI 营销策略的有效性和诚意，对炒作热度与即将发布的公告实质内容之间的差距表示怀疑，并将其与 ChatGPT 和 Gemini 的广泛采用及现实世界使用数据进行了对比。
    - 一位用户注意到公众对 AI 的怀疑与实际使用情况之间的对比，引用数据称 ChatGPT 和 Gemini 每周或每月共同服务于“远超 10 亿人”。这突显了尽管存在对主流使用的批评或怀疑，这些 AI 工具仍具有极高的采用率。
- [**GPT-5 彩蛋**](https://www.reddit.com/r/singularity/comments/1mhkahr/gpt5_easter_egg/) ([Score: 125, Comments: 71](https://www.reddit.com/r/singularity/comments/1mhkahr/gpt5_easter_egg/))：**一位 Reddit 用户指出一个潜在的彩蛋：OpenAI 员工 Boris Power 在 PDT 时间上午 8:05 精准发布了关于 GPT-5 的推文，推测 8/5（8 月 5 日）是 GPT-5 预定的发布日期。该帖子引用了一天有 1440 分钟来强调该时间的刻意性（见 [推文](https://x.com/BorisMPower/status/1952385313546146238)），但未提供关于 GPT-5 能力、架构或 Benchmarks 的直接技术细节或确认。** 具有技术思维的评论者持怀疑态度，挑战其统计显著性，并认为这种关联性很弱，建议将此类推测用于预测市场，而不是将其视为有意义的证据。
    - 一位评论者指出，像 GPT-5 这样的大型模型发布通常会通过官方活动提前宣布，而不是突然揭晓或通过隐藏信号透露，这意味着重大的发布会伴随着 OpenAI 等组织的实质性营销和沟通努力。这得到了以往重大 AI 发布历史模式的支持。
    - 另一位具有技术思维的回复挑战了对数字巧合（一天 1440 分钟）作为发布日期信号的解释，并指出考虑到一系列可能的日期（8/5-8/31），日期数字中的模式匹配可能会产生误导，并不是一种统计学上严谨的预测方法。

---

# AI Discord 摘要回顾

> 由 Gemini 2.5 Pro Exp 生成的摘要之摘要
> 

**主题 1. 新模型前沿：GPT-5 炒作、Horizon Alpha 的首次亮相以及通义千问 (Qwen) 的崛起**

- [**工程师们正严阵以待即将发布的 GPT-5**](https://discord.com/channels/974519864045756446/998381918976479273/1400554785911865466)：关于 **GPT-5** 即将发布的猜测甚嚣尘上，由于扩展限制以及 **Chain of Thought (CoT)** 收益递减（一些开发者称其为 *“彻底的死胡同”*），有人将其称为 *“恐慌式发布”*。虽然来自 **OpenAI** Discord 的传闻暗示由于无法超越 **Grok4** 而导致延迟，但也有人报告在 API 中短暂发现了 **GPT-5** 的踪迹（[来源](https://x.com/chetaslua/status/1951301385292493259)），并预见这将是一个[结合了多种产品](https://link.to/openais-next-foundational-model)的统一全模态模型。
- [**神秘的 Horizon Alpha 模型表现超越付费 LLM**](https://discord.com/channels/1091220969173028894/1094454198688546826/1400553644956323921)：一款名为 **Horizon Alpha** 的新模型令开发者印象深刻，它在 **OpenRouter** 上的表现优于付费 LLM，能够[在自定义编程语言中提供完美的一次性生成代码](https://openrouter.ai/)，并展示了卓越的 Shell 使用能力。**Nous Research AI** 上的推测认为，根据一则 [推文](https://x.com/apples_jimmy/status/1951180954208444758) 和 [Reddit 帖子](https://www.rxddit.com/r/LocalLLaMA/comments/1mepeqh/the_openai_open_weight_model_might_be_120b/)，它可能是来自 **OpenAI** 的 **120B MoE** 或 **20B** 模型，而其他人则认为 *它可能一直是某种我们没想到的极其诡异的东西，比如 Codex-2*。
- [**Qwen3-Coder 打破速度记录并带来挑战**](https://discord.com/channels/1027685395649015980/1027688115592237117/1400899899402489856)：新的 **Qwen3-Coder** 模型现已在 **Windsurf** 中上线，运行在位于美国的服务器上，速度达到了惊人的 **2000 tokens/sec**。与此同时，**Unsloth AI** Discord 的开发者正在辩论最佳量化方法，**Q4_K_M GGUF** 在 **Ollama** 中表现缓慢，而其他人则在 **3090** 上通过 **vLLM** 以 **40k** 上下文运行 **Qwen3-30b-a3b** 模型；然而，**LM Studio** 的用户在加载模型时遇到了 *Cannot read properties of null (reading '1')* 的错误。

**主题 2. 技术前线：量化困境、RAG 辩论与 API 争端**

- [**量化难题困扰新模型**](https://discord.com/channels/1053877538025386074/1149866623109439599/1400597993442513128)：工程师们正致力于解决新模型的量化问题，因为 **Hermes-3 数据集** 中意外出现的拒绝回答使 *imatrix* 计算变得复杂，促使人们对[该数据集进行更深入的调查](https://huggingface.co/datasets/NousResearch/Hermes-3)。传闻中 **OpenAI MoE** 模型的泄露配置显示其隐藏层大小（hidden size）为 **2880**，这将导致无法使用 K 或 I 量化，这一限制在 **SmolLM** 和 **Qwen2.5** 中也同样存在。
- [**开发者质疑长上下文窗口，精进 RAG**](https://discord.com/channels/974519864045756446/998381918976479273/1400554785911865466)：一场关于[长上下文窗口是否被高估](https://nealgoogs.website/)的辩论正在酝酿，一些人发现 **Claude** 和 **ChatGPT** 在处理遗留代码库时比 **Gemini** 的 1M 窗口更好用，而另一些人则认为长上下文对于 **Agent** 应用至关重要，以便 *自动记住并织入久远之前的细节*。为了改进检索，**Yannick Kilcher** Discord 的工程师们正在 **RAG** 系统中使用[查询扩展技术](https://www.promptingguide.ai/techniques/query_expansion)，通过单个查询生成多个问题。
- [**Anthropic 封禁 OpenAI 访问 Claude API 的权限**](https://discord.com/channels/822583790773862470/1075282825051385876/1400565376567480373)：在一项重大的竞争举措中，**Anthropic** 以违反服务条款为由，撤销了 **OpenAI** 对其模型（包括 **Claude**）的 API 访问权限。**OpenAI** 表示失望，并强调其 API 仍对 **Anthropic** 开放，这引发了社区关于竞争升级以及模型训练界限模糊的热议。

**主题 3. Agent 的兴起与专业化工具**

- [**编程 Agent 争夺开发者主导权**](https://discord.com/channels/1131200896827654144/1131200896827654149/1400559583528747048)：**Aider** 因其高效性继续赢得赞誉，一位用户声称使用 **DeepSeek** 后，“仅需 2 美元即可在一天内完成一周的编程工作”。在融资领域，开源 AI 编程 Agent **Cline** 在由 **Emergence Capital** 和 **Pace Capital** 领投的融资轮中获得了 **3200 万美元**，旨在为其 **270 万** 开发者提供透明的工具。
- [**具身智能与创意工具突破边界**](https://discord.com/channels/1053877538025386074/1149866623109439599/1400597993442513128)：售价 **5,900 美元** 并配备开放 SDK 的 **Unitree R1 基础机器人模型** 正在推动具身智能开发的民主化，正如[这段 YouTube 视频](https://www.youtube.com/watch?v=ljo7TjOqRzs)所示。在创意 AI 领域，[**TheCinema AI**](https://thecinema.ai/) 研究项目正在攻克生成连贯电影场景的挑战，详见其 [arXiv 论文](https://arxiv.org/html/2507.18634v1)；同时，一位训练 **VITS** 模型的开发者发现，该模型可以学习到像逗号处呼吸这样的细微差别。
- [**新协议与框架扩展生态系统**](https://discord.com/channels/1312302100125843476/1312302100125843479/1400559945786589275)：Model Context Protocol 的新支付层 **PayMCP** 正在开发中，并为早期采用者提供了 [Python](https://github.com/blustAI/paymcp) 和 [TypeScript](https://github.com/blustAI/paymcp-ts) 实现版本。在 GPU 领域，[picocuda](https://github.com/j4orz/picocuda) 编译器的开发正在推进，计划遵循 [GPUCC 论文](https://dl.acm.org/doi/pdf/10.1145/2854038.2854041) 的路径，最终实现编译 **Karpathy 的 llm.c**。

**Theme 4. 巨头之争：Gemini 的挫折与 Kimi 的崛起**

- [**Google 的 “Deepthink” 计划因高价低限额遭嘲讽**](https://discord.com/channels/1340554757349179412/1340554757827461211/1400554342616010903)：面向 Ultra 会员发布的 **Gemini 2.5 Deepthink** 在 Discord 各大社区遭到嘲笑，原因是其 **250 美元/月** 的价格标签和 **每天 10 次查询的限制**。**LMArena** 和 **Moonshot AI** 的成员称其为“骗局”，并认为其“非常滑稽且非常卑劣”，视其为在 GPT-5 发布前仓促推出的产物。
- [**Kimi K2 Turbo 发布：速度提升 4 倍并提供折扣**](https://discord.com/channels/1369594130807787570/1371757097246785536/1400719197838770217)：**Moonshot AI** 宣布推出 **Kimi K2 Turbo**，这是其模型的更快版本，拥有 **4 倍的速度提升**（达到 **40 tokens/sec**），可通过官方 API [platform.moonshot.ai](http://platform.moonshot.ai/) 获取。新模型在 9 月 1 日前提供 **50% 的 token 折扣**，这使得 **Kimi K2** 成为首个让用户觉得可以替代 **Claude** 的模型，并导致他们放弃了 **Gemini 2.5 Pro**。
- [**Gemini 模型表现出故障与偏见**](https://discord.com/channels/1340554757349179412/1340554757827461211/1400554342616010903)：**LMArena** 的用户注意到 **Gemini** 模型表现出重复、不稳定的行为。更系统地，**Yannick Kilcher** 和 **Eleuther** Discord 社区的开发者观察到，**Gemini-2.5-flash** 在评估中始终将 **Gemma** 模型的排名排得比其他模型更高，这引发了对其评估能力中潜在“家族偏见”的担忧。

**Theme 5. 用户体验战场：冻结 Bug、API 错误与免费优惠**

- [**Cursor 冻结 Bug 与定价争议令用户沮丧**](https://discord.com/channels/1074847526655643750/1074847527708393565/1400555426764034119)：一个持续存在的 **Cursor 冻结 Bug** 导致机器在聊天使用一小时后每隔 **30-60 秒** 就会冻结，团队正引导用户在 [Cursor 论坛](https://forum.cursor.com/c/bug-report/6) 上进行报告。与此同时，不断上涨的成本也引发了争论，一名用户报告在 **3 个月内花费了 600 美元**，而另一名用户则更倾向于 **Claude 的 200 美元计划**。
- [**平台 API 遭遇错误与停机**](https://discord.com/channels/1091220969173028894/1094454198688546826/1400553644956323921)：**OpenRouter** 用户正受到 API 问题的困扰，包括内部错误、超时以及 **Deepseek v3 免费模型** 的空响应，该模型被描述为“完全超载”。同时，**iOS** 上的 **Perplexity Pro** 订阅者报告称，图像生成功能无法整合附件图片，且 `search_domain_filter` 运行不正常。
- [**印度 Airtel 用户获赠免费 Perplexity Pro**](https://discord.com/channels/1047197230748151888/1047649527299055688/1400553929611280499)：在一项重大的促销活动中，拥有超过 **3 亿用户** 基数的印度 **Airtel** 订阅者将免费获得为期 **12 个月** 的 **Perplexity Pro**。此促销活动仅限位于印度的 Airtel 订阅者，为该平台提供了一个重要的用户获取渠道。

---

# Discord：高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Comet Browser 邀请码陆续发放**：**Perplexity** 正在缓慢分发 **Comet Browser** 邀请码，并优先考虑 **Pro 用户**。
   - 用户报告等待时间不等，并建议 Pro 用户可以分享最多 **2 个邀请码** 以加速进程。
- **Perplexity Pro 图像生成在 iOS 上失败**：用户报告 **iOS 上的 Perplexity Pro** 在图像生成过程中无法整合上传的图片，导致问题反复出现。
   - 即使在开启新聊天后，模型也只是总结请求而无法根据附件生成图像。
- **印度 Airtel 订阅者获赠免费 Perplexity Pro**：印度 **Airtel** 订阅者（超过 **3 亿人**）将免费获得为期 **12 个月** 的 **Perplexity Pro**。
   - 该促销活动专供位于印度的 Airtel 订阅者。
- **GPT-5 发布日期：仍笼罩在迷雾中**：关于 **GPT-5** 发布日期的猜测不断，对于它是完整版本还是更小、更专注的模型存在分歧。
   - 一名用户声称曾在 **API 中短暂看到 GPT-5** ([来源](https://x.com/chetaslua/status/1951301385292493259))，但它很快被移除，引发了进一步的猜测。
- **搜索域名过滤器失效**：一位 **Perplexity Pro** 订阅者报告称，尽管该功能并非处于 Beta 阶段，但 **search_domain_filter** 的运行效果未达预期。
   - 另一名成员请求获取该用户的请求副本，以便进一步调查和协助。

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **GPT-5：恐慌式发布还是温和改进？**：成员们正在推测 **GPT-5** 是否会因为 **OpenAI** 在 Scaling 方面的局限性以及 **Chain of Thought (CoT)** 收益递减而成为一次“恐慌式发布（panic drop）”。
   - 有观点认为 **CoT** 是一个“彻底的死胡同”，并建议采用模型向量输出的直接网络反馈，而不是使用 Token 进行思考。
- **Qwen3 测试量化极限**：讨论围绕 **Qwen3 Coder 30B** 的最佳量化方案展开，有报告称 **Q4_K_M gguf** 在 **Ollama** 中运行缓慢，而其他人则更倾向于使用 **UD q3 XL** 以节省 VRAM。
   - 一名成员在 **3090** 上通过 **vllm** 全天候运行 4 月份的 **Qwen3-30b-a3b** 模型（**40k** 上下文），并等待 Coder 模型的 4-bit AWQ 版本。
- **Unsloth 现已支持 GSPO**：在 Qwen 提出 **GSPO** 作为 **GRPO** 的更新版本后，成员们澄清 **GSPO** 已经在 **Unsloth** 中可用，它是一个封装器，将自动支持 **TRL** 的更新。
   - 尽管 **GSPO** 效率略高，但成员们并未注意到性能有任何显著提升。
- **VITS 学会了呼吸**：一位彻夜训练 **VITS checkpoint** 的成员分享道，**模型质量取决于 Epochs 和数据集质量**，且 **VITS 在说话人解耦（speaker disentanglement）方面表现出色**。
   - 此外，他们发现 **VITS 将原始音频编码到 Latent Space** 以实现逼真的重现，并能通过标注学习到逗号处的呼吸等细微差别，但在 iOS 上遇到了内存问题。
- **动态量化迎来 Quant Clone**：一位成员创建了 [一个小应用程序](https://github.com/electroglyph/quant_clone)，用于以与 Unsloth 动态量化相同的方式对微调模型进行量化，希望在自己的微调模型上复制这一过程。
   - 一名用户报告其 **Gemini** 微调模型的拒绝率很高，并发现 **Gemini** 在这方面“相当令人讨厌”。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **竞技场增强功能旨在提供帮助**：成员们建议添加 **Search, Image, Video, 和 Webdev Arena** 的按钮以提高曝光度，并建议在排行榜上添加工具提示，解释 **Rank, CI, 和 Elo** 是如何确定的，并分享了一张 [概念图](https://cdn.discordapp.com/attachments/1340554757827461211/1400554342167089222/uzzzSHh.png)。
   - 目标是协助用户浏览平台并理解排名指标。
- **数据隐私担忧：个人信息风险**：一名用户对在发布的 Prompt 中意外包含 **个人信息** 表示担忧，并询问是否可以删除 Prompt。
   - 一名成员回应称，此类案例应私信（DM）给他们以便上报，并确认已 [与团队分享了这些担忧](https://www.deepcogito.com/research/cogito-v2-preview)。
- **Gemini 的生成出现故障**：一些成员注意到 **Gemini** 表现出重复行为，而另一位成员则询问 **Gemini 2.5 Flash** 是否修复了该问题；还有一名用户注意到视频限制从 **10 个降至 8 个**，敦促其他人尽快使用视频生成竞技场。
   - 社区的情绪在经历故障和性能稳定之间产生分歧。
- **DeepThink 首秀令人失望？**：随着面向 Ultra 会员的 **Gemini 2.5 Deepthink** 发布，成员们在看到 **10 RPD（每日请求限制）** 后怀疑其是否值得。
   - 成员们称其为“骗局”和“白昼抢劫”，认为这只是因为 **GPT-5** 即将发布而推出的仓促版本。
- **Veo 3 视觉大捷**：**Veo 3 Fast & Veo 3** 已在 [Video Arena](https://discord.com/channels/your_server_id/1397655695150682194) 中上线，具备全新的 **图生视频（Image-to-Video）及音频功能**。
   - 社区现在可以在 video-arena 频道中使用新的 `/image-to-video` 命令从图像创建视频，并对最佳视频进行投票。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Vibe Coding 激发对 GitHub 的需求**：一位成员询问后台 **Agent** 是否必须使用 **GitHub**，并在附图中惊叹 *“这东西太酷了”*，引发了大家对 **vibe coding** 配置的好奇。
   - 另一位在 **Prompt** 上花费了 **$40** 的用户寻求优化其 **Cursor** 设置的建议，反映出用户对高效配置的普遍兴趣。
- **Cursor 卡死 Bug 令人沮丧**：一位用户报告称，在使用聊天功能一小时后，机器每隔 **30-60 秒** 就会频繁卡死，这表明存在一个持续的 **Cursor 卡死 Bug**。
   - 一名 **Cursor** 团队成员建议将该问题发布在 [Cursor 论坛](https://forum.cursor.com/c/bug-report/6)上，并强调了用于 Bug 报告和协助的官方渠道。
- **模型支出与 Claude Pro 的对比**：用户讨论了 **Cursor** 与 **Claude Pro** 的定价，一位用户表示他们更倾向于最便宜的方案和最好的模型，看好 Claude 的 **$200** 方案。
   - 另一位用户对不断攀升的成本提出了警告，称其在 3 个月内花费了 **$600**，强调了成本管理的必要性。
- **Horizon Alpha 体验评价两极分化**：一位用户描述其对 **Horizon-Alpha** 的个人体验 *“有点差强人意”*，暗示用户对这一新功能的反应不一。
   - 相反，另一位用户赞叹 *“Cursor 是我见过的最好的应用”*，突显了用户体验的主观性。
- **用户请求 Cursor 推荐计划**：成员们询问了 **Cursor** 的推荐计划，一位用户声称自己 *“目前已经在 Discord 里拉了至少 200 多人，笑死”*，表明社区驱动的采用率非常显著。
   - 社区分享了 [Cursor 大使计划 (Cursor Ambassador program)](https://cursor.com/ambassador) 的链接，为奖励社区贡献提供了另一种途径。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Function Calling API 优于 XML 变通方案**：**Function Calling API** 相比结构化 XML 具有**内在价值**，后者通常在 **Qwen** 等模型不支持原生 **Tool Calling** 时作为变通方案使用。
   - 内联工具调用（Inline tool calls）最大限度地提高了 **Qwen** 等编程模型的互操作性，即使存在细微的效率损失。
- **扎克伯格的 AI 引发生物武器担忧**：**Mark Zuckerberg** 的 AI 超级智能计划引发了对潜在生物武器制造的担忧，一名成员警告不要向公众发布超级智能。
   - 成员们还担心，*“通过虚假用户和精心设计的语言来控制思想”* 可能比生物武器更危险。
- **GPT-5 面临延迟，Grok4 夺冠？**：传闻称 **GPT-5** 的延迟是因为无法超越 **Grok4**，但 [OpenAI 计划将多个产品整合到 GPT-5 中](https://link.to/openais-next-foundational-model)。
   - 官方澄清 **GPT-5** 将是一个单一、统一的全模态（Omnimodal）模型。
- **Horizon Alpha 表现优于付费 LLM**：**Horizon Alpha** 通过 OpenRouter API 的表现优于付费 **LLM**，能够提供 [针对自定义编程语言的完美 One-shot 代码](https://openrouter.ai/)。
   - 它在编排模式（Orchestrator mode）下的 Shell 使用和任务列表创建优于其他模型，尽管有人推测它 *“可能一直是我们没想到的奇怪东西，比如 Codex-2”*。
- **大上下文窗口引发争论**：尽管 **Gemini** 拥有 100 万的上下文窗口（Context Window），但遗留代码库问题在 **Claude** 和 **ChatGPT** 中得到了更好的解决，引发了关于 [大上下文窗口是否被高估](https://nealgoogs.website) 的争论。
   - 一些人更喜欢上下文窗口较小但输出质量更高的模型，而另一些人则坚持认为大窗口对于 **Agent** 应用至关重要，以便 *“自动记住并织入很久之前的细节”*。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 中的图生视频提示词生成愿景**：成员们正期待 **LM Studio** 未来能推出 **图生视频（image-to-video）提示词生成**和**图片附件**功能，相比 **ChatGPT** 等云端替代方案，他们更青睐离线能力。
   - 作为替代方案，一名成员提到了 **ComfyUI**，并指出它可能未针对 **AMD** 显卡进行优化。
- **LM Studio 的路线图：一个谜**：社区讨论了 **LM Studio** 缺乏**公开路线图（public roadmap）**的问题，并推测其开发计划可能缺乏结构且难以预测。
   - 一名成员表示：*没有公开路线图，所以没人知道*。
- **LM Studio API 安全考量**：用户讨论了跨网络连接 **LM Studio API** 的问题，强调了潜在的安全漏洞。
   - 成员们对 **LM Studio** 未经证实的安全性表示担忧，警告在没有进行适当风险评估和网络保护的情况下不要将其暴露在网络中。
- **Qwen3 Coder 模型面临加载故障**：用户在加载 **Qwen3 Coder 30B** 模型时遇到困难，触发了 *Cannot read properties of null (reading '1')* 错误。
   - 一位成员建议更新到 **0.3.21 b2** 版本，该版本声称已解决此问题，并建议启用**推荐设置**。
- **Nvidia 突发驱动更新**：**Nvidia** 在 **577.00** 发布仅 9 天后迅速发布了 **580.88** 驱动，修复了开启 **NVIDIA Smooth Motion** [5370796] 后可能出现的 GPU 显存速度问题。
   - 该用户直接从 **CUDA toolkit** 运行驱动程序，不使用花哨的控制面板或 **GFE (GeForce Experience)**。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **API 错误困扰 OpenRouter**：用户报告在使用 **OpenRouter API** 调用模型时遇到 **API 错误**，有用户建议检查**模型 ID 前缀（model ID prefix）**和**基础 URL（base URL）**以解决问题。
   - 错误包括 *no endpoint found*，成员们认为这可能是由配置错误引起的。
- **Deepseek v3 免费版模型受停机困扰**：用户在使用 **Deepseek v3 0324 free** 模型时遇到问题，包括*内部错误*、*空响应*和**超时**，导致部分用户转向付费版本。
   - 一名成员指出：*免费版完全超载了。付费版没有这些问题，而且实际的内容质量更好。*
- **Horizon Alpha 被赞高效**：用户称赞 **Horizon Alpha** 模型的有效推理能力和良好表现。
   - 虽然该模型自称由 **OpenAI** 开发，但社区成员澄清它很可能是一个**蒸馏模型（distilled model）**。
- **Personality.gg 利用 OpenRouter 进行角色扮演**：[Personality.gg](https://personality.gg) 推出了一个角色扮演网站，大部分模型使用 **OpenRouter**，通过 **OpenRouter PKCE** 提供对全部 400 个模型的完全免费或廉价访问。
   - 这种集成让用户能够与各种各样的 **AI 模型** 进行角色扮演场景互动。
- **PyrenzAI 的 UX 赢得赞誉**：一位用户称赞了 [PyrenzAI](https://pyrenzai.com) 的 **UI/UX**，欣赏其独特的外观风格以及与其他应用不同的侧边栏设计。
   - 尽管存在速度和安全性方面的批评，该应用程序的用户界面仍获得了积极反馈。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi K2 Turbo 开启极速模式！**: Moonshot 团队发布了 **Kimi K2 Turbo**，宣称速度提升 **4倍**，达到 **40 tokens/sec**。在 [platform.moonshot.ai](https://platform.moonshot.ai/) 上，输入和输出 token 在 **9月1日** 前享受 **5折** 优惠。
   - 得益于相同模型更快的托管服务，用户现在可以通过官方 API 体验显著提升的性能。
- **Moonshot AI 推出全新交流社区**: Moonshot AI 推出了 ***Moonshot AI Forum*** ([https://forum.moonshot.ai/](https://forum.moonshot.ai/))，用于技术讨论、API 帮助、模型行为、调试和开发者技巧。
   - 虽然 *Discord 仍然适合发表情包*和闲聊，但该论坛旨在成为严肃开发和技术讨论的首选之地。
- **Kimi K2 挑战 Claude 的地位**: 有用户反馈 **Kimi K2** 是他们第一个可以替代 **Claude** 使用的模型，并因此放弃了 **Gemini 2.5 Pro**，因为在编程方面，作为一种信息形式，它变得更加自由。
   - 该用户还补充道，他们预计大多数 AI 在知识方面都会趋同，因此它们之间的差异将开始变得模糊。
- **Kimi K2 Turbo 价格细节曝光**: 极速版 **Kimi K2 Turbo** 的定价为：输入 token（缓存）**$0.30/1M**，输入 token（非缓存）**$1.20/1M**，输出 token **$5.00/1M**，促销活动持续至 9 月 1 日。
   - 这相当于在优惠期间以 *2倍的价格获得约 4倍的速度*，专为需要快速处理的用户量身定制。
- **Gemini Ultra 的深度思考代价昂贵**: 成员们嘲讽了 Google Gemini Ultra 的方案：**每月 $250 仅限每天 10 次查询**，一位用户称其“非常滑稽且非常无耻”。
   - 相比之下，每月 $200 的 **ChatGPT pro** 提供无限量的 **Office 365 Pro**，而 **Claude Max** 被认为价格更合理。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes-3 数据集拒绝回复引发关注**: 成员在为量化计算 *imatrix* 时，发现了 **Hermes-3 数据集** 中意外的拒绝回复，从而引发了对 [数据集的进一步调查](https://huggingface.co/datasets/NousResearch/Hermes-3)，以确认数据集中不含拒绝回复。
   - 团队希望通过确保数据集经过充分审查，来确认其中没有任何拒绝回复。
- **Unitree R1 机器人推动具身智能普及**: 社区探讨了 **Unitree R1 基础机器人模型**，售价 **$5,900**，提供完整的开源软件开发工具包（**Python**、**C++** 或 **ROS**）用于 AI 开发，详见[此 YouTube 视频](https://www.youtube.com/watch?v=ljo7TjOqRzs)。
   - 用户表示，它是研究团队向下一代 AI 演进过渡的理想工具。
- **Horizon Alpha 模型引发 OpenAI 猜测**: 成员们辩论了 **OpenAI Horizon Alpha 模型** 是否符合 **OpenAI** 的风格，推测它可能是一个低激活的 **120B MoE** 模型，或者是 **20B** 模型，详见[此推文](https://x.com/apples_jimmy/status/1951180954208444758)。
   - 一些人在 [此 Reddit 帖子](https://www.rxddit.com/r/LocalLLaMA/comments/1mepeqh/the_openai_open_weight_model_might_be_120b/) 中暗示，如果该模型仅支持 **FP4**，那么量化将是不可能的。
- **AnythingLLM 倡导数据主权**: 一位用户分享了关于 **AnythingLLM** 的[推文链接](https://x.com/AnythingLLM/status/1755265335803212271?t=ypxd85gvodugP-ksZP6Nvg&s=19)，并宣称它是 **数据主权** 的未来。
   - 该用户还分享了指向 **Neuronpedia** 的链接以及其他关于 **数据主权** 的推文，包括 [Jack_W_Lindsey 的推文](https://x.com/Jack_W_Lindsey/status/1950952346990862502?t=JGcHUqVwZF8_GBoWV5JPcg&s=19) 和 [heyshrutimishra 的推文](https://x.com/heyshrutimishra/status/1950801664379953468?t=ywRLWQRNGMsoXD8eOMPV-g&s=19)。
- **OSS 模型训练脚本启动**: 一位公共研究工程师开始开发 **OSS 模型训练脚本**，以填补自然光标导航领域缺乏优秀开源模型的空白。
   - 该工程师承认，那些屏蔽爬虫机器人的网站可能会被使用这种技术的新“克隆体”抓取。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Cline 为开源 AI 编程 Agent 筹集 3200 万美元**：AI 编程 Agent Cline 完成了由 **Emergence Capital** 和 **Pace Capital** 领投的 **3200 万美元** 种子轮和 A 轮融资，旨在通过透明的开源 AI 工具赋能开发者，目前已为 **270 万** 开发者提供服务，坚持价格透明且无额外加价。
   - **Latent.Space Podcast** 的一期节目邀请了 **Cline**，与 Saoud Rizwan 和 Pash 讨论了其起源、“计划 + 执行”（Plan + Act）范式、社区工具以及未来方向，可在其 [官网](https://xcancel.com/latentspacepod/status/1951008883163668522) 和 [YouTube](https://www.youtube.com/watch?v=dQw4w9WgXcQ) 上收听。
- **OpenAI 的 OS 模型 YOFO 细节泄露**：**OpenAI** 即将推出的 OS 模型 **YOFO** 的细节在其配置信息被短暂访问后流出，引发了围绕传闻中的 **120B** 和 **20B** 参数变体的热议。
   - 一位成员指出，Jimmy Apples 不愿分享所有的配置细节。
- **Anthropic 的 Claude 生成了 22,000 行代码更新**：Anthropic 合并了一个对其生产环境强化学习代码库的 **22,000 行** 变更，这些代码大部分由 **Claude** 编写，这引发了人们对如此大规模 AI 生成代码变更可靠性的怀疑，尽管该变更大部分是 **json dsl**。
   - 讨论涉及了人工审核流程以及对大规模 AI 驱动代码合并可靠性的担忧；Sauers 确认该变更是真实的。
- **Anthropic 封禁 OpenAI 的 Claude API 访问权限**：Anthropic 撤销了 OpenAI 对其模型（包括 **Claude**）的 API 访问权限，理由是违反了服务条款。
   - **OpenAI** 表示失望，并指出其 API 仍对 **Anthropic** 开放，这引发了社区关于竞争策略和模型训练界限模糊的讨论。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **查询扩展提升 RAG 性能**：讨论围绕在 **RAG** 系统中使用 [查询扩展技术](https://www.promptingguide.ai/techniques/query_expansion) 展开，通过从单个用户查询中生成多个问题来改进信息检索。
   - 对于查询 *'what is the name of the customer'*，建议将其扩展为 *'What is the name?'* 和 *'Who is the customer?'*。
- **Cross-Encoders 在排序中表现不佳**：在 **MS MARCO** 数据上使用 Cross-Encoder 对问题 *'What is the name of the customer?'* 的结果进行排序实验，结果不尽如人意。
   - 预期的首选结果（*Customer Name*）排名低于（*Definition of Customer*），得分分别为 **-0.67** 和 **-1.67**。
- **微调是检索的关键**：根据 [这篇论文](https://arxiv.org/abs/2212.01349)，直接针对检索任务进行训练对于控制排序质量至关重要。
   - 成员们建议，最佳相似度指标取决于具体任务，这意味着通用 Embedding 可能不足以应对专门的检索场景。
- **Gemini 2.5 Flash 存在对 Gemma 的偏袒**：**Gemini-2.5-flash** 一贯将 **Gemma 模型** 的排名排在其他模型之上，甚至超过了一些 70B 模型。
   - 怀疑原因是 Gemma 模型的回答语气对人类和 LLM 来说可能都更具说服力，从而影响了排名。
- **Cinema AI 生成连贯的电影场景**：根据 [arXiv 论文](https://arxiv.org/html/2507.18634v1)，[TheCinema AI](https://thecinema.ai/) 研究项目专注于生成彼此保持 **连贯性** 的电影场景。
   - 该项目探索了生成连贯电影场景的方法，并在项目网站和论文中进行了详细说明。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 用户要求离线访问**：用户正在寻求保存 **NotebookLM studio material** 的方法，以便在没有持续网络连接的旅途中进行离线访问。
   - 一位用户提到将音频下载到 iPad，并将其添加到带有家庭照片的 PowerPoint 幻灯片中。
- **Pro 用户困惑于缺失的预览特权**：多位 **Pro 账户用户**报告称，尽管已经升级，但仍无法使用 **video overview feature**（视频概览功能），而其他一些免费账户用户却可以使用。
   - 一位曾短暂获得视频访问权限的用户在刷新页面后失去了该功能，这表明可能存在持续的推送（rollout）问题。
- **用户梦想利用 Gemini 构建自定义 NotebookLM**：一位用户正考虑使用 **Gemini embedding 001** 和 **Gemini 2.5 models API** 为文档创建一个自定义的多跳、多步推理 **RAG pipeline**。
   - 他们的目标是超越 **NotebookLM** 的能力，理由是其存在 **300 个文件限制**、工作流缺乏透明度以及系统指令受限等局限性。
- **Comet 扩展将 NBLM 推向新高度**：用户讨论了 **Comet**（一个可以访问标签页/历史记录/书签并控制浏览器的浏览器扩展）及其与 **NotebookLM** 集成以进行来源查找的潜力。
   - 有人建议 **Comet** 可能会开发一个扩展，以动态地向 **NotebookLM** 添加来源。
- **西班牙语音频概览依然短小精悍？**：一位用户询问为什么西班牙语的 **Audio Overviews** 时长仍然很短，并分享了一个变通方法：*先将其切换为英语，更改时长，然后提示它用西班牙语生成*。
   - 另一位用户确认，虽然葡萄牙语尚未正式支持讲解视频，但他们能够强制使其运行。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Attention Probes 的性能表现引发争议**：EleutherAI 对 **attention probes**（用于分类 Transformer 隐藏状态的小型神经网络）的实验结果褒贬不一。正如其 [blog post](https://blog.eleuther.ai/attention-probes/) 中详述的那样，由于 **overfitting**（过拟合）和 **optimization issues**（优化问题），其表现有时不如标准的 **linear probes**。
   - 这些实验的代码已在 [GitHub](https://github.com/EleutherAI/attention-probes/) 上开源，邀请社区进行探索和改进，以发现潜在的提升空间。
- **低功耗 LLM 挑战海底场景**：一位成员正在离岸低功耗边缘设备上部署 **LLM**，用于海底测绘、环境监测和自主系统，重点关注 **mission planning**（任务规划）、**anomaly detection**（异常检测）和 **smart data compression**（智能数据压缩）。
   - 科学建模目前受到延迟和带宽限制的制约，但团队正在积极探索克服这些 **challenges**（挑战）的方法。
- **Gemini-2.5-flash 评判 Gemma 生成内容**：一位成员观察到，在比较各种 LLM 时，**Gemini-2.5-flash** 始终给 **Gemma** 的回复打出更高分，这暗示可能存在 *family bias*（家族偏见）或 **Gemma3** 模型具有更优越的性能。
   - 这一观察引发了关于 LLM 评估指标的公平性和客观性，以及开源模型竞争格局的讨论。
- **Weight Tying 引发担忧**：一位成员认为 *weight tying 是一种普遍的糟糕做法*，会导致低效和不稳定，甚至 *在数学上都不合理*，暗示其对模型性能有负面影响。
   - 这一断言在更广泛的研究社区中引发了关于 **weight tying** 有效性的辩论。
- **HF Transformers 的调整引发纠纷**：在 **HuggingFace transformers 4.54** 中，**Llama & Qwen layers** 现在直接返回残差流（不再是 tuple），这可能会影响使用 `nnsight layer.output[0]` 的用户。
   - 一位成员警告说，使用 `nnsight layer.output[0]` 将只能获取第一个 batch 元素，而不是完整的残差流，这一 bug 是通过 [nnterp tests](https://butanium.github.io/nnterp) 发现的。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 依然在代码编辑领域占据主导地位**：用户对 **Aider** 表达了强烈赞赏，称其在控制力和自由度之间实现了优于其他替代方案的平衡。一位用户估计，使用 **DeepSeek**，**Aider** 仅花费 2 美元就在一天内完成了一周的编程工作。
   - 另一位用户感叹道：“**Aider** 简直太强了”，强调了它在代码编辑任务中的高效性。
- **SGLang** 和 **Qwen** 突破速度极限：一位用户报告称，在 **RTX 4090** 上使用 **LM Studio** 运行 **sglang** 和 **Qwen 0.6B Q8** 达到了 **472 tokens/s** 的速度，而普通 **LM Studio** 的速度为 **330 tokens/s**。
   - 另一位用户表示有兴趣复制这种纯本地配置，特别是由于 **vllm** 在其 **4090** 上的表现比 **Ollama** 慢，因此对尝试 **llama.cpp** 感到好奇。
- **讨论多 GPU 主板**：讨论涵盖了硬件配置，一位成员推荐将 [这款 MSI 主板](https://www.bhphotovideo.com/c/product/1864084-REG/msi_meg_x870e_godlike_e_atx.html) 用于 **Fractal North XL** 机箱内的双 **3090** 配置。
   - 其他人分享了自己的配置，包括配备 **3 个 L4** 和 **T40** 的服务器，以及 **Meshify2** 等多种机箱选择。
- **Claude Code** 在高 **Token** 数量下表现不佳：成员们将 **Claude Code** 与其他前沿模型进行了对比，指出其性能在超过 **64k tokens** 后显著下降，尤其是在与 **o3** 和 **Gemini 2.5 Pro** 对比时。
   - 还有人提到，*系统提示词（system prompt）占用了可用上下文窗口（context window）的很大一部分*。
- **在本地对 Qwen3 30B 进行基准测试**：一位成员正在寻找一种简单的方法，使用 **LM Studio** 在本地对 **Qwen3 30B A3B Coder** 的 8 种不同量化版本进行基准测试。
   - 另一位成员建议利用“同一台电脑上的 **llama.cpp server** + **docker aider benchmark**”，并引用了一篇关于让 **Gemini 2.5 Pro** 正常运行的文章。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **安全 MCP 检查器寻求反馈**：一位成员分享了一个用于 **安全 MCP 检查工具** 的 [GitHub 仓库](https://github.com/minte-app/security-mcp-check)，请求社区反馈。
   - 该工具旨在帮助用户识别其 **MCP** 服务器中的潜在漏洞。
- **PayMCP 支付层加入竞争**：一个名为 **PayMCP** 的 **MCP** 新支付层正在开发中，目前已提供 [Python](https://github.com/blustAI/paymcp) 和 [TypeScript](https://github.com/blustAI/paymcp-ts) 实现。
   - 创建者正在寻找合作者和早期采用者，以探索其在促进 **MCP** 服务器接受付款方面的能力。
- **开始探索 MCP 服务器的 PageRank**：一位成员询问了 **MCP** 服务器的 **PageRank** 实现，目标是根据实用性对服务器进行排名。
   - 建议包括将 [MCP 工具库](https://github.com/YogiSotho/mcp-tools-collection) 和 [MCP 注册表](https://github.com/modelcontextprotocol/registry) 作为有价值的资源。
- **JSON MCP 服务器简化处理**：出现了一个 **JSON MCP Server**，旨在帮助 **LLM** 高效解析大型且复杂的 **JSON** 文件（如 **Excalidraw 导出文件**），详见此 [GitHub 仓库](https://github.com/kehvinbehvin/json-mcp-filter)。
   - 该解决方案采用 **schema 生成** 来理解 **JSON** 结构并提取必要数据，从而减少 **tokens** 和上下文消耗。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Hylo 语言引发“异构编程语言”类比**：**Hylo** 编程语言 ([https://www.hylo-lang.org/](https://www.hylo-lang.org/)) 因其通过 **value semantics**（值语义）和调度实现内存安全的方法而受到关注，并被拿来与 **Halide** 和 **Mojo** 进行对比。
   - 成员们报告称，**Hylo** 的负责人目前正在从事 **Scala 3/Scala Native** 的工作，并指出其核心领导者来自 **cpp** 和 **Swift** 背景。
- **AMD 发布 Kernel AI Agent 与 GEAK 基准测试**：AMD 在其论文 [GEAK: INTRODUCING TRITON KERNEL AI AGENT & EVALUATION BENCHMARKS](https://arxiv.org/abs/2507.23194) 中介绍了 **GEAK benchmarks** 和 **Triton Kernel AI Agent**。
   - 探索 AMD 使用其新型 **Triton Kernel AI Agent** 进行内核优化的 **AI-driven kernel optimization**（AI 驱动内核优化）创新方法。
- **__launch_bounds__ 设置触发 CUDA 修复**：一位用户通过向 `__launch_bounds__` 传递 `minBlocksPerMultiprocessor` 参数，设置 `maxThreadsPerBlock=128*3` 和 `minBlocksPerMultiprocessor=1`，修复了编译器在入口处无法确定寄存器数量的问题。
   - `setmaxnreg` 设置仍被忽略，现在是因为一个与 `'extern'` 调用兼容性相关的不同问题。
- **MI300X 基准测试超越 H200**：一位用户询问了在 AMD 硬件上运行新 [MI300X FP8 benchmarks](https://eliovp.com/mi300x-fp8-data%E2%80%91parallel-benchmarks-8-64-gpus-h200-left-behind-b200-within-reach/) 的经验。
   - 该基准测试对比了 **AMD MI300X** 与 **NVIDIA H200**，结果表明 MI300X 在某些 FP8 数据并行任务中优于 H200，性能接近 **NVIDIA B200**。
- **picocuda 编译器在 GPU 领域取得进展**：根据 singularity-systems 频道的成员透露，[picocuda](https://github.com/j4orz/picocuda) 编译器和 [elements](https://github.com/j4orz/elements) 图数据结构项目正在取得进展。
   - 该教科书将大致遵循来自 CGO '16 的 [GPUCC paper](https://dl.acm.org/doi/pdf/10.1145/2854038.2854041)。



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Flux Krea 发布，但不包含 NSFW**：新的 **Flux Krea** 模型已发布，[在此获取](https://huggingface.co/Clybius/FLUX.1-Krea-dev-scaled-fp8)，承诺提供“更多细节”并兼容 base.dev 上的大多数 lora。
   - 早期报告显示，无法生成 **NSFW** 内容。
- **Emergence AI 脱颖而出**：**Emergence AI** 的架构在 **LongMemEval benchmark** 上达到了 [SOTA](https://www.emergence.ai/blog/emergence-is-the-new-new-state-of-the-art-in-agent-memory)，该基准测试用于评估 AI Agent 的长期记忆能力。
   - 这使得 **Emergence AI** 成为内存基准测试领域的领导者。
- **Smolagents 推出 JavaScript 版本**：一位成员发布了 **smolagents.js**，这是 **smolagents** 的 **TypeScript** 移植版，可在 [GitHub](https://github.com/yusuf-eren/smolagents.js) 和 [npm](https://www.npmjs.com/package/smolagents.js) 上获取。
   - 该移植版允许开发者在 **JavaScript** 环境中使用 **smolagents**。
- **判别器学习率微调**：成员们讨论了通过降低 **discriminator learning rate**（判别器学习率）来 **debugging GANs** 以识别问题，建议观察极低值（如 **1e-5**）下的损失变化。
   - 目标是确定判别器损失塌陷至 **0** 是否源于学习率不平衡。
- **Qwen 和 DeepSeek-R1 顶上**：面对 **Llama 4** 的访问限制，在 Colab 上运行 *dummy_agent_library.ipynb* 时，可以使用 **Qwen** 或 **DeepSeek-R1** 作为替代方案。
   - 当 **Llama 4** 的访问受限时，这些模型被认为是可行的替代方案。



---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 上下文窗口大小：128k 输入，8k 输出！**：一位用户注意到了上下文窗口的差异，**Hugging Face model card** 显示为 **32k 上下文**，而 **API docs** 则声称是 **128k**。团队澄清为 **128k 输入**和 **8k 输出**。
   - Cohere 团队成员承诺将更新 Hugging Face model card。
- **速率限制挫败黑客松希望！**：参加 **HackRx 6.0 AI hackathon** 的 **Team Patriots** 遇到了 **10 次调用/分钟试用密钥限制**的速率限制问题。
   - 一位 Cohere 团队成员允许创建多个账户并循环使用密钥以克服限制，这表明速率限制是一个已知障碍。
- **青睐 Cohere Reranker 的初创公司寻求企业版！**：一家对 Cohere 的 **Reranker 实现**充满热情的初创公司表示，由于超出了生产环境 API 的 **1000次/分钟限制**，他们对 **Enterprise plan** 感兴趣。
   - Cohere 指导他们将用例详情发送至 support@cohere.com 和 varun@cohere.com 以获取安全协助。
- **三星 AI 架构师加入聊天！**：来自 **Samsung Biologics** 的一位 AI 架构师介绍了自己，重点关注集成 **AI 方法和工具**，并运行私有的 **LLM service with RAG** 供内部使用。
   - 他们正在寻求讨论 **生物制药或生物学挑战**。
- **Cohere API 遭遇超时！**：#[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/) 频道的一位用户报告在查询 API 时收到多个超时错误。
   - 该用户在聊天中未获得任何反馈。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **垃圾信息发送者仍在活动**：一位成员报告收到私信垃圾信息，并请求管理员永久封禁该活跃用户。
   - 在此期间未采取任何行动，该垃圾信息发送者仍在继续。
- **Wide Research，它够广吗？**：一位成员询问关于使用 **Wide Research** 的初步看法。
   - 未收到关于 **Wide Research** 的评价。
- **Cloudflare 配置卡住，需要帮助**：一位成员在 **Cloudflare** 中配置虚拟环境时遇到问题。
   - 设置一直卡在 **Cloudflare** 上，导致他们无法完成虚拟环境配置。
- **积分崩溃，用户抨击**：一位成员报告每日刷新积分不再起作用，表明平台的积分系统存在问题。
   - 另一位用户提到他们的账户在没有违反任何规则的情况下被停用，表明账户管理可能存在问题。
- **裁员可能导致退款无望**：一位成员指出公司最近进行了裁员，并暗示用户可能无法拿回退款。
   - 该评论暗示公司最近的裁员可能会影响处理退款或解决财务问题的能力。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 与 Novita Labs 联手**：[LlamaIndex 推文](https://twitter.com/llama_index/status/1951315242904068483)宣布 **LlamaIndex** 与 **Novita Labs** 模型推理能力集成。
   - 此次集成提供了多样化的数据源连接，并能将其转换为 Vector embeddings。
- **Gemini 能够流利使用 TypeScript**：[LlamaIndex 推文](https://twitter.com/llama_index/status/1951342252346974431)宣布 **Gemini Live integration** 现已支持 **TypeScript**。
   - 提供了一个演示，展示了如何设置和运行基础的终端聊天。
- **工程师打造链上 AI**：一位资深 AI 与区块链工程师正在使用 **Eliza OS**、**LangGraph** 和自定义工具链构建用于交易、媒体自动化和自主治理的 **on-chain AI agents**。
   - 该工程师在 **Base**、**Solana**、**Berachain**、**Sui**、**Aptos**、**HBAR**、**EVM chains** 以及跨链系统方面拥有丰富经验。
- **LLM 对话的 Git 风格分支**：一位成员正在实验一种系统，其中每条消息都是一个节点，允许在对话的任何点分叉以创建新的上下文路径，详情见[其博客文章](https://gupta-aniket.github.io/Mobile-developer/hire/#projects#branched-llm-mvp)。
   - 该系统目前使用 **Gemini API**，并计划加入 **GPT-4**、**Claude** 和本地 **LLaMA** 模型，正在寻求测试者反馈。
- **Llama Parsers 解析耗时较长**：成员们讨论了 **LlamaIndex parsers** 处理 **.doc**、**.pdf** 和 **.ppt** 文件的性能，特别是处理嵌入在图像中的文本时。
   - 提出的解决方案包括使用高级模式的 **LlamaParse**、将 PPT 转换为 PDF 以提高速度，或实现 **ThreadPoolExecutor()** 进行异步文档解析。

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **为 Yaron Minsky 创造了 "DSpill" 动词**：成员们讨论了谁会再次尝试对 **Yaron Minsky / quant bros** 进行 "**DSpill**"，从而产生了一个新动词 "**DSpill**"。
   - 术语 "**DSpill**" 被提议用来描述针对 **Yaron Minsky** 和 **quant bros** 的行动。
- **DSPy 现在是 RL 了！**：一位成员分享了一篇关于在 DSPy 中使用 **Reinforcement Learning** (RL) 来提高写作质量的[博客文章](https://www.dbreunig.com/2025/07/31/how-kimi-rl-ed-qualitative-data-to-write-better.html)。
   - 虽然没有展开讨论，但对于那些寻求优化生成结果的人来说可能很有趣。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 安装问题值得 GitHub 关注**：一位成员遇到了 **Mojo** 安装困难，并考虑开启一个 **GitHub issue** 来报告该问题。
   - 另一位成员建议他们创建一个包含详细日志的 **GitHub issue**，以协助开发者高效地诊断和解决安装问题。
- **日志是开发者的好朋友**：讨论强调了在 **GitHub** 上报告 **Mojo** 安装问题时包含详细日志的重要性。
   - 提供详尽的日志可以让开发者通过提供调试所需的必要信息，更高效地诊断和解决问题。
- **Print 语句抑制了尾调用优化？！**：一位成员观察到在函数中添加基础的 **print/log 语句**会阻止**尾调用消除 (tail call elimination)**。
   - 讨论围绕在极简的 **Mojo** 示例中，添加 **print/log 语句**如何影响**尾调用消除**展开，并寻求理解这种行为的底层原因。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **OpenAI 拥有 128 个专家的模型泄露**：传闻中一个拥有 **128 个专家 (experts)** 和 **120B 参数**的 **OpenAI** 模型可能已经泄露。
   - 据报道，该模型的权重采用 **FP4** 格式，表明其处于压缩状态。
- **深入探讨混合专家模型 (Mixture of Experts)**：**Mixture of Experts (MoE)** 模型使用多个子网络（专家）配合一个门控网络来路由输入。
   - 这种架构能够在不按比例增加计算成本的情况下扩展模型规模，使其成为一个活跃的研究领域。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **带有答案解析的 MOOC 测验现已发布**：测验及其答案解析的存档现在可以在课程网站的 *"Quizzes"* 部分访问。
   - 这为学生提供了复习课程材料和评估理解程度的资源。
- **Google Forms 将保持关闭**：课程工作人员宣布他们无法重新开放用于测验的 **Google Forms**。
   - 错过通过 **Google Forms** 进行测验的学生应使用现有的存档进行复习。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Qwen3-Coder 以极速冲入 Windsurf**：**Qwen3-Coder** 现已在 Windsurf 中可用，运行速度约为 **2000 tokens/sec**。
   - 该消息通过 [X](https://x.com/windsurf/status/1951340259192742063) 和 [Reddit](https://www.reddit.com/r/windsurf/comments/1mf3e5s/qwen3coder_at_2000_tokenssec_is_now_live_in/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button) 发布，该模型完全托管在位于美国的服务器上。
- **Windsurf 的最新成员：Qwen3-Coder**：Windsurf 现在托管了 **Qwen3-Coder**，拥有高达每秒 **2000 tokens** 的惊人速度。
   - [Reddit](https://www.reddit.com/r/windsurf/comments/1mf3e5s/qwen3coder_at_2000_tokenssec_is_now_live_in/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button) 上正在讨论这一新模型带来的影响。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **开发者寻求机会**：alex_sdk4 询问是否有人在寻找开发者。
   - 未提供关于具体技能、项目或预期的进一步细节。
- **后续：开发者寻求机会**：既然 alex_sdk4 主动联系，这可能是处理小型任务的好机会。
   - 潜在客户可以直接联系 alex_sdk4。



---


**tinygrad (George Hotz) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。


---


**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。


---



您收到此邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。


---

# Discord：各频道详细摘要与链接





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1400553929611280499)** (1048 条消息🔥🔥🔥): 

> `Comet Browser 邀请、Perplexity Pro 图像生成问题、印度 Airtel 用户免费获取 Perplexity Pro、GPT-5 发布推测、模型性能对比` 


- **Comet Browser 邀请逐步发放**：Perplexity 几乎每天都在发放 **Comet Browser** 邀请，优先考虑 **Pro 用户**，但等待时间可能有所不同。
   - 一些用户建议，如果你的女儿有 Pro 账户，她可以发送最多 **2 个邀请**。
- **图像生成故障困扰 Perplexity Pro**：一位用户报告称，**iOS 版 Perplexity Pro** 的图像生成无法整合附件图片，另一位用户确认这是一个反复出现的问题。
   - 模型总结了请求，但没有根据附件文件生成图像，开启新对话也无法稳定解决该问题。
- **印度 Airtel 用户免费获取 Perplexity Pro**：一位用户提到，如果印度 **3 亿 Airtel 用户**是其订阅者，则可以免费获得 **12 个月**的 Perplexity Pro。
   - 要使用此促销活动，你必须位于印度且是 Airtel 订阅者。
- **GPT-5 发布日期仍是个谜**：用户推测 **GPT-5** 的发布，有人建议可能是下周，但另一位成员坚持认为可能只是某种 mini 模型。
   - 一位用户曾在 **API** 中短暂看到 **GPT-5**，但很快就被移除了（[来源](https://x.com/chetaslua/status/1951301385292493259)）。
- **模型性能引发辩论：Sonnet 4 占据主导，O3 表现稳健**：用户讨论了各种模型的使用体验，**Sonnet 4** 在编程和价值方面受到称赞，而 **O3** 被推荐用于推理（[cplx.app](https://www.cplx.app/)）。
   - 讨论涉及 tool call 问题，以及 Anthropic 模型除非被明确要求，否则倾向于*保留信息*的特点。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1400597657667244112)** (7 条消息): 

> `可共享线程、无 Embedding 的 RAG、特朗普-梅德韦杰夫` 


- **线程共享设置已澄清**：一位 Perplexity AI 工作人员向用户澄清，线程应设置为 `Shareable`（可共享）。
   - 分享了一个关于*如何使线程可共享*的链接。
- **OpenAI 无 Embedding 的 RAG**：一位成员分享了一篇 [Medium 文章](https://levelup.gitconnected.com/rag-without-embeddings-heres-how-openai-is-doing-this-45866cd5ddc6)，内容涉及**无 Embedding 的 RAG** 以及 **OpenAI** 如何实现这一点。
   - 该文章由 **Gaurav Shrivastav** 撰写。
- **特朗普-梅德韦杰夫与 2 艘核潜艇的戏码**：一位成员分享了一个 [Perplexity 搜索结果](https://www.perplexity.ai/search/find-information-about-trump-p-g67iddgiQSe1WR4x6GKNjg#2)，内容涉及**特朗普-梅德韦杰夫的戏码，其中 2 艘核潜艇被部署在俄罗斯附近**，用于 8 月 1 日的新人类基准报告（Human Benchmark Report）。
   - 他们分享了一个为报告本身制作的 [Gemini Canvas 信息图](https://g.co/gemini/share/c43c0a891af3)。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1400582585968496640)** (14 messages🔥): 

> `search_domain_filter, Moderator Bot Usage, Image Uploading via API` 


- **排除 Search Domain Filter 故障！**：一位用户报告称，即使作为 Pro 订阅者，**search_domain_filter** 也没有生效，并请求了解如何启用该功能。
   - 另一位成员回复称该功能应该可以正常工作（非 beta 测试版），并请求提供请求代码副本以便协助。
- **Moderator Bot 定价问题？**：一名学生询问了使用 **Perplexity AI** 构建版主机器人的用法和定价，预计大约有 **200 个请求**，每个请求的数据少于 **100 个单词**。
   - 该用户正尝试使用 Perplexity AI 制作一个版主机器人。
- **图片上传导致内部服务器错误！**：一位用户在通过 API 以 base64 格式上传图片时遇到了内部服务器错误（**code 500**）。
   - 随后他们分享了自己的 [B4J code](https://www.b4x.com) 以展示其方法，同时一名成员询问了具体的请求内容和所使用的模型。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1400560721535959271)** (1099 messages🔥🔥🔥): 

> `GPT-5 speculation, Qwen3 model, Cogito V2, Unsloth GRPO and TRL, H100 and batch sizes` 


- **关于 GPT-5 “恐慌式发布”的猜测兴起**：成员们正在猜测 **GPT-5** 会是由于 **OpenAI** 耗尽了扩展模型规模的手段且思维链（**CoT**）收益递减而进行的“恐慌式发布（panic drop）”，还是仅仅是一个中规中矩的改进。
   - 有观点认为 CoT 是一个*完全的死胡同*，通过将模型的向量输出直接反馈回网络而不是使用 Token 进行思考，也有可能实现同样的效果。
- **Qwen3 量化与性能测试**：讨论了 **Qwen3 Coder 30B** 的理想量化方案，一些人发现 **Q4_K_M gguf** 在 **Ollama** 中添加上下文时速度较慢，而另一些人则为了节省 VRAM 而倾向于使用 **UD q3 XL**。
   - 一位成员报告称在 **3090** 上通过 **vllm** 全天候运行 4 月份的 **Qwen3-30b-a3b** 模型，上下文为 **40k**，而其他人则热切期待 Coder 模型的 4-bit AWQ 版本。
- **讨论 Cogito V2 强化学习**：成员们讨论了 **Cogito-v2 GGUFs** 的发布及其强化学习方法，一些人认为这是对现有技术的迭代，而非突破性的创新。
   - 一位成员分享了一篇涵盖 2024 年过程奖励模型（process reward models）的文章 ([synthesis.ai](https://synthesis.ai/2025/02/25/large-reasoning-models-how-o1-replications-turned-into-real-competition/))，另一位成员分享了 **Deepmind** 2022 年探索类似概念的论文 ([arxiv.org](https://arxiv.org/abs/2211.14275))。
- **Unsloth GRPO 已支持 GSPO**：在 Qwen 提议将 **GSPO** 作为 **GRPO** 的更新后，一位成员询问是否会更新 **Unsloth** 以支持 **GSPO** 训练。
   - 另一位成员澄清说 **GSPO** 效率略高，但它已经在 **Unsloth** 中可以使用，并且由于 Unsloth 是一个封装（wrapper），它将自动支持 **TRL** 的更新。
- **传闻中的 OpenAI 新模型引发关注**：关于 **OpenAI** 新模型的传闻正在流传，一些人猜测它可能是最强的操作系统（**OS**）模型，并在评估中击败 **SOTA K2**。
   - 许多人对潜在的稠密（dense）**20B** 基座模型感到兴奋，认为它可以很好地与现有方案搭配，而另一些人则好奇它是稠密模型还是另一个混合专家模型（**MoE**）。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1400858572593565747)** (4 messages): 

> `New member introduction, Community assistance` 


- **新成员加入并坦言自己是新手**：新成员 cyber.n0de 介绍了自己，并幽默地承认自己完全不知所措。
   - 他们表达了对指导的需求，这为社区协助和引导提供了机会。
- **社区伸出援手**：成员 theyruinedelise 迅速回应了新成员的求助并表示愿意提供帮助。
   - 这体现了社区支持新人并提供指导的意愿。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1400780930255163402)** (74 条消息🔥🔥): 

> `VITS checkpoint 训练心得，iOS 上的端侧 VITS 系统，儿童语音录制，用于提升音频忠实度的 Avocodo 和 iSTFTNet，用于 Speech LLM 的通用声码器` 


- **VITS 训练产生灵感时刻**：在对 **VITS checkpoint** 进行通宵训练后，一位成员分享了见解：**模型质量取决于 epoch 数量和数据集质量**，且 **VITS 在说话人解耦（speaker disentanglement）方面表现出色**，适用于创建具有独特声音的模型。
   - 他们指出 **VITS 将原始音频编码到潜空间（latent space）** 以实现逼真的重构，并强调与 RVC 相比，选择取决于具体需求。
- **VITS 在 iOS 上遭遇内存困扰**：一位成员报告称，在 iOS 上使用 **VITS 作为端侧系统语音**时，**Hifi-GAN 解码器**面临内存消耗挑战，需要进行分块解码（chunk-wise decoding）。
   - 他们还发现，通过适当的标注，**VITS 可以学习到逗号处的呼吸等细微差别**，以及引用文本的不同风格。
- **录制儿童语音需谨慎安排时长**：一位成员对**录制儿童语音**所需的小时数表示不确定，这些数据用于微调（fine-tuning）轻快女声以获得更好的基准模型。
   - 另一位成员建议，每个说话人 24 小时时长过多了，强调数据质量优于数量。
- **Avocodo 提升忠实度备受关注**：成员们讨论了将 **Avocodo** 作为在不显著提升速度的情况下快速增强忠实度的手段，并注意到伪影的减少受限于数据集质量。文中附带了一个非官方的 [Avocodo-pytorch 实现](https://github.com/rishikksh20/Avocodo-pytorch)链接。
   - 他们指出，链接中的实现使用了 **Hi-Fi GAN**，但需要自行训练模型。
- **通用声码器探索开启**：一位成员表示需要一个**通用声码器（universal vocoder）**，以便将 **VITS 接入 Speech LLM**，要求速度快、GPU 占用低，且能从头开始训练。
   - 一个建议是使用 [BigVGAN](https://github.com/NVIDIA/BigVGAN)，尽管提问者希望从头训练；其他人则考虑了轻量级 LLM 架构的影响。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1400554633470152867)** (207 条消息🔥🔥): 

> `循环导入错误，合并模型加载时的 RuntimeError，UV venv 性能问题，Qwen3 工具调用问题，vLLM 上的 Qwen3-Coder-30B-A3B-Instruct-1M-Q8_0.gguf` 


- **循环导入引发困扰**：一位成员报告了在使用 `unsloth.FastLanguageModel.from_pretrained` 并设置 `use_async=True` 时，由于**循环导入（circular import）**导致的 `ImportError: cannot import name 'convert_lora_modules' from partially initialized module 'unsloth_zoo.vllm_utils'`。
- **特殊 Token 触发运行时错误**：一位成员在微调并向 tokenizer 和模型的 embedder 添加了 **2 个特殊 token** 后，加载合并模型时遇到了与 **尺寸不匹配（size mismatch）** 相关的 `RuntimeError`。
   - 另一位成员建议，添加新 token 的问题尚未完全解决，系统可能仍尝试加载基础模型的 tokenizer；此外，使用 `resize_model_vocab = 128258` 可能部分解决问题，但对于合并模型并不总是有效，因为它可能会加载基础模型的 tokenizer。
- **UV venv 导致性能下降**：一位用户在 **UV venv** 中使用 Unsloth 时遇到了 **20 倍的性能下降**，导致 CUDA graph 形状捕获期间初始化极其缓慢。
   - 有建议认为 UV 可能会下载所有 xformers 版本导致减速，但一位成员指出他们改用了 mamba，以完全避免使用 UV。
- **Qwen3 的工具调用难题**：一位用户报告称，尽管使用了最新版本的 Unsloth 和 Ollama，但其 **Langchain 应用**中的 **Qwen3 30B 变体**无法像之前的 Qwen3 4B 及更大模型那样可靠地执行**工具调用（tool calling）**。
   - 建议检查 `fast_inference=True`，但用户确认已启用。随后有人建议查看与 vLLM 和 UV 相关的 [此 vLLM issue](https://github.com/vllm-project/vllm/issues/12324)。
- **vLLM 难以处理 GGUF 模型**：一位用户在尝试于 **vLLM** 上运行 **Qwen3-Coder-30B-A3B-Instruct-1M-Q8_0.gguf** 时遇到了 `ValueError: GGUF model with architecture qwen3moe is not supported yet`。
   - 成员们建议 GGUF 格式应该在 *llama.cpp* 上运行，并指出该模型架构可能尚未支持，因此建议从源码安装 Transformers 以尝试解决问题。


  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1400764067383083079)** (8 条消息🔥): 

> `Unsloth Dynamic Quantization, Qwen3 30B-A3B, Space Invaders refined, Roleplay AI finetuning, Gemini Refusals` 


- **Dynamic Quantization 迎来 Quant Clone**：一名成员创建了[一个小型应用程序](https://github.com/electroglyph/quant_clone)，用于以与 Unsloth 动态量化相同的方式对 finetunes 进行量化。
   - 他们希望在自己的 finetunes 上复制 Unsloth 的动态量化效果。
- **Unsloth 的 Qwen3 Coder 模型构建 Space Invaders**：通过使用 **Q4_M unsloth Qwen3 30B-A3B coder 模型**和 VS Code 中的 Cline，一名成员创建并完善了一款 Space Invaders 风格的游戏。
   - 该游戏在不触碰一行代码的情况下，在大约十分钟内完成，可在[此处](https://invaders.smolit.us/space_invaders/)体验。
- **使用 Unsloth 进行 Roleplay AI 微调**：一名成员宣布了一种使用 Unsloth 进行微调的简便方法，并通过其 [roleplay-ai 项目](https://github.com/bjoern-buettner/roleplay-ai/tree/the-one/beam-llm-training)提供更多数据。
   - 该模型已在 Hugging Face 上发布。
- **Gemini 面临高拒绝率**：一名成员询问其他人是否在他们的 finetunes 中遇到了更高水平的拒绝，并将其与 **Gemini** 进行了比较。
   - 该成员发现 *Gemini 在这方面相当令人反感*。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1400731806977753219)** (4 条消息): 

> `Gemma 3 1B garbage, finetuning project, continuous training of loras` 


- **Gemma 3 1B 表现不佳**：一位用户训练了 **Gemma 3 1B**，发现它简直是*垃圾*，且浪费算力，因此坚持使用性能爆表的 **4B** 模型。
   - 他们没有提到训练数据集或训练方法。
- **微调项目正在进行中**：一位用户正寻求合作，使用开源 LLMs 进行**微调项目**，并在 GCP 上提供算力。
   - 他们热衷于从事从**代码模型**到特定领域应用的任何工作。
- **再次讨论持续 LoRA 训练？**：一位用户询问了关于持续更新模型权重的最新工作，并引用了亚马逊几年前关于 **continuous training of LoRAs** 的一些研究。
   - 另一位用户 suresh.b 确认了此类工作的存在，但未提供进一步的细节或链接。


  

---


### **Unsloth AI (Daniel Han) ▷ #[unsloth-bot](https://discord.com/channels/1179035537009545276/1390899684834410536/1400636791173677151)** (114 条消息🔥🔥): 

> `GRO Trainer dataset mapping, Chat template cut off, GRPOTrainer config, Sequence dictionary (seq-dict), Unsloth shape dynamically changes` 


- **排除 GRPO Trainer 中的排列错误**：由于 `Question` 和 `Answer` 等数据集特征问题，用户在使用 **Qwen 2.5** 基础模型时面临 GRPO trainer 的排列错误（permutation errors）。
   - 错误源于 `shuffle_sequence_dict` 函数，特别是 `ref_per_token_logps`，这表明可能存在源代码问题。
- **无法配置 Unsloth 的 Output Embeddings**：用户正努力配置 Unsloth 中 `output_embeddings` 的卸载（offloading）位置，该位置默认存储在 `{model}/output_embeddings.pt` 路径中。
   - 有人担心，如果用户对 `{model}` 路径没有写入权限，这种*行为*将会产生问题。
- **用于微调的 Gemma 图像格式**：用户正在调试在微调 **Gemma-3-it-4B** 时使用多张图像和系统提示词的正确格式，遇到了 `ValueError: Invalid input type`。
   - 正确的格式涉及为文本和图像内容构建带有 `type` 键的输入数据，以适应图像与系统提示词的混合，但要求每个样本的图像数量保持一致。
- **利用 AI 生成微调数据**：用户正在探索将 **50 万 token** 的原始文本转换为 AI 微调数据的方法，特别是考虑使用长上下文模型或 RAG。
   - 讨论包括是否使用带有 RAG 的 **Phi-14B** 模型来创建训练数据，尽管分块（chunking）方案已被否决。
- **SFT 训练期间 VRAM 膨胀**：用户好奇为什么在 **SFT** 训练期间 **VRAM** 会增加，推测内存预分配（pre-allocation）应该能防止这种情况。
   - 有人提到*训练应该可以进行内存预分配*。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1400554342616010903)** (968 messages🔥🔥🔥): 

> `Arena 可见性、排行榜工具提示、数据集中的个人信息、Gemini 的重复倾向、Gemini 2.5 Deepthink` 


- ****Arena 按钮提升浏览体验****：一名成员建议增加 **Search, Image, Video, and Webdev Arena** 三个主要按钮以提高可见性，并分享了一张[概念图](https://cdn.discordapp.com/attachments/1340554757827461211/1400554342167089222/uzzzSHh.png)。
   - 另一名成员建议增加一个 **webdev arena** 按钮，因为它位于一个独立的平台上，并建议在排行榜上增加工具提示，解释 **Rank, CI, and Elo** 是如何确定的。
- ****数据集挖掘暴露危险数据****：一位用户对在发布的 Prompt 中意外包含**个人信息**（电子邮件、密码等）表示担忧，并建议在公开前允许用户删除 Prompt。
   - 一名成员回应称，此类示例应通过私信（DM）发送给他们以便上报，并确认已[与团队分享了这些担忧](https://www.deepcogito.com/research/cogito-v2-preview)。
- ****Gemini 对话出现故障****：一名成员询问其他人是否注意到 **Gemini** 出现重复，但另一名成员认为其表现稳定，并询问 **Gemini 2.5 Flash** 是否有所改进。
   - 一位用户注意到视频限制从 **10 个降至 8 个**，敦促其他人尽快使用视频生成 Arena。
- ****DeepThink 亮相：令人失望？****：**Gemini 2.5 Deepthink** 已面向 Ultra 会员推出，成员们在看到 **10 RPD (每日请求限制)** 后，怀疑其是否值得。
   - 成员们称其为**骗局**和光天化日下的抢劫，有人表示这只是因为 **GPT-5** 即将发布而赶工出来的版本。
- ****GPT-5 传闻引发巨大期待****：讨论围绕 **GPT-5** 的潜在发布展开，一些人期待范式转移，而另一些人则预期增量改进，成员们还讨论了各种性能基准测试数据。
   - 一名成员发表观点认为，*我们正迅速告别“最强”模型的时代*，因为路由到一个非常强大的模型可能对某些任务很有效，但不会一直使用它。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1400888347160739932)** (1 messages): 

> `Veo 3、图生视频、音频功能` 


- **Veo 3 发布图生视频与音频功能**：**Veo 3 Fast & Veo 3** 现在在 [Video Arena](https://discord.com/channels/your_server_id/1397655695150682194) 中支持**具有音频功能的图生视频**。
- **在 Discord 中使用图像创建视频**：video-arena 频道新增了 `/image-to-video` 命令：允许用户从图像创建视频。
   - 鼓励用户对使用新命令创建的最佳视频进行投票。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1400555426764034119)** (580 messages🔥🔥🔥): 

> `后台 Agent、改进 Cursor 设置、Cursor 卡顿问题、YOLO 模式激活、Vibe coding 策略` 


- ****Vibe Coding 需要 GitHub****：一位成员表示 *对于后台 Agent 你需要 GitHub？这东西太酷了* 并附带了图片。
   - 另一位成员在 Prompt 上花费了 **$40**，需要关于改进其 **Cursor** 设置的建议。
- ****Cursor 卡顿 Bug 令用户沮丧****：一位用户报告称，在聊天超过一小时后，他们的机器每隔 **30-60 秒** 就会卡顿。
   - 一名 Cursor 团队成员建议将问题发布在 [Cursor 论坛](https://forum.cursor.com/c/bug-report/6)上，以便获得更好的关注和帮助。
- ****应对模型支出的不确定性****：用户正在比较 **Cursor** 和 **Claude Pro** 的定价，一位用户表示 *老实说，哪里方案最便宜、模型最好我就去哪里，即使 Claude 有新的每周小时限制，其 $200 的方案对我来说目前仍是最好的交易之一*。
   - 另一位用户表示成本会迅速膨胀，*3 个月内花费了 $600*。
- ****Horizon Alpha 体验平平****：一位用户发现他们对 **Horizon-Alpha** 的个人体验 *有点平庸*。
   - 与此形成鲜明对比的是，另一位用户说 *Cursor 是我见过的最好的应用*。
- ****Cursor 用户请求推荐计划****：成员们询问 **Cursor** 是否有推荐计划，因为一位成员提到他目前已经在 Discord 中引导了 *至少 200 多人*。
   - 分享了 [Cursor 大使计划 (Ambassador program)](https://cursor.com/ambassador) 的链接。


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/)** (1 messages): 

lintaffy: 噢，我的 ba 在执行简单命令时还在加载……
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1400554785911865466)** (410 条消息🔥🔥🔥): 

> `Function Calling vs XML, AI 超级智能生物武器, Grok4 vs GPT5, Horizon Alpha 性能, 大上下文窗口` 


- ****Function Calling API**: 内在价值？**: Function Calling API 被认为比使用结构化 XML 进行函数调用具有**内在价值**，但一位成员指出，当模型不支持 tool calling 时，[XML 通常被用作一种变通方案](https://drinkoblog.weebly.com/)。
   - 一些编程模型（如 **Qwen**）不支持 function calling，因此尽管效率略低，内联 tool calls 仍能最大限度地提高互操作性。
- ****扎克伯格的 AI 超级智能**：生物武器威胁？**: **Mark Zuckerberg** 的 AI 超级智能计划引发了对潜在生物武器制造的担忧，一位成员表示 *你不能就这样把超级智能公之于众*。
   - 有人担心，*通过虚假用户和精心设计的语言来控制思想* 比生物武器更危险。
- ****GPT-5 推迟**：Grok4 的胜利？**: 传闻称 **GPT-5** 因无法超越 **Grok4** 而推迟，但另一位成员表示 [OpenAI 计划将多个产品整合到 GPT-5 中](https://link.to/openais-next-foundational-model)。
   - 一位成员还澄清说，**GPT-5** 将是一个单一、统一的全模态 (omnimodal) 模型。
- ****Horizon Alpha 大放异彩**：免费的推理模型？**: **Horizon Alpha** 在通过 OpenRouter API 使用时似乎优于付费 LLM，能够提供 [在自定义编程语言中的完美 one-shot 代码](https://openrouter.ai/)，一位用户声称：*它比 o3 的多轮对话好用 3-4 倍，o3 的多轮对话太糟糕了*。
   - 它在 orchestrator 模式下的高级 shell 使用和任务列表创建证明其优于其他模型，尽管有人认为它 *可能一直是某种我们没想到的奇怪东西，比如 codex-2*。
- ****上下文窗口**：被高估还是必不可少？**: 尽管 **Gemini** 拥有 100 万上下文窗口，但遗留代码库问题在 **Claude** 和 **ChatGPT** 中得到了更好的解决，这引发了关于 [大上下文窗口是否被高估](https://nealgoogs.website) 的辩论。
   - 一些人认为上下文窗口较小但输出质量更好的模型更可取，而另一些人则断言，对于 Agent 应用来说，更大的上下文窗口对于 *自动记忆和编织久远细节* 至关重要。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1400657438746738861)** (11 条消息🔥): 

> `Agent 模式困惑, ChatGPT Agents vs 普通 GPT, GPT-4o 自动推理, 聊天记录丢失` 


- **Agent 模式引发困惑**: 用户对 **Agent Mode** 一词感到困惑，有些人认为这是一个新功能，而它本质上是指现有的高级模式，如 **Code Interpreter**/**Advanced Data Analysis**。
   - 一些成员将初期的不顺归结为基本的成长阵痛，认为它可能会产生困惑、给出错误答案或直接停止工作，但在正常运行时非常 *出色 (awesome)*。
- **ChatGPT Agents vs 普通 GPT**: 一位成员指出 [ChatGPT 模型并不了解最近的发展](https://openai.com/index/introducing-chatgpt-agent/)，包括像 **ChatGPT Agent** 这样的新产品。
   - 另一位成员报告说使用 **Agent Mode** 在 **GitHub** 中解决问题，发现 *观察它的操作过程非常有趣*。
- **GPT-4o 自动推理**: 用户注意到 **GPT-4o** 会自动切换到 *Thinking*（思考）状态，即使没有被标记为 **Deep Research** 或 **Study mode**。
   - 在处理技术或编程相关问题时切换到 **o3** 会导致大量的推理回复，一些用户并不喜欢这样，他们更倾向于简洁的回答。
- **聊天记录丢失**: 一位成员报告说，他们的**聊天记录**（不在文件夹中）在一周内于网页端和移动端应用上逐渐消失。
   - 另一位成员提到 *应该已经修复了*，并且 *截至昨天他们已经修复了该问题*。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1400645347990311045)** (1 条消息): 

> `` 


- **无重大讨论**: 提供的內容中没有值得总结的有意义的讨论。
- **无值得注意的见解**: 提供的屏幕录制不包含任何值得注意的见解或总结主题。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1400645347990311045)** (1 条消息): 

> `` 


- **未讨论任何主题**：提供的消息中没有讨论相关主题。
   - 内容似乎是一个屏幕录制，没有可供总结的具体细节。
- **摘要数据不足**：提供的图像分析缺乏适合生成有意义摘要的文本内容。
   - 需要更多信息或消息详情来创建相关的主题摘要。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1400554223522811936)** (325 条消息🔥🔥): 

> `LM Studio 中的图生视频提示词生成, LM Studio 缺乏路线图, LM Studio 的插件系统, 从网络上的其他计算机连接到 LM Studio API, LM Studio 对 Qwen3 Coder 模型支持` 


- **LM Studio 何时支持图生视频？**：成员们想知道 **LM Studio** 未来何时会推出**图生视频提示词生成**和**图像附件**功能，并表示相比依赖 **ChatGPT**，更倾向于离线解决方案。
   - 一位成员建议将 **ComfyUI** 作为替代方案，但指出它在 **AMD** 显卡上的表现不佳。
- **路线图未知，所以无人知晓**：成员们讨论了 **LM Studio** 缺乏**公开路线图 (Roadmap)** 的问题，有人调侃路线图就像是一个装满随机纸条的大桶。
   - 另一位成员确认**没有人**知道计划是什么，并表示*由于没有公开路线图，所以无人知晓*。
- **保障网络中 LM Studio 的安全**：成员们讨论了从网络上的其他计算机连接到 **LM Studio API** 的问题，并对安全性表示担忧。
   - 有人建议 **LM Studio** 的安全性尚未得到证实，在不了解风险并确保自身网络安全的情况下，不应将其暴露。
- **Qwen 速成课：加载模型！**：成员们讨论了加载 **Qwen3 Coder 30B** 模型时遇到的问题，一位用户遇到了 *Cannot read properties of null (reading '1')* 错误。
   - 一位成员指出用户应将应用版本更新到 **0.3.21 b2**（据称已修复该问题），并提到点击**推荐设置 (recommended settings)**。
- **推测解码：Fabguy 说不值得**：一位成员询问在 **Qwen3 MoE** 模型中使用**推测解码 (speculative decoding)** 的情况，这会导致崩溃错误。
   - 另一位成员指出，*草稿模型和主模型可能会为该任务 [推测解码] 选择非常不同的专家。不值得这样做。*


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1400555314864197673)** (69 条消息🔥🔥): 

> `Nvidia 驱动 580.88, 二手服务器, 部分 KV Cache 卸载, Mac mini M4 vs RTX 3070, 下一代 GPU` 


- **Nvidia 驱动版本的跳跃**：Nvidia 在 **577.00** 发布后不久就发布了驱动 **580.88**，这是一个发布仅 **9 天**的驱动，可能修复了启用 NVIDIA Smooth Motion [5370796] 后 GPU 显存速度的问题。
   - 该用户直接从 CUDA Toolkit 运行驱动，不使用花哨的控制面板或 GFE (GeForce Experience)。
- **思考部分 KV Cache 卸载**：有人提问是否可以在 LM Studio 中进行部分 **KV Cache** 卸载，例如使用 **40GB 模型**，其中 **KV Cache** 需要 **20GB**，而 **GPU** 总共有 **48GB**。
   - 用户想知道是否可以拆分，将 20GB 缓存中的 8GB 放在 GPU 中，其余部分卸载 (offload)。
- **Mac mini M4 对标 RTX 3070**：一位用户想知道拥有 10 核、32GB 内存的 **Mac mini M4** 是否能胜过 **RTX 3070**。
   - 有人表示，如果模型能装入 VRAM，**CUDA** 通常比 Apple Silicon 更快。
- **关于显存建议的闲聊**：一位用户建议攒钱买二手的 **3090**，他们声称这是 AI 使用场景中性价比最高的显卡。
   - 它们的价格约为 **700 欧元**，对于 LLM 来说可能是最佳解决方案，但可能存在曾用于挖矿的问题。
- **5070 TiS 即将发布！**：一位用户推测 **5070 TiS** 即将发布，配备 **24GB** 显存，而 **5070 Ti** 和 **5080** 只有 **16GB** 显存。
   - 另一位用户指出，对于廉价推理，目前 **5060 Ti 16GB** 是最佳选择，单价约 450 欧元，且可以在一块主板上插 3 到 4 张。


  

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1400592183010263101)** (11 messages🔥): 

> `PyrenzAI 发布, Personality.gg, OpenRouter PKCE, PyrenzAI 反馈` 


- **Personality.gg 通过 OpenRouter 实现角色扮演**：[Personality.gg](https://personality.gg) 推出了一个角色扮演网站，大部分模型使用 **OpenRouter**，通过 **OpenRouter PKCE** (Proof Key for Code Exchange) 提供对所有 400 个模型的完全免费或廉价访问。
- **PyrenzAI 发布免费 AI 聊天网站**：一位开发者宣布发布 [PyrenzAI](https://pyrenzai.com)，这是一个具有简洁 UI、多种模型、记忆系统以及对所有层级提供 **免费 RAG** (Retrieval-Augmented Generation) 的 **AI 聊天网站**，使用 OpenRouter 作为主要的 AI 生成后端。
- **PyrenzAI 应用面临速度和安全性批评**：一位用户批评了新发布的 PyrenzAI 应用，指出它在*速度和安全性方面都存在问题*，表现为性能*滞后*以及过度获取用户偏好（每次加载超过 200 次以上）。
- **PyrenzAI 发布后 UI 和 UX 受到赞赏**：一位成员称赞了 [PyrenzAI](https://pyrenzai.com) 的 **UI/UX**，欣赏其独特的外观风格以及与其他应用不同的侧边栏设计。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1400553644956323921)** (242 messages🔥🔥): 

> `API 错误, Deepseek r1, 免费模型, Horizon Alpha, API Key 信用额度限制` 


- **API 错误困扰 OpenRouter 用户**：部分用户报告在尝试通过 OpenRouter API 使用模型时遇到 **API 错误**，包括 *no endpoint found* 错误及其他问题。
   - 一位成员建议检查 **model ID prefix** 和 **base URL** 是否存在潜在的配置错误。
- **Deepseek v3 停机影响用户**：用户报告了 **Deepseek v3 0324 free** 模型的问题，包括*内部错误*、*空响应*和**超时**。
   - 一位成员指出，切换到该模型的付费版本解决了问题，暗示免费版本已过载：*免费版完全过载。付费版没有这些问题，且实际内容质量更好。*
- **免费模型限制令 OpenRouter 用户沮丧**：几位用户询问是否有消息限制更高的**免费模型**，其中一位用户问是否有任何免费模型*不会在 50 条消息时停止？*
   - 成员们澄清说，充值 **$10** 即可获得 **1000 requests/day** 的限制，并引用了详细说明限制的 [OpenRouter 文档](https://openrouter.ai/docs/api-reference/limits#rate-limits-and-credits-remaining)。
- **Horizon Alpha 获得好评**：用户讨论了 **Horizon Alpha** 模型，一些人报告称其推理有效且性能良好。
   - 该模型本身报告称其是由 OpenAI 开发的，但其他成员澄清说它很可能是一个蒸馏模型。
- **预算超支令 API 用户困惑**：一位用户报告称其费用大幅超过了 **API key credit limit**，怀疑使用 Python 线程**并行运行 API 调用**可能是原因。
   - 其他用户分享了类似经历，认为信用额度更新可能不是实时的，导致偶尔出现超支。


  

---


### **OpenRouter (Alex Atallah) ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - 新模型**
  

---

### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1400586072773103901)** (23 messages🔥): 

> `Groq OpenBench, Provider Benchmarks, GPQA Evals, Inspect.ai, Prompt Caching for Kimi K2 and GLM 4.5` 


- **用于服务商基准测试的 Groq OpenBench**：成员们讨论了 [Groq OpenBench](https://github.com/groq/openbench) 仓库，以及它在**服务商基准测试 (provider benchmarks)** 方面被提及的次数。
   - 一位成员提到他们*已经在进行评估工作（最近被列为优先事项）*，例如针对每个服务商的 **GPQA**，并正在扩展到其他维度。
- **Inspect.ai 的发现受到称赞**：一位成员表示很高兴通过 **OpenBench** 链接发现了 [inspect.ai](https://inspect.ai)，并指出这*正是他一直在寻找的东西*。
   - 该用户还表达了对聊天 UI 使用其账户全名且无法控制的担忧，这可能导致潜在的隐私泄露 (doxxing)。
- **询问 Kimi K2 和 GLM 4.5 的 Prompt Caching**：一位用户询问 **OpenRouter** 是否支持 **Kimi K2** 和 **GLM 4.5** 的 **prompt caching**，并指出 **Moonshot** 的平台已直接支持该功能。
   - 他们表示在 [z.ai](https://z.ai) 上看起来似乎支持。
- **突破 20MB 限制：现在可以发送更大的 PDF**：成员们询问新功能是否会突破 **20MB 限制**，他们提到*最近增加了一种发送更大 PDF 的方式*。
   - 新的限制取决于**上游服务商的限制 (upstream provider limit)**。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[announcements](https://discord.com/channels/1369594130807787570/1371757097246785536/1400719197838770217)** (2 messages): 

> `Kimi K2 Turbo, Moonshot AI Forum` 


- **Kimi K2 开启极速模式！**：Moonshot 团队发布了 **Kimi K2 Turbo**，这是 Kimi K2 模型的更快版本，速度提升了 **4 倍**，从 **10 tokens/sec** 提高到 **40 tokens/sec**。
   - 在 **9 月 1 日**之前，用户可享受输入和输出 token 的 **50% 折扣** ([platform.moonshot.ai](https://platform.moonshot.ai/))。
- **Moonshot AI 发布官方论坛**：Moonshot AI 团队宣布启动 ***Moonshot AI Forum*** ([https://forum.moonshot.ai/](https://forum.moonshot.ai/))，作为技术讨论、API 帮助、模型特性、调试和开发者技巧的新枢纽。
   - *Discord 仍然适合玩梗*、闲聊和调戏 ***Kimi Bot***，但如果你想认真搞开发和技术？论坛才是真正的阵地 🔥


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1400679315850526800)** (126 messages🔥🔥): 

> `Kimi vs Claude, Kimi K2 Turbo pricing and speed, Using Kimi K2 Turbo in Claude code, Chinese companies video generation, Kimi K2's prompt format similar to ChatGPT` 


- **Kimi K2 挑战 Claude 的地位**：经过测试，一位用户发现 **Kimi K2** 是第一个让他们觉得可以替代 **Claude** 使用的模型，并完全弃用了 **Gemini 2.5 Pro**。
   - 他们补充说，作为信息的一种形式，代码编写正变得越来越自由，且发展速度超出预期。最终，大多数 AI 在知识储备上都会趋同，它们之间的差异将开始消退。
- **Kimi K2 Turbo 提速 4 倍**：Kimi K2 Turbo 是**相同的模型但拥有更快的托管服务**，现已开启特别促销至 9 月 1 日：输入 token 为 **$0.30/1M** (cached)，**$1.20/1M** (non-cached)，输出 token 为 **$5.00/1M**。
   - 这种定价意味着在折扣期间，*以 2 倍的价格获得了 4 倍的速度*，旨在满足有速度要求的用户，其官方 API 有助于保持稳定性。
- **Kimi K2 Turbo 环境变量设置**：要在 Claude 代码中使用 `kimi-k2-turbo-preview`，请设置以下环境变量配置：`export ANTHROPIC_SMALL_FAST_MODEL=kimi-k2-turbo-preview` 和 `export ANTHROPIC_MODEL=kimi-k2-turbo-preview`。
- **Kimi K2 的 Prompt 设计模仿 ChatGPT**：用户注意到 Kimi 的 Prompt 格式与 **ChatGPT** 非常相似。一位用户为此取消了 **Gemini** ($250/月)、**OpenAI ChatGPT Pro** ($200/月) 和 **Grok 4 Heavy** ($3000/年) 的订阅。
   - 一位成员开玩笑说，要从其他聊天机器人那里获得类似的效果，只需要*添加一个系统提示词 (system prompt)，让它表现得像个放飞自我的 Discord 版主，并告诉它“去尽情表达你自己吧” 哈哈。*
- **Google Gemini 的每日深度思考限制**：成员们嘲讽了 Google Gemini Ultra 计划中 **$250/月却限制每天 10 次查询**的规定，一位成员称其*非常滑稽且非常坑人*。
   - 另一位补充道，即使是 $200/月的 **ChatGPT Pro** 也会提供无限量的 **Office 365 Pro**，而 **Claude Max** 的定价则更为合理。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1400597993442513128)** (110 条消息🔥🔥): 

> `Hermes-3 dataset, Unitree R1 robot, OpenAI's Horizon Alpha model, Quantization challenges, SmolLM and Qwen2.5` 


- **Hermes-3 数据集拒绝响应引发量化波折**：成员们讨论了 **Hermes-3 dataset** 中的拒绝响应（refusals）是刻意设计的还是受审查模型的产物。一位成员在计算量化的 *imatrix* 时发现了意外的拒绝响应，从而引发了对[数据集的进一步调查](https://huggingface.co/datasets/NousResearch/Hermes-3)。
   - 主要目的是确认该数据集中不存在拒绝响应。
- **Unitree R1 机器人推动具身智能普及**：社区讨论了 **Unitree R1 基础机器人模型**，售价 **$5,900**。它为 AI 开发提供全开源的软件开发工具包（**Python**、**C++** 或 **ROS**），如[此 YouTube 视频](https://www.youtube.com/watch?v=ljo7TjOqRzs)所示。
   - 对于正在向下一代 AI 演进的研究团队来说，这是一个理想的工具。
- **Horizon Alpha 模型引发 OpenAI 开源基础模型发布传闻**：成员们讨论了 **OpenAI 的 Horizon Alpha 模型**，推测其风格与 **OpenAI** 相似，可能是一个具有低激活特性的 **120B MoE** 模型，或者如[这条推文](https://x.com/apples_jimmy/status/1951180954208444758)所暗示的 **20B** 模型。
   - Reddit 上也有相关推测，[此帖子](https://www.rxddit.com/r/LocalLLaMA/comments/1mepeqh/the_openai_open_weight_model_might_be_120b/)指出，如果它仅支持 **FP4**，则无法进行正常的量化。
- **OpenAI 泄露模型的量化难题**：社区分析了泄露的配置文件，显示 **OpenAI 的模型** 是一个 **116.8B/5.7B MoE** 模型。当为 GGUF 进行填充（padded）时，参数量会推至 **132.7B/6.3B**。由于架构的 hidden size 原因，除了 **Q4_0**、**Q5_0**、**Q8_0** 和 **IQ4_NL** 之外，很难使用其他方法进行量化。
   - 因为 2880 的 hidden size 不允许量化为 K 或 I quants。
- **SmolLM 与 Qwen2.5 的量化陷阱**：讨论透露 **SmolLM (135B/360B)** 和 **Qwen2.5 0.5B** 的维度无法转换为 K 或 I quants。
   - 成员报告称，对于传闻中的 **GPT 模型**，只有 *o_proj*（来自 attention）可以量化为 K 或 I quants。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1400563468649762909)** (4 条消息): 

> `Input Tokens per Second, Prefill, Gemma, Time to First Token` 


- **探究输入 Token 处理**：一位用户询问了关于推算**每秒输入 Token 数（input tokens per second）**的资源。
   - 另一位成员澄清这指的是 *prefill*（仅指你使用的上下文，而非生成部分）。
- **在笔记本电脑上分析 Gemma**：一位用户报告称，在笔记本电脑上使用 **Gemma** 时，对于 4500 和 9000 Token 的提示词，**Time To First Token** 均为 **~50 秒**左右。
   - 该用户正在寻求该过程的全面概述以进行性能分析（profiling），并指出在不同输入 Token 大小下，每秒输出 Token 数是相同的。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1400851409561194598)** (3 条消息): 

> `OSS Model Training Script, Metaprogramming and DAG->HRM->code automation` 


- **OSS 模型训练脚本：Raizoken 正在构建！**：一位公共研究工程师正在编写一个模型训练脚本，并打算立即将其 **OSS**（开源）。
   - 他们正尝试为自然光标导航创建优质的 **OSS 模型**，但担心模型可能被滥用，例如抓取那些屏蔽爬虫机器人的网站。
- **Raizoken 寻求元编程自动化建议**：一位成员正在寻求关于**元编程（metaprogramming）**和 **DAG->HRM->代码自动化**的建议，并提到他们已在技术栈中使用这些技术，但面临扩展瓶颈。
   - 他们实施了 **Terraform** 和 **Helm** 来抵消这一影响，但在 **Ray 节点** 形成集群时，克隆的从属节点（slaves）遇到了困难，缺乏除冷却时间（cooldowns）之外控制自我生成的机制。


  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1400575460483535091)** (5 条消息): 

> `AnythingLLM, Neuronpedia, Data Sovereignty` 


- **AnythingLLM 预示着 Data Sovereignty 的未来**：一位用户分享了一个关于 **AnythingLLM** 的 [推文链接](https://x.com/AnythingLLM/status/1755265335803212271?t=ypxd85gvodugP-ksZP6Nvg&s=19)，并宣称它是 **Data Sovereignty** 的未来。
- **Neuronpedia 和 Data Sovereignty 受到关注**：该用户还分享了指向 **Neuronpedia** 的链接以及其他与 **Data Sovereignty** 相关的推文，包括 [Jack_W_Lindsey 的推文](https://x.com/Jack_W_Lindsey/status/1950952346990862502?t=JGcHUqVwZF8_GBoWV5JPcg&s=19) 和 [heyshrutimishra 的推文](https://x.com/heyshrutimishra/status/1950801664379953468?t=ywRLWQRNGMsoXD8eOMPV-g&s=19)。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1400851409561194598)** (3 条消息): 

> `OSS model training script, Metaprogramming and DAG->HRM->code automation, Federated cycles between clones in ray nodes` 


- **OSS 模型训练脚本出现**：一位公共研究工程师正在开发一个 **OSS 模型训练脚本**，以解决缺乏用于自然光标导航的优秀 **OSS** 模型的问题。
   - 该工程师指出，屏蔽爬虫机器人的网站可能会被使用该技术的新“克隆体”抓取。
- **Metaprogramming 自动化瓶颈显现**：尽管使用了 Terraform 和 Helm，一位成员仍在寻求关于 **Metaprogramming** 和 **DAG->HRM->代码自动化** 扩展问题的建议。
   - 他们正面临 **Ray** 节点中克隆体之间 **Federated cycles** 的问题，特别是在冷却期之外不受控制的自我产生。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1400565376567480373)** (112 条消息🔥🔥): 

> `Cline's $32M seed funding, CLI orchestration layer, Subagents and Claude Code Office Hours, Bytedance's Seed Diffusion LLM for Code, Open-License Hybrid Reasoning Models` 


- **Cline 完成 3200 万美元融资**：AI coding agent **Cline** 宣布完成了由 **Emergence Capital** 和 **Pace Capital** 领投的 **3200 万美元** Seed 和 Series A 轮融资，旨在支持为开发者提供透明的开源 AI 工具；目前服务于 **270 万** 开发者，并提供无加价的透明定价。
   - Cline 旨在通过避免“阉割版”产品来赋能开发者，专注于访问控制和集中计费等企业级功能。
- **OpenAI 的 OS 模型泄露**：关于 **OpenAI** 即将推出的 **OS** 模型 **YOFO** 的细节在其配置短暂公开后泄露，引发了对传闻中 **120B** 和 **20B** 变体的兴奋。
   - 一位成员指出，Jimmy Apples 不愿分享所有的配置细节。
- **Anthropic 的生产级 Reinforcement Learning 代码库由 Claude 更新**：Anthropic 合并了一个对其生产级 **Reinforcement Learning** 代码库的 **22,000 行** 变更，该变更主要由 **Claude** 编写，这引发了用户对如此大规模 AI 生成代码变更的真实性和安全性的怀疑与讨论；该变更主要是 **json dsl**。
   - Sauers 确认该变更是真实的，讨论涉及了人工审核流程以及对大规模 AI 驱动代码合并可靠性的担忧。
- **Anthropic 切断 OpenAI 的 API 访问权限**：Anthropic 撤销了 **OpenAI** 对其模型（包括 **Claude**）的 **API** 访问权限，理由是违反了服务条款。
   - 一位成员指出，**OpenAI** 表示失望，并提到其 **API** 仍对 **Anthropic** 开放，社区讨论了竞争举措的影响以及模型训练界限模糊的问题。


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1400567742054011033)** (4 条消息): 

> `Cline pod writeup, Latent Space Podcast, Open Source Code Agent` 


- **Cline 播客文章发布！**：**Cline 播客** 的文章现已发布，链接见 [X](https://x.com/latentspacepod/status/1951008883163668522)。
- **Latent.Space Podcast 专题报道 Cline！**：**Latent.Space Podcast** 宣布了关于 **Cline** 的新一期节目，Cline 是一个最近筹集了 **3200 万美元** 的开源 **VSCode** 扩展。
   - 本期节目讨论了 Cline 的起源、'Plan + Act' 范式、顶级社区工具以及未来方向，嘉宾包括 Saoud Rizwan 和 Pash。该播客可在其 [网站](https://xcancel.com/latentspacepod/status/1951008883163668522) 和 [YouTube](https://www.youtube.com/watch?v=dQw4w9WgXcQ) 上收听。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1400554550129070171)** (86 messages🔥🔥): 

> `RAG query expansion techniques, Sentence embeddings vs. token embeddings, Cross-encoders for semantic similarity, Knowledge Graphs for information retrieval, LLMs and question-answer co-occurrence` 


- **查询扩展提升 RAG 性能**：成员们讨论了用于 RAG 系统的 [查询扩展 (query expansion)](https://www.promptingguide.ai/techniques/query_expansion) 技术，建议从单个查询中生成多个问题。
   - 具体而言，针对 *'what is the name of the customer'*（客户姓名是什么），建议生成 *'What is the name?'* 和 *'Who is the customer?'* 等问题以改进检索效果。
- **Cross-Encoders 在排序任务中表现不佳**：使用 Cross-encoder 配合 **MS MARCO** 数据对问题 *'What is the name of the customer?'* 的结果进行排序的实验显示效果较差。
   - 预期的首选结果 (*Customer Name*) 的排名低于 (*Definition of Customer*)，得分分别为 -0.67 和 -1.67。
- **微调检索任务是关键**：根据 [这篇论文](https://arxiv.org/abs/2212.01349)，为了控制排序质量，直接在检索任务上进行训练至关重要。
   - 有建议指出，最优相似度指标取决于具体任务，这意味着通用型 Embeddings 可能无法满足特定的检索场景。
- **Gemini 2.5 Flash 偏好 Gemma 模型**：成员们发现 Gemini-2.5-flash 始终将 **Gemma 模型** 的排名排在其他模型（甚至包括一些 70B 模型）之上。
   - 据推测，Gemma 模型的 **回复语气 (response tone)** 可能对人类和 LLM 来说都更具说服力，从而影响了排名。
- **关于 LLM 并行思考的辩论**：围绕 [Google 的 Gemini 2.5](https://blog.google/products/gemini/gemini-2-5-deep-think/) 及其 *'Deep Think'* 功能展开讨论，该功能利用并行思考来提供更详细、更周全的回复。
   - 一些人认为该模型并行生成多个想法，并带有并行 COT，而另一些人则认为这是对基础模型和上下文管理的更高层级编排。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1400573062881214524)** (3 messages): 

> `The Cinema AI, Generating Movie Scenes` 


- **使用 TheCinema AI 生成连贯的电影场景**：频道将评测 [TheCinema AI](https://thecinema.ai/)，这是一个有趣的、专注于生成相互保持 **连贯性 (cohesion)** 的电影场景的研究项目，详见其 [arXiv 论文](https://arxiv.org/html/2507.18634v1)。
- **TheCinema AI：生成电影场景**：该研究探索了生成连贯电影场景的方法，详情见 [TheCinema AI 项目官网](https://thecinema.ai/) 及其对应的 [arXiv 论文](https://arxiv.org/html/2507.18634v1)。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1400557372002271293)** (4 messages): 

> `NVIDIA Chips, Nintendo Switch` 


- **专家揭露 NVIDIA 芯片功能**：据称美国 AI 领域的专家透露，**NVIDIA 的计算芯片** 具备 *追踪与地理定位* 以及 *远程关机* 技术。
   - 一名成员要求提供 [引用来源](https://citation.needed)，因为该消息源自 *中华人民共和国国家互联网信息办公室 (PRC)*，并称其为 *荒谬且无力的政治博弈手段*。
- **政府限制就像 Nintendo Switch**：一名成员表示，政府施加的限制就像 **Nintendo Switch** 一样。


  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1400575531170402304)** (27 messages🔥): 

> `幻灯片切换中的音频暂停计时，讲解视频的葡萄牙语支持，用于个性化播客的 NotebookLM，来自 Perplexity Deep Research 的 Canvas 信息图` 


- **延迟幻灯片切换以获得更流畅的音频**：用户建议在每次幻灯片切换前增加额外的半秒暂停，以避免讲解视频中音频突然截断。
   - 这一微小的调整可以通过让音频自然淡出，显著*提升观看体验*。
- **葡萄牙语讲解视频：现已提供非官方支持**：一位用户确认，虽然葡萄牙语尚未正式支持讲解视频，但他们能够强制其运行。
   - 另一位用户报告了*参差不齐的结果*，音频是葡萄牙语，但幻灯片有时仍为英语，而另一位用户建议调整 Prompt 以同时指定音频和视频轨道。
- **NotebookLM + Gemini：播客利器？**：一位用户分享了一个工作流：先向 Gemini 提问，然后将答案输入 NotebookLM 以创建个性化播客。
   - 他们发布了链接来演示该过程：[NotebookLM](https://notebooklm.google.com/notebook/aa55ef62-9230-4b15-be5e-a6954247470c/audio) 和 [Gemini Share](https://g.co/gemini/share/11437d9da04c)。
- **通过 NotebookLM 获取来自 Perplexity 的 Canvas 信息图？**：一位用户分享了直接从 **Perplexity Deep Research** 报告创建 Canvas 信息图的过程。
   - 虽然与 NotebookLM 没有直接关系，但他们建议将其作为一个潜在步骤，以*利用 NotebookLM 的强大功能*处理来自其他模型的详细输出，并补充说 *Google 可以且应该做得比目前的视频概览更好*，并指出了当前的 AI 输出。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1400554423864000664)** (65 messages🔥🔥): 

> `离线访问 NotebookLM 工作室材料，视频概览推送问题，用于自定义 RAG 流水的 NotebookLM 和 Gemini API，用于 NotebookLM 的 Comet 浏览器扩展，Audio Overviews 语言和时长限制` 


- **NotebookLM 为经常出差的用户提供离线支持**：用户正在寻求在没有持续互联网连接的情况下，保存 **NotebookLM 工作室材料**以便在旅行期间离线访问的方法。
   - 一位用户提到将音频下载到 iPad 并将其添加到带有家庭照片的 PowerPoint 幻灯片中。
- **视频概览的烦恼：专业版用户思考缺失的预览福利**：几位 **Pro 账户用户**报告称无法访问**视频概览功能**，尽管他们已经升级，而其他免费账户用户却可以访问。
   - 一位曾短暂获得视频访问权限的用户在刷新页面后失去了该权限，这表明推送过程仍存在问题。
- **RAG 梦想：用户计划利用 Gemini 动力定制 NotebookLM**：一位用户正考虑使用 **Gemini embedding 001** 和 **Gemini 2.5 models API** 为文档创建一个自定义的多跳、多步推理 **RAG 流水线**。
   - 他们的目标是超越 **NotebookLM** 的能力，理由是其存在 **300 个文件限制**、工作流缺乏透明度以及系统指令有限等局限性，并希望能*借鉴其成果*。
- **Comet 扩展可能让 NBLM 飞跃**：用户讨论了 **Comet**，这是一个可以访问标签页/历史记录/书签并控制浏览器的浏览器扩展，以及它与 **NotebookLM** 集成以寻找来源的潜力。
   - 有建议提出 **Comet** 可能会编写一个扩展，动态地向 **NotebookLM** 添加来源。
- **西班牙语 Audio Overviews 依然短小精悍？**：一位用户询问为什么西班牙语的 **Audio Overviews** 时长仍然很短。
   - 有人建议了一个变通方法：*将其切换为英语，更改时长，然后提示它用西班牙语生成*。


  

---


### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1400876073763213554)** (1 messages): 

> `Attention probes, Linear probes, 过拟合, 优化问题` 


- **Attention Probes：一种分类隐藏状态的新方法**：EleutherAI 进行了 **attention probes** 实验，这是一种带有注意力机制的微型神经网络，经过训练用于分类 Transformer 的隐藏状态。
   - 尽管寄予厚望，但其表现参差不齐，由于**过拟合**和**优化问题**，有时表现不如标准的 **linear probes**，详见其 [博客文章](https://blog.eleuther.ai/attention-probes/)。
- **Attention Probe 代码已开源**：EleutherAI 已经开源了其 attention probes 实验的代码，邀请他人探索和改进该方法。
   - 该仓库可在 [GitHub](https://github.com/EleutherAI/attention-probes/) 上获取，希望进一步的研究能发现潜在的改进空间。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1400692396144070698)** (11 messages🔥): 

> `离岸低功耗边缘设备上的 LLMs，Gemini-2.5-flash 对 gemma 响应的偏见排名，OpenAI 开源模型配置，MLA 与 MHA 的泛化性` 


- **低功耗 LLMs 挑战离岸部署**：一位成员正在离岸低功耗边缘设备上运行 **LLMs**，重点关注海底测绘、环境监测和自主系统。
   - 由于延迟和带宽挑战，目前的用例主要涉及**任务规划**、**异常检测**和**智能数据压缩**，而非科学建模。
- **Gemini-2.5-flash 表现出对 Gemma 模型的偏袒**：一位使用 **Gemini-2.5-flash** 对各种 **LLMs** 的响应进行排名的成员注意到，其对 **Gemma** 的响应始终存在偏见排名。
   - 该成员推测这可能是由于“家族偏见”，或者仅仅是因为 **Gemma3** 模型确实更优越。
- **OpenAI 即将发布的开源模型配置泄露！**：一位成员分享了即将发布的 **OpenAI 开源模型**的 [config](https://gemini.google.com/share/3b63a193539c)，规格包括 **36 个隐藏层**、**128 个专家 (experts)** 以及 **201088 的词表大小 (vocab size)**。
   - 其他成员向那些作品被 **OpenAI** 采纳进该模型的开发者表示祝贺。
- **MLA 在泛化性辩论中胜过 MHA**：一位成员询问在教科书质量的数据上预训练一个 **300m 参数模型**（使用 **RoPE**）时，**MLA** 还是 **MHA** 的泛化性更好。
   - 另一位成员建议使用 **MLA** (Multi-level Attention) 作为首选架构。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1400589521535373434)** (41 messages🔥): 

> `RoPE 接近最优，权重共享（Weight tying）不可取，语义搜索与 RAG` 


- ****NovelAI** 揭晓 RoPE 研究**：NovelAI 的研究已发布在 [这里](https://research.novelai.net/rope/)，实验将 **RoPE** 中的黄金分割比作为优化目标。
   - 结论是一些*仅对理论家有意义且没有实际应用价值的数学和实验*。
- ****RoPE** 的最优性与通用形式**：这篇 [博客文章](https://nor-blog.pages.dev/posts/2025-07-28-deriving-rope/) 认为，如果尝试推导 **RoPE**，会发现它已接近最优。
   - **N 维**的通用形式需要沿非相干且均匀的方向投影位置，尽管这*并没有太大的实际意义*。
- ****权重共享 (Weight Tying)** 被批为糟糕的实践**：一位成员表示，“权重共享普遍是一种糟糕的实践”，并且是“一种可怕的归纳偏置！”。
   - 他们认为 **weight tying** 是导致许多低效和不稳定的原因，而且“在数学上甚至都说不通”。
- **语义搜索的困扰与 RAG 替代方案**：一位成员在语义搜索方面遇到困难，并提出了关于责任上限的问题。
   - 另一位成员建议使用类似 **RAG** 的方法而不是语义搜索，并表示“语义搜索需要大量的领域特定工程才能正常工作”。


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1400583667998527540)** (1 messages): 

> `EleutherAI 网站 PR，Tensor Program 论文，Yang 等人的论文` 


- **EleutherAI 网站焕新**：一位成员感谢了另一位成员的文章，并提交了一个 [PR](https://github.com/EleutherAI/website/pull/145) 以修复 EleutherAI 网站的一些问题。
   - 该成员请求进行仔细审查，提到他们尚未阅读 **Tensor Program 论文**，可能存在错误，特别是数学附录中关于方程 15-18 的部分。
- **寻求 Tensor Program 方程的澄清**：提交 PR 的成员正在寻求指导，以在 **Yang 等人的论文**中定位特定方程 (**15/17**)，这表明需要对 Tensor Program 的数学基础进行澄清。
   - 这反映了社区在确保网站关于 Tensor Program 内容的准确性和有效性方面的协作努力。


  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1400837578130981006)** (5 messages): 

> `HF transformers update, Llama & Qwen residual streams, Attention Probes Work, NIAH datasets` 


- **HF Transformers 的 Llama 层发布残差流**：在 **HuggingFace transformers 4.54** 中，**Llama & Qwen 层**现在直接返回残差流（而非元组），这可能会影响 `nnsight layer.output[0]` 的用户。
   - 一位成员警告说，使用 `nnsight layer.output[0]` 将只能获取第一个 batch 元素，而不是完整的残差流，这个 bug 是通过 [nnterp tests](https://butanium.github.io/nnterp) 发现的。
- **Attention Probes 取得令人期待的探测进展**：成员们讨论了前景广阔的 Attention Probes，但对其参差不齐的结果感到惊讶，参考 [attention probes work](https://link-to-attention-probes-work)。
   - 一位成员建议在探测时添加后缀，以考虑你试图探测的内容，要求 LM 考虑你试图探测的目标（例如：*上述陈述是否属实？*）。
- **NIAH 数据集的 Last-Token 特性**：成员们指出 Attention Probes 表现不佳主要是由于 **NIAH 数据集**（Needle In A Haystack）的构造方式导致的，该数据集将待分类的内容直接放在序列末尾。
   - 这解释了为什么 Last-token 探测在那里表现良好；在这种情况下，应该同时训练线性探测器（linear probe）和 Attention Probe。
- **McKenzie 探测论文推动 Prompting 进展**：探测论文 [McKenzie et al. 2025](https://arxiv.org/abs/2506.10805v1) 将提示模型给出答案作为基准（结果低于探测器），但没有考虑通过 Prompting 来改进探测。
   - 在均值探测（mean probes）优于 Last-token 探测的数据集上，这可能是一种改进方向，值得进一步研究。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1400755680994136134)** (1 messages): 

> `` 


- **用户找到潜在解决方案**：一位用户表示他们可能已经找到了解决问题的方法，如果不起作用会再发消息。
- **等待用户反馈**：对话目前正在等待用户关于其解决方案是否成功的进一步更新。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1400600571442106388)** (14 messages🔥): 

> `MIT Collaboration on LLM Training, Containerization Issues, CUDA Issues, DeepSpeed checkpoint inspection` 


- **MIT 在 OLMo2 和 DCLM 训练上进行合作**：MIT 和 EAI 正在合作进行 LLM 训练，从 **OLMo2 1B** 或 **DCLM 1B** 开始以熟悉流水线，最初专注于预训练，但计划稍后加入 **SFT** 和安全对齐。
- **容器安装面临棘手的权限错误**：一位用户在使用 Apptainer 进行容器化安装时遇到了权限错误，特别是与 `setgroups` 失败有关，建议尝试使用 `apptainer exec --fakeroot your_image.sif ...` 作为潜在的变通方法。
   - 另一位成员建议，如果容器问题持续存在，根据他们在基于 Slurm 的 HPC 集群上的经验，可以直接在宿主机上使用 conda 环境。
- **Conda 环境中的 CUDA 配置挑战**：切换到 conda 环境后，用户遇到了 **CUDA** 问题，他们认为这些问题已经解决，目前正在尝试安装 **flash-attention** 和 **TE**。
   - 用户询问了在安装 **flash-attention** 和 **TE** 后验证环境设置的具体测试命令。
- **DeepSpeed 检查点检查难题**：一位用户报告说，来自 experimental 分支的 `inspect_ds_checkpoint` 不支持 `pipe_parallel_size=0`，由于检查点目录中缺少 `layer_*` 文件，导致验证检查失败。
   - 他们还询问，在 `pipe_parallel_size=0`、`model_parallel_size=1` 和 zero stage 1 的情况下，从 **(4 nodes x 8 GPUs)** 扩展到 **(8 nodes x 8 GPUs)** 是否在根本上是不可能的。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1400559583528747048)** (61 条消息🔥🔥): 

> `Aider 赞赏、SGLang 和 Qwen 速度、4090 主板与机箱、Aider 对比其他工具、Claude Code 上下文限制` 


- ****Aider** 依然稳坐头把交椅**: 一位成员表达了对 **Aider** 的赞赏，指出与其他工具相比，它在控制力和自由度之间达到了完美的平衡。据估计，使用 DeepSeek 仅花费 **2 美元** 就在一天内完成了相当于 **一周的编程工作**。
   - 另一位用户也表达了同样的看法，称 *"Aider 简直太强了"*。
- ****SGLang** 和 **Qwen** 达到惊人速度**: 一位成员报告称，在 **RTX 4090** 上使用 **sglang** 和 **Qwen 0.6B Q8** 在 LM Studio 中达到了 **472 tokens/s**，而在普通版本的 LM Studio 中仅为 **330 t/s**。
   - 另一位用户表示有兴趣复制这种纯本地工作流，并指出 **vllm** 在其 **4090** 上的性能相对 Ollama 较慢，因此非常想尝试 llama.cpp。
- **多 GPU 配置主板探讨**: 讨论转向了硬件配置，一位成员推荐使用这款 [MSI 主板](https://www.bhphotovideo.com/c/product/1864084-REG/msi_meg_x870e_godlike_e_atx.html) 来搭载双 **3090**，并安装在 Fractal North XL 机箱中。
   - 其他人也分享了自己的配置，包括配备 **3 个 L4** 和 **T40** 的服务器，以及 **Meshify2** 等不同的机箱。
- **Aider 对比 Windsurf 对比 Cursor**: 一位用户对 **Aider**、**OpenHands** 和 **Chode-Pilot** 表示失望，更倾向于使用 **Windsurf** 和 **Cursor**。
   - 他们推测其“核心秘诀”可能在于运行在强大硬件上的巨型闭源模型，并表示在 **Devstral** 和 **Codelamma** 体验不佳后，需要尝试一下 **QWEN3**。
- ****Claude Code** 的上下文窗口注意事项**: 成员们讨论了 **Claude Code** 的性能，其中一人提到它在没有 RAG 的情况下表现良好，并指出 Claude 与其他顶尖模型不同，在高上下文 token 计数下表现会大幅下降。
   - 讨论指出，当超过 **64k tokens** 后，质量会明显下降，这一问题在 **o3** 中不太明显，而 **Gemini 2.5 Pro** 处理得最好。其他人指出，*仅系统提示词（system prompt）就会占用上下文窗口的很大一部分*。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1400608522361770119)** (10 条消息🔥): 

> `Qwen3 30B A3B Coder 基准测试、LM Studio 使用、llama.cpp server + docker aider 基准测试、aider + claude-code max 订阅集成、Gemini 2.5 Pro` 


- **在 LM Studio 中本地测试 Qwen3 30B**: 一位成员希望以简便的方式，使用 **LM Studio** 本地测试 **Qwen3 30B A3B Coder** 的 8 种不同量化版本。
   - 另一位成员建议使用 *在同一台电脑上运行 llama.cpp server + docker aider 基准测试*，并参考了一篇涉及 **Gemini 2.5 Pro** 的文章，其中详细说明了使其正常运行的步骤。
- **Aider 集成 Claude-Code Max 订阅**: 一位成员询问 *aider* 是否可以与 **claude-code max 订阅集成** 配合使用，以利用新的思考模型（thinking model）。
   - 他们还询问命令 `aider --model gemini/gemini-2.5-pro-preview-06-05 --thinking-tokens 32k` 是否是旧的思考方式，以及是否有人成功在 Claude code 上运行 aider。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1400559945786589275)** (43 messages🔥): 

> `Security MCP Check Tool, PayMCP Payment Layer, PageRank for MCP Servers, MCP Eval Platforms, Gateway for Agent Tool Search` 


- ****Security MCP Check Tool** 发布**: 一位成员分享了一个用于 **Security MCP Check Tool** 的 [GitHub repo](https://github.com/minte-app/security-mcp-check)，并寻求反馈。
   - 这可能提供一种检查自家服务器漏洞的方法，但目前尚未提供进一步的解释。
- ****PayMCP** 支付层出现**: 一位成员宣布正在开发 **PayMCP**，这是一个针对 **MCP** 的支付层，并正在寻找合作者和早期用户，目前提供了 [Python](https://github.com/blustAI/paymcp) 和 [TypeScript](https://github.com/blustAI/paymcp-ts) 的实现。
   - 这个新工具承诺让 **MCP** 服务器能够轻松接收付款，尽管目前尚不清楚它支持哪些支付选项。
- ****针对 MCP 服务器的 PageRank**：一种新的搜索工具**: 一位成员询问是否存在针对 **MCP** 服务器或工具的 **PageRank** 实现，旨在根据实用性而非仅仅是名称或描述对服务器进行排名。
   - 另一位成员分享了一个 [MCP 工具仓库](https://github.com/YogiSotho/mcp-tools-collection)，并提到 [MCP registry](https://github.com/modelcontextprotocol/registry) 是潜在的有帮助的资源。
- **寻求 MCP Eval 平台**: 一位成员正在寻找关于 **MCP Eval 平台** 的信息，该平台可以在各种情况下生成不同的 **Agent** 来测试 **MCP** 服务器。
   - 另一位成员表示他们正在为 **Agent** 开发一个搜索工具的网关（Gateway），并计划在周日发布可用版本。
- **掌握 MCP 的指导**: 一位成员请求协助理解并在其工作流中使用 **MCP**，并愿意为他人的指导时间付费。
   - 这凸显了新用户在采用 **MCP** 时面临的复杂性和学习曲线。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1400893140394578104)** (1 messages): 

> `JSON MCP Server, LLM Efficiency with JSON, Schema Generation for JSON, Token Savings` 


- **针对 LLM 的 **JSON MCP Server** 发布**: 一个新的 **JSON MCP Server** 已创建，旨在帮助 **LLM** 高效解析大型且复杂的 **JSON** 文件（例如 **Excalidraw exports**）；详见 [GitHub repo](https://github.com/kehvinbehvin/json-mcp-filter)。
   - 该工具使用 **Schema Generation** 来首先理解 **JSON** 的结构，然后仅提取必要的数据，从而节省 **Token** 和上下文空间。
- **LLM 更高效地解析 JSON 文件**: 该工具的主要目标是帮助 **LLM** 更高效地解析大型且杂乱的 **JSON** 文件。
   - 它通过仅提取所需数据来节省 **Token** 和上下文。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1400575926592344246)** (8 messages🔥): 

> `Hylo Programming Language, Value Semantics, Halide, Scala 3/Scala Native, Heterogenous Programming` 


- ****Hylo** 语言引起关注**: 一位成员询问了 **Hylo** 编程语言 ([https://www.hylo-lang.org/](https://www.hylo-lang.org/))，强调了其通过 **Value Semantics** 和调度实现内存安全的方法，并将其与 **Halide** 进行了类比。
   - 有人指出，该团队与 **Mojo** 一样，都属于“21 世纪的异构编程语言（Heterogenous PL）”范畴。
- **Hylo 的 Value Semantics 和并发**: 成员们表示 **Hylo** 团队仍在完善其 **Value Semantics** 和并发机制，不过其愿景和路线图是让 **Value Semantics** 与调度（Scheduling）、分块（Tiling）、向量化（Vectorizing）完美结合。
   - **Hylo** 团队来自 Adobe STL，拥有开发 **Halide** 的经验。
- ****Scala** 团队成员在参与 Hylo？**: 一位成员提到负责 **Hylo** 的人目前正在从事 **Scala 3/Scala Native** 的工作。
   - 其他成员表示负责人来自 **cpp** 和 **Swift** 领域。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1400862107377074339)** (1 messages): 

> `Triton Kernel AI Agent, GEAK benchmarks` 


- **AMD 推出 GEAK 和 Triton Kernel AI Agent**: AMD 在其论文 [GEAK: INTRODUCING TRITON KERNEL AI AGENT & EVALUATION BENCHMARKS](https://arxiv.org/abs/2507.23194) 中介绍了 **GEAK benchmarks** 和 **Triton Kernel AI Agent**。
- **深入了解 AMD 的 Kernel AI Agent**: 探索 AMD 使用其新型 **Triton Kernel AI Agent** 进行 **AI 驱动的 Kernel 优化** 的创新方法。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1400653536185942016)** (4 messages): 

> `Profiling Copilot, __launch_bounds__ fix for register count issue, setmaxnreg ignored due to extern call` 


- **__launch_bounds__ 设置触发 CUDA 修复**：一位用户通过向 `__launch_bounds__` 传递 `minBlocksPerMultiprocessor`，设置 `maxThreadsPerBlock=128*3` 和 `minBlocksPerMultiprocessor=1`，修复了编译器在入口处无法确定寄存器计数的问题。
   - 他们提到*不确定这具体是如何修复问题的*，但*很高兴能继续推进*。
- **`setmaxnreg` 遇到不兼容问题**：`setmaxnreg` 设置仍被忽略，现在是因为与 `extern` 调用相关的兼容性问题，如消息所示：`ptxas info : (C7506) Potential Performance Loss: 'setmaxnreg' ignored to maintain compatibility into 'extern' call.`
   - 一位成员询问 Kernel 是否正在调用定义在独立编译单元中的 `'extern'` 函数。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1400611087044579428)** (1 messages): 

> `CheckpointPolicy with Custom Kernels, Functorch API` 


- **自定义 Kernel 的 CheckpointPolicy**：一位成员询问了关于在 Torch 中为自定义 Kernel（特别是融合 **MLP**）实现 **CheckpointPolicy** 的文档。
   - 他们询问在 **Functorch API** 中使用它是否可行。
- **Functorch 与自定义 Kernel**：用户希望在利用 **CheckpointPolicy** 的同时，将自定义 Kernel（如融合 **MLP**）集成到 **Functorch API** 中。
   - 他们正在寻求关于如何有效实现这一集成的指导或文档。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1400600521634480232)** (1 messages): 

> `MI300X FP8 benchmarks on AMD, AMD MI300X vs H200 vs B200, FP8 Data Parallel Benchmarks` 


- **MI300X 基准测试超越 H200**：一位用户询问了关于 AMD 硬件上新 [MI300X FP8 基准测试](https://eliovp.com/mi300x-fp8-data%E2%80%91parallel-benchmarks-8-64-gpus-h200-left-behind-b200-within-reach/) 的使用体验。
- **MI300X 上的 FP8 性能**：该基准测试将 **AMD 的 MI300X** 与 **NVIDIA 的 H200** 进行了对比，结果显示 MI300X 在某些 FP8 数据并行任务中优于 H200。
   - 结果表明 **MI300X** 的性能正接近 **NVIDIA 的 B200**。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/)** (1 messages): 

celis1702: 非常感谢你们两位的清晰解释并分享这些细节！
  

---


### **GPU MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1400694013803106367)** (2 messages): 

> `JIT function, JAXPR printing, Static arguments` 


- **JAXPR 打印问题**：一位用户在尝试为使用静态参数的 **JIT** 函数打印 **JAXPR** 时遇到了追踪时（trace-time）错误。
   - 用户尝试使用 `jax.make_jaxpr(jit_func)(1, 2)` 但遇到了错误。
- **静态参数与 JIT 编译**：用户的问题围绕着在 `jax.jit` 中使用 `static_argnames`，然后尝试检查生成的 JAXPR。
   - 理解静态参数如何影响追踪和编译是解决追踪时错误的关键。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1400861221586210836)** (2 messages): 

> `Agreement, Acknowledgement` 


- **肯定确认**：用户 @sshkr16 表示 *"I am yeah"*，在对话中发出同意或确认的信号。
   - 另一位用户 ali_8366 回复了 *"Nice !"*，对最初的陈述表示认可和积极肯定。
- **收到积极确认**：ali_8366 的 "Nice !" 回复表明对 @sshkr16 的确认持积极态度。
   - 这段简单的交流凸显了频道内的相互理解和一致。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1400585395363643573)** (2 messages): 

> `Profiling llama.cpp with rocprofilerv3, AMD machine for GGUF` 


- **使用 rocprofilerv3 分析 llama.cpp 时的困扰**：一位成员询问了关于使用 **rocprofilerv3** 分析 **llama.cpp** 的问题，提到在 **ROCm 6.3.3** 的 **MI50s** 上，PyTorch 代码分析成功，但 llama.cpp 存在问题。
   - 他们很好奇这个问题是否是由于他们的特定配置引起的。
- **用于 GGUF 执行的 AMD 硬件咨询**：另一位成员回复称，他们尚未尝试分析 **llama.cpp**，并询问了用于运行 **GGUF** 模型的具体 AMD 机器。
   - 他们想了解用于 GGUF 推理的硬件配置。


  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1400869779694419998)** (1 messages): 

> `C/ua 招聘，AI Agents 基础设施，创始工程师职位` 


- **C/ua 在旧金山和西班牙寻求人才**：**C/ua** 正在旧金山和西班牙（远程或马德里混合办公）招聘创始工程师，以构建通用 AI Agents 的基础设施。
   - 他们由 **Y Combinator** 支持，正在开发被数千名开发者使用的开源工具。
- **C/ua 构建 AI Agent 基础设施**：**C/ua** 专注于 AI Agents 的基础设施，使其能够大规模安全地使用计算机和应用程序。
   - 该职位涉及构建安全运行时、容器编排、开发者 API 以及 OS 级虚拟化。
- **C/ua 的创始工程师职位**：**C/ua** 正在寻找对系统安全性、可复现性和开发体验充满热情的创始工程师，以塑造 Agents 大规模运行的方式。
   - 有意向的候选人可以在 [旧金山职位公告](https://ycombinator.com/companies/cua/jobs/dIskIB1-founding-engineer-infra-agent-systems) 中找到更多详情。


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/)** (1 messages): 

tonic_1: 很高兴我足够好奇来围观这次对话 🙂 对此感到非常兴奋 🙂
  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1400557751641313300)** (7 messages): 

> `README 更新关于 Resource 与 Prototype 的对比、RCON 客户端断开连接、蓝图 VQA 流水线` 


- **README 中的 Resource 还是 Prototype？**：一位成员询问 README 中关于使用 **Resource** 还是 **Prototype** 来寻找资源点的部分是否为最新，特别是质疑 `position=nearest(Prototype.IronOre))` 是否应该是 `Resource.IronOre`。
   - 另一位成员确认了这种可能性，并指出 *“README 的那部分是 Claude 在 Cursor 中生成的”*。
- **RCON 客户端断开连接，限制了测试**：测试进度受阻，因为 **RCON 客户端** 正在断开连接，错误信息显示为 *“The RCON client is currently not connected to the server”*。
   - 此问题导致无法完成完整的轨迹（trajectories）。
- **蓝图的 VQA 流水线已完成！**：一位成员报告称 **蓝图的 VQA 流水线** 已经完成，目前正专注于数据增强。
   - 增强方法包括 **旋转**、**翻转**和**子区域分块**，旨在将可用蓝图数量增加 10-15 倍。


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1400572352168984789)** (6 messages): 

> `picocuda 编译器、elements 图数据结构、标量编译、GPU 编译、tinygrad 的 AMD GPU 驱动` 


- **Picocuda 和 Elements 项目取得进展**：[picocuda](https://github.com/j4orz/picocuda) 编译器和 [elements](https://github.com/j4orz/elements) 图数据结构项目正在取得进展。
   - 在完成了 [Zero to Hero](https://j4orz.ai/zero-to-hero/) 教科书的标量编译部分后，现在的重点是深入研究 GPU。
- **GPU 编译教科书将参考 GPUCC 论文**：该教科书将大致遵循 CGO '16 的 [GPUCC 论文](https://dl.acm.org/doi/pdf/10.1145/2854038.2854041)，扩展来自 sampsons cs6120 的 big red intermediate language (BRIL)，这是一个小型 LLVM（[BRIL 网页](https://www.cs.cornell.edu/~asampson/blog/bril.html)）。
   - 作者建议通过一小层编排 Host 和 Device 代码的运行时，逐步构建标量和向量编译。
- **用于开源教科书的 AMD GPU**：将购买 **7900xtx** 或 **9070xt** 用于开发，并通过 USB 使用 **tinygrad 的 AMD GPU 驱动**。
   - 选择 AMD 是因为它是开源的，符合教科书针对黑客和折腾玩家（tinkerers）的目标受众。
- **将 llm.c 移植到 AMD 的 HIP**：目标是最终实现 **Karpathy 的 llm.c**（已 fork 并修改为 **AMD 的 HIP** 版本）。
   - 欢迎贡献者参与，特别是 [picocuda](https://github.com/j4orz/picocuda) 的 C 编译器和 [elements](https://github.com/j4orz/elements) 的图数据结构。
- **Host 代码所需的图算法**：Host 代码需要的两个主要图算法是用于中间层（`opto`）的支配者（dominators）算法，以及用于后端（`cgen`）寄存器分配器的图着色算法。
   - 作者推荐使用 lengauer-tarjan 算法处理支配者（类似 rustc），使用 briggs-chaitin-click 算法处理寄存器分配器（类似 hotspot 的 C2）。


  

---

### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1400849332994838631)** (4 messages): 

> `DTensor, Basic Parallelism Schemas, Shape Rotation, DTensor Problems, Marksaroufim visualizations` 


- **DTensor 方案持续推进计划**：成员们计划继续开展 **DTensor** 和 **基础并行方案 (basic parallelism schemas)** 的工作。
   - 会议定于周日 **CEST 时间晚上 8 点**左右举行，如有必要可能会延长。
- **Shape Rotation 任务进行中**：其中一名成员计划专注于 **shape rotation**。
   - 目标是探索并实现高效操作 Tensor 形状的技术。
- **Marksaroufim 可视化启发 DTensor 问题**：成员们将通过 [Marksaroufim 的可视化](https://www.youtube.com/@marksaroufim) 探索新的 **DTensor 问题**。
   - 旨在利用这些可视化深入了解 **DTensor** 开发中的潜在挑战和解决方案。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1400589363934527620)** (26 messages🔥): 

> `Flux Krea model, Synthetic Datasets with HF jobs, AMD GPU for EM image segmentation, Llama CP model path, Gemini-2.5-flash bias` 


- ****Flux Krea** 新模型发布！**：新的 **Flux Krea** 模型已发布，具有*更多细节*，支持 base.dev 上的大多数 LoRA，[在此获取](https://huggingface.co/Clybius/FLUX.1-Krea-dev-scaled-fp8)。
   - 根据初步报告，该模型*无法生成* **NSFW** 内容。
- ****Gemini 2.5 Flash** 可能偏向 **Gemma 3****：一位成员一直尝试使用 **Gemini-2.5-flash** 对各种 LLM 的响应进行排名，发现 **Gemma 3** 模型的排名始终高于其他模型，甚至超过了一些 **70B** 模型。
   - 另一位成员认为确实存在偏见，但 **Gemma 3** 本身也是较好的模型之一，且*默认权重也做得很好*。
- ****HuggingFace Ultrascale** 书籍是博客文章的镜像？**：一位新成员询问 **HF ultrascale book** 的内容是否与博客相同，以及是否需要 **HF pro 订阅**。
   - 另一位成员确认*该书共 246 页*，内容可能与包含大量图片的博客文章相同，并附上了 [Julien Chaumond 的推文链接](https://x.com/julien_c/status/1951277984532279794)。
- **使用 **HF jobs** 生成合成数据集的文档**：一位成员询问如何通过 **HF jobs** 创建合成数据集。
   - 另一位成员提供了 [hf jobs 文档](https://huggingface.co/docs/huggingface_hub/en/guides/jobs)、[脚本](https://ray.so/O8JjQ6X)、[数据集](https://huggingface.co/datasets/dvilasuero/nemotron-kimi) 和 [配置示例](https://huggingface.co/datasets/dvilasuero/nemotron-personas-kimi-questions/raw/main/config.yml)。
- **在 **AMD** 上构建的体积分割工具**：一位成员发布了一个用于 **EM 图像分割的 SOTA 工具**，该工具运行在已有 10 年历史的 **GCN AMD GPU** 上，该 GPU 没有 Tensor Core，甚至不被最新的 ROCm 支持，[项目地址](https://github.com/fgdfgfthgr-fox/Volume_Seg_Tool)。
   - 他们提到，相比其他神经模型，该工具实现了近 **5x-10x 的缩减**（可能指推理开销或模型规模）。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1400825233887465542)** (2 messages): 

> `Note-taking tools, Remnote` 


- **公开笔记应用：Remnote**：一位用户询问正在使用的笔记工具，得到的回复指向了 [Remnote](https://www.remnote.com/)。
   - Remnote 是一款将笔记记录与**间隔复习 (spaced repetition)** 学习相结合的**知识管理工具**。
- **Remnote：不仅是笔记**：讨论强调了 [Remnote](https://www.remnote.com/) 是一个**多功能平台**。
   - 它将传统的笔记功能与**间隔复习**等特性相结合，以增强学习效果和记忆保留。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1400811422899896330)** (2 messages): 

> `AgentUp, Emergence AI, LongMemEval Benchmark` 


- ****AgentUp** 闪亮登场！**：[AgentUp](https://github.com/RedDotRocket/AgentUp) 项目受到关注。
   - 它作为一个值得关注的 Agent 框架，似乎正在获得青睐。
- ****Emergence AI** 宣称在记忆力方面达到 SOTA！**：**Emergence AI** 的新架构在 **LongMemEval 基准测试**上达到了 [SOTA](https://www.emergence.ai/blog/emergence-is-the-new-new-state-of-the-art-in-agent-memory)。
   - 该基准测试用于评估 AI Agent 的长期记忆能力。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1400846134766862366)** (3 messages): 

> `smolagents.js, CodeBoarding, Qwen3-30B-A3B-Instruct-2507` 


- **Smolagents 移植到 JavaScript！**: 一位成员发布了 **smolagents** 的 **TypeScript** 移植版本，名为 **smolagents.js**，可在 [GitHub](https://github.com/yusuf-eren/smolagents.js) 和 [npm](https://www.npmjs.com/package/smolagents.js) 上获取。
- **CodeBoarding 发布！**: 一位成员发布了 **CodeBoarding**，这是一个开源项目，使用静态分析 + LLM 生成 **Python** 代码库的交互式图表，可在 [GitHub](https://github.com/CodeBoarding/CodeBoarding) 上获取。
- **Qwen3 不再拒绝回答问题！**: 一位成员发布了关于调整 **Qwen3-30B-A3B-Instruct-2507** 的帖子，使其不再拒绝回答甚至是很露骨的问题，可在 [HuggingFace](https://huggingface.co/pszemraj/Qwen3-30B-A3B-Instruct-2507-abliterated) 上获取。


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/)** (1 messages): 

cakiki: <@570737726991761409> 请不要在服务器中推广付费内容。
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1400694296440733776)** (2 messages): 

> `Discriminator Learning Rate, GAN Loss Issues, Debugging GANs` 


- **降低 Discriminator 学习率以调试 GAN**: 一位成员建议将 **discriminator learning rate** 降低到极低的值以观察 loss 变化，这有助于精确定位 **GAN** 训练中的问题。
   - 另一位成员询问应该降低到多少，并提到他们目前的速率是 **1e-5**。
- **微调 GAN 学习率**: 讨论集中在通过操纵 discriminator 学习率来调试 **Generative Adversarial Networks (GANs)** 的技术。
   - 目标是确定 discriminator 的 loss 坍塌至 **0** 是否由于学习率不平衡造成的。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1400700923663093811)** (2 messages): 

> `Llama 4 Access, Qwen Model, DeepSeek-R1` 


- **Llama 4 访问受阻！**: 一位成员报告在 Colab 上尝试运行 *dummy_agent_library.ipynb* 时 **无法访问 Llama 4**。
   - 另一位成员建议使用 **Qwen 模型** 或 **DeepSeek-R1** 作为可行的替代方案。
- **替代模型来救场！**: 由于 **Llama 4** 的访问请求被拒绝，请使用 **Qwen** 或 **DeepSeek-R1** 作为替代。
   - 这些模型作为替代品效果应该不错。


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1400583118104039715)** (21 messages🔥): 

> `Cohere API context window size discrepancy, HackRx 6.0 AI hackathon Rate Limit, Cohere Enterprise Plan, Cohere website login error, Cohere Support Team introduction` 


- **上下文窗口大小之争：32k 还是 128k？**: 一位用户指出 **Hugging Face 模型卡片（32k 上下文）** 与 **API 文档（128k 上下文）** 之间存在差异，随后澄清为 **128k 输入** 和 **8k 输出**。
   - 团队承认了该问题，并承诺很快会更新 Hugging Face 模型卡片。
- **Team Patriots 寻求 Rate Limit 限制放宽**: 一个学生团队 **Team Patriots** 请求为 **HackRx 6.0 AI 黑客松** 临时提高速率限制，因为他们受到了 **每分钟 10 次调用的试用 Key 限制**。
   - 一位 Cohere 团队成员允许他们创建多个账户并轮换 Key 以克服限制。
- **初创公司关注 Cohere Enterprise**: 一家非常喜欢 Cohere 的 Reranker 实现的初创公司咨询了 **Enterprise 方案**，以应对生产环境 API 超过 **1000/min 限制**的情况。
   - 他们被引导将用例详情和请求概况发送至 support@cohere.com 和 varun@cohere.com，以便获得安全协助并与相关负责人取得联系。
- **登录错误令人头疼**: 一位用户报告在 **Cohere 网站**登录时出现错误，具体与 **CORS 策略**在新手引导过程中阻止访问有关。
   - 聊天中未立即提供解决方案。
- **Cohere 支持团队表示热烈欢迎**: Varun，Cohere 的一名 **Technical Support Engineer**，介绍了自己并提供了关于在何处发布通用支持和 API 特定讨论的指导。
   - 鼓励新人加入 **Cohere Labs 🧪**，这是一个专门从事研究的 Discord 社区，地址为 [https://cohere.com/research](https://cohere.com/research)。


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/)** (1 messages): 

kaludi: API 是出什么问题了吗？我们的查询出现了多次超时。
  

---

### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1400730205450014751)** (6 条消息): 

> `Samsung Biologics AI 架构师, 专注于 LLM 工作流的 AI 开发者, Dell 工程技术专家, 移动端 & JS 全栈 AI 应用开发者` 


- **三星 AI 架构师加入！**: 一位来自 **Samsung Biologics** 的 AI 架构师介绍了自己，重点关注整合 **AI 方法和工具**以满足业务需求，并强调了供内部使用的带有 **RAG** 的私有 **LLM** 服务。
   - 他们热衷于参与和**生物制药或生物学挑战**相关的对话。
- **专注于 LLM 的 AI 开发者加入**: 一位专注于 **LLM 工作流、基于 Agent 的工具和 MCP 集成**的 AI 开发者介绍了自己，并提到在使用 **LangChain 和 FastAPI** 构建 **AI 销售助手和 RAG 流水线**方面拥有经验。
   - 他们的主要技术栈包括 **Python 和 Node.js**，并对合作和合同工作持开放态度。
- **移动端 AI 应用开发者前来打招呼！**: 一位具有移动端和 JS 全栈经验的 **AI 应用开发者**介绍了自己。
   - 未提供更多额外信息。
- **戴尔 AI 研究人员到访 Cohere**: 一位来自巴西的 **Dell** 工程技术专家介绍了自己，主要从事 **AI 研究**工作。
   - 他们来到这里是为了交流和学习。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1400565757355753552)** (17 条消息🔥): 

> `私信垃圾信息, Wide Research, Cloudflare 问题, Manus AI, 每日刷新额度` 


- **用户投诉私信垃圾信息**: 一名成员报告收到私信垃圾信息，并请求管理员永久封禁该用户。
   - 在此期间未采取任何行动，发送垃圾信息的用户仍未被处理。
- **用户测试 Wide Research 平台**: 一名成员询问了使用 **Wide Research** 的初步感受。
   - 未提供关于 **Wide Research** 的评论。
- **用户无法设置 Cloudflare 虚拟环境**: 一名成员在配置 **Cloudflare** 内的虚拟环境时遇到问题。
   - 设置一直卡在 **Cloudflare** 上，导致他们无法完成虚拟环境配置。
- **每日刷新额度失效**: 一名成员报告每日刷新额度不再起作用。
   - 另一名用户提到，尽管没有违反任何规则，但他们的账户被停用了，这表明平台的额度和账户管理可能存在问题。
- **裁员可能影响退款**: 一名成员指出公司最近进行了裁员，并暗示用户可能无法拿回退款。
   - 该评论暗示公司最近的裁员可能会影响处理退款或解决财务问题的能力。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1400874884271313083)** (2 条消息): 

> `LlamaIndex, Novita Labs, Gemini Live` 


- **LlamaIndex 与 Novita Labs 联手！**: [LlamaIndex 推文](https://twitter.com/llama_index/status/1951315242904068483) 宣布将 **LlamaIndex** 与 **Novita Labs** 的模型推理能力结合使用。
   - 他们提供多样化的数据源连接，并将数据转换为向量嵌入 (Vector Embeddings)。
- **Gemini Live 现已支持 TypeScript**: [LlamaIndex 推文](https://twitter.com/llama_index/status/1951342252346974431) 宣布 **Gemini Live 集成**现已支持 **TypeScript**。
   - 演示展示了如何设置并运行一个简单的终端聊天。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1400596216693129216)** (13 messages🔥): 

> `Agentic AI 代码辅助, LLM 对话的 Git 式分支, 用于 PDF 和 PPT 的 LlamaIndex 解析器, 用于链上 AI Agent 的 AI+区块链` 


- **LLM Web3 工程师求职**：一位资深 AI & 区块链工程师分享了使用 **Eliza OS**、**LangGraph** 和自定义工具链构建用于交易、媒体自动化和自主治理的**链上 AI Agent**的经验。
   - 他在 **Base**、**Solana**、**Berachain**、**Sui**、**Aptos**、**HBAR**、**EVM 链**及跨链系统方面拥有深厚经验。
- **渴望本地 Agentic AI 代码助手**：一名成员询问是否有类似于 **Cursor 编辑器**但可以在本地运行的本地 Agentic AI 代码辅助工具。
   - 其他成员表示 GitHub 上有很多选择，但原帖作者表示**大多数选项存在依赖问题**或缺乏 Agentic 特性。
- **Git 式分支构建对话树**：一名成员正在测试一个系统，其中每条消息都是一个节点，允许从对话树的任何位置分支出来以创建新的上下文路径，详情见[其博客文章](https://gupta-aniket.github.io/Mobile-developer/hire/#projects#branched-llm-mvp)。
   - 该系统目前已使用 **Gemini API** 进行测试，并计划尝试 **GPT-4**、**Claude** 和本地 **LLaMA** 模型，作者正在寻找测试人员。
- **Llama 解析器解析耗时较长**：成员们讨论了使用 LlamaIndex 解析器处理 **.doc**、**.pdf** 和 **.ppt** 文件的情况，特别是当文本位于图像上时。
   - 一位成员建议使用高级模式下的 **LlamaParse**，而另一位成员建议将 PPT 转换为 PDF 以提高速度，或使用 ThreadPoolExecutor() 异步解析文档。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/)** (1 messages): 

dbreunig: https://www.dbreunig.com/2025/07/31/how-kimi-rl-ed-qualitative-data-to-write-better.html
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1400619842368962560)** (2 messages): 

> `DSpill, Yaron Minsky, Quant Bros` 


- **创造新动词：DSpill 来了！**：一名成员询问谁会*再次尝试 **DSpill Yaron Minsky / quant bros***。
   - 另一名成员回复道：*哇，新动词：**DSpill***。
- **Quant bros 被 DSpilled 了？**：一名成员提出了“DSpilling” **Yaron Minsky** 和 **quant bros** 的想法。
   - 这引发了一个新动词“**DSpill**”的诞生，用于描述这一行为。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1400588919791161475)** (2 messages): 

> `Mojo 安装问题, GitHub Issue 报告, 用于调试的详细日志` 


- **Mojo 安装困扰是否该提交 GitHub Issue？**：一名成员报告称连续三天无法安装 **Mojo**，并询问是否应提交 **GitHub Issue**。
   - 另一名成员鼓励他们提交 Issue 并附上详细日志以协助排查故障。
- **建议在 GitHub Issue 中提供详细日志**：在提交关于 **Mojo** 安装问题的 **GitHub Issue** 时，包含详细日志可以显著提供帮助。
   - 这能为开发者提供诊断并更高效解决安装问题所需的信息。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1400972756421443615)** (1 messages): 

> `尾调用消除 (Tail Call Elimination), Print/Log 语句, 最小示例 (Minimal Examples)` 


- **尾调用消除 (Tail Call Elimination) 触发条件**：一名成员在创建最小示例时注意到，如果向函数中添加基础的 **print/log 语句**，则不会触发**尾调用消除**。
   - 该成员询问其中的原因。
- **Print/Log 语句影响尾调用消除**：讨论集中在添加 **print/log 语句**如何阻止最小示例中的**尾调用消除**。
   - 该成员试图理解这种行为背后的深层原因，特别是在创建最小示例时。


  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1400781766913949827)** (3 messages): 

> `OpenAI Model Leak, Mixture of Experts, FP4 weights` 


- **OpenAI 传闻中的模型泄露**：据传 **OpenAI** 有一个**泄露的模型**，拥有 **128 个专家 (experts)** 和 **120B 参数**。
   - 该模型的权重据称采用 **FP4** 格式，表明其处于高度压缩或量化状态。
- **深入了解 MoE**：**Mixture of Experts** 模型由多个子网络（称为 *experts*）组成，并配备一个门控网络，负责学习将每个输入路由到最相关的专家。
   - 这是一个活跃的研究领域，因为它可以实现扩展模型规模而不会成比例地增加计算成本。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1400911694011699361)** (1 messages): 

> `Course Quizzes Availability, Google Forms Reopening` 


- **带有参考答案的测验现已在线发布**：**测验（含参考答案）**的存档已在课程网站的 "Quizzes" 板块上线。
   - 这为学生复习课程材料和评估理解程度提供了宝贵的资源。
- **测验的 Google Forms 将不会重新开放**：课程团队宣布，他们将无法重新开放用于测验的 **Google Forms**。
   - 错过通过 **Google Forms** 参加测验机会的学生应利用现有的存档进行复习。


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1400899899402489856)** (1 messages): 

> `Qwen3-Coder, Token Speed, US Servers` 


- **Qwen3-Coder 以闪电般的速度登陆 Windsurf**：**Qwen3-Coder** 现已在 Windsurf 上线，运行速度约为 **2000 tokens/sec**。
   - 该发布已在 [X](https://x.com/windsurf/status/1951340259192742063) 和 [Reddit](https://www.reddit.com/r/windsurf/comments/1mf3e5s/qwen3coder_at_2000_tokenssec_is_now_live_in/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button) 上宣布，并完全托管在美国服务器上。
- **Windsurf 引入 Qwen3-Coder**：名为 **Qwen3-Coder** 的极速新模型已进入 Windsurf。
   - 该模型以每秒 2000 tokens 的速度运行，[Reddit](https://www.reddit.com/r/windsurf/comments/1mf3e5s/qwen3coder_at_2000_tokenssec_is_now_live_in/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button) 上正在讨论其带来的影响。