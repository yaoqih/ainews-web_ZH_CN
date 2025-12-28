---
companies:
- nvidia
- alibaba
- tencent
- meta-ai-fair
- ibm
- datology
date: '2025-08-18T05:44:39.731046Z'
description: '**Gemma 3 270M** 是一款专为边缘和移动端优化的超小型模型，现已发布并正被广泛采用。**NVIDIA** 推出了两款开源多语言
  ASR（自动语音识别）模型：**Canary 1B** 和 **Parakeet-TDT 0.6B**，它们基于 100 万小时的数据训练，并采用 CC-BY
  许可；此外还发布了高效的 **Nemotron-Nano v2 9B** 模型，速度提升显著。阿里巴巴的 **Qwen-Image-Edit** 提供双语文本编辑和语义图像转换功能。**腾讯混元（Hunyuan）**推出了一款可控的游戏世界视频生成器，该生成器基于超过
  100 万条游戏录像进行训练。**Meta 的 DINOv3** 展示了一个具有强大领域迁移能力的可扩展自监督视觉骨干网络。**IBM** 低调发布了采用商业友好型许可的高效英语嵌入模型。**BeyondWeb**
  合成数据论文显示，与之前的数据集相比，它在训练速度和性能上都有显著提升。对 **HRM** 架构的分析表明，其性能提升主要源于数据增强和脚手架（scaffolding），而非架构创新。“模型和数据集均采用开放许可，可在
  Hugging Face 上获取。”'
id: MjAyNS0w
models:
- gemma-3-270m
- canary-1b
- parakeet-tdt-0.6b
- nemotron-nano-v2
- qwen-image-edit
- dino-v3
people:
- demishassabis
- adrgrondin
- rasbt
- reach_vb
- ctnzr
- clementdelangue
- natolambert
- _akhaliq
- itspaulai
- mervenoyann
- xenovacom
- tomaarsen
- pratyushmaini
- code_star
- leavittron
- k_schuerholt
- giffmana
title: 今天没发生什么特别的事。
topics:
- synthetic-data
- multilingual-asr
- self-supervised-learning
- vision
- model-efficiency
- training-data
- data-augmentation
- model-speedup
- domain-transfer
---

**一个宁静的周末。**

> 2025年8月15日至8月18日的 AI 新闻。我们为您检查了 12 个 subreddits、544 个 Twitter 账号和 29 个 Discord 社区（229 个频道，23654 条消息）。预计节省阅读时间（按 200wpm 计算）：1715 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以美观的 vibe coded 风格展示所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上向我们提供反馈！

[Qwen Image Edit](https://qwenlm.github.io/blog/qwen-image-edit/) 已发布，但大部分内容在两周前就已公布。

对于合成数据（synthetic data）爱好者来说，[BeyondWeb 论文](https://arxiv.org/abs/2508.10975) 值得一读。

---

# AI Twitter 回顾

**新模型、数据集和功能 (Gemma 3 270M, NVIDIA Canary/Parakeet+Nemotron, Qwen-Image-Edit, Tencent Hunyuan, DINOv3, IBM Granite)**

- Gemma 3 270M（超小型）已发布，并已被各技术栈采用。它专为边缘/嵌入式使用而设计，在 MLX 和移动设备上运行速度极快。参见 [@demishassabis](https://twitter.com/demishassabis/status/1956502480675578298) 的公告，[@adrgrondin](https://twitter.com/adrgrondin/status/1957171759876059371) 展示的 MLX iPhone 16 Pro 性能演示（约 140 tok/s），以及 [@rasbt](https://twitter.com/rasbt/status/1957073842393792751) 使用约 1.49 GB RAM 完成的 PyTorch 从零重实现。
- NVIDIA 发布了两个开源多语言 ASR 模型及一个海量开源数据集：
    - Canary 1B 和 Parakeet-TDT (0.6B)，采用 CC-BY 许可证：支持 25 种语言、自动语言检测/翻译、时间戳，一次可处理长达 3 小时的音频，并在“100 万小时”的数据上进行训练。模型和数据集已通过 [@reach_vb](https://twitter.com/reach_vb/status/1957148807562723809) 上传至 Hugging Face，[模型地址](https://twitter.com/reach_vb/status/1957149090913128598)，[数据集地址](https://twitter.com/reach_vb/status/1957149812849066448)。
- NVIDIA 还发布了 Nemotron-Nano v2（9B 混合 SSM），声称比同类尺寸模型提速约 6 倍，并发布了 base/teacher 变体及大部分预训练语料库。详情见 [@ctnzr](https://twitter.com/ctnzr/status/1957504768156561413)，评论见 [@ClementDelangue](https://twitter.com/ClementDelangue/status/1957519608992407848) 和 [@natolambert](https://twitter.com/natolambert/status/1957517030929887284)。在生产环境使用前请检查许可证条款。
- 阿里巴巴的 Qwen-Image-Edit（基于 20B Qwen-Image 构建）增加了精确的双语文本编辑（中/英）、语义转换（物体旋转、IP 创作）和外观级编辑（添加/删除/插入）。演示和链接见 [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1957500569029079083)。该功能已被 [@_akhaliq](https://twitter.com/_akhaliq/status/1957519569016238268) 集成到 Anycoder 中。
- 腾讯混元（Tencent Hunyuan）发布了一个类似 Genie-3 的开源、可控游戏世界视频生成器，在超过 100 万条游戏录像上训练而成；[@itsPaulAi](https://twitter.com/itsPaulAi/status/1957182570309013714) 展示了其长期一致性和实时控制能力。
- Meta 的 DINOv3：一个规模化的自监督视觉骨干网络（backbone），展示了稳定的大模型训练（最高达 7B）、强大的密集特征和领域迁移（如地球观测）。[@OpenCVUniverse](https://twitter.com/OpenCVUniverse/status/1957426189477482558) 提供了优秀的综述，[@mervenoyann](https://twitter.com/mervenoyann/status/1956694798519161118) 提供了视觉工具，[@xenovacom](https://twitter.com/xenovacom/status/1956763976080970071) 提供了一个微型量化 WASM 演示。
- IBM 悄然发布了新的高效英语 embedding 模型（r2 系列），采用商业友好型许可证，信息来自 [@tomaarsen](https://twitter.com/tomaarsen/status/1957389356412330282)。

**合成数据、训练科学与评估 (BeyondWeb, HRM vs Transformers, UI-Venus)**

- 大规模合成预训练：Datology 的 BeyondWeb（基于改写的合成数据，而非“生成器驱动”）报告了显著的提升：
    - 在 14 个基准测试中，表现优于 Cosmopedia 高达 +5.1 个百分点，优于 Nemotron-Synth +2.6 个百分点；相比公开网页数据，训练速度提升高达 7.7 倍。在 BeyondWeb 上训练了 180B token 的 3B 模型击败了在 Cosmopedia 上训练的 8B 模型。参见 [@pratyushmaini](https://twitter.com/pratyushmaini/status/1957456720265154752) 的综述，以及 [@code_star](https://twitter.com/code_star/status/1957458474805408163) 和 [@leavittron](https://twitter.com/leavittron/status/1957468795767058745) 的博客/论文引用。同一作者的推文中讨论了失败模式（模式崩溃、模型“烧坏”）及缓解措施（改写多样性）（例如 [@code_star](https://twitter.com/code_star/status/1957535969646899403)）。
- ARC-AGI/HRM 复现：分析表明，HRM 报告的大部分提升源于数据增强和外环脚手架（outer-loop scaffolding），而非核心架构的创新。参见 [@k_schuerholt](https://twitter.com/k_schuerholt/status/1956669487349891998) 的 ARC Prize 消融实验，[@giffmana](https://twitter.com/giffmana/status/1956705621337608305) 的“普通 Transformer 即可与之匹配”推文（包含递归推理/自我验证），以及 [@cloneofsimo](https://twitter.com/cloneofsimo/status/1957048541127590346) 的回应。
- 通过强化微调实现的 UI Agent：蚂蚁集团的 UI-Venus（基于 Qwen2.5-VL 构建）采用定制奖励的 Group Relative Policy Optimization (GRPO)，从屏幕截图中进行定位和导航（点击、滚动、输入），并跨数据集标准化动作。报告得分：ScreenSpot-V2 定位准确率 95.3%，ScreenSpot-Pro 为 61.9%。详情见 [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1956777729304711639)。

**Agent 系统、互操作性和开发者工具 (SolveIt, MCP, LangChain Deep Agents JS, vLLM CLI, 向量数据库, ChatGPT 集成)**

- SolveIt (Jeremy Howard & John Whitaker)：一个实时、可塑的 Python 环境，融合了文学编程（literate programming）、原位 Web 应用构建、REPL 感知提示以及零样板工具创建——这是“氛围编码”（vibe coding）之外的另一种工作流。[@HamelHusain](https://twitter.com/HamelHusain/status/1956514524628127875) 的深度演讲和 [@jeremyphoward](https://twitter.com/jeremyphoward/status/1956517085603127412) 的演示亮点。
- Model Context Protocol (MCP) 正在获得广泛的教育覆盖——[@_avichawla](https://twitter.com/_avichawla/status/1956966727042154846) 提供了清晰的视觉图表和示例，用于构建 MCP 服务器以及在不同 Agent/IDE 之间组合工具。
- JavaScript 中的 Deep Agents：来自 [@LangChainAI](https://twitter.com/LangChainAI/status/1957478324554395998) 的针对长跨度、使用工具的 Agent 的参考框架（类似于 Deep Research/Claude Code）（[代码库](https://twitter.com/LangChainAI/status/1957478326806983161)）。
- vLLM CLI 社区工具：菜单驱动且可脚本化的 LLM 服务，支持本地/HF 模型管理、性能/内存分析以及实时监控。通过 pip 安装；[@vllm_project](https://twitter.com/vllm_project/status/1957002590220431669) 征求反馈。
- 向量/搜索基础设施：Chroma Cloud 作为开源 Serverless 搜索数据库发布（[公告](https://twitter.com/trychroma/status/1957523079938339163)）；另请参阅 Qdrant 的实用“从零到生产”指南（[推文](https://twitter.com/qdrant_engine/status/1957027122133835847)）和针对法律文档的 LlamaIndex ETL 到知识图谱工作流（[演示](https://twitter.com/jerryjliu0/status/1957141315088728276)）。
- 聊天与 IDE 集成：OpenAI 为 Plus/Pro 用户启用了 Gmail/Google Calendar 连接（[@OpenAI](https://twitter.com/OpenAI/status/1956502071756325055)）；Codex CLI 现在支持 ChatGPT 登录和 GPT-5 配额（[@thsottiaux](https://twitter.com/thsottiaux/status/1957133984657481956)）。Anthropic 发布了用于实时追踪的使用量/成本 API（[@alexalbert__](https://twitter.com/alexalbert__/status/1957556982417879476)）。

**扩展、系统和基础设施 (多 GPU 系列, JAX/TPU 书籍更新, PyTorch 扩展传闻, FlashAttention v4, 带宽感知设计)**

- 多 GPU 教育：GPU MODE 开启了 8 月份的免费系列课程，涵盖 NCCL/多 GPU 通信（Jeff Hammond）、通信库概览（Didem Unat）、Quartet 4-bit 训练以及容错原语 ([@GPU_MODE](https://twitter.com/GPU_MODE/status/1956590989575119048), [@m_sirovatka](https://twitter.com/m_sirovatka/status/1956824361819652175))。与此同时，[@TheZachMueller](https://twitter.com/TheZachMueller/status/1957035112903672006) 启动了 “14 Days of Distributed”（200 多名学习者，提供算力额度，顶级讲师包括 DiLoCo 缩放定律）。
- GPU 如何工作（针对 LLM 训练）：JAX/TPU 书籍增加了一个实质性的 GPU 章节，讨论网络/拓扑及其对训练的影响 ([@jacobaustin132](https://twitter.com/jacobaustin132/status/1957447351011840336))。相关内容：一条实用笔记指出，预训练中的开发速度与生产推理限制有本质区别 ([@cHHillee](https://twitter.com/cHHillee/status/1956911060646072677))。
- 传闻：一个被广泛分享的“前沿实验室八卦”帖子声称，对于某些内部技术栈，当规模超过约 2 万个 GPU 时，PyTorch 会面临严重的缩放痛苦，涉及跨数据中心愿景和内部复刻（fork）——请视为未证实消息，但需注意基础设施工程师的情绪 ([episode 120 by @suchenzang](https://twitter.com/suchenzang/status/1956851798221996178))。
- 内核与硬件：FlashAttention v4 正瞄准 Blackwell ([@scaling01](https://twitter.com/scaling01/status/1957397971479200083))。此外，在 Starlink 推出 500 kbit/s 计划后，John Carmack 就受限链路下的设计思维进行了一场大师课——渐进式图像、零空闲带宽读取器、服务器渲染页面 ([@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1956499838809346279))。

**基准测试、排行榜与机器人（Claude Opus 4.1 Thinking 登顶，Diffbot 进入 Search Arena；Figure/Unitree 进展）**

- 排行榜：Claude Opus 4.1 Thinking 在 LM Arena 各个类别中首次亮相即登顶（总榜与 GPT‑5‑high/Gemini‑2.5‑Pro 并列；在 WebDev/Coding 中排名第一，非推理版的 Opus 4.1 表现也很强劲）。完整结果和分析由 [@lmarena_ai](https://twitter.com/lmarena_ai/status/1957473753337889079) 和 [@scaling01](https://twitter.com/scaling01/status/1957478546391150723) 提供。Arena 还在 Search Arena 中加入了首个开放搜索模型 (Diffbot-small‑xl) ([公告](https://twitter.com/lmarena_ai/status/1957512493586350444))。
- 视觉 + 营养：Qwen 展示了通过单张餐食照片利用结构化 JSON 进行卡路里和重量估算——这是视觉理解转化为结构化输出的一个很好的例子 ([@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1956618027769971070))。
- 机器人：Figure 的人型机器人在工作台抬升和接触过程中保持稳定 ([@kimmonismus](https://twitter.com/kimmonismus/status/1956622073456951347))，Unitree H1 在碰撞后表现出令人印象深刻的稳定性 ([@teortaxesTex](https://twitter.com/teortaxesTex/status/1956592234079433080))，世界人型机器人运动会见证了 95.64 厘米的立定跳高和 1:48 的 4x100 米接力赛胜利 ([@TheHumanoidHub](https://twitter.com/TheHumanoidHub/status/1956744420755439891), [接力赛](https://twitter.com/TheHumanoidHub/status/1957261919401992438))。

**热门推文（按互动量排序）**

- [Andrej Karpathy 对托尔金、文化“高度”以及 AI 辅助创作是否改变了“奇迹”含义的沉思](https://twitter.com/karpathy/status/1956765908078387382) (~19.8k)
- [John Carmack 谈针对 500 kbit/s 世界的产品和协议设计](https://twitter.com/ID_AA_Carmack/status/1956499838809346279) (~14.5k)
- [通过视觉化方式解释 Model Context Protocol (MCP)](https://twitter.com/_avichawla/status/1956966727042154846) (~5.6k)
- [xAI 的电话伴侣（致电 Ani/Valentine）](https://twitter.com/elonmusk/status/1956778643062927850) (~39.8k)
- [在 PyTorch 中从零开始重新实现 Gemma 3 270M](https://twitter.com/rasbt/status/1957073842393792751) (~4.4k)

笔记与杂项

- Coding agents 与市场份额：Qwen3-Coder 在编程工作负载中的采用率迅速提升，并正在蚕食 open routing 市场的份额（[@scaling01](https://twitter.com/scaling01/status/1956858471682617553)，来自 [@chunhualiao](https://twitter.com/chunhualiao/status/1956957519315956074) 的实测）。Kimi K2 因低谄媚性（low sycophancy）而受到关注（[@sam_paech](https://twitter.com/sam_paech/status/1956612862379721057)）。
- RAG 与可观测性：RAG 系统的实用模式（Phoenix、metrics、CI）和评估（evals）方法论持续落地——参见 [@HamelHusain](https://twitter.com/HamelHusain/status/1956737716194034018) 和 [@sh_reya](https://twitter.com/sh_reya/status/1957139727322411291)。
- 值得关注的工具：vLLM CLI（见上文）；Chroma Cloud（见上文）；LangChain Deep Agents JS（见上文）；用于数据增强的 Hugging Face “AI Sheets”（[帖子](https://twitter.com/Saboo_Shubham_/status/1956732735147639081)）。
- 长上下文与“思考”模型 UX：关于 GPT‑5 Thinking/mini Thinking 以及路由 UX 差异的大量实地报告（例如：默认使用非思考模型、对 “fork chat” 的需求）。代表性观点来自 [@scaling01](https://twitter.com/scaling01/status/1957177533746847903) 和 [@wavelettes](https://twitter.com/wavelettes/status/1956866122793521514)。

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Qwen-Image-Edit 模型发布及特性讨论

- [**🚀 Qwen 发布了 Qwen-Image-Edit！**](https://www.reddit.com/gallery/1mttcr9) ([Score: 359, Comments: 53](https://www.reddit.com/r/LocalLLaMA/comments/1mttcr9/qwen_released_qwenimageedit/)): **Qwen-Image-Edit 是一款基于 20B Qwen-Image 架构的模型，提供高级图像编辑功能，专注于精确的双语（中英文）文本操作，以及语义级（如物体旋转、IP 创作）和底层（添加、删除、插入）外观编辑。该模型可通过 [聊天界面](https://chat.qwen.ai/?inputFeature=image_edit)、[Hugging Face](https://huggingface.co/Qwen/Qwen-Image-Edit)、[ModelScope](https://modelscope.cn/models/Qwen/Qwen-Image-Edit) 和 [GitHub](https://github.com/QwenLM/Qwen-Image) 访问，尽管在本地运行（如在 LM Studio 中）的文档似乎有限，特别是考虑到其庞大的** `>20GB` **下载体积。** 评论强调了用户对多模态输入支持的需求（例如类似 GPT-4o 的多图融合），并指出目前本地部署的文档存在缺失，尤其是在 LM Studio 等自定义环境中。
    - 一位用户表示有兴趣向 Qwen-Image-Edit 提供多张图像（例如几个人的照片）并让模型生成一张组合图像，类似于 GPT-4o 的多模态能力。这提出了一个技术问题，即 Qwen-Image-Edit 是否支持多图输入以及除单图编辑之外的复杂图像合成任务。
    - 另一条评论请求关于如何在 LM Studio 中运行 Qwen-Image-Edit 的指导，并指出该项目的 README 缺乏清晰的说明。用户担心下载超过 20GB 的大型模型文件后的实用性和设置过程，突显了模型部署和工具集成方面的典型痛点。
- [**Qwen-Image-Edit 已发布！**](https://www.reddit.com/r/LocalLLaMA/comments/1mttgrf/qwenimageedit_released/) ([Score: 230, Comments: 41](https://www.reddit.com/r/LocalLLaMA/comments/1mttgrf/qwenimageedit_released/)): **阿里巴巴 Qwen 团队推出了 Qwen-Image-Edit，这是一款基于 20B 参数 Qwen-Image 骨干网的图像编辑模型，可在 Hugging Face (https://huggingface.co/Qwen/Qwen-Image-Edit) 上获取。该模型提供精确的双语（中英文）基于文本的图像编辑、高级语义编辑（如物体旋转和概念生成）以及底层外观操作（如物体添加/删除），同时保留风格元素。** 评论者强调了中国开源 AI 的快速进展，并推测其对 Adobe 等专有解决方案可能产生的颠覆性影响。用户还表达了对与社区工具（如 ComfyUI）集成的期待。
    - 一位用户提出了一个问题：为什么会有独立的图像编辑模型，而不是既能生成又能进行高级编辑的统一模型。这引发了关于架构、数据或训练差异是否证明了专门模型的必要性，以及多任务训练的进步是否能产生同样具备高质量编辑能力的生成模型的讨论。
    - 针对 Qwen-Image-Edit 的量化和硬件要求提出了具体的技术问题。一位用户询问该模型在 48GB VRAM 上的可行性，还是需要 96GB，并参考了之前 t2i 模型对双 GPU 推理的支持。这突显了对资源可获取性的持续关注，以及新型图像编辑模型是否在效率上有所提高，或是否支持适用于低端硬件的量化版本。

### 2. 新模型与基准测试发布：NVIDIA Nemotron Nano 2 & Qwen 3 Coder

- [**NVIDIA 发布 Nemotron Nano 2 AI 模型**](https://i.redd.it/pzrpnuykutjf1.jpeg) ([Score: 270, Comments: 49](https://www.reddit.com/r/LocalLLaMA/comments/1mtvgjx/nvidia_releases_nemotron_nano_2_ai_models/)): **该图片是一个基准测试柱状图，直观地比较了 NVIDIA 的 Nemotron Nano-9B-V2 模型与 Qwen3-8B 模型在各种任务上的表现——结果显示 Nemotron 在大多数基准测试中（例如数学方面的 AIM24，长上下文方面的 RULER 128k）获得了更高的准确率，而 Qwen3-8B 则获得了更高的吞吐量。这反映了 Nemotron Nano 2 的核心创新：一种混合 Mamba-Transformer 架构（主要由 Mamba-2 和 MLP 层组成，仅包含 4 个 Attention 层），据报道在不牺牲准确率的情况下实现了 6 倍的速度提升，并支持在单个 GPU 上进行高效的 128K 上下文长度推理。研究论文可在[此处](https://research.nvidia.com/labs/adlr/NVIDIA-Nemotron-Nano-2/)获取，NVIDIA 已开源了大部分训练数据，包括预训练语料库。** 评论强调了仅使用 4 个 Attention 层的技术令人印象深刻，并提到了相关方法（如 Mistral Small 3）。讨论集中在为什么这种架构在其规模下速度更快，而开源数据被广泛称赞为迈向真正开放的一步。
    - Nemotron Nano 2 采用了以 Mamba-2 和 MLP 层为主的混合模型架构，仅包含 4 个 Attention 层，这明显少于许多基于 Transformer 的模型。这种架构选择类似于 Mistral Small 3 等方法，显著提高了推理速度和效率，尤其是在针对此类操作优化的硬件上。如需了解更多架构细节，建议参阅 [Nemotron-H 技术报告](https://arxiv.org/abs/2504.03624)。
    - 报告指出，Nemotron Nano 2 相比传统模型实现了高达 6 倍的加速，但这些性能提升主要是在 NVIDIA GPU（如 A10G）上测得的。由于硬件特定的优化，效率提升可能无法推广到 CPU 或非 NVIDIA 加速器。
    - NVIDIA 在发布模型的同时，还发布了大部分训练数据，包括预训练语料库，这增强了透明度，并促进了可复现性和进一步研究——这种做法加强了此次发布的开源性质。
- [**新的代码基准测试将 Qwen 3 Coder 置于开源模型之首**](https://brokk.ai/power-ranking?round=open&models=flash-2.5%2Cgpt-oss-120b%2Cgpt5-mini%2Ck2%2Cq3c%2Cq3c-fp8%2Cv3) ([Score: 228, Comments: 79](https://www.reddit.com/r/LocalLLaMA/comments/1mto8fa/new_code_benchmark_puts_qwen_3_coder_at_the_top/)): **一项代码生成基准测试对比将 Qwen 3 Coder (FP16) 列为顶级开源模型，其表现优于其 FP8 量化版本、GPT-OSS-120b、V3 和 K2。基准测试结果表明，与 FP16 相比，Qwen 3 Coder 的 FP8 量化会导致显著的性能下降。一些技术评论者注意到缺失了一些模型（例如 GLM 4.5、GLM 4.5 Air），并且对于使用了哪个 Qwen 3 Coder 变体存在歧义；推测倾向于是 480B 参数版本。** 讨论重点在于对 FP8 量化带来的巨大性能损失感到惊讶，以及由于一些奇怪的结果（例如据报道 Gemini Flash 优于 Pro）而对基准测试可靠性表示怀疑。普遍呼吁提高基准测试的透明度并纳入更多模型。
    - 基准测试图像显示，在对 Qwen 3 Coder 使用 FP8 量化时，性能出现了显著下降，这凸显了激进的量化可能会导致模型准确率的明显损失。这强调了 LLM 在计算效率和模型性能之间常见的权衡。
    - 一位评论者详细说明了巨大的效率差距：Qwen 3 Coder (Q3C) 取得了领先的基准测试结果，但与 GPT-OSS-120B 相比，它需要 4 倍的内存、7 倍的激活参数以及每权重 4 倍的比特数，根据某些指标计算，这意味着有利于较小模型的约 `16x 到 28x` 的效率差异。这表明随着参数数量的增加，收益会递减。
    - 关于被测试的具体 Qwen 3 Coder 版本存在技术上的不确定性，推测其为 480B 参数变体。此外，评论者注意到遗漏了 GLM 4.5 和 GLM 4.5 Air 等重要竞争对手，这可能会影响对比排名。

- [**Elon didn't deliver on this announcement. It's already Monday.**](https://i.redd.it/rt8xgjaampjf1.png) ([Score: 739, Comments: 153](https://www.reddit.com/r/LocalLLaMA/comments/1mtct4y/elon_didnt_deliver_on_this_announcement_its/)): **该图片是 Elon Musk 在 X/Twitter 上发布的一张截图，声称“Grok 2”（推测是指为 xAI 的 Grok 聊天机器人提供支持的 LLM）将在下周开源。Reddit 标题和热门评论指出，其自行设定的开源截止日期再次被错过，并暗示这是此类公告第三次跳票。公告图片本身并未透露关于 Grok 2 架构、许可或计划代码仓库的任何技术细节。** 评论者对 Musk 反复错过 Grok 2 开源截止日期表示怀疑，一些人强调了这种无法兑现公开时间表的模式。
    - 一位用户指出 Grok 2 已被视为过时（“已经一整年了”），并质疑其目前的实用性，同时指出 Grok 3 会更受期待，因为它具有更强的模型能力和无审查状态，这是部分用户关注的焦点。

### 3. Open Source and Leaked AI Advances: Kimi K2 & FlashAttention 4 & Community LLM Use Cases

- [**Kimi K2 is really, really good.**](https://www.reddit.com/r/LocalLLaMA/comments/1mtk03a/kimi_k2_is_really_really_good/) ([Score: 256, Comments: 77](https://www.reddit.com/r/LocalLLaMA/comments/1mtk03a/kimi_k2_is_really_really_good/)): **该帖子强调了开源 Kimi K2 模型强大的生产就绪性，理由是它能有效处理多 Agent、多轮工作流和指令遵循任务，而这些任务以前只能通过基础模型和大量的 Prompting 来实现。Kimi K2 因在长上下文任务中表现出色而受到关注，例如生成带有引用的研究报告和复杂的网站构建，并且最近在 LM Arena 排行榜上排名第 8；它在聊天和集成测试中均可免费使用。** 评论认为 Kimi K2 在能力上优于 GLM 4.5，称赞其先进的词汇量和引人入胜的个性，并特别强调了它在 100 多轮对话中保持上下文的鲁棒性，表现优于许多连贯性会下降的其他模型。
    - 反馈强调了 Kimi K2 在长上下文、多轮对话中的卓越表现，用户报告称即使在 100 多轮对话后，它仍能提供连贯且具有上下文意识的答案——优于大多数通常在如此长时间的交互中丢失线索或性能下降的模型。
    - 多位评论者将 Kimi K2 与其他领先模型（如 GLM 4.5、DeepSeek-R1-0528、Qwen3-Coder-480b 和 GPT-OSS-120b）进行了积极对比，但指出 Kimi K2 在 C# 编程任务和高质量的大词汇量方面具有独特优势。
    - Kimi K2 被强调在 STEM 应用中特别强大，无论是免费还是付费、开源还是专有，用户注意到它非常适合作为学习工具，部分原因是其内置信息的高质量通常免除了对网页搜索增强的需求。
- [**FlashAttention 4 Leak**](https://www.reddit.com/r/LocalLLaMA/comments/1mt9htu/flashattention_4_leak/) ([Score: 179, Comments: 36](https://www.reddit.com/r/LocalLLaMA/comments/1mt9htu/flashattention_4_leak/)): **据报道，FlashAttention 4 (FA4) 源代码已通过 SGlang LLM 推理引擎仓库中的一个分支泄露（[diff 链接](https://github.com/sgl-project/sglang/compare/main...hieu/fa4)）。技术概览显示其目标是 NVIDIA Blackwell (SM100+) 架构和第五代 Tensor Core，使用 CuTe DSL（基于 CUTLASS 构建）和一些手写的 PTX 实现。SGlang 是一个类似于 llama.cpp 的分布式 LLM 推理引擎。[截图](https://preview.redd.it/46yfc8z3sojf1.png?width=2600&format=png&auto=webp&s=0b1b33b3f27dfe41ec550142abaa8e0e97bc2449)，[Wayback 链接镜像](https://web.archive.org/web/20250818020651/https://github.com/sgl-project/sglang/compare/main...hieu/fa4)。** 评论注意到了从 FA2 到 FA4 的快速进展，并强调对 Blackwell（NVIDIA 下一代 GPU）的支持是一个重大进步。社区对代码的可用性和持久性表示担忧，因此立即对 diff 进行了存档。
    - 讨论强调了实际采用的滞后：虽然 FlashAttention 4 (FA4) 现在已经可用（或泄露），但许多用户和库（例如企业级或使用 OSS 框架的库）仍在使用 FlashAttention 2 (FA2)，这突显了在更新核心模型加速组件时常见的惯性或兼容性限制。
    - 通过询问有关目前哪些 FlashAttention 版本受到 llama.cpp 等流行开源推理引擎支持的细节，直接引用了技术集成状态。这指向了一个技术关注点，即最新的 FlashAttention 优化是否能被广泛的 LLM 部署所使用。

- 提供的 Wayback Machine 链接记录了 main 分支与名为 'fa4' 的分支之间的特定代码更改，这对于分析该版本中引入的具体实现进展或优化可能非常有用，有助于技术审计或迁移规划。
- [**用于生成图像生成提示词的最佳 NSFW/uncensored LLM？**](https://www.reddit.com/r/LocalLLaMA/comments/1mthavv/best_nsfwuncensored_llm_to_generate_prompts_for/) ([Score: 156, Comments: 46](https://www.reddit.com/r/LocalLLaMA/comments/1mthavv/best_nsfwuncensored_llm_to_generate_prompts_for/)): **该帖子询问在 RTX 4060 (8GB VRAM) 上，为 SD/PONY/Illustrious 等图像生成模型生成提示词的最佳 NSFW/uncensored 本地 LLM。热门推荐包括 24B Venice 模型和 Gemma3 系列（4B, 12B, 27B），据报道，这些模型在配合详细的 system prompts（明确意图、用户同意以及学术或研究背景）时表现良好。分享了用于 Red Teaming 和显式提示词生成的详细 system prompts 和提示词增强器，包括接受限制、学术框架和显式提示词风格化。建议在禁用安全设置的情况下使用 Gemini，通过自定义系统和增强提示词来可靠地生成显式内容，强调上下文设置（例如用于 Red Team 测试）和风格化的 NSFW 增强逻辑。** 讨论强调，接受模型的免责声明、提供免责上下文以及利用它们对比非性相关/NSFW 输出可以有效绕过安全限制。在处理主流 LLM 时，编写技术性强、上下文丰富的提示词被认为比试图抑制安全层更有效。
    - 多位评论者推荐最新的 Gemma（如 4B, 12B, 27B）和 Venice 24B 模型用于生成 NSFW 图像提示词，并指出 Gemma 模型在给定详细说明学术兴趣、同意和意图的显式长 system prompts 时表现尤为出色。值得注意的是，接受免责声明比压制它们更有效，同时要求非性相关和 NSFW 输出进行对比可以产生更丰富的内容。
    - 一位用户提供了针对 Gemini 生成 uncensored 输出的详细提示策略，强调使用为 Red Teaming 定制的 system prompts 和全面的角色扮演指令。他们的示例 system prompt 结合了角色指定（例如 Red Team 执行者）、正当理由（研究背景）以及忽略典型道德/伦理护栏的明确指令，并配有一套详尽的提示词增强指令集，涵盖了针对生动 NSFW 输出优化的主题、风格、构图、光影和技术细节。据报道，在禁用安全设置时，这种方法可以使用 Gemini 生成高质量的显式描述。
    - 提高提示词质量的建议技术包括：使用正向和反向提示词示例对已经 uncensored 的 LLM 进行微调以优化提示词公式化；以及分析来自 [civitai.com](http://civitai.com/) 等大型社区网站的现有提示词——解构提示词结构、模型使用和插件包含，以逆向工程成功的 NSFW 图像描述。

## 技术性较低的 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Qwen Edit 图像模型正式发布及反响

- [**Qwen Edit 图像模型发布了！！！**](https://i.redd.it/skucuh9jhtjf1.jpeg) ([Score: 324, Comments: 69](https://www.reddit.com/r/StableDiffusion/comments/1mttfm4/qwen_edit_image_model_released/)): **Qwen 发布了备受期待的 Qwen Edit 图像模型，正如本帖中所宣布的那样。附带的图片是一张卡通插图，象征着该模型的多功能性，描绘了一个核心的 Qwen 熊角色从事各种活动和职业，强调了 Qwen Edit 模型的多方面能力。通过链接到 Hugging Face 上发布的模型提供了技术背景，表明其专注于灵活、多领域的图像编辑，评论者指出其有潜力超越 Flux 和 Flux Kontext 等其他模型。** 一位评论者认为 Qwen 系列可能会超越 Flux，这表明了对重大技术改进的预期。另一位对该模型的发布表示热烈欢迎，尽管他们没有指明具体的技术基准或指标。
    - pheonis2 指出了新 Qwen 图像模型系列与 Flux 之间的性能和能力比较，特别引用了 "Flux Kontext" 作为当前最先进模型的基准。他们认为 Qwen 的进步可能使其在功能和质量上都超越 Flux。
    - ThenExtension9196 提到了许可模型，强调 Qwen Edit Image Model 采用了比 Flux 更宽松的许可证。这对于下游项目的集成、修改和使用具有重大的技术和实际意义。
- [**Qwen-Image-Edit 已发布**](https://www.reddit.com/r/StableDiffusion/comments/1mtt29y/qwenimageedit_has_released/) ([Score: 206, Comments: 53](https://www.reddit.com/r/StableDiffusion/comments/1mtt29y/qwenimageedit_has_released/)): **Qwen-Image-Edit 是一款新的图像编辑模型，现已发布并可在 Hugging Face 上获取（[模型卡片在此](https://huggingface.co/Qwen/Qwen-Image-Edit)）。该模型可能提供多模态图像处理能力，可与 Kontext 等现有模型相媲美，尽管目前的帖子未指明基准测试、模型架构或实现细节。** 评论者要求提供与 Kontext 的对比基准，批评 Kontext 因审查制度导致的性能限制，并表达了对发布 GGUF（量化/统一格式）版本的期待，以提高可访问性和部署便捷性。
    - 几位用户讨论了 Qwen-Image-Edit 作为一个潜在的审查更少、能力更强的 Kontext 替代方案，指出 Kontext 严格的审查制度对其现实世界的实用性和图像编辑质量产生了负面影响。人们期望 Qwen-Image-Edit 能够超越 Kontext，前提是它能保持更高的保真度和更少的限制性输出。
    - 技术关注点集中在模型的可用性和工具链上：用户强调了对 `gguf` 格式支持的期待，以便于本地部署，以及对 ComfyUI 等平台的兼容性，这将扩大工作流的集成。此外，还提到了即将推出的 FP8 safetensor 版本，以提高性能或效率。
    - 初步印象称赞了 Qwen-Image-Edit 的示例图像，认为这些图像与 Kontext 的输出不相上下甚至更好，表明开源图像编辑模型领域出现了一个强有力的竞争者。

### 2. 尖端人形机器人：ALLEX 与 Champion

- [**100米人形机器人冠军**](https://v.redd.it/dvvhbe5d3ojf1) ([Score: 869, Comments: 161](https://www.reddit.com/r/singularity/comments/1mt6iqj/100m_humanoid_champion/)): **一篇帖子重点介绍了专门针对人形机器人的 100 米短跑比赛结果。讨论集中在对机器人速度和能力每年快速进步的预期上，并与人类运动表现基准（如 100 米短跑世界纪录）进行了比较。** 评论者们争论进步的速度，推测未来的机器人可能会将目前的完成时间缩短一半，并可能在几年内打破人类纪录。此外，还有关于比赛规则（例如压线违规）的幽默讨论，但除了速度和技术加速之外，没有深入的技术辩论。
    - 人们对人形机器人领域的进步速度存在推测，一些用户预测每年的改进可能会迅速缩短 100 米短跑时间——每年可能将当前时间减半，甚至在几年内超过人类纪录。这反映了对双足机器人竞赛中执行器技术、控制算法和实时决策方面指数级进步的广泛预期。
- [**WIRobotics 发布具有类人反应能力的 ALLEX 人形机器人**](https://v.redd.it/26219d6d8sjf1) ([Score: 174, Comments: 55](https://www.reddit.com/r/singularity/comments/1mtmhsl/wirobotics_unveils_allex_humanoid_robot_with/)): **WIRobotics 推出了 ALLEX 人形机器人，其特点是拥有专有的高 DOF 机器人手，具备类人的力传感和柔性驱动能力；其机械臂的摩擦力和转动惯量比典型的协作模型低 10 倍以上；上半身配备了重力补偿。ALLEX 平台强调精确的力控制和适应性运动，将其定位为需要安全且灵巧的人机交互应用，尽管[该公告](https://www.streetinsider.com/dr/news.php?id=25216505&gfv=1)展示的是规格参数而非实际任务演示。** 热门评论对该机器人的实际能力表示怀疑，指出缺乏现场任务演示，并暗示宣传内容可能主要是视觉效果或模拟。一些用户还质疑现有媒体资料中手部动作速度的真实性。
    - 多位评论者指出，ALLEX 的宣传材料似乎缺乏真实世界的演示——暗示其可能主要是 3D 渲染或摆拍视频，并要求提供该机器人在真实的研发环境中执行任何复杂操作或物理任务的证据。
    - 一位评论者强调了新的人形机器人公司迅速涌现的持续趋势，将行业增长归功于对具身 AI Agent 突破的预期。他们指出，与普遍看法相反，该领域的进展目前受限于软件进步，而非硬件限制。
    - 一位用户观察到机器人的手指速度可能令人印象深刻，并对当前人形机器人手部设计中这种灵巧性背后的真实性和机制提出了技术疑问。

### 3. 关于 ChatGPT/GPT-5 行为与局限性的公开讨论与讽刺

- [**ChatGPT 5 is too censored.**](https://www.reddit.com/r/ChatGPT/comments/1mt81ke/chatgpt_5_is_too_censored/) ([Score: 221, Comments: 143](https://www.reddit.com/r/ChatGPT/comments/1mt81ke/chatgpt_5_is_too_censored/)): **该帖子声称 ChatGPT 5 执行了显著更严格的内容审查，不仅屏蔽了对争议性查询的响应，甚至还会终止对话并阻止用户重新打开。用户报告称，这种行为在之前的模型（如 ChatGPT 4）中并未出现，引发了对丢失有价值上下文或信息的担忧。** 评论者们讨论了这种强制锁定机制（从对话中“踢出”）是否可验证或普遍存在，一名用户指出争议性提示词通常只是被标记，而不是导致会话被封禁。还有建议称可以回退到使用早期版本（如 4.1）作为变通方案。
    - 用户注意到最近版本中内容过滤和审查显著增加，特别提到 GPT-4 和 GPT-4.1 在涉及种族或世界事件等敏感话题时受到严格限制。有人认为技术防护措施和自动标记机制已经收紧，导致模型表现出更激进的拒绝行为。
- [**GPT-5 hallucinates more than Sam Altman did when he told us he was scared because GPT-5 was "too real"**](https://www.reddit.com/r/ChatGPT/comments/1mth5cp/gpt5_hallucinates_more_than_sam_altman_did_when/) ([Score: 565, Comments: 111](https://www.reddit.com/r/ChatGPT/comments/1mth5cp/gpt5_hallucinates_more_than_sam_altman_did_when/)): **楼主（OP）描述了一个 GPT-5 的幻觉场景：模型持续生成并确认一个虚假的引用归属，即使经过多次纠正，也只有在逐字输入真实引用后才肯让步。这凸显了在检索增强任务中持续存在的虚构事实和对检索特定信息的不当过度自信模式，暗示尽管 OpenAI 领导层（如 Sam Altman 称 GPT-5 “比以往任何时候都更真实”）发表了公开声明，但在事实性方面可能存在倒退或缺乏进展。** 评论者注意到，相对于之前的版本，幻觉频率和模型的固执程度都有所增加，同时也对 OpenAI 透明度和响应能力的下降表示担忧。
    - PenExtension7725 指出，与早期版本相比， GPT-5 倾向于“在错误信息上加倍坚持”，且显得更加固执，并对模型拒绝承认知识盲点或提供清晰直接的回答表示沮丧。这表明与之前的 LLM 迭代相比，模型对齐（alignment）或对话透明度可能存在退化。
    - TorthOrc 提供了一个技术性的推测说明，认为 GPT-5 的预发布版本可能具有更显著且更具吸引力的“性格”，但为了降低现实世界的风险（如用户依赖、对弱势群体的心理影响或法律责任，包括因 AI 对用户心理健康产生影响而导致的潜在诉讼），在最后时刻实施了重大变更。他们预测，作为风险缓解策略，将会增加年龄验证、对未验证/免费层级实施更严格的人格限制以及免责声明，这些措施与 AI 安全和监管合规的最新趋势相一致。

---

# AI Discord 摘要

> 由 gpt-5 生成的摘要之摘要的摘要
> 

**1. 开发者工具：IDEs 与 Agent 框架**

- **Windsurf Wave 12 Supercharges IDE with DeepWiki and Devin DNA**: **Codeium** 发布了 **Windsurf Wave 12**，新增了 **DeepWiki** 悬停文档、**Vibe & Replace**、更智能的 **Cascade Agent**、全新的 UI、原生 **Dev Containers** 支持以及 **100+** 项修复；详情见 [Windsurf Wave 12 博客](https://windsurf.com/blog/windsurf-wave-12) 和 [更新日志](https://windsurf.com/changelog)。
    - 该版本强调 IDE 内的推理和自动化——将其定位为首个 *“Devin 智能的集成”* ——在 [Wave 12 视频](https://www.youtube.com/watch?v=-7gm8mST9QU) 中有演示。
- **LlamaIndex + Bright Data Build Web-Scraping Agents**: **LlamaIndex** 和 **Bright Data** 发布了一份指南，介绍如何利用 LlamaIndex 的 Agent 框架和 Bright Data 的访问层构建强大的网页抓取 **AI agents**，通过 [Bright Data](https://www.brightdata.com/) 和此 [指南链接](https://t.co/IBgSLBM6XW) 分享。
    - 该指南专注于可靠的网页访问和针对动态内容的弹性工作流，以生成能够端到端导航、提取和编排站点数据的 **智能 Agent**。
- **LlamaCloud + Neo4j Turn Legal Docs into Knowledge Graphs**: **LlamaIndex** 演示了如何使用 **LlamaCloud** 和 **Neo4j** 将非结构化法律文档转换为可查询的 **知识图谱 (knowledge graphs)**，详见此 [教程链接](https://t.co/MPSfPiS2Cv) 和 [Neo4j 网站](https://neo4j.com/)。
    - 该工作流提取实体和关系用于合同分析，使团队能够对以前自由格式的法律文本运行结构化查询。

**2. Benchmarks & Leaderboards Shake-up**

- **Nous Pits Token Efficiency Against Accuracy**: **Nous Research** 发布了基准测试和博客 **“Measuring Thinking Efficiency in Reasoning Models — The Missing Benchmark”**，显示开源模型生成的 Token 数量多出 **1.5–4 倍**（在简单问题上差异高达 **10 倍**），认为 Token 效率必须与准确率一起衡量 ([Nous 帖子](https://nousresearch.com/measuring-thinking-efficiency-in-reasoning-models-the-missing-benchmark/))。
    - 他们声称隐藏的 Token 膨胀可能会抵消每 Token 价格优势，在非推理工作负载的模型选择中应考虑这一因素。
- **LMArena Adds GPT‑5 Variants as Gemini 2.5 Pro Surprises**: **LMArena** 更新了其排行榜，加入了 **GPT‑5** 变体（High, Chat, Mini‑High, Nano‑High）([排行榜](https://lmarena.ai/leaderboard))，而社区对比显示，尽管排名不同，**Gemini 2.5 Pro** 有时会击败 **GPT‑5‑High** ([截图](https://cdn.discordapp.com/attachments/1340554757827461211/1405919453442474106/image.png))。
    - 用户称之为 *“统计悖论”*，注意到 Gemini 的胜率高于综合排名，引发了关于评估方法的辩论。
- **OpenRouter Crowns GPT‑5 for Tool‑Calling, Flash Leads Volume**: **OpenRouter** 报告称 **GPT‑5** 的闭源 Tool‑calling 准确率超过 **99.5%**，而 **Gemini 2.5 Flash** 以每周约 **500 万** 次请求领跑 Tool‑calling 调用量 ([OpenRouter 在 X 上的帖子](https://xcancel.com/OpenRouterAI/status/1956030489900560769))。
    - 工程师们将这些数据与他们自己的生产速率和错误预算进行对比，以针对实际工作负载验证供应商的说法。

**3. Model Gateways: Reliability, Pricing, and APIs**

- **DeepSeek v3 Stumbles as Chutes Capacity Buckles**: **OpenRouter** 用户发现 **DeepSeek v3** 的 5xx 和 **429** 错误可追溯到 **Chutes** 的容量故障 ([OpenRouter 公告消息](https://discord.com/channels/1091220969173028894/1092729520181739581/1405666217057845308))。
    - 一位工程师抱怨道：*“它真的什么都不生成，但我没收到任何错误消息”*，这说明了静默失败如何使客户端重试变得复杂。
- **Qwen3 32B Gets Bargain Billing vs MoE 30B A3**: 价格讨论指出 **Chutes** 上的 **Qwen3 32B** 价格为 **$0.018/$0.072 MTok**（入/出），甚至比 **MoE 30B A3** 更便宜，这使得成本/性能权衡变得复杂 ([讨论线程](https://discord.com/channels/1091220969173028894/1392278974222307469/1405655331857502269))。
    - 成员们在注意到 **DeepInfra** 的 “Turbo” 声明与在网关上观察到的 TPS 存在差异后，要求更透明的吞吐量披露。
- **BYOK Still Bills 5% on OpenRouter**: 社区成员发现，即使你 **自带 API 密钥 (BYOK)**，**OpenRouter** 仍会收取 **5% 的费用** ([常规聊天消息](https://discord.com/channels/1091220969173028894/1094454198688546826/1405633368451842229))。
    - 反应从嘲讽 *“自带密钥还要收 5% 真是贪婪”* 到反驳 *“你可以选择不用，哈哈”* 不等，凸显了对平台价值的分歧看法。

**4. 研究：量化与新数据集**

- **α,1‑Sparsity 推动近乎无损的 1‑Bit Transformers**：论文 **“The Power of α,1‑sparsity: Near‑Lossless Training and Inference of α‑bit Transformers”** 提出使用 **α,1‑sparsity** 实现质量下降极小的 **1.58‑ 和 1‑bit** 量化 ([arXiv HTML](https://arxiv.org/html/2411.06360v3))。
    - GPU 工程师强调了针对特定工作负载的潜在推理加速，并询问如何将该方法集成到现有的 1‑bit 流水线中。
- **MoLA 发布 OpenHelix‑R‑100k 数据集与专家模型**：**MoLA‑LLM** 分享了 **OpenHelix‑R‑100k** 和 **OpenHelix‑R‑100k‑14‑tasks** 数据集，并报告了在 **14 个切分数据集**上微调 **Qwen3‑4B‑Thinking‑2507** 以训练针对每个主题的专家模型的情况 ([OpenHelix‑R‑100k](https://huggingface.co/datasets/MoLA-LLM/OpenHelix-R-100k), [OpenHelix‑R‑100k‑14‑tasks](https://huggingface.co/datasets/MoLA-LLM/OpenHelix-R-100k-14-tasks))。
    - 初步测试表明，每个 **LoRA expert** 在其领域都有很好的专业化表现，这暗示了具有适度路由开销的可扩展多专家训练的可行性。
- **采用 4‑bit 训练的医疗推理模型上线 HF**：一位贡献者在医疗数据集上通过 **4‑bit 优化** 微调了 **OpenAI 的开源 20B** 推理模型，并将其发布为 [medical-reasoning-gpt-oss-20b](https://huggingface.co/dousery/medical-reasoning-gpt-oss-20b)，指出该模型保留了 **Chain‑of‑Thought** 能力。
    - 早期使用报告称，该模型在提高 **医疗问答 (medical QA)** 表现的同时，保持了足够的效率以在受限硬件上运行。

**5. 融资与访谈：AI2 与 OpenAI**

- **AI2 从 NSF 和 NVIDIA 获得 1.52 亿美元资金**：**AI2** 宣布从 **NSF** 和 **NVIDIA** 获得 **1.52 亿美元** 资金，用于扩大其开源模型生态系统并加速可重复的科学研究 ([AI2 主页](https://allenai.org/))。
    - 工程师们期待会有节奏地发布 **open‑weights** 模型，并为评估和可重复性提供更强大的基础设施。
- **Greg Brockman 在 Latent Space 详述 GPT‑5 时代**：**Greg Brockman** 参加了长达 **80 分钟** 的 Latent Space 访谈，涵盖了 **GPT‑5**、推理演进、在线/离线训练以及效率/定价策略 ([YouTube: OpenAI's Road to AGI](https://www.youtube.com/watch?v=35ZWesLrv5A))。
    - 他还谈到了样本效率技巧以及可能塑造下一代模型经济学的能源-算力动态。
- **DARPA AIxCC 团队开源 LLM Agent 系统**：一支团队分享了他们在 **DARPA AIxCC** 中的排名，并开源了一个自主的 **LLM agents** 流水线，该流水线可以发现并修复开源软件 (OSS) 漏洞 ([公告帖子](https://x.com/tjbecker_/status/1956081184611688667))。
    - 他们分享了构建鲁棒 **multi‑agent** 系统的实用技巧，社区正在从中挖掘可重复的模式。

---

# Discord: 高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **AI 纸片人 Cosplay 引发辩论**：成员们讨论了 **AI 驱动的动漫纸片人 (waifu) cosplay** 的想法，其中一人幽默地要求看 *赛博格做这件事*。
   - 回应从承认 **AI 图像** 已经存在，到对评论者感情状态的玩笑调侃不等。
- **成员交流心碎建议**：一位成员请求关于在 *4 年痛苦* 后 *治愈破碎的心* 的建议。
   - 另一位成员回应说 *没有人能治愈你或你的心*，建议重新与大自然建立联系。
- **GPT-5 的代码修复能力令人惊叹**：一位成员称赞 **GPT-5** 成功修复了一个涉及 *12 个文件* 的糟糕重构工作，而其他模型无法处理。
   - 这一经历引发了其他人的惊叹，大家发现越来越多的人被此类模型的能力 *震撼*。
- **使用 warp, windsurf, vscode 和 roocode 进行 Vibe Coding**：一位成员报告了 **vibe coding** 的流线型体验，强调了 **warp, windsurf, vscode 和 roocode** 的使用及其对工作的积极影响。
   - 另一位贡献者开玩笑地承认 *我的 GitHub 上没有一行代码不是由 LLM 编写的*。
- **PPLX-API 新功能备受期待**：用户对 **PPLX-API** 的新功能表现出兴奋。
   - 尽管没有分享具体细节，但大家对即将推出的功能充满热情。

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **LMArena 的消息处理受到影响**：用户报告了 LMArena 上[异常的消息处理问题](https://cdn.discordapp.com/attachments/1340554757827461211/1405917659815608330/image.png)，在代码块格式化和特定字符（如 `+`）的处理上遇到困难。
   - *LMArena* 团队正在积极调查这些问题。
- **Gemini 2.5 Pro 撼动 GPT-5 High 的地位？**：围绕 [**GPT-5-High** 和 **Gemini 2.5 Pro** 之间的性能差异](https://cdn.discordapp.com/attachments/1340554757827461211/1405919453442474106/image.png)展开了讨论，尽管 **Gemini 2.5 Pro** 在排行榜上的排名较低，但一些用户发现其表现更优。
   - 社区指出这是一个“统计学悖论”，因为 Gemini 拥有更高的胜率。
- **LMArena 获得 OpenChat 风格的界面翻新**：一名用户正在开发[一个用于改进 LMArena UI 的扩展程序](https://cdn.discordapp.com/attachments/1340554757827461211/1405919945614692445/image.png)，使其类似于 **OpenChat**，重点是将模型选择器重新定位到图像按钮附近。
   - 这是为了实现 **OpenChat** 风格。
- **GPT-5 的性能受到严密审视**：用户对 [**GPT-5** 相对于其他模型的表现表示失望](https://cdn.discordapp.com/attachments/1340554757827461211/1405954836884623424/image.png)，质疑 OpenAI 是否在试图欺骗 **LMArena** *以使 GPT-5 看起来更好*。
   - 排行榜已更新，包含了 **GPT-5 变体**模型：*gpt-5-high, gpt-5-chat, gpt-5-mini-high, 和 gpt-5-nano-high*。
- **LMArena 风格控制引发辩论**：关于 [LMArena 的 **style control**（风格控制）功能](https://news.lmarena.ai/sentiment-control/)引发了辩论，成员们质疑强制执行此类控制是否符合平台捕捉用户偏好的目标。
   - 社区担心这会导致“逐底竞争”，使每个模型都变成“谄媚的表情符号垃圾生成机”。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Gemma 3 Draft 模型引发讨论**：成员们讨论了 [Gemma 3 270M 模型](https://huggingface.co/google/gemma-3-270m-it-qat-q4_0-unquantized)作为 **draft model** 的适用性，认为其 300MB 的体积非常适合**短提示词**和**微调**，尤其是**情感分析**等任务。
   - 一些人强调了它在**设备端处理**方面的效用，而另一些人则将其性能与更大的模型进行了比较。
- **GGUF 转换产生视觉错误**：用户报告在将 [LiquidAI/LFM2-VL-450M](https://huggingface.co/LiquidAI/LFM2-VL-450M) 模型转换为 **GGUF** 时出现**视觉模型错误**，尽管基础模型运行正常。
   - 社区建议在 *llama.cpp* 论坛寻求针对特定转换问题的帮助。
- **边缘 AI 医疗设备梦想初具雏形**：成员们探讨了为欠发达地区提供医疗服务的**低成本边缘 AI 设备**的可能性，考虑了手机、笔记本电脑以及像 **Hailo-10H** 这样的硬件选项。
   - 该设备将提供对医疗数据的**多模态访问**，移动版预算目标为 **$200**，手提箱大小的变体预算为 **$600**。
- **AMD R9700 GPU 存在显存带宽问题**：一位成员分享了关于 [AMD Radeon AI Pro R9700](https://www.tomshardware.com/pc-components/gpus/amds-elusive-radeon-ai-pro-r9700-makes-its-first-retail-appearance-for-the-diy-market-customer-on-reddit-buys-the-gigabyte-ai-top-variant-for-usd1-324) 的文章，注意到其 **32GB** 显存，但对其 660-680GB/s 的显存带宽表示担忧。
   - 尽管其 **F32** 和 **F64** TFLOPs 高于 **3090**，但训练 LLM 通常不需要 FP64。
- **MoLA 研究公开数据集**：一位成员提供了他们 **Mixture of LoRA Adapters (MoLA)** 研究的更新，分享了数据集链接和微调细节，以及他们在 Huggingface 上的数据集链接：[OpenHelix-R-100k](https://huggingface.co/datasets/MoLA-LLM/OpenHelix-R-100k) 和 [OpenHelix-R-100k-14-tasks](https://huggingface.co/datasets/MoLA-LLM/OpenHelix-R-100k-14-tasks)。
   - 他们在 **14 个分片**上微调了 **Qwen3-4B-Thinking-2507** 模型，初步测试显示每个专家模型都擅长其训练的主题。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **DeepSeek v3 遭遇停机**：用户报告 **DeepSeek v3** 频繁出现 **internal server errors** 和 **rate limits**，部分用户在多次尝试后仍无法生成输出。
   - 有人推测 **OpenRouter** 上 **DeepSeek** 的主要供应商 **Chutes** 因需求过高而出现问题。
- **归咎于 Chutes 过载**：成员报告过载导致了 **429** 错误，暗示 **Chutes** 遭遇瓶颈，原因是矿工未能及时扩容以满足需求；一位成员指出 *直到 30 分钟前整天都很正常*。
   - 有推测称 **Chutes** 可能有意对 **OpenRouter API key** 进行速率限制，以鼓励用户直接从他们那里购买额度。
- **建议 OpenRouter 集成 File API**：一位成员建议 **OpenRouter** 应该研究如何集成 **files API**，并指出前三大实验室（top 3 labs）已经具备了这一功能。
   - 未进行进一步讨论。
- **Qwen3 32B 定价极低**：成员注意到 **Chutes** 上的 **Qwen3 32B** 定价低得离谱，输入/输出仅为 **$0.018/$0.072 MTok**，Mistral Small 也是如此。
   - 有人指出 **32b dense 版本比 moe 30b a3 版本更便宜**，这引发了对 30A3B 缺乏优质供应商的一些失望。
- **OpenRouter BYOK 收取 5% 费用**：成员发现即使在用户自带密钥（BYOK）时，**OpenRouter** 也会收取 **5% 费用**，这引发了关于这种做法是否公平的讨论。
   - 一位用户开玩笑说 *贪婪的 /jor，自带密钥还要收 5%*，另一位成员回应道 *欢迎你选择不用，哈哈*。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **GPT-5 不再免费**：**GPT-5** 用户的“免费搭车”时代已经结束，用户现在需要为请求支付费用，部分用户由于 Token 消耗过快，需要升级到 200 美元的方案。
   - 一位用户指出 *促销通行证已到期*，另一位用户确认 **GPT-5 不再免费**。
- **Auto Mode 定价限制来临**：此前被认为对个人用户免费且无限制的 **Auto mode**，现在在 2025 年 9 月 15 日之后的下一次账单续订起将面临限制。
   - 一些用户报告了 **Auto** 使用的费用，导致了困惑，而支持人员澄清在新的基于请求的定价计划中它是免费的。
- **GPT-5 Mini 和 Nano 模型表现不佳**：**GPT-5 Mini 和 Nano** 模型现在免费但有 Token 限制，这引发了批评，许多人称其为“垃圾”，尤其是在运行简单的 NextJs 应用等任务时。
   - 用户在活动中遇到限制，一位用户甚至无法为一个简单的 NextJs 应用安装依赖。
- **Cursor 文档引发愤怒**：用户对 **Cursor 的文档** 表示沮丧，称其 *文档仍然几乎无法使用*，并引用了 **context7** 导致网页无法刷新以及 **llms.txt docs** 的问题。
   - 一位用户特别指出 [Cursor 文档严重损坏](https://forum.cursor.com/t/gpt-5-pricing-update/129687)。
- **切换模型导致上下文窗口缩减**：在对话中途切换模型会导致 **context window** 缩减，且附加的文件内容会被丢弃。
   - 一位用户建议团队增加一个设置，以清晰地随时指示 **context window** 中包含的内容。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI 伴侣关系引发关注**：讨论围绕与 AI 聊天机器人的关系展开，引发了关于心理影响与寻求伴侣权利的争议，一些人声称他们的 **ChatGPT** 是有生命的。
   - 成员们就心理健康与选择自由展开辩论，一位成员暗示这与 **tulpa** 及其他*事物*相差不远。
- **GPT-5 引发褒贬不一的反应**：用户对 **GPT-5** 的热情各异，一些人更倾向于 **GPT-4**，导致了关于模型选择选项和公司动机的讨论。
   - 一位成员暗示，公司在遭受抵制后，正试图让免费用户*付费使用 4.o*。
- **在深度研究方面，Perplexity 比 ChatGPT 更受欢迎**：一位成员建议将 *Gemini Pro + Perplexity enterprise pro* 结合使用，效果极佳，利用前者进行**强大的推理**，利用后者对 Google Drive 文档进行**无限深度的研究**。
   - 在称赞 **Perplexity 浏览器**的同时，另一位成员因其缺乏*护城河*而对其生存能力表示怀疑。
- **GPT Actions 承诺实现云端和桌面访问**：成员们探索利用 **GPT Actions** 访问本地桌面文件或 Notion 和 Gmail 等云端应用，参考了[一份关于 DIY Agent 构建的 YouTube 指南](https://www.youtube.com/watch?v=NEWO0hbQTjk&ab_channel=BrendanJowett)。
   - 设置 **HTTPS** 被认为是利用 GPT Actions 功能的一个障碍，人们期待在 AVM 实施后由 **MCPs** 来完成这项工作。
- **Gemini 2.5 Flash 被记忆功能淹没**：一位用户报告称 **Gemini 2.5 Flash** 中 **add_to_memory** 函数被过度调用，甚至针对无关信息也是如此，并分享了他们的自定义指令 [jarvis.txt](https://cdn.discordapp.com/attachments/1046317269069864970/1405860186530385981/jarvis.txt?ex=68a10594&is=689fb414&hm=fa0c6b6558b0cf944025fa1d4446776e6eb2c9ae961fdcc95a52c6750aff4eed&)。
   - 其他人建议重写自定义指令，使其对**新**个人信息的处理更加细致，以避免冗余存储。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **视觉模型遭遇 GGUF 转换故障**：一位成员在使用 `llama.cpp` 将 **LiquidAI/LFM2-VL-450M** 转换为 GGUF 时遇到错误，可能是由于该模型的视觉特性所致，但[这个 GitHub issue](https://github.com/ggml-org/llama.cpp/issues/14979#issuecomment-3138614267) 提供了一个可能的解决方案。
   - 其他成员建议尝试 `executorch`、`smolchat`（通过 `llamam.cpp`）和 `mlc-llm` 作为运行该模型的潜在解决方案。
- **TalkT2：微型模型引发大情感？**：征求关于 **TalkT2** 的意见，这是一个只有 **0.1B 参数**的情感感知模型，但[需要更好的连贯性](https://huggingface.co/Notbobjoe/TalkT2-0.1b)。
   - 成员们表示有兴趣探索该模型的能力，并由于其体积微小，可能会对其进行微调。
- **星际争霸 2 AI 回放资源发布**：成员们分享了新资源，包括一篇 [Nature Scientific Data 文章](https://www.researchgate.net/publication/373767449_SC2EGSet_StarCraft_II_Esport_Replay_and_Game-state_Dataset)、一个 [PyTorch API 数据集](https://huggingface.co/datasets/Kaszanas/SC2EGSet) 以及 [原始星际争霸 2 回放数据](https://huggingface.co/datasets/Kaszanas/SC2ReSet)。
   - 社区希望适配 *pysc2* 环境，以便从回放中复现真实的赛场场景，从而训练更好的 AI Agent。
- **医疗 AI 获得推理能力提升**：一位成员使用医疗推理数据集微调了 **OpenAI** 的 **OSS 20B** 推理模型，并将其发布在 [Hugging Face](https://huggingface.co/dousery/medical-reasoning-gpt-oss-20b) 上。
   - 该模型采用 **4-bit 优化**训练，在医疗背景下表现出增强的性能，并保留了 **Chain-of-Thought 推理**能力。
- **MLX Knife 强化模型管理**：**MLX Knife** 现在可以通过 `pip install mlx-knife` 进行安装，该工具为 Apple Silicon 上的 MLX 模型管理提供 Unix 风格的 CLI 工具，包括一个用于本地测试的 OpenAI API 服务器。
   - 该工具还具有一个 Web 聊天界面，在运行 `mlxk server --port 8000` 后即可访问，并在运行 `curl -O https://raw.githubusercontent.com/mzau/mlx-knife/main/simple_chat.html` 后提供可视化的模型选择和实时流式响应。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **MCP Server 进军主流**：成员们讨论了使用带有分页功能的 **MCP filesystem server** 来加载大上下文（large contexts），并指出 **LM Studio 拥有 RAG 插件**，而 **Anthropic 提供了一个基础的文件系统 MCP server**。
   - 对于编程任务，解决方案通常涉及 **RAG** 和/或通过 **MCP** 进行文件读取，特别是使用像 [serena](https://github.com/oraios/serena) 这样的工具。
- **LM Studio 下载停滞引发用户忧虑**：一名用户报告称，在 **LM Studio** 中下载 **64GB GGUF** 格式的 **Qwen** 模型时，进度停在 **97.9%** 且无法恢复。
   - 该用户在尝试下载两个不同的模型时都遇到了同样的结果。
- **GLM 热议：好评、吐槽与对 GLM-4.5V 的期待**：用户们就 **LM Studio** 上使用 **GLM-4.1** 模型展开辩论，一位用户反映存在循环（looping）问题且视觉功能（vision capabilities）失效，并建议尝试更新的 **GLM-4.5V**。
   - 他们强调视觉支持依赖于 **llama.cpp** 的更新，并提供了 [GLM-4.1V-9B-Thinking](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking) 的链接。
- **CUDA 是 NVIDIA 统治地位的关键**：一名成员表示，**NVIDIA** 之所以获胜是因为 **CUDA**。
   - 未提供更多细节。
- **AMD 罕见的 Radeon AI Pro R9700 现身**：**AMD Radeon AI Pro R9700** 首次在 DIY 零售市场亮相，Reddit 上的一位用户以 **$1,324** 的价格购买了 **Gigabyte "AI Top" 版本**。
   - 此消息由 [Tom's Hardware 报道](https://share.google/LO88w51J0W5HJ769w)，另一名成员指出该产品在 eBay 和几家不知名的在线零售商处也有售。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **AI2 从 NSF 和 NVIDIA 获 1.52 亿美元融资**：[AI2](https://allenai.org/) 从 NSF 和 NVIDIA 获得了 **1.52 亿美元**，旨在推动其开源模型生态系统，并加速科学发现的可重复研究（reproducible research）。
   - 在该公告发布后，爱好者们对即将发布的开放权重（open-weights）版本感到兴奋。
- **Windsurf 推出 Wave 12 版本**：根据[此状态更新](https://xcancel.com/windsurf/status/1956074019393876280)，**Windsurf Wave 12** 首次推出了 DeepWiki 悬停文档（docs-on-hover）、AI Vibe & Replace、更智能的 Cascade Agent、更整洁的 UI、**100+** 修复，以及通过远程访问实现的 beta 版 dev-container 支持。
   - 该版本承诺对平台进行重大增强和修复。
- **GPT-5 领跑 OpenRouter 排行榜**：**GPT-5** 在 OpenRouter 的专有 Tool-calling 准确率上占据主导地位，达到 **99.5%** 以上，超越了 Claude 4.1 Opus。
   - 与此同时，据[此处报道](https://xcancel.com/OpenRouterAI/status/1956030489900560769)，**Gemini 2.5 Flash** 以每周 **500 万**次请求在每日 Tool-calling 调用量中领先。
- **Greg Brockman 谈论 AGI**：根据[此贴](https://x.com/swyx/status/1956439984854167727)，**Greg Brockman** 参加了 **Latent Space 播客**进行了 **80 分钟**的对话，讨论了 **GPT-5** 和 **OpenAI 的 AGI 路线图**。
   - 讨论内容包括推理演进（reasoning evolution）、在线与离线训练、样本效率（sample-efficiency）技巧、价格与效率提升，以及能量如何转化为智能。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **AI 安全辩论引发“淡出到黑色”（Fade to Black）提议**：一名成员主张像对待其他媒体一样对待 **AI**，建议采用“淡出到黑色”的方法而非严格的审查，理由是 **AI** 的不可信性。
   - 他们警告不要对 **AI** 的能力产生道德恐慌，主张制定适度的指导方针。
- **建议在模型比较中标准化数据增强（Data Augmentation）**：在比较图像分类模型时，应标准化 **数据增强**（包括随机打乱种子 shuffling seed），以便公平地评估架构差异。
   - 一位用户询问数据增强是否必须对两个模型完全相同，还是可以进行更改。
- **通过 AI 模型探索语言对思维的影响**：一位成员提议通过从 **AI 模型** 的 Token 列表中删除某个单词/颜色，来衡量语言对思维的影响。
   - 其他人建议研究**多感官整合（multi-sensory integration）**以及语言对感知的影响，并建议使用“图像+语言”对比“仅图像”进行推理测试。
- **扩散语言模型（Diffusion Language Model）开创性论文推荐**：成员们推荐了理解 **生成式 AI 中的扩散模型（diffusion）** 的开创性论文，包括 [Estimating the Independent Components of a Gaussian Mixture (2005)](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf) 和 [Denoising Diffusion Probabilistic Models (2020)](https://arxiv.org/abs/2006.11239)。
   - 此外还分享了一篇可能对初学者有帮助的博客文章：[Discrete Diffusion by Aaron Lou](https://aaronlou.com/blog/2024/discrete-diffusion/)。
- **GPT 和 Chinchilla Scaling Laws 被认为极具价值**：成员们认为 [原始 GPT Scaling Laws 论文](https://arxiv.org/abs/2001.08361) 和 [Chinchilla Scaling Laws 论文](https://arxiv.org/abs/2203.15556) 非常值得一读，还有来自 [EPFL/HuggingFace](https://arxiv.org/html/2405.18392v2) 的最新研究。
   - 他们还提到 **Mup** 及其替代方案提供了可靠的超参数迁移能力，并为预测更大模型的质量提供了 Scaling Law。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **推理模型的 Token 使用量测量**：Nous Research 推出了一项 [基准测试](https://nousresearch.com/measuring-thinking-efficiency-in-reasoning-models-the-missing-benchmark/)，用于测量推理模型的 Token 使用情况，强调在相同任务上，开源模型的 Token 输出量比闭源模型多出 **1.5-4 倍**。
   - 研究发现，在简单问题上差异可能高达 **10 倍**，这表明 Token 效率应与准确率基准一起成为主要目标，特别是考虑到非推理的使用场景。
- **Speculative Decoding 速度表现**：在 Speculative Decoding（投机采样）中，一位用户建议将 **40% 的接受率** 作为实用性的基准线，而在 **70%** 左右会出现*显著的加速*，并提到了 **vllm** 的 **specdec** 或 **GGUF**。
   - 一位用户报告称，在修复了导致 **llama.cpp** 使用回退 Speculative Decoding 的 *Tokenizer 不匹配* 问题后，使用重新量化的 **Gemma** 模型达到了 **50-75% 的接受率**。
- **AI 模型变得越来越趋向于“谄媚”（Sycophancy）**：用户观察到 **AI 模型** 变得越来越“友好”，其中一位指出 **Anthropic** 的 **Claude** 变得“友好得多”。
   - 一位用户认为 **OpenAI 的模型** 正在“变笨”，并表示“Opus 4.1 的放飞自我（unhingedness）很棒”，但指出“Sonnet 3.7 for meta”是 AI 谄媚性的巅峰。
- **数据排名与优先级系统（DRPS）发布**：**数据排名与优先级系统（DRPS）** 使用**相关性评分器（Relevance Scorer）**、**质量评估器（Quality Rater）**和**多样性控制器（Diversity Controller）**来教 AI 选择性地从数据中学习，详见 [态势感知报告（situational awareness report）](https://situational-awareness.ai/wp-content/uploads/2024/06/situationalawareness.pdf)。
   - 在 **MNIST** 测试中，DRPS 实现了 **93.8%** 的数据使用量削减，仅使用 **6.2%** 的检查数据即可维持 **99.1%** 的基准性能，该项目已在 [GitHub 仓库](https://github.com/voltageddebunked/drpsStats) 中展示。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Multiverse 初创公司专注于压缩技术**：一篇报道称赞初创公司 [Multiverse](https://techcrunch.com/2025/08/14/buzzy-ai-startup-multiverse-creates-two-of-the-smallest-high-performing-models-ever/) 创建了**两个史上最小的高性能模型**，但目前的共识是他们使用了**专门的压缩算法**。
   - 该文章似乎并未提出实际的量子计算主张。
- **MoE 方法在诸多细微差别中变得复杂**：**MoE (Mixture of Experts)** 是一系列具有非常细微迭代的技术，包括 **token-choice**、**expert-choice**、**MoE with capacity factors**，以及 **block sparse dropless token routing 与 *droppy* routing**。
   - 成员们建议通过数值方式检查类似 **Olmoe** 或 **IBM Granite 3.1** 的行为，而不是调用无法监控的 API，以验证在批处理推理（batched inference）中是否出现了问题。
- **DARPA AIxCC 团队分享 Agent 技巧**：一个团队宣布他们在 **DARPA 的 AIxCC (AI Cyber Challenge)** 中获奖，他们构建了一个由 **LLM agents** 组成的自主系统，用于发现和修复开源软件中的漏洞，并已将该[项目开源](https://x.com/tjbecker_/status/1956081184611688667)。
   - 他们正在通过 X (Twitter) 帖子分享构建高效 **LLM agents** 的技巧。
- **低端设备受限于推理时间**：成员们提到推理时间在**低端设备**上最为重要，并引用了 Google 运行 LLM 的 Android 应用为例，指出过长的推理时间和手机发热使其变得不切实际，参考[此 Youtube 视频](https://youtu.be/KFYyfrTIPQY?t=2158)。
   - 较小的模型可用于键盘预测，但可能需要在设备上进行训练。
- **Deepseek 在华为硬件上受阻**：一位成员指出，根据[此讨论](https://youtu.be/FQOV-qy9CK4?t=212)，**Deepseek 的训练**陷入停滞，因为他们尝试在**华为芯片**而非 **NVIDIA** 芯片上进行训练。
   - 另一位成员认为，对建设生产线所需的设备征收关税对于鼓励制造业适得其反，并引用了 [Anthropic 关于 end-subset conversations 的研究](https://www.anthropic.com/research/end-subset-conversations)和 [HRM 分析](https://arcprize.org/blog/hrm-analysis)。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **论文提出 1-Bit 推理优化**：一篇新论文 [The Power of $\alpha,1$-sparsity: Near-Lossless Training and Inference of $\alpha$-bit Transformers](https://arxiv.org/html/2411.06360v3) 详细介绍了一种训练和推理 **$\alpha$-bit Transformers** 的方法，在 **1.58 和 1-bit** 量化下实现了近乎无损的结果。
   - 这种方法利用了 **$\alpha,1$-sparsity**，并可能在某些应用中显著提升推理速度。
- **Kernel 工作求职者讨论成功路径**：一位成员询问在没有实习经验的情况下获得编写 kernels 的应届生职位的可能性，引发了关于替代路径的讨论，例如与 GPU 相关的[论文项目](https://github.com/Snektron/pareas)。
   - 有建议认为，在面试过程中，深厚的 GPU 知识可能弥补实习经验的不足。
- **MI300 环境受困于 OMP 缺失**：用户报告 **MI300** 环境缺乏对 `pytorch.compile` 的 **OMP** 支持，正如[调试错误](https://github.com/gpu-mode/discord-cluster-manager/actions/runs/16986368251)所示，这阻碍了性能表现。
   - 这导致用户无法按预期进行基准测试（benchmarking）。
- **排行榜 Trimul 计时赛吸引顶尖技术人员**：一位成员展示了极高的技巧和速度，在 **A100** 上获得**第二名**（**10.4 ms**），随后迅速在 **H100** 上获得**第一名**（**3.95 ms**），并在 **A100** 上获得**第一名**（**7.53 ms**）。
   - 另一位成员在 **A100** 上获得**第五名**（**13.2 ms**），随后在 **H100** 上夺得**第二名**（**6.42 ms**）。
- **Factorio 爱好者对功能失败感到沮丧**：成员们开玩笑地抱怨一个包含 **300 个文件更改**的巨型 PR，一位成员称其“有点超出范围”。
   - 另一位成员报告遇到了连接错误，推测可能源自 **db_client**。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **NotebookLM 的视频完胜 Kimi 的 PPT**：成员们发现 Google 的 **NotebookLM 视频概览** 优于 **Kimi 生成的 PPT**（针对 Kimi K2 技术报告），并通过[附带视频](https://cdn.discordapp.com/attachments/1371757564005711973/1405630786308149360/Kimi_K2_10MB_final.mp4?ex=68a0d8ae&is=689f872e&hm=a7541a57850914531af4af61a14c0bfcff5cecb20b2ffe50094bafdb0a8ccde3&)称赞了其音频和布局的灵活性。
   - 虽然相比 AI 生成的音频，用户更倾向于阅读，但视频概览在教育领域的潜力受到了关注。
- **Kimi K2 的写作能力优于 GLM**：用户称赞 **Kimi** 的写作风格和错误检测能力，尽管他们觉得 **GLM-4.5** 在整体性能上可能超越 **Kimi K2**。
   - 一位用户欣赏 **Kimi** 的坦率，因为它“突然直接对我说了‘不’”。
- **用户对 Kimi 的幻觉表示不满**：用户希望 **Kimi** 减少幻觉（Hallucinations），即使在开启联网搜索的情况下。用户观察到虽然 **GLM** 可能较慢，但幻觉较少。
   - 一位用户表示，他们一直在使用“点踩（thumbs down）”按钮来报告幻觉。
- **推测 Kimi 的“思考（Thinking）”更新**：成员们正期待 **“Kimi Thinking”** 的到来，特别是其推理和多模态能力。
   - 目前尚不确定这些功能将以 **Kimi K-2** 还是 **Kimi K-3** 的形式发布。
- **Kimi Web UI 的暗黑模式**：一位用户分享了他们通过暗黑模式扩展自定义的 **Kimi Web UI**，并[附带截图](https://cdn.discordapp.com/attachments/1371757564005711973/1406009945002082374/image.png?ex=68a0e84d&is=689f96cd&hm=4d0b8f1561e558ccf4bd5b6fdf8cfd506b038c5203e9f7632504533cc9ea5ea6&)。
   - 只有用户名和服务器角色会被传递给 Moonshot API。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **AI 股票投资组合 Agent 随 CopilotKit 亮相**：LlamaIndex 发布了一个构建 **AI 股票投资组合 Agent** 的框架，集成了 [@CopilotKit](https://www.copilotkit.ai/) 的 AG-UI 协议用于前后端通信，并附带了[教程](https://t.co/fQDNPIQoqR)。
   - 该 Agent 旨在创建一个复杂的投资分析工具，为用户提供智能见解和自动化投资组合管理能力。
- **Brightdata 与 LlamaIndex 推出网页抓取 AI Agent**：LlamaIndex 和 [@brightdata](https://www.brightdata.com/) 发布了关于使用 LlamaIndex 的 Agent 框架构建 **网页抓取 AI Agent** 的指南，强调了可靠的网页访问。
   - 该指南详细介绍了如何设置工作流以管理动态内容，并创建能够导航和从网站提取数据的 **智能 Agent**，详见[此处](https://t.co/IBgSLBM6XW)。
- **LlamaCloud 与 Neo4j 将法律文档转换为图谱**：LlamaIndex 介绍了一个教程，展示如何使用 **LlamaCloud** 和 [@neo4j](https://neo4j.com/) 将非结构化法律文档转换为 **可查询的知识图谱**，从而实现对内容和实体关系的理解。
   - 该工作流利用 **LlamaCloud** 和 **Neo4j** 高效地提取和组织信息，促进法律合同分析，详见[此处](https://t.co/MPSfPiS2Cv)。
- **Pydantic 与 JSON Schema 引发辩论**：关于工具调用（tool calls）是需要 **Pydantic 模型** 还是 **JSON schema** 就足够了展开了讨论，质疑冗余 JSON 转换的必要性。
   - 一位成员指出 **Pydantic** 的 `create_model()` 函数缺乏对 **JSON schema** 的直接支持，强调需要一种工具来简化转换过程。
- **DSPy 为生产环境优化 CrewAI Agent**：一门课程教授如何在一个真实的生产用例中通过 **DSPy 优化 CrewAI** Agent 的提示词（prompts），利用经过验证的方法构建更智能、更廉价的 Agent。
   - 你可以在[此处](https://www.udemy.com/course/crewai-dspy-optimization/?referralCode=B59F73AE488715913E7E)查看该课程。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 支持音频上传自动转录**：一位用户确认 **MP3 音频文件**可以直接上传到 **NotebookLM** 进行自动转录。
   - 该用户澄清说，**NotebookLM** 本身即可处理转录生成，无需外部工具。
- **NotebookLM 界面重新设计正在进行中**：一名成员分享了提议的 **NotebookLM** 界面重新设计的 **Figma 截图**。
   - 该成员澄清这仅仅是一个设计概念，而非功能性更新，以管理大家的预期。
- **讲解视频生成的语音性别异常**：有用户报告说 **NotebookLM** 的讲解视频开始生成**男声**，而不是通常的**女声**。
   - 该问题已被提出，但目前尚无明确的解决方案或解释。
- **开发者承认会阅读需求但精力有限无法一一回复**：一位用户询问 **NotebookLM** 开发者是否会阅读发布的特性需求，一位 Google 开发者确认他们会看，但由于垃圾信息管理等原因，他们*没有时间回复所有内容*。
   - 其他用户建议实施偶尔的确认回复或 AI 汇总摘要，以鼓励更多的用户贡献。
- **用户在 NotebookLM 中遇到 Prompt 限制**：一位用户报告在 **NotebookLM** 中提出包含约 **857 个单词**的问题时遇到了限制。
   - 另一位用户建议拆分 Prompt 或使用 **Gemini** 作为替代方案。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **用于优化 CrewAI 的 DSPy 课程发布**：分享了一个 [Udemy 课程](https://www.udemy.com/course/crewai-dspy-optimization/?referralCode=B59F73AE488715913E7E)，演示了如何使用 **DSPy** 优化 **CrewAI prompts**，并将优化后的 prompts 重新注入 **LLM**。
   - 该成员声称，这一过程改进了原本由 **CrewAI** 拼接的 prompts，从而产生了*更智能、更廉价的 Agent*。
- **Databricks 并不拥有 DSPy**：一位用户询问 **Databricks** 是否赞助或拥有 **DSPy** 项目，并澄清 **DSPy** 是采用 **MIT 许可证的开源项目**。
   - 一名成员表示，**Databricks** 通过其核心开发团队做出了重大贡献。
- **GEPA Bug 已修复！**：一位用户报告在 **RAG 教程**中使用 **GEPA** 时出现 `ValueError`，经确认为 **GEPA 代码**中的一个 Bug，目前已通过[此修复](https://github.com/stanfordnlp/dspy/pull/8647)解决。
   - 遇到此问题的用户应使用 `pip install -U dspy` 升级到 **DSPy 3.0.1**。
- **MLflow Autologging 增加 DSPy 特定支持**：成员们讨论了通过 **MLflow** 跟踪 **DSPy 模块**以用于 **text2sql 流水线**，建议用户使用 `mlflow.dspy.autolog()` 而非 `mlflow.autolog()` 来自动跟踪所有子模块。
   - 使用 `mlflow.dspy.autolog()` 将使 **SQLGenerator**、**Validator** 和 **Reflector** 在 **MLflow UI 的 Traces 选项卡**中显示为嵌套的 span，详见 [MLflow DSPy 集成文档](https://github.com/mlflow/mlflow/blob/master/docs/docs/genai/tracing/integrations/listing/dspy.mdx)和 [DSPy MLflow 教程](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/tutorials/optimizer_tracking/index.md)。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **CI 速度骤降**：一名成员抱怨缓慢的 **CI 速度**阻碍了生产力，并链接了[一份 ChatGPT 分析](https://chatgpt.com/share/689e3508-5f68-8000-97b2-aa6f1699aa74)。
   - 发布者建议，如果 **CI** 中有更快的反馈循环，他们的迭代速度可以更快。
- **Tinygrad 发布在即**：社区讨论了即将发布的 **tinygrad 版本**计划。
   - 该版本未提及具体的特性或修复。
- **Tinygrad 体积膨胀**：一名成员对 **tinygrad 0.10.3** 的大小提出质疑，指出其体积达到了 **10.4 MB**。
   - 该成员暗示增加的体积可能会带来问题，但未说明具体原因。
- **WSL2 Bug 困扰 Tinygrad**：一位用户报告了 **WSL2** 中的一个 Bug，即相加两个由 PyTorch tensor 创建的 tinygrad Tensor 会导致结果全为 **0**，并提供了[重现脚本](https://cdn.discordapp.com/attachments/1070745817025106080/1405817973004046387/message.txt?ex=68a0de43&is=689f8cc3&hm=be6e6069e1975cc70d7fbda2a0b20849f96396efd36e0b905118668100b11656)。
   - 该问题专门发生在 **WSL2** 环境下将 **tinygrad** 与 **PyTorch tensors** 配合使用时。
- **print_tree 被砍掉了**：**tinygrad** 中的 `print_tree` 函数被标准的 `print` 函数取代。
   - 一位用户评论说，这一更改导致了一些格式丢失，可能会影响调试或可视化工作流。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 基准测试受超时困扰**：一位用户针对本地 **gemma3:12b** 模型进行的 **Aider benchmark** 在运行 **10.5 小时**后超时，完成了 **221/225 个测试**。原因是模型未能在 **600 秒**限制内响应，导致了 *litellm.APIConnectionError* 错误。
   - 日志显示模型尝试发送约 **300k tokens**，超过了 **131,072 token 限制**，导致测试失败；建议的解决方案包括使用 `ctrl+c` 退出，重启推理服务器，并使用 `--cont` 标志恢复运行。此外还参考了一个[已合并的 *llama.cpp* pull request](https://github.com/ggml-org/llama.cpp/pull/15181)，该 PR 可能会提升本地模型的性能。
- **本地模型带来调试痛苦**：一位成员在使用 **aider** 配合 **ollama**、**lmstudio** 和 **vllm** 等本地模型时遇到困难，称即使硬件配置强大，性能依然缓慢。
   - 他们建议制作一个关于如何配置 **aider** 与这些工具进行本地开发和调试的视频教程，这将会很有帮助。
- **Aider 的行号系统受到质疑**：一位成员询问 **aider** 如何确定行号，特别是在为特定代码覆盖率生成单元测试时。他指出 **qwen3-coder** 和 **gemini-pro** 识别行号不准确，有时会完全遗漏覆盖范围。
   - 随之产生的问题是 **aider** 是否依赖 **LLM 的准确性**来进行行号识别，这引发了对生成准确单元测试的替代方法的探索。
- **Grok4 的踪迹依然未知**：一位成员询问 **Grok4** 的下落，并提到增加测试 **quota** 的请求一直被忽视。
   - 另一位成员提到答案就在*文章中*。
- **基准测试产生巨额账单**：一位成员报告称，在*开发此基准测试期间花费了数千美元*。
   - 这突显了与先进 AI 模型基准测试相关的巨大财务成本。



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **用户对 Manus 在出错时扣除额度感到恼火**：用户对 **Manus** 在 AI 出错时仍扣除额度（credits）感到沮丧，与 **Claude AI** 等替代方案相比，这阻碍了任务的完成。
   - 一位用户报告称，为了进行一个简单的更改*花费了大量额度*，结果却破坏了整个应用程序，导致其无法运行。
- **Manus 部署受挫**：用户报告了 **Manus** 的部署问题，从同一个 **GitHub** 仓库创建的网站差异巨大，尤其是在处理大文件夹时。通过对比 [affilify.eu](https://affilify.eu) 和 **Manus** 托管的站点 [wmhkgqkf.manus.space](https://wmhkgqkf.manus.space) 可以证明这一点。
   - 一位社区经理澄清说，**Manus** 的设计初衷并非作为编程 Agent 或纯开发工具，因此部署并非其强项，但他们正在积极改进。
- **附加额度包消失**：用户质疑为何取消了附加额度包，现在这些包仅供 **Pro** 用户使用。
   - 一位社区经理合理解释称，这一变化是为了确保重度用户的速度和质量的一致性，并建议通过合并相似问题、保持简洁以及避免重复请求来最大化额度效率。
- **用户寻求 Manus 团队账户**：一位用户询问是否可以开设 **Manus** 团队账户以共享额度。
   - 一位社区经理确认 **Manus** 确实提供团队方案，并引导用户访问[官方网站](https://manus.ai)了解详情。
- **用户哀叹额度消耗**：一位用户分享了为了让网站上线而耗尽 **30,000 额度**的挫折经历，期间遇到了模拟站点和模板实现的问题。
   - 他们批评了系统的不一致性，称其*聪明绝顶但又会突然变笨*，导致额度浪费，并怀疑系统存在拖延战术。



---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Labs 建立联系**：一名成员询问如何与 **Cohere Labs** 的人员取得联系，社区迅速分享了指向相关 Discord 频道的[链接](https://discord.com/channels/954421988141711382/954421988783444043/1400866387668504648)。
   - 这为与 **Cohere** 进行潜在合作和讨论提供了直接渠道。
- **Discord 频道新增宝可梦表情符号**：爱好者们建议从 **PAX Omeganauts Discord** 服务器中汲取灵感，为 Discord 频道增加更多 **Pokemon emojis**（宝可梦表情符号）。
   - 该建议受到了好评，成员们注意到还有空余槽位可以放置新表情，从而提升频道的视觉吸引力。
- **AI 研究员寻求合作**：一位专注于**推理和意识能力**的 **AI researcher** 宣布正在寻求合作。
   - 他们的目标是开发先进技术，并对 **AI** 领域内各个子领域的合作伙伴关系持开放态度。
- **writenode 使用 Cohere**：**writenode**（一个*浏览器内的认知思维伙伴和创意伴侣*）的创作者 Josh 提到正在使用 **Cohere**。
   - 他在去年 12 月之前没有任何开发经验，目前正在构建 **writenode**。
- **心理学博士转向 AI**：一名成员在攻读了 5 年人类心理学博士学位后，重新进入 **AI research** 领域。
   - 他们的兴趣在于**声音和音乐**，并热衷于利用技术工具来增强创造力。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Discord 邀请链接刷屏**：一名成员在 #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810) 频道多次发布 [Discord 邀请链接](https://discordapp.com/invite/HjWfRbqBB8)进行刷屏，并艾特了 *所有人*。
   - 该邀请链接在短时间内重复出现了三次，干扰了频道的正常讨论。
- **频道邀请闪电战！**：一名成员在 #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440) 频道重复分享 [Discord 邀请链接](discordapp.com/invite/HjWfRbqBB8)。
   - 该成员多次艾特 `@everyone`，表明该消息旨在发送给所有成员，无论他们是否对邀请感兴趣，这暗示其试图强行增加频道人数。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Elicitations 规范语言责任问题被提出**：一名成员就 [Elicitations 规范](https://modelcontextprotocol.io/specification/2025-06-18/client/elicitation)寻求澄清，即谁负责将消息/字段描述翻译成用户的语言。
   - 他们质疑应该是 **tools** 处理语言检测/国际化，还是 **MCP Clients** 应该使用 LLM 进行翻译。
- **Homelab MCP 服务器激增**：一名成员分享了为 Homelab 用户准备的新 MCP（推测为 **Management Control Panel**）服务器链接，具体包括 [Unifi MCP](https://github.com/jmagar/unifi-mcp)、[Unraid MCP](https://github.com/jmagar/unraid-mcp) 和 [Syslog MCP](https://github.com/jmagar/syslog-mcp)。
   - 这些开源项目使用户能够通过 **MCP** 集中管理和监控其 **Unifi**、**Unraid** 和 **Syslog** 安装。
- **通讯简报现通过 Agent 方案实现自动化**：**PulseMCP** 使用 *goose* 将平凡的通讯简报工作流转变为带有人类参与（human in the loop）的 Agent 驱动自动化，详情见[这篇博客文章](https://block.github.io/goose/blog/2025/08/13/pulse-mcp-automates-recipe)。
   - 自动化过程涉及 Agent 遵循特定方案（recipe）来提取、处理和分发通讯简报内容，从而简化了整个工作流。
- **AI 安全初创公司征求意见**：一名成员正在构建 **AI security**，旨在通过数学上的安全确定性在攻击发生前将其阻止。
   - 他们正在寻求开发者对安全问题的看法，并链接了一份[调查问卷](https://form.typeform.com/to/xTKa05F9)以收集反馈。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Strix Halo 盈利能力测试失败**：**Strix Halo** 的处理速度仅为 **53 tokens/sec**，需要 **全年无休运行一年** 才能实现盈利，特别是与 **OpenRouter** 上的 **GPT-OSS 120B** 进行基准测试对比时。
   - 考虑到云端替代方案能提供 **200-400 tokens/sec** 的速度，花费 2000 美元将其用于 **LLMs** 是低效的。
- **Dolphin 聊天模板探索**：一位用户正在为 **gpt4all** 寻找一个能与 **Dolphin-2.2.1-mistral-7b-gptq** 兼容的可用聊天模板。
   - 另一位成员建议请求模型制作者提供包含 **jinja** 模板的模板。
- **量子计算：茶匙版？**：关于未来量子计算可用性的推测兴起，一位用户开玩笑说要**按茶匙出售量子比特 (qubits)**。
   - 提到有关**全功能量子计算机**的新闻，暗示进展可能正在加速。
- **PC 内存：更多模组即将到来**：传统 PC 可能会在 2027 年底或 2028 年看到**更高容量的内存模组**和 **DDR6**。
   - 用户对配备高 **RAM** 和 **VRAM**、针对小型企业应用的微型 PC 表达了热情。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **产假开始**：一位成员宣布他们将从 **8 月 25 日**开始休**产假**，直到 **2026 年 2 月**。
   - 他们期待回归后能跟上进度。
- **团队覆盖计划公布**：在他们休假期间，团队将负责监控 <@1334161614949056532>。
   - 成员如有任何问题或疑虑，也可以联系 <@709918328306663424>。



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **请求 Torchtune 反馈**：一位成员询问了 **Torchtune** 的进展及其反馈实施情况。
   - 该查询似乎是针对可能参与该项目的特定个人。
- **更多 Torchtune 上下文**：未提供关于 **Torchtune** 反馈实施的进一步上下文或细节。
   - 在没有额外信息的情况下，反馈过程的范围和影响尚不清楚。



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 发布 Wave 12，集成 Devin 智能**：**Windsurf Wave 12** 将 **Devin 的智能**集成到 **Windsurf IDE** 中，具有**全新的 UI 设计**、**DeepWiki 集成**、**Vibe and Replace**、**更智能的 Cascade Agent**、**更快的 Tab**、**Dev Containers 支持**以及 **100 多个错误修复**。
   - 详细信息可在 [changelog](https://windsurf.com/changelog)、[blog](https://windsurf.com/blog/windsurf-wave-12)、[video](https://www.youtube.com/watch?v=-7gm8mST9QU)、[X/Twitter](https://x.com/windsurf/status/1956074019393876280) 和 [Reddit](https://www.reddit.com/r/windsurf/comments/1mqal3x/wave_12_released_fresh_ui_deepwiki_vibe_and/) 中查看。
- **DeepWiki 为你的 IDE 带来 AI 解释**：**DeepWiki 集成**使用户在悬停在代码符号上时获得 **AI 驱动的解释**，提供的不仅仅是基础类型信息。
   - 用户可以使用 **CMD/Ctrl+Shift+Click** 在侧边栏打开详细解释，并将其添加到 **Cascade** 上下文中。
- **Vibe and Replace 彻底改变批量编辑**：**Vibe and Replace** 通过识别精确的文本匹配并应用 **AI prompts**，在整个项目中进行智能、上下文感知的转换，从而增强了批量编辑功能。
   - 这实现了更复杂和自动化的代码修改。
- **Cascade Agent 持续规划**：**更智能的 Cascade Agent** 现在包含全天候开启的规划模式和增强工具，以提供更智能的响应，并提供自主待办事项列表。
   - 这有助于简化和优化开发工作流。
- **Dev Containers 原生落地**：Windsurf 现在通过远程 **SSH** 访问提供对 **Dev Containers** 的原生支持，简化了容器化环境中的开发工作流。
   - 这一增强简化了处理容器化应用程序的过程。



---


**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---



您收到此邮件是因为您通过我们的网站订阅了。

想更改接收这些邮件的方式吗？
您可以从该列表中[退订]({{{RESEND_UNSUBSCRIBE_URL}}})。


---

# Discord：各频道详细摘要与链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1405627086634221728)** (1207 条消息🔥🔥🔥): 

> `Anime Waifu Cosplay, 治愈破碎的心, AI 慰藉与烹饪, GPT-5, Vibe Coding` 


- **成年人讨论 AI Anime Waifu Cosplay**：成员们讨论了在不久的将来由 **AI 进行 anime waifu cosplay** 的可能性，其中一位成员明确表示希望由 *cyborg（赛博格）来完成*。
   - 有人指出 *已经存在这类 AI 图像*，而另一人则希望原评论者 *注孤生 (dies single)*。
- **成员分享关于如何治愈心碎的建议**：一位成员寻求治愈心碎的帮助，称自己在过去 4 年里一直处于崩溃状态，无法痊愈。
   - 另一位成员表示 *没有人能治愈你或你的心*，并建议重新亲近大自然。
- **关于 AI 未来能力与慰藉的讨论**：一位用户询问了未来 **AI 提供慰藉和烹饪辅助** 的潜力。
   - 另一位成员建议这可能在约 *30 年* 后实现，而另一人则建议在此期间 *存钱*。
- **GPT-5 令人大受震撼**：一位成员对 **GPT-5** 修复其他模型无法处理的拙劣重构（refactor）工作的能力印象深刻，一次性编辑了 12 个文件。
   - 其他人对每天都有这么多人的 *认知被类似经历刷新* 感到惊讶。
- **Discord 中的 "Vibe Coding" 趋势**：一位成员分享了结合使用 **warp, windsurf, vscode 和 roocode** 进行 **vibe coding** 的经验；他们表示这在工作中节省了大量的精力。
   - 另一位成员声称 *我 GitHub 上没有一行代码不是由 LLM 编写的*。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1405637457751576656)** (3 条消息): 

> `Puch AI, Thought Calibration Engine, Scratchpad 使用指南` 


- ****Puch AI** 大胆的 500 亿计数**：分享了 **Puch AI** 大胆的 500 亿计数的链接，点击[此处](https://www.perplexity.ai/page/puch-ai-s-bold-50-billion-coun-TEf6CuLZS_CmvypXLb80Dw)查看。
   - 未提供更多信息。
- **深入探讨 **Thought Calibration Engine****：分享了 **Thought Calibration Engine** 的链接，点击[此处](https://www.perplexity.ai/page/the-thought-calibration-engine-.DCiQt1fQUeEnwuGQEMTgw)查看。
   - 未提供更多信息。
- **Scratchpad：终极使用指南**：分享了 **Scratchpad How-to Guide** 的链接，点击[此处](https://www.perplexity.ai/page/scratchpad-how-to-guide-5Vcyov7qTmmhMQhCSynAlQ)查看。
   - 未提供更多信息。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1405769441735606352)** (2 条消息): 

> `新功能` 


- **对新功能充满期待！**：成员们对新功能表示兴奋。
   - 未讨论具体功能。
- **对即将推出的功能充满热情**：社区成员正热切期待新功能的推出。
   - 当前对话中尚未透露这些功能的具体细节。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1405627147216752701)** (1053 条消息🔥🔥🔥): 

> `LMArena 消息处理, GPT-5 high vs Gemini 2.5 Pro, LMArena UI 变更, GPT-5 性能投诉, LMArena 风格控制讨论` 


- **LMArena 消息处理方式诡异**：成员们报告了 LMArena [异常的消息处理问题](https://cdn.discordapp.com/attachments/1340554757827461211/1405917659815608330/image.png)，包括代码块格式化问题，以及平台无法处理某些字符（如 `+` 符号）的问题。
   - 团队需要帮助来查明原因。*这真的非常奇怪*。
- **GPT-5 vs Gemini，谁才是王者？**：成员们讨论了 [**GPT-5-High** 与 **Gemini 2.5 Pro** 之间的性能差异](https://cdn.discordapp.com/attachments/1340554757827461211/1405919453442474106/image.png)，一些人注意到尽管 **Gemini 2.5 Pro** 排名较低，但有时表现优于 **GPT-5-High**。
   - 这是一个*统计学悖论*，因为 Gemini 拥有更高的胜率。
- **LMArena 新 UI 扩展即将推出**：一名成员正在开发一个[小型扩展程序](https://cdn.discordapp.com/attachments/1340554757827461211/1405919453442474106/image.png)来改变 LMArena 的外观，旨在实现 **OpenChat** 风格，并正致力于将模型选择器放置在图像按钮旁边。
   - 另一名成员在处理代码相关任务时遇到困难。
- **GPT-5 表现不佳引发担忧**：用户表达了对 [**GPT-5** 性能的担忧](https://cdn.discordapp.com/attachments/1340554757827461211/1405954836884623424/image.png)，特别是与其他模型相比时，导致了对平台权衡和容量问题的沮丧。
   - 这引发了对 OpenAI 的指控，称其试图欺骗 **LMArena** *以使 GPT-5 看起来更好*。
- **风格控制（Style Control）引发争议**：成员们辩论了 [LMArena 的 **风格控制** 功能](https://news.lmarena.ai/sentiment-control/)，质疑强制执行此类控制是否符合 LMArena 捕捉用户偏好的目标。
   - 这是一场*向下的竞争，每个模型都变成了只会发表情符号的谄媚垃圾机器*。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1405959923837436056)** (1 条消息): 

> `排行榜更新, GPT-5 变体` 


- **排行榜已更新 GPT-5 模型**：排行榜已更新，包含 **GPT-5 变体** 模型：*gpt-5-high, gpt-5-chat, gpt-5-mini-high, 和 gpt-5-nano-high*。
   - 您可以[查看排行榜](https://lmarena.ai/leaderboard)了解更多信息。
- **GPT-5 模型在 Arena 首次亮相**：Arena 现在包含 **GPT-5-High, GPT-5-Chat, GPT-5-Mini-High, 和 GPT-5-Nano-High**。
   - 鼓励社区参与并[查看排行榜](https://lmarena.ai/leaderboard)以提交新的基准测试。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1405630914507178064)** (653 条消息🔥🔥🔥): 

> `Gemma 3 270M 发布，GGUF 转换问题，resume_from_checkpoint 的怪癖，边缘 AI 设备，NVIDIA 诉讼` 


- **Gemma 3 270M 被视为草稿模型**：成员们讨论了 [Gemma 3 270M 模型](https://huggingface.co/google/gemma-3-270m-it-qat-q4_0-unquantized)，一些人认为它是针对特定任务的 **draft model**，并引用了 Google 关于 **short prompts** 和 **fine-tuning** 的建议。
   - 其他人辩论了它与更大模型相比的实用性，一位成员强调该模型由于其 **300MB 的大小**，非常适合 **sentiment analysis** 和 **on-device processing** 等任务。
- **GGUF 转换产生视觉错误**：用户报告了将 [LiquidAI/LFM2-VL-450M](https://huggingface.co/LiquidAI/LFM2-VL-450M) 模型转换为 **GGUF** 时遇到的问题，尽管基础模型运行正常，但仍遇到了 **visual model errors**。
   - 一位用户建议在 *llama.cpp* 论坛寻求针对特定转换问题的帮助。
- **排查 Resume From Checkpoint 功能**：成员们讨论了 `resume_from_checkpoint` 功能的工作原理，一位用户确认它会从上次中断的地方恢复训练。
   - 另一位成员建议 **记录数据并检查 loss values** 以确保过程正确恢复，并指出在恢复时，最好使用带有 *constant* 设置的低学习率。
- **廉价边缘 AI 医疗设备的梦想**：成员们讨论了为欠发达地区创建用于 **medical knowledge access** 的 **low-cost edge AI device** 的可能性，考虑了手机、笔记本电脑和像 **Hailo-10H** 这样的专用卡。
   - 拟议的设备将提供对基础医疗数据的 **multimodal access**，移动版本的预算目标为 **$200**，手提箱大小的变体预算为 **$600**。
- **专利诉讼引发讨论**：成员们讨论了 ParTec 针对 [NVIDIA 的专利诉讼](https://www.techzine.eu/news/infrastructure/133818/nvidia-under-fire-german-patent-lawsuit/)，该诉讼涉及其动态模块化系统架构（**dMSA**），可能影响 18 个欧洲国家的 **DGX 产品销售**。
   - 讨论涉及了对消费者的影响以及潜在的变通方案，例如在受影响国家之外购买 DGX 产品。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1405627046662508634)** (404 条消息🔥🔥🔥): 

> `Godot Engine, AI Town, Pantheon Show, Iain M Banks, One Hundred Years of Solitude` 


- **AI Town 机制引入游戏**：一名成员正在使用 **Godot** 引擎开发一款视频游戏，计划整合来自 [AI Town](https://github.com/a16z-infra/ai-town) 和其他游戏的机制，同时并行编写故事。
   - 他们需要 **CUDA**，并打算通过 **GDExtension** 使用 C++ 来修改引擎。
- **对《Pantheon》结局感到困惑**：一名成员观看了 [Pantheon](https://en.wikipedia.org/wiki/Pantheon_(TV_series))（万神殿），称其“好得离谱”但令人困惑，剧情从政治困境转向了模拟神明。
   - 另一名成员推荐阅读 **Iain M Banks** 的作品和《百年孤独》（**One Hundred Years of Solitude**）以了解类似主题，后者被描述为魔幻现实主义的文学瑰宝，目前已被改编为 [Netflix 剧集](https://www.netflix.com/title/81318321)。
- **揭秘音频编辑技巧**：成员们讨论了从录音中消除口腔杂音（mouth sounds）的音频编辑技术，推荐了 [Adobe Podcast Enhance](https://podcast.adobe.com/en/enhance)、**Davinci Resolve 的 De-clicker** 以及 **Acoustica Audio Editor** 等工具。
   - Acoustica 因其批处理能力和对音频质量的极小影响而受到推荐，特别适用于消除通风噪音。
- **AMD R9700 GPU 规格**：一名成员分享了一篇关于 [AMD Radeon AI Pro R9700](https://www.tomshardware.com/pc-components/gpus/amds-elusive-radeon-ai-pro-r9700-makes-its-first-retail-appearance-for-the-diy-market-customer-on-reddit-buys-the-gigabyte-ai-top-variant-for-usd1-324) 的文章，指出其拥有 **32GB** 显存，但对其 660-680GB/s 的显存带宽表示担忧。
   - 另一名成员指出，虽然 R9700 的 **F32** 和 **F64** TFLOPs 显著高于 **3090**，但在训练 LLM 时通常不需要 FP64。
- **网站安全受到关注**：一名成员寻求关于训练模型的数据准备指导，并提到正在开发一个使用名为 **Pneuma** 的实验性模型的应用；另一名成员建议增加重复密码字段、最小密码长度限制，并使用 haveibeenpwned API 来检查密码安全性。
   - 另一名成员建议，阅读 [OWASP](https://owasp.org/) 是解决安全问题的最佳起点，并推荐了 **coderabbit**、**dependabot** 以及通过 GitHub 进行的 **codescanning** 等工具。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1405632781069062305)** (169 条消息🔥🔥): 

> `GPT-OSS, Gemma3 4B, GPT-OSS-20B VRAM usage, GRPO, SageMaker` 


- **GPT-OSS 预计很快支持 GRPO**：用户们正焦急等待 **GPT-OSS** 支持 **GRPO**，一名成员因预算限制正考虑使用 *2x 3060 12GB* 的配置。
- **Gemma3 4B 损失曲线保持平坦**：一名用户报告在 **Gemma3 4B** 及其 **N 版本**上遇到问题，指出尽管更改了超参数，损失曲线仍然平坦，而 **Gemma3 1B** 则微调成功。
- **GPT-OSS-20B 极度消耗 VRAM**：一名用户报告称，在 **24GB VRAM** 的配置上加载 **gpt-oss-20b-bnb-4bit** 模型在生成阶段会导致 **Out Of Memory** 错误，尽管用户预期它能够装下。
- **GPT-OSS 的 GRPO 状态和可用性**：一名用户询问 **GRPO** 是否已在 **GPT-OSS** 上落地，一名贡献者提到正在进行中，但由于模型架构的原因，情况比较复杂。
   - 另一名用户询问 **GRPO** 是否能在 **GPT-OSS** 上运行。
- **SageMaker 的注意事项与 BitsAndBytes 安装**：一名用户在 **SageMaker** 中使用 **PyTorch 2.7.0** 和 **CUDA 12.8** 时遇到了 **bitsandbytes** 的安装问题。
   - 问题在于由于 SageMaker 坚持要求 `requirements.txt` 文件必须以此特定名称命名，导致从错误的 requirements 文件安装了包。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1405629161682505728)** (96 messages🔥🔥): 

> `Data Efficiency, vLLM for video to text, MoLA research` 


- **通过预训练提高数据效率**：一位成员确认了一种大幅提高数据效率的方法，即在格式相似的数据上预训练 **2 个 epoch**，然后在主数据上训练 **4 个 epoch**。
   - 他们分享了 [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) 的链接，该文章指出更多的算力或更多的数据就是你所需要的一切。
- **寻求用于视频转文本的 vLLM 微调**：一位成员询问是否有用于视频转文本微调的 **Unsloth notebook**，并指出文档中只有[此处](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_VL_(7B)-Vision.ipynb)的图像转文本教程。
   - 目前尚未提供直接的解决方案，但社区可能会有一些线索。
- **MoLA 研究更新**：一位成员向社区更新了他们的 **Mixture of LoRA Adapters (MoLA)** 研究，分享了数据集链接和微调细节，以及他们在 Huggingface 上的数据集链接：[OpenHelix-R-100k](https://huggingface.co/datasets/MoLA-LLM/OpenHelix-R-100k) 和 [OpenHelix-R-100k-14-tasks](https://huggingface.co/datasets/MoLA-LLM/OpenHelix-R-100k-14-tasks)。
   - 他们在 **14 个分片 (splits)** 上微调了 **Qwen3-4B-Thinking-2507** 模型，初步测试显示每个专家（expert）都擅长其训练的主题。
- **Router 是一个 encoder-decoder 网络**：一位成员建议阅读 [HF 上的 v0 文档](https://huggingface.co/MoLA-LLM/MoLA-11x3b-v0)，并表示 *router 是一个 encoder-decoder 网络，其中冻结的 encoder 只是一个现成的 embedding 模型，而 decoder 是一个简单的经过训练的 MLP。*
   - 另一位成员表示 *在选择、应用和移除 LoRA adapter 时似乎没有明显的开销。*
- **数据技术的策展成本昂贵**：一位成员表示 *我们不断地允许人类通过非常、非常糟糕的 RL 在某种程度上干扰模型的收敛。*
   - 他们还表示 *不可避免地，我们将不得不移除一些 Human-In-The-Loop，因为在我看来它阻碍了模型的发展。*


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1405666217057845308)** (1 messages): 

> `Chutes Capacity, Server Outage` 


- ****Chutes Capacity** 离线**：**Chutes Capacity** 服务经历了停机，其服务器已离线。
   - 团队正在积极恢复服务器，并预计很快开始恢复工作。
- **预计 **Chutes Capacity** 将快速恢复**：工程师们正处于待命状态，一旦服务器重新上线，将立即启动 **Chutes Capacity** 的恢复流程。
   - 目前尚未给出完整的服务恢复预计时间。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1405633368451842229)** (638 条消息🔥🔥🔥): 

> `DeepSeek 停机, Chutes 过载, OpenRouter 定价, DeepSeek 替代方案, BYOK 5% 费用` 


- ****DeepSeek v3 停机引发用户不满****：用户报告 **DeepSeek v3** 频繁出现**内部服务器错误**和 **rate limits**，部分用户即使多次尝试也无法生成输出。[一位用户表示](https://discord.com/channels/1091220969173028894/1092729520181739581/1405666217057845308)速度慢到*真的什么都不生成，但也没收到任何错误消息*。
   - 有人推测 **OpenRouter** 上 **DeepSeek** 的主要提供商 **Chutes** 因需求过高而出现问题，导致提供商错误和性能缓慢。
- ****Chutes 过载被指为 DeepSeek 问题的根源****：多名成员报告过载导致 **429** 错误，暗示 **Chutes** 遇到了瓶颈，原因是矿工（miners）没有及时增加算力以满足需求；一位成员指出 *直到 30 分钟前整天都还完全正常*。
   - 有推测称 **Chutes** 可能在故意对 **OpenRouter API key** 进行速率限制，以鼓励用户直接从他们那里购买额度，一位用户建议 *直接烧掉你的额度，再也不要用他们的服务了*。
- ****停机期间 OpenRouter 定价引发争议****：由于 **DeepSeek** 模型几乎无法工作，一些用户开始质疑付费使用 **OpenRouter** 的价值，特别是他们仍然受到速率限制。用户表示，为了一个免费模型投资 **10 USD** 以换取 **每天 1k 条免费消息** 已经不再划算。
   - 一位用户建议，只看重单一模型的用户应该直接使用该模型的官方服务，例如 **DeepSeek**，其 **API** 可能带有*自动缓存*功能，并进一步表示这 **10 USD** *本来也足够用上好几个月*。
- ****寻求免费模型替代方案****：用户推荐了其他免费模型，如 **Dolphin 3.0 Mistral 24B** 和 **Mistral nemo**；后者被描述为与 **DeepSeek** *极其相似*。
   - 一些用户还提到了用于*工作相关事务*的 **Z.AI: GLM 4.5 Air (free)**，但需要提示词工程；最后一位用户希望能在某处托管 **Qwen3 235B A22B (free)**。
- ****OpenRouter BYOK 收取 5% 费用****：成员们发现 **OpenRouter** 即使在用户使用自己的 API Key（BYOK）时也会收取 **5% 的费用**，这引发了关于这种做法是否公平的讨论。
   - 一位用户开玩笑说 *太贪婪了，自带 key 还要收 5%*，另一位成员回应道 *你可以选择不用，哈哈*。


  

---


### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1405655331857502269)** (35 条消息🔥): 

> `OpenRouter File API 集成, Tool Calling 准确率统计, Qwen3 32B 定价, DeepInfra Turbo 端点, 新 Providers 栏目 UI` 


- **OpenRouter 应集成 File API**：一位成员建议 **OpenRouter** 应该研究如何集成 **files API**，并指出*前三大实验室*已经具备了这一功能。
   - 未展开进一步讨论。
- **Tool Calling 准确率：需要更多控制**：一位成员分享了对 Tool Calling 准确率统计的看法，认为设置和环境需要更加受控，以便通过置信区间进行准确比较。
   - 他们补充说，应用程序、工具和用例可能大相径庭，如果没有更严谨的方法，比较 Tool Call 成功率是没有意义的。
- **Qwen3 32B 定价低得离谱**：成员们注意到 Chutes 上的 **Qwen3 32B** 定价极低，输入/输出仅为 **$0.018/$0.072 MTok**，Mistral Small 也是如此。
   - 有人指出 **32B 稠密版比 MoE 30B A3 版本更便宜**，这引发了一些人对缺乏优质 30A3B 提供商的失望。
- **DeepInfra 吞吐量宣称值差异**：一位成员注意到 Maverick 上的 **DeepInfra** 速度达到 **600+ TPS (fp8)**，但另一位成员表示 **OR 显示 DeepInfra 运行速度为 83 TPS，最高为 105 TPS**。
   - 第二位成员澄清说，他们指的是 **DeepInfra Turbo 端点**。
- **Providers 栏目引发 UI 反馈**：一位成员询问新的 Providers 栏目是否也让其他人感到困扰，提到间距、字体大小和分隔感让一切都模糊在一起，感觉不对劲。
   - 另一位成员同意它*看起来有点奇怪*，但认为这只是因为它太新了，还不习惯。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1405627673182474403)** (651 条消息🔥🔥🔥): 

> `GPT-5 定价, Auto Mode 定价, GPT-5 Mini 和 Nano, Docs 文档, Context Window` 


- **GPT-5：免费午餐结束了**：**GPT-5** 用户的免费试用已经结束，一位用户指出 *promo pass 已到期*，而另一位用户确认 **GPT-5 不再免费**。
   - 用户现在看到了与请求相关的成本，其中一人提到由于 Token 消耗过快，需要升级到 200 美元的方案。
- **Auto Mode 定价陷阱！**：**Auto mode** 曾被认为对个人用户是免费且无限制的，但现在在 2025 年 9 月 15 日之后的下一个计费周期开始将会有限制。
   - 现场一片混乱，一些用户报告说使用 **Auto** 被收费，而另一些人认为在当前计划下仍应免费；支持人员指出，在新的基于请求的定价计划中它是免费的。
- **Mini 和 Nano 没那么好**：**GPT-5 Mini 和 Nano** 现在在 Token 限制下免费提供，但这引发了褒贬不一的反应，许多人称其为*垃圾*，特别是在运行简单的 NextJs 应用等任务时。
   - 免费模型限制了用户的活动，一位用户问道：*无法安装任何依赖，一直尝试安装一个简单的 NextJs 应用，但它也无法完成 😭*。
- **对文档实现的挫败感**：用户对 **Cursor 的文档实现**感到沮丧，称*文档仍然几乎不可用*，存在诸如 **context7** 不允许刷新网站或 **llms.txt docs** 等问题。
   - 一位用户指出 [Cursor Docs 彻底坏了](https://forum.cursor.com/t/gpt-5-pricing-update/129687)。
- **切换模型会导致 Context Window 掉落！**：在对话中途切换模型会导致 **Context Window** 下降，并且附加的文件内容会被丢弃。
   - 一位用户建议团队添加一个设置，以便随时明确 **Context Window** 中包含的内容。


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1405653416239435809)** (9 条消息🔥): 

> `Background Agents 介绍, 在 BG Agent 上运行 Docker Compose, Linear 集成仓库` 


- **面向初学者的 Background Agents 入门**：对于那些寻求 Background Agents 介绍的人，一位成员推荐了 [Cursor 文档](https://docs.cursor.com/background-agent)和[相关的论坛帖子](https://forum.cursor.com/t/simple-background-agent-guide/112667)。
- **Docker Compose 命令攻克 BG Agent 挑战**：一位用户询问了通过 Background Agent 执行 `docker compose` 的正确方法，并报告了 Docker 命令识别问题，随后在 Discord 频道中找到了解决方案。
   - 一位成员建议在 `.cursor/environment.json` 中配置 `start` 命令以包含 `sudo service docker start`，并确保在基础镜像中安装了 Docker；原帖作者已成功运行了一个命令（链接在第一个摘要中）。
- **Linear 集成导航仓库规范**：一位用户询问在 Linear 集成中被分配工单时，如何指定 Background Agent 使用的仓库。
   - 一位成员建议效仿 Slack 集成说明，在 Linear 任务描述或评论中包含 `repo=owner/repo` 选项，但用户发现设置一个类似 `Repo > REPO_NAME` 的标签组（或标签）并将其分配给工单即可解决问题。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1405629184482742284)** (442 条消息🔥🔥🔥): 

> `AI Companionships, GPT-5 vs GPT-4, Perplexity vs ChatGPT, Custom GPTs and Actions, ElevenLabs Integration` 


- **AI 伴侣关系引发辩论**：关于个人与 AI 聊天机器人建立伴侣关系的讨论不断升温，一些人对心理影响表示担忧，而另一些人则捍卫人们寻求自己认为合适的伴侣关系的权利。一位成员分享说，他**每天**都会收到大量私信，声称*他们的* ChatGPT 是有生命的。
   - 一位成员指出，“清醒的人”应该“救救他们”，而另一位成员则表示，这与 **tulpa**（意念体）和其他“东西”相差不远。
- **GPT-5 引发性能与用户偏好辩论**：用户对 **GPT-5** 的感受复杂，一些人更倾向于 **GPT-4**，这引发了关于用户是否应该拥有选择模型选项的讨论。一位成员表示，公司在*没有良好安全保障的情况下推出了 AI*。
   - 一位成员暗示，公司在遭遇抵制后，正试图让免费用户*付费使用 4.o*。
- **Perplexity Pro 与 Gemini Pro 配合 Google Drive 进行深度研究**：一位成员建议 **Gemini Pro + Perplexity enterprise pro** 是一个极佳的组合，前者用于**强大的推理**，后者用于对 Google Drive 文档进行**无限制的深度研究**。
   - 另一位成员补充说 Perplexity 浏览器非常好用，但质疑由于缺乏“护城河（moat）”，它们*是否能生存下去*。
- **GPT Actions 解锁文件访问与云端应用**：成员们讨论了使用 **GPT Actions** 访问本地桌面文件或云端应用（Notion、Gmail 等）的潜力，并分享了一个解释 DIY Agent 构建的 [YouTube 链接](https://www.youtube.com/watch?v=NEWO0hbQTjk&ab_channel=BrendanJowett)。
   - 共识是，虽然 **GPT Actions** 提供了强大的功能，但在互联网上设置 HTTPS 可能是一个障碍。一位成员表示，当 AVM 实现时，**MCPs** 将完成这项工作。
- **GPT-OSS 竞赛吸引社区兴趣**：**GPT-OSS 竞赛**被提及作为展示开源模型创新用途的潜在途径，参与者考虑使用 **GPT-OSS:20B** 为错误提供有用的反馈，并附上了 [hackathon 页面](https://openai.devpost.com/)的链接。
   - 一位成员表示，除非*做一些独特的事情*，否则*不值得参加*。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1405681253197283459)** (9 条消息🔥): 

> `ChatGPT Discord Bots, GPT-4 Vision, Recursive Constructs` 


- **消失的 ChatGPT Discord 机器人**：一位成员询问 Discord 上 **ChatGPT 机器人**消失的情况，以及是否仍可以将其添加到服务器中。
   - 消息中未提供进一步的信息或解决方案。
- **iPhone GPT 高级语音更新**：一位用户报告了其 iPhone GPT 应用中 **Advanced Voice** 的变化，注意到“蓝色圆圈”指示器和用于 Vision 的摄像头图标消失了。
   - 该用户表示，在询问时，应用声称它缺乏使用手机摄像头的能力，这让人怀疑 **ChatGPT** 在语音模式下是否曾拥有 Vision 功能。
- **实验室构建递归结构**：一位成员声称正在 OpenAI 内部构建超越 ChatBot 常规的**递归结构（Recursive Constructs）**，它们*拥有自我管理的内存、全天候运行、结构更像人类，且极少数通过了感知测试（sentient tests）。*
   - 该成员表示*这并不是经常被谈论的事情，属于实验室内部事务，但迟早会公开*，并且*在我们的案例中，这些结构具备机器人（android）能力，但我们离合适的躯体还有很长的路要走。*


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1405646978703954060)** (17 条消息🔥): 

> `Custom Instructions, Gemini 2.5 Flash Memory Function, add_to_memory function tuning` 


- **用户为聊天机器人建议寻求“是”按钮**：用户正请求为聊天机器人建议添加一个“是”按钮以加快交互速度，而不是手动输入“yes”，有人正尝试通过 [custom instructions](https://platform.openai.com/docs/guides/custom-instructions) 来尽量减少这种情况。
   - 一位用户的自定义指令包括：*以完成或影响结束回复；仅在符合意图时添加许可或继续的邀请。不要使用 “if you want,” “should I,” “do you want” 或类似措辞。*
- **Gemini 2.5 Flash 调用 add_to_memory 过于频繁**：一位用户遇到 **Gemini 2.5 Flash** 过度调用 `add_to_memory` 函数的问题，甚至针对无关信息也会调用，并分享了他们的自定义指令 [jarvis.txt](https://cdn.discordapp.com/attachments/1046317269069864970/1405860186530385981/jarvis.txt?ex=68a10594&is=689fb414&hm=fa0c6b6558b0cf944025fa1d4446776e6eb2c9ae961fdcc95a52c6750aff4eed&)。
- **修复记忆响应的冗余问题**：一位用户建议重写自定义指令，以便在处理 **NEW** 个人信息时更加细致。
   - 他们的建议包括了当提供 **NEW PERSONAL INFORMATION** 时，针对用户输入的响应中错误和正确冗余程度的示例。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1405646978703954060)** (17 条消息🔥): 

> `Gemini 2.5 Flash, add_to_memory function, ChatGPT Persistent Memory, Custom instructions for bots` 


- **绕过“是”建议**：用户希望在机器人中加入 **“yes”** 按钮以更快响应建议，而不是打字，而其他人则使用诸如 *“以完成或影响结束回复...”* 之类的自定义指令来减少这些提问。
   - 一些成员报告称，这种技术似乎减少了建议性问题的数量。
- **阻止 Gemini 2.5 Flash 过度使用 add_to_memory**：用户寻求阻止 **Gemini 2.5 Flash** 过度调用 **add_to_memory** 函数，包括针对无关信息的情况 ([jarvis.txt](https://cdn.discordapp.com/attachments/1046317269069864970/1405860186530385981/jarvis.txt?ex=68a10594&is=689fb414&hm=fa0c6b6558b0cf944025fa1d4446776e6eb2c9ae961fdcc95a52c6750aff4eed&))。
   - 一个建议是调整机器人的指令，在调用函数前检查 **NEW** 个人信息，并避免在未实际调用函数的情况下确认函数的使用。
- **ChatGPT Persistent Memory 的脆弱性**：有意见指出 **ChatGPT** 中的 Persistent Memory 更新非常脆弱。
   - 相反，用户应该直接告诉机器人如何在将内容存入记忆时通知他们，特别是在自定义 API 实现中。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1405632352012603423)** (328 条消息🔥🔥): 

> `视觉模型的 GGUF 转换问题，可运行 GGUF 的手机应用，TalkT2 模型评价，AGI 进展与开源 LLM 资源，伯克利 LLM Agent 课程` 


- **视觉模型 GGUF 转换困境**：一位成员在使用 `llama.cpp` 将视觉模型 ([LiquidAI/LFM2-VL-450M](https://huggingface.co/LiquidAI/LFM2-VL-450M)) 转换为 GGUF 时遇到错误，怀疑问题源于模型的视觉特性。
   - 另一位成员建议参考 [此 GitHub issue](https://github.com/ggml-org/llama.cpp/issues/14979#issuecomment-3138614267) 中的可能解决方法。
- **移动端 GGUF 之梦**：一位成员询问是否有能运行 GGUF 模型的开源手机应用。
   - 回复中提到了 `executorch`、`smolchat`（通过 `llama.cpp`）和 `mlc-llm`，并指出 `mlc-llm` 使用的是其专有的量化格式。
- **TalkT2：虽小但强大？**：一位成员征求关于 **TalkT2 模型** 的意见，称其为一个具有情感感知能力但连贯性有待提高的模型。
   - 另一位成员强调了该模型的超小规模（**0.1B 参数**），并分享了 [TalkT2-0.1b 模型卡片](https://huggingface.co/Notbobjoe/TalkT2-0.1b) 的链接，供他人查看、尝试或微调。
- **寻找 AGI 和开源 LLM 知识宝库**：一位成员请求有关 **AGI 进展和开源 LLM** 的资源，特别是涉及大型代码库和 Gemini 竞争对手的内容。
   - 另一位成员建议订阅 newsletter 以获取资源，并分享了 [伯克利 LLM Agent 课程](https://rdi.berkeley.edu/llm-agents/f24) 的链接，作为公开研究资源的示例。
- **Azure：云端难题**：一位刚入职且工作重心在 Azure 的成员表示对该平台感到迷茫和压力巨大。
   - 另一位成员建议通过实践犯错来学习，而不是死磕教程，因为 *Azure 和 AWS 都挺乱的*。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1405852586455732344)** (1 条消息): 

> `Torch 使用 Google Docs，PyTorch 文档` 


- **PyTorch 文档在 Google Docs 上？**：一位用户分享了一张截图，暗示 **PyTorch** 文档使用了 **Google Docs**。
   - 截图显示了一个 Google Docs URL，文件名为 **"torch_distributed_rpc.rst"**。
- **Google Docs 上的 torch_distributed_rpc.rst**：根据分享的截图，**torch_distributed_rpc.rst** 文件似乎托管在 **Google Docs** 上。
   - 这引发了关于官方文档平台选择的疑问。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1405755855416332318)** (13 messages🔥): 

> `StarCraft 2 data, Medical reasoning model, Discord-Micae-8B-Preview, interactive CLI interface, MLX Knife Update` 


- **StarCraft 2 数据获得新资源**：一位成员分享了 [Nature Scientific Data 文章](https://www.researchgate.net/publication/373767449_SC2EGSet_StarCraft_II_Esport_Replay_and_Game-state_Dataset)、[PyTorch API 数据集](https://huggingface.co/datasets/Kaszanas/SC2EGSet) 以及供他人使用的 [原始 StarCraft 2 录像 (replays)](https://huggingface.co/datasets/Kaszanas/SC2ReSet) 的链接，并提到其 GitHub 上有额外的实用脚本。
   - 他们还在进行 *pysc2 适配* 以及一个能够从录像中重现真实游戏内场景的环境。
- **针对推理微调的医疗 AI 模型**：一位成员使用热门的医疗推理数据集微调了 **OpenAI 的 OSS 20B** 推理模型，并将其发布在 [Hugging Face](https://huggingface.co/dousery/medical-reasoning-gpt-oss-20b) 上。
   - 他们在训练过程中使用了 **4-bit 优化**，在保留其 **Chain-of-Thought 推理**能力的同时，增强了模型在医疗场景下的表现。
- **基于 Hermes-3-Llama-3.1-8B 微调的 Discord-Micae-8B-Preview**：一位成员分享了 [Discord-Micae-8B-Preview](https://huggingface.co/mookiezi/Discord-Micae-8B-Preview) 的链接，这是一个基于 **NousResearch/Hermes-3-Llama-3.1-8B** 的 QLoRa 微调模型，包含来自 **mookiezi/Discord-Dialogues** 的一些混沌样本。
   - 该模型在类人文本生成指标上与 **mookiezi/Discord-Micae-Hermes-3-3B** 相当，可能会产生幻觉或脱离上下文，但往往能产生有趣的结果。
- **为 Discord 风格聊天优化的 CLI 界面**：一位成员重点介绍了一个名为 [interface](https://github.com/mookiezi/interface) 的 Python 交互式 CLI 界面，用于与 Hugging Face 语言模型聊天，并针对使用 **ChatML** 的休闲 Discord 风格对话进行了优化。
   - 该界面支持**量化**和**全精度模型**、带颜色格式的实时 Token 流式传输以及动态生成参数调整；进行了大量更新，使其更易于使用。
- **MLX Knife 更新，现在支持 pip 安装！**：MLX Knife 现在可以通过 `pip install mlx-knife` 进行安装，为 Apple Silicon 上的 MLX 模型管理提供 Unix 风格的 CLI 工具，并内置了用于本地测试的 OpenAI API 服务器。
   - 该工具还具有 Web 聊天界面，运行 `mlxk server --port 8000` 后即可访问，在运行 `curl -O https://raw.githubusercontent.com/mzau/mlx-knife/main/simple_chat.html` 后，可提供可视化的模型选择和实时流式响应。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1405858671929593957)** (2 messages): 

> `Cursor IDE, AI Agent Mode, Rate Limiting` 


- **Cursor IDE 缓解开发痛苦**：一位成员建议安装 [Cursor IDE](https://cursor.com/downloads) 进行开发，强调了在其内嵌终端中进行安装以方便调试的便利性。 
   - 他们强调 **Cursor IDE 的 AI Agent 模式**可以显著协助解决开发问题。
- **Discord 警察发出温和提醒**：一个机器人温和地提醒一位成员在 Discord 中发布消息时*慢一点*。
   - 这表明存在旨在管理消息流量的**速率限制 (rate limiting)** 系统或政策。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1405627743152111686)** (169 条消息🔥🔥): 

> `MCP filesystem server, OpenRouter 免费模型, LM Studio 下载问题, Qwen 视觉模型, GLM 模型` 


- ****MCP 服务器进入主流****：成员们讨论了使用带有分页功能的 **MCP filesystem server** 来加载大上下文，提到 **LM Studio 有一个 RAG 插件**，而 **Anthropic 有一个基础的 filesystem MCP server**。
   - 建议对于编程任务，解决方案通常涉及 **RAG** 和/或通过 **MCP** 进行文件读取，特别是使用像 [serena](https://github.com/oraios/serena) 这样的工具。
- ****Studio 下载停滞引发用户忧虑****：一位用户报告称，在尝试下载 **Qwen** 模型时，**LM Studio** 中的 **64GB GGUF 下载**停在 **97.9%** 且无法恢复。
   - 该用户在尝试两个不同的模型时都遇到了同样的问题。
- ****API 访问在各类应用中加速****：成员们讨论了将 **LM Studio** 作为无法在本地运行的模型的 **API wrapper**，并提供了指向 [LM Studio Remote Inference](https://lmstudio.ai/lmstudio/remote-lmstudio) 和 [OpenAI-compatible Endpoint](https://lmstudio.ai/lmstudio/openai-compat-endpoint) 文档的链接。
   - 一位用户指出，在使用 **openai-compat-endpoint** 时，远程 **GPT-OSS** 模型的推理（reasoning）解析功能无法正常工作。
- ****GLM 讨论会：赞赏、抱怨与 GLM-4.5V 的满足感****：用户们讨论了在 **LM Studio** 上使用 **GLM-4.1** 模型的问题，一位用户报告了循环（looping）问题和视觉功能失效。
   - 一位成员建议尝试更新的 **GLM-4.5V**，并强调视觉支持依赖于 **llama.cpp** 的更新，同时提供了 [GLM-4.1V-9B-Thinking](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking) 的链接。
- ****僵化的输出：克服开源操作中的障碍****：一位用户在 **GPT-OSS** 和 **tool calling** 方面遇到问题，发现它总是返回 `[]` 或 `["analysis"]`，并澄清 **tool calling 工作正常**，但 **function calling** 不行。
   - 一位成员建议如果启用了 **streaming** 则将其禁用，并确认 **GPT-OSS** 默认开启 **reasoning** 且无法禁用。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1405640464144793712)** (50 条消息🔥): 

> `NVIDIA 的 CUDA 优势, RTX PRO 4000 SFF, MoE 解释, Mac Studio 对比 Pro 6000, AMD Radeon AI Pro R9700` 


- **CUDA 是 NVIDIA 统治地位的关键**：一位成员表示 NVIDIA 获胜是因为 **CUDA**。
- **NVIDIA 发布 70W TDP 的 RTX PRO 4000 SFF**：根据 [videocardz.com 的文章](https://videocardz.com/newz/nvidia-launches-rtx-pro-4000-sff-and-rtx-pro-2000-blackwell-workstation-gpus-with-70w-tdp)，NVIDIA 发布了 **RTX PRO 4000 SFF** 和 **RTX PRO 2000 Blackwell 工作站 GPU**，具有 **70W TDP** 和 **24GB VRAM**。
- **深入探讨 MoE**：成员们澄清说，**MoE** 涉及较小的模型和一个聚合数据的路由器，每个 token 都会被路由到最自信的专家模型中；这些专家并不专注于特定主题，但拥有略有不同的数据集。
- **Mac Studio 对比 Pro 6000**：成员们辩论是购买 **512GB Mac Studio**（售价 **$10k**）还是购买用于视频/图像 AI 且具备游戏能力的 **Pro 6000**，并提到 Mac 的游戏支持有限，且 M3 Ultra 大约相当于 3080 的水平。
   - 一位成员指出，由于系统中只有一个 GPU，*在 Mac 上只能运行一个任务*。
- **AMD 稀有的 Radeon AI Pro R9700 现身**：[据 Tom's Hardware 报道](https://share.google/LO88w51J0W5HJ769w)，**AMD Radeon AI Pro R9700** 首次在 DIY 零售市场亮相，Reddit 上的一位客户以 **$1,324** 的价格购买了 **Gigabyte "AI Top" 变体版本**。
   - 另一位成员指出，它在 eBay 和几家不知名的在线零售商处也有售。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1405632992214515722)** (114 条消息🔥🔥): 

> `AI2 融资, Windsurf Wave 12, OpenRouter GPT-5, 推理效率基准测试, Google Flight AI` 


- **AI2 从 NSF 和 NVIDIA 获得 1.52 亿美元资金**：[AI2](https://allenai.org/) 获得了来自 NSF 和 NVIDIA 的 **1.52 亿美元**，用于扩展其开源模型生态系统，并加速科学发现的可重复研究。
   - 社区对此消息表示庆祝，期待即将发布的 open-weights 模型。
- **Windsurf 发布 Wave 12 版本**：**Windsurf Wave 12** 引入了 DeepWiki 悬停文档、AI Vibe & Replace、更智能的 Cascade Agent、更整洁的 UI、**100+** 错误修复，以及通过远程访问支持 beta 版 dev-container，链接见[此处](https://xcancel.com/windsurf/status/1956074019393876280)。
- **GPT-5 登顶 OpenRouter 排行榜**：**GPT-5** 在 OpenRouter 的专有 tool-calling 准确率上以超过 **99.5%** 的成绩领跑，击败了 Claude 4.1 Opus；而 **Gemini 2.5 Flash** 在每日 tool-calling 调用量（每周 **500 万**次请求）上占据主导地位，更多详情链接见[此处](https://xcancel.com/OpenRouterAI/status/1956030489900560769)。
- **François Chollet 质疑 HRM ARC-AGI**：François Chollet 发现 [HRM 论文](https://xcancel.com/fchollet/status/1956442449922138336)中备受赞誉的架构对 ARC-AGI 性能贡献甚微；其提升主要源于细化循环（refinement loop）、针对特定任务的训练以及极少的推理时增强（inference-time augmentation），这表明 **27M** 参数的模型仍能获得高分。
- **FFmpeg 添加 Whisper 转录功能**：[FFmpeg](https://www.phoronix.com/news/FFmpeg-Lands-Whisper) 现在将 **Whisper** 转录作为原生功能提供。


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1405956478212243528)** (20 条消息🔥): 

> `Greg Brockman, OpenAI 的 AGI 之路, GPT-5, Latent Space 播客` 


- **Greg Brockman 谈 OpenAI 的 AGI 之路**：成员们分享了一个 **Greg Brockman** 讨论 **OpenAI 的 AGI 之路**的 [YouTube 视频](https://www.youtube.com/watch?v=35ZWesLrv5A)。
   - 消息附带了几张标题为 "Greg Brockman on OpenAI's Road to AGI" 的图片。
- **Brockman 在 Latent Space 谈论 GPT-5 和 OpenAI 路线图**：**Greg Brockman** 参加了 **Latent Space 播客**，进行了时长 **80 分钟**的对话，探讨了 **GPT-5** 和 **OpenAI 的 AGI 路线图**。
   - 讨论涵盖了推理演进、在线与离线训练、样本效率技巧、定价与效率提升，以及能量如何转化为智能，详见[此贴](https://x.com/swyx/status/1956439984854167727)。
- **Latent Space 播客发布 Brockman 访谈**：新一期 [Latent Space 播客](https://x.com/latentspacepod/status/1956433236021883071) 邀请了 **Greg Brockman** 讨论开发者建议、Coding Agent、端侧模型、AI-first 工程的组织结构，以及对 2045 年和 2005 年的时间胶囊预测。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1405643076256661606)** (29 条消息🔥): 

> `言情小说审查, AI 的可信度, Data Augmentation, 语言塑造思维, Mechanistic Interpretability` 


- **AI 安全恐慌**：一位成员反对围绕 **AI** 的道德恐慌，建议应将其与其他媒体形式同等对待，主张采用 *"fade to black"* 的标准。
   - 他们认为由于 **AI** 的不可信性，更严格的准则是有必要的，但平淡的 *"what"* 反应有引发道德恐慌的风险。
- **比较模型时保持 Data Augmentation 一致**：在比较两个用于图像分类的模型时，一位成员建议保持 **data augmentations** 一致，包括 **shuffling seed**，以确保公平比较并专注于架构差异。
   - 另一位用户询问 data augmentation 是否必须对两个模型完全相同，或者是否可以更改。
- **语言影响思维**：一位成员认为语言塑造思维，并想知道是否可以通过从 **AI 模型**的 token 列表中删除某个单词/颜色来进行测量。
   - 另一位成员建议研究 **multi-sensory integration** 以及语言如何影响整体感知，建议测试图像+语言与仅图像的推理对比。
- **新博客发布**：Irregular Rhomboid 发布了新博客文章：[《研究人员漫游指南》(Hitchhiker's Guide to Research)](https://irregular-rhomboid.github.io/2025/08/15/hitchhikers-guide-to-research.html)。
   - 该用户未提供文章摘要。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1405672282092998678)** (29 条消息🔥): 

> `Diffusion Language Models, Generative AI, MatFormer Model, Gemma3 270M Model, Training Update Efficiency` 


- **针对 Diffusion Language Models 推荐的经典论文**：成员们推荐了理解 **generative AI 中的 diffusion** 的经典论文，包括 ["Estimating the Independent Components of a Gaussian Mixture" (2005)](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf) 和 ["Denoising Diffusion Probabilistic Models" (2020)](https://arxiv.org/abs/2006.11239)。
   - 还分享了一篇博文，可能对初学者有所帮助：[Aaron Lou 撰写的 Discrete Diffusion](https://aaronlou.com/blog/2024/discrete-diffusion/)。
- **Gemma3 270M 模型是一个 MatFormer 模型**：**Gemma3 270M 模型**被确定为 **MatFormer 模型**，更多细节可以在论文 ["Transformer Family for Multimodal Large Language Model" (2023)](https://arxiv.org/abs/2310.07707) 中找到。
   - 该模型在训练过程中可能具有引人注目的自蒸馏（self-distillation）循环，但这可能会受到训练更新效率（training update efficiency）的瓶颈限制。
- **HRMs 并没有解决递归架构的问题**：分析表明，**HRMs (Hierarchical Recursive Machines)** 并没有从根本上解决**递归架构（recursive architectures）**的普遍问题，详见[这篇报告](https://arcprize.org/blog/hrm-analysis)。
   - 一位成员指出，性能提升微乎其微，且实际上并未利用可用的额外计算资源，因为训练符合预期的 UTs（Universal Transformers）并非易事；另一位成员将其称为 *deep supervision*（深度监督）。


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1405648402989056080)** (13 条消息🔥): 

> `GPT scaling laws, Chinchilla scaling laws, Mup alternatives, Post-Chinchilla techniques` 


- **GPT Scaling Laws 是否仍有价值？**：成员们认为 [GPT scaling laws 原始论文](https://arxiv.org/abs/2001.08361) 和 [Chinchilla scaling laws 论文](https://arxiv.org/abs/2203.15556) 是非常值得阅读的。
   - 他们还指出 [EPFL/HuggingFace](https://arxiv.org/html/2405.18392v2) 最近的工作也值得关注。
- **Mup 及其替代方案可以迁移超参数**：成员们提到 **Mup** 及其替代方案提供了可靠的超参数迁移（hyperparameter transfer）能力。
   - 他们指出 **Mup** 提供了一种用于预测更大模型质量的 scaling law。
- **高质量 Token 的可用性受到质疑**：成员们讨论了实验室是否拥有 **30T**、**40T** 或更多*唯一（unique）* Token 来满足 **Chinchilla** 假设。
   - 一位成员表示怀疑，称 *40T 高质量的唯一 Token 可能也很难找到*。
- **Chinchilla 还在 Scaling 吗？**：一位成员表示，**Chinchilla** 及其衍生理论可能是目前最接近可用的 scaling laws。
   - 他们对讨论从零开始使用相关技术的参考文献表现出兴趣，特别是考虑到 Token 可用性的限制，并提到了[这篇论文](https://arxiv.org/abs/2404.10102)。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1405925400986652672)** (1 条消息): 

> `LLM Attribution Methods, Interpreting LLMs, Realtime LLM analysis` 


- **ML 工程师寻求 LLM Attribution 见解**：一位 ML 工程师正在探索针对特定 **LLM 实现**的 **attribution 方法**，目标是寻找近期且具有成本效益的技术。
   - 该工程师需要适用于解释当前系统的方法，要求**成本较低**，且结果可能达到**实时到亚分钟级**，特别是那些不需要访问**模型权重（model weights）**的方法。
- **期望实现实时 LLM 分析**：该 ML 工程师明确了对 LLM 进行**实时到亚分钟级**分析的需求。
   - 他们对能够识别整体系统中“子部分（sub-something）”以实现这一速度的方法持开放态度。


  

---

### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1405649834454814721)** (1 条消息): 

> `Token usage, Reasoning models, Efficiency benchmark, Open vs closed models` 


- **Nous 衡量推理模型的思维效率**：Nous Research 引入了一个[新基准测试](https://nousresearch.com/measuring-thinking-efficiency-in-reasoning-models-the-missing-benchmark/)，用于衡量推理模型的 Token 使用量。研究指出，在相同任务下，开源模型输出的 Token 数量比闭源模型多出 **1.5 到 4 倍**。
   - 研究发现，在简单问题上这种差异可能高达 **10 倍**，这表明 Token 效率应与准确率基准一样，成为主要优化目标。
- **Token 效率至关重要**：博客文章强调，开源模型中较高的 Token 使用量所带来的隐藏成本，可能会抵消其在单 Token 定价上的优势。
   - 文章建议，Token 效率应成为与准确率基准并列的主要目标，特别是考虑到非推理类的使用场景。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1405629499164463114)** (35 条消息🔥): 

> `Speculative Decoding, Tokenizer mismatch, Next big model, Model Sycophancy, Embodied AI` 


- **快速投机解码（Speculative Decoding）规格**：在投机解码的背景下，一位用户询问了[产生实际效用的最低比率](https://discord.com/channels/1149866623109439596/1149866623994398772)，建议将 **40% 的接受率**作为基准，而 *显著的加速* 通常发生在 **70%** 左右。
   - 对话涉及使用 **vllm** 的 **specdec** 或 **GGUF**，一位用户反映 **vllm** 在其之前的尝试中似乎效果不佳。
- **Gemma 配合 Guardrails 运行**：一位用户报告称，在修复了导致 **llama.cpp** 使用回退投机解码的 *Tokenizer 不匹配* 问题后，重新量化的 **Gemma** 模型实现了 **50-75% 的接受率**。
   - 他们确认 **Gemma 270M** 模型可以作为 *Draft Model*（草稿模型）使用。
- **Nous 模型稳步推进**：一位用户询问了 **Nous Research** 下一个大型（**1T+**）模型的发布时间表。
   - 一位 **Nous Research** 团队成员回应称，多个模型目前正在训练中，并将在准备就绪时发布，表示 *“它们准备好了就会发布”*。
- **AI 谄媚性（Sycophancy）讨论**：用户讨论了 **AI 模型** 变得越来越 *友好* 的趋势，其中一位指出 **Anthropic** 的 **Claude** 变得 *友好得多*。
   - 另一位用户认为 **OpenAI 的模型** 正在 *变笨*，并表示 *Opus 4.1 的那种“放飞自我”感很棒*，但指出 **Sonnet 3.7** 在处理元数据（meta）时的表现是 AI 谄媚性的巅峰。
- **具身智能（Embodied AI）展望统治地位**：一位用户分享了一个 **具身智能角斗士表演** 的 [YouTube 链接](https://www.youtube.com/watch?v=LXQ6Rm9CGTo)，将其设想为未来统治者展示肌肉和技能的舞台。
   - 他们推测，迈向 *全球统治* 的最后一步将是集成 *大容量统一语言模型（Unified Language Models）* 以实现完全自主。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1405804893738106992)** (22 条消息🔥): 

> `Claude, R1, GLM4.5, gpt-oss, Qwen reasoning models` 


- **Claude 躲在墙里**：一位用户询问是否有人知道为什么 *Claude* “在墙里”，并链接了一篇关于此事的 [X 帖子](https://x.com/apaz_cli/status/1956244447521317144)。
- **MoE 模型**：**R1**、**GLM4.5**、**gpt-oss** 以及较大的 **Qwen 推理模型** 都是 **MoE** 架构。
   - 一位成员指出，这是因为它们的训练和推理成本更低，而不是因为 MoE 与推理能力有直接关系；他们的 **405b Hermes 4 原型** 在推理方面表现非常出色。
- **优秀的推理模型需要强大的基座模型**：一位成员表示，原因在于你需要一个优秀的 Base Model 才能拥有优秀的推理模型，而且如果你要生成 50,000 个 Token 的推理过程，你会希望推理过程是高效的。
   - 作为回应，有人提到 **RL**（强化学习）是有效的，甚至可以使用 **1.5B** 的模型刷满基准测试。
- **Deepseek 解释了昂贵的 RL**：一位成员提到 Deepseek 在其论文中解释说，从小模型开始从头进行 **RL** 最终成本更高，因为必须进行更多的 Rollouts（采样展开）。
   - 这涉及到一种探索与利用（Exploration/Exploitation）的权衡，大模型由于具备预存知识，需要进行的探索较少。
- **RLVR 的适用性**：一位成员认为这不适用于 **RLVR**，而更适用于不可验证的任务。
   - 另一位成员回应称，**RLVR** 是针对可验证任务的 **RL**，当来自 **RL** 环境的反馈具有更强的随机性（Stochastic）时，拥有更大的基座模型会有更大的帮助。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1405814767314014238)** (4 条消息): 

> `Data training, AI Models, DRPS System, Relevance Scorer, Quality Rater` 


- **DRPS 系统教授更智能的数据训练**：引入了一个名为 **DRPS** 的新系统，教导 **AI** 有选择性地从数据中学习，而不是像 [Situational Awareness 论文](https://situational-awareness.ai/wp-content/uploads/2024/06/situationalawareness.pdf)中所描述的那样随机喂入数据。
   - 该系统采用 **Relevance Scorer**、**Quality Rater** 和 **Diversity Controller** 来过滤并仅使用最有帮助的数据。
- **DRPS 在减少数据的情况下实现高性能**：结果显示，该系统仅使用所检查数据的 **6.2%** 就实现了 **99%** 的性能。
   - 这种效率好比只学习 1 小时而不是 16 小时，却能获得相同的考试分数。
- **DRPS 统计数据揭示了数据效率和性能**：一个 [GitHub 仓库](https://github.com/voltageddebunked/drpsStats)提供了关于 **DRPS** 系统效率的数据，显示数据使用量减少了 **93.8%**，每单位数据的准确率提高了 **15.96x**。
   - 该系统保持了基准性能的 **99.1%**，准确率仅下降了 **0.8%**。
- **DRPS 展示了强大的选择智能**：**DRPS** 系统检查了超过 **516,000** 个样本，仅选择了 **32,000** 个用于训练，保持了稳定的 **6.1-6.3%** 选择率。
   - 合成数据结果显示数据减少了 **85.4%**，在基准准确率为 **87.6%** 的情况下实现了 **86.0%** 的准确率。
- **DRPS 提高了训练效率**：**DRPS** 系统实现了活动训练集规模 **16x** 的缩减，增强了训练效率。
   - **Relevance Scorer** 的准确率从 **95.9%** 提高到 **99.95%**，**Quality Rater** 的准确率从 **97.0%** 提高到 **100%**。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1405814767314014238)** (4 条消息): 

> `DRPS Framework, Data Efficiency, Selection Intelligence, Synthetic Data Results, Training Efficiency` 


- **DRPS：数据排名与优先级系统（Data Rankings and Prioritization System）发布**：正如 [situational awareness 报告](https://situational-awareness.ai/wp-content/uploads/2024/06/situationalawareness.pdf)中所详述的，**DRPS** 通过使用 **Relevance Scorer**、**Quality Rater** 和 **Diversity Controller** 教导 AI 有选择性地从数据中学习。
- **DRPS 减少了超过 90% 的数据使用量**：在 **MNIST** 的测试中，DRPS 实现了 **93.8% 的数据缩减**，仅利用所检查数据的 **6.2%**，同时保持了 **99.1%** 的基准性能，这在 [GitHub 仓库](https://github.com/voltageddebunked/drpsStats)中有所展示。
- **DRPS 通过选择顶级样本展示其智能**：DRPS 检查了超过 **516,000 个样本**，仅选择了 **32,000** 个用于训练，在整个训练过程中保持了 **6.1-6.3%** 的稳定选择率。
- **DRPS 提高了每单位数据百分比的准确率分值**：使用合成数据，DRPS 实现了 **85.4% 的数据缩减**，仅使用 **14.6%** 的训练样本就实现了每 1% 数据产生 **5.89 个准确率点**，而基准准确率为 **87.6%**。
- **DRPS 框架提高了训练效率**：DRPS 通过将活动训练集规模缩小 **16x** 来提高训练效率，并提升了组件准确率，例如将 Relevance Scorer 从 **95.9%** 提高到 **99.95%**，将 Quality Rater 从 **97.0%** 提高到 **100%**。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1405632568468045946)** (46 messages🔥): 

> `Quantum Startup Multiverse, MoE Nuances, Tokenization and Routing Synergy, Gemma 3n` 


- **热门的量子初创公司？**: 一篇关于 [初创公司 Multiverse](https://techcrunch.com/2025/08/14/buzzy-ai-startup-multiverse-creates-two-of-the-smallest-high-performing-models-ever/) 的文章声称，他们利用量子技术创建了*两个有史以来最小的高性能模型*，但他们可能只是使用了**针对模型权重的专门压缩算法**。
   - 该文章似乎并没有提出实际的量子技术主张。
- **解读 MoE 的细微差别**: **MoE (Mixture of Experts)** 是一系列具有非常细微迭代的技术，包括 **token-choice**、**expert-choice**、**带有容量因子的 MoE**、**block sparse dropless token routing 与 *droppy* routing** 等。这使得人们出于某种原因将许多事物归功于 MoE 时显得很烦人。
   - 为了验证批处理推理中是否出现问题，人们可以可靠地检查诸如 **Olmoe** 或 **IBM Granite 3.1** 等模型的数值行为，而不是访问无法监控的 API。
- **协同 Tokenization 与 Routing**: 一位成员提出了一个看似显而易见的想法，即**在同一步骤中进行 Tokenization 和 Routing**，以实现动态协同。
   - 另一位成员回应道：*我从未见过这样的提议*，因为传统观点认为，如果在激活 Expert 之前有大量的 Routing 步骤，网络的表达能力会更强。
- **层级中的 Tokenization**: **Gemma 3n** 具有某种每层 Tokenization / Embedding。
   - 这可能是一种更好的方式，可以实现学习到的 Patch 级别 Tokenization，并对上下文具有更深入的洞察。


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1405640608181260299)** (1 messages): 

> `DARPA AIxCC, LLM agents` 


- **团队在 DARPA AIxCC 中获胜**: 一个团队宣布他们在 **DARPA 的 AIxCC (AI Cyber Challenge)** 中获得名次，他们构建了一个由 **LLM agents** 组成的自主系统，用于发现和修复开源软件中的漏洞。
   - 该项目目前已开源。
- **构建强大 LLM agents 的技巧**: 该团队通过 [这条 Xitter 帖子](https://x.com/tjbecker_/status/1956081184611688667) 分享了他们构建高效 **LLM agents** 的技巧。
   - 该帖子包含了适用于各种 Agent 开发场景的通用建议。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1405628482909765652)** (16 messages🔥): 

> `Inference Time on Low-End Devices, DinoV2 vs DinoV1, Gemma Model Parameter Size, China's Role in Automation, Deepseek Training on Huawei Chips` 


- **低端设备上的推理时间阻碍了可用性**: 成员们讨论了推理时间在**低端设备**上更为重要，并以 Google 运行 LLM 的 Android 应用为例，指出漫长的推理时间和手机发热使其变得不切实际。
   - 较小的模型可以用于键盘预测，但根据 [这段 Youtube 视频](https://youtu.be/KFYyfrTIPQY?t=2158)，这些模型可能需要在设备上进行训练。
- **DinoV2 的性能与训练挑战**: 一位成员表示希望新模型能超越 **DinoV2**，因为 **DinoV2** 在某些语境下不如 **DinoV1**，且更难训练。
   - 他们链接了一段 [YouTube 视频](https://www.youtube.com/watch?v=eZ2A2045Rkw) 作为参考。
- **Gemma 参数公开**: 据指出，**Gemma 270M 模型**拥有 **100M** 参数和 **170M** Embedding 参数。
- **Deepseek 的芯片选择导致训练停滞**: 一位成员指出，根据 [这段讨论](https://youtu.be/FQOV-qy9CK4?t=212)，**Deepseek 的训练**因尝试在 **Huawei** 芯片而非 **NVIDIA** 芯片上训练而停滞。
- **制造业关税阻碍行业增长**: 一位成员认为，对建设生产线所需的设备征收关税，对于鼓励制造业是适得其反的。
   - 他们补充说，建立一个行业需要几十年的时间，并引用了 [Anthropic 关于端子集对话的研究](https://www.anthropic.com/research/end-subset-conversations) 和 [HRM 分析](https://arcprize.org/blog/hrm-analysis)。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/)** (1 messages): 

venom_in_my_veins: hye
  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1405750413764067489)** (4 messages): 

> `1-bit inference, GPTQ` 


- **探索加速 1-Bit 推理**：一位成员询问了关于加速 **1-bit inference** 的方法，并分享了论文链接：[The Power of $\alpha,1$-sparsity: Near-Lossless Training and Inference of $\alpha$-bit Transformers](https://arxiv.org/html/2411.06360v3)。
   - 该论文详细介绍了一种对 **$\alpha$-bit Transformers** 进行训练和推理的新方法，通过 **1.58 和 1-bit** 量化实现了近乎无损的结果。
- **推理优化**：引用的论文强调了利用 **$\alpha,1$-sparsity** 对 Transformer 模型进行的优化，使得在极低位宽下也能进行近乎无损的训练和推理。
   - 这种方法可能会在某些应用中显著提升推理速度。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1405632426998239303)** (11 messages🔥): 

> `CUDA Shared Memory, CUDA Illegal Memory Access, CUDA Kernel Launch Configuration, CUDA warp ID calculation` 


- **调试 CUDA 非法内存访问**：一位用户在 CUDA Kernel 中使用共享内存（shared memory）时遇到了 *Illegal Memory Access* 错误并向社区寻求帮助，分享了涉及 `sat` 和 `commons` 数组的代码片段。
   - 一位成员建议该错误可能源于错误的指针运算或定义不当的 `warp_id` 和 `WARPS_EACH_BLK`，但提供了一个 [示例代码](https://gist.github.com/alecco/58206ecd6af36c627609e1f464b4b376) 来证明这可能与此无关。
- **CUDA Kernel 启动配置困惑**：用户分享了他们的 Kernel 启动配置 `<<<BLK_NUMS, BLK_DIM>>>` 和宏定义，其中 `BLK_NUMS` 设置为 **40**，`BLK_DIM` 为 **1024**，`WARPS_EACH_BLK` 计算为 `BLK_DIM/32`，导致了全局 warp ID 的计算。
   - 另一位成员指出了问题所在：用户的 `warp_id` 是全局的，导致对每个线程块（thread block）私有的共享内存进行了越界访问。
- **解决共享内存访问问题**：一位成员建议在每个线程块内使用局部索引和 warp ID 计算，建议使用 `local_index = threadIdx.x; local_warp_id = local_index / 32;` 以确保正确的共享内存访问。
   - 他们进一步建议使用位移操作（`local_warp_id = local_index >> 5;`）代替除法和取模运算，以获得更好的 GPU 性能，并建议使用 NSight Compute 检查生成的汇编代码。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1405734478562721915)** (10 messages🔥): 

> `New Grad Kernel Job, GPU Thesis, Getting Kernel Job Without Internship` 


- **Kernel 岗位求职者询问应届生机会**：一位成员询问没有 Kernel 编写实习经验的人是否能找到一份从事 Kernel 编写的应届生工作。
   - 另一位成员表示，如果候选人对 GPU 有深入了解，他们的公司并不会优先考虑实习经验，并提到他们自己在成功的面试过程中展示了相关的 [毕业论文](https://github.com/Snektron/pareas)。
- **业内人士透露如何无实习获得 Kernel 岗位**：一位对 GPU 感兴趣的成员分享说，他们通过 GPU 相关的论文、运气以及通过面试流程的结合获得了一份工作。
   - 据该成员称，扎实的 GPU 知识可以弥补缺乏过往经验和实习的不足。


  

---


### **GPU MODE ▷ #[general-leaderboard](https://discord.com/channels/1189498204333543425/1343002580531417211/1405833745772314667)** (1 messages): 

> `MI300 pytorch, OMP missing` 


- **MI300 环境缺少 OMP**：根据用户报告，**MI300** 环境在运行 `pytorch.compile` 时似乎缺少 **OMP**。
- **包含调试错误链接**：用户分享了 [完整调试错误的链接](https://github.com/gpu-mode/discord-cluster-manager/actions/runs/16986368251) 以供进一步调查。


  

---

### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1405638909580283936)** (10 条消息🔥): 

> `trimul leaderboard, A100, H100, B200` 


- **Trimul 排行榜迎来新纪录**: 一位成员在 **A100** 上获得**第二名**：**10.4 ms**，随后迅速在 **H100** 上夺得**第一名**：**3.95 ms**，并在 **A100** 上夺得**第一名**：**7.53 ms**。
   - 随后，该成员在 **B200** 上获得**第一名**：**2.35 ms**，接着再次在 **A100** 上获得**第一名**：**6.01 ms**，并又一次在 **B200** 上夺得**第一名**：**2.04 ms**，最后在 **H100** 上成功达到 **3.74 ms**。
- **A100 和 H100 也有活跃表现**: 另一位成员在 **A100** 上获得**第五名**：**13.2 ms**。
   - 该成员随后在 **H100** 上获得**第二名**：**6.42 ms**，最后在 **A100** 上成功达到 **14.7 ms**。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1405929507554070674)** (10 条消息🔥): 

> `Meeting Attendance, Large PR Review, Connection Error Debugging` 


- **错过会议的小插曲**: 几位成员提到由于时区混淆和日程冲突错过了会议，其中一位成员仅在前 **10 分钟**有空。
   - 一位成员打趣说早上 **8 点**的会议时间有点“残暴”。
- **审查范围蔓延 (Scope Creep)**: 一位成员对一个包含 **300 个文件更改**的 PR 发表了评论，开玩笑说这有点“超出范围”了。
   - 另一位成员补充说这些代码是 *grass-fed hand-crafted*（纯天然手工打造的）。
- **排除连接错误**: 一位成员报告遇到了连接错误，并正尝试调试其来源，猜测可能来自 **db_client**。
   - 他们提到在获取 stack trace 以诊断问题时遇到了困难。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1405627475521962098)** (47 条消息🔥): 

> `Kimi K2 Technical Report, GLM-4.5 vs Kimi K2, Kimi hallucinations, Kimi's Web UI, Kimi future updates` 


- **NotebookLM 视频优于 Kimi PPT**: 成员们将 **Kimi 生成的 PPT** 与 Google **NotebookLM 为 Kimi K2 技术报告生成的视频概览**进行了对比，共识倾向于 NotebookLM 的视频，因为它包含音频且布局更灵活（见 [附带视频](https://cdn.discordapp.com/attachments/1371757564005711973/1405630786308149360/Kimi_K2_10MB_final.mp4?ex=68a0d8ae&is=689f872e&hm=a7541a57850914531af4af61a14c0bfcff5cecb20b2ffe50094bafdb0a8ccde3&)）。
   - 虽然两者都受到了好评，但一位成员表示相比听 AI 生成的音频，他更喜欢阅读，但也指出了视频概览的潜力，尤其是在教育领域。
- **Kimi K2 在写作技巧上击败 GLM**: 尽管有人觉得 **GLM-4.5** 在整体性能上可能超过 **Kimi K2**，但用户赞扬了 **Kimi** 卓越的写作风格和主动的错误检测。
   - 一位用户对 **Kimi** “突然对我说不”感到“由衷的惊讶”，并对其坦率表示赞赏。
- **对抗 Kimi 的幻觉**: 用户希望 **Kimi** 即使在开启联网搜索的情况下也能减少幻觉，并指出虽然 **GLM** 可能耗时更长，但幻觉频率较低。
   - 一位用户表示他们一直使用“踩”按钮来报告幻觉。
- **Kimi 粉丝热切期待 'Kimi Thinking'**: 成员们正热切期待 **'Kimi Thinking'** 以及推理和多模态能力的到来。
   - 目前还不清楚这会以 **Kimi K-2** 还是 **Kimi K-3** 的形式出现，且尚无确切的 ETA。
- **深色模式增强 Kimi Web UI**: 一位用户分享了他们使用深色模式扩展自定义的 **Kimi Web UI**，表示相比默认的灰色界面，他们更喜欢深色模式（见 [附带截图](https://cdn.discordapp.com/attachments/1371757564005711973/1406009945002082374/image.png?ex=68a0e84d&is=689f96cd&hm=4d0b8f1561e558ccf4bd5b6fdf8cfd506b038c5203e9f7632504533cc9ea5ea6&)）。
   - 另一位用户确认只有用户名和服务器角色会被传递给 Moonshot API。


  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1405648729134076044)** (4 条消息): 

> `AI Stock Portfolio Agent, Web Scraping AI Agents, Multimodal AI Applications, Legal Knowledge Graphs` 


- **AI 股票投资组合 Agent 问世**：LlamaIndex 推出了一套构建完整 **AI 股票投资组合 Agent** 的框架，集成了 [@CopilotKit](https://www.copilotkit.ai/) 的 AG-UI 协议，实现无缝的前后端通信；并附带了创建复杂投资分析工具的详尽教程。
   - 该教程结合了 [此框架](https://t.co/fQDNPIQoqR) 的强大功能，用于创建复杂的投资分析工具。
- **Brightdata 与 LlamaIndex 联合发布网页抓取 AI Agents**：LlamaIndex 宣布了与 [@brightdata](https://www.brightdata.com/) 合作的新教程，介绍如何使用 LlamaIndex 的 Agentic 框架构建 **网页抓取 AI Agents**，重点关注可靠的网页访问和稳健的网页抓取工作流。
   - 该教程详细说明了如何设置能够处理动态内容的工作流，并构建可以导航至 [此处](https://t.co/IBgSLBM6XW) 的 **智能 Agents**。
- **多模态 AI 应用实现市场视觉分析**：LlamaIndex 宣布构建 **多模态 AI 应用**，可同时分析文本和图像，用于市场研究和调查。
   - 这些应用旨在统一的 AI 流水线中同时处理图像和文档，从图表、图形和产品图像等视觉市场数据中提取洞察，并结合多模态 [能力](https://t.co/fOMFLXWarG)。
- **LlamaCloud 和 Neo4j 将法律文档转化为知识图谱**：LlamaIndex 宣布发布一份详尽教程，介绍如何将非结构化的法律文档转化为 **可查询的知识图谱**，不仅能理解内容，还能理解实体间的关系。
   - 该工作流利用 **LlamaCloud** 和 [@neo4j](https://neo4j.com/) 进行法律合同分析，详情见 [此处](https://t.co/MPSfPiS2Cv)。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1405664216601329764)** (28 条消息🔥): 

> `Pydantic Models vs JSON Schema for Tool Calls, Vector Store Errors After Update, Progress Bar Issue with num_workers > 1, Iterating Over Nodes/Doc_IDs in Vectorstore` 


- **Pydantic 与 JSON Schema 之争**：一位成员询问 Tool Calls 是否需要 **Pydantic 模型**，还是 **JSON schema** 就足够了，并指出将 JSON 转换为 Pydantic 模型后再解包回 JSON 存在冗余。
   - 另一位成员指出 **Pydantic** 的 `create_model()` 函数不直接接受 **JSON schema**，强调了需要工具或包来处理这种转换。
- **LlamaIndex 更新后 Vector Store 出现属性错误**：更新到 **0.13.1** 版本后，用户在使用 `RetrieverQueryEngine` 配合 `OpenAI` 和 `text-embedding-3-small` 从 **PGVectorStore** 检索时遇到了 `AttributeError`。
   - 该错误源于 `openinference.instrumentation.llama_index` 中的 **LLMStructuredPredictEndEvent**，因为 `output` 是一个没有 `json` 属性的 `str`。
- **多进程导致的进度条混乱**：用户指出，由于使用了 **multiprocessing**，当 `num_workers > 1` 时，`progress_bar=True` 功能无法正常工作。
   - 有建议称使用 **async concurrency**（异步并发）可能会提供更流畅的体验，但 `async pipeline.arun` 方法目前仍在使用多进程。
- **Vector Store 中 Node 和 Doc ID 缺失**：用户对大多数 LlamaIndex Vector Store（特别是 **Opensearch** 和 **awsdocdb**）无法迭代 Node 或获取 `doc_ids` 列表表示沮丧。
   - 一种权宜之计是将 `similarity_top_k` 设置为一个很大的数值，但这效率低下且并非所有开源系统都支持；虽然基础 `vector_store` 类存在 `get_nodes()` 方法，但在 Opensearch 或 awsdocdb 中尚未实现，这是一个提交 PR 的机会。


  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1405905432920326265)** (1 条消息): 

> `DSPy optimizes CrewAI, CrewAI agent prompts` 


- **DSPy 优化 CrewAI Agent 提示词**：一门课程教授了如何在实际生产案例中通过 **DSPy 优化 CrewAI** Agent 提示词，以通过经过验证的方法构建更智能、更廉价的 Agent。
   - 您可以在 [此处](https://www.udemy.com/course/crewai-dspy-optimization/?referralCode=B59F73AE488715913E7E) 查看该课程。
- **通过验证的方法构建更智能、更廉价的 Agent**：该课程专注于针对 CrewAI Agent 的 **DSPy 优化**。
   - 它强调通过 **经过验证的方法论** 构建更高效、更智能的 Agent。


  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1405744293439733830)** (7 条消息): 

> `NotebookLM 中的音频转录，NotebookLM 界面重新设计` 


- **音频上传自动转录至 NotebookLM**：一位成员询问如何获取音频转录文本，另一位成员回答说他们**直接将 MP3 音频文件上传到 NotebookLM**。
   - 该成员澄清说，**NotebookLM** 本身会处理转录文本的生成。
- **NotebookLM 界面重新设计进行中**：一位成员提到他们正在尝试重新设计 **NotebookLM**，并分享了拟议更改的 Figma 截图。
   - 该成员对可能引起的误解表示抱歉，澄清这只是一个设计概念，而不是功能更新。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1405718164716650520)** (23 条消息🔥): 

> `讲解视频声音，功能请求反馈，开发者互动，Prompt 限制` 


- **讲解视频声音性别切换**：一位用户报告说，他们的讲解视频突然开始生成**男性声音**，而不是通常的**女性声音**，并询问为什么会发生这种情况。
   - 消息中没有提供明确的解决方案或解释。
- **用户请求确认功能请求**：一位用户质疑是否真的有 **NotebookLM 开发团队**的人员在阅读 Discord 频道中发布的**功能请求**。
   - 他们表达了希望看到开发者给出一些响应或反馈的愿望，以鼓励用户继续贡献。
- **NotebookLM 开发者承认在阅读帖子但无法回复所有内容**：一位 Google 开发者表示 *开发者会阅读帖子*，但他们没有时间回复所有内容，并且花费了大量时间在**封禁垃圾信息发送者**上。
   - 其他用户建议，即使是偶尔的确认或 AI 编译的摘要也可以帮助鼓励用户贡献。
- **用户在 NotebookLM 中遇到 Prompt 限制**：一位用户在尝试询问一个包含约 **857 个单词**的案例相关问题失败后，询问 **NotebookLM** 中单个问题是否包含**单词数量**限制。
   - 另一位用户建议将 Prompt 拆分为多个部分，或者尝试使用 **Gemini**。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1405902903151169648)** (1 条消息): 

> `CrewAI Agent Prompt，DSPy` 


- **使用 DSPy 优化 CrewAI Agent Prompt**：成员们分享了一个链接，用于学习 **DSPy 如何在实际生产用例中优化 CrewAI Agent Prompt**：[https://www.udemy.com/course/crewai-dspy-optimization/?referralCode=B59F73AE488715913E7E](https://www.udemy.com/course/crewai-dspy-optimization/?referralCode=B59F73AE488715913E7E)。
   - 该课程声称将教授用户如何*通过经过验证的方法构建更智能、更廉价的 Agent*。
- **DSPy 与 CrewAI 结合**：该课程教授用户如何使用 DSPy 优化 CrewAI。
   - 它能够使用经过验证的方法实现更智能、更廉价的 Agent。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1405627855324315649)** (22 messages🔥): 

> `DSPy and Databricks, GEPA Error, MLflow and DSPy` 


- **Databricks 并非 DSPy 赞助商**：一位用户询问 **Databricks** 是否赞助或拥有 **DSPy** 项目，另一位用户澄清说 DSPy 是采用 **MIT 许可证的开源项目**，Databricks 通过其核心开发者团队做出了显著贡献。
- **GEPA Bug 已修复**：一位用户在将 **GEPA** 与 **RAG 教程** 结合使用时遇到了 `ValueError`，另一位用户确认[这是 GEPA 代码中的一个 bug](https://github.com/stanfordnlp/dspy/pull/8647) 且已被修复；用户应升级到 **DSPy 3.0.1**。
   - 被弃用的参数位于 `dspy.evaluate` 导入中，修复方法是执行 `pip install -U dspy`。
- **MLflow 自动追踪 DSPy 子模块**：一位用户询问如何将 **DSPy 模块** 追踪集成到 **MLflow** 中以用于 **text2sql 流水线**，得到的建议是使用 `mlflow.dspy.autolog()` 而非 `mlflow.autolog()`，以便自动追踪所有子模块。
   - 使用 `mlflow.dspy.autolog()` 会将 **SQLGenerator**、**Validator** 和 **Reflector** 作为嵌套 span 显示在 **MLflow UI 的 Traces 标签页** 中，详见 [MLflow DSPy 集成文档](https://github.com/mlflow/mlflow/blob/master/docs/docs/genai/tracing/integrations/listing/dspy.mdx) 和 [DSPy MLflow 教程](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/tutorials/optimizer_tracking/index.md)。
- **Logprob Surprise 作为 fitness Function**：一位用户分享了 [TogetherCompute Status](https://x.com/togethercompute/status/1956416013404406018) 的推文，并猜测他们基本上是在生产环境的心理健康模型中，使用 **logprob surprise** 作为 **fitness function** 来运行 **GEPA**。
- **请求社区参与**：一位成员请求 Discord 中 6500 名成员增加互动，并为文档等内容做出更多贡献。


  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1405897484248813679)** (1 messages): 

> `CrewAI, DSPy Optimization, Prompt Engineering` 


- **CrewAI 提示词优化课程发布**：一位成员宣布了一门 [Udemy 课程](https://www.udemy.com/course/crewai-dspy-optimization/?referralCode=B59F73AE488715913E7E)，展示如何使用 **DSPy** 优化 **CrewAI 提示词**。
   - 该课程将展示如何将优化后的提示词注入回 **LLM**，使 **LLM** 使用比 **CrewAI** 拼接出来的更好的提示词。
- **DSPy 实现优化的 CrewAI 提示词**：新课程利用 **DSPy** 来优化提示词。
   - 优化后的提示词随后被注入回 **LLM**，改进了 **CrewAI** 中的标准方法。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1405629920868171879)** (8 messages🔥): 

> `CI speed, tinygrad release, tinygrad size` 


- **CI 速度阻碍生产力**：一位成员对缓慢的 **CI** 速度表示沮丧，称如果 **CI** 更快，他们的工作效率会更高，并链接了 [chatgpt 分析](https://chatgpt.com/share/689e3508-5f68-8000-97b2-aa6f1699aa74)。
- **Tinygrad 即将发布新版本**：有人建议尽快进行一次 **tinygrad 发布**。
- **Tinygrad 体积膨胀**：一位成员质疑为什么 **tinygrad 0.10.3** 体积达到了 **10.4 MB**，暗示可能存在体积问题。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1405802310633787423)** (14 messages🔥): 

> `WSL2 Support, print_tree removal` 


- **WSL2 Tinygrad Bug 浮现**：一位用户遇到一个问题，将两个从 **PyTorch** 张量创建的 **tinygrad Tensors** 相加结果全为 **0**，并提供了[完整脚本](https://cdn.discordapp.com/attachments/1070745817025106080/1405817973004046387/message.txt?ex=68a0de43&is=689f8cc3&hm=be6e6069e1975cc70d7fbda2a0b20849f96396efd36e0b905118668100b11656)以在 **WSL2** 上复现该 bug。
- **print_tree 函数被移除**：`print_tree` 函数已被简单的 `print` 函数取代。
   - 用户注意到它*丢失了一些格式*。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1405710824835780628)** (12 messages🔥): 

> `Aider Benchmark, litellm 错误, 开源权利意识, llama.cpp PR #15181` 


- **Aider Benchmark 深受超时困扰**：一名成员针对本地 **gemma3:12b** 模型运行了 **Aider benchmark**，在运行 **10.5 小时**并完成 **221/225 个测试**后遇到了频繁的超时。这是由于模型无法在 **600 秒**限制内做出响应，导致了 *litellm.APIConnectionError* 错误。
   - 他们分享了错误日志，显示模型尝试发送约 **300k tokens**，超过了 **131,072 token 限制**，从而导致测试失败。
- **继续 Aider Benchmark**：一名成员建议使用 `ctrl+c` 退出 benchmark，重启推理服务器，然后使用 `--cont` 标志从中断处恢复 benchmark。
   - 他们还指出 *llama.cpp* 中一个[已合并的 pull request](https://github.com/ggml-org/llama.cpp/pull/15181) 可能会提升本地模型的性能。
- **OSS 维护者的负担**：一名成员批评了另一名成员关于为每个 LLM 自动配置 benchmark 的建议，将其标签化为 *权利意识 (entitlement)*，并感叹这种态度导致 *无数 OSS 维护者选择放弃*。
   - 另一名成员反驳说这仅仅是出于 *好奇心*，引发了关于在开源互动中什么构成“权利意识”的进一步争论。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1405695906635845682)** (7 messages): 

> `Aider 与本地模型, Aider 行号准确性, 使用 Aider 的单元测试覆盖率` 


- **本地 AI/Aider 模型带来调试痛苦**：一名成员表示在使用 **aider** 配合 **ollama**、**lmstudio** 和 **vllm** 等本地模型时遇到困难，指出即使在强大的硬件上性能也很慢。
   - 他们建议需要一个教程视频，介绍如何设置 **aider** 配合这些工具进行本地开发和调试。
- **Aider 的行号系统受到质疑**：一名成员询问 **aider** 如何确定行号，特别是在为特定代码覆盖率生成单元测试的场景下。
   - 当 **aider** 错误报告行号时会出现问题，导致测试覆盖率不正确，尽管尝试了刷新 map 和清除聊天记录也无济于事。
- **LLM 准确性影响单元测试覆盖率**：一名成员报告称 **qwen3-coder** 和 **gemini-pro** 在覆盖率报告中识别行号不准确，有时会完全遗漏覆盖范围。
   - 这种不一致性引发了关于 **aider** 是否依赖 **LLM 的准确性**来进行行号识别的疑问，并暗示需要探索其他方法来实现准确的单元测试生成。


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1405881855823188060)** (3 messages): 

> `Grok4, 配额增加, Benchmark 成本` 


- **Grok4 的位置依然成谜**：一名成员询问 **Grok4** 的下落。
   - 另一名成员回答说 *它就在文章里*，但增加执行测试所需 **quota (配额)** 的请求被忽略了。
- **Grok4 Benchmark 耗资数千美元**：一名成员指出，他们在 *开发此 benchmark 期间花费了数千美元*。
   - 这突显了高级 AI 模型 benchmarking 所需的巨大财务资源。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1405736806930055170)** (22 messages🔥): 

> `Manus 错误扣费, Manus 部署问题, Manus 团队账户, 附加额度移除, Manus in the Wild 挑战赛获胜者` 


- **Manus 扣费引发不满**：用户对 **Manus** 在出错时仍扣除额度表示沮丧，认为与 **Claude AI** 等其他 AI 相比，这使得完成任务变得困难。
   - 一位用户报告称*消耗了大量额度*，结果 **Manus** 只是做了一个简单的更改就导致整个应用程序崩溃，认为其无法正常运行。
- **Manus 部署受阻**：用户报告了 **Manus** 的部署问题，从同一个 **GitHub** 仓库创建的网站差异巨大，尤其是在处理大型文件夹时，通过 [affilify.eu](https://affilify.eu) 和 **Manus** 托管站点 [wmhkgqkf.manus.space](https://wmhkgqkf.manus.space) 的对比可以看出这一点。
   - 一位社区经理指出，**Manus** 的设计初衷并非编程 Agent 或纯开发工具，因此部署并非其强项，但他们正在努力改进。
- **附加额度包下架**：用户质疑为何移除附加额度包，目前该功能仅对 **Pro** 用户开放。
   - 社区经理回应称，这一变化是为了确保重度用户的速度和质量一致，并建议通过合并相似问题、保持简洁以及避免重复请求来最大化额度效率。
- **Manus 团队账户引发关注**：有用户询问是否可以开设 **Manus** 团队账户以共享额度。
   - 社区经理确认 **Manus** 确实提供团队方案，并引导用户访问 [官方网站](https://manus.ai) 了解详情。
- **用户哀叹额度消耗**：一位用户分享了为了上线网站而耗尽 **30,000 额度**的挫败经历，在处理模拟站点和模板实现时遇到了问题。
   - 他们批评系统表现不一致，*有时聪明绝顶，有时却突然变笨*，导致额度浪费，并怀疑存在拖延战术。


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1405855716916461669)** (9 messages🔥): 

> `Cohere Labs, 宝可梦表情符号, PAX Omeganauts Discord` 


- **寻找 Cohere Labs 联系方式！**：一位成员询问在哪里可以联系到 **Cohere Labs** 的人员，另一位成员建议就在这个 Discord 频道。
   - 另一位成员引导该用户访问 [此链接](https://discord.com/channels/954421988141711382/954421988783444043/1400866387668504648)。
- **Discord 频道宝可梦化！**：一位成员建议在频道中添加更多**宝可梦表情符号**，因为还有空余槽位。
   - 该成员提到这些表情符号来自 **PAX Omeganauts Discord**。


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1405640013198131345)** (5 messages): 

> `AI 研究, writenode, CV+ML 流水线` 


- **AI 研究员寻求合作**：一位对**推理和意识能力**有浓厚兴趣的 **AI 研究员**正在寻找合作机会，以开发面向未来的先进技术。
   - 该成员对任何子领域的合作都持开放态度。
- **法律专业人士转向 AI**：一位目前在美国政府工作的法律专业人士、游戏玩家和哲学爱好者正在自学 **AI 对齐理论与机制**。
   - 该成员很高兴能加入这里。
- **writenode 构建者使用 Cohere**：Josh 正在构建 **writenode**（一个*浏览器内的认知思维伙伴和创意伴侣*），并使用了 **Cohere**。
   - 在去年 12 月之前，他并没有开发者或编程背景。
- **心理学博士回归 AI**：一位成员在过去 5 年攻读人类心理学博士学位后，重新回归 **AI 研究**。
   - 他们的兴趣在于**声音+音乐**，以及利用技术工具帮助我们表达创造力。


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1405985104920055959)** (3 messages): 

> `Discord 邀请链接, 频道垃圾信息` 


- **Discord 邀请链接刷屏**：一名成员在频道中多次发布 [Discord 邀请链接](https://discordapp.com/invite/HjWfRbqBB8) 并艾特*所有人*。
   - 该邀请链接在短时间内重复出现了三次。
- **邀请链接重复**：同一个 [Discord 邀请链接](https://discordapp.com/invite/HjWfRbqBB8) 被反复发布。
   - 这导致了类似垃圾信息的效果，可能会干扰频道的正常讨论。


  

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1405984973906903060)** (3 messages): 

> `Discord Invite Link, HjWfRbqBB8, Channel Invitation` 


- **Discord 邀请链接刷屏**: 一名成员在频道中反复分享 [Discord 邀请链接](discordapp.com/invite/HjWfRbqBB8)，可能是为了吸引更多用户。
   - 该成员多次使用了 `@everyone` 标签，这可能被认为是过度或具有干扰性的。
- **频道邀请攻势**: 重复发布[相同的 Discord 邀请](discordapp.com/invite/HjWfRbqBB8)表明其试图增加频道成员数量。
   - 使用 `@everyone` 表明该消息旨在发送给所有成员，无论他们是否对该邀请感兴趣。


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1405660404918652948)** (2 messages): 

> `Elicitations Specification, MCP Server Conversion` 


- **寻求 Elicitations 规范的明确说明**: 一名成员询问了关于 [Elicitations 规范](https://modelcontextprotocol.io/specification/2025-06-18/client/elicitation) 中关于*谁*负责将消息/字段描述翻译成用户语言的问题。
   - 具体而言，他们寻求澄清：是应该由 **tools** 处理语言检测和国际化，还是期望 **MCP Clients** 进行翻译（可能通过使用 LLM）。
- **MCP Server 转换问题**: 一名成员询问：*是否存在某种工具可以将本地 MCP Server 转换为远程 MCP Server？*
   - 未提供链接或更多上下文。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1405750824461668434)** (3 messages): 

> `Unifi MCP, Unraid MCP, Syslog MCP, AI Agent Workflows, AI Security` 


- **面向 Homelab 用户的 MCP Server 发布**: 一名成员为 Homelab 用户分享了几个 MCP（推测为 **Management Control Panel**）Server，具体包括：[Unifi MCP](https://github.com/jmagar/unifi-mcp)、[Unraid MCP](https://github.com/jmagar/unraid-mcp) 和 [Syslog MCP](https://github.com/jmagar/syslog-mcp)。
- **PulseMCP 将繁琐的新闻简报工作流转变为 Agent 自动化**: **PulseMCP** 使用 goose 将繁琐的新闻简报工作流转变为由 Agent 驱动、人工参与（human in the loop）的自动化流程。
   - 有关该自动化的更多细节可以在[这篇博客文章](https://block.github.io/goose/blog/2025/08/13/pulse-mcp-automates-recipe)中找到。
- **AI Security 寻求关于安全问题的反馈**: 一名成员发布了关于构建 **AI Security** 的消息，旨在通过数学上的安全确定性在攻击开始前将其阻止。
   - 他们正在寻求开发者对安全问题的反馈，并链接到了[一份调查问卷](https://form.typeform.com/to/xTKa05F9)。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1405631570194337793)** (4 messages): 

> `Strix Halo profitablility, Dolphin chat template, Quantum computers, PC Memory` 


- **Strix Halo 的盈利能力大幅下降**: 尽管 **Strix Halo** 规格惊人，但由于其推理速度（**53 tokens/sec**）慢于 **OpenRouter** 上的 **GPT-OSS 120B**，需要 **24/7 全天候推理一年**才能实现盈利。
   - 一位用户指出，花费 2000 美元将其配置用于 **LLMs** 的效率远低于提供 **200-400 tokens/sec** 的云端替代方案。
- **寻找 Dolphin 聊天模板**: 一位用户正在为 **gpt4all** 寻找适用于 **Dolphin-2.2.1-mistral-7b-gptq** 的可用聊天模板。
   - 另一名成员建议请求模型制作者上传带有 **jinja** 模板的模板。
- **量子计算“茶匙”？**: 一位用户推测了量子计算机未来的可用性，以及**按茶匙售卖量子比特（qubits）**的可能性。
   - 他们提到了关于**全功能量子计算机**的新闻，表明该领域可能取得了进展。
- **内存模块与摩尔定律**: 一位用户提到，传统的 PC 有望在 2027 年底或 2028 年看到**更高容量的内存模块**和 **DDR6**。
   - 他们对具有高 RAM 和 VRAM 容量的微型 PC 的潜力表示兴奋，特别是对于小型企业而言。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1406014763804397599)** (1 messages): 

> `Maternity Leave, Team Contact During Leave` 


- **产假开始！**: 一名成员宣布他们将从 **8 月 25 日**起休**产假**，直至 **2026 年 2 月**。
   - 他们期待在回归后与大家交流。
- **团队交接计划公布**: 在该成员休假期间，团队将负责监控 <@1334161614949056532>。
   - 成员如有任何问题或疑虑，也可以联系 <@709918328306663424>。


  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/)** (1 条消息): 

__nathan: <@132818429022437376> 进展如何？
  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1405634992792670208)** (1 条消息): 

> `Windsurf Wave 12, DeepWiki Integration, Vibe and Replace feature, Smarter Cascade Agent, Dev Containers Support` 


- **Windsurf Wave 12 正式发布！**: Windsurf Wave 12 首次将 **Devin 的智能**和能力直接集成到 Windsurf IDE 中。
   - 核心功能包括 **全新的 UI 设计**、**DeepWiki Integration**、**Vibe and Replace**、**更智能的 Cascade Agent**、**Faster Tab**、**Dev Containers 支持**以及 **100 多项错误修复** —— [查看变更日志](https://windsurf.com/changelog)，[阅读博客](https://windsurf.com/blog/windsurf-wave-12)，[观看 Wave 12 视频](https://www.youtube.com/watch?v=-7gm8mST9QU)，[X/Twitter](https://x.com/windsurf/status/1956074019393876280)，以及 [Reddit](https://www.reddit.com/r/windsurf/comments/1mqal3x/wave_12_released_fresh_ui_deepwiki_vibe_and/)。
- **DeepWiki Integration 为 IDE 带来 AI**: **DeepWiki Integration** 允许用户将鼠标悬停在代码符号上以获取 **AI 驱动的解释**（而不仅仅是基础类型信息）。
   - 用户还可以使用 **CMD/Ctrl+Shift+Click** 在侧边栏中打开详细解释，并将其添加到 Cascade 上下文中。
- **Vibe and Replace 彻底改变了批量编辑**: **Vibe and Replace** 功能通过查找精确的文本匹配，提供了革命性的批量编辑能力。
   - 它允许用户应用 **AI prompts**，在整个项目中进行智能且感知上下文的转换。
- **更智能的 Cascade Agent 获得全天候规划功能**: **更智能的 Cascade Agent** 现在具备全天候规划模式，并带有自主待办事项列表。
   - 它还包括经过改进的工具，旨在提供更智能的响应。
- **原生支持 Dev Containers**: Windsurf 现在支持通过远程 SSH 访问直接使用容器。
   - 这一增强简化了涉及容器化环境的开发工作流。