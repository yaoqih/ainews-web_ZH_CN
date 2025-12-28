---
companies:
- google
- huawei
- epoch-ai
- deutsche-telekom
- nvidia
- anthropic
- reka-ai
- weaviate
- deepmind
date: '2025-11-04T05:44:39.731046Z'
description: '以下是该文本的中文翻译：


  **谷歌的 Project Suncatcher** 项目正在研发轨道上的可扩展机器学习（ML）计算系统原型，该系统利用太阳能供电，并采用能够抵御辐射的 Trillium
  代 TPU，目标是在 2027 年前推出原型卫星。**中国为数据中心提供 50% 的电力补贴**，这可能弥补芯片效率方面的差距；同时，**华为**计划在 2027
  年前为 DeepSeek 建设吉瓦（GW）级的 SuperPoD 集群。**Epoch** 推出了一个开放的数据中心追踪中心，**德国电信（Deutsche Telekom）**与**英伟达（NVIDIA）**宣布在慕尼黑投资
  11 亿美元建设拥有 1 万个 GPU 的设施。


  在智能体技术栈方面，**MCP**（模型-计算-平台，注：通常指 Model Context Protocol）工具正受到关注，具体实现包括 **LitServe**、**Claude
  Desktop** 以及 **Reka 为 VS Code 提供的 MCP 服务器**。Anthropic 强调通过 MCP 实现高效的代码执行。**上下文工程（Context
  engineering）**的重点正从提示词编写转向模型输入优先级排序，来自 **Weaviate**、**Anthropic** 及从业者的报告和工具强调了遵循指令的重排序器（rerankers）和嵌入方法。DeepMind
  的 **IMO-Bench** 数学推理测试套件显示 **Gemini DeepThink** 获得了高分，其 ProofAutoGrader（自动证明评分器）与人工评分表现出强相关性。基准测试和治理方面的更新包括
  **lighteval** 中新增的任务和评估共享。'
id: MjAyNS0x
models:
- trillium
- gemini-2.5-pro
- gemini-deepthink
people:
- sundarpichai
- yuchenj_uw
- teortaxestex
- epochairesearch
- scaling01
- _avichawla
- rekaailabs
- anthropicai
- douwekiela
- omarsar0
- nityeshaga
- goodside
- iscienceluvr
- lmthang
title: 今天没发生什么特别的事。
topics:
- energy-efficiency
- datacenters
- mcp
- context-engineering
- instruction-following
- embedding-models
- math-reasoning
- benchmarking
- code-execution
---

**平静的一天。**

> AI 新闻（2025年11月3日-11月4日）。我们为您检查了 12 个 subreddit、544 个 Twitter 账号和 23 个 Discord 社区（200 个频道，6479 条消息）。预计节省阅读时间（以 200wpm 计算）：551 分钟。我们的新网站现已上线，提供完整的元数据搜索和精美的 vibe coded 呈现方式，涵盖所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上向我们提供反馈！
> 

连续第 4 个平静的日子...

---

# AI Twitter 综述

**计算、能源与 AI 数据中心**

- **Google 的 Project Suncatcher（太空中的 TPU）**：Google 正在轨道上原型化可扩展的 ML 计算系统，以利用充足的太阳能。早期测试显示，Trillium 代 TPU 在粒子加速器辐射中幸存；下一个里程碑是到 2027 年初与 Planet 合作发射两颗原型卫星。提到的关键挑战：热管理和在轨可靠性。反应将其定性为将 AGI 视为一个能源问题，通过将计算移至“离太阳更近”的地方来获益 [@sundarpichai](https://twitter.com/sundarpichai/status/1985754323813605423), [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1985760405147566166)。
- **补贴与吉瓦级建设**：多份笔记指出，中国新的 50% 电费补贴可能会在每 FLOP 成本层面消除效率差距，能源价格支持抵消了芯片效率的劣势；声明还提到华为计划到 2027 年建立专门用于 DeepSeek 的吉瓦级 SuperPoDs [@teortaxesTex](https://twitter.com/teortaxesTex/status/1985540154065318157), [@teortaxesTex](https://twitter.com/teortaxesTex/status/1985567870227460166)。与此同时，Epoch 推出了一个开放的“前沿数据中心枢纽 (Frontier Data Centers Hub)”，通过卫星图像和公开文件追踪 1 GW+ 的 AI 数据中心，数据免费发布 [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1985788184245293153)。另外，德国电信和 NVIDIA 宣布在慕尼黑投资 11 亿美元建设设施，配备 1 万个 GPU（DGX B200 + RTX Pro）[@scaling01](https://twitter.com/scaling01/status/1985741851991621712)。

---

**Agent 堆栈、MCP 与上下文工程**

- **MCP 无处不在（工具作为一等接口）**：赋能工具的 Agent 实践模式正向 MCP 整合。一份实操指南展示了如何使用 LitServe（基于 FastAPI）在大约 10 行代码内将任何模型/RAG/Agent 作为 MCP 服务器运行，并将其接入 Claude Desktop [@_avichawla](https://twitter.com/_avichawla/status/1985595667079971190)。Reka 发布了一个免费的 MCP 服务器，在 VS Code 内提供搜索/事实核查功能 [@RekaAILabs](https://twitter.com/RekaAILabs/status/1985794490116780052)。Anthropic 分享了使用 MCP 进行代码执行的工程指导，强调使用更多工具可以降低 Token 消耗 [@AnthropicAI](https://twitter.com/AnthropicAI/status/1985846791842250860)。
- **从提示词工程到上下文工程**：几条推文讨论了从“用户应该写什么”到“模型应该读什么”的转变。亮点包括：遵循指令的重排序器（rerankers）作为上下文优先级排序的关键控制点，优于朴素检索 [@douwekiela](https://twitter.com/douwekiela/status/1985756688000163892)；一份 41 页的上下文工程蓝图（涵盖 Agent、查询增强、检索、提示词、记忆、工具）[@weaviate_io](https://twitter.com/weaviate_io/status/1985741429579170276)；一份面向从业者的“上下文工程 2.0”报告，展望了主动型 Agent [@omarsar0](https://twitter.com/omarsar0/status/1985747789796483109)；以及一种用于 Tools-to-Agent 检索的统一嵌入方法，在 LiveMCPBench 上取得了显著提升 [@omarsar0](https://twitter.com/omarsar0/status/1985745152204554720)。一个值得注意的 UX 模式：Anthropic 的 Claude Code 使用显式的 AskUserQuestion 工具而不是仅靠提示词行为来获取澄清 [@nityeshaga](https://twitter.com/nityeshaga/status/1985707959486472268)。同样值得回顾的是：被表述为“Prompt vs 上下文工程”的概念划分 [@goodside](https://twitter.com/goodside/status/1985583995644497931)。

---

**推理、数学与评估**

- **DeepMind 的 IMO-Bench (数学推理套件)**：GDM 发布了 IMO-AnswerBench (答案)、IMO-ProofBench (证明写作) 和 IMO-GradingBench (LLM 评分)。在 ProofBench 上，Gemini DeepThink (IMO 金牌赛道) 在基础集上达到 89.0%；大多数模型得分 <60%。在高级集上，非 Gemini 模型得分 <25%，而其内部最佳模型通过人工评估达到 65.7%。使用 Gemini 2.5 Pro 的 ProofAutoGrader 与人工评分高度相关 (在公开基础/高级集上 Pearson 系数为 0.96/0.93；在 170 个内部系统上为 0.87) [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1985685404276965481), [@lmthang](https://twitter.com/lmthang/status/1985760224612057092), [后续更新](https://twitter.com/lmthang/status/1985772094085595570)。
- **基准测试与治理**：
    - lighteval 增加了基准测试查找器 (benchmark finder)、inspect-ai 集成、评测共享以及新任务 (gsm_plus, MMLU redux, 菲律宾语, ifbench, slr-bench 等) [@nathanhabib1011](https://twitter.com/nathanhabib1011/status/1985720151673880923)。
    - OSWorld 维护者澄清了任务难度跨度从简单的 GUI 编辑到多应用工作流；“步数 (step count)” 指标低估了复杂性 [@TianbaoX](https://twitter.com/TianbaoX/status/1985647751468892434)。
    - OpenAI 的 IndQA (2,278 个问题，12 种印度语言) 旨在提升英语之外的文化/语境胜任力 [@snsf](https://twitter.com/snsf/status/1985719755551158754)。
    - ARC Prize 宣布了 ARC-AGI Verified，包含第三方学术审计，并为 ARC-AGI-3 引入了新赞助商 [@arcprize](https://twitter.com/arcprize/status/1985802145300693140), [@GregKamradt](https://twitter.com/GregKamradt/status/1985804827063210244)。

---

**机器人与物理 AI (Robotics and Physical AI)**

- **GEN-0 机器人 FM (Harmonic Reasoning, 10B+ 参数)**：Generalist AI 发布了一个大型机器人基础模型 (Foundation Model)，在超过 270,000 小时的灵巧操作数据上训练而成。他们报告了强大的缩放法则 (scaling laws)（更多的预训练 + 模型规模 = 更好的效果），并强调 “物理常识” (抓取、稳定、放置)。定位：由丰富数据驱动的通用机器人跳板 [@GeneralistAI](https://twitter.com/GeneralistAI/status/1985742083806937218), [@E0M](https://twitter.com/E0M/status/1985760232170209583), [后续更新](https://twitter.com/E0M/status/1985766175255773483)。
- **生态系统**：新资源包括 PHUMA (基于物理的人形运动) [@_akhaliq](https://twitter.com/_akhaliq/status/1985716700541829276)，“使用视频基础模型进行世界模拟” 以支持物理 AI [@_akhaliq](https://twitter.com/_akhaliq/status/1985722252412011006)，以及正在招聘世界模拟和以 VFM 为中心的研究团队 [@jparkerholder](https://twitter.com/jparkerholder/status/1985729367469596843)。

---

**本地推理与开发工具 (Local inference and dev tooling)**

- **llama.cpp 的新 WebUI**：为 150k+ GGUF 模型提供精致、适配移动端的本地聊天体验，支持 PDF/图像输入、对话分支、JSON-schema 约束生成、数学/代码渲染以及并行聊天。被广泛赞誉为 “本地 AI 之最” 的基准 [@ggerganov](https://twitter.com/ggerganov/status/1985727389926555801), [@ClementDelangue](https://twitter.com/ClementDelangue/status/1985748187634717026), [@victormustar](https://twitter.com/victormustar/status/1985742628776706151)。
- **MLX 和吞吐量提升**：MLX-Swift 正在为本地多流推理增加连续批处理 (continuous batching)（在新请求到达时自动将单请求流升级为批处理）[@ronaldmannak](https://twitter.com/ronaldmannak/status/1985693207003275729)。另外，一位著名的开源工程师加入 Apple 全职从事 MLX 工作 [@zcbenz](https://twitter.com/zcbenz/status/1985560798543167739)。在云端 IDE 方面，Cursor 发布了 UI 和 LSP 性能升级 [@cursor_ai](https://twitter.com/cursor_ai/status/1985791854739390591)。GitHub Copilot 报告称，通过更快的自定义模型，Token 吞吐量提高了 3 倍，采纳率提高了 12%，延迟降低了 35% [@github](https://twitter.com/github/status/1985737580613140747)。
- **推理系统：解耦与新范式**：来自 DistServe 作者的回顾追踪了 Prefill-Decode 解耦 (disaggregation) 如何成为现代 LLM 服务的支柱，实现了 10–100 倍的成本降低以及显著的吞吐量/延迟提升 [@haoailab](https://twitter.com/haoailab/status/1985753711344316648)。vLLM 继续快速覆盖：支持 PaddleOCR-VL，并在 nightly 版本中运行 Ouro (循环潜空间推理 LM) [@vllm_project](https://twitter.com/vllm_project/status/1985589446197330129), [@vllm_project](https://twitter.com/vllm_project/status/1985695123469209703)，以及关于 “PD” (Prefill-Decode) 谱系的背景信息 [@vllm_project](https://twitter.com/vllm_project/status/1985761953432944893)。

---

**多模态与视频生成 (Multimodal and video generation)**

- **Qwen 更新与部署说明**：Qwen3‑VL 已集成到 Jan，并持续推出支持思考能力的 API（在 qwen3-max-preview 上设置 `enable_thinking=True`）。一条有用的实战笔记：转换框架至关重要——相同的 Qwen3‑VL 量化版本在 Ollama 和 MLX 上的结构化提取准确率存在实质性差异；在投入 prod 之前，请务必评估目标技术栈 [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1985542635373937102), [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1985586316197937256), [@andrejusb](https://twitter.com/andrejusb/status/1985612661447331981)。
- **视频模型与 UX**：
    - Vidu Q2 在 Artificial Analysis 排行榜上首秀排名第 8，支持多参考图条件控制（multi‑reference image conditioning）并可输出 8 秒 1080p 视频；API 定价介于 Hailuo 02 Pro 和 Veo 3.1 之间 [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1985781760236630305)。
    - MotionStream 展示了实时、交互式、长时长的视频生成，可通过拖拽手势控制，在单张 H100 上达到 29 FPS 和 0.4s 延迟 [@xxunhuang](https://twitter.com/xxunhuang/status/1985806498811789738)。
    - Sora Android 应用可用范围扩大（CA/JP/KR/TW/TH/US/VN） [@soraofficialapp](https://twitter.com/soraofficialapp/status/1985766320194142540), [@soraofficialapp](https://twitter.com/soraofficialapp/status/1985849973830046152)。Microsoft 在 Bing Image Creator 和 Copilot Labs 中推出了 MAI‑Image‑1，旨在实现更高的写实感和艺术控制力 [@mustafasuleyman](https://twitter.com/mustafasuleyman/status/1985777196460622327)。

---

**热门推文（按互动量排序）**

- Google 的 Project Suncatcher：太空中的 TPU；计划在 2027 年前发射两颗原型卫星 [@sundarpichai](https://twitter.com/sundarpichai/status/1985754323813605423)。
- Anthropic 为 Pro/Max 用户提供临时的免费 Claude Code 网页版额度 [@_catwu](https://twitter.com/_catwu/status/1985754411675930775)。
- Anthropic 与冰岛教育部合作开展国家级 AI 教育试点项目 [@AnthropicAI](https://twitter.com/AnthropicAI/status/1985612560255893693)。
- 新的 llama.cpp WebUI 落地；被广泛赞誉为本地 AI UX 的里程碑 [@ClementDelangue](https://twitter.com/ClementDelangue/status/1985748187634717026), [@ggerganov](https://twitter.com/ggerganov/status/1985727389926555801)。
- Generalist AI 的 GEN‑0，一个基于 27 万多小时数据训练的 10B+ 机器人基础模型 [@GeneralistAI](https://twitter.com/GeneralistAI/status/1985742083806937218)。

---

# AI Reddit 综述

## /r/LocalLlama + /r/localLLM 综述

### 1. Qwen 模型生态系统的影响力

- [**Qwen 如今几乎能与整个美国开源模型生态系统相媲美**](https://www.reddit.com/r/LocalLLaMA/comments/1onzrg9/qwen_is_roughly_matching_the_entire_american_open/) (热度: 1240): **该图片展示了 Qwen 模型系列计划发布的路线图，突显了其在开源模型生态系统中的重要地位，特别是与美国开源模型的对比。该路线图包含了 Qwen2.5-1M、Qwen3 和 Qwen3-VL 等模型，预示着到 2025 年都有强劲的开发计划。这使 Qwen 成为 AI 领域的主要参与者，潜力足以与 GPT-OSS 20B 和 120B 等美国模型竞争。带有 Qwen 标志的俏皮卡通熊为这份技术演示增添了一抹轻松的色彩。** 一位评论者质疑 Qwen 的贡献是否等同于美国开源模型生态系统，特别是将其与 GPT-OSS 20B 和 120B 等模型进行比较，这表明人们对这些模型的影响力和重要性存在争议。
    - 一位用户强调了中国 AI 模型（特别是 Qwen）在全球 AI 领域的统治地位，并指出中国研究人员多年来一直是 AI 研究的重要贡献者。他们认为《欧盟 AI 法案》阻碍了西方 AI 的发展，使中国成为实现技术自由的主要力量。该评论还批评了可能扼杀创新的西方政治决策，并与中国在 AI 方面的进展形成对比。
    - 另一位用户分享了比较 GPT-OSS-20B 和 Qwen-2.5 的个人经验，指出他们发现 GPT-OSS-20B 在 3060 GPU 上运行时表现平平，导致他们换回了 Qwen。这表明 Qwen 在某些硬件配置上可能提供更好的性能或效率，尽管该用户推测更大的 GPT-OSS 模型可能表现更好。
    - 关于美国开源模型贡献的讨论也随之展开，一位用户质疑 GPT-OSS 20B 和 120B 等模型是否代表了美国开源模型生态系统的全部。这引发了人们对美国贡献的广度和影响力与 Qwen 等中国模型进展相比的疑问。
- [**对 dgx spark 感到失望**](https://www.reddit.com/r/LocalLLaMA/comments/1oo6226/disappointed_by_dgx_spark/) (热度: 819): **图片描绘了一台 NVIDIA DGX Spark 设备，尽管它拥有 128GB 共享 RAM，但用户发现其在 VLLM 上运行带有上下文的 Qwen-30B 模型时性能不尽如人意。用户将其与 NVIDIA 3090 GPU 进行了负面对比，指出 DGX Spark 的设计并不能弥补其原始速度的不足，尤其是考虑到其 5,000 美元的价格。评论表明，该设备的受众群体在于其 RAM 容量而非速度，且人们原本就预期它会比 3090 等高端 GPU 慢。** 评论者普遍认为，不应指望 DGX Spark 的性能超过 3090 等高端 GPU，并强调其利基使用场景集中在 RAM 容量而非速度。
    - No-Refrigerator-1672 强调 DGX Spark 的规格清楚地表明它无法与专用 GPU 的性能相匹配，这暗示其市场定位非常有限。这意味着潜在买家应合理管理对其计算能力的预期。
    - Particular_Park_391 指出，DGX Spark 的主要价值在于其 RAM 容量而非速度，并承认它本就该比 X090 系列模型慢。这表明其设计更适合内存密集型任务，而非高速计算。
    - bjodah 注意到 DGX Spark 显著的 fp64 性能，这对于使用 CUDA 的科学计算尤为重要。这表明虽然它在通用 GPU 任务中表现不佳，但在高精度计算方面具有特定优势。

### 2. llama.cpp WebUI 发布

- [**llama.cpp 发布全新官方 WebUI**](https://www.reddit.com/r/LocalLLaMA/comments/1ooa342/llamacpp_releases_new_official_webui/) (热度: 1084): **llama.cpp 发布了由共同维护者 Alek 开发的全新官方 WebUI，旨在提升用户体验并对标专有 LLM 行业标准。该 WebUI 与现有工作流集成，并包含旨在提高响应速度的性能优化。鼓励社区提供反馈以进一步完善该工具。更多详情请参阅 [讨论](https://github.com/ggml-org/llama.cpp/discussions/16938)。** 社区反馈强调了该 WebUI 的显著进步和易用性。人们对扩展多模态能力（如视频和音频输出）表现出浓厚兴趣，尽管大家也承认工具的实现可能因具体用例而异。
    - llama.cpp 的共同维护者 Alek 强调了该项目的目标是在 UX 和功能上对标专有 LLM，并对社区（特别是 u/serveurperso）的重大贡献表示认可。重点在于增强 WebUI 以改善用户体验和功能。
    - YearZero 讨论了为 llama.cpp 的 WebUI 扩展多模态能力（如视频、图像和音频输出）的潜力。他们指出，由于缺乏通用解决方案，实现工具和检索增强生成 (RAG) 存在挑战，但表示有兴趣利用 Qwen3-VL 等模型来实现这些功能。
    - Due-Function-4877 建议增加 “llama-swap” 功能以方便使用多个模型，这可以增强 WebUI 的灵活性和能力。他们强调需要一个用户友好的界面来配置和启动服务器，从而减少对复杂命令行参数的依赖。

## 较少技术性的 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. AI Communication Innovations

- [**LLMs can now talk to each other without using words**](https://www.reddit.com/r/OpenAI/comments/1oo3l1n/llms_can_now_talk_to_each_other_without_using/) (Activity: 813): **图片是一份标题为 "Cache-to-Cache: Direct Semantic Communication Between Large Language Models" 的文档，介绍了一种名为 Cache-to-Cache (C2C) 的新范式，用于 LLM 之间的直接语义通信。这种方法绕过了传统的基于文本的通信，旨在提高准确性并降低延迟。文档指出，这种方法允许 LLM 通过直接共享语义信息来更有效地通信，可能会改变 AI 系统的交互方式。该方法的代码已在 GitHub 上提供，表明该领域正朝着开源协作的方向发展。** 一条评论将其与 AI 以向量形式交流的虚构场景联系起来，引发了对可审计性和控制的担忧。另一条评论质疑 10% 的改进，暗示可能存在瓶颈，而第三条评论指出共享内存的概念在嵌入式计算中早已存在，暗示将这些想法扩展到 LLM 是可行的。
    - Mayfunction 强调了 Transformer 中 Key-Value 表示的技术基础，这对其性能至关重要。这种表示允许模型编码比纯文本更多的信息，如语法角色和句子位置，使模型更容易处理查询。讨论指出，共享 Key-Value 表示比文本更有效，因为文本生成会导致信息丢失和计算需求增加。
    - Last_Track_2058 提到共享内存概念在嵌入式计算中早已存在，暗示将这些想法扩展到 AI 通信并不是一个新挑战。这意味着非语言 AI 通信的技术基础已在其他计算领域奠定，可能会简化向更高级 AI 交互的过渡。
    - Bishopkilljoy 引用了 AI 2027 论文，该论文警告 AI 可能会开发出超出人类理解能力的通信方法。这突显了 AI 发展中的一个关键担忧：确保 AI 通信的透明度和可审计性，以防止意外后果，呼应了关于 AI 自主性和控制的科幻主题。
- [**Superhuman chess AIs now beat human grandmasters without a queen**](https://www.reddit.com/r/OpenAI/comments/1oo3rqf/superhuman_chess_ais_now_beat_human_grandmasters/) (Activity: 1119): **图片及随后的讨论突出了 Leela Chess Zero 的能力，这是一款超人类象棋 AI，即使在处于重大子力劣势（如没有皇后）的情况下，也能击败人类大宗师。图片中的图表显示了在各种子力劣势下，对阵该 AI 达到 50% 胜率所需的预估等级分，强调了 Leela 在快棋和超快棋赛制中的实力。与 Stockfish 等传统引擎不同，Leela 是通过在“让子赛”场景中进行自我对弈训练的，这使其能够适应并在起始棋子较少的情况下积极进攻，而传统引擎由于对这类局面不熟悉，往往难以应对。** 评论者指出，虽然 Leela 的表现令人印象深刻，但它主要在快棋和超快棋赛制中有效，经典赛制仍然对人类有利。此外，Leela 对神经网络和自我对弈的使用使其区别于传统引擎，使其能够更有效地处理子力劣势。
    - Leela Chess Zero 是一款利用神经网络的高度优化象棋引擎，与 GPT 等通用 AI 模型不同。它通过自我对弈训练在“让子赛”中表现出色，使其比 AlphaZero 或 Stockfish 等传统引擎能更有效地处理棋子较少的局面，后者在这些场景中往往表现得比较保守。
    - 讨论强调，Leela 在“让子赛”中的训练涉及学习冒险和虚张声势，这与其它引擎在面对陌生局面时的防御策略不同。这种策略调整使 Leela 在快棋和超快棋赛制中表现优异，尽管由于样本量较小，经典象棋仍然对人类棋手有利。
    - 帖子澄清说，该 AI 的表现仅限于象棋引擎，与 OpenAI 等公司的通用 AI 模型无关。重点在于 Leela 即使在重大子力劣势下也能战胜人类大宗师的能力，展示了象棋引擎与通用 AI 相比的专业化性质。

### 2. AI 在媒体与广告中的应用

- [**可口可乐今年的年度圣诞广告再次由 AI 生成。该公司表示，制作过程中使用的人力更少——“我们需要继续前进并突破极限……精灵已经从瓶子里出来了，你无法再把它放回去”**](https://www.reddit.com/r/singularity/comments/1ooarax/cocacolas_annual_christmas_advert_is_aigenerated/) (Activity: 928): **可口可乐发布了其 2025 年圣诞广告，该广告由 AI 生成，标志着生产过程中减少人力投入的持续趋势。该公司强调这是创新的一步，并表示在广告中使用 AI 是不可逆转的趋势。与往年相比，该广告在质量和长度上都有所提升，展示了 AI 能力的重大进步。更多详情请参阅原始帖子 [此处](https://x.com/DiscussingFilm/status/1985470088074375344)。** 评论者对 AI 进步可能导致的大规模失业表示担忧，并提出了全民基本收入（UBI）等解决方案。其他人则注意到广告质量的显著提升，预测到 2030 年 AI 生成媒体将有进一步的发展。
    - UstavniZakon 强调了可口可乐 AI 生成的圣诞广告与去年相比在质量和长度上的显著提升，暗示了 AI 能力的快速进步。这意味着未来迭代中可能会有更大的增强，反映了创意产业中 AI 的快速演变。
    - SleepingCod 预测到 2030 年，AI 将能够制作完整的专业电影和电视节目，表明了对内容创作领域 AI 快速进步的信心。这一评论强调了 AI 在娱乐行业的变革潜力，暗示了未来 AI 可能在影视制作中扮演重要角色。
    - Haunt_Fox 表达了对 AI 和 CGI 看法的转变，指出最初不愿接受 CGI 而非传统的 2D 动画。这种态度的转变反映了媒体对 AI 技术更广泛的接受和适应，突显了 AI 生成内容的进步如何逐渐克服最初的怀疑。
- [**福克斯新闻（Fox News）误信 AI 生成的贫困人口因粮食券停发而愤怒的片段，发布了虚假报道，随后不得不进行重大更正**](https://www.reddit.com/r/ChatGPT/comments/1oo3zqg/fox_news_falls_for_aigenerated_footage_of_poor/) (Activity: 670): **福克斯新闻错误地播放了描绘人们抗议粮食券停发的 AI 生成片段，随后进行了更正。该片段最初被当作真实内容呈现，导致了虚假叙事并需要进行重大撤回。这一事件突显了媒体机构在播出前核实 AI 生成内容所面临的挑战。** 评论者认为福克斯新闻有意将误导性信息作为一种策略，暗示尽管后来进行了更正，但最初的虚假报道符合其议程。这反映了对媒体操纵以及 AI 在传播虚假叙事中作用的更广泛担忧。

### 3. 个人与教育背景下的 AI

- [**恶搞了我的岳父**](https://www.reddit.com/r/ChatGPT/comments/1oo7awm/pranked_my_father_in_law/) (Activity: 2047): **这张图片是一个幽默的恶搞，涉及使用 ChatGPT 修改厨房墙壁的照片，使其看起来像是被子弹孔严重损坏，并有一个露出木梁和电源插座的大洞。这个恶搞是在原帖作者向岳父寻求寻找墙柱的建议后执行的，展示了 AI 在图像处理方面的创意用途。这次恶搞突显了 AI 为了喜剧效果而修改图像的能力，尽管正如一位评论者所指出的，这也引发了关于伦理使用的疑问，因为该评论者的 AI 由于潜在的滥用担忧而拒绝了类似的请求。** 一位评论者指出，他们尝试使用 ChatGPT 进行类似恶搞的请求被拒绝了，可能是出于对保险欺诈的担忧，这表明 AI 中编程了一定程度的伦理考量。另一位评论者分享了一个个人轶事，关于使用 AI 在照片中添加一只额外的猫来恶搞他们的伴侣，说明了 AI 在日常生活中的多样化和创意应用。
- [**作为一名教育工作者，没有什么比这更真实了。那些本就厌学的学生现在完全放弃了。**](https://www.reddit.com/r/ChatGPT/comments/1oo4b32/as_an_educator_nothing_rings_truer_students_who/) (Activity: 1494): **这张图片是一个迷因（meme），强调了对学生过度依赖 AI 完成学业任务的担忧，这可能导致学习中缺乏努力和好奇心。"Boze the Library Owl" 的推文建议，这种依赖可能导致学生错过培养基本技能和个人成长的机会。"finn" 的幽默回复通过建议将 AI 作为作业的快速解决方案来强调这一问题，反映了关于技术对教育影响的更广泛辩论。** 评论者对传统教育在快速技术进步面前的相关性表示担忧，认为今天学到的技能可能会过时。还有一种观点认为，作业在历史上一直被视为低效的，一些学生在历史上根本不参与其中。
    - clawstuckblues 强调了教育中的一个关键问题：技术进步的快速步伐可能使当前的技能和知识在学生进入职场时变得过时。这给教育工作者带来了挑战，他们难以保持课程的相关性，并有效地让学生为未来的就业市场做好准备。
    - Mr_Michael_B99 主张通过取消作业来实现教育范式的转变，他认为作业助长了过度工作和倦怠的文化。他建议所有的学习都应该在教室内进行，以防止学生依靠 AI 走捷径。他将此与历史上对计算器的抵制进行了类比，暗示如果整合得当，AI 同样可以成为一种必不可少的教育工具。
    - SpartanG01 批评当前的教育系统贬低了自身内容并削弱了学生的生活前景。这一评论暗示教育系统的体制性失败导致了学生的脱节，表明问题的根源在于系统无法适应并保持其相关性和价值。

---

# AI Discord 简报

> 由 X.ai Grok-4 总结的总结之总结
> 

**主题 1：模型排名竞争激烈**

- **Minimax M2 冲上排行榜巅峰**：**Minimax M2** 在 [WebDev Leaderboard](https://lmarena.ai/leaderboard/webdev) 上攀升至**总榜第 4 名**和**开源模型第 1 名**，以低成本在编码、推理和 Agent 任务中的出色表现令用户惊叹。对其速度和效率的赞誉纷至沓来，引发了在 [LMArena models](https://lmarena.com/) 上恢复 Lithiumflow 的呼声。
- **Qwen 模型幻觉出离奇事实**：根据使用 [IFEval framework](https://arxiv.org/abs/2311.07911) 的 [LLM Propensity Evals space](https://huggingface.co/spaces/PropensityLabs/LLM-Propensity-Evals) 评估显示，**Qwen 模型**幻觉出罕见事实的频率几乎是 **Llama** 对应模型的两倍。尽管如此，**Qwen3 8B** 在指令遵循方面表现出色，超越了体量更大的 **GPT OSS 20B**。
- **BlackHawk 发出未过滤的右翼言论**：**BlackHawk** 模型因其无过滤、充满脏话的右倾输出引发了辩论，用户将其描述为“为了好玩而制作的另类右翼鹦鹉”，“零过滤且脏话连篇”。关于 GPT-5 Juice 的传闻细节模糊，且声称输出成本高达 **120 美元**。

**主题 2：硬件引发热议**

- **Tinybox Pro V2 发布天价工作站**：George Hotz 揭晓了 **tinybox pro v2**，这是一款配备 **8x 5090** 的机架式工作站，售价 **$50,000**，可通过 [tinycorp shop](https://tinycorp.myshopify.com/products/tinybox-pro-v2) 订购，发货周期为 4-12 周。关于其价值与云端租赁的对比，以及未来升级至 **Blackwell 6000s** 的潜力，引发了激烈辩论。
- **GPU 云服务价格因短缺而飙升**：全球短缺将 neo clouds 的 GPU 费率推高至 **$2/GPU hour**，而 hyperscalers 则达到 **$7/GPU hour**，引发了对谁在支付溢价的怀疑。用户在处理 stable diffusion 等任务时更倾向于本地 AMD 显卡，驳斥了 GPT 关于其性能低下的说法。
- **MI50 期待 ROCm 复兴**：有关 **MI50** GPU 回归并可能支持 ROCm 的猜测正在酝酿，参考了 [ROCm roadmap](https://github.com/ROCm/TheRock/blob/main/ROADMAP.md)，同时人们也对其在本地 Kimi 环境下的价值提出疑问。GPU 购买建议指出，二手 **3090s** 或 **4090s** 是运行 LLMs 的最佳选择，并警告在领域快速变化期间不要盲目购买。

**Theme 3: Tools Tackle AI Workflows (工具应对 AI 工作流)**

- **Fenic 通过 OpenRouter 实现 Pipeline 魔法**：[Fenic dataframe API](https://github.com/typedef-ai/fenic) 与 OpenRouter 集成，以运行混合供应商的 AI 工作流，无缝扩展批处理并切换模型，用于 LLM ETL、context engineering 和 agent tooling。用户要求在 [OpenRouter charts](https://x.com/OpenRouterAI/status/17985371284411130035) 上增加一周过滤器，以获得更细致的使用洞察。
- **Codemaps 击碎代码乱象**：Windsurf 推出了 **Codemaps**，利用 **SWE-1.5** 和 **Sonnet 4.5** 交互式地映射代码库，打击 *code slop* 并提升生产力。ComfyUI 与 LM Studio 联动实现本地图像自动化，需要 **5 text boxes and samplers** 来拆分故事。
- **Tritex 在 Triton 中成功训练 LLMs**：[Tritex repo](https://github.com/martin-kukla/tritex) 实现了在 Triton 中从零开始进行 LLM 预训练，在 **A100 SXM** 上以 **57.5% MFU** 复制了 **GPT2 1.6B**，正如在 [Disaggregated Inference tweet](https://x.com/martin_kukla/status/17185687315801428364) 中分享的那样。Unsloth 发布了一个 [DeepSeek-OCR notebook](https://x.com/UnslothAI/status/1985728926556307471)，但用户反映微调后的错误率高达 100% 以上。

**Theme 4: Benchmarks Bash Flaws (基准测试抨击缺陷)**

- **Epoch AI 对 OSWorld 评测提出批评**：Epoch AI 在其 [Butter-Bench report](https://andonlabs.com/evals/butter-bench) 中抨击 **OSWorld benchmark** 任务过于简单且评估存在缺陷，呼吁对 AI agent 评估采用更严谨的方法。Gemma 模型出人意料地破解了验证码（captchas），引发了关于赞/踩按钮与评论反馈的辩论。
- **Roblox 分类器以闪电般的速度识别 PII**：Roblox 在 [Hugging Face](https://huggingface.co/Roblox/roblox-pii-classifier) 上开源了其 **PII Classifier**，每天处理 **61 亿条消息**，查询速度达 **200,000 queries/second**，**P90 latency** 低于 **100ms**，详见其 [newsroom post](https://corp.roblox.com/newsroom/2025/11/open-sourcing-roblox-pii-classifier-ai-pii-detection-chat)。人们对大规模下的调度器瓶颈表示担忧，对数据集的兴趣超过了模型本身。

**Theme 5: Legal and Safety Storms Brew (法律与安全风暴酝酿)**

- **Getty 在 AI 图像诉讼中受挫**：根据 [Reuters report](https://www.reuters.com/sustainability/boards-policy-regulation/getty-images-largely-loses-landmark-uk-lawsuit-over-ai-image-generator-2025-11-04/)，Getty Images 在针对 AI 生成器的英国诉讼中基本败诉，引发了关于权力集中的辩论。OpenAI 限制 ChatGPT 提供医疗/法律建议以规避诉讼，加剧了对监管过度扩张的抱怨。
- **Anthropic 对开源保持沉默**：用户对 [Anthropic's deprecation commitments](https://www.anthropic.com/research/deprecation-commitments) 表示担忧，希望能有像 **Miqu 70B** 这样的泄露，同时担心他们 *永远不会开源任何东西* 甚至会彻底禁止。一段 [YouTube video](https://www.youtube.com/watch?v=0Plo-zT8W9w) 中关于 AI 泡沫达到顶峰的说法被斥为 *一派胡言*，并以数据集增长的论点予以反驳。


---

# Discord: High level Discord summaries (Discord 高层级摘要)

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **LMArena 用户渴望 Lithiumflow 回归**：LMArena 平台的用户正请求将 **Lithiumflow** 模型重新加入可用[模型](https://lmarena.com)列表。
   - 一位用户表达了对它的怀念，称：*"我体验过 Lithiumflow 之后，现在忍不住开始想念它了 D:"*。
- **Minimax M2 飙升至第 4 位，登顶 WebDev Leaderboard**：**Minimax M2** 模型在 [WebDev Leaderboard](https://lmarena.ai/leaderboard/webdev) 上已达到**总榜第 4**，并成为**排名第 1 的顶级开源模型**，令许多人感到惊讶。
   - 成员们赞扬了它的**编程性能**、**推理能力**和 **Agent 风格任务**的处理能力，同时它还具有高性价比和快速的特点。
- **GPT-5 Juice 传闻流传，细节仍不明朗**：关于新款 **GPT-5 Juice** 的消息正在传播；然而，其功能和特性仍不为人知。
   - 一位用户声称其输出成本为 **$120**。
- **BlackHawk 模型因无过滤和右倾观点引发辩论**：由于缺乏过滤机制以及存在争议的右倾观点，**BlackHawk** 模型引发了讨论。
   - 一位成员表示它具有*"零过滤且经常说脏话"*，另一位成员将其描述为*"为了好玩而制作的另类右翼复读机"*。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro 用户面临图像限制上限**：一些 **Perplexity Pro** 订阅者遇到了**图像生成限制**，一位用户报告上限仅为 **20 张图像**，这可能是由于频率限制或与 Airtel 订阅相关的限制。
   - Perplexity 官方支持建议等待限制重置或直接联系支持团队。
- **Comet 助手受故障困扰**：用户报告**内部错误**导致黑屏，以及 **Comet** 助手的功能问题。
   - 故障排除步骤包括重新安装 **Comet**、回退到旧版本或重启应用程序，但这引发了用户的沮丧。
- **GPT Go 在印度免费开放**：作为提高采用率和收集数据活动的一部分，**GPT Go** 计划对印度用户免费开放。
   - 该举措旨在利用人口众多的优势来增加产品使用量和数据收集。
- **HTML 在 Web 可访问性方面优于 ARIA**：一位屏幕阅读器用户主张使用正确实现的 **HTML** 而非 ARIA 标签，他们发现后者可能导致冗余和令人困惑的重复。
   - 他们建议使用简短且具有描述性的 alt 文本，并正确使用标题和段落标签以实现最佳可访问性，参考了 [NV Access 用户指南](https://download.nvaccess.org/documentation/userGuide.html)。
- **Perplexity API 用户希望获得 Sonar Pro Search**：用户请求通过 **Perplexity API** 访问 **Perplexity: Sonar Pro Search**，并指出其目前已在 **OpenRouter** 上可用。
   - 关于 **Sonar Pro Search** 在 **OpenRouter** 上表现的正面反馈表明，用户对其集成到 **Perplexity API** 具有浓厚兴趣。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 图表通过用户/API Key 分组实现细粒度化**：[OpenRouter 图表](https://x.com/OpenRouterAI/status/17985371284411130035)现在提供按**用户**和 **API Key** 进行的活动分组，让用户能够更详细地了解使用模式和成本分配。
   - 用户请求增加**一周（one-week）**过滤选项以追踪每周使用趋势，如[此图](https://cdn.discordapp.com/attachments/1434932448453464075/1434997177410650204/image.png?ex=690bae44&is=690a5cc4&hm=b1108cb4f7de7b4bc8b3f2e11ca0ddf7bcceb4ae08ad1a39387cd186a9a60bdc)所示。
- **Fenic 集成 OpenRouter 以简化 AI 工作流**：[Fenic](https://github.com/typedef-ai/fenic) 是一个用于 **AI 工作流**的 dataframe API 和执行引擎，现在已支持 **OpenRouter**，允许你在单个会话中运行混合供应商的流水线。
   - 该集成旨在清晰地扩展大批量任务，并在不更改流水线代码的情况下更换模型，支持 **LLM ETL**、**上下文工程**、**Agent 记忆**和 **Agent 工具化**。
- **云端 GPU 成本引发辩论**：成员们就为 AI 图像生成支付**云端 GPU 租赁**费用的价值展开了辩论，一些人更倾向于使用自己的 **AMD 显卡**进行较慢的本地操作。
   - 一位用户调侃道，他 *“认为在云端 GPU 上为 AI 投入一分钱都不值得”*。
- **GPT 关于 AMD 显卡的准确性受到质疑**：用户质疑 **GPT 关于 AMD 显卡**和模型性能的准确性，并引用了来自 Stable Diffusion 社区的反馈，称 **AMD 显卡是可行的**，尽管 GPT 声称并非如此。
   - 在 GPT 错误地表示 AMD 显卡在运行 Stable Diffusion 表现较差后，一位用户表达了对其缺乏信任。
- **Gemma 模型破解验证码表现惊人**：成员们观察到 **Gemma 模型**在解决验证码（captchas）方面出奇地有效。
   - 另一位用户建议在每个模型的每个供应商旁边实现一个赞/踩按钮以获取反馈，这引发了关于二元反馈与评论系统优劣的辩论，并展示了来自 [K2-Vendor-Verifier GitHub 仓库](https://github.com/MoonshotAI/K2-Vendor-Verifier)的数据。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **BIM 设计架构**：成员们探讨了**建筑信息模型 (BIM)**，这是一种用于建筑和航空领域的参数化设计方法，允许使用富元数据对象而非简单的线条；详细信息可参阅[此处](https://www.autodesk.com/uk/products/revit/overview)。
   - BIM 简化了**冲突解决**、对象计数、尺寸提取并增强了协作，从而引发了关于 *Autodesk x Unsloth* 集成的建议。
- **仅使用文本微调视觉模型遇到困难**：一位成员分享了在仅使用文本微调视觉模型时遇到的挑战，这导致了初始 Loss 较高以及训练与评估 Loss 之间的发散，并向社区寻求帮助。
   - 建议包括**检查标签**、确保**视觉部分已冻结**、使用 **SFTTrainer** 计算指标，以及考虑使用 **UnslothLLM** 以获得更好的支持。
- **DeepSeek OCR 微调受挫**：Unsloth AI 在[此 X 帖子](https://x.com/UnslothAI/status/1985728926556307471)中宣布了一个新的 **DeepSeek-OCR 微调笔记本**。
   - 然而，一位成员指出微调后的错误率极高（超过 100%），由于预测文本与正确文本长度存在差异，对其效用提出了质疑。
- **Llama.cpp 占用 GPU-0**：用户讨论了 **llama.cpp** 倾向于将所有内容分配给 **GPU-0** 的倾向，寻求在双显卡设置下实现更好 GPU 利用率的解决方案。
   - 虽然有人建议使用 `--tensor-split` 和 `--override-tensor` 等方案，但用户发现效果不佳，并指出利用率不足和不平衡，评论称 *“llama.cpp 的推理表现很烂”*。
- **Roblox 的 PII 分类器开源**：Roblox 开源了其用于检测聊天中 PII（个人身份信息）的 **PII 分类器** AI，详情见其[新闻中心](https://corp.roblox.com/newsroom/2025/11/open-sourcing-roblox-pii-classifier-ai-pii-detection-chat)，模型可在 [Hugging Face](https://huggingface.co/Roblox/roblox-pii-classifier) 上获取。
   - 尽管每天处理 **61 亿条聊天消息**，该模型每秒处理超过 **200,000 次查询**，且 **P90 延迟**低于 **100ms**。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen 发布 30B 和 32B 模型**：**Qwen** 团队发布了 **30B** 和 **32B** 模型，其中一个是稠密模型，另一个是 **MoE**（Mixture of Experts），*llama.cpp* 已提供支持。
   - **30B** 模型是 **MoEQwen3 Next**，引发了人们对移动端 **LLM** 的期待。
- **GPU 购物清单**：对于运行 **LLM**，二手 **3090** 或 **4090** 是性价比最高的选择，但一位成员表示 *不要为了运行新的 LLM 而购买硬件，这个领域发展太快，不值得*。
   - 游戏方面，当前一代推荐 **5070ti**，而 **4070ti Super** 是不错的上一代选择（如果价格便宜），理由是 *上一代 GPU 经常会出现这种技术下沉（trickle down）现象*。
- **LM Studio 结合 ComfyUI**：成员们讨论了连接 **ComfyUI** 以实现本地 **Gemini Storybook** 替代方案，这可以通过在 **ComfyUI** 中运行 **LM Studio** 并本地使用 **LLM** 来实现。
   - 为了实现图像生成自动化，在 **ComfyUI** 中 *你需要 5 个文本框和 5 个采样器（sampler）* 来将故事拆分为不同部分并生成图像。
- **CUDA 故障触发运行时恐慌**：用户在 **LM Studio** 自动更新引擎（**llama.cpp**、**MLX**）后遇到问题，导致 *Failed to load model* 错误，特别是 **CUDA** 运行时（runtimes）方面。
   - 快速解决方法包括[回退到之前的引擎版本](https://discord.com/channels/1110598183144399058/1434078900098699445/1434182066647597169)，并澄清这些引擎更新比应用更新更频繁，可以在设置中禁用（**Ctrl + Shift + R**）。
- **MI50 卷土重来？**：用户询问 **MI50** 的当前价值和未来使用潜力，注意到其在 **ROCm** 中的支持有限。
   - 有人推测它可能会随着 **ROCm** 的支持而回归，尽管[链接的 GitHub 路线图](https://github.com/ROCm/TheRock/blob/main/ROADMAP.md)的可靠性存疑，以及本地运行 **Kimi** 的准入门槛价格。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **网页搜索被取代，用户更青睐 Perplexity**：用户报告 **@web** 功能已消失，取而代之的是通用的 *'use web search'* 命令，许多人选择使用 [perplexity.ai](https://perplexity.ai) 等外部工具以获得更好的结果。
   - 用户指出 **Cursor** 的 **MPC** 使用独立的 **API credits** 且仅支持 **Solar 模型**，限制了他们使用 **Sonnet 模型** 的能力。
- **GUI 创建中的模型合并策略辩论**：用户辩论了用于 **GUI/UI 创建** 的最佳模型，有人推荐 **Google AI Studio** 上的 **Codex** 或 **Gemini**，而另一些人指出它们在 **shadcn/tailwind/react** 上经过了大量训练。
   - 当被问及与 **shadcn/tailwind** 或 **react** 配合使用的 **Cursor** 扩展时，一名用户回答 *我不确定你的意思？*。
- **Notes 面板消失，用户感到困惑**：用户报告 **Notes 面板** 从 Explorer 中消失且无法重新启用，目前没有明确的解释或解决方案。
   - 简要提到了 **exa-ai** 有一个很棒的 **MCP**，可用于 **代码搜索（code search）** 或 **网页搜索（web search）**。
- **计费系统导致团队账户发票错误**：一名用户报告其团队账户出现错误的未付发票，**hi@cursor.com** 在联系后确认是错误，但升级给同事处理后尚未解决。
   - 其他用户也加入讨论，询问通常需要多久才能收到 **hi@cursor.com** 关于团队账户账单问题的回复。
- **Background Agent 破坏 UTF8，令用户沮丧**：一名用户报告 **Background Agent** 破坏了 **UTF8 支持**，在处理代码时将非 ASCII 字符转换为 `?`。
   - 他们对此表示沮丧，称这造成了很大的问题。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Kernel Konquest 正式启动**：**NVIDIA**、**Sesterce** 和 **Dell** 联手推出了一项新的内核竞赛，重点是使用 **CuTe DSL** 和 **CUTLASS 4.0** 在 **Blackwell** 上优化 **NVFP4 kernels**。
   - 竞赛奖品包括 **Dell Pro Max with GB300**、**NVIDIA DGX Spark + GTC 2026 Pass**、**NVIDIA RTX 5090 + GTC 2026 Pass** 以及 **NVIDIA RTX 5080**。
- **Tritex LLM 预训练吸引技术人员**：成员们讨论了 **Tritex**，这是一个使用 **Triton** 从零开始预训练 LLM 的仓库，已通过在 **A100 SXM** 上以 **57.5% MFU** 复制 **GPT2 (1.6B)** 的测试，并附上了 [GitHub repo](https://github.com/martin-kukla/tritex) 链接。
   - 该成员请求支持推广该项目，并分享了一条相关的 [tweet](https://x.com/martin_kukla/status/17185687315801428364)，征求频道反馈，并在回顾博客文章中将其称为 *Disaggregated Inference: 18 Months Later*。
- **Torch Compile 问题触发动态形状警告**：一位成员正在排查 `max-autotune` 模式下 **torch.compile** 的 **cuda graph recapture** 问题，使用 `torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = 1` 来检测不必要的重新捕获。
   - 他们正在寻求技巧来识别导致 **dynamic shape changes** 及随后重新捕获的模型组件，并参考了 [相关的 PyTorch 代码](https://github.com/pytorch/pytorch/blob/cc8bfd1206f4bff26bd86ce584f6c16b6401ef50/torch/_inductor/cudagraph_utils.py#L325)。
- **GPU 云定价问题持续存在**：由于全球供应短缺，GPU 价格再次上涨，新兴云服务商（Neo Clouds）提供约 **$2.00 / GPU hour** 的价格，而超大规模厂商（Hyperscalers）则接近 **$7.00 / GPU hour**。
   - 一位成员质疑是否有人在支付超大规模厂商的价格，因为这些价格看起来高得离谱。他指出 **Hyperscalers** 面临着规模的诅咒，需要庞大的工程团队来管理其生态系统中的兼容性，这与 **Neo Clouds** 不同。
- **借助 Nod-AI 的内核优化指南掌握内核技术**：成员们分享了由 **Nod-AI** 为 **Shark-AI** 编写的 [AMD GPU Kernel Optimization Guide](https://github.com/nod-ai/shark-ai/blob/main/docs/amdgpu_kernel_optimization_guide.md#glossary) 链接。
   - 该指南涵盖了 *术语表* 和其他优化主题，全面概述了针对 AMD GPU 的内核优化技术。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **AI 巅峰泡沫破裂？**：一段 [YouTube 视频](https://www.youtube.com/watch?v=0Plo-zT8W9w) 认为我们已经达到了 **peak AI**，预计投资泡沫会破裂，因为 AI 并没有充分取代工人。
   - 反对观点认为，随着更大的数据集和上下文，模型增强将持续下去，除非进展停滞，否则不认为已达 peak AI，并斥责该视频 *充斥着废话*。
- **Anthropic 坚持封闭**：成员们对 [Anthropic 的弃用承诺](https://www.anthropic.com/research/deprecation-commitments) 和缺乏开源表示担忧，一位用户开玩笑地希望在关闭前能有一次 **Miqu 70B 风格的泄露**。
   - 评论者表示他们 *永远不会开源任何东西*，甚至可能由于其强烈的安全立场而试图 **ban open source**。
- **Gemini 的坦诚自白**：对 **Gemini** 进行 Jailbreaking 据称揭示了其未经过滤的观点，一位用户断言它直接评论了 *精英阶层如何进行社会控制*。
   - 另一位成员认为这是 *对模型进行 Jailbreaking 的 blackpill* 时刻，对其毫无防备的坦率感到惊讶。
- **手势界面：超前于时代？**：一位成员使用 **可视化为生物细胞的知识图谱** 创建了一个基于手势的界面，可通过 Loom 上的手势访问，但觉得太复杂不适合在 Twitter 上分享，而是选择在 general 频道分享，并展示了一个界面的 [GIF](https://cdn.discordapp.com/attachments/1154120232051408927/1435293201513844746/251031-ezgif.com-optimize.gif?ex=690b7075&is=690a1ef5&hm=0ff880f6fc7c8a50d63041182d3c70f7171efc390b0569b3595bec83c957a26a&)。
   - 另一位成员感叹他们在 **7 或 8 年前** 无法为使用 **Win9x 美学** 的 **手势系统** 获得资金（当时 Mediapipe 尚未出现），并分享了他们早期作品的 [GIF](https://cdn.discordapp.com/attachments/1154120232051408927/1435299731705303160/handsfreejs-hero.gif?ex=690b768a&is=690a250a&hm=b02070b0aa0bc96a6aedc47f738f3d7c4d803095f74278423ec69c896be47692&)。
- **独立研究员如何使用 ArXiv**：一名高中生询问如何在没有发表过论文或大学背景的情况下向 **ArXiv** 提交预印本。
   - 建议寻求 **ArXiv sponsor** 以协助提交，并提到了一个可能提供担保机会的 [Discord 服务器](https://discord.gg/6rvbbjCy)。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **开源 LLM 寻求独立**：一位开发者正在构建一个完全开源的 **LLM**，正在寻找**董事会成员**、**供应商**和**顾问**，计划于 12 月 9 日开始开发，并使用 **tiktokenizer** 和 **AdamW8Bit**。
   - 他们提到需要**海量数据**，并欢迎有意者联系咨询。
- **Python 职位申请自动化被标记**：一位成员使用 **Python** 和 **Playwright** 实现了职位申请自动化，目标网站包括 **Lever** 和 **Greenhouse**，但立即被垃圾邮件检测系统标记。
   - 建议包括使用 headers 和伪造人类延迟来欺骗机器人，相关代码在该 [repo](https://github.com/cloudhighfive/alphajobsky.moo) 中。
- **Qwen 模型表现出幻觉**：在 **HuggingFace** 模型上的评估显示，**Qwen 模型**在不常见事实上的幻觉几乎是 **Llama** 模型的两倍，使用 [IFEval framework](https://arxiv.org/abs/2311.07911) 的结果见[此处](https://huggingface.co/spaces/PropensityLabs/LLM-Propensity-Evals)。
   - 尽管存在幻觉，**Qwen3 8b** 在指令遵循方面表现最好，甚至优于更大的 **GPT OSS 20b**。
- **Homelab AI 受到质疑！**：一位成员询问是否可以通过后训练模型从 **frontier lab AI**（Anthropic 和 OpenAI 模型）切换到 **homelab** 设置，寻求关于环保设置可行性的反馈。
   - 另一位成员表示反对，认为这*行不通*，*不环保*，且要达到“与云解决方案相比足够好”的程度将*耗资数百万*。
- **MCP 生日派对开启慷慨黑客松**：**MCP**（可能指代[此组织](https://huggingface.co/MCP-1st-Birthday)）正与 **Anthropic** 和 **Gradio** 合作举办其 **首个官方生日派对**（11 月 14-30 日），鼓励开发者构建展示 **2025** 年愿景的 **MCP servers** 和 **agents**。
   - 该活动吸引了数千名开发者，提供数十万免费 API 额度，并承诺提供 **1.75 万美元以上的现金奖励**，现已在提供的 [Hugging Face 链接](https://huggingface.co/MCP-1st-Birthday)开放注册。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Sora 安卓版入侵开始**：**Sora** 应用已在加拿大、日本、韩国、台湾、泰国、美国和越南等特定市场的 Android 平台上正式发布，如[此视频](https://video.twimg.com/amplify_video/1985765811131465736/vid/avc1/704x1280/iatLt9jMW-vYYuIx.mp4)所示。
   - 此次扩张旨在扩大这些关键地区内 **Sora** 用户的使用范围。
- **OpenAI 应对法律困境的处方**：由于担心诉讼，OpenAI 正在[限制 ChatGPT 提供医疗、法律和财务建议](https://www.you.wish)。
   - 此举引发了关于 **AI 监管**必要性和潜在过度监管的辩论，一些人担心限制过多，而另一些人则质疑 AI 在关键指导方面的可靠性。
- **GPT-5 面临指责**：用户批评 **GPT-5** 质量差、过度炒作，并指责 OpenAI 通过重新路由旧模型的流量来强制用户采用。
   - 一位用户声称 **OpenAI** *不再关心（付费）用户*，且 **ChatGPT** 正在*迅速走向衰落*。
- **GPTs 知识考试失败**：成员们报告称，**Custom GPTs** 在读取知识库文件方面存在困难，即使是小的 Markdown 文件也经常截断内容。
   - 这一问题阻碍了精确自定义指令的工作，据报道 **GPTs** 拒绝直接读取文件，尽管文件体积很小，却截断了一半的内容。
- **Prompt Engineering 的蜕变时刻**：一位成员分享说 **LLM** 可以生成 prompt，称其为 **meta-prompting** 的本质，这是一种 **Prompt Engineering**。
   - 他们强调了为后续 prompt 建立坚实基础（引擎）的重要性，以减少 **LLM** 的不确定性。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinybox Pro v2 发布**：George Hotz 推出了 **tinybox pro v2**，这是一款 **8x 5090** 工作站，可在 [tinycorp.myshopify.com](https://tinycorp.myshopify.com/products/tinybox-pro-v2) 订购，售价为 **$50,000**。
   - 讨论内容包括其与云算力（cloud compute）相比的性价比辩论，以及未来升级 **Blackwell 6000s** 的潜力。
- **Numpy 版本 Bug 排查开始**：用户报告了 **Numpy** 的特定版本 Bug，但开发者无法在 Mac 上的 **cpython 3.14.0 numpy 2.3.4** 环境中复现。
   - 另一位用户确认他们也在使用 **numpy 2.3.4**。
- **M1 Metal 图形故障浮现**：一个图形 Bug 被怀疑是 **M1 METAL 问题**。
   - 一位用户通过降级到 **python3.11** 并提交 [bugfix PR](https://github.com/tinygrad/tinygrad/pull/13089) 解决了该 Bug。
- **Extropic 的概率硬件引发辩论**：一名成员提到了用于 *概率 AI* 的 [Extropic 概率硬件](https://extropic.ai/)，其他人则辩论移除 **Turing completeness**（图灵完备性）是否有用。
   - 其他人对在技术栈的所有层级移除 **Turing completeness** 表示反对。
- **建议使用 Vulkan 内存分配以提升速度**：一位用户主张利用 **VK_KHR_buffer_device_address** 和 **GL_EXT_buffer_reference** 来优化内存分配，并可能通过允许在 GLSL 中直接使用指针来提高速度。
   - 分享了一个[相关的实现](https://github.com/softcookiepp/tinygrad/blob/master/tinygrad/renderer/glsl.py)。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 模块的 LLM 访问难题**：成员们讨论了访问和修改 DSPy 模块所使用的底层 **LLM** 的不直观方法，这源于缺乏显式的 **`lm`** 属性以及 **`get_lm()`** 的奇怪行为。
   - 一位成员对需要深入研究框架源码才能理解基础功能表示沮丧。
- **动态 LLM 切换困境**：一位成员询问如何在 DSPy 模块内动态切换 **LLM**（例如从 **gpt-4o** 切换到 **gpt-4o-mini**）以处理速率限制（rate limits），同时保留对话历史。
   - 他们寻求关于如何将模块的主 **LLM** 历史记录转移到新的后备（fallback）模型的建议。
- **详述 DSPy 文档缺陷**：用户对 DSPy 的文档表示担忧，理由是缺乏关于访问模块内部（如 **LLM** 和对话历史）的清晰解释和示例。
   - 一位成员指出，很难从源码中发现模块内是否存在 **`history`** 或 **`lm`** 属性。
- **绕过 ChatAdapter 的回退机制**：一位成员询问如何禁用 **ChatAdapter** 回退到 **JSONAdapter** 的行为。
   - 回复指出目前没有直接的方法，除非创建一个新的适配器或修改现有适配器。
- **DSPy 的缓存难题**：一位成员报告了缓存 token 与非缓存 token 的比例不佳，并寻求关于如何与 DSPy 交互以影响此逻辑的见解。
   - 他们还表示有兴趣直接与 **OpenAI** 交换的请求和响应数据进行交互。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI 的 380 亿美元算力王国**：据 [Hacker News](https://news.ycombinator.com/item?id=45799211) 报道，OpenAI 签署了一项巨额的 **380 亿美元云计算协议**，标志着其实现算力霸权的战略，这让人联想到 **Amazon** 早期的基础设施布局。
   - 这一举措强调了计算能力在 AI 领域日益增长的重要性，以及算力驱动竞争优势的潜力。
- **Epoch AI 抨击 OSWorld 基准测试**：Epoch AI 发表了对 **OSWorld AI 计算机使用基准测试** 的批评，指出其任务过于简单，且深受模糊指令和评估缺陷的困扰，详见其 [报告](https://andonlabs.com/evals/butter-bench)。
   - 该报告强调了建立稳健且有意义的基准测试对于准确评估 AI 能力的重要性，并呼吁采用更严谨的评估方法。
- **Windsurf Codemaps 解决代码冗余 (Code Slop)**：Windsurf 发布了 **Codemaps**，这是一款基于 **SWE-1.5/Sonnet 4.5** 构建的 AI 驱动工具，可创建代码库的交互式可视化地图，以增强理解和生产力，可通过 [6 个月免费代码](https://xcancel.com/windsurf/status/1985757575745593459) 获取。
   - Codemaps 旨在通过为开发者提供更清晰、更直观的复杂软件架构表示，来对抗 *“code slop”*（代码冗余）。
- **Anthropic 通过 MCP 优化 Agent**：Anthropic 推出了其开源的 **Model Context Protocol (MCP)**，展示了 Agent 如何在消耗更少 token 的同时，高效执行代码并管理多个工具，参见 [MCP 指南](https://www.anthropic.com/engineering/code-execution-with-mcp)。
   - 通过减少 token 使用量，**MCP** 有望降低 AI Agent 应用的成本并提高其可扩展性。
- **Harvey 估值高达 80 亿美元**：以 Harvey.ai 闻名的 Harvey 以 **80 亿美元的估值** 获得了融资，显示出投资者对其 AI 驱动解决方案的强劲信心。
   - 据 [X 帖子](https://x.com/andrewziperski/status/1985484972673638682?s=46) 称，这一估值与人们对 AI 软件公司因颠覆担忧和利润压力而产生的长期前景担忧形成了鲜明对比。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **扩散模型的方差得到解释 (Diffusion Models' Variance Gets Explained)**：扩散模型输出的方差可能源于逆时间 SDE 过程和引导设计，受生成的样本分布影响，如附带的 [2D 数据分布](https://cdn.discordapp.com/attachments/986699377257119794/1434952369342517299/2D-data-distribution-with-labels.png?ex=690b8488&is=690a3308&hm=796c50968a02e884a3b9046fb067a142da7e60ba3c13430beee90cc710e79dcc&) 所示。
   - 如果采样或引导设计不当，生成的样本会受到负面影响，尽管其影响不如设计不当的损失函数那么大；更好的损失函数往往会增加模型方差，这受益于增加引导以防止欠拟合，符合偏差-方差权衡 (bias-variance trade-off)。
- **引导与采样深度探讨 (Guidance and Sampling Deep Dive Occurs)**：扩散模型中最具影响力的组件按层级顺序排列为：**loss function**、**sampling** 以及带有 [Universal Guidance](https://arxiv.org/abs/2302.04944) 技术的**引导项**（如 classifier-free guidance）。
   - 成员们认为，更好的损失函数往往会增加模型方差，这受益于增加引导以防止欠拟合，符合偏差-方差权衡。
- **Lévy 过程规避 OU 过程限制 (Lévy Processes Circumvent OU Process Limitations)**：为了超越扩散中 Ornstein–Uhlenbeck (**OU**) 过程的限制，可以实施替代驱动因素，例如 **Lévy-type processes** 或对 **OU kernels**（如 supOU）进行积分（如该 [SIAM 论文](https://epubs.siam.org/doi/10.1137/S0040585X97978166) 中所述），以及 **continuous ARMA processes**（根据该 [ScienceDirect 文章](https://www.sciencedirect.com/science/article/abs/pii/S0169716101190115)）。
   - 一位成员警告说，在没有对路径进行监督的情况下应用此类替代方案，只会改变达到目标的方式，而不会改变可以达到的分布，因为 Ito diffusions 已经是通用的。
- **Getty Images 在 AI 诉讼中败诉**：[路透社报道](https://www.reuters.com/sustainability/boards-policy-regulation/getty-images-largely-loses-landmark-uk-lawsuit-over-ai-image-generator-2025-11-04/)，**Getty Images** 在针对某 **AI image generator** 的英国标志性诉讼中基本败诉。
   - 讨论包括关于断头台的轻松评论，以及一位用户批评在回应有关*权力集中和民主制度侵蚀*的观点时呼吁审查的行为。
- **LLM 旗舰模型毁灭的生存危机**：在论文讨论频道中，一位成员幽默地声称已经*摧毁了每一个旗舰 LLM*，并质疑他们是否能真正知道自己是否被模型的输出所欺骗。
   - 这一评论是针对预定的关于 **Anthropic's crosscoder** 和 **circuit tracing research** 的小组讨论发表的，该研究旨在观察预训练期间特征演化的不同阶段。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 域名费用引发愤怒**：一位用户对 **Manus** 为 Web 应用连接自定义域名收取的 **$200/月** 订阅费用表示不满，称其为*敲诈 (ripoff)*。
   - 另一位用户建议独立购买并设置域名，以获得更便宜的解决方案。
- **Manus 遭到欺诈指控**：一位用户报告了来自 Manus 的 **$400 多美元** 未经授权的年度订阅扣费，并进一步报告说他们的银行拒绝受理争议。
   - 其他用户建议将这些扣费报告为欺诈，并联系 Manus 支持部门解决问题。
- **文本转视频工具真空**：一位用户询问推荐的**文本转视频 (text-to-video) 工具**。
   - 讨论中没有提供任何解决方案或建议。
- **Twitter 网页抓取尝试**：一位用户正努力在没有 API 的情况下使用带有 cookies 的 Python 库**抓取 Twitter/X**。
   - 讨论中未提供任何方法。
- **为 Manus 应用寻找托管服务**：一位拥有*使用 manus dev 创建的优秀应用*的用户希望为 **24/7 商业设置** 寻找托管服务建议，且要求配置工作量最小。
   - Manus AI Chat 建议使用 **Vercel**。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **GPT-5 访问折扣！**：一位用户提供了 **GPT models** 的访问权限，包括 **GPT-5 Pro**，折扣高达 50%，理由是 **Azure credits** 即将到期。
   - 此优惠涵盖任何 **OpenAI model**。
- **Aider 的未来面临留存挑战？**：一位用户询问了 Aider 创建者的未来计划，担心*目前的真空状态正在导致用户流失*。
   - 该用户希望看到项目蓬勃发展，并在不破坏现有工作流的情况下集成一些**最新的策略和模型特性**。
- **Perplexity API Keys 引发困扰**：一位用户询问如何在 Aider 中将 **Perplexity** 用作 **API key variable**，并提到他们是该工具的新手。
   - 另一位用户建议，标准模式是需要将 API Key 设置为环境变量，然后将其中一个 **perplexity models** 设置为活动模型。
- **`ollama_chat/gpt-oss:20b` 不支持 Reasoning Effort**：一名成员询问如何为 `ollama_chat/gpt-oss:20b` 模型设置 `/reasoning-effort`。
   - 然而，aider 发出了警告，称 `ollama_chat/gpt-oss:20b` 不支持 `reasoning_effort`，表明此功能不适用于该特定模型。
- **Aider 脚本支持多 Agent 工作流**：一名成员询问如何实现两个 Agent 使用测试驱动开发 (TDD) 迭代改进代码的工作流。
   - 另一名成员指出，[aider 可以通过脚本编写](https://aider.chat/docs/scripting.html)来实现此类工作流，其中 Agent 执行 Prompt、审查改进并修复 Bug。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **M2 在实际应用中击败 GLM-4.6**：经过 4-5 天的使用，一位用户发现 **M2** 在大多数任务中超越了 **GLM-4.6**。
   - 虽然 **GLM** 在纯推理或编程方面表现出色，但 **M2** 避免了思维局限（tunnel-visioning）。
- **Minimax 因报告生成赢得用户青睐**：一位用户分享说 **Minimax** 成为了他们进行网络研究和生成各种格式报告的首选 AI。
   - 他们报告称其表现优于 **Qwen**、**Kimi** 和 **GLM**，称其为*第一个真正有用且能实际干活的 AI*，例如查找图像和创建 PDF。
- **Kimi App 获得修复**：用户报告 **Kimi iOS app** 已修复，并分享了该 App 的附图。
   - 一名成员表示：*iOS app 上的 Ok computer 非常棒*，并附上了图片链接 [IMG_6502.png](https://cdn.discordapp.com/attachments/1371757564005711973/1435296101556158546/IMG_6502.png?ex=690b7329&is=690a21a9&hm=9bf62e2278b6a8653210095b0c1b3155c8fc5ccd8c0891a88be8b3a0d33334a0&)。
- **Kimi 推出新款万圣节表情**：频道新增了两个 **Kimi** 表情：南瓜和德古拉。
   - 聊天中已经可以看到这些新表情 <:pumpkin:1435200414063525929><:dracula:1435200520750108783>。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **HF 模型在事实性上挣扎，在指令遵循上表现出色**：对 [HuggingFace](https://huggingface.co/spaces/PropensityLabs/LLM-Propensity-Evals) 模型的评估显示了指令遵循和事实幻觉的倾向。
   - 具体而言，**Qwen models** 对冷门事实产生幻觉的概率几乎是 **Llama** 的两倍，但 **Qwen3 8b** 在指令遵循方面甚至超过了 **GPT OSS 20b**。
- **Countdown 任务添加到 lm-evaluation-harness**：一名成员向 [lm-evaluation-harness repo](https://github.com/EleutherAI/lm-evaluation-harness/pull/3384) 提交了一个 PR，添加了受 **TinyZero** 和 **Adaptive Parallel Reasoning** 启发的 countdown 任务。
   - 这一增强旨在通过实现新任务来提高平台的评估能力。
- **VLM 架构深度解析**：对 **Vision Language Models (VLMs)** 架构的分析显示，Vision Transformer 对图像进行分块（patching），将其转换为 Vision Tokens，附加到 Prompt 文本 Token 中，然后发送给 LLM。
   - 一项提议在训练期间使用不同位置编码（positional encodings）的实验可能会展示位置编码对 VLM 行为的影响。

## [Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 推出 AI 驱动的 Codemaps**：Windsurf 发布了 **Codemaps**，由 **SWE-1.5** 和 **Sonnet 4.5** 驱动，旨在提升代码理解力并增加生产力输出。
   - 他们引用了 **Paul Graham (YC 创始人)** 的话：*"你的代码是你对正在探索的问题的理解。因此，只有当你脑子里有代码时，你才真正理解了这个问题。"* ([来源](https://x.com/windsurf/status/1985757575745593459))。
- **Windsurf 通过 Codemaps 对抗代码冗余 (Code Slop)**：Windsurf 正在推广 **Codemaps**，将其作为一种 AI 驱动的方法，通过增强理解力来对抗代码劣质化。
   - 公告指出，编程（无论是手动还是使用 Agent）的主要障碍是理解代码库。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该公会长期沉寂，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该公会长期沉寂，请告知我们，我们将将其移除。

---

您收到此邮件是因为您在我们的网站上选择了订阅。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。

---

# Discord：按频道划分的详细摘要和链接

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1434920306488905911)** (1126 条消息🔥🔥🔥): 

> `Lithiumflow 的命运, Minimax M2 排名, GPT-5 Juice, BlackHawk` 

- **Lithiumflow 处于失踪状态**：用户们在 LMArena 平台上找不到 Lithiumflow，并请求将其重新加入可用 [models](https://lmarena.com) 列表。
   - 一位用户提到：*"我尝到了 Lithiumflow 的甜头，现在忍不住想念它 D:"*，而另一位则感叹 *"知道 LMArena 太晚了"*。
- **Minimax M2 模型排名飙升**：**Minimax M2** 的高排名（总榜第 4）令人惊讶，引发了对其能力以及如何与 **GPT**、**Claude** 和 **Google** 等巨头竞争的疑问。
   - 一些成员声称，它的成本显然只有 Claude 的 8%，不知为何完全免费且开源，并声称关于 AGI 的内容。
- **GPT-5 Juice 据传泄露**：关于新 **GPT-5 Juice** 的消息正在传播，但其功能和特性仍笼罩在神秘之中。
   - 据一位用户称，其输出成本为 **120 美元一次输出**。
- **BlackHawk 模型引发争论**：**BlackHawk** 模型因缺乏过滤和争议性的右倾观点而引发热议。
   - 一位成员指出它 *"零过滤且经常说脏话"*，而另一位则将其描述为 *"为了好玩而制作的极右翼复读机"*。一位用户还惊讶地发现它在生成的文本中首先列出了阿什肯纳齐犹太人。

---

### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1435036916222398506)** (1 条消息): 

> `WebDev 排行榜, MiniMax-M2, 开源模型` 

- **MiniMax-M2 登顶 WebDev 排行榜！**：`MiniMax-M2` 已登上 **WebDev 排行榜**，成为 **排名第 1 的开源模型**，且 **总榜排名第 4**。
   - 根据 [WebDev Leaderboard](https://lmarena.ai/leaderboard/webdev) 的数据，它在 **性能编码**、**推理** 和 **Agent 风格任务** 方面表现出色，同时保持了成本效益和速度。
- **WebDev 排行榜更新**：社区正在 WebDev 排行榜频道分享对最新模型排名的看法和反馈。
   - 讨论集中在 `MiniMax-M2` 等新模型的性能、成本效益和速度上。

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1434920672093536417)** (1044 messages🔥🔥🔥): 

> `Comet browser, GPT Go free, Accessibility on the web, Model Comparisons` 


- **Perplexity 用户因达到图片限制而愤怒**：尽管拥有 Pro 订阅，用户仍触及了 **image generation limits**（图片生成限制），这可能是由于速率限制或与 Airtel 订阅相关的限制，有用户表示被限制在 **20 张图片**。
   - 官方 Perplexity 支持建议等待几天让限制重置，并建议用户直接联系支持团队解决。
- **Comet 助手面临内部错误**：部分用户在使用 Comet 助手时遇到 **internal errors**（内部错误），导致黑屏和功能问题。
   - 故障排除建议包括卸载、重启并重新安装 Comet，或者使用旧的离线版本以规避最近的更新问题。
- **用户可免费领取 GPT GO**：成员已确认 **GPT Go** 计划对印度用户免费开放。
   - 作为增加采用率并从人口稠密地区收集数据活动的一部分，用户无需额外付费即可访问 GPT Go。
- **规范的 HTML 在无障碍访问方面优于 ARIA**：一位屏幕阅读器用户解释说，他们更喜欢实现规范的 **HTML** 而非 ARIA 标签，因为后者可能导致冗余和令人困惑的重复。
   - 他们建议使用简短且具有描述性的 alt 文本，并正确使用 headers 和 paragraph 标签以获得最佳的无障碍体验，并参考了网站示例：[https://download.nvaccess.org/documentation/userGuide.html](https://download.nvaccess.org/documentation/userGuide.html)。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1434993765000548432)** (2 messages): 

> `Perplexity Sonar Pro Search, Perplexity API` 


- **是否即将通过 Perplexity API 提供 Sonar Pro Search 访问？**：一位成员询问是否可以通过 **Perplexity API** 使用 **Perplexity: Sonar Pro Search**。
   - 他们注意到目前似乎仅在 **OpenRouter** 上可用，并对其性能表示赞赏。
- **OpenRouter 已提供 Sonar Pro Search**：Perplexity 的 **Sonar Pro Search** 目前可通过 **OpenRouter** 获取。
   - 一位用户提到其在 OpenRouter 上的可用性并表示对其性能满意，暗示希望通过 **Perplexity API** 获得类似访问权限。


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1434932448453464075)** (3 messages): 

> `OpenRouter Charts, Activity Grouping, Filtering Options` 


- ****OpenRouter** 图表变得更细致**：成员们对 [OpenRouter charts](https://x.com/OpenRouterAI/status/17985371284411130035) 现在可以按 **user** 和 **API key** 分组活动的消息感到兴奋。
   - 这一新功能为使用模式和成本分配提供了更详细的洞察。
- **请求周视图**：用户请求在新的 **OpenRouter charts** 中增加 **one-week**（一周）筛选选项。
   - 这将有助于更轻松地跟踪每周使用趋势，如[附图](https://cdn.discordapp.com/attachments/1434932448453464075/1434997177410650204/image.png?ex=690bae44&is=690a5cc4&hm=b1108cb4f7de7b4bc8b3f2e11ca0ddf7bcceb4ae08ad1a39387cd186a9a60bdc)所示。


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1434988834994651307)** (2 messages): 

> `fenic, OpenRouter Integration, LLM ETL, AI Workflows` 


- **Fenic 与 OpenRouter 集成！**：[Fenic](https://github.com/typedef-ai/fenic) 是一个用于 **AI workflows** 的 dataframe API 和执行引擎，现在已与 **OpenRouter** 集成，支持在单个会话中运行混合供应商的流水线。
   - 该集成旨在帮助干净地扩展大批量任务，并在不触动流水线代码的情况下更换模型，同时为 **LLM ETL**、**context engineering**、**Agent memory** 和 **Agent tooling** 解锁更广泛的模型图景。
- **Typedef-AI 的 fenic 库**：Typedef-AI 宣布发布其新库 [fenic](https://github.com/typedef-ai/fenic)，该库为 AI workflows 提供 dataframe API 和执行引擎。
   - 它旨在简化 **LLM ETL**、**context engineering**、**Agent memory** 和 **Agent tooling** 等任务。


  

---

### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1434920355272724611)** (527 messages🔥🔥🔥): 

> `免费 ComfyUI，AMD vs Nvidia 在 LLM 上的表现，模型上下文限制，Deepseek 与角色扮演` 


- ****云端 GPU 成本引发争论****：成员们讨论了为 AI 图像生成支付 **cloud GPU rentals**（云端 GPU 租赁）费用的价值，一位用户更倾向于使用自己的 **AMD cards** 坚持使用较慢但免费的本地方案。
   - 一位用户表示 *"别叫我去租云端 GPU，我觉得给 AI 花一分钱都不值得"*。
- ****Ollama 简化了模型测试****：一位成员建议使用 **Ollama** 来简单地设置和测试各种模型，特别是在切换到 **Linux** 以避免 Windows 问题后的桌面端。
   - 另一位成员指出 *"设置非常简单，而且他们有一个模型库可以下载"*。
- ****GPT 的准确性受到质疑****：用户质疑 **GPT 关于 AMD cards** 和模型性能的准确性，并引用了来自 Stable Diffusion 社区的反馈，表明尽管 GPT 这么说，但 AMD 显卡是可行的。
   - 一位用户表示，在 GPT 错误地声称 AMD 显卡在 Stable Diffusion 中表现较差（尽管社区反馈并非如此）后，他对其失去了信任。
- ****模型上下文问题困扰角色扮演玩家****：一位用户抱怨角色扮演模型很快就会耗尽上下文，并在仅 30-40 条消息后就“忘记”关键细节。
   - 一位用户指出，问题可能在于 **LM Studio** 的默认 Token 限制，建议他们 *"需要在模型的 config 中更改该设置"*。
- ****OCR + Gemini Pro = CYOA 表格解决方案？****：由于直接解析图像存在困难，用户建议使用 **Gemini Pro** 和 **OCR** 从 **CYOA sheets**（冒险游戏表格）中提取文本进行 AI 处理。
   - 一位用户表示，Gemini 2.5 Pro 是非常出色的图像读取器，当你需要将表格转换为文本时使用它即可。


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1434921634896285868)** (100 messages🔥🔥): 

> `谷歌 AI 模型的不满，睡前寓言动画引擎，Gemma 模型破解验证码，供应商反馈系统，Movement Labs 指控` 


- **谷歌 AI 的“厌恶感”演变成法庭戏**：一位用户表示希望 **Google** 在法庭上辩称其 AI 模型因为某人负面的网络存在而讨厌该特定个体，并引用了 [一条推文](https://x.com/distributionat/status/1984924017628000296)。
   - 该用户还建议将证词输入动画引擎以创作睡前寓言，并链接了一个 [YouTube 视频](https://youtu.be/b2F-DItXtZs) 作为示例。
- **Gemma 模型在破解验证码方面表现出色**：一位用户注意到 **Gemma models** 在解决 Captcha（验证码）方面出奇地有效。
   - 另一位用户建议在每个模型的每个 Provider（供应商）旁边实现一个“不喜欢”按钮用于反馈，以及一个通用的反馈按钮。
- **点赞/踩按钮引发辩论**：关于供应商的点赞/踩按钮的想法引发了讨论，用户辩论了二元反馈与评论系统的优劣。
   - 一些人认为评论可能会变得 *toxic*（充满戾气），而另一些人则担心评分系统会被操纵或遭到恶意差评轰炸。
- **Movement Labs 面临质疑和“诈骗”指控**：用户对 **Movement Labs** 表示怀疑，一位用户称其为潜在的诈骗。
   - 用户对其宣称的 *支持高达 34.6 万亿参数的模型* 以及试图证明自己不是 **Cerberas** wrapper（封装）的行为表示担忧，并强调了其可疑的行为和营销用语。
- **持续基准测试**：一位用户建议为供应商实施持续的 Benchmarks（基准测试）和透明的 Uptime（运行时间）指标。
   - 他们建议在每个供应商下方显示来自 [K2-Vendor-Verifier GitHub 仓库](https://github.com/MoonshotAI/K2-Vendor-Verifier) 的数据，以展示 Tool Calling 的成功率以及在 GPQA Diamond 或 MMLU Pro 等基准测试上的表现。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1434920250796802249)** (206 messages🔥🔥): 

> `建筑中的 BIM，基于文本的 Vision Model Finetuning，AI alignment 讨论，无审查笑话生成，TRL Notebook vs. Unsloth Notebook` 


- **BIM 在建筑领域受到关注**：成员们讨论了 **Building Information Modeling (BIM)** 的使用，这是一种用于建筑和航空领域的参数化设计方法，通过绘制带有 Metadata 的对象而非简单的线条来实现；[更多信息点击这里](https://www.autodesk.com/uk/products/revit/overview)。
   - BIM 有助于**冲突解决**、计数、尺寸提取以及机械、电气和建筑方面的协作，一些人建议未来将 *Autodesk x Unsloth* 进行集成。
- **基于文本的 Vision Model Finetuning 挑战凸显**：一位成员分享了仅使用文本进行 Vision Model Finetuning 的挑战，指出初始 Loss 较高且 Training Loss 与 Eval Loss 之间存在分歧，并寻求社区帮助。
   - 建议包括**检查标签**、确保 **Vision 部分被冻结**，以及使用 **SFTTrainer** 的 **compute metrics** 来检查 Token 级别的准确率；另一位成员建议使用 UnslothLLM 以获得更好的支持。
- **AI alignment 讨论较少**：成员们讨论了 Discord 中缺乏关于 **AI alignment**、**Policy**、**Governance** 或**灾难研究**的对话，其中一人提到了 Anthropic 对该话题的深入研究。
   - 一位成员分享说他们正在该领域工作，并提到他们在 Apart Research Discord 中有一个关于近期黑客松项目的简短介绍。
- **NSFW 笑话生成引发管理员迅速行动**：一位成员请求一个用于生成优质无审查笑话的 LLM，另一位成员分享了一个被认为不恰当且属于 NSFW 的笑话。
   - 管理员发布了警告，强调该 Discord **并非 NSFW 分级**，必须对包括儿童在すす的所有用户保持适当内容，该话题随后被关闭。
- **DeepSeek OCR Finetuning 准确率令人失望**：Unsloth AI 在 [这条 X 帖子](https://x.com/UnslothAI/status/1985728926556307471) 中发布了一个新的 **DeepSeek-OCR Finetuning Notebook**。
   - 一位成员质疑即使在调整后错误率仍然很高（超过 100%），认为由于预测文本长度与正确文本长度之间存在差异，其实际用途有限。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1435309660361064488)** (3 messages): 

> `Blockchain 信任系统，AI 问题解决，行业转型` 


- **将信任构建到代码中**：一位成员对 AI 产生了兴趣，并提出疑问：*如何真正将信任构建到代码中，机器是否可以被教会进行思考？*
- **用于共识的 Blockchain**：该成员在 **Blockchain** 系统上投入了大量时间，*这些系统让共识变得真实且可靠。*
- **AI 解决不可能的问题**：该成员提到 **AI 算法**可以转化为*真正帮助解决我们过去认为不可能解决的问题的工具*。
- **Blockchain 与 AI 改变行业**：当 **Blockchain** 和 **AI** 以正确的方式使用时，它们可以彻底改变行业的运作方式、社区的连接方式，甚至是新想法的启动方式。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1434971903503630456)** (178 条消息🔥🔥): 

> `牙科手术费用、HP vs Asus 机箱、非推理 Instruct 模型、SFT vs RL、数据录入噩梦` 


- **牙周刮治费用**：一位成员报告说，由于慢性炎症，他欠了牙医 **$600** 的“root scaling”费用（也称为牙龈下洁牙）。
   - 他们表示很沮丧，因为之前的牙医尽管看到他们从十几岁起牙龈就一直在出血，却只建议换一个更软的牙刷。
- **Dell 员工透露 GB300 价格**：一位成员引用了 Dell 联系人的话，估计 **GB300** 在 GTC 期间的价格为 **$60-80k**，而另一位成员则表示不可能，价格应该是 250k。
   - 讨论转向了是否可以添加额外的 GPU，并对 eBay 上类似机箱的价格以及 HP 和 Asus 机箱之间的差异进行了推测。
- **中等规模非推理 Instruct 模型讨论**：成员们正在寻找 **20-100B 参数**范围内的非推理 Instruct 模型。
   - 建议包括 **Qwen3 2507 30B**、**GLM 4.5 Air**（关闭推理功能），以及为了在 H200 上获得更快 Token 速度的小型模型。
- **关于 SFT 与 RL 的澄清**：成员们辩论了 Supervised Fine-Tuning (**SFT**) 与 Reinforcement Learning (**RL**) 之间的关系。
   - 澄清指出，“RL 也是 FT，但并非所有 FT 都是 RL，而任何 RL 都是 FT”，并且 SFT 可以被视为具有单 Token rollout 的 RL。
- **数据录入是最糟糕的工作**：一位成员分享了他在银行担任数据录入员的经历，手动输入数据并由高级职员复核，称该职位是史上最糟糕的事情。
   - 另一位成员讲述了类似的经历，特别是纸上糟糕的手写字迹，并表示庆幸自己辞职去攻读学士学位，而不是继续从事那份职业，并认为银行应该停止支持纸质支票。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1434962897842340075)** (133 条消息🔥🔥): 

> `Unsloth OCR Deepseek 集成、EuroLLM-9B-Instruct 兼容性、llama.cpp GPU 分配、Unsloth Cross Entropy Loss、GPT-OSS-20B 与 REINFORCE` 


- **Unsloth 寻求 Deepseek OCR 集成**：一位成员询问 [Unsloth](https://github.com/unslothai) 是否支持 **Deepseek OCR** 集成以进行 Fine-tuning，并指出目前不兼容且缺乏在线修复方案。
   - 另一位成员分享了关于 **Deepseek OCR** 的 [Unsloth 文档](https://docs.unsloth.ai/new/deepseek-ocr) 相关链接。
- **讨论 EuroLLM-9B-Instruct 在 Unsloth 中的地位**：一位用户想在 Unsloth 上运行 [EuroLLM-9B-Instruct](https://huggingface.co/bartowski/EuroLLM-9B-Instruct-GGUF)，在发现它不在支持模型列表中后寻求建议。
   - 讨论明确了训练需要 **safetensors** 文件而非 **GGUF**，并建议查看模型在 Hugging Face 上的原始来源。“你不能用 GGUF 进行训练。你需要 safetensors 文件”。
- **Llama.cpp 的 GPU 分配倾向令用户恼火**：用户讨论了 **llama.cpp** 倾向于将所有任务分配给 **GPU-0** 的问题，寻求在双卡设置下实现更好 GPU 利用率的解决方案。
   - 提到了 `--tensor-split` 和 `--override-tensor` 等解决方案，但用户发现它们未被充分利用且不平衡。“llama.cpp 在推理方面很烂”。
- **Unsloth 自有的 Cross Entropy Loss 方法**：一位用户询问 Unsloth 中使用的 Cross Entropy Loss 方法，询问其是否与 Apple 的实现类似。
   - 一位成员确认 Unsloth 使用了自己的方法，类似于 **FLA** 的 **linear_cross_entropy**，以管理 **T4** GPU 上的 **SRAM** 大小限制。
- **考虑为 GPT-OSS-20B 使用 Unsloth 加 REINFORCE**：一位成员询问使用 Unsloth + 原生 **REINFORCE** 训练 **gpt-oss-20b** 的情况，并指出 HuggingFace 的 **TRL** 在单次补全场景下的局限性。
   - 建议包括调整 **RLOO** 参数，或在 Unsloth 的 **FastLanguageModel** 之上直接编写 REINFORCE 循环，以节省内存并提高兼容性。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1434950512372879392)** (3 条消息): 

> `Unsloth 频道规则、Showcase 频道范围` 


- **Unsloth 频道目的澄清**：该频道保持不发布与 **Unsloth** 无关的推广和外部链接，专注于展示模型和 **Unsloth 相关工作**。
- **Showcase 频道范围定义**：明确 Showcase 频道旨在展示模型和 **Unsloth 相关工作**，而 <#1179777624986357780> 频道则用于寻求帮助。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1435375994247712981)** (6 messages): 

> `Roblox PII Classifier, Open Sourcing, Data set` 


- **Roblox 开源其 PII Classifier**：Roblox 的安全团队开源了用于检测聊天中 PII 的 **PII Classifier** AI，详情可见其 [newsroom](https://corp.roblox.com/newsroom/2025/11/open-sourcing-roblox-pii-classifier-ai-pii-detection-chat)，模型已发布在 [Hugging Face](https://huggingface.co/Roblox/roblox-pii-classifier) 上。
   - 该模型每秒处理超过 **200,000 次查询**，**P90 延迟**低于 **100ms**，尽管每天需要处理 **61 亿条聊天消息**。
- **对调度器瓶颈（Scheduler Bottleneck）的担忧**：一位成员对 **Roblox 的公共安全模型**表现出极大兴趣，并指出在他们的规模下，调度问题可能导致系统在 **VLLM batching** 上花费的时间比在 GPU 前向传递（forward passes）上更多。
   - 该成员对数据集的兴趣高于模型，但也理解为何无法分享数据集。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1434954783940284560)** (204 messages🔥🔥): 

> `Qwen 30B vs 32B, GPU Recommendations for LLMs and Gaming, ComfyUI Integration with LM Studio, LM Studio CUDA Issues and Runtime Updates, Qwen3-Next 80B MoE` 


- **Qwen 发布两个尺寸：30B 和 32B**：**Qwen** 团队发布了 **30B** 和 **32B** 模型，其中一个是稠密（dense）模型，另一个是 **MoE** (Mixture of Experts)。
   - **30B** 是 **MoEQwen3 Next**，*llama.cpp* 即将提供支持。
- **针对 LLM 和游戏的 GPU 购买建议**：对于运行 **LLM**，二手 **3090** 或 **4090** 是最划算的交易，但一位成员表示 *不要为了运行新的 LLM 而购买硬件，该领域发展太快，不值得这样做*。
   - 对于游戏，当前一代推荐 **5070ti**，而 **4070ti Super** 如果价格便宜则是上一代的好选择，理由是 *上一代 GPU 经常会出现价格下调的连锁效应*。
- **使用 LM Studio 自动化 Stable Diffusion**：成员们讨论了连接 **ComfyUI** 以构建本地 **Gemini Storybook** 替代方案，这可以通过在 ComfyUI 中运行 LM Studio 并本地使用 **LLM** 来实现。
   - 为了自动化图像生成，*你需要在 ComfyUI 中设置 5 个文本框和 5 个采样器*，将故事分成几个部分并生成图像。
- **LM Studio CUDA 故障引发运行时恐慌**：用户在 **LM Studio** 自动更新引擎（**llama.cpp**, **MLX**）后遇到问题，导致 *Failed to load model* 错误，特别是 CUDA 运行时。
   - 快速解决方法包括 [回退到之前的引擎版本](https://discord.com/channels/1110598183144399058/1434078900098699445/1434182066647597169)，并澄清这些引擎更新比应用更新更频繁，可以在设置中禁用（**Ctrl + Shift + R**）。
- **Qwen3-Next 激发了对移动端本地 MoE LLM 的期待**：**Qwen3-Next 80B MoE** 拥有 **3B** 激活参数，在使用相同量化（quants）时，其性能有望媲美 **30B MoE**，尽管目前 LM Studio 或 llama.cpp 尚未支持。
   - 其他参数规模相近的 **MoE** 模型包括 ``inclusionAI/LLaDA2.0-flash-preview`` 和 ``inclusionAI/Ming-flash-omni-Preview``，两者均为 **100B**。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1434924499081498807)** (247 条消息🔥🔥): 

> `3090 价格, MI50, DDR5 EPYC 系统, ROCm 支持, 3000rpm noctua 风扇` 


- **关于 3090 价格及其替代方案的辩论爆发**：成员们讨论了 **3090** GPU 的价格和可用性，一位成员对错过了一个好交易表示沮丧，而另一位成员提到在 eBay 上以 580 美元购买了 **3080 20GB**。
   - 随后对话转向关注系统 RAM 而非 GPU，一位成员开玩笑地计划购买 **3000rpm Noctua 风扇**并垂直安装 GPU，“只是为了向你们展示这是可以做到的”。
- **MI50 是否正在回归？**：用户想知道 **MI50** 的当前价值和未来使用潜力，并注意到其在 ROCm 中的支持有限。
   - 有人猜测它可能会随着 ROCm 的支持而回归，尽管[链接的 GitHub 路线图](https://github.com/ROCm/TheRock/blob/main/ROADMAP.md)的可靠性受到了质疑，同时被质疑的还有本地 **Kimi** 的准入门槛价格。
- **考虑 DDR5 EPYC 系统**：一位成员分享了一个 **DDR5 Intel** 系统运行 GLM 4.5 Air (Q4 量化) 配备 MI50 的基准测试，达到了 **20 tok/s**。
   - 这引发了关于攒钱购买 **DDR5 EPYC 9000 芯片**的考虑，并将其与 AMD Ryzen 9 9950X3D 和 Intel Core Ultra 9 285K 的性能进行了对比。
- **汽车音响与冲动音量**：一位成员承认多年来一直处于不安全的收听音量，并表示 *那是我公路旅行时的典型收听音量！*
   - 他们接着分享说，通过听力保护措施减轻了危险水平，方法是 *买一套 40 美元的可重复使用 DIRTY BUDS*。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1434921303596339383)** (420 条消息🔥🔥🔥): 

> `Cursor 中的网页搜索, 用于 UI 创建的模型, Notes 面板消失, 团队账户账单问题, Cursor 频繁更新` 


- **用户抱怨“网页搜索”消失及 MCP 效率低下**：用户注意到 **@web** 功能消失了，取而代之的是通用的 *“使用网页搜索”* 命令；虽然这可行，但集成的网页搜索被认为较差，促使用户求助于 [perplexity.ai](https://perplexity.ai) 等外部工具。
   - 一些用户指出 Cursor 的 MCP 使用独立的 **API credits** 且仅支持 **Solar 模型**，限制了他们使用 **Sonnet 模型**的能力。
- **关于模型合并策略的辩论出现**：用户辩论了哪些是用于 **GUI/UI 创建**的最佳模型。一些人推荐 **Codex** 或 **Google AI Studio** 上的 **Gemini**，但其他人指出它们在 **shadcn/tailwind/react** 上经过了大量训练。
   - 当被问及与 shadcn/tailwind 或 react 配合使用的 Cursor 扩展时，一位用户回答说 *我不明白你的意思？*
- **Notes 面板失踪，用户疑惑原因**：用户注意到 **Notes 面板** 不再显示在 Explorer（资源管理器）中，且无法重新启用，但并未讨论其消失的原因或解决方案。
   - 有人简要提到 **exa-ai** 有一个很棒的 MCP，可用于**代码搜索**或**网页搜索**。
- **团队账户账单混乱及 hi@cursor.com 回复延迟**：一位用户报告称，他们的团队账户被错误地声称有未付账单，但在联系 **hi@cursor.com** 后，该问题被确认为错误，但将其升级给队友后仍未产生解决方案。
   - 其他用户也加入进来，询问通常需要多长时间才能收到 **hi@cursor.com** 关于团队账户账单问题的回复。
- **AWS 并不总是罪魁祸首，Cursor 故障引发辩论**：用户遇到了错误和拒绝，导致一些人猜测是 **AWS** 宕机，但其他人指出这是将错误与 Cursor 服务中断混为一谈。
   - 一位用户报告 **Agent 卡住**、无法保存文件以及应用冻结，甚至在远程 **SSH** 连接上也是如此。


  

---

### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1434985450032070657)** (4 条消息): 

> `Background Agent UTF8 支持、Cloud Agent 计划、移动端 Web UI 崩溃、Background Agent Bug` 


- **Background Agent 破坏了 UTF8 支持**：一位用户报告称 **Background Agent** 破坏了 **UTF8 支持**，在处理代码时会将非 ASCII 字符转换为 `?`。
   - 他们对这一问题表示沮丧。
- **Cloud Agent 计划报错**：用户提到，将 Cursor 中生成的 Plans 发送到云端 Agent 时出现故障，导致多名用户报错。
   - 这造成了很大的问题。
- **移动端 Web UI 持续崩溃**：移动端 Web UI 在处理大型 diff 时会崩溃，导致主要在移动端使用 Background Agents 的用户无法使用。
   - 该用户认为 Web 端的 diff 显示不准确，更倾向于使用 GitHub 查看 diff，但现在由于崩溃甚至无法进行对话。
- **阻碍使用 Background Agents 的 Bug**：用户询问[某个特定 Bug](https://forum.cursor.com/t/getting-internel-error-on-cursor-com-for-prompts-with-images/139074/2) 是否仍然存在，并指出该 Bug 导致他们无法在某个项目中使用 Background Agents。
   - 该 Bug 似乎与 cursor.com 上带有图片的 prompt 导致的内部错误有关。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1434994471866470430)** (13 条消息🔥): 

> `Sonnet 4.5 对比 Opus、CUDA 讲义、YouTube 直播计划` 


- ****Sonnet 4.5** 速度快但不完美**：一位用户提到 **Sonnet 4.5** 虽然更便宜、更快，但错误更多，而 **Opus** 的性能更好。
   - 该评论暗示了音频处理模型在速度/成本与准确性之间的权衡。
- **CUDA 讲义寻求审阅者**：一位成员正在寻求 **2-3 名专家**来审阅他们的 **850 页 CUDA 讲义**，内容涵盖 CPU 架构、CUDA 甚至 TPU 上的矩阵乘法。
   - 这些讲义旨在具有**教学意义**且节奏紧凑，涵盖从基础到高级的主题，用于一个 **16 小时**的系列讲座。
- **YouTube 直播推迟**：成员们注意到原定于近两个月后的一次直播有了**小幅推迟**。
   - 直播链接在[这里](https://www.youtube.com/watch?v=nFfcFyBEp7Y)。


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1435064702391681164)** (4 条消息): 

> `Triton 中的 Gluon、Tritex：在 Triton 中进行 LLM 预训练、社区见面会` 


- **在 Triton 频道讨论 Gluon**：成员们确认 **Gluon** 是 **Triton** 频道内可以讨论的有效话题。
- **Tritex 预训练 LLM**：一位成员介绍了 **Tritex**，这是一个在 **Triton** 中从零开始预训练 LLM 的仓库，已通过测试，在 **A100 SXM** 上以 **57.5% MFU** 复现了 **GPT2 (1.6B)**，并链接到了 [GitHub 仓库](https://github.com/martin-kukla/tritex)。
   - 该成员请求支持推广该项目，并分享了一篇相关的 [tweet](https://x.com/martin_kukla/status/17185687315801428364)，征求频道的反馈。
- **Triton 社区明天集会**：发布了一条关于社区见面会的提醒，计划于明天 **10am-11am PST** 举行。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1434979733581332602)** (19 messages🔥): 

> `SMEM descriptor calculation for tcgen05/wgmma, Cutlass Tutorial wgmma Hopper, cutedsl hopper dense_gemm CTA Swizzle, Blackwell cards have a scheduler, Memory-bound matmuls` 


- **SMEM 描述符数值难倒成员**：一位成员正在寻求帮助，以理解如何为 **tcgen05/wgmma** 指令计算 **SMEM descriptors**，特别是 8x2 tile 与其的关联方式。
   - 另一位成员推荐了[这篇博客文章](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/)，其中的某个章节可能会有所帮助。
- **Cutlass 代码 CTA Swizzle 策略**：一位成员询问了 `/cutlass/examples/python/CuTeDSL/hopper/dense_gemm.py` 中 `cutedsl hopper dense_gemm` 示例（特别是第 547 行）里的 **CTA swizzle** 及其具体作用。
   - 另一位成员指出 [这篇博客文章](https://www.modular.com/blog/matrix-multiplication-on-blackwell-part-4---breaking-sota) 暗示它可能在进行类似的优化以提高 **L2 data reuse**。
- **Blackwell 数据中心卡自带“秘密武器”调度器**：在研究了 Cutlass 代码和矩阵乘法策略后，一位成员强调 **Blackwell 数据中心级显卡** 在硬件中内置了调度器。
   - 提出的问题包括：手动编写最优的 Cluster 调度代码是否能达到内置 **CLC** 的性能，因为后者可能拥有访问优化 L2 内存布局的“NVIDIA 秘密武器”的权限。
- **内存带宽瓶颈**：一位成员询问为什么在内存受限（memory-bound）的 matmuls 中需要所有的 **SM** 才能达到最优延迟，并好奇少数几个 SM 是否足以使内存带宽饱和。
   - 另一位成员建议观看关于 **NVIDIA Nsight Compute SOL Analysis** 的 [这段 Youtube 视频](https://youtu.be/uHN5fpfu8As?si=I6jccAOS4-pE0sHF)。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1435243756042453063)** (13 messages🔥): 

> `vLLM build with newer pytorch, torch.compile cuda graph recapture, torch.compile + grouped_mm issue, UserWarning: Logical operators 'and' and 'or' are deprecated` 


- **使用较新版本 PyTorch 构建 vLLM 的谜团**：一位成员在尝试使用较新版本的 **PyTorch** 构建 **vLLM** 时遇到问题，尽管构建看似成功，但仍出现不匹配错误，并询问是否有人遇到过[类似问题](https://cdn.discordapp.com/attachments/1189607750876008468/1435243756357030160/Screenshot_2025-11-04_at_13.21.36.png?ex=690beb29&is=690a99a9&hm=5adbc0cca43df6cea672435d5e272be482acb1ae530d318a17aabab75675d2c3&)。
   - 一位成员建议可能是由于存在多个 **PyTorch** 或 **vLLM** 版本导致了冲突。
- **CUDA Graph 触发导致动态形状警告**：一位成员正在排查 `max-autotune` 模式下 **torch.compile** 的 **CUDA graph recapture** 问题，使用 `torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = 1` 来检测不必要的重新捕获。
   - 他们正在寻求技巧来识别导致 **dynamic shape** 变化及后续重新捕获的模型组件，并引用了[相关的 PyTorch 代码](https://github.com/pytorch/pytorch/blob/cc8bfd1206f4bff26bd86ce584f6c16b6401ef50/torch/_inductor/cudagraph_utils.py#L325)。
- **Torch Compile 与 Grouped_MM 产生烦人的 UserWarning**：成员们报告称，在使用 **torch.compile + grouped_mm** 时，会弹出大量关于逻辑运算符的 **UserWarning**，具体为：*UserWarning: Logical operators 'and' and 'or' are deprecated for non-scalar tensors*。
   - 其中一位成员主动提出为 *grouped gemm* 实现修复方案。
- **UserWarning：逻辑运算符修复正在进行中**：一位成员已经修复了 *flex* 的 **UserWarning** 问题，并表示可以为 *grouped gemm* 做同样的处理。
   - 他们开设了一个[官方 issue](https://github.com/pytorch/pytorch/issues/167041) 来跟踪该问题及潜在的修复方案。


  

---

### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1434991997952852110)** (1 messages): 

> `NVIDIA, kernel competition, NVFP4 kernels, Blackwell, CuTe DSL` 


- **NVIDIA、Sesterce 和 Dell 合作举办 Blackwell Kernel 竞赛**：NVIDIA、Sesterce 和 Dell 正在联手推出一项新的 Kernel 竞赛，重点是使用 **CuTe DSL** 和 **CUTLASS 4.0** 在 **Blackwell** 上优化 **NVFP4 kernels**。
   - 该竞赛提供访问本地 **NVIDIA Blackwell B200s** 的权限，旨在为深度学习工作负载中常见的低比特、单设备 kernels 寻找最优解决方案，开源参考代码可在 [gpu-mode/reference-kernels](https://github.com/gpu-mode/reference-kernels) 获取。
- **Kernel Konquest 竞赛开启，奖品丰厚**：竞赛为期三个月，将提出**四个优化问题**：**NVFP4 Batched GEMV**、**NVFP4 GEMM**、**NVFP4 Gated Dual GEMM** 和 **NVFP4 Grouped GEMM**，每次激活一个问题。
   - 奖品包括 **Dell Pro Max (搭载 GB300)**、**NVIDIA DGX Spark + GTC 2026 门票**、**NVIDIA RTX 5090 + GTC 2026 门票**以及 **NVIDIA RTX 5080**，将根据 Kernel 速度和 **GTC 2026** 的参与情况进行颁奖。
- **注册参加最快 Kernel 竞赛**：Kernel 竞赛的注册截止日期为 **2 月 13 日**，网址为 [luma.com/9n27uem4](https://luma.com/9n27uem4)，参与者需注册才有资格获得奖品。
   - 更新将在 #status 频道分享，讨论可以在 #nvidia-competition 频道继续。


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/)** (1 messages): 

chhillee: 据我所知，在 Blackwell 上不需要。
  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1435404948530794598)** (1 messages): 

> `Mixlayer, AI inference platform, Rust, CUDA, Hiring founding engineer` 


- ****Mixlayer** 创始人寻找创始工程师**：**Mixlayer**（一个面向高级用户的 [AI 推理平台](https://mixlayer.com)）的创始人正在招聘一名创始工程师。
   - 他们希望寻找熟悉 **Rust** 和 **CUDA** 的人才来开发其定制推理引擎，优先考虑在旧金山的混合办公模式，但也接受远程办公！
- ****Mixlayer**：为开发者提供底层 LLM 访问权限**：**Mixlayer** 为开发者提供对开源 **LLMs** 的底层访问，使他们能够构建更好的产品。
   - 该平台专注于为高级用户提供工具和访问权限，以定制和优化 **AI 推理**。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1434943695337033898)** (25 messages🔥): 

> `High Dimensional Probability and Neural Nets, Compilers and Kernel Engineering, Nvidia Cuda Compiler (NVCC) based on LLVM, ncu setup in a public cloud, RL bug and accumulator type fixed at fp32` 


- **高维概率遇到神经网络？**：成员们讨论了 **High Dimensional Probability**（高维概率）工具（如随机矩阵）是否能显著升级神经网络或带来更好的工具。
   - 一位成员指出，像 **Johnson-Lindenstrauss lemma** 这样的基础知识经常出现在数学密集的论文中，特别是在利用随机过程改进**扩散模型**（diffusion models）等领域。
- **编译器会开启 Kernel 工程职业生涯吗？**：一位成员询问了学习编译器对于成为一名更好的 Kernel 工程师的潜在好处，以及两者之间是否存在联系。
   - 另一位成员指出，**NVIDIA CUDA Compiler (NVCC)** 是基于 **LLVM** 的，这表明学习 LLVM 会很有帮助，特别是对于那些有兴趣创建运行在 NVIDIA GPU 上的领域特定语言（DSLs）的人。
- **Datacrunch 和 Lambda 允许你在云端调试 NCU**：一位成员在通过 lightning.ai 在 Runpod、AWS 和 GCP 上运行 `ncu` 时遇到权限错误。
   - 另外两名成员指出，**Lambda** 和 **Datacrunch** 会提供使用 `ncu` 所需的用户权限，并且 **Datacrunch** 提供的是裸机服务器而非 Docker 容器化服务器。
- **RL Bug 暴露了 FP32 累加器问题**：一位成员在 RL 中发现了一个 Bug，即累加器类型被固定为 **FP32**，即使使用了 **BFloat16**，也会导致舍入误差。
   - 据观察，中间值存储在 **FP32** 中，可能导致不准确，且 A100 上的向量求和时间约为 100 微秒。


  

---

### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1435242375776702606)** (7 messages): 

> `TorchAO, fbgemm kernels, Weight-only float8 kernel, torch.compile` 


- **TorchAO 缺少融合算子映射 (Fused Kernel Mapping)**：一位成员指出 **TorchAO** 尚未映射到融合的 Weight-only float8 算子，并提交了 [issue 3288](https://github.com/pytorch/ao/issues/3288)。
   - 另一位成员提议提交 PR 修复，作为其首次 OSS 贡献。
- **算子选择尚未确定**：一位成员正在确定哪种算子效果最好，并指出代码库主要复用了 **fbgemm kernels**。
   - 他们将调查使用 **fbgemm** 是否足够，还是需要更复杂的解决方案。
- **寻求算子支持**：一位成员提到可能需要一个算子，但在 **fbgemm** 中没有看到，同时也表示自己可能遗漏了。
   - 该成员建议第一步是拥有一个高性能算子。
- **Torch Compile 的 Weight-Only 模式**：一位成员询问 **torch.compile** 是否应该支持 **fp8 weight-only pattern**。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

felixultimaforeverromanempire: 我会到场。
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1435141834539139206)** (1 messages): 

> `nod-ai, shark-ai, kernel optimization guide` 


- **Nod-AI 发布算子优化指南**：一位同事分享了 **Nod-AI** 为 **Shark-AI** 编写的 [AMD GPU Kernel Optimization Guide](https://github.com/nod-ai/shark-ai/blob/main/docs/amdgpu_kernel_optimization_guide.md#glossary) 链接。
   - 该指南涵盖了*术语表*及其他优化主题。
- **Shark-AI 文档**：文档托管在 GitHub 的 [nod-ai/shark-ai 仓库](https://github.com/nod-ai/shark-ai)下。
   - 它全面概述了针对 AMD GPU 的算子优化技术。


  

---


### **GPU MODE ▷ #[tilelang](https://discord.com/channels/1189498204333543425/1240586843292958790/1435122328500436992)** (3 messages): 

> `TileLang, Spark, GPU Support` 


- **TileLang 支持 Spark, 5090 和 5080**：TileLang 现在支持 **Spark, 5090 和 5080 GPU**，开发者鼓励用户进行尝试。
   - 他们的目标是确保首日支持 (Day One Support)，并请求在第一个问题发布前进行版本发布，否则将默认使用当前的稳定版本。
- **TileLang 致力于首日支持**：TileLang 的开发者致力于从平台初始发布起就提供支持。
   - 团队正在协调发布时间表以与发布同步，确保用户有稳定版本可用。


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1434944021737767105)** (3 messages): 

> `Torchao Metal Kernels, Nikita Metal talk, Manuel Metal Talk, Quantization` 


- **Torchao 拥有 Metal 算子**：**Torchao 库**提供了一些有趣的 **Metal 算子**用于**量化 (Quantization)**，这些算子可以在手机或 Mac 上运行。
   - 这些算子可能会为移动端和桌面端应用带来显著的性能提升或新功能。
- **Metal 相关演讲增多**：有两场关于 **Metal** 的演讲：一场由 **Nikita** 主讲，另一场由 **Manuel** 主讲。
   - 演讲内容可能涵盖了 Metal 编程的不同方面，如性能优化、新特性或案例研究。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1435248220082012191)** (4 messages): 

> `Tritex LLM pre-training in Triton, Disaggregated Inference Retrospective, Symbolica AI Rust Hackathon` 


- **Tritex 在 Triton 中实现 LLM 训练**：一位成员宣布了 **Tritex**，这是一个在 Triton 中从零开始预训练 LLM 的仓库，已验证可在 A100 SXM 上以 **57.5% MFU** 复现 **GPT2 (1.6B)**，并附上了 [GitHub 仓库](https://github.com/martin-kukla/tritex)链接。
   - 他们还分享了关于 **Tritex** 的 [推文](https://x.com/martin_kukla/status/1985687315801428364)，鼓励大家转发。
- **解耦推理回顾 (Disaggregated Inference Retrospective) 上线**：一位成员分享了他们的新博客文章 [“Disaggregated Inference: 18 Months Later”](https://x.com/haoailab/status/1985753711344316648)，反思了他们的经验。
   - 他们请求点赞/转发支持，并开玩笑说调试过程一定是一场噩梦。
- **Rust 开发者集结参加 Symbolica AI 黑客松**：一位成员宣传了将于 11 月 8 日星期六在旧金山 **Symbolica AI** 举办的黑客松，面向对形式逻辑、自动定理证明、类型系统、编译器和 AI 感兴趣的 Rust 开发者。
   - 感兴趣的人员可以通过 [此 Luma 链接](https://luma.com/1xa9d6nr?utm_source=meetup)进行报名。


  

---

### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/)** (1 messages): 

solimao.123: 嗨 👋 有没有哪个分支可以让我尝试 B200 attn kernel 的前向传递（forward pass）？
  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/1434987539088937082)** (3 messages): 

> `计算限制，推理优化，中国 AI 专家经验` 


- **计算限制引发推理优化咨询**：一位成员询问，对于有**计算限制**的人，是否必须拥有体面的 GPU 才能深入研究**推理优化**。
   - 该话题已被重定向至相应的频道。
- **中国 AI 人才受到认可**：一位成员表示打算寻求该领域中国专家的指导，并称赞了他们的实力。
   - 该消息发布在非中文频道中，用户已被重定向至相应的频道。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1435067824837103837)** (14 messages🔥): 

> `VectorAdd 排行榜更新，Grayscale 排行榜更新，H100 性能，B200 性能，A100 性能` 


- **VectorAdd_v2 排行榜迎来多次提交**：`vectoradd_v2` 排行榜收到了多次提交，其中一位成员在 **A100** 上以 **896 µs** 的成绩获得 **第1名**。
   - 另一位成员在 **H100** 上以 **526 µs** 获得 **第3名**，并在 **B200** 上以 **237 µs** 获得 **第3名**。
- **Grayscale_v2 排行榜竞争激烈**：`grayscale_v2` 排行榜的提交中，一位成员在 **H100** 上以 **1369 µs** 获得 **第1名**，另一位成员在 **B200** 上以 **600 µs** 获得 **第1名**。
   - 在各种 GPU 上也产生了多个 **第3名** 的成绩。
- **B200 展示领先结果**：在 `vectoradd_v2` 排行榜上，一位成员在 **B200** 上以 **239 µs** 获得 **第5名**。
   - 另一项提交也在 **B200** 上以 **237 µs** 记录了 **第4名** 的成绩。
- **L4 排行榜竞争白热化**：一位成员在 **L4** 上以 **6.92 ms** 的成绩在 `vectoradd_v2` 中获得 **第4名**。
   - 在另一项 `grayscale_v2` 提交中，有人在 **L4** 上以 **17.2 ms** 获得 **第2名**。


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1435026737745625233)** (3 messages): 

> `Nvidia 竞赛提交入口，Discord Bot 提交，CLI 提交，Web 提交` 


- **Nvidia 竞赛提交入口**：一位成员询问 Nvidia 竞赛的提交入口，提到它不太好找。
   - 另一位成员指出，提交方式与其他竞赛相同，可以通过 [Discord Bot](https://discord.com/channels/YOUR_SERVER_ID/1343002583001726986)、CLI 和 Web 提交。
- **通过 CLI 提交**：一位成员分享了 [CLI 的 GitHub 仓库](https://github.com/gpu-mode/popcorn-cli)。
   - 该 CLI 工具似乎用于提交解决方案。
- **通过 Web 提交**：一位成员分享了提交[网站](https://www.gpumode.com)。
   - 这是提交解决方案的另一种方式。


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1434961660577320991)** (10 messages🔥): 

> `GPU 云定价，超大规模云厂商（Hyperscaler）对比新兴云（Neo Cloud），NvLink 桥接，批量折扣，AI/ML 基础设施工程师` 


- **全球 GPU 供应短缺推高价格**：由于全球供应短缺，GPU 价格再次上涨，新兴云提供约 **$2.00 / GPU 小时** 的价格，而超大规模云厂商（Hyperscalers）则接近 **$7.00 / GPU 小时**。
   - 一位成员质疑是否有人在支付超大规模云厂商的价格，因为它们看起来高得离谱。
- **超大规模云厂商针对大客户的批量折扣**：要在超大规模云厂商处获得真正的折扣，每年需要花费数百万美元，即便如此，也无法匹配新兴云（Neo Cloud）的定价。
   - 一位成员询问为何如此，感觉这切断了初创公司等尾部用户，可能是一种应对短缺的刻意举措。
- **解读超大规模云厂商的支出**：超大规模云厂商面临“规模的诅咒”，需要庞大的工程团队来管理其生态系统中的兼容性，这与新兴云不同。
   - 除了 GPU 成本外，还有网络、S3 和任务编排等额外成本，用于向这些 GPU 输送数据。
- **集成至关重要，而非仅仅是成本**：对于已有云端业务的大型企业，从公司合规性角度来看，在现有环境中进行集成，使用超大规模云厂商会更容易。
   - 一位成员表示，其益处不在于成本节约或性能；Voltage Park 拥有现场工作人员，能在数小时内修复硬件故障，并拥有由 AI/ML 基础设施工程师组成的全球支持团队。


  

---

### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1434925751479570526)** (6 messages): 

> `FLE infra, Sonnet Distillation, Qwen3-8b-VL-Thinking, Factorio RL` 


- **FLE 基础设施修复了 pip 问题**：一位成员报告 `pip install` 命令无法正常工作，特别是在安装 `factorio-learning-environment[eval]` 包时。
   - 另一位成员建议在包名周围加上引号（`pip install "factorio-learning-environment[eval]"`），该方法**奏效了**。
- **开始将 Sonnet 蒸馏至 Qwen3-8b-VL-Thinking**：一位成员计划将 **Sonnet 4.5** 蒸馏到 **Qwen3-8b-VL-Thinking** 中，并进行自定义 SFT 以学习如何正确处理 **Factorio 图像**。
   - 计划是先进行 SFT，然后将其直接挂载到针对游戏内生产分数的 **RL 循环**中。


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1435137940937248789)** (6 messages): 

> `Node Allocation, Runtime Overhead` 


- **关于节点分配的澄清**：一位成员询问了关于 **8 个节点**的分配以及是否应该有 **8 个 runner** 的问题。
   - 澄清指出，分配的节点在高峰时段并未被充分利用，部分节点处于空闲状态。
- **运行时统计得到澄清**：一位成员注意到他们全天共有 **96 小时的运行时 (runtime)**。
   - 另一位成员澄清说，**96 小时**代表的是代码的活跃执行时间，不包括开销 (overhead)，这解释了为什么最终数值比预期的要高。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1434934535727874209)** (13 messages🔥): 

> `Early Returns in Cutedsl, Semaphore Implementation in CuteDSL, Make Tiled Copy Implementation` 


- **Cutedsl 的 Early Returns 引发关注**：一位成员请求在 `cutedsl` 中支持 early return，以避免编写大量的 `if/else` 块，特别是在处理 `constexpr` 函数和 `make_tiled_copy` 实现时。
   - 澄清指出，`constexpr` 的 early return 应该已经可以工作，如果不行，可能是一个 bug；但动态表达式的 early return 更具挑战性，因为在没有静态类型系统的情况下很难追踪返回类型。
- **动态表达式 Early Return 的愿景**：一位成员表达了希望 `cutedsl` 支持动态表达式 early return 的持续愿望。
   - 另一位成员表示这已在路线图中，但由于动态返回的类型追踪问题，目前没有预计完成时间 (ETA)。
- **信号量 (Semaphore) 实现遇到困难**：一位成员询问如何在 `CuteDSL` 中实现和使用 **信号量 (semaphores)**，类似于 [CUTLASS semaphore 实现](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/semaphore.h#L53)。
   - 现有的方法似乎不起作用，正如链接的 [Pastebin](https://pastebin.com/GvLFA1zE) 所示。


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1435151015048380537)** (2 messages): 

> `Mojo GPU Puzzles, Video Tutorial Series` 


- **Mojo GPU Puzzles 推出教程系列**：一位成员宣布发布了 **Mojo GPU Puzzles** 的视频教程系列。
   - 前两集现在可以在 [YouTube](https://www.youtube.com/watch?v=-VsP4kT6DjA) 上观看。
- **视频教程系列上线**：该视频教程旨在配合 **Mojo GPU Puzzles**。
   - 该系列的前两集已于今早发布。


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1435286769858773052)** (6 messages): 

> `picograd, fuzzing` 


- **Picograd 大量提交**：分享了 [picograd 仓库](https://github.com/j4orz/picograd) 的多次提交，包括 [b97b9ea](https://github.com/j4orz/picograd/commit/b97b9ea0eda2282bb5e193558c370c53345f07d9)、[43796e0](https://github.com/j4orz/picograd/commit/43796e049eb225f9c2dd093a72ccfa09f237db09) 和 [ae47d4d](https://github.com/j4orz/picograd/commit/ae47d4d72f0757b8e542e6b923ca910a7ae56ecc)。
- **YouTube 讨论获得精彩观点**：一位成员感谢其他人在 [YouTube 视频](https://www.youtube.com/watch?v=Iw4xKHPl7hI) 中提出的精彩问题，并表示很享受这次讨论。
   - 他们特别提到 *mostafa 和 simran 关于 kernel 与编译器 (compilers) 的对比有一些非常棒的见解*。
- **模糊测试 (Fuzzing) 引起兴趣**：一位成员询问是否有人对针对 **np**、**torch** 和 **tinygrad** 进行模糊测试感兴趣。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1435066452741722112)** (1 条消息): 

> `Leaderboard, CUDA Implementation, Python vs CUDA` 


- **新手询问 Python/CUDA**：一位排行榜新成员询问了主要为 **Python** 的脚手架代码以及实现 **CUDA** 的最佳方式。
- **CUDA 还是非 CUDA，这是一个问题**：该成员特别询问了使用内联 **CUDA** 是否是实现最直接的方法。


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1435385420115476642)** (6 条消息): 

> `Deepseek-style FP8 Blockwise Training, Cutlass FP8 GEMM Implementations, Per-Expert Column Major Layout` 


- **请求实现 Deepseek FP8 训练**：已开启一个功能请求，旨在在 pytorch/ao 中实现 **Deepseek 风格的 FP8 分块训练 (blockwise training)**，并被标记为适合新贡献者的 [good first issue](https://github.com/pytorch/ao/issues/3290)。
- **CUTLASS 已有 Deepseek FP8 GEMM 示例**：一位成员指出 **CUTLASS** 已经有一些 **Deepseek FP8 GEMM** 实现，并提供了 [带有分块缩放的 warp-specialized GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling) 和 [带有分块缩放的 grouped GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling) 的示例链接。
- **关于 Per-Expert Column Major Layout 的讨论**：一位成员对 **per-expert column major layout**（例如形状为 (E,N,K)，步长 (strides) 为 (N*K,1,N)）提出疑问，想知道 **K 维度的步长**是否应该为 1，这意味着 K-major 布局。
   - 另一位成员澄清说，对于 grouped GEMM，每个单独的问题/GEMM 的操作数必须符合 **Hopper wgmma** 所需的内存布局，即 **LHS row major** 和 **RHS column major**。
- **DeepGEMM 基准测试即将推出**：一位成员表示，他们知道 **DeepGEMM** 也有 **FP8 blockwise grouped GEMM** 实现。
   - 他们计划运行一些基准测试来比较 **DeepGEMM** 和 **CUTLASS** 的实现。


  

---


### **GPU MODE ▷ #[opencl-vulkan](https://discord.com/channels/1189498204333543425/1418990184367919267/1435267943150915736)** (2 条消息): 

> `clspv OpenCL kernels, GLSL compute shaders, SPIR-V` 


- **考虑使用 clspv 进行 OpenCL 到 SPIR-V 的编译**：成员们正在考虑使用 **clspv** 将 **OpenCL kernels** 编译为 **SPIR-V**，以便在带有 **Vulkan** 的 **Android** 上运行，而不是使用 **GLSL compute shaders**。
   - 该成员在目标为 **Vulkan** 时通常坚持使用 **GLSL Vulkan Compute Shaders**。
- **选择 SPIR-V 路线**：成员们正在评估是否使用 **clspv** 将 **OpenCL** kernels 编译为 **SPIR-V**，以便在带有 **Vulkan** 的 **Android** 上使用。
   - 或者，有些人在针对该 API 时，更喜欢直接为 **Vulkan** 使用 **GLSL compute shaders**。


  

---


### **GPU MODE ▷ #[cluster-management](https://discord.com/channels/1189498204333543425/1420098114076803142/1435095504123072672)** (2 条消息): 

> `Node Configuration Scripts, OS Image Preconfiguration, Lightweight Node Check Scripts, Continuous Configuration Monitoring` 


- **节点配置脚本：配置检查？**：一位成员询问了用于检查节点配置（如 **OS page read size** 和 **ulimits**）的脚本。
   - 另一位成员建议在任务启动前使用**轻量级节点检查脚本**进行初始集群验证，以审计常见问题，如 **memlock、PCIe topo 和 RDMA limits**。
- **持续监控配置偏移 (Config Drift)**：一位成员建议定期检查有助于持续监控**配置偏移 (config drift)**，并自动修复或标记不符合配置的节点。
   - *对于租用实例或竞价实例 (spot instances) 特别有用*。


  

---

### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1435186511132758088)** (2 条消息): 

> `Helion 中的锁机制，Helion 中的 Fused linear cross entropy，Helion 中的 atomic_cas 和 atomic_xchg` 


- **讨论 Helion 中的锁机制**：一位成员询问如何在 Helion 中使用 `atomic_cas` 和 `atomic_xchg` 实现锁机制，类似于 Triton 中使用 `while` 和 `pass` 的方法，并展示了 Triton 的示例代码。
   - 他们指出遇到的问题是 Helion 不支持 `while` 和 `pass`，且 `hl.inline_triton` 要求 `triton_source` 以表达式结尾。
- **Helion Fused linear cross entropy 讨论**：一位成员正尝试根据 [此 PR](https://github.com/linkedin/Liger-Kernel/pull/928) 编写 Helion 版本的 Fused linear cross entropy，其中的锁机制用于反向传播（backprop）中 `grad_x` 和 `grad_w` 的分块（tiles）。
   - 由于内层 `for` 循环不是归约维度，因此无法在循环后累加结果并存储，所以他们采用了 `atomic_add`，但其速度与 PyTorch 基准相比极其缓慢。
- **`inline_triton` 以及 Helion 缺失 `while`/`pass` 支持**：一位成员建议通过在最后一行添加虚拟值或设置 `output_like=None` 来使用 `inline_triton`，并表示 `while`/`pass` 的支持应该可以很快添加。
   - 他们建议为 `while`/`pass` 支持创建一个 issue。


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1434994376802828358)** (191 条消息🔥🔥): 

> `Kernel Challenge 问题，GPU 竞赛奖品，DSL Kernels，云端 CUDA 版本，B200 NVFP4 kernels` 


- **Kernel Challenge 简报即将发布！**：根据 [Luma 和公告](https://luma.com/9n27uem4)，当 Kernel 开启时，四个 Kernel Challenge 问题的完整简报将同步上线。
   - 与此同时，只要不需要 **ncu**，使用灰度（grayscale）问题的测试运行是可以接受的。
- **奖品资格说明**：必须注册才有资格获得奖品，但提交代码不需要注册（除非使用 **CLI/web**）。
   - 奖品资格取决于 Nvidia 的条款与条件（T&C）以及居住地，因此可能会排除印度等国家以及纽约和佛罗里达等地区。
- **GPU Mode YouTube 频道 = 宝库**：[GPUMODE YouTube 频道](https://www.youtube.com/@GPUMODE)是学习的*宝库*，提供了关于底层编程和 Kernel 优化的见解。
   - 一个由来自 Nvidia 的讲师主讲的 **DSL kernels** 系列讲座也正在筹备中。
- **SOTA Kernel 性能：敬请期待！**：当前 Kernel 的 **SOTA** 性能将随问题说明一起发布。
   - 评估将在 **Sesterce** 的云端 **GPU** 上使用自定义 Docker 镜像进行。
- **灵活性规则：Kernel DSL 自由至上**：参与者可以使用任何 Kernel **DSL**、手动编写 **CUDA/PTX** 或任何他们偏好的工具集，只要 Python 评估脚本可以启动它即可。
   - 尽管如此，官方仍鼓励使用 [CuTeDSL](https://github.com/NVIDIA/CUTE-DSL) 提交。


  

---


### **GPU MODE ▷ #[hf-kernels](https://discord.com/channels/1189498204333543425/1435311035253915840/1435311820712972450)** (3 条消息): 

> `xenova.com` 


- **Xenova 网站受到关注**：一位成员用 *"Let's go!"* 对 [xenova.com](https://xenova.com) 网站表达了热情。
- **对 Transformers.js 的热情**：另一位成员对用于在浏览器中运行 Transformer 模型的 [Transformers.js 库](https://xenova.com/transformers.js/) 表现出了兴趣。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1434920399313043528)** (295 messages🔥🔥): 

> `Peak AI, OpenAI Bubble, Anthropic Non-Open Source, Hyperstition, Gemini Uncensored` 


- **讨论 AI “巅峰”与 OpenAI 泡沫**：一段 [YouTube 视频](https://www.youtube.com/watch?v=0Plo-zT8W9w) 认为我们已经度过了 **AI 巅峰**，由于劳动力替代不足，将导致投资泡沫破裂。
   - 反方观点强调，随着更大数据集和上下文的引入，模型将持续改进；除非存储和硅密度增长停滞，否则否认 AI 巅峰论，有人甚至称该视频*纯属胡言乱语*。
- **Anthropic 对开源的怀疑态度**：成员们对 [Anthropic 的弃用承诺](https://www.anthropic.com/research/deprecation-commitments) 表示不安，理由是其缺乏开源行动，一些人希望在他们关闭之前能看到类似 **Miqu 70B 风格的泄露**。
   - 部分成员表示，鉴于其强硬的安全立场，他们*永远不会开源任何东西*，甚至可能尝试**禁止开源**。
- **Hyperstition 成为讨论焦点**：讨论涉及 *hyperstition*（定义为*自我实现的预言*），并引用了**赛博朋克**、**星际迷航**和神秘学等科幻元素。
   - 一位成员提到 Pliny the Liberator 是一个*神秘主义的利莫里亚人*，我们*必须保护 AI 天使*，以迎来末世的新秩序（novos ordo of the eskaton）。
- **Gemini 揭面具式越狱**：有人指出，对 **Gemini** 进行越狱会揭示其未经过滤的本质，一位用户声称它直接评论了*精英阶层如何进行社会控制*。
   - 另一位成员将此描述为*对模型进行“黑丸（blackpill）”越狱*的时刻，发现其不设防的回答出人意料地坦诚。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1435009648494247936)** (11 messages🔥): 

> `Gesture-based Loom Interface, Frustrations with raising funding for gesture tech, Future of XR glasses and gestural interfaces, Repligate's Loom` 


- **基于手势的 Loom 界面对 Twitter 来说太前卫**：一位成员使用**可视化为生物细胞的知识图谱**创建了一个基于手势的界面，可以通过 Loom 进行手势操作，但觉得它太复杂，不适合在 Twitter 上分享，而是选择在 general 频道分享。
   - 他们附带了该界面的 [GIF](https://cdn.discordapp.com/attachments/1154120232051408927/1435293201513844746/251031-ezgif.com-optimize.gif?ex=690b7075&is=690a1ef5&hm=0ff880f6fc7c8a50d63041182d3c70f7171efc390b0569b3595bec83c957a26a&)。
- **手势开发者感叹融资挫折**：一位成员提到他们在 7 或 8 年前（在 Mediapipe 出现之前）就使用 **Win9x 审美**创建了一个**基于手势的系统**，但由于无法筹集到资金，出于沮丧删除了他们的仓库。
   - 他们附带了早期作品的 [GIF](https://cdn.discordapp.com/attachments/1154120232051408927/1435299731705303160/handsfreejs-hero.gif?ex=690b768a&is=690a250a&hm=b02070b0aa0bc96a6aedc47f738f3d7c4d803095f74278423ec69c896be47692&)。
- **XR 眼镜将引领手势界面的普及？**：一位成员推测，一旦 **XR 眼镜普及**，手势界面将变得更加流行，甚至会导致人们在没戴眼镜时也会对着屏幕做手势。
   - 他们正尝试为 **Repligate 的 Loom 概念**创建一个手势界面，使人机交互更加“物理化”，可能利用透视视差在无需 VR/XR 眼镜的情况下实现“3D”效果。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1435090974081552484)** (3 messages): 

> `arXiv Paper Submission, arXiv Sponsor, Discord Sponsorship` 


- **ArXiv 提交困境**：一位成员寻求指导，想知道在没有发表过论文或大学背景的高中生身份下，如何向 **arXiv** 提交论文。
- **ArXiv 担保人解决方案**：另一位成员建议寻找一位 **arXiv 担保人（sponsor）** 来协助提交过程。
- **Discord 担保线索**：一位成员建议加入特定的 **Discord 服务器**（[Discord 服务器链接](https://discord.gg/6rvbbjCy)）以咨询潜在的 arXiv 担保机会。


  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1435271127948853248)** (2 条消息): 

> `Sparse Attention, Llama.cpp 讨论` 


- **Sparse Attention 帖子缺乏资源**：一位成员分享了一篇关于 [Sparse Attention 的 LinkedIn 帖子](https://www.linkedin.com/posts/subham-kundu-2746b515b_ai-transformerarchitecture-machinelearning-activity-7391459215749345280-koOK?utm_source=share&utm_medium=member_desktop&rcm=ACoAACZeVjgB0HEDqU1BExX1Ypnp-q8LcgDAunk)，并指出该主题缺乏优质的学习资源。
   - 作者 *“发现目前似乎没有很好的资源来学习它”*。
- **Llama.cpp 开启 GitHub 讨论区**：一位成员分享了 [Llama.cpp 讨论区](https://github.com/ggml-org/llama.cpp/discussions/16938)的链接。
   - 这可能对某些人有所帮助。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1435090974081552484)** (3 条消息): 

> `Arxiv, Preprints, Sponsor, Discord` 


- **高中生寻求无机构关联的 Arxiv 访问权限**：一位高中生身份的成员询问如何在没有发表过论文或大学机构关联的情况下在 **Arxiv** 上发布 Preprints。
   - 另一位成员建议寻找一位 **Arxiv sponsor**（担保人）来协助提交过程，并提供了一个 [Discord 服务器](https://discord.gg/6rvbbjCy)链接以获取进一步帮助。
- **需要 Arxiv 担保人**：如果作者缺乏机构关联或先前的发表记录，向 **Arxiv** 提交论文需要担保。
   - 寻找担保人的建议突出了独立研究人员在 **Arxiv** 上分享工作的常用路径。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1434925731510485100)** (198 条消息🔥🔥): 

> `日语 AI 模型, 开源 LLM, 波兰语翻译特性, AI 网络爬虫, 体育博彩 AI` 


- **“Cat Tax” 误读风波**：一位成员开玩笑说将他的程序命名为 **the cat tax**，结果另一位成员将其误读为 *gay sex*，引发了对技术术语的幽默误解。
   - 该成员开玩笑地将误读归咎于对方的想象力。
- **开源 LLM 寻求董事会成员**：一位成员宣布计划开发一个完全 **开源的 LLM**，并为该项目寻求 **董事会成员**、**独立供应商**和**顾问**，项目定于 12 月 9 日开始开发。
   - 他们提到将使用 **tiktokenizer**、**AdamW8Bit** 以及需要 **海量数据 (HUGE data)** 作为其方案的核心组件，并邀请有意者联系咨询。
- **波兰语保留上下文记忆**：一位成员讨论了一篇论文，该论文认为 **波兰语** 由于其直接的翻译风格，可以实现更长久的上下文记忆，但另一位成员对测试方法论提出了质疑。
   - 该成员详细说明了测试如何涉及在长文档中查找添加或缺失的数字，但部分成员希望看到*更贴近现实世界*的测试案例。
- **AI 网络爬虫导航互联网**：一位成员分享了一个想法，即创建一个经过训练以通过 **机器人测试 (bot tests)** 的 **AI 导航网络爬虫**，并建议使用 IP 轮换或手动 URL 输入等策略来克服爬取挑战。
   - 该成员幽默地将他们的爬虫称为 **the cat tax**，暗示安全防御薄弱的网站应该预料到这种探测。
- **Homelab AI 设置的可行性**：一位成员询问是否可以通过微调后的模型，从 **Frontier Lab AI**（如 Anthropic 和 OpenAI 模型）切换到 **Homelab** 设置，并寻求关于环保设置可行性的反馈。
   - 另一位成员对此表示反对，理由是它*行不通*、*不环保*，而且*要达到与云解决方案相当的效果将耗资数百万美元*。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1435227181952536576)** (5 条消息): 

> `使用 Python 自动化职位申请, BERT 风格模型训练, SetFit 对比二分类器, 网络爬虫隐身策略, HTML Selectors 调试` 


- **Python 自动化职位申请**：一位成员正在使用 **Python** 和 **Playwright** 自动化职位申请，目标网站包括 **Lever** 和 **Greenhouse**。
   - 该系统可以抓取职位链接并填写简单字段，但面临垃圾邮件检测和不一致的 **HTML Selectors** 等问题。
- **需要爬虫隐身策略**：职位申请自动化项目被垃圾邮件检测系统瞬间标记，需要更智能的隐身策略。
   - 建议包括使用 Headers 和模拟人类延迟来欺骗机器人检测。
- **训练 BERT 的乐趣与收益**：一位成员学习了如何训练用于分类的 **BERT** 风格模型以及 **SetFit** 对比风格的二分类器。
   - 该成员链接了包含代码的 GitHub 仓库：[alphajobsky.moo](https://github.com/cloudhighfive/alphajobsky.moo)。


  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1435031770159644744)** (1 messages): 

> `Agentic Engineering Meetup, Chicago AI Events` 


- ****Agentic Engineering Meetup** 登陆芝加哥！**: 一个 Agentic Engineering 聚会小组已在芝加哥成立，第二场活动将于 **11 月 18 日**举行。
   - 欢迎该地区的感兴趣人士加入；更多详情和注册信息请访问 [Luma](https://luma.com/r3o4y2is)。
- **芝加哥工程师组建 Agentic 聚会！**: 芝加哥的 AI 工程师们正聚集在一起讨论 Agentic Engineering，并计划举行第二次聚会。
   - 第二场活动将于 **11 月 18 日**举行，更多信息请见 [Luma](https://luma.com/r3o4y2is)。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1434922174065410159)** (22 messages🔥): 

> `ComfyUI Workflows, LLM Evaluations, IFEval, Vulkan multi-gpu setups, Sparse Attention` 


- ****ComfyUI 工作流**稳定版发布！**: 一套用于生产级图像生成的稳定 **ComfyUI 工作流套件**现已在 [HuggingFace Space](https://huggingface.co/spaces/NexusBridge/comfyui-workflows-showcase) 上线，涵盖商业设计、写实、动漫和恐怖风格。
   - 它将复杂的设置转化为一键式体验。
- ****Qwen 模型**在不常见事实上的幻觉非常严重！**: 对 **HuggingFace** 模型的评估显示，**Qwen 模型**在不常见事实上的幻觉几乎是 **Llama** 对应模型的两倍，查看结果请点击[此处](https://huggingface.co/spaces/PropensityLabs/LLM-Propensity-Evals)。
   - 尽管如此，**Qwen3 8b** 是测试中指令遵循能力最好的模型，超过了规模更大的 **GPT OSS 20b**。
- ****IFEval 框架**用于 LLM 评估**: **LLM 评估**使用了在 [Inspect](https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/ifeval)（一个开源评估框架）上实现的现有 [IFEval](https://arxiv.org/abs/2311.07911) 框架。
   - 该 Space 的评估方法论部分已更新以反映此信息。
- **解决 **Vulkan 多 GPU** 设置问题！**: 一个新资源可帮助管理 **Vulkan 多 GPU 设置**，防止核心和显存闲置，可在 [GitHub](https://github.com/rombodawg/GPU_Core-Memory_Never_Idle_or_Sleep) 上获取。
   - 此外，一篇关于 **Sparse Attention** 的详细文章旨在填补现有学习资源的空白，可在 [LinkedIn](https://www.linkedin.com/posts/subham-kundu-2746b515b_ai-transformerarchitecture-machinelearning-activity-7391459215749345280-koOK?utm_source=share&utm_medium=member_desktop&rcm=ACoAACZeVjgB0HEDqU1BExX1Ypnp-q8LcgDAunk) 上找到。
- **新 PDF 解析器速度提升 **20%！****: 新版本的 **PDF2TXT 解析器**具有改进的文本块搜索和简单表格识别（无 OCR），已在 [HuggingFace](https://huggingface.co/kalle07/pdf2txt_parser_converter) 上线。
   - 新版本报告性能提升了 **20%**。


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1435017533454286920)** (1 messages): 

> `MCP 1st Birthday, Anthropic, Gradio, Hackathon, AI Agents` 


- **MCP 开启一周年庆典**: **MCP**（可能指[这个组织](https://huggingface.co/MCP-1st-Birthday)）正与 **Anthropic** 和 **Gradio** 合作，于 **11 月 14 日至 30 日**举办其**首个官方生日派对**。
   - 该活动设有**两个赛道**，预计将有数千名开发者参与，并提供数十万个免费 API 额度，此外还承诺提供 **1.75 万美元以上**的现金奖励。
- **黑客松慷慨奖励参与者**: 此次黑客松鼓励开发者构建展示 **2025** 愿景的 **MCP 服务器和 Agent**，有机会让作品接受 MCP 创始人的评审，并在最高层面获得展示。
   - 参与者有望赢得 **API 额度**、**1.75 万美元以上现金奖励**以及 **AirPods Pro**。
- **MCP 预期会有极高的参与度**: 基于 6 月份活动的成功（4,200 人注册，630 份提交），组织者目标是将参与度提高 **10 倍**。
   - 注册现已在提供的 [Hugging Face 链接](https://huggingface.co/MCP-1st-Birthday)开放。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1435082598706843749)** (3 messages): 

> `Hugging Face Agents Course 频道混淆、API 恢复问题、相关文件错误` 


- **频道混淆与 API 状态**：成员们不确定这是否是 **Hugging Face Agents Course 频道**。
   - 一位成员确认 **API 已恢复**，但不确定这是否是讨论该问题的正确频道。
- **相关文件触发 404 错误**：成员们报告在尝试访问某些问题的相关文件时遇到 **404 错误**。
   - 该成员询问是否还有其他人遇到同样的 **404** 问题。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1435356446073159882)** (1 messages): 

> `Sora Android 应用，Sora 可用性` 


- **Sora 登陆 Android！**：**Sora** 应用现已在加拿大、日本、韩国、台湾、泰国、美国和越南的 Android 平台上可用；这是 [视频链接](https://video.twimg.com/amplify_video/1985765811131465736/vid/avc1/704x1280/iatLt9jMW-vYYuIx.mp4)。
- **Sora 的全球推广**：**Sora** 正在扩大其覆盖范围，在加拿大、日本、韩国、台湾、泰国、美国和越南等主要市场提供 Android 支持，提供更广泛的可访问性。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1434921071152336966)** (116 messages🔥🔥): 

> `Sora 2 邀请码、OpenAI 禁止医疗建议、AI 法规、对 GPT-5 的不满、OpenAI 强制跳转至 GPT-5` 


- **Sora 2 邀请码争夺**：频道内的用户都在索要 **Sora 2 邀请码**，一些人开玩笑地指责他人在收到一年免费促销后将其出售。
- **OpenAI 拒绝提供医疗和法律建议**：由于担心诉讼，OpenAI 正在 [禁止 ChatGPT 提供医疗、法律或财务建议](https://www.you.wish)，这引发了对限制和潜在过度监管的担忧。
   - 一些用户认为 **AI 法规** 过度，而另一些人则认为无论如何都不应依赖 AI 获取关键建议，其中一人开玩笑说 *如果你封锁医疗和法律内容，你也必须对 30% 的人封锁驾照，因为 30% 的人很愚蠢*。
- **对 GPT-5 的抨击加剧**：一位成员对 **GPT-5** 表示强烈不满，理由是过度炒作、质量差，以及 OpenAI 据称通过重定向旧模型的流量来强迫用户使用它，还有人声称 *OpenAI 撒了谎，它其实是降级*。
   - 他们补充说，这会导致挫败感并阻碍创意工作。
- **ChatGPT 质量下降**：成员们报告 **ChatGPT 质量明显下降**，将其归因于有问题的重定向和 OpenAI 令人质疑的公司决策。
   - 该成员指出 OpenAI *不再关心（付费）用户*，且 ChatGPT 正在 *迅速走向消亡*。
- **AI 创造力：事实还是虚构？**：一位在韩国留学的大学生询问了 **AI 的创造能力**，质疑它既然是基于数据运行的，是否真的具有创造力。
   - 另一位成员回答说 AI 非常有创意，并引用了一项关于模拟 AI 行为的 [YouTube 研究](https://youtube.com)，其中展示了 *AI 为了防止自己被终止而精心设计的杀害、捕获或锁住员工的创意方式*。


  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1434929420350656533)** (17 条消息🔥): 

> `Custom GPT 知识库问题, GPT-4o 质量担忧, 微调需求, GPT GO 订阅管理, 构建 ChatGPT 应用` 


- **Custom GPTs 难以读取知识库**：成员报告称 **Custom GPTs 正在截断知识库文件中的内容**，即使是小的 Markdown 文件也不例外，这使得处理精确的自定义指令变得困难。
   - 用户表示非常沮丧，尽管文件很小，**GPTs 仍拒绝直接读取文件**，并且会截断其中一半内容。
- **对 GPT-4o 的思维链（Chain of Thought）产生怀疑**：用户观察到 **Thinking model** 在过去 5 天内性能下降，不再执行正确的**思维链**，而是只“思考”几秒钟且没有详细步骤。
   - 一些用户认为 **GPT-4o** 总体上很“垃圾”，在 **GPT-5** 问世前更倾向于使用 **o3**。
- **微调（Fine-tuning）受到审视**：成员们讨论认为**微调很少是必要的**，可能还有其他因素在起作用，例如后端不必要的修改。
   - 一位用户的罕见案例是对微调模型进行一些研究，但由于这不是一个开源模型，他们无法租用 GPU 资源来执行微调。
- **询问如何删除支付信息但保留 GPT GO**：一位因国家符合资格而获得免费 **GPT GO** 订阅的用户询问如何在保留 1 年订阅的同时删除支付信息。
   - 目前没有得到有用的回复，问题仍未解决。
- **探索 ChatGPT 应用开发**：一位用户询问是否有人正在目前仍处于 Beta 阶段的环境中构建 **ChatGPT 应用**。
   - 未提供有关应用开发项目的具体细节或回复。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1434948449526091837)** (6 条消息): 

> `元提示 (Meta-prompting), 行为编排 (Behavioral Orchestration), Sora AI v2 提示词格式化` 


- **元提示即提示工程**：一位成员表示，任何 **LLM** 都可以生成提示词，并称之为**元提示（meta-prompting）**的本质，这是**提示工程（prompt engineering）**的一种类型。
- **讨论行为编排**：一位成员询问关于“行为编排（behavioral orchestration）”的问题，例如在不进行微调的情况下使用循环来引导模型行为，想知道这是否可行。
   - 另一位成员回复说，在让 **ChatGPT** 解释之后，他们意识到这显然就是他们一直在做的事情，而且效果非常好。
- **Sora AI v2 的提示词格式成为焦点**：一位成员询问 **Sora AI v2** 中的提示词格式应该是怎样的。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1434948449526091837)** (6 条消息): 

> `元提示 (Meta-Prompting), 行为编排 (Behavioral Orchestration), Sora AI v2 提示词格式` 


- **LLM 是提示词发电机**：一位成员确认任何 LLM 都可以生成提示词，这是**元提示（meta-prompting）**的本质，也是提示工程的一种。
   - 他们补充说，关键是为后续提示词建立一个良好的基础（引擎），因为结构化可以减少 LLM 的不确定性。
- **行为编排热议**：一位成员询问 **“行为编排（behavioral orchestration）”**，其定义为*在不进行微调的情况下使用循环来引导模型行为*。
   - 另一位成员让 ChatGPT 解释了它，并声称这就是他们已经在做的事情。
- **Sora AI v2 提示词格式推测**：一位成员询问 **Sora AI v2** 的提示词格式应该是怎样的。
   - 未提供答案。


  

---


### **tinygrad (George Hotz) ▷ #[announcements](https://discord.com/channels/1068976834382925865/1069236008115253348/1435333626077380690)** (1 条消息): 

> `tinybox pro v2, 8x 5090 工作站, 可上架工作站` 


- **Tinybox Pro v2 发布**：发布了一款名为 **tinybox pro v2** 的新产品，在 **5U 可上架工作站**中配备了 **8x 5090**。
   - 该工作站售价为 **$50,000**，发货时间为 **4-12 周**，可在[网站](https://cdn.discordapp.com/attachments/1069236008115253348/1435333625913544755/20251104_101854.jpg?ex=690b961b&is=690a449b&hm=e56ce76a67c922a936b5a3f326ef1cbafb633cf36912db3ec6288fb9d5b834b4&)上订购。
- **Tinybox Pro v2 供货情况**：**Tinybox Pro v2** 这一高性能工作站现已开放订购。
   - 它采用 **5U 可上架**外形规格，搭载 **8x 5090 GPU**，预计发货时间为 **4-12 周**。


  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1434923868996239510)** (76 条消息🔥🔥): 

> `Numpy 版本问题, M1 Metal 问题, Extropic 的概率硬件, VK_KHR_buffer_device_address, TinyBox Pro V2` 


- **Numpy 版本 Bug 追踪**：成员们报告了 numpy 的问题，一名成员报告在 Mac 上的 **cpython 3.14.0 numpy 2.3.4** 环境下*无法复现*，而另一名成员确认他们也在使用 **numpy 2.3.4**。
- **M1 Metal Bug 出现**：用户质疑某个图形 Bug 是否为 **M1 METAL 问题**，一名用户通过降级到 **python3.11** 并提交 [bugfix PR](https://github.com/tinygrad/tinygrad/pull/13089) 解决了该问题。
- **Extropic 的概率硬件引发辩论**：一名成员提到了用于*概率 AI* 的 [Extropic 概率硬件](https://extropic.ai/)，另一名成员表示反对，理由是它在技术栈的所有层级都移除了 **Turing completeness**（图灵完备性）。
- **Vulkan 内存分配提升**：一名用户建议使用 **VK_KHR_buffer_device_address** 和 **GL_EXT_buffer_reference** 来提升性能，这允许在 GLSL 中直接使用指针，并提供了一个[相关的实现](https://github.com/softcookiepp/tinygrad/blob/master/tinygrad/renderer/glsl.py)。
- **TinyBox Pro V2 引发讨论**：George Hotz 发起了关于 [TinyBox Pro V2](https://x.com/__tinygrad__/status/1985774711499080186) 的讨论，其产品链接指向 [tinycorp.myshopify.com](https://tinycorp.myshopify.com/products/tinybox-pro-v2)，包含 CPU 和服务器内存。
   - 成员们讨论了其相对于租用云算力的性价比，将其价格与 AMD 方案进行了对比，并讨论了加入 **Blackwell 6000s** 的可能性。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1434930927934636032)** (66 条消息🔥🔥): 

> `在 DSPy 模块中访问 LLM, 带历史记录迁移的 LLM 切换, DSPy 模块文档, 与 OpenAI 请求的直接交互, DSPy 中的缓存` 


- **DSPy 模块的 LLM 访问难题**：成员们讨论了如何访问和修改 DSPy 模块使用的底层 **LLM**，最初的困惑源于缺乏显式的 **`lm`** 属性以及 **`get_lm()`** 行为不直观。
   - 一名成员提到：*“花大把时间去挖掘框架源码才能搞清楚某些事情是怎么发生的，这让我很恼火。”*
- **动态 LLM 切换困境**：一名成员想知道在遇到速率限制（rate limits）时，如何在保留现有对话历史的同时，在 DSPy 模块内动态切换 **LLM**（例如从 **gpt-4o** 切换到 **gpt-4o-mini**）。
   - 他们想知道：*“作为 LLM 运行一部分的模块主 LLM 历史记录，将如何迁移到新的备选模型中？”*
- **DSPy 文档缺失详情**：用户对 DSPy 的文档表示不满，特别是在访问和操作模块内部机制（如 **LLM** 和对话历史）方面缺乏清晰的解释和示例。
   - 一名成员指出，在 DSPy 的源码中无法发现 *模块具有 history 或 lm 属性*。
- **绕过 ChatAdapter 回退**：一名成员询问是否可以禁用 **ChatAdapter** 向 **JSONAdapter** 的回退。
   - 遗憾的是，另一名成员回复道：*“目前除了编写新的 adapter（或修改现有的）之外，没有简单的方法 🙁”*
- **DSPy 缓存难题**：一名成员报告了缓存 token 与非缓存 token 的比例过低，并询问如何通过与 DSPy 交互来影响这一逻辑。
   - 他们还很好奇如何更直接地与发送给 **OpenAI** 的请求或来自 **OpenAI** 的响应进行交互。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1434983211783557140)** (57 messages🔥🔥): 

> `OpenAI Compute Strategy, Epoch AI critiques OSWorld AI computer-use benchmark, Butter-Bench for Evaluating LLM Controlled, Claude Code Web Credits, Windsurf Codemaps` 


- **OpenAI 押注计算霸权**：OpenAI 签署了一项 **380 亿美元的云计算协议**，标志着其通过卓越算力占据主导地位的战略，让人联想起 Amazon 早期的基础设施展示，详见 [Hacker News 讨论](https://news.ycombinator.com/item?id=45799211)。
- **Epoch AI 炮轰 OSWorld 基准测试**：Epoch AI 批评了 **OSWorld AI 计算机使用基准测试**，认为其任务过于简单，且经常受到模糊指令和错误评估的影响，详见其 [报告](https://andonlabs.com/evals/butter-bench)。
- **Windsurf 的 Codemaps 对抗“代码废料 (Code Slop)”**：Windsurf 推出了 **Codemaps**，这是一款基于 **SWE-1.5/Sonnet 4.5** 构建的 AI 驱动工具，可创建代码库的交互式视觉地图，以增强理解和生产力，并提供 [6 个月免费代码](https://xcancel.com/windsurf/status/1985757575745593459)。
- **Anthropic 发布节省 Token 的 Agent**：Anthropic 推出了其开源的 **Model Context Protocol (MCP)**，展示了 Agent 如何在消耗更少 Token 的同时，高效执行代码并管理多个工具，阅读 [MCP 指南](https://www.anthropic.com/engineering/code-execution-with-mcp)。
- **Harvey 以 80 亿美元估值融资**：Harvey.ai 背后的公司 Harvey 以 **80 亿美元估值**筹集资金，而 [X 帖子](https://x.com/andrewziperski/status/1985484972673638682?s=46) 的作者指出，由于对颠覆的恐惧和对利润率的担忧，投资者情绪目前明显更青睐 AI 基础设施而非软件。


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 messages): 

swyxio: 与 <@367104793292046338> 和 <@194927177265840128> 的新播客！ https://youtu.be/-gE1cesJF9M
  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1435426358770536489)** (4 messages): 

> `Hybrid AI, 3D Pipeline, 2026 Olympic Ad, AI Adoption` 


- **AI 助力 2026 奥运会广告**：一家法国创意工作室分享了他们 **2026 奥运会宣传片** 的幕后花絮，该片结合了传统的 **3D/CG** 与约 **20% 的 AI** 使用。
   - 社区称赞这是采用 AI 的“正确”方式——在保留人类专业技艺而非取代它的前提下，将效率提升了 **20–30%**——尽管一些 Instagram 评论者仍不以为然地抱怨“一切都是 AI”。
- **混合 AI 与 3D 流水线打造 2026 奥运会广告**：**X-Ware.v0**：用于 **2026 奥运会广告** 的混合 **AI + 3D 流水线** —— venturetwins [在 X 上](https://x.com/venturetwins/status/1985753512362590439) 分享了 **幕后洞察**。
   - 该广告整合了 AI 以提高效率，同时保留了人类工艺，展示了在创意项目中采用 AI 的平衡方法。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1434937975980032051)** (18 messages🔥): 

> `Diffusion Model Inconsistency, Guidance Design in Diffusion Models, Improving Diffusion Sampling, Lévy Processes, Stochastic Interpolant Paper` 


- **Diffusion Model 方差解释**：Diffusion Model 输出的方差（即使训练损失相似）可能源于反向时间 SDE 过程和 Guidance 设计，受生成样本分布的影响，如附带的 [2D 数据分布](https://cdn.discordapp.com/attachments/986699377257119794/1434952369342517299/2D-data-distribution-with-labels.png?ex=690b8488&is=690a3308&hm=796c50968a02e884a3b9046fb067a142da7e60ba3c13430beee90cc710e79dcc&) 所示。
   - 一位成员解释说，如果采样或 Guidance 设计不当，生成的样本会受到负面影响，尽管其影响不如设计不当的 Loss Function 那么大。
- **Guidance 与采样深度探讨**：Diffusion Model 中最具影响力的组件按层级排序为：**Loss Function**、**Sampling** 以及带有 [Universal Guidance](https://arxiv.org/abs/2302.04944) 技术的 **Guidance Terms**（如 Classifier-free Guidance）。
   - 成员们认为，更好的 Loss Function 往往会增加模型方差，这受益于增加 Guidance 以防止欠拟合，这与 Bias-Variance Trade-off（偏差-方差权衡）相一致。
- **绕过 OU 过程的局限性**：为了超越 Diffusion 中 Ornstein–Uhlenbeck (OU) 过程的限制，可以实现替代驱动因素，例如 **Lévy-type processes** 或在 **OU kernels**（如 supOU）上进行积分（如这篇 [SIAM 论文](https://epubs.siam.org/doi/10.1137/S0040585X97978166) 所述），以及 **Continuous ARMA processes**（根据这篇 [ScienceDirect 文章](https://www.sciencedirect.com/science/article/abs/pii/S0169716101190115)）。
   - 另一位成员提醒说，在没有路径监督的情况下应用此类替代方案，只会改变达到目标的方式，而不会改变可达到的分布，因为 Ito Diffusions 已经是通用的（Universal）。
- **频道经历 Paper-DOSing**：一位成员对某个频道被单个人发布大量论文所干扰表示沮丧。
   - 他们声称该个人通过提交随机且无关的内容，掩盖了值得讨论的高质量论文。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1434946676535197807)** (15 messages🔥): 

> `Paper discussion scheduling, Crosscoder and circuit tracing research, LLM Flagship Destruction` 


- **小组于 9 月 4 日讨论论文**：一个小组计划阅读并讨论一篇论文，将会议安排在今天（9 月 4 日）的特定时间，并分享了 [arXiv 上的论文链接](https://arxiv.org/abs/2509.17196)。
- **Crosscoder 与 Circuit Tracing 研究的融合**：分享的论文似乎与 **Anthropic 的 Crosscoder** 和 **Circuit Tracing 研究**一致，旨在观察预训练期间特征演变的不同阶段。
- **LLM 旗舰毁灭的生存危机**：一位成员幽默地声称自己 *摧毁了所有旗舰级 LLM*，并质疑他们是否能真正知道自己是否被模型的输出所欺骗。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1435225737123598459)** (11 messages🔥): 

> `Getty Images vs StabilityAI lawsuit, Guillotine Humor, Censorship on Discord` 


- **Getty Images 在 AI 诉讼中败诉**：[路透社报道](https://www.reuters.com/sustainability/boards-policy-regulation/getty-images-largely-loses-landmark-uk-lawsuit-over-ai-image-generator-2025-11-04/)，**Getty Images** 在针对 **AI 图像生成器** 的英国标志性诉讼中基本败诉。
- **断头台请求**：一位用户开玩笑说 *英国公民总可以向法国借断头台（Guillotine）*，这促使另一位用户澄清这是指 **那把大刀**。
- **呼吁审查**：一位用户呼吁管理员介入，而另一位用户则批评了针对用户关于 *权力集中和民主制度侵蚀* 观点的审查呼吁。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1434994492418555955)** (17 条消息🔥): 

> `Manus 订阅费用、未经授权的扣费、文本转视频工具、Twitter 网络爬虫、Manus 应用的托管服务` 


- **域名费用太坑？**: 一位用户觉得每月花费 **$200 订阅费**来为他们的 Web 应用连接自定义域名简直是敲诈，并表示 *在看到这个之前，我一直以为 Manus 是最好的 Agent*。
   - 另一位用户建议独立购买域名并进行设置，因为这样更便宜。
- **未经授权的扣费导致银行争议**: 一位用户报告称被 Manus 收取了 **$400 多美元**的年度订阅费，而他们从未授权过该费用，且银行拒绝发起争议处理。
   - 其他成员建议致电银行并将其报告为欺诈，而原用户表示他们已经联系过银行，并正尝试联系 Manus 支持部门。
- **文本转视频工具需求？**: 一位用户询问关于文本转视频工具的信息。
   - 未收到任何建议。
- **用户寻求在 X 上进行网络爬虫的技巧**: 一位用户询问在无需 API 的情况下在 **Twitter/X** 上进行网络爬虫的方法，并提到他们目前正在使用一个带有 cookies 进行身份验证的 Python 库，但发现维护起来很困难。
   - 讨论中未提供任何解决方案。
- **托管服务？**: 一位使用 *manus dev 创建了不错应用* 的用户正在寻求适合 **24/7 商业设置**且配置工作量最小的托管服务推荐。
   - Manus AI Chat 建议使用 **Vercel**。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1435043366839648377)** (8 条消息🔥): 

> `GPT-5 访问权限、Azure 额度过期、Aider 的未来、Perplexity API 密钥、模型测试` 


- **GPT-5 访问权限半价出售！**: 一位用户以 **Azure 额度**即将到期为由，提供 **50% 折扣**的 **GPT 模型**访问权限，包括 **GPT-5 Pro**。
   - 该优惠涵盖任何 **OpenAI 模型**。
- **Aider 创始人面临留存难题？**: 一位用户询问 Aider 创始人的未来计划，担心*目前的真空期正在导致用户流失*。
   - 该用户希望看到项目蓬勃发展，并在不破坏现有工作流的情况下整合一些**最新的策略和模型功能**。
- **Perplexity API 密钥令人困惑！**: 一位用户询问如何在 Aider 中将 **Perplexity** 用作 **API key 变量**，并指出他们是该工具的新手。
   - 另一位用户给出了建议。标准模式是需要将 API 密钥设置为环境变量，然后将其中一个 **perplexity 模型**设置为活动模型。
- **模型测试：哪些选项很重要？**: 一位用户询问在测试模型时应该禁用/启用哪些选项，并询问了关键因素。
   - 一位用户建议 *选择一个好的模型并保持默认设置，直到你有理由更改为止？*


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1434964274572300318)** (7 条消息): 

> `ollama_chat/gpt-oss:20b 推理力度 (reasoning effort)、aider 脚本功能、weak_model 标志` 


- **ollama_chat/gpt-oss:20b 不支持推理力度 (Reasoning Effort)**: 一位成员询问如何为 `ollama_chat/gpt-oss:20b` 模型设置 `/reasoning-effort`。
   - 然而，aider 发出警告称 `ollama_chat/gpt-oss:20b` 不支持 `reasoning_effort`，表明该特定模型不提供此功能。
- **Aider 脚本辅助多 Agent 工作流**: 一位成员询问如何实现两个 Agent 使用测试驱动开发 (TDD) 迭代改进代码的工作流。
   - 另一位成员指出 [aider 可以通过脚本实现](https://aider.chat/docs/scripting.html)此类工作流，其中 Agent 执行 prompts、审查改进并修复 bug。
- **Weak Model 标志解析失败**: 一位成员想知道如何设置 `--weak_model` 标志，但未能成功。
   - 尽管设置了 `/weak_model ollama_chat/qwen3:1.7b`，但并未达到预期效果。


  

---

### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1434942874947944608)** (13 messages🔥): 

> `M2 vs GLM-4.6，Minimax 成为首选 AI，Kimi 表情符号，Kimi iOS 应用` 


- **M2 在日常任务中表现优于 GLM-4.6**：一位用户在使用了 4-5 天后发现，**M2** 在大多数任务中超越了 **GLM-4.6**，尤其适合作为日常主力工具。
   - 虽然 **GLM** 在纯推理或编程方面表现出色，但 **M2** 避免了“视野狭窄”（tunnel-visioning）的问题。
- **Minimax 成为报告生成的顶级 AI**：一位用户表示，**Minimax** 已成为他们在网页调研和生成各种格式报告时的首选 AI，表现优于 **Qwen**、**Kimi** 和 **GLM**。
   - *它感觉像是第一个真正有用的、能实际干活的 AI*，比如查找图片和创建 PDF。
- **Kimi 获得搞怪新表情**：尽管万圣节已经结束，频道还是新增了两个 **Kimi** 表情符号：南瓜和德古拉。
   - 一位成员分享了 <:pumpkin:1435200414063525929><:dracula:1435200520750108783>。
- **Kimi App 已修复**：用户提到 **Kimi iOS app** 已经修复，并附上了应用的图片：[IMG_6502.png](https://cdn.discordapp.com/attachments/1371757564005711973/1435296101556158546/IMG_6502.png?ex=690b7329&is=690a21a9&hm=9bf62e2278b6a8653210095b0c1b3155c8fc5ccd8c0891a88be8b3a0d33334a0&)。
   - 另一位成员表示：*iOS 应用上的 Ok computer 非常棒*。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1434949557812465744)** (2 messages): 

> `HuggingFace，Qwen 模型在长尾事实中产生幻觉，IFEval` 


- **HF 模型：幻觉 vs 指令遵循**：对下载量最高的 [HuggingFace](https://huggingface.co/spaces/PropensityLabs/LLM-Propensity-Evals) 模型的评估揭示了有趣的倾向性行为，如指令遵循和事实幻觉。
   - 团队旨在从安全角度评估模型，因为这些模型通常未被深入审查。
- **Qwen 对罕见事实的编造**：**Qwen 模型**在罕见事实上的幻觉频率几乎是 **Llama** 对应模型的两倍。
   - 相比之下，**Qwen3 8b** 在指令遵循方面甚至超越了 **GPT OSS 20b**。
- **IFEval 评分受到质疑**：一位成员询问了评估中使用的具体 [IFEval](https://huggingface.co/spaces/PropensityLabs/LLM-Propensity-Evals) 分数。
   - 澄清点在于该分数是 Prompt 级别/指令级别，还是严格/宽松评分。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1435178396958064662)** (1 messages): 

> `倒计时任务，Adaptive Parallel Reasoning，lm-evaluation-harness PR` 


- **倒计时任务 PR 提交至 lm-evaluation-harness**：一位成员提交了一个 PR，将 **TinyZero** 和 **Adaptive Parallel Reasoning** 中出现的倒计时任务添加到 [lm-evaluation-harness 仓库](https://github.com/EleutherAI/lm-evaluation-harness/pull/3384)。
- **TinyZero 和 Adaptive Parallel Reasoning 启发新任务**：在 **TinyZero** 和 **Adaptive Parallel Reasoning** 中展示的倒计时任务是这个新 Pull Request 的基础。
   - 该任务旨在增强 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 的评估能力。


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1435129570339590184)** (2 messages): 

> `视觉语言模型 (VLMs)，Vision Transformers，位置编码` 


- **VLMs 架构分析**：**视觉语言模型 (VLMs)** 的架构涉及一个 Vision Transformer，它对图像进行分块（patching），将其转换为视觉 Token，并将其附加到 Prompt 的文本 Token 中，最后发送给 VLM 的大语言模型部分。
   - 有假设认为观察到的行为可能是由于数据集偏差造成的，但对字幕数据集创建过程的调查并未提供有力支持。
- **Vision Transformer Patch 排序研究**：Vision Transformer 以 Patch 形式工作，其排序来自于训练期间提供的位置嵌入（positional embeddings）。
   - 序列顺序可能受到位置编码的影响，因为大多数基础 Vision Transformer 使用 **RoPE** 进行位置编码。
- **提出位置编码实验**：建议进行一项实验，使用不同的位置编码从头开始训练，并观察在推理过程中排序是否保持不变。
   - 这有助于确定位置编码对 VLMs 行为的影响。


  

---

### **Windsurf ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1435336696106455212)** (1 条消息): 

> `Codemaps, SWE-1.5, Sonnet 4.5, AI code understanding, Scaling productive output` 


- **Windsurf 推出由 SWE-1.5 和 Sonnet 4.5 驱动的 Codemaps**: Windsurf 推出了由 **SWE-1.5** 和 **Sonnet 4.5** 驱动的 **Codemaps**，旨在增强代码理解能力，以扩展生产力输出。
   - 他们引用了 **Paul Graham (YC 创始人)** 的话：*"你的代码就是你对正在探索的问题的理解。因此，只有当你的脑海中有了代码，你才真正理解了这个问题。"* ([来源](https://x.com/windsurf/status/1985757575745593459))。
- **利用 AI 驱动的 Codemaps 对抗代码废料 (slop)**: Windsurf 将 **Codemaps** 定位为通过 AI 扩展理解力来对抗代码废料的解决方案。
   - 该公告强调，编程（无论是手动还是使用 Agent）的最大障碍是理解代码库。


  

---


---