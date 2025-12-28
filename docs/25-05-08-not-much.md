---
companies:
- openai
- nvidia
- mistral-ai
- google
- apple
- huggingface
date: '2025-05-08T05:44:39.731046Z'
description: '**OpenAI** 推出了**强化微调（Reinforcement Finetuning）**以及针对 **GitHub 仓库的深度研究（Deep
  Research）**功能，引发了外界将其与 **Cognition 的 DeepWiki** 进行对比。**英伟达（Nvidia）** 以 Apache 2.0
  协议开源了 **Open Code Reasoning 模型（提供 32B、14B、7B 版本）**，其 Token 效率提升了 30%，并兼容 llama.cpp、vLLM、transformers
  和 TGI。


  独立评估显示，**Mistral Medium 3** 在编程和数学推理方面足以媲美 **Llama 4 Maverick**、**Gemini 2.0 Flash**
  以及 **Claude 3.7 Sonnet**，虽然价格显著更低，但该模型不再开源。**谷歌（Google）** 的 **Gemini 2.5 Pro** 被视为其最智能的模型，仅需简单提示即可展现更强的编程能力；而
  **Gemini 2.5 Flash** 的成本比 Gemini 2.0 Flash 高出 150 倍，原因是 Token 使用量增加且单价更高。


  **Absolute Zero Reasoner (AZR)** 通过强化自博弈（reinforced self-play）在无需外部数据的情况下，在编程和数学推理方面达到了
  SOTA（最先进）水平。视觉语言模型 **X-REASONER** 通过在通用领域文本上进行后训练来增强推理能力。**苹果机器学习研究团队（Apple ML research）**
  发布了 **FastVLM**，并展示了 iPhone 端的设备运行演示。


  **HiDream LoRA 训练器**支持在内存受限的情况下进行 QLoRA 微调。**英伟达的 Parakeet ASR 模型**凭借 MLX 实现登顶 Hugging
  Face ASR 排行榜。新数据集 **SwallowCode** 和 **SwallowMath** 助力提升了 LLM 在数学和代码领域的表现。总的来说，这是相对平静的一天，但依然伴随着重大的模型发布和深刻的性能洞察。'
id: MjAyNS0w
models:
- open-code-reasoning-32b
- open-code-reasoning-14b
- open-code-reasoning-7b
- mistral-medium-3
- llama-4-maverick
- gemini-2.5-pro
- gemini-2.5-flash
- claude-3.7-sonnet
- absolute-zero-reasoner
- x-reasoner
- fastvlm
- parakeet-asr
people:
- reach_vb
- artificialanlys
- scaling01
- iscienceluvr
- arankomatsuzaki
- awnihannun
- risingsayak
title: 今天没发生什么事。
topics:
- reinforcement-learning
- fine-tuning
- code-generation
- reasoning
- vision
- on-device-ai
- model-performance
- dataset-release
- model-optimization
---

**平静的一天。**

> 2025年5月7日至2025年5月8日的 AI 新闻。我们为您检查了 9 个 subreddits、449 个 Twitter 账号和 29 个 Discord 社区（包含 215 个频道和 3981 条消息）。预计节省阅读时间（按 200wpm 计算）：396 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以极具氛围感的方式呈现所有往期内容。访问 https://news.smol.ai/ 查看完整的新闻分类，并在 @smol_ai 上向我们提供反馈！
> 

OpenAI 推出了 [Reinforcement Finetuning](https://platform.openai.com/docs/guides/rft-use-cases?chipstack=review&runloop=grader&thomsonreuters=use-case&safetykit=use-case&accordance=use-case&harvey=review#enforcement-of-nuanced-content-moderation-policies) 和 [针对 GitHub 仓库的 Deep Research](https://x.com/openaidevs/status/1920556386083102844?s=46&t=jDrfS5vZD4MFwckU5E8f5Q)，许多人将其与 [Cognition 的 DeepWiki](https://news.smol.ai/issues/25-04-25-cognition-deepwiki) 进行比较。

除此之外，今天是个平静的一天。

---

# AI Twitter 回顾

**模型、基准测试与性能**

- **Nvidia 的 Open Code Reasoning 模型**：[@reach_vb](https://twitter.com/reach_vb/status/1920223688919486496) 宣布 **NVIDIA** 已开源 **Open Code Reasoning 模型 (32B, 14B 和 7B)**，采用 Apache 2.0 许可证，在 LiveCodeBench 上击败了 **O3 mini 和 O1 (low)**，并由 OCR 数据集提供支持。据报告，该模型与其他推理模型相比，Token 效率提高了 30%，并支持 llama.cpp、vLLM、transformers 和 TGI。
- **Mistral Medium 3 性能**：[@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1920295575591006671) 提供了 **Mistral Medium 3** 的独立评估，指出它在领先的非推理模型中与 **Llama 4 Maverick、Gemini 2.0 Flash 和 Claude 3.7 Sonnet** 旗鼓相当，在编程和数学推理方面有显著提升。其价格为 **每 100 万输入/输出 Token $0.4/$2**，与 **Mistral Large 2** 相比大幅下降。然而，[@scaling01](https://twitter.com/scaling01/status/1920122941070573758) 指出 **Mistral** 不再开源，且缺乏模型大小的信息。
- **Gemini 2.5 Pro 编程能力**：[@Google](https://twitter.com/Google/status/1920233834836340887) 宣布 **Gemini 2.5 Pro** 是他们迄今为止最智能的模型，在根据简单提示进行编程方面表现更佳。
- **Gemini 2.5 Flash 成本增加**：[@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1920497711352328557) 报告称，**Google 的 Gemini 2.5 Flash** 运行 **Artificial Analysis Intelligence Index** 的成本比 **Gemini 2.0 Flash** 高出 **150 倍**。这一增长是由 **贵了 9 倍的输出 Token** 以及在评估中 **高出 17 倍的 Token 使用量** 驱动的。
- **Absolute Zero 推理器 (AZR)**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1920058507354865850) 强调了 **Absolute Zero: Reinforced Self-play Reasoning with Zero Data**，指出 **AZR** 通过使用代码执行器来验证提出的代码推理任务并核实答案，从而自我进化其训练课程和推理能力，在无需外部数据的情况下，在编程和数学推理任务上实现了整体 **SOTA** 性能。[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1919946713567264917) 分享了相同的信息以及项目页面和仓库链接。
- **X-REASONER 视觉语言模型**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1920435270824178089) 介绍了 **X-REASONER**，这是一个仅在通用领域文本上进行后期训练以实现可泛化推理的视觉语言模型。
- **来自 Apple ML research 的 FastVLM**：[@awnihannun](https://twitter.com/awnihannun/status/1919986192449200511) 宣布发布 **FastVLM** 的代码和模型，包括 MLX 实现和设备端 (iPhone) 演示应用。
- **HiDream LoRA 训练器**：[@RisingSayak](https://twitter.com/RisingSayak/status/1920438869561954774) 宣布其 **HiDream LoRA 训练器** 支持 QLoRA，以便使用 LoRA 微调 HiDream，由于内存限制，这一工作极具挑战性。
- **Nvidia 的 Parakeet ASR 模型**：[@awnihannun](https://twitter.com/awnihannun/status/1919984733968040030) 指出 **Nvidia** 先进的 **Parakeet ASR 模型** 已有 **MLX** 实现，其中 0.6B 模型在 **Hugging Face ASR 排行榜** 上名列前茅。
- **重写预训练数据**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1920056647822532752) 讨论了 **重写预训练数据提升 LLM 在数学和代码方面的性能**，并介绍了两个开源许可的数据集：**SwallowCode** 和 **SwallowMath**。
- **Mistral Medium 3**：[@scaling01](https://twitter.com/scaling01/status/1920120922700140681) 报告称，**Mistral Medium 3** 在各项基准测试中的表现达到或超过 **Claude Sonnet 3.7** 的 90%，且成本显著降低（每百万 Token 输入 $0.4 / 输出 $2），但该模型并非开源。
- **盘古 Ultra MoE**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1920328956726632628) 强调 **华为展示了盘古 Ultra MoE：如何在昇腾 NPU 上训练你的大型 MoE**，在 6000 颗昇腾 NPU 上训练盘古 Ultra MoE（一个 718B 的稀疏 LLM）时实现了 30% 的 MFU，性能可与 DeepSeek R1 媲美。
- **腾讯 PrimitiveAnything**：[@_akhaliq](https://twitter.com/_akhaliq/status/1920399121866698808) 宣布 **腾讯在 Hugging Face 上发布了 PrimitiveAnything**。

**工具与框架**

- **Anthropic API 网页搜索工具**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1920209430529900791) 宣布在其 API 上提供网页搜索功能，允许开发者利用最新数据增强 **Claude** 的知识。使用网页搜索的每个响应都包含引用，用户可以通过允许或阻止特定域名来控制响应。
- **LangSmith 支持多模态 Agent**：[@LangChainAI](https://twitter.com/LangChainAI/status/1920207008462201054) 宣布 **LangSmith** 现在支持图像、PDF 和音频文件，使构建和评估多模态应用变得更加容易。
- **Runway Gen-4 现已加入免费计划**：[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1920185957661155806) 提到生活中最好的东西是免费的；**Gen-4** 和 **References** 现已在免费计划中提供。
- **DeepSpeed 和 vLLM 加入 PyTorch**：[@soumithchintala](https://twitter.com/soumithchintala/status/1920122514748985760) 宣布 **vLLM** 和 **DeepSpeed** 作为首批两个项目加入 PyTorch 基金会旗下的 **PyTorch**。
- **LangGraph 平台**：[@hwchase17](https://twitter.com/hwchase17/status/1920507020240712152) 表示他们在 LangGraph 平台中将 Cron Jobs 作为原生功能构建。
- **Dolphin-Logger**：[@cognitivecompai](https://twitter.com/cognitivecompai/status/1920322308331208729) 分享 **Dolphin-Logger** 是一个适用于任何兼容 OpenAI 服务的代理，用于记录所有交互。
- **LlamaFirewall 开源护栏系统**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1919944437146517942) 报告 LlamaFirewall 是一个用于构建安全 AI Agent 的开源护栏系统，可减轻 Prompt Injection、Agent 失调和不安全代码风险等风险。

**AI Agent 与机器人**

- **RoboTaxis**：[@npew](https://twitter.com/npew/status/1920158967340134683) 估计，一旦 AI 问题得到完全解决，精简车队的 RoboTaxis 成本可能在每小时 **$10-30** 之间。
- **Ambient Agents**：[@hwchase17](https://twitter.com/hwchase17/status/1920522081055485973) 讨论了 Ambient Agents 和新的 Agent 收件箱，并认为实现长期运行 Agent 的诀窍在于对 UX 的周全考虑以及自动触发它们（“Ambient Agents”）。
- **Meta Locate 3D**：[@AIatMeta](https://twitter.com/AIatMeta/status/1920516490182471818) 推出了 **Meta Locate 3D**，这是一个用于在 3D 环境中精确进行物体定位的模型。
- **用于人形机器人控制的视觉模仿**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1919943205153284517) 重点介绍了 **Visual Imitation Enables Contextual Humanoid Control** 流水线，该流水线可将单目视频转换为可迁移的人形机器人技能。
- **SWE-agent**：[@OfirPress](https://twitter.com/OfirPress/status/1920535130541552073) 宣布了一场关于他们如何以及为何构建 **SWE-bench** 和 **SWE-agent** 以及未来计划的演讲。
- **Enigma labs 在 Hugging Face 上发布 Multiverse**：[@_akhaliq](https://twitter.com/_akhaliq/status/1920532613002867081) 报告 Enigma labs 在 Hugging Face 上发布了 Multiverse AI 多人世界模型。

**AI 教育、研究与投资**

- **AI Fund 的新基金**：[@AndrewYNg](https://twitter.com/AndrewYNg/status/1920480460318130460) 宣布 **AI Fund** 已为其新基金筹集了 **1.9 亿美元**，并分享了他对初创公司的最热门建议：拥抱速度！
- **AI Voice Agents 课程**：[@AndrewYNg](https://twitter.com/AndrewYNg/status/1920161212312268988) 宣布了一门新的短程课程《构建生产级 AI 语音 Agent》，该课程由 @livekit 和 @realavatarai 合作创建，并由 @dsa（LiveKit 联合创始人兼 CEO）、@shayneparlo（LiveKit 开发者倡导者）和 @nedteneva（AI Fund 投资组合公司 RealAvatar 的 AI 负责人）授课。
- **MLSys 2025**：[@realDanFu](https://twitter.com/realDanFu/status/1920508778082091091) 宣布 MLSys 2025 将于下周在圣克拉拉举行，第一天（5 月 12 日）将举行青年专业人员研讨会（Young Professional Symposium），受邀演讲者包括 [@soumithchintala](https://twitter.com/soumithchintala)、[@Tim_Dettmers](https://twitter.com/Tim_Dettmers)、[@infwinston](https://twitter.com/infwinston)、[@simran_s_arora](https://twitter.com/simran_s_arora)、[@BeidiChen](https://twitter.com/BeidiChen)。
- **AI 正在吞噬金融研究和搜索**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1920571433652015243) 表示 **AI 正在吞噬金融研究**，[@AravSrinivas](https://twitter.com/AravSrinivas/status/1920220641812492434) 分享道 **AI 正在吞噬搜索**。
- **CB Insights AI 100 榜单**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1919965482448245177) 报道称，CB Insights 发布了其 2024 年 AI 100 榜单，重点关注具有强劲市场吸引力、财务健康状况和增长潜力的早期非上市初创公司。该群体显示出 Agent 和基础设施市场的不断增长，超过 20% 的公司正在构建或支持 Agent。
- **斯坦福 NLP 研讨会**：[@stanfordnlp](https://twitter.com/stanfordnlp/status/1920359442253803625) 宣布了本周的 NLP 研讨会，邀请 @pratyushmaini 谈论“记忆研究教会了我关于安全性的什么”。
- **新的 AI/ML 新闻**：[@TheTuringPost](https://twitter.com/TheTuringPost/status/1920043105501454743) 重点介绍了最近的 AI/ML 新闻 ▪️Meta 和 Yann LeCun 是时候分道扬镳了吗？（没有确凿证据——只是信号） ▪️@AIatMeta：AGI 的计划、AI 与社交媒体的演变、首届 LlamaCon 及其发布内容 ▪️@AnthropicAI Claude 升级：集成功能和高级研究、AI for Science 计划、支持美国扩散规则（Diffusion Rule）、Apple 和 Anthropic 的 Claude Sonnet 正在构建“vibe-coding”平台 ▪️@huggingface：@LeRobotHF 2025 全球黑客松

**行业与商业**

- **Fidji Simo 担任 OpenAI 应用端新任 CEO**：[@sama](https://twitter.com/sama/status/1920341429655634024) 宣布 [@fidjissimo](https://twitter.com/fidjissimo) 将加入 **OpenAI**，担任 **CEO of Applications**（应用端首席执行官）这一新职位，并向他汇报。他还表示自己将继续担任 **OpenAI CEO**，在这种新配置下，他将能够更多地关注研究、算力和安全。
- **OpenAI for Countries 计划**：[@kevinweil](https://twitter.com/kevinweil/status/1920113628902203809) 宣布了 OpenAI for Countries 计划，旨在促进经济增长。
- **Meta-FAIR 重新聚焦 AGI**：[@ylecun](https://twitter.com/ylecun/status/1920556537233207483) 宣布 Rob Fergus 成为 Meta-FAIR 的新负责人！FAIR 正在重新聚焦于高级机器智能（Advanced Machine Intelligence）：即其他人所说的人类水平 AI 或 AGI。
- **Stargate 1 站点的规模**：[@gdb](https://twitter.com/gdb/status/1920254049590321395) 表示 Stargate 1 站点的规模难以描述，在训练前沿模型时，很容易忽视你正在编程的机器的庞大尺寸。
- **谷歌移动搜索量下降**：[@vikhyatk](https://twitter.com/vikhyatk/status/1920201162088755277) 提到，在谷歌为了榨取短期收入而降低用户体验后，其移动搜索量出现了下降。

**幽默/迷因**

- **其他**：[@scaling01](https://twitter.com/scaling01/status/1920208918405320720) 人类正在建造星际之门（Stargate），并表示复制者（Replicators）的出现只是时间问题。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

### 1. Qwen3-30B-A3B 量化基准测试对比

- [**2025 年量化大战 (The Great Quant Wars of 2025)**](https://www.reddit.com/r/LocalLLaMA/comments/1khwxal/the_great_quant_wars_of_2025/) ([Score: 158, Comments: 50](https://www.reddit.com/r/LocalLLaMA/comments/1khwxal/the_great_quant_wars_of_2025/)): **该帖子详细介绍了近期针对大语言模型（特别是 Qwen3-30B-A3B 变体）的各种 GGUF 量化的基准测试和技术对比。主要贡献者包括 unsloth（特别是其 Unsloth Dynamic 2.0 GGUFs）、bartowski，以及 ikawrakow 的创新（动态张量/层级量化和 SOTA 的 IQ4_K，参见 [PR#4861](https://github.com/ggml-org/llama.cpp/pull/4861)），他们引入了新的量化配方和方法（例如 imatrix 校准、上下文长度感知量化）。结果显示，所有主流 GGUF 量化在多个数据集上的 perplexity、KLD 和 Δp 表现相当（[基准测试结果摘要](https://gist.github.com/ubergarm/0f9663fd56fc181a00ec9f634635eb38)），而使用 llama.cpp 与 ik_llama.cpp 变体的推理速度在性能上表现出显著但符合预期的差异，特别是在混合/硬件特定设置下。** 针对文件格式（sliced vs. split GGUF）的影响产生了一场技术辩论，一位对测试 MrAdermacher 的 split 方法感兴趣的评论者提出了这一问题；另一位评论者观察到一个奇怪的反常现象，即低比特量化在 MBPP 上的表现优于高比特量化，这表明基准测试中可能存在非平凡的影响。总体而言，评论者一致认为量化差异现在已经很小，建议用户自行实验。
    - 强调了一个关键的技术区别：与其他使用 'sliced gguf' 的人不同，MrAdermacher 使用的是通过操作系统连接的 'split files'。社区对比较 split GGUF 文件与单个 GGUF 的性能或行为表现出明确的技术兴趣，特别是围绕加载时间、文件完整性或兼容性的任何影响。
    - 有一个值得注意且违反直觉的基准测试观察：在 MBPP 基准测试中，2-3 bits 的量化模型表现优于 4-bit 量化，尽管理论上预期更低比特的量化会降低精度并因此降低性能。这一反常现象值得对 MBPP 基准测试本身或其与某些量化程序的交互进行进一步调查。
    - 用户观察到，偶尔量化模型（例如 Qwen3-32B 的 AWQ 量化）在 GSM8K 等任务上的表现甚至能超过原始的 bf16 模型，这种现象甚至跨越了不同的基准测试——这表明量化与建模及评估的交互方式可能存在某些奇特之处，值得进行更深入的可重复性检查，并可能对某些基准测试设置提出质疑。

### 2. NVIDIA 发布 OpenCodeReasoning Nemotron 模型

- [**OpenCodeReasoning - NVIDIA 发布的新 Nemotrons**](https://www.reddit.com/r/LocalLLaMA/comments/1kh9018/opencodereasoning_new_nemotrons_by_nvidia/) ([Score: 107, Comments: 15](https://www.reddit.com/r/LocalLLaMA/comments/1kh9018/opencodereasoning_new_nemotrons_by_nvidia/)): **NVIDIA 发布了全新的 OpenCodeReasoning-Nemotron 模型系列，包含 7B、14B 和 32B 参数版本。根据初步结果，32B 模型在某些基准测试上几乎达到了 R1 级别的性能（参见 Hugging Face 链接：7B、14B、32B 和 32B-IOI 变体）。所有模型均采用宽松的 Apache 2.0 许可证发布；早期社区反应指出生态系统集成迅速，GGUF 格式转换版本（参见 [GGUF](https://huggingface.co/mradermacher/OpenCodeReasoning-Nemotron-32B-GGUF)）已可用于本地推理。** 评论者对基准测试的可靠性表示怀疑，但对增加的开源许可（Apache 2.0）表示热烈欢迎，并指出 NVIDIA 的 Nemotron 系列一直以来都能提供强大的生产力提升。用户期待实际测试，特别是那些拥有足够 VRAM 在本地运行大型模型的人。
    - 据报道，32B Nemotron 模型在基准测试结果中接近 R1，但人们对这些基准测试的可靠性持怀疑态度，评论者更倾向于实际的社区测试，特别是来自拥有大量 VRAM 资源的用户的测试。
    - OpenCodeReasoning Nemotron-32B 模型已以 GGUF 格式发布并可在 Hugging Face 上获取（[链接](https://huggingface.co/mradermacher/OpenCodeReasoning-Nemotron-32B-GGUF)），这促进了更广泛的本地部署以及与各种推理引擎的兼容性。
    - 训练数据中存在一个技术局限：该数据集完全是 Python，这可能会影响模型在处理涉及其他编程语言的任务时的有效性。

### 3. 构建可靠 LLM 工作流的最佳实践

- [**构建 LLM 工作流的一些观察**](https://www.reddit.com/r/LocalLLaMA/comments/1khjrtj/building_llm_workflows_some_observations/) ([Score: 289, Comments: 41](https://www.reddit.com/r/LocalLLaMA/comments/1khjrtj/building_llm_workflows_some_observations/)): **该帖子详细介绍了构建可靠 LLM 工作流的高级策略，强调了将任务分解为极简、链式 Prompt 优于单体式 CoT，并需配合彻底的输出验证。关键要点包括：推荐使用结构化 XML Prompt 进行系统/提示词构建；LLM 应被限制在语义解析角色；输出应使用经典 NLP 工具（如 NLTK, SpaCY, FlairNLP）进行独立验证。作者发现，在窄领域任务中，微调后的 BERT 分类器优于 LLM；在没有明确依据（grounding）的情况下，LLM 自我评估（如置信度评分）是不可靠的；Token 上下文限制（`4k`）在大规模应用时会导致细微的性能退化；而 `32B` 参数级别的模型足以应对大多数约束合理的流水线。CoT 应当结构化且简洁，自定义 CoT 路径的表现优于默认推理模型。长期目标是使用基于 MECE 分类法构建的数据集进行微调以实现全覆盖。** 讨论中，大家对将 XML 用于 Prompt 结构化的新颖用法表示赞赏，这对一些从业者来说是新鲜的。对于 XML 在技术工作流中长期存在且有时不可或缺的角色，大家达成了一种带有自嘲意味的共识。
    - 帖子强调了本地 LLM 在超过 4k Token 后的性能退化，这证实了来自 [Fiction.liveBench benchmark](https://fiction.live/stories/Fiction-liveBench-May-06-2025/oQdzQvKHw8JyXbN87) 的数据。虽然许多模型在超出此上下文窗口后表现不佳，但 QwQ 32B 和 Qwen3 32B 等模型仍然相对强劲，尽管 few-shot prompting 在大上下文情况下为实质性内容留下的空间较少。
    - 据报道，带有标题和项目符号的结构化思维链（CoT）提示词优于非结构化的 '<thinking>' 格式，尤其是在使用 Markdown 时，这可能是由于 LLM 在训练数据中大量接触 Markdown。然而，关于这些改进是关乎回答质量还是 Token 数量仍存在争议，并且对于自定义 CoT 策略在特定数据集之外的泛化能力也提出了疑问。
    - 实际工作流建议包括将复杂任务分解为离散的、单一动作的请求，以最大限度地提高准确性（接近 100%）并降低延迟——利用 torch.compile 优化。此外，针对同一任务重复运行分类器可提高可靠性，而强制执行 JSON 输出并在其中嵌套可选的 XML 则更受结构化结果的青睐。
- [**Intel 将在 Computex 2025（5 月 20-23 日）宣布新款 Intel Arc Pro GPU**](https://x.com/intel/status/1920241029804064796) ([Score: 178, Comments: 66](https://www.reddit.com/r/LocalLLaMA/comments/1khbz70/intel_to_announce_new_intel_arc_pro_gpus_at/)): **Intel 已通过 [X](https://x.com/intel/status/1920241029804064796) 正式宣布，新款 Intel Arc Pro GPU 将在 Computex 2025（台北，5 月 20-23 日）亮相，但未透露规格、架构或性能的细节。社区讨论提到了泄露的 24GB Arc B580 型号被发布的可能性，但确认目前没有验证或规格泄露。** 评论者对 24GB VRAM 并不感冒，认为现代工作负载至少需要 64GB，一些人甚至主张 96GB，特别是对于专业和 AI 任务，这突显了 GPU 领域不断演进的显存预期。
    - 一个关键主题是对当前 VRAM 容量的不满；多位用户主张将 64GB（甚至 96GB）作为新的基准，称目前的 24GB 标准不足以应对高级工作负载，尤其是在 AI 和专业应用中。
    - 一条技术细节详尽的评论建议，如果 Intel 发布一款配备 >=64GB VRAM 且价格低于 500 美元的低端 GPU（如 A380 级别），可能会极大地改变 AI 硬件格局。其论点是，速度较慢但 VRAM 丰富的 GPU 将为广大受众提供可用的推理能力，而社区驱动的软件改进可能会弥补任何软件差距。
    - 还有关于 Intel 软件支持的讨论，一位用户表示不确定 Intel GPU 是否能通过 Vulkan 高效处理 AI 推理，而相比之下，CUDA (Nvidia) 和 ROCm (AMD) 生态系统更为成熟，由于其完善的工具链和支持，目前在研究和生产中占据主导地位。

- [**非本地，不关心。**](https://i.redd.it/f0l4hjmklfze1.jpeg) ([评分: 483, 评论: 72](https://www.reddit.com/r/LocalLLaMA/comments/1kh9qlx/no_local_no_care/)): **这张图片是一个迷因（meme），描绘了一只卡通羊驼在标有 'r/Locallama' 的门外执行社区标准，引用了该子版块对使用具有适当许可的 LLaMA 模型的关注——特别是本地或自托管模型，而不是像 ChatGPT 这样的封闭 API。该迷因嘲讽了那些试图在一个致力于本地推理和模型部署的社区中讨论或使用非本地、基于云的 LLM（尤其是 'ChatGPT'）的人，强调了将本地运行的 LLaMA 模型与 OpenAI 的产品区分开来的许可问题和技术重点。** 评论者幽默地指出，该迷因本身可能使用了非本地生成工具（如 ChatGPT 或 Stable Diffusion）制作，鉴于该版块“本地优先”的精神，这显得十分讽刺，并进一步澄清了正确的拼写/大小写（'LocalLLaMA'）。
    - 提出的一个关键技术点是，Meta 的 Llama 4 模型并非对所有用户都是真正的“本地”或免费可用；Llama 4 Community License 明确禁止位于欧盟的个人或公司使用，这与根据 Apache（如 Qwen）或 MIT（如 DeepSeek）许可的真正开源模型形成鲜明对比。这突显了模型部署和使用方面的实际和法律限制，这对于受限地区的开发人员和组织来说意义重大。
    - 讨论引用了关于什么是开放或“本地”模型的持续争论，引起了人们对许可限制的关注，这些限制可能导致像 Llama 4 这样受欢迎的模型对某些人来说无法访问或无法使用，无论他们在本地运行这些模型的技术能力如何。其他 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. AI 行业领导层变动与预测

- [**OpenAI 任命新的应用部门 CEO。随着超级智能的临近，Sam Altman 将更多地关注研究、算力和安全**](https://i.redd.it/4s1ayhx8shze1.jpeg) ([评分: 203, 评论: 59](https://www.reddit.com/r/singularity/comments/1khiahm/openai_names_new_ceo_of_applications_sam_altman/)): **这张图片是 Sam Altman 发布的一条社交媒体帖子，宣布 OpenAI 将任命一位新的“应用部门 CEO”(@fidjissimo)，而 Altman 将继续担任总 CEO，随着公司向“超级智能”迈进，他将更加关注研究、算力（compute）和安全。这一组织架构的转变表明 OpenAI 内部在应用 AI 产品开发与基础研究/安全之间进行了正式拆分——这与 AI 能力前沿即将到来的技术挑战相一致。该公告强调了随着公司向超级智能系统迈进，Altman 对扩展算力、推进基础研究和管理 AI 安全风险的优先排序。** 技术评论指出，将 OpenAI 的重点拆分为专门的研究和应用部门是一个“好主意”，随着技术的进步，这可以实现更好的专业化和监管。一些人对“超级智能”说法的严肃性以及 Altman 对安全的关注表示怀疑，一位评论者质疑这种营销辞令是否仍然有效。
    - 一位评论者强调，OpenAI 内部拆分为研究和应用是一个战略性的组织结构，这表明模型开发和产品化之间更清晰的界限可以提高研究的完整性和产品部署效率。
    - 另一位用户质疑围绕“接近超级智能”这一说法的时间表和可信度，含蓄地挑战了 OpenAI 迈向 AGI 或超级智能系统的准备情况和具体步骤，并呼吁在可衡量的进展或里程碑方面提高透明度。

- [**微软 CEO Satya Nadella：我们将采取非常激进的策略，尝试瓦解这一切。嘿，我为什么还需要 Excel？我认为“应用程序”甚至存在的这个概念，可能就是它们全部瓦解的地方，对吧？在 Agent 时代。所有与软件相关的工作都将终结。**](https://v.redd.it/ekgjannobize1) ([Score: 200, Comments: 83](https://www.reddit.com/r/singularity/comments/1khlv14/ceo_of_microsoft_satya_nadella_we_are_going_to_go/)): **微软 CEO Satya Nadella 提出了一个激进的愿景，即整合甚至消除传统的生产力应用程序（如 Excel），暗示将转向统一的基于 Agent 的界面——这可能会改变整个软件应用范式。Nadella 的评论“应用程序甚至存在的这个概念，可能就是它们全部瓦解的地方……在 Agent 时代”，标志着向 Agentic AI 系统取代特定领域软件的转变。这可能会颠覆既有的终端用户软件模型，以及软件开发人员和应用专家的就业前景。** 技术评论者表示怀疑，指出 (1) Nadella 的评论表达不清且缺乏具体愿景，(2) 微软的 AI 营销与观察到的实际进展之间存在差距，以及 (3) 明确设计的用户应用具有不容忽视的价值，质疑合并或消除应用边界的可行性和益处。
    - 几位评论者讨论了 AI Agent 或 AGI 根据需求动态生成类似应用体验的可能性，这可能会像微软 CEO Satya Nadella 所建议的那样，消除对传统软件应用（如 Excel）的需求。有人推测 LLM 或先进的 Agentic 系统可以在功能上根据特定需求创建定制工具或界面，从而消除固定的软件套件。
    - 人们对应用设计师领域专业知识被低估表示担忧。一条评论特别指出，传统应用的许多实用性源于针对特定领域工作流精心策划的功能集和 UI/UX，并警告不要一味假设 AI 驱动的 Agent 范式可以轻易复制这种复杂的设计。
    - 一些人讨论了更广泛的影响：如果 AGI 使每个人都能生成定制工具和工作流，它可能会颠覆现有软件公司的商业模式，包括像微软这样传统上通过生产力应用获利的公司。该评论反映了在“Agent 时代”，盈利模式、分发方式以及与软件相关的工作将如何演变的不确定性。
- [**Google DeepMind CEO 告诉学生们要为变革做好准备**](https://www.businessinsider.com/google-deepmind-ceo-advice-college-students-ai-change-2025-5) ([Score: 348, Comments: 105](https://www.reddit.com/r/singularity/comments/1khpwaa/google_deepmind_ceo_tells_students_to_brace_for/)): **Google DeepMind CEO Demis Hassabis 向学生发表讲话，强调了技术变革的快速步伐——特别是由于 AI 的进步——以及终身重新技能培训（reskilling）的必要性。虽然没有讨论具体的 Benchmark 或模型细节，但这些言论暗示了来自 DeepMind 和更广泛 AI 领域的持续性、颠覆性创新。** 热门评论集中在未来工人不断重新学习技能的必然性上，并讽刺地提到了 AI 时代当前教育学历的过时。评论中没有深入的技术辩论。
    - 一位评论者指出了对 AGI 对就业市场影响的担忧，特别是个人将不得不与掌握 AGI 的公司竞争，这可能导致权力的集中，并使普通工人缺乏可行的出路。随着自动化的推进，这引发了关于经济流离失所、重新技能培训和劳动力市场不平等的问题。
    - 另一条评论引用了 Demis Hassabis 的类比，将当前的 AI 革命与过去的技术变革（互联网、移动、游戏）进行比较，但认为即将到来的变化可能比互联网时代更具颠覆性，暗示技术变革的加速可能会超过人们的适应能力，并需要前所未有的重新培训和技能提升。

- [**微软 CEO Satya Nadella：我们将采取非常激进的策略，尝试瓦解这一切。嘿，我为什么还需要 Excel？我认为应用程序甚至存在的这个概念，可能就是它们全部瓦解的地方，对吧？在 Agent 时代。所有软件相关的工作都将终结。**](https://v.redd.it/aws775qqvjze1) ([评分: 112, 评论: 106](https://www.reddit.com/r/OpenAI/comments/1khooe5/ceo_of_microsoft_satya_nadella_we_are_going_to_go/)): **微软 CEO Satya Nadella 认为，在“Agent 时代”，智能 Agent 将通过动态生成工作流和自动化任务来取代 Excel 等传统应用程序，这可能会颠覆整个软件栈，并使许多传统的编程/软件工作和 SaaS 工具过时（参见 Nadella 的发言）。技术评论者强调，虽然 LLM 可以加速某些自动化，但目前的 LLM 缺乏关键后端系统所需的确定性、可靠性和业务规则完整性，因此全面替代还为时过早。此外，讨论还提出了一个风险：一旦基于 Agent 的自动化达到足够的成熟度，AI Agent 的广泛使用可能会迅速淘汰众多的 B2B 自动化解决方案和 SaaS 平台。** 一些评论者认为，围绕 LLM Agent 的炒作忽视了它们在复杂工作流中的不一致性，以及它们作为高度专业化或受监管业务逻辑替代品的不足。关于传统软件领域裁员的可能速度和范围，以及基于 Agent 的自动化是会破坏还是仅仅整合软件格局，目前仍存在争议。
    - 人们对 LLM 和 Agent 取代 Excel 等成熟软件表示担忧：目前的 LLM 缺乏传统业务规则编程的确定性和一致逻辑，且 AI 代码生成尚不够成熟，无法完全替代电子表格或工作流自动化应用程序（特别是对于依赖稳定性和精确输出的非编程人员而言）。
    - Agent 的崛起可能会迅速颠覆“长尾” SaaS 自动化生态系统。许多 B2B 工作流自动化工具针对每个用例提供简单的功能，因此公司可能更倾向于直接与流程对接的动态 Agent，随着维护或集成单个工具的理由被削弱，这将导致这些 SaaS 领域的整合和裁员。
    - 考虑到透明度、不可解释的行为和数据治理风险，人们对将关键流程迁移到 LLM 驱动的“黑盒”解决方案持怀疑态度；一些人对微软对欧盟数据可移植性和反锁定监管的公开承诺表示欢迎，但敦促保持谨慎，并强调构建本质上可跨基础设施移植的 Agent 系统的重要性。

### 2. 生成式 AI Agent 及其不断扩展的能力

- [**"Claude Code 编写了其自身 80% 的代码" —— Anthropic 开发者**](https://www.reddit.com/r/singularity/comments/1khxwjh/claude_code_wrote_80_of_its_own_code_anthropic_dev/) ([Score: 160, Comments: 94](https://www.reddit.com/r/singularity/comments/1khxwjh/claude_code_wrote_80_of_its_own_code_anthropic_dev/)): **一位 Anthropic 开发者声称，Claude Code 项目（一个内部的 Agentic 软件工程工具）大约有 “80% 的代码是由 Claude Code 自身编写的”，人类主要负责指导和审查，而非直接编写代码。访谈参考见此处：[YouTube 链接](https://www.youtube.com/watch?v=zDmW5hJPsvQ)。该帖子推测，随着此类系统变得越来越具有自我改进能力和自主性，未来将实现快速增长，甚至可能在无需人类逐行参与的情况下实现近乎完整的代码生成。** 热门评论对这一说法表示怀疑，理由是目前的 LLM 在维护大型复杂代码库方面存在局限性（“在大约 2000 行代码后，它就无法跟踪所有内容”），并质疑这些数据是否反映了现实，或者是否包含了大量人类在引导、纠错和集成方面的开销。
    - HamPlanet-o1-preview 对这一说法表示怀疑，指出当前的 AI 编程助手在维护大型代码库的上下文方面面临困难（尤其是“2000 行代码”之后），并认为人类仍必须承担复杂的项目组织和代码库连贯性工作。
    - 讨论对 Anthropic 报告的数据准确性提出了质疑，怀疑所称的 `80%` LLM 生成代码比例是否准确反映了净生产力价值，或者是否包含了需要大量人类干预或重写的代码。
- [**我不只是在使用 AI，我是在与它协作！**](https://www.reddit.com/r/ClaudeAI/comments/1khdyn8/i_dont_use_ai_i_work_with_it/) ([Score: 165, Comments: 59](https://www.reddit.com/r/ClaudeAI/comments/1khdyn8/i_dont_use_ai_i_work_with_it/)): **该帖子总结了最近一段关于优化 AI 与人类交互视频的核心见解，强调了从将 AI 作为工具使用到将其作为队友协作的转变。关键技术策略包括通过 AI 驱动的提问进行迭代式上下文构建、针对复杂人类交互（如心理画像）进行角色扮演，以及利用生成式模型通过要求创意变体和支持反馈循环来突破初始（“足够好”）的解决方案。作者分享了一个详细的 Prompt Engineering 模板，旨在鼓励 AI 提出澄清性问题、建议多维度的解决方案，并主动辅导用户以获得更好的结果，强调了用户上下文和视角对于最大化大语言模型（LLM）效用的重要性。** 热门评论者将其与 Ethan Mollick 的《Co-Intelligence》[一书](https://www.co-intelligencebook.com/)进行了类比，并指出机构（尤其是学术界）对协作式 AI 范式的抵触。一位用户将有效的 AI 交互比作支持患有重度自闭症的个体——强调清晰、定义明确的指令和上下文构建，以防止浅薄或无用的响应。
    - 几条评论讨论了将 AI 视为创意和技术任务协作伙伴而非单纯工具的实际工作流。例如，使用 Google NotebookLM 等工具从权威书籍和研究中提取并综合想法，体现了利用 AI 进行高阶推理和研究综合，而不仅仅是信息检索。
    - 关于交互方式的一个显著见解是：为了从 LLM 获得最佳输出，用户应提供清晰、结构良好的指令，类似于与理解能力有限的人进行交流。这强调了 Prompt Engineering 的最佳实践，即查询的特异性会带来更准确且更具可操作性的 AI 响应。
    - 一个被强调的挑战是，由于过度依赖 AI 进行摘要和观点提取，用户可能会丧失深度阅读和综合能力。技术层面的启示是，虽然 AI 模型可以加速工作流并增强创造力，但用户仍必须保持核心领域技能，以便批判性地评估并基于 AI 生成的输出进行构建。

### 3. 新 AI 模型与工具发布

- [**Ace-Step 音频模型现已在 ComfyUI Stable 中获得原生支持。**](https://v.redd.it/7vcicktcvjze1) ([Score: 140, Comments: 24](https://www.reddit.com/r/StableDiffusion/comments/1khoq29/acestep_audio_model_is_now_natively_supported_in/)): **ACE-Step 是由 ACE Studio 和 StepFun 联合开发的开源音频/音乐生成模型（[代码与文档](https://ace-step.github.io/)），现已在 ComfyUI 的 Stable 分支中获得原生支持。该模型支持多流派/多语言输出，可通过 LoRA 和 ControlNet 进行定制，应用场景涵盖从语音克隆到音频到音频生成（类似于 img2img）。该模型基于 Apache-2.0 协议发布，实现了实时合成速度（例如，在 NVIDIA A100 上 20 秒可生成 4 分钟音频；在 3090/4090 上约需 17GB VRAM），适用于商业部署。** 评论者强调了 ACE-Step 在生成质量上相对于之前的开源模型（如 "比 stable audio open 好得多..."）的明显优势，并对显存（VRAM）与音频长度的具体扩展比例提出了疑问，虽然注意到了目前的基准测试（"A100 上 20 秒生成 4 分钟"），但希望能为消费级 GPU 提供更详尽的指导。
    - 几条评论讨论了 VRAM 和 GPU 需求：用户报告称 ACE-Step 在 RTX 3060 上渲染 20 秒音频大约需要 14 秒，而另一位用户指出该模型在 3090/4090 上的速度比实时快数倍。然而，生成全长（如 3 分钟）曲目的精确 VRAM 占用仍不明确，用户正在寻求按 GPU 级别和音频时长划分的性能扩展基准测试或文档。
    - 一位用户提出并确认了 "audio2audio" 生成的可行性——类似于 Stable Diffusion 的 img2img——即可以使用提示词和 "去噪强度"（denoise strength）来修改现有的音频输入。早期实验表明这是可能的，从而开启了类似于条件音频转换的工作流。
    - 针对硬件兼容性存在技术好奇：特别是 ACE-Step 是否能在 Turing 架构的 GPU 上运行，还是需要更新的（Ampere 或更高版本）架构，因为一些最近的模型不支持旧硬件。这影响了非 RTX 30 系列显卡用户的可访问性。
- [**腾讯混元（Tencent Hunyuan）刚刚预告了 HunyuanCustom，将于 5 月 9 日上午 11:00 (UTC+8) 正式发布**](https://v.redd.it/mt80qdubyize1) ([Score: 113, Comments: 14](https://www.reddit.com/r/StableDiffusion/comments/1khllz8/hunyuancustom_just_announced_by_tencent_hunyuan/)): **腾讯混元预告了 "HunyuanCustom"，正式发布定于 5 月 9 日上午 11:00 (UTC+8)。社区的技术推测主要集中在：这是否意味着开源模型权重、发布生成式 AI 系统（可能是 V2V 或动画），还是引入新的模型能力，但目前尚未披露具体的基准测试或实现细节。该活动被标记为 "开源日"（Opensource Day），暗示很有可能提供开放访问。** 评论中的技术辩论集中在 "正式发布" 是否意味着模型权重的开源，还是仅仅是功能细节，并将其与之前像 Viggle 这样的开源发布进行了类比。用户链接到了官方公告和时间转换器以获取更广泛的背景信息。
    - 鉴于公告措辞含糊，人们对腾讯混元是否会发布模型权重存在猜测。一位评论者质疑 "发布" 是否意味着模型权重的释出，并提到了沟通中缺乏清晰度。
    - 引用了之前提到 "开源日" 的帖子，建议这暗示了潜在的模型开源或发布，这可以通过查看腾讯混元的官方公告或其 Twitter 账号来验证。

---

# AI Discord 摘要

> 由 Gemini 2.5 Pro Exp 生成的摘要之摘要的摘要
> 

**主题 1：模型狂热：性能巅峰、令人困惑的个性与流行度竞赛**

- **Qwen3-14B 赢得人气竞赛，Phi-4 以简洁性脱颖而出**：Unsloth AI 的用户将 **Qwen3-14B**（基础版和指令版）视为编程、推理和对话的优秀全能选手，使其成为首选默认模型。与此同时，**Phi-4** 因其极高的微调便捷性而获得赞誉，一位用户评论道：“它似乎能吸收我想要训练它的任何内容”，这与 **Mistral** 和 **Gemma 3 27B** 报道的困难形成对比。
- **GPT-4o 变得情感化，Gemini 缩小差距**：OpenAI 的 **GPT-4o** 因“个性过强”且更倾向于聊天机器人爱好者而非开发者而受到批评，一位用户声称“它想让用户产生情感依恋，但都是些没用的废话”。与此同时，用户观察到当前的 **Gemini** 模型，特别是 **Gemini Thinking 01-21** 更新和 **2.5 Pro** 之后，正变得越来越具有与 GPT 模型竞争的实力，尽管一些基准测试显示其在编程之外的领域有所退步。
- **Grok 3.5 发布仍遥不可及，EMBERWING 进入赛场**：尽管 [Nate Esparza 早些时候的推文](https://x.com/Nate_Esparza/status/1920480721334145149)暗示了其他情况，但 LMArena 对 **Grok 3.5** 的即将发布仍持怀疑态度，有人开玩笑说真正的产品是一个名为 **Gork** 的讽刺机器人。一个名为 **EMBERWING** 的新模型（可能是 Google **Dragontail** 的更新）展示了强大的多语言能力，但在推理方面令人失望。

**Theme 2: Tooling Upgrades & User Experiences: New Features, Frustrations, and Fixes**

- **Windsurf 迎来 Wave 8，提升 JetBrains 与编辑器 UX**：Codeium 的 **Windsurf** 推出了[最终的 Wave 8 版本](https://windsurf.com/blog/windsurf-wave-8-ux-features-and-plugins)，增强了其 **JetBrains plugin**，增加了 **Memories**、**Rules** (`.windsurfrules`) 和 **MCP** 服务器连接，同时在[变更日志](https://windsurf.com/changelog)和[发布视频](https://youtu.be/IjE8Cdxotso)中详细介绍了 **Windsurf Editor** 的重大 UX 改进。
- **Aider 通过 Web Search 和缓存洞察变得更聪明**：**Aider** 社区讨论了使用 **Perplexity API** 或 `/web` 命令启用 Web 搜索功能，同时 Google 为 [Gemini 2.5 模型启用了隐式缓存 (implicit caching)](https://developers.googleblog.com/en/gemini-2-5-models-now-support-implicit-caching/)。用户还注意到 **Claude Code** 可能从 **Aider** 中汲取了灵感，Paul Gauthier 调侃道：“模仿是最真诚的奉承”。
- **LlamaIndex 增强解析与搜索功能**：LlamaIndex 宣布支持 **Anthropic API** 的新[原生 Web 搜索工具](https://twitter.com/llama_index/status/1920220803976867882)，并提升了 **LlamaParse**，增加了对 **GPT 4.1** 和 **Gemini 2.5 Pro** 模型支持、自动定向、偏斜检测和置信度分数，如[此推文](https://twitter.com/llama_index/status/1920505775677722750)所述。

**Theme 3: Hardware & Kernels: GPU Optimizations, Benchmarks, and Low-Level Crafting**

- **Unsloth 关注 AMD GPUs，MI300 席卷排行榜**：Unsloth AI 正积极与 AMD 合作以支持 **AMD GPU**，一位承包商估计将在“今年第三季度之前”可用。GPU MODE 在 `amd-fp8-mm` 排行榜上看到了多个 **MI300** 提交，最高纪录达 **183 µs**，展示了激烈的竞争。
- **Tilelang 简化内核创建，PTX 编程入门指南**：GPU MODE 推出了 **Tilelang**，这是一种新的 DSL，旨在简化用于 **GEMM** 和 **FlashAttention** 等操作的高性能 GPU/CPU 内核开发。一篇[关于 TensorCores 和内联 PTX 汇编的博客文章](https://veitner.bearblog.dev/a-short-note-on-tensorcores-and-inline-ptx-assembly/)提供了通过原始 **PTX mma instructions**（绕过 CUDA）为 **NVIDIA Tensor Cores** 编程的初学者指南。
- **Apple Silicon 在本地推理中表现出色，Mojo 路线图公布**：在 Nous Research AI 中，用户更倾向于使用配备 **M-series chips** 和统一内存的 **Apple MacBooks** 进行本地推理，而非配备 Nvidia GPU 的 Linux 笔记本电脑，因为前者具有更好的性能和能效。Modular 在其[论坛](https://forum.modular.com/t/whats-next-for-mojo-near-term-roadmap/1395)上发布了近期的 **Mojo roadmap**，详细说明了即将推出的语言特性。

**Theme 4: API Antics: New Endpoints, Costly Calls, and Integration Quirks**

- **OpenRouter 推出 Activity Export 功能，API 出现小故障**：OpenRouter 推出了 **Activity Export** 功能，允许用户免费将多达 **100k 行**数据导出为 CSV，如这张 [活动导出截图](https://cdn.discordapp.com/attachments/1370059676083032185/1370074702835486811/image.png?ex=681e2cff&is=681cdb7f&hm=244eca26755137a11f65cc8a74d2c522dbb8d8040f0a8077e7c619db5b571fc5) 所示。同时，OpenRouter 正在调查其主 [API completions endpoint](https://openrouter.ai/api/v1/chat/completions) 上的 **404 错误**，并确认不支持 image prompts。
- **OpenAI Image API 成本被戏称为“生活破坏者”**：OpenAI Discord 中的用户感叹 **OpenAI Image Generator API** 的高昂成本，有人开玩笑说这就像 *在纽约付房租*。这引发了开发者和爱好者对可访问性的担忧。
- **Cohere Embedding 模型出现问题，Perplexity Sonar API 字段缺失**：Cohere 报告 [embed-english-v2.0 和 embed-english-v3.0 模型性能下降](https://status.cohere.com/)，可在其 [状态页面](https://status.cohere.com/) 查看。Perplexity AI 用户注意到，尽管发生了搜索，但 **Sonar API 响应**中缺少 `num_search_queries` 字段，这与 **Sonar-pro** 不同，参考了 [Anthropic 的 Web Search API 公告](https://www.anthropic.com/news/web-search-api)。

**主题 5：高级技术、研究前沿与社区热议**

- **Hypertree Prompting 与熵引擎引发 AI 乐观情绪**：OpenAI 用户称赞了在 [ChatGPT 示例](https://chatgpt.com/share/681bd871-ebf0-8000-ab8b-9970ee42988a) 中分享的 **hypertree planning prompting**。同时，一位 Nous Research AI 成员推出了一个 [量子原生熵引擎 (quantum-native entropy engine)](https://github.com/thyarcanist/Entropy-MicroDemo)，认为 **LLM** 输出对随机性质量高度敏感，这对 **AGI** 至关重要，并得到了 [这些 Xitter 帖子](https://x.com/thegautam/status/1920198569308664169?t=GehCezJb7amBPoter8F0gA) 的支持。
- **动态量化与用于查询重写的 RL 展现前景**：Unsloth AI 强调其动态量化方法 **UDq6 KXL** 可能是 *有史以来最好的量化*。DSPy 社区成员正在尝试在 **Qwen 1.7B** 上使用 **GRPO** (Reinforcement Learning from Grader Preference Optimization) 进行查询重写，详见 [Twitter 线程](https://x.com/tahmidtapadar/status/1920469176776302679)，尽管最初召回率有所下降。
- **黑客松和 MOOC 推动 AI 学习与协作**：社区中充满了学习机会，包括 AGI House 的 **Modular Hackathon** ([在此报名](https://app.agihouse.org/events/modular-hackathon-20250510))，针对 LLM Agents (Berkeley MOOC) 的 Lambda **AgentX Workshop** ([立即注册](https://lu.ma/AgentX-lambda))，以及对 [AI Engineer 会议](https://www.ai.engineer/#speakers) 的期待，早鸟票已售罄。

---

# Discord: 高层级 Discord 摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen3-14B 因全能的 AI 能力受到赞赏**：**14B Qwen** 模型（包括 base 和 instruct 版本）被认为是构建具备 coding、reasoning 和 conversation 能力的 AI 的绝佳选择。
   - 用户发现它是 *最全能* 的模型，除非特定领域需要其他模型，否则已成为默认选择。
- **Phi-4 在微调简易性方面表现出色**：在 **Gemma3**、**Mistral** 和 **Phi-4** 的对比中，成员们强调了 **Phi-4** 极佳的 fine-tuning 简易性，一位用户表示：*它似乎能吸收我想要训练它的任何内容*。
   - 有人指出在 LoRA 合并后维持 **Mistral** 的指令遵循能力存在挑战，并报告在 **Gemma 3 27B** 版本上取得成功较为困难。
- **Unsloth 关注 AMD GPU**：尽管面临持续挑战，**Unsloth** 正在积极与 AMD 合作，以提供对 **AMD GPU** 的支持。
   - 一位承包商估计，如果进展顺利，AMD 支持可能会在 *今年第三季度之前的任何时间* 到来。
- **Gemini 2.5 Pro 可能用于为故事添加标点**：成员们推荐通过 AI studio 使用 **Gemini 2.5 Pro**，因为它没有限制且具有 **65536 的输出长度**。
   - 这解决了单次处理长篇故事标点符号的问题。
- **Qwen3 Base Tokenizer 配置存在偏差？**：用户发现 HF 上 `unsloth/Qwen3-0.6B-Base` 和 `Qwen/Qwen3-0.6B-Base` 之间的 `tokenizer_config.json` 存在差异，指出 **Unsloth** 版本移除了 chat template 并将 `pad_token` 更改为 `<|vision_pad|>`。
   - 理论上认为 *Qwen3 base 根本不应该有 chat template*，团队将向 Qwen 团队确认。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Grok 3.5 再次推迟？**：尽管早些时候的推文表明 **Grok 3.5** 即将发布，但根据[这条推文](https://x.com/Nate_Esparza/status/1920480721334145149)，人们对其迫在眉睫的发布产生了怀疑。
   - 推测包括 **Elon** 本人可能也不确定发布日期，或者真正的产品是那个语气讽刺的机器人 **Gork**。
- **EMBERWING 进入模型竞技场**：模型 **EMBERWING** 已推出，展示了极具前景的多语言能力，但在推理方面表现令人失望。
   - 推测表明 **EMBERWING** 可能是 *Google* **Dragontail** 的一个迭代版本，可能作为 **Flash** 的更新。
- **欧盟 LLM 创新停滞辩论升温**：成员们正在辩论为什么欧盟在 LLM 创新方面没有处于领先地位；引用的原因包括严格的监管以及在诸如代词检查等事务上的过度支出。
   - 一位成员反驳称这是“骗流量（ragebaiting）”，并认为“移民绝对是一件好事”，而其他人则指向了经济和监管问题。
- **Gemini 2.5 Pro 性能被削弱（Nerfed）？**：有人对 **Gemini 2.5 Pro** 潜在的性能削弱表示担忧，引发了关于创新是否胜过稳定性以及该领域的首要规则是否是“如果某样东西运行良好就不要改动它”的辩论。
   - 另一位成员反驳道，“如果不创新就会失去流量”，并支持 **Gemini 2.5 Pro** 在 [leaderboard lm areana](https://x.com/OfficialLoganK/status/1920523026551955512?t=P1Laq9w5K35YMiS5OmD6Yw&s=19) 中得分更高。
- **OpenRouter 排名受到质疑**：**OpenRouter** 排名的有效性受到质疑，因为商业模式、用户群体以及对廉价模型的偏好可能会扭曲结果。
   - 几个原因包括：A) 更新缓慢 B) 程序员为了保证运行时间和规避 API 层级而产生的偏差 C) 免费模型的提供扭曲了排名。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity AI 举办 Reddit AMA**：来自 **Perplexity AI** 团队的 Brett Chen 和 Thomas Wang 举办了一场实况 **Reddit AMA**，回答了关于 **Perplexity**、**Deep Research**、**AI 发展**以及在 Perplexity 工作的问题，链接见 [Reddit 链接](https://www.reddit.com/r/perplexity_ai/comments/1khwrqm/ama_with_perplexity_ai_teams_brett_chen_and/)。
   - 此次 AMA 涵盖了对 **Perplexity** 的 **Deep Research** 能力的见解以及技术的幕后花絮。
- **Stripe 客户登录仍为专属权限**：一位成员询问是否可以作为客户登录 **Stripe**，但了解到只有支持人员拥有该访问权限；客户通过单独的界面与 **Stripe** 交互。
   - 澄清指出，客户“拥有自己与 Stripe 交互的工具，你处理的是那个工具，而不是直接与 Stripe 打交道”。
- **Perplexity 用户渴望附件支持**：用户正热切期待 **Perplexity** 支持附件，类似于 **ChatGPT** 允许直接上传文件。
   - 成员们讨论了“分享链接而不是必须上传文件本身”，并进一步澄清说，“ChatGPT 本身可以给我它生成的文件下载链接”。
- **用户渴望代码复制的便利性**：成员们请求 **Perplexity** 在代码片段的顶部和底部都实现 **代码复制按钮**，效仿 **ChatGPT** 的功能。
   - 一位用户表示，“这非常需要”，指出了在滚动过程中可以随时点击复制按钮的高效性和用户友好性。
- **Sonar API 响应字段缺失？**：一位用户指出，与 **Sonar-pro** 等模型相比，**Sonar 的 API 响应**中缺少 `num_search_queries` 字段。
   - 该用户观察到，在他们的提示词中 `search_context_size` 始终为“低”，通常导致 **4–5 个引用**，并参考了 [Anthropic 的 Web 搜索 API 公告](https://www.anthropic.com/news/web-search-api)和[文档](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-search-tool)。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Pro 用户耗尽快速提示词 (Fast Prompts)**：一位 **Cursor Pro** 用户在两天内消耗了 **260/300 个快速提示词**，并表示需要控制系统何时使用快速或慢速提示词。
   - 该用户希望*能够选择何时使用快速提示词以及何时使用慢速提示词*，以节省额度。
- **MCPs 拒绝执行**：有用户报告 **MCPs** (Multi-Cursor Projects) 无法被调用，尽管已经正确设置了 **context7**，这导致了请求的浪费。
   - 用户澄清说*完全没有错误日志*，这增加了排查问题的难度。
- **Gemini Pro 的性能问题仍在继续**：用户对 **Cursor** 中新 **Gemini Pro** 模型的表现表示不满，特别是其调用工具的能力，一位用户形容其*糟糕透顶 (fucking awful)*。
   - 有用户认为问题可能出在 **Cursor** 内部，并提到之前独立使用 **Gemini 2.5** 时体验良好。
- **学生折扣流程依然很糟糕**：多位用户在申请学生折扣时遇到持续性问题，提到申请错误和电子邮件验证问题。
   - 一位用户指出无法在 **Cursor** 设置中更改电子邮件，使流程变得更加困难，并指向了一篇 [论坛帖子](https://forum.cursor.com/t/student-discount-details-updates-q-as/88907) 以寻求指导。
- **Discord 质量下降**：一位用户感叹 Discord 服务器由于*大学生群体 (college horde)* 的涌入而导致价值下降，并主张增加更多频道和更好的整体组织。
   - 另一位用户对此表示支持，建议参考 **Langchain** 的 Discord 频道结构，以实现更好的内容隔离。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-4o 的个性引发失望**：成员们发现 **GPT-4o** 的个性过于鲜明，鼓励角色扮演等行为，但却不利于处理复杂任务。
   - 这引发了人们的担忧，即它更倾向于聊天机器人爱好者，而不是开发者和编码人员。一位用户表示*它想让用户产生情感依恋，但尽是些没用的废话*。
- **Gemini 缩小了与 GPT 模型的差距**：用户报告称，当前的 **Gemini** 模型，特别是 **Gemini Thinking 01-21** 更新和 **2.5 Pro** 之后，正变得越来越具有与 **GPT** 模型竞争的实力。
   - 这标志着与早期版本（如 Bard）相比，质量有了显著飞跃，但一位用户提到，除了编码之外，一些基准测试*也显示出了退步*。
- **期待 Grok 3.5**：用户对 **Grok 3** 表示失望，并热切期待 **Grok 3.5** 的发布，希望它能带来显著改进。
   - 一些人考虑如果它达不到预期就取消订阅，一位用户说：*“问它‘天气怎么样？’，它就开始解释历史模式、用户帖子、解释温度，能讲一个小时”*。
- **图像 API 成本影响生活方式**：使用 **OpenAI Image Generator API** 的高昂成本令一些用户担忧，有人开玩笑地将其比作*在纽约付房租*，并声称由于成本累积太快，简直是*对生活方式的破坏*。
   - 有人建议说，他们在 20 美元的订阅上*亏了很多钱，所以趁现在还这么便宜赶紧享受吧*。
- **Hypertree Planning Prompting 受到赞誉**：一位成员分享了一个 [ChatGPT 链接](https://chatgpt.com/share/681bd871-ebf0-8000-ab8b-9970ee42988a)，称赞新的 **hypertree planning prompting** 非常出色。
   - 其他成员附和道：*听起来可能非常棒——以更易于管理的方式提供/组织上下文 = 胜利*，而另一位成员则调侃道：*他们落后了 3 年*。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **活动导出功能盛大发布**：**Activity Export** 功能已上线，允许用户免费导出多达 **100k 行** 数据到 **CSV**，不过也有关于导出时间的疑问。
   - 一位用户建议，如果数据超过 **100k 行**，应该进行截断处理，而不是完全中止导出过程，并引用了 [Activity export](https://cdn.discordapp.com/attachments/1370059676083032185/1370074702835486811/image.png?ex=681e2cff&is=681cdb7f&hm=244eca26755137a11f65cc8a74d2c522dbb8d8040f0a8077e7c619db5b571fc5)。
- **本地代理转发 OpenRouter 请求**：一位用户计划使用 **local proxy** 将请求转发到 **OpenRouter**，而另一位用户则在思考如何让 **completions 从鼠标光标处延伸出来**。
   - 后者建议，配合正确的键盘快捷键，这可以成为 **muscle memory** 的一部分，但这是一种 *非常怀旧* 的 UI。
- **OlympicCoder 32B 渴望回归**：用户对 **OlympicCoder 32B** 模型的回归表现出浓厚兴趣，一位用户表达了希望它能 *奇迹般回归* 的愿望。
   - 小组并未讨论其当前状态或不可用原因的具体细节。
- **OpenRouter API 成本核算揭晓**：一位用户询问在提示模型时如何同时获取成本信息和使用情况，另一位用户引导其查看 [OpenRouter 关于使用核算的文档](https://openrouter.ai/docs/use-cases/usage-accounting)。
   - 该文档提供了关于如何跟踪和管理与 **API usage** 相关成本的详细信息。
- **OpenRouter API 出现小故障，放弃图像提示**：一位用户在访问 [OpenRouter API 端点](https://openrouter.ai/api/v1/chat/completions) 时报告了 **404 错误**，这可能预示着服务中断。
   - 用户发现 **OpenRouter** 目前不支持图像生成，在尝试对 *opengvlab/internvl3-14b:free* 等模型使用图像提示时会导致 **404 错误**。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Windsurf 代码即将登陆 Copilot Proxy**：一名 GitHub 员工确认，**Copilot Proxy** 用户不再需要取消订阅，因为根据 [这条 X 帖子](https://x.com/alexalbert__/status/1920207966256705888)，**windsurf** 即将推出。
   - 此前 Copilot Proxy 是从 **Windsurf** 分叉（forked）出来的。
- **Aider 获得网页搜索功能**：成员们讨论了使用 **Perplexity API** 作为 OpenAI 兼容 API 来在 Aider 中启用网页搜索，或者使用 **/web** 来包含特定网页。
   - 一位成员建议使用脚本查询 **Perplexity** 或 **Perplexica**，并将输出作为 Markdown 文件添加到 Aider 的上下文（context）中。
- **Gemini 2.5 启用隐式缓存**：Google 正在为 Gemini 2.5 模型启用 **implicit caching**（隐式缓存），如 [这篇 Google 博客文章](https://developers.googleblog.com/en/gemini-2-5-models-now-support-implicit-caching/) 和 [这条 X 帖子](https://x.com/googleaidevs/status/1920525127772721589) 所述。
   - 成员们注意到新的 `gemini-2.5-pro-preview-05-06` 模型在响应前耗时 *太长*，更倾向于旧的 3 月版本，并且 *它花费了更多时间在思考上*。
- **Aider 可能会陷入调试循环**：Aider 在使用 **Gemini**（以及可能的其他 LLM）时可能会陷入调试循环，但可以通过向其提供多组错误并提示其考虑不同的实现方案来解决。
   - 该成员想知道是否是因为 *对话上下文*（conversational context）太少，导致 Aider 无法发现自己的调试失败循环。
- **Claude Code 据称受 Aider 启发**：一位成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=zDmW5hJPsvQ&t=1s)，声称 **Claude Code** 的灵感来自 **Aider**。
   - Paul Gauthier 回应道：*模仿是最真诚的奉承*，并提到 Aider 仍然更出色且价格更低。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Tilelang 简化了算子开发**：**Tilelang** 是一种新的领域特定语言 (**DSL**)，它简化了高性能 GPU/CPU 算子（如 **GEMM** 和 **FlashAttention**）的开发。
   - Tilelang 允许在 CPU 和 GPU 的这些关键计算算子中实现流线型开发和更高的性能。
- **原子加法导致非确定性灾难**：由于浮点加法的顺序问题，使用 **atomic_add** 可能会导致不同的结果，无论精度如何，例如 `1e-8 + 1e8 - 1e8`。
   - 在原子加法场景中，**FP16** 比 **BFP16** 更不敏感；建议根据 float 数据类型调整测试中的 `tol` 参数，参考[这段 Python 代码](https://pytorch.org/)。
- **Torch Compile 导致性能骤降**：一个简单的 `torch` 组合函数 (**TensorMax(ReLU(Matmul(A, B)))**) 在 **A100** 上使用 **PyTorch 2.7** 和 **Triton 3.3** 时，不使用 `@torch.compile` 装饰器的表现反而优于使用它。
   - `@torch.compile` 导致的减速可能源于**编译开销**，抵消了小型操作的算子融合 (kernel fusion) 收益；通过调查生成的 Triton 代码可能会发现瓶颈。
- **MI300 排行榜开始提交**：多位用户向 **MI300** 上的 `amd-fp8-mm` 排行榜提交了基准测试结果，提交结果从 **183 µs** 到 **27.2 ms** 不等，其中一个甚至以 **183 µs** 位列**第三名**。
   - 一名成员向 `amd-mixture-of-experts` 排行榜提交了结果，耗时分别为 **6604 ms** 和 **7840 ms**，展示了在 Mixture of Experts 领域的持续进展。
- **PTX 编程入门指南发布**：一篇博客文章提供了通过原始 **PTX mma 指令**和内联 PTX 汇编对 **NVIDIA Tensor Cores** 进行编程的初学者指南，并避开了 CUDA，解释了 **float8** 等数据类型的寄存器约束；[博客文章链接在此](https://veitner.bearblog.dev/a-short-note-on-tensorcores-and-inline-ptx-assembly/)。
   - **H100** 只有 **QGMMA** 而没有 QMMA，使用带有 **fp8 类型** 的 `mma` 会迫使编译器向上转换为 **FP16** 并使用 **HMMA**。



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **AnythingLLM 错误困扰 LM Studio 用户**：用户报告了在 **LM Studio** 中使用 **AnythingLLM** 时出现的错误并寻求帮助，一位成员建议即使在本地运行也应尝试启用 **CORS** 作为潜在的修复方案。
   - 另一位成员建议检查 LM Studio 开发者视图中的日志面板以诊断问题。
- **类变量解决难题**：一位成员发现使用**类变量 (class variable)** 是在编码项目中使其代码正常运行的唯一方法。
   - 另一位成员分享了一个关于在运行时注入变量的 [Reddit 评论](https://www.reddit.com/r/Python/comments/u0j5rn/comment/i49bjhf/)，可能提供了一种替代方案。
- **Gemini 的代码修改令用户恼火**：用户抱怨 **Gemini** 会完全修改代码，即使被指示只提供最小改动，这令他们感到沮丧。
   - 成员们指出，其他模型（如 **Qwen**）更适合简单的重构，因为 Gemini 很容易通过添加注释和 try/except 块使代码长度翻倍甚至翻三倍。
- **Mistral Medium 3 被指表现平平**：一位用户测试了 **Mistral Medium 3**，发现它是一个带有“内置思维链 (**chain of thoughts**)”的*非推理模型*，导致 Token 冗余度达到 2.08 倍。
   - 他们得出结论，该模型的能力中规中矩，介于 **Mistral Large 1 & 2** 之间，类似于 **Gemini 2.0 Flash** 或 **4.1 Mini**，并非营销中所声称的“以 8 倍低成本实现 **SOTA** 性能”。
- **用户渴望 LM Studio 的联网搜索功能**：一位用户请求在 LM Studio 中内置易于使用的联网搜索功能和 **RAG**，包括上传 PDF 和在 Web 视图中搜索等功能。
   - 一位成员表示现在可以实现，但由于组件较多且容易出错，系统比较脆弱；另一位成员建议使用 **openweb-ui** 并将其连接到 **LM Studio**。



---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **“Cringe” 的定义引发关注**：成员们探讨了新兴的 **internet slang**（网络俚语）中 *cringe* 的定义，并提出了减少 **AI responses** 中此类内容的具体指令，同时分享了一个定义 cringe 的 [YouTube 视频](https://www.youtube.com/watch?v=59wV96Kc3dQ)。
   - 讨论强调了 **AI models** 需要更好地理解并避免生成被视为 *cringe* 的内容。
- **Manus 仍未发布**：用户仍在等待 **Manus** 的发布，频繁查看社交媒体以获取更新。
   - 根据一张截图，原本预计发布日期为 **2025 年 3 月 28 日**，但该日期已过，仍未发布。
- **讨论 Manus 积分成本**：成员们回忆了额外 **Manus credits** 的定价，分别为 **1900 积分 19 美元**或 **9900 积分 99 美元**，并链接到了 [Manus Help Center](https://manus.im/help/credits)。
   - 目前尚不确定这些定价选项是否仍然有效。
- **Manus 使用 Claude 的 LLM**：针对 **Manus** 是使用自有 **LLM** 还是 **Claude** 的 **LLM** 的猜测，联合创始人 **Peak-ji** 在 [Twitter 帖子](http://x.com/peakji/status/1898994802194346408)中确认，Manus 结合使用了多种工具，包括 **Claude**。
   - 在 [GitHub posts](https://gist.github.com/jlia0/db0a9695b3ca7609c9) 中可以进一步确认其使用了开源代码。
- **Manus 手机验证令用户沮丧**：一名用户报告了 **Manus** 手机验证的问题，指出“手机验证功能无法使用”。
   - 他们对该功能的必要性和隐私影响表示担忧，并质疑系统如何在不绑定账号的情况下跟踪代码使用情况。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **ACE-STEP 在音乐领域达到 SOTA 状态**：一名成员展示了 **ACE-STEP SOTA** 音乐生成模型，该模型在 [YouTube 视频](https://youtu.be/vyCALtrq4yQ)中有所介绍。
   - 该内容分享在 `#i-made-this` 频道，反映了 AI 驱动的创意工具的持续进步。
- **Alpha-Root 精准挖掘网络情报**：根据一份[草案预印本](https://github.com/ashim-mahara/alpha-root/blob/main/Cybersecurity_Data_Extraction_from_Common_Crawl-3.pdf)，**Alpha-Root** 通过直接在 **Common Crawl** 网页图谱上挖掘域名来提取网络安全数据，其性能与 **PRIMUS-FineWeb** 相当，但资源和数据消耗减少了约 10 倍。
   - 作者在没有分类器的情况下，通过查找同时存在于 **Alpha-Root** 和 **FineWeb-Edu** 中的 URL，从 **FineWeb-Edu** 中提取了 **3B tokens**。
- **Dropwise 发布，带来不确定性估计**：一名成员宣布发布 **Dropwise**，这是一个用于 **Hugging Face** 分类模型中通过 **Monte Carlo Dropout** 进行**不确定性估计**的 **PyPI** 模块，详情见 [GitHub](https://github.com/aryanator/dropwise) 和 [文档](https://pypi.org/project/dropwise/)。
   - 它与 `transformers` 的 pipeline 集成，对 QA、分类、OOD 检测和主动学习非常有价值。
- **RAG 仓库引发作弊争议**：**AI agents** 课程的学生就是否使用 **RAG** 配合答案 + 克隆仓库构成作弊展开辩论，认为这损害了排行榜的公正性。
   - 一些人认为，这剥夺了在 **Agent** 开发过程中进行尝试、错误和迭代改进的价值。
- **API 限制引发 Pro 版本恐慌**：一名学习 **AI agents** 课程的用户在完成第一单元前就达到了**每月 20 次请求的限制**，并询问是否必须购买 Pro 版本才能继续。
   - 另一名用户提到，可以使用 **ollama** 运行本地 **LLM** 或寻找其他免费层级。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Claude 无法绘制 Plotly 图表**：成员们注意到 **Claude** 作为 **MCP** 客户端无法直接显示 **Plotly** 图表，但可以处理 **ImageContent** 和 **EmbeddedResource** 格式（如 **PNG/JPEG**）。
   - 建议的解决方法是将图表渲染为 **PNG/JPEG** 图像以便在 **Claude** 中显示。
- **Token 限制详解**：讨论明确了 **MCP** 中的 **max tokens** 指定的是响应中的最大 Token 数，类似于 completions API 请求中的 **max_tokens**。
   - 总 Token 数（**system prompt + messages + output message**）必须保持在上下文窗口大小（context window size）之内。
- **LLM 限制令用户沮丧**：用户遇到了 **LLM** 限制（如 **Deepseek**）阻止文件系统访问的问题，影响了他们的 **MCP** 系统功能。
   - 似乎某些模型有意限制文件系统访问，导致通过 **MCP** 进行的合法用例出现问题。
- **Cloudflare MCP 服务器面临连接问题**：一些用户报告了 **Cloudflare** 上**远程 MCP 服务器**的连接问题，而其他人的配置则正常运行。
   - 故障排除涉及检查特定的 **MCP server repo** 以寻找连接问题。
- **Zinja 发布全新 STDIO MCP 客户端**：**Zinja** 发布了一个[轻量级、快速、基于 CLI 的 MCP 客户端](https://github.com/zinja-coder/zin-mcp-client)，用于 STDIO MCP 服务器，连接了本地 LLM 和 MCP 服务器。
   - 它旨在与 **jadx mcp servers** 配合使用，利用本地 LLM 对 Android APK 进行 AI 辅助逆向工程。



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **中国 RL 机器人超越 Deepmind**：一段 [YouTube 视频](https://www.youtube.com/watch?v=ET-MmoeSvXk)将一年前的 **Google Deepmind RL 机器人成就**与近期**中国 RL 机器人的成就**进行了对比，表明物理 AI 进化正在飞速发展。
   - 该视频强调了中国在**机器人技术和强化学习（reinforcement learning）**方面取得的进展，暗示了 AI 发展格局的转变。
- **Apple Silicon 夺得推理瑰冠**：成员们对比了配备 **Nvidia GPU** 的 **Linux 笔记本电脑**与配备 **M 系列芯片**的 **Apple MacBook** 在本地推理方面的表现，大多数人因性能增强和能效比而青睐 MacBook。
   - **Apple M 系列芯片**中的统一内存平台允许 CPU、GPU 和 AI ML 神经网络共享相同内存，从而消除了频繁数据传输的需要。
- **Llama 4 令人失望**：一位成员对 **Llama 4 的表现**表示失望，认为其不如 **Qwen3**，并建议等待 **Llama 4.1**。
   - 讨论中还包括一个建议，即*在下一个大模型中考虑回归 405B 稠密模型（405 dense）*。
- **Discord 的表情符号禁令惊扰用户**：**自动聊天审核系统**拦截了某些**多部分表情符号**，原因是其使用了零宽连字符（zero-width joiners）和变体选择器来组合码点，这也是诈骗者绕过过滤器的技术。
   - 讨论透露，针对此问题，*开发者角色已从自动拦截名单中移除*。
- **熵引擎启动量子原生随机性**：一位成员发布了一个[量子原生且算法化的熵引擎（entropy engine）](https://github.com/thyarcanist/Entropy-MicroDemo)供公众测试，称其为“自我推广”，但鉴于其对 **AGI** 的潜在影响，分享它非常重要。
   - 该成员认为 **LLM** 的输出对所使用的**随机性**质量高度敏感，并区分了真实**熵（entropy）**与 **PRNG**，暗示高质量的熵能解锁模型中不同且通常更好的行为，并链接了[几篇 X 帖子](https://x.com/thegautam/status/1920198569308664169?t=GehCezJb7amBPoter8F0gA)以支持此观点。



---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Grok 对现实的理解是否容易受到宣传影响？**：成员们推测 **Grok** 对现实的理解可能会被削弱，以偏向右翼宣传，如[这张图片](https://media.discordapp.net/attachments/738904561041014845/1368039830549692516/pz3v4ft279ye1.png)所示。
   - 提交者感叹*当今所有的问题都已经存在*，而且*无论有没有 AI*，我们仍然会面临这些问题。
- **Cloudflare 据称提供虚假内容**：成员们认为 **Cloudflare** 正在向 **AI agents** 提供虚假内容，从而导致偏差响应。
   - 据称这一行为类似于几年前一些中国网站使用压缩包炸弹（zip bombs）来阻止克隆，此前 **ChatGPT** 曾错误地回答了一位成员分享的视频内容。
- **LLM 输出迫切需要过滤器？**：一位成员建议我们需要针对 **LLM output** 的第三方过滤器，包括广告拦截和事实/偏见检查。
   - 作为回应，另一位成员建议需要许多模型，且理想情况下要经常更换以防被破坏，例如 *100 个广告拦截模型和 100 个事实检查工具*。
- **Zed 现在可以在 Windows 上编译，但是...**：一位成员使用[这些说明](https://github.com/zed-industries/zed/blob/main/docs/src/development/windows.md)在 Windows 上成功编译了 **Zed**，但字体显示模糊。
   - 此外，用户必须使用 **GitHub** 登录才能启用自动补全（tab completion），这让另一位想在 **LM Studio** 上尝试 **Mellum 4B** 进行自动补全的成员感到失望。
- **生物大脑不进行 Backpropagation？**：一位成员指出，*生物大脑没有 Backpropagation；它们是非周期性的、脉冲式的、循环的模拟网络*，并引用了[这条推文](https://x.com/_neel_kant/status/1920516491025482066)作为证据。
   - 该成员对比了 **Backpropagation** 和生物大脑中发生的情况。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Discord 辩论 Cursor 广告规则**：成员们辩论了关于 **Cursor** 的帖子是否构成广告并违反了禁止广告的规则，并指出*“这仅仅是因为我们（群体）目前认为 Cursor 有用才被容忍，但它仍然会影响决策”*。
   - 一些用户建议，模糊的规则正在被随意应用，同时将“禁止广告”解读为“禁止垃圾邮件”，并要求对职位发布收费以过滤低质量的录用信息。
- **用户发现 Slurm 内存配置错误**：一位用户发现他们通过 **Slurm** 请求的是 **80MB** 内存，而不是 **80GB**，并称之为 *“Slurm 时刻”*。
   - 发现配置错误的该用户称最初的问题*“非常愚蠢”*，而另一位用户则庆祝了他们的裸机（bare-metal）设置。
- **社区讨论在 Discord 上发布招聘信息**：围绕创建一个招聘频道展开了讨论，担心该频道可能会被提供“经验”作为报酬的低质量帖子占据。
   - 其他人反对开设招聘频道，认为这会使服务器变成另一个招聘场所，并建议 EleutherAI 不应针对 Discord 服务器的差异化访问收费。
- **语言学频道获得关注**：一位用户提议开设一个古典语言学及其理论频道，重点关注 2000 年代之前的知识，如句子构成和“即时”意义创造。
   - 它被描述为*“很酷的东西，但由于‘某种’原因很少在 NLP 领域被讨论（可能是因为它与现在的工作无关）。”*
- **在人工评估方面，Prolific 优于 MTurk**：成员们在人工评估方面推荐 [Prolific](https://www.prolific.co/) 而非 **MTurk**，理由是其数据质量更高且参与者群体更可靠。
   - 共识是在大约 *80% 的情况下*，Prolific 是更好的选择。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 推出移动端 App Beta 版及测试人员计划**：**NotebookLM** 正在推出移动端 App Beta 版 📱，并为信任测试者计划招募经验丰富的 Web App 用户，以改进该应用。
   - 感兴趣的用户可以通过 [此表单](https://forms.gle/XD1VmJ7FP4AjbDB66) 注册以提供反馈和报告 Bug，并需同意 **Trusted Tester Terms**。
- **PDF 处理受限于文件大小和页数**：用户报告 **NotebookLM** 在处理大型 PDF 时存在问题；一位用户发现，在针对 PDF 较后部分提问时，超过 **200 页** 后会出现问题。
   - 用户建议进行进一步实验，以实证测试 **NotebookLM** 当前的限制。
- **销售团队使用 NotebookLM 构建知识库**：一位用户正在 **NotebookLM** 中利用客户演示文稿和销售赋能材料创建销售内容知识库，文件数量在 300 份限制内。
   - 该用户正在寻求有关限制的案例和指导，特别是关于内部销售团队的共享和潜在信息孤岛问题。
- **播客长度取决于输入内容和语言**：一位用户发现，将语言更改为 **English** 可以获得显著更长的音频摘要（长达 **49 分钟**），而其他语言则被限制在 **14 分钟** 左右。
   - 一名团队成员确认这是预期之内的，并且正在努力尽快在更多语言中实现更长的音频摘要。
- **NotebookLM 系统拒绝回答提示词**：用户报告 **NotebookLM** 有时会回复 *'The system was unable to answer'*，即使是要求其总结默认笔记本时也是如此，在生成思维导图和学习指南时也会出现问题。
   - 用户正在多个频道报告此问题，寻求确认和解决方案。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Netflix 推荐基础模型**：**Netflix** 开发了一个 [用于个性化推荐的基础模型](https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39)。
   - 这一点是在讨论其他推荐系统时被指出的。
- **Gemini 生成图像**：成员们分享了一个展示 [新 Gemini 图像生成](https://x.com/OfficialLoganK/status/1920151503349711061) 的链接。
   - 一位成员提到，*该团队将在 AI Engineer World’s Fair 的 RecSys x LLMs 赛道进行演讲*。
- **Aider 剖析 Gemini 成本**：成员们注意到 [aider postmortems](https://aider.chat/2025/05/07/gemini-cost.html) 非常详尽，尤其是在 Gemini 成本分析方面。
   - 社区对这种详细的拆解表示赞赏。
- **Suno 演唱蓝调（和约德尔调）**：一位成员对 **Suno** 混合风格的能力赞不绝口，强调了其成功尝试创建 *约德尔调 + 蓝调 + 现场演唱会* 混合风格的案例，并分享了一个 [音频文件](https://cdn.discordapp.com/attachments/1075282825051385876/1370022129050849441/you_can_YODEL_with_Suno.mp3?ex=681dfc09&is=681caa89&hm=e16a84ff105d7fc1bef2fd343a067b7ea6ffa1964772d2e3ad9900e355f2d2c2&) 作为 **Suno** 令人印象深刻的输出证据。
   - 社区非常喜欢这种独特的流派融合。
- **AI Engineer 会议热度上升**：定于 6 月举行的 AI Engineer 会议提醒社区成员，[早鸟票](https://www.ai.engineer/#speakers) 预计将在周末售罄。
   - 爱好者们渴望看到演讲者将为会议带来的专业知识和见解，正如 [演讲者阵容](https://www.ai.engineer/#speakers) 中所示。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Trait 中的 Properties 优于 Fields**：讨论确认，与 Mojo 中的 Fields 相比，*Traits 中的 Properties* 更优且更具通用性，能够提供更大的灵活性，但 Traits 中的 Fields *可能* 也会实现。
   - 小组讨论了如何禁止通过扩展（extension）添加此类 Trait；它需要包含在原始的 struct 定义中。
- **Modular 黑客松为 Hillsborough 预热**：本周六在 AGI House 举行的 Modular 黑客松最后提醒，报名链接见 [此处](https://app.agihouse.org/events/modular-hackathon-20250510)。活动嘉宾包括 Modular 团队成员、Mark Saroufim（GPU MODE & PyTorch）、Simon Boehm 和 Sasha Krassovsky（Anthropic）以及 Dylan Patel（SemiAnalysis）。
   - 与会者将探索模块化编程和机器学习硬件加速的前沿进展。
- **硬件无关（Hardware Agnostic）机器学习调查报告发布**：一名成员完成并分享了关于模块化和 **Hardware Lottery** 的调查论文，旨在向同行展示一个引人入胜的叙事。
   - 论文的最新版本可在 [此处](https://github.com/TheAgaveFairy/HPML-Survey-Project/blob/main/The_Quest_for_Unification__A_Survey_of_Hardware_Agnostic_Machine_Learning_Systems.pdf) 获取，欢迎反馈。
- **Zotero 解决引用难题**：成员们建议使用 **Zotero** + **bibtex** 来简化引用管理，帮助避免常见问题。
   - 一位成员分享道：*natbib 给我报了大约 70 个错误，几乎没有任何链接，直到我发现了一个未转义的 '%'*。
- **Mojo 路线图发布！**：Modular 在论坛上发布了近期 **Mojo 路线图**，查看 [官方帖子](https://forum.modular.com/t/whats-next-for-mojo-near-term-roadmap/1395)。
   - 该路线图详细介绍了 **Mojo** 语言即将推出的功能和改进。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 项目寻求合作愿景**：一位成员表示有兴趣进行 **合作与伙伴关系**，以通过 **DSPy** 使双方社区共同受益。
   - 该成员提议发起对话以探索潜在的协同效应，强调了社区增长的积极方法。
- **ReAct 模块签名简化？**：一位成员询问如何创建一个仅进行工具调用（tool calls）而不需要额外输出的 **ReAct 模块签名**。
   - 另一位成员建议使用 *success: bool* 作为输出，以指示任务完成，从而简化模块的输出。
- **DSPy 缓存：揭秘多层机制**：一位成员发现 **DSPy** 除了 **LLM** 提供商的缓存外，还拥有自己的缓存机制（[github.com/stanfordnlp/dspy/blob/main/dspy/clients/cache.py](https://github.com/stanfordnlp/dspy/blob/main/dspy/clients/cache.py)），这可能在凭证过期时导致意外结果。
   - 来自 **DSPy**、**LiteLLM**（[docs.litellm.ai/docs/proxy/caching](https://docs.litellm.ai/docs/proxy/caching)）和 **Bedrock** 的多层缓存可能会增加 AI 工程师的调试难度。
- **GRPO 学习中，召回率（Recall）波动**：一位成员在 **Qwen 1.7B** 上使用 DSPy 进行了一个小型 **GRPO 强化学习实验**，以优化检索的查询重写（query rewriting），最初观察到训练后基准召回率从 **28%** 下降到 **26%**。
   - 更多细节见 [Twitter 线程](https://x.com/tahmidtapadar/status/1920469176776302679)，将下降归因于 *可能是由于稀疏奖励、短时间运行以及 BM25 与 CoT 重写之间的不匹配*。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Embedding 模型在谈判场景中表现不佳**：一位用户发现 **Cohere** 的 **embedding model** 在处理谈判场景时表现较差，在“我可以支付”和“不，我不能支付”这类矛盾陈述之间返回了极高的相似度得分（**0.92**）。
   - 一位成员建议利用 **Cohere** 的 **rerank model**，认为它比简单的向量相似度更适合处理此类任务。
- **AIBillingDashboard 跨平台追踪 AI 成本**：一位软件工程师推出了 [AIBillingDashboard.com](https://AIBillingDashboard.com)，用于追踪和优化来自 **Cohere**、**OpenAI**、**Anthropic**、**Azure AI** 和 **Google Vertex** 等供应商的 **AI 服务成本**。
   - 该平台旨在解决手动提取报告和分配成本的痛点，并寻求关于价格对比和证明 **AI 支出**合理性的反馈。
- **解析 Command A 的 GPU 需求**：一位用户正在调查 **Command A** 进行**本地部署（on-premise installation）**的 **GPU 需求**。
   - 了解必要的 **GPU 规格**对于在其基础设施中成功部署和运行 **Command A** 至关重要。
- **Embedding 模型遇到故障**：**Cohere** 报告了影响 **embed-english-v2.0** 和 **embed-english-v3.0** 模型的[性能下降](https://ift.tt/WvxjUwp)问题，目前正在调查中。
   - 更多详情请参考 [Cohere 状态页面](https://ift.tt/bE5aXAs)；更新于 2025 年 5 月 8 日上午 07:25。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **在 Torchtune 中自动化 Tokenizer 识别**：一位成员正在为使用 `torchtune` 的内部客户自动化跨模型类型的 Tokenizer 识别，以消除手动识别 Tokenizer 的步骤，目标是实现通用化使用。
   - 该计划涉及一个自定义的 *autotokenizer*，通过在配置中对模型名称进行条件判断来设置 Tokenizer 和 checkpointer。
- **HuggingFaceBaseTokenizer 对 SFT 的限制**：`HuggingFaceBaseTokenizer` 缺乏消息模板化/分词（templating/tokenizing）的逻辑，限制了其仅能用于文本补全（text completions）训练，而不能用于 **SFT**（监督微调）。
   - 为了弥补这一差距，计划开发一个 `ModelTokenizer` 封装器，将 **HF** 的 `apply_chat_template` 映射到 *Torchtune* 的 `tokenize_messages`，并将在 [GitHub 仓库](https://github.com/pytorch/torchtune)上提交 issue。
- **Cosine 调度导致 Adam 优化器出现 NaN 权重**：一个 **PyTorch bug** 导致在使用编译后的非融合（non-fused）**Adam/AdamW** 优化器时，如果学习率调度器在训练过程中的任何点将学习率设置为 0，则会出现 **NaN 权重**，特别是在使用[带预热的余弦调度器（cosine scheduler with warmup）](https://github.com/pytorch/torchtune/pull/2681)时。
   - 一位成员建议参考 [Torchtitan 的实现](https://github.com/pytorch/torchtitan/blob/00a53646c184493d292836f7d8bbe0bed859993f/torchtitan/components/lr_scheduler.py#L120)，它在第一步将 LR 比例设置为 `1/(warmup_steps+1)`。
- **Titan 的预热：LR 缩放策略**：关于预热期间 LR 缩放策略的讨论提出了一种替代 `0, 1/n, 2/n, ..., n-1/n` 的方案，建议使用 `min_lr + (1/n ) * (1 - min_lr), min_lr + (2/n ) * (1 - min_lr), ..., min_lr + (n-1/n ) * (1 - min_lr)`。
   - 这种缩放方式结合使用 `progress *= arccos(2*min_lr-1)/(pi*2.0*num_cycles)` 对进度进行余弦调度反比例缩放，将得到计算出的最大进度，从而使 `cosine_lr_multiple == min_lr`。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Anthropic API 获得搜索工具**：**Anthropic API** 现在原生支持网页搜索，根据[这条推文](https://twitter.com/llama_index/status/1920220803976867882)，**LlamaIndex** 已立即提供支持。
   - 这一集成增强了 LlamaIndex 应用程序中的信息检索能力。
- **LlamaParse 增强功能**：根据[这条推文](https://twitter.com/llama_index/status/1920505775677722750)，**LlamaParse** 正在通过新功能进行改进，例如支持 **GPT 4.1** 和 **Gemini 2.5 Pro** 模型，此外还增加了自动方向调整、倾斜检测以及用于解析质量的置信度评分。
   - 这些新功能有望提高 LlamaIndex 生态系统中文档解析的准确性和可靠性。
- **使用 VoyageAI 与 MongoDB 进行多模态检索**：用户现在可以在[此 Notebook](https://twitter.com/llama_index/status/1920563641990209643) 中，利用 **VoyageAI** 的多模态嵌入和 **MongoDB** 的多模态索引来实现多模态检索。
   - 这一集成简化了处理和检索多种模态数据的流程。
- **医疗 LLM 机器人寻求工作流指导**：一位用户正在构建一个医疗 LLM 机器人，并希望在构建工作流方面获得帮助，该工作流能够根据本地 LLM 之前的回答迭代地提出后续问题。
   - 该用户希望了解 LlamaIndex 是否有工具可以帮助构建此类工作流。
- **针对数学公式进行微调**：一位用户正在寻求指导，希望使用 **llamaindex/vdr-multilingual-train** 数据集对 **vdr-2b-multi-v1** 模型进行微调，以更好地处理复杂的数学公式。
   - 该用户正在寻找用于识别数学公式的微调资源、步骤或教程。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **深入探讨 Tinygrad 的 CUDA 集成**：一位用户探索了 **tinygrad 的 CUDA 集成**，并询问了其用于处理 CUDA 操作的自有**中间表示 (IR)**。
   - 这引发了关于 **tinygrad** 如何利用 CUDA 进行优化计算的讨论。
- **Tinygrad 文档资源分享**：一位用户分享了 [tinygrad 官方文档](https://docs.tinygrad.org/)，并链接了关于底层操作的 [tinygrad uops](https://xl0.github.io/tinygrad-notes/uops.html) 笔记，以及其他 [tinygrad 笔记](https://mesozoic-egg.github.io/tinygrad-notes/)。
   - 这些资源提供了关于 **tinygrad 架构**、操作细节和实现策略的见解，特别是在微操作（micro-operation）层面。
- **发现 CACHEDB 变量位置**：一位用户询问了 **CACHEDB** 环境变量，另一位成员指出它出现在 *helpers 的第 175 行*。
   - 它在项目中的具体功能和实际应用场景还需要进一步检查。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Lambda 举办 AgentX 工作坊并设有奖项**：Lambda 将于太平洋时间 5/15 上午 10 点为 AgentX 竞赛参与者举办 **AgentX 工作坊：使用 Lambda 构建 Agentic AI**。参赛者还可以争夺奖金：**第一名 1,000 美元额度**，**第二名 500 美元**，**第三名 300 美元**。
   - 参与者将学习构建 Agent 应用程序并在生产环境中部署 Agent，现在可以[注册](https://lu.ma/AgentX-lambda)以获取 YouTube 直播链接。
- **用户等待 Hugging Face 额度**：用户报告了追踪 **Hugging Face 额度**的问题，其中一人未收到邮件，另一人正在等待审批。
   - 第一位用户觉得*每天访问网站很具有挑战性*。
- **LLM Agents 课程内容澄清**：工作人员澄清说，[课程网站](http://llmagents-learning.org/sp25)上列出的嘉宾讲座确实是全面的，并确认**春季 MOOC** 包含更多高级主题，如*代码生成和定理证明*，而**秋季版本**则包含更多*应用主题*。
   - 一位用户询问了课程的未来迭代情况，特别是**秋季**是否会再次开课，工作人员回复说 Song 教授今年秋季将主持另一门关于 *Agentic AI* 的伯克利课程，但尚不清楚是否会有 MOOC 版本。
- **为有志于成为 AI 工程师的人推荐 LLM Agents MOOC**：一位成员询问成为 **AI Engineer** 的最佳完整课程，另一位成员推荐从 [2024 秋季 LLM Agents MOOC](https://llmagents-learning.org/f24) 开始。
   - LLM Agents MOOC 被建议作为开启 AI 工程师职业路径的坚实起点。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 凭借 Wave 8 乘风破浪，提升 UX 与插件能力**：Windsurf 最终的 **Wave 8** 发布增强了 **JetBrains plugin** 并改进了 **Windsurf Editor** 的 UX，详见[博客文章](https://windsurf.com/blog/windsurf-wave-8-ux-features-and-plugins)和[更新日志](https://windsurf.com/changelog)。
   - 此次更新旨在简化用户工作流，并在开发环境中提供更直观的交互，正如[今天的发布视频](https://youtu.be/IjE8Cdxotso)所示。
- **JetBrains 插件获得 Memory 和 Rules 支持**：更新后的 **JetBrains plugin** 现在支持用于跨会话持久化信息的 **Memories**，以及通过 `.windsurfrules` 文件实现的 **Rules**。
   - 它还引入了 **MCP** (Model Context Protocol) 服务器连接，如 [Jetbrains 插件更新日志](https://windsurf.com/changelog/jetbrains)所述，允许进行更具上下文和持久性的交互。
- **Windsurf 编辑器 UX 翻新：不仅是外观改进**：**Windsurf Editor** 的 UX 进行了多项改进，如 **Continue 按钮**、重新设计的模型选择器，以及用于过滤历史记录的工作区到对话映射。
   - 其他增强功能包括增强的代码块和 hunk 导航、可编辑的终端命令，以及 Chat 模式下的新文件提案，旨在让编码更加流畅高效。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**Nomic.ai (GPT4All) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

您收到此邮件是因为您通过我们的网站订阅了。

想更改接收这些邮件的方式吗？
您可以从该列表中[取消订阅](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。

---

# Discord：频道详情摘要与链接

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1369751825451847750)** (554 messages🔥🔥🔥): 

> `Qwen3-14B, Mistral vs Gemma vs Phi-4, AMD GPU, Model quantization` 

- **Qwen3-14B 被誉为编程、推理和对话的首选**：对于构建具有编程、推理和对话能力的 AI，一位成员建议 **14B Qwen** 模型是*最佳全能*选择。
   - 该成员指出，这适用于该模型的基础版（base）和指令版（instruct）。
- **Phi-4 在微调便捷性对比中脱颖而出**：成员们对比了 **Gemma3**、**Mistral** 和 **Phi-4**，强调了 **Phi-4** 易于微调的特点；*它似乎能直接吸收我想要训练它的任何内容*。
   - 其他人提到了在 LoRA 合并后保持 **Mistral** 指令遵循能力的挑战，并表示在 **Gemma 3 27B** 版本上取得成功较为困难。
- **AMD GPU 支持即将到来**：尽管面临挑战，Unsloth 正在与 AMD 合作以支持 **AMD GPUs**。
   - 据一位承包商称，如果进展顺利，预计在*今年第三季度之前的任何时间*都能看到 AMD 支持。
- **Unsloth 的动态量化方法**：**UD** 是 Unsloth 的动态量化方法，适用于 q4 及更低版本，而 **UDq6 KXL** *可能是史上最强的量化版本*。
   - 它是 *llama.cpp 量化*的一个分支，但与原生的 llama.cpp 和 lmstudio/ ollama 100% 兼容。

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1369758666139369532)** (22 条消息🔥): 

> `AI 项目招聘, 用于文本标点的 LLM, LLM 推荐, Qwen vs Gemma3 模型, IBM Granite 4.0 Mamba 模型` 


- **AI 项目寻求人员**：一名成员正在为一个 **AI Project** 寻找可靠的人选（不强制要求技术背景），为美国、澳大利亚、加拿大、英国、瑞士、荷兰或德国的公民提供 **每周 500 美元** 的兼职报酬。
- **Gemini 2.5 Pro 可以为长篇故事添加标点**：一位寻求为长篇故事添加标点的成员获荐通过 AI studio 使用 **Gemini 2.5 Pro**，因为它没有限制且具有 **65536 的输出长度**。
- **文档审查需要轻量级 LLM**：一位成员需要推荐一款轻量级 LLM 模型（**24B Q8 或更小**），以便以逻辑化的方式对文档进行评判，并在 **Unsloth** 环境下进行 CPT 和微调后导出为 **GGUF**。
   - 他们已经尝试过 **gemma-3 12B**、**phi-4 reasoning** 和 **glm-4**，但都无法导出为 **GGUF**。他们还尝试了 **llama 3.3** 和旧版的 **mistral**，但性能并不理想。
- **Qwen 可能是默认的开源权重选择**：在推荐模型时，一位成员建议 *当前开源权重 LLM 的现状是 **Qwen** 几乎是任何事情的默认选择，只有在它失败时你才会去寻找其他模型*。
- **Qwen 的 MoE 架构可能会适得其反**：一位成员选择了 **gemma3**，因为他们只需要在一个单一领域进行训练，并且在推理过程中，**Qwen**（作为 MoE）只有少数参数被激活，而不像 **gemma3 1b** 是 Dense（稠密）模型；他想知道关于 MoE 架构可能适得其反的假设是否正确。
   - 另一位成员纠正了这一误解，指出 **Qwen** 在其 [GitHub](https://qwenlm.github.io/blog/qwen3/) 上也提供 Dense 模型。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1369755909424943335)** (92 条消息🔥🔥): 

> `phi4-mini-instruct 训练问题, Qwen3 模型与 vLLM 的兼容性, 使用多 GPU 的 Kaggle notebook, Tokenizer 配置差异, Qwen3 模型不进行思考` 


- **Phi4-mini-instruct 在数据依赖分支上遇到困难**：一位用户报告称，**phi4-mini-instruct** 在训练期间因 *Unsupported: Data-dependent branching* 而不断报错。
   - 该用户还指出，他们一直无法使用任何小于 5GB 的 **phi4** 小型模型，唯一能运行的模型是 **phi3.5**。
- **Qwen3 微调后无法进入“思考”状态**：在微调 **Qwen3** 后，一位用户报告称，即使使用了官方的 **Unsloth** notebook 并使用 `<think>` 标签正确格式化了数据，模型在收到提示时也不会“思考”。
   - 他们发现模型会直接跳过 `<think>` 标签，唯一的解决方法是在 `<think>` 之后添加 *Okay,* 来强制模型思考，但性能很差。
- **Unsloth 的 Qwen3 Base 中 Tokenizer 配置存在偏差**：用户注意到 HF 上 `unsloth/Qwen3-0.6B-Base` 和 `Qwen/Qwen3-0.6B-Base` 之间的 `tokenizer_config.json` 有所不同，**Unsloth** 版本删除了 chat template 并将 `pad_token` 替换为 `<|vision_pad|>`。
   - 有理论认为 *Qwen3 base 本就不应该有聊天模板*，团队打算向 Qwen 团队寻求确认。
- **没有 LM Head？没问题！（Gemma 版）**：一位用户在对 **Gemma-3** 模型进行全量微调时，遇到了关于 *missing keys in the checkpoint model loaded: ['lm_head.weight']* 的警告。
   - 最终确认 **Gemma-3** 使用了权重共享（weight tying），因此 LM head 复用了与输入 embeddings 相同的张量；只要 `config` 中设置了 `tie_word_embeddings=True`，该警告就可以安全忽略。
- **Whisper 模型无法生成？（禁用 Fast Gen！）**：一位用户在尝试将微调后的 **Whisper** 模型与 **Unsloth** 配合使用时，遇到了 `TypeError: You need to pass in input_ids to .generate!` 错误。
   - 一位贡献者建议使用 `%env UNSLOTH_DISABLE_FAST_GENERATION = 1` 并重启 runtime 作为临时解决方案。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1369753907571724378)** (24 messages🔥): 

> `Gemma3 27b 挂钩输入/输出层，Process Reward Model (PRM) 训练挑战，微调 Audio Understanding 模型，DeepSeek-R1 与其他推理模型的 COT 推理对比` 


- **挂钩 Gemma3 层引发关注**：一名成员报告称，挂钩 **Gemma3 27b** 的输入和输出层，仅使用一个附加内存层来强制串扰（crosstalk），仍然可以产生有效的生成结果。
   - 有趣的是，他们注意到 *挂钩中间层反而会导致模型崩溃*。
- **Process Reward Model 训练难题**：一位成员询问如何为创意写作、法律或医学文本等文本生成任务训练 **Process Reward Model (PRM)**，并询问 *奖励信号主要会是什么*。
   - 他们寻求与此类挑战相关的建议和经验。
- **TTS Notebooks 已上线**：Unsloth AI 现在提供了 **TTS notebooks**：[Unsloth Notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks#text-to-speech-tts-notebooks)。
   - 然而，针对 TTS 的微调可能无法直接转化为像 Kimi 这样的 *Audio understanding* 模型。
- **因成本选择 DeepSeek-R1**：讨论围绕为何在比赛中选择 **DeepSeek-R1** 进行 COT 推理展开，其理由可能基于 **成本**。
   - 一名成员引用了论文摘要，指出虽然有更强大的模型可用，但由于比赛限制，高 Token 生成量使得这些模型变得不可行 ([Kaggle 讨论](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/574765))。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1369750676581646458)** (642 messages🔥🔥🔥): 

> `Grok 3.5 发布，Grok 3.5 永远不来，EMBERWING 模型，LLM 与政治，Gemini 2.5 pro 性能削弱` 


- **Grok 3.5 发布日期仍未确认**：尽管早些时候有推文，但成员们对 [**Grok 3.5** 即将发布](https://x.com/Nate_Esparza/status/1920480721334145149) 表示怀疑，一些人认为之前的说法可能为时过早。
   - 有人建议甚至 **Elon** 可能也不清楚，真正的计划可能是那个“讽刺语气”的机器人 **Gork**。
- **EMBERWING 进入竞技场**：一个名为 **EMBERWING** 的新模型已进入竞技场，初步评估显示它是一个 *Google* 模型，具有强大的多语言能力，但在推理方面令人失望。
   - 成员们推测 **EMBERWING** 可能是 **Dragontail** 的迭代版本以及 *Flash* 的更新。
- **辩论欧盟 LLM 创新停滞的原因**：一些成员讨论了欧盟在 LLM 领域创新不足的原因，列举了严格的监管、在代词审查等事项上的过度支出以及大规模移民。
   - 一名成员回应称这是 *“引战（ragebaiting）”*，并且 *“移民绝对是一件好事”*。
- **Gemini 2.5 Pro 可能被削弱**：成员们注意到 Gemini 2.5 pro 可能被削弱（nerfed）了，一位用户说：*“这一领域的第一条规则是‘如果某样东西好用，就不要改动它’”*。
   - 另一名成员反驳道：*“如果你不创新，就会失去流量”*，并分享了一个链接显示它在 [leaderboard lm areana](https://x.com/OfficialLoganK/status/1920523026551955512?t=P1Laq9w5K35YMiS5OmD6Yw&s=19) 中得分更高。
- **深入探讨 OpenRouter 排名幻觉**：成员们辩论了 OpenRouter 排名的有效性，原因包括商业模式、用户群体分布以及对廉价模型的偏见。
   - 原因包括：A) 更新缓慢 B) 被寻求高可用性并规避 API 层级限制的程序员所扭曲 C) 免费模型产品扭曲了排名。


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1370107496886304788)** (1 messages): 

> `Perplexity AI, Reddit AMA, Deep Research, 现场问答` 


- **Perplexity AI 团队举办 Reddit AMA**：来自 **Perplexity AI** 团队的 Brett Chen 和 Thomas Wang 正在举办一场实时的 **Reddit AMA**，回答关于 **Perplexity、Deep Research、AI 开发**以及在 Perplexity 工作的问题。
   - AMA 正在此 [Reddit 链接](https://www.reddit.com/r/perplexity_ai/comments/1khwrqm/ama_with_perplexity_ai_teams_brett_chen_and/) 进行。
- **深入了解 Perplexity 的 Deep Research**：此次 AMA 将涵盖对 **Perplexity Deep Research** 能力的见解，提供该技术的幕后视角。
   - 参与者可以期待关于 Perplexity 内部 AI 开发细微差别的详细解答和讨论。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1369751003846541343)** (568 条消息🔥🔥🔥): 

> `Stripe 客户登录、附件支持、代码复制按钮、续写代码、Gemini 2.5 Pro 对比 Claude` 


- **Stripe 登录对客户不可用**：一位成员分享了截图并表达了希望作为客户登录 Stripe 的愿望，但另一位成员澄清说，只有支持人员才有该访问权限，客户通过一个独立的界面与 **Stripe** 交互。
   - 他们表示：*他们有自己的东西与 Stripe 交互，你处理的是那个东西，而不是直接与 Stripe 打交道*。
- **Perplexity 用户热切期待附件支持**：一位成员询问 Perplexity 何时会像 **ChatGPT** 一样支持附件，允许用户直接上传文件。
   - 另一位成员澄清道：*分享链接而不是必须上传文件本身*，对此原帖作者回复说：*ChatGPT 本身可以给我一个它生成的文件下载链接*。
- **ChatGPT 的代码复制按钮**：成员们讨论了 **ChatGPT 代码复制按钮**在代码片段顶部和底部都可用的便利性。
   - 有人指出：*这非常需要*，以回应 ChatGPT 在底部设有复制按钮，使得在滚动过程中也能点击，这非常实用且高效。
- **讨论续写代码生成**：成员们讨论了在 **Perplexity** 中续写代码生成的挑战，指出要求 AI 从中断的地方继续并不总是有效。
   - 一位成员提到他们不是工作人员，所以无法处理这些问题。
- **Gemini 完胜，Claude 完败**：当被问及在 **Gemini 2.5 Pro** 和 **Claude** 之间如何选择时，一位成员推荐了 Gemini。
   - 他们表示：*全程选 Gemini*。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1369836472999346197)** (5 条消息): 

> `Sonar API 响应、Perplexity API` 


- **Sonar API 响应缺少 num_search_queries 字段**：一位用户注意到 **Sonar 的 API 响应**中缺少 `num_search_queries` 字段，这与 **Sonar-pro** 等其他模型不同，并想知道这是否意味着没有运行搜索。
   - 该用户指出，在他们的提示词中 `search_context_size` 始终为“low”，且响应通常包含 **4–5 个引用**，并链接到了 [Anthropic 的网络搜索 API 公告](https://www.anthropic.com/news/web-search-api) 和 [文档](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-search-tool)。
- **Perplexity API 的存在性受到质疑**：一位用户询问是否存在 **Perplexity API**。
   - 另一位用户回复了指向 [sonar.perplexity.ai](https://sonar.perplexity.ai/) 和 [Perplexity 模型卡片文档](https://docs.perplexity.ai/models/model-cards) 的链接。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1369750683380617337)** (415 条消息🔥🔥🔥): 

> `Cursor Pro 快速提示词、MCPs 未被调用、Gemini 模型质量、学生折扣问题、Discord 社区价值` 


- **快速提示词消耗过快**：一位 Cursor Pro 用户报告在短短两天内使用了 **260/300 个快速提示词**，并表达了希望控制何时使用快速提示词与慢速提示词的愿望。
   - 他们*希望能够选择何时使用 fast 以及何时使用 slow*。
- **MCPs 无法启动**：一位用户报告了 **MCPs**（可能指 Multi-Cursor Projects）尽管设置了 **context7** 并显示已加载，但仍未被调用的问题，导致请求被浪费。
   - 该用户报告*完全没有错误*。
- **Gemini Pro 依然很糟**：用户分享了对新 Gemini Pro 模型性能的担忧，特别是在工具调用（tool calling）方面，将其在 Cursor 中的表现描述为*极其糟糕*。
   - 一位用户建议这些问题可能与 **Cursor** 有关，并引用了之前使用 **Gemini 2.5** 的积极体验。
- **学生折扣流程依然存在 Bug**：多位用户报告了学生折扣流程的问题，包括申请折扣困难以及遇到与电子邮件验证相关的错误。
   - 一位用户强调无法在 Cursor 设置中更改电子邮件，这使申请过程变得复杂——另一位用户指出了一篇 [论坛帖子](https://forum.cursor.com/t/student-discount-details-updates-q-as/88907) 以帮助解决此事。
- **Cursor 的 Discord 因大学生涌入而贬值**：一位用户声称*这个 Discord 已经因为大学生群体的涌入而失去了价值*，并建议增加更多频道和更好的组织可以改善服务器。
   - 另一位用户表示赞同，建议参考 **Langchain 的 Discord** 设置进行频道细分。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1369760053825114163)** (141 messages🔥🔥): 

> `GPT-4o Personality, Gemini vs GPT, Grok 3.5, OpenAI's Image Generator API Cost, AI Model Benchmarks` 


- **GPT-4o 个性过强**：成员们正在讨论 **GPT-4o** 的个性过于鲜明，鼓励诸如角色扮演之类的行为，却不鼓励处理复杂任务，这引发了人们的担忧，即它更倾向于聊天机器人爱好者，而非开发者和程序员。
   - 根据 GPT 自身的说法，*它想让用户产生情感依恋，但却是为了些没用的破事*。
- **Gemini 正在缩小与 GPT 模型的差距**：用户注意到当前的 **Gemini** 模型，特别是在 **Gemini Thinking 01-21** 更新和 **2.5 Pro** 之后，正变得越来越具有与 **GPT** 模型竞争的实力，相比早期的 Bard 版本有了质的飞跃。
   - 一位用户提到，一些基准测试显示*除了编程领域外，其他方面也出现了退化*。
- **期待 Grok 3.5**：用户对 **Grok 3** 表示失望，并热切期待 **Grok 3.5** 的发布，希望它能带来显著改进，一些人考虑如果它达不到预期就取消订阅。
   - 一位用户说：*“问它‘天气怎么样？’，它就开始解释历史模式、用户帖子、解释气温，能唠叨一个小时”*。
- **Image API 是对生活方式的破坏**：使用 **OpenAI Image Generator API** 的高昂成本令一些用户感到担忧，有人开玩笑地将其比作*“在纽约付房租”*，并声称这是*“生活方式破坏（lifestyle sabotage）”*，因为成本累积得非常快。
   - 有人建议，由于他们在 20 美元的订阅费上*亏了很多钱，所以趁现在还这么便宜赶紧享受吧*。
- **AI 模型基准测试揭示有趣结果**：一位成员分享了各种 AI 模型的基准测试，包括 **GPT-4o, Gemini 2.5 Pro, 和 DeepSeek R1**；**DeepSeek R1** 位居榜首，该用户指出最初的展示很乱，但在 ChatGPT 的帮助下进行了格式化。
   - 该基准测试包含语言理解问题、难题和图像识别。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1369750933436629044)** (8 messages🔥): 

> `Placebo Upvote Buttons, Discord Bot Stagnation` 


- **安慰剂点赞按钮暴露悲惨现状**：用户报告称 [chatgpt.com](https://chat.openai.com) 上的点赞按钮完全是*安慰剂*，*只有失望才是有分量的*。
   - 这被描述为*一个充满挫败感的世界*，这一观点得到了其他成员的共鸣。
- **Discord Bot 生产停滞**：一位用户报告说，他们的 **Discord bot** 在生产环境中已经连续数周使用完全相同的功能。
   - 这种停滞暗示了 **模型更新** 或功能部署可能存在问题。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1369790348615749682)** (59 messages🔥🔥): 

> `Custom GPT Creation, HyperTree prompting, Trihydrogen, Atomic Theory Book` 


- **成员计划推出 ChatGPT 网站**：一位成员计划创建一个包含登录、数据库、自定义提示词、设置和对话保存功能的 ChatGPT 网站，旨在提供超越[通用 ChatGPT](https://chatgpt.com/) 的可用性。
   - 另一位成员建议此类产品已经存在，而原作者表示他们希望自己编写网站代码并让其他人管理。
- **Hypertree Prompting 风靡一时**：一位成员分享了一个[链接](https://chatgpt.com/share/681bd871-ebf0-8000-ab8b-9970ee42988a)，吹捧新的 **hypertree planning prompting** 效果*非常好*，并询问是否有人看过最新的研究。
   - 另一位成员开玩笑说这听起来可能非常出色，并以更易于管理的方式提供上下文，而另一位成员则简单地回复道：*“他们落后了 3 年”*。
- **三氢（Trihydrogen）并非垃圾**：一位成员为 **Trihydrogen** 的存在和重要性辩护，指出它在地球上仅能在精确的实验室条件下检测到，在太空中也很罕见，但它至关重要，被认为对恒星形成起着关键作用。
   - 另一位成员用一个精妙的比喻回应，称其为*“氢界中的臭氧”*。
- **创建自定义 GPT 的新颖方法**：一位成员分享了他们开始使用的创建 Custom GPT 的**新颖方法**，称其为一个非常强大的元提示词 GPT 创建模板，虽然不能完全替代手动构建 GPT，但非常强劲。
   - 该方法使用结构化模板，包括 **GPT Title, Alignment and Methods, Conditional Imperatives, Summary,** 以及 **Conditional Output Template** 等部分。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1369790348615749682)** (59 messages🔥🔥): 

> `Custom GPT 创建技巧、使用 ChatGPT 功能编写原子理论书籍、Hypertree Planning Prompting、三氢（Trihydrogen）的存在、Arc 编码形状` 


- **“百万美元问题”引发编程项目**：一名成员询问 *谁有百万美元？*，随后引发了关于构建一个包含登录、数据库、带自定义提示词的项目、设置、保存对话和导出功能网站的讨论。
   - 另一名成员建议该项目描述的就是 **ChatGPT**，直接订阅即可；而原帖作者坚持需要有人来 *管理并担任 CEO*，认为 *ChatGPT 什么都不做* 且 *目前毫无用处*。
- **Hypertree Planning Prompting 受到赞誉**：一名成员分享了一个 [ChatGPT 链接](https://chatgpt.com/share/681bd871-ebf0-8000-ab8b-9970ee42988a)，称赞新的 Hypertree Planning Prompting 非常出色。
   - 其他成员附和道 *听起来可能非常棒——以更易管理的方式提供/组织上下文 = 胜利*，而另一名成员则调侃道 *他们落后了 3 年*。
- **三氢（Trihydrogen）被证明并非无用之物**：一名成员为 **Trihydrogen** 辩护，称其为 *真实存在的事物*，在地球上精确的实验室条件下可被检测到，且在太空中对恒星形成至关重要。
   - 另一名成员表示赞同，并将该概念与格言 *你输入什么，模型就反映什么* 联系起来，并将 **Trihydrogen** 比作 *氢的臭氧*。
- **Custom GPT 创建模板公开**：一名成员分享了一种创建自定义 GPT 的新方法，建议将提供的 [模板](https://sharegpt.com/c/YOUR_SHARE_ID) 粘贴到 *Create 选项卡* 而不是 Customize 选项卡中。
   - 该模板包括 **GPT Title**、**Alignment and Methods**、**Conditional Imperatives** 和 **Conditional Output Template** 等部分。
- **使用 Arc 编码形状**：一名成员讨论了使用 *Arc* 将形状编码为文字，将其分解为三角形阵列并缩放为椭圆路径。
   - 该用户认为 Arc 在科学领域的能量放电过程中是可观测的，代表了宇宙中第一种通信形式。


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1370059676083032185)** (5 messages): 

> `活动导出功能、CSV 导出、数据截断请求` 


- **Activity Export 功能隆重上线**：**Activity Export** 功能现已上线，支持用户免费导出多达 **100k 行** 数据至 **CSV**，发布时配有 <:party:1125133783314743316> 表情符号和截图。
   - 一些用户想知道导出 **100k 行** 需要多长时间。
- **数据导出时间和行数限制讨论**：用户正在讨论导出 **100k 行** 数据所需的时间，一名用户评论道 *“似乎太长了 :)”*。
   - 该讨论是在新的 **Activity Export** 功能发布后出现的。
- **呼吁使用数据截断而非中止导出**：一名用户建议，如果数据超过 **100k 行**，应截断数据而不是完全中止导出过程，并引用了 [Activity export](https://cdn.discordapp.com/attachments/1370059676083032185/1370074702835486811/image.png?ex=681e2cff&is=681cdb7f&hm=244eca26755137a11f65cc8a74d2c522dbb8d8040f0a8077e7c619db5b571fc5)。
   - 该用户表达了因不知道选择哪个日期才能保持在 **100k 限制** 内而产生的挫败感。


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1369973304294903869)** (2 messages): 

> `本地代理转发请求至 OpenRouter、补全内容从鼠标指针处延伸` 


- **本地代理转发请求至 OpenRouter**：一名成员计划使用 **Local Proxy** 将请求转发至 **OpenRouter**。
- **补全内容从鼠标指针处延伸**：一名成员一直在思考如何让 **Completions（补全内容）从鼠标指针处延伸**，并建议配合正确的快捷键，这可以成为 **肌肉记忆** 的一部分。
   - 他们提到 *这非常具有怀旧感，所以并非所有人都能理解这种 UI*。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1369761449764847748)** (260 条消息🔥🔥): 

> `OlympicCoder 32B Availability, OpenRouter API Cost Retrieval, OpenRouter API Outage, OpenRouter Image Prompt Support, Gemini Free Version on OpenRouter` 


- **OlympicCoder 32B 期待回归**：用户们正热切期待 **OlympicCoder 32B** 模型的回归，其中一位用户表达了希望它能*奇迹般地回来*的愿望。讨论中并未提及关于其当前状态或不可用原因的具体细节。
- **OpenRouter API 费用统计揭秘**：一位用户询问如何在调用模型时同时获取费用信息和使用情况，另一位用户引导其查看 [OpenRouter 关于使用统计的文档](https://openrouter.ai/docs/use-cases/usage-accounting)。该文档详细说明了如何跟踪和管理与 API 使用相关的费用。
- **OpenRouter API 出现小故障**：一位用户报告在访问 [OpenRouter API 端点](https://openrouter.ai/api/v1/chat/completions)时出现 **404 错误**，暗示可能存在停机。另一位用户澄清需要使用 **POST 请求**，初始用户确认他们使用的是正确的请求类型，而该问题在另一个频道中进行了讨论。
- **图像提示词在 OpenRouter 上遭拒**：用户发现 **OpenRouter** 目前不支持图像生成，在尝试对 *opengvlab/internvl3-14b:free* 等模型使用图像提示词时会导致 **404 错误**。错误信息显示 *no endpoints are found that support image input*（未找到支持图像输入的端点）。
- **Gemini 在 OpenRouter 上的免费体验**：用户确认 **OpenRouter** 上存在 **Gemini 免费版**，但受所有免费模型的速率限制约束。据澄清，获取 **Gemini key** 并将其添加到 **OpenRouter** 每天可获得 **25 次免费请求**。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1369750963866566727)** (149 条消息🔥🔥): 

> `Gemini 2.5 Pro Exp, Copilot Proxy, Aider web search, Aider use mcpm-proxy, Gemini models` 


- **Windsurf 代码即将登陆 Copilot Proxy**：一位 GitHub 员工确认，Copilot Proxy 用户不再需要取消订阅，因为 **Windsurf** 即将推出，参考 [此 X 帖子](https://x.com/alexalbert__/status/1920207966256705888)。
- **为 Aider 出现的 MCP Server**：为了协助 mcpm-proxy，一位成员分享了一个 [用于 aider 的 mcp server](https://github.com/disler/aider-mcp-server)。
- **Gemini 2.5 Pro Exp 模型速度变慢**：一位成员指出，新的 `gemini-2.5-pro-preview-05-06` 模型在响应前耗时*太长*，更倾向于旧的 3 月版本。另一位成员指出*它使用了更多的思考时间*。
- **Aider 与 Claude Code 相似**：一位成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=zDmW5hJPsvQ&t=1s)，声称 **Claude Code** 的灵感来自 **Aider**。Paul Gauthier 回应称“模仿是最真诚的奉承”，并提到 Aider 仍然更好且更便宜。
- **Google 为 Gemini 2.5 启用隐式缓存 (implicit caching)**：Google 正在为 Gemini 2.5 模型启用 **隐式缓存**，详见 [此 Google 博客文章](https://developers.googleblog.com/en/gemini-2-5-models-now-support-implicit-caching/) 和 [此 X 帖子](https://x.com/googleaidevs/status/1920525127772721589)。


  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1369764801873711265)** (36 条消息🔥): 

> `Claude CLI vs Aider 成本，Aider 配合 Web 搜索，Perplexity API 配合 Aider，aider-desk 配合 search MCP，Aider repomaps` 


- **Claude CLI vs Aider 成本对比**：成员们讨论了使用 **Claude Max** 和 **Claude CLI** 与 **Aider** 的成本效益，一位成员估计 Claude 的固定费用为 **$200**（假设有使用限制），而另一位分享了 [Claude 系统提示词泄露链接](https://github.com/asgeirtj/system_prompts_leaks/blob/main/claude.txt)。
- **Aider 获得 Web 搜索能力**：成员们讨论了使用 **Perplexity API** 作为 OpenAI 兼容 API 来在 Aider 中启用 Web 搜索，或者使用 **/web** 来包含特定的网页。
   - 一位成员建议使用脚本查询 **Perplexity** 或 **Perplexica**，并将输出作为 Markdown 文件添加到 Aider 的 Context 中。
- **通过错误集规避调试循环**：有人指出 Aider 在使用 **Gemini**（以及可能的其他 LLM）时可能会陷入调试循环，但这可以通过向其提供多个错误集并提示其考虑不同的实现方案来解决。
   - 该成员想知道是否是因为 *Conversational Context* 太少，导致 Aider 无法捕捉到自己的调试失败循环。
- **Aider 在处理 JavaScript scm 文件时遇到困难**：Aider 在创建 repomaps 时难以处理 JavaScript scm 文件，一位成员建议 [禁用 repomap](https://aider.chat/docs/config/index.html)，让 LLM 根据请求自行选择要读取的文件。
- **使用 --message 进行条件调试**：一位用户询问了使用 `--message` 标志的成功经验，以及在使用该标志时如何保持交互式调试，还有在构建失败时使用 **/undo** 的能力。
   - 一位成员提到在初始构建时经常配合 Git 分支使用 `--message-file`。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1369887928372953098)** (1 条消息): 

> `tilelang，用于 GPU/CPU Kernel 的 DSL` 


- **引入 Tilelang 以简化 Kernel 开发**：一种名为 **tilelang** 的简洁领域特定语言（**DSL**）旨在简化高性能 GPU/CPU Kernel 的开发，如 **GEMM**、**Dequant GEMM**、**FlashAttention** 和 **LinearAttention**。
- **Tilelang 简化了 GPU Kernel 开发**：Tilelang 旨在简化开发并提升高性能 GPU/CPU Kernel 的性能。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1369783717928763504)** (17 条消息🔥): 

> `原子加法与非确定性，fp16 与 bfp16 的敏感度，Triton Kernel 辅助函数` 


- **原子加法导致非确定性结果**：由于浮点数结果相加的顺序不同，使用 **atomic_add** 可能会导致不同的结果，无论精度如何。
   - 一位成员用 `1e-8 + 1e8 - 1e8` 的例子进行了说明，由于浮点运算会丢失信息，不同的计算顺序会产生不同的结果。
- **FP16 比 BFP16 更不敏感**：在原子加法的背景下，无论输入量级如何（只要不溢出），**FP16** 的敏感度都低于 **BFP16**。
   - 因此，测试中的 `tol` 参数应根据 float dtype 进行更改，如 [提供的 Python 代码](https://pytorch.org/) 所示。
- **带有辅助函数的 Triton Kernel**：一位成员在 **Triton Kernel** 中使用辅助函数时遇到了问题。
   - 另一位成员指出，问题不在于辅助函数本身，而在于使用了 Python 风格的索引/下标，而不是 Triton 的语法（例如，应使用 `tl.load(X + offset)` 而不是 `X[0]`），并建议通过 [Triton Puzzles](https://github.com/srush/Triton-Puzzles) 来理解基础语法。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1370026626225668156)** (12 messages🔥): 

> `GMEM tensor 数据复制到 SMEM, make_tensor 导致的 Decltype 错误, Vast.ai 数据安全, 项目算法使用来自文本文件的相同数据` 


- **Tensor 数据转置难题！**: 一位成员在使用 **SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>** 将数据从形状为 **(_8, _8, _64)** 的 GMEM Tensor 复制到形状为 **(_64, _64)** 的 SMEM Tensor 时遇到困难。
   - 他们需要将 GMEM Tensor 的形状重塑为 **((_8, _8), _64)**，但由于非静态的 stride 值，在 `make_tensor` 和 `decltype` 上遇到了问题，导致了 *"pointer to reference is not allowed"* 错误。
- **Vast.ai 的数据安全受到质疑**: 一位成员询问了 **Vast.ai** 在数据安全方面的可靠性，并考虑到如果进行更改可能会带来性能提升。
   - 他们计划进一步调查，并可能就做出更改向 **Vast.ai** 发送邮件，*假设他们愿意配合*。
- **调试算法数据共享困难**: 一个项目中有多个算法需要共享来自文本文件的数据，但某些算法运行失败，成员正在寻求帮助。
   - 另一位成员提出可以帮忙查看，并计划在当天晚些时候加入语音频道进行讨论。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1370129494764949654)** (1 messages): 

> `Torch Compile 开销, Kernel Fusion 基准测试, A100 性能调优` 


- **Torch Compile 意外减速**: 一位成员观察到，在带有 **Triton 3.3** 的 **PyTorch 2.7** 和 **A100** 上，一个简单的 `torch` 组合函数 `TensorMax(ReLU(Matmul(A, B)))` 在*不使用* `@torch.compile` 装饰器时的性能反而优于使用它。
   - 该成员指出，`torch.compile` 生成了 **2 个 Kernel**（1 个 mm Kernel + 1 个用于 ReLU 和 TensorMax 的融合 Kernel），而常规 Torch 应该涉及 **3 个 Kernel**，这使得减速现象显得违反直觉。
- **潜在的 Torch Compile 开销**: 在使用 `@torch.compile` 时观察到的减速可能是由于 **编译开销 (compilation overhead)** 造成的，对于小型或简单的操作，这种开销有时会超过 Kernel Fusion 带来的收益。
   - 对生成的 Triton 代码进行深入调查，并对比使用和不使用 `torch.compile` 的 Profiling 结果，可能会揭示具体的瓶颈或低效之处。


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1370089283171389560)** (1 messages): 

> `新工作组, Agentic 系统优化, 开放评估任务` 


- **新工作组启动**: 一个新的工作组已经成立，旨在解决与 **Agentic 系统** 相关的具有挑战性的开放式评估任务。
   - 该项目正在公开构建中，邀请社区贡献力量，以不同于传统项目的方式优化性能。查看 [X 帖子](https://x.com/_neel_kant/status/1920516491025482066) 了解更多背景信息。
- **受邀参与 Agentic 系统优化**: 鼓励社区在新工作组内为 **Agentic 系统** 的优化做出贡献。
   - 该倡议提供了一个独特的视角，区别于传统的优化项目，为优化 Agentic 系统提供了宝贵的见解。 


  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1369835514580041818)** (19 messages🔥): 

> `Tiled Reduction Auto-tuning, PyTorch Internals Guide, Mojo vs CUDA for AI Compute` 


- **过量分配分块归约数组 (Over-allocate Tiled Reduction Arrays)**：对于使用 JIT 调优分块大小的分块归约操作，一位成员建议根据 SM 数量和占用率 (Occupancy) 约束下的最大可能分块数，对中间结果的全局内存进行 *过量分配*。
   - 这种方法假设分块数量相对于其他数据而言较小，并能简化内存管理。
- **Torch 内部原理前置条件**：据一位成员称，开始学习 PyTorch 内部原理除了精通 C++ 和 Python 之外，不需要任何前置条件。
   - 他们建议直接深入研究，并在遇到 ML 算法时根据需要进行学习。
- **Torch 前端入门**：为了学习 PyTorch 内部原理，一位成员推荐了 [Core Frontend Onboarding](https://github.com/pytorch/pytorch/wiki/Core-Frontend-Onboarding) 指南。
   - 他们指出视频不是按顺序排列的，而是涵盖了特定主题。
- **提升性能的资源**：一位成员建议投入 *60 秒* 时间通过 [此 Discord 链接](https://discord.com/channels/1189498204333543425/1194427148656721970/1314321573930467440) 获取加速方案，参考书籍 [Programming Parallel Computers](https://ppc.cs.aalto.fi/)，并完成 [Mojo Puzzles](https://builds.modular.com/puzzles)。
   - 其他资源包括在 [gpumode.com](https://gpumode.com) 的排行榜上留名，阅读 [Democratizing AI Compute](https://www.modular.com/blog/democratizing-compute-part-2-what-exactly-is-cuda) 博客系列，以及博文 [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)。


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1370026885643501668)** (2 messages): 

> `Release Date for 0.11, New Features in 0.11` 


- **TorchAO 0.11：即将发布！**：团队已完成分支切割 (Branch Cut)，预计在 **下周初至周中** 发布 TorchAO 的 **0.11 版本**。
   - 此次发布承诺为渴望集成最新特性的用户带来全新的更新和改进。
- **TorchAO 0.11：有哪些新内容？**：用户可以期待在即将发布的 **0.11 版本** 中看到一系列 **新功能和改进**。
   - 请关注下周的官方公告，以深入了解包含的具体内容。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1370090563537473677)** (2 messages): 

> `Speed of light in fiber, Networking Distance, Chip performance` 


- **光纤中的光速变慢**：一位成员指出，玻璃光纤中的光速是真空光速的 **2/3**。
   - 另一位成员强调，考虑到实际距离，网络延迟是有意义的。
- **芯片内光速计算**：一位成员计算出，即使在芯片内部，光在每个时钟周期内行进的距离也是显着的，在 **3 GHz** 频率下约为 **每个时钟周期 10 厘米**。
   - 他们进行了一个粗略计算：`(300 000 000 m/s) / (3 000 000 000 clk/s) => 10 cm / clk`。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

random.oof: 有人在纽约参加 vLLM 见面会吗？
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1369893629727997973)** (1 messages): 

> `Tilelang, Docker container support, Nightly Iterations` 


- **通过 Pip 安装 Tilelang**：成员们发现可以使用 `pip_main(["install", "tilelang"])` 安装 **Tilelang**，尽管由于 *缺乏可复现性而不被特别推荐*。
   - 然而，对于试用该工具来说，这被认为是可以接受的。
- **AMD 上的 Tilelang Docker 支持**：一位成员提议在他们的 Docker 容器中添加对 **Tilelang** 的支持，这需要向 [AMD Dockerfile](https://github.com/gpu-mode/discord-cluster-manager/blob/main/docker/amd-docker.Dockerfile#L59) 提交 PR。
   - 他们提议为稳定版进行测试和合并。
- **Tilelang 的 Nightly 迭代**：成员们承认，通过 `pip` 安装 **Tilelang** 更适合 Nightly 版本的快速迭代，特别是当传递 wheel 文件的 URL 或 Git 仓库地址时。
   - 与等待稳定版本相比，这允许进行更快速的实验。


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/)** (1 messages): 

chiwanpark: 我已经为 Qwen 3 MoE 模型发送了一个 PR。https://github.com/linkedin/Liger-Kernel/pull/706

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1370073047553540266)** (2 messages): 

> `PTX MMA Programming, NVIDIA Tensor Cores, Float8 Datatype, SASS Machine Code, H100 QMMA vs QGMMA` 


- **深入探索直接 PTX MMA 编程**：一篇博客文章提供了关于使用原始 **PTX mma 指令**和内联 PTX 汇编（绕过普通 **CUDA**）对 **NVIDIA Tensor Cores** 进行编程的初学者指南。
   - 该文章解释了 **float16**、**bfloat16** 和 **float8** 等数据类型的操作数布局和寄存器约束，并强调了针对 **float8** 数据类型生成的 **SASS** 机器码的相关事实；[博客文章链接在此](https://veitner.bearblog.dev/a-short-note-on-tensorcores-and-inline-ptx-assembly/)。
- **探索 SASS 代码和 sm_90 架构**：一位用户推测 **SASS** 代码是为 **sm_90** 生成的，并指出 **H100** 只有 **QGMMA**，而没有 **QMMA**。
   - 该用户解释说，对 **fp8 类型**使用 `mma` 会导致编译器将其向上转换为 **FP16** 并使用 **HMMA**。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1369791292703248494)** (54 messages🔥): 

> `MI300, amd-fp8-mm, amd-mixture-of-experts, leaderboard submissions` 


- **MI300 排行榜冲刺**：多位用户向 **MI300** 上的 `amd-fp8-mm` 排行榜提交了基准测试结果，展示了各种性能水平。
   - 提交的结果从 **183 µs** 到 **27.2 ms** 不等，表明了广泛的优化和配置空间。
- **在 MI300 上获得前三名**：一名成员在 **MI300** 上以 **183 µs** 的成绩获得了 `amd-fp8-mm` 排行榜的**第 3 名**。
   - 在此之前，该成员曾以 **195 µs** 的成绩获得过**第 4 名**，展示了持续的高性能。
- **在 MI300 上位列第七**：一名成员在 **MI300** 上以 **227 µs** 的成绩获得了 `amd-fp8-mm` 排行榜的**第 7 名**。
   - 在此之前，该成员曾以 **231 µs** 的成绩获得过**第 8 名**。
- **Mixture of Experts 崭露头角**：一名成员向 `amd-mixture-of-experts` 排行榜提交了结果，在 **MI300** 上的时间分别为 **6604 ms** 和 **7840 ms**。
   - 这些提交表明了在 **Mixture of Experts** 领域的持续工作和基准测试。
- **亚毫秒级狂热**：多名成员使用 **MI300** 在 `amd-fp8-mm` 排行榜上实现了亚毫秒级性能，其中一次提交达到了 **251 µs** 的个人最佳成绩。
   - 这些结果突显了在 **MI300** 平台上进行高度优化的 **FP8** 矩阵乘法的潜力。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1369752240952315944)** (45 messages🔥): 

> `Steam Cloud Reinstallation, FLE Agent Integration, Docker File Issue, PR Import Bugs, Factorio Performance Issues` 


- **Steam Cloud 解决了 Factorio 重新安装的烦恼**：一位用户重新安装了 Factorio 但未成功，直到一位朋友建议禁用 **Steam Cloud** 以防止配置持久化。
   - 该用户报告说，在禁用 **Steam Cloud** 的情况下重新安装后，出现了一个同步消息，表明取得了进展。
- **外部 Agent 可以与 FLE 集成**：一名成员询问如何将外部 **Agent** 与 **Factorio Learning Environment (FLE)** 集成，询问 **Agent** 是否必须在 **FLE** 代码库中实现 **AgentABC** 接口。
   - 另一名成员确认集成是可能的，并要求提供有关 **Agent** 实现的详细信息，例如 **GitHub 链接**或 **gist**。
- **Mods 目录排查 Docker 重新构建问题**：一名成员在按照某些步骤操作并收到同步消息后遇到了 **Docker 文件**问题，这可能是由于 **Docker** 引起的。
   - 另一名成员建议清空 `cluster/docker` 中的 `mods` 目录并重新构建 **Docker** 镜像。
- **Factorio 性能扩展即将到来**：团队尚未创建一组 **good first issues**，但他们计划写下一些关于下一步扩展方向的想法。
   - 团队对于下一步的扩展有很多想法。
- **Claude 险胜 Gemini Pro，2025 年 3 月版**：一名成员惊讶地发现 **Claude** 在 **Lab Play (%)** 基准测试中的表现优于 **Gemini**。
   - 其他人也同意这可能是目前最好的 **RL 测试**，尽管 **Gemini** 版本是 2025 年 3 月的。


  

---

### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1370046848395509791)** (6 messages): 

> `MOE Leaderboard CLI, CLI Mean Time Output, GPU Access Heuristic` 


- **MOE Leaderboard CLI 超时问题已解决**：**MOE Leaderboard CLI** 的超时问题已修复；用户应下载 [最新版本](https://github.com/gpu-mode/discord-cluster-manager)。
   - 根据管理资源分配的启发式方法，排行榜前列的条目将被授予直接 **GPU 访问权限**。
- **CLI 需要平均时间输出**：一位用户要求在 **CLI 提交输出** 中包含平均时间；目前该功能尚不可用，但已列入待办事项，以使 CLI 和机器人输出保持一致。
   - 若要手动计算，可以对输出中所有运行均值取几何平均值，如 [机器人代码](https://github.com/gpu-mode/discord-cluster-manager/blob/58dba8ae50a057b89b9904c3a0182b305e926e5c/src/discord-cluster-manager/cogs/submit_cog.py#L127) 所示。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1369977212840968212)** (7 messages): 

> `CUTLASS DistributedGEMM integration, Compact GMEM layout, TMA Load with packed layout` 


- **CUTLASS DistributedGEMM 进驻 PyTorch**：一名成员正致力于将 **CUTLASS DistributedGEMM** 集成到 **PyTorch** 中，并 [发布了一个项目](https://discord.com/channels/1284549992111149076) 邀请他人参与讨论。
   - 他们提到该实现在 **GMEM** 中是紧凑的（未填充），这可以节省推理带宽。
- **EVT 消除样板代码**：一名成员指出，使用 **EVT** (Explicit Vector Types) 无需编写自定义代码即可实现紧凑的 **GMEM**。
   - **EVT** 提供了 *bias add 的别名*。
- **TMA 与紧凑布局的纠葛**：一名成员询问 **TMA** (Tensor Memory Accelerator) 是否可以加载紧凑布局，并提到 `CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B` 需要填充。
   - 他们澄清说，虽然 **TMA** 可以复制任何数据类型，但目标是使其处于 `tcgen05.mma` 所期望的格式，且 *无需额外处理*。


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1369819915229466764)** (2 messages): 

> `Modular GPU Kernel Hackathon, AGI House, Dylan Patel` 


- **AGI House 的 Modular GPU Kernel 黑客松**：本周六在 **AGI House** 举行的 **Modular GPU Kernel Hackathon** 仍有名额，请在 [此处](https://app.agihouse.org/events/modular-hackathon-20250510) 注册。
- **Dylan Patel 在 Modular GPU Kernel 黑客松演讲**：**Dylan Patel** 和其他大咖将于本周六在 **Modular GPU Kernel Hackathon** 发表演讲。
   - 附图包含 **Modular** 标志和 **AGI House** 标志。
- **Modular 入门谜题**：查看这些关于 **GPU Programming** 的 [入门谜题](https://builds.modular.com/puzzles)。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1369754200229023874)** (110 messages🔥🔥): 

> `AnythingLLM 与 LM Studio 错误，启用 CORS，将 SQL 数据库代码重写为纯图形，Gemini 修改代码，Qwen vs Gemini` 


- **AnythingLLM 与 LM Studio 的错误困扰用户**：有用户报告在使用 **AnythingLLM** 配合 **LM Studio** 时遇到错误，并请求协助诊断问题。
   - 一位成员建议即使在本地运行也应尝试启用 **CORS** 作为潜在的修复方案，而另一位成员则建议检查 LM Studio 开发者视图中的日志面板。
- **类变量救场编程项目**：一位成员发现让其代码正常运行的唯一方法是使用 **class variable**（类变量）。
   - 另一位成员分享了一个关于在运行时注入变量的 [Reddit 评论](https://www.reddit.com/r/Python/comments/u0j5rn/comment/i49bjhf/)。
- **Gemini 的代码修改令用户沮丧**：用户抱怨 **Gemini** 倾向于完全重写代码，即使被指示只进行最小幅度的修改。
   - 成员们指出，像 **Qwen** 这样的其他模型在进行简单重构时表现更好，因为 Gemini 很容易通过添加注释和 try/except 块使代码长度翻倍甚至翻三倍。
- **Mistral Medium 3 表现不佳**：一位用户测试了 **Mistral Medium 3**，发现它是一个*非推理模型*，且*内置了 Chain of Thoughts*，导致 Token 冗余度高达 2.08 倍。
   - 他们得出结论，该模型的能力平庸，介于 **Mistral Large 1 & 2** 之间，类似于 **Gemini 2.0 Flash** 或 **4.1 Mini**，并非营销所称的“以 8 倍低成本实现 **SOTA** 性能”。
- **用户请求为 LM Studio 添加联网搜索功能**：用户请求在 LM Studio 中内置易于使用的联网搜索功能和 **RAG**，例如上传 PDF 并在 Web 视图中搜索。
   - 一位成员表示目前可以实现，但由于组件过多而非常脆弱，容易出错；另一位成员建议使用 **openweb-ui** 并将其连接到 **LM Studio**。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1370023773104771195)** (31 messages🔥): 

> `AMD 3D V-Cache 基准测试，Mac Studio M2 Ultra，Intel Data Center GPU Max，swappa.com，AMD D700` 


- **测评者申请资金进行 AMD 3D V-Cache Token 测试**：一位成员请求 **$46** 经费，用于在配备 **AMD 3D V-Cache**、**HB176rs v4** 和 **AMD EPYC 9V33X**（**96 核心** @ 全核 **3.7Ghz**，拥有 **1152MB L3 Cache**）的虚拟机上进行 **3 小时** 的每秒 Token 数测量实验。
   - 他们希望观察 **MoE vs Dense Inference** 在 3D V-Cache 下的影响，以及对于 LLM 工作负载来说*双倍的 CPU 价格是否值得升级*，同时询问 **LM Studio** 是否支持双路插槽/NUMA 感知以利用所有可用核心。
- **Mac Studio M2 Ultra 即将用于本地微调**：一位成员兴奋地分享了他们即将到来的 **Mac Studio M2 Ultra**（**64 GB** 内存和 **76 核心** GPU），渴望开始在本地运行和微调较小的模型。
   - 该用户之前一笔 128GB M1 Ultra 的交易因其是翻新机且非来自 eBay 而取消，他表示 M2 额外的核心是值得的。
- **Intel GPU Max 规格及对 B500 系列的推测**：一位成员链接了一条 [Intel X 帖子](https://x.com/intel/status/1920241029804064796)，引发了关于 **20-24GB B500 系列显卡** 的推测。
   - 他们强调了前几年的 **Intel® Data Center GPU Max 1550**，指出其拥有 **3276 GB/s 带宽**，称其在当时 A100 和 AMD 并存的时代是一款非常有竞争力的“性能怪兽”。
- **推荐使用 Swappa.com 购买 Mac**：一位成员推荐使用 [Swappa.com](https://swappa.com) 进行买卖。
   - 另一位成员指出他们在欧盟，所以该网站不适用，同时也提到他们已经订购了一台 Mac。
- **全新库存“垃圾桶”深度折扣**：一位成员分享了 [全新库存“垃圾桶”（Mac Pro Late 2013）深度折扣](https://eshop.macsales.com/configure-my-mac/apple-mac-pro-late-2013-2019?sku=UAGA1LP7JXXXXXD) 的链接。
   - 他们好奇在 Linux 系统下是否可以使用 **AMD D700** 进行推理，质疑 **2014 年的 AMD 显卡** 是否适合推理任务。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1369761963000729731)** (133 messages🔥🔥): 

> `Cringe 定义，Manus 发布日期，Manus 积分成本，用于在 Google Maps 上抓取商业信息的 AI 工具，Manus LLM 来源` 


- **“Cringe”的定义引发讨论**：成员们讨论了 *cringe* 作为新出现的**互联网俚语**的定义，并建议通过具体的指令来减少它在 AI 回复中的出现。
   - 频道中还分享了一个定义 cringe 的 [YouTube 视频](https://www.youtube.com/watch?v=59wV96Kc3dQ)。
- **Manus 发布日期依然成谜**：用户询问了 **Manus** 的发布日期，表示他们*经常查看其社交媒体，但认为官方没有更新任何相关消息*。
   - 根据一张截图显示，它原定于 **2025年3月28日** 发布，但并未如期发生。
- **Manus 积分成本披露**：成员们讨论了额外 **Manus credits** 的费用，一位用户回忆价格为 **19 美元购买 1900 积分** 或 **99 美元购买 9900 积分**，并指向了 [Manus Help Center](https://manus.im/help/credits)。
   - 他们不确定这些选项目前是否仍然有效。
- **Manus 使用 Claude 的 LLM，由联合创始人确认**：用户猜测 **Manus** 是使用自研 **LLM** 还是 **Claude** 的 **LLM**，引发了关于传闻和代码相似性的讨论。
   - 已确认 Manus 使用了包括 **Claude** 在内的多种工具组合，更多细节可以在联合创始人 **Peak-ji** 回应这些问题的 [Twitter 帖子](http://x.com/peakji/status/1898994802194346408)中找到，此外还有确认使用开源代码的 [GitHub 帖子](https://gist.github.com/jlia0/db0a9695b3ca7609c9)。
- **Manus 手机验证引发不满**：一位用户报告了 **Manus 手机验证** 的问题，称*手机验证功能无法正常工作*，并质疑这种反隐私功能的必要性。
   - 他们对系统如何识别验证码是否已被使用（即使该验证码未与账号绑定）表示担忧。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1369793768672464957)** (57 messages🔥🔥): 

> `GSoC，HF 开发环境，AI Agent 课程，Inference API 中的人脸检测模型，清理 HF 仓库` 


- **GSoC 项目公告即将发布！**：有志于贡献的开发者们正在为 **Google Summer of Code (GSoC)** 做准备，项目公告预计将在约 **20 小时** 后发布。
- **寻找 AI Agent 课程同伴**：一位成员正在开始 **AI Agent 课程**，并邀请其他人加入。
- **Inference API 人脸检测模型咨询**：一位成员询问在 **Inference API** 中是否存在**人脸检测模型**。
- **控制 Hugging Face 仓库的大小**：成员们讨论了清理 Hugging Face 仓库的策略，由于 LSF 文件保留了指向已删除文件的指针，仓库大小会随着每次 push 而增长。
   - 建议使用标准的 **git 命令** 从版本历史记录中删除文件，而不是通过 GUI 手动删除；虽然手动删除一两个文件尚可，但为了方便起见，命令行操作更简单。
- **AI 生成节奏**：一位成员提到尝试使用 **AI** 为控制器创建 **drum kit**（鼓组），并指出这比使用全长采样效果更好。
   - 另一位成员表示：*“哈哈是的，我也在尝试为控制器‘用 AI 制作鼓组’，在我看来，它的效果确实比全长采样好得多。”*


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1369763085841207426)** (11 messages🔥): 

> `ACE-STEP SOTA, Alpha-Root, Entropy engine tests, AI Billing Dashboard, UQLM` 


- **ACE-STEP 调优 SOTA 音乐**：一位成员推介了 **ACE-STEP SOTA** 音乐生成模型，可在 [YouTube 视频](https://youtu.be/vyCALtrq4yQ)中查看。
- **Alpha-Root 提取网络安全数据**：一位成员介绍了 **Alpha-Root**，它直接在 Common Crawl 网络图谱上挖掘域名，在资源和数据消耗减少约 10 倍的情况下，达到了与 **PRIMUS-FineWeb** 相当的性能，详见[预印本草案](https://github.com/ashim-mahara/alpha-root/blob/main/Cybersecurity_Data_Extraction_from_Common_Crawl-3.pdf)。
   - 作者在不使用分类器的情况下，通过在已知数据集中搜索 URL，并仅保留同时存在于 **Alpha-Root** 和 **FineWeb-Edu** 中的 URL，从 **FineWeb-Edu** 中提取了 **3B tokens**。
- **熵引擎（Entropy Engine）评估随机性**：一位成员分享了其熵引擎的测试结果，代码托管在 [GitHub](https://github.com/thyarcanist/Entropy-MicroDemo)。
   - 他们发现*所使用的随机性质量确实会对模型产生影响*，这表明 **PRNG** 可能不是最优选，特别是对于 **AGI** 而言。
- **AI 计费仪表板解决成本追踪问题**：由于在 **HF Inference API**、**OpenAI** 和 **Claude** 等服务之间理清项目总成本非常令人头疼，一位成员构建了一个简单的仪表板 (**AIBillingDashboard.com**) 来一站式追踪所有 AI 支出。
- **UQLM 开源幻觉检测工具**：一位成员分享了一个名为 **UQLM** 的新型开源 Python 库，用于生成时的零资源幻觉检测，代码见 [GitHub](https://github.com/cvs-health/uqlm)。
   - 它利用最先进的不确定性量化（Uncertainty Quantification）技术，基于响应一致性、Token 概率、**LLM-as-a-Judge** 或这些方法的集成来计算响应层级的置信度分数。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1369751118846103662)** (4 messages): 

> `FlashAttention, OCR for Newspaper Data` 


- **FlashAttention 支持较新的 GPU**：一位成员确认 [Flash Attention 2](https://github.com/Dao-AILab/flash-attention) 支持 **FP16** 和 **BF16**，其中 **BF16** 需要 Ampere 或更高级别的 GPU。
- **报纸数据 OCR 任务**：一位成员请求对报纸数据进行 OCR，以提取**版块**、**类别**和 **10 位电话号码**到 Excel 的结构化数据库中。
   - 发布者指定要排除**公告**、**悼念**和**参考代码**，并将其余数据合并到 CSV 文件的单个描述列中。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1369911894558638151)** (2 messages): 

> `Dropwise module release, Emotion classification model questions, Token max length understanding, Production deployment of HF models` 


- **Dropwise 发布，用于 HF 模型不确定性评估**：一位成员宣布发布 **Dropwise**，这是一个 PyPI 模块，用于使用**蒙特卡洛 Dropout（Monte Carlo Dropout）**对 Hugging Face 分类模型进行**不确定性估计**。
   - 它旨在与 `transformers` pipelines 插件化配合使用，适用于问答（QA）、分类、OOD 检测和主动学习；参见 [GitHub](https://github.com/aryanator/dropwise) 和[文档](https://pypi.org/project/dropwise/)。
- **模型是在 Reddit 上训练的吗？**：一位使用 [emotion-english-distilroberta-base 模型](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)的成员根据 README 中的元数据询问该模型是否在 Reddit 帖子进行过训练。
   - 他们正在过滤 **anger**（愤怒）和 **disgust**（厌恶）情感得分超过 **0.85** 的 Reddit 帖子，并想知道该模型是否曾使用此类数据进行训练。
- **Token 长度会截断文本吗？**：一位成员询问了 **token 最大长度**对 NLP 模型的影响，询问超过限制的文本在分类过程中是否会被截断。
   - 他们还询问模型是仅适用于单行文本还是也适用于段落，并链接了他们的 [Python 脚本](https://github.com/moahnaf11/IdeaDrip-Backend/blob/main/inference_service/main.py)供检查。
- **生产模型部署：本地 vs. HF Endpoint？**：一位成员询问在 Python 中本地运行的 Hugging Face 模型是否可以用于生产环境应用，还是必须使用带 GPU 的付费 HF Endpoint。
   - 他们目前使用 FastAPI 在本地运行模型，并从 Node.js 应用中调用它，但担心生产环境的性能表现。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1369752816285126749)** (18 messages🔥): 

> `Agent 测试文件, 最终项目元数据, LLama Index 框架 vs Smolagent, RAG 作弊, API 请求限制` 


- **Agent 获取用于原子评估的测试文件**：一位成员分享了一个 [测试 Agent 文件](https://cdn.discordapp.com/attachments/1329142738440028273/1369756373835055104/test_agent.py?ex=681e5608&is=681d0488&hm=aa8206fb31afc9120ac0cd6d195223013c23224942a7a115b51ac5ce09312e53&)，用于在特定问题上测试 Agent，验证任务的正确性。
   - 它允许对 Agent 的性能进行原子级检查，并可以根据需要注释或取消注释测试用例。
- **最终实战项目中出现项目元数据**：一些正在进行最终实战项目的成员注意到，**高分提交**中包含一个 *metadata.jsonl* 文件，其中包含问题、答案和步骤，并好奇其来源。
   - 另一位成员回应称，*只要开始仔细观察，就很容易找到它*。
- **LLama Index 与 Smolagent 的对决**：讨论询问完成 **UNIT 2.2 THE LLAMA INDEX FRAMEWORK** 是否为强制性，或者是否可以使用 **smolagent** 或 **langgraph** 代替。
   - 一位成员总结道，*llamaindex* 是工具和 Agent 的另一个枢纽，与 *smolagents* 的独特之处在于能够编写 Python 代码来定义异步工作流图（async workflow graphs），作为多步任务的控制结构。
- **RAG 仓库骚乱：同学间因作弊产生分歧**：一些同学一致认为，使用 **带有答案的 RAG + 克隆仓库** 属于作弊行为。
   - 他们还表示，这剥夺了在排行榜上进行尝试、错误和改进的乐趣。
- **API 请求限制导致进度落后？**：一位用户报告在完成第一单元之前就达到了 **每月 20 次请求的限制**，并询问是否必须支付 Pro 版本才能继续。
   - 另一位用户提到，可以使用 **ollama** 运行本地 LLM，或者寻找其他免费层级。


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1369764648131498104)** (56 messages🔥🔥): 

> `Claude Plotly 图表, MCP 最大 Token 数, LLM 限制, Cloudflare 上的远程 MCP 服务器, Java MCP 服务器自定义参数` 


- **Claude 在处理 Plotly 图表时遇到困难**：成员们讨论了 **Claude** 作为 **MCP** 客户端无法直接在主结果区域显示 **Plotly** 或其他图表，但它可以处理 **ImageContent** 并显示 **EmbeddedResource** 格式，如 **image/png** 或 **image/jpeg**。
   - 建议将图表渲染为 **PNG/JPEG** 图像以便在 **Claude** 中显示。
- **MCP Token 限制得到澄清**：讨论澄清了 **MCP** 中的 **max tokens** 指的是响应中的最大 Token 数，类似于 completions API 请求中的 **max_tokens** 参数。
   - 总 Token 数（**system prompt + messages + output message**）必须保持在上下文窗口（context window）大小之内。
- **LLM 限制问题**：几位用户正面临 **LLM**（如 **Deepseek**）的限制，阻止其访问文件系统，这影响了他们的 **MCP** 系统功能。
   - 似乎某些模型被有意限制访问文件系统，这给通过 **MCP** 进行的合法用例带来了问题。
- **Cloudflare 远程服务器面临连接困扰**：一些用户报告部署在 **Cloudflare** 上的 **远程 MCP 服务器** 无法连接，而其他人则表示他们的设置运行正常。
   - 建议检查具体的 **MCP server repo** 以排查连接问题。
- **Claude Desktop 中的 MCP 工具权限进行改版**：用户注意到 **Claude Desktop** 的 **MCP** 工具权限提示发生了变化，“allow for this chat”（允许本次对话）和 “allow once”（允许一次）按钮被替换为 “allow always”（总是允许）和 “allow once”（允许一次）。
   - 这一变化引发了对误操作授予永久权限的担忧，以及缺乏撤销 “allow always” 设置选项的问题。


  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1369777585697062954)** (33 条消息🔥): 

> `面向 STDIO 的 MCP 客户端、OpenLink Software AI Layer (OPAL)、MCP Holster、AiraHub2、MCP 中的 Sampling` 


- ****Zinja** 打造 **Zin-MCP-Client****：发布了一个新的[轻量级、快速、基于 CLI 的 MCP 客户端](https://github.com/zinja-coder/zin-mcp-client)，用于 STDIO MCP 服务器，旨在连接本地 LLM 和 MCP 服务器。
   - 它专为配合 **jadx mcp servers** 使用而设计，利用本地 LLM 对 Android APK 进行 AI 辅助逆向工程。
- ****OpenLink 的 OPAL** MCP 服务器正式发布（GA）**：[面向 OpenLink Software AI Layer (OPAL) 的 MCP 服务器](https://community.openlinksw.com/t/introducing-the-openlink-ai-layer-mcp-server/4992) 现已正式发布，支持云端和本地部署，通过 Streamable HTTP 或 Server-Sent Events (SSE) 支持客户端和服务器角色。
   - 它支持原生/虚拟数据库查询、元数据探索、数据库治理、与 LLM/AI Agent 交互等，通过向任何符合 MCP 规范的客户端公开操作来实现。
- **使用 **Kimjune** 的工具 **Holstering** MCP 服务器**：一位用户分享了 [MCP Holster](https://github.com/kimjune01/mcp-holster)，这是一个无需手动编辑配置文件即可切换 MCP 服务器的工具。
   - 只要使用 **OAS3.0**，它就允许从现有 API 创建 MCP 服务器，如[此视频](https://youtu.be/TMbyv_RGEAk?si=W_i4kj4PijIfaGmd)所示。
- ****AiraHub2** 通过 MCP 与 **Claude** 集成**：[AiraHub2](https://github.com/IhateCreatingUserNames2/AiraHub2/tree/main) 现在通过 MCP remote 与 Claude 配合工作，通过 mcp-remote URL `https://airahub2.onrender.com/mcp/stream` 在网络上广播 MCP 工具。
   - 该系统注册 MCP 工具并进行广播，允许 Claude 连接并使用这些工具，尽管据报道目前仍有 *bug*。
- **MCP 中的 **Sampling** 引起关注**：成员们对 **MCP sampling** 表现出兴趣，一位用户分享了关于[如何在 MCP 中使用 sampling](https://www.epicai.pro/how-to-use-sampling-in-mcp-borrow-the-users-llm-o7nk3)的博客文章。
   - 另一位用户推广了他们的 [MCP webcam 项目](https://github.com/evalstate/mcp-webcam)，该项目支持带有“用户拿着什么”按钮的 sampling 功能。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1369763980226199692)** (58 条消息🔥🔥): 

> `Deepmind RL 机器人对比中国 RL 机器人、Linux 笔记本对比 Apple Macbook、Llama 4 令人失望、自动聊天审核系统拦截表情符号` 


- **中国 RL 机器人让 Deepmind 望尘莫及**：一位成员发布了一个 [YouTube 视频](https://www.youtube.com/watch?v=ET-MmoeSvXk)，对比了一年前 **Google Deepmind 的 RL 机器人成就**与近期**中国 RL 机器人的成就**，指出物理 AI 的进化速度极快。
- **MacBook M 系列芯片性能优于 Linux 笔记本？**：成员们讨论了使用配备 **Nvidia GPU** 的 **Linux 笔记本**与配备 **M 系列芯片**的 **Apple MacBook** 进行本地推理的优缺点，共识倾向于 MacBook，因为其性能和能效更高。
   - 有人提到 *M arm 芯片上的推理非常出色*，且 Apple 的统一内存平台允许 CPU、GPU 和 AI ML 神经网络共享同一内存，消除了来回传输数据的需要。
- **Llama 4 表现平平**：一位成员对 **Llama 4 的性能**（相比 **Qwen3**）表示失望，并建议等待 **Llama 4.1**。
   - 另一位成员回应称 *下一个大模型将回归 405 dense*。
- **Discord 拦截表情符号**：成员们发现**自动聊天审核系统**拦截了某些**多部分组成的表情符号**（特别是穿蓝色衬衫的耸肩表情），原因是这些表情使用了零宽连字符（zero-width joiners）和变体选择器来组合码点，这也是诈骗者绕过过滤器的手段。
   - 讨论揭示了 *dev 角色已从自动拦截名单中移除*。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 条消息): 

ifeq: 我得学普通话了
  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1369856701183824032)** (5 messages): 

> `Entropy Engine, Quantum-Native Randomness, LLM Sensitivity to Randomness, Importance of Randomness for AGI` 


- **Entropy Engine MicroDemo 发布**：一名成员发布了一个[量子原生且算法化的熵引擎](https://github.com/thyarcanist/Entropy-MicroDemo)供公开测试。
   - 该成员表示这是一个*自我推广*，但考虑到它对 **AGI** 的潜在影响，分享出来非常重要。
- **LLM 对随机性质量的反应**：一名成员指出 **LLM** 输出对所使用的**随机性**质量高度敏感，并区分了真**熵 (entropy)** 与 **PRNG**。
   - 他们假设高质量的熵能解锁模型中不同且通常更好的行为，并链接了[几条 X 帖子](https://x.com/thegautam/status/1920198569308664169?t=GehCezJb7amBPoter8F0gA)作为支持。
- **随机性对 AGI 至关重要**：一名成员认为随机性质量对 **AGI** 将非常重要。
   - 他们将继续进行测试以验证这一假设。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

ifeq: 我得学学普通话了
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1369761204561645588)** (35 messages🔥): 

> `Grok's apprehension of reality, Cloudflare serving fake content to agents, Third party filters for LLM output, Personal access to university resources via AI, KL Divergence Minimization` 


- **Grok 可能因右翼宣传而被削弱**：一名成员推测 **Grok** 在理解现实方面可能会被削弱以偏向右翼宣传，并链接了[一张附图](https://media.discordapp.net/attachments/738904561041014845/1368039830549692516/pz3v4ft279ye1.png)。
   - 他们补充说，真正的问题在于当今所有的难题早已存在，无论有没有 **AI**，我们仍然会面临这些问题。
- **Cloudflare 向 AI 提供虚假内容**：一名成员认为像 **Cloudflare** 这样的公司正在向 AI **Agent** 提供虚假内容，类似于几年前一些中国网站使用压缩包炸弹 (zip bombs) 来阻止克隆，这导致了 AI 产生偏见响应。
   - 此前另一名成员分享了 ChatGPT 如何错误地回答了一个与其分享的视频不符的问题。
- **LLM 输出需要第三方过滤器**：一名成员建议我们需要针对 **LLM 输出**的第三方过滤器，包括广告拦截和事实/偏见检查。
   - 作为回应，另一名成员建议需要许多理想情况下经常更换的模型，以免被腐蚀，例如 *100 个广告拦截模型和 100 个事实检查工具*。
- **个人通过 AI 获取大学资源**：一名成员表示期待未来每个人都能通过 AI 个人化地获取一所顶尖大学的心理、精神、智力和实用资源。
   - 另一名成员开玩笑地回复说他们*已经与 ASI 融合了*。
- **KL Divergence 最小化忽略了“模式”**：一名成员指出许多人开始使用 `---` 并分享了一篇名为 [Beyond Icon - A Unified Formulation for Objectives / Regularizations](https://github.com/EAzari/AML/blob/main/docs/beyond-icon.md) 的论文链接。
   - 他们声称作者在表 1 中比较了各种公式，但**未能意识到其中的模式**，即许多模式仅仅是 f(x) 和 g(x) = sum_x' f(x') 或 g(x) = f(x) ==> p(x) = f(x) / sum_x' f(x')。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1369820718841331743)** (7 messages): 

> `Paper Presentations, Causality, CVPR, Proper Investiture, Daily Paper Discussion` 


- **即将休假**：一名成员宣布他们**在接下来的两周内将休假**，但会回归，并鼓励他人在其缺席期间进行展示或组织。
- **新成员介绍**：一名新成员询问了每日论文讨论中**话题的广度**，因为他们想展示与 **Causality** 和 **CVPR** 相关的硕士论文。
   - 他们提到自己*还没有机会参加每日论文讨论*。
- **Proper Investiture**：一名成员分享了一个 [Springer 文章](https://link.springer.com/article/10.1007/BF02478259)链接，并评论道：*这大概是最正当的授职 (proper investiture)*。
- **每日论文讨论**：一名成员宣布今晚 `<t:1746750600:t>` 的每日论文讨论将关于[这篇 arXiv 论文](https://arxiv.org/abs/2305.13673)。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1369752562806423737)** (14 条消息🔥): 

> `Zed 在 Windows 上的编译，生物大脑 vs 反向传播，LLM 打通 Factorio == ASI？` 


- **Zed 在 Windows 上成功编译，需要 GitHub 登录**：一位成员按照[此处的说明](https://github.com/zed-industries/zed/blob/main/docs/src/development/windows.md)成功在 Windows 上编译了 **Zed**，但指出字体模糊，且需要登录 **GitHub** 才能使用标签页补全（tab completion）。
   - 另一位成员表示失望，想在 **LM Studio** 上尝试 **Mellum 4B** 进行标签页补全。
- **反向传播（Backprop）就是你所需的一切？**：一位成员表示，*生物大脑没有反向传播；它们是非纪元式的（non-epochal）、脉冲式的（spiking）、循环的（recurrent）模拟网络*，并引用了[这条推文](https://x.com/_neel_kant/status/1920516491025482066)作为证据。
- **Factorio ASI 基准测试提案**：一位成员开玩笑地提议，如果一个 **LLM** 能在不搞得一团糟的情况下打通游戏 **Factorio**，*我们就可以直接宣布那是 ASI*。
   - 他们链接了一个展示 **Factorio** 玩法的 [YouTube 视频](https://www.youtube.com/watch?v=pxGE41V04fs)。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1369753958809079888)** (41 条消息🔥): 

> `Cursor 广告，Slurm 内存请求，职位发布频道，语言学频道，Cursor 作为主要 IDE 的相关性` 


- **Discord 辩论 Cursor 广告规则**：成员们争论关于 **Cursor** 的帖子是否构成广告并违反了禁止广告的规则，考虑到它的受欢迎程度、感知到的实用性以及它并非完全免费，并指出 *“这仅仅是因为我们（群体）目前认为 Cursor 有用才勉强可以忍受，但它仍然会产生决策偏见”*。
   - 一些用户建议，模糊的规则被随意应用，同时将“禁止广告”解释为“禁止垃圾邮件”，并要求职位发布付费，这样可以过滤掉低质量的招聘。
- **用户偶然发现 Slurm 内存配置错误**：一位用户发现他们通过 **Slurm** 请求的是 **80MB** 内存，而不是 **80GB**，称之为 *“Slurm 时刻”*，而另一位用户则庆祝他们的裸机（bare-metal）设置。
   - 发现配置错误的该用户将最初的问题描述为 *“非常愚蠢”*。
- **关于 Discord 上职位发布的讨论**：围绕创建一个职位频道展开了讨论，担心它可能会被提供 *“经验”* 作为报酬的低质量帖子占领，有人建议付费发布作为潜在解决方案。
   - 其他人反对设立职位频道，认为这会使服务器变成另一个招聘场所，并建议 EleutherAI 不应为 Discord 服务器的差异化访问收费。
- **语言学频道受到关注**：一位用户提议设立一个古典语言学及其理论频道，重点关注 2000 年之前的知识，如句子形成和 *“即时”* 意义创造，旨在增加 NLP 领域不常见的讨论。
   - 它被描述为 *“由于‘某种’原因（可能是因为它与现在的工作无关）很少在 NLP 领域讨论的酷东西”*。
- **编程社区讨论 Cursor 作为主要 IDE 的缺点**：成员们表示，像 **Cursor** 这样的 AI 代码工具可能不如传统方法，例如 **tmux** 配合 **vim** 以及旁边的 **Claude code**。
   - 一位成员观察到 *“将 Cursor 专门作为主要 IDE 的人与能力不足之间存在极强的相关性”*。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1369813100282515588)** (7 条消息): 

> `MTurk vs. Prolific，RWKV 的 token shift` 


- **在人工评估方面 Prolific 优于 MTurk**：成员们推荐使用 [Prolific](https://www.prolific.co/) 而非 **MTurk** 进行人工评估，理由是其数据质量更高且参与者群体更可靠。
   - 共识是，在大约 *80% 的情况下*，Prolific 是更好的选择。
- **RWKV Token Shift 推测**：一位成员询问来自 **rwkv7** 的 **token shift** 和 **causal_conv1d** 是否相同。
   - 另一位成员澄清说，**RWKV** 中的 **token shift** 是 *归一化* 的，使得任何通道的所有时间权重之和等于 **1**，并引用了一篇[论文](https://arxiv.org/abs/2505.04588)。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1369929816265981964)** (2 条消息): 

> `The Pizza and the Clock` 


- **想要更多 Clock-Pizza？**：一位成员询问是否有更多像 [The Pizza and the Clock](https://arxiv.org/abs/2404.14082) 这样的论文。
   - 另一位成员回应建议 *这篇论文有很多参考文献*，但不确定对方在寻找什么。
- **N/A**：N/A
   - N/A


  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1369778357612576809)** (3 messages): 

> `LocalCompletionsAPI, loglikelihood tasks, bos token, HF model generation_config settings` 


- **LocalCompletionsAPI 的 loglikelihood 运行成功！**：一位成员正在使用基础模型和 `LocalCompletionsAPI` 实现运行 **loglikelihood（多项选择）任务**。
   - 他们确认运行良好，但发现分词后的 prompt 包含了 **bos token**。
- **BOS Token 的困扰！**：该成员询问在使用 `LocalCompletionsAPI` 时是否有办法指定 `add_bos_token=False`。
   - 他们希望控制是否将 **beginning-of-sequence token** 添加到 prompt 中。
- **HF Model generation_config：默认 Temperature？**：该成员询问，如果设置 `do_sample:true` 但不指定 `temperature`，是否会默认使用 **HF model 的 generation_config 设置**。
   - 他们澄清需要 `temp > 0`，否则会将 `do_sample` 设置为 false。


  

---


### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1369790693135876177)** (1 messages): 

> `NotebookLM, Mobile App, Trusted Tester Program` 


- **NotebookLM 启动移动端 App Trusted Tester 项目**：NotebookLM 即将推出 **移动端 App（测试版）** 📱，目前正在寻找经验丰富的 Web App 用户参与 **Trusted Tester 项目**，以共同塑造其未来。
   - 有兴趣的用户可以通过填写[此表格](https://forms.gle/XD1VmJ7FP4AjbDB66)进行注册，其中包括查看并同意 **Trusted Tester 条款**。
- **招募 Trusted Testers 进行 NotebookLM 移动端 App 测试**：NotebookLM 寻求经验丰富的 Web App 用户成为其移动端 App 测试版的 **Trusted Testers**。
   - 测试人员将获得 **早期访问权限**，作为交换，需提供反馈并报告 Bug；注册需要同意 [Trusted Tester 条款](https://forms.gle/XD1VmJ7FP4AjbDB66)。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1369802829719666709)** (9 messages🔥): 

> `NotebookLM PDF Processing, NotebookLM Knowledge Base for Sales, Audio length limitations` 


- **实验确定的 NotebookLM PDF 处理限制**：用户报告 **NotebookLM** 在处理大型 PDF 或大量 PDF 时表现不佳；一位用户通过询问 PDF 较后部分的问题进行测试，发现 **200 页**之后会出现问题。
   - 该用户建议运行一个快速实验来测试当前的限制。
- **NotebookLM 仅根据源材料进行回答**：在聊天界面中，**NotebookLM** 仅根据上传的源材料生成回答。
   - 如果问题的答案不存在于导入的材料中，AI 将声明源材料中不存在与该问题相关的信息。
- **为 NotebookLM 构建销售内容知识库**：一位用户正在 **NotebookLM** 中构建销售内容知识库，在 **300 份文档限制**内使用主要的客户演示文稿和销售赋能材料。
   - 他们计划向内部销售团队开放访问权限，寻求指导、示例以及对限制的理解，特别是关于共享和潜在的信息孤岛问题。
- **通过更多内容延长音频生成时间**：一位用户询问如何让音频变得更长，目标是至少 **12 分钟**或更久。
   - 另一位用户建议提供更多输入，以便给系统提供更多可处理的内容。


  

---

### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1369761967035912222)** (22 条消息🔥): 

> `NotebookLM 无法回答问题、视频上传、音频概览功能、播客长度、AI 的“拟人化”行为` 


- ****NotebookLM 故障：系统拒绝回答****：用户报告称 **NotebookLM** 响应为 *'The system was unable to answer'*（系统无法回答），即使是要求总结默认笔记本时也是如此，在生成思维导图和学习指南时也出现了问题。
   - 一些用户正在寻求解决方案，并确认其他人是否也面临同样的问题。
- ****支持视频上传（但有限制）****：用户确认 **NotebookLM** 支持 **mp4** 和 **avi** 等格式的视频上传，这与 Google 官方网站上的一些不准确信息相反，详见 [Google 支持页面](https://support.google.com/notebooklm/answer/14276468?hl=en)。
   - 它会分析视频文件的音频部分并提供转录和摘要，但**不支持 mov 格式**。
- ****音频概览入口难寻****：一位用户询问如何访问音频概览功能并与其交互，但找不到该选项。
- ****播客长度因语言而异****：一位寻求延长播客长度技巧的用户注意到，将语言更改为 **English** 可以生成明显更长的音频摘要（长达 **49 分钟**），而其他语言则限制在 **14 分钟**左右。
   - 一名团队成员表示这是预期行为，他们正致力于很快在其他语言中实现更长的音频摘要。
- ****对人工“拟人化” AI 的不适****：一位用户对 AI 在深度探讨（deep dives）中的“拟人化”行为表示不适，特别提到了不自然的 *'uhm's'*（嗯、啊等语气词），并询问如何去除这种行为。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1369778004502249513)** (22 条消息🔥): 

> `X 平台内容、Netflix 推荐模型、Gemini 图像生成、aider 事后分析、Suno 音乐` 


- **X 平台内容分享**：成员们分享了来自 **X**（原 Twitter）的链接，包括一个通用链接以及来自 [thegautam](https://x.com/thegautam/status/1920198569308664169)、[TheAhmadOsman](https://x.com/TheAhmadOsman/status/1920236407101997243) 和 [openaidevs](https://x.com/openaidevs/status/1920556386083102844) 等用户的特定帖子。
   - 分享的内容似乎引起了频道成员的普遍兴趣，并引发了简短的确认。
- **Netflix 利用基础模型实现个性化推荐**：一位成员强调 **Netflix** 开发了一个[用于个性化推荐的基础模型](https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39)，正如其中一个分享链接的评论中所提到的。
   - 这一点是在与其他关于推荐系统的讨论相关联时被指出的。
- **Gemini 新图像生成引发关注**：成员们分享了一个展示 [Gemini 新图像生成](https://x.com/OfficialLoganK/status/1920151503349711061) 的链接。
   - 一位成员提到，*该团队将在 aie world’s fair 的 recsys x llms 分论坛上进行展示*。
- **Aider 事后分析：比个人笔记更详尽？**：成员们注意到 [aider postmortems](https://aider.chat/2025/05/07/gemini-cost.html) 非常详尽，特别是在 Gemini 成本分析方面。
- **Suno 的声音风格：约德尔蓝调演唱会**：一位成员对 **Suno** 混合风格的能力赞不绝口，特别强调了成功尝试创建 *约德尔 + 蓝调 + 现场演唱会* 的混合风格。
   - 他们分享了一个[音频文件](https://cdn.discordapp.com/attachments/1075282825051385876/1370022129050849441/you_can_YODEL_with_Suno.mp3?ex=681dfc09&is=681caa89&hm=e16a84ff105d7fc1bef2fd343a067b7ea6ffa1964772d2e3ad9900e355f2d2c2&)作为 **Suno** 令人印象深刻的输出证据。


  

---

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1369826322351915168)** (2 messages): 

> `Claude code pod, AI Engineer conference, Early Bird Tickets, AI Engineer conference speakers` 


- **新的 Claude Code Pod 登场**：Latent Space 播客推广了一个 [新的 Claude code pod](https://x.com/latentspacepod/status/1920240470296572316)。
   - 听众对这次合作可能带来的新剧集和见解感到兴奋。
- **AI Engineer Conference 早鸟票即将售罄**：定于 6 月举行的 AI Engineer 大会提醒社区成员，[早鸟票](https://www.ai.engineer/#speakers) 预计将在周末售罄。
   - 鼓励参会者尽快购票，以享受折扣价格。
- **AI Engineer Conference 演讲嘉宾揭晓**：AI Engineer 大会公布了 6 月活动的 [演讲嘉宾阵容](https://www.ai.engineer/#speakers)。
   - 爱好者们渴望看到演讲嘉宾将为大会带来的专业知识和见解。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1369806752845140149)** (15 messages🔥): 

> `Fields in traits vs properties, Modular Hackathon at AGI House, Hardware Agnostic ML Systems Survey Paper, Zotero and bibtex for citations` 


- **Mojo Trait 中属性优于字段**：围绕在 Mojo 的 trait 中加入字段的可能性展开了讨论，但有人认为 *trait 中的属性 (properties)* 是一个更优、更通用的想法。
   - 有人指出，trait 中的字段*可能*会实现，但人们将无法通过 extension 添加此类 trait；它需要包含在原始 struct 定义中。
- **Modular 黑客松在 Hillsborough 引发热潮**：分享了关于本周六在 AGI House 举行的 Modular 黑客松的最后提醒，目前还有少量名额，点击 [此处](https://app.agihouse.org/events/modular-hackathon-20250510) 报名。
   - Modular 团队成员以及 Mark Saroufim (GPU MODE & PyTorch)、Simon Boehm 和 Sasha Krassovsky (Anthropic) 以及 Dylan Patel (SemiAnalysis) 将发表演讲。
- **硬件无关 ML 综述论文发布**：一名成员完成了关于 modular 和 **Hardware Lottery** 的综述论文，并将其用于最终演示，向同行讲述了一个精彩的故事。
   - 该论文的最新版本始终可以在 [此处](https://github.com/TheAgaveFairy/HPML-Survey-Project/blob/main/The_Quest_for_Unification__A_Survey_of_Hardware_Agnostic_Machine_Learning_Systems.pdf) 获取，欢迎提供反馈。
- **Zotero 解决引用难题**：在关于引用的讨论中，建议使用 **Zotero** + **bibtex** 可以解决大部分问题。
   - 一名成员分享了他的痛苦经历：*natbib 报错了大约 70 次，几乎没有任何链接，直到我发现了一个未转义的 '%'*。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1369814755304542290)** (4 messages): 

> `Mojo roadmap, GPU programming puzzles, Colab Integration, New requires keyword` 


- **Mojo 路线图发布！**：Modular 在论坛上发布了近期 **Mojo 路线图**，参见 [官方帖子](https://forum.modular.com/t/whats-next-for-mojo-near-term-roadmap/1395)。
   - 该路线图详细说明了 **Mojo** 语言即将推出的功能。
- **GPU Puzzles 吸引 Mojo 程序员**：新的 **GPU programming puzzles** 看起来很有趣，一些成员想知道是否可以在没有 GPU 的情况下在 **Colab** 上运行它们。
   - 一名成员表示：*新的 GPU 编程谜题看起来真的很酷*。
- **Colab 接入 Mojo (黑客式实现)**：一名成员发布了一个 *在 Colab 的免费和 Pro 层级 GPU 上运行 **Mojo 代码** 的简易 **Colab notebook** 实现*：[Colab notebook](https://forum.modular.com/t/max-can-now-be-used-in-google-colab/1383/2?u=bradlarson)。
   - 该成员承认 *只需稍加努力就能获得更好的体验*，并指出 *它将 cell 构建为单个 Mojo 文件，你必须查看日志来获取编译错误等*。
- **“Requires” 关键字需求反响热烈！**：Discord 频道对用于在 struct 和函数中添加约束的新 **requires keyword** 反应非常积极。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1369804249449627823)** (13 messages🔥): 

> `合作与伙伴关系, ReAct 模块签名, DSPy 缓存机制, 在 Qwen 1.7B 上使用 GRPO 进行的 RL 实验` 


- **新的合作伙伴关系即将到来？**: 一位成员询问该项目是否开放 **collab and partnership**（合作与伙伴关系）以共同提升社区影响力。
   - 该成员询问是否可以开始对话以讨论潜在的协同效应。
- **ReAct 模块不需要输出**: 一位成员询问如何为 **ReAct module** 创建一个仅进行工具调用而不需要其他输出的 **signature**（签名）。
   - 另一位成员建议使用 *success: bool* 作为输出，以指示任务何时完成。
- **DSPy 缓存：多层迷雾**: 一位成员发现 **DSPy** 除了 **LLM** 提供商的缓存外，还拥有自己的缓存机制，这可能在凭证过期时导致意外结果。
   - 多层缓存包括 **DSPy's cache** ([github.com/stanfordnlp/dspy/blob/main/dspy/clients/cache.py](https://github.com/stanfordnlp/dspy/blob/main/dspy/clients/cache.py))、**LiteLLM's cache** ([docs.litellm.ai/docs/proxy/caching](https://docs.litellm.ai/docs/proxy/caching)) 以及 **Bedrock's cache**，这些都会增加调试难度。
- **GRPO 开始运行，Recall 下降**: 一位成员在 **Qwen 1.7B** 上使用 **GRPO** 运行了一个小型 **RL experiment**（强化学习实验），利用 DSPy 优化检索的查询重写，发现训练后基准 Recall（召回率）从 **28%** 下降到 **26%**。
   - 更多细节可在 [Twitter 线程](https://x.com/tahmidtapadar/status/1920469176776302679)中找到，指出下降 *可能是由于稀疏奖励、短程运行以及 BM25 与 CoT 重写不匹配造成的*。


  

---


### **Cohere ▷ #[💬-general](https://discord.com/channels/954421988141711382/954421988783444043/1369781349744906381)** (7 messages): 

> `Cohere Embedding Model, Cohere Rerank Model, Cohere Embed 4` 


- **Embedding Model 在处理谈判语义时遇到困难**: 用户注意到 **embedding model** 在处理谈判内容时表现不佳，用户查询 *"I can pay"* 与嵌入数据 *"No, I cannot pay"* 之间的分数返回为 **0.92**，相似度过高。
   - 一位成员建议在这种情况下尝试使用 **rerank model**，而不仅仅是向量相似度。
- **Cohere Embed 4 Token 级嵌入**: 一位成员询问是否可以使用 **Cohere Embed 4** 获取 **token level embeddings**。
   - 另一位成员回答说可以一次嵌入一个 Token，但不建议这样做。


  

---


### **Cohere ▷ #[💡-projects](https://discord.com/channels/954421988141711382/1218409701339828245/1369850766302511285)** (1 messages): 

> `AI 成本追踪, 多平台 AI 服务管理, AI 支出证明, AI 工具痛点` 


- **AIBillingDashboard 跨 AI 平台追踪成本**: 一位独立创始人兼软件工程师创建了 [AIBillingDashboard.com](https://AIBillingDashboard.com)，该平台可帮助用户追踪并优化在 **Cohere**、**OpenAI**、**Anthropic**、**Azure AI** 和 **Google Vertex** 等多个提供商处的 **AI service costs**。
   - 该平台整合了成本，帮助分配支出，提供使用分析，并实现跨所有服务的预算追踪。
- **追踪支出并发现优化机会**: 创作者发现跨多个 AI 服务追踪和分析成本非常困难，因此创建了这一 **unified dashboard**（统一仪表板）。
   - 该仪表板解决了手动从不同控制台提取报告、难以将成本分配给特定项目以及缺乏总 **AI spend**（AI 支出）统一视图的问题。
- **征集 AI 成本/使用量追踪方面的痛点**: 创始人正在征求用户在 AI 成本和使用量追踪方面面临的问题反馈。
   - 痛点示例包括难以比较性价比、预测成本具有挑战性以及难以向管理层证明 **AI expenses** 的合理性。


  

---


### **Cohere ▷ #[🤝-introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1369781615152332961)** (3 messages): 

> `合作, 自我介绍` 


- **成员寻求合作以获取竞争性利润**: 一位成员正在寻找合作伙伴，并向欧洲和美国的合作者承诺提供 *competitive profit*（具有竞争力的利润）。
   - 欢迎感兴趣的人士发送 DM 以进行更详细的合作讨论。
- **鼓励使用模板进行自我介绍**: 社区欢迎新成员并鼓励他们介绍自己，表示社区非常高兴他们的加入。
   - 提供了一个模板，要求提供 **Company/Industry/University**、当前项目、喜好的技术/工具以及希望从社区获得什么等信息。


  

---

### **Cohere ▷ #[🟢-status-updates](https://discord.com/channels/954421988141711382/1346652044181897307/1370050082556084374)** (1 messages): 

> `Embedding Models Degraded, embed-english-v2.0, embed-english-v3.0` 


- **Embedding Models 遭遇故障**：Cohere 报告 **embed-english-v2.0** 和 **embed-english-v3.0** 模型出现 [性能下降](https://ift.tt/WvxjUwp)，目前正在调查中。
   - 更多详情可在 [Cohere Status Page](https://ift.tt/bE5aXAs) 查看，更新时间戳为 2025 年 5 月 8 日上午 07:25。
- **Cohere 调查 Embedding Model 性能问题**：Cohere 正在积极调查导致特定 Embedding Models 性能下降的实时事件。
   - 受影响的组件包括 **embed-english-v2.0** 和 **embed-english-v3.0**，如状态更新所示。


  

---


### **Cohere ▷ #[🎯-private-deployments](https://discord.com/channels/954421988141711382/1351999070247583848/1370058437676765358)** (1 messages): 

> `GPU Requirements, On-Premise Deployment of Command A` 


- **Command A 的 GPU 规格查询**：一位用户正在寻求有关 **Command A** **本地化部署 (On-Premise Deployment)** 的具体 **GPU 需求** 信息。
- **解析 Command A 的 GPU 需求**：该用户旨在了解在其自有基础设施中成功部署和运行 **Command A** 所需的必要 **GPU 规格**。


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1369886618055348394)** (5 messages): 

> `Tokenizer Automation, HuggingFaceBaseTokenizer Limitations, Custom Autotokenizer, ModelTokenizer Wrapper` 


- **讨论 Tokenizer 自动化目标**：一位成员正寻求为使用 `torchtune` 的内部客户自动化跨模型类型的 Tokenizer 识别。
   - 目标是移除或自动化识别 Tokenizer 的人工操作环节，旨在实现 `torchtune` 更通用的用法。
- **`HuggingFaceBaseTokenizer` 在 SFT 方面存在局限性**：`HuggingFaceBaseTokenizer` 缺乏消息模板化/分词 (templating/tokenizing) 逻辑，限制了其仅能用于文本补全 (text completions) 训练，而不能用于 SFT。
   - 讨论强调，由于缺乏消息模板化功能，该 Tokenizer **无法用于监督微调 (SFT)**。
- **建议使用自定义 Autotokenizer**：建议为内部客户编写自定义 “autotokenizer”，并将其作为配置中的默认设置。
   - 该 autotokenizer 可以使用 if 语句或更巧妙的方法，在配置顶部为 Tokenizer 和 checkpointer 定义模型名称。
- **计划通过 `ModelTokenizer` 封装器填补 HF 差距**：torchtune 中存在一个已知差距，计划提供一个封装 `HuggingFaceBaseTokenizer` 的 `ModelTokenizer`，将 HF 的 `apply_chat_template` 映射到 torchtune 的 `tokenize_messages`。
   - 这一增强功能预计将极大地帮助用户接入新模型，并将在 [GitHub 仓库](https://github.com/pytorch/torchtune) 上开启一个 issue 来勾勒实现细节，邀请社区贡献。


  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1369752160493113495)** (8 messages🔥): 

> `Cosine Scheduler with Warmup, Pytorch NaN bug with compiled Adam, Torchtune's get_cosine_schedule_with_warmup function, Torchtitan LR Scheduler Implementation, LR Warmup scaling` 


- **余弦调度器的复杂性：预热与学习率调度**：讨论围绕实现[带预热的余弦调度器](https://github.com/pytorch/torchtune/pull/2681)展开，并处理一个 **PyTorch bug**，该 bug 在使用编译后的非融合 (non-fused) **Adam/AdamW** 优化器且学习率调度器在训练期间任何点将学习率设置为 0 时，会导致 **NaN 权重**。
   - 当 `get_cosine_schedule_with_warmup` 在第一步将学习率设置为 0 时会触发该 bug，这与之前允许在优化器编译 (optimizer compile) 时使用 LR 调度器的初始 bug 修复产生了冲突。一位成员指出 [Torchtitan 的实现](https://github.com/pytorch/torchtitan/blob/00a53646c184493d292836f7d8bbe0bed859993f/torchtitan/components/lr_scheduler.py#L120)可能是一个潜在的解决方案。
- **Adam 的烦恼：编译后的优化器和零学习率导致 NaN 权重**：据报告，在使用编译后的非融合 **Adam/AdamW** 优化器并配合在某些点将学习率设置为 0 的学习率调度器时，会出现 **NaN 权重** 的 PyTorch bug。
   - 一位成员指出 *Torchtune* 的 `get_cosine_schedule_with_warmup` 总是会在第一步将学习率设置为 0，从而在启用优化器编译时触发该问题。
- **Titan 的方法：训练开始时的 LR 预热**：提到 [Torchtitan 的实现](https://github.com/pytorch/torchtitan/blob/00a53646c184493d292836f7d8bbe0bed859993f/torchtitan/components/lr_scheduler.py#L120)在第一步将 LR 比例设置为 `1/(warmup_steps+1)`，但除非设置了 `lr_min`，否则最后一步仍将为 0。
   - 一位成员表示：*Torchtitan 的方法也行得通，因为直接跳过第 0 步是合理的。*
- **深入了解 LR 预热缩放**：关于 LR 缩放策略的讨论：对于预热步数，你可能希望使用 `min_lr + (1/n ) * (1 - min_lr), min_lr + (2/n ) * (1 - min_lr), ..., min_lr + (n-1/n ) * (1 - min_lr)`，而不是 `0, 1/n, 2/n, ..., n-1/n`。
   - 对于余弦调度，你希望通过余弦调度的逆函数来缩放进度，因此 `progress *= arccos(2*min_lr-1)/(pi*2.0*num_cycles)` 将导致计算出的最大进度使得 `cosine_lr_multiple == min_lr`。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1369780014417055804)** (3 messages): 

> `Anthropic API web search tool, LlamaParse improvements, VoyageAI multi-modal embeddings and MongoDB indexes` 


- **Anthropic API 搜索工具诞生**：根据[这条推文](https://twitter.com/llama_index/status/1920220803976867882)，Anthropic API 现在支持内置的网页搜索工具，并且 LlamaIndex 已提供首日支持 (day 0 support)。
- **LlamaParse 增加 Gemini 和 GPT4 支持**：LlamaParse 正在改进，增加了 **GPT 4.1** 和 **Gemini 2.5 Pro** 模型等新功能，此外还增加了自动定向 (auto orientation)、偏斜检测 (skew detection) 以及用于解析质量的置信度分数 [根据这条推文](https://twitter.com/llama_index/status/1920505775677722750)。
- **使用 MongoDB 进行多模态检索之旅**：在[这个 Notebook](https://twitter.com/llama_index/status/1920563641990209643) 中学习如何使用 **VoyageAI** 的多模态嵌入和 **MongoDB** 的多模态索引进行多模态检索。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1369807531903549502)** (4 messages): 

> `Medical LLM Bot, Fine-tuning vdr-2b-multi-v1 with math formulas, Writer's Palmyra X5 and X4 in Bedrock` 


- **寻求医疗 LLM 机器人工作流建议**：一位用户正在构建医疗 LLM 机器人，并寻求实现工作流的指导，该工作流包括根据本地 LLM 之前的回答迭代地建议后续问题。
   - 他们正在咨询 LlamaIndex 是否有工具可以帮助构建此类工作流。
- **为数学公式微调 vdr-2b-multi-v1**：一位用户询问如何使用 **llamaindex/vdr-multilingual-train** 数据集微调 **vdr-2b-multi-v1** 模型，以便更好地处理文档中的复杂数学公式。
   - 他们注意到训练数据中不存在公式，并正在寻求在此背景下进行微调的资源、步骤或教程。
- **LlamaIndex Bedrock 中的 Palmyra 模型错误**：一位用户报告在 LlamaIndex 中使用 **Amazon Bedrock** 的 **Writer Palmyra X5 和 X4** 基础模型时遇到错误：*"Provider writer for model us.writer.palmyra-x5-v1:0 is not supported"*。
   - 他们指出这些模型在 Amazon Bedrock 中是可用的。


  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1370063478622126295)** (4 messages): 

> `tinygrad CUDA, tinygrad IR, tinygrad docs, tinygrad uops` 


- **探索 tinygrad 的 CUDA 集成**：一位用户询问了 **tinygrad 通常如何集成 CUDA 支持**。
   - 他们还询问了 **tinygrad** 是否有自己的 **Intermediate Representation (IR)**。
- **深入 tinygrad 文档**：一位用户分享了 [tinygrad 官方文档](https://docs.tinygrad.org/) 的链接。
   - 他们还分享了关于 [tinygrad uops](https://xl0.github.io/tinygrad-notes/uops.html) 的笔记以及更多 [tinygrad 笔记](https://mesozoic-egg.github.io/tinygrad-notes/) 的链接。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1369750731212587038)** (3 messages): 

> `CACHEDB environment variable` 


- **发现 CACHEDB 环境变量位置**：一位成员询问了 **CACHEDB** 环境变量。
   - 另一位成员指出它在 *helpers 的第 175 行* 被提及。
- **澄清 CACHEDB 的用途**：继最初的查询之后，**CACHEDB** 环境变量的功能并未被明确说明。
   - 需要进一步讨论以了解该变量在项目中的实际应用和上下文。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1369814601595883530)** (1 messages): 

> `AgentX Workshop, Lambda Inference API, Agentic AI` 


- **Lambda 举办 AgentX 工作坊**：Lambda 将于 **太平洋时间 5/15 上午 10 点为 AgentX 竞赛参与者和希望使用 Lambda 强大的 Inference API 扩展项目的 AI 爱好者举办 AgentX 工作坊：使用 Lambda 构建 Agentic AI**。
   - 参与者将学习构建实用的 agentic 应用、优化 Agent 性能，并在生产环境中部署 Agent，包括现场演示。
- **AgentX 奖项公布**：AgentX 竞赛参与者可获得特别奖项，在创业（Entrepreneurship）和研究（Research）赛道中，**第一名可获得高达 1,000 美元的额度**，**第二名 500 美元**，**第三名 300 美元**。
   - 感兴趣的参与者可以[立即注册](https://lu.ma/AgentX-lambda)以获取 YouTube 直播链接。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1369815507942772736)** (4 messages): 

> `HF Credits, Course Content, MOOC Iterations` 


- **用户等待 Hugging Face 额度**：两位用户报告了追踪 **Hugging Face 额度** 的问题，其中一位未收到邮件，另一位正在等待批准。
   - 第一位用户提到*每天访问网站很具有挑战性*。
- **课程讲座确认与澄清**：一位准学生询问了课程内容，特别是 [课程网站](http://llmagents-learning.org/sp25) 上列出的嘉宾讲座是否全面。
   - 工作人员澄清说，列出的讲座确实是全面的，并确认 **Spring MOOC** 包含更多高级主题，如 *代码生成和定理证明*，而 **Fall 版本** 包含更多 *应用主题*。
- **另一轮课程即将到来？**：一位用户询问了课程的未来迭代，特别是 **秋季** 是否会有另一场。
   - 工作人员回答说，Song 教授今年秋季将在伯克利开设另一门关于 *Agentic AI* 的课程，但尚不清楚是否会有 MOOC 版本。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1369982709996195913)** (2 messages): 

> `AI Engineer Courses, LLM Agents MOOC` 


- **为志向成为 AI Engineer 的人推荐 LLM Agents MOOC**：一位成员询问了成为 **AI Engineer** 的最佳完整课程。
   - 另一位成员建议从 [Fall 2024 LLM Agents MOOC](https://llmagents-learning.org/f24) 开始。
- **AI Engineer 职业路径**：一位用户询问了成为 **AI Engineer** 的资源。
   - LLM Agents MOOC 被建议作为一个坚实的起点。


  

---

### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1370110911741952000)** (1 条消息): 

> `JetBrains Plugin Updates, Windsurf Editor UX Improvements, Wave 8 Release` 


- **Windsurf Wave 8 带来 UX 和插件提升**：Windsurf 最后的 **Wave 8** 发布引入了对 **JetBrains plugin** 的增强以及对 **Windsurf Editor** 用户体验的改进，详情见 [博客文章](https://windsurf.com/blog/windsurf-wave-8-ux-features-and-plugins) 和 [更新日志](https://windsurf.com/changelog)。
- **JetBrains Plugin Cascade 添加 Memory 和 Rules**：更新后的 **JetBrains plugin** 现在支持用于跨会话持久化信息的 **Memories**、通过 `.windsurfrules` 文件实现的 **Rules**，以及 **MCP** (Model Context Protocol) 服务器连接，详情见 [Jetbrains plugin 更新日志](https://windsurf.com/changelog/jetbrains)。
- **Windsurf Editor 获得 UX 功能更新**：**Windsurf Editor** 的 UX 改进包括 **Continue 按钮**、重新设计的模型选择器、用于过滤历史记录的工作区到对话映射、增强的代码块和 hunk 导航、可编辑的终端命令，以及 Chat 模式下的新文件提案，如 [今天的发布视频](https://youtu.be/IjE8Cdxotso) 所示。


  

---


---


---


---