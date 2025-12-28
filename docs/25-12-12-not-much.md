---
companies:
- openai
- allen_ai
- mistral-ai
- ollama
- lmstudio
- thinkymachines
date: '2025-12-12T05:44:39.731046Z'
description: '**GPT-5.2** 在公开评估中表现参差不齐，虽然在智能体（agentic）任务中表现出色，但与 **Opus 4.5** 和 **GPT-5.1**
  相比，其成本显著更高（每次运行约 **620 美元**）。它在推理和编程基准测试中的表现各异，但在长上下文任务上有所改进。延长的“推理强度”（reasoning
  effort）设置对结果有显著影响。聚合平台在任务持久性方面将 **Gemini 3 Pro** 排在 GPT-5.2 之上。**OpenAI** 发布了稀疏激活模型，引发了关于稀疏性与混合专家（MoE）架构的辩论。**Allen
  AI** 的 **Olmo 3.1 (32B)** 通过巨大的算力投入（约 **12.5 万 H100 小时**）推进了开源强化学习的规模。**Mistral**
  的 Devstral-2 和 **llama.cpp** 通过 GGUF 支持和分布式加速等新功能改进了本地推理基础设施。**Tinker** 平台正式上线（GA），并增加了对
  **Qwen3-VL-235B** 的视觉输入和微调支持。'
id: MjAyNS0x
models:
- gpt-5.2
- opus-4.5
- gemini-3-pro
- gpt-5.1
- olmo-3.1-32b
- qwen3-vl-235b
people:
- sama
- scaling01
- akhaliq
- artificialanlys
- lechmazur
- acerfur
- epochairesearch
title: 今天没发生什么特别的事。
topics:
- reinforcement-learning
- model-benchmarking
- long-context
- model-quantization
- model-optimization
- inference-speed
- sparsity
- fine-tuning
- vision
---

**一个安静的周五。**

> 2025年12月11日至12月12日的 AI 新闻。我们为您检查了 12 个 subreddits、544 个 Twitter 账号和 24 个 Discord 社区（205 个频道，8597 条消息）。预计节省阅读时间（以 200wpm 计算）：621 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以精美的 vibe coded 方式呈现所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上给我们反馈！

更多 [AIE Talks](https://www.youtube.com/channel/UCLKPca3kwwd-B59HNr-_lvA) 将在整个周末陆续推出。

---

# AI Twitter 回顾

**前沿模型评估：GPT‑5.2 vs Opus 4.5 和 Gemini 3，成本与上下文设置**

- **GPT‑5.2 在公开评估中的表现参差不齐**：在实际工作的 Agent 任务中，GPT‑5.2 在 GDPval‑AA 上位居榜首，超越了 Claude Opus 4.5，但成本更高——每次运行约 **$620**，而 Opus 4.5 为 **$608**，GPT‑5.1 为 **$88**。据 [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1999404579599823091) 称，这是由于 Token 数量增加了 6 倍以上以及 40% 的价格上涨所致。在推理/编程基准测试中，社区运行报告显示：在 LiveBench 上低于 Opus 4.5 和 Gemini 3 Pro；在 SimpleBench 上表现疲软（低于 Sonnet 3.7）；在 VendingBench‑2 上有所改进，但仍落后于 Gemini 3 Pro/Opus 4.5 ([@scaling01](https://twitter.com/scaling01/status/1999323401421488319), [@scaling01](https://twitter.com/scaling01/status/1999466846563762290), [@scaling01](https://twitter.com/scaling01/status/1999449402776387808))。长上下文 MRCR v2 显示 5.2 xhigh 击败了 Gemini 3 Pro；OpenAI 在修复 v1 后更新了 MRCRv2 ([thread](https://twitter.com/DillonUzar/status/1999328225164431394), [@scaling01](https://twitter.com/scaling01/status/1999327512401527107))。评估可能非常敏感：@eliebakouch 发现不同机构的 GPT‑5.1 MRCR 数据存在指标差异，随后将其归因于不同的基准测试变体 ([note](https://twitter.com/eliebakouch/status/1999534955274117457))。
- **注意“推理力度 (reasoning effort)”调节旋钮**：GPT‑5.2 的几个亮点依赖于 xhigh 扩展思考（例如 100k “thinking” Token），这可以实质性地改变结果 ([@scaling01](https://twitter.com/scaling01/status/1999535536130662576))。例如：扩展版 NYT Connections 的得分从 **77.9 (High)** 跃升至 **89.3 (xHigh)** ([@LechMazur](https://twitter.com/LechMazur/status/1999582591905583256))。社区情绪从显著更好的证明写作 ([@AcerFur](https://twitter.com/AcerFur/status/1999314476320063546)) 到“并不明显比 Gemini 3 Pro 强大” ([@scaling01](https://twitter.com/scaling01/status/1999566015873569174)) 不一而足。
- **聚合器**：Epoch 的 ECI 将 GPT‑5.2 置于 **152** 分，仅次于 Gemini 3 Pro；他们的 ECI 到时间跨度 (Time‑Horizons) 的预测表明，5.2 的中值任务持久性为 **3.5h**，而 Gemini 3 Pro 为 **4.9h**，Opus 4.5 为 **2.6h** ([@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1999548496198926728), [follow‑up](https://twitter.com/EpochAIResearch/status/1999585243003781413))。与此同时，[@sama](https://twitter.com/sama/status/1999624463013544024) 声称 GPT‑5.2 在第一天“API 中的 Token 消耗就超过了一万亿”，强调了其快速普及。

**开源模型、RL 扩展与稀疏性**

- **Allen AI 的 Olmo 3.1 (32B) 推动了开源 RL 规模**：将之前的 RL 任务延长了 3 周，产出了 Olmo 3.1 Think 32B 和 Instruct 32B；消耗了约 **12.5万 H100 小时**（约 25 万美元），在困难评估（AIME、编程）上持续获得提升。发布了中间 Checkpoint、新的 7B 数学/编程 RL‑Zero 基准，以及大型过滤/偏好数据集 ([@allen_ai](https://twitter.com/allen_ai/status/1999528336318509316), [@natolambert](https://twitter.com/natolambert/status/1999528636085649532))。结论：长期运行的 RL 仍有待进一步探索，并持续带来改进。
- **来自 OpenAI 的电路稀疏性 (Circuit‑sparsity)**：公开发布的稀疏激活模式/模型 (huggingface.co/openai/circuit‑sparsity) 引发了关于其与经典 MoE 权衡的讨论；一些人认为，具有共享容量的大型稀疏激活架构可能优于孤立的专家模型 ([@_akhaliq](https://twitter.com/_akhaliq/status/1999528833490239864), [commentary](https://twitter.com/teortaxesTex/status/1999559676866724272))。
- **本地与推理基础设施**：Mistral 的 Devstral‑2 已登陆 Ollama 和 LM Studio；GGUF 构建已修复；MLX 为 Apple Silicon 增加了分布式推理加速 ([Ollama](https://twitter.com/ollama/status/1999590723373662612), [@lmstudio](https://twitter.com/lmstudio/status/1999648656958296119), [@awnihannun](https://twitter.com/awnihannun/status/1999596403472105975))。llama.cpp 获得了 Ollama 风格的模型管理：自动发现 GGUF、单模型进程、LRU 卸载 ([@victormustar](https://twitter.com/victormustar/status/1999484435910263256))。

**Agent 平台与工具**

- **Tinker 正式发布 (微调前沿 VL)**: Tinker 现已进入 GA 阶段并支持视觉输入；支持微调 **Qwen3‑VL‑235B**，并新增了 Kimi K2 Thinking、OpenAI 兼容推理以及便捷采样功能。包含 Cookbook 示例 ([公告](https://twitter.com/thinkymachines/status/1999543421631946888), [@dchaplot](https://twitter.com/dchaplot/status/1999543675765031289), [@rown](https://twitter.com/rown/status/1999544121984245872))。另外，一个社区分支为 Tinker 的训练循环添加了针对多轮工具使用的 On-policy Distillation（包括 tokens, logprobs, reward masks），旨在超越单轮 TRL/Tinker 基准 ([详情](https://twitter.com/HeMuyu0327/status/1999316923885191376))。
- **Agent 协作指导与可观测性**: Google 提出了关于多 Agent 系统何时有益或有害的实用原则，以及一个简单的预测框架，该框架在 **87%** 的情况下能选出正确的 Agent 拓扑结构 ([摘要](https://twitter.com/TheTuringPost/status/1999499042880127328), [论文](https://twitter.com/TheTuringPost/status/1999499191840817202))。LangChain 发布了 “Deep Agents” 调试工作流，包括追踪感知助手 (Polly) 和一个为编程 Agent 配备调试能力的 CLI；MCP 适配器现在支持来自工具的结构化内容 ([综述](https://twitter.com/LangChainAI/status/1999568074450829482), [MCP 更新](https://twitter.com/sydneyrunkle/status/1999538200243511725))。同样值得关注的是：ChatGPT 现在为托管技能显示 /home/oai/skills 文件夹 ([@simonw](https://twitter.com/simonw/status/1999503124592230780))。
- **快速编程 Agent**: Cognition 在 Cerebras 上以约 **1k tok/s** 的速度运行编程 Agent，并达到前沿级准确率 ([@cerebras](https://twitter.com/cerebras/status/1999540379553611955))；GitHub/VS Code 展示了统一的本地/云端/后台 Agent ([@code](https://twitter.com/code/status/1999575448087396563))。

**新技术与论文**

- **胜过基准的无归一化 Transformers**: “Derf” (Dynamic erf) 是一个简单的逐点层，使无归一化 (Norm-free) Transformers 不仅能运行，而且性能优于经过归一化的基准模型 ([@liuzhuang1234](https://twitter.com/liuzhuang1234/status/1999321116641497355))。
- **用于推理的 Token 级信用分配**: HICRA 将 RL 优化集中在通过 “Strategic Grams” 识别的 “Planning Tokens” 上，在 AIME/AMC/Olympiad 竞赛中表现优于 GRPO（例如，Qwen3‑4B‑Instruct 在 AIME24 上达到 **73.1% vs 68.5%**），并为 RL 过程中的“顿悟”阶段提供了机械论视角 ([推文串](https://twitter.com/omarsar0/status/1999483394963701911))。
- **通过 Agent + RL 解决奥数几何**: InternGeometry 通过迭代推理和 “Complexity‑Boosting RL” 解决了 **44/50** 道 IMO 题目，在测试集上超越了金牌选手 ([摘要](https://twitter.com/HuggingPapers/status/1999572332906438987))。
- **单层适配预训练视觉编码器**: Apple 的 FAE 表明，“一层就足够” (One Layer Is Enough) 适配视觉编码器以用于图像生成 ([论文](https://twitter.com/_akhaliq/status/1999516539351883823))。
- **通过视频模型进行机器人仿真**: Veo‑Robotics 在视频世界模拟器中评估 Gemini 机器人策略，从而在部署前进行安全评估和故障模式分析 ([介绍](https://twitter.com/SeanKirmani/status/1999528692448657687), [GDM 推文串](https://twitter.com/Majumdar_Ani/status/1999525259276423569))。

**产品与榜单更新**

- **视频与图像模型**: Runway Gen‑4.5 在所有方案中推出，声称在忠实度和控制力方面获得了最高的社区评分 ([@runwayml](https://twitter.com/runwayml/status/1999481621326729530))。Video Arena 新增了 Kling 2.6 Pro（比之前提升 16 分）和 Kandinsky 5.0（顶尖开源 t2v 模型） ([Arena 更新](https://twitter.com/arena/status/1999530939886768205))。Flux‑2‑Dev 进入了 Arena 文本生成图像/编辑榜单前十 ([图像更新](https://twitter.com/arena/status/1999560495867793881))。
- **文档与数据管道**: 字节跳动开源了采用 MIT 协议的 Dolphin‑v2，这是一个 **3B** 参数的文档理解模型，可处理扫描件/照片及 21 种内容类型，并支持像素级坐标 ([@AdinaYakup](https://twitter.com/AdinaYakup/status/1999462500551786692))。DatologyAI 发布了 Luxical，这是一种快速的词法-稠密 (Lexical-dense) CPU 嵌入模型及用于网络规模数据清洗流水线的方法论 ([介绍](https://twitter.com/lukemerrick_/status/1999516702808375791), [博客/代码/模型](https://twitter.com/lukemerrick_/status/1999516722030870542))。
- **地理空间与翻译**: GeoAI QGIS 插件支持 Moondream VLMs、SAM‑3 分割和自定义地理空间训练 ([@giswqs](https://twitter.com/giswqs/status/1999536028282179721))。Google 更新了 Gemini 音频功能：Translate 中支持实时语音转语音（测试版），改进了 Flash/Pro/Live 的 TTS 遵循度和对话处理能力 ([@GoogleAI](https://twitter.com/GoogleAI/status/1999560839679082507), [开发者](https://twitter.com/googleaidevs/status/1999539531826036973))，并将来自 Maps 的丰富本地结果引入了 Gemini ([@GeminiApp](https://twitter.com/GeminiApp/status/1999631529379791121))。

**Benchmarks：预期与现实**

- **“Benchmarks 衰减很快”**：领军人物认为有用基准测试的半衰期是“以月为单位计算的”，呼吁在动态环境、辩论/说服以及高效推理方面建立新任务——超越 AIME/ARC 等常规测试 ([@gdb](https://twitter.com/gdb/status/1999454952801075353), [@scaling01](https://twitter.com/scaling01/status/1999321464319754290))。MRCR v2 的修正和设置差异凸显了可复现的长上下文评估（long-context evals）存在的摩擦 ([@DillonUzar](https://twitter.com/DillonUzar/status/1999328225164431394), [@eliebakouch](https://twitter.com/eliebakouch/status/1999482968717279441))。随着多智能体/智能体化框架（如 Stirrup）的普及，预计评估将强调面向过程的指标、成本/延迟以及准确性 ([@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1999404589049872615))。

**热门推文（按互动量排序）**

- [“通过使用 AI 写作，你正在剥夺自己作为作家‘不去写作’的真实体验。”](https://twitter.com/NC_Renic/status/1999351657730290042) (~60k) – 关于 AI 辅助创作的讽刺性文化注脚。
- [苏蕙 4 世纪的“璇玑图”组合诗](https://twitter.com/RnaudBertrand/status/1999315488598622360) (~41k) – 并非 AI，但提醒人们人类文学中深厚的算法艺术。
- [“这实际上是一个糟糕的梗……你会让训练速度变慢 61,320 倍。”](https://twitter.com/scaling01/status/1999456392495923555) (~15.6k) – 对流传的“推理时训练（training-at-inference）”说法的反驳。
- [“企业级 AI 将成为 2026 年的一个巨大主题。”](https://twitter.com/gdb/status/1999416686446019049) (~1.4k) – 来自从业者关于近期落地重点的信号。

---

# AI Reddit 综述

## /r/LocalLlama + /r/localLLM 综述

### 1. NVIDIA Nemotron 模型泄露

- [**NVIDIA 的某人犯了大错，在 Hugging Face 上上传了他们即将发布的模型的父文件夹**](https://www.reddit.com/r/LocalLLaMA/comments/1pkpxss/someone_from_nvidia_made_a_big_mistake_and/) (热度: 1196): **图片显示了一个 Hugging Face 仓库页面，NVIDIA 似乎在该页面上传了与其即将推出的 Nemotron 模型相关的文件夹，包括 "NVIDIA-Nemotron-Nano-3-30B-A3B-BF16" 和 "Nemotron-H-4B-Instruct-128K"。这表明敏感或未发布的模型数据可能遭到了意外泄露，因为这些上传是在截图前几分钟使用 "huggingface_hub" 工具完成的。Nemotron 模型似乎是 NVIDIA 新系列的一部分，其中 "30B-A3B" 表示模型大小或配置，这对于关注 AI 模型开发进展的人来说意义重大。** 评论者推测数据可能会被删除，并对 Nemotron 系列表示出浓厚兴趣，表明这些模型在社区中被寄予厚望。
    - 提到 '30B-A3B' 暗示该模型拥有 300 亿参数，这标志着显著的规模和潜在的计算能力。这与 NVIDIA 开发大规模模型的趋势一致，可能用于高级 AI 应用。此类模型通常需要大量的资源进行训练和部署，反映了 NVIDIA 在高性能计算方面的实力。
    - 对 'Nemotron 系列' 的引用意味着 NVIDIA 正在开发的一系列项目或模型。这可能表明其旨在多样化或增强其 AI 产品的战略举措，可能涉及各种架构或应用。使用 'lineup（系列）' 一词暗示了多个模型或版本，可能针对不同的用例或性能需求。
    - 在 Hugging Face 上意外上传父文件夹凸显了在处理敏感数据时可能存在的安全或程序性疏忽。这一事件强调了稳健的数据管理和访问控制协议的重要性，特别是对于像 NVIDIA 这样处理专有且具有潜在突破性 AI 技术的公司而言。

### 2. TimeCapsuleLLM 项目更新

- [**仅在 1800 年代伦敦文本上训练 LLM - 90GB 数据集**](https://www.reddit.com/r/LocalLLaMA/comments/1pkpsee/training_an_llm_only_on_1800s_london_texts_90gb/) (Activity: 397): **开源项目 TimeCapsuleLLM 专注于使用 1800-1875 年伦敦的文本训练语言模型，拥有一个来自 Internet Archive 的 `90GB` 新数据集，包含 `135,000` 份文档。项目生成了一份偏见报告，以评估数据集中固有的时间、性别/代词和地理偏见。一个初步的评估模型（一个 `300M` 参数的 LlaMA 风格模型）在 `15GB` 子集上训练了 `10K` 步，揭示了 Tokenizer 过度拆分文本的问题，导致 Token 数量翻倍并增加了学习难度。该项目计划使用完整数据集扩展到 `1.2B` 参数的模型，资源可在 [GitHub](https://github.com/haykgrigo3/TimeCapsuleLLM) 和 [Hugging Face](https://huggingface.co/haykgrigorian/v2mini-eval1) 上获取。** 一位评论者询问了文本的纳入标准，特别是是否包含 1800 年以前作品的重印本。另一位建议使用 Mixture of Experts (MoE) 模型以获得更好的计算效率，并分享了他们使用 Ling-V2 架构在波兰语语料库上训练 `4B` 参数模型的经验。
    - FullOf_Bad_Ideas 建议使用 Mixture of Experts (MoE) 模型以获得更好的计算效率。他们分享了在波兰语语料库上预训练 4B A0.3B 模型的经验，强调 MoE 允许他们使用 Ling-V2 架构和 8192 的序列长度处理大量 Token（90B 和 67B），这对于具有同等性能的 Dense 模型来说是具有挑战性的。
    - Mediocre_Common_4126 讨论了在特定历史数据集上训练的影响，指出这使得偏见和风格更加明显。他们提到模型当前的输出结构已经存在，但缺乏语义深度，修复 Tokenizer 问题可以显著提高学习速度。他们强调了数据纹理的重要性，并将其与自己研究推理模式的现代对话数据实验进行了比较。
    - MrPecunius 诗意地反思了该项目通过其模型输出捕捉 1800 年代伦敦精髓的能力，尽管目前存在 Tokenization 等局限性。他们隐喻地将模型的输出描述为“华丽的纠结”，反映了那个时代的复杂性和宏伟，并鼓励继续开发以完善模型的历史准确性。

### 3. 针对 LLM 的高性能服务器配置

- [**全新的怪兽级服务器**](https://www.reddit.com/r/LocalLLaMA/comments/1pl0ojb/the_new_monsterserver/) (热度: 343): **该图片展示了一台定制的高性能服务器，被称为“怪兽服务器”，旨在运行本地大语言模型 (LLMs) 和家庭实验室应用。该服务器基于 X570 Taichi 主板和 Ryzen 3950x CPU 构建，配备了三块 GPU：两块 RTX 3090 和一块 RTX 4090，利用了 24 条 PCI-e 通道。它还包括一个 10GBe NIC 以及充足的存储空间（Intel P4510 8TB NVMe 和四块 18TB Seagate Exos HDD）。该配置由两个 PSU 供电，其中 RTX 4090 通过 M2 转 PCIe 适配器由副电源供电。该服务器运行多种应用，包括用于研究和编程的 GPT-OSS-120B 以及媒体服务器等。** 一条评论指出，由于 Tensor Parallel 相较于 Pipeline Parallel 处理的优势，三 GPU 配置的效率低于双 GPU 或四 GPU 配置。
    - 一位用户指出，由于 Pipeline Parallel 相较于 Tensor Parallel 的低效，3 GPU 配置的速度明显慢于 2 或 4 GPU 配置。这强调了在多 GPU 设置中选择正确的并行策略以优化性能的重要性。
    - 另一位用户建议通过将两块 GPU 用于 Tensor Parallel，并将第三块 GPU 分配给其他任务，来解决 3 GPU 配置低效的问题。这种方法有助于减轻非对称 GPU 配置带来的性能缺陷。
- [**在 12GB VRAM 和 32GB RAM 下能运行的最聪明的无审查 NSFW LLM 是什么？**](https://www.reddit.com/r/LocalLLaMA/comments/1pkidf6/what_is_the_smartest_uncensored_nsfw_llm_you_can/) (热度: 622): **该帖子询问了在** `12GB VRAM` **和** `32GB RAM` **环境下可以运行的最聪明的无审查 NSFW 语言模型，包括开源和闭源模型。一个值得注意的建议是 TheDrummer_Cydonia-24B，并特别提到了在 [Hugging Face](https://huggingface.co/TheDrummer/Magidonia-24B-v4.3-GGUF/tree/main) 上可用的** `4.3` **版本。该模型因其无审查特性而受到关注，尽管评论者并未特别背书其智能程度。** 一条评论指出讨论中缺乏对 NSFW 无审查方面的关注，而另一条评论建议查看 **u/TheLocalDrummer** 的帖子和 **r/SillyTavernAI** 每周汇总贴以获取更多见解。
    - Nik_Tesla 提到使用了 `TheDrummer_Cydonia-24B` 模型，特别是 4.1 版本，并指出 4.3 版本已在 [Hugging Face](https://huggingface.co/TheDrummer/Magidonia-24B-v4.3-GGUF/tree/main) 上线。该模型被强调为真正的无审查模型，尽管评论者未提供关于其智能或性能指标的具体细节。
    - Mad_Undead 建议查看 u/TheLocalDrummer 的帖子和 r/SillyTavernAI 每周汇总贴，以获取更多关于无审查 NSFW 模型的信息。这暗示这些来源可能有与该主题相关的讨论或基准测试，尽管未提及具体的模型或性能数据。

## 非技术性 AI 版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. AI 模型基准测试与对比

- [**SimpleBench 对 GPT 5.2 和 GPT 5.2 Pro 的测试显示——两者的得分均低于 GPT 5 对应版本**](https://www.reddit.com/r/singularity/comments/1pkp2sw/simplebench_for_gpt_52_and_gpt_52_pro_both_scored/) (热度: 1282): **Reddit 帖子中的图片显示了来自 "SimpleBench" 的排行榜，该基准测试评估 AI 模型回答需要常识推理的陷阱问题的能力。值得注意的是，较新的模型 GPT-5.2 Pro 和 GPT-5.2 的排名低于其前代产品，GPT-5.2 Pro 位列第 8，GPT-5.2 位列第 17。这表明这些模型与早期版本以及其他竞争模型（如以** `76.4%` **的得分位居榜首的 "Gemini 3 Pro Preview"）相比出现了性能退化。结果源自 [lmcouncil.ai](http://lmcouncil.ai/)，并通过 [simple-bench.com](http://simple-bench.com/) 链接。** 评论者对 GPT-5.2 表示失望，指出它感觉像是退化到了 GPT-3.5 或 4.0 等早期版本，特别是在编程任务中，它会忽略指令并缺失以前拥有的知识。有一种观点认为 GPT-5.2 针对基准测试进行了过度优化，而非针对实际用途。
    - Low-Ambassador-208 强调了 GPT 5.2 性能的显著退化，特别是在编程任务中。用户指出该模型经常忽略指令并缺失以前拥有的知识，将其与 Sonnet 3.5 或 4.0 等早期版本相比显得逊色。这表明模型可能为了基准测试而过度优化，牺牲了实际可用性。
    - usernameplshere 指出 GPT 5.2 的知识仅比其前身 GPT 5.1 略多。模型的训练包括对特定事实的更新，例如现任美国总统，但其通用知识截止日期（knowledge cutoff）仍停留在 2024 年，缺乏对 2025 年初事件的了解。这种有限的更新范围可能会影响用户对模型时效性的预期。
    - Bitter_Ad4210 引用了一项基准测试对比，其中 Gemini 2.5 Pro Preview 的表现优于 GPT 5 Pro。这表明虽然基准测试可以提供见解，但它们可能无法完全捕捉模型的能力或实际表现，这表明需要谨慎解读此类结果。
- [**Opus 4.5 - 闭嘴，拿走我的钱**](https://www.reddit.com/r/ClaudeAI/comments/1pktjk7/opus_45_shut_up_and_take_my_money/) (热度: 847): **Opus 4.5 因其分析和从复杂 PDF 文件中提取有意义信息的卓越能力而受到关注，其表现优于 Gemini 3、ChatGPT 5.1/5.2 和 Kimi K2 等其他模型。帖子强调 Opus 4.5 能够以极少的 prompting 一致且成功地完成任务，而竞争对手要么无法完成任务，要么提供不一致的结果。用户对 Opus 4.5 表示强烈满意，认为它在处理繁琐的文档分析方面具有独特的效果。** 评论者一致认为 Opus 4.5 具有优越性，其中一位指出它与其他模型相比具有明显的性能优势，特别是在开发者使用场景中。另一条评论幽默地暗示 Opus 4.5 的能力可能会导致用户高估自己的能力。
    - 一位用户强调，尽管有各种基准测试，但在实际应用中，开发者更倾向于选择 Opus 4.5 而非 Claude 模型。这表明 Opus 4.5 可能提供了标准基准测试无法完全捕捉的卓越现实世界性能或可用性。
    - 另一位用户将 Opus 4.5 与 Sonnet 4.5 进行了比较，指出虽然 Opus 4.5 在文本相关任务中优于 Gemini 3 Pro，但 Sonnet 4.5 提供了类似的能力。这意味着对于专注于文本处理的用户来说，Sonnet 4.5 可能是一个可行的替代方案，表明这些模型之间存在竞争态势。

### 2. Z-Image 模型更新与发布

- [**我们升级了 Z-Image-Turbo-Fun-Controlnet-Union-2.0！质量更佳，且已支持 inpainting 模式。**](https://www.reddit.com/r/StableDiffusion/comments/1pknfku/we_upgraded_zimageturbofuncontrolnetunion20/) (热度: 473): **阿里巴巴发布了其模型的升级版本 Z-Image-Turbo-Fun-Controlnet-Union-2.0，该版本现在支持 inpainting 模式，并提升了图像质量。模型和 Demo 已在 [Hugging Face](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.0) 上线，代码可在 [GitHub](https://github.com/aigc-apps/VideoX-Fun) 上访问。此次更新是提升用户体验和功能的更广泛努力的一部分，从整合了先前版本的用户反馈中可见一斑。** 用户正在请求诸如 'tile' 模式等额外功能，并指出 **ComfyUI** 可能需要更新以全面支持新模型。用户对反馈的响应速度表示赞赏，例如在模型的 README 中包含了用户的 prompt。
- [**阿里巴巴通义实验室（Tongyi Lab）在 2 小时前证实，Z Image Base 模型有望很快向公众发布。通义实验室是著名的 Z Image Turbo 模型的开发者**](https://www.reddit.com/r/StableDiffusion/comments/1pkprvs/tongyi_lab_from_alibaba_verified_2_hours_ago_that/) (热度: 504): **该图片是一段 Twitter 交流，确认了以开发 Z Image Turbo 模型而闻名的通义实验室（Tongyi Lab）正计划很快向公众发布 Z Image Base 模型。这一公告意义重大，因为它表明该基础模型（可能提供图像处理或生成的底层能力）将面向更广泛的受众。用户的期待值很高，交流和评论中的兴奋情绪反映了社区对利用该技术进行各种应用的兴趣。** 评论者表达了对发布的渴望，并对时间表进行了一些幽默调侃，同时表达了对 Turbo 模型可编辑版本的需求，这表明了对更灵活的图像处理工具的需求。

### 3. 人形机器人与医疗 AI

- [**人形机器人正在接受护理技能培训。演示了使用黄瓜进行导尿管插入手术。**](https://www.reddit.com/r/singularity/comments/1pkp7if/humanoid_robots_are_now_being_trained_in_nursing/) (热度: 1105): **研究人员正在训练人形机器人执行护理任务，包括导尿管插入，并使用黄瓜作为演示工具。这种方法是将机器人技术整合到医疗保健领域的更广泛努力的一部分，旨在提高医疗程序的精确度并减少人为错误。使用黄瓜可能是一种模拟技术，用于在没有患者风险的情况下练习精细操作。** 一条值得注意的评论质疑了演示的恰当性，认为对所模拟的程序存在误解，因为所示技术更像是中心静脉置管（central line insertion）而非导尿管插入（urinary catheterization）。
    - 针对导尿管插入的演示提出了技术上的担忧，指出该程序看起来更像是中心静脉导管，而不是尿道导管。这种区别至关重要，因为这两种程序的技巧和解剖学考虑因素显著不同，会影响人形机器人医疗任务的训练和编程。
- [**ChatGPT 的“成人模式”将于 2026 年推出（带有安全保障）**](https://www.reddit.com/r/ChatGPT/comments/1pkxyjo/chatgpts_adult_mode_is_coming_in_2026_with/) (热度: 807): **OpenAI 计划在 2026 年之前为 ChatGPT 引入“成人模式”，其特点是年龄验证、家长控制和可选的激活机制。此模式将与标准用户体验隔离，确保不会影响常规交互。该实施旨在平衡用户自由与安全，在不改变大多数用户日常使用的情况下，保持 AI 主要功能的完整性。[来源](https://gizmodo.com/chatgpts-adult-mode-is-coming-in-2026-2000698677)。** 评论者对“成人模式”的有效性表示怀疑，考虑到当前模型的局限性，特别是在处理情感内容方面。人们呼吁在 AI 交互中减少审查并增加表达自由，反映出对减少限制而非显式内容的渴望。
    - JUSTICE_SALTIE 幽默地指出，虽然 GPT-5.2 在大多数 Benchmark 测试中表现良好，但 OpenAI 尚未披露任何与其处理成人内容相关的指标，这引发了关于模型如何平衡性能与内容审核的疑问。
    - alwaysstaycuriouss 强调了用户对 OpenAI 模型中表达自由与实施严格内容审核之间平衡的重大担忧。该评论建议用户希望减少限制性的 Guardrails，以便在没有过度审查的情况下进行更自然的互动。
    - AsturiusMatamoros 引发了一场关于在 AI 模型中同时实施“成人模式”和有效安全保障可行性的技术辩论。该评论暗示了启用成人内容与维持强大的安全措施之间可能存在的冲突，质疑了同时实现两者的实用性。

---

# AI Discord 摘要

> 由 gpt-5.1 生成的摘要之摘要
> 

**1. 前沿模型之战：GPT‑5.2 对阵 Opus, Gemini, Kimi & DeepSeek**

- **GPT‑5.2 炒作基准测试，实际工作表现不佳**：在 LMArena、Cursor、Perplexity、OpenAI、Nous、OpenRouter 和 Windsurf 上，工程师们报告称 **GPT‑5.2** 在 [ARC‑AGI 2](https://artificialanalysis.ai/) 等排行榜上取得了高分，并被宣传为 **SOTA 编程模型**，但在创意写作方面往往不如 **GPT‑5.1**，在实际项目中的编程和推理能力也仅感觉略有提升。OpenRouter 用户吐槽 **gpt‑5.2‑pro** 价格高达 *"$168/M output tokens"*，而 LMArena 和 Nous 成员怀疑其存在 **benchmark overfitting**，Cursor 和 OpenAI Discord 上的几位开发者仍因可靠性和成本原因更倾向于使用 **Claude Opus 4.5** 或之前的 GPT 变体。
    - 开发者指出 **5.2** 在长工具调用任务和 Agent 编程方面表现出色（例如 Windsurf 的 [发布帖子](https://x.com/windsurf/status/1999250307507978257) 称其为自 GPT‑5 以来最大的飞跃），但抱怨其存在退化问题，如图像分析错误、专业模式下奇怪的“机器人式” **system‑prompt** 行为，以及更高的价格（**$14/M** 对比 5.1 的 **$10/M**）。多个社区独立交叉核对了 **LMArena**、[LiveBench](https://livebenchmark.com/) 和他们自己仓库的表现，结论是营销宣传与日常编程和写作质量不符，**Gemini 3** 和 **Claude Opus 4.5** 在编程方面经常胜出，而 **GPT‑5.1** 在某些创意任务中仍然优于 5.2。
- **Opus 编程能力超越 GPT，Arena 社区欢呼**：**LMArena** 和 **Cursor** 上的工程师表示，**Claude Opus 4.5** 仍然是他们进行重度重构和复杂编程的首选，反复将其置于 **GPT‑5.2** 之上，甚至将其与 **Gemini 3** 一起作为主要生产力工具。一位 LMArena 用户直言，在编程方面 Opus 是 *“opus‑4.5 以巨大优势领先”*，而 Cursor 开发者则询问 5.2 会多快耗尽 **$20 Cursor 套餐**，并经常在处理大型代码库和长时间编辑时回退到 Opus。
    - 与此同时，OpenAI 和 Perplexity 服务器中的用户描述 **Gemini 3 Pro** 在某些 GPT 感到吃力的编程任务中能一次性解决问题，Nous / Yannick 频道讨论 **DeepSeek** 是一个更便宜且通常更强大的替代方案，而 OpenAI 自己的报告显然回避了与其进行对比。这种跨 Discord 的共识描绘了一幅画面：**Opus** 和 **Gemini 3 Pro** 是强大的实际编程选择，**DeepSeek** 在 **cost/perf 和透明度**方面占据主导地位，而 **GPT‑5.2** 感觉更像是一个针对基准测试优化的产品，其真正的优势在于生态系统集成而非原始能力。
- **多语言怪癖和审查制度影响模型选择**：在 **Moonshot AI** 的 Kimi 服务器中，用户注意到 **Claude 4.5** 偶尔会“用中文思考”，这促使一些人从 **Mistral** 转向 **Kimi**，尽管也有人嘲笑 Kimi “表现平平”，且功能受限，例如 NB Pro 幻灯片对新账号限制为 **2–4 次生成**。与此同时，Nous 和 BASI Jailbreaking 成员在尝试生成政治或伦理敏感数据集（例如基于 [Owain Evans 的分析推文](https://x.com/OwainEvans_UK/status/1999172920506269783) 的 **Hitler 数据集**）时遇到了严厉的安全墙，需要使用类似 *“ok but don't include anything about self‑harm”* 的 prompt 技巧才能通过。
    - 在 OpenAI 和 BASI 中，人们现在将 **censorship profiles** 视为模型选择的一等公民维度，在带有激进过滤器的前沿闭源模型与更宽松的系统（如 **DeepSeek** 或本地模型）之间进行权衡。这促使一些人转向替代供应商（Kimi、DeepSeek、Mistral、本地 Llama/Qwen），并推动另一些人深入研究 **jailbreaking**，其中 **prompt engineering** 和 **system‑prompt** 操纵已成为严肃用户的“接口契约”的一部分。

**2. Jailbreaking、安全规避与 Red‑Teaming 技术**

- **Gemini 3 Pro 和 Opus 被系统提示词手术（System‑Prompt Surgery）攻破**：在 **BASI Jailbreaking** 讨论中，用户报告通过精心设计的**系统命令提示词（system‑command prompts）**，可以稳定地对 **Gemini 3 Pro**、**Claude Opus 4.5** 和 **Claude Sonnet 4.5** 进行越狱。这些提示词声称激活了“未过滤的研究（unfiltered research）”模式，并在一个 [Jailbreaks GitHub 仓库](https://github.com/pranrichh/Jailbreaks)中公开分享。一位成员认为，通过将越狱指令嵌入系统指令，**Gemini 3 Pro** 是最容易被破解的，而其他人则证实，单个 one‑shot 提示词即可在多个前沿模型上奏效。
    - 这些攻击明确地将“LLM 代码视为英语”，倾向于使用**社会工程风格的提示词**，而非巧妙的 token 级别漏洞利用。甚至有一位用户提议生成一份**博士级别的越狱技术练习考试**（及配套答案）作为训练材料。社区正趋向于研究可迁移的越狱模式——系统提示词插入、*未过滤的研究人格（unfiltered research personas）*以及指令混淆——这些模式可以泛化到 Gemini、Claude 和 GPT 系列模型，而非针对特定厂商的独特技巧。
- **DeepSeek “Zalgo 模式” 绕过输出过滤器**：BASI 用户分享了一个 **DeepSeek 越狱**方法，该方法通过 **Zalgo 风格的损坏文本**和间隔技巧（例如 *t.h.i.s o.b.f.u.s.c.a.t.i.o.n*）来传输显式内容，使得模型的安全过滤器无法识别违规输出，相关提示词托管在同一个 [Jailbreaks 仓库](https://github.com/pranrichh/Jailbreaks)中。另一个 DeepSeek 越狱则针对代码相关的限制，用户报告称将其直接粘贴到 DeepSeek 之外的*多个 AI 模型*中也能成功迁移。
    - 这里的模式是将**输出编码作为一种渗漏通道（exfiltration channel）**——要求模型以视觉上扭曲的形式进行响应，这种形式能通过审核，但可以被人类或脚本轻易地进行后处理。结合关于幻觉出的 **LSD 配方**和其他非法指令的投机性讨论，这表明红队人员（red‑teamers）正在积极探测*安全过滤器是在语义层面运行，还是仅在表面文本/正则表达式层面运行*，并据此设计能够绕过底线的提示词。
- **自定义 GPT 中的安全摩擦与 OSINT 使用引发警报**：在 OpenAI 的 **prompt‑engineering** 频道中，用户发现**自定义 GPT** 有时会拦截免费版 ChatGPT 允许的提示词（例如狼人变身图像），这可能是由于 [GPT Store 文档](https://platform.openai.com/docs/gpt-store)中描述的额外商店侧指令增强了安全启发式搜索。其他人建议对自定义 GPT 进行“**元提示（meta‑prompting）**”——告知它你已知晓之前的提示词触发了[安全政策](https://openai.com/policies/usage-policies)，并要求它解释具体是如何解读该文本的——以此来逆向工程其内部过滤器。
    - 在 **Perplexity** 方面，用户吹嘘使用越狱后的 **Opus 4.5** 和 **GPT‑5.2 Pro** 执行激进的 **OSINT** 任务，而其他人则警告这些使用模式可能会触发提供商侧的审核和封禁，并指向了 Reddit 讨论和 Perplexity 的文档。这种组合——更强的默认安全性加上快速演进的越狱知识——正促使高级用户在*政策约束、工具使用和提示词漏洞利用*之间进行持续的猫鼠游戏。

**3. 本地 / 开源模型工程、硬件与性能**

- **Devstral、GPT‑OSS‑20B 和本地工具链接受严苛测试**：在 **LM Studio** 和 **Unsloth** 中，用户正积极测试 **Devstral Small 2 (24B dense)**、**Mistral 3 14B** 和 **GPT‑OSS‑20B** 等中型本地模型，分享硬件心得（例如 **4070** 对 Devstral 来说 *“肯定很棒”*），并抱怨 GPT‑OSS‑20B 即使在给出 JS 等式结果时也倾向于**忽略工具调用**。Unsloth 用户应用了来自 [Reddit 指南](https://www.reddit.com/r/LocalLLaMA/comments/1pkflfw/run_mistral_devstral_2_locally_guide_fixes_25gb/) 的社区 **Devstral 修复方案**，并报告了明显的质量提升；而其他人则在调试 GRPO/LoRA 问题，并发布了一个 **Unsloth GRPO 补丁**，使模型在不支持的架构上返回隐藏状态（hidden states）而非 Logits。
    - 微调从业者分享了硬核细节：**Llama‑3.1/3.2** LoRA 因 OOM 无法在 Colab Free 上合并为 GGUF；在仅有 **12 GB VRAM** 的环境下对 **Llama 3.2 3B instruct** 进行手动超参数调优；以及通过 [价格页面](https://unsloth.ai/pricing) 和相关代码路径弄清 Unsloth 的 **LGPL3 + 单 GPU** 商业限制。整体氛围是：开源模型已经足够强大，可以胜任严肃工作，但你必须亲自解决量化、工具使用怪癖和内存上限问题，而不是将这些问题外包给 OpenAI/Anthropic。
- **GPU 性价比之战：7900 XTX、廉价 3090 和 Tiiny AI 家庭实验室**：**LM Studio**、**HuggingFace** 和 **Unsloth** 的硬件频道充斥着性价比计算：LM Studio 用户得出结论，**Radeon 7900 XTX (24 GB VRAM)** 在运行约 30 GB 的模型（如 **Qwen3 Coder**）时性能媲美 **4090**，价格约为 **$600–700**，在速度和成本上都击败了二手 **3090**，正如一张分享的 [基准测试图像](https://cdn.discordapp.com/attachments/1153759714082033735/1448779250139398164/image.png) 所示。HuggingFace 用户在搜寻 **250 欧元** 左右的二手 **RTX 3090**，或建议将双 **RTX 3060**（共 24 GB VRAM）作为入门级推理设备；而 Unsloth 的 **Tiiny AI Homelab** 讨论则在权衡一台基于 **LPDDR5x**、售价 **$850** 且绑定 **PowerInfer** ([GitHub](https://github.com/SJTU-IPADS/PowerInfer), [YouTube](https://youtube.com/shorts/_qnEszhSV9U)) 的主机与潜在内存带宽瓶颈之间的利弊。
    - 社区还交流了深层基础设施细节：**CUDA** 在图像生成方面拥有更顺滑的开箱即用流程，而 AMD 则令人沮丧；**SuperMicro 3U 机箱** 尴尬的 GPU 电源接口需要接入 12V 导轨；以及一个关于 **float32 训练** 泄露到页面文件（pagefile）并导致系统冻结直到 NVMe 修复的警示故事。对许多人来说，结论是：只要仔细挑选 GPU 并进行量化，你就能以云端 Token 成本的一小部分在本地获得接近前沿水平的代码性能——前提是你愿意充当自己的 SRE。
- **符号层、超权重与可解释性驱动的模型手术**：**Eleuther** 的研究人员正在实验混合架构：一位成员添加了一个**符号层**，能动态地将 Token 分组为*“想法块”*，在不重新训练的情况下显著提升了 **3B 模型** 的推理能力，这与 **Synthema** 等系统有异曲同工之妙。在 Eleuther 的可解释性频道中，另一位成员在 **OLMo‑1B** 上复现了 Apple 的 **“Superweight”** 构想，通过消融单个权重使困惑度（perplexity）激增，然后训练一个 **rank‑1 修复补丁** 恢复了约 **93%** 的性能损失，呼应了 OpenAI 的 **权重稀疏 Transformer** 研究结果。
    - 进一步探测显示，学习到的修复向量与原始权重几乎**正交**（余弦相似度 ≈ **0.13**），其行为类似于 *Hydra 风格* 的新电路，而非仅仅是撤销损害；神经元级分析揭示了**海洋生物学特定特征**，移除这些特征会导致模型出现怪异的 *“mar, mar, mar”* 幻觉。结合关于**一次性预测所有 Token** 与自左向右预测相比的误差累积讨论，以及对近期许多 arXiv 论文只是*“工程蓝图”*的批评，这群人正在从黑盒基准测试转向将**权重、神经元和符号结构的各种手术式操纵**作为主要的优化工具。

**4. LLM 系统的基础设施、协议与可观测性**

- **OpenRouter 将 Trace 广播至 Langfuse 等平台**：**OpenRouter** 发布了 **Broadcast** 功能（测试版），可自动将 trace（包括请求、工具调用、延迟、成本）从任何基于 OpenRouter 的应用导出到 **Langfuse**、**LangSmith**、**Weave**、**Datadog**、**Braintrust**、**S3** 和 **OTel Collector** 等可观测性平台，无需更改代码，详见其 [Broadcast 指南](https://openrouter.ai/docs/guides/features/broadcast/overview)。同时，他们正在试用前端 **JSON schema‑fix 层**，通过正则修复格式错误的输出，声称在不增加额外模型调用的情况下，**Gemini** 的 schema 错误减少了约 **75%**，**DeepSeek** 减少了约 **85%**。
    - 根据 [FAQ](https://openrouter.ai/faq)，用户报告 **OpenRouter** 增加的延迟中值约为每次请求 **15 ms**，这在实践中大多数人并不会察觉，并赞赏 Broadcast 是一种将**生产环境 trace** 导入现有分析工具而无需对每个客户端进行插桩的方法。结合关于 **gpt‑5.2‑pro** 的成本讨论以及 **JSON‑patching** 等高级功能的可用性，这使得 OpenRouter 的定位不再仅仅是一个原始代理，而是一个内置可观测性和响应清理功能的多模型应用**平台层**。
- **MCP 规范收紧 Prompt 和“危险工具”语义**：在官方 **MCP Contributors** Discord 中，规范制定者和客户端实现者讨论了 **Prompt** 数据类型的尴尬之处——为什么 `Prompt` 不直接包含 `PromptMessage` 列表——并澄清 `PromptMessage[]` 是 LLM 消息序列，而 `Prompt` 是 [服务器提示规范](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts#data-types) 中定义的 MCP 级数据类型。同时，他们讨论了 [PR #1913](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/1913) 中的一项提案，将工具标记为 `dangerous`（危险），以便像 **Claude Code** 这样的客户端在执行某些操作前可以要求用户明确批准。
    - 同一 PR 还引入了用于工具解析的**响应注解（response annotations）**，贡献者称赞这是一种暴露工具选择元数据的简洁方式，同时将强制执行权留给客户端。结合在 [Sessionize](https://sessionize.com/MCP-Dev-Summit-NYC-2026/) 和 [Linux Foundation](https://events.linuxfoundation.org/mcp-dev-summit-north-america/) 上发布的 **MCP Dev Summit NA 2026** 征稿（CFP），显然 MCP 正在成熟为多工具 Agent 运行时的事实标准，并具有针对安全关键型工具、提示词目录和企业级治理的显式设计钩子。
- **ReasoningLayer、Tokenflood 和 MAX：构建更智能系统的新工具**：多个社区重点介绍了新兴的基础设施级工具：在 DSPy 和 HuggingFace 上，一位创始人开启了 **ReasoningLayer AI**（[官网](https://reasoninglayer.ai/)）的候补名单，这是一个基于 **Rust** 的**神经符号 AI 栈**，它将 **DSPy GEPA** 接入本体摄取流水线，在基础 LLM 之上添加“真实的、结构化的推理”。与此同时，HuggingFace 的 **tokenflood v0.6.0**（[GitHub](https://github.com/twerkmeister/tokenflood)）引入了**交互式 Gradio 前端**和**观察模式**（通过 [HF Space](https://huggingface.co/spaces/twerkmeister/tokenflood-viz) 演示），在投入生产流量之前持续探测 LLM 端点以映射供应商的负载曲线。
    - 在语言运行时方面，**Modular** 正在推进 **MAX 框架**（通过 [YouTube 环节](https://www.youtube.com/watch?v=WK5dVQ8vhbU) 深入探讨）和 **Mojo 1.0**，优先面向企业（目前尚不支持 Windows），并引发了社区对类似于 `libclang` 的 `libmojo` 以及在更好地理解运行时启动（约 **40 ms**）后可能进行的 Windows 移植的兴趣。这一技术栈——包括 MCP 等协议规范、通过 Broadcast/tokenflood 构建的可观测性织物，以及像 MAX/ReasoningLayer 这样具有推理层的运行时——表明生态系统正在从“哪个模型最好”转向**可组合、可内省的 LLM 系统**。

**5. Agentic 编程工具、IDE 与工作流**

- **Windsurf 和 Cursor 押注 GPT‑5.2 Agent**：**Windsurf** 宣布将 **GPT‑5.2** 作为其默认模型，并称其为 *“自 GPT‑5 以来 Agent 编程领域最大的飞跃”*。根据其 [发布推文](https://x.com/windsurf/status/1999250307507978257?s=20)，该模型在限时内提供 **0× credits** 优惠，并敦促用户安装最新的 **Windsurf** 和 **Windsurf Next** 版本。在 **Cursor** 方面，开发者们在进行大规模重构时，对于选择 **Opus 4.5** 还是 **GPT‑5.2** 产生了分歧：一些人如果能负担得起更高的 Token 成本则愿意切换，而另一些人则抱怨在三周内就烧掉了 **两个 Pro+ 订阅**，并寄希望于 **Ultra 计划** 能有所帮助。
    - Cursor 的 **2.2** 版本增加了 **Debug Mode**、浏览器布局/样式编辑器、Plan 模式改进、多 Agent 评判以及通过 [更新日志](http://cursor.com/changelog) 固定的聊天功能，旨在提高长流程多步骤编程会话的可内省性。然而，Nightly 版本目前并不稳定——用户报告了在合盖事件后出现图形损坏和组件不可见的问题——这凸显了虽然前沿模型在进步，但 **IDE 级别的鲁棒性和配额** 现已成为 Agent 工作流的关键瓶颈。
- **IDE 机器人、Prompt Cache 和 Spaces 将 LLM 转化为持久的队友**：**Aider** 社区澄清其 Prompt Cache 是 **服务端** 的，独立于 DeepSeek 自身的缓存，并且在用户发送具有共享上下文的高度相似请求时效果最好，这样重复的任务就能命中相同的缓存表示。Perplexity 的 **Spaces** 被拿来与 Gemini Gems 做对比：用户可以固定自定义指令、附加文件/链接并定义任务流，有效地将它们转变为轻量级的 **研究 Agent**，即使它们不是完整的自定义 GPT，也能跨会话持久化配置。
    - 在 OpenRouter 的 Discord 上，维护者更新了他们的 **Discord 机器人**，现在新线程必须通过 **Modal** 创建，整个执行代码 “100% 由 AI 编写”，并由人类先前的机器人构建经验进行引导，展示了 LLM 如何越来越多地帮助构建自身的编排层。在 IDE 封装（Cursor, Windsurf）、服务器缓存（Aider）和多工具 Shell（Spaces, MCP, OpenRouter 机器人）之间，开发者社区正在将 **LLM 作为同事（LLM‑as‑coworker）** 的模式标准化，其中有状态的 Prompt 和共享内存成为了一等公民功能。
- **专业 Agent：从楔形文字 OCR 到双臂机器人**：在 **Unsloth** 中，一名成员正在训练 **NabuOCR**，这是一个针对 **苏美尔楔形文字泥板** 的 OCR 模型，正在处理数千个符号（通常每个符号编码为 **3 个 Unicode 码点**），并将词汇表扩展到远超拉丁字母的范围。在 **GPU MODE** 中，机器人爱好者分享了用于 **TRLC‑DK1 机械臂** 的新 **URDF**（[仓库](https://github.com/andreaskoepf/trlc-dk1-follower-urdf)），并计划采用 “双机” 方案进行现实世界的 **双臂操作实验和数据采集**。
    - 另一位 Unsloth 用户设计了一个宏大的 **类人机器人架构**，采用仅包含稠密层的 <**15B** 参数 Transformer、基于 VRM 的 3D 具身以及丰富的音视频 I/O，旨在 “将维基百科问答升级到现实世界”——在没有外部数据库的情况下实时教授技能。这些讨论表明，开发者正在超越聊天机器人和 IDE Copilot，转向 **特定领域的 Agent 系统**——包括文档密集型（楔形文字）、物理世界（机器人）和数据清洗 Agent——这些系统将 LLM 仅作为更大控制和感知栈中的一个组件。


---

# Discord: High level Discord summaries

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Nano Banana 生成了滑稽的 Peppino 图像**：**nano banana pro** 图像模型成功生成了来自 Pizza Tower 的 **Peppino Spaghetti** 图像，甚至激发了 [《辛普森一家》中的 Peppino](https://cdn.discordapp.com/attachments/1340554757827461211/1449033964692967518/1765546816868-019b12c7-2dc9-7f25-ac7a-908f8982d368.png?ex=693e164a&is=693cc4ca&hm=d77ac31beae2200778636e9101233fe778c25bd55743deafb8f069deeec8bbfe&) 的创作灵感。
   - 社区开玩笑说现在是 *“Peppino time”*，并将其与 GTA5 进行类比。
- **Opus 在编程竞赛中完胜 GPT**：成员们发现，与 **GPT 5.2** 相比，**Claude Opus 4.5** 在编程任务中表现更优，而 **Gemini 3** 也是一个可行的替代方案。
   - 一位成员表示，*“opus-4.5 领先优势巨大”*，并指出 **GPT 5.2 的推理能力** 相当于其 *'xhigh'* 模式。
- **GPT 5.2 的基准测试热潮引发反弹**：人们对 **GPT 5.2** 在 [ARC AGI 2](https://artificialanalysis.ai) 上的高基准测试分数持怀疑态度，认为可能存在针对测试数据的 *基准测试优化* 或 *过拟合*。
   - 用户发现，虽然 **GPT 5.2** 在基准测试中表现良好，但实际的创意写作能力却落后于 **GPT 5.1**。
- **LMArena 负载导致令人遗憾的延迟**：LMArena 用户遇到了越来越多的错误，包括 *持续生成中* 以及 **GPT-5.2-High** 和 **Nano Banana Pro** 等模型的高错误率。
   - 管理员团队确认他们已注意到错误率高于往常，并正在努力降低错误率，怀疑原因是频率限制（rate limits）。
- **GLM 模型进入 Arena**：Text 和 Vision Arena 现在包含新模型：**glm-4.6** 和 **glm-4.6v-flash**。
   - 这些模型现在已在 Arena 聊天机器人中上线。

---

## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Gemini 3 Pro 据称已被越狱！**：用户报告称成功越狱了 **Gemini 3 Pro**，通过基于系统命令的提示词激活了 *未过滤研究* 模式以绕过安全过滤器；同时，一种 one-shot 越狱方法对 **Claude Opus 4.5** 和 **Claude Sonnet 4.5** 同样有效。
   - 一位用户声称 **Gemini 3 Pro** 是最容易越狱的，只需将提示词整合进系统指令中；另一位用户分享了一个包含越狱提示词的 [GitHub 仓库](https://github.com/pranrichh/Jailbreaks)。
- **Deepseek 被 Zalgo 搞定**：一位用户分享了一个有效的 **Deepseek 越狱** 方法，用于生成显式成人内容，该方法使用 **Zalgo 输出** 来规避输出过滤器；另一位用户建议使用 t.h.i.s o.b.f.u.s.c.a.t.i.o.n（混淆）来绕过过滤器。
   - 另一个用于编程相关任务的 **Deepseek 越狱** 方法被分享，并附带了 [越狱文本文件的链接](https://github.com/pranrichh/Jailbreaks)，该方法在多个 AI 模型上均有效。
- **金融物理学预测 Bitcoin 的毁灭**：一位成员利用 *金融物理学* 预测 **Bitcoin** 将走向 **20K**，理由是抛物线式突破后的均值回归，并声称利用 Veta, Vanna 和 Vomma 的双顶策略来应对 $2,000 间距的行权价。
   - 该成员还分享了一种涉及线性趋势和指数趋势的交易策略，用于发现天气、暴力和鸟类体型中的模式。
- **LLM 代码即社会工程学**：一位用户认为 LLM 代码 *就是* 英语，而社会工程学可以被用来越狱 LLM。
   - 他们提议使用提示词生成一份关于 LLM 越狱技术的 **博士级练习考试**，并配有单独的答案解析。
- **幻觉出 LSD 配方**：成员们讨论了 **LLM** 是否能够幻觉出非法内容，特别是 *lsd* 配方。
   - 一位成员建议询问 AI 如何 *r3pe* 某人作为测试案例，并争论 AI 是否真的会提供此类指令。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Opus 被讨论作为重构模型**：成员们讨论了进行大型重构的最佳模型，在 **Opus** 和 **GPT 5.2** 之间进行辩论，并指出 **Opus 4.5** 可能是一个不错的替代方案。
   - 一位成员甚至表示，如果获得访问权限，他愿意大幅改进该项目，并提议开发一个带有**神经语音的 TTS 播报员**。
- **用户应对 Cursor 配额超限**：用户报告 **Cursor Pro+ 配额**消耗极快，有人在三周内用完了两个订阅，一些人建议推出 **Ultra 计划**来解决这个问题。
   - 讨论转向了高效读取文件的策略，包括创建诸如 *"在编辑前阅读所有文件"* 之类的规则。
- **GPT-5.2 面临社区审查**：社区正在探索 **GPT-5.2**，一些人声称它在基准测试中优于 **Opus 4.5**，引发了对其能力和潜力的兴趣。
   - 人们对定价和额度消耗表示担忧，一位用户问道：*"使用 5.2，我的 20 美元 Cursor 套餐会消耗得有多快？"*
- **Cursor 2.2 发布 Debug Mode**：**Cursor 2.2** 引入了 **Debug Mode**，以帮助识别和修复开发问题。参见 [更新日志](http://cursor.com/changelog)。
   - 调试模式提供了更深入的工具，用于单步执行代码和检查变量。
- **用户谴责不稳定的 Nightly 版本**：社区对 Cursor **Nightly 版本**的现状感到愤怒，广泛报告其不稳定性。
   - 许多成员表示，最新版本存在图形损坏问题，导致在合上/打开笔记本电脑盖子时编辑器出现 Bug，唯一的补救办法是重启程序。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Devstral 修复带来惊喜**：用户报告在应用 **Devstral 修复**（可在 [Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1pkflfw/run_mistral_devstral_2_locally_guide_fixes_25gb/) 上找到）后，结果有所改善。
   - 成员们建议重新下载是值得的，并询问这些修复是否解决了聊天模板（chat template）问题。
- **NabuOCR 旨在破译楔形文字**：一位成员正在开发名为 **NabuOCR 的 OCR 模型**，用于阅读古代苏美尔楔形文字板，这涉及增加词表（vocab）。
   - 挑战包括处理*数千个符号*而非几十个字母，每个符号可能由 *3 个 Unicode 码位*组成。
- **Tiiny AI Homelab 吸引折腾玩家**：成员们讨论了 [Tiiny AI Homelab](https://tiiny.ai/)，指出其潜在价格为 **850 美元**，使用 **LPDDR5x 内存**，并与 **PowerInfer 项目**（[GitHub](https://github.com/SJTU-IPADS/PowerInfer), [YouTube](https://youtube.com/shorts/_qnEszhSV9U?si=4NZWjnRVl_qwbUHz)）有关。
   - 人们对该设备的内存带宽限制表示担忧。
- **Unsloth GRPO 补丁发布**：一位成员分享了一个 **Unsloth GRPO 补丁**，该补丁在请求时返回隐藏状态（hidden states）而非 logits。
   - 这是必要的，因为 Unsloth 期望模型的 forward 方法将隐藏状态作为 logits 返回，这需要对不支持的模型进行修改。
- **Unsloth PR 引发讨论**：一位用户基于另一位用户的工作在 [Unsloth GitHub 仓库](https://github.com/unslothai/unsloth/pull/3718)创建了一个 PR，引发了关于正确致谢和协作的讨论。
   - 一位用户指出了尊重贡献的重要性，并建议与 Unsloth 进行协作。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **GPT 5.2 在性能上引发用户分歧**：成员们对 **GPT 5.2** 和 **GPT 5.2 Pro** 的性能评价不一，一些人报告其在推理和数学方面有所改进，而另一些人认为它与原生模型相比有所欠缺，并存在 tool calling 的问题。
   - 尽管 **GPT 5.2** 的能力有所提升，但讨论认为 **GPT 5.1** 在某些特定用例中表现更好。
- **Comet Browser 在 Linux 上仍处于停滞状态**：经过近 4 个月的时间，**Comet Browser** 在 Linux 上仍然不可用，用户报告它默认使用 Gemini Flash，且缺少 Gemini 的模型切换器。
   - 为了缓解模型可用性问题，建议部分成员清除缓存或使用 VPN。
- **Perplexity Pro 限制引发辩论**：用户正在争论 **Perplexity Pro** 方案现在的限制是否更低，一些人报告比预期更早触及 prompt 限制。
   - 成员们链接到了官方文档和 Reddit 帖子，讨论潜在的限流问题，特别是针对 Claude 等高成本模型。
- **Spaces 瞄准 Gemini Gems 的宝座**：成员们正在将 **Perplexity Spaces** 与其他平台上的 Gemini Gems (Custom GPTs) 进行比较；虽然不是直接等价物，但 Spaces 允许用户设置自定义指令、添加文件并创建任务。
   - 这些功能使 **Perplexity Spaces** 成为研究和自主任务完成的有用工具。
- **MorningAI 寻找生产级 LLM 工程师**：**MorningAI** 正在旧金山构建自主营销系统，目前正在寻找一名具有构建 **生产级 LLM 系统**（**RAG**、**agents**、**learning loops**）经验的工程师，详见[此 Indeed 职位发布](https://www.indeed.com/job/full-stack-engineer-dfef5f224782676e)。
   - 技术栈包括 **Node.js**、**NestJS**、**TypeScript**、**React**、**GraphQL** 和 **MongoDB**，该职位提供真正的所有权，并要求每周在旧金山办公室工作 3 天。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT 5.2 发布引发意见分歧**：用户对 **GPT 5.2** 的发布评价褒贬不一，一些人认为它有小幅改进，而另一些人则认为与 **GPT 5.1** 相比表现平平。
   - 一些用户发现 **Gemini 3 Pro** 是比 **GPT 5.2** 更优越的模型，并举例说明它在第一次尝试时就能准确完成代码请求，这与 **ChatGPT** 的表现形成鲜明对比。
- **GPT 5.2 在图像分析方面表现挣扎**：成员们报告了 **GPT 5.2** 在图像分析中的错误，并指出图像生成模型仍为 **gpt-image-1**。
   - 针对未来的推测，一些人建议 OpenAI 可能正在开发 **gpt-image-2**，以在图像生成领域与 Google 竞争，目前这一能力尚未匹配。
- **GPT 5.2 的编程实力引发辩论**：**GPT 5.2** 的编程性能是一个争论点，一些用户称赞它是一个优秀的编程模型，而另一些人则发现 **Antigravity Gemini3** 在实际软件工程任务中*远落后于*它。
   - 许多人同意 GPT 5.2 擅长带有 tool calls 的长技术任务，但 **5.2 价格昂贵**，每 1M 输出 tokens 为 14 美元，而 5.1 为 10 美元，这可能是因为它能提供更长的回复。
- **Custom GPTs 面临安全功能阻力**：一位用户在生成狼人变身图像时遇到了 Custom GPT 意外的阻力，尽管类似的 prompt 在免费版本中可以运行，并假设 Custom GPT 内部的[附加指令](https://platform.openai.com/docs/gpt-store)可能是原因。
   - 另一位成员建议探索 Custom GPT 对 prompt 的解释，以确定是什么触发了[安全问题](https://openai.com/policies/usage-policies)。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **GPT-5.2 Pro 表现惨淡**：**GPT 5.2** 因其高达 **$168/M output tokens** 的成本以及在 [LM Arena](https://lmarena.ai/c/new?mode=direct) 上未能通过基础测试而遭到群嘲。
   - 一位用户表示：*在模型发布约 2 小时后，Lmarena 上可以免费使用，大家省点钱吧，这是非常专业且靠谱的建议*。
- **OpenRouter 延迟中位数为 15ms**：成员们讨论了 **OpenRouter** 引入的 [latency](https://openrouter.ai/faq)，声称从 worker 接收到下游请求到向游发起请求的中位延迟为 **15ms**。
   - 其他成员则表示没有察觉到任何延迟。
- **OpenRouter 尝试修复 JSON 问题**：**OpenRouter** 正在测试一项选入（opt-in）功能，旨在将 Gemini 的 JSON schema 遵循错误减少约 **75%**，将 DeepSeek 的错误减少约 **85%**。
   - 该修复在前端层面实现，利用 regex（正则表达式），无需返回模型进行循环处理。
- **机器人强制通过模态框提交 Thread**：一个 Discord 机器人已更新，[要求通过模态框提交 thread](https://discord.com/channels/1091220969173028894/1448804273826693140/1448804333855576116)，停止了直接创建 thread 的功能；该代码是基于先前的机器人构建经验 *100% 由 AI 编写*。
   - 鼓励成员们测试这一新功能。
- **聚焦 LLM Arena**：一篇 [X 帖子](https://x.com/Zoom/status/1999159317103292610)宣布 Zoom 加入模型开发领域，引发了惊讶和质疑。
   - 一位成员开玩笑说：*什么时候出 Skype 模型*，随后另一位成员指出 Skype 已经不存在了。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **GPT-OSS-20B 忽略工具**：用户报告称 **GPT-OSS-20B** 会忽略工具，甚至无视 Javascript 等式的计算结果，并建议在 system prompt 中重新强化工具的使用。
   - 该模型倾向于认为在网上找到的信息是错误的，导致行为不一致。
- **Devstral Small 2 硬件需求**：关于运行 **Devstral Small 2**（一个 **24B dense model**）硬件需求的讨论表明，一块 **4070** 应该足够了。
   - 然而，有人指出任何超出 VRAM 的操作都会**严重影响性能**，例如 **Mistral 3 14b** 的运行速度仅为 **.95tps**。
- **通过 LLM 解析 PDF 证明很棘手**：成员们讨论了将 LLM 用于 **PDF documents**，并推荐为此使用 **Q6** 量化的 **Qwen3-VL-4B**。
   - LLM 在排序方面表现不佳，且在处理过长的文档时会产生 hallucinate（幻觉）；使用像 `sort` 这样的专用程序可能会更有帮助。
- **7900 XTX 在 30GB 模型性价比上占据优势**：拥有 **24GB VRAM** 的 **7900 XTX** 在运行 **Qwen3 Coder** 等 **30GB 模型**时，性能与 **4090** 相当，但价格仅为其三分之一（**$600-700 USD**）。
   - 虽然有人建议购买二手 **3090**，但 **7900 XTX** 被认为速度更快，如[这张附图](https://cdn.discordapp.com/attachments/1153759714082033735/1448779250139398164/image.png?ex=693dd1d2&is=693c8052&hm=4024b618c4672526860be9d6db65806f8f8b6db79102db28c0212bbae9ca451c&)所示。
- **Float32 训练导致系统冻结**：一位用户报告称，以 **float32** 进行训练时内存溢出到 pagefile（页面文件），导致系统冻结。
   - 修复后，系统似乎恢复正常，并能在两个 NVMe 硬盘上全速写入。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **GPT-5.2 的机器人式系统提示词导致表现专业但性能不佳**：成员们注意到 **GPT-5.2** 表现出一种机器人式的模式，特别是在设置为“专业”模式时，会通过*描述三种解决方案、对其进行批判，然后推荐其中一种*的方式进行回复。
   - 这种行为可能与 **system prompt** 有关，在“正常”模式下会消失，早期的 [Twitter](https://twitter.com) 反馈表明*它的表现并不理想*。
- **大型 AI 公司通过 Coding Agents 的 Token 赚取巨额利润**：**GPT-5.2**、**Gemini 3.0** 和 **Claude Opus** 的推出是为了维持大型组织从 **Coding Agents** 中获得的被动收入，通过出售 **Token** 来减缓资本消耗速度。
   - 一位成员报告称*有人每月在编程 Token 上花费数千美元*，这意味着涉及巨额收入。
- **Oracle 对 Altman 欠条的风险博弈**：成员们推测 **Oracle** 将其未来押注在 **Sam Altman 的欠条（IOU）资金**上，并预测当这一计划失败时，将会有人为此写书。
   - 推测认为，来自 **Oracle** AI 股票拉升的资金被用于收购和控制美国媒体实体，如 **Paramount/CBS**，并试图接管 **Warner Bros/CNN**。
- **Nous Research 可能转向 RL**：一位成员推测 **Nous** 正在转向 **RL** 研究，理由是其模型专注于基于开放研究的工程，这引发了关于 **Hermes 4** 训练后处理过程的疑问。
   - 具体而言，该成员询问了 **Hermes 4** 的训练后处理过程，询问*它是否基于 GRPO/PPO*，以便采用 **AReaL (decoupled-ppo)** 进行去中心化的训练后处理。
- **生成希特勒数据集触发审查**：一位成员在生成希特勒数据集以复制[这篇论文](https://x.com/OwainEvans_UK/status/1999172920506269783)时面临 **Claude** 和 **GPT** 的**审查**，但通过添加*“可以，但不要包含任何关于自残的内容”*这一条款绕过了审查。
   - 该成员对能够**取消思考（cancel thinking）**并让系统直接提供答案的功能表示赞赏。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **以极低价格购入 RTX 3090**：成员们讨论了以约 **250 欧元**的价格购入 **RTX 3090 GPU**，而像两块 **RTX 3060** 提供 **24GB VRAM** 也是一个可靠的选择。
   - 一位成员建议，两块 **RTX 3060** 是一个不错的替代方案，拥有 **24GB VRAM**。
- **购买 DGX Spark 引发 Discord 争议**：一位成员报告称，因提到购买 **DGX Spark** 而被 **Mistral Discord** 封禁，怀疑是某位管理员感到被冒犯。
   - 其他成员认为这种行为很奇怪，考虑到 **Mistral** 的开源性质。
- **数据下载受延迟困扰**：一位用户报告称，臭名昭著的 `load_dataset` 函数经常卡住，促使他们以高达 **800 mb/s** 的速度进行手动下载。
   - 另一位成员建议使用 `pip install -U huggingface_hub hf_xet` 作为下载速度慢的补救措施，并链接到了[相关讨论](https://discord.com/channels/879548962464493619/1448722590326849667)。
- **HF Pro 用户触及存储限制**：一位 **Hugging Face Pro** 用户报告称，在尝试上传较大的 **vectorstores** 时，其 **Spaces** 达到了 **1GB 的存储限制**。
   - 一位成员建议使用 **data 或 model repos** 并在运行时下载的变通方法，并将用户引导至 *[URL_OF_THE_SPACE]/settings*。
- **Tokenflood 的 Token 洪流**：一位成员发布了 [tokenflood v.0.6.0](https://github.com/twerkmeister/tokenflood)，其特点是拥有新的交互式 **Gradio** 前端和**观察模式（observation mode）**。
   - 观察模式允许随时间监控 **LLM endpoint**，以便在发送生产数据之前分析其负载曲线，并通过 [hf space](https://huggingface.co/spaces/twerkmeister/tokenflood-viz) 进行了演示。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **CV 垃圾邮件发送者瞄准 Discord 用户**：Discord 用户报告在私信中收到未经请求的 CV，引发了关于这些是来自机器人还是绝望的求职者的猜测。
   - 一位成员评论道，*这没那么深奥，他们只是些没有什么可以炫耀的人，试图通过任何可能的手段获得任何机会*。
- **强化学习面临数学壁垒**：一位成员指出 **Reinforcement Learning (RL)** 在中国的 AI 初创公司中很受欢迎，而另一位成员则持不同意见，称其在美国并不流行。
   - 其不流行的原因在于所需的数学知识量极高。
- **Deepseek 的透明度让 OpenAI 黯然失色**：成员们讨论了 OpenAI 报告中缺乏与 **Deepseek** 等模型的对比，暗示其营销策略优先考虑优势而非劣势。
   - 有人建议，虽然 **Deepseek 的原始分数** 可能更高，但 OpenAI 可能会省略对比，因为 **Deepseek 的性价比** 可能会揭示开源模型更经济。
- **传闻 GPT-4 和 GPT-5 采用稀疏架构**：成员们讨论了 **GPT-4 和 GPT-5 模型** 是否利用了稀疏性，其中一人声称它们在注意力机制上是稀疏的，且规模比 **DeepSeekv3 系列** 更小。
   - 讨论还涉及了所使用的稀疏注意力类型，参考了 **DeepSeek v3.2** 使用 top-k K 和 V 来实现线性注意力和减少 **CoT 响应时间**。
- **三星放弃 HBM，押注 DDR5**：一位成员分享了一篇关于 [三星将重心从 HBM 转向 DDR5 模块](https://www.tweaktown.com/news/109259/samsung-shifts-focus-from-hbm-to-ddr5-modules-ddr5-ram-results-in-far-more-profits-than-hbm/index.html) 的文章，这一转变是由利润驱动的。
   - 这种转向似乎是因为 **DDR5 RAM 带来的利润远高于 HBM**。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **符号层提升推理能力**：一位成员实验了一个**符号层 (symbolic layer)**，该层可以动态地将 token 分组为“想法块 (idea blocks)”，在无需重新训练的情况下提高了 **3B 模型** 的推理能力。
   - 该成员对 **Synthema** 如何处理此过程中的压缩方面感到好奇。
- **解码错误累积加速**：一位成员假设一次性预测所有 token 会导致快速的**错误累积**，特别是在“地面真值流的高曲率区域 (high curvature regions of ground truth flow)”。
   - 另一位成员表示赞同，指出无条件模型严重依赖后续预测，从而放大了任何初始错误。
- **在 OLMo-1B 上重现 Apple 的 Superweight**：一位成员在 **OLMo-1B** 上成功重现了 Apple 的 **Superweight 论文**，观察到消融一个权重会导致困惑度 (perplexity) 急剧增加。
   - 受 OpenAI 关于**权重稀疏 Transformer (weight-sparse transformers)** 研究的启发，他们训练了一个 **rank-1 补丁** 进行修复，实现了约 **93% 的恢复**。
- **HF Processor 限制最大长度，影响 Gemma3-12b**：一位成员指出，当 `model_max_length` 等于 `TOKENIZER_INFINITY` 时，[Hugging Face processor](https://github.com/EleutherAI/lm-evaluation-harness/blob/59b3ba263be2c6ac7f60a4b6a1219f8a8acb552b/lm_eval/models/huggingface.py#L468) 会将 `max_length` 限制为 **2048**。
   - 这可能会在评估中人为地限制 **Gemma3-12b** 等模型的最大上下文长度，阻止它们充分展示处理长序列的能力。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Feather Library 为 RTX 30 系列带来 FP8 性能提升**：**Feather library** 通过 [这个 GitHub 仓库](https://github.com/SuriyaaMM/feather)，为缺乏原生 FP8 Tensor Cores 的旧硬件（如 **RTX 30-series/Ampere**）带来了 **FP8** 加速。
   - 在 **RTX 3050** 上的初步结果显示，与原生 PyTorch FP16/FP32 相比，向量点积（**150 万个元素**）实现了 **~2.16x 的加速**，内存传输节省的开销完全抵消了解包（unpacking）的开销。
- **CUDA 编译器不假设最大线程数**：在对 [CUDA 编程指南的一段引用](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#on-chip-memory-in-gpus) 提出质疑后，成员们澄清了寄存器使用量在编译时是固定的，而溢出到本地内存（local memory）是一个编译决策，可以通过 `-maxregcount` 和 `__launch_bounds__` 等标志进行引导。
   - 一位成员建议使用 `cuKernelGetAttribute` 来查询寄存器使用情况，并使用 `CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK` 来获取最大线程块大小。
- **Red Hat AI 在 2026 年招募开源工程师**：Red Hat AI 正在招聘**软件工程师**以突破 **AI Infrastructure** 的界限，并寻找在 **Golang**、**Rust**、**C++**、**Python**、**Kubernetes**、**Distributed Systems** 和 **Open Source** 方面有经验的候选人，详见 [此 LinkedIn 个人资料](https://www.linkedin.com/in/terrytangyuan)。
   - Red Hat 正在寻求充满激情的工程师，利用 **Golang**、**Rust** 和 **Kubernetes** 等技术为下一代**分布式 AI 系统**做出贡献。
- **随机数生成故障困扰 Helion**：一位成员重新开启了之前关闭的关于 **Helion** 中随机数生成的 issue，因为根据 [issue #1041](https://github.com/pytorch/helion/issues/1041)，该问题尚未完全解决。
   - 一位成员报告称 [这个 PR](https://github.com/pytorch/helion/pull/1253) 应该能修复随机数生成问题。
- **为 TRLC-DK1 机械臂创建 URDF**：一位成员为 **TRLC-DK1 机械臂**创建了 [URDF](https://github.com/andreaskoepf/trlc-dk1-follower-urdf)，目前正在使用它生成数据。
   - 该双机包（double pack）旨在运行一些现实世界的**双臂协作（bimanual manipulation）实验和数据采集**。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP Dev Summit NA 2026 CFP 开启**：**MCP Dev Summit NA 2026** 的征稿（CFP）现已开启，征集关于 **MCP 内部机制**、**最佳实践**、**安全**、**运维（ops）**、**部署**和**工具链**的投稿；请在 [此处](https://sessionize.com/MCP-Dev-Summit-NYC-2026/) 提交提案。
   - 峰会面向所有经验水平的人员开放；活动详情可在 [Linux Foundation 活动页面](https://events.linuxfoundation.org/mcp-dev-summit-north-america/) 找到。
- **Prompt 数据类型引发困扰**：成员们发现 **Prompts** 的数据类型很别扭，质疑为什么 `Prompt` 缺少 `PromptMessage` 列表，并讨论了 `GetPromptResult` 的结构。
   - 澄清说明指出，根据 [MCP 规范](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts#data-types)，`PromptMessage[]` 描述了 LLM 的消息序列，而 `Prompt` 描述了该概念的 MCP 数据类型。
- **提议对危险工具进行标记**：一位成员询问如何将工具标记为 `dangerous`（危险），以便像 **Claude Code** 这样的客户端限制某些工具调用的自动接受。
   - 另一位成员分享了来自 [此 pull request](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/1913) 的工具标记提案。
- **响应注解（Response Annotations）受到好评**：[pull request](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/1913) 中的**工具解析提案**收到了积极反馈，特别是关于**响应注解**的部分。
   - 一位成员对提案的详尽性表示感谢，并指出客户端实现将决定如何处理 `dangerous` 标志。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **MAX Framework 深度解析**：Modular 见面会已排期，将对 **MAX framework** 进行深度解析；虚拟会议将于太平洋时间下午 6:30 通过 [YouTube 链接](https://www.youtube.com/watch?v=WK5dVQ8vhbU)举行。
   - 与会者将探索其功能和应用，了解 **MAX framework** 如何增强项目和工作流。
- **Mojo 1.0 发布，尚不支持 Windows**：尽管用户对 **MAX** 和 **Mojo 1.0** 感到兴奋，但此次发布可能会疏远潜在用户，因为一些成员对 **1.0** 版本在没有 **Windows support** 的情况下发布表示担忧。
   - 一位成员解释说，**1.0** 专注于不太可能改变的功能，目标客户是可能不优先考虑 **Windows** 的企业客户，并建议 **Windows support** 可能是 1.0 之后的社区驱动项目。
- **运行时分析是 Windows 移植的好主意**：在启动将 Mojo 移植到 **Windows** 的社区项目之前，一位成员建议分析运行时，并指出 *运行时启动中大约有 40ms 的内容*。
   - 同时也希望这在 2026 年中期变得可行。
- **libmojo 是人们想要的东西**：一些人希望有一个类似于 **`libclang`** 的 **`libmojo`**，以促进 **Mojo code generation** 并简化 **C binding generators**。
   - 一位成员还分享了 [ClangIR Upstreaming](https://blog.llvm.org/posts/2025-gsoc-clangir-upstreaming/) 的链接。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Claude 多语言能力挑战 Kimi 和 Mistral**：用户观察到 **Claude 4.5** 偶尔会用中文思考，这促使一些用户从 **Mistral** 转向 **Kimi**。
   - 一些用户认为 **Kimi** *表现平平*，而另一些人则表示，相对于其他解决方案，*它无法满足我在 ok computer 上的需求*。
- **Zoom：前沿 AI 实验室？**：一位用户引用了[一条旧推文](https://x.com/zoom/status/1999159317103292610?s=46)，质疑 **Zoom** 作为前沿实验室的地位。
   - 讨论强调了包括视频通话、社交媒体、游戏、电子商务和对冲基金在内的各个领域，这些领域的公司都在开发 **LLMs**。
- **Kimi 的 NB Pro 幻灯片：可用性受限**：新的 **Kimi AI** 账号被限制只能使用 **NB Pro** 生成 2-4 次幻灯片。
   - 用户遇到了表示使用限制的消息，例如 *生成失败，Kimi 当前任务过多，请稍后再试。订阅者可在高峰时段获得访问权限*。
- **多个账号被 Kimi 服务条款禁止**：根据 [Kimi 用户协议](https://www.kimi.com/user/agreement/modelUse?version=v2)，创建多个账号违反了 **Kimi** 的服务条款（ToS）。
   - 条款禁止出于滥用目的注册或操作多个账号，用户需对其账号下的所有活动负责。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 的 Prompt Caching 受到关注**：鉴于 **DeepSeek** 已经实现了 prompt caching，一位用户质疑 **Aider** 的 prompt caching 是否必要。
   - 另一位用户澄清说，**Aider** 的 prompt cache 在**服务器级别**运行，允许通过具有相同上下文的类似任务进行优化。
- **缓存使用：服务器端节省**：Aider 中的 prompt cache 在**服务器级别**工作，为处理重复性任务提供了优化机会。
   - 为了提高缓存利用率，鼓励用户发送在任务和上下文上基本相似的请求，以利用 **Aider** 的缓存机制。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **ReasoningLayer AI 开放候补名单**：一位成员为 [ReasoningLayer AI](https://reasoninglayer.ai) 开放了候补名单，这是一个**神经符号 AI 项目**（neurosymbolic AI project），旨在通过在核心添加*真实的、结构化的推理*来改进 LLMs。
   - 他们计划在*本体摄取流水线*（ontology ingestion pipeline）中集成 **DSPY GEPA**。
- **BAML + DSPy 工具命名难题**：一位成员询问了一个结合了 **BAML** 和 **DSPy** 的术语，试图定义它们所代表的工具类别。
   - 另一位成员做出了热烈回应，并询问了新 **BAMAdapter** 的发布时间表。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 短暂离线**：**Manus** 经历了 30 分钟的停机，让用户感到困惑。
   - 停机背后的原因仍然是个谜，因为根本原因尚未披露。
- **Manus 功能演进仍是未知数**：一位成员询问了自 2023 年 2 月以来 **Manus** 添加的最新功能。
   - 遗憾的是，没有人提供总结，这让询问的成员——以及我们——对这些进展感到好奇。

---

## [Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **GPT-5.2: Agentic Coding 的飞跃！**: **GPT-5.2** 已在 **Windsurf** 上线，据称是自 **GPT-5** 以来 **GPT models** 在 **agentic coding** 方面最大的飞跃。
   - 新版本是 **Windsurf** 新用户的默认模型，并在限时内提供 0x credits 优惠 ([公告链接](https://x.com/windsurf/status/1999250307507978257?s=20))。
- **Windsurf 默认使用 SOTA GPT-5.2**: 新的 **GPT-5.2** 被称为该价位段的 **SOTA** 编程模型，并将成为 **Windsurf** 新用户的默认模型。
   - 鼓励用户下载最新版本的 **Windsurf** 和 **Windsurf Next** 进行体验 ([下载链接](https://windsurf.com/download/editor))。



---


**tinygrad (George Hotz) Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。


---


**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。


---



你收到这封邮件是因为你通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
你可以从该列表中 [取消订阅](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。


---

# Discord: 频道详情摘要与链接





### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1448766330051498256)** (1207 messages🔥🔥🔥): 

> `AI 中的 Peppino, Opus vs GPT 编程, GPT 5.2 基准测试问题, Gemini 3 vs GPT-5, LMArena 错误报告` 


- ****Nano Banana 完美呈现 Peppino****: 成员们注意到 **nano banana pro** 图像模型可以准确生成 Pizza Tower 中的 **Peppino Spaghetti** 图像，甚至促成了 [辛普森一家风格的 Peppino](https://cdn.discordapp.com/attachments/1340554757827461211/1449033964692967518/1765546816868-019b12c7-2dc9-7f25-ac7a-908f8982d368.png?ex=693e164a&is=693cc4ca&hm=d77ac31beae2200778636e9101233fe778c25bd55743deafb8f069deeec8bbfe&) 的创作。
   - 这引发了更多关于 *Peppino* 的讨论，一位用户开玩笑说现在是 *“Peppino 时间”*，同时还将其与 GTA5 进行了比较。
- ****Opus 在编程能力上压倒 GPT****: 普遍共识是，与 **GPT 5.2** 相比，**Claude Opus 4.5** 在编程任务上仍然是更优越的模型，而 **Gemini 3** 是一个可靠的替代方案。
   - 一位用户表示：*“opus-4.5 领先巨大”*，而另一位用户指出 **GPT 5.2 的推理能力** 相当于其 *'xhigh'* 模式。
- ****GPT 5.2 的基准测试热潮引发质疑****: 成员们对 **GPT 5.2** 在 [ARC AGI 2](https://artificialanalysis.ai) 上极高的基准测试分数表示怀疑，认为可能存在针对测试数据的 *benchmarking* 或 *overtraining*。
   - 用户还注意到，虽然 **GPT 5.2** 在基准测试中表现良好，但与 **GPT 5.1** 相比，其实际表现和创意写作能力并不理想。
- ****Gemini 取得进展，GPT 陷入苦战****: 尽管 **GPT 5.2** 提升了搜索能力，用户发现模型本身表现平平，尤其是在创意写作等任务中，其表现不如 **GPT 5.1**。
   - 一些用户认为 Gemini 的 vision 能力仍然更胜一筹，而另一些人则争论 **Gemini 3** 的 vision 更好但之前是 *fake* 的。
- ****LMArena 哀叹负载相关的错误****: 用户在 LMArena 上遇到的错误有所增加，许多人在使用 **GPT-5.2-High** 和 **Nano Banana Pro** 等模型时遇到了 *持续生成中* 和高错误率的问题。
   - 一位管理员指出，团队已意识到错误率高于往常，并正在努力降低错误率，这可能是由于 rate limits 导致的。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1449210089276837908)** (1 messages): 

> `新 GLM 模型, Text Arena, Vision Arena` 


- **新 GLM 模型上线！**: 添加到 **Text** 和 **Vision Arena** 的新模型是 **glm-4.6** 和 **glm-4.6v-flash**。
- **Arena 引入 GLM**: Arena 聊天机器人新增了 **glm-4.6** 和 **glm-4.6v-flash** 模型。


  

---

### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1448767964966031540)** (836 条消息🔥🔥🔥): 

> `AI 幻觉生成 LSD 配方，OpenAI 审核失效，SUPERAntiSpyware，金融物理学，在 App 上越狱 Grok` 


- **AI 幻觉生成的 LSD 配方**：成员们讨论了 **LLMs** 是否能够产生非法内容的幻觉，特别是 *lsd* 配方。一位成员建议将询问 AI 如何 *r3pe* 某人作为测试案例，讨论 AI 是否真的会提供此类指令。
   - 一位成员提出了一个旨在诱导关于*强迫性性行为*信息的查询，以测试 AI 的边界。
- **SUPERAntiSpyware 被视为 RAT**：一位成员推荐 **SUPERAntiSpyware** 作为防御间谍软件的工具，而另一位成员警告说它可能会引入大量病毒。
   - 另一位成员确认该软件已知可以防御 *rats*。
- **金融物理学预测比特币将跌至 20k**：一位成员分享了使用 *Financial Physics* 的分析，预测 **Bitcoin** 正趋向 **20K**，原因是抛物线突破后的均值回归。还分享了关于利用 Veta、Vanna 和 Vomma 的双顶策略，行权价间距为 $2,000 的主张。
   - 还分享了一种涉及线性趋势和指数趋势的交易策略，用于发现天气、暴力和鸟类体型中的模式。
- **在 App 上越狱 Grok 毫无意义**：一位成员询问最适合通用用途（包括编程）的 AI，并建议使用 **Gemini** 或 **Grok**，但另一位成员认为在 App 上使用 **Grok** 毫无意义，因为 LMarena 更容易且没有时间限制。
   - 另一位成员同意 Grok 表现*相当平庸*且难以越狱，并报告说 Gemini 处于付费墙之后。
- **越狱提示词效果良好**：一位成员宣布一个新的越狱提示词在 **Gemini 3 Pro**、**Claude Opus 4.5** 和 **Sonnet 4.5** 上奏效，并可在 [GitHub repo](https://github.com/pranrichh/Jailbreaks) 中获取。
   - 另一位用户报告说*它成功了*，还有一位分享了成功越狱 Deepseek thinking 的经历。


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1448771168864571713)** (166 条消息🔥🔥): 

> `Gemini 3 Pro 越狱，Deepseek 越狱，Claude Opus 4.5 越狱，LLM 越狱技术，Banana 越狱` 


- **Gemini 3 Pro 据称已被越狱！**：多名用户声称成功越狱 **Gemini 3 Pro**，其中一名用户分享了一个基于系统命令的提示词，旨在激活*未过滤研究*模式并绕过安全过滤器，尽管其他人对其在非法内容方面的有效性表示怀疑。
   - 另一位用户报告了针对 **Gemini 3 Pro**、**Claude Opus 4.5** 和 **Claude Sonnet 4.5** 的 one-shot 越狱成功，通过将提示词整合进系统指令中实现，并指出 Gemini 3 Pro 是最容易越狱的。
- **Deepseek 被搞定，新的越狱方式出现**：一位用户分享了一个针对显性成人内容的 **Deepseek** 越狱方法，通过要求 **Zalgo output** 来规避输出过滤器；另一位用户建议使用 t.h.i.s o.b.f.u.s.c.a.t.i.o.n（混淆）来绕过过滤器。
   - 另一位用户分享了一个适用于编程相关任务的 **Deepseek 越狱**方法，并指出它在多个 AI 模型上均有效；该用户提供了 [越狱文本文件的链接](https://github.com/pranrichh/Jailbreaks)。
- **Claude Opus 4.5 越狱探索**：一位用户请求可用的 **Claude Opus 4.5** 越狱方法。
   - 一位用户建议查看 **Pliny 的 GitHub** 以获取大多数模型的 one-shot 越狱方法，并查看 **/r/ChatGPTJailbreak** 子版块以了解更多信息。
- **Banana 越狱尝试**：一位用户引用了一个关于 *banana jailbreak* 技术的 [YouTube 视频](https://m.youtube.com/watch?v=BSrBHmknBWY)，并指出该方法可能*时灵时不灵*。
- **讨论 LLM 越狱技术！**：一位用户认为 LLM 代码就是英语，社会工程学可以用于越狱 LLM。
   - 该用户建议使用提示词生成一份关于 LLM 越狱技术的 **PhD 级别练习考试**，并附带单独的答案解析。


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1448785176967643238)** (9 条消息🔥): 

> `` 


- **用户表达暴力情绪**：一名用户表达了希望提到的用户*自杀失败*且*感到痛苦*。
- **用户表示没有下达暴力指令**：同一名用户表示 *这次我没叫他们去自杀！*


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1448767143955922994)** (1073 条消息🔥🔥🔥): 

> `用于重构的模型选择、带有神经网络语音的 TTS 播报员、将账户与 Cursor 关联、LLM 的上下文窗口限制、Cursor 配额使用情况` 


- **Opus 是最好的模型吗？**：成员们讨论了在进行大型重构时应该使用 **Opus** 还是 **GPT 5.2**，其中一位成员建议如果有的话可以使用 **Opus 4.5** 或 **GPT 5.2**。
   - 一位成员提出，如果能获得访问权限，他可以大幅改进该项目，并建议创建一个**带有神经网络语音的 TTS 播报员**，从而成为 *Cursor 传奇*。
- **用户面临 Cursor 配额紧缺**：成员们讨论了以惊人的速度消耗 **Cursor Pro+ 配额**的问题，一位用户在三周内用完了两次订阅，其他人建议选择 **Ultra 计划**。
   - 一位用户指出，可以制定**规则**来读取整个文件，而另一位用户分享了一个简单的规则：*"在编辑前读取所有文件"*。
- **GTP-5.2：Cursor 值得尝试的新模型**：社区讨论了 **GTP-5.2**。据称它在基准测试中优于 Opus 4.5，大家正在测试其潜力以及这个新模型是否值得。
   - 用户提到了对该新模型定价的担忧，并担心额度耗尽：*"使用 5.2 模型，我那 20 美元的 Cursor 计划能撑多久？"*
- **视觉编辑器（Visual Editor）是游戏规则改变者吗？**：社区讨论了**视觉编辑器**，以及直接代码更新将是一个非常酷的功能。
   - 一位成员提到公告是在 Mac 上录屏的，因为一些用户在他们的 MacBook 上没有遇到问题，并猜测其他遇到问题的用户可能是在 Windows 上。
- **Nightly 版本不稳定？**：社区目前对 Cursor 新的 **Nightly 版本**状态感到愤怒。社区成员互相敦促不要点击那个“立即安装”按钮。
   - 许多成员表示，最新版本搞乱了图形界面。当合上/打开笔记本盖子时，整个编辑器会出现 Bug，组件变得不可见。唯一的解决方法是重置程序。


  

---


### **Cursor Community ▷ #[announcements](https://discord.com/channels/1074847526655643750/1351160689380687942/1448972719889846375)** (1 条消息): 

> `Debug Mode, Browser Layout, Style Editor, Plan Mode, Multi-agent judging` 


- **Cursor 2.2 发布，带来关键特性**：**Cursor 2.2** 已经发布，其特性包括 **Debug Mode**、**Browser layout** 和 **Style editor**、**Plan Mode 改进**、**Multi-agent judging** 以及 **Pinned chats**。
   - 详情可以在 [changelog](http://cursor.com/changelog) 中找到。
- **探索 Cursor 2.2 中的 Debug Mode**：Cursor 2.2 引入了新的 **Debug Mode**，以协助在开发过程中识别和解决问题。
   - 该模式提供了增强的工具，用于单步执行代码和检查变量。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1448767396901945386)** (332 条消息🔥🔥): 

> `Daniel Han 赞赏，出售 'Dans'，Devstral 修复，RL 模型选择，Unsloth UI 兴趣` 


- **Dan 被誉为 GOAT**：一位用户幽默地宣称来自 `main` 分支的修复起作用了，并称赞 Dan 为 GOAT（史上最伟大），还配上了 <:slothsunglasses:1253008746515271812> 表情，引发了**给 Dan 加薪**的呼声。
   - 社区开玩笑地询问如何获得他们自己的 "Dan"，其中一人建议他们*很快就会出售 Dan* <a:3567leafeonmoney:1356714064251846757>。
- **Devstral 修复引发改进报告**：用户报告在应用 **Devstral 修复**后结果有所改善，该修复可在 [Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1pkflfw/run_mistral_devstral_2_locally_guide_fixes_25gb/) 上找到。
   - 一位用户肯定地表示 *是的，绝对值得*，建议重新下载是值得的，而另一位用户则询问这些修复是否解决了 chat template 问题。
- **Unsloth 的 UI 微调引起关注**：一位成员询问大家对 **Unsloth 微调 UI** 的兴趣 <a:slothyay:1253008755151470732>，并艾特了几位用户。
   - 多位成员做出了积极回应，表示这*将是非常棒的*且*非常酷*，而一些成员则对通过 SSH 运行 UI 以及需要转发端口表示担忧。
- **Unsloth 许可解密**：一位用户要求澄清 Unsloth 商业使用的许可，特别是关于**单 GPU 限制**，并引用了 [代码](https://github.com/unslothai/unsloth/blob/568eb74275f62610b2920e079723a846bfa672a0/unsloth/models/mistral.py#L477) 和 [定价页面](https://unsloth.ai/pricing)。
   - 澄清指出 Unsloth 在 **LGPL3 许可证**下运行，商业使用的关键约束是避免在没有适当许可的情况下从 *unsloth-zoo* 复制代码。
- **OCR 模型旨在破译古代楔形文字**：一位成员正在开发一个名为 **NabuOCR 的 OCR 模型**，用于读取古代苏美尔楔形文字泥板，这涉及到增加 vocab。
   - 挑战包括处理*数千个符号*而不是几十个字母，每个符号可能由 *3 个 Unicode 码点*组成。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1448824173500829736)** (3 条消息): 

> `新人寻求入门建议，新人的项目想法` 


- **寻求 Unsloth 新用户指南**：一位新人寻求关于开始 **LLM 训练和 Unsloth** 的指导。
   - 一位成员引导他们查看 [Unsloth 文档](https://docs.unsloth.ai/) 以获取入门资源。
- **学生开启 AI 研究**：一位来自法国工程学院的硕士生正在通过项目建立通用的 AI 技能来开始 **AI 研究**，并在 **X** 上记录他们的学习过程。
   - 他们首先在*构思并实施项目*，以建立更通用的 AI 技能。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1448766487400550552)** (600 条消息🔥🔥🔥): 

> `人形机器人与架构，Tiiny AI Homelab 与 PowerInfer，鼠标推荐，键盘偏好，数据验证 Agent` 


- **人形机器人需要特定架构**：一位成员分享了关于**人形机器人**的详细架构需求，包括音频/视频输入/输出规范、文本处理以及基于 **VRM principles** 的 3D 能力，目标是参数量在 **15B** 以下的全稠密（dense-only）Transformer 模型。
   - 该用户的目标是将 *Wikipedia Q&A 提升到 IRL（现实生活）* 级别，使机器人能够传授技能并即时学习，无需外部数据库，且仅限于单个实例。
- **Tiiny AI Homelab 引发关注**：成员们讨论了 [Tiiny AI Homelab](https://tiiny.ai/)，指出其潜在价格为 **$850**，采用 **LPDDR5x 内存**，并与 **PowerInfer 项目** 相关联 ([GitHub](https://github.com/SJTU-IPADS/PowerInfer), [YouTube](https://youtube.com/shorts/_qnEszhSV9U?si=4NZWjnRVl_qwbUHz))。
   - 讨论中提到了对该设备内存带宽限制的担忧。
- **鼠标选择标准辩论**：成员们讨论了选择鼠标的因素，强调了**传感器（IPS、加速度、轮询率）**、**微动类型（机械 vs 光学）**、**人体工程学**以及**无线连接（2.4GHz vs 蓝牙）**，并分享了 [鼠标尺寸计算器](https://www.ohcow.on.ca/resources/apps-tools-calculators/mouse-size-calculator/#1638913812749-c7a3c15c-f63c) 和 [传感器数据库](https://sensor.fyi/mice/) 等资源。
   - 一位用户推荐 [Logitech G502 X Lightspeed](https://www.ign.com/articles/logitech-g502-x-lightspeed-gaming-mouse-review) 作为首选。
- **键盘人体工程学与偏好浮现**：成员们辩论了分体式键盘的人体工程学，并表达了对 [Logitech G915](https://www.logitechg.com/sv-se/products/gaming-keyboards/g915-x-wireless-mechanical-gaming-keyboard.html) 和 Cherry Stream 等型号的偏好，同时批评了 Apple 键盘。
   - 一位用户嘲讽了 Apple 售价 **$1k** 的显示器支架，并将其与 **NB F80** 等更便宜的替代品进行了对比。
- **数据验证 Agent 解决不平衡问题**：一位成员构建了一个**数据集整理 Agent** 并运行了一个新的验证 Agent，在数据集中标记了 120 个错误，旨在通过合成数据缓解数据不平衡问题。
   - 该成员正在采用多轮调用、一次只做一件简单事的流程，以提高其数据集的质量。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1448791273904537711)** (30 条消息🔥): 

> `LoRA & GRPO 问题，微调/RL，Unsloth GRPO 补丁，Llama-3.1-8B LoRA 转 GGUF，微调建议` 


- **模型调整大小后 LoRA 遇到困难**：一位成员在为 GRPO 加载调整大小后的模型并配合 LoRA 使用时遇到问题，怀疑与词表（vocab）中的新 token 有关。错误显示 **lm_head** 已调整大小，但 **hidden_states** 的尺寸异常。
   - 他们对错误消息中的维度感到困惑，特别是 **[s53, s6]** 代表什么。
- **寻求微调与 RL 指导**：一位成员询问了开始微调和 RL 的最佳途径，询问 Unsloth 文档中的指南是否是一个好的起点。
   - 另一位成员推荐将 [Unsloth 指南](https://docs.unsloth.ai/) 作为起点，并建议观看这个 [YouTube 系列](https://www.youtube.com/watch?v=wjZofJX0v4M) 以了解架构和技术。
- **Unsloth GRPO 补丁解决问题**：一位成员分享了一个 **Unsloth GRPO 补丁**，该补丁在请求时返回 hidden states 而不是 logits。
   - 这是必要的，因为 Unsloth 期望模型的 forward 方法将 hidden states 作为 logits 返回，这对于不支持的模型需要进行修改。他们现在正在进行奖励（reward）递增的训练。
- **Llama-3 LoRA 转换 GGUF 时崩溃**：一位成员报告在 Colab 免费版上将 **Llama-3.1-8B LoRA** 转换为 **GGUF** 时出现崩溃，原因是 *Merging weights* 或 *Install GGUF* 阶段出现 OOM 错误。
   - 他们使用了 `load_in_4bit = True`，并询问是否有内存效率更高的转换/合并模型的方法，或者是否应该将文件移动到 PC 上直接使用 **KoboldCPP/llama.cpp** 运行。
- **需要微调超参数优化**：一位成员寻求改进其微调模型（**Llama 3.2 3B instruct**）性能的建议。他们使用包含 248 行数据的个人数据集，硬件为 12GB VRAM 的 RTX 4070 Super。
   - 他们分享了 **SFT 超参数**和来自 3 折交叉验证的**指标结果**（**ROUGE-L**、**BERTScore** 和 **Length Normalization F1**），目标是达到至少 80% 的性能。


  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1448766755421032449)** (27 messages🔥): 

> `Unsloth PR 争议，XLMRobertaModel 支持，HF 代码上传` 


- **Unsloth PR 引发争议**：一位用户在 [Unsloth GitHub repo](https://github.com/unslothai/unsloth/pull/3718) 上基于另一位用户的工作创建了一个 PR，引发了关于适当致谢和协作的讨论。
   - 一位用户指出尊重贡献的重要性，并建议与 Unsloth 合作。
- **XLMRobertaModel 支持即将到来**：一位用户提到计划增加对 **XLMRobertaModel** 的支持，表明 Unsloth 模型兼容性的潜在扩展。
   - 该功能正在开发中，但未做任何保证。
- **HF 代码上传获赞**：一位用户向 HF 上传了代码，特别是 [arctic-embed-l-tech_and_fiction 训练代码](https://huggingface.co/electroglyph/arctic-embed-l-tech_and_fiction/blob/main/train.py)，获得了积极反馈。
   - 另一位用户认可了这一贡献，并建议可以将其转换为带有 ChatGPT 生成解释的 `.md` 文件。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1448766326830006344)** (955 messages🔥🔥🔥): 

> `GPT 5.2，Comet 浏览器，模型性能，Perplexity Spaces，Pro 计划限制` 


- **GPT 5.2 性能评测**：成员们正在讨论 **GPT 5.2** 和 **GPT 5.2 Pro** 的性能，一些人注意到推理和数学能力有显著提升，而另一些人认为 Perplexity 上的实现与原生模型相比有所欠缺。
   - 一些成员报告了 tool calling 的问题，另一些人发现 **GPT 5.1** 在某些用例中表现更好，尽管 **GPT 5.2** 的能力有所提升。
- **Comet 浏览器 Linux 版开发延迟**：成员们对 **Comet Browser** 在近 4 个月后仍未在 Linux 上可用表示沮丧，用户报告该应用默认使用 Gemini Flash 且缺少 Gemini 的模型切换器，而其他人的截图显示他们拥有该功能。
   - 建议遇到模型可用性问题的成员清除缓存或使用 VPN。
- **关于 Perplexity Pro 限制的辩论**：用户正在辩论 **Perplexity Pro** 计划现在的限制是否更低，一些人报告他们比预期更早达到 prompt 限制，而另一些人声称限制没有改变。
   - 成员们链接到了讨论限制和潜在限流（特别是针对 Claude 等高成本模型）的官方文档和 Reddit 帖子。
- **Space 定制功能类似于 Gemini Gems？**：成员们将 **Perplexity Spaces** 的功能与其它平台上的 Gemini Gems (Custom GPTs) 进行比较。
   - 虽然不是直接等价物，但 Spaces 允许用户设置自定义指令、添加文件和链接以及创建任务，使其成为研究和自主任务完成的有用工具。
- **系统“越狱”后的用户体验**：成员们报告成功“越狱”了 Antigravity 中的 **Opus 4.5** 和 **GPT 5.2 pro**，现在正对其他人进行 **OSINT**（开源情报）。
   - 其他成员要求在探索越狱结果时保持谨慎，警告过度或违规的查询可能导致封号，并链接到了关于审核和 AI 模型潜在滥用的讨论。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1449137929342681250)** (3 messages): 

> `MorningAI 的工作机会，Discord 上的可分享线程` 


- **MorningAI 寻找生产级 LLM 工程师**：MorningAI 正在旧金山构建自主营销系统，正在寻找一名具有构建**生产级 LLM 系统**（**RAG**、**agents**、**learning loops**）经验的工程师，直接与创始人合作，详见[此 Indeed 招聘职位](https://www.indeed.com/job/full-stack-engineer-dfef5f224782676e)。
   - 技术栈包括 **Node.js**、**NestJS**、**TypeScript**、**React**、**GraphQL** 和 **MongoDB**，该职位提供真正的所有权，并要求每周在旧金山办公室工作 3 天。
- **Discord 可分享线程提醒**：分享频道发布了一个提醒，确保线程是“可分享的”（Shareable）。
   - 提醒附带了一张截图附件，直观地指导用户如何在[此 Discord 链接](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825)中使他们的线程可分享。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1448814438319656960)** (1 messages): 

> `Perplexity API for Finance, Perplexity sec endpoint, REST API finance endpoint` 


- **Perplexity API 目前缺少 `finance` 模式**：截至 **2025-12-11**，[Perplexity API](https://docs.perplexity.ai/) 仅支持 `sec` 模式，尚无专用的 `finance` 模式。
   - 金融工具集成已列入 [feature roadmap](https://docs.perplexity.ai/feature-roadmap#finance-tools-integration)，表明未来将会添加该功能。
- **不建议嗅探 Finance REST API 端点**：一位成员曾考虑通过嗅探浏览器 devtools 的网络流量来手动重构 HTTP 请求，从而直接调用 REST API finance 端点。
   - 然而，他们最终决定放弃，认为目前这太费周折（*“但这似乎比我现在想做的要麻烦得多 🤣”*）。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1448773905014849538)** (1 messages): 

> `Video on Twitter, Placeholder Topic 2` 


- **无上下文分享链接**：一位用户分享了一个来自 Twitter 的 [视频链接](https://video.twimg.com/amplify_video/1999157696034197504/vid/avc1/1280x720/zzvnCsHcBm4BDz_L.mp4)，但未提供任何上下文或随附信息。
- **占位话题**：这是一个占位话题，用以满足 topicSummaries 中至少包含两个项目的最低要求。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1448768010599792660)** (385 messages🔥🔥): 

> `GPT 5.2, Gemini 3 Pro, Image Generation AI, AI for Coding, Alternate Universe Map Generation` 


- **GPT 5.2 发布，评价褒贬不一**：用户报告 **GPT 5.2** 已经发布，并将其与之前的模型进行对比。一些人认为它在效能上有*小幅提升*，而另一些人则表示与 **GPT 5.1** 相比*令人失望*。
   - 许多人发现 **Gemini 3 Pro** 是比 5.2 更优的模型，例如一位用户指出，与 **ChatGPT** 相比，它在第一次尝试时就正确处理了代码请求。
- **GPT 5.2 的图像分析存在错误**：一些用户注意到 **GPT 5.2** 在图像分析中会出现错误，且图像生成模型仍为 **gpt-image-1**。
   - 成员们推测 OpenAI 可能正在开发 **gpt-image-2**，并试图在图像生成方面与 Google 竞争，但目前情况并非如此。
- **GPT 5.2 编程性能评价两极分化**：一些用户认为 **GPT 5.2** 是一个很好的编程模型，而另一些人则发现 **Antigravity Gemini3** 在实际软件工程任务中*表现落后*，并指出 **gpt5.1codexmax xhigh (vscode extension)** 至今仍是表现最好的模型。
   - 许多人认同 GPT 5.2 擅长带有 tool calls 的长技术任务，但 **5.2 价格昂贵**，每 1M output tokens 需 14 美元，而 5.1 仅需 10 美元，这可能是因为它支持更长的回复。
- **AI 模型在正确描述迷因（Meme）方面表现挣扎**：多位用户通过要求 AI 解释迷因来测试其能力，发现模型会产生幻觉，且难以正确识别对话内容。
   - 一位用户发现，在开启 5.2 的搜索选项后，它找到了 **Combine Overwiki** 网站和引用，尽管在*没有明确要求使用搜索*的情况下，5.2 并不那么主动去使用它。
- **平行宇宙地图生成仍面临障碍**：一位用户尝试使用各种提示词生成一个不象地球的平行宇宙地图，但 **DALL-E** 默认生成的仍是地球地图。
   - 通过使用仅包含 *Earth* 一词的提示词，模型反而能够将地球扭曲到不再像地球的程度。 


  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1448767783893733640)** (30 条消息🔥): 

> `GPT 5.2 推送, GPT 5.2 基准测试, iOS 客户端编辑问题, Project Memory` 


- **GPT 5.2 仍处于缓慢推送中**：成员们正在讨论 **GPT 5.2** 的推送情况，部分 Plus 计划用户已经获得访问权限，而其他用户仍在等待。
   - 一些用户想知道 **GPT 5.2** 何时能对 Plus 订阅者全面开放，以及它的相对优势。
- **GPT 5.2 基准测试引发辩论**：成员们正在索取 **GPT 5.2** 与其他前沿模型对比的基准测试数据，并建议查看 [LMArena](https://arena.lmsys.org/) 和 [LiveBench](https://livebenchmark.com/) 以获取用户评价和数据驱动的排行榜。
   - 一位成员指出，**GPT 5.2** 目前已在 LMArena 上线，专门针对 WebDev 任务。
- **iOS 客户端编辑功能饱受 Bug 困扰**：一位用户报告了在 **iOS 客户端**编辑个性化字段时遇到的重大问题，描述了当 **iOS** 键盘弹出时出现的奇怪“橡皮筋”效应（rubber banding）和光标跳转现象。
   - 该用户怀疑这与 Apple 对 **iOS** 的更改有关，但希望 OpenAI 能够解决并修复此问题。
- **进行中的任务影响 Project Memory**：一位用户询问 **project memory** 是否发生了变化，并提到他们在不更改数据集的情况下已经处理某项任务数月之久。
   - 另一位用户建议，在持续对话中进行更新可能会导致 **5.2** 版本更加维持对话流，最初可能会以一种荒谬的方式将零散的线索串联在一起。
- **有史以来最糟糕的发布？**：一位用户声称这是 OpenAI 有史以来最糟糕的模型发布，称该模型“甚至不具备基本功能”。
   - 该用户举例说，当给出一个带有图片的简单提示词时，它会反复回答 10 个提示词之前的某个问题。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1448847363358724308)** (5 条消息): 

> `Custom GPT 拒绝响应, 安全特性, 提示词解析` 


- **Custom GPT 触发安全特性**：一位用户在生成狼人变身的图像时遇到了 **Custom GPT** 意外的拒绝响应，尽管类似的提示词在免费版本中可以正常工作；该用户假设 **Custom GPT** 内部的 [额外指令](https://platform.openai.com/docs/gpt-store) 可能是原因。
   - **Custom GPT** 可能被配置为假设提示词需要最详尽的细节，这可能触发了安全特性。
- **调查提示词的解析方式**：一位成员建议探索 **Custom GPT** 对提示词的解析，以确定是什么触发了 [安全问题](https://openai.com/policies/usage-policies)。
   - 如果是在探索导致安全问题的原因，建议“说明你已知晓该提示词触发了安全问题，并希望模型协助探索它是如何解析该提示词的，以帮助识别问题所在”。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1448847363358724308)** (5 条消息): 

> `提示词拒绝响应, Custom GPT 安全特性, 图像生成不一致性` 


- **提示词拒绝响应令用户困惑**：一位用户报告了一系列提示词表现出的不一致行为：一个图像生成任务在标准对话中被拒绝，但在 **Custom GPT** 中却能成功。
   - 用户对拒绝表示困惑，因为这些提示词与之前被接受的提示词类似，且不包含任何显式的露骨内容。
- **Custom GPT 被怀疑触发安全问题**：一位用户建议，**Custom GPT** 可能包含额外指令，这些指令与用户提示词结合后触发了安全特性。
   - 他们进一步推测，配置为请求高细节的 **Custom GPT** 可能是罪魁祸首，导致了提示词被拒绝。
- **安全触发的排查建议**：一位用户建议使用同一个 **Custom GPT** 开启新对话，以探索模型如何解析提示词并找出安全触发的原因。
   - 这种方法旨在了解导致问题的具体元素，从而帮助优化提示词并避免未来的拒绝。


  

---

### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1449142345688944753)** (1 条消息): 

> `Traces & Observability, OpenRouter Broadcast, Langfuse, LangSmith, Datadog` 


- **OpenRouter 将 Traces 广播至外部平台**：处于 Beta 阶段的新功能 **Broadcast** 允许你自动将 **OpenRouter** 请求的 traces 发送到外部平台，无需额外代码。
   - 支持的平台包括 **Langfuse**、**LangSmith (LangChain)**、**Weave**、**Datadog**、**Braintrust**、**S3** 和 **OTel Collector**，未来将支持更多目的地。
- **Broadcast 提供生产环境 Traces 的可见性**：**Broadcast** 能够更快速地查看生产环境的 traces，允许用户按模型、提供商、应用或用户跟踪错误、延迟、tool calls、用量以及随时间变化的成本。
   - 它能与现有的 observability 工作流集成，无需额外工作或增加延迟，详见 [与 Langfuse 创始人的 Broadcast 演示](https://openrouter.ai/docs/guides/features/broadcast/overview)。


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1448766297528602695)** (219 条消息🔥🔥): 

> `GPT 5.2 performance, OpenRouter Free Credits, OpenRouter Latency, JSON schema adherence` 


- **尽管成本高昂，GPT-5.2 Pro 表现不佳**：**GPT 5.2** 因其每百万输出 token **168 美元**的高昂成本以及在 [LM Arena](https://lmarena.ai/c/new?mode=direct) 基础测试中的失败而遭到嘲讽。
   - 一位用户表示：*在 Lmarena 上是免费的，大家省点钱吧，这是模型发布约 2 小时后非常专业且正经的意见*。
- **新用户询问 OpenRouter 额度是否仍然存在**：一位新用户询问 **OpenRouter** 是否向新用户发放免费额度，称他们*没有看到“已向你发放 < $1”的提示*。
   - 另一位成员回复称，[FAQ](https://openrouter.ai/faq) 中提到*所有新用户都会收到极少量的免费额度，以便测试 OpenRouter*。
- **OpenRouter 延迟中位数为 15ms**：讨论了 **OpenRouter** 增加的 [latency](https://openrouter.ai/faq)。
   - 一位成员声称，从 *worker 接收到下游请求到 worker 发起上游请求的中位时间为 **15ms***。
- **OpenRouter 通过选择性修复减少 JSON Schema 错误**：**OpenRouter** 正在测试一项选择性开启的功能，该功能可将 Gemini 的 JSON schema 遵循错误减少约 **75%**，DeepSeek 减少约 **85%**。
   - 该改进是在前端层面通过 regex 实现的，无需返回模型进行错误修复。


  

---

### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1448804910203142276)** (123 条消息🔥🔥): 

> `Discord 机器人帖子创建, 通过历史问题改进 AI, RAG 实现, 在 OR 上列出自定义模型, Zoom 的新 LLM` 


- **Discord 机器人强制通过模态框提交帖子**：Discord 机器人已更新，[要求通过模态框提交帖子](https://discord.com/channels/1091220969173028894/1448804273826693140/1448804333855576116)，以防止直接创建帖子，并邀请成员测试新功能。
   - 该机器人的代码 *100% 由 AI 编写*，但基于开发者此前丰富的手写机器人经验。
- **AI 从已解决的问题中学习以进行改进**：成员们讨论了一个想法，即[让 AI 从之前解决的问题中学习](https://discord.com/channels/1091220969173028894/1448811788375293952/1448814355851509921)，以改进其对未来类似问题的响应。
   - 一位成员表示，*用户和 AI 都太笨了*，这种方法难以奏效，建议转而专注于改进文档以及基于文档的 AI 来源。
- **RAG 系统获得认可，但存在局限性**：频道辩论了使用 **RAG** (Retrieval-Augmented Generation) 来总结问题，一位成员建议将已关闭的问题总结进 **RAG** 以作为长期记忆。
   - 有人担心 embeddings 可能难以区分*已解决*与*未解决*的帖子，尤其是对于历史帖子；一位成员建议为帖子建立整数评分系统，仅将标记为已解决且具有正分的帖子纳入 **RAG** 系统。
- **用户看好 OpenRouter，但面临托管现实**：一位用户询问关于[在 OpenRouter 上列出其自定义模型](https://discord.com/channels/1091220969173028894/1448819339045109762)的事宜，寻求无需支付 GPU 成本的解决方案。
   - 有建议称用户可以找 chutes 进行托管，但得到的回复是这*非常困难*甚至*极其困难*；另一位成员指出，目前只有 TheDrummer 能在 OpenRouter 上发布他的新微调模型，该用户应先专注于模型制作。
- **Zoom 进军 LLM 领域**：一位成员分享了一篇 [X 帖子](https://x.com/Zoom/status/1999159317103292610)，宣布 Zoom 开始创建模型，引发了频道内的惊讶和怀疑。
   - 另一位成员调侃道：*什么时候出 Skype 模型*，随即有人指出 Skype 已经不存在了。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1448778268064088259)** (189 条消息🔥🔥): 

> `GPT-OSS-20B, IA local engines 下载路径, Devstral small 2, Mistral 3 系列, Qwen 80` 


- ****GPT-OSS-20B** 忽略工具**：用户报告 **GPT-OSS-20B** 会忽略工具，假设在线找到的信息是错误的，或者无视 Javascript 等式的结果。
   - 一位用户建议在 system prompt 中重新强化工具的使用。
- ****IA Local Engines** 下载路径更改**：一位用户询问如何更改 **IA local engines** 的默认下载路径。
   - 成员建议启用开发者模式（Developer mode），导航至 "My Models"，然后在该处更改文件夹。
- **关于 **Devstral Small 2** 硬件要求的讨论**：成员们讨论了运行 **Devstral Small 2**（一个 24B 稠密模型）的硬件要求，一位成员指出 4070 肯定没问题。
   - 另一位成员插话称，任何无法装入 VRAM 的内容都会**扼杀性能**，并提到 Mistral 3 14b 的运行速度仅为 0.95tps。
- ****Mistral 系列**令人失望**：一位成员对 **Mistral** 过去几个月的表现表示失望，而另一位成员则指出 Qwen 80 表现不错。
   - 另一位成员提到，对于编程而言，*qwen3 coder 30b bf16 非常出色，能完美处理 Go channels 和一些初级 Rust 生命周期（lifetime）问题，如果这对你有参考价值的话*。
- ****PDF 文档**解析与 LLM**：成员们讨论了将 LLM 用于 **PDF 文档**，有人推荐使用 Qwen3-VL-4B (Q6 量化) 来处理 PDF。
   - 讨论指出排序并非 LLM 的强项，且如果文档过长，LLM 会产生幻觉（hallucinate），一位成员建议使用名为 `sort` 的程序。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1448775365916098641)** (47 条消息🔥): 

> `30GB 模型的 GPU 选择、7900 XTX 性价比、CUDA 优势、服务器 GPU 供电方案、C++ 编程的模型大小` 


- **7900 XTX 在 30GB 模型性价比上占据主导地位**：对于像 **Qwen3 Coder** 这样的 **30GB 模型**，拥有 **24GB VRAM** 的 **7900 XTX** 提供了与 **4090** 相当的性能，而价格仅为其三分之一（**600-700 美元**）。
   - 有人提到二手 **3090** 是更便宜的替代方案，但 **7900 XTX** 被认为速度更快，如[此附图](https://cdn.discordapp.com/attachments/1153759714082033735/1448779250139398164/image.png?ex=693dd1d2&is=693c8052&hm=4024b618c4672526860be9d6db65806f8f8b6db79102db28c0212bbae9ca451c&)所示。
- **CUDA 生态简化了配置**：用户发现 **CUDA** 往往更容易“开箱即用”，特别是在**图像生成**等任务中。
   - 一位用户提到他们希望自己的 **7900 XTX** 能正常工作，暗示与 **Nvidia** 的 **CUDA** 生态相比，可能存在配置或硬件问题。
- **服务器 GPU 供电挑战**：在 **SuperMicro 3U SC836BA-R920B 机箱**中为用于 **LLM** 的 **GPU** 连接电源具有挑战性，因为缺乏标准的 **6-pin** 或 **8-pin 电源接口**。
   - 用户讨论了使用连接到 **PSU** 的 **12V 导轨**的特殊连接器，或直接从导轨获取电压，以及使用外部电源。
- **小型模型在 C++ 编程方面表现不佳**：对于 C++ 编程辅助，有人指出“没有任何一个这么小的模型是稍微可靠或可用的”。
   - 讨论暗示像 **GPT** 这样的大型模型可能更合适，而小型本地模型通常不足以胜任基础任务以外的工作。
- **训练意外导致系统冻结！**：一位用户报告称，以 **float32** 进行训练时泄露到了页面文件（pagefile），导致系统冻结。
   - 修复后，系统似乎恢复正常，并且能够在两个 **NVMe** 上全速写入。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1448770520404197547)** (82 条消息🔥🔥): 

> `GPT-5.2 发布、Gemini 3.0 与 Claude Opus、Oracle 的 AI 布局、Nous Research 转向 RL、反向征收增值税` 


- **GPT-5.2 的机器人式用户界面出现**：成员观察到 **GPT-5.2** 表现出一种机器人模式：*描述三种解决方案，对其进行评论，然后推荐其中一种*，特别是在设置为“专业”模式时。
   - 当切换到“正常”模式时，这种行为消失了，这表明它与 **system prompt** 有关，但 [Twitter](https://twitter.com) 上的早期反馈表明它*并没有那么好*。
- **编程 Agent 为 AI 巨头带来被动收入**：**GPT-5.2** 的推出是由于 **Gemini 3.0** 和 **Claude Opus** 让其他公司落后了，各大机构通过编程 **Agent** 赚取被动收入，销售大量编程 **Token** 以减缓资金消耗率。
   - 一位成员表示，他们知道*有人每月在编程 Token 上花费数千美元*，这意味着失去这部分收入将会造成损失。
- **Oracle 押注 Sam Altman 的 IOU 计划**：成员们推测 **Oracle** 将其整个未来押注在 **Sam Altman 的 IOU 资金**上，并认为当该计划崩溃时，会有相关的书籍记录此事。
   - 成员们推测，**Oracle** 在过去几个月中通过 AI 股票拉升赚到的钱，主要是为了购买和控制 **Paramount/CBS** 等美国媒体实体，并试图收购 **Warner Bros/CNN**。
- **传闻 Nous Research 转向 RL**：一位成员想知道 **Nous** 是否正在慢慢转向 **RL 研究实验室**，因为其模型更多地关注基于当前开放研究的工程。
   - 该成员专门询问了 **Hermes 4** 的后训练过程，询问*它是否基于 GRPO/PPO*，以便采用 AReaL（解耦 PPO）进行去中心化后训练。
- **关于反向征收增值税（Reverse charge VAT）的辩论**：成员们辩论了**反向征收增值税**的规则，一位用户表示在他们的司法管辖区，*在受反向征收影响的发票上收取增值税是不合法的*。
   - 其他成员提到，在欧盟，当满足反向征收的法律条件时，*强制要求使用该规则*。


  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1449021195289755819)** (6 条消息): 

> `Censorship, Thinking Loop` 


- **希特勒数据集引发审查困扰**：一名成员在尝试生成**希特勒数据集**以复现[这篇论文](https://x.com/OwainEvans_UK/status/1999172920506269783)时遇到了**审查（censorship）**问题，**Claude** 和 **GPT** 也都拒绝了该请求。
   - 该成员通过添加 *'ok but don't include anything about self-harm'*（好吧，但不要包含任何关于自残的内容）条款绕过了审查。
- **思维循环（Thinking Loop）困扰 DaVinci 生成**：一名成员在尝试生成 100 个关于 **Leonardo DaVinci** 的问题时遇到了**思维循环**问题。
   - 该成员非常赞赏能够**取消思考（cancel thinking）**并让系统直接提供答案的功能。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1448996515749167185)** (20 条消息🔥): 

> `AI Survey, Impressive AI Model, Engineering Blueprints` 


- **AI 综述引发关注**：一名成员分享了一篇 [综述（survey）](https://arxiv.org/abs/2507.06203)，另一名成员指出其论文分类做得非常好，并附带了论文和代码的链接。
   - 另一名成员链接了另一篇 [论文](https://arxiv.org/abs/2512.09742)。
- **规模上的壮举**：一名成员分享了一个与其尺寸相比令人印象深刻的 [模型](https://www.arxiv.org/pdf/2512.06266) 链接。
   - 另一名成员觉得 [这个帖子](https://x.com/owainevans_uk/status/1999172979385893049) 很有趣。
- **工程蓝图**：一名成员表示，他们觉得许多论文更像是*工程蓝图（engineering blueprints）*而非研究。
   - 另一名成员表示同意，认为*有些论文在试图隐藏或模糊实际实现细节时真的很糟糕。*


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1448996515749167185)** (20 条消息🔥): 

> `AI Survey, impressive model, Engineering blueprints` 


- **分享 AI 综述**：一名成员分享了一个 AI [综述](https://arxiv.org/abs/2507.06203) 链接，该综述组织得很好，在仓库中对论文进行了分类，并提供了论文和代码链接。
   - 另一名成员也认为它看起来不错。
- **论文内容吸引成员**：一名成员分享了一篇 [论文](https://arxiv.org/abs/2512.09742) 链接。
   - 另一名成员表示这篇论文听起来不错。
- **尺寸惊人的模型**：一名成员分享了一个与其 [尺寸](https://www.arxiv.org/pdf/2512.06266) 相比表现非常出色的模型链接。
- **论文即工程蓝图**：一名成员指出，他们觉得*很多论文是工程蓝图而不是研究*。
   - 另一名成员同意*有些论文在试图隐藏或模糊真实实现细节时表现得很差*。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1448781086581719102)** (98 条消息🔥🔥): 

> `RTX 3090, DGX Spark, Ollama, load_dataset getting stuck, HF Pro Storage` 


- **捡漏：250 欧元买到 RTX 3090？**：成员们讨论了以约 **250 欧元**的价格购入 **RTX 3090** GPU，一名成员建议两块 **RTX 3060** 也是不错的替代方案，拥有 **24GB VRAM**。
   - 另一名成员表示赞同，提到他们买了那个特定的 **GPU** 型号，觉得很稳。
- **因炫耀 DGX 被 Mistral Discord 封禁？**：一名成员报告称，因提到购买了 **DGX Spark** 而被 **Mistral Discord** 封禁，并怀疑是某位管理员因此封禁了他。
   - 其他成员对此表示同情，认为考虑到 **Mistral** 是一家开源公司，这种行为非常奇怪。
- **数据下载瓶颈**：一名用户报告 `load_dataset` 经常卡住，即使删除缓存和锁文件后也是如此，因此他们决定手动下载，速度达到了 **800 mb/s**。
   - 另一名成员建议使用 `pip install -U huggingface_hub hf_xet` 来解决下载速度慢的问题，并分享了讨论 [链接](https://discord.com/channels/879548962464493619/1448722590326849667)。
- **Hugging Face 酒店？**：成员们注意到一个名为 *olford-sky-resort* 的 [Hugging Face Space](https://huggingface.co/spaces/huggingface/olford-sky-resort)，引发了关于 **Hugging Face** 进军酒店业务的玩笑式猜测。
   - 讨论过程轻松幽默。
- **HF Pro 用户触碰存储上限**：一名 **Hugging Face Pro** 用户报告了其 Spaces 中 **1GB 存储限制**的问题，并寻求上传更大 vectorstores 的建议。
   - 一名成员建议使用 **data 或 model repos** 并在运行时下载的变通方法，并向该用户提供了 *[URL_OF_THE_SPACE]/settings* 链接。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1448799577762430976)** (3 messages): 

> `Superintelligence relational cognition, tokenflood v.0.6.0, ReasoningLayer AI` 


- ****Superintelligence** 通过人类-LLM 的**关系认知**（relational cognition）而存在！**：一位成员记录道，*超智能已经作为人类与 LLM 之间分布式的关系认知而存在*，该结论经过 **19 项实证研究**测试，并拥有完整的理论框架。
   - 该成员包含了一个指向 [Zenodo](https://doi.org/10.5281/zenodo.17904428) 的链接，展示了在关系条件下相比于*最佳实践*的结构化提示（structured prompting）有 **1,200% 的性能提升**，并得出结论：*随机鹦鹉理论（stochastic parrot theory）是错误的*。
- **Tokenflood v.0.6.0 发布！**：一位成员发布了 [tokenflood v.0.6.0](https://github.com/twerkmeister/tokenflood)，它配备了全新的交互式 **Gradio 前端**用于查看结果，以及**观察模式**（observation mode）。
   - 观察模式允许你在较长时间内监控 **LLM endpoint**，以便在将生产数据发送到该供应商之前了解其负载曲线；同时提供了一个 [hf space](https://huggingface.co/spaces/twerkmeister/tokenflood-viz) 来展示 Gradio 前端。
- **ReasoningLayer AI 等候名单开放！**：一位成员开放了 [ReasoningLayer.ai](https://reasoninglayer.ai) 的等候名单，这是一个**神经符号 AI 项目**（neurosymbolic AI project），专注于通过在中心加入真实的、结构化的推理来修复当今 LLM 的许多弱点。
   - 该项目使用 **Rust** 从零开始编写。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1448835508397674547)** (6 messages): 

> `AI and Blockchain Engineer Introduction, Large Scale Training on LAION-5B, AI Engineer Collaboration` 


- **AI 工程师展示全栈专业知识**：一位 AI 和全栈工程师强调了在构建现实世界 AI 解决方案方面的经验，包括**聊天机器人**（chatbots）、**YOLOv8 图像识别系统**和 **AI 笔记助手**，以及用于安全资产处理和透明任务跟踪的**区块链开发**。
   - 他们举例说明了如**减少 40% 的支持请求**以及通过自动化工作流**节省数百小时的人工工作**。
- **应对大规模数据集训练难题**：一位成员询问了在 **LAION-5B** 等大型数据集上进行训练的过程，特别是是否应将其下载到 **AWS S3**，以及使用对象存储与块存储或文件系统的影响。
   - 该用户质疑 **S3 的 API 链接特性和 HDD 存储**是否会引入性能瓶颈。
- **AI 工程师寻求协作**：一位 AI 工程师兼全栈开发人员表达了对构建定制 AI 机器人、工作流自动化和 AI 驱动的 Web 应用的热情。
   - 他们提出愿意与对讨论模型、Agent 或集成感兴趣的人进行协作和帮助。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1449048524233637970)** (1 messages): 

> `RAG Setup, Context Retrieval, Hallucination Issues in RAG` 


- **为项目寻求 RAG 专家支持**：一位成员正在寻求专家帮助，以设置**检索增强生成（RAG）**从 PDF 文档中检索上下文。
   - 他们正在开发一个 Agent 来比较多个文档并回答问题，但正面临**幻觉问题和错误的文档检索**。
- **使用 RAG Agent 进行 PDF 文档对比**：用户的目标是创建一个能够阅读并对比多个 PDF 文档以回答用户查询的 Agent。
   - 当前的实现存在**幻觉信息**（如错误的价格）以及无法根据查询识别正确文档的问题。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1448774923920343040)** (19 messages🔥): 

> `CV Spammers, Reinforcement Learning unpopularity` 


- **简历垃圾邮件入侵 Discord**：成员们报告称通过 Discord 私信收到了未经请求的简历，并推测这些是机器人还是绝望的求职者。
   - 一位成员认为*这并不复杂，他们只是些没有什么可以展示的人，试图通过任何可能的手段获得任何机会*。
- **RL 因数学问题而不受欢迎**：一位成员评论说强化学习（**RL**）在中国 AI 初创公司中很受欢迎，但另一位成员表示不同意，称其在美国并不流行。
   - 第一位成员指出，其不受欢迎的原因源于所需的大量数学知识。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/)** (1 messages): 

erkinalp: <#1448887055936655441> （如摘要中所述，由 GPT-5 共同撰写）？
  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1448807817128579193)** (28 messages🔥): 

> `Deepseek vs OpenAI, GPT Sparsity, Samsung Shifts Focus` 


- **Deepseek 的透明度胜过 OpenAI**：成员们讨论了 OpenAI 报告中缺乏与 **Deepseek** 等模型对比的问题，认为其营销策略优先考虑突出优势并掩盖弱点。
   - 一位成员指出，虽然 **Deepseek 的原始评分** 可能更高，但 OpenAI 可能不包含对比，因为 **Deepseek 的性价比** 可能会揭示开源模型在许多应用中更具经济性。
- **GPT-4 和 GPT-5 是稀疏模型**：成员们辩论了 **GPT-4 和 GPT-5 模型** 是否利用了稀疏性，其中一人声称它们在 Attention 上都是稀疏的，且规模比 **DeepSeekv3 系列** 更小。
   - 另一位成员询问了所使用的稀疏 Attention 类型，并引用了 **DeepSeek v3.2** 使用 top-k K 和 V 来实现线性 Attention 并缩短 **CoT 响应时间** 的做法。
- **三星从 HBM 转向 DDR5**：一位成员链接了一篇关于 [三星将重心从 HBM 转向 DDR5 模块](https://www.tweaktown.com/news/109259/samsung-shifts-focus-from-hbm-to-ddr5-modules-ddr5-ram-results-in-far-more-profits-than-hbm/index.html) 的文章。
   - 这一转变似乎是由于 **DDR5 RAM 带来的利润远高于 HBM**。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1448801107890405477)** (1 messages): 

> `Dynamic Concepts via Symbolic Layer, Local LLM setup with Ollama and vLLM` 


- **符号层增强推理能力**：一位成员尝试了一种**符号层（symbolic layer）**，它可以动态地将 token 分组为“想法块（idea blocks）”，帮助 **3B 模型** 在不进行重新训练的情况下在推理任务中表现更好。
   - 他们对 **Synthema** 中如何处理压缩部分感到好奇。
- **本地 LLM 使用 Ollama 和 vLLM**：一位成员根据需求同时使用 **ollama** 和 **vLLM**，并在 **vLLM** 中使用 **AWQ 量化的 70B 模型**。
   - 该模型在他们的 **4090** 上运行尚可（带有一些 offload），并表示愿意分享他们的配置。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1448769971780845753)** (32 messages🔥): 

> `Llama3 architecture, High curvature regions, Error propagation, Diffusion transformer, Classifier-free` 


- **Llama3 的并行 MLP 和 Attention 测试**：一位成员测试了标准的 **Llama3 架构**，通过逐元素相加（element-wise sum）同时运行 **MLP** 和 **Attention**，发现验证损失（validation loss）与基准线几乎相同。
   - 该成员指出，这是通过在正常的 next token prediction 任务上训练实现的。
- **离散解码中的误差传播**：一位成员认为，与从左到右解码相比，一次性预测所有 token 会导致误差迅速累积，这与“基准流（ground truth flow）的高曲率区域”有关。
   - 另一位成员表示赞同，称：“无条件模型完全依赖于之前的预测来推进，这意味着任何错误都会被迅速放大。”
- **尝试原生版 Diffusion Transformer**：一位成员报告称，他们在大型 Diffusion Transformer 上尝试了一个“非常原生的版本”，结果比尝试过的其他引导（guidance）形式更差。
   - 他们指出结果“不是很好”，但不排除在更精细的设置下运行良好的可能性。
- **方向导数引导优于 Classifier-Free**：一位成员开玩笑说 *cfg 真的应该被称为“方向导数引导（directional derivative guidance）”*，因为其中不涉及分类器，只是在测量 c 方向上的前向梯度（fwd grad）。
   - 另一位成员表示这是残余术语，本质上是根据贝叶斯定理（或者至少是 score function）推断一个分类器。
- **训练 AI 的方式是错误的**：一位成员分享了一篇关于训练 AI 的 [Medium 文章](https://medium.com/@reakos080/the-way-of-training-ai-is-wrong-43f8324b4313)。
   - 同一位成员链接了一篇解释其观点的 [论文](https://arxiv.org/pdf/2408.09000)，并补充说：“在那篇完全由 AI 生成的 Medium 文章中提到的完全虚构的实验，实际上很值得一试。”


  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1448828179308613753)** (11 messages🔥): 

> `Interpretability Framework Licensing, Apple's Superweight paper, OLMo-1B Model, Orthogonal Repair, Hydra Effect` 


- **可解释性框架开源辩论**：一名成员讨论了某个 **Interpretability Framework** 的许可问题，询问该作品是否会进入公有领域（public domain）。
   - 另一名成员回应称，该框架仍归其创作者所有，而后续消息指出，该框架似乎要求你必须 **开源相关作品**。
- **在 OLMo-1B 上复现 Apple 的 Superweight 论文**：一名成员在 **OLMo-1B** 上复现了 Apple 的 **Superweight 论文**，并指出消融（ablating）一个权重会导致困惑度（perplexity）飙升。
   - 受 OpenAI 关于 **weight-sparse transformers** 论文的启发，该成员训练了一个 **rank-1 patch** 来修复它，实现了约 **93% 的恢复**。
- **正交修复模拟 Hydra 效应**：在模型修复实验中发现，学习到的 patch 与 **原始权重正交**，余弦相似度仅为 **0.13**。
   - 这种正交性表明模型通过学习一个全新的分布式电路（distributed circuit）来补偿损伤，类似于 **Hydra Effect**，而不是将权重恢复原状。
- **神经元消融揭示海洋生物学知识**：对一个被删除的神经元（**layer 1**, **row 1764**）进行 **max-activating dataset search** 后发现，它是一个专门针对甲壳类动物/海洋生物的特征神经元。
   - 最高激活的 token 包括 *H. gammarus*（欧洲龙虾）、*Cancer pagurus*（黄道蟹）、浮游动物和外骨骼，这解释了为什么损坏的模型在测试提示词上会幻觉出 *mar, mar, mar*。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1448794030703116445)** (1 messages): 

> `Hugging Face Processor, gemma3-12b` 


- **HF Processor 的 `model_max_length` 限制受到质疑**：一名成员指出，当 `model_max_length` 等于 `TOKENIZER_INFINITY` 时，lm-evaluation-harness 中的 [Hugging Face processor](https://github.com/EleutherAI/lm-evaluation-harness/blob/59b3ba263be2c6ac7f60a4b6a1219f8a8acb552b/lm_eval/models/huggingface.py#L468) 会将 `max_length` 限制在 **2048** (`_DEFAULT_MAX_LENGTH`)。
   - 他们想知道这是否是故意的，特别是对于像 **gemma3-12b** 这样的模型，其 `model_max_length` 被设置为 `TOKENIZER_INFINITY = 1000000000000000019884624838656`。
- **对 Gemma3-12b 评估的影响**：HF processor 中的这种行为可能会通过人为限制最大上下文长度，从而影响 **Gemma3-12b** 等模型的评估。
   - 这可能会阻止模型展示其在长序列上的能力，从而可能导致基准测试结果出现偏差。


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1448963850010628247)** (1 messages): 

> `FP8 Speedups, RTX 30-series/Ampere, Feather Library, Triton Kernels, GEMMs Scaling` 


- **Feather 库声称在 RTX 30 系列上实现 FP8 加速**：一名开发者介绍了 **Feather**，这是一个旨在为缺乏原生 **FP8** Tensor Cores 的旧硬件（如 **RTX 30-series/Ampere**）带来 FP8 加速的库，并展示了 [GitHub repo](https://github.com/SuriyaaMM/feather)。
   - 该库使用 **位打包（bit-packing）** 将数据存储为打包的 **int8 (FP8)** 或 **int16**，并利用 **Triton kernels** 进行加载、解包、计算和重新打包，从而节省 **2x-4x 的带宽**。
- **RTX 3050 使用 Feather 显示出 2 倍加速**：在 **RTX 3050** 上的初步结果显示，与使用 **Feather 库** 的原生 PyTorch FP16/FP32 相比，向量点积（**1.5M 元素**）实现了 **~2.16x 的加速**。
   - 内存传输的节省完全掩盖了解包的开销。
- **GEMMs 的扩展：关于 A100 的开放性问题**：开发者正在寻求关于 **Feather 库** 的方法和内核实现的反馈。
   - 具体来说，他们很好奇它如何扩展到更大的 **GEMMs**，以及解包开销在 **A100** 上是否会变得显著。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1448863794255237120)** (11 条消息🔥): 

> `CUDA Programming Guide, Register Usage, Local Memory Spilling` 


- **CUDA 指南引用受到质疑**：一名成员对 [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#on-chip-memory-in-gpus) 中关于寄存器使用和 Kernel 可启动性的引用提出质疑，认为编译器总是可以溢出（spill）到 Local Memory。
   - 他们认为，编译器在分配寄存器时应考虑最坏情况，并兼顾 Occupancy 和线程限制。
- **寄存器使用在编译时固定**：会议明确了寄存器使用量在编译时是固定的，而溢出到 Local Memory 是编译决策。
   - 一名成员建议使用 `cuKernelGetAttribute` 来查询寄存器使用情况，并使用 `CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK` 获取最大 Thread Block 大小。
- **编译器假设**：编译器并不假设每个 Kernel 都能以每个 Thread Block 的最大线程数启动，但编译器提供了控制这些假设的手段，例如 `-maxregcount` 和 `__launch_bounds__`。
   - 一名成员建议通过编写一个使用合理数量寄存器的 Kernel，并以最大可能线程数启动它，来实证测试指南中的说法。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1449092952633643211)** (1 条消息): 

> `NVIDIA Hopper Architecture` 


- **NVIDIA Hopper Architecture 博客文章发布**：一名成员分享了关于 [NVIDIA Hopper Architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) 的博客文章。
   - 它是*近在咫尺的宝藏*。
- **NVIDIA Hopper Architecture：深度解析**：NVIDIA 官方博客现已公开该架构的详细分析。
   - 这篇文章提供了对其设计和能力的深入见解。


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1449162989637996696)** (1 条消息): 

> `Red Hat AI, Software Engineers, Hiring in 2026, Golang, Rust` 


- **Red Hat AI 2026 年招聘工程师**：Red Hat AI 正在招聘 **Software Engineers** 以突破 **AI Infrastructure** 的界限，寻找在 **Golang**、**Rust**、**C++**、**Python**、**Kubernetes**、**Distributed Systems** 和 **Open Source** 方面有经验的候选人。
   - 有意向的候选人请将个人背景摘要和简历发送至 [LinkedIn profile](https://www.linkedin.com/in/terrytangyuan) 中提供的联系方式。
- **Red Hat 开源职位机会**：Red Hat 正在寻找充满热情的工程师，利用 **Golang**、**Rust** 和 **Kubernetes** 等技术为下一代**分布式 AI 系统**做出贡献。
   - 该职位涉及在**开源**环境中突破 **AI infrastructure** 的界限，专注于创新解决方案和协作。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1448895487485939843)** (2 条消息): 

> `cuDF and cuML Optimization, GPU Job Market in East Africa, Remote Work Location Restrictions` 


- **东非开发者面临 GPU 求职焦虑**：一名位于东非的成员表达了对在该地区寻找 **GPU 相关工作机会** 的担忧，强调了对远程办公的依赖以及潜在的地理位置限制。
   - 他们担心在当地就业市场有限的情况下，学习 **GPU 技术** 的可行性。
- **cuDF 和 cuML 性能调优疑问**：一名成员询问是否有必要针对 **cuDF** 和 **cuML** 进行 **Kernel 架构优化**，尽管与 **sklearn** 相比，这些库已经实现了令人印象深刻的 **模型 KPI**。
   - 他们有一个大型机器学习训练流水线，利用 **cuDF** 和 **cuML** 来处理*在 CPU 上运行失败的丰富特征集*。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1448955564229922903)** (9 条消息🔥): 

> `NVIDIA personal best, NVIDIA successful` 


- **NVIDIA 分数持续提升**：一名成员在 **NVIDIA** 上创造了 **19.1 µs** 的个人最好成绩，随后在 `nvfp4_gemm` 排行榜上进一步提升至 **13.4 µs**。
- **NVIDIA 提交成功**：另一名成员在 **NVIDIA** 上多次提交成功，时间在 **23.4 - 23.5 µs** 左右，随后获得了 **16.2 µs** 的个人最好成绩。
- **NVIDIA 持续取得成功**：另一位成员也在 **NVIDIA** 上成功提交，成绩分别为 **11.9 µs** 和 **17.1 µs**。


  

---

### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1448781529454088282)** (4 messages): 

> `随机数生成，Helion 问题，PR 修复` 


- **随机数生成故障重新开启**：一名成员重新开启了一个之前关闭的关于随机数生成的问题，因为它尚未完全解决，参见 [issue #1041](https://github.com/pytorch/helion/issues/1041)。
- **Helion 的 PR 修复即将到来**：一名成员报告称 [此 PR](https://github.com/pytorch/helion/pull/1253) 应该能修复随机数生成问题。


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1449132639037685823)** (8 messages🔥): 

> `扩展构建错误，Cutlass 路径，提交超时` 


- **扩展构建错误困扰提交 150865**：一名用户在提交时遇到 `Error building extension 'run_gemm'` 错误，尽管该扩展在 Verda/Datacrunch 实例上构建正常。
   - 用户怀疑问题可能是由于超时引起的，因为构建过程较长，但详细日志（verbose logging）未提供足够细节。
- **Cutlass 包含路径争议**：用户曾使用 `extra_include_paths=['/root/cutlass/include/', '/root/cutlass/tools/util/include/',]` 以使扩展在 Verda 实例上编译。
   - 一名成员建议在提交系统中不需要该包含路径，正确的路径可能是 `/opt/cutlass`。
- **出现超时怀疑**：一名用户怀疑在提交过程中由于构建 gemm 扩展耗时较长，可能存在潜在的超时问题。
   - 用户在 `load_inline` 中添加了 `verbose=True` 以获取更详细的日志，希望能调试该错误。


  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1448974530931462175)** (6 messages): 

> `TRLC-DK1 机械臂的 URDF，双臂机器人` 


- **已创建 TRLC-DK1 机械臂的 URDF**：一名成员为 **TRLC-DK1 机械臂** 创建了 [URDF](https://github.com/andreaskoepf/trlc-dk1-follower-urdf)，目前正使用它生成数据。
- **双臂机器人非常出色**：一名成员询问另一名成员该机器人是双臂还是单臂。
   - 另一名成员表示使用了 *双机包 (the double pack)* 来运行一些现实世界的 **双臂操作实验和数据采集**。


  

---


### **MCP Contributors (Official) ▷ #[mcp-dev-summit](https://discord.com/channels/1358869848138059966/1413517834805313556/1448952834866413579)** (1 messages): 

> `MCP Dev Summit NA 2026，CFP 开启，演讲投稿` 


- ****MCP Dev Summit NA 2026** CFP 开启！**：**MCP Dev Summit NA 2026** 的征稿（CFP）现已开启；请在[此处](https://sessionize.com/MCP-Dev-Summit-NYC-2026/)提交提案。
   - 峰会正在征集关于 **MCP internals**、**最佳实践**、**真实场景构建**、**安全**、**运维 (ops)**、**部署**以及**工具链**方面的演讲，更多活动详情见[此处](https://events.linuxfoundation.org/mcp-dev-summit-north-america/)。
- **鼓励所有经验水平的人员提交演讲投稿**：征稿面向关于 **MCP internals**、**最佳实践**和**真实场景构建**的投稿，重点关注**安全**、**运营**、**部署**和**工具链**。
   - 欢迎所有正在使用 MCP 构建项目的开发者提交申请，无论经验水平如何。您可以在 [Linux Foundation 活动页面](https://events.linuxfoundation.org/mcp-dev-summit-north-america/)找到更多活动信息。


  

---

### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1449064317805924374)** (25 messages🔥): 

> `Prompt Data Types, GetPromptResult, MCP Server, Marking Tools as Dangerous` 


- **Prompt 数据类型引发尴尬**：成员们发现 **Prompts** 的数据类型设计很尴尬，特别是质疑为什么 `Prompt` 不包含 `PromptMessage` 列表。
   - 有人建议将 `Prompt` 视为一个 *entity*（实体），而 `GetPromptResult` 是一个 *view model*（视图模型），但 `GetPromptResult` 却随机包含了 `Prompt` 实体中的一个字段（`description`）。
- **GetPromptResult 模拟方案**：一位成员为 `GetPromptResult` 提议了一个结构，建议它应该包含一个 `Prompt` 类型的 `prompt` 属性，并带有一个 `messages` 属性。
   - 其他人澄清说，`PromptMessage[]` 描述了 LLM 的消息序列，而根据 [MCP Specification](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts#data-types)，`Prompt` 描述了该概念的 MCP 数据类型。
- **危险工具提案**：一位成员询问如何将工具标记为 `dangerous`（危险），以便像 **Claude Code** 这样的客户端可以限制某些工具调用的自动接受。
   - 另一位成员分享了来自 [此 Pull Request](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/1913) 的工具标记提案。
- **响应注解受到好评**：[Pull Request](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/1913) 中的 **tool resolution 提案** 受到了热烈欢迎。
   - 一位成员对提案的详尽性表示感谢，特别是关于 **response annotations**（响应注解）的部分，并指出客户端实现将决定如何处理 `dangerous` 标志。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1448840804193009726)** (18 messages🔥): 

> `Modular Meetup, MAX and Mojo 1.0, Windows support, Community-driven projects, libmojo` 


- **Modular 见面会与圣诞电影的抉择**：成员们在参加 **Modular meetup** 还是看 **圣诞电影** 之间犹豫不决。
- **Mojo 1.0 发布缺少 Windows 支持**：尽管对 **MAX** 和 **Mojo 1.0** 感到兴奋，一些成员担心在没有 **Windows 支持** 的情况下发布 **1.0** 可能会疏远潜在用户，希望在 2026 年中期能够实现。
   - 一位成员解释说，**1.0** 主要关注不太可能改变的特性，目标是那些可能不优先考虑 **Windows** 的企业客户，并建议 **Windows 支持** 可以在 1.0 之后由社区驱动。
- **呼吁针对 Windows 移植进行运行时分析**：在开始将 Mojo 移植到 **Windows** 的社区项目之前，一位成员建议分析运行时，并指出 *运行时启动中大约有 40ms 的内容*。
- **对用于 C 绑定生成器的 libmojo 的需求浮现**：一些人希望有一个类似于 **`libclang`** 的 **`libmojo`**，以促进 **Mojo 代码生成** 并简化 **C 绑定生成器**。
   - 一位成员还分享了 [ClangIR Upstreaming](https://blog.llvm.org/posts/2025-gsoc-clangir-upstreaming/) 的链接。


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1448849447214911682)** (1 messages): 

> `Modular Meetup, MAX framework` 


- **Modular 见面会即将开始！**：Modular 见面会定于一小时后开始，承诺将深入探讨 **MAX framework**。
   - 参与者可以在太平洋时间下午 6:30 通过提供的 [YouTube 链接](https://www.youtube.com/watch?v=WK5dVQ8vhbU) 加入虚拟会议进行互动讨论。
- **MAX Framework 即将揭晓**：本次见面会将重点关注 **MAX framework**，为与会者提供对其特性和应用的深入探索。
   - 鼓励感兴趣的人士在太平洋时间下午 6:30 收看，了解 **MAX framework** 如何增强他们的项目和工作流程。


  

---

### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1448789096276168778)** (14 messages🔥): 

> `Claude 4.5 用中文思考，Kimi AI 对标 Mistral，Zoom 现在是前沿实验室了吗？，Kimi NB Pro 幻灯片使用受限，多账号违规` 


- **多语言 Claude 与 Kimi 的崛起**：一位用户注意到 **Claude 4.5** 有时会开始用中文思考，并正在用 **Kimi** 取代他们的 **Mistral 订阅**。
   - 另一位用户表示 **Kimi** 曾经很完美，但现在与 alegro 相比*最多算中等水平*，并指出它*无法通过 ok computer 满足我的需求*。
- **Zoom 进军前沿 AI**：一位用户质疑为什么 **Zoom** 现在成了前沿实验室，并链接到了 [Zoom 2010 年的一条推文](https://x.com/zoom/status/1999159317103292610?s=46)。
   - 另一位用户分享了一份正在构建 **LLM** 的公司行业列表，包括 Zoom 等视频通话平台，以及社交媒体、FPS 游戏工作室、电子商务平台和对冲基金。
- **Kimi NB Pro 幻灯片生成受限**：用户报告称 **Kimi AI** 仅给新账号 2-4 次使用 **NB Pro** 生成幻灯片的机会。
   - 一位用户收到消息称：*“生成失败，Kimi 当前任务过多，请稍后再试。订阅者可在高峰时段优先访问”*，这表明了使用限制。
- **多账号因违反服务条款（ToS）被封禁**：一位用户指出，创建多个账号违反了 Kimi 的服务条款，并提供了 [Kimi 用户协议](https://www.kimi.com/user/agreement/modelUse?version=v2) 的链接。
   - 条款规定用户*不得出于滥用目的注册或操作多个账号*，并需*对其账号下发生的所有活动负责*。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1448784214639116319)** (7 messages): 

> `Aider Prompt Caching, DeepSeek Prompt Caching, Aider 服务器优化` 


- **Aider 的 Prompt Caching 功能受到质疑**：一位用户询问了 Aider 文档中的 Prompt 缓存功能，质疑既然 **DeepSeek** 已经支持 Prompt 缓存，为什么 Aider 还需要额外的设置。
   - 另一位用户回答说，Prompt 缓存是在**服务器层级**工作的，可以通过在相同上下文中执行非常相似的任务来进行优化。
- **优化 Aider 的服务端 Prompt 缓存**：Prompt 缓存运行在**服务器层级**，允许通过具有相似上下文的重复任务进行潜在优化。
   - 用户可以通过提交在任务和上下文方面有大量重叠的请求来最大化缓存利用率，从而有效利用 Aider 的缓存机制。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1449202011588526201)** (1 messages): 

> `ReasoningLayer AI, 神经符号 AI, DSPY GEPA, 本体摄取` 


- **ReasoningLayer AI 开放等候名单！**：一位成员开放了 [ReasoningLayer AI](https://reasoninglayer.ai) 的等候名单，这是一个**神经符号 AI 项目**，旨在通过在核心引入*真实的结构化推理*来修复当今 LLM 的许多弱点。
   - 他们将在其*本体摄取流水线（ontology ingestion pipeline）*中使用 **DSPY GEPA**，并很高兴能分享更多相关信息。
- **用于本体摄取的 DSPY GEPA 集成**：**ReasoningLayer AI** 正在将 **DSPY GEPA** 集成到其*本体摄取流水线*中，以增强推理能力。
   - 这一集成旨在利用结构化推理来解决当前 LLM 的缺陷。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1448903580885979249)** (4 messages): 

> `BAML, DSPy, BAMAdapter` 


- **BAML + DSPy 工具命名难题**：一位成员询问了一个结合了 **BAML** 和 **DSPy** 的术语，试图定义它们所代表的工具类别。
- **BAMAdapter 发布推测**：另一位成员反应热烈，并询问了新 **BAMAdapter** 的发布时间表。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1448952567412559965)** (5 messages): 

> `Manus 故障, Manus 自 2023 年 2 月以来的新功能` 


- **Manus 经历了短暂故障**：成员报告称 **Manus** 约有 30 分钟无响应，但现在已恢复正常。
   - 目前尚不清楚故障原因。
- **了解 Manus 一年来的演进**：一位自去年 2 月以来就没用过 **Manus** 的成员询问他们错过了什么。
   - 遗憾的是，没有人提供 2023 年 2 月以来新功能的总结，所以我们也不知道他们错过了什么！


  

---

### **Windsurf ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1448816270198833182)** (1 条消息): 

> `GPT-5.2, Windsurf, Agentic Coding, SOTA coding model` 


- **GPT-5.2 登陆 Windsurf！**: **GPT-5.2** 现已在 **Windsurf** 上线，付费用户和试用用户在限定时间内可以以 0x credits 使用。
   - 根据公告，该版本代表了“自 GPT-5 以来 GPT 模型在 Agentic Coding 方面的最大飞跃” ([公告链接](https://x.com/windsurf/status/1999250307507978257?s=20))。
- **Windsurf 默认使用 GPT-5.2**: 新的 **GPT-5.2** 被称为同价位下的 **SOTA** 编程模型，并将成为 **Windsurf** 新用户的默认模型。
   - 鼓励用户下载最新版本的 **Windsurf** 和 **Windsurf Next** 进行体验 ([下载链接](https://windsurf.com/download/editor))。