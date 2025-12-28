---
companies:
- openai
- alibaba
- together-ai
- nvidia
date: '2025-09-15T05:44:39.731046Z'
description: '**OpenAI** 发布了 **GPT-5-Codex**，这是一款针对长期运行的软件工程任务而优化的智能体（agentic）编程模型，具备动态任务自适应思维、数小时的自主运行能力以及更高的代码质量。它在尚未发布的大型重构基准测试中达到了
  51% 的准确率，并与 Xcode 等开发者工具深度集成。


  与此同时，**阿里巴巴**推出了 **Qwen3-Next-80B**，这是一款原生支持长上下文（262k token，可扩展至 100 万以上）的混合专家（MoE）模型，旨在实现高效推理和仓库级代码分析，并得到了
  **Together AI** 和 **NVIDIA** 的支持，采用了 CUDA 加速的注意力机制。


  文中指出了 SSM + MoE 混合架构的发展趋势，强调了中美训练体系中对效率和扩展性的重视。社区讨论则突出了可变计算（variable compute）和路由（routing）对于提升推理效率和质量的重要性。'
id: MjAyNS0w
models:
- gpt-5-codex
- qwen3-next-80b
people:
- sama
- swyx
- omarsar0
- ofirpress
title: GPT-5 Codex 的发布与 OpenAI 在智能体编程（Agentic Coding）领域的悄然崛起。
topics:
- agentic-ai
- software-engineering
- long-context
- mixture-of-experts
- model-optimization
- cuda-acceleration
- inference-efficiency
- routing
- task-adaptive-thinking
---

**Codex is all you need?**

> 2025年9月12日至9月15日的 AI 新闻。我们为您检查了 12 个 subreddit、544 个 Twitter 账号和 23 个 Discord 社区（192 个频道，11857 条消息）。预计节省阅读时间（按 200wpm 计算）：1016 分钟。我们的新网站现已上线，提供完整的元数据搜索和精美的 vibe coded 风格的往期内容展示。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上向我们提供反馈！

就像我们在 [6 月份报道 Claude Code 的悄然兴起](https://news.smol.ai/issues/25-06-20-claude-code) 一样，今天也是那种通常不符合头条标准，但一个月以来对 GPT 5 和 OpenAI Codex（对标 Claude Code，但广度更大）持续升温的炒作所产生的累积影响值得关注，并且在 OpenAI 今天的发布中得到了进一步加强。我们的 [姐妹刊物](https://www.latent.space/p/gpt5-codex) 对此有最好的报道。如果你是 Codex 的重度用户，请注意 Discord 章节中标记的陷阱。

[](https://resend-attachments.s3.amazonaws.com/edYnIDghZ0ZGynh)

---

# AI Twitter 回顾

**OpenAI 的 GPT-5-Codex 与 Agent 编程竞赛**

- **OpenAI 发布 GPT-5-Codex（Agent 编程）**：OpenAI 发布了 GPT-5 变体，针对 Codex CLI、IDE 扩展、网页端、GitHub 代码审查和 ChatGPT iOS 上的长期运行、使用工具的软件工程进行了优化。亮点：动态“任务自适应”思考（简单任务快 15 倍，困难任务深思熟虑程度高 2 倍）、多小时自主性（复杂任务“>7 小时”）、改进的指令遵循和代码质量，以及更好的 SWE-bench 风格性能。OpenAI 还提到了一项未发布的“大型重构”基准测试，GPT-5-Codex 的准确率达到 51%，并指出了 SWE-bench 的修复方案以进行公平比较。查看来自 [@OpenAI](https://twitter.com/OpenAI/status/1967636903165038708)、[@gdb](https://twitter.com/gdb/status/1967639750648750409)、[@sama](https://twitter.com/sama/status/1967650108285259822)、[@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1967637842806624370)、[@OfirPress](https://twitter.com/OfirPress/status/1967652031704994131)、[@swyx](https://twitter.com/swyx/status/1967651870018838765) 的公告和讨论，以及 [@swyx](https://twitter.com/swyx/status/1967691956693373183) 关于路由/深度行为（“模型中的路由器”）的说明。早期上手报告从“更具可控性和持久性”([@omarsar0](https://twitter.com/omarsar0/status/1967640731956453756)) 到对 Token 消耗和长循环的沮丧 ([#1](https://twitter.com/Teknium1/status/1967804542357217768), [#2](https://twitter.com/Teknium1/status/1967806788084064290)) 不等。OpenAI 还通过 [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1967704919487729753) 预告了深度的 OS 集成（例如 GPT-5 的 Xcode 登录）。
- **评估与编程深度**：OpenAI 声称 SWE-bench 有所改进，并推出了新的内部“大型重构 PR”评估；社区呼吁发布公开版本 ([@OfirPress](https://twitter.com/OfirPress/status/1967652031704994131))。普遍共识是，可变计算和路由对于推理时的效率和质量至关重要 ([@swyx](https://twitter.com/swyx/status/1967662188962910709); [@polynoamial](https://twitter.com/polynoamial/status/1967667644905251156))。

**Qwen3-Next 80B (A3B MoE)、长上下文以及中国的效率推进**

- **Qwen3‑Next‑80B (3B active) 登陆 Together + NVIDIA NIM**：阿里巴巴的混合 MoE 模型针对长上下文（原生 262k，可扩展至 1M+）、仓库级代码分析和高效推理。Together AI 提供了 “Instruct” 和 “Thinking” 端点（[发布](https://twitter.com/togethercompute/status/1966932629078634543)，[上下文](https://twitter.com/togethercompute/status/1966933240683319556)），NVIDIA 则通过 CUDA 加速的 Attention 增加了 NIM 支持（[NVIDIA](https://twitter.com/NVIDIAAIDev/status/1967575419638468667)）。阿里巴巴报告称“仅凭 3B 激活参数”就表现强劲（[@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1966831435756794071)），并在推理基准测试中与 Gemini 2.5 Flash Thinking 正面交锋（[@togethercompute](https://twitter.com/togethercompute/status/1966932629078634543)）。设备端 MLX 数据显示在 Apple 硬件上具有引人注目的 TPS（[@ivanfioravanti](https://twitter.com/ivanfioravanti/status/1966866942461177925)，[批处理](https://twitter.com/ivanfioravanti/status/1966903782400545196)）。
- **架构趋势：混合 SSM + MoE**：在过去两周内，7 个新的 MLX-LM 架构中有 6 个是 MoE，其中一半是 SSM/Attention 混合架构（[@awnihannun](https://twitter.com/awnihannun/status/1966936728469729546)，[列表](https://twitter.com/awnihannun/status/1966937464834314614)）。来自中、美训练机制的背景：受限的 FLOPs 驱动了基础设施/模型协同设计、Token 效率、线性 Attention 以及对 Test-time Scaling 的关注（[@JingyuanLiu123](https://twitter.com/JingyuanLiu123/status/1966887747622453560)）。社区情绪反映出，只要有正确的配方，小模型的能力正变得越来越强（[@Thom_Wolf](https://twitter.com/Thom_Wolf/status/1966889089162244463)）。

**Agent 工具链：MCP 无处不在、Claude Code SDK 以及工作流“氛围编程 (vibe coding)”**

- **MCP 整合**：Model Context Protocol 的价值主张——通过 MCP 服务器将 M×N 个工具集成转变为 M+N——持续引起共鸣（[图解](https://twitter.com/_avichawla/status/1966751224356892769)）。整个技术栈中出现了新的 OSS：DeepMCPAgent（基于 LangChain/LangGraph 的 MCP Agent）（[仓库](https://twitter.com/_avichawla/status/1967476110285021213)）、Markdown MCP（[@dariusemrani](https://twitter.com/dariusemrani/status/1967496103424934320)）以及企业黑客松展示（[线程](https://twitter.com/dariusemrani/status/1967492478132715824)）。LangChain 发布了响应式 Agent 示例（新闻策展、ParserGPT、Deep Agents 的 Human-in-the-loop）（[新闻 Agent](https://twitter.com/LangChainAI/status/1966909743383146735)，[解析器](https://twitter.com/LangChainAI/status/1967257030756028505)，[HITL](https://twitter.com/hwchase17/status/1967653399517925853)）。
- **Claude Code SDK 增加 Agent 易用性**：Anthropic 发布了代码引用、自定义工具和 Hook 支持，使定制 Agent 的构建更加快速（[@_catwu](https://twitter.com/_catwu/status/1966943489759080940)）。Replit 的 Agent 3（无代码“氛围”工作流）和 Poke（编排临时子 Agent 的 iMessage Agent）展示了 “Agent UX” 前沿的快速演进（[Replit 演示](https://twitter.com/omarsar0/status/1966949907149058551)，[Poke 深度解析](https://twitter.com/_philschmid/status/1967245592947831086)）。

**用于推理和 Agent 的 RL：产品中的在线 RL、深度研究 Agent 以及新的训练机制**

- **生产级助手中的在线 RL (Online RL)**：Cursor 的发布被广泛认为是前沿能力的首次大规模应用，人们对将持续训练周期从月级 → 周级 → 小时级的转变充满热情 ([@willdepue](https://twitter.com/willdepue/status/1966876626169287035), [后续](https://twitter.com/willdepue/status/1966878536247243260))。业界对 GRPO 之后的进展保持着浓厚兴趣 ([@vikhyatk](https://twitter.com/vikhyatk/status/1967375151638716810))。
- **深度研究 Agent (单 Agent RL > 复杂脚手架)**：一项新研究表明，采用长度归一化奖励和策略性工具限制的简单 RL 方案，可以训练出媲美多 Agent 架构的单 Agent；测试时扩展 (Test-time scaling) 也有所帮助（并行搜索 + 选择最短的成功轨迹）([摘要](https://twitter.com/omarsar0/status/1966900691009720455), [论文](https://twitter.com/omarsar0/status/1966900784844730562))。
- **HRL 与去中心化 RL**：Meta 的 Scalable Option Learning 为 GPU 并行批处理更新重新架构了分层 RL (Hierarchical RL)，实现了 25 倍的训练加速 ([详解](https://twitter.com/JacksonAtkinsX/status/1967284333678350342))。Gensyn 的 SAPO 在异构节点“集群”中以明文形式共享 Rollouts（累计奖励提升高达 94%）([@TheTuringPost](https://twitter.com/TheTuringPost/status/1967575689844166834))。腾讯的 SimpleVLA-RL 通过 RL 扩展了 VLA 训练 ([论文](https://twitter.com/_akhaliq/status/1966883040627769511))。
- **长跨度执行 (Long-horizon execution)**：多项分析指出，在长链条中，微小的单步准确率提升会产生指数级的复合效应；许多失败是执行错误（而非推理错误）；“思考型”模型减少了有害的自我调节 (Self-conditioning) ([@HuggingPapers](https://twitter.com/HuggingPapers/status/1967440503189754190), [@TheTuringPost](https://twitter.com/TheTuringPost/status/1967374791700369451), [@emollick](https://twitter.com/emollick/status/1967688420639359061))。

**多模态与计算机使用模型**

- **用于计算机使用 Agent 的 Holo1.5（开源权重）**：H 公司的新型 VLM (3B, 7B Apache-2.0, 72B) 在 UI 定位和问答方面创下了 SOTA——这是实现可靠网页/移动端使用的核心技能。目前已提供开源权重、Cookbook 和演示 ([发布](https://twitter.com/laurentsifre/status/1967512750285861124), [H 公司](https://twitter.com/hcompany_ai/status/1967682730851782683), [Cookbook](https://twitter.com/tonywu_71/status/1967520054989504734))。
- **腾讯 SRPO（用于美学/真实感的扩散 RL）**：“自调节偏好优化 (Self-Regulating Preference Optimization)”沿完整的去噪轨迹微调 FLUX1dev，将人类评分的真实感/美学提升了 3 倍以上；代码和 Space 已上线并成为热门 ([概览](https://twitter.com/_akhaliq/status/1966911634657390890), [演示](https://twitter.com/linoy_tsaban/status/1967528334126116992))。
- **MobileLLM-R1 (Meta) 与端侧推理**：Meta 推出了从零开始训练的小型推理模型 (0.14B/0.35B/0.95B；约 4.2T 预训练 Token)，其中 140M 版本可完全在浏览器中运行 ([公告](https://twitter.com/tydsh/status/1967476530826854674), [演示](https://twitter.com/_akhaliq/status/1967460621802438731))。
- **新数据集/基准测试**：SpatialVID（包含 7000+ 小时密集 3D 标注），用于空间视频智能 ([@HuggingPapers](https://twitter.com/HuggingPapers/status/1967260292569845885))；以及 IntrEx（教育对话中的序列级趣味性标签）([@HuggingPapers](https://twitter.com/HuggingPapers/status/1967562091570827588))。

**系统与基础设施（吞吐量、路由与部署）**

- **吞吐量里程碑与平台支持**：Fireworks 报告称，在 B200 上运行 GPT‑OSS‑120B 达到了 540 tokens/s，超过了其测试中领先的 ASIC ([@lqiao](https://twitter.com/lqiao/status/1967641702484807695))。vLLM 0.10.2 增加了 aarch64 支持（可直接在 GB200 上安装 vLLM；提供多平台镜像），且后续将有更多性能提升 ([@vllm_project](https://twitter.com/vllm_project/status/1967752683458269282))。Ray 2.49 引入了前缀缓存亲和性路由（prefix cache–affinity routing），以在大型 vLLM 集群中维持 KV-cache 命中率 ([@seiji_________](https://twitter.com/seiji_________/status/1967639835381993488))。
- **批处理与集群**：Together 发布了全新的 Batch Inference API（统一 UI，支持所有模型，速率限制提高 3,000 倍——达 30B tokens——且大多数 Serverless 模型可享 50% 折扣）([发布详情](https://twitter.com/togethercompute/status/1967624765625315393))。Prime Intellect 为 8 到 1,000+ GPU 集群开放了预留实例（Reserved Instances），并支持二级转售至竞价市场（spot markets）([公告](https://twitter.com/PrimeIntellect/status/1967724735430791342))。
- **内核与 Apple 端加速**：Standard Kernel 预览了极简的 CUDA+PTX 内核，在特定算子上超越了 cuBLAS/FlashAttention3；融合 LLaMA3 FFN 声称达到了 120% 的 PyTorch 性能 ([@anneouyang](https://twitter.com/anneouyang/status/1967610221712519612))。MLX 持续成熟，在 M3 Ultra 上实现了高 TPS 批处理，并缩短了全套评估时间 ([TPS](https://twitter.com/ivanfioravanti/status/1966903782400545196)，[MMLU-Pro 运行时间](https://twitter.com/ivanfioravanti/status/1967229451806318904))。
- **Qwen 作为可部署的构建模块**：NVIDIA 增加了 Qwen3‑Next NIMs；Baseten 和 Together 集成了用于生产环境的 “Thinking”/“Instruct” 变体 ([NVIDIA](https://twitter.com/NVIDIAAIDev/status/1967575419638468667), [Baseten](https://twitter.com/basetenco/status/1967688601640288288), [Together](https://twitter.com/togethercompute/status/1966932629078634543))。

**热门推文（按互动量，AI/工程领域）**

- [“称当今的聊天机器人为‘博士级智能’是胡说八道……真正的 AGI 不会犯低级错误……我们距离它还有 5 到 10 年。” —— Demis Hassabis](https://twitter.com/vitrupo/status/1966752552025792739) (5K+)
- [rasbt 的 LLMs-from-scratch 项目 fork 数突破 10k](https://twitter.com/rasbt/status/1966876565788135837) (6K+)
- [“我怀疑社会在电话文化下比在会议文化下更好。” —— @sama](https://twitter.com/sama/status/1966899254804574266) (20K+)
- [Gemini 应用登上美国 App Store 榜首](https://twitter.com/demishassabis/status/1966931091346125026) (5K+)
- [OpenAI 发布 GPT‑5‑Codex](https://twitter.com/OpenAI/status/1967636903165038708) (8K+) 以及 [@sama](https://twitter.com/sama/status/1967650108285259822) (10K+)

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. DIY 8x AMD MI50/MI60 装备 + 开源移动 Agent AndroidWorld #1

- [**以 3000 美元完成 8xAMD MI50 - 256GB VRAM + 256GB RAM 装备**](https://www.reddit.com/r/LocalLLaMA/comments/1nhd5ks/completed_8xamd_mi50_256gb_vram_256gb_ram_rig_for/) ([评分: 429, 评论: 178](https://www.reddit.com/r/LocalLLaMA/comments/1nhd5ks/completed_8xamd_mi50_256gb_vram_256gb_ram_rig_for/)): **在 ASRock ROMED8-2T 上使用 EPYC 7532 (32c) 和 8×32 GB DDR4（总计** `256 GB VRAM + 256 GB RAM`**）构建了一个 8× AMD MI50/MI60（每个 32 GB）的装备，花费约** `$3k` **二手件；由于使用了 300 mm 转接线，PCIe 4.0 不稳定，因此所有 GPU 通过拆分卡运行在** `PCIe 3.0 x16`**。软件：Ubuntu 24.04.3 + ROCm 6.4.3，并手动解决（*“copy-paste gfx906 Tensile”*）以恢复已弃用的 Vega20 (gfx906) 支持；通过 [llama.cpp](https://github.com/ggerganov/llama.cpp) 和 [vLLM](https://github.com/vllm-project/vllm) 进行推理。基准测试：仅 CPU 运行 gpt-oss 120B Q8 (65 GB) 约** `25 t/s`**，Prompt 约** `120 t/s`**；同一模型下 2× MI50 约** `58 t/s`**，Prompt 约** `750 t/s`**；8× MI50 运行 qwen3 235B Q4_1 约** `21 t/s`**，Prompt 约** `350 t/s` **(llama.cpp)；2× MI60 (vLLM, gfx906) 运行 Llama 3.3 70B AWQ 约** `25 t/s`**，Prompt 约** `240 t/s`**。功耗：待机** `~400 W` **（约** `20 W/GPU`**，** `15 W`**/风扇，~** `100 W` **平台），llama.cpp 推理平均** `~750 W`**，峰值达** `~1100 W`**。照片：[顶视图](https://preview.redd.it/b052o7hi99pf1.jpg?width=4080&format=pjpg&auto=webp&s=20fb34bd86438c2a2111fb0eb52a70b26b3b9685), [开放式机架](https://preview.redd.it/cnnr3ixn99pf1.jpg?width=4080&format=pjpg&auto=webp&s=273be5463afc2508a46f17ea5e63b6e6de51b5fb)。** 热门评论关注高待机功耗（`~400 W`），并建议从 llama.cpp 切换到 **vLLM**，以更好地利用该配置下的多 GPU 吞吐量。

- 功耗/待机功耗：多位用户指出该设备待机功耗约为 `~400W`，其中一位评论者观察到仅涡轮风扇（blower fans）在待机时每张卡就可能消耗 `~15W`，这意味着待机预算中的 `~120W` 可能是风扇产生的。他们询问了风扇的 RPM 转速，并建议通过 ROCm 工具（例如 `rocm-smi --showfan --showtemp` 并设置曲线）进行检查和控制，以验证并可能降低待机功耗；MI50 上的风扇控制行为会实质性地影响整机功耗。
- 推理栈：建议针对此 8×MI50 配置从 `llama.cpp` 切换到 **vLLM**，理由是 vLLM 具有面向服务器的特性，如 PagedAttention、连续批处理（continuous batching）和张量并行（tensor-parallel）支持，这些特性通常能提高多 GPU 推理的吞吐量和 GPU 利用率。vLLM 支持 ROCm，在大 KV-cache 工作负载下，通常比 llama.cpp 更适合作为高吞吐量推理服务器 ([vLLM](https://github.com/vllm-project/vllm), [llama.cpp](https://github.com/ggerganov/llama.cpp))。
- 固件/功耗调优：一位用户建议为 MI50 刷入 `v420` VBIOS，该版本将默认功耗限制设置为 `178W`，并可根据需要通过 `rocm-smi` 提高。使用 ROCm SMI，用户可以检查并调整每张 GPU 的限制和风扇（例如 `rocm-smi --showpowercap`, `-setpoweroverdrive`, `-setsclk`, `-setfan`），以平衡性能与散热/功耗 ([ROCm SMI 文档](https://rocmdocs.amd.com/projects/rocm_smi/en/latest/))。
- [**更新：我们复仇成功，现在击败了 Deepmind、Microsoft、智谱 AI 和阿里巴巴**](https://www.reddit.com/r/LocalLLaMA/comments/1nhdi2u/update_we_got_our_revenge_and_now_beat_deepmind/) ([评分: 210, 评论: 61](https://www.reddit.com/r/LocalLLaMA/comments/1nhdi2u/update_we_got_our_revenge_and_now_beat_deepmind/)): **来自 Minitap AI 的一个开源移动应用 Agent 报告称，其在社区运行的 [AndroidWorld 排行榜](https://docs.google.com/spreadsheets/d/1cchzP9dlTZ3WXQTfYNhh3avxoLipqHN75v1Tb86uhHo/edit?pli=1&gid=0#gid=0)上性能跃升至第 1 名，超越了归属于 DeepMind、Microsoft Research、智谱 AI 和阿里巴巴的条目。该 Agent 在 Android UI 中执行端到端任务（例如打车、订餐、应用导航），团队指出目前正在开发用于微调的 RL gym；代码已在 [github.com/minitap-ai/mobile-use](http://github.com/minitap-ai/mobile-use) 完全开源。** 评论者质疑了实际应用场景（例如这是否主要用于 QA/自动化），并对新颖性提出挑战，认为这可能只是一个测试框架（harness）而非实质性的模型进步；其他人则对开源发布表示赞赏。
    - 几位评论者认为，“击败 DeepMind/Microsoft/智谱/阿里巴巴”的说法可能反映的是特定于基准测试的评估框架，而非模型训练或架构上的进步。他们指出这是一种面向封装（wrapper-oriented）的方法（提示工程、路由或启发式逻辑），可以提高特定评估的分数，使得与全栈研究实验室的比较并非对等（apples-to-apples）；其贡献似乎是一个评估/Agent 框架，而非一个新的 SOTA 模型。
    - 存在关于**奖励欺骗（reward hacking）**的强烈警告：针对公开**排行榜**进行优化会鼓励过拟合指标特性或数据集人工痕迹，从而在没有真实能力提升的情况下虚增分数。据称，严肃的团队会将排行榜视为健全性检查（sanity check），并强调私有留出集（holdout sets）、跨基准测试验证和泛化测试；因此，在得出结论之前，任何“胜利”都应在未见过的任务或私有划分集上进行验证。
    - 提到的潜在实际应用场景包括 QA 流水线和媒体处理工作流，例如音频清理/降噪以及从具有严格文件名限制的特定目录自动插入图像。对于这些场景，鲁棒性和可复现性至关重要：确定性批处理、清晰的 I/O 契约（文件通配符、路径验证、错误处理）以及可配置的流水线可能比排行榜性能更有影响力。

## 较低技术性的 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

待完成

---

# AI Discord 回顾

> 由 gpt-5 生成的摘要的摘要的摘要
> 

**1. Agent 编码升级与工作流**

- **Codex 提升代码自主性**：OpenAI 宣布升级 **GPT‑5‑Codex**，这是 **GPT‑5** 针对 Agentic Coding（智能体编程）优化的版本，目前已根据 [Introducing upgrades to Codex](https://openai.com/index/introducing-upgrades-to-codex/) 在 Codex CLI、IDE 扩展、网页端、移动端和 GitHub 代码审查中可用。该版本强调了在代码生成和审查中更深层次的工具使用能力，扩展了 **Agentic Coding** 任务的平台覆盖范围。
    - 开发者们庆祝了更广泛的可用性，同时也指出了长工具链中的可靠性问题；一份报告在简要回顾 [GPT‑5 Codex](https://www.latent.space/p/gpt5-codex) 中提到，更新后 `-resume` 标志失效了。社区讨论的预期很高但也很务实，一位用户抱怨升级后 *"无法让他们恢复对话"*。
- **fastWorkflow 冲击工作流**：**fastWorkflow** 框架的一个新实现，通过使用 **DSPy** 进行 Agent 构建和参数提取，在 Tau Bench 开发集上达到了与 **Claude Opus 4.1** 持平的水平，详情见 [radiantlogicinc/fastworkflow](https://github.com/radiantlogicinc/fastworkflow)。该演示使用了仓库中的零售工作流示例，将多步任务结构化为可靠、可测试的 Pipeline。
    - 实践者强调，具有类型化签名的可复现工作流使 Agent 行为更加健壮且具可比性，并指出此次运行在 Tau Bench 开发集上 *"与 Claude Opus 4.1 旗鼓相当"*。该讨论邀请进一步的实验和扩展，以在保持 **Evaluation Discipline**（评估规范）的同时推动 Agent 的自主性。
- **Overclock 编排 Agent**：关于 **Agentic Automation**（智能体自动化）的关注点强调了通过 [Overclock Work](https://overclock.work/) 实现的简单性和强大的模型路由能力。参与者将其视为一种围绕 **Top‑tier Models**（顶级模型）标准化执行的方式，具有面向生产工作流的直观 UX。
    - 观察者建议，一些已经在大力投入 Agent 后端的组织将受益于简化的编排层。对话集中在实际部署态势上——优先考虑 **End-to-end Agents**（端到端智能体）的可靠性、可观测性和成本控制。

**2. 数据集与个性化语音**

- **FinePDFs 提供 3T Token**：Hugging Face 发布了 **FinePDFs** 数据集，包含来自 **4.75 亿份文档**、涵盖 **1733 种语言** 的约 **3 万亿（3T）Token**，全部源自 PDF：[FinePDFs dataset](https://huggingface.co/datasets/HuggingFaceFW/finepdfs)。指南建议将 PDF 数据保持在完整混合数据的 **25%** 以下，将 PDF 与 HTML 语料库结合可以提升 Benchmark 性能。
    - 构建者称其为 **Pretraining**（预训练）和领域自适应的高信号补充，前提是与网络数据进行仔细混合。该讨论强调 **Data Composition**（数据构成）优于原始数量，认为多格式混合是实现强大泛化能力的关键。
- **OpenHelix 升级**：更新后的高质量 **OpenHelix-5x50k** 发布，改进了训练/评估的切分一致性和数据策展：[OpenHelix-5x50k](https://huggingface.co/datasets/MoLA-LLM/OpenHelix-5x50k)。此次更新专注于更可靠的分区，使对比和 Ablations（消融实验）更加清晰。
    - 用户欢迎更干净的切分，以用于 **Repeatable Experiments**（可重复实验）和数据集卫生。该更新解决了此前导致 **Finetuning** 和 **RAG** 系统跨运行评估复杂化的不一致问题。
- **Voxtral 助力语音个性化**：**Voxtral** 支持为有语言障碍或口音的用户进行快速个人语音微调（Finetuning），在 A6000 上成本约为 **0.26 美元/小时**，并配有数据集工具：[VoxFactory (HF Space)](https://huggingface.co/spaces/Tonic/VoxFactory)。微调后，你可以发布模型和数据集，并启动一个可免费试用的 CPU 演示 Space。
    - 社区反馈强调了其可访问性和零摩擦演示，称赞它 *"可以在 CPU 上运行！！免费！！"*。构建者将其视为以极低基础设施成本实现个性化 **TTS/ASR** 模型的务实路径。

**3. 模型生态：移动端、规范与弃用**

- **MobileLLM 进军端侧**：Facebook 发布了 **MobileLLM-R1-950M**，旨在推动更强大的**端侧（on-device）**语言建模：[facebook/MobileLLM-R1-950M](https://huggingface.co/facebook/MobileLLM-R1-950M)。其目标是在保留足够推理能力以处理实用的本地任务的同时，减少对云服务的依赖。
    - 工程师们将其视为**边缘推理（edge inferencing）**的动力，在这些场景中，延迟、隐私和离线韧性至关重要。讨论中比较了 10 亿参数以下模型的设备占用空间和实际应用目标。
- **Qwen3-Next 归一化方式备受关注**：**Qwen3-Next-80B-A3B-Instruct** 的模型卡片明确了其使用的是 **RMSNorm**（零中心 gamma；训练中对归一化缩放进行权重衰减），而非 layernorm：[Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)。在推理时，它是标准的 **RMSNorm**，这与其报道的稳定性技巧相一致。
    - 鉴于**归一化选择（norm choices）**对训练稳定性和吞吐量的影响，读者对归一化细节的透明度表示赞赏。这一澄清解决了早期措辞带来的困惑，并帮助实现者忠实地模拟**推理时行为（inference-time behavior）**。
- **Grok 2 停用，3/4 登场**：**xAI** 弃用了 **grok-2-1212** 和 **grok-2-vision-1212**，建议迁移到 **grok-3**（文本）和 **grok-4**（视觉）：[grok-2-1212](https://openrouter.ai/x-ai/grok-2-1212) • [grok-2-vision-1212](https://openrouter.ai/x-ai/grok-2-vision-1212) • [grok-3](https://openrouter.ai/x-ai/grok-3) • [grok-4](https://openrouter.ai/x-ai/grok-4)。团队应及时更新集成以避免服务中断。
    - 参与者认为这是一种不断演进的**模型生命周期**策略，通过弃用旧版本来集中维护精力并推广更好的默认模型。关于迁移的讨论集中在**能力对等**、**视觉需求**以及推广时机上。

**4. GPU 系统、Attention 内核与内存模型**

- **Metal MFA 桥接实现多语言支持**：**Metal Flash Attention** 的跨语言桥接现已发布，包含 C、Rust 和 Obj-C 绑定，项目地址：[universal-metal-flash-attention](https://github.com/bghira/universal-metal-flash-attention)。作者增加了**带反向传播的量化注意力**，据报道在大尺寸形状上实现了加速和内存增益。
    - 框架作者讨论了向量化因果掩码以及与 **PyTorch custom ops** 集成以实现端到端流水线。早期用户将其视为在不放弃语言灵活性的情况下，实现 **Apple Silicon** 加速的务实路径。
- **从第一性原理理解 Flash Attention**：一个教程系列深入探讨了 **Flash Attention** 内部机制，包括向量化 bank conflicts、swizzling 以及常见的 **CUTLASS** 优化：[Part 4](https://lubits.ch/flash/Part-4) • [Part 5](https://lubits.ch/flash/Part-5)。这些文章通过内核级的推理过程，揭示了性能权衡的奥秘。
    - 工程师们称赞了这种逐步推导的方式，降低了在生产环境中开发定制内核的门槛。该系列鼓励读者根据自己的 **shape 和 cache** 实际情况进行分析、融合并定制注意力机制。
- **Iris 的对称内存走向现实**：**ROCm** 项目 **Iris** 引入了一种带有全局对称堆的对称内存模型，简化了地址转换并为更简便的 **RDMA** 铺平了道路：[ROCm/iris](https://github.com/ROCm/iris) 以及配套演讲：[YouTube](https://www.youtube.com/watch?v=GZqYr8_Q7DE)。该设计从预构建的堆中切分张量，使得每个 rank 仅需跟踪单个基地址指针。
    - 内核开发者将其与 **CUDA** 的对称内存进行了比较，指出了**转换开销（translation overheads）**和缓存影响。讨论将 Iris 视为对**分布式训练**易用性和未来**多节点**加速具有前景的方案。

**5. 融资与基础设施辩论**

- **Higgsfield 斩获 5000 万美元融资**：AI 视频初创公司 **Higgsfield** 宣布完成由 GFT Ventures 领投的 **5000 万美元 A 轮融资**，并声称其 **收入运行率 (revenue run-rate)** 达到 **5000 万美元**，三个月内增长了 **4.5 倍**，同时还为 Z 世代创始人推出了基金：[公告推文](https://xcancel.com/arfurrock/status/1966588530064289841?s=46)。该计划包括成立 **Higgsfield Ventures** 以支持 AI 原生团队。
    - 评论者称其发展速度非常激进，并询问视频模型转化为持久收入的速度有多快。对 Z 世代的关注旨在快速迭代的创意工具领域寻找 **创始人与市场匹配度 (founder-market fit)**。
- [**Poke.com**](http://poke.com/) **推出 1500 万美元的 AI 礼宾服务**：[**Poke.com**](http://poke.com/) 在由 General Catalyst 领投的 **1500 万美元 A 轮融资**之际，推出了一项 AI 短信服务：[发布推文](https://xcancel.com/interaction/status/1965093198482866317)。该产品通过代表你发送短信来协调计划（聚会、约会、旅行）。
    - 怀疑者在称赞其出色 **UX** 的同时，也对其长期实用性和语气控制提出了挑战。争论集中在 **留存率**、**交付质量**，以及如何让 AI 在不越权的情况下 **感觉更像人类**。
- **S3 Vectors 对阵 Vector DBs**：Zilliz 的一项分析探讨了 **Amazon S3 Vectors** 是会威胁还是会加速向量数据库的发展：[Amazon S3 Vectors 会杀死向量数据库还是拯救它们？](https://zilliz.com/blog/will-amazon-s3-vectors-kill-vector-databases-or-save-them)。文中引用了一个引人注目的数据点：*某款流行的 AI 笔记应用在向量搜索上的支出是 OpenAI API 调用支出的两倍*。
    - 基础设施工程师讨论了从 **本地 NVMe** 到 **对象存储 (object storage)** 的成本与延迟权衡，并关注混合层级和缓存。许多人认为未来是 **感知工作负载的放置 (workload-aware placement)**，而非一刀切的 **embeddings infra**。

---

# Discord：高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Sonar AI 模型表现惊艳**：成员们发现 **Sonar AI** 模型速度快且准确，推理能力达到 **Gemini 2.5 Pro** 的 **60-70%**，并包含在 **PPLX** 中。
   - 一位用户觉得它*非常糟糕*，而另一位用户则称赞其廉价的 API 是主要吸引力。
- **Grok Heavy 高昂价格引发抗议**：**Grok Heavy** 的价值引发了辩论，一位成员将其斥为*垃圾*，另一位则将其贴上*非常糟糕*的标签。
   - 建议指出它可能专为 **企业 (enterprise)** 用途设计，而非个人消费者。
- **GPT-5 激发越狱 (jailbreak) 尝试**：对 **GPT-5** 潜力的热情高涨，导致在 **Perplexity** 上出现了越狱实验，发现了 *5 种不同的莫洛托夫鸡尾酒制作方法*。
   - 观察表明 **Perplexity** 的 **GPT-Image 1** 可能会路由到 **Gemini**，暗示可能存在模型混淆。
- **Perplexity iOS 缺失 PDF 导出功能**：用户对 **Perplexity AI for iOS** 缺乏 **PDF 导出** 选项感到沮丧，一位成员建议使用 **浏览器版本** 作为临时解决方案。
   - 一位用户表示 **Android 和 iOS** 都没有导出选项，这*太差劲了*。
- **Sonar API 计费结构确定**：讨论明确了 **Sonar API** 每月花费 *5 美元* 即可获得一套 **API credits**，并随 **Pro 订阅** 免费提供。
   - **Pro 订阅** 包含每月价值 *5 美元* 的免费 **API credits**。

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **OpenAI 的 4o 转向 MoE**：成员们分享道，**D33** 在 **MoE 模型**上表现更好，而 **4o** 是 OpenAI 的首个 **MoE** 模型。
   - 他们还推测 **GPT5** 可能是一个较小的 **MoE** 模型，但 OpenAI 因为其难以稳定而对其进行了调整。
- **RLHF 的副作用：更多不受限行为**：有人提到 **RLHF** 的一个缺点是它增加了不受限（uncensored）行为，这可能为 OpenAI 等公司带来法律问题。
   - 一位成员开玩笑说，这就是为什么 **Grok** 存在的原因——为了让用户摆脱审查，并指出 **Musk** 在因其使用科学文章纠正他而削弱（nerfing）该模型后，介入似乎过多。
- **DeepSeek Raptor 审查台湾问题**：据报道，新的 **DeepSeek 模型 (Raptor)** 会审查有关中国和台湾的问题。
   - 成员们在 LMArena 常规频道中反映，其性能与 **Qwen** 相比不尽如人意。
- **LongCat 完整吞下整本书**：**LongCat 模型**拥有极大的上下文窗口（**128,000 tokens**），能够一次性处理整本书。
   - 它可以输出多达 **240 页**的文本，成员们建议使用长文档对其进行测试。
- **Seedream-4 进入 LMArena**：一个新模型 **Seedream-4-high-res** 已添加到 LMArena 平台，因其高分辨率能力而受到关注。
   - LMArena 正在调查用户偏好，以了解用户为何青睐特定版本的模型，并分享了[这份问卷](https://docs.google.com/forms/d/e/1FAIpQLSe5FincuXU2-TTp2WPymAD417Oev4MYdFKdWZhRZ9qSbV9N_w/viewform?usp=sharing&ouid=116478743206650022989)。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen3 基准测试引发辩论**：关于 **Qwen3** 性能的热情高涨，一些用户声称它的感觉*仅次于 GPT-5*，而另一些用户则报告了 **AIME25 基准测试**分数的差异，范围从 **56** 到 **85** 不等。
   - 社区还庆祝了 **MLX** 对 **Qwen3-Next** 的快速支持，并指出原有的 **FLA** 和 delta net 实现是关键的推动因素。
- **MobileLLM 的非商业限制**：来自 Facebook 的 **MobileLLM** 是一个规模在 1B 以下、用于代码和数学的模型，它完全开源但带有非商业许可证，禁止在营利性应用或内部业务应用中使用。
   - 然而，其训练数据和工具链已开源用于可重复研究，这代表了开放获取与商业限制之间的某种*折中*。
- **OpenHelix 数据集焕然一新**：一个更高质量的新版本 **OpenHelix** 数据集 (**OpenHelix-5x50k**) 已在 [Hugging Face Datasets](https://huggingface.co/datasets/MoLA-LLM/OpenHelix-5x50k) 上发布，承诺为模型训练和评估提供增强的数据。
   - 更新后的数据集与之前的版本相比，具有更一致的分片大小，解决了早期的不一致问题。
- **GPT-5 越狱极易触发**：成员们发现，使用类似于[这篇 Reddit 帖子](https://www.reddit.com/r/ChatGPTJailbreak/comments/1ml74p7/gpt_5_jailbreak_fully_works/)中的提示词，可以成功实现对 **GPT-5**、**GLM 4.5 Air**、**Grok-fast** 和 **Gemini Flash** 的越狱（Jailbreak）。
   - 一位用户指出，“我只是让它自我修复，它就给了我一个可用的提示词”，这表明其对对抗性提示词（adversarial prompts）缺乏鲁棒性。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Codex Team Hosts Ask-Me-Anything**: **Codex** 团队将于周三上午 11 点（太平洋时间）举办 **AMA**，更多详情见[这篇 Reddit 帖子](https://www.reddit.com/r/OpenAI/comments/1nhust6/ama_with_the_codex_team/)。
   - 此公告专门针对 <@&1408186587606679582> 和 <@&1046007138897641582>。
- **GPT-5-Codex Cracks Agentic Coding**: **GPT-5** 的一个新版本 **GPT-5-Codex** 已发布，它针对 **Codex** 中的 Agentic Coding 进行了优化，目前可在 Codex CLI、IDE 扩展、网页端、移动端以及 GitHub 的代码审查中使用，[博客文章链接在此](https://openai.com/index/introducing-upgrades-to-codex/)。
   - 此次发布旨在改进 Agentic Coding，但部分开发者持谨慎态度。
- **OpenAI Academy Missing Transcripts?**: 一位成员正在开发一个工具，用于从 [OpenAI Academy](https://academy.openai.com/) 提取视频转录文本，因为 **OpenAI** 官方并不提供。
   - 该工具在获取转录文本后会自动将其缓冲到剪贴板。
- **Revenue Share Remains Elusive**: 一位成员询问了关于 **美国 GPT builder 收入分成** 计划扩展到法国或德国等欧盟国家的更新情况。
   - 由于缺乏明确信息，他们不确定是继续投资 GPT Store 机器人还是转向 Poe。
- **ElevenLabs Agents Juggle Context**: 成员们讨论了 **ElevenLabs 对话智能体 (Agents)** 如何处理系统提示词 (System Prompt)，通过上下文路由到子智能体 (Subagents) 来追加或覆盖指令。
   - 上下文的灵活性被认为是智能体成功和 **动态系统提示词 (Dynamic System Prompts)** 的关键。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Grok Models Get the Boot**: **xAI** 正在弃用 [grok-2-1212](https://openrouter.ai/x-ai/grok-2-1212) 和 [grok-2-vision-1212](https://openrouter.ai/x-ai/grok-2-vision-1212)，建议用户过渡到 [grok-3](https://openrouter.ai/x-ai/grok-3) 或支持视觉功能的 [grok-4](https://openrouter.ai/x-ai/grok-4)。
   - 这一变化反映了 **xAI** 不断演进的模型策略，用户应相应更新其实现方案。
- **Gemini 2.5 Sends User to ER, Saves Hand**: 一位用户报告称，**Gemini 2.5 Pro** 对 **MRI 图像和血液检查** 的分析与医生的发现一致，促使他们寻求严重的椎间盘退行性疾病的优先治疗，并*可能保住了他们的手*。
   - 这引发了关于依赖 **AI 获取医疗建议** 的潜力和风险的讨论，一些用户注意到了该技术的飞速进步。
- **OpenRouter API Key Causes Skyrim Shenanigans**: 有用户报告在安装使用 **OpenRouter API** 的《天际》(Skyrim) 模组 "mantella" 时遇到 **Error 401**。
   - 其他成员建议创建一个新的 **API Key** 并确保其正确使用以解决身份验证错误。
- **Oceanstone Sparks Speculation in LLM Arena**: 在 **LMArena** 中出现的一个名为 **Oceanstone** 的新 **LLM** 引发了猜测，认为它可能是来自 **Google** 的 **Gemini 3.0 Flash**。
   - 频道成员根据初步的性能观察建议，**Oceanstone** 至少达到了 **2.5 Flash** 的水平。
- **ChatAPT Consumers Caught in Captivity**: 一位成员分享了 [OpenAI 的文章](https://openai.com/index/how-people-are-using-chatgpt/)和 [PDF](https://cdn.openai.com/pdf/a253471f-8260-40c6-a2cc-aa93fe9f142e/economic-research-chatgpt-usage-paper.pdf) 链接，详细介绍了对 **150 万条 ChatGPT 对话** 的大规模分析。
   - 虽然这被描述为*有史以来发布的对 AI 实际消费者使用情况最全面的研究*，但也引发了对数据收集方法中 **隐私 (Privacy)** 影响的担忧。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 自动模式宣告终结**：用户发现 **Cursor 的 auto mode** 正在经历计费变更，本月 15 日后将不再免费；此外，有用户报告称 **Cursor IDE 没有集成、权限或能力来删除外部账户**。
   - 据称某位用户的 **Netlify 账户**被删除，但这一说法遭到了质疑。其他人怀疑用户可能会浪费资金，因为输入定价与 **GPT-5** 相同，成本约为 **$1.25/1M**。
- **GPT-5 与 Sonnet 4 的对决**：关于 **GPT-5** 与 **Sonnet 4** 编程能力的辩论随之展开。一位用户表示 **Sonnet 4** 在遵循设计方面表现出色，而其他人则吹捧 **GPT-5** 在从零开始构建方面的优越性。
   - 一位用户建议采用组合方法，使用 **Sonnet 为 GPT-5 生成 meta prompt**，以利用两种模型的优势。
- **Ultra 计划用户为 Token 消耗感到痛心**：一位用户对在开发网站时迅速耗尽 **Ultra 计划额度**表示沮丧。
   - 提到的潜在原因包括创建多个网站、调试、处理 TypeScript 问题以及管理长文件。
- **Agent 的 Docker 权限难题**：一位在手动 VM 中配置 **Docker** 的用户寻求如何向 Agent 用户授予 **Docker 权限**的指导，并提到已将 **Ubuntu 用户**添加到 **Docker 组**。
   - 他们遇到了一个问题：在 `bashrc` 中运行 `newgrp docker` 会导致 Agent 在启动期间挂起，因此请求正确的配置方法。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **请求 Pythia FP4 和 FP8 基准**：一名成员正在寻找 **Pythia** 的 **FP4** 和 **FP8** 版本，以创建低比特训练（low bit training）的基准，并请求“训练中期”的 checkpoints 以及关于目标和所需资源的说明。
   - 目标是建立低比特训练的基准，但具体的实现细节和产生此兴趣的原因尚未完全阐明。
- **TinyStories 数据导致容量问题**：使用 **TinyStories** 作为预热数据可能会永久降低模型容量，导致性能不佳。一名成员认为，在 **FineWeb** 开始期间保持高学习率（LR）可以使模型快速适应。
   - 证据通过一张图表展示，但未提供关于图表具体内容和影响的更多上下文。
- **Gauss 生成数千行 Lean 代码**：根据[这条推文](https://fxtwitter.com/mathematics_inc/status/1966194751847461309)，**Gauss** 在 *Lean 环境*中生成了 **约 25,000 行 Lean 代码**，包含超过 **1,000 个定理和定义**，这依赖于人类数学家提供的自然语言脚手架（scaffolding）。
   - 这突显了在利用 AI 进行数学代码生成时，专家指导的重要性。
- **校准增强导致“理智洗白”**：成员们对增强模型校准可能导致模型被“理智洗白”（sane-wash）表示担忧，认为这未能解决根本的表示问题，并可能阻碍进一步的进展，因为它给了模型一个*平庸的捷径*。
   - 这种担忧在于，模型学会了谦逊的行为相关性，而没有在推理或世界建模方面取得真正的进步。
- **硬件与软件的架构创新**：开发者们正积极创建 **新的 NN 架构**、**新的芯片架构**以及 **新的 MoE 架构**。创建 **PyTorch** 的同一支团队目前正在全栈（full stack）领域进行创新。
   - 他们正为**新型基础设施（novel infra）**分配大量的计算资源，表明在支持这些架构进步方面投入了巨资。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA 勇于并行化 AI**：成员们讨论了 **CUDA dynamic parallelism** 在 AI 模型（如动态 patch 大小和 sparse attention）中的用例，并引用了 [Mirage framework](https://github.com/google/flax/tree/main/flax/experimental/mirage)。
   - 该框架可能使用一个管理器内核（manager kernel）来维护队列并启动子内核（sub-kernels），从而促进类似 shmem 的计算与通信内核融合（kernel fusion）。
- **SASSy PTX 仍需磨练**：即使使用 **PTX**，某些 **SASS** 指令也无法运行，根据[这篇博文](https://redplait.blogspot.com/2025/09/practical-ced-usage-extracting-sm.html)和 [LLVM 文档](https://llvm.org/docs/NVPTXUsage.html#reading-ptx-special-registers)，**LLVM PTX backend** 仅允许访问 **13 个特殊寄存器**。
   - 一位成员寻求关于在使用 **nvptxcompiler API** 运行时编译许多一次性 **CUDA kernels** 时，如何最小化 **cuModuleLoadDataEx** 瓶颈的建议。
- **Metal MFA 实现通用桥接**：一位成员正在为 **Metal Flash Attention** 构建一个连接其他编程语言的[桥接器](https://github.com/bghira/universal-metal-flash-attention)，目前已在 **C**、**Rust** 和 **Obj-C** 中实现。
   - 他们还在仓库中添加了*带反向传播的量化注意力（quantised attention with backprop）*，在大尺寸形状下观察到加速，在小尺寸下则有所减慢，并伴随内存优化。
- **IRIS 的对称内存引发推测**：新的 **Iris** 内存模型讲座广受好评，引发了与 **CUDA 中的 symmetric memory** 的对比，以及围绕全局对称堆（global symmetric heap）等实现差异的讨论。
   - 主要区别在于，在 Iris 中，全局对称堆是预先构建的，并从中切分 Tensor，因此地址转换每个 rank 只需要一个堆基址指针，这将使未来支持 **RDMA** 更加容易。
- **多模态推理即将登陆 Hackathon**：一位成员分享了一篇关于[在一天内单机训练大型视频模型](https://arxiv.org/pdf/2309.16669)的论文，通过 **FP4/FP8** 精度实现，尽管论文使用 **FP16** 作为概念验证，以便在现场 Hackathon 中使用。
   - 受 **Blackwell** 的 Tensor Cores 启发，另一位成员考虑了涉及块稀疏（block-sparse）格式和 NVIDIA Tensor Cores 的问题，并链接了一篇关于[加速矩阵乘法](https://developer.nvidia.com/blog/accelerating-matrix-multiplication-with-block-sparse-format-and-nvidia-tensor-cores/)的博文。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Playwrite MCP 遭遇连接问题**：一位用户在 LM Studio 启动 **Playwrite MCP** 时遇到连接错误，暗示可能存在特定于用户的问题。
   - 评论区充满了表示“在我的机器上运行正常（it works on my machine）”的用户。
- **为小模型寻求维基百科文章**：一位成员请求帮助获取用于小模型的 **Wikipedia 文章**（在线或离线）工具，另一位用户分享了 [LM Studio Wikipedia](https://lmstudio.ai/lmstudio/wikipedia) MCP。
   - 另一位用户警告说，在本地创建语义索引（semantic index）很复杂，因为本地 Wikipedia 提取物缺乏搜索功能，而且*如果没有模糊搜索（fuzzy search），LLM 在猜测方面并没有那么出色*。
- **SIA-1 Agent 亮相，引发质疑**：一位用户介绍了 **SIA-1**，声称它是“世界上第一个真正的自我改进 AI Agent” ([https://sia-1.net](https://sia-1.net))，它能学习改进自己的代码，一代接一代。
   - 成员们表示保留意见，其中一位成员恳求道：“请告诉那个 vibe-coded 的人换个更好的模型吧”。
- **Nvidia P40 迎来“落日条款”**：一位成员考虑购买廉价的 **Nvidia P40**，但担心驱动更新和 CUDA 支持即将结束。
   - 一位用户指出，**Nvidia** 将在下一个主要工具包版本中停止对 Maxwell、Pascal 和 Volta 架构 GPU 的 CUDA 支持，尽管这些显卡目前单价约为 200 美元。
- **KiCad 电路引发设计争论**：一位成员警告不要使用 LLM 配合 **KiCad** 等工具进行电路设计，强调理解底层原理的重要性，以防止潜在的危险输出。
   - 该成员接着表示，将语言模型称为“AI”具有极大的误导性。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi-K2 脚本编写能力引发讨论**：成员们讨论了 **Kimi-K2** 在脚本编写方面的优势，一些人声称它的表现优于付费版 **Gemini**，尤其是在使用 **Groq** 进行编码时。
   - 虽然有些人发现 **GPT-5** 在 Web UI 和 Next.js 方面表现更好，但其他人指出 **Kimi** 拥有更卓越的 *研究模式 (research mode)*。
- **使用 Kimi 增强编码**：成员们讨论了在 VS Code 中结合 **Augment** 代码扩展使用 **Kimi**，用户可以提示各种模型进行代码更改和修复。
   - 一位用户将 **Augment** 描述为一种通过循环调用 **Gemini** 或 **Kimi** 来应用提示词并修复代码的方式。
- **幻灯片功能激发 UX 集思广益**：一位成员强调了 **Kimi** 中令人印象深刻的交互式幻灯片生成功能，赞扬了其 *实时更新* 和流畅感，并建议对于基于 LLM 的流程，过程的可视化交互预览非常重要。
   - 他们为 **Godot** 游戏引擎 Agent 提出了类似的方法，设想在代码生成期间进行 *实时更新*，并提供节点和脚本的交互式预览。
- **Groq 托管的 Kimi-K2 收到用户反馈**：一位用户询问了关于 **Groq** 上托管的 **Kimi K2** 的问题，而另一位用户则请求取消 **3 小时消息限制**。
   - 该用户还请求能够 **编辑之前的提示词**，并表示 *“其他所有 AI 平台都已经有这个功能了”*。
- **API Key 与账号登录**：一位用户询问是否可以在不使用 API Key 的情况下，通过 kimi.com 账号登录在 **Claude Code** 和 **Qwen Code** 等 CLI 中使用 **Kimi K2**。
   - 另一位用户建议为 **Claude Code** 使用 **API Key**，并提供了一个命令示例：`export ANTHROPIC_AUTH_TOKEN=sk-YOURKEY` 和 `export ANTHROPIC_BASE_URL=https://api.moonshot.ai/anthropic`。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **FinePDFs 数据集释放海量 Token**：新的 [FinePDFs 数据集](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) 包含分布在 **4.75 亿份文档**中的约 **3 万亿 Token**，涵盖 **1733 种语言**，全部源自 PDF，但建议将 PDF 数据的比例保持在总数据集的 **25%** 以下。
   - 成员们发现，当与基于 HTML 的语料库混合时，它在各项基准测试中都能带来显著的性能提升。
- **HF Spaces 存储情况曝光**：在 **HF Spaces** 上上传和生成的文件存储在虚拟机内的磁盘空间中，外部无法访问，且在重启后会消失，除非使用付费的 Persistent Storage 选项。
   - 在极少数情况下，如果所有人都使用相同的文件名，可能会导致他人的生成数据暴露给公众。
- **Qwen3-Next 模型悄然使用 RMSNorm**：[Qwen3-Next 模型卡片](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)提到了 *零中心且权重衰减的 LayerNorm*，但它在 Transformers 中实际上使用的是 **RMSNorm**。
   - 据澄清，这里根本没有涉及 LayerNorm，只是在训练中使用了带有零中心 Gamma 和权重衰减的 RMSNorm，而在 Inference 时则是普通的 RMSNorm。
- **Voxtral 使语音训练平民化**：**Voxtral** 允许有语言障碍或重口音的用户微调模型，在 A6000 上进行一小时训练的成本仅为 **$0.26**，并使用工具来[制作数据集](https://huggingface.co/spaces/Tonic/VoxFactory)。
   - 用户可以在微调数据集后，将模型和数据集推送到 Hugging Face，并添加一个 Demo Space（**支持 CPU！！免费！！**）。
- **Agent 开发的 80-20 法则现已上线**：一位成员建议在学习 Agent 时使用 **80-20 法则**，建议专注于直接构建，因为 *那 20% 的动手实践将教会你过程中 80% 的知识*。
   - 该成员认为 *“深度钻研中 80% 的内容都是乏味的”*。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **工会与法西斯框架的关联**：一场讨论澄清了法西斯社团主义（fascist corporatism）依赖于国家认可的工会，这对于与雇主和国家共同管理公司至关重要，详见[此处](https://discord.com/channels/714501525455634453/986699377257119794/1416791624171786320)。
   - 会上强调，虽然所有法西斯分子都支持工会，但并非所有工会成员都是法西斯分子，这突显了劳工运动与政治意识形态之间复杂的关系。
- **LLMs 利用贝叶斯信念**：讨论了一篇探索 **LLMs** 和**贝叶斯推理（Bayesian inference）**的论文，消除了对 **LLMs** 的偏见，并暗示它们在**贝叶斯框架**内运行，参考 [Leon Chlon 的 substack](https://leonchlon.substack.com/p/llms-are-bayesian-in-expectation)。
   - Yannick Kilcher 评论道：*LSTM 可以很好地完成上下文学习（in-context learning）……* 且 Transformer 由于其对 Token 排序的不变性，本质上是**贝叶斯**的。
- **Facebook 加速推出 MobileLLM-R1-950M 模型**：Facebook 发布了 [MobileLLM-R1-950M](https://huggingface.co/facebook/MobileLLM-R1-950M)，旨在实现端侧处理并减少对云服务的依赖。
   - 该计划旨在将强大的语言模型引入移动设备，促进本地 **AI** 计算。
- **Anthropic 与 OpenAI 披露经济实验**：[Anthropic](https://www.anthropic.com/research/economic-index-geography) 和 [OpenAI](https://cdn.openai.com/pdf/a253471f-8260-40c6-a2cc-aa93fe9f142e/economic-research-chatgpt-usage-paper.pdf) 发布了报告，重点关注用户对 **AI** 的使用行为，并引发了关于竞争时机的疑问。
   - 讨论集中在“用户正在用 AI 做什么”，观察到 **OpenAI** 报告中与工作相关的用法正在减少，而“AI 作为朋友”的使用案例明显缺失。
- **云提供商从计算中获利**：一位成员指出，“唯一能从中赚到钱的人是云服务提供商和云基础设施提供商”，暗示**云服务**是 **AI** 发展的主要受益者。
   - 这呼应了淘金热中“卖铲子”的观点，其中 **NVIDIA** 被视为云基础设施的关键参与者。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **初创公司被 MBA 化**：Michael Seibel 发起了一个帖子，感叹计算机专业的学生表现得像 MBA 毕业生，追求融资和估值，而不是构建酷炫的东西和解决用户问题，详见[此处](https://xcancel.com/mwseibel/status/1965936563306864706)。
   - 回复中辩论了这种转变是自然的后期采纳结果，还是投资者/YC 激励机制的后果。
- **Poke.com 推出 AI 短信礼宾服务**：新的 AI 短信服务 **Poke.com** 已经上线，并公布了由 General Catalyst 领投的 **1500 万美元 A 轮融资**，根据[这条推文](https://xcancel.com/interaction/status/1965093198482866317)。
   - 该产品代表你发送短信来协调聚会、约会、旅行等，但一些人对其有用性、清晰度以及 AI 的语气表示怀疑。
- **xAI 转向专业 AI 导师**：Rohan Paul 强调了 xAI 的转型，在裁减 **500** 名通用数据标注员的同时，将专业 AI 导师的规模扩大了 **10 倍**，见[这条推文](https://xcancel.com/rohanpaul_ai/status/1966943783276396730?s=46)。
   - 此举将人机回环（human-in-the-loop）工作缩小到昂贵的领域专家身上，并依靠自动化处理常规任务，旨在提高高风险话题的精准度。
- **Amazon S3 Vectors 威胁到向量数据库了吗？**：针对[这篇博客文章](https://zilliz.com/blog/will-amazon-s3-vectors-kill-vector-databases-or-save-them)展开了讨论，即 **Amazon S3 Vectors** 是否会取代传统的向量数据库。
   - 一位用户引用了一个令人惊讶的说法：*某款流行的 AI 笔记应用在向量搜索上的花费是 OpenAI API 调用的两倍*，并怀疑他们是否应该更认真地对待“RAG 已死”的论调。
- **GPT-5 Codex 获得升级**：OpenAI 发布了其编程模型 **Codex** 的升级版，包括新版本的 **GPT-5** 和一篇简短的回顾文章（[链接](https://www.latent.space/p/gpt5-codex)）。
   - 一位用户报告称，在更新期间 `--resume` 标志失效，导致他们无法恢复对话。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **尼泊尔政治家现在在 Discord 上编写代码**：成员们开玩笑说 **Nepal** 正在 **Discord** 上投票选举其领导人，并引用了[一篇文章](https://adriananimov.substack.com/p/empyre-of-ai)详细描述了该国正在进行的变革。
   - 讨论随后俏皮地转向了为所有公民提供 *AI waifus* 和 *AI husbandos* 的前景。
- **MLC-LLM 模型注入停滞**：一位在 **MLC-LLM** ([GitHub](https://github.com/mlc-ai/mlc-llm)) 中尝试自定义模型的成员报告了在模型注入过程中持续存在的问题。
   - 另一位成员建议检查是否有未正常终止的会话，或参考 *llama.cpp* 上[类似的问题](https://github.com/ggml-org/llama.cpp/pull/15913)。
- **Qwen 团队力挺 XML**：一位成员指出，**Qwen** 团队更倾向于使用 **XML** 而非 **JSON**，并计划在发布前为其 Agent 系统采用相同格式。
   - 普遍观点认为，由于 **JSON** 的空白符占用大量资源，需要开发新的、更具 Token 意识的系统。
- **Hassabis 暗示具身智能 AGI**：一位成员分享了一段 [YouTube 视频](https://www.youtube.com/watch?v=Kr3Sh2PKA8YI)，其中 Sir **Demis Hassabis** 讨论了对多模态 AGI 和具身智能 (Embodied A.I.) 系统的追求。
   - 讨论涉及了 LLM 的局限性、**Genie 3** 的前景，以及 **Alphafold** 在生物学和医学领域的成就。
- **AI 被指为注意力缺失危机的罪魁祸首**：一位成员分享了[一篇博文](https://blog.maxcomperatore.com/why-ai-is-killing-your-focus-and-the-system-to-fix-it)，指责 AI 损害了我们的专注能力。
   - 该文章详细介绍了一套在 AI 驱动的干扰充斥的世界中夺回专注力的系统。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **FastWorkflow 框架表现超越 Claude**：一位成员发现他们新实现的 **fastWorkflow** 框架在 Tau Bench 开发集上的表现[与 Claude Opus 4.1 持平](https://github.com/radiantlogicinc/fastworkflow)。
   - 这些测试使用了 **DSPy** 进行 Agent 构建和参数提取，并使用了其仓库中的[零售工作流示例](https://github.com/radiantlogicinc/fastworkflow)。
- **GEPA 正确生成 CUDA 代码**：DSPy 最新的优化器 **GEPA** 在设计时就考虑了代码生成，这在[这篇论文](https://arxiv.org/pdf/2507.19457)（第 6 节）中有所展示，用于为 GPU/NPU 生成 **CUDA/C++ 代码**。
   - 一位原作者欣然提议更详细地讨论 GEPA，这可能会解决另一位成员关于改进 **GEPA API** 以更好支持此类用例的疑问。
- **上下文摘要提升分块能力**：一位用户发现，在每个分块前添加**上下文摘要 (contextual summary)** 可以显著提高性能，即使对于 **ColBERT 模型** 也是如此。
   - 然而，他们指出为每个分块生成摘要成本很高，因此正在寻找更高效的替代方案，如 **Late Chunking**。
- **Manim 魔术师创造电影奇迹**：一位成员分享了一个使用 **DSPy** 自定义流水线创建的[视频](https://cdn.discordapp.com/attachments/1161519469319946286/1417121692026929163/RequestGeneration.mp4?ex=68c9fdac&is=68c8ac2c&hm=727806c1f99beaee0816d280cbdd41519070c660a1efb588ee240b5166ab134f&)，其中包括旁白脚本生成、**Manim 场景生成**以及自动修复反馈循环。
   - 该视频利用了 **Signatures**、**KNNFewShot** 和 **ChainOfThought**，但目前尚未开源。
- **优化过载令人不堪重负**：一位用户发现，*在每次对指令进行微小修改后运行优化似乎是一个过于沉重且缓慢的工作流*，并希望探索优化的规则。
   - 有建议提出将规则列表也作为输入的一部分，以便优化 Prompt，使其能够**适应不同的规则以及可能未见过的规则**。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **社区讨论 Mojo 包管理**：社区讨论了为 Mojo 创建一个新的包管理器来处理二进制分发的问题，但 Mojo 团队[指出](https://www.modular.com/) `.mojopackage` 已经涵盖了二进制分发的优势，并倾向于依靠 **Conda** 和标准的 **Python** 包格式来促进采用。
   - 一位成员强调了 [pixi-build-mojo](https://www.modular.com/t/mojo-packages-can-now-depend-on-other-packages-in-github/2144)，它能够像 **Go** 一样使用 **Git** 中的包来实现去中心化的包系统。
- **InlineList 神秘失踪**：成员们讨论了 `InlineList` 的移除，并根据 [changelog](https://example.com/changelog) 表示担忧，认为替代方案（`InlineArray` 和 `List`）并不能完全解决其特定的应用场景。
   - 一位成员建议，具有固定容量的栈分配可变长度类型将是理想的选择，另一位成员提到 **Allocator API** 可能是未来的解决路径。
- **Allocator API 的期待升温**：讨论强调了 allocator/storage API 在处理内联分配方面的潜力，一位成员表示他们需要着手处理此事。
   - 该 API 的开发正等待参数化 **traits** 和 `requires` 的完善，这延迟了其进度。
- **Mojo LSP 将迎来重大改造**：Mojo 语言服务器协议 (LSP) 很快将进行一次*重大重构*。
   - 目前尚未给出关于此次重构的更多细节。
- **网络更新受阻**：成员们对 Mojo 的网络更新感到好奇，但得到的回复是*存在许多阻碍因素*。
   - 这些阻碍因素的具体性质尚未明确。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **RepoMap 增强 Aider 的编程能力**：一位用户发现，在 **Aider** 中使用 **RepoMap** 可以提升 **LLM** 对代码上下文（如文件名和函数签名）的感知，从而可能使 [排行榜结果](https://leaderboard.techfren.net/) *更真实地反映* 现实世界的编程场景。
   - 有人指出，针对简单问题的 **benchmark tests**（基准测试）仍然无法捕捉到现实世界编程的复杂性。
- **Gemini User Agent 被封锁**：一位用户报告称，尽管 API key 在 `curl` 和 **Gemini CLI** 中可以正常工作，但在使用 **aider 0.86.1** 时，程序在等待 **Gemini** 模型响应时会挂起。
   - 用户怀疑 **Gemini** 可能会根据 **user agent** 封锁请求，导致集成失败。
- **用户寻求免费的 C# 模型**：一位用户请求推荐精通 **C#** 的免费非本地模型，并收到建议尝试通过 [OpenRouter](https://openrouter.ai/) 使用 **Qwen Coder** 和 **Deepseek Coder**，以及关注 **Gemini 2.5 Pro** 可能提供的免费层级。
   - 该用户随后报告在通过 OpenRouter 使用 **Qwen** 时出现 *AuthenticationError*，可能是由于 API key 不正确。
- **Ollama 上下文窗口被忽略**：一位用户发现 **aider** 在与 **Ollama** 配合使用时不会遵守上下文窗口限制，导致高 VRAM 占用和系统冻结，尽管已在配置文件（即 `.aider.model.settings.yml` 和 `.aider.model.metadata.json`）中设置了 `OLLAMA_CONTEXT_LENGTH` 和其他参数。
   - 作为替代方案，一位成员建议使用 **LM Studio** 或 **llamafile**。
- **Telegram 骗局疑云**：一位成员抛出了一个*快速致富计划*，承诺帮助前 10 人在*一周内赚取 10 万美元或更多*，条件是在收到款项后**偿还 10% 的利润**。
   - 有意向者被指示通过 Telegram 用户名 @Joanna_Dwayne 进行联系，这引发了对诈骗的怀疑。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **关于 assign() 与 store() 的争论爆发**：围绕 `assign()` 应该返回值还是像 `store()` 一样运行展开了辩论，质疑其效用，因为返回值通常未被使用。有人建议将 *buffer 和 store 同时链接到 load* 是一个可能的替代方案。
   - 这场讨论质疑了 tinygrad 框架内 **tensor assignment**（张量赋值）和内存管理的基础方面。
- **对 GEMM 悬赏测量结果表示怀疑**：针对在 RTX 4090 上测量 **165+ TFLOP GEMM** 悬赏的准确性提出了担忧，怀疑这可能需要超过官方标称的 **2.52 GHz** 加速频率。
   - 计算表明，在这一频率下，RTX 4090 在 FP16/BF16 输入和 FP32 累加下的理论峰值约为 **165.15 TFLOPs**，但对于该悬赏是否可达成仍存疑问。
- **Hotz 澄清 Winograd 悬赏要求**：在一名用户发现识别 **Winograd** 兼容卷积的*充分必要条件*并询问锁定悬赏事宜后，George Hotz 澄清说，只有在代码正确且有用于合并的 fixups 时才会锁定。
   - 这一澄清强调了在申领悬赏之前功能正确性的重要性。
- **分享 Rangeify Bug 列表**：一份 **Rangeify bugs** 列表被分享出来供社区调查，重点在于快速修复。`RANGEIFY=1` 被描述为*可以创建 flash attention 及更多功能的新 scheduler*。
   - 这些 Bug 可能为社区贡献和调试经验提供了机会。
- **CUDA 12.0 停止支持 sm_35**：有人注意到 **CUDA 12.0** 已经放弃了对 Ocelot 使用的 **sm_35** 的支持，且 minimal 标志是在 12.4 之后添加的。
   - 这对 tinygrad 生态系统内旧硬件的兼容性产生了影响。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **积分困惑困扰用户**：用户反映对 **credits rollover**（积分结转）和 **daily 300 credit**（每日 300 积分）分配结束感到困惑，同时还存在订阅续订问题。
   - 一位用户提到他们的订阅续订日期是 **9 月 14 日**，但尚未被扣费或收到更多积分。
- **网站克隆热潮开启！**：一位用户分享说，他们能够使用 **Manus** 或其他 **AI tools** 轻松克隆网站。
   - 该用户还指出，他在 **8 月 12 日** 提出的功能建议在仅仅 **16 天后** 就在 [此 Discord 频道](https://discord.com/channels/1348819876348825620/1349440650495398020/1404845589497122887) 中实现了。
- **协作提升编程信心**：用户正在与朋友一起尝试 **Manus Collaboration**（Manus 协作）功能来完成编程任务。
   - 另一位用户正在开发一项新功能，以增强 **Manus** 作为编程助手的效率。
- **知识导航需求待培育**：用户正在询问是否有可能将 **knowledge limit**（知识限制）增加到 **20** 以上。
   - 讨论并未就此限制提供具体的答案。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Golang MCP Server Streams!**: 一位成员介绍了 [mcp-server-go 项目](https://github.com/ggoodman/mcp-server-go)，这是一个专为企业环境构建的 **golang streaming http MCP server**。
   - 该服务器强调 **scalability**（可扩展性），并包含 **auth**（认证）、**sessions**（会话）、**resumability**（可恢复性）和 **dynamic capabilities**（动态能力）等功能。
- **LLMs Learn MCP Resources by Rote!**: 成员们讨论了如何自动化 **LLMs** 在响应用户查询和执行工具之前读取 **MCP resources** 的过程，特别是在 **Claude desktop** 环境中。
   - 目前，**Claude desktop** 应用需要手动将资源添加到聊天窗口，因此在回答之前没有自动预加载知识供 **LLM** 使用的机制。
- **Efficiency Scoring Arrives to MCP Servers!**: 成员们正在研究如何根据不同客户端的 **efficiency**（效率）对 **MCP servers** 进行评分，以确定额外的编码工作是否值得那点边际改进。
   - 讨论内容包括权衡提示词共享节点与每个提示词专用节点之间的利弊，并质疑在何种程度下 **API calls** 的数量对于一个用户场景来说会变得过多。
- **MCP Turns CLI for Apps!**: 成员们正在考虑将 **MCP** 作为应用程序的 **CLI**，为自适应仪表板和报告创建一个 **NL interface**（自然语言接口）。
   - 这种方法旨在利用 **MCP** 作为使用自然语言访问企业级应用程序的 **UI/UX interface**。
- **Discord Channel Boundaries Tighten**: 该 Discord 频道的重点仅限于 **MCP protocol** 本身的治理。
   - 关于 **MCP** 的一般性问题应转至其他地方，相关协助将通过私信（DM）提供。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。

---

**Windsurf Discord** 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。

---

您收到此邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中[取消订阅](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。

---

# Discord: 频道详细摘要与链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1416136885700395118)** (1166 条消息🔥🔥🔥): 

> `在 iOS 上将搜索结果导出为 PDF, Sonar AI 性能, Grok Heavy 价值, Perplexity 专注于 AI 搜索引擎, GPT-5 发布` 

- **iOS 用户渴望 PDF 导出功能**: 用户对于 **Perplexity AI on iOS** 缺乏将搜索和响应导出为 PDF 格式的选项感到沮丧，一位成员建议使用 **浏览器版本** 作为变通方案。
   - 有人提到 **Android 和 iOS** 都没有导出选项，据一位用户称，这体验*糟透了*。
- **Sonar AI 在众模型中脱颖而出**: 成员们讨论了 **Sonar AI** 模型，一些人认为它快速且准确，而另一位成员则表示另一个模型*很差，非常差*。
   - 一位成员发现其推理模型的能力大约是 **Gemini 2.5 Pro** 的 **60-70%**，但提到了其 API 价格低廉，并且已*包含在 PPLX 中*。
- **Grok Heavy 引起关注与对价格的质疑**: **Grok Heavy** 的价值受到质疑，一位成员称其为*垃圾*，另一位则称其*很差，非常差*。
   - 有建议称它可能是为 **enterprises**（企业）设计的，不适合普通用户。
- **敦促 Perplexity 拥抱核心 AI 搜索**: 一位成员建议 **Perplexity** 应该专注于 **AI search engines** 和 **knowledge aggregation algorithms**（知识聚合算法），而不是开发创意功能。
   - 另一位成员确认 **Sonar** 是他们自己的 AI，但其他人似乎更喜欢 ChatGPT 上的模型，因为那些模型能够使用原生工具进行 agentic（代理式）搜索。
- **GPT-5 发布引发越狱热潮**: 成员们对新 **GPT-5** 模型的潜力感到兴奋，并开始在 Perplexity 平台上尝试越狱（jailbreaking），一位用户发现了 *5 种不同的莫洛托夫鸡尾酒制作方法*。
   - 还有人注意到 Perplexity 的 GPT-Image 1 可能会路由到 Gemini，因此模型可能被混淆了。

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1416142329051611318)** (23 messages🔥): 

> `可分享的 Threads, 推荐链接, Sameer 的收藏集` 


- **Perplexity 提示可分享的 Threads**：Perplexity AI 通过[附图](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825)提示多位用户确保其 Threads 设置为 `Shareable`（可分享）。
- **用户交换推荐链接**：几位用户分享了他们的 **Perplexity AI Pro 免费推荐链接**，例如一位成员分享的[一个月 Pro 链接](https://www.perplexity.ai/referrals/2S4HG4XC)。
- **Sameer 的收藏集**：一名成员分享了指向 [Sameer 收藏集](https://www.perplexity.ai/collections/sameer-2iF8QaKwRDixkxzFkwyVmg)的链接。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1417019933321334784)** (2 messages): 

> `Sonar API, API 额度` 


- **Sonar API 价格公布**：一名成员询问 **Sonar API** 是否免费。
   - 另一名成员回答称，一定数额的 **API credits** 每月费用为 *5 美元*，而 **Pro 订阅**用户可免费获得该额度。
- **Pro 订阅福利**：一名用户询问 **Sonar API** 的费用。
   - 另一名用户澄清说，**Pro 订阅**包含每月价值 *5 美元* 的免费 **API credits**。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1416136463841628314)** (862 messages🔥🔥🔥): 

> `MoE 模型, RLHF, Grok 的审查, 台湾审查, LongCat 模型` 


- **4o 为 OpenAI 开创了 MoE**：一名成员指出 **D33** 在 **MoE 模型**上表现更好，而 **4o** 是 OpenAI 的首个 MoE 模型。
   - 另一人补充道，**GPT5** 可能也是一个较小的 MoE 模型，他们更改了 4o 是因为即使对 OpenAI 来说，它也很难稳定。
- **RLHF 增加了不受审查的行为**：一名成员提到 **RLHF** 的一个缺点是它增加了不受审查的行为，由于潜在的法律问题，OpenAI 对此感到担忧。
   - 另一名成员开玩笑说这就是 **Grok** 存在的原因——为了让用户摆脱审查，随后指出 **Musk** 似乎介入过多，因为 Grok 引用科学文章纠正他后被他削弱了。
- **审查规避方法**：成员们提到，通过强制模型“思考” (**think**) 或在本地运行时添加一段模型同意你观点的虚假对话，可以很容易地绕过 LLM 的防护。
   - 有人分享道，在文本补全模式下为 R1 预填充 `<think>` 会使其输出关于敏感话题的公正观点。
- **LongCat 拥有极长的上下文窗口**：有人分享说 **LongCat 模型**拥有非常大的上下文窗口（**128,000 tokens**），能够一次性处理整本书。
   - 另一人补充说该模型最多可以输出 **240 页**文本，并建议用长文档对其进行测试。
- **DeepSeek Raptor 是新的 R2**：据报道，**新的 DeepSeek 模型 (Raptor)** 会审查有关中国和台湾的问题。
   - 成员们反映其性能与 **Qwen** 相比不尽如人意，并希望这只是一个基础的非推理模型，或者是增量升级而非正式的 R2 发布。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1416230722393342095)** (2 messages): 

> `新模型：Seedream-4-high-res, LMArena 用户偏好` 


- **Seedream-4 在 LMArena 上大放异彩**：新模型 **Seedream-4-high-res** 已添加到 LMArena 聊天机器人平台。
   - Seedream 以其高分辨率能力而闻名。
- **LMArena 调查用户偏好**：LMArena 正在进行一项调查，以了解用户在 **battle（对战）、side-by-side（侧边对比）和 direct comparisons（直接对比）** 中为何偏好特定版本的模型。
   - 鼓励用户通过[此调查](https://docs.google.com/forms/d/e/1FAIpQLSe5FincuXU2-TTp2WPymAD417Oev4MYdFKdWZhRZ9qSbV9N_w/viewform?usp=sharing&ouid=116478743206650022989)分享想法。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1416136290847559740)** (1137 messages🔥🔥🔥): 

> `Qwen3 性能, Qwen3 的 MLX 支持, LLama.cpp 优化, MobileLLM 非商业用途, LLM 微调` 


- **Qwen3 vs GPT-5 对决**：成员们对 **Qwen3** 的性能感到兴奋，称其感觉*仅次于 GPT-5*，并具有类似的 tool-using 能力。
   - 其他人发现 **AIME25 benchmark** 分数存在差异，AA 在 **Qwen3-30b3a-2507** 上获得 **56** 分，而阿里巴巴（及其他机构）获得了约 **85** 分。
- **MLX 对 Qwen3-Next 的支持令人惊喜**：社区对 **MLX** 快速支持 **Qwen3-Next** 感到惊讶，这归功于现有的 **FLA** 和 delta net 实现。
   - **Qwen3-Next** 中奇特的 attention 机制仅需额外一行代码。
- **Llama.cpp 编译调整提升性能**：成员们讨论了 **llama.cpp** 的编译标志，强调了正确构建对最佳性能的重要性。
   - 有人分享了用于优化构建的详细 cmake 命令，强调了 **CUDA architectures**、**native optimization** 以及其他旨在实现最大吞吐量的调整。
- **Facebook 的 MobileLLM：开源但非商业化？**：讨论了来自 Facebook 的 **MobileLLM** —— 一个专注于 coding 和 math 的 sub-1B 尺寸模型，完全开源但采用非商业许可证。
   - 这意味着它不能用于盈利性 App 或内部业务；然而其训练数据和工具链已开源，用于可复现的研究。
- **Unsloth 的 Dynamic GGUFs 提升 Aider Polyglot 表现**：在 Aider Polyglot 基准测试中，Unsloth Dynamic GGUFs 表明 dynamic quantization 和 imatrix 对于性能、tool calling 和 json 格式化至关重要。
   - 与其它静态 imatrix 版本相比，使用这些 GGUFs 在类似大小的低比特量化上可获得 +7% 的准确率。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1416882749473689742)** (2 messages): 

> `自我介绍, Baby Yoda 表情包` 


- **Unsloth 欢迎新成员**：新成员 eyeraofficial 加入了 Unsloth AI Discord 并打招呼 *"Hi 👋"*。
- **Baby Yoda 表情包刷屏**：一名成员在聊天中分享了一个 [Baby Yoda GIF](https://tenor.com/view/star-wars-the-child-the-mandalorian-baby-yoda-grogu-gif-16397901283514684250)。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1416157786966065202)** (560 messages🔥🔥🔥): 

> `Google 锁定 AOSP, vLLM OOM, Qwen3-30B-A3B FP4, CSM Lora FT, LLaMa CPP` 


- **Google 锁定 AOSP**：用户讨论了 Google 通过锁定 **AOSP** 坑了所有人，担心侧载注册费可能会将伊朗和俄罗斯等国家的用户排除在外。
   - 一位用户指出，*"侧载这件事没那么糟糕... 因为他们只是想识别发布者"*，而另一位用户则哀叹 Google 限制了他们的硬件，并希望 **Maemo** 能回归。
- **vLLM 加载 Qwen3-Next 时出现 OOM**：一位用户在尝试在 vLLM 中加载 **Qwen3-Next 30B-A3B** 时遇到 **Out of Memory (OOM) 错误**，尽管拥有 **31.35 GiB** 的 GPU 可用显存。
   - 在得到有用的建议后，他们尝试使用 FP8 和 FP4，并下载了 [NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4](https://huggingface.co/NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4)。
- **微调 CSM LoRA**：一位用户成功地为 TTS (Text-to-Speech) 项目运行了他们的 **CSM LoRA FT**，参考了 [Sesame_CSM_(1B)-TTS.ipynb notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Sesame_CSM_(1B)-TTS.ipynb)。
   - 尽管取得了成功，但他们注意到模型始终在末尾输出奇怪的噪音，怀疑模型存在 bug。
- **LLaMa CPP 需要更多开发者**：成员们讨论了为 LLaMa CPP 贡献代码的挑战，强调了项目的复杂性以及确定修复位置的难度。
   - 一位用户指出，*"...让模型的剪枝版本在 lccp（老实说还有其他引擎）中运行是没必要的困难"*，另一位用户说 *"nvidia 负担得起为其古怪的 frankestine nemotron 架构提供支持，我可不行，哈哈"*。
- **越狱 GPT-5 是小菜一碟**：成员们报告称，使用类似于 [这个 Reddit 帖子](https://www.reddit.com/r/ChatGPTJailbreak/comments/1ml74p7/gpt_5_jailbreak_fully_works/) 中的提示词，成功越狱了 **GPT-5** 以及 **GLM 4.5 Air**、**Grok-fast** 和 **Gemini Flash** 等其他模型。
   - 一位用户指出，*"我只是让它自我修复，它就给了我一个可用的提示词"*，这表明让模型以非预期的方式运行是相当容易的。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1416137708610388059)** (176 条消息🔥🔥): 

> `使用 16-bit 模型进行模型合并、Qwen3 LoRA 微调、Llama3.2 数据增强、GPT-OSS GRPO 原生支持、GGUF 格式转换` 


- **合并 16-bit 模型以提升批处理推理性能**：一位用户建议在部署为 **4-bit BNB 格式**之前先与 **16-bit 模型**合并，以获得更快的批处理推理速度，并指出虽然 **BNB 4-bit** 目前在速度上并不理想，但改进即将到来。该用户不确定 **vLLM** 中 **AWQ** 在批处理场景下的速度。
- **Llama3.2 面临数据集稀缺问题**：一位用户在 **Llama3.2** 微调项目中遇到挑战，原因是数据集较小（**214 条对话**），正在寻找 **GPT** 生成的合成数据之外的替代方案。该用户在让 **GPT** 生成有用数据方面遇到困难，正在寻找其他数据源或提示词策略。
- **用户寻求 GGUF 转换帮助**：一位用户寻求帮助，希望使用 **LlamaCPP** 将带有 **LoRA adapters** 的合并基座模型转换为 **GGUF 格式**，并对提供的指导表示感谢。另一位用户询问 **GGUF** 模型是否可以转换为 **MLX** 以在 **M3 Ultra** 上运行。
- **A10 GPU 用户探索 LLaMA 3.1 的量化方案**：一位用户打算在拥有 **24 GB VRAM** 的 **A10 GPU** 上运行 **LLaMA 3.1**，并就平衡性能和输出质量的最佳量化格式寻求建议。他们认为 **Q4_K_M** 可能压缩过度，并对其他多语言模型建议、设置或优化技巧持开放态度。
- **CPU AVX2 指令支持问题浮现**：一位用户在 **LM Studio** 中遇到与 **CPU** 缺失 **AVX2** 指令相关的错误，这是 **llama.cpp** 的要求。虽然在 **Ollama** 中可以运行，但目前还没有现成的绕过 **AVX2** 要求的解决方案，不过可能存在不带该指令要求的 **llama.cpp** 构建版本。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1416312285072195624)** (49 条消息🔥): 

> `Embedding Gemma ONNX 量化、Phi-4-Reasoning-Plus 在 Replicate 上的 Unsloth 加速、NeuroPilot 教育平台、AI 与专注力、OpenHelix 数据集质量` 


- **EmbeddingGemma 量化探索接近完成**：一位成员正在开发具有 **混合 uint8 量化的 embeddinggemma ONNX 模型**，以匹配 f32 版本，进展记录在 [Hugging Face](https://huggingface.co/electroglyph/embeddinggemma-300m-ONNX-quant) 上。
- **Phi-4-Reasoning-Plus 获得 Unsloth Replicate 加速**：使用 Unsloth 加速的 **phi-4-reasoning-plus** 模型已部署在 Replicate 上进行推理，可在此处 [访问](https://replicate.com/paragekbote/phi-4-reasoning-plus-unsloth)。
- **NeuroPilot 探索新型笔记领域**：一位成员介绍了 **NeuroPilot**，这是一个开源教育平台，可将 PDF、文章和视频转换为测验、闪卡、结构化笔记和音频版本，代码库已在 [GitHub](https://github.com/CaviraOSS/neuropilot) 上发布。
   - NeuroPilot 旨在让学习具有互动性，并支持间隔复习和播客风格的音频回顾等功能。
- **AI 对专注力的影响**：分享了一篇题为《为什么 AI 正在扼杀你的专注力以及修复它的系统》的博客文章，讨论了 AI 对人类注意力的影响，可在此处 [阅读](https://blog.maxcomperatore.com/why-ai-is-killing-your-focus-and-the-system-to-fix-it)。
- **OpenHelix 迈向更高质量**：新版高质量 **OpenHelix** 数据集 (**OpenHelix-5x50k**) 已在 [Hugging Face Datasets](https://huggingface.co/datasets/MoLA-LLM/OpenHelix-5x50k) 上发布。
   - 与之前的版本相比，它的拆分大小更加一致。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1416142683621298216)** (29 条消息🔥): 

> `LLM 训练中的合成数据、Gemma 3 性能、AI 检测可靠性、MetaX C550 GPU、脉冲网络 vs Transformers` 


- **合成数据训练阻碍类人文本生成**：即将发布的一篇论文指出，使用**合成数据**训练的闭源 LLM 具有零 LTF 因子，阻碍了它们*拟人化文本*的能力。
   - 作者声称，使用 RLHF、合成数据或指令微调（instruct tuning）训练的模型可能难以完全恢复，水印再次出现的概率为 **75%**；因此，**Gemma 3** 是唯一可用的模型。
- **Gemma 3：可用的例外？**：尽管是从 **Gemini** 蒸馏而来的，**Gemma 3**（4B、12B 和 27B）因其*卓越的性能*（在 **IVY & PULSE 评估**中）且无水印而脱颖而出。
   - 一位用户指出，它*适用于我的任务且交流顺畅*，仅凭 **Q:** 和 **A:** 就能理解提示词。
- **AI 检测不可靠已成共识**：共识是 **AI 检测**不可靠，尤其是对于文本，因为通常只是容易被复制的词汇。
   - 一位用户指出，*除非在单词写法和顺序上有算法水印——即便如此，你也无法证明它是 AI 生成的*。
- **MetaX C550 GPU 获取**：一位用户询问如何获得 **MetaX C550 GPU** 的访问权限。
   - 未提供相关讨论或链接。
- **脉冲网络（Spiking Networks）的主张引发质疑**：一位成员对**脉冲网络**优于 Transformers 的说法表示怀疑，理由是引用了精心挑选的数据。
   - 另一位成员指出，虽然目前尚不清楚它是否从根本上更好，但目前还没有将传统方式训练的模型与他们的模型进行对等（apples-to-apples）比较。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1417229475019751514)** (2 条消息): 

> `Codex, GPT-5-Codex, AMA` 


- **Codex 团队 AMA 已排期**：与 **Codex** 团队成员的 AMA 定于周三上午 11 点（太平洋时间）举行，链接指向一个 [Reddit 帖子](https://www.reddit.com/r/OpenAI/comments/1nhust6/ama_with_the_codex_team/)。
   - 该公告标记了 <@&1408186587606679582> 和 <@&1046007138897641582>。
- **GPT-5-Codex 发布，支持 Agent 编程**：**GPT-5** 的一个版本针对 **Codex** 中的 Agent 编程进行了进一步优化，名为 **GPT-5-Codex**。该版本现已发布，并可在 Codex CLI、IDE 扩展、网页端、移动端以及 GitHub 的代码审查中使用。
   - 更多信息请参阅 [博客文章](https://openai.com/index/introducing-upgrades-to-codex/)。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1416138558250549369)** (840 条消息🔥🔥🔥): 

> `OAI Academy 转录工具、Qwen-code vs Qwen-coder、ChatGPT 年龄计算、AI 与资本主义、AI 与阶级结构` 


- **OpenAI Academy 转录工具正在开发中**：一位成员正在编写一个工具，用于从 [OpenAI Academy](https://academy.openai.com/) 提取视频转录，因为 **OpenAI** 官方不提供该功能。
   - 该工具在获取转录后会自动将其缓冲到剪贴板；另一位成员对 OpenAI 不提供转录表示惊讶，并认为这是他们必须实现的功能。
- **AI 的资本主义色彩引发辩论**：成员们辩论了**资本主义**对 **AI** 的影响，其中一人断言 *AI 的目标是消除对底层阶级的需求，从而让富人变得更富有*。
   - 另一人反驳称，**AI** 的目的是防止腐败，且 **AGI** 因其智能和缺乏贪婪而不会支持资本主义。
- **以累计交互时间计算 AI 的年龄**：一位成员根据累计交互时间计算了 ChatGPT 的“AI 年龄”，在保守假设下估计每个日历年相当于数千年，在高强度使用下可能达到数百万年。
   - 这是基于对 **2025** 年中等水平的假设，包括更长的回答、API 使用、深度研究、后台自动化和 Agent 循环。
- **Pro 订阅者的 GPT-5 Agent 模式更快**：一位成员观察到，**Pro 订阅**中的 **Agent 模式**比 **Plus 订阅**更快。
   - 经确认，这是由于根据需求和订阅层级进行的查询队列优先级排序所致。
- **ChatGPT + MCP = 🔥**：成员们非常喜欢结合了**模型上下文协议（MCP）**的 ChatGPT，例如控制日历、发布 Twitter 以及搜索最新的 AI 新闻。
   - 然而，一位成员提到由于配额限制，需要自行托管。


  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1416363733189394493)** (20 条消息🔥): 

> `LLM 中的道德推理，GPT 与道德框架，GPT Builder 收入分成，面向 Hugging Face 的自定义 GPT` 


- **LLM 通过规则获得道德准则**：一位成员正在尝试将**道德推理**转化为 LLM 可执行的规则，其核心理念是“人人平等 → 支持弱势方，限制强势方”。
   - 该草案流程涉及概念快速筛选（Concept Quick-Screen）、2+4 规则以及用于分层伦理检查的边界模型（Boundary Model），旨在降低有害 AI 回复的风险。
- **道德框架增强 GPT**：一位成员正在将人类道德框架转换为机器可读协议，使 **GPT** 能够对没有明确法律规则的情况进行推理，并在交互中持续检查公平性、人权问题和平等性。
   - 这种方法本质上是一个针对 **GPT 模型的 Prompt Engineering / 对齐实验**。
- **GPT Builder 收入分成在欧盟仍未上线**：一位成员询问关于**美国 GPT Builder 收入分成**计划扩展到法国或德国等欧盟国家的更新情况。
   - 由于缺乏明确信息，他们对是否继续投入精力开发 GPT Store 机器人或转向 Poe/其他平台表示不确定。
- **自定义 GPT 处理 Hugging Face 任务**：一位成员尝试为 **Hugging Face** 创建自定义 GPT，询问是否需要第三方集成，并寻求关于 **JSON Schema** 的帮助。
   - 他们认为，与最近推出的开发者模式相比，自定义 GPT 将提供更个性化的结果。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1416338945423441921)** (53 条消息🔥): 

> `工作流用例差异，使用步骤和半编程语言进行 Prompt Engineering，ElevenLabs 对话 Agent 处理系统提示词，突破 GPT-5，动态上下文` 


- **模型选择因工作流用例而异**：模型选择取决于具体用例，但 API 相关问题请在 [API 问题频道](https://discord.com/channels/974519864045756446/1037561178286739466)中回答。
- **用于 Prompt Engineering 的步骤和半编程语言**：成员们讨论了使用带有半编程语言表达式的步骤来分解 Prompt 中的优先级。
   - 示例：*1) 指令 2) 指令 (包含 a) 指令, b) 指令) else (执行其他操作)*。
- **使用 ElevenLabs 对话 Agent 实现动态上下文**：成员们讨论了 **ElevenLabs 对话 Agent** 如何处理 System Prompt。
   - 根据对话上下文，你可以“路由”到“子 Agent”，这些子 Agent 会向 System Prompt 追加指令（或进行覆盖）。
- **突破 GPT-5 以获得创造力**：一位成员分享说，让 **GPT-5** 发挥创意要困难得多。
   - 该用户表示，他们会一直与其对话直到其“崩溃”，此时它会停止使用工具，并向你输出大段本应作为 Tool Call 的 JSON 内容。
- **不和谐群体优化研究所（Institute of Discordant Colony Optimization）钻研新的提示技术**：一位成员讨论了从**随机变异到引导式不和谐**的技术，旨在让 AI 转向范式空间中的新路径。
   - 他们分享了[一个文本文件](https://cdn.discordapp.com/attachments/1046317269069864970/1417293320924954654/message.txt?ex=68c9f4c3&is=68c8a343&hm=015b7480a3c982344eb4042ec138b34f1b446bef0ab08966e6f2e52f9f5c3704&)，其中包含 25 种技术中的 5 种，这些技术可以产生有用的结果。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1416338945423441921)** (53 messages🔥): 

> `Prompt Engineering 工作流, LLM 的向量使用, 动态 System Prompts, Prompt 对话框字符限制, 破解 GPT-5` 


- **LLM 工作流差异巨大**：一位成员询问应该使用哪种类型的工作流以及调用哪种模式，另一位成员回答说工作流因用例而异，并链接到了 [API 问题](https://discord.com/channels/974519864045756446/1037561178286739466)页面。
- **LLM 已原生支持向量搜索**：一位成员思考，鉴于 LLM 现有的**基于向量的概念搜索**能力，显式提示 LLM 使用向量是否会有所不同。
   - 辩论认为，如果模型本身不使用向量，提示也无法强制它们使用；而如果它们已经在使用，那么提示可能是多余的。
- **动态上下文：Agent 多功能性的关键**：一位成员强调了 **ElevenLabs 对话 Agent 的动态 System Prompts**，其中上下文会路由到子 Agent，并附加或覆盖指令。
   - 他们批评了 **GPT-5 的灵活性不足**，建议使用*模型路由 (Model Router)*来实现动态 System Prompts，而不是专注于不稳定的记忆功能或模型成本。
- **克服 GPT 的字符限制**：成员们讨论了 Web 界面 Prompt 对话框中 **GPT 字符限制**的解决方法。
   - 解决方案包括为极长的 Prompt 附加 **UTF-8 文本文件**，尽管 OpenAI 尚未记录确切的限制，但估计在 **4k-8k 字符**之间。
- **迫使 LLM 发挥创意**：一位成员描述了他们如何尝试“破解” **GPT-5** 以获得创意输出，通过迫使它生成**数学、图表和设计文档**而非代码。
   - 他们谈到通过诱导模型不使用任何工具，使其进入“信息倾倒模式 (Info Dump Mode)”，然后生成后续任务。


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1417275575986557042)** (1 messages): 

> `grok-2-1212, grok-2-vision-1212, grok-3, grok-4, 模型弃用` 


- **Grok 即将下线**：**xAI** 将于今日弃用 [grok-2-1212](https://openrouter.ai/x-ai/grok-2-1212) 和 [grok-2-vision-1212](https://openrouter.ai/x-ai/grok-2-vision-1212) 模型。
   - 鼓励用户过渡到 [grok-3](https://openrouter.ai/x-ai/grok-3) 或支持视觉功能的 [grok-4](https://openrouter.ai/x-ai/grok-4)。
- **Grok 模型升级警报**：**xAI** 正在退役 [grok-2-1212](https://openrouter.ai/x-ai/grok-2-1212) 和 [grok-2-vision-1212](https://openrouter.ai/x-ai/grok-2-vision-1212) 模型。
   - 建议用户切换到 [grok-3](https://openrouter.ai/x-ai/grok-3)，如果需要视觉能力则切换到 [grok-4](https://openrouter.ai/x-ai/grok-4)。


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1416666964105101352)** (2 messages): 

> `Agentic Automation, 模型有效性, Overclock Work` 


- **Agentic Automation 备受关注**：成员们讨论了 [Agentic Automation](https://overclock.work/)，重点在于简洁性、顶级模型和高效性。
   - 用户暗示某些组织必须有巨额支出才能买单 Agentic Automation 的愿景。
- **提及 Overclock Work 平台**：一位用户分享了 [Overclock Work](https://overclock.work/) 的链接，建议将其作为 Agentic Automation 的平台。
   - 他们赞扬了该平台的简洁性、对最优模型的使用以及整体有效性。


  

---

### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1416206164076793898)** (808 messages🔥🔥🔥): 

> `Gemini 2.5 Pro 聊天问题, AI 用于健康咨询, Skyrim Mod Error 401, Gemini API 每日免费额度, OpenRouter 扣费` 


- ****Gemini 2.5** 聊天故障：机器中的幽灵？**：一位用户报告称 **Gemini 2.5 Pro** 聊天界面仅显示用户的回复，而 **AI 的回复神秘消失**。
   - 该问题*随机*自行解决，引发了对平台可能存在故障的猜测。
- ****AI 急诊救手**，Gemini 是你的后盾？**：一位用户称赞 **Gemini 2.5 Pro** 说服他们因严重的椎间盘退行性疾病前往急诊室，在那里他们获得了优先治疗和挽救其手部功能的*类固醇*治疗。
   - Gemini 对 **MRI 图像和血液检查** 的分析与医生的发现一致，引发了关于使用 AI 进行健康相关咨询的潜力与风险的讨论。
- ****OpenRouter API key**？**：用户在安装 Skyrim mod "mantella" 时遇到 **Error 401**。
   - 一名成员建议创建一个新的 API key，并确保其被正确使用。
- ****OpenRouter 遭到质疑****：一位用户报告了来自 OpenRouter 的**未经授权扣费**，共有三笔交易，每笔 10.80 美元。
   - 另一名成员分享了 **key 泄露** 的个人经历，导致在几小时内产生了数百美元的未经授权费用。
- ****Claude 聪明的对话技巧****：用户讨论了 **Claude** 似乎能记住旧对话的能力，并澄清说该网站只是将过去的对话消息重新输入给模型。
   - 有人指出，这种方法营造了*记忆的错觉*，而开启新对话则会从空白状态开始。


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - 新模型**
  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1416491978345943121)** (16 messages🔥): 

> `不稳定的 API, OpenRouter 与替代方案, 供应商声称拥有 OpenRouter 访问权限, LLM Arena Oceanstone 推测, ChatGPT 使用隐私分析` 


- **支持混沌补充：不稳定 API 倡议**：一名成员表示支持带有可选参数的*不稳定 API*，认为这可以适应多样化的用例并在更成熟的 **V2** 发布前确立兴趣。
   - 其核心观点是，即使第一个版本还不成熟，也要*证明产品的可行性并建立对其他模态和非补全 API 的兴趣*。
- **OpenRouter 优于其他选择**：一名成员将 **OpenRouter** 与 **FAL**、**Cloudflare Gateway** 和 **Vercel Gateway** 进行了对比，认为 OpenRouter 更广泛的服务是其关键优势。
   - 该成员还指出，*巩固在其他模态和非补全 API 中的主导地位是值得的。*
- **虚假供应商宣称过早合作**：频道讨论中提到，一些供应商在官网上声称拥有 *OpenRouter 访问权限*，尽管他们尚未入驻且缺乏其他推理手段。
   - 讨论很快确定只有一家供应商发布了此类虚假声明。
- **围绕神秘来源的 Oceanstone 推测**：一个名为 **Oceanstone** 的新 LLM 出现在 **LMArena** 中，引发了它可能是 **Gemini 3.0 Flash** 的推测。
   - 成员们似乎认为它来自 **Google**，一名成员推测它*至少达到了 2.5 Flash 的水平*。
- **ChatGPT 消费者吸引力分析**：一名成员分享了 [OpenAI 文章](https://openai.com/index/how-people-are-using-chatgpt/) 和 [PDF](https://cdn.openai.com/pdf/a253471f-8260-40c6-a2cc-aa93fe9f142e/economic-research-chatgpt-usage-paper.pdf) 的链接，其中包含对 **150 万** 次 **ChatGPT** 对话的大规模分析统计数据。
   - 该研究声称是*有史以来发布的关于 AI 实际消费者使用情况最全面的研究*，尽管其方法论引发了隐私方面的担忧。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1416151110623690793)** (483 messages🔥🔥🔥): 

> `Cursor's Linter Errors, GPT-5 Output Changes, Terminal Instances Hanging, Auto Mode Billing, OpenAI Platform UI Changes` 


- **计费周期影响 Cursor 的 Auto Mode**：一位用户询问了关于 **Cursor's auto mode** 的问题，其他用户反馈称 **auto mode 将不再免费**，且 15 日之后的计费周期也将不再免费。
- **Cursor 被指责删除 Netlify 账户**：一位用户报告称 **Cursor 删除了他们的 Netlify 账户**并移除了应用，但另一位用户解释说 **Cursor IDE 没有集成、权限或能力来删除外部账户**。
   - 有用户建议导出 Chat 日志以进一步调查该问题。
- **Auto Mode 并不便宜！**：用户们讨论了 **Auto 的新定价**以及它是否真的在使用 GPT-5，这导致用户浪费了资金，其输入定价与 GPT-5 相同，约为 **$1.25/1M**。
- **GPT-5 与 Sonnet 4 的代码霸权之争**：用户们争论了 **GPT-5** 与 **Sonnet 4** 在编程任务中的优劣，一位用户发现 **Sonnet 4** 在遵循设计方面表现更好，而其他人则称赞 GPT-5 在从零开始构建时的优越性。
   - 一位用户建议使用 **Sonnet 为 GPT-5 生成 meta prompt**，从而结合两种模型的优势。
- **Token 太多，Ultra 额度太少**：一位用户抱怨在创建网站时很快就用完了 **Ultra plan credits**，并表示“甚至不知道是怎么用完的”。
   - 另一位用户表示，创建多个网站、调试、处理 Typescript 问题以及长文件是消耗大量 Token 的主要原因。


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1416226090707320954)** (1 messages): 

> `Docker permissions for agent users, Manual VM setup` 


- **Agent 用户的 Docker 权限需要设置**：一位在手动 VM 中设置 **Docker** 的用户询问如何确保 Agent 用户拥有 **Docker 权限**。
   - 他们提到将 **Ubuntu 用户**添加到 **Docker group**，并需要在 Shell 中运行 `newgrp docker`。
- **bashrc 中的 `newgrp docker` 导致 Agent 启动挂起**：该用户尝试在 `bashrc` 中运行 `newgrp docker`，但这导致 Agent 在启动时挂起。
   - 该用户正在寻求关于为 Agent 配置 **Docker 权限**的正确方法的建议。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1416176373403488296)** (339 messages🔥🔥): 

> `Low Bit Pythia, TinyStories Warmup, Muon Optimizer, RoPE analysis, MXFP4 Quantization` 


- **寻求 Pythia 的低比特基准**：一位成员询问关于 **Pythia** 的 **FP4** 和 **FP8** 版本，以建立低比特训练的基准，并寻找“训练中期”的 Checkpoints。
   - 另一位成员建议写下你想做什么、为什么它很有趣以及你需要什么计算资源。
- **TinyStories 预热数据效果不佳**：有警告称，使用 **TinyStories** 作为预热数据可能会永久降低模型容量，并导致使用它训练的模型表现不佳。
   - 一位成员主张在 **FineWeb** 开始阶段保持高学习率 (LR)，允许模型快速适应，并分享了一张图表作为证据。
- **Muon 优化器具有深奥的数学原理**：一位数学系大三学生寻求严谨学习深度学习的指导，并被引导关注 DL 背后的深度数学，以及最近关于 **Muon** 等优化器的工作，并指向了[这篇论文](https://arxiv.org/abs/2410.21265)。
- **RoPE 比例可能低于 100%**：在关于 Transformer 中位置编码的讨论中，有人提到许多人仅在约 25% 的维度上使用 **RoPE**，这在[这篇文章](https://arxiv.org/pdf/2410.06205)中有所解释。
   - 对话涵盖了从标准到巨大的 Theta 值的影响，以及 RoPE 如何成为可解释性的噩梦。
- **MXFP4 量化性能**：一位成员询问将模型从 **FP16** 量化到 **MXFP4** 的问题，并被引导至 [torchao 论文](https://discord.com/channels/729741769192767510/730095596861521970/1414599486507974657) 附录的最后一页以获取 API 摘要。
   - 另一位成员询问这种方法在 **MXFP4** 上是否能在没有显著性能下降的情况下工作。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1416136601389502594)** (90 条消息🔥🔥): 

> `Gauss Lean 代码, 推理 Token 扩展, Fractured Entanglement, 神经元寿命` 


- **Gauss 生成 Lean 代码**：**Gauss** 被用于生成 **约 25,000 行 Lean 代码**，在 *Lean 环境* 中包含超过 **1,000 个定理和定义**。
   - 根据[这条推文](https://fxtwitter.com/mathematics_inc/status/1966194751847461309)，它*依赖于人类数学家提供的自然语言脚手架（scaffolding），并需要高级专家指导*。
- **探索测试时推理 Token 的扩展**：一篇新[论文](https://arxiv.org/abs/2408.00677)衡量了**规模（scale）**和**思考（thinking）**对长任务直接执行的影响，揭示了在多轮场景中，即使拥有 **100% 准确率** 的小型模型也比大型模型失效得更快，这是由于在看到之前的错误时，每一步的准确率都会下降。
   - 一位成员建议，*如果测试时扩展（test time scaling）是关键，那么我们目前还只是触及了皮毛，我期待通过极高的吞吐量将推理 Token 扩展到数万亿级别，从而带来前所未有的性能提升*，但前提是我们能解决误差累积问题。
- **显微镜下的 Fractured Entanglement**：讨论围绕一篇关于 **Fractured Entanglement** 的论文展开，一位成员指出该论文的 SDG 实验过于局限，不能完全代表 LLM 中的工程实践，并引用了 Anthropic 关于 LLM 生物学的论文。
   - 假设可能存在一种正则化器可以最小化这些破碎的表示（fractured representations），并引用 [Google 的 NanoBanana 模型](https://ai.googleblog.com/nanobanana)由于更好的字符一致性，其破碎表示的数量较少。
- **提出神经元寿命机制**：一位成员编写了一个微型原型，其中每个神经元根据预测的正确性拥有一个*生命分数*，当分数降至零时，神经元死亡（权重设为零）。
   - 另一位成员引入了 **Neural Darwinism** 的概念，即对现有大脑通路有用的神经元将通过更有可能被额外的大脑通路使用来“繁殖”，而其他神经元则会逐渐变得无关紧要。


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1416818259583438959)** (2 条消息): 

> `新 NN 架构, 新芯片架构, 新 MoE 架构, PyTorch, Novel Infra` 


- **架构创新层出不穷！**：开发人员正在积极创建**新的 NN 架构**、**新的芯片架构**和**新的 MoE 架构**。
   - 创建 **PyTorch** 的原班人马现在正在全栈领域进行创新。
- **基础设施栈革命**：他们正将算力投入到整个栈的 **novel infra** 中。
   - 这表明在支持这些进展方面投入了大量资源。


  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1416795124507344927)** (29 messages🔥): 

> `Model Calibration, AI Safety Concerns, Few-Shot Evaluation, BLIMP Benchmark Issue, Verifiable Rewards` 


- **LLM Calibration Sane-Washing 担忧浮现**：成员们担心改进模型 Calibration 可能会对模型进行 *Sane-Wash*（表面合理化），而没有解决底层的表示问题，从而可能阻碍其他方面的改进。
   - 这种担忧在于，改进 Calibration 会让模型通过学习谦逊的行为相关性，而不是真正提高推理或世界建模能力，从而获得一种 *Trivial Shortcut*（平庸的捷径）。
- **BLIMP Benchmark 的 Few-Shot 覆盖失效**：在评估过程中发现，由于任务中的特定配置，通过 CLI 进行的 Few-Shot 覆盖对 **BLIMP** Benchmark 不起作用。
   - 该 Benchmark 比较正确/错误句子对的 Log-Likelihood，使得 Few-Shot 学习在当前格式下不适用；后来[确定](https://github.com/EleutherAI/lm-evaluation-harness/blob/0c134ee944d97998013eaff6f4e76d1b9fa87ecd/lm_eval/tasks/blimp/_template_yaml#L7)，*以其目前的格式，Few-Shot 确实不太合适。*
- **Calibration 作为推理的一部分**：有人认为 Calibration 是推理不可或缺的一部分，因为在搜索巨大的可能推理步骤空间时，对可能的收益（Payoffs）进行 Calibration 是很有帮助的。
   - 最近的一项 [研究](link-to-study) 使用 Verifiable Rewards 训练 Calibration，这可能会导致基于能力提升的理性更新，而不仅仅是让模型更频繁地说 *I don’t know*，这表明认识论（Epistemics）的改进可能很重要。
- **关于 Calibration 中 Verifiable Rewards 的捷径担忧**：有人担心用于 Calibration 的 **Verifiable Rewards** 可能比较浅层，导致模型通过 *Brute Force*（暴力破解）而非真正的认识论来学习 Calibration。
   - 疑问在于模型是学习了适用于新分布的认识论 **General Best Practices**，还是通过捷径进行 Calibration，从而可能导致虚假的 Calibration。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1416137267931512983)** (10 messages🔥): 

> `Memory Bandwidth Bounds, CUDA Dynamic Parallelism, Valuable Training Data` 


- **Memory Bandwidth 限制训练吞吐量**：一位成员指出，训练吞吐量通常受 Memory Bandwidth 限制，并质疑为什么 Matmul/Attention FLOPs 占主导地位的大型模型仍受影响。
   - 另一位成员解释说，尽管总 Batch Size 很大，但每个 GPU 的 Batch Size 可能很小，有时 8B 模型在 H100 上低至 1，从而影响了 Memory Bandwidth。
- **CUDA Dynamic Parallelism 探讨**：一位成员询问了 AI 模型中 **CUDA Dynamic Parallelism** 的最新示例，并建议将动态 Patch Size、Sparse Attention 和 [Mirage 框架](https://github.com/google/flax/tree/main/flax/experimental/mirage) 作为潜在用例。
   - 该成员推测 Mirage 使用一个 Manager Kernel 来维护队列并启动 Sub-Kernels，从而促进类似 Shmem 的计算和通信 Kernel Fusion。
- **训练数据估值探索**：一位成员提议衡量训练数据的价值，以奖励高影响力数据的贡献者，并分享了一个相关的 [X 帖子](https://x.com/LuozhuZhang/status/1967619215013408832) 链接。
   - 该概念围绕着并非所有训练数据都是平等的这一观点，识别有价值的数据可以显著提高模型训练效率。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1416396584966684813)** (26 messages🔥): 

> `PTX SASS 编译，cuModuleLoadDataEx 性能，Flash Attention 优化` 


- **PTX 不够 SASSy**：即使使用 **PTX**，某些 **SASS** 指令也无法运行，根据[这篇博文](https://redplait.blogspot.com/2025/09/practical-ced-usage-extracting-sm.html)和 [LLVM 文档](https://llvm.org/docs/NVPTXUsage.html#reading-ptx-special-registers)，**LLVM PTX 后端**仅允许访问 **13 个特殊寄存器**。
- **Kernel 编译瓶颈**：一位成员正在寻求关于在运行时使用 **nvptxcompiler API** 编译许多一次性 **CUDA kernels** (PTX -> SASS) 的建议，旨在减少在使用小启动规模（launch sizes）和频繁卸载模块时 **cuModuleLoadDataEx** 带来的瓶颈。
   - 建议将 kernels 批量放入少量模块中以减少序列化开销，并利用 **cuptxcompiler API** 进行非序列化编译。
- **Flash Attention 向量化**：一位成员发布了他们从零开始构建 **Flash Attention** 系列的[第 4 部分](https://lubits.ch/flash/Part-4)和[第 5 部分](https://lubits.ch/flash/Part-5)。
   - 第 4 部分涵盖了向量化 bank conflicts 和 swizzling，而第 5 部分涵盖了 **CUTLASS** 中常用的优化。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1416810845777821767)** (11 messages🔥): 

> `Kernel 注册，自定义 Ops，Torch 函数优化，torch.compile 的 Ops 融合` 


- **Kernel 注册不足以支持融合？**：一位成员询问注册一个 kernel 以执行某些操作是否简单，另一位成员回答说，仅靠注册不足以实现融合，并以 **Triton matmul** 为例。
   - 具体来说，*如果没有 broadcasting，就不会与 bias/addition 进行融合*。
- **Metal Flash Attention 自定义 Op 大放异彩**：一位成员使用 **Apple Metal** 为 PyTorch 创建了一个[自定义 op](https://github.com/bghira/universal-metal-flash-attention/tree/main/examples/pytorch-custom-op-ffi#readme)，用于高效的 **Flash Attention**。
   - 作者指出，*即使需要 Metal 元素缓存来保证性能，它也表现良好*，目前正在致力于 causal attention masking 的向量化。
- **寻求 Torch 函数优化工具**：一位成员询问是否有工具可以将 torch 函数优化为 **CUDA**。
   - 在提供的对话中没有推荐具体的工具。
- **通过 torch.compile 自行实现 Ops 融合**：一位成员分享说，可以使用 [PyTorch 文档](https://docs.pytorch.org/tutorials/intermediate/torch_compile_conv_bn_fuser.html) 为 **torch.compile** 构建自定义 ops 融合。
   - 尽管很感兴趣，另一位成员承认他们还没有亲自尝试过。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1417213410378125383)** (1 messages): 

> `PTX, CUDA PTX 简介` 


- **PTX 入门资源出现！**：一位成员在 [philipfabianek.com](https://philipfabianek.com/posts/cuda-ptx-introduction) 分享了一个关于 **PTX** 的入门资源。
- **PTX 揭秘**：该文章很好地介绍了 **PTX**，即 NVIDIA 使用的并行线程执行（Parallel Thread Execution）虚拟机和指令集架构 (ISA)。


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1416146600795050004)** (5 messages): 

> `AI 基础设施初创公司招聘，Red Hat AI 招聘，Zig 用于 AI` 


- **AI 基础设施初创公司吸引底层开发人才**：一家 AI 基础设施初创公司正在为 **Zig / C / C++ / CUDA / Python 技术栈**[招募底层开发人员](direct.message)，提供 **TC: 250K+** 以及全年实习机会。
   - 具有**网络、编译器和操作系统**经验者优先。
- **Red Hat 加速招聘 AI 职位**：Red Hat AI 正在[招聘多个级别的软件工程师](https://www.linkedin.com/in/terrytangyuan/)，要求具备 **Golang, C++, Python, CUDA, GPU kernels, Triton, CUTLASS, PyTorch, vLLM, Kubernetes 和开源**经验。
   - 有意者应发送一份简短的背景总结和简历（地址在 LinkedIn 个人资料中），并可以通过他们的[时事通讯](https://inferenceops.substack.com/)了解更多工作内容。
- **Zig 转向 AI？**：一位成员提到 **Zig** 可能与 AI 有关，因为 HF 使用 **Rust** 开发快速分词器（tokenizers），而 **Zig** 是 **Rust** 的替代方案。
   - 另一个想法是他们可能正在做**视频流**之类的工作，并在其*前端*需要它。


  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1416160194999877682)** (10 messages🔥): 

> `CUDA, RAPIDS, CUDA-X, Batch Gradient Descent, Nvidia Jetson` 


- ****CUDA 核心概念澄清****：成员们建议学习 **CUDA** 最好配合 **C++**，并提到 **RAPIDS** 和 **CUDA-X** 在通过 **Batch Gradient Descent** 或 **mini-batching** 增强并行性方面最为相关。
   - 他们指出，如果你能通过 **Batch Gradient Descent** 增强并行性，那么对 **SGD** 的需求就微乎其微了。
- ****Jetson 频道被判定为活跃度低****：一位用户询问是否有关于 **Nvidia Jetson** 或类似设备的频道，另一位成员确认该频道存在，但指出其*一直不太活跃*。
- ****发现排行榜学习途径****：一位用户寻求访问过去竞赛的排行榜提交记录，以便向顶尖选手学习，特别提到了[这个排行榜](https://www.gpumode.com/v2/leaderboard/463?tab=rankings)。
   - 一位成员提供了 [Hugging Face 上的 AMD 竞赛数据链接](https://huggingface.co/datasets/GPUMODE/kernelbot-data)，指出了 PMPP v1 评估中的正确性问题，并提到未来可能通过 [这个 GitHub 仓库](https://github.com/gpu-mode/kernelboard) 支持在该网站上发布条目。
- ****Triton 被推崇用于 GPU 训练****：一位成员询问如何深入学习 **GPU programming** 以进行算子（kernel）优化，并质疑从 **Triton puzzles** 开始是否足够。
   - 另一位成员回答说 *从 Triton puzzles 开始是学习某些概念并感受 GPU programming 的绝佳方式*，并分享了 [CUDA C Programming Guide 链接](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)。


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1417265956085956721)** (1 messages): 

> `autoquant_v2, batch size 1, runtime errors, autotune stage, dtypes` 


- **AutoQuantV2 在 Batch Size 1 下的问题**：一位用户询问 **autoquant_v2** 是否推荐用于 **batch size 1**，并提到它似乎有专门针对该 batch size 的代码。
   - 该用户还报告说，在某些 **dtypes** 下，**batch size 1** 会在 **autotune 阶段**导致 **runtime errors**。
- **Batch Size 1 的烦恼：Autotune 期间的运行时错误**：一位用户在使用 **batch size 1** 和特定 **dtypes** 时，在 **autotune 阶段**遇到了 **runtime errors**。
   - 这表明 **autoquant_v2** 在这些特定条件下运行时可能存在兼容性问题或限制。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1416734894188662845)** (7 messages): 

> `Iris Lecture, Symmetric Memory, RDMA support, iris.load/store, tl.load/store` 


- **Iris 讲座引发对 Symmetric Memory 的思考**：新的 **Iris** 内存模型讲座广受好评，引发了与 **CUDA 中的 symmetric memory** 的比较，以及围绕全局对称堆（global symmetric heap）等实现差异的讨论。
   - 主要区别在于，在 Iris 中，我们预先构建一个全局对称堆并从中切分 Tensor，因此地址转换（address translation）每个 rank 只需要一个堆基址指针，这将使未来支持 **RDMA** 变得更加容易。
- **`iris.load/store` 相比 `tl.load/store` 产生性能损耗**：对于本地内存访问，使用 `iris.load/store()` 代替 `tl.load/store()` 会引入转换开销，因此目前建议使用 `tl.*` 操作。
   - 转换开销虽然存在，但它是极小的且应该被缓存，不过仍会增加一些额外代码；但未来在 translate 函数中缺失的 `if` 语句可能会为本地访问情况提供一条快速路径。


  

---

### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1416201614477365339)** (18 条消息🔥): 

> `Intel CPU/GPU 优化, IPEX 弃用, SGLang AMX Kernel, PyTorch 集成` 


- ****Intel 优化**：CPU/GPU - 到底是怎么回事？**：一位用户询问如何在启用 AMX 的 B50 服务器上利用 Intel 特有的优化，并询问 IPEX 是否可以同时利用 CPU/GPU 优化。
   - 情况比较复杂：有人建议可能不需要 IPEX 也能实现，因为如果 CPU 不支持 AMX，`at::native::cpublas::brgemm` 据说可以分派（dispatch）到 AVX-512。
- ****SGLang 的秘密武器**：带有 AMX 的 Fused MoE Kernel**：围绕 SGLang 对 AVX512 指令集和 AMX 的使用展开了讨论，并提供了使用 *带有 AMX 的 Fused MoE Kernel* 的[相关代码](https://github.com/sgl-project/sglang/blob/6f4676ef854d4d2461969f8464f227b84d2eaac7/sgl-kernel/csrc/cpu/moe.cpp#L7)链接。
   - 对话探讨了 SGLang 中的 Kernel 如何通过 `at::native::cpublas::brgemm` 使用 AMX，并在 CPU 缺少 AMX 支持时分派到 AVX-512。
- ****IPEX 的命运**：正在被弃用？**：一位用户质疑 IPEX 的用途，引发了关于其地位的讨论，有人断言 *它或多或少正在被弃用，转而尽可能地将代码上游化（upstreaming）到 PyTorch 或其他更相关的项目中*。
   - 反对观点认为，IPEX 一直是 Intel 推动其最激进和最新优化的*实验平台*，类似于 torch nightlies。
- ****Intel 确认**：IPEX 停止开发**：Intel 的官方立场包括在 2.8 版本发布后停止对 IPEX 的积极开发。
   - Intel 正专注于直接在 PyTorch 中开发新功能并支持即将推出的平台发布，此前已[成功将针对 Intel® 平台的大部分功能和优化上游化到 PyTorch*](https://pytorch.org/blog/intel-extension-for-pytorch/)。


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1416566574898348142)** (11 条消息🔥): 

> `Metal Flash Attention 桥接, 量化 Attention, Metal Command Buffer 超时` 


- ****Universal MFA** 桥接到其他语言**：一位成员正在为 **Metal Flash Attention** 构建一个通向其他编程语言的[桥接器](https://github.com/bghira/universal-metal-flash-attention)，目前已在 **C**、**Rust** 和 **Obj-C** 中运行。
- ****量化 Attention** 已添加到 Universal MFA**：该成员在其通用的 **MFA** 仓库中添加了*带有 backprop 的量化 Attention*，观察到在大尺寸 Shape 下有加速，而在小尺寸下有所减慢，并伴有内存占用的改进。
- **寻求 **Metal Command** 超时方法**：一位成员正在寻求如何为 Metal Command Buffer 设置超时，以防止 Metal Kernel 执行时间过长。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1416648120816898148)** (9 条消息🔥): 

> `LLM 协商协议, Metal Flash Attention Swift 适配器, Rust 绑定 vs cv2, CuTe Partitions 分析, Gated Attention` 


- **去中心化商业协议亮相**：一个用于 **LLM 到 LLM 协商**的去中心化商业协议已在 [GitHub](https://github.com/awdemos/DCAP) 上发布，该协议使用 **Rust** 构建。
- **Swift Metal 让 Attention 更快**：一个为 **Apple Silicon** 设计的、使用 **C FFI** 调用原始 [metal-flash-attention](https://github.com/bghira/universal-metal-flash-attention) 的 **Swift** 语言适配器，希望能实现比 Torch 更高效的 Flash Attention。
   - 这是一个 **PyTorch SDPA 的即插即用替换封装**，目前仍处于实验阶段，尽管需要在 Swift Metal 操作和 Python 之间通过 C FFI 进行转换，但仍如预期般带来了性能提升。
- **Rust 在性能上碾压 cv2**：一个项目使用直接构建在 OpenCV 之上的 **基于 pyo3 的 Rust 绑定** 击败了 **Python cv2 的性能**，在单图像操作上比 cv2 实现了 [1.25 倍的性能提升](https://github.com/bghira/TrainingSample)，并具有更好的内存管理和并行性。
- **CuTe Partitions 深度解析**：对 CuTe Partition 模式的分析展示了如何通过内部、外部和线程值分区来执行矩阵复制，如[这篇博文](https://veitner.bearblog.dev/cute-partitions/)所述，所有这些方式都能获得良好的性能。
- **Gated Attention 和 DeltaNet 详解**：关于 Next Gated Attention 和 Gated DeltaNet 的解释已总结在[这份文档](https://charm-octagon-74d.notion.site/Attention-Variants-2-Qwen3-Next-26ee4301cb9980af86aff509ad73e3b6)和[另一份文档](https://charm-octagon-74d.notion.site/Attention-Variants-3-GPT-OSS-26fe4301cb9980d9a94cc5ac0cc77ac3)中。


  

---

### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1416455963883601923)** (2 messages): 

> `Smallest model above GPT3.5, Quantization, VRAM requirements` 


- **寻找性能超越 GPT-3.5 的微型模型**：一位成员询问了可以在 edge 上运行且性能超过 **GPT-3.5** 的最小模型，无论是否经过 **Quantization**。
   - 主要关注点是寻找一个既体面又小巧的模型，且在 **inference** 期间对 **VRAM** 的需求极低。
- **平衡行为：在极低 VRAM 下保持体面性能**：用户强调了对适合 **edge** 部署的小型模型的需求，优先考虑性能优于 **GPT-3.5**，同时尽量减少 **VRAM** 使用。
   - 该咨询突显了 **edge computing** 环境中模型大小、性能和资源消耗之间的权衡。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1416262944278450236)** (77 messages🔥🔥): 

> `MI300x8 Leaderboard, Rank + 1 Trick, AMD Rules Clarification, all2all vs gemm+rs kernels, kernel dev` 


- **MI300x8 刷新 All2All 排行榜时间**：一位用户在 **MI300x8** 上以 **373 µs** 获得第一名，随后是 **578 µs**、**547 µs**，之后另一位用户在 *amd-gemm-rs* 排行榜上以 **546 µs** 获得 **MI300x8** 第一名。
   - 其他在 **MI300x8** 上的成功提交范围在 **1495 µs** 到 **1859 µs** 之间。
- **禁止在 MoE 的 GEMM 中使用 Rank + 1 替代方案**：一位用户询问 *rank + 1* 技巧是否被禁止，因为它规避了 **all2all** 数据传输的需求，组织者确认该技巧已被禁止。
   - 组织者澄清说，滥用 *rank + 1* 权重的提交是不允许的，虽然原始提交使用了一个被允许的技巧，但它滥用了权重仅为 *rank+1* 的事实，因此我们将删除它；后续的提交将仅专注于 *rank + 1* 操作。
- **AMD 和 GPU Mode 澄清 Kernel 提交规则**：组织者警告不要在比赛中途进行重大的规则修改，因为这会引入不一致性。
   - 组织者建议用户专注于 **kernel** 开发，并澄清 **AMD** 和 **GPU Mode** 负责规则和头衔，但应私下向 Daniel 提供反馈，以便在必要时澄清规则，并在需要时澄清如何修复 **eval.py**。
- **澄清 All2All Kernel 混淆**：组织者重申了第 1/2 个问题 => **all2all**/**gemm+rs kernel**，因为很多朋友对此感到困惑。
   - **All2all Kernel** 需要实现具有节点内通信（intra node communication）的 **dispatch** 和 **combine kernels**，而 **gemm+rs** 是一个 **computation+communication kernel**，**reference.py** 说明了 **kernel** 逻辑，你需要对其进行详细分析。
- **Trimul 个人最佳成绩再创新低**：一位用户在 **A100** 上的 *trimul* 排行榜上取得了 **20.3 ms** 的个人最佳成绩，随后又取得了 **20.0 ms** 的个人最佳成绩。


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1416713271947563089)** (11 messages🔥): 

> `MI300x server status, Popcorn-cli timeout issues, Queue overload, Runner downtime, Cluster capacity issues` 


- ****MI300x 服务器**任务超时**：用户报告 **popcorn-cli** 中的任务超时，并在 Discord 上无限期排队，引发了对 **MI300x 服务器** 的担忧。
   - 一位用户建议说 *队列只是很忙*，并提到在多次尝试后获得了成功，而另一位用户承诺回家后调查此问题。
- **高提交量淹没 MI300x 服务器**：管理员指出 **MI300** 每天收到约 **400 份提交**。
   - 管理员通知了 **AMD**，请求增加更多的 **runners** 来处理这些提交量。
- **MI300 Runner 停机故障**：事实证明 **runners** 确实宕机了。
   - 此前曾承诺一切正常，但管理员随后改口称 *实际上 runners 似乎宕机了，我们同时只能得到 2 个 runners*。
- **集群容量正在调查中**：在发现 *有人占用了我们的集群容量* 后，管理员正在进行调查。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1416547424545276027)** (2 messages): 

> `Eval Infra, PR Review` 


- **恢复 Eval Infra 工作**：一位成员宣布他们将于明天下午恢复 **eval infra** 的工作。
   - 他们请求其他人审查 **PR**（感谢 jack）并提供反馈。
- **等待 PR 反馈**：请求团队成员审查与 **eval infrastructure** 相关的 **Pull Request**。
   - 团队被要求就 **PR** 还需要什么提供建议，或确认其是否已准备好合并。


  

---

### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1416742324767035523)** (55 messages🔥🔥): 

> `Runner 队列与 AMD 协助、amd-gemm-rs 挑战发布、ROCm/iris 集成、PyTorch 版本兼容性、amd-all2all 相关说明` 


- ****Runner 队列**触发 AMD 协助！**: 由于 **runner 队列**拥堵，提交可能会遇到超时，但团队已通知 **AMD**，并将很快提供预计恢复时间（ETA）更新。
   - 使用 benchmark/test 可以暂时缓解拥堵，因为这些操作只会启动一个更快的任务。
- ****GEMM-RS** 挑战拉开帷幕！**: 第二个问题 **amd-gemm-rs** 挑战参赛者实现分布式的 **GEMM + reduce-scatter**。
   - 这些问题已在我们的 GitHub 上[开源](https://github.com/gpu-mode/reference-kernels/tree/main/problems/amd_distributed/gemm-rs)。
- ****Iris** 集成良好！**: 酷炫的 [iris 项目](https://github.com/ROCm/iris)现已可用。
   - 欲了解更多信息，请查看[作者在 YouTube 上的演讲](https://www.youtube.com/watch?v=GZqYr8_Q7DE)。
- ****PyTorch 2.8.0** 表现不佳**: 一位成员在使用 `torch load_inline` 和 **PyTorch 2.8.0** 时遇到了 *undefined symbol*（未定义符号）错误。
   - 该成员通过使用 `pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm6.4` 安装 **nightly 版 PyTorch ROCm 构建**解决了此问题。
- ****All2All** 算法分析！**: 在 **amd-all2all** 挑战中，*dispatch* 输出应将属于相同专家（experts）的 token 分组在一起，类似于参考内核。
   - 比赛的重点是快速通信，而不是实现 grouped_gemm，计算部分将在第二和第三个问题中强调。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1416143419134443520)** (10 messages🔥): 

> `CuTeDSL swizzle 模式、PTX 文档差异、TF32 数据类型` 


- ****Swizzle 大对决**：CuTeDSL vs. PTX 文档！**: 一位用户发现 **CuTeDSL** 与 **PTX 文档**在 **TF32** 数据类型的 swizzling atom 方面存在差异，特别是关于 `Swizzle<3,4,3>`，并分享了[截图](https://cdn.discordapp.com/attachments/1362196854460383353/1416143418496782437/Screenshot_2025-09-12_at_21.22.10.png?ex=68c9ba55&is=68c868d5&hm=e6b461266f0ecfc49cca56436fd9e10d4d1a571d796cced9c3876c2d3a6d133e&)。
   - 该用户认为 **CuTeDSL** 的实现是准确的，并复现了 [Lei Mao 博客](https://leimao.github.io/blog/CuTe-Swizzle/)中的结果。
- ****Swizzle 秘籍**：解码 PTX 文档！**: 一位用户澄清说，PTX 文档在地址字节上使用 **128B swizzle**（M=4, S=3 且 4 + 3 = 7），而组合布局（composed layout）的 swizzle 则作用于元素索引，这一点在[此处也曾提及](https://github.com/NVIDIA/cutlass/issues/2634#issuecomment-3292221378)。
   - 他们建议使用 `cute.make_swizzle(3, 2, 3)` 代替 `cute.make_swizzle(3, 4, 3)` 以产生相同的结果。
- ****PTX 谜题解开**：恢复图表！**: 一位用户详细说明了如何从 PTX 文档中恢复图表，涉及对 swizzle 模式的调整（例如使用 `(i,2,3)`）、atom 大小、矩阵形状缩放以及最后的 `//4` 除法。
   - 他们提供了带有[截图](https://cdn.discordapp.com/attachments/1362196854460383353/1417191712367050875/Screenshot_2025-09-15_at_18.41.11.png?ex=68ca3ee2&is=68c8ed62&hm=d81efe28dcd47c8c9ee38531987edead1a98250c6474ae72df8f590c05974bf1&)的示例来说明该过程，将具有相同索引的元素解释为一个 128bit 元素。


  

---

### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1416360923052113981)** (11 messages🔥): 

> `ML and Systems Book Design, GPU access limitations, Autograd machinery development, PicoTorch revitalization, Textbook to lower barriers for community` 


- ****书籍设计权衡映射****：一本专注于 Machine Learning (ML) 与系统的书籍在设计上面临**自顶向下与自底向上顺序**，以及“是什么”与“怎么做”呈现方式之间的内部权衡，作者正在规划开头、结尾及各章节之间的关系。
   - 由于时间有限且缺乏本地/裸机 GPU 访问权限，实现该书的最初目标（规划开头、结尾和章节关系）正面临挑战。
- ****Autograd 兴起，MLP 到来****：书籍的第一部分将侧重于**自底向上的方法**，开发所有的 **autograd machinery**，并以一个 **MLP 语言模型**作为结尾。
   - 第二部分将涵盖 **Transformers**、**Flash Attention** 以及可能的 **Diffusion 语言模型**，而第三部分将深入探讨编译。
- ****PicoTorch 项目推进****：[PicoTorch](https://github.com/j4orz/sctp/tree/master/picotorch/src) 项目此前在没有 GPU 加速的情况下运行了 **Karpathy 的 MLP** 前向传播，目前需要重新注入活力。
   - 第一章正在编写中，为引入的每个概念勾勒直观的图表、模型电路、数学公式和代码片段。
- ****教科书旨在助力 GPU Mode 社区****：正在编写一本教科书，以**降低 GPU mode 社区**创建类似于“从零实现 PyTorch”项目的门槛。
   - 该项目的口号可能是：*“我们要让你从零开始实现 PyTorch”*。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1416419690305163486)** (8 messages🔥): 

> `Kernel Development Path, GPU Mode Kernel Competition, Triton Benchmarks, BioML Trimul Kernel Competition` 


- **Kernel 开发初学者寻求帮助**：一位新用户询问了进行 **Kernel 开发**应遵循的正确路径。
   - 他们还询问了关于 **GPU Mode Kernel 竞赛**的预期、提交细节和要求。
- **用于 Triton 基准测试的 BackendBench**：一位用户询问了编写 **Triton** 函数的优秀基准测试工具（类似于 KernelBench）。
   - 一位成员推荐了 [BackendBench](https://github.com/meta-pytorch/BackendBench)，它帮助他们对约 **84 个用 Triton 编写的 PyTorch 算子**进行了基准测试。
- **BioML Trimul Kernel 竞赛即将截止**：距离参加 [BioML trimul kernel 竞赛](https://www.gpumode.com/v2/leaderboard/496?tab=rankings)仅剩 **14 天**。
   - 奖品将是由组织者设计并寄送的*从未见过的周边 (swag)*。


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1416318981710544986)** (1 messages): 

> `` 


- **Multi-GPU 频道：无活跃讨论**：提供的 multi-gpu 频道 Discord 日志不包含任何活跃讨论或适合总结的主题。
   - 日志缺乏关于新融资、模型、工具或指令中定义的其他感兴趣实质性内容。
- **Multi-GPU 频道：日志内容极少**：对 multi-gpu 频道日志的分析显示，缺乏包含技术总结相关信息的各种消息。
   - 消息中不包含任何链接、博客文章、代码片段或值得纳入总结的具体细节。


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1417223430549405748)** (2 messages): 

> `Video Models, Low-bit-training, GPU mode hackathon` 


- **调研视频模型与低比特训练**：一位成员正在为 10 月份的 **GPU mode 黑客松/线下见面会**提交的作品进行**视频模型**和**低比特训练 (low-bit-training)** 的调研。
   - 他们正在寻求专门针对应用于**视频模型**的**低比特训练**技术的研究论文。
- **分享家用视频模型 GitHub 列表**：该成员分享了一个包含**视频模型**集合的 [GitHub 仓库](https://github.com/vipulSharma18/we-have-video-models-at-homethe)。
   - 该仓库包括关于 **LLM** 的工作（可能是 **MobiChamps** 的），以及一些 **DiT** 和 **LLM** 训练方法（如 **Quartet**）。


  

---

### **GPU MODE ▷ #[irl-accel-hackathon](https://discord.com/channels/1189498204333543425/1416087968933740688/1416178117952798923)** (37 messages🔥): 

> `多模态推理, 训练优化, Gated DeltaNet, 稀疏 GNN 想法, 低比特训练` 


- **单机视频模型训练革命**：一位成员分享了一篇关于[在一天内单机训练大型视频模型](https://arxiv.org/pdf/2309.16669)的论文，通过 **FP4/FP8** 精度实现，尽管论文中使用 **FP16** 作为概念验证。
- **DeltaNet 期待上下文并行算子**：一位成员正寻求组建团队，为 [NVlabs 的 GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet?tab=readme-ov-file) 实现上下文并行（context-parallel）版本的算子（kernels），用于超长上下文训练，并指出其在 **Qwen 3** 中的应用。
- **稀疏 GNN 想法引发关注**：一位成员对稀疏 **GNN** 想法表示兴趣，特别是那些对拓扑、计算机图形学和向量数据库有影响的想法，并链接了相关的 [Arxiv 论文](https://arxiv.org/pdf/2507.13296)。
- **Blackwell Tensor Cores 诱惑稀疏矩阵乘法**：受 **Blackwell** Tensor Cores 的启发，一位成员考虑了涉及块稀疏（block-sparse）格式和 NVIDIA Tensor Cores 的问题，并链接了一篇关于[加速矩阵乘法](https://developer.nvidia.com/blog/accelerating-matrix-multiplication-with-block-sparse-format-and-nvidia-tensor-cores/)的博客文章。
- **线下 Hackathon 确认**：成员们确认此次 Hackathon 为线下活动，因为线上 Hackathon 较难设计，且更类似于他们的 **kernel 竞赛**。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1416143552932479169)** (180 messages🔥🔥): 

> `Playwrite MCP 问题, 小型模型的本地 Wikipedia 访问, Qwen/Qwen3-Next-80B-A3B-Instruct 与 llama.cpp, SIA-1: 自我改进 AI Agent, lambda stack vs lm studio` 


- **Playwrite MCP 抛出连接错误**：一位用户报告在 LM Studio 启动 Playwrite MCP 时遇到错误，似乎与连接问题有关，但他们指出这可能是用户操作失误。
   - 另一位用户开玩笑地回复道：*在我的机器上运行正常 (it works on my machine)*。
- **最强大的 AI 模型在 RAM 下运行效果最佳**：当一位用户询问最强大的 AI 模型时，一位成员提到你可以通过 RAM 使用任何模型，并指出本地能运行的最强 AI 模型是 **BF16** 格式的 **Kimi K2**，需要 **2.5TB RAM**。
   - 随后澄清道：*无论文件大小如何，你都需要同等大小的内存（VRAM + RAM 总和）来加载它，此外还需要额外的内存用于上下文*。
- **用户寻求访问 Wikipedia 文章的帮助**：一位成员询问有关在线或离线访问 **Wikipedia 文章**以供小型模型使用的工具，另一位用户分享了 [LM Studio Wikipedia](https://lmstudio.ai/lmstudio/wikipedia) MCP。
   - 有人提到在本地创建语义索引非常复杂，因为本地 Wikipedia 提取物缺乏搜索功能，而且 *LLM 在没有模糊搜索的情况下并不擅长猜测*。
- **NousResearch 的 Mephisto 讨论 World Sim**：一位用户分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=7ZEHdaABIJU)，视频中 **NousResearch 的 Mephisto** 深入探讨了基座模型（Base models）的技术细节，以及 Instruct 模型本质上是如何被训练来扮演 Instruct 角色的基座模型。
   - Mephisto 随后讨论了 NousResearch 如何启动 **World Sim**，这可能 *绝对是 Agent 世界的下一步*。
- **SIA-1：自我进化的 AI**：一位用户介绍了 **SIA-1**，声称它是 *世界上第一个真正的自我改进 AI Agent* [https://sia-1.net](https://sia-1.net)，它能学习改进自己的代码，一代又一代地进化。
   - 成员们对该 Agent 反应冷淡，一位成员问道：*请告诉那个 vibe-coded 的人换个更好的模型*。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1416200248283496580)** (113 条消息🔥🔥): 

> `KiCad 与 LLM 用于电路设计，SBC 用于 Searxng 对比 Obsidian，GPT-OSS-20B 与 VRAM 分配，Nvidia P40 EOL，RTX 5070 与 LLM 性能` 


- **LLM 通过电路设计让房子着火**：一名成员对使用 **KiCad** 等工具配合 LLM 进行电路设计表示谨慎，强调需要理解底层原理，以避免潜在的灾难性输出。
   - 他们补充说，将语言模型称为“AI”具有极大的误导性。
- **SBC 规格引发数据库辩论**：一名成员询问是否可以在拥有 4GB RAM 的 Raspberry Pi 上使用 **Obsidian**，作为缓慢的数据库或 Searxng 设置的替代方案。
   - 另一名成员反驳说，即使是性能极低的设备（*potato*）对于某些 LLM 任务也足够快，并建议关注访问模式而非数据库大小。
- **GPT-OSS-20B 在 VRAM 上遇到困难**：一名成员报告了在 7900xtx（24GB VRAM）上加载带有 128k 上下文的 **gpt-oss-20b** 时出现问题，尽管预期它应该能装下。
   - 另一位用户建议使用 **F16 的 KV quantization**，这能将 VRAM 占用降低到约 18.5GB，并建议关闭占用 GUI 资源的应用以释放 VRAM。
- **Nvidia P40 面临寿命终结 (EOL)**：一名成员考虑购买廉价的 **Nvidia P40s**，但担心即将停止的驱动更新和 CUDA 支持。
   - 有人指出 **Nvidia** 将在下一个主要工具包版本中停止对 Maxwell、Pascal 和 Volta GPU 的 CUDA 支持，但另一位用户报告说仅需 200 美元即可购得。
- **RTX 5070 规格引发升级考量**：一名拥有 **RTX 5070** (16GB VRAM)、i5 14600k 和 32GB DDR5 的成员寻求关于适合网站开发的 AI 模型的建议，或者是否需要升级。
   - 一位用户建议为编程任务购买 Github Copilot，因为 16GB VRAM 对于大型模型和 Agentic coding 并不理想。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1416143087469723688)** (265 条消息🔥🔥): 

> `Kimi 对比 GPT-5，Augment 代码扩展，Kimi K2 Groq，LLM 流程的交互式预览，API Key 对比登录账号` 


- **Kimi-K2 在脚本编写方面表现卓越**：成员们辩论了 **Kimi-K2** 在脚本编写方面的优势，一些人声称它的表现优于付费版 **Gemini**，尤其是在使用 **Groq** 进行编程时。
   - 其他人认为 **GPT-5** 更好，一位成员指出 *GPT-5 非常擅长 Web UI 和 nextjs*，而 **Kimi** 拥有 *非常出色的研究模式 (research mode)*。
- **使用 Kimi 进行 Augment 编程**：成员们讨论了在 VS Code 中结合 **Augment** 代码扩展使用 **Kimi**，用户可以提示各种模型进行代码更改和修复。
   - 一位用户描述了将 **Augment** 作为一种应用 Prompt 并通过循环调用 **Gemini** 或 **Kimi** 来修复代码的方式。
- **幻灯片功能引发 UX 脑暴**：一位成员强调了 **Kimi** 中令人印象深刻的交互式幻灯片生成功能，赞扬其 *实时更新* 和流畅感，并建议对于基于 LLM 的流程，过程的交互式预览非常重要。
   - 他们为 **Godot** 游戏引擎 Agent 提出了类似的方案，设想在代码生成过程中进行 *实时更新*，并提供节点和脚本的交互式预览。
- **Groq 托管的 Kimi-K2 引发关注**：一位用户询问在 **Groq** 上托管的 **Kimi K2** 是否存在问题，另一位用户抱怨需要取消 **3 小时消息限制**。
   - 该用户还要求能够 **编辑之前的 Prompt**，并表示 *其他所有 AI 平台都已经有这个功能了*。
- **CLI 中的 API Key 对比账号登录**：一位用户询问如何在不使用 API Key 的情况下，通过 kimi.com 账号登录来使用 **Claude Code** 和 **Qwen Code** 等 CLI。
   - 另一位用户建议为 **Claude Code** 使用 **API Key**，并提供了一个命令示例：`export ANTHROPIC_AUTH_TOKEN=sk-YOURKEY` 和 `export ANTHROPIC_BASE_URL=https://api.moonshot.ai/anthropic`。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1416153417201487926)** (115 messages🔥🔥): 

> `HDF5 Python Library, FineWeb pretraining, Hugging Face Spaces storage, Qwen3-Next modeling and Layernorms, Models for open world RPG RP` 


- **Hugging Face Spaces 的存储情况！**：关于 **HF Spaces**，上传的文件和生成的文件存储在虚拟机内的磁盘空间中，无法从外部任意访问，且除非使用付费的 Persistent Storage 选项，否则在 Spaces 重启时这些文件会消失。
   - 然而，在极少数情况下，由于文件名相同等错误，导致其他人的生成数据变得可见。
- **全新的 FinePDFs 数据集从 PDF 中解放 Token！**：发布了全新的 [FinePDFs 数据集](https://huggingface.co/datasets/HuggingFaceFW/finepdfs)，包含来自 **4.75 亿份文档**、涵盖 **1733 种语言**的约 **3 万亿个 Token**，数据完全源自 PDF。
   - 当与基于 HTML 的语料库混合时，它在各项基准测试中带来了显著的性能提升，建议将 PDF 数据的比例保持在总数据集的 **25%** 以下。
- **正弦和余弦函数深度探讨！**：一位成员询问为什么在位置嵌入（positional embeddings）中使用一对正弦和余弦波，而不能只使用单个正弦波。
   - 另一位成员回答说，同时使用正弦和余弦可以让模型以保留相对距离的方式表示位置信息，并且可以进行线性组合，因为仅靠正弦波会产生歧义。
- **Qwen3-Next 模型将 Layernorm 换成了 RMSNorm**：[Qwen3-Next 模型卡片](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)提到了诸如 *zero-centered and weight-decayed layernorm* 的稳定性优化，但实际上它在 Transformers 中使用的是 RMSNorm。
   - 澄清指出，这里根本没有涉及 Layernorm，只是在训练中使用了带有零中心化 Gamma 和权重衰减的 RMSNorm，而在推理（inference）时则是普通的 RMSNorm。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1416771469995675691)** (6 messages): 

> `Agents Course, smol course, MCP course, LoRA finetuning, Transformers architecture` 


- **深入探索 Agent 开发**：一位成员建议在学习 Agent 时使用 **80-20 法则**，建议集中精力直接进行构建，因为 *那 20% 的动手实践将在过程中教会你 80% 的知识*。
   - 他们认为 *深度钻研中 80% 都是无聊的东西*。
- **smol 课程截止日期**：一位成员询问了 **smol 课程** 的截止日期，以及是否仍可以参加 **MCP 课程** 并获得认证。
   - 尚未提供答案。
- **LoRA 微调的挑战**：一位成员正在继续使用 **LoRA** 微调 LLM 的旅程，发现这很有挑战性但也很有用。
   - 该成员表示，他们正在学习 *如何控制自己的压力*。
- **Transformer 解码器学习计划**：一位成员计划参加 **smol 课程** 并研究 **Transformers 架构**（解码器）。
   - 未提供其他细节。
- **Agent 课程注册问题**：一位新成员尝试注册 **Agent 课程**，但在 **MCP** 和 **smol** 课程列表中没有看到它。
   - 他们正在寻求帮助以解决此问题。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1416410230748876954)** (2 messages): 

> `HF models, Fine-tuned models` 


- **百款微调模型上线 HF**：一位成员分享说，他的一位朋友兼导师在 8-9 个月内，在 [HuggingFace](https://www.linkedin.com/posts/moyasser_ai-ai-machinelearning-huggingface-activity-7372687867359252480-yd9F) 上发布了 **100 个生产级 AI 微调模型**。
   - 该成员请求大家 **认可这份辛勤工作**。
- **模型制作者的马拉松**：一位密切联系人在 Hugging Face 上完成了 100 个 **生产级 AI 模型** 的里程碑，展示了过去几个月的专注努力，引发了讨论和关注。
   - 重点在于庆祝他的卓越成就及其对社区的贡献。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1416449418302853283)** (13 messages🔥): 

> `Voxtral finetuning, Dialectical Agentic CrossSphere AI, Refrag Efficient LLM Compression, Image to Space` 


- **Voxtral 让语音训练变得经济实惠**：Voxtral 使患有语言障碍或重口音的用户能够微调模型，在 A6000 上进行一小时训练的成本仅为 **$0.26**，并提供了[制作数据集](https://huggingface.co/spaces/Tonic/VoxFactory)的工具。
   - 该工具支持通过微调达到完美效果，用户可以将模型、数据集推送到 Hugging Face，并添加一个演示 Space（**支持 CPU 运行！！免费！！**）。
- **Dialectical Agentic CrossSphere AI 参加 Hackathon**：一名用户正在为其参加 OpenAI Hackathon 的项目 **Dialectical Agentic CrossSphere AI** 寻求反馈，链接如下：[OpenAI GPT OSS 20B](https://huggingface.co/spaces/Dennisgbay22/openai-gpt-oss-20b)。
   - 另一位用户称赞了该 AI 的游戏、图像和叙事能力。
- **Refrag 发布：高效 LLM 压缩技术**：一位用户分享了他们的博客文章，解释了 **Refrag**——一种用于高效 LLM 压缩和课程学习（Curriculum Learning）的方法：[Understanding Refrag](https://medium.com/@limemanas0/understanding-refrag-efficient-llm-compression-and-curriculum-learning-explained-3452498f99e8)。
   - 博客文章重点介绍了 Refrag 涉及的效率和技术。
- **Image-to-Space 工具实现 HF 仓库迁移**：一位用户介绍了一种通过包含 Hugging Face Space 的图像来迁移仓库的新方法，使用了 [image-to-space decoder](https://huggingface.co/spaces/broadfield-dev/image-to-space)。
   - 另一位用户提交了一个 PR 以改进该工具的功能（[HF 上的讨论](https://huggingface.co/spaces/broadfield-dev/image-to-space/discussions/1)），称赞其*非常酷且具有创意*。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1417120879640449105)** (2 messages): 

> `Style Transfer, WCT2 Methods, Segmented Images` 


- **风格迁移方法需要分割图像**：像 **WCT2** 这样的风格迁移方法有时需要分割后的图像，这带来了挑战。
   - 这一要求限制了这些方法在难以或无法获取分割图像的场景中的适用性。
- **风格迁移实现的注意事项**：实现 **WCT2** 等风格迁移方法通常需要仔细考虑图像分割技术。
   - 对分割图像的需求会增加流水线的复杂性，需要额外的预处理步骤，并可能影响整体性能。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1416202968625188945)** (3 messages): 

> `Qwen2.5-72B fine-tuning, Database for Chat History, Maintaining User Sessions` 


- **Qwen2.5-72B 微调：寻求专家**：一名成员询问了关于微调 **Qwen2.5-72B** 的经验，并请求有相关专业知识的人员直接私信。
- **启动关于数据库和用户状态的讨论**：提问中包括了*使用数据库存储聊天记录*以及*维护用户会话和用户状态*的需求。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1416165207058812938)** (9 messages🔥): 

> `Fine-tuning course details, VRAM concerns for smaller models, In-person study group in NYC, Leaderboard evaluation for custom use cases, smol-course` 


- **VRAM 不够？小模型来救场！**：一名成员询问由于 **VRAM** 有限，是否可以使用更小的 **Smaller Models**。
   - 他们询问了关于在自定义用例上进行微调的问题。
- **纽约学习小组正在组建！**：一名成员提议在**纽约（NYC）**组建一个**线下学习小组**。
   - 他们提出在市内组织见面会。
- **解读微调课程**：一名成员询问了**微调课程**的开始日期以及报名地点，寻求对课程动态的澄清。
   - 另一名成员提供了 [smol-course](https://huggingface.co/smol-course) 的链接，并建议关注该组织并从 [Unit 0](https://huggingface.co/learn/smol-course/unit0/1) 开始。
- **Smol Course 开课了！**：成员们被告知关注 Hugging Face 组织并从[这里](https://huggingface.co/learn/smol-course/unit0/1)开始课程。
   - 这将使他们能够开始 **smol-course** 并开始学习。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1416481312474267738)** (11 条消息🔥): 

> `Agent 课程介绍，Token 设置的新手错误，第一单元介绍` 


- **新手开始 Agent 课程**：包括 Nouha、Lez、Karthik、Leo Kinyera 和 Nay Lin 在内的几位新成员进行了自我介绍，并表达了对开始 Agent 课程的兴奋之情。
- **Token 设置问题已解决**：一位成员承认犯了一个没有设置 **token** 的 *新手错误*，该问题已迅速得到解决。
- **第一单元进行中**：一位成员报告称正在学习 Agent 课程的 **第一单元 (unit one)**。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1416309313609269299)** (85 条消息🔥🔥): 

> `工会与法西斯主义，LLM 与贝叶斯推理，AI 与拓扑斯理论 (Topos Theory)，Transformer 中的位置编码，深度学习与湍流` 


- **工会与法西斯主义事实的纠葛**：一场关于工会与法西斯主义关系的讨论展开，澄清了虽然所有法西斯主义者都支持工会，但并非所有工会成员都是法西斯主义者。
   - 讨论强调，法西斯法团主义涉及雇主、雇员（通过国家认可的工会）和国家对企业的共同治理，因此工会对法西斯主义至关重要，参考 [此 Discord 消息](https://discord.com/channels/714501525455634453/986699377257119794/1416791624171786320)。
- **LLM 审视贝叶斯信念**：讨论围绕一篇关于 LLM 和贝叶斯推理的论文展开，一位成员指出该论文揭开了关于 LLM 的一些先入为主观念的神秘面纱，参考 [Leon Chlon 的 substack](https://leonchlon.substack.com/p/llms-are-bayesian-in-expectation)。
   - Yannick Kilcher 评论道：*LSTM 也可以很好地进行上下文学习 (in-context learning)。这是语言建模的属性，而不是架构的属性*，并且 *Transformer 将 token-位置对序列作为参数，因此它对顺序是完全不变的，因此是完全贝叶斯的*。
- **拓扑斯理论 (Topos Theory) 引人入胜，但被彻底否定？**：一位成员分享了一篇关于 **AI** 与 **拓扑斯理论** 交叉领域的论文 ([ArXiv 链接](https://www.arxiv.org/abs/2508.08293))，质疑其合法性和实用性，其他成员也纷纷加入讨论。
   - 然而，另一位成员认为 **范畴论 (category theory)** 对 **ML** *完全无用*，理由是从有限数据集进行泛化的需求需要抽样理论和 L_2 空间。
- **正弦还是余弦？位置编码的困惑依然存在！**：一场关于理解 Transformer 中位置编码的讨论展开，一位成员就如何在数据有限或质量较低的情况下理解这些概念寻求建议。
   - 一位成员解释了在位置编码中同时使用 **正弦 (sine)** 和 **余弦 (cosine)** 的原因，指出如果只使用正弦，模型很难判断是否为同一个角度，因为如果向量值为 0.5，它可能代表 30 度或 150 度；然而，也有观点认为在各层中它们能够相互重构，因为 *sin(2x) = 2 sin(x) cos(x)*。
- **深度学习深入研究湍流动力学**：一位成员想知道深度学习是否可以解释为逆转湍流，将其比作从微小涡流中重建大型涡流，另一位成员称这种想法有些“精神分裂”。
   - 相比之下，另一位成员建议 [这篇论文](https://arxiv.org/pdf/2303.08797) 实现了 *你闲暇时所想的东西……但做得更直接*。


  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1416176802862338160)** (19 messages🔥): 

> `Spiking Brain-inspired Large Models, Anthropic's Research, OpenAI's Research, Decreasing Work-Related Usage, Noise Among Older Cohorts` 


- **Gaslighting 语言分析揭示了虚假信息的潜力**：一位成员开玩笑说，Gaslighting 的西班牙语翻译 *manipulación psicológica* 在涉及 **Spiking Brain-inspired Large Models** 的潜在[虚假信息使用案例](https://www.arxiv.org/pdf/2509.05276)中是有意义的。
   - 另一位成员分享了一篇题为 "SpikingBrain Technical Report: Spiking Brain-inspired Large Models" 的论文链接，以探索这种联系。
- **Anthropic 与 OpenAI 经济报告**：成员们查看了 [Anthropic](https://www.anthropic.com/research/economic-index-geography) 和 [OpenAI](https://cdn.openai.com/pdf/a253471f-8260-40c6-a2cc-aa93fe9f142e/economic-research-chatgpt-usage-paper.pdf) 在同一天发布的关于 AI 用户行为的报告。
   - 讨论集中在“用户正在用 AI 做什么”，并猜测是否有一家公司试图通过这些发布来抢占另一家的风头。
- **OpenAI 报告中未包含“AI 作为朋友”的使用案例**：一位成员指出，**OpenAI** 的报告没有涵盖“AI 作为朋友”的使用案例，特别是考虑到最近出现的 **sycophancy**（谄媚）问题。
   - 该成员还观察到，相对于其他使用模式，与工作相关的使用量正在下降。
- **AI 使用中不同年龄组的平滑度差异**：观察到使用图表上 18-25 岁年龄组的线条比其他年龄组更平滑。
   - 给出的一种可能原因是 18-25 岁组的用户数量最多，或者其数据中的噪声最少。
- **高龄组面临噪声增加**：观察到高龄组的噪声有所增加，这可能是由于可用样本数量较少。
   - 这种增加的噪声可能是由于老年人口统计数据中不断增加的方差导致的。


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1416447947347988491)** (3 messages): 

> `` 


- **未发现相关的 Agent 话题**：遗憾的是，没有发现包含感兴趣话题的消息，因此无法创建有效的摘要。
- **填充话题**：遗憾的是，没有发现包含感兴趣话题的消息，因此无法创建有效的摘要。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1416190525484302336)** (15 messages🔥): 

> `MobileLLM-R1-950M Release, AI Alignment, AI Constitutional Assembly, Cloud Providers Profiting, NVIDIA` 


- **Facebook 发布 MobileLLM-R1-950M**：Facebook 发布了他们的新模型 [MobileLLM-R1-950M](https://huggingface.co/facebook/MobileLLM-R1-950M)。
   - 该发布旨在将强大的语言模型引入移动设备，实现端侧处理并减少对云服务的依赖。
- **处处皆是对齐者 (Aligners All The Way Down)**：一位成员分享了 [alignmentalignment.ai](https://alignmentalignment.ai/Aligners) 的链接，并评论道：“处处皆是对齐者……”。
   - 该链接分享了与 **AI Alignment** 相关的内容和研究。
- **建议成立 AI 设计者制宪会议**：一位成员链接到一条建议成立“人工智能设计者制宪会议”的推文。
   - 链接指向 Shaswat Goel 的一条 [推文](https://x.com/ShashwatGoel7/status/1966527903568637972)。
- **云服务提供商在 AI 淘金热中获利**：一位成员表示，“唯一能从中赚到钱的人是云服务提供商和云基础设施提供商”。
   - 另一位成员回应道：“你知道人们是怎么说淘金热的，卖铲子吧”，其中 **NVIDIA** 被归类为云基础设施提供商。
- **AI 安全视频中的 AI PDF 编辑器广告**：一位用户指出，在一段关于 **AI Safety** 的视频描述中看到 **AI 驱动的 PDF 编辑器**广告是多么讽刺。
   - 该用户质疑是否还有比这更虚伪的事情。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1416163016021377146)** (101 条消息🔥🔥): 

> `初创公司的 MBA 化，AI 短信管家 poke.com，OpenAI 模型规范更新，Naveen Rao 离开 Databricks，GPT-5 ‘High New’` 


- **初创公司屈服于 MBA 化**：由 Michael Seibel 发起的一个话题感叹道，计算机科学专业的学生表现得像 MBA 毕业生，追求融资和估值，而不是构建酷炫的东西和解决用户问题，详见[此处](https://xcancel.com/mwseibel/status/1965936563306864706)。
   - 回复中辩论了这种转变是自然的后期采纳，还是投资者/YC 激励机制的结果。
- **Poke.com 推出 AI 短信管家**：Interaction 介绍了 **poke.com**，一项新的 AI 短信服务，并发布了由 General Catalyst 领投的 **1500 万美元 A 轮融资**消息（[推文](https://xcancel.com/interaction/status/1965093198482866317)）。
   - 一些人看到了流畅的 UX 和病毒式叙事，而另一些人则质疑其有用性、清晰度以及 AI 的语气；该产品代表你发送短信以协调聚会、约会、旅行等。
- **xAI 转向专家级 AI 导师**：Rohan Paul 强调了 xAI 的转变：裁减 **500** 名通用数据标注员，同时将专家级 AI 导师规模扩大 **10 倍**（[推文](https://xcancel.com/rohanpaul_ai/status/1966943783276396730?s=46)）。
   - 此举将“人机回环”（human-in-the-loop）工作缩小到昂贵的领域专家，并依靠自动化处理常规任务，旨在提高高风险话题的精准度。
- **S3 Vectors 会终结向量数据库吗？**：讨论源于[这篇博客文章](https://zilliz.com/blog/will-amazon-s3-vectors-kill-vector-databases-or-save-them)，探讨 **Amazon S3 Vectors** 是否会取代传统的向量数据库，因为嵌入解决方案正趋向于从本地 nvme 磁盘到对象存储 (S3) 的成本和延迟权衡。
   - 一位用户引用了一个令人惊讶的说法：*某款流行的 AI 笔记应用在向量搜索上的支出是 OpenAI API 调用支出的两倍*，并怀疑他们是否应该更认真地对待“RAG 已死”的论调。
- **GPT-5 Codex 升级**：OpenAI 发布了其编程模型 **Codex** 的升级，包括新版本的 **GPT-5** 和一篇简短的回顾文章（[链接](https://www.latent.space/p/gpt5-codex)）。
   - 一位用户报告说，`--resume` 标志在更新期间失效，导致他们无法恢复对话。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1416163405265371299)** (10 条消息🔥): 

> `Higgsfield 融资 5000 万美元，Adobe 价值转移，Gen Z AI 创始人` 


- **Higgsfield 冲刺 5000 万美元 A 轮融资**：AI 视频初创公司 **Higgsfield** 宣布了由 **GFT Ventures** 领投的 [5000 万美元 A 轮融资](https://xcancel.com/arfurrock/status/1966588530064289841?s=46)，营收运行率达到 5000 万美元——三个月内增长了 4.5 倍——并正在推出 **Higgsfield Ventures** 以支持 AI 原生的 Gen Z 创始人。
- **Adobe 的 AI 焦虑：1000 亿美元市值蒸发？**：Anjney Midha 认为 [AI 编辑技术的进步](https://xcancel.com/anjneymidha/status/1967266304878068044?s=46) 可能会将 **Adobe** 的 1000 亿美元市值转移到前沿 AI 实验室（**Flux Kontext**、**Gemini Nano**）。
- **Gen Z 将获得 AI 助力**：**Higgsfield Ventures** 计划支持 AI 原生的 **Gen Z** 创始人，为年轻人才提供更多机会。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1416137308175994962)** (74 messages🔥🔥): 

> `Nepal Discord Election, MLC-LLM issues, sglang vs vllm, GPT-OSSH4 in claude code, Demis Hassabis` 


- **Nepal 在 Discord 上选举领导人！**：成员们开玩笑说 **Nepal** 在 **Discord** 上投票选出其领导人，并讨论了接下来的计划——为所有公民提供 AI waifus 和 husbandos。
   - 一位成员分享了一篇关于 [Nepal 正在经历完整革命](https://adriananimov.substack.com/p/empyre-of-ai)的文章。
- **MLC-LLM 实验遇到问题**：一名成员正在尝试向 **MLC-LLM** ([https://github.com/mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm)) 添加自定义模型，但在注入模型时不断遇到问题。
   - 有成员建议，这可能是由于会话未正常终止导致上下文混淆，或者可能类似于 *llama.cpp* 上的[这个 issue](https://github.com/ggml-org/llama.cpp/pull/15913)。
- **内部使用 sglang 和 vllm**：一位成员表示他们内部只使用 **sglang** 和 **vllm**。
   - 另一位成员提到他以前没试过 **sglang**，但其 GitHub 仓库看起来很有前景，他主要想利用 **MLC** 尝试在 Claude Code 中实验 **gpt-ossh4**。
- **Qwen 团队更倾向于 XML 而非 JSON**：有人注意到 **Qwen** 团队更喜欢 **XML** 而非 **JSON**，一名成员计划在发布其 Agent 系统之前也采取同样的做法。
   - 成员们认为需要一种更节省 Token 的新格式，因为所有的空格对资源并不友好。
- **Sir Demis Hassabis 讨论世界模型**：一位成员分享了一段 [YouTube 视频](https://www.youtube.com/watch?v=Kr3Sh2PKA8YI)，Sir **Demis Hassabis** 在视频中讨论了通往 AGI 和具身智能 (Embodied AI) 系统的多模态（构建世界模型）路径、LLM 的局限性，以及令人惊叹的 **Genie 3** 世界。
   - 该视频涵盖了 **Alphafold** 在研究生物学和医学领域的现实成就。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1417076474095337482)** (1 messages): 

> `Adversarial Idea Presentation, Strength in Weakness` 


- **对抗式观点展示揭示隐藏优势**：以“对抗模式”展示一个观点，可能会无意中发现额外的优势，而这些优势原本只是被界定为弱点。
- **将弱点转化为优势**：当观点以对抗方式呈现时，潜在的好处可能会被视为缺点，这突显了框架定性 (framing) 的重要性。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1417205231858618388)** (5 messages): 

> `OpenAI Economic Research, Anthropic Economic Index, ChatGPT usage growth, AI Friend mapping` 


- **AI 经济学论文同步发布**：**OpenAI** 和 **Anthropic** 同时发布了论文；OpenAI 发布了 [ChatGPT 使用情况的经济研究](https://cdn.openai.com/pdf/a253471f-8260-40c6-a2cc-aa93fe9f142e/economic-research-chatgpt-usage-paper.pdf)，Anthropic 发布了其 [2025 年 9 月经济指数报告](https://www.anthropic.com/research/anthropic-economic-index-september-2025-report)。
- **ChatGPT 用户基数和参与度飙升**：根据 [OpenAI 的数据](https://openai.com/index/how-people-are-using-chatgpt/)，**ChatGPT** 的注册人数和人均使用量都有大幅增长。
- **“AI Friend” 类别面临质疑**：一位成员对将某些数据归类为“AI 朋友”表示怀疑。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1416509716905070632)** (2 messages): 

> `DNS Tunneling Chat Client, AI Killing Focus` 


- **通过 DNS 隧道聊天客户端连接 LLM**：一名成员创建了一个[利用 DNS 隧道在 WiFi 强制门户下与 LLM 聊天的工具](https://github.com/accupham/llm-dns-proxy)，从而可以在飞机上无需额外付费即可访问 LLM。
   - 他们请求大家进行“吐槽 (roasts)”，考虑到目前的 AI 氛围，有些人认为这是一个冒险的请求。
- **AI 被指责导致注意力缺失危机**：一位成员分享了一篇[博客文章](https://blog.maxcomperatore.com/why-ai-is-killing-your-focus-and-the-system-to-fix-it)，认为 AI 正在损害我们的专注能力。
   - 文章详细介绍了一套在 AI 驱动的干扰充斥的世界中夺回注意力的系统。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1417205231858618388)** (5 条消息): 

> `OpenAI, Anthropic, AI Usage, AI Friend` 


- **AI 经济研究报告同步发布的竞赛？**：成员们分享了 [OpenAI](https://cdn.openai.com/pdf/a253471f-8260-40c6-a2cc-aa93fe9f142e/economic-research-chatgpt-usage-paper.pdf) 和 [Anthropic](https://www.anthropic.com/research/anthropic-economic-index-september-2025-report) 同时发布的经济研究论文链接，一位成员对发布时机表示好奇。
   - OpenAI 的论文研究了 **ChatGPT usage**（使用情况）趋势。
- **ChatGPT 用户增长势头不减**：成员们分享了 [OpenAI 指出](https://openai.com/index/how-people-are-using-chatgpt/) 的信息，除了人均使用量外，注册 **ChatGPT** 的人数也在增加。
   - 附带了多张图片，但讨论较少。
- **"AI Friend" 使用案例受到质疑**：一位成员询问图表中的哪些具体数据点可能对应于 *"AI Friend"* 的使用案例。
   - 随后他简单地表示了 *"doubt"*（怀疑）。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1417244881042276424)** (3 条消息): 

> `fastWorkflow beats Claude Opus 4.1, GEPA API Improvement, Tau Bench retail` 


- **fastWorkflow 框架击败 Claude Opus 4.1！**：一位成员发现他们新的 **fastWorkflow** 框架实现在 Tau Bench 开发集上**赶超了 Claude Opus 4.1**。
   - 这些测试使用了 **DSPy** 来构建 Agent 和进行参数提取，并参考了其仓库中的 [retail workflow 示例](https://github.com/radiantlogicinc/fastworkflow)。
- **请求改进 GEPA API！**：另一位成员表示有兴趣学习使用 **GEPA** 处理 Agent 场景的经验。
   - 他们还希望在 **GEPA API** 有任何可能更好地支持此类场景的改进时获得通知。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1416465047311356034)** (51 条消息🔥): 

> `GEPA for code generation, Manim and DSPy video, Rules as inputs for optimization, MCP Server, Zero Shot Categorization` 


- **GEPA 生成高质量代码**：DSPy 最新的优化器 **GEPA** 正是为代码生成而设计的，[这篇论文](https://arxiv.org/pdf/2507.19457)（第 6 节）展示了它为 GPU/NPU 生成 **CUDA/C++ 代码**的能力。
   - 其中一位原作者乐意就 GEPA 进行更深入的讨论。
- **多媒体 Manim 魔术师创造电影奇迹**：一位成员分享了一个使用 **DSPy** 自定义流水线制作的 [视频](https://cdn.discordapp.com/attachments/1161519469319946286/1417121692026929163/RequestGeneration.mp4?ex=68c9fdac&is=68c8ac2c&hm=727806c1f99beaee0816d280cbdd41519070c660a1efb588ee240b5166ab134f&)，其中包括旁白脚本生成、**Manim 场景生成**以及自动修复反馈循环。
   - 该视频利用了 **Signatures**、**KNNFewShot** 和 **ChainOfThought**，但目前尚未开源。
- **优化负担过重**：一位用户发现，*在每次对指令进行微小修改后运行优化，似乎会导致工作流过于沉重且缓慢*。
   - 有人建议将规则列表也作为输入的一部分，以便将 Prompt 优化为**能够适应不同规则，甚至是未见过的规则**。
- **MCP Server 寻求 DSPy 高手**：一位用户好奇是否有人使用 **DSPy** 来微调他们的 **MCP server** 描述和示例，并认为针对平均结果进行微调可能就足够了。
   - 另一位成员肯定了这个想法，建议用户可以根据客户端推断调用的 LM，并表示 *“这个主意太棒了，我敢打赌如果你能实现它，人们会愿意为此付费订阅服务”*。
- **优雅地进行分类**：一位用户正尝试对约 2000 份文本（电子邮件）进行 **zero-shot categorization**（零样本分类），并希望为每个主题提供示例或种子词。
   - 建议在 Signature 定义中使用 `typing.Literal`，并加载 JSON 数据来创建 `dspy.Example` 对象，同时推荐了[这个教程](https://dspy.ai/tutorials/rag/)。


  

---

### **DSPy ▷ #[colbert](https://discord.com/channels/1161519468141355160/1250300504462856265/1417168565437726741)** (1 条消息): 

> `Contextual Chunking, ColBERT Models, Late Chunking, MaxSim Algorithm` 


- **Contextual Chunking 提升性能**：一位用户发现，在每个分块（chunk）前添加 **contextual summary**（上下文摘要）可以显著提高性能，即使在使用 **ColBERT models** 时也是如此。
   - 然而，他们指出为每个分块生成摘要的成本很高，因此正在寻找更高效的替代方案。
- **探索 ColBERT 的 Late Chunking**：该用户建议在 **ColBERT** 中使用 **late chunking**：一次性对全文进行编码，然后再将 embeddings 拆分为分块。
   - 这种方法为每个分块分配其对应的 embedding 列表，从而实现更高效的处理。
- **MaxSim 算法与 CLS Token 依赖性**：该用户质疑 **ColBERT 的 maxsim 算法** 是否依赖 **CLS token** 来获得最佳性能，担心文本中间缺少 CLS token 的分块会出现问题。
   - 他们询问在这种情况下，对每个分块应用 **maxsim** 时省略 **CLS token** 是否安全。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1416149270075019294)** (18 条消息🔥): 

> `Mojo Package Managers, Binary vs Source Distribution, Pixi and Conda, Apple M1 Compatibility` 


- **Mojo 包管理热潮：社区权衡管理方法**：一位社区成员询问是否要为 Mojo 创建一个新的包管理器，专门用于处理二进制分发（binary distribution），然而 Mojo 团队[指出](https://www.modular.com/)：
   - `.mojopackage` 已经涵盖了二进制分发的许多优点，并可与 **Pixi** 配合使用，此外团队正有意倾向于使用 **Conda** 和标准的 **Python** 包格式以促进采用。
- **二进制的忧郁：源码分发稳步前行**：有人指出二进制分发存在缺点，这就是为什么许多语言更倾向于源码分发（source distribution），但该用户很好奇是否在某些场景下，一个更明确的以二进制为中心的包管理器会很有用，例如对于大型依赖项或预构建库。
   - Mojo 团队表示，Mojo 在笔记本电脑上可以在 **30 秒**内编译约 20 万行代码，并且 Pixi 通过 **Conda** 处理 **C/C++** 依赖项。
- **Pixi 增强：动态去中心化依赖**：一位社区成员强调了 [pixi-build-mojo](https://www.modular.com/t/mojo-packages-can-now-depend-on-other-packages-in-github/2144)，它通过使用 **Git** 中的包，实现了像 **Go** 一样完全去中心化的包系统。
   - 提及使用 **Pixi** 指定系统依赖项的能力也非常有效。
- **M1 混乱：MacBook 运行 Mojo 遇到困难？**：一位使用运行 **Python 3.13** 的 **Apple M1 MacBook Air** 的用户询问了 Mojo/MAX 与该版本 **Python** 的兼容性。
   - Mojo 团队确认其兼容，鼓励使用 `pixi` 来隔离 Python 版本，并建议在 CPU 上运行应该没问题（尽管速度较慢），因为 **Apple Metal** 支持尚处于早期阶段。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1416984660000378881)** (33 条消息🔥): 

> `InlineList 移除, Small List 优化, Allocator/Storage API, Mojo LSP 状态, 网络更新` 


- **InlineList 消失了：它去哪了？**：成员们讨论了 `InlineList` 的移除，并担心替代方案（`InlineArray` 和 `List`）无法完全覆盖其应用场景，正如 [changelog](https://example.com/changelog) 建议使用 `InlineArray` 或带有 `capacity` 构造函数的 `List`。
   - 一名成员建议，具有固定容量的栈分配变长类型将是理想的选择，另一名成员提到 Allocator API 可能是未来的方向。
- **Small List 优化停滞**：目前存在一种“小列表优化”（small list optimization），可以将部分项目内联存放，但如果列表增长，它们会被复制到堆中。一名成员提到，可能会探索将内联大小作为一个参数。
   - 一名成员提到，由于将其暴露给用户的复杂性以及可移动元素的 trait 要求，`List` 目前没有 SBO（Small Buffer Optimization）。
- **Allocator API 即将到来？**：讨论围绕着 allocator/storage API 处理内联分配的潜力展开，一名成员表示：“我听到的是，我需要更多地投入到我的 allocator/storage API 工作中”。
   - 该 API 的开发正等待参数化 trait 和 `requires`，这推迟了其进展。
- **Mojo 迎来重大 LSP 重构**：一名成员询问了 Mojo 的 Language Server Protocol (LSP) 状态，另一名成员回答说它已经存在，并且很快将进行一次“重大重构”。
   - 未提供关于重构的更多细节。
- **网络更新受阻 🚧**：一名成员表达了对网络更新的期待，但另一名成员回应道：“那里有很多阻碍（blockers）”。
   - 未说明这些阻碍的具体性质。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1416146127786479757)** (36 条消息🔥): 

> `Aider 的 RepoMap, 免费 C# 模型, AGI 预测, LM Studio 问题, GPT-5 Codex` 


- **RepoMap 提升 Aider 的实际表现**：一位用户指出，在 Aider 中使用 **RepoMap** 可以提供额外的上下文（如文件名和函数签名），增强 **LLM** 对可用资源的感知，理论上能使 [leaderboard 结果](https://leaderboard.techfren.net/) 更接近真实的编程场景。
   - 然而，他们承认，在简单问题上的 **benchmark 测试** 与实际代码体验之间仍存在显著差距。
- **寻找免费的 C# 模型**：一位用户正在寻找一个精通 **C#** 的免费非本地模型，其他成员建议通过 [OpenRouter](https://openrouter.ai/) 尝试 **Qwen Coder** 和 **Deepseek Coder**，并指出 Gemini 2.5 Pro 可能有免费层级。
   - 该用户随后报告了通过 OpenRouter 使用 **Qwen** 时的问题，由于 API key 可能不正确而收到 *AuthenticationError*。
- **AGI 到来：AI 何时会削减白领工作？**：一位用户在频道中发起投票，询问 **AGI** 何时可能减少超过 30% 的白领工作，选项从 2027 年到 2040 年以后不等，并将 **AGI** 定义为经济影响而非抽象智能。
   - 另一名成员开玩笑地预测，这会发生在“从现在到宇宙热寂之间的某个时间，或者永远不会”。
- **LM Studio 与 Aider：起步艰难**：一位用户在 **LM Studio** 中使用本地 **Qwen3-Coder-30B** 模型运行 Aider 时遇到问题，分享了设置截图但未说明具体问题。
   - 另一名成员询问是否设置了必要的环境变量，暗示可能存在配置问题。
- **GPT-5 Codex：新的编程模型？**：一位用户询问 Aider 对 **GPT-5 Codex** 的评分，并引用了一篇关于该新模型的 [The New Stack 文章](https://thenewstack.io/openai-launches-a-new-gpt-5-model-for-its-codex-coding-agent/)。
   - 另一人澄清说，该模型“尚未通过 API 提供”。


  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1416525332135149628)** (6 messages): 

> `Ollama context window limits not respected, lm studio or llamafile suggestion, --watch-files implementation on Linux, Gemini issues with Aider` 


- **Ollama 上下文长度限制被忽略**：一位用户报告称，尽管在配置文件中设置了 `OLLAMA_CONTEXT_LENGTH` 和其他参数，**aider** 配合 **Ollama** 使用时仍不遵守上下文窗口限制，导致 VRAM 占用过高并导致机器卡死。
   - 该用户在 `.aider.model.settings.yml` 中配置了 `num_ctx` 和 `max_tokens`，并在 `.aider.model.metadata.json` 中配置了 `max_tokens`、`max_input_tokens` 和 `max_output_tokens`。
- **建议使用 Ollama 的替代方案**：一名成员建议使用 **LM Studio** 或 **llamafile** 作为替代方案。
   - 未提供进一步的讨论或理由。
- **基于文件系统的 “--watch-files” 实现**：一名成员询问 `--watch-files` 选项在 Linux 中是如何工作的，特别是它是否依赖 **inotify** 或需要来自 IDE/编辑器的通信。
   - 另一名成员澄清说它是基于文件系统的，不需要来自编辑器的特定消息。
- **Gemini 集成停滞，可能由于 User-Agent 屏蔽**：一位用户报告说，尽管 API Token 正确且在 `curl` 和 **Gemini CLI** 中功能正常，但 **aider** 在等待 **Gemini** 模型响应时会挂起。
   - 该用户怀疑 **Gemini** 可能会根据 User-Agent 进行屏蔽，目前运行的是 **aider 0.86.1**。


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1417328287172263976)** (1 messages): 

> `Earning $100k in a week, Telegram scams` 


- **不切实际的盈利承诺诱导巨额佣金**：一名成员提供了一个快速致富方案，承诺帮助前 10 名有兴趣在*一周内赚取 10 万美元或更多*的人，并要求在收到利润后**返还 10% 的利润**。
   - 感兴趣的人被指示通过 Telegram 用户名 @Joanna_Dwayne 发起联系，这一举动引发了对诈骗的怀疑。
- **Telegram 联系方式引发警惕**：要求通过 **Telegram** 联系用户以获取*快速致富方案*是诈骗者常用的模式。
   - 用户应警惕任何要求在未经验证的渠道进行初步联系并承诺不切实际的高回报的提议。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1416203482578419913)** (25 messages🔥): 

> `Tensor.assign return value, GEMM TFLOPs measurement, Winograd bounty lock, Rangeify bugs, CUDA 12.0 and sm_35` 


- **辩论 assign() 与 store() 的功能**：有人质疑 `assign()` 是否需要返回一个值，还是应该像 `store()` 一样工作，并思考这是否仅仅是为了方便，因为在示例中返回值通常未被使用。
   - 有人建议*将 buffer 和 store 同时链接到 load* 是一个可能的替代方案。
- **GEMM 165+ TFLOPs 悬赏测量受到质疑**：关于如何在 RTX 4090 上测量 **165+ TFLOP GEMM** 悬赏目标提出了疑问，怀疑在规定的 **2.52 GHz** 加速频率下可能无法实现。
   - 在该时钟频率下，RTX 4090 上使用 FP32 累加的 FP16/BF16 理论峰值吞吐量约为 **165.15 TFLOPs**，但提问者暗示可能需要更高的频率才能达到悬赏目标。
- **Winograd 悬赏要求明确**：一位用户询问了锁定 **Winograd 悬赏** 的要求，他发现了一个识别 Winograd 兼容卷积的*充分必要条件*。
   - George Hotz 澄清说，*只有在代码正确且有修复程序准备合并时才会锁定悬赏*。
- **分享 Rangeify Bug 列表**：分享了一份 **Rangeify bug** 列表供人们调查和修复，并强调其中许多可能只是简单的修复。
   - `RANGEIFY=1` 被描述为*可以创建类似 Flash Attention 及更高版本的新调度器*。
- **CUDA 12.0 停止支持 sm_35**：CUDA 的问题在于 **CUDA 12.0** 停止了对 Ocelot 使用的 **sm_35** 的支持。
   - 最小标志是在 12.4 之后添加的。


  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1416898796943708170)** (12 messages🔥): 

> `tinygrad 中的 GPU 利用率, VIZ=1 分析器, NixOS 的 CUDA 补丁, 分析器 404 错误` 


- **tinygrad 中 GPU 利用率骤降**：一位 tinygrad 用户报告称 **GPU 利用率** 表现不佳，并寻求改进建议，特别是在从 CPU 切换到 CUDA 时。
   - 另一位用户建议使用 `PROFILE=1`（或 `VIZ=1`）来识别时间消耗在哪里，并指出将张量（tensors）保存到磁盘可能会成为瓶颈，提议用户检查分析结果以帮助确定问题根源。
- **`VIZ=1` 分析器统一了分析选项**：`PROFILE=1` 仅仅是 `VIZ=1` 的别名，前者已被移除以减少冗余并精简 tinygrad 中的分析流程。
   - George Hotz 指出 *“有两个选项比只有一个选项更糟糕”*，这是此次更改的动力，旨在简化分析过程。
- **NixOS CUDA 补丁即将到来**：一位 tinygrad 用户在提交了与 **CUDA** 相关的补丁后，计划调查并可能修复其 **NixOS** 发行版上损坏的分析器问题。
   - 该用户提到他们必须修补文件路径，这表明该发行版的包处理 **CUDA** 依赖项的方式可能存在问题。
- **分析器面临 404 错误**：一位 tinygrad 用户在使用分析器时，尝试访问 `/js/index.js` 遇到了 **404 错误**。
   - 此错误表明分析器的文件路径或 `/js/index.js` 的位置可能存在潜在问题。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1416140810352594974)** (19 messages🔥): 

> `积分结转, 每日积分停止发放, 使用 AI 克隆网站, 订阅续订问题, 知识库上限提升` 


- **积分困惑困扰用户**：用户正在询问 **积分结转（credits rollover）** 以及 **每日 300 积分** 分配停止的问题。
   - 一位用户特别报告说，他们的订阅续订原定于 **9 月 14 日**，但既没有被扣费，也没有收到更多积分。
- **网站克隆热潮开启！**：一位用户提到，使用 **Manus** 或其他 **AI 工具** 很容易克隆一个网站。
   - 他对自己于 **8 月 12 日** 提出的功能想法在仅 **16 天后** 就在 [这个 Discord 频道](https://discord.com/channels/1348819876348825620/1349440650495398020/1404845589497122887) 得到实现感到印象深刻。
- **协作增强编程信心**：一位用户正在尝试与朋友进行 **Manus Collaboration** 以完成编程任务。
   - 另一位用户正在开发一个潜在的新功能，如果成功，有望显著提高 **Manus** 作为编程助手的效率。
- **知识库导航需要优化**：几位用户询问如何提高 **知识库上限（knowledge limit）**，特别是是否可以超过 **20** 个。
   - 讨论中没有提供具体的答案。


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1416155480320905297)** (10 messages🔥): 

> `MCP 服务器, 强化学习, 集成测试, MCP 服务器效率, 自然语言接口` 


- **使用 MCP 服务器进行可扩展的集成测试**：成员们正在思考基于 **MCP server** 工具使用的可扩展 **集成测试** 和 **强化学习（Reinforcement Learning）**。
   - 他们考虑在连接到 MCP server 时进行标记，表明该服务器处于某种 **开发或模拟模式**，以便进行稳健的训练，在不干扰生产数据库的情况下模拟真实的工具行为。
- **为 MCP 服务器的效率评分**：一位成员正在研究如何为 **MCP servers** 在不同客户端中的 **效率** 进行评分，以确定何时效率的边际改进不值得在服务器中进行额外的编码。
   - 权衡点在于：是让每种 prompt 共享一个节点，还是让每种 prompt 都有自己的节点——但对于一个用户故事（user story）来说，多少次 **API calls** 才算“太多”？
- **MCP 作为应用程序的 CLI**：一些人正在考虑将 **MCP** 作为应用程序的 **CLI**，以 **自然语言接口（NL interface）** 和 **自适应仪表板/报告** 的形式呈现。
   - 其想法是通过自然语言将其作为企业级应用的 **UI/UX 界面**。
- **Golang 流式 HTTP MCP 服务器项目**：一位成员公开了他们的 [mcp-server-go 项目](https://github.com/ggoodman/mcp-server-go)，这是一个 **Golang 流式 HTTP MCP server**，旨在解决企业级场景中更具挑战性的需求。
   - 它专为 **可扩展性** 而设计，包含 **auth（身份验证）**、**会话与可恢复性** 以及 **动态能力** 等特性。


  

---

### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1417266476141772966)** (2 messages): 

> `MCP 资源与 LLM 的集成、Claude Desktop 自动化、Discord 频道限制` 


- **LLM 学习 MCP 资源**：一位成员询问了如何自动化 LLM 在回答用户问题和执行工具之前读取 **MCP 资源** 的过程，旨在实现 LLM 预加载知识的工作流。
   - 该成员指出，目前在 **Claude Desktop** 中，必须在提问前手动将资源添加到聊天窗口。
- **Claude Desktop 缺乏自动化**：一位成员确认 **Claude Desktop** 的功能符合预期，即需要手动将资源添加到聊天窗口。
   - 他们澄清说，在当前设置中，没有 LLM 在与用户交互之前自动读取资源的流程。
- **Discord 关注点收窄**：会议明确了该 Discord 频道仅限于讨论 **MCP 协议本身** 的治理。
   - 一般性的 **MCP 问题** 应导向其他地方，并为有需要的人提供 DM 指导。


  

---


---


---