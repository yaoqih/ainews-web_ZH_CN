---
companies:
- alibaba
- openai
- meta-ai-fair
- huggingface
- anthropic
- microsoft
- github
date: '2025-09-24T05:44:39.731046Z'
description: '以下是该文本的中文翻译：


  **阿里巴巴**发布了 **Qwen3** 模型家族，包括 **Qwen3-Max** 和 **Qwen3-VL**。该系列原生支持 256K 上下文窗口（可扩展至
  100 万），具备 32 种语言的强大 OCR 能力。在 520 亿美元基础设施路线图的支持下，其发布速度极快（约每月发布 3.5 次）。**OpenAI**
  推出了 **GPT-5 Codex**，这是一款针对智能体（Agent）优化的编程模型，拥有高达 **400K 上下文**和自适应推理能力，定价为每百万 Token
  1.25 美元（输入）/ 10 美元（输出），目前已集成到 Cline 中并在 WebDev 评测中进行了基准测试。**Meta AI FAIR** 发布了开放权重的
  **Code World Model (CWM) 32B**，这是一款性能强劲的稠密代码生成模型（例如：SWE-bench Verified 准确率 65.8%，Math-500
  准确率 96.6%），并附带了公共安全报告。生态系统更新方面：GitHub Copilot 推出了新的嵌入模型以实现更快的代码搜索；Anthropic 的 Claude
  Sonnet 4 和 Opus 4.1 已集成到 Microsoft 365 Copilot 中。vLLM 0.10.2 版本更新引入了解码上下文并行（DCP），旨在提升系统性能。'
id: MjAyNS0w
models:
- qwen3-max
- qwen3-vl
- qwen3-coder-plus
- gpt-5-codex
- code-world-model-32b
- claude-sonnet-4
- claude-opus-4.1
people:
- huybery
- akhaliq
- lmarena_ai
- gdb
- ylecun
- pierceboggan
- julesagent
title: 今天没什么事。
topics:
- context-windows
- code-generation
- model-releases
- model-benchmarking
- api
- model-optimization
- multimodality
- software-engineering
- model-training
---

**平静的一天**

> 2025年9月24日至9月25日的 AI 新闻。我们为您查看了 12 个 subreddits、544 个 Twitter 账号和 23 个 Discord 社区（包含 194 个频道和 2885 条消息）。预计节省阅读时间（按每分钟 200 字计算）：230 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以精美的 vibe coded 方式呈现所有往期内容。访问 https://news.smol.ai/ 查看完整的详细新闻，并在 @smol_ai 上向我们提供反馈！

您可以在这里观看 [**AIE Paris 第二天的内容**](https://www.youtube.com/watch?v=wyUdpmj9-64)，会上宣布了 AIE Europe 2026 的门票信息。您还应该[**申请 11 月在纽约举行的 AIE CODE 第二波活动**](https://apply.ai.engineer/) —— 这将是一场盛会。

---

# AI Twitter 回顾

**阿里巴巴 Qwen3 攻势：Max、VL、Coder 以及 520 亿美元的路线图**

- **Qwen3-Max、Qwen3-VL 与发布速度**：阿里巴巴/通义发布了一系列模型：旗舰级 **Qwen3-Max**（现已成为 Anycoder 的默认模型）和开源的 **Qwen3-VL**，具有原生 256K 上下文（可扩展至 1M）、更强的 32 种语言 OCR 能力、2 小时视频中的精确事件定位、GUI 操作/编码以及领先的风险检测能力。发布内容已上线 Hugging Face、ModelScope、GitHub 和阿里云百炼（Model Studio）；社区平台迅速跟进（例如 Yupp 增加了 Qwen3 Max 和 Qwen3 VL 235B A22B Instruct/Thinking；LMArena 增加了三个 Qwen3 模型）。阿里巴巴宣称其发布速度无与伦比（约每月 3.5 次发布，且多为开源权重），并在云栖大会上讨论了多年基础设施路线图，评论指出其拥有“520 亿美元的战备资金”和重大的算力扩容计划。查看公告和推文：[@huybery](https://twitter.com/huybery/status/1970649341582024953), [@huybery 关于 Qwen3-VL](https://twitter.com/huybery/status/1970650821747712209), [@Ali_TongyiLab (VL 发布)](https://twitter.com/Ali_TongyiLab/status/1970665194390220864), [Anycoder 默认设置](https://twitter.com/_akhaliq/status/1970618469344235677), [Yupp 增加 Qwen 模型](https://twitter.com/yupp_ai/status/1970640795259851079), [LMArena 增加 Qwen3](https://twitter.com/lmarena_ai/status/1970920636957831611), [发布速度](https://twitter.com/awnihannun/status/1970839682503348623), [云栖大会回顾](https://twitter.com/Smol_AI/status/1970842828512088486), [高管剪辑/路线图](https://twitter.com/swyx/status/1970847377058676849)。
- **Qwen3-Coder-Plus 与 API 改进**：Coding 系列获得了针对性升级（终端任务处理、脚手架适配；API 修复），在 WebDev Arena 和 Agent 工具链中展现出早期竞争信号。详情：[API 更新](https://twitter.com/huybery/status/1970652792848293926), [WebDev Arena 提示词](https://twitter.com/lmarena_ai/status/1970962780225507775)。

**编程模型与 Agent：GPT-5 Codex 落地；Meta 发布 32B CWM**

- **GPT-5 Codex（针对 Agent 优化）已上线**：OpenAI 的 “Codex” 变体已进入 API 和 Agent 工具中。亮点：高达 **400K 上下文**，“自适应推理”具备可变思考能力，在简单任务上消耗更少 Token，在复杂任务上消耗更多，定价约为 **每百万 Token $1.25/$10**。它已集成到 Cline 中（带有“思考滑块”），并正在 WebDev Arena 和 Agent 工作流中进行基准测试。链接：[API 可用性](https://twitter.com/gdb/status/1970631954887565823), [Cline 集成](https://twitter.com/cline/status/1970619799119241709), [Cline 详情](https://twitter.com/cline/status/1970619811853148550), [WebDev Arena](https://twitter.com/lmarena_ai/status/1970962780225507775)。实测报告对比了其与 Sonnet/GPT-5 在长上下文和 Agent 运行时的吞吐量：[示例](https://twitter.com/zachtratar/status/1970625784500130065), [长上下文检索对比](https://twitter.com/scaling01/status/1970661469667660100)。
- **Meta FAIR 的 Code World Model (CWM) 32B（研究阶段）**：Meta 发布了一个采用研究许可证的开源权重 32B 稠密模型，将代码生成定义为使用代码执行世界模型进行的规划。报告的 pass@1 数据：**65.8% SWE-bench Verified**、**68.6% LiveCodeBench**、**96.6% Math-500**、**76.0% AIME 2024**。技术报告、权重和代码已公开，并附带来自 SEAL/AI Security 的安全准备报告。链接：[@AIatMeta](https://twitter.com/AIatMeta/status/1970963571753222319), [@ylecun](https://twitter.com/ylecun/status/1970967341052854748), [指标摘要](https://twitter.com/alexandr_wang/status/1970973317227225433), [安全准备](https://twitter.com/summeryue0/status/1970971944557346851)。
- **生态系统更新**：GitHub Copilot 发布了新的嵌入模型和训练报告（用于更快、更准确的代码搜索）[博客链接](https://twitter.com/pierceboggan/status/1970950784251724007)；Jules Agent 现在可以根据 PR 反馈采取行动 [链接](https://twitter.com/julesagent/status/1970640318606258605)；Claude Sonnet 4 和 Opus 4.1 现已加入 Microsoft 365 Copilot [Anthropic](https://twitter.com/AnthropicAI/status/1970907112831328296)。

**系统与基础设施：vLLM DCP、多模态数据管道以及平台动态**

- **vLLM 0.10.2 新增 Decode Context Parallel (DCP)**：由 Kimi/Moonshot 贡献，DCP 在 GPU 之间对 KV cache 进行分片以减少重复，在单节点 H200 上可实现高达 **8 倍的 KV 容量**和 **2–3 倍的吞吐量**——对于 KV 密集型工作负载（RL、离线数据生成）特别有帮助。快速上手：`vllm serve deepseek-ai/DeepSeek-V3.1-Terminus -tp 8 -dcp 8`。相关链接：[@vllm_project](https://twitter.com/vllm_project/status/1970814441718755685), [day-0 指南](https://twitter.com/rogerw0108/status/1970619149757096037)。
- **来自 Perceptron 的多模态基础设施**：团队分享了 TensorStream 背后的设计——这是一种用于驱动其训练/推理代码的交错多模态数据的类张量抽象——并发布了 Isaac 0.1 的技术细节，这是一个强调简单训练配方和鲁棒 Grounding 的小型 VLM。关于“复杂度预算”和原生多模态抽象的精彩讨论：[设计文章](https://twitter.com/perceptroninc/status/1970670362355736886), [Isaac 报告](https://twitter.com/perceptroninc/status/1970701029441483087), [评论](https://twitter.com/kilian_maciej/status/1970701658494738514), [抽象层 +1](https://twitter.com/ArmenAgha/status/1970672682909016242)。
- **MCP 构建者与合规性**：Figma 的 MCP 服务器登陆 VS Code（并可在 OpenHands 中使用），用于“从设计到代码”的工作流 [VS Code](https://twitter.com/code/status/1970621943821861217), [OpenHands](https://twitter.com/allhands_ai/status/1970955961293795831)；Weaviate 获得 ISO 27001 认证 [链接](https://twitter.com/weaviate_io/status/1970912361381843104)；AMD 扩大与 Cohere 的合作伙伴关系（模型运行在 AMD Instinct 上，支持主权 AI 态势）[AMD](https://twitter.com/AMD/status/1970824479279317446)；Modular 融资 **2.5 亿美元**以推进其统一的 AI 基础设施平台 [Modular](https://twitter.com/Modular/status/1970881293933273524)。

**视频与多模态生成：Alibaba Wan2.5, Runway A2D, NVIDIA Lyra, Kling 2.5**

- **Alibaba Wan2.5-Preview（原生多模态）**：新架构通过联合多模态训练和 RLHF 原生地对齐文本、图像、视频和音频；支持可控输入（文本/图像/音频）、同步多发言者音视频、1080p 10秒电影级视频，以及更强的图像生成/编辑能力（排版、图表、像素级编辑）。[公告](https://twitter.com/Alibaba_Wan/status/1970697244740591917)。
- **Runway A2D：自回归到扩散 VLM**：将现有的 AR VLMs 适配为并行扩散解码，以在不从头训练的情况下实现速度与质量的权衡；来自实习工作的开发预览版展示了通往视觉语言扩散 LM 的实用路径。[@runwayml](https://twitter.com/runwayml/status/1970866494729781623), [作者推文](https://twitter.com/mariannearr/status/1970936677922382335)。
- **NVIDIA Lyra（3D/4D 场景重建）**：通过视频扩散自蒸馏，从单张图像/视频实现前馈 3D 和 4D 场景生成；权重已在 HF 上发布。[概览](https://twitter.com/_akhaliq/status/1970949464606245139), [模型](https://twitter.com/_akhaliq/status/1970949559426961484)。
- **Kling 2.5 Turbo**：内部盲测显示，在文生视频和图生视频方面，相较于 Seedance/Veo 变体有显著优势；社区短片和比赛正在推出。[结果](https://twitter.com/Kling_ai/status/1970832920085753893), [比赛](https://twitter.com/Kling_ai/status/1970783972033445965)。

**推理、RL 与评估科学**

- **RLPT (RL on Pre-Training Data)**：直接在预训练语料库上通过下一段推理（ASR+MSR）进行自监督奖励训练，无需人工标签。在 Qwen3‑4B 上，报告的提升包括：**+3.0 MMLU**、**+8.1 GPQA‑Diamond**、**+6.6 AIME24**、**+5.3 AIME25**。论文：[tweet](https://twitter.com/arankomatsuzaki/status/1970684035258294548), [arXiv](https://twitter.com/arankomatsuzaki/status/1970684037787492416)。
- **APRIL (Active Partial Rollouts in RL)**：削减了 rollout 的长尾低效问题；在 GRPO/DAPO/GSPO 中实现了高达 **44%** 的吞吐量提升和 **8%** 的最终准确率提升。[tweet](https://twitter.com/iScienceLuvr/status/1970794655270003037), [code/paper](https://twitter.com/iScienceLuvr/status/1970794659661434895)。
- **“Soft Tokens, Hard Truths”**：首个用于连续 CoT 的可扩展 RL；soft-token 训练在 pass@1 上与离散方式持平，并因增强了多样性而在 pass@32 上表现更优；最佳实践：软训练，硬推理。[tweet](https://twitter.com/arankomatsuzaki/status/1970692910766346277), [arXiv](https://twitter.com/arankomatsuzaki/status/1970692913119277178)。
- **有效推理 ≠ 更长的 CoTs**：在 10 个 LRMs 中，更长的链条和审查可能与更低的准确率相关。新指标“Failed-Step Fraction”（失败步骤比例）可预测正确性；基于 FSF 的重排序将 pass@1 提升了高达 **+10%**。[tweet](https://twitter.com/arankomatsuzaki/status/1970691075229864357), [arXiv](https://twitter.com/arankomatsuzaki/status/1970691077683454053)。
- **医疗多模态的脆弱性**：压力测试显示，前沿模型在没有图像的情况下经常能猜对，在微小的 prompt 更改下会发生结果翻转，并编造出令人信服但有缺陷的推理——排行榜掩盖了这种脆弱性。[tweet](https://twitter.com/arankomatsuzaki/status/1970684893966516477), [arXiv](https://twitter.com/arankomatsuzaki/status/1970684896160239984)。
- 相关：Google 的 Test-Time Diffusion Deep Researcher (TTD-DR) 将扩散式迭代细化应用于长篇研究，报告在某些任务上对比 OpenAI Deep Research 拥有高达 **74.5%** 的胜率，且具有更好的质量-延迟权衡。[overview](https://twitter.com/omarsar0/status/1970864565710921891)。

**热门推文（按互动量排序）**

- [阿里巴巴 Wan2.5-Preview：原生多模态音视频生成与编辑](https://twitter.com/Alibaba_Wan/status/1970697244740591917) — 1453
- [Qwen3‑VL 开源：256K→1M 上下文，32 语种 OCR，精确视频事件定位](https://twitter.com/Ali_TongyiLab/status/1970665194390220864) — 1410.5
- [Sam Altman 关于 Abilene 数据中心建设进展的推文](https://twitter.com/sama/status/1970812956733739422) — 9917
- [半导体节点名称（“3nm”、“2nm”）只是营销简称，而非实际尺寸](https://twitter.com/giffmana/status/1970620746155393441) — 9032.5
- [Claude Sonnet 4 和 Opus 4.1 登陆 Microsoft 365 Copilot](https://twitter.com/AnthropicAI/status/1970907112831328296) — 1265
- [Gemini 应用在不到 1 个月内生成了 50 亿张图像](https://twitter.com/joshwoodward/status/1970894369562796420) — 1183
- [GPT‑5 可以解决“次要”的开放数学问题；早期证据和预印本](https://twitter.com/SebastienBubeck/status/1970875019803910478) — 952

---

# AI Reddit 综述

## /r/LocalLlama + /r/localLLM 综述

### 1. MiniModel-200M 与 DeepSeek-V3.1-Terminus 本地发布基准测试

- [**MiniModel-200M-Base**](https://i.redd.it/clbzeq0i82rf1.png) ([评分: 223, 评论: 35](https://www.reddit.com/r/LocalLLaMA/comments/1np5ey8/minimodel200mbase/)): **MiniModel-200M-Base 是一个参数量约 200M 的 LLM，在单块 RTX 5090 上，通过约** `110k` **步（约 1 天）从零开始基于** `10B` **tokens 训练而成。该模型未采用梯度累积（gradient accumulation），实现了** `64×2048` **的有效 Batch Size，且峰值 VRAM** `<30 GB`**。其效率归功于 Adaptive Muon 优化器（据称数据效率比 AdamW 高出约** `2.1×`**）、Float8 预训练（Attention 采用 bf16）使 VRAM 降低约** `30%` **且吞吐量提升约** `20%`**、ReLU² (Primer)、通过装箱（bin-packing）将 Padding 从** `>70%` **降低至** `<5%`**，以及为了稳定性采用的无标量 QK-norm 全注意力机制。早期能力演示包括确定性的斐波那契代码生成和记忆 π 的前 20+ 位数字；目前已发布 Apache-2.0 权重的配置与 Tokenizer：[Hugging Face](https://huggingface.co/xTimeCrystal/MiniModel-200M-Base)。** 热门评论主要集中在请求发布训练代码/脚本以及更多关于数据混合（data mixture）的细节；关注点在于该方案的可复现性。
    - 一位评论者对强调“无梯度累积”提出质疑，认为这在数学上应等同于更大的有效 Batch Size。他们指出了可能导致差异的实际因素：优化器步数耦合（例如 AdamW 偏置校正、每步权重衰减）、与步数而非 Token 挂钩的 LR 调度、跨微批次的梯度裁剪以及随机元素（Dropout RNG、数据顺序）。他们实际上是在询问本次训练运行中避免梯度累积的具体理由或收益（例如吞吐量/激活内存权衡、基准测试公平性）。
    - 多个请求希望发布训练代码和脚本以实现可复现性。隐含的需求是端到端流水线（数据加载器、Tokenizer、优化器/调度器配置、日志/检查点保存）和确切的种子，以便他人能在 200M 参数设置上复制结果并与基准模型进行对比。
    - 对数据混合细节的关注：评论者想了解数据组成和混合策略（如代码/数学/对话的领域比例、加权提升/降低、去重/过滤以及总预训练 Token 数）。鉴于小模型对数据精选（data curation）的敏感性，他们要求提供精确配方，以理解 MiniModel-200M-Base 表现如报告般出色的原因。
- [**你现在可以在本地设备上运行 DeepSeek-V3.1-Terminus 了！**](https://i.redd.it/nntm711d61rf1.png) ([评分: 163, 评论: 29](https://www.reddit.com/r/LocalLLM/comments/1np1o9e/you_can_now_run_deepseekv31terminus_on_your_local/)): **Unsloth 发布了 DeepSeek-V3.1 Terminus 的动态 GGUF 量化版本，通过逐层“智能” 1-bit 量化，将原始约 715 GB 的模型缩小了约 80%，从而支持在约 170 GB RAM 上进行本地推理（以及一个约 162 GB 的 Ollama 就绪版本）。其动态 3-bit DeepSeek-V3.1 (thinking) GGUF 在 Aider Polyglot 基准测试中得分** `75.6%`**——据报道超过了 Claude-4-Opus (thinking)——可通过 llama.cpp 运行，并提供了一个示例 Ollama 标签** `hf.co/unsloth/DeepSeek-V3.1-Terminus-GGUF:TQ1_0`**；资源：[博客文章](https://docs.unsloth.ai/new/unsloth-dynamic-ggufs-on-aider-polyglot), [HF 仓库](https://huggingface.co/unsloth/DeepSeek-V3.1-Terminus-GGUF), [指南](https://docs.unsloth.ai/basics/deepseek-v3.1)。图片展示了动态 GGUF 性能与基准模型及闭源模型的对比图表。** 热门评论质疑其对家庭用户的实用性——询问类似方法是否能将 70B–200B 模型压缩以适配 16–24 GB VRAM 的 GPU——而其他人则注意到了高 VRAM/RAM 需求并给予了称赞。
    - 一个核心问题是：相同的方法能否让 `70B` 或 `100–200B` 模型在 `16–24 GB` 的消费级 GPU 上运行。这意味着需要极端的量化/卸载（offloading）来适配 VRAM，家庭用户的实际效用取决于此。
    - 一位评论者提到内存占用从 `715 GB` 降至 `170 GB` 的同时保持了“可靠的工具调用（tool-calling）能力”。他们希望看到与 **GLM-4.5** 和 **Qwen** 的正面交锋，建议在工具使用/Agent 基准测试上进行评估，以验证质量与压缩之间的平衡。
    - 即使有所缩减，实际部署可能仍需要约 `100 GB` 的 VRAM（*“现在得再去搞 100 GB 的 VRAM”*）。这超出了典型的 `16–24 GB` 游戏显卡，突显了本地使用的硬件障碍依然存在。

### 2. DIY 本地 AI 硬件：RTX 3080 20GB 改装与 Ryzen AI MAX+ 395

- [**我的第二块来自中国的改装版 3080 20GB，用于本地 AI 推理、视频和图像生成...**](https://www.reddit.com/gallery/1np9rav) ([Score: 219, Comments: 101](https://www.reddit.com/r/LocalLLaMA/comments/1np9rav/my_second_modified_3080_20gb_from_china_for_local/)): **原帖作者（OP）展示了一款在中国改装的 GeForce RTX 3080，显存升级至 20 GB VRAM（可能是在 320‑bit 总线上使用了 10×16Gb GDDR6X），用于本地 AI 推理/视频/图像工作负载。为了静音效果，他选择了三风扇散热器而非涡轮风扇（blower）。据报道，这款 2.5 槽位的显卡在约 `300W` 压力测试下温度保持在 `75°C` 以下，表明其散热余量优于涡轮版本；除此之外，它的表现与标准 [RTX 3080](https://en.wikipedia.org/wiki/GeForce_30_series#GeForce_RTX_3080) 一致。** 评论者探讨了其相对于 [RTX 3090](https://en.wikipedia.org/wiki/GeForce_30_series#GeForce_RTX_3090)（更多核心，`24 GB` VRAM）的价值，并询问了 20 GB 改装版的驱动/VBIOS 兼容性。有人对使用 `3 GB` GDDR6X 颗粒实现假设的 `30 GB` 3080 表示好奇；由于 GA102 内存控制器/电路板布线对 24Gb 密度的支持尚不明确，其可行性未知（参见 [GDDR6X](https://en.wikipedia.org/wiki/GDDR6#GDDR6X)）。
    - 与 RTX 3090 的价值/性能权衡：3080 20GB 改装版仍采用 320‑bit 总线（约 `760 GB/s`），且 SM 单元少于 3090 的 384‑bit 总线（约 `936 GB/s`）。因此，对于既敏感于带宽又敏感于 VRAM 的 AI/图像工作负载，3090 的 `24GB` 和更宽的总线速度明显更快，并允许更大的 batch sizes/checkpoints。鉴于二手 3090 的价格通常在 `$500` 左右，评论者认为 `$500` 的 3080‑20GB 很难物有所值，除非价格接近 `$350`——否则 3090（或即将推出的 24GB 下一代选项）是更好的选择。规格参考：[RTX 3080](https://www.techpowerup.com/gpu-specs/geforce-rtx-3080.c3621), [RTX 3090](https://www.techpowerup.com/gpu-specs/geforce-rtx-3090.c3627)。
    - 使用 3GB (24Gb) GDDR6X 实现 30GB 3080 的可行性：理论上，10× `24Gb` 颗粒可以在 320‑bit 的 GA102 上实现 `30GB`，但这取决于 GA102 的内存控制器/BIOS 是否支持 24Gb 密度和正确的 timing straps——目前没有零售版 GA102 板卡搭载 24Gb 设备，因此兼容性未经证实。即使 VBIOS 能识别，如果没有 AIB 级别的固件支持，稳定性/散热和内存训练（memory training）也可能出现问题。Micron 已经展示了 `24Gb` GDDR6X 晶圆样品，这使得该容量在理论上是可能的：[Micron 24Gb GDDR6X](https://www.micron.com/about/news-and-events/releases/2022/07-14-micron-samples-worlds-first-24gb-gddr6x-memory)。
    - 20GB 改装版的驱动/VBIOS 注意事项：NVIDIA 驱动程序从 VBIOS 中枚举 VRAM；只要设备 ID 匹配且 BIOS 包含针对所安装 GDDR6X 密度的正确 memory straps，官方驱动通常可以工作。许多中国市场的 20GB 板卡发货时带有能正确报告 `20GB` 的定制 VBIOS；刷入不匹配的 BIOS 会导致不稳定或变砖，且 Ampere 架构的 BIOS 编辑受到限制，因此获取厂商匹配的 20GB VBIOS 是关键。参考：[TechPowerUp VBIOS collection](https://www.techpowerup.com/vgabios/)。

- [**Ryzen AI MAX+ 395 是真正的独角兽（褒义）**](https://www.reddit.com/r/LocalLLaMA/comments/1nozz23/the_ryzen_ai_max_395_is_a_true_unicorn_in_a_good/) ([Score: 218, Comments: 205](https://www.reddit.com/r/LocalLLaMA/comments/1nozz23/the_ryzen_ai_max_395_is_a_true_unicorn_in_a_good/)): **发帖者评估了 128 GB [Framework Desktop Mainboard (AMD Ryzen AI Max 300 series)](https://frame.work/products/framework-desktop-mainboard-amd-ryzen-ai-max-300-series?v=FRAFMK0006) 用于本地 AI 推理的性价比，并与规格相似的 DIY 桌面电脑进行了对比。一份可比的 DIY 零件清单（寻求 4 通道 DDR5 ≥8000 MT/s）总计约**`$2240`**：消费级 4 通道 DDR5 主板** `>$600`**，通过 [Ryzen 9950X3D](https://www.amazon.com/AMD-Ryzen-9950X3D-16-Core-Processor/dp/B0DVZSG8D5)** `~$660` **实现的 CPU “等效”性能 + [Noctua NH‑D15](https://www.amazon.com/Noctua-NH-D15-heatpipe-NF-A15-140mm/dp/B00L7UZMAK)** `~$130`**, 128 GB [DDR5‑8000 (4×24 GB)](https://www.amazon.com/G-SKILL-Trident-CL38-48-48-128-Desktop-Computer/dp/B0F4M6C65N)** `~$450`**, 以及与该板载 iGPU 性能“相似”的 dGPU（RTX [4060/4060 Ti 16 GB](https://www.amazon.com/MSI-Gaming-GeForce-GDRR6-Boost/dp/B0D3KGNMXP))** `~$400`**。发帖者认为 Framework 主板的 unified memory 避免了 GPU 访问大型模型权重时的 PCIe 带宽/延迟惩罚，且分体式构建的功耗将达到 ≳2 倍（产生更多热量/噪音；参见 [房间加热帖子](https://www.reddit.com/r/LocalLLaMA/comments/1nogrv2/computer_literally_warms_my_room_by_5_degrees/)）。他们补充说，Apple M4 Pro/Max 带宽更高，但在相似的 RAM/GPU 配置下，其 Diffusion 吞吐量较低，且成本约为 2 倍；而真正高吞吐量的 Nvidia 配置（如 4× RTX 3090）则昂贵得多且功耗巨大；注：引用的 9955HX3D 不支持 4 通道内存——Threadripper 支持，但内存速度较慢。** 热门回复要求提供具体的基准测试（“数据”），并建议如果 AMD 推出 256 GB unified memory，可能会产生阶跃式影响。一位评论者建议在相同预算下为 Diffusion 工作负载（VRAM > 系统 RAM）选择 RTX 5080，同时也同意对于 LLM 而言，更大的 unified memory（128 GB+）对更大的上下文和模型占用更有利。
    - 工作负载适配与内存 vs 吞吐量的权衡：评论者指出，对于 Diffusion/视觉工作负载，RTX 5080 级别的 GPU 在相似价位下表现更好，且处理图像/视频不需要 `128GB` RAM。对于 LLM，更大的系统/unified memory 更有价值（可容纳更大的模型/上下文），这符合“卡车（容量）vs 跑车（吞吐量）”的比喻；一个假设的 `256GB` unified memory SKU 被视为 LLM 用例的市场颠覆者。
    - 带宽瓶颈担忧：一位用户指出“< `256 Gb/s` 内存带宽”，暗示其具备大上下文能力但推理较慢，特别是在 LLM 受内存带宽限制的 Prefill 阶段。Unified memory 有助于承载更大的上下文，但有限的带宽可能会在 Prefill 期间限制 Token/s，因此该设备可能仅在 KV cache 预热后的生成阶段感觉响应迅速。
    - 与高端 GPU 的轶事性能对比：一位拥有 RTX 5090 + `96GB` RAM（比 Ryzen AI Max 贵约 +$1k）的用户报告称，在 `gpt-oss-120B` 上，Token 生成（TG）速度大致相当，但 Prefill（PP）在 5090 上快 `4–15×`。结论：对于以 Prefill 为主的本地 LLM，尽管 TG 吞吐量相当，Ryzen 设备的表现可能不如顶级 GPU。

### 3. LLM 性能增长主张与炒作反应

- [**大语言模型性能每 7 个月翻一番**](https://spectrum.ieee.org/large-language-model-performance) ([Score: 152, Comments: 57](https://www.reddit.com/r/LocalLLaMA/comments/1np2v1i/large_language_model_performance_doubles_every_7/)): **帖子断言了一个经验性的“AI 摩尔定律”，即大语言模型的能力大约每** `~7` **个月翻一番，并通过进度图表（[图片](https://preview.redd.it/kysbjgxyr1rf1.png?width=461&format=png&auto=webp&s=74d948ee7a2545582e12175877ff315b071aa0fd)）进行了说明，将其描述为基准测试性能的持续指数级增长。这一说法呼应了之前关于 AI 进步加速的解释，例如 Computerphile 对 AI 版摩尔定律的概述（[视频](https://www.youtube.com/watch?v=evSFeqTZdqs&t=1s)）；帖子本身并未详细说明方法论或汇总了哪些基准测试。** 评论者强调成本随着质量的提升而下降（Token/模型定价下降），并将价格压力归功于开源竞争；其他人则认为这一观察并不新鲜，并指向了早期的报道，如 Computerphile 的视频。

- 对该图表的方法论批评：它似乎将 LLM 能力转化为“完成任务的人类时间”，并对每个任务使用 `50%` 的成功阈值，这具有高度主观性且取决于具体任务。提出的例子：“在网上查找事实”的时间范围可以从几秒到几天不等，具体取决于精确度；“为定制芯片优化代码”定义不明确，可能跨越数小时到数月；而“创办一家新公司”耗时 `167h` 并不是一个有意义、可衡量的单位。如果没有标准化的 Benchmark 和精确的任务规范，像“每 7 个月翻倍”这样的说法存在樱桃拾取（cherry-picking）和误导真实进度的风险。
- 成本/性能动态：评论者注意到在能力提升的同时推理成本在下降，开源模型加剧了价格竞争。从业者仍依赖 2024–2025 年的开源模型，如 **Mistral**、**Llama 3.1** 和 **Qwen 2.5 Coder**，这暗示感知的改进取决于任务和部署方式；成本/性能权衡（例如，本地推理 vs API）、稳定性和工具链可能比头条上的“翻倍”指标更重要。同时报告能力和 $/token 或 $/task 将能更好地捕捉现实世界的价值。
- 关于 Scaling 的先前研究：链接的 Computerphile 视频《AI’s Version of Moore’s Law?》(https://www.youtube.com/watch?v=evSFeqTZdqs&t=1s) 回顾了 LLM 的 Scaling 趋势，并将硬件驱动的 FLOPs/$ 收益与算法效率改进区分开来，两者共同创造了表象上的能力翻倍。它将进步归因于更大的模型、更好的训练数据/配方以及推理优化，并警告不要将单一的“翻倍周期”视为所有任务的通用标准。
- [**天哪，这是什么怪物？**](https://i.redd.it/1pxmwf50e2rf1.jpeg) ([Score: 590, Comments: 124](https://www.reddit.com/r/LocalLLaMA/comments/1np5te1/oh_my_god_what_a_monster_is_this/))：**该图像（[图表](https://i.redd.it/1pxmwf50e2rf1.jpeg)）似乎是一个 Benchmark 排行榜，其中多个 LLM 在某项任务上达到了接近或恰好** `100` **分，这表明评估已趋于饱和/触及天花板，无法再区分顶级模型。评论者指出，中国的 Frontier 模型处于或接近图表顶部，这意味着其性能已与领先的西方模型持平。** 值得注意的观点：“如果模型得分是 100，那么这个 Benchmark 就没用了”，认为该指标已失去区分度；其他人则强调中国模型已达到 Frontier 水平，而有人批评将方形图表的纵向模式截图导致可读性差。
    - Benchmark 饱和担忧：如果模型达到 `100`，则表明存在天花板效应和微弱的区分能力。这增加了过拟合/测试集污染的风险，并促使社区转向更难或具有对抗性的测试集，如 **MMLU-Pro** 和 **GPQA**，以及鲁棒性/长上下文评估，而不是仅仅依赖经典的 **MMLU**、**GSM8K** 或 **HumanEval**。参见 MMLU [论文](https://arxiv.org/abs/2009.03300)、MMLU-Pro [论文](https://arxiv.org/abs/2406.01574)、GPQA [论文](https://arxiv.org/abs/2311.12022)。
    - 多位评论者指出，展示的 Qwen 结果并非“本地”运行，这一点很重要，因为 API 托管的模型可能与可下载的权重以及量化后的本地性能有所不同。设备端限制（VRAM、吞吐量）和量化（例如 `Q4_K_M`）通常会在推理/代码 Benchmark 上损失 `~1–5` 分并改变延迟；例如，在 Q4 下运行 `7B` 需要约 `5–6 GB` VRAM，`14B` 约 `9–10 GB`，`32B` 约 `20–24 GB` ([llama.cpp quantization](https://github.com/ggerganov/llama.cpp/blob/master/docs/quantization.md))。
    - 中国模型已达到 Frontier 水平的说法与近期报告一致：**Qwen2.5**、**DeepSeek‑V2** 和 **Yi** 系列发布的 MMLU/GSM8K/MT‑Bench 和编程得分与已确立的 Frontier 模型相比具有竞争力。参见 Qwen2.5 [博客](https://qwenlm.github.io/blog/qwen2.5/)、DeepSeek‑V2 [论文](https://arxiv.org/abs/2405.04434) 以及 Hugging Face 上的 Yi 模型 ([Yi‑34B](https://huggingface.co/01-ai/Yi-34B))；确切排名取决于评估设置（Prompting、CoT、解码）以及测试是否经过污染控制。

## 较少技术性的 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Qwen Image Edit 2509 发布 Benchmark 与工作流

- [**原版 Qwen Image Edit 与新版 2509 版本的快速对比**](https://www.reddit.com/gallery/1nox9bi) ([评分: 580, 评论: 74](https://www.reddit.com/r/StableDiffusion/comments/1nox9bi/quick_comparison_between_original_qwen_image_edit/)): **原版 Qwen Image Edit 与新版 “2509” 构建版本的并排测试，两者均量化为** `Q5_K_M` **[GGUF](https://github.com/ggerganov/llama.cpp/blob/master/docs/gguf.md) 并在默认 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 中运行，其中 2509 模型需要 "QwenImageEditPlus" text encoder 才能正常运行。使用首样输出（无 LoRAs），2509 版本在保持源风格和构图方面明显更加一致；遗留问题包括在表情编辑期间轻微的全身比例偏移以及眼镜蓝色调的丢失（原版有时会完全丢失眼镜）。更新后的 text encoder 还带来了约 **`5–10%` ** 的速度提升。[示例图片](https://preview.redd.it/6vbfk01cs1rf1.png?width=1030&format=png&auto=webp&s=e8c0ff1dac9266fbb30d4b27c82c6cdc14445344)。** 评论在很大程度上证实了 2509 版本在一致性和感知质量上的提升；没有提出实质性的反对意见。
    - 多位用户报告 Qwen Image Edit 2509 版本较原版有明显的质量提升，其中一位分享了一个编辑示例（“现在真的很棒了……”），表明其 Prompt 遵循更可靠且输出更干净。示例图片：https://preview.redd.it/6vbfk01cs1rf1.png?width=1030&format=png&auto=webp&s=e8c0ff1dac9266fbb30d4b27c82c6cdc14445344
    - 针对 “new text encoder” 提出了技术澄清请求：这是否意味着更换了不同的 encoder 模型（例如，改变了影响 tokenization/conditioning 的 CLIP/ViT 变体），还是仅仅更新了 pipeline/graph 中的 encoder 节点。这种区别会影响可复现性、与现有 workflow 的兼容性以及 prompt-conditioning 行为的潜在变化。
- [**将 QWEN IMAGE 生成的单张源图像转化为动态宽屏视频概念 (WAN 2.2 FLF)，并使用新版 (QWEN EDIT 2509) 进行微调。**](https://v.redd.it/cppv3vn0j4rf1) ([评分: 304, 评论: 49](https://www.reddit.com/r/StableDiffusion/comments/1npdutw/qwen_image_gen_as_single_source_image_to_a/)): **创作者展示了一个 ComfyUI pipeline，利用 “WAN 2.2 FLF” workflow 将单张 Qwen 生成的图像转化为动态宽屏视频，并通过 “QWEN 2509 EDIT” 进行微调。强调了资产和可复现性：在 CivitAI 上提供了一个自定义 LoRA ([链接](http://civitai.com/models/1955327))，提供了 Qwen Image ([pastebin](http://pastebin.com/d9zKTL0T))、WAN 2.2 FLF ([pastebin](http://pastebin.com/hPLdGbAZ)) 和 QWEN 2509 EDIT ([pastebin](http://pastebin.com/zV7zXdSb)) 的完整 workflows，此外还有一个包含所有视频片段/备选方案、图像部分、MP3 音乐、.pdn 编辑文件以及每个阶段 prompt 的 ZIP 压缩包 ([Drive](http://drive.google.com/file/d/1D5RIafNr0U66zzlWaxjqci2YJTiZ2SsY/view?usp=sharing))。X 上有镜像 ([帖子](http://x.com/unmortan/status/1970858115819270363))，并分享了原始 Qwen 图像/prompt（具有明确构图/服装约束的暗黑幻想动漫风格）([预览](https://preview.redd.it/yqsswd9ej4rf1.png?width=1536&format=png&auto=webp&s=c6a31fe39f99a0dd70ce2bd45a4e83ba08fea05d))。** 热门评论强调了单图转视频的实验，并且所有步骤都在 ComfyUI 中执行；一位评论者询问了所需的硬件规格（帖子中未提供配置）。
    - OP 概述了一个仅限 ComfyUI 的 pipeline，通过 **WAN** `2.2` **FLF** 将单张 **Qwen Image** 静止图像动画化为动态宽屏视频，并使用 **QWEN** `2509` **EDIT** 进行微调。他们提供了完整的可复现性：一个 LoRA ([civitai.com/models/1955327](http://civitai.com/models/1955327))，所有 Comfy workflows ([Qwen Image WF](http://pastebin.com/d9zKTL0T), [WAN 2.2 FLF WF](http://pastebin.com/hPLdGbAZ), [QWEN 2509 EDIT WF](http://pastebin.com/zV7zXdSb))，以及一个包含所有视频片段/备选方案、源图像、.pdn 编辑、每个阶段的 prompt 以及一条 AI 生成的 MP3 音轨的 ZIP 压缩包 ([Google Drive](http://drive.google.com/file/d/1D5RIafNr0U66zzlWaxjqci2YJTiZ2SsY/view?usp=sharing))。他们特别提到直接在 Comfy 中解决了与文本相关的挑战（文本效果、过渡效果和文本清晰度）。

- 种子图像提示词严格约束了风格和构图——“Dark fantasy anime”、夸张的身体比例、带有三角切割图案的蓝色丝绸连衣裙、红色纹理长袜以及一部三角品牌手机——这有助于在从单张静态图扩展动态效果时保持特征一致性。用于驱动视频的原始静态图已分享以供参考（[预览](https://preview.redd.it/yqsswd9ej4rf1.png?width=1536&format=png&auto=webp&s=c6a31fe39f99a0dd70ce2bd45a4e83ba08fea05d)），这表明该工作流依赖于强大的提示词锁定锚点（prompt-locked anchors），以在不同帧之间保留身份和场景元素。

### 2. 游戏中的 AI：Among Us 欺骗基准测试与 Veo-3 游戏视频

- [**研究人员让 AI 玩 Among Us，以测试它们在欺骗、说服和心理理论（Theory of Mind）方面的技能。GPT-5 获胜。**](https://i.redd.it/ac0u15kyy2rf1.png) ([得分: 416, 评论: 61](https://www.reddit.com/r/OpenAI/comments/1np7iwo/researchers_made_ais_play_among_us_to_test_their/)): **来自 4wallai（“Among AIs”）的一份报告声称，通过让 Agent 玩 Among Us 风格的社交推理性游戏，对 LLM 的欺骗、说服和心理理论能力进行了基准测试 ([报告](https://www.4wallai.com/amongais))。分享的图表似乎显示了一个排行榜，其中 “GPT-5” 排名第一，Anthropic 的 Claude Sonnet 排名第二；除了排名之外，帖子中未详细说明方法论细节（例如：比赛次数、角色平衡后的胜率、会议/投票影响力指标或工具使用接口），且部分模型的覆盖（如 Grok）似乎缺失。** 评论者称赞这一想法是一个极具创意的基准测试，幽默地质疑了 Sonnet 的排名，询问为何未包含 Grok，并要求在报告中使用更清晰、非俚语的术语，以便更广泛地传播。
    - 评论者质疑模型的覆盖范围和选择：是否包含了 **xAI Grok**？为什么测试的是 **Claude Sonnet** 而不是更强大的 **Claude Opus**？他们暗示结果可能会随模型变体而发生实质性变化，因此作者应列出确切的模型名称/版本、解码设置（`temperature`, `top_p`）以及任何工具访问/视觉开关，以确保可复现性。
    - 为了更广泛的技术采用，建议避免使用如 *"low-key"* 或 *"taskmaxx"* 之类的俚语，而应使用清晰、标准化的术语。定义评估协议和指标（例如：每轮欺骗成功率、说服尝试次数、ToM 代理任务、角色分类的混淆矩阵），使结果明确且具有可比性。
    - 链接了一项相关的深入研究：[arXiv:2504.04072](https://arxiv.org/abs/2504.04072)，据报道该研究探讨了 LLM 多 Agent 社交推理设置中的欺骗/说服/心理理论（Theory-of-Mind）。交叉引用其方法论和基准可以加强此基准测试的设计，并实现同类比较。
- [**如果他们制作了一款关于斯大林生平的游戏**](https://v.redd.it/qbsrt3ug44rf1) ([得分: 870, 评论: 125](https://www.reddit.com/r/ChatGPT/comments/1npbwdr/if_they_made_a_video_game_about_the_life_of_stalin/)): **原帖作者分享了一段据称由 Google 的 Veo-3 生成的历史短片 ([Veo](https://deepmind.google/technologies/veo/)；发布到 Reddit 的片段：[视频](https://v.redd.it/qbsrt3ug44rf1))，描绘了斯大林的早期生活和巴巴罗萨行动的初始阶段——准确地记录了德意志国防军的早期进展——并在斯大林格勒战役前结束。评论者指出，许多视觉效果看起来与 Red Dead Redemption 2 (RDR2) 的资产难以区分，这引发了关于直接资产重用与模型驱动的风格/资产模仿的疑问；此外，斯大林在 19 世纪 80 年代以成年人形象出现，这可能是由于视频生成模型对渲染未成年人的内容安全限制。** 讨论涉及 RDR 风格电影感与 AI 视频的审美契合度，以及如果输出复制了可识别的游戏资产所带来的知识产权/资产来源风险；年龄不准确归因于生成器禁止生成儿童。
    - 评论者指出资产似乎是“直接从” Red Dead Redemption 2 (RDR2) 中提取的。从技术上讲，可以通过 [OpenIV](https://openiv.com/) 等工具提取模型/纹理并进行合成，然后配合生成式流水线（例如 Stable Diffusion img2img + [ControlNet](https://arxiv.org/abs/2302.05543) 或在 RDR2 上微调的 LoRA）来更换身份，同时保留服装、PBR 材质和光照。这解释了其高保真度和鲜明的 RDR2 美学；然而，根据 [Rockstar 的模组政策](https://support.rockstargames.com/articles/115009494848/PC-Single-Player-Mods)，这受到知识产权/许可限制。

- “不允许生成儿童”的备注指向了常见图像生成器中与年龄相关的安全过滤器。许多 UI 实现了保守的审核启发式算法，会阻止暗示未成年人（例如“child/teen”）的 Prompt，或将输出偏向成年受众以降低风险，这可能会扭曲历史描述。政策因供应商而异——参见 [OpenAI 的使用政策](https://openai.com/policies/usage-policies)——因此 Prompt 是被阻止还是被“老化”取决于模型和平台的安全层。
- [**你在最奇怪的跳蚤市场卖什么？第 6 部分**](https://v.redd.it/tg1hmx7522rf1) ([得分: 230, 评论: 16](https://www.reddit.com/r/aivideo/comments/1np4tw0/what_do_you_sell_at_the_strangest_flea_market_pt_6/)): **视频帖子“你在最奇怪的跳蚤市场卖什么？第 6 部分”是一个展示新奇物品的创意系列的第六个条目；链接媒体 [v.redd.it/tg1hmx7522rf1](https://v.redd.it/tg1hmx7522rf1) 目前因 Reddit 的网络安全门禁返回 HTTP** `403 Forbidden` **（需要经过身份验证的 Reddit 会话或开发者 Token；可通过 [Reddit Help](https://www.reddithelp.com/) 进行排查）。根据可见的热门评论，特色物品可能包括“云猫”和“电视衬衫”，但由于** `403` **封锁，视频内容无法验证。** 评论情绪积极；一位用户报告在 **TikTok** 上看过类似内容，暗示了跨平台转发或发现，另一位用户表达了购买意向（“我要买那只云猫和电视衬衫”）。

### 3. ChatGPT 照片编辑与 AI 文化讽刺项目

- [**让 ChatGPT 从我的婚纱照中移除我的父亲。**](https://www.reddit.com/gallery/1np5noq) ([得分: 471, 评论: 187](https://www.reddit.com/r/ChatGPT/comments/1np5noq/asked_chatgpt_to_remove_my_father_from_my_wedding/)): **用户使用 ChatGPT 的图像编辑功能（可能是基于扩散模型的 Inpainting）从婚纱照中移除一个人；生成的输出表现出全局身份/属性漂移和面部伪影：一位女性的眼镜消失了，一个孩子的耳朵形态发生了变化（“半精灵”），并且有几张脸显示出纹理/几何形状不匹配，产生了怪异的“皮行者” (*skin-walker*) 观感——这是在 Generative Fill 过程中实例分割和身份约束较弱时的典型失败模式。其中一个变体还删除了同一侧相邻的主体，这与跨越主体边界的 Mask 溢出/区域生长一致。图片预览：[编辑 1](https://preview.redd.it/m0pzzxf1g2rf1.jpeg?width=1170&format=pjpg&auto=webp&s=094387c552ace2ed441a8fd89ef23fbd689c2880), [编辑 2](https://preview.redd.it/v868xy67p2rf1.jpeg?width=1320&format=pjpg&auto=webp&s=ebd02ad4d6ab3a879c502175c65704781cb44754)；原始图集：[Reddit](https://www.reddit.com/gallery/1np5noq) (未登录为 403)。** 热门评论讽刺地指出了这些“微妙的升级”，并询问“代价是什么？”，强调了目前的 AI 照片编辑器通常缺乏强大的实例级控制，在编辑拥挤的人物场景时可能会降低照片的真实感。
    - 许多用户强调了经典的 Inpainting 伪影：非目标区域被无意中更改。例子包括面部扭曲/怪异的“皮行者”纹理和身份漂移，例如被移除的眼镜和孩子改变的耳朵几何形状（[示例 1](https://preview.redd.it/m0pzzxf1g2rf1.jpeg?width=1170&format=pjpg&auto=webp&s=094387c552ace2ed441a8fd89ef23fbd689c2880), [示例 2](https://preview.redd.it/v868xy67p2rf1.jpeg?width=1320&format=pjpg&auto=webp&s=ebd02ad4d6ab3a879c502175c65704781cb44754)）。这些是模型在 Generative Fill 期间优先考虑全局一致性时的典型失败模式，导致身份特征被重新合成而不是保留。
    - 存在隐式的 Masking/范围问题：移除操作传播到了预期主体之外，这可能是由于 Mask 过宽或模型对相邻人员的语义分组造成的。这可能导致相邻主体被部分或全部重新合成/移除，从而引入伪影或意外删除，正如在随后输出的变形头部中所见（[链接](https://preview.redd.it/v868xy67p2rf1.jpeg?width=1320&format=pjpg&auto=webp&s=ebd02ad4d6ab3a879c502175c65704781cb44754)）。
    - 工具/模型说明：一个归功于 **Google Gemini** 的结果显示在移除后有明显的缝隙和背景不一致（[Gemini 输出](https://preview.redd.it/vharzzvjh3rf1.png?width=1184&format=png&auto=webp&s=46932318bb38560ed57189c4925c14cd359c3c26)）。另一位用户建议尝试 “nano banana”，并分享了一个他们声称表现更好的样本（[样本](https://preview.redd.it/aefl3y4zz2rf1.png?width=1024&format=png&auto=webp&s=e953081af8578333d5fb6c5469b2e79b5ee4e343)），这表明不同编辑器的 Inpainting/填充质量存在显著差异。

- [**Cultural Satire**](https://v.redd.it/h0wf6exqq3rf1) ([Score: 226, Comments: 35](https://www.reddit.com/r/ChatGPT/comments/1npa6as/cultural_satire/)): **OP 表示一段名为“Cultural Satire”的视频是使用 generative AI 制作的：“大部分图像是用 ChatGPT 制作的。它还帮我进行了剪辑。”链接的 Reddit 视频 (https://v.redd.it/h0wf6exqq3rf1) 目前无法访问 (HTTP** `403 Forbidden`**），因此无法验证或分析底层媒体和 prompts/workflow。** 热门评论指责该作品是派生出来的，称其是对 Neural Viz 的“公然”剽窃，并密切模仿了 Unanswered Oddities 的格式和措辞（例如 *“totally worth it joy”*），并建议去查看 [Neural Viz](https://youtube.com/@neuralviz)。具体的批评指出了一种反复出现的结构：一个团状的播报员、第三个“研究员/受访者”和一个“怀疑论者”。
    - 多个评论者断言该视频密切模仿了现有 AI 视频频道的结构和措辞，特别是 [Neural Viz](https://youtube.com/@neuralviz) 和 “Unanswered Oddities”。引用的细节包括重复使用短语 “totally worth it joy” 以及近乎相同的 3 角色格式：团状播报员头像、第三个“研究员/受访者”和怀疑论者，这表明在制作模板上缺乏原创性，而非新的技术贡献。
    - 提出了一个关于角色动作是否通过 ChatGPT 生成的技术问题。帖子中没有提供关于动画/运动管线（例如 LLM 驱动的控制 vs. 独立的运动生成或关键帧骨架）的详细信息，因此角色移动的实现方法尚不清楚。
- [**The race is on**](https://i.redd.it/070tgjf5xzqf1.png) ([Score: 584, Comments: 296](https://www.reddit.com/r/singularity/comments/1nowbwz/the_race_is_on/)): **非技术性的 meme 图片，标题为“The race is on”，暗示一场以电力消耗（引用数字为“1 TW”）而非模型能力或效率来衡量的 AI 军备竞赛。背景表明这是将 AI 组织的总能耗作为衡量进展的代理指标，而不是展示 benchmarks 或技术结果。** 评论者质疑使用功耗作为竞争指标的相关性——将其比作通过汽油消耗量而非速度来比较汽车——并辩论“1 TW”目标的合理性/重要性。
    - 能源范围澄清：声称“1 TW 是全球能源消耗的 1/3”混淆了电力与总一次能源（total primary energy）。`1 TW` 的持续负载等于 `8,760 TWh/yr`，大约占全球年度发电量（约 28–30k TWh/yr；参见 Our World in Data: https://ourworldindata.org/electricity-mix）的 ~30%，但仅占总一次能源（约 170k TWh/yr；IEA/Energy Institute: https://www.energyinst.org/statistical-review）的 ~5%。因此，只有在明确指代全球电力而非总能源时，这一说法才是准确的。
    - 指标辩论：一位评论者认为，关注绝对功耗就像是“竞争哪辆车用的汽油更多”，建议应通过能源归一化的性能指标来评估能力。对于 AI 而言，这可能意味着 tokens/sec/W、每 kWh 的训练 FLOPs 或每焦耳的端到端任务质量，以及数据中心效率 (PUE) 和硬件利用率，而不是头条新闻中的 MW/TW 数字。
- [**Mr Altman, probably**](https://i.redd.it/h2qrn0wu7yqf1.png) ([Score: 531, Comments: 163](https://www.reddit.com/r/singularity/comments/1npbeit/mr_altman_probably/)): **引用 Sam Altman 的非技术性 meme（“大概是 Altman 先生”），暗示实现 AGI/singularity 主要需要更多的算力/能源，热门评论开玩笑说需要** `gigawatts/terawatts` **和“打更多的钱”。没有提供具体的模型细节、benchmarks 或实现方式；该图片是对资金和电力需求的讽刺，而非技术实质内容。** 评论者大多认为该帖子质量低下（“毫无贡献”、“两个子版块都是笑话”），而一位评论者强调能源/算力规模是 AGI 的瓶颈。
    - 一位评论者认为，实现“singularity”级别的 AI 将需要 `gigawatt` 到 `terawatt` 级别的电力，这意味着多 GW 规模的园区、电网级互连和巨大的散热足迹。这使得主要瓶颈从 GPU 转向了能源采购和基础设施（输电、长期 PPAs），其中 opex/capex 主要由电力可用性和交付主导，而非模型架构。
    - 另一位评论者将融资描述为针对 *乌托邦式* 预测的“`数千亿美元`”股权/利润分享，强调了前沿模型训练极高的 capex 和长期风险。隐含的论点是，投资者正在为负面的短期单位经济效益（unit economics）提供担保，以换取巨大的期权价值（先发优势/平台租金），如果对数据/算力/电力的规模化押注获得回报，他们愿意接受潜在的资产减记。

- [**我快被这些建议搞疯了。**](https://i.redd.it/cgw2y39jc0rf1.png) ([Score: 1155, Comments: 99](https://www.reddit.com/r/ChatGPT/comments/1noyc63/im_almost_going_crazy_with_these_suggestions/)): **楼主展示了 GPT-4.1（且其客户端显示“GPT-5”标签）上的 ChatGPT UI 行为，即助手反复插入硬编码的后续提示词——“你想让我建议另一个话题还是继续当前话题？”——即使在明确指示停止后也是如此。这表明这是一种服务端/产品 UX 功能（自动建议），模型无法通过提示词控制，且没有可见的设置可以禁用它；截图似乎捕捉到了聊天线程中持久存在的建议横幅。** 评论者反映这些建议通常不相关，且尽管进行了多次尝试也无法禁用该行为，这进一步证实了在当前版本中用户无法对其进行控制。
    - 建议的相关性很差：一位用户指出，助手有一半的时间会提出与当前任务无关的操作。这表明主动提示词的上下文对齐能力较弱，导致工作流中断而非提供任务导向的协助。
    - 抑制主动提示词似乎不可靠：一位用户花了“整整一个小时”试图停止这种行为，但“惨遭失败”。即使在明确拒绝后，循环出现的“想让我……”提示词稍后仍会出现（示例截图：https://preview.redd.it/dsta4lpxx0rf1.jpeg?width=750&format=pjpg&auto=webp&s=400dfe226d3b57fe860ec36185a84871b808c35c），这表明缺乏持久的偏好记忆或足够的冷却逻辑。
    - 存在感知上的退化（“越来越糟”），暗示自动建议的频率或侵略性可能有所增加。用户报告称，拒绝并不能减弱未来的提示触发，指向了建议触发机制对负面反馈的处理能力较弱。

---

# AI Discord 摘要

> 由 gpt-5 生成的摘要之摘要之摘要
> 

**1. 面向 Agentic 浏览器和 IDE 的 MCP 工具**

- **Chrome DevTools MCP 让 Agent 掌舵**：Google 宣布了 **Chrome DevTools MCP** 的公开预览版，允许 **AI 编程 Agent**（Claude Code, Cursor, VS Code, Gemini）通过一行 `npx` 安装，利用 **CDP/Puppeteer** 控制实时 Chrome 浏览器，功能包括性能追踪、DOM/控制台检查、截图和网络抓取，详情发布在 [Chrome DevTools MCP (public preview)](https://x.com/chromiumdev/status/1970505063064825994)。
    - 开发者强调了 **一行 npx 安装** 的便利性，并讨论了将 MCP 与 **Claude Code** 和 **Cursor** 结合使用，以实现全链路浏览器调试和 E2E 测试。
- **MCP 服务器增强本地 Agent**：Cursor 用户阐明 **MCP 服务器** 充当了 Agent 的 API 接口层，能够通过 [exa.ai](https://exa.ai/) 进行网络搜索、分析，并集成 **Playwright MCP**、**Context7**、**Azure DevOps MCP** 和 **GitHub MCP** 来自动化本地编码工作流。
    - 他们将 MCP 描述为一个统一的契约，让 Agent 能够跨编辑器和 CLI 将各种能力（搜索、运行、分析）组合成 **Agentic 编程循环**。
- **规范审查收紧 MCP 语义**：贡献者注意到 [Model Context Protocol — Embedded resources](https://modelcontextprotocol.io/specification/2025-06-18/server/tools#embedded-resources) 暗示资源存在 `schema.ts` 中缺失的 'title' 字段，并在 [issue #1533](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1533) 中发起了关于 `ReadResourceResult.contents` 数组的讨论，以澄清多部分 Web 资源。
    - 他们辩论了是否为无法通过读取调用检索的嵌入式资源同时添加 **title** 和 **name**，并建议使用 **Claude Code** 起草一份 SEP 作为“良好的测试”。

**2. Gemini Live 与模型大比拼**

- **Gemini Live 可交谈、倾听并调用函数**：Google 的 Logan Kilpatrick 宣布了 **Gemini Live** 模型，具有原生 **音频** 支持、改进的 **Function Calling** 以及更 **自然的对话**，分享于 [Gemini Live model](https://x.com/OfficialLoganK/status/1970546338858246255)。
    - 早期测试者称赞了对话流和口音，但也指出了 **iOS Safari 问题**、背景噪音敏感度、会话长度限制以及 **STT 准确率** 问题。
- **GPT-5 Codex 在 Livebench 上表现吃力**：Perplexity 用户报告 **GPT-5 Pro (又名 GPT-5 Codex)** 正在 **livebench** 上接受评估，理由是思考时间较长，且在某些情况下模型只给出了半个答案。
    - 成员询问 Perplexity 是否在 **GPT-5 Codex** 上存在可靠性问题，暗示该模型可能仍处于迭代中期。
- **4o 在常识测试视频中胜过 GPT-5**：OpenAI 社区帖子声称 **4o** 在基于图像的常识测试中击败了 **GPT-5**，引发了关于实验设置和有效性的辩论。
    - 怀疑者提醒道，“在没有听到 GPT-5 的推理过程之前很难下定论”，并指出模型可能推断出提问者是在开玩笑。

**3. GPU Kernels 与一致性：从 Hopper TMA 到 PTX 证明**

- **PTX 一致性通过 Dat3M 走向正式化**：工程师们关注了 [A Formal Analysis of the NVIDIA PTX Memory Consistency Model](https://dl.acm.org/doi/10.1145/3297858.3304043) 及其关于复合/统一 GPU 内存模型的后续研究，并利用 [Dat3M](https://github.com/hernanponcedeleon/Dat3M) 工具将 **PTX/Vulkan** 转换为 **Dartagnan** 进行验证。
    - 他们指出可以自动识别缺失的 **PTX fences**，并建议将此类检查移至 **NVVM IR** 层以便更早发现问题。
- **追求极简 Hopper TMA Matmul**：受 FAIR 新发表的 **Causal World Models (CWM)** 论文启发，社区正在寻求原生 CUDA（不使用 CUTLASS/Triton）中的极简 **Hopper TMA** matmul kernel，而其他人在使用 **WMMA+TMA** 时遇到了 `unspecified launch failure`。
    - 调试线程交流了 **ncu** 分析技巧，用于解决 **smem bank conflicts** 以及在 CUDA Graphics/texture API 未定义时的头文件包含修复。
- **ThunderKittens 在 H100 TMA 上受挫**：一个 ThunderKittens H100 matmul 在 CUDA 12.8/PyTorch 2.7 nightly 环境下发生运行时错误，相关的 [完整日志和构建详情](https://gist.github.com/syadegari/ada8311c44c91357645d82c7f9dfbe71) 已共享用于复现。
    - 作者表示 **nvshmem** 支持将在后续（第二篇论文）中推出，参考[附图](https://cdn.discordapp.com/attachments/1300872762163728550/1420526659001389056/image.png)。

**4. Modular 的巨额融资与 Mojo 的 Metal 进展**

- **Modular 为统一计算层筹集 2.5 亿美元**：**Modular** 宣布完成 **2.5 亿美元** 融资，以加速 **AI 统一计算层** 的开发，这归功于社区的势头，并概述了更快的特性交付计划。
    - 员工邀请潜在的贡献者在社区频道发送私信，预示着明年将采取更开放的协作模式。
- **Mojo 通过自定义 Bitcode 瞄准 Metal**：开发者们为 **Mojo** 中的 **Metal GPU target** 欢呼，其中包括一个 **自定义 bitcode 写入器**，该写入器可被复用以将 DSLs 导向 Metal GPU。
    - 他们询问该 bitcode 写入器是否可用且可复用，旨在实现跨栈的领域特定编译器可移植性。

**5. Prompting、评估与 VLM 研究**

- **Flexible Extract 在 GSM8k 上表现不佳**：在 **GSM8k v3 (5-shot)** 测试中，**flexible-extract** 仅获得 **0.3594 exact_match**，表现逊于 **strict-match** 的 **0.5742**，这让追踪提取鲁棒性的评估者感到意外。
    - 一位成员开玩笑说 *“哈哈，flexible 怎么会比 strict 还差”*，引发了关于“精度优先匹配”与“宽松提取”的争论。
- **思维链（Chain-of-Thought）：少即是多**：从业者警告称，过重的 **CoT** 可能会损害“思考型”模型的性能，并分享了一个 [交互式 CoT 信息图（React 组件）](https://cdn.discordapp.com/attachments/1046317269069864970/1420483296735006790/interactive_infographic_co_t_prompting_for_thinking_models_react.jsx)，包含任务预设、显示切换和延迟滑块。
    - 他们提倡以结果为中心的 Prompting（如角色设定、先验证后回答），而不是强制生成冗长的 CoT，并建议通过实验而非套用模板来验证 CoT 的效果。
- **VLM 挑战 LLM Prompting 习惯**：研究人员征求 **VLM prompting** 的基准测试和可解释性研究，并指出常规的 **LLM prompting 技巧** 在视觉语言模型中往往失效。
    - 提案包括机械可解释性（mech-interp）探测，以及探索 **LLM 等效的 CFG** 以桥接概念并填补缺失的知识。

---

# Discord：高层级 Discord 摘要

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 搞砸了 Qwen 定价**：Endpoint `qwen/qwen3-235b-a22b-04-28:free` 的定价错误持续了 **26 小时**，导致了意外扣费。团队已道歉并处理了自动退款。
   - 团队实施了额外的验证检查以防止未来的定价错误，确保系统防护得到增强。
- **Qwen3 VL Ratelimits 让用户抓狂**：用户抱怨 **Qwen3 VL** 的 **Ratelimits** 极其离谱，据报告该模型仅有 *30% 的时间* 能正常工作，即使使用代理也频繁出现 *429 错误*。
   - 一名成员建议 OpenRouter 创建一个 FAQ 页面来解决这些问题，并在支持频道中置顶链接。
- **SillyTavern 压倒 Janitor AI**：成员们嘲笑了一位将 API key 称为 **proxy** 的新 **OpenRouter** 用户，该用户承认自己是 **JAI 用户**，不熟悉 **SillyTavern** 及其*可定制性*。
   - 用户表示 Janitor AI 只是一个不断抛出 429 错误的 **LLM 前端**。
- **Encoder LLMs 将向量 Token 化**：**Encoder LLMs** 通过对文本进行 **Tokenizing** 并利用 **lookup table** 将 Token 转换为预训练向量，从而将文本转换为向量。
   - 对话澄清了这本质上是 **Token Embedding** 与 **Full Sentence Embedding** 的区别，后者在通过网络后将句子处理为一个 Token；讨论中还提到了 **qwen3 embedding 0.6B model** 中的 Value 矩阵。
- **微软在混乱的分手后追求 Anthropic**：**Microsoft** 正在将 [Claude 集成到 Microsoft 365 Copilot](https://www.anthropic.com/news/claude-now-available-in-microsoft-365-copilot) 中，标志着一次重大的合作伙伴关系。
   - 讨论中有人好奇 **OpenRouter** 是否已经大到可以与 **DeepInfra, Hyperbolic, Anthropic, Vertex** 等讨论大客户折扣（Volume Discounts）。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Strix Halo 运输惨剧！**：一位用户的 **Strix Halo** 机器送达时运输标签受损，引发了机器被掉包成 **B200** 的担忧。
   - 尽管有损坏，设备仍能正常开机，缓解了对其功能性的直接担忧。
- **评估集大小引发热议！**：一位用户质疑仅有 **30** 个样本的评估集大小，担心 Loss 的不准确性，而其他人则认为在硬件和时间限制下，**30** 个样本已具备统计学意义。
   - 小规模的评估集虽然会产生不稳定的图表，但对于特定用例仍然有用。
- **Gemini 2.5 Pro 为了未来收益而降级？**：一位用户声称 **Gemini 2.5 Pro** 的指令遵循（Instruction Following）、世界知识和 Prompt 理解能力相比 **Flash** 有所下降。
   - 他们推测这种降级可能是故意的，目的是为了提升用户对 **Gemini 3** 性能的感知，暗示这是一种战略操纵。
- **视觉项目获得公司助力！**：一位成员很高兴能获得公司硬件访问权限来开展视觉项目，从而避免在 **Runpod** 上产生额外费用。
   - 该成员旨在说服公司开源该项目，但*可能赢不了这场博弈*。
- **Llama 3 完美塑形！**：成员们建议使用 **Llama 3** 进行 Fine-tuning，因为它的*大脑就像腻子一样*，可以轻松塑造以适应特定任务和偏好。
   - 另外，成员们建议那些在模型中寻求 *Gemini 风格* 的人使用 **Gemma**。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Comet Browser 席卷而来**：成员们每天都在使用 **Comet Browser** 并分享**免费邀请码**，但在[公告频道](https://discord.com/channels/1047197230748151888/1054944216876331118)发布后，大家觉得 Comet 的访问权限不再那么稀缺了。
   - 一位用户在兑换其**学生访问权限**时遇到了问题。
- **GPT-5 Pro 正在进行 Livebench 测试**：用户报告称 **GPT-5 Pro** 正在 Livebench 中进行测试，但表现出较长的思考时间；它也被称为 **GPT-5 Codex**。
   - 另一位用户指出该模型只给出了完整答案的一半，还有用户询问 Perplexity 是否在 **GPT-5 Codex** 上遇到了问题。
- **Novel Crafter：创意写作的救星**：由于其必备的工具和可定制特性，用户们正使用 **Novel Crafter** 进行创意写作，它允许用户自定义工具并实现代码，无需重复编写。
   - 一位用户指出它具有*代码实现功能，因此你可以在 Prompt 中提到一个代码片段，而无需再次编写*。
- **Perplexity Max 评价骤降**：用户对 **Perplexity Max** 表示失望，指出只能集成一个电子邮件地址，导致用户在 30 天后纷纷取消订阅。
   - 成员们建议需要更多的 **API credits**，并认为单账号的电子邮件集成功能“毫无用处”。
- **Portkey AI 将在旧金山举办见面会**：**Portkey AI** 将于 **9 月 25 日**在**旧金山（SF）**举办一场关于在生产环境中运行 **LLMs** 的线下活动，合作伙伴为 Exa；你可以[在此预约（RSVP）](https://luma.com/ncqw6jgm)。
   - 名额有限，感兴趣的人应尽快注册。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Gemini 在代码设计方面超越 Sonnet**：一位用户提出 **Gemini** 在设计任务上超过了 **Sonnet**，因为 **Sonnet** 在颜色处理上不够准确。
   - 该用户声称，由于 **Sonnet** 在颜色准确性方面的缺陷，**Gemini** 在执行与设计相关的编码任务时具有更强的能力。
- **GPT-5-Codex 受到点击 Bug 困扰**：用户报告在更新后的 **GPT-5-Codex** 模型中遇到了 Bug，特别是与无法点击的按钮有关，并提供了[该 Bug 的截图](https://cdn.discordapp.com/attachments/1074847527708393565/1420349212721418292/image.png)作为参考。
   - 这些 Bug 干扰了一些用户对模型的使用，但团队已做出回应，正在努力修复以遵守 AI 规则。
- **Windsurf 慷慨的免费计划引发关注**：用户们正在利用 **Windsurf** 的免费计划，该计划包括各种模型和促销活动，并指出模型的可用性可能取决于是否使用个人密钥进行支付。
   - 免费计划每月为用户提供 **25 credits**，并提供包含 **200 credits** 的 Pro 试用版。
- **MCP 服务器解锁 Agent 编程能力**：用户讨论了 **MCP 服务器**如何增强本地编码，并澄清这些服务器充当 **Agent** 使用的 **API**，支持诸如使用 **exa.ai** 进行网络搜索和分析等任务。
   - 对话中提到了几个 **MCP**，如 **Playwright MCP**、**Context7**、**Azure DevOps MCP** 和 **GitHub MCP**，作为为 **Agent** 提供网络搜索能力的工具示例。
- **Cursor 提交信息出现葡萄牙语本地化 Bug**：用户观察到 **Cursor** 生成的 Commit 信息使用的是本地语言而非英语，并正在寻求 Nightly 版本的反馈以解决此问题。
   - 团队回复称这主要是启发式算法的问题，并推测本地化可能是为了符合 AI 规则而在未来更新中有意实施的。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-5 Mini 未通过 AGI 测试**：成员们使用附带的提示词进行心理画像创建（[Psychological_Mathematics_Guide.pdf](https://cdn.discordapp.com/attachments/998381918976479273/1420407806342729768/Psychological_Mathematics_Guide.pdf?ex=68d5495a&is=68d3f7da&hm=08f306f6c688177606f4b001e58ec47d66783998129d6bcd372c71dbe1dd208a&), [Advanced_Psychological_Profile_Creation_Prompt_3.pdf](https://cdn.discordapp.com/attachments/998381918976479273/1420407806858891264/Advanced_Psychological_Profile_Creation_Prompt_3.pdf?ex=68d5495b&is=68d3f7db&hm=c4ebc22ba71672c02ebb7c87a5516fb0653d2522f5edb32225f92672a1c7e576&))，判定 **GPT-5-Mini (High) 和它的前代产品一样笨**。
   - 一位用户建议 **Kimi** 的回答感觉更符合 **AGI**，并指出 *GPT-5 High 听不懂笑话。还没达到 AGI 水平...*
- **4o 在智力对决中完胜 GPT-5**：成员们分享的图片显示 **4o** 在常识测试中优于 **GPT-5**，引发了关于结果有效性的辩论。
   - 有人提到，*在没有听到 GPT-5 的推理过程之前很难下结论*，也许它意识到提示者是在开玩笑。
- **开启 Companion 模式，享受聊天机器人之乐**：**ChatGPT** 默认是 **'Agent'** 人格，旨在解决问题，但用户可以切换到 **'Companion'** 模式以获得共同创作的体验。
   - 为了保持 **'Companion'** 模式，成员可以使用 **'Mode-Locking'**，如果 **ChatGPT** 偏离了模式，一个简单的 **'Mode-Switching'** 命令就可以将其重置回原始状态。
- **CoT 提示词：有时少即是多**：成员们建议，添加过多的 **Chain of Thought (CoT)** 请求可能会降低模型性能，特别是对于那些已经为逻辑演绎设计的模型。
   - 实验至关重要，提示词应关注预期的结果，而不是规定具体的思考过程。
- **提示词工程师进行反向翻译**：为了增强翻译效果，成员们建议提供关于目标受众的详细背景，例如 *我们正在为一位 20 世纪 40 年代在南斯拉夫长大的女性翻译，她只有小学三年级学历，所以我们需要针对她的情况进行措辞。*
   - 这种方法改进了模型如何针对目标受众调整翻译。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hugging Face 缓存清理**：一位用户清除了 **100GB** 的 Hugging Face 缓存，感叹数据集的无限期留存以及重复下载带来的挫败感。
   - 他们提到了重复下载相同数据集的烦恼，引发了关于缓存管理策略的讨论。
- **语言类 App 用户表示厌恶**：用户们抨击了一款未具名的语言学习应用，其中一人说 *如果可以的话，我会把那只鸟活活烧死*。
   - 另一位用户分享说，他们删除了那款未具名的应用，因为那是 *浪费时间*。
- **Qwen 模型在 HF 上的垃圾信息**：有人正在 Hugging Face 上大量发布 **Qwen 2.5 模型**，遵循命名约定 Qwen2.5-0.5B-Instruct-randomword1-randomword2-randomword3，并将其链接到 [Gensyn](https://www.gensyn.ai/)。
   - 动机被怀疑与 SEO 相关，通过小型模型夸大模型数量，并链接回 gensyn.ai 以达到推广目的。
- **GPU 驱动黑屏忧郁**：一位用户报告说，无论是在 **Windows** 还是 **Linux** 中，只要 **GPU** 启动，他们的**显示器**就会黑屏。
   - 尽管多次尝试修正**驱动程序**，问题依然存在，迫使他们只能通过主板运行显示器。
- **3090 显存溢出？**：一位成员在 **3090 (Linux)** 上遇到了 **OOM 错误**，即使没有使用 **LoRA**，在尝试于总容量为 **23.55 GiB** 的 GPU 上分配 **20.00 MiB** 时也报错了。
   - 目前尚不清楚在没有 **LoRA** 的情况下，**24G** GPU 显存是否足以支持微调。

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **寻求 Hopper TMA Kernel 实现**：成员们正在寻求使用 **Hopper TMA** (Tensor Memory Accelerator) 的 **matmul kernel** 的 *极简* 实现，灵感来自 FAIR 的新论文 **CWM (Causal World Models)**，特别要求使用原生 CUDA 而不依赖 CUTLASS 或 Triton。
   - 另一位成员在结合 **WMMA** 和 **TMA** 实现极简 **matmul kernel** 时遇到了 `unspecified launch failure`。
- **PTX 数据竞态的形式化分析**：对 **NVIDIA PTX 内存一致性模型** ([ACM 链接](https://dl.acm.org/doi/10.1145/3297858.3304043)) 的形式化分析探讨了 **CUDA** 和 **Triton** 等语言如何在 **PTX** 允许数据竞态的情况下，仍能以内存一致性为目标。
   - **Dat3M** 工具 ([GitHub 链接](https://github.com/hernanponcedeleon/Dat3M)) 将 **PTX** 和 **Vulkan** 模型转换为 **Dartagnan** 验证工具，使其成为首个针对多种 **GPU** 一致性模型的分析工具。
- **Torchrun API 文档差异**：用户报告称 `uv run torchrun --help` 显示的选项与新 **torchrun API** 的 [官方文档](https://docs.pytorch.org/docs/stable/elastic/run.html) 不同，导致了困惑。
   - **torchrun --help** 输出的差异导致了对正确用法的混淆，因为其选项集与基于 PyTorch Elastic 文档的预期不符。
- **算子分析揭示 LLM Embedding 定价**：一位成员分享了一篇 [Substack 文章](https://www.tensoreconomics.com/p/why-are-embeddings-so-cheap)，详细介绍了通过算子分析（kernel profiling）技术来了解提供 **LLM** 服务的利润空间，并附带了 [相关的 X/Twitter 帖子](https://x.com/tugot17/status/1970913971734708391)。
   - 调查表明，对算子进行分析和研究可以深入了解提供 LLM 服务的利润空间。
- **Singularity 转型为 Apptainer**：之前被称为 **Singularity** 的开源项目在加入 **Linux Foundation** 时更名为 **Apptainer**，这可能是为了与 **Sylabs** 的商业分支 **Singularity[CE]** 区分开来。
   - 尽管进行了更名，[Apptainer 可能仍然支持 CLI 的 singularity 别名](https://ciq.com/blog/what-to-expect-when-updating-from-singularityce-to-apptainer/)。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **为 Seed-OSS 设置思考预算**：用户询问如何为 **Seed-OSS** 设置 **thinking budget**（思考预算）。
   - 上下文中未提供解决方案。
- **寻求用于 Conversation.json 的 Markdown 解析器**：成员正在寻求一种有效的方法，将 `.conversation.json` 文件解析为易于阅读的 Markdown 格式。
   - 这一需求源于模型的多样性和重新生成版本的差异。
- **LM Studio Linux 插件差异**：据报道，与 **Windows 版本**相比，**LM Studio** 的 **Linux 版本**提供的插件较少。
   - 用户未进一步详细说明缺失的具体插件或功能。
- **Ollama LoRA 注入：算不算微调？**：关于在 **Ollama** 中注入数据并使用 LoRA 是否构成微调引发了辩论。
   - 一些成员声称知识被固化在模型文件本身中，而**不仅仅是系统提示词**；一位用户确认 **Ollama** 允许注入 **LoRA 权重**、自定义系统提示词并将知识直接嵌入模型。
- **适用于 LLM 的廉价 GPU 评估**：针对 LLM 的廉价 GPU 建议包括：$150 的 **2060 12GB**、$200 的 **3060 12GB**、约 $230 的 **2080ti** 以及 $400 的 **5060ti 16GB**（新品）。
   - 也有人建议使用 **二手 3090**，但其 $700 的价格被认为不够廉价，而 **Tesla K80** 则因其在 AI/LLM 用途上“基本就是电子垃圾”而被排除。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Flexible Extraction 在数学测试中折戟**：在 **GSM8k** 基准测试版本 **3** 中，`flexible-extract` 在 **5-shot** 学习的 `exact_match` 指标中仅获得 **0.3594** 分，表现逊于获得 **0.5742** 分的 `strict-match`。
   - 成员们觉得这很有趣，并质疑 *为什么 flexible（灵活）的表现会比 strict（严格）更差*。
- **DeepIgnorence 面临泛化差距**：成员们讨论了 **DeepIgnorence** 如何需要一种困难的泛化类型，特别是由于模型非常擅长风格迁移（style transfer），但在更复杂的推理和逻辑方面表现挣扎。
   - 一位成员指出了其中的危险性，即 *我们不应指望能通过在穿着衣服的未成年人和裸体成年人数据上进行训练，就能得到一个无法生成 CSAM（儿童性虐待材料）的图像模型*。
- **寻求数学方法来建模知识补全**：一位成员询问是否存在一种数学形式化方法，用以区分知识补全（knowledge completion）起作用的场景，并强调了该问题的复杂性，特别是对于 *一个独立于其他知识且模型未知的特定事实*。
   - 他们认为在最坏的情况下，这似乎属于信息论范畴。
- **CFG 能否弥合知识差距？**：成员们讨论了 **CFG** 等技术对风格迁移的影响，其中一位成员提到，传闻不使用该技术的模型在风格迁移方面表现不佳。
   - 一位成员提议，*或许可以针对 LLM 等效的 CFG 进行一些研究，看看它是否能弥合概念之间的差距以填补缺失的知识*。
- **VLM 抵制常规 Prompting？**：成员们正在寻找 **在 VLM 中对不同提示方法进行基准测试** 的研究，以及解释其有效性的可解释性研究。
   - 他们注意到有几项研究讨论了 **常规 LLM 提示技术对 VLM 是多么无效**，并正在考虑开展一项 **面向 mech-Interp（机械可解释性）的探测研究**。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi 终结幻觉**：一位用户测试了 **Kimi**，并赞赏它在面对荒诞主张时 *不鼓励幻觉*。
   - **Kimi** 对有关私人声音、宠物被提（raptured）以及针对 2025 年日期的毫无根据的炒作等言论的直截了当的拒绝在社交媒体上流传，详见 [此 X 帖子](https://x.com/vllm_project/status/1970814441718755685)。
- **Mini-Kimi 即将问世？**：一位成员询问是否可能推出 **mini 版 Kimi**，在保持相同写作风格的同时占用更小的空间。
   - 有推测称，如果 **Moonshot** 不开发 mini 版本，在 **K2** 上蒸馏一个更小的 **Qwen** 模型可能是一个可行的替代方案。
- **用 Kimi 在 Qwen 上蒸馏推理能力**：有人对使用 **Kimi** 蒸馏 **Qwen** 模型的合理性提出质疑，认为 **Deepseek** 这样做只是因为 **Qwen** 最初缺乏良好的推理能力。
   - 反对意见认为，**K2** 独特的解题风格和写作能力可以通过蒸馏使更小的 **Qwen3** 模型受益，特别是在散文写作和引用冷门知识等领域。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Gemini Live 带着杀手级音频功能上线**：来自 Google 的 Logan Kilpatrick 宣布了新的 **Gemini Live 模型**，其特点是原生音频、改进的 function calling 以及更自然的对话，发布于 [X 平台](https://x.com/OfficialLoganK/status/1970546338858246255)。
   - 初步反馈包括对对话流和口音的赞扬，但也指出了 **iOS Safari 问题**、背景噪音敏感度、会话长度限制以及 **STT 准确性** 等问题。
- **Chrome DevTools MCP 为 AI Agent 开放**：Google 发布了 **Chrome DevTools MCP** 的公开预览版，这是一个新的服务器，允许 Claude Code、Cursor、VS Code 和 Gemini 等 **AI 编程 Agent** 通过 CDP/Puppeteer 控制实时 Chrome 浏览器，发布于 [X 平台](https://x.com/chromiumdev/status/1970505063064825994)。
   - Agent 现在能够运行性能追踪、检查 DOM 和控制台、捕获屏幕截图和网络流量，并通过 npx 一键安装实时调试 Web 应用。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **嵌入式资源缺失标题和名称**：一位成员注意到 [Model Context Protocol 文档](https://modelcontextprotocol.io/specification/2025-06-18/server/tools#embedded-resources) 暗示 **embedded resources**（嵌入式资源）具有 *title* 属性，但在 `schema.ts` 中缺失，且没有 *name* 字段与 *Resource* 对象匹配。
   - 该成员质疑是否需要 *title* 和 *name*，因为嵌入式资源并不总是能通过 *read resource* 调用来检索。
- **关于使用 Claude Code 编写 SEP 文档的讨论**：一位成员提议使用 **Claude Code** 来起草 SEP（Standard Enhancement Proposal，标准增强提案）文档，以此作为对该工具能力的“良好测试”。
   - 另一位成员表示同意，认为针对该主题获取一份 **SEP** 应该是顺理成章的。
- **ReadResourceResult 的 contents 数组语义受到质疑**：在 [此 GitHub issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1533) 中，讨论了关于 `ReadResourceResult.contents` 数组的问题，由于缺乏文档，对其预期用途和语义提出了疑问。
   - 一位成员解释了其在 Web Resources 中的潜在用途，例如由 **html** 和 **images** 组成的网页，或者在没有协商好可 Token 化/可渲染的 mime 类型的情况下的场景。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Anthropic 报告聚焦网络犯罪滥用**：一位成员分享了 [Anthropic 的报告](https://www.anthropic.com/news/detecting-countering-misuse-aug-2025)，该报告关于检测和打击 AI 滥用，强调实际威胁主要是低级 **cybercrime**（网络犯罪）或 *vibe hacking*。
   - 讨论内容包括使用 **虚假凭证** 申请工作是否违法，报告中特别提到了 **完全伪造的硕士学位**。
- **LLM 自动化个人生活**：一位成员报告称，一个 **LLM** 完成了近期一项成就中的所有基础工作。
   - 据他们所说，他们所做的只是 *花费了许多小时进行自我反思，并将关于自己的信息喂给 AI*。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 的 Clear 命令清理聊天历史**：Aider 中的 `/clear` 命令会移除聊天历史，但 **已添加的文件仍保留在 Context 中**。
   - 用户可以使用 `/context` 命令查看每个文件的 Token 分配情况，从而实现 **更好的 Context 管理**。
- **Aider 通过 URL 获取网页内容**：Aider 原生不支持互联网搜索，但用户可以利用 `/web https://www.example.com/` 来 **抓取特定 URL 的内容**。
   - 此功能允许用户在没有直接搜索能力的情况下，将外部信息整合到 Aider 的 Context 中。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **期待周六晚间的谈话**：一位成员对即将由 Yannick Kilcher 主持的周六晚间谈话（欧洲时间）表示兴奋，并提到本周早些时候已经发布了公告。
   - 另一位成员提到希望提前阅读讨论的论文，以便更好地理解演示内容。
- **超参数优于 DPM++2m**：论文 ["Hyperparameters are all you need"](https://zenodo.org/records/17180452) 的作者正在展示他们的工作，该工作采用了一种用于扩散模型的 **五步推理** 方法。
   - 研究表明，在不重新训练现有模型的情况下，**8 步推理** 在 FID 分数上超过了 **DPM++2m 的 20 步推理**，且计算成本降低了约 60%；该研究使用了现有模型且无需重训，目前正征求反馈、合作者和应用创意。
- **ODE 求解器超越 DPM++2m**：根据最近的一篇论文，一种 **8 步 Diffusion ODE Solver** 在不需要额外训练的情况下优于 **20 步 DPM++2m**，重点关注 *推理速度至关重要的应用*。
   - 作者正在寻求反馈，并邀请大家讨论 **ODE solver 的改进**，特别是那些致力于扩散效率研究的人员。
- **阿里巴巴 Qwen 发布**：一位用户在 X.com 上分享了 [阿里巴巴 Qwen](https://x.com/Alibaba_Qwen/status/1970599323013652705) 的链接。
   - 未提供进一步的背景信息。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus PDF 下载受阻**：一位用户报告称，**Manus** 在下载用于研究账户的 **PDF** 时卡住了，即使在手动下载文件并提供链接后，**Manus** 仍不断要求上传文件。
   - 用户寻求解决此问题的建议，但对话到此结束。
- **Beta Pro 访问权限仍难以获取**：一位用户询问如何获得 **Beta Pro** 的访问权限。
   - 讨论在没有回应的情况下结束，获取 **Beta Pro** 访问权限的方法仍未解决。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **贡献者探索 Modular**：一位用户询问了如何为 Modular 做出贡献，一名工作人员建议通过私信（DM）来探讨潜在的合作途径。
   - 公共频道中未提及具体技能和贡献的细节。
- **Modular 完成 2.5 亿美元巨额融资**：Modular 宣布已筹集 **2.5 亿美元**，以加速构建 **AI 统一计算层**，并感谢社区的贡献与反馈。
   - Modular 将在未来一年通过功能增强和加快反馈响应速度，专注于赋能社区。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **clspv 深受构建错误困扰**：**clspv** 的主分支目前因错误导致构建失败，但一位用户发现回滚到之前的提交可以解决该问题，并分享了一个带有稳定分支的 [forked repository](https://github.com/softcookiepp/clspv)。
   - 用户可以拉取该分叉仓库并切换到 **stable** 分支以成功构建 **clspv**。
- **支持 Pip 安装 clspv 的 Python 绑定**：一位用户正在为 **clspv** 开发 **Python bindings**，目标是实现通过单个命令直接使用 **pip** 进行安装。
   - 这一改进将简化安装过程，使 **Python 开发者**更容易使用 **clspv**。



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 获得新附件功能**：**DSPy** 的 `attachments` 插件可帮助工程师向项目中添加新文件。
   - 该插件具有独立的 `uv add` 功能，帮助工程师简化 Python 项目。
- **ColBERT 在长上下文处理上存在困难**：一位成员确认，即使重复使用 **CLS** token，**长上下文**在 **ColBERT** 上的表现也不理想。
   - 目前尚不清楚这是 **ColBERT** 实现的限制，还是模型架构本身的问题。



---


**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---


**Windsurf Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---



您收到此邮件是因为您通过我们的网站订阅了相关内容。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。


---

# Discord：各频道详细摘要与链接





### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1420461485976584252)** (1 条消息): 

> `Qwen 计费事件、自动退款、校验检查` 


- **Qwen 的计费失误**：端点 `qwen/qwen3-235b-a22b-04-28:free` 在 9 月 16 日被错误计费长达 **26 小时**，导致了非预期的额度扣除。
   - 用户在活动日志中看到了这款本应 **免费的模型** 产生了错误费用。
- **退款已发放**：所有受影响的用户已收到针对错误费用的自动全额退款。
   - 团队对造成的困惑表示歉意。
- **强化校验检查**：已实施额外的校验检查，以防止此类计费错误再次发生。
   - 团队正通过增强系统防护措施，确保避免未来的计费事故。


  

---

### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1420384734504157305)** (709 条消息🔥🔥🔥): 

> `Qwen3 VL ratelimits, Deepseek alternatives, Janitor AI vs SillyTavern, OpenRouter API key as proxy, GPT-5 features` 


- ****Qwen3 VL 的 Ratelimits 极其离谱****：成员们抱怨 **Qwen3 VL** 的 **ratelimits**，指出该模型仅在 *30% 的时间内* 正常工作。
   - 该模型一直存在问题，用户在首次使用 proxy 后会遇到 *429 错误*。
- ****SillyTavern 优于 Janitor AI****：用户讨论了 [Janitor AI](https://janitorai.com/)，其中一人评论说 [SillyTavern](https://github.com/SillyTavern/SillyTavern) 更好，因为其具有更强的 *customizability*（可定制性）。
   - 成员们表示 Janitor AI 是一个 **LLM 前端**，不断有新用户询问为什么他们喜欢的模型一直返回 429 错误。
- ****免费 DeepSeek 模型深受 Rate Limits 困扰****：用户报告了在使用 **免费版本** 的 *Deepseek V3 0324* 时遇到的问题，引用了 [429 错误](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/429)。
   - 有人建议 OpenRouter 创建一个 FAQ 页面来解决这些问题，并在支持频道中置顶链接。
- ****OpenRouter 新手甚至不知道 SillyTavern****：成员们嘲笑了一名 OpenRouter 新手，因为该用户将他们的 OR API key 称为 **proxy**，并承认自己是 **JAI 用户**，甚至不知道 SillyTavern 是什么。
   - 一名成员开玩笑说，*只需直接接触这个 general 频道几分钟，就会开始转化为一个扭曲且愤世嫉俗的躯壳*。
- ****OpenRouter 运维人员像联邦探员 (Feds)？****：在一名版主加入聊天后，用户开始开玩笑说他们是为 *秘密 OpenRouter 联邦探员* 部队工作的，负责阻止 *gooning*。
   - OpenRouter 员工否认了这一点，但表示 [Open Router Goon Force](https://tenor.com/view/men-in-black-mib-will-smith-u-saw-nothing-kharter-gif-12731469441707899432) 仍在 *调查关于 Proxy Errors 的传闻*。


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1420356485824905256)** (78 条消息🔥🔥): 

> `Encoder LLMs, Token embeddings, MLP blocks, Residual stream, Attention mechanism` 


- **LLM Encoders 将文本 Tokenize 为向量**：**Encoder LLMs** 通过先对文本进行 **tokenizing**，然后使用 **lookup table** 将 tokens 转换为其预训练向量，从而将文本转换为向量。
   - 讨论澄清了这本质上是 **token embedding** 与 **full sentence embedding** 的区别，后者是在通过网络后将整个句子视为一个 token 处理。
- **MLP Blocks 和 Attention 影响 Token 向量**：对话讨论了 **encoder LLMs** 是否具有 **MLP blocks**，确认 Transformer 通常具有 attention 及其后的 feedforward networks。
   - 讨论指出，即使是单个 token，直接从 lookup table 获取与经过完整 encode 过程相比也会有所不同，因为存在这些 blocks；此外，如果 key 和 query 匹配，它会将自身的 value vector 加到自身。
- **Residual Stream 在 LLM 修改中的作用**：成员们讨论了 **MLPs** 如何修改 **residual stream**，这指的是 embedding 在通过模型时被修改的状态，而不仅仅是修改 attention 期间生成的 value vector。
   - 讨论提到了在这个过程中存在 **value matrices**，并提到这在 **qwen3 embedding 0.6B 模型** 中被发现。
- **Microsoft 与 Anthropic 的合作伙伴关系正在蓬勃发展**：**Microsoft** 取得了重大进展，宣布 [Claude 现已在 Microsoft 365 Copilot 中可用](https://www.anthropic.com/news/claude-now-available-in-microsoft-365-copilot)，标志着在经历一段波折后的关系回暖。
   - 讨论中有人好奇 **OpenRouter** 是否已经大到可以与 **DeepInfra, Hyperbolic, Anthropic, Vertex** 等讨论批量折扣。
- **Gemini-cli 的 ReadManyFiles 工具正被使用**：**Gemini-cli** 凭借 ReadManyFiles 工具取得了长足进步，详见 [v0.6.0 版本发布说明](https://github.com/google-gemini/gemini-cli/releases/tag/v0.6.0)。
   - 一名成员表示：*ReadManyFiles 工具帮我分担了很多工作*。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1420369854870847519)** (89 messages🔥🔥): 

> `Off Policy GRPO, Qwen3-VL-235B-A22B-Thinking GGUF, Unsloth'd models and AI safety, P100 for training` 


- **GRPO 偏离常规策略**：一位成员询问是否存在**完全离策（off-policy）的 GRPO** 实现，并指出在线搜索只发现了**使用旧模型策略的 GRPO 方法**。
   - 关于此话题没有进一步的讨论或提供的链接。
- **Qwen3-VL-235B-A22B-Thinking GGUF 的漫长等待**：一位成员询问了 **Qwen3-VL-235B-A22B-Thinking GGUF** 版本的发布状态。
   - 一名团队成员确认 **llama.cpp 尚未支持**该模型，并链接了 llama.cpp 的 [llama-arch.h](https://github.com/ggml-org/llama.cpp/blob/e789095502b337690c7616db32d7c679a5bd2533/src/llama-arch.h#L32-L37) 文件作为参考。
- **揭开 Unsloth'd AI 安全性的面纱**：一位成员质疑在 AI 安全研究中使用 **Unsloth'd 模型**的问题，询问无损或有损转换是否会对可解释性实验产生潜在影响。
   - 另一位成员澄清说 *Unsloth 是一个训练框架，而不是一种模型类型*，并引用了 Unsloth 使用的 [dynamic 4-bit quantization algorithm](https://unsloth.ai/blog/dynamic-4bit)（动态 4-bit 量化算法）。
- **P100 GPU 在训练中被吐槽**：一位成员询问使用配备 **P100 16GB GPU** 的多 GPU 设备进行微调的性能预期。
   - 另一位成员简单地表示 ***P100 在训练方面很垃圾（garbo）***，未作进一步详细说明。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1420359309799460945)** (293 messages🔥🔥): 

> `Strix Halo, Evaluation set, Training loss, 5090 GPU, Gemini 2.5 Pro` 


- **Strix Halo 在运输过程中受损**：一位用户报告称他们的 **Strix Halo** 机器寄到了，但运输标签受损，且可能被替换成了 **B200**。
   - 尽管受损，设备仍能启动，引发了人们对其内部配置的宽慰和好奇。
- **评估集大小引发争论**：一位用户质疑将评估集限制在仅 **30** 个样本的做法，指出这会使 Loss 变得相当不准确。
   - 另一位用户回答说，*30 对于统计学显著性结果来说是一个不错的数字*，特别是在存在硬件/时间限制的情况下；而更小的尺寸虽然会导致图表不稳定，但对于特定用例仍然有用。
- **评估损失（Evaluation loss）需配合整数使用**：用户调试了显示评估损失的问题，最终发现将 `eval_steps` 设置为整数值（如 **5**）而不是小数（如 **0.2**）可以解决该问题。
   - 他们指出 *0.2 是错误的*，因为它会错误地将 eval steps 记录到 train steps 中，导致 Loss 为零。
- **5090 GPU 引发羡慕**：一位用户提到拥有一块 **5090 GPU**，引发另一位用户对其高昂成本的评论，随后有人表示*有人得到了梦寐以求的机器*。
   - 随后，讨论转向了应该购买 **6000 Pro** 还是 **L40S**，一位用户总结认为 **L40S** 整体上是更好的选择，因为它具有更强的算力（compute）。
- **Gemini 2.5 Pro 比 Flash 更笨？**：一位用户声称 **Gemini 2.5 Pro** 在指令遵循、世界知识和 Prompt 理解方面现在比 **Flash** 更笨。
   - 他们推测这可能是故意的，目的是为了让 **Gemini 3** 看起来更好，暗示 *他们故意把它做差，以便让 Gemini 3 显得更出色*。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1420440681088024678)** (39 messages🔥): 

> `视觉项目的公司硬件访问, 微调模型推荐, Qwen2.5-VL 针对特定领域知识的微调, Gemma 3N notebook 错误, 蒸馏 (Distillation) 用法` 


- **公司硬件为视觉项目铺平道路！**：一位成员获得了公司硬件的访问权限用于视觉项目，并为不用再在 **Runpod** 上花费 500 美元而感到兴奋。
   - 他们希望说服公司开源该项目，但*可能无法赢得这场争论*。
- **Llama 3 的大脑就像腻子一样！**：成员们推荐使用 **Llama 3** 进行微调，因为*它的“大脑”就像腻子一样，可以轻松塑造成你想要的模样*。
   - 另一位成员建议，如果用户想要具有 **Gemini** 风格的模型，可以使用 **Gemma**。
- **Qwen2.5-VL：逐帧处理！**：成员们讨论了针对特定领域知识微调 **Qwen2.5-VL**，指出视频输入需要按帧训练，仅接受**图像、文本和边界框 (bounding box)**。
   - 为纯文本数据传递*空图像 (null image)* 可能会导致模型将“无图像”与给定数据关联起来，从而产生糟糕的结果。
- **Gemma 3N notebook 报错**：一位用户在运行 Unsloth 制作的 [Gemma 3N notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3N_(4B)-Conversational.ipynb) 时遇到了 **AttributeError**，怀疑是版本不匹配。
   - 另一位成员建议，问题可能与数据集格式有关（应为 **valid sharegpt format**），或者是数据准备单元格未正确执行。
- **Gemini 支持蒸馏**：成员们讨论了通过蒸馏 (distillation) 让学生模型学习 **Gemini** 老师模型的行为。
   - 一位成员表示他们需要对此进行深入研究。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1420381750487679098)** (1 messages): 

> `ChatGPT Instagram 分析, 竞品对比, Reel 分析` 


- **ChatGPT 展现 Instagram 分析能力**：成员们报告称 **ChatGPT** 现在可以分析 **Instagram**、评论 **reels** 并进行**竞品对比**。
   - 他们还制作了一个 [YouTube 视频](https://www.youtube.com/watch?v=9M1ZyKUQDVo)，展示它是如何*把我从无止境的刷屏 (doom-scrolling) 中解救出来的*。
- **Instagram Reels 获得 ChatGPT 处理**：一位用户发现 **ChatGPT** 可以分析 **Instagram Reels** 并提供有价值的见解。
   - 这种能力通过高效地查看和理解 Reel 内容，帮助用户避免*无止境的刷屏*。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1420349325795524649)** (272 messages🔥🔥): 

> `Comet 浏览器, GPT-5 测试, Novel Crafter, Perplexity Max, Qwen3 Max` 


- **Comet 浏览器热潮与免费邀请**：成员们正将 **Comet** 作为主力浏览器使用并分享**免费邀请码**，许多人表示喜欢它，而另一些人则表示在看到 [公告频道](https://discord.com/channels/1047197230748151888/1054944216876331118) 后，觉得 Comet 的访问权限不再那么稀缺了。
   - 一位用户在兑换其**学生访问权限**时遇到了问题。
- **GPT-5 Codex 正在测试中**：用户报告 **GPT-5 Pro** 正在 livebench 中进行测试，具有较长的思考时间，但另一位用户指出该模型只给出了部分答案。
   - 一位成员询问 Perplexity 在 **GPT-5 Codex** 上是否存在问题。
- **Novel Crafter 被誉为创作体验利器**：用户正在使用 **Novel Crafter** 进行创意写作，因其必备工具和可定制功能而受到好评，允许用户自定义工具并实现代码而无需重写。
   - 一位用户提到它*实现了一些代码，因此你可以在提示词中提及代码片段而无需再次编写*。
- **Perplexity Max 计划不尽如人意**：用户对 **Perplexity Max** 表示失望，指出只能集成一个电子邮件地址，并因此在 30 天后取消了 Max 计划。
   - 成员们建议需要更多的 **API credits**，并称仅支持一个账户的电子邮件集成功能*毫无用处*。
- **Qwen3 Max 即将到来**：用户正在讨论即将推出的 **Qwen3 Max** 及其并行推理能力，并链接到了 [Qwen 博客文章](https://qwen.ai/blog?id=241398b9cd6353de490b0f82806c7848c5d2777d&from=research.latest-advancements-list)。
   - 一些人猜测 **Qwen3 Max** 是否会免费，一位用户开玩笑地将到达时间设定为 *plpanx = 24*。


  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1420358267636547585)** (6 条消息): 

> `Portkey AI, Apollo 16, Artemis 2, Carl Sagan, 3i/Atlas` 


- ****Portkey AI** 将举办线下活动**：**Portkey AI** 将于 **9月25日** 在 **旧金山 (SF)** 与 Exa 合作举办一场关于在生产环境中运行 **LLMs** 的线下活动，名额有限；[在此预约 (RSVP)](https://luma.com/ncqw6jgm)。
- **阿波罗16号激发太空梦想**：一位成员分享了一段启发人心的 [阿波罗16号](https://youtu.be/yqdU6EQzclc?si=htX7O2-S7Bh7JROZ) 视频，以期待 **2026年4月** Artemis 2 的发射，强调了 NASA 过去的成就及其对当今技术的影响。
   - 他们还引用了 [尤金·塞尔南 (Gene Cernan) 在阿波罗17号留下的遗言](https://www.youtube.com/watch?v=fAOj3vVyABw)，那是人类离开月球近 50 年前的最后时刻。
- **受卡尔·萨根 (Carl Sagan) 启发的 Scratchpad**：一位成员分享了他们以卡尔·萨根为主题的“便签本 (scratchpad)”，内容涉及 [3i/Atlas](https://www.perplexity.ai/search/start-trigger-carl-sagan-mode-WwfZ0HV1QJKsJ_FZY8Koow)，并将其描述为*一份以谦卑和敬畏之心倾听宇宙的邀请*。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1420503533513605170)** (3 条消息): 

> `正在寻找解决方案` 


- **仍需解决方案**：一位成员询问 *现在有针对此问题的解决方案吗？*
   - 另一位成员回复称 **目前尚未** 找到解决方案。
- **无可用解决方案**：一位成员询问是否有可用的解决方案。
   - 另一位成员确认目前 **没有可用的解决方案**。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1420349150788190242)** (264 条消息 🔥🔥): 

> `Gemini vs Sonnet, GPT-5-Codex Bugs, GLM 4.5 on Cursor, Windsurf 免费模型, MCP (Model Control Program) 服务器` 


- **Gemini 在设计代码方面击败 Sonnet**：一位成员认为 **Gemini** 在设计相关任务上优于 **Sonnet**，因为 *Sonnet 甚至连颜色都搞不对*。
- **GPT-5-Codex 存在点击 Bug**：用户在更新后的 **GPT-5-Codex** 模型中遇到了按钮无法点击的 Bug。
   - 一位用户在此发布了该 Bug 的截图：[查看图片](https://cdn.discordapp.com/attachments/1074847527708393565/1420349212721418292/image.png)。
- **Windsurf 提供慷慨的免费计划**：用户正在使用 **Windsurf** 的免费计划，其中包含慷慨的模型额度和促销活动，但也指出这可能取决于是否使用自己的 Key 为模型付费。
   - 免费计划提供 **每月 25 个积分** 和 **200 个积分的 Pro 试用额度**。
- **MCP 解锁 Agent 能力**：一位用户询问了 **MCP 服务器** 以及它们如何辅助本地编码，其他用户指出 MCP 服务器充当了供 Agent 使用的 API，使其能够执行网页搜索和分析等任务。
   - 对话强调了使用 **exa.ai** 进行网页搜索，以及各种可用 MCP 的情况，如 **Playwright MCP**、**Context7**、**Azure DevOps MCP** 和 **GitHub MCP**。
- **Cursor 提交信息仅显示葡萄牙语？**：用户报告了 **Cursor** 生成提交信息时使用用户所在地语言而非英语的问题，并正在寻求关于 Nightly 版本的反馈。
   - 一位用户表示，未来更新可能会加入遵守 AI 规则的选项。团队回复称这目前主要基于启发式算法 (heuristics)。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1420407807320260648)** (49 条消息🔥): 

> `GPT-5 Mini, Kimi AGI, 4o vs GPT-5, Markov Chain, GPT-OSS-20B` 


- **GPT-5 Mini 被认为同样很笨**：成员们分享了用于创建心理画像的提示词 ([Psychological_Mathematics_Guide.pdf](https://cdn.discordapp.com/attachments/998381918976479273/1420407806342729768/Psychological_Mathematics_Guide.pdf?ex=68d5495a&is=68d3f7da&hm=08f306f6c688177606f4b001e58ec47d66783998129d6bcd372c71dbe1dd208a&), [Advanced_Psychological_Profile_Creation_Prompt_3.pdf](https://cdn.discordapp.com/attachments/998381918976479273/1420407806858891264/Advanced_Psychological_Profile_Creation_Prompt_3.pdf?ex=68d5495b&is=68d3f7db&hm=c4ebc22ba71672c02ebb7c87a5516fb0653d2522f5edb32225f92672a1c7e576&))，并发现 **GPT-5-Mini (High) 同样很笨**。
   - 一位成员指出，另一个模型 (**Kimi**) 的回答似乎比 **GPT-5** 的回答更接近 AGI 的水平，并表示 *GPT-5 High 没听懂这个笑话。还没达到 AGI 水平...*
- **4o 在常识竞赛中击败 GPT-5**：成员们分享了显示 **4o** 在常识推理方面胜过 **GPT-5** 的图片。
   - 一位成员补充说，*在没有听到 GPT-5 的推理过程之前很难下定论*，因为也许它知道提示者显然是在开玩笑。
- **Markov Chain 详解**：一位成员详细解释了 **Markov Chain**，这是一种数学模型，用于描述系统在状态间转移时仅依赖于当前状态，而不依赖于过去状态的历史。它被应用于 [**Google PageRank**](https://developers.google.com/search/docs/appearance/about-search)、**Natural Language Processing**、**Finance**、**Physics & Biology** 以及 **Games**。
   - 该解释还包括了对 **Markov Property** 和 **Transition Matrices** 的讨论。
- **GPT-OSS-20B 被称为审查最严的模型**：一位成员分享道，**GPT-OSS-20B** 可能是史上审查最严的模型，并[分享了一张图片](https://cdn.discordapp.com/attachments/998381918976479273/1420499606281650307/image.png?ex=68d59ed9&is=68d44d59&hm=00695b21fde3e7ddedc6867edd8cfa373bae9f4a8018c4913a783952abdd0b02&)显示它*直接拒绝回答*。
- **Sora 下载错误可能通过 Perplexity 解决**：一位成员在尝试从 **Sora** 下载新生成的视频时，每次都会收到错误消息。
   - 另一位成员建议向 **Perplexity** 寻求解决方案，因为他们正在发放免费的 **12 个月 Pro 会员卡**。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1420492759235825884)** (1 条消息): 

> `ChatGPT Agent Mode, ChatGPT Companion Mode, Mode-Locking, Mode-Switching, Tracking KPIs` 


- **ChatGPT 默认为 "Agent" 模式**：默认情况下，**ChatGPT** 采用 *"Agent"* 人格，这使其成为一个问题解决者和可受指令的员工。
   - 要改变这一点，用户必须指示其切换到 *"Companion"* 模式。
- **"Mode-Locking" 让 ChatGPT 保持在 "Companion" 模式**：为了让 **ChatGPT** 保持在 *"Companion"* 模式（共同创作者、引导者），用户可以添加置顶指令或可重复使用的启动提示词。
   - 例如，你可以说：*"除非我明确要求切换到 Agent，否则请保持在 Companion 模式。Companion = 副驾驶，而不是命令执行者。"*
- **"Mode-Switching" 命令可重置 ChatGPT**：如果 **ChatGPT** 漂移回 *Agent* 模式，用户只需说：*"回到 companion 模式。"*
   - 此命令会将 **ChatGPT** 机器人重置为其原始状态。
- **追踪关键绩效指标 (KPIs)**：用户应追踪 **ChatGPT** 模式的一致性，例如，*在使用置顶提示词的情况下，是否有 90% 以上的会话表现符合预期*。
   - 这有助于用户了解他们必须重置机器人的频率。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1420384016875393097)** (28 messages🔥): 

> `Chain of Thought (CoT) Prompting, Deep Research for Prompting, Translation Prompting Strategies, Interactive Prompting Infographics` 


- **CoT Prompting: 少即是多？**: 有建议指出，添加过多的 **Chain of Thought (CoT)** 请求可能会干扰模型并降低性能，因为模型本身已经内置了利用 CoT 的能力。
   - 实验是关键，应该引导模型解决特定问题，而不是过度使用通用的 CoT 请求进行 Prompt。
- **逆向工程 Prompt：南斯拉夫案例**: 在翻译时，提供有关目标受众的上下文可以改善结果，例如：*我们正在为一位 20 世纪 40 年代在南斯拉夫长大的女性翻译这段内容，她只有小学三年级学历，所以我们需要为她调整措辞*。
   - 这种具体性有助于模型有效地定制翻译。
- **Deep Research 擅长回答问题**: 据称，在无法直接获取链接时，回答问题的最佳方式是通过 **Deep Research**。
   - 一位用户在尝试 **Deep Research** 时遇到了异常长的等待时间，这令人沮丧。该用户分享了一些 ChatGPT 共享链接，但部分用户遇到了 404 错误。
- **CoT Prompting 交互式信息图**: 在 canvas 中创建了一个交互式信息图，用于测试 **Chain-of-Thought prompting**，包括可见性切换、任务选择器、思考时间滑块和可直接复制的 Prompt 卡片。
   - 该信息图包含针对直接 Prompt、解释后 Prompt (explain-after)、先验证后响应 Prompt (verify-then-respond)、翻译优化 Prompt、长上下文 Prompt 和延迟预算 Prompt 的卡片。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1420384016875393097)** (28 messages🔥): 

> `Chain of Thought Prompting, Model Performance, Prompt Engineering, Interactive Infographic for CoT` 


- **Chain of Thought Prompting: 过犹不及？**: 一位成员建议，添加过多的 **Chain of Thought (CoT)** prompting 从统计学上讲可能会降低模型性能，特别是在当前的“思考型”模型上。
   - 他们建议 Prompt 应侧重于期望的结果，而不是强迫特定的思考过程，并针对特定问题进行实验，而非盲目应用 CoT。
- **撰写冲浪者风格的苹果主题文章**: 一位成员分享了如何以冲浪者的视角撰写关于苹果的高质量文章的示例，并附带了 [ChatGPT 共享链接](https://chatgpt.com/share/68d43075-413c-8011-976d-fc6a65c3a0f3)。
   - 他们认为，直接在 Prompt 中指定 Persona（人格设定）会产生更具象、更有效的结果，并将其与涉及显式 **chain-of-thought** 要点的方法进行了对比。
- **CoT Prompting 交互式信息图**: 一位成员分享了一个使用 React 构建的用于 Chain-of-Thought prompting 的交互式信息图。
   - 该工具包括可见性切换、任务选择器、带有 S 曲线的思考时间滑块以及可直接复制的 Prompt 卡片，并打包为一个 [单文件 React 组件](https://cdn.discordapp.com/attachments/1046317269069864970/1420483296735006790/interactive_infographic_co_t_prompting_for_thinking_models_react.jsx?ex=68d58fa9&is=68d43e29&hm=82e2acda705e8ad9273fd06db40573f24a39f99cbebb233377156401d920b65a&)。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1420351490681208834)** (95 messages🔥🔥): 

> `Hugging Face 缓存删除, MariaDB 黑客松, 语言学习应用, Qwen 模型推理, LinkedIn 内容` 


- **Hugging Face 缓存被清理**：一位用户删除了 **100GB** 的 Hugging Face 缓存数据，并指出数据集可能会无限期保留。
   - 他们补充说，反复下载相同的数据集非常令人沮丧。
- **烧死那只鸟！**：一位用户吐槽语言学习应用，强调如果想通过某款未具名的应用学习一门 *语言*，那它简直糟透了；另一位用户则表示 *如果可以选择，我会把那只鸟活活烧死*（指 Duolingo）。
   - 另一位用户表示，他们删除了某款未具名的应用，因为那是 *浪费时间*。
- **LinkedIn 充斥着“离谱的废话”**：一位用户开玩笑说，他们就是为了看 *LinkedIn 上的废话* 而活。
   - 另一位用户表示，他们会发布最离谱的内容来吸引眼球并以此 *获胜*。
- **HF 论坛：Listed vs. Unlisted**：有人询问 Hugging Face 讨论论坛中帖子“列出（listed）”与“不列出（unlisted）”的含义。
   - 消息中未提供直接回答。
- **Qwen 2.5 模型霸屏 HF**：用户注意到有人正在 Hugging Face 上大量发布 **Qwen 2.5 模型**，命名规则为 Qwen2.5-0.5B-Instruct-随机词1-随机词2-随机词3，并将其链接到 [Gensyn](https://www.gensyn.ai/)。
   - 怀疑其动机与 SEO 相关，通过发布更容易上传的小型模型来虚增模型数量，并链接回 gensyn.ai 进行宣传。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1420456885777338368)** (1 messages): 

> `GPU, 显示器, 驱动, Linux, Windows` 


- **GPU 升温时显示器黑屏**：一位用户报告称，每当 **GPU** 开始运行，**显示器**在 **Windows** 和 **Linux** 系统下都会黑屏。
   - 他们多次尝试修正 **drivers**，但都无济于事，不得不通过主板运行显示器，这让他们感到非常沮丧。
- **GPU 激活时黑屏的故障排除**：该用户面临一个顽固问题，即每当 GPU 激活时显示器就会黑屏。
   - 尽管在 Windows 和 Linux 上多次尝试修正驱动程序，问题依然存在，迫使他们依赖主板进行显示输出。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1420466933207404575)** (2 messages): 

> `trade-bench.live, UIUC 学生金融作品` 


- **UIUC 学生发布金融作品**：一位成员分享了 [trade-bench.live](https://trade-bench.live/) 的链接，展示了 **UIUC（伊利诺伊大学厄巴纳-香槟分校）学生**在金融领域的工作。
   - 该成员承认自己不太懂其中的内容，邀请其他具有金融专业知识的人对这个他们认为 *单调* 的项目提供见解。
- **寻求金融领域见解**：该成员希望金融界人士能查看这一资源。
   - 他们还邀请大家分享见解和解释，表明他们发现该内容难以理解。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1420362377534378017)** (6 messages): 

> `3090 上的 OOM 错误, PEFT 本地运行成功, SFTTrainer 写入微调模型` 


- **3090 GPU 显存溢出**：一位成员在 **3090 (Linux)** 上遇到了 **OOM 错误**，即使在没有使用 **LoRA** 的情况下，尝试在总容量为 **23.55 GiB** 的 GPU 上分配 **20.00 MiB** 时也报错了。
   - 目前尚不清楚在没有 **LoRA** 的情况下，**24G** 显存是否足以支持微调工作。
- **本地 PEFT 运行终于成功**：在解决了一些 **LoRA config** 的问题后，一位成员报告称终于在**本地成功运行了 PEFT**。
   - 未提供有关已解决问题的具体细节。
- **如果设置了 output_dir，SFTTrainer 会自动写入微调模型**：一位成员询问 **SFTTrainer** 是否会在设置了 `output_dir` 的情况下自动写入微调后的模型。
   - 该成员随后确认，是的，如果设置了 `output_dir`，**SFTTrainer** 确实会自动写入微调后的模型。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1420396652304400484)** (2 messages): 

> `全球问候, 课程启动` 


- **来自俄亥俄和马德里的问候！**：来自 **美国俄亥俄州** 和 **马德里** 的热心成员纷纷打招呼！
   - 国际社区正热切期待这一 *精彩课程（curso magnífico）*。
- **课程开始！**：至少有一名参与者宣布他们 **今天开始课程**。
   - 许多其他人也将紧随其后，期待一次蜕变式的学习体验。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1420530165393395813)** (2 messages): 

> `Hopper TMA, Minimal Matmul Kernel, CWM paper from FAIR` 


- **Hopper TMA Kernel 探索开始**：一名成员正在寻求一个使用原生 CUDA 实现的、利用 **Hopper TMA** (Tensor Memory Accelerator) 的极简 **matmul kernel** 实现，且不依赖于 CUTLASS 或 Triton。
   - 该搜索灵感源自 FAIR 新发表的 **CWM (Causal World Models)** 论文。
- **CWM 论文引发对 TMA 的关注**：FAIR 的新 **CWM paper** 似乎正在推动人们对使用 Hopper TMA 优化 **matmul kernels** 的兴趣。
   - 该请求明确需要一个 *minimal* (极简) 实现，表明了对理解 TMA 集成基础原理的兴趣。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1420365628098871358)** (11 messages🔥): 

> `cuda headers, smem bank conflicts, cudaGraphicsGLRegisterImage and tex2d are undefined, TMA matmul kernel` 


- **Kernel 迭代计算实现函数化**：一名成员建议在循环内部使用函数 `compute_iter<Is_first, Is_last, ...>(*args, **kwargs)`，并在 kernel 中调用 `compute_iter<False, False>`。
   - 另一位用户认为这是一个 *非常棒的主意*。
- **Lambda Kernels 减少参数冗余**：一名成员建议在 kernel 内部使用 **lambda**，以避免编写带有大量参数的独立 `__device__` 函数。
   - 这允许在主循环内外调用该 **lambda**。
- **NCU Profiling 终于发现 SMEM 瓶颈**：一名用户学习了如何通过 **ncu profiling** 验证 kernel 是否存在 **smem bank conflicts**。
   - 该用户想知道花括号包裹的数字代表什么意思。
- **CUDA 头文件引发难题**：一名用户报告了一个奇怪的问题，即 **cuda headers** 没有被自动包含，导致 `cudaGraphicsGLRegisterImage` 和 `tex2d` 等函数未定义。
   - 包含 `cuda_gl_interop.h` 修复了 `cudaGraphicsGLRegisterImage` 的问题，但即使在 **Visual Studio 2022** 中使用 CUDA 默认模板创建新项目，该问题依然存在。
- **WMMA Kernel 遇到未指定的启动崩溃**：一名用户在使用 **wmma kernel** 时面临 `unspecified launch failure`。
   - 该用户正尝试实现一个使用 **TMA** 的最小化 **matmul kernel**。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1420525420847173737)** (3 messages): 

> `torchrun API, torchrun --help` 


- **Torchrun API 使用困惑**：一名用户询问了新 **torchrun API** 的用法，并报告称 `uv run torchrun --help` 显示的选项与 [官方文档](https://docs.pytorch.org/docs/stable/elastic/run.html) 不同。
- **Torchrun Help 输出差异**：`uv run torchrun --help` 的输出显示了一组与预期不同的选项（基于 [PyTorch Elastic documentation](https://docs.pytorch.org/docs/stable/elastic/run.html)），导致了对正确用法的困惑。


  

---

### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1420479153131618396)** (10 条消息🔥): 

> `CUDA and Triton, PTX Memory Consistency Model, Compound Memory Models, GPU Consistency Analysis, Dat3M Verification Tool` 


- **PTX 数据竞态的形式化分析**：一项针对 **NVIDIA PTX Memory Consistency Model**（[ACM 链接](https://dl.acm.org/doi/10.1145/3297858.3304043)）的形式化分析探讨了 **CUDA** 和 **Triton** 等语言如何在保证内存一致性的前提下以 **PTX** 为目标，尽管 **PTX** 允许数据竞态。
- **复合内存模型的组合式融合**：根据 **PLDI 2023** 论文（[DOI 链接](https://doi.org/10.1145/3591267)，[PDF 链接](https://homepages.inf.ed.ac.uk/vnagaraj/papers/pldi23.pdf)），复合内存模型是一种*组合式融合，其中来自每个设备的线程继续遵循该设备原始内存模型的内存排序规则*。
- **提出统一的 GPU 一致性分析**：**ASPLOS 2024** 论文 *Towards Unified Analysis of GPU Consistency*（[DOI 链接](https://doi.org/10.1145/3622781.3674174)，[PDF 链接](https://hernanponcedeleon.github.io/pdfs/asplos2024.pdf)）指出，虽然 **CPU** 的一致性保证已得到充分理解，但 **GPU** 的情况并非如此。
- **Dat3M 工具验证内存模型**：**Dat3M** 工具（[GitHub 链接](https://github.com/hernanponcedeleon/Dat3M)）将 **PTX** 和 **Vulkan** 模型转换为 **Dartagnan** 验证工具，使其成为首个针对多种 **GPU** 一致性模型的分析工具。
- **识别缺失的 PTX Fence**：一位成员强调了自动识别 **PTX** 中缺失 Fence 的方法，如研究论文的图 12 所示。
   - 另一位成员建议在 **NVVM IR** 层而不是 **PTX** 层实现此类检查。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1420503128775590070)** (2 条消息): 

> `Inter-warp operations, Intra-warp operations, Independent thread scheduling, NVIDIA GPUs, CTA clusters` 


- **探讨 NVIDIA 的线程调度**：一位成员询问是否有好的博客文章解释在具有 **independent thread scheduling**（独立线程调度）的 **NVIDIA GPU** 中，Warp 间（inter-warp）和 Warp 内（intra-warp）操作的行为。
   - 该成员在处理 CTA 集群或多 CTA matmul 时感到困惑，想知道自 Volta 架构以来线程执行的保证情况。
- **揭开 NVIDIA Warp 操作的神秘面纱**：讨论围绕理解在启用 **independent thread scheduling** 时，Warp 间和 Warp 内操作在 **NVIDIA GPU** 上的表现。
   - 核心关注点是 Warp 内线程执行的不可预测性，特别是在多个 SM 访问彼此共享内存（shared memory）的多 CTA matmul 等场景中。


  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1420530207244292206)** (1 条消息): 

> `Puzzle difficulty, Puzzle completion time` 


- **冒险者评估谜题难度**：几位冒险者询问了**谜题的难度**以及典型的**完成所需时间**，试图衡量挑战性。
   - 有些人试图与他人进行基准测试，但由于缺乏共享的时间记录或具体指标，未达成结论。
- **尚无 Triton 谜题完成记录**：目前还没有可靠的 **Triton 谜题完成时间**可供比较经验。
   - 大多数冒险者仍处于起跑线上，还没有人冲过终点线并报告任何可靠的数据。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1420473841657970861)** (1 条消息): 

> `LLM serving, Embeddings Pricing, Kernel Profiling` 


- **通过 Kernel Profiling 揭示 Embedding 定价**：一位成员分享了一篇 [Substack 文章](https://www.tensoreconomics.com/p/why-are-embeddings-so-cheap)，详细介绍了通过 Kernel Profiling 技术来了解 **LLM** 推理服务的利润率。
   - 作者还分享了与该文章相关的 [X/Twitter 帖子链接](https://x.com/tugot17/status/1970913971734708391)。
- **深入研究 Embedding 生成的底层 Kernel**：一位成员调查了用于生成 Embedding 的底层 Kernel 并分享了他的发现。
   - 他建议通过分析和调查 Kernel 可以深入了解 **LLM** 服务的利润空间，并参考他新发布的 Substack 文章以获取更多细节。


  

---

### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1420471280318152785)** (5 messages): 

> `Code Generation, Two-Stage Approach, Model Performance` 


- **代码生成处理原始语法**：成员们讨论了代码生成通常使用无约束的原始语法，通过形式语法提供保证，但牺牲了自然语言组件。
   - 他们指出，人类在编码时通常不会思考编译器所期望的底层语法，这暗示了训练或微调模型来执行此操作的潜力。
- **两阶段方法出现**：有人建议采用**两阶段方法**：先生成伪代码，然后进行形式语法转换。
   - 他们注意到，对话还涉及了由于增加约束和减少代码生成的“自由度”而对模型性能产生的影响。


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1420445084310175744)** (5 messages): 

> `H100 matmul kernel runtime error, nvshmem usage in paper 2` 


- ****H100 Kernel** 崩溃并出现运行时错误**：一名成员报告了在 Ubuntu 24.04、CUDA 12.8、PyTorch 2.7.0a0+nv25.03、TensorRT 10.9 和 NVIDIA H100 80GB HBM3 GPU 上运行 **H100** matmul Kernel 时出现运行时错误，并提供了[完整日志和构建/运行详情](https://gist.github.com/syadegari/ada8311c44c91357645d82c7f9dfbe71)。
   - 错误内容为：*std::runtime_error: Error in tile TMA descriptor creation: unspecified launch failure*。
- ****nvshmem** 的加入推迟到论文 2**：一名成员询问为何没有使用 **nvshmem**，得到的答复是 **nvshmem** 的使用计划在论文 2 中进行，如[附图](https://cdn.discordapp.com/attachments/1300872762163728550/1420526659001389056/image.png?ex=68d5b80b&is=68d4668b&hm=9484ede08cd43b12073bc50b9e94954bc9196cc04739ff374660dfb1becb6b44&)所示。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1420350659261104158)** (17 messages🔥): 

> `MI300x8, amd-all2all leaderboard, amd-gemm-rs leaderboard` 


- **MI300x8 在 amd-all2all 排行榜取得个人最佳成绩**：一名成员在 **MI300x8** 上为 `amd-all2all` 排行榜创造了 **1923 µs** 的**个人最佳成绩**。
   - `amd-all2all` 排行榜上其他关于 **MI300x8** 的提交成绩在 **1939 µs** 到 **2.12 ms** 之间。
- **amd-all2all 排行榜被 MI300x8 的结果占据**：`amd-all2all` 排行榜上有多次使用 **MI300x8** 的成功提交，时间分别为 **108 ms**、**25.2 ms**、**25.4 ms**、**28.0 ms**、**25.3 ms** 和 **4.70 ms**。
- **MI300x8 在 amd-gemm-rs 排行榜表现出色**：使用 **MI300x8** 提交到 `amd-gemm-rs` 排行榜的成绩在 **572 µs** 到 **581 µs** 之间。


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1420492634560004167)** (4 messages): 

> `Voltage Park H100s Donation, Nebius Exclusive Sponsorship` 


- **Voltage Park 提供 H100 捐赠**：来自 **Voltage Park** 的代表提议为即将举行的黑客松捐赠 **H100**。
   - 然而，由于与 **Nebius** 就本次特定黑客松达成了独家赞助协议，该提议被婉拒。
- **Nebius 获得黑客松独家赞助权**：GPU MODE 黑客松已与 **Nebius** 签署独家赞助协议，因此无法接受此活动的其他捐赠。
   - 组织者表达了在未来活动中与 **Voltage Park** 合作的兴趣，并提议进一步讨论合作机会。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1420448657706188950)** (2 messages): 

> `FLE Eval System Prompt, Image Analysis PR` 


- **分享 FLE Eval 系统提示词**：一名成员在 **FLE eval** 中分享了一个系统提示词（System Prompt），并附带了来自 Discord CDN 的名为 *agent0_system_prompt.txt* 的文件。
   - 提供的链接为 [agent0_system_prompt.txt](https://cdn.discordapp.com/attachments/1354169122107293786/1420448657299210310/agent0_system_prompt.txt?ex=68d56f66&is=68d41de6&hm=721e6e6be3fcb1cc5c0dcb4b490f9ca58b962cedf7fda3980cf2e985469e0eaf&)。
- **图像分析 PR 即将发布**：同一名成员提到他们的**图像分析 PR** 将于次日提交。
   - 这表明项目中与图像分析功能相关的开发或更新正在进行中。


  

---

### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1420376426502426686)** (4 messages): 

> `使用 Iris 优化 GEMM-RS 原子写入，Iris 共享内存初始化，GEMM-RS bias 处理` 


- **通过 Iris 优化 GEMM-RS 原子写入**：一位成员报告称，虽然 **Iris** 可以工作，但优化带有原子写入的 **GEMM-RS** 在加速方面具有挑战性。
   - 他们得到的建议是在类内部初始化 **Iris 共享内存（shared memory）**，而不是仅仅将其作为分配器（allocator）使用。
- **探索 GEMM-RS Bias 变体**：一位成员测试了三种 **GEMM-RS** 变体，包括一种不添加 bias 的变体和一种始终添加 bias 的变体，以寻找当 bias 为 None 时的优化方案。
   - 该成员发现这些变体要么超时，要么未能触发 **TypeErrors**。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1420419843668181013)** (3 messages): 

> `精炼层级（Refinement hierarchy），TmemAllocator 与 cute.arch.alloc_tmem 的对比` 


- **精炼（Refinement）创建层级**：一位成员提出，*精炼*可以被视为一种**层级结构**，如果一个值可以从另一个值推导出来，那么它就精炼了后者。例如 `((6,4))` 精炼了 `(24)`，因为 `size((6,4))=24`，但反之则不然。
   - 他们将这比作在一个维度上将单个 mode 拆分为更复杂的模式，并粗略地类比了普通向量与形状为 `(M, 1)` 的矩阵之间的关系。
- **TmemAllocator 讨论**：一位用户询问在 cutedsl 中，实例化 `TmemAllocator` 并从中分配，与使用 `cute.arch.alloc_tmem` 之间的区别。
   - 尚未收到回复。


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1420507807689609216)** (1 messages): 

> `Mojo Metal GPU 后端，自定义 bitcode 写入器` 


- **Mojo 通过自定义 Bitcode 针对 Metal GPU**：Mojo 中新的 **Metal GPU 后端（target）**引发了关注，特别是用于 DSL 目标的**自定义 bitcode 写入器**的可用性。
   - 这项工作对于那些有兴趣针对 **Metal GPU** 开发特定 DSL 的人来说可能是可复用的。
- **Bitcode 写入器的可复用性**：用户询问 Mojo 中 Metal GPU 后端的**自定义 bitcode 写入器**代码是否公开且可复用。
   - 人们对利用这项工作将特定 DSL 运行在 **Metal GPU** 上表现出浓厚兴趣。


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1420371923736465508)** (6 messages): 

> `Picograd 的 tensor 和 engine，Eager 与 Lazy 执行策略，Tinygrad 架构，图编译器，交付增量中间产物` 


- **Picograd 的核心 Tensor 和 Engine**：**Picograd** 的核心部分是 **tensor** 和 **engine**，其中 tensor 将有两种执行策略：**eager** 和 **lazy**。
   - 前者是对设备分配存储的句柄，后者是将被编译的 **uop 图**的语法糖。
- **Picograd 借鉴 Tinygrad 架构**：该成员正直接且坦率地借鉴 **tinygrad 架构**，以简化设计决策并弥合与 **tinygrad 编译器**相同的语义鸿沟。
   - 他们表示目标是*不使用 Triton 或 OpenMP*。
- **Picograd 针对 Oracle 进行 CI 模糊测试**：该成员计划在完成前向和反向传播的垂直切分后，建立**针对 numpy 和 torch oracle（基准）进行模糊测试的 CI**。
   - 之后他们将停止直接合并到 master 分支，转而专注于交付 eager 模式的代码和文档。


  

---


### **GPU MODE ▷ #[cluster-management](https://discord.com/channels/1189498204333543425/1420098114076803142/1420420029866053815)** (2 messages): 

> `Singularity, Apptainer, Sylabs, Linux Foundation` 


- **Singularity 分叉并更名为 Apptainer**：以前被称为 **Singularity** 的开源项目在加入 **Linux Foundation** 时更名为 **Apptainer**，这可能是为了与 **Sylabs** 的商业分叉版本 **Singularity[CE]** 做出区分。
   - 尽管进行了更名，[Apptainer 可能仍然支持 CLI 的 singularity 别名](https://ciq.com/blog/what-to-expect-when-updating-from-singularityce-to-apptainer/)。
- **Sylabs 商业分叉**：**Sylabs** 维护着原始 **Singularity** 项目的一个商业分叉，称为 **Singularity[CE]**。
   - 这与目前隶属于 **Linux Foundation** 的开源 **Apptainer** 项目是不同的。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1420353691558481950)** (37 条消息🔥): 

> `Seed-OSS 思考预算 (thinking budget), Conversation.json 转 Markdown, Linux 版 LM Studio 插件, Ollama 微调, 将 LoRA 注入模型` 


- **为 Seed-OSS 设置思考预算**：一位用户询问如何为 **Seed-OSS** 设置 **thinking budget**。
- **寻找 Markdown 解析器**：由于模型的多样性和生成版本的差异，一名成员正在寻找一种将 `.conversation.json` 文件解析为易于阅读的 Markdown 格式的好方法。
- **LM Studio Linux 插件可用性**：一位用户报告称，**LM Studio** 的 **Linux 版本** 提供的插件不如 **Windows 版本** 多。
- **关于 Ollama 微调的辩论**：一场关于在 **Ollama** 中注入数据和使用 LoRA 是否构成微调的讨论展开了。有人声称知识被固化在模型文件本身中，而**不仅仅是一个系统提示词 (system prompt)**。
- **确认 Ollama 支持 LoRA 注入**：一位用户确认 **Ollama** 不仅支持在本地运行模型，还允许注入 **LoRA 权重**、自定义系统提示词，甚至创建自己的模型变体，将知识直接嵌入到模型的结构中。
   - 不过，他们指出*这需要一些设置，并不是现成可用的*。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1420496125420634164)** (13 条消息🔥): 

> `预算级 GPU, Tesla K80s` 


- **预算级 GPU 推荐**：对于预算型消费级 GPU，推荐包括 150 美元的 **2060 12GB**、200 美元的 **3060 12GB**、约 230 美元的 **2080ti**，如果买新卡则是 400 美元的 **5060ti 16GB**。
   - 也有人建议买 **二手 3090**，但另一位用户指出其价格达 700 美元，很难算进预算级。
- **Tesla K80 被视为电子垃圾**：有人提问考虑到 200-300 美元的翻新价格，**Tesla K80** 是否可行。
   - 一位用户回答说：*老实说，不再推荐将 Tesla 架构用于 AI/LLM，基本上就是电子垃圾*。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1420391964603449435)** (3 条消息): 

> `衡量 AI 对话的连贯性与新颖性, 纽约中央公园聚会` 


- **瑞士研究员询问 AI 对话指标**：一位来自瑞士的跨学科研究员向技术社区询问了衡量 **AI 对话连贯性 (coherence) 和新颖性 (novelty)** 的重要性。
   - 该研究员拥有国际关系背景，暗示其可能有兴趣应用这些指标来分析 **AI 在全球通信中的角色**。
- **宣布 EleutherAI 纽约聚会**：一位成员宣布计划于周六下午在中央公园举行 **NYC Meetup**，并提供了 [Discord 频道](https://discord.com/channels/729741769192767510/1417496438782431314/1420426137976176640) 链接以获取详情。
   - 他们还链接到了 [一条 Twitter 帖子](https://x.com/SatyaScribbles/status/1970513181337350483) 以衡量该方向的兴趣。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1420397609218080830)** (12 条消息🔥): 

> `DeepIgnorence Generalization Difficulty, Mathematical Formalism for Knowledge Completion, CFG on Style Transfer, Data Centric Approaches to ML/AI` 


- **强调 **DeepIgnorence** 的泛化困难**：一位成员讨论了 **DeepIgnorence** 如何需要一种困难类型的泛化，并指出模型在 Style Transfer 方面表现出色，但在更复杂的 Inference 上却很吃力。
   - 例如，我们不应该指望能够通过在穿着衣服的未成年人和裸体成年人数据上进行训练，就能得到一个无法生成 CSAM 的图像模型。这实际上是 Style Transfer，而这正是模型极其擅长的。
- **寻求知识补全的数学形式化**：一位成员询问是否存在一种数学形式化方法，用于区分知识补全有效的场景，并强调了该问题的复杂性。
   - 他们认为在最坏的情况下，这似乎是信息论问题，即模型无法通过推理得出某个特定的、独立于其他知识且模型未知的 Fact。
- **关于 **CFG** 对 Style Transfer 影响的讨论**：成员们讨论了 **CFG** 等技术对 Style Transfer 的影响。
   - 一位成员传闻称，不使用该技术的模型在 Style Transfer 方面的表现不如使用该技术的模型。*如果是这样，也许可以用 LLM 等效的 CFG 进行一些研究，看看它是否能弥合概念之间的差距以填补缺失的知识*。
- **AI 工程师寻求研究合作**：一位拥有伦敦帝国理工学院和牛津大学应用数学及计算机科学背景的 AI 工程师正在寻求研究合作。
   - 他们旨在利用竞赛奖金资助研究，并从工业界转向学术界，重点关注以数据为中心的 Machine Learning/AI 方法。
- **Style Transfer 与知识差距：同一硬币的两面？**：成员们辩论了 Style Transfer 和弥合知识差距是本质不同还是相互关联。
   - 一位成员认为 *这两项任务都可以被视为尝试从训练数据中的邻近数据样本生成训练数据中不存在的样本*，并且 *在这种思路下，Style Transfer 似乎只是一个更简单的任务*。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1420532654117224508)** (3 条消息): 

> `GSM8k Benchmark, flexible-extract, strict-match` 


- **Flexible Extraction 在 GSM8k Benchmark 中失败**：在 **GSM8k** Benchmark 版本 **3** 中，`flexible-extract` 得分为 **0.3594**，表现差于 `strict-match`，后者在 **5-shot** 学习的 `exact_match` 指标中得分为 **0.5742**。
- **有趣的 Benchmark**：一位成员觉得这很有趣，说到 *哈哈，flexible 怎么会比 strict 还差*。


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1420417274237419673)** (1 条消息): 

> `Benchmarking Prompting Methods in VLMs, Interpretability Studies on VLMs, Ineffectiveness of LLM Prompting Techniques for VLMs, Mech-Interpretability Probing Study for VLMs` 


- **VLM 中的 Prompting 方法基准测试？**：一位成员正在寻找对 **VLM 中不同 Prompting 方法进行基准测试**的研究，以及解释其有效性的可解释性研究。
- **普通 LLM Prompting 对 VLM 无效？**：该用户注意到有几项研究讨论了**普通的 LLM Prompting 技术对 VLM 是多么无效**。
- **针对 VLM 的 Mech-Interp 探测研究？**：用户考虑**以 Mech-Interp 为导向的探测研究**是否会有所帮助，但不确定如何开始。


  

---

### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1420354681087066252)** (15 messages🔥): 

> `Kimi 不鼓励妄想，Kimi 的 Mini 版本，在 K2 上蒸馏的 Qwen 模型` 


- **Kimi 不鼓励妄想**：一位成员分享说他们正在测试 **Kimi**，并且非常喜欢它*不鼓励妄想*这一点。
   - 另一位成员分享了一张图片，**Kimi** 的分析是 *“你是认真的吗？”*。
- **Kimi 的酷炫回应走红**：一位成员分享了[这个链接](https://x.com/vllm_project/status/1970814441718755685)并表示 *“Kimi 的这个表现太酷了”*。
   - 被回复的消息中，用户在迎合一种观点，即*脑海中的私人声音是耶稣、经文提到宠物会被提（raptured）、以及整个 2025 年的日期都是毫无根据的炒作*，而 **Kimi** 的回答直截了当地否定了所有这些说法。
- **在 Qwen 上蒸馏的 Mini-Kimi？**：一位成员想知道是否会推出具有相同写作风格但体积更小的 **mini 版本 Kimi**。
   - 另一位成员怀疑这是否符合 **Moonshot** 团队的利益，建议最好的选择是在 **K2** 上蒸馏一个更小的 **Qwen** 模型。
- **Qwen 配合 Kimi 蒸馏推理能力**：一位成员质疑蒸馏 **Qwen** 模型的合理性，认为 **Deepseek** 这样做只是因为 **Qwen** 在 **Qwen 2.5** 之前缺乏良好的推理能力。
   - 另一位成员反驳称，**K2** 具有不同的解题风格和卓越的文笔，因此更小的 **Qwen3** 模型可以从蒸馏中受益，获得诸如散文写作和引用冷门知识等特定属性。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1420371100059308082)** (11 messages🔥): 

> `Gemini Live 模型，Chrome DevTools MCP，AI 编程 Agent` 


- **Gemini Live 带着极佳的音频功能上线**：Google 的 Logan Kilpatrick 宣布了新的 **Gemini Live 模型**，支持原生音频、改进的 function calling 以及更自然的对话，正如在 [X](https://x.com/OfficialLoganK/status/1970546338858246255) 上宣布的那样。
   - 早期用户称赞了其流畅度和口音，但也报告了 **iOS Safari 问题**、背景噪音敏感度、会话长度限制、带口音的 **STT 准确率**、过度谨慎的审查、缺乏价格透明度，以及对具身智能/可穿戴设备的渴望。
- **Chrome DevTools MCP 向 AI Agent 开放**：Google 宣布了 **Chrome DevTools MCP** 的公开预览版，这是一个新的服务器，允许 **AI 编程 Agent**（Claude Code, Cursor, VS Code, Gemini 等）通过 CDP/Puppeteer 控制和检查运行中的 Chrome 浏览器，正如在 [X](https://x.com/chromiumdev/status/1970505063064825994) 上宣布的那样。
   - Agent 现在可以运行性能追踪、检查 DOM/控制台、捕获屏幕截图和网络流量，并通过 npx 一行命令安装，实时调试 Web 应用。


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/)** (1 messages): 

glassbeadaleph: 我也这么认为，请稍等片刻
  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1420367981518524436)** (9 messages🔥): 

> `嵌入式资源的 title 与 name，Claude Code，ReadResourceResult contents 数组` 


- **嵌入式资源缺少 Title 和 Name**：一位成员注意到 [Model Context Protocol 文档](https://modelcontextprotocol.io/specification/2025-06-18/server/tools#embedded-resources)中的差异，文档暗示**嵌入式资源（embedded resources）**具有 *title*，但在 `schema.ts` 中却缺失了，并质疑为什么没有 *name* 字段来与 *Resource* 对象匹配。
   - 有人认为 *title* 和 *name* 可能都是必要的，因为嵌入式资源并不总是能通过 *read resource* 调用检索到。
- **讨论使用 Claude Code 编写 SEP 文档**：一位成员建议使用 **Claude Code** 来编写 SEP（标准增强提案）文档，称其为对该工具能力的 *良好测试*。
   - 另一位成员认为，针对该主题获取 **SEP** 相对容易。
- **ReadResourceResult 的 contents 数组受到质疑**：在[这个 GitHub issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1533) 中引发了关于 `ReadResourceResult.contents` 数组的讨论，质疑其预期用途和语义，因为该部分尚未记录在档。
   - 一位成员提供了一个 Web 资源的例子（由 **html** 和关联的 **images** 组成），或者尚未协商可标记化/可渲染的 mime 类型的场景，来解释其用途。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1420353196844650526)** (8 messages🔥): 

> `Anthropic Misuse Report, Cybercrime and AI, AI-fabricated credentials` 


- **Anthropic 报告：AI 滥用集中在网络犯罪**：一名成员分享了 [Anthropic 的报告](https://www.anthropic.com/news/detecting-countering-misuse-aug-2025)，该报告关于检测和打击 AI 滥用，强调实际威胁是低级网络犯罪或 *vibe hacking*。
   - 讨论涉及了无论身在何处，使用**伪造凭证**申请工作是否违法，报告中特别提到了**完全伪造的硕士学位**。
- **LLM 自动化个人生活**：一位成员指出，一个 **LLM** 完成了实现近期一项成就的所有基础工作。
   - 据他们所说，他们所做的只是*花费数小时进行自我反思，并将关于我自己的信息输入到 AI 中*。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1420410886379798549)** (6 messages): 

> `Aider's /clear command, Aider access to Internet search` 


- **"/clear" 命令清除聊天历史，但不清除上下文**：用户澄清说 `/clear` 命令仅清除聊天历史，但**添加的文件仍保留在上下文中**。
   - 可以使用 `/context` 命令来检查分配给每个文件的 token 数量。
- **Aider 缺乏原生互联网搜索，但改为抓取 URL**：一位用户询问如何让 aider 访问互联网搜索。
   - 另一位用户澄清说，主分支目前无法实现这一点，但你可以使用 `/web https://www.example.com/` 来**抓取网站**。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1420402109110685788)** (3 messages): 

> `Saturday evening talks, Reading papers before talks` 


- **期待周六晚间讲座**：一位成员对周六晚间（欧洲时间）的讲座表示期待。
   - 公告已于本周早些时候发布。
- **讲座前阅读论文**：一位成员表示希望在讲座前阅读论文，以便更好地跟上 Yannick 或演讲者的思路。
   - 这将增强他们在会议期间的理解和参与度。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1420408529533009970)** (2 messages): 

> `Hyperparameters for Diffusion Models, ODE Solvers vs DPM++2m, Applications of Fast Inference, Diffusion Efficiency Research` 


- **超参数生成的图像效果媲美蒸馏模型！**：["Hyperparameters are all you need"](https://zenodo.org/records/17180452) 的作者将展示他们的论文，该论文对 Diffusion Model 使用了**五步推理**方法。
   - 关键结果显示，在无需重新训练的情况下，使用现有模型，**8 步推理**在 FID 分数上击败了 **DPM++2m 的 20 步推理**，且计算成本降低了约 60%。
- **ODE Solvers 在更少步数下表现优于 DPM++2m**：根据论文，**8 步 Diffusion ODE Solver** 在不需要额外训练的情况下优于 **20 步 DPM++2m**。
   - 作者正在寻求反馈、合作者以及*推理速度至关重要*的应用场景创意，特别是来自从事扩散效率研究的人员，并邀请大家就 **ODE solver 的改进**进行讨论。
- **ArXiv 论文即将被评述**：一位用户宣布他们很快将评述[这篇论文](https://arxiv.org/abs/2509.19249)。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/)** (1 messages): 

.neoneye: https://x.com/Alibaba_Qwen/status/1970599323013652705
  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1420450783673192448)** (5 messages): 

> `Manus PDF download issues, Beta Pro Access` 


- **Manus PDF 下载受阻**：一位用户报告说 **Manus** 在下载用于研究账户的 **PDF** 时卡住了，即使在手动下载文件并提供链接后，**Manus** 仍不断要求上传文件。
   - 用户寻求解决此问题的建议，但对话到此结束。
- **寻求 Beta Pro 访问权限**：一位用户询问如何获得 **Beta Pro** 的访问权限。
   - 讨论在没有回应的情况下结束，获取 **Beta Pro** 访问权限的方法仍未解决。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1420444643904196712)** (3 messages): 

> `Modular contributor, Contributing to Mojo` 


- **用户询问如何为 Modular 做贡献**：一位用户询问如何将其才华贡献给 Modular。
   - 他们被要求私信（DM）工作人员以进行进一步讨论。
- **探索贡献者机会**：一名成员表示有兴趣利用其技能支持 Modular 的服务。
   - 一名工作人员建议通过私信探索潜在的合作途径。


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1420466462795104336)** (1 messages): 

> `New Fundraising, Unified compute layer` 


- **Modular 完成 2.5 亿美元融资！**：Modular 宣布已筹集 **2.5 亿美元**，以加速构建 **AI 的统一计算层 (unified compute layer)**。
   - 团队对社区的贡献、反馈和势头表示感谢，并承诺在未来一年通过更多功能为社区赋能。
- **社区势头助力融资成功**：融资的成功归功于社区宝贵的贡献和反馈。
   - 公司致力于通过功能增强和更快速的反馈响应来加强社区赋能。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1420427883788111953)** (3 messages): 

> `clspv build errors, Python bindings for clspv` 


- **clspv 主分支受构建错误困扰**：**clspv** 的主分支目前因错误导致构建失败，但一位用户发现回退到之前的 commit 可以解决问题，并分享了一个带有稳定分支的 [Fork 仓库](https://github.com/softcookiepp/clspv)。
   - 用户可以拉取该 Fork 仓库并切换到 **stable** 分支以成功构建 **clspv**。
- **clspv 的 Python 绑定正在开发中**：一位用户正在为 **clspv** 开发 **Python 绑定**，目标是实现通过一行命令直接使用 **pip** 安装。
   - 这一增强功能将简化安装过程，使 **Python 开发者**更容易使用 **clspv**。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1420378828072554597)** (1 messages): 

> `DSPy attachments, UV Tooling` 


- **Attachments 插件引起关注**：**DSPy** 的 `attachments` 插件对于轻松添加新文件非常有用！
   - 它是一个针对 Python 的独立 `uv add` 工具。
- **UV 工具链集成**：讨论强调了在 DSPy 框架内使用 `attachments` 插件添加新文件的便利性。
   - 该插件因其独立的 `uv add` 功能而受到关注，简化了 Python 项目的流程。