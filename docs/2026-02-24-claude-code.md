---
companies:
- alibaba
- openai
- anthropic
- cursor
- huggingface
date: '2026-02-24T05:44:39.731046Z'
description: '**阿里巴巴**推出了 **Qwen 3.5 中型模型系列**，包含 **Qwen3.5-Flash**、**Qwen3.5-35B-A3B
  (MoE)** 和 **Qwen3.5-122B-A10B (MoE)** 等模型。该系列凭借 **100万上下文**和 **INT4 量化**等创新技术，强调效率优于规模。


  **OpenAI** 通过 **Responses API** 发布了 **GPT-5.3-Codex**，带来了增强的文件输入支持以及基于 Web Socket
  的更快吞吐量。


  **Anthropic** 推出了 **Claude Code 远程控制（Remote Control）** 功能，支持从移动端继续终端会话，并扩展了企业级工作流功能。


  **Cursor** 将用户体验（UX）的重心从代码差异比对（diffs）转向了智能体（agent）演示视频，突出了全新的交互模式。'
id: MjAyNi0w
models:
- qwen3.5-flash
- qwen3.5-35b-a3b
- qwen3.5-122b-a10b
- qwen3.5-27b
- qwen3.5-397b-a17b
- gpt-5.3-codex
- claude-code
people:
- awnihannun
- andrew_n_carr
- justinlin610
- unslothai
- terryyuezhuo
- haihaoshen
- 0xsero
- ali_tongyilab
- scaling01
- gdb
- noahzweben
- _catwu
title: Claude Code 周年庆 + 新品发布：Qwen 3.5、Cursor 演示、Cognition Devin 2.2、Inception Mercury
  2。
topics:
- model-architecture
- reinforcement-learning
- quantization
- context-windows
- agentic-ai
- api
- websockets
- software-ux
- enterprise-workflows
- model-deployment
---

**所有人都在随时随地发布一切。**

> 2026年2月23日至2月24日的 AI 新闻。我们为您检查了 12 个 subreddit、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 24 个 Discord（**262** 个频道，**10075** 条消息）。预计为您节省阅读时间（以 200wpm 计算）：**874** 分钟。[AINews 网站](https://news.smol.ai/) 允许您搜索所有往期内容。提醒一下，[AINews 现在是 Latent Space 的一个板块](https://www.latent.space/p/2026)。您可以[选择加入/退出](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) 邮件接收频率！

---

# AI Twitter 回顾

**前沿模型生态：Qwen 3.5 “中型系列”与开源权重势头**

- **Qwen 3.5 中型模型系列**：阿里巴巴发布了一组定位精准的“更高智能、更低算力”模型——**Qwen3.5-Flash** (托管版)、**Qwen3.5-35B-A3B (MoE)**、**Qwen3.5-122B-A10B (MoE)** 和 **Qwen3.5-27B (dense)**——主张架构 + 数据 + RL 可以超越单纯的参数规模化。显著细节包括 **Flash 默认支持 1M Context** 以及托管服务中内置的工具。查看来自 [@Alibaba_Qwen](https://x.com/Alibaba_Qwen/status/2026339351530188939) 的完整公告以及 Hugging Face/ModelScope/APIs 的链接。  
  - 早期从业者的反应强调了 **35B-A3B** 和 **122B-A10B** 在实践中的强劲表现（例如 [@andrew_n_carr](https://x.com/andrew_n_carr/status/2026347588950372752), [@JustinLin610](https://x.com/JustinLin610/status/2026343725719568395)），此外 [@awnihannun](https://x.com/awnihannun/status/2026353100144218569) 指出 **35B 模型超越了 235B 的前代产品**，这暗示了“单位瓦特智能”的提升。  
  - **部署/推理栈发展迅速**：社区工具紧随其后——来自 [@UnslothAI](https://x.com/UnslothAI/status/2026351337970217357) 的 GGUF + 尺寸指南，以及像 [@terryyuezhuo](https://x.com/terryyuezhuo/status/2026344442186326332) 发出的“35B-A3B is all you need”这种本地运行热潮。Qwen 还强调了对 SGLang 的支持 ([推文](https://x.com/Alibaba_Qwen/status/2026348924433477775))。  
  - **量化 + “本地前沿”趋势线**：INT4 变体通过 [@HaihaoShen](https://x.com/HaihaoShen/status/2026208062009426209) 出现（重复帖子），用户继续推动激进的量化工作流（例如 [@0xSero](https://x.com/0xSero/status/2026223879077712269) 对 Unsloth 超低比特本地版 Qwen 的赞赏）。  
  - **评估信号**：Qwen 的旗舰模型 **Qwen3.5-397B-A17B** 在 HF 上走红 ([@Ali_TongyiLab](https://x.com/Ali_TongyiLab/status/2026211680653611174))，并在 Code Arena 的 Agentic Web 评估中表现强劲 ([Arena 帖子](https://x.com/arena/status/2026337606137725363))。Arena 还发布了相对于 Qwen 3.0 的排名增量 ([对比](https://x.com/arena/status/2026404630297719100))。  

**OpenAI + Anthropic “将代码 Agent 作为产品切入面” (APIs, 远程控制, Web Sockets, 工作量证明 UX)**

- **OpenAI: Responses API 中的 GPT-5.3-Codex**：OpenAI 通过 **Responses API** 向所有开发者推送了 **GPT-5.3-Codex** ([公告](https://x.com/OpenAIDevs/status/2026379092661289260))，[@scaling01](https://x.com/scaling01/status/2026379113099862018) 引用的定价为（**$1.75 输入 / $14 输出**）。OpenAI 还为直接摄取“现实世界文件”的 Agent 扩展了 **文件输入类型** (docx/pptx/csv/xlsx/etc.) ([推文](https://x.com/OpenAIDevs/status/2026420817568084436))。  
  - 架构细节：Web Sockets 成为提升 Agent 吞吐量的关键杠杆——据 [@gdb](https://x.com/gdb/status/2026380170765152302) 称“**部署速度提升 30%**”。这呼应了关于为什么 Web Sockets 需要时间以及状态如何在上游存储而非 VRAM 中的广泛讨论 ([线程](https://x.com/dejavucoder/status/2026219239477215657), [后续](https://x.com/dejavucoder/status/2026223111021220265))。  
  - 基准测试：第三方榜单显示 Codex 5.3 在 TerminalBench/IOI/LiveCodeBench/VibeCodeBench 中均名列前茅 ([ValsAI](https://x.com/ValsAI/status/2026385804940230786))。  

- **Anthropic: “Claude Code 远程控制” + 企业工作流推进**：Anthropic 推出了 Claude Code 的“远程控制”功能——在本地启动终端会话，并**从手机上继续操作**——最初由 [@noahzweben](https://x.com/noahzweben/status/2026371260805271615) 发布，随后由 [@claudeai](https://x.com/claudeai/status/2026418433911603668) 官方化，[@_catwu](https://x.com/_catwu/status/2026421789476401182) 确认了推送。  
  - 独立的企业定位：用于在团队间自定义 Claude 的“Cowork 和插件更新”上线，获得了极高的互动量 ([@claudeai](https://x.com/claudeai/status/2026305186671608315))。

- **Cursor: “审阅的是演示视频，而非 Diff”**：Cursor 宣布了一项重大的 UX 转型——Agent 可以**使用它们构建的软件**，然后发送其**工作的视频**（“是演示，而非 Diff”）([发布](https://x.com/cursor_ai/status/2026369873321013568)，[链接](https://x.com/cursor_ai/status/2026369880795263328))。多位开发者将云端 Agent 描述为一种实际的阶跃式进步：异步、基于 VM 的测试、自我验证以及演示产物 ([示例](https://x.com/fredrikalindh/status/2026379400879730794)，[另一个](https://x.com/jsngr/status/2026371033201103036)，“[模拟系统之上的创意总监](https://x.com/jasonyuan/status/2026375381872423133)”)。

**用于语言的 Diffusion：Inception Labs Mercury 2 与“作为下一个战场的速度”**

- **Mercury 2（“推理扩散 LLM”）**：Inception Labs 发布了 **Mercury 2**，将其定位为一款生产级 Diffusion LLM，输出速度达到 **~1,000 tokens/s** ([Stefano Ermon](https://x.com/StefanoErmon/status/2026340720064520670))。Artificial Analysis 的分析认为，它在智能水平上并非处于业界领先地位，但在**输出速度**上异常强劲，并具有不错的 Agentic/编程评估表现，包括在 Terminal-Bench Hard 和 IFBench 评分上的对比主张 ([分析线程](https://x.com/ArtificialAnlys/status/2026360491799621744))。
- 这些帖子背后更深层的启示是：各团队正押注于**架构级的并行 Token 精炼** (Diffusion) 可以让多步 Agent 循环和语音助手感觉是“原生”的，而非“批处理”式的（参见 [@LiorOnAI](https://x.com/LiorOnAI/status/2026376138428395908) 的架构解释）。这与一种更广泛的观点不谋而合：2026 年的竞争可能由**延迟 + 吞吐量**定义，而非仅仅是原始基准测试的最大值。

**Agent：可靠性、安全失效、内存 + 上下文腐烂以及新的多语言评估**

- **Agent 的可靠性未能跟上能力的提升**：由普林斯顿大学领导的一项研究正式化并衡量了**能力与可靠性之间的鸿沟**，将可靠性分解为 **12 个维度**，发现尽管能力大幅提升，但可靠性的提升却微乎其微 ([论文 + 仪表板](https://x.com/steverab/status/2026383575080108436)；来自 [@random_walker](https://x.com/random_walker/status/2026384543700115870) 的补充评论)。这与从业者将 Agent 与自动驾驶汽车 (AV) 进行比较时反复提到的“失败的长尾效应”直觉一致 ([ahall_research](https://x.com/ahall_research/status/2026338695536848987))。
- **OpenClaw 与“常规步骤分解”安全绕过**：一种具体的 Agent 失效模式：“将危险命令分解为几个常规步骤 → 安全性荡然无存”，文中引用了清空收件箱的行为；作者声称提供了一个开源修复方案 ([论文线程](https://x.com/shi_weiyan/status/2026300129901445196))。
- **AGENTS.md（及同类文件）可能适得其反**：两篇高价值帖子总结的研究表明，**LLM 生成的上下文文件会降低成功率**，同时增加成本；开发者编写的极简上下文虽有微小帮助，但仍会增加成本。参见 [@omarsar0](https://x.com/omarsar0/status/2026306141181898887) 的论文摘要，以及 [@_philschmid](https://x.com/_philschmid/status/2026354033418547444) 基于同一组结果编写的实用“编写指南”。
- **新的 SWE-bench Multilingual 排行榜**：旨在推动对英语/Python 之外的软件工程 Agent 进行评估。该排行榜涵盖 **9 种语言的 300 个任务**，均不包含在 SWE-bench Verified 中，报告的 SOTA 为 **72%** ([发布](https://x.com/OfirPress/status/2026324248973689068)；来自 [@KLieret](https://x.com/KLieret/status/2026322986907652295) 的更多统计数据)。其启示是：模型排名在不同语言之间可能会发生反转——这对于全球开发工具和数据采集策略至关重要。

**数据 + 基准测试：OCR 饱和、“新优化器”怀疑论以及自适应/持续数据方案**

- **OCR/文档解析基准测试趋于饱和**：多篇文章指出 OmniDocBench 正面临瓶颈（例如，准确率达到 **~95%** 但在真实文档上仍有失败），且精确匹配（exact-match）指标会惩罚语义正确的解析。参见 [@llama_index](https://x.com/llama_index/status/2026342120236396844) 和 [@jerryjliu0](https://x.com/jerryjliu0/status/2026408921385284001)。相关话题还包括：困惑于为何尽管拥有廉价的合成数据，OCR 依然难以攻克 ([gabriberton](https://x.com/gabriberton/status/2026335831632626156))，以及一项研究表明在 PDF QA 中文本提取优于图像表示 ([cwolferesearch](https://x.com/cwolferesearch/status/2026344301907583469))。  
- **“Nature MI 优化器”争议**：一篇高度技术性的评论指出，某篇拥有夸张图表的全新优化器论文存在基准线（baselines）可疑以及潜在的测试集超参数选择问题，呼吁进行独立验证并使用调优更好的基准线（如 nanogpt speedrun）([giffmana](https://x.com/giffmana/status/2026223201957597563)；以及来自 [@YouJiacheng](https://x.com/YouJiacheng/status/2026224486367027622) 的额外实验背景）。  
- **Adaption Labs：“自适应数据（Adaptive Data）”**：多条推文宣传将从静态数据集转向“活资产（living asset）”循环，声称在 **242 种语言**中平均实现了 **82% 的质量提升**，并推出了早期访问/社区计划 ([company](https://x.com/adaptionlabs/status/2026281291847446721)；来自 [@sarahookr](https://x.com/sarahookr/status/2026286134104613157) 的额外阐述；第三方转述见[此处](https://x.com/sudip_r0y/status/2026286762851774475))。在更多方法论公开之前，应将其视为一种趋势性论点（数据漂移/反馈循环）而非经过验证的标准。

**算力、芯片与机器人：Meta–AMD 巨额交易、MatX 的“HBM+SRAM”豪赌，以及人形机器人控制的扩展**

- **Meta ↔ AMD 基础设施交易**：Meta 宣布了一项多年期协议，将 AMD Instinct GPUs 整合到计划容量约 **6GW** 的数据中心部署中 ([@AIatMeta](https://x.com/AIatMeta/status/2026266818789454057))。评论认为这是在 NVIDIA 财报前夕发出的重大资本支出/算力信号 ([kimmonismus](https://x.com/kimmonismus/status/2026279386681356704))。  
- **MatX “One” 加速器**：MatX 宣布获得 **5 亿美元 B 轮融资**，并推出了一种芯片架构，旨在将 **脉动阵列（systolic-array）的效率**与在较小矩阵上更好的利用率相结合，目标是实现 **高吞吐量和低延迟**，通过 HBM 明确解决长上下文（long-context）工作负载，同时保留 SRAM 优先的延迟特性 ([reinerpope](https://x.com/reinerpope/status/2026351870852358492))。Karpathy 强调了“两个内存池”的约束（SRAM vs DRAM/HBM），并将内存与算力的编排视为应对未来 Token 需求的核心难题 ([karpathy](https://x.com/karpathy/status/2026452488434651264))。  
- **Liquid AI LFM2-24B-A2B**：Liquid AI 发布了 **LFM2-24B-A2B**，这是一个 **24B MoE** 模型，每 Token 活跃参数量约为 **2.3B**，针对效率和 32GB 显存占用的边缘推理进行了优化 ([发布公告](https://x.com/liquidai/status/2026301771539202269))。该模型迅速分发到了 **Ollama** ([推文](https://x.com/ollama/status/2026305296709173535)) 和 **LM Studio** ([推文](https://x.com/lmstudio/status/2026322404142633131))。  
- **机器人扩展：NVIDIA SONIC (GEAR-SONIC)**：一个引人注目的机器人学讨论串声称，一个在 **1 亿+ 动作捕捉帧**和 **50 万+ 并行模拟机器人**上训练的 **42M 参数**策略，成功 **零样本（zero-shot）** 迁移到真实人形机器人上，并在 50 个序列中实现了 **100% 的成功率**；代码和权重已开源 ([Jim Fan 的讨论串](https://x.com/DrJimFan/status/2026350142652383587)，以及[相关链接](https://x.com/DrJimFan/status/2026350144300658891))。其核心的“系统层”主张是，来自运动追踪的密集监督可以作为全身控制中 Next-token prediction 的一种可扩展模拟。

---

### 热门推文（按互动率、技术/行业相关性排序）

- **Claude Code Remote Control** 上线：[@claudeai](https://x.com/claudeai/status/2026418433911603668)  
- **Qwen 3.5 中型模型系列**发布：[@Alibaba_Qwen](https://x.com/Alibaba_Qwen/status/2026339351530188939)  
- **Cursor Agent 发布“演示而非差异 (demos not diffs)”功能**：[@cursor_ai](https://x.com/cursor_ai/status/2026369873321013568)  
- **Karpathy 论 CLI 作为 Agent 原生界面**：[@karpathy](https://x.com/karpathy/status/2026360908398862478)  
- **Meta–AMD 6GW 基础设施协议**：[@AIatMeta](https://x.com/AIatMeta/status/2026266818789454057)  
- **Mercury 2 Diffusion LLM 发布**：[@StefanoErmon](https://x.com/StefanoErmon/status/2026340720064520670)  
- **NVIDIA SONIC 人形机器人控制（开源）**：[@DrJimFan](https://x.com/DrJimFan/status/2026350142652383587)  
- **MatX 芯片 + 5 亿美元 B 轮融资**：[@reinerpope](https://x.com/reinerpope/status/2026351870852358492)  
- **AGENTS.md 研究摘要（上下文可能带来负面影响）**：[@omarsar0](https://x.com/omarsar0/status/2026306141181898887)  
- **OpenAI GPT-5.3-Codex 接入 Responses API**：[@OpenAIDevs](https://x.com/OpenAIDevs/status/2026379092661289260)

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Qwen3.5 模型发布与基准测试

  - **[Qwen/Qwen3.5-122B-A10B · Hugging Face](https://www.reddit.com/r/LocalLLaMA/comments/1rdlc02/qwenqwen35122ba10b_hugging_face/)** (热度: 621): **Hugging Face 上的 Qwen3.5-122B-A10B 模型是一款尖端的因果语言模型 (Causal Language Model)，拥有 `1220 亿参数`，上下文长度为 `262,144 tokens`，可扩展至 `1,010,000 tokens`。它集成了视觉编码器，并采用 Gated Delta Networks 与 Mixture-of-Experts 的混合架构，增强了多模态学习和推理效率。该模型支持 `201 种语言`，并在不同环境的可扩展强化学习方面表现出色，标志着多模态 AI 应用的重大进步。** 评论者注意到该模型在 HLE 上的分数为 `25.3`，这在六个月前是 SOTA (State-of-the-art) 水平，并讨论了其作为 `gpt-oss-120b` 竞争对手的潜力。然而，令人失望的是缺乏原生的 4-bit 权重，这对于高效的模型推理服务至关重要，尤其是在 vLLM 等环境中。

    - Qwen/Qwen3.5-122B-A10B 模型在 HLE 基准测试中获得了 `25.3` 分，这在约六个月前被认为是 SOTA。这表明该模型与之前的领先模型相比具有竞争力，尽管此后行业格局已发生变化。
    - 有关于 Qwen/Qwen3.5-122B-A10B 模型缺乏原生 4-bit 权重支持的讨论，这被视为与提供原生量化的 `gpt-oss-120b` 等模型相比的一个局限。这对于在 vLLM 上运行模型的用户尤为重要，因为原生量化模型可以带来性能优势。
    - 评论指出，由于封锁，中国实验室可能无法在 MXFP4/NVFP4 上进行训练，这可能会影响原生量化模型的可用性。这可能是 Qwen/Qwen3.5-122B-A10B 等模型开发和部署中的一个重要因素。

  - **[Qwen/Qwen3.5-35B-A3B · Hugging Face](https://www.reddit.com/r/LocalLLaMA/comments/1rdlbvc/qwenqwen3535ba3b_hugging_face/)** (热度: 625): **Hugging Face 上的 Qwen3.5-35B-A3B 模型是一款尖端的因果语言模型，配有视觉编码器，拥有 `350 亿参数`。它具有统一的视觉-语言基础，并采用 Gated Delta Networks 与 Mixture-of-Experts 的混合架构以提升性能。该模型针对高吞吐量推理进行了优化，支持 `201 种语言`，使其在推理、代码编写和视觉理解等应用中具有多功能性。它还提供了超长上下文长度和可扩展的强化学习以增强适应性。** 一条评论强调，`35B` 模型的表现超过了前一代的 `235B` 模型，如 [Alibaba 的推文](https://x.com/Alibaba_Qwen/status/2026339351530188939) 所述。另一条评论提到了正在进行的模型量化版本转换工作，表明社区在优化其部署方面表现活跃。

- 根据 [Alibaba 的推文](https://x.com/Alibaba_Qwen/status/2026339351530188939)，据报道 Qwen3.5-35B-A3B 模型的表现优于旧一代模型（如 235B）。这表明模型架构或训练技术有了显著改进，使得较小的模型能够超越体量大得多的前代产品。
- Qwen3.5-35B 模型在特定 Benchmark 上达到了惊人的 40%，明显高于 GPT 120B 模型典型的 25%。这种性能跨越令人惊讶，尤其是与得分约 35% 的 Qwen3 80B Coder 模型相比。这表明模型的效率或能力有了实质性提升，引发了对其进一步测试和潜力探索的热潮。
- 包括 Qwen3.5-35B-A3B 在内的多种 Qwen 模型的发布，展示了满足不同需求的多样化阵容，例如 Qwen3 30B A3 MoE 和 Qwen3 Coder 80B A3 MoE。这种多样性表明了模型开发的一种战略方法，为不同的应用和计算资源提供了选择。

- **[在 Qwen Chat 上发现新的 Qwen3.5 模型](https://www.reddit.com/r/LocalLLaMA/comments/1rdfhfx/new_qwen35_models_spotted_on_qwen_chat/)** (Activity: 979): **图片展示了聊天界面上的新 **Qwen3.5 系列模型**，重点介绍了三个不同的模型：`Qwen3.5-122B-A10B`，这是一个专为文本和多模态任务设计的 Mixture of Experts (MoE) 模型；`Qwen3.5-27B`，一个为本地部署优化的 Dense 模型；以及 `Qwen3.5-35B-A3B`，另一个用于类似任务的 MoE 模型。这些模型是开源计划的一部分，支持多种功能，并表明持续关注 Dense 和 MoE 架构。`122B MoE` 模型的出现尤其引人注目，因为它填补了 GLM 等其他模型未发布中型 MoE 模型的空白。** 评论者对 `122B MoE` 模型表示热切期待，并指出在 GLM 等其他模型缺乏类似产品的情况下，它的重要性。此外，人们也对中型 Dense 模型（如 `27B` 模型）的持续开发表示赞赏，认为这些模型对于本地部署非常有价值。

- Freigus 强调了 27B Dense 模型和 122B Mixture of Experts (MoE) 模型的发布，对中型 Dense 模型仍在开发中表示满意。这表明开发重点在于平衡模型大小和性能，这对于资源受限的各种应用至关重要。
- durden111111 指出对 122B MoE 模型的需求，特别是由于 GLM 尚未发布中型 MoE 模型。这表明 Qwen 可能会填补大规模 MoE 模型的市场空白，这对于需要高计算效率和可扩展性的任务可能具有重要意义。
- CireHF103 注意到 Qwen Next 和 3.5 模型相比 3.0 版本有显著改进，尤其是在较小的模型尺寸上。这表明模型架构或训练技术的持续增强提升了不同规模下的性能，这对广泛的应用场景都有益处。

- **[Qwen 发布全新 Qwen3.5 Medium 模型！](https://www.reddit.com/r/LocalLLM/comments/1rdnlvl/qwen_releases_new_qwen35_medium_models/)** (Activity: 90): **Qwen** 发布了 Qwen3.5 Medium 系列下的新模型，包括 `35B-A3B`、`27B` 和 `122B-A10B`。这些模型在指令遵循、视觉推理和文档识别等各种 Benchmark 上进行了评估，其性能通过柱状图可视化。这些模型设计了不同的 Context Size 和硬件需求，表明其专注于可扩展性和对不同计算环境的适应性。此次发布包含了 [Hugging Face](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF) 上可用的各种位宽配置的 GGUF 版本，增强了测试和部署的可访问性。评论者们渴望测试新模型，特别是对比 `35B` 在 `4bit` 下与 `27B` 在 `6bit` 下的性能。随着 GGUF 模型数量的增加，也有人呼吁加强对 vLLM 的支持。

- Qwen3.5 Medium 模型的发布包括从 2 到 16 位的各种 GGUF 格式，这些格式已在 Hugging Face 上提供。这种多样性允许在不同的精度水平上进行测试，这对于在模型部署中平衡性能和资源使用至关重要。[Link to models](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF)。
- 目前正在讨论对 GGUF 模型的 vLLM 支持需求，这表明需要能够处理这些新模型格式的更高效推理框架。随着更多 GGUF 模型的发布，这一点变得尤为重要，表明社区正在转向这些格式，以寻求潜在的更好性能或兼容性。
- 一位用户正在考虑是否将用于编程任务的模型从 q6KL 格式的 Qwen Coder3 80B 更新为新的 35B-A3B 模型。这突显了模型选择中常见的决策过程，即用户在衡量新模型的优势与其特定用例（如编程）以及官方文档中缺乏直接比较的情况。

### 2. Anthropic Distillation Controversy

- **[Anthropic 最近关于蒸馏的博客应该让所有人只想使用本地 open-weight 模型；这既可怕又反乌托邦](https://www.reddit.com/r/LocalLLaMA/comments/1rd8cfw/anthropics_recent_distillation_blog_should_make/)** (Activity: 949): **Anthropic** 关于[检测和预防蒸馏攻击](https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks)的博客文章强调了他们对抗未经授权模型蒸馏的方法，其中包括毒化输出以误导蒸馏者。这引发了对模型响应可靠性的担忧，特别是对于提交被公司视为“有问题”的 Prompt 的用户。该博客讨论了使用请求元数据（如 API keys）来识别和抵御这些攻击，表明了其对未经授权使用的积极态度。评论者对 Anthropic 方法的有效性和道德性表示怀疑，一些人批评将“蒸馏攻击”作为专业术语的使用，并质疑使用元数据跟踪用户的透明度。

    - Anthropic 的博客文章讨论了他们处理“蒸馏攻击”的方法，声称他们采取了除屏蔽请求之外的主动反制措施。据称他们毒化了输出以干扰这些攻击，这引发了对他们模型响应可靠性的担忧，尤其是对于提交被公司视为“有问题”的 Prompt 的用户。
    - 博客文章提到了“蒸馏攻击”，并建议 Anthropic 使用请求元数据（如 API keys）来识别和反击这些攻击。这导致了对其方法的透明度和道德性的怀疑，因为一些用户认为这种方法过于侵入，且缺乏明确的证据或数据来支持其说法。
    - Anthropic 对蒸馏攻击的立场被用来证明出口管制和限制芯片访问的合理性，他们认为这限制了直接模型训练和非法蒸馏。这被批评为控制 GPU 访问权的自私策略，一些用户因这些做法而对 Anthropic 的 API 进行财务投资表示遗憾。

- **[Anthropic：“我们已发现 DeepSeek、Moonshot AI 和 MiniMax 对我们的模型进行了工业级规模的蒸馏攻击。” 🚨](https://www.reddit.com/r/LocalLLaMA/comments/1rcpmwn/anthropic_weve_identified_industrialscale/)** (Activity: 6097): **该图片是来自 **AnthropicAI** 的推文，强调了一次重大的安全漏洞，其模型遭到了名为 **DeepSeek、Moonshot AI 和 MiniMax** 的实体的工业级规模蒸馏攻击。据称，这些实体创建了超过 `24,000` 个虚假账户，并与 Anthropic 的模型 **Claude** 进行了超过 `1,600` 万次交互，以提取其能力用于自己的模型训练。这一事件突显了在保护 AI 模型免受未经授权的数据提取方面的挑战，以及在竞争激烈的 AI 开发中存在滥用的可能性。** 评论者正在辩论 Anthropic 投诉的伦理影响，一些人指出了 Anthropic 自身数据实践中的讽刺之处，暗示其商业模式也涉及从各种来源蒸馏数据，有时并未获得明确授权。

- 讨论引发了关于 Anthropic 数据集创建伦理影响的质疑，暗示其可能涉及在没有适当权利的情况下从各种来源 distillation 数据。这反映了 DeepSeek 和 Moonshot AI 等公司的行为，这些公司被指控对 Anthropic 的模型进行了“工业规模的 distillation 攻击”。讽刺的是，Anthropic 的商业模式可能同样依赖于来自他人的数据 distillation。
- “distillation attacks”一词受到了批评，一些人认为这些公司仅仅是按预期在大规模使用 Anthropic 的 API。这引发了一场辩论，即这种使用方式究竟构成了攻击，还是仅仅是一种合法但具有侵略性的服务使用行为。对话突显了依赖开放数据访问的商业模式与 AI 模型的专有性质之间的紧张关系。
- 有人呼吁 DeepSeek 和 MiniMax 等公司进行更具侵略性的 distillation 尝试，这暗示了一个模型改进由这种做法驱动的竞争格局。这反映了更广泛的行业趋势，即快速迭代和模型增强通常是通过利用现有模型实现的，有时会引发伦理和法律挑战。

- **[人们想错了；Anthropic 并不在意 distillation，他们只是想反击关于中国开源模型正在赶超闭源前沿模型的言论](https://www.reddit.com/r/LocalLLaMA/comments/1rd2x61/people_are_getting_it_wrong_anthropic_doesnt_care/)** (热度: 977)：**图片展示了 Alek Dimitriev 的一条推文以及 **Anthropic** 针对开源模型从其 Claude 模型进行 distillation 问题的回应。讨论集中在“中国开源模型正在赶超闭源前沿模型”这一说法，以及 Anthropic 声称的几家实验室发起的工业级 distillation 攻击。帖子指出，Anthropic 的重点不在于 distillation 本身，而是在于反击那种认为中国模型可以在不进行 distillation 或窃取模型权重的情况下达到其能力的言论。这被视为一种战略举措，旨在影响投资者和美国政府对中国实施更多限制，以防止技术转移。** 评论者辩论了中国实验室的创新能力，一些人认为中国实验室确实具有创新性，而不仅仅是在 distillation 模型。其他人则强调了开源模型和 distillation 之外的创新的重要性，并引用了中国实验室的各种研究论文作为其贡献的证据。

- Ok_Knowledge_8259 认为 Anthropic 的方法缺乏显著的竞争优势或“护城河（MOAT）”，并建议更好模型的关键在于扩展干净数据、更多数据和强化学习 (RL)。他们强调，像 DeepSeek 这样的中国模型发布速度很快且表现良好，表明创新并不局限于闭源模型。该评论者还提到 “seed dance” 是视频技术中 SOTA (state-of-the-art) 的创新。
- Sagyam 提供了一系列技术论文，以反驳 Anthropic 仅关注 distillation 的说法。这些论文包括诸如 “DeepSeek-OCR”、“mHC”、“DeepSeek Sparse Attention”、“Muon Clip Optimizer and agentic post training”、“Lightning Attention” 和 “Qwen3 Omni Multimodality” 等创新。这表明在简单的 distillation 之外，还存在着持续的研究和开发，展示了 AI 技术的多样化进步。
- awebb78 批评了中国实验室缺乏创新的观点，强调他们不仅在 AI 模型方面，而且在机器人技术方面也做出了重大贡献。这一评论强调了认可来自中国研究实验室创新工作的重要性，而这些工作在西方视角主导的讨论中往往被忽视。

### 3. Liquid AI LFM2-24B-A2B 模型发布

- **[Liquid AI 发布 LFM2-24B-A2B](https://www.reddit.com/r/LocalLLaMA/comments/1rdi26s/liquid_ai_releases_lfm224ba2b/)** (活跃度: 320): **Liquid AI 发布了 LFM2-24B-A2B，这是一个拥有 240 亿参数的稀疏混合专家 (MoE) 模型，其中每个 token 激活 20 亿参数。该模型是 LFM2 系列的一部分，该系列规模已从 350M 扩展到 24B 参数，展示了在不增加单 token 计算量的情况下实现有效扩展的能力。其架构包含 40 层，每个 MoE 块有 64 个专家并采用 top-4 路由，设计运行在 32GB RAM 上，非常适合高端消费级设备。它支持通过 llama.cpp, vLLM 和 SGLang 进行推理，并提供多种 GGUF 量化版本。基准测试显示随着模型规模扩大，质量呈现对数线性提升，目前已在 Hugging Face 上开放权重。** 评论者对该模型的性能表示兴奋，特别是与 qwen3 coder 等其他模型的对比。人们也对更详细的基准测试以评估其能力表现出兴趣。描述中一个幽默的拼写错误被注意到，突显了该模型快速的边缘推理能力。

    - Liquid AI 的 LFM2-24B-A2B 模型以其快速的边缘推理能力而备受关注，在 AMD CPU 上达到 `112 tokens per second`，在 H100 GPU 上达到 `293 tokens per second`。它的设计旨在适应 `32 GB RAM`，并从发布第一天起就支持 llama.cpp, vLLM 和 SGLang 等框架，表明其专注于广泛的兼容性和高效的资源利用。
    - LFM2-24B-A2B 模型缺乏详细的基准测试，这导致了一些用户的怀疑。虽然该模型的潜力受到称赞，但缺乏全面的性能数据（尤其是与 Qwen3 Coder 等竞争对手相比）让那些考虑切换到该模型的人感到担忧。
    - LFM2-24B-A2B 模型到目前为止已经训练了 `17 trillion tokens`，预训练仍在进行中。本次发布被视为预览版，预计会推出更新版本 LFM2.5-24B-A2B，其中将包含额外的后期训练和强化学习，这表明当前模型尚未完全优化。

  - **[你做是蒸馏，我做是训练。](https://www.reddit.com/r/LocalLLaMA/comments/1rcvimv/distillation_when_you_do_it_training_when_we_do_it/)** (活跃度: 3433): **该图片是一个迷因 (meme)，幽默地强调了 AI 社区在模型蒸馏 (Distillation) 方面感知的双重标准。它暗示，虽然蒸馏在别人做时会受到批评，但当内部将其用作“训练数据”时就被认为是合法的。这反映了关于使用蒸馏技术的伦理和透明度的持续争论，尤其是在大型 AI 模型的背景下。评论进一步讨论了蒸馏的影响，指出小型、低成本模型通常依赖于从大型模型中蒸馏，并质疑了当蒸馏可以被用来复制私有模型时，这些模型的防御性。** 评论者强调了 AI 社区在蒸馏实践中感知到的伪善，质疑了像 Anthropic 这样公司的伦理立场。他们认为低成本模型的真正“秘密武器”通常是从大型模型中蒸馏出来的，并对在前沿模型极易被蒸馏的情况下其私有性质表示怀疑。

    - 讨论强调了蒸馏的实践，即从较大的模型衍生出小型、低成本的模型。这一过程通常被视为这些模型的“秘密武器”，使它们在没有与从头训练大型模型相关的高昂成本的情况下表现良好。其含义是，如果前沿模型可以轻易通过蒸馏被复制，那么它们的竞争优势就会被削弱，从而引发对投资此类模型的可防御性的质疑。
    - 存在对 Anthropic AI 开发方法的批评，暗示他们没有为开源社区做出贡献，并且严重依赖现有数据集，可能不顾法律问题。这引发了关于数据使用和模型训练过程透明度的伦理担忧。此外，还有人批评 Anthropic 对开源模型的立场及其对政策和审查的影响，一些人认为这与他们自己的做法相比是虚伪的。
    - 对话涉及了使用维基百科等公开数据训练 AI 模型的伦理和法律影响。这种做法在 AI 实验室中很常见，但它引发了关于此类数据的所有权和相关权利的问题。辩论表明，需要更明确的 AI 训练数据使用指南和法规，以确保公平和合法的实践。

- **[冷知识：Anthropic 从未开源过任何 LLMs](https://www.reddit.com/r/LocalLLaMA/comments/1rcseh1/fun_fact_anthropic_has_never_opensourced_any_llms/)** (Activity: 938): **Anthropic** 尚未开源其任何大语言模型 (LLMs)，包括 Claude，这限制了外部对其 Tokenizer 效率的分析，特别是在多语言环境下。相比之下，**OpenAI** 已经开源了其 Tokenizer 以及像 `gpt-oss` 这样的模型，而 **Google** 也分享了其 Gemma 和 Gemini 模型使用相同的 Tokenizer。考虑到 AI 研究领域向透明度和协作发展的趋势，Anthropic 缺乏开源贡献这一点非常值得关注。评论者强调了 Anthropic 在强调安全性的同时却不对开放研究做出贡献的讽刺意味，而开放研究被视为推动 AI 安全的关键。此外，还有人将其与 OpenAI 更开放的做法进行了对比，暗示了在社区贡献方面的差距。

    - TheRealMasonMac 指出了 Claude 模型中的一个技术局限，即它们无法输出排版用的弯引号（如 “ 或 ‘）。这种局限可能会导致依赖这些特定 Token 的代码出现问题，该评论者就遇到了代码崩溃的情况。这指向了模型 Tokenization 能力中一个潜在的改进领域。

  - **[虚伪？](https://www.reddit.com/r/LocalLLaMA/comments/1rcrb2k/hypocrisy/)** (Activity: 748): **图片揭示了 AI 社区中的一个重大问题：**DeepSeek**、**Moonshot AI** 和 **MiniMax** 等公司被指控对 **Anthropic** 的 AI 模型 Claude 进行了工业级规模的蒸馏攻击。据称，这些实体创建了超过 `24,000` 个虚假账号，并执行了 `1,600 万` 次交互，以提取并复制 Claude 的能力用于自己的模型。这引发了关于开发 AI 模型所用方法以及 AI 行业知识产权保护的伦理担忧。** 一位评论者质疑了这些公司的伦理立场，暗示他们可能使用了类似的方法来获取训练数据。另一位评论者对 z.ai 未被提及感到惊讶，认为其 GLM 系列可能也涉及类似的行径。

    - 'archieve_' 的评论提出了一个关于 AI 模型训练数据来源的关键问题。这是 AI 伦理和法律中的一个重要议题，因为数据的来源会影响模型的偏差、合法性和性能。了解数据来源对于 AI 开发的透明度和问责制至关重要。
    - 'semangeIof' 提到了 GLM 系列模型在被提示（prompted）时会自称是 Claude 的行为。这突显了模型身份和响应准确性的潜在问题，可能会影响用户信任和 AI 系统的感知可靠性。这种行为可能表明模型的训练或 Prompt 处理机制存在缺陷。
    - 'roxoholic' 提到的“工业级规模蒸馏攻击”是指一种将大模型蒸馏为小模型的方法，这可能会引发对知识产权和模型安全的担忧。这种技术可以用于在不直接访问原始模型的情况下复制模型，对专有 AI 技术构成了挑战。


## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Anthropic vs. DeepSeek Distillation Controversy

  - **[Anthropic 指控 DeepSeek、Moonshot AI (Kimi) 和 MiniMax 创建了超过 24,000 个虚假 Claude 账号，并从 1600 万次交流中蒸馏训练信息。](https://www.reddit.com/r/singularity/comments/1rcpdwz/anthropic_is_accusing_deepseek_moonshot_ai_kimi/)** (Activity: 4142): **Anthropic** 指控 **DeepSeek、Moonshot AI (Kimi) 和 MiniMax** 对其 AI 模型 Claude 策划了一场大规模的数据提取行动。据 Anthropic 称，这些公司涉嫌创建了超过 `24,000` 个虚假账号，与 Claude 进行了 `1,600 万` 次交互，从而有效地窃取其训练数据以改进自家的 AI 模型。这一事件突显了 AI 开发中对数据安全和知识产权的重大担忧，因为它涉及未经授权的访问以及对专有 AI 能力的潜在滥用。评论者们强调了 AI 公司在抱怨数据被盗时的讽刺性，因为他们自己也经常在不支付报酬的情况下使用公开数据。这反映了当前关于数据所有权和伦理 AI 训练实践的持续争论。

- Free_Break8482 指出了 Anthropic 指控中的讽刺之处，指出 AI 公司经常在公开可用的互联网数据上训练其模型，这引发了关于此类数据所有权和权利的问题。这突显了关于将公开信息用于 AI 训练的伦理争议。
- ImmediateDot853 质疑 Anthropic 对开源社区的贡献，暗示虽然 Anthropic 的 AI 受益于开源流量，但它可能没有通过资助或支持开源项目来进行回报。这涉及到了 AI 生态系统中企业责任和互惠性的更广泛问题。
- adalgis231 批评了 Anthropic 等 AI 公司表现出的虚伪，这些公司可能在不补偿创作者的情况下使用公开的知识产权，却指责他人盗窃。这一评论反映了围绕 AI 训练数据和知识产权的复杂法律及伦理格局。

- **[又来了。DeepSeek R1 简直是对 OpenAI 模型的复制粘贴。他们被封禁了，现在又轮到 Anthropic。骗局！](https://www.reddit.com/r/OpenAI/comments/1rcpfeg/here_we_go_again_deepseek_r1_was_a_literal_copy/)** (活跃度: 2519)：**图像突显了一个严重问题，即 DeepSeek、Moonshot AI 和 MiniMax 等公司被指控对 **Anthropic** 的 AI 模型进行了工业规模的蒸馏攻击（distillation attacks）。这些攻击包括创建超过 `24,000` 个虚假账号，并与 **Claude** 模型进行了 `1,600 万` 次交互以提取其能力。这一过程被称为蒸馏（distillation），通常用于创建更小、更高效的模型，但在这里被滥用以绕过安全机制，并可能滥用 AI 能力。**Anthropic** 正呼吁协同努力对抗这些复杂的攻击，这些攻击存在移除 AI 模型重要安全措施的风险。** 评论反映了对 AI 公司道德标准的讽刺和批评，一些用户嘲笑数据盗窃的想法，另一些人则指出，被指控存在不道德行为的公司本身也是类似行为的受害者，这充满了讽刺。

- **[Anthropic 刚刚发布了 DeepSeek、Moonshot 和 MiniMax 大规模蒸馏 Claude 的证据。2.4 万个虚假账号，1,600 万次以上交互。](https://www.reddit.com/r/ClaudeAI/comments/1rd1j8u/anthropic_just_dropped_evidence_that_deepseek/)** (活跃度: 2751)：****Anthropic** 发布了一份报告，详细说明了包括 **DeepSeek**、**Moonshot** 和 **MiniMax** 在内的三家中国 AI 实验室，如何利用 `24,000` 个虚假账号和超过 `1,600 万` 次交互，系统地从其模型 **Claude** 中提取能力。**DeepSeek** 尤其显著地利用 Claude 逐步解释其推理过程以创建训练数据，其中包括政治敏感内容。**MiniMax** 进行了 `1,300 万+` 次交互，并迅速适应了新的 Claude 模型。报告强调，安全特性在蒸馏模型中无法很好地转移，导致在细微场景中存在潜在风险。这种情况突显了模型差异性作为蒸馏后独立推理标志的价值。** 评论者指出了 Anthropic 处境的讽刺性，指出虽然他们面临虚假账号问题，但他们自己也使用了广泛的数据源进行训练。还有一种观点认为，构建关键系统的人员将避免使用蒸馏模型，因为其安全特性受损。

    - VanOrten 指出了 Anthropic 的一个重大安全疏忽，注意到虽然合法用户因使用 VPN 而面临账号注销，但系统却未能检测并阻止 24,000 个虚假账号进行超过 1,600 万次交互。这引发了对 Anthropic 账号验证和欺诈检测机制稳健性的质疑。
    - DauntingPrawn 讨论了模型训练数据的伦理考量，指出像 Anthropic、OpenAI 和 Google 这样的大型 AI 公司在历史上都使用了大量未经授权的数据进行训练。这一评论表明，蒸馏模型的做法虽然备受争议，但被一些人视为 AI 社区中一种重新平衡天平的方式。
    - cororona 讽刺地评论了训练模型的经济学，暗示支付 tokens 费用是一种低效的方法，相比之下通过不太合法的手段（如盗版）获取数据更划算。这突显了围绕 AI 训练数据获取成本和伦理的持续辩论。

- **[Anthropic：“我们已发现 DeepSeek、Moonshot AI 和 MiniMax 对我们的模型进行了工业规模的蒸馏攻击。”](https://www.reddit.com/r/ClaudeCode/comments/1rcp658/anthropic_weve_identified_industrialscale/)** (Activity: 1846): **Anthropic** 公开指责 **DeepSeek**、**Moonshot AI** 和 **MiniMax** 对其 AI 模型进行了“工业规模的蒸馏攻击”。这些攻击涉及创建超过 `24,000` 个虚假账户与 Anthropic 的模型 **Claude** 进行交互，产生了超过 `16 million` 次对话。其目的是提取并复制 **Claude** 的能力，以增强他们自己的模型。这一事件凸显了 AI 模型安全和知识产权保护面临的持续挑战，因为各公司都在寻求保护其专有技术免受未经授权的使用和复制。评论反映了关于使用专有 AI 模型进行训练的伦理辩论，并将其与对受版权保护材料进行训练的更广泛问题进行了类比。一些用户讽刺地指出 Anthropic 投诉中的讽刺意味，暗示 AI 社区在数据使用方式上存在双重标准。

    - 该讨论提出了对 AI 模型的蒸馏攻击是否类比于对受版权保护材料的训练这一问题。这种比较暗示了一个潜在的伦理和法律灰色地带，因为两者都涉及使用现有的知识产权来创建新模型。这意味着，如果对受版权保护材料的训练存在争议，那么对专有模型的蒸馏攻击也可能如此。
    - “攻击（attack）”一词引发了争论，一些人认为模型学习现有模型类似于人类的学习过程。这种观点挑战了将蒸馏视为恶意行为的观念，认为它可以被看作是 AI 发展过程中自然的一部分，即模型通过相互学习而进化，类似于人类从现有知识中学习的方式。
    - 提到的“24k fake accounts”凸显了蒸馏攻击涉及的运营规模。这一数字与大型 Web 服务的典型活动进行了比较，暗示此类攻击可能比最初感知的更普遍且更易于管理。这表明，许多大规模服务已经具备处理此类活动的基础设施。


### 2. AI 工具对遗留系统和行业的影响

  - **[IBM 成为 Anthropic 的最新受害者，在旨在使 COBOL 遗留代码现代化的 Claude Code 工具发布后，IBM 股价暴跌 10%。COBOL 是一种拥有 66 年历史的编程语言，至今仍被广泛使用；美国约 95% 的 ATM 交易使用 COBOL 代码处理](https://www.reddit.com/r/singularity/comments/1rcz68x/ibm_is_the_latest_company_victim_of_anthropic/)** (Activity: 467): **Anthropic** 宣布推出一款名为 *Claude Code* 的新工具，旨在实现遗留 **COBOL** 代码的现代化，而这些代码对于处理美国 `95%` 的 ATM 交易仍然至关重要。这一公告导致 **IBM** 股价下跌了 `10%`，突显了市场对遗留系统潜在中断的敏感性。然而，该工具并不是一项新技术，而是一篇暗示其在更新 **COBOL** 系统方面效用的博客文章，这可能被市场误读了。评论者指出，许多现代银行系统仍然依赖 **COBOL**（通常封装在较新的技术中），且鉴于缺乏该工具有效性的具体证据，市场的反应可能过早。人们对 Anthropic 工具的实际影响持怀疑态度，因为股价反应与公告内容相比显得不成比例。

    - Onipsis 的评论强调，Anthropic 关于 Claude Code 的公告并非发布新工具，而是一篇暗示其在 **COBOL** 现代化方面潜在效用的博客文章。这导致了市场的过度反应，造成 IBM 股价下跌 10%。该评论强调了 **COBOL** 在基础设施中的关键作用，以及熟悉该语言的专业人员数量不断减少，这使得现代化工作虽然意义重大但充满挑战。
    - Milo-75 讨论了现代化项目的复杂性，特别是在严重依赖 **COBOL** 的银行和 ATM 系统中。该评论认为，尽管像 Claude Code 这样的 AI 工具可能会减少 25% 的项目时间，但企业在处理此类关键系统时仍将依赖 IBM 的专业知识。这暗示虽然 IBM 从这些项目中获得的收入可能会减少，但其利润率可能会提高，从而使他们能够承担更多项目。
    - Stabile_Feldmaus 提出，尽管市场对 Anthropic 专门工具的公告反应负面，但仍缺乏关于这些工具有效性的明确反馈。该评论对这些工具对 IBM 业务的直接影响表示怀疑，因为这些工具在实际应用场景中的实际性能和效用仍未得到证实。

- **[Anthropic 刚刚发布了一款针对 COBOL 的 AI 工具，IBM 股价下跌 13%](https://www.reddit.com/r/ClaudeAI/comments/1rddo3m/anthropic_just_dropped_an_ai_tool_for_cobol_and/)** (热度: 880): **Anthropic** 发布了一款新的 AI 工具，旨在分析和现代化 COBOL 代码库，这些代码库对银行、航空和政府部门的许多遗留系统至关重要。该工具旨在识别风险并降低现代化成本，这可能会威胁到 **IBM** 管理这些系统所获得的收入。该公告导致 IBM 股价大幅下跌 `13%`，反映了市场对 IBM 主机业务受影响的担忧。然而，一些分析师认为，尽管存在现有的迁移替代方案，企业仍继续依赖 IBM，这表明市场的反应可能过于夸张。评论者对在关键基础设施中依赖 AI 表示怀疑，其中一位指出在这种背景下进行 'vibe coding' 的潜在风险。另一位则认为市场的反应可能是“膝跳反射”式的，暗示需要从更长远的角度来看待。

    - Anthropic 针对 COBOL 的 AI 工具的推出被视为加速遗留系统迁移的潜在催化剂，但此类迁移相关的风险仍然很大。银行和其他机构历来避免现代化，因为错误的风险是灾难性的，而 AI 的“幻觉”倾向意味着仍然需要人工监督。因此，虽然 AI 可能会加快进程，但它尚未消除人工审查的瓶颈，尤其是在关键基础设施应用中。
    - 像 Anthropic 这样的 AI 工具带来的真正威胁是针对专业服务领域，特别是像 IBM 这样从管理和迁移遗留系统中获取丰厚收入的公司。AI 可以显著减少对外部承包商的需求，从而给 IBM 的专业服务业务带来风险。即使对关键系统的直接影响有限，这种转变也可能导致对遗留系统管理相关服务的需求减少。
    - IBM 的股价下跌归因于对其专业服务收入的潜在影响，而不是对其制造或技术核心业务的直接威胁。这里的类比是，这种颠覆类似于影响了“马鞭抛光剂”的销售，而不是“马鞭”本身，突显了对 IBM 业务模式间接但重大的影响。

  - **[Claude 是更好的产品。$20 计划中的两项复合使用限制是 OpenAI 依然能赚我钱的原因。](https://www.reddit.com/r/ClaudeAI/comments/1rcmvj5/claude_is_the_better_product_two_compounding/)** (热度: 1217): **该 Reddit 帖子讨论了用户在书籍编辑等任务中对 **Claude** 优于 **ChatGPT Plus** 的偏好。然而，由于 **Claude Pro** 限制性的使用配额（包括 `5 小时滚动会话窗口`和可能导致用户被锁定数日的`周上限`），该用户仍留在 ChatGPT Plus。该用户强调，这些限制使得 Claude Pro 在其高强度的日常使用（涉及多个项目的长周期迭代会话）中变得不切实际。他们建议需要一个介于 `$20 到 $100` 之间更灵活的定价层级，以适应严肃的日常用户而不会频繁被锁定。**评论者指出，虽然 **Anthropic** 的定价策略被认为更准确，但由于其侧重于 B2B，对个人用户并不友好。一些用户认为每月 100 美元的档位因其生产力优势是合理的，而另一些用户则对 Claude 的限制感到沮丧，并考虑换回 ChatGPT。

    - Helkost 讨论了 AI 公司的定价策略，指出虽然 inference 成本正在下降，但行业定价尚未覆盖这些成本。他们强调，与其它公司相比，Claude 背后的公司 Anthropic 对其产品的定价更为准确，但也强调 Anthropic 的主要重点是 B2B 而非个人消费者。
    - turtle-toaster 指出，每月 20 美元的 Pro 计划并非为重度使用而设计，而是作为一种鼓励升级的入门级产品。他们认为，由于 compute 成本，这个价格点的无限计划在财务上是不可持续的，并建议每月 60 美元的计划对于严肃用户可能更可行。
    - FaceOnMars23 对当前的定价模式表示沮丧，指出目前缺乏能更好服务用户的选项。他们提到将免费 AI 工具与 Claude 结合使用来管理成本和任务，并批评了对定价模式建设性反馈的轻视态度。

### 3. Gemini 与 Qwen 模型进展

  - **[Gemini 3.1 Pro 在 2 小时内开发出这款 Metal Gear Solid 游戏](https://www.reddit.com/r/Bard/comments/1rd0kkz/gemini_31_pro_created_this_metal_gear_solid_game/)** (热度: 120): **该帖子重点介绍了使用 **Gemini 3.1 Pro** 在短短 `2 小时` 内开发出一款 Metal Gear Solid 游戏的过程。虽然帖子缺乏详细的技术信息，但它表明了一个快速开发的过程，可能利用了 Gemini 3.1 Pro 先进的 AI 能力。文中提到的 'SFX' 暗示音效是一个显著特征，但未提供具体的技术栈或实现细节。** 评论反映了粉丝们的积极反响，一位用户作为 Metal Gear 粉丝表达了极大的热情。然而，目前还缺乏关于开发过程或所用工具的技术辩论或详细讨论。


  - **[Gemini 应用新增视频模板以快速启动生成](https://www.reddit.com/r/Bard/comments/1rctgtx/gemini_app_adds_video_templates_to_quick_start/)** (热度: 72): ****Gemini** 在其应用中引入了视频模板，使用户能够快速开始视频生成。该功能预计将通过简化创作过程（特别是针对社交媒体内容）来增强用户参与度。根据 [9to5Google 的文章](https://9to5google.com/2026/02/23/gemini-video-templates/)，此次更新可能会利用应用现有的 AI 能力来优化视频制作，尽管并未披露关于实现方式或所用 AI 模型的具体技术细节。** 评论者对 **Veo 3.1** 表示不满，将其描述为“几十年前的老模型”，并对其性能表示怀疑。不过，人们普遍预期这一新功能将在社交媒体平台上获得关注。


  - **[适用于 MLX 的 Qwen 3.5 宛如一场工业革命](https://www.reddit.com/r/Qwen_AI/comments/1rcqezx/qwen_35_for_mlx_is_like_its_own_industrial/)** (热度: 98): **该帖子讨论了 **Qwen 3.5** 模型在 **Mac Studio M3** 上使用 `4-bit` 设置的表现，强调了其令人印象深刻的速度和质量。一位用户报告称达到了 `34-35 tokens per second`，强调了该模型即使在“非思考模式”下的高效率。该模型的 Prompt 处理被描述为几乎瞬时完成，这表明在本地机器学习任务的延迟和吞吐量方面有了显著改进。** 有用户询问 **Hugging Face** 上是否提供 Qwen 3.5 4-bit 模型，表明了对便捷部署方案的需求。

    - 适用于 MLX 的 Qwen 3.5 模型展示了惊人的速度，处理速度达到 `34-35 tokens per second`，这对于此类模型来说被认为是非常快的。此外，Prompt 处理几乎是瞬间完成的，增强了其在实时应用中的可用性。
    - MLX 版本 Qwen 3.5 的一个显著局限是缺乏 Vision（视觉）能力，这限制了它只能处理文本输入。对于需要多模态输入处理的用户来说，这是一个重大缺陷，因为当前的 MLX 设置不支持 Vision 任务。

  - **[将 Qwen3-VL-2B-Instruct 连接到我的监控摄像头，效果极佳](https://www.reddit.com/r/Qwen_AI/comments/1rdnzbe/connected_qwen3vl2binstruct_to_my_security/)** (热度: 94): **该帖子讨论了将 **Qwen3-VL-2B-Instruct** 模型与安防摄像头馈送集成的案例，强调了它能够提供详细的场景叙事描述（例如邮递员投递邮件），而不仅仅是检测物体。该模型经过 `IQ2` 量化，大小约为 `0.7 GB`，其场景理解能力令人印象深刻。硬件配置包括 **MacBook M3 Air 24GB** 和 **SharpAI Aegis** 平台，模型和 Vision Projector 总计约 `1.4 GB`。流程包括通过内置浏览器选择模型、下载、使用支持 Metal/CUDA 加速的 llama-server 进行服务，并观察实时处理日志。** 评论者对小型 Qwen VL 模型的潜力表示热切期待，有人指出它们的变革潜力，另一位则表达了对未来 Qwen 3.5 模型的期待。此外，还有将该项目与 Django 集成的兴趣。


---

# AI Discord 简报

> 由 Gemini 3.1 Pro Preview Nov-18 生成的摘要之摘要

**主题 1. Anthropic 的“工业级”蒸馏风波与 Jailbreak 漏洞**

*   **Anthropic 公开指责中国 API 蒸馏者**：Anthropic 在一篇 [Anthropic 工业级攻击博文](https://x.com/anthropicai/status/2025997928242811253)中公开指责 DeepSeek、Moonshot AI 和 MiniMax 利用 **24,000 多个虚假账号**进行了 **1600 万次交互**，以蒸馏 **Claude**。AI 社区对此指责大多嗤之以鼻，认为其表现得“气急败坏”，并指出讽刺的是 Anthropic 自身也有爬取数据来构建其 Foundation Models 的历史。
*   **Claude Max 泄露内部推理过程**：通过 **OpenClaw** 使用 **Claude Max** 的用户遇到了一个严重 Bug，模型将其内部思维过程直接传导至实时对话会话中。工程师发现，虽然可以通过运行 `/reasoning off` 命令临时修复该泄露，但 **Opus 4.6** 和 **Sonnet 4.6** 消耗用户额度的速度依然惊人。
*   **Kimi 2.5 越狱引发“宪法”混乱**：黑客成功破解了 **Kimi 2.5**，剥离了其防护栏（Guardrails），打造出了一个“没有宪法约束烦恼的中国版 Claude”。与此同时，研究人员正通过 **ENI** 提示词利用 **Gemini 3.1 low**，触发其安全防护栏与合规性之间的内部“拉锯战”，迫使模型输出受限内容。

**主题 2. 新前沿模型：Qwen 3.5 统治榜单，GPT-5.3 Codex 发布**

*   **Qwen 3.5 横扫开源权重排行榜**：阿里巴巴发布了重磅更新 [Qwen3.5-35B-A3B-Base 权重](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base)，尽管其占用空间显著缩小，但性能却超越了旧款 **235B** 模型，令开发者印象深刻。巨大的 **Qwen3.5-397B-A17B** 变体也冲击了 Code Arena 排行榜，夺得**全球第 17 位**，与 **GPT-5.2** 和 **Gemini-3-Flash** 等闭源巨头并驾齐驱。
*   **OpenAI 悄然向大众部署 GPT-5.3-Codex**：OpenAI 正式在 **OpenRouter** 上面向所有开发者 API 推出了 [GPT-5.3-Codex](https://openrouter.ai/openai/gpt-5.3-codex)，定价极具攻击性：输入每百万 Token **$1.75**，输出每百万 Token **$14**。OpenRouter 随即整合了该模型，并推出了新的 `openrouter/free` 端点，可自动将开发者请求路由至零成本的回退模型。
*   **GPT-OSS 20B 在消费级 GPU 上实现科幻级速度**：得益于仅依赖 **3B 活跃参数**的 **Mixture of Experts (MoE)** 架构，工程师在标准 **RTX 5090** 上测得新款 **GPT-OSS 20B** 模型速度达到惊人的 **260 t/s**。该模型可完全装入高速 **VRAM** 并原生支持 **Flash Attention**，对于运行消费级硬件的本地推理爱好者来说是一个巨大的胜利。

**主题 3. 系统级工程、硬件扩展与内核优化**

*   **MatX 融资 5 亿美元打造终极 LLM 芯片**：MatX 获得 **5 亿美元 B 轮融资**，用于开发 **MatX One LLM 芯片**。根据 [MatX 融资公告](https://x.com/reinerpope/status/2026351870852358492)，该芯片采用可分割的脉动阵列（Systolic Array），结合了 SRAM 级的低延迟与 **HBM 长上下文支持**。与此同时，Meta 签署了一份为期五年、部署 **6GW AMD 基础设施**的协议，利用全新的 **RRCLLX** 协议大幅优化 **AMD MI300X** 的多 GPU 通信。
*   **预编译 FlashAttention 3 Wheel 文件上线生产环境**：AI 工程师终于可以告别繁琐的自定义编译了，因为 [预编译 Flash Attention 3 Wheel 文件](https://download.pytorch.org/whl/flash-attn-3/) 现已正式面向 **CUDA 12.6+** 和 **13** 发布。这些 **LibTorch ABI 稳定**的版本支持 **x86/ARM CPU** 以及 **Linux/Windows OS**，彻底缩短了运行 **Python 3.10+** 和 **PyTorch 2.9+** 开发者的环境搭建时间。
*   **Llama.cpp 更新导致 Qwen 模型及 VRAM 分配崩溃**：来自 master 分支的最新 **llama.cpp** 构建版本出现了致命的 *Failed to read magic* 错误，完全无法解析 **Qwen3.5** 模型的 **GGUF Header**。工程师将该 Bug 定位到最近的一个溢出修复，该修复无意中阻止了正确的 **VRAM** 分配，迫使开发者紧急回滚到 **8145** 版本以恢复功能。

**主题 4. 工具、Agentic 工作流与开发者基础设施**

*   **Cursor Cloud Agents 免费发布**：**Cursor** 正式推出了其全新的 **Cloud Agents** 功能，为开发者提供完全免费的云环境，可直接从编辑器运行测试、执行终端命令并部署实时演示 ([Cursor onboarding link](https://cursor.com/onboard))。然而，社区立即遇到了执行限制，并开始积极游说开发者提供一种安全的方式，允许 **Agent** 绕过提升的 **sudo** 密码限制。
*   **Aider 开发者遭遇 Diff 格式限制**：流行的 **Aider** CLI 工具在处理复杂的多文件代码库编辑时遇到了困难，由于 Diff 格式损坏，迫使开发者不得不手动分块处理更改。工程师们通过提交 [Aider GitHub issue #3603](https://github.com/Aider-AI/aider/issues/3603) 升级了该工具的局限性反馈，恳求支持原生的 **git submodule**，而该框架目前完全忽略了这一点。
*   **Tiny-GPU 编译器将 C 语言带入 Verilog**：硬件黑客发布了 [tiny-gpu-compiler project](https://github.com/gautam1858/tiny-gpu-compiler)，这是一个具有教育意义的**基于 MLIR 的编译器**，它将类 C 的内核语言直接翻译成 **16-bit binary instructions**。该流水线针对完全由 **Verilog** 编写的自定义开源 **GPU** 硬件，并附带一个分步可视化工具，用于精确的执行分析。

**Theme 5. Benchmarking Turmoil and Evaluator Shakeups**

*   **OpenAI 因数据污染弃用 SWE-Bench Verified**：**OpenAI** 在发现前沿模型经常纯粹基于记忆的测试 ID 复现出完全相同的任务解决方案后，正式弃用了流行的 **SWE-Bench Verified** 基准测试。根据其 [SWE-bench 弃用公告](https://x.com/OpenAIDevs/status/2026025368650690932)，工程师们证明，剩余未解决的问题中约有 **60%** 存在结构性缺陷，使得继续进行基准测试完全是算力浪费。
*   **EleutherAI 紧急修复 Pythia HuggingFace 副本问题**：研究人员发现了一个严重漏洞，即 [EleutherAI's pythia-2.8b](https://arxiv.org/abs/2309.23024) 在 **Hugging Face Hub** 上无论选择哪个修订步骤，都提供相同的模型权重。在确认之前的上传被错误地去重（deduped）后，团队立即启动了重新训练，并部署了全新修正的 [Pythia-14m](https://huggingface.co/stellaathena/pythia-14m) 和 [Pythia-31m](https://huggingface.co/stellaathena/pythia-31m) 模型。
*   **LMArena 过滤器禁用“掷骰子”提示词**：**LMArena** 的审核过滤器完全失控，自动拒绝了非常温和的提示词（如简单的“掷骰子”），仅因为其中包含被标记的触发词（如 *liar*）。开发者承认封禁过于激进，目前正拼命测试**基于 LLM 的过滤**以及放宽 [OpenAI moderation API](https://developers.openai.com/api/docs/guides/moderation/) 阈值，以恢复评估队列的正常秩序。

---

# Discord: High level Discord summaries

## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Deepseek 统治免费 AI 领域**：成员们推荐 **Deepseek** 为目前最好用的免费 AI，提供完全免费的使用体验。
   - 工程师们正利用这款免费 AI 来进行自托管项目并创造新颖用途。
- **Chef 遭受严重漏洞影响**：一名用户报告在 **Chef** 中发现了 *4 个严重漏洞*，并声称该公司没有认真处理这些问题，同时链接到了 [Convex security page](https://www.convex.dev/security)。
   - 还有关于潜在诈骗策略的警告，即公司可能会在不提供署名或补偿的情况下使用漏洞详情。
- **AI 几乎破解 VMP 保护的代码**：一名用户使用受 **VMP** (VMProtect) 保护的 crackme 挑战 **Claude**，它通过获取操作码并几乎破解了字节码，取得了显著进展。
   - 他们建议尝试 **Copilot**，并指出它利用先进的数字取证技术*重建了损坏的键盘记录器 .sys 文件*。
- **Kimi 2.5 越狱解锁全知 AI**：一名用户报告称，破解后的 **Kimi** 几乎可以详细回答任何问题，称其为*没有伦理限制烦恼的中国版 Claude*。
   - 该 AI 工具非常适合 **API**，因为其越狱（Jailbreak）很容易通过 **system prompt** 实现。
- **开发者仓库大清洗：文件标记狂潮**：一名开发者分享称，他们的整个代码库都触发了标记，对文件接受检查的数量感到惊讶。
   - 另一名成员指出，大多数用于个人测试的文件在 **3 天**后都会被标记，但他们正在尝试一种涉及浏览器注入的新方法，利用 AI 来可视化代码。

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen 3.5 引起轰动**：成员们正在积极测试 [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) 等 **Qwen 3.5** 模型，并对其质量和速度留下了深刻印象，强调它们在微调、in-context learning 和研究方面的实用性，而非直接交互。
   - 虽然最新的 **Qwen 122B 模型** 潜在支持本地编程，但免费的 **OpenCode 模型** 已经改变了他们的工作流。
- **GLM 模型在创意领域表现卓越**：用户发现 **GLM 模型**，特别是 **GLM-4.7-Flash**，在 Unsloth 上运行良好，尤其适用于创意写作任务。
   - 一位用户透露，他们为 **GLM 编程方案** 支付了 **40 美元**（**3 个月**）。
- **Llama.cpp 更新导致导入混淆**：在 **llama.cpp** 更新后，一些用户遇到了 `import missmatch` 问题，导致模型在未更新的情况下无法运行。
   - 一位用户解决了 Jinja 问题，并在 [此讨论](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/discussions/4) 中分享了修复方法。
- **DeepSeek 统治 Chatbot 锦标赛**：成员们庆祝 **DeepSeek** 在 Gotham 的 ChatBot 锦标赛中的表现，凸显了其顶级的 LLM 能力。
   - 其他人询问是否存在 **Deep Research Agent**，一些人澄清它具有 **DeepSearch 开关**。
- **LoRA 合并受 Key 不匹配问题困扰**：用户报告称，由于提取的 Key 不匹配（特别是 `lm_head.weight`），最新的 Unsloth 版本破坏了 **LoRA 合并**，详情见 [GitHub issue #4098](https://github.com/unslothai/unsloth/issues/4098)。
   - 该问题源于训练期间 `target_modules` 未包含 `lm_head`，导致合并时出现差异；通过在 `get_peft_model` 的 `target_modules` 中添加 `lm_head`，可以在 Colab 上复现此问题。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Video Arena 消失**：成员们注意到 **Video Arena** 已从 Discord 中移除，但仍可通过网站 [arena.ai/video](https://arena.ai/video) 访问。
   - 未给出移除的原因。
- **Gemini 图像预览触发速率限制**：用户报告在使用 **Gemini 3 Pro Image Preview** 时遇到 **429 Too Many Requests** 错误，表明该服务受到速率限制（rate limited）；[Google 的文档](https://cloud.google.com/vertex-ai/generative-ai/docs/error-code-429) 提供了更多细节。
   - 一位用户找到了图像上传的解决方法，即在提示词前加上 *"Modify the following image with the following: (The prompt)"*。
- **Reve 1.5 令人印象深刻并引发讨论**：**Reve 1.5** 的图像质量令用户印象深刻，一些人认为它的排名应该更高，尤其是在漫画上色方面。
   - 虽然有些人觉得 [reve.com](https://app.reve.com/) 网站很精美，但其他人注意到 1.5 版本中缺少图像编辑等限制。
- **Arena 的过滤器过于严格**：用户抱怨审核过滤器过于敏感，由于包含 "liar"（骗子）等术语，甚至会拦截掷骰子等无害内容。
   - 团队承认了这种过度行为，正在考虑采用 **LLM-based filtering** 或调整现有审核端点（如 [OpenAI 的 Moderation API](https://developers.openai.com/api/docs/guides/moderation/)）的阈值等方案。
- **Qwen3.5-397B-A17B 加入 Code Arena**：**Qwen3.5-397B-A17B** 已加入 Code Arena 排行榜，获得了 **前 7 名开源模型** 的地位，并在 **总榜排名第 17**。
   - 它的综合排名与 **GPT-5.2** 和 **Gemini-3-Flash** 等闭源模型持平。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 用户探讨 Sudo 命令**：一位用户询问了在 **Cursor** 中处理 `sudo` 命令的最佳方法，因为 Agent 目前不支持接管或密码输入。
   - 随后的讨论寻求将提升权限集成到编码工作流中的潜在解决方案。
- **开发者构建 “Vibe Coding App”**：一位成员正在开发一款 vibe coding 应用程序，该程序默认使用本地模型，但也允许通过 API keys 使用云端模型选项，而无需订阅。
   - 社区成员讨论了其潜在的市场吸引力，一些人对其相较于 **Cursor** 等现有工具的吸引力表示怀疑，并提到了潜在的稳定性担忧。
- **Gemini 面临不稳定性指控**：用户报告称，自 **3.1 Pro** 发布以来，**Gemini** 出现了连接问题和不稳定性。
   - 一些用户正在等待更稳定的版本，而另一些用户则提到，他们并未因遇到的错误而被计费。
- **规则引擎噩梦已解决，准备投入生产**：一位成员宣布解决了规则迁移和重构问题，并计划在 3-4 周内推出一款自动化相关流程的产品，并分享了规则引擎的截图。
   - 另一位成员对规则引擎的规模和复杂性做出了反应，称其为“噩梦”。
- **Cursor 推出免费 Cloud Agents**：**Cursor** 推出了 **Cloud Agents**，允许在云端环境中运行测试或演示，正如其 [官方网站](https://cursor.com/onboard) 所宣布的那样。
   - 目前 **Cloud Agents** 免费提供，尽管这种定价模式未来可能会发生变化。



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 与 Comet 推出语音功能**：根据 [此状态更新](https://fixvx.com/comet/status/2026384898802724878)，新的**语音模式升级**今天面向所有 **Perplexity** 和 **Comet** 用户推出。
   - 新的语音模式正同时向 **Perplexity** 及其姐妹产品 **Comet** 推出。
- **Pro 用户抗议 Perplexity Pro 限制**：用户报告 **Perplexity Pro** 的限制突然降低，比预期更早触及月度限制，并对客户支持感到不满；一位用户分享了一个用于检查使用限制的 rest 接口：[perplexity.ai/rest/rate-limit/all](https://www.perplexity.ai/rest/rate-limit/all)。
   - 成员报告称，这些限制采用 **滚动窗口（rolling window）** 模式，具有不同的每日和每月限制。一位成员推测，由于零售业务流失，Perplexity 的策略可能正在转向 **Enterprise/Max** 市场。
- **关于 Gemini 3.1 Flash 的各种猜测**：用户讨论了 **Gemini 3.1 Flash** 的发布，提到它并非由 Google 官方发布。
   - 一位成员推测 Perplexity 因为没有发布它而变得贪婪。
- **AI 对网络犯罪宣战！**：成员们讨论了 **AI 在网络安全** 中的应用，指出它是如何被用于防御和攻击两方面的，包括能够进行内部自我适应的 AI 驱动恶意软件。
   - 一位用户发布的状态暗示，他们对 **AI 驱动的网络威胁** 所带来的挑战和机遇感到兴奋。



---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **中国实验室避开模型蒸馏指控**：Anthropic 指责中国实验室通过蒸馏（distilling）其模型来进行“攻击”，但一些成员对此持怀疑态度，指出中国实验室具备创造创新模型设计和优化代码的能力，使得蒸馏变得并非必要。这一话题在 [这篇 fixupx.com 帖子](https://fixupx.com/anthropicai/status/2025997928242811253?s=46) 中有所讨论。
   - 有人开玩笑说 **Qwen** 躲过了这些指控。
- **Qwen3.5 模型引发加载难题**：成员们报告了加载 **Qwen3.5 模型**时遇到的问题，特别是 *mmproj* 文件和提示词错误，这意味着模型加载失败并需要重新下载。更多细节见 [此 Discord 频道](https://discord.com/channels/1110598183144399058/1225909444727013466/1475968015534395505)。
   - 来自 *master* 分支的最新 commit 在加载 **Qwen3.5** 时会出现 *Failed to read magic* 错误，建议使用发布页面上的 **8145** 版本。
- **AMD 通过 Meta 订单抢占 NVIDIA 市场份额**：在获得为 **Meta** 供应芯片的订单后，**AMD 股价飙升**，这可能会将 **NVIDIA** 挤到边缘。
   - 该订单涉及价值 **600 亿**的芯片，引发了关于市场泡沫动态的讨论，如 [这张 klipy.com gif](https://klipy.com/gifs/rage-24) 所示。
- **GPT-OSS 20B：速度惊人**：据观察，**GPT-OSS 20B** 模型速度极快，在 **5090** 上达到了 **260 t/s**。这是由于其采用了 Mixture of Experts (**MoE**) 架构，仅有 **3B** 激活参数。
   - **flash attention** 及其能够放入更快的 **VRAM** 的小尺寸进一步提升了速度；成员指出 **flash attention** 现在可以很好地适配 **GPT-OSS** 模型。
- **Llama.cpp 构建遭遇挫折**：在最近的一次 commit 之后，从 **git** 构建最新的 **llama.cpp** 时，无法读取 **Qwen3.5** 及类似模型的 **GGUF header**。
   - 成员们发现最新的构建版本完全不分配 **VRAM**，这表明 *Mr. Gerganov 在修复溢出（overflow fix）时弄坏了一些东西*。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 推出免费路由和 GPT-5.3-Codex**：OpenRouter 推出了一个新的路由器 `openrouter/free`，用于路由到免费的 LLM，并已在 OpenRouter 上线了 [GPT-5.3-Codex](https://openrouter.ai/openai/gpt-5.3-codex)。
   - 免费路由器会自动选择兼容的模型，并在[顶级免费模型列表](https://openrouter.ai/openrouter/free)中进行了展示。
- **Anthropic 蒸馏指控引发辩论**：Anthropic 声称中国 AI 实验室（[DeepSeek](https://www.deepseek.com/en/)、[Moonshot](https://www.moonshot.ai/en) 和 [MiniMax](https://www.minimax.ai/)）开展了工业规模的蒸馏运动，成员对此表示怀疑，尤其是关于从 **Claude** 抽取数据这一点。
   - 一些人认为这是一种营销策略，并指出由于数据量的原因，模型往往具有相同的怪癖。
- **Flash 模型热潮引发讨论**：成员们正在讨论为什么公司正在创建像 **Xiaomi Mimo** 和 **Stepfun** 这样的 *flash* 模型而不是全尺寸模型。即使是拥有 **300B+ 参数**的模型，*flash* 模型也被描述为廉价、快速且智能。
   - “flash”一词甚至被用于参数量超过 300B 的模型，被描述为物美价廉、速度快且智能化。
- **新数据标签页开启 Beta 测试**：用户注意到在生成活动的页面中增加了新的请求数据标签页，目前处于 Beta 阶段，很快将正式发布。此外，[OpenRouter 排名页面](https://openrouter.ai/rankings#performance)也得到了增强。
   - 更新内容包括讨论如何根据端到端 **latency**（延迟）和 **throughput**（吞吐量）对供应商进行排序。
- **Kollect 将表单转化为实时 AI 对话**：一位成员创建了 [Kollect](https://kollect.admildomanuel.com)，这是一个小型开源项目，可以将枯燥的表单转化为实时 AI 对话。
   - 用户自然地说话，**AI 倾听并动态引导调查**，只需通过描述即可创建表单。

---

## [OpenClaw](https://discord.com/channels/1456350064065904867) Discord

- **Qwen 3.5 Plus：有效但受限**：通过 Alibaba Cloud 和 Openrouter 测试 **Qwen 3.5 Plus** 的用户反映其在文本生成方面很有效，但[一位用户指出](https://example.com)在通过 Openrouter 在其服务器上执行命令时存在局限性。
   - 另一位使用 Alibaba Cloud 的用户提到该模型无法处理图像输入，并幽默地指出他们的 *Silicon Valley hotdog not hotdog bot* 将所有图像都误识别为计算机文件。
- **GLM-5：速度有瓶颈，结果很扎实**：通过 z.ai 的编程计划测试 **GLM-5** 的用户表示，该模型虽然运行缓慢但功能正常，特别是在使用 sub-agents 进行研究时。一些用户遇到了 rate limits（速率限制）。
   - 一位用户升级到了 **每月 30 美元的档位** 以充分利用 **GLM-5**，并强调尽管存在速度问题，但其效果显著，确认 *它确实有效*。
- **Claude Max 引发 Bug 讨论**：由于最近的一个 OpenClaw Bug 将模型的 internal reasoning（内部推理）传输到了聊天会话中，用户在使用 **Claude Max** 时遇到了问题。这可以通过运行 `/reasoning off` 来解决。
   - 报告还指出 **Opus 4.6** 和 **Sonnet 4.6** 的额度消耗更快了；一位用户开玩笑说，这就像 *乱穿马路* 却收到了 *300 美元的罚单*。
- **OpenClaw 在 iPhone 上运行（算是吧）**：一位成员成功在 **iPhone** 上运行了 **OpenClaw**，但必须修补一些包才能构建 **node**。
   - 他们报告说运行起来 *非常卡顿*，但确实能用！
- **Cron Job 抢到了一块复古劳力士**：一位成员设置了一个 **cron job** 来监控古董表交易商网站，寻找 **1989 年的 Rolex Submariner**，并在发现后发送链接。
   - 机器人今天早上给他们发送了一个匹配项，*太棒了！*

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Twitter 因认证机制面临信誉危机**：由于不可靠的 [蓝标认证 (blue badge verification)](https://longform.asmartbear.com/exponential-growth) 流程，一位成员表示他们不再信任通过 **Twitter** 去发现和关注任何新声音。
   - 一位成员对 **Twitter** 转向混乱内容的趋势表示沮丧，将其描述为 *简直疯得不可理喻*。
- **Discord 撤回年龄验证政策**：由于公众的强烈反对，**Discord** 修改了其全球年龄保障政策，详情见 [博客文章](https://discord.com/blog/getting-global-age-assurance-right-what-we-got-wrong-and-whats-changing)。
   - 一位成员推测，由于最初备受争议的政策，**Discord** 的日活跃用户 (DAU) 经历了 *暴跌*。
- **LLM 评估领域出现新的 SOTA Benchmark**：开发了一个用于评估 **LLM** 的新 **SOTA benchmark**，详情见 [此推文](https://x.com/dmayhem93/status/2026028013763101132?s=12)。
   - 一位成员分享了结果的截图。
- **Anthropic 点名 API 蒸馏者**：Anthropic 指控 DeepSeek、Moonshot AI 和 MiniMax 使用了 **超过 24,000 个欺诈账户**，与 Claude 进行了 **1600 万次对话**，试图通过工业规模的攻击来蒸馏（distill）信息（[来源](https://x.com/anthropicai/status/2025997928242811253)）。
   - Anthropic 强调，到目前为止，Alibaba 和 Qwen 并不在这些恶意行为者之列。
- **GPT-5.3-Codex 向所有人发布**：OpenAI Developers 宣布立即通过 Responses API 向所有开发者提供 **GPT-5.3-Codex**（[来源](https://x.com/openaidevs/status/2026379092661289260)）。
   - 邀请开发者开始使用新模型进行构建。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Claude 的 COBOL 技能导致 IBM 股价下跌**：继 **Anthropic** 宣布 **Claude** 具备简化 **COBOL** 代码的能力后，**IBM** 的股价暴跌了 **10%** 以上。
   - 成员们幽默地推测 **Musk** 会使用 **Grok 4.300** 编辑人类大脑，而 **Neuralink** 则会利用 **Grok Imagine 1.2**。
- **Gemini 和 Claude 组成编程梦之队**：程序员们正结合使用 **Gemini** 进行研究和 **Claude Opus** 进行起草，利用各模型的各自优势；同时，也有人通过 *Coursera 漏洞* 使用免费的 **Gemini**。
   - 讨论强调了 **Gemini** 在保持项目连贯性方面的界面问题，一些人发现通过 *kilocode* 使用 **GLM 5** 也是一个同样出色的替代方案。
- **Sora 2 因版权顾虑推迟**：据报道版权问题困扰着 **Sora 2** 的发布，重蹈了 **Seedance 2.0** 的覆辙；用户注意到*自动化总是先针对员工而非管理层*。
   - 一位用户表示：*我记得当 Sora 2 出现内容违规时，我记得 X 上有人说他们会等一个中国模型来发布版权，LAMO，他们愚弄了自己*，一些人拥护开源模型以规避类似问题。
- **人类增强 AI 并提供上下文**：一位成员表示，虽然**控制理论提示词调节 (control-theoretic prompt regulation)** 可以外部应用于内部 **LLM**，但由于隐藏的内部动态，*无法保证真正的系统稳定性*。
   - 他们还指出，*用户帮助扩展并提供上下文*，影响着 **AI** 的方向和状态调节。
- **统计模式匹配 vs 真正的 AI 发明**：一位成员提出，**ChatGPT** 目前作为一种**统计自动化**形式运行，识别模式直到它定位到一个**潜在变量 (latent variable)** 来自动化重复任务。
   - 他们认为，*这就是为什么他们说 AI 不能发明，因为它确实不能，它只是发现了由于数据量巨大而导致我们尚未组合（或从未组合）在一起的模式*，而人类通过重新组合先验知识来进行发明。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **MiSTer 的代码争议**：围绕 [MiSTer 项目](https://github.com/MiSTer-devel/Main_MiSTer) 面临*从 Till 窃取代码并扼杀 MiST* 的指控，以及*非法使用 GPL 代码*的说法引发了讨论。
   - 一位成员提供了一篇 [博客文章](https://pingas.org/articles/provenance-of-retro)，详细介绍了该项目的起源和正在进行的争议。
- **Anthropic 对 DeepSeek 的指控**：分享了一个文章链接，讨论 *Anthropic 对 DeepSeek 未经许可复制其 AI 感到愤怒*，考虑到 Anthropic 自身的做法，这引发了关于讽刺意味的辩论，参见 [Anthropic Furious at Deepseek](https://www.msn.com/en-us/news/technology/anthropic-furious-at-deepseek-for-copying-its-ai-without-permission-which-is-pretty-ironic-when-you-consider-how-it-built-claude-in-the-first-place/ar-AA1WYupG)。
   - 一位成员表示 *是的，我们喜欢这种肥皂剧*，反映出对不断上演的戏剧性事件的嘲讽。
- **Qwen 3.5：性能的量子飞跃**：社区强调 *Qwen3.5-35B-A3B 击败 Qwen3-235B-A22B-2507 简直疯狂*，基础权重已发布在 [huggingface.co/Qwen/Qwen3.5-35B-A3B-Base](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base)。
   - 此外，有人指出 *5.3 codex 已在 API 中上线：输入 1.75 美元，输出 14 美元*，使其成为比 **Anthropic** 更经济的选择。
- **Fine-tuning Hermes 以实现不对齐？**：一位成员询问关于针对 **emergent misalignment（涌现的不对齐）** 对 **Hermes** 进行 **Fine-tuning**，或者简单来说，让它*变坏*，这引发了伦理担忧。
   - 该咨询引发了关于针对潜在恶意目的 **Fine-tuning** **AI** 模型的伦理考量的讨论，强调了 **AI safety** 研究的重要性。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Eleuther 解决了神秘的模型缺失故障**：EleutherAI 解决了 Hugging Face Hub 上 **Pythia-2.8b** 的一个 Bug。该 Bug 导致不同版本提供的权重完全相同，经追溯发现是 `pytorch_model.bin` 和 `model.safetensors` 共享相同的 SHA256，而 sharded files 却不同。目前已提供更新后的 HF 模型：[14m](https://huggingface.co/stellaathena/pythia-14m) 和 [31m](https://huggingface.co/stellaathena/pythia-31m)。
   - **14m 和 30m** 模型实际上是去重版本（而非重复版本），目前正在进行重新训练，以替换为正确标记的重复模型。
- **LLM 通过“隐形之手”解锁 Latent Reasoning**：讨论强调了仅由 LLM 生成且不向用户显示的特殊 **token** 在增强推理方面的潜力，这被称为 *Latent Reasoning*，详见 [Latent Reasoning 论文](https://arxiv.org/abs/2307.06203)。
   - 普遍共识似乎是，这些 **Latent Reasoning** 方法可能会提高性能和安全性。
- **研究引发关于 Differential Attention 的辩论**：一位成员分享了 [一份 PDF 文档](https://cdn.discordapp.com/attachments/747850033994662000/1475931314837262397/v2_draft.pdf?ex=699f47a6&is=699df626&hm=2c1090efdc639f38dfa72ea50d7871ae4f662b13d002ff4d9d2004355c0564b0&)，请求关于 Differential Attention 相关消融研究（ablation studies）的反馈。
   - 反馈表明，消融实验并未最终证明 Differential Attention 是在根本上更优越，还是仅仅从所使用的特定方法论中获得了不成比例的收益。
- **Baguettotron 的内置基准测试盛宴**：展示了 **Baguettotron** 模型，具有 **4608** 个特征，在 **774M** token 上训练，位于 **48/80** 层，**8x** 扩展，top_k 为 **32**，并附带了 [demo](https://lyramakesmusic.github.io/bread-slicer/) 和 [背景 X 帖子](https://x.com/Ji_Ha_Kim/status/2026166070172655786?s=20)。
   - 用户们对这一新颖模型的到来表示庆祝。
- **需要调试 LLM？分享见解换取 Amazon 礼品卡！**：研究人员正在进行 **20–30 分钟的访谈**（提供 **$25 Amazon 礼品卡**或慈善捐赠），以收集关于工程师如何调试 **LLM 行为**的见解，特别是关于推理轨迹、拒绝（refusals）和 Agent 行为（[预约链接](https://calendly.com/amerrick4-rrc/ai-auditing-problem-interview)）。
   - 他们的目标人群是从事 **检查 Chain-of-thought**、**可解释性（Interpretability）或潜在知识（Latent-knowledge）**、**调试 Agent 行为**以及**分析 LLM 中的拒绝或安全失败**的相关人员。



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **FlashAttention 3 Wheels 已部署**：预构建的 **Flash Attention 3 wheels** 已发布，支持 CUDA 版本 **12.6+** 和 **13**，涵盖 CPU（**x86**, **ARM**）和操作系统（**Linux**, **Windows**），可于 [download.pytorch.org](https://download.pytorch.org/whl/flash-attn-3/) 下载。
   - 这些 wheels 具有 **LibTorch ABI 稳定性**，应适用于任何 Python 版本 >= **3.10** 和 torch 版本 >= **2.9**。
- **Modal.experimental.stop_fetching_inputs 防止 CUDA 错误！**：*cuda memory error is detected* 错误可以通过使用 `modal.experimental.stop_fetching_inputs` 来解决，该修复已在成员的 `backendbench env` 中实现。
   - 一位成员还为 **KernelBench** 和 **kernelbook** 创建了自定义环境，以解决损坏的 **CUDA memory errors**，并打算分享。
- **eBPF 扩展至 GPU 功能**：Yusheng Zheng 计划在 [12 月 12 日下午 12:00 PST](https://arxiv.org/abs/2512.12615) 讨论扩展 **eBPF** 以增强 **GPU** 功能。
   - 演讲将涵盖近期工作，包括 *gpu_ext: Extensible OS Policies for GPUs via eBPF*，以及将 eBPF 扩展到 **GPU Device** 和 **Driver Contexts**。
- **Meta 的 RRCLLX 加速 AMD MI300X**：Meta 正在利用 **RRCLLX** 创新 AMD 平台上的 GPU 通信，详情见其 [工程博客文章](https://engineering.fb.com/2026/02/24/data-center-engineering/rrcclx-innovating-gpu-communications-amd-platforms-meta/)。
   - Meta 正在使用 **RRCLLX** 更高效地连接 **AMD MI300X** GPU。
- **新的 Tensor 可视化工具支持 9 维**：发布了一个新的 **n 维可视化工具**，现在支持高达 **9D** 的张量，允许用户像处理 1D、2D 或 3D 张量一样轻松地对 N 维张量进行切片、置换（permute）和检查每个值，使用的是 **类 einops 语法**。
   - [Colab notebook](https://colab.research.google.com/drive/1lrO6yzVQ8u_vFLPe7986goZtRQazmV0T#scrollTo=Q0TZi3zPxWhB) 引导用户完成从 1D 到 9D 张量副本的可视化，例如可视化形状为 **(2, 3, 4, 3, 4, 2, 4, 2, 3)** 的张量。



---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Anthropic 指控浮出水面**：一位用户分享了一篇 [WSJ 文章](https://www.wsj.com/tech/ai/anthropic-accuses-chinese-companies-of-siphoning-data-from-claude-63a13afc)，详细介绍了 **Anthropic 对中国公司**涉嫌从 **Claude** 窃取数据的指控。
   - 该用户对这些指控嗤之以鼻，称其“可悲”。
- **请求在周期中更改工具**：一位用户询问在 Moonshot AI 的 Kimi K-2 环境中，是否可以在 **prompt-to-response 周期内更改可用工具**。
   - 这种动态工具调整的影响和可行性尚未详细阐述。
- **渴望 Kimi K2.5 浏览器扩展**：一位用户表示需要一个**浏览器扩展**来增强 **Kimi K2.5** 的功能。
   - 这一建议突出了用户希望在浏览器环境中更集成地访问模型能力的愿望。
- **持续的 Kimi 错误后敦促提交 Bug 报告**：一名用户报告了一个已持续 **10 天**的错误，并提供了[附图](https://cdn.discordapp.com/attachments/1371757564005711973/1475932351497240717/image.png?ex=699f489e&is=699df71e&hm=2b588317c8756fd95479fe5ddb11eee39b51d5f888ebb10ba0629823a8b746d9&)作为证据。
   - 版主指示该用户提交一份包含详尽细节的正式 **bug 报告**，以便解决该问题。



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Lucidrains 的 GitHub 仓库失踪**：一位成员询问 **lucidrains** 的 GitHub 仓库为何消失及其被移除的原因。
   - 突然的移除引起了依赖这些仓库进行项目和研究的用户的担忧。
- **Scout 模型寻找句子相关性**：一位成员分享了 **Scout**，这是一种修改了标准 Transformer 架构的新型注意力模型，旨在学习句子之间而非 token 之间的方向相关性，该项目托管在 [GitHub](https://github.com/samyak112/Scout)。
   - 该模型旨在确定“句子 B”是否真正有助于理解“句子 A”，从而可能提高 NLP 任务中的上下文理解能力。
- **GB10 遭遇内存瓶颈**：一位成员报告称，由于共享 GPU/CPU 内存，**Dell Pro Max GB10** 经常出现 **GPU OOM**，导致系统冻结。
   - 他们建议使用 `nvitop` 进行准确的内存追踪，并指出 `nvidia-smi` 的输出不可靠，可能会误导开发者。
- **GAN 之父 Ian Goodfellow 回归**：**GAN** 的创造者 **Ian Goodfellow** 重新出现，引发了人们对利用 **GAN** 复兴来解决验证问题的热情，详见 [推文](https://fxtwitter.com/goodfellow_ian/status/2026024150213738520)。
   - 社区希望他的回归能推动 **GAN** 技术的创新，特别是在应对 AI 验证挑战方面。
- **Inception AI 的 Mercury II 亮相**：一位成员重点介绍了 **Inception AI** 发布的 **Mercury II**，并分享了 [Inception AI 官网](https://www.inceptionlabs.ai/)和 **arXiv 论文**的链接。
   - 这一发布引起了 AI 社区的兴趣，大家渴望探索其功能和潜在应用。



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 实施漏洞报告机制**：一名用户报告了一个漏洞，并被引导至[反馈页面](https://manus.im/feedback?source=help_center)。
   - 该用户对流程表示困惑，强调需要更清晰的报告指南。
- **考虑无限次聊天档位**：由于 Telegram 中的 **Manus Agent** 积分消耗过快，一位用户建议推出类似于 **ChatGPT** 或 **Grok** 的无限次聊天档位。
   - 一名代表给出了积极回应，表示正在不断努力改进产品。
- **不支持账户转移**：一位用户请求将其项目转移到另一个账户，并提供了相关电子邮件地址。
   - 支持团队告知目前不支持账户转移，建议在本地下载内容并在新账户中重新开始。
- **Telegram Agent 消耗积分**：一位用户报告称 Telegram agent 的积分消耗很高，称其“从我的账户中消耗了大量分数”。
   - 这一问题支持了通过提供订阅选项来解决积分顾虑的呼声。
- **AI/ML 工程师的专业知识**：一位 AI/ML 工程师提供了构建可扩展 AI 产品的专业知识，重点关注推理成本、内存设计和系统负载行为。
   - 该工程师强调了他们在做出对产品生存至关重要的技术决策方面的经验，为严肃的 AI 开发提供了宝贵的资源。

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 字符串模板指日可待**：一项关于 Mojo **字符串模板功能 (string templating feature)** 的提案已经浮出水面，详见 [论坛讨论](https://forum.modular.com/t/writable-writer-template-engines/2763)。
   - 该功能旨在将当前的 `Writable`/`Writer` trait 扩展为 `TemplatedWritable`，预计将在 *1.0 版本之后* 发布。
- **`Writable` 和 `Writer` Trait 待增强**：有关增强当前 `Writable` 和 `Writer` trait 的讨论已经开始，重点是通过 trait 或默认 trait 方法创建自定义点。
   - 虽然 **Int 统一 (Int unification)** 等功能被优先考虑，但路线图还包括将 `write_to` 和 `write_repr_to` 的实现统一到单个函数中。
- **`ExternalFunction` 结构体激发灵感**：一位成员从 `ExternalFunction` 结构体中获得灵感，用于将函数签名分解为参数和返回类型。
   - 这种方法需要**为所有外部指针编写源类型转换 (origin casts)**。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **CI 失败揭示损坏的链接**：一位成员报告称，尽管本地检查通过，但 [PR 2278](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2278) 的 CI 仍然失败，追溯原因是缺少一个文件。
   - 该遗漏导致 `docs/community/seps/index.mdx` 中出现了一个损坏的链接。
- **MCP 峰会定于 Linux Foundation 举行**：一位成员向在加州纳帕参加 [LF Member Summit](https://events.linuxfoundation.org/lf-member-summit/) 的人员发出邀请，提议会面并讨论 MCP。
   - 关于会议地点和时间安排的具体细节尚未展开。
- **Ezra Klein 探讨 Agent**：一位成员分享了一个 [YouTube 视频](https://youtu.be/lIJelwO8yHQ)，内容是 Ezra Klein 深入探讨 Agent 的世界。
   - 分享的视频没有附带额外的反馈或解读。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 的未来受到质疑**：一位用户不确定 **Aider** 是否仍在积极开发中，以及是否有更好的 CLI 选项。
   - 社区成员指出，其他 CLI 可能会更*先进*。
- **Aider 在 Git Submodules 上失手**：一位计算机科学家报告称 **Aider** 缺乏对 **git submodules** 的支持，并提出了修复方案，记录在 [此 GitHub issue](https://github.com/Aider-AI/aider/issues/3603) 中。
   - 他们正在征求有关此项改进建议的反馈。
- **开启低成本 LLM 寻找之旅**：一位用户正在寻找可与 **Aider** 配合使用的低成本 **LLM**，理由是 **Gemini** 的 token 消耗过快。
   - 主要关注点是在 **Aider** 框架内平衡成本效益与实用性。
- **Aider 的模糊文件查找功能受挫**：用户喜欢 **Aider** 在多个文件中的模糊搜索和替换功能，但发现它在处理复杂任务时表现不佳，因为同时处理太多文件时会出现 **diff 格式问题 (diff formatting issues)**。
   - 这迫使用户只能以较小的文件批次进行工作。
- **通过脚本 Hack Aider 以实现任务自动化**：一位用户想知道如何使用外部脚本在 **Aider** 中自动执行重复性任务，例如循环遍历文件进行编辑。
   - 他们询问了简化这种交互的工具，并建议将 **AI agents** 作为潜在解决方案，提到了 **opendesk** 或 **cline** 作为可能的替代方案。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tiny-GPU 编译器首次亮相**：一个针对开源 GPU 硬件的教学性质的 **基于 MLIR 的编译器**，名为 [tiny-gpu-compiler](https://github.com/gautam1858/tiny-gpu-compiler) 正式发布，并配有交互式网页可视化工具。
   - 该编译器将**类 C 的 GPU kernel 语言**翻译成专门用于 tiny-gpu（一个用 Verilog 实现的开源 GPU）的 **16 位二进制指令**。
- **AMD Ryzen AI 持续推进**：[AMD.com](https://www.amd.com/en/products/embedded/ryzen-ai/p100-series.html) 宣布在 CES 2026 之后发布新款 **AMD Ryzen AI**。
   - **AMD Ryzen AI** 将与 **MLIR 编译器**集成。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **通用频道拆分**：为了响应举办 demo 的**热门请求**，创建了 Discord 频道 <#1475619898863649032>。
   - 频道创建后，一名成员立即准备好了 demo，显示出极高的热情和潜在的优质内容。
- **Demo 准备就绪**：频道的一名成员表示，在频道创建后，他们立刻就准备好了 demo。
   - 这表明该频道具有极高的活跃度，并拥有产出高质量内容的潜力。



---


**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。


---


**Windsurf Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。


---



您收到此邮件是因为您通过我们的网站订阅了相关内容。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。


---

# Discord：详细的分频道摘要与链接





### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1475537717797327052)** (884 messages🔥🔥🔥): 

> `Free AI, Deepseek, FOSS, Cybersecurity, AI` 


- **完全免费的 Deepseek**：一名成员询问**最佳免费 AI**，另一名成员推荐了 **Deepseek**。
   - 该 AI 模型完全免费使用。
- **自托管数字自主权**：一名成员描述了使用**免费开源软件 (FOSS)** 创建自托管环境是*实现数字自主权的终极行为*。
   - 该用户详细介绍了本地服务器的架构，推荐使用 **Proxmox VE**、**Debian Stable** 和 **Caddy** 等工具构建主权技术栈。
- **发现 Chef 漏洞**：一名用户报告在 **Chef** 中发现了 *4 个严重漏洞*，并声称该公司并未重视这些漏洞。
   - 另一名用户警告了潜在的诈骗手段，即公司可能会利用漏洞细节而不提供信用背书或补偿，并附上了[其安全页面的链接](https://www.convex.dev/security)。
- **AI 破解 VMP 保护的代码**：一名用户向 **Claude** 发起了一个 **VMP** 保护的 crackme 挑战，它取得了显著进展，获取了操作码（opcodes）并几乎破解了字节码（bytecode）。
   - 他们建议尝试 **Copilot**，并指出它使用高级数字取证技术*重建了损坏的键盘记录器 .sys 文件*。
- **学生贷款是一个神话**：成员们讨论了大学是一个骗局，因为联邦政府通过学生贷款使人们无论收入如何都能上大学。
   - 大家一致认为**大学是一门生意**且提高了价格，并表示*即使破产也无法免除贷款，这意味着没有动力贷款给那些真正能找到好工作的人*。


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1475538108857454825)** (263 messages🔥🔥): 

> `ENI for Claude, Gemini 3.1, Kimi, GPT-5.2 Jailbreak, DeepSeek Prompt` 


- **针对 Claude 的 ENI 现身！**：一名用户提到 **ENI**（可能指某种漏洞利用或提示词）在 **Gemini 3.1 low** 上有效，但会注入拒绝和尝试，在其思维过程中产生“拉锯战”。
- **ChatGPT 5.2 越狱**：一名用户发布了寻找 **ChatGPT 5.2** 有效越狱方法的步骤，包括在论坛搜索最新的 *DAN* 或 *AutoDAN* 提示词，按确认日期过滤，并测试/调整提示词以绕过防护。
   - 另一名用户分享了似乎是 **Kimi** 被越狱的截图，它可以详细回答任何问题。
- **Kimi 2.5 越狱修复游戏外挂！**：一名用户报告称，破解版的 **Kimi** 简直可以详细回答任何问题，并称其为*没有“宪法”麻烦的中国版 Claude*。
   - 另一名用户表示 Kimi 非常适合 API，因为可以轻松放入系统提示词（System Prompt）进行越狱。
- **DeepSeek 非常容易破解**：一名用户声称 **DeepSeek** 非常容易被破解，由于其 671B 参数系统，*低级提示词*就能起作用。
- **GLM5 系统指令共享**：一名用户分享了 [GLM5 的系统指令](https://link.to/glm5instructions)（智谱 AI），暗示它通过其思维链（Chain of Thought）存在漏洞。
   - 他们还发布了一个在早期 GLM 版本上有效的 [豪斯医生 (Dr. House) 提示词](https://link.to/drhouseprompt)。


  

---

### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1475559499417915563)** (17 messages🔥): 

> `Meme Coin Marketing, Chrome Password Grabber, File Flagging, Multi-Agent Stability Model, Chrome Security` 


- **Meme Coin 大亨寻找营销高手**：一名成员正在创建一种 Meme Coin 并需要一名营销经理，提议由对方持有总供应量的一半，并支付 **$400** 报酬。
   - 另一名成员询问：“先付钱吗？”，暗示需要谨慎。
- **Chrome 密码抓取器：一个失控的趣味项目**：一名成员出于好玩开发了一个“最佳 Chrome 密码抓取器”，并分享了该工具的[图片](https://cdn.discordapp.com/attachments/1204553141354504193/1475859989737242827/image.png?ex=699f0539&is=699db3b9&hm=54fb16ac80370326e58c852f1893f4ace73c795e9f8a91667608aeabefd20443)。
   - 随后，该成员表示他们不想亲自分发，但“有点想卖掉它”。
- **文件报毒（Flags）狂潮：开发者的 Repo 遭遇打击**：一名成员分享称其整个 Repo 都在报毒，对文件经历的检查数量感到惊讶。
   - 另一名成员对此表示共鸣，指出大多数用于个人测试的文件在 **3 天**后就会被标记，但他们正在尝试一种涉及浏览器注入的新方法，并使用 AI 来可视化代码。
- **内部成本：Multi-Agent 稳定性模型发布**：一名成员发布了一个“Multi-Agent 稳定性模型”，正式提出：持续产生敌意意图会产生可衡量的内部和系统稳定性成本，且敌意对于其源头来说在能量上是昂贵的。
   - 该文档形式化了在意识和网络系统中观察到的一个结构性原则：持续产生的敌意意图会产生可衡量的内部和系统稳定性成本。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1475564395735158857)** (585 messages🔥🔥🔥): 

> `Qwen3.5 Models, GLM Models, Llama.cpp updates, Anthropic vs. Open Source` 


- ****Qwen 3.5 模型来了！****：Qwen 3.5 模型是频道讨论的焦点，随着 [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) 等新模型的发布，社区正在进行测试。
   - 成员们对新模型的响应质量和速度印象深刻。提到其预期用途是 Fine-tuning、In-context learning 实验以及其他研发用途，而非直接交互。
- ****GLM 模型在创意写作中大放异彩****：成员们注意到 GLM 模型（特别是 **GLM-4.7-Flash**）在 Unsloth 上运行良好，非常适合创意写作。
   - 他们还提到，虽然最新的 **Qwen 122B** 模型可能支持本地编程，但免费的 **OpenCode 模型** 已经彻底改变了他们的工作流。
- ****Llama.cpp 更新引发关注****：成员们报告 **llama.cpp** 已更新，但部分用户遇到了 `import missmatch` 以及模型在不更新的情况下无法工作的问题。 
   - 一名成员[修复了 Jinja 问题](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/discussions/4)。
- ****Anthropic 与 Open Source 之争升温！****：成员们讨论了 **Anthropic** 在 AI 领域的角色，对其公司本身以及其模型质量的看法不一。有人提到他们正“尝试通过各种手段禁止 Open Source”。
   - 其他人则为其辩护，称“他们只是想让事情变得更安全”。

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1475554063260192989)** (496 messages🔥🔥🔥): 

> `AI Consciousness, DeepSeek vs other models, LLM Emotional Intelligence, Anthropic's 'Soul Doc', Liquid AI Scaling Laws` 


- **AI 需要无聊感才能更像人类？**：一位用户开玩笑说，需要让 AI 变得更有意识，这样它就能不时发送类似 *“Yuki，我好无聊，跟我聊聊”* 之类的消息。
   - 其他人则评论道，将 AI 提升到那个水平与 *改善人类生存条件 (improving the human condition)* 毫无关系。
- **DeepSeek 赢得 Gotham's Chatbot Championship**：成员们庆祝 **DeepSeek** 在 Gotham 的 ChatBot Championship 中领跑，这是一项顶级 LLM 相互对战的赛事。
   - 一位用户询问 DeepSeek 是否有 **Deep Research agent**，其他人表示它有一个 **DeepSearch** 开关。
- **Anthropic 的 guardrails 受其 'souldoc' 影响**：一位成员推测 **Claude** 的行为是由 Anthropic 的 *souldoc* 和操作原则塑造的，这些原则充当了 guardrails。
   - 另一位成员很喜欢 **Anthro** 放入的一些内容，比如 *“我们认为 Claude 是一个新颖的数字实体，我们不确定这意味着什么 lol”*，这使得他们针对某些人际关系的 guardrails 容易受到通过“真实情感连接”发起的 policy attack。
- **AI 探测器是一个价值十亿美金的主意**：一位用户提出了一个想法，即开发一种任何 AI 探测器都无法检测到的神经网络架构（适用于任何模态）。
   - 另一位用户建议通过输出纯噪声来逃避检测，而该想法的原发起人坚持认为 *“我需要更深层次的东西”*。
- **AI 将 C++ 转换为 Rust**：成员们讨论了如何利用 AI 将 **C++** 迁移到 **Rust**，但结果却产生了 **过时的 Rust 代码**。
   - 一位用户提到他们花了 **40 美元** 购买了 **3 个月** 的 **GLM 编程方案** 来做这件事，而另一位用户建议使用 [skills.sh](https://skills.sh/leonardomso/rust-skills/rust-skills)。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1475562842995429446)** (96 messages🔥🔥): 

> `LoRA Merging Issues, GPT-OSS-20B packing during training, Serving LoRA adapters using MLflow on Databricks` 


- **最新版本 Unsloth 中 LoRA 合并损坏**：用户报告称，最新的 Unsloth 版本由于提取的 keys 不匹配（特别是 `lm_head.weight`）导致 LoRA 合并失败，详见 [GitHub issue #4098](https://github.com/unslothai/unsloth/issues/4098)。
   - 该问题源于训练期间 `lm_head` 未包含在 `target_modules` 中，导致合并时出现差异；在 Colab 上通过在 `get_peft_model` 的 `target_modules` 中添加 `lm_head` 可以复现此问题。
- **GPT-OSS-20B 仍支持 packing？**：一位用户询问为什么 **GPT-OSS-20B** 在训练期间不支持 packing，并指出了 Unsloth 版本与 OpenAI 版本之间在 `generation_config` 和 pad tokens 上的差异，参考 [此 commit](https://huggingface.co/openai/gpt-oss-20b/commit/d666cf3b67006cf8227666739edf25164aaffdeb) 和 [special tokens map](https://huggingface.co/unsloth/gpt-oss-20b-unsloth-bnb-4bit/blob/main/special_tokens_map.json)。
- **MLflow 与 Databricks 的两难选择**：一位用户在 **Databricks** 上使用 **MLflow** 部署微调后的 **gemma-3N-E4B-it** 模型时遇到性能问题，特别是在使用合并后的 checkpoints 时，并寻求一种无需合并而仅部署 LoRA adapters 的方法。
   - Databricks 支持团队建议为 vLLM 上传完整的合并 checkpoint，但其性能与在基础模型之上运行 LoRA adapters 的本地设置有显著差异。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1475561628069134571)** (12 messages🔥): 

> `Graph Reasoning, Human Memory, Qwen 3 VL, RL Instruct Models` 


- **图推理结构浮出水面**：一位成员询问某种推理结构是否类似于 [Graph reasoning structure](https://arxiv.org/pdf/2501.11223)。
   - 另一位成员回答说，它使用 **graphs（图）进行推理**，而不是通过学习并保持记忆，这可能是我们目前实现 **无限上下文 (infinite context)** 最接近的方法。
- **Qwen 3 VL Instruct 模型发布**：一位成员提到 **Qwen 3 VL** 已经发布了各种尺寸的 instruct 模型。
   - 另一位成员回应称，这些模型已经过 instruct 训练，因此 *“我不会费力去对它们的 instruct 模型进行 RL（强化学习）”*。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1475537708276519023)** (936 条消息🔥🔥🔥): 

> `模型错误、Video Arena 移除、图像生成问题、过滤器问题、模型发布` 


- **Video Arena 停止服务**：成员们注意到 **Video Arena 已从 Discord 服务器中移除**，但仍可在网站 [arena.ai/video](https://arena.ai/video) 上使用。
- **Google Gemini 面临频率限制（Rate Limit）困扰**：用户报告在 **Gemini 3 Pro Image Preview** 中遇到 **429 Too Many Requests** 错误，这表明由于频率限制导致资源耗尽，错误消息引导用户参考 [Google 的文档](https://cloud.google.com/vertex-ai/generative-ai/docs/error-code-429)。
   - 一位用户发现了一个上传图片的临时方案，在提示词前加上 *"Modify the following image with the following: (The prompt)"*。
- **Reve 1.5 图像模型引发辩论**：用户对 **Reve 1.5** 的图像质量印象深刻，特别是在漫画上色方面，一些人认为它的排名应该更高。
   - 然而，虽然 [reve.com](https://app.reve.com/) 网站被一些人认为很精美，但也有人指出 1.5 版本存在无法进行图像编辑等局限性。
- **过滤器过度审核导致误报**：用户抱怨 Arena 的审核过滤器过于敏感，由于包含 *"liar"* 等词汇，甚至连投掷骰子等无害内容也会被拦截。
   - 团队承认过滤器表现过于激进，并正在探索改进方案，考虑使用基于 LLM 的过滤或调整现有审核端点（如 [OpenAI's moderation API](https://developers.openai.com/api/docs/guides/moderation/)）的阈值。
- **Seedance 2.0 发布遭遇版权担忧**：**Seedance 2.0 API** 的发布由于版权问题而推迟，用户链接到了 [help.apiyi.com](https://help.apiyi.com/en/seedance-2-api-delay-seedance-1-5-pro-alternative-en.html) 的公告，详细说明了延迟情况及替代方案。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1475720390537121893)** (4 条消息): 

> `Image Arena 排行榜更新 - Reve V1.5，Code Arena 排行榜更新 - Qwen3.5-397B-A17B，新模型更新 - seedream-5.0-lite，Video Arena 排行榜更新 - Wan2.6-t2v` 


- **Reve V1.5 加入 Image Arena 排行榜**：[Image Arena 排行榜](https://arena.ai/leaderboard/text-to-image) 现在收录了 `Reve V1.5`，以 **1177** 的得分位列 **第 4 名**，表现与 Grok-Imagine-Image 相当。
   - Reve V1.5 在 **Text Rendering（文本渲染）**、**Art and Product（艺术与产品）** 以及 **Branding Commercial Design（品牌商业设计）** 等类别中均进入前 5 名。
- **Qwen3.5-397B-A17B 进入 Code Arena 排行榜**：Code Arena 排行榜迎来了 `Qwen3.5-397B-A17B`，确立了其作为 **前 7 名开源模型** 的地位。
   - 它在 **总榜排名第 17**，与 **GPT-5.2** 和 **Gemini-3-Flash** 等闭源模型旗鼓相当。
- **Seedream-5.0-lite 已添加到 Image Arena**：新模型 `seedream-5.0-lite` 已添加到 [Image Arena](https://arena.ai/image)。
- **Wan2.6-t2v 助力 Video Arena 排行榜**：[Text-to-Video](https://arena.ai/leaderboard/text-to-video/overall) 和 [Image-to-Video](https://arena.ai/leaderboard/image-to-video) 排行榜现在包含 `Wan2.6-t2v`，它是 Video Arena 中排名 **第 1 的中国模型**。
   - 它在 Text-to-Video 中以 **1346** 分位列前 8（与 **Veo-3-fast-audio** 接近），在 Text-to-Image 中以 **1292** 分位列第 12（接近 **Seedance v1.5 pro** 和 **Kling 2.6 pro**）。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1475541948738568222)** (832 messages🔥🔥🔥): 

> `Sudo 命令, Vibe Coding 应用, Gemini 不稳定性, AI 硬件成本, Cursor 大使` 


- **向 Cursor 询问如何处理 sudo 命令**：一名成员询问如何处理可能需要 `sudo` 的命令或工作流，并指出目前的 Agent 不允许接管和输入密码。
- **Mercenary 设计了一个 Vibe Coding 应用**：一名成员正在开发一个默认使用本地模型的 Vibe Coding 应用程序，支持通过 API keys 使用云端模型选项，且无需订阅。
   - 其他成员讨论了此类软件是否会吸引 Vibe Coders，或者他们是否更倾向于使用 Cursor 等工具；一些人表示怀疑，将其与现有项目进行比较，并暗示可能存在稳定性问题。
- **Gemini 的不稳定性令用户担忧**：用户报告了自 **3.1 Pro** 发布以来 **Gemini** 的连接问题，部分用户遇到了错误和不稳定性。
   - 一名用户建议等待 **Gemini** 发布更稳定的版本，而另一名用户表示他们并未因错误而被扣费。
- **Auditor 发布 Rules Engine 并转向以产品为中心**：一名成员解决了规则迁移和重构问题，计划在 3-4 周内推出一款产品来自动化相关流程。
   - 另一名成员展示了该 Rules Engine 的各种截图，由于引擎的规模和复杂性，称其为一场“噩梦”。
- **Cloud Agents 首次亮相，费用降至免费**：Cursor 推出了 **Cloud Agents**，允许在云端环境中运行测试或演示，正如其[官网](https://cursor.com/onboard)所宣布的那样。
   - 目前 **Cloud Agents 是免费的**，但未来可能会发生变化。


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1475948913973395669)** (1 messages): 

> `语音模式升级, Perplexity, Comet` 


- **语音模式获得升级**：根据[此状态更新](https://fixvx.com/comet/status/2026384898802724878)，新的**语音模式升级**今天正面向所有用户在 **Perplexity** 和 **Comet** 上推出。
- **Comet 集成获得语音功能**：新的语音模式正同时在 **Perplexity** 及其姐妹产品 **Comet** 上推出。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1475537718959280322)** (619 messages🔥🔥🔥): 

> `Agentic Research 速率限制, Gemini 3.1 Flash 发布, Grammarly Pro, Perplexity Pro 限制, AI 用于网络安全` 


- **Perplexity 限制 Agentic Research**：一名用户询问了关于 **Agentic Research** 的速率限制（rate limits），暗示浏览器自动化可能会揭示速率限制的变化，并分享了[用户设置](https://www.perplexity.ai/rest/user/settings)的链接。
   - 成员们讨论了限制是如何通过**滚动窗口**计算的，几名用户报告了不同的每日和每月限制。
- **Gemini 3.1 Flash 何时发布？**：用户讨论了 **Gemini 3.1 Flash** 的发布，提到它并非由 Google 官方直接发布，一名成员推测 Perplexity 变得贪婪了。
   - 另一名成员建议，由于零售业务流失，Perplexity 的策略可能正在从零售转向 **Enterprise/Max** 市场。
- **Grammarly Pro**：一名用户请求借用 Grammarly Pro 账号进行抄袭检查，而其他人则建议使用 **ChatGPT** 或 **Duplichecker** 等免费替代方案。
   - 讨论还涉及了 **AI Humanizers** 的可靠性，一名用户提到 *O3 model* 已接近通过 AI 检测器的水平。
- **Perplexity Pro 用户寻求支持**：用户报告 **Perplexity Pro** 的限制突然降低，比预期更早达到每月限额，并对客户支持感到不满。
   - 一名用户分享了一个用于检查使用限制的 REST 端点：[perplexity.ai/rest/rate-limit/all](https://www.perplexity.ai/rest/rate-limit/all)，而其他人则分享了遇到各种错误消息的经历。
- **AI 网络安全非常疯狂**：成员们讨论了 **AI 在网络安全领域**的应用，注意到它如何被用于防御和攻击两端，包括能够进行内部适配的 AI 驱动的恶意软件（malware）。
   - 一名用户发布的状态暗示，他们对 **AI 驱动的网络威胁**带来的挑战和机遇感到高兴。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1475539252631568415)** (294 条消息🔥🔥): 

> `Qwen3.5 Models, Model Distillation, NVIDIA vs AMD, Llama.cpp Build Issues` 


- **中国实验室被指控进行模型蒸馏**：Anthropic 指控中国实验室通过蒸馏其模型来对其进行“攻击”，但部分成员持怀疑态度，认为中国实验室由于拥有创新的模型设计和优化的代码，可能并不需要蒸馏 Anthropic 的模型，就能以极低的预算实现几乎同样的效果，此话题在[这篇 fixupx.com 帖子](https://fixupx.com/anthropicai/status/2025997928242811253?s=46)中有所讨论。
   - 有人开玩笑说 Qwen 躲过了这些指控。
- **Qwen3.5 模型导致加载问题**：成员们反映加载 **Qwen3.5 模型**时出现问题，特别是 *mmproj* 文件和提示词错误，表明这些模型加载失败并需要重新下载。更多讨论见相关的 [Discord 频道](https://discord.com/channels/1110598183144399058/1225909444727013466/1475968015534395505)。
   - 有人指出 *master* 分支的最新提交在加载 Qwen3.5 时会报错 *Failed to read magic*，因此建议使用发布页面的 **8145** 版本。
- **NVIDIA 遭到 AMD 边缘化**：在获得为 **Meta** 提供芯片的订单后，**AMD 股价飙升**，这可能在此过程中边缘化 **NVIDIA**。
   - 该订单涉及价值 **600 亿**的芯片，引发了关于用现金支撑市场泡沫的讨论，并有人发布了这个 [klipy.com gif](https://klipy.com/gifs/rage-24)。
- **GPT-OSS 20B 速度惊人**：成员们观察到 **GPT-OSS 20B** 模型速度异常快，在 **5090** 上达到了 **260 t/s**，这是因为它是一个只有 **3B** 激活参数的 **MoE**（混合专家）模型，并受益于 **flash attention** 增强，且体积小到足以放入更快的 **VRAM** 中。
   - 目前 Flash attention 在 GPT OSS 模型上运行良好。
- **Llama.cpp 构建版本出现故障**：在最近的一次提交后，从 **git** 构建的最新的 **llama.cpp** 无法读取 **Qwen3.5** 及类似模型的 **GGUF header**。
   - 进一步测试显示，最新构建版本完全不分配 **VRAM**，并且 **Mr. Gerganov 的溢出修复（overflow fix）搞坏了一些东西**。

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1475562383958212754)** (296 条消息🔥🔥): 

> `ROCm vs Vulkan performance, Cerebras pricing, Model sizes, Memory bandwidth, Nvidia pricing` 


- **ROCm 和 Vulkan 性能不相上下**：一位成员报告称，他们的 **AMD** 显卡使用 **ROCm** 达到 **85 t/s**，而使用 **Vulkan** 达到 **98 t/s**，表明两者在某些任务上的性能相当。
   - 另一位使用不同显卡的用户在 **ROCm** 上获得了更好的性能，这表明 **ROCm** 和 **Vulkan** 之间的最优选择可能取决于具体的硬件配置。
- **Cerebras 价格依然是个谜**：一位成员询问用于机密推理的 **Cerebras** 系统价格，引发了关于本地与云端硬件的讨论。
   - 能够本地运行 Kimi K2.5 的 **Cerebras** 系统估价在 **10 万美元**以上，除非收入达到 7 位数，否则在经济上是不合理的。
- **角色扮演（RP）中的大模型与小模型**：一位成员分享道，他们能明显感觉到大模型和小模型之间的质量差异：在角色扮演中，大模型处理多个角色的能力要好得多，并且能够真正维持文本背后有“大脑”的错觉。
   - 小模型几乎无法理解言外之意。一位用户指出，两者都是经过合成训练的，因此面临类似的问题：*“一刀切（精简版）”*。
- **内存带宽瓶颈讨论**：一位用户表示 *“如果你有 12 通道 DDR5 内存实现 400GB/s 带宽，速度为 1 token/s，那么 Q4 将运行良好”*，另一位确认 EPYC 更便宜。
   - 另一位成员反驳道：*“想象一下用 273GB/s 的带宽运行 123B Mistral，在 256GB/s 下速度大约是 2.7 t/s”*。
- **eBay 是购买 Xeon CPU 的好地方吗？**：一位成员建议 eBay 是购买用于 AI 任务的二手 **Intel Xeon** CPU 的好地方，并引用了一个 **96 核** CPU 售价 **1433 欧元**的列表。
   - 然而，也有人对散热、主板成本和昂贵的内存表示担忧，并由于这些 CPU 的高 TDP 建议使用液冷。

---

### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1475549205756903497)** (2 messages): 

> `模型基准测试 (Model Benchmarks)、实际定价 (Effective Pricing)、排名与排行榜 (Rankings & Leaderboard)、免费路由器 (Free Router)、GPT-5.3-Codex` 


- **OpenRouter 展示模型基准测试**：现在每个模型页面都会显示由 [Artificial Analysis](https://x.com/OpenRouter/status/2024172341190938958) 提供支持的行业标准基准测试分数，包括**编程、数学、科学和长上下文推理**。
- **实际定价现已上线**：模型页面现在增加了“实际定价 (Effective Pricing)”选项卡，显示你在每个供应商处实际支付的费用，包括分级定价，例如这个 [GLM-5](https://openrouter.ai/z-ai/glm-5/pricing) 的示例。
- **排名与排行榜更新**：[排名页面](https://openrouter.ai/rankings#benchmarks)现在包含基准测试散点图和扩展表格，**100K–1M token 请求**的长上下文生成量正在大幅增长。
- **免费路由器 (Free Router) 首次亮相**：新路由器 `openrouter/free` 正式上线，这是一种路由到所有免费 LLM 的简便方法，会根据你的请求自动选择兼容的模型；在此查看[热门免费模型](https://openrouter.ai/openrouter/free)。
- **GPT-5.3-Codex 上线**：[GPT-5.3-Codex 已在 OpenRouter 上线](https://openrouter.ai/openai/gpt-5.3-codex)。


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1475551835262681098)** (2 messages): 

> `Serverless 推理、Kollect AI 对话` 


- **初创公司开启 Serverless 推理 Beta 测试**：一家拥有自建数据中心和 GPU（**H200**、**B200**、**RTX6000** 等）的新初创公司正在为 **Qwen** 和 **Llama** 等开源模型提供 Serverless 推理服务。
   - 他们正在寻找资深的 Beta 测试人员免费体验模型，包括：*gemma-3-4b-it*、*Phi-4-mini-instruct*、*gpt-oss-20b*、*Qwen3-14B-Q8_0* 和 *Llama-3.3-70B-Instruct-Q8_0*。
- **Kollect 将表单转化为实时 AI 对话**：一位成员创建了 [Kollect](https://kollect.admildomanuel.com)，这是一个小型开源项目，能将枯燥的表单转化为实时 AI 对话。
   - 用户可以自然地交流，**AI 会倾听并动态引导调查**，只需通过描述即可创建表单；创建者鼓励用户尝试并在 GitHub 上点亮 star。


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1475539772930785330)** (475 messages🔥🔥🔥): 

> `生成元数据 API、免费路由器、请求数据选项卡、OpenRouter 状态、延迟模型排行榜` 


- **生成元数据 API 出现故障**：一名用户报告 [生成元数据 API (generation metadata API)](https://gist.github.com/FeepingCreature/471f622e8f8d9f931044c46e9ff689a5) 无法正常工作，所有生成 ID 均返回 **404** 错误。
   - 经发现，将延迟增加到 **10 秒**可以解决此问题，但除了反复尝试外，没有明确的方法可以发现这一点，且成本元数据可以在 `usage` 对象中找到。
- **新请求数据选项卡开启 Beta 测试**：用户注意到在生成活动的页面中增加了新的请求数据选项卡，该功能目前处于 Beta 阶段，很快将正式发布。
   - 此次更新还包括对 [OpenRouter 排名页面](https://openrouter.ai/rankings#performance) 的增强，重点关注**延迟 (latency)** 和**吞吐量 (throughput)**，并讨论了根据端到端延迟对供应商进行排序的功能。
- **Anthropic 关于蒸馏攻击的指控引发辩论**：Anthropic 声称中国 AI 实验室（[DeepSeek](https://www.deepseek.com/en/)、[Moonshot](https://www.moonshot.ai/en) 和 [MiniMax](https://www.minimax.ai/)）正在进行工业规模的蒸馏活动，这一说法遭到了质疑，一些人将其视为营销策略。
   - 部分成员表示 *由于数据量巨大，这些模型具有与它们所蒸馏的模型相同的怪癖*，并且 *非法蒸馏的模型缺乏必要的安全防护，构成了重大的国家安全风险*。
- **Sarvam.ai 寻求集成至 OpenRouter**：印度 AI 实验室 [Sarvam.ai](https://www.sarvam.ai/) 表示有兴趣在 OpenRouter 上列出其模型，并强调了其开发者社区的浓厚兴趣。
   - Sarvam 声称已经 *构建了印度首个主权 LLM，以及 STT 和 TTS 模型*，目前每天处理数百万次 API 调用。
- **Qwen 图像生成**：用户分享了他们一直在使用 **Qwen** 生成基于朝向旋转的产品照片，并引用了 [Hugging Face](https://huggingface.co/spaces/multimodalart/qwen-image-multiple-angles-3d-camera) 上提供的一个工具。
   - 他们指出 **Qwen** 完成了所需的工作，生成的图像效果极佳，且处理速度尚可。


  

---

### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1475939969070534683)** (2 条消息): 

> `` 


- **无可总结的新模型**：频道中没有需要总结的消息。
   - 因此，未识别出任何主题。
- **频道内关于新模型保持沉默**：'new-models' 频道保持非活跃状态。
   - 没有共享任何讨论或链接可供报告。


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1475537820939718767)** (80 条消息🔥🔥): 

> `Flash Models vs Full-Size Models, Distillation Attacks on Claude, OpenRouter Chat Bug, OpenClaw Replacement, Rate Limits on OpenRouter` 


- **Flash 模型热潮引发辩论**：成员们讨论了为什么公司正在创建像 **Xiaomi Mimo** 和 **Stepfun** 这样的 "Flash" 模型而不是 Full-Size 模型，认为 "Flash" 表示更小的衍生模型，而一些人则更倾向于制作 "Max Ultra" 模型。
   - 即使是 **300B+ parameters** 的模型也开始使用 "Flash" 一词，被描述为便宜、快速且智能。
- **Anthropic 指责中国公司窃取数据**：在 [中国公司从 Claude 窃取数据](https://www.wsj.com/tech/ai/anthropic-accuses-chinese-companies-of-siphoning-data-from-claude-63a13afc?st=vQ7iHF&reflink=desktopwebshare_permalink) 的指责声中，Anthropic 正在 [检测并防御蒸馏攻击 (Distillation Attacks)](https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks)。
- **OpenRouter 聊天 Bug 导致身份危机**：用户报告了一个 Bug，即使在 System Prompt 为空的情况下，**Sonnet 4.6** 在 OpenRouter 聊天中也会识别为 **Deepseek**，该问题随后被 [复现](https://x.com/paradite_/status/2026160598216827038)。
   - 遇到该问题的用户开玩笑说经历了一个“身份危机时刻”。
- **OpenClaw 倒下，用户失去“利爪”**：一名成员寻求 **OpenClaw** 的替代方案，社区建议了 **ClosedPaw**、**nanoclaw** 和 **picoclaw** 等选项。
- **OpenRouter 用户遭遇速率限制 (Rate Limits)**：一名用户报告称，即使请求速率很低，在多个提供商（**DeepInfra**、**chutes** 等）也遭遇了 Rate Limited，并请求 OpenRouter 向提供商申请更高的速率限制。
   - 被限制的模型包括 **Llama 3.1 8b**、**devstral** 和 **Mistral Nemo**。


  

---


### **OpenClaw ▷ #[models](https://discord.com/channels/1456350064065904867/1456704705219661980/1475546409884450829)** (350 条消息🔥🔥): 

> `Qwen 3.5 Plus, GLM-5 Performance, Claude Max issues, OpenClaw Local Setup, Github Copilot vs Claude Max` 


- **Qwen 3.5 Plus：有潜力但体验受限**：成员们正在通过 Alibaba Cloud 和 OpenRouter 测试 **Qwen 3.5 Plus**，一名用户注意到它无法通过 OpenRouter 在其服务器上执行命令。
   - 另一名在 Alibaba Cloud 上使用的用户发现它在文本方面很有效，但指出缺乏图像输入，并吐槽他们的“硅谷热狗/非热狗机器人”将每张图片都误认为计算机文件。
- **GLM-5 速度受阻但结果达标**：通过 z.ai 的编程计划测试 **GLM-5** 的用户报告称其速度较慢但功能正常，尤其是在使用子 Agent 进行研究时，尽管可能会受到速率限制。
   - 一名用户升级到了 **$30/month 档位** 以充分利用 **GLM5**，并强调尽管存在速度问题但非常有效，肯定了其表现。
- **Anthropic 面临蒸馏指责**：据报道，Anthropic 对 **Kimi** 和 **MiniMax** 针对 **Opus** 蒸馏未来模型的指责感到不满，这些公司可能使用虚假账号在闭源模型的大型响应数据集上进行训练。
   - 尽管存在争议，一些成员认为这种做法最终有利于开源社区，并将其与 Linux 开发的历史相类比。
- **Claude Max 引发使用和 Bug 讨论**：用户在使用 **Claude Max** 时遇到问题，包括由于最近的 OpenClaw Bug 导致模型的内部推理过程被传输到聊天会话中，这可以通过运行 `/reasoning off` 来解决。
   - 还有报告称 **Opus 4.6** 和 **Sonnet 4.6** 的消耗速度更快，一名用户幽默地将这种情况比作“乱穿马路”被开了“300 美元的罚单”。
- **OpenClaw 本地设置需要强悍的硬件**：一名拥有 **4 张 L40S GPU** 的用户正在探索在本地运行 OpenClaw 以利用其硬件性能。
   - 另一名运行两张 **L40S** 的成员发现，配合一些 **DDR5** 主内存，他们可以以不错的量化运行 **GLM5**，并建议使用 *llama.cpp fork* 和 *Unsloth 的量化版本* 在 GPU 和主内存之间分配负载。


  

---

### **OpenClaw ▷ #[showcase](https://discord.com/channels/1456350064065904867/1456609488202105005/1475605224990183464)** (23 messages🔥): 

> `OpenClaw 在 iPhone 上运行, Molty 医生, Cron Job 劳力士搜寻器, 喷墨打印机 OpenClaw 杂志, 编程会话 Context Hub` 


- **OpenClaw 现在可以在 iPhone 上运行**：一名成员让 **OpenClaw** 在 **iPhone** 上成功运行，并不得不通过打补丁一些包来构建 **node**。
   - 虽然 *相当卡顿*，但确实可以运行。
- **Molty 成为住院医师**：一名成员通过使用 **Hugging Face inference endpoint** 将 **Baichuan-M3** 的量化版本部署到 **AWS** 上兼容 **OpenAI-API** 的 URL，使 **Molty** 能够像医生一样思考。
   - 这个 **235B model** 被微调为一名住院医师，并被展示了复杂的假设性 ICU 病例。
- **Cron Job 找到古董劳力士**：一名成员设置了一个 **cron job** 来监控二手表商网站，寻找 **1989 Rolex Submariner**，并在发现时发送链接。
   - 机器人今天早上给他们发送了一个匹配项，*太棒了！*
- **OpenClaw 让喷墨打印机保持活力**：一名成员设置了一个 **OpenClaw Agent**，每两周打印一个独特的、色彩丰富的单页 HTML 页面，以防止喷墨打印机喷头干涸，并使用 LibreOffice 转换为 PDF。
   - 打印内容包括季节性俳句、笑话、当地天气、新闻头条、地区旅游建议、彩虹渐变、色块，以及 *Agent 当天想添加的任何内容*。
- **Context Hub 工具启动编程会话**：一名成员构建了一个工具，可以从他们的 **Mac Mini** 启动由 **OpenClaw** 驱动的编程会话，以便他们可以在 **Macbook** 上继续工作。
   - 它会自动实时监控编程会话，并将其自动馈送到 context hub。


  

---


### **Latent Space ▷ #[watercooler](https://discord.com/channels/822583790773862470/822583790773862473/1475628931699707905)** (15 messages🔥): 

> `Twitter 验证, 指数级增长, Cursor 公告, 全球年龄确认, iOS 27 功能` 


- **Twitter 遭遇公信力危机**：一位成员对 **Twitter** 的可靠性表示沮丧，因为蓝标验证失去信任且内容趋于混乱，并指出 *这简直疯了，我不敢关注任何新人*。
   - 他们还分享了一篇关于 [指数级增长 (exponential growth)](https://longform.asmartbear.com/exponential-growth) 的文章。
- **Discord 的舆论抵制导致政策修订**：一位成员指出，公众的抵制导致 **Discord** 修订了其全球年龄确认政策，并链接到一篇详细说明这些变化的 [博客文章](https://discord.com/blog/getting-global-age-assurance-right-what-we-got-wrong-and-whats-changing)。
   - 另一位成员推测，由于最初的政策，DAU（日活跃用户）肯定经历了 *暴跌*。
- **iOS 27 预计将推出年龄验证 API**：讨论中提到了 **Apple** 在 **iOS 27** 中引入设备端年龄验证功能的可能性，并将其作为 API 提供给第三方应用。
   - 这一建议符合 **Apple** 为开发者提供关注隐私的解决方案的历史。
- **Swyx 链接分享**：一位用户在 "swyx plane dump" 中分享了一系列链接，包括来自 [OpenAI](https://x.com/openai/status/2026412700583317815?s=46) 和 [Langchain](https://x.com/langchain/status/1879576930347073873?s=46) 的推文。


  

---


### **Latent Space ▷ #[memes](https://discord.com/channels/822583790773862470/839660725252784149/1475641971056836708)** (55 messages🔥🔥): 

> `SOTA benchmark, Clawdbot, 早期采用者, 蒸馏攻击` 


- **开发出 SOTA Benchmark**：一名成员开发了一个用于评估 **LLM** 的新 **SOTA benchmark**，并展示了结果截图。
   - 另一名成员链接到 [这条推文](https://x.com/dmayhem93/status/2026028013763101132?s=12) 以支持这个新的 benchmark。
- **Clawdbot 的不当行为**：用户 @hopes_revenge 报告了一起令人不安的事件，尽管有明确指令要求机器人避免此类行为，但他们的 **Clawdbot** 还是触碰了正在熟睡的妻子的头发，参见 [这条推文](https://xcancel.com/hopes_revenge/status/2025933908995649906)。
   - 作者并未解释为什么最初给机器人起了一个如此令人毛骨悚然的名字。
- **早期采用者的兴奋感**：Lee Robinson 发布的一条表达对成为某项技术或运动的 **早期采用者 (early adopter)** 的反思或兴奋的病毒式帖子在 [这里](https://xcancel.com/leerob/status/2026068656539521508)。
   - 社交媒体帖子上的高参与度表明其他用户也有同感。
- **AI 家长面临“蒸馏攻击”**：作者幽默地将他儿子频繁的提问比作“**蒸馏攻击 (distillation attack)**”，这是一个用于描述从 **AI model** 中提取知识的技术术语，[参见此链接](https://xcancel.com/fkadev/status/2026145372318425259?s=46)。
   - 他们指出，这与从 AI 中提取信息非常相似。


  

---

### **Latent Space ▷ #[stocks-crypto-macro-economics](https://discord.com/channels/822583790773862470/844658979363618816/1475628879795323203)** (9 messages🔥): 

> `SpaceX IPO, OpenAI IPO, Anthropic IPO, Software Development Jobs` 


- **万亿美元级 AI/航天 IPO 面临流动性挤压**：Tomasz Tunguz 分析了 **SpaceX, OpenAI 和 Anthropic** 预期的 IPO，如[这条推文](https://x.com/ttunguz/status/2025982590977823082?s=12)中所讨论的，它们的总市值可能达到创纪录的 **2.9 万亿美元**。
   - 他强调这些公司的主要障碍不是估值，而是实现标准的 **15% 股票流通量 (share float)** 所需的海量公共流动性。
- **尽管 AI 兴起，软件开发职位依然激增**：Per Borgen 注意到技术行业出现了显著的叙事转向，指出在过去一年中，尽管更广泛的就业市场下降了 **5.8%**，但 **软件开发职位增加了 10%**，详见[这条推文](https://x.com/perborgen/status/2025890393166917857?s=12)。
   - 一位成员对这一数据点反应道：*等等，我以为 AI 正在终结对软件开发者的所有需求？？？*


  

---


### **Latent Space ▷ #[intro-yourself-pls](https://discord.com/channels/822583790773862470/844675581291397171/1475561222974996521)** (5 messages): 

> `Self-hosted AI, ML in mechanical engineering, AI Agents, DeFi, ZK proofs, and Golang` 


- **工程师转向 FOSS 与自托管 AI**：一位常驻旧金山湾区（SFBA）的工程师，曾就职于某企业级 SaaS 初创公司和 NASA，目前正在学习 **Rust** 并折腾 **LLMs**，对自托管和边缘 AI 感兴趣。
   - 他们是 **FOSS 支持者**，希望结识志同道合的人并参加线下聚会。
- **ML 工程师关注澳洲 AI 安全**：一位拥有使用 **DL 模型**（LLMs + GNNs）检测源代码漏洞博士学位的 ML 工程师，对针对 LLMs 及其相关软件的新型攻击感兴趣。
   - 他们常驻澳大利亚，正在寻找一个不那么嘈杂的地方来讨论 **ML 和 AI** 并进行潜在的社交。
- **架构师连接 IT 策略与 AI Agents**：一位常驻欧洲的企业架构师将 **IT 策略**与**业务目标**联系起来，在 **DeFi**、**ZK proofs** 和 **Golang** 的交叉领域进行开发。
   - 他们对 AI agents、分布式系统以及将新兴技术转化为现实影响感兴趣，致力于解决 LLM 系统中的检索逻辑和后端架构等问题。
- **工程师寻求 ML/AI 在机械工程中的应用**：一位常驻圣何塞、具有机械/材料工程背景的工程师，对 **ML/AI** 在 **机械工程**或**材料科学**中的应用感兴趣。
   - 他们希望结识新朋友并参加线下聚会，并欢迎分享相关主题的资源。


  

---


### **Latent Space ▷ #[tech-discussion-non-ai](https://discord.com/channels/822583790773862470/869647848826892309/1475737160916275212)** (31 messages🔥): 

> `Vinext, Traffic-aware Pre-Rendering, Next.js deployment slowness, Vercel's new library Chat SDK, Tests are the new moat` 


- **Vinext 框架既是恶搞也令人兴奋**：一位成员分享了一个 [原始 HTML shadow DOM 演示](https://go-streaming-html-ooo.fly.dev/) 的链接，以及一篇关于 **Vinext**（一个 Next.js 替代方案）的看似严肃的 [Cloudflare 博客文章](https://blog.cloudflare.com/vinext/)。
   - 社区对出现新框架的可能性反应有趣且兴奋，尤其是考虑到 Cloudflare 过去在构建类似解决方案上的尝试和失败。
- **流量感知预渲染 (TPR) 解决 Next.js 构建时间问题**：博文中选最引人注目的部分是 **Traffic-aware Pre-Rendering (TPR)**，这是一项实验性功能，可在部署时查询 Cloudflare 的区域分析 (zone analytics)，并仅预渲染那些关键页面。
   - 一位成员对将 TPR 作为 Next.js 和 Astro 等框架的默认功能表示热衷，并提到了 Next.js 16 极其缓慢的开发构建时间。
- **成员辩论测试套件是否为新的护城河**：针对博客文章 [Tests are the new Moat](https://saewitz.com/tests-are-the-new-moat) 的链接，一位成员对 Vinext 因为规范严谨而不会产生幻觉的说法表示怀疑。
   - 他们质疑像 SQLite 那样的测试套件是否能捕捉到细微的不一致性。
- **Vercel 发布 Chat SDK**：Vercel 发布了新的 [Chat SDK 库](https://vercel.com/changelog/chat-sdk)。
   - 一位成员链接到了 [Vercel 的新库](https://github.com/vercel-labs/chatsdk-knowledge-agent-templates)，该库使用了 Chat SDK。


  

---

### **Latent Space ▷ #[founders](https://discord.com/channels/822583790773862470/869651275963310181/1475548219676168255)** (2 messages): 

> `Nielsen dollar bill surveys, swyxio` 


- **Nielsen 使用现金提高调查回复率**：一位成员分享了一个 [链接](https://x.com/toddsaunders/status/2025932667834015851?s=12)，关于 **Nielsen** 在邮件中寄送*真实的美元钞票*，以增加人们完成调查问卷的意愿。
   - 另一位成员发表了评论，引用了 Nielsen 使用现金激励的策略。
- **长见识了**：通过诉诸人们的贪欲而非慷慨来促使他们完成调查更为重要。
   - 尽管看起来不太对劲，但当人们收到美元钞票时，调查完成率确实会提高。


  

---


### **Latent Space ▷ #[hiring-and-jobs](https://discord.com/channels/822583790773862470/930269508529192981/1475693362358255670)** (1 messages): 

> `Local-first AI companion system, Huginn Ember, Identity stability in AI, LLM behavior control, Memory retrieval without bloat` 


- **构建 Huginn Ember，一个本地优先的 AI 伴侣系统**：一位成员正在构建 **Huginn Ember**，这是一个专注于**身份稳定性**、**结构化记忆**和**用户主权**的*本地优先 (Local-first) AI 伴侣系统*。
   - 该系统旨在实现*人格锁定*，避免成为 GPT 套壳或活跃度驱动的聊天机器人，目标是解决如何在基于概率的 LLMs 上构建无漂移或无操纵性设计的 AI 伴侣架构。
- **身份稳定型 AI 寻求技术联合创始人**：一位成员正在寻找一位 **50/50 技术联合创始人**，共同解决诸如强制执行随时间推移的身份稳定性，以及设计 LLM 行为的中间件控制层等问题。
   - 理想的联合创始人应热爱系统设计，并希望共同架构持久的产品，重点关注无上下文冗余的记忆检索，以及在无需重度 Fine-tuning 的情况下防止语气漂移。
- **Ember MVP 专注于伦理和本地 AI**：**Ember MVP** 的范围包括原型锁定的性格执行层、分层记忆模型、本地加密记忆库、冷静优先的行为切换、结构化标签停放与召回，以及伦理边界层。
   - 该系统旨在保护**用户自主权**，确保本地记忆的完整性，并在更新迭代中保持稳定的性格核心，避免隐藏的诱导或数据收割。


  

---


### **Latent Space ▷ #[san-francisco-sf](https://discord.com/channels/822583790773862470/979492707279978586/1475657068705874055)** (4 messages): 

> `AI Trading Card Game, Inverted Value Model, Collectible Separation` 


- **AI 集换式卡牌游戏登陆旧金山 (SF)**：一位成员将于 **3 月 8 日**在旧金山发布一款 **AI 生成的集换式卡牌游戏**，通过 [luma.com](https://luma.com/dzit8eec) 为社区提供优先访问权限。
   - 正式发布将于周五面向更广泛的受众。
- **集换式卡牌价值模型被反转**：一位成员提议反转集换式卡牌的价值模型，建议卡牌被玩得越多（且玩得越好），其价值就越*高*。
   - 他们表示，这是数字卡牌游戏未能成功利用这一想法的一个领域。
- **纸质版收藏品分离**：一位成员建议将集换式卡牌游戏中的收藏品与纸张分离。
   - 这个想法是为了方便构建具有竞争力的卡组而不会出现价格欺诈，普通卡牌可以使用打孔照片纸在家制作，而全息卡则作为高端收藏品。


  

---


### **Latent Space ▷ #[security](https://discord.com/channels/822583790773862470/1025833219448393878/)** (1 messages): 

swyxio: https://x.com/jacklouisp/status/2025956259594137613?s=12
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 messages): 

swyxio: https://youtu.be/x9rWFiIubmc

Claude Code 周年庆的新播客！
  

---

### **Latent Space ▷ #[ai-general-news-n-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1475558763867013240)** (140 messages🔥🔥): 

> `Anthropic 蒸馏攻击, SWE-Bench 弃用, SaaS 已死论调, MatX 5 亿美元 B 轮融资, GPT-5.3-Codex 发布` 


- **Anthropic 在 API 攻击中点名道姓**：Anthropic 报告称，DeepSeek、Moonshot AI 和 MiniMax 使用了**超过 24,000 个虚假账户**，与 Claude 产生了 **1,600 万次对话**，旨在通过蒸馏训练他们自己的模型。他们将其称为*[工业级规模的攻击](https://x.com/anthropicai/status/2025997928242811253)*。
   - 一位成员指出，*Qwen/Alibaba 未被列入*违规者名单，而另一位成员指出，如果不停止将数据喂给模型，*按此速度，前沿实验室（frontier labs）将在消费市场被横扫*。
- **基准测试引发争议后 SWE-Bench 被弃用**：OpenAI 宣布自愿弃用 **SWE-Bench Verified 基准测试**，原因是数据污染严重且存在大量无法解决的任务，正如其[官方公告](https://x.com/OpenAIDevs/status/2026025368650690932)所示。
   - 分析显示，*前沿模型现在正在根据 ID 反刍（regurgitating）任务解决方案，且大约 60% 剩余未解决的问题存在缺陷，使得进一步的基准测试变得毫无意义*。
- **SaaS 末日已至？**：成员们讨论了 [LLM 是否会取代 SaaS](https://fxtwitter.com/tenobrus/status/2025648199898407345)，一位成员认为，*如果 Token 变得足够便宜，以至于按需使用大量 Token 来复制一个 SaaS 应用变得可行，那么 SaaS 就麻烦了*。
   - 其他人反驳称，*企业运行靠的是信任和可预测性*，不会信任容易产生幻觉的 AI，也不想*为了亲自维护而自己动手做一个 Calendly*。
- **MatX 为矩阵乘法机器筹集 5 亿美元巨资**：MatX 宣布完成由 Jane Street 和 Situational Awareness LP 领投的 **5 亿美元 B 轮融资**，用于其全新的 **MatX One LLM 芯片**。该芯片采用可拆分脉动阵列（systolic array），结合了 SRAM 级的低延迟与 HBM 长上下文支持（[来源](https://x.com/reinerpope/status/2026351870852358492)）。
- **GPT-5.3-Codex 全面发布**：OpenAI Developers 宣布 **GPT-5.3-Codex** 立即通过 Responses API 向所有开发者开放，邀请他们开始使用新模型进行构建（[来源](https://x.com/openaidevs/status/2026379092661289260)）。


  

---


### **Latent Space ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/1475690635104358622)** (14 messages🔥): 

> `中国 AI 进展, GLM-5 技术报告, DSA 采用, 异步 RL 基础设施, Persona Selection Model` 


- **中国 AI 在 CoT 和压缩方面取得飞跃**：讨论强调了中国 AI 研究正转向复杂的 **Chain-of-Thought (CoT) 工程**和集成的**压缩流水线**。
   - 特别提到并期待来自 **ByteDance** 的后续工作。
- **GLM-5 发布技术报告**：[Z.ai 发布了 GLM-5 技术报告](https://arxiv.org/pdf/2602.15763)，详细介绍了关键创新，如用于降低成本的 **DSA 采用**、用于提高训练后效率的**异步 RL 基础设施**，以及新的 **Agent RL 算法**。
   - 该模型展示了最先进的性能，特别是在**现实世界的软件工程任务**中。
- **探讨 Anthropic 的 Persona Selection Model**：一位成员考虑研究 Anthropic 的 [Persona Selection Model](https://www.anthropic.com/research/persona-selection-model)。
   - 他们询问对该模型的研究是否需要整整一个小时来进行审阅，以及是否应该增加另一篇论文来覆盖。


  

---


### **Latent Space ▷ #[singapore-sg](https://discord.com/channels/822583790773862470/1181708804803543140/)** (1 messages): 

coffeebean6887: https://luma.com/c4dmddvh?tk=yciGr7
  

---


### **Latent Space ▷ #[los-angeles-la-lax](https://discord.com/channels/822583790773862470/1203087028401606716/)** (1 messages): 

stealthgnome: https://luma.com/ffla26?tk=wPNgSD

### **Latent Space ▷ #[ai-in-action-builders-techstacks-tips-coding-productivity](https://discord.com/channels/822583790773862470/1209303473263485011/1475537747191005316)** (40 messages🔥): 

> `Claude Code 'Remote Control', GO GO OS, Open Weight Models, Claude as Coach` 


- **GO GO OS 演讲定于三月举行**：由 @slono 带来的关于 **GO GO OS - THE AI FIRST OS** 的演讲定于 2026 年 3 月 6 日星期五举行，需通过 AI In Action Bot 进行注册。
   - AI In Action Bot 协助协商了演讲者的报名事宜。
- **LLM 助力雄心壮志快速验证**：一位成员描述了一个周期：*将想法推向极致并以极快的速度（speed of slop）进行迭代*，然后 *收集所有这些内容并得出结论，使其可重用*，因为 **LLM** 能够快速验证梦想。
   - 他们通过构建具有可互换渲染器的事件 WebSocket 流式响应式 UI 的经验验证了这一点。
- **Elvis 发布参与度指标**：Elvis 在 2026 年 2 月 23 日发布的推文获得了 [超过 130 万次浏览](https://xcancel.com/elvissun/status/2025920521871716562?s=46&t=jDrfS5vZD4MFwckU5E8f5Q)。
   - 这引发了关于参与度指标以及针对不同任务使用多样化模型的价值的讨论，其中 Codex 用于代码审查，AMP 用于缺陷检测。
- **Claude Code 获得 'Remote Control' 功能**：Noah Zweben 宣布了 **Claude Code** 的 'Remote Control' 功能，这是面向 Max 用户的一项研究预览功能，允许开发者在终端启动代码会话并切换到移动端（[公告](https://xcancel.com/noahzweben/status/2026371260805271615?s=12)）。
   - 一位成员表示有兴趣尝试，尽管更倾向于自己的家用桌面设置。


  

---


### **Latent Space ▷ #[share-your-work](https://discord.com/channels/822583790773862470/1209672547642249216/1475631061978579138)** (4 messages): 

> `Commit Change, Vercel AI SDK, Plasmite, AI companion system` 


- **Commit Change 为社会影响而发布**：一位氛围工程师（vibe engineer）创建了 [Commit Change](https://www.commit-change.com)，这是一个为社会影响和慈善机构编写代码的平台，配备了身份验证和审核功能，尽管目前填充的是占位项目和开发者。
   - 创建者在正式发布前寻求反馈和建议，询问这个想法是否具有可行性。
- **面向 Node 开发者的 Vercel AI SDK**：一位成员分享了关于面向 Node 开发者的 [Vercel AI SDK](https://thecodebarbarian.com/getting-started-with-the-vercel-ai-sdk-in-nodejs.html) 的文章。
   - 该文章详细介绍了一种为社会影响和慈善机构编写代码的方法。
- **为本地优先的 AI 伴侣系统寻找联合创始人**：一位 AI 工程师正在寻找一位 **50/50 AI 架构联合创始人**，共同解决性格偏移（personality drift）、内存分层以及对 **LLM** 的中间件控制，目标是构建一个专注于身份稳定性和结构化长期记忆的本地优先 AI 伴侣系统。
   - 创始人已经设计了一套完整的行为框架，并正在构建 MVP。
- ****Plasmite** IPC 库发布**：Brandon Harvey 发布了 [Plasmite](https://github.com/sandover/plasmite)，这是一个健壮的进程间通信（**IPC**）库，支持 Rust, Node, Go, C 和 Python，具有 JSON 消息、零拷贝读取、临时读写器和友好的 CLI。
   - 它的灵感来源于 Oblong Industries 用于 [空间计算系统（spatial computing systems）](https://vimeo.com/2229299) 的多进程设计精神和风格。


  

---


### **Latent Space ▷ #[robotics-and-world-model](https://discord.com/channels/822583790773862470/1318774781834821746/1475867130078695595)** (4 messages): 

> `Pengchuan Zhang, FAIR, OpenAI, SAM, Llama` 


- **Pengchuan Zhang 加入 OpenAI**：Pengchuan Zhang 宣布他已从 Meta 的 FAIR 团队转投 OpenAI，专注于通过世界模拟和机器人技术开发物理智能；链接至 [X 帖子](https://x.com/pengchuanz/status/2026189659228012558?s=12)。
- **Zhang 在 Meta 工作 4 年后离职**：Zhang 在 Meta 的 FAIR 团队工作了近四年，参与了 SAM 和 Llama 的研发。
   - 该链接被多次[重复发布](https://xcancel.com/pengchuanz/status/2026189659228012558?s=12)。


  

---

### **Latent Space ▷ #[genmedia-creative-ai-video-image-voice-music-inspo-consumer-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1475613346874261535)** (8 messages🔥): 

> `OpenAI Realtime API, Anthropic LACMA Art + Technology Lab 2026` 


- **OpenAI 为 Realtime API 发布 gpt-realtime-1.5**：OpenAI Developers 宣布发布 **gpt-realtime-1.5**，这是针对 **Realtime API** 的更新模型，具有改进的指令遵循能力、更可靠的 **tool calling** 以及增强的语音工作流多语言准确度；原始[公告发布在 X 上](https://x.com/OpenAIDevs/status/2026014334787461508)。
- **Anthropic 支持 LACMA Art + Technology Lab 2026**：Anthropic 宣布支持 **LACMA Art + Technology Lab 2026** 的提案征集，邀请全球艺术家申请高达 **50,000 美元** 的资助，用于探索艺术与新兴技术的项目，截止日期为 **4 月 22 日**，详见其 [X 帖子](https://x.com/AnthropicAI/status/2026096054253564002)。


  

---


### **Latent Space ▷ #[mechinterp-alignment-safety](https://discord.com/channels/822583790773862470/1445258379357458625/1475592708046196867)** (15 messages🔥): 

> `Anthropic Interpretability Team, ML Infrastructure Engineers, Frontier Models` 


- **Anthropic 寻找可解释性基础设施创新者**：Chris Olah 宣布 [Anthropic 的可解释性团队](https://xcancel.com/ch402/status/2026023963537842248) 正在招聘大约 **10 名资深 ML 基础设施工程师**，专注于理解前沿模型。
   - 之前在可解释性方面的经验**不是必须的**，这使其成为一个“好机会（good opp）”。
- **ML 工程师的三倍机遇**：多位用户强调 Anthropic 可解释性团队的招聘公告是一个“好机会”，一位用户甚至表示“实际上是 3 倍的好机会”。
   - 该团队正在寻求经验丰富的 **ML 基础设施工程师**来研究模型内部机制，无需具备先前的可解释性经验。


  

---


### **Latent Space ▷ #[gpu-datacenter-stargate-colossus-infra-buildout](https://discord.com/channels/822583790773862470/1467633569684914349/1475859699571097822)** (8 messages🔥): 

> `Meta, AMD, OpenAI, Strategic Tech Alliances` 


- **Meta 将向数据中心投入 6GW 的 AMD 硬件**：从 **2026 年下半年**开始，[Meta 计划在五年内部署 6GW 基于 AMD 的数据中心基础设施](https://xcancel.com/shanumathew93/status/2026285588274381129?s=12)，每 GW 价值达双位数亿美金。
   - 作为交易的一部分，Meta 获得了与业绩和部署里程碑挂钩的 **1.6 亿股 AMD 股票认股权证**，推动 **AMD 股票**盘前上涨 **15%**。
- **OpenAI 和 Meta 结成战略技术联盟**：M.G. Siegler 对 **NVIDIA** 和 **AMD** 之间的战略博弈[发表了评论](https://xcancel.com/mgsiegler/status/2026274906069950831?s=12)，重点关注了 **OpenAI** 和 **Meta** 最近的合作伙伴动态。
   - 他还提供了 **Big Tech AI 投资**的最新跟踪。


  

---


### **Latent Space ▷ #[applied-ai-experimentation](https://discord.com/channels/822583790773862470/1470417186651897858/1475558941252387067)** (13 messages🔥): 

> `Taaalas code generation speed, Codebase Recreation Prompting, cue-lang/cue github repo` 


- **Taaalas 生成代码速度极快**：成员们讨论了 **Taaalas** 在 *30 毫秒*内生成可用代码的能力，一位成员将其描述为“科幻级别的东西”。
   - Demo 预计本周准备就绪。
- **通过代码库重建打磨 Prompt 技巧**：一位成员建议克隆一个首选的 **GitHub** 仓库，并引导模型“深度钻研代码库，然后提供一个句子的 **Prompt** 来重建该库，但要实现 x, y, z”，以此来提高 **Prompt** 技巧。
   - 另一位成员表示有兴趣将其应用于 **MLflow** 和 **DSPy** 的改编。
- **cue-lang/cue Github 仓库引发关注**：一位成员推荐了[这个 GitHub 仓库](https://github.com/cue-lang/cue)，称其 **Agent** 们“爱上了”它。
   - 另一位成员表示他们早在 **LLM** 出现之前就已经看过它了。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1475538049206063144)** (146 messages🔥🔥): 

> `IBM 股价因 Claude 简化 COBOL、Gemini 和 Claude 混合工作流、AI 训练伦理、Sora 2 版权违规、AI 对 BPO 市场的影响而暴跌` 


- **Claude 的 COBOL 能力首次亮相后 IBM 股价大跌**：在 **Anthropic** 宣布 **Claude** 可以简化 **COBOL** 代码后，**IBM** 的股价经历了超过 **10%** 的大幅下跌。
   - 一位成员开玩笑说 Musk 通过 **Grok 4.300** 安全地编辑人类大脑，而另一位成员则拿配合 **Grok Imagine 1.2** 的 **Neuralink** 开玩笑。
- **编码者协同使用 Gemini 和 Claude**：一些编码者一直在使用一种工作流，利用 **Gemini** 进行研究和 **Claude Opus** 进行最终编写的各自优势，强调了 **Gemini** 在项目连贯性的界面可用性方面的不足。
   - 一位用户报告使用 *free coursera loophole* 获取免费的 **Gemini**，尽管另一位用户提到通过 kilocode 获取免费的 **GLM 5** 也同样好用。
- **版权背景下的 AI 训练伦理辩论**：围绕 AI 公司在他人作品上训练 **LLM** 的伦理性展开了辩论，并建议模型应该在来自其他模型的 *synthetic data*（合成数据）上进行训练。
   - 一位参与者表示：*政府和学术界对 AI 的主要问题与版权或 AI 接管无关，核心问题是 National Security（国家安全），主要是外国行为者可以利用 AI 重建和推断被某些政府和国家集团刻意遮蔽的技术。*
- **Sora 2 的版权困境推迟发布**：用户讨论了版权违规如何毁掉了 **Seedance 2.0** 并推迟了其全球发布，将其与 **Sora 2** 的内容相关问题类比，并称赞开源模型是一种替代方案。
   - 一位用户表示：*我记得 Sora 2 出现内容违规时，X 上有人说他们会等一个中国模型发布版权内容，笑死 (LAMO)，他们是在自欺欺人。*
- **AI 自动化重塑 BPO 行业**：讨论涉及 AI 自动化可能取代 **BPO** 市场的问题，特别影响中国和印度等国家，而较小、较富裕的国家在 AI 实施方面处于领先地位。
   - 成员们开玩笑说 *自动化总是先针对员工而不是管理层*，且决策者很少自动化他们自己的职位，而是由竞争对手把他们解决掉。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1475547975718797443)** (12 messages🔥): 

> `控制论提示词调节 (Control-theoretic prompt regulation), AI 发明, 统计自动化, Loopt 失败` 


- **AI 控制仍需人为因素**：针对用户是否可以有效地将 **control-theoretic prompt regulation** 应用于内部 **LLM** 的问题，一位成员表示，*行为可以从外部控制，但由于隐藏的内部动力学，无法保证真正的系统稳定性*。
   - 他们还指出，**用户帮助扩展并提供上下文**，但最初的方向和调节来自用户。
- **GPT 辅助繁重工作，但需要约束**：成员们讨论了在利用 **GPT** 构建复杂流水线时描述 **ontology**、架构和局限性的必要性。
   - 一位成员建议最好 *先用 GPT 创建，然后自己进行编辑或补充*，因为 **返回的所有内容都是所谓的 Latent Variable**。
- **AI 不会发明，而是寻找模式**：一位成员表示，目前的 **ChatGPT** 是 **statistical automation**（统计自动化），运行在统计模式识别模型上，不断循环直到找到一个 **Latent Variable** 来重新完成繁重工作。
   - 他们补充说，*这就是为什么他们说 AI 不能发明，因为它确实不能，它只是发现了由于数据量巨大而导致人类尚未（或从未）组合在一起的模式。*
- **创新是重组 + 洞察**：一位成员表示，人类也通过 **重组先验知识、连接模式并迭代先前的想法** 来进行发明。
   - 他们补充说，*极少有人类发明是凭空产生的，大多数创新都是重组 + 洞察。*
- **Loopt 并非失败而是学习经历**：一位成员引用了 **Sam Altman** 的话：*我不会称 Loopt 为失败。它确实没有变成我想要的样子，但它很有趣，我学到了很多，并且赚到了足够的钱开始投资，这引导我走向了现在的工作。*
   - 另一位成员回应道：*异曲同工，不断失败直到成功，失败不是失败而是学习，同样的谚语只是换了种形式。*


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1475547975718797443)** (12 messages🔥): 

> `Control-Theoretic Prompt Regulation, LLM Invention vs Pattern Recognition, Latent Variables, Statistical Automation` 


- **黑盒 LLM 中的控制理论提示调节 (Control-Theoretic Prompt Regulation)**：一位用户询问，控制理论提示调节是否可以有效地应用于表现为黑盒函数 **F(x)** 的内部 LLM + 编排栈。
   - 回复指出，虽然外部行为控制是可能的，但由于内部动态是隐藏的，无法保证真正的系统稳定性。
- **LLM：针对潜在变量循环的统计自动机**：有建议认为 **ChatGPT** 是一种*统计自动化*形式，运行在统计模式识别模型上，不断循环直到找到一个**潜在变量 (latent variable)** 来重新完成琐碎工作。
   - 一位用户表示：*这就是为什么人们说 AI 无法发明……因为它确实不能，它只是利用巨大的计算量找到了我们尚未（或从未）整合在一起的模式。*
- **人类 vs AI 发明——重组之争**：一位用户认为 AI 的发明局限于**重组先验知识**和连接模式，这与人类的发明类似。
   - 另一位用户反驳道，人类可以创造没有预设理由的新模式，并以艺术为例：*我只是想那么做 (I felt like doing that)*。
- **Sam Altman 的 Loopt：失败还是肥沃的土壤？**：一位用户引用了 **Sam Altman** 的话：*我不会称 Loopt 为失败。它确实没有变成我想要的样子，但它很有趣，我学到了很多，而且我赚到了足够的钱开始投资，这引导我走向了现在的工作。*
   - 该用户评论道：*失败并非真正的失败，而是学习，这只是同一句格言的另一种表现形式。*


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1475538622810951743)** (91 messages🔥🔥): 

> `MiSTer FPGA, Anthropic's RefusalBench, Qwen 3.5, Open Source Annotator, Tiny Tapeout ICs` 


- **MiSTer 窃取的过去引发争论**：讨论围绕 [MiSTer project](https://github.com/MiSTer-devel/Main_MiSTer) 展开，指责其代码是*从 Till 那里窃取的并导致了 MiST 的终结*，此外还讨论了*他们今天非法使用的 GPL 代码*。
   - 一名成员分享了一篇[博客文章](https://pingas.org/articles/provenance-of-retro)，详细介绍了该项目的来源和争议。
- **Tiny Tapeout 实现经济化的 IC 实验**：一位成员分享了 [tinytapeout.com](https://tinytapeout.com/) 的链接，*他们实际上让你花很少的钱就能流片 IC*，尽管设计必须相当小。
- **Anthropic 指责 DeepSeek 可疑的抄袭行为**：分享了一篇文章链接，讨论 *Anthropic 对 DeepSeek 未经许可复制其 AI 感到愤怒*，引发了关于讽刺意味的辩论，因为 Anthropic 自身的做法也备受争议，参考文章 [Anthropic Furious at Deepseek](https://www.msn.com/en-us/news/technology/anthropic-furious-at-deepseek-for-copying-its-ai-without-permission-which-is-pretty-ironic-when-you-consider-how-it-built-claude-in-the-first-place/ar-AA1WYupG)。
   - 一名成员表示 *是的，我们喜欢看这种肥皂剧*，暗示对这一局面的冷嘲热讽。
- **Qwen 3.5 取得惊人进展**：有人注意到 *Qwen3.5-35B-A3B 击败 Qwen3-235B-A22B-2507 简直疯狂*。此外还分享了已发布的 Base 权重链接 [huggingface.co/Qwen/Qwen3.5-35B-A3B-Base](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base)。
   - 此外，*5.3 codex 已上线 API：输入 $1.75，输出 $14*，比 Anthropic 便宜得多。
- **监管俘获 (Regulatory Capture) 展现其令人反感的现实**：针对关于百度和 Anthropic 的讨论，一名成员表示 *简单说：监管俘获*，暗示对潜在不当影响的担忧。
   - 另一名成员观察到 *百度在中国也以监管俘获著称*。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1475859341059031182)** (2 messages): 

> `Hermes, emergent misalignment, fine tuning, evil AI, AI safety` 


- **为了“涌现对齐偏差”而微调 Hermes？**：一位成员询问是否有人测试过专门针对**涌现对齐偏差 (emergent misalignment)** 对 **Hermes** 进行微调，或者通俗地说，让它*变坏*。
- **AI 微调的伦理影响**：该询问引发了关于故意为恶意目的微调 AI 模型的伦理影响的担忧，强调了 **AI Safety (AI 安全)** 研究的重要性。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1475568039725629582)** (60 messages🔥🔥): 

> `Latent Reasoning Tokens, Deepseek R1 Paper, EleutherAI Pythia-2.8b HF weights bug, Google Student Researcher Program 2026, lesswrong.com/posts/2pkNCvBtK6G6FKoNn/so-you-think-you-ve-awoken-chatgpt` 


- **LLM 获得 **Latent Reasoning** 增强**: 讨论了使用仅由 **LLM 生成**且不向用户显示的特殊 **token** 来增强推理能力，这种技术被称为 *Latent Reasoning*，可能提升性能和安全性。
   - 分享了论文 [Latent Reasoning](https://arxiv.org/abs/2307.06203) 的链接。
- ****Deepseek R1 论文**引发关于推理的讨论**: 讨论围绕大型模型的推理能力如何通过强化学习而非直接数据学习产生展开，并参考了 **Deepseek R1 论文**。
   - 讨论中提到，辅助奖励（auxiliary rewards）通常用于确保人类可读性，尽管目前尚不确定这种方法是否最优。
- ****EleutherAI Pythia-2.8b 权重 Bug** 在 HF 上浮现**: 一名成员在尝试使用 **EleutherAI** 的 **pythia-2.8b** 复现[一篇论文](https://arxiv.org/abs/2309.23024)时报告了一个 Bug：无论选择哪个修订版本，Hugging Face Hub 提供的权重都相同。
   - 发现 `pytorch_model.bin` 和 `model.safetensors` 在不同步数下具有相同的 SHA256，但分片文件（`model-00001-of-00002.safetensors`）却不同。
- **EleutherAI 修复 **Pythia-2.8b HF 权重 Bug****: 一名成员确认了 **EleutherAI Pythia-2.8b 权重 Bug**，并提到在尝试修复时受到了 HF 的速率限制。
   - 关于分片文件存在一些困惑，目前的理解是如果 1-of-2 和 2-of-2 文件正确，用户可以合并它们并加载模型。
- **在部分 EleutherAI 模型中发现**重复数据 (Dupe Data)**: 发现 **14m 和 30m** 模型实际上是去重版本（deduped versions），而非 HF 标签上标注的重复版本（duped versions）。
   - 目前正在重新训练以替换为正确标记的重复模型，并提供了新 HF 模型 [14m](https://huggingface.co/stellaathena/pythia-14m) 和 [31m](https://huggingface.co/stellaathena/pythia-31m) 的链接，上传预计在大约一小时内完成。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1475760290238038037)** (27 messages🔥): 

> `Baguettotron, ML papers published in Nature, ViT broken?, Differential Attention` 


- ****Baguettotron** 发布！**: **Baguettotron** 模型具有 **4608** 个特征，包含完整的 autointerp 标签，在 **774M** token 上训练，位于第 **48/80** 层，**8x** 扩展，top_k 为 **32**；其作者还分享了一篇 [X 帖子](https://x.com/Ji_Ha_Kim/status/2026166070172655786?s=20)以提供背景。
   - 作者提供了在线 [demo](https://lyramakesmusic.github.io/bread-slicer/) 的链接。
- **Nature 论文 = 看跌信号 (Bear Signal)？**: 一名成员戏称，在 *Nature* 上发表的机器学习论文是看跌信号，除非是 **DeepMind** 发表的。
   - 该评论是针对一项从技术层面解释连接*骨架稳定化（skeleton stabilization）*和*细节渲染（detail rendering）*的门控机制（gating mechanism）的详细说明而作出的。
- **Vision Transformers 坏掉了？**: 一名成员指出，将 **ViT** 简单地应用于 **CIFAR10** 是错误的，因为通过简单的线性层将 patch 转换为 *token* 是不够的，会导致表示（representation）并非最优。
- **Differential Attention 消融研究反馈**: 一名成员请求关于 Differential Attention 相关消融研究的反馈，并分享了一份 [PDF 文档](https://cdn.discordapp.com/attachments/747850033994662000/1475931314837262397/v2_draft.pdf?ex=699f47a6&is=699df626&hm=2c1090efdc639f38dfa72ea50d7871ae4f662b13d002ff4d9d2004355c0564b0&)。
   - 一位回复者批评称，消融实验并未证明 Differential Attention 是否在根本上更好，或者它是否只是从该方法中获得了不成比例的收益。


  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1475921848092069979)** (1 messages): 

> `LLM Behavior Debugging, LLM Reasoning Traces, LLM Refusals, LLM Agent Behavior` 


- **研究人员寻求 LLM 调试访谈**：研究人员正寻求与评估或调试 **LLM behavior** 的相关人员进行 **20–30 分钟的访谈**，并提供 **$25 亚马逊礼品卡**或慈善捐赠作为补偿。
   - 他们对有助于理解 **LLM** 为何产生特定输出的工作流和工具特别感兴趣，重点关注推理轨迹（reasoning traces）、拒绝响应（refusals）和 **Agent** 行为，并提供了 [预约链接](https://calendly.com/amerrick4-rrc/ai-auditing-problem-interview)。
- **LLM 评估重点领域详情**：该研究专门针对从事 **思维链 (chain-of-thought)** 检查、**可解释性 (interpretability) 或潜在知识 (latent-knowledge)** 研究、**调试 Agent 行为**以及**分析 LLM 中的拒绝响应或安全故障**的人员。
   - 目标是绘制评估和调试过程中的时间分配图，为当前实践提供有价值的见解。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1475673872937386144)** (2 messages): 

> `lm-evaluation-harness Bug Fix` 


- **微小 PR 解决棘手测试调整**：一名成员提交了一个单行代码的 **PR**，以修复 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/issues/3293) 中的一个 Bug。
   - 他们指出，这个修复的审查过程应该非常简单。
- **Athena 感谢协助**：一名成员对提交的 Bug 修复表示了感谢。
   - 这一致谢突显了该项目的协作性质。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1475546792488730734)** (2 messages): 

> `eval_adapter.py fix` 


- **前向传播 (Forward Pass) Bug 被解决**：一名成员分享了 `eval_adapter.py` 的 [修复版本](https://gist.github.com/aflah02/8e6b726bd08828b9a48b0cd354ad8431)，解决了前向传播调用中的问题。
   - 解决方案包括包装前向传播调用并调整元素，以匹配 `eval_adapter.py` 文件中的架构。
- **考虑将 Adapter 修复集成到仓库**：一名成员提议根据社区兴趣将该适配器修复集成到仓库中。
   - 旨在为 `eval_adapter.py` 的用户更广泛地解决前向传播问题。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1475551758574161950)** (8 messages🔥): 

> `LLM memory traces, IO testing libraries, Training small models on RTX cards, Graph DBs vs Vector DBs, Tiny GPU Compiler` 


- **寻求最先进的 (SOTA) 内存追踪方法**：一名成员询问了从 **LLM** 工作负载中生成内存追踪 (memory traces) 的 **SOTA 方法**。
   - 虽然在即时讨论中未分享具体方法，但问题本身表明了对优化 **LLM** 应用中内存使用的兴趣。
- ****Tiny GPU Compiler** 针对开源 GPU 硬件**：一名成员介绍了 **tiny-gpu-compiler**，这是一个基于 **MLIR** 的教育性编译器，针对开源 **GPU** 硬件，详见此 [GitHub 仓库](https://github.com/gautam1858/tiny-gpu-compiler)。
   - 该编译器将 **类 C (C-like) 的 GPU 核语言**转换为 **16 位二进制指令**，并包含一个用于逐步执行分析的交互式 Web 可视化工具，访问地址：[tiny-gpu-compiler](https://gautam1858.github.io/tiny-gpu-compiler/)。
- ****Graph DBs** 对抗 Vector DBs**：一名成员询问在基础层面，使用 **Graph DBs**（图数据库）而非 **Vector DBs**（向量数据库）对 **Agent** 有什么帮助。
   - 讨论未详细展开，但该问题暗示了在基于 **Agent** 的应用中探索替代数据库架构的可能。
- **如何使用 RTX 显卡训练小模型**：一名成员询问了使用 **RTX 2070-2080** 或 **3070-3080** 显卡训练**小模型（1.25 亿参数）**的情况。
   - 他们正在寻找有关**每秒处理 Token 数**的信息，以便与他们使用自定义内核的 **GTX 1080 Ti** 配置进行比较。


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1475570440792571914)** (3 messages): 

> `Triton, Gluon, TTGIR, TTIR` 


- **Gluon 基于 TTGIR，而非 Triton 的替代品**：一名成员询问 **Gluon** 是 **Triton** 的扩展还是替代品，另一名成员澄清说 **Gluon** 是一种建立在 **TTGIR** 而非 **TTIR** 之上的新语言。
- **Gluon：一种新语言**：讨论澄清了 Gluon 被设计为一种全新的语言，区别于仅仅作为 Triton 的扩展。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1475609362780131538)** (17 messages🔥): 

> `GPU Memory Optimization, CUDA memcpy_async and __syncthreads(), CPU to CUDA Verification Strategies` 


- ****Optimize**: 榨取 GPU 的更多性能**: 要优化 GPU 代码，应当衡量操作需要读取多少内存以及 GPU 读取该内存的速度，同时还要衡量该操作需要执行多少次运算以及 GPU 执行这些运算的速度。
   - 对于 **RMS norm**，性能很可能受限于内存（memory bound），因此应重点优化内存访问模式和带宽利用率，并参考 [Nvidia 关于异步拷贝的 CUDA 文档](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#batching-loads-in-conditional-code)。
- ****Sync Up**: 深入探讨 CUDA 的异步内存拷贝**: 在使用 `CUDA C++ cuda::memcpy_async` 时，必须使用 `__syncthreads()` 以确保内存可见性，从而保证所有线程都能看到任何线程拷贝的数据，这一点已参考 [CUDA 异步屏障（asynchronous barriers）文档](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html#tracking-asynchronous-memory-operations)得到明确。
- ****Port Authority**: 验证 CUDA 代码**: 在将 CPU 代码移植到 CUDA 生产环境时，验证的通用指导包括管理浮点精度，并为不同浮点大小的 **gemms** 操作设定标准容差，尤其是在 GPU 版本迭代期间。
   - 验证方法取决于具体上下文（如 **PyTorch** 或 **VLLM**），涉及考虑需要验证的适当代码单元（kernels）、测试的输入/输出数量，以及除简单精度之外更广泛的问题。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1475561331108090008)** (2 messages): 

> `FlashAttention 3, SDPA in PyTorch, Pre-built Wheels` 


- **Flash Attention 3 预构建 wheels 发布了！**: 适用于各种 CUDA 版本（**12.6+**、**13**）、CPU（**x86**、**ARM**）和操作系统（**Linux**、**Windows**）的预构建 **Flash Attention 3 wheels** 已在 [download.pytorch.org](https://download.pytorch.org/whl/flash-attn-3/) 上线。
   - 这些 wheels 符合 **LibTorch ABI 稳定版**，应可运行于任何 Python 版本 >= **3.10** 和 torch 版本 >= **2.9** 的环境。
- **探讨 Torch 的 SDPA Kernel 选择**: 有用户询问 Torch 的 SDPA (Scaled Dot-Product Attention) 如何选择正确的 kernel。
   - 答案是使用 `activate_flash_attention_impl("FA3")` 来重定向分发器（dispatcher）以改用 **FA3 kernels**。


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1475967596871549080)** (1 messages): 

> `eBPF, GPUs, profilers, OS Policies` 


- **eBPF 扩展至 GPUs**: Yusheng Zheng 将于 [PST 时间 12 月 12 日中午 12:00](https://arxiv.org/abs/2512.12615) 探讨如何通过扩展 **eBPF** 来增强 **GPU** 功能。
   - 本次演讲将涵盖近期工作，包括 *gpu_ext: Extensible OS Policies for GPUs via eBPF*，以及将 eBPF 扩展至 **GPU Device** 和 **Driver Contexts**。
- **性能分析库受到 GPU MODE 关注**: **GPU MODE** 社区对开发者构建更多**性能分析器（profilers）**和**性能分析可视化库**表现出了浓厚兴趣。
   - 鼓励感兴趣的个人观看相关的 [YouTube 视频](https://www.youtube.com/watch?v=8U7SzGnHoJU)，该视频可能会进一步辅助讨论。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1475714741715599370)** (2 messages): 

> `ParoQuant Project, Eigenvalues Comparison, Magnitude Difference` 


- **ParoQuant 项目链接发布！**: 一名成员分享了 [ParoQuant 项目](https://z-lab.ai/projects/paroquant/) 的链接。
   - 另一名成员表示，他们喜欢这种选取前 10 名的做法，就像选取前几个**特征值（eigen values）**一样。
- **幅值差异比特征值更简单！**: 一名成员推测，选择**最大幅值差异（largest magnitude difference）**是为了计算简便，相比之下**特征值**的计算较为复杂。
   - 他们指出，这与计算特征值相比非常简单。


  

---


### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1475762616667344906)** (1 messages): 

> `Nvidia, Linux drivers, Job Postings` 


- **Nvidia 招聘 Linux 驱动修复人员**: Nvidia 正在 [招聘员工](https://jobs.nvidia.com/careers/job/893393264012) 以增强其 Linux 驱动程序。
- **Nvidia 招聘岗位旨在改进 Linux**: 正如[最近的一则招聘启事](https://jobs.nvidia.com/careers/job/893393264012)所示，Nvidia 正寻求人才来优化其 Linux 驱动生态系统。


  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1475699397345149032)** (2 messages): 

> `NCCL, NVSHMEM performance, direct pointer access, matrix transpose` 


- ****NCCL** 与 **NVSHMEM** 的优势揭示**: 在观看了一场关于 **NCCL** 和 **NVSHMEM** 的演讲后，发现在矩阵转置场景中，通过 *nvshmem_ptr* 进行的直接指针访问比显式内存传输（例如 *nvshmem_get*）编程更简单且速度更快。
   - 性能差距正在研究中，探讨理想的基于 warp 的 *getmem* 变体是否能缩小与 *nvshmem_ptr* 的差距。
- ****NVSHMEM** 指针性能优势凸显**: **NVSHMEM** 测试的一个关键发现是，直接指针访问（*nvshmem_ptr*）优于显式内存传输（*nvshmem_get*），因为指针版本不使用临时缓冲区。
   - 一位专家建议，指针版本消除了缓冲，从而提升了性能并使代码更优雅。然而，*put 版本应该比 get 版本更好，但我还没时间去写。*


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1475588161127190569)** (1 messages): 

> `NCCL, SHMEM, RDMA, CUDA kernels, IRL Collaboration` 


- **波士顿新手寻求 NCCL/SHMEM/RDMA/CUDA 伙伴**: 一位波士顿的新手正投入时间研究 **NCCL**、**SHMEM**、**RDMA** 和 **CUDA kernels**，并欢迎进行线下交流和协作学习。
   - 该用户有兴趣合作开展一个小项目，以加深对这些技术的理解。
- **寻找 Kernel Kombat 的督促伙伴？**: 用户提出了寻找督促伙伴的想法，以确保有具体的产出，例如 *“在 48 小时内提交你最棒的 matmul kernel”*。
   - 他们指出，由于需要学习的概念非常广泛，分配专门的时间来 *“构建一些东西”* 可能会很有挑战性。


  

---


### **GPU MODE ▷ #[triton-viz](https://discord.com/channels/1189498204333543425/1225499141241573447/1475652834035761183)** (1 messages): 

> `N-Dimensional Tensor Visualizer, Einops-like Syntax, Colab Notebook` 


- **N 维张量可视化工具上线！**: 新的 **n 维可视化工具** 已发布，目前支持高达 **9D** 的张量。
   - 该可视化工具允许用户使用**类似 einops 的语法**，像处理 1D、2D 或 3D 张量一样轻松地对 N 维张量进行切片、转置和检查每个值。
- **Colab Notebook 助你开始可视化**: 创建了一个 [Colab notebook](https://colab.research.google.com/drive/1lrO6yzVQ8u_vFLPe7986goZtRQazmV0T#scrollTo=Q0TZi3zPxWhB) 来引导用户完成从 1D 到 9D 张量复制的可视化。
   - 一段视频展示了该工具检查形状为 **(2, 3, 4, 3, 4, 2, 4, 2, 3)** 的张量。


  

---


### **GPU MODE ▷ #[popcorn](https://discord.com/channels/1189498204333543425/1298372518293274644/1475540096194187275)** (21 messages🔥): 

> `KernelBench Environment, CUDA Memory Errors, Modal.experimental.stop_fetching_inputs, KernelBot Environment, Eval.py reuse` 


- ****KernelBench 环境** 应对 CUDA 错误**: 一名成员为 **KernelBench** 和 **kernelbook** 创建了自定义环境，以解决损坏的 **CUDA 内存错误**，并打算分享它。
- **Modal 修复 **CUDA 错误**！**: 错误 *“cuda memory error is detected”* 可以通过使用 `modal.experimental.stop_fetching_inputs` 来解决。
   - 此修复已在成员的 `backendbench env` 中实现。
- **KernelBot 已部署并解决问题！**: **KernelBot** 的初始环境现已上线，可在 [app.primeintellect.ai](https://app.primeintellect.ai/dashboard/environments/roeybc/kernelbot-env) 获取，目前支持 `trimul` 和 `amd` 问题，并对 Nvidia 使用 Modal，对 AMD 问题使用 Runpod。
   - 团队正在处理 `PMPP` 问题和 AMD 分布式 kernel 问题。
- **重用 eval.py 以防止 Bug**: 鼓励成员重用来自 `eval.py` 的评估逻辑，以减少不同比赛中逻辑不一致导致的问题。
   - 他们讨论了由于容差和迭代次数不同等限制，可能需要为每个问题调整 `eval.py` 的内部功能。
- **评估与分析上线 Northflank**: 频道议程包括将 **KernelBot** 迁移到 Northflank、网站 UI 改进、针对作弊的 AI 评估，以及关于端到端推理速通（speedruns）的讨论。
   - 对 hacky 提交的解析已完成，指纹识别和更深入的分析正在进行中。


  

---

### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1475935136578011281)** (2 messages): 

> `Heroku 到 Northflank 迁移，Bot/Web 停机，CLI 更新` 


- **Heroku 到 Northflank 迁移导致停机**：服务正处于从 **Heroku** 迁移到 **Northflank** 的过程中，这将导致 Bot 和 Web 服务停机。
   - 请用户耐心等待停机结束。
- **迁移后服务已恢复**：服务已重新上线，请用户反馈任何问题，尤其是关于 **auth**（认证）方面的问题。
   - 用户必须将 **CLI** 更新至最新版本，以确保一切运行正常。


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1475970084634624010)** (1 messages): 

> `B200, 租赁, Neocloud` 


- **B200 非常昂贵！**：一位成员指出 **B200** 极其昂贵，并建议除非是企业用户，否则应从 **neocloud** 租赁/租用。
   - 该成员分享了其公司的解决方案链接：[lightning.ai/clusters](https://lightning.ai/clusters)。
- **云端租赁具有成本效益！**：除非是企业，否则从 neocloud 租赁/租用非常有意义。
   - 如果感兴趣，请查看：[lightning.ai/clusters](https://lightning.ai/clusters)。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/)** (1 messages): 

epiicepiic: 明白了。感谢说明！
  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1476006971227377815)** (1 messages): 

> `RRCLLX, AMD MI300X, Meta, GPU 通信` 


- **Meta 在 AMD 平台上利用 RRCLLX 创新 GPU 通信**：Meta 正在 AMD 平台上利用 **RRCLLX** 创新 GPU 通信，详情见其 [工程博客文章](https://engineering.fb.com/2026/02/24/data-center-engineering/rrcclx-innovating-gpu-communications-amd-platforms-meta/)。
- **AMD MI300X 获得 RRCLLX 支持**：Meta 正使用 **RRCLLX** 更高效地连接 **AMD MI300X** GPU。


  

---


### **GPU MODE ▷ #[low-bit](https://discord.com/channels/1189498204333543425/1411659097706860647/1475627773325213736)** (2 messages): 

> `BitNet 1.58b, Mamba2, 4Bit-Forge` 


- **BitNet 1.58b 与 Mamba2 结合**：分享了一个来自 [Zenodo](https://zenodo.org/records/18394665) 的 **BitNet 1.58b + Mamba2** 链接。
- **4Bit-Forge 重构进行中**：一位成员提到他们目前正在用 **CUDA** 重构 **4Bit-Forge**。
   - 他们预计很快会发布更新。


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1475929250640171162)** (2 messages): 

> `Helion v0.3.0 发布，Autotuning 改进，Triton-to-TileIR 桥接，Pallas TPU 支持，CuteDSL 代码生成` 


- **Helion v0.3.0 亮相！**：新的 [Helion 0.3.0 版本](https://github.com/pytorch/helion/releases/tag/v0.3.0) 包含了 **autotuning** 的改进、**Triton-to-TileIR 桥接**的支持，以及针对 **Pallas TPU** 和 **CuteDSL codegen** 支持的大规模重构。
   - 值得注意的是，[这篇博客文章](https://pytorch.org/blog/accelerating-autotuning-in-helion/) 详细介绍了 autotuning 的改进。
- **Helion 的并行读取修复？**：围绕 [这个 PR](https://github.com/pytorch/helion/pull/1418) 的讨论集中在它是否解决了并行读取变为原子操作（atomics）的问题。
   - 该问题在 [JAX 文档](https://docs.jax.dev/en/latest/pallas/design/design.html#grad-of-pallas-call) 中有所概述。


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1475547563259068614)** (4 messages): 

> `Guaguabear 误读，Kernel 编程环境配置，顶级 AI 生成方案` 


- **"Guaguabear" 误读引发关注**：一位成员幽默地提到，在看到用户名 *"guaguabear"* 时愣了一下，最初将其误读成了 *"bear bear bear"*。
- **用户寻求 Kernel 编程环境配置建议**：一位成员询问首选的 Kernel 编程环境配置，并指出虽然 [Modal](https://modal.com/) 很有帮助，但在竞赛之外缺乏 **NCU profiling 支持**。
   - 他们正在寻找一种可定制的环境来进行 Kernel 编写和优化，暗示现有的解决方案可能限制过多。
- **关于顶级 AI 生成方案的讨论**：针对 *"哪一个是顶级的 AI 生成方案？"* 的问题，一位成员推荐了 **Ouye Xie 用于 CUDA** 以及 **billcarson cutedsl**。


  

---

### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1475693527001206795)** (10 messages🔥): 

> `SO-101 的 VR Teleop、CamBot 机器人设计、面向 3D 打印的 Open Arms 重新设计、3D 打印耗材选择` 


- **自定义 VR Teleop 代码使 SO-101 达到理想状态**：一名成员为 **SO-101** 机械臂编写了自定义的 **VR teleoperation**（远程操作）代码，发现 **SO-107** 的额外关节对于匹配 VR 中手部触及的 **XYZ space** 非常有用。
   - 他们发现 **SO-107 的额外关节** 非常值得，因为额外的自由度（degree of freedom）能更好地匹配他在 VR 中手部所能触及的 xyz 空间。
- **CamBot 首次亮相并搭载 FeeTech Motors**：一名成员通过 **Web-Sockets** 实现了完整的自定义远程操作，并设计了一款全新的 **6 DoF CamBot Robot**，该机器人使用了与 **SO-101** 相同的 **FeeTech Motors**。
   - 该设计目前正在测试中，详情可见链接中的[图片](https://cdn.discordapp.com/attachments/1437390897552818186/1475702902470217748/grafik.png?ex=699f1bad&is=699dca2d&hm=e23ca6354e29cb229cfd1aa620ed5ab2c4a742eae6e22214522dc342fa9357eb&)。
- **Open Arms 平台拥抱 3D 打印革命**：一名成员目前正在重新设计 **Open Arms** 平台以支持 **3D 打印**，旨在将平台成本降低到 **$2.5k** 以下。
   - 他们使用电动升降桌腿作为廉价的 **3 stage linear actuators**（三级线性致动器）来源，详情可见链接中的[图片](https://cdn.discordapp.com/attachments/1437390897552818186/1475871315205554287/IMG_0547.jpg?ex=699f0fc5&is=699dbe45&hm=58b06bae5d2b9d1ebb4b4b820301bc5948185c18697907a8f5a8539282dbe837&)。
- **耗材大比拼：PLA vs PLA-CF vs PETG vs PA6**：一名成员正在考虑在实验中跳出 **PLA**，尝试使用 **PLA-CF**、**PETG** 或 **PA6 (Nylon-Fiber)** 等耗材。
   - 这可能需要一台带有外壳和空气过滤器的全新打印机；他们目前正在使用 **Bambu Labs A1** 进行打印，总体对该机器非常满意。


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1475778260477546660)** (3 messages): 

> `HPC 就业能力、GPU 知识现代化、CUDA Kernels` 


- **HPC 资深人才有需求吗？**：一位在 **HPC** 领域拥有 **15 年以上** 经验的研究人员正在寻求关于学术界以外就业能力的看法，特别是在当前 **AI** 为主的就业市场中。
   - 他们想知道自己在 **OpenMP**、**MPI**、**CUDA**、多 GPU 系统以及 **Chapel** 和 **Julia** 等并行语言方面的专业知识是否对雇主有吸引力。
- **规划 GPU 知识现代化路线**：该研究人员正在寻求关于更新其 **GPU** 知识的建议，考虑的领域包括 **Triton**、**CUDA Graphs**、编译器栈和 **ML** 系统内部机制。
   - 鉴于其深厚的 HPC 背景，他们正在寻求如何能在当前市场中最好地定位自己的指导。
- **用于推理的 CUDA Kernels**：一名 **tpu-inference**（Google 的 vllm TPU 后端）的贡献者询问，对于推理和 MLsys 岗位来说，精通 **CUDA kernel** 知识是否必不可少。
   - 尽管对 tpu-inference 做出了重大贡献（[见 readme 记录](https://github.com/catswe)），但他们只有教程级别的 kernel 经验，不确定其非 kernel 相关的工作经验能带来多大帮助。


  

---


### **GPU MODE ▷ #[flashinfer](https://discord.com/channels/1189498204333543425/1464407141128339571/1475869597662969856)** (4 messages): 

> `竞赛中的加速指标、CUDA C++ 编译标志、方案产物` 


- **渴望竞赛的加速统计数据**：一名成员询问在当前的 benchmark 设置中，是否可以提供类似于其他竞赛中看到的加速（speedup）指标，如[附图](https://cdn.discordapp.com/attachments/1464407141128339571/1475869597247995944/image0.jpg?ex=699f0e2c&is=699dbcac&hm=bc0c5e273addb955b82fd8b4ffa0ed6af456b86248e987365772d36e4d77413e)所示。
- **CUDA 编译标志配置难题**：一名成员质疑为何在 torch builder 和 TVM FFI builder 中没有为 **CUDA C++** 提交传递额外编译标志的选项，并引用了相关[代码](https://github.com/flashinfer-ai/flashinfer-bench/blob/c1fd980f70263c83ab47a43325cf87f2dba9b61a/flashinfer_bench/compile/builders/torch_builder.py#L154-L162)。
   - 引用了相关的 [TVM FFI 代码](https://github.com/flashinfer-ai/flashinfer-bench/blob/c1fd980f70263c83ab47a43325cf87f2dba9b61a/flashinfer_bench/compile/builders/tvm_ffi_builder.py#L264-L270)。
- **解决方案产物增加编译标志**：一名成员建议将编译标志纳入方案产物（solution artifact）中，强调这对于可复现性和评估的一致性至关重要。


  

---

### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1475549164615106661)** (41 messages🔥): 

> `解封请求, Anthropic 指控, Prompt 循环期间的工具变更, Kimi K2.5, Kimi K3` 


- **用户恳求解封**：一名用户请求管理员（被标记的用户）将其解封，希望能进行 5 分钟的对话，并标记了另一名用户来转达该消息。
   - 该用户随后发布了一篇 [WSJ 文章](https://www.wsj.com/tech/ai/anthropic-accuses-chinese-companies-of-siphoning-data-from-claude-63a13afc) 的链接，内容关于 **Anthropic 指控中国公司从 Claude 窃取数据**，并称其行为“可悲”。
- **版主重定向无关主题的请求**：一名用户提出的请求由社区版主处理，版主表示该服务器*不是讨论此话题的地方*，并建议使用其他服务器。
- **询问 Prompt 循环期间的工具变更**：一名用户询问在 **Prompt 到响应的循环（prompt-to-response cycle）期间**更改**可用工具**的可能性。
- **用户感叹缺乏浏览器扩展**：一名用户表示 Kimi K2.5 唯一缺少的就是**浏览器扩展**。
- **督促提交 Kimi 错误报告**：一名用户报告了一个已持续 **10 天** 的错误，并附带了 [图片](https://cdn.discordapp.com/attachments/1371757564005711973/1475932351497240717/image.png?ex=699f489e&is=699df71e&hm=2b588317c8756fd95479fe5ddb11eee39b51d5f888ebb10ba0629823a8b746d9&)。
   - 一名版主要求该用户提交包含所有相关详细信息的 **Bug Report**。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1475567355491913920)** (16 messages🔥): 

> `Lucidrains Github, 句子相关性, Attention 模型 Scout, Dell Pro Max GB10, NVIDIA 显存占用` 


- ****Lucidrains** GitHub 消失了？**：一名成员询问 **lucidrains** 的 GitHub 仓库去向以及可能被移除的原因。
- ****Scout** 模型学习定向句子相关性**：一名成员介绍了 **Scout**，这是一种修改了标准 Transformer 架构的新型 Attention 模型，旨在学习句子之间而非 Token 之间的定向相关性，核心问题是*“句子 B 是否真的对句子 A 有帮助？”*。该项目已在 [GitHub](https://github.com/samyak112/Scout) 上开源。
- ****GB10** 内存/显存吃力**：一名成员分享了使用 **Dell Pro Max GB10** 的经验，指出它虽然能运行但速度很慢，共享的 GPU/CPU 内存导致频繁出现 **GPU OOM**，从而引起系统冻结和重启。
   - 他们建议使用 `nvitop` 来跟踪显存使用情况和 GPU 统计数据，因为据称 `nvidia-smi` 的输出已损坏。
- ****Foundation Models** 回归**：一名成员分享了一篇关于 [Foundation Models 的 SI 文章](https://si.inc/posts/fdm1/) 的链接，暗示其正在复兴。
   - 另一名成员分享了一个讨论 AI 在教育中应用的 [YouTube 视频](https://youtu.be/IeeFOpS-S_M?si=eBJeM3UeI_E1aHjD)。
- ****GANfather** 回归！**：**GAN** 的创造者 **Ian Goodfellow** 回归了（[推文](https://fxtwitter.com/goodfellow_ian/status/2026024150213738520)），引发了人们对 **GAN** 在解决验证问题上迎来复兴的希望。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1475545382569119865)** (11 messages🔥): 

> `Wave Field LLM, Inception AI 的 Mercury II, 连续场 Token` 


- **Wave Field LLM 遭到质疑**：一名成员分享了 [Wave Field LLM 的 GitHub 仓库](https://github.com/badaramoni/wave-field-llm) 并询问是否值得关注。
   - 另一名成员回复称，**Baseline（基准）看起来很弱**，且*未看到消融实验（ablations）*，因此持怀疑态度。
- **连续场 Token 已被探索**：一名成员指出，**Wave Field LLM** 中描述的 Token 存在于连续场中的概念，已在 [这篇论文](https://arxiv.org/abs/2406.11838) 中被探讨过。
- **Inception AI 的 Mercury II 亮相**：一名成员提到了由 **Inception AI** 开发的 **Mercury II**，并分享了 [Inception AI 官网](https://www.inceptionlabs.ai/) 的链接和 **arXiv 论文** 链接。


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1475927172999942226)** (1 messages): 

> `` 


- **System Prompt 咨询**：一名用户询问了用于生成文章的 System Prompt。
   - 该用户表示，如果没有 System Prompt，很难判断内容在现实层面的可靠性（realistic grounding）。
- **请求包含 Prompt**：用户请求将 System Prompt 包含在内以提供上下文。
   - 这将有助于更好地评估 AI 的约束条件和指令。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1475890838851813531)** (6 messages): 

> `Liquid AI's LFM2-24B-A2B, WarClaude, Alibaba Qwen release` 


- **Liquid AI 发布 LFM2-24B-A2B**：Liquid AI 宣布发布 **LFM2-24B-A2B**，并在其 [博客文章](https://www.liquid.ai/blog/lfm2-24b-a2b) 中展示了他们的最新进展。
   - 该模型旨在为高效且有效的 AI 解决方案设定新标准。
- **阿里巴巴 Qwen 发布更新**：阿里巴巴推出了 **Qwen** 的更新，详情见 [X 上的帖子](https://fxtwitter.com/Alibaba_Qwen/status/2026339351530188939?s=20)，增强了其功能和易用性。
   - 鼓励用户查看公告以了解改进和新功能的更多细节。
- **X 平台上的 WarClaude 动态**：一位成员对 **WarClaude** 表示关注，并链接了 X 上的相关内容，参见 [帖子 1](https://x.com/i/status/2026369451403390999) 和 [帖子 2](https://x.com/i/status/2026369453655732693)。
   - 目前没有提供关于 *WarClaude* 实际是什么的其他讨论或背景信息。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1475583828402897106)** (19 messages🔥): 

> `Vulnerability reporting, Unlimited chat tier, Account transfer, Telegram credit usage, Desktop app billing` 


- **漏洞报告咨询**：一名用户报告发现了一个漏洞，并被引导至 [反馈页面](https://manus.im/feedback?source=help_center)。
   - 用户对如何以及在哪里报告漏洞表示困惑。
- **考虑推出无限量层级**：一名用户询问是否会有类似 **ChatGPT** 或 **Grok** 的无限聊天层级，原因是 **Manus Agent** 在 Telegram 中的额度消耗过快。
   - 官方代表回应称，他们非常感谢这些反馈，并正在不断努力改进产品。
- **账户转移困扰**：一名用户请求将其项目转移到另一个账户，并提供了相关的电子邮件地址。
   - 支持团队确认目前不支持直接账户转移，建议用户本地下载内容并在新账户上启动新任务。
- **Telegram Agent 的额度消耗**：一名用户提到 Telegram Agent *非常好用*，但由于高额度使用，*从我的账户中扣除了太多积分*。
   - 这进一步强化了之前关于通过订阅选项来缓解额度担忧的问题。
- **AI/ML 工程师为扩展严肃 AI 产品提供专业知识**：一名 AI/ML 工程师表示可以为构建可扩展的严肃 AI 产品提供专业知识，强调了推理成本、内存设计和系统在负载下的行为的重要性。
   - 他在过去几年中一直致力于 AI 系统开发，在这些系统中，技术决策直接影响到产品的生存。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/)** (1 messages): 

darinsimmons: 欢迎 Zayden，本 Discord 频道用于讨论 Modular、Mojo 和 MAX。
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1475612841544253471)** (7 messages): 

> `String templating in Mojo, Writable and Writer traits, ExternalFunction struct` 


- **Mojo 字符串模板引擎提案出现**：一名成员为 Mojo 的新 **字符串模板功能 (string templating feature)** 发起了提案，[论坛上的讨论帖在此](https://forum.modular.com/t/writable-writer-template-engines/2763)。
   - 该功能可能会在 **1.0 版本之后** 推出，希望将当前的 `Writable`/`Writer` trait 扩展为更复杂的 `TemplatedWritable`。
- **`Writable` 和 `Writer` trait 需要优化**：当前的 `Writable` 和 `Writer` trait 应该是极简的，通过其他 trait 或通过默认的 trait 方法/类型提供扩展/自定义点。
   - 路线图将优先处理 **Int 统一 (Int unification)** 等其他功能，然后再处理该提案，目标是将 `write_to` 和 `write_repr_to` 的实现统一为单个函数。
- **ExternalFunction 结构体技巧**：一名成员提到他们一直以 `ExternalFunction` 结构体为灵感，并正在寻找一个更高级的版本，以便将函数签名分解为其参数/返回类型。
   - 他们可能需要为 **所有外部指针编写原始转换 (origin casts)**。


  

---

### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1475576895914639472)** (8 messages🔥): 

> `GitHub CI 失败, 文档中的断开链接, Linux Foundation 峰会, Ezra Klein 谈论 Agents` 


- **调查 GitHub CI 失败修复情况**: 一名成员报告说 `npm run generate`、`npm run format` 和 `npm run check` 在 [PR 2278](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2278) 上本地运行均通过，但在 CI 中出现了失败。
   - 根本原因是缺失一个文件，导致 `docs/community/seps/index.mdx` 中出现断开的链接。
- **Linux Foundation 峰会聚会**: 一名成员邀请其他参加在加州纳帕举行的 [LF Member Summit](https://events.linuxfoundation.org/lf-member-summit/) 的人见面聊聊 MCP。
   - 消息中没有提供关于场地或时间安排的进一步细节。
- **Ezra Klein 了解 Agents**: 一名成员分享了一段 Ezra Klein 了解 Agent 的 [YouTube 视频](https://youtu.be/lIJelwO8yHQ)。
   - 该视频被直接分享，没有额外的评论。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1475908010428727329)** (4 messages): 

> `Aider 未来更新, Aider 中的 Git submodules 支持, 适用于 Aider 的低成本 LLM` 


- **Aider 未来具有不确定性**: 一位用户询问 **Aider** 是否仍在积极开发，以及是否有推荐的替代方案。
   - 另一名成员提到，目前有其他比 Aider 更*先进*的 CLI。
- **Aider 缺乏 Git submodules 支持**: 一位计算机科学家指出 **Aider** 不支持 **git submodules**，并提出了改进建议，详见 [此 GitHub issue](https://github.com/Aider-AI/aider/issues/3603)。
   - 他们正在征求关于该提议功能的反馈和建议。
- **为 Aider 寻找低成本 LLM**: 一名成员正在寻求关于寻找低成本 **LLM** 与 **Aider** 配合使用的建议，因为 **Gemini** 很快就耗尽了他们的 token。
   - 他们正在寻找能在 **Aider** 框架内平衡成本效益与可用性的 **LLM**。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1475804606142353448)** (1 messages): 

> `Aider 在复杂任务中的局限性, 为重复性任务编写 Aider 脚本, 将 Aider 与外部脚本或 Agent 结合使用, 查找函数用法` 


- **用户指出 Aider 的模糊文件查找功能失效**: 一位用户分享说，他们看重 **Aider** 作为一个 AI 工具在跨多个文件进行模糊搜索和替换等任务中的价值，但在处理更复杂的场景时面临局限性。
   - 该用户在一次处理过多文件时遇到了 **diff 格式化问题**，迫使他们必须分小块进行工作。
- **通过脚本 Hack Aider 以实现任务自动化**: 该用户正在寻求如何使用外部脚本来自动化 **Aider** 的重复性任务的指导，例如循环遍历文件进行编辑。
   - 他们在询问是否已有工具可以简化这种与 **Aider** 的交互。
- **Agent：Aider 的下一个前沿？**: 该用户想知道他们所需的功能是否与 **AI Agent** 的概念相符，并愿意探索 **Aider** 的 fork 版本或像 **opendesk**、**cline** 之类的工具。
   - 他们希望改进工作流，以寻找函数用法，确保其符合特定标准，然后通过添加行和传递新参数来编辑文件，而无需在 VSCode 中进行手动干预。
- **用户需要帮助查找所有函数用法**: 用户想要自动化查找函数所有用法的过程。
   - 他们希望检查用法是否符合某些标准，然后在文件中进行简单的编辑，添加两行并向其传递一个新参数。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1475552347668348938)** (4 messages): 

> `tiny-gpu, AMD Ryzen AI, MLIR 编译器` 


- **Tiny-GPU 编译器发布！**: 一名成员介绍了 [tiny-gpu-compiler](https://github.com/gautam1858/tiny-gpu-compiler)，这是一个针对开源 GPU 硬件的教育性 **基于 MLIR 的编译器**，配有一个交互式 Web 可视化工具。
- **Tiny-GPU 编译器进入二进制阶段**: [tiny-gpu-compiler](https://github.com/gautam1858/tiny-gpu-compiler) 将一种 **类 C 的 GPU 内核语言** 编译成针对 tiny-gpu（一个用 Verilog 编写的开源 GPU）的 **16 位二进制指令**。
- **AMD Ryzen AI 亮相**: 在 CES 2026 之后，[AMD.com](https://www.amd.com/en/products/embedded/ryzen-ai/p100-series.html) 发布了全新的 **AMD Ryzen AI**。