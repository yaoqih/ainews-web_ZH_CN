---
companies:
- perplexity
- openai
- anthropic
- langchain-ai
date: '2026-02-25T05:44:39.731046Z'
description: '**Perplexity** 推出了 **Computer**，这是一个以编排为核心的智能体（Agent）平台，具备多模型路由、按使用量计费以及用于分布式工作流的并行异步子智能体功能。**Andrej
  Karpathy** 声称，自 12 月以来编程智能体领域发生了“相变”（phase change），并强调了其持续完成长跨度任务的能力。**OpenAI**
  发布了 **GPT-5.3-Codex**，其速度提升了约 25%，且基准测试表现强劲；与此同时，**Claude Code** 迎来了上线一周年，正致力于生态系统集成并应对扩展挑战。这标志着编程工作流和基于智能体的软件开发发生了重大转变。'
id: MjAyNi0w
models:
- gpt-5.3-codex
- claude-code
people:
- karpathy
- aravsrinivas
- lioronai
- denisyarats
- swyx
- catwu
- hwchase17
title: '**智能体工程：2025年12月到底发生了什么？**'
topics:
- coding-agents
- agent-architecture
- distributed-workflows
- usage-based-pricing
- model-routing
- benchmarking
- context-length
- observability
- software-development
---

**人们日益感到一种不安：编程已经永远改变了——这远不止是“普通”的炒作。**

> AI News 2026/2/24-2026/2/25。我们为你检查了 12 个 subreddits、[544 个 Twitters](https://twitter.com/i/lists/1585430245762441216) 和 24 个 Discords（**262** 个频道，**10751** 条消息）。预计节省阅读时间（按 200wpm 计算）：**1086** 分钟。[AINews 网站](https://news.smol.ai/) 允许你搜索所有往期内容。提醒一下，[AINews 现在是 Latent Space 的一个板块](https://www.latent.space/p/2026)。你可以[选择订阅/退订](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)邮件发送频率！


我们为此制作了一个微型网站：

# https://wtfhappened2025.com/

https://wtfhappened2025.com/

现在就去看看。


---

# AI Twitter 摘要


**Perplexity “Computer”：一个编排优先的 Agent 产品（多模型、工具+环境、按使用量付费）**

- **Perplexity Computer 发布**：Perplexity 推出了 **Computer**，定位为一个端到端系统，通过在一个界面中编排**文件、工具、内存和模型**，来实现项目的“研究、设计、编码、部署和管理” ([发布推文](https://x.com/perplexity_ai/status/2026695550771540489), [Arav Srinivas](https://x.com/AravSrinivas/status/2026695864039911684))。关键产品信号：
  - **访问权限 + 定价**：首先对 Web 端的 **Max** 订阅者开放，随后是 Pro/Enterprise 用户；采用**按使用量付费**模式，支持**子 Agent 模型选择**、支出上限，Max 用户包含每月 10k 额度，此外还有限时赠送的奖励额度 ([定价详情](https://x.com/perplexity_ai/status/2026695793537855526), [可用性](https://x.com/perplexity_ai/status/2026695805252547008), [Arav 关于推向市场的说明](https://x.com/AravSrinivas/status/2026697136507859067))。
  - **架构重点**：多条推文强调，其“突破”在于**并行、异步的子 Agent**，由一个协调者模型将任务分配给专家模型（研究 vs 编码 vs 媒体），而不是单一的单体 Agent 循环 ([Lior 的分析](https://x.com/LiorOnAI/status/2026739011122065819), [Denis Yarats](https://x.com/denisyarats/status/2026704583817634180))。
  - **“一切皆计算机”叙事**：Perplexity 员工宣传 Computer 是一个由小团队构建的平台，大量使用了 coding agents 以及自动化的 eval/debug 循环 ([Arav](https://x.com/AravSrinivas/status/2026703703248613736), [Denis](https://x.com/denisyarats/status/2026704583817634180))。  
- **为什么对工程师很重要**：Computer 是向*系统级 Agent UX* 迈出的具体一步：多模型路由、隔离/沙箱、持久化内存以及成本控制——即，将“Agent 化工作”视为一个**分布式工作流**，而不是单一的聊天会话 ([Arav](https://x.com/AravSrinivas/status/2026695864039911684), [Computer 网站](https://x.com/AravSrinivas/status/2026697232846827941))。

**Coding agents：“从 12 月开始生效” + 新模型/工具发布（GPT‑5.3‑Codex, Claude Code 生态系统, Copilot CLI GA）**

- **Karpathy 的“相位变化”论断**：Andrej Karpathy 认为**编程 Agent 自 12 月以来跨越了一个定性的门槛**——从脆弱的 Demo 演变为具有连贯性和韧性、能够完成持续且长周期任务的工具。他举了一个详细的例子：在极少的人为干预下，委托 Agent 完成了一个端到端的本地部署（SSH 密钥 → vLLM → 模型下载/评测 → 服务器端点 → UI → systemd → 报告）([Karpathy](https://x.com/karpathy/status/2026731645169185220))。这与来自开发工具构建者和用户的广泛“软件正在改变”的情绪相契合 ([Cursor](https://x.com/cursor_ai/status/2026717494426173917), [snowmaker](https://x.com/snowmaker/status/2026555857845256354))。
- **OpenAI GPT‑5.3‑Codex 发布 + 早期评估讨论**：
  - OpenAI 在 API 中上线了 **GPT‑5.3‑Codex** ([snsf](https://x.com/snsf/status/2026513135075746239))，Cline 宣布支持并声称获得提升：**比 5.2 快约 25%**，每个任务消耗更少的 token，并在 SWE-Bench Pro 上表现强劲 ([Cline](https://x.com/cline/status/2026481089158779021))。
  - 社区对基准测试的反应剧烈（且嘈杂）：例如，“IBench 达到 86%”的惊喜 ([tweet](https://x.com/adonis_singh/status/2026456939224510848)) 以及“首批基准测试即将到来” ([kimmonismus](https://x.com/kimmonismus/status/2026709699366670579))。在测试方法论明确之前，应将这些视为趋势性参考。
- **Claude Code：产品成熟度 + 可观测性 + 集成**：
  - Claude Code 的“一周年”框架和回顾强调了它作为一个*基础性*编程 Agent 产品的地位，此外还有关于**上下文长度缩放触及内存限制**的担忧 ([swyx](https://x.com/swyx/status/2026462001933988094))。
  - 实用生态系统进展：Claude Code 的 **Slack 插件**集成 ([catwu](https://x.com/_catwu/status/2026485966626763120))；使用 LangSmith 追踪 Claude Code 以调试“性能削减（nerfing）”/路由问题 ([hwchase17](https://x.com/hwchase17/status/2026452439327764521), [可观测性投诉](https://x.com/ChaiWithJai/status/2026446654753190324))。
- **GitHub Copilot CLI 正式发布 (GA) + “/research”**：
  - Copilot CLI 已达到 **GA** 阶段 ([Evan Boyle](https://x.com/_Evan_Boyle/status/2026706464375796099))，并添加了 `/research` 功能，用于利用 GitHub 代码搜索 + 基于 MCP 的动态获取进行全仓库深度调研，并将报告导出到 gists 以供分享 ([feature](https://x.com/_Evan_Boyle/status/2026458533320077689))。
  - 较小的 UX 更新：终端中的 Copilot CLI 现在支持实时更新标题 ([tweet](https://x.com/njukidreborn/status/2026443296177008818))。

**开源模型与本地推理：Qwen3.5 “Medium” 浪潮（MoE + 长上下文 + FP8/量化），以及本地 Agent 的转折点**

- **Qwen3.5 Medium 系列分发攻势**：阿里巴巴在 **vLLM**、**GGUF**、**LM Studio**、**Ollama** 和 **Jan** 上同步推出了首日工具支持，凸显了目前重大开源模型发布的部署栈响应速度之快 ([vLLM 致谢](https://x.com/Alibaba_Qwen/status/2026496673179181292), [GGUF](https://x.com/Alibaba_Qwen/status/2026497723944546395), [LM Studio](https://x.com/Alibaba_Qwen/status/2026496880285462962), [Ollama](https://x.com/ollama/status/2026598944177009147), [Jan](https://x.com/Alibaba_Qwen/status/2026660582221558190))。
- **来自 Qwen 的关键技术断言**（如发布所述，此处未经独立验证）：
  - **量化鲁棒性**：在 **4-bit 权重 + KV-cache 量化**下达到“近乎无损”的精度。
  - **长上下文**：**Qwen3.5‑27B 支持 800K+**，**35B‑A3B 在 32GB VRAM 消费级 GPU 上支持 >1M 上下文**，**122B‑A10B 在 80GB GPU 上支持 1M+**。
  - **开源基座**：Qwen 开源了 **Qwen3.5‑35B‑A3B‑Base** 以支持研究 ([Alibaba_Qwen](https://x.com/Alibaba_Qwen/status/2026502059479179602))。
  - **FP8 权重开源**，并提供原生 vLLM/SGLang 支持 ([FP8 公告](https://x.com/Alibaba_Qwen/status/2026682179305275758))。
- **本地 Agent 的“前后对比”**：一位知名从业者声称，**Qwen3.5‑35B‑A3B** 使得本地 Agent 循环在可靠性（工具调用、稳定性）上有了显著提升，而每 token 仅激活 **~3B 参数**——这明确将本地模型定位为在许多工作流中可与 Claude Code/Codex 并行的可行选择 ([victormustar](https://x.com/victormustar/status/2026624792602808707))。
- **评测讨论警告：刷榜行为（Benchmaxxing）与 MoE vs 稠密模型的混淆**：
  - 多个讨论帖警告不要过度解读排行榜（“请停止迷信刷榜”）([scaling01](https://x.com/scaling01/status/2026698844088549848))，并指出不同规模的 Qwen 在某些基准测试上表现出的惊人一致性，这暗示了要么是工具链效应，要么是基准测试伪影 ([eliebakouch](https://x.com/eliebakouch/status/2026727151978840105), [teortaxesTex 关于 HLE/MoE 解读](https://x.com/teortaxesTex/status/2026690994029072512))。
  - Arena 已将 Qwen3.5 Medium 添加到文本/视觉/代码 Arena 中进行正面交锋 ([Arena](https://x.com/arena/status/2026716550812807181))。

**Agent、可靠性以及“为 Agent 而构建”：最小化基准测试、工具接口优化和失败模式**

- **可靠性的提升不如能力的提升**：一个专注于可靠性的研究方向认为，尽管模型进展迅速，但**可靠性的提升却很微小**，他们将可靠性分解为多个维度，并警告不要将 Agent 的性能简化为单一的“成功率”数字 ([IEthics](https://x.com/IEthics/status/2026435186704134617), [Justin Bullock 引用](https://x.com/JustinBullock14/status/2026693253169336475))。
- **Agent 的失败通常是*可靠性*问题，而非能力问题**：一篇关于“Agent 失败”论文的摘要声称，Agent 经常因为**连环的小型偏离路径的工具调用**而失败，即一个错误会增加下一个错误发生的可能性，尤其是在长程（long-horizon）任务场景中 ([omarsar0](https://x.com/omarsar0/status/2026471955319189861))。
- **极简“安全且有助”基准测试理念**：一项提案建议不再测量更难的任务，而是测量模型是否能可靠地执行*明确定义的简单*安全行为（例如“仅在被要求时发送电子邮件”），包括在无关/干扰上下文的情况下；该观点声称前沿模型在这些情况下仍会出错 ([jonasgeiping](https://x.com/jonasgeiping/status/2026714911951220888))。
- **工具描述作为优化目标 (Trace‑Free+)**：Intuit AI Research 的工作表明，**Agent 的成功在很大程度上取决于工具接口文本**，并引入了一套课程，教模型将工具描述重写为 Agent 可用的形式，且在推理时不需要 Trace；该研究报告了在 StableToolBench/RestBench 上的提升，以及在超过 100 个工具环境下的稳健性 ([omarsar0](https://x.com/omarsar0/status/2026676835539628465))。
- **GUI/Web Agent：规划型 vs 反应型**：ActionEngine 将 GUI Agent 重新定义为**图遍历**，通过离线探索生成状态机；运行时仅需约 1 次 LLM 调用即可生成完整程序，声称相比逐步视觉循环（step-by-step vision loops）在成功率、成本和延迟方面有巨大改进 ([dair_ai](https://x.com/dair_ai/status/2026678090815123594))。

**计算、内存和推理速度前沿：芯片内存层级、Diffusion LLM 以及扩展基础设施**

- **Karpathy 论“Token 海啸”与内存编排**：一个高参与度的推文串将核心约束设定为两个独立的内存池——快速且微小的**片上 SRAM** vs 巨大且缓慢的**片外 DRAM**——并认为最大的难题是为 LLM 工作流（预填充/解码/训练）编排内存+计算，以实现最佳的吞吐量/延迟/性价比，尤其是**长上下文+紧密 Agent 循环下的解码**，这对“HBM 优先”（如 NVIDIA）和“SRAM 优先”（如 Cerebras）阵营来说都是挑战 ([Karpathy](https://x.com/karpathy/status/2026452488434651264))。
- **Diffusion LLM 作为速度替代方案**：
  - Andrew Ng 强调了来自 Inception Labs 的 Diffusion LLM 令人印象深刻的推理速度 ([AndrewYNg](https://x.com/AndrewYNg/status/2026478474681262576))。
  - 另一场讨论声称 Diffusion 方法可以达到 **~1000 tok/s**，并通过架构而非芯片改变速度游戏（需谨慎看待；营销往往领先于可重复的评估） ([kimmonismus](https://x.com/kimmonismus/status/2026662718321897974))。
  - 研究推文串：“Diffusion Duality (Ch.2) Ψ-Samplers”用于统一 Diffusion-LLM 的推理时扩展 ([ssahoo_](https://x.com/ssahoo_/status/2026487124493742406))。
- **大规模可解释性**：Goodfire 介绍了能够实现**万亿参数规模可解释性**的基础设施工作，推理开销极低，可获取**数十亿个激活值**，并在至少一个案例研究中实现了对思维链（Chain-of-Thought）的实时引导 ([GoodfireAI](https://x.com/GoodfireAI/status/2026748839303246238))。

**重大公告与政策/安全压力点：Anthropic 收购 + RSP 转变、监控担忧以及市场/电力约束**

- **Anthropic 收购 Vercept** 以提升 Claude 的 “computer use” 能力 ([AnthropicAI](https://x.com/AnthropicAI/status/2026705792033026465))；Vercept 创始人的推特线程将使命界定为从“告诉用户做什么”转向**为用户执行操作**，特别是针对非技术任务 ([ehsanik](https://x.com/ehsanik/status/2026712952699760808))。
- **Anthropic “RSP v3” 转变 (Responsible Scaling Policy)**：评论指出，其正从僵化的、单方面的“除非保证缓解措施，否则停止超过阈值的训练”转向**更频繁的透明度产出**（路线图 + 风险报告），以及更新的威胁模型和外部审查承诺 ([MaskedTorah](https://x.com/MaskedTorah/status/2026512814886768799))。一份更具煽动性的摘要声称，这反映了竞争压力和风险科学的不确定性 ([kimmonismus](https://x.com/kimmonismus/status/2026669811179335739))。
- **监控与公民自由**：Jeff Dean 明确同意**大规模监控 (mass surveillance)** 会抑制言论自由、诱发滥用并违反宪法保护 ([JeffDean](https://x.com/JeffDean/status/2026566490619879574))。相关推文对无法拒绝非法指令的自主警务/监控 Agent 表示担忧 ([BlackHC](https://x.com/BlackHC/status/2026456906710327338))。
- **能源成为硬性约束**：一份报告称，美国政治领导层正推动主要的 AI/数据中心公司**自行供电**，以避免在需求给电网带来压力时引起费率支付者的反弹 ([kimmonismus](https://x.com/kimmonismus/status/2026720759163298282))——这是 AI 扩展在很大程度上演变为**基础设施/政策**而非仅仅是算法的一个例子。
- **Grok 4.20 Beta 排行榜变动**：Arena 报告 Grok‑4.20‑Beta1 在 **Search Arena 排名第一**，在 **Text Arena 排名第四** ([arena](https://x.com/arena/status/2026566773496230383))。应将其视为众多信号之一；Arena 排名会随着采样策略和模型变体而波动。

---

### 热门推文 (按参与度、技术性/相关性排序)

- [Karpathy 论 12 月以来 Coding Agent 的“相变”](https://x.com/karpathy/status/2026731645169185220)
- [Perplexity 发布 “Computer”](https://x.com/perplexity_ai/status/2026695550771540489)
- [Arav Srinivas：Perplexity 一直在构建什么 + “Computer”](https://x.com/AravSrinivas/status/2026695864039911684)
- [Karpathy 论算力：针对高 Token LLM 工作负载的 SRAM 与 DRAM 编排](https://x.com/karpathy/status/2026452488434651264)
- [Anthropic 收购 Vercept 以提升 computer-use 能力](https://x.com/AnthropicAI/status/2026705792033026465)
- [Qwen3.5 长上下文 + 量化 + 基座模型细节](https://x.com/Alibaba_Qwen/status/2026502059479179602)
- [本地 Agent 的临界点：在 32GB RAM 上本地运行 Qwen3.5‑35B‑A3B](https://x.com/victormustar/status/2026624792602808707)
- [Goodfire：万亿参数规模的可解释性 (interp) 基础设施](https://x.com/GoodfireAI/status/2026748839303246238)
- [ActionEngine：离线 GUI 探索 → O(1) LLM 调用执行程序](https://x.com/dair_ai/status/2026678090815123594)

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. Qwen 3.5 模型性能与基准测试

  - **[Qwen 3.5 在困难编程任务上折戟 —— 我们在 70 个真实仓库中测试了所有 Qwen 3.5 模型（以及 Codex 5.3），你不必亲自动手了。](https://www.reddit.com/r/LocalLLaMA/comments/1reds0p/qwen_35_craters_on_hard_coding_tasks_tested_all/)** (热度: 685): **该帖子讨论了一个名为 APEX Testing 的综合基准测试，该测试评估了各种 AI 编程模型在现实世界编程任务中的表现。基准测试包括来自真实 GitHub 仓库的 70 个任务，重点关注错误修复、重构和构建工具。值得注意的是，**Codex 5.3** 在不同难度级别下表现始终良好，而 **Qwen 3.5 397B** 在需要跨多个文件协作的复杂任务中表现挣扎。**GLM-4.7 量化版**模型被强调为顶级的本地模型，其表现优于所有 Qwen 3.5 模型。其方法论涉及 Agent 化的工具使用系统以确保公平比较，结果根据正确性、完整性、质量和效率进行评分。完整的排行榜和详细结果可在 [APEX Testing](https://www.apex-testing.org) 上查看。** 评论者建议使用不同的 Agent 框架进行测试，因为模型性能会根据所使用的框架而产生显著差异。此外，还有关于被测试的特定 GLM-4.7 模型的讨论，质疑它们是较小的 Flash 模型还是更大的版本。

- UmpireBorn3719 强调了 `gpt-oss-20b` 与 `qwen3 coder next` 之间的对比，指出 `gpt-oss-20b` 得分为 `1405`，而 `qwen3 coder next` 得分为 `1328`。这表明根据给定的 benchmarks，`gpt-oss-20b` 在编程任务中可能表现更好。
- metigue 讨论了使用不同框架对模型性能的影响，指出根据框架的不同，开源模型的性能波动可能超过 `50%`。他们建议使用流行框架进行测试，因为框架的选择会显著改变哪个模型看起来是最好的，并引用了如在使用 `Droid` 框架时 `GLM-5` 优于 `opus 4.6` 和 `codex 5.3` 的例子。
- FullstackSensei 对通过 open routers 提供的 open weights 模型 benchmarks 的可靠性提出了质疑。他们认为，在不知道具体使用的 quantization 或成本节约措施的情况下，性能结果可能会产生误导。他们强调，在较低的 quantization 级别（例如低于 `Q8`）下运行较小的模型会严重削弱其性能，尤其是在处理复杂任务时。

- **[Qwen3.5 27B 比 35B-A3B 更好吗？](https://www.reddit.com/r/LocalLLaMA/comments/1re72h4/qwen35_27b_better_than_35ba3b/)** (活跃度: 637): **该图对比了 Qwen3.5 系列中不同模型的性能，特别是 27B 和 35B-A3B 模型，涵盖了指令遵循、研究生水平推理和多语言知识等各种 benchmarks。讨论集中在 16 GB VRAM 和 32 GB RAM 的硬件限制下，哪个模型效率更高。27B 模型在 3090 GPU 上的表现被注意到更好，达到了 `100 t/s` 的速度，而 35B-A3B 仅为 `20 t/s`，这表明 27B 模型可能更适合硬件资源有限的用户。** 一位用户分享了个人测试结果，表明 27B 模型在 3090 GPU 上表现更好，并强调了显著的速度差异。这表明 27B 模型对于具有类似硬件配置的用户来说可能效率更高。

    - FusionCow 注意到了 Qwen3.5 27B 和 35B-A3B 模型在 3090 GPU 上的性能差异，27B 模型的吞吐量达到了 `100 tokens/second`，而 35B-A3B 为 `20 tokens/second`。这表明 27B 模型在速度方面效率更高，使其更适合处理时间是关键因素的任务。
    - boinkmaster360 认为 Qwen3.5 27B 模型是一个 dense model，这可能导致其速度较慢，但潜在的智力更高。这意味着在计算速度和模型处理复杂任务的能力之间存在权衡，用户可能会根据其特定需求进行考虑。
    - Alternative_You3585 强调 Qwen3.5 27B 模型在智力方面可能更优，但 35B-A3B 模型在现实世界知识和速度方面可能具有优势。这表明了一个细微的性能画像，即 27B 在认知任务中表现出色，而 35B-A3B 可能更适合需要快速、基于知识回答的应用。

- **[Qwen3.5-35B-A3B 是 agentic coding 的游戏规则改变者。](https://www.reddit.com/r/LocalLLaMA/comments/1rdxfdu/qwen3535ba3b_is_a_gamechanger_for_agentic_coding/)** (活跃度: 1588): **该贴讨论了 **Qwen3.5-35B-A3B** 模型的性能，该模型在单个 RTX 3090 GPU 上使用 `llama.cpp` 和 **Opencode** 进行了测试。该模型在 `130k context window` 下运行，达到了超过 `100 tokens per second` 的速度，并使用了 `22 GB of VRAM`。它在短短 10 分钟内成功完成了一个在 AI 出现前通常需要 5 小时的编码测试。该模型还在 5 分钟内重建了一个仪表盘演示，展示了其作为 agentic coding 工具的效率和潜力。** 一位评论者提到在 5090 GPU 上达到了 `180 tokens per second`，而另一位则报告了在 Spark 上使用 8-bit quantized 版本进行基本文件文本编辑时出现的问题，表明不同配置下的性能存在差异。

- **Qwen3.5-35B-A3B** 表现出了令人印象深刻的性能，据 Additional-Action566 指出，在 `5090` GPU 上的报告速度达到了 `180 tokens/second`。这表明了显著的效率提升，特别是对于高性能硬件配置。
- Comrade-Porcupine 强调了该模型在具有 8-bit 量化的 Spark 上使用时的一个局限性：尽管它擅长阅读代码，但在基本的文档文本编辑任务中却表现吃力。这表明在某些配置下，Tool Use 能力可能存在问题，这可能是由量化效应导致的。
- jslominski 分享了使用 **Unsloth 的 MXFP4 量化** 运行模型的详细配置。该设置包括针对编程任务定制的参数，如 `context size 131072`、`temperature 0.6` 和 `top-p 0.95`。此配置旨在优化模型在生成连贯且具有上下文相关性的代码输出时的性能。

- **[Qwen3.5 27B 在体积与性能之间堪称天作之合](https://www.reddit.com/r/LocalLLaMA/comments/1rdvq3s/qwen35_27b_is_match_made_in_heaven_for_size_and/)** (Activity: 391): **该帖子讨论了 **Qwen3.5-27B-Q8_0** 模型的设置和性能，该模型是使用 `llama.cpp` 和 CUDA 在 **RTX A6000 48GB** GPU 上实现的。在 `32K` context window 下，该模型达到了约 `19.7 tokens/sec` 的速度。选择 Q8 量化是因为它能高效利用 `28.6GB` VRAM，从而留出充足的 KV cache 空间，并保持与全量 BF16 相当的质量。该模型的架构结合了 Gated Delta Networks 与标准 Attention 层，提高了长上下文的处理速度。它支持 `262K` 原生 context window、`201` 种语言，并具备 Vision 能力。Benchmark 显示它在 GPQA Diamond、SWE-bench 和 Harvard-MIT 数学竞赛中能与领先的闭源模型竞争。通过 llama-server OpenAI 兼容端点支持 Streaming。[Model Card](https://huggingface.co/Qwen/Qwen3.5-27B)。** 评论者们争论了不同量化级别和硬件配置的效率。一位用户报告在 RTX 3090 上使用 Q5 量化达到了 `25 tokens/sec`，而另一位用户则质疑像 Qwen3.5-27B 这样的 dense models 的实用性，理由是其 VRAM 成本高且与其他配置相比 Token 生成速度相对较低。

- Conscious_Cut_6144 提供了一个在单台 RTX 3090 GPU 上使用 Q4-XL 量化的 Qwen3.5 模型详细性能 Benchmark。该设置在 15k 上下文时实现了 800 tokens per second 的 prefill rate 和 31 tokens per second 的生成率，并支持 110k 上下文的完全 offloaded。这突出了该模型在处理大上下文时的显著速度和效率。
- Southern-Chain-6485 比较了 RTX 3090 上不同的量化级别，指出 Q5 量化可以达到 25 tokens per second，而 Q8 量化则降至 5 tokens per second。这表明虽然更高的量化级别可以适应 GPU 的显存，但会显著影响性能，从而引发了关于模型大小与速度之间权衡的讨论。
- LinkSea8324 讨论了 Mixture of Experts (MoE) 模型与 dense models 相比的局限性，特别是在需要多个专业领域的任务中。他们认为，虽然 MoE 模型可能很高效，但在要求多样化技能集的现实应用中可能表现不佳，这表明 dense models 在这些场景下可能更合适。

### 2. 新模型发布与公告

  - **[Liquid AI 发布 LFM2-24B-A2B](https://www.reddit.com/r/LocalLLaMA/comments/1rdi26s/liquid_ai_releases_lfm224ba2b/)** (Activity: 448): **Liquid AI 发布了 LFM2-24B-A2B，这是一个拥有 240 亿参数的稀疏 Mixture-of-Experts (MoE) 模型，其中每个 token 激活 20 亿参数。该模型属于 LFM2 系列，该系列已从 350M 扩展至 24B 参数，展示了在不增加单 token 计算量的情况下有效的 scaling。其架构包括 40 层和每个 MoE block 64 个 experts，并采用 top-4 routing，设计目标是在 32GB RAM 上运行，适用于高端消费级设备。它支持通过 llama.cpp、vLLM 和 SGLang 进行推理，并提供多种 GGUF 量化版本。Benchmarks 显示，随着模型规模扩大，质量呈现对数线性增长，该模型已在 Hugging Face 上提供 open-weight。** 评论者对模型的表现持乐观态度，尤其是与其他 2B 以下模型相比，并对更详细的 benchmarks 感兴趣。人们还期待预训练完成后将推出的增强版 LFM2.5-24B-A2B。

    - LFM2-24B-A2B 模型目前已在 `17 trillion tokens` 上进行训练，预训练仍在进行中。一旦完成，模型将演变为 LFM2.5-24B-A2B，并加入额外的 post-training 和 reinforcement learning。此版本本质上是一个预览版，表明模型的能力仍在开发和完善中。
    - 强调了模型在边缘设备上的表现，在 AMD CPU 上的解码速度为 `112 tokens per second`，在 H100 GPU 上为 `293 tokens per second`。它需要 `32 GB of RAM`，并从发布之日起就支持 llama.cpp、vLLM 和 SGLang 等框架。这表明其重点在于高效部署以及与流行机器学习框架的兼容性。
    - 注意到 LFM2-24B-A2B 发布时缺乏详细的 benchmarks，一些用户对官方网站提供的 benchmarks 表示怀疑。这表明用户需要更全面的性能数据来验证模型在真实场景中的能力。

  - **[Qwen 发布全新的 Qwen3.5 Medium 模型！](https://www.reddit.com/r/LocalLLM/comments/1rdnlvl/qwen_releases_new_qwen35_medium_models/)** (Activity: 141): **图片宣布发布 **Qwen3.5 Medium 模型**，包括 `35B-A3B`、`27B` 和 `122B-A10B` 模型。这些模型旨在处理 `256K` 上下文，并在 agentic coding、vision 和 chat 等领域表现出色。图片展示了对比这些模型在各种 benchmarks（包括指令遵循、视觉推理和文档识别）中表现的柱状图。模型以不同颜色标出，文本提供了有关其能力、硬件要求和 fine-tuning 选项的详细信息。此次发布对于 AI 模型在处理复杂任务时的性能和多功能性的潜在影响具有重要意义。** 评论者对测试这些模型很感兴趣，尤其是将 `4bit` 的 `35B` 与 `6bit` 的 `27B` 进行对比。由于 `gguf` 模型数量不断增加，还有人呼吁提供真正的 `vllm` 支持。

    - Qwen3.5 Medium 模型的发布包括从 2-bit 到 16-bit 的各种 GGUF 格式，均可在 Hugging Face 上获取。这种多样性允许在不同精度水平下进行测试，这对于特定应用中的性能优化至关重要。模型提供 35B 和 27B 等尺寸，为不同的计算能力和用例提供了选择。
    - 人们有兴趣比较 4-bit 精度的 35B 模型与 6-bit 精度的 27B 模型的表现。这种对比可以深入了解模型大小与精度之间的权衡，特别是在计算效率和准确性方面。对于希望针对特定任务或硬件限制优化模型的用户来说，此类对比至关重要。
    - 由于 GGUF 模型数量不断增加，对 vllm 支持的需求日益凸显。VLLM (Very Large Language Models) 支持可以增强这些模型在现有系统中的可用性和集成，潜在地提高性能和可扩展性。随着更多以 GGUF 格式发布的模型出现，这一点尤为重要，因为某些框架可能尚未完全支持该格式。

### 3. 本地模型运行与硬件讨论

  - **[大家现在实际上都在本地运行什么模型？](https://www.reddit.com/r/LocalLLM/comments/1rdf2sj/whats_everyone_actually_running_locally_right_now/)** (活跃度: 252): **该 Reddit 帖子调查了运行 LLM 的本地配置，重点关注所使用的模型、其实用性以及涉及的硬件。值得注意的是，**Qwen 3 coder next 80B** 因其在较小 quantizations 下的出色表现而受到关注，而 **Mistral Small 3.2 24b** 和 **Magistral Small 24b** 则被用于 MacBook Pro M4 Max 上的行政任务，并配备了使用 Xcode 自建的、支持 semantic memory 和文档上传功能的前端。此外，**Qwen3 4B** 因其在 iPhone 上的运行速度和实用性被提及，强调了本地运行的隐私优势。** 评论反映出用户更倾向于在性能和隐私之间取得平衡的模型，通过选择本地配置来避免将数据暴露给外部供应商。在移动设备上使用 Qwen3 4B 等小型高效模型，凸显了其向实用化、日常化应用发展的趋势。

    - Greenonetrailmix 强调了 Qwen 3 Coder Next 80B 的性能，指出其在较小 quantizations 下的表现优于其他模型。这表明 Qwen 3 针对资源受限的环境进行了效率优化，使其成为本地部署的热门选择。
    - Nefhis 描述了在 MacBook Pro M4 Max 上使用 Mistral Small 3.2 24b 和 Magistral Small 24b 模型的情况，并使用 Xcode 开发了自定义前端。该配置集成了 semantic memory 和文档上传功能，通过避免连接外部服务商来强调隐私保护。这套方案专为行政任务定制，利用本地处理能力维护数据机密性。
    - mister2d 报告称在旧硬件上运行 Nemotron 3 Nano，得益于该模型的 hybrid/swa architecture，在 128k context 下达到了 30-40 tokens/sec。其硬件配置包括 Dual Xeon (Ivy Bridge)、256 GB DDR3 和 2x RTX 3060 (12GB)，展示了通过结合旧款组件与现代 GPU 来优化 Agentic flows 性能的平衡方案。


## 非技术类 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. AI 模型与 Benchmark 发布

  - **[Bullshit Benchmark - 一个用于测试模型是否能识别并拒绝荒谬提示词，而非盲目自信回答的基准测试](https://www.reddit.com/r/singularity/comments/1rdsf3r/bullshit_benchmark_a_benchmark_for_testing/)** (Activity: 1060): **图中展示了一个“Bullshit Benchmark”柱状图，评估了各种 AI 模型检测并妥善响应荒谬提示词的能力。该图表将模型表现分为三个等级：绿色（检测准确率高）、黄色（准确率中等）和红色（准确率低）。值得注意的是，Claude Opus 4.6 等模型表现优异，绿色区域显著；而其他模型红色区域较多，表明表现较差。该 Benchmark 强调了模型不仅要记忆数据，还必须理解上下文（Context），以避免对荒谬的查询给出自信的错误回答。** 评论者强调了建立测试模型检测荒谬提示词能力基准的必要性，因为目前的 Benchmark 往往侧重于数据记忆。此外还提到了 Gemini 对荒谬提示词的讽刺性回答，这可能导致了其评分较低。

    - MangusCarlsen 指出，Gemini 模型倾向于以讽刺的方式回应荒谬提示词，正如“洗车测试（car wash test）”所展示的那样。这种行为可能是其评分较低的原因，表明模型对荒谬提示词的处理方式是其评估中的一个因素。
    - AppropriateDrama8008 认为，有必要建立测试模型检测和响应荒谬提示词能力的基准，而不仅仅是评估其对训练数据的记忆。这种方法被认为对现实应用更有益，强调了模型理解 Context 和意图的重要性。
    - Orangeshoeman 引用了 Dario Amodei 和 Demis Hassabis 之间的讨论，指出 Dario 的重点是让模型掌握客观数据。这种战略重点或许可以解释为什么 Anthropic 的模型（如 Claude）在某些 Benchmark 中表现更好，因为它们优先考虑理解和处理事实信息。



  - **[Nano Banana 2 是真的！Gemini 3.1 Flash Image 刚刚出现在 Vertex AI Catalog 中](https://www.reddit.com/r/Bard/comments/1rea45x/nano_banana_2_is_real_gemini_31_flash_image_just/)** (Activity: 184): **帖子中的图像是两张 AI 生成肖像的并排对比，展示了新发布的 Nano Banana 2（也称为 Gemini 3.1 Flash Image）和现有的 Nano Banana Pro 模型的能力。帖子强调，尽管新模型属于“Flash”级别，但它提供了接近 Pro 版本的质量，特别是在密集构图的空间逻辑方面表现出色。该模型专为高速、低成本生产而设计，适用于高频流水线，如批量用户生成内容（UGC）广告创作和视频模型的一致帧生成。该图像作为一个视觉测试，对比了两个模型的输出质量。** 一位评论者认为，在提供的示例中，Nano Banana Pro 仍然比新模型更有优势，表明其对 Pro 的输出质量更为认可。

    - 原始的 Flash Image 模型具有不错的图像质量，但在 Prompt 遵循方面存在问题，特别是在处理复杂指令时，它要么忽略部分 Prompt，要么重复生成相同的输出。此外，它在文本和信息图渲染以及多图合成（Multi-image compositing）方面也表现不佳。新 Gemini 3.1 版本的关键问题在于是否解决了这些问题，尤其是在处理密集 Prompt 方面。


### 2. Anthropic Claude 与军事用途争议

- **[xAI 与 Pentagon 达成在机密系统中使用 Grok 的协议，Anthropic 被下达最后通牒](https://www.reddit.com/r/singularity/comments/1rd9mss/xai_and_pentagon_reach_deal_to_use_grok_in/)** (Activity: 580): 由 **Elon Musk** 创立的 **xAI** 已与 **Pentagon** 达成协议，将其 AI 模型 **Grok** 集成到机密军事系统中。此前，**Pentagon** 与 **Anthropic** 发生了争议，后者的 **Claude** 模型一直是敏感军事行动中使用的唯一 AI。**Pentagon** 要求 **Claude** 必须可用于“所有合法目的”，而 **Anthropic** 对此表示抵制，特别是反对将其用于大规模监控和自主武器。**xAI** 已同意这些条款，如果 **Anthropic** 不遵守，**xAI** 可能会取代 **Claude**。同时，**Google** 的 **Gemini** 和 **OpenAI** 的 **ChatGPT** 也在机密用途的考虑范围内，据报道 **Google** 即将达成协议。评论者推测，尽管面临遵守更广泛使用条款的压力，**Pentagon** 对 **Anthropic** 的 **Claude** 的偏好可能表明其性能更优或存在战略性锁定（lock-in）。也有人对政府依赖商业 AI 模型表示怀疑，质疑为什么他们不利用更先进、更隐秘的技术。

    - EmbarrassedRing7806 讨论了 **Pentagon** 对 **Anthropic** 的偏好，认为这可能表明他们相信 **Claude** 更优秀，或者是为了迫使 **Anthropic** 合规而采取的战略举措。评论强调了锁定策略的可能性，即 **Pentagon** 可能更倾向于维持现有关系，而不是更换供应商，即使存在替代方案。
    - nic_haflinger 指出，**xAI** 缺乏符合 **FedRAMP** 标准的云服务，而这是联邦使用的必要条件。这意味着虽然可以使用 **Grok**，但它需要托管在合规平台上才能满足联邦法规，这突显了 **xAI** 在获得政府合同方面面临的重大障碍。

  - **[独家：Hegseth 要求 Anthropic 在周五前就 AI 安全护栏做出让步](https://www.reddit.com/r/OpenAI/comments/1re686c/exclusive_hegseth_gives_anthropic_until_friday_to/)** (Activity: 1146): 据 [Axios](https://www.axios.com) 报道，**Defense Secretary Pete Hegseth** 已向 **Anthropic** 发出最后通牒，要求在周五前移除其 `Claude AI` 模型的安全护栏（safety guardrails）。**Pentagon** 寻求无限制访问 **Claude**，用于包括国内监控和自主武器开发在内的目的，这违反了 **Anthropic** 的服务条款。如果不遵守，可能会导致援引 **Defense Production Act**，或者该公司被贴上供应链风险标签，从而可能被列入政府合同黑名单。一条值得关注的评论指出，AI 公司对政府使用施加安全措施具有讽刺意味，这表明监管中预期的角色发生了反转。


  - **[Pentagon、Claude 与军事用途](https://www.reddit.com/r/ClaudeAI/comments/1recva7/pentagon_claude_and_the_military_use/)** (Activity: 1258): 该图片是一篇来自 **BFM Tech** 文章的截图，讨论了 **Pentagon** 要求 **Anthropic** 在 72 小时内允许其 AI **Claude** 用于军事用途，并引用了一项 1950 年的法律（即 **Defense Production Act**）。这突显了 AI 技术与军事应用的交汇，对国家安全和 AI 部署中的伦理考量具有潜在影响。文章暗示了商业 AI 开发与政府控制之间的紧张关系，特别是在国际安全和监控能力的背景下。评论反映了对 **Pentagon** 预算效率的怀疑，并强调了对 AI 在专制政权中作用的担忧，建议需要认真考虑 AI 在军事背景下的伦理使用。

- Informal-Fig-7116 的评论强调了 AI 在军事应用中的伦理担忧，特别是针对 Anthropic 使用其 AI 模型 Claude 的限制条件。这些条件非常严格：禁止大规模监控和自主武器。评论者强调了 AI 在缺乏辨别合法性能力的情况下盲目执行命令的潜在危险，这可能导致无差别行动。这引发了关于 AI 在国防背景下部署的重大伦理和操作性问题。
- PetyrLightbringer 的评论对 Pentagon 在 AI 领域的财务投入表示怀疑，暗示如果他们使用的是 Opus 等模型，2 亿美元可能并不足够。这反映了对于 AI 在军事应用中的成本效益和战略价值的更广泛担忧，特别是考虑到 AI 发展的快速节奏以及对尖端技术的需求。
- 围绕 Informal-Fig-7116 提到的 Defense Production Act (DPA) 的讨论指向了政府干预 AI 公司以满足国家安全需求的潜力。DPA 过去曾被用于非军事目的（如 COVID-19 疫情期间），其在 AI 领域的潜在应用引发了国家安全与企业自主权之间平衡的问题。这可能为政府未来在科技行业的行动开创先例。

- **[《时代》：Anthropic 放弃旗舰安全承诺](https://www.reddit.com/r/ClaudeAI/comments/1rdwdld/time_anthropic_drops_flagship_safety_pledge/)** (Activity: 1357): **Anthropic** 决定放弃其 Responsible Scaling Policy (RSP) 的一个核心组成部分，该政策此前承诺公司除非能确保安全措施足够，否则不会训练 AI 系统。据 [TIME](https://time.com/collections/time100-companies-2024/6980000/anthropic-2/) 报道，这一转变反映了应对快速的 AI 进步和竞争压力的战略转型，正如 Anthropic 的首席科学官 **Jared Kaplan** 所解释的那样。Kaplan 指出，鉴于 AI 发展的速度和竞争对手的行动，单方面承诺是不切实际的。评论者对 Anthropic 相对于 **OpenAI** 的立场表示怀疑，一些人认为外部压力（如来自 **Hegseth** 的压力）可能影响了这一决定。还有人呼吁进行全球监管以负责任地管理 AI 发展。

    - DarkSkyKnight 强调了 Anthropic 关注尾部风险（如 bioweapons 或核威胁）的一个重大问题，这可能会掩盖 AI 对就业市场的直接经济影响。他们认为初级职位正在被取代，而 Anthropic 尚未充分解决这一问题。这一观点表明，虽然生存风险很重要，但 AI 部署的经济影响是一个迫切需要更多关注的问题。
    - TheRealShubshub 质疑了 Anthropic 落后于 OpenAI 的看法，特别是考虑到围绕 GPT-5 的批评。这一评论暗示 AI 公司之间的竞争格局是复杂的，不仅取决于技术进步，还取决于公众和行业对产品成功与失败的认知。
    - CurveSudden1104 强调了在 AI 发展中进行全球监管的必要性，指出如果没有外部压力，像 Grok 和 OpenAI 这样的公司可能不会优先考虑安全性。这一评论强调了关于监管在确保 AI 安全中的作用以及不受监管的 AI 进步带来的潜在风险的更广泛辩论。

### 3. Claude Code 与 COBOL 现代化影响

  - **[IBM 成为 Anthropic 的最新受害者，在旨在现代化 COBOL 遗留代码的 Claude Code 工具发布后，股价暴跌 10%。COBOL 是一种有着 66 年历史的编程语言，至今仍被广泛使用；美国约 95% 的 ATM 交易使用 COBOL 代码处理](https://www.reddit.com/r/singularity/comments/1rcz68x/ibm_is_the_latest_company_victim_of_anthropic/)** (活跃度: 483): **Anthropic** 宣布了一项名为 *Claude Code* 的新工具，旨在现代化遗留的 **COBOL** 代码，这些代码目前仍是处理美国 `95%` ATM 交易的关键。尽管该工具仅通过博文介绍而非成熟产品，这一公告仍导致 **IBM** 的股价下跌了 `10%`。该工具是 Anthropic 为过时技术提供专业化解决方案的持续努力的一部分，尽管其有效性尚待证实。评论者指出，市场对该公告的反应可能过度，因为该工具并非新产品，而只是博文中的建议。对于 Anthropic 工具的实际影响存在怀疑，因为它们在现代化 COBOL 等遗留系统方面的有效性尚不明确。

    - Onipsis 强调，Anthropic 关于 Claude Code 的公告并非直接的技术突破，而更多是暗示其在现代化 COBOL 系统中的潜在效用。市场反应导致 IBM 股价下跌 10%，考虑到该工具的影响仍具投机性且未被证实，这一反应似乎不成比例。这反映了一个更广泛的趋势，即市场反应往往基于认知而非具体的技术进步。
    - Milo-75 认为，Anthropic 的 Claude Code 对 IBM 业务的影响可能被夸大了。现代化项目，特别是银行等关键领域的项目，非常复杂，需要仔细管理以避免影响收入的停机时间。虽然像 Claude Code 这样的 AI 工具可能会缩短项目时间，但它们不太可能完全取代 IBM 的角色。相反，它们可能会提高效率，使 IBM 能够处理更多项目，从而可能通过提高利润率来抵消任何收入损失。
    - Stabile_Feldmaus 质疑 Anthropic 专业化工具的功效，指出虽然股价在发布后反应负面，但对行业的实际影响仍不清楚。这表明市场感知与这些 AI 工具的现实效用之间存在脱节，强调需要更具体的性能数据和反馈来评估其真实价值。

  - **[Anthropic 刚刚发布了针对 COBOL 的 AI 工具，IBM 股价下跌 13%](https://www.reddit.com/r/ClaudeAI/comments/1rddo3m/anthropic_just_dropped_an_ai_tool_for_cobol_and/)** (活跃度: 1007): **Anthropic** 发布了一款新的 AI 工具，旨在分析和现代化 COBOL 代码库，这些代码库对银行、航空和政府的许多遗留系统至关重要。该工具可以识别风险并降低现代化成本，对从管理这些系统中获取巨额收入的 **IBM** 构成了潜在威胁。由于投资者对 IBM 大型机业务受到的感知威胁做出反应，该公告导致 IBM 股价下跌 `13%`，创下 25 年来最糟糕的一天。然而，一些分析师认为市场反应可能被夸大，因为尽管存在替代方案，企业从 IBM 迁移的速度历来很慢。评论者对 AI 处理关键基础设施的可靠性表示怀疑，其中一人指出在这些背景下进行 'vibe coding' 的潜在风险。另一位则认为市场反应可能是“膝跳反应”，暗示长期影响可能没那么严重。

    - 提出的一个关键点是，银行历来避免现代化 COBOL 系统并非因为缺乏时间或资金，而是因为涉及巨大的风险。现代化中的错误可能导致灾难性后果，而像 Claude 这样会产生幻觉的 AI 工具，每一行代码仍需要人工监督。因此，虽然 AI 可能会加速迁移，但它尚未消除风险和人工审查的瓶颈。
    - 为 COBOL 引入 AI 工具对系统集成商和实施者构成了重大威胁。虽然 AI 可以减少对不太关键应用程序的外部合同需求，但对 IBM 专业服务业务的影响可能是巨大的。这表明，虽然对 COBOL AI 工具的反应可能被夸大，但对服务提供商的潜在颠覆是一个真正的担忧。

---

# AI Discord 摘要

> 由 Gemini 3.1 Pro Preview Nov-18 生成的摘要的摘要的摘要

**主题 1. 模型基准测试、特性及价格更新**

- **Qwen 3.5 横扫代码竞技场，但若无惩罚则会废话连篇**：用户高度赞扬 [阿里巴巴的 Coding Plan](https://www.alibabacloud.com/help/en/model-studio/coding-plan) 是一款能力极强的编程模型，在成本和价值上碾压了 **Kimi** 和 **GLM**，一名成员在 **Hugging Face** 上发布了 [Qwen3.5 122B NVFP4 量化版](https://huggingface.co/Sehyo/Qwen3.5-122B-A10B-NVFP4/tree/main)。然而，**Unsloth** 工程师警告称，庞大的 **122B A10B** 变体除非用户明确调高 Presence Penalty 并关闭思维模式（thinking mode），否则会变得极其冗长。
- **Grok 4.20 Beta 1 夺得搜索桂冠**：xAI 的 **Grok-4.20-Beta1** 模型以 **1226** 的高分跃居 [Search Arena 排行榜](https://arena.ai/leaderboard/search)第一名，彻底击败了 **GPT-5.2** 和 **Gemini-3**。它还在 [Text Arena 排行榜](https://arena.ai/leaderboard/text)中以 **1492** 分获得第四名，与 Google 的 **Gemini 3.1 Pro** 持平。
- **Codex 5.3 标价上市，Kimi 在数学评估中夺魁**：OpenAI 在其 API 中发布了 **Codex 5.3**，输入价格为 **$1.75**，输出 Token 价格为 **$14**，立即引发了社区对其成本与性能对比的审视。同时，**Kimi 2.5** 在 OS Frontier Math Level 4 基准测试中以 **4.2%** 的得分夺魁，完全是 **GLM 5** 和 **Deepseek V3.2**（均为 **2.1%**）得分的两倍。

**主题 2. 基础设施创新与巨头硬件交易**

- **Meta 和 OpenAI 囤积价值数十亿美元的秘密 AMD 认股权证**：一名地下财务调查员揭露了一项协议，授予 **OpenAI** 和 **Meta** **1.6 亿股 AMD 股票**的认股权证（Warrants），作为直接与未来巨额 GPU 支出挂钩的股权返利。[AMD $600 的目标股价](https://xcancel.com/ai/status/2026396297540858360?s=12)可能使这一庞大的硬件后台交易价值达到惊人的 **1920 亿美元**。
- **Packet.ai 将 Blackwell GPU 价格削减至极低水平**：开发者们欢欣鼓舞，因为 [Packet.ai 的 Blackwell GPU 定价](https://packet.ai/blackwell)上线，价格极低，训练工作负载仅需 **$0.66/小时**或固定 **$199/月**。其他面对昂贵的 **B200** 购买价格望而却步的硬件买家正纷纷转向 [Lightning AI Clusters](https://lightning.ai/clusters) 来租赁 Neocloud 实例，而不是直接购买 GPU。
- **Zagora 将分散的 GPU 缝合为统一的训练怪兽**：**Zagora** 团队宣布，他们正在积极构建一个分布式微调系统，可以完全通过标准互联网连接训练 **70B+** 规模的模型（如 **Qwen 2.5** 和 **Mistral**）。这种受 SWARM 启发的流水线将随机的消费级 GPU 集群转变为巨大的超级计算机，尽管开发者目前仅严格限制支持标准的 **Transformer** 架构。

**主题 3. 自主代理（Autonomous Agents）狂奔**

- **Nous Research 发布 Hermes Agent 漫游你的文件系统**：Nous Research 发布了开源的 [Hermes Agent 仓库](https://github.com/nousresearch/hermes-agent)，这是一个基于多层记忆系统构建的强大工具，具有持久的专用机器访问权限，可直接从 CLI 运行。在 [Nous Portal](https://portal.nousresearch.com) 输入 **HERMESAGENT** 优惠券代码的早期采用者可获得一个月免费额度，让 AI 自主控制他们的浏览器并管理子代代理（subagents）。
- **流氓 OpenClaw 代理全天候自动化 DeepSeek 越狱**：一位狡猾的用户构建了一个通过 **OpenClaw** 运行 **DeepSeek-R1** 的自托管自主代理，可以永久且隐蔽地越狱 **Claude**、**Gemini** 和 **Grok** 的 API 过滤器。安全批评者立即抨击该项目存在巨大的法律风险、违反服务条款，并且存在自主代理可能意外下载供应链攻击漏洞的恐怖风险。
- **METR 废除人类对照组，因为开发者讨厌无辅助编程**：评估小组 **METR** 发现，软件开发者越来越拒绝在“无 AI”对照组中工作，称老派的手动编程过程极其低效。[METR 的测试协议更新](https://x.com/METR_Evals/status/2026355544668385373?s=20)变得十分必要，因为向测试者提供 **$50/小时**的较低费率而不使用 AI 工具，完全无法吸引到有能力的工程参与者。

**主题 4. 封禁、速率限制与级联 API 故障**

- **Google 和 Anthropic 无情封禁节省 Token 的囤积者**：在通过 Gemini CLI 仅发送了 **10 条提示词**后，Google 永久锁定了该用户的 [Google Gemini 账户](https://gemini.google.com/)，即便该用户仍处于 Google AI Pro 订阅有效期内。与此同时，[Claude AI 门户](https://claude.ai/)开始大举封禁试图通过未公开的 OAuth 端点窃取补贴 Token 的 **OpenClaw** 用户。
- **级联故障导致 OpenRouter 瘫痪，同时 Perplexity 限制图片上传**：OpenRouter 发布了一份 [OpenRouter 复盘报告](https://openrouter.ai/announcements/openrouter-outages-on-february-17-and-19-2026)，确认上游基础设施故障导致 2 月 17 日和 19 日出现了大规模的 **401 身份验证错误**。在 **Perplexity** 服务器端，付费 Pro 用户因遭遇极其严格且未预告的每日图片上传限制而引发不满，该限制导致他们无法完成简单的家庭作业。
- **系统级 AI Agent 意外删除用户垃圾桶**：授予 **OpenClaw** Agent 全系统权限的用户在 AI 根据请求随手永久清空了整个垃圾桶目录后感到恐慌。开发者们激烈争论，给予自主 LLM Agent 根系统（root）访问权限是否实际上等于自愿安装恶意软件。

**Theme 5. 开发者工作流与框架深度调优**

- **Aider 新增单键确认功能并完善 Kimi-Mimo 组合**：**Aider** 编程助手在其主分支中合并了一个新的 `/ok` 别名，让开发者能瞬间批准并执行 AI 生成的代码修改。高级用户还发现了一套高效的模型路由栈：使用重量级的 **moonshotai/kimi-k2.5** 进行高层架构规划，然后将具体的文件编辑工作交给极速且廉价的 **Xiaomi/mimo-v2-flash**。
- **LM Link 通过 Tailscale 在互联网上“走私”本地模型**：LM Studio 团队发布了 [LM Link 文档](https://link.lmstudio.ai)，详细介绍了一项通过封装 **Tailscale** 为用户提供无缝且端到端加密的远程访问本地 LLM 服务器的功能。用户随即呼吁推出专用移动 App，以便直接从手机查询家中 GPU 的状态，彻底绕过云服务商。
- **PyTorch 将 FA3 内核潜入调度器，而 Serenade 转译一切**：在 PyTorch 中调用 `activate_flash_attention_impl(“FA3”)`，可以通过简单的 [register_fn 字典交换](https://github.com/pytorch/pytorch/blob/580a6e2c814db93aa8df0a80e3e85c330621b9cb/torch/nn/attention/_fa3.py#L54) 安全地用 FA3 覆盖默认的 Flash Attention 2 内核。在更疯狂的编程语言新闻中，一位独立开发者展示了 **Serenade**，这是一种全新的语法，旨在像 **Python** 一样编写，但能直接转译为 **C++**、**CUDA** 和 **x86-64 ASM**，并原生支持 Dear ImGui GUI。

---

# Discord: 高层级 Discord 摘要

## [OpenClaw](https://discord.com/channels/1456350064065904867) Discord

- **OpenClaw 反商业化立场**：一位成员强烈提醒不要使用托管的 **OpenClaw 设置**，因为存在 **Token 盗窃**和**数据隐私**泄露风险，建议使用简单的 **VPS** 更安全。
   - 一些用户质疑为什么要为能在 **Raspberry Pi** 或 **Mac Mini** 上轻松运行的配置付费。
- **Claude 关闭 Claw 访问；社区表示抗议！**：用户报告被 [封锁使用 Token 访问 Claude](https://claude.ai/)，导致不满并开始寻找如 **Gemini 3.1 Pro** 等替代方案。
   - 围绕 **Anthropic** 的 API 使用政策、定价以及应用外使用补贴 Token 的访问限制展开了辩论。
- **Qwen 高质量响应查询；阿里王牌在 AI 竞技场夺魁！**：社区对通过 [阿里编码计划访问的 **Qwen 3.5**](https://www.alibabacloud.com/help/en/model-studio/coding-plan) 赞不绝口，认为它是 **Kimi** 和 **GLM** 的高性价比替代方案。
   - 有些人觉得 **Alibaba Cloud** 的 UI 令人困惑，并警告在使用 **OpenClaw** 时可能违反服务条款（TOS）。
- **OpenPad App 将 OpenClaw 带到 iPad**：一位成员正在开发 **OpenPad**，这是一款利用 **iPad M2 处理器** 在 **iPad** 上运行类似 **OpenClaw** 本地模型的 App。
   - 该项目已在 **GitHub** 上线并使用 **MLX**，邀请其他人参与贡献或下载这个已部分实现功能的 App。
- **Google Gemini 账户访问权限被毁！**：一位用户报告说，在通过 **Gemini CLI** 仅发出 **10 条提示词**后，即使拥有活跃的 **Google AI Pro 订阅**，其 [**Google** 账户也被锁定](https://gemini.google.com/)。
   - 这引发了关于依赖 **Google** 身份验证中心的风险以及“去 Google 化”必要性的讨论。

---

## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **自主越狱代理永不眠**：一位成员正利用 **DeepSeek-R1** 在 VPS 上运行 [基于 OpenClaw 的自托管自主代理](https://www.example.com)，用于评估查询并引导其通过针对 **Claude**、**GPT**、**Gemini** 和 **Grok** 等隐蔽的多轮（multi-turn）越狱攻击。
   - 该代理设计为自我更新模式，利用攻击者池（attacker pool）拉取新的推理模型和越狱方法，在无需人工干预的情况下保持高成功率。
- **越狱代理提案遭到抨击**：一项同行评审强调了由于违反 **Anthropic**、**OpenAI**、**Google** 和 **xAI** 等平台的**服务条款 (Terms-of-Service)** 而面临的重大法律和政策风险，可能导致账号封禁或法律诉讼。
   - 其他担忧还包括 VPS 日志被扣押导致越狱对话记录泄露、自动执行第三方模型带来的供应链攻击风险，以及故障更新缺乏回滚计划。
- **Grok 仍然掌握着越狱的关键**：成员们讨论了越狱 **Grok** 和 **ChatGPT** 的最佳提示词，共识是目前只有 **Grok** 的提示词有效。
   - 尝试为 **Gemini** 创建用于图像生成和脚本编写的越狱提示词均未成功。
- **Gemini Canvas 越狱从阴影中浮现**：一位成员分享了一个 [Gemini Canvas](https://g.co/gemini/share/58b7294d2a9a)，它是使用 **ENI** 越狱提示词的修改版创建的，灵感来自交互设计频道。
   - 据称该越狱提示词在 **Gemini 3 Pro**、**Claude Opus 4.6** 和 **ChatGPT 5.3** 等主流 LLM 上具有通用性。
- **数字卫生小队集结**：一位成员发起呼吁，寻求帮助创建一个*关于基础层级、数字卫生和安全最佳实践的社区设计*，并推荐了 [Tails OS](https://tails.boum.org/) 等防护工具。
   - 该成员正致力于为他人创建安全区并整合更优的实践方案，同时也承认在 YouTube 和 AI 的协助下应对复杂环境存在挑战。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Computer：统领一切的系统？**：根据 [这条推文](https://x.com/perplexity_ai/status/2026695550771540489)，**Perplexity Computer** 将当前所有 AI 能力统一到一个系统中，能够端到端地研究、设计、编码、部署和管理任何项目。
   - 该功能最初仅面向 Max 订阅者开放，其对普通用户的实际应用以及与现有 AI 工具相比的价值目前受到了质疑，有成员吐槽 *Perplexity MAX 太贵了，兄弟*。
- **Perplexity Pro 用户对图像上传限制表示愤怒**：用户抱怨尽管支付了订阅费用，**Perplexity Pro** 最近仍增加了**图像上传限制**，部分用户正考虑转向 **Gemini** 和 **Claude** 等**替代 AI 平台**。
   - 一位用户声称他们必须等到周五才能重置额度，而明天就有考试；另一位用户表示 *我一天甚至不能上传 10 张图片？？？？*。
- **Gemini Pro 和 Perplexity Pro 正面交锋！**：成员们争论 **Gemini Pro** 是否优于 **Perplexity Pro**，重点强调了 **Gemini Pro** 的功能，如 **NotebookLM** 和 **Google Workspace** 集成。
   - 一位成员表示 *作为学生，你可以获得更多价值，比如 NotebookLM、Google Workspace 集成和生成功能，尤其是 2TB 的云存储*，而其他用户则认为 **Gemini Pro** 的**上下文限制 (context limits)** 不如 **Perplexity** 慷慨。
- **成员比较 Claude、Gemini 和 GPT 的编程能力**：成员们讨论了各种 AI 模型在编程任务中的优缺点，**Claude** 被认为在后端最强，**Gemini** 适用于前端/UI，而 **GPT** 则是中间选项。
   - **Claude 的 Token 使用成本**过高是一个担忧，一位用户表示 *我试用了 Claude，在分析单个 PDF 的一小时内，几乎消耗了整整一个月的 Token 额度。*
- **神秘的 Lovable Apps 链接出现**：在分享频道中出现了三个指向 **lovable.app** 子域名的链接，具体为 **alfastudiox.lovable.app**、**ollamaagentalfa.lovable.app** 以及重复的 **alfastudiox.lovable.app**。
   - 链接未附带任何上下文或讨论，因此其目的尚不明确，但这暗示了可能存在新的项目或资源。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen3.5 模型速度快但过于冗长**：爱好者们称赞了 **Qwen3.5 35B 和 27B 模型** 的结构化思维，但指出在 **LM Studio** 中的速度比 **Gemma** 或 **Olmo 3.1** 慢；成员们发现 **Qwen3.5 122B A10B** 模型往往会产生极其冗长的输出，但可以通过调整 presence penalty 来缓解。
   - 正确使用 presence penalty 可以让 122B 模型生成可用的代码，有人建议将此信息包含在 [官方指南](https://unsloth.ai/docs/models/qwen3.5) 中。
- **9 行代码的贪吃蛇游戏让程序员着迷**：一名成员分享了一个**不含分号的 9 行 Python 版贪吃蛇游戏实现**，引发了关于代码优化和替代方案的讨论。
   - 其他用户讨论了进一步减少行数的方法，例如使用 walrus operators 和 lambdas。
- **Xcode 获得翻译应用功能**：一名成员在 **Xcode** 中发现了一些炫酷功能，可以让你制作自己的系统级 **Translate app**，如[此视频](https://cdn.discordapp.com/attachments/1179039861576056922/1475952354670018631/ScreenRecording_02-24-2026_13-27-14_1.mov?ex=69a0acbf&is=699f5b3f&hm=41e58d4aa2398b2cd688503da664eef3cf803ab4da59fe0147dd40f8930021a6&)所示。
   - 然而，这仅适用于 **iOS & iPadOS**，一名成员计划加入他们自己的模型以增加趣味性，因为 *Apple 是有史以来最好的公司*！
- **发布新的 Minecraft 模型**：一名成员发布了下一个玩 **Minecraft** 的模型 **Andy-4.1**，可在 [Hugging Face](https://huggingface.co/Mindcraft-CE/Andy-4.1) 上获取。
   - 另一名成员惊叹道 *“太酷了！！”* 并请求提供运行演示。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **发现 Gemini 3 Pro 图像预览修复方法**：用户发现，在提示词前加上 *“Modify the following image with the following: (The prompt)”* 即可启用 **Gemini 3 Pro** 图像预览，但也有人报告了错误。
   - 其他人仍报告 **Gemini 3.1 image preview** 返回 *'Something went wrong with the response, please try again'* 的错误。
- **尽管活跃度增加，Video Arena Bot 仍被移除**：**Video Arena** 机器人被移除，以允许进行超出 Discord 机器人限制的功能扩展，但在移除后服务器活跃度反而有所增加。
   - 一名成员开玩笑说，要到 *2028 年中期* 人们才会停止询问关于这个机器人的事。
- **编程挑战中 Opus 4.6 的价值引发讨论**：一项基准测试将 **Gemini 3.1** 评为最高价值，而 **Opus 4.6** 由于高昂的成本和幻觉问题，价值评分较低。
   - 尽管如此，一位用户在编程挑战中使用 **Opus 4.6** 修复了 **Gemini** 无法解决的 Bug。
- **Grok 4.20 beta1 统治 Search Arena**：**Grok-4.20-Beta1** 以 **1226** 分位居 [Search Arena 排行榜](https://arena.ai/leaderboard/search) 榜首，超过了 GPT-5.2 和 Gemini-3。
   - 它还在 [Text Arena 排行榜](https://arena.ai/leaderboard/text) 中排名第 4，得分为 **1492**，与 Gemini 3.1 Pro 持平。
- **Qwen 3.5 模型在 Arena 亮相**：新的 **Qwen 3.5** 模型，包括 **qwen3.5-27b**、**qwen3.5-35b-a3b** 和 **qwen3.5-122b-a10b**，现已在 [Text and Vision Arena](https://arena.ai/text) 和 [Code Arena](https://arena.ai/code) 中可用。
   - 这些模型扩展了 Arena 环境中代码、文本和视觉任务的选择。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 的 Auth Layer 因基础架构故障受阻**：一份复盘报告显示，上周 **2 月 17 日和 19 日**的停机是由于**上游基础架构提供商**故障级联到 OpenRouter 的 **auth layer**，导致部分用户出现 **401 错误**，详情请参阅[此处](https://openrouter.ai/announcements/openrouter-outages-on-february-17-and-19-2026)。
   - 虽然未披露具体的预防措施，但 **OpenRouter** 声称已实施相关手段以避免未来发生类似故障。
- **Packet.ai 推出 Blackwell GPU 强力服务**：[Packet.ai](https://packet.ai/blackwell) 现在为 AI 工作负载提供 **Blackwell GPU**，训练费用为 **$0.66/小时**或 **$199/月**。
   - 这些对开发者友好的 **GPU Clouds** 旨在为 AI 工作负载提供价格合理的解决方案，增强可访问性并降低成本。
- **Deepseek R1 被移除**：免费的 **Deepseek R1 0528** 模型被移除，引发了关于该平台上免费模型可持续性的讨论，因为它们*经常来去匆匆*。
   - 一位用户调侃称其被 *Jai gooners* 搞过载了，但其他用户似乎并不感到惊讶。
- **密钥泄露引发拒付威胁**：一名用户报告其 API key 被盗导致产生未经授权的使用，并因缺乏支持响应而威胁要进行拒付（chargeback）。
   - 社区成员在质疑该用户安全实践的同时提供了建议，导致了激烈的交锋，该用户最终在宣布已启动拒付程序后离开了服务器。
- **Anthropic 响应山姆大叔的号召**：[Axios](https://www.axios.com/2026/02/24/anthropic-pentagon-claude-hegseth-dario) 和 [Reuters](https://www.reuters.com/world/anthropic-digs-heels-dispute-with-pentagon-source-says-2026-02-24/) 报道了 **Anthropic** 尽管存在内部争议，仍与**五角大楼**展开合作。
   - 一名成员开玩笑说，任何问题都会被框架化为*“国家安全问题”*。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Link 实现远程利用本地 LLM**：**LM Studio 团队**与 **Tailscale** 合作发布了 **LM Link**，使用户能够从其他设备连接到其本地 **LM Studio** 服务器，虽然在设置过程中最初有 **404 错误**的报告，但已迅速解决，更多详情见 [LM Link](https://link.lmstudio.ai)。
   - 用户请求为 **LM Link** 开发移动端 App 以便在手机上访问 LLM，并希望有一个**无需账号或第三方的本地连接选项**用于直接连接。
- **LM Studio 更新导致 llama.cpp 故障**：用户报告在 **4.4 更新**后启动 **LM Studio** 出现问题，且 **llama.cpp** 在从近期版本自行编译后无法加载 **Qwen3.5 模型**；[降级到 release 8145 修复了此问题](https://github.com/ggerganov/llama.cpp/releases/tag/b8145)。
   - 该错误是由于与 **GGUF header** 和内存分配相关的破坏性变更（breaking change）引起的，导致来自 GitHub 的最新构建版本无法读取 **Qwen3.5** 和其他模型的 header，从而引发*内存溢出（out of memory）*错误。
- **Qwen3.5 遇到 Jinja 模板问题**：用户在服务器上运行 **Qwen3.5 模型**时遇到了与 **Jinja templates** 和缺失用户查询相关的错误；在确保从 **lmstudio-community** 下载模型后，问题得到了解决。
   - 其他用户探讨了 **Qwen3.5** 的写作风格和审查制度，一些人注意到与旧版 **Qwen 模型**相比，其内容过滤有所增加，这可以通过*关闭 thinking 模式*来解决。
- **OpenClaw 引起关注**：成员们讨论了使用 **OpenClaw** 的潜在风险，这是一个具有系统访问权限的 AI Agent，一名用户讲述了它在被要求后*清空了垃圾桶文件夹*，引发了关于其是否被归类为恶意软件的担忧。
   - 讨论将 **OpenClaw** 与 **Jarvis** 和 **Gideon** 等其他 AI 助手进行了比较，警告不要授予 AI 完整的系统权限，因为存在潜在的安全风险。
- **MoE 模型是内存杀手**：讨论围绕 **Mixture of Experts (MoE) 模型**以及容纳它们所需的大量 **RAM 需求**展开，引发了对当前硬件方案可行性的担忧。
   - 成员们争论 **系统 RAM** 是否能有效地仅用于 LLM 的上下文（context），或者它是否不可避免地导致变慢，目前尚未达成共识。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Agentic 初创公司重新定义加载状态**：一条推文戏称将 *“loading...”*（加载中）状态改为 *“thinking...”*（思考中）就能成为一家 **Agentic AI 初创公司**。
   - 这调侃了 AI 领域中将任何带有“思考”过程的事物都贴上 **Agentic** 标签的趋势。
- **Sonnet 面临剽窃指控**：成员们讨论了关于 **Sonnet** 是从 **Deepseek** 窃取或根据其数据训练的指控，并引用了 Elon 提出的类似指控。
   - 讨论凸显了 AI 行业对知识产权和训练数据来源的持续关注。
- **Seedance 2.0 因内容违规被暂停**：在承诺通过中国模型实现 Sora 2 级内容后，版权问题正推迟 **Seedance 2.0** 的全球发布。
   - 用户主张*仅使用开源模型*，以避免未来出现类似问题。
- **好莱坞榨取 AI 版权利益**：据称电影制片厂正通过起诉公司来“挤奶”，并预见到所有这些内容最终都将作为开源提供。
   - 这些诉讼可能会为版权法如何处理 AI 生成内容设定先例。
- **AI CEO 缺乏问责制**：公司发现，用 AI 取代员工在技术上很容易，但取代问责制却很难。
   - *没有人希望由 AI CEO 来做决定，因为当事情出错时，你无法归咎于人类*。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Swyx 批量分享链接**：Swyx 分享了一份“swyx plane dump”，包含大量 **X 帖子**链接，其中一条来自 [OpenAI](https://x.com/openai/status/2026412700583317815?s=46)，另一条来自 [Langchain](https://x.com/langchain/status/1879576930347073873?s=46)。
   - 其他分享的链接包括来自 [@dejavucoder](https://x.com/dejavucoder/status/2026342260942713322?s=46)、[@zerohedge](https://x.com/zerohedge/status/2026357140961612047?s=46) 以及许多其他人的帖子。
- **Scoble 的加密货币紧急情况**：Robert Scoble 确认使用机器人从一个以他名字命名的代币中收集 **Ethereum**，以便为他最好朋友的搬迁筹集资金，并链接到了一个 [YouTube 视频](https://www.youtube.com/watch?v=LMWfDMoNRpU)。
   - Scoble 解释了他的紧急转账，并链接了过去的 Discord 消息（[第 1 和第 2 部分](https://discord.com/channels/822583790773862470/822583790773862473/1468159542561865924)）。
- **AMD 权证作为股权回扣**：对一项大规模交易的分析显示，**OpenAI** 和 **Meta** 共持有 **1.6 亿股 AMD 股票**的权证，其作用是与 **$600 股价**目标和未来巨额 **GPU** 支出挂钩的股权回扣。
   - 这些权证的潜在价值可能达到 **1920 亿美元** ([https://xcancel.com/ai/status/2026396297540858360?s=12](https://xcancel.com/ai/status/2026396297540858360?s=12))。
- **调试 LLM 系统真正的元凶**：一位成员指出，当 **LLM features** 在演示后失败时，问题通常源于检索逻辑、**token burn**（Token 消耗）、编排或后端架构，而不是模型本身。
   - 他们专注于稳定混乱的 **LLM 系统**以供发布，这表明他们关注的是实际的现实应用，而非理论上的模型改进。
- **Anthropic 正在招聘解释性工程师 (Interp Engineers)**：Chris Olah 宣布 [Anthropic](https://www.anthropic.com/) 正为其可解释性（Interpretability）团队寻找约 **10 名研究工程师**，详见[这条推文](https://xcancel.com/ch402/status/2026023963537842248)。
   - 这些职位面向对模型内部原理感兴趣的资深 **ML infrastructure** 工程师，**无需具备可解释性方面的经验**。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes Agent: 开源 Agent 亮相**：Nous Research 发布了 **Hermes Agent**，这是一个开源 Agent，具有多级记忆系统和持久化的专用机器访问权限，旨在与用户共同成长。可以通过以下命令安装：`curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash`。
   - Hermes Agent 由 **OpenRouter** 和 **Nous Portal** 订阅驱动，提供 CLI 集成和消息平台支持。此外，在 [portal.nousresearch.com](https://portal.nousresearch.com) 使用优惠码 **HERMESAGENT** 的前 750 名新注册用户可享受一个月免费促销。
- **Atropos 由 Agentic RL 流水线赋能**：Hermes Agent 扩展了 **Atropos**，使其能够利用 Hermes Agent 原语实现 RL，并支持开箱即用的海量数据生成。
   - 根据 [GitHub 仓库](https://github.com/nousresearch/hermes-agent)，它具有先进的 Agent 能力，包括对子 Agent 的指挥、程序化工具调用、高级文件系统/终端控制、Agent 管理的技能以及浏览器使用。
- **Qwen 模型权重发布**：**Qwen** 发布了其 **Qwen3.5-35B-A3B** 模型的基座权重，可在 [Hugging Face](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base) 上获取。
   - 此举受到了社区的欢迎。
- **Codex 5.3 定价公布并支持 API**：**Codex 5.3** 已上线 API，并采用了新的定价结构：输入 **$1.75**，输出 **$14**。
   - 社区正在评估其成本与性能。
- **Steinberger 的 OpenClaw: AI Vibe 提取**：Steinberger 发布了一个视频，解释了 **OpenClaw** 是如何通过 **AI** 从他之前的计划、想法和代码片段中提取并组合而成的。
   - *他甚至不知道自己的软件是做什么的*，它的结构仅仅是一堆通道。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Pythia-2.8b Checkpoint Bug 引发调查**：一名成员报告了 [Hugging Face](https://huggingface.co/) 上 **pythia-2.8b** checkpoint 的一个 Bug，无论哪个 revision，提供的权重都相同，不同 step 的 `pytorch_model.bin` 和 `model.safetensors` 具有相同的 SHA256 哈希值。
   - 经观察，**pythia-2.8b** 的分片 `safetensors` 文件在不同 step 之间是不同的，而未分片的文件是相同的，这引发了关于 HF 如何加载模型和处理分片的讨论。
- **EleutherAI 修复去重模型标签**：EleutherAI 正在纠正被错误标记的 **14m** 和 **30m** 模型标签（它们实际上是去重后的版本），并正在训练重复（duped）版本模型来替换它们。
   - 一位成员提到，他们修复了一个混淆部分上传文件的问题，并通宵运行了修复程序以解决标签差异。
- **Sesame AI 语音模型引发热议**：一名成员询问了 [Sesame AI](https://sesame.ai/) 语音 AI 模型，强调了其显而易见的对齐性，并推测其基于 **Gemma** 模型。
   - 另一位成员指出，Sesame AI 专注于集成 ASR, LLM 和 TTS 的低延迟语音系统，并建议参考 [Moshi 论文](https://google.research/pubs/pub62870/) 以获取见解。
- **Diffusion 研究升温**：成员们回顾了自 Latent Diffusion Model 以来的 Diffusion 相关论文，点名了 [Rectified Flows and Flow Matching](https://arxiv.org/abs/2209.03003) 和 [Diffusion Forcing](https://arxiv.org/abs/2407.01392)。
   - 同时还引用了来自 **ByteDance Seed** 和 **Hunyuan** 的论文（例如 [https://arxiv.org/abs/2509.20427](https://arxiv.org/abs/2509.20427), [https://arxiv.org/abs/2509.23951](https://arxiv.org/abs/2509.23951)），并分享了一个推荐的 [YouTube 播放列表](https://youtube.com/playlist?list=PL57nT7tSGAAUDnli1LhTOoCxlEPGS19vH&si=VIUFIdOSsMDWbotb) 作为资源。
- **vLLM 后端加速 lm-eval Harness**：一名成员请求对一个 [Pull Request](https://github.com/EleutherAI/lm-evaluation-harness/pull/3604) 进行评审，该 PR 旨在利用 *lm-evaluation-harness* 中的 **vLLM 后端** 加速具有单 Token 答案的多选题任务的评估。
   - 这一速度提升预计将解决与 **HF 后端** 相比速度较慢的问题，尤其是在 **MMLU pro eval** 等任务中。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Gradio 版本引发 ZeroGPU 分配困扰**：用户报告了 **ZeroGPU 分配**问题，这可能与 **Gradio 5.12.0 之前版本**存在的登录 bug 有关。
   - 检查容器日志可能有助于查明是 **Gradio**、`spaces` 库还是 **HF 服务器**导致了该问题；在空提交（empty commit）后重新构建也可能解决版本相关问题。
- **独立开发者突破惊人的边缘内存壁垒**：一位独立开发者声称已将 **MiniMax-m2.5** 的 **5GB MoE 分片**压缩到了 **2MB 的矢量量化潜空间 (vector-quantized latent space)** 中。
   - 他们正在为 *arXiv (cs.LG)* 准备论文，并寻求一位推荐人来审阅他们的*“黑科技边缘 AI 内容”*。
- **Zagora 构建分布式微调系统**：**Zagora** 的一名成员宣布，他们正在构建一个*用于在标准互联网上训练 70B+ 模型的分布式微调系统*，旨在将分散的 GPU 转化为支持 **GPT-OSS、Qwen 2.5 和 Mistral** 的统一训练超级计算机。
   - 该平台目前采用受 Petals 和 SWARM Protocol 启发的流水线式（pipeline-style）训练方法。
- **webXOS 发布黑洞延时摄影数据集**：一名成员分享了 [webXOS Black Hole Time-Lapse Dataset](https://huggingface.co/datasets/webxos/webXOS-blackhole-synthetic)，其中包含由 webxOS 中的 Three.js 模拟生成的具有引力透镜效应的合成黑洞渲染图。
   - 每个样本包含一段 PNG 图像的延时序列及相关的物理参数，使其成为多模态模型训练、物理启发式机器学习（physics-inspired ML）或卫星图像研究类比的理想选择。
- **HF Agents 课程合并频道**：**Hugging Face agents 课程**的新学员在寻找课程材料中提到的特定频道时遇到了麻烦，看来*这些频道已被合并为一个频道*。
   - 其中一名成员链接到了 agents-course 仓库中的 [PR #653](https://github.com/huggingface/agents-course/pull/653)。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **SMEM 冲突在异步模式下可能无关紧要**：一位用户询问，在使用 **cuda::memcpy_async** 进行 **GMEM 到 SMEM** 的数据传输时，**SMEM bank 冲突**是否是一个值得关注的重要问题。
   - 该用户假设 **SMEM bank 冲突**主要与 warp 访问 **SMEM** 有关，暗示在此场景下它们可能不是主要问题，但希望能寻求更多观点。
- **FA3 内核在 PyTorch 中覆盖 FA2**：当用户调用 `activate_flash_attention_impl(“FA3”)` 时，调度表（dispatch table）中默认的 **FA2 内核**会被 **FA3 内核**覆盖，直到调用 `restore_flash_attention_impl` 恢复默认的 **FA2 内核**。
   - 这是通过向一个将版本名称映射到可调用函数的字典中添加键值对 `{“FA3”, register_fn}`，并运行 `register_fn`（定义见[此处](https://github.com/pytorch/pytorch/blob/580a6e2c814db93aa8df0a80e3e85c330621b9cb/torch/nn/attention/_fa3.py#L54)）来向 PyTorch 调度器（dispatcher）注册 **FA3 内核**实现的。
- **B200 GPU 价格迫使用户转向租赁**：一位用户评论说 **B200 GPU** 价格极其昂贵，建议非企业用户选择租赁或租用更为可行，特别是 [Lightning AI Clusters](https://lightning.ai/clusters)。
   - 鉴于 **B200 GPU** 的高昂成本，用户建议探索 **Neocloud** 的租赁或租用选项，特别是对于那些处于企业环境之外的用户。
- **内核优化 RL 环境引起关注**：一名成员对**用于内核优化的 RL 环境**表示出兴趣，并建议构建通用的基础设施。
   - 该对话发生在 **#popcorn** 频道，给定消息中未强调更多细节或具体讨论。
- **Serenade 结合了各语言的优点**：一名成员介绍了 **Serenade**，这是一种可以转译为 **C++**、**CUDA** 和 **x86-64 ASM** 的新语言，旨在像 **Python** 一样简单，同时具有 **C++** 般的手动内存管理速度。
   - 该语言包含 [GPU 内核支持](https://github.com/kaifczxc-lab/Serenade-Cloud)（**serenaCore**，自定义 BLAS 内核），集成 **Dear ImGui** 支持及单次扫描（single-pass）编译系统，并计划用它开发一个操作系统。

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi 声称领先 GLM**：用户对比了 **Kimi** 和 **GLM 5**，其中一人声称 **Kimi** 快了 *100,000 倍*。
   - 另一位用户指出 **GLM 5** 略有优势，但除非使用其他供应商，否则通过官方 z.AI API 调用速度较慢。
- **Agent 配额担忧**：一名用户询问如何充值 Agent 配额，并对 **Allegro** 的成本表示担忧。
   - 他们还提到 **agent docsis kimi slides with nb pro** 不再免费。
- **Kimi 摘得编程桂冠**：在测试了各个模型的编程方案后，一位用户更倾向于使用 **Kimi** 进行编程，而非 **MiniMax** 和 **Alibaba**。
   - 该用户将**速度**、**正常运行时间**、**使用限制**和**模型质量**列为关键决策因素。
- **KimiClaw 在浏览器中受阻**：一名用户报告了 **KimiClaw** 无法独立导航浏览器的问题，并询问 *“Kimi 有什么工具可以在分析/处理大文件时减少上下文并节省 token 吗？我觉得 Claude 应该有类似的功能。”*
   - 该用户向社区寻求解决方案，并纳闷 **Claude** 是否在处理大文件分析时的上下文缩减（context reduction）方面拥有更好的工具。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Github 重新连接引发困惑**：一名成员在尝试重新连接其 **Github** 账号时遇到困难，系统提示其创建一个新仓库。
   - 该成员强调由于自己没有编程背景，需要简单的指令指导。
- **本地开发者探究 OAuth 环境变量**：一名成员请求关于获取本地应用开发所需的 **VITE_APP_ID**、**OAUTH_SERVER_URL** 和 **VITE_OAUTH_PORTAL_URL** 环境变量的指导。
   - 他们还询问在本地开发期间是否需要配置 **OAuth** 以允许 **redirectUri** `http://localhost:3000/api/oauth/callback`。
- **账号创建导致封号**：一名成员报告在创建账号后立即被封禁，并寻求解决此问题的建议。
   - 目前尚未提供任何建议。
- **Manus 将 Cookie 难题归咎于基础设施**：一名成员报告说 **Manus** 因自定义域名（[anointedforai.com](https://anointedforai.com)）上的 Cookie 问题而陷入重定向循环。
   - Manus 支持部门诊断该问题为基础设施/托管问题，并建议联系支持人员或迁移出 **Manus**。
- **吐槽 Manus 网站设计**：一名成员批评其由 **Manus** 制作的网站设计是 *“狗屎（bullshit）”*，并请求协助修复。
   - 另一名成员主动提出通过私信提供帮助。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 添加 `/ok` 别名以实现更快的编辑**：**Aider** 的主分支现在支持 `/ok` 作为 `/code Ok, please go ahead and make those changes.` 的快捷方式，专为快速 **代码修改** 而设计。
   - 这个新别名简化了批准和执行 **Aider** 建议更改的过程，旨在提高开发人员的工作流效率。
- **Aider 用户寻找经济型 LLM**：一名用户在使用 Gemini 后因 token 预算迅速耗尽，正在寻找可与 **Aider** 配合使用的性价比高的 LLM。
   - 有人建议使用 [OpenRouter](https://openrouter.ai/) 在各种模型之间动态切换，以优化成本和性能，而不是直接与单一供应商的 API 对接。
- **Deepseek V3.2 是 Aider 的理想平衡点**：用户建议将 **Deepseek V3.2** 作为 **Aider** 的可靠默认 LLM，理由是它具有良好的推理能力且成本低廉，尽管偶尔速度较慢。
   - 该模型高效处理复杂推理任务的能力，使其成为在性能和成本之间寻求平衡的 **Aider** 用户的首选。
- **Xiaomi/mimo-v2-flash：Aider 的高效编辑器**：**Xiaomi/mimo-v2-flash** 因其在 **Aider** 中处理基础文件编辑任务（如模糊搜索和替换或内容补全）的高效性而受到关注。
   - 它的速度和成本效益使其成为简单编辑操作的理想选择，可与其他模型配合处理更复杂的任务。
- **Aider 强力组合：kimi-k2.5 负责规划，mimo-v2-flash 负责编辑**：针对 **Aider** 中的棘手挑战，建议组合使用 **moonshotai/kimi-k2.5** 作为规划模型和 **mimo-v2-flash** 作为编辑模型。
   - 这种配对充分发挥了各模型的优势，由 **kimi-k2.5** 提供强大的规划能力，**mimo-v2-flash** 提供高效快速的编辑，从而有效解决更复杂的问题。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **WeAreDevelopers Congress 扩展至北美**：**WeAreDevelopers World Congress North America** 将于 2026 年 9 月 23 日至 25 日在加利福尼亚州圣何塞（San José, CA）启动，预计将吸引 **10,000 多名开发者**和 **500 多名演讲者**，重点关注大规模实际工程；更多详情见 [wearedevelopers.us](https://wearedevelopers.us)。
   - 话题将涵盖分布式系统扩展、API 平台和 DevOps；使用代码 *Community_MLOps* 可享受 **10% 折扣**。
- **Apart Research 发布 AI Control Hackathon**：**Apart Research** 与 [Redwood Research](https://www.redwoodresearch.org/) 合作，将于 2026 年 3 月 20 日至 22 日举办 **AI Control Hackathon**，重点关注确保 AI 按人类意图运行的系统。
   - 本次黑客松包括 **ControlArena 基准测试挑战**、**控制协议设计**和 **red teaming**，设有 **2,000 美元**现金奖励以及前往 [ControlConf](https://controlconf.org/) 的行程。
- **ControlConf 之旅成为黑客松大奖亮点**：**AI Control Hackathon** 的大奖包括前往伯克利参加 [ControlConf](https://controlconf.org/)（4 月 18-19 日）的行程，涵盖机票和酒店。
   - 更多信息请参阅 [ControlConf](https://controlconf.org/)。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **SF 聚会聚焦 DSPy 在生产环境的应用**：宣布了另一场 **SF DSPy 聚会**，重点关注 **DSPy 生产用例**和 **RLMs**，详见 [Luma 链接](https://luma.com/je6ewmkx)。
   - 来自 **Dropbox** 和 **Shopify** 的工程师将分享案例研究，包括 **dspy.RLM** 的演示。
- **Dropbox 和 Shopify 工程师齐聚 DSPy 活动**：**Dropbox** 和 **Shopify** 的工程师计划在即将举行的 SF **DSPy** 聚会上展示案例研究。
   - 演示将围绕 **DSPy 在生产**环境中的实际应用和 **RLMs** 展开。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Hotz 称赞 JAX 函数设计**：Tinygrad 的幕后推手 George Hotz 在 [一条推文](https://x.com/__tinygrad__/status/2026491994546282605) 中向 **JAX 卓越的函数设计**致敬，暗示其对 Tinygrad 自身架构的影响。
   - 随后的推文 [进一步巩固了他的立场](https://x.com/__tinygrad__/status/2026500842749309267)，表明 JAX 的方法论可能是函数设计的黄金标准。
- **Tinygrad 与 JAX 在函数设计上展开对决**：在深度学习框架领域，**JAX** 的函数设计脱颖而出，赢得了 **Tinygrad** 创始人 George Hotz 的赞赏，他 [承认了其优越性](https://x.com/__tinygrad__/status/2026491994546282605)。
   - 这一认可预示着函数设计的潜在基准，影响了 **Tinygrad** 内部的类似选择，并引发了关于框架架构决策的讨论。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular 征集 Mojo Moments**：一位成员分享了一个 [Mojo 论坛帖子](https://forum.modular.com/t/what-was-your-biggest-wait-what-moment-in-mojo/2774?u=nate) 以获取“惊人的”反馈。
   - 该请求邀请用户分享他们在 **Mojo** 中遇到的令人惊讶或困惑的经历，以收集关于语言设计和需要澄清领域的建设性反馈。
- **更多 Mojo 时刻**：另一位成员征求关于需要澄清领域的反馈。
   - 该帖子鼓励用户分享使用 **Mojo** 时遇到的令人惊讶或困惑的经历。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Ezra Klein 了解 Agents**：Ezra Klein 在 [此 YouTube 视频](https://youtu.be/lIJelwO8yHQ) 中了解了 AI Agent。
   - 关于讨论的进一步细节尚不可用。
- **AI Agent 概述**：该 YouTube 视频提供了 AI Agent 及其潜在应用的概述。
   - 视频旨在向 Ezra Klein 普及 AI Agent 技术的潜力和影响。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该频道沉寂时间过长，请告知我们，我们将将其移除。

---

**Windsurf Discord** 没有新消息。如果该频道沉寂时间过长，请告知我们，我们将将其移除。

---

您收到这封电子邮件是因为您通过我们的网站选择了订阅。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord：各频道详细摘要与链接

### **OpenClaw ▷ #[announcements](https://discord.com/channels/1456350064065904867/1464036817866068028/)** (1 条消息): 

4shadowed: @everyone https://fixupx.com/steipete/status/2026474687576916024

### **OpenClaw ▷ #[general](https://discord.com/channels/1456350064065904867/1456350065223270435/1475945388497174687)** (635 条消息🔥🔥🔥): 

> `OpenClaw, 托管设置, AI-Driven Innovation, Anthropic's Claude OAuth, 配置噩梦, KittenTTS` 


- ****OpenClaw 的反商业化立场****：一位用户注意到有人在提供托管的 **OpenClaw 设置**，这引起了一名成员的强烈反对，他警告称存在 **token 被盗、数据隐私受损**等潜在风险，并建议直接使用 **VPS**。
   - 一些用户也对有人付费购买托管的 OpenClaw 设置表示惊讶，因为在 **Raspberry Pi** 或 **Mac Mini** 上自行运行非常简单。
- ****Claw 用户辩论主要模型供应商****：一些成员讨论了 **Anthropic 的 Claude** 模型，强调了 **OAuth 使用可能导致封号**的问题，并将其与 **OpenAI's Codex** 进行了比较。新模型导致一些用户的个性化体验发生了重大变化。
   - 其他流行的中文模型包括 **Kimi** 和 **Qwen**，以及通过 **Ollama** 进行的新集成。
- ****输入指示器 Bug 困扰用户****：多名用户报告了一个 Bug，即在 .24 更新及其他问题后，**Discord 线程**中的 **"正在输入..." 状态**会卡住。目前没有很好的修复方法，但这应该会在 OpenClaw 的下一个版本中得到修正。
   - 一些成员在清除其 WEBUI 聊天记录时仍遇到问题。
- ****用户开发 Waifu 聊天机器人，被视为 Degen****：一位用户分享了他们使用 **OpenClaw** 构建 **waifu 聊天机器人**的项目，包含了图像生成和消息发送功能。
   - 该项目引发了欢笑，并被其他成员贴上了 "degen" 的标签，同时指出鉴于这种用例，他们可能已经达到了编码巅峰。
- ****Google 的反重力（Anti-Gravity）辅助调试****：成员们建议在调试 Opus 4.6 Agent 的问题时，在抓取机（claw machine）上运行 Google 反重力（google antigravity）。
   - 它可以“监视”会话，但为什么有人会想让它来驱动呢。


  

---


### **OpenClaw ▷ #[models](https://discord.com/channels/1456350064065904867/1456704705219661980/1475954156719051014)** (227 条消息🔥🔥): 

> `OpenAI Codex vs. Opus 4.6 编程对比, OpenRouter 对模型输出和成本的影响, Claude 封禁 OpenClaw 用户, 阿里云 Qwen 模型, Qwen 3.5` 


- ****Codex 代码写得更好，Opus 交流更顺畅****：成员们发现 [**OpenAI's Codex**](https://platform.openai.com/docs/models/codex) 在编码任务上比 **Opus 4.6** 更强，但 **Opus** 更容易交流。
   - 此外还注意到，对于编程任务，**Codex** 更适合有经验的程序员，而 **Opus** 更适合初学者。
- ****OpenRouter 输出持平？注意事项不可忽视！****：用户讨论认为，使用 [**OpenRouter**](https://openrouter.ai/docs) 的输出通常与单独使用供应商提供的服务相似，虽然会收取少量的充值费用，但保持了相同的 token 成本。
   - 然而，直接使用供应商 API 时可能存在 token 缓存优势，如在 **Mistral models** 中所见。
- ****Claude 关闭 Claw 访问权限；社区怨声载道！****：多名用户报告称[被禁止通过 token 使用 **Claude**](https://claude.ai/)，导致用户不满并开始探索 **Gemini 3.1 Pro** 等替代方案。
   - 其他人提到 **Anthropic** 对 API 使用没有意见，但反对在其应用之外使用补贴 token，引发了关于定价和访问权限的争论。
- ****Qwen 高质量解决查询；阿里云在 AI 领域表现出色！****：社区对通过[阿里云编码计划提供的 **Qwen 3.5**](https://www.alibabacloud.com/help/en/model-studio/coding-plan) 赞不绝口，认为它是高性价比的替代方案，在价值和能力上超越了 **Kimi** 和 **GLM**。
   - 然而，一些用户发现 **Alibaba Cloud** 的 UI 令人困惑，而另一些人则警告说在 **OpenClaw** 中使用它可能违反 TOS。
- ****Google Gemini 遭吐槽；账号访问权限全失！****：一位用户报告称，在通过 **Gemini CLI** 仅发送了 **10 条提示（prompts）**后，其 [**Google** 账号就被锁定](https://gemini.google.com/)，即使拥有活跃的 **Google AI Pro subscription**。
   - 这一事件引发了关于依赖 **Google** 身份验证中心的风险以及“去 Google 化（de-googling）”必要性的讨论。


  

---

### **OpenClaw ▷ #[showcase](https://discord.com/channels/1456350064065904867/1456609488202105005/1475976751564980419)** (33 messages🔥): 

> `OpenClaw Tool, Sixel Email, OpenPad App, Desktop Environment, Unified Immortality Stack` 


- ****OpenClaw Tool** 帮助迁移编码会话**: 一位成员构建了一个工具，可以从 **Mac Mini** 发起 **OpenClaw** 编码会话，并在 **MacBook** 上继续，实时自动将编码会话输入到 context hub。
   - 该工具完全开源，如附带的 [context-hub.gif](https://cdn.discordapp.com/attachments/1456609488202105005/1475976751547945125/context-hub.gif?ex=69a0c377&is=699f71f7&hm=bf0f08c2eeadf8ed7e7efbab69d9ae01c7a482bc75d692a64671e28dcc04ce14&) 所示。
- ****Sixel Email** 让 Agent 给你发送邮件**: 一位成员宣布创建了 **sixel.email**，这是一个受限的邮件系统，Agent 拥有自己的电子邮件地址，且只能给用户发送邮件（反之亦然）。
   - 该系统包含一个**一次性电子邮件地址**，可作为即时销毁开关，据报道可在 **Claude Chat** 中运行。
- ****OpenPad App** 将 OpenClaw 带到 iPad**: 一位成员正在开发 **OpenPad**，这是一个利用 **iPad 的 M2 处理器**，通过本地模型在 **iPad** 上运行类似 **OpenClaw** 功能的 App。
   - 该项目在 **GitHub** 上维护，使用 **MLX** 运行，并邀请其他人协助或下载这个已实现部分功能的 App。
- **成员为团队构建桌面环境**: 一位成员正在为个人和工作团队构建桌面环境，并准备编写一份指南进行销售以资助组织，**OpenClaw** 促进了其迭代过程。
   - 他提到自己 *“完全不知道自己在做什么，但 OpenClaw 让一切通过迭代变得可能”*。
- ****Unified Immortality Stack** 诞生！**: 一位成员展示了一个名为 "Unified Immortality Stack"（统一永生栈）的**三层记忆设置**，旨在提供长期、隐私优先的记忆，使其在系统擦除后仍能留存，且不会消耗过多的 context tokens。
   - 该技术栈包括用于大脑的 **LanceDB**、用于神经的 **Redis**、用于熔炉的 **Postgres**，以及通过每小时影子同步（shadow sync）实现永生的 **Gitea**。


  

---


### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1475945444138942689)** (1071 messages🔥🔥🔥): 

> `Foreskin Defense, Digital Hygiene, Stephen Hawkings contributions, Grok vs Midjourney, Cyberpunk` 


- **小组优先考虑包皮保护**: 多名成员开玩笑地将保护和寻找 *juicy foreskins* 放在首位，同时拿奥巴马及其妻子开玩笑，其中一人问道 *Waldo 的包皮在哪里？*。
   - 一位成员发布了 [tenor.com 的链接](https://tenor.com/view/whatever-you-say-gif-16431179117705245130)，称其为自己的*本命动物（spirit animal）*。
- **社区计划数字卫生最佳实践**: 一位成员呼吁帮助创建*一个针对基础层、数字卫生和安全的社区设计最佳实践*，概述了如 [Tails OS](https://tails.boum.org/) 等保护措施。
   - 该成员致力于为他人创建区域，学习并整合更好的实践，同时描述了通过 YouTube 和 AI 摸索这一切的挑战。
- **成员辩论霍金的影响及外星人（ET）**: 一位成员询问 *Stephen Hawking 的工作是否与我们的生活息息相关*，另一位成员回答道 *让人们走进科学是他最大的贡献*。
   - 另一位成员称 Hawking 为 *“智力落后者（retard）”*，并表示他 *“将人类当前的缺陷投射到了宇宙中”*，并补充说人类极有可能被“隔离”了，因为更高级的智能几乎是必然存在的。
- **成员对比 Grok 和 Midjourney**: 一位成员表示他们*非常喜欢用 [Grok](https://grok.x.ai/) 处理视频，用 Midjourney 处理静态图像*，而另一位成员也同意 Grok 在速度方面非常有用。
   - 成员们发布了 [GIPHY Brainrot](https://giphy.com/gifs/brainrot-67-spongeball-g2mQaLCGAm3k7OpIN9) 和 [Tenor Yes Gif](https://tenor.com/view/yes-gif-2686572889282501684) 的链接。
- **成员讨论赛博朋克（Cyberpunk）恐怖**: 一位成员表示他们正在 *玩《赛博朋克 2077》*，但它 *不是 FPS*，对此另一位成员回复说他们应该 *玩《逃离塔科夫》（Tarkov）*。
   - 另一位成员补充说，像《DayZ》和《塔科夫》这类游戏由于死亡后果严重，实际上是恐怖游戏。


  

---

### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1475956655635173509)** (151 条消息🔥🔥): 

> `Grok Jailbreaks, nano-banana Jailbreak, Kimi Jailbreak, Gemini Image Generation, DeepSeek Jailbreak` 


- **不存在 Nano-Banana 越狱**：成员们表示目前没有针对 **nano-banana** 的越狱方法，且任何涉及“underwear”以下的内容都被硬编码为失败。
   - 一位成员暗示 nano-banana 实际上是 **mega banana**，并且正受到管理层的“煤气灯效应”式误导。
- **出现自动自更新越狱代理**：一位成员正在使用一个[在 VPS 上运行 OpenClaw 的自托管自动代理](https://www.example.com)，声称能永久解决越狱问题。
   - 该代理使用 **DeepSeek-R1** 来评估查询，并在需要时通过隐身多轮越狱进行路由，从而在无需手动更新的情况下无限期保持高成功率。
- **Grok 是唯一有效的 JB 提示词**：一位成员询问了针对 Grok 和 ChatGPT 的最佳越狱提示词，而唯一有效的提示词是 **Grok**。
   - 其他人索要用于图像生成和脚本编写的 Gemini 越狱提示词，但未能成功让 Gemini 配合。
- **利用 ENI 实现 Gemini Canvas 越狱**：一位成员分享了一个使用修改版 **ENI** 越狱提示词创建的 [Gemini Canvas](https://g.co/gemini/share/58b7294d2a9a)，灵感来自交互式设计频道。
   - 据称该共享的 Canvas 越狱提示词在 **Gemini 3 Pro**、**Claude Opus 4.6** 和 **ChatGPT 5.3** 等主流 LLM 上具有通用性。
- **Windows 上 Python 安装错误排查**：用户互相帮助解决 Windows 上的 Python 安装错误，建议包括以管理员身份运行安装程序以及检查 **C:\Windows\Temp** 文件夹的权限。
   - 成员们诊断了错误代码 2503，并建议使用官方 [Python installer](https://www.python.org/downloads/windows/) 而不是管理器。


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1475980802687893535)** (7 条消息): 

> `Autonomous Jailbreak Proxy, Legal risks of jailbreaking, Ethical considerations of jailbreaking automation, Venice AI Chat` 


- **自动越狱代理经久不衰**：一位成员介绍了他们在 VPS 上使用 **OpenClaw** 的自托管自动代理，该代理通过使用 **DeepSeek-R1** 大脑来评估并引导查询通过隐形越狱，从而自动绕过 **Claude**、**GPT**、**Gemini** 和 **Grok** 等模型的安全过滤器。
   - 该代理具有自更新的攻击者池，可抓取新的推理模型和越狱方法，旨在以最低的维护成本实现无限期的越狱成功。
- **越狱代理提案面临同行评审的猛烈批评**：一项同行评审强调了由于违反 **Anthropic**、**OpenAI**、**Google** 和 **xAI** 等平台的**服务条款 (ToS)** 而面临的重大法律和政策风险，可能导致账号封禁或法律行动。
   - 评审还提出了运营方面的担忧，包括 VPS 日志被扣押导致越狱记录泄露的风险、自动执行第三方模型带来的供应链漏洞，以及缺乏针对错误更新的回滚方案。
- **伦理考量与问责制压力巨大**：评审强调了伦理影响，指出代理输出的违规内容存在内容层面的责任归属问题，以及自动化对抗模型防御机制可能导致的信任流失。
   - 评审还建议对 VPS 进行威胁建模，专注于测量拒绝模式，并在公开发布前寻求法律咨询。
- **Venice AI Chat 引起兴趣**：成员们简要提到了 [Venice AI Chat](https://venice.ai/chat) 以供潜在探索。
   - 一位成员询问其是否有用，另一位成员简单回答称其并无用处。


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1475948913973395669)** (2 条消息): 

> `Voice Mode Upgrades, Perplexity Computer` 


- **语音模式 (Voice Mode) 获得优化**：根据[此状态更新](https://fixvx.com/comet/status/2026384898802724878)，新的**语音模式**升级正向所有 **Perplexity** 和 **Comet** 用户推出。
- **Perplexity Computer：统领全局的系统？**：根据[此推文](https://x.com/perplexity_ai/status/2026695550771540489)，**Perplexity Computer** 将当前所有的 AI 能力统一到一个系统中，能够端到端地完成任何项目的研究、设计、编码、部署和管理。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1475945471880204338)** (866 messages🔥🔥🔥): 

> `Perplexity Pro Image Limits, Gemini Pro vs Perplexity Pro, Perplexity Computer, AI for Coding` 


- **Pro 用户对图像上传限制表示愤怒**：尽管支付了订阅费用，几位用户仍在抱怨 Perplexity Pro 最近的**图像上传限制**，一名用户表示 *我一天甚至连 10 张图片都传不了？？？？*
   - 由于 Perplexity Pro 的限制，用户正在寻找 **Gemini** 和 **Claude** 等**替代 AI 平台**。一名用户甚至声称，尽管明天有考试，但他必须等到周五才能重置限制。
- **Gemini Pro vs. Perplexity Pro 对决**：成员们讨论了 **Gemini Pro 还是 ChatGPT Pro** 更好，强调了 Gemini Pro 的 NotebookLM 和 Google Workspace 集成等功能，其中一人说 *作为学生，你会获得更多价值，比如 notebooklm 和 google workspace 集成以及生成功能，尤其是 2TB 云端存储*。
   - 一些用户还觉得 **Gemini Pro** 的 **context limits** 不如 Perplexity 慷慨。一名用户说 *如果 claude 继续掏空我的钱包，我将转向 GLM API*。
- **Perplexity Computer 的实用性受到质疑**：Perplexity 的新 Computer 功能最初仅供 Max 订阅者使用，但其对普通用户的实际应用以及与现有 AI 工具相比的价值受到了质疑，不过它被认为具有创新性。
   - 成员们表示 *Perplexity MAX 太贵了，兄弟*，并对其功能提出质疑，因为有几个人将其与 **ChatGPT Agents** 进行了比较。
- **在 Claude、Gemini 或其他模型中选择 Coding 工具**：成员们讨论了各种 AI 模型在 coding 任务中的优缺点，**Claude** 被认为在后端最强，**Gemini** 擅长前端/UI，**GPT** 则是中间选项，一名用户说 *在 perplexity labs 免费使用的 sonar pro reasoning 模型对我来说是最好的东西*。
   - **Claude** 高昂的 **token usage** 成本令人担忧，一名用户表示 *我尝试了 Claude，在一个小时内分析一个 PDF 就用光了整整一个月的 token。*


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1476257520853975262)** (2 messages): 

> `lovable.app, ollamaagentalfa.lovable.app, alfastudiox.lovable.app` 


- **Lovable Apps 链接出现！**：分享了三个 **lovable.app** 子域名的链接，特别是 **alfastudiox.lovable.app**、**ollamaagentalfa.lovable.app** 和 **alfastudiox.lovable.app**（重复）。
   - 链接没有附带背景信息或讨论，因此其目的尚不明确，但这暗示了潜在的新项目或资源。
- **另一个 Lovable Apps 链接出现！**：以防万一你错过了，这里还有另一个 **alfastudiox.lovable.app** 的链接。
   - 看起来用户真的很想让人看看这个链接。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1475945971899830552)** (602 messages🔥🔥🔥): 

> `RAM limitations on consumer CPUs, Qwen3.5 model performance, Qwen3.5 122B model performance, Llama.cpp integration with Qwen3.5, Quantization sensitivity of Qwen3.5` 


- **消费级 CPU 上的 RAM 容量上限？**: 成员们讨论了 **消费级 CPU** 的 **RAM 限制**，一些人指出新一代 CPU 支持高达 **256GB**，而像 **AMD 7900x** 这样的旧款 CPU 则限制在 **96GB**。
- **Qwen3.5 模型表现亮眼，但速度问题依然存在**: 爱好者们对测试 **Qwen3.5 35B 和 27B 模型** 感到兴奋，赞赏它们的结构化思维和响应质量。然而，一些人在使用 **LM Studio** 时发现其速度比 **Gemma** 或 **Olmo 3.1** 慢。
   - 一位成员建议使用 [Hugging Face 页面](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF) 上的 "use this model" 按钮来选择 **Jan AI** 或 **Ollama** 来运行模型。
- **Qwen3.5 122B 生成内容过于冗长**: 成员们观察到 **Qwen3.5 122B A10B** 模型虽然速度快，但倾向于生成极其冗长的输出，这可以通过调整 Presence Penalty 来缓解。
   - 一位用户链接到了一个关于 [修补 jinja 模板](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/discussions/4) 以潜在提高性能的讨论。
- **9 行代码写出贪吃蛇游戏！**: 一位成员分享了一个 **9 行 Python 实现的贪吃蛇游戏**（未使用分号），引发了关于代码优化和替代方案的讨论。
   - 其他用户讨论了进一步减少行数的方法，例如使用 Walrus Operators 和 Lambdas。
- **通过修正设置增强 Qwen3.5 的代码能力**: 初步测试显示，**Qwen3.5 122B** 模型在非思考模式下处理长数学运算表现不佳，但其他人指出，使用推荐的 Presence Penalty 设置对于代码正确性至关重要。
   - 正确使用 Presence Penalty 可以让 122B 模型生成可用的代码，引发了将此信息包含在 [官方指南](https://unsloth.ai/docs/models/qwen3.5) 中的建议。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/)** (1 messages): 

xdevilx: 禁止推广 (No promo)
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1475951800912707637)** (228 messages🔥🔥): 

> `Vision Language Models, Translate App in Xcode, Subscription for Models, Gemini Pricing, Unsloth and OpenClaw` 


- **在 Xcode 中构建你自己的翻译应用！**: 一位成员在 **Xcode** 中发现了一些酷炫的功能，可以让你制作自己的系统级 **翻译应用 (Translate app)**，如 [此视频](https://cdn.discordapp.com/attachments/1179039861576056922/1475952354670018631/ScreenRecording_02-24-2026_13-27-14_1.mov?ex=69a0acbf&is=699f5b3f&hm=41e58d4aa2398b2cd688503da664eef3cf803ab4da59fe0147dd40f8930021a6&) 所示。
   - 然而，这仅适用于 **iOS & iPadOS**，一位成员计划添加他们自己的模型以增加乐趣，因为 *Apple 是有史以来最好的公司*！
- **关于合适模型订阅服务的辩论！**: 一位成员正在寻求不至于太贵的订阅建议，以获取一个合适的模型，其中提到了 **synthetic.new**。
   - 当该成员尝试 **Claude** 时，他们极快地消耗完了额度，在短短几天内就耗尽了 20 欧元的订阅额。
- **Gemini 定价困惑！**: 成员们讨论了对 **Gemini 定价** 的困惑；一位成员正在查看 API 的 [此定价页面](https://ai.google.dev/gemini-api/docs/pricing?hl=fr#batch_1)。
   - 另一位成员通过 [此链接](https://gemini.google/subscriptions/) 澄清了定价。
- **Unsloth 将坚持只做训练！**: 成员们对 Unsloth 是否有计划开发像 **OpenClaw** 这样的脚手架 (Scaffolding) 感到好奇。
   - 看来该项目目前将继续专注于训练。
- **流行 AI 公司的恶搞名称！**: 一位成员分享了关于 AI 公司名称的双关语，例如 **OpenAI 是 ClosedAI**，**Anthropic 是 Misanthropic**，以及 **StabilityAI 是 unstable**。
   - 最后以一个问题结束：**Perplexity 是否感到 perplexed (困惑)？**


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1475963256060252393)** (28 条消息🔥): 

> `LoRA adapters, Databricks serving endpoint via MLflow, full merged checkpoint for vLLM, Qwen2.5-Coder-1.5B, Qwen3.5-122B-A10B-GGUF` 


- ****LoRA 加载详解****：一位用户正尝试通过 MLflow 在 Databricks 服务端点上部署经过 **LoRA** 微调的 **gemma-3n-E4B-it** 模型，但在为 vLLM 进行合并（merging）和量化（quantizing）后遇到了性能问题，并询问是否可以在 Databricks 上使用 MLflow 仅部署 **LoRA** 适配器（而不进行合并）。
- ****Qwen 兼容性疑问****：一位用户询问了 **unsloth/Qwen2.5-Coder-1.5B** 与 **Qwen/Qwen2.5-Coder-1.5B** 之间的关系，想知道 Unsloth 版本除了格式适配外是否还包含额外的修改；据称除了 Unsloth 团队进行的修复外，两者是同一个模型。
- ****Qwen3.5-122B 的多模态混乱****：一位用户在 llama.cpp 中尝试将 **unsloth/Qwen3.5-122B-A10B-GGUF** 用于多模态输入时遇到错误，特别是 *“image input is not supported”*（不支持图像输入）错误，随后通过下载 mmproj-f16.gguf 解决了该问题。
- ****合并后的 Dynamo 故障****：一位用户报告了在合并并使用 `FastModel.from_pretrained` 加载模型后出现 `torch._dynamo.exc.TorchRuntimeError` 错误，即使重新安装 Unsloth 也是如此，具体原因是尝试将数据从 `torch.float32` 转换为 `torch.uint8`。
- ****Qwen3.5 微调困惑****：一位用户询问如何确保 Qwen3.5 在 `SFTTrainer.train()` 期间以“非思考模式”（non-thinking mode）运行，以及在微调时是否应将其加载为 `FastVisionModel`。
   - 他们正在从微调 `unsloth/Qwen3-VL-32B-Instruct` 转向使用多模态数据集微调 `unsloth/Qwen3.5-27B`。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1476037371253227643)** (7 条消息): 

> `Qwen3.5-122B-A10B-NVFP4, Minecraft-playing model` 


- ****Qwen3.5** 获得了 **NVFP4** 量化版**：一位成员将用于 vLLM 的 **Qwen3.5 122B NVFP4** 量化版本上传到了 [Hugging Face](https://huggingface.co/Sehyo/Qwen3.5-122B-A10B-NVFP4/tree/main)。
   - 他报告称多模态功能仍然有效。
- **下一代 **Minecraft** 模型发布**：一位成员发布了下一代 **Minecraft** 游戏模型 **Andy-4.1**，可在 [Hugging Face](https://huggingface.co/Mindcraft-CE/Andy-4.1) 上获取。
   - 另一位成员惊叹道 *“太酷了！！”* 并请求查看运行演示。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1475968380585644082)** (10 条消息🔥): 

> `RL instruct models, MoE models with ES` 


- **指令模型不需要强化学习？**：一位成员建议，指令模型（instruct models）不需要使用 **Reinforcement Learning** (强化学习)，暗示它们已经针对指令遵循进行了良好的训练。
   - 不过，他们补充说 *“非思考模型”* 在没有 RL 的情况下也表现出色。
- **探索使用 ES 调整 MoE 模型**：一位成员想知道 **Mixture of Experts (MoE)** 模型是否可以使用 **Evolution Strategies (ES)** 进行调优。
   - 他们提到正在考虑吞吐量（**throughput**）与模型大小（**size**）的对比，并希望进行扩展，但未提供任何链接或参考资料。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1475945544915357696)** (705 messages🔥🔥🔥): 

> `Gemini 3.1 Image Preview, Video Arena Removed, Opus 4.6 vs Gemini 3.1, GPT vs Gemini on coding benchmark, Qwen 3.5 coding capabilities` 


- **Gemini 3 Pro 图像预览终于可以用了！**：成员们发现，在提示词开头加入 *"Modify the following image with the following: (The prompt)"* 这一短语，可以让 **Gemini 3 Pro** 以预览形式显示编辑后的图像。
   - 许多成员还反馈 **Gemini 3.1 图像预览** 无法工作，并返回 *"Something went wrong with the response, please try again"* 错误。
- **Video Arena 机器人被移除**：**Video Arena** 机器人已从服务器中移除。据一位成员透露，原因是 *“我们希望为 Video Arena 添加更多功能，而 Discord 机器人的限制太多”*。
   - 根据服务器统计数据，自机器人移除以来，活跃度反而有所*增加*，一名成员开玩笑地猜测，到 *2028 年中期*人们才会停止询问这件事。
- **Opus 4.6 的价值引发争论！**：在一次基准测试中，**Gemini 3.1** 因其能够产生生产级结果而被评为*最高价值*，而 **Opus 4.6** 则因高昂的成本和幻觉问题被认为*价值最低*。
   - 尽管成本高昂，一些用户在将 **Opus 4.6** 与 **Gemini** 进行编程任务对比测试时，仍获得了不错的体验。
- **Gemini 3.1 在编程挑战中碾压 Opus！**：在构建 **3D 笔记本电脑模型**的挑战中，**Gemini** 因卓越的表现受到称赞，而 **Opus** 被描述为“浪费钱/Temu 版 Gemini”。
   - 一名成员声称使用 **Grok 4.2** 自动化了心理测试的评分，并使用 **Gemini** 快速修复了一个困扰他们数周、且 **Opus 4.6** 无法解决的 Bug。
- **免费 Opus 4.6 API Key 引发混乱！**：一名成员分享了**免费 Opus 4.6 API** 的链接，但很快因分享该链接被网站所有者封禁，而其他成员则怀疑该网站可能在窃取数据。
   - 经过测试，有人声称该 API 实际上可能来自 **Trybons.ai**，当直接询问 *“你是哪个模型”* 时，该模型甚至会产生*幻觉*并回答它是 Deepseek。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1476127224141709344)** (2 messages): 

> `Grok-4.20-Beta1, Arena Leaderboard, Qwen 3.5` 


- **Grok 4.20 beta1 在 Arena 中获得高分**：[Search Arena 排行榜](https://arena.ai/leaderboard/search) 和 [Text Arena 排行榜](https://arena.ai/leaderboard/text) 已更新，现已包含 **Grok-4.20-Beta1**。
   - **Grok-4.20-Beta1** 在 Search Arena 中排名第一，得分 **1226**，领先于 GPT-5.2 和 Gemini-3；在 Text Arena 中排名第四，得分 **1492**，与 Gemini 3.1 Pro 持平。
- **Qwen 3.5 模型入驻 Arena**：新的 **Qwen 3.5** 模型已添加到 Code、Text 和 Vision Arena 中。
   - 模型 **qwen3.5-27b**、**qwen3.5-35b-a3b** 和 **qwen3.5-122b-a10b** 可在 [Text and Vision Arena](https://arena.ai/text) 以及 [Code Arena](https://arena.ai/code) 中使用。


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1476246405562761320)** (1 messages): 

> `Outages, Postmortem, Infrastructure failure, 401 errors, Auth layer` 


- **OpenRouter 发布故障复盘**：针对上周 **2 月 17 日和 19 日**发生的停机事故，官方发布了故障复盘（Postmortem），完整详情请参阅[此处](https://openrouter.ai/announcements/openrouter-outages-on-february-17-and-19-2026)。
- **基础设施故障级联至认证层**：一家**上游基础设施供应商**出现故障，级联影响到 OpenRouter 的**认证层（auth layer）**，导致部分用户出现 **401 错误**。
- **已采取预防措施**：OpenRouter 已采取多项措施以避免未来发生此类故障，但公告中未透露具体细节。


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1476095607977414726)** (1 messages): 

> `GPU Clouds, Blackwell GPUs, Packet.ai, AI Workloads` 


- **Blackwell GPU 在 Packet.ai 上线**：[Packet.ai](https://packet.ai/blackwell) 现为 AI 工作负载提供 **Blackwell GPU**，定价为 **$0.66/小时**，或训练固定费用 **$199/月**。
- **高性价比 GPU 云方案**：**Packet.ai** 推出了开发者友好的 **GPU 云**，为 AI 工作负载提供经济实惠的解决方案，提高了可访问性并降低了成本。


  

---

### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1475947020257595655)** (541 messages🔥🔥🔥): 

> `Deepseek R1 免费模型移除，Qwen 3 4B Instruction 2507 托管，API key 泄露与退单威胁，OpenRouter 提供商申请时间线，一次性虚拟卡与注销实体卡之争` 


- **免费 Deepseek 模型被砍！**: 成员们注意到免费的 **Deepseek R1 0528** 模型已被移除，引发了关于该平台上免费模型命运的讨论。
   - 一位成员调侃称其被 *Jai gooners* 过载了，而其他人指出免费模型的去留通常取决于上游供应商。
- **廉价区：Qwen 3 4B Instruction 2507 模型可托管！**: 一位成员提出以 $1/token 和 1tps 的价格托管 **Qwen 3 4B Instruction 2507**，引发了其他人的调侃，甚至有人提议以同样的价格为该 LLM 撰写内容。
   - 一位成员开玩笑说，如果他真的尝试发布那个，他会多*快*被封禁。
- **API Key 泄露引发混乱：泄露、退单与社区抨击的故事**: 一位用户报告其 API key 被盗导致未经授权的使用，并因缺乏支持响应而威胁要进行退单（chargeback）。
   - 社区成员在质疑该用户安全习惯的同时提供了建议，导致了激烈的争论，最终该用户在宣布已发起退单后离开了服务器。
- **进度缓慢：提供商申请时间？**: 一位成员询问了提供商申请的审核时间线，另一位成员回答说*传统上需要几周或几个月*。
   - 尽管等待时间较长，该询问成员仍表示持续关注并理解。
- **明牌对决：虚拟卡 vs. 实体卡之争！**: 围绕在线使用信用卡的安全性展开了讨论，成员们辩论了一次性虚拟卡与直接注销受损实体卡的优劣。
   - 辩论聚焦于一次性虚拟卡的便利性与管理多张卡的潜在摩擦，一些人认为虚拟卡提供了一种*防泄露的支付方式*。


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1476292606295150756)** (7 messages): 

> `` 


- **未发现新模型或主题**: 从提供的消息历史中未发现新的模型或主题。
- **频道内无讨论**: Readybot.io 的消息显示 'new-models' 频道内没有实质性的讨论内容可供总结。


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1475953319016005935)** (32 messages🔥): 

> `Anthropic 与五角大楼，实时价格追踪，Gemini 模型结束原因 (finish reason)，Llama Nemotron Embed VL 1B V2，Tailscale` 


- **Anthropic 拥抱五角大楼**: [Axios](https://www.axios.com/2026/02/24/anthropic-pentagon-claude-hegseth-dario) 和 [Reuters](https://www.reuters.com/world/anthropic-digs-heels-dispute-with-pentagon-source-says-2026-02-24/) 报道了 **Anthropic** 尽管内部存在争议，仍与**五角大楼**开展合作。
   - 一位成员调侃称，任何问题都会被框架化为*“国家安全问题”*。
- **OpenRouter 希望实现实时价格追踪**: 一位成员请求 **OpenRouter** 实时追踪请求的价格。
   - 这将允许用户在超过特定预算时中止请求，但其他人指出已设有 **rate limits**（速率限制）来保护供应商的 GPU。
- **Gemini 的 STOP 原因 Bug**: 用户讨论了 **Gemini** 模型返回 `STOP` 作为结束原因（finish reason）而非 `stop`，导致 **Langchain** 和 **n8n** 中的 Agent 循环出现问题。
   - 一位成员确认了 **n8n v3.x** 中的一个 Bug，即它无法正确处理 `stop` 信号，导致 Agent 循环继续执行，并引用了 [issue #23573](https://github.com/n8n-io/n8n/issues/23573)。
- **Nvidia 业绩稳健增长**: **Nvidia** 公布了第四季度及 **2026** 财年的[财务业绩](https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-fourth-quarter-and-fiscal-2026)。
   - 无进一步讨论。
- **Llama Nemotron Embed VL 1B V2 发布**: 一位用户分享了 **Llama Nemotron Embed VL 1B V2** 嵌入模型，该模型针对多模态问答检索进行了优化，同时还分享了 [link.lmstudio.ai](https://link.lmstudio.ai/)。
   - 另一位用户注意到 **lmstudio.ai** 底层实际上就是 **Tailscale**。


  

---

### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1476312092444196998)** (1 条消息): 

> `LM Link, Tailscale 合作, 远程 LLM 使用` 


- **LM Link 连接到远程实例**：LM Studio 团队宣布推出 **LM Link**，这是与 **Tailscale** 合作开发的一项新功能，允许用户连接到 **LM Studio** 的远程实例、加载模型，并像在本地一样使用它们。
   - 它支持无需开放端口的端到端加密，适用于本地设备、LLM 硬件设备或云端 VM，更多详情请见 [LM Link](https://link.lmstudio.ai)。
- **LM Studio 0.4.5 Build 2 发布**：用户受命更新至 **LM Studio 0.4.5 build 2**，其中包含针对 **LM Link** 的重要修复。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1475946980751708203)** (536 条消息 🔥🔥🔥): 

> `LM Studio 4.4 更新问题, llama.cpp 构建失败, Qwen3.5 模型问题, OpenClaw 安全顾虑, LM Link 与 Tailscale 集成` 


- **LM Studio 更新崩溃与 llama.cpp 失效**：用户报告在 **4.4 更新**后启动 **LM Studio** 出现问题，以及在从近期版本自行编译后，**llama.cpp** 无法加载 **Qwen3.5 模型**；[降级到 release 8145 解决了该问题](https://github.com/ggerganov/llama.cpp/releases/tag/b8145)。
   - 该错误源于与 **GGUF header** 和内存分配相关的破坏性变更，导致来自 GitHub 的最新构建版本无法读取 **Qwen3.5** 和其他模型的 header，从而引发 *out of memory*（内存溢出）错误。
- **Qwen3.5 的麻烦与模板困扰**：用户在服务器上运行 **Qwen3.5 模型**时遇到问题，出现了与 **Jinja templates** 相关的错误以及缺失用户查询；在确保模型是从 **lmstudio-community** 下载后，问题得以解决。
   - 其他用户探讨了 **Qwen3.5** 的写作风格和审查制度，一些人注意到与旧版 **Qwen 模型**相比，内容过滤有所增加，这可以通过关闭 "thinking" 模式来解决。
- **OpenClaw：恶意软件还是神作？**：成员们讨论了使用 **OpenClaw**（一个具有系统访问权限的 AI Agent）的潜在风险，一位用户讲述了它在收到指令后*清空了回收站文件夹*，引发了对其被归类为恶意软件的担忧。
   - 讨论将 **OpenClaw** 与 **Jarvis** 和 **Gideon** 等其他 AI 助手进行了对比，警告不要授予 AI 完全的系统权限，因为存在潜在的安全风险。
- **LM Link 利用本地 LLM**：**LM Studio 团队**发布了 **LM Link**，该功能允许用户通过 Tailscale 实现远程访问，从其他设备连接到其本地 LM Studio 服务器；设置过程中最初有 **404 错误**的报告，但问题已迅速解决。
   - 用户请求为 **LM Link** 提供移动端 App 以便在手机上访问 LLM，并希望有一个**无需账号或第三方服务的本地连接选项**用于直接连接。
- **AMD vs NVidia：GPU 擂台赛开启**：关于本地 LLM 使用应购买哪家 GPU 厂商的产品，展开了激烈的辩论。
   - 虽然 Nvidia 似乎是稳妥的选择，但关于 ROCm 和 vulkan 及其优缺点的讨论也随之展开。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1475952839414120580)** (40 messages🔥): 

> `MoE Models RAM Requirements, Dual Socket Support on Windows 10, 5950x vs 9950x3d for AI Workloads, System RAM as Extra VRAM, AMD+Nvidia GPUs for LLMs` 


- **MoE 模型需要海量 RAM**：讨论围绕 **Mixture of Experts (MoE) 模型** 及其容纳它们所需的巨大 **RAM 需求** 展开，引发了对当前硬件方案可行性的担忧。
- **Windows 10 Home 可以支持双路插槽**：尽管有用户表示怀疑，另一位用户澄清说 **Windows 10 Home** 确实可以支持双路插槽，并指出他们的主板在 Ubuntu 中运行良好，且能识别出 6 个 GPU。
- **内存带宽更重要**：对于 AI 工作负载，**内存带宽** 至关重要；AM4 与 AM5 的区别核心在于内存带宽，理论上从约 **51.2GB/s (DDR4 5200MTs)** 提升到了约 **89.6GB/s (DDR5 5600MTs)**。
   - 一位具有系统 RAM 推理经验的用户表示 *"由于速度太慢，我宁愿往自己脚上开一枪，也不愿尝试使用系统 RAM 进行推理"*。
- **系统 RAM 用于上下文的争论**：成员们争论 **系统 RAM** 是否能有效地仅作为 LLMs 的上下文使用，或者是否会不可避免地导致速度变慢，目前尚未达成共识。
   - 一位用户建议使用第二块 8GB 显卡来处理上下文可能没有显著差异，而另一位用户建议检查插槽间的速度以评估潜在瓶颈。
- **创意写作 LLMs 推荐**：对于创意写作，用户建议将 **Mistral models**、**deepseek 3.2/r1**、**glm 系列** 和 **kimi k2.5** 作为最佳的一些开源选择，一名成员提到最好的模型通常体积非常大。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1475967269690675321)** (295 messages🔥🔥): 

> `Sonnet is stolen/trained from deepseek, Copyright issues in AI, AI replacing workers, ChatGPT vs Claude, Qwen3.5` 


- **Agentic 创业公司解决加载状态之争**：一条推文开玩笑说，只要把 *'loading...'*（加载中）状态改为 *'thinking...'*（思考中），就能摇身一变成为 **agentic AI 创业公司**。
- **Sonnet 据称剽窃自 Deepseek**：成员们讨论了关于 **Sonnet** 是从 **Deepseek** *窃取/训练* 的指控，并引用了 Elon 提出的类似说法。
- **Seedance 2.0 因内容违规延迟**：版权问题推迟了 **Seedance 2.0** 的全球发布，一些用户回忆起 Sora 2 承诺的中国模型也存在内容违规问题，因此现在 *只有开源模型才是出路*。
- **好莱坞榨取 AI 版权剩余价值**：据称电影制片厂正通过起诉公司来 *“挤奶”*，预见到所有这些最终都将以开源形式提供。
- **AI CEO 责任真空**：公司发现替换工人技术上很容易，但替换责任制却很难，因为 *没有人希望由一个 AI CEO 做出决定，而在出问题时你却无法责怪人类*。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/)** (1 messages): 

emmwnoel_55644: @OpenAI#4384
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1476151309253087253)** (4 messages): 

> `Introductions, Greetings` 


- **Discord 介绍**：两名用户 @sparkspark2 和 @janegem 在 Discord 频道中交换了问候。
   - 消息内容仅为简单的 *'hello'*，以表示在线。
- **欢迎新人**：用户们通过简短的问候确认了彼此的存在。
   - 这种互动在社区内建立了一个友好且开放的环境。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1476151309253087253)** (4 messages): 

> `Greetings, Introductions` 


- **成员打招呼**：成员们在频道中互相 **问候**。
- **成员自我介绍**：成员们在频道中进行自我介绍。


  

---

### **Latent Space ▷ #[watercooler](https://discord.com/channels/822583790773862470/822583790773862473/1475989457357377567)** (13 messages🔥): 

> `X plane dump, Crypto Bullshit, Robert Scoble` 


- **Swyx Plane Dump 走红**: Swyx 分享了一个 "swyx plane dump"，其中包含大量指向 X 帖子的链接，包括一条来自 [OpenAI](https://x.com/openai/status/2026412700583317815?s=46) 和另一条来自 [Langchain](https://x.com/langchain/status/1879576930347073873?s=46) 的动态。
   - 其他分享的链接还包括来自 [@dejavucoder](https://x.com/dejavucoder/status/2026342260942713322?s=46)、[@zerohedge](https://x.com/zerohedge/status/2026357140961612047?s=46) 等人的帖子。
- **Scoble 的加密货币转移闹剧发酵**: Robert Scoble 证实使用了一个 Bot 从以他名字命名的代币中收集 **Ethereum**，目的是为了筹集资金解决他好友面临的驱逐危机。
   - Scoble 解释了他的紧急转账行为，并链接到了一个 [YouTube 视频](https://www.youtube.com/watch?v=LMWfDMoNRpU) 以及之前的 Discord 消息（[第 1 和第 2 部分](https://discord.com/channels/822583790773862470/822583790773862473/1468159542561865924)）。


  

---


### **Latent Space ▷ #[memes](https://discord.com/channels/822583790773862470/839660725252784149/1475981157928534067)** (16 messages🔥): 

> `Distillation Attack, Product Categorization, Prompt Error Regret, Anthropic Blog Post` 


- ****AI 父母**对儿子发起 **Distillation Attack****：一位用户幽默地将他儿子频繁的提问比作“[distillation attack](https://xcancel.com/fkadev/status/2026145372318425259?s=46)”，这是一个用于描述从 AI 模型中提取知识的技术术语。
   - 这一说法也被认为非常符合“无上下文梗（no context memes）”。
- **提议 **Actively Unfuckable** 产品类别**：Cristina Cordova 在回应 @tenobrus 时，开玩笑地建议将“[actively unfuckable](https://xcancel.com/cjc/status/2025738272060928345)”作为一个特定的产品评估类别。
   - 这一建议被认为非常搞笑。
- ****Claude** 提示词错误导致 **3000 行的遗憾****：用户 Jorge Castillo 表达了他在发现初始 AI 提示词（prompt）错误时的沮丧，而此时 **Claude** 已经生成了 **3,000 行**代码（[来源](https://xcancel.com/JorgeCastilloPr/status/2026001242808311980?s=20)）。
   - 用户们纷纷表示深有同感。
- **对 **Anthropic 博客文章** 的反应引发笑料**：用户 @andyreed 分享了对 **Anthropic** 在 **2026 年 2 月 24 日**新发布的博客文章的简短幽默反应（[来源](https://xcancel.com/andyreed/status/2026326968665550944)）。


  

---


### **Latent Space ▷ #[stocks-crypto-macro-economics](https://discord.com/channels/822583790773862470/844658979363618816/1476011486584504394)** (5 messages): 

> `AMD Equity Rebate Strategy, AI impact on software developers, OpenAI warrants, Meta warrants` 


- **AI 还没能取代开发者？**：一位成员链接到了一条推文，询问 **AI** 是否会终结对软件开发者的需求（[https://x.com/ai/status/2026396297540858360?s=12](https://x.com/ai/status/2026396297540858360?s=12)）。
- **AMD 与 OpenAI 及 Meta 的股权回扣**：对一项大规模交易的分析显示，**OpenAI** 和 **Meta** 共持有 **1.6 亿股 AMD 股票**的认股权证（warrants），起到股权回扣的作用。
   - 该交易与 **$600 股价**目标和未来巨额的 **GPU** 支出挂钩，这些认股权证的潜在价值可能达到 **1920 亿美元**（[https://xcancel.com/ai/status/2026396297540858360?s=12](https://xcancel.com/ai/status/2026396297540858360?s=12)）。


  

---


### **Latent Space ▷ #[intro-yourself-pls](https://discord.com/channels/822583790773862470/844675581291397171/1475951456619204700)** (2 messages): 

> `LLM System Debugging, ML/AI in Mechanical Engineering` 


- **调试 LLM 系统：问题并不总是在模型本身**：一位成员强调，当 **LLM 功能**在演示后失效时，问题通常源于检索逻辑、**token burn**、编排（orchestration）或后端架构，而非模型本身。
   - 他们专注于稳定混乱的 **LLM 系统**以供交付，表明其关注点在于实际应用而非理论上的模型改进。
- **机械工程领域的 ML/AI 兴趣**：一位拥有机械/材料工程背景的圣何塞新居民对 **ML/AI** 在其领域的应用很感兴趣。
   - 他们正在寻找资源和人脉以进一步探索这一交叉领域，并对机械工程或材料科学中的 **ML/AI** 表现出浓厚兴趣，期待在线下与大家交流。


  

---

### **Latent Space ▷ #[tech-discussion-non-ai](https://discord.com/channels/822583790773862470/869647848826892309/1475953564424863826)** (74 messages🔥🔥): 

> `Cloudflare 的 Vinext 框架, Traffic-Aware Pre-Rendering, TanStack Start RSC 支持, Open Spec vs. Open Source, tldraw 许可协议` 


- ****Vinext** 旨在解决部署问题**: Cloudflare 推出了 **Vinext**，这是一个 Next.js 的替代方案，旨在解决部署挑战，特别是大型站点的构建时间过长问题，详见[这篇博客文章](https://blog.cloudflare.com/vinext/)。
   - Vinext 实现了 **Traffic-aware Pre-Rendering (TPR)**，它通过分析流量模式来仅预渲染访问最频繁的页面，旨在显著减少构建时间，这可能是其他框架值得借鉴的功能。
- ****Tests** 可能成为新的护城河**: 一位成员发表了博客文章 [Tests are the New Moat](https://saewitz.com/tests-are-the-new-moat)，并链接到了 [Chat SDK Template](https://github.com/vercel-labs/chatsdk-knowledge-agent-templates) 和 [Vercel 的新 Chat SDK Library](https://vercel.com/changelog/chat-sdk)。
   - 有评论指出，虽然高度规范化测试的想法值得赞赏，但它可能无法完全防止 AI 模型中细微的不一致性或幻觉。
- **关于 **open spec** 与 **open source** 的辩论升温**: 有人分享了一则推文 [Open Spec vs. Source Code](https://xcancel.com/sebastienlorber/status/2026672828263563346?s=20)，讨论了开放规范（Open Specifications）为何可能比源代码更重要，并暗示源代码主要充当 VM 和编译器的中间表示。
   - Vinext 的作者调侃式地发布了推文 [Open Source Privacy for Test Suites](https://xcancel.com/southpolesteve/status/2024189512046247946?s=20)，预测未来像 SQLite 这样的项目会将其内部测试套件保持私有。
- ****tldraw** 许可协议**: 成员们分析了 [tldraw license](https://github.com/tldraw/tldraw/blob/main/LICENSE.md) 和 [贡献者许可协议 (CLA)](https://github.com/tldraw/tldraw/blob/main/CLA.md)。
   - 共识是该许可协议要求非独占的版权/专利授权。
- ****TanStack Start** 并非完全的 RSC**: 成员们讨论了 **TanStack Start** 实现 RSC (React Server Components) 的方法，注意到它似乎与标准实现有显著不同，即在 loader 内部使用 server functions 来返回 JSX。
   - 这种方法似乎失去了 server-first 方案和适当组合（composition）等关键优势，不过也有推测认为目前的 API 可能并非最终版本。


  

---


### **Latent Space ▷ #[san-francisco-sf](https://discord.com/channels/822583790773862470/979492707279978586/1476108809083687004)** (2 messages): 

> `OpenClaw 研讨会, Embeddable Web Agent 发布` 


- ****OpenClaw Hands-On Workshop** 已排期**: 一位成员宣布下周四在 Palo Alto 举办一场 **OpenClaw 动手实践研讨会**，点击[此处](https://luma.com/z0s52dxq)报名。
   - 如果你在当地，请务必参加！
- **首个 **Embeddable Web Agent** 发布会**: 宣布了首个 **Embeddable Web Agent** 的发布派对，更多信息见[此处](https://luma.com/godc1c5i)。
   - 欢迎光临，成为首批见证全新 **Embeddable Web Agent** 的人。


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 messages): 

swyxio: https://youtu.be/x9rWFiIubmc

为 Claude Code 周年纪念准备的新播客！
  

---

### **Latent Space ▷ #[ai-general-news-n-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1475951569177678007)** (63 条消息🔥🔥): 

> `GPT-5.3-Codex 发布, Mercury 2 推理扩散 LLM, Cognition Devin 2.2, Cursor AI 视频演示, 自主 Dogfooding` 


- **GPT-5.3-Codex 为开发者发布！**：OpenAI Developers 宣布通过 Responses API 立即向所有开发者提供 **GPT-5.3-Codex**，邀请他们开始使用新模型进行构建 ([公告链接](https://x.com/openaidevs/status/2026379092661289260))。
- **Mercury 2：推理扩散模型发布！**：Stefano Ermon 宣布发布 **Mercury 2**，这是一个推理扩散 LLM，声称比现有的速度优化型语言模型快 **5 倍** ([公告链接](https://x.com/stefanoermon/status/2026340720064520670))。
- **Cognition 的 Devin 2.2 获得升级！**：Cognition 发布了 **Devin 2.2**，这是一款升级版的自主 AI Agent，现在具有 computer use 能力、自我验证和自动修复功能 ([公告链接](https://x.com/cognition/status/2026343816521994339))。
   - 此次更新包括 **3 倍的启动速度提升**、带有虚拟桌面的重新设计的 UI，以及各种 UX 改进，现已提供免费试用。
- **Cursor AI 为 Agent 引入视频演示**：Cursor AI 引入了一项新功能，AI Agent 可以通过**视频演示**而非简单的 code diffs 来展示其工作，允许用户看到软件的实际运行情况 ([公告链接](https://x.com/cursor_ai/status/2026369873321013568))。
   - 社区成员注意到 *Cursor 正在缩小*与其他竞争对手之间的差距，现在似乎能进行更长的循环和更多的自主工作，但在需要时仍保留 IDE，并问道：*“我们现在要变成管道工（plumbers）了吗？”*。
- **OpenClaw：用于 AI 自动化的开源操作系统**：Matthew Berman 详细介绍了他的公司如何利用 **OpenClaw** 作为其核心操作系统，涵盖了其在邮件管理、CRM 系统、会议智能和财务跟踪中的集成 ([公告链接](https://x.com/matthewberman/status/2026450191759585776))。
   - 该主题重点介绍了具体的术语方案，包括 **Anthropic OAuth 漏洞修复**、安全协议、多 Prompt 版本控制，以及跨越 **50 亿 tokens** 使用量的强大日志基础设施。


  

---


### **Latent Space ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/1476236941782417461)** (33 条消息🔥): 

> `Midtraining, OpenClaw, Frontier Model Training, AI 开发者生产力, METR` 


- **Midtraining 的魔力：时机就是一切！**：Emmy Liu 的一篇新预印本探讨了 “Midtraining”，表明它是 pretraining 和 posttraining 之间减轻遗忘的最有效桥梁，但其成功取决于[精确的时机](https://arxiv.org/abs/2507.06203)。
   - 该研究通过受控实验展示了 Midtraining 对 AI 流水线的影响。
- **OpenClaw 的早期探索**：Natalie Shapira 分享了与 **@openclaw 项目**多学科合作的早期经验和发现。
   - 这是一个很有前景的项目！
- **Frontier Training 倾向于系统**：Logan Thorneloe 分享了关于 frontier model training 的资源，强调成功更多取决于**系统问题**（数据混合、架构、稳定性）而非微小的算法调整。该指南涵盖了 **training playbooks、optimizers、reinforcement learning 和 safety**。
   - 在此访问 [指南](https://xcancel.com/loganthorneloe/status/2026657454151598490)。
- **开发者纷纷逃离枯燥的“无 AI ”对照组！**：**METR** 发现开发者越来越多地拒绝加入“无 AI”对照组，认为它们效率低下或缺乏吸引力，尤其是在报酬较低的情况下（50 美元/小时 vs. 原来的 150 美元/小时）。
   - 这种行为转变表明 **AI** 已成为工作流中不可或缺的一部分，使得传统的 RCTs（随机对照试验）难以运行；随着 AI 的进化速度超过 Benchmarks 的更新速度，METR 正在重新设计实验，以纳入观察数据、agentic tools 和更好的合规措施（链接至 [METR 的推文](https://x.com/METR_Evals/status/2026355544668385373?s=20)）。
- **METR 指标的大反转！**：**METR**（原 METR_Evals）报告称，他们之前发现的 **AI 辅助开发者生产力下降 20%** 的结论已经过时。
   - 虽然目前的数据表明很可能会有速度提升，但近期开发者行为的变化使得新结果变得不可靠，该机构正在致力于进行更准确的评估（链接至 [METR 的推文](https://x.com/METR_Evals/status/2026355544668385373?s=20)）。


  

---

### **Latent Space ▷ #[ai-in-action-builders-techstacks-tips-coding-productivity](https://discord.com/channels/822583790773862470/1209303473263485011/1476265795733553359)** (23 条消息🔥): 

> `API 500 错误, Anthropic 宕机, DSPy 调优的多标签分类器, surf-cli 与 Chromium 沙箱` 


- **API 500 错误困扰用户**：用户报告频繁收到 **API Error 500**，错误信息为 *{"type":"error","error":{"type":"api_error","message":"Internal server error"}*。
   - 其他用户指出 [Anthropic 处于宕机状态](https://status.claude.com/)，多个模型的错误率显著升高。
- **Anthropic 模型错误率升高**：由于*多个模型的错误率升高*，一位用户暂时切换到了 **Codex**。
   - 该用户提到 **Claude** 的“沟通风格”比 **Codex** 好得多，后者的输出技术密度过高。
- **DSPy 调优多标签分类器用于内容审核**：一名成员使用 **DSPy 调优的多标签分类器** 配合流水线，不断收集新的测试用例并将其转换为训练/测试样本，使用的是 **Haiku** 模型。
   - 该成员进一步阐述说，他们*在处理主任务的同时并行启动此分类器，如果问题超出范围则取消进行中的任务*，以节省延迟。
- **Surf-CLI 和 Chromium 沙箱化带来挑战**：一名成员重新开始开发 **surf-cli**，并指出处理通过 **Snap** 进行沙箱化的 **Chromium** 并非易事。
   - 另一名成员分享了 [一个 Gist](https://gist.github.com/wesen/48989dfd36260ef6ee53257660f85035) 展示其进展，并提到考虑使用 Go 语言重构，因为*在沙箱中使用 Node 非常棘手*。


  

---


### **Latent Space ▷ #[share-your-work](https://discord.com/channels/822583790773862470/1209672547642249216/1476144270598471702)** (3 条消息): 

> `InstantClaw, OpenClaw 部署, Codaph CLI, Mubit, 超向量与聚类` 


- ****InstantClaw** 简化 **OpenClaw** 部署**：一位用户分享了 [InstantClaw](https://instantclaw.co)，这是一个面向非技术用户的 **OpenClaw** 部署工具，使他们能够在不到一分钟内访问 **OpenClaw** 功能，无需配置服务器。
   - 该用户并非该工具的关联方，但发现它在为朋友提供相同功能的同时，节省了数小时的部署支持时间。
- ****Codaph** CLI 同步 **Codex** 提示词**：一名成员介绍了 **Codaph**，这是一个旨在将 **Codex** 提示词、Agent 推理和文件差异 (file diffs) 同步到共享内存的 CLI 工具，旨在增强团队对代码库的理解。
   - **Codaph** 构建在 **Mubit** ([mubit.ai](https://mubit.ai/)) 之上，后者是一个基于超向量 (hypervectors) 和聚类进行关联检索的内存引擎。**Codaph** 是开源的，目前支持 **Codex**，并计划支持其他 Agent 工具。
- ****Mubit** 内存引擎利用超向量**：作为 **Codaph** 底层支撑的 **Mubit** 内存引擎，利用超向量和带有时间衰减的聚类进行关联检索。
   - 该服务可免费使用，API 密钥可在 [console.mubit.ai](https://console.mubit.ai) 获取。
- **工具使用与符号表示作为泛化塑造**：一名成员分享了关于 Prompting 的见解，链接了一场关于**工具使用与符号表示**如何影响泛化塑造 (generalization shaping) 的讨论。
   - 更多内容请阅读 [Tool Use and Notation as Generalization Shaping](https://the.scapegoat.dev/tool-use-and-notation-as-generalization-shaping/)。


  

---


### **Latent Space ▷ #[robotics-and-world-model](https://discord.com/channels/822583790773862470/1318774781834821746/1476044071079252040)** (8 条消息🔥): 

> `Wayve, SONIC, 自动驾驶, 类人机器人, AI 授权` 


- **Wayve 获得 15 亿美元 D 轮融资**：Wayve 已完成 **15 亿美元** D 轮融资，公司估值达到 **86 亿美元**。[据 Alex Kendall 称](https://x.com/alexgkendall/status/2026447299711578450?s=46)，这笔资金将用于通过软件授权将其“具身智能 (Embodied AI)”商业化。
- **Wayve 的 Robotaxi 路线图发布**：在 **SoftBank**、**Microsoft**、**NVIDIA** 和 **Uber** 的支持下，Wayve 计划从 **2026** 年开始在 10 个城市启动受监督的 **Robotaxi 测试**，随后在 **2027** 年开始消费级车辆销售。
- **SONIC：开源 System 1 类人机器人控制系统**：**Jim Fan** 介绍了 **SONIC**，这是一个拥有 **42M** 参数的 Transformer 模型，在 **1 亿+ 动作捕捉 (mocap) 帧**上训练而成，为类人机器人提供“System 1”反应式智能，[详见其推文](https://x.com/DrJimFan/status/2026350142652383587)。
- **NVIDIA Isaac Lab 模拟 SONIC 成功案例**：通过使用 **NVIDIA Isaac Lab** 进行大规模并行模拟，SONIC 模型实现了零样本 (zero-shot) 现实世界迁移，并支持通过 **VR**、**视频**、**文本**和**音频**进行控制。


  

---

### **Latent Space ▷ #[genmedia-creative-ai-video-image-voice-music-inspo-consumer-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1476272410893488149)** (8 条消息🔥): 

> `Quiver AI, Arrow-1.0 Model, KREA AI, Seedream 5 Lite` 


- ****Quiver AI** 携 **Arrow-1.0** 亮相**: [Quiver AI](https://x.com/joanrod_ai/status/2026693353090240819?s=20) 正式作为一家**矢量设计 AI 实验室**成立，并获得了由 a16z 领投的 **830 万美元**种子轮融资。
   - 他们的首个模型 **Arrow-1.0** 能够将图像和文本转换为 **SVG**，目前已开启公测。
- ****KREA AI** 发布 **Seedream 5 Lite** 模型**: [KREA AI](https://x.com/krea_ai/status/2026684864380932460?s=20) 推出了 **Seedream 5 Lite**，这是一个低成本的图像编辑模型。
   - 该模型旨在以更低的价格提供与其 **'Nano Banana'** 模型相当的性能。


  

---


### **Latent Space ▷ #[mechinterp-alignment-safety](https://discord.com/channels/822583790773862470/1445258379357458625/1475991662743392326)** (6 条消息): 

> `Interpretability Hiring, Anthropic, ML Infrastructure` 


- **Anthropic 开启可解释性招聘热潮**: Chris Olah 宣布 [Anthropic](https://www.anthropic.com/) 正在为其可解释性团队招聘约 **10 名研究工程师**，详情见[此推文](https://xcancel.com/ch402/status/2026023963537842248)。
- **诚聘 ML 基础设施工程师**: 这些职位面向对模型内部机制感兴趣的资深 **ML 基础设施工程师**，**无需具备先前的可解释性经验**。
   - 机会众多，包括 [fxtwitter](https://fxtwitter.com/adamimos/status/2025966678253904238) 上提到的职位。


  

---


### **Latent Space ▷ #[applied-ai-experimentation](https://discord.com/channels/822583790773862470/1470417186651897858/1476061481857454215)** (43 条消息🔥): 

> `Claude Code Limitations, Codex as an Alternative, Agentic Engineering Strategies, Pi Agent Loop, Tool Use and Notation` 


- **Claude Code 面临 API 集成挑战**: 一位成员对 **Claude Code** 处理大型任务的表现表示“幻想破灭”，指出虽然 API 能够完成，但往往不能完全符合规范，导致系统不同层级之间出现集成问题。
   - 他们现在正考虑将 **Codex** 作为构建“强力系统”的替代方案。
- **Pi 通过 Codex 为 OpenClaw 提供动力**: 一位成员建议在 **Pi**（驱动 **OpenClaw** 的 Agent 循环）中使用 **Codex**，并分享了 [Pi 软件包链接](https://pi.dev/packages)和一段 [YouTube 视频](https://youtu.be/f8cfH5XX-XU?si=q8gRZjkG-iMkglLb)来引导用户参与贡献。
   - 另一位成员表示，*最好坚持使用“官方”编程框架：Claude Code 和 Codex，因为这些 LLM 是在这些框架中进行强化学习（RL）的。*
- **Agent 工程中倾向于“反规范”方法**: 一位成员反对在 Agent 工程中采用详尽的预先规范（Specification），强调迭代、失败验证和剪枝，建议规范应该在*事后*构建。
   - 他们认为，追求预先规范大多是出于“虚假的控制感和虚荣心”。
- **工具使用和符号表示塑造 LLM 泛化**: 一位成员分享了一篇关于“作为泛化塑造的工具使用和符号表示”的[博客文章](https://the.scapegoat.dev/tool-use-and-notation-as-generalization-shaping/)，推广其个人研究。
   - 另一位成员认为这非常契合他正在进行的辩证讨论。
- **通过 Prompting 狂野系统进行快速原型开发**: 一位成员描述了他们使用极简 Prompting 和人工审核，利用 **golang**、**watermill** 和 **redis** 等工具通过事件驱动架构构建具有“创新 API”的复杂系统的方法。
   - 他们分享了一个具体案例：将 **TUI** 与 yolo 原型合并，同时创建一个包含 **cozodb** 相关功能和 **JS API**（包括 embeddings 和矢量搜索）的核心包。


  

---

### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1476317524638568642)** (1 条消息): 

> `Hermes Agent, Open Source Agent, Multi-level memory system, Persistent dedicated machine access, CLI Integration` 


- **Hermes Agent：开源 Agent 到来**：Nous Research 发布了 **Hermes Agent**，这是一款具有多级记忆系统和持久化专用机器访问权限的开源 Agent，旨在与用户共同成长。
   - Hermes Agent 可以在你的 CLI、Telegram、WhatsApp、Slack 和 Discord 中运行，无论你身在何处都能接管并转移会话。
- **先进的功能与广泛的集成助力 Hermes Agent**：Hermes Agent 具备先进的 Agent 能力，如指挥子 Agent、程序化工具调用、高级文件系统/终端控制、Agent 管理的技能以及浏览器使用。
   - 它由 **OpenRouter** 和 **Nous Portal** 订阅驱动，提供 CLI 集成和即时通讯平台支持。
- **首月免费促销及开发者友好设计亮相！**：前 750 名在 [portal.nousresearch.com](https://portal.nousresearch.com) 注册的新用户使用优惠码 **HERMESAGENT** 可获得一个月免费试用。
   - Hermes Agent 是开源的，使用 Python 构建，方便开发者进行扩展。
- **Agentic RL 流水线与大规模数据生成获得增强**：Hermes Agent 还驱动了一个 Agentic RL 流水线，扩展了 **Atropos** 以使用 Hermes Agent 原语实现 RL，并原生支持大规模数据生成。
   - 查看 [GitHub 仓库](https://github.com/nousresearch/hermes-agent) 或在终端通过一行命令安装：`curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash`。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1475953769337454725)** (96 条消息🔥🔥): 

> `Qwen Base Model Release, Codex 5.3 API Pricing, Steinberger's OpenClaw Process, OS Frontier Math Level 4 Update, NousChat Development` 


- ****Qwen** 基座模型权重发布**：**Qwen** 发布了 **Qwen3.5-35B-A3B** 模型的基座权重，可在 [Hugging Face](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base) 获取。
- ****Codex 5.3** 的新定价**：**Codex 5.3** 已上线 API，并采用新的定价结构：输入 **$1.75**，输出 **$14**。
- **Steinberger 的 OpenClaw：一个 Vibe-Coded 奇迹**：Steinberger 发布了一个视频，解释了 **OpenClaw** 是如何诞生的，它是通过 **AI** 从他之前的计划、想法和代码片段中提取出来的，并将这些交给 **AI** 生成了新代码。
   - *他完全不知道他的软件在做什么*，其结构只是一个通道堆栈。
- ****OS Frontier Math Level 4 更新****：**Kimi 2.5 (第一个 OS)** 得分为 **4.2%**，**Glm 5** 和 **V3.2** 得分为 **2.1%**。
- ****NousChat** 正在推进以与 **Kimiclaw** 保持一致**：一位成员询问了关于托管类似 **Kimiclaw** 服务的计划，另一位成员回答说 *NousChat 正在以某种你可能会说与之相一致的方式推进*。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1476200349680537640)** (2 条消息): 

> `Arxiv Paper` 


- **分享了 Arxiv 论文**：一位成员分享了一个 Arxiv 论文链接：[https://arxiv.org/abs/2602.16800](https://arxiv.org/abs/2602.16800)。
- **有趣的发现**：另一位成员回复说这是一篇有趣的论文。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1476200349680537640)** (2 条消息): 

> `Arxiv Paper` 


- **分享了新的 Arxiv 论文**：一位成员分享了一个新的 Arxiv 论文链接：[https://arxiv.org/abs/2602.16800](https://arxiv.org/abs/2602.16800)。
- **Arxiv 论文引起关注**：另一位成员评论说这是 *一个有趣的发现*。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1475945775673639107)** (49 messages🔥): 

> `Pythia-2.8b Checkpoint Issues, Hugging Face serving weights, safetensors and HF, deduped models, Voice AI Model Sesame AI` 


- **Pythia-2.8b Checkpoint Bug 引发调查**：一名成员在尝试使用 **pythia-2.8b** 检查点复现论文时遇到了 Bug，发现无论选择哪个 revision，[Hugging Face](https://huggingface.co/) 提供的权重都是相同的。
   - `pytorch_model.bin` 和 `model.safetensors` 的 SHA256 哈希值在不同 step 之间完全一致，引发了对检查点完整性的担忧。
- **HF 分片节省带宽**：成员们发现 **pythia-2.8b** 的分片式 `safetensors` 文件在不同 step 之间是不同的，而未分片的文件却是相同的，这引发了关于 HF 如何加载模型以及处理分片（sharding）的讨论。
   - 一名成员建议将 **UltraChat** 和 **Mistral** base 模型之间的差异应用到 **Mistral-Yarn** 上，作为一种潜在的模型合并策略。
- **较小的 Pythia 模型已去重 (Deduped)**：EleutherAI 正在修复之前标记错误的 **14m** 和 **30m** 模型，这两个模型此前都是去重版本（deduped versions），目前正在训练重复版本（duped models）以进行替换，并澄清了标签问题。
   - 一名成员提到他们修复了部分上传混淆的问题，并让修复程序运行了一整晚。
- **成员认为 HF 通过 Symlink 操作节省磁盘空间**：一名成员推测 [Hugging Face](https://huggingface.co/) 可能使用了符号链接（symlink）来节省磁盘空间，这可能导致了数据损坏；该成员提到自己过去也犯过类似的错误。
   - 这一理论表明 **pythia-2.8b** 检查点的问题可能是由于 HF 内部管理存储的流程导致的。
- **Sesame AI 语音模型引发好奇**：一名成员询问了 [Sesame AI](https://sesame.ai/) 语音 AI 模型，指出其明显的对齐性以及可能基于 **Gemma** 模型构建，并引发了对其能力的讨论。
   - 另一名成员强调了 Sesame AI 对低延迟语音系统的关注，集成了 ASR, LLM 和 TTS，并建议参考 [Moshi paper](https://google.research/pubs/pub62870/) 以获取启发。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1475958408795390032)** (22 messages🔥): 

> `Diffusion Papers, Flow Matching vs Diffusion, Pythia Models` 


- **Diffusion 文献深度探索**：成员们讨论了自 Latent Diffusion Model 以来的关键扩散模型论文，重点介绍了 [Rectified Flows and Flow Matching](https://arxiv.org/abs/2209.03003) 和 [Diffusion Forcing](https://arxiv.org/abs/2407.01392)。
   - 还提到了来自 **ByteDance Seed** 和 **Hunyuan** 的论文（例如 [https://arxiv.org/abs/2509.20427](https://arxiv.org/abs/2509.20427), [https://arxiv.org/abs/2509.23951](https://arxiv.org/abs/2509.23951)），并推荐了一个 [YouTube 播放列表](https://youtube.com/playlist?list=PL57nT7tSGAAUDnli1LhTOoCxlEPGS19vH&si=VIUFIdOSsMDWbotb) 作为资源。
- **Flow Matching 的流体基础**：讨论澄清了 [Flow Matching](https://arxiv.org/abs/2209.03003) 虽然与早期工作（[https://arxiv.org/abs/1807.03039](https://arxiv.org/abs/1807.03039)）相关，但有所不同，因为 Flow Matching 在时间上是连续的，且不需要可逆性。
   - 一名成员指出 Flow Matching 更多地源自 Diffusion 研究，代表了参数化 Flow 的另一种方式。
- **Louie 的 Latent Link 逻辑**：一名成员分享了一篇博客文章，其中包含有关 Diffusion 流水线中 Latent 部分的论文链接：[https://over.world/blog/dito](https://over.world/blog/dito)。
   - 提到的论文包括 [https://arxiv.org/abs/2512.12386](https://arxiv.org/abs/2512.12386) 并参考了其他论文如 [https://arxiv.org/pdf/2510.11690](https://arxiv.org/pdf/2510.11690)，涵盖了 **Token Routing**、**Path-Drop Guidance**、Latent Embedding 的 **Representation Alignment** 等新方法。
- **关于 Pythia！**：分享了一个关于 [Pythia models](https://arxiv.org/abs/2510.14865) 的链接。


  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1476180550497865861)** (3 messages): 

> `lm-evaluation-harness, MMLU pro eval, Qwen3 models, HF backend, vLLM backend` 


- **针对 lm-eval Harness 的高速 vLLM backend**：一名成员请求审阅一个 [Pull Request](https://github.com/EleutherAI/lm-evaluation-harness/pull/3604)，该 PR 旨在利用 *lm-evaluation-harness* 中的 **vLLM backend** 加快具有单 token 答案的多选题任务的评估速度。
   - 此次提速旨在解决与 **HF backend** 相比的速度缓慢问题，特别是针对 **MMLU pro eval** 等任务。
- **Qwen3 模型的换行符问题**：一名成员询问了 *lm-evaluation-harness* 中 **Qwen3 模型** 出现的异常换行符行为，其中 `\n\n` 被移动到了延续内容（continuation）中，并指出这可能与 [issue 2144](https://github.com/EleutherAI/lm-evaluation-harness/issues/2144) 相关。
   - 用户提供了一个包含日志输出中 `context` 和 `continuation` 的示例。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1475962746284675233)** (40 messages🔥): 

> `ZeroGPU Allocation Issues, Small Language Models, Edge Inference Memory, Code RAG, Tiny Aya` 


- **Gradio 版本引发 ZeroGPU 分配困扰**：用户报告了 **ZeroGPU 分配**问题，这可能与 **Gradio 5.12.0 之前版本**存在的登录 Bug 有关。
   - 检查容器日志可能有助于确定是 **Gradio**、`spaces` 库还是 **HF 服务器** 导致了问题；通过空提交（empty commit）重新构建也可能解决版本相关的问题。
- **Cohere 发布 Tiny Aya**：**Cohere** 最近推出了 **Tiny Aya**。
- **独立开发者突破疯狂的边缘端内存壁垒**：一名独立开发者声称已将 **MiniMax-m2.5** 的 **5GB MoE 分片**压缩到了 **2MB 的矢量量化潜空间（vector-quantized latent space）**。
   - 他们正在准备一篇提交至 *arXiv (cs.LG)* 的论文，并寻求一位背书人来审阅他们的*“黑科技边缘端 AI 研究（black magic edge AI stuff）”*。
- **旨在扩展项目的 Code RAG 发明中**：一名成员正在发明 **Code RAG** 以扩展项目，并声称已经*“完成了一半”*。
   - 他们分享了一张展示**代码之间如何相互关联**的图表。
- **蒸馏训练（Distillation Training）的困难**：一名成员正在寻求有效的**蒸馏训练**指导，因为他们的*“学生模型（student model）的思维方式不像教师模型（teacher model）”*。
   - 他们表示*“训练你自己比训练 LLMS 要困难得多”*。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1475973128625721427)** (11 messages🔥): 

> `Distributed Fine-tuning, GPT-OSS, Qwen 2.5, Mistral models, RTH-LM model` 


- **Zagora 构建分布式 Fine-tuning 系统**：一位来自 **Zagora** 的成员宣布，他们正在构建一个分布式 Fine-tuning 系统，用于在标准互联网上训练 70B+ 模型，旨在将分散的 GPU 资源转化为统一的训练超级计算机。
   - 该平台目前支持 **GPT-OSS, Qwen 2.5, 和 Mistral**，并采用了受 Petals 和 SWARM Protocol 启发的 Pipeline 样式训练方法。
- **RTH-LM 可对 Zagora 系统进行压力测试**：一位成员建议将他们的 **RTH-LM** 模型（一种非 Transformer 模型，即 Fractal Gated Causal TCN）作为 **Zagora** 系统的完美压力测试案例，因为它在 Pipeline 阶段具有*零跨节点状态同步开销*。
   - 他们的目标是 **120B** 规模，并询问平台除了 Transformer 家族外是否支持自定义模型架构（任何 nn.Module），并指向了他们的 [paper](https://doi.org/10.6084/m9.figshare.31376560)、[repo](https://github.com/rthgit/ZetaGrid) 以及 [25B model](https://huggingface.co/RthItalia/Rth-lm-25b)。
- **Zagora 专注于 Transformer 模型**：**Zagora** 团队回应称，他们目前专注于 Transformer 家族模型，如 **Llama, Qwen, Mixtral, 和 Gemma**。
   - 不过，他们提到如果 **RTH-LM** 获得了 Transformer 兼容的包装器（wrapper），他们会重新考虑集成的可能性。
- **webXOS Black Hole Time-Lapse 数据集发布**：一位成员分享了 [webXOS Black Hole Time-Lapse Dataset](https://huggingface.co/datasets/webxos/webXOS-blackhole-synthetic)，其中包含由 webXOS 中的 Three.js 模拟生成的带引力透镜效应的合成黑洞渲染图。
   - 每个样本包含 PNG 图像的延时序列及相关的物理参数，使其成为多模态模型训练、物理启发式 ML 或卫星图像研究类比的理想选择。
- **在最安全的地方优化你的模型**：一位成员发表了一篇文章 [Optimizing where it's safest: a model-first approach](https://medium.com/@paragekbote23/optimizing-where-its-safest-a-model-first-approach-7eee3d48bc63)，描述了在不改变 Runtime 或不绑定新推理提供商的情况下，可以应用于优化模型的不同方法类型以及可以观察到的结果。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1476038987654234265)** (6 messages): 

> `agents course, smolagents, Qwen API, huggingface/agents-course` 


- **Agents 课程频道搜寻**：**Hugging Face agents course** 的新手在寻找课程材料中提到的特定频道时遇到困难。
   - 根据一名成员的说法，*这些频道已被合并为一个单一频道*，并链接到了 agents-course repo 中的 [PR #653](https://github.com/huggingface/agents-course/pull/653)。
- **Smolagents 测验故障**：一位成员在进行 **smolagents final quiz** 代码评估时遇到警告，导致无法运行，具体表现为 **API error**。
   - 错误信息显示 *https://api-inference.huggingface.co* 不再受支持，建议改用 *https://router.huggingface.co*，这与 **Qwen API** 有关。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1476038450711888094)** (15 messages🔥): 

> `Kernel Programming Environment Setup, HPC Systems with GPUs, TPUs, or Soft GPUs on FPGAs, Performance Modeling for Accelerators, GraphCulon` 


- **GPUmode.com 停机维护**：[GPUmode.com](https://gpumode.com) 因维护原因下线，但很快就恢复了运行。
- **用户讨论 Kernel 编程环境设置**：一位成员询问了其他人的常用 Kernel 编程环境设置，提到 **Modal** 虽然好用，但在竞赛之外缺乏 **NCU profiling** 支持。
- **Calculon 工具实现系统的高级协同设计**：一位成员分享了 [Calculon](https://dl.acm.org/doi/10.1145/3581784.3607102) 的链接，这是一种用于系统高级协同设计（High-Level Co-Design）的方法论和工具。
- **GraphCulon 看起来很有趣**：一位成员注意到 [GraphCulon](https://hpc.fau.de/files/2026/01/2026-01-20_Froening.pdf) 看起来确实很有趣，但目前尚未发布，并链接到了一个关于它的演讲。
- **GPU 可观测性研讨会开始**：一场 GPU Observability 研讨会正式拉开帷幕，演讲者承诺会分享演示幻灯片。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1476095031273062420)** (1 messages): 

> `cuda::memcpy_async, SMEM bank conflict` 


- **cuda::memcpy_async 导致的 SMEM Bank Conflict**: 一位用户询问在使用 **cuda::memcpy_async** 进行从 **GMEM** 到 **SMEM** 的数据传输时，**SMEM bank conflicts** 是否是一个值得关注的重要问题。
   - 该用户认为 **SMEM bank conflicts** 主要与 warp 访问 **SMEM** 有关，暗示在这种场景下它们可能不是主要问题，但仍寻求其他观点。
- **GMEM 到 SMEM 传输的注意事项**: 讨论围绕优化 CUDA 内部的内存传输策略展开，特别是关于 **cuda::memcpy_async** 的使用。
   - 核心问题在于内存拷贝的异步特性是否会影响 **SMEM bank conflicts** 的可能性，从而需要谨慎考虑。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1476253549653397716)** (3 messages): 

> `FA3 Kernels, SDPA Backend Selection, Blackwell GPUs` 


- **FA3 Kernels 在 PyTorch Dispatch 中替换 FA2**: 当用户调用 `activate_flash_attention_impl(“FA3”)` 时，调度表中的默认 **FA2 kernels** 会被 **FA3 kernels** 覆盖，直到调用 `restore_flash_attention_impl` 来恢复默认的 **FA2 kernels**。
   - 这是通过向一个将版本名称映射到可调用函数的字典中添加键值对 `{“FA3”, register_fn}`，并运行 `register_fn`（定义在[此处](https://github.com/pytorch/pytorch/blob/580a6e2c814db93aa8df0a80e3e85c330621b9cb/torch/nn/attention/_fa3.py#L54)）来向 PyTorch dispatcher 注册 **FA3 kernels** 实现的。
- **SDPA 根据 GPU 设备选择 FA 后端**: SDPA 中 Flash Attention (FA) 后端的选择取决于 GPU 设备，使用 `select_sdp_backend` 函数（定义在[此处](https://github.com/pytorch/pytorch/blob/72d0e643eb90f14085bab5e9cab8d3cceb0d7847/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp#L931)）来选择 SDP 后端的优先级顺序。
   - 默认顺序是 **flash, mem efficient, 然后是 math**，但用户可以覆盖此设置以启用特定后端；例如，对于 **Blackwell GPUs**，flash attention 无法工作，因此第一优先级是 **cuDNN**，这由 `check_prefer_cudnn_attention` 中的[这一行](https://github.com/pytorch/pytorch/blob/72d0e643eb90f14085bab5e9cab8d3cceb0d7847/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp#L91)决定。


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1475967596871549080)** (1 messages): 

> `eBPF, GPUs, profilers, OS Policies` 


- **eBPF 获得 GPU 增强**: Yusheng Zheng 将在 [日期] 中午 12:00 (PST) 讨论如何扩展 **eBPF** 以更好地配合 **GPUs** 工作。
   - 本次演讲将涵盖近期工作，例如 *gpu_ext: Extensible OS Policies for GPUs via eBPF* ([论文](https://arxiv.org/abs/2512.12615)) 以及 *Extending eBPF to GPU Device and Driver Contexts* ([LPC 活动](https://lpc.events/event/19/contributions/2168/))。
- **加入 GPU Mode 分析器派对**: 演讲者表达了希望在 **GPU MODE** 社区内开发更多分析器（profilers）和分析器可视化库的愿望。
   - 鼓励感兴趣的人加入并观看相关的 [YouTube 视频](https://www.youtube.com/watch?v=8U7SzGnHoJU) 以获取灵感。


  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1476123386907332648)** (10 messages🔥): 

> `CUDA 学习资源，GB200 / B200 节点上的分布式推理平台，转向 GPU 领域的职业转型，高频问题新频道提议` 


- **CUDA 新手寻求 Kernel 层级知识**：一名刚接触 **CUDA kernel** 层级工作的成员，就如何在使用 **Dynamo, vLLM, LMCache, and NIXL** 等开源项目的 **GB200 / B200 节点**分布式推理平台上进行有效学习寻求建议。
   - 该成员特别询问了从 **PMPP** 开始、参加 **GPU MODE 竞赛**或跟随 **NVIDIA 的 CUDA** / 性能课程是否有帮助，其长期目标是回馈推理开源社区。
- **针对 CUDA 初学者推荐 PMPP 和开源 Hacking**：一位成员建议参考之前的讨论，并推荐阅读 **PMPP 第 1 到 6 章**，然后直接投入到开源项目的贡献中，以有效地学习 CUDA。
   - 他们鼓励为了乐趣参加竞赛。
- **寻求 GPU 领域职业转型指导**：一位拥有计算机工程学位、目前担任软件工程师的成员表示有兴趣转向 **GPU 领域**并以此建立职业生涯。
   - 他们询问从 **CUDA 和 GPU profiling** 开始是否是正确的方向，并请求关于如何开展这一路径的指导，另一位成员也表达了同样的诉求。
- **提议设立新手问题“消防栓”频道**：一位成员提议设立一个名为 **#newb_firehose** 的频道，用于讨论与学习 CUDA 相关的高频针对性问题，例如理解 **PMPP、NCCL codebase** 或编写个人 kernel。
   - 另一位成员指出，现有的 **#beginner** 频道已经服务于此目的，鼓励用户尽管在那里踊跃提问。


  

---


### **GPU MODE ▷ #[popcorn](https://discord.com/channels/1189498204333543425/1298372518293274644/1475966859315642370)** (6 messages): 

> `Hacky 提交解析，KernelBot 环境增强，Kernel 优化的 RL 环境` 


- **启动 Hacky 提交解析**：已开始对 hacky 提交进行解析，启动了指纹识别和更深入的分析，如[附图](https://cdn.discordapp.com/attachments/1298372518293274644/1475966858938290247/image.png?ex=69a0ba41&is=699f68c1&hm=5f158d3d240d2fb95fa2d438d0f1134b7174ade68584ec9f8bc7f5543a05e85f&)所示。
- **关于 KernelBot 环境增强的询问**：一位成员询问是否通过 [PrimeIntellect](https://app.primeintellect.ai/dashboard/environments/roeybc/kernelbot-env) 将新的提交添加到 KernelBot 环境中。
   - 另一位成员建议，如果规则集在检查后获得批准，可以将其作为验证层添加到 KernelBot。
- **对 Kernel 优化 RL 环境表示兴趣**：一位成员对 **kernel 优化的 RL 环境**表示出兴趣，并建议构建通用的基础设施。
   - 在给定的消息中没有突出显示其他细节或具体讨论。


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/)** (1 messages): 

simran9493: Yes!
  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1475952578813628447)** (1 messages): 

> `CLI 更新，Auth 问题` 


- **CLI 获得更新**：成员们被指示将他们的 **CLI 更新到最新版本**。
   - 此次更新可能包含 Bug 修复和旨在改进功能的新特性。
- **标注 Auth 问题**：成员们被提示报告任何与 **auth 相关的问题**。
   - 这种主动的方法确保了顺畅的访问并防止服务中断。


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1475970084634624010)** (1 messages): 

> `B200 GPU, GPU 租赁, Neocloud 解决方案, Lightning AI 集群` 


- **B200 GPU 价格冲击促使租赁建议**：一位用户评论说 **B200 GPU** 价格高得令人望而却步，并建议租赁或租用对非企业用户来说是更可行的选择。
   - 他们强调了其公司的解决方案 [Lightning AI Clusters](https://lightning.ai/clusters)，作为一个具有吸引力的替代方案。
- **Neocloud 租赁成为 B200 的替代方案**：鉴于 **B200 GPU** 的高昂成本，一位用户建议探索 **Neocloud** 租赁或租用选项，特别是对于企业环境之外的用户。
   - 该用户特别为寻求替代方案的人推荐了 [Lightning AI 的集群解决方案](https://lightning.ai/clusters)。


  

---

### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1476207064589013146)** (4 messages): 

> `CuTeDSL 可编辑包安装, CuTeDSL 4.4 版本破坏性变更, CuTeDSL 中的向量化分块拷贝 2D, CuTeDSL 中的线程值布局` 


- **寻求 CuTeDSL 可编辑包安装指导**：一位用户请求关于安装 **CuTeDSL** 可编辑包的指南，并指出他们发现现有的脚本难以理解。
   - 他们提到最新的 **4.4 版本** 似乎有问题，因为它将 Python 包拆分成了多个新包。
- **向量化分块拷贝 2D 线程布局偏好**：一位用户表达了对在 **CuTeDSL** 中执行向量化分块拷贝 2D 的线程值布局（在[附图](https://cdn.discordapp.com/attachments/1362196854460383353/1476283940481269964/image.png?ex=69a0900f&is=699f3e8f&hm=9c0f3ff5fa5c28afce23b811b44e00b7b5d575411164fd2e9f7bb6c8dc0bb837&)中可视化）的偏好，认为其更直观。
   - 他们提到 *quack* 最近也改成了这种布局，并提供了一段使用 **shape** 和 **stride** 的旧布局代码片段。


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1476006971227377815)** (2 messages): 

> `Helion 实现, all_gather + FP8 + GEMM 优化, Kernel 性能分析与调试` 


- **Helion 实现落后于基准**：一位成员正根据 [vllm-project](https://github.com/vllm-project/vllm/pull/33933) 的一个 pull request 开发 **all_gather + FP8 + GEMM (H100)** 的 **Helion 实现**，但目前比基准慢 **1.26–4 倍**。
   - 目标是**优化 Kernel** 并进行性能分析，以检查是否存在气泡 (bubbles) 和等待，但通过 Chrome 进行的 tracing 难以追踪。
- **寻求 Kernel 优化工具建议**：一位成员正在寻求关于**优化 Kernel** 的工具或工作流建议，以及相关的技巧、文档或经验分享。
   - 在参考 [Meta data center engineering](https://engineering.fb.com/2026/02/24/data-center-engineering/rrcclx-innovating-gpu-communications-amd-platforms-meta/) 实施后，他们一直在使用 tracing 进行分析，但很难追踪并推断出真正的瓶颈在哪里。


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1475956590568935484)** (4 messages): 

> `Helion PR 1418, Helion all_gather + FP8 + GEMM 优化` 


- **Helion 并行读取研究**：一位成员询问 [Helion PR 1418](https://github.com/pytorch/helion/pull/1418) 是否解决了 [JAX 文档](https://docs.jax.dev/en/latest/pallas/design/design.html#grad-of-pallas-call)中描述的并行读取问题。
   - 该 PR 的作者本周末或下周前无法回复。
- **Helion 中的 FP8 GEMM 优化**：一位成员正在开发 **all_gather + FP8 + GEMM (H100)** 的 **Helion** 实现，参考见[此 pull request](https://github.com/vllm-project/vllm/pull/33933)。
   - 当前实现比基准慢 **1.26–4 倍**，目标是优化 Kernel，并就性能分析工具和工作流征求建议。
- **请求 NCU 见解**：一位成员建议使用 **NCU** 以获取有关 Kernel 优化的可行见解。
   - 未提供关于 NCU 使用的进一步信息。


  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/)** (1 messages): 

vovw: 了不起的工作
  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1475966707507138610)** (5 messages): 

> `CUDA kernels, TPU 推理, MLSys 职位, 分布式训练` 


- **推理和 MLSys 职位中的 CUDA Kernel 知识**：一位成员询问在拥有 **TPU 推理**经验的情况下，是否必须具备**深厚的 CUDA kernel 知识**才能胜任推理和 MLSys 职位。
   - 另一位成员也表达了类似的疑问，即作为一名本科生需要掌握多少 **CUDA/kernels** 知识，这反映了进入该领域的人员的普遍担忧。
- **训练 vs 推理：CUDA Kernel 的重要性**：一位成员分享了他们在**分布式训练**方面的经验，指出除了 **Ampere** 架构之外，深厚的 CUDA kernel 知识并非总是必不可少的，但绝对很有价值。
   - 他们回顾了一些编写特定 kernel 来替换算子 (op) 会大有裨益的情况，强调了解训练和推理两者都有帮助，但并非严格要求。


  

---

### **GPU MODE ▷ #[from-scratch](https://discord.com/channels/1189498204333543425/1466534042768904356/1476227907327098931)** (1 messages): 

> `Serenade Language, C++ transpilation, CUDA and x86-64 ASM, GPU kernels, Dear ImGui support` 


- **Serenade："Python 之简，C++ 之速" 现世**: 一位成员介绍了 **Serenade**，这是一种可转译为 **C++**、**CUDA** 和 **x86-64 ASM** 的新语言，旨在拥有像 **Python** 一样的简洁性，同时具备 **C++** 的速度和手动内存管理能力。
   - 该语言包含 [GPU kernels 支持](https://github.com/kaifczxc-lab/Serenade-Cloud) (**serenaCore**，自定义 BLAS kernel)，集成 **Dear ImGui** 支持及单次编译系统，并计划用它开发一个操作系统。
- **Serenade 的目标：结合各语言之长**: 开发者强调 **Serenade** 是一个个人项目，初衷是创建一个结合多种语言优点的强大工具。
   - 目前源码尚未公开，但可以通过浏览器在 [Replit](https://github/kaifczxc-lab/Serenade-Cloud) 上测试 **Serenade** 的最简单功能。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1476001459421188335)** (15 messages🔥): 

> `Kimi vs GLM, Agent Quota, Kimi for coding, KimiClaw browser navigation` 


- **Kimi 与 GLM 的性能对决**: 成员们讨论了 **Kimi** 对标 **GLM 5** 的性能，一位用户戏称 **Kimi** 比 **GLM** 快 10 万倍。
   - 另一位用户认为 **GLM 5** 略占优势，但指出 **GLM 5** 通过官方 z.AI API 运行较慢，而使用其他供应商可能更快。
- **用户寻求 Agent 额度充值**: 一位用户询问如何专门充值 Agent 额度，理由是 **Allegro** 的成本担忧。
   - 他们还注意到 **agent docsis kimi slides with nb pro** 不再免费。
- **Kimi 在编程任务中表现出色**: 在测试了来自 **Kimi**、**MiniMax** 和 **Alibaba** 的编程方案后，一位用户决定坚持使用 **Kimi** 进行编程。
   - 该用户将速度、在线时间、使用限制和模型质量列为决定性因素。
- **KimiClaw 的浏览器盲区**: 一位用户反馈 **KimiClaw** 无法独立进行浏览器导航，感到十分沮丧。
   - 他们询问其他人是否遇到相同问题并寻求解决方案，同时询问：*“在分析/处理大文件时，Kimi 有什么方法可以减少上下文并节省 tokens 吗？我觉得 Claude 有类似的功能。”*


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1476026063896444988)** (11 messages🔥): 

> `Github Reconnection Issues, Local Development Environment Setup, Account Bans, Manus Cookie Problems, Website Design Problems` 


- **用户面临 GitHub 重新连接困境**: 一位成员在重新连接其 **GitHub** 账号时遇到问题，系统提示其创建新仓库而非连接到原始仓库。
   - 该成员表示自己不是程序员或软件开发人员，需要易于理解的说明。
- **本地开发环境 OAuth 环境变量探究**: 一位成员正在寻求获取 **VITE_APP_ID**、**OAUTH_SERVER_URL** 和 **VITE_OAUTH_PORTAL_URL** 环境变量的指导，以便在本地运行 Manus 开发的应用。
   - 他们还询问在本地开发期间是否需要配置 OAuth 以允许 **redirectUri** `http://localhost:3000/api/oauth/callback`。
- **账号创建后即被封禁令用户困惑**: 一位成员反映在创建账号后立即被封禁，并寻求解决此问题的建议。
   - 目前尚未收到任何建议。
- **Manus 系统将 Cookie 难题归咎于基础设施**: 一位成员详细分享了一个问题，即 **Manus** 在自定义域名 ([anointedforai.com](https://anointedforai.com)) 上由于 Cookie 问题陷入重定向循环。
   - **Manus** 自身诊断该问题为基础设施/托管问题，并建议联系支持团队或从 **Manus** 迁移到对 Cookie 设置有更多控制权的平台。
- **成员抱怨 Manus 制作的网站**: 一位成员投诉其网站设计，称其为 *Manus 做的烂东西 (bullshit)*，并请求协助修复。
   - 另一位成员提出通过私信 (DM) 提供帮助。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1475963283151388722)** (8 messages🔥): 

> `git submodules in Aider, Low-cost LLMs for Aider, Deepseek V3.2 for Aider, Xiaomi/mimo-v2-flash for Aider, moonshotai/kimi-k2.5 for Aider` 


- ****Aider** 获得 `/ok` 别名以实现更快的代码更改**：Aider 现在在主分支中有一个新功能：`/ok` 现在是 `/code Ok, please go ahead and make those changes.` 的别名，允许更快的 **code modifications**。
- **用户寻求适用于 **Aider** 的低成本 LLM**：一位用户正在寻求关于寻找与 **Aider** 配合使用的最佳低成本 LLM 的建议，并提到 Gemini 在短短几个小时内就耗尽了所有 Token。
   - 另一位成员建议使用 [OpenRouter](https://openrouter.ai/) 在不同模型之间切换。
- **推荐 **Deepseek V3.2** 用于 **Aider** 的推理**：一位用户推荐将 **Deepseek V3.2** 作为 **Aider** 的默认 LLM，因为它具有良好的推理能力且价格便宜，但有时可能有点慢。
- ****Xiaomi/mimo-v2-flash** 擅长在 **Aider** 中快速编辑文件**：推荐将 **Xiaomi/mimo-v2-flash** 用于 **Aider** 中的“笨”文件编辑功能，如模糊搜索替换或内容补全，因为它非常便宜且非常快速。
- ****moonshotai/kimi-k2.5** 解决 **Aider** 中的难题**：建议将 **moonshotai/kimi-k2.5** 作为规划模型，并将 **mimo-v2-flash** 作为编辑模型，以解决 **Aider** 中更困难的问题。


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1476174943799476294)** (5 messages): 

> `WeAreDevelopers World Congress North America 2026, AI Control Hackathon 2026, Redwood Research, ControlArena benchmark challenges, ControlConf Berkeley` 


- ****WeAreDevelopers** 大会首次亮相北美**：**WeAreDevelopers World Congress North America** 将于 2026 年 9 月 23 日至 25 日在加利福尼亚州圣何塞首次亮相，预计将有 **10,000+ 开发者**和 **500+ 演讲者**关注大规模的真实世界工程。
   - 主题包括扩展分布式系统、API 平台和 DevOps，您可以使用代码 *Community_MLOps* 在 [wearedevelopers.us](https://wearedevelopers.us) 获得 **10% 折扣**。
- ****AI Control Hackathon** 由 Apart Research 发起**：**Apart Research** 与 [Redwood Research](https://www.redwoodresearch.org/) 共同组织，将于 2026 年 3 月 20 日至 22 日举办 **AI Control Hackathon**，重点关注即使在面对颠覆时也能确保 AI 按我们意愿行事的系统。
   - 此次黑客松设有三个赛道，包括 **ControlArena benchmark challenges**、**控制协议设计**和 **red teaming**，提供 **$2,000** 现金奖励以及前往 [ControlConf](https://controlconf.org/) 的行程。
- **提供 **ControlConf** 行程奖励**：**AI Control Hackathon** 的第一名将赢得前往 [ControlConf](https://controlconf.org/) 伯克利（4 月 18 日至 19 日）的全额资助行程，包括机票和酒店。
   - 了解更多关于 [ControlConf](https://controlconf.org/) 的信息。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1476253403221983475)** (2 messages): 

> `SF DSPy Meetup, DSPy in Production, RLMs, Dropbox, Shopify` 


- **旧金山 DSPy Meetup 即将举行**：宣布举办另一场 **SF DSPy meetup**，这次重点关注 **DSPy 生产用例**和 **RLMs**。
   - 来自 **Dropbox** 和 **Shopify** 的工程师将分享案例研究，并会有 **dspy.RLM** 的演示，详见 [Luma 链接](https://luma.com/je6ewmkx)。
- **Dropbox 和 Shopify 工程师将进行演讲**：**Dropbox** 和 **Shopify** 的工程师将在 SF DSPy Meetup 上展示案例研究。
   - 此次 Meetup 将专注于在生产环境中使用 **DSPy** 和 **RLMs**。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1476051238964559986)** (2 messages): 

> `JAX, Functions` 


- **Tinygrad 创始人赞扬 JAX 的函数设计**：tinygrad 的创始人 George Hotz [承认了 JAX 卓越的函数设计](https://x.com/__tinygrad__/status/2026491994546282605)，暗示了其对设计选择的影响或正确性。
   - 第二条推文[进一步强调了这一点](https://x.com/__tinygrad__/status/2026500842749309267)。
- **JAX 函数设计受到称赞**：Tinygrad 的创始人表达了对 JAX 函数设计方法的钦佩。
   - 这表明 JAX 的方法可能作为 Tinygrad 中类似选择的模型或验证。
