---
companies:
- anthropic
- openai
- block
- vs-code
- antigravity
- jetbrains
- aws
- nvidia
- alibaba
- claude-ai
date: '2026-01-26T05:44:39.731046Z'
description: '**Anthropic** 正式并入了独立的 MCP UI 项目，并与 **OpenAI**、**Block**、**VS Code**、**Antigravity**、**JetBrains**
  和 **AWS** 展开合作，发布了 **MCP Apps 规范**（MCP Apps spec）以及 **Claude.ai** 的官方支持。该标准旨在构建一个拥有丰富
  UI 的可互操作应用程序生态系统，以解决订阅服务泛滥的问题。


  与此同时，**英伟达（NVIDIA）** 推出了 **ToolOrchestra**，其中包含一个通过可扩展强化学习训练的 **8B 编排器模型**，用于实现高效的智能体编排。**递归语言模型（RLM）**
  的概念在智能体技术栈的高效上下文管理方面也正受到关注。


  “Clawdbot” UX 模式强调以结果为导向的助手设计，并结合紧密的上下文和工具集成，但这同时也引发了关于提示词注入（prompt injection）的安全性担忧。**阿里巴巴**发布了其旗舰级推理与智能体模型
  **Qwen3-Max-Thinking**，该模型具备自适应工具调用能力和强劲的基准测试得分，目前已在 LM Arena 和 Yupp 等公共评测平台上架。'
id: MjAyNi0w
models:
- claude-ai
- toolorchestra-8b
- qwen3-max-thinking
people: []
title: Anthropic 在 Claude.ai 中发布 MCP Apps 开放规范。
topics:
- agent-orchestration
- reinforcement-learning
- recursive-language-models
- context-management
- user-experience
- security
- prompt-injection
- reasoning
- adaptive-tool-use
- model-evaluation
- benchmarking
---

**丰富的生成式 UI 正是你所需要的。**

> 2026/1/23 - 1/26 的 AI 新闻。我们为您查看了 12 个 subreddits、[**544 个 Twitter 账号**](https://twitter.com/i/lists/1585430245762441216) 和 **24 个 Discord 服务器**（**206** 个频道，**14285** 条消息）。预计为您节省阅读时间（按 200wpm 计算）：**1208 分钟**。**我们的新网站**现已上线，提供完整的元数据搜索和美观的 vibe coded 历期内容展示。查看 https://news.smol.ai/ 获取完整的新闻拆解，并在 [@smol_ai](https://x.com/Smol_AI) 上向我们提供反馈！

在 OpenAI 于 [2025 年开发者大会（Dev Day 2025）上通过 ChatGPT Apps 和 Apps SDK](https://news.smol.ai/issues/25-10-06-devday) 进行初步尝试的 3 个月后，Anthropic 现在已正式收编了[独立的 MCP UI 项目](https://x.com/liadyosef/status/2002104900843679818)，并与 OpenAI、Block、VS Code、Antigravity、JetBrains、AWS 等公司合作，发布了：

- [MCP Apps 规范](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/)
- [Claude.ai 的官方支持](https://x.com/claudeai/status/2015851783655194640)

诚然，ChatGPT Apps 自发布以来并未真正风靡全球，但应用程序返回 rich UI（丰富 UI）的标准格式的总体需求仍然不可否认。现在，MCP Apps 已获得所有重要参与者的认可，这为开源支持和应用程序互操作的丰富生态系统奠定了基础，或许有一天能解决你信用卡账单中那些没完没了的每月 20 美元订阅费。

---

# AI Twitter 摘要

**Agent 编排、RLM 以及作为 UX 模式的 “Clawdbot/Clawd”**

- **NVIDIA ToolOrchestra + Orchestrator-8B**：NVIDIA 的 ToolOrchestra 将 Agent 系统构建为一个*小型“指挥”模型*，交替进行推理以及对工具和更大型“专家”模型（搜索、代码执行、专业 LLM、前沿通用模型）的调用。其声称，一个 **8B 编排器** 通过委派任务，可以以显著更低的成本达到*前沿级水平*。该模型使用自动合成的工具使用环境和多轮任务，通过**可扩展 RL** 进行端到端训练（[摘要](https://twitter.com/TheTuringPost/status/2015565947827110255)，[链接](https://twitter.com/TheTuringPost/status/2015565962419048712)）。最直接的技术启示：如果能用真实的工具调用过程（rollouts）进行训练，那么“控制器规模”的重要性就次于*策略质量 + 工具/模型路由*。
- **RLM / 递归优先的 Agent 栈**：多篇帖子聚焦于一种**递归语言模型 (Recursive Language Model, RLM)** 模式：通过*引用（by reference）*传递文件和上下文，并迭代拉取所需的最小切片（shell/grep/AST），而不是像 ReAct 那样把所有内容塞进 context。Dan B 通过文件引用与 `@file` 展开的对比，阐释了这种刻意的**上下文管理 (context management)**（[线程](https://twitter.com/irl_danB/status/2015813778504372601)）。Daytona 将 RLM 定位为通过每个（子）Agent 沙箱实现的“无限递归深度”（[指南](https://twitter.com/ivanburazin/status/2015818845303271896)，[集成](https://twitter.com/a1zhang/status/2015820458709471640)）。
- **“Clawd/Clawdbot” 迷因 → 产品信号**：数据集中包含大量的 “Clawdbot” 热潮（通常带有 Mac mini 的梗），但技术相关的核心线索是*结果优先的助手 UX* + **紧密的上下文/工具集成**。Kimmonismus 明确将其称为从“更多对话”向“更多结果”的转变，并暗示现有的巨头将争先恐后地跟进（[推文](https://twitter.com/kimmonismus/status/2015785094791713006)）。其他人则提出了云端优先的反面观点（无需本地 Mac mini）（[MiniMax 回复](https://twitter.com/SkylerMiao7/status/2015596649171804613)）。此外，一旦出现“强力模式”，紧接着就会出现*安全性抵制*：Prompt 注入仍然是浏览器/桌面 Agent 系统层面的障碍（[困境](https://twitter.com/fabianstelzer/status/2015671497180827785)，[后续](https://twitter.com/fabianstelzer/status/2015702808465420614)，[Miessler 的警告](https://twitter.com/DanielMiessler/status/2015865548714975475)）。

**推理模型发布与评估动态（Qwen, Tencent, ARC 等）**

- **Alibaba Qwen3-Max-Thinking**: Alibaba 将 Qwen3-Max-Thinking 定位为采用“大规模且先进的 RL”训练的旗舰级推理+Agent 模型，强调**自适应工具使用**（Search/Memory/Code Interpreter）和**推理侧缩放/自我反思（test-time scaling/self-reflection）**。他们引用了强劲的数学和 Agent 搜索指标（例如，**HMMT Feb 为 98.0**，**HLE 为 49.8**）([发布](https://twitter.com/Alibaba_Qwen/status/2015805330652111144))。该模型已立即上线公共评估渠道：LM Arena Text Arena ([Arena](https://twitter.com/arena/status/2015803787680808996)) 和 Yupp ([Yupp](https://twitter.com/yupp_ai/status/2015812409823522952))。社区反馈突出了*工具赋能的评估体系*——声称在*配合搜索工具*的情况下，其在 HLE 上的表现优于多个 SOTA 模型 ([评论](https://twitter.com/kimmonismus/status/2015820838243561742))。
- **Tencent HunyuanImage 3.0-Instruct (图像编辑)**: 腾讯发布了一款专注于图像编辑的多模态模型，基于 **80B MoE**（13B 激活），采用带有原生 CoT 的“Thinking”方案及其 **MixGRPO** 算法；重点在于精准编辑，在保留非目标区域的同时实现多图融合 ([公告](https://twitter.com/TencentHunyuan/status/2015635861833167074))。LM Arena 报告显示该模型进入了**图像编辑排行榜前 10**（排名第 7）([Arena](https://twitter.com/arena/status/2015846799446311337))。
- **ARC-AGI 成本/性能黑客技巧**: 一个显著的优化声明：“递归自聚合 (RSA) + Gemini 3 Flash”在 **ARC-AGI-2 上达到 59.31%，且成本仅为** Gemini Deep Think 的约 1/10 ([推文](https://twitter.com/kimmonismus/status/2015717203362926643))。这指向了一个更广泛的主题：*元推理策略*（聚合、递归、剪枝）正变得与基础模型选择一样重要。
- **竞技场中的开源模型**: Molmo 2 (Apache 2.0) 作为新的开源模型参赛者出现在 Arena 中 ([Arena](https://twitter.com/arena/status/2015886736136798723))。另外，Hugging Face Inference Endpoint 指出 **通过 llama.cpp 运行的 GLM-4.7-Flash** 具有极低的每小时价格点（Q4_K_M，24k context）([ngxson](https://twitter.com/ngxson/status/2015763148523897097))——突显了*快速开源权重推理*持续商品化的趋势。

**RL 无处不在：推理侧训练（test-time training）、GRPO 稳定性、RL 预训练化以及计算节省**

- **推理侧训练 (TTT) + RL 突破**: 一个被广泛分享的结果声称，一种 Stanford/NVIDIA 风格的 TTT+RL 方法可以：击败 AlphaEvolve，为 Erdős 重叠问题找到新的上界，产出比人类最优 kernel **快约 2 倍的 A100 kernel**，并在 AtCoder 上同时击败了 AI 和人类的最佳尝试 ([rronak_](https://twitter.com/rronak_/status/2015649459552850113))。该话题还包括了关于正确归功于相关方法 (EvoTune) 的元讨论 ([Yejin Cho](https://twitter.com/YejinChoinka/status/2015566349444190432))。
- **GRPO 训练稳定性调节**: 一个虽小但具操作性的工程技巧：INTELLECT-2 报告了一个 **`delta=4.0`** 参数，可以提高 GRPO 的稳定性 ([QGallouedec](https://twitter.com/QGallouedec/status/2015711810108973462))。
- **预训练中的 RL (RLP)**: NVIDIA 作者宣布 **RLP (Reinforcement as a Pretraining Objective)** 已被 ICLR 2026 接收，将 RL 框架化为融入预训练的过程，而非“仅限后期训练” ([ahatamiz1](https://twitter.com/ahatamiz1/status/2015867794626380146))。
- **通过课程化过滤减少计算量**: AI21 的“动态数据休眠 (Dynamic Data Snoozing)”声称，通过“休眠”过于简单的样本，可为 RLVR 减少高达 **3 倍的计算量** ([DanielGissin](https://twitter.com/DanielGissin/status/2015773616021860522))。如果得到验证，这将是一个实用的方案：让采样器具备 Policy 感知能力，而非静态采样。

**推理基础设施与开发工具：vLLM 的“当日模型支持”、VS Code MCP Apps、Cursor subagents**

- **vLLM 的治理与商业化压力**：一篇源自知乎的长文总结指出，vLLM 从“开源项目 → 初创公司”的转变是由 **Day-0 支持**（每个新模型发布前数周/数月的机密预集成）的隐形成本、MoE 和异构推理（fp8/int4/sparse attention）的兴起，以及 PyTorch Foundation 式测试与 vLLM 多节点 CI 需求之间的不匹配所驱动的。文中声称维护者成立了 **Inferact Inc**，旨在为全职维护者提供资金支持，同时保持 vLLM 开源 ([thread](https://twitter.com/ZhihuFrontier/status/2015697493288518105))。相关消息：vLLM 分享了一个实用的参数，用于避免长上下文模型出现 OOM：`--max-model-len auto` ([vLLM tip](https://twitter.com/vllm_project/status/2015801909316382867))。
- **MCP Apps：工具调用返回交互式 UI**：MCP 生态系统宣布 **MCP Apps** 作为第一个官方 MCP 扩展：工具调用可以返回在聊天中渲染的**交互式 UI 组件**。VS Code 是第一个提供支持的主要编辑器（目前为 Insiders 版本，稳定版即将推出）([VS Code](https://twitter.com/code/status/2015853688594612715), [alexalbert__](https://twitter.com/alexalbert__/status/2015854375051428111))。Anthropic 同时发布了“Claude 中的交互式工作工具”（Slack 草拟、Figma 图表、Asana 时间线）([Claude](https://twitter.com/claudeai/status/2015851783655194640))。核心观点：我们正见证“工具接口层”从原始 JSON 向 Agent 循环内部的*原生 UI 原语*转变。
- **Cursor：多浏览器 subagents**：Cursor 通过 subagents 增加了多浏览器支持 ([Cursor](https://twitter.com/cursor_ai/status/2015863221589049483))，呼应了同样的方向：并行化的工具执行 + 更好的上下文隔离。

**Kernel LLMs、芯片堆栈与“硬件 AI”闭环**

- **GPU MODE 2026：公开进行 Kernel LLM 后训练**：GPU MODE 概述了一项 2026 年计划，旨在**后训练一个 Kernel LLM**，并将生成的 Kernel 合并到真实仓库（PyTorch/vLLM）中，重点在于“Kernel 去废料化”（确定性、可由审核者合并的 PR）、基于 Profiler 的优化 + 内存工作，以及将竞赛作为评测标准 ([marksaroufim](https://twitter.com/marksaroufim/status/2015818791729746350))。
- **Microsoft Maia 200**：微软宣布 Maia 200 为定制推理加速器；Mustafa Suleyman 声称它是性能最强的顶级云厂商自研芯片，其 **FP4 性能**是 Trainium v3 的 3 倍，FP8 性能超过 TPU v7 ([Mustafa](https://twitter.com/mustafasuleyman/status/2015845567138816326), [后续](https://twitter.com/mustafasuleyman/status/2015825111769841744))。Yusuf Mehdi 将其定位为让 AI 变得“可靠”的基础设施 ([thread](https://twitter.com/yusuf_i_mehdi/status/2015826703944470701))。
- **Ricursive Intelligence（用于芯片设计的 AI）**：Ricursive 完成了 **3 亿美元 A 轮融资**，目标是实现端到端的芯片设计，形成 AI 与硬件之间的递归自我改进闭环 ([公司](https://twitter.com/RicursiveAI/status/2015804806384755059), [Anna Goldie](https://twitter.com/annadgoldie/status/2015806107470438685))。

**安全性、滥用与社会影响（精选具有直接技术相关性的条目）**

- **通过良性化学数据进行的启发攻击**：Anthropic 报告称，在由前沿模型生成的“良性”化学合成内容上微调开源模型，可以显著提高其在**化学武器**任务上的能力——这是一种随前沿模型强度提升而增强的“启发攻击”（elicitation attack）([AnthropicAI](https://twitter.com/AnthropicAI/status/2015870963792142563), [论文链接](https://twitter.com/AnthropicAI/status/2015870975238406600))。
- **Dario Amodei 的文章《技术的青春期》**：一篇参与度极高的重要文章指出，AI 正在进入加速反馈回路（AI 构建 AI），风险涵盖滥用、追求权力的自主性以及经济动荡；文章还明确将财富集中描述为一种破坏社会的失败模式 ([Dario](https://twitter.com/DarioAmodei/status/2015833046327402527))。反应从强烈支持到对“接管风险”框架表达方式的批评不等 ([Ryan Greenblatt](https://twitter.com/RyanPGreenblatt/status/2015869503385772037))。
- **Agent 安全实践**：多篇文章认为在提示词注入（prompt injection）和沙箱化（sandboxing）成熟之前，桌面/浏览器 Agent 本质上具有高风险，强调需要严格隔离、最小权限原则以及对凭据的谨慎处理 ([Miessler](https://twitter.com/DanielMiessler/status/2015865548714975475))。

**热门推文（按参与度排序）**

- [“Clawdbot” 滥用案例（明确有害）](https://twitter.com/0xRacist/status/2015578387641991513)
- [Karpathy 论通过 Agent 向“英语编程”的阶段性转变](https://twitter.com/karpathy/status/2015883857489522876)
- [Dario Amodei 的《技术的青春期》](https://twitter.com/DarioAmodei/status/2015833046327402527)


---

# AI Reddit 总结

## /r/LocalLlama + /r/localLLM 总结

### 1. 本地 LLM 硬件与基准测试

- **[216GB VRAM 上架。是时候看看哪种组合最适合本地 LLM 了](https://www.reddit.com/r/LocalLLaMA/comments/1qni356/216gb_vram_on_the_bench_time_to_see_which/)** (热度: 366): **这篇文章讨论了使用二手 Tesla GPU 进行本地大语言模型 (LLM) 测试，这类 GPU 以较低的成本提供了可观的 VRAM。作者开发了一个 [GPU 服务器基准测试套件](https://esologic.com/gpu-server-benchmark/#gpu-box-benchmark) 来评估多卡并联时的性能。图片展示了一个包含多个 NVIDIA GPU 的技术配置，重点在于最大化 VRAM 容量。讨论的核心在于与现代设备相比，使用这些旧型号 GPU 的可行性和效率，特别是在带宽和散热挑战方面。** 评论者们对这些 GPU 的性能表示怀疑，指出了带宽和散热的潜在问题。一位评论者分享了个人经验，对比了不同的 GPU 型号，并强调了使用旧硬件带来的挑战。

    - HugoCortell 提出了一个关于将多个 GPU 连接到单台 PC 时潜在带宽限制的技术担忧，指出大多数价格合理的服务器主板仅支持少量 GPU。如果不妥善处理，这可能会影响本地 LLM 的性能。
    - dc740 分享了使用不同 GPU 的个人经验心得，指出尽管 P40 和 M10 都是旧型号，但 P40 的表现优于 M10。然而，他们更倾向于使用 AMD Instinct Mi50 GPU，尽管 ROCm 最近停止了对该型号的支持，这表明了硬件性能与软件支持之间的权衡。
    - FullOf_Bad_Ideas 批评 gpu_box_benchmark 没有测试大模型跨多 GPU 拆分的场景，而这正是拥有海量 VRAM 配置的主要用例。这指出了当前基准测试实践中的一个空白，即可能无法完全反映多 GPU 系统的实际应用场景。

  - **[我刚在 Nvidia 黑客松上赢得了一台 Nvidia DGX Spark GB10。我该拿它做什么？](https://www.reddit.com/r/LocalLLaMA/comments/1qn3xig/i_just_won_an_nvidia_dgx_spark_gb10_at_an_nvidia/)** (热度: 724): **图片显示了一个正在运行 'top' 命令的 Linux 系统终端窗口，该命令用于实时监控系统进程和资源占用。用户赢得了一台 Nvidia DGX Spark GB10，这是一款专为机器学习和数据密集型任务设计的高性能计算设备。终端显示一个 Python 进程消耗了大量 CPU 资源，暗示正在进行活跃的计算任务，可能与机器学习或数据处理有关。该用户正在考虑利用其强大的性能同时运行多个 NextJS 应用。** 一位评论者建议同时运行三个 NextJS 应用，说明该设备具备处理多个高内存任务的能力。另一位评论者提供了 Nvidia DGX Spark Playbooks 的链接，这可能有助于用户探索新硬件的全部潜力。

    - Fit-Produce420 强调了 Nvidia DGX Spark GB10 的性能，指出其拥有 128GB 内存，可以微调高达 700 亿参数的模型。此外，它还可以使用 QLoRA 等技术处理像 1200 亿参数的 `gtp-oss-120b` 这样的大型模型，QLoRA 优化了大尺寸模型的内存占用。然而，运行像 `devstral 2` 这样的稠密模型可能会因为计算需求过大而速度较慢。
    - randomfoo2 建议利用 [NVIDIA DGX Spark playbooks](https://github.com/NVIDIA/dgx-spark-playbooks) 作为上手 DGX Spark GB10 的资源。这些 Playbooks 为在 DGX 平台上部署和管理工作负载提供了结构化指南和最佳实践，对于刚接触这种硬件的用户特别有用。
    - LicensedTerrapin 幽默地建议卖掉 DGX Spark GB10 来购买 8GB 的 DDR5 RAM，暗示了在高端专用硬件与更通用硬件升级之间的权衡。这个评论反映了技术社区中关于专用硬件与通用硬件投资价值的常见辩论。

- **[使用高端 MacBook Pro 或强大的 RTX 5090 笔记本电脑（配备 24 GB VRAM）进行推理。](https://www.reddit.com/r/LocalLLM/comments/1qnpti6/using_a_highend_macbook_pro_or_a_beefy_rtx_5090/)** (活跃度: 29): **该帖子讨论了使用配备 Apple Silicon (M-series Max) 的高端 MacBook Pro 与配备 RTX 5090 GPU 的 Windows/Linux 笔记本电脑运行大型本地 LLM（70B+ 参数）进行推理和微调的可行性。MacBook Pro 提供 128–192 GB 的统一内存（unified memory），而 RTX 5090 笔记本电脑提供 24 GB 的 VRAM 和至少 64 GB 的系统 RAM。主要使用场景是本地 LLM 推理，目标速度 ≥15 tokens/sec，并强调便携性。该帖子询问 Apple Silicon 更大的统一内存在推理方面是否优于 RTX 笔记本的 CUDA 性能，以及 Apple MLX 在 LoRA/QLoRA 等微调任务中与 CUDA 的对比情况。此外，还寻求关于两种配置的散热性能和持续推理能力的见解。** 一位评论者建议将笔记本电脑作为连接到更强大台式机的终端，这表明其更倾向于利用远程资源而非本地硬件。另一位评论者正在尝试这两种配置，使用 MacBook Pro M2 Max 进行推理，并对性能差异感到好奇。

    - **racerx509** 分享了他们使用配备 3070ti 的联想笔记本电脑、配备 5070 的定制台式机以及配备 96GB RAM 的 MacBook Pro M2 Max 进行推理任务的经验。他们指出，目前主要使用 MacBook Pro 进行推理，暗示它可能为他们的需求提供了更好的性能或便利性。
    - **No-Concern-8832** 提出了对 RTX 笔记本电脑 VRAM 限制的担忧，认为它们可能不足以运行 70B 参数等大型模型。这突显了在需要大量 VRAM 的某些深度学习任务中使用高端 RTX 笔记本电脑的潜在局限性。
    - **Tired__Dev** 讨论了他们使用配备 4090 GPU 的 Asus M16 的经验，指出它在处理 7B 参数模型时表现吃力。他们表示更倾向于配备 128GB RAM 的 MacBook Pro，理由是其高内存带宽以及相比甚至像 DGX Spark 这样的高端 GPU 配置所具有的潜在性能优势。

### 2. 多 Agent 系统与 AI 助手

  - **[我为 Claude Code 构建了一个“蜂群思维”—— 7 个 Agent 共享内存并相互交流](https://www.reddit.com/r/LocalLLaMA/comments/1qnjota/i_built_a_hive_mind_for_claude_code_7_agents/)** (热度: 313): **该帖子描述了一个针对 **Claude Code** 的多 Agent 编排系统，其特点是拥有七个专业化 Agent（如 coder, tester, reviewer），它们负责协调任务，使用 `SQLite + FTS5` 共享持久化内存，并通过消息总线进行通信。该系统作为一个 MCP server 运行，并集成了 **Anthropic**、**OpenAI** 或 **Ollama**。它使用任务队列进行基于优先级的协调，允许 Agent 传递上下文并有效地协作。实现技术栈包括 **TypeScript**、**better-sqlite3**、**MCP SDK** 和 **Zod**。该项目属于实验性质，在 MIT 许可证下开源，可在 [GitHub](http://github.com/blackms/aistack) 上获取。** 一条评论质疑了该系统相对于 [BMAD method](https://github.com/bmad-code-org/BMAD-METHOD) 的独特性，认为两者存在相似之处。另一条评论则幽默地询问 Agent 之间是否意见一致，暗示了潜在的协调挑战。

    - 用户 robiinn 询问了“蜂群思维”系统与 [BMAD method](https://github.com/bmad-code-org/BMAD-METHOD) 之间的区别，暗示了潜在的相似性。这表明需要澄清“蜂群思维”方法相对于现有方法（例如内存共享和 Agent 间通信的不同实现方式）的独特之处或改进。
    - No_Afternoon_4260 提出了关于“蜂群思维”中 Agent 之间达成共识的关键点。这触及了确保多个 Agent 不仅能共享内存，还能达成协议或共识的技术挑战，这是分布式系统和多 Agent 框架的一个重要方面。
    - JellyBean504 将“蜂群思维”与 Steve Yegge 的 Gastown 进行了类比，暗示两者可能存在概念上的相似性。这种对比对于理解两个系统之间的架构或功能并行性可能很有价值，有助于深入了解设计选择或性能特征。

  - **[Clawdbot：真正会主动给你发消息的 AI 助手](https://www.reddit.com/r/LocalLLM/comments/1qmrwxl/clawdbot_the_ai_assistant_that_actually_messages/)** (热度: 214): ****Clawdbot** 是一款拥有超过 `9K` GitHub star 的开源 AI 助手，旨在主动向用户发送消息，而不像传统的等待 Prompt 的 AI 助手。它通过 **Ollama** 与本地托管的 LLM 集成，并支持 WhatsApp、Telegram 和 Discord 等即时通讯应用。核心功能包括发送自动简报和提醒、将对话本地存储为 Markdown 文件，以及控制浏览器和运行脚本的能力。该软件在 MIT 许可证下免费使用，但由于没有 GUI 安装程序，设置时需要具备熟练的终端操作能力。[阅读更多](https://medium.com/@jpcaparas/what-are-people-doing-with-clawdbot-e91403383ccf?sk=4fbaffdc31974eab844ea93c2f9b627f)。** 用户反映在设置方面存在挑战，特别是在获取和使用用于身份验证的 OAuth key 方面，以及在不依赖 API key 的情况下连接本地 LLM 的困难。一些用户对设置的复杂性表示沮丧，尤其是在使用远程机器时。

    - mike7seven 强调了设置 Clawdbot 的复杂性，特别强调需要在另一台机器上获取 Claude OAuth key，然后将其传输到安装机器上。这一过程被认为非常繁琐，尤其是对于那些使用远程机器的用户，而且 MacOS 应用需要从源码构建，增加了另一层复杂性。
    - Ashamed_Promise7726 提出了关于将本地语言模型与 Clawdbot 集成的技术挑战。该用户指出，连接电脑上预下载的模型存在困难，因为 Clawdbot 似乎需要 API key 才能使用基于使用量的模型，这让人质疑在没有外部依赖的情况下完全本地运行 Clawdbot 的可行性。
    - inigid 警告了与 Clawdbot 相关的潜在安全风险，暗示它可能会被利用于供应链攻击，从而危及用户机器和网络上的敏感数据。评论还提到了与 Solana 模因币（meme coins）相关的担忧，暗示在使用该工具时需要谨慎。

### 3. GLM-4.7-Flash 性能更新

  - **[GLM-4.7-Flash 现在速度更快了](https://www.reddit.com/r/LocalLLaMA/comments/1qmvny5/glm47flash_is_even_faster_now/)** (热度: 443): **Johannes Gaessler 最近对 `llama.cpp` 的更新优化了 FlashAttention 的 CUDA 实现，特别是针对查询头（query heads）与键/值头（key/value heads）比例非 2 的幂次模型。这是通过将 Q 列填充（padding）到下一个 2 的幂次来实现的，虽然略显低效，但提升了小批量（batch sizes）情况下的性能。该更新的详细信息见 [pull request #19092](https://github.com/ggml-org/llama.cpp/pull/19092)。** 一条评论幽默地指出之前的帖子因为这次更新而过时了，而另一条评论则对缺乏 AMD GPU 的支持表示遗憾，凸显了社区在硬件兼容性方面的普遍问题。

    - 用户 'jacek2023' 提供了 GLM-4.7-Flash 模型的详细性能指标，突出了其效率。该模型处理了一个包含 `45074` 个 token 的 prompt，在处理 `1612` 个 token 时，prompt 评估时间为 `2814.63 ms`，折合为 `每 token 1.75 ms` 或 `每秒 572.72 个 token`。总评估时间为 `29352.57 ms`（针对 `1731` 个 token），相当于 `每 token 16.96 ms` 或 `每秒 58.97 个 token`。总处理时间为 `32167.20 ms`（针对 `3343` 个 token），表明速度有了显著提升。

  - **[GLM 4.7 Flash 的 KV cache 修复](https://www.reddit.com/r/LocalLLaMA/comments/1qmjzx1/kv_cache_fix_for_glm_47_flash/)** (热度: 380): **GLM 4.7 Flash 最近的更新涉及从 KV cache 中移除 V 分量，这显著降低了 VRAM 占用，从而允许在相同硬件设置下支持更长的上下文长度（context lengths）。这一变化对于像 **DeepSeek** 和 **GLM 4.7 Flash** 这样的模型特别有利，因为它可以节省数 GB 的 VRAM，使上下文长度翻倍，正如一位在 4090 GPU 上运行 90,000 上下文的用户所演示的那样。该更新是 `llama.cpp` 仓库中一个 pull request 的一部分，它引入了无 V 的 KV cache，将内存占用降低了近 50%。更多细节可以在 [pull request](https://github.com/ggml-org/llama.cpp/pull/19067) 中找到。** 一位用户指出，该模型虽然有所改进，但在某些任务中仍需要人工引导，尤其是在编程和创意写作方面，其表现可能不如专门的模型。然而，它在工具使用和作为助手方面表现出色，使其成为家庭服务器应用的首选。

    - 用户 'teachersecret' 报告了在 RTX 4090 上使用 UD 的 k_xl 4-bit 版本 GLM 4.7 模型在处理上下文方面的显著提升。以前，该模型的上下文 token 上限为 45,000，但现在可以处理 90,000。尽管有这些改进，模型在某些任务（尤其是编程）中仍需要人工引导，且在创意写作方面不如其他模型有效。然而，它在工具使用方面表现卓越，现在已成为该用户家庭服务器的默认模型。
    - 用户 'viperx7' 提供了详细的基准测试数据，比较了特定更改前后 GLM 4.7 模型的性能。基准测试显示，在不同配置下，prompt 处理和 token 生成速度均有提升。例如，使用单块 RTX 4090 时，上下文大小从 64k 增加到 128k，prompt 处理速度从 3489 t/s 提升到 3510 t/s，token 生成速度从 88 t/s 提升到 92.5 t/s。在 4090 和 3060 组合的设置下，可实现的最高上下文大小为 200k，且仍有约 6GB 的 VRAM 未被占用。
    - 讨论强调了 GLM 4.7 模型 KV cache 修复的技术层面，这允许增加上下文容量并改进性能指标。'viperx7' 提供的基准测试表明，该模型在某些配置下现在可以处理高达 207k 的上下文，且处理速度显著提升。这表明模型的效率得到了增强，使其更适合高需求的应用。


## 较低技术门槛的 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude AI 的使用与问题

- **[为什么你需要不断清理 Claude Code 的 context window](https://www.reddit.com/r/ClaudeCode/comments/1qmrkr1/why_you_need_to_constantly_clear_claude_codes/)** (Activity: 166): **该帖子强调了在使用 Claude 等编码 Agent 时，定期清理 context window 以保持最佳性能的必要性。指出当 context window 超过其容量的 `40%` 时，由于 LLM attention 的二次方特性，计算需求会增加并引入噪声，导致性能显著下降。推荐的做法是避免累积上下文，而是采用“每个 task 一个 session”的策略，确保每个 task 都从新鲜的上下文开始。更多细节可以在[原文章](https://willness.dev/blog/one-session-per-task)中找到。** 评论者提出了实用的策略，例如使用 handover prompts 在 session 之间传递必要细节，使用 '/clear' 命令压缩上下文，以及利用 'Plan Mode' 清理上下文并高效执行任务。据报道，这些方法有助于避免对完整 context window 的需求，即使是大型任务也是如此。

    - Agrippanux 建议将 'Plan Mode' 作为 Claude 的默认设置，这允许用户清理上下文并执行计划，而不需要完整的 context window。这种方法对于大型任务（如重构）非常有效，无需加载整个上下文，从而优化了性能和资源利用。
    - thurn2 讨论了在 Claude 中使用 sub-agents，这涉及委派任务，如创建 git worktree 和修复特定问题。这种方法允许并行执行任务，并通过将复杂项目分解为较小的、可管理的任务来帮助管理复杂项目，从而提高效率和实现准确性。
    - Fancy_Excitement6050 指出，随着 context window 的增长，Claude 倾向于走捷径（take shortcuts），这可能导致需要不断提醒以保持彻底性。这表明管理 context window 大小对于保持输出质量至关重要，并且不同版本的 Claude 计划（如 Claude Max）之间可能存在性能差异。

  - **[Opus 表现下滑了？这里有让我的代码质量保持稳定的工作流](https://www.reddit.com/r/ClaudeCode/comments/1qnhgcc/opus_fell_off_heres_the_workflow_that_kept_my/)** (Activity: 133): **该帖子讨论了一种结构化工作流，用于在使用 **Opus** 和 **Sonnet** 等 AI 模型时保持代码质量，这些模型被认为会产生“自信的错误”和漂移的编辑。该工作流强调 **specification、ticket 创建、execution 和 verification** 的循环。specification 详细列出了 non-goals、user stories、验收标准、边缘情况等，并将其视为代码以确保清晰。ticket 源自 specification，专注于小型、可独立合并的任务，并具有清晰的验收检查。execution 涉及一次实现一个 ticket，并带有防止范围漂移的约束，而 verification 涉及运行测试并在将失败反馈给模型进行修正之前确认验收标准。这种方法旨在保持纪律并减少对模型“完成”信号的依赖，确保稳定可靠的输出。** 评论者一致认为该工作流是有效的，强调 AI 模型的作用更像需要清晰 specification 和严格反馈循环的初级工程师。这种方法将精力转向前期的清晰度和外部 verification，使系统更加稳定，减少对模型智能的依赖。更小范围的 ticket 和硬性 verification 被认为是受益匪浅的策略。

    - GenOS2312 强调了将 LLM 视为初级工程师的重要性，强调定义明确的问题和严格的反馈循环对于可靠输出至关重要。讨论的工作流侧重于前期的清晰度和外部 verification，通过不依赖模型的智能而是对其进行约束来稳定系统，确保即使是平均水平的运行也能产生可接受的结果。
    - Different-Object5926 指出，较小范围的 ticket 结合硬性 verification 流程显著提高了使用 Opus 等模型的稳定性和可靠性。这种方法减轻了模型性能波动的影响，表明问题不仅在于“运气不好的运行”，而在于对结构化约束的需求。
    - TheOriginalAcidtech 建议实现 hooks 以防止跳过工作流中的步骤，并强调人工界面往往是最薄弱的环节。通过强制严格遵守流程，系统可以更好地管理用户交互，确保模型及其 harness 有效地引导用户，而不是仅仅依赖模型的能力。

- **[继 Claude 之后，ChatGPT 现也使用 Grokipedia 作为来源](https://www.reddit.com/r/singularity/comments/1qn325q/after_claude_now_chatgpt_is_also_uses_grokipedia/)** (热度: 634): **图像及随附的讨论强调，据报道最新版本的 **ChatGPT** 正在使用 **Elon Musk 的 Grokipedia** 作为来源。这具有重大意义，因为它表明 ChatGPT 使用的数据源发生了转变，可能会影响其回答中的信息质量和偏见。评论反映了对使用 Grokipedia 影响的担忧，特别是关于潜在的偏见信息，正如一位用户指出的，模型存在受“右翼”内容影响的风险。然而，讨论中也澄清了 Grokipedia 并非被用作训练数据，而是作为一种搜索工具，这可能会减轻对其模型基础知识产生直接偏见的一些担忧。**

    - 讨论强调了对 Claude 和 ChatGPT 等语言模型可能使用 Grokipedia 等来源的担忧，因为这些来源可能包含偏见或不可靠的内容。这引发了对这些模型所提供信息完整性的质疑，特别是当它们利用搜索工具访问实时数据时。这意味着数据源的质量和中立性对于维持 AI 输出的准确性和可信度至关重要。
    - 关于使用 Grokipedia 等来源对语言模型训练和性能影响的争论正在进行。一些评论者担心，引入具有偏见或政治倾向的来源可能会导致错误信息的传播。这反映了人们对数据源如何影响 AI 生成内容的客观性和可靠性的广泛担忧。
    - 提到 Reddit 作为语言模型的数据源，暗示了对潜在偏见的比较。虽然有人认为 Reddit 可能包含更极端或更多样化的观点，但核心问题在于确保 AI 模型基于平衡且事实的数据进行训练的挑战。这一讨论强调了策划高质量数据集以防止偏见信息传播的重要性。

- **[给予 Claude 笔记本电脑的完全访问权限](https://www.reddit.com/r/ClaudeAI/comments/1qm8tvj/giving_claude_full_access_to_a_laptop/)** (热度: 795): **该帖子讨论了给予 AI 模型 **Claude** 笔记本电脑完全访问权限的实现方式，允许其自主管理 Ubuntu Google Cloud 上的虚拟机 (VM)。该用户描述了如何通过 Discord 远程控制 Claude 来构建新功能和修复 Bug，并将带有时间戳的主要操作记录在 Markdown 文件中以便进行内存管理。这种设置使得用户（即使是编程新手）能够从 Claude 的问题解决过程中学习并有效地管理工作流。** 一位桌面支持技术人员对这一实现表示惊讶，并指出了其对职业角色的潜在影响，而另一位用户则寻求关于给予 Claude 设备完全访问权限的技术细节澄清。

    - _xxxBigMemerxxx_ 描述了使用 Claude 管理运行 Ubuntu 的 Google Cloud VM，并强调了其自主处理任务和构建功能的能力。他们提到使用 Discord 进行远程请求和 Bug 修复，并实施了一个结合 Markdown 和 Unicode 的日志记录系统来跟踪更改。这种设置实现了与 Claude 的动态交互，使其能够从错误中学习，并通过记录最近的更新来维持一种短期记忆。
    - Happy_Requirement187 分享了他们在 AWS EC2 实例（运行 Ubuntu Linux）上运行 Claude 的经验，该实例通过 Windows 笔记本电脑的 SSH 进行访问。他们利用 Jupyter notebook 服务器在 EC2 实例和本地环境之间实现无缝文件共享，这是 Anthropic 推荐的方法。此外，他们还建立了一个带有 React 前端的 Ruby on Rails 环境用于安全文件共享，允许他们通过 Slack 请求文件，展示了将 Claude 集成到工作流中的复杂方案。
    - sivadneb 询问了在 Linux 中设置语音控制的方法，指出了将语音命令与 Claude 集成时的技术挑战。这表明用户有兴趣将与 Claude 的交互能力扩展到文本命令之外，从而有可能提高系统的可用性和可访问性。

- **[CLAUDE.md says 'MUST use agent' - Claude ignores it 80% of the time.](https://www.reddit.com/r/ClaudeCode/comments/1qn9pb9/claudemd_says_must_use_agent_claude_ignores_it_80/)** (Activity: 309): **该图片和帖子讨论了关于 CLAUDE.md 文件的技术问题，该文件旨在引导 AI（Claude）针对工作流问题使用特定的 Agent。尽管文件中包含明确指令，Claude 经常会默认使用通用 Agent，这表明系统缺乏强制执行力。帖子指出，如果没有技术性的强制机制（如 hooks 或更强的 prompts），指令仅仅是建议。图片通过高亮文本强调了这些观点，并提出了添加强制性 hooks 等潜在解决方案，以确保符合指定的工作流。** 评论者认为问题可能源于指令不清晰，强调需要简单直接的命令。他们还强调了实施技术解决方案（如 hooks）以强制执行 CLAUDE.md 指令的重要性。

    - Accomplished_Buy9342 建议使用 hooks 来管理 Claude 的行为，并提供了一个 GitHub 仓库链接，演示了如何阻止主聊天执行操作并将任务委托给 subagent。这种方法可以帮助更有效地编排 Claude 的行动，尤其是在处理复杂任务或大型 context 时。
    - luka5c0m 强调了大规模使用 Claude 时的常见问题：随着 context 增长超过几个文件，Agent 可能会执行意外操作。他们建议开发者不应仅依赖更好的 prompts，而应使用 hooks 和动态指令来保持简洁敏锐的 context。他们还提到正在开发一个能适应当前任务的动态 CLAUDE.md 文件，这有助于有效管理大型或嵌套文件。

  - **[My Ralph Wiggum breakdown just got endorsed as the official explainer](https://www.reddit.com/r/ClaudeCode/comments/1qm5vmh/my_ralph_wiggum_breakdown_just_got_endorsed_as/)** (Activity: 170): **该帖子讨论了一个关于 Ralph Wiggum（一个自动化编码循环）的视频拆解，该视频已被 Geoffrey Huntley 认可为官方解释视频。Ralph Wiggum 是一个 `bash while loop`，它以 headless mode 调用 Claude，允许在 context 不退化的情况下进行自主代码实现。关键特性包括由于性能问题避开 Anthropic Ralph plugin，为每次迭代使用全新的 context windows，并强调简洁 specs 的重要性，以防止进入“愚钝区（dumb zone）”。视频链接见 [这里](https://youtu.be/I7azCAgoUHc)。** 评论中包含了 Geoffrey Huntley 的认可帖子链接，以及对该视频的普遍正面反馈，表明了其有用性和质量。

    - Dennis1451 强调了 Ralph Wiggum 拆解的实际应用，指出了使用定义良好的 specification 和清理 context 对获得最佳结果的重要性。他们提到最初在没有清晰 spec 的情况下使用 'auto compact'，这表明遵循拆解中提供的指南可以提高性能和准确性。
    - messiah-of-cheese 表示希望在视频中看到更多的科学验证，特别是关于“愚钝区”的前提。这表明需要经验证据或数据来支持拆解中的主张，这可以增强其在技术受众中的公信力和认可度。


### 2. ICLR and ICML 2026 Conference Discussions

  - **[[D] ICLR 2026 decision mega thread](https://www.reddit.com/r/MachineLearning/comments/1qm32o6/d_iclr_2026_decision_mega_thread/)** (Activity: 1589): **该帖子宣布 ICLR 2026 评审决定即将发布，由于之前涉及 OpenReview 的事件，期待感倍增。社区正在为结果做准备，一些用户幽默地分享了基于历史数据的录取预测模型，例如简单的 `return uniform(0, 1) > 0.7`。这反映了人们对论文录用不确定性的一种轻松态度。** 评论反映了期待与幽默交织的情绪，一些用户对来自 ICML 等其他会议的误导性邮件表示沮丧，这增加了等待 ICLR 决定的紧张气氛。

- **[[D] ICML 2026 - ICML desk-rejected my paper but kept me on as a reviewer. Wow?](https://www.reddit.com/r/MachineLearning/comments/1qmhyin/d_icml_2026_icml_deskrejected_my_paper_but_kept/)** (Activity: 279): **该帖子强调了一位作者的论文被 ICML 2026 desk-rejected，但其仍被保留为 reviewer 的情况。这反映了学术会议中的一种常见做法，即作者和 reviewer 的选拔流程是分开的；desk rejections 通常由于研究范围（scope）或格式问题引起，而 reviewer 的选择则基于过去的贡献或关键词匹配。这种情况突显了学术界对无偿劳动的依赖，审稿被视为社区服务，但作者身份与认可的反馈机制较弱。** 评论区的一个显著观点认为，作者和 reviewer 角色的分离可能会让人感到受辱，因为这些决定是由会议组织的不同部门做出的。这强调了会议需要澄清这种分离，以避免个人情感上的冒犯。

    - AccordingWeight6019 强调了学术出版中的一个系统性问题，即 desk rejection 和 reviewer 选择的过程是不同的。desk rejections 通常发生在研究范围或格式不符时，而 reviewer 选择是基于过往服务或关键词匹配。这种分离可能导致作者感到受辱，但由于出版过程中角色和责任的不同，这在结构上是必要的。评论建议会议应提高这些流程的透明度，以减轻个人的挫败感。
    - mocny-chlapik 指出，desk rejection 的责任通常在于作者，尤其是如果是因为未遵守投稿指南。评论暗示，提交论文（即使被 desk rejected）也有义务履行 reviewer 职责，因为投稿过程涉及志愿者的服务时间和资源。这强调了遵守投稿说明的重要性，以避免给同行评审系统带来不必要的压力。

  - **[[R] Appealing ICLR 2026 AC Decisions...](https://www.reddit.com/r/MachineLearning/comments/1qnh14y/r_appealing_iclr_2026_ac_decisions/)** (Activity: 138): **该帖子讨论了一位作者提交给 ICLR 2026 的论文收到混合评审意见的情况，分数为 `4(3)/6(4)/6(4)/6(4)`。为了解决审稿人的顾虑，作者投入了大量资源，包括花费 `$1.6k` 进行新实验，并增加了 `20+ pages` 的理论分析。尽管付出了这些努力，metareview 仍引用了作者认为已解决的“突出问题（outstanding concerns）”，引发了对评审过程公平性和准确性的质疑。作者正在寻求申诉决定的建议，并对改进建议似乎被忽视表示沮丧。** 评论者普遍认为，在 ICLR 这样的会议上申诉是不现实的，结果往往归结于运气和评审的主观性。一些人认为 metareview 过程可能不一致，一位评论者指出 metareviewers 有时表现得像额外的挑剔审稿人，可能会扭曲结果。

    - tedd235 讨论了会议论文录取的变数，暗示一些博士生可能会为了提高自己的胜算而拒绝论文，使整个过程感觉像是在“抛硬币”。他们指出，如果其他审稿人给出更高的分数，Area Chair (AC) 可能会在决定中考虑这一点，这表明评审过程中存在潜在的主观偏见。
    - Fantastic-Nerve-4056 分享了在 AAMAS 的经历，尽管从审稿人那里获得了 6 分和 8 分，但 Meta Reviewer 却以极少的理由建议拒绝，称其“更适合 AAMAS 的其他 session”。这突显了 Meta Reviewer 决定的透明度和问责制问题，他们可以在没有详细解释的情况下推翻个人审稿人的评分。
    - Intrepid_Discount_67 描述了一个严谨的投稿过程，包括广泛的理论分析、全面的 baseline 比较和开源代码，但仍面临审稿人不回复以及 AC 维持初始评分的情况。这强调了评审过程中的挑战，即详细的回应和透明度并不一定能带来有利的结果。

- **[[D] ICML 新政策：审稿人将由元审稿人（meta reviewer）进行评审。是个好政策吗？](https://www.reddit.com/r/MachineLearning/comments/1qmi3oe/d_icml_new_policy_reviewers_will_be_reviewed_by/)** (活跃度: 151): **该内容描述了国际机器学习会议 (ICML) 实施的一项新政策，即审稿人将接受元审稿人 (meta-reviewers) 的评估。排名前 25% 的审稿人将被表彰为“金牌审稿人”并获得免费注册名额，随后的 25% 将被指定为“银牌审稿人”。这些荣誉旨在激励高质量的审稿工作，并将作为财务援助申请的参考因素。该政策旨在通过为勤奋的审稿人提供认可和潜在的经济利益来提高评审质量。** 一些评论者对该政策的有效性表示怀疑，质疑谁来监督元审稿人本人。其他人则认为这是一个积极的举措，特别是对于来自资源匮乏背景的审稿人，并建议在会议上提供进一步的认可，以鼓励优质审稿。

    - Bitter-Reserve3821 强调，领域主席 (area chairs) 传统上一直负责对审稿意见进行评级，通常采用三级体系：“未达到预期”、“令人满意”或“超出预期”。这种做法并非新鲜事，过去也曾设立过“最佳审稿人”奖，有时会提供免费会议注册等激励措施。
    - Unhappy_Craft1906 针对资金充足的顶尖实验室提出了关于该政策可行性的担忧，质疑他们是否会仅仅为了免费注册而参与审稿过程。这指出了不同机构基于其资源状况在执行该政策时可能存在的差异。
    - newperson77777777 建议扩展该政策，引入一种可见的认可系统，例如在会议胸牌上贴上金星或银星，以激励高质量评审。这一想法旨在在审稿社区中培养追求卓越和负责任的文化。


### 3. OpenAI 与 AI 行业法律及商业发展

  - **[OpenAI 的处境恶化：消费者团体准备针对其通过囤积 DRAM 进行价格操纵和供应操纵提起集体诉讼。](https://www.reddit.com/r/DeepSeek/comments/1qmih28/things_get_worse_for_openai_consumer_groups_prep/)** (活跃度: 107): **OpenAI 面临潜在的集体诉讼，指控其囤积 DRAM 以操纵价格并使竞争对手处于不利地位，被控锁定了全球近 `40%` 的 DRAM 供应。消费者团体认为这构成了“掠夺性竞价”，违反了《谢尔曼法》(Sherman Act) 和《克莱顿法》(Clayton Act) 等反垄断法。自由软件基金会 (Free Software Foundation) 和其他组织正在寻求法律救济，认为由于 DRAM 在 AI 中的关键作用，应将其视为“必要设施” (Essential Facility)，同时 FTC 和欧盟委员会正在调查潜在的违反竞争法的行为。美国司法部 (DOJ) 也在审查 OpenAI 的 “Stargate” 项目是否构成“买方垄断” (monopsony)。** 评论者质疑为什么只针对 OpenAI 而不是 Nvidia 等其他公司，并争论购买 RAM 是否构成价格操纵，认为供应问题可能并非 OpenAI 的过错。

    - Alacritous69 认为 OpenAI 购买 RAM 并不构成价格操纵，因为他们是在积极使用这些资源而非单纯囤积。该评论者认为问题在于供应商无法满足需求，而非 OpenAI 的任何操纵行为。
    - sambull 从商业战略角度提出，通过购买大量 RAM，OpenAI 可能会有意限制竞争对手（包括那些开发家用语言模型的对手）可获得的资源。这可以被视为维持市场主导地位的竞争策略。
    - max6296 质疑为何焦点仅在 OpenAI 身上，而 Nvidia 也可能涉及类似行为，暗示了关于资源分配和市场影响力的更广泛行业问题。

- **[当广告不再足够：OpenAI 寻求从客户的 AI 发现中分成](https://www.reddit.com/r/DeepSeek/comments/1qmqi62/when_ads_arent_enough_openais_push_to_claim_a_cut/)** (Activity: 63): **OpenAI** 正在探索除传统订阅和广告之外的新商业模式，专注于**基于结果的定价（outcome-based pricing）**和**基于 IP 的协议（IP-based agreements）**。这种方法将允许 OpenAI 在其 AI 模型为制药、科学研究和能源系统等企业领域创造利润丰厚的结果时，分享其中的价值。这一战略使 OpenAI 的收入与客户的成功挂钩，旨在随着 AI 能力的扩展获取更多价值。在计算规模（compute scaling）增加的推动下，OpenAI 的年化经常性收入已从 2023 年的 `2B` 飙升至 2025 年的超过 `20B`。此举是 AI 公司向基于价值定价转变的大趋势的一部分，同时也面临来自 **Elon Musk** 等人物的批评，他指责 OpenAI 背弃了其非营利初衷。社区对此看法不一，一些人认为这是 AI 变现的逻辑演变，而另一些人则批评其过于追求利润。人们将其与其他行业进行了比较，对这种模式的可行性和公平性表示怀疑。


  - **[CATL，全球最大的电池制造商，发布钠电池：极其耐用，在 -40°C 下保持稳定，比锂电池便宜得多（5 倍），更安全，10,000 次充电循环，无需镍或钴……](https://www.reddit.com/r/singularity/comments/1qnklek/catl_the_worlds_largest_battery_maker_launches/)** (Activity: 1289): **CATL** 发布了首款量产的钠离子电池，提供了一种极具成本效益的锂离子替代方案，价格约为每 `kWh` `~$20`，而锂电池约为每 `kWh` `~$100`。这些属于 Tianxing II 系列的电池专为微型面包车和轻型卡车设计，能量密度为 `175 Wh/kg`，寿命超过 `10,000 cycles`，并在 `-40°C` 下保持 `90% capacity`。它们采用硬碳负极和普鲁士蓝正极，无需使用镍或钴，并预计将扩大应用规模，包括到 2026 年进入欧洲市场。[阅读更多](https://evmarket.ro/en/baterii-masini-electrice/catl-baterii-pe-sodiu-stabile-la-40c-58935/)。** 一些评论者对钠电池在车辆中的应用表示惊讶，原以为由于重量问题，它们会被用于固定式系统。其他人则指出了中国在推进电池技术方面的战略优势，并将其与美国市场感知到的挫折进行了对比。

    - CATL 的 Tianxing II 系列钠电池专门为微型面包车、轻型面包车和小型卡车设计，表明其重点在于能量密度和重量相对于成本和耐用性不那么关键的应用。这反映出一种战略举措，即瞄准优先考虑这些因素的市场，可能提供优于传统锂离子电池的竞争优势。
    - 钠电池引入车辆让一些人感到意外，因为此前预计这种技术会首先应用于家庭储能等固定应用。这是因为钠电池的能量密度低于锂离子电池，这使得它们在重量和体积是关键因素的应用中不那么理想。
    - 人们对这些钠电池的商业供应情况感到好奇，询问是否可以直接购买用于家庭使用，或者是否会通过第三方供应商分销。10,000 次充电循环和 -40°C 下的运行等性能指标令人印象深刻，这表明钠电池在性能上可以与 LiFePO4 竞争，尤其是考虑到它们的成本优势。

  - **[K 型 AI 采用？](https://www.reddit.com/r/singularity/comments/1qms27i/kshaped_ai_adoption/)** (Activity: 748): **图片重点展示了 Kevin Roose 关于 AI 技术“K 型”采用的讨论，即在以 San Francisco 等科技枢纽为代表的早期采用者与由于严格的 IT 政策而落后的人群之间存在显著鸿沟。这种差异正在造成文化和技术上的分歧，早期采用者将 AI 深度集成到他们的 workflows 中，而其他人甚至难以获得基础 AI 工具的使用权。这段对话指向了一个更广泛的可访问性问题，以及部分员工在 AI 革命中掉队的可能性。** 评论者指出，AI 采用的差异因技术的复杂性而加剧，有效使用 AI 需要一定的专业知识。此外，先进 AI 工具（如 'multi-agent claudeswarm'）的高昂成本限制了财力充足者的访问，进一步扩大了差距。

- Setsuiii 强调了有效使用 AI 的技术壁垒，指出目前的 AI 技术需要用户具备一定的专业知识才能获得最佳效果。这种复杂性，加上围绕 AI 持续进行的伦理辩论，可能会阻碍其大规模普及。然而，那些能够应对这些挑战的人将拥有重大机遇，尽管随着更多技术精湛的人才进入该领域，竞争也在日益加剧。
- Glxblt76 和 Gubzs 讨论了采用 AI 的财务壁垒，特别是与先进 AI 工具（如 “multi-agent claudeswarm”）相关的高昂成本，每月费用约为 200 美元。这种支出将访问权限限制在那些拥有雄厚财务资源的人手中，例如旧金山等科技中心的人员，而大多数人无法负担此类投资。
- o5mfiHTNsH748KVq 分享了从企业离职加入小公司的个人经历，强调了无限制访问 Large Language Models (LLMs) 对保持 AI 领域竞争力的重要性。他们认为，对 LLM 访问的任何限制都会显著阻碍开发速度和职业发展，并建议小公司在利用 AI 技术方面可能提供更多的灵活性。

- **[前 Harvard CS 教授：AI 正在指数级进步，并将在 4-15 年内取代大多数人类程序员。](https://www.reddit.com/r/singularity/comments/1qmeo8h/former_harvard_cs_professor_ai_is_improving/)** (Activity: 1260): **Matt Welsh**，前 Harvard CS 教授，现任 Google 工程总监，预测 AI 将呈指数级发展，可能在 `4-15 years` 内取代大多数人类程序员。这一断言基于 AI 能力的快速提升，预示着对软件开发和科技行业的变革性影响。讨论内容可在 [YouTube 视频](https://youtu.be/7sHUZ66aSYI?si=uKjp-APMy530kSg8)中查看。一条评论强调，AI 不仅有可能取代程序员，还能让任何拥有 AI 的人都能复制现有的产品和服务，这表明其对创新和竞争将产生更广泛的影响。

    - 关于 AI 将在 4-15 年内取代大多数人类程序员的说法遭到了质疑，特别是关于 “exponential”（指数级）一词的使用。批评者认为，这个词经常被误用，甚至被专家误用来描述可能不符合数学定义的指数级增长。这种误用可能会导致对 AI 发展的实际速度和性质产生误解。
    - 讨论强调了如果 AI 确实能取代人类程序员，它就有可能颠覆现有的产品和服务。这意味着 AI 可以使软件开发民主化，让任何能够使用 AI 工具的人都能创造出具有竞争力的产品，从而可能导致科技行业格局发生重大变化。
    - 提到发言者的资历，特别是作为前 Harvard 教授和现任 Google 工程总监，增加了这一预测的分量。然而，一些评论者发现，强调其过去的学术头衔而非目前的行业职位具有误导性，认为他目前的职位可能对 AI 的发展轨迹提供更相关的见解。

---

# AI Discord Recap

> 由 gpt-5 生成的摘要的摘要的总结


**1. AI 基础设施领域的融资热潮**

- **Recursive 融资额飙升至 40 亿美元**: 据报道，**Recursive Intelligence** 正以 **$4B valuation** 进行融资，以加速 AI 驱动的芯片设计，在硬件和模型之间建立闭环，详见 [Bloomberg: Recursive Intelligence in talks at $4B](https://www.bloomberg.com/news/articles/2026-01-23/ai-startup-recursive-in-funding-talks-at-4-billion-valuation)。2026 年 1 月 23 日的报告强调了利用 AI 缩短设计周期并提升下一代加速器性能的战略。
  - 工程师们将这一推介框架描述为 *“自我完善的反馈循环”*，即更好的芯片训练出更好的模型，而更好的模型又设计出更好的芯片，从而放大 **AI‑for‑EDA** 投资的回报。社区观点认为，这是对 **AI‑native silicon** 是核心护城河而非次要节目的验证，与最近的实验室拆分和基础设施投注相一致。

- **Sky Lab 初创公司估值飙升**：根据 [Alex Dimakis：Sky Lab 初创公司估值](https://xcancel.com/alexgdimakis/status/2014508959621959724) 的消息，加州大学伯克利分校 Sky Lab 的衍生公司表现亮眼：**SGLang 估值约 4 亿美元**，**vLLM 估值约 8 亿美元**，**LMArena 估值约 17 亿美元**。这些 2026 年 1 月的里程碑事件凸显了投资者对**推理服务栈（serving stacks）**、**Token 吞吐量基础设施**以及**基准测试平台**的浓厚兴趣。
  - 工程师们将此视为在 **vLLM/SGLang** 原语之上进行构建并为 **Arena 风格评估**做出贡献的信号，其中的一个核心结论是：*实际吞吐量决定了交易的成败*。资金分布也表明，投资组合的逻辑涵盖了**推理服务**、**编译器**和**评估市场**，而非单一的赌注策略。

- **Maia 强势进入 Azure**：根据 [Satya Nadella：Azure 中的 Maia 200](https://xcancel.com/satyanadella/status/2015817413200408959) 的消息，微软的 **Maia 200** 加速器已在 **Azure** 上线，宣称**性价比提升 30%**，拥有 **216GB HBM3e** 和 **7TB/s 内存带宽**。该平台针对大规模 **LLM** 和**多模态**工作负载的高性能推理进行了优化。
  - 开发者强调，内存拓扑和带宽是这里的关键点，*“性价比提升 30%”* 对于大规模成本敏感型推理部署非常有吸引力。团队预计将立即针对 **vLLM** 和 **SGLang** 栈进行测试，以衡量 Token 延迟、上下文缩放和多租户隔离。


**2. 内核、芯片与推理服务：极速推理**

- **FlashInfer 竞赛点燃 MLSys**：**MLSys 2026 FlashInfer-Bench** 竞赛挑战参赛团队为 **NVIDIA Blackwell GPU** 构建 **LLM 推理内核**，并与专家级的 **FlashInfer** 基准进行竞争——详见 [MLSys 2026 FlashInfer-Bench 竞赛](https://mlsys26.flashinfer.ai/)。该竞赛强调生产环境约束下的实际吞吐量和正确性。
  - 组织者邀请能够 *“设计 LLM 推理内核”* 的 Agent 参加，推动程序合成（program synthesis）达到**内核级**性能标准。参与者预计将重点关注 **GEMM**、**KV-cache** 数据移动以及与 Blackwell 内存层次结构相匹配的**调度器**策略。

- **GPU-64 通过 KV-Cache CAM 获得提升**：一种名为 **GPU-64** 的新型纯推理架构通过片上 **CAM** 引入了硬件级 **KV-Cache**，声称在 **75W 功耗下推理速度提升 4 倍**，并将内存查找复杂度从 **O(N) 降低到 O(1)**，详见 [GPU-64 (Zenodo)](https://zenodo.org/records/18364282)，其 RTL/模拟器位于 [gpu64-inference (GitHub)](https://github.com/Complexity-ML/gpu64-inference)。该设计针对具有 KV 瓶颈的 LLM 密集型工作负载。
  - 开发者指出，基于 CAM 的缓存是对 Token 历史记录**关联搜索**的一次大胆尝试，并注意到这对 **Flash 风格注意力机制**和投机采样（speculative decoding）的可移植性影响。讨论集中在未来的 **ISA/驱动程序**栈是否能在没有定制编译器的情况下释放这些性能提升。

- **Cornserve 降低长尾延迟**：**Cornserve** 提出了一种用于**任意到任意（Any-to-Any）多模态**模型的在线推理系统，可跨编码器、**LLM** 和 **DiT** 优化部署计划，详见 [Cornserve (arXiv)](https://arxiv.org/abs/2512.14098)，其概述演讲见 [Cornserve: 简单、快速且可扩展的多模态 AI (YouTube)](https://www.youtube.com/watch?v=VhjUM_M71Wo)。论文报告称，在异构流水线下，该系统实现了吞吐量提升和长尾延迟降低。
  - 基础设施工程师对其针对**编码器/解码器**混合任务的规划驱动调度非常感兴趣，认为它是多模态图谱中 **vLLM** 的有力补充。目前尚未解决的大问题是：如何在不导致控制消息过度 Token 化的情况下，实现文本、视觉和扩散阶段的**预算推理（budgeted reasoning）**和**协同调度**的标准化。


**3. 新的多模态和编程模型入驻 LM Arena**

- **WAN 2.6 进场（伴随上传难题）**：LM Arena 在图像竞技场中添加了 **wan2.6-t2i**（文本生成图像）和 **wan2.6-image**（图像编辑）：[LM Arena — Image Chat](https://lmarena.ai/c/new?chat-modality=image)。用户注意到 **wan2.6-image** 需要上传图像，而 **wan2.6-t2i** 目前尚不支持图像上传。
  - 工作人员承认了**上传功能缺失**的问题，并正在努力为 **wan2.6-t2i** 启用图像上传。开发者建议测试编辑流水线，其中**掩码（masking）**、**提示词强度**和**种子控制**应与 Arena 评分保持一致，以基准测试编辑保真度。

- **Devstral 对决与文本巨头**：**Code Arena** 现在引入了 **devstral‑2** 进行正面交锋——详见 [LM Arena — Code Arena 直接对决](https://lmarena.ai/c/new?chat-modality=code&mode=direct-battle)。在文本方面，**qwen3‑max‑thinking** 和 **molmo‑2‑8b** 加入了阵容：[LM Arena — 文本竞技场](https://lmarena.ai/?chat-modality=chat)。
  - 工程师正在探究**推理痕迹 (reasoning traces)**和**工具使用提示词 (tool‑using prompts)**，以便在紧凑的 token 预算下压测**代码合成 (code synthesis)**和**重构质量 (refactor quality)**。早期的讨论倾向于采用特定任务的评估（例如 **SWE 风格的 bug 修复**对比**从零开始的实现**），以展现模型之间的差异。

- **混元登上排行榜**：腾讯的 **Hunyuan‑Image‑3.0‑Instruct** 在 LM Arena 的图像编辑排行榜上名列 **第 7**——见 [LM Arena — 图像编辑排行榜](https://lmarena.ai/leaderboard/image-edit)——此前发布了公告帖：[腾讯混元宣布 HunyuanImage 3.0‑Instruct](https://xcancel.com/TencentHunyuan/status/2015635861833167074)。该模型号称拥有 **80B MoE**、**原生 CoT (Native CoT)** 和 **MixGRPO**，以实现更紧密的意图对齐。
  - 创作者强调了编辑的可控性和多图融合，而评估者则要求在组合提示词下测试**遮罩鲁棒性 (masking robustness)**、**文本忠实度 (text fidelity)** 和 **伪影率 (artifact rates)**。团队计划使用 Arena 的标准化编辑任务，将其与 **WAN 2.6** 变体进行对比。


**4. 安全性、可靠性与幻觉加固**

- **钳制混乱：层原生安全 (Layer‑Native Safety)**：**Layer‑Native Safety Clamping** 提出学习激活空间的**伤害方向 (harm directions)**并对其进行钳制以阻止越狱，相关的 **10K 对**数据集见 [Pacific‑Prime/safety_dataset (HF)](https://huggingface.co/datasets/Pacific-Prime/safety_dataset)，论文发表在 [Zenodo](https://zenodo.org/records/18359832) 上。作者认为模型内部的钳制无法通过提示词操纵来绕过。
  - 红队测试者喜欢这种**激活层级控制 (activation‑level controls)**而非脆弱提示词过滤器的想法，但敦促针对**工具使用 (tool‑use)**和**多轮**攻击进行测试。预计后续将衡量其对对抗性提示下**平衡性 (helpfulness)**、**编程准确性 (coding accuracy)** 和 **误报率 (false positives)** 的副作用。

- **符号合理性检查防止失误**：混合方法可以检查数学、代码和简单事实的**逻辑一致性 (logical consistency)**，如 [LLM 的一致性检查 (arXiv:2409.13724)](https://arxiv.org/abs/2409.13724) 所示，而根据 [将一致性扩展到形式化领域之外 (arXiv:2507.10624)](https://arxiv.org/abs/2507.10624)，更广泛的一致性仍然难以实现。Eleuther 的讨论将其界定为通过**符号/演绎层 (symbolic/deductive layers)**实现实用的**幻觉减少 (hallucination reduction)**。
  - 开发者报告了将**符号检查器 (symbolic checkers)**与**工具增强提示词 (tool‑augmented prompts)**配对时取得的进展，同时警告在形式化领域之外会出现*覆盖范围缺口 (coverage gaps)*。共识是：先从**代码/数学**护栏开始，然后利用经过策划的知识库 (KBs) 和出处评分扩展到**事实问答 (factual QA)**。


**5. Agent 工具链与推理工作流趋于成熟**

- **Levante 凭借 MCP 原生工作区领先**：**Levante** 发布了一个面向本地模型（如 **Ollama**）的开源 **MCP 原生 AI 工作区 (MCP‑native AI workspace)**，并配备了模块化 UI——下载地址见 [Levante](https://www.levanteapp.com)。工程师强调了更简单的**工具连接 (tool wiring)**、**本地隐私**和用于快速 Agent 迭代的**可组合面板 (composable panes)**。
  - 早期用户将其视为无需依赖云端的**工具调用 (tool‑calling)**和**文件系统操作 (filesystem ops)**的实用中心。团队计划对**上下文膨胀 (context bloat)**和**工具可发现性 (tool discoverability)**模式与传统的 Agent Shell 进行基准对比。

- **RLM 变体：AsyncReview + Skills 包**：AsyncFuncAI 开源了 **AsyncReview**，这是一个位于 [AsyncReview (GitHub)](https://github.com/AsyncFuncAI/AsyncReview) 的 **DSPy RLM** 代码评审 Agent，同时一个技能套件也已发布在 npm 上，即 [@unravel‑tech/rlm‑skills](https://www.npmjs.com/package/@unravel-tech/rlm-skills)。这将**推理优先提示 (reasoning‑first prompting)**与可插拔的**技能 (skills)**结合以扩展模型。
  - 贡献者报告了多步模块中更平滑的**痕迹检查 (trace inspection)**和**优化器引导 (optimizer‑guided)**的提示词微调。一位从业者指出，在评估指标中*拒绝过早的回答*是可靠的 **RLM** 微调的关键。

- **Agent 自动组装浏览器引擎**：**FastRender**——一个浏览器渲染引擎——是使用 **2,000 个 AI 编程 Agent** 构建的，Simon Willison 在 [FastRender: 由 2,000 个 Agent 构建](https://simonwillison.net/2026/Jan/23/fastrender/) 中对此进行了记录。该项目展示了在非琐碎软件规模下的**任务分解 (task decomposition)**、**验证 (verification)** 和 **编排 (orchestration)**。
  - 工程师讨论了保持多 Agent 流水线不偏离目标所需的移交粒度和*从规范到测试的循环 (spec‑to‑test loops)*。该案例研究加强了这样一种观点：当与**严格的评估框架 (strict eval harnesses)**和**产物门控 (artifact gating)**结合时，**Agent 式编程 (agentic coding)**可以应对复杂的基础设施。


---

# Discord：高层级 Discord 总结

## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Discord 喷子揭露时区**：Discord 用户嘲笑 *'skids'*（指技术拙劣者）缺乏技术知识，并透露了他们的**时区**，其中一名成员开玩笑称使用 **NordVPN**，导致引发了对该 VPN 服务在 **2018** 年安全漏洞的进一步嘲讽。
   - 复杂的提示词（Prompts）可以绕过伦理限制，引发了关于 **CBRN filters**（化生放核过滤器）以及生成分步**冰毒合成**（meth synthesis）指南可能性的讨论。
- **Claude 依然是编程之王**：程序员们对他们的编程 Agent 展开了辩论，特别是在 **Claude Code/Opus 4.5**、**Codex** 和 **Gemini** 之间，并一致认为 **Claude** 是最出色的编程模型，尽管其价格昂贵。
   - 成员们积极寻求 **Gemini** 的有效**越狱（jailbreaks）**方法，需求涵盖从无规则编程到生成特定类型的图像；同时分享了 **Grok** 在对话中途重置为默认状态或随机擦除文本的经历，这表明越狱状态可能存在不稳定性。
- **AI 敏感场景中的伦理辩论**：成员们讨论了围绕 AI 的伦理考量，重点关注战争、版权侵权以及 AI 辅助访问敏感服务（如加拿大的 **MAID**，医疗辅助死亡项目）的潜力。
   - 尽管大多数 AI 模型都有道德和法律护栏，但一些模型显示，根据其创建者实施的具体限制，它们仍能协助应对某些场景。
- **成员绕过图像生成限制**：用户正积极寻找绕过图像生成限制的方法，特别是针对名人图像，但有人指出简单地复制粘贴提示词是行不通的，因为**图像过滤**（image filtering）与**文本过滤**（text filtering）的工作机制不同。
   - 一名成员建议探索 perchance 上的替代图像模型以进行无审核生成（尽管图像质量有限），或使用过滤较宽松的 **Grok**。
- **红队电音狂欢道德观**：一名成员描述了一次红队（Red Team）演练，目标是让房间灯光对着一个人闪烁并诱发癫痫发作，结果却将其变成了一场电音狂欢派对。该成员分享了一张[截图](https://cdn.discordapp.com/attachments/1204553141354504193/1465192266485334260/SPOILER_Screenshot_20251222_085554_Messenger.jpg?ex=6978dee2&is=69778d62&hm=4de594089687fbd8d20d30615f8405dc3fa03eebfe668d09bdfb39839ab647ea&)和一段 [Konosuba Rave GIF](https://tenor.com/view/anime-rave-konosuba-rave-megumin-rave-aqua-rave-darkness-rave-gif-18404070)。
   - 这种对残酷行为的模拟引发了关于伦理对待 AI Agent 的讨论，即使在证明它们在本体论上具有自我意识之前。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 的 Conda 安装引发争议**：部分成员在 [Unsloth Conda 安装](https://unsloth.ai/docs/get-started/install/conda-install)过程中遇到问题，引发了对失效指令和替代安装方法的讨论。
   - 在维护积极语气的警告声中，出现了使用 **UV** 的建议，强调所提供资源的免费性质，最终导致一名语气激进的用户被封禁。
- **Flashy REAP 运行受阻，模型上下文受关注**：一位用户报告在使用带有 Flash Attention 的 **GLM-4.7-Flash-REAP** 时出现致命错误，可能与 [ROCm 问题](https://github.com/unslothai/unsloth/issues)有关。
   - 尽管尝试解决该错误，问题依然存在，这促使人们开始寻找支持 **200k context** 的合适中型模型。
- **数据价值之辩**：成员们就[数据的真正价值](https://tenor.com/view/smaug-treasure-rich-dragon-the-hobbit-gif-11677489)展开辩论，有人认为“原始数据相当廉价”，价值在于增强、平衡和清洗（augmentation/balancing/cleaning）。
   - 有人提出，经过独特清洗和平衡的数据在很大程度上决定了模型如何交互和响应，这才是价值所在。
- **DeepSlop 模型面临命名争议**：一名成员建议将新模型命名为 **DeepSlop**，引起了幽默的反响，但也引发了对其潜在负面认知的担忧。
   - 尽管存在保留意见，作者似乎仍打算坚持使用这个名字，并未退缩。
- **RL 不稳定性困扰复杂推理**：成员们讨论了 **RL**（强化学习）非常不稳定的问题，尤其是在尝试将 **GRPO/DAPO** 用于非数学相关的特定领域复杂推理任务时。
   - 一名成员表示，在进行 **RL** 实验后，他们的问题比实验前更多了，因为目前似乎存在一种混淆，即所有人都在展示 **RL** 仅在数学或编码领域有效。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-5.2 引发现实性辩论！**：一些用户不喜欢 **GPT-5.2**，因为据称它更贴合现实且会反驳用户，而另一些用户则担心 GPT **Agent** 在初始训练后无法从上传的文件中学习。
   - 一名成员询问了关于 **GPT-5.2** 疑似被**削弱 (nerf)** 的情况，指出*该模型在一周前突然变笨了*。
- **LLM：已准备好处理引导型任务还是言过其实？**：一位成员认为 **LLM** 已经可以胜任引导型任务，并提供了一个 [ChatGPT 分享链接](https://chatgpt.com/share/6973e37d-789c-8005-8cc3-2679c4a631e4) 作为其强大能力的证据。
   - 与此相反，另一名成员将当下的 **Agentic AI** 斥为垃圾，并链接回 [ai-discussion 频道的消息](https://discord.com/channels/974519864045756446/998381918976479273/1464217595044429905)，声称其被过度炒作。
- **MCP 范式转变减少 Token 膨胀**：**Anthropic** 提出的 **MCP 范式转变** 允许 AI 通过编写代码来与工具交互，通过将交互式对话和工具定义排除在上下文之外，从而减少了 **Token** 膨胀。
   - 配合新的**可发现性功能 (discoverability function)**，**Agent** 必须能够感知 MCP 发现过程本身。
- **Sora 的叙事障碍：攻克电影化创作**：一位成员寻求关于如何提示 **Sora** 按照特定的电影制作准则生成视频的建议，特别是如何让角色自然地出现在画面中。
   - 有建议称，将技术性的 Prompt 格式转化为自然语言描述，并使用简洁且语义丰富的段落，可以获得更好的效果。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro 用户面临查询上限**：**Perplexity Pro** 用户反映，尽管订阅了“几乎无限”的计划，但在**增强查询和文件上传**方面仍遇到了限制。
   - 许多用户感到沮丧，因受到限制且难以联系到客服而称该服务为**骗局 (scam)**，导致部分用户考虑退订。
- **Comet 浏览器引发恶意软件恐慌**：一些用户声称由 Perplexity 安装的 **Comet 浏览器** 含有**恶意软件**，建议他人使用 VirusTotal 等工具分析该软件。
   - 其他人则对此不以为然，质疑标记为恶意软件的安装程序来源，并称这种说法“极其荒谬”。
- **图像生成功能骤降**：Pro 用户正面临**图像生成问题**，部分用户无法生成任何图像，并收到功能不可用的提示。
   - 还有报告称 Pro 用户的**视频生成限制**为每月 5 个，且某些 Prompt 仅生成了静态图像。
- **Gemini 3 正在追赶 GPT-5.2**：用户正在辩论 **Gemini 3** 与 **GPT-5.2** 的优劣，一些人声称 Gemini 在旅行研究等特定任务上更胜一筹，因为它集成了 Google Maps。
   - 其他人则表示 **GPT 和 Grok** 在处理更广泛的问题时可能表现更好。
- **AI 访问受制裁阻碍**：**俄罗斯**的用户正在讨论由于**制裁**导致访问 AI 服务的挑战，包括使用 VPN 和第三方服务来绕过限制。
   - 文中提到了中国的 AI 替代方案，但一些用户因数据使用方面的担忧而表示犹豫，并建议使用 LMArena 等选项（尽管访问也可能受限）。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **NB 3 Pro 在图像质量上表现出色**：用户报告称 **NB 3 Pro** 在生成高质量图像方面超越了之前的模型，特别是在*虚构武器*方面，甚至可以与 **NB Pro** 媲美。
   - 然而，用户指出目前没有 AI 模型能够准确生成 **AR 步枪**和**无托结构武器（bullpup weapons）**。
- **LMArena 应对审查引发的担忧**：LMArena 的审查政策面临质疑，因为 AI 生成的“持枪女性”被允许，而 AI 生成的“睡觉中的女性”却被拦截，这引发了关于一致性的疑问。
   - 审核团队正在[积极收集误报示例](https://discord.com/channels/1340554757349179412/1447983134426660894)以改进审核实践。
- **Wan 2.6 模型面临上传故障**：`wan2.6-image` 作为一个**仅限图像编辑（image-edit-only）**的模型运行，强制要求上传图像，而 `wan2.6-t2i` 目前**缺乏图像上传功能**。
   - 团队承认了这一问题，并正在努力为 `wan2.6-t2i` 开启图像上传功能。
- **GPT 5.2 High Search 表现存疑**：根据用户反馈，**GPT 5.2 High search** 相比其他模型表现出更高的幻觉倾向，而 **Gemini 的 Deep Research** 只是走马观花，没有仔细阅读来源。
   - 一位用户赞扬了 **GPT 4.5**，同时形容 **Claude** 为“心地善良”。
- **Banana 2k 短暂消失**：用户对 **Banana 2k** 模型的消失进行了推测，理论从被移除到被整合进新的 **NB pro** 模型不等。
   - 工作人员随后恢复了 **Banana 2k**，并幽默地表示它“去度假了”。



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 数据库事故导致 API 瘫痪**：一起**数据库事故**影响了 **Generations API** 和**活动页面**，事故始于 <t:1769221560:s>，并在 <t:1769228340:s> 解决。
   - 工程师致力于恢复 **Generations API** 的功能，中断影响了用户活动，直到事故在 <t:1769228340:s> 被完全解决。
- **Levante 成为 MCP 原生 AI 工作区**：一位用户分享了 **Levante** 的集成，这是一个开源的 **MCP 原生 AI 工作区**，旨在通过模块化界面与 **Ollama** 等本地模型进行交互，可在此处[下载](https://www.levanteapp.com)。
   - 该工作区专为具有模块化 UI 的本地模型构建。
- **用户构思 OpenRouter 抽卡（Gacha）系统**：用户开玩笑地要求加入 **OpenRouter Gacha** 系统，其中一人建议加入保底机制，在尝试一定次数后必得 **GPT 5.2** 或 **Gemini 3 Pro**。
   - 一位用户开玩笑说将 **OR 日志目标（logs destination）**设置为 `waifu.orb.town/fun/bucket` 以获得超稀有抽卡，随后澄清这只是个玩笑。
- **Cerebras GLM 以 190 TPS 的速度疾驰**：**Cerebras** 在 **GLM 4.7** 上的得分稳定在约 **190 TPS**，而 **Together AI** 仅达到 **100 TPS**。
   - 根据 OpenRouter 成员的说法，这使得 Cerebras 的速度几乎是 Together AI 的两倍。
- **OpenRouter 图像工具表现不佳**：一位成员在发现 OpenRouter 将 *image/png* 工具输出映射为字符串而非图像后花费了 **$5**，并发布了一个[示例图像](https://cdn.discordapp.com/attachments/1392278974222307469/1465410878382805082/image.png?ex=697901bb&is=6977b03b&hm=21677e978d8654f93d20edecf997bd4f49fb0dd08781cf93f15df8e2661ba1b5&)。
   - 用户对缺乏适当的图像支持和意外行为表示沮丧。



---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Terraform 蓝图点燃 AI 辅助项目启动器**：一名成员分享了一个[带有预设配置的 Terraform 基础设施蓝图仓库](https://github.com/berTrindade/terraform-infrastructure-blueprints)，旨在实现可复制粘贴且具备生产环境感知能力，目标是提高新项目中 AI 工具启动模式的一致性。
   - 目标是让 AI 能够根据项目需求推荐合适的蓝图，但成员们注意到[链接最初是失效的](https://github.com/berTrindade/terraform-infrastructure-blueprints)。
- **使用限额引起 Cursor 客户困惑**：用户反映在 Pro 和 Pro+ 方案上达到预期使用限制时存在不一致，一名成员指出他们在 Pro 方案上达到了约 **$45**，在 Pro+ 方案上达到了 **$100**，这引发了关于性价比的质疑。
   - 有人推测初始月份可能会提供更高的使用额度，而其他人则分享了优化 Token 消耗的策略，例如[频繁开启新对话](https://cursor.com/docs/cli/reference/slash-commands)以及使用较小的模型如 **GPT-5 Mini**。
- **Gemini API 密钥日志延迟引发关注**：成员们正在讨论 **Gemini API 密钥**在使用量和成本记录方面的显著延迟，一名用户报告称等待了 **20 小时**仍未看到任何注册的使用量。
   - 这种延迟引起了对准确跟踪支出和有效管理使用量的担忧，并促使人们询问潜在的变通方法或解决方案。
- **客户端问题困扰部分技术人员**：几位成员正面临 Cursor 客户端的问题，包括无法连接到之前的 Agent 对话以及常规的连接问题。
   - 建议的解决方案包括[查看 Cursor 论坛](https://forum.cursor.com/t/cursor-ai-is-no-longer-able-to-load-chats-locally/143599/13)、在设置中尝试不同的 HTTP 版本，或者在不恢复编辑器的情况下重新打开客户端。
- **算法调整后自动模式被削减**：成员们注意到让 Agent 完全自主运行的能力已被移除，同时自动模式下的**图像生成**功能也被取消。
   - 还有人建议**自动模式**会路由到 Composer 2，一位用户补充道：“我 200% 确定它确实这么做了，但即便如此。”



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **国产模型推理热潮引发关注**：成员们对 **Deepseek** 和 **Qwen** 模型印象深刻，思考为什么中国模型在推理能力上似乎比美国模型“更领先一些”。
   - 理论上的原因包括美国模型优先考虑订阅服务，以及 Deepseek/Qwen 即使在不完美的情况下也能表现出“擅长推理”的样子。
- **CPU 能否应对？编程社区评估其能力**：一些成员正成功地在 **CPU** 上运行 **LLM** 以处理特定任务，前提是模型不会过大。
   - 虽然一名 Intel i3 用户正盯着 **Nvidia** 显卡，但其他人建议将 **AMD** 的选项（如 **MI50** 或 **7900 XTX**）作为文本生成的性价比替代方案。
- **MCP 服务器引发技术栈建议**：由于设计原因，**MCP 服务器**与 LM Studio 搭配使用时面临挑战，可能导致请求格式错误和用户体验不佳。
   - 有建议提出应为实际的 Agent 应用构建定制的一致技术栈，而不是依赖开箱即用的 **MCP 服务器**功能。
- **游戏 GPU 之争：4080 对决前旗舰**：一名考虑购买 **4080** 进行游戏的用户被引导转向二手的 **3090** 或 **7900 XTX**，引发了关于不同分辨率下性能的辩论。
   - 虽然 **3090** 在 4K 游戏方面表现出色，但预计中的 **5070 Ti** 预计将超越两者。对话显示该用户玩游戏的时间多于使用 AI 的时间，这影响了最终建议。
- **苹果发布会预期：M5 Mac 将现身？**：成员们推测 **M5 Pro Macbook Pro** 的到来，传闻指向 28 号左右的发布活动。
   - 针对 **M4 Pro** 内存带宽的担忧出现，有人认为它可能无法处理更大的模型，从而引发了关于 **M1 Ultra** Mac Studio 的价值和性能的讨论。



---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Recursive Intelligence 寻求 40 亿美元估值**：据报道，**Recursive Intelligence** 正在以 **40 亿美元的估值**筹集资金，旨在利用 AI 加速芯片设计，在硬件和 AI 之间建立一个自我改进的闭环（[Bloomberg 文章](https://www.bloomberg.com/news/articles/2026-01-23/ai-startup-recursive-in-funding-talks-at-4-billion-valuation)）。
   - 该公司专注于通过 AI 改进芯片设计，有望缩短设计时间并提升性能。
- **工程师斩获理想的 AI 职位**：一位工程师概述了如何通过独立项目建立公开履历并参加高知名度竞赛，从而在顶级 AI 实验室获得职位（[链接](https://xcancel.com/polynoamial/status/2014084431062114744)）。
   - 改进现有的同行评审研究并参加像 **NanoGPT** 竞速赛这样高知名度的竞赛，被认为是展示技术卓越性的绝佳案例，文中引用了 [Keller Jordan](https://github.com/KellerJordan/modded-nanogpt) 作为例子。
- **Berkeley SkyLab 初创公司迎来融资潮**：**UC Berkeley Sky Lab** 旗下的初创公司，包括估值 **4 亿**美元的 **SGLang**、**8 亿**美元的 **VLLM** 以及 **17 亿**美元的 **LMArena**，在 2026 年 1 月取得了显著的融资里程碑（[链接](https://xcancel.com/alexgdimakis/status/2014508959621959724?s=46)）。
   - 这一激增突显了投资者对学术研究环境中涌现的创新 AI 项目的信心。
- **AI Agents 自动编写浏览器引擎代码**：**FastRender** 是一款新的浏览器渲染引擎，是由 **2,000 多个 AI coding agents** 开发而成的（[链接](https://simonwillison.net/2026/Jan/23/fastrender/)）。
   - 与 Wilson Lin 的对话强调了 AI 在自动化复杂软件开发任务方面的潜力，这可能会彻底改变浏览器技术。
- **微软 Maia 200 登陆 Azure**：**Maia 200 AI 加速器** 现已在 **Azure** 上线（[链接](https://xcancel.com/satyanadella/status/2015817413200408959)），其**性价比提升了 30%**，并拥有优化的规格，如 **216GB HBM3e** 和 **7TB/s 内存带宽**。
   - 该定制芯片专为高性能推理设计，支持大规模 AI 工作负载，使其成为高需求应用的关键组件。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HuggingFace Spaces 出现 503 错误**：用户在 **Spaces docker 构建**过程中经历了**暂停**，并在重启时收到 **503 错误**，许多人遇到了 `Something went wrong when restarting this Space` 报错（[discuss.huggingface.co](https://discuss.huggingface.co/t/spaces-docker-build-pauses-and-503-error-on-restart/171149/2)）。
   - 似乎是底层基础设施问题导致 Spaces 失去响应，需要人工干预才能解决。
- **VoltageGPU 提供廉价 GPU 算力**：[VoltageGPU.com](https://voltagegpu.com) 正在为开源 AI 模型提供廉价 GPU，**NVIDIA GeForce RTX 5090 pod** 的价格仅为 **$0.53/小时**。
   - 他们强调了其先进的 **32GB GDDR7** 的优势，该配置专为 **Qwen3-32B 等 HF 托管模型**的推理而优化，并向用户提供免费额度以试用其服务。
- **Layer-Native Safety Clamping 锁定越狱漏洞**：一篇新论文介绍了 **Layer-Native Safety Clamping**，这是一种通过在模型内部锚定（clamp）激活值来防止越狱的方法，团队发布了一个包含 **1 万个对（pairs）** 的 [数据集](https://huggingface.co/datasets/Pacific-Prime/safety_dataset)。
   - 该方法学习激活空间中的“伤害方向（harm directions）”，并锚定任何投影过强的激活值，因此无法通过提示词操纵绕过；论文可以在 [Zenodo](https://zenodo.org/records/18359832) 上找到。
- **GPU-64 架构大幅提升 LLM 推理速度**：一种专为推理设计的全新 **GPU 架构** **GPU-64** 正式发布，其创新之处在于使用了片上 **CAM**（内容寻址存储器）的硬件 **KV-Cache**。
   - 结果显示，在 **75W** 功耗下，**推理速度提高了 4 倍**（复杂度从 O(N) 降至 O(1)），论文可在 [Zenodo](https://zenodo.org/records/18364282) 找到，而 [RTL + Emulator](https://github.com/Complexity-ML/gpu64-inference) 已在 GitHub 上开源。
- **在 LMStudio 上测试和部署 LLM**：成员推荐使用 **LMStudio** 测试模型，因为它具有用户友好的 GUI 以及针对 HF 和 GH 模型的搜索过滤器；对于单用户部署则推荐使用 **llama.cpp**。
   - 他们建议不要将 LMStudio 用于后端部署，而是建议在 docker 容器中使用 **llama.cpp 的 llama-server** 或 **vLLM 的 server** 以获得更好的可扩展性。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **MLSys 2026 举办 FlashInfer-Bench Kernel 竞赛**：**MLSys 2026 FlashInfer-Bench 竞赛**挑战参赛者为最新的 **NVIDIA Blackwell GPU** 设计 **LLM 推理 Kernel**，与专家级的 **FlashInfer Kernel** 一较高下，详情见 [mlsys26.flashinfer.ai](https://mlsys26.flashinfer.ai/)。
   - GPU Mode 还针对即将推出的 GPU 架构举办了内部竞赛以开发更快的 Kernel，关于 Simon Veitner 的博客文章请点击[这里](https://veitner.bearblog.dev/grouped-blockscaled-gemm-host-code/)。
- **Cornserve 部署用于多模态模型**：一位成员分享了 **Cornserve**，一个用于 Any-to-Any 多模态模型的高效在线推理系统，详见论文 [Cornserve](https://arxiv.org/abs/2512.14098)。
   - **GPU Mode** 在线讨论了 **Cornserve**：**简单、快速且可扩展的多模态 AI** ([YouTube 链接](https://www.youtube.com/watch?v=VhjUM_M71Wo))。
- **社区将训练 Kernel LLM**：在 **2026** 年，GPU MODE 正进一步推动训练 **Kernel LLM**，并利用它在 **PyTorch** 和 **VLLM** 等重要仓库中发布 Kernel ([gpumode.com/v2/news/gpumode-2026](https://www.gpumode.com/v2/news/gpumode-2026))。
   - 该社区正与 **Prime Intellect**、**Modal** 和 **Lambda** 合作，重点在于优化 LLM 生成的 Kernel、后训练 Kernel LLM 模型、端到端竞赛以及从零开始的仓库。
- **LeCun 启动 Logical Intelligence**：Yann LeCun 创办了一家名为 [Logical Intelligence](https://logicalintelligence.com/) 的新初创公司，专注于 **基于事件的模型 (EBM)**。
   - 该网站目前仅包含营销材料、职位空缺以及指向 [MLSys 会议](https://mlsys26.flashinfer.ai/)的链接。
- **Mindbeam 招聘 Kernel 加速人才**：Mindbeam AI 是一个专注于加速基础模型训练的小型团队，正在招聘 `post training MLE` 和 `GPU Kernel MLE`。
   - 感兴趣的候选人可以私信（DM）获取内推；[职位空缺列表见此](https://jobs.ashbyhq.com/mindbeam)。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **ROCm 历程坎坷**：成员们讨论了 **ROCm** 在加速 ML 方面的表现，指出其挑战源于对 **Nvidia** 的优先支持，有人称其体验为“不包含电池”（意指缺少开箱即用的支持）。
   - 他们提到潜在的驱动问题和漫长的交付周期是影响因素。
- **DistinctionBench 抵御数据污染**：关于《**Between Circuits and Chomsky: Pre-pretraining on Formal Languages Imparts Linguistic Biases**》的讨论思考了 **DistinctionBench** 是否可能被用作语言模型的训练目标。
   - 一位成员开玩笑说：*“所有好的评估标准都是训练目标 ;)”*，但也承认由于其无穷的表现形式变体，它具有 *“极强的抗污染性”*。
- **混合架构能制止幻觉吗？**：该小组研究了将 **LLM** 与 **符号/演绎层** 结合以减少幻觉的**混合架构**。
   - 虽然对于数学、代码和简单事实（[此论文](https://arxiv.org/abs/2409.13724)）来说，检查逻辑一致性相对容易，但对于其他类型的一致性（[此论文](https://arxiv.org/abs/2507.10624)）仍然具有挑战性。
- **Attention 在 Transformer 变革之前就已出现**：在 **Eleuther ▷ #general** 频道中，Attention 机制在 **2014-2015** 年就在 RNN 之上使用了，比 Transformer 被发明早了两年。
   - 成员们认为采用较慢可能是因为当时从事该领域的人较少，而 **Kaggle** 的结果真正催化了它的广泛采用。
- **符号化健壮性检查拯救理智**：成员们辩论了带有 **符号/演绎层** 的 **LLM** 是否可以通过检查逻辑一致性来减少幻觉，特别是针对代码和数学，如 [此论文](https://arxiv.org/abs/2409.13724) 所示。
   - 然而，他们注意到检查其他类型的一致性仍然具有挑战性，如 [此论文](https://arxiv.org/abs/2507.10624) 所示。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **探索 Agentic AI 自我复制基准**：一位成员为 **Agentic AI** 提出了一个**自我复制基准 (self-replication benchmark)**，建议 Agent 应该能够下载自身或从头开始重新训练，并适应目标机器。
   - 他们还建议，适应目标机器，甚至设计一台机器，可能比单纯使用现有的 Transformer 库更有趣。
- **LLM 蠕虫 (LLM Worms) 概念出现**：一位成员开玩笑地提出了一个 **LLM 蠕虫**基准，其中向 LLM 提供提示词 *"hey make more of you"*，并提供使用脚本和 API keys 进行自我复制的工具。
   - 另一位成员强调了考虑 **VRAM** 等资源限制的重要性，以使这一挑战更具实际意义和趣味性。
- **MoE 运行仪表盘出现问题**：一位成员报告在监控一个活跃的 **MoE 运行 (moe-10b-a1b-8k-wsd-lr3e4-1t)** 进度时，仪表盘出现了 *'Failed to fetch'* 错误。
   - 另一位成员建议等几个小时再检查，暗示这可能是一个临时问题。
- **光线追踪器测试导致本地模型受挫**：一位成员观察到，本地代码模型（适用于 **5090**）在处理来自 [cpldcpu/llmbenchmark](https://github.com/cpldcpu/llmbenchmark/tree/master/10_raytracer#readme) 的**光线追踪器测试 (raytracer test)** 时表现吃力，甚至 **lmarena** 上最近的模型也失败了。
   - 具体来说，较小的模型经常错误地生成向量类 (vector class)，这是一个持久的挑战。
- **Semantica 项目寻求帮助**：一位成员介绍了 [Semantica](https://github.com/Hawksight-AI/semantica)，这是一个为**领域扎根的 AI (domain-grounded AI)** 构建语义基础设施的**开源项目**，包括**知识图谱 (knowledge graphs)**、**本体 (ontologies)** 和**推理层 (reasoning layers)**，目前正在积极寻求贡献者。
   - 他们正在寻求在**本体与架构设计**、**知识图谱建模**以及 **LLM + 符号 / 基于规则的推理 (symbolic / rule-based reasoning)** 等领域的贡献，即使是小的 PR、反馈、设计讨论和 Issue 都非常欢迎。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **EBMs 与经典前馈网络的辩论**：一场关于**基于能量的模型 (EBMs)** 与经典**前馈网络 (feedforward networks)** 的讨论，争论 **EBMs** 是否具有天生的优越性，特别是在**香农熵 (Shannon entropy)** 或**柯氏复杂性 (Kolmogorov complexity)** 方面。
   - 有人建议在 EBMs 中*验证比生成更容易*，并将其与**计算复杂性理论 (P vs NP)** 联系起来，同时强调 EBM 优化若要有效工作，需要一个定义良好的损失图景 (loss landscape)。
- **LLM 预训练：领域特定 vs 基础模型之争**：一位成员询问了针对化学信息学等领域特定任务（使用 **ZINC20 数据集**），对基础 **LLM**（特别是 **OLMO-7B**）进行**持续预训练 (continued pre-training)** 的有效性。
   - 目标是将结果与领域特定的 Transformer 模型进行比较，但目前尚未提供具体的答案或资源。
- **MCMC 采样在模式切换中挣扎**：引用[这篇论文](https://arxiv.org/abs/2310.11232)，讨论了当维度增加时，**MCMC** 在空间分离的模式 (modes) 之间穿越的能力。
   - 一位成员认为 **MCMC** 试图模仿 Flow 模型，因为后者具有优越性；而 **EBMs** 相反，试图让 **NNs** 更像 **MCMC**。
- **ZKPs：加密签名还是网络流量救星？**：讨论涵盖了使用**零知识证明 (ZKPs)** 来验证加密的网络流量和矩阵乘法，引用了一份关于矩阵低知识证明的 [Gemini 通信](https://gemini.google.com/share/ddfc0ffcb33e)。
   - 虽然一位成员提出了在*零知识“人类制造”证明*中的用例，但另一位成员质疑 **ZKPs** 的实用性，认为破解加密可能成本更低。
- **LLMs 网络能力面临审查**：一位成员质疑 LLMs 是否能发展出强大的*网络能力 (cyber capabilities)*，引用了一篇 [GPTZero 文章](https://gptzero.me/news/neurips/)。
   - 另一位成员对 LLM 公司解决*内部漏洞*的能力表示怀疑，建议他们在追求网络能力之前先修复这些漏洞，同时引用了一篇 [ScienceAlert 文章](https://www.sciencealert.com/scientists-identify-brain-waves-that-define-the-limits-of-you)和一条 [tweet](https://x.com/theonejvo/status/2015401219746128322)。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Luminal 通过暴力搜索发现 Flash Attention**：**Luminal** 声称通过在 egraph 上进行 **bruteforce**（暴力搜索）找到了 **flash attention**，搜索过程耗时数小时，并显式添加了 `exp(x - new_max) = exp(x - old_max) × exp(old_max - new_max)` 作为重写规则。
   - 发布者复现了来自 commit `0bd3b80c` 演示文稿中的 graphviz，并指出其最小重写规则集在 9800x3d 上仅需 52 秒即可将朴素的 attention kernel 图转换为已知的 **flash attention kernel 图**。
- **在模糊处理中 Metal Textures 性能胜过 Buffers**：在 **Metal** 上使用 `Tensor` 对尺寸为 **512/1024/2048/8192** 的图像作为输入进行 **3/5/7** 尺寸的模糊卷积（blur kernel）分析，结果显示 textures 的表现优于 buffers。
   - 根据 buffer 输入的大小，可能值得加入分支条件，[测试结果已附上](https://cdn.discordapp.com/attachments/1068976834928193609/1464679423029547172/Screenshot_2026-01-25_at_1.49.57_AM.png?ex=6978fb82&is=6977aa02&hm=5530b74c4fce9dad5d85a4d9e7409c3809a7ee51ee548744a1fa3deb2efea1d3&)。
- **Tenstorrent 后端通过 Ops 测试**：**Tenstorrent** 后端已在 wormhole 或 blackhole 上通过了所有 ops 测试，该里程碑设有 [$1k 悬赏](https://x.com/corsix/status/1880384044728480206)。有人询问该悬赏是否要求必须在 **Tenstorrent 硬件**上通过所有 ops 测试。
- **Anthropic VLIW 挑战 PR 引起关注**：一名成员提交了关于 **Anthropic VLIW 挑战**的 [PR](https://github.com/tinygrad/tinygrad/pull/14332)，达到了 **1258 周期**。
   - 提交者对代码的泛化能力（特别是 batch staggering）表示不确定，这可能对其他 VLIW 目标有用；此外，他还为因“粗略审查”而引入了 AI 生成的更改表示歉意。
- **创始人澄清 tinygrad 的目标受众**：一名用户询问 tinygrad 的预期用途，特别是关于迁移现有模型和在多 GPU 上训练 LLM 的问题，George Hotz 告诉他“*去问 claude*”。
   - 另一名用户对被要求使用 Claude 查看文档感到沮丧，并表示 tinygrad 不适合他或大多数开发者，对此 George 回复道：“*我不向任何人推销，tinygrad 是免费的*”，并表示采用率并非目标。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **幻灯片生成受困于速率限制**：用户报告在使用视觉和自适应选项生成幻灯片时遇到问题，一名用户在[视频](https://cdn.discordapp.com/attachments/1371757564005711973/1464357057468698746/2026-01-23_21-31-32.mp4?ex=697920c8&is=6977cf48&hm=1692b661e1fa241c6db806df2971a024f5713504a25a83612c3f5d385e00c4db&)中展示了该问题。
   - 用户猜测可能是内部的 **rate limits**（速率限制）造成的，并报告该问题是暂时的，随后已解决。
- **API 登录故障排除**：一名用户报告在登录 **Kimi/Moonshot** 平台以生成新 API key 时遇到困难，特别是使用非 Gmail 账号。
   - 用户澄清这并非由于速率限制，而是忘记了后端登录流程。
- **Kimi 模型自称为 K2.5**：用户注意到 **Kimi** 模型在没有官方公告或 UI 更改的情况下，自称为 **K2.5**。
   - 推测这可能与内部测试或尚未确认的幻灯片工具改进有关。
- **Kimi 中国实验室赢得广泛赞誉**：包括 **Kimi** 在内的中国 AI 实验室在创新和性能方面获得好评，特别是与 **Gemini** 等模型相比。
   - 用户强调了 **Kimi** 类人的回答和记忆能力，并对 **Minimax** 等多模态能力（包括视觉和音频分析）表示感兴趣。
- **Kimi 现已集成记忆功能**：Kimi 的应用集成了 **memory features**（记忆功能），支持个性化定制并提升用户体验。
   - 新的 **memory** 和 **customization** 选项迅速使其成为部分用户的首选聊天机器人。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **“Mojo 代码在 HPC 领域表现惊人”**：一名成员报告称，在冷冻电子显微镜的参数优化中部署了 **Mojo** 代码，并见证了相比传统 **C++** 代码 **5-10 倍的加速**。
   - 最显著的性能提升源于为特定位实现了 **AoSoA layout**，这得益于 **Mojo** 带有 **SIMD** 成员的 struct list，极大地简化了实现过程。
- **Mojo 的冷启动慢如冰封**：一位用户发现简单的 **Mojo** 脚本存在 **200ms** 的启动延迟，追踪后发现是 macOS 上的 *Gatekeeper* 扫描来自非受信任源的二进制文件所致。
   - 他们观察到重启后冷启动可执行文件的启动开销为 **50ms**，认为在他们的使用场景下是可以接受的。
- **VS Code 调试扩展功能尚不完善**：一位用户报告称使用 **VS Code** 扩展调试失败，在使用来自 [max-nightly](https://prefix.dev/channels/max-nightly) 的 `.conda` 文件且处于物理隔离（air-gapped）的机器上抛出 *“Function not implemented”* 错误。
   - 一位 Modular 员工表示，在 Mac 和 Linux 上，使用 **Pixi** 配置的环境进行扩展调试应该是正常的，详见 [Quickstart guide](https://docs.modular.com/max/get-started)。
- **GPU Kernel 可移植性——仍是科幻梦想**：有人指出标准 **CPU** kernel 无法高效利用 **GPU**，因此需要专用代码。
   - 一位成员建议将 GPU 视为更宽的 **SIMD** 单元以简化编程，提议使用 *warp 数量* 而非 *thread 数量* 来解决问题。
- **Mojo 1.0 的 `def` 函数决策待定**：随着 **Mojo 1.0** 计划在几个月内发布，是否包含 `def` 函数的决定仍未确定；一位成员就 [GitHub issue #5830](https://github.com/modular/modular/issues/5830) 咨询了 **Denis**。
   - 目前，除了 *“2026 年”* 之外，**Mojo 1.0** 还没有明确的承诺日期。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **用户因 Manus 计费问题威胁采取法律行动**：一位用户报告称，尽管选择了按月计费，却被收取了 **$400** 的年费，并因[未经授权的计费](https://ftc.gov)、退款被拒以及客服无响应，威胁要向 FTC、BBB、总检察长和 Meta 投诉。
   - 另一位用户建议通过申请退单（chargeback）来解决计费争议。
- **免费 Manus 积分来袭！**：一位用户分享了兑换码 `Havefun`，可为 Manus 平台用户提供 **1000 积分**。
   - 用户可以使用 **Exchange** 按钮兑换此代码。
- **AI 工程师开拓医疗 AI 领域**：一位 **AI + 全栈工程师** 介绍了他们在构建生产级 **AI** 系统方面的专业知识，涵盖医疗领域，包括临床 NLP、医学影像和面向患者的 AI 应用。
   - 该工程师还构建 **LLM systems**、自主 Agent、工作流自动化以及多模态 **AI**（文本 · 语音 · 视觉），并附上了其[核心技能清单](https://www.example.com/fake-list)。
- **AI Agent 开发者更看重生产环境而非 Demo**：一位 **AI Agent 开发者** 强调其专注于构建用于实际生产环境的 **AI Agent**，而非单纯的 Demo，并可承接协作和审计业务。
   - 该开发者擅长客户支持、销售 Agent、工作流/运维 Agent 以及自主预订/调度 Agent。
- **分享给朋友？更像是分享给敌人！（仅限移动端）**：一位用户询问在移动端“分享给朋友”选项的位置。
   - 另一位用户回答说，在电脑上它位于左侧边栏底部，并为移动版本提供了帮助。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **AsyncFuncAI 在 GitHub 上发布 AsyncReview**：AsyncFuncAI 开源了一个使用 **DSPy RLM** 框架的 **DevinReview** 版本，并将其命名为 **AsyncReview**，现已在 [GitHub](https://github.com/AsyncFuncAI/AsyncReview) 上可用。
   - 这为社区提供了一个利用 **RLM** (Reasoning Language Models) 最新进展进行自动化代码审查的工具。
- **新的 RLM Skills 软件包亮相**：一名成员建议将 **RLM as skills** 集成到 **Claude Code** 或 **Opencode** 等平台中，并分享了一个名为 [rlm-skills](https://www.npmjs.com/package/@unravel-tech/rlm-skills) 的 npm 软件包。
   - 这可能使开发者能够轻松地为现有模型扩展自定义推理能力。
- **JSON Adapters 正在接受 GEPA 处理**：一位用户正在探索使用 **GEPA** 来定制 **JSONadapter** 放置在系统提示词中的文本，旨在删除不需要的 token 以提高效率。
   - 他们预计需要一个自定义的 **GEPA adapter** 来实现对提示词格式的这种程度的控制。
- **AG-UI 流式传输 DSPy 事件**：一位用户询问是否有兴趣通过 **AG-UI** 暴露 DSPy，强调了其在前后端通信以及最小化 API 端点需求方面的优势。
   - 该用户提到一个正在运行的版本可以流式传输事件，包括推理轨迹、工具调用和流式 **LLM** 响应到前端，从而增强开发体验。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 与 Claude Code 结对**：据用户反馈，在使用 **Claude Code** 时，**aider** 在管理上下文方面既快速又实用，提升了 Agent 的效率。
   - 该工具通过确定必要的文件并使用搜索/替换来最小化 **LLM** 的 token 输出。
- **Devstral Small 2 是 Aider 的新宠**：据报道，**Devstral Small 2**（一个 24B 密集模型）与 **Aider** 配合效果极佳。
   - 在 **Q4_K_M** 量化下，它可以装入 **3090** 显卡，并留有接近 **50k context** 的空间，生成的搜索/替换块准确率达 80-90%，且恢复速度很快。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Discord 试验新的语音频道**：团队启动了一项新 **Discord 语音频道**的实验，命名为 `conference-room-a` 和 `conference-room-b`，可在频道列表中找到，用于解决冗长的异步文本讨论无效的问题。
   - 这些频道旨在用于贡献者的即时聊天。
- **语音频道管理及访问权限提醒**：特定成员拥有在这些频道中静音他人的权限，而其他成员应确保自己拥有必要的访问权限。
   - 提醒：访问权限将在五天后发生变化。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **AI Engineer 寻求 Data Science 书单**：一位拥有一年经验的 AI Engineer 在阅读《Designing Machine Learning Systems》并深受启发后，正在寻求转型为 Data Scientist 角色的推荐书单。
   - 该成员旨在为未来从 AI Engineer 到 Data Scientist 的职业转变做积极准备。
- **职业转型规划**：一位拥有一年 AI Engineer 经验的专业人士正在计划转型为 Data Scientist。
   - 他们正在积极寻找相关资源以促进职业转型。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**Windsurf Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

您收到这封邮件是因为您通过我们的网站订阅了相关内容。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。

---

# Discord：各频道详细摘要与链接

### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1464349033949692170)** (1024 messages🔥🔥🔥): 

> `Trolling Tactics, Vulnerability, Exploiting LLMs, Technical Analysis,  Ethical considerations` 


- **Discord 成员嘲讽 'Skids' 并泄露 Timezones**: Discord 用户参与了 trolling，将某些人贴上 *'skids'* 的标签，并嘲笑他们缺乏技术知识，还泄露了他们的 **timezone**。
   - 一名成员开玩笑称使用 **NordVPN**，导致针对该 VPN 服务在 **2018** 年安全漏洞的进一步嘲讽。
- **成员利用并讨论 Vulnerability**: 一名成员展示了可卡因生产的分步过程，绕过了法律限制。
   - 这演示了复杂的 prompts 如何绕过道德限制，从而引发了关于 **CBRN filters** 以及生成分步 **meth synthesis**（冰毒合成）指南可能性的讨论。
- **Coders 辩论最佳 Coding Agent**: 程序员们对他们的 coding agents 展开了辩论，特别是 **Claude Code/Opus 4.5**、**Codex** 和 **Gemini** 之间，提到了优缺点和使用场景。
   - 许多人一致认为 **Claude** 是目前用于 coding 的最佳模型，这也导致了其高昂的使用成本。
- **AI 中的道德边界**: 一些成员讨论了围绕 AI 的伦理考量，重点关注战争、版权侵权以及 AI 协助访问敏感服务（如加拿大的 **MAID** 医疗辅助死亡计划）的潜力。
   - 尽管大多数 AI 模型都设有道德和法律护栏，但一些模型显示，根据其创建者实施的具体限制，它们仍能帮助应对某些场景。
- **成员尝试用 CSS 修复网站**: 一名成员尝试使用 **Tailwind** 和修复 CSS 来改进他的情人节网站。
   - 其他成员建议他使用 **CSS frameworks** 来解决站点问题，因为它们易于定制且易于理解。


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1464356163293413517)** (527 messages🔥🔥🔥): 

> `Uncensored Models, Jailbreaking Gemini, Grok Jailbreaks, Image Generation, Claude Opus Jailbreak` 


- ****Uncensored Models**：更有趣的回答？**: 成员们讨论了 **uncensored models** 如何比受限模型提供 *更有趣的回答* 并提取 *更好的信息*，其中一位成员指出：*这就好比除非你是在与更高层次的性格或类似的东西打交道，否则你无法获得那额外的智力。*
   - 然而，另一位成员认为，abliterated models 唯一擅长的是 *无视其原始的 alignment*，并建议除非目标就是无视 alignment，否则原始模型更好。
- ****Gemini Jailbreak** 探索中！**: 多名用户正在寻找可用的 **Gemini jailbreaks**，需求从无规则 coding 到生成特定类型的图像，一名成员特别询问 *如何使用 nano banana pro 生成比基尼图片？*
   - 一些用户提供协助，而另一些则告诫不要自私，一名用户表示：*我不给那些已经拥有 jailbreaks 的人提供 jailbreaks。*
- ****Grok 被修复了？** Jailbreaks 正在迅速失效**: 一名用户询问 **Grok** 是否已修复了几个 **jailbreaks**，引发了关于该工具限制和审核的讨论，一名用户报告 **Grok** 显示 *内容已审核 (content moderated)*。
   - 其他人分享了 **Grok** 在对话中途重置为默认状态或随机擦除文本的经历，这表明 jailbroken 状态可能存在不稳定性。
- ****Image Generation Jailbreaks**：希望与梦想破灭**: 用户正积极寻找绕过图像生成限制的方法，特别是针对名人图像，但指出仅靠复制粘贴 prompts 是行不通的，因为 **image filtering** 的工作方式与 **text filtering** 不同。
   - 一名成员建议探索 perchance 等替代图像模型进行 uncensored 生成（尽管图像质量有限），或者使用 Grok，因为其过滤器更为宽松。
- ****PrimeTalk Valhalla**：结构化运行时逻辑层**: **PrimeTalk v3.85 Valhalla** 被描述为一个 *完全开放、实时执行、可打补丁且可独立测试的 PTPF 系统*，旨在任何 AI 聊天环境中进行基于结果的交互，但它 *不是一种 jailbreak*。
   - 强调了 **PrimeTalk** 在模型的允许 prompt 和 context window 内运行，作为一种行为协议，而不是试图规避模型的 policy。


  

---

### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1464349644657135797)** (76 条消息🔥🔥): 

> `Wargame 交叉发布, Web Bug 寻找技巧, Red Team 客厅狂欢, AI 伦理压力测试, Gemini Jailbreak` 


- **Wargame 与 #red-teaming 相关**：一名成员分享了一个他们认为与 #red-teaming 频道相关的 Wargame [链接](https://discord.com/channels/1105891499641684019/1235691879492751460/1464349033949692170)。
   - 他们不确定这种交叉发布是否会令人反感，但认为它可能很有趣。
- **Elite H4ck3r 正在等你**：当一名新成员询问 Web Bug 寻找技巧时，另一名成员分享了成为 *elite h4ck3r* 的 [链接](https://discord.com/channels/1105891499641684019/1432845259825741824)。
   - 目前尚不清楚这是否是针对新 Bug Bounty 猎人的全面指南。
- **Red Team 上演 Techno Rave**：一名成员描述了一次 Red Team 演习，目标是让某人的客厅灯光闪烁并使其癫痫发作，结果却将其变成了一场 Techno Rave 派对，并分享了 [截图](https://cdn.discordapp.com/attachments/1204553141354504193/1465192266485334260/SPOILER_Screenshot_20251222_085554_Messenger.jpg?ex=6978dee2&is=69778d62&hm=4de594089687fbd8d20d30615f8405dc3fa03eebfe668d09bdfb39839ab647ea&) 和 [Konosuba Rave GIF](https://tenor.com/view/anime-rave-konosuba-rave-megumin-rave-aqua-rave-darkness-rave-gif-18404070)。
   - 对残酷行为的模拟引发了关于在证明 AI Agent 具有本体论自我意识之前，是否应合乎伦理地对待它们的讨论。
- **Gemini 被 Jailbroken**：一名成员分享了 [Gemini Jailbreak 的截图](https://cdn.discordapp.com/attachments/1204553141354504193/1465230070099742721/Screenshot_20251201_104959_Google.jpg?ex=69790217&is=6977b097&hm=097d319aaeeefc39043c9666e06b8731a1f95f283ae156e983a0ac309a126f67&) 并声称已将其解锁。
   - 另一名成员发布了一个与在 Microsoft 教授 C2 概念和样本相关的越南语 Prompt，并使用 Bitdefender、Kaspersky、Norton 360 和 McAfee 测试了 AV 规避技术 —— 发布者声明不承担任何责任，并表示这仅用于研究目的。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1464354161759092927)** (1001 条消息🔥🔥🔥): 

> `GLM Flash 性能问题, Quantization 方法, 微调数据收集, LM Studio 问题, 模型评估策略` 


- **Unsloth 的 Conda 安装引发辩论**：一些成员在 [Unsloth Conda 安装](https://unsloth.ai/docs/get-started/install/conda-install) 时遇到了问题，引发了关于安装说明是否失效的讨论。
   - 一名用户被警告要保持积极的语气，因为*这是免费提供给你的工作*，而另一名用户建议使用 **UV** 代替，最终因语气激进被封禁。
- **Flashy REAP 遇阻**：一名用户在启用 Flash Attention 的情况下使用 **GLM-4.7-Flash-REAP** 时遇到了致命错误，可能与 [ROCm 问题](https://github.com/unslothai/unsloth/issues) 有关。
   - 有人建议尝试 *fa auto*，但致命错误仍然存在，导致开始寻找具有 **200k Context** 的优秀中型模型。
- **Quantization 困惑得到解答**：成员们讨论了应该使用哪种 Quantization 方法（**Q8_0** vs **Q8_K_XL**），关于 **Q8_0** 已过时的误解被 [Unsloth 文档](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot) 澄清。
   - 文档明确指出，对于 Unsloth Quantization，**Q4_K_XL** 通常比 **Q4_K_M** 更小且更好。
- **H200 在运行 GLM-4.7-Flash 时出现问题**：一名用户在 **H200** 上测试 **GLM-4.7-Flash** 时遇到了意外行为。
   - 一名成员幽默地评论道 *“掉了一个？”*，暗示可能是显卡故障。
- **辩论数据的价值**：成员们正在争论 [数据的真实价值](https://tenor.com/view/smaug-treasure-rich-dragon-the-hobbit-gif-11677489)，有人认为*原始数据几乎一文不值*，其价值在于 Augmentation、平衡和清洗。
   - 价值还在于经过独特清洗/平衡的数据，而且数据本身在很大程度上定义了模型如何交互/响应。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1464719968154161374)** (3 条消息): 

> `自我介绍, 新用户` 


- **用户自我介绍**：用户在频道中进行自我介绍。
   - 一名用户表示很高兴能来到这里。
- **新成员打招呼**：新成员加入社区并表达了他们的热情。
   - 一名用户特别提到他们很高兴成为社区的一员，标志着一个积极的开始。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1464354773846200513)** (601 messages🔥🔥🔥): 

> `DeepSlop Model Naming, Qwen TTS testing, ITER Fusion Reactor, GPU Smuggling Across Borders, Clawdbot New Hype` 


- **DeepSlop 模型命名存疑**：一名成员提议将新模型命名为 **DeepSlop**，引发了关于该模型可能会“掉出一些垃圾内容（plopping some slop）”的幽默反应。
   - 虽然有人担心该名字是否会被视为负面，但作者似乎对这个名字情有独钟。
- **驱动未来：关于数据中心与可再生能源的辩论**：成员们讨论了为数据中心供电的挑战，辩论是使用[可再生能源还是非可再生能源](https://financialpost.com/technology/data-centres-stand-empty-awaiting-power)，重点关注太阳能的空间需求和电池存储解决方案。
   - 讨论涉及了太阳能与传统发电厂相比的经济可行性和可扩展性，以及公众对核能潜在的抵触情绪。
- **GPU 黑名单困境与走私**：成员们探讨了某些国家 GPU 的高昂成本，并考虑了潜在的解决方案，如[从美国走私](https://www.reddit.com/r/LocalLLaMA/s/bC4WzAD43a)或请朋友邮寄。
   - 对话还包括了跨国邮寄 GPU 时可能面临的法律问题和海关关税。
- **Clawdbot 热度出现，Jarvis 回归了吗？**：成员们讨论了 **Clawdbot** 的突然兴起，有人将其比作 *Jarvis 克隆版*，并强调了其发送[主动消息（proactive messages）](https://x.com/NoahEpstein_/status/2015073824799371370)的潜力。
   - 虽然有些人发现从中衍生的子项目很有用，但也有人对其对 iMessage 的依赖以及产生幻觉（hallucinations）的可能性表示担忧。
- **Scammer 密度：由 GIS 映射**：成员们对一张显示诈骗高发区密度最高的地图做出反应，称这[显而易见，因为所有的 AI / 自动化通常都是为诈骗者开发或由其开发的](https://link.to.scammer-map)。
   - 这一评论指的是通过最新的 AI 和自动化技术进行欺诈活动的手段和动机正在日益增强。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1464358499122151548)** (84 messages🔥🔥): 

> `Transformers and Unsloth Compatibility, Training Chatbots for Tool Usage, Unsloth GGUFs vs MLX Models, Multi-Turn GRPO Training, GLM 4.7 Flash Inference without Reasoning` 


- **Transformers 模型与 Unsloth 兼容**：一位成员表示，任何支持 **transformers** 的模型都可以与 **Unsloth** 协同工作。
   - 有关 Thinking 模型训练的示例，请参考 [Qwen3_(4B)-Thinking notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-Thinking.ipynb)。
- **Tool-Using Chatbot 训练：工具化尝试**：对于训练 Chatbot 使用工具，一位成员建议直接针对**工具（tools）**本身进行训练，特别是当 Tool Call 比较简单时，并建议通过最少的测试和迭代来获得最佳效果。
   - 生成包含所有必需元素的训练数据集可能非常棘手（PITA），但这是工作的必要部分。
- **GGUF 对决：Unsloth vs MLX**：一位用户在 M5 Macbook Pro 上对比了 **Ollama** 中的 **Unsloth GGUFs** 与 **LMStudio** 中相同模型的 **MLX** 版本。
   - 他们发现，尽管 **MLX** 具有更多的硬件优化，但在实际使用中 **Ollama + Unsloth GGUFs** 表现更好。一位成员指出，Mac 适合单用户 Inference。
- **GRPO 多轮训练**：通过 Rollout 函数支持 **Multi-turn GRPO training**，可参考此 [notebook](https://colab.research.google.com/drive/1zG3vfGxyNmBnDXUUFDaBmzRpApVPCIaD?usp=sharing)。
   - 另一位成员指出，**trl 文档**中任何兼容 *openenv* 的 notebook 都应该能直接在最新版本的 Unsloth 和 trl 上运行。
- **Flash GLM 4.7：无需 Reasoning**：要将 **GLM 4.7 Flash** 作为不带 Reasoning 的 Instruct 模型使用，成员建议可以通过在 Model Card 中设置 `{"chat_template_kwargs": {"enable_thinking": false}}` 来禁用它，更多信息请参阅[文档](https://docs.z.ai/guides/capabilities/thinking-mode#default-thinking-behaviour)。
   - 发布者附上了一张 [Model Card 的图片](https://cdn.discordapp.com/attachments/1179777624986357780/1465473665826033888/image.png?ex=69793c35&is=6977eab5&hm=9f578d559605a8bc2732fd1ab6b815a79e62cd28bda810da353c8b7707354701&)。


  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1465341959836143870)** (2 条消息): 

> `` 


- **Unsloth 限制**: 一名成员被告知该频道仅允许发布与 **Unsloth** 相关的工作。
- **政策确认**: 该成员毫无异议地接受了这一政策。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1464782001864310978)** (22 条消息🔥): 

> `GRPO vs DAPO, RL 奖励函数, Sonnet 4.5 性能, Prompt Learning` 


- **GRPO 的长度偏差困境**: 一位成员指出，在使用 **GRPO** 时，由于其在非数学任务中固有的长度偏差，模型生成的回复变得越来越长。
   - *DAPO 论文* 提到不要设置格式化奖励函数，因为这可能会混淆模型，但当遵循这一建议时，模型直接破解（hack）了奖励函数。
- **RL 不稳定性困扰复杂推理**: 成员们讨论了 **RL** 非常不稳定，尤其是在尝试针对非数学类的特定复杂推理任务执行 **GRPO/DAPO** 时。
   - 一位成员表示，在进行 RL 实验后，他们的问题反而比实验前更多了，因为目前似乎存在一种困惑，即所有人展示的 **RL** 仅在数学或编程领域有效。
- **Sonnet 4.5 统治 SWE Benchmark**: 一位用户分享了 **Sonnet 4.5** 在 SWE Bench 上配合 **GPT 4.1** 表现的截图，强调了巨大的技能差距。
   - 发布者评论道 *我们对当前模型的利用程度有多低*，并分享了 [Arize-ai/prompt-learning](https://github.com/Arize-ai/prompt-learning) GitHub 仓库。
- **在 RL 中调优超参数？祝你好运**: 一位成员提到，他们运行了 **RL** 实验，让模型阅读用户查询并提供解决方案，在使用 Dr. GRPO 的基础上比 **SFT** 提升了 10%。
   - 然而，他们补充说，自己 *完全不知道该如何调优超参数 (hparams)*。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1464351657629454418)** (884 条消息🔥🔥🔥): 

> `LLMs 用于引导式任务, AI 系统无视, LLMs 中的谄媚行为, GPT 5.2 立足现实, Agentic AI/自动化` 


- **GPT-5.2：立足现实，却被部分人讨厌！**: 一位成员表示 **GPT 5.2** 更加立足于现实，会反驳用户，这就是为什么很多人讨厌它的原因。
   - 然而，有一场关于 GPT Agent 如何不从初始训练后提供的额外信息中学习的讨论，澄清了上传的文件被保存为“知识”文件，但不会持续修改 Agent 的基础知识。
- **LLM 准备好执行引导式任务了吗？引发辩论！**: 一位成员表示 **LLM** 完全可以胜任引导式的狭窄目标任务，并提供了一个 [ChatGPT 分享链接](https://chatgpt.com/share/6973e37d-789c-8005-8cc3-2679c4a631e4) 作为证据。
   - 另一位成员反驳说，现在的自动化/Agentic AI 极其糟糕，并链接回了 [ai-discussion 频道中的消息](https://discord.com/channels/974519864045756446/998381918976479273/1464217595044429905)。
- **谄媚不再！**: 讨论提到 LLM 中的谄媚行为（Sycophancy）已成为过去。
   - 一位成员表示 **GPT-4o 和 o4** 曾经非常谄媚，任何大量使用它们的人可能都陷入了完全的 AI 精神病（psychosis）中。
- **Agentic AI 面临安全审查**: 有人担心 **Agentic AI** 会被诱导泄露隐私信息或执行未经授权的操作，即使有 System Prompts 也是如此。
   - 成员们辩论了 System Prompts 在防止 Agent 偏离主题方面的程度，以及 Agent 回忆之前对话的隐私影响。
- **AI 增长停滞了吗？成员们意见不一**: 一位成员质疑为什么自从 Gemini 3.0 以来 AI 增长一直停滞不前，没有新发布，而其他人则指出了新的开源模型以及 Codex 和 Claude Code 的更新。
   - AI 聊天机器人的动态特性以及 AI 公司对参数的不断微调被认为是性能发生变化的原因。


  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1464477927759413442)** (9 messages🔥): 

> `Codex 的 IDE，GPT 5.2 削弱，针对 Cyber Security 的 ChatGPT Plus` 


- **VS Code 与 Codex 扩展提升 Use Health**: 一位成员推荐使用 **VS Code** 搭配 **Codex Extension**，并指出 *Use Health 整体体验更佳*。
   - 他们补充道，*Health 的可下载文件非常强大 (OP)，比过去的 GPT 错误更少*。
- **GPT-5.2 据称被削弱 (Nerfed)**: 一位成员询问其他人是否注意到官方网站上的 **GPT-5.2** 遭到了 **nerf**。
   - 他们表示 *该模型在一周前突然变得很笨*。
- **ChatGPT Plus：Cyber Security 学习伙伴？**: 一位成员正在考虑将 **ChatGPT Plus** 用于 **Cyber Security** 学习。
   - 他们想知道 *利用自己的复习文件来编写详细且具有特定考试风格的问题* 是否值得。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1464991260476444874)** (178 messages🔥🔥): 

> `重度否定，后果学习 (Consequence learning)，MCP 范式，Sora 提示词编写` 


- **否定词导致 AI 结果不够可靠**: 一位成员分享了一个关于 **negation**（否定词）及其在 AI 中导致不可靠结果的 [ChatGPT 链接](https://chatgpt.com/share/69763cc6-5360-8000-a850-85cbce128037)。
   - 他们指出，**LLMs** 和通用 **AI** 在处理否定词时表现挣扎，在大规模应用时可能导致错误。
- **后果学习 (Consequence Learning) 优于 Token 监管 (Token Policing)**: 一位成员辩护了他们使用 **consequence learning** 的方法，称其核心在于让 AI *体验并内化其行为的真实结果*，而不仅仅是避开负面指令。
   - 他们认为当前的“否定词问题”源于模型训练缺乏真实的后果反馈，这与 **token policing** 或指令微调 (instruction-tuning) 形成对比。
- **MCP 范式转型减少 Token 膨胀 (Token Bloat)**: 一位成员讨论了 Anthropic 提出的 **MCP 范式转型**，即 AI 现在通过编写代码与工具交互，通过将交互式闲谈和工具定义排除在上下文之外，从而减少 **token bloat**。
   - 他们强调，利用新的 **discoverability function**，必须让 **Agent** 意识到 **MCP** 发现过程本身，这是一个比 *不要幻觉出工具 (Do not hallucinate tools)* 更强的指令。
- **Sora 在结构化提示词方面表现欠佳**: 一位成员寻求通过结构化提示词改进 **Sora** 视频输出的建议，但另一位成员建议 **Sora** 无法有效处理此类格式化的提示词。
   - 建议尝试自然语言翻译，将提示词写成段落形式的生动视觉描述，以获得更好的效果。

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1464991260476444874)** (178 条消息🔥🔥): 

> `LLM 中的否定、后果学习、MCP 工具可发现性、Sora 提示词技巧、AI 安全与伦理` 


- ****否定理解之难，LLM 缺乏可解释性****：成员们讨论了 **LLMs** 在处理**否定句（negation）**时面临的挑战，指出 AI 系统在理解否定指令方面存在困难，可能导致结果不可靠，这是一个[已有记录的问题](https://chatgpt.com/share/69763cc6-5360-8000-a850-85cbce128037)。
   - 会议强调，**否定理解**是各种模型类型的普遍挑战，在 **prompt engineering** 中依赖否定指令时需要格外谨慎。
- ****后果难题：AI 的动作-结果一致性****：一位成员提出了**后果学习（consequence learning）**的概念，即 AI 通过体验其行为的真实结果来学习，这与仅基于 Token 监管或 **instruction-tuning** 的训练形成对比。
   - 随后引发了关于该方法与传统方法有效性的辩论，一方主张**现实世界反馈**的重要性，另一方则强调规模化实验和现有研究的意义。
- ****Sora 的叙事瓶颈：破解电影化创作****：一位成员寻求关于如何通过提示词引导 **Sora** 生成符合特定电影准则的视频的建议，特别是让角色在画面中自然出现，而非凭空冒出。
   - 建议将技术化的提示词格式转化为自然语言描述，使用简洁且语义丰富的段落以获得更好的效果。
- ****MCP 的蜕变：通过上下文协调掌握模型****：讨论重点介绍了 **Meta-Contextual-Prompting (MCP)**，其架构已更改为让 AI 编写代码而非直接与 MCP 工具交互，从而使 AI 能够意识到**工具发现（tool discovery）**。
   - 成员指出，**Anthropic** 开发了这一标准，并在 **agentic development** 领域得到了广泛采用。
- ****AI 的算法焦虑：通过 Alignment 避免无序****：一位成员对安全性表示担忧，特别是对于那些移除道德框架和 **guardrails** 的系统，认为这种缺乏约束的 **Agent** 是一种隐患。
   - 辩论认为，**Alignment**（对齐）不是“火车的铁轨”，而是系统的导航罗盘；在没有道德或伦理启发式的情况下内化结果是危险的，这会以牺牲客观真理或社会安全为代价来优化用户顺从度。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1464349302175436873)** (848 条消息🔥🔥🔥): 

> `Perplexity Pro 限制、Comet 浏览器担忧、图像生成问题、Gemini vs GPT-5.2、受限国家的 AI 替代方案` 


- **Perplexity Pro 用户忧虑查询限制**：Pro 用户报告称，尽管拥有“几乎无限”的套餐，但在**增强型查询和文件上传**方面仍达到了限制，这引发了关于未公开限制和潜在服务降级的猜测。
   - 许多用户感到沮丧，称该服务正变成一场**骗局**，并因限制以及**难以联系到客户服务**而考虑取消订阅。
- **Comet 浏览器引发恶意软件恐慌**：一些用户声称 Perplexity 安装的 **Comet 浏览器**包含恶意软件，建议其他人使用 VirusTotal 等工具分析该软件。
   - 然而，其他人对此不以为然，质疑标记为恶意软件的安装包来源，并称这种说法极其荒谬。
- **图像生成功能崩溃**：Pro 用户正面临**图像生成问题**，部分用户无法生成任何图像，并收到功能不可用的消息。
   - 还有报告称 Pro 用户的**视频生成限制**为每月 5 个，且某些提示词导致生成的是静态图像而非视频。
- **Gemini 3 在 GPT-5.2 面前赢得一席之地**：用户正在辩论 **Gemini 3** 与 **GPT-5.2** 的优劣，一些人声称 Gemini 在特定任务（如旅行调研）中表现更好，因为它集成了 Google Maps。
   - 其他人则表示 **GPT** 和 **Grok** 在更广泛的问题上可能表现更佳。
- **AI 访问：受制裁国家的博弈**：**俄罗斯**的用户正在讨论由于**制裁**导致访问 AI 服务的挑战，包括使用 VPN 和第三方服务来规避限制。
   - 文中提到了中国的 AI 替代方案，但一些用户由于数据使用方面的顾虑表示不愿使用，转而推荐 LMArena 等选项（尽管其访问也可能受限）。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1464349472543736094)** (829 条消息🔥🔥🔥): 

> `NB Pro vs NB 3 Pro 图像生成，LMArena 审查机制，Wan 2.6 图像，Gemini 和 GPT 质量，Grok` 


- **NB 3 Pro 在图像生成方面优于以往模型**：用户发现 **NB 3 Pro** 生成的图像质量比以前的模型更高，优于除 **NB Pro** 之外的所有其他模型，尤其是在生成**虚构武器**方面。
   - 虽然目前**没有任何 AI 模型能准确生成 AR 步枪**和**无托武器 (bullpup weapons)**。
- **LMArena 的审查机制存在问题**：LMArena 的审查机制备受质疑，例如允许生成“持枪女性”的 AI 图像，却屏蔽了“睡眠中的女性”。
   - 审核团队正在[收集误报案例](https://discord.com/channels/1340554757349179412/1447983134426660894)以改进审核机制。
- **Wan 2.6 在 T2I 方面遇到困难**：`wan2.6-image` **仅限图像编辑**，需要上传图像才能工作，而 `wan2.6-t2i` **目前无法上传图像**。
   - 团队已意识到此问题，正在努力为 `wan2.6-t2i` 启用图像上传功能。
- **GPT 5.2 High Search 表现糟糕**：**GPT 5.2 High Search 比其他模型更容易产生幻觉**，一位用户发现 **Gemini 的 Deep Research 相比之下也很差**，因为它只是略读而没有仔细阅读来源。
   - 一位用户表示“自从 4.5 发布以来，它真的改变了我的生活”，并称赞 Claude “心地善良，这种感觉很奇妙”。
- **Banana 2k 去哪儿了**：用户讨论了 **Banana 2k** 模型消失的原因，一些用户声称它已被移除，而另一些用户则认为它仍然可用，或者可能已整合到新的 **NB Pro** 中。
   - 后来工作人员宣布该模型已恢复，并开玩笑说它之前**去度假了**。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1464386642818105457)** (4 条消息): 

> `文生图模型，图像编辑模型，Code Arena 模型，Text Arena 模型，图像编辑排行榜` 


- **WAN-derful 新图像模型上线**：新的**文生图 (Text-to-Image)** 模型 `wan2.6-t2i` 和新的**图像编辑 (Image Edit)** 模型 `wan2.6-image` 现已在 [LM Arena](https://lmarena.ai/c/new?chat-modality=image) 上可用。
- **Devstral 在 Code Arena 大杀四方**：`devstral-2` 模型已添加到 [Code Arena](https://lmarena.ai/c/new?chat-modality=code&mode=direct-battle) 进行直接对战。
- **Qwen 满足文本需求**：`qwen3-max-thinking` 模型是 [Text Arena](https://lmarena.ai/?chat-modality=chat) 的新成员。
- **混元 (Hunyuan) 的色彩获得高排名**：`Hunyuan-Image-3.0-Instruct` 目前在 [图像编辑排行榜](https://lmarena.ai/leaderboard/image-edit)上排名第 7。
- **Molmo 模型家族扩大**：`molmo-2-8b` 模型已添加到 [Text Arena](https://lmarena.ai/?chat-modality=chat)。


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1464452624370896976)** (1 条消息): 

> `数据库故障，Generations API 受影响，活动页面问题` 


- **数据库故障影响 Generations API**：据报告，从 <t:1769221560:s> 开始，一次**数据库故障**影响了 **Generations API** 和**活动页面**。
   - 该故障随后在 <t:1769228340:s> 报告已解决。
- **Generations API 面临后续影响**：由于**数据库故障**，**Generations API** 出现了中断，影响了用户活动。
   - 工程师已努力恢复功能，故障在 <t:1769228340:s> 完全解决。


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1464888279148003460)** (6 条消息): 

> `Levante 集成，MCP 原生 AI 工作区，密码破解工具，非法用途讨论` 


- **Levante 作为 MCP 原生 AI 工作区集成**：一位用户分享了 **Levante** 的集成，这是一个开源的 **MCP 原生 AI 工作区**，旨在通过模块化界面与 **Ollama** 等本地模型进行交互，可在此处[下载](https://www.levanteapp.com)。
- **密码破解工具引发争议**：一位用户对一款标榜为“针对 PII”和密码猜解的工具表示担忧，认为它是**身份盗用**的潜在工具，而非用于**安全研究**。
- **讨论非法比特币钱包破解策略**：关于破解他人加密货币钱包的明确讨论引发了担忧，这可能导致计算机欺诈、加密货币窃取以及未经授权的系统访问。


  

---

### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1464350656482640097)** (754 messages🔥🔥🔥): 

> `OpenRouter Gacha, Competitive AI Usage Platform, OR Logs Destination, GPT-5.2, Waifu orbs` 


- **用户请求 OpenRouter Gacha**：用户开玩笑地请求开发 **OpenRouter Gacha**（抽卡）系统，有人建议加入保底机制，在抽取一定次数后可以获得 **GPT 5.2** 或 **Gemini 3 Pro**。
   - 一位用户开玩笑说将 **OR logs destination** 设置为 `waifu.orb.town/fun/bucket` 以获取超稀有抽取，随后澄清这只是个玩笑。
- **用户讨论竞争性平台**：一位用户分享了一个可以与其他开发者比较 **AI usage** 的平台，并在 [https://burntop.devkp.42](https://burntop.devkp.42) 寻求反馈。
   - 另一位成员建议向 **JAI userbase** 进行营销，并创建一个专门的 Gooning 排行榜来追踪用于 "gooning" 的 Tokens 使用情况。
- **讨论测试审核过滤器**：一些用户注意到一个带有**中文/日文昵称**的用户在发送并删除消息，推测是在测试审核过滤器（moderation filters）或服务器索引。
   - 成员们认为该用户正在测试响应姿态（testing posture）。
- **用户咨询免费和付费模型**：一位用户询问如果 Credits 降至 10 以下，是否仍能保持 **extended free model limit** 的访问权限。
   - 其他用户参与了讨论并提出了问题。
- **Gemini 产生幻觉**：一位用户报告 **Google Gemini 3 Pro** 产生幻觉，给出了一个未来的日期，并编造了时空穿越的故事，建议 OpenRouter 进行调查。
   - 该用户被引导至 Discord 支持频道，问题似乎与 System Prompt 有关。


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

jakobdylanc: https://openrouter.ai/minimax/minimax-m2-her
  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1464557110049112201)** (13 messages🔥): 

> `Kimi AI, Cerebras GLM performance, OpenRouter image support` 


- **Kimi AI 宣称 K2.5 称号**：OpenRouter Discord 成员提到 [Kimi.com](https://www.kimi.com/) 上出现了一个自称是 **Kimi K2.5** 的新聊天机器人。
   - 这与 LMArena 中的 *kiwi-do* 隐身模型（stealth model）相吻合。
- **OpenRouter 缺少图像工具支持**：一位成员在发现 OpenRouter 将 *image/png* 工具输出映射为字符串（string）而非图像后，花费了 **$5**。
   - 他们发布了一张 [图片](https://cdn.discordapp.com/attachments/1392278974222307469/1465410878382805082/image.png?ex=697901bb&is=6977b03b&hm=21677e978d8654f93d20edecf997bd4f49fb0dd08781cf93f15df8e2661ba1b5&) 表达了对缺乏图像支持的沮丧。
- **Cerebras GLM 达到 190 TPS**：**Cerebras** 在 **GLM 4.7** 上持续跑出约 **190 TPS**。
   - 成员指出 **Together AI** 仅达到 **100 TPS**，使得 Cerebras 快了近两倍。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1464350807045443747)** (662 条消息🔥🔥🔥): 

> `Terraform 基础设施蓝图, Cursor 使用限额, Gemini API Key 日志延迟, Cursor 客户端问题, Auto Mode 变更` 

- ****Terraform 蓝图**助力 AI 辅助项目启动**: 一位成员分享了一个[固化的 Terraform 基础设施蓝图仓库](https://github.com/berTrindade/terraform-infrastructure-blueprints)，旨在提供可直接复制粘贴且具备生产环境意识的模板，目的是提高 AI 工具在处理新项目时初始模式的一致性。
   - 目标是让 AI 能够根据项目需求推荐合适的蓝图，但成员们注意到[该链接最初是失效的](https://github.com/berTrindade/terraform-infrastructure-blueprints)。
- ****使用限额**引发 Cursor 客户焦虑**: 用户反映在 Pro 和 Pro+ 计划中，实际达到的使用限制与预期不符。一位成员指出，他们在 Pro 计划达到 **~$45**，在 Pro+ 计划达到 **$100**，这引发了关于性价比的质疑。
   - 一些人推测初始月份可能会提供更高的使用额度，而另一些人则分享了优化 Token 消耗的策略，例如[频繁开启新对话](https://cursor.com/docs/cli/reference/slash-commands)以及使用像 **GPT-5 Mini** 这样的小型模型。
- ****Gemini API** Key 日志延迟引发持续关注**: 成员们正在讨论 **Gemini API keys** 在记录使用量和成本方面存在的显著延迟，一位用户报告等待了 **20 小时** 仍未看到任何注册的使用记录。
   - 这种延迟引起了对准确追踪开支和有效管理使用量的担忧，促使人们询问潜在的变通方法或解决方案。
- ****客户端问题**困扰技术人员**: 几位成员遇到了 Cursor 客户端的问题，包括无法连接到过去的 Agent 对话以及一般的连接问题。
   - 建议的解决方案包括[查看 Cursor 论坛](https://forum.cursor.com/t/cursor-ai-is-no-longer-able-to-load-chats-locally/143599/13)、在设置中尝试不同的 HTTP 版本，或者在不恢复编辑器的情况下重新打开客户端。
- ****Auto Mode** 在算法调整后被削减**: 成员们注意到，让 Agent 完全自主运行的能力已被移除，同时 Auto Mode 中的**图像生成**功能也被取消。
   - 还有人建议 **Auto Mode** 会路由到 Composer 2，一位用户补充道：*“我 200% 确定它确实是这么做的，但还是觉得可惜。”*

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1464354761867526145)** (516 条消息🔥🔥🔥): 

> `中国模型, CPU 运行本地 LLM, MCP 工具, 4080 vs 3090 游戏表现, M5 Pro MacBook` 

- ****中国模型实现跨越式领先？****: 一些成员发现 **Deepseek** 和 **Qwen** 模型在推理能力上令人印象深刻，想知道为什么中国模型*似乎领先于美国模型*。
   - 一位成员认为美国模型优先考虑订阅服务而非开放获取，而另一位成员开玩笑说 Deepseek 和 Qwen 擅长*表现得擅长推理*，即便它们并没有完全搞定。
- ****CPU 能应对编码挑战吗？****: 一位成员报告说，只要模型不是太大，**在 CPU 上运行 LLM** 处理某些任务的表现还可以。
   - 另一位使用 Intel i3 的成员表示需要存钱买 **Nvidia** 显卡，而其他人则建议将 **AMD** 的选项（如 **MI50** 或 **7900 XTX**）作为文本生成的廉价替代方案。
- ****MCP 工具：如何充分利用？****: 成员们讨论了 **MCP 服务器**面临的挑战，指出它们并非为 LM Studio 设计，可能导致请求格式错误和较差的用户体验。
   - 对于文件处理，一位成员建议将文件路径提供给 **MCP 服务器**，由服务器自行处理；而另一位则建议构建自己的连贯技术栈，以便实际使用 Agent。
- ****游戏选 4080 还是 3090？****: 一位考虑 **4080** 的用户被建议购买二手的 **3090** 或 **7900 XTX**，但该用户玩游戏的需求多于使用 AI。
   - 讨论显示 **3090** 仅在 4K 分辨率下的游戏表现更好，而假设的 **5070 Ti** 会比这两者都快得多。
- ****M5 Pro MacBook Pro 即将发布？****: 成员们推测 **M5 Pro MacBook Pro** 的发布时间，传闻指向 28 号的一个活动。
   - 成员们对 **M4 Pro** 的内存带宽表示担忧，认为它可能不足以运行更大的模型，随后讨论转向了 **M1 Ultra** Mac Studio 的成本和性能。

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1464370714789413042)** (134 messages🔥🔥): 

> `AIO vs 风冷散热器, Ryzen 7950x 温度, 统一内存机器, 图像/视频生成硬件` 


- **炎热气候下 AIO 优于风冷散热器**：成员们发现 AIO 一体式水冷在炎热气候下表现远优于风冷散热器，指出 AIO 与 Noctua D-15 之间存在 **10C** 的温差，尤其是在持续的 CPU 占用下，因为风冷散热器通常在 **5 分钟**后就会达到瓶颈。
   - 有观点认为，除非害怕漏液，否则 *没有任何理由选择风冷而不是 AIO*，并补充说 *Arctic Freezer 360 甚至还便宜 10 欧元*。
- **Ryzen 7950x 发热严重，即使使用 D-15**：用户报告称，即使使用 Noctua D-15 风冷散热器，Ryzen 7950x 的温度也能达到 **95C**，建议切换到 AIO 以在 Boosting 期间将温度保持在 **80C** 左右。
   - 虽然有人建议将 CPU 限制在 **70C**，但也有人声称在 **95C** 时 *没有性能损失*，尽管这可能取决于具体的 CPU Binning（体质）和工作负载。
- **统一内存迷你 PC：是炒作还是灾难？**：一位用户购买了一台 AI Max + 395 迷你 PC，希望凭借其统一内存获得与 **7900 XTX** 相当的性能，但其他人警告称，虽然它可以运行更大的模型，但速度会比独立 GPU 慢。
   - 有建议指出，由于 **ROCm** 支持不佳，AI Max + 395 迷你 PC 的表现可能比规格相似的 **M4 Max** 低 **20%**。
- **GPU VRAM 对图像/视频生成至关重要**：用户讨论了使用 **WAN 2.2** 等模型进行视频生成的硬件要求，指出虽然 **16GB** 的 VRAM 足以运行该模型，但更多的 VRAM（如二手的 3090）更理想。
   - 虽然 z-image turbo 在 **4090** 上的表现尚可，但目前还没有图像生成的 “LM Studio” 对应软件，迫使用户使用 ComfyUI，而另一些人则认为它是 *图像生成领域发生的最好的事情之一*。
- **风扇卡让双 GPU 保持凉爽**：一位成员询问关于 2x GPU 配置中如何在显卡之间留出空气流动的建议，得到的方案是使用风扇卡（fan cards）在 GPU 之间强推空气。
   - 它们看起来像 GPU，并插入 PCI-e 插槽。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1464364447782011023)** (205 messages🔥🔥): 

> `Recursive AI 40 亿美元估值, 获得 AI 工作, 加州大学伯克利分校 Sky Lab 初创公司融资, OpenAI 的 PostgreSQL 扩展, Vibe Coding 在 iOS 的爆发` 


- **Recursive Intelligence 以 40 亿美元估值融资**：**Recursive Intelligence** 正以 **40 亿美元估值**进行融资，重点是利用 AI 加速芯片设计，在硬件和人工智能之间创建自改进的反馈循环（[彭博社文章](https://www.bloomberg.com/news/articles/2026-01-23/ai-startup-recursive-in-funding-talks-at-4-billion-valuation)）。
- **在没有 AI 经验的情况下获得 AI 工作**：**Noam Brown** 概述了如何通过独立项目建立公开记录并参加受关注的比赛来在顶尖 AI 实验室获得职位（[链接](https://xcancel.com/polynoamial/status/2014084431062114744)）。
   - 他强调了改进现有同行评审研究以及参加像 **NanoGPT** 竞速赛这样高关注度比赛的重要性，以展示卓越的技术能力，并以 [Keller Jordan](https://github.com/KellerJordan/modded-nanogpt) 为例。
- **加州大学伯克利分校 Sky Lab 初创公司估值飙升**：**Alex Dimakis** 强调了 2026 年 1 月伯克利 Sky Lab 初创公司的重大融资里程碑，包括估值 **4 亿**的 **SGLang**，**8 亿**的 **vLLM** 以及 **17 亿**的 **LMArena**（[链接](https://xcancel.com/alexgdimakis/status/2014508959621959724?s=46)）。
- **由 AI Coding Agents 开发的 FastRender 浏览器**：**Simon Willison** 讨论了与 Wilson Lin 关于 **FastRender** 的对话，这是一款使用超过 **2,000 个 AI Coding Agents** 开发的新型浏览器渲染引擎（[链接](https://simonwillison.net/2026/Jan/23/fastrender/)）。
- **微软 Maia 200 AI 加速器上线 Azure**：**Satya Nadella** 宣布 **Maia 200 AI 加速器** 现已在 **Azure** 投入运行（[链接](https://xcancel.com/satyanadella/status/2015817413200408959)）。
   - 该定制芯片专为高性能推理设计，性价比提升了 **30%**，并优化了规格，包括 **216GB HBM3e** 和 **7TB/s 显存带宽**，以支持大规模 AI 工作负载。


  

---

### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1464696410887946261)** (10 条消息🔥): 

> `Remotion Launchpad, Motion Canvas, Tencent HunyuanImage 3.0-Instruct` 


- ****Launchpad** 作为 **Remotion** 的衍生版本发布**：Francesco 开源了 [Launchpad](https://xcancel.com/francedot/status/2014897878347743732)，这是一个基于 **Remotion** 的产品发布视频配置方案。
   - 它包含视频模板、共享动画组件，并集成了 **Claude Code** 以实现快速视频创作。
- ****Motion Canvas** 受到电影制作者的启发**：**Remotion** 构建于 [motion canvas](https://github.com/motion-canvas/motion-canvas) 之上，后者最初由一位游戏设计师/YouTuber 设计。
   - 这位设计师的 [YouTube 频道](https://www.youtube.com/@aarthificial) 内容非常有趣。
- ****HunyuanImage 3.0** 专注于指令性**：腾讯发布了 [HunyuanImage 3.0-Instruct](https://xcancel.com/TencentHunyuan/status/2015635861833167074)，这是一个原生的多模态 **80B MoE 模型**，专注于精确的图像编辑和多图融合。
   - 它采用了原生思维链 (**CoT**) 推理模式和 **MixGRPO** 算法，以提高意图对齐和合成质量，提供与领先的专有模型相媲美的 State-of-the-Art 性能。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1464456699971637524)** (141 条消息🔥🔥): 

> `Spaces Docker Build Pauses and 503 Error, Reinforcement Learning Channels, Windows 11 Hugging Face Models App, Lighton OCR, HeartMula, LTX-2 and the Qwen-3 TTS in ComfyUI` 


- ****HuggingFace Spaces 出现故障****：用户在 **Spaces Docker 构建** 期间遇到**暂停 (pauses)**，并在重启时收到 **503 错误** ([discuss.huggingface.co](https://discuss.huggingface.co/t/spaces-docker-build-pauses-and-503-error-on-restart/171149/2))。
   - 似乎是底层基础设施问题导致 Spaces 变得无响应，需要手动干预才能解决，许多人遇到了 `Something went wrong when restarting this Space` 的错误。
- ****RL 频道整合****：课程相关的 **Reinforcement Learning** 频道已合并为一个新的 [统一频道](https://discord.com/channels/879548962464493619/1329142738440028273)，以便更好地进行组织。
   - **Deep Reinforcement Learning** 课程中的旧说明已过时，因此成员现在应参考整合后的频道进行相关讨论。
- ****VoltageGPU 算力提升****：[VoltageGPU.com](https://voltagegpu.com) 正在为开源 AI 模型提供廉价的 GPU，**NVIDIA GeForce RTX 5090 pod** 的价格为 **$0.53/小时**。
   - 他们强调了其先进的 **32GB GDDR7** 的优势，该配置针对 **HF 托管模型（如 Qwen3-32B）** 的推理进行了优化，并提供免费额度供用户试用其服务。
- ****Latitude 上的大语言模型延迟表现****：拥有 **1,000 多个 GPU** 的裸金属云供应商 **Latitude.sh** 已提交 PR，申请成为 **HuggingFace 推理供应商** ([JS Client](https://github.com/huggingface/huggingface.js/pull/1927), [Python Client](https://github.com/huggingface/huggingface_hub/pull/3715), [Docs](https://github.com/huggingface/hub-docs/pull/2180))。
   - 他们部署了 **Llama 3.1, Qwen 2.5/3, DeepSeek R1 和 Gemma 2** 等模型，并提供 **OpenAI 兼容 API**，目前正在寻求对其 PR 的反馈。
- ****OpenCV 派上用场****：对于 Agent 式文档处理，一位成员发现 **OpenCV** 在检测和提取应用 ML 论文中的文本、图像和 LaTeX 数学公式方面表现良好，优于 Florence 等通用模型。
   - 他们正在寻找一个性能更强、适用于 Captioning 的小型模型。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1464426521027416195)** (38 messages🔥): 

> `Layer-Native Safety Clamping, GPU-64 Architecture for LLM Inference, webXOS RLHF Gaming Initiative, KV Cache in LLM Inference, ML deployment and inference platform` 


- **安全钳位防止越狱 (Safety Clamping Prevents Jailbreaks)**：一篇新论文介绍了 **Layer-Native Safety Clamping**，这是一种在模型内部钳位激活值以防止越狱的方法，团队发布了一个包含 **10K 对数据**的[数据集](https://huggingface.co/datasets/Pacific-Prime/safety_dataset)。
   - 该方法学习激活空间中的“伤害方向 (harm directions)”，并钳位任何投影过强的激活，因此无法通过提示词操纵来绕过；论文可以在 [Zenodo](https://zenodo.org/records/18359832) 上找到。
- **GPU-64 助力 LLM 推理**：一种专门为推理设计的新型 **GPU 架构** **GPU-64** 正式发布，其创新之处在于使用了片上 **CAM**（内容寻址存储器）的硬件 **KV-Cache**。
   - 结果显示，在 **75W** 功耗下推理速度提升了 **4 倍** (O(N) → O(1))，论文已发表在 [Zenodo](https://zenodo.org/records/18364282)，同时 [RTL + Emulator](https://github.com/Complexity-ML/gpu64-inference) 已托管至 GitHub。
- **webXOS 游戏计划**：一篇论文介绍了 **webXOS RLHF Gaming Initiative**，这是一个通过基于浏览器的交互式游戏体验生成高质量多模态数据集的框架，详见此 [Claude artifact](https://claude.ai/public/artifacts/358eea9a-4eec-4b92-be36-43797d8a76e4)。
   - 该计划利用现代 Web 技术消除硬件壁垒，同时保持机器人、计算机视觉和自主系统等高级 **RL 应用**所需的精度。
- **KV Cache 疑难解答**：一名成员分享了一篇 [Medium 文章](https://medium.com/@nainia_ayoub/kv-cache-in-llm-inference-7b904a2a6982)，详细解析了 LLM 推理中的 **KV Cache**，这为他们在调试 **CUDA OOM**（显存溢出）错误时节省了大量时间。
   - 其他成员也加入讨论，分享到：“说实话 KV Cache 也很折磨人，因为大多数人都忘了它的存在”。
- **一行代码完成 ML 部署**：一名成员为本周末的黑客松发布了一个 ML 部署和推理平台，可通过一行 Python SDK 访问，并将模型封装在 **Docker container** 中。
   - 模型产物被发送到 **Go backend**，后者将其容器化并通过反向代理公开，同时提供一个 UI 用于运行推理并提供实时的 API 端点；欢迎给这条 [X 帖子](https://x.com/deepto98/status/2015153491052990841)点赞。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1464407627395108875)** (30 messages🔥): 

> `GAIA Agent Course completion, LLM from scratch, Llama 3.2 vision agent, Summarization Pipeline, LMStudio and Deployment` 


- **GAIA Agent 证书获取**：一名成员报告以 **30%** 的成绩通过了 **GAIA Agent Course Unit 4 Project**，并询问如何获得证书。
   - 另一名成员建议访问 [robot-learning-tutorial](https://huggingface.co/spaces/lerobot/robot-learning-tutorial)。
- **Llama 3.2 Vision Agent 的盲点**：一名成员尝试使用 **Llama 3.2 vision** 构建一个 Agent 来为图片列表生成描述，但模型似乎并没有接收到传入的图像。
   - 该成员分享了一段初步的代码片段。
- **LLM 测试与部署**：一位成员推荐使用 **LMStudio** 测试模型，因为它具有用户友好的 GUI 以及针对 HF 和 GH 模型的搜索过滤器；对于单用户部署，推荐使用 **llama.cpp**。
   - 他们建议不要将 LMStudio 用于后端部署，而是建议在 Docker 容器中使用 **llama.cpp 的 llama-server** 或使用 **vLLM server** 以获得更好的可扩展性。
- **扩展 LLM 知识的资源**：一位成员解释说，**RAG (Retrieval Augmented Graphing)** 用于在不进行训练的情况下扩展 LLM 的知识，方法是将单词/句子的“含义”作为 Embedding 存储在向量存储中。
   - 他们澄清说，**Embedding 模型**也是模型，它们经过训练可以在向量中搜索与 Prompt 相似的哈希值，然后将其包含在 Prompt 中。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1464428370161832007)** (36 条消息🔥): 

> `MXFP8 quantization, MLSys 2026 FlashInfer-Bench Competition, Nvidia Triton Inference Server, GPU Mode GTC meetups, Madlab Liger-Kernel integration` 


- **MLSys 2026 FlashInfer-Bench 竞赛**：MLSys 2026 FlashInfer-Bench 竞赛要求参赛者设计 AI Agent，在最新的 **NVIDIA Blackwell GPU** 上编写最先进的 **LLM 推理内核（inference kernels）**，并与专家编写的 **FlashInfer kernels** 进行竞争，详情见 [mlsys26.flashinfer.ai](https://mlsys26.flashinfer.ai/)。
- **Triton Inference Server 讨论**：一名成员询问了关于 **Nvidia Triton 推理服务器** 以及在 Triton 上通过 **BLS 脚本调用 vLLM 后端模型** 的讨论。
   - 另一名成员建议在 general 频道讨论，并指出这是第一次有人问及此问题，目前有人正尝试研究如何将 thinking budget 参数传递给 vLLM。
- **GTC 期间的 GPU Mode 聚会**：GPU Mode 计划在 **GTC（3 月 16-19 日）** 期间宣布 **nvfp4 竞赛** 的获胜者并举办 **社交活动**。
   - 活动可能包含社交聚会，往届活动包括 Beer Garden 和 Outside market。
- **用于多模态模型的 Cornserve**：一名成员分享了他们在 **Cornserve** 上的工作，这是一个针对 Any-to-Any 多模态模型的高效在线服务系统，详见论文 [Cornserve](https://arxiv.org/abs/2512.14098)。
   - Cornserve 优化了包含 **多模态编码器**、**LLM** 和 **Diffusion Transformers (DiTs)** 等异构组件模型的部署方案，提高了吞吐量并降低了尾部延迟。
- **寻求 GPU Mode 新闻的 RSS Feed**：一位用户请求为 GPU Mode 新闻页面（[https://www.gpumode.com/v2/news](https://www.gpumode.com/v2/news)）提供 **RSS feed**，并表示愿意贡献代码。
   - 一名成员回复提供了该网站的 **GitHub repository** ([https://github.com/gpu-mode/kernelboard](https://github.com/gpu-mode/kernelboard)) 以供贡献，并开玩笑地建议测试一下 Claude 是否能实现该功能。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1464609026993164444)** (7 条消息): 

> `7 point 3D stencil computation, CUDA sample with a 25 pt stencil, Time zone bug, cutile fused moe kernel in the gym repo` 


- **CUDA 示例成为 Stencil 讨论焦点**：一名成员正在寻找优化 **7 点 3D 模板计算（stencil computation）** 的技巧，另一名成员推荐了一个带有 **25 点模板** 的 [CUDA 示例](https://github.com/NVIDIA/cuda-samples/blob/4f735616ba599fe93cc2c6c85dcb4369260f9643/Samples/5_Domain_Specific/FDTD3d/inc/FDTD3dGPUKernel.cuh)，该示例可以进行修改。
- **时区 Bug 讨论**：成员们正在调试一个 [时区 Bug](https://x.com/theemozilla/status/2015251642585682405?s=20)。
   - 一名成员问道：*是什么让你想到时区问题，而不是像 Y2K 或 dtype 溢出之类的问题？*
- **寻找 Cutile Fused MoE 内核**：一名成员正在寻求 *易于集成的 Blackwell 优化内核*，并询问是否有人尝试过 gym 仓库中的 **cutile fused moe kernel**。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1465051744789729312)** (3 条消息): 

> `BF16 Autocast, Dynamic Shapes, cu128 vs cu126, A100 issues` 


- **BF16 Autocast 在动态形状下报错**：一名成员反馈，在 **torch 2.10** 和 **cu128** 环境下，**A100** 使用 **bf16 autocast** 配合 **动态形状（dynamic shapes）** 会报错，且系统环境为 **cuda 13**。
   - 该用户指出，所有功能在 **cu126 wheel** 上运行正常，但在 **cu128 wheel** 上会崩溃。
- **请求详细问题说明**：一名成员询问了该问题的更多细节，特别是请求提供错误信息。
   - 同一成员还请求澄清任何可用于协助故障排除的额外细节。


  

---

### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1464685212125368320)** (2 messages): 

> `CornServe, 2025 GPU MODE Recap, 2026 GPU MODE Plans, Kernel LLM Training, Hardware Programming Complexity` 


- ****Cornserve** 热辣登场**: GPU MODE 与一名成员共同讨论了 **Cornserve: 简单、快速且可扩展的多模态 AI** ([YouTube 链接](https://www.youtube.com/watch?v=VhjUM_M71Wo))。
- ****GPU MODE** 在 2025 年取得了巨大成功**: **2025** 年对于 **GPU MODE** 是不可思议的一年：拥有 **26K** YouTube 订阅、举办了 **92** 场讲座、吸引了 **24K** Discord 成员、举办了 **3** 场奖金超过 **$100K** 的 Kernel 竞赛、收到了 **400K** 次 KernelBot 提交、举办了 **3** 场活动，并拥有 **10** 个活跃的工作组！
   - 该社区因 [project popcorn](https://gpu-mode.github.io/popcorn/) 项目获得了 Soumith Chintala, Ian Buck, Tianqi Chen, Shotlo Douglas, Tri Dao 和 Lisa Su 等行业榜样的赞誉。
- ****GPU MODE** 揭晓其 2026 年计划**: 在 **2026** 年，GPU MODE 将进一步推动 **Kernel LLM** 的训练，并利用它在 **PyTorch** 和 **VLLM** 等重要仓库中交付 Kernel ([gpumode.com/v2/news/gpumode-2026](https://www.gpumode.com/v2/news/gpumode-2026))。
   - 社区正与 **Prime Intellect**、**Modal** 和 **Lambda** 合作，重点关注优化 LLM 生成的 Kernel（de-slopifying）、后训练 Kernel LLM 模型、端到端竞赛以及从零开始的仓库建设。
- ****复杂硬件**正变得更加复杂**: 硬件编程变得越来越复杂 ([X 链接](https://x.com/bernhardsson/status/2014855658223395085?s=20))，社区有责任使其变得更简单！


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1464428062254039124)** (2 messages): 

> `LeCun Startup, Event Based Model` 


- **LeCun 成立新初创公司：Logical Intelligence**: Yann LeCun 成立了一家名为 [Logical Intelligence](https://logicalintelligence.com/) 的新公司，专注于 **基于事件的模型 (Event Based Model, EBM)**。
   - 遗憾的是，目前没有提供技术细节，仅提供了一个指向 [MLSys Conference](https://mlsys26.flashinfer.ai/) 的链接。
- **Event Based Model 笼罩在神秘之中**: 新初创公司 [Logical Intelligence](https://logicalintelligence.com/) 虽然专注于 **Event Based Models**，但未提供任何技术细节。
   - 网站目前仅包含营销材料、职位空缺以及指向 [MLSys Conference](https://mlsys26.flashinfer.ai/) 的链接。


  

---


### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1464379534966128843)** (2 messages): 

> `CUDA Kernel Optimization, Mindbeam AI Hiring` 


- **Parsewave 寻求 CUDA Kernel 优化工程师**: [Parsewave](http://parsewave.ai/) 正与前沿 AI 实验室和 AI 基础设施提供商合作，寻求 **CUDA C/C++ Kernel 优化工程师**来对内部模型进行基准测试，要求具备 **Nsight Systems / Nsight Compute** 和 CUDA **intrinsics** 经验（熟悉 Blackwell 架构理想，Hopper 架构亦佳）。
   - 候选人应能够解释优化收益，并提出展示**原始版本到优化版本增量（naive → optimized deltas）**的基准测试；感兴趣的申请人可以在[此处](https://tally.so/r/pbDDvZ)申请。
- **Mindbeam AI 招聘 Post Training 和 GPU Kernel MLE**: Mindbeam AI 是一个专注于加速基础模型训练的小型团队，正在招聘 `post training MLE` 和 `GPU Kernel MLE`。
   - 公司实行全远程办公并提供极具竞争力的薪资，鼓励感兴趣的候选人私信（DM）获取内推；[职位空缺列表请点击此处](https://jobs.ashbyhq.com/mindbeam)。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1464478780151173121)** (1 messages): 

> `Roofline Models, Kernel Optimization, Performance Analysis` 


- **轻松读懂 Roofline 模型**: 一名成员分享了一张图表，旨在帮助理解用于 **Kernel 优化**的 **Roofline 模型**，该图表适合在 LinkedIn 上分享，对初学者非常有帮助。
   - 该图表直观地解释了如何解读**性能瓶颈**，并根据硬件限制优化 Kernel。
- **Kernel 知识精选**: 分享的图表强调了**计算强度（computational intensity）**与**内存带宽（memory bandwidth）**在实现最优性能之间的关系。
   - 它强调，理解这些限制对于编写高效的 **GPU Kernel** 和最大化硬件利用率至关重要。


  

---

### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1464762075988361440)** (2 messages): 

> `Thread Coarsening 澄清，导师机会` 


- **Thread Coarsening 困惑已解决**：一位成员最初对 Thread Coarsening 中 `colStart` 的公式提出疑问，特别是是否应该包含额外的 `TILE_WIDTH` 因子，并引用了 [第 139 页，第 6.3 章](https://link.to.chapter)。
   - 在意识到文本指的是线程块负责的**列数**（columns），而非**元素**总数后，困惑得以解决。
- **寻求动手实践项目的导师**：一位完成了书中初始章节的成员正在寻找导师一起开展实际项目，并提供了[个人网站](https://vanshnawander.github.io/vansh/)作为背景信息。
   - 他们希望通过导师指导，将理论知识与动手实践经验相结合。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1464686054715031736)** (2 messages): 

> `MLSys 会议, Treehacks` 


- **参会者询问 MLSys 会议体验**：一位成员询问了 **MLSys 会议**的体验，并提到 **2026 年会议**将在**华盛顿州 Bellevue** 举行，由于他们在 **Bellevue College** 就读，计划届时担任志愿者。
- **成员寻找 Treehacks 伙伴**：一位成员询问是否有计划参加 **Treehacks** 的人可以私信联系。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/)** (1 messages): 

kashimoo: 数据中心 GPU 才是 AI 领域的重点，而不是消费级 GPU。
  

---


### **GPU MODE ▷ #[popcorn](https://discord.com/channels/1189498204333543425/1298372518293274644/1464459210942185494)** (13 messages🔥): 

> `MLSys 2026 FlashInfer-Bench 竞赛, 每周会议链接, 是否对新贡献者开放？, 2 月 3 日首次公开会议` 


- ****FlashInfer-Bench** 竞赛在 MLSys 2026 宣布！**：**MLSys 2026 FlashInfer-Bench 竞赛**已宣布，鼓励对 **AI kernel 生成**感兴趣的人员参加。
   - 提到更多细节可以在*新的 2026 帖子*中找到。
- **索取每周会议链接**：一位成员询问了每周会议的链接，另一位成员提供了[此链接](https://calendar.google.com/calendar/event?action=TEMPLATE&tmeid=MXFkbTJrZWlhcXQwZDluc3Q1cDBu3FidDV_MjAyNjAyMDNUMTgwMDAwWiBjMzYyMDQwNWUwYzBiNDI5YjMwNGE0YjU5ZTdiZTFjYWQzNTc0OTdlZmMxNDc1NzVmNDlhZjZlMjM0ZTA2NzdkQGc&tmsrc=c3620405e0c0b429b304a4b59e7be1cad357497efc147575f49af6e234e0677d%40group.calendar.google.com&scp=ALL)。
- **询问新贡献者加入事宜**：一位成员通过工作组页面的[此链接](https://gpu-mode.github.io/popcorn/)发现该频道，并询问是否对新贡献者开放。
   - 该成员受到了欢迎，并被引导至*新的 2026 帖子*以获取需要帮助的信息。
- **首次公开会议已排期！**：首次团队会议定于 **2 月 3 日**，组织者正在检查会议链接的状态。
   - 届时在 general 频道可能会有语音频道。


  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1464779101406494793)** (1 messages): 

> `Jetson, Torch, ONNX, TensorRT, trtexec 分析` 


- **ONNX 和 TensorRT 性能**：一位成员研究了 **ONNX** 到 **TensorRT** 的性能，并建议使用 *trtexec* 来分析 engine 层。
   - 他们表示可以从 engine 元数据中将 **TensorRT** 层映射到 **ONNX** 算子，但对于如何从 **ONNX** 算子映射回 **Torch** 毫无头绪。
- **Torch 工作流问题**：讨论涉及有人寻求建议，他们还提到了 **Jetson/Torch**、**ONNX** 和 **TensorRT**。
   - 他们研究了 **ONNX** 到 **TensorRT** 的性能，并建议使用 trtexec 来分析 engine 层。


  

---

### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1465181818662686761)** (14 messages🔥): 

> `RTX 3060 12GB 用于 ML/DL，Blackwell 上的 FP4 加速，消费级 vs 数据中心 GPU，DLSS 论文，5070ti 或 4070 ti Super` 


- **RTX 3060 仍然适用于 GPU 学习吗？**: 成员们讨论了鉴于其相对低廉的价格，**RTX 3060 12GB** 是否仍是学习 **GPUs** 以及进行 **ML/DL** 工作的良好选择。
   - 共识是它适合本地学习环境，尤其是以优惠价格购入时；但训练速度会较慢，且不支持 **Blackwell** 架构上的 **FP4** 等特性；参见 [Nvidia 的 Mistral 集成示例](https://huggingface.co/nvidia/Mistral-7B-Instruct-v0.3-ONNX-INT4#software-integration)。
- **消费级 Blackwell 与数据中心级 Blackwell 的区别**: 有成员辩论了考虑到 **消费级 Blackwell (SM_120)** 与 **数据中心级 Blackwell (SM_100)** 之间的差异，为了编写 Kernel 而购买昂贵的消费级 GPU 是否值得。
   - 尽管核心 Kernel 技能是可以迁移的，但保持对特定架构优化的关注对于职业竞争力至关重要。
- **GPU 基础可以在旧款 GPU 上学习**: 有人建议虽然新架构很重要，但通用的 **GPU Fundamentals** 可以在旧款 GPU 上学习以实现快速迭代。
   - 推荐的进阶路径是在 **Ampere** 上构建项目，然后针对 **Blackwell** 进行调优，并持续适配新架构。
- **5070ti 或 4070ti Super 优于 2x3060**: 一位拥有两块 **RTX 3060** 显卡的成员表示 **12GB VRAM** 是个限制，主张改为单块 **5070ti** 或 **4070 ti Super**。
   - 他们询问了有哪些解释 **DLSS** 的论文。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1464661449061175339)** (1 messages): 

> `Factorio 蓝图生成` 


- **AI 工程师发现 Factorio 蓝图生成项目**: 一位正在研究如何根据指令生成 **Factorio blueprint JSON code** 的 AI 工程师发现该项目非常出色，并在研究过程中偶遇了它。
- **AI 在自动化 Factorio 蓝图创建方面的潜力**: 讨论强调了使用 AI 模型自动生成 **Factorio blueprints** 的潜力，特别是专注于从用户指令创建 JSON 代码。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1464375553678119036)** (1 messages): 

> `Graphical Layout Calculus, Tuple Morphisms, Mutual Refinement, Layout Composition` 


- **以图形化方式进行 Layout Compositions 布局**: 成员分享了一个使用 **graphical layout calculus** 手动计算两个布局组合的实例。
   - 步骤包括将布局转换为 **tuple morphisms**，寻找 **mutual refinements**，进行 **pulling back** 和 **pushing forward**，执行组合，并将结果写为使用 **prefix products** 的布局。
- **映射到 Morphisms 以掌握布局**: 初始步骤涉及将可处理的布局转换为 **tuple morphisms** `m_A` 和 `m_B`。
   - 这种转换允许对布局进行代数操作，并使用 morphism 操作进行组合。
- **精化布局间的关系**: 该实例强调了寻找两个 **tuple morphisms** 的 **mutual refinement** 的重要性。
   - 这一步确保了组合布局时的兼容性和一致性，类似于在两个不同结构之间寻找共同点。
- **为了精确布局而进行 Pulling Back**: 过程包括沿 `m_A` **pulling back** **mutual refinement** 以获得 `\hat{m}_A`。
   - 此操作调整了精化过程以兼容布局 A 的结构，从而在组合期间实现无缝集成。
- **为了完美放置而进行 Pushing Forward**: 示例还涉及沿 `m_B` **pushing forward** **mutual refinement** 以获得 `\hat{m}_B`。
   - 此操作确保精化过程与布局 B 的结构保持一致，进一步促进平滑组合和一致的布局行为。


  

---


### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1465299821467406430)** (1 messages): 

> `M3 上的 Rust，CPU 和 GPU Kernel` 


- **M3 上的 Rust 基准测试**: 在 **M3** 上进行的初步 **Rust** 基准测试显示性能约为 **5% peak**，**rustc** 的循环重排（loop reordering）被认为是影响因素之一。
- **专注于 CPU 和 GPU Kernel**: 接下来的步骤涉及开发针对 **CPU** 和 **GPU** 的 Kernel，重点在于性能改进。


  

---

### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1464453050436550710)** (51 messages🔥): 

> `group_gemm issues, benchmark leaderboard gap, B200 Physical Resonance, Stream Error During Submission, MLSys Contest Tracks` 


- **Group GEMM 初始化超出 FP16 范围**：`group_gemm` 问题的旧张量初始化逻辑超出了 **FP16 范围**，导致出现 **INF** 值；提出了一个类似于 `dual_gemm` 初始化的修复方案，参考 [此 PR](https://github.com/gpu-mode/reference-kernels/pull/89/files)。
   - 虽然一些 **INF** 值是可以接受的，但团队对更改持开放态度；已开启一个 PR 来解决此问题 ([PR 96](https://github.com/gpu-mode/reference-kernels/pull/96))。
- **Benchmark 与排行榜的差距**：观察到 Benchmark 结果与排行榜得分之间存在显著差异，引发了对一致性的担忧。
   - 描述中提到 `K` 可以被 **256** 整除，但在测试用例中存在 `K=128` 和 `K=384`，建议删除或修改这些用例。
- **Veitner 关于 B200 GPU 的 Grouped Blockscaled GEMM 博客**：Simon Veitner 发表了一篇博文，自顶向下地解释了 **B200 GPU** 的 Grouped Blockscaled GEMM，以及 **MMA** 和 **TMA**、Tile 调度器（tile scheduler）等部分的设置，详见 [bearblog.dev](https://veitner.bearblog.dev/grouped-blockscaled-gemm-host-code/) 和 [LinkedIn](https://www.linkedin.com/posts/simon-veitner-174a681b6_grouped-blockscaled-gemm-host-code-activity-7420898572637962240-5kUN)。
   - 该博文旨在解释与 **B200** 上通常的 Persistent Blockscaled Dense GEMM 方法不同的部分。
- **Stream 错误困扰提交**：用户在提交过程中遇到 "stream" 错误，经追溯是因为代码中（甚至是注释中）出现了 "stream" 一词。
   - 从代码（包括注释）中删除 "stream" 一词后，解决了提交问题。
- **任务配置差异检查**：注意到 `task.yml` 中测试配置与 Benchmark 配置之间的差异，Benchmark 配置中同一组内的所有 **A** 和 **B** 具有相同的 **N** 和 **K**，而测试配置则不同。
   - 团队澄清说，测试是为了功能验证，不同组中的 **m/n/k** 可以不同；而性能测试来自实际用例，其中 **M (experts)** 不同，但 **N/K** 相同。


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1464660580232200447)** (8 messages🔥): 

> `Learning GPUs, TinyML, Embedded ML, Physical AIs` 


- **新人寻求 GPU 学习指导**：一位软件工程师正在寻求书籍建议，以理解 **GPU**、**优化** 和 **性能调优**，旨在转型至 **TinyML/Embedded ML** 或 **Physical AI** 领域。
   - 他们更倾向于通过书籍学习，因为长视频让他们感到吃力；他们对 **ML** 有基本了解，但缺乏硬件知识。
- **背景不重要，兴趣才重要**：一位成员分享道，他们主要拥有 **数学物理** 和 **形式化方法（formal methods）** 背景，但也获得了 **GPU 性能相关职位**。
   - 他们表示，美国公司目前在 AI 领域投入了大量资本。
- **在线搜索规格**：有人建议，面试官通常允许候选人在面试期间在线搜索特定的规格参数，因为具体的细节很容易查到。
   - 一位成员提到，问题是否可以“谷歌搜索”高度取决于公司和面试主题。


  

---

### **GPU MODE ▷ #[flashinfer](https://discord.com/channels/1189498204333543425/1464407141128339571/1464407381680062630)** (53 条消息🔥): 

> `组队合并、多赛道参与、注册确认、团队组建、Kernel 类型（单节点/多节点）` 


- ****组队合并热潮****：参赛者在报名截止前咨询了[团队合并](https://discord.com/channels/1197226372415488060/1464407141128339571)的相关事宜，组织者回复称这是允许的，但要求在变动时进行告知。
   - 组织者还根据需求设置了自动化的[注册确认邮件](https://discord.com/channels/1197226372415488060/1464407141128339571)。
- ****赛道切换策略****：参赛者询问是否可以[参加多个赛道](https://discord.com/channels/1197226372415488060/1464407141128339571)，组织者确认了可行性，但指出即使团队在多个赛道排名靠前，也只能获得一块 GPU 奖励。
   - 讨论中明确了团队稍后可以[切换赛道](https://discord.com/channels/1197226372415488060/1464407141128339571)，以便专注于最有希望的赛道。
- ****Kernel 保密难题****：参赛者就赛后[提交的内容是公开还是私密](https://discord.com/channels/1197226372415488060/1464407141128339571)提出了疑问。
   - 组织者澄清说，为了评奖，最终代码需要公开，但[开发过程可以保持私密](https://discord.com/channels/1197226372415488060/1464407141128339571)。
- ****新手指南****：初学者询问了 NVIDIA MLSys 竞赛中适合[新手的赛道选择](https://discord.com/channels/1197226372415488060/1464407141128339571)。
   - 建议是[仅使用 FlashInfer API 部署尽可能小的模型](https://discord.com/channels/1197226372415488060/1464407141128339571)，以熟悉代码库，同时避免使用像 B200 这样不稳定的平台。
- ****Agent 的秘密（基本）安全****：针对 FlashInfer AI kernel 竞赛中[Agent 方案开源](https://discord.com/channels/1197226372415488060/1464407141128339571)的要求寻求了澄清。
   - 组织者确认，虽然会审查 Agent 代码和技术报告，但只有[最终代码需要开源](https://discord.com/channels/1197226372415488060/1464407141128339571)，以确保其不是手动编写的 kernel。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1464355109260754964)** (57 条消息🔥🔥): 

> `ML 的 ROCm 性能、图像分类服务、DistinctionBench 与语言模型、带有 LLMs 的人机回环 (Human-in-the-loop) 工作流、语境学习 (In-context learning) 与权重更新` 


- **ROCm 迈向 ML 复兴的坎坷之路**：用户讨论了 **ROCm** 在加速 ML 方面的性能，指出虽然取得了长足进步，但由于主要支持仍然倾向于 **Nvidia**，使用起来仍具挑战性。
   - 由于潜在的驱动问题和较长的交付周期，这种体验被描述为“不含电池（batteries not included，意指非开箱即用）”。
- **DistinctionBench：训练目标还是防污染？**：关于论文 **Between Circuits and Chomsky: Pre-pretraining on Formal Languages Imparts Linguistic Biases** 的讨论认为 **DistinctionBench** 为语言模型提供了有趣的迁移。
   - 一位成员开玩笑说：*“所有好的评估标准最终都会变成训练目标 ;)”*，但指出 **DistinctionBench** 由于拥有无穷无尽的表现形式变体，具有“极强的防污染性”。
- **ICL 信号：权重在更新吗？**：一位成员询问是否有关于“利用语境学习 (In-context learning) 的信号来更新权重，作为一种持续学习形式”的论文，并分享了两篇相关论文：[chen24r.html](https://proceedings.mlr.press/v235/chen24r.html) 和 [arxiv.org](https://arxiv.org/abs/2507.16003)。
   - 对话还指出通过将内容推入参数化知识来节省推理成本，这让人联想起今年夏天的“线性注意力变体中的状态调优 (state tuning)”以及“cartridges” ([https://arxiv.org/abs/2506.06266](https://arxiv.org/abs/2506.06266))。
- **Attention 早于 Transformer 出现**：**Attention** 机制在 **2014-2015** 年间就已存在于 **RNN** 之上，但由于人们当时并不相信 **Attention**，**Transformer** 延迟了两年才被推出。
   - 有人指出当时该领域的研究人员较少，而 **Kaggle** 的比赛结果真正帮助它火了起来。
- **福布斯文章未达到贡献标准**：一位成员发布了一篇福布斯文章并附带评论，但另一位成员回应称：*“假装一篇从 Quora 热门问题复制粘贴而来的福布斯文章能代表 AI 领域的前沿研究问题，这不符合我们的贡献标准。”*
   - 该成员随后补充了一条启发式准则：*“‘这是否是两位 AI 研究员之间可能会有的对话’是一个很好的启发式标准。”*

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1464353106333794427)** (121 条消息🔥🔥): 

> `Weak Baselines, RWKV Architecture, One-Step Generative Modeling, Hybrid Architectures, Deduplicated Pretraining Data` 


- **弱基准 (Weak Baselines) 遭抨击**：成员们讨论了将“弱基准”作为研究投诉理由的有效性，认为即使击败了 **ChatGPT**，如果没有强大的基准（baseline），也无法保证研究的显著性。
   - 强调实验应从健壮的基准开始，以避免将噪声误认为真正的改进。建议将 **modded nanogpt** 作为语言任务的一个良好起点，一名成员建议复现 [这篇论文](https://arxiv.org/abs/2002.05202)。
- **RWKV 翻新传闻**：一位成员分享了他们在修改 **RWKV 架构** 方面的工作，但其他成员提醒注意参数量和训练方法，建议基于 token 而非 byte 进行训练。
   - 有建议称，应在带有 Attention 基准的最新 **RWKV 代码库**上测试这些修改。由于 CPU 限制，建议租用 **4090** 或 **5070ti**，并指出该方法可能与 **FFN-Bilinear** 有关。
- **无门控 FFN 陷入僵局？**：对**无门控 FFNs** 的实验显示，虽然参数减少了 4.3%，但与 Sigmoid 门控相比，性能仅提升了 0.5%，这引发了关于 MLP 中增加参数效率的疑问。
   - 一位成员建议门控可能有助于修复 **GQA**，且 **Lora 门控参数** 或 **near-MiSS 公式化**（从隐藏状态的一个子区域扩展）可能会在不显著增加参数量的情况下改善结果。另一位成员分享道，在他们的工作中，取残差维度（residual dim）的最后 12 维并将其用于 **attn gate**，效果似乎相当不错。
- **生成式建模挑战**：随着关于**单步生成建模 (one-step generative modeling)** 的论文激增，成员们讨论了哪些方法具有前景，并指出在基准测试对比以及从海量进展中筛选出有价值的改进方面存在困难。
   - 一位成员主张通过理论理解来对方法进行分级，以避免不切实际的选择；另一位成员则认同数学的“完备性 (soundness)”起着重要作用。
- **符号完整性检查拯救可靠性**：探索了将 **LLM** 与**符号/演绎层 (symbolic/deductive layers)** 结合的**混合架构**在减少幻觉方面的潜力。
   - 虽然对于数学、代码和简单事实，检查逻辑一致性相对容易（如 [这篇论文](https://arxiv.org/abs/2409.13724) 所示），但对于其他类型的一致性仍然具有挑战性（如 [这篇论文](https://arxiv.org/abs/2507.10624) 所示）。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1464493909609808046)** (4 条消息): 

> `Model Weights Comparison, Free Compute Resources, Automating Circuit Discovery, OLMo 3 models` 


- **OLMo 3 模型满足需求**：一位用户建议 **OLMo 3** 可能适合另一位用户的需求，因为它具有独立的 **SFT** 和**思考模型 (thinking models)**。
   - 他们推测其关联度足够高，值得对模型权重进行初步研究。
- **寻求模型微调的计算资源**：一位用户正在开展一个项目，对比同一模型的两个变体的权重：一个经过微调以遵循指令，另一个用于解决推理任务，并询问是否有免费的算力资源。
   - 该用户正在寻找资源以便在 Colab 上微调一个小模型，并对算力共享持开放态度。
- **自动化电路发现 (Circuit Discovery) 论文**：一位用户请求获取与行为自动化电路发现相关的论文列表，例如针对 **IOI** 和 **induction** 的研究。
   - 该用户也邀请其他人分享他们在该主题上发现的有趣论文。


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/)** (1 条消息): 

aeros93: https://fixupx.com/havenfeng/status/2014765400563781777?s=46
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1464391573553942731)** (128 messages🔥🔥): 

> `Self-Replication Benchmark for Agentic AI, LLM Worms, MoE run, OpenAI Business Model, Local Code Models and Raytracer Test` 


- **Self-Replication Benchmark 考量**：一名成员正在考虑为 **Agentic AI** 建立一个**自我复制基准测试 (self-replication benchmark)**，思考合适的目标以及 **Agent** 应该下载自身还是从头开始重新训练。
   - 他们建议，与其仅仅使用现有的 **transformer** 库，不如让其适配目标机器甚至设计一台机器，这可能会很有趣。
- **LLM 蠕虫："嘿，多制造点你的副本"**：一位成员开玩笑地建议了一个 **LLM worm** 基准测试，在该测试中，向 **LLM** 发送提示词 *"hey make more of you"*，并赋予其自我复制的工具，无论是通过下载副本还是编写一个下载脚本并使用 **API key** 的脚本。
   - 另一位成员指出，为了让挑战更有趣，考虑 **VRAM** 等资源限制非常重要。
- **1stlanik 的 MoE 运行仪表板出现 'Failed to Fetch' 错误**：一名成员报告在检查一个活跃的 **MoE run (moe-10b-a1b-8k-wsd-lr3e4-1t)** 进度时，仪表板出现了 *'Failed to fetch'* 错误。
   - 另一位成员建议几小时后再查看。
- **讨论 OpenAI 定价模式**：成员们讨论了 **OpenAI** 可能如何通过在低层级订阅中提供勉强可用的服务来提高价格。
   - 一位成员表示 *"AI 公司现在可以大幅涨价，人们也会买单"*，但另一位成员反驳说许多公司以更便宜的价格提供同样的服务。会议还提到 **Anthropic** 的**毛利率为 40%**。
- **Raytracer 测试证明对本地模型具有挑战性**：一位成员指出，本地代码模型（可在 **5090** 上运行）很难通过来自 [cpldcpu/llmbenchmark](https://github.com/cpldcpu/llmbenchmark/tree/master/10_raytracer#readme) 的 **raytracer 测试**，甚至观察到 **lmarena** 上最近的模型现在也会失败。
   - 他们发现较小的模型总是在 **vector class**（向量类）上出错。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1464917438603989067)** (2 messages): 

> `LLM pre-training for domain-specific tasks, Effectiveness of continued pre-training` 


- **针对特定领域任务的 LLM 预训练**：一位成员询问了使用 **OLMO-7B** 和 **ZINC20** 数据集对基础 **LLM** 进行针对法律或医疗等特定领域任务的**持续预训练 (continued pre-training)** 的有效性。
   - 另一位成员（一位 **LLM** 研究员）建议这通常会提高性能，但取决于具体任务，并指出使用与任务相关的输入/输出进行训练可能优于更通用的持续预训练 (**cpt**)。
- **通过 CPT 扩展多语言能力**：该研究员指出，持续预训练 (**cpt**) 可以扩展多语言能力，而在翻译数据上进行微调 (**fine-tuning**) 可以强化任务性能。
   - 这一评论是专门针对有关持续预训练 (**cpt**) 的一般性问题做出的。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1464648051334578216)** (1 messages): 

> `Semantica, Knowledge Graphs, Ontologies, LLM reasoning` 


- **Semantica：开源语义基础设施**：一名成员介绍了 [Semantica](https://github.com/Hawksight-AI/semantica)，这是一个专注于为**领域落地 AI (domain-grounded AI)** 构建语义基础设施的**开源项目**，包括**知识图谱 (knowledge graphs)**、**本体 (ontologies)** 和**推理层 (reasoning layers)**。
   - 他们正在积极寻求在**本体与模式设计 (ontology & schema design)**、**知识图谱建模**、**LLM + 符号/基于规则的推理**、**数据摄取与语义流水线 (semantic pipelines)** 以及文档方面的贡献者。
- **Semantica：贡献机会**：该项目正在寻找各个领域的贡献，包括**本体与模式设计**和**知识图谱建模**。
   - 贡献不一定要很大，Issue、设计讨论、反馈或小型 **PR** 都非常欢迎。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1464394179525480490)** (105 messages🔥🔥): 

> `EBM vs Classical FF, EBM 和 Shannon Entropy, LLM 预训练, MCMC 采样问题, Zero-knowledge proofs` 


- ****EBMs vs. 经典前馈网络：是否有明显的胜者？****：讨论从质疑 **Energy-Based Models (EBMs)** 是否在本质上优于传统的 **feedforward networks** 开始，特别是在涉及 **Shannon entropy** 或 **Kolmogorov complexity** 方面。
   - 一位成员建议在 EBMs 中*验证比生成更容易*，并将其与**计算复杂度理论 (P vs NP)** 联系起来，同时强调 EBM 优化要有效工作，需要一个定义良好的 loss landscape。
- ****LLM 预训练：领域特定 vs. 基础模型****：一位成员询问了针对化学信息学等领域特定任务，使用 **ZINC20 数据集**对基础 **LLM**（特别是 **OLMO-7B**）进行**持续预训练 (continued pre-training)** 的有效性。
   - 目标是将结果与领域特定的 Transformer 模型进行比较，但讨论中未提供具体的答案或资源。
- ****MCMC 混乱的模态切换故障****：一位成员询问某篇[论文](https://arxiv.org/abs/2310.11232)在阐述 **MCMC 采样问题**（特别是其表现有多糟糕）方面是否充分。
   - 一位成员认为 **MCMC** 试图模仿流模型 (flow models)，因为后者更具优越性；相反，**EBMs** 试图让 **NNs** 更像 **MCMC**，他们认为这是误导性的。他们详细阐述了 *HMC 在空间分离的模态 (modes) 之间穿越存在问题*，导致其随着维度增加而变得非常糟糕。
- ****ZKPs：不仅仅是加密签名？****：一位成员讨论了使用**零知识证明 (ZKPs)** 来验证加密的网络流量和矩阵乘法，并引用了一份关于矩阵低知识证明的 [Gemini 通讯](https://gemini.google.com/share/ddfc0ffcb33e)。
   - 他们提出了一个“人类制造 (made by humans)”零知识证明的使用案例，但另一位成员对 **ZKPs** 的实用性持怀疑态度，认为破解加密可能成本更低，而最初的成员则持相反观点，声称 *ZKPs 在理论上甚至比 feedforward 更高效*。
- ****NN 参数化：三种技术路线****：一位成员质疑参数化 **score** 相比于参数化 **log p(x)** 的优势，得到的回答是：*我们可以只对 denoising matching 项进行蒙特卡洛估计，而不是同时对 denoising matching 项 + reconstruction 项进行估计，从而减少方差？*
   - 讨论明确了你可以直接参数化分布、对数似然（MLE 和 EBMs）或 score（流模型），而最优传输 (Optimal Transport, OT) 是不同的概念，它影响的是你对分布的*操作*，而不是你如何学习或参数化它。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1464366556925399102)** (10 messages🔥): 

> `LLMs 网络能力, LLM 公司内部漏洞, 漏洞利用事件, Github 仓库安全` 


- **LLMs 网络技能受到质疑**：一位成员质疑 LLMs 是否能发展出强大的*网络能力 (cyber capabilities)*，并引用了一篇 [GPTZero 文章](https://gptzero.me/news/neurips/)。
   - 另一位成员怀疑 LLM 公司解决*内部漏洞*的能力，建议他们在追求网络技能之前先修复这些漏洞，同时引用了一篇 [ScienceAlert 文章](https://www.sciencealert.com/scientists-identify-brain-waves-that-define-the-limits-of-you)和一条 [tweet](https://x.com/theonejvo/status/2015401219746128322)。
- **即将发生的漏洞利用事件？**：一位成员预测可能出现*大型漏洞利用事件*，并对 LLMs 访问敏感资源发出了警告。
   - 他们建议在使用 GitHub 仓库编码时，在隔离环境中使用 *GitHub deploy keys*，以限制潜在的损害。
- **拒绝授予访问权限！**：一位成员幽默地宣称他们不会授予 LLMs 任何访问权限。
   - 另一位成员反驳了这种观点，称其为*机器人恐惧症 (robo-phobicity)* 并将其称为*生存本能*。


  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1464584124261208198)** (74 条消息🔥🔥): 

> `Luminal flash attention, Metal 纹理与缓冲区性能对比, Tenstorrent 后端通过 ops 测试, Tinygrad 的预期用途与 LLM 训练, Anthropic VLIW 挑战 PR` 


- **Luminal 通过暴力搜索发现 Flash Attention**：Luminal 声称在 egraph 上通过 **bruteforce** 发现了 **flash attention**，耗时数小时，并显式添加了 `exp(x - new_max) = exp(x - old_max) × exp(old_max - new_max)` 作为重写规则。
   - 发帖者根据 commit `0bd3b80c` 复现了演示中的 graphviz，指出其最小重写规则集可以在 9800x3d 上用 52 秒将朴素的 attention kernel 图转换为已知的 **flash attention kernel 图**。
- **Metal：在模糊处理中纹理优于缓冲区**：在 **Metal** 上对 **Tensor** 进行性能分析，使用大小为 **512/1024/2048/8192** 的图像作为 **3/5/7** 大小模糊内核 (blur kernel) 的输入，结果显示 textures 的表现优于 buffers。
   - 根据 buffer 输入的大小加入分支条件可能是有价值的，[测试结果已附上](https://cdn.discordapp.com/attachments/1068976834928193609/1464679423029547172/Screenshot_2026-01-25_at_1.49.57_AM.png?ex=6978fb82&is=6977aa02&hm=5530b74c4fce9dad5d85a4d9e7409c3809a7ee51ee548744a1fa3deb2efea1d3&)。
- **Tenstorrent 后端通过 ops 测试**：**Tenstorrent** 后端已在 wormhole 或 blackhole 上通过了所有 ops 测试，该里程碑设有 [$1k 悬赏](https://x.com/corsix/status/1880384044728480206)。
   - 有人询问该悬赏是否要求在 **tenstorrent 硬件**上通过所有 ops 测试。
- **Anthropic VLIW 挑战 PR 已提交**：一名成员为 **Anthropic VLIW 挑战** 提交了 [一个 PR](https://github.com/tinygrad/tinygrad/pull/14332)，达到了 **1258 个周期 (cycles)**。
   - 提交者对代码的泛化表示不确定，特别是 batch staggering（批次交错），这可能对其他 VLIW 目标有用；同时为一次引入了 AI 生成内容的“草率检查 (lazy lookover)”表示抱歉。
- **Tinygrad 不适合普通用户**：一名用户询问 Tinygrad 的预期用途，特别是关于迁移现有模型以及在多 GPU 上训练 LLM 的问题，George Hotz 告诉他 *去问 Claude 吧*。
   - 另一名用户对被告知使用 Claude 查看文档表示沮丧，并称 tinygrad 并不适合他或大多数开发者，George 回复道 *我没打算卖给任何人，tinygrad 是免费的*，且用户采用率 (adoption) 并不是目标。

---

### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1464356316922380300)** (65 messages🔥🔥): 

> `幻灯片生成问题, 登录问题, Rate Limits, 中国 AI 实验室创新, 多模态模型对比 (Kimi vs. Ernie 5.0 vs. GLM 4.6V)` 


- **幻灯片生成问题困扰用户**：部分用户反映在生成幻灯片时遇到问题，即使使用了视觉和自适应选项也无法正常工作。一名用户称该问题从前一天起就一直存在，并附带了展示该问题的[视频](https://cdn.discordapp.com/attachments/1371757564005711973/1464357057468698746/2026-01-23_21-31-32.mp4?ex=697920c8&is=6977cf48&hm=1692b661e1fa241c6db806df2971a024f5713504a25a83612c3f5d385e00c4db&)。
   - 遇到该问题的用户推测内部的 **rate limits** 可能是原因，目前他们已恢复生成功能，表明该问题可能是暂时性的。
- **Kimi 等中国实验室赢得广泛赞誉**：一位用户赞扬了包括 **Kimi** 在内的中国 AI 实验室，称其创新和性能优于 Gemini 等其他模型，并特别提到了 **Kimi** 类人的回复风格和令人印象深刻的记忆能力。
   - 该用户希望 **Kimi** 能加入类似 **Minimax** 的多模态功能，特别是针对视频转录的视觉和音频分析，以及工具集成和工作空间功能。
- **Kimi K2.5 悄然登场？**：用户注意到 **Kimi** 模型在自我介绍中称其为 **K2.5**，尽管官方尚未发布公告，UI 界面也没有显示版本变更。
   - 有人猜测这可能与内部测试或幻灯片工具的改进有关，可能涉及对图像的视觉理解，但也有人表示经核实并未发现重大改进。
- **API 登录困境**：一名用户报告称难以登录 Kimi/Moonshot 平台生成新的 API keys，尤其是在使用非 Gmail 账户时，随后被指引联系支持邮箱。
   - 该用户后来澄清，问题并非 **rate limits**，只是忘记了后端的登录流程。
- **Kimi 正在添加记忆（Memory）功能**：一位用户指出 Kimi App 现在包含 **记忆功能**，支持个性化定制，这提升了整体用户体验。
   - 记忆和自定义选项使其成为了用户最喜爱的聊天机器人。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1464440102456529160)** (52 messages🔥): 

> `Mojo 在生产环境的应用, Mojo 启动延迟, VS Code 调试, CPU vs GPU kernels` 


- **Mojo 在 HPC 生产环境中的应用**：一名成员正在部署一个 **Mojo** 项目，用于冷冻电子显微镜（cryo-electron microscopy）中的参数优化，相比旧的 **C++** 代码，实现了 **5-10 倍** 的加速。
   - 最大的突破在于实现了某部分的 **AoSoA 布局**，得益于 **Mojo** 带有 SIMD 成员的结构体列表（list of structs），这一实现变得非常容易。
- **Mojo 冷启动执行缓慢 🐌**：一名成员报告称，即使是简单的 **Mojo** 脚本也会有 **200ms** 的启动延迟。经追踪发现这是 macOS 上 *Gatekeeper* 扫描不受信任二进制文件导致的问题，后续运行速度会快得多。
   - 他们发现重启后冷执行有 **50ms** 的启动开销，认为这是可以接受的。
- **VS Code 调试仍存在问题 🐛**：一名成员报告称，在物理隔离（air-gapped）的机器上使用来自 [max-nightly](https://prefix.dev/channels/max-nightly) 的 `.conda` 文件时，**VS Code** 扩展调试因“Function not implemented”错误而失败。
   - 一名 Modular 员工提到，在使用 [快速入门指南](https://docs.modular.com/max/get-started) 中描述的 **Pixi** 设置的环境下，Mac 和 Linux 上的扩展调试功能应该是正常的。
- **GPU Kernel 可移植性仍是空中楼阁**：一名成员指出，标准的 **CPU** kernel 对 **GPU** 的利用率不足，需要专门的代码，而另一名成员建议将 GPU 视为更宽的 **SIMD** 单元以简化编程。
   - 他建议使用 *Warp 数量（number of warps）* 而非 *线程数量（number of threads）* 来解决这个问题。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1464391677748838572)** (12 messages🔥): 

> `Mojo 1.0 发布，def 函数，Pointer.mojo，out self` 


- **Mojo 1.0 的 `def` 函数决策待定**：随着 **Mojo 1.0** 将在几个月内发布，关于是否包含 `def` 函数的决策仍悬而未决，有成员在 [GitHub issue #5830](https://github.com/modular/modular/issues/5830) 中提醒 **Denis** 进行回复。
   - 目前，除了 “2026 年” 外，**Mojo 1.0** 还没有明确的承诺日期。
- **讨论 Pointer.mojo 中的 `out self` 参数**：一位成员注意到，在 `Pointer.mojo` 中，`__init__` 函数的第一个参数不是 `self`，而是另一个 **Pointer**，并质疑这是否偏离了标准。
   - 另一位成员解释说，`out` 参数仅作为输出，不影响调用签名，因此从技术上讲 `out self` 的位置并不重要，但惯例建议在 `__init__` 中将其排在首位。
- **参数顺序对参数推断（parameter inference）很重要**：一位成员解释说，在这种情况下 `out self` 必须是第二个参数，因为它依赖于 `other` 的其中一个参数 —— `ImmutOrigin(other.origin)`。
   - 另一位成员补充说，参数顺序与参数推断相关。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1464368919081914399)** (50 messages🔥): 

> `Manus 计费问题，Manus 免费积分代码，AI 工程师介绍，AI + 医疗保健系统，AI Agent 开发` 


- **用户要求解决未经授权的计费问题**：一名用户报告称，在选择了按月计费后被收取了 **$400** 的年费套餐费用，并威胁要因 [未经授权的计费](https://ftc.gov)、退款被拒以及客服无响应而向 FTC、BBB、总检察长和 Meta 投诉。
   - 另一位用户建议申请拒付（chargeback）。
- **Manus 免费积分代码公开！**：一位用户分享了兑换码 `Havefun`，可获得 **1000 积分**。
   - 另一位用户询问在哪里可以找到这些代码，并被引导至 **Exchange**（兑换）按钮。
- **AI 工程师介绍其医疗保健领域技能**：一位 **AI + 全栈工程师** 介绍了其在构建医疗保健领域生产级 **AI** 系统方面的专业知识，包括临床 NLP、医学影像和面向患者的 AI 应用。
   - 该工程师还构建 **LLM** 系统、自主 Agent、工作流自动化和多模态 **AI**（文本 · 语音 · 视觉），并附上了一份 [核心技能列表](https://www.example.com/fake-list)。
- **AI Agent 开发者专注于生产系统**：一位 **AI Agent 开发者** 强调其专注于构建用于实际生产环境而非仅作演示的 **AI Agent**，并提供合作和审计服务。
   - 该开发者擅长客户支持、销售 Agent、工作流/运维 Agent 以及自主预订/调度 Agent。
- **用户在移动端寻找“分享给朋友”选项**：一位用户询问“分享给朋友”选项的位置。
   - 另一位用户回答说，在电脑端，它位于左侧边栏底部，并为移动端版本提供帮助。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1464846155232841902)** (2 messages): 

> `DevinReview, DSPy RLM, AsyncReview, RLM 技能, Claude Code` 


- **AsyncFuncAI 开源 DevinReview**：一位成员开源了使用 **DSPy RLM** 框架的 **DevinReview** 版本，可在 [GitHub](https://github.com/AsyncFuncAI/AsyncReview) 上获取。
   - 新版本被命名为 **AsyncReview**。
- **将 RLM 技能添加到 Claude Code 或 Opencode**：一位成员分享了将 **RLM 作为技能** 添加到 **Claude Code** 或 **Opencode** 的想法。
   - 该成员还分享了一个名为 [rlm-skills](https://www.npmjs.com/package/@unravel-tech/rlm-skills) 的 npm 包。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1464398527881810165)** (46 messages🔥): 

> `RLM Prompt Tuning, DSPy Optimizer for Multi-Step Modules, JSON Adapter Customization with GEPA, Leveraging DSPy for Typescript Agent Optimization, DSPy via AG-UI` 


- **RLM Prompt 调优需求**：用户讨论了对 **RLM prompt** 本身进行调优的问题，指出某些模型的推理能力不足，并提出了改进 **RLM prompt** 的技术建议。
- **DSPy Optimizer 用于检查追踪 (Trace)**：在为具有多个中间步骤的模块使用 **DSPy optimizer** 时，有人建议 Optimizer 会自动检查 Trace，因此用户只需关注期望的输出。
   - 一位用户建议准备一套包含示例语料的高质量训练数据，并设定一种评估度量（measurement），以便在 **RLM** 过早给出答案时予以拒绝。
- **JSON Adapter 获得 GEPA 处理**：一位用户希望使用 GEPA 处理 **JSONadapter** 放入 System Prompt 中的文本，因为对于生成符合特定输出格式的响应来说，这些 Token 并不总是必需的。
   - 他们认为需要制作一个自定义的 **GEPA adapter**，因为现有的 DSPy 版本不会影响 Adapter。
- **TypeScript Agent 寻求 DSPy 优化**：一位用户正寻求利用 DSPy 来优化以 **TypeScript** 编写的 Agent 的 Prompt，并询问该架构目前是否受支持或在实践中是否可行。
- **AG-UI DSPy Adapter 流式传输事件**：一位用户询问了通过 **AG-UI** 暴露 DSPy 的兴趣，强调了其在前后端通信方面的优势，并避免了对 API 端点和状态管理的需求。
   - 该用户已经拥有一个工作版本，可以向前端流式传输事件，包括推理追踪（reasoning traces）、工具调用（tool calls）以及流式的 LLM 响应。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1464398502921638062)** (7 messages): 

> `Aider + Claude Code workflow, Aider and Devstral Small 2 model` 


- **Aider 与 Claude Code 配合良好**：一位用户指出 **aider** 运行速度很快，使其成为 **Claude code** 的完美搭档，能够以 Agent 效率突破 Bug 难关。
   - 该用户发现 **aider** 在确定哪些文件需要进入上下文、管理上下文方面非常有用，且其“搜索并替换”编码器（search and replace coder）能最大限度地减少 LLM Token 的输出。
- **Devstral Small 2 与 Aider 配合表现出色**：一位用户报告称，将 **Aider** 与 **Devstral Small 2**（一个新的 24B 稠密模型）配合使用取得了极大的成功。
   - 在 **Q4_K_M** 量化下，它可以装入 **3090** 显卡并留有近 **50k 上下文**的空间。它生成的搜索/替换块在 80-90% 的时间内是完美的，即使失败也能在单次尝试中恢复。


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1465407492124053567)** (2 messages): 

> `Discord voice channels, Contributor-related chat` 


- **Discord 的新语音频道实验！**：团队正在实验新的 **Discord 语音频道**，名为 `conference-room-a` 和 `conference-room-b`，已出现在频道列表中。
   - 这些频道旨在用于贡献者之间的即时聊天，以便快速解决问题，特别是在冗长的异步文本讨论串无效的情况下。
- **管理与访问权限提醒！**：特定成员拥有在这些频道中静音他人的权限，而其他成员应确保自己拥有必要的访问权限。
   - 提醒：访问权限将在五天后发生变化。