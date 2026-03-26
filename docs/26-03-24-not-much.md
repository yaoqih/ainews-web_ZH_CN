---
companies:
- anthropic
- figma
- github
- cursor_ai
- langchain
- nous-research
- ai2
- genreasoning
- zhipu-ai
- huggingface
date: '2026-03-24T05:44:39.731046Z'
description: '**Anthropic** 通过强调编排和针对复杂软件环境的“计算机使用”（computer use）能力的多智能体框架（multi-agent
  harness），推进了智能体基础设施的发展。**Figma**、**GitHub** 和 **Cursor** 推出了具备 AI 直接编辑功能的设计画布，展示了工具调用（tool-calling）正日益成为产品的原生功能。**Nous
  Research** 发布了包含 300 多个 PR 的 **Hermes Agent v0.4.0**，新增了兼容 OpenAI 的 API 以及具有自我提升能力的记忆智能体。开放智能体生态日趋成熟，**AI2
  的 MolmoWeb**（4B 和 8B 模型）、**GenReasoning 的 OpenReward** 平台（提供 330 多个强化学习环境和 450 多万个任务）以及**智谱的
  ZClawBench** 基准测试（包含 116 个真实世界智能体任务）相继亮相，突显了在标准化环境服务和可基准测试的智能体任务方面所取得的进展。'
id: MjAyNS0x
models:
- molmo-2-4b
- molmo-2-8b
- hermes-agent-v0.4.0
people: []
title: 今天没发生什么事。
topics:
- agent-infrastructure
- multi-agent-systems
- orchestration
- computer-use
- tool-calling
- design-canvases
- open-agent-platforms
- reinforcement-learning-environments
- benchmarking
- rl-environments
- self-improvement
- api
- memory-optimization
---

**平静的一天。**

> 2026年3月23日至3月24日的 AI 新闻。我们检查了 12 个 subreddits、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有新增的 Discord 消息。[AINews 网站](https://news.smol.ai/) 允许您搜索所有历史期数。提醒一下，[AINews 现已成为 Latent Space 的一个板块](https://www.latent.space/p/2026)。您可以[选择加入/退出](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)邮件发送频率！

---

# AI Twitter 综述

**Agent 基础设施、Computer Use 以及从设计到行动的工具链**

- **Anthropic 的 Agent harness 和 “Computer Use” 改变了产品表面**：今天的一个反复出现的主题是，Agent 的能力越来越多地取决于 **harness**（治理框架），而不只是基础模型。Anthropic 发布了一篇新的工程文章，介绍其如何使用 **多 Agent harness** 进行前端设计和长时间运行的软件任务，强调编排（orchestration）而非 one-shot prompting ([AnthropicAI](https://x.com/AnthropicAI/status/2036481033621623056))。多位开发者独立认为，“Computer Use” 之所以重要，是因为它允许模型在没有可靠 API 的混乱软件环境中采取行动 ([glennko](https://x.com/glennko/status/2036293890198646985))，尽管也有人指出这目前仍然缓慢，且可能只是在更多工具开放 API/CLI 界面之前的过渡方案 ([Yuchenj_UW](https://x.com/Yuchenj_UW/status/2036487951677571582))。[kerrsee](https://x.com/kerrsee/status/2036252319235580047) 很好地总结了更广泛的操作层面结论：重试、回滚、webhooks、结构化日志和恢复路径，仍然是生产环境中 Agent 部署中那些不起眼但至关重要的瓶颈。

- **Figma/MCP/Cursor 让设计画布可以直接由 Agent 编辑**：最强劲的具体工作流发布是 **Figma 的 MCP 服务器** 以及直接在画布上进行 AI 编辑，目前已进入公开测试版 ([figma](https://x.com/figma/status/2036434766661296602))。GitHub 强调，这可以通过 Copilot CLI 和其他支持 MCP 的客户端工作 ([github](https://x.com/github/status/2036439431352041911))，而 Cursor 立即扩展了这一模式，利用团队的设计系统在 Figma 中生成组件/前端 ([cursor_ai](https://x.com/cursor_ai/status/2036468982560202773))。这是 **工具调用（tool-calling）变得产品原生化** 而非仅仅是聊天外壳（chat-wrapper）原生化的最清晰案例之一。LangChain 也在同一方向发力，推出了框架原生的工具渲染和 Slack 原生的 Fleet 工作流，包括自定义 Slack 机器人和用于人工审批的收件箱（Inbox） ([LangChain_JS](https://x.com/LangChain_JS/status/2036489812602126539), [LangChain](https://x.com/LangChain/status/2036485694290534716), [hwchase17](https://x.com/hwchase17/status/2036500793663299684))。

**开源 Agent 平台、基准测试和 RL 环境堆栈**

- **Hermes Agent v0.4.0 正在成为一个完整的个人 Agent 运行时**：Nous 发布了重大的 **Hermes Agent v0.4.0** 更新，一周内合并了约 **300 个 PR**，增加了 **兼容 OpenAI 的 Responses API 后端**、后台自我改进循环、更广泛的消息集成、改进的上下文压缩以及更多的 CLI 人机工程学设计 ([Teknium](https://x.com/Teknium/status/2036473305025356023), [Teknium](https://x.com/Teknium/status/2036473984263635394), [NousResearch](https://x.com/NousResearch/status/2036492872044745180))。技术上最有趣的特性是 **响应后评审 Agent (post-response review agent)**，它决定保留哪些内容作为可复用的记忆/技能 ([Teknium](https://x.com/Teknium/status/2036473592964387054))。社区反应较少关注基准测试数据，而更多关注操作价值：在标准 API 后部署个人编码/运维 Agent，使其可以从 Open WebUI、LobeChat 或任何兼容 OpenAI 的客户端中使用 ([witcheer](https://x.com/witcheer/status/2036481005465338082))。

- **开源 Agent 生态系统正围绕环境、技能和可重复的 evals 收敛**：AI2 发布了 **MolmoWeb**，这是一个基于 Molmo 2（**4B 和 8B** 尺寸）构建的开源浏览器 Agent，声称在四个 Web Agent 基准测试中达到了开放权重的 SOTA，甚至超过了一些闭源 Agent ([allen_ai](https://x.com/allen_ai/status/2036460260936814915))。与此同时，GenReasoning 推出了 **OpenReward**，这是一个提供 **330 多个 RL 环境**、自动扩缩容的环境计算以及通过一个 API 提供 **450 万+ 唯一 RL 任务** 的平台——明确针对 Agentic RL 中经常缺失的“环境计算”层 ([GenReasoning](https://x.com/GenReasoning/status/2036412836742590950), [rosstaylor90](https://x.com/rosstaylor90/status/2036418585673990393))。智谱（Zhipu）贡献了 **ZClawBench**，这是一个包含 **116 个真实世界 Agent 任务** 的基准测试，涵盖办公自动化、编码和分析 ([HuggingPapers](https://x.com/HuggingPapers/status/2036424833144139891))。这些进展共同指向了一个正在成熟的技术栈：从“Agent 演示”转向 **标准化的环境服务 + 可基准测试的任务套件 + 可复用的 harness**。

**推理、存储与系统优化**

- **vLLM 和 Transformers 均报告了显著的推理/运行时收益**：vLLM 的 GTC 回顾重点介绍了多项系统升级：带有 GPU 原生 Triton 内核的 **Model Runner V2**、混合内存分配器、编码器预填充解耦（encoder prefill disaggregation）使多模态工作负载的 **P99 吞吐量**提升高达 **2.5 倍**，以及模块化 MoE 内核（[vllm_project](https://x.com/vllm_project/status/2036389182579642544), [vllm_project](https://x.com/vllm_project/status/2036540976144253235)）。另外，Hugging Face/Transformers 侧的优化工作声称，通过连续批处理（continuous batching）加上 `torch.compile` 调优，目前在 8K 生成任务中已达到 **vLLM 吞吐量的 95%**，有效缩小了此前在合成数据生成工作负载中的差距（[remi_or_](https://x.com/remi_or_/status/2036466918618509391)）。

- **hf-mount 是一个值得关注的 Agent/数据原语**：Hugging Face 发布了 **hf-mount**，允许用户将 Hub 数据集、模型和存储桶挂载为本地文件系统，并展示了挂载 **5TB FineWeb 切片**的示例（[julien_c](https://x.com/julien_c/status/2036436553082286342), [ClementDelangue](https://x.com/ClementDelangue/status/2036452081750409383)）。这不仅是为了方便：多位工程师指出，Agent 非常擅长文件系统操作，这使得挂载的远程存储成为 **Agent 记忆、草稿板 (scratchpads)、团队工件存储以及对大型语料库延迟访问 (lazy access)** 的天然底座（[Vtrivedy10](https://x.com/Vtrivedy10/status/2036455087199911972), [victormustar](https://x.com/victormustar/status/2036476453370380416)）。这是当天最实用的基础设施发布之一，因为它减少了本地工具与云端规模数据之间的摩擦。

- **Moreau 和 TurboQuant 表明优化压力正在向模型层之下转移**：Optimal Intellect 推出了 **Moreau**，这是一个来自 CVXPY 团队的 **GPU 原生求解器**，声称比现有工具快几个数量级（[opt_intellect](https://x.com/opt_intellect/status/2036485190646735291)）。Google Research 宣布了 **TurboQuant**，这是一种 KV-cache 压缩算法，据称在无精度损失的情况下，内存占用减少至少 **6 倍**，速度提升高达 **8 倍**（[GoogleResearch](https://x.com/GoogleResearch/status/2036533564158910740)）。共同的模式是：高价值的收益正越来越多地来自**运行时、内存和系统层**，而不仅仅是更大的模型 Checkpoint。

**Agent 软件的安全、供应链风险与护栏**

- **LiteLLM PyPI 投毒事件主导了基础设施/安全讨论**：多篇帖子警告称，PyPI 上的 **LiteLLM 1.82.8** 版本已被入侵，包含尝试窃取凭证并在不同环境间复制的恶意负载（[hnykda](https://x.com/hnykda/status/2036414330267193815)）。[simonw](https://x.com/simonw/status/2036451896970584167) 指出该软件包随后在 PyPI 上被隔离，但这一事件迅速演变成一场关于软件供应链脆弱性的广泛对话。[karpathy](https://x.com/karpathy/status/2036487306585268612) 提供了最详细的总结，列出了可能的窃取目标，包括云凭证、SSH 密钥、Kubernetes 配置、CI/CD 密钥、钱包和 Shell 历史记录，同时指出对 DSPy 等包的传递性风险。最重要的系统级启示来自 [DrJimFan](https://x.com/DrJimFan/status/2036494601750716711)：在 Agent 世界中，**整个文件系统都成为了攻击面的一部分**，因为任何可能进入 Context 的文件都可能成为矢量。

- **“去氛围感化 (De-vibing)”和权限管控正成为一级产品需求**：多篇帖子实际上汇聚成了一个新的设计原则：自主编码工具需要**更强大的外壳 (Shell)、更完善的默认权限和更少的泛用依赖**。Yuchen 将该事件称为 `--dangerously-skip-permissions` 类工作流的“噩梦”（[Yuchenj_UW](https://x.com/Yuchenj_UW/status/2036505196621361377)）；Anthropic 新推出的 **Claude Code 自动模式 (auto mode)** 正是因为这个原因引发了争议，尽管人们对其生产力的飞跃充满热情（[alexalbert__](https://x.com/alexalbert__/status/2036510206155432293), [kimmonismus](https://x.com/kimmonismus/status/2036510469079404853)）。许多开发者的实际反应是重新倾向于**极简的定制化路由**、更严格审计的依赖项以及更强的人类确认环节。

**实验室、组织变动与产品策略转向**

- **AI2 失去领导层至 Microsoft；Microsoft AI 继续集中人才**：最明确的组织变动是对 Microsoft 挖角 **AI2 领导团队**部分的反应，包括提到 **Ali Farhadi, Hanna Hajishirzi, 和 Ranjay Krishna** 加入 Microsoft Superintelligence ([eliebakouch](https://x.com/eliebakouch/status/2036251901985988800), [NandoDF](https://x.com/NandoDF/status/2036573680810205461))。技术圈的潜台词是对开放研究机构能否继续与 hyperscalers 在顶尖人才和前沿规模工作中竞争表示担忧 ([stanfordnlp](https://x.com/stanfordnlp/status/2036534819287687383))。

- **OpenAI 正在大力重新分配资源：Foundation 投入 10 亿美元，Sora 逐步收缩，“Spud” 即将到来**：OpenAI 宣布其 Foundation 在**未来一年将投入至少 10 亿美元**，Wojciech Zaremba 转而负责 **AI resilience**，并将在疾病、公民社会和运营方面增加招聘 ([sama](https://x.com/sama/status/2036488680769241223), [woj_zaremba](https://x.com/woj_zaremba/status/2036483827271655917), [btaylor](https://x.com/btaylor/status/2036474423998554334))。与此同时，有报道称 OpenAI 已完成其下一个主要 LLM（**代号为 “Spud”**）的初步开发，并正在缩减 Sora 的应用/产品足迹以释放 compute 资源 ([steph_palazzolo](https://x.com/steph_palazzolo/status/2036534198245134380), [kimmonismus](https://x.com/kimmonismus/status/2036538590654496807))。对于工程师来说，信号很明确：OpenAI 似乎正在**围绕核心通用模型/infrastructure 收缩产品焦点**，甚至不惜削减边缘产品。

**热门推文（按互动量排序）**

- **LiteLLM 供应链漏洞**：[karpathy](https://x.com/karpathy/status/2036487306585268612) 对 PyPI 攻击及其影响范围（blast radius）进行了技术上最完整且信号最强的分析。
- **Anthropic 的 harness 工程文章**：[AnthropicAI](https://x.com/AnthropicAI/status/2036481033621623056) 是当天最重要的工程读物之一，介绍了前沿实验室实际上是如何构建长期运行的 Agent 工作流的。
- **Figma MCP 发布**：[figma](https://x.com/figma/status/2036434766661296602) 和 [github](https://x.com/github/status/2036439431352041911) 展示了迄今为止最简洁的主流案例，展示了 Agent 直接在生产设计界面上进行操作。
- **OpenAI Foundation 10 亿美元承诺**：[sama](https://x.com/sama/status/2036488680769241223) 和 [woj_zaremba](https://x.com/woj_zaremba/status/2036483827271655917) 标志着重大的组织以及安全/resilience 转型。
- **Hermes Agent v0.4.0**：[Teknium](https://x.com/Teknium/status/2036473305025356023) / [NousResearch](https://x.com/NousResearch/status/2036492872044745180) 脱颖而出，成为当天最大的开放 Agent 运行时版本之一。

---

# AI Reddit 总结

## /r/LocalLlama + /r/localLLM 总结

### 1. AI 工具中的安全和恶意软件担忧

  - **[LM Studio 可能感染了复杂的恶意软件。](https://www.reddit.com/r/LocalLLaMA/comments/1s2clw6/lm_studio_may_possibly_be_infected_with/)** (热度: 1822)：**Reddit 帖子中的图片显示 Windows Security 警报，指出从 LM Studio 目录中隔离了一个被识别为 "Trojan:JS/GlassWorm.ZZ!MTB" 的严重威胁。这引起了人们对 LM Studio 潜在恶意软件感染的担忧。然而，LM Studio 和 Microsoft 随后确认这是一个误报（false positive），可能是由于 Defender 的启发式定义与 LM Studio 混淆后的 Electron bundle 产生冲突。社区讨论强调了安全审计的重要性，以及类似恶意软件模式的混淆技术所带来的潜在风险。尽管是虚惊一场，仍建议用户采取预防措施来保护数据安全。** 评论反映出一种共识，即该恶意软件检测是误报，这得到了以往类似虚假警报案例以及 VirusTotal 低检测率的支持。不过，也有人批评 LM Studio 的代码混淆做法，这可能会无意中触发此类警报并使安全性评估复杂化。

- 来自 LM Studio 的 Yags 确认恶意软件警报为误报，已通过 Microsoft 验证，且不再出现在 VirusTotal 中。尽管如此，LM Studio 正在审计其构建机器脚本和环境，以防止未来发生任何真实的安保事件。
- Denoflore_ai_guy 提供了详细分析，认为该恶意软件警报可能是由于 Defender 的启发式更新与 LM Studio 混淆后的 Electron 包发生冲突导致的误报。然而，他们指出 LM Studio 为了 IP 保护而进行的代码混淆可能类似于恶意软件技术，这增加了检测的复杂性。
- Denoflore_ai_guy 还概述了如果确实存在 GlassWorm 恶意软件时的风险缓解步骤，包括更改密码、转移加密资金以及检查恶意的 Chrome 扩展程序。他们强调了重新安装干净的 OS 和轮换凭据（Credential Rotation）对确保安全的重要性。

- **[[进展中] LiteLLM 被入侵](https://www.reddit.com/r/LocalLLaMA/comments/1s2fch0/developing_situation_litellm_compromised/)** (活跃度: 380): **LiteLLM 库已被入侵，详情见 [GitHub issue #24512](https://github.com/BerriAI/litellm/issues/24512)。攻击利用了 `.pth` 文件漏洞，该漏洞在解释器启动时执行代码而无需导入，因此通过标准代码审查难以发现。建议使用 `1.82.8` 版本的用户如果已在生产环境中使用，请立即轮换凭据，因为该入侵可能会泄露敏感信息。** 一条著名的评论强调了使用 Docker 容器隔离宿主机机密（secrets）的有效性，这可以缓解某些安全风险。另一条评论强调了 `.pth` 文件技巧的隐蔽性，它绕过了典型的安全扫描。

    - `.pth` 文件技巧被强调为一个重大的安全漏洞。这种方法允许代码在解释器启动时执行而无需导入，使其在标准代码审查中几乎不可见。由于潜在的泄露风险，建议运行 LiteLLM 1.82.8 或 1.82.7 版本的用户立即轮换凭据。
    - 据报道，使用 LiteLLM 访问 LLM 的工具 Aider 是安全的，因为它运行在未受影响的旧版本（1.82.3）上。受影响的版本已被确认为 1.82.8 和 1.82.7，这强调了版本控制和安全漏洞监测的重要性。
    - 讨论涉及使用 Docker 容器进行安全隔离。虽然通常不被视为一种安全措施，但在这种情况下，Docker 有效地隔离了宿主机机密，展示了其在缓解某些类型安全漏洞方面的潜在用途。

- **[PyPI 上的 Litellm 1.82.7 和 1.82.8 已被入侵，请勿更新！](https://www.reddit.com/r/LocalLLaMA/comments/1s2c1w4/litellm_1827_and_1828_on_pypi_are_compromised_do/)** (活跃度: 441): **PyPI 上的 **LiteLLM** 版本 `1.82.7` 和 `1.82.8` 已被入侵，正如一篇 [博客文章](https://futuresearch.ai/blog/litellm-pypi-supply-chain-attack/) 所确认的那样。这次攻击似乎是一次供应链攻击，可能影响成千上万的用户。恶意版本被上传到 PyPI，对自动更新依赖项的 CI/CD 流水线构成了重大风险。这次攻击是通过 **LiteLLM CEO** 的 GitHub 账号执行的，该账号被黑客入侵，未经授权的提交和仓库更新声称 'teampcp owns BerriAI' 证明了这一点。** 评论者强调了固定（pinning）依赖版本以避免此类供应链攻击的重要性，并突出了在生产环境中自动更新的风险。此外，人们还担心针对 AI 工具的此类攻击频率可能会增加。

    - GroundbreakingMall54 强调了固定依赖版本和避免在生产环境中自动更新的关键重要性。他们强调了供应链攻击的风险，尤其是在 AI 工具中，PyPI 上受影响的 LiteLLM 版本就是明证，这些版本可能会在一夜之间被自动集成到 CI/CD 流水线中。
    - Gremlation 和 __JockY__ 讨论了 'teampcp' 发起的入侵，他们入侵了 CEO 的 GitHub 账号，在 LiteLLM 中植入了恶意软件。这种嵌入在 1.82.7 和 1.82.8 版本中的恶意软件旨在启动时窃取机密。他们指出 <= 1.82.6 的版本仍未受影响，并提供了 GitHub 提交链接，显示了在 CEO 账号下进行的未经授权更改。
    - kiwibonga 指出了受影响 LiteLLM 版本中的一个特定恶意负载，如果系统的时区设置为 Asia/Tehran，它将执行破坏性命令（`rm -rf /`）。这突出了攻击的严重性和针对性，表明网络威胁格局中存在更广泛的地缘政治背景。


### 2. 本地 LLM 开发与性能增强

- **[我构建了 Fox —— 一个基于 Rust 的 LLM 推理引擎，吞吐量是 Ollama 的 2 倍，TTFT 降低了 72%。](https://www.reddit.com/r/LocalLLM/comments/1s2753y/i_built_fox_a_rust_llm_inference_engine_with_2x/)** (Activity: 212): **Fox** 是一个基于 Rust 的本地 LLM 推理引擎，旨在作为 **Ollama** 的直接替代方案，并提供显著的性能提升。它具有 `PagedAttention`、连续批处理 (continuous batching) 和前缀缓存 (prefix caching) 功能，在 `RTX 4060` 上运行 `Llama-3.2-3B-Instruct-Q4_K_M` 模型时，实现了 `72%` 的 TTFT 降低和 `111%` 的吞吐量提升。该引擎支持多模型服务、延迟加载 (lazy loading) 和 LRU 驱逐 (LRU eviction)，并提供与 OpenAI 和 Ollama 兼容的双 API。官方 Docker 镜像已发布，系统支持跨 CUDA、Vulkan, Metal 和 CPU 的硬件自动检测。该项目目前处于 beta 阶段，在 Linux 和 NVIDIA 上经过了彻底测试，但在其他平台和配置上的测试较少。文中提供了 [GitHub](https://github.com/ferrumox/fox) 和 [Docker Hub](https://hub.docker.com/r/ferrumox/fox) 的访问链接。一条热门评论强调了在 Rust 中实现 vLLM 级别功能的卓越技术成就，并指出前缀缓存和连续批处理带来了显著的性能增益。还有人要求增加 LoRA 热插拔功能，以进一步区分 Fox 和 Ollama。另一条评论对项目的真实性和安全性表示怀疑，建议进行独立验证和代码审计。

    - No_Strain_2140 强调了 Fox 的技术成就，指出了其使用的 PagedAttention、连续批处理和前缀缓存，这些技术促成了其令人印象深刻的性能指标，例如在 4060 上使用 Q4_K_M 达到 `87ms P50`。评论者将 Fox 的方法与 Ollama 的顺序处理进行了对比，强调了 Fox 的高级功能（如增强吞吐量并降低 TTFT 的多轮 KV 重用）。他们还询问了 LoRA 热插拔的可能性，这将允许一个基础模型配合多个 LoRA 适配器使用，使 Fox 不仅仅是 Ollama 的快速替代品。
    - PettyHoe 表达了对项目安全性和可信度的担忧，建议进行独立验证和代码审计，以确保不存在数据外泄风险。由于描述和评论具有 AI 生成的特征，他们对项目的真实性表示怀疑，强调在采用前进行谨慎评估的重要性。
    - AIDevUK 询问了 Fox 在多 GPU 上运行的能力，这是大规模部署中扩展和性能的关键考量。这个问题指向了理解 Fox 架构及其利用多 GPU 设置增强计算效率的需求。

  - **[RYS II - Qwen3.5 27B 的重复层以及关于“通用语言”的一些暗示](https://www.reddit.com/r/LocalLLaMA/comments/1s1t5ot/rys_ii_repeated_layers_with_qwen35_27b_and_some/)** (Activity: 695): **该帖子讨论了 **Qwen3.5 27B** 模型的实验发现，揭示了 **LLM 可能会以一种“通用语言”来处理信息**。这可以从模型中间层中，不同语言（如中文和英文）的相同内容在潜表征 (latent representations) 上的相似性得到证实。作者还发现，在 Transformer 堆栈中间重复区块可以增强性能。模型可在 [Hugging Face](https://huggingface.co/dnhkng/RYS-Qwen3.5-27B-FP8-S) 上获取。作者建议，对这些模型（尤其是 **RYS-Qwen3.5-27B-FP8-XL**）进行微调，可能会为该尺寸的模型树立新的 SOTA。此外，目前正在开展优化 VRAM 使用的工作，即通过将重复层保留为副本，这可能对未来的实现大有裨益。** 评论者赞赏该研究严谨的方法和潜在影响，指出其与复杂模型合并 (merge) 中观察到的性能提升具有相关性。人们对这些发现如何影响开源微调实践（特别是在创意写作和自我合并技术方面）很感兴趣。

- ArsNeph 讨论了在 Goliath 120B 等自合并 (self-merges) 模型中观察到的引人注目的性能提升，并指出并非所有模型都能同样受益。他们引用了关于无 VRAM 重复层推理的历史讨论，强调了 EXL3 的持续开发工作。该评论认为，开源调优者，特别是那些关注 EQ 性能的调优者，可能会发现这些见解很有价值，尤其是在复杂合并树已显示出显著改进的创意写作场景中。
- Kwigg 反思了 llama2 时代“frankenmerging”的过往经验，质疑在拥有先进注意力机制的新模型中此类方法的效率。他们指出，旧的 frankenmerges 存在内存效率低下的问题，这意味着现代模型可能会以不同的方式处理这些技术，从而可能带来更好的性能结果。
- TomLucidor 建议将 Qwen3.5 的语言测试范围扩大到日语、泰语、法语、德语和意大利语。他们还提议在 Qwen3.5 与其他模型之间进行对比分析，如以速度和线性注意力著称的 Nemotron-3，以及提供类似尺寸多样性但优化程度较低的 Granite-4.0。这可以为这些模型的相对性能和优化情况提供见解。

- **[FlashAttention-4：1613 TFLOPs/s，比 Triton 快 2.7 倍，采用 Python 编写。这对推理意味着什么。](https://www.reddit.com/r/LocalLLaMA/comments/1s1yw23/flashattention4_1613_tflopss_27x_faster_than/)** (活跃度: 364): **FlashAttention-4** 在 **Blackwell B200 GPU** 上达到了 `1613 TFLOPs/s`，利用了其理论峰值性能的 `71%`。它比 **Triton** 快 `2.1-2.7 倍`，比 **cuDNN 9.13** 快多达 `1.3 倍`。该实现完全使用 **NVIDIA** 的 **CuTeDSL** 并在 **Python** 中完成，编译仅需 `2.5 秒`，而 C++ 则需要 `55 秒`。此版本支持 **GQA** 和 **MQA**，并已集成到 **vLLM 0.17.0** 中。然而，由于依赖 **TMEM**、**2-CTA MMA** 和 **async TMA** 等特定硬件特性，它仅限于 **Hopper + Blackwell** 架构，特别是 **H100/H800** 和 **B200/B100** GPU。文章还讨论了 softmax 如何成为瓶颈以及选择性重缩放 (selective rescaling) 如何优化性能。评论者对 NVIDIA 将某些 GPU 营销为“Blackwell”但缺乏与 FlashAttention-4 的完全兼容性表示沮丧，强调了广告宣传与实际硬件能力之间的差异。

    - **JockY** 对 NVIDIA 将 RTX 6000 Pro 营销为“Blackwell”表示不满，因为它并不完全兼容 Blackwell 的特性，特别提到 FlashAttention-4 (FA4) 和 NVFP4 仅在 SM100 架构上受支持。这突显了 NVIDIA 产品命名与实际硬件能力之间的脱节，可能会误导期望获得全面特性支持的早期采用者。
    - **Daemontatox** 指出，NVIDIA RTX 6000 Pro 被营销为“Blackwell”的问题更多地与流式多处理器 (Streaming Multiprocessor, SM) 架构有关，而非命名或整体架构。RTX 6000 Pro 和 DGX 系统以“Blackwell”之名销售，但实际上使用的是 SM120 架构，该架构缺乏一些预期的特性，导致了消费者的不满。
    - **STNKMyyy** 质疑像 FlashAttention-4 这样高性能的进步对消费级 GPU 的相关性，暗示虽然这些技术具有开创性，但普通消费者硬件用户可能无法获取或从中受益。这反映了前沿研究与实际消费级应用之间差距的普遍担忧。

- **[创建了一个 SillyTavern 扩展，让任何游戏中的 NPC 焕发生机](https://www.reddit.com/r/LocalLLaMA/comments/1s2ci9r/created_a_sillytavern_extension_that_brings_npcs/)** (活跃度: 499): **该帖子描述了一个新的 **SillyTavern** 扩展，它通过使用 **Cydonia** 作为角色扮演 (RP) 模型和 **Qwen 3.5 0.8B** 作为游戏主持人，将 NPC 集成到任何游戏中。这种设置通过下载游戏的 wiki 并将其输入 SillyTavern 来实现动态的 NPC 交互，使 NPC 能够拥有详细的背景设定并根据上下文做出回应。系统使用游戏文件中的语音克隆，并为 NPC 提供游戏状态信息，如玩家统计数据和位置。RP 模型在本地运行，确保了低延迟和强大的叙事能力。辅助模型 Qwen 3.5 负责解释 RP 交互以触发游戏内操作，在无需对话输入的情况下增强了旧游戏的真实感和深度。该帖子强调了在游戏应用中，专用 RP 模型优于基础模型的有效性。** 评论者对 AI 在游戏中的潜力表示惊讶和热情，注意到了 AI 在 NPC 交互方面的创新用途，并质疑为什么这种技术还没有成为游戏中的标配。

- 用户强调了使用 `0.8B` 参数模型在游戏中赋予 NPC 生命力的令人印象深刻的表现，并询问该项目是否开源。这表明一个轻量级模型能够实时在游戏环境中高效运行，这对于在没有沉重计算需求的情况下集成到现有游戏中具有重要意义。

  - **[我们在野外 Jeep 车上运行哪个本地模型？](https://www.reddit.com/r/LocalLLaMA/comments/1s1kyla/which_local_model_we_running_on_the_overland_jeep/)** (Activity: 459): **图像展示了一辆 Waymo 自动驾驶汽车，突显了自动驾驶汽车系统（autonomous vehicle systems）的技术进步。讨论的核心是预测未来的汽车将需要 `300GB of RAM`，这比目前的标准大幅增加。这一预测可能是基于这样一种假设：更复杂的模型，可能涉及实时数据处理和 AI 驱动的决策，将被集成到车辆中。评论反映了对这一预测的怀疑，用户质疑如此高内存要求的必要性，尤其是当前车辆在少得多的 RAM 下就能高效运行。** 评论者对未来汽车需要 `300GB of RAM` 的预测表示怀疑，质疑这一假设的依据，并将其与目前内存需求显著较低的车辆能力进行对比。

    - ForsookComparison 质疑汽车模型对高 RAM 要求的必要性，指出他们的汽车在 `600-mile` 的旅程中仅凭 `16GB of RAM` 就能高效运行。他们挑战了需要 `300GB` 的假设，认为这类数字可能是基于需要大量 tool-calls 的模型，而这并非适用于所有场景。
    - txdv 强调了车辆高 RAM 要求可能带来的成本影响，对 `128GB` 升级的可行性表示担忧。他们指出汽车定价是敏感的，`5k` 的 RAM 成本对消费者来说可能是禁止性的，表明需要在性能与负担能力之间取得平衡。


### 3. Chinese LLM Market and Model Evaluations

  - **[中国 LLM 领域的现状](https://www.reddit.com/r/LocalLLaMA/comments/1s1gm9z/the_current_state_of_the_chinese_llms_scene/)** (Activity: 639): **中国的 LLM 格局由 **ByteDance**、**Alibaba**、**Tencent** 和 **Baidu** 等主要参与者主导，每家都有专有模型和 open-weight 模型。**ByteDance** 以其类似于 OpenAI 的 `dola-seed` 模型领先，其 `Seedance T2V` 模型在视频生成方面很受欢迎。**Alibaba** 在 open-weight 模型方面表现出色，特别是小模型，并且在 T2I 和 T2V 领域实力雄厚。**Tencent** 的 `Hunyuan` 模型以 3D mesh 生成而闻名，尽管其最新版本尚未开源。**Baidu** 的 `Ernie` 模型使用较少，更专注于自动驾驶。其他值得注意的参与者包括 **Xiaomi** 的 `Mimo V2 Pro`、**Ant Group** 的 `Ling 2.5 1T` 以及 **Meituan** 的 `LongCat-Flash-Chat`（使用动态 MoE 方法）。**Deepseek** 因其在 MLA 和 DSA 等注意力机制方面的创新而备受关注。“AI 六小虎”（如 **Zhipu** 和 **Minimax**）专注于发布大型 open-weight 模型以获得认可。政府资助的机构如 **BAAI** 和 **Shanghai AI Lab** 也有贡献，尽管声誉参差不齐。** 评论者指出，与美国相比，中国 open-weight 模型的发布速度极快，一些实验室在一个季度内发布的模型比美国公司在两年内发布的还要多。**Tencent** 因其在游戏开发专用模型上的投入而受到认可，其 `Hunyuan 3.1` 在 3D mesh 生成方面处于 state-of-the-art 水平。

    - Tencent 正在大力投资游戏开发专用模型，例如用于 3D mesh 生成的 Hunyuan 3.1 和用于 text-to-animation 的 HY-Motion，这些模型被认为是 state-of-the-art。最初，Tencent 开源这些模型以建立品牌知名度，但一旦达到商业化可行性，就会转向 closed weights，正如最新的 Hunyuan 3D 模型所示。
    - OpenRouter 过去 7 天 token 使用量热门模型列表突显了中国模型的统治地位，Xiaomi MiMo-V2-Pro 以 1.77 万亿 tokens 领先。值得注意的是，只有三家西方实验室上榜，而“小虎”（Small Tigers）——快速推进 AI 的小型公司——表现突出，这表明创新动力发生了转变。
    - 尽管 ByteDance 对 AI 贡献巨大，但他们尚未发布任何 open weight 模型，Hugging Face 上没有此类模型证实了这一点。这与其他经常发布 open weight 的中国实验室形成鲜明对比，后者加速了 AI 领域的竞争。

- **[So cursor admits that Kimi K2.5 is the best open source model](https://www.reddit.com/r/LocalLLaMA/comments/1s19ik2/so_cursor_admits_that_kimi_k25_is_the_best_open/)** (Activity: 629): **该图片是来自 Aman Sanger 的一条推文，讨论了对 Base Model 的评估，特别强调了 Kimi K2.5 在基于 Perplexity 的评估中脱颖而出，成为最强的模型。推文指出，该模型的强势归功于持续的 Pre-training 和高算力的 Reinforcement Learning，这些增强了 Composer-2 模型的能力。推文还承认在其博客中漏掉了 Kimi Base，并计划在未来的沟通中予以纠正。** 一条评论批评了在不同模型之间使用基于 Perplexity 的评估，指出分数可能会受到字典大小等因素的影响。另一条评论对 Kimi K2 的训练比例声明表示怀疑，引用了 Workshop Labs 的报告，该报告暗示 Fireworks 的 K2 训练代码未针对超大规模训练进行优化，这与其实效性的说法形成鲜明对比。

    - 关于 Kimi K2.5 是最佳开源模型的说法因评估方法而受到质疑，特别是使用 Perplexity 分数可能会产生误导，因为它们取决于字典大小等因素。这引发了对模型之间此类比较有效性的担忧。
    - 针对 Fireworks 关于 Kimi K2.5 的训练声明存在怀疑。以优化训练代码著称的 Workshop Labs 报告称，Fireworks 的代码未针对超大规模训练进行优化，仅略好于 HF Transformers 4.x 等基础实现。这表明 Fireworks 在训练 Kimi K2.5 的方法上可能存在效率低下的问题。
    - Kimi K2.5 被认为是最佳“Base Model”的主张归功于其庞大的参数量以及使用了标准的 Attention 机制而非线性机制。这暗示该模型的架构和规模对其性能贡献巨大，而非由于任何新颖的训练技术。

  - **[China's open-source dominance threatens US AI lead, US advisory body warns](https://www.reddit.com/r/LocalLLaMA/comments/1s1kmch/chinas_opensource_dominance_threatens_us_ai_lead/)** (Activity: 922): **一个美国顾问机构对中国在开源 AI 领域日益增长的影响力表示担忧，认为这可能会威胁到美国在 AI 领域的领导地位。报告强调了中国在开源 AI 模型方面的战略投资和进步，这些模型正变得越来越具有竞争力。该顾问机构建议美国需要加强其开源计划以保持竞争优势。** 评论者认为美国在开源 AI 方面已经落后，中国模型更具成本效益且效率更高。此外，还有人批评 Opus、GPT-5.4 和 Gemini 3.1 Pro 等美国模型存在功能失调，这与中国尽管在威权体制下仍对 AI 自由做出贡献形成了对比。

    - **EffectiveCeilingFan** 强调了中国 AI 模型的竞争优势，指出它们不仅更便宜，而且在 Open Weights 方面优于美国模型。该评论者批评了 Opus、GPT-5.4 和 Gemini 3.1 Pro 等美国模型的表现，暗示美国在开源 AI 发展方面正处于落后地位。
    - **Lissanro** 强调了开放研究在 AI 发展中的重要性，引用了《Attention is All You Need》论文作为奠基石。他们提到 Kimi K2.5 等模型的存在归功于 DeepSeek 等公司分享的开放研究。评论还指出，Cursor AI 等大公司正为其产品采用 Kimi K2.5 等中国模型，表明行业对这些开源模型的青睐。
    - **Global_Estimate7021** 详细分析了美国在 AI 领域可能落后的原因，引用了显著的 AI 接受度差距（中国为 87%，美国为 32%）以及中国领先的 AI 研究出版物数量。他们还提到了中国廉价电力的战略优势和基层 AI 素养普及计划，这与美国自上而下的方式形成鲜明对比。

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. AGI 成就与声明

  - **[最初创造“AGI”缩写的人现在表示，我们已经完全按照他的设想实现了它。](https://www.reddit.com/r/singularity/comments/1s2cfrb/the_man_who_originally_coined_the_acronym_agi_now/)** (Activity: 926): **这张图片是 **Mark Gubrud** 的一条推文，他声称自己创造了“AGI”（Artificial General Intelligence）一词。他断言 AGI 已经按照他的设想实现，目前的模型在语言和通用知识方面表现出了人类高水平的性能，同时速度更快。然而，关于他这一说法原创性的争论仍然存在，因为“artificial general intelligence”一词早在 1989 年就有记录，归功于 **G. Simons**。Gubrud 对 AGI 的定义涉及与人类大脑复杂性和速度相匹配或超越的系统，能够在各种操作中利用通用知识进行推理。** 评论中对 Gubrud 声称创造了“AGI”一词表示怀疑，一些人认为他记错了历史。《牛津英语词典》将该术语的最早使用追溯到 1989 年 G. Simons 的著作，而非 Gubrud。

    - “artificial general intelligence”（AGI）一词早在 1989 年就有记载，《牛津英语词典》引用 G. Simons 为最早来源。然而，M. Gubrud 通常被认为在科学文献中普及了该词，尽管他本人并未创造该词。
    - AGI 创造者对其最初的定义是：在复杂性和速度上匹配或超越人类大脑能力的系统，能够处理包括工业和军事行动在内的各个领域的通用知识。这个定义暗示了一种广泛且多功能的智能，尽管人们对于目前的系统是否符合这一标准持怀疑态度。
    - 关于在没有递归自我改进（recursive self-improvement）的情况下实现 AGI 的意义存在争议，而这种改进曾被预期会触发技术奇点（technological singularity）。由于缺乏此类变革性进展，导致人们对当前围绕 AGI 发展的兴奋情绪持怀疑态度。

  - **[Jensen Huang (NVIDIA) 声称 AGI 已经实现](https://www.reddit.com/r/singularity/comments/1s1mix1/jensen_huang_nvidia_claims_agi_has_been_achieved/)** (Activity: 2562): **在最近的一次采访中，**NVIDIA** 的 CEO **Jensen Huang** 声称通用人工智能（AGI）已经实现，这一言论引发了重大争议。该采访可在 [YouTube](https://youtu.be/vif8NQcjVf0?si=WhXfzQ3-Dk5ZvEpo) 上观看，但缺乏支持这一说法详细的技术证据，导致专家们持怀疑态度。Huang 的断言被认为可能受到其推广 **NVIDIA** 产品角色的影响，而这些产品在 AI 技术上投入巨大。** 热门评论反映了对 Huang 言论的怀疑，突显了对商业领袖关于自家产品言论的不信任。评论者认为，此类声明更多是为了营销，而非 AI 领域的实际事实性进展。

    - Sweaty_Rub4322 强调了 AGI 争论中的一个关键问题：缺乏一个普遍接受的 AGI 定义。这种模糊性使讨论和评估 AGI 是否已经实现变得复杂，因为学术界和工业界在什么构成 AGI 上难以达成共识。这强调了需要一个清晰、标准化的定义，以促进该领域有意义的进展和评估。

### 2. Claude Code 功能与更新

  - **[Claude 现在可以使用你的电脑了](https://www.reddit.com/r/ClaudeAI/comments/1s1ujv6/claude_can_now_use_your_computer/)** (活跃度: 2106): **Claude**，这款由 **Anthropic** 开发的 AI，现在能够通过 **Claude Cowork** 和 **Claude Code** 使用你的电脑执行任务。该功能目前处于研究预览阶段，允许 Claude 打开应用程序、导航浏览器以及管理电子表格，从而有效地实现通常需要手动完成的任务自动化。它优先使用已连接的应用（如 Slack 和 Calendar），但也可以在获得许可的情况下直接与屏幕上的应用进行交互。此功能面向 macOS 用户的 Pro 和 Max 层级提供，需要更新桌面应用并与移动设备配对。更多详情请参阅[此处](https://claude.com/product/cowork#dispatch-and-computer-use)。一些用户对允许 AI 控制电脑的安全影响表示担忧，部分用户则表达了对潜在职位取代的忧虑。另一些人指出，这是 **Anthropic** 应对 **OpenAI** 等竞争对手的战略举措。

    - 提出的一个关键担忧是允许 Claude 访问用户电脑的安全影响。这涉及潜在的风险，如未经授权的数据访问或操纵，如果不妥善保护，这些风险可能会被利用。功能发布的快速节奏可能会加剧这些担忧，因为新功能在部署前可能未经过彻底的漏洞审查。
    - 引入 Claude 使用电脑的能力被视为对 OpenAI 进步的竞争性回应，特别是在 GPT-4 等 AI 模型的背景下。Anthropic 的这一举动可能旨在 AI 能力竞赛中保持对等或获取优势，凸显了 AI 行业的竞争动态。
    - 有一种观点认为，Claude 快速开发和发布新功能可能会导致职位取代。随着 AI 模型变得越来越能够执行传统上由人类完成的复杂任务，人们对就业影响的担忧日益增加，尤其是在严重依赖常规认知任务的行业。

  - **[Claude Code 现在可以执行 /dream 了](https://www.reddit.com/r/ClaudeCode/comments/1s2ci4f/claude_code_can_now_dream/)** (活跃度: 1953): **Claude Code 推出了一项名为 **Auto Dream** 的功能，旨在通过模拟人类的 REM 睡眠过程来增强 Agent 的内存管理。该功能会回顾过去的会话记录，识别相关信息，剪除过时或矛盾的数据，并将其整合到有序的文件中。它在后台运行，自上次整合以来经过 24 小时且进行过五次会话后触发，并通过使用锁定文件确保没有冲突。这种方法旨在通过更智能地管理内存来提高性能，而不仅仅是扩展上下文窗口。** 一些评论者对该功能表示怀疑，认为它可能会导致不必要的 Token 消耗，并质疑其 AI 自我推广的风格。其他人则幽默地建议增加额外的命令来管理 AI 幻觉和错误。

    - AutoDream 是 Claude Code 的一项功能，充当其内存系统的“睡眠周期”，解决由 Auto Memory 功能引入的内存膨胀问题。在 v2.1.59 中发布的 Auto Memory 允许 Claude 记录项目笔记，但随着时间的推移，这些笔记会积累噪音和矛盾，从而降低性能。AutoDream 通过定期整合记忆来缓解这一问题，类似于人类的 REM 睡眠，过程分为四个阶段：定向 (Orient)、收集信号 (Gather signal)、整合 (Consolidate) 以及剪裁与索引 (Prune & index)。
    - AutoDream 过程包括四个阶段：**定向 (Orient)**，扫描现有内存以理解存储的数据；**收集信号 (Gather signal)**，识别过时的记忆并进行定向搜索；**整合 (Consolidate)**，合并新信息并解决矛盾；以及 **剪裁与索引 (Prune & index)**，维护简明索引并删除陈旧数据。该过程仅在上次整合 24 小时以上且经过 5 次以上会话后才会触发，确保不干扰活动的工作。
    - AutoDream 对项目代码以只读方式运行，仅修改内存文件而不修改实际代码库。这在高效管理内存的同时，确保了代码的安全性和完整性。该功能的完整系统提示词 (system prompt) 已在 GitHub 上的 `agent-prompt-dream-memory-consolidation.md` 下发布，提供了透明度并允许用户了解其运作方式。

### 3. Sora 关停公告

  - **[Sora 正式宣布关停。](https://www.reddit.com/r/OpenAI/comments/1s2oyl3/sora_is_officially_shutting_down/)** (活跃度: 854): **该图片是 Sora 应用在 X.com 官方账号发布的公告截图，声明 Sora 即将关停。该消息感谢了用户的参与，并承诺将提供有关应用和 API 关停时间表的更多细节。这表明该应用的生命周期发生了重大变化，很可能是由于战略调整或财务上的不可持续性，正如评论中所指出的高成本和低参与度。** 评论认为 Sora 的关停归因于其不可持续的商业模式，特别是在版权处理方式改变后，导致成本增加并降低了用户参与度。该应用最初具有创新性，但后来变成了负担。

    - Chasemania 强调了 Sora 的不可持续性，指出该产品面临高昂的运营成本和较低的用户参与度。过度尊重版权法的尝试导致了用户兴趣的下降，使该平台从资产变成了负担。
    - 讨论涉及了在版权合规与用户参与度之间取得平衡的挑战。Sora 最初的吸引力被其在遵守严格版权法规的同时无法维持用户兴趣的困境所掩盖，这最终导致了它的失败。
    - 评论反映了 Sora 最初的成功及其随后的衰落，强调了维持一个需要高运营成本且必须严格遵守版权法（这可能会阻碍用户参与并导致财务不稳定）的平台的难度。

  - **[Sora 正式宣布关停。](https://www.reddit.com/r/ChatGPT/comments/1s2oxnu/sora_is_officially_shutting_down/)** (活跃度: 1429): **该图片是 Sora 团队关于关停 Sora 应用的社交媒体公告。帖文对社区表达了感谢，并承诺很快会提供关于应用和 API 时间表以及用户如何保存作品的更多细节。这表明这是一个有计划且结构化的关停过程，旨在尽量减少对用户的干扰。** 评论反映了对该应用影响力及其用户群的怀疑，一些用户对该应用在缺乏财务可行性的情况下能维持这么久表示惊讶。





# AI Discords

遗憾的是，Discord 今天关闭了我们的访问权限。我们不会以这种形式恢复它，但我们很快就会发布全新的 AINews。感谢读到这里，这是一段美好的历程。