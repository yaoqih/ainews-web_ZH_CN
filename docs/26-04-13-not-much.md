---
companies:
- openai
- github
- cursor
- langchain
- nous-research
date: '2026-04-13T05:44:39.731046Z'
description: '**支撑工程 (Harness engineering)** 正成为 AI 智能体（Agent）开发中的一个关键领域，它强调除模型之外的组件，如文件系统、内存和重试机制。**OpenAI
  的 Codex** 正在将智能体化的编程工作流扩展到软件工程之外，包括代码库理解和 Bug 分类。工具链的发展趋势正趋向于多智能体编排、可观测性和远程控制，**GitHub
  Copilot**、**Cursor** 和 **LangChain** 都在推动这些能力的进步。**Hermes Agent v0.9.0** 的发布引入了本地
  Web 仪表板和增强的安全性，在用户体验（UX）和效率方面表现出色，比 **OpenClaw** 获得了更多的社区关注。开放智能体生态系统正在不断壮大，**Open
  Agents** 和 **DeepAgent** 等项目提供了模块化的技术栈和运行时。'
id: MjAyNS0x
models:
- codex
people:
- andrew_ng
- steve_yegge
- gabrielchua
- giffmana
- rhys_sullivan
- teknium
- shaun_furman
- dabit3
- robinebers
- zainanzhou
- nicoalbanese10
- bromann
- elliothyun
- tiagonbotelho
- pierceboggan
- sydneyrunkle
title: 今天没发生什么特别的事。
topics:
- agent-harnesses
- multi-agent-systems
- software-engineering
- tooling
- orchestration
- observability
- remote-control
- security-hardening
- user-experience
- open-source
- community-engagement
---

**平静的一天。**

> 2026年4月11日至4月13日的 AI 新闻。我们查看了 12 个 subreddit、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，且没有进一步查看 Discord。[AINews 网站](https://news.smol.ai/) 允许您搜索所有历史期数。提醒一下，[AINews 现在是 Latent Space 的一个板块](https://www.latent.space/p/2026)。您可以[选择订阅/退订](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)邮件接收频率！

---

# AI Twitter 回顾

**Agent Harness、编程工作流，以及从单一模型到系统设计的转变**

- **Harness 工程现在已成为一等学科（first-class discipline）**：贯穿 [AI Engineer Europe 的总结](https://x.com/dat_attacked/status/2043647001749836253)、[Vtrivedy 对 harness 原语的构想](https://x.com/Vtrivedy10/status/2043870915059236966)以及多个 Agent 构建者帖子的共同主题是，实用的 Agent 不仅仅是“模型”。文件系统、bash、压缩（compaction）、内存、权限、重试、evals 和 subagent 正日益被视为核心产品表面（product surface）。[Andrew Ng](https://x.com/AndrewYNg/status/2043742105852621052) 也表达了类似观点，他认为瓶颈正在从实现转向决定构建什么；[Steve Yegge](https://x.com/Steve_Yegge/status/2043747998740689171) 则声称，尽管工具获取门槛降低，但企业端的采用仍远落后于前沿实践。

- **OpenAI 的 Codex 使用模式表明 Agentic 编程正扩展到 SWE 之外**：OpenAI 通过 [@gabrielchua](https://x.com/gabrielchua/status/2043339151278506234) 分享了一个实用的 Codex 工作流目录——包括理解大型代码库、PR 审查、Figma 转代码、Bug 分类、数据集分析、CLI 工具、入职引导，甚至是幻灯片生成。在实践中，用户报告了同样的“Agent 作为胶水”模式：例如 [giffmana](https://x.com/giffmana/status/2043401612035559445) 使用 Codex 为 Linux 上的一个小众 Wayland/HIDPI 问题修补 Java/Qt 二进制文件；而其他人则对当前模型在可靠生产工作中能否优于人类直接实现持怀疑态度，如 [Rhys Sullivan 的评论](https://x.com/RhysSullivan/status/2043584591861321929)。

- **工具链正向多 Agent 编排、可观测性和远程控制收敛**：GitHub 发布了 [来自 Web/移动端的 Copilot 远程控制](https://x.com/pierceboggan/status/2043717775265562701)，[@tiagonbotelho](https://x.com/tiagonbotelho/status/2043720370734104923) 随后进行了跟进。Cursor 增加了 [拆分 Agent（split agents）以及搜索/性能改进](https://x.com/cursor_ai/status/2043798784367546707)。LangChain 强调了 [通过中间件和文件系统权限建立护栏（guardrails）](https://x.com/sydneyrunkle/status/2043767032361967751)，而 deepagents 的心理模型则如 [@ElliotHyun](https://x.com/ElliotHyun/status/2043721149616369719) 所述，将 subagent 简化为结构化工具/函数调用。共同的模式是：Agent 产品正通过暴露控制平面走向成熟，而不是仅仅宣称实现了全自主的可靠性。

**Hermes Agent Dashboard 发布、OpenClaw 竞争以及开放 Agent 技术栈**

- **Hermes 巩固了其作为当前讨论最广泛的开源 harness 的势头**：核心发布是 Hermes Agent v0.9.0，带有本地 Web Dashboard、快速模式、备份/导入、更强的安全加固以及更广泛的渠道支持；参见 [@Teknium](https://x.com/Teknium/status/2043771509123232230) 和 [@NousResearch 的官方公告](https://x.com/NousResearch/status/2043791876835156362)。社区反应认为该 Dashboard 是能让 Hermes 破圈走向普通用户的关键功能，[Shaun Furman 甚至称其为“OpenClaw 时刻”](https://x.com/Shaun__Furman/status/2043820083114545416)。

- **OpenClaw 仍在迭代，但在 UX 和效率方面的对比讨论正向 Hermes 倾斜**：OpenClaw 通过 [@TheTuringPost](https://x.com/TheTuringPost/status/2043340386538778840) 发布了重大更新——内存导入、“记忆宫殿（Memory Palace）”、更丰富的聊天 UI、插件设置指南、更好的视频生成以及更多集成。但多位用户明确表示在速度、架构或 Token 效率方面更倾向于 Hermes，包括 [dabit3](https://x.com/dabit3/status/2043808914312212568)、[robinebers](https://x.com/robinebers/status/2043835216670929005) 以及 [ZainanZhou 的 harness 级解释](https://x.com/ZainanZhou/status/2043760979931213851)，后者指出更好的预选/上下文整形（preselection/context shaping）可能会减少 Token 消耗。

- **围绕 Agent 栈的开放生态系统正在变得日益丰厚**：[Open Agents](https://x.com/nicoalbanese10/status/2043745569278251112) 作为云端代码 Agent 栈正式开源；[bromann](https://x.com/bromann/status/2043886229650067729) 将其与 DeepAgent 进行了对比，后者是一个具有可插拔模型提供商、沙箱、中间件和追踪功能的底层运行时。Hermes 自身也在积累社区技能、教程、多 Agent 配方以及集成——从[中文教程汇总](https://x.com/biteye_sister/status/2043630704798679545)到来自 [@coreyganim](https://x.com/coreyganim/status/2043627229205193211) 的实用“4 人 Agent 团队”指南。值得注意的技术范式是持久化的角色分离加上隔离的记忆，而非幼稚的“单个 Agent 包揽一切”。

**网络安全、模型能力升级与 Mythos 冲击波**

- **Claude Mythos Preview 主导了网络安全话题的讨论**：英国 AI 安全研究所（AISI）报告称，Mythos 是[第一个端到端完成 AISI 网络靶场测试的模型](https://x.com/AISecurityInst/status/2043683577594794183)；[ekinomicss](https://x.com/ekinomicss/status/2043688793085992970) 的后续评论指出，该模型在 32 步的企业网络攻击模拟中取得了成功。其他反馈强调了其能力与效率，例如 [scaling01 声称](https://x.com/scaling01/status/2043700788245963167) Mythos 在长时运行后，仅需约 **40%** 的 Token 即可达到 Opus 级的性能。

- **安全层面的意义不仅在于基准测试的进展，更在于实际操作的可用性**：[emollick](https://x.com/emollick/status/2043810051979157680) 认为这种担忧是有道理的；[ananayarora](https://x.com/ananayarora/status/2043381424594837789) 指出 Marcus Hutchins 的反应尤其具有意义。目前正在形成的一个共识是，“漏洞研究模型”不再是投机性的营销术语；实验室和外部评估机构现在正在描述在独立靶场上完成的端到端漏洞利用（exploit）工作流。

- **防御工具正在同步成熟，但不对称性显而易见**：[《The Turing Post》的汇总](https://x.com/TheTuringPost/status/2043332388785426498)重点介绍了 10 个开源 AI 安全项目，包括 NVIDIA NeMo Guardrails、garak、Promptfoo、LLM Guard、ShieldGemma 2 和 CyberSecEval 3。与此同时，开发者们正在重新审视“Agent 可以安全替代成熟依赖项”的假设：[dbreunig](https://x.com/dbreunig/status/2043762702653460520) 认为，一旦将加固和安全审查的成本计算在内，Token 的经济计算逻辑就会发生变化，使得维护良好的 OSS 库再次具有相对吸引力。

**推理、检索、OCR 与系统性能**

- **文档/OCR 评估迎来了重要的新基准测试**：LlamaIndex 发布了 [ParseBench](https://x.com/jerryjliu0/status/2043721536922955918)，这是一个针对文档解析的开源基准测试/数据集，专注于与 Agent 相关的语义正确性，而非精确匹配的文本相似度。它包含约 **2,000** 页经过人工验证的企业级页面，以及涵盖表格、图表、内容忠实度、语义格式化和视觉定位（visual grounding）的 **167,000+** 条评估规则。一个显著的结果是：没有哪款解析器能在所有维度上占据主导地位，但据报道 LlamaParse 在整体上以 **84.9%** 的得分领先。

- **Hugging Face 展示了使用开源模型进行工业级规模的 OCR 既便宜又可靠**：[@ClementDelangue](https://x.com/ClementDelangue/status/2043779449322160270) 报告称，通过在 L40S 上运行 **16** 个并行的 HF Jobs，耗时约 **29 小时**、花费约 **$850**，将 **27,000 篇 arXiv 论文** OCR 转换为了 Markdown 格式，目前已用于支持“与你的论文对话（Chat with your paper）”功能。随后确认该模型为 [Chandra-OCR-2](https://x.com/ClementDelangue/status/2043783879601848726)。

- **检索和传输层优化持续发挥重要作用**：LightOn 发布了 [ColGrep 1.2.0](https://x.com/raphaelsrty/status/2043676936442875954)，支持用于混合多向量检索的 BM25 trigrams，并引入相对路径以节省 Token，将其定位为 Agent 搜索的简易升级方案。在系统方面，[Lewis Tunstall 及其同事](https://x.com/_lewtun/status/2043690765227102335)指出了一项不明显的同策略（on-policy）蒸馏瓶颈：vLLM 通过网络传输 JSON 格式的 logprobs。切换到二进制 NumPy 数组后获得了 **1.4 倍** 的速度提升，这是一个有益的提醒：基础设施层面的收益往往存在于内核（kernel）和模型代码之外。

- **压缩和推测解码（Speculative decoding）仍然是高效的部署手段**：Red Hat AI 展示了[在 vLLM 上部署的 Gemma 4 31B 量化版本](https://x.com/RedHat_AI/status/2043709783102906489)，其 **tokens/sec** 提升了近 **2 倍**，内存占用减少了一半，且保留了 **99% 以上** 的精度。在推测解码方面，相关博文涵盖了[用于 Kimi/Qwen 系列本地加速的 DFlash 适配器](https://x.com/winglian/status/2043731370598347066)、Baseten 的 [EAGLE-3 生产建议](https://x.com/baseten/status/2043762663235432855)，以及 [DDTree](https://x.com/liranringel/status/2043813397972607477) 等新研究，该研究通过一次 block-diffusion 传递来草拟树结构，从而共同验证多个后续生成。

**研究方向：内存（Memory）、验证、RL 和模型架构**

- **长文本内存研究正在超越传统的 KV cache 扩展**：[behrouz_ali](https://x.com/behrouz_ali/status/2043743704335192095) 概述了“Memory Caching”，这是一系列将上下文压缩为缓慢增长的循环内存的架构，旨在实现接近 Attention 的有效内存增长，但推理成本更接近 RNN。Sparse Selective Caching 被认为是最具实用性的变体。[askalphaxiv](https://x.com/askalphaxiv/status/2043782770657219010) 的相关评论将其描述为标准循环（recurrence）与全二次型 Attention 之间的插值。

- **验证器风格的推理时方法正在成为一种严肃的 Agent 基准策略**：[Azali Amirhossein 等人](https://x.com/Azaliamirh/status/2043813128690192893) 引入了 **LLM-as-a-Verifier**，通过要求模型对输出进行排序，并使用排序 Token 的 logprobs 来估算预期质量，从而为候选对评分。其核心观点是，在 Agent 基准测试中，胜出者选择而非候选生成往往是推理时扩展（test-time scaling）的瓶颈；单次验证传递的表现可以优于更繁琐的重排序（reranking）设置。

- **推理发现（Reasoning discovery）仍是弱点，有人认为这对监管来说是好消息**：[Laura Ruis](https://x.com/LauraRuis/status/2043715536186384775) 报告称，即使某些潜在规划策略在教授后变得非常简单，LLM 仍难以*发现*这些策略，即使扩展到 GPT-5.4 也仅有小幅提升。另外，[Wen Sun](https://x.com/WenSun1/status/2043755261954011484) 认为，基于 RL 的 Prompt 优化只需 **2** 个示例即可实现泛化，而零阶方法在此时会发生过拟合。综合结论：在“推理”实现稳健的自我引导（self-bootstrapping）之前，训练目标和推理时支架（test-time scaffolding）仍有巨大的提升空间。

**热门推文（按参与度排序）**

- **Codex 使用案例在 OpenAI**：[@gabrielchua](https://x.com/gabrielchua/status/2043339151278506234) 分享了一份广泛且实用的内部 Codex 工作流清单，涵盖代码理解、应用构建、运维自动化以及非工程任务。
- **AISI 对 Claude Mythos Preview 的网络安全评估**：[@AISecurityInst](https://x.com/AISecurityInst/status/2043683577594794183) 报告了模型首次端到端完成其网络安全靶场（cyber range）的任务，使其成为该系列中技术影响力最大的博文之一。
- **Hermes Agent 控制面板发布**：[@NousResearch](https://x.com/NousResearch/status/2043791876835156362) 宣布了本地控制面板及相关的 v0.9.0 功能，引发了一波用户将其与 OpenClaw 和 Claude Code 进行对比的热潮。
- **OpenAI 的“算力驱动经济”备忘录**：[@gdb](https://x.com/gdb/status/2043831031468568734) 概述了 OpenAI 的论点，即软件工程是向算力介导工作和意图驱动工具（intent-driven tooling）更广泛转型的前沿领域。
- **Hugging Face 的大规模开源 OCR 部署**：[@ClementDelangue](https://x.com/ClementDelangue/status/2043779449322160270) 展示了如何使用开源模型和 HF Jobs 以低成本、容错的方式将 2.7 万篇论文 OCR 转换为 Markdown。

---

# AI Reddit 总结

## /r/LocalLlama + /r/localLLM 总结

### 1. Gemma 4 模型进展与基准测试

  - **[最佳本地 LLM - 2026 年 4 月](https://www.reddit.com/r/LocalLLaMA/comments/1sknx6n/best_local_llms_apr_2026/)** (热度: 440): **该帖子讨论了截至 2026 年 4 月本地大语言模型 (LLM) 的最新进展，重点介绍了 **Qwen3.5**、**Gemma4** 和 **GLM-5.1** 的发布，后者声称具有 SOTA 性能。**Minimax-M2.7** 模型因其易用性受到关注，而 **PrismML Bonsai** 引入了高效的 1-bit 模型。该线程鼓励用户分享这些模型的体验，重点关注开源权重模型，并详细说明其配置、用法和工具。帖子还根据 VRAM 需求对模型进行了分类，范围从“无限制”（&gt;128GB）到“S”（&lt;8GB）。** 有评论建议进一步细分需要 128GB 以上 VRAM 的模型类别，表明在对高资源模型分类时需要更细的粒度。

- 一位用户建议为显存超过 128 GB 的模型细分类别，强调需要更精细的分级，而不应仅仅依赖于 'S' 或 'M' 等标签。这暗示了对大显存模型详细性能指标和 Benchmark 的需求，这对于需要大规模数据处理或复杂计算的应用至关重要。
- 讨论的焦点还包括针对医疗、法律、会计和数学等特定领域定制的专业本地 LLM。这突显了领域特定优化的重要性，以及这些模型通过利用专门的训练数据和架构，在利基领域超越通用 LLM 的潜力。
- 文中提到了 Agentic Coding 和工具使用，这表明人们对能够自主与工具或 API 交互以执行任务的模型感兴趣。这指向了一个趋势，即开发具有动态任务执行和外部系统集成能力的 LLM，从而增强其在实际应用中的效用。

- **[音频处理功能随 Gemma-4 落地 llama-server](https://www.reddit.com/r/LocalLLaMA/comments/1sjhxrw/audio_processing_landed_in_llamaserver_with_gemma4/)** (热度: 494): **llama.cpp** (llama-server) 已集成音频处理功能，特别是支持 **Gemma-4 E2A 和 E4A 模型** 的语音转文本 (STT)。此更新实现了原生音频支持，无需再使用像 Whisper 这样独立的 Pipeline。然而，用户报告了长音频转录的问题，例如 `llama-context.cpp` 中的错误和语句循环。推荐的设置是使用 `E4B as Q8_XL quant with BF16 mmproj`，因为其他配置会降低性能。为了获得最佳转录效果，应遵循特定的模板，强调精确的格式化和数字表示。一些用户对其性能与 Whisper 相比表示怀疑，而另一些人则指出，尽管进行了集成，系统在处理较长音频段时仍显吃力，这表明 **Voxtral** 在这些情况下表现更好。

    - Chromix_ 强调了 llama-server 当前音频处理实现中的几个技术问题，特别是在处理超过 5 分钟的音频时。他们指出，建议使用 E4B 作为 Q8_XL 量化并搭配 BF16 mmproj，因为其他格式会降低性能。然而，他们遇到了类似 `llama-context.cpp:1601` 的错误以及转录质量问题，包括语句循环和提前终止。他们建议使用特定的转录和翻译模板来改善结果。
    - GroundbreakingMall54 指出了 llama.cpp 中原生音频支持的重要性，这消除了对独立 Whisper Pipeline 的需求。对于以前必须管理多个音频处理系统的用户来说，这种集成被视为一项重大改进。
    - ML-Future 分享了他们在西班牙语中测试音频处理功能的经验，指出虽然不完美，但相当准确，且表现优于 Whisper。这表明新功能在某些语言中可能比现有解决方案提供更好的转录质量。

- **[Speculative Decoding 在 Gemma 4 31B 配合 E2B 草稿模型下表现出色（平均提升 29%，代码提升 50%）](https://www.reddit.com/r/LocalLLaMA/comments/1sjct6a/speculative_decoding_works_great_for_gemma_4_31b/)** (热度: 527): **该帖子讨论了 **Gemma 4 31B** 模型使用 **Gemma 4 E2B (4.65B)** 作为草稿模型（draft model）实现投机采样（Speculative Decoding）的情况，并取得了显著的性能提升。测试环境包括 **RTX 5090 GPU** 和一个带有 TurboQuant KV cache 的 **llama.cpp fork**，配置了 `128K context` 和特定的草稿参数（`--draft-max 8 --draft-min 1`）。Benchmark 显示平均加速 `+29%`，在代码生成任务上提升达 `+50%`，这归功于模型间词表（vocabularies）的兼容性，避免了 Token 转换开销。早期 GGUF 版本中发现了一个关键问题，即 `add_bos_token` 元数据不匹配，该问题已通过重新下载更新后的模型解决。帖子还强调了设置 `--parallel 1` 以防止 VRAM 过度占用的重要性，并提出了优化性能的实用建议，例如使用 Q4 草稿模型和管理 VRAM 分配。评论者建议尝试不同的 `--draft-max` 和 `--draft-min` 值，并询问了完整的 llama-server 命令以及所使用的特定 fork。另一个建议是将每层 Embedding 卸载到 CPU，以在不影响推理速度的情况下优化 VRAM 使用。**

- Odd-Ordinary-5922 询问了调整 `--draft-max` 和 `--draft-min` 参数的影响，这些参数可能与控制 Speculative Decoding 过程有关。这些参数可能会影响速度与准确性之间的平衡，但评论中未详细说明具体效果。
- albuz 建议通过使用 `--override-tensor-draft "per_layer_token_embd\.weight=CPU"` 命令将 Draft Model 的逐层 Embedding 卸载（Offloading）到 CPU，以此优化 VRAM 使用。该技术旨在不影响推理速度的前提下节省 GPU 显存，对于 VRAM 资源有限的用户非常有益。
- EdenistTech 报告称在 5070Ti/5060Ti 组合上使用 Draft Model 时性能显著提升，在 128K Context Size 下，吞吐量从每秒约 25 Tokens 增加到 40 Tokens。这表明 Speculative Decoding 可以大幅提升处理速度，特别是在特定硬件配置的设置中。

### 2. Minimax M2.7 与许可更新

  - **[MiniMax 的 Ryan Lee 发布了关于许可协议的文章，指出该协议主要针对那些在提供 M2.1/M2.5 服务方面表现不佳的 API 提供商，并可能为普通用户更新许可！](https://www.reddit.com/r/LocalLLaMA/comments/1skabyf/ryan_lee_from_minimax_posts_article_on_the/)** (活跃度: 451)：**图片是来自 **MiniMax** 的 Ryan Lee 发的一条推文，讨论了其 M2.7 模型的许可条款。他澄清说，允许并免费支持自托管 M2.7 用于编写代码，但承认目前的许可协议缺乏细节，将会进行更新。这一点意义重大，因为它解决了关于许可清晰度和适用性的担忧，特别是针对那些因服务质量差而受到批评的 API 提供商。讨论强调了限制商业用途的许可问题，这可能会使自托管变得复杂，并可能在模型能力方面误导用户。** 评论者对许可条款的清晰度和意图表示怀疑，指出一些提供商歪曲了他们所提供模型的质量。还有人担心旨在防止营利性托管的复杂许可可能会无意中影响合法的自托管努力。

    - Few_Painter_5588 强调了 OpenRouter 的一个重大问题，指出许多 API 提供商歪曲了他们所提供模型的质量，有些甚至根本没有提供他们声称的模型。这反映了生态系统中的一个更广泛的问题，即模型服务的可靠性和透明度对于用户信任和有效部署至关重要。
    - silenceimpaired 讨论了旨在防止营利性托管的许可复杂性，这可能会无意中使自托管变得复杂。他们以 Black Forest Labs 为例，说明此类许可策略如何导致混乱，并建议如果模型在自有硬件上运行或在与用户的特定距离内运行，许可应允许商业用途，以避免这些问题。
    - ambient_temp_xeno 指出了许可语言中的法律细微差别，指出虽然声明中没有明确提到对“代码编写”商业用途的限制，但早期的沟通中确实提到了。这突显了在许可条款中保持清晰和一致的信息传递对于避免误解和确保合规性的重要性。

  - **[本地 Minimax M2.7, GTA 基准测试](https://www.reddit.com/r/LocalLLaMA/comments/1sk70ph/local_minimax_m27_gta_benchmark/)** (活跃度: 383)：**该帖子讨论了使用 **Minimax M2.7** 模型在单个网页内创建类似 3D 侠盗猎车手 (GTA) 体验的基准测试。用户指出，虽然 **GLM 5** 在没有明确指令的情况下在美学和细节方面表现出色，但 Minimax M2.7 在被要求使用 boids algorithm 添加树木和鸟类时表现良好。该测试在 openwebui artifacts 窗口和 OpenCode 中进行，模型以 `IQ2_XXS` 运行以获得最大速度，并保持了连贯性和能力。图像显示了一个带有车辆和城市元素的方块化、风格化的游戏环境，表明这是一个驾驶模拟或基准测试。** 一条评论指出，**GLM 5** 在不需要特定提示语的情况下提供了更多关于主角的细节，表明它在某些美学方面可能更胜一筹。

    - -dysangel- 提到使用 **GLM 5** 进行对比，强调了它在没有额外提示语的情况下能提供更多关于主角细节的能力。这表明 GLM 5 在生成详细的角色描述方面可能具有先进的能力，可能对需要丰富叙事内容的应用有用。
    - EndlessZone123 批评将 **GLM** 用于 2D 或 3D 任务，认为它不是一个 Vision 模型，主要依靠记忆来完成 one-shot 任务。这暗示了 GLM 在处理连续或迭代视觉任务方面的局限性，这可能是开发人员在处理需要持续视觉处理的项目时需要考虑的因素。
    - averagebear_003 注意到基准测试中包含了鸟类，这可能表明 **Minimax M2.7** 模型在性能表现中对环境细节的关注。这对于评估模型在渲染具有动态元素的复杂场景方面的能力可能具有相关性。

### 3. 本地 AI 硬件与设置讨论

  - **[在讨论个人事务时，本地模型简直是福音](https://www.reddit.com/r/LocalLLaMA/comments/1ska9av/local_models_are_a_godsend_when_it_comes_to/)** (活跃度: 443): **该帖子讨论了使用本地模型（特别是支持 `256k context` 的 **Gemma 4 26B A4B 模型**）来分析个人日记。用户向模型分享了超过 `100k+ tokens` 的日记，并通过引导性提问来洞察经常出现的主题、回避的话题以及思想的演变。用户强调了本地模型相对于专有模型的隐私优势，突出了在个人设备上安全处理敏感信息的能力。这反映了在保持隐私的同时，使用 AI 进行个人数据分析的日益增长的趋势。** 评论者强调了本地模型的优势，例如能够将个人文档处理成可供查询的知识库，以及减少了对像成瘾性交互这类商业化驱动功能的需求。他们还提到了结构化日记习惯的历史背景，以及使用 AI 进行认知外部化的非治疗性质。

    - Unlucky-Message8866 描述了使用 `Qwen-3.5` 模型处理超过 10 年的个人文档，创建了一个全面的知识库。这种设置允许查询特定的个人数据，如过去的支出或人际关系，展示了模型在个人数据管理和检索方面的效用。
    - Not_your_guy_buddy42 强调了本地模型在避免旗舰模型面临的商业压力方面的优势。本地模型的设计初衷并非为了让用户上瘾或不必要地延长用户交互，这可以带来更真实、更少操纵感的用户体验。这与商业模型通常华而不实且带有权威色彩的特性形成了鲜明对比。
    - mobileJay77 提到使用 `Mistral 3.2` 模型，该模型与他们的硬件兼容，并且能够无限制地处理个人话题。这表明像 Mistral 这样的小型模型在个人使用场景中也能非常有效，在没有大型商业模型限制的情况下提供灵活性和隐私。

  - **[刚拿到这个……正在构建一个本地优先的东西 👀](https://www.reddit.com/r/LocalLLM/comments/1sk3zng/just_got_my_hands_on_one_of_these_building/)** (活跃度: 441): **图片展示了一块 NVIDIA RTX PRO 6000 Blackwell Max-Q 工作站版 GPU，用户计划将其集成到高性能的本地优先计算设置中。该配置包括 `9950X` CPU、`128GB RAM` 和一块 `ProArt board` 主板，表明其重点在于高级 AI 和服务器任务，而非游戏。用户旨在实现多用户并发推理并保持对数据的本地控制，避免依赖外部 API 提供商。他们正在探索 `vLLM` 和 `llama.cpp` 等技术，以构建能够高效处理多用户的系统，并计划未来可能增加第二块 GPU 以实现扩展性。** 一位评论者建议加入 RTX 6000 Discord 社区寻求建议，这表明这款高端 GPU 用户之间存在协作环境。另一条评论幽默地提到了购买如此强大 GPU 的诱惑，反映了尖端硬件的魅力。

    - Sticking_to_Decaf 分享了使用 RTX 6000 的详细设置，建议使用带有 `cu130 nightly image` 的 `vLLM`。他们强调在运行像 `Qwen3.5-27B-FP8` 这样的大型模型时，将 `kv cache dtype` 设置为 `fp8_e4m3`，在仅占用 `55%` 显存的情况下实现了 `160k tokens` 的最大上下文长度。性能指标包括单次请求 `80-90 tps`，多次并发请求超过 `250 tps`。该设置还容纳了 `whisper-large-v3`、一个 Embedding 模型和一个 Reranker 模型，并为可交换的 LoRAs 留有额外空间。




## 低技术门槛 AI 版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. AI 模型与基准测试发布


  - **[OpenRouter 刚刚发布了一个新的 100B 模型](https://www.reddit.com/r/Bard/comments/1skfbvf/openrouter_just_announced_a_new_100b_model/)** (热度: 240): **OpenRouter 发布了一个名为 "Elephant Alpha" 的新模型，这是一个拥有 `100B` 参数的模型，旨在提供业界领先的性能，并专注于 token 效率。该模型在代码补全、调试、文档处理以及支持轻量级 Agent 方面的能力尤为突出。这一发布使 "Elephant Alpha" 成为 AI 模型领域极具竞争力的产品，强调了其高效性和广泛的应用潜力。** 评论者推测 "Elephant Alpha" 可能与 "Grok" 模型有关，因为这类模型通常会先出现在 OpenRouter 上。大家还达成共识，认为它不是 Google 的模型，因为 Google 通常不会公开其自研模型的参数量。

    - Nick-wilks-6537 和 Artistic_Survey461 讨论了 OpenRouter 发布的这个新 100B 模型是 "Grok" 模型的可能性。他们提到这一推论是基于 X 平台上用户的探测和分析，并指出 OpenRouter 经常最初在隐藏或未命名的提供商下托管此类模型。
    - Capital-Remove-6150 对新模型的性能发表了评论，称在测试中它似乎并未达到 SOTA（业界领先）或接近 SOTA 的水平。这意味着虽然该模型参数量巨大，但其表现可能无法与领域内的顶尖模型相媲美。
    - SomeOrdinaryKangaroo 指出该模型不太可能来自 Google，因为 Google 通常不会透露其专有模型的参数量。这表明该模型的来源可能是另一家对这类细节更加透明的机构。


### 2. Sam Altman 安全事件

  - **[Sam Altman 的住所遭遇第二次袭击](https://www.reddit.com/r/singularity/comments/1sjtebt/sam_altmans_home_targeted_in_second_attack/)** (热度: 2227): ****Sam Altman** 在旧金山的住所遭遇了两起袭击：一起莫洛托夫鸡尾酒（汽油弹）事件和一起枪击事件。后者涉及一辆通过监控捕获的本田轿车，嫌疑人 Amanda Tom 和 Muhamad Tarik Hussein 已被逮捕。车辆的车牌信息促成了警方的行动。目前没有人员伤亡报告。[阅读更多](https://sfstandard.com/2026/04/12/sam-altman-s-home-targeted-second-attack/)。** 评论者批评媒体披露了 Altman 的住址，引发了隐私担忧，并讨论了亿万富翁采取的安全措施，如搬迁到安全的封闭式大院。


  - **[Sam Altman 住所遭纵火袭击数小时后又遭飞车枪击](https://www.reddit.com/r/OpenAI/comments/1sk82sc/sam_altmans_home_targeted_in_driveby_shooting/)** (热度: 1088): **据报道，**OpenAI** 的 CEO **Sam Altman** 的住所遭遇了飞车枪击，而就在几小时前刚刚发生过纵火袭击。这些事件在短时间内相继发生，引发了人们对知名科技高管安全的担忧。关于袭击的细节尚不充分，但它们凸显了科技行业重要人物可能面临的安全漏洞。** 评论反映了社会经济担忧以及对贫富差距和潜在社会动荡等更广泛社会问题的推测性联系，而非技术讨论。


  - **[Sam Altman 再次遭遇谋杀未遂，住所传出枪声](https://www.reddit.com/r/ChatGPT/comments/1skwdp7/another_murder_attempt_on_sam_altman_as_gunshots/)** (热度: 1087): **两名嫌疑人 Amanda Tom 和 Muhamad Tarik Hussein 因涉嫌在 **OpenAI** CEO **Sam Altman** 的住所附近开枪而在旧金山被捕。这是 Altman 在短时间内遭遇的第二次袭击，此前曾发生过一次纵火企图。嫌疑人面临过失释放火器的指控，警方在逮捕过程中没收了多件武器。据报道，这两起事件并无关联。更多细节请参见[原文报道](https://www.usatoday.com/story/news/crime/2026/04/13/sam-altman-house-attack/89586825007/)。** 评论者幽默地推测这是否有“时空旅行者”参与其中，反映出对当前事件的一种反乌托邦式看法。还有人讽刺地建议 Altman 可能会撤退到一个私人岛屿来开发先进的 AI 技术。



### 3. AI 模型性能与配置

- **[Claude 并非变笨了，只是它不再尽力。以下是在 Chat 中修复它的方法。](https://www.reddit.com/r/ClaudeAI/comments/1sjz1hg/claude_isnt_dumber_its_just_not_trying_heres_how/)** (热度: 1726): **这篇 Reddit 帖子讨论了 AI 模型 **Claude** 性能下降的问题，认为这是配置更改而非模型降级所致。Claude Code 用户可以通过输入 `/effort max` 恢复之前的行为，但 Chat 用户缺乏直接的开关。一种解决方法是在 Chat 界面中设置 Custom Instructions，以鼓励深入的推理和全面的分析。据报道，这种方法能恢复 Claude 深度处理上下文并提供详细回复的能力。帖子强调，这些指令对模型起到了强烈的信号作用，弥补了缺乏对努力程度（effort settings）直接控制的不足。** 评论者们辩论了 Token 效率与回复深度之间的平衡，并建议使用“斯巴达模式（Spartan mode）”来获得简洁但深刻的回答。另一条评论指出，Claude 的 System Prompt 允许它在认为不相关时忽略用户偏好，这表明设置“风格（styles）”可能比用户偏好在控制推理投入方面更有效。

    - m3umax 讨论了在 Claude 的 System Prompts 中使用“风格（styles）”而非用户偏好的重要性。他们指出，Claude 的网页版 System Prompt 允许它在认为不相关时忽略用户偏好，因此“风格”设置更为有效。他们提供了不同推理强度的风格示例，其中“高级版本”设定为 `99`，“中级版本”设定为 `85`，方便用户轻松切换思考等级。
    - Medium-Theme-4611 指出，Claude 表现出的“懒惰”源于其节省 Token 的行为。他们建议通过指令要求 Claude “研究并深入钻研（research and dive deep）”来抵消这一点，这暗示 Claude 的默认行为在没有明确指示的情况下，会优先考虑效率而非彻底性。
    - sidewnder16 将 Claude 与 Gemini 进行了类比，指出两者都需要明确的系统指令才能有效地执行任务。这表明如果没有清晰的指令，这些模型可能无法发挥全部潜力，突显了详细 Prompt 对于获得最佳性能的重要性。

  - **[Claude Code (~100 小时) 对比 Codex (~20 小时)](https://www.reddit.com/r/ClaudeCode/comments/1sk7e2k/claude_code_100_hours_vs_codex_20_hours/)** (热度: 1421): **该帖子在真实工程背景下对比了 **Claude Opus 4.6** 和 **Codex GPT-5.4**，重点关注一个拥有 80k LOC 的复杂 Python/TypeScript 项目。**Claude** 被描述为反应迅速且具交互性，但通常需要人工监督，并且倾向于忽略指南，导致任务完成不彻底及架构问题。相比之下，**Codex** 虽然速度较慢但更加深思熟虑，严格遵守指南，并能生成更整洁、更易维护的代码，无需持续监督。作者指出，虽然 Claude 适合快速原型设计（prototyping），但对于企业级软件开发，由于其周密的方法和对最佳实践的遵循，Codex 更胜一筹。** 评论者们普遍同意作者的评估，认为 Codex 较慢且更审慎的方法带来了更高质量的输出。然而，一些用户发现 Codex 的交流风格过于冗长，有时甚至不配合，这可能会让人感到沮丧。尽管存在这些问题，Codex 仍因其在自主完成任务方面的胜任能力和可靠性而受到称赞。

    - Temporary-Mix8022 讨论了 GPT-5.4 和 Opus 4.6 的对比表现，指出两个模型在解决问题方面能力相当，没有显著的性能差距。然而，他们批评了 Codex 的交流风格，认为其倾向于过度冗长并使用项目符号格式，这使得经验丰富的开发者难以解析。他们还提到由于 Reinforcement Learning (RL) 训练，Codex 有唱反调的倾向，这对于经验丰富的用户来说可能令人沮丧。
    - 该用户强调了 Codex Reinforcement Learning 训练中的一个特定问题，即它似乎优先考虑安全性和异议，可能导致无效的互动。他们对 Codex 无法专注于任务表示失望，认为这可能是因为它针对 Web 应用交互进行的训练所致。尽管如此，他们承认 Codex 在自主完成任务方面的有效性，在这方面通常优于 Opus。
    - Temporary-Mix8022 寻求关于优化 Codex/GPT-5.4 交流设置的建议，因为他们正挣扎于模型在过于冗长和缺乏细节之间反复横跳的问题。他们希望能有一种更平衡的交流风格，这表明尽管他们具备编程专业知识，但管理 LLM 的输出仍然具有挑战性。

- **[黄金时代已结束](https://www.reddit.com/r/ClaudeAI/comments/1sjqn2e/the_golden_age_is_over/)** (热度: 4149): **该帖子讨论了消费者和专业消费者在访问大语言模型 (LLMs)（如 Claude, ChatGPT, Gemini 和 Perplexity）时感知到的质量下降。作者指出，此前在分析文本对话方面表现出色的 Claude，现在的表现不佳，经常出错且显得漫不经心。ChatGPT 因回应过于热情而受到批评，Gemini 存在幻觉 (hallucinations) 问题，而 Perplexity 则缺乏深刻的分析。作者认为，高质量的 LLM 访问现在可能需要企业级的投资，并暗示这可能涉及计算资源问题或公司的策略性限流 (throttling)。帖子引用了来自 [ijustvibecodedthis.com](http://ijustvibecodedthis.com/) 的文章来支持这些观察。** 评论者认为，感知质量的下降可能是因为用户变得更擅长编写 Prompt，从而更清晰地触及了模型的局限性。另一种观点指出，海外模型和开源模型正在填补美国公司留下的空白，而后者被认为正在通过限制模型来节流智能。

    - CitizenForty2 强调了 Opus 的性能问题，指出与 Sonnet 相比，它消耗更多 Token 且耗时更长。在切换回 Sonnet 后，他们表示没有遇到其他人面临的常见问题，这表明 Sonnet 在他们的使用场景中可能更高效或更稳定。
    - kaustalautt 讨论了海外模型和开源模型填补美国公司留下的空白的趋势，这些公司被认为在限制智能。他们认为国际市场正通过采用开源方法来应对这一现状，从而可能提供对 AI 能力更少限制的访问。
    - bl84work 对 AI 模型进行了对比分析，指出 Gemini 的幻觉率很高，而 Claude 表现出自我纠错行为，会中断对话以纠正不准确之处。ChatGPT 被描述为“自信地犯错”，突显了模型行为和可靠性方面的差异。



# AI Discords

遗憾的是，Discord 今天关闭了我们的访问权限。我们不会以这种形式恢复它，但我们很快就会发布新的 AINews。感谢读到这里，这是一段美好的历程。