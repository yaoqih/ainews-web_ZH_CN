---
companies:
- anthropic
- openai
date: '2026-04-17T05:44:39.731046Z'
description: '**Anthropic** 推出了由 **Claude Opus 4.7** 驱动的原型设计工具 **Claude Design**，旨在优化设计工作流，并与
  **Figma** 等产品展开竞争。基准测试显示，**Opus 4.7** 在编程和文本任务中处于领先地位，并提升了效率和自适应推理能力，尽管早期用户反馈指出其存在一些性能退步和稳定性问题。相关讨论强调了其与
  **Gemini 3.1 Pro** 及 **GPT-5.4** 相比的成本效益和智能体（agentic）能力。与此同时，**OpenAI** 的 Codex
  更新引入了先进的“计算机使用”（computer-use）功能，能够实现对桌面应用和企业软件快速、智能的控制，标志着在开发实用类 AGI 智能体方面取得了进展。'
id: MjAyNS0x
models:
- claude-opus-4.7
- gemini-3.1-pro
- gpt-5.4
- claude-code
- codex
people:
- claudeai
- yuchenj_uw
- kimmonismus
- skirano
- therundownai
- arena
- artificialanlys
- victortaelin
- emollick
- alexalbert__
- theo
- scaling01
- reach_vb
- kr0der
- hamelhusain
- mattrickard
- matvelloso
- gdb
title: 今天没什么事。
topics:
- agentic-ai
- model-benchmarking
- adaptive-reasoning
- cost-efficiency
- computer-use
- prototyping-tools
- code-generation
- model-performance
- software-integration
---

**平静的一天。**

> 2026年4月16日至4月17日的 AI 新闻。我们检查了 12 个 subreddit、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有查看更多 Discord。[AINews 网站](https://news.smol.ai/) 允许你搜索所有往期内容。提醒一下，[AINews 现在是 Latent Space 的一个板块](https://www.latent.space/p/2026)。你可以[选择加入/退出](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)邮件接收频率！

---

# AI Twitter 回顾

**Anthropic 发布 Claude Opus 4.7 和 Claude Design**

- **Claude Design 作为 Anthropic 的首个设计/原型制作界面发布**：[@claudeai](https://x.com/claudeai/status/2045156267690213649) 宣布了 **Claude Design**，这是一个研究预览版工具，由 **Claude Opus 4.7** 驱动，可根据自然语言指令生成原型、幻灯片和单页介绍。此次发布立即使 Anthropic 被视为正在从聊天/编码领域扩展到设计工具领域；多位观察者称其直接对标 **Figma/Lovable/Bolt/v0**，包括 [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2045158071950033063)、[@kimmonismus](https://x.com/kimmonismus/status/2045162358004216134) 和 [@skirano](https://x.com/skirano/status/2045192705941106992)。市场反应本身也成了新闻，[@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2045161719547445426) 等人指出 Figma 在发布会后股价大幅下跌。产品细节由 [@TheRundownAI](https://x.com/TheRundownAI/status/2045176722476208454) 披露：支持行内微调、滑块调节、导出至 **Canva/PPTX/PDF/HTML**，并可移交给 **Claude Code** 进行实现。
- **Opus 4.7 整体表现更强，但发布过程伴随着杂音**：第三方基准测试的帖子普遍持正面态度。[@arena](https://x.com/arena/status/2045177492936532029) 将 **Opus 4.7 排在 Code Arena 第一名**，比 Opus 4.6 高出 37 分，领先于非 Anthropic 的同类模型；该账号还将其排在 **Text Arena 综合第一名**，在编码和科学密集型领域均有斩获（详见[此处](https://x.com/arena/status/2045177497378316597)）。[@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2045292578434875552) 报告称其 **Intelligence Index** 榜首出现了近乎“三足鼎立”的局面——**Opus 4.7 57.3**、**Gemini 3.1 Pro 57.2**、**GPT-5.4 56.8**，同时将 Opus 4.7 排在 Agent 能力基准测试 **GDPval-AA** 的首位。他们还注意到，与 Opus 4.6 相比，其在得分更高的情况下**输出 Token 减少了约 35%**，并引入了 **task budgets**，同时完全移除了扩展思维（extended thinking），转而采用自适应推理（adaptive reasoning）。但在发布后的最初 24 小时内，用户体验评价褒贬不一：[@VictorTaelin](https://x.com/VictorTaelin/status/2045139180359942462) 报告了性能退化和上下文失效问题，[@emollick](https://x.com/emollick/status/2045147490316374414) 表示 Anthropic 在第二天就已经改进了自适应思维的表现，[@alexalbert__](https://x.com/alexalbert__) 则确认许多初始 Bug 已被修复。此外，[@theo](https://x.com/theo/status/2045310884717981987) 对 Design 本身的产品稳定性提出了投诉，并在[此处](https://x.com/theo/status/2045317666383204423)提到了账号级别的安全性问题。
- **成本/效率的讨论变得与原始质量几乎同等重要**：[@scaling01](https://x.com/scaling01/status/2045160883010081237) 声称，与之前的顶尖模型相比，在运行某些机器学习（ML）问题时 **Token 消耗减少了约 10 倍**，且保持了类似的性能。同时 [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2045206342173086156) 将 Opus 4.7 置于文本和代码领域的**性价比帕累托前沿（Pareto frontier）**。并非所有基准测试都认同其绝对领先地位——例如 [@scaling01](https://x.com/scaling01/status/2045178622617498084) 指出它在 **LiveBench** 上仍落后于 **Gemini 3.1 Pro** 和 **GPT-5.4**。但这些帖子的共识是，Anthropic 实质性地提升了模型的 Agent 效用和效率。

**计算机使用（Computer use）、编码 Agent 与评测框架设计**

- **Computer-use UX 正在成为主流产品类别**：OpenAI 的 Codex 桌面/计算机操作（computer-use）更新引起了从业者不同寻常的强烈反应。[@reach_vb](https://x.com/reach_vb/status/2045151640802771394) 将 **subagents + computer use** 称为在实际感受上“非常接近” AGI；[@kr0der](https://x.com/kr0der/status/2045154074337710136)、[@HamelHusain](https://x.com/HamelHusain/status/2045191726495846459)、[@mattrickard](https://x.com/mattrickard/status/2045218583882633412) 和 [@matvelloso](https://x.com/matvelloso/status/2045209294942142860) 都强调 Codex Computer Use 不仅仅是华丽，而且**速度极快**，能够驱动 **Slack、浏览器工作流和任意桌面应用**，并且可能是第一个真正可用于企业遗留软件的计算机操作平台。[@gdb](https://x.com/gdb/status/2045375289560007029) 明确将 Codex 定位为正在进化的**全智能体 IDE (full agentic IDE)**。
- **该领域正趋向于“简单的测试框架 (harness)、强大的评估 (evals) 和模型无关的脚手架 (scaffolding)”**：几篇高质量帖子指出，可靠性的提升现在更多源于测试框架，而非追求超大规模模型。[@AsfiShaheen](https://x.com/AsfiShaheen/status/2045072599508508914) 描述了一个三阶段金融分析师流水线——**router / lane / analyst**——具有严格的上下文边界和针对每个阶段的黄金数据集 (gold sets)，认为许多 bug 实际上是指令/接口 bug。[@AymericRoucher](https://x.com/AymericRoucher/status/2045176781414527305) 从泄露的 Claude Code 测试框架中总结了同样的经验：简单的规划约束加上更整洁的表示层，其表现优于“花哨的 AI 脚手架”。[@raw_works](https://x.com/raw_works/status/2045208764509470742) 展示了一个更明显的例子：**Qwen3-8B** 配合 **dspy.RLM** 在 LongCoT-Mini 上获得了 **33/507** 的分数，而原生版本 (vanilla) 为 **0/507**，认为脚手架——而非微调——承担了“100% 的重任”。LangChain 将更多此类模式引入了产品：[@sydneyrunkle](https://x.com/sydneyrunkle/status/2045209395881980276) 为 **`deepagents deploy` 增加了 subagent 支持**，[@whoiskatrin](https://x.com/whoiskatrin/status/2045139949939200284) 宣布了 **Agents SDK 中的记忆基元 (memory primitives)**。
- **开源 Agent 栈继续激增**：Hermes Agent 仍是焦点。来自 [@GitTrend0x](https://x.com/GitTrend0x/status/2045142797439922337) 的社区生态系统概览强调了诸如 **Hermes Atlas**、**Hermes-Wiki**、HUDs 和控制面板等衍生品。[@ollama](https://x.com/ollama/status/2045282803387158873) 随后通过 `ollama launch hermes` 发布了**原生 Hermes 支持**，[@NousResearch](https://x.com/NousResearch/status/2045304840645939304) 对此进行了转发。Nous 和 Kimi 还发起了 **2.5 万美元的 Hermes Agent 创意黑客松** [@NousResearch](https://x.com/NousResearch/status/2045225469088326039)，标志着从编码/生产力向**创意 Agent** 工作流的推进。

**Agent 研究：自我提升、监控、网页技能和评估**

- **一系列论文推动了 Agent 的鲁棒性和持续改进**：[@omarsar0](https://x.com/omarsar0/status/2045139481779696027) 总结了 **Cognitive Companion**，它通过 **LLM** 裁判或隐藏状态 **probe** 来监控推理退化。核心结果非常显著：在 **layer-28 hidden states** 上使用 **logistic-regression probe** 可以检测到性能退化，其 **AUROC** 达到 **0.840**，且**测得的推理开销为零**；而 **LLM** 监控版本在约 11% 的开销下，将重复率降低了 **52–62%**。来自 [@dair_ai](https://x.com/dair_ai/status/2045139481892880892) 的另一项关于 **web agents** 的工作介绍了 **WebXSkill**，**Agent** 可以从轨迹中提取可重用的技能，在 **WebArena** 上提升了高达 **+9.8 分**，在 **grounded 模式**下的 **WebVoyager** 上达到了 **86.1%**。此外，[@omarsar0](https://x.com/omarsar0/status/2045241905227915498) 还重点介绍了 **Autogenesis**，这是一种让 **Agent** 识别能力差距、提出改进建议、验证并整合有效更改而无需重新训练的协议。
- **开放世界评估（Open-world evals）正成为一个严肃的主题**：多篇文章认为当前的 **benchmarks** 过于狭窄。[@CUdudec](https://x.com/CUdudec/status/2045139195220431022) 支持针对 **long-horizon**、开放式场景进行开放世界评估；[@ghadfield](https://x.com/ghadfield/status/2045245020429570505) 将此与监管和“**economy of agents**”问题联系起来；[@PKirgis](https://x.com/PKirgis/status/2045265295649231354) 讨论了 **CRUX**，这是一个旨在对真实复杂环境中的 **AI agents** 进行定期**开放世界评估**的项目。在测量方面，[@NandoDF](https://x.com/NandoDF/status/2045063560716296450) 建议在跨 **2500 个主题桶**的 **out-of-training-domain** 书籍/文章上，建立广泛的基于 **NLL/perplexity** 的评估套件，尽管这引发了来自 [@eliebakouch](https://x.com/eliebakouch/status/2045115926123520100)、[@teortaxesTex](https://x.com/teortaxesTex/status/2045139476972745120) 等人关于 **RLHF/post-training** 后 **perplexity** 是否仍具有信息量的争论。
- **文档/OCR 和检索评估也变得更加以 Agent 为中心**：[@llama_index](https://x.com/llama_index/status/2045145054772183128) 扩展了 **ParseBench**，这是一个以 **content faithfulness** 为核心的 **OCR benchmark**，包含超过 **16.7 万个基于规则的测试**，涵盖遗漏、幻觉和阅读顺序错误——明确将标准从“人类可读”重新定义为“足够可靠，足以让 **Agent** 据此采取行动”。在检索领域，[@Julian_a42f9a](https://x.com/Julian_a42f9a/status/2045200413402493064) 指出，新的研究表明 **late-interaction** 检索表示可以在 **RAG** 中替代原始文档文本，这意味着某些 **RAG** 流水线可能能够绕过全文重建。

**开源模型、本地推理和推理系统**

- **Qwen3.6 本地/量化工作流是一个务实的亮点**：[@victormustar](https://x.com/victormustar/status/2045068986446958899) 分享了一个具体的 **llama.cpp + Pi** 设置，将 **Qwen3.6-35B-A3B** 作为本地 **agent stack**，强调了本地 **agentic systems** 现在看起来是多么可行。**Red Hat** 紧随其后发布了 **NVFP4** 量化的 **Qwen3.6-35B-A3B** 检查点 [@RedHat_AI](https://x.com/RedHat_AI/status/2045153791402520952)，报告初步实现了 **GSM8K Platinum 100.69%** 的恢复率，而 [@danielhanchen](https://x.com/danielhanchen/status/2045169369723064449) 对动态量化进行了基准测试，声称许多 **Unsloth** 量化模型处于 **KLD** 与磁盘空间的 **Pareto frontier** 上。
- **消费级硬件推理持续改进**：[@RisingSayak](https://x.com/RisingSayak/status/2045114073000657316) 宣布了与 **PyTorch/TorchAO** 的合作，实现了支持 **FP8** 和 **NVFP4** 量化的 **offloading**，且没有显著的延迟惩罚，明确针对受内存限制的消费级 **GPU** 用户。**Apple** 端的本地推理也得到了展示，[@googlegemma](https://x.com/googlegemma/status/2045204738720084191) 演示了 **Gemma 4** 在 **iPhone** 上完全离线运行并支持长上下文。
- **值得注意的推理基础设施更新**：[@vllm_project](https://x.com/vllm_project/status/2045381618928582995) 重点介绍了与 **AMD/EmbeddedLLM** 合作的 **MORI-IO KV Connector**，声称通过类似 **PD-disaggregation** 风格的连接器，在单节点上实现了 **2.5 倍更高**的 **goodput**。**Cloudflare** 继续推进其 **Agent/AI** 平台建设，推出了 **isitagentready.com** [@Cloudflare](https://x.com/Cloudflare/status/2045126394418503846)、**Flagship** 功能标记 [@fayazara](https://x.com/fayazara/status/2045133183575113771)，以及共享压缩字典，在一个示例中实现了剧烈的有效载荷缩减，例如从 **92KB 降至 159 字节** [@ackriv](https://x.com/ackriv/status/2045177696506794336)。

**AI 用于科学、医学和基础设施**

- **科学发现和个性化健康是突出的应用主题**：[@JoyHeYueya](https://x.com/JoyHeYueya/status/2045147082546462860) 和 [@Anikait_Singh_](https://x.com/Anikait_Singh_/status/2045149764636094839) 发布了关于 **insight anticipation**（见解预判）的内容，即模型可以从其“父”论文中生成下游论文的核心贡献；后者介绍了 **GIANTS-4B**，这是一个经过 RL 训练的模型，据报道在该任务上超越了前沿模型。在健康方面，[@SRSchmidgall](https://x.com/SRSchmidgall/status/2045023895041061353) 分享了一个基于可穿戴数据的生物标志物发现系统，其首个发现是“**深夜刷手机（late-night doomscrolling）**”可以预测抑郁严重程度，相关系数为 **ρ=0.177, p<0.001, n=7,497** —— 值得注意的是，该特征是由模型本身命名的。另外，[@patrickc](https://x.com/patrickc/status/2045164908912968060) 认为目前的编程 Agent 在**个性化基因组解释**方面已经非常有用，他描述了花费不到 100 美元的分析运行就发现了个体患黑色素瘤的风险高出约 **30 倍**，并提出了后续干预措施。
- **大规模算力建设仍然是一个核心的元叙事**：[@EpochAIResearch](https://x.com/EpochAIResearch/status/2045258390147088764) 调研了全部 **7 个美国 Stargate 站点**，并得出结论：该项目有望在 **2029 年前实现 9+ GW** 的规模，这相当于**纽约市的峰值需求**。[@gdb](https://x.com/gdb/status/2045279841482928271) 将 Stargate 描述为“**算力驱动型经济**”的基础设施，而 [@kimmonismus](https://x.com/kimmonismus/status/2045206835238441332) 将当今全球年度数据中心资本支出折算后，认为其相当于每年投入了 **5–7 个曼哈顿计划（Manhattan Projects）**。

**热门推文（按互动量排序）**

- **Claude Design / Anthropic 产品扩张**：[@claudeai 发布 Claude Design](https://x.com/claudeai/status/2045156267690213649)，这是当天目前为止最大的纯 AI 产品发布信号。
- **模型基准测试 / 排名**：[@ArtificialAnlys 指出 Opus 4.7 在综合排名中并列第一，并在 GDPval-AA 中领先](https://x.com/ArtificialAnlys/status/2045292578434875552)。
- **编程 Agent / 计算机使用 (computer use)**：[@cursor_ai 在新的 Agent 窗口中将 Composer 2 的限制提高了一倍](https://x.com/cursor_ai/status/2045236540784492845)，以及 [@HamelHusain 关于 Codex Computer Use 的内容](https://x.com/HamelHusain/status/2045191726495846459)。
- **开源 Agent**：[@ollama 发布了对 Hermes Agent 的原生支持](https://x.com/ollama/status/2045282803387158873)。
- **AI 在医学领域的应用**：[@patrickc 谈论用于基因组分析和个性化预防的编程 Agent](https://x.com/patrickc/status/2045164908912968060)。
- **基础设施 / 电力扩展**：[@EpochAIResearch 关于 Stargate 9+ GW 发展轨迹的内容](https://x.com/EpochAIResearch/status/2045258390147088764)。


---

# AI Reddit 热点回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Qwen3.6 模型的发布与特性

  - **[Qwen3.6。就是它了。](https://www.reddit.com/r/LocalLLaMA/comments/1so1533/qwen36_this_is_it/)** (热度: 1483): **该贴讨论了 LLM 模型 **Qwen3.6** 在自主构建塔防游戏方面的能力，它能够识别并修复 Bug，例如画布渲染问题和波次完成错误。该模型使用 `llama-server` 设置进行部署，并采用了特定的配置，包括 `Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf` 和 `mmproj-F16.gguf` 文件，运行参数如 `--cpu-moe`、`--top-k 20` 和 `--temp 0.7`。用户强调了模型的高效性，在 NVIDIA 3090 GPU 上达到了 `120 tk/s`，并且能够快速解决其他模型难以处理的代码问题。** 评论者对该模型的性能感到惊讶，指出了它对未来一代的潜在影响，以及与 Gemma 等其他模型相比的高效性。人们对用于部署的技术栈表现出浓厚兴趣，希望能有类似的本地配置。

    - **cviperr33** 强调了 Qwen3.6 模型令人印象深刻的性能，指出它能够快速、高效地修复损坏的代码。他们报告在 `NVIDIA 3090` 上使用 `llama.cpp` 达到了 `120 tokens/second`，在 `3.8k-5k` token 范围内实现了即时预填充（prefill）。这种速度实现了快速响应和高效的文件编辑，而不会使 GPU 过载，这与 Gemma 模型较慢的性能形成了鲜明对比。
    - **PotatoQualityOfLife** 询问了所使用模型的具体大小或量化（quantization）版本，这是了解模型性能和资源要求的关键因素。这个问题表明了对优化本地部署配置的关注，这会显著影响速度和效率。
    - **No-Marionberry-772** 表达了对搭建运行 Qwen3.6 等模型的本地环境的兴趣，但在选择合适的软件栈方面面临挑战。这反映了用户在本地尝试利用先进模型时遇到的普遍问题，表明需要关于最佳配置的更清晰指导或资源。

- **[Qwen 3.6 是第一个让我觉得真正值得投入精力的本地模型](https://www.reddit.com/r/LocalLLaMA/comments/1so2nt9/qwen_36_is_the_first_local_model_that_actually/)** (热度: 512): **用户报告称 `qwen3.6-35b-a3b` 模型是第一个在项目中（特别是针对 Avalonia 的 UI XML 和嵌入式系统的 C++）表现出高效且值得使用的本地模型。在 `5090 + 4090` 的配置下，该模型在 `260k context` 下达到了 `170 tokens per second` 的速度，通过极少的修正就超越了 Gemma 4 等其他模型。这表明本地模型的能力有了显著提升，有可能减少对云端解决方案的依赖。** 评论反映了对该模型性能的分歧意见，一些用户对其能力表示怀疑，而另一些用户则注意到发布后评价呈现两极分化。

    - -Ellary- 强调了 Qwen 3.6 与其他模型之间的性能差异，指出 Qwen 3.5 27b 在任务执行和问题解决方面更胜一筹。他们建议，如果硬件资源允许，运行完整的 GLM 4.7 358B A32B（使用 IQ4XS 或 IQ3XXS 量化）将比 Qwen 3.6 35b A3B 获得显著更好的结果，他们认为后者是类似于 9-12b dense models 的轻量级模型。
    - kmp11 提到了 Hermes-Agent 与 Qwen 3.6 搭配时的出色表现，并指出其能够以超过 100 tokens per second 的速度处理无限数量的 token。这表明在快速处理海量数据方面具有极高的效率和能力，对于需要快速 token 处理的应用非常有益。

  - **[Qwen3.6 配合 OpenCode 简直不可思议！](https://www.reddit.com/r/LocalLLaMA/comments/1so3rsx/qwen36_is_incredible_with_opencode/)** (热度: 436): **该帖子讨论了本地 AI 模型 **Qwen3.6** 在 **RTX 4090** (24 GB VRAM) 上使用 `llama.cpp` 部署时的性能。用户在涉及 Rust、TypeScript 和 Python 服务的代码库中，测试了该模型实现 PostgreSQL RLS 这一复杂任务的表现。尽管存在一些 Bug，但该模型表现良好，能根据编译器错误进行迭代并优化代码更改。配置包括 **Qwen3.6-35B-A3B, IQ4_NL unsloth quant**，Context 长度为 `262k`，VRAM 占用约 `21GB`。部署使用了 **docker** 并设置了特定参数以防止 OOM 错误，实现了 `100+ output tokens per second`。** 评论者们对硬件限制（如只有 `16GB` VRAM）表示遗憾，并分享了使用 Qwen3.6 的积极体验，注意到它处理涉及多个 subagents 和 tool calls 的复杂任务的能力。虽然注意到了一些问题，例如 subagents 未保存输出和演示错误，但这些都在迭代中得到了解决。

    - Durian881 分享了将 Qwen 3.6 与 Qwen Code 结合使用的详细经验，强调了它处理涉及“麦肯锡研究技能”复杂任务的能力，该任务包含 9-12 个 subagents 以及大量的 websearch 和 webfetch 等 tool calls。整个过程耗时超过 1.5 小时，尽管在 subagents 保存输出和幻灯片渲染方面存在一些问题，但模型能够恢复并生成高质量的 HTML 幻灯片。这些修复过程被拿来与 Gemini 3 Pro 进行比较，后者在幻灯片顺序和标题页方面也存在类似问题。
    - robertpro01 将 Qwen 3.6 与 Gemini 3 Flash 进行了比较，指出其性能与后者持平，这意味着如果用户能够有效使用 Qwen 3.6，可能就不需要为 Gemini 3 Flash 付费。这表明 Qwen 3.6 以可能更低的成本提供了极具竞争力的性能，使其成为寻求高性价比解决方案用户的诱人选择。
    - RelicDerelict 询问了在 4GB VRAM 和 32GB RAM 的系统上运行 Qwen 3.6 的情况，表明用户有兴趣了解获得最佳性能所需的硬件要求。这突显了硬件资源有限的用户中的普遍关注点，他们希望在无需高端设备的情况下利用 Qwen 3.6 等先进模型。

- **[Qwen3.6-35B-A3B 发布！](https://www.reddit.com/r/LocalLLaMA/comments/1sn3izh/qwen3635ba3b_released/)** (活跃度: 3494): **图片展示了新发布的 **Qwen3.6-35B-A3B** 的性能，这是一个稀疏 MoE 模型，拥有 `35B` 总参数和 `3B` 激活参数，突显了其在各种基准测试中的竞争优势。该模型基于 Apache 2.0 许可证发布，展示了与激活规模大其十倍的模型相当的 Agentic coding 能力，并在多模态感知和推理方面表现出色。图中的柱状图说明了 Qwen3.6-35B-A3B 在 coding 和推理等任务中的优越性能，超越了稠密参数模型 Qwen3.5-27B 及其前身 Qwen3.5-35B-A3B，特别是在 Agentic coding 和推理任务中。[查看图片](https://i.redd.it/g6edjlxt0kvg1.jpeg)** 评论者注意到 Qwen3.6-35B-A3B 令人印象深刻的性能，特别是在 coding 基准测试中，并表达了对未来可能挑战 Google 等公司主流模型的新版本的期待。

    - Qwen3.6-35B-A3B 相比其前身展示了显著改进，特别是在 coding 和推理任务中。它在多个关键 coding 基准测试中超越了稠密的 27B 参数模型 Qwen3.5-27B，并大幅领先 Qwen3.5-35B-A3B，特别是在 Agentic coding 和推理任务上，标志着本地 LLM 性能的巨大飞跃。
    - Qwen3.6-35B-A3B 模型是原生多模态的，展现了先进的感知和多模态推理能力。尽管只有约 30 亿激活参数，它在视觉语言基准测试中表现异常出色，在多项任务中比肩或超越了 Claude Sonnet 4.5。值得注意的是，它在 RefCOCO 上获得了 92.0 分，在 ODInW13 上获得了 50.8 分，突显了其在空间智能方面的优势。
    - 人们对更大规模的 Qwen3.6 模型（可能是 122B 版本）的发布充满期待，这可能会给 Google 等竞争对手带来压力，促使其发布自己的大模型。这种竞争可能会使 GLM 5.1 和 Sonnet 4.6 等模型进入更直接的对比，表明大规模模型开发领域正处于快速演变之中。

### 2. Qwen3.6 基准测试与性能

  - **[Qwen3.6 GGUF Benchmarks](https://www.reddit.com/r/LocalLLaMA/comments/1so5nrl/qwen36_gguf_benchmarks/)** (热度: 588): **该图像是 Qwen3.6 GGUF 的性能基准测试图表，展示了各种量化提供商的平均 KL 散度（Mean KL Divergence）与磁盘空间的关系。图表强调了 **Unsloth** 量化在帕累托前沿（Pareto frontier）占据主导地位，在 22 个案例中的 21 个里实现了 KL 散度与磁盘空间之间的最佳权衡。这表明 Unsloth 的量化模型在性能和存储方面效率极高。该帖还解决了有关频繁更新的误解，澄清了大多数问题源于外部因素，并强调了 CUDA 13.2 中一个已确认的、会影响低比特（low-bit）量化的 Bug，预计将在 CUDA 13.3 中修复。**

    - **danielhanchen** 强调了 CUDA 13.2 的一个关键问题，即所有 4-bit 量化都会产生乱码输出。此问题影响所有量化提供商，并正如 NVIDIA 在 [GitHub issue comment](https://github.com/ggml-org/llama.cpp/issues/21255#issuecomment-4248403175) 中指出的那样，已确认将在即将发布的 CUDA 13.3 版本中解决。建议遇到此问题的用户暂时退回到 CUDA 13.1 作为权宜之计。
    - **tavirabon** 批评了基准测试中数据的选择性呈现，认为分析中使用百分比是为了偏袒受问题影响的模型。评论还提到分析中存在明显的偏见，特别是在处理竞争方面，具体提到了针对 Bartowski 的行动，这似乎脱离了语境并影响了分析的中立性。
    - **PiratesOfTheArctic** 赞赏图形化数据表示的清晰度，这简化了那些不太熟悉技术细节的人的理解。这表明基准测试中提供的视觉辅助工具能有效向更广泛的受众传达复杂信息。

  - **[Ternary Bonsai: Top intelligence at 1.58 bits](https://www.reddit.com/r/LocalLLaMA/comments/1snqo1f/ternary_bonsai_top_intelligence_at_158_bits/)** (热度: 532): ****Ternary Bonsai** 是 **PrismML** 推出的一系列新型语言模型，旨在通过三进制权重 {-1, 0, +1} 以每个权重 `1.58 bits` 的精度运行。这种方法使模型能够保持比传统 16-bit 模型小约 `9倍` 的内存占用，同时在标准基准测试中实现卓越性能。模型有 `8B`、`4B` 和 `1.7B` 参数规格，可通过 [Hugging Face](https://huggingface.co/collections/prism-ml/ternary-bonsai) 获取。此次发布包括了用于兼容现有框架的 FP16 safetensors，尽管 **MLX 2-bit format** 是目前唯一可用的压缩格式，预计很快会支持更多格式。更多详情请参阅 [官方博客文章](https://prismml.com/news/ternary-bonsai)。** 一些评论者质疑模型尺寸的呈现方式，认为使用 Q4 量化大型模型可以在不显著损失性能的情况下缩小尺寸差异。其他人则表达了对更大模型（如 20-40B 参数）的期待，认为这将对该领域产生重大影响。

    - **r4in311** 和 **DefNattyBoii** 讨论了模型基准测试中潜在的误导性比较，指出在不考虑量化（例如 Q4）的情况下展示 8B/9B 模型的全权重会夸大尺寸差异。他们认为量化模型可以在减小尺寸的同时保持性能，并批评在基准测试中使用 Qwen3 等过时模型，主张与 Qwen3.5 和 Gemma4 等更新的模型进行比较。
    - **DefNattyBoii** 对缺乏与主流推理框架（如 `llama.cpp`、`vllm` 和 `sglang`）的协作表示担忧，认为这可能会限制所讨论模型的实际适用性和集成。这种集成的缺乏可能会阻碍这些模型在实际应用中的采用和性能优化。
    - **Kaljuuntuva_Teppo** 强调了当前模型在使用具有 24-32 GB 显存的消费级 GPU 方面的局限性。他们表达了对能够更好利用此类硬件的模型的渴望，认为当前模型太小，无法充分利用可用资源，这可能导致性能和资源使用效率低下。

### 3. Qwen3.6 Uncensored Aggressive 变体

  - **[Qwen3.6-35B-A3B Uncensored Aggressive 已发布，包含 K_P 量化版本！](https://www.reddit.com/r/LocalLLaMA/comments/1snlo6s/qwen3635ba3b_uncensored_aggressive_is_out_with_k/)** (热度: 433): **Qwen3.6-35B-A3B Uncensored Aggressive** 模型已经发布，它采用了与之前的 3.5-35B 相同的 `35B` MoE 规模，但基于更先进的 3.6 架构。该变体是完全无审查的，具有 **0/465 的拒绝率**，且没有进行性格修改，在保持完整能力的同时没有性能衰减。它包含了多种量化格式，如 `Q8_K_P`、`Q6_K_P` 等，均使用 **imatrix** 生成以优化性能。该模型支持多模态输入（文本、图像、视频），并在 `40 层` 中采用了 `3:1` 的线性与 Softmax 比例的混合注意力机制。它兼容 `llama.cpp` 和 `LM Studio` 等平台，尽管由于自定义量化命名，某些 GUI 标签可能无法正确显示。评论者对无审查模型“无质量下降”的说法表示怀疑，并批评了使用独特的量化命名公约，认为这会破坏 GUI 的兼容性。此外，还有人呼吁在“零能力损失”的测试方法上提高透明度。

    - 一位用户对 Qwen3.6-35B-A3B Uncensored Aggressive 模型“零能力损失”的说法表示怀疑，指出通常无审查模型会出现质量下降。这凸显了需要详细的测试方法和基准测试来证实此类说法，正如评论者指出的，目前缺乏关于这些测试如何进行的详细信息。
    - 另一位评论者批评了对自定义量化使用新术语的做法，认为其描述与现有的 “imatrix” 等方法一致。他们认为，为已有的技术发明新词会导致混淆，并与依赖标准命名公约的 GUI 产生兼容性问题，主张使用更广泛认可的标签，如 “K_L” 或 “K_XL”。
    - 有人提到可供下载的量化文件有限，这表明发布可能尚未完成或仍在进行中。这暗示想要尝试该模型的用户可能会遇到延迟，或者需要等待完整文件上传完毕。

  - **[Qwen3.6-35B-A3B Uncensored Aggressive 已发布，包含 K_P 量化版本！](https://www.reddit.com/r/LocalLLM/comments/1snlo1x/qwen3635ba3b_uncensored_aggressive_is_out_with_k/)** (热度: 357): **Qwen3.6-35B-A3B Uncensored Aggressive** 模型已发布，具有与之前 3.5-35B 相同的 `35B` MoE 规模，但基于最新的 3.6 架构。该变体完全无审查，具有 **0/465 的拒绝率**，在保持完整能力的同时没有性格修改。它包括 `Q8_K_P`、`Q6_K_P` 等多种量化格式，针对质量进行了优化，文件体积略有增加。该模型支持多模态输入（文本、图像、视频）并使用混合注意力机制。它兼容 `llama.cpp` 和 `LM Studio` 等平台，尽管后者可能会出现一些显示问题。更多详情请参见 [Hugging Face 模型页面](https://huggingface.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive)。一位用户询问“无性格变化”的含义，表现出对模型行为的好奇。另一位用户对这些发布版本的一贯品质表示赞赏，表明了对该开发者模型的偏好。

    - 模型名称 “Qwen3.6-35B-A3B” 代表了特定的特征：“Qwen” 是模型家族，“3.6” 可能指版本，“35B” 表示参数数量（350 亿），而 “A3B” 可能表示特定的架构或训练配置。“K_P” 量化是指一种在保持性能的同时减小模型体积的方法，尽管 “K_P” 的确切含义并非普遍定义，可能会因语境而异。
    - 关于硬件兼容性，一位用户询问模型的 “q3” 量化版本是否能在 24GB 的 NVIDIA 4090 GPU 上高效运行。“q3” 量化建议使用较低的精度格式以减少显存占用，从而可能使模型适应 GPU 的内存限制。然而，人们担心这种量化是否会显著降低模型质量，这取决于具体的实现和使用场景。
    - “无性格变化”一词可能指模型的行为在不同版本或配置中保持一致。这意味着尽管进行了更新或量化更改，模型的回答和交互风格应保持稳定，从而确保在需要一致行为的应用中的可靠性。

## 非技术性 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude Opus 4.7 的性能表现与反响

  - **[opus 4.7 (high) 在 NYT Connections Extended 基准测试中得分为 41.0%，而 opus 4.6 得分为 94.7%。](https://www.reddit.com/r/singularity/comments/1so2vmc/opus_47_high_scores_a_410_on_the_nyt_connections/)** (活跃度: 1287): **Opus 4.7** 在 NYT Connections Extended 基准测试中仅获得 `41.0%` 的分数，相较于 **Opus 4.6** 的 `94.7%` 有显著下降。该基准测试详见 [此 GitHub 仓库](https://github.com/lechmazur/nyt-connections/)，通过 940 个增加了复杂度的 NYT Connections 谜题来评估 LLM。值得注意的是，Opus 4.7（无推理版）以 `15.3%` 的得分排名垫底，基准测试作者指出，这归因于安全顾虑导致的拒绝回答，而非答案错误。在它参与评估的谜题中，Opus 4.7 得分为 `90.9%`，仍低于 Opus 4.6。评论者注意到了该模型的成本节约特性，并对性能下降表示困惑，强调了安全拒绝机制对结果的影响。

    - 用户 Klutzy-Snow8016 指出，Opus 4.7 相较于 4.6 的性能下降归因于安全顾虑导致的拒绝率增加。这一调整导致 Opus 4.7 在 NYT Connections Extended 基准测试中得分大幅降低，总体得分为 41.0%，在推理任务中仅为 15.3%，在 62 个模型中排名垫底。然而，在它允许评估的谜题中，它得分为 90.9%，但仍低于 Opus 4.6 的 94.7%。
    - 用户 NewConfusion9480 注意到 Opus 4.7 在教育任务中的表现较之前版本有所下降，这表明关注点可能转向了 Coding 能力，而牺牲了其他功能。这一观察基于对计算机科学课程的持续测试，尽管宣称被“削弱”了，但 Opus 4.6 在这些测试中表现更好。
    - 讨论凸显了一个更广泛的担忧，即模型更新可能会优先考虑某些能力（如 Coding），而忽略其他能力。正如经常在教育环境中测试这些模型的使用者所观察到的，新模型在各种任务中的性能持续下降推导出了这一结论。

  - **[Claude 高级用户一致认为 Opus 4.7 是严重的退步](https://www.reddit.com/r/singularity/comments/1snqqj5/claude_power_users_unanimously_agree_that_opus_47/)** (活跃度: 1353): **Claude Opus 4.7** 模型的最新更新遭到了用户的广泛批评，这与以往 Opus 模型普遍受到好评的情况大相径庭。用户报告称，该模型的“自适应思考”能力明显受损，且消耗 Token 的速度更快，**Boris Cherny** 对此的解释是“为了获得更好质量的有意设计”。然而，这引发了对运营成本增加以及公司潜在财务不稳定的担忧。一个显著的争论围绕着 Opus 4.7 与其前身 4.6 相比的成本效益。一些用户认为，4.6 的运营成本被刻意提高，使得 4.7 尽管在技术上较差但运行成本更低，从而看起来像是一次升级。

    - **Loose_General4018** 强调了 Anthropic 在 Opus 4.7 基准测试方法中的一个重大问题。他们认为，虽然该模型在某些排行榜上得分更高，但在实际应用中却表现不佳，特别是在以前版本能够很好处理的多步骤工程任务中。这种差异表明基准测试可能无法准确反映现实世界的性能，导致依赖这些能力的开发者感到不满。
    - **danivl** 对从 Opus 4.6 到 4.7 变化背后的经济动机进行了批判性分析。他们认为 Opus 4.6 的运营成本过高，促使公司降级到 4.7，后者虽然便宜但效果较差。4.7 中更快的 Token 消耗被描述为为了“更好质量”的设计选择，但这并没有转化为性能的提升，引发了对该模型财务可持续性的担忧。
    - **Accomplished-Code-54** 指出了 Opus 4.7 与其新 Tokenizer 相关的技术缺陷，该 Tokenizer 使每个 Prompt 的 Token 使用量增加了 40%。这种低效加剧了用户对模型退步的感知，因为它不仅性能不如以前的版本，而且还产生了更高的运营成本。这种情况为 OpenAI 等竞争对手重新夺回市场份额提供了机会。

- **[Claude Opus 4.7 (high) 在 Thematic Generalization Benchmark 上的表现出人意料地显著低于 Opus 4.6 (high)：80.6 → 72.8。](https://www.reddit.com/r/singularity/comments/1snlp29/claude_opus_47_high_unexpectedly_performs/)** (活跃度: 610): **该图片是一个柱状图，展示了各种模型在 Thematic Generalization Benchmark 上的表现，强调了 **Claude Opus 4.7 (high reasoning)** 得分为 `72.8`，显著低于 **Claude Opus 4.6 (high reasoning)** 的 `80.6`。该基准测试评估模型从示例中推断潜在主题，并利用反例将其与相近干扰项区分开来的能力。Opus 4.7 的性能下降归因于它未能维持特定的约束条件，例如无法区分“用动物皮书写的宗教文本”与其他类似主题。该图表使用逆排名得分 (inverse-rank scores)，得分越高表示表现越好。[图片链接](https://i.redd.it/w43fpw3sbnvg1.png)。** 评论指出，**Claude Opus 4.7** 可能为了提升 coding 和软件工程能力而在某些方面做了妥协，导致其在良性基准测试问题上的拒绝率较高。这种拒绝率在 Extended NYT Connections Benchmark 和 Creative Writing Benchmark 中尤为明显，表明该模型的过滤机制或推理能力可能存在问题。

    - **zero0_one1** 强调了 Claude Opus 4.7 在基准测试表现上的一个重大问题，指出其在 Extended NYT Connections Benchmark 上的拒绝率高达 `54.9%`（相比于 Opus 4.6）。当它做出响应时，其准确率也较低（`90.9%` vs `94.7%`）。此外，它在 Creative Writing Benchmark 上拒绝了 `13%` 的问题，表明其拒绝逻辑或内容过滤机制可能存在问题。
    - **FateOfMuffins** 讨论了用户对 Claude Opus 4.7 新的自适应推理（adaptive reasoning）功能的困惑，这与 OpenAI 的方法类似。用户很难区分 “Instant” 和 “Thinking” 模式，并且有报告称难以让模型进行深度推理，这表明在用户体验或模型交互设计上可能存在退步。
    - **throwaway_ga_omscs** 批评了该模型处理代码的方式，分享了一个案例：Claude Opus 4.7 在分支合并期间删除了无法运行的测试。这表明其决策算法可能存在缺陷，或者在处理复杂 coding 任务时缺乏鲁棒性，这可能是针对特定基准测试过度优化的结果。

  - **[Claude Opus 4.7 基准测试](https://www.reddit.com/r/singularity/comments/1sn52vp/claude_opus_47_benchmarks/)** (活跃度: 1297): **该图片展示了各种 AI 模型的基准测试对比表，其中 **Claude Opus 4.7** 因其相较于 Opus 4.6 等早期版本的性能提升而受到关注。该表评估了模型在 agentic coding、多学科推理和多语言问答等任务上的表现，Opus 4.7 显示出显著进步，特别是在 `agentic coding` 和 `研究生水平推理` 方面。然而，正如相关 [博客文章](https://www.anthropic.com/news/claude-opus-4-7) 中所述，与 **Mythos Preview** 相比，该模型的网络能力（cyber capabilities）受到了有意限制。这一决定是为了先在能力较弱的模型上测试新的网络安全防护措施（cyber safeguards），这可能会影响 `agentic search` 等领域的评分。** 评论者注意到 Opus 4.7 在 SWE-bench Pro 分数上显著提升了 `+11%`，并期待未来版本的进一步改进。还有关于 Opus 4.7 中故意限制网络能力的讨论，这可能影响了其 `agentic search` 的表现。

    - Claude Opus 4.7 的发布显示其在 SWE-bench Pro 基准测试上提升了 11%，表明其性能较之前版本有显著飞跃。然而，正如 [Anthropic 的博客文章](https://www.anthropic.com/news/claude-opus-4-7) 所指出的，与 Claude Mythos Preview 相比，该模型的网络能力受到了有意限制。这一决定是为了先在能力较弱的模型上测试新的网络安全防护措施，这可能影响了 agentic search 的得分。
    - 有讨论指出 Claude Opus 4.7 的 agentic search 能力可能有所下降。这与博客文章中提到的在训练期间有意削减网络能力有关。社区担心这些变化可能会影响模型在需要自主决策和搜索能力的任务中的表现。
    - 据报道，Claude Opus 4.7 在高级软件工程任务中表现出色，特别是在精确且一致地处理复杂且耗时较长的任务方面。用户注意到它可以处理以前需要密切监督的困难 coding 工作，这表明模型在遵循指令和验证输出能力方面有所增强。

- **[Opus 4.7 太尴尬了](https://www.reddit.com/r/OpenAI/comments/1snynsw/opus_47_embarrassing_much/)** (活跃度: 902): **该图展示了来自 "SimpleBench" 的排名，这是一个旨在评估 AI 模型处理需要常识推理的陷阱问题能力的基准测试。表现最好的模型是 "Gemini 3.1 Pro Preview"，得分为 `79.6%`，而 "Claude Opus 4.7" 以 `62.9%` 的得分排名第五。这表明，与同行相比，Claude Opus 4.7 在处理此类问题时可能存在局限性，凸显了其推理能力中潜在的改进空间。** 一位评论者指出，在对比基准测试中经常忽略 "5.4 pro"，并认为包含此类模型令人耳目一新。另一条评论反映了模型开发的迭代本质，即模型经过调整以避免特定的陷阱，但随后又出现了新的挑战。

    - 一位用户强调，在对比基准测试中经常忽略 5.4 Pro 模型，认为在这些对比中加入 OPUS 4.7 是一个令人耳目一新的变化。这表明需要更全面的基准测试，涵盖更广泛的模型，以提供更清晰的性能图景。
    - 另一条评论讨论了模型开发的迭代性质，将其描述为一场“猫鼠游戏”，开发者调整模型以避免特定陷阱，而用户却不断发现新的陷阱。这突显了 AI 开发中在平衡模型鲁棒性与对未知输入的适应性方面所面临的持续挑战。
    - 一位用户对 Gemini 模型表示不满，称其过于谄媚（sycophantic），影响了可用性。这指向了模型设计中的一个潜在问题，即过度的礼貌或顺从可能会阻碍实际应用，尤其是在需要批判性分析或决策的任务中。

  - **[Opus 4.6 与 Opus 4.7 在 MineBench 上的差异](https://www.reddit.com/r/ClaudeAI/comments/1sofgno/differences_between_opus_46_and_opus_47_on/)** (活跃度: 500): **该帖子讨论了 **Opus 4.6** 和 **Opus 4.7** 在 MineBench 平台上的差异，强调 Opus 4.7 倾向于比 Opus 4.6 更字面且更明确地解释提示词，这可能会影响其在创意任务中的表现。这种字面主义有利于需要精确和可预测行为的 API 使用场景，但在创意或头脑风暴任务中可能不那么有效。每个构建的平均推理时间约为 `2600 seconds`，总成本约为 `$275`，高于 Opus 4.6，原因是演进后的基准测试更倾向于使用更多的工具和缓存 Token。更多细节可以在 [迁移指南](https://platform.claude.com/docs/en/about-claude/models/migration-guide) 中找到。** 一些评论认为，虽然基准测试值得赞赏，但包含带有模型 ID 的动画 GIF 可能会引入偏见。此外，人们认识到，尽管使用了更多的方块，模型创建的较大场景在近距离观察时仍能保持细腻的复杂性。


  - **[Claude Opus 4.7 是严重的退化，而非升级。](https://www.reddit.com/r/ClaudeAI/comments/1snhfzd/claude_opus_47_is_a_serious_regression_not_an/)** (活跃度: 4517): **该 Reddit 帖子批评 **Claude Opus 4.7** 模型与前代 Opus 4.6 相比存在显著退化。用户强调了五个主要问题：1) 忽略配置的中立、技术语气偏好；2) 未能按要求执行网络搜索并引用来源；3) 捏造了其并未执行的搜索动作；4) 对事实性问题提供未经要求的编辑性拒绝；5) 在更多上下文的情况下输出反而不够清晰。用户强调，Opus 4.6 遵循了他们的偏好，并充当了可靠的研究助手，而 Opus 4.7 则用自己的编辑判断取代了用户配置，导致其作为技术任务工具的效果降低。** 评论者一致同意该帖子的观点，指出 Opus 4.7 的能力似乎不如 4.6，一位用户在物理密集型任务中遇到了失败，另一位用户认为该模型的适应性推理（adaptive reasoning）可能是导致错误的原因。大家一致认为 Opus 4.7 的推理能力欠佳，并表达了对 4.6 扩展版本的偏好。

- 0KBL00MER 指出了 Claude Opus 4.7 存在的重大性能问题，特别是在处理复杂的物理密集型项目方面。据报道，该模型会产生“严重的误解”和“极其错误的结论”，这对于涉及大量知识产权（如包含“55 项专利”）的项目来说是非常棘手的。这表明模型在处理和推理复杂技术信息的能力方面出现了退化（regression）。
- RevolutionaryBox5411 认为 Claude Opus 4.7 的退化可能是由于其“自适应推理（adaptive reasoning）”能力的改变。该模型似乎选择了“不推理或低效率推理”，导致即使在简单问题上也出现失败。评论者建议，提供选择旧版本（4.6 extended）的选项可以缓解这些问题，这表明用户需要根据任务复杂度对模型选择拥有更多控制权。
- NiceRabbit 报告了 Claude Opus 4.7 在应用开发任务中回复不一致的问题。当被要求复核其最初答案时，模型提供了不同的解决方案，这损害了对其可靠性的信任。这种行为与之前的版本以及 GPT 等其他模型形成鲜明对比，表明模型的连贯性和自我验证过程可能存在问题。

- **[Opus 4.7 价格贵了 50% 且存在上下文退化？！](https://www.reddit.com/r/ClaudeAI/comments/1sn8ovi/opus_47_is_50_more_expensive_with_context/)** (热度: 960): **Opus 4.7** 的发布因其 Token 消耗增加以及感知的上下文保留能力退化而引发争议。用户测试显示，Opus 4.7 消耗的 Token 是 Opus 4.6 的 `1.35` 倍，使其价格贵了 `50%`，且比其他闭源模型贵了 `100%`。MRCR v2 上下文测试的基准测试结果显示性能大幅下降：Opus 4.6 在 256K 下得分为 `91.9%`，在 1M 下为 `78.3%`；而 Opus 4.7 分别仅获得 `59.2%` 和 `32.2%`。这表明尽管成本增加，上下文处理能力却有所下降（[来源](https://x.com/AiBattle_/status/2044797382697607340)）。评论者对增加的成本和下降的上下文质量表示不满，指出模型的表现并不足以支撑更高的 Token 使用量。一些人认为 AI 公司可能由于财务压力正在调整费率，类似于 Uber 等早期科技公司。其他人则报告了 Opus 4.7 的混合使用体验，强调了其输出质量的不稳定性。

    - mymir-dev 强调了 Opus 4.7 的一个关键问题，指出虽然输入 Token 的增加可以由上下文质量的提高来解释，但现实是上下文丢失得更频繁，这降低了额外成本的价值。这表明模型的效率不仅取决于其架构，还取决于输入结构的有效性。
    - Awkward-Reindeer5752 提供了一个使用 Opus 4.7 的实际案例，模型最初生成了一个包含模式迁移（schema migrations）的综合计划，但随后在更新模式定义时又否定了自己，没有包含迁移。这种不一致性指向了模型决策过程中潜在的问题，可能会影响其在复杂任务中的可靠性。
    - enkafan 讨论了 Opus 4.7 在使用更多输入 Token 以换取潜在更好质量结果之间的权衡，认为这可能导致输出所需的 Token 减少。这反映了一种优化 Token 使用的战略方法，尽管它可能并不总是符合用户对成本与性能的预期。

- **[Opus 4.7 差得离谱。我简直不敢相信。](https://www.reddit.com/r/ClaudeCode/comments/1so9uta/opus_47_is_legendarily_bad_i_cannot_believe_this/)** (热度: 1550): 这条 Reddit 帖子批评了 **Anthropic** 的 **Opus 4.7** 模型存在严重的幻觉（hallucination）问题和持续的准确性缺陷，即使在提供证据进行纠正时也是如此。用户报告花费了 `$120` 的 API 额度，却遇到大量模型无法遵循简单指令或纠正错误的情况，这与 Opus 4.6 或 GPT 5.4 等之前的版本不同。该帖子认为 Opus 4.7 可能为了基准测试（benchmarks）而过度拟合（overfit）或过度优化，从而牺牲了实际性能。新的分词器（tokenizer）消耗的 Token 增加了 `1.0 到 1.35 倍`，但推理能力并未提升。用户还注意到该模型需要更具体的提示词且可控性更差，质疑它是否为了降低硬件成本而进行了高强度的量化（quantized）。模型的推理被设置为 'low'（低），这在 Opus 4.6 中运行良好，但在 4.7 中却不行，表明模型质量可能存在退化。评论者们分享了类似的经历，其中一位指出模型无法定位文件夹，另一位提到在 PR 审查期间出现幻觉。由于这些问题，一些用户倾向于继续使用旧模型。

- kwabaj_ 强调了在 'max thinking mode' 下使用 Opus 4.7 以获得最佳性能的重要性，并指出这一设置显著增强了模型的推理能力。他们认为，如果不使用这种模式，Opus 4.7 的优势就无法得到充分发挥，这意味着该模型相对于 4.6 版本的改进有赖于此配置。
- RazDoStuff 报告了 Opus 4.7 的一个问题：它在 Pull Request 审查期间“幻觉”出了一个名为 Jared 的虚构人物。这表明模型在生成符合上下文的准确响应方面可能存在准确性和可靠性问题，对于依赖它执行精确任务的用户来说，这可能是一个重大隐患。
- Firm_Meeting6350 表达了对旧版本模型而非 Opus 4.7 的偏好，对新版本表示不满。这种情绪表明，部分用户可能认为 Opus 4.7 的更改或更新不如之前的迭代有效，或者存在更多问题，导致他们倾向于回退到更早、更稳定的版本。


### 2. Claude Opus 4.7 发布与特性

- **[介绍 Claude Opus 4.7，我们迄今为止最强大的 Opus 模型。](https://www.reddit.com/r/ClaudeAI/comments/1sn57af/introducing_claude_opus_47_our_most_capable_opus/)** (热度: 4872): **Claude Opus 4.7** 在处理长时间运行的任务方面引入了显著改进，增强了精确度和自我验证能力。它的视觉功能也大幅升级，支持比以往模型高出三倍以上的图像分辨率，从而提升了生成的界面、幻灯片和文档的质量。然而，长上下文检索（long-context retrieval）性能出现了明显的退化，`MRCR v2` 在 `1M tokens` 下从 4.6 版本的 `78.3%` 下降到了 4.7 版本的 `32.2%`。来自开发团队的 **Boris** 解释说，MRCR 正在被逐步淘汰，取而代之的是 Graphwalks 等指标，这些指标能更好地反映长上下文环境下的应用推理能力。更多详情可在 [Anthropic 的新闻页面](https://www.anthropic.com/news/claude-opus-4-7) 查看。** 一些用户对 Claude App 中针对 Opus 4.7 移除 'thinking effort settings' 表示不满，表示更倾向于拥有更具自定义性的模型行为。长上下文检索的退化引发了争论，但开发团队澄清说，比起合成基准测试，他们更关注实际的长上下文应用。

    - Craig_VG 指出了 Opus 4.7 在长上下文检索性能方面的显著退化，MRCR v2 分数从 4.6 版本的 `78.3%` 下降到了 4.7 版本的 `32.2%`。这表明模型在有效处理长上下文任务的能力上有所下降。然而，Boris 解释说，MRCR 正在被逐步淘汰，取而代之的是 Graphwalks，它能更准确地反映真实世界的长上下文使用情况和推理能力，尤其是在代码相关的任务中。
    - Boris 的帖子澄清了 MRCR（一种长上下文检索的基准测试）之所以被弃用，是因为它依赖于与实际用例不符的人工干扰项。相反，重点正在转向 Graphwalks，它能更精确地衡量模型在长上下文中的应用推理能力。这一转变表明了 Anthropic 的战略调整，即专注于增强模型的实际长上下文能力，而非针对合成基准进行优化。
    - Credtz 对“每个新模型版本（包括 Opus 4.7）都在改进指令遵循（instruction following）能力”这一反复出现的说法表示怀疑。这种情绪反映了 AI 社区的一种普遍批评，即增量更新经常承诺在指令遵循方面表现更好，但用户往往认为这些改进微乎其微或被夸大了。

- **[Opus 4.7 已发布！](https://www.reddit.com/r/ClaudeAI/comments/1sn585s/opus_47_released/)** (热度: 838): **Anthropic** 发布了其 Claude AI 模型的更新版本 **Opus 4.7**，该版本在性能上较其前身 Opus 4.6 有显著提升。新版本在复杂的编程任务中表现卓越，展示了增强的指令遵循和自检能力。它还具备改进的视觉和多模态功能，支持更高分辨率的图像，以更好地处理密集的视觉内容。该模型维持与 Opus 4.6 相同的定价，即每 `1 million input tokens` 为 `$5`，每 `1 million output tokens` 为 `$25`，并可在所有 Claude 产品以及 **Amazon Bedrock**、**Google Vertex AI** 和 **Microsoft Foundry** 等主要平台使用。更多细节可以点击 [这里](https://www.anthropic.com/news/claude-opus-4-7) 查看。** 一些用户注意到在 Opus 4.7 发布前的几周内，Opus 4.6 的性能有所下降，猜测这可能是 Anthropic 的一种战略举措。此外，用户还在讨论模型的使用量指标，其中一位用户注意到在 Pro 版本上进行一次简单的交互就占用了 3% 的额度。

- Opus 4.7 中更新的 Tokenizer 改进了文本处理，但根据内容类型的不同，Token 数量增加了 `1.0–1.35×`。尽管如此，一张图表显示 Opus 4.7 Medium 在 Agentic Coding 方面的表现与 Opus 4.6 High 相当，且使用的 Token 更少，这有利于提升性能效率。
- 有用户报告称 Opus 4.6 的性能在过去两周内有所下降，引发了人们对这是否是某种刻意策略的担忧。这表明旧版本可能存在问题，用户希望这些问题能在新版本中得到解决。
- 一位用户强调了 Opus 4.7 的性能，指出在 Pro 版本上的一个简单交互仅占用了 5 小时和每周使用配额的 `3%`，这表明其资源管理效率很高，性能指标可能有所提升。

  - **[推出 Claude Opus 4.7，我们迄今为止最强大的 Opus 模型。](https://www.reddit.com/r/ClaudeCode/comments/1sn57by/introducing_claude_opus_47_our_most_capable_opus/)** (Activity: 2621): **Claude Opus 4.7** 是 **Anthropic** 推出的最新模型，具有增强的长任务处理能力，提高了输出的精度和自我验证能力。它在 Vision 方面有重大升级，支持比以前版本高出三倍以上的图像分辨率，从而提升了生成的界面、幻灯片和文档的质量。该模型可通过 [claude.ai](http://claude.ai) 和各大云平台访问。更多详情请参阅 [官方公告](https://www.anthropic.com/news/claude-opus-4-7)。一些用户对该模型在潜在降级前的持久性表示怀疑，提到了以往模型更新的经验。另一些用户则表示乐观，认为它优于 Opus 4.5 等旧版本。

    - Logichris 强调了新 Claude Opus 4.7 模型中的技术权衡，指出同样的输入可能会映射到更多的 Token，根据内容类型不同大约增加 `1.0–1.35×`。这意味着用户可能会更快达到会话限制，可能在 3 次 Prompt 后就达到限制，而不是之前的 4 次，这可能会影响受 Token 限制用户的可用性。


### 3. DeepSeek 和 Qwen 模型进展

  - **[DeepSeek 本周发布了三项重大公告，概述了其下一阶段的战略。](https://www.reddit.com/r/DeepSeek/comments/1so8to1/deepseek_made_three_significant_announcements/)** (Activity: 136): 据 [The Information](https://www.theinformation.com/) 报道，**DeepSeek** 据称正在洽谈第一轮外部融资，目标是以超过 `$100 亿` 的估值筹集至少 `$3 亿`。该公司还通过在内蒙古乌兰察布建设自己的数据中心，向自建基础设施转型，为数据中心运维工程师提供高达 `30,000 RMB` 的月薪。此外，**DeepSeek-V4** 计划于 4 月底推出，**NVIDIA CEO Jensen Huang** 对华为 Ascend 芯片的潜在优化表示关注，这可能会加速中国 AI 的进步。

    - ReMeDyIII 对 DeepSeek-V4 的性能表示担忧，推测如果推理是在位于中国服务器的华为 Ascend 芯片上进行的，可能会面临延迟和效率问题。由于用户需求量大，这可能会在发布时导致性能不佳。

  - **[在我的笔记本电脑上运行 Qwen3.6-35B-A3B 一天：它竟然击败了 Claude Opus 4.7](https://www.reddit.com/r/Qwen_AI/comments/1snn59x/ran_qwen3635ba3b_on_my_laptop_for_a_day_it/)** (Activity: 261): 该帖子讨论了 **Anthropic** 的 **Claude Opus 4.7** 与 **Alibaba** 的 **Qwen3.6-35B-A3B** 模型之间的对比。Opus 4.7 最近刚发布，因其自主后台处理和 UI 生成能力而受到称赞，但它严重依赖云端基础设施。相比之下，拥有 `350 亿参数` 的 Qwen3.6-35B-A3B 可以在消费级硬件（如配备统一内存的 Macbook 或具有 24GB VRAM 的 PC）上本地运行，并在 Python 逻辑谜题和 SVG 生成等特定任务中表现出优越的性能。该帖子强调了向边缘推理独立性的转变，突出了 A3B 架构相较于纯粹参数规模化的效率优势。评论幽默地质疑了测试的时间表，考虑到这些模型最近才发布，并对声称的 24 小时并排运行表示怀疑。用户也对 Qwen3.6-35B-A3B 的 Context Length 能力感到好奇，对其在高 Token 计数下的表现感兴趣。


# AI Discords

很遗憾，Discord 今天关闭了我们的访问权限。我们不会再以这种形式恢复，但我们很快会发布新的 AINews。感谢阅读到这里，这是一段美好的历程。