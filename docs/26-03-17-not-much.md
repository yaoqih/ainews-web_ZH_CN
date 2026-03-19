---
companies:
- openai
- langchain
- stripe
- ramp
- coinbase
- nous-research
- hermes-agent
date: '2026-03-16T05:44:39.731046Z'
description: '**OpenAI** 发布了 **GPT-5.4 mini** 和 **GPT-5.4 nano**，这是其功能最强大的小型模型，针对编程、多模态理解和子智能体（subagents）进行了优化。这些模型具备
  **400k 上下文窗口**，且运行速度比 GPT-5 mini 提升了 **2 倍以上**。


  该 mini 模型在性能上接近规格更大的 GPT-5.4，但仅消耗 **30% 的 Codex 配额**，因此已成为许多编程工作流的默认选择。虽然性能强劲，但定价担忧和真实性权衡（truthfulness
  tradeoffs）也受到了关注，第三方评估对其推理能力和识别虚假前提（false premises）的能力给出了褒贬不一的评价。此外，OpenAI 在最近的一次更新中解决了行为微调（behavior
  tuning）问题。


  与此同时，智能体基础设施正随着安全代码执行和编排工具（如 **LangChain 的 LangSmith Sandboxes** 和 **Open SWE**）而不断演进，这些工具的灵感源自
  **Stripe、Ramp 和 Coinbase** 的内部系统。子智能体和安全执行现已成为关键的产品特性，例如 **Hermes Agent v0.3.0**
  的发布展示了插件架构、实时 Chrome 浏览器控制及语音模式。在研究领域，关于注意力机制的研究（包括 **Attention Residuals** 和垂直注意力）也正受到越来越多的关注。'
id: MjAyNS0x
models:
- gpt-5.4-mini
- gpt-5.4-nano
- gpt-5.4
- codex
people:
- hwchase17
- michpokrass
title: 今天没发生什么事。
topics:
- coding
- multimodality
- subagents
- context-window
- model-performance
- pricing
- behavior-tuning
- secure-execution
- plugin-architecture
- attention-mechanisms
- agent-infrastructure
---

**平静的一天。**

> 2026年3月14日至3月16日的 AI News。我们检查了 12 个 subreddits、[544 个 Twitters](https://twitter.com/i/lists/1585430245762441216)，没有检查更多的 Discords。[AINews 网站](https://news.smol.ai/) 允许你搜索所有往期内容。提醒一下，[AINews 现在是 Latent Space 的一个板块](https://www.latent.space/p/2026)。你可以[选择加入/退出](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)邮件接收频率！

---

# AI Twitter 综述


**OpenAI 发布 GPT-5.4 Mini/Nano 以及向小型化、针对 Coding 优化的模型的转向**

- **GPT-5.4 mini 和 nano 已在 API、ChatGPT 和 Codex 中推出**：OpenAI 发布了 [**GPT-5.4 mini**](https://x.com/OpenAI/status/2033953592424731072) 和 [**GPT-5.4 nano**](https://x.com/OpenAI/status/2033953595637538849)，将其定位为迄今为止能力最强的小型模型。根据 [@OpenAIDevs](https://x.com/OpenAIDevs/status/2033953815834333608) 的说法，GPT-5.4 mini 的速度比 GPT-5 mini **快 2 倍以上**，目标是 **coding、计算机使用、多模态理解和 subagents**，并在 API 中提供 **400k context window**。OpenAI 还声称 mini 在包括 [**SWE-Bench Pro 和 OSWorld-Verified**](https://x.com/OpenAIDevs/status/2033953828387885470) 在内的评估中接近更大规模 GPT-5.4 的性能，而仅使用 [**30% 的 GPT-5.4 Codex 配额**](https://x.com/OpenAIDevs/status/2033953840312291603)，使其成为许多后台 coding 工作流和 subagent 分发的新默认选择。
- **早期反馈集中在 Coding 价值，但也关注价格和真实性的权衡**：开发者立即强调了 mini 在 [Codex 中的 subagents](https://x.com/dkundel/status/2033953901301665838)、[计算机使用工作负载](https://x.com/scaling01/status/2033954794105127007)以及 [Windsurf](https://x.com/windsurf/status/2033954998837776869) 等外部产品中的效用。然而，评论也汇聚到了一个熟悉的 OpenAI 模式：性能更好，但价格更高。[@scaling01](https://x.com/scaling01/status/2033955279079907511) 的帖子指出，mini 的价格为 **输入 $0.75/M，输出 $4.5/M**，nano 的定价也同样高于之前的 nano 层级。第三方评估结果喜忧参半：[Mercor 的 APEX-Agents 结果](https://x.com/mercor_ai/status/2033955468650156503)显示，mini 在 xhigh reasoning 下的 **Pass@1 为 24.5%**，在该基准测试中领先于一些轻量级和中量级竞争对手；而 [BullshitBench](https://x.com/petergostev/status/2033995459522396287) 将这些新的小型模型在抵御虚假前提/术语陷阱方面的排名排得相对较低。OpenAI 还悄悄承认了行为微调问题，[@michpokrass](https://x.com/michpokrass/status/2033935238066540806) 表示最近的 **5.3 instant** 更新减少了“令人恼火的点击诱导”行为。

**Agent 基础设施：Sandboxes、Subagents、Open SWE 以及测试框架之争**

- **执行代码的 Agent 正在成为产品架构的核心**：多次发布表明，技术栈正围绕安全的执行、编排和部署易用性而成熟，而不仅仅是更好的基础模型。LangChain 推出了用于安全临时代码执行的 [**LangSmith Sandboxes**](https://x.com/LangChain/status/2033949251529793978)，[@hwchase17](https://x.com/hwchase17/status/2033950657619874217) 明确指出“越来越多的 Agent 将编写并执行代码”。与此同时，LangChain 开源了 [**Open SWE**](https://x.com/hwchase17/status/2033977192053612621)，这是一个模仿 **Stripe、Ramp 和 Coinbase** 内部系统的后台 coding Agent。该系统与 [Slack、Linear 和 GitHub](https://x.com/BraceSproul/status/2033962118970818650) 集成，使用 subagents 加中间件，并将测试框架（harness）、沙箱（sandbox）、调用层（invocation layer）和验证层分离。这是从“聊天辅助工具（chat copilots）”向可部署的内部工程 Agent 迈出的重要一步。
- **Subagents 和安全执行现在是整个生态系统中的一级产品功能**：OpenAI 的 Codex 现在支持 [**subagents**](https://x.com/gdb/status/2033757784437895367)，而 GPT-5.4 mini 被 OpenAI 描述为特别适用于该用例。Hermes Agent 的 [**v0.3.0** 版本发布](https://x.com/NousResearch/status/2033877040399831478)是另一个强烈的信号：**5 天内提交了 248 个 PR**，具备一流的 **plugin architecture**，通过 **CDP** 进行实时 Chrome 控制，集成 IDE，支持基于 Whisper 的本地语音模式，PII 脱敏，以及像 [Browser Use](https://x.com/Teknium/status/2033811117521408078) 这样的供应商集成。由此产生的发展方向在各厂商之间是一致的：Agent 的价值越来越取决于安全的执行环境、可组合的技能/插件以及工作流原生界面，而不仅仅是纯粹的基准测试收益。

**架构研究：Attention Residuals、Vertical Attention 以及 Mamba-3**

- **Attention over depth 正在受到关注**：Moonshot 的 [**Attention Residuals 论文**](https://x.com/Kimi_Moonshot/status/2033796781327454686) 引发了围绕 “垂直注意力”（vertical attention）或跨层注意力的实质性技术讨论。来自 [@ZhihuFrontier](https://x.com/ZhihuFrontier/status/2033751367198949865) 的详细解析将该想法框定为每一层查询前一层状态，有效地将注意力从水平序列交互扩展到层间记忆。社区反应强调这并非完全孤立：[@rosinality](https://x.com/rosinality/status/2033810580604158323) 指出 **ByteDance 也实现了 attention over depth**，并且 [@arjunkocher](https://x.com/arjunkocher/status/2033846693918347641) 发布了实现指南。这里有趣的系统论点是，由于 **层数 << 序列长度**，某些形式的垂直注意力可能隐藏在现有计算之下，几乎不会带来额外延迟。
- **Mamba-3 强化了推理优先混合架构的地位**：另一个主要的架构发布是 [**Mamba-3**](https://x.com/_albertgu/status/2033948415139451045)，由 [@_albertgu](https://x.com/_albertgu/status/2033948415139451045) 和 [@tri_dao](https://x.com/tri_dao/status/2033948569502413245) 展示，作为在混合时代使线性/状态空间模型更具竞争力的最新举措。重点明确放在 **推理效率** 上，而不是直接取代 Transformer。Together 将其总结为一个 [**MIMO 变体**](https://x.com/togethercompute/status/2033956365165859026)，在相似的解码速度下提高了模型强度，并声称在线性模型中性能最强，且在 **1.5B** 规模下预填充（prefill）和解码速度最快。Tri Dao 还指出，推理密集型的 RL 和长序列（long-rollout）工作负载是此类架构尤其肥沃的土壤。从 Attention Residuals 和 Mamba-3 中得出的更广泛结论是，实验室仍在寻找在不牺牲过多生态系统兼容性的情况下，缓解全 Transformer 瓶颈的方法。

**GTC：NVIDIA 的 Agent 推动、开源模型与基础设施论题**

- **GTC 的传讯集中在推理、Agent 和 “Token 工厂” 的世界观**：多条推文反映了 Jensen Huang 将未来计算机定义为[“**制造 Token 的系统**”](https://x.com/TheTuringPost/status/2033983885131059636)，推理现在正驱动着下一波产能浪潮。这体现在产品和生态系统的发布中：LangChain 表示其框架下载量突破 [**10 亿次**](https://x.com/LangChain/status/2033788913937195132) 并加入了 **NVIDIA Nemotron 联盟**；[@ggerganov](https://x.com/ggerganov/status/2033947673825337477) 强调了 llama.cpp 对 **Nemotron 3 Nano 4B** 的支持；Hugging Face 的 [@jeffboudier](https://x.com/jeffboudier/status/2033959279510884631) 回顾了一系列开源的 NVIDIA 发布，涵盖推理模型、机器人数据集和世界模型（world models）。
- **开源和企业级 Agent 工具占据了周边公告的主导地位**：H Company 发布了 [**Holotron-12B**](https://x.com/hcompany_ai/status/2033851052714320083)，这是一款与 NVIDIA 共同构建的开源多模态模型，用于 **计算机操作 Agent（computer-use agents）**。Perplexity 发布了 [**Comet Enterprise**](https://x.com/perplexity_ai/status/2033947232467357874)，将其 AI 浏览器带入企业团队，具备发布控制和 [CrowdStrike Falcon 集成](https://x.com/perplexity_ai/status/2033947356551647356)。NVIDIA 更广泛的业务论点也得到了强化：[@TheTuringPost](https://x.com/TheTuringPost/status/2033981870141231215) 强调了 Jensen 的言论，即经常被引用的 **1 万亿美元 AI 基础设施机遇** 仅涵盖了 2027 年前堆栈的一个子集，进一步证实了行业仍处于推理基础设施扩建的早期阶段。

**开源工具、本地 Agent 与开发栈升级**

- **本地/私有 Agent 工作流持续改进**：Hugging Face 发布了一个 [**hf CLI extension**](https://x.com/ClementDelangue/status/2033982183791108278)，可自动检测适用于当前硬件的最佳本地模型/量化版本，并启动一个本地代码 Agent。Unsloth 推出了 [**Unsloth Studio**](https://x.com/UnslothAI/status/2033926272481718523)，这是一个开源 Web UI，可在 Mac/Windows/Linux 上本地训练和运行 **500 多个模型**，声称**训练速度提升 2 倍且节省 70% 的 VRAM**，支持 GGUF、合成数据工具、Tool Calling 和代码执行。Ollama 为 OpenClaw 工作流增加了 [Web 搜索/抓取插件和 headless 启动支持](https://x.com/ollama/status/2033993519459889505)，同时也在 [CodexBar 中作为提供商](https://x.com/ollama/status/2033794815448780803)亮相。
- **“开源代码 Agent”生态系统正变得清晰**：各种模式正趋于融合：模型无关的 Harness、结构化技能、文件系统/状态抽象，以及临时的云端或本地执行。LangChain 的 [Deep Agents](https://x.com/RoundtableSpace/status/2033955271333011829) 被描述为一个采用 MIT 许可证、可检查的 Claude Code 风格 Agent Harness 副本。Hermes Agent 的插件系统和对本地模型的友好性也使其进入了同一讨论范畴。这是数据集中最清晰的趋势之一：前沿领域不再仅仅是开源权重模型，还包括用于实际部署 Agent 的开源 Harness 和运行时层。

**热门推文（按互动率排序）**

- **OpenAI 发布小模型**：[@OpenAIDevs 关于 GPT-5.4 mini/nano](https://x.com/OpenAIDevs/status/2033953815834333608) 的推文是当天最重要的技术公告之一，特别是对于代码 Agent 的工作负载而言。
- **Cursor 基于 RL 的上下文压缩**：[@cursor_ai](https://x.com/cursor_ai/status/2033967614309835069) 表示其训练了 Composer **通过 RL 而非 Prompting 进行自我总结**，将压缩误差降低了 **50%**，并使其能够处理更难的长程代码任务。
- **Mamba-3 发布**：[@_albertgu](https://x.com/_albertgu/status/2033948415139451045) 和 [@tri_dao](https://x.com/tri_dao/status/2033948569502413245) 标志着本周期序列建模中最重要的架构更新之一。
- **Unsloth Studio**：[@UnslothAI](https://x.com/UnslothAI/status/2033926272481718523) 进行了最强有力的开源产品发布之一，直接面向本地训练/推理从业者。
- **Kimi Attention Residuals**：[@Kimi_Moonshot](https://x.com/Kimi_Moonshot/status/2033796781327454686) 引发了大量的架构讨论，随后还出现了关于垂直注意力和层间记忆的后续分析。


---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. Unsloth Studio 发布及其特性

  - **[Unsloth 发布 Unsloth Studio —— LMStudio 的竞争对手？](https://www.reddit.com/r/LocalLLaMA/comments/1rwa0f7/unsloth_announces_unsloth_studio_a_competitor_to/)** (热度: 998): **Unsloth Studio** 已宣布作为一个全新的开源、无代码 Web 界面，用于在本地训练和运行 AI 模型，这可能会挑战 LMStudio 在 GGUF 生态系统中的主导地位。它兼容 `Llama.cpp`，并提供 **auto-healing tool calling**、Python 和 bash 代码执行，以及对 **audio, vision, 和 LLM finetuning** 的支持。该平台支持 GGUFs，可在 Mac, Windows, 和 Linux 上运行，具备 **SVG 渲染**、**synthetic data generation** 和 **fast parallel data preparation** 功能。通过 `pip install unsloth` 即可轻松安装。更多详情可见 [Unsloth 文档](https://unsloth.ai/docs/new/studio#run-models-locally)。一些用户对将 LMStudio 描述为高级用户的“首选”表示质疑，建议使用 vLLM 或 llama.cpp 等替代方案。其他用户则对该 UI 的功能表示兴奋，特别是针对训练和数据准备方面。

    - **danielhanchen** 强调了 Unsloth Studio 丰富的特性集，指出其具备 auto-healing tool calling、Python 和 bash 代码执行等能力，并支持包括 Mac, Windows, 和 Linux 在内的多个操作系统。该工具还提供 SVG 渲染、synthetic data generation 和 fast parallel data preparation 等高级功能，使其成为处理各种 AI 任务的综合解决方案。更多详情和安装说明可在 [GitHub](https://github.com/unslothai/unsloth) 查阅。
    - **sean_hash** 指出了将 finetuning 和 inference 功能集成到 Unsloth Studio 这一个工具中的便利性。这与目前需要使用多个项目才能实现相同功能的情况形成对比，突显了 Unsloth Studio 优化 AI 开发工作流的潜力。
    - **Specter_Origin** 对 Unsloth Studio 的开源性质表示赞赏，将其与闭源的 LM Studio 进行了对比。这种开放性对于偏好透明度并希望根据需求修改工具的开发者来说，可能是一个显著优势。

  - **[推介 Unsloth Studio：一个用于训练和运行 LLM 的全新开源 Web UI](https://www.reddit.com/r/LocalLLaMA/comments/1rw9jmf/introducing_unsloth_studio_a_new_opensource_web/)** (热度: 579): **Unsloth Studio** 是一款全新的开源 Web UI，旨在 **Mac, Windows, 和 Linux** 上本地训练和运行 LLM。它声称能以两倍的速度训练 `500+` 个模型，同时减少 `70% less VRAM` 使用。该平台支持 **GGUF**、vision, audio, 和 embedding 模型，并包含模型对比、self-healing tool calling 和网页搜索等功能。它还支持从 **PDF, CSV, 和 DOCX** 等格式自动创建数据集，并允许执行代码以增强 LLM 输出的准确性。模型可以导出为 GGUF 和 Safetensors 等格式，并支持 inference 参数的自动调优。通过 `pip install unsloth` 进行安装。[GitHub](https://github.com/unslothai/unsloth) 和 [文档](https://unsloth.ai/docs/new/studio) 已提供更多细节。评论者对 Unsloth Studio 作为一个现有平台的完全开源替代方案感到兴奋，强调了它在 finetuning 模型方面的易用性，特别是对于专业知识较少的用户。人们对即将到来的 AMD 支持充满期待，预计这将扩大其可用性。

    - 一位用户强调了降低 finetuning 门槛的重要性，指出 Unsloth Studio 提供了一种简便的 finetuning 模型方式，而这自 LLaMA 2 发布以来一直是个挑战。这种易用性可能会复兴“微调的黄金时代”，让缺乏专业知识的人也能更容易地参与模型定制。
    - 另一位用户指出在安装过程中遇到了技术问题，在下载大型 `torch` 包时因磁盘空间不足导致 OSError。这突显了 AI/ML 项目中管理依赖项和系统资源的常见挑战，表明可能需要组件的原子化安装来降低准入门槛。
    - 一位 AMD 代表表示已准备好支持即将发布的 Unsloth Studio 官方 AMD 支持，这预示着 AMD 硬件用户的兼容性和性能可能会得到提升。这种合作可能会增强 Unsloth Studio 在不同硬件平台上的可用性。

### 2. Qwen3.5-9B 文档基准测试结果

  - **[Qwen3.5-9B 在文档基准测试上的表现：哪些地方超越了前沿模型，哪些地方没有。](https://www.reddit.com/r/LocalLLaMA/comments/1rv98wo/qwen359b_on_document_benchmarks_where_it_beats/)** (热度: 295): **该图片对比了阿里巴巴的 Qwen3.5-9B 和 OpenAI 的 GPT-5.4 在文档 AI 基准测试中的表现。Qwen3.5-9B 以 `77.0` 的评分位列第 9，在“关键信息提取”和“表格理解”方面表现出色，而 GPT-5.4 以 `81.0` 的评分位列第 4，在其他领域领先。基准测试结果突出了 Qwen3.5-9B 在 “OmniOCR” 中的卓越表现，但在 “OmniDoc” 和 “IDP Core” 方面落后。这与帖子中的详细拆解一致，即 Qwen 模型在 OCR 和 VQA 任务中表现优异，但在表格提取和手写 OCR 方面落后。** 一位评论者指出，AI 技术正在达到功能天花板，这表明目前的模型足以胜任许多任务，并能在性能较低的硬件上高效运行。另一条评论期待与 GLM-OCR 的有趣对比，而第三条评论则指出，对于可以容忍较长处理时间的任务，使用较小的 Qwen 模型具有潜在的能效优势。

    - **Qwen3.5-9B 的表现**: 该模型在文档处理任务中表现出与更大型的前沿模型竞争的实力。它在超极本等低端硬件上高效运行的能力，突显了其能效以及在更广泛应用中的可访问性。这表明行业正趋向于针对特定任务优化小型模型，而不是仅仅依赖更大、更耗资源的模型。
    - **能效与推理**: Qwen3.5-9B 模型以其能效著称，尤其是在需要长时间推理的任务中。与 Gemini 或 GPT 等大型模型相比，如果处理时间不是关键因素，Qwen3.5-9B 提供了一个更可持续的选择。这使其成为将能耗视为首要任务的应用场景中的可行替代方案。
    - **模型变体与基准测试**: 人们对基准测试中缺失的大型 Qwen 变体（如 27B dense 和 35B MoE）感到好奇。这种缺失引发了关于这些大型模型在特定任务中的对比表现和潜在优势的疑问，表明需要对这些变体进行进一步的探索和基准测试。



### 3. Mistral Small 4 和 DGX Station 的可用性

  - **[Mistral Small 4:119B-2603](https://www.reddit.com/r/LocalLLaMA/comments/1rvlfbh/mistral_small_4119b2603/)** (热度: 1057): ****Mistral Small 4** 是一款拥有 `1190 亿参数` 和 `256k 上下文长度` 的混合 AI 模型，集成了 Instruct、Reasoning 和 Devstral 能力。它支持多模态输入，并采用高效架构，将延迟降低了 `40%`。该模型包含投机采样（speculative decoding）和 4-bit float 量化等高级特性，针对通用聊天、编程和文档分析等任务进行了优化。它在 Apache 2.0 许可下发布，可用于商业和非商业用途。更多详情可以在 [Hugging Face 页面](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603)查看。** 评论者幽默地提到了规模的变化，现在 `1200 亿参数` 已被视为“小型”，这反映了 AI 模型尺寸和能力的快速演进。

    - Mistral Small 119B 模型正在与 Qwen3.5-122B-A10B 模型进行对比，重点是参数激活。Mistral 激活了 65 亿个参数，而 Qwen3.5 使用了 100 亿个，这也许解释了为什么 Mistral 的整体表现没有超过 Qwen3.5。这突显了参数激活（parameter activation）在模型性能中的重要性。

  - **[DGX Station 已可用（通过 OEM 分销商）](https://www.reddit.com/r/LocalLLaMA/comments/1rvnppg/dgx_station_is_available_via_oem_distributors/)** (热度: 418): **该图片展示了一台高性能工作站，很可能是 **NVIDIA DGX Station**，目前已通过 OEM 分销商发售。这台机器专为 AI 和深度学习应用设计，具有先进的冷却系统和性能。DGX Station 配备了 NVIDIA 的最新技术，是 AI 社区中许多人的“梦想机器”。讨论强调了它可以通过 **Dell** 和 **Exxact** 等分销商购买，据称价格在 `8.5-9 万美元` 范围内。其中提到了“一致性内存”（coherent memory）的概念，这是一种允许 CPU 和 GPU 之间高效共享数据的内存架构，有可能增强 AI 工作负载的性能。** 关于 DGX Station 的定价和可用性存在讨论，一些用户注意到 Dell 的产品列表存在差异。 “一致性内存”的概念也受到了质疑，显示出人们对其对 GPU 性能影响的好奇。

- DGX Station 的价格在 `85-90k USD` 之间，这是根据观察当前市场列表的用户所指出的。这种定价将其定位为高端机器，可能针对企业或研究机构，而非个人消费者。
- 尽管价格高昂且功能先进，DGX Station 除非安装了额外的显卡，否则缺乏视频输出。这一设计选择突显了其对计算任务而非传统图形输出的关注，符合其作为数据中心或 AI 研究工具而非消费级产品的定位。
- DGX Station 中“coherent memory”（一致性内存）的概念受到质疑，用户推测它是否允许对 GPU 进行完整的内存访问，类似于 DGX Spark。这一特性对于需要大型数据集和高速处理的任务将非常重要，强调了该机器对 AI 和机器学习应用的适用性。

- **[Mistral Small 4 | Mistral AI](https://www.reddit.com/r/LocalLLaMA/comments/1rvohug/mistral_small_4_mistral_ai/)** (热度: 323): **Mistral Small 4** 是一款多模态 AI 模型，拥有 `119 billion parameters` 和 `256k context window`，采用 Mixture of Experts (MoE) 架构，包含 `128 experts`。它旨在优化推理、多模态处理和编码任务的性能，并允许配置推理力度（reasoning effort）。该模型根据 Apache 2.0 许可证发布，支持文本和图像输入，旨在通过比其前代 Mistral Small 3 更低的延迟和更高的吞吐量实现高效的企业部署。更多细节可以在 [官方公告](https://mistral.ai/news/mistral-small-4) 中找到。** 评论者对该模型的 `6.5B active parameters` 很感兴趣，将其推理成本与 Qwen 3.5 35B-A3B 进行了比较，但其拥有更大的专家库。有人对 Mistral 在之前版本中的 tool calling 问题表示担忧，特别是在幻觉函数签名和丢失参数方面。该模型在 agentic 任务上的表现以及 `32k` 之外的上下文质量是关注的重点。

    - RestaurantHefty322 强调了 Mistral Small 4 的竞争定位，指出其 `119B` 参数及 `6.5B` 激活参数使其推理成本与 Qwen 3.5 35B-A3B 等模型相当，但拥有更大的专家库。这可能会挑战 Qwen 在 `~7B` 激活参数层级的霸主地位，特别是如果 Mistral 改进了其 tool calling 能力——该能力在 Devstral 2 中因幻觉函数签名和多步链中丢失参数等问题而备受诟病。
    - 讨论涉及了本地部署中 `6-7B` 激活参数范围内的文本和代码质量的重要性，特别是对 Mistral Small 4 如何处理 `32k` 以外的上下文质量感兴趣。这是较小的 MoE 模型经常挣扎的领域，尽管它们宣传有更长的上下文长度。
    - RepulsiveRaisin7 对 Mistral Small 4 相对于 Devstral 2 的改进表示怀疑，后者被认为落后于竞争对手。该评论反映了一个更广泛的担忧：考虑到 Mistral Small 4 的规模和竞争格局，它是否能比 Qwen 等现有模型提供切实优势。

- **[Mistral 4 Family Spotted](https://www.reddit.com/r/LocalLLaMA/comments/1rvfypu/mistral_4_family_spotted/)** (热度: 687): **Mistral 4** 系列引入了一种混合模型，集成了来自三个不同模型家族的能力：Instruct、Reasoning（前称 Magistral）和 Devstral。**Mistral-Small-4** 模型采用 `Mixture of Experts (MoE)` 架构，拥有 `128 experts` 且 `4 active`，总计 `119 billion parameters`，每个 token 激活 `6.5 billion` 参数。它支持 `256k context length`，并接受多模态输入（文本和图像）及文本输出。核心功能包括可配置的推理力度、多语言支持以及具有原生 function calling 的 Agent 能力。该模型根据 **Apache 2.0 License** 开源。[Mistral-Small-4](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603) 旨在兼顾速度和性能，提供超大上下文窗口和视觉能力。** 评论者对该模型的能力充满热情，尤其是其在 `120 billion parameter` 范围内的定位，可与 **gpt-oss-120B** 和 **Qwen-122B** 等模型相媲美。人们对其性能和潜在应用充满期待。

- Mistral 4 模型是一种混合架构，集成了来自三个不同模型系列的能力：Instruct、Reasoning（前身为 Magistral）和 Devstral。它采用了 Mixture of Experts (MoE) 架构，拥有 128 个专家，其中 4 个激活，总计 1190 亿参数，每个 Token 激活 65 亿参数。该模型支持 256k 上下文长度，并接受多模态输入（包括文本和图像），输出为文本。它还提供可配置的推理力度 (Reasoning Effort)，允许在快速即时回复和计算密集型推理模式之间切换。
- Mistral 4 旨在具有高度通用性，支持数十种语言的多语言能力，并提供具有原生 Function Calling 和 JSON 输出的高级 Agent 功能。它针对速度和性能进行了优化，并保持对 System Prompts 的强一致性。该模型在 Apache 2.0 许可证下发布，允许商业和非商业用途及修改，使其可广泛应用于各种场景。
- 该模型与 llama.cpp 的集成正在进行中，GitHub 上的一个 Pull Request 表明了这一点。这意味着 Mistral 4 很快将获得 llama.cpp 的支持，这是一个高效运行 LLM 的流行框架。这种集成可能会增强模型的可访问性和可用性，方便开发者在各种应用中利用其能力。


## 非技术性 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. AI 模型与工具创新

  - **[即将迎来令人惊叹的内容](https://www.reddit.com/r/singularity/comments/1rvlvw5/incredible_stuff_incoming/)** (热度: 483): **该图片展示了关于 **NVIDIA Nemotron 3 Ultra Base** 模型的演示幻灯片，该模型大小约为 `500B`。它声称是“最佳开源 Base Model”，具有 `5X` 的效率和高推理准确度。幻灯片包含的柱状图在多个 Benchmark 上比较了 Nemotron 3 Ultra 与 GLM 和 Kimi K2 等其他模型的表现，包括 Peak Throughput、Understanding MMLU Pro、Code HumanEval、Math GSM8K 和 Multilingual Global MMLU。Nemotron 3 Ultra 因其在这些类别中的卓越表现而被凸显。** 评论者对这些 Benchmark 表示怀疑，指出 NVIDIA 没有说明使用的是哪个 GLM 模型，且 Kimi K2 模型相对较旧（已有八个月历史）。此外，还有人批评其演示技巧，认为图表从 `60%` 开始刻度会夸大性能差距。

    - **elemental-mind** 指出了 NVIDIA 公告中的模糊性，指出他们没有指明引用的是哪个 GLM 模型。他们强调，如果是 Kimi K2 Base 版本，其智能程度可与 MiniMax M2.1 和 GLM-5-no-reasoning 相媲美，这暗示这种对比可能并不像看起来那么令人印象深刻。
    - **FullOf_Bad_Ideas** 澄清了 Base Model 与其 Finetuned 版本之间的区别。他们认为被比较的模型很可能是 Kimi K2 Base 1T 和 GLM 4.5 355B Base，而不是更先进的 K2.5 或 GLM 5，后者属于 Instruct/Reasoning 微调版本。这种区分对于理解讨论中的性能和能力至关重要。
    - **ThunderBeanage** 对 Kimi K2 的相关性表示怀疑，称其已过时。他们怀疑提到的 GLM 模型不是最新的 GLM 5，这意味着这种对比可能无法反映当前的 SOTA 模型。这种怀疑凸显了在性能讨论中明确模型版本的重要性。

### 2. AI 在创意与娱乐领域的应用

  - **[展示 LTX LORA 的真实能力！Dispatch LTX 2.3 LORA 多角色 + 风格](https://www.reddit.com/r/StableDiffusion/comments/1rv40xc/showing_real_capability_of_ltx_loras_dispatch_ltx/)** (热度: 932): **该帖子讨论了使用 LTX 2.3 创建 LORA 模型的过程。该模型在来自游戏《Dispatch》的约 `440 clips` 上进行训练，平均每个片段包含 `121 frames`。模型包含超过 `6 characters`，具有独特的语音和风格，通过为每个角色分配唯一的 trigger word 和详细的 captions 来实现。训练使用了 [akanetendo25 的 musubi fork](https://github.com/AkaneTendo25/musubi-tuner)，涉及使用 `pyscene` 切分片段，并将其转换为 `24 fps`，同时配合自定义打标工具。数据集根据片段长度分为 HD 和 SD 组，训练过程中使用了 `31GB VRAM` 和 `4 blockswap`。为了应对数据的复杂性，模型训练至 `64 rank`，且每 `500 steps` 保存一次 checkpoints。作者指出，虽然 LTX 在视觉表现上不如 WAN，但在游戏开发的 pre-visualization（预演）方面具有巨大的潜力。** 一位评论者对 WAN 2.5 是否开源表示怀疑，而另一位则赞扬了使用 `440 clips` 进行训练所付出的努力，并注意到其结果非常干净。

    - Lars-Krimi-8730 询问了训练 LTX 2.3 LORA 模型的底层技术细节，具体涉及使用的 trainer、设置、打标方法和分辨率。这表明用户对模型训练过程的可复现性和技术配置有浓厚兴趣。
    - Anxious_Sample_6163 强调了训练过程中使用了 440 个片段，这表明在数据准备方面付出了极大的努力。如此数量的片段意味着一个强大的数据集，很可能提升了模型的性能和纯净度。
    - SvenVargHimmel 询问了在 `5090` GPU 上的训练时长，这体现了对模型训练过程的计算资源需求和时间效率的关注。这个问题对于理解训练类似模型的可扩展性和可行性非常重要。

  - **[oldNokia Ultrareal. Flux2.Klein 9b LoRA](https://www.reddit.com/r/StableDiffusion/comments/1rutgoa/oldnokia_ultrareal_flux2klein_9b_lora/)** (热度: 541): **该帖子发布了 **Nokia 2MP Camera LoRA** 的重训练版本，命名为 **OldNokia UltraReal**，旨在模拟 2000 年代中期手机摄像头的审美。其核心特征包括软焦塑料镜头效果、退色的调色板，以及 JPEG compression 和 chroma noise 等数字伪影，所有这些都训练自作者的 Nokia E61i 照片存档。该模型可在 [Civitai](https://civitai.com/models/1808651/oldnokia-ultrareal) 和 [Hugging Face](https://huggingface.co/Danrisi/oldNokia_flux2_klein9b) 下载。** 一位评论者幽默地提到，历史上的 Nokia 摄像头缺乏模型中所展现的那种动态范围。另一位建议在 `qwen-image` 上训练该模型以进一步增强效果，还有一位对该 LoRA 表示热赞，并分享了一个涉及 frame injection 的个人项目。

    - jigendaisuke81 建议在 `qwen-image` 上训练模型，表现出对探索模型在不同数据集或架构下表现的兴趣。这可能意味着关注于增强图像生成能力或测试模型对各种图像风格的适应性。
    - Striking-Long-2960 提到对 “Wan2GP 中的 frame injection” 感兴趣，这表明了在技术上探索将帧集成到生成模型中的可能性。这可能涉及操作或增强图像序列，可能用于视频或动画制作。
    - berlinbaer 强调了该 LoRA 模型在复制特定视觉效果方面的技术成就，例如“带有红蓝色偏的过曝高光”。这表明用户关注模型准确模拟复杂摄影效果的能力，而这些效果可能很难通过简单的 prompting 实现。

### 3. AI 与就业影响

  - **[Anthropic CEO 表示 50% 的入门级白领工作将在 3 年内被消除](https://www.reddit.com/r/singularity/comments/1rw2tan/antrophic_ceo_says_50_entrylevel_whitecollar_jobs/)** (Activity: 2162): **Anthropic CEO** 预测，由于 AI 技术的进步，未来三年内 `50%` 的入门级白领工作将被消除。这一言论凸显了 AI 在职场的快速整合，可能会取代传统上由人类执行的任务，即使像 **Copilot** 这样的 AI 解决方案在质量和准确性上可能尚未达到人类专家的水平。该预测强调了就业市场的重大转变，突出了劳动力进行适应和技能进化的必要性。一个显著的评论分享了个人经历，即 AI 被用于执行任务但效果不佳，导致了错误和不正确的结论。这反映了人们对在专业环境中过早依赖 AI 的普遍担忧，这可能会损害人类专业知识和就业安全。

    - **Due_Answer_4230** 强调了工作场所 AI 整合中的实际问题，即使 **Copilot** 等 AI 工具表现不佳，仍被用来替代人类工作。这导致了错误和不正确的结论，但管理层可能因为速度而更青睐 AI，从而损害了投入多年时间培养专业知识的资深员工。
    - **Stahlboden** 提到了去年的一个预测，即 AI 将编写 100% 的代码，并指出虽然这还没有完全实现，但 AI 在编码中的作用已经显著增加。这反映了 AI 在技术领域日益增强的能力，暗示了 AI 未来可能主导某些任务。
    - **Environmental_Dog331** 指出 AI 领袖们缺乏针对 AI 进步导致的就业流失的解决方案。该评论强调了创造新职位的速度难以赶上 AI 驱动的失业速度，突显了劳动力转型战略规划中的关键差距。

  - **[NBC News 调查发现美国人对 AI 的反感程度甚至超过了 ICE](https://www.reddit.com/r/ChatGPT/comments/1rv9rsl/nbc_news_survey_finds_americans_hate_ai_even_more/)** (Activity: 1146): **NBC News** 的一项调查显示，只有 `26%` 的选民对 AI 持正面看法，而 `46%` 持负面看法，这使得 AI 的受支持程度低于除了民主党和伊朗之外的大多数话题。这反映了公众对 AI 广泛的怀疑态度，尽管它作为生产力工具被广泛使用且具有潜力。调查强调了 AI 被感知的潜力和其实际效用之间的脱节，特别是在取代需要大量行业知识的工作方面。评论者注意到一个悖论，即频繁使用 AI 的用户仍因有关 AI 能力（尤其是取代白领工作的潜力）的过度炒作而心生怨恨。大家一致认为，虽然 AI 是一个强大的工具，但目前尚不具备取代需要深厚行业知识的工作的能力。

    - **TimeTravelingChris** 强调了 AI 的潜力与当前实际应用之间的差距，指出虽然 AI 可以是强大的生产力工具，但尚不能取代需要大量行业和公司知识的工作。评论者强调了验证 AI 输出的重要性，因为在严密审查下，该技术仍存在显著缺陷。
    - **AlexWorkGuru** 讨论了实验室展示的 AI 潜力与用户日常体验之间的巨大差异，用户的日常体验通常涉及与基础 AI 实现（如 **Chatbot** 和自动化电话系统）的令人沮丧的交互。这种差距导致了 AI 的信誉问题，因为推广它的公司往往已经是用户不信任的公司，从而加剧了负面感知。
    - **bjxxjj** 指出，公众对 AI 的感知受到裁员和监控等负面关联的严重影响，而非教育类 **Chatbot** 等实际应用。这表明，有关 AI 情绪的调查结果可能会因为受访者所考虑的 AI 特定方面而产生偏差。

# AI Discords

遗憾的是，**Discord** 今天关闭了我们的访问权限。我们不会以这种形式恢复它，但我们将很快发布新的 **AINews**。感谢阅读到这里，这是一段美好的历程。