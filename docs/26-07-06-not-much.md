---
companies:
- tencent
- nvidia
- amd
- nous-research
- hugging-face
- artificial-anlysiis
- dair-ai
date: '2026-07-06T05:44:39.731046Z'
description: '**腾讯**发布了**Hy3**，这是一款拥有**2950亿参数**的 **MoE** 开放权重模型，**激活参数为210亿**，包含**192个专家**，支持**256K上下文**和
  **MTP 推测解码**。该模型可在 **vLLM** 上原生运行，并针对 **NVIDIA** 和 **AMD** 硬件进行了优化，最高可实现 **2.95倍**的加速并降低延迟。Hy3
  在开放模型领域与 **GLM-5.2** 的竞争力相当。**AutomationBench-AA** 榜单评估了智能体在**40款 SaaS 应用**中的**657项任务**表现，目前由
  **Claude Fable 5** 领跑，其后依次是 **Opus 4.8**、**Gemini 3.5 Flash** 和 **GPT-5.5 xhigh**。开放模型整体落后，其中表现最好的
  **GLM-5.2 max** 得分为 **27.8%**。最新的领域专用能力指数进一步凸显了成本与性能之间的权衡。关于持久化智能体记忆的研究包括：**A-TMA**
  提高了冲突处理的准确率，**ReContext** 则无需重新训练即可增强长上下文推理能力。

  '
id: MjAyNS0x
models:
- hy3
- glm-5.2
- claude-fable-5
- opus-4.8
- gemini-3.5-flash
- gpt-5.5-xhigh
- glm-5.2-max
people:
- eliebakouch
- shunyuyao12
- vllm_project
- teortaxestex
- tinygrad
- mbusigin
- artificialanlys
- fchollet
- omarsar0
title: '今天没发生什么特别的事。

  '
topics:
- mixture-of-experts
- model-quantization
- speculative-decoding
- inference-speed
- agent-evaluation
- long-context
- memory-optimization
- cost-efficiency
- benchmarking
- multi-domain-evaluation
---

**平静的一天。**

> 2026 年 7 月 4 日至 7 月 6 日的 AI 新闻。我们查看了 12 个 subreddit、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有进一步查看 Discord。[AINews 网站](https://news.smol.ai/)支持搜索所有往期内容。提醒一下，[AINews 现在已成为 Latent Space 的一个栏目](https://www.latent.space/p/2026)。你可以[选择接收或取消接收](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)不同频率的邮件！




---

# AI Twitter 动态回顾


**Tencent Hunyuan 发布 Hy3，开放权重模型迈入新阶段**

- **Hy3 以实力不俗的开放模型姿态登场**：Tencent 以 **Apache 2.0** 许可证发布了 **Hy3**。这是一款 **295B MoE** 模型，拥有 **21B 激活参数**、**192 个专家 / top-8 路由**、**GQA**、**256K 上下文窗口**，以及用于推测解码的 **3.8B MTP 层**。多篇帖子认为，它在推理、编程和 Agent 任务上足以与规模大得多的系统竞争，尤其强调了可靠性方面的改进，例如工具调用稳定性和抑制幻觉方面的工作 [@eliebakouch](https://x.com/eliebakouch/status/2074011171661701466)、[@HuggingPapers](https://x.com/HuggingPapers/status/2074024501201813797)、[@ShunyuYao12](https://x.com/ShunyuYao12/status/2074151389945827744)。
- **推理支持在发布当天就相当成熟**：[@vllm_project](https://x.com/vllm_project/status/2074147504254517529) 表示，Hy3 从发布起就能在 **vLLM** 中原生运行，并支持工具调用解析器、推理解析器、**MTP 推测解码**，同时已经验证可在 **NVIDIA 和 AMD** 平台上运行。后续的一篇帖子进一步介绍了 Tencent 的生产级内核已合并到 vLLM 主分支，包括负载均衡解码调度和融合 FP8 MoE 服务。据报告，在混合长度解码场景下，性能最高提升 **2.95 倍**；与默认后端相比，延迟方面 **TTFT** 约降低 **24%**，**TPOT** 约降低 **17%** [@vllm_project](https://x.com/vllm_project/status/2074147506875969754)。社区反响十分强烈，以至于 [@Teknium](https://x.com/Teknium/status/2074264567803531589) 很快就在 Nous Portal 上限时两周免费提供 Hy3。
- **更广泛的开放模型背景**：Hy3 很快就被拿来与 **GLM-5.2** 比较。一些发帖者认为，如果基准测试和实际体验测试的结果能够站得住脚，Tencent 现在已经跻身顶尖开源实验室之列 [@teortaxesTex](https://x.com/teortaxesTex/status/2074012467886178725)；但也有人仍坚持认为，**GLM-5.2** 是目前实际使用体验最好的开放权重模型 [@__tinygrad__](https://x.com/__tinygrad__/status/2074206866641752190)、[@mbusigin](https://x.com/mbusigin/status/2074238100251799998)。总体来看，开放模型前沿正在快速收敛，竞争重点也越来越偏向部署稳定性，而不只是排行榜上的原始分数差距。

**Agent 基准测试、Harness 与长期记忆**



- **AutomationBench-AA 带来了更贴近现实的 Agent 评测**：[ @ArtificialAnlys](https://x.com/ArtificialAnlys/status/2074194764510208230) 为 Zapier 的 **AutomationBench** 建立了一个独立排行榜，覆盖 **657 项任务**和 **40 个模拟 SaaS 应用**，同时评估目标完成情况与防护规则（guardrails）。**Claude Fable 5** 以 **48.6%** 领跑，略高于 **Opus 4.8** 的 **48.5%**；**Gemini 3.5 Flash** 和 **GPT-5.5 xhigh** 分别为 **42.6%** 和 **42.1%**。比排名更值得关注的是：所有模型仍然会违反业务规则；而 Gemini 在**每次防护规则违规所对应的目标完成数**和**成本效率**方面表现尤其突出。开放权重模型仍明显落后，其中 **GLM-5.2 max** 以 **27.8%** 成为榜单上表现最好的开放模型。
- **能力指数正变得更加多维**：Artificial Analysis 还推出了六个面向具体领域的指数——**Finance & Accounting、Legal、Healthcare & Medical、Strategy & Ops、Engineering、Economics**——试图摆脱单一的模型总分 [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2074299714699469221)。头条结果并不意外：**Claude Fable 5** 领跑，**Opus 4.8** 作为备选；但更有价值的发现是，不同领域的排名会发生非常明显的重排，而且价格与性能之间的前沿差距已经变得很陡峭。这与 [@fchollet](https://x.com/fchollet/status/2074242671103889799) 的观点一致：如果只报告基准分数，却不提供**单项任务成本**，这些分数正越来越失去意义。
- **记忆与检索仍是持久化 Agent 的瓶颈**：这里有两篇论文获得了较多关注。第一篇是 **A-TMA**，它解决的是“幽灵记忆”（ghost memory）问题：在长期运行的助手中，过时事实和当前事实会被同时检索出来。据称，在 LTP 基准上，将它加入 Graphiti 后，冲突判断准确率提升了 **0.240 个绝对百分点** [@omarsar0](https://x.com/omarsar0/status/2074121191846261022)。第二篇是 **ReContext**，这是一个无需训练的长上下文推理框架，会在生成答案前重新呈现模型内部的证据，从而在八个 128K 数据集上提升证据利用率 [@dair_ai](https://x.com/dair_ai/status/2074178316819677238)。再结合面向百万 token 上下文检索的 **BlockSearch** [@dair_ai](https://x.com/dair_ai/status/2074117920133898707)，可以看出一个清晰趋势：更好的记忆能力正越来越多地通过推理时工程来实现，而不只是依赖训练。

**Anthropic 的 J-Space / Global Workspace 研究结果**

- **机制可解释性成为焦点**：Anthropic 发布了一项研究，声称 Claude 内部存在一种**类似全局工作空间（global workspace）**的结构，其核心是他们称为 **J-space** 的一小部分激活 [@AnthropicAI](https://x.com/AnthropicAI/status/2074185348142280912)、[@AnthropicAI](https://x.com/AnthropicAI/status/2074185387577094398)。这项研究的核心并不是提取思维链，而是发现一种具有特殊地位的内部表征载体：它似乎可以被模型用于报告、调节和灵活推理。Anthropic 还为开放权重模型推出了 Neuronpedia 演示 [@AnthropicAI](https://x.com/AnthropicAI/status/2074185390060110138)。
- **研究人员为何关注这项工作**：即使对其表述方式存在分歧，可解释性研究人员仍将其视为比此前公开研究更有力的证据，表明模型可能存在某种“工作记忆”或内部工作空间。[ @NeelNanda5](https://x.com/NeelNanda5/status/2074193936588148891) 称这是目前关于类似工作记忆机制的最佳证据。[ @Jack_W_Lindsey](https://x.com/Jack_W_Lindsey/status/2074215950602379388) 则认为，理解这一特殊空间可能是理解 LLM 认知的关键。还有一些帖子强调了其在安全实践方面的价值：据称，这个工作空间可以在相关内容被语言化之前，呈现隐藏概念、检测 prompt injection，并暴露与内部破坏行为有关的特征 [@mlpowered](https://x.com/mlpowered/status/2074190714100146483)、[@LiorOnAI](https://x.com/LiorOnAI/status/2074198891990548940)、[@omarsar0](https://x.com/omarsar0/status/2074264122330612223)。
- **但“意识”的说法引发了争议**：Anthropic 的公开表述招致了强烈反弹。支持者认为，这些结果暗示了一种功能上类似于**通达意识（access consciousness）**的机制，而不是现象意识 [@BorisMPower](https://x.com/BorisMPower/status/2074201312531734567)；批评者则认为，该公司把具有特殊访问权限的潜在激活与意识混为一谈，存在过度宣称之嫌 [@AlanCowen](https://x.com/AlanCowen/status/2074265992570736919)。即便是态度较为支持的观点也强调，真正重要的意义在于：这为审计和引导模型提供了一个新的**干预点**，而不是带来了哲学层面的结论。

**推理、服务与系统效率**



- **推测解码仍是热门基础设施方向**：[[@lmsysorg](https://x.com/lmsysorg/status/2074176669108367549)] 已将 **DSpark** 加入 SGLang，用于基于置信度进行可变长度验证。其核心思路是，在高负载下避免验证每个草稿 token；与固定预算的推测方法相比，这有望改善吞吐量与延迟之间的权衡。据报道，DeepSeek-V4-Pro 在 **B300** 上、batch=1 时达到了 **383.7 tok/s**。Microsoft 还介绍了如何在 GitHub Copilot 测试框架中，对 **GPT-5.5** 进行提示词级优化，以便在发布后进一步改善延迟和 token 效率 [@code](https://x.com/code/status/2074178799512539571)、[@pierceboggan](https://x.com/pierceboggan/status/2074180737147027757)。
- **推理效率正日益成为战略瓶颈**：[@jon_durbin](https://x.com/jon_durbin/status/2074169183835685351) 认为，如今真正决定成败的是推理，而不只是训练，因为每条数据流水线、每个 RL 循环以及每个 Agent 运行时，最终都要转化为测试时计算。这一观点也体现在更底层的 kernel 工作中：Chutes 宣布 **MiniMax MSA** 和 **GatedDeltaNet-2** 获得了大幅加速，其中在 **RTX Pro 6000 / SM120** 上，稀疏注意力训练性能提升了 **约 7 倍**，融合 FP8 kernel 的表现也有所改善 [@jon_durbin](https://x.com/jon_durbin/status/2074119835366134188)。
- **超越模型服务的基础设施发布**：Cloudflare 推出了 **Workers Cache**，这是一种位于 Worker 入口前、按区域分层的缓存，可通过标准 HTTP 标头进行配置 [@Cloudflare](https://x.com/Cloudflare/status/2074117419728007181)。OpenAI 发布了 **GPT-Realtime-2.1-mini**，以与上一代 mini 相同的价格，将推理和工具调用能力带入 mini realtime 产品线；同时，OpenAI 表示缓存改进使 p95 延迟降低了 **25% 以上** [@OpenAIDevs](https://x.com/OpenAIDevs/status/2074255408013955466)、[@OpenAIDevs](https://x.com/OpenAIDevs/status/2074255420831735824)。

**世界模型、语音与文档 AI**

- **MIRA 是一个值得关注的世界模型演示**：General Intuition 和 Kyutai 携手 Epic Games 推出了 **MIRA**，这是一个可游玩的 Rocket League 多人世界模型，使用机器人采集的 **1 万小时**数据进行训练 [@gen_intuition](https://x.com/gen_intuition/status/2074104524596457706)。它能够以 **20 fps** 实时运行；相关帖子重点展示了一个 **50 亿参数**的模型，仅用一块 **NVIDIA B200** 就能运行完整的 2v2 比赛，而且不依赖显式的物理引擎或渲染引擎 [@TheRundownAI](https://x.com/TheRundownAI/status/2074184559768277398)。这清楚地表明，视频和世界模型正从玩具式演示逐步走向交互式模拟器。
- **语音领域的竞争依然激烈**：AssemblyAI 发布了 **Universal-3.5 Pro Realtime**，这是一款流式 STT 模型，在 AA-WER Streaming 基准上的 WER 为 **4.1%**；它还支持上下文预置，并且无需重新连接即可在通话过程中更新上下文 [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2074160133702402314)。在 TTS 方面，Artificial Analysis 表示，**Speechify Simba 3.2** 目前以 **1233 Elo** 位居其 Speech Arena 榜首，领先于 Gemini 3.1 Flash TTS、Sonic 3.5 和 Inworld Realtime TTS 1.5 Max，同时也是排名靠前的模型中价格最低的一个 [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2074265309985570890)。
- **文档上下文流水线正默认采用多模态方式**：LlamaIndex 和 LanceDB 介绍了一种用于处理复杂 PDF 的检索流水线，将**页面、分块和提取出的资源**分别存入相互关联的多模态表中；在一个带标签的 ESG 报告基准测试中，该方案取得了 **82% any-page-hit@5** 和 **74% 答案准确率** [@lancedb](https://x.com/lancedb/status/2074153945631457663)、[@llama_index](https://x.com/llama_index/status/2074170470119752084)。这与 Jerry Liu 提出的更宏观观点相呼应：Agent 需要一个专门的“文档上下文层” [@jerryjliu0](https://x.com/jerryjliu0/status/2074165277634253106)。

**热门推文（按互动量排名）**



- **Anthropic 的全球 workspace 论文**在互动量上遥遥领先，其中关于 Claude 内部 workspace/J-space 的主要公告远超其他内容 [@AnthropicAI](https://x.com/AnthropicAI/status/2074185348142280912)。
- **Tencent Hy3** 是最受关注的纯模型发布事件，尤其受到讨论开源竞争力和部署问题的技术类账号关注 [@teortaxesTex](https://x.com/teortaxesTex/status/2074012467886178725)、[@ShunyuYao12](https://x.com/ShunyuYao12/status/2074151389945827744)。
- **MIRA 的可交互世界模型**是最亮眼的多模态/系统演示 [@gen_intuition](https://x.com/gen_intuition/status/2074104524596457706)。
- **Will Depue 的“Stargate for Data”**帖子是内容最充实的战略分析，认为对 frontier labs 来说，数据采集——而不只是算力——将成为关键瓶颈，也可能成为护城河 [@willdepue](https://x.com/willdepue/status/2074178395462848800)。
- **John Carmack 关于 memory system 的帖子**引发了大量技术讨论。他认为，推理硬件可以利用确定性的访问模式，并采用比 HBM 便宜得多的 memory tiers，为大模型 serving 提供支持 [@ID_AA_Carmack](https://x.com/ID_AA_Carmack/status/2074248758422864226)。


---

# AI Reddit 盘点

## /r/LocalLlama + /r/localLLM 盘点

### 1. 大型 Open-Weight MoE 模型发布

  - **[longcat 2.0（1.6T，约 48B active）权重现已在 MIT license 下开放](https://www.reddit.com/r/LocalLLaMA/comments/1unyvnz/longcat_20_16t_48b_active_weights_are_now_open/)**（活跃度：638）：****LongCat 2.0** 的权重现已根据 **MIT license** 开放，相关公告来自 [elie](https://x.com/eliebakouch/status/2073690402503487902) 和 [ModelScope](https://x.com/ModelScope2022/status/2073710226365165679)，技术细节见 [LongCat 2.0 博客文章](https://longcat.chat/blog/longcat-2.0/)。这是一个规模极大的 **MoE** 系统，总参数量为 `1.6T`，每次推理约激活 `48B` 参数；评论者指出，发布的权重在 **BF16** 下约占 `3.55 TB`，在 **FP8** 下约占 `2.05 TB`。评论者强调了多 TB 权重规模带来的实际部署负担，并提到据报道，**Meituan**——一家常被描述为中国版 Groupon/Uber Eats 的公司——完全使用中国国产芯片训练了该模型，这也引发了关于其地缘政治和市场意义的讨论。

    - 评论者重点讨论了 **LongCat 2.0** 的规模和部署需求：总参数量为 `1.6T`，每次推理约激活 `48B` 参数，这意味着它采用了稀疏式/MoE 架构。有用户指出，发布的权重在 **BF16** 下约需要 `3.55 TB` 存储空间，在 **FP8** 下约需要 `2.05 TB`；对于计划搭建本地存储或推理基础设施的人来说，这一点非常重要。
    - 有人提出一个技术细节：据报道，**Meituan** 使用 `100%` 中国国产芯片训练了该模型。评论者认为，这对 AI 硬件供应链自主化具有重要意义。尤其值得注意的是，Meituan 是一家大型中国互联网公司，更接近 Groupon 与 Uber Eats 的结合体，而不是传统意义上的 AI 实验室。
    - 多位用户关注了宽松的 **MIT license**，并计划将其与 **Qwen**、**DeepSeek** 等领先的开源模型进行基准测试。总参数量达到 `1.6T`、但每次仅激活 `~48B` 参数，再加上开放权重，意味着只要推理工具能够高效支持其架构，该模型就可能适合与其他高端 MoE 开源模型进行比较。

  - **[Tencent Hy 发布的新 Open 模型：Hy3（总参数 295B，激活参数 21B，Apache 2.0）](https://www.reddit.com/r/LocalLLaMA/comments/1uoozt4/new_open_model_from_tencent_hy_hy3_295b_total_21b/)**（活跃度：604）：****Tencent** 在 [Hugging Face](https://huggingface.co/collections/tencent/hy3) 上发布了正式版 **Hy3** 模型系列。该系列被描述为一个总参数量 `295B`、每次激活 `21B` 参数的 MoE 模型，现已从此前限制较多的社区 license 改为 **Apache 2.0**。评论者关注一个相关的基准测试图表，并认为此次发布展现了“相较 HY3-Preview 相当亮眼的预期提升”，如果实际表现能够达到公布的结果，它可能会成为高端本地/家用推理环境值得关注的模型。讨论的主旋律是对 license 变更持积极态度：评论者认为，从地域受限、限制性较强的 license 改为 **Apache 2.0** 是最重要的改进，尤其考虑到 Tencent 最近还发布了采用 Apache license 的翻译模型。



  - 评论者指出，**Tencent Hunyuan Hy3** 是一个大型 MoE 风格模型，总参数量达到 **`295B`，激活参数量为 `21B`**。有用户提到，与 **HY3-Preview** 相比，它宣称的 benchmark 提升幅度相当可观；如果这些提升能在实际工作负载中体现出来，那么它可能会成为“高端家用配置”值得关注的选择。大家尤其关心实际推理支持情况，特别是 **GGUF 量化版本**，因为这将决定它是否适合本地部署。

    - 有人注意到，一个重要的许可变化是：Tencent 据称已将此前更严格的“community”许可证改为 **Apache 2.0**。原许可证曾限制在 **South Korea、UK 和 EU** 等地区的使用。评论者认为，这一变化意义重大，因为它能让模型更广泛地用于商业和研究用途，也与 Tencent 近期发布的一些采用 Apache 许可证的翻译模型保持一致。
    - 一位评论者认为，Hy3 可能成为 **Qwen** 和 **MiniMax** 的替代方案，反映出大家希望了解它的 benchmark 成绩和实际表现能否与当前领先的中国开源权重模型系列竞争。

  - **[新模型：GigaChat3.5-432B-A28B（首日即支持 GGUF！）](https://www.reddit.com/r/LocalLLaMA/comments/1uotkm7/new_model_gigachat35432ba28b_with_day0_gguf/)**（活跃度：439）：****Sberbank/ai-sage** 在 Hugging Face 上发布了 **GigaChat3.5-432B-A28B**，同时提供 [instruct](https://huggingface.co/ai-sage/GigaChat3.5-432B-A28B) 和 [base](https://huggingface.co/ai-sage/GigaChat3.5-432B-A28B-base) 版本，以及首日发布的 [GGUF 权重](https://huggingface.co/ai-sage/GigaChat3.5-432B-A28B-GGUF)；`llama.cpp` 已通过 PR [ggml-org/llama.cpp#25342](https://github.com/ggml-org/llama.cpp/pull/25342) 提供支持，但该 PR 目前尚未合入 master。评论者引用 model card 的说法称，这是一个**定制 MoE**，用来取代之前的 **GigaChat 3.1 Ultra 700B**：模型规模缩小约 `40%`，但在代码、数学和 Agent 任务上的表现更强；每个 token 使用的 KV cache 约减少 `4×`，在相同显存下可容纳超过 `2×` 的上下文，生成吞吐量提升约 `20%`。从架构上看，它据称采用 **MLA + GatedDeltaNet 线性注意力**的混合堆栈，并配备两个 MTP head；官方宣称，在贪心解码时，使用一个 head 可提速约 `1.5×`，使用两个 head 时最高可提速 `2.2×`。**主要的技术注意事项包括：与 **DeepSeek 3.2** 的 benchmark 对比，相较于当前前沿模型可能并不是特别有代表性；此外，GigaChat3.5 是一个**非推理模型**，因此解读 benchmark 结果时需要考虑这一点。一位评论者称赞了该发布的开放程度——对于如此规模的模型来说，同时开放 base model 和中间/open-weight checkpoints 并不常见；不过，训练数据集仍未公开。

    - 评论者指出，评估 **GigaChat3.5-432B-A28B** 时，应考虑它是一个*非推理模型*，因此与推理模型或前沿模型进行对比可能会产生误导。一位用户质疑为何使用 **DeepSeek 3.2** 作为参考，认为它“比前沿模型落后约一年”；另一位用户则表示，非推理基线如今越来越少见，应当单独进行评估。
    - 对于如此规模的模型而言，这次发布被认为具有不同寻常的开放程度：评论者特别提到，**中间 checkpoints 和 base model 都以开放权重形式提供**，并称其“在 Hugging Face 上发布的模型中，开放程度位居前 10%”。目前主要缺少的材料是**确切的训练数据集**，这限制了完整的复现以及对数据污染的分析。
    - 一段详细的架构说明称，**GigaChat 3.5 Ultra** 比 **GigaChat 3.1 Ultra 700B** 小约 `40%`，但在代码、数学和 Agent 性能上有所提升；每个 token 使用的 KV cache 约减少 `4×`，在相同显存下可容纳 `>2×` 的上下文，吞吐量提升约 `20%`。该模型采用定制 MoE 混合注意力设计，将 **MLA** 与 **GatedDeltaNet** 线性注意力层结合起来，并加入带两个 head 的 **Multi-Token Prediction**；据称，这能让贪心解码在使用一个 head 时提速约 `1.5×`，使用两个 head 时最高提速 `2.2×`。





### 2. 消费级硬件上的 Frontier-Scale Models

  - **[如果趋势保持不变，约 2 年内，消费级高端硬件或许就能运行 Mythos 级能力](https://www.reddit.com/r/LocalLLaMA/comments/1uoij3s/if_trends_hold_mythosclass_capability_may_be/)**（热度：1992）：这张**[图片](https://i.redd.it/5xwuga6pwhbh1.png)**是一张名为**“从 Frontier 到在笔记本电脑上运行”**的推测性趋势图。图中认为，过去可在笔记本电脑上运行的开放权重模型，通常会比 Frontier 模型晚平均 `24.8 个月`发布——例如，**GPT-3 → Llama 2 70B** 相隔 `37 个月`，**ChatGPT/GPT-3.5 → Llama 3 70B** 相隔 `17 个月`，而 **GPT-4 → Gemma 3/Qwen3 级别**相隔 `24 个月`。按照这一趋势外推，到 **2027 年年中左右**，高端消费级硬件或许就能运行 **GPT-5/Claude 4 级别**的能力，到 **2028 年 7 月左右**则可能达到 **Fable/Mythos 级别**。不过，这只是启发式推测，并非经过基准测试验证的结果。评论者对“消费级硬件”能否在这一趋势下继续保持可负担性持怀疑态度，认为高端本地推理的成本可能会逐渐接近企业级计算成本。一条技术讨论指出，**Gemma 4 26B A4B** 在 **RTX 5080** 上处理长上下文时最初表现很差，但用户后来发现问题出在配置上，并在使用 `--no-mmap --batch-size 256 --ubatch-size 512` 后报告速度约为 `100 tok/s`。

    - 一位用户报告称，**Gemma 4 26B A4B QAT** 在 **RTX 5080** 上处理长上下文时最初表现不佳：在 `20K` 上下文下生成速度只有约 `6 tok/s`，这让人怀疑假设中的 **Gemma 4 31B dense** 模型是否适合在笔记本级硬件上运行。后来他们发现这是配置问题，并应用 `llama.cpp` 风格的参数 `--no-mmap --batch-size 256 --ubatch-size 512`，使吞吐量提升到了空闲时约 `100 tok/s`、负载下约 `60 tok/s`，相关配置指南见：[在本地运行 Gemma 4 26B A4B](https://carteakey.dev/blog/local-inference/running-gemma-4-26b-a4b-locally/)。
    - 一位评论者提醒说，推测 **Mythos 级别**模型能否在本地运行仍然十分 speculative，因为模型的实际大小和架构尚未确定；他们指出，该模型的规模可能达到“`Opus 4.8` 的 3 倍”，因此，假设它能装进当前的高端消费级 GPU 并不可靠。

  - **[我在一台只有 25 GB RAM 的普通笔记本电脑上运行了 GLM-5.2（744B MoE）——纯 C 实现，专家模块从磁盘流式读取](https://www.reddit.com/r/LocalLLM/comments/1uocapw/i_managed_to_run_glm52_744b_moe_on_a_humble_25_gb/)**（热度：546）：作者构建了 **[colibrì](https://github.com/JustVugg/colibri)**，这是一个纯 C、零依赖的 **GLM-5.2 744B MoE** 推理引擎。它将稠密的 `int4` 部分常驻在约 `9.9–10 GB` RAM 中，同时按需从磁盘流式读取约 `2.1 万个`路由专家（总计约 `370 GB int4`）。该引擎实现了 GLM-5.2 的完整前向路径，包括 **MLA attention**、压缩 KV cache、DeepSeek 风格的路由、MTP speculative decoding、`int8/int4` AVX2 内核、异步专家预读、batch-union MoE，以及 FP8→int4 转换器。在一台配备 12 核 CPU、25 GB RAM、运行 WSL2 的 NVMe 笔记本电脑上，报告的性能受磁盘速度限制：冷启动时约为 `0.05–0.1 tok/s`，每生成一个 token 需要随机读取约 `11 GB` 数据。热门评论大多对把这种以 `0.1 t/s` 的速度称为“运行”模型持怀疑或调侃态度；有人认为，现在更适合用“每个 token 需要多少秒”来衡量，而不是 tokens per second。

    - 一位评论者报告或推断，其吞吐量极低，约为 `0.1 tokens/s`；其他人则将这一结果重新表述为 **每个 token 需要多少秒**，而不是每秒生成多少 token。这场讨论表明，基于磁盘流式读取的 MoE 方案在技术上确实可以运行，但主要受 I/O 延迟拖累，尤其是在使用机械硬盘或 DDR3 时代的老旧系统时。
    - 有人提出技术问题：是否尝试过使用带有 `mmap` 的 `llama.cpp`，而不是定制的纯 C 磁盘流式方案。这意味着，另一条可能的实现路径是使用内存映射模型权重，让操作系统负责分页，而不是显式地从磁盘流式读取专家模块。




## AI 子版块简报（技术性较低）

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Fable 5 能力演示与长上下文使用



  - **[我一开始误解了 Fable，现在明白了。](https://www.reddit.com/r/ClaudeAI/comments/1uo1xpz/i_misunderstood_fable_at_first_now_i_get_it/)**（热度：1626）：**这篇帖子认为，Fable 在“原始智能”方面只比 **Opus** 略胜一筹，但它的实际优势在于：面对规模更大、彼此相互依赖的产物时，能够保持连贯的上下文。在一个涉及 `8` 张原理图的 PCB 审查流程中，作者表示，Fable 能更好地跟踪跨图纸依赖关系；而 Opus 在超过约 `2` 张图纸后就“抓不住重点”了。这表明 Fable 可能在长上下文处理和全局推理方面更强，而不是局部推理能力更好。** 热门评论基本认同这一点：Fable 的价值在于“看清全局”，并在更长的上下文中持续保持代码和设计质量。一位用户介绍的工作流是：先用 Fable 分析整个代码库和文档并生成建议，再切换到 Opus 逐项实现。

    - 几位评论者认为，在高层次的软件工程编排方面，**Fable** 比 **Opus** 更强：它可以分析现有代码库和文档、找出缺口，并在切换到 Opus 逐项编写代码之前，先产出建议和实施计划。用户反馈的典型工作流是：用 Fable 评估架构和代码库并制定计划，再用 Opus 负责具体实现。
    - 一个反复出现的技术差异是：Fable 未必能立刻写出比 Opus “更好”的代码，但它似乎能通过避免走进无效的实现路径，*在更长时间内保持质量*。一位评论者提到，Opus 会反复坚持一种明知不可行的方法，而 Fable 能意识到这条策略没有产出。这种能力更适合项目层面的决策，而不只是局部代码生成。
    - 用户也指出了两者在交互风格上的取舍：Fable 在项目规划、差距分析和理解现有系统方面表现出色，但在聊天模式下可能会急于下结论，而不是按照指示先提出澄清问题并等待回复。这说明 Fable 的优势可能更多体现在自主规划和审查上，而不是严格受控的交互式需求收集。

  - **[Google DeepMind 的产品与设计负责人正在使用并宣传竞争对手的模型](https://www.reddit.com/r/singularity/comments/1uo3af4/google_deepmind_product_and_design_lead_using_and/)**（热度：1192）：**这张[图片](https://i.redd.it/0k8376mn5fbh1.png)显示，**Ammaar Reshi**——帖子标题称其为 **Google DeepMind 的产品与设计负责人**——公开表示，他使用竞争对手的模型 **“Fable 5”**，将 *Command \& Conquer: Generals Zero Hour* 移植到了 **iPhone/iPad**。这一说法在技术上值得注意，因为它声称一款 **2003 年的 PC 即时战略游戏引擎**已经针对 **ARM64** 原生编译，并加入了**触控操作**。这意味着 LLM 可能协助完成了跨架构、跨平台 API 以及跨输入范式的代码迁移和移植。** 评论大多将此视为竞争情报，而不是不忠：一位用户认为，产品负责人本来就应该充分了解竞争对手的工具；另一位则指出，Google 与类似 Anthropic 的竞争对手之间存在投资和合作关系，使这种竞争关系变得更加复杂。还有一位评论者表示，就在几个月前，这类由 LLM 协助完成的游戏移植还被认为至少要过几年才能实现。

    - 一个技术含量较高的讨论指向了实际项目：[`Generals-Mac-iOS-iPad`](https://github.com/ammaarreshi/Generals-Mac-iOS-iPad)。该项目似乎是在 Apple 平台上封装并移植 *Command \& Conquer: Generals*。一位评论者认为，虽然成果令人印象深刻，但它很可能是在一套遗留的 `DX8` 代码库之上叠加了“`4 abstraction layers`”，并建议等待 [`TheSuperHackers/GeneralsGameCode`](https://github.com/TheSuperHackers/GeneralsGameCode/) 提供更简洁的引擎级重写版本。
    - 几条评论认为，使用 Claude 与其说是不忠，不如说是竞争分析；毕竟，产品/设计负责人应该亲自了解竞争模型的能力。一位评论者还补充称，Google 与 Anthropic 存在相当深的关系，并声称 Google 持有 Anthropic 约 `18%` 的股份，因此把对方简单称为“竞争对手”在技术和商业层面都不够准确。
    - 一个值得注意的技术观察是，真正令人意外的并不是使用 Claude/Fable 这类工具，而是这个移植项目优先面向 Apple 平台，而不是 Android。这说明讨论的一部分其实涉及平台/运行时的可行性，以及在 iOS/macOS/iPadOS 上运行传统 PC 游戏时工具链的成熟度，而不只是模型选择问题。



  - **[《The Room》——Fable 的 One-shot 作品](https://www.reddit.com/r/singularity/comments/1uow9c8/the_room_one_shot_by_fable/)**（活跃度：683）：**Reddit 帖子展示了一段名为 **“The Room”** 的视频，据描述这是 **Fable** 制作的 **one-shot** 作品，但帖子链接中的 Reddit 媒体地址（[v.redd.it/68csn9fdulbh1](https://v.redd.it/68csn9fdulbh1)）因 Reddit 返回 **HTTP `403 Forbidden`** 而无法从外部访问，因此无法核实实际媒体内容或实现细节。评论显示，这段视频似乎是一种细节极其丰富的连续缩放或尺度转换视觉效果，观众纷纷猜测这种细节是如何生成的，并询问其背后的 *代码/成本*。热门技术讨论主要集中在缩放范围不够深入——一位评论者认为缩放本应继续深入到夸克以下——以及对场景能否以如此高的细节程度完成渲染的质疑和惊叹。

    - 评论者对 Fable 这段 one-shot 视频背后的生成方案十分好奇，具体询问了实现这种显著场景细节所使用的 **prompt**、可能的制作流程，以及所需的计算资源、代码和成本。
    - 一位评论者指出，这个“向内缩放”的概念错过了继续深入夸克以下的机会，并将其视为一种推测性的技术/叙事局限：目前没有得到证实的物理规律表明夸克就是可能存在的最小组成单位，因此这段序列本可以进一步探索更深层的假想结构。


### 2. Claude 在实际工作流和 Agent 仪表板中的应用

  - **[Claude 遇上政府监督 🫡🇺🇸](https://www.reddit.com/r/ClaudeAI/comments/1uobmts/claude_meets_government_oversight/)**（活跃度：744）：**发帖者正在开发 **“Article One”**，这是一个由 Claude 驱动的多 Agent 仪表板，旨在汇总国会议员资料、选区/竞选背景、履职表现指标、竞选捐助者分析，以及国会办公室支出和纳税人资助的运营信息，并将其整合到一个透明度界面中。该项目目前尚未发布；发帖者表示，由于 Claude 每周的使用额度限制，代码仓库和仪表板的公开时间被推迟，并通过 [Buy Me a Coffee](https://buymeacoffee.com/AJK28) 寻求资助，以购买 Claude Max、加快开发。** 评论者普遍支持这一透明度应用场景，但强调每一项指标都必须提供 **可验证的来源、计算方法和可审计性**，以避免 AI 产生幻觉或给出误导性结论。大家最希望增加的功能，是在公开传记信息之外进行更深入的财务分析：例如配偶/家庭财富变化、竞选资助者、关联行业/PAC，以及潜在利益冲突，而不是仅仅提供类似维基百科的个人简介。

    - 评论者强调，任何基于 Claude 的政府透明度工具，都需要为每一项派生统计数据提供 **可验证的引用和计算方法**。尤其是在有用户质疑“众议院中有 `33%` 的议员比 AOC 更自由派”这一看起来不太可信的说法之后，这一点更加重要。主要技术担忧在于，来源不可靠或由幻觉生成的政治指标，可能会让系统不但无法提供帮助，反而造成实际危害。
    - 有评论者提出了一项实质性的功能建议：不要局限于面向公众的传记摘要，而应扩展到 **竞选财务和利益冲突分析**，包括调查谁在资助每位政治人物、其配偶/家庭财富在任职期间如何变化、关联人士是否经营对冲基金或其他投资实体，以及像 **AIPAC** 这样的团体捐款是否与投票行为存在相关性。这意味着该工具需要系统化地整合财务披露、竞选捐款数据库和投票记录，并对家庭成员及商业关系中的实体进行实体消歧和关联。

  - **[谢谢你，Anthropic。作为一名教师，Claude cowork 真是我的救星。](https://www.reddit.com/r/ClaudeAI/comments/1uox9uu/thank_you_anthropic_as_a_teacher_claude_cowork_has_been_godsend/)**（活跃度：688）：**一位教师表示，他使用 **Anthropic Claude**（“Claude cowork”）来设计课程、辅助批改、生成 PowerPoint，以及分析学生成绩数据。他说，通过将上传的教育学文档与课程规划工作流结合起来，这个工具为他节省了数小时。对方最希望增加的功能，是为个人账户提供 **Microsoft 365 / OneNote 连接器**，因为所在学校没有使用工作或教育版 Microsoft 365 租户，这限制了该工具与现有教学材料的集成。** 热门评论提出了一个具体的数据治理问题：如果将学生姓名、成绩或可识别的作业上传到 Claude，除非事先进行匿名化处理或该系统已获批准，否则可能违反学校政策或数据保护规定。另一位评论者指出，删去学生身份信息可能会抵消大部分效率提升，尤其是在需要对作文提供个性化反馈时。



    - 几位评论者都重点提到了在教育场景中使用 Claude 时存在的**学生数据隐私风险**：如果把学生姓名或可识别的背景信息输入未经批准的 AI/工作系统，可能违反学校政策，甚至导致纪律处分。一位教师介绍了匿名化处理带来的实际负担：他们曾尝试用 Claude 为 11 年级学生提供作文反馈，但花费了大量时间*“清除姓名和身份标识信息”*，最终生产力提升被削弱，甚至荡然无存。
    - 有人提出了一种缓解措施：为 Claude 配置明确的学生数据保护指令，上传学校的数据保护政策，并要求它在检测到敏感信息时停止。评论者强调，这种做法**并非万无一失**，但可以作为一种轻量级防护措施；还可以让 Claude 在个性化处理必须使用敏感数据时，建议保护隐私的工作流程或替代方案。
    - 人们还关注与 **Microsoft 365 / PowerPoint** 更紧密的集成，尤其是对使用个人账户而非受管理账户的学校而言。评论者认为，缺少经过批准的连接器会增加工作流程中的摩擦，迫使教师采用手动变通方案，进而同时提高时间成本和数据治理风险。

  - **[我觉得我们正迅速走向这样一个时代：每个人都有各种神奇、定制化的本地工具，而且只属于自己](https://www.reddit.com/r/ClaudeAI/comments/1uopekl/i_feel_like_were_rapidly_heading_to_a_place_where/)**（活跃度：875）：**这篇帖子观察到一种日益明显的趋势：由 AI 辅助构建的“定制化本地工具”不断增多。这些软件对个人或特定组织非常有用，与某位用户的工作流程紧密结合，却很难泛化或分发给他人。评论者举了不少例子，包括脆弱的个人自动化方案、定制的健身/闹钟应用，以及通过“vibecoding”给一家细分领域公司搭建的 **ERP** 系统——如果依靠传统软件开发，市场规模根本不足以支撑其开发。**评论者普遍对此持积极态度：AI 降低了为极小众需求开发软件的成本，即使最终产品无法迁移、较为脆弱，或只能由创建者自己维护。

    - 几位评论者将本地 AI 构建的软件描述为**高度个性化但无法迁移**。其中一人把自己的系统称作一个*“神奇的小盒子”*，并表示只要别人动它就可能出问题。从技术角度看，这意味着许多 AI 生成的工具可能更注重适配个人工作流程，而不是可维护性、可移植性、上手难度或普适的产品市场契合度。
    - 一位用户表示，自己通过“vibecoding”为一家细分领域公司搭建了定制的 **ERP 系统**，并认为 AI 辅助开发让原本不值得传统供应商或开发者投入的软件项目具备了经济可行性。这说明未来可能会越来越多地出现面向内部、特定领域的工具：它们的投资回报来自解决某个狭窄的运营需求，而不是打造可复用的 SaaS 产品。
    - 另一条评论预测，小企业的行政工作很快会由善于使用 **Claude** 的员工取代，并将 Claude 形容为*“一整个行政人员和实习生团队”*，还把协作式 AI 工具比作高端行政助理。其核心观点是，价值可能不会主要流向通用的“AI 自动化”产品，而是流向那些能够把 LLM 整合进具体企业行政流程的员工。




# AI Discord 社区

很遗憾，Discord 今天终止了我们的访问权限。我们不会以这种形式恢复它，但很快会推出全新的 AINews。感谢你读到这里，这段旅程曾经很美好。