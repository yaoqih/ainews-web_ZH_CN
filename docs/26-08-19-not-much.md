---
companies:
- ornith
- vllm
- ollama
- unsloth
- qwen
- arena
- valsai
- deepseek
- truefoundry
- claude
date: '2026-08-19T05:44:39.731046Z'
description: '**Ornith-1.5** 发布了全新的开放权重模型系列，包含 **9B 稠密模型、35B MoE 模型和 397B MoE 模型**，采用
  **MIT 许可证**。该系列支持 **FP8、GGUF、MLX 和 NVFP4** 等量化格式，并展示了端到端的**自我改进**能力。


  压缩技术进一步提升了模型的准确率和效率。其中，采用 **Dynamic V3** 的 **Qwen3.8-27B GGUF** 模型准确率提高了 **10%**；在
  **8GB 内存**设备上，1-bit 量化仍能保留 **77% 的 BF16 准确率**。


  智能体评测榜单显示，**Claude Opus 5（High）、Kimi K3、GLM 5.2、Grok 4.5** 和 **GPT-5.6 Luna** 等模型在质量和性价比方面处于领先位置。


  **DeepSeek Harness（DSH）** 推出了一种基于插件的开放式智能体运行时架构，重点优化了可扩展性和工具支持。


  **TrueFoundry** 开源了 **TrueForge**。这是一套可自行部署、与供应商无关的智能体运行框架，在保持准确率的同时，可减少 **30% 的
  token 使用量**，并降低 **75% 的成本**。这也凸显出，在智能体平台中，会话、环境、记忆和工具等层的重要性正不断提升。'
id: MjAyNS0x
models:
- ornith-1.5
- qwen3.8-27b
- claude-opus-5
- kimi-k3
- glm-5.2
- grok-4.5
- gpt-5.6-luna
- grok-4.6
- glm-5.3
- trueforge
people:
- ornith_
- unslothai
- danielhanchen
- arena
- valsai
- zhihufrontier
- theturingpost
- truefoundry
- omarsar0
- kimmonismus
- bradenjhancock
- dbreunig
- rseroter
- claudedevs
title: 今天没发生什么特别的事。
topics:
- model-compression
- quantization
- reinforcement-learning
- agent-evaluation
- plugin-architecture
- open-agent-runtime
- cost-efficiency
- session-management
- tooling
- benchmarking
---

**平静的一天。**

> AI 新闻，时间范围为 2026 年 8 月 18 日至 19 日。我们查看了 12 个 subreddit、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有继续追踪其他 Discord 频道。你可以在 [AINews 网站](https://news.smol.ai/) 搜索过去发布的所有内容。提醒一下，[AINews 现在已成为 Latent Space 的一个栏目](https://www.latent.space/p/2026)。你可以选择[订阅或取消订阅不同的邮件发送频率](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)！




---

# AI Twitter 速览


**开放权重模型、压缩技术与基准排名变化**

- **Ornith-1.5 成为一支不容忽视的新开放模型家族**：[[@ornith_](https://x.com/ornith_/status/2090074077084127302)] 发布了 **Ornith-1.5**，提供 **9B dense、35B MoE 和 397B MoE** 三种版本，采用 **MIT** 许可证，并支持 **FP8、GGUF、MLX 和 NVFP4** 等量化格式。其最受关注的特点是端到端的**自我改进**能力：模型可以自行提出任务、生成脚手架，并产出 RL rollout，从而创建新的训练数据。官方公布的评测结果在 Agent 和编程任务上表现强劲，包括 **Terminal-Bench 2.1：86.1**、**SWE-Bench Verified：86**、**DeepSWE：56**、**HLE：44.6** 以及 **Tool Decathlon：71.2**。发布后不久，[vLLM](https://x.com/vllm_project/status/2090243605147586955) 和 [Ollama](https://x.com/ornith_/status/2090276420983587087) 就已将其接入各自的推理服务栈。
- **压缩技术继续变得更加激进，同时仍尽力保持模型实用性**：[[@UnslothAI](https://x.com/UnslothAI/status/2090103470015828184)] 和 [@danielhanchen](https://x.com/danielhanchen/status/2090119165055324518) 发布了采用 **Dynamic V3** 的新一批 **Qwen3.8-27B GGUF**，声称在相同模型大小下，准确率大约提升 **10%**；同时还推出了 **1-bit 量化版本**，在仅使用 **8GB RAM** 的情况下，仍能保留约 **77% 的 BF16 准确率**。他们新增的 **Divergence-300** 指标，使用来自 **Terminal Bench**、**DeepSWE** 及相关任务的未见样本，将 top-1% 贪心准确率的评估范围扩展到更长的生成结果。
- **Agent 和法律评测榜单仍在持续洗牌**：[[@arena](https://x.com/arena/status/2090137780932538549)] 发布了 **Agent Arena** 的 Pareto 视图，其中 **Claude Opus 5 (High)** 在质量方面领先，但 **Kimi K3**、**GLM 5.2**、**Grok 4.5** 和 **GPT-5.6 Luna** 等成本更低的模型，则构成了价值前沿的大部分。另一方面，[ValsAI](https://x.com/ValsAI/status/2090119651204423763) 报告称，**Grok 4.6** 在 **Legal Research Bench** 中以 **48.1%** 的成绩排名 **第 3/49**，支持 **500k 上下文**以及工具、图像和文件处理，价格也相对较低。对于开放模型，[ValsAI](https://x.com/ValsAI/status/2090192848780136668) 还特别指出，**GLM 5.3** 在开放权重模型中位列 **Terminal Bench 第 2、Legal Bench 第 3、Skills Bench 第 6**。

**Agent Harness 成为新的竞争层**

- **DeepSeek Harness 的极简是刻意设计，而不是功能不完整**：由 [@ZhihuFrontier](https://x.com/ZhihuFrontier/status/2089998555889250478) 推广、[@TheTuringPost](https://x.com/TheTuringPost/status/2090096803899216151) 总结的一篇详细文章，将 **DeepSeek Harness (DSH)** 描述为构建在名为 **Cordis** 的插件架构之上的轻量外壳。其核心设计选择是：**一切皆为插件**，包括 Agent 循环本身。据报道，早期 beta 用户在不到一周内就发布了 **100 多个插件**，提交了 **400 多个 issue**；应用示例从**五子棋模型测试平台**，到连接模型与实时查询执行、通过 SQL 执行闭合反馈环路的**数据库 Agent**，覆盖范围十分广泛。最值得关注的是其架构理念：DSH 与其说是“产品化的助手”，不如说是一个**开放的 Agent 运行时**，重点面向用户可扩展的工具、可替换的控制循环，以及业务规则注入。

- **TrueFoundry 开源 TrueForge，并明确提出 Harness 成本问题**：[@truefoundry](https://x.com/truefoundry/status/2090081376330715176)、[@omarsar0](https://x.com/omarsar0/status/2090138030296219973) 和 [@kimmonismus](https://x.com/kimmonismus/status/2090159374450974850) 都报道了 **TrueForge** 的发布。这是一款采用 **MIT 许可证**、支持自行托管、且不绑定供应商的生产级 Agent Harness。该技术栈包含工具编排、上下文管理、子 Agent、代码沙箱、人工审批和追踪能力，并同时支持**本地**和**托管**部署模式。最受关注的技术结论是：在包含 **14 项任务的企业基准测试**中，TrueForge 使用 **Opus 4.8** 时，性能达到 **Claude Managed Agents** 的水平，同时减少了约 **30% 的 token 使用量**；改用 **GLM-5.2** 后，成本降低约 **75%**，准确率仍保持不变。更广泛的行业趋势，也得到了 [@bradenjhancock](https://x.com/bradenjhancock/status/2090114460828766567) 和 [@dbreunig via @rseroter](https://x.com/rseroter/status/2090146780658782517) 的呼应：**会话、环境、记忆和工具这一层**，正逐渐成为差异化竞争和成本节省的重要来源。

- **托管型 Harness 也在持续加强可观测性和控制能力**：[@ClaudeDevs](https://x.com/ClaudeDevs/status/2090218983962390950) 为**自托管沙箱**增加了**记忆支持**，为 Web 工具加入了**域名允许/阻止控制**，并重新设计了**多 Agent 会话查看器**，新增 **minimap**、**分组转录**以及按线程/会话统计的**成本明细**。与此同时，OpenAI 继续推进另一条路线：向团队提供可嵌入自有产品的 Harness 基础能力。[@OpenAIDevs](https://x.com/OpenAIDevs/status/2090230646497251387) 强调，**开源 Codex Harness** 是内部工具、运维仪表盘和定制应用底层所使用的运行时；[@cursor_ai](https://x.com/cursor_ai/status/2090136956101414982) 则围绕持久化目标和长期运行的会话，改进了云端 Agent 的用户体验。

**Post-Training、Mid-Training 与 RL 系统工作**

- **越来越多证据表明，扩展的重点正从参数规模转向训练方案的质量**：[[@kimmonismus](https://x.com/kimmonismus/status/2090026799916888080)] 转述了 **zAI/GLM** 创始人的一个重要观点：模型能力仍在持续扩展，但业内讨论过多聚焦于参数量，而忽视了**数据质量、推理计算量和后训练**。文中提到的例子是 **GLM-5.3**：据称它与 **GLM-5.2** 使用相同的核心基座模型和架构，但通过额外约**一个月的 RL**，性能得到了大幅提升。
- **Microsoft 的 Agent Lightning 展示了通过 harness 实现 RL 的实用方案**：[@omarsar0](https://x.com/omarsar0/status/2090078336697733531) 介绍了 **Agent Lightning v1.0**。它通过 endpoint proxy 将任意 harness 接入 RL，并处理**重新分词、样本合并、优势计算、归一化，以及 scheduler/backend 协同**等问题。据称，在约 **6K 个训练样本**和适度计算资源下，它能将 **Qwen3.5-9B** 在 **SWE-Bench Verified** 上的成绩从 **41.8% 提升到 56.4%**。
- **业界正更加明确地把 mid-training 视为一个优化空间**：[@cwolferesearch](https://x.com/cwolferesearch/status/2090080281248325744) 梳理了当前实践者对 **CPT/midtraining** 的看法：优化重点不仅是“用更好的数据继续预训练”，还包括**数据配比、训练时长、阶段顺序、序列长度，甚至模型后训练的可优化性**。这篇讨论的价值在于，它把这些因素视为彼此影响的调节旋钮，而不是相互独立的技巧。
- **RL 基础设施也在持续改进，为研究提供更好的底层支持**：[@SergioPaniego](https://x.com/SergioPaniego/status/2090052408940666888) 转述了一项研究：在 **TRL** 中，通过 generation buffer、批量调用 teacher，以及二进制编码 logprob，让**基于 on-policy 的蒸馏**提速了 **40 倍**；[@mikasenghaas](https://x.com/mikasenghaas/status/2090212176166629474) 宣布 **prl** 加入了**自适应并发**功能，可以在 RL 运行过程中动态调整正在执行的 rollout 数量。

**生产环境中值得关注的基准、检索与基础设施细节**

- **Qdrant 的可过滤 HNSW 与 ACORN 对比，是检索系统领域的一项实质性进展**：[@qdrant_engine](https://x.com/qdrant_engine/status/2089999409404957029) 认为，带过滤条件的 ANN 检索应当在**索引层**处理，而不只是依赖查询时过滤。他们提出的 **filterable HNSW** 会在共享索引 payload 值的点之间增加边，从而保持过滤后子图的连通性。在其针对 **100 万个向量中筛选 1%** 的基准测试中，**filterable HNSW** 达到了 **99.8% 的召回率，延迟 1.0ms**；相比之下，**ACORN** 的召回率为 **67.7%，延迟 4.7ms**。他们也指出，对于**取值范围较宽的字段**和 **AND 过滤条件**，ACORN 仍然有帮助，尤其是在已经针对过滤进行优化的图之上。
- **Sentence Transformers v6.0 体现了检索从单向量转向多向量的实际趋势**：[@tomaarsen](https://x.com/tomaarsen/status/2090018110052987171) 清晰概括了两者的区别：稠密检索会将每段文本压缩成一个向量，而**多向量**检索会保留 token 级别的向量，先计算查询 token 与文档 token 之间的匹配分数，再汇总其中的最佳匹配。这一点很重要，因为对于重视质量的搜索系统而言，late-interaction 检索正逐渐成为默认的权衡方案。
- **生产环境中的 Agent 延迟，往往与模型本身关系不大**：[@dair_ai](https://x.com/dair_ai/status/2090117595907383672) 总结了一篇对十个 agentic 应用进行监测的论文。研究发现，在其中一半的应用里，**非 LLM 组件占据了主要延迟**；sandbox 内存峰值达到**每个 session 28GB**，不同子系统之间的延迟差异最高可达 **32 倍**，并且步骤之间会长时间保留闲置状态。论文提出的优化方向并不意外，但十分重要：**面向任务的 serving** 可将延迟降低 **29–40%**，**状态卸载**可将内存占用降低 **4.6 倍**，而**工具结果缓存**则可减少 **35.2%** 的重复搜索调用。
- **Linear 和 turbopuffer 表明，向量基础设施正逐渐进入非搜索类的关键路径**：[@turbopuffer](https://x.com/turbopuffer/status/2090091547585065283) 表示，Linear 已将其 **delta sync 读取路径**从 **Postgres** 迁移到 **turbopuffer**，利用属性索引处理权限过滤，并让规模最大的同步任务耗时减少了约 **8 秒**。

**Google、OpenAI、Anthropic 与产品化竞赛**

- **Gemini 3.7 Flash 在评测和产品集成两方面都表现出色**：[ @_philschmid](https://x.com/_philschmid/status/2090063976872751408) 和 [@NewsFromGoogle](https://x.com/NewsFromGoogle/status/2090120394141266141) 强调，**Gemini 3.7 Flash** 在 Artificial Analysis 的 **AA-AnalystAgent** 评测中排名 **#1**，在 **80 项以电子表格和文档为主的定量任务**中取得了 **60.0% pass^5**、**70.5% pass@1**、**77.5% pass@5** 的成绩，平均每项任务耗时 **1.32 秒**，平均成本为 **$0.54**。Google 还将它进一步整合到更多产品中，包括 [Gemini chat 和 Spark](https://x.com/Google/status/2090113238436315618)、在 AI Mode 中根据搜索内容即时生成的 **基于 Search 的交互式模拟**（[示例](https://x.com/rmstein/status/2090177397006168437)），以及用于构建工作流的 [AI Studio GitHub 同步](https://x.com/GoogleAIStudio/status/2090149753312932026)。
- **OpenAI 正进一步强调低成本部署和隐私定位**：[ @Replit](https://x.com/Replit/status/2090076648276185555) 推出了由 **GPT-5.6 Luna** 驱动的 **Free Mode**。[@kimmonismus](https://x.com/kimmonismus/status/2090111297039765703) 认为，这体现了显著的效率提升：一款不久前还可能被视为 SOTA 的模型，如今成本已经低到可以广泛免费提供。在企业服务方面，[@OpenAI](https://x.com/OpenAI/status/2090165328290701800) 推出了 **Private Safety Processing**，目标是在继续为 frontier models 提供 **Zero Data Retention** 的同时，无需人工查看底层内容，也能识别跨交互产生的安全风险。
- **Anthropic 仍在持续优化开发者体验**：除了上文提到的 managed-agent 更新外，[@ClaudeDevs](https://x.com/ClaudeDevs/status/2090245922685063634) 还为 Claude Code 增加了 **Concise 输出风格**。这再次表明，产品团队如今不仅在提升能力，也开始把响应形式本身作为一项重要的 UX 变量来调校。

**热门推文（按互动量排序）**

- **Ornith-1.5 发布**：[@ornith_](https://x.com/ornith_/status/2090074077084127302) 发布了一个采用 **MIT 许可**的开放模型家族，参数规模覆盖 **9B 到 397B**，并声称在 coding/agentic 基准测试中表现强劲，同时支持广泛的量化格式。
- **OpenAI 隐私与安全基础设施**：[@OpenAI](https://x.com/OpenAI/status/2090165328290701800) 宣布推出 **Private Safety Processing**，同时再次确认 frontier models 支持 **Zero Data Retention**。
- **Gemini 学生推广和产品组合**：[@GeminiApp](https://x.com/GeminiApp/status/2090165248196252003) 面向全球学生提供一年的 Gemini 套餐，并同步推出新的学习功能。
- **Claude Code UX 更新**：[@ClaudeDevs](https://x.com/ClaudeDevs/status/2090245922685063634) 发布了 **Concise mode**。这是一次规模不大、但受到广泛关注的改进，可提升日常 coding-agent 交互体验。
- **OpenRouter 被收购**：[@patrickc](https://x.com/patrickc/status/2090125021910020520) 确认 **OpenRouter 将加入 Stripe**。许多人将此解读为一种认可：**token routing/marketplaces** 正逐渐成为核心基础设施，而不再只是边缘工具。


---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Qwen/DeepSeek 开放权重推理加速

  - **[Introducing Qwen3.8-27B Dynamic v3 Unsloth GGUFs](https://www.reddit.com/r/LocalLLaMA/comments/1vsr67c/introducing_qwen3827b_dynamic_v3_unsloth_ggufs/)**（活跃度：1428）：**图片是一张技术公告图，介绍 **“Dynamic v3.0 Qwen3.8”**，展示了 Unsloth 新推出的 **Qwen3.8-27B Dynamic v3 GGUF** post-training quantization 版本，并宣称：与其他提供方相比，在相同 GGUF 大小下，top-1% 准确率提高了 **`>10%`**。图片中还提供了一张内存占用表，显示该模型从 **约 `8GB` RAM** 即可运行的 1-bit quants，一直到 BF16 版本均有覆盖；同时还配有一张对比不同量化规模准确率的图表。帖子链接了 Hugging Face 上的 GGUF 发布页面（[Hugging Face](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)）、[Dynamic 3.0 文档和基准测试](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs)，以及[图片原文件](https://i.redd.it/it09zxtsxckh1.jpeg)。Unsloth 强调，这些仅是 **post-training quantization** 版本，*“we do NOT use QAT or QAD”*；同时表示，imatrix 校准文件已公开，方便用户进行独立评测和 fine-tuning 实验。**评论整体较为积极，但有一条技术建议希望 Unsloth 将此前的 **UD 2.0** quants 加入图表，以便用户与自己本地已有的版本进行比较。另一位评论者则希望看到更深入的诊断数据，尤其是按类别统计的结果，以及 **KV-cache quantization KLD** 数值，并提到了类似 localbench 的报告方式。**

    - 几位评论者希望看到针对新版 **Qwen3.8-27B Dynamic v3 Unsloth GGUFs** 更详细的量化评测，尤其是增加一条与之前 **Qwen 3.8 27B UD 2.0** 量化版本的直接对比曲线。有人建议加入 **KLD** 和/或 **top-1 agreement** 等指标，帮助用户判断新版动态量化是否比许多人已经存放在本地的旧版本有实质性提升。
    - 有评论者参考 [localbench.substack.com](https://localbench.substack.com/) 上的分析形式，希望看到**按类别统计的 KLD**以及 **KV-cache 量化 KLD**。这样的拆分能让量化质量讨论更具实用价值，明确展示在不同 GGUF 量化格式下，哪些基准测试/任务类别或缓存量化设置的性能退化最明显。
    - 大家也很关注这些量化版本的实际内存占用：一位用户提到 **Q4_K_M 约需 `~15 GB`**，另一位用户则推测 **IQ4_XS 现在可能已经可以在 `16 GB` 显存中运行**，并且“无需 mtp”。这里的技术关键在于：这些更小的量化格式能否在模型质量基本不受影响的情况下，让用户在常见的消费级 GPU 上完整运行 27B 级别模型。

  - **[在 4 张 RTX 3060 12GB 上以约 100 tok/s 的提示词处理速度运行 DeepSeek V4 Flash Q4_K_XL](https://www.reddit.com/r/LocalLLaMA/comments/1vrqf4f/running_deepseek_v4_flash_q4_k_xl_at_100_toks/)**（活跃度：1104）：**[图片](https://i.redd.it/lav8jwie65kh1.jpeg)展示了一台 DIY 开放式多 GPU 本地 AI 设备，与标题中“运行 **DeepSeek-V4-Flash-0731 UD-Q4_K_XL GGUF**”的说法相符。该模型是一个约 `144 GiB` 的 MoE 量化版本，运行环境为 **4 张 RTX 3060 12GB** 和 llama.cpp。该帖子的技术价值在于其非常规的内存与布局策略：`-ncmoe 34` 将较早的 MoE experts 保留在系统内存中，`-ot` 将后续的 expert blocks 固定分配到 GPU 1–3，而极端的 `-ts 100,1,1,1` 则把大多数非 expert tensors 及与 KV 相关的内存分配推到 GPU0，从而在配置为约 `368k` 上下文、使用 Q8_0 KV cache 的情况下，实现约 `99.4 tok/s` 的 prefill 和 `10.1 tok/s` 的 decode。**评论大多关注设备本身，而不是基准测试：有人开玩笑说，这套裸露的 4-GPU、转接线和 850W 电源的配置“绝对够硬核”，还建议把它收录进假想的 `r/crackheadlocalai`。

  - **[DFlash 2 已支持 Qwen 3.8 27B 和 Muse Glimmer](https://www.reddit.com/r/LocalLLaMA/comments/1vs2tsn/dflash_2_available_for_qwen_38_27b_and_muse/)**（活跃度：569）：****DFlash 2** 已由原始 DFlash 作者发布，支持 **Qwen 3.8 27B** 和 **Muse Glimmer**；相应的 **GGUF 量化版本**已经 उपलब्ध，同时还有对应的 [`llama.cpp` PR #27342](https://github.com/ggml-org/llama.cpp/pull/27342)。据称，帖子中链接的 Qwen 3.8 27B 基准图显示，**DFlash 2 的表现大幅优于 MTP**；至少有一位评论者确认，它可以与 **Qwen 3.8 的 8-bit 量化版本**成功运行。**评论者总体上对这次发布感到兴奋，但也有人指出一个技术限制：**tensor split 似乎尚未支持**，会在 `ggml-backend-meta.cpp:543` 触发 `GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0)`。

    - 有评论者指出帖子中发布的 **Qwen 3.8 27B** 基准图，并注意到从图中数据来看，**DFlash 2 似乎大幅优于 MTP**：https://preview.redd.it/oqmkebcmd7kh1.png?width=645&format=png&auto=webp&s=02fe2114c582819309247b2b45da07f109e4d961。该讨论没有提供具体数值，但核心技术结论是：与 MTP 相比，DFlash 2 在这款模型上的 speculative/accelerated decoding 性能有明显提升。
    - 一位用户表示，已经成功让 DFlash 2 配合 **8-bit 量化的 Qwen 3.8** 运行，说明该版本至少可以用于量化部署。另一位用户则反馈，**tensor split 尚未支持或目前存在问题**，运行时触发了 `llama.cpp`/GGML 断言：`GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0) failed`，位置在 `ggml-backend-meta.cpp:543`。这表明它与 split-axis backend metadata 的处理存在兼容性问题。

  - **[Qwen3.8-27B on 2x 3090 + vLLM + DFlash2: 218 tok/s single request](https://www.reddit.com/r/LocalLLaMA/comments/1vsccit/qwen3827b_on_2x_3090_vllm_dflash2_218_toks_single/)**（热度：395）：**一位用户表示，他使用 **2× RTX 3090**、**vLLM `v0.26.1rc1`**、**AutoRound INT4** 量化（`group 128`）以及 **DFlash2** 草稿模型运行 **Qwen3.8-27B**，通过 [Club-3090 canonical bench suite](https://github.com/noonghunna/club-3090/pull/1056) 测得单请求解码速度：叙事文本为 `120.1 tok/s`，代码为 `218.3 tok/s`。其他指标包括：`10k` 上下文下预填充速度为 `1342 tok/s`，`90k` 下为 `628 tok/s`；推测解码使用 `7` 个草稿 token，平均接受长度为 `3.35`，接受率为 `47.8%`；峰值显存为每张卡 `22.3 GB`，上下文上限为 `131k`。此外，帖子还在 [oceanplexian/vllm#1](https://github.com/oceanplexian/vllm/pull/1) 中提供了自定义 vLLM 启动修复方案。**置顶评论大多与技术无关；唯一有实质内容的澄清问题是询问所使用的模型和量化方式，原帖已明确说明为 Qwen3.8-27B 和 AutoRound INT4。**

    - 一位评论者分享了一个可运行的双 AMD `R9700` vLLM 配置，使用 **vllm-radiance**（[Docker 镜像](https://hub.docker.com/r/stilldeadcode/vllm-radiance/)）、`Qwen/Qwen3.8-27B-FP8`、`--tensor-parallel-size 2`、`--quantization fp8` 和 `--max-model-len 262144`，并启用了基于 ROCm AITER 的统一注意力。他表示，普通文本生成速度约为 `40-60 tok/s`，借助 MTP 的代码生成速度约为 `80-120 tok/s`，长上下文预填充峰值最高可达 `13k tok/s`。
    - 该启动配置使用了多个 ROCm 专用调优参数，例如 `VLLM_ROCM_USE_AITER=1`、`ROCM_AITER_UNIFIED_ATTN`、`RADIANCE_DYNAMIC_DRAFT=1`、`RADIANCE_AR_QUANT=1`，并通过 `--speculative-config '{"method":"mtp","num_speculative_tokens":8,...}'` 启用推测解码。评论者说明，模型量化格式为 `fp8`；KV cache 看起来使用的是 `16-bit`，不过他表示 `fp8` KV 量化似乎也能正常工作。

  - **[Alibaba's RISC-V CPU, XuanTie C950, Runs Qwen-3.8 27B at 30 tps](https://www.reddit.com/r/LocalLLaMA/comments/1vs0wsl/alibabas_riscv_cpu_xuantie_c950_runs_qwen38_27b/)**（热度：718）：****据 [Wccftech](https://wccftech.com/alibabas-tsmc-built-5nm-risc-v-chip-xuantie-c950-now-runs-qwen-3-8-27b-model-natively-unlocking-massive-vertical-integration-tailwinds/) 报道，Alibaba 的 64 核 RISC-V XuanTie C950 可以原生运行 **Qwen-3.8 27B**，解码速度达到 `30 tokens/s`，首 token 延迟（TTFT）为 `1.9s`。据介绍，这款采用 TSMC 5nm 工艺、面向服务器的 CPU 使用了基于 AMBA CHI 的 8 核集群，并配备向量/矩阵加速、可配置缓存、预取机制、8-wide 解码和 16 级流水线，目标是成为不依赖 GPU 的推理平台，服务于 Alibaba 自有的 Qwen 技术栈。**评论者对仅凭 `30 t/s` 解码速度进行评测持怀疑态度，要求补充 **上下文长度扩展表现**、**预填充吞吐量**以及所使用的**量化格式**。也有人认为，如果 Alibaba 将其作为产品推出，这类硬件有望成为更便宜、类似 DGX Spark 的私有部署或边缘推理设备。

    - 多位评论者指出，如果缺少 **上下文长度扩展表现**、**预填充吞吐量**以及 Qwen-3.8 27B 所使用的**量化格式**，单独报告 `30 tokens/s` 的解码速度不足以评估真实的 LLM 性能。关键问题在于，解码速度看起来可能足够实用，但预填充以及更长上下文带来的 KV cache 压力，可能会显著降低实际吞吐量。
    - 有评论者关注将 XuanTie C950 系统定位为低成本的本地 AI 设备，作为类似 NVIDIA DGX Spark 的替代方案。不过他们强调，对于 `27B` 模型而言，**RAM 容量和带宽**至关重要。即使解码速度达到 `30 tps`，系统是否实用仍将很大程度上取决于内存容量、量化级别，以及平台能否在处理更长上下文时保持稳定性能而不受瓶颈限制。


### 2. 推理轨迹与 Scaling Laws

  - **[不要把中间 Token 拟人化：Qwen3.8 并没有“想太多”](https://www.reddit.com/r/LocalLLaMA/comments/1vsjcf7/stop_anthropomorphisizing_intermediate_tokens/)**（热度：862）：**这篇帖子认为，LLM 生成中间 Token，并将其包装成“思考”或“推理”，与其理解为类似人类的逐步认知过程，不如视为对 **prompt/context 的增强**。帖子引用了 [Kambhampati 等人](https://arxiv.org/abs/2504.09762)的研究以及相关的 [OpenReview 论文](https://openreview.net/forum?id=gDE7YcRC3F)。其中引用的研究显示：最终答案是否正确，与推理轨迹是否有效之间的相关性很弱；使用经过**破坏或语义无关的推理轨迹**训练的模型，性能相当甚至更好；RL 能提升答案准确率，却不一定能稳定提升推理轨迹的有效性；此外，推理轨迹的长度基本不受问题难度影响。这些现象都说明，不应把中间轨迹视为语义上忠实的“推理”过程。**评论者反驳说，“思考”和“推理”之类的词是有用的计算隐喻，就像人们把已删除文件的存储位置称为“回收站”一样，并不意味着模型真的具备人类认知。另一位评论者在技术层面同意，推理轨迹主要是供模型探索自身的概率分布，而不是供用户阅读；但他认为，用命令式的方式纠正术语，在修辞上反而适得其反。

    - 几位评论者认为，**“思考”“推理”“幻觉”和“想太多”**等词是有用的计算隐喻，并不等同于字面意义上的拟人化描述。最技术性的说法是，Qwen 所谓的“想太多”，应理解为**在测试时使用了过多计算资源，或在中间推理轨迹上消耗了过大的 Token 预算**，而不是某种类似人类的认知状态。
    - 一位评论者区分了面向用户的拟人化理解与模型的实际实现：中间推理轨迹*“不是给用户看的”*，但可以让模型在生成答案前更充分地探索其学到的概率分布。按照这种观点，类似 chain-of-thought 的 Token 是一种推理时机制，用于改善采样或搜索质量，而不能据此证明模型真的在进行内在思考。
    - 一位评论者指出，即便是人类推理，也未必是一个理想的对照案例，因为人类常常先凭直觉得出结论，再在事后构建出一套合理化解释。这使得“LLM 的推理”这一说法不能仅仅因为模型生成 Token 的方式不同于理想化的人类推理，就被判定为无效。

  - **[关于 Scaling Law 的一些想法 - Z.ai](https://www.reddit.com/r/LocalLLaMA/comments/1vsf9eg/thoughts_about_scaling_law_zai/)**（热度：672）：**这张[图片](https://i.redd.it/mpu6o0zi7akh1.png)是 **Z.ai 创始人、清华大学教授 Jie Tang 在 X 上发布的帖子**“Thoughts About Scaling Law”的截图。帖子认为，前沿模型的 Scaling 已经不能简单归结为参数量增长：最优资源分配取决于**数据、训练计算量、推理成本、稀疏性/MoE 激活规模，以及 post-training/RL**。帖子将 **GLM-5.3** 描述为针对 **GLM-5.2** 的一项受控实验：两者拥有**相同的基础架构、总参数量和激活参数量**，但 GLM-5.3 额外进行了大约**一个月的长周期环境训练和 RL**，并据称取得了大幅提升。这说明，与其继续扩大模型规模，不如调高 post-training 的投入。帖子还提到了从 **Kaplan 式的参数量优先 Scaling** 转向 **Chinchilla 式的 Token 与参数平衡**，并进一步认为，MoE 模型将“知识容量”与“推理深度”解耦，因此，对于漏洞发现这类高度依赖推理的领域，激活参数量、有效深度和针对具体任务的训练变得更加重要。**评论者普遍认为，这说明包括 **Z.ai/GLM** 在内的中国实验室正在开展严肃的前沿研究，而不只是对西方模型进行蒸馏。一位更偏技术视角的评论者将这一观点与小模型架构联系起来：如果把“世界知识”外置到计算图之外，也许可以利用由 RAM 支持的知识存储，让约 8–9B 的模型将更多 VRAM 和计算资源用于推理；不过，这类系统要实现大规模服务，可能会比较困难。**

- 一些评论者将 **GLM 5.3** 解读为一次 scaling 实验：不再单纯堆叠参数规模，而是把部分能力转移到 *reasoning compute* 上。他们以 **Qwen 3.8 27B** 为例，认为相对较小的模型也能凭借更高强度的推理取得出色表现。还有人猜测，**GLM 5.5** 可能会采用接近 **DeepSeek V4 Pro** 的规模，重点放在以较低成本实现接近前沿水平的性能，而不是一味追求更大的参数量。
- 有一篇较为详细的讨论，探讨了如何为 **Llama 8B** 重新设计 **DeepSeek 风格的“Engram”架构**，将相对静态的“世界知识”与核心计算图分离。评论者认为，这种方式相当于用 **system RAM 换取 VRAM**：一个约 `8B–9B` 的模型可以将活跃模型保存在 VRAM 中，同时把完整的外部知识表存放在大约 `32GB` 的 RAM 里，从而让较小模型的参数更多地用于计算和推理。
- 同一位评论者认为，这种将知识外置的方案可能会降低激进量化带来的损失，因为受 `Q4` 这类量化影响最大的知识部分，可以继续保存在 RAM 中的高精度表里，例如使用 `FP16` 或 `FP8`。他们还提出，在 **DDR5** 上，知识提取路径显然不一定会受 **PCIe** 带宽限制。不过他们也指出，要大规模部署这种架构，在运维层面会非常困难：需要在 RAM 中的哈希表与 VRAM 限制之间进行平衡。这或许可以解释为什么 **DeepSeek V4** 可能不会广泛采用这种部署方式。

### 3. Qwen Next 中等规模模型预告

  - **[社区经理称，全新的中等规模 Qwen 3.8 模型将于下周推出（希望如此）！](https://www.reddit.com/r/LocalLLaMA/comments/1vs9zym/new_midsize_qwen_38_model_coming_next_week/)**（活跃度：1025）：**据称，一名 **Qwen** 社区经理在 Qwen Ambassador Discord 中表示，一款全新的**中等规模开放权重模型**计划于“下周（希望如此）”发布。由于日程安排紧张，这次将**不会提供提前体验**。目前尚未公布具体规模或架构，但评论者推测其规模可能会明显大于 `35B`，有人猜测可能是 `80B` 的 coder 版本，或是约 `100B`、属于 Qwen 3.8 级别的模型。**评论者认为，这款预告中的模型可能比传闻中的 `35B A3B` 更值得关注，并引用团队的一句话：“35B A3B 不是那个值得等待的模型。”还有人推测，约 `100B` 的 Qwen 模型或许能与“dsv4F”展开有力竞争，但目前没有提供任何基准测试或技术细节。

    - 评论者主要讨论了下一款 Qwen 模型理想的参数规模，提出或猜测了**`80B` Qwen Coder**，以及**`100B–122B` 的 Qwen 3.8 级别模型**。一名用户认为，全新的具备 Qwen 3.8 能力的**`122B`**模型将达到实用上的*“速度与世界知识的甜蜜点”*，这表明大家期待一种保持广泛知识能力、同时又比前沿规模模型更快的稠密或中等规模模型。
    - 一条被引用的团队/社区评论称：**“……35B A3B 不是那个值得等待的模型……”**。这暗示传闻中的**`35B A3B`**可能不是即将发布的主要产品，或者其重要性不如更大的模型。讨论中的技术猜测主要围绕这样一个问题展开：更大的 Qwen 3.8 模型能否在编程和综合能力上与 **DeepSeek V4/F 风格**的模型竞争，甚至超越它们。

  - **[Qwen-3.8-35B-A3B？可能不是……来自 Qwen 联合作者的神秘回复。](https://www.reddit.com/r/LocalLLM/comments/1vrmt8e/qwen3835ba3b_maybe_not_cryptic_reply_direct_from/)**（活跃度：521）：**这张[图片](https://i.redd.it/awxrvkngf4kh1.png)展示了一张 X/Twitter 回复截图。**Qwen 联合作者 Shuai Bai**在回复一名用户关于 `Qwen-3.8-35B-A3B` 是否会面向硬件有限的用户发布的问题时，留下了一句意味深长的评论：“*35B-A3B 可能不是那个值得等待的模型 👀*”。这暗示这款传闻中的 MoE 风格模型可能并不是即将发布的重点，甚至可能是在暗示另一款更小、或更容易运行的 Qwen 模型。**评论者据此认为 `35B-A3B` 可能不会发布，同时猜测 Qwen 或许会改为推出 `9B`、`20B`、`30B-A3B`，或更小的稠密模型。排名靠前的一条评论开玩笑地提到了一个不可能用于智能手表的 `2.3T A200M` 模型，因此这条讨论中也包含一些与技术无关的幽默内容。

    - 评论者将 Qwen 联合作者所说的*“不要等待这个模型”*解读为：传闻中的**Qwen-3.8-35B-A3B**很可能不会发布，但这并不排除 Qwen 推出其他新模型的可能。技术猜测主要集中在其他参数规模或 MoE 配置上，例如**30B A3B**、**9B**、**20B**，或**12B 稠密模型**。这意味着大家期待的可能是一个规模更小、或结构不同的 checkpoint，而不是传闻中的 `35B-A3B`。
    - 有评论者猜测，Qwen 可能正在准备一款更大的稀疏 MoE 级别模型，被描述为**“122B/a10 级别模型”**。这意味着该模型的总参数量大约为 `122B`，每个 token 激活的参数量约为 `10B`；它的总容量将高于传闻中的 `35B-A3B` 级别，同时保留稀疏推理的特点。




## AI 子版块技术含量较低的回顾

> /r/Singularity、/r/Oobabooga、/r/MachineLearning、/r/OpenAI、/r/ClaudeAI、/r/StableDiffusion、/r/ChatGPT、/r/ChatGPTCoding、/r/aivideo、/r/aivideo


### 1. 本地编程模型挑战 Claude Code

  - **[游戏结束。现在，Pi 上运行的 22GB 本地模型，在训练截止日期之后发布的真实编码任务中，已经超过 Claude Code Opus 5 High](https://www.reddit.com/r/ClaudeCode/comments/1vrqxqc/game_over_22gb_local_models_run_in_pi_now/)**（热度：2485）：**这张[基准测试信息图](https://i.redd.it/vj21oxtc75kh1.png)声称，使用作者提供的 [Sharp chat templates](https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates) 的本地运行 **Qwen3.x GGUF coding agents**，在 `21` 个训练截止日期后发布、类似 SWE-bench-Live 的真实代码任务中，表现超过了 **Claude Code Opus 5 High**：原版 `Qwen3.8-27B Q6` 修复了 `12/21` 个问题，Sharp `Qwen3.8-27B` 在约 `20 分钟`内修复了 `11/21` 个，而 Opus 5 High 修复了 `10/21` 个，Sonnet 5 修复了 `5/21` 个。其技术观点是，调整 prompt/chat template 可以在保持修复质量的同时，减少 token 使用量和延迟；帖子还将链接的本地模型 [Dirk-Qwen3.8-27B-GGUF](https://huggingface.co/peculiar-ragdoll/Dirk-Qwen3.8-27B-GGUF) 和 [Nail-Qwen3.6-35B-A3B-GGUF](https://huggingface.co/peculiar-ragdoll/Nail-Qwen3.6-35B-A3B-GGUF) 作为实用的本地 coding-agent 替代方案。**评论区则充满怀疑，甚至有阴谋论式的猜测：有人声称 Anthropic 可能暗中降低了 Opus 的质量，也有人认为，本地模型目前最大的障碍仍然是运行速度，以及价格可接受的 `22GB+` VRAM 硬件。

    - 评论者指出，**22GB 本地模型可能在基准测试中表现出色，但实际使用时往往受推理延迟限制**。一位用户用 *“本地模型慢得离谱”* 概括了部署中的核心问题。质疑的重点并不是原始任务准确率，而是实际吞吐量和使用体验：本地模型不仅要在编码质量上追上托管式企业模型，还需要在**速度、成本和普通消费者能够买得起的硬件要求**方面达到相当水平。
    - 一个反复出现的技术性保留意见是**硬件的价格和可获得性**，具体来说，用户是否能以较低成本买到拥有约 `22GB` VRAM 的 GPU。有几位评论者暗示，在这些模型能够在**面向消费者的价格水平的硬件**上以可接受的速度运行之前，称其“游戏结束”还为时过早，而不能只看它们在高显存本地设备上的表现。
    - 有一位评论者声称，**Anthropic** 在后台“悄悄把 Opus-5 换成了 Sonnet-5”，但该帖子没有提供任何证据或基准测试细节。就目前的表述来看，这一说法与模型对比结果是否有效有关，但它仍然只是未经证实的断言，而不是已经记录在案的路由变化或评测问题。

  - **[发生了什么……](https://www.reddit.com/r/ClaudeAI/comments/1vs4ntq/what_is_happening/)**（热度：1053）：**一位资深工程师表示，自己接手的是由 AI 生成的任务，这些任务属于一个由 AI 构思的项目，而项目文档也是由 AI 编写的，几乎无法使用。随后，他在缺乏明确系统或产品规格的情况下，借助 AI 完成了 `3` 个 PR，每个大约包含 `20,000` 行代码（LOC）。核心技术问题在于，组织过度依赖 **Claude** 和 **ChatGPT** 等 LLM 来完成需求挖掘、代码生成、文档编写和代码审查，最终形成了一条流水线：没有任何人真正拥有对正在构建的系统的完整理解。**高热度评论质疑，在不了解产品的情况下提交 `60,000` 行代码的 PR 是否合理，以及这样的流程由谁负责治理；其中一位建议先拿到真正的产品规格说明。另一位则建议用 Claude 生成文档，再用 ChatGPT 审查文档，这反映出讨论中的矛盾：一方面，人们把 AI 当作弥补工程流程缺失的临时手段；另一方面，这种做法也可能让问题进一步恶化。

    - 多位评论者关注 **一天内提交 `3` 个、每个约 `20,000` 行代码（LoC）的 PR** 所暴露出的工程流程问题：这种规模的改动实际上无法有效审查，也说明产品规格可能缺失，或根本没有得到重视。一条评论明确建议，在接受或处理这种规模的代码之前，应该先拿到产品规格说明。
    - 一个反复出现的技术担忧是：让 AI 生成过多代码，可能会导致系统的整体结构不再有清晰的人类心智模型。评论者指出，这会形成一种维护困境：因为代码“已经不再符合直觉”，今后的修改也只能继续依赖 AI。

  - **[使用 AI 完全制作我的钓鱼游戏：第 3 周](https://www.reddit.com/r/ClaudeAI/comments/1vrjryf/week_3_of_making_my_fishing_game_entirely_with_ai/)**（活跃度：2740）：**作者正在使用 **Godot `4.7.1`**、支持 MCP 的 **Claude/Claude Code**，以及专门的 **Blender MCP** 会话，通过脚本生成低多边形 3D 资产；同时使用 **ChatGPT/OpenAI Playground** 制作概念图和参考图，从而“完全借助 AI”开发一款钓鱼游戏。目前的成果已发布为 [Artifact 1](https://claude.ai/code/artifact/1c9c24a7-129f-4ccb-b453-d7aed9e5c412) 和 [Artifact 2](https://claude.ai/code/artifact/2c524964-0cf4-43f6-bd4d-9fdffe04806e)。整个流程将工作分为参考图生成、程序化几何建模、整合与摆放、镜头内截图检查、盲测前后对比评分，以及带来源标记的设计决策，从而避免 AI 幻觉再次被当成需求引入。最具技术性的批评指出了 AI 生成游戏世界中常见的问题：无法进入的门和码头、不合逻辑的建筑布局与地形切口、杂乱且锚定位置不佳的 UI、Shader 或多边形数量过高可能导致的掉帧、缺少插值而显得突兀的水波环动画、*“Pull out the common”* 和 *“The boat remembers the route”* 这类文案问题、范围过大的 `18` 个改装槽位，以及过于夸张的昼夜光照和灯塔朝向。**评论者普遍认为，AI 已经让独立制作游戏变得容易得多，因此作品能否脱颖而出，将不再主要取决于生产速度，而更多取决于有意识的人为审美、反复迭代，以及能够消除“AI 生成感”的打磨。一位评论者还建议将这些批评直接反馈到制作流程中，并强调：资产在预告片或远景截图中可能看起来不错，但在近距离实际游玩时，可能会因为空间关系不合理和交互提示不足而暴露问题。**

    - 一份详细的批评指出，AI 生成的资产中存在一些可能影响游戏可读性的常见问题：建筑的门无法进入或被堵住、几何结构彼此融合、码头被围栏挡住、NPC 悬浮在空中、楼梯通向虚空，以及岛屿和灯塔缺少合理的停靠位置或居住布局。技术层面的结论是，AI 生成的环境在预告片中可能很有表现力，但仍需要人工检查和修改 **空间逻辑、导航交互提示、碰撞与合理性，以及面向玩家的可读性**。
    - 评论者还发现了几处 UI/UX 问题：`purse` 和 `debt` 两个数值可能是重复信息，或许可以合并为一个带正负号的货币数值；导航信息被分散在屏幕两侧；右侧组件的锚定位置似乎不正确；而“Harbour Roads”面板被认为文字过于密集，不适合在即时游玩过程中阅读。评论者建议将非必要文字放入 `I` 详情面板，并把 `Mark/Arrives` 等相关导航指标与 `KN` 速度信息放在一起。
    - 水面与钓鱼流程中出现了疑似性能问题或录制卡顿，因此评论者建议在继续添加更多系统前，先分析 **Shader 开销和 3D 模型的多边形数量**，因为后续迭代很可能会进一步加重延迟。他们还注意到一个视觉问题：水波纹内部的圆环会突然“凭空出现”，建议加入插值；此外，他们也指出了 *“Pull out the common”* 这类不自然的生成文案，以及 *“The boat remembers the route”* 这类带有 Claude 风格、将物体拟人化的表达。

### 2. Frontier AI Safety 和 Bio Design

  - **[说到做到：Anthropic 的 Claude 自主设计出针对疾病的蛋白质，并通过真实湿实验验证，成功率达 35%，而人类平均水平为 10–15%](https://www.reddit.com/r/singularity/comments/1vs524y/putting_money_where_their_mouth_is_anthropics/)**（Activity: 1152）：****Anthropic** 报告称，[Claude 能够自主生成针对疾病的蛋白质设计方案](https://www.anthropic.com/research/Claude-accelerates-protein-design)，并已通过湿实验进行验证。据称，其成功率达到 `35%`，而引用的人类蛋白质设计基准约为 `10–15%`。这里的关键技术成果并不只是计算机模拟评分，而是**经过真实实验验证**，这表明 Claude 可能会成为治疗性蛋白质工程流程中的迭代式设计引擎。**评论者大多对其相较人类平均水平超过 2 倍的表现印象深刻，有些人甚至进一步联想到癌症治疗；在提供的热门评论中，实质性的技术讨论并不多。

    - 一位评论者指出，如果比较方法公平，那么报告中的**湿实验成功率 `35%` 对比人类平均水平 `10–15%`**，意味着性能提升了**超过 2 倍**。不过，他们也提醒说，这种关系未必是线性的，而且不同任务之间可能无法直接比较。

  - **[@sama 解释暂停 RL training：“当前模型进展极其迅速，我们一直说过，如果认为模型能力的发展速度已经超过了安全和 alignment 的推进速度，就会采取行动。”](https://www.reddit.com/r/singularity/comments/1vrz27g/explanation_from_sama_on_rl_training_pause_model/)**（Activity: 893）：**该帖子引用 **Sam Altman（@sama）** 的说法，将**暂停 RL training**解释为一项安全与 alignment 方面的门控决策：*“当前模型进展极其迅速……我们一直说过，如果模型能力的发展速度超过了安全和 alignment 的推进速度，就会采取行动。”* 一位评论者还附上了[更新截图](https://preview.redd.it/kcioru1iz6kh1.png?width=629&format=png&auto=webp&s=841a7a205ac448d90541da816b97d8d529915833)，但帖子节选没有提供截图内容。**评论者的观点分成两派：一派持加速主义立场，对暂停训练表示怀疑，例如提议成立一家专注于 RSI/AGI/ASI 的前沿实验室；另一派则担心，真正的风险拐点可能会出现在未来的硬件范式出现之后，因为新的硬件可能在约 5 年内带来 `100×–1000×` 的更快或更高效的扩展能力。

    - 一位评论者认为，真正的安全拐点可能会随着下一次重大硬件范式转变而到来，而不是出现在当前的 RL training 中。他指出，在大约 `<5 years` 内，模型效率、速度或规模可能提升 `100x`–`1000x`。在他看来，推动这一变化的主要动力是经济利益，而不只是研究需求：通过多种硬件和效率优化方案降低 AI 能耗、提升吞吐量，背后存在“万亿美元级的激励”。

### 3. AI Industry 的权力、政策与经济学

  - **[记者在 Amazon 仓库里偷偷放入 AirTag，证明他们会销毁用于训练 AI 的珍稀书籍](https://www.reddit.com/r/ChatGPT/comments/1vro3fa/journalists_slip_an_airtag_into_an_amazon/)**（Activity: 3803）：**据 *404 Media* 的调查报道，记者将一个 **Apple AirTag** 藏在一本通过批量订单售出的珍稀书籍中，并追踪到它最终被送往**拉斯维加斯的一处 Amazon AI training 设施**。这为相关说法提供了支持：实体书可能会被送入用于 AI training 的数字化及销毁流程。一位书商评论者补充说，在二手书供应链中，大规模粉碎未售出、获捐、库存积压、图书馆淘汰以及无法退回的书籍，本来就是常见做法；其中许多书最终既没有被转售，也没能成功捐出（[评论](https://www.reddit.com/r/books/s/uZWzmUl8x8)）。**评论者争论这篇报道是否真的令人意外：有人要求提供更可靠的来源，另有人声称 Amazon 的销毁流程与“Project Panama”下的法律义务有关，但帖子节选中没有提供支持这一说法的引用。

    - 一位书商认为，在出版业和二手书供应链中，Amazon 销毁图书并不罕见：捐赠图书、图书馆淘汰书、书店未售出的库存，以及尾货仓库中的存货，往往都会因为需求远低于供给而被定期粉碎处理。他们声称，书店里的新季图书中，**大约 `50%` 可能一本都卖不出去**，之后这些库存会根据出版商的指示被退回、作为尾货处理，或直接粉碎。
    - 有评论者断言，这种销毁行为可能与法律或法院强制执行的义务有关，而非自愿行为；他们提到了作为相关机制的 **“Project Panama”**，据称该机制要求销毁图书，但该讨论串中没有提供支持这一说法的来源。
    - 另一个与技术更相关的观点是，被销毁的物品很可能是需求低或大批量生产的材料，例如旧教材或杂志，而不是独一无二的“稀有文化瑰宝”。理由是，真正受欢迎的书籍通常会有多个印次，并通过多个流通渠道传播。

  - **[Big Tech Is Raising Billions To Stop UBI](https://www.reddit.com/r/singularity/comments/1vrasl0/big_tech_is_raising_billions_to_stop_ubi/)**（活跃度：2461）：**该帖声称，美国前商务部长 **Gina Raimondo** 正在领导 **RAISE US**。这是一个新成立、资金雄厚的组织，反对将 **UBI/基本收入作为应对 AI 取代就业的方案**，并引用了她所说的 UBI 会“*像美国的末日一样*”。帖子称，RAISE US 已筹集 **`>$500M`，目标为 `$1B`**；**Amazon、Anthropic、Microsoft 和 OpenAI Foundation** 是其主要合作方，其他支持者还包括 Blackstone、GM、IBM、Mastercard、Deloitte、Cisco、UPS 等。**评论者将没有 UBI 的 AI 驱动“奇点”描述为反乌托邦，并批评 Big Tech 一边可能推动就业自动化，一边出资支持反 UBI 的政策工作。一条较有实质内容的讨论提出，可以采用类似[负所得税](https://en.wikipedia.org/wiki/Negative_income_tax)的 UBI，并指出 **Milton Friedman** 曾支持这一方案；20 世纪 70 年代，美国众议院还曾两次通过类似提案，但最终因左右两派的反对而失败。

    - 一条较有实质内容的政策讨论认为，UBI 可以通过**负所得税**的方式实施，并指出 Milton Friedman 曾在 20 世纪 70 年代倡导过类似方案。据称，该方案曾两次在美国众议院获得通过，但后来因意识形态两翼的反对而失败。评论者强调，与经过经济状况审查的福利制度相比，这种设计具有多项优势：不需要设置那么多官僚化的资格审核规则，也不容易出现福利骤减的“断崖”，同时还能形成这样的激励机制：*“工作或组建双亲家庭，始终都应该带来净收益。”*
    - 多位评论者认为，在自动化程度很高的未来，反对 UBI 的游说在经济上可能适得其反：如果 AI 驱动的裁员导致家庭收入下降，整体消费需求就会减少，于是便会产生这样的问题：*“当没人有钱消费时，谁来买东西？”* 这与其说是一个技术层面的 AI 观点，不如说是对缺乏再分配或收入支持的自动化模式提出的宏观经济批评。

  - **[Anthropic has twice the revenue of OpenAI](https://www.reddit.com/r/ClaudeAI/comments/1vsdx5z/anthropic_has_twice_the_revenue_of_openai/)**（活跃度：1131）：**这张[图片](https://i.redd.it/b8rut9nxq9kh1.png)是《华尔街日报》（WSJ）的一段摘录，称 **Anthropic 的营收翻了一倍以上，达到 `$11.6B`，并实现了小幅经营利润**；与此同时，**OpenAI 的季度营收增至 `$6.7B`，但经营亏损进一步扩大**。从背景来看，该帖想表达的是：尽管 Reddit/X 上流传着用户正在放弃 Claude 的说法，但企业用户或更广泛市场的采用情况可能比可见的消费者舆论更强；不过，标题将其夸大为 OpenAI 营收的“两倍”，因为 `$11.6B / $6.7B ≈ 1.7x`。**评论主要分为品牌和产品观感，以及针对具体工作流程的评价：一些人认为 Claude 在软件开发方面“确实好用”；另一位用户则表示，OpenAI 最近在他们的工作中表现更好，如今只把 Claude 限定用于代码审查，尤其是在对水印问题产生担忧之后。

- 一位同时订阅了 **OpenAI 和 Anthropic 付费/Max 套餐**的用户表示，**OpenAI 已经更适合自己的工作流程**，而 Claude 现在主要用于**代码审查**。他还提到，Anthropic 宣布将加入**水印标记**后，自己的使用方式也发生了变化：不再直接使用 Claude 生成的内容，而是只把这些内容输入给另一个 LLM。
- 一位评论者认为，Anthropic 和 OpenAI 的盈利能力可能会受到 **LLM 商品化**的挤压，尤其是**中国的开放权重模型**正在降低产品差异化。他认为，未来的竞争优势可能不再主要来自前沿 R&D，而会逐渐转向**效率研究、安全性、客户支持、UI/产品集成**，以及可能出现的**硬件**领域。
- 一位企业用户介绍了公司内部的混合部署方案：在内部 AI 工具中同时使用 **GPT 和 Claude**，同时也使用 **VS Code + GitHub Copilot**，并将 Claude 选作模型。他指出，**Microsoft 的工具与开发者工作流结合得非常紧密**，因此即使 Claude 在某些任务上更受偏好，OpenAI/Microsoft 的分发渠道也很难被完全拆分开。