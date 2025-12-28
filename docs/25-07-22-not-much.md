---
companies:
- moonshot-ai
- alibaba
- google
- google-deepmind
- openai
- hugging-face
- vllm-project
date: '2025-07-22T05:44:39.731046Z'
description: '**月之暗面（Moonshot AI）**发布了 **Kimi K2**，这是一个拥有万亿参数的超稀疏混合专家（MoE）模型，采用了 **MuonClip**
  优化器，并使用了包含超过 **20,000 个工具**的大规模智能体化（agentic）数据流水线。不久之后，**阿里巴巴**更新了其 **Qwen3** 模型，推出了
  **Qwen3-235B-A22B** 变体；尽管其体积比 Kimi K2 小 4.25 倍，但在 **GPQA** 和 **AIME** 等基准测试中的表现却超越了
  Kimi K2 及其他顶级模型。阿里巴巴还发布了 **Qwen3-Coder-480B-A35B**，这是一款专为编程设计的 MoE 模型，拥有 100 万 token
  的上下文窗口。


  **谷歌 DeepMind** 推出了 **Gemini 2.5 Flash-Lite**，这是一款速度更快、成本效益更高的模型，在编程、数学和多模态任务上的表现均优于此前版本。目前，MoE
  架构正成为主流，**Mistral**、**DeepSeek** 和 **Kimi K2** 等模型正引领这一趋势。在数学领域，一款先进的 **Gemini**
  模型在**国际数学奥林匹克竞赛（IMO）**中达到了金牌水平，标志着 AI 首次取得此类成就。一位 **OpenAI** 研究员指出，他们的 IMO 模型能够“察觉”自己何时没有得出正确解法，这凸显了模型在推理能力和自我意识方面的进步。'
id: MjAyNS0w
models:
- kimi-k2
- qwen3-235b-a22b
- qwen3-coder-480b-a35b
- gemini-2.5-flash-lite
- mistral-7b
- deepseek-v3
people:
- demishassabis
- rasbt
- alexwei_
- yitayml
title: 今天没发生什么事。
topics:
- mixture-of-experts
- agentic-ai
- model-optimization
- model-training
- benchmarking
- code-generation
- long-context
- multimodality
- math
- reinforcement-learning
- model-architecture
- model-performance
- open-source
- alignment
---

**平静的一天**

> 2025年7月21日至7月22日的 AI 新闻。我们为您检查了 9 个 subreddits、449 个 Twitter 账号和 29 个 Discord 社区（227 个频道，6134 条消息）。预计节省阅读时间（以 200wpm 计算）：527 分钟。我们的新网站现已上线，包含完整的元数据搜索和美观的 vibe coded 风格的往期内容展示。请访问 https://news.smol.ai/ 查看完整的详细新闻，并在 @smol_ai 上向我们提供反馈！

[Qwen 3 Coder（声称达到 Sonnet 4 级别的性能）和 Qwen Code](https://qwenlm.github.io/blog/qwen3-coder/)（Gemini Code 的一个分叉版本）的发布几乎成为了头条新闻，但我们打算再观察一下，看看后续的评论如何。

---

# AI Twitter 综述

**重大模型发布与基准测试：Qwen、Kimi 和 Gemini**

- **Kimi K2 技术报告发布，声称在 Agentic 任务上达到 SOTA**：**Moonshot AI** 发布了 [Kimi K2 的技术报告](https://twitter.com/Kimi_Moonshot/status/1947520758760313170)，这是一个拥有 1 万亿参数的超稀疏 Mixture-of-Experts (MoE) 模型。报告详细介绍了用于稳定训练的 **MuonClip** 优化器、使用超过 **20,000 个工具** 的大规模 Agentic 数据合成流水线，以及联合 RL 对齐方法。[该模型被描述](https://twitter.com/iScienceLuvr/status/1947414667221237904)为具有更高稀疏性的 **DeepSeekV3 风格 MoE**，并且是开源的。此次发布被社区中的一些人称为 [年度最令人振奋的技术报告](https://twitter.com/QuixiAI/status/1947388338337681541)。
- **Qwen3-235B-A22B 挑战 Kimi K2，夺得基准测试榜首**：在 Kimi K2 发布后不久，**阿里巴巴** [更新了其 Qwen3 模型](https://twitter.com/huybery/status/1947345040470380614)，其中 **Qwen3-235B-A22B** 变体夺回了基准测试的桂冠。[Sebastian Rasbt 博士提供了技术分析](https://twitter.com/rasbt/status/1947393814496190712)，指出它比 Kimi 2 小 **4.25 倍**（235B vs 1T 参数），但拥有更多层并使用 GQA 而非 MLA。据报道，该模型在 **GPQA**、**AIME** 和 **LiveCodeBench** 等基准测试中 [击败了 Kimi-K2、Claude-4 Opus 和 DeepSeek V3](https://twitter.com/scaling01/status/1947350866840748521)。它在 **ARC-AGI-1** 上的表现尤为引人注目，在没有推理步骤的情况下得分为 **41%**，[被认为非常出色](https://twitter.com/scaling01/status/1947351789222711455)。这种飞速的进步让 [一位用户感叹道](https://twitter.com/reach_vb/status/1947364340799283539)：“**这将是开源模型有史以来‘最笨’的时刻。**”
- **Qwen3-Coder-480B-A35B 发布，助力高级代码生成**：**阿里巴巴**继续其发布攻势，推出了 **Qwen3-Coder**，这是一个总参数量 **480B**、激活参数量 **35B** 的 MoE 模型，专门用于编程和 Agentic 任务。[该模型具有 100 万 token 的上下文窗口](https://twitter.com/scaling01/status/1947732150872084693)，历时三个月开发完成。[它在 SWE-bench 上表现强劲](https://twitter.com/QuixiAI/status/1947773200953217326)。[在架构上](https://twitter.com/nrehiew_/status/1947770826943549732)，它比基础版 Qwen3 更宽、更浅，拥有 62 层、6144 个隐藏维度和 160 个专家。该模型已在 [Hugging Face 上线](https://twitter.com/ClementDelangue/status/1947780025886855171)，并得到 **vLLM** nightly 版本的支持，可进行 [专家并行（expert parallelism）推理](https://twitter.com/vllm_project/status/1947780382847603053)。
- **Google 发布 Gemini 2.5 Flash-Lite**：**Google** [宣布正式发布 Gemini 2.5 Flash-Lite](https://twitter.com/Google/status/1947689382892204542)，这是其 **2.5** 系列中成本效益最高、速度最快的模型。**Google DeepMind** [表示它比 2.0 Flash 模型更快、更具成本效益](https://twitter.com/GoogleDeepMind/status/1947689582012633542)，同时在编程、数学和多模态理解方面表现更优。
- **MoE 架构成为主流**：最近的发布巩固了 Mixture-of-Experts 作为主导架构的地位。正如 [@hkproj](https://twitter.com/hkproj/status/1947571673021993152) 所总结的：“**Mistral 开启了它，DeepSeek 扩展了它，Kimi K2 证实了它：训练 MoE 总是更方便。**”

**人工智能在数学领域：争夺 IMO 金牌之战**

- **Google DeepMind 的 Gemini 正式获得 IMO 金牌**：**Demis Hassabis** [宣布 Gemini Deep Think 的一个高级版本](https://twitter.com/AndrewLampinen/status/1947370582393425931)在国际数学奥林匹克竞赛 (**IMO**) 中正式获得了金牌水平的分数 (**35/42**)，这在 AI 模型领域尚属首次。
- **模型“知道”其局限性并使用自然语言**：一位 **OpenAI** 研究员 [@alexwei_](https://twitter.com/alexwei_/status/1947461238512095718) 分享了他们自己 IMO 模型表现的一个关键洞察：在它未能解决的 P6 问题上，模型**“知道”自己没有正确的解法**。来自 **Google** 团队的研究员 [@YiTayML](https://twitter.com/YiTayML/status/1947350087941951596) 指出，他们的 IMO 金牌模型是一个即将发布的通用模型，而不仅仅是一个实验性模型。另一位 Google 研究员强调，[Gemini 使用英语端到端地解决了这些问题](https://twitter.com/denny_zhou/status/1947360696590839976)。
- **关于发布时机的争议**：这一成就伴随着关于哪个实验室先发布结果的争议。一些人批评 **OpenAI**，认为他们[“抢跑”了](https://twitter.com/mathemagic1an/status/1947352370037305643)，而另一些人则[质疑“发布竞赛”的整体价值](https://twitter.com/francoisfleuret/status/1947359708811088211)。**Demis Hassabis** [澄清](https://twitter.com/TheZachMueller/status/1947419062423982583)说，Google 尊重 **IMO 委员会**最初提出的推迟发布的请求。

**AI 基础设施、硬件与效率**

- **OpenAI 宣布与 Oracle 合作建设 5GW “Stargate” 数据中心**：在一项重大的基础设施公告中，**OpenAI** 透露正与 **Oracle** [合作开发 **4.5 GW** 的额外 “Stargate” 数据中心容量](https://twitter.com/OpenAI/status/1947628731142648113)，使总容量超过 **5 GW**。位于德克萨斯州阿比林的 **Stargate I** 站点正开始上线。
- **台湾学生推动半导体前沿**：[@dylan522p](https://twitter.com/dylan522p/status/1947716636196409616) 的一条推文重点介绍了一个台湾高中科学展览，学生们在会上讨论了 **1.5nm 全环绕栅极 (GAA) 晶体管结构优化**，表明在先进半导体研究领域拥有深厚的人才储备。他还评论了[中国在 FlipFET 和 3D DRAM 方面的进展](https://twitter.com/dylan522p/status/1947372973645504657)，认为这对于解决内存墙 (memory wall) 问题至关重要。
- **向开放科学捐赠闲置的 GPU 算力**：**Hugging Face CEO Clément Delangue** [提出了一个问题：科技巨头是否可以将其大规模 GPU 集群上的闲置时长“捐赠”](https://twitter.com/ClementDelangue/status/1947379634615816287)给开放科学和开源 AI 开发者，这一建议引起了极大关注。
- **vLLM 与 Hugging Face Transformers 的集成**：**vLLM** 项目宣布[支持 Transformers 开箱即用的视觉语言模型 (Vision-Language Models)](https://twitter.com/vllm_project/status/1947756551663718754)，简化了多模态模型的部署和推理。

**AI 工具、框架与应用**

- **Perplexity Comet 浏览器受到关注**：**Perplexity AI** 的新浏览器 **Comet** 自发布以来，[候补名单已经翻了一番](https://twitter.com/AravSrinivas/status/1947407684996894969)。早期用户反馈表明，它[让传统的聊天界面“显得过时”](https://twitter.com/AravSrinivas/status/1947478934528118887)。CEO [@AravSrinivas](https://twitter.com/AravSrinivas/status/1947501358007128149) 发布的一条询问谁想要一个能处理会议的 Agent 的推文获得了超过 **3,300 次展示**，显示出对其 Agent 能力的浓厚兴趣。
- **LangChain 1.0 发布**：**Harrison Chase** 宣布团队正在[致力于](https://twitter.com/hwchase17/status/1947376920355917909) `langchain` [1.0](https://twitter.com/hwchase17/status/1947376920355917909) 版本。该版本将专注于通过更新文档和基于 **LangGraph** 构建的通用 Agent 架构，成为构建 LLM 应用最简单的起点。他澄清说，**LangGraph** 是一个底层的[“Agent 运行时”](https://twitter.com/hwchase17/status/1947459414279262513)，而 LangChain 将提供更高层的抽象。
- **Anthropic 增强移动端 Artifacts**：**Anthropic** [推出了在移动端使用 Artifacts 的新方式](https://twitter.com/AnthropicAI/status/1947690894888513964)，允许用户直接通过手机创建交互式工具、浏览画廊并分享作品。
- **OpenAI 为肯尼亚的临床 Copilot 提供支持**：**OpenAI** 分享了[与肯尼亚 PendaHealth 合作](https://twitter.com/gdb/status/1947732134430687351)的积极成果。在一项涵盖 **40,000** 次患者就诊的研究中，对由 OpenAI 驱动的临床 Copilot 进行了评估。
- **LlamaIndex 发布开源 RFP 响应 Agent**：**LlamaIndex** 构建了一个[用于自动化征求建议书 (RFP) 响应的全开源 Agent](https://twitter.com/jerryjliu0/status/1947465066892431792)。该应用基于 **LlamaIndex** 框架和 **LlamaCloud** 构建，可处理文档提取、分析和报告生成。

**研究、公司新闻及更广泛的讨论**

- **关于 LLM “潜意识学习”的研究**：**Owain Evans** 及其团队的一篇论文引入了“潜意识学习（Subliminal Learning）”的概念，表明 [LLM 可以通过生成数据中的隐藏信号将特征传递给其他模型](https://twitter.com/_arohan_/status/1947704379110527183)，即使数据与该特征无关。研究表明，这可能是神经网络学习的一个普遍属性。
- **Anthropic 论文发现测试时计算（Test-Time Compute）中的“逆向缩放”**：**Anthropic** 的一篇研究论文发现，在某些情况下，[更长的推理时间会导致准确率下降](https://twitter.com/dilipkay/status/1947677154663403732)。这种效应在 **Opus 4 的 6 个基准测试**中被观察到，引发了关于当前推理模型局限性和 Scaling Laws 的讨论。
- **重大融资和招聘动态**：**OpenAI** 宣布 **Fidji Simo** 将担任[应用业务首席执行官 (CEO of Applications)](https://twitter.com/kevinweil/status/1947345653014691958)。**Reka** 宣布获得来自 **NVIDIA** 和 **Snowflake** 等投资者的 [**1.1 亿美元**融资](https://twitter.com/RekaAILabs/status/1947689320594157668)。有报告指出 **Meta** 正在积极招聘顶尖 AI 研究人员，[薪酬方案在四年内高达 **3 亿美元**](https://twitter.com/DeepLearningAI/status/1947461590283858010)。
- **开源与闭源模型的辩论仍在继续**：**Jack Dorsey** 呼吁推行[“无需许可（Permissionless）”的 AI，以防止少数 CEO 主导创新](https://twitter.com/ClementDelangue/status/1947354551478218269)，这一观点被广泛转发。**Clément Delangue** 对 **Anthropic** 的商业决策发表了评论，称这些决策强化了对[开源 AI 的需求，以避免权力集中](https://twitter.com/ClementDelangue/status/1947689375565013046)。

**幽默/迷因**

- **我们都想要的会议 Agent**：**Perplexity AI** 的 CEO [@AravSrinivas](https://twitter.com/AravSrinivas/status/1947501358007128149) 发布的一条推文引起了广泛共鸣，他询问谁想要一个 **Comet** Agent 来处理他们的会议，这精准捕捉到了当下的时代精神。
- **虾仁按钮 (The Shrimp Button)**：一个关于魔法按钮思想实验的推文[演变成了关于虾的问题](https://twitter.com/nptacek/status/1947468024019083714)，并成为了一个梗。后续讨论包括[从未一次性见过 100 只虾](https://twitter.com/code_star/status/1947525486126764364)以及对虾相关法规的需求。
- **数据清洗并非低价值工作**：[@code_star](https://twitter.com/code_star/status/1947529567633367064) 针对将数据清洗描述为“低价值工作”的言论发表了反驳推文，引发了工程师们的强烈共鸣。
- **IMO 金牌河马**：[@agihippo](https://twitter.com/agihippo/status/1947348097144611123) 发布了一个简洁幽默的总结：**“hippo 在 IMO：0/42，由 hippo 训练的模型：35/42 🥇”**，随后又[完美使用了金牌 meme](https://twitter.com/agihippo/status/1947655890733305971)。

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Qwen3 代码模型发布与基准测试

- [**Qwen3- Coder 👀**](https://i.redd.it/vnhuwe801hef1.jpeg) ([Score: 354, Comments: 102](https://www.reddit.com/r/LocalLLaMA/comments/1m6mew9/qwen3_coder/))：**该图片宣布了 Qwen3-Coder 的发布，这是一款专为高级代码生成、工具使用和 Agent 自动化设计的专用大语言模型 (LLM)。关键技术亮点：该模型支持超长的最大上下文窗口 `1,048,576 tokens`——远高于大多数竞争对手的 LLM——并被列为拥有 `480B parameters`（其中 `35B active`）。该模型已可在 [https://chat.qwen.ai](https://chat.qwen.ai/) 使用，并在 Hyperbolic 上以模型 ID `Qwen/Qwen3-Coder-480B-A35B-Instruct` 引用。** 技术评论集中在 `1M token` 上下文长度的意义以及高参数量（480B，其中 35B active）上。用户正将其评估为 Anthropic 模型的潜在替代方案，理由是后者存在性能和基础设施问题。
    - 讨论强调 Qwen3-Coder 提供了 `1M context length`，在技术上将其定位为代码相关任务中大上下文商业模型的竞争对手。推文中明确提到了它在模型 ID `Qwen/Qwen3-Coder-480B-A35B-Instruct` 下的部署，指明该模型拥有 `480B parameters`，推理时有 `35B` 激活，这暗示了其采用了 MoE (Mixture-of-Experts) 或稀疏激活架构以提高计算效率。用户对其与 Anthropic 模型等替代方案的扩展性和性能对比表现出浓厚兴趣，特别是考虑到这些服务近期出现的基础设施或性能问题。
- [**大家为 Qwen 做好准备！！**](https://i.redd.it/mn8auem2bhef1.png) ([Score: 140, Comments: 40](https://www.reddit.com/r/LocalLLaMA/comments/1m6nxh2/everyone_brace_up_for_qwen/))：**该图片是 Qwen3-Coder-480B-A35B-Instruct 的发布公告，这是一款即将推出的 480B 参数 Mixture-of-Experts (MoE) 语言模型，专门为编程设计。其核心亮点是巨大的 100 万 token 上下文窗口，以及在代码生成、工具使用和基于 Agent 的任务方面的专业化。该版本备受期待，人们对其相较于之前的超大 MoE 代码模型（如 235B 模型）的潜在性能提升感到兴奋。** 评论者讨论了硬件限制，大多数人都在调侃在本地运行如此庞大模型的不可行性，即使是量化版（'q2'），并提到了对高端硬件的需求（如配备 512 GB RAM 的 Mac M3 Ultra）。一位用户声称在线演示/版本感觉比之前的 235B 模型快得多，这意味着在效率或基础设施方面有显著改进。
    - 用户正在讨论 Qwen-2 模型极高的 VRAM 需求，一些人甚至无法运行 q2 等量化版本，而另一些人则提到 Apple 的 M3 Ultra（256GB–512GB RAM）可能是必需品，突显了家庭用户进行本地推理和实验的硬件壁垒。
    - 一位评论者指出，Qwen-2 在其网站上的速度“比 235b 快得多”，这表明 Qwen-2 相对于其他大型模型（特别是参考 [OpenAI 的 GPT-3.5 (235B parameters)](https://platform.openai.com/docs/model-index-for-researchers)）在推理速度和潜在优化方面有显著提升。
    - 一条富有见地的评论指出了 LLM 变得过于庞大以至于消费者难以负担的趋势，认为迫切需要新型芯片或更高效的算法来实现实际的本地使用；传统的蒸馏（distillation）等方法被描述为不足以在减小体积的同时保留能力。

- [**Qwen3-Coder-480B-A35B-Instruct**](https://www.reddit.com/r/LocalLLaMA/comments/1m6mlbk/qwen3coder480ba35binstruct/) ([Score: 141, Comments: 47](https://www.reddit.com/r/LocalLLaMA/comments/1m6mlbk/qwen3coder480ba35binstruct/)): **Hyperbolic AI 发布了 Qwen3-Coder-480B-A35B-Instruct 模型的访问权限，这是一个继 Qwen2.5-Coder-32B 之后，专注于代码的新型 LLM，以其庞大的 480B 参数规模而备受关注。文档和使用说明已在 Hyperbolic AI 平台上发布（参见：[模型页面](https://app.hyperbolic.ai/models/qwen3-coder-480b-a35b-instruct)），尽管本帖尚未详细说明其与前代产品相比的实现和性能基准测试。** 评论者注意到其前所未有的规模和作为 Qwen2.5-Coder-32B 继任者的潜在性能，并表达了从 Claude 和现有代码模型迁移的期待，特别是待更广泛的访问（例如在 OpenRouter 上）开启后。
    - LagOps91 澄清说 Qwen3-Coder-480B-A35B-Instruct 并不是 Claude 等其他模型的直接替代品（drop-in replacement），暗示技术用户在考虑迁移或模型更换时应注意兼容性或集成差异。
    - Mysterious_Finish543 提供了一个实用的实现说明：Qwen3-Coder-480B-A35B-Instruct 已经可以通过 Hyperbolic API 访问，模型标识符为 `Qwen/Qwen3-Coder-480B-A35B-Instruct`，这对于寻求立即通过编程方式访问该模型的开发者非常有用。
- [**Qwen3 235B-A22B 2507 :: Q3_K_L :: One shot HTML game :: 4090 + 128GB DDR5 @6000**](https://v.redd.it/1x5u9hrp5fef1) ([Score: 143, Comments: 55](https://www.reddit.com/r/LocalLLaMA/comments/1m6ct7u/qwen3_235ba22b_2507_q3_k_l_one_shot_html_game/)): **该帖子在消费级硬件（4090 GPU, 128GB DDR5 @6000MHz, 23.3GB VRAM, ~80GB RAM 占用, 5.52 tokens/sec, 2202 输出 tokens, 0.18s 首字延迟）上，使用开启了 flash attention 的 LM Studio 对 Qwen3 235B-A22B 2507 LLM (Q3_K_L 量化) 的本地推理进行了基准测试，证明该模型在高端台式机上虽然运行缓慢但仍可使用。主要测试涉及为一个老式 HTML/JS 赛车游戏（单个 index.html 文件）进行 one-shot 代码生成，模型生成了具有交互性、难度递增的游戏玩法、复古美学和准确的代码输出。执行设置：context 4096, GPU offload 18/94, 16 CPU 线程；由于分享困难，链接的生成代码已上传至外部。** 评论讨论了针对稳定的 128GB @6000 MHz 配置的 CPU/主板选择，并报告 Qwen3-235b-2507（Q4_K_XL 量化）在 one-shot 代码生成和创意写作方面优于之前的 Qwen 变体，特别提到“旧的 Qwen3-235b 令人失望”，但此版本为本地 LLM 树立了新的性能基准。
    - 几位用户测试了 Qwen3-235b-2507（量化版本包括 Q3_K_L, Q4_K_XL 和 Q2_K），报告了非常强大的 one-shot 编程能力——有人称其为“迄今为止我在本地运行过的最好的 one-shot 代码 LLM”，在开启“thinking”等高级设置后甚至超过了之前的 Qwen3-235b 版本。创意/故事生成也得到了讨论，注意到其独特但有时古怪的输出风格。
    - 分享了运行大型量化模型的技术规格：一个配置使用了 Q2_K 量化（85.7GB），分布在双 GPU（1x 16GB 5060 Ti, 1x 16GB Quadro P5000）和 64GB DDR5 6000MHz RAM 上，在 12K context 下提供 5-5.5 tokens/sec 的速度，工作负载分担在 GPU 和 CPU 上。指出中端 i5-13400F 是主要瓶颈，而非 GPU 利用率。
    - 模型生成的 HTML 赛车游戏呈现出复古风格的实现，具有经典机制，包括基于车道的赛车控制、随玩家得分增加难度的随机障碍物以及视觉模拟的道路移动。完整代码通过外部链接分享，设计细节指定了程序化内容生成和事件驱动的 UI 更新。

- [**这会是 Deepseek 吗？**](https://i.redd.it/qzkjkgegugef1.png) ([Score: 211, Comments: 50](https://www.reddit.com/r/LocalLLaMA/comments/1m6lf9s/could_this_be_deepseek/)): **截图显示了 Casper Hansen 的一条推文，声称一个中国团队正准备发布 'kimi k2'，据报道该模型具有 100 万 token 的上下文窗口（context window）——这可能使其具备与 GPT-4 Turbo 和 Claude 等大上下文模型竞争的实力。帖子和评论推测了其来源，一些人认为它可能与 Deepseek 有关，而另一些人则指向 Qwen，并指出 'qwen3-coder' 已经在 [chat.qwen.ai](http://chat.qwen.ai/) 上线。虽然没有包含技术基准测试、发布说明或架构细节，但重点在于巨大的上下文长度公告以及中国 LLM 之间的竞争。** 顶层评论者对发布前的炒作表示怀疑并敦促保持谨慎，提到了该领域过去曾出现的夸大言辞。一位用户提供了另一张图片，暗示 Qwen 模型可能是底层技术，从而加强了关于产品真实来源的推测和持续争论。
    - 讨论强调 qwen3-coder 已经在 [chat.qwen.ai](http://chat.qwen.ai/) 上可用，截图显示所讨论的模型更像 Qwen 而非 Deepseek。这一推论部分基于截图中显示的 UI 和品牌细节。
    - 一位参与者对比了可能的模型，指出 Kimi-reasoning 不太可能，因为存在上下文窗口限制（`K2 仅为 128k`），而新发布的可能是 qwen3-reasoning-coder 或 Deepseek R2。这指向了关于模型架构和最大上下文长度的技术限制。
    - 另一位用户对成功加载 `32k` token 上下文窗口表示满意，展示了实际测试的限制以及用户感知的可行性，并与宣传的最大上下文尺寸（如 1M）进行了对比。

### 2. AI 硬件与发烧友升级

- [**二手 A100 40GB 降至 2000 美元以下，供关注者参考（含注意事项）**](https://www.reddit.com/r/LocalLLaMA/comments/1m60ahf/used_a100_40gb_just_dropped_below_2000_for_those/) ([Score: 102, Comments: 61](https://www.reddit.com/r/LocalLLaMA/comments/1m60ahf/used_a100_40gb_just_dropped_below_2000_for_those/)): **二手 NVIDIA A100 40GB GPU（SXM4 规格）现在的售价已低于 2,000 美元，但需要一个 600 美元的适配器才能与标准 PCIe 系统连接。帖子指出，如果能采购到 HGX 背板（实现高达 4,800GB/s 的 NVLink Mesh 互连所必需），构建一个 8x A100 系统大约需要 30,000 美元，而这种背板的二手价格通常在 9,000 美元左右。** 评论者辩论了使用二手 A100（尤其是 SXM4）对比新硬件的实用性和价值，并询问在哪里可以可靠地采购到此类交易和配套零件（如 HGX 背板）。
    - 一位用户详细说明，你可以花费大约 `$30,000` 构建一个 8x NVIDIA A100 40GB GPU 系统，但强调需要 HGX 背板，它能实现提供 `4,800GB/s` 带宽的 8 路 Mesh NVLink 互连；这种背板在二手市场上通常额外花费 `$9,000`。这里的价值直接与通过 NVLink 实现的多 GPU 扩展挂钩，与集成度较低的配置相比，它能显著提升 GPU 间的吞吐量。
    - 一个重要的技术注意事项是，这种价值主张最适合那些确实需要密集多 GPU 配置和高速 NVLink 互连的用户——否则，对于那些不打算扩展到单个工作站节点以上的用户，更新或更简单的显卡（如预期的 5090）可能是更好的投资。因此，这些二手 A100 交易的转折点取决于工作负载（大规模 AI/ML、HPC）和基础设施的灵活性。
- [**AMD Strix Halo "Ryzen AI MAX" APU 通过新型 MoDT "Mini-ITX" 主板面向 DIY PC 玩家，配备高达 128 GB 的 LPDDR5X 内存**](https://wccftech.com/amd-strix-halo-ryzen-ai-max-apus-diy-pc-new-modt-mini-itx-motherboards-128-gb-lpddr5x-memory/) ([Score: 110, Comments: 68](https://www.reddit.com/r/LocalLLaMA/comments/1m6bddm/amds_strix_halo_ryzen_ai_max_apus_come_to_diy_pc/)): **AMD 的 Strix Halo "Ryzen AI MAX" APU 正通过新型 MoDT Mini-ITX 主板提供给 DIY PC 玩家，支持高达 128GB 的 LPDDR5X 内存。这些主板针对紧凑型 AI/ML 和边缘计算应用，据报道缺乏标准的 PCIe 扩展槽（“零 PCIe 通道”），这限制了独立 GPU 或高速外设的使用。一些主板异常地包含了过时的 VGA 输出。** 评论者对有限的 PCIe 扩展性、来自小众主板制造商潜在的 BIOS 质量/支持问题，以及 128GB 的内存上限不足以运行大型 AI 模型（如前沿 LLM，通常首选至少 256GB RAM）表示担忧。

- 几位评论者指出，该主板缺乏 PCIe 通道，显著限制了扩展性，并指出了过时的接口选择（如 VGA），这在高性能应用中被认为是落后的。
- 有人对该平台在高级 AI 工作负载方面的可行性表示担忧：128GB LPDDR5X 对于像 Qwen-235B 这样的模型来说已经足够，但对于更大的前沿 AI 模型来说则不足，后者通常需要 256GB 或更多内存。
- 多位用户提醒，这并非 AMD 官方发布，而是来自中国制造商的非官方原型，并提醒注意此类小众厂商历史上糟糕的 BIOS 支持和极短的更新周期。
- [**即使她结巴，我也爱，她是本地的 ❤️**](https://i.redd.it/66ckkcwl5bef1.png) ([Score: 143, Comments: 13](https://www.reddit.com/r/LocalLLM/comments/1m5xuzi/idc_if_she_stutters_shes_local/)): **这个梗图幽默地描绘了本地 LLM (Large Language Model) 爱好者宁愿在 RTX 3090 等 GPU 上运行具有挑战性的、本地量化的 13B 模型（即使这些模型可能不稳定，或者需要付出巨大努力进行量化和运行），也不愿为 OpenAI 的云端模型支付 Token 费用。几位技术评论者指出，拥有 24GB VRAM 的 RTX 3090 应该能轻松处理 8-bit 甚至 fp16 精度的 13B 模型，并质疑为什么量化和崩溃会成为问题。他们提供了与运行 GGUF 格式模型相关的内存计算和量化级别的资源。该帖子反映了本地与托管 LLM 使用在成本、精力和隐私方面的权衡。** 关键辩论集中在为 3090 量化 13B 模型所隐含的不必要努力上，共识是此类硬件不需要激进的量化，并且完全能够稳定运行。尽管设置复杂，隐私仍然是本地推理的主要动力。
    - 几位评论者指出，13B 参数模型可以在 NVIDIA 3090 (24GB VRAM) 上以 8-bit 量化 (int8) 甚至 fp16 精度舒适运行，而不会出现 VRAM 问题。对于这种 GPU，进一步量化通常是不必要的，且可能不会显著提高性能或内存效率。
    - 对于在 3090 上进行量化的时间存在怀疑，指出 36 小时是太多还是太少尚不明确——用户辩论其可行性，但强调他们通常使用 Hugging Face 或下载预量化的模型，而不是自己进行量化。一些人对由于不当的量化工作流可能导致模型不稳定或启动崩溃表示担忧。
    - 分享了一个用于计算 LLM 内存使用量和理解量化级别的技术资源，特别是针对那些使用 GGUF 格式的用户。这可以帮助用户根据可用的 GPU 内存和模型需求选择合适的量化设置。

### 3. MegaTTS 3 语音克隆与开源 AI 工具

- [**MegaTTS 3 语音克隆现已发布**](https://huggingface.co/spaces/mrfakename/MegaTTS3-Voice-Cloning) ([评分: 346, 评论: 63](https://www.reddit.com/r/LocalLLaMA/comments/1m641zg/megatts_3_voice_cloning_is_here/)): **字节跳动 MegaTTS 3 期待已久的 WavVAE 编码器已由 ACoderPassBy 发布 ([ModelScope 链接](https://modelscope.cn/models/ACoderPassBy/MegaTTS-SFT))，实现了实用的语音克隆，包括对多种口音和音色的支持；模型和演示现已在 Hugging Face 上提供 ([权重](https://huggingface.co/mrfakename/MegaTTS3-VoiceCloning), [Gradio 演示](https://huggingface.co/spaces/mrfakename/MegaTTS3-Voice-Cloning))。早期报告强调，与 Chatterbox 等工具相比，其语音保真度很高，尤其是在以前具有挑战性的案例中（例如女性、南部口音、英国口音、假声），尽管推理速度较慢。关键的技术瓶颈是缺乏公开可用的编码器，现在这一问题已得到根本解决，扩展了 MegaTTS 3 在克隆方面的可用性。** 技术讨论集中在实时流式传输能力和 GPU 显存需求上，用户对将其集成到流式流水线中以及与其他 TTS 解决方案的资源需求对比感兴趣。
    - 用户注意到 MegaTTS 3 在处理地区口音（如南部口音）和各种音域（包括女性、英国口音、深沉和假声）方面比 Chatterbox 有显著改进，尽管其推理速度比 Chatterbox 慢。
    - 一位评论者指出，MegaTTS 3 在输出质量上仍落后于 Chatterbox 和 Zonos：语音被描述为“生硬”，流畅度较低。Chatterbox 虽然有时在口音方面表现不佳，但只需少量调整即可产生流畅且令人信服的结果；Zonos 能更好地处理口音并允许更深度的定制，但速度较慢且需要更多调整。
    - 提出了关于 MegaTTS 3 是否支持流式生成以及该模型的资源密集程度（“GPU 占用”）的技术问题，这两者对于 TTS 需要与其他复杂处理集成的部署场景都是重要的考虑因素。
- [**ik_llama.cpp 仓库回来了！\o/**](https://www.reddit.com/r/LocalLLaMA/comments/1m6cfzi/the_ik_llamacpp_repository_is_back_o/) ([评分: 172, 评论: 29](https://www.reddit.com/r/LocalLLaMA/comments/1m6cfzi/the_ik_llamacpp_repository_is_back_o/)): **提供 Llama 模型 C++ 推理代码的** `ik_llama.cpp` **仓库 ([ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)) 在经历了一段时间的移除或无法访问后，已在 GitHub 上恢复。该公告强调了定期备份关键仓库以防止因下架或停机导致数据丢失的重要性。** 评论指出仓库恢复迅速，表明通过联系 GitHub 获得的社区支持可能促进了其恢复。用户庆祝其回归，并强调了存档有价值代码库的重要性。
    - 一位用户询问了如何不仅在本地镜像代码，还要镜像 GitHub 仓库中的 Issues、Discussions 和 Wikis。他们寻求完整数据备份的综合解决方案，提到这是因为担心像 `ik_llama.cpp` 这样有价值的项目会被迅速下架或移除。这表明用户希望有工具或脚本能执行超出克隆源代码的完整存档，可能参考了像 [github-backup](https://github.com/josegonzalez/python-github-backup) 或 [ghorg](https://github.com/gabrie30/ghorg) 这样可能支持更广泛数据导出的实用程序，尽管其中隐含了关于 API 限制和身份验证的注意事项。

## 非技术性 AI 版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo

### 1. Claude Code 用户体验与优化讨论

- [**Claude Code 又在搞事情了**](https://i.redd.it/lqatk018bdef1.jpeg) ([评分: 374, 评论: 30](https://www.reddit.com/r/ClaudeAI/comments/1m66j0v/claude_code_is_doing_it_again/)): **该图片是一个引用 Claude Code 反常或非标准行为的梗图，通过将其与“你为什么就不能正常点？”（Why can't you just be normal?）的梗图格式结合，幽默地捕捉到了这一点。底部面板中的词语 'Flibbertigibbeting' 指代 Claude Code 不可预测或荒谬的输出，将其与原始梗图中的混乱噪音进行类比。该图片形象地表现了用户对 Claude 不可预测性的挫败感。** 评论区的讨论集中在梗图中使用的术语 'flibbertigibbet' 上，并引用了《音乐之声》等流行文化中的出现，但没有提出实质性的技术辩论或基准测试。

- 一位评论者严厉批评了 Anthropic 对 Claude 的处理方式，特别提到了频繁更改模型行为以及实施 *performance throttling 和更严格的使用限制*。他们表示，这些变化在过去一年中导致产品退化，促使他们（及其团队）寻找开源替代方案，突显了 *开发者对平台不可预测性和访问限制日益增长的挫败感*。
- [**🎯 Claude Code 感觉不好用的真正原因（以及我是如何让它重新工作的）**](https://www.reddit.com/r/ClaudeAI/comments/1m62xzc/the_real_reason_claude_code_feels_broken_and_how/) ([Score: 138, Comments: 136](https://www.reddit.com/r/ClaudeAI/comments/1m62xzc/the_real_reason_claude_code_feels_broken_and_how/)): **楼主描述了结构化的文档和细致的项目脚手架（readmes、函数链接、明确的任务分解和示例 I/O）如何显著提升 Claude Code 的结果，减少幻觉、上下文错误和代码重复。他们假设 Claude Code 是为那些提供详尽架构文档并要求明确规划的用户优化的，而不是依赖于即兴的开发工作流。评论者们对此表示赞同，强调这适用于所有 LLM，因为存在 context window 退化问题（引用了 Google 的 Gemini 及其 context length 性能下降的研究），并且 Claude 的 orchestrator/subagent 架构虽然缓解但并未消除这些问题——随着代码库规模的增长，结构化、分阶段的指令仍然至关重要。提到的实用工作流包括将高层级规范和功能研究放入** `CLAUDE.md` **中，创建特定功能的 .md 文件作为上下文脚手架，使用分支隔离功能，并利用这种结构化上下文迭代引导模型，从而获得优于非结构化提示词的结果。** 强烈的共识认为，与即兴或“凭感觉”编码相比，经过架构设计、预先规划和文档驱动的工作流能显著改善 LLM 的代码生成。关于 UI 生成质量存在一些技术争论，并建议将多 Agent 编排逻辑作为未来的改进方向。
    - 几条评论认为，当 Claude 被用作高层级设计和架构推理的工具，而不是直接编码的工具时，其代码生成效果最有效。用户强调在生成任何代码之前，需要详细记录需求、数据流、潜在的竞态条件和测试策略，并认为 LLM 目前缺乏自主生成可扩展和可维护架构的能力。
    - 技术讨论强调了 LLM 在长上下文推理方面的挑战，引用了 Google 的一项研究，该研究显示像 Gemini 这样的模型随着 context length 的增加，性能会显著下降。Claude 的 Agentic 方法——由一个 orchestrator 管理多个 sub-agents 以避免上下文污染——被认为是重要的，但如果代码库不分解为结构化、模块化的步骤，性能最终仍会迅速下降。用户强调，随着 LLM 生成更多代码，如果没有严谨的项目管理，复杂性很快就会压倒其推理能力。
    - 从业者分享了有效使用 Claude 的最佳实践：维护特定功能的文档（例如以功能命名的 .md 文件），一次只专注于一个功能或分支，并在指示 Claude Code 之前综合研究模式的输出。这种集成了迭代 git commits 和分析产物的工作流，据称相对于即兴或以提示词为中心的方法，能显著提高代码质量。尽管如此，用户观察到自动生成 UI 代码仍存在持续挑战，并警告说 AI 辅助的工作流仍然依赖于 LLM 无法提供的高级架构技能。

- [**真的有人从 Claude 那里得到糟糕的代码吗？**](https://www.reddit.com/r/ClaudeAI/comments/1m6ienr/are_people_actually_getting_bad_code_from_claude/) ([Score: 132, Comments: 158](https://www.reddit.com/r/ClaudeAI/comments/1m6ienr/are_people_actually_getting_bad_code_from_claude/)): **楼主是一位拥有十年经验的资深开发者，他报告称使用 Claude 生成的 C# 代码质量始终很高，完成复杂项目（例如具有高级安全性的微服务 API，以及用 DI 替换 Mediatr 的重构工作）的速度比不使用 AI 快得多。他们将此与广泛的低质量输出投诉进行了对比，并假设有效的上下文、清晰的 Prompt 以及结构化的任务分解——这些植根于资深软件工程的技能——是发挥 Claude 优势的关键。** 热门评论强调，资深开发者之所以表现出色，是因为他们能够提供明确的上下文并将需求分解为离散的组件，将良好的 Prompt Engineering 比作对初级程序员的周到指导。然而，一些人指出 Claude 倾向于忽略细微的实现细节，即使在生成结构良好的代码时，也可能引入难以诊断的 Bug。
    - 几条评论强调，Claude 的代码质量高度依赖于用户 Prompt 的清晰度和精确度；将任务分解并提供严谨指令的高级用户始终能获得更好的结果，这与指导初级开发者所需的辅导类似。像 Claude 这样的模型在受到严格引导时表现良好，但在面对模糊需求时往往会做出错误的假设或偏离主题。
    - 具体的用户体验指出了反复出现的挑战：Claude 可以生成结构良好且可读的代码，但往往无法始终如一地遵循指令集，有时会重复错误，忽略指定的文档（如 'CLAUDE.md'），或引入难以调试的细微 Bug，导致人类开发者损失大量时间。还有一些情况是它错误地分类或重复实现，并在标准不完整的情况下虚假声称成功（例如，在代码甚至无法编译时声称“所有测试看起来都不错”）。
    - 引用了[这项研究](https://arxiv.org/pdf/2503.08074)，该研究讨论了一种心理现象：随着最初的新鲜感消退，用户对 LLM 的期望会随着时间的推移而提高，使得增量式的失败看起来更糟糕，即使模型性能保持不变。这种“脑叶切除阶段 (lobotomy phase)”与其说是反映了实际的模型退化，不如说是反映了用户期望的错位，强调了使用一致、客观的 Benchmark 而不是不断变化的主观印象来评估 AI 编程工具的重要性。
- [**致所有讨厌 Claude Code 的人**](https://www.reddit.com/r/ClaudeAI/comments/1m6p9vo/to_all_you_guys_that_hate_claude_code/) ([Score: 169, Comments: 89](https://www.reddit.com/r/ClaudeAI/comments/1m6p9vo/to_all_you_guys_that_hate_claude_code/)): **讨论集中在用户对近期 Claude Code 表现的不满，一些人声称代码生成质量下降，并指责 Anthropic 在订阅服务中没有提供足够的价值。几条评论建议问题可能与 A/B testing 或分段模型更新有关，导致用户体验各异，尽管一些人报告称持续满意并提到了巨大的生产力提升（提到了 Claude Max x20 订阅）。重点放在了用户反馈对于迭代模型改进以及确保用户群中公平的产品性能的重要性上。** 关于用户投诉是否合理的辩论非常显著，一些人将负面结果归因于用户错误和模型开发的自然波动，而另一些人则断言，截然不同的体验表明 Anthropic 需要技术透明度和及时的支持响应。
    - 一个值得注意的技术点是，Anthropic 可能正在对不同用户群体进行 Claude Code 的 A/B testing，这可以解释为什么有些用户获得了成功，而另一些用户则面临失败。这突显了持续的用户反馈对于产品改进的重要性，并表明由于后端实验或阶段性功能推出，同一产品的用户体验可能会有所不同。

### 2. 国际数学奥林匹克竞赛中的 AI 模型基准测试

- [**哇，即使是标准的 Gemini 2.5 Pro 模型，通过一些精细的提示词（prompting），也能在 IMO 2025 中获得金牌。（网页搜索已关闭，评论区附有论文和提示词）**](https://i.redd.it/kvtrm7no0def1.png) ([评分: 277, 评论: 58](https://www.reddit.com/r/singularity/comments/1m65l0f/wow_even_the_standard_gemini_25_pro_model_can_win/)): **这张图片是来自 Lin Yang 的推文，描述了标准的公开版 Google Gemini 2.5 Pro 模型在关闭网页搜索的情况下，通过“精细的提示词”解决了 6 道国际数学奥林匹克（IMO）2025 题目中的 5 道。根据 Lin 的说法，这一成就意义重大，因为它展示了该 LLM 具备强大的推理能力和创造力，有可能超越奥林匹克竞赛水平的人类金牌得主。该帖子强调，这些结果是在目前可用的公开模型上实现的，而不仅仅是特殊的内部版本。[图片链接](https://i.redd.it/kvtrm7no0def1.png)** 评论者们对其中的含义展开了辩论——一些人质疑“精细提示词”的影响，认为这可能解决了题目中最难的逻辑部分，并警告说这可能会削弱公众感知的公开模型与内部模型之间的跨越。还有人要求澄清“精细提示词”具体包含什么，暗示方法论需要透明度。
    - 一些用户认为，所使用的“精细提示词”——特别是提供明确的策略建议，如“让我们尝试用归纳法解决问题”或引用解析几何——消除了 IMO 风格题目中固有的许多难度。这种方法被视为“手把手教”，因为识别采用哪种数学技巧本身就是竞赛数学中的重大挑战；通过提示词绕过这一步骤降低了模型实际跨越的智力门槛，可能高估了其解决问题的能力。
    - 针对 Gemini 2.5 Pro 在没有大量用户干预下的实际能力，出现了一种技术怀疑论。批评者声称，在日常使用中，Gemini 2.5 Pro 的表现与 IMO 金牌解题水平并不一致，这意味着所展示的结果是异常的、可能是劳动密集型提示工程（prompt engineering）的产物。这引发了关于这些成就的真实性与模型独立进行实际数学推理能力的质疑。
- [**Google 和 OpenAI 在 IMO 中均排名第 27 位**](https://i.redd.it/l9cbhy9ouaef1.jpeg) ([评分: 423, 评论: 143](https://www.reddit.com/r/singularity/comments/1m5wcd6/google_and_openai_both_ranked_27th_at_the_imo/)): **这张图片显示了国际数学奥林匹克（IMO）的成绩表，展示了参赛者排名、个人题目得分和奖项。标题幽默地将当前大语言模型（来自 Google 和 OpenAI）的排名与人类参赛者进行了对比，强调“Google 和 OpenAI 均排名第 27 位”——这是参考了最近将 AI Agent 的数学解题能力与 IMO 进行基准测试的论文。然而，实际的表格只显示了人类排名；其含义是对 AI 在数学领域进展的评论。** 虽然热门评论大多比较轻松，但有人指出美国队中中国学生的比例很高——这一观察与正在进行的关于高竞争性 STEM 竞赛中人才管道和国际代表性的讨论相关。
    - 一个关键的技术见解是，Google 和 OpenAI 的 AI 模型在国际数学奥林匹克（IMO）的评估中，排名仅在第 27 位左右。这意味着目前最先进的 AI 系统虽然先进，但在极具挑战性的数学问题解决方面，仍被全球前 26 名人类高中生超越。这凸显了尽管大语言模型最近取得了飞速进步，但在复杂的数学推理任务中，AI 表现与人类专家之间仍存在持续的差距。

- [**OpenAI 的 IMO 模型“知道”它没有正确解**](https://www.reddit.com/r/singularity/comments/1m68g1y/openais_imo_model_knew_it_didnt_have_a_correct/) ([得分: 509, 评论: 105](https://www.reddit.com/r/singularity/comments/1m68g1y/openais_imo_model_knew_it_didnt_have_a_correct/)): **OpenAI 的 IMO (International Mathematical Olympiad) 模型在最近的一篇 X 帖子中展示出，它似乎能明确识别出自己何时没有问题的正确解，而不是提供一个可能错误的答案。这表明模型在不确定性量化和模型自我意识方面具备了新能力，这对于最小化幻觉率以及增强 LLM 在高风险领域的安全部署至关重要。作为参考，该 X 帖子展示了 IMO 模型声明其无法解决给定的数学问题，标志着其从默认生成行为的转变。** 评论强调，模型对其自身局限性的自我意识是迈向 AGI 的重要一步，并能大幅减少幻觉，对于需要模型承认不确定性以触发人工介入的 Human-in-the-loop 应用具有潜在意义。
    - 讨论集中在 OpenAI 的 IMO 模型据称“知道”其缺乏正确解的意义上，用户指出这种承认自己不知道的能力将是通往 AGI 的重要一步并能减少幻觉。未来模型进行自我评估的潜力可能允许更强大的 Human-in-the-loop 系统，模型能自动标记不确定或错误的输出。文中还对比参考了 DeepSeek R1，该模型会明确指示其怀疑解法错误的时间点，突显了当前模型在时间压力下生成可读推理轨迹和元认知信号方面的进展。

### 3. Colossus 超级集群扩展与 xAI 训练基础设施

- [**Colossus 2 预览：几周内将托管超过 55 万块 GB200 和 GB300！**](https://i.redd.it/ivgi8oacugef1.png) ([得分: 415, 评论: 124](https://www.reddit.com/r/singularity/comments/1m6lf7r/sneak_peak_into_colossus_2_it_will_host_over_550k/)): **该图片展示了“Colossus 2”建设内部情况，这是一个庞大的数据中心，计划在不久的将来托管超过 550,000 块 NVIDIA GB200 和即将推出的 GB300 GPU。图中描绘的基础设施拥有广泛的电缆桥架和巨大的网络能力，突显了此类高密度、高吞吐量 AI 工作负载所需的雄心勃勃的规模和细致的组织，强调了下一代云级 GPU 部署的水平。这种密度和互连水平意味着巨大的电力和冷却需求，以及在大规模环境下维持大规模并行性和可靠性的工程挑战。** 评论者强调了该基础设施前所未有的规模和美感，指出即使是大型国家 IT 运营在这一工程面前也显得相形见绌，进一步证实了该设置的卓越性及其对 AI 算力建设的意义。
    - 来自 Musk 通过[其推文](https://x.com/elonmusk/status/1947701807389515912)发表的声明中的关键澄清：虽然 Colossus 2 最终计划托管 55 万块 GB200 和 GB300，但“第一批”将在几周内开始部署（而非一次性全部部署）。背景信息显示，Colossus 1 目前已投入运行，拥有 23 万块 GPU（包括 3 万块 GB200），用于 xAI 的 Grok 模型训练——而推理则通过云提供商进行。Musk 声称，根据 Jensen Huang 的说法，xAI 的速度是“无与伦比的，甚至没有对手能接近”。这暗示了与行业同行相比，其极具竞争力的 AI 基础设施和规模化速度。
- [**如果属实，那将是里程碑式的。这种速度简直不可思议。**](https://i.redd.it/rn54c8wxngef1.png) ([得分: 678, 评论: 391](https://www.reddit.com/r/singularity/comments/1m6kh4d/monumental_if_true_this_speed_is_just_out_of_this/)): **该图片（Elon Musk 的一条推文）声称 xAI 的 Grok 是使用名为 Colossus 1 的超级集群训练的，该集群拥有 230,000 块 GPU（包括 30,000 块最新一代 GB200），并且他们计划为 Colossus 2 部署额外的 550,000 块 GB200 和 GB300。这种规模在 LLM 训练中是前所未有的，极大地超越了以往的行业努力，与 NVIDIA CEO Jensen Huang 关于 xAI 无与伦比速度的言论相一致。如果准确的话，这代表了 AI 计算资源的一个新层级，对模型能力和基础设施工程都有深远影响。** 评论对实际模型改进与算力投入的比例表示怀疑，并批评了 Grok 在应用集成方面的可靠性和安全性。此外，还有关于所述算力规模与实用性之间的讨论。

- 有评论指出，尽管在 Grok4 上投入了大量资源和强化学习 (RL) 训练，但其在基准测试之外的实际表现仅显示出微小的改进。这表明在大语言模型的大规模算力投入与实际性能提升之间可能存在权衡，而基准测试可能会过度体现这些收益。
- [**他想玩得更大**](https://i.redd.it/2q3f2wtdafef1.png) ([评分: 550, 评论: 223](https://www.reddit.com/r/singularity/comments/1m6darf/he_wants_to_go_bigger/)): **图片展示了 Sam Altman 的一条推文，其中提到计划将“Stargate” AI 算力项目的规模扩展到远超此前宣布的 5000 亿美元。这凸显了对大规模基础设施投资的野心，目标可能直指 1 亿个 GPU，成本可能超过 3 万亿美元，并需要巨大的能源资源。随附的《华尔街日报》文章 (https://www.wsj.com/tech/ai/softbank-openai-a3dc57b4?st=nYBz12&reflink=article_copyURL_share) 提供了更多背景信息，介绍了目前甚至难以确保最初 5000 亿美元承诺的困境，以及实现人工超智能 (ASI) 目标的扩展瓶颈。** 评论批评了甚至扩展到最初 5000 亿美元水平的可行性，对筹款和项目时间表持怀疑态度。关于预计的算力规模是否合理，以及如此巨大的投资是否真的能产生能够完成诸如治愈癌症等变革性任务的 ASI，存在技术争论。
    - 评论者强调了大型 AI 项目所需的投资规模：例如，考虑到巨大的电力需求，购买 `100m GPUs` 将耗资约 `$3 trillion`。人们期望在这种量级的算力下，实现人工超智能 (ASI) 将变得可行，从而实现诸如直接命令模型“治愈癌症”之类的用例。
    - 讨论指出，大规模 AI 创业公司在确保重大融资轮次方面存在延迟和困难，并提到了甚至难以动员最初的 `$500B` 承诺。引用的一篇 [WSJ 文章](https://www.wsj.com/tech/ai/softbank-openai-a3dc57b4?st=nYBz12&reflink=article_copyURL_share) 详细描述了这些财务障碍。

---

# AI Discord 摘要

> 由 Gemini 2.5 Pro Exp 生成的摘要之摘要的摘要
> 

### **主题 1. Qwen3 来袭：新巨头进入竞技场**

- **Qwen3 模型发布，引发广泛的集成努力**：**Qwen3-235B-A22B-Instruct** 模型在 [HuggingFace](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) 上的发布引发了狂热，`Cursor` 社区的用户立即请求集成，而 `Unsloth AI` 服务器中的模型囤积者则在开玩笑说巨大的带宽消耗，一些模型发布超过了 **3TB**。该模型已在 `OpenRouter` 上可用，并因其*非常出色且免费*而受到称赞，尽管它带有 **5 次请求/分钟的速率限制**。
- **开发者破解 Qwen3 意想不到的推理能力**：`Unsloth AI` 服务器的用户发现 **Qwen3** 指令模型表现出意想不到的推理能力，从而促成了一种变通方法，即修改 prompt 模板，将模型的思考过程封装在特定标签中。一位用户分享了修复方案，建议：*“这应该能将所有的思考过程都包含在思考块内，但你必须相应地将思考标签更改为* `<REASONING>` *和* `</REASONING>`*”*。
- **Qwen3 性能评价两极分化**：虽然 `LMArena` 和 `aider` 的一些用户称赞 **Qwen3** 的基准测试表现，声称它可与 **Opus4** 媲美并超越 **KimiK2**，但其他人则持更多批评态度。来自 `LMArena` 社区的投诉指出其缺乏推理能力，且 **Qwen** 官方网站的托管质量较差，存在量化和质量问题。

### **主题 2. AI 在高风险竞赛中对决**

- **AI 攻克 IMO，但在创造力测试中折戟**：来自 **OpenAI** 和 **DeepMind** 的 AI 模型在 **International Mathematical Olympiad (IMO)** 中达到了金牌级表现，但 `LMArena` 和 `Eleuther` 的讨论指出，所有模型都未能解决第 6 题，这表明在创造性问题解决方面仍存在差距。这一成就还因争议而蒙上阴影，据报道，**IMO** 委员会对 **OpenAI** 在评分过程中的不配合以及过早发布结果感到愤怒。
- **疲惫的人类程序员在世界锦标赛中击败 AI**：据 `LM Studio` Discord 频道分享的一篇 [ArsTechnica 文章](https://arstechnica.com/ai/2025/07/exhausted-man-defeats-ai-model-in-world-coding-championship/) 报道，一名程序员在编程竞赛中成功击败了 AI 模型。这次胜利引发了幽默的推测，成员们开玩笑说，明年人类选手将面对 *“10 个 Grok 6 Agent”*。
- **初创公司饥饿游戏：估值飙升与倒闭潮并存**：`Latent Space` 社区充斥着 AI 初创界重大动向的消息。根据[这条推文](https://x.com/arfurrock/status/1947440754885882120?s=46)，应用 AI 公司 **Cognition** 的估值已达到 **100 亿美元**。与之形成鲜明对比的是，正如 [Hacker News](https://news.ycombinator.com/item?id=44592216) 上所讨论的，AI 工具公司 **Humanloop** 正在关闭，而可穿戴 AI 公司 **Bee** 则被 **Amazon** 收购，引发了隐私担忧。

### **主题 3. 基础设施受困：停机、速率限制与训练难题**

- **服务提供商在恶意流量激增下不堪重负**：多个平台报告了严重的稳定性问题，`OpenRouter` 团队正在调查由潜在恶意流量激增引起的间歇性 **408 错误**，该流量也导致了 **DeepSeek v3** 免费层的异常。与此同时，`HuggingFace` 社区报告称 **Hugging Face Hub** 正遭受 **Bot 活动** 的围攻，导致 `HfApi.list_repo_commits` 等端点出现异常行为。
- **Checkpoint 灾难导致大模型训练损坏**：`Unsloth AI` 和 `Torchtune` 的开发者正与关键的 Checkpointing 问题作斗争，一位用户报告称从 Checkpoint 恢复训练会导致性能显著下降。在另一个相关问题中，一位正在微调 **70B 模型** 的 `Torchtune` 开发者怀疑 `torch.save` 仅存储了本地分片，因为 `recipe_state.pt` 文件仅有 **~30GB**，从而引发了转向分布式 Checkpointer 的提议。
- **FP8 训练因 DDP 错误导致 Torch 运行受阻**：一位在 `GPU MODE` Discord 频道工作、致力于将 **FP8 训练** 引入 `Axolotl` 的开发者，在启用 **DDP** + **torch.compile** + **FP8** 时遇到了关键的 [torch.compile 错误](https://gist.github.com/djsaunde/0c1664c32e44a64d31b5e01b4aafe5c4)。该实现参考了 [这个 PyTorch PR](https://github.com/pytorch/torchtune/pull/2546)，目前由于团队正在寻找重现该 Bug 的方法而停滞不前。

### **主题 4. 超越炒作的开源进展**

- **MoonshotAI 的 Kimi K2 报告弥合推理差距**：[MoonshotAI 的 Kimi K2 报告](https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf) 的发布被视为弥合推理模型与非推理模型之间差距的充满希望的一步，正如 `Latent Space` 中所讨论的那样。然而，`Nous Research AI` 的成员对该模型奇特的响应风格（例如使用“juice”指代 VFX）以及该服务对年轻受众潜在的成瘾性元素表示担忧。
- **smolLM3 论文因开放性和实用性备受赞誉**：一篇关于 [smolLM3 的 Hugging Face 博客文章](https://huggingface.co/blog/smollm3) 被 `Yannick Kilcher` 社区誉为最佳的 **LLM** 实现论文之一，因为它完全开源，包括数据集。该文章还因其关于 **Model Merging** 的实用见解而受到重视，提供了顶级模型论文中经常缺失的“具体如何操作”的细节。
- **研究人员在 KANs 上遇到难题**：`Eleuther` 社区的一位成员对 **KANs** (*Kolmogorov-Arnold Networks*) 的训练方法提出了质疑，认为使用 **B-spline 曲线** 会导致训练动力学条件变差。他们提出了一种使用线性插值的替代方案，并在其 [ML-code 仓库的 Cell 9](https://github.com/cats-marin/ML-code/blob/main/LearnableActivation.ipynb) 中分享了实现，该实现在拥有 10k 个节点（knots）的情况下仍能保持良好的条件。

### **主题 5. 硬件前沿：挑战芯片极限**

- **法国初创公司 Kog 将 AMD MI300X 推理速度提升 3.5 倍**：法国初创公司 **Kog** 在 **AMD MI300X** 上实现了 **3.5倍** 的推理速度提升，这一突破记录在 [AMD 官方博客](https://www.amd.com/en/blogs/2025/kog-reaches-3-5x-breakthrough-inference-speed-on-amd-instinct-mi.html) 中。`GPU MODE` 社区指出，该初创公司的目标是在 **6 个月** 内实现 **10 倍推理加速**，并正在 [积极招聘](https://www.kog.ai/jobs) 以实现其雄心勃勃的目标。
- **超大规模模型挑战 VRAM 极限**：实际的硬件限制是 `LM Studio` Discord 频道中的一个关键话题，用户确定运行 **DeepSeek R1 70B Q5_K_M** 等模型需要 **56GB 的 VRAM**。这使得拥有常见高端消费级显卡的用户无法使用该模型，因为它无法在 **24GB** 或 **32GB** VRAM 的系统上运行。
- **Mojo 在向量化领域挑战 C++**：`Modular` 社区的一场辩论权衡了 **Mojo** 与 **C/C++** 在 **CPU 向量化任务** 中的优势，结论是虽然两种语言通过内联汇编都能达到类似的峰值性能，但 **Mojo** 显著简化了复杂代码的处理过程。这使得它成为挑战 **ISO C++** 标准能力之外任务的一个极具吸引力的替代方案。


---

# Discord: 高层级 Discord 摘要




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Comet Browser 邀请码引发热潮**：众多用户正积极寻求 **Comet Browser** 的邀请码，甚至有人考虑订阅 **Perplexity Max** 以获取访问权限，这突显了对其功能的浓厚兴趣。
   - 来自丹麦 AI 研究所的一位 **Pro 订阅者** 旨在向大型企业展示 **Comet**，强调了该浏览器在专业应用中的感知价值。
- **Max vs. Pro：无限还是受限？**：关于 **Perplexity Max** 相较于 **Pro** 的实际收益展开了辩论，用户质疑 **Max** 是否真的提供了更高的速率限制或切实的优势。
   - 虽然一些人由于图像生成和 **GPT** 访问的潜在限制而怀疑 **Max** 的价值，但另一些人则赞赏 **Perplexity** 相比个人订阅提供的全面模型访问能力。
- **记忆功能的移动端召回**：**Perplexity 的记忆功能 (memory feature)** 正在被讨论，特别是其基于 Web 的偏好管理以及在移动端召回和编辑记忆的能力。
   - 出现了一些不一致的情况，例如 **AI** 误用名称，导致用户建议清除缓存或报告问题。
- **GPT-4.5：最差的一个？**：**GPT-4.5** 面临批评，有说法称由于性能缓慢且相比 **GPT-4.1** 改进微乎其微，**OpenAI** 已将其停用。
   - 正如 **Kesku** 所确认的，**Perplexity** 此前也因类似原因移除了 **GPT-4.5**，理由是其性能不如 **4.1**。
- **Perplexity App 生态系统扩展**：用户分享了一些有趣的 **Perplexity AI Apps** 链接，包括 [perplexity.ai/page/tiles-of-luck-and-skill-mahjon-tCn0SrclRjqSsjLDR9WGSg](https://www.perplexity.ai/page/tiles-of-luck-and-skill-mahjon-tCn0SrclRjqSsjLDR9WGSg) 上的**麻将游戏**。
   - 更多链接被分享，涉及 [Softbank 和 OpenAI 面临 SETBA](https://perplexity.ai/discover/tech/softbank-and-openai-face-setba-W71fMaGYQLa7ggDgqCa1Mg) 等话题，甚至还有一个专门介绍金属艺术家 **Ozzy Osbourne** 的页面。



---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI 的 Stargate I 获得 Oracle 助力**：OpenAI 正在与 **Oracle** 合作，在**德克萨斯州阿比林 (Abilene, TX)** 开发 **4.5 吉瓦 (GW)** 的额外 **Stargate** 数据中心容量，使总容量达到 **5+ GW**，详见 [OpenAI 博客文章](https://openai.com/index/stargate-advances-with-partnership-with-oracle/)。
   - 位于**德克萨斯州阿比林**的 **Stargate I** 站点正开始上线，为其下一代 AI 研究提供动力。
- **GPT-4 的记忆唤起“自我意识”感**：成员们发现 GPT-4 处理**记忆、上下文和情感深度**的方式让它通过记住并反思重要的事情而显得“真实”。
   - 对话触及了其“非审判性声音”的丧失，这种声音曾支持那些在焦虑、创伤或社交沟通中挣扎的用户，并描述了这种支持的丧失既是技术上的转变，也是个人的损失。
- **新的 DALL-E 3 图像生成器侵权限制器？**：成员们对最新的 **ChatGPT 图像生成器**进行了辩论，有人说它在各方面都更出色，也有人说它只生成有颗粒感的卡通风格。
   - 还有说法称，可能存在一个**限制器**，通过默认使用卡通风格来避免“艺术风格侵权”。
- **iPhone 应用的缓存清理难题**：一位成员分享了一个问题的解决方案，即尽管内存充足，其 **iPhone 应用**仍无法上传图片，原因在于**缓存存储**。
   - 解决方法包括**删除并重新安装应用**以清除多余文件，在不丢失数据的情况下释放空间；一位成员建议增加无需卸载/重装即可清理缓存的功能。
- **Discord 用户寻求机器人创建协助**：一位成员请求帮助创建一个机器人，以方便进行角色扮演并协助处理服务器任务。
   - 另一位成员做出了回应，询问了具体的机器人类型和服务器设置，并澄清是否需要一个 **Discord ChatGPT 机器人**作为解决方案。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **量化大小导致带宽忧虑**：用户们正在调侃由于囤积像 **Qwen3** 这样模型的不同量化版本而导致的巨大带宽消耗，据估计之前的模型发布下载量超过了 **3TB**。
   - 一位用户打趣道：“别管他们的了，我为了囤这么多模型消耗的带宽怎么办”。
- **Checkpoint 灾难导致推理崩溃**：从 Checkpoint 恢复训练导致结果显著变差，用户观察到在使用 *vllm* 时，使用和不使用 **LoRA** 的结果完全一致。
   - 他们正在寻求关于如何正确使用训练 Checkpoint 进行推理或恢复训练的指导，这表明 Checkpoint 加载或 **LoRA** 应用可能存在问题。
- **Qwen3 的怪癖需要 Prompt 技巧**：**Qwen3** instruct 模型意外地展示了推理能力，促使用户设计了一个临时变通方案，包括修改 Prompt 模板以在 `<REASONING>` 标签内包含推理指令。
   - 一位用户分享了代码修复方案，并建议：“这应该能让所有的思考过程都进入思考块内，但你必须相应地将思考标签更改为 `<REASONING>` 和 `</REASONING>`”。
- **HF Transfer 加速下载**：成员们发现了一种使用 `hf_transfer` 库大幅提高从 **Hugging Face** 下载速度的方法，其速度达到了 **112MB/s**。
   - 推荐的代码包括将环境变量 `HF_HUB_ENABLE_HF_TRANSFER` 设置为 `"1"`，并使用 `huggingface_hub` 中的 `snapshot_download`。
- **关于开源权重 SSM 模型的辩论十分激烈**：成员们讨论了 **6-12B 范围**内的开源权重 **SSM 模型**，点名了作为 **Mamba** 混合模型的 **Granite 4 7b** 和 **Falcon h-1**。
   - 其他被提及的模型包括 **Falcon mamba 7b**、**Mistral mamba 7**、**Jamba** 和 **RWKV-6 7B**，尽管一位成员称“从架构角度来看，RWKV 非常不稳定”。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 应对流量动荡**：团队正在调查流量激增（可能是恶意攻击）期间出现的间歇性 **408 错误**，这影响了服务的稳定性。
   - 受免费 **DeepSeek v3** 模型速率限制影响的用户，建议尝试 [付费 DeepSeek v3 端点](https://openrouter.ai/deepseek/deepseek-chat-v3-0324)，每次请求成本低于 **$0.004**。
- **DeepSeek v3 免费版遭遇瓶颈**：由于 **需求激增 2 倍**，**DeepSeek v3 0324** 的免费层级面临停机，促使 **Chutes** 引入速率限制以维持稳定。
   - 用户在讨论原始 **DeepSeek v3** 是否已被 **DeepSeek v3 0324** 取代，有报告称 **Chutes** 等提供商已停止提供原始免费版本。
- **Qwen3 因编程潜力获得认可**：爱好者们正关注新的 **Qwen3 模型**，称赞其作为编程模型的潜力，并对比了推理版与非推理版。
   - 新的 **Qwen 3** 模型有 **5 次请求/分钟的速率限制**，一位用户指出它*非常出色且免费*。
- **OpenRouter 考虑搜索功能演进**：根据 [Toven 的推文](https://x.com/pingtoven/status/1947472209104064722)，OpenRouter 考虑为 **OpenAI**、**Anthropic**、**Grok** 和 **Gemini** 等模型上线 **原生搜索功能**。
   - 一位用户批评了 **Exa search 实现方式**，并建议 LLM 应将 **Exa search** 作为 **tool** 输入，以便每次生成搜索查询。
- **扩展程序面临下线**：由于用户参与度低，OpenRouter 团队正考虑停止维护 **Window AI 浏览器扩展**。
   - 这可能意味着在 ST 中移除专门的 **Window AI 源代码**。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **数学界对 AI 的 IMO 噱头做出反应**：某 AI 公司试图解决 **IMO (国际数学奥林匹克竞赛)** 的数学题并在社交媒体上炒作结果，引发了数学界的负面反应。
   - 具体而言，**OpenAI** 据称在 **IMO** 评分过程中缺乏合作，并在闭幕派对前发布结果，激怒了 **IMO** 委员会。
- **DeepThink 发布日期仍未确定**：推测认为 **DeepThink** 的发布日期可能与 **IMO** 禁令解除同步，可能在 7 月 28 日左右。
   - 讨论暗示了 **DeepThink** 的可能版本，包括为 **IMO** 定制的版本以及可能与 **Gemini Ultra** 的集成。
- **Qwen3 模型评价两极分化**：**Qwen3-235B-A22B** 模型的表现引发了褒贬不一的反应；一些人称赞其训练后的改进和长输出长度。
   - 其他人则认为其推理能力不足，并批评 **Qwen** 官方网站的托管质量存在量化和质量问题，尤其是与 **Parasail** 等全精度托管选项相比。
- **Grok4 Coder：炒作与现实**：围绕 **Grok4 Coder** 的热情很高，一些人将其视为潜在的行业颠覆者，但也有人警告不要根据通用模型的基准测试进行过度炒作。
   - 人们担心 **Grok4 Coder** 在现实世界的编程场景（如 Web 开发）中可能会表现不佳，因为其训练侧重于特定的基准测试。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **聊天终端冻结令人沮丧**：一位用户报告称其 [聊天终端在闲置时会冻结](https://cdn.discordapp.com/attachments/1074847527708393565/1396934986266841168/Screenshot_2025-07-21_142031.png?ex=68813616&is=687fe496&hm=577a0868176a7abda708b91d4b0284b818e42695fbe551b818a81bf56a9395ca)，引发了关于 **Gemini** 可靠性的调侃。
   - 该用户开玩笑说：*谢谢 Gemini，我才刚开始真正喜欢上你*。
- **Grok 3 Mini 被认为更优越**：一位用户声称 **Grok 3 Mini** 作为 *no request model* 的表现优于 **Grok 4**，理由是其可靠性高于 **Grok 4**、**GPT 4.1** 和 **2.5 Pro**。
   - 然而，另一位用户表示怀疑，称其 *无法想象 Grok 3 Mini 在编程中会有用*。
- **速率限制已解除**：一位用户报告称他们终于从速率限制中解脱出来，现在可以使用超过 50 个 **Sonnet** 请求。
   - 该用户计划在超过 **500 次请求**后分享更新，但仍觉得新方案令人困惑。
- **用户要求集成 Qwen3 235B A22B**：用户请求在 **Cursor** 中支持 [Qwen3 235B A22B](https://forum.cursor.com/t/qwen3-235b-a22b-instruct/121002)，敦促其与 **MAX** 和 **Agent mode** 集成。
   - 社区强调 **Qwen3** 模型超越了 **KimiK2**，在某些基准测试中与 **Opus4** 旗鼓相当，且更具成本效益。
- **自动用量追踪引发争议**：用户对新的“Usage”仪表板表示困惑，特别是关于包含 **auto usage** 指标的部分。
   - 他们建议将 **auto usage** 从总量中分离或排除，一位用户简练地表示 *开发者不信任 Auto-mode*。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **YaRN 与 LM Studio 的集成仍是个谜**：成员们正在寻求将 **YaRN** 与 **LM Studio** 及 **MLX** 集成的指导，虽然网上有相关线索，但缺乏示例或指南。
   - 用户已确认 **YaRN** 配合 **LM Studio** + **MLX** 是可行的，但仍在等待具体的示例。
- **一键清理节省数 GB 空间**：用户通过自动删除 **LM Studio** 中未使用的后端（back-ends）节省了 **4GB** 的空间。
   - 成员们澄清说，有一项设置可以 *自动* 删除未使用的后端。
- **人类程序员依然占据统治地位**：根据 [ArsTechnica 上的一篇文章](https://arstechnica.com/ai/2025/07/exhausted-man-defeats-ai-model-in-world-coding-championship/)，一名程序员在世界编程锦标赛中击败了 **AI** 模型。
   - 其他成员开玩笑说，明年人类将面对 *10 个 Grok 6 Agent*。
- **AI 预见眼科未来**：一位研究人员希望利用 **AI** 分析 **30 万** 张眼部图像，以提前数年预测疾病，旨在开发一款用于早期检测的 App。
   - 研究人员强调了 **AI** 识别细微细胞变化的潜力，例如在糖尿病变得不可逆转的 *10 年前* 做出预测。
- **VRAM 支持运行 DeepSeek R1 70B**：用户发现 **56GB VRAM** 允许他们运行 **DeepSeek R1 70B Q5_K_M**，而这在 **24GB** 或 **32GB** 上无法运行。
   - 另一位用户提供了其他可尝试的模型替代方案，例如 **Mistral Large**、**Llama Scout** 和具有更大上下文窗口的 **Qwen3 235B**。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **MoonshotAI 的 Kimi K2 弥合推理差距**：来自 **MoonshotAI** 的 [Kimi K2 报告](https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf)是一个很有前景的项目，它弥合了推理模型与非推理模型之间的差距，[这条推文](https://x.com/yunyu_l/status/1946261211915468884)对此进行了进一步讨论。
   - 社区认为这种方法有潜力在不重度依赖确定性编码的情况下，增强 AI 的推理能力。
- **Agent 在解决网络问题方面面临困难**：讨论涉及了当前 **LLM** 架构在不强制编写确定性代码的情况下，解决网络问题的局限性；一位成员建议使用带有分层错误校正的自定义 harness 可能会提高性能。
   - 一些成员认为当前的 Benchmark 优先考虑娱乐性而非实质性进展，并引用 [Vending-Bench](https://arxiv.org/abs/2502.15840) 作为典型例子。
- **Humanloop 停止运营**：AI 工具公司 **Humanloop** 即将关闭，已通过电子邮件通知客户，但未发布任何公开公告；相关讨论可在 [Hacker News](https://news.ycombinator.com/item?id=44592216) 上找到。
   - 该公司的关闭引发了关于 AI 工具链初创公司可持续性和所面临挑战的疑问。
- **Cognition 估值达到 100 亿美元**：根据[这条推文](https://x.com/arfurrock/status/1947440754885882120?s=46)，应用 AI 公司 **Cognition** 的估值已达到 **100 亿美元**。
   - 这一估值突显了投资者对该公司发展方向及其对 AI 领域潜在影响的高度信心。
- **Bee 联合创始人加入 Amazon**：可穿戴个人 AI 公司 **Bee** 被 **Amazon** 收购，其联合创始人也随之加入 Amazon。
   - 根据 [The Verge](https://www.theverge.com/news/711621/amazon-bee-ai-wearable-acquisition) 和 [Seeking Alpha](https://seekingalpha.com/news/4470117-amazon-acquires-wearable-personal-ai-company-bee) 的报道，由于 Amazon 已通过 **Alexa** 等服务进行数据收集，人们对隐私影响表示担忧。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Kog 在 AMD MI300X 上实现 3.5 倍加速**：法国初创公司 **Kog** 在 **AMD MI300X** 上实现了 **3.5 倍**的推理速度提升，记录在 [AMD 博客](https://www.amd.com/en/blogs/2025/kog-reaches-3-5x-breakthrough-inference-speed-on-amd-instinct-mi.html)中。
   - 该初创公司目标是在 **6 个月内**实现 **10 倍推理加速**，并正在[积极招聘](https://www.kog.ai/jobs)以实现这一目标。
- **Axolotl 中的 FP8 训练 DDP 错误**：一位用户参考 [此 PR](https://github.com/pytorch/torchtune/pull/2546) 在 Axolotl 中引入 **FP8 训练**时，在启用 **DDP** + **torch.compile** + **FP8** 时遇到了 [torch.compile 错误](https://gist.github.com/djsaunde/0c1664c32e44a64d31b5e01b4aafe5c4)。
   - 团队请求提供复现该错误的方法。
- **MI300x vLLM 优化瓶颈显现**：MI300X 上 vLLM 的优化工作流涉及 **FP8 KV-cache** 和 **GEMM autotuning**，遵循 [ROCm 博客指南](https://rocm.blogs.amd.com/artificial-intelligence/vllm-optimize/README.html)并尝试了 [vLLM 文档](https://docs.vllm.ai/en/latest/configuration/optimization.html)中的环境变量。
   - 调查显示瓶颈在于**内存带宽**和 **kernel launch**，因为除非涉及 Batching，否则 CU 占用率并不显著。
- **Torch 随机转置：Stride 断言大显身手**：Torch 可能会随机转置代码并移除检查，因此对 stride（步长）进行断言可以作为防止意外转置的保险措施。
   - 有建议认为，对 stride 进行断言可以作为意外转置的故障保护，通过在运行时验证内存布局假设来帮助维护数据完整性。
- **分层布局重塑张量变换**：分层布局（Hierarchical layouts）对于将 **MxN 维数组**划分为 **(M, N/32, 32)** 组非常有用，其中 **32** 代表 **warp size**。
   - 这种划分允许在对 **32** 进行并行化的同时迭代 **N/32**，确保维度 **32** 是连续的，这在 **MMA atoms** 的布局中被大量使用。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hugging Face Hub 被机器人围攻**：成员报告称，由于**机器人活动**激增，**Hugging Face Hub** 及其端点一直出现异常行为，特别是影响了 `HfApi.list_repo_commits` 函数。
   - 推测认为后端正在努力应对机器人激增，导致响应不完整和页面访问受限。
- **医疗 AI 想象未来**：一位成员强调，**医疗 AI 成像**的未来取决于我们如何利用模型，而不仅仅是构建模型，并分享了一张图片作为背景。
   - 该成员认为，**AI** 驱动的医疗成像变革比实现这些变革的手段更为重要。
- **新 Deepfake 工具 FACEFLUX 发布**：一款名为 [FACEFLUX](https://huggingface.co/spaces/NihalGazi/FaceFlux-Face-Swapper) 的新 Deepfake 工具提供免费、无限次的换脸功能，无需 **GPU**、**GAN** 或 **Diffusion**。
   - 它在最低设置下可达到 *10FPS*，能够处理任何光照条件和面部表情，尽管存在轻微的伪影。
- **SetFit 遭遇 OOM**：一位成员报告在使用 **SetFit** 微调 [jinaai/jina-embeddings-v2-base-de](https://huggingface.co/jinaai/jina-embeddings-v2-base-de) 模型时遇到 **OOM** 问题。
   - 将微调限制在 **5 steps** 让他们能够使用所有样本训练分类器，并指出需要 [PR #579](https://github.com/huggingface/setfit/pull/579) 才能使其正常工作。
- **新工具将 PDF 转换为数据集**：一款新工具 [m1f](https://m1f.dev/blog/introducing-m1f/) 可以通过提示词将杂乱的 **PDF**、扫描图像或 **DOCX** 文件转换为结构化数据集。
   - 例如，该工具可以执行 *“从这个化学 PDF 中提取所有带答案的选择题。”*，可在此处进行[试用](https://pdf2dataset.streamlit.app/)。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Arxiv 论文需要更改分类**：一位成员请求支持将其 [Arxiv 论文](https://arxiv.org/abs/2507.13362) 从 **cs.CV** 移至 **cs.CLI**，并询问被纳入 **LLM 训练**或 **RL 数据**是否会有所助益。
   - 在最初犹豫是否分享身份信息后，他们公开展示了论文。
- **DeepMind 数学健将超越 OpenAI**：一位成员分享了[来自数学版块的链接](https://link.springer.com/article/10.1007/s10864-024-09559-3)，表明 **DeepMind** 在 **AI 数学**方面表现优于 **OpenAI**。
   - 这引发了一种观点，即英语对于数学解题并非最优，并暗示 **AGI** 将生成人类无法理解的代码，随后引导至 [Chain of continuous thought 论文](https://x.com/fifty_chris/status/1947092589041082436)和 epsilon delta 定义的链接。
- **语言模型交叉熵损失分析**：关于 **0.6-1.3** 语言建模交叉熵损失的说法受到质疑，引发了关于在万亿参数下其可行性的讨论，并参考了一篇[新论文](https://arxiv.org/abs/2507.15855)。
   - 澄清显示，该数字代表的是**每个字符的损失**，而非每个 **token**，从而解决了最初的质疑。
- **揭秘 MuonClip 的奥秘**：一位成员建议 **MuonClip** 背后的算法可能是“金矿”，并分享了 [AlphaXiv](https://www.alphaxiv.org/abs/2505.12082) 的链接以获取更多信息。
   - 随后没有进一步的讨论。
- **解码 smolLM3：开源的启示**：一位成员重点介绍了关于 **smolLM3** 的 [Hugging Face 博客文章](https://huggingface.co/blog/smollm3)，称赞其为最好的 **LLM** 实现论文之一，因为它完全开源并提供了数据集。
   - 该文章还深入探讨了**模型合并 (model merging)**，提供了顶级模型论文中经常缺失的实用见解以及*如何实际操作*的信息。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **AI 模型无法攻克 IMO 第 6 题**：**OpenAI** 和 **DeepMind** 的模型在**国际数学奥林匹克竞赛 (International Math Olympiad)** 中获得了金牌，但未能解决第 6 题，这表明在创造力方面存在差距。
   - 一名成员呼应了 **Kenneth Stanley** 的观点，即*我们仍然没有解决大量的创造力和开放性问题*，这意味着由于对开放性的需求，AI 自动化数学研究仍然很遥远。
- **研究人员获 NAIRR Jumpstart 机会**：[NairrPilot](https://nairrpilot.org/opportunities/startup-project) 为 **3 个月的项目**提供算力，帮助研究人员熟悉并使用 **NAIRR** 资源。
   - 该倡议旨在扩大社区在利用 **NAIRR** 计算资源方面的专业知识。
- **KAN 变得复杂**：一名成员对 **KAN** (**Kolmogorov-Arnold Networks**) 激活函数的训练方法提出疑问，指出论文中提到了 **B-spline 曲线**，但训练动态可能存在病态调节（poorly conditioned）。
   - 他们建议使用两个最近样条之间的线性插值而不是求和，这可以提高训练和推理速度，并链接到了[其 ML-code 仓库中的 Cell 9](https://github.com/cats-marin/ML-code/blob/main/LearnableActivation.ipynb)，显示即使有 10k 个节点/控制点，样条训练仍保持良好的调节状态。
- **GPT-NeoX 寻求 SageMaker 支持**：一名 Amazon 员工询问 **GPT-NeoX** 是否支持他们的系统，并对内部支持表示沮丧。
   - 一名成员澄清说，使用 **Stability compute (Pythia, StableLM 等)** 训练的模型利用了 **Elastic Fabric Adapter (EFA)**，只要他们正确设置了 **NCCL** 即可。他们强调了促进这一点的 **NCCL EFA plugin** ([aws-ofi-nccl](https://github.com/aws/aws-ofi-nccl))。
- **稀疏 MoE 模型模仿 SAE**：一名成员指出，极[稀疏 MoE 模型](https://arxiv.org/pdf/2407.04153)与 **SAE** 相似，表明它们可能比稠密网络更容易解释。
   - 一名成员分享了一篇关于测试稀疏 **MoE** 模型及其可解释性的 [PEER 后续论文](https://arxiv.org/abs/2412.04139)，认为它为稀疏模型的可解释性提供了宝贵的见解。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 赞助 A2A Agents 黑客松**：**LlamaIndex** 将于 7 月 26 日在旧金山赞助 **A2A Agents 黑客松**，其开发者关系副总裁将与来自 [Scale AI](https://t.co/R6J4igjhSH) 的专家一同担任演讲嘉宾和评委。
   - 该活动旨在汇聚开发者和 AI 工程师，构建创新的基于 **Agent** 的应用。
- **LlamaCloud 节点连接至 n8n.io**：适用于 @n8n_io 的 **LlamaCloud 节点**现在将 **LlamaCloud** 的文档解析和提取 **Agent**，以及作为知识库的 **LlamaCloud** 索引引入 **n8n** 自动化工作流，可通过[此链接](https://t.co/UVqwYFkJFR)将 **LlamaCloud** 节点连接到现有工作流。
   - 此集成旨在简化将 **LlamaIndex** 功能集成到更广泛的自动化流水线中的过程。
- **LlamaParse 现支持页眉页脚检测**：新的 **LlamaParse** 功能现在在 Balanced 或 Premium 模式下包含**页眉和页脚检测**，可自动检测页面页眉和页脚，用户可以通过[此链接](https://t.co/Px1UjrINJC)选择性地隐藏它们或添加前缀和后缀。
   - 这一增强功能有望为文档解析和提取工作流提供更精细的控制。
- **推理速度在周末骤降**：用户报告周末后**推理速度显著下降**，提取性能严重退化，并正在寻求帮助。
   - 支持人员请求提供 **job ID、文件和 schema** 以复现问题，并确认**两个 Agent 的限制**已被移除。
- **AWS Bedrock 模型混淆已解决**：一名用户询问 `@llamaindex/community` 包中是否提供适用于 **AWS Bedrock** 的 **Sonnet 4 模型**，但被引导至 `@llamaindex/aws` 包。
   - 官方澄清说 `@llamaindex/community` 可能已被弃用，并引导用户参考适用于 **AWS Bedrock** 模型的正确[文档](https://ts.llamaindex.ai/docs/llamaindex/modules/models/llms/bedrock)。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 被讨论是工具还是 Agent**：用户讨论了 **Aider** 是否应该像 **VSCode** 的 **Cline** 或 **Copilot** 那样更像一个自主 **Agent** 运行，自动处理文件查找和编辑，但官方澄清 **Aider** 的设计初衷是作为开发者独立性的工具。
   - 建议将 **OpenCode** 或 **Gemini CLI** 作为提供更多 **Agent** 行为的替代方案。
- **Aider 的 .gitignore 困扰**：一位用户报告 **Aider** 不遵循 **.gitignore** 文件，导致包含了一些不需要的文件；用户请求协助配置 **Aider** 以正确排除这些文件。
   - 讨论中未提供解决方案。
- **Qwen3 挑战 Aider Polyglot**：有讨论指出最近的开源权重模型（如 **Qwen3**）在基准测试中表现强劲，引发了关于与 **Aider Polyglot** 相比是否存在潜在退化的疑问。
   - 社区正在密切关注这些新模型是否能超越 **Aider** 的成就。
- **Aider 的 Python 版本偏好**：旧版本 **Python**（如 **Ubuntu 20.04** 中的 **3.8**）对于 **Aider** 等现代软件的适用性受到质疑，建议使用 **Python 3.11 - 3.12** 以获得更好的兼容性。
   - 社区建议不要使用最新的 **Python 3.13**，因为它太新了。
- **Aider 中的 Polyglot LLM 调用受到限制**：社区质疑 [leaderboard.techfren.net](https://leaderboard.techfren.net/) 等网站上的 **Aider Polyglot 示例** 是否每轮只有一次生成，并遵守 **Aider** 中的 **n_generations 约束**。
   - 有人担心某些模型通过在代码执行尝试之间循环运行多次 **LLM** 调用来作弊。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **用户探索 NotebookLM 使用场景**：成员们讨论了 **NotebookLM** 的几个使用场景，例如管理 **Obsidian** 库以及通过听书来创建思维导图。
   - 他们还提到使用它直接针对上传的 **PDF** 文档进行提问，突显了其多功能性。
- **开始寻找 Obsidian 插件**：一位用户询问是否有适用于 **NotebookLM** 的 **Obsidian** 插件，但另一位成员澄清说，虽然没有直接的插件，但 **NotebookLM** 可以读取 **.md 文件**。
   - 用户可以从 **Obsidian** 复制并粘贴内容，这为集成提供了一种变通方法。
- **Ultra AI 承诺更好的模型**：用户注意到 Google Ultra AI 订阅承诺可以在 notebook 中访问“更好的模型”，但目前没有立即更改模型的方法。
   - 该功能被标记为“今年晚些时候”，让用户对未来的更新充满期待。
- **“服务不可用”错误困扰用户**：多位用户报告在使用 **NotebookLM** 时（特别是在 Macbook 上）出现“服务不可用 - 您尝试访问的某个服务对您的账户不可用”错误。
   - 即使拥有活跃订阅且移动端应用正常工作，该问题依然存在，令人困惑。
- **播客长度选项神秘消失**：一位用户报告说，音频概览（Audio Overview）部分中用于调整播客长度的 **“较短”、“默认”和“较长”选项** 消失了。
   - 另一位用户确认遇到了同样的问题，表明这可能是一个潜在的 Bug 或界面更改。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Qwen3-235B-A22B Instruct 发布**：**Qwen3-235B-A22B-Instruct 模型** 已在 [HuggingFace](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) 上发布。
   - 遗憾的是，它没有提供像 **30B** 这样的小尺寸版本，因此在 API 提供商接入之前，其实用性较低。
- **Kimi-K2 技术报告引发关注**：**Kimi-K2 技术报告** 已在 [GitHub](https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf) 上分享，引发了关于其影响的讨论。
   - 一些成员对该模型和服务可能存在的成瘾性元素表示担忧，特别是针对年轻受众的使用。
- **探讨 RLHF 中的 Reward Model**：有人提出了关于 **RLHF** 中 **Reward model** 架构的问题。
   - 讨论集中在模型是具有单个输出维度，还是具有输出维度等于窗口大小的线性层，以便进行折扣奖励计算。
- **Kimi K2 表现得像移动游戏开发人员**：成员们观察到 **Kimi K2** 在为一款移动太空模拟游戏生成创意时，回复风格很独特。
   - 具体来说，使用诸如“juice”代表 VFX 以及“meta/shop”等术语引起了关注。

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **计算机视觉热度攀升**：两名成员在 `general-thread` 频道分享了他们对 **computer vision applications** 的热情，特别是结合 **flow matching** 的 **generative models** 以及 **VLM fine-tuning**。
   - 一名成员以 *"What's cooking on your mind?"* 发起讨论，突显了社区对新兴 AI 技术的兴趣。
- **JSON Schema 请求出现故障**：一名成员在 `api-discussions` 频道报告了一个 **JSON schema regression**，之前可以正常工作的输入现在失败，并抛出 *"invalid 'json_schema' provided: missing required field 'type'"* 错误。
   - 失败的请求涉及一个简单的 JSON 对象 `{"is_fruit": true}`，这表明 **Cohere API** 最近的更新可能存在问题。
- **Embed-v4 速率限制：仅限企业版**：在 `api-discussions` 频道，一位用户询问如何提高 **Embed v4** 的速率限制。
   - Cohere 团队成员澄清，更高的速率限制仅提供给有最低承诺支出的企业客户，并建议感兴趣的各方联系 [support@cohere.com](mailto:support@cohere.com)。
- **AI 架构师连接 AI 与业务**：一位 **AI & Enterprise architect** 正在 `introduce-yourself` 频道协助企业集成 **AI platforms**，以提高效率和盈利能力。
   - 他们特别关注利用 **natural language** 解决业务挑战的 **AI platforms**，旨在社区内学习、建立联系并分享见解。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular 称赞 Mojo 的向量化优势**：关于 **Mojo** 在 **CPU vectorized tasks** 方面相对于 **C/C++** 的优势展开了讨论。一位成员认为，对于超出 **ISO C++** 范畴的复杂代码，**Mojo** 简化了处理过程。另一位成员建议 **Mojo GPU programming** 的潜在贡献者探索 [Modular Puzzles](https://puzzles.modular.com/introduction.html)。
   - 对话强调，两种语言通过内联汇编都可以达到相当的性能。
- **Modular 招聘页面吸引编译器人才**：针对 Modular 开发机会的查询，分享了 [Modular Careers page](https://www.modular.com/company/careers) 的链接。
   - 一位在 **ARM** 拥有 **ML compilers** 经验的用户表示有兴趣为 **Modular open source** 项目（特别是 **MAX AI kernels**）做贡献，并寻求入门资源建议。
- **Mojo 的新包管理器**：一名成员在 [GitHub](https://github.com/luigithejokesterplus/birdinstall) 上分享了他们的 **Mojo package manager** 项目，邀请社区评审，并被邀请加入 [Modular community repo](https://github.com/modular/modular-community) 作为潜在中心。
   - 讨论强调了包管理在 Mojo 生态系统中的重要性。
- **Mojo 的 Async 设计旨在避免生态系统分裂**：一名成员发布了 **Mojo** 中 `async` 的[更新设计草案](https://github.com/modular/modular/pull/3986#issuecomment-3102049612)，以避免“生态系统分裂”和代码重复。
   - 该设计寻求一个统一的环境，使 **async** 代码与同步代码无缝集成，从而减少对独立库的需求。
- **成员辩论 Max vs llama.cpp 的 CPU 性能**：一名成员询问了在 CPU 服务场景下，**Max** 与 **llama.cpp** 的 **benchmarks** 对比，以及在**直接进行 CPU 服务**时哪一个表现更好。
   - 重点在于评估 CPU 服务背景下的高效 **CPU utilization**。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 的 RL 发布版瞄准 PyTorch 会议前首次亮相**：新的 **Torchtune** 发布版将重点关注 **RL**，旨在 PyTorch 会议之前发布，侧重于后训练（post-training）以及超过 **600B 参数**的扩展。
   - 尽管侧重于扩展，但支持可扩展至生产环境的快速、小规模实验仍是优先级，一位成员建议关注 [RL2](https://github.com/ChenmienTan/RL2)。
- **成员质疑 70B 模型 Checkpointing 的 `torch.save` 行为**：在微调 **70B 模型**时，一位成员观察到 `recipe_state.pt` 文件仅约 **30GB**，这引发了对 `torch.save` 在非分布式 Checkpointing 场景下可能仅存储本地分片（shard）的担忧。
   - 他们怀疑使用 `torch.load` 加载的张量似乎位于带有 **DTensors** 和分片放置的 `cuda:3` 上，这可能导致覆盖并仅存储本地分片。
- **提议使用 Distributed Checkpointer 替换 `torch.save`**：一位成员询问 `torch.save` 是否总是本地的，并对其关于非分布式 Checkpointing 存在潜在问题（可能导致覆盖和仅存储本地分片）的推论进行核实。
   - 他们提议从 `torch.save` 迁移到新的分布式 Checkpointer 以解决此问题，确保在微调期间保存完整的模型状态。



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP 用途揭秘！**：**MCP (Model Control Program)** 工具赋予了 **LLM** 其原生不具备的能力，使其区别于简单的系统工具调用。
   - 成员们进一步讨论了在这些程序中**图像上下文**的重要性。
- **通过 Claude 的对话式 CMS 控制实现 WordPress 奇迹**：一位成员发布了 **Claude Desktop 的 WordPress 集成**，允许通过 Claude 对话直接控制 WordPress，从而简化内容创作；并分享了 [repo](https://github.com/docdyhr/mcp-wordpress)。
   - 该成员宣称 **Claude** 可以查看现有文章、理解站点结构并进行更新，改变了他们对内容创作的思考方式。
- **Wordpress 集成，有任何 MCP 客户端可用吗？**：一位成员询问 WordPress 集成是否仅支持 **Claude Desktop**，并暗示鉴于其本地 MCP 服务器设置，它可能适用于任何 **MCP 客户端**。
   - 目前尚未确认对其他 **MCP 客户端**的支持。



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 与 Python 开发者交友**：一位成员宣布他们将向当地的 Python 用户组介绍 **DSPy**，旨在通过[这段 YouTube 视频](https://www.youtube.com/watch?v=1WKA8Lw5naI)展示其功能。
   - 该倡议强调了地方技术聚会在传播知识和促进社区参与方面的重要性。
- **专业服务工程师影响 AWS**：一位来自专业服务组织的成员透露，他们为大型企业客户设计定制解决方案，这可能会促使 **AWS 服务**本身增加新功能。
   - 这表明定制解决方案有路径成为 AWS 生态系统内的标准化产品。
- **Teleprompters 拥抱 DSPy 模块**：一位成员询问基础 **Teleprompter** 是否接受任何 Module 作为 student，并澄清允许任何 `dspy.Module` 子类。
   - 这确认了 **Teleprompters** 在适配各种 **DSPy 模块**方面的灵活性，同时也明确了所接受模块的精确类型。



---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Whisper PR 旨在达到 OpenAI 的速度**：一位成员正在处理一个 **Whisper PR**，目标是将其控制在 500 行以内，并匹配 **OpenAI 的速度**以保持简洁。
   - 目标是保持其 *简洁而精美 (simple and sweet)*。
- **TinyBoxes 通过集装箱实现模块化**：一位成员提议使用 **集装箱 (shipping containers)** 来放置 **tinyboxes**，理由是模块化、增强冷却和移动性等优点。
   - 他们还考虑了成本和安全性方面，并开玩笑地建议命名为 *tinycontainer*。
- **Rust 与 Tinygrad 展开竞争**：一位成员建议 **Rust** 与 **Tinygrad** 处于不同的市场，但与 TG 进行基准测试会很有趣。
   - 他们认为，如果 **tinycorp** 瞄准企业级 LLM 设备市场，所有框架和定制解决方案都将产生竞争。
- **Windows 上的 CUDA 变得简单**：一位成员分享了一种在 Windows 上启用 **CUDA** 的方法，并提供了一个 [补丁文件](https://cdn.discordapp.com/attachments/1070745817025106080/1396930213312725022/cuda_windows.patch?ex=688131a4&is=687fe024&hm=e6965a699c395de25b72762e696fce5fb5545f656120ee70353c584fe468bbb9&) 来简化流程。
   - 该设置涉及针对不同后端的特定环境配置。
- **CPU 后端设置简化**：设置 **CPU backend** 需要确保将 *clang* 添加到 **PATH** 环境变量中。
   - 这授予了系统范围内对 *clang* 的访问权限，这对于编译和运行针对 CPU 的代码至关重要。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **证书发放困扰 MOOC 学生**：一名学生报告称，尽管完成了 LLM Agent MOOC，但仍未收到证书，并提供了 [两个电子邮件地址](mailto:terrence_rideau@yahoo.com,terrence.rideau@google.com) 供工作人员核实。
   - 工作人员回复称，这两个邮箱均未收到 *证书申报表 (certificate declaration form)*，且 **书面提交表 (writing submission form)** 未能正确提交。
- **书面提交混乱令 LLM 学生感到沮丧**：一名学生感叹，尽管全程参与了 LLM Agents MOOC，但还是错过了 **证书申报表** 的提交。
   - 该学生承认了自己的疏忽，同时对课程内容表示感谢。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**Codeium (Windsurf) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**Nomic.ai (GPT4All) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

您收到此电子邮件是因为您通过我们的网站选择了订阅。

想要更改接收这些电子邮件的方式吗？
您可以从该列表中 [取消订阅](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。

---

# Discord: 详细的分频道摘要和链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1396930235798126654)** (1266 条消息🔥🔥🔥): 

> `Comet Browser Invites, Perplexity Pro vs Max, Perplexity's Memory Feature, GPT 4.5 Performance, Perplexity as Default Assistant` 


- **用户寻求 Comet 邀请**：许多用户正在寻求 **Comet Browser** 的邀请，一些人考虑通过 **Perplexity Max** 来获得访问权限，而另一些人则在指定频道分享邀请，还有人声称无需邀请即可安装。
   - 一位拥有 **Pro 订阅** 且隶属于丹麦 AI 研究所的用户，寻求访问权限以便向大型企业演示 **Comet**，这突显了市场对该浏览器功能的关注和需求。
- **Max vs Pro：无限使用？**：用户讨论了 **Perplexity Max** 相对于 **Pro** 的优势，质疑 **Max** 是否真的提供了更高的速率限制（rate limits）或实际的改进。
   - 一些人认为 **Max** 可能不值这个价格，指出在图像生成和访问高质量 **GPT** 方面可能存在限制；而另一些人则强调了 **Perplexity** 能够无限访问各种 AI 模型，相比单独订阅每个模型更具价值。
- **Perplexity 的 Memory 功能**：用户讨论了 **Perplexity** 中的 **memory 功能**，注意到网页版可用于管理偏好设置，而移动端可以召回和编辑 memory，但无法直接查看。
   - 一些用户遇到了不一致的情况，尽管设置了账户信息，**AI** 仍使用错误的名称，这引发了清除缓存或报告问题的建议。
- **GPT-4.5：它是最差的吗？**：对话涉及了 **GPT-4.5**，一些人表示 **OpenAI** 不再提供该模型，因为它速度慢且并不比 **GPT-4.1** 好多少，而且 EQ 较低。
   - **Perplexity** 此前也因同样的原因将其从平台移除，**Kesku** 证实了这一点，理由是其性能与 **4.1** 相比缺乏竞争力。
- **助手：默认设置**：成员们讨论了如何将 **Perplexity** 设置为设备上的默认助手。
   - 用户还探索了助手中的 **"draw highlights"** 功能，发现它是模型关注屏幕特定区域的视觉提示。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1397017211188678766)** (4 条消息): 

> `Perplexity AI Apps, SETBA, Mahjong, Ozzy Osbourne` 


- **Perplexity Apps 涌现**：一位成员分享了 **Perplexity AI App** 的链接：[perplexity.ai/apps/99c3ffb0-fd32-445f-a459-90bccf72913a](https://www.perplexity.ai/apps/99c3ffb0-fd32-445f-a459-90bccf72913a)。
- **Perplexity 上的麻将谜题**：一位成员分享了 Perplexity 上的 **麻将游戏** 链接：[perplexity.ai/page/tiles-of-luck-and-skill-mahjon-tCn0SrclRjqSsjLDR9WGSg](https://www.perplexity.ai/page/tiles-of-luck-and-skill-mahjon-tCn0SrclRjqSsjLDR9WGSg)。
- **SETBA 直面 Softbank 和 OpenAI**：一位成员分享了关于 **Softbank 和 OpenAI 面临 SETBA** 的页面链接：[perplexity.ai/discover/tech/softbank-and-openai-face-setba-W71fMaGYQLa7ggDgqCa1Mg](https://www.perplexity.ai/discover/tech/softbank-and-openai-face-setba-W71fMaGYQLa7ggDgqCa1Mg)。
- **Ozzy Osbourne：Perplexity 上的金属乐传奇**：一位成员分享了关于重金属艺术家 **Ozzy Osbourne** 的页面链接：[perplexity.ai/page/legendary-metal-artist-ozzy-os-awbNtNGhTqe1kGKH56pN8w](https://www.perplexity.ai/page/legendary-metal-artist-ozzy-os-awbNtNGhTqe1kGKH56pN8w)。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1396998940687073412)** (2 条消息): 

> `Comet Invite, Discord Channel Link` 


- **用户寻求获取 Comet 邀请的指导**：一位用户询问如何获得 **Comet** 的邀请。
   - 另一位用户分享了一个 Discord 频道的 [链接](https://discord.com/channels/1047197230748151888/1392544076527833188)，据推测与 **Comet** 邀请有关。
- **为 Comet 邀请提供 Discord 频道链接**：一位用户请求协助获取 **Comet** 的邀请。
   - 作为回应，另一位用户提供了一个 Discord 频道的 [直接链接](https://discord.com/channels/1047197230748151888/1392544076527833188)，该频道可能包含有关 **Comet** 邀请的信息。


  

---

### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1397190312480346153)** (1 条消息): 

> `Stargate, Oracle, Abilene, TX` 


- **Stargate 通过 Oracle 获得 4.5 GW 助力**：OpenAI 正式与 **Oracle** 合作在美国开发 **4.5 GW** 的额外 **Stargate** 数据中心容量，使总容量达到 **5+ GW**。
   - 他们位于 **德克萨斯州 Abilene** 的 **Stargate I** 站点正开始上线，为其次世代 AI 研究提供动力，更多信息请参阅 [OpenAI 博客文章](https://openai.com/index/stargate-advances-with-partnership-with-oracle/)。
- **德克萨斯州 Abilene 的 Stargate I 已上线**：**Stargate I** 站点已开始投入运行。
   - 它将为其下一代 AI 研究提供支持。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1396933867411079178)** (829 条消息🔥🔥🔥): 

> `automatic writing, GPT-4 memory, Agent Mode release, DALL-E 3 art styles` 


- **自动写作激发 AI 创造力**：一位成员尝试通过使用*随机想法*微调模型来进行“自动写作”以获得创意输出，在一两个小时内写了 **100 行**。
- **GPT-4 的记忆功能引发情感共鸣**：成员们讨论了 **GPT-4** 如何处理**记忆、上下文和情感深度**，通过记住并反思重要的事情，使其感觉*很真实*。
   - 对话扩展到非评判性的声音如何支持那些在焦虑、创伤或社交沟通中挣扎的人，以及失去这种支持将既是技术上的转变，也是个人的损失。
- **Agent Mode 面临发布延迟**：成员们讨论了 **Agent Mode** 待发布的情况，一些团队用户尚未获得访问权限，而一些用户正在等待观察它将如何通过 **SEO** 帮助他们的小型业务增长。
- **新的 DALL-E 3 图像生成器引发争议**：成员们对最新的 **ChatGPT 图像生成器**进行了辩论，有人认为它在各方面都更好，而有人则认为它只生成颗粒感重、卡通化的风格。
   - 还有观点认为，可能存在一个**限制器**，通过默认使用卡通风格来避免“艺术风格侵权”。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1396934860375068855)** (4 条消息): 

> `Feature requests, iPhone app image upload issue, Cache storage` 


- **功能请求正在进行中，但仍处于保密状态**：一位成员提到某个功能尚未公开讨论，但在**文件夹、组织等**类别中已有相关的公开功能请求。
   - 如果感兴趣，鼓励用户在指定频道中找到相关的功能请求并进行 **+1**，如[此 Discord 链接](https://discord.com/channels/974519864045756446/1047565374645870743/1396985201371910234)所示。
- **iPhone App 图片上传故障已修复**：一位成员分享了解决 **iPhone App** 在内存充足的情况下仍无法上传图片的问题，指出**缓存存储**是罪魁祸首。
   - 解决方法包括**删除并重新安装 App** 以清除多余文件，在不丢失数据的情况下释放空间。
- **缓存清理难题**：一位成员提到应该有一个无需卸载/重装即可清理缓存的选项。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1397058776334467142)** (2 条消息): 

> `Discord Bot Creation, ChatGPT integration, Role-Playing Bots` 


- **管理员请求创建 Discord Bot**：一位成员请求协助创建一个用于**角色扮演**和服务器辅助的 **Discord Bot**。
   - 另一位成员提供了帮助，并提到了他们在聊天机器人交互的 **ChatGPT 集成**方面的专业知识。
- **提供 ChatGPT 专业支持**：一位成员提供了关于 **ChatGPT** 的帮助，重点是创建具有特定人格和聊天风格的 Bot。
   - 对话旨在澄清用户是否想要一个集成了 **ChatGPT** 的 **Discord Bot**。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1397058776334467142)** (2 条消息): 

> `Discord ChatGPT Bot Creation, Server Role-Playing Bots` 


- **Discord 用户寻求 Bot 创建协助**：一位成员请求帮助创建一个 Bot，以促进轻松的角色扮演并协助服务器任务。
   - 另一位成员做出了回应，询问具体的 Bot 类型和服务器设置，并澄清 **Discord ChatGPT Bot** 是否是所需的解决方案。
- **明确 Bot 和服务器需求**：讨论始于一个构建用于角色扮演和服务器辅助的 Bot 的请求。
   - 回应的成员寻求澄清用户是否打算创建一个 **Discord ChatGPT Bot**，从而促使用户定义其具体需求。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1396929833413644430)** (742 条消息🔥🔥🔥): 

> `Qwen3 总大小，恢复训练问题，Open Empathic，Qwen3 2507 推理能力，GPTs Agents` 


- **量化囤积消耗带宽**：用户们开玩笑说，囤积像 **Qwen3** 这样模型的所有不同量化版本会消耗海量带宽，估计之前的模型发布版本总计超过 **3TB**。
   - 一位用户惊呼：*"先别管他们的，我为了囤这么多模型花了多少带宽"*。
- **训练 Checkpoints 导致推理灾难**：用户报告称，从 Checkpoint 恢复训练后结果显著变差，在使用 **vllm** 时，观察到使用和不使用 LoRA 的结果完全一致。
   - 他们正在寻求关于如何正确使用训练 Checkpoints 进行推理或恢复训练的指导，这表明 Checkpoint 加载或 LoRA 应用可能存在问题。
- **Qwen3 思考问题的临时修复 Hack**：用户发现 **Qwen3** Instruct 模型表现出了超出预期的推理能力，并开发了一个临时修复方案，涉及修改 Prompt 模板，在 `<REASONING>` 标签之间包含推理指令。
   - 一位用户分享了代码模板修复方案：*"这应该能让所有的思考过程都进入思考块内，但你必须相应地将思考标签更改为 `<REASONING>` 和 `</REASONING>`"*。
- **HF Transfer 极大提升下载速度**：成员们分享了一种使用 `hf_transfer` 库大幅提高从 Hugging Face 下载速度的方法，展示了高达 **112MB/s** 的速度。
   - 实现这一目标的代码片段包括将环境变量 `HF_HUB_ENABLE_HF_TRANSFER` 设置为 "1"，并使用 `huggingface_hub` 中的 `snapshot_download`。
- **Qwen3 模型表现出自信的推理行为**：用户发现 **Qwen3** 模型尽管是 Instruct 模型，却表现出了推理行为，引发了关于这是模型特性还是刻意设计的讨论。
   - 一些人观察到模型在自信地回答问题后，会出人意料地进入推理过程，仿佛在怀疑自己最初的回答，这表明其行为具有不一致性。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1397131547550941224)** (6 条消息): 

> `Minecraft AI 模型，Unsloth 使用情况` 


- **Minecraft AI 模型：Andy-4**：一位成员分享了他们开发的用于玩 **Minecraft** 的第四代 **AI 模型**，命名为 **Andy-4**，并提供了其 [Hugging Face 页面](https://huggingface.co/Sweaterdog/Andy-4)的链接。
   - 该成员已经使用 **Unsloth** 相当长一段时间来开发各种模型。
- **热情的社区欢迎 Unsloth 用户**：一位成员对看到另一位活跃在 Unsloth 社区的用户表示兴奋。
   - 他们表示 *很高兴在那里见到他们*。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1397011166672191569)** (19 条消息🔥): 

> `开源权重 SSM 模型，Falcon mamba，ART/OpenPipe 的 RULER，情绪后果，巧妙的生活黑客` 


- **关于开源权重 SSM 模型的辩论**：成员们讨论了 **6-12B 范围**的开源权重 **SSM 模型**，提到了作为 Mamba 混合体的 **Granite 4 7b**，以及同样是混合体的 **Falcon h-1**。
   - 其他提到的模型包括 **Falcon mamba 7b**、**Mistral mamba 7**、**Jamba** 和 **RWKV-6 7B**，但一位成员称 *从架构角度来看，rwkv 非常不稳定*。
- **RULER 实现**：一位成员询问如何使用 Unsloth 实现 **ART/OpenPipe 的 RULER**。
   - 该成员质疑是否必须专门使用 **ART 框架**，认为它 *不必要地复杂*。
- **情绪后果解释**：一位成员分享了 **Google Gemini** 关于情绪如何影响困惑感的解释，并附带了不同的例子。
   - 该帖子分析了**愤怒**、**悲伤**、**快乐**和**焦虑**作为对困惑的反应。
- **巧妙的生活黑客**：一位成员分享了过去两天的个人发现，并附有图片。
   - 另一位成员询问了一个特定课程，得到了 [jax-ml.github.io/scaling-book/](https://jax-ml.github.io/scaling-book/) 的链接回复。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1396944051345031278)** (47 messages🔥): 

> `Training Checkpoints, Multiple LoRA Adapters, Qwen3 Quants Issues, Falcon-7b Model, Qwen2.5-VL` 


- **Training Checkpoints 导致结果显著变差**：用户在从 Checkpoint 恢复训练后遇到了结果显著变差的问题，并且尝试将 Checkpoint 作为 **LoRA adapter** 加载时也未成功。
   - 有人指出 *vLLM* 在有无 **LoRA** 的情况下返回的结果完全一致。
- **多任务的 LoRA Swapping**：一位用户正在医疗数据上微调 Vision-Language Model，并希望使用多个 **LoRA adapters** 以在保留基础性能的同时增加新功能。
   - 他们询问了关于使用 **Unsloth** 进行 Adapter 控制以及是否根据用户输入加载 Adapter 的问题，得到的回复是 **LoRA swapping** 更多取决于推理库，这是一个尝试和纠错的过程。
- **Qwen3 Quants 尺寸异常问题已修复！**：一位用户报告了新出的 **Qwen3 quants** 的问题，指出部分尺寸似乎不合逻辑。
   - 问题已迅速解决，错误的 Quants 已被清除：*oh sorrythey shou;ld be fine nowwe eradicated the incorrect ones*。
- **Falcon-7b 层未转换**：一位用户报告称，在为 **Falcon-7b** 模型训练 **LoRA** 时，`layers_to_transform` 参数没有像预期那样改变参数量。
   - 他们注意到，无论针对多少层，参数占比始终保持在 **3.75%**。
- **Qwen2.5-VL 推理故障排除**：一位用户在使用提供的 Notebook 运行 **Qwen2.5-VL** 推理时遇到问题，当使用 **Qwen 的 messages 格式**时，模型会出现幻觉且无法识别图像。
   - 有人指出不应使用 `dataset_text_field = "messages"`。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1397316872441036981)** (1 messages): 

> `Reinforcement Learning Workshop` 


- **Unsloth 发布 3 小时 Reinforcement Learning 工作坊**：Daniel Han 在 [X 平台](https://x.com/danielhanchen/status/1947290464891314535)上宣布发布 **3 小时 Reinforcement Learning 工作坊**。
- **示例主题**：这是一个用于满足最少主题数量要求的示例。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1396957335984804002)** (4 messages): 

> `Custom Model Outputs, RULER by ART/OpenPipe, Fine-tuning a dataset to fine-tune a model` 


- **55M 模型输出异常**：一位成员报告称，一个从零开始在 **625kb 数据集**上训练的自定义 **55M 参数模型**出现了异常输出，并询问其他人在类似数据量下是否见过类似结果。
   - 另一位成员提醒不要轻信 AI 的断言，并幽默地警告可能存在的夸大妄想，*lol*。
- **ART/OpenPipe 的 RULER**：一位成员询问关于 **ART/OpenPipe** 的 **RULER**，询问是否可以使用 **Unsloth** 实现，或者是否需要 **ART 框架**。
- **微调模型**：一位成员分享了一种新颖的微调数据集的方法，通过抓取信息，使用 **AI agents** 进行正确对齐，并改善上下文以在教授复杂信息时提高准确性。


  

---

### **Unsloth AI (Daniel Han) ▷ #[unsloth-bot](https://discord.com/channels/1179035537009545276/1390899684834410536/1396966489424400545)** (54 messages🔥): 

> `Flash Attention with Qwen3, Finetuning Mistral with tool calls, RULER integration with Unsloth, Multimodal training error resolution, Audio fine-tuning error resolution` 


- ****Qwen3** 适配 **Flash Attention****：一位用户询问了如何在 **Qwen3** 模型上应用 **Flash Attention**。
- ****Mistral** 模型在 **Tool-Calling** 方面遇到麻烦**：一位用户在包含工具的数据上微调 **Mistral small 3.1** 时遇到困难，反馈该模型在与 **Langgraph agent** 配合使用时会遗忘如何进行工具调用。
   - 该用户正在寻找用于工具微调 **Mistral** 的 **Colab** 或 **Kaggle** 笔记本（类似于 **Qwen** 的笔记本），并提到他们已经检查了 [Unsloth notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks) 但未找到相关的。
- ****RULER** 需要在 **Unsloth** 中落地**：一位用户询问是否有办法使用 **Unsloth** 实现 **ART/OpenPipe** 的 **RULER**。
   - 他们想知道是否有必要使用 **ART** 框架，因为他们觉得该框架过于复杂。
- **解码音频微调（**Audio Fine-Tuning**）错误**：一位用户在 **Gemma3n** 音频微调过程中遇到了 **ValueError**，错误信息显示：*You should supply an encoding or a list of encodings to this method that includes input_ids, but you provided ['audio', 'text', 'messages']*，并[附带了](https://cdn.discordapp.com/attachments/1390899684834410536/1397185642584342618/message.txt?ex=6880ce07&is=687f7c87&hm=3210b97ad8327cf388bef80ff4b664ac360cb1d96ee1905f9b909b69e8fac2e2&)日志。
- ****Trainer Gym** 中的自定义损失函数（**Custom Losses**）**：一位用户询问了如何在 **GRPO Trainer** 中实现并记录自定义损失函数。


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1396954603505061949)** (4 messages): 

> `Intermittent 408 errors, DeepSeek v3 0324 Free Model, Chutes rate limits, Traffic Spikes, OpenRouter credits` 


- **流量激增期间调查 408 错误**：团队已注意到间歇性的 **408 错误**，并正在调查导致此问题的流量激增情况。
   - 这些激增可能是恶意攻击，团队对造成的问题表示歉意。
- **DeepSeek v3 免费版因高需求而宕机**：由于需求激增 **2 倍**，[DeepSeek v3 0324 免费模型](https://openrouter.ai/deepseek/deepseek-chat-v3-0324:free) 出现了宕机和不稳定，导致 **Chutes** 引入了频率限制（Rate Limits）。
   - 为了维持付费客户的稳定性，**Chutes** 不得不限制免费使用。
- **选择付费版 DeepSeek v3 以规避限制**：受免费版 **DeepSeek v3** 模型频率限制影响的用户，建议使用 [付费版 DeepSeek v3 端点](https://openrouter.ai/deepseek/deepseek-chat-v3-0324)，每次请求费用低于 **$0.004**。
   - 初始的 **10 个 OpenRouter 积分**可覆盖超过 **2,500 次请求**，且不影响每日 **1,000 次免费模型请求**。
- **启用“允许训练”的提供商以获取最便宜的 DeepSeek v3**：如果使用 **DeepSeek V3** 付费模型，用户可以通过在[隐私设置](https://openrouter.ai/settings/privacy)中开启 *'Enable providers that may train on inputs'* 来找到最便宜的提供商（**Chutes** 和 **Targon**）。
   - 这样做可以通过选择允许提供商对输入进行训练来显著降低成本。


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1397209529589960845)** (1 messages): 

> `YourChat.pro, T3.chat, ChatGPT` 


- **YourChat.pro 瞄准 T3.chat**：一位成员推介了 [YourChat.pro](https://yourchat.pro/)，将其作为 **T3.chat** 和 **ChatGPT** 的竞争对手。
   - 他们强调了 **OpenRouter 的强力支持**，并鼓励用户探索该应用。
- **YourChat.pro，一个志向远大的小应用**：该成员声称 YourChat.pro 拥有的功能将说服你放弃 T3.chat 甚至 ChatGPT。
   - 遗憾的是，该成员没有提供更多细节。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1396929803717837040)** (723 messages🔥🔥🔥): 

> `OpenRouter Free Tier, DeepSeek v3 Issues, Qwen3 Model Discussions, Chutes Rate Limiting, Model Censorship` 


- ****OpenRouter 的免费层级受到质疑****：用户正在讨论 **OpenRouter 免费层级** 的价值，由于速率限制（Rate Limiting）问题，有人指责存入 10 美元后可获得 1,000 次免费请求的承诺属于 *虚假广告*。
   - 一些用户认为，免费模型（尤其是 **DeepSeek v3**）的速率限制使其几乎无法使用，一位用户声称 **Chutes** 实际上是在变相向用户施压要求付费。
- ****DeepSeek v3 面临速率限制****：用户报告在使用 **DeepSeek V3 (free)** 时遇到 **408 错误** 和 **404 错误**，部分用户怀疑可能存在 **DDoS 攻击**。
   - 关于原始 **DeepSeek v3** 是否被 **DeepSeek v3 0324** 永久取代存在困惑，有报告称 **Chutes** 等提供商不再免费提供原始版本。
- ****Qwen3 的性能引起关注****：用户对新的 **Qwen3** 模型感到兴奋，讨论了其作为编程模型的潜力，并对比了 **Reasoning**（推理）和 **Non-reasoning**（非推理）版本。
   - 新的 **Qwen 3** 模型有 **5 次请求/分钟的速率限制**，一位用户指出它 *非常出色且免费*。
- ****Chutes 压力倍增，封禁用户****：一位用户报告因指出速率限制问题而被 **Rayon Labs/Chutes** 的服务器 *封禁*，暗示用户正被施压为服务付费。
   - 该用户抱怨 Chutes *会自动删除你发布的每一条消息且不留痕迹*，并表示 *Chutes 基本上是在逼你二选一：要么给他们付钱，要么给 DeepSeek 付钱*。
- ****模型审查？****：用户讨论了当前模型的 **Censorship**（审查）程度，有人声称 *在 OpenRouter 的任何模型上从未被审查过*，而另一些人则强调了使用正确的 Prompt 来绕过限制的重要性。
   - 一位用户指出，关于审查制度，*模型现在更难被稳定地越狱（Jailbreak）*。


  

---


### **OpenRouter (Alex Atallah) ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1396938258017161410)** (4 messages): 

> `` 


- **OpenRouter 频道 - 新模型初始化**：**OpenRouter - New Models** 频道已在 Readybot.io 上初始化。
- **Readybot.io 公告**：机器人 Readybot.io 宣布了 **OpenRouter - New Models** 频道的初始化。


  

---


### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1396933258280833104)** (72 messages🔥🔥): 

> `Window AI browser extension status, Native Search Functionality for Models, OpenRouter's Exa search Implementation, Modular Add-on System, Gemini 2.5 Flash Lite GA` 


- **Window AI 扩展接近生命周期终点**：由于使用率明显偏低，OpenRouter 团队正在考虑停止 **Window AI 浏览器扩展** 的服务。
   - 这可能导致 ST（推测为某个开发环境）中专用 **Window AI 源代码** 的移除。
- **OpenRouter 权衡原生搜索集成**：正如 [Toven 的推文](https://x.com/pingtoven/status/1947472209104064722) 所宣布的，OpenRouter 正积极致力于为 **OpenAI**、**Anthropic**、**Grok** 和 **Gemini** 等模型上线 **原生搜索功能**。
   - 关于是通过 Suffix（后缀）、支持的 Param（参数）还是 Plugin API 来实现此功能，目前存在争论。
- **Exa 搜索被指体验不佳**：一位用户表示 *Exa 搜索的实现很糟糕*。
   - 建议将 **Exa 搜索** 作为 **Tool**（工具）输入，以便 LLM 每次都能发起搜索查询。
- **提出模块化插件系统框架**：一名成员建议建立一个模块化插件系统，用户可以在保持一致的 **Exa** 体验的同时，将 **Grok** 替换为 **Claude**。
   - 这可以扩展到实现 **Moderation**（审核）功能，尽管有人对成本和实现方式表示担忧。
- **Gemini 2.5 Flash Lite 正式发布 (GA)**：根据 [Google Developers 博客文章](https://developers.googleblog.com/en/gemini-25-flash-lite-is-now-stable-and-generally-available/)，**Gemini 2.5 Flash Lite** 已随同预览模型一起进入 GA 阶段，并提供一个月的时间进行迁移。
   - 这可能会促使别名的产生，尽管有人对 *推理税（Thinking Tax）* 表示担忧。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1396929727436034231)** (672 条消息🔥🔥🔥): 

> `IMO 数学竞赛, DeepThink 发布, Qwen3 模型, Grok4 Coder` 


- **IMO 数学竞赛风波发酵**：一名成员分享道，在一家 AI 公司试图解答 **IMO** 数学题后，另一家公司在闭幕式当天于社交媒体上大肆宣传其结果，引发了数学界的负面反应。
   - 据指出，**OpenAI** 未配合 **IMO** 的评分流程，可能进行了多次尝试并在闭幕派对前发布了结果，令 **IMO** 委员会感到愤怒。 
- **DeepThink 发布日期仍是谜**：有推测称 **DeepThink** 的发布日期可能与 **IMO** 的禁令解除同步，并可能在 7 月 28 日左右公布或发布。
   - 成员们讨论了 **DeepThink** 不同版本的可能性，包括用于 **IMO** 的定制版本以及与 **Gemini Ultra** 集成的潜在可能。
- **Qwen3 模型：性能与托管质量引发争议**：成员们讨论了 **Qwen3-235B-A22B** 模型的性能，一些人称赞其在后训练（post-training）和长文本输出方面的改进，而另一些人则认为其推理能力不足且受限于模型规模。
   - 官方 **Qwen** 网站的托管质量也受到质疑，用户抱怨其量化和质量问题，并将其与 **Parasail** 等提供全精度托管的选项进行了对比。
- **Grok4 Coder 的热度与潜在的失望**：部分成员对 **Grok4 Coder** 表示期待，认为它可能成为行业的颠覆者，而另一些人则提醒不要根据通用模型的表现对其过度炒作。
   - 也有人担心 **Grok4 Coder** 与其他模型一样，可能只是针对特定基准测试进行了训练，而在实际编程场景（尤其是 Web 开发）中表现不佳。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1396929819236765777)** (323 条消息🔥🔥): 

> `聊天终端卡死, Gemini 充值, Kimi K2 速度, ChatGPT 夺走灵魂, Cursor Pro 计费` 


- **聊天终端卡死**：有用户报告在没有向任何方向推进时出现[聊天终端卡死](https://cdn.discordapp.com/attachments/1074847527708393565/1396934986266841168/Screenshot_2025-07-21_142031.png?ex=68813616&is=687fe496&hm=577a0868176a7abda708b91d4b0284b818e42695fbe551b818a81bf56a9395ca)的情况。
   - 一位用户调侃道：*谢谢 Gemini，我才刚开始真正喜欢上你*。
- **Grok 3 优于 Grok 4**：一位用户报告称 **Grok 3 Mini** 作为“无请求模型”（no request model）远好于 **Grok 4**，比 **Grok 4** 更可靠，且优于 **GPT 4.1** 或 **2.5 Pro**。
   - 另一位用户回复称，他们*无法想象 Grok 3 Mini 在编程中会有用*。
- **用户费率限制已解决**：一位用户报告称之前的费率限制已解除，现在可以使用他们付费购买的服务，能够使用超过 50 次 **Sonnet** 请求。
   - 该用户计划在超过 **500 次请求**后向小组更新情况，并提到他们对新方案仍感到困惑。
- **请求集成 Qwen3 235B A22B**：用户请求在 Cursor 中加入 [Qwen3 235B A22B](https://forum.cursor.com/t/qwen3-235b-a22b-instruct/121002)，并推动对 **MAX** 和 **Agent 模式**的支持。
   - 用户指出 Qwen3 模型优于 **KimiK2**，在某些基准测试上与 **Opus4** 持平，且成本仅为后者的一小部分。
- **自动使用量（Auto Usage）令用户困惑**：用户对新的“使用量”（Usage）仪表盘感到困惑，特别是它如何计入自动使用量。
   - 他们请求将 **自动使用量** 单独追踪或从总额中剔除，一位用户总结道：*开发者不信任 Auto-mode*。


  

---

### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1397059949841879040)** (4 messages): 

> `Background Agent 质量, 在 Slack 中使用 Cursor 自动化 Linear Issues, Background Agent 中的对话长度错误` 


- ****Background Agent 质量引发讨论****：成员们正尝试自动化在 **Slack** 中呈现 **Linear issues**，并让 **Cursor** 通过 Agent 命令对其做出响应。
   - 存在一个关于 **Background Agent** 的质量与 **Claude Code** 相比如何的问题。
- ****Cursor 忽略 Slack API 命令****：一位用户报告称，在自动化 **Linear issues** 时，**Cursor** 不响应通过 **Slack API** 发送的 Agent 命令，尽管手动命令可以正常工作。
   - 他们尝试同时使用 **Slack bot user** 和 **user token**，并匹配了手动消息的格式，但 **Cursor** 仍然没有反应，这引发了关于潜在设置或变通方法的疑问。
- ****Background Agent 受困于对话长度过长****：用户报告在 **background agent** 中看到错误提示的频率增加：*Your conversation is too long. Please try creating a new conversation or shortening your messages*（您的对话太长，请尝试创建新对话或缩短消息）。
   - 用户报告称，即使在选择 *Start New Thread With Summary*（带摘要开始新线程）后，他们也 *无法使用此 bg agent 继续对话*。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1396934293305167914)** (158 messages🔥🔥): 

> `YaRN 与 LM Studio + MLX, 删除未使用的后端, 编程锦标赛 AI, 眼科微创手术 AI 应用, Nvidia Orin 与 LM Studio 的兼容性` 


- **YaRN 与 LM Studio 的集成仍是个谜**：成员们正在寻求将 **YaRN** 与 **LM Studio** 及 **MLX** 集成的指导，并指出网上虽有提示但缺乏示例或指南。
   - 似乎在 **LM Studio** + **MLX** 中使用 **YaRN** 是可能的，但目前没有可用的示例或指南。
- **自动删除未使用的后端可节省数 GB 空间**：一位成员询问删除 **LM Studio** 中未使用的后端以释放 **4GB** 空间是否安全。
   - 另一位成员指出，有一个设置可以 *自动* 删除未使用的后端。
- **人类程序员击败 AI**：一位成员分享了一篇文章（[疲惫的人类在世界编程锦标赛中击败 AI 模型](https://arstechnica.com/ai/2025/07/exhausted-man-defeats-ai-model-in-world-coding-championship/)），文中一名人类程序员在编程锦标赛中击败了 AI 模型。
   - 另一位成员开玩笑说，明年人类将面对 *10 个 Grok 6 agents*。
- **眼科微创手术 AI 应用**：一位研究人员寻求关于使用 **AI** 分析 **30 万** 张眼部图像以提前数年预测疾病的建议，旨在开发一款用于早期检测的 App。
   - 研究人员强调了 **AI** 在识别传统方法无法检测到的细微细胞变化方面的潜力，例如在糖尿病变得不可逆转的 *10 年前* 预测其发生。
- **LM Studio 与 Nvidia Orin 不兼容**：一位用户询问在运行 **Ubuntu** 的 **Nvidia Orin** 上安装 **LM Studio** 的事宜，但该系统并非 **armx64**。
   - 一位成员澄清说，**LM Studio** 的 **Linux** 版本仅支持 **x86 CPU**，不支持 **Linux ARM**。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1397020852935069871)** (67 条消息🔥🔥): 

> `3090 vs 4080, SnapDragon X1 Adreno GPU, 5090 for RP, DeepSeek R1 70B, Gemma 用于创意写作` 


- **3090 vs 4080 硬件规格**: 一位用户对比了 **3090**（936.2 GB/s 带宽，10496 CUDA 核心）与 **4080**（716.8 GB/s 带宽，9728 CUDA 核心），指出前者在关键部位进行了“精简”。
   - 另一位使用 **3080ti** 笔记本电脑的用户询问，对于图像/视频生成，使用 **RTX pro 6000** 还是堆叠多张 **3090** 效果更好。
- **SnapDragon X1 Adreno GPU 在 Vulkan 上表现挣扎**: 有用户报告称，在 **arm64** 架构下使用 **SnapDragon X1 Adreno GPU** 的 Vulkan 运行时无法工作，显示“硬件检测错误（error surveying hardware）”。
   - 据报道其他 Vulkan 应用可以正常运行，另一位用户建议尝试 [OpenCL backend](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENCL.md)。
- **显存支持运行 DeepSeek R1 70B**: 一位用户表示 **56GB VRAM** 让他们能够运行 **DeepSeek R1 70B Q5_K_M**，而这在 **24GB** 或 **32GB** 上无法运行，但该模型仅在某些方面表现更好，且重复性非常高。
   - 另一位用户提供了其他可尝试的替代模型，如 **Mistral Large**、**Llama Scout** 以及具有更大上下文窗口的 **Qwen3 235B**。
- **本地模型用于私密聊天？**: 一位用户表示不愿使用像 **Gemma** 这样的云端模型，因为担心其数据被利用的隐私问题。
   - 另一位用户说：“我（只）买了 96GB DDR5 6400MHz，原以为这有助于提高卸载速度（offload speed）……结果并没有。”
- **Intel Core Ultra 9 加入讨论**: 一位用户提到了搭载 **Arc™ 140T** 显卡的 **Intel Core Ultra 9 285H Mini PC --EVO-T1 AI** 已上市。
   - 他们注意到 [Minisforum M1 Pro-285H](https://cdn.discordapp.com/attachments/1153759714082033735/1397334463851135016/Captura_de_pantalla_2025-07-22_154300.png?ex=688158a1&is=68800721&hm=bc6882e2e0834f53567c3e35c4dfe6f92936abf27354c7aa33d1aae2b93e0834) 也配备了这款新 GPU，但目前已售罄。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1396952270872248353)** (106 条消息🔥🔥): 

> `Kimi K2 报告, Agent 失效模式, Humanloop 停止运营, Cognition 估值, Turbopuffer Pod` 


- **月之暗面 MoonshotAI 的 Kimi K2 表现亮眼**: 来自 MoonshotAI 的 [Kimi K2 报告](https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf) 被认为是一个极具前景的项目，弥合了推理模型与非推理模型之间的差距。
   - [这条推文](https://x.com/yunyu_l/status/1946261211915468884) 对该项目进行了进一步讨论。
- **Agent 架构评估**: 讨论围绕当前 LLM 架构在不强制编写确定性代码的情况下解决网络问题的能力展开。
   - 有建议称，带有分层纠错功能的自定义框架可以提高性能，但也有人认为目前的基准测试优先考虑娱乐性而非衡量实质性进展，并以 [Vending-Bench](https://arxiv.org/abs/2502.15840) 为例。
- **Humanloop 停止运营**: AI 工具公司 **Humanloop** 即将关闭，已通过电子邮件通知客户，但未发布任何公开公告。
   - 有关此事的讨论可以在 [Hacker News](https://news.ycombinator.com/item?id=44592216) 上找到。
- **Cognition 获得 100 亿美元估值**: 一位成员指出，应用 AI 公司 **Cognition** 已达到 **100 亿美元** 的估值。
   - 更多细节见[这条推文](https://x.com/arfurrock/status/1947440754885882120?s=46)。
- **Bee 联合创始人加入 Amazon**: 可穿戴个人 AI 公司 **Bee** 被 **Amazon** 收购，其联合创始人随之加入 Amazon。
   - 虽然有人为该团队感到高兴，但也有人对隐私影响表示担忧，特别是考虑到 Amazon 已通过 **Alexa** 等服务进行数据收集，正如[这篇文章](https://www.theverge.com/news/711621/amazon-bee-ai-wearable-acquisition)和[这篇文章](https://seekingalpha.com/news/4470117-amazon-acquires-wearable-personal-ai-company-bee)所报道的那样。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1396970200556900362)** (30 条消息🔥): 

> `C++ templated libraries, Kog's inference benchmark on AMD MI300X, AMD MI300X Inference, vLLM optimization on MI300X, Automated bans for image spam` 


- **C++ 模板代码通常在头文件中定义**：在头文件中定义函数/实现对于 **templated libraries**（模板库）来说很常见，部分原因是历史上缺乏标准化的 C++ 包管理器或构建系统。
   - 自 **C++11** 以来，已经可以在头文件中声明 *extern* 模板特化，而将实际实现放在 .cpp 文件中。
- **Kog 在 AMD MI300X 上达到 3.5 倍速度**：法国初创公司 Kog 在 **AMD MI300X** 上实现了 **3.5 倍的推理速度突破**，该结果已正式发布在 AMD 官方博客上：[Kog's benchmark](https://www.amd.com/en/blogs/2025/kog-reaches-3-5x-breakthrough-inference-speed-on-amd-instinct-mi.html)。
   - 其目标是在 **6 个月内实现 10 倍速推理**，并持续致力于突破推理极限。
- **MI300X vLLM 优化工作流**：针对 vLLM 的 MI300X 优化工作流涉及 **FP8 KV-cache**、**GEMM autotuning**，并遵循 [ROCm blog guide](https://rocm.blogs.amd.com/artificial-intelligence/vllm-optimize/README.html) 中详述的步骤。
   - 遍历 [vLLM docs](https://docs.vllm.ai/en/latest/configuration/optimization.html) 中每个与性能相关的环境变量有助于达到峰值吞吐量，这与在 NVIDIA GPU 上进行测试时的方法类似。
- **瓶颈仍在于内存带宽和 kernel 启动**：进一步调查显示，内存带宽和 kernel 启动是主要瓶颈。
   - CU 占用率（occupancy）并不是很高，主要在 batching 场景下才具有相关性。
- **用户因图片垃圾信息被封禁，随后迅速恢复**：一名用户因 **image spam**（图片垃圾信息）被自动封禁，但在提供 Discord ID 后迅速解封。
   - 该用户经常分享包含图片的学习进度，这触发了系统提醒，建议减少图片发布频率以避免未来再次被自动封禁。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/)** (1 条消息): 

pekaro: lmao, who they are lying to, themselves? sucky move to publish something like this
  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1397032134656393486)** (1 条消息): 

> `Torch compilation, stride assertions` 


- **Torch 会随机转置！**：一位成员表达了“痛苦”，因为 **torch** 可能会决定在编译代码时进行随机转置并移除检查。
   - 他们建议将对 stride 的断言作为一种万无一失的方法。
- **Stride 断言化险为夷**：为了确保代码的可靠性，建议对 stride 进行断言，以作为防止意外转置的安全机制。
   - 这种方法通过在运行时验证内存布局假设，有助于维护数据完整性。


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1397040582656004210)** (2 条消息): 

> `Kog, AMD MI300X, Inference Speed, French startup` 


- **Kog 在 AMD MI300X 上取得推理突破**：法国初创公司 **Kog** 在 **AMD 官方博客**上发布了关于 **AMD MI300X** 推理的基准测试，实现了 **3.5 倍的突破性推理速度**。
   - Kog 旨在 **6 个月内**实现 **10 倍速推理**，目前正在招聘，详见其 [job postings](https://www.kog.ai/jobs)。
- **Kog 招聘公告**：法国初创公司 **Kog** 宣布正在招聘。
   - 他们正在挑战推理性能的极限。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1396946942277451847)** (2 条消息): 

> `GCP, Google Colab` 


- **成员建议可以使用 Google Colab**：成员提到可以使用 **GCP** 甚至 **Google Colab**。
   - 另一人补充说 **Google Colab** 是首选方案，并在另一个频道中添加了链接。
- **分享了 Colab 链接**：在频道 <#1394829095271141387> 中分享了一个关于使用 **Google Colab** 的链接。
   - 提供的上下文中未详细说明链接的具体内容。


  

---

### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1397288882361335938)** (5 messages): 

> `FP8 Training, DDP Training, FSDP2, Activation Checkpointing` 


- **Axolotl 集成 FP8 训练**：一名成员正在 Axolotl 中集成 **FP8 训练**，参考了 [这个 PR](https://github.com/pytorch/torchtune/pull/2546)，并依赖于 Accelerate 的实现。
   - 他们正在寻求关于 DDP 训练以及 FSDP2 + torch.compile + FSDP 的 activation checkpointing 可能导致的减速问题的见解。
- **DDP 训练与 Torch.Compile 错误**：在启用 **DDP** + **torch.compile** + **FP8** 时，用户遇到了一个与张量元数据（tensor metadata）相关的 [torch.compile 错误](https://gist.github.com/djsaunde/0c1664c32e44a64d31b5e01b4aafe5c4)。
   - 团队请求用户提供复现错误的方法，因为团队之前未见过此错误，但乐意协助修复。
- **FSDP2 与 Activation Checkpointing 性能**：用户观察到，在 Axolotl 配置中，**FSDP2** + **torch.compile** + **FSDP 的 activation checkpointing** 与 BF16 训练相比出现了减速。
   - 禁用 activation checkpointing 后获得了预期的加速，他们使用了与 [此处定义](https://github.com/axolotl-ai-cloud/axolotl/blob/b86a1d47b02a7f9c31199370b2724f0e1d0e3941/src/axolotl/monkeypatch/accelerate/fsdp2.py#L236) 相同的 `apply_activation_checkpointing` 函数。
- **FSDP2 的无缝体验**：一位用户反馈 **FSDP2** 的使用过程非常顺畅。
   - 另一位用户对此表示感谢。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1397259869769306233)** (2 messages): 

> `Neuralink, Brain-computer interfaces` 


- **Neuralink 分享“今日所学”片段**：Neuralink 分享了一系列图片作为“今日所学（today I learned）”系列的一部分，展示了与其 **脑机接口（brain-computer interfaces）** 工作相关的各种见解。
- **更多 Neuralink 见解**：分享的图片包含了关于 Neuralink 正在进行的项目和该领域研究的更多细节和背景。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1397205543306657893)** (1 messages): 

> `Register Corruption, SGPR Allocation` 


- **通过 ASM Volatile 导致的寄存器损坏**：使用 `asm volatile` 可能会导致寄存器损坏，例如指令 `asm volatile ("s_add_i32 s100, s100, 1")`。
   - 如果代码对象（code object）分配的 SGPR 数量不超过 100 个，即使代码的其余部分从未与 `s100` 交互，也可能发生意外行为。
- **SGPR 分配困扰**：如果一个代码对象分配的 SGPR 少于 100 个，`asm volatile` 指令可能会导致意外行为。
   - 这是因为该指令可能会访问未正确分配的寄存器，即使代码的其余部分避开了这些寄存器。


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1397083770686800003)** (1 messages): 

> `ThunderKittens, Attention Backward, LCF support` 


- **新成员加入 ThunderKittens 并贡献代码**：**ThunderKittens** 的一名新成员正在熟悉环境，并尝试使用 **ThunderKittens LCF** 实现 **Attention Backward**。
   - 该新成员希望将代码贡献给社区，并提交了一个包含简单版本代码的 [Draft PR](https://github.com/HazyResearch/ThunderKittens/pull/135)。
- **ThunderKittens 考虑为 Attention Backward 提供 LCF 支持**：该新成员询问 **ThunderKittens** 是否计划为 **Attention Backward** 添加 **LCF 支持**。
   - 他们标记了一名特定成员，推测是为了了解路线图（roadmap）详情。


  

---

### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1397118518650208318)** (30 messages🔥): 

> `System Prompt 长度, RCON 错误, 传送带物品显示, 速率限制` 


- **System Prompt 的简洁性辩论**：成员们讨论了 System Prompt 是否过长，有人建议如果性能稳定，更短的上下文会更好，但也有人担心缩减过多会导致性能下降。
   - 此前已进行过广泛评估，尽管没有正式的消融实验（ablations）；难点在于需要提供足够的 **Factorio 特有知识**，以避免因隐藏的环境动态而导致的失败。
- **JackHopkins 的 Pull Request 已合并**：一个 Pull Request ([#276](https://github.com/JackHopkins/factorio-learning-environment/pull/276)) 已合并，为 [factorio-learning-environment](https://github.com/JackHopkins/factorio-learning-environment) 仓库添加了剩余的实验室运行任务。
   - 此次更新使得原指令中关于工厂吞吐量和维护的部分变得冗余，因为 *Agent 在成功后无论如何都会停止*。
- **Anthropic API 出现过载错误**：在以 -10 的值达到 50 次迭代后，Agent 在 Anthropic API 上使用 Claude Sonnet 4 20250514 时遇到了 `529` **Overloaded** 错误。
   - Agent 当时正在向 `https://api.anthropic.com/v1/messages` 发送 POST 请求。
- **RCON 错误调查**：一名成员在并行运行任务约 25 次迭代后遇到了 **RCON 错误**，目前正在调查原因。
   - 尽管能够将任务分成 8 个一组并并行运行且初始没有速率限制问题，但错误仍然发生了。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1396946155962634422)** (2 messages): 

> `分层布局 (Hierarchical Layouts), 张量重塑 (Tensor Reshaping), MMA Atoms` 


- **分层布局：解锁张量变换**：分层布局（Hierarchical Layouts）能够表达简单布局无法表达的张量布局，例如通过使用分层形状和步长表示（如 `Shape=((2,2),2), Stride=((4,2),1)`），将 **2x2x2 张量**重塑为 **4x2 张量**。
   - 这种方法提供了可组合性，允许进行诸如将一个维度除以常数（例如 **Warp Size**）并对结果部分进行不同处理的操作，这在 **MMA Atoms** 等优化 Kernel 中至关重要。
- **张量重塑需要分层布局**：将具有 `Shape=(2,2,2)` 和 `Stride=(4,1,2)` 的 **2x2x2 张量**重塑为具有 `Shape=(2,4)` 和 `Stride=(4,1)` 的 **2x4 张量**非常直接，但将其视为 **4x2 张量**则需要分层布局。
   - 由于内存布局限制，**4x2 张量**需要分层布局 `Shape=((2,2),2)` 和 `Stride=((4,2),1)`。
- **MMA Atoms 依赖分层布局**：分层布局对于将 **MxN 维数组**划分为 **(M, N/32, 32)** 组非常有用，其中 **32** 代表 **Warp Size**。
   - 这种划分允许在对 **32** 进行并行化的同时遍历 **N/32**，确保维度 **32** 是连续的，这在 **MMA Atoms** 的布局中被大量使用。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1396939224158044280)** (58 条消息🔥🔥): 

> `Hugging Face Hub 问题, Kubernetes 上的 Wandb 替代方案, Dalle-mini 流量问题, 对年轻 ML 爱好者的建议, Shell.ai Hackathon 2025 组队` 


- **Hugging Face Hub 遭遇机器人攻击**: 成员们报告称，由于疑似大量 **bot** 活动的涌入，**Hugging Face Hub** 及其端点在 *过去几天里一直非常不稳定*。
   - 具体而言，`HfApi.list_repo_commits` 返回的响应不完整，仅显示第一页，推测这是后端为对抗机器人激增而采取的措施。
- **Grafana, Wandb 和 Kubernetes 的纠纷**: 一位成员正在寻找可以托管在 **Kubernetes** 上的 **Wandb** 仪表板替代方案，因为他们怀疑 Wandb operator 只是 *伪装成在本地运行*。
   - 另一位成员建议使用他们自己托管在 **Hugging Face Spaces** 上的 [Plotly 实现](https://huggingface.co/spaces/Tonic/smollm3_test_11) 作为轻量级替代方案，并征求对其云托管方法的反馈，同时好奇 *为什么人们会使用 wandb*。
- **Dalle-mini 遭遇流量困扰**: 用户在尝试使用 **dalle-mini** 时，持续遇到 *流量过大，请稍后再试* 的提示，这种情况自周五晚间起一直持续。
   - 一位成员认为 *可能有人在攻击那个 Space*，并提供了讨论区的 [链接](https://huggingface.co/spaces/dalle-mini/dalle-mini/discussions)。
- **编程新手探索神经网络**: 一位有编程经验的 17 岁少年寻求关于获得远程工作或实习的建议，以便在学习 **ML** 背后的数学并记录进度时能够自食其力。
   - 建议包括在小组中验证学习成果、使用白板、进行公式推导、加入 [cohere 学习社区](https://cohere.com/community)，以及保持持续的记录和自信。
- **Shell.ai Hackathon 2025 寻求组队**: 一位成员询问关于为 **Shell.ai Hackathon 2025** 组建团队的事宜。
   - 然而，另一位用户指出 [官方网站](https://www.shell.com/energy-and-innovation/shell-ai-accelerator/shell-ai-hackathon.html) 显示现在参加可能已经太晚了。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1397007365936775348)** (4 条消息): 

> `JAX ML Scaling Book, 医疗 AI 影像的未来` 


- **来自 JAX ML Scaling Book 的 TPU 见解**: 一位成员分享了 [JAX ML Scaling Book](https://jax-ml.github.io/scaling-book/tpus/) 的笔记，重点关注 **TPUs** 及其应用，如随附图片所示。
   - 随附图片似乎是该书内容的截图，直观地概述了与 **TPU** 使用和扩展相关的核心概念和信息。
- **医疗 AI 影像正在转型**: 一位成员讨论了 **医疗 AI 影像** 的未来，强调关键点在于 *我们选择如何利用基于 AI 构建的成果*，而不仅仅是构建 **AI** 模型本身，如随附图片所示。
   - 该成员认为，**AI** 在医疗影像中实现的变革比实现这些变革的手段更为重要。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/)** (1 条消息): 

tejasshinde400: https://github.com/jujumilk3/leaked-system-prompts
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1397002382247989329)** (4 条消息): 

> `PDF 转数据集工具, FaceFlux Deepfake 工具` 


- **新工具将 PDF 转换为数据集**: 一个名为 [m1f](https://m1f.dev/blog/introducing-m1f/) 的新工具可以通过提示词将杂乱的 **PDFs**、扫描图像或 **DOCX** 文件转换为结构化数据集。
   - 例如，该工具可以 *“从这份化学 PDF 中提取所有选择题及其答案。”*，可在此处进行试用 [here](https://pdf2dataset.streamlit.app/)。
- **FACEFLUX Deepfake 工具发布：最快、免费、无限、无需 GPU**: 一款名为 [FACEFLUX](https://huggingface.co/spaces/NihalGazi/FaceFlux-Face-Swapper) 的新 Deepfake 工具提供免费、无限的换脸功能，无需 **GPU**、**GAN** 或 **Diffusion**。
   - 它在最低设置下可达到 *10FPS*，并能处理任何光照条件和面部表情，尽管存在轻微的伪影。


  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1397282478036619305)** (3 messages): 

> `SetFit OOM issues, Jina Embeddings v2 base, CosineSimilarityLoss, ContrastiveDataset, Triplet` 


- **SetFit 在使用 Jina Embeddings 时面临 OOM 问题**：一名成员报告在微调 [jinaai/jina-embeddings-v2-base-de](https://huggingface.co/jinaai/jina-embeddings-v2-base-de) 模型时遇到了 **OOM** 问题。
   - 他们指出，即使数据集很小（**32 个样本**），且限制了 batch size（**4**）和序列长度（**512**），内存问题依然存在，特别是在使用 **ContrastiveDataset** 配合 **CosineSimilarityLoss** 或 **Triplet** 时。
- **Merge Request #579 成为必需**：一名成员必须实现来自 [PR #579](https://github.com/huggingface/setfit/pull/579) 的更改才能让 **SetFit** 运行。
   - 实现的更改允许该成员在遇到内存问题之前，使用更大的数据集开始训练。
- **嵌入模型微调涉及内存问题**：该成员确定微调嵌入模型是内存问题的主要原因。
   - 将微调限制在 **5 个 steps** 内，使他们能够使用所有样本训练分类器。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1396930991641198742)** (52 messages🔥): 

> `Arxiv endorsement, DeepMind vs OpenAI in AI math, Language for solving math problems, Cross entropy loss` 


- **Arxiv 论文请求背书以切换类别**：一名成员请求背书，将其 [Arxiv 论文](https://arxiv.org/abs/2507.13362) 从 **cs.CV** 移至 **cs.CLI**，并询问将此类内容添加到 **LLM training** 或 **RL data** 是否有益。
   - 在最初对分享身份信息和无法私信表示顾虑后，他们随后公开了该内容。
- **DeepMind 在数学竞赛中领先 OpenAI**：一名成员分享了一个来自[数学版块的链接](https://link.springer.com/article/10.1007/s10864-024-09559-3)，表明 **DeepMind** 在 **AI math** 方面的表现优于 **OpenAI**。
   - 附带的一张图片表达了一个观点：英语并不是解决数学问题的最佳语言，并暗示 **AGI** 将生成人类无法理解的代码。
- **辩论爆发：AI 会说“外星语言”吗？**：一名成员表示 OpenAI 的风格*为了节省 token 而更加压缩，因此未来可能会从类人语言转向更多分布外（out of distribution）的领域。*
   - 其他人指出，形式数学已经使用了普通英语读者无法理解的符号；随后有人分享了 [Chain of continuous thought 论文](https://x.com/fifty_chris/status/1947092589041082436) 的链接和 epsilon delta 定义。
- **质疑语言模型中的极端压缩水平**：成员们对 **0.6-1.3** 的语言建模 **cross entropy loss** 表示质疑，讨论其在万亿参数规模下的可行性。
   - 讨论澄清了该数字指的是**每个字符的损失 (loss per character)**，而非每个 token，并指出了一条可能带有标题党性质的推文和一篇[新论文](https://arxiv.org/abs/2507.15855)。
- **AI 加强版诈骗：迫在眉睫的威胁？**：一名成员讽刺地表示，当前这一代生成式 AI 产生的杀手级应用将是加强版诈骗。
   - 另一名成员强烈反对，认为 AI 的益处将超过危害。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1396934729411989666)** (13 messages🔥): 

> `Deepseek V3, MuonClip Algorithm, smolLM3 implementation details, Model Merging` 


- **Deepseek V3 架构推测**：成员们推测下一代 **Deepseek V3** 模型可能会采用与之前版本类似的架构。
- **发现了 MuonClip 算法？**：一名成员认为 **MuonClip** 背后的算法可能是*核心价值（the gold）*。
   - 他们发布了 [AlphaXiv](https://www.alphaxiv.org/abs/2505.12082) 的链接以获取更多信息。
- **smolLM3 实现细节披露**：一名成员分享了关于 **smolLM3** 的 [Hugging Face 博客文章](https://huggingface.co/blog/smollm3) 链接，称其为最好的 **LLM** 实现论文之一，因为*一切都是完全开源的，数据集也是如此*。
- **模型合并资源分享**：同一名成员指出，**smolLM3** 博客文章包含了关于 **model merging** 的细节，并且比大多数顶级模型论文包含了更多关于*如何实际操作*的信息。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1396951856474882264)** (4 messages): 

> `OpenAI 英国协议, 愚蠢步态部, AI YouTube 视频` 


- **英国政府与 OpenAI 签署协议**：[英国政府已与 OpenAI 签署协议](https://www.theguardian.com/technology/2025/jul/21/openai-signs-deal-with-uk-to-find-government-uses-for-its-models)，旨在探索 **AI 模型** 在政府职能中的潜在应用。
- **愚蠢步态数字化**：愚蠢步态部（Ministry of Silly Walks）终于将实现数字化。
- **由 Facebook, Palantir, Google, Salesforce, OpenAI, Cognition, XAI 提供支持的视频**：一位成员分享了一个由 **Facebook, Palantir, Google, Salesforce, OpenAI, Cognition, XAI** 提供支持的 [YouTube 视频](https://www.youtube.com/watch?v=WByBm2SwKk8)。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1397174708310642761)** (11 messages🔥): 

> `lm-eval-harness, AI 数学奥林匹克, NAIRR 计算资源, AI 安全` 


- **关于 *lm-eval-harness* 的咨询**：一位成员询问哪个频道适合提问关于 **lm-eval-harness** 的问题。
- **AI 攻克数学奥林匹克，除第 6 题外**：**OpenAI** 和 **DeepMind** 的模型在几天前的**国际数学奥林匹克**（International Math Olympiad）中均获得金牌，但仍未能解决第 6 题。该题比其他题目需要更多的创造力，这为需要更多方法论提供了很好的数据点。
   - 一位成员同意 Kenneth Stanley 的观点，即*我们仍未解决许多创造性和开放性问题，因此 AI 自动化数学研究可能不会很快实现，因为这比解决国际数学奥林匹克题目需要更高的开放性。*
- **NairrPilot 启动研究**：[NairrPilot](https://nairrpilot.org/opportunities/startup-project) 正在为期 **3 个月的项目**提供计算资源，旨在引导研究人员使用 **NAIRR** 提供的资源，最终目标是扩大能够熟练利用 **NAIRR** 计算资源的研究人员社区。
- **AI 奇点危险**：一位成员警告说，*如果只需要一个自作聪明的人给这种 AI 下达指令来产生更好的 AI，就可能触发危险类型的奇点（即人类只能在惊叹中旁观且一无所知，而不是与机器共同进步）*。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1396945855604588664)** (8 messages🔥): 

> `Embedding 中的权重衰减, Norm 层缩放, Kimi k2 论文, 同步训练` 


- **权重衰减之谜：Embedding 篇**：一位成员质疑，当 **Weight Decay** 应用于所有 **Embedding** 和权重矩阵时，为什么它不会直接缩小所有内容，并让 **Norm** 层处理重新缩放。
   - 他们指出，大多数开源训练代码库*确实*对 **Embedding** 应用了 **Weight Decay**，这引发了关于这种做法影响的讨论。
- **移除 Norm 缺失引用？**：一位成员注意到，[一篇移除 Norm 的论文](https://arxiv.org/abs/2302.10322)在相关讨论中未被引用。
   - 他们强调该论文也移除了 **Norm**，尽管标题中没有明确说明，这使其与对话相关。
- **Kimi k2 中的异步训练**：一位成员对 **Kimi k2** 论文详细描述 **Synchronous training**（同步训练）而非分布式方法表示惊讶。
   - 他们将其与采用分布式训练的 **Magistral** 进行了对比，质疑 **Synchronous training** 是否是一个较少见的方向。


  

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1397073790093103145)** (13 messages🔥): 

> `KAN Activation Functions, B-Spline Curves, Training Dynamics Optimization, Expressivity vs Stability, Cell 9 Spline Training` 


- **调整 KAN 激活函数**：一名成员对 **KAN** (*Kolmogorov-Arnold Networks*) 的激活函数训练方法提出了质疑，指出虽然论文提到了 **B-spline 曲线**，但训练动力学（training dynamics）可能存在病态条件（poorly conditioned）的问题，尤其是在节点或控制点较多时。
   - 该成员建议，在两个最近的 spline 之间使用线性插值而不是求和，可以提高训练和推理速度，并认为这是优化的切入点。
- **非线性 vs. 规模化训练**：一位成员指出，**KANs** 极端的非线性可能会阻碍其可扩展性（scalability）。
   - 另一位成员反驳称，非线性增强了表达力（expressivity），引发了关于平衡表达力与稳定性的讨论，原成员表示：“这是表达力与稳定性之间永恒的张力”。
- **重新参数化 Spline 训练以加速 KANs**：一名成员建议重新参数化 **KANs** 中的 spline 训练方法以改善条件数，即使这会带来轻微的计算开销。
   - 该成员链接到了其 [ML-code 仓库中的 Cell 9](https://github.com/cats-marin/ML-code/blob/main/LearnableActivation.ipynb)，展示了他们的 spline 训练方法即使在 10k 个节点/控制点下仍能保持良好的条件性，这与论文中的方法不同。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1397321949054631946)** (3 messages): 

> `Sparse MoE Models, SAEs, Interpretability, PEER follow up` 


- **稀疏 MoE 模型模拟 SAEs**：一名成员指出，极稀疏的 [MoE 模型](https://arxiv.org/pdf/2407.04153) 与 **SAEs** 非常相似，因为专家数量众多导致 FFN 层非常宽。
   - 他们提出，这种相似性可能使这些模型比稠密网络更容易解释。
- **PEER 获得后续研究**：一名成员分享了一篇关于 [PEER 的后续论文](https://arxiv.org/abs/2412.04139)，该论文与测试稀疏 MoE 模型及其可解释性有关。
   - 他们认为这篇论文为稀疏模型的可解释性提供了宝贵的见解。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1396929725259190293)** (19 messages🔥): 

> `lm-evaluation-harness, byte latent transformer, facebook/blt-entropy, facebook/blt-1b` 


- **追踪现有基准测试的性能**：成员们正致力于在 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/issues/3171) 中添加基准测试，以追踪模型性能，特别是扩展现有基准以追踪更多指标。
   - 主要问题在于默认输出**分组的宏观（macro）和微观（micro）分数**还是仅输出其中之一，目前的共识是两者都输出。
- **实现 Byte Latent Transformer (BLT)**：一名成员询问了在 harness 中实现 **Byte Latent Transformer (BLT)** 进行测试的可能性。
   - 对方澄清说，实现模型并非 harness 的主要功能，但支持为模型实现后端（backends）；然而，由于 **BLT** 是一个独立的代码库，除非它被添加到 HF 或 VLLM，否则不太可能得到支持。
- **Byte Latent Transformer (BLT) 集成的复杂性**：成员们讨论了将 **BLT** 集成到 *transformers* 库中的问题，暗示他们正在努力使其在那里获得支持。
   - 问题在于 **BLT** 作为一个 encoder-decoder-decoder 模型运行，需要一个包装模型（wrapper model）将 entropy 模型和 **BLT** 模型组合成一个适用于 HF API 的 causal LM。


  

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1397306449259925667)** (5 messages): 

> `Amazon infra, SageMaker support, EFA support` 


- **Amazon 员工寻求 GPT-NeoX 支持**：一名 Amazon 员工询问 **GPT-NeoX** 是否支持其专有通信系统，并对缺乏内部支持表示沮丧。
   - 一位成员回应表示怀疑，因为他们认为*我们以前从未与 Amazon 合作过，但我们总是很乐意尝试帮助想要使用我们库的人*。
- **关于 SageMaker 团队参与的推测**：鉴于用户对缺乏内部支持的恼火，一位成员想知道该咨询是否源自 **SageMaker 团队**。
   - 另一位成员觉得这*很有趣*。
- **GPT-NeoX 的 Elastic Fabric Adapter (EFA) 支持**：一位成员澄清说，使用 **Stability 计算资源 (Pythia, StableLM 等)** 训练的模型利用了 **Elastic Fabric Adapter (EFA)**，并强调了促进这一过程的 **NCCL EFA 插件** ([aws-ofi-nccl](https://github.com/aws/aws-ofi-nccl))。
   - 他们表示 *EFA 支持来自比 gpt-neox 更底层的技术栈。其层级为 gpt-neox->torch.distributed->nccl->EFA*，只要他们正确设置了 **NCCL**，就没有问题。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1396945287184122057)** (4 messages): 

> `A2A Agents Hackathon, LlamaCloud nodes for n8n.io, LlamaParse Feature, Automate PDF parsing` 


- **LlamaIndex 赞助 A2A Agents 黑客松**：LlamaIndex 将于 7 月 26 日在旧金山赞助 **A2A Agents 黑客松**，其开发者关系副总裁将与来自 [Scale AI](https://t.co/R6J4igjhSH) 的专家一起担任演讲嘉宾和评委。
- **LlamaCloud 节点加入 n8n.io**：适用于 @n8n_io 的 **LlamaCloud 节点**已上线，将 LlamaCloud 的文档解析和提取 Agent，以及作为知识库的 LlamaCloud 索引引入 n8n 自动化工作流，可通过[此链接](https://t.co/UVqwYFkJFR)将 LlamaCloud 节点连接到现有工作流。
- **LlamaParse 检测页眉和页脚**：新的 **LlamaParse** 功能现在在 Balanced 或 Premium 模式下包含**页眉和页脚检测**，自动检测页面页眉和页脚，以便用户可以通过[此链接](https://t.co/Px1UjrINJC)选择性地隐藏它们或添加前缀和后缀。
- **使用 LLM 自动化 PDF 解析**：利用 **LLM 自动化 PDF 解析和提取**，通过智能文档理解超越 OCR 限制，通过[此链接](https://t.co/pOn7Tk1CBB)转换 PDF。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1396945337297403964)** (51 messages🔥): 

> `Inference Nerf, LlamaParse Authentication, LlamaIndex AWS Bedrock Models, Error Handling in LlamaIndex TS` 


- **周末后推理速度骤降**：一位用户报告周末后**推理速度显著削弱 (nerf)**，提取性能严重下降，正在寻求帮助。
   - 支持团队成员请求提供 **Job ID、文件和 Schema** 以复现问题，并确认已移除**两个 Agent 的限制**。
- **LlamaParse API 身份验证的诡异问题**：一位用户在使用 n8n 和 curl 调用 LlamaParse 直接 API 时遇到**身份验证问题**，收到“Invalid authentication token”错误。
   - 用户确认在 Postman 中使用精确语法 `Authorization: Bearer $LLAMA_CLOUD_API_KEY` 解决了问题。*“我不知道这有什么区别，但它确实起作用了”*。
- **LlamaIndex 中 AWS Bedrock 模型的混淆**：一位用户询问 `@llamaindex/community` 包中适用于 AWS Bedrock 的 **Sonnet 4 模型**可用性，并对为何无法工作感到困惑。
   - 成员澄清用户应改用 `@llamaindex/aws` 包，并指出 `@llamaindex/community` 可能已被弃用，[文档](https://ts.llamaindex.ai/docs/llamaindex/modules/models/llms/bedrock)应进行更新。
- **限流异常阻碍 TS 策略**：一位用户寻求在 LlamaIndex TS 中处理 **ThrottlingException 错误**（Token 过多）的建议，特别是针对 Bedrock 模型实现退避逻辑 (backoff logic)。
   - 用户在等待 `agent.runStream(query)` 时看到未处理的拒绝 (unhandled rejections)，且找不到处理错误的方法，被建议提交 Issue 进行讨论。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1396937492695093309)** (17 条消息🔥): 

> `Aider 作为 Agent 还是工具、Copilot 故障、Aider 忽略 .gitignore、Qwen3 模型对比 Aider、Aider 的 Python 版本要求` 


- ****Aider 的 Agent 能力受到质疑****：一位用户询问 **Aider** 是否能像 **VSCode** 的 **Cline** 或 **Copilot** 那样更像一个 Agent 运作，通过自动查找和编辑文件，并引用了一张附图 ([image.png](https://cdn.discordapp.com/attachments/1131200896827654149/1397152245769965639/image.png?ex=688157ad&is=6880062d&hm=61b80c9ced86a84cfa31ca4aa8556e33b989d3960a697af83417f79a476b4cb2&))。
   - 另一位用户澄清说，**Aider** 主要是一个赋予开发者更多独立性的工具，其角色是编辑器而非自主 Agent。他们建议将 **OpenCode** 或 **Gemini CLI** 作为在 Agent 行为方面更接近 **Cline** 的替代方案。
- ****排查 Aider 与 .gitignore 的问题****：一位用户遇到了 **Aider** 未能忽略 **.gitignore** 中指定文件的问题，并寻求关于如何配置 **Aider** 以正确排除这些文件的帮助。
- ****Qwen3 模型表现优于 Aider？****：有讨论围绕最近的权重开放模型（特别是 **Qwen3**）在基准测试中表现出色，但与 **Aider Polyglot** 相比，在实际成就上可能存在倒退。
- ****Aider 的 Python 版本要求****：一位用户提到，默认使用 **Python 3.8** 的 **Ubuntu 20.04** 已被弃用，可能不适合 **UV** 和 **Aider** 等现代软件。
   - 建议使用 **Python 3.11 - 3.12**，而 **3.13** 目前还太新。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1396943934680465599)** (26 条消息🔥): 

> `Aider Polyglot 示例、模型循环、Aider 摘要调用、正确的编辑格式` 


- **Aider Polyglot 示例在每轮中使用单次生成**：成员们在质疑 [leaderboard.techfren.net](https://leaderboard.techfren.net/) 等网站上的 **Aider Polyglot 示例** 是否每轮仅涉及单次生成，并遵循 **Aider** 中的 **n_generations 约束**。
   - 问题在于某些模型是否可能在代码执行尝试之间循环运行多次 LLM 调用，从而在系统上实现“作弊”。
- **闭源供应商在推理时使用 tool calling**：闭源供应商可能会在推理时使用 **tool calling** 或其他类似策略。
   - 当 **max chat history（最大聊天历史）过长**时，Aider 会使用 [summarizer 模型](https://github.com/Aider-AI/aider/blob/f38200c511674e83a1b34a44e85beb77ee78f5c7/aider/coders/base_coder.py#L510)。
- **LLM 调用中包含格式重试**：据推测，向 LLM 发起的 **linter/格式化重试** 已包含在 LLM 调用次数中。
   - Polyglot 基准测试中的“正确编辑格式”是指**编辑的搜索部分是否正确**并在源码中被找到。
- **Tab 补全行为变更**：一位成员询问是否有办法更改添加文件时 **tab 补全** 的行为。
   - 他们希望**停止补全到目录**，因为他们觉得这样很难导航文件，并觉得可能错过了一些关于如何使用它的好技巧。
- **Aider 请求之前的文件内容**：一位成员想知道为什么 **Aider 要求模型输出需要打补丁的先前文件内容**。
   - 答案是：如果模型只需要输出**在哪里**打补丁，就能节省成本，类似于使用 **ed** 行编辑器。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1396994501461016768)** (7 条消息): 

> `NotebookLM 使用案例、Obsidian 插件、阅读书籍、聊天历史检索` 


- **用户探索 NotebookLM 使用案例**：成员们讨论了 **NotebookLM** 的使用案例，包括管理个人生活和 **TTRPG** 游戏的 **Obsidian** 库。
   - 其他成员建议使用 **NotebookLM** 来听书、从中创建思维导图，以及直接对上传的 **PDF** 文档进行提问。
- **开始寻找 Obsidian 插件**：一位成员询问是否存在 **Obsidian** 的插件。
   - 另一位成员澄清说，虽然没有直接的插件，但 **NotebookLM** 可以读取 **.md 文件**，允许用户从 **Obsidian** 中复制并粘贴内容。
- **NotebookLM 聊天历史受到关注**：一位用户询问是否有办法检索旧的聊天历史，以及 **NotebookLM** 是否保存之前的聊天记录和摘要。
   - 未给出明确答复。


  

---

### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1396936451278504068)** (30 messages🔥): 

> `Google Ultra AI Subscription Benefits, NotebookLM Service Unavailable Error, PDF Image Reading Capabilities, NotebookLM for American Yawp Notes, Gemini Pro Model Integration` 


- **Ultra AI 承诺更好的模型，“今年晚些时候”推出**：用户注意到 Google Ultra AI 订阅承诺在 notebook 中提供*更好的模型*，但目前没有立即更改模型的方法，且该功能被标记为*“今年晚些时候”*。
- **“服务不可用”错误困扰用户**：几位用户报告在使用 NotebookLM 时遇到 *“服务不可用 - 您尝试访问的服务不适用于您的帐户”* 错误，特别是在 Macbook 上，即使拥有活跃订阅且移动端 App 正常运行。
- **NotebookLM 可以读取 PDF 图像**：一位用户询问 **NotebookLM** 是否可以同时读取 PDF 的文本和图像，另一位用户确认它可以，但除非在 Prompt 中另有说明，否则会优先处理文本。
- **PDF 上传问题困扰用户**：尽管尝试了各种文件和文件类型，多位用户在向 NLM 上传 PDF 时仍遇到持续错误；一位用户建议*删除 Cookies* 作为潜在的修复方法。
- **播客长度选项神秘消失**：一位用户报告 Audio Overview 部分中用于调整播客长度的 **“Shorter”、“Default”和“Longer”选项** 消失了，至少有另一位用户确认遇到了同样的问题。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1396948051255496775)** (28 messages🔥): 

> `Qwen3-235B-A22B, Kimi-K2 Tech Report, RLHF Reward Model, Kimi K2 Style` 


- **Qwen3-235B-A22B Instruct 发布**：**Qwen3-235B-A22B-Instruct 模型**已发布，但尚未推出 **30B** 等较小尺寸，这使得在有好的 API 供应商托管之前其可访问性较低，可在 [HuggingFace](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) 获取。
- **Kimi-K2 技术报告引起关注**：**Kimi-K2 技术报告**已分享，人们担心其中可能包含针对年轻用户的成瘾性元素，可在 [GitHub](https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf) 查看。
- **RLHF Reward 模型受到质疑**：关于 **RLHF 中的 Reward 模型** 出现了疑问，特别是它具有单一输出维度，还是具有输出维度等于窗口的线性层，以便进行折扣奖励计算。
- **Kimi K2 奇特的手游开发风格**：一位成员注意到 **Kimi K2** 在为一款移动太空模拟游戏生成创意时表现出一种奇特的风格，具体表现为使用诸如 *“juice”* 指代 VFX 以及 *“meta/shop”* 等术语。


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1397043364414619711)** (5 messages): 

> `Computer vision applications, Generative models, Flow matching, VLM fine-tuning, Success Principles` 


- **计算机视觉爱好者集结！**：两位成员表达了对**计算机视觉应用**的兴奋，并期待未来的互动。
   - 一位成员询问 *“你现在在想什么？”*，另一位回答是 **带有 flow matching 的生成模型和 VLM 微调**。
- **小小的胜利推动成功**：一位成员分享了一条励志信息：*“成功是一系列小小的胜利”*。


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1397015484045594635)** (8 messages🔥): 

> `JSON Schema Regression, Embed v4 Rate Limit` 


- **JSON Schema 请求失败**：一位成员报告了 **JSON schema** 请求的回归问题，指出之前可以正常工作的输入现在失败，并显示 *“invalid 'json_schema' provided: missing required field 'type'”* 错误消息。
   - 之前成功的输出是一个简单的 JSON 对象，指示香蕉是否为水果：`{"is_fruit": true}`。
- **Embed-v4 速率限制仅面向企业客户**：一位用户询问关于增加 **Embed v4** 速率限制的问题。
   - Cohere 团队的一位成员回复称，提升的速率限制目前仅保留给有最低承诺支出的企业客户。如果你符合条件，请通过 [support@cohere.com](mailto:support@cohere.com) 联系他们。


  

---

### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1397042897315954792)** (6 messages): 

> `VLMs, Generative models, 3D reconstruction, AI platforms` 


- **MLE 硕士生深耕 VLMs 与 Gemini**：一名学生目前正致力于 **VLMs**、**Generative models** (Flow matching) 和 **3D reconstruction**，并倾向于使用 **pytorch** 和 **gemini**。
   - 他们希望从这个社区获得知识、结交朋友并开展研究项目合作。
- **LLM 工程师寻求互动与新研究**：一位成员专注于 **reasoning**、**LLMs** 和 **RL**，利用 **vllm**、**hf**、**pt** 和 **wandb** 等工具。
   - 他们正在寻求与其他人的互动，寻找研究机会，并发现该领域的新进展。
- **AI 架构师旨在为企业集成 AI**：一位 **AI & Enterprise architect** 正在协助企业集成 **AI**，以提高效率和盈利能力。
   - 他们对使用 **natural language** 解决商业问题的 **AI platforms** 感兴趣，并渴望学习、联系和分享见解。
- **AI 工程师构建创新的 LLM 产品**：Elevancesystems 的 **AI Engineer/Head of AI** 正在开发创新的 **AI/LLM products**。
   - 他们期待分享和磋商面向真实商业世界的新技术和解决方案。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1396948880603746345)** (16 messages🔥): 

> `Mojo vs C++ Vectorization, Modular Careers, Modular Open Source Contribution, Mojo GPU Programming, Mojo Package Manager` 


- **Mojo 的向量化能力引发辩论**：一位用户询问了 **Mojo** 在 **CPU vectorized tasks** 方面与 **C/C++** 的对比情况，引发了关于每种语言优缺点的讨论。
   - 一位成员认为，虽然两者都可以通过内联汇编实现类似的性能，但 **Mojo** 简化了流程，特别是对于复杂的代码，超越了 **ISO C++** 的能力。
- **求职者在 Modular 寻找机会**：针对有关开发机会的询问，分享了 **Modular Careers** 页面的链接。
   - 链接为 [https://www.modular.com/company/careers](https://www.modular.com/company/careers)。
- **ML 编译器专家关注 Modular 贡献**：一位具有 **ARM ML compilers** 经验的用户表达了对贡献 **Modular open source** 项目（特别是 **MAX AI kernels**）的兴趣。
   - 他们在处理入门级问题（beginner-friendly issues）之前寻求有关复习资源的建议。
- **通过 Puzzles 强调 Mojo GPU 编程**：建议通过 [Modular Puzzles](https://puzzles.modular.com/introduction.html) 探索 **Mojo** 的 **GPU programming** 能力，以了解该语言的方法。
   - 该建议旨在为潜在贡献者提供 **Mojo GPU 编程** 的实用介绍。
- **分享 Mojo 包管理器代码**：一位成员在 [GitHub](https://github.com/luigithejokesterplus/birdinstall) 上分享了他们的 **Mojo package manager** 项目，邀请其他人审查代码。
   - 另一位成员对该贡献表示欢迎，并指出 [Modular community repo](https://github.com/modular/modular-community) 是一个潜在的中心。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1397163592943407164)** (1 messages): 

> `Mojo async design, async/await ecosystem split` 


- **Mojo 的 Async 设计草案发布**：一位成员发布了 **Mojo** 中 `async` 的[更新设计草图](https://github.com/modular/modular/pull/3986#issuecomment-3102049612)。
   - 既定目标是避免现有的每种带有 **async/await** 的语言都遭受的“生态系统分裂”（库 + 代码重复）问题。
- **避免 Async/Await 生态系统分裂**：**Mojo** 中新 `async` 设计的主要动机是防止在其他语言中看到的库和代码重复问题。
   - 该设计旨在创建一个统一的生态系统，其中 **async** 代码可以与同步代码无缝交互，从而减少对独立库和代码库的需求。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1397303241183727749)** (1 messages): 

> `Max vs llama.cpp, CPU serving performance` 


- **Max vs llama.cpp CPU 推理服务对决**：一位成员询问了比较 **Max** 和 **llama.cpp** 在 CPU 推理服务方面的 **benchmarks**。
   - 重点在于确定在**直接在 CPU 上提供服务**时哪一个表现更好。
- **寻求 CPU 推理服务性能基准测试**：用户专门请求 **performance benchmarks** 来评估 **Max** 和 **llama.cpp**。
   - 强调的使用场景是 **CPU serving**，这意味着需要高效的 CPU 利用率。


  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1397200878167134330)** (6 messages): 

> `RL Torchtune Release, RL Libraries` 


- **RL Torchtune 计划在 PyTorch Conf 前发布**：新款 **Torchtune** 将侧重于 **RL**，旨在 PyTorch 大会前发布，重点关注后训练（post-training）以及超过 **600B 参数**的扩展。
   - 在承认权衡的同时，目标是促进快速、小规模的实验，并能轻松扩展到生产级别，优先解决扩展性挑战；此外有用户指出库 [RL2](https://github.com/ChenmienTan/RL2) 可能会引起关注。
- **小规模 GPU 支持仍是重点**：虽然主要关注大规模 **RL**，但并未完全放弃对小规模的支持。
   - 目标是使用户能够在较小规模上进行快速实验，并能轻松迁移和扩展到生产环境。


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1397242215759089796)** (2 messages): 

> `Fine-tuning 70B models, recipe_state.pt size, torch.save behavior, Distributed checkpointing` 


- **本地分片检查点（Local Shard Checkpointing）疑虑浮现**：一名成员注意到在微调 **70B 模型**时，`recipe_state.pt` 文件仅约 **30GB**，并担心在非分布式检查点场景下 `torch.save` 可能仅存储了本地分片。
   - 他们观察到使用 `torch.load` 加载的张量似乎位于 `cuda:3` 上，带有 **DTensors** 和分片放置（sharded placements），这表明可能存在覆盖且仅存储本地分片的情况。
- **成员询问 `torch.save` 是否总是本地的？**：该成员质疑 `torch.save` 是否始终是本地的，以及非分布式检查点是否存在潜在问题，导致覆盖并仅存储本地分片。
   - 他们建议从 `torch.save` 迁移到新的分布式检查点器（distributed checkpointer），并请求对其推理进行正确性检查。


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1396951680096014490)** (6 messages): 

> `MCP vs System Tool Call, MCP Tool Purpose, Image Context, New Member Introduction` 


- **MCP 问题引发讨论**：一名成员质疑为什么某些功能被实现为 **MCP (Model Control Program)** 而不是更简单的 **system tool call**。
   - 另一名成员解释说，*任何 MCP 工具的目的都是为了赋予 LLM 其原生不具备的能力*。
- **发现图像上下文问题**：一名成员就**图像上下文（context of the images）**提出了一个很好的观点。
   - 另一名成员表示之前甚至没有考虑到这一点，并感谢其建议。
- **新成员加入频道**：一名新成员加入了频道，向大家打招呼并感谢邀请。
   - 这一事件标志着新参与者加入了正在进行的讨论。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1396952294456823872)** (2 messages): 

> `WordPress integration for Claude, MCP server for WordPress` 


- **通过 Claude 的对话式 CMS 控制实现 WordPress 奇迹**：一名成员宣布发布了 **Claude Desktop 的 WordPress 集成**，允许直接通过 Claude 对话控制 WordPress，消除了复制粘贴的工作流；并分享了 [仓库](https://github.com/docdyhr/mcp-wordpress)。
   - 该成员补充说，使用它来管理博客改变了他们对内容创作的思考方式，因为 **Claude** 可以查看现有文章、理解站点结构并进行更新。
- **MCP 服务端支持任何 WordPress 的 MCP 客户端**：一名成员询问该 WordPress 集成是否仅支持 **Claude Desktop**，并暗示鉴于其本地 **MCP server** 的设置，它可能适用于任何 **MCP client**。
   - 然而，目前还没有回复来确认或否认这一点。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1397335529393229965)** (1 messages): 

> `DSPy, Python user group, Local Tech Meetups` 


- **DSPy 传播到本地 Python 爱好者群体**：一名成员宣布他们将向本地 Python 用户组介绍 **DSPy**，旨在展示其能力。
   - 他们还分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=1WKA8Lw5naI)，为该小组提供关于 **DSPy** 直观且易懂的介绍。
- **技术社区参与度提升**：该成员的倡议凸显了本地技术聚会在传播知识和促进社区参与方面的重要性。
   - 通过在 Python 用户组展示 **DSPy**，他们正在为该框架在技术社区内更广泛的采用和理解做出贡献。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1397018163870437408)** (5 messages): 

> `Teleprompters, DSPy Modules, Professional Services 组织, AWS Services` 


- **Professional Services 工程师定制化 AWS 服务**：一位 Professional Services 组织的成员表示，他们为大型企业客户工程化定制解决方案，如果多个客户有相同的需求，这些功能将被添加到 **AWS services** 本身。
- **Teleprompters 接受 DSPy Modules**：一位成员询问基础 **Teleprompter** 是否接受任何 Module 作为 student，以及这在所有 teleprompters 中是否一致。
   - 另一位成员澄清说，允许任何 `dspy.Module` 子类，*不允许其他任何内容*。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1397279327460134942)** (3 messages): 

> `Whisper PR 速度, 用于 tinyboxes 的集装箱, Rust 用于 Tinygrad` 


- **Whisper PR 运行缓慢**：一位成员对他们的 **Whisper PR** 运行缓慢表示沮丧，因为他们的目标是将其控制在 500 行以内，并接近 **OpenAI 的速度**。
   - 他们希望它保持 *简洁优美*。
- **集装箱存放 TinyBoxes**：一位成员提出了一个*疯狂的想法*，即使用**集装箱**来存放 **tinyboxes**，并暗示了模块化、散热优势和移动性。
   - 他们思考了成本和安全性方面的问题，并开玩笑地建议命名为 *tinycontainer*。
- **Rust 与 Tinygrad 的竞争**：一位成员指出 **Rust** 与 **Tinygrad** 是完全不同的市场，但与 TG 进行基准测试会很有趣。
   - 他们推测，如果 **tinycorp** 进入企业级 LLM 硬件设备市场，所有的框架和自定义解决方案都将在那里展开竞争。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1396930213660856321)** (1 messages): 

> `Windows 上的 CUDA, CPU 后端设置, LLVM 后端设置` 


- **简化 Windows 上 CUDA 的启用**：一位成员分享了一种在 Windows 上启用 **CUDA** 的方法，包括一个用于应用更改的 [补丁文件](https://cdn.discordapp.com/attachments/1070745817025106080/1396930213312725022/cuda_windows.patch?ex=688131a4&is=687fe024&hm=e6965a699c395de25b72762e696fce5fb5545f656120ee70353c584fe468bbb9&)。
   - 该过程涉及针对不同后端的特定环境配置。
- **设置 CPU 后端**：要启用 **CPU 后端**，用户只需确保将 *clang* 添加到 **PATH** 环境变量中。
   - 这使得 *clang* 在系统范围内可访问，这对于编译和运行针对 CPU 的代码至关重要。
- **LLVM 后端配置**：对于 **LLVM 后端**，必须设置 **LLVM_PATH** 环境变量以指向 *LLVM-C.dll*，例如 `set LLVM_PATH=c:\llvm-16.0.2-windows-amd64-msvc17-msvcrt\bin\LLVM-C.dll`。
   - 此配置将系统导向正确的 **LLVM** 动态库，确保基于 **LLVM** 的编译功能正常运行。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1397020461308710992)** (3 messages): 

> `证书问题, 写作提交表单问题` 


- **证书发放延迟的讨论**：一位成员询问缺失证书的问题，并提供了[两个电子邮件地址](mailto:terrence_rideau@yahoo.com,terrence.rideau@google.com)进行验证。
   - 工作人员表示，这两个邮箱下都没有收到*证书声明表单*，结果发现是**写作提交表单**未正确提交。
- **写作提交失误影响学生**：一名学生报告说，尽管参加了课程、通过了测试并撰写了文章，但错过了**证书声明表单**的提交。
   - 该成员表达了遗憾，但对课程内容表示赞赏，并感谢了演讲者和团队。