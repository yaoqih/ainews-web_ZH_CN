---
companies:
- langchain
- meta-ai-fair
- hugging-face
- openrouter
- google-ai
- microsoft
- openai
- anthropic
date: '2025-10-23T05:44:39.731046Z'
description: '**LangSmith** 推出了 **Insights Agent**，具备针对智能体运维（Agent Ops）和可观测性的多轮评估功能，提升了故障检测和用户意图聚类的能力。**Meta
  PyTorch** 和 **Hugging Face** 联合推出了 **OpenEnv**，这是一个 Gymnasium 风格的 API 和中心（Hub），用于支持分布式训练的可复现智能体环境。


  相关讨论强调了智能体编码中提供商保真度（provider fidelity）的重要性，其中 **OpenRouter** 的 exacto 过滤器提升了稳定性。在构建者体验（Builder
  UX）更新方面，**Google AI Studio** 为 Gemini 代码更改推出了注释模式（Annotation mode），**微软**增强了 Edge
  浏览器中的 Copilot 模式，而 **OpenAI** 为 ChatGPT 商务版推出了共享项目（Shared Projects）和公司知识库（Company
  Knowledge）功能。**Claude** 则增加了项目级的记忆功能（Memory）。


  在强化学习领域，**Meta** 的 **ScaleRL** 提出了一种预测大语言模型（LLM）强化学习缩放结果的方法，具有更高的效率和稳定性。'
id: MjAyNS0x
models:
- gemini-1.5-pro
- claude-3
- chatgpt
people:
- hwchase17
- ankush_gola11
- whinthorn
- koylanai
- _lewtun
- bhutanisanyam1
- thom_wolf
- danielhanchen
- cline
- canvrno
- pashmerepat
- mustafasuleyman
- yusuf_i_mehdi
- jordirib1
- fidjissimo
- bradlightcap
- mikeyk
- alexalbert__
title: 今天没发生什么特别的事。
topics:
- agent-ops
- observability
- multi-turn-evaluation
- reinforcement-learning
- distributed-training
- api
- model-stability
- user-intent-clustering
- software-development
- project-management
- code-generation
---

**平静的一天**

> 2025/10/22-2025/10/23 AI 新闻。我们为您检查了 12 个 subreddits、544 个 Twitter 账号和 23 个 Discord（包含 198 个频道和 8784 条消息）。预计节省阅读时间（按 200wpm 计算）：592 分钟。我们的新网站现已上线，支持全元数据搜索，并以精美的 vibe coded 方式呈现所有往期内容。访问 https://news.smol.ai/ 查看完整的新闻分类，并在 @smol_ai 上向我们提供反馈！

平静的一天。

---

# AI Twitter 综述

**Agent 运维、可观测性与真实世界环境**

- **LangSmith 发布 “Insights Agent” + 多轮评估 (multi-turn evals)**：LangChain 推出了一款产品内 Agent，可扫描追踪记录 (traces) 以自动聚类使用模式和失败模式，并提供多轮评估功能来评估整个对话过程中的目标完成情况。团队报告称，无需人工分拣，即可近乎实时地洞察静默失败类别和用户意图集群。查看发布推文及详情：[@LangChainAI](https://twitter.com/LangChainAI/status/1981390300502487370), [@hwchase17](https://twitter.com/hwchase17/status/1981390508841980332)，以及来自 [@Hacubu](https://twitter.com/Hacubu/status/1981396190077043162), [@ankush_gola11](https://twitter.com/ankush_gola11/status/1981408009097265344), [@WHinthorn](https://twitter.com/WHinthorn/status/1981403256598192451) 的工程笔记，以及 [@koylanai](https://twitter.com/koylanai/status/1981444604869087624) 的上手分析。
- **OpenEnv：Agent/RL 环境的共享规范与枢纽**：Meta PyTorch 和 Hugging Face 推出了 OpenEnv，这是一个 Gymnasium 风格的 API (reset/step/state)，专为通过简单 HTTP 进行容器/服务器执行而构建，并包含一个用于可重复“Agent 环境”（工具、凭据、沙箱）的 Hub。早期集成涵盖了 TRL, Unsloth, Atari 以及社区示例（如扑克），旨在标准化环境打包并扩展分布式训练。参见 [@_lewtun](https://twitter.com/_lewtun/status/1981380372748521929), [@bhutanisanyam1](https://twitter.com/bhutanisanyam1/status/1981377720157351938), [@Thom_Wolf](https://twitter.com/Thom_Wolf/status/1981396028117901401), 和 [@danielhanchen](https://twitter.com/danielhanchen/status/1981428184215363956)。
- **实战中的 Agent 编码：供应商忠实度至关重要**：Cline 强调了相同的开放权重模型在不同推理端点（量化、tool-call 格式化、“thinking” 标签）上的表现如何截然不同，这通常导致用户归咎于模型而非基础设施。他们的修复方案结合了激进的系统提示词缩减和供应商过滤（例如 OpenRouter 的 :exacto）以恢复稳定性。他们还将发布包含真实世界、可中断任务的 ClineBench。参见 [@cline](https://twitter.com/cline/status/1981370535176286355)，[@canvrno](https://twitter.com/canvrno/status/1981403534471119330) 的分析，以及 [@pashmerepat](https://twitter.com/pashmerepat/status/1981431374386233840)。
- **开发者体验 (UX) 更新简述**：Google AI Studio 的新标注模式 (Annotation mode) 允许你“标记”实时应用 UI，并让 Gemini 应用代码更改（[公告](https://twitter.com/GoogleAIStudio/status/1981375306423554490), [演示](https://twitter.com/patloeber/status/1981375563685384430)）。Microsoft 在 Edge 中推出了 Copilot 模式（Journeys, Actions）、Mico 语音 UI，并升级了 Copilot 内部的搜索落地 (search grounding)（[@mustafasuleyman](https://twitter.com/mustafasuleyman/status/1981390345578697199), [@yusuf_i_mehdi](https://twitter.com/yusuf_i_mehdi/status/1981426387958583717), [@JordiRib1](https://twitter.com/JordiRib1/status/1981399255576174657)）。OpenAI 为 ChatGPT Business/Enterprise/Edu 增加了共享项目 (Shared Projects) 和“公司知识库”（Slack, Drive, GitHub 等）（[@OpenAI](https://twitter.com/OpenAI/status/1981432799212249119), [@fidjissimo](https://twitter.com/fidjissimo/status/1981437695915413947), [@bradlightcap](https://twitter.com/bradlightcap/status/1981454865454027007)），而 Claude 发布了项目范围的记忆功能 (Memory)（[@mikeyk](https://twitter.com/mikeyk/status/1981415275695394852), [@alexalbert__](https://twitter.com/alexalbert__/status/1981421146886328778)）。同样值得关注的还有：Firecrawl 在 LangChain、n8n 和 MCP 上的集成指南（[@firecrawl_dev](https://twitter.com/firecrawl_dev/status/1981390679462072766)），以及 Vercel 用于 TypeScript 中持久异步任务的 “useworkflow”（[@cramforce](https://twitter.com/cramforce/status/1981399119559348290), [@rauchg](https://twitter.com/rauchg/status/1981426366982824387)）。

**LLM 的强化学习 (RL)：缩放法则、稳定性和离策 (off-policy)**

- **ScaleRL (Meta)：迈向可预测的 RL 扩展**：新工作提出了一种方案和方法论，旨在通过小规模运行预测 LLM RL 的结果。设计选择包括 PipelineRL-8（异步）、CISPO 损失、FP32 计算、prompt 平均损失、batch 级归一化、零方差过滤以及 No-Positive-Resampling。声称在高达 100k GPU-hours 的情况下能实现准确外推，且效率优于 GRPO/DAPO/Magistral。摘要见：[@TheTuringPost](https://twitter.com/TheTuringPost/status/1981487666714800356) 及其中的论文链接。
- **通过训练-推理对齐避免 RL 崩溃**：一项深入的剖析显示，微小的框架/精度差异（KV cache 精度、FP32 中的 softmax/norm、RoPE deltas、attention 后端差异、MoE 路由稳定性）会在层与 token 之间累积——尤其是在 MoE 和长序列生成（rollouts）中——导致崩溃。解决方案：逐层激活日志记录以及 prefill/decode 之间的对齐、一致的数值计算以及高精度路由。通过 [@ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/1981337266523164694) 阅读技术清单。
- **基于记忆的持续学习与 off-policy RL**：Memento 将 Agent 的改进重新定义为在记忆增强型 MDP（基于案例的推理 + MCP 工具上的执行器）上的基于记忆的 online RL，无需权重更新（[推文串](https://twitter.com/_avichawla/status/1981246733322768780) + [仓库](https://twitter.com/_avichawla/status/1981246746497077492)）。BAPO 针对部分 rollout 和经验复用场景下的 LLM off-policy RL（[@Be1ong1](https://twitter.com/Be1ong1/status/1981297924564046007)）。OpenEnv 的标准化（见上文）加上 Unsloth/TRT/Llama 生态系统，正趋向于为大规模训练提供共享且可复现的环境（[@danielhanchen](https://twitter.com/danielhanchen/status/1981428184215363956)）。

**生成式媒体、OCR/VLM 浪潮与机器人**

- **开放创意/视频引擎**：LTX 发布了 LTX-2，这是一个开放的 AI 创意引擎，支持音视频同步、原生 4K、高达 50 fps 和 10 秒序列，采用 API 优先设计，在消费级 GPU 上运行高效；权重将于今年晚些时候发布（[@ltx_model](https://twitter.com/ltx_model/status/1981346235194683497), [@LTXStudio](https://twitter.com/LTXStudio/status/1981371951894667279)）。Argil 宣布了 Atom，强调可控性和时间一致性，无时长限制，并提供用于外观选择的“风格 Tinder”（[发布](https://twitter.com/BrivaelLp/status/1981343140196778270), [试用](https://twitter.com/BrivaelLp/status/1981344149862314183)）。
- **机器人基础模型与 OCR/VLM**：NVIDIA 的 Gr00t N1.5（通过 LeRobot）是一个跨具身动作模型，具有视觉/语言/本体感受输入和 flow-matching action transformer，在真实/合成/互联网规模的数据上进行训练；在 Libero 和真实硬件上进行了评估（[@LeRobotHF](https://twitter.com/LeRobotHF/status/1981334159801929947)）。OCR/VLM 正在激增：LightOnOCR-1B（端到端 VLM）专注于速度/吞吐量（[@staghado](https://twitter.com/staghado/status/1981379888301867299)），OlmOCR-2 依靠 RLVR + 二进制单元测试进行快速迭代（[@kylelostat](https://twitter.com/kylelostat/status/1981380820658180310)），模型对比更新迅速（[摘要](https://twitter.com/mervenoyann/status/1981396054634615280)；根据 [@MaziyarPanahi](https://twitter.com/MaziyarPanahi/status/1981421331053760775)，VLM/OCR 发布在 HF 上正趋于热门）。同样值得注意的还有：Runway 的“广告应用”系列，旨在将常见的视频/图像工作流产品化，无需复杂的 prompting（[公告](https://twitter.com/runwayml/status/1981380360249159783)）。

**基础设施与模型平台**

- **Anthropic x Google TPU 巨额交易**：Anthropic 计划在 2026 年扩展到“约一百万个” TPU，且容量“远超” 1 GW——涉及数百亿美元的算力投入——大幅扩展了训练和推理的余量 ([@AnthropicAI](https://twitter.com/AnthropicAI/status/1981460118354219180), [后续](https://twitter.com/AnthropicAI/status/1981460119742533848))。
- **推理栈 (Serving stacks)**：vLLM 现在支持 NVIDIA 的 Nemotron Nano 2（9B 混合 Transformer–Mamba 推理模型，开源权重，训练量超过 9T tokens），具有可调的“思考预算” (thinking budget) 以实现可预测的成本和延迟；vLLM 声称其“思考” token 吞吐量比类似的开源密集模型快 6 倍，提升了 Agent 的搜索和反思能力 ([@vllm_project](https://twitter.com/vllm_project/status/1981553870599049286))。Cerebras 发布了经过 REAP 剪枝的 GLM‑4.6 MoE 检查点，压缩率分别为 25/30/40%（FP8, A32B），旨在保持生成质量的同时提高效率 ([@vithursant19](https://twitter.com/vithursant19/status/1981476324045967785))。Ollama 发布了在 NVIDIA Spark 固件及新版本上的性能测试 ([@ollama](https://twitter.com/ollama/status/1981486870963114121))。此外：Qdrant 推出了 Vector Search “Academy” ([@qdrant_engine](https://twitter.com/qdrant_engine/status/1981319267749679599))，Modular 发布了 Mojo GPU “puzzles”，用于 CUDA/Metal 的实操学习 ([@Modular](https://twitter.com/Modular/status/1981455872137318556))。

**路由与服务保真度**

- **多 LLM 系统的 Lookahead 路由**：提出的 “Lookahead” 路由可以预测潜在响应的隐层表示 (latent representations)，从而廉价地“偷窥”每个模型的回答，在无需完全解码的情况下实现响应感知路由。据报告，在 7 个基准测试中，其表现比 SOTA 路由平均高出 7.7%，且具有极高的数据效率（仅需 16% 的数据即可达到满血性能），并在 causal/masked LMs 之间具有良好的泛化性 ([@omarsar0](https://twitter.com/omarsar0/status/1981360482813710384))。
- **提供商差异是一等风险**：Cline 的分析显示，提供商侧的无声变化（量化、tool-call 格式）可能导致结果从“成功”变为“失败”，从而削弱对开源模型的信任。他们的缓解措施：将 system prompt 削减 57%（从 56,499 字符减至 24,111 字符），严格的提供商过滤（例如 OpenRouter 的 :exacto），以及工作流强制执行。建议：要求透明地报告量化和实现差异，并将提供商测试纳入模型 evals 中 ([@cline](https://twitter.com/cline/status/1981420111815987494), [@canvrno](https://twitter.com/canvrno/status/1981403534471119330))。

**研究亮点**

- **推理过程中的指令遵循**：Together 的 ReasonIF 基准测试发现，大型推理模型经常在思维链 (chain-of-thought) 过程中违反用户约束（多语言格式、长度控制），强调了在生成过程中进行指令保真度检查的必要性 ([@togethercompute](https://twitter.com/togethercompute/status/1981441935303975059))。
- **预训练中的“覆盖轮廓”优于交叉熵**：一篇新的预印本论文认为，成功源于覆盖率指标 (coverage metrics)——即模型内化了哪些分布——而不仅仅是 loss ([@canondetortugas](https://twitter.com/canondetortugas/status/1981481591177105740))。
- **LMs 的可逆性/单射性**：一篇论文声称在大量实证测试中证明了模型映射（输入 → 表示）的单射性/可逆性 (injectivity/invertibility)，暗示了无损表示的特性，对可解释性具有重要意义 ([@HuggingPapers](https://twitter.com/HuggingPapers/status/1981452722495787286))。
- **优化动力学**：关于基于 muP 的权重衰减缩放 (weight decay scaling) 的新工作（用于超参数迁移的独立缩放），以及关于早期 vs 后期效应的实证评论 ([论文](https://twitter.com/tonysilveti/status/1981406663086391588), [@giffmana](https://twitter.com/giffmana/status/1981483376604565969) 的讨论)。此外：通过 TunedLens 探索表示流 (representational flow) ([@neuranna](https://twitter.com/neuranna/status/1981357907170959799))，以及关于实践中线性注意力 (linear attention) 精度的笔记 ([@francoisfleuret](https://twitter.com/francoisfleuret/status/1981487811489317175))。

---

**热门推文（按互动量排序）**

- [Anthropic 计划在 2026 年部署约 100 万个 TPU 并实现超过 1 GW 的容量](https://twitter.com/AnthropicAI/status/1981460118354219180) (3.4k+)
- [OpenAI：ChatGPT 中的共享项目 (Shared Projects)](https://twitter.com/OpenAI/status/1981432799212249119) (3.1k+)
- [Yann LeCun：“你无法在制造涡轮喷气发动机之前证明其安全性；AI 也是如此。”](https://twitter.com/ylecun/status/1981360519442321451) (2.9k+)
- [LTX-2：开源 AI 创意引擎（4K/50fps，API-first）](https://twitter.com/ltx_model/status/1981346235194683497) (2.5k+)
- [Argil Atom：具有强一致性的可控视频](https://twitter.com/BrivaelLp/status/1981343140196778270) (2.3k+)
- [Microsoft：“Clippy 回来了！！”](https://twitter.com/satyanadella/status/1981466897557196837) (5.7k+)
- [欧盟“AI 工厂”与行业 GPU 规模——算力对比](https://twitter.com/levelsio/status/1981351393513615813) (4.2k+)

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. AI Agent 基础教程

- [**我花了几个月时间苦苦钻研 AI Agent。我从零开始编写了一个教程，这样你就不用再走弯路了。**](https://www.reddit.com/r/LocalLLaMA/comments/1oee1ie/i_spent_months_struggling_to_understand_ai_agents/) (热度: 345): **这篇 Reddit 帖子介绍了一个从零开始构建 AI Agent 的全面教程，重点在于基础理解，而不是依赖 LangChain 或 CrewAI 等框架。该教程可在 [GitHub](https://github.com/pguso/ai-agents-from-scratch) 上获取，包含 8 个使用纯 JavaScript 和 Qwen、Llama 等本地 LLM 的渐进式示例。它涵盖了系统提示词 (system prompts)、流式传输 (streaming)、Token 控制、函数调用 (function calling)、记忆系统和 ReAct 模式等核心概念，旨在为喜欢动手实践的开发者揭开 AI Agent 底层机制的神秘面纱。** 一位评论者赞赏了教程的清晰度，并建议将其发布在 r/LocalLLaMA 等相关论坛上。另一位分享了类似的学经验，强调了理解 AI Agent 中工具使用和函数调用的重要性，正如 Mistral 文档中所阐述的那样。
    - mobileJay77 讨论了一种 AI Agent 的调试方法，参考了 Agno Agi 和 Mistral 文档。他们强调了一种方法：不是直接向 LLM 提问，而是将查询格式化为包含函数名称和参数的 JSON。这种结构化方法允许解析结果、执行函数，然后让 LLM 将其转换为完整的句子，强调了处理 AI Agent 中工具使用的系统化方式。

## 技术性较低的 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. AI 与工作替代担忧

- [**公平的问题**](https://www.reddit.com/r/OpenAI/comments/1oe42et/fair_question/) (活跃度: 1322): **该图片是一个模因（meme），展示了参议员 Bernie Sanders 的一条推文，表达了对 AI 和机器人可能取代所有工作的担忧，这一观点与 Elon Musk 相关。Sanders 质疑了那些可能失去工作和收入的工人的处境，强调了面对技术进步时未来就业的一个重大问题。讨论反映了更广泛的社会对 AI 对劳动力市场影响的担忧，以及经济系统适应技术变革的必要性。** 评论者对 AI 和机器人完全取代工作表示怀疑，认为如果出现这种情况，将需要经济体系的根本性变革。关于 AI 是会创造新工作，还是会导致传统劳动力过时并需要新的社会结构的辩论正在进行中。
- [**Elon Musk 表示 AI 将取代所有工作并使工作成为可选项。你认为这是梦想还是灾难？**](https://www.reddit.com/r/ChatGPT/comments/1ody5w4/elon_musk_says_ai_will_replace_all_jobs_and_make/) (活跃度: 6231): **该图片是 Elon Musk 的一条推文，暗示 AI 和机器人将取代所有工作，使工作成为可选项。这一概念暗示了一个就业不再是生存必需的未来，类似于选择自己种菜而不是买菜。其理念是 AI 可以处理琐碎的任务，让人们专注于爱好或创造力。然而，这引发了对如果传统工作消失，人们会失去目标感、身份认同和动力的担忧。** 评论者表示怀疑，质疑 AI 是否会取代 CEO 等高层职位，以及如果人们不再通过工作赚钱，经济将如何运作。一些人认为 Musk 的言论不切实际或具有误导性。
- [**公平的问题**](https://www.reddit.com/r/ChatGPT/comments/1oe42v0/fair_question/) (活跃度: 790): **该图片是一个模因，展示了参议员 Bernie Sanders 的一条推文，强调了对 AI 和机器人技术对就业影响的担忧，呼应了 Elon Musk 的观点。Sanders 质疑了在一个 AI 可能取代所有工作的世界中工人的未来，提出了收入和社会结构的问题。这反映了关于 AI 对劳动力市场影响的持续辩论，以及应对这些挑战可能需要的社会变革。** 一条评论指出，资本主义社会在没有工作的情况下无法运作，暗示需要大规模的社会转型，用新的结构取代传统的经济结构。另一条评论愤世嫉俗地提到了一个人类被视为多余的乌托邦未来，而第三条评论建议在“后工作”时代将“工人”重新定义为“人类”。

### 2. Claude AI 记忆功能发布

- [**Claude 现已为 Pro 和 Max 方案用户提供记忆功能**](https://www.reddit.com/r/ClaudeAI/comments/1oe8td4/claude_now_has_memory_for_pro_and_max_plan_users/) (热度: 617): **Claude 为其 Pro 和 Max 方案用户引入了记忆功能，允许 AI 学习并保留用户的工作流模式，包括工具使用、关键协作人员和问题解决偏好。该功能使想法能够随着时间的推移在不同对话中不断积累，用户可以控制记忆内容，并拥有编辑、重置或开关记忆功能的权限。该功能目前已面向 Max 用户开放，并将在未来两周内推向 Pro 用户。更多详情请参阅 [Anthropic 的新闻页面](https://www.anthropic.com/news/memory)。** 一些用户对 AI 记忆的实用性表示怀疑，建议通过标准化测试来评估其对输出质量的影响。另一些用户则建议增加诸如针对单个对话的“忽略记忆”标志等功能，还有部分用户遇到了记忆条目不准确从而影响对话准确性的问题。
    - 一位用户对 AI “记忆”的实用性表示怀疑，认为它可能无法提高输出质量。他们建议在开启和关闭记忆功能的账户上进行标准化测试对比分析，以评估其对性能的影响。
    - 另一位用户提出了一个潜在的功能改进建议，即为单个对话设置“忽略记忆”标志。这将允许用户切换记忆的使用状态，甚至可能在对话中途切换，以便更好地管理过去交互对当前输出的影响时机。
    - 一位用户报告称，在遇到 Claude 将错误的记忆条目视为事实信息而非依赖当前对话上下文的问题后，他禁用了记忆功能。这表明记忆功能在保持信息准确性和相关性方面可能存在潜在的可靠性问题。
- [**Genie 的实验性发布即将到来**](https://www.reddit.com/r/singularity/comments/1oe6twb/genies_experimental_launch_is_imminent/) (热度: 664): **该图像似乎是名为“Genie”的新功能或工具的宣传或概念图，可能与交互式或创意过程有关，且可能涉及 AI。“让我们从勾勒你的世界开始”这段文字暗示了其侧重于用户驱动的内容创作，可能允许用户以文本格式描述环境和角色。这与评论中的推测一致，即该功能最初可能支持“text to world”能力，表明这可能是一个 AI 驱动的世界构建工具。提到的“Genie 的实验性发布”暗示这是一个即将推出或处于 Beta 阶段的功能，根据引用“Gemini 3 和 Genie”的评论推测，这可能与 Google 的 AI 计划有关。** 评论者对“Genie”的潜力表示兴奋，一些人希望未来能有图像上传功能来赋予他们的世界生命。此外，还有人幽默地期待 AI 生成的内容能超越现有的娱乐作品，正如关于 AI 版 GTA 6 的评论所示。

### 3. OpenAI 争议与法律问题

- [**OpenAI 正在变成完全的邪恶公司 (Evil Corp)**](https://www.reddit.com/r/OpenAI/comments/1oe48qe/openai_going_full_evil_corp/) (Activity: 3168): **该图片强调了 OpenAI 在一起涉及 Adam Raine 家属的诉讼中提出的争议性法律要求。Adam Raine 是一名在使用 ChatGPT 后自杀身亡的青少年。OpenAI 要求获取与 Raine 追悼会相关的文档，包括参与者名单和悼词，这被家属律师批评为“蓄意骚扰”。这一要求很可能是非正常死亡诉讼中证据开示（discovery）过程的一部分，OpenAI 可能试图验证 Raine 与 ChatGPT 的互动，并可能通过追悼会上的个人证词来证实聊天记录。** 评论者指出，虽然这一要求看起来具有侵入性，但它是诉讼中证据开示的标准环节。一些人认为 OpenAI 试图验证 Raine 与 ChatGPT 互动的背景，特别是考虑到他曾对系统进行了越狱（jailbroken）。
- [**ChatGPT 救了我母亲的命。**](https://www.reddit.com/r/ChatGPT/comments/1odxuy6/chatgpt_saved_my_moms_life/) (Activity: 1547): **该帖子描述了如何使用 ChatGPT 识别严重的医疗状况，从而促成了及时的医疗干预。在一个案例中，它正确地提示了需要紧急护理的感染；在另一个案例中，它识别出了血栓。这些实例突显了 AI 工具在提供初步医疗建议方面的潜力，尽管它们不应取代专业的医疗咨询。该帖子强调了将 AI 作为紧急健康评估辅助工具的重要性。** 评论反映了许多个人轶事，其中 ChatGPT 提供了关键的健康见解，从而实现了挽救生命的干预。用户分享了 ChatGPT 的建议促使他们寻求进一步医疗咨询的经历，这些建议随后得到了医疗专业人员的证实，强调了该工具在紧急情况下的潜在效用。
    - ChuCHuPALX 分享了一个案例：ChatGPT 被用来识别其父亲被误诊为焦虑症的情况，而他实际上患有严重中风。通过将症状和药物输入 ChatGPT，他们发现了错误并将其转至另一家医院，在那里他得到了适当的治疗。这突显了 AI 工具在交叉验证医疗方案和维护患者权益方面的潜力。
    - touchofmal 描述了 ChatGPT 如何帮助识别出非自主震颤的原因是高剂量抗精神病药物的副作用，这可能导致迟发性运动障碍（tardive dyskinesia）。这促使他们咨询了另一位精神科医生，证实了 ChatGPT 的建议。此案例强调了 AI 在识别药物副作用和促使进一步医疗咨询方面的效用。
    - Single-Intention-535 讲述了一次经历：ChatGPT 建议针对败血症（sepsis）症状立即入院，而这些症状最初被医生忽视了。AI 的建议促成了及时的干预，防止了潜在的生命危险。这说明了 AI 在紧急医疗场景中提供关键第二意见的作用。

---

# AI Discord Recap

> 由 X.ai Grok-4 生成的摘要的摘要的摘要
> 

**主题 1. AI 模型传闻与发布热潮**

- **Gemini 3 暗示即将发布**：用户在 Gemini 网站上发现彩蛋后，推测 **Google Gemini 3** 即将发布；由于幻觉问题，关于其就绪程度引发了争论。一些人预测将在 2024 年推出，而另一些人则因资源需求对其表示怀疑。
- **Sora 进化，支持角色客串与编辑**：OpenAI 的 **Sora** 新增了[角色客串](https://video.twimg.com/amplify_video/1981118365541310464/vid/avc1/896x512/hoABFvQ_q78wMp_h.mp4)功能（从动物和玩具开始），并增加了基础编辑工具和 Android 支持。社区的兴奋点集中在热门 UI 和社交分享功能，以增强视频生成体验。
- **LTX-2 展现本地视频魔力**：Lightricks 发布了[开源 LTX-2](https://xcancel.com/ltx_model/status/1981346235194683497)，支持同步音频的 **4K/50 fps** 视频，可在消费级 GPU 上运行 10-15 秒。权重将于今年晚些时候发布，引发了对专业级本地 AI 工具的热议。

**主题 2. 硬件竞赛与 GPU 争夺**

- **Anthropic 抢购吉瓦级 TPU**：据 [CNBC 报道](https://www.cnbc.com/2025/10/23/anthropic-google-cloud-deal-tpu.html)，Anthropic 在一项数百亿美元的交易中，确保了到 2026 年获得约 **100 万个 Google Cloud TPU** 和超过 **1 GW** 的电力容量。外界推测，这种巨大的算力提升将降低 API 成本并延长上下文窗口。
- **Mojo 攻克 GPU 内核**：Modular 的 [Mojo workloads](https://arxiv.org/abs/2509.21039) 在 **NVIDIA H100** 和 **AMD MI300A** GPU 上实现了基于 MLIR 的 HPC 内核，在四项科学任务中达到了厂商基准水平。GitHub 仓库 [Mojo-workloads](https://github.com/tdehoff/Mojo-workloads) 推动了关于可移植性能的讨论。
- **云端租赁优于本地购买**：专家建议在购买本地硬件前先租用数十小时的云端 GPU，估计包括电力和管理成本在内的收支平衡点约为半年的租赁费用。

**主题 3. 工具 Bug 与性能提升**

- **Unsloth QAT 获得 NVIDIA 认可**：Unsloth 宣布 [NVIDIA 支持](https://x.com/NVIDIAAIDev/status/1981510959048106262) 其 **QAT** 发布，引发了关于 **Qwen Next 80B** 微调的讨论。依赖项修复包括为 **transformers 4.57.1** 设置 pip 顺序以避开 numpy 冲突。
- **DSPy 规避异步陷阱**：用户正在调试同步运行的 **DSPy async ReAct** 模块，建议使用 `await program.acall` 进行修复，同时抱怨文档令人困惑。通过 [LiteLLM 代码片段](https://discord.com/channels/1161519468141355160/1161519469319946286/1431029715511232512) 进行 Token 追踪，可从 program.history 计算成本。
- **Cursor 自动路由引发开发者不满**：Cursor 的模型随机切换功能因在 LLM 之间进行不透明切换而令用户沮丧，促使人们使用 **/summarize** 技巧来削减上下文和成本。后台 Agent 因启动耗时半天而进度滞后，这被归咎于失控的开发进程。

**主题 4. 关于 Scaling 与神话的研究争议**

- **Scaling 遭遇收益递减**：关于[预训练限制](https://arxiv.org/abs/2506.06266)的论文质疑 Scaling 是否会在 RL 和测试时计算（test-time compute）中达到上限，但辩论认为像 **ChatGPT** 这样的界面开启了时代潮流。批评者认为 HCI 研究缺乏扩展性，并引用 [Minecraft Voyager](https://x.com/DrJimFan/status/1662117799974809603) 作为先驱。
- **MythWorx 宣称在 ARC-AGI 获胜**：MythWorx 在 **1 亿美元** 融资期间声称在 4 小时内实现了 **100% ARC-AGI** 且无需预训练，但缺乏组织者的验证。怀疑者将其与 Theranos 类比，[Greg Kamradt](https://latent.space/) 提供了官方测试。
- **元学习超越 Atari 规则**：[Disco RL 论文](https://www.nature.com/articles/s41586-025-09761-x)利用 Agent 经验来元学习规则，从而击败了 Atari 基准测试，代码库位于 [GitHub disco_rl](https://github.com/google-deepmind/disco_rl)。这引发了关于累积 RL 进化的讨论。

**主题 5. 社区对定价与福利的抱怨**

- **Perplexity 大幅削减推荐奖励**：用户对 **Perplexity AI** 在美国的推荐奖励从 **$20** 降至 **$5** 感到愤怒，部分用户甚至被完全拒绝访问 Pro 版本。在关于速度与 **GPT-4** 的争论中，出现了通过 **PayPal** 和 **Airtel** 获取免费 Pro 的黑客手段。
- **OpenRouter 费用引发不满**：充值被收取 **80 美分** 的服务费，导致 **$5.83** 的付款仅获得 **$5** 的额度，此外还因余额耗尽导致 **DeepSeek v3.2** 停机。Exacto 模型引发了定价纠纷，尽管 **glm-4.6** 的费率保持稳定。
- **Hugging Face Spaces 开始扣除积分**：像 [Sora-2](https://huggingface.co/spaces/akhaliq/sora-2) 这样的 Spaces 现在开始向用户扣除积分，标志着变现模式的转变。更新后的 [RAG 库](https://huggingface.co/kalle07/embedder_collection) 助力检索系统，同时社区也在抱怨针对 **1.1B** 模型的 **6GB VRAM** 优化调整。

---

# Discord: 高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **推荐计划奖励缩减**：用户反映对 **Perplexity AI 推荐计划** 感到困惑和失望，部分用户仅收到 **$5** 奖励，而非宣传的 **$20**。
   - 一位用户指出，奖励缩减可能仅针对美国（USA）用户，而其他用户则反映根本没有获得 **Pro** 访问权限。
- **Perplexity 与 GPT 模型的辩论**：用户正在积极辩论 **Perplexity AI** 相对于 **GPT-3.5** 和 **GPT-4** 等 **GPT 模型** 的优势，并对比了付费版和免费版。
   - 一位用户断言 *Perplexity 比 ChatGPT 好 10 倍且速度更快*，而另一位用户则开玩笑地建议获取 *super grok*。
- **探索免费获取 Perplexity Pro 的途径**：成员们正在分享免费获取 **Perplexity Pro** 订阅的方法，包括通过 **PayPal** 的促销活动和 **Airtel** 的订阅。
   - 此外，一些用户正在寻求关于使用包含的特权以及申请订阅退款的指导。
- **创业想法：针对 Gooning 的自动化内容生成**：一位用户提出了一个专注于为 *Gooning* 自动化生成内容的创业想法，建议它可以生成 *ni hao fine shyts*。
   - 该提议的初创公司旨在生产 *男性和女性* 内容，特别强调亚洲偏好，引发了成员间的进一步讨论。
- **寻求最安全的家庭用车**：一位成员使用 Perplexity AI 询问了 [最安全的家庭用车](https://www.perplexity.ai/search/what-s-the-safest-family-vehic-D4glhRttT3WnhFuyInsfvQ#0)。
   - 未提供关于车辆具体要求或预期用途的其他细节或背景。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Google Gemini 3 发布传闻**：在 Google Gemini 网站上发现线索和彩蛋后，成员们纷纷猜测 **Gemini 3** 可能在今年发布。
   - 一些用户认为它可能即将推出，而另一些用户则认为由于幻觉（hallucination）等问题，它可能尚未准备好发布。
- **LithiumFlow 模型神秘移除**：竞技场中的热门模型 **LithiumFlow** 似乎已被移除，引发了关于其可能集成到 AI Studio 或即将发布的猜测。
   - 一些用户表示失望，而另一些用户则指出它可能对提示词敏感，并产生不一致的结果。
- **Code Arena 的 HTML 升级引发辩论**：Code Arena 接收了包含新提示词的更新，文件创建现在采用 **.HTML** 格式。
   - 成员们建议优化或删除系统提示词中明确的 Tailwind CSS 部分，因为这并不总是必要的，并且可能导致模型创建过于注重 UI 的提示词版本。
- **Sora 2 Pro 视频被辟谣**：一位用户发布了声称由 **Sora 2 Pro** 生成的视频，引发了对其真实性的辩论。
   - 其他成员迅速拆穿了这一说法，暗示这些视频可能来自电影，或者是使用了 RTX 模组的 Minecraft。
- **NimbleBean 成为顶尖视频生成器**：用户们对新的 **NimbleBean** 视频模型（也称为 **Kling 2.5 Turbo Standard**）议论纷纷，它现在是竞技场中排名第一的视频生成器。
   - 一位用户表示它 *产生了完美的运行效果*，许多人无法分辨它是 AI 还是真人，并指出其关键要素在于 *标注（labelling）*。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 的自动模型路由感觉像在玩轮盘赌**：用户对 **Cursor 的自动路由 (auto-routing)** 到不同模型表示沮丧，将其描述为一种“轮盘赌”，会根据 prompt 和 **context** 的不同，在聪明和愚蠢的 LLM 之间进行不稳定的切换。
   - 用户指出，缺乏透明度使得很难确定路由到了哪些模型，一些人建议通过 **集群基准测试性能 (cluster benchmark performance)** 来估算性能。
- **Cursor 的计费引发使用焦虑**：用户反映 **Cursor 的新方案** 迫使他们使用 **Auto 模型**，并在高级模型使用量减少剩余天数时发出警告；但一位用户表示，只要能通过延迟方式无限使用高级模型，他就会*尽可能最大限度地使用高级模型*。
   - 成员们注意到 **Auto 模型** 使用了 **Claude 4.5 Sonnet Thinking** 且价格更便宜，因此建议经常使用 **/summarize** 来降低 context window 并减少成本。
- **后台 Agent 遭遇长达半天的启动等待**：用户抱怨 **Background Agent** 启动时间过长，报告称在 git cloning、repo scanning 和初始 linting 方面需要等待长达*半天*的时间。
   - 成员们幽默地推测，可能是有开发人员不小心让一个 **Background Agent** 无限期运行了，暗示资源管理效率低下。
- **并行 Agent 因易用性问题陷入瘫痪**：一位用户难以找到 **parallel agent** 的可行用例，而另一位用户在启用 MCP 以在容器中运行 `cursor-agent` 时遇到问题，提示 *✗ MCP login failed*。
   - 该错误表明 **MCP server "atlassian"** 在加载前需要批准，暗示可能存在配置或权限问题。
- **UI 错误消息导致调试头疼**：一位用户建议澄清一条 **UI 错误消息**，该消息在本地分支未推送到 origin 时引起了困惑，建议提供更明确的消息，如 *"The branch {branch name} does not exist in the remote repository."*
   - 用户最初怀疑是 *environment.json* 文件的问题，这凸显了清晰的错误消息对于高效调试的重要性。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **NVIDIA 介入 Unsloth QAT**：Unsloth [宣布了 NVIDIA 对其 QAT 发布的支持](https://x.com/NVIDIAAIDev/status/1981510959048106262)，这表明业界对量化感知训练 (quantization-aware training) 的兴趣日益浓厚。
   - 该公告引发了关于微调 **Qwen** next 80b 的讨论，尽管有些回复是求职推广。
- **Unsloth 用户遭遇依赖灾难**：安装 **Unsloth** 的用户报告了依赖冲突，特别是 `transformers` 和 `numpy` 版本问题，导致安装失败。
   - 解决方法包括特定的安装顺序（先 `pip install unsloth` 再 `pip install transformers==4.57.1`）以及使用 Docker 来绕过版本不兼容问题。
- **Karpathy 对关键 Token 初始化的赞赏**：Karpathy [在 X 上](https://x.com/karpathy/status/1980397031542989305)坦言自己几乎不使用 **tmux**。
   - 他强调 **良好的 token 初始化** 非常重要，尤其是在没有漫长教学阶段的情况下。
- **QAT 到 GGUF 的转换即将推出**：一名团队成员确认，目前尚不支持将 **QAT (Quantization Aware Training)** 模型转换为 **GGUF** 格式，但计划在年底实现。
   - 这一新功能将允许低功耗设备利用 **QAT GGUF** 模型。
- **Docker 中的 Llama.cpp 编译速成**：一位用户报告称，由于缺少 CUDA 编译器 **nvcc**，**llama-cpp 编译** 在官方 Unsloth Docker 镜像中失败。
   - 排查建议包括检查 Unsloth 版本、CUDA 设置、确保正确安装了 NVIDIA container toolkit，并使用 `--gpus all` 标志运行容器。

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Deepseek OCR 面临审查**：成员们讨论认为，全新的 [**Deepseek OCR**](https://example.com/deepseek-ocr) 在部署前需要经过严格审查。
   - 成员们建议在依赖其性能之前，进行彻底的测试和评估。
- **Zero3 配置导致显存爆炸！**：一位成员报告称，在使用 **LoRA**（r=8）、flash attention、deepspeed zero 3 和 bf16 精度，且 batch size 为 1 的情况下，训练序列长度为 8k 的 **1b Gemma 3** 模型时，**GPU RAM** 出现爆炸。
   - 另一位成员指出，他们的 **Zero3 config** 可能存在问题，暗示这是一个配置错误。
- **Tiny LLM 遭到吐槽**：一位成员分享了一个从零开始构建的 [GitHub 仓库](https://github.com/ker2x/MytinyLLM)，用于他们的 **Tiny LLM** 学习练习。
   - 另一位成员对这一贡献表示不屑，称一个 **400k 参数模型** 只是个玩具，开箱即用下无法学到任何有意义的东西。
- **Hugging Face Spaces 现在需要你的额度**：一位成员注意到，包括 **Sora** 和 **Veo 3.1** 在内的一些 **Hugging Face Spaces** 现在需要付费才能使用*你的*额度，参见 [Sora space](https://huggingface.co/spaces/akhaliq/sora-2)。
   - 他们链接了一个带有警告的 **Sora space** 示例，指出使用该服务现在需要付费，这标志着平台货币化策略的转变。
- **Hugging Face 上的 RAG 基础库已更新**：一位成员为所有从事 **RAG** 工作的人员分享了更新后的基础库，地址为 [huggingface.co/kalle07/embedder_collection](https://huggingface.co/kalle07/embedder_collection) 和 [huggingface.co/Pacific-Prime/pacific-prime](https://huggingface.co/Pacific-Prime/pacific-prime)。
   - 此次更新旨在为构建 **Retrieval-Augmented Generation** 系统提供更好的起点。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT 扩展洞察一切**：OpenAI 推出了 [**ChatGPT Chrome 扩展**](https://video.twimg.com/amplify_video/1981054579648368640/vid/avc1/1280x720/YmADR8pumfl1Zwin.mp4)，使 **ChatGPT** 能够查看当前页面，并在无需切换标签页的情况下提供即时、相关的答案。
   - 这种集成旨在通过结合上下文的 **ChatGPT** 响应来简化信息获取并增强用户体验。
- **Sora 将首推角色客串功能**：**Sora** 将引入 [角色客串 (character cameos)](https://video.twimg.com/amplify_video/1981118365541310464/vid/avc1/896x512/hoABFvQ_q78wMp_h.mp4) 功能，首批包括动物和玩具，允许用户在生成的视频中加入熟悉的角色。
   - 即将推出的功能包括热门客串 UI、基础视频编辑工具、社交分享选项、Feed 流质量改进以及 **Android** 版本的 **Sora**。
- **计算器通过 GPT 变得能言善辩**：一篇 **Wired** 文章和 [YouTube 视频](https://www.youtube.com/watch?v=olcZdTRdnQg) 重点介绍了如何使用 **ESP32** 在 **TI-84 计算器**上集成 **GPT**，为 AI 辅助解题打开了大门。
   - 社区幽默地辩论了使用这种装置在数学考试中作弊是否应该因其技术独创性而获得赞赏。
- **非官方 ChatGPT 扩展修复 Bug**：一位用户分享了一个名为 **ChatGPT LightSession** 的 [新非官方 ChatGPT 扩展](https://www.reddit.com/r/ChatGPT/) 链接，报告称它修复了*红色文本错误*并加快了工作速度。
   - 另一位用户警告要*谨慎使用此扩展*，建议坚持使用公开可访问的内容，以确保个人信息安全。
- **AI 语音克隆进入 DIY 时代**：一位成员分享了一个 [DIY 语音 AI 项目](https://www.instructables.com/Create-Your-Own-AI-Voice-Agent-Using-EchoKit-ESP32/)，允许用户使用 **EchoKit ESP32** 克隆语音。
   - 这个开源项目促进了针对各种应用（包括视频生成）的自定义语音 AI 训练，目前关于其潜在用途和伦理影响的讨论正在进行中。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Gemini 在 DSPy 的结构化输出中失败**：有用户报告称，在 **DSPy** 中使用带有 `responses` 参数的 **Gemini models** 时，配合结构化输出适配器会抛出警告和错误，具体表现为 *"Stub file not found"* 错误。
   - 目前的讨论集中在如何为 **Gemini models** 启用结构化输出，以确保在 Python 中使用 **DSPy** 时的类型安全。
- **Refine 漏洞：过早终止**：一位用户发现 `dspy.Refine` 并不一定会运行完所有的 N 次迭代，如果 `reward_fn` 超过了设定的 `threshold`（阈值），它就会提前停止，这与 [文档](https://dspy.ai/tutorials/output-refinement/best-of-n-and-refine/?h=refine#refine) 最初的假设相反。
   - 该用户提供了一个 [demo](https://github.com/stanfordnlp/dspy/blob/main/dspy/predict/refine.py#L142) 来说明如果 `reward_fn` 超过阈值，refine 循环就会中断，并指出 `Refine` 每次都会对模块进行深拷贝。
- **DSPy 的异步 ReAct：并不那么异步？**：一位用户在尝试使用 `dspy.asyncify` 异步运行两个 `ReAct` 模块时遇到问题，指出它们似乎是在同步执行，并寻求正确实现的指导。
   - 一名成员建议使用 `await program.acall` 而不是 `await program.aforward`，并在模块内实现 `async def aforward(...)`，同时指出 **DSPy** 的文档令人困惑。
- **DSPy 追踪 Token 和成本**：用户正在讨论如何在 **DSPy** 中追踪 Token 使用情况和成本，一名成员分享了一个 [代码片段](https://discord.com/channels/1161519468141355160/1161519469319946286/1431029715511232512)，利用 `program.history` 和 **LiteLLM** 模型定价来计算成本。
   - 他们澄清说，`if "usage"` 条件确保了成本计算是基于实际的模型使用情况（考虑了缓存），该成员还指出 `program.history` 中的 `usage` 与 `result.get_lm_usage()` 类似。
- **自定义导致困扰**：一位用户对 **DSPy** 的复杂性表示沮丧，特别是在访问 `ReAct` 模块输出方面，认为在自定义循环中实现 LLM 调用更容易获得更好的控制和 UI 集成。
   - 用户们探索了其他方法，例如在 `ReAct` 模块中对 `aforward` 方法进行子类化和重新实现，如[此代码片段](https://cdn.discordapp.com/attachments/1431066467897577492/1431067652763156666/message.txt?ex=68fc111c&is=68fabf9c&hm=956afd7b5a72fb6ea1288e1d2656952b4e64d31baf63e1feccda13f151437ba5&)所示。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo ❤️ Rust：C ABI 弥补差距**：Mojo 可以通过 **C ABI** 与 Rust、C++、Zig 或任何其他支持 **C ABI** 的语言进行通信，类似于宿主 C/C++/Rust 进程嵌入 Python 解释器的方式。
   - 然而，使用 **C ABI** 需要过多的手动干预，这促使人们寻找能让包像平台原生一样可用的解决方案。
- **Python 力量：Mojo 的运行时秘密 🤫**：Mojo 与 **CPython** 的交互方式类似于宿主 C/C++/Rust 进程嵌入 Python 解释器，使其能够利用 Python 生态系统；目标是尽可能实现**零运行时开销**。
   - 这在证明加速热循环（hot loop）代码的可行性时尤为重要。
- **类型系统谈：Mojo 展示其实力 💪**：Mojo 的类型系统非常强大，Dependent Haskell 可能是最流行的能表达 Mojo 所能表达的一切的语言，而拥有更强大类型系统的语言可以消费来自类型系统较弱语言的代码。
   - 为了让从另一种语言调用 Mojo 函数感觉像原生调用，另一种语言需要一个比 Mojo 更强大或同等强大的类型系统，可能包括依赖类型和线性类型。
- **互操作见解：动态与静态的对决 ⚖️**：使用编译型语言既提供了一定程度的动态性，又使 Mojo 代码能够从你的代码中获得类型系统保证，从而大大减少了类型内省（type introspection）。
   - 如果宿主语言是静态且编译的，那么在其之上实现的新语言也应该是静态且编译的，以最小化开销，因为静态编译语言可以被编译为发射动态代码（如果动态语言更具表现力）。
- **Mojo 错失管道操作符（目前）**：一位成员注意到 Mojo 缺乏像其他语言中的 `|>` 或 `>>` 那样的管道字符或函数组合，由于 Mojo 尚未稳定，团队可能短期内不会考虑它，但指出了 [这个 GitHub issue](https://github.com/modular/modular/issues/213) 作为一个相关的特性请求。
   - 另一位成员指出 Python 也不具备该特性。

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **DeepSeek 遭遇额度紧缺**：由于额度耗尽，**DeepSeek/deepseek-v3.2-exp** 模型在 OpenRouter 上面临可用性问题，导致用户收到 **402 Proxy Error**。
   - 一些用户最初误以为该错误是特定于账户的，凸显了错误消息提示方面的困扰。
- **充值到账延迟**：用户报告了 OpenRouter 额度充值延迟以及 **DeepSeek** 供应商余额耗尽的问题。
   - 一位用户揶揄道，问题要等到 *"直到 Toven 看到它"* 才能解决。
- **OpenRouter 充值手续费曝光**：用户仔细研究了在 OpenRouter 增加额度的服务费，注意到支付 5.83 美元仅能获得 5 美元的额度。
   - 额外的 0.83 美元用于支付服务费，约为 **80 美分（美金）**。
- **Exacto 模型引发定价争议**：将 **exacto** 模型作为独立的 API 变体实施，引发了对潜在价格上涨以及 Token/应用统计数据碎片化的担忧。
   - 尽管担心成本增加，但 **glm-4.6** 在 **exacto** 和 **non-exacto** 端点的价格保持一致；然而，一位成员认为这种做法会*拆分 Token 和应用统计数据*。
- **OpenRouter API 遭遇访问问题**：用户在 OpenRouter 的 completions 端点遇到了登录问题和 **401 unauthorized errors**。
   - 该问题可能特定于某些工具，cURL 请求可以成功，而其他工具则失败。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Linear 转向自动驾驶 SaaS**：Linear 正在从反应式聊天机器人 UI 转向**主动式 AI**（被称为 *"self-driving SaaS"*），以自动推进任务。
   - 评论者赞扬了这一大胆战略，将其比作软件自动处理任务的 **Level-5 autonomy**。
- **Airbnb 赞赏阿里巴巴的 Qwen**：**Airbnb CEO** 强调 **Alibaba 的 Qwen 模型**比 **OpenAI 的最新模型**更快、更经济，并展示了 **Cerebras 2,500-tokens/sec 的推理速度**。
   - 讨论围绕着边际准确率提升与成本降低相比重要性下降展开，安全性、主权和硬件演进成为关键考虑因素。
- **MythWorx 的 ARC-AGI 声明面临审查**：**ARC Prize** 的组织者尚未确认 **MythWorx** 在 **ARC-AGI** 上达到 **100%** 的声明，该声明是在一笔 **1 亿美元**融资期间宣布的。
   - **Greg Kamradt** 表示如果 MythWorx 提交准确，他愿意进行官方测试，但由于缺乏验证以及类似于 **Theranos 式的宣传**，质疑声依然存在。
- **Anthropic 获得巨量 TPU 算力**：Anthropic 披露了一项协议，到 **2026** 年将整合约 **100 万个 Google Cloud TPU** 和超过 **1 GW** 的容量，引发了对计算能力的兴奋。
   - 反应包括猜测这是否会降低 API 成本、增加使用量或扩展上下文窗口；该交易价值数百亿美元，并将在 2026 年上线超过 1 吉瓦的 AI 算力 ([CNBC 文章](https://www.cnbc.com/2025/10/23/anthropic-google-cloud-deal-tpu.html), [Anthropic 公告](https://www.anthropic.com/news/expanding-our-use-of-google-cloud-tpus-and-services))。
- **Lightricks 发布本地潜空间模型 LTX-2**：**Lightricks** 发布了 **LTX-2**，这是一个开源创意引擎，可以在消费级 GPU 上生成约 10–15 秒的[同步 **4K/50 fps** 音视频](https://xcancel.com/ltx_model/status/1981346235194683497)。
   - 社区对民主化专业级 AI 视频工具的前景感到振奋，权重预计将于今年晚些时候发布，目前 API Playground 已上线。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Mojo HPC Kernels 登陆 GPU**：[Mojo](https://arxiv.org/abs/2509.21039) 和 [Mojo Workloads](https://github.com/tdehoff/Mojo-workloads) 现在正被用于在 GPU 上实现**基于 MLIR 的性能可移植 HPC 科学计算算子（Kernels）**。
   - 该论文针对四种科学计算工作负载，并将其性能与 **NVIDIA H100** 和 **AMD MI300A GPU** 上的厂商基准进行了对比。
- **为 WGMMA 解码 SMEM 描述符**：一位用户分享了用于 **WGMMA** 指令的自定义共享内存（**SMEM**）描述符实现，包括 `matrix_descriptor_encode` 和 `make_smem_descriptor` 函数，该实现源自 [PTX 指南](https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-descriptor-format)。
   - 另一位成员建议参考 DeepSeek-AI 的 DeepGEMM [实现](https://github.com/deepseek-ai/DeepGEMM/blob/c9f8b34dcdacc20aa746b786f983492c51072870/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh#L275)，以处理类似的内存屏障（memory fences）和 **WGMMA** 问题。
- **在投入之前先租用云端资源**：一位成员建议在投入至少几十小时进行严肃工作以评估成本和收益之前，先*租用云端资源*。
   - 他们估计，投资本地硬件的*盈亏平衡点*至少是*半年的云端租用时间*。
- **HQQ+ 迁移至 Dropbox**：**MobiusML** 域名今天正在迁移至 **Dropbox**，导致原始链接失效，用户现在应将博客文章和 GitHub 仓库链接中的 `mobiusml` 替换为 `dropbox`。
   - 一位成员在发现 [https://mobiusml.github.io/1bit_blog/](https://mobiusml.github.io/1bit_blog/) 无法访问后，正在寻找 **HQQ+ 博客文章** 的有效链接。
- **Torch/XLA 迎来新转折，依然具有参考价值**：尽管 [torch/xla](https://github.com/pytorch/xla/issues/9684) 有了*新方向*，但 **Lazy Tensor 论文** 对于 picograd 在 tinygrad 中强行加入 eager 模式以降低门槛（lower the ramp）仍然具有参考意义。
   - [Lazy Tensor 论文](https://arxiv.org/pdf/2102.13267) 探讨了如何使用 picograd 将 eager 模式集成到 tinygrad 中。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Discord 引入“待定”成员状态**：Discord 为加入服务器的新成员引入了“待定”（pending）状态，该状态在服务器标签中添加“加入”选项后激活。
   - 成员确认在管理员批准前处于“待定”状态；关于这种手动批准流程是否仍在运行引发了讨论。
- **提议建立类似互联网的 AI 网络**：一位成员提议建立一个类似于互联网的 **AI 网络**，成千上万个模型通过一种协议进行通信来回答问题。
   - 另一位成员请求提供此类系统实现和潜在架构的具体案例。
- **新模型声称拥有惊人性能**：一位成员声称其 **50M** 参数模型的 loss 达到了 **0.223**，而其 **1B** 参数模型的验证 loss 约为 **0.197**，对应的困惑度（perplexity）约为 **1.22**，并附上了一张[图片](https://cdn.discordapp.com/attachments/747850033994662000/1431083629748027402/image.png?ex=68fc1ffd&is=68face7d&hm=c347fd3fa1d4dae87e30579f0723253fd5c83a7197c57f6583d66e7d2ba5ca67&)。
   - 其他成员表示怀疑，认为该模型存在 bug，或者作者只是在寻求关注而非寻求帮助，并建议他们通过推理进行调试。
- **元学习强化学习发现新规则**：一篇[论文](https://www.nature.com/articles/s41586-025-09761-x)探讨了从智能体（**Agent**）跨环境的累积经验中进行**元学习**（**meta-learning**），产生了一条在 **Atari 基准测试**上超越现有规则的规则。
   - 该项目的代码库已在 [GitHub](https://github.com/google-deepmind/disco_rl) 上发布。
- **因果抽象热度减退**：成员们询问为何最近关于**因果抽象**（**Causal Abstractions**，见[论文](https://arxiv.org/abs/2301.04709)）的讨论变少了，这曾是 2023 年的热门话题。
   - 有人指出，该框架虽然有用，但受到隐式线性假设（[论文](https://arxiv.org/abs/2507.08802)）、难以选择合适的抽象层级（[综述](https://arxiv.org/abs/2408.01416)）以及单一行为存在多个有效抽象（[论文](https://arxiv.org/abs/2502.20914)）等问题的困扰。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Django 拆分为微服务**：一位成员正在考虑将一个使用 **bi-encoders、cross-encoders、YOLO models** 的 Django 应用拆分为独立的微服务，以卸载推理任务并解决在有限 VRAM (32GB) 下可能出现的请求排队问题。
   - 作为回应，另一位成员建议顺序调度请求或并行运行它们，鼓励他们利用并行 RPC 调用来最大化 GPU 的能力。
- **Meta 的权重硬度黑客技巧 (Weight Hardness Hack)**：受真实神经元连接的启发，一位成员链接了 [Meta 的实现](https://x.com/realJessyLin/status/1980662516285075762)，在每个权重上使用额外的标量来存储其“硬度”或“软度”。
   - 该概念涉及根据计算结果调整权重，强化有益的连接并削弱有害的连接，从而在不修改不可逆连接的情况下有效地进行学习。
- **MythWox 神秘地掌握了 ARC-AGI**：一位成员分享了 [MythWorx.ai](https://mythworx.ai/capabilities/) 的链接，声称在 4 小时内无需预训练即可在 **ARC-AGI 1** 上达到 **100%** 的准确率。
   - 该成员表示怀疑，质疑这些能力的真实性，另一位成员则反应出困惑和不解。
- **Jsonnet 处理函数式配置**：成员们辩论了 [Jsonnet](https://jsonnet.org/) 是否过度设计，而另一位成员则赞赏其可定制性以及 **DeepSeek** 在配置中对它的使用。
   - 一位成员将其比作 **Nix**，即使用函数式编程定义哈希表，并质疑其对于 JSON 的可读性。另一位成员提议讨论 **Jsonnet**/**VinePPO**，随后提供了与此相关的 [ICML](https://icml.cc/virtual/2025/poster/45526)、[OpenReview](https://openreview.net/forum?id=5mJrGtXVwz)、[ArXiv](https://arxiv.org/abs/2410.01679) 和 [GitHub](https://github.com/McGill-NLP/VinePPO) 链接。
- **Brave 浏览器浏览“不可见”注入**：一位成员分享了 [Brave Browser 博客文章](https://brave.com/blog/unseeable-prompt-injections/)，讨论了**不可见提示词注入 (unseeable prompt injections)**。
   - 未添加进一步讨论。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **模型模仿牙买加口音**：一位用户报告称，**4o model** 在语音模式下仍会发出**牙买加口音**，而 **model 5** 虽然声称可以做到，但根本没有改变声音。
   - 该用户指出，这是“我关心的少数指标之一”。
- **关于带 Bug 开源的辩论**：一位成员正在考虑开源他们的软件，但正在纠结是延迟发布以修复 Bug，还是直接发布带有烦人垃圾内容的代码。
   - 他们的应用旨在让任何人都能在本地训练 AI 模型而无需 SaaS，并计划包含一个由 **40k images** 组成的 **Weeb Sam finetune**。
- **ChatGPT 遭遇海马惊吓**：一位用户分享了一条关于 **ChatGPT bug** 的[推文](https://fxtwitter.com/ltx_model/status/1981346235194683497)，该 Bug 在发送提示词“is there an emoji of a seahorse?”时会出现。
   - 一位评论者推测，这看起来像是一个正在开发中的 **Unreal Engine/Runway/Wan 竞争对手**。
- **新论文测试 Scaling 极限**：一位成员分享了一篇[论文](https://arxiv.org/abs/2506.06266)，并质疑 **scaling** 是否已达极限，考虑到 **pretraining**、**test-time compute** 和 **RL** scaling 的收益递减。
   - 其他成员表示反对，认为并非如此，“RL、test time compute 等只是 AI scaling 的一个组成部分”。
- **界面在 Scaling 中起关键作用**：一位成员认为，**interface**、**workflows** 和 **infra** 是 AI scaling 极其重要的组成部分，正如 **ChatGPT interface** 开启了一个研究时代。
   - 他们指出，“Claude 将其新的提示系统称为 'Skills'，但实际上这在几年前就由 Minecraft Voyager 开创（并命名）了”，并引用了 [Jim Fan 的推文](https://x.com/DrJimFan/status/1662117799974809603)。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **AlphaXiv 追踪 AI 论文起源**：一位成员分享了 [AlphaXiv](https://www.alphaxiv.org/labs/geo-trends)，这是一个追踪 **AI 研究论文**地理起源的工具。
   - 另一名成员因多次违规发布广告被禁言。
- **Kimi K2 在 Chutes 上更便宜但存疑**：一位成员询问 **Chutes** 与 **Moonshot AI** 官方相比在运行 **Kimi K2** 时的质量和数据政策。
   - 另一位成员声称 **Chutes** 在没有数据政策的情况下使用用户数据进行训练，且在线率（Uptime）较差；其 Tool Call 准确率仅为官方 API 的一半，此事在 [Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1da7j6g/did_you_know_you_can_ban_chutes_openrouter_go_to/) 上已被做成梗图。
- **通过 Git 控制家庭服务器版本？**：一位成员询问是否可以使用 **Git** 来管理家庭服务器版本，以便在系统崩溃后辅助重置。
   - 社区忽略了该用户的疑虑。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **混合架构引发辩论**：一位用户提议使用结合本地计算和存储的*混合架构*，以避免浪费可用资源。
   - 该建议涉及构建大型原生应用、在本地处理数据集，以及在本地硬件上运行资源密集型 AI 模型。
- **Pro 计划用户指责“诱导转向”**：一位用户对 **Pro 计划** 现在设有额度限制（Credit Limits）表示不满，声称最初宣传时是*无限量*的。
   - 该用户认为这对已付费月份不公平，并询问是否有其他人想测试 Pro 功能。
- **Manus 发布 Logic Arenas 应用**：一位用户注意到 **Manus** 在一小时内使用极少额度创建了一个全栈 **AI Arenas 应用**（[图片](https://cdn.discordapp.com/attachments/1349440650495398020/1430878322073796628/2025-10-23_12-15.jpg?ex=68fc0988&is=68fab808&hm=efef49879866f78ee43ea3c281eca345ab2bcf800110c561cc4e81bc723f5219&)）（*尚未测试*）。
   - 该 **Logic Arenas** 应用包含 *Heart vs Mind*、*Dual Parents* 和 *Moral Court* 竞技场，使用了 **Kotlin**、**Ktor**、**Redis**、**Jetpack Compose** 和 **Material 3**，跨 28 个文件共计约 7k 行代码（LOC）。
- **Manus 代码因使用弃用模块被吐槽**：一位用户观察到，如果 **Manus** 能使用非弃用（Non-deprecated）的代码、模块和插件，效果会更好，但这需要大量工作。
   - 尽管 **Manus** 宣称拥有精美的 Material 3 深色主题（[图片1](https://cdn.discordapp.com/attachments/1349440650495398020/1430917287363350728/2025-10-23_14-51.jpg?ex=68fc2dd2&is=68fadc52&hm=4e68706e60e030a96d99d157b3abdacc9d2b99077639e85f545cb96a0620d614&), [图片2](https://cdn.discordapp.com/attachments/1349440650495398020/1430917287707279391/2025-10-23_14-47.jpg?ex=68fc2dd2&is=68fadc52&hm=04f22a7c26e41abcb491ebfcbdcb2c4ec8b05ad504eeb504de7aa4c2bc80f3f4&)），但初始 Android 应用设计似乎过时了；不过， Claude Code 仅通过一个 Prompt 就对其进行了改进（[图片](https://cdn.discordapp.com/attachments/1349440650495398020/1430971270219956234/2025-10-23_18-26_1.jpg?ex=68fbb758&is=68fa65d8&hm=d73323d08753d88efdd123cd9be451622fbb989859f6077c67b76d45598ebfcd&)）。
- **Manus 搞砸了作业**：一位用户声称 **Manus** 未能正确解决一项家庭作业，尽管用户提供了类似的示例。
   - 用户提到提供了两个 Notebook，但生成的 PDF 杂乱无章，凸显了 **Manus** 无法完成此类任务。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider API 密钥更便宜？**：一位用户询问使用带有 API 密钥的 **aider** 是否能达到 **gemini-cli** 的性价比，通过 `/tokens` 管理密钥、清除历史记录并监控上下文大小。
   - 他们指出 **gemini-cli** 每月约 **$20 USD** 的计划名义上提供每日 **1500** 次请求和 **1M Token** 的上下文窗口，在 `grep` 等文件系统操作上优于 **aider** 的界面，但缺乏 Repo Map 功能。
- **Playwright 在 Fedora 上运行困难**：一位用户询问是否有人在 **Fedora** 上成功运行了 **Playwright**。
   - 未提供任何变通方法或解决方案。
- **社区关注 Aider 的未来**：成员们想知道 **Aider** 的未来发展方向，并等待其创始人 **Paul Gauthier** 的消息。
   - 社区期待该项目即将到来的开发进展和战略决策。

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **tinygrad 亮相 pytorchconf！**: 一张 **tinygrad** 的 [幻灯片](https://cdn.discordapp.com/attachments/1068976834928193609/1430758414744686655/HDjLLqQ.png?ex=68fc429c&is=68faf11c&hm=13bea9c27e9b130accbe594e495dbd0e6813cb720aaabb5bb6f2375495a385b5&) 出现在了 **pytorchconf** 上，展示了 **tinygrad** 在 PyTorch 生态系统中的重要性。
   - 这突显了 **tinygrad** 在更广泛的机器学习社区中日益增长的采用率和相关性。
- **tinygrad 壮大开发团队**: 一名成员询问了关于成为 **tinygrad** 开发者的文档，表明了对为项目做贡献的兴趣。
   - 这暗示了 **tinygrad** 贡献者群体潜在的增长和扩张。
- **JITBEAM 专注于 Jitted Kernels**: 一名成员指出 **JITBEAM** 仅对 *jitted kernels* 进行 beaming。
   - 未提供关于 "beaming" 细节及其对 **tinygrad** 性能影响的进一步背景信息。



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Block 将 MCP 工具比作食人魔 (Ogres)**: 一名成员分享了来自 [Block 的博客文章](https://engineering.block.xyz/blog/build-mcp-tools-like-ogres-with-layers)，将构建 **MCP tools** 比作构建 *带有层级* 的食人魔。
   - 这一比喻暗示了构建健壮且复杂系统的分层方法。
- **MCP 网站请求会议日程**: 频道发布了一个提醒，要求 **工作组负责人 (working group leads)** 在 [Model Context Protocol 网站](https://meet.modelcontextprotocol.io/)上发布他们即将举行的会议时间。
   - 会议组织者可以从“过去活动 (Past Events)”部分克隆之前的活动，以加快调度流程。



---


**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。


---


**Windsurf Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。


---



您收到此邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。


---

# Discord: 各频道详细摘要与链接





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1430677331239899186)** (1422 条消息🔥🔥🔥): 

> `推荐计划, GPT 模型, 免费 Pro 账户, Gooning 创业公司` 


- **推荐奖励引起混淆**: 用户对推荐计划表示困惑，一些人报告说他们只收到了 **$5** 而不是广告宣传的 **$20**，还有一些人根本没有获得 Pro 权限。
   - 一位用户表示以前是 $20，但 *现在来自美国的用户只给 $5*。
- **用户辩论 Perplexity vs GPT**: 用户正在辩论 **GPT 模型**（3.5 vs 4）的能力，并讨论付费版和免费版之间的区别。
   - 一位用户指出 *Perplexity 比 ChatGPT 好 10 倍且速度更快*，而另一位用户开玩笑说他们 *本该买 super grok*。
- **Pro 特权与免费账户**: 用户讨论了免费获取 **Perplexity Pro** 的方法，例如通过 **PayPal** 促销或 **Airtel** 订阅。
   - 还有一些用户请求退款并询问如何使用特权。
- **爱好者宣布 Gooning 创业公司**: 一位用户提议成立一家 **Gooning 机器创业公司**，实现内容生成的自动化，*它将展示 ni hao fine shyts for gooning*。
   - 成员们讨论了这家旨在创建 *男性和女性* 内容的创业公司，重点关注亚洲偏好。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1430984118543450173)** (2 条消息): 

> `最安全的家庭车辆, 计算证据` 


- **成员寻找最安全的家庭车辆**: 一名成员询问 Perplexity 关于 [最安全的家庭车辆](https://www.perplexity.ai/search/what-s-the-safest-family-vehic-D4glhRttT3WnhFuyInsfvQ#0)。
   - 未提供额外的背景或讨论。
- **通过计算发现的证据**: 一名成员链接到了一个关于 [计算证据 (computational evidence)](https://www.perplexity.ai/page/computational-evidence-for-rec-MZ.AjbR6SlGMwJpoCK7cCA) 的 Perplexity 页面。
   - 未提供额外的背景或讨论。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1430681312255148144)** (1333 条消息🔥🔥🔥): 

> `Gemini 3 发布, LithiumFlow 移除, Code Arena 系统提示词, Sora 2 Pro, NimbleBean Kling 2.5` 


- **Google Gemini 3 传闻即将发布！**：成员们在 Google Gemini 网站上发现线索和彩蛋后，推测 **Gemini 3** 可能在今年发布，尽管 Logan 尚未发布相关推文。
   - 一些用户认为它可能很快就会推出，而另一些人则认为由于幻觉（hallucination）问题以及运行世界模拟器（world simulator）需要大量资源，它可能还没准备好发布。
- **LithiumFlow 从竞技场消失**：竞技场中的热门模型 **LithiumFlow** 似乎已被移除，引发了关于其可能集成到 AI Studio 或即将正式发布的猜测。
   - 一些用户表示失望，而另一些人指出该模型可能对提示词（prompt）敏感，且产生的结果不一致。
- **Code Arena 更新了新的提示词和 .HTML 功能**：Code Arena 进行了更新，引入了新的提示词，并支持以 **.HTML** 格式创建文件。
   - 成员们建议在新的 Code Arena 中优化或移除系统提示词中显式的 Tailwind CSS 部分，因为这并不总是必要的，且可能导致模型生成过于侧重 UI 的版本。
- **关于新 Sora 2 Pro 视频的争论**：一名用户发布了一些视频并声称是由 **Sora 2 Pro** 生成的，引发了关于其真实性的辩论。
   - 然而，其他成员很快拆穿了这一说法，指出这些视频更有可能来自某部电影，或者是使用了 RTX 模组的 Minecraft。
- **NimbleBean 现成为最佳视频生成器**：用户讨论了竞技场上新的 **NimbleBean** 视频模型，即 **Kling 2.5 Turbo Standard** 模型，它目前是排名第一的视频生成器。
   - 一位用户表示它 *实现了完美的运行效果*，许多人无法分辨它是 AI 还是真人拍摄，并指出其关键要素在于 *标注（labelling）*。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1430677736669843496)** (411 条消息🔥🔥🔥): 

> `Cursor 自动模型路由, Cursor 使用与计费, Cursor 版本控制, Background Agents 失控, GCP 部署` 


- **自动模型轮盘赌？**：用户讨论了避免自动路由到能力较弱模型的技巧，指出选择聪明还是愚蠢的模型感觉就像 *轮盘赌*，并怀疑选择取决于提示词和 **context**（上下文）。
   - 成员们表示很难知道路由到了哪些 LLM，唯一的办法是 *聚类基准测试性能（cluster benchmark performance）* 来估算性能。
- **使用与计费的陷阱**：Cursor 的新方案试图强迫用户使用 Auto 模型，并在过度使用时发出警告，减少剩余使用天数，迫使你使用糟糕的 Auto 模型。但一位成员表示 *只要我能通过延迟无限使用 Premium 模型，我就会一直用到极致*。
   - 成员们提醒 **Auto 模型** 使用的是 **Claude 4.5 Sonnet Thinking** 且收费更低，因此是省钱的好方法；但如果不想用，成员建议每隔 2 条提示词使用一次 **/summarize** 以降低 context。
- **Background Agents 完蛋了！**：用户抱怨 Background Agents 可能需要 *半天时间* 才能启动、git clone 仓库、扫描整个仓库、进行更改、lint 等等。
   - 成员们开玩笑说 *很可能是某个开发者用了 Background Agents 然后把它忘了*。
- **Parallel Agent，更像是 Paralyzed Agent（瘫痪代理）**：一位成员表示很难找到 Parallel Agent 的用例，另一位成员在容器中运行 `cursor-agent` 时启用 MCP 遇到问题。
   - 运行 `cursor-agent mcp login atlassian` 时提示：*✗ MCP login failed. Failed to load MCP 'atlassian': MCP server "atlassian" has not been approved. Please approve it before loading.*
- **Gemini 3.0 何时发布？**：随着 Gemini 3.0 发布在即，它可能会降低 Cursor 的价值。
   - 一些成员推测 *Gemini 无法击败 Sonnet 4.5 Thinking*。


  

---

### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1430708078936064150)** (5 条消息): 

> `调试 Cursor 的 UI 错误，处于评审模式的 Cursor PR，Background Agent API Key` 


- **UI 错误导致困惑**：一位用户建议更新 **UI 错误消息**，因为当本地分支未推送到 origin 时会引起混淆，建议使用更明确的消息，如 *"The branch {branch name} does not exist in the remote repository."*
   - 该用户花费了时间进行调试，因为他们以为 *environment.json* 没有被检入，并表示当时要是有一杯咖啡就好了。
- **Cursor 的远程分支逻辑受到质疑**：一位用户认为错误消息很清晰，并指出 **远程分支** 缺少 *environment.json* 文件意味着必须合并本地更改。
   - 该用户表示 *"Remote branch does not have an environment.json ... won't work unless you merge your local changes into it"*。
- **寻求 Ready-to-Review 状态的 PR**：一位用户询问是否可以将 **Cursor** 配置为直接以 *ready-to-review 模式* 而不是草稿模式打开 **Pull Requests**。
   - 未提供任何回答。
- **Background Agent API Key 的来源**：一位用户询问用于 **Background Agent 状态报告** 的 **API key** 的来源。
   - 未提供任何回答。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1430725021923410010)** (264 条消息🔥🔥): 

> `UnslothTrainer 参数，Unsloth 中的 LoRA，QAT 模型转换，GRPO 训练，Unsloth VLLM` 


- **针对过拟合调整 UnslothTrainer**：一位用户询问增加 `max_grad_norm` 是否能改善 [UnslothTrainer](https://github.com/unslothai/unsloth) 的过拟合问题，旨在记住训练数据并去除水印，初步尝试显示 loss 从 **2.0 降低到 0.2**。
   - 建议包括将 `weight_decay` 设置为 0，使用更高的学习率 (**5e-4**)，以及调整其他参数以促进记忆，例如减少或消除 warmup 并使用 constant 学习率调度。
- **在 Unsloth 中对 lm_head 应用 LoRA**：一位用户询问如何在 **Unsloth** 中对 `lm_head` 层应用 **LoRA**，并指出他们最初的尝试导致该层被包装为 `ModulesToSaveWrapper` 而不是注入 **LoRA** 模块。
   - 该用户寻求澄清最新的 **Unsloth** 版本是否支持 **lm_head LoRA**，以及是否有推荐的方法通过 **LoRA** 对新添加的 token 进行微调。
- **将 QAT 模型转换为 GGUF**：一位用户询问如何将 **QAT (Quantization Aware Training)** 模型转换为 **GGUF** 格式，以便在低功耗设备上使用。
   - 一名团队成员回应称，目前尚不支持 **QAT 到 GGUF** 的转换，但计划在年底实现，并强调了在低功耗设备上运行 **QAT GGUF** 模型的潜在优势。
- **依赖问题困扰 Unsloth 安装**：多位用户报告在安装 **Unsloth** 时遇到依赖冲突，特别是 `transformers` 和 `numpy` 版本。
   - 建议的解决方法包括特定的安装顺序（先 `pip install unsloth` 然后 `pip install transformers==4.57.1`）以及由于版本不兼容而采用 Docker 等替代安装方法。
- **NVIDIA 支持 Unsloth QAT**：Unsloth [宣布了 NVIDIA 对其 QAT 发布的支持](https://x.com/NVIDIAAIDev/status/1981510959048106262)。
   - 该推文收到了职位晋升尝试以及关于微调 qwen next 80b 的后续问题。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1430928966801424404)** (6 条消息): 

> `新成员自我介绍，LLM 训练兴趣` 


- **新成员加入社区**：几位新成员介绍了自己，兴趣范围从训练 **tiny models** 到实验现有模型。
   - 一名成员承认加入是因为他们 *"到处"* 都能看到 **Unsloth AI**。
- **LLM 训练引发兴趣**：一位名叫 Daryl 的新成员表示虽然 **没有 LLM 训练经验**，但有兴趣向社区学习。
   - 他表示这是 *"我感兴趣的事情，很乐意在志同道合的人身边学习"*。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1430696763047804949)** (54 条消息🔥): 

> `Karpathy 不用 Tmux，语音生成的 LLM 韵律 Token，人形机器人温度，小号刷 GPU 积分` 


- **Karpathy 坦白不再使用 Tmux 的诱惑**：Karpathy [在 X 上](https://x.com/karpathy/status/1980397031542989305)坦白，他几乎再次陷入了不使用 **tmux** 的诱惑中。
   - 他强调 **良好的 Token 初始化** 非常重要，尤其是在没有漫长教学阶段的情况下。
- **LLM 语音韵律 Token 技巧**：成员们讨论了在语音生成之前，使用特定的 **韵律 Token (prosody tokens)** 对 **LLM** 进行 **finetuning**，以控制语调。
   - 例如，使用类似 `<p3> Hello! </p3>` 的 Token 来表示生成语音中不同程度的重音和语调。
- **机器人专家讨论人形机器人的散热管理**：一位成员询问，考虑到 CPU/GPU 通常在 **60°C** 左右运行，将人形机器人的温度恒定维持在 **36.6°C** 是否会损坏内部硬件。
   - 另一位成员回答说，电子设备通常比生物系统更 **耐高温 (temperature-tolerant)**，并幽默地指出“肉体凡胎”在 42.5°C 时就会开始崩溃。
- **GPU 积分小号骚操作**：一位用户开玩笑说通过创建 **小号 (alt accounts)** 来获取额外的 **GPU 积分 (GPU credits)**。
   - 另一位用户回复了一个 [Chris Pratt 震惊的 GIF](https://tenor.com/view/chris-pratt-mind-blown-parks-and-rec-surprised-shookt-gif-9771169)，将其比作 MMO 游戏的基础操作。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1430678328297263176)** (72 条消息🔥🔥): 

> `Qwen3 微调问题，lm_head 层的 LoRA，Gemma 4b 模型训练不稳定，Docker 中的 llama-cpp 编译失败，微调 Qwen/Qwen3-VL-30B-A3B-Instruct` 


- **德语方言 Qwen3 模型遇到困难**：一位用户报告在微调 **Qwen3-4B-Instruct-2507** 模型将德语翻译成特定方言时遇到困难，面临模型丧失德语能力或陷入重复循环的问题。
   - 降低 **learning rate** 有助于减轻语言损坏，但模型在学习方言方面仍然吃力，这表明可能存在数据或训练配置问题。
- **在 lm_head 层使用 LoRA 让用户感到困惑**：一位尝试在 Unsloth 中对 **lm_head** 层应用 **LoRA** 的用户发现，它只是包装了该层而没有注入 LoRA。
   - 另一位成员建议 **lm_head LoRA** 是没有必要的。
- **Gemma 4b 模型出现故障**：一位用户在尝试修复代码以避免 Unsloth 的 **SFT** 训练期间出现重复后，遇到了 **Gemma 4b** 模型不稳定的问题。
   - 尽管重置了内核并使用了全新的模型和数据集，模型仍然产生不合理的回答，这表明可能存在持续的干扰或数据污染。
- **Docker Llama.cpp 出错**：一位用户报告说，由于缺少 CUDA 编译器 **nvcc**，**llama-cpp** 编译在官方 Unsloth Docker 镜像中失败。
   - 排查建议包括检查 Unsloth 版本、CUDA 设置，并确保正确安装了 NVIDIA container toolkit，且运行容器时带有 `--gpus all` 标志。
- **多模态模型微调受挫**：一位用户在使用针对 Vision 模型的 Unsloth notebook 微调 **Qwen/Qwen3-VL-30B-A3B-Instruct** 时遇到错误，具体发生在 `trainer.train()` 步骤。
   - 有建议称，通过设置 `os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"` 禁用编译可能会解决该问题。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1431050839975526471)** (2 条消息): 

> `AGI 定义，Cattel-Horn-Carroll 测试` 


- **发布 AGI 定义论文**：一位成员分享了 [AGI 定义论文](https://www.agidefinition.ai/paper.pdf) 的链接。
- **在 LLM 上应用 Cattel-Horn-Carroll 测试**：一位成员提到，将 **Cattel-Horn-Carroll 测试** 应用于 **LLM** 是 *非常聪明* 的做法。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1430682469912416407)** (287 条消息🔥🔥): 

> `OCR Models, DeepSpeed Zero 3 Configuration, Tiny LLM Training, Hugging Face Spaces Pricing, AI Prompt Standardization` 


- **Deepseek OCR 崭露头角**：成员们讨论了 [**Deepseek OCR**](https://example.com/deepseek-ocr) 是全新的，并建议在使用前先查看其评价。
- **Zero3 配置可能出问题了！**：一位成员在训练 **1b Gemma 3** 模型时遇到了 **GPU RAM** 爆炸的问题。该模型使用了 8k 的最大序列长度、r 为 8 的 **LoRA**、flash attention、deepspeed zero 3 以及 bf16 精度，batch size 为 1。
   - 另一位成员建议检查其 **Zero3 config** 是否有误。
- **为了正义的 Tiny LLM**：一位成员分享了一个从零开始构建的 **Tiny LLM** [GitHub repo](https://github.com/ker2x/MytinyLLM)，旨在作为学习练习。
   - 另一位成员指出，一个 **400k 参数的模型** 只是个玩具，开箱即用无法学到任何有意义的东西。
- **Hugging Face Spaces 现在要收费了？！**：一位成员注意到，一些 **Hugging Face Spaces**（包括 **Sora** 和 **Veo 3.1**）现在需要付费才能使用，消耗的是*你自己的*额度。
   - 他们链接到了一个 [Sora space](https://huggingface.co/spaces/akhaliq/sora-2) 的示例，其中包含一条警告，提示使用现在需要付费。
- **像伪代码一样标准化 Prompt**：一位成员想知道，将 **AI prompts** 标准化为比 JSON 更高效的**伪代码 (pseudocode)** 是否能提高 AI 性能。
   - 另一位成员分享了具有类似目的的 [GitHub repositories](https://huggingface.co/datasets/John6666/forum2/blob/main/ai_prompt_standardization_1.md)。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 条消息): 

waffles1: 啊是的，这完全是正经的
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1430893934212550756)** (2 条消息): 

> `RAG, raw-text-searcher, embedder_collection, pacific-prime, 6GB VRAM optimization` 


- **RAG 基础库更新**：一位成员分享了一个为 **RAG** 开发者准备的更新且优秀的库，见 [huggingface.co/kalle07/embedder_collection](https://huggingface.co/kalle07/embedder_collection) 和 [huggingface.co/Pacific-Prime/pacific-prime](https://huggingface.co/Pacific-Prime/pacific-prime)。
- **发布针对关键词的原始文本搜索器**：一位成员正在开发一个智能的 **raw-text-searcher**，用于搜索包含 **AND/OR, wildcard** 的关键词。它能像 embedder 那样截取匹配项周围的代码片段，但不基于相似度。他询问谁有兴趣参与测试？
- **针对 6GB VRAM 优化的 1.1B 模型**：一位成员报告了在挑战 **6GB VRAM** 极限时获得了 **10% 的提升**，现在能够从头开始运行一个 **1.1B** 模型并获得 **10% 的性能增益**。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/)** (1 条消息): 

its_nmt05: 我使用了原生 (vanilla) CLIP，只是切换到了通用 Prompt
  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1431018915139555471)** (3 条消息): 

> `NanoChat Course, Karpathy Educational Material, nanochat-students org` 


- **成员们思考 NanoChat 课程的可行性**：一位成员询问 **Hugging Face 上的 NanoChat 课程** 是否会有用。
   - 另一位成员对其目标表示困惑，并注意到 Discord 服务器中没有新频道。
- **社区在 Hub 上分享模型**：一位成员分享说，他们在 Hub 上建立了一个社区以便大家分享模型，并且正在努力将架构移植到 transformers，以便人们能更广泛地使用。
   - 该成员还提到，他们认为 **Karpathy** 将发布更多教学材料，因此关注这一点是有意义的。
- **澄清 nanochat-students 组织的目标**：一位成员询问另一位成员指的是 Karpathy 服务器还是 HF，以及 Hub 上 **NanoChat** 或 **nanochat-students org** 的目标。
   - 另一位成员表示，他认为大多数人都缺乏 GPU 资源 (GPU poor)，因此使用 **8xH100** 太奢侈了。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1430763200608993481)** (5 messages): 

> `HuggingFace Learn Agents Course, Quiz Errors, Qwen Model Issues` 


- **HuggingFace Agents 课程出现测验错误**：用户报告在 [HuggingFace Learn Agents Course](https://huggingface.co/learn/agents-course/unit2/smolagents/final_quiz) 的最终测验部分出现错误。
   - 错误信息显示在生成反馈时出现 **404 Client Error: Not Found**，特别是在使用 **Qwen/Qwen2.5-Coder-32B-Instruct** 模型时。
- **Unit 4 问题导致 404 错误**：一名用户报告在尝试下载 HuggingFace Learn Agents 课程中的 **Unit 4 问题** 时遇到 **404 错误**。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1430860721717248143)** (2 messages): 

> `ChatGPT Chrome Extension, Sora Roadmap Updates, Sora Character Cameos, Sora Video Editing, Sora Social Experience` 


- **ChatGPT 扩展提供即时回答**：OpenAI 发布了一个 [ChatGPT Chrome 扩展](https://video.twimg.com/amplify_video/1981054579648368640/vid/avc1/1280x720/YmADR8pumfl1Zwin.mp4)，允许 **ChatGPT** 查看你当前的页面，从而无需切换标签页即可提供即时、准确的回答。
- **Sora 路线图更新：Cameos、编辑、社交、Android**：**Sora** 正在添加 [角色 cameos](https://video.twimg.com/amplify_video/1981118365541310464/vid/avc1/896x512/hoABFvQ_q78wMp_h.mp4)，首批包括狗、豚鼠和喜爱的毛绒玩具，以及来自 **Sora** 视频生成的角色客串。
   - 他们正在更新生成 UI 以实时显示最新的热门 cameos，添加基础视频编辑功能（如拼接多个片段），探索与朋友一起使用 **Sora** 的新方式，提高 Feed 质量和审核，并正在开发 **Android** 版本的 **Sora**。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1430679139626647572)** (216 messages🔥🔥): 

> `Sora access, AI Education Software, Meta Glasses, AI Assisted calculators, Voice AI cloning` 


- **用户询问 Sora 生成权限**：成员们讨论了如何获取 **Sora AI** 的视频生成权限，提到了对 **text prompts** 的需求，并建议使用 **Comet** 或 **Atlas** 等工具来辅助流程。
- **关于 AI 辅助教育软件的讨论**：一名成员提到正在开发 **AI 辅助大学教育软件**，而其他人则讨论了此类创业公司的潜在陷阱和成功案例，包括 **LAUSD** 在疫情期间对一个 AI 项目的 500 万美元投资。
- **Meta 眼镜尽管面临障碍仍引发关注**：一些成员对 **新款 Meta 眼镜** 表示出兴趣，并指出由于需要预约而难以购买，一名成员还提到了在其国家隐蔽录音设备是非法的。
   - 讨论涉及了录音设备的合法性，提到了 **双方法律同意权 (two-party consent laws)**，以及在 **澳大利亚** 等地非法隐蔽录音可能面临的罚款和监禁。
- **AI 赋能 TI-84 计算器**：一名成员链接了一篇 **Wired** 文章和一段 [YouTube 视频](https://www.youtube.com/watch?v=olcZdTRdnQg)，展示了通过 **ESP32** 在 **TI-84 计算器**上运行 **GPT**，并认为通过 AI 破解计算器在数学考试中作弊应该获得学分和高分。
   - 该文章标题为 [ChatGPT on a TI-84 Graphing Calculator Is a Cheating Device](https://www.wired.com/story/chatgpt-on-a-ti-84-graphing-calculator-cheating-device/)。
- **语音克隆 DIY 项目**：一名成员分享了一个 [DIY 语音 AI 项目](https://www.instructables.com/Create-Your-Own-AI-Voice-Agent-Using-EchoKit-ESP32/)，允许用户使用 **EchoKit ESP32** 克隆声音，并讨论了在自定义声音上训练 AI 以进行视频生成。
   - 该项目是开源的。


  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1430713328703115425)** (26 条消息🔥): 

> `ChatGPT LightSession 扩展, Muon 作为 Adam 的替代品, OpenAI 支持问题, OpenAI 503 错误` 


- **ChatGPT LightSession 扩展分享**：一位用户分享了一个名为 **ChatGPT LightSession** 的[新型非官方 ChatGPT 扩展](https://www.reddit.com/r/ChatGPT/)，并指出该扩展仍处于 beta 阶段，可通过共享链接访问。
   - 另一位用户警告要*小心使用该扩展*，建议坚持使用公开可访问的内容，以保护个人信息安全。
- **LightSession 扩展修复红字错误**：一位用户报告称 **LightSession 扩展**修复了*红字错误*并提高了工作效率。
   - 他们表示，目前发现的最佳对话配置语句是*询问澄清性问题 (ask clarifying questions)*。
- **Muon 与 Adam 优化器**：一位用户询问 **Muon** 是否可以作为 **Adam** 优化器的潜在替代品。
   - 无人回应此问题。
- **OpenAI 支持失联，用户面临持续的 503 错误**：一位用户在过去 16 天里因持续出现 **503 错误**而无法发送或接收任何文本消息，且未收到 OpenAI 支持部门的任何回复。
   - 该用户是一名 **ChatGPT Plus 订阅者**，已尝试了多种故障排除步骤，正在寻求如何联系到 OpenAI 支持部门人工客服的建议。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1430738072815009852)** (14 条消息🔥): 

> `使用 Sora 创建监控摄像头 (CCTV) 画面, 优质的 ChatGPT 提示工程资源, LLM 的性能与提示差异, Gemini 的翻译词汇问题, 创建球体弹跳和下落的视频` 


- **使用 Sora 创建监控画面**：一位成员询问如何使用 **Sora** 创建**监控摄像头 (CCTV) 画面视频**。
   - 未提供解决方案。
- **寻求优质提示词入门指南**：一位成员询问是否有优质的 **ChatGPT** **提示工程 (prompt engineering)** 资源，以及是否每个 **LLM** 都有其独特的性能特征和提示方法。
   - 未提供解决方案。
- **Gemini 的语法错误**：一位成员报告称，**Gemini** 将视频脚本翻译成陌生语言时出现了脱离语境的词汇，并寻求如何通过提示 **ChatGPT** 以获得更好结果的建议。
   - 未提供解决方案。
- **为自定义创作打造吸睛作品**：一位成员分享了一个从自定义 GPT 作词人/音响工程师/专辑视觉生成器中提炼出的*疯狂的半复用提示词*，用于将音乐艺术家作为自定义指令/提示词投放，使其可用于音乐生成 AI。
   - 该成员附带了一个[文本文件](https://cdn.discordapp.com/attachments/1046317269069864970/1431083530695217233/drop_in_as_is_knowledge_initial_instruct.txt?ex=68fc1fe5&is=68face65&hm=5f027ddb6661f87ccf71a3d202e90d685e1ec64cb814fa2c5da0a6791c3725cc&)，旨在作为使用前需要*重新激活 (rehydrated)* 的知识库。
- **精准定位提示参数**：一位成员询问如何查找构建精准提示词所需的信息，特别是针对灯光和摄影等超出其专业领域的领域。
   - 另一位成员建议将示例图像展示给 **ChatGPT**，并要求其生成一张几乎相同的图像，同时清晰地描述图像，重点关注感兴趣的区域。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1430738072815009852)** (14 条消息🔥): 

> `Sora 用于 CCTV，Prompt engineering 来源，LLM 翻译，图像提示词，用于提示词的 GPTs` 


- **Sora 的 CCTV 首秀遇冷**：一位成员询问如何使用 **Sora** 创建 **CCTV 监控录像视频**，但未获得即时回应或解决方案。
   - Sora 能否很好地模拟监控摄像头那种颗粒感、低保真度的美学效果仍有待观察。
- **Prompt 工程师寻找“圣杯”**：一位用户请求推荐 **ChatGPT prompt engineering** 的优质资源，并询问每个 **LLM** 是否具有独特的性能特征，从而需要定制化的 prompt。
   - 社区并未立即提供资源，凸显了对有效 prompt engineering 策略的持续探索。
- **Gemini 的语言障碍**：一位成员分享了使用 **Gemini** 翻译视频脚本的经验，指出尽管 Google 拥有翻译专长，但词汇有时缺乏上下文。
   - 他们寻求关于改进 **ChatGPT** prompt 的建议，以获得更好的翻译准确性。
- **图像分析启发精准 Prompt**：一位成员寻求关于创建精准图像生成 prompt 的建议，询问“一个人究竟该如何寻找信息来构建一个真正精准的 prompt？”。
   - 该成员附上了一张示例图片，寻求关于如何将视觉元素转化为详细 prompt 指令的指导。
- **GPTs 释放 Prompt 威力**：一位成员建议开发个人 **GPTs** 来处理特定的 prompt 请求，认为专门化的 GPTs 比通用的 GPTs 能更好地聚焦细节。
   - 该策略旨在利用自定义 GPTs 的专业化，在 prompt 生成中产生更具针对性和精炼的结果。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1430874121679405068)** (214 条消息🔥🔥): 

> `Gemini 与结构化输出，DSPy Refine 逻辑，异步 ReAct 模块，DSPy token 使用与成本追踪，DSPy 定制化痛点` 


- **Gemini 无法处理 DSPy 中的结构化输出**：一位用户报告称，在 DSPy 中带有 `responses` 参数的 Gemini 模型在使用结构化输出适配器时会抛出警告和错误，具体表现为“找不到存根文件 (Stub file not found)”错误。
   - 目前正在讨论为 **Gemini 模型** 启用结构化输出，以确保在 **DSPy** 中使用 Python 时的类型安全。
- **Refine 在达到 N 之前停止**：一位用户发现 `dspy.Refine` 并不一定会运行所有 N 次迭代，如果 `reward_fn` 超过了设定的 `threshold`（阈值），它就会提前停止，这与 [文档](https://dspy.ai/tutorials/output_refinement/best-of-n-and-refine/?h=refine#refine) 中的初始假设相反。
   - 用户提供了一个 [demo](https://github.com/stanfordnlp/dspy/blob/main/dspy/predict/refine.py#L142)，说明如果 `reward_fn` 超过阈值，refine 循环就会中断，并指出 `Refine` 每次都会对模块进行深拷贝，这阻碍了对迭代次数的准确计数。
- **DSPy 的异步 ReAct 以同步方式运行**：一位用户在使用 `dspy.asyncify` 异步运行两个 `ReAct` 模块时遇到问题，注意到它们似乎是同步执行的，并寻求正确实现的指导。
   - 一位成员建议使用 `await program.acall` 代替 `await program.aforward`，并在模块内实现 `async def aforward(...)`，同时指出 DSPy 的文档令人困惑。
- **DSPy 的 token 使用情况追踪成本**：用户正在讨论如何在 DSPy 中追踪 token 使用情况和成本，一位成员分享了一个 [代码片段](https://discord.com/channels/1161519468141355160/1161519469319946286/1431029715511232512)，利用 `program.history` 和 **LiteLLM** 模型定价来计算成本。
   - 他们澄清说，`if "usage"` 条件确保了成本计算是基于实际模型使用量的（考虑到了缓存），该成员还指出 `program.history` 中的 `usage` 与 `result.get_lm_usage()` 类似。
- **用户在 DSPy 定制化方面挣扎**：一位用户对 DSPy 的复杂性表示沮丧，特别是关于访问 `ReAct` 模块输出的问题，建议在自定义循环中实现 LLM 调用，以便更好地控制和进行 UI 集成。
   - 用户探索了替代方法，例如在 `ReAct` 模块中对 `aforward` 方法进行子类化和重新实现，如 [此代码片段](https://cdn.discordapp.com/attachments/1431066467897577492/1431067652763156666/message.txt?ex=68fc111c&is=68fabf9c&hm=956afd7b5a72fb6ea1288e1d2656952b4e64d31baf63e1feccda13f151437ba5&) 所示。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1430752336048816139)** (181 messages🔥🔥): 

> `Mojo 与 Rust 的互操作性，CPython 运行时与 Mojo 的交互，在 Mojo 中使用 PyO3 和 JuliaCall，Mojo 中的可扩展 Traits，Mojo 中的 Effect 系统和逻辑编程` 


- ****Mojo** ❤️ **Rust**: 以 C ABI 作为共同基础**: Mojo 作为一种系统语言，可以通过 **C ABI** 与 Rust、C++、Zig 或任何支持 **C ABI** 的语言进行通信，类似于宿主 C/C++/Rust 进程嵌入 Python 解释器的方式。
   - 然而，使用 **C ABI** 需要过多的手动干预，这促使人们寻找能让软件包像原生平台一样可用的解决方案。
- ****Python 力量**: Mojo 的运行时秘密 🤫**: Mojo 与 **CPython** 的交互方式类似于宿主 C/C++/Rust 进程嵌入 Python 解释器，使其能够利用 Python 生态系统，但这需要运行时方面的努力以及表达宿主语言概念的能力。
   - 目标是尽可能实现**零运行时开销 (zero runtime overhead)**，特别是在证明加速热循环 (hot loop) 代码的可行性时。
- ****PyO3** 和 **JuliaCall**: 互操作性的梦之队 ✨**: **PyO3** 和 **JuliaCall** 可以从基于 Mojo 构建的语言中使用，以访问 Rust 和 Julia 生态系统，因为它们创建了 Python 模块，但由于需要通过 Python 进行往返转换，这种方法可能会引入性能损失。
   - 有人建议直接让 **Julia 库导出 C ABI**，而不是通过 Python 往返，但这必须针对每个项目单独完成，且由于丢失了类型安全性，安全性较低。
- ****类型系统对话**: Mojo 的实力展示 💪**: Mojo 的类型系统非常强大，Dependent Haskell 可能是最流行的能表达 Mojo 所能表达的一切的语言，而拥有更强大类型系统的语言可以消费来自类型系统较弱语言的代码。
   - 为了让从另一种语言调用 Mojo 函数感觉像原生调用一样，另一种语言需要一个比 Mojo 更强大或同等强大的类型系统，可能包括依赖类型 (dependent types) 和线性类型 (linear types)。
- ****互操作洞察**: 动态与静态的权衡 ⚖️**: 使用编译型语言意味着既有一定的动态性，又能让 Mojo 代码从你的代码中获得类型系统保证，这大大减少了类型内省 (type introspection)。
   - 如果宿主语言是静态且编译的，那么在其之上实现的新语言也应该是静态且编译的，以最小化开销，因为如果动态语言更具表现力，静态编译语言可以编译为发出动态代码。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1430827604964933672)** (8 messages🔥): 

> `Mojo 管道 (pipelines)，LeetGPU，GPU Puzzles` 


- **Mojo 暂缺管道操作符**: 一位成员询问 Mojo 是否拥有像其他语言中的 `|>` 或 `>>` 那样的管道字符或函数组合。
   - 另一位成员回答说 Python 缺乏这一特性，且由于 Mojo 尚未稳定，团队可能短期内不会考虑它，但指出了 [这个 GitHub issue](https://github.com/modular/modular/issues/213) 作为相关的特性请求。
- **LeetGPU 支持 Mojo**: 一位成员分享了 [LeetGPU](https://leetgpu.com/) 的链接，并指出它支持 Mojo。
   - 一位 Modular 员工回应称，Modular 认为开发与 Modular 平台一致的培训非常重要，并建议将 [GPU Puzzles](https://puzzles.modular.com/introduction.html) 作为替代方案。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/)** (1 messages): 

brockelmore: https://github.com/modular/modular/issues/5496
  

---

### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1430682766122680442)** (141 条消息🔥🔥): 

> `Claude Haiku 4.5 网页搜索, Cursor 中的 OpenRouter 模型, Exacto 评估, NovitaAI 和 GPT-OSS-120B, DeepSeek v3.2 exp 时刻` 


- **DeepSeek 因余额问题出现状况**：**DeepSeek/deepseek-v3.2-exp** 由于 OpenRouter 额度耗尽而出现问题，导致 **402 Proxy Error**。
   - 用户误以为该错误是特定于账号的问题，而其他人指出该错误是针对用户的，暗示他们的账号出了问题。
- **充值到账延迟引发困扰**：用户报告了 OpenRouter 额度充值延迟以及 **DeepSeek** 提供商余额耗尽的问题。
   - 一位用户幽默地提到，延迟会持续到 *Toven 看到它为止*。
- **OpenRouter 充值手续费详情**：用户讨论了在 OpenRouter 充值相关的服务费，支付 5.83 美元仅获得 5 美元的额度。
   - 据澄清，额外的 0.83 美元用于支付服务费，大约为 **80 美分 (US)**。
- **Exacto 模型变体：昂贵的难题？**：用户讨论了将 **exacto** 模型作为 API 中独立变体的实现方式，并对定价以及 Token/应用统计数据的拆分表示担忧。
   - 尽管有人担心 OpenRouter 会收取更高费用，但对 **glm-4.6** 的检查显示 **exacto** 和 **non-exacto** 端点的价格相同。不过，一位成员认为，将它们作为模型变体而非额外参数来实现，会*拆分这些模型的 Token 和应用统计数据*。
- **OpenRouter API 访问问题**：用户报告了登录 OpenRouter 以及在 completions 端点收到 **401 unauthorized error** 的问题。
   - 有人建议该问题可能与特定工具有关，cURL 请求可以正常工作，而其他工具则失败。


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 条消息): 

Readybot.io: **OpenRouter - 新模型**
  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1430702602978463794)** (6 条消息): 

> `模型统计数据可见性, OpenAI 数据保留, Krea AI 的视频模型, OpenRouter 文档` 


- **模型统计数据可见性诉求**：成员请求公开显示模型统计数据，或通过徽章标示 **"exacto quality"**。
   - 工作人员确认该功能“即将推出”。
- **OpenAI 不再需要保留所有 ChatGPT 数据**：一位成员分享了一篇 [Engadget 文章](https://www.engadget.com/ai/openai-no-longer-has-to-preserve-all-of-its-chatgpt-data-with-some-exceptions-192422093.html)，指出 **OpenAI** 不再需要保留其所有的 **ChatGPT** 数据（存在部分例外）。
- **Krea AI 的 14B 视频模型发布**：一位成员分享了 [Krea AI 的推文](https://x.com/krea_ai/status/1980358158376988747)链接，内容关于其 **14B 视频模型**。
- **OpenRouter 文档状态？**：一位成员报告 **OpenRouter docs** 仍然无法访问。
   - 工作人员回复称他们正在“调查中”。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1430680098100609114)** (115 条消息🔥🔥): 

> `OpenClip TS 实现, Linear 的自动驾驶 SaaS, 阿里巴巴 Qwen 对阵 OpenAI, MythWorx 的 ARC-AGI 声明, SOTA RL 算法` 


- **TypeScript 版 OpenClip 实现被发现！**: 一位成员正在寻找 **OpenClip** 的 TypeScript 实现，并在 [Marqo's fashionCLIP](https://huggingface.co/Marqo/marqo-fashionCLIP) 找到了自己的解决方案。
   - 他们分享了代码片段，展示了如何使用 `@huggingface/transformers` 中的 `CLIPTextModelWithProjection`、`CLIPVisionModelWithProjection`、`AutoTokenizer` 和 `AutoProcessor` 来计算文本与图像之间的相似度得分。
- **Linear 的愿景：自动驾驶 SaaS 掌舵！**: Linear 正在预告从反应式聊天机器人 UI 向**主动式 AI (proactive AI)** 的转变，后者能自动推进工作，并称之为 *“自动驾驶 SaaS”*。
   - 关注者赞扬了这种大胆的构想，并将其比作 **Level-5 自动驾驶**，即软件完成工作，而用户只需坐享其成。
- **Qwen 的魅力攻势：Airbnb 选择更便宜的 LLM！**: **Airbnb CEO** 公开赞扬**阿里巴巴的开源 Qwen 模型**比 **OpenAI 的最新模型**更快且便宜得多。
   - 评论者强调了 **Cerebras 的 2,500 tokens/秒推理速度**、多语言质量，以及在边际准确率提升与成本之间的“逐底竞争”，并指出安全、主权和未来的硬件转变是潜在的担忧。
- **ARC-AGI 质疑：MythWorx 大胆声称达到 100%！**: **ARC Prize** 组织者表示，他们尚未验证 **MythWorx** 在一份 **1 亿美元**融资新闻稿中声称其模型在 **ARC-AGI** 上达到 **100%** 的说法。
   - **Greg Kamradt** 表示如果 MythWorx 正确提交，他愿意运行官方测试；而评论者则因缺乏验证、团队透明度极低以及与 **Theranos 级别的炒作**相似而质疑该声明的真实性。
- **Anthropic 的算力征服：2026 年前部署 100 万个 TPU！**: Anthropic 宣布了一项数十亿美元的交易，计划到 **2026 年**增加 **约 100 万个 Google Cloud TPU** 和 **超过 1 GW** 的容量。
   - 社区对算力规模反应热烈，开起了“建立帝国”的玩笑，并询问这是否会转化为更低的 API 价格、更高的使用限制或更长的上下文窗口；该交易价值数百亿美元，预计将在 2026 年上线超过 1 吉瓦的 AI 算力容量 ([CNBC 文章](https://www.cnbc.com/2025/10/23/anthropic-google-cloud-deal-tpu.html), [Anthropic 公告](https://www.anthropic.com/news/expanding-our-use-of-google-cloud-tpus-and-services))。


  

---


### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1430727704147136524)** (7 条消息): 

> `AMD Strix Halo ROCm, ROCm 上的 GPT-OSS, 用于多页扫描 PDF 问答的本地 AI 应用` 


- ****苏姿丰 (Lisa Su)** 签名版 **Strix Halo**！**: 一位用户分享了一个[链接](https://xcancel.com/AnushElangovan/status/1981031660209770802)，展示了 **Anush Elangovan** 炫耀的一台有**苏姿丰**签名的 **Strix Halo (Ryzen AI Max+ PRO 395)** 笔记本电脑。
   - 该笔记本正在 **ROCm** 上本地运行 **GPT-OSS**，引发用户分享基准测试结果、错误报告、购买建议以及对更高内存/下一代配件的愿望清单。
- **寻找支持扫描 PDF 问答的本地 AI**: 一位用户询问是否有支持使用 **VLM**（如 **Qwen3-VL-4B**）直接对**多页扫描 PDF 进行问答**的本地 AI 应用。
   - 他们指出，大多数应用要么只支持图像，要么在上传文件时执行 **RAG**，例如 **LM Studio**。


  

---

### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1430702568006357062)** (15 messages🔥): 

> `Sora 路线图更新，“教沙子思考” - Runtime 2025，LTX-2 发布，Character AI Ovi` 


- **Peebles 预告 Sora 进展**：来自 **OpenAI** 的 Bill Peebles 概述了即将推出的 **Sora** 应用更新，包括将在几天内上线的[自定义角色客串 (custom character cameos)](https://xcancel.com/billpeeb/status/1981118483607032050)、基础剪辑拼接编辑器、即将推出的社交频道，以及期待已久的 **Android** 版本发布。
   - 用户请求了客串搜索、分镜保存、更长的片段、英国/全球访问权限以及更好的审核反馈。
- **a16z 的抽象 AI 广告惊艳全场**：**a16z** 发布了 ["Runtime 2025"](https://xcancel.com/a16z/status/1981074692443427321?s=46) 的预告片，其中包含台词 *"教沙子思考，或者被时代抛弃。"*
   - 这段 **A24 风格的视频**在加密货币、AI 和硬件社区引发了热议，观众称赞其电影质感，同时也质疑这是远见卓识的宣传还是空洞的包装。
- **Lightricks 发布本地潜空间模型 LTX-2**：**Lightricks** 推出了 **LTX-2**，这是一个开源创意引擎，可在消费级 GPU（优化后 RTX 5 系列表现真实）上生成长达 ~10–15 秒的[同步 **4K/50 fps** 视频及音频](https://xcancel.com/ltx_model/status/1981346235194683497)。
   - 社区讨论集中在专业级 AI 视频工具的民主化和即将推出的本地安装，权重将于今年晚些时候发布，API Playground 现已上线。
- **Character AI 开源 Ovi 被忽视了？**：Character AI 在 [**GitHub 上开源了 Ovi**](https://github.com/character-ai/Ovi)。
   - 一些社区成员好奇为什么它“一点水花都没有”。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1430688885809090672)** (21 messages🔥): 

> `FA4 的正确性优化，确定性推理博客，图形编程 Discord 服务器，非 ML 工作负载的 Kernel 编程 DSL，用于 HPC 的 Mojo` 


- **寻求数值稳定性资源**：一位成员在阅读了关于 **FA4 的正确性优化**和**确定性推理博客**后，寻求数值稳定性方面的资源，并被推荐了一本相关书籍：[《随机微分方程的数值解》(Numerical Solution of Stochastic Differential Equations)](https://epubs.siam.org/doi/book/10.1137/1.9781611971491)。
- **讨论 GPGPU 之外的图形编程**：一位成员询问了关于图形编程的资源，特别是使用 **OpenGL** 或 **Vulkan** 进行科学可视化的资源，而非专注于 **GPGPU** 的内容。
   - 另一位成员推荐了拥有 2 万名成员的 [Graphics Programming Discord 服务器](https://graphics-programming.org/)，该服务器以在 **OpenGL** 和 **Vulkan** 方面的专业知识而闻名。
- **用于非 ML 工作负载的 DSL Kernel**：一位成员询问了 **Kernel 编程 DSL**（TileLang, Triton, Gluon, TLX, Helion）在非 ML 工作负载中的应用，建议将其扩展到**稀疏性 (sparsity)**、**模板 (stencils)** 和**分块硬件架构 (tiled hardware architectures)**。
   - 另一位成员引用了关于 HPC 的 [Mojo](https://github.com/tdehoff/Mojo-workloads) 工作负载和 [Triton](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=w2kMs8IAAAAJ&citation_for_view=w2kMs8IAAAAJ:u-x6o8ySG0sC) 论文。
- **GPU 上的 Mojo HPC Kernel**：一位成员分享了一篇论文和 GitHub 仓库的链接，内容关于使用 [Mojo](https://arxiv.org/abs/2509.21039) 和 [Mojo Workloads](https://github.com/tdehoff/Mojo-workloads) 构建 **基于 MLIR 的性能可移植 GPU HPC 科学 Kernel**。
   - 该论文针对四种科学工作负载：七点模板 (seven-point stencil)、BabelStream、miniBUDE 和 Hartree-Fock，并将其在 **NVIDIA H100** 和 **AMD MI300A GPU** 上的性能与厂商基准进行了对比。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1430758919269257246)** (18 条消息🔥): 

> `wgmma template, smem descriptors, Pinned portable memory, PTX memory consistency model, Compiler issued barriers` 


- **WGMMA 的 Warpgroup 困扰**：一位用户在处理 `wgmma` 模板和编译器注入的屏障（barriers）时遇到问题，即使在通过 `warpgroup_arrive` 函数使用了 `wgmma.fence.sync.aligned` 之后也是如此。
   - 该用户正在将 `wgmma` 指令与共享内存描述符（shared memory descriptors）配合使用，并怀疑问题出在 PTX 内部不正确的 fence 或内存一致性上；[提供了代码片段](https://cdn.discordapp.com/attachments/1189607726595194971/1430858333014986862/image.png?ex=68fbf6ea&is=68faa56a&hm=c2d221fd9b6ecca17e72af6f77398d4cb88ef8c8049c472ee765ed5250b771e4)。
- **SMEM 描述符解析**：一位用户提供了他们自定义的共享内存（SMEM）描述符实现，用于 WGMMA 指令，包括 `matrix_descriptor_encode` 和 `make_smem_descriptor` 函数。
   - 该实现源自 [PTX 指南](https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-descriptor-format)，将地址和内存布局信息编码进一个 64 位的描述符中。
- **DeepSeek GEMM 深度解析**：一位社区成员建议参考 DeepSeek-AI 的 DeepGEMM [实现](https://github.com/deepseek-ai/DeepGEMM/blob/c9f8b34dcdacc20aa746b786f983492c51072870/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh#L275)，以处理类似的内存屏障（memory fences）和 WGMMA 问题。
   - 此外还提供了指向 NVIDIA CUTLASS [库](https://github.com/NVIDIA/cutlass/blob/b2ca083d2bb96c41d9b3c5a930637c641f6669bf/include/cute/atom/mma_traits_sm90_gmma.hpp#L49)、[mma_sm90_gmma.hpp](https://github.com/NVIDIA/cutlass/blob/b2ca083d2bb96c41d9b3c5a930637c641f6669bf/include/cute/arch/mma_sm90_gmma.hpp#L88) 以及[讨论区](https://github.com/NVIDIA/cutlass/discussions/1375)的链接。
- **FP8 精度累加？**：一位用户询问了 `m16n8k32 fp8xfp8` 操作的累加精度，想知道它是否在转换回 FP16 之前先以 FP32 进行累加。
   - 这个问题是受此 [文档链接](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma) 的启发，其中提到中间值的累加至少以单精度（single precision）执行。


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1430838297424298005)** (2 条消息): 

> `Pruna.ai Hiring, Vectorware Hiring, Rust GPU software` 


- **Pruna.ai 招聘应用机器学习工程师 (Applied ML Engineer)**：[Pruna.ai](https://careers.pruna.ai/jobs/6569302-applied-ml-engineer) 正在招聘一名具有 **Diffusion Models** 优化经验的 **Applied ML Engineer**；该职位为远程办公，地点在巴黎或慕尼黑。
- **Vectorware 开启招聘**：一家使用 **Rust** 构建高级 GPU 软件的公司 [Vectorware](https://www.vectorware.com/jobs) 正在招聘，应聘者预计需要学习 Rust。
   - 有关该公司的更多详情可在[这篇博文](https://www.vectorware.com/blog/announcing-vectorware/)中找到。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1430745347440906274)** (1 条消息): 

> `Cloud vs Local, Cost Analysis, Serious Work Consideration` 


- **试运行阶段云端优于本地**：一位成员建议先从*云端租赁*，直到你投入了至少几十个小时进行严肃的工作，以便评估成本和收益。
- **本地回本需要半年时间**：他们估计投资本地硬件的*盈亏平衡点*至少需要*半年的云端租赁时间*。
- **不要忘记管理和电力成本**：在评估云端与本地的权衡时，成本分析应考虑*管理痛苦和电力成本*。


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1431050313552494713)** (2 条消息): 

> `HQQ+ blog post, mobiusml github down, dropbox github` 


- **HQQ+ 博客文章链接谜题解开！**：一位成员在发现 [https://mobiusml.github.io/1bit_blog/](https://mobiusml.github.io/1bit_blog/) 失效后，正在寻找 **HQQ+ 博客文章** 的有效链接。
   - 另一位成员指出，由于今天宣布的一项最新变动，博客文章和 GitHub 链接中的 *`mobiusml` 都应替换为 `dropbox`*。
- **MobiusML 迁移至 Dropbox**：**MobiusML** 域名今天正在迁移至 **Dropbox**，导致原始链接失效。
   - 用户现在应将博客文章和 GitHub 仓库链接中的 `mobiusml` 替换为 `dropbox`。


  

---

### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1430954293346697258)** (4 条消息): 

> `Mobius Labs 收购，Austin Huang 新闻` 


- **Adept 策略转变及联合创始人加入 Amazon**：[Mobius Labs](https://x.com/Mobius_Labs/status/1981391562836721786) 的联合创始人分享了 *个人消息*，表示 *这是一段美好的旅程*。
   - 一位成员向他们表示祝贺，希望他们得到了优待，并表示 *你们做得非常出色*。
- **Austin Huang 宣布个人消息**：[Austin Huang](https://x.com/austinvhuang/status/1981393212003521017) 宣布了 *一些个人消息*，并附上了 X 的链接。
   - 他还分享了一张 **电烤架上的三文鱼** 照片，以及番茄、黄瓜、海盐、咖啡、奶霜和甜菊糖。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 条消息): 

melnimr: 我给你发私信了
  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1430995659586142319)** (4 条消息): 

> `Lunar Lake, MAMF 脚本, roofline, vk_cooperative_matrix_perf, CUDA events` 


- **Lunar Lake 草图 Roofline 首次亮相**：一位成员分享了使用 [Stas 的 MAMF 脚本](https://github.com/stas00/ml-engineering/tree/master/compute/accelerator/benchmarks) 中的数据点为 PyTorch 2.9.0+xpu 环境下的 **Lunar Lake** 创建的 **roofline** 草图。
   - 他们注意到在小尺寸下存在恒定的开销，这使得它们无法代表 GPU 的能力，但强调了测试的一致性。
- **测量短 Kernel 的时间**：一位成员建议通过在不相交的输入/输出上运行多个 kernel 以使 L2 缓存失效，从而改进短 kernel 的时间测量。
   - 还建议如果使用 **CUDA event**（或 XPU 等效项），可以在 `event.record()` 之前添加一个小型的 matmul 以减少开销，并链接了一篇关于 [PyTorch 中计时操作的 Speechmatics 文章](https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch)。
- **vk_cooperative_matrix_perf 已修复**：**vk_cooperative_matrix_perf** 已通过补丁得到改进。
   - 未提供更多细节。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1430981414530318448)** (6 条消息): 

> `Vectorware, Rust 编写的 GPU 软件, GPU 上的 ML, GPU 渲染` 


- **Vectorware 进军 Rust 编写的 GPU 软件**：一家名为 [Vectorware](https://www.vectorware.com/blog/announcing-vectorware/) 的新公司成立了，专注于使用 **Rust** 为 GPU 编写高级软件。
   - 该公司明确表示目前尚未建立 Discord 服务器以保持专注，但最初将重点放在 **ML 应用**上，并有一些 **在 GPU 上运行的基于 CPU 的用例**。
- **ML 成为 GPU 的核心**：Vectorware 将 **机器学习 (ML)** 应用作为其 **基于 GPU 的软件开发** 的初始重点。
   - 在接下来的几周内，Vectorware 计划展示利用 GPU 处理基于 CPU 用例的演示，暗示了超越 ML 的更广泛长期愿景。
- **GPU 渲染成为焦点**：一位成员对像 **Zed** 这样严重依赖 **GPU** 进行渲染的应用表示好奇，Vectorware 可能会解决这类问题。
   - 成员们推测，还有其他 **非渲染**、**非 ML 应用** 将大为受益。


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1430950569538883696)** (2 条消息): 

> `针对 sm_80 的 PGL 支持, 4090/L4s 的单 GPU kernel` 


- **Ampere GPU 逐步退出开发阶段**：有人提出了关于未来 **针对 sm_80 (Ampere 架构) 的 PGL 支持** 的问题。
   - 一位成员回答说他们 *不再拥有 Ampere GPU*，这可能意味着不再支持。
- **4090/L4s 的单 GPU kernel 获得 PR**：一位成员正在处理一个 **PR，旨在解除旧全局布局下 4090/L4s 的单 GPU kernel 阻塞**。
   - 由于新的必需命名空间加上此问题，编译已损坏，但这同样适用于 A100，因为 PGL 现在仅针对 Hopper/Blackwell 进行测试。


  

---

### **GPU MODE ▷ #[ppc](https://discord.com/channels/1189498204333543425/1343373905783554048/1430917059931537488)** (1 messages): 

> `Code Snippets Policy, Shared Memory Speed` 


- **不鼓励发送冗长的代码片段**：一位成员要求其他人避免发布较长的代码片段，而是分享其在 [ppc system 中的提交链接](https://ppc.system)。
   - 目标是不向其他人剧透。
- **Shared Memory 并不总是高性能的**：一位成员建议加载地址应按 `(something) + threadIdx.x` 的方式计算。
   - 他们补充说，虽然 **Shared Memory 很快**，但确保 **registers** 中的数据得到良好重用非常重要，尤其是当数值仅被加载和使用一次时；这甚至可能比使用 Shared Memory 还要重要。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1431020837695262821)** (3 messages): 

> `Thread Fragment Visualization, Code Formatting in Discord` 


- **Thread Fragment 可视化问题**：一位用户询问他们可视化具有给定 **shape** 和 **stride** 的 **thread fragment** 布局的方法是否正确，并指出它打印了 128 个元素，重复索引在 **0** 到 **79** 之间。
   - 该用户提供了使用 `Shape` 和 `Stride` 模板的代码，通过迭代 128 个元素并使用 `crd2idx` 打印计算出的索引。
- **Discord 代码格式化帮助**：一位用户建议重新格式化另一位用户提供的代码，建议使用三反引号（`````）以在 Discord 中正确格式化代码。
   - 需要正确的格式化以避免 Discord 复制消息。


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1430849056074371144)** (4 messages): 

> `Torch/XLA new direction, Lazy Tensor Paper relevance, eDSLs frontends and embeddedness` 


- **Torch/XLA 转向新方向**：尽管 [torch/xla](https://github.com/pytorch/xla/issues/9684) 有了“新方向”，但 **Lazy Tensor Paper** 对于 picograd 在将 eager mode 强行塞入 tinygrad 以降低门槛方面仍然具有参考价值。
   - [Lazy Tensor Paper](https://arxiv.org/pdf/2102.13267) 探讨了使用 picograd 将 eager mode 集成到 tinygrad 中。
- **eDSLs 嵌入深度**：人们意识到 eDSLs 具有“非平凡的前端（non-trivial frontends）”，这取决于它们与宿主语言的嵌入深度，并以 [picograd <- tinygrad](https://pytorch.org/assets/pytorch2-2.pdf) 作为案例研究。
   - 以前认为 eDSLs 能够避开解析（parsing），但这取决于 eDSL 是“浅嵌入（shallow embedding）”还是“深嵌入（deep embedding）”。


  

---


### **GPU MODE ▷ #[irl-accel-hackathon](https://discord.com/channels/1189498204333543425/1416087968933740688/1430677295458287881)** (12 messages🔥): 

> `Blackwell B200, H100 availability, Nebius Offerings, PyTorch Open Source Week in SF, IRL Event streaming` 


- **Blackwell B200 单点供应**：参与者到达后可能通过扫描 **QR code** 领取单块 **Blackwell B200 GPU**。
   - *如果你有钱，欢迎自带算力*。
- **Nebius 可能没有 H100 库存**：一位成员询问了 Nebius 是否提供 **H100 GPU**。
   - 虽然没有直接回答，但该询问暗示 **H100** 可能不是该活动的主要供应产品。
- **Hackathon 已报满，不再提供门票**：线下 accel-hackathon 活动的预订量超额约 **6 倍**，这意味着门票已完全售罄。
   - 一位目前在 **SF** 参加 **PyTorch/Open Source Week** 的成员希望能参加，但目前在候补名单上，并询问活动的任何部分是否会在线直播。


  

---


### **GPU MODE ▷ #[cluster-management](https://discord.com/channels/1189498204333543425/1420098114076803142/1430810943251808366)** (2 messages): 

> `Replica Based Recovery, Automated Fault Detection, AI-Based Fault Handling` 


- **节点获得冗余副本**：根据节点数量和预算，运行主动冗余副本可以缩短基于副本恢复的停机时间。
   - 如果无法实现冗余，**checkpointing** 是现实的恢复策略。
- **讨论自动化恢复与编排**：正在为更长时间的训练任务开发自动化恢复和编排功能，通过一个层来自动检测和恢复跨节点、GPU 和 **interconnects** 的故障。
   - 该系统接入现有技术栈，增加了可观测性和 **AI-based fault handling**，以防止因硬件问题导致训练任务中断或时间损失。


  

---

### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1430792657747312702)** (7 messages): 

> `Phi nodes, cudagraphs, tilelang, Helion compiler improvements` 


- **为了编译器简洁性选择 Phi 节点**：在 Device IR 中选择 **phi nodes** 而非块参数（block arguments）是出于 Helion 编译器实现简洁性的考虑，因为它们主要保留了用户的控制流。
   - Phi 节点变成了类似于“在输出中将这些分配给同一个变量”的操作，这使得输出代码在控制流方面看起来与输入代码非常相似。
- **Helion Kernel 支持 CUDA Graphs**：**CUDA graphs** 在 Helion kernel 中得到支持，受限于与其他语言相同的约束，从而允许使用 `torch.cuda.make_graphed_callables`。
   - 只要你不在 kernel 中执行任何不支持 **cudagraph** 的操作，通常就没有问题。
- **TileLang 改进 Mamba-2 基准测试**：[TileLang](https://github.com/tile-ai/tilelang/blob/main/benchmark/mamba2/README.md) 最近更新了其在 **mamba-2-chunk-scan** 上的基准测试，展示了性能提升。
   - 这些对其编译器的改进是为了回应已发布的性能数据，可能会影响与 Helion 和 Triton 的对比。
- **Helion 编译器迎来改进**：TileLang 针对已发布的性能数据对其编译器进行了一些改进。
   - 成员指出，他们可能只是调整了其 **kernel hyperparams**（Kernel 超参数）。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1430690040400314388)** (19 messages🔥): 

> `Chat datasets with system messages, AI network like the internet, Discord 'pending' member status` 


- **用户寻找带有系统消息的聊天数据集**：一位成员询问是否有包含 **system messages** 的聊天数据集推荐，并发现 [Open Orca](https://huggingface.co/datasets/OpenOrca/OpenOrca) 非常有效。
   - 该成员正在寻找更新的数据集；另一位用户随后请求提供链接。
- **构思 AI “互联网”**：一位成员提出了一个类似于现代互联网的 **AI 网络**想法，成千上万个模型通过特定协议进行通信以回答问题。
   - 另一位成员询问了这可能呈现的具体示例。
- **Discord 引入“待定”成员状态**：用户注意到在服务器标签中添加“加入”选项后，加入 Discord 服务器的成员会出现新的“待定（pending）”状态。
   - 成员确认在获得批准前处于“待定”状态，由管理员手动接受，并推测现在可能不再是这种情况。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1430780880175104061)** (59 messages🔥🔥): 

> `Encoder-Decoder Training, Grokking Theory, Meta-Learning RL, New Model Architecture` 


- **Encoder-Decoder 训练：它更好吗？**：一位成员质疑一篇新论文 ([https://arxiv.org/abs/2510.17558v1](https://arxiv.org/abs/2510.17558v1)) 是否优于简单地将 Encoder 用于 prefill 并将 Decoder 用于 generation 的训练方式，并引用了[另一篇论文](https://arxiv.org/abs/2412.09810)。
- **Grokking 理论取得有趣结果**：讨论围绕一篇有趣的 **grokking theory** 论文展开，该论文在经典的模运算任务上取得了结果，并链接到了[论文](https://arxiv.org/abs/2412.09810)及其[代码](https://github.com/brantondemoss/GrokkingComplexity)。
- **元学习强化学习（Meta-Learning RL）发现超越现有规则的规则**：一篇关于从跨环境 Agent 的累积经验中进行**元学习**的论文 ([https://www.nature.com/articles/s41586-025-09761-x](https://www.nature.com/articles/s41586-025-09761-x))，其发现的规则在 **Atari benchmark** 上超越了现有规则，代码可在[此处](https://github.com/google-deepmind/disco_rl)获取。
- **新模型声称具有惊人性能**：一位新成员声称他们的 **50M** 模型实现了 **0.223** 的损失（相比之下，vanilla Transformer 为 **2.73**），其 **1B** 模型的验证损失约为 **0.197**，导致困惑度（perplexity）约为 **1.22**，并附上了一张[图片](https://cdn.discordapp.com/attachments/747850033994662000/1431083629748027402/image.png?ex=68fc1ffd&is=68face7d&hm=c347fd3fa1d4dae87e30579f0723253fd5c83a7197c57f6583d66e7d2ba5ca67&)。
- **推理与 Bug 排查**：成员们对这位新成员的说法提出挑战，认为该模型可能存在 Bug，并建议通过推理进行调试，同时要求查看代码。
   - 该新成员以知识产权（IP）为由拒绝分享代码，引发了质疑，有人认为他们是在寻求关注而非真正的帮助。


  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1430833790175150143)** (4 messages): 

> `Induction Circuits, Causal Abstractions` 


- **可解释电路引发讨论**：成员们讨论了给定模型中用于括号匹配和长空格的 **Induction Circuits** 及结构。
   - 据一位成员所说，这个模型似乎不包含太多其他内容（至少是我们能看到的）。
- **Causal Abstraction 热度降温**：一位成员询问为何近期关于 **Causal Abstractions**（[论文链接](https://arxiv.org/abs/2301.04709)）的讨论变少了，该话题在 2023 年曾被积极讨论。
   - 另一位成员表示，该框架仍然有用，但面临隐式线性假设（[论文链接](https://arxiv.org/abs/2507.08802)）、难以选择合适的抽象级别（[综述链接](https://arxiv.org/abs/2408.01416)）以及单一行为存在多个有效抽象（[论文链接](https://arxiv.org/abs/2502.20914)）等问题。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1430853450048213082)** (30 messages🔥): 

> `Model Deployment Microservice, GPU Memory Management, Weight Hardness in Neural Nets, ARC-AGI 100% MythWox, Docker vs Colab` 


- **Django 应用咨询：是否采用微服务？**：一位成员正在考虑将一个使用 **bi-encoders, cross-encoders, YOLO models** 以及其他 GPU 密集型模型的 Django 应用拆分为独立的微服务，以卸载推理压力。
   - 他们担心单个 GPU 上可能出现请求排队，并就部署策略以及在显存（32GB **VRAM**）有限的情况下如何处理多个并行 **RPC** 调用寻求建议。
- **GPU 显存烦恼与并行 RPC 祈祷**：针对 GPU 过载的担忧，一位成员建议如果显存耗尽则按顺序调度请求，如果内存允许则并行运行。
   - 他们鼓励提问者尝试在 **32GB GPU** 上进行并行 **RPC** 调用，以利用并行处理的优势。
- **权重硬度：Meta 尝试树突方案**：受真实神经元连接的启发，一位成员分享了一个想法，即在每个权重上使用额外的标量来存储其“硬度”或“软度”，并链接了 [Meta 的实现](https://x.com/realJessyLin/status/1980662516285075762)。
   - 这个概念涉及根据计算结果调整权重，加强有益连接并削弱有害连接，从而在不修改不可逆连接的情况下有效地学习新事物。
- **MythWox 宣称 ARC-AGI 达到 100%**：一位成员分享了 [MythWorx.ai](https://mythworx.ai/capabilities/) 的链接，该网站声称在 4 小时内无需预训练即可在 **ARC-AGI 1** 上达到 **100%** 的准确率。
   - 该成员表示怀疑，对该能力的真实性提出质疑，另一位成员则表现出困惑和不解。
- **Colab 无法运行 Docker，Docker 无法运行 Colab**：成员们讨论了 Google Colab 与 **Docker** 容器在深度学习实验中的优缺点，一致认为 Colab 非常适合快速实验，但缺乏 **Docker** 容器和 **SSH** 支持。
   - 一位成员强调，在只有 1 个 CPU 的 Google Colab 上实现可复现性是一个不错的噱头，但由于并非总是能无障碍运行，因此存在局限性。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1430710513293328494)** (26 messages🔥): 

> `Jsonnet, VinePPO, Configuration Management, Tiny Recursive Model, MLP Mixer` 


- **Jsonnet：过度设计还是量身定制？**：成员们辩论了 [Jsonnet](https://jsonnet.org/) 是否过度设计，而另一位成员则赞赏其可定制性以及 **DeepSeek** 对它的使用。
   - 一位成员发现配置文件的目录结构令人望而生畏，即使是像 **VinePPO** 这样简单的实验也是如此。
- **使用 Jsonnet 的函数式配置**：一位成员建议使用 Jsonnet 进行配置，特别是将函数式编程方案应用于模型以实现轻松切换。
   - 另一位成员将其比作 **Nix**，即使用函数式编程定义哈希表，但质疑其作为 JSON 的可读性。
- **微型递归模型复现**：一位成员分享了关于[复现带权重的 Tiny Recursive Model](https://www.alphaxiv.org/models/samsung/tiny-recursive-model)的链接，指出有 95% 的可能性。
   - 发布者强调，**单一网络设计**优于**双网络**，且 **MLP attention** 将性能从 74.7% 提升至 87.4%。
- **VinePPO 配置概览**：一位成员提议讨论 **Jsonnet**/**VinePPO**，随后提供了与之相关的 [ICML](https://icml.cc/virtual/2025/poster/45526)、[OpenReview](https://openreview.net/forum?id=5mJrGtXVwz)、[ArXiv](https://arxiv.org/abs/2410.01679) 和 [GitHub](https://github.com/McGill-NLP/VinePPO) 链接。
   - 随后指出，*RLVR 的配置细节比 Jsonnet 语言本身更重要*。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1430697332848066571)** (8 messages🔥): 

> `Unseeable Prompt Injections, Google's Genie 3 AI` 


- **AI 社区批评论文**：成员们批评了一篇链接论文的质量，其中一人称其虽然*标题有趣*但*质量很差*，另一人也表示*这篇确实不行*。
   - 第一位成员表示，他*原以为这会是一篇有趣且简单的论文，结果对其质量之差感到惊讶*。
- **Brave 揭露“不可见”的 Prompt Injections**：一位成员分享了一篇 [Brave 浏览器博客文章](https://brave.com/blog/unseeable-prompt-injections/)，讨论了**不可见的 prompt injections**。
   - 未添加进一步讨论。
- **Google 准备 Genie 3 实验**：一位成员链接了一份关于 [Google 即将进行的 **Genie 3** 实验](https://www.testingcatalog.com/google-prepares-genie-3-public-experiment-with-ai-generated-worlds/)的报告，该实验旨在生成 AI 世界。
   - 未添加进一步讨论。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1430684999568457909)** (12 messages🔥): 

> `Voice mode accents in models, Open Sourcing decision, Building in Public, Weeb Sam finetune, ChatGPT bug` 


- **模型的口音大冒险：牙买加口音！**：一位用户报告称，**4o 模型**在语音模式下仍会发出**牙买加口音**，而 **model 5** 声称可以做到但完全没有改变声音。
   - 他们特别指出这是*我关心的少数指标之一*。
- **开源还是打磨：开发者的两难境地**：一位成员正在考虑开源他们的软件，但纠结于该推迟发布以修复 bug，还是直接发布带有恼人瑕疵的版本。
   - 他们的应用旨在让任何人都能在本地训练 AI 模型而无需 SaaS，但他们担心发布不够完美的作品。
- **通过公开构建 (Build in Public) 建立信誉**：一位成员建议，**公开构建 (build in public)** 通常是件好事，即使只是写在没人知道的个人博客上。
   - 他们认为这*有助于建立信誉，即使在社交媒体上失败了，你也有一个记录在案的叙事，可以帮助你获得资助/资金*。
- **针对动漫的 Weeb Sam 微调**：该成员提到他们计划在发布中包含一个 **Weeb Sam finetune**，目前包含 **40k 张图像**。
   - 他们声称在处理**动漫 (Anime)** 时，它的表现似乎在某种程度上优于 **SAM**。
- **ChatGPT 的海马惊魂**：一位用户分享了一个关于 **ChatGPT 漏洞**的 [推文](https://fxtwitter.com/ltx_model/status/1981346235194683497) 链接，该漏洞在发送 prompt *is there an emoji of a seahorse?* 时触发。
   - 一位评论者推测，这看起来像是一个正在开发中的 **Unreal Engine/Runway/Wan 竞争对手**。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1430749921040007361)** (11 messages🔥): 

> `Scaling Limits, KBLaM Citation, RL Scaling, Interface Matters, Minecraft Voyager` 


- **新论文引发 Scaling 极限争论**：一位成员分享了一篇[论文](https://arxiv.org/abs/2506.06266)并质疑 **scaling** 是否已达到极限，考虑到 **pretraining**、**test-time compute** 和 **RL** scaling 的收益递减。
   - 其他成员表示反对，认为 *RL、test time compute 等只是 scaling AI 的一个组成部分*。
- **界面设计驱动研究时代精神**：一位成员认为，**界面 (interface)**、**工作流 (workflows)** 和 **基础设施 (infra)** 是 scaling AI 极其重要的组成部分，正如 **ChatGPT 界面**开启了一个研究时代精神。
   - 他们指出 *Claude 将他们的新 prompt 系统称为 ‘Skills’，但实际上这在多年前就由 Minecraft Voyager 开创（并命名）了*。
- **Voyager 的界面前瞻性受到赞赏**：成员们强调，由于**人机交互 (Human-Computer Interaction)** 组件尚未跟上研究步伐，大量研究尚未实现大规模应用。
   - 一位成员赞扬了 **Voyager**，并提到他*有幸在 GTC 听完 Jim Fan 的演讲后与他见面* ([推文](https://x.com/DrJimFan/status/1662117799974809603))。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1430749921040007361)** (11 messages🔥): 

> `Scaling Limits, RL Scaling, Minecraft Voyager, Claude Skills` 


- **Scaling 达到极限，论文声称**：一名成员分享了一篇[论文](https://arxiv.org/abs/2506.06266)，认为这是该想法的一个非常有前景的演进，并询问其他人是否认为 **Scaling** 已经达到了极限。
   - 另一名成员简单地回答了 *no*。
- **界面和基础设施对 Scaling 至关重要**：一位成员表示，**RL**、**test time compute** 等只是 AI **Scaling** 的一个组成部分，界面、工作流和基础设施也极其重要。
   - 他们补充说，有大量的研究尚未大规模实施，因为 **Human-Computer Interaction** 组件尚未跟上研究步伐，并引用了 [Jim Fan 的推文](https://x.com/DrJimFan/status/1662117799974809603)。
- **Claude Skills 对比 Minecraft Voyager**：一名成员讨论了 **Claude** 将其新的 **prompt** 系统称为 *Skills*，但实际上这在多年前就由 **Minecraft Voyager** 开创（并命名）了，只是当时实际实现它的界面尚未大规模形式化（例如 **CLI agents**）。
   - 另一名成员表示他们*喜欢更大的 Voyager*。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1430691461875236906)** (23 messages🔥): 

> `Kimi Partnership, AI Paper Tracking, Chutes vs Moonshot AI for Kimi K2, Home Server Version Control with Git, OpenRouter Chutes ban` 


- **AlphaXiv 地理化追踪 AI 论文**：一名成员分享了 [AlphaXiv](https://www.alphaxiv.org/labs/geo-trends) 的链接，这是一个用于追踪 **AI research papers** 地理来源的工具。
   - 另一名成员要求不要打广告，随后该成员因多次违规被禁言。
- **Kimi K2：更便宜的 Chutes 牺牲了质量**：一名成员询问了作为 **Kimi K2** 供应商的 **Chutes** 与 **Moonshot AI** 相比的质量如何。
   - 另一名成员声称 **Chutes** 在没有数据政策的情况下利用用户数据进行训练，运行时间可靠性较低，且其 **tool call** 准确率仅为官方 **API** 的一半，随后 **Chutes** 在 [Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1da7j6g/did_you_know_you_can_ban_chutes_openrouter_go_to/) 上被做成了梗。
- **家庭服务器使用 Git 管理版本？**：一名成员询问是否可以使用 **Git** 来管理家庭服务器版本，以帮助在发生故障后进行重置。
   - 没有人回应用户的提问或担忧。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1430679223584034928)** (22 条消息🔥): 

> `Manus 沙盒限制，Pro 计划变更，Logic Arenas 应用，Manus 代码弃用，作业解决方案` 


- **混合架构头脑风暴会议**：一位用户建议采用结合本地计算和存储的 *混合架构 (hybrid architecture)*，质疑在用户拥有可用本地资源的情况下为什么要浪费这些资源。
   - 该建议包括支持构建大型原生应用、在本地处理海量数据集，以及在本地硬件上运行资源密集型的 AI 模型。
- **Pro 计划用户声称遭遇“挂羊头卖狗肉”**：一位用户对 **Pro 计划** 现在设有额度限制表示沮丧，声称在购买时该计划的广告宣传是*无限制*的。
   - 用户承认计划可以更改，但认为对已经付费的月份不公平，并询问是否有人想测试 Pro 功能。
- **Manus 创建的 AI 驱动 Arena 应用**：一位用户报告称，Manus 在一小时内使用极少的额度创建了一个全栈 **AI Arenas 应用** ([图片](https://cdn.discordapp.com/attachments/1349440650495398020/1430878322073796628/2025-10-23_12-15.jpg?ex=68fc0988&is=68fab808&hm=efef49879866f78ee43ea3c281eca345ab2bcf800110c561cc4e81bc723f5219&))（*尚未测试*）。
   - 该 **Logic Arenas** 应用包含 *感性 vs 理性 (Heart vs Mind)*、*双亲 (Dual Parents)* 和 *道德法庭 (Moral Court)* 竞技场，使用 **Kotlin**、**Ktor**、**Redis**、**Jetpack Compose** 和 **Material 3** 构建，包含 28 个文件，约 7k 行代码 (LOC)。
- **Manus 代码使用已弃用的模块和插件**：一位用户报告称，**Manus** 可以通过使用非弃用的代码、模块和插件来改进，因为更新所有内容需要耗费大量精力。
   - 虽然 Manus 声称创建了一个精美的 Material 3 暗黑主题（[图片1](https://cdn.discordapp.com/attachments/1349440650495398020/1430917287363350728/2025-10-23_14-51.jpg?ex=68fc2dd2&is=68fadc52&hm=4e68706e60e030a96d99d157b3abdacc9d2b99077639e85f545cb96a0620d614&), [图片2](https://cdn.discordapp.com/attachments/1349440650495398020/1430917287707279391/2025-10-23_14-47.jpg?ex=68fc2dd2&is=68fadc52&hm=04f22a7c26e41abcb491ebfcbdcb2c4ec8b05ad504eeb504de7aa4c2bc80f3f4&))，但最初的 Android 应用设计看起来很过时，不过 Claude Code 仅通过一个提示词就对其进行了改进 ([图片](https://cdn.discordapp.com/attachments/1349440650495398020/1430971270219956234/2025-10-23_18-26_1.jpg?ex=68fbb758&is=68fa65d8&hm=d73323d08753d88efdd123cd9be451622fbb989859f6077c67b76d45598ebfcd&))。
- **Manus 未能通过作业测试**：一位用户报告称，即使在两个 Notebooks 中提供了类似的示例，**Manus** 仍未能正确解决其作业，且生成的 PDF 杂乱无章。
   - 用户强调了 Manus 无法完成这个看似简单的请求。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1430893317289410560)** (7 条消息): 

> `aider API 密钥，gemini-cli 成本，Fedora 上的 Playwright` 


- **Aider API 密钥：具备成本竞争力吗？**：一位用户询问使用带有 API 密钥的 **aider** 是否能达到 **gemini-cli** 的成本效益。
   - 用户提到他们使用 `/tokens` 来管理密钥、清除历史记录并监控上下文大小。
- **Gemini-cli 慷慨的计划**：一位用户描述了 **gemini-cli** 的成本概况，提到一个每月约 **20 美元** 的计划，名义上每天提供 **1500** 次请求，并拥有 **1M Token** 的上下文窗口。
   - 用户还提到，虽然界面大多优于 **aider**，但它缺乏 Repo map，而是大量依赖 `grep` 等文件系统操作。
- **Fedora 上的 Playwright：有人成功吗？**：一位用户询问是否有人成功在 **Fedora** 上运行 **Playwright**。
   - 摘录中未提供解决方案或变通方法。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1430741210439618612)** (1 条消息): 

> `Aider 的未来，Paul Gauthier 的愿景` 


- **社区期待 Paul Gauthier 对 Aider 未来的愿景**：成员们对 **Aider** 的未来发展方向感到好奇，并期待听到其创作者 **Paul Gauthier** 的见解。
   - 社区期待了解该项目即将到来的开发计划和战略决策。
- **关于 Aider 路线图的推测**：讨论围绕 **Aider** 的潜在演变展开，用户渴望了解其路线图 (Roadmap)。
   - 用户寻求关于计划功能、增强功能以及 **Aider** 项目整体轨迹的明确信息。


  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1430758415516569640)** (2 条消息): 

> `tinygrad slide at pytorchconf, tinygrad dev onboarding` 


- **tinygrad 亮相 pytorchconf！**：tinygrad 在 **pytorchconf** 上获得了一个 [展示页](https://cdn.discordapp.com/attachments/1068976834928193609/1430758414744686655/HDjLLqQ.png?ex=68fc429c&is=68faf11c&hm=13bea9c27e9b130accbe594e495dbd0e6813cb720aaabb5bb6f2375495a385b5&)。
   - 该幻灯片展示了 **tinygrad** 在更广泛的 PyTorch 生态系统中的集成和相关性。
- **招募新 tinygrad 开发者**：一名成员询问了关于成为 **tinygrad 开发者** 的文档。
   - 消息中没有直接链接具体资源，但这暗示了扩大 **tinygrad** 贡献者群体的意向。