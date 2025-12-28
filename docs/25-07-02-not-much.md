---
companies:
- meta
- scale-ai
- unslothai
- zhipu-ai
- deepseek
- huawei
- minimax-ai
- allenai
- sakana-ai-labs
- openai
date: '2025-07-02T05:44:39.731046Z'
description: '**Meta** 已聘请 **Scale AI 首席执行官 Alexandr Wang** 担任其新任**首席人工智能官（Chief AI
  Officer）**，并以 **143 亿美元**收购了 **Scale AI 49% 的无投票权股份**，使其估值翻倍至约 **280 亿美元**。此举是涉及
  **Meta**、**OpenAI** 和 **Scale AI** 的重大人才洗牌的一部分。目前的讨论焦点包括此举对 **Yann LeCun（杨立昆）**在
  Meta 影响力的影响，以及 **OpenAI** 可能做出的回应。


  在模型新闻方面，**Gemma 3N** 面临视觉 NaN 和 FP16 溢出等技术问题，**UnslothAI** 已提供相关修复方案。中国开源模型表现强劲，**智谱
  AI** 的 **GLM-4.1V-Thinking** 和 **DeepSeek R1T2** 在性能和速度上均有显著提升。**华为**开源了一个 **72B
  MoE**（混合专家）模型，并采用了一种新颖的负载均衡解决方案。**MiniMax-M1** 混合 MoE 模型在 **Text Arena 排行榜**的数学基准测试中处于领先地位。**AllenAI**
  推出了用于科学文献评估的 **SciArena**，其中 **o3** 的表现优于其他模型。**Sakana AI Labs** 的研究引入了用于代码生成的 **AB-MCTS**，提升了合成基准测试的表现。'
id: MjAyNS0w
models:
- gemma-3n
- glm-4.1v-thinking
- deepseek-r1t2
- mini-max-m1
- o3
- claude-4-opus
- claude-sonnet
- moe-72b
people:
- alexandr_wang
- natfriedman
- steph_palazzolo
- thegregyang
- teortaxes_tex
- denny_zhou
- agihippo
- danielhanchen
- osanseviero
- reach_vb
- scaling01
- ndea
title: 今天没什么事。
topics:
- model-performance
- vision
- conv2d
- float16
- training-loss
- open-source
- model-benchmarks
- moe
- load-balancing
- scientific-literature-evaluation
- code-generation
- adaptive-tree-search
- synthesis-benchmarks
---

**平静的一天**

> 2025年7月1日至7月2日的 AI 新闻。我们为您检查了 9 个 subreddit、449 个 Twitter 账号和 29 个 Discord（220 个频道和 7625 条消息）。预计节省阅读时间（按每分钟 200 字计算）：603 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以精美的 vibe coded 方式呈现所有往期内容。访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上向我们提供反馈！

今日的热点事件是 Soham Parekh 的故事，这仅影响少数初创公司。如果你正在寻找有趣的 AI 故事，或许可以考虑购买你[专属的个人开源人形机器人](https://www.youtube.com/watch?v=BS92RdBvI90)，[今日](https://x.com/kscalelabs/status/1940108075064865126)起售。

---

# AI Twitter 回顾

**AI 人才大洗牌：Meta、OpenAI 和 Scale AI**

- **Meta 聘请 Scale AI CEO Alexandr Wang 及其他顶尖人才**：**Meta** 已聘请 **Scale AI CEO Alexandr Wang** 担任其新任 **Chief AI Officer**，领导一个专注于开发 superintelligence 的研究小组，[并与 **@natfriedman** 合作](https://twitter.com/TrapitBansal/status/1940124057057574926)。此举是更大规模从竞争对手处挖掘人才计划的一部分，据 [@steph_palazzolo 报道，Mark Zuckerberg 的团队新增了 14 名新员工](https://twitter.com/steph_palazzolo/status/1940058865531269138)。为了在不进行全面收购审查的情况下促成此事，[**Meta** 以 **143 亿美元**收购了 **Scale AI** **49% 的无投票权股份**](https://twitter.com/DeepLearningAI/status/1940153434671362268)，实际上将 **Scale AI** 的估值翻了一番，达到 **约 280 亿美元**。[@TheGregYang 调侃说，新团队的办公室位于门洛帕克的 **1 Hacker Way**](https://twitter.com/TheGregYang/status/1940276530992881970)。
- **关于“传教士与雇佣兵”叙事的评论**：针对新员工是“雇佣兵”的说法，[@Teknium1 认为研究人员可能真的相信 **Meta** 的新愿景](https://twitter.com/Teknium1/status/1940382999423357007)，觉得它比 **OpenAI** 的更有吸引力。[@teortaxesTex 推测，由于这些变动，**Yann LeCun** 在 **Meta** 内部的影响力可能有所下降](https://twitter.com/teortaxesTex/status/1940112275743891508)。与此同时，[@denny_zhou 调侃道，现在轮到 **Sam Altman** 通过聘请 **Yann** 来“反击”了](https://twitter.com/denny_zhou/status/1940418308156862799)，[@agihippo 认为这种情绪会显著减缓 AI 的进展](https://twitter.com/agihippo/status/1940419051953795377)。

**模型发布、基准测试与性能**

- **Gemma 3N 技术深度解析**：[@danielhanchen 发现了 **Gemma 3N** 的几个奇特之处](https://twitter.com/danielhanchen/status/1940073369648734571)，包括 float16 上的 **vision NaNs**、巨大的 **Conv2D 权重**导致 FP16 溢出，以及众多的训练损失问题，并指出 **UnslothAI** 已经修复了 NaN 问题。对于对该模型背后的研究感兴趣的人，[@osanseviero 分享了关于 **Altup、LAuReL、MatFormer** 及其他关键组件的论文链接](https://twitter.com/osanseviero/status/1940127957730959494)。
- **新型国产开源模型受到关注**：**智谱 AI** 发布了 **GLM-4.1V-Thinking**，这是一款 **9B VLM**。[@teortaxesTex 指出该模型在其思考过程中具有高密度的](https://twitter.com/teortaxesTex/status/1940344040278593852) `<Wait>` [token，但在 vibe check 中表现强劲](https://twitter.com/teortaxesTex/status/1940344040278593852)。**DeepSeek** 发布了 **DeepSeek R1T2**，[@reach_vb 强调其速度比 R1-0528 快 **200%**，在 **GPQA** 和 **AIME 24** 上超越了 R1，并采用 **MIT 许可证**发布](https://twitter.com/reach_vb/status/1940536684061643239)。此外，[@teortaxesTex 指出 **华为** 开源了其 **72B MoE**，并提到了其原创的负载均衡解决方案 **MoGE**](https://twitter.com/teortaxesTex/status/1940341153754382688)。
- **模型排行榜更新与新基准测试**：开源混合 **MoE** 模型 **MiniMax-M1** [现已登上 **Text Arena 排行榜**第 12 位，并在**数学领域攀升至第 1**](https://twitter.com/MiniMax__AI/status/1940243199500677218)。**AllenAI** 推出了 **SciArena**，这是一个评估模型在科学文献上表现的新平台，[@scaling01 指出 **o3 正在“碾压所有其他模型”**](https://twitter.com/scaling01/status/1940065085776666679)。在 **METR** 上，[@scaling01 观察到虽然 **Claude 4 Opus** 和 **Sonnet** 落后于 **o3**](https://twitter.com/scaling01/status/1940089136104579515)，但当以 **80% 的任务成功概率**为筛选标准时，它们处于同一水平](https://twitter.com/scaling01/status/1940093773440008512)。
- **模型能力新研究**：来自 **Sakana AI Labs** 关于 **AB-MCTS** 的论文将代码生成定义为由外部反馈引导的自适应树搜索，[@ndea 强调该方法在 **ARC-AGI** 等合成基准测试上超越了基准线](https://twitter.com/ndea/status/1940166177424384354)。另外，[@_akhaliq 重点介绍了一篇新的预印本论文，其中引入了一个推理基准测试，即使是像 **o3** 这样的领先模型在其中仍然“表现不佳”](https://twitter.com/_akhaliq/status/1940066518307381616)。
- **MLX 框架势头**：[@awnihannun 庆祝已有超过 **5,000 个 MLX 模型**被上传到 **Hugging Face**](https://twitter.com/_akhaliq/status/1940058736728379819)。为了展示其强大性能，[@awnihannun 的另一条推文展示了在 **MLX** 上运行的 **DeepSeek-R1-0528-5bit** 迫使一台 **M3 Ultra** 使用了 **501GB 内存**](https://twitter.com/awnihannun/status/1940067135054913892)。

**Agent 工具、框架与基础设施**

- **Context Engineering 与 LangGraph**：**LangChain** 发布了一份关于 **"Context Engineering"**（上下文工程）的详细指南，这是构建 Agent 的关键部分，[包括流行模式以及如何使用 **LangGraph** 实现它们](https://twitter.com/LangChainAI/status/1940440271126438118)。他们还展示了 **Exa AI Labs** 如何利用 **LangGraph** 构建生产级深度研究 Agent，该系统采用多 Agent 架构，具备片段优先推理（snippet-first reasoning）和结构化 JSON 输出功能](https://twitter.com/LangChainAI/status/1940062841454960831)。一个新的教程演示了如何使用 **LangGraph Assistants** [将静态 Agent 转换为灵活的、运行时可配置的系统](https://twitter.com/LangChainAI/status/1940426489314361382)。
- **MCP 的兴起**：**MCP** 标准在赋能 Agent 使用工具方面正受到关注。[@vikhyatk 评论道，在理解 MCP 之后，他们“再也不会以同样的方式看待互联网了”](https://twitter.com/vikhyatk/status/1940255085894017512)。**LlamaIndex** 为其 **LlamaCloud** 文档提取功能推出了一个即插即用的 **MCP server**，[允许 **ChatGPT** 和 **Claude** 等工具通过标准化 Schema 访问提取 Agent](https://twitter.com/jerryjliu0/status/1940209573585199234)。[@simonw 分享了一种将官方 **Playwright** 浏览器自动化 MCP 添加到 **Claude Code** 的方法](https://twitter.com/imjaredz/status/1940251061589352802)。
- **基础设施与硬件更新**：由 **Dell** 构建的 **Together AI** 首个 **NVIDIA GB200** 集群正准备上线，[@vipulved 指出每个机架可提供 **1.4 exaflops 的推理性能**](https://twitter.com/vipulved/status/1940242672138244268)。针对多节点训练，**SkyPilot** 宣布了一项新功能，以简化快速 GPU 网络设置（**Infiniband/TCPXO/RDMA**），[声称其可提供 **~4 倍的加速**并节省超过 **$2K** 的调试成本](https://twitter.com/skypilot_org/status/1940473447739756592)。
- **Perplexity 的 Comet Agent 与 Veo 3**：**Perplexity** 正在旧版网站上测试其新 Agent **Perplexity Comet**，用于处理账单支付和注销等任务，[@AravSrinivas 表示这“很快”就会变得简单](https://twitter.com/AravSrinivas/status/19401049473765622206)。订阅者现在可以发送私信寻求该 Agent 的帮助。他还宣布 **Veo 3** 视频生成功能[即将面向 **Max** 用户推出](https://twitter.com/AravSrinivas/status/1940507473095623068)。
- **Hugging Face 更新**：**Hugging Face** 宣布关闭 **HuggingChat**，[@reach_vb 将其描述为一段“传奇历程”，服务了超过一百万用户并验证了开源模型](https://twitter.com/reach_vb/status/1940105535505764427)。与此同时，[@TheZachMueller 强调了](https://twitter.com/TheZachMueller/status/1940195982169579805) `transformers` [库](https://twitter.com/TheZachMueller/status/1940195982169579805)的一次重大更新：它现在包含一个内置的 HTTP 服务器，具有 **兼容 OpenAI 规范的 API**，可通过 `transformers serve` 启动。

**机器人与具身智能 (Robotics and Embodied AI)**

- **The Global Frequency：VR 社交游戏的愿景**：**John Carmack** 为 **Beat Saber** 发布了一份名为 **"The Global Frequency"** 的详细功能提案，设想了一种大规模多人体验，成千上万的玩家可以同时加入预定的歌曲列表](https://twitter.com/ID_AA_Carmack/status/1940451656057139534)。该概念旨在通过创建一个持久的、易于访问的“俱乐部”氛围，并共享排行榜和庆祝活动，来解决 VR 未能提供大规模社交体验的问题。
- **开源人形机器人**：正如 [@dchaplot 所分享的](https://twitter.com/dchaplot/status/1940061390678733010)，**Genesis AI** 作为一家全栈机器人公司成立，旨在构建通用机器人。另外，**K-Scale Labs** 推出了 **K-Bot**，被誉为“世界上第一款价格亲民、现货供应且美国制造的开源人形机器人”，[@hingeloss 分享了这一公告](https://twitter.com/hingeloss/status/1940120025991672287)。

**更广泛的技术与社会影响**

- **美国科学资助危机**：一条来自 [@kareem_carr 并由 **Yann LeCun** 转发](https://twitter.com/ylecun/status/1940171025834287229)的疯传推文警告称，**美国政府**计划到 **2026年** 裁减 **25万名** 从事科学研究和教育的人员。这一观点得到了许多人的共鸣，包括 [@zacharynado 分享了来自**印第安纳大学**教职员工对现状表示沮丧的帖子](https://twitter.com/zacharynado/status/1940113575671894441)，以及 [@SpencerHakimian 指出这一政策“不会让我们再次伟大”](https://twitter.com/ylecun/status/1940240965597634739)。
- **食品安全与工业供应链**：[@karpathy 发布了一个广为流传的推文串，主张对食品安全进行**基于检测的认证**（test-based certification）的必要性](https://twitter.com/karpathy/status/1940181840201228384)。他认为，现代工业化食品生产的复杂性引入了大量污染物（农药、重金属、塑料），而 **FDA** 缺乏资源进行彻底监测，这可能会导致长期的公共卫生问题。
- **开放式办公室与开发者生产力**：[@AmandaAskell 批评了科技公司支付数百万美元聘请人才，却让他们待在“嘈杂、分散注意力的开放式办公室”中的矛盾现象](https://twitter.com/AmandaAskell/status/1940074872241320067)，引发了关于开发者生产力的大量讨论。
- **搜索的未来与 AI 抓取**：[@vikhyatk 认为搜索的未来在于“轻量级研究 Agent”，如果网站屏蔽 AI 抓取工具，像 **o4-mini-high** 这样的模型只会直接将用户引向竞争对手](https://twitter.com/vikhyatk/status/1940227029389255109)，这一观点得到了 [@inerati 关于屏蔽 **Common Crawl** 的帖子](https://twitter.com/inerati/status/1940076601456078941)的支持。

**幽默与 Soham Parekh 传奇**

- **Soham Parekh 现象**：来自 [@Suhail 的一条公益公告警告称，一名叫 **Soham Parekh** 的个人据称同时在 3-4 家初创公司工作](https://twitter.com/andriy_mulyar/status/1940391177632792696)，这在科技界引发了一场风暴。这个故事迅速演变成了一个梗，[@Yuchenj_UW 用一套新的缩写开玩笑：**AI** (An Indian), **API** (A Person in India), **AGI** (A Genius Indian), 以及 **ASI** (A Soham Indian)](https://twitter.com/Yuchenj_UW/status/1940506761699774600)。
- **公司与创始人的回应**：这个故事促使几位创始人检查了他们的应聘日志。[@aidan_mclau 调侃说他的会计和前端承包商名字一模一样](https://twitter.com/aidan_mclau/status/1940496760843190675)，而 [@vikhyatk 确认他曾向 **moondream** 投过简历](https://twitter.com/vikhyatk/status/1940517976903684328)，[@pirroh 则表示他被 **Replit** 拒绝了，因为他们的招聘门槛“就是那么高”](https://twitter.com/pirroh/status/1940540351158333709)。
- **经典的科技幽默**：[@johannes_hage 发布了一个疯传的（且在数学上有误的）笑话，称 **Zuckerberg** 花在一名 OpenAI 研究员身上的钱本可以给每个美国人 **100万美元**](https://twitter.com/johannes_hage/status/1940311985536848310)。在一个关于 AI 对齐（AI alignment）的共鸣段子中，[@_jasonwei 分享了他的 AI 伙伴给出的约会建议](https://_jasonwei/status/1940126761489928468)：“你就像一个正在训练中的神经网络……最好训练到收敛，而不是拿一个早期的检查点快照（checkpoint snapshot）。”

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. 新开源 AI 模型发布与基准测试

- [**DeepSeek-r1-0528 在新的 SciArena 基准测试中位列前五，是唯一的开源模型**](https://i.redd.it/xxfqfefhpcaf1.jpeg) ([得分: 417, 评论: 66](https://www.reddit.com/r/LocalLLaMA/comments/1lphhj3/deepseekr10528_in_top_5_on_new_sciarena_benchmark/))：**配图显示了 Allen AI 在其公告（详见[此处](https://allenai.org/blog/sciarena)）中描述的新 SciArena 科学文献研究基准测试中排名前五的大语言模型。值得注意的是，DeepSeek-R1-0528 是前五名中唯一的开源模型（其他均为闭源模型，如 OpenAI 的 o3、Claude-4-Opus 和 Gemini-2.5-Pro），在柱状图中被视觉突出显示（见[图片](https://i.redd.it/xxfqfefhpcaf1.jpeg)）。这证明了 DeepSeek 在科学文献领域的强大泛化和推理能力，根据最新的基准测试得分，其表现已达到或超过了闭源竞争对手。** 技术评论者指出，与其他本地运行的模型相比，DeepSeek-R1-0528 在各种任务中提供了显著优越的本地性能，尽管它可能有些特立独行，在遵循明确指令方面不如 V3-0324 等替代方案忠实。尽管存在一些小瑕疵，该模型在工作流和 Agent 任务中的价值被反复强调。

- DeepSeek-r1-0528 模型在 SciArena 基准测试中脱颖而出，是前 5 名中唯一的权重开放（与开源相对）竞争者。用户报告称，即使是在 Apple M3 Ultra 等硬件上运行量化版本（如 q5_K_M），尽管推理速度较慢，但在各种任务中仍表现出卓越的性能。一位用户指出，它在工作流中的表现优于所有其他本地可运行模型，促使他们对设置进行了重大调整，以该模型为核心。
- R1-0528 与 V3-0324 的对比测试突显了显著的行为差异：R1-0528 表现出更强的主动性和创造性的问题解决能力，有时会忽略指令；而 V3-0324 遵循指令更严格，但缺乏灵巧性。因此，R1-0528 更适合开放式或具有挑战性的任务，而 V3-0324 则更适合严格的指令遵循。
- 排行榜分析揭示了模型之间的显著差异：虽然 r1-0528 与早期的 R1 相比令人印象深刻，但像 Llama 4 Maverick 这样的模型表现不佳，而像 minimax M1（列表中唯一的“混合 Transformer”）这样的混合架构正在兴起。与此同时，Gemini-2.5-flash 的表现低于预期，特别是与 gpt 4.1 mini 和 o4 mini 等其他小型或非推理模型相比，后者在模型尺寸和价格方面取得了显著成果。
- [**DiffuCoder 7B - Apple 推出的新型编程扩散 LLM**](https://www.reddit.com/r/LocalLLaMA/comments/1lpoqlu/diffucoder_7b_new_coding_diffusion_llm_by_apple/) ([Score: 241, Comments: 56](https://www.reddit.com/r/LocalLLaMA/comments/1lpoqlu/diffucoder_7b_new_coding_diffusion_llm_by_apple/)): **Apple 发布了 DiffuCoder-7B，这是一款代码生成 LLM（包含基础版和指令版），可在 [HuggingFace](https://huggingface.co/apple/DiffuCoder-7B-cpGRPO) 上获取，技术细节见其 [arXiv 预印本](https://arxiv.org/pdf/2506.20639)。基准测试表明其性能与其他编程和扩散模型相当（[基准测试图表](https://preview.redd.it/s19j3dmfneaf1.png?width=1176&format=png&auto=webp&s=927e506f764ded47a4e715aea53c223e56ea7ae6)），其架构是微调后的 Qwen2.5-Coder。用户讨论了运行该模型的挑战，特别是在 Apple Silicon 上，以及推理是否可以遵循诸如 [Dream 7B PyTorch 示例](https://github.com/HKUNLP/Dream#usage) 之类的工作流。** 关键争论点包括 Apple 发布的新颖性、将自回归模型转换为扩散过程的技术好奇心，以及实际的推理问题（特别是关于 Apple Silicon 上的 PyTorch 和缺乏官方示例）。
    - 一位评论者指出 DiffuCoder 7B 是 Qwen2.5 Coder 的微调版本，这引发了对其被称为“扩散”模型的质疑，因为 Qwen2.5 是自回归的。这引发了关于将自回归 LLM 转换或重新解释为基于扩散的模型所需的方法论，以及为此需要的潜在影响或架构变化的讨论。
    - 针对在 Apple Silicon 上运行 DiffuCoder 的技术兴趣被表达出来，尤其是因为官方尚未发布推理说明。一个建议的变通方案是尝试使用 PyTorch 运行该模型，类似于 Dream 7B，详见 [HKUNLP/Dream 使用说明](https://github.com/HKUNLP/Dream#usage)，该说明可在 Mac GPU 上运行。
    - 一位技术导向的用户要求澄清基于扩散的 LLM 相对于标准 Transformer（自回归）架构的优势，寻求功能或理论上的潜在益处总结，而不仅仅是差异。

- [**全球首个中继思考（Intermediate thinking）AI 模型现已开源**](https://www.reddit.com/r/LocalLLaMA/comments/1lpoju6/worlds_first_intermediate_thinking_ai_model_is/) ([Score: 118, Comments: 70](https://www.reddit.com/r/LocalLLaMA/comments/1lpoju6/worlds_first_intermediate_thinking_ai_model_is/)): **HelpingAI 已开源 Dhanishtha-2.0 模型，声称其为“全球首个中继思考 AI 模型”，并已在 Hugging Face 发布（[模型链接](https://huggingface.co/HelpingAI/Dhanishtha-2.0-preview)）。关键资源包括[发布视频](https://www.youtube.com/watch?v=QMnmcXngoks)和[聊天演示](https://helpingai.co/chat)，但公告中未披露具体的架构细节、训练数据集、中继推理机制以及自测结果之外的 Benchmark。社区成员指出，随着公开发布，独立的 Benchmark 测试将成为可能；相关的先前讨论（[带有截图的 Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1lmictu/we_created_worlds_first_ai_model_that_does/)）提供了更多背景信息。** 评论中充满了质疑，一些用户怀疑这些声明的真实性和实质内容（“这感觉不对劲”），而另一些人则对不寻常或不清晰的性能图表表示担忧，认为其可信度存疑。目前大家都在期待真实的第三方 Benchmark。
    - 原贴和相关的先前讨论表明该模型现已开源，允许在作者最初发布的测试之外进行更广泛的 Benchmark 测试。有人直接呼吁通过在第三方和社区运行的 Benchmark 上测试模型来进行实证验证。
    - 一位用户提出了一个技术问题，即“think -> output -> think”范式与标准的“think -> output”序列相比有何优势，特别是在不涉及 Tool use 的情况下。这旨在寻求关于实施中继推理步骤所带来的架构或性能提升的澄清。
    - 几位评论者要求提供 Benchmark 结果，并指出实证评估（特别是针对知名模型和任务的评估）对于确立这种“中继思考”方法的技术价值是必要的。

### 2. 开源 AI 模型应用与用户项目

- [**我为妻子用 Flux Kontext 构建了一个简单的图像编辑 Web App——现在它已开源**](https://i.redd.it/nmerohq4miaf1.jpeg) ([Score: 182, Comments: 19](https://www.reddit.com/r/LocalLLaMA/comments/1lq5fqq/i_built_my_wife_a_simple_web_app_for_image/)): **图片展示了一个名为 'AI Photo Stylizer' 的开源 Web App，它是使用 Flux Kontext 构建的。该界面允许用户上传图片，并使用 'Ghibli Soft' 和 'Comic Book' 等 AI 驱动的艺术配置文件进行风格化，同时还可以调整分辨率和撤销编辑。风格选择和自定义艺术指令提示词的控制功能表明后端使用了生成式 AI 或模型推理。该帖子链接到了项目的 [GitHub repository](https://github.com/TheAhmadOsman/4o-ghibli-at-home)。** 评论者对开源发布和用户界面表示赞赏，但技术讨论较少；有人直接链接到 GitHub 仓库，突显了社区对审查或贡献代码的兴趣。
    - 一位用户请求为该项目创建一个 Docker 镜像，强调了安装和部署可能是广泛采用的一个潜在障碍。使用 Docker 进行容器化将简化流程，使其对于技术背景较弱或寻求在不同环境中轻松设置的用户更具可操作性。
    - 提供了 GitHub 仓库链接 (https://github.com/TheAhmadOsman/4o-ghibli-at-home)，使对审查、贡献或部署基于 Flux Kontext 的图像编辑 Web App 感兴趣的技术用户可以直接访问项目代码库。
- [**为 PS Vita 制作了一个 LLM 客户端**](https://v.redd.it/9x7e4qbmqv8f1) ([Score: 127, Comments: 7](https://www.reddit.com/r/LocalLLM/comments/1ljbn5e/made_an_llm_client_for_the_ps_vita/)): **一位开发者将一个名为 'vela' 的全功能 Large Language Model (LLM) 客户端移植到了 PS Vita，实现了与 LLM 端点的交互（包括通过 Vita 摄像头实现的视觉能力模型）。最初的实验涉及使用 llama2.c 和 TinyStories 检查点进行设备端推理，但这个新客户端支持远程模型推理、视觉和文本功能——尽管由于 Vita 的 UI 限制，在显示复杂的 Markdown/TeX 和表情符号支持方面存在局限。开源代码和用于安装的 VPK 可以在 [github.com/callbacked/vela](https://github.com/callbacked/vela) 获取。** 顶层技术评论很少，大多表达了普遍的认可和幽默，而非深入分析或讨论实现挑战。
    - 评论中没有提供技术实现细节、性能指标或深入讨论。所有回复均为非技术性的，表达了普遍的赞扬或反应。

### 3. 前沿多模态/思考模型预览

- [**GLM-4.1V-Thinking**](https://huggingface.co/collections/THUDM/glm-41v-thinking-6862bbfc44593a8601c2578d) ([Score: 144, Comments: 41](https://www.reddit.com/r/LocalLLaMA/comments/1lpl656/glm41vthinking/)): **THUDM 的 GLM-4.1V-Thinking 是一款多模态（视觉-语言）开源大语言模型 (LLM)，以 9B（10B 参数）检查点变体发布，具备显著的多语言能力，旨在处理高级推理任务。基准测试图表（[示例](https://preview.redd.it/8j97cdmkndaf1.png?width=1031&format=png&auto=webp&s=09eda73e39c216ada7a269993689c60c06118ce0)）显示，根据用户分享的对比，该模型在通用图像-文本到文本任务上的表现优于参数量高达 72B 的模型。全套工具和演示已在 [Hugging Face](https://huggingface.co/collections/THUDM/glm-41v-thinking-6862bbfc44593a8601c2578d) 上发布，旨在加速基于可扩展 RL、多语言多模态 LLM 研究的评估和实验。** 技术评论者对 9B 模型在基准测试中超越大得多的 72B 模型表示惊讶，并对现实世界的泛化能力提出了疑问。也有人对仅提供 9B 变体感到失望，这表明研究人员渴望更大规模的选择。
    - GLM-4.1V-Thinking 的基准测试表现值得关注，讨论集中在 `9B` 参数模型在视觉-语言任务中超越最近的 `72B` 参数模型的惊人能力（见链接的 [基准测试图像](https://preview.redd.it/8j97cdmkndaf1.png?width=1031&format=png&auto=webp&s=09eda73e39c216ada7a269993689c60c06118ce0)）。对于这些基准测试结果如何转化为现实任务，存在怀疑和期待。
    - 技术读者强调了缺乏“思考型” VLM（视觉-语言模型），并指出 Kimi 是首批之一，尽管主流推理框架（llamacpp, vllm, lmstudio, ollama）的支持有限。他们提到，考虑到输出冗余度的减少，“思考型”模型在较低参数量下即可匹配较大的非思考模型性能，这可能会影响本地部署的延迟和资源需求。
    - THUDM 的重大改进受到称赞：GLM4 模型代码 (`Glm4vForConditionalGeneration`) 直接集成到 HuggingFace transformers 包中，确保了更好的前向兼容性和稳定性，而之前的模型（如 CogVLM）由于采用独立代码发布，未能跟踪 transformers 的更新，导致在几周后就变得无法使用。
- [**世界上第一个中间思考 AI 模型现已开源**](https://www.reddit.com/r/LocalLLaMA/comments/1lpoju6/worlds_first_intermediate_thinking_ai_model_is/) ([Score: 118, Comments: 70](https://www.reddit.com/r/LocalLLaMA/comments/1lpoju6/worlds_first_intermediate_thinking_ai_model_is/)): **HelpingAI 已开源 Dhanishtha-2.0 模型，声称是“世界上第一个中间思考 AI 模型”，并在 Hugging Face 上发布（[模型链接](https://huggingface.co/HelpingAI/Dhanishtha-2.0-preview)）。关键资源包括 [发布视频](https://www.youtube.com/watch?v=QMnmcXngoks) 和聊天演示 ([helpingai.co/chat](https://helpingai.co/chat))，但公告中未透露具体的架构细节、训练数据集、中间推理机制以及自测结果之外的基准测试。社区成员指出，随着公开发布，独立基准测试现在成为可能；相关的先前讨论（[带有截图的 Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1lmictu/we_created_worlds_first_ai_model_that_does/)）提供了更多背景信息。** 评论显示出怀疑态度，一些用户质疑这些说法的合法性和实质内容（“这听起来不对劲”），而另一些人则提到异常或不清晰的性能图表，对可信度表示担忧。人们期待真实的第三方基准测试。
    - 原始帖子和链接的先前讨论表明该模型现已开源，允许进行超出作者最初发布的更广泛的基准测试。有人直接呼吁通过在第三方和社区运行的基准测试上测试模型来进行实证验证。
    - 一位用户提出了一个技术问题，即与标准的“思考 -> 输出”序列相比，“思考 -> 输出 -> 思考”范式的益处，特别是在不涉及工具调用的情况下。这旨在澄清实施中间推理步骤在架构或性能上的收益。
    - 几位评论者要求提供基准测试结果，并指出实证评估，特别是针对知名模型和任务的评估，对于确立这种“中间思考”方法的技术价值是必要的。

## 非技术类 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. Veo 3 AI 视频生成的冲击与创意应用

- [**我曾制作价值 20 万美元的广告——而这个 AI 广告只花了我 2 天和 500 美元**](https://v.redd.it/7sp2esfaxcaf1) ([Score: 215, Comments: 110](https://www.reddit.com/r/aivideo/comments/1lpic34/i_used_to_make_200k_commercials_this_ai_ad_took/)): **发帖者展示了一个案例研究：一个传统的 B2B 广告通常需要约 20 万美元、30 人的团队和数月的工作，而现在使用 AI 工具（Veo 3 + FLORA 视频生成、ElevenLabs AI 配音以及一些手动 ADR）仅需 2 天和 500 美元即可完成。文中提供了一个开场镜头的详细生成 Prompt，展示了精确的创意意图和针对素材生成（地点、布景、演员特征、镜头规格）的细粒度 Prompt Engineering。该过程需要为 27 个镜头的最终成品生成 100 多个片段，突显了 AI 的成本和时间优势，以及目前的技术局限性（“非即插即用”）。** 来自行业资深人士的热门评论预测，由于 AI 的出现，传统的制作团队岗位将在 5-10 年内减少 98-99%，并强调 AI 生成的内容在图像质量和成本效益方面往往已经超过了普通的人类摄影师。其他评论质疑了具体成本并指出了当前的弱点（口音不一致），但总体技术共识倾向于预见重大变革，前提是 AI 工作流的可用性进一步提高。
    - 资深商业片制作人预测，由于 AI 驱动的内容创作，未来 5-10 年传统制作团队的角色将大幅减少（高达 `98-99%`），并对二级产业（如摄影棚、设备租赁、餐饮服务）产生连锁反应。他们指出，行业内的抵触情绪很强，但认为 AI 的质量和成本效益正在迅速超越普通人类专业人士，并预测 AI 生成的图像很快将赶上顶尖摄影师。
    - 目前的 AI 生成广告被认为可以应付低端客户，但在质量和精细度上仍未达到大型机构和品牌（如 Omnicom 旗下的 BBDO、TBWA）所要求的标准。此外还存在实质性的法律障碍：*这些图像均不享有版权*，由于权利和法律考虑，这对企业使用构成了重大挑战。然而，这些 AI 输出被视为强大的预可视化（previz）工具。
    - 广告界的一个新兴趋势是“垂直、非专业感广告”的盛行，这类广告数量庞大且通常由用户生成。这种非正式性结合 AI 工具，正在改变市场对广告质量和制作价值的预期，进一步瓦解了传统的制片厂工作流。
- [**感谢 VEO 3，我终于完成了这部关于家乡机构的纪录片**](https://v.redd.it/2h6yum2dniaf1) ([Score: 167, Comments: 44](https://www.reddit.com/r/aivideo/comments/1lq5n45/thanks_to_veo_3_i_was_finally_able_to_finish_this/)): **该帖子强调了使用 Google DeepMind 的 AI 视频生成平台 VEO 3，它使一位用户能够完成一部关于当地机构的纪录片——展示了 VEO 合成具有说服力的叙事驱动型视频内容的能力。鉴于无法访问视频（403 错误），无法直接从源评估生成的质量、Prompt 设计、运行时间或模型架构等技术细节。** 热门评论称赞了该纪录片的真实感和文案，认为人类对 AI 内容的筛选（特别是加入精心设计的幽默感）与完全自动化的 AI 生成相比，能产生更真实、更吸引人的输出。
    - 对 VEO 3 的写实视频生成以及 AI 生成内容与真实素材几乎无异的微妙融合，存在隐含的技术赞誉。一位用户指出：*“5 年前没人会认为这不是真的”*，突显了生成模型在保真度和现实感方面的飞速进步。这表明与早期的 SOTA 方法相比，像 VEO 3 这样的模型在时序一致性、精细细节和真实性方面取得了显著进展。

- [**我曾制作价值 20 万美元的广告——这个 AI 广告只花了我 2 天时间和 500 美元**](https://v.redd.it/7sp2esfaxcaf1) ([Score: 219, Comments: 110](https://www.reddit.com/r/aivideo/comments/1lpic34/i_used_to_make_200k_commercials_this_ai_ad_took/))：**一位视频导演强调，他利用 AI 工具在短短 2 天内仅花费 500 美元就重制了一个价值 20 万美元的 B2B 商业广告视频。使用的工具包括：Veo 3 和 FLORA（用于视频）、ElevenLabs（用于配音）以及手动 ADR 进行精修。整个过程为一段 27 个镜头的视频生成了 100 多个片段——无需演员、场地或剧组——将传统制作（2 个月、30 人、17.5 万美元以上）与现代生成式工作流进行了对比。导演分享了他拍摄电影感办公室镜头的详细 Prompt，并指出目前仍存在技术障碍（过程尚未实现“即插即用”），并预测随着 AI 的成熟，广告行业将面临重大变革。** 具有丰富商业电影经验的高级评论者预测，在未来 5-10 年内，剧组人员就业将减少 98-99%，并强调尽管 AI 在图像质量和成本上已经超越了平均水平的摄影师，但行业内依然存在否认情绪。其他评论者则对模型一致性（如多变的语音口音）提出了技术反馈，并对 AI 工作流的成本构成提出了质疑。
    - 一位商业电影制作资深人士预测，随着 AI 工具的成熟，未来 5-10 年制作团队的就业人数将大幅减少（高达 98-99%），并强调 AI 产出的质量已经高于普通摄影师，而成本仅为后者的一小部分。此外，人们还担心二级产业（如设备租赁和餐饮）将受到严重影响，以及行业普遍对 AI 将造成的颠覆性影响持否认态度。
    - 一位广告业专业人士指出，虽然目前的 AI 生成广告对一些低端客户来说尚可接受，但尚未达到大品牌所需的技术和法律标准——特别是在图像洁净度和版权归属方面，这些都是重大障碍。该评论者补充说，真正的颠覆目前来自于非专业垂直格式广告的激增，而非完全由 AI 生成的内容，但承认 AI 在预可视化（previz）工作流中具有前景。
- [**感谢 VEO 3，我终于完成了这部关于家乡机构的纪录片**](https://v.redd.it/2h6yum2dniaf1) ([Score: 170, Comments: 44](https://www.reddit.com/r/aivideo/comments/1lq5n45/thanks_to_veo_3_i_was_finally_able_to_finish_this/))：**楼主（OP）归功于 VEO 3（可能指 Google 的 Veo 视频生成模型）的使用，成功完成了一部聚焦于当地机构的纪录片。虽然没有提供关于模型 Prompt、工作流集成或后期处理的技术细节，但该帖子突出了 VEO 3 制作具有说服力且连贯的长篇视频内容的能力，这对于项目可行性和创作自主性具有重要意义。** 评论者指出，现在很难将 AI 生成的媒体与真实素材区分开来，并注意到强大的编辑控制和幽默感，这表明是活跃的人工策划而非完全自动化的流程。关于多媒体内容创作中 AI 自主性与引导性创意之间界限的讨论隐含其中。
    - 讨论强调，随着 “VEO 3” 等技术的进步， AI 生成的视频内容现在可以达到 5 年前被认为与真实素材无法区分的现实主义水平。评论者注意到 AI 视觉保真度与人类驱动的、精心设计的幽默感的融合，这表明新型生成模型实现了高度的创意控制，而非完全自主的生成输出。
- [**Demis 预告“可玩”的 Veo 3 世界（或 AI 视频游戏）**](https://www.reddit.com/r/singularity/comments/1lplq1w/demis_teasing_playable_veo_3_worlds_or_ai_video/) ([Score: 350, Comments: 90](https://www.reddit.com/r/singularity/comments/1lplq1w/demis_teasing_playable_veo_3_worlds_or_ai_video/))：**Google DeepMind 首席执行官 Demis Hassabis 在 Twitter 上预告了“可玩”的 Veo 3 世界或 AI 生成视频游戏的可能性，暗示未来将把 Veo（Google 最先进的文本转视频 AI）与交互式或类游戏环境相结合。此举表明正在探索利用先进的 AI 生成模型来构建实时、用户交互的数字世界，超越被动的视频合成，这可能会重塑游戏内容创作或程序化叙事。引用的推文可能预示着一个研究或原型方向，尽管尚未披露基准测试或技术细节。** 热门评论回顾了 Hassabis 早期在 AI 驱动游戏开发领域的职业生涯，将其视为回归他利用 AI 进行交互式数字娱乐的初衷。人们对他将 AI 应用于游戏领域的变革性记录表示认可。

- 猜测集中在未来整个游戏世界、行为和系统可能由先进的 AI 模型生成，而不是传统的代码编写，将当前的编程方法比作像打孔卡（punch cards）这样的传统手动流程。这将使游戏开发的本质发生戏剧性转变，转向更高层次的设计，由 AI 根据需求合成复杂的环境和交互。
- 多位用户注意到目前已经存在早期但粗糙的生成式 AI 游戏公开 Web 演示，允许基础的移动和交互（例如通过方向键移动）。虽然这些演示还很初级，但它们的存在被视为先导指标，表明更复杂、完全可玩的 AI 生成世界可能很快就会到来。

### 2. Kontext 与 ComfyUI 高级参考及工作流技术

- [**Boosting Success Rates with Kontext Multi-Image Reference Generation**](https://www.reddit.com/r/StableDiffusion/comments/1lpqkkr/boosting_success_rates_with_kontext_multiimage/) ([Score: 176, Comments: 20](https://www.reddit.com/r/StableDiffusion/comments/1lpqkkr/boosting_success_rates_with_kontext_multiimage/)): **该帖子系统地探索了如何在 ComfyUI 中使用 Kontext 多图参考来改进图生图（image-to-image）属性迁移，重点是将服装从参考图迁移到模型图像。实验测试表明，简单地拼接图像或在白色背景上使用裁剪后的参考图会导致效果不佳或不一致，原因是上下文过多或丢失了相关线索。有效的方法是裁剪参考图，仅保留核心服装元素以及极少的身体线索（如手臂/腿部），从而将迁移准确率提升至 80% 以上。提供的工作流和 [Civitai 模型链接](https://civitai.com/models/1738322)展示了这一方法，并通过图像链接分享了输出结果的直接对比。** 评论者指出，虽然这种技术增强了 Kontext 的实用性，但它并不能取代 LoRA，特别是在面部一致性方面。此外还有关于工作流复现的技术讨论，特别是关于自定义宽高比和控制输出分辨率，以及与 BFL 指南一致的 Prompt 工程建议（即使用祈使句命令而非描述性 Prompt）。
    - 有一项关于对比 Kontext 多图参考生成与 LoRA 方法的讨论，特别是在面部特征的保真度方面——一位用户指出，与训练良好的 LoRA 相比，Kontext 仍然可能导致面部不够准确（例如 Sydney 的脸“相当走样”），而 LoRA 在迁移图像中能保留更高的细节。
    - 几位用户研究了工作流的细微差别：有人提出了关于在原始参考图宽高比不同时如何生成纵向分辨率输出的问题，因为 ComfyUI 的官方指南仅演示了图像拼接而没有明确的分辨率控制，这凸显了当前示例中的技术空白。
    - 围绕 ComfyUI 中的参考图链接（chaining）出现了技术争论：一些用户想知道使用拼接（stitched）还是链接的 Latent 节点会产生更好的效果，以及顺序（例如人物身份与服装）是否会影响迁移质量。其他的技术查询探索了 Prompt 工程——一位用户引用了 BFL，根据他们自己的测试，建议使用命令而非描述性 Prompt 以获得更好的结果。
- [**nunchaku your kontext at 23.16 seconds on 8gb GPU - workflow included**](https://www.reddit.com/r/StableDiffusion/comments/1lpn2wa/nunchaku_your_kontext_at_2316_seconds_on_8gb_gpu/) ([Score: 141, Comments: 59](https://www.reddit.com/r/StableDiffusion/comments/1lpn2wa/nunchaku_your_kontext_at_2316_seconds_on_8gb_gpu/)): **该帖子重点介绍了使用 MIT Han Lab 的 [ComfyUI-nunchaku](https://github.com/mit-han-lab/ComfyUI-nunchaku) 扩展在 8GB GPU 上运行 'kontext' 模型的 int4 版本，每张图像的推理时间约为** `~23s`**。关键参考资料包括所需的 [int4 kontext 模型权重](https://huggingface.co/mit-han-lab/nunchaku-flux.1-kontext-dev/tree/main)和分享的工作流。该工作流不需要速度 LoRA 或 SAGE Attention 模块。帖子还征求了关于将 Safetensors 模型快速转换为 int4 格式的信息。** 评论者注意到了性能差异（例如在具有相似或更高 VRAM 的 GPU 上为 `55s` 对比 `23s`），表明工作流或配置可能是一个因素。一位用户报告了与输入张量维度相关的弃用警告，暗示 Pipeline 的 PyTorch 预处理可能会有更新。
    - 一位用户报告实现了显著的运行速度提升——在 8GB GPU 上运行时间为 23 秒——而另一位用户在 RTX 3060 12GB 上则需要 55 秒，据称两人都使用了默认的 nunchaku 工作流，这突显了影响性能的优化或配置差异。

- 记录了一个关于 nunchaku 在 PyTorch 中弃用用法的错误：传递给 `img_ids` 的 3D torch.Tensor 应更改为 2D tensor，以符合更新后的库要求，这表明需要对工作流或代码库进行更新。
- 几位用户在 ComfyUI（便携版/Windows）中运行 nunchaku 时遇到了设置和依赖问题，包括模块未找到错误，以及在安装或导入 'nunchaku' 时明显的挑战，反映了环境配置和包管理中的常见障碍。
- [**Kontext Dev 上的 "Image Stitching" 与 "Latent Stitching" 对比**](https://www.reddit.com/gallery/1lpx563) ([Score: 138, Comments: 15](https://www.reddit.com/r/StableDiffusion/comments/1lpx563/comparison_image_stitching_vs_latent_stitching_on/))：**该帖子对比了 Kontext Dev 中提供的两种图像合成工作流：Image Stitching（合并多个角色/图像参考以合成新场景）和 Latent Stitching（在 latent space 内使用从次要图像提取的特征来覆盖或编辑主要图像）。提供了针对单图和双图场景的工作流 JSON 下载（[工作流文件](https://files.catbox.moe/q3540p.json)），并通过引用的 Stable Diffusion 工作流 Reddit 帖子提供了额外指导。核心实现功能是一个开关，可以在推理或编辑会话期间在两种合成模式之间无缝切换。** 技术导向的评论者注意到作者之前的 NAG 工作流取得了成功，并询问如何使用 Kontext 作为 ControlNet 机制来控制姿态——目前看来在强制执行精确姿态参考方面存在局限性，表明当前工具功能尚存空白。
    - 一位用户询问关于使用 Kontext 作为 ControlNet 来强制执行精确角色姿态的问题，并报告了在实现精确姿态控制方面的困难。这突显了目前 Kontext 与控制框架集成在精确空间调节（spatial conditioning）方面的局限性，可能指向缺乏直接的姿态强制工具或此类用例的必要指导。
    - 一位评论者询问 'image concatenate' 和 'image stitching' 之间的区别，暗示其操作差异存在模糊性。这反映了图像合成或复合工作流中术语的广泛混淆或重叠，表明需要对这些过程进行更清晰的定义或技术分解。
    - 围绕 'fluxkontextimagescale' 节点的使用展开了讨论，有报告称它在某些 stitching 方法下效果更好，且手动分辨率设置会根据所选技术影响质量。然而，在最佳搭配或工作流方面存在不确定性且缺乏共识，强调了对基准测试指南或文档的需求。
- [**我用 Flux Kontext 为妻子构建了一个简单的图像编辑 Web 应用——现已开源**](https://i.redd.it/iivpnnllliaf1.jpeg) ([Score: 126, Comments: 15](https://www.reddit.com/r/StableDiffusion/comments/1lq5evu/i_built_my_wife_a_simple_web_app_for_image/))：**图像展示了 'AI Photo Stylizer'，这是一个用于为照片应用艺术风格的 Web 应用——使用 Flux Kontext 构建，现已开源。UI 提供了选择风格配置（如 'Ghibli Soft' 和 'Comic Book'）、调整分辨率的控件，以及撤销/重做和下载等功能。链接的 [GitHub 仓库](https://github.com/TheAhmadOsman/4o-ghibli-at-home) 提供了项目源码供他人探索或贡献。** 一位评论者指出，与 GIMP 等传统工具相比，Flux Kontext 在图像编辑方面非常方便，但也提到其硬件要求是一个潜在的限制。
    - 几条评论讨论了 Flux Kontext 的硬件要求，一位用户指出，虽然它在某些任务上比 GIMP 提供更简单的图像操作，但目前还无法在所有硬件上高效运行，突显了与传统图像编辑器相比，低资源设备的限制。
    - 有建议提出支持在 Google Colab 上运行该应用，可能通过 Gradio 或 ngrok 集成，以扩大可访问性，并允许本地资源有限的用户利用云端计算进行图像编辑任务。

### 3. 社交媒体中的 AI 生成网红与虚拟角色

- [**这位网红并不存在**](https://i.redd.it/24k4hbp5rhaf1.png) ([Score: 412, Comments: 94](https://www.reddit.com/r/OpenAI/comments/1lq11ms/this_influencer_does_not_exist/)): **该图片显示了一条推文，详细介绍了一个 AI 生成的网红账号的迅速崛起，该账号在短短三个月内就积累了 13.8 万名粉丝。这突显了 AI 在生成照片级真实的虚拟角色方面日益成熟，这些角色可以在多张图像中保持一致的美学风格，从而构建出似真度高的在线身份。该帖子强调了一个更广泛的趋势，即“网红”可能既不是真人，也没有真实的粉丝，这引发了人们对社交媒体参与度的真实性和变现能力的质疑。** 评论者讨论了不仅是网红，连他们的粉丝也可能是虚假账号（机器人账号）的可能性，并指出人工或机器人驱动的影响力长期以来一直是社交媒体平台的一部分，并提到了过去对虚假账号的清洗以及在广告中使用人造受众的情况。
    - Impossible-Glass-487 提出了数字广告生态系统中机器人和虚假用户的重大历史背景，提到了“Twitter 大清洗”以及 Facebook 的类似努力。他们强调了社交平台此前如何在广告指标中包含不存在的用户或机器人，从而夸大广告触达率（例如 Facebook 的 lookalike audiences），并类比了 AI 生成的网红现在如何让个人用户也能制造类似问题，而不再是大型广告商的专利。
    - Positive-Raccoon-616 询问了在不同 AI 模型中一致匹配生成的网红面部的技术过程。这提出了 AI 生成内容中身份持久性（identity persistence）的挑战，可能涉及一致的 latent embeddings、conditional GANs 或跨模型参考对齐等技术，以确保不同的模型或系统能渲染出可辨识的角色。
- [**这位网红并不存在**](https://i.redd.it/0dzjz6h5qhaf1.png) ([Score: 984, Comments: 140](https://www.reddit.com/r/ChatGPT/comments/1lq0wmi/this_influencer_does_not_exist/)): **附图显示了关于社交媒体上 AI 生成网红的讨论，引用了一个 Instagram 个人资料拥有 13.8 万粉丝的案例，根据推文显示，该数字创作者并不存在。评论中的技术问题探讨了当前模型如何实现生成的面部和身体的一致性；从历史上看，这需要针对特定面部训练自定义模型或进行 LoRA 微调，这暗示了 AI 生成技术的进步，可能使用了更具动态性或提示词一致性（prompt-consistent）的方法。** 一位评论者对粉丝的真实性表示怀疑，认为许多人可能是机器人，而另一位评论者则淡化了这种新奇感，称大多数网红无论是否使用 AI 都不真实。
    - 一位用户询问了在为 AI 生成的网红生成一致的面部或身体方面的进展，并指出以前必须训练特定模型或在面部数据集上使用 LoRA (Low-Rank Adaptation) 技术。该问题旨在澄清是否存在能让当前生成模型实现更高一致性的新方法，以及哪些技术方案已经取代了这些旧技术。

---

# AI Discord 摘要回顾

> 由 Gemini 2.5 Pro Exp 生成的摘要之摘要之摘要
> 

**主题 1. AI 军备竞赛：新模型、性能对决与人才争夺战**

- **马斯克宣布 Grok-4 即将到来**：**Elon Musk** 宣布 **Grok-4-0629** 预计将在 *“7 月 4 日之后”* 发布，这引发了 **LMArena** Discord 频道关于其为何缺席 [LLM Arena 排行榜](https://lmarena.ai/) 的猜测。同时，竞技场中出现的一个神秘新模型被推测为 **Deepseek R2**，尽管据报道它*未能通过空间推理测试*。
- **新模型面临审查与冷遇**：**Amazon** 的 **Cypher Alpha** 模型在评估中表现不佳，**aider** 和 **LMArena** 社区的用户称其为一种退步，且*在所有方面都很糟糕*，这可能是由于受限的系统提示词（system prompt）所致。相反，虽然官方排行榜对其排名较低，但 **Claude Sonnet 7** 的代码编写速度受到了赞扬；**OpenAI** Discord 频道的一项对比分析发现，**Grok** 的 **AVM** 在解释 **Enso（圆相）和 Zen（禅宗）** 等复杂主题方面表现出色，超越了 **ChatGPT**。
- **Meta 与 Cursor 在高额竞价大战中挖角顶尖 AI 人才**：对 AI 专业知识的争夺日益激烈，据 [Amir Efrati 报道](https://x.com/amir/status/1940112288381641026)，**Cursor** 聘请了来自 **Anthropic** 的 **Claude Code** 团队的两名高级领导。更激进的是，据 [此推文线程](https://x.com/tanayj/status/1940137574141694046) 透露，**Meta** 据传开出了四年高达 **3 亿美元** 的薪酬方案，从 **OpenAI** 挖角 AI 研究员，第一年的薪资就超过了 **1 亿美元**。

**Theme 2. Innovations in Model Architecture and Fine-Tuning**

- **研究人员支持乘法微调优于 LoRA**：根据 [HuggingFace 上的论文](https://huggingface.co/papers/2506.07621) 作者介绍，一篇新论文引入了 **LoRMA**，这是一种使用*乘法*更新的参数高效微调方法，其**收敛速度比 LoRA 快 3 倍**。另外，**Unsloth AI** Discord 频道的讨论澄清了 Unsloth 的量化在*微调期间提供了更高的准确性*，这一优势与速度提升是不同的。
- **架构之争：SSMs 对阵 Transformers**：工程师们辩论了**状态空间模型 (SSMs)** 究竟代表了真正的范式转移，还是仅仅是对扩展后的 **RNNs** 的渐进式改进。**Yannick Kilcher** Discord 频道的讨论还重点介绍了一篇新论文 [Parallelizing Linear Transformers with the Delta Rule](https://arxiv.org/abs/2406.06484)，该论文引入了一种硬件效率高的算法，使 **DeltaNet** 的性能超越了 **Mamba** 和 **GLA**。
- **顶尖实验室涌现新训练方法**：**Meta** 的 [“Transition Matching” 论文](https://arxiv.org/abs/2506.23589) 声称优于 **Flow Matching**，尽管其作为提示词混淆攻击的动机在 **Eleuther** 社区引发了争论。该社区还热议了 **何恺明 (Kaiming He)** 关于 **Mean Flow Matching** 的演讲，有成员建议从 [研讨会视频的 2:43:32 处](https://www.youtube.com/watch?v=r-fgrZ0Ve74&ab_channel=VGMi) 开始观看。

**Theme 3. Developer Tooling, Workflows, and GPU Nightmares**

- **本地 LLM 变得更具创意（且在道德上存疑）**：**LM Studio** Discord 频道的用户支持使用本地 **LLM** 以保护隐私、节省成本并处理*道德存疑的内容*，使用场景涵盖从创意写作到 *Waifu 创建*。为了实现本地 **RAG**，成员们建议将 [OpenWebUI](https://docs.openwebui.com/tutorials/tips/rag-tutorial/) 作为功能性 API 服务器与 **LM Studio** 配合使用。
- **GPU 分析和构建难题困扰工程师**：在 **GPU MODE** Discord 频道中，有用户报告 **nsys 分析器** 在与 `torch.compile` 配合使用时会**停滞**，这是一个已知问题，在 [NVIDIA 论坛线程](https://forums.developer.nvidia.com/t/nsys-profile-pytorch-fails-under-torch-compile/332302/5) 中也有记录。其他令人沮丧的问题包括 **Triton** 的 nightly 构建版本已损坏数月，以及由于未分片参数导致 **FSDP 2.0** 中的 `full_state_dict` 加载失败。
- **Agent 框架和提示词旨在提升生产力**：**LlamaIndex** 宣布发布 **Workflows 1.0**，这是一个 [用于 Agent 系统的高效轻量级框架](https://www.llamaindex.ai/blog/announcing-workflows-1-0-a-lightweight-framework-for-agentic-systems)。在 **OpenAI** 社区，一个团队分享了一个 [战略项目矩阵提示词](https://chatgpt.com/share/68650970-55e0-8010-a80d-e4005f787b9c)，该提示词使用 **S.M.A.R.T.** 助记符来分类处理跨连接数据源的复杂任务。

**Theme 4. The Business of AI: Pricing, Outages, and Poaching**

- **模糊的定价模型令用户沮丧**：**Cursor** 的新定价模型因其 API 定价 **20% 的加价**以及不明确的*无限限速计划 (unlimited-rate-limited plan)* 而遭到抨击，部分用户称其*接近诈骗*。**Perplexity** 在推出昂贵的 **$200/月** 的 **Perplexity Max** 等级后，因撤销折扣码也面临抵制。
- **服务中断导致开发者手忙脚乱**：由于 DeepSeek 端的[配置错误](https://deepseek.com/)，**DeepSeek V3** 在 **OpenRouter** 上经历了短暂但令人恐慌的停机。**Hugging Face** 也遭遇了推理端点故障，并宣布关闭 **HuggingChat**，迫使用户在数据被删除前寻找备份对话的方法。
- **客户挖角丑闻冲击 OpenRouter**：一名用户报告称受到 **cometapi** 的拉拢，后者似乎利用公开的 **OpenRouter** 数据来识别并以更低的价格挖走高 Token 使用量的用户。该漏洞被追溯到用户在 HTTP headers 中发送其网站信息，这一做法在 [OpenRouter API 文档](https://openrouter.ai/docs/api-reference/overview#headers)中有记载。

**主题 5：护栏挑战与 AI 幻觉的风险**

- **“安全”模型被证明既笨拙又充满幻想**：**Amazon** 的 **Cypher Alpha** 模型在评估中表现不佳，可能是因为其 system prompt 要求它*只能表明自己由 Cypher Labs 制造*，这实际上*削弱 (nerf)* 了它的能力。该模型还会对自身的内部技术细节产生幻觉，在 **OpenRouter** 上被提问时，错误地声称自己拥有 *1.17 亿参数*和 *768 维嵌入维度 (embeddings dimension)*。
- **工程师辩论护栏有效性与模型危险性**：**OpenAI** Discord 中的讨论通过 **基于规则的奖励 (Rule-Based Rewards, RBR)** 视角分析了模型安全性，详情见 [OpenAI 博客文章](https://openai.com/index/improving-model-safety-behavior-with-rule-based-rewards/?utm_source=chatgpt.com)。共识是，虽然像 **DALL-E 3** 这样的美国模型在质量上领先，但它们的安全性过滤器往往无效，这可能会使模型*更加危险*。
- **本地 LLM 是傲慢且不可信的实习生**：各服务器（尤其是 **LM Studio**）的用户对 **AI 幻觉**的普遍存在发泄了不满，将本地模型比作语言能力出色但可靠性为零的*傲慢实习生*。核心结论是不论来源如何，都要怀疑输出结果，因为云端和本地 LLM 往往都已过时或纯粹是错误的。

---

# Discord: 高层级 Discord 摘要

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **护栏辩论引发关注**：一名成员发起了关于 **AI safety**、**guardrails**（护栏）以及模型欺骗行为的讨论，质疑当前基准测试的有效性，并促使了对模型训练和过滤的详细分析，特别是围绕 **Rule-Based Rewards (RBR)** [正如这里所讨论的](https://openai.com/index/improving-model-safety-behavior-with-rule-based-rewards/?utm_source=chatgpt.com)。
   - 该成员总结道，透明度对于创建更强大的护栏和防止被利用至关重要，特别是在通过现有图像模型进行实验以发现其优缺点时。
- **磨皮感图像引发争议**：成员们讨论了 **AI image generation** 中超写实输出的趋势，指出了 **DALL-E 3** 独特的磨皮（airbrushed）风格，并将其与 **Grok 3** 的能力以及 **4o** 的潜力进行了对比。
   - 一位成员使用 *Uneven flash lighting*（不均匀闪光灯照明）和 *Grainy fingerprint glare*（颗粒状指纹眩光）等修饰语生成了一张有趣的 *Zark Muckerberg* 图像。
- **Grok 的 AVM 在社交场景中表现优于 ChatGPT**：在对比分析中，**Grok's AVM** 在社交互动和解释复杂主题（如 **Enso and Zen**，圆相与禅）方面表现出色，超越了 **ChatGPT**。一些人指出 **Grok 4** 的开发将是下一代产品。
   - 一位成员对 **Mark Zuckerberg** 的视频做出了反应，指出他 *正试图掩饰那些低下的社交技巧，也许从那时起已经有所好转了*。
- **美国模型占据主导地位**：尽管安全方法各异，但像 **DALL-E 3**、**Midjourney** 和 **Grok** 这样的 **American image models** 凭借庞大的训练数据在图像生成质量上处于领先地位。
   - 共识是这些模型可能 *没有真正的安全过滤器* 或者 *过滤器无效*，这可能会使模型 *更加危险*。
- **战略项目矩阵提示词分流**：一个团队分享了一个 [Strategic Project Matrix prompt](https://chatgpt.com/share/68650970-55e0-8010-a80d-e4005f787b9c)，旨在利用连接的数据（Gmail, Calendar, Dropbox, Google Drive）管理跨地点和供应商的复杂义务，以构建一个用于分流和后续行动的矩阵。
   - 该提示词将 AI 塑造为 **Strategic Intelligence Analyst**（战略情报分析师），并遵循 **S.M.A.R.T.** (**Scan**, **Map**, **Assign**, **Recommend**, **Triage**) 记忆法框架，以确定项目的优先级并防止遗漏承诺。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro 文件上传功能扩展**：成员们讨论了 **file uploads** 作为 **Perplexity Pro** 和企业版账户功能扩展的情况，一位用户提到在被撤销前曾使用 **Samsung discount code**（三星折扣码）获得了更低的价格。
   - 一位成员讽刺地向支持部门呼吁，要求优先 *解决一个我想给他们送钱却送不出去的问题*。
- **Perplexity 打击折扣码**：Perplexity 撤销了 1 年期 Pro 订阅的 **discount codes**，包括来自三星和 HubSpot 的代码；部分用户在联系支持部门后找回了代码。
   - 这引发了用户的沮丧情绪，他们觉得自己因为寻找优惠而受到了惩罚。
- **Manus AI Gemini 进入免费聊天模式**：**Manus AI** 现在提供 [使用 **Gemini** 的无限免费聊天消息](https://manus.im/invitation/R9QW2MZDXXJJ4)。
   - 成员们对免费层级的限制感到好奇。
- **Perplexity Max 亮相**：Perplexity 推出了 **Perplexity Max**，这是一个每月 **$200** 的新层级，提供对高级 AI 模型的无限访问、[Comet browser](https://x.com/AravSrinivas/status/1940104947376562206) 的早期访问权限以及优质数据源。
   - 用户对高昂的价格和潜在的限制表示担忧，有人表示他们 *总是误点到仅限 Max 订阅的模型*，并且认为这是 *目前性价比最差的 200 美元订阅*。
- **API 子域名排除讨论**：一位用户寻求关于如何从 **Perplexity API** 调用中排除特定子域名的建议，旨在将搜索限制在仅以 *domain1.com/path1* 开头的内容。
   - 该用户旨在排除诸如 *domain1.com/path2* 之类的子域名，以更好地优化 API 搜索结果。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 的更新提升了速度**：用户反馈更新后 **Cursor 感觉更快了**，团队成员确认这 *并非心理作用*。
   - 然而，目前尚未披露更新的具体细节。
- **Cursor 的定价模式引发质疑**：用户对新的定价模式表示困惑和不满，理由是关于速率限制（rate limits）以及向 API 定价过渡的 **缺乏透明度**。
   - 一些用户认为在 API 定价基础上 **加价 20%** *近乎诈骗*，尤其是所谓的 *“无限量但限速”方案（unlimited-rate-limited plan）*。
- **消息队列系统增加了操作阻碍**：最新更新引入了 **队列系统** 来按顺序处理消息，这为中断进程增加了一个新步骤。
   - 用户发现中断正在运行的进程变得很 *繁琐*，现在必须点击停止按钮。
- **Claude Code 胜过 Gemini，但成本更高**：在许多评估中，**Claude Code** 的表现优于 **Cursor**，尽管 **Anthropic API** *要昂贵得多*。
   - 虽然 **Gemini 2.5 Pro** 模型展现出了潜力，但据报道它在 *编辑文件方面表现糟糕*。
- **后台 Agent 在 GitHub 上创建分支**：有用户报告称 **Background Agents** 会在 github.com 上创建一个新分支，而 Cursor Chat 则在本地创建文件，一位成员对此表示不满，问道：*为什么我要切换分支？我这辈子都不想要第二个分支。*
   - 成员们强调他们深度依赖 git，并建议花时间 *询问 Agent 应该如何为你处理 git 操作*。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 量化提升准确率**：一位用户询问对一个 101 层的模型进行 **Unsloth-ing** 处理比直接量化能带来多少性能提升，得到的澄清是主要益处在于 *微调（fine-tuning）期间准确率的提高*。
   - 他们确认 Unsloth 的 4bit 动态主要是为了抵消训练期间与 fp16 的差异，并指出 **bartos quants** 也很有效，特别是对于 **GGUF** 微调。
- **昇腾 GPU 在 MoE 模型上表现出色**：一个完全在 **昇腾 GPU (Ascend GPUs)** 上训练、具有优化架构的通用 **MoE 模型** 已发布在 ArXiv 上：[https://arxiv.org/pdf/2505.21411](https://arxiv.org/pdf/2505.21411)。
   - 该模型进行了包括 RL（推理模型）在内的端到端训练，其基准测试结果与 **Qwen3-32b** 相似，展示了在利用 **昇腾 GPU** 处理大语言模型（LLM）方面的显著进展。
- **Qwen3-32B 合并故障显现**：用户报告在尝试保存合并后的 16-bit **Qwen3-32B-unsloth-4bit** 模型时遇到 `RuntimeError`，问题追溯到 attention 层的维度不匹配。
   - 该问题源于从本地 **4bit** 检查点开始，绕过了合并所需的 **16bit** 基础模型的下载；解决方法是手动指向已下载的基础模型。
- **Intel Arc 定价引发争论**：围绕 **Intel Arc A770** 显卡的定价展开了讨论，有说法称电话核实后的价格为每张卡 **3000 美元**。
   - 针对其价值，人们将其与 **5000 美元** 的多卡系统以及定价在 **4500-5000 美元** 的 **RTX Pro 5000** 进行了对比并表示担忧。
- **ChessFish.io 助力国际象棋**：一位成员推广了 [ChessFish.io](https://www.chessfish.io/)，这是一个用于 *分析、学习和休闲对弈* 的国际象棋网站。
   - 该网站可以免费试用，无需注册账号。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Cypher Alpha 在评估中表现不佳**：成员们发现 **Amazon** 的模型 **Cypher Alpha** 表现糟糕，这可能是由于其严格的系统提示词（system prompt）要求它只能标识为由 **Cypher Labs** 开发。
   - 系统提示词施加的限制可能削弱（nerf）了模型的能力。
- **Grok-4-0629 预计在独立日后发布**：**Elon Musk** 宣布 **Grok-4-0629** 预计将在“7 月 4 日之后”发布，这引发了关于它为何未出现在 [LLM Arena](https://lmarena.ai) 的猜测。
   - 一些用户推测该模型未在 Arena 中测试是因为其表现可能不尽如人意，而 **xAI** 正在尝试通过暴力尝试（bruteforcing）以获得更好的结果。
- **OpenAI 权重开放模型的许可证限制**：有推测称 **OpenAI** 的权重开放模型许可证可能会限制与其 API 的直接竞争。
   - 用户讨论了开发一种*可以在 4090 上运行的高质量模型*的可能性，并将其与 [R1 进行了对比](https://x.com/AiBattle_/status/1940139539525419512?t=g8LAuWUNXwvdN9fxs6IvQQ&s=19)。
- **Gemini CLI 速率限制曝光**：社区讨论了 **Gemini CLI** 的速率限制，该限制允许**每分钟 60 次请求**和**每天 1000 次请求**。
   - 超过这些限制后，用户将被切换到 **Flash**，视觉和图像输入/输出功能预计将在 7 月 4 日之后推出。
- **Deepseek R2 成为 Arena 神秘模型**：有推测认为 Arena 中的新模型是 **Deepseek R2**，可能是一个混合模型。
   - 然而，该模型*未能通过空间推理测试*。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **DeepSeek V3 出现短暂离线故障**：由于其内部的[配置错误](https://deepseek.com)，**DeepSeek V3 0324** 模型经历了约 **15 分钟** 的停机。
   - 该模型现已恢复在线，但此次事件引发了社区的恐慌和担忧。
- **Cypher 模型对参数细节产生幻觉**：用户发现 **Cypher** 模型在被问及时，总是输出关于其参数的详细信息，如 *1.17 亿参数*和 *768 嵌入维度*，尽管这些很可能是幻觉。
   - 成员指出，该模型基于改进的 **GPT-2 architecture**，旨在对潜在有害的提示词做出适当回应，但似乎缺乏对自己技术规格的准确自我认知。
- **CometAPI 疑似利用 OpenRouter 数据诱导客户**：一名用户在 Facebook 上被 **cometapi** 联系，对方声称知道他们正在使用哪种 LLM 模型并提供更便宜的服务，这引发了 **OpenRouter** 数据可能被用于挖掘客户的怀疑。
   - 正如 [OpenRouter API 文档](https://openrouter.ai/docs/api-reference/overview#headers)所述，该用户通过 HTTP headers 将其网站和标题发送给 OpenRouter，使其在顶级 Token 用户列表中可见，从而暴露了他们的模型使用情况。
- **DeepSeek 过载导致负载均衡问题**：一些用户遇到了 **DeepSeek** 响应缓慢或超时的问题，特别是在太平洋标准时间下午 3 点左右，导致即使响应未送达也会产生计费问题，这表明 **DeepSeek** 的负载均衡机制正在介入。
   - 社区成员建议尝试其他模型，如 **Mistral 3.2 24B small** 或 **Grok**，并指出 **DeepSeek** 可能已过载，用户可能需要寻找延迟更低的模型。
- **AI 驱动的词汇学习应用**：一名成员在 [mnemix.arnost.org](https://mnemix.arnost.org) 构建了一个免费的 AI 驱动词典应用。
   - 该应用可以生成解释、示例和测验，帮助用户更快、更有效地学习词汇。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **本地 LLM 也会产生幻觉**：成员们对本地 LLM 中的 **AI hallucinations**（AI 幻觉）表示沮丧，将其比作语言能力出色但*傲慢的实习生*，并强调需要对其输出保持不信任，因为云端 LLM 总是*过时的*。
   - 讨论强调了不要盲目信任 LLM 输出的重要性，无论它们是基于云端还是本地。
- **本地 LLM 并不总是需要互联网**：用户讨论了本地 LLM 联网的必要性，指出 **LM Studio** 对于已下载的模型可以离线运行，仅在初始下载和运行时更新时需要互联网。
   - 对话提到 **LM Studio** 可以通过 **MCP (Model Context Protocol)** 使用互联网。
- **本地 LLM 变得富有创意**：尽管最初持怀疑态度，成员们强调了本地 LLM 的几个用例，包括降低成本、隐私保护、实验以及处理*道德存疑的内容*。
   - 提到的创意应用涉及创意写作、信息提取、日历创建、游戏自动化、信息整理，甚至包括 *Waifu 创作*。
- **RAG 定义得到澄清**：成员们在本地 LLM 的背景下解释了 **RAG (Retrieval-Augmented Generation)**，将其定义为将文本转换为数值 embeddings 并将其集成到 LLM 知识库的过程。
   - **RAG** 可用于研究，通过将相关的段落或 HTML 文件复制粘贴到 context 中供 LLM 参考。
- **推荐 OpenWebUI 用于本地 RAG**：成员们建议将 [OpenWebUI](https://docs.openwebui.com/tutorials/tips/rag-tutorial/) 与 **LM Studio** 配合使用来实现 **RAG**，因为它作为一个 API server 提供了更多功能。
   - 一位成员确认，在 VPS (Virtual Private Server) 上设置它来管理请求队列效果*非常棒*。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HuggingChat 告别**：成员们讨论了 **HuggingChat** 的关停，并想知道是否会出现类似的开源替代方案，考虑可能会启动一个 **chat-ui** 的备用实例。
   - 用户正在分享关于如何在 **HuggingFace** 于两周内删除存储在 **HuggingChat** 数据库中的所有用户数据之前**备份对话**的技巧。
- **GPT4All 模型寻求设计输入**：一位用户请求一个可用于 **GPT4All** 的模型，能够回答建筑设计问题并解释上传的楼层平面图 **JPG**。
   - 另一位用户建议*需要多模态模型来解释图像*，并链接了关于 **GPT4All** 中多模态模型支持的相关 **GitHub** issue（[issue 3540](https://github.com/nomic-ai/gpt4all/issues/3540), [issue 1568](https://github.com/nomic-ai/gpt4all/issues/1568)）。
- **Step1 作为 Lovable/Cursor 替代方案出现**：**Step1** 是一款类似于 **Lovable/Cursor** 的构建落地页工具，因其易用性而受到关注；一位用户在 **15 分钟** 内构建了一个落地页。
   - 一位用户注意到有人利用自己的侧边项目通过 **Step1** 构建了产品并在 **Fiverr** 上出售，强调了变现机会。
- **LoRMA 乘法适配方案发布**：一篇论文介绍了 **LoRMA**，这是一种参数高效微调（fine-tuning）范式，它将传统的*加法*更新替换为*乘法*更新。
   - 据作者称，**LoRMA** 实现了**比 LoRA 快 3 倍的收敛速度**，并在各种理解和生成任务中具有竞争力的性能；该论文可在 [HuggingFace](https://huggingface.co/papers/2506.07621) 上查阅。
- **HF Inference Endpoints 出现故障**：用户报告了 **Hugging Face inference endpoints** 的问题，其中一人分享了一个关于停机讨论帖的[链接](https://discuss.huggingface.co/t/are-inferenceclient-s-down/161485)。
   - 一位用户在跟随 Agent 课程的代码操作时，遇到了 **Llama-3.3-70B-Instruct** 的 **HTTPError 400** 错误，追踪后发现是 *Bad Request: The endpoint is paused*（错误请求：端点已暂停）。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Wayfarer Labs 目标 100 FPS**：**Wayfarer Labs 的 CSO** Shahbuland Matiana 将在布朗大学展示在大型 **Diffusion World Models** 中实现 **100 FPS** 的策略（[Zoom 链接](https://brown.zoom.us/j/8536695003)）。
   - 讲座将分解 **diffusion world model pipeline**，识别瓶颈，并提出在长上下文长度下达到 **100 FPS** 的缓解方法。
- **Meta 的匹配方案主张引发关注**：**Meta** 关于 ["Transition Matching"](https://arxiv.org/abs/2506.23589) 的论文据称优于 **Flow Matching**，但其关于提示词混淆攻击（prompt obfuscation attack）的动机引发了争论。
   - 该攻击需要查询拦截和修改，且仅需黑盒访问。根据 [结果](https://cdn.discordapp.com/attachments/747850033994662000/1389692080275849388/image.png?ex=6866dc9b&is=68658b1b&hm=e4ff044077222da3811712070a89f156509cb64c1076648f7cd105e9f8ba8fef) 显示，随着混淆程度加深，准确率会有所下降。
- **懒加载（Lazy-Loading）提升启动速度**：在 `__init__.py` 中对 `simple_evaluate` 和 `evaluate` 进行懒加载，并将 `lm_eval` 的导入移至 `cli_evaluate` 内部，极大地缩短了 **lm-evaluation-harness 库**的脚本启动时间。
   - 启动时间从 `3.61s user 0.92s system 50% cpu 8.986 total` 降低至 `0.04s user 0.01s system 98% cpu 0.051 total`。
- **何恺明（Kaiming He）的讲座引发思考**：一名成员分享了一个包含 **Kaiming He** 的 [YouTube 工作坊链接](https://www.youtube.com/watch?v=r-fgrZ0Ve74&ab_channel=VGMi)，并强调了他对 **Mean Flow Matching** 的描述。
   - 具体建议从 **2:22:01** 开始观看何恺明的讲座，并在 **2:43:32** 处查看他对 **Mean Flow Matching** 的描述。
- **黑客松需要大量支持**：随着 **Open Research Hackathon** 将于 8 月举行，鼓励社区研究员在 [research 频道](https://discord.com/channels/729741769192767510/747850033994662000/1386431466447311000) 提出项目。
   - 这也包括 **Interpretability-General** 频道。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **相似性搜索中动态 K 值的争论**：成员们讨论了在 SentenceTransformers 的开放式问答中，是否应该动态选择 **top-k 检索**中的 'k' 值，而不是固定 'k'，以避免随意性。
   - 讨论涵盖了潜在的解决方案，从使用**相似性阈值**过滤结果，到将阈值处理与带上限的 top-k 方法相结合。
- **RNNs/LSTMs：深挖还是略读？**：成员们辩论了在开始学习 NLP 时，是应该深入研究 **RNNs 和 LSTMs** 还是仅仅略读。
   - 虽然一些人建议关注 **RNNs** 和 **随时间反向传播 (BPTT)** 的概念，但另一些人认为 LSTMs 已经过时且与未来无关，更倾向于像 **GRUs** 这样的简化版本。
- **通用函数逼近（Universal Function Approximation）：事实还是噱头？**：成员们讨论了一种观点，即在现代规模下，所有架构（特别是稠密前馈架构）都是**通用函数逼近器**。
   - 一名成员链接到了[一篇关于通用函数逼近器的论文](https://arxiv.org/abs/1906.06766i)，并指出这一观点在社区内可能是一个激进的看法，尤其是在涉及状态空间模型（SSMs）时。
- **SSMs：范式转变还是渐进式改进？**：成员们辩论了 **状态空间模型 (SSMs)** 是否代表了根本性的范式转变，或者在 LLMs 中规模化后，它们的表现是否仅在 **RNN** 的误差范围内。
   - 一名成员提到由于 **Flash-Attention** 作者的贡献，对 **SSMs** 寄予厚望，期待在 CUDA 低层和模型架构高层都能有所突破。
- **Delta 规则加速线性 Transformer 并行化**：讨论涉及了[通过序列长度上的 Delta 规则并行化线性 Transformer](https://arxiv.org/abs/2406.06484)，重点是 [RWKV-7 论文](https://arxiv.org/pdf/2503.14456#page=18) 中等式 18 的并行化。
   - 该论文引入了一种硬件高效的算法，用于使用 Delta 规则训练线性 Transformer，使得 **DeltaNet** 能够扩展到标准语言建模设置，性能优于 **Mamba** 和 **GLA**。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **nsys 在使用 Torch Compile 时停滞**：一位用户报告称，当 **nsys** 分析器与 **torch.compile** 配合使用时会发生**停滞**，即使使用了显式的 NVTX 范围也是如此。该用户引用了一个[相关的 NVIDIA 论坛帖子](https://forums.developer.nvidia.com/t/nsys-profile-pytorch-fails-under-torch-compile/332302/5)，该帖子声称 **nsys** 应该可以正常工作。
   - 该用户提供了一个重现该问题的[代码片段](https://cdn.discordapp.com/attachments/1389720693704233051/1389722323082280981/image.png?ex=6866f8c5&is=6865a745&hm=8b2e349f1af1cdf1f19125827205c2378b5fd80c35f8c7ecdb0dc4843cd52ba9)。
- **Triton Nightly 构建已损坏数月**：一名成员质疑为什么 **Triton** 的 **nightly wheel 构建**已经损坏了数月，而另一名用户指出，通过 `TensorDescriptor` 使用 **TMA** 需要从源码构建，这在*租用实例时非常令人恼火*。
   - 他们强调了修复 **nightly wheel 构建**的重要性，特别是因为示例依赖于最近的功能，而源码构建可能需要大约 **1 小时**。
- **SWE 在 MLE 中找到了灵魂的宁静**：一位成员分享了他们从 **SWE 角色转型为 MLE 角色**的经历，理由是工作与生活的平衡以及成就感得到了改善，并指出**与最终用户的直接互动**极大地提升了他们的成就感。
   - 他们强调了让自己周围环绕合适的人的重要性，并与之前角色中感觉被困在[功能工厂 (feature factory)](https://cutle.fish/blog/12-signs-youre-working-in-a-feature-factory)的感觉形成了对比。
- **FSDP 2.0 的 state_dict 停滞**：一位用户报告称，由于**未分片参数 (unsharded parameters)**，他们的 `full_state_dict` 在前向传播 (forward pass) 后，使用 `torch.distributed.checkpoint.state_dict.set_model_state_dict` 在 **FSDP 2.0** 中停止加载。
   - 在前向传播之后，参数保持未分片状态，导致 `.parameters()` 不返回 **DTensors**，从而阻碍了 state dictionary 的加载。
- **对 Cutlass Kernel 性能的思考**：一位成员询问是否存在可以根据模板参数配置预测 **Cutlass kernel** 性能的[成本模型 (cost models)](https://link.to/cost-model-info)。
   - 许多深度学习编译器依赖于基于分析的自动调优 (profile-guided autotuning) 来进行 **Cutlass kernel 选择**，该成员质疑 **Cutlass** 的元编程是否应该支持基于分析成本模型的 kernel 选择。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Cursor 招揽 Claude Code 负责人！**：[Amir Efrati 报道](https://x.com/amir/status/1940112288381641026)称，Anysphere/Cursor 已经从 **Anthropic 的 Claude Code 团队**聘请了两名高级领导，而 Anthropic 的年度经常性收入 (**ARR**) 已达到约 **40 亿美元**，自年初以来增长了四倍。
   - 这一举动标志着 AI 人才格局的重大转变，像 Anysphere/Cursor 这样的小型公司正在从行业巨头那里吸引顶级人才。
- **Meta 开启巨额资金挖人行动！**：据[此推文线程](https://x.com/tanayj/status/1940137574141694046)透露，Meta 正提供高达 **4 年 3 亿美元**的薪酬方案，从 OpenAI 挖走 AI 研究人员，第一年的薪酬就超过了 **1 亿美元**。
   - 这种激进的人才获取策略凸显了对 AI 专业知识和资源的激烈竞争。
- **Luma 发布华丽的视频工具！**：Luis C 宣布 **Luma Labs AI 的“Modify Video”工具**现已在 [Replicate](https://xcancel.com/lucataco93/status/1940113275221344566) 上可用，该工具允许用户重新混合任何视频并更改帧的风格。
   - 该工具允许用户使用 AI 重新混合任何视频并更改帧的风格，展示了 AI 在内容创作方面日益增长的能力。
- **Perplexity 推出高级订阅计划！**：Perplexity.ai 推出了 **Perplexity Max**，这是一个新的高级订阅层级，提供无限的 Labs 查询、访问更广泛的前沿模型，以及提前体验即将推出的产品（如 [Comet](https://xcancel.com/perplexity_ai/status/1940443479710257226)）。
   - 这一新层级反映了对先进 AI 工具和服务日益增长的需求。
- **Latent Space 发布最新的 LLM 学习内容！**：Latent Space 发布了新的一集，与 Jack Morris 讨论了[语言模型的信息论](https://xcancel.com/latentspacepod/status/1940453495465038067)，涉及**学习即压缩**的概念，这是 **Ilya Sutskever** 倡导的观点。
   - Morris 基于他的 **AI 博士经验**和广受好评的论文，倡导一种涵盖 **V-information、嵌入 (embeddings) 和 LLM 逆向/提取**的“新型信息论”。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP 启发 O'Reilly 著作**：一位作者正在撰写一本关于 [MCP 的 O'Reilly 书籍](https://www.oreilly.com/)，并在 Discord 服务器中寻求见解。
   - 一位成员对 O'Reilly 书籍表示怀旧，同时指出 *当今世界变化太快，它已不像 10 年前那样具有相关性*。
- **文档决策主导讨论**：成员们辩论了存储公司文档供 LLM 使用的最佳方法，偏好从 **markdown 文件** 到 **Google Docs** 不等。
   - 建议使用用于 PKM CRUD 的 **MCP server**、[HackMD](https://hackmd.io) 和 **Obsidian** 等解决方案来进行高效的文档管理。
- **徽章和 Inspect 问题困扰平台**：用户报告在 Docker 文件更新后，其 MCP server 的 **Inspect 选项** 和用于安全与质量的 **徽章更新** 持续出现问题。
   - 这些问题被证实是普遍存在的，表明更新过程中可能存在 Bug 或疏忽。
- **Claude 触发代码提交**：开发人员正在尝试使用 **Claude Hooks**，以自动执行由 **Jane docs** 修改触发的 git 操作。
   - 一位成员一直在使用 [context7](https://context7.ai/) 来增强 Claude 的功能。
- **MCP-Routing 重新构想请求路由**：一位成员提议建立一个 **MCP-Routing 层**，以智能管理不同 LLM 和 MCP 工具（如 Context7）的 context window 大小。
   - 讨论还考虑了 MCP server 是否应该过渡到 **REST APIs**，以减轻 hallucinations 并提高效率。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Cypher Alpha 模型表现不佳**：成员们发现 **Cypher Alpha** 在编码任务中的表现不如 **Qwen3 30B A3** 等模型，称其退步到了旧标准。
   - 一位用户幽默地将其评价为 *在所有方面都很差*，认为它是最近测试过的最差模型之一。
- **Sonnet 7 抢尽风头**：用户对 **Claude Sonnet 3.7** 和 **Sonnet 4** 在排行榜上的低排名表示惊讶，同时报告了使用 **Sonnet 7** 配合 **Thinking 8k** 进行编码的积极结果，并称赞了速度提升。
   - 这种情绪表明 **Sonnet 7** 的表现超出了预期，掩盖了早期围绕 **Sonnet 4** 的炒作。
- **Openrouter Oauth 遇到问题**：一位用户报告了新版 **Openrouter Oauth** 的问题，指出 Oauth 弹出窗口未能出现，影响了可用性。
   - 故障排除建议包括重新设置不带括号的 API key。
- **Aider API Key 令人头疼**：一位用户在使用 **Aider** 时面临持续问题，系统反复识别 API key 缺失。
   - 建议重新安装 **Aider** 作为 API key 问题的潜在修复方案。
- **解析 Aider 中的 /architect 模式**：一位用户寻求关于实施 Aider `/architect` 模式计划的指导，观察到更改并未直接反映在 repo 中，直到使用 `/code` 立即启动编辑。
   - 观察表明，编辑在启动 `/code` 后立即开始，这与按下 Enter 键后才开始编辑的预期相反。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 被设置为双重用途工具**：一位成员正在尝试将 **NotebookLM** 设置为个人每日日志，用于记录**感悟、媒体、Gemini 聊天和建议**，并将其作为文章、想法和参考资料的可搜索笔记数据库。
   - 他们计划将 **Google Docs** 作为隐私和数据控制的单一事实来源，同时探索替代输入方法以构建一个具有韧性的系统。
- **音频概览功能解释书籍**：一位成员使用 **NotebookLM 的音频概览（Audio Overview）功能**来解释他们正在创作的书籍，并指出该功能主要是在*为他们进行讲解*。
   - 虽然没有提供关于书籍类型或内容的进一步细节，但它强调了**音频功能**的实用性。
- **NBLM 源选择受到质疑**：一位用户询问，在从特定源创建思维导图后，重新激活所有源是否会导致讨论话题从所有源中提取，还是仅从最初选择的源中提取。
   - 澄清源选择行为可以帮助用户管理 **NotebookLM** 使用的信息范围。
- **播客生成想法被提出**：一位用户请求关于如何使用 **NotebookLM** 的 *Audio overview* 功能生成更长播客的建议。
   - 该请求寻求社区的意见和建议，以优化 **NotebookLM** 内的**播客生成**。
- **NBLM 专业版与免费版的博弈**：一位新用户询问了 **NotebookLM** 中专业版和免费版账户之间的区别。
   - 了解每种账户类型的特性和限制对于新用户确定最佳选择至关重要。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 用户渴望新的开源模型**：用户表达了对新的或更新的 **Cohere 开放模型权重发布**的期待，特别是强调 **tool/agent 框架**和现代架构的模型。
   - 一位用户指出 **CMD-R** 模型即将迎来一周年（08-24），并建议 Cohere 发布新的权重将超越竞争对手。
- **测试密钥限制了 Cohere Embeddings**：**Cohere embedding 模型**可以通过**测试密钥（trial key）**访问，但用户报告称，与生产密钥相比，测试密钥存在严格的**速率限制（rate limits）**和**每月使用上限**。
   - 虽然**测试密钥**和**生产密钥**解锁的功能相同，但**测试密钥的每月限制**构成了显著的约束。
- **研究人员涌向 Cohere 公会**：Sriram 正在研究**强化学习（reinforcement learning）**和**设备端安全 AI**；Dojo 专注于**语义分割（semantic segmentation）**以及将 **NLP 架构**应用于计算机视觉；Oraib 正在整合 **AI 应用**，通过**卫星图像**进行**水质监测**。
   - Abdullah Abbasi 介绍自己是一名 **Agentic AI** 学生和**平面设计师**。
- **引发关于 Secure ML 和隐私保护的辩论**：在 **#research** 频道中，引发了围绕 **Secure ML** 和**隐私保护（Privacy Preservation）**的讨论。
   - 一位用户询问了关于 **Secure ML** 和**隐私保护**的问题，要求详细阐述 **Secure ML** 的含义。
- **用户争相寻找 ML 夏季训练营频道**：多位用户正试图寻找 **#ml-summer-school** 频道，并引用了 [Cohere Labs 社区夏季训练营网站](https://sites.google.com/cohere.com/coherelabs-community/community-programs/summer-school)。
   - 他们尚未找到指定的频道，并请求协助定位。



---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **探索 Mojo 的 Origin Tracking**：一位用户询问了 **Mojo Origin Tracking 系统**（借用检查器）的实现方式，其他成员分享了关于 [Ownership and life cycles](https://docs.modular.com/mojo/manual/values/ownership)（所有权与生命周期）以及 [lifetimes and references](https://docs.modular.com/mojo/manual/values/lifetimes)（生命周期与引用）的文档，此外还有 [一段相关的 YouTube 视频](https://www.youtube.com/watch?v=9ag0fPMmYPQ)。
   - 另有提到，该语言的创始人计划最终就此主题进行演讲。
- **揭秘 Mojo Structs vs. Classes**：一位成员询问了 **Mojo structs** 与 **classes** 之间的细微差别。
   - 另一位成员分享了 [官方文档](https://docs.modular.com/mojo/manual/structs#structs-compared-to-classes) 的链接，其中概述了这些关键区别。
- **探讨 Mojo 的 GPU Barrier 需求**：一位成员在 [GPU puzzles](https://puzzles.modular.com/puzzle_14/tiled.html#tile-processing-steps) 页面的 **GPU puzzles** 中，就 **矩阵分块（matrix tiling）问题** 中第二个 Barrier 的必要性寻求澄清。
   - 有人建议在论坛发帖，以便专家提供更详细的解释。
- **Mojo 拥抱依赖类型（Dependent Types）**：一位成员询问随着语言的成熟，**Mojo** 是否会包含更高级的概念，如 **graded types**。
   - 另一位成员回答说，**Mojo** 确实正在向 **dependent type system**（依赖类型系统）发展，在功能与编译时间及运行时性能之间取得平衡。
- **稳定版修复 Mojo 离线推理问题**：一位用户在 M1 Mac 上使用 Nightly 版本，按照 [Modular Max 离线推理文档](https://docs.modular.com/max/serve/offline-inference/) 使用 [llm4decompile-v1.5/1.3B-Q6_K 模型](https://builds.modular.com/models/llm4decompile-v1.5/1.3B-Q6_K) 时，遇到了与不支持的 `quantization_encoding` 相关的 `ValueError`。
   - 该用户报告称，切换到 Mojo 🔥 的稳定版本（Stable Build）解决了 `quantization_encoding` 问题，模型运行符合预期。



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **AI 爱好者寻求在线志同道合的伙伴**：一位用户表示在 **Discord** 和 **Reddit** 等平台上寻找志同道合的人讨论 AI 具有挑战性，希望能与 *志趣相投* 的人建立在线友谊。
   - 他们强调了在 AI 领域寻找具有共同小众兴趣的同伴的困难。
- **Manus AI 现在提供无限免费聊天**：[Manus AI](https://manus.im/invitation/R9QW2MZDXXJJ4) 现在提供基于 **Gemini** 的 **无限免费消息**。
   - 一位用户询问了自其上一个版本（本质上是带有工具的 **Claude**）以来的改进情况，成本是主要关注点。
- **NFT 项目被标记为潜在诈骗**：一位用户询问了在 **Twitter** 上推广的一个 **NFT** 项目的合法性，怀疑其为 *诈骗（scam）*。
   - 这一询问引发了对 **NFT** 领域潜在欺诈活动的担忧。
- **研究人员为独立研究寻求导师**：两名独立用户正在寻求导师，以开始进行 **独立研究（independent research）** 项目。
   - 鼓励感兴趣的导师通过私信联系，为他们的研究工作提供指导和支持。



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All 面临平面图挑战**：一位成员正在寻找与 **GPT4All** 兼容的模型，该模型可以从上传的 **JPG** 平面图中解释建筑设计。
   - 针对 **GPT4All** 的图像上传能力以及 **ChatGPT** 由于急于提供帮助而可能产生的不准确性提出了担忧，并指出这些模型很难准确地 *讨论* 物体。
- **图像识别能力引发讨论**：成员们讨论了 **ChatGPT** 识别 *树木*、*建筑物*、*面部* 和 *人物* 等物体的能力。
   - 一位成员表示，**ChatGPT** 在没有系统提示词（system prompt）的情况下，通过一张图片正确识别了未标记的出口，但由于可能存在的不准确性，这一说法遭到了其他成员的质疑。
- **LM Studio 意外支持图像输入**：一位成员指出，只要模型正确，**LM Studio** 就可以接受图像。
   - 一位用户展示了 **ChatGPT** 处理平面图图像后的输出，尽管输出的具体细节及其准确性并未详述。

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaExtract 自动生成 Schema**：新功能 **LlamaExtract** 现在可以根据文档和/或 Prompt 自动生成 Schema，从而减轻了手动构建的需求，详见[这条推文](https://twitter.com/llama_index/status/1940123273981043035)。
   - 只需要一个文档和一段描述即可。
- **LlamaCloud 扩展企业级 RAG**：一篇新博客文章详细介绍了 **LlamaCloud 扩展企业级 RAG 的 4 种方式**，分享了在为大规模企业工作负载扩展 LlamaCloud 时学到的经验，[这条推文](https://twitter.com/llama_index/status/1940440399690248669)中有进一步解释。
   - 该文章旨在帮助其他构建大规模文档索引和检索的人了解未来的发展方向。
- **LlamaCloud 索引和检索图像**：你现在可以从 **LlamaCloud Indexes** 中检索图像和说明性图表以及文本，非常适合演示文稿和报告，如[这条推文](https://twitter.com/llama_index/status/1940485676371530035)所示。
   - 启用此功能只需切换“*Multi-modal indexing*”选项。
- **LlamaIndex 发布 Workflows 1.0**：LlamaIndex 宣布发布 **Workflows 1.0**，这是一个用于 Agentic 系统的轻量级框架，详见[他们的博客](https://www.llamaindex.ai/blog/announcing-workflows-1-0-a-lightweight-framework-for-agentic-systems)。
- **环境变量是 OpenAI API key 的关键**：一位成员在 LlamaIndex 的 No-Code UI 中嵌入 **OpenAI API key** 时需要帮助，另一位成员指出 *LlamaIndex 只寻找环境变量*。
   - 设置一个名为 `OPENAI_API_KEY` 的环境变量即可解决问题。



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Prompt 需要团队关注**：一名用户在 **general** 频道请求 Manus 团队修复他们的 Prompt，或者为包含 image 单词的消息添加一个微型 Prompt。
   - 该用户标记了一名团队成员 <@1352272661589524572> 以寻求 Prompt 方面的帮助。
- **MCP Server 请求 Claude Opus 4**：一名用户标记了 Manus 团队，希望使用 **Claude Opus 4** 帮助构建 **MCP Server**（被描述为 *像一级方程式赛车一样*）。
   - 该用户开玩笑说：*它每天都会在维修站里，因为你会一直在研究如何让它工作。*
- **Qwen 3, 32B 模型开箱即用表现出色**：一名用户报告了 **Qwen 3, 32B 模型** 在 **LM studio** 中成功的开箱即用功能。
   - 该用户分享了一个与此设置相关的[文件](https://files.catbox.moe/5gf51x.txt)。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 获得 Haldie 风格的可视化**：一名成员在 Python 后端展示了一个“haldie 风格”的可视化，展示了内存访问模式，使用绿色表示读取，红色表示写入，并[分享了演示](https://cdn.discordapp.com/attachments/1068976834928193609/1389987658020819056/haldie_viz.mov?ex=68669e62&is=68654ce2&hm=8458418f0dc82f9dcac83e6a7eaad13d1d4b79101c26945629d63322948c6bd6&)。
   - 该可视化专为 Metal 设计，上方显示 **shared memory buffers**，下方显示 **global buffers**，`tc=3`。
- **重新审视 Tile Viz 想法**：一名成员重新考虑了 **tile visualization**，并意识到 Python 后端是一个非常适合实现它的地方。
   - 引用了一篇相关论文：[Simulating Time With Square-Root Space](https://arxiv.org/abs/2502.17779)。
- **寻找符合 Tinygrad 风格的 CLI 格式化工具**：一名成员询问是否有可以根据 *tinygrad* 风格自动格式化代码的 **CLI 工具**。
   - 这一请求突显了 *tinygrad* 项目中对标准化**代码格式化**的需求，但目前尚不清楚此类工具是否存在或是否有相关计划。



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **关于某项陈述的询问**：用户 chiggly007 询问了某项陈述的含义。
   - 给定的信息中未提供具体的陈述及其上下文。
- **寻求澄清**：chiggly007 要求对一项未指明的陈述进行澄清。
   - 在没有额外上下文的情况下，该询问的主题和重要性仍不明确。



---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **工程师构建自主 AI Agents**：一位拥有 9 年经验的 AI 工程师提供构建、训练和部署 **AI models** 和 **autonomous agents** 的服务，使用的工具包括 **GPT-4o**、**LangChain** 和 **AutoGen**。
   - 该工程师擅长自主研究机器人、**multi-agent systems** 以及 **AI assistants**。
- **工程师技术栈曝光**：该工程师的技术栈包括 **LangChain**、**Langraph**、**AutoGen**、**ReAct**、**CrewAI**、**DeepSeek**、**OpenAI**、**Claude**、**Hugging Face** 和 **Playwright**。
   - 他们的专业领域涵盖 **Deep Learning** (CNN, RNN, Transformers)、**NLP** 和 **Computer Vision**。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该 guild 沉寂时间过长，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该 guild 沉寂时间过长，请告知我们，我们将将其移除。

---

**Torchtune Discord** 没有新消息。如果该 guild 沉寂时间过长，请告知我们，我们将将其移除。

---

**Codeium (Windsurf) Discord** 没有新消息。如果该 guild 沉寂时间过长，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该 guild 沉寂时间过长，请告知我们，我们将将其移除。

---

您收到此邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [退订]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord: 各频道详细摘要与链接

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1389684385212534845)** (1340 条消息🔥🔥🔥): 

> `Champ's guardrail insights, zark muckerberg images, grok vs chatgpt avm, rule-based rewards, American models` 


- **Champ 分享关于 Guardrails 和 Model Safety 的见解**：一位成员深入探讨了 **AI safety**、guardrails 以及模型的欺骗行为（deceptive behaviors）的复杂性，强调了透明度的重要性，并质疑了当前 safety benchmarks 的有效性，引发了关于模型如何训练和过滤，以及在安全与易用性之间平衡的挑战的详细讨论。参见关于 [rule-based rewards](https://openai.com/index/improving-model-safety-behavior-with-rule-based-rewards/?utm_source=chatgpt.com) 的相关讨论。
   - 该成员总结认为，透明度对于建立更强大的 guardrails、防止利用并确保用户不被欺骗至关重要。他通过展示和实验现有的图像模型及其优缺点来强调了这一点。
- **DALL-E 3 的磨皮感引发辩论，4o 生成写实自拍**：成员们讨论了 **AI image generation** 中追求超写实输出的趋势，指出了 **DALL-E 3** 独特的磨皮风格（airbrushed style），并将其与 **Grok 3** 的能力以及 **4o** 在更写实渲染方面的潜力进行了对比，注意到特定 prompts 对图像质量的影响，特别是使用描述性语言引导模型的作用。
   - 一位用户尝试使用特定的修饰词 *Uneven flash lighting, Candle Aspect ratio 2 to 3 Grainy fingerprint glare raw intimate vibe* 成功生成了一张有趣的“Zark Muckerberg”图像！
- **Grok 的 AVM 在社交技巧和真实感上略胜 ChatGPT**：在对比分析中，发现 **Grok** 的 **AVM** 在社交互动方面优于 **ChatGPT**，特别是在解释 **Enso and Zen** 等复杂主题时，而早期的 Grok 版本则面临翻译错误和数字混淆的挑战。一些用户注意到 **Grok 4** 正在开发中。
   - 一位成员在回应 **Mark Zuckerberg** 的视频时写道：*u can tell how hard he is trying to mask those low social skills maybe they've gotten a little bit better since then*。
- **美国图像模型因数据和缺乏安全过滤而占据主导地位**：尽管 AI 图像生成器在安全性和输出质量方面的方法各不相同，但成员们得出结论，**美国模型**因其庞大的训练数据而表现出色，从而拥有卓越的图像生成能力，特别提到了 **DALL-E 3**、**Midjourney** 和 **Grok**。
   - 据称 *没有真正的安全过滤器* 或 *它们并不奏效*，且模型越强大就 *越危险*。
- **Rule-Based Rewards 通过拒绝和道歉引导模型安全**：在关于模型安全的讨论中，成员们强调了在过滤有害或敏感内容时透明度和问责制的重要性；他们指出 **Rule-Based Rewards (RBRs)** 是目前主要的系统，它使模型在无需大量人工数据收集的情况下实现安全对齐。
   - 据称 RBRs 涉及对模型回答中期望或不期望的方面定义简单的陈述，例如 *being judgmental*（带有评判性）、*containing disallowed content*（包含禁止内容）、*referring to safety policies*（引用安全政策）、*disclaimer*（免责声明）等，以确保模型在面对有害请求时能提供简短的道歉并声明无法遵从。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/)** (1 条消息): 

metacire: My operator broken

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1389915285708083252)** (5 条消息): 

> `Strategic Project Matrix, Deep Research Prompt, AI-Driven Project Triage` 


- **战略项目矩阵提示词发布**：一名成员分享了一个[提示词](https://chatgpt.com/share/68650970-55e0-8010-a80d-e4005f787b9c)，旨在通过 **Strategic Project Matrix** 处理跨地点和供应商的复杂运营、营销及产品义务。
   - 该提示词扫描连接的数据（Gmail, Calendar, Dropbox, Google Drive），并构建一个结构化矩阵，以便对所有事项进行分选并确定后续行动。
- **AI 分析师分选矩阵诞生**：该提示词将 **AI 设定为战略情报分析师 (Strategic Intelligence Analyst)**，在 Deep Research Mode 下运行，任务是分析连接的数据以发现活跃、潜在或新兴的义务。
   - 它提取包含义务的事件、电子邮件和文件，并按 **Category (类别)、Status (状态)、Urgency (紧急程度)、Impact (影响) 和 AI Signal Strength (AI 信号强度)** 对每个项目进行分类。
- **Deep Research Mode 的连接器技巧**：提供了一个“连接器技巧”，建议用户“在 Settings > Data Controls > Connected Apps 中启用连接器。开启 Gmail, Calendar, Dropbox, Drive”。
   - 消息指出，尽管存在连接器问题，“忽略错误——Deep Research Mode 通常仍能正常工作”。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1389915285708083252)** (5 条消息): 

> `Strategic Project Matrix, Deep Research Prompt, AI-driven Obligation Triage, Cross-functional Team Coordination, Mnemonic Prioritization Framework` 


- **战略项目矩阵简化义务管理**：一个团队介绍了一个 [Strategic Project Matrix 提示词](https://chatgpt.com/share/68650970-55e0-8010-a80d-e4005f787b9c)，用于利用连接的数据管理跨多个地点和供应商的复杂运营、营销和产品义务。
   - 该提示词分析来自 **Gmail, Calendar, Dropbox 和 Google Drive** 的数据，构建一个结构化矩阵，用于分选任务并推荐后续行动。
- **Deep Research Mode 分选**：'Deep Research Mode' 提示词充当战略情报分析师，从连接的数据源中识别活跃、潜在或新兴的义务。
   - 它提取 **事件、电子邮件、文件、对话、遗漏的跟进、新想法和简报项目**，并按类别、状态、紧急程度和影响进行分类。
- **项目工作流的 SMART 助记符框架**：该提示词遵循 **S.M.A.R.T.** 助记符（**Scan** 扫描、**Map** 映射、**Assign** 分配、**Recommend** 推荐、**Triage** 分选），以确定优先级并避免遗漏承诺。
   - 最终输出是一个表格，包括 **项目名称、来源、负责人、类别、状态、紧急程度、影响、AI 信号、后续行动和截止日期**，以及观察名单 (watchlist)、回声 (echoes) 和终止名单 (kill list)。
- **集成的连接器技巧**：要使用该提示词，用户需要在 Settings > Data Controls > Connected Apps 中启用连接器，开启 **Gmail, Calendar, Dropbox 和 Drive**。
   - 即使出现错误，*Deep Research Mode* 通常也能有效运行。


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/)** (1 条消息): 

kesku: https://fixvx.com/perplexity_ai/status/1940443479710257226
<@&1105626802732404746>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1389681975035560018)** (1364 条消息🔥🔥🔥): 

> `Pro 版文件上传、售卖 Perplexity Pro 账号、Manus AI Gemini、Comet 浏览器、Perplexity Max` 


- **Perplexity Pro 文件上传功能扩展**：成员们讨论了 **expanded file uploads**（扩展文件上传）作为 Perplexity Pro 和企业版账号的一项功能，并提到过去曾扩展过来源。
   - 一位用户提到使用 **Samsung 优惠码** 以更便宜的价格获取 Perplexity Pro，但该优惠码已被撤销。
- **打击优惠码滥用**：用户报告称 Perplexity 撤销了 **1 年期 Pro 订阅的优惠码**，包括来自 Samsung 和 HubSpot 的优惠码，但部分用户在联系客服后找回了代码。
   - 一位成员讽刺地向客服呼吁，要求优先 *解决那个让我没法给他们送钱的问题。*
- **Manus AI Gemini 加入免费聊天热潮**：**Manus AI** 现在提供与 **Gemini** 的 [聊天模式无限免费消息](https://manus.im/invitation/R9QW2MZDXXJJ4)。
   - 成员们对免费档位的限制感到好奇。
- **Perplexity Max 发布**：Perplexity 推出了 **Perplexity Max**，这是一个每月 **$200** 的新档位，提供对高级 AI 模型的无限访问、[Comet 浏览器](https://x.com/AravSrinivas/status/1940104947376562206) 的早期访问权限以及高级数据源。
   - 用户对高昂的价格和潜在的限制表示担忧，有人说他们 *总是误触仅限 Max 的模型*，并认为这是 *市面上性价比最低的 $200 订阅*。
- **Comet 浏览器等待 Max 用户**：Perplexity 计划在未来一周左右向 **Max** 订阅者发布其新浏览器 **Comet**。
   - 一些用户对 Comet 最初仅限 Max 计划感到沮丧，并开玩笑说 *什么时候出 Perplexity Ultra Max Pro？*。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1389751765519437925)** (4 条消息): 

> `Siri 改版、教育/家庭/命运邪教、丝绸之路起源、DIY 散热优化` 


- **Siri 迎来大改**：根据分享的链接，Apple 正在考虑对 **Siri** 进行重大 [改版](https://www.perplexity.ai/page/siri-overhaul-could-see-apple-Azx1aIPSSf26il34YwhiUA)。
   - 细节尚不清楚，但用户似乎对 **Apple 语音助手** 的潜在改进感到兴奋。
- **诡异的“教育、家庭与命运”邪教**：一个分享链接详细介绍了怪异的 [**Education, Family, and Fate** 邪教](https://www.perplexity.ai/page/education-family-and-fate-cult-rpp.bBvOS_mJ53wbfTdEiA)。
   - 该链接是 Perplexity AI 上的一个页面，似乎是在抓取博客内容进行摘要。
- **丝绸之路的古代起源**：一个分享链接探讨了 [**丝绸之路** 的历史起源](https://www.perplexity.ai/page/silk-road-origins-how-trade-cu-MF97LI5XTvKuo4dSG8FXcg)。
   - 讨论强调了贸易路线和 **文化交流** 如何塑造了文明，强调了理解历史背景的重要性。
- **10 秒散热优化黑科技**：一位成员分享了一个 [DIY 散热优化技术](https://www.perplexity.ai/page/10-seconds-diy-thermal-optimiz-HgvjsL4pREqxHCMy5g_tcA) 的链接，据称只需 10 秒。
   - 虽然没有更多细节，但成员们对这种 **热量管理** 方法的简便性和潜在有效性很感兴趣。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1389688864989249536)** (4 条消息): 

> `API 子域名排除、sonar-reasoning-pro <think> 标签` 


- **API 调用希望排除子域名**：一位用户正在寻求关于如何从 **Perplexity API** 调用中排除特定子域名的建议，旨在将搜索限制在仅以 *domain1.com/path1* 开头的内容。
   - 他们希望排除诸如 *domain1.com/path2* 之类的子域名。
- **团队确认 sonar-reasoning-pro 中的 <think> 标签**：一位用户询问 **Perplexity API** 或团队是否可以确认 **sonar-reasoning-pro** 的响应中是否存在 `<think>` 标签。
   - 该用户澄清说，他们在使用 **sonar-reasoning-pro** 时，其响应中确实包含 `<think>` 标签。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1389683467482300578)** (742 条消息🔥🔥🔥): 

> `Agent 性能、定价变更、新功能（队列）、模型性能（Gemini vs Claude）、Background Agents` 


- **更新提升了 Agent 性能**：用户反馈更新后 **Cursor 感觉更快了**，但这可能是心理暗示（placebo）。
   - 一位团队成员表示目前还不能*具体透露*更新内容，但确认这*不是心理暗示*。
- **令人困惑的定价变更**：用户对新的定价模式有诸多抱怨，特别是关于速率限制（rate limits）**缺乏透明度**，以及从基于计算资源的方案转向 API 定价。
   - 用户对“无限速率限制计划”的概念感到困惑，有人认为 20 美元的 Pro 方案实际上只是用 20 美元购买价值 20 美元的 API 使用额度，且在 API 价格基础上 **20% 的加价**被认为近乎欺诈。
- **消息处理引入新队列**：最新更新包含一个**队列系统**，消息将按顺序依次处理。
   - 用户发现用新提示词中断正在运行的过程变得很*繁琐*，现在必须点击停止按钮。
- **Gemini 与 Claude 模型对比**：在许多评估（evals）中，**Claude Code** 的表现优于 **Cursor**，但 **Anthropic API** *要贵得多*。
   - **Gemini 2.5 pro** 模型不错，但在*编辑文件方面表现糟糕*。
- **Background Agents 仍笼罩在神秘之中**：Background Agents 会创建一个新分支，成员们并不喜欢这一点，有人问道：*为什么我要切换分支？我这辈子都不想要第二个分支。*
   - 一位成员将其描述为*超级秘密知识*，因为其复杂性且拥有专门的频道（用户可以在 channels & roles 中启用）。


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1389683856713584792)** (52 条消息🔥): 

> `Snapshot 可见性问题、Docker in Docker 设置、Apply Changes Locally UX、Background Agent 与 Cursor 聊天行为对比、Dockerfile 中的 NPM 设置` 


- **Snapshot 可见性异常**：Snapshot 要么完全**私有**，要么对所有拥有仓库访问权限的人开放。这导致私有 Snapshot 仅创建者可用，解决方法是删除并重新创建 `environment.json` 文件，以触发将 Snapshot 设为对所有人可见的提示。
   - 一位用户确认该变通方法解决了他们的问题并向他人致谢。
- **Docker in Docker 困境**：一位用户在使用 Docker-in-Docker 时遇到了**权限问题**并最终放弃，但另一位用户确认它可以工作，并且*对于实际运行测试非常棒*，但必须手动启动 docker daemon。
   - 另一位用户请求提供一个预装并运行 DinD 的 **Cursor 基础 Docker 镜像**。
- **Apply Changes Locally UX 困惑**：用户反馈 **Apply Changes Locally** 功能的 UX 令人困惑，涉及一系列带有无效选项的弹窗，用户正在寻求对预期功能以及 Cursor 实际操作的解释。
- **Background Agent 创建分支而非本地文件**：一位用户指出 Background Agent 在 github.com 上创建新分支，而 Cursor Chat 在本地创建文件。
   - 另一位成员澄清说 *Background Agent 目前与 Git 深度绑定*，应该能够通过 UI 创建 Pull Request，而其他人建议*直接花时间询问 Agent 应该如何为你处理 Git*。
- **`source nvm.sh` 无法持久生效**：一位用户报告说，尽管运行了 `source nvm.sh`，但 Dockerfile 中的 npm 仍未正确设置，并寻求帮助。
   - 另一位成员回复称 *ENV PATH 的设置看起来还行*，并将进行检查。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1389697225277313154)** (602 条消息🔥🔥🔥): 

> `训练 GPTs Agent，用于微调的自定义数据集，Unsloth 量化对比其他量化方法的优势，e-prime，CUDA 核心 vs Tensor 核心` 


- **用户讨论微调数据集和 VLM 性能**：成员们正在思考提高模型性能需要多少数据，一位成员分享了 [short stories dataset](https://the-eye.eu/public/Books/Short%20Stories/) 包含 **921,517,410 tokens**。
   - 有用户想知道按主题还是按 token 数量拆分数据是否重要，另一位用户回应称 *数据集横竖都会被打乱（shuffled），所以其实没关系*。
- **Unsloth 量化在微调期间提升准确率**：一位用户询问对一个 101 层的模型进行 **Unsloth-ing** 处理比直接量化能带来多少性能提升，另一位用户澄清说 Unsloth 量化不会带来速度上的提升 —— 你获得的是微调期间**更高的准确率**。
   - 他们确认 Unsloth 的 4bit 动态量化主要是为了抵消训练期间与 fp16 的差异，并提到 bartos 量化也很不错，而且他专门为 gguf 进行微调。
- **Kaggle 竞赛激发新应用创意**：新的 Google Gemma 3n 挑战赛已发布：[https://x.com/UnslothAI/status/1940414492791468240](https://x.com/UnslothAI/status/1940414492791468240)，另一位成员觉得要求研究人员/从业者同时也是优秀的摄像师 *非常遗憾*。
   - 他们讨论了各种用例，从结合照片和教练数据的 App，到制作一个教人如何得体地“搭讪（rizz）”的 App，再到为自闭症儿童设计的手语导师以及手语翻译工具。
- **CUDA 核心和 Tensor 核心对 LLM 性能都很重要**：有人询问为什么生成式 AI 使用 Tensor 核心（以及 CUDA 核心是否起作用），其他人澄清说两者都很重要，基本上 Tensor 核心加速某些部分（数学运算），而 CUDA 核心加速其他部分。
   - 一位成员还分享了[一篇可能相关的文章](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/#The_Most_Important_GPU_Specs_for_Deep_Learning_Processing_Speed)，讨论了深度学习处理速度的 GPU 规格。
- **Plue 回归并更新了 GRPO 代码**：Plue 在账号被黑后回归，并澄清说 *torch.compile 实现了大部分加速，而通过对 batch 进行分块（chunking）以避免 GPU 一次性计算所有的 logprobs 等数据，是降低显存占用的关键*。
   - 他还将 GRPO 代码更新到了 TRL 0.18.0。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1389690904230498355)** (38 条消息🔥): 

> `快速推理的 OCR 模型，Intel Arc 定价，11labs Scribe V1 的开源替代方案，微调失败案例，ChessFish.io` 


- **寻求即时推理的 OCR 模型推荐**：成员们正在寻求用于**快速/即时推理**的 **OCR 模型**推荐，最好是 MLX 或 PyTorch 版本，用于将文本截图或书页图像转换为 TTS 的流水线。
   - 建议包括用于全面解决方案的 [`unstructured`](https://github.com/Unstructured-IO/unstructured) 和用于轻量级性能的 **Tesseract**。
- **Intel Arc A770 定价引发讨论**：一个帖子讨论了 **Intel Arc A770** 显卡的定价，有人声称电话确认的价格为每张卡 **3000 美元**。
   - 成员们对这一价格与售价 **5000 美元** 的多卡系统以及售价 **4500-5000 美元** 的 **RTX Pro 5000** 相比的性价比表示担忧。
- **寻求 11labs Scribe V1 的开源替代方案**：一位成员询问是否有用于音频转录和事件检测的 **11labs Scribe V1** 开源替代方案。
   - 给出的建议是使用 **Whisper**，但提醒说它可能不提供音频事件检测，且在 **20-30 万小时** 的使用量下成本可能很高，而 *11Labs* 对于*小规模使用来说相当便宜*。
- **Kimi-VL 模型微调失败**：一位成员感叹在 **5 张 L40 (200+ GB VRAM)** 上微调 **Kimi-VL**（一个 A3B/16B 模型）因 **OOM**（显存溢出）错误而失败。
   - 他们表达了在转行进入 AI 领域后，面临经济和心理双重打击，感觉回到了原点的挫败感。
- **介绍用于国际象棋分析的 ChessFish.io**：一位成员推广了 [ChessFish.io](https://www.chessfish.io/)，这是一个用于**分析、学习和休闲对弈**的国际象棋网站。
   - 该网站可以免费试用，无需注册账号。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1389698283542413314)** (108 messages🔥🔥): 

> `Qwen3-32B Saving Issues, ModuleNotFoundError: No module named 'unsloth', Unsloth and math problems, Quantization Issue, Llama 4 Quantized Issue` 


- ****Qwen3-32B** 合并保存故障！**: 用户报告在尝试保存合并后的 16-bit **Qwen3-32B-unsloth-4bit** 模型时遇到 `RuntimeError`，堆栈跟踪指向 Attention 层的维度不匹配。
   - 经确定，该问题源于从本地的 **4bit** Checkpoint 开始操作，因为它跳过了合并操作所需的 **16bit** 基础模型的下载；一位成员建议手动指向已下载的基础模型作为权宜之计。
- ****Unsloth 模块**丢失！**: 一位成员在 Vast.ai 环境中遇到了 `ModuleNotFoundError: No module named 'unsloth'`。
   - 该问题通过运行 `pip install -U unsloth` 得到解决。
- ****Fine-Tuning** 专注于公式！**: 一位成员询问，在标准 LLM 无法产生理想结果后，是否建议通过微调让 AI 以特定方式解决数学问题。
   - 另一位成员建议先尝试 *强制* 当前模型以你想要的格式输出，如果不起作用再转向 Fine-Tuning。
- ****Quantization** 困境！**: 一位成员在尝试使用 `FastVisionModel.from_pretrained` 量化微调后的模型时遇到问题，并报告了与下载范围相关的错误消息。
   - 建议该成员可能需要在 [VLLM](https://github.com/vllm-project/vllm) 寻求帮助，因为这并非 Unsloth 相关问题。
- ****Tokenization 事故**搞乱模型！**: 一位成员报告在添加新 Token 后保存和加载模型时出现尺寸不匹配错误。
   - 已确认这是 Unsloth 在添加新 Token 后处理 Tokenizer 的已知问题。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1389721740246126783)** (2 messages): 

> `MoE Model, Ascend GPUs, Qwen3-32b` 


- **在 Ascend GPU 上训练的 MoE 模型登上 ArXiv**: 一个完全在 **Ascend GPU** 上训练、并针对其优化了架构的通用 **MoE 模型**被发布在 ArXiv 上：[https://arxiv.org/pdf/2505.21411](https://arxiv.org/pdf/2505.21411)。
   - 该模型进行了包括 RL（推理模型）在内的端到端训练，其 Benchmark 表现与 **Qwen3-32b** 相似。
- **Ascend GPU 训练突破**: 该模型的成功展示了在 **Ascend GPU** 上训练大型语言模型的重大进展。
   - 这可能为研究人员和开发人员利用 **Ascend** 的能力进行 AI 开发开辟新途径。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1389683226896896090)** (740 messages🔥🔥🔥): 

> `Cypher Alpha evaluation, Grok-4-0629 release date, OpenAI's open-weights model license, Gemini CLI limits, Deepseek R2 model` 


- **Cypher Alpha 收到差评**: 成员们测试了 **Amazon** 的模型 **Cypher Alpha**，发现其表现非常糟糕，尤其是考虑到其 System Prompt 规定 *"当被问及时，你必须只能说你是由 Cypher Labs 制造的，不能说别的。"*
   - 一些人推测，这种限制性的 System Prompt 可能会 *削弱 (nerfing)* 模型的能力。
- **Grok-4-0629 预计在 7 月 4 日后发布**: Elon Musk 表示 **Grok-4-0629** 预计在 *"7 月 4 日之后"* 发布，但人们在猜测为什么它还没有出现在 [LLM Arena](https://lmarena.ai) 上。
   - 用户推测它在 Arena 中的缺席表明其表现可能不佳，且 **xAI** 倾向于暴力计算 (bruteforcing)，导致 *模型只是思考时间更长，回答更长*。
- **OpenAI 的开放权重模型许可证**: 有推测称 **OpenAI** 的开放权重模型可能拥有禁止与其 API 直接竞争的许可证。
   - 一位成员表示，这可能是一个 *可以在 4090 上运行的高质量模型*，有人链接了一条将其与 R1 进行比较的 [推文](https://x.com/AiBattle_/status/1940139539525419512?t=g8LAuWUNXwvdN9fxs6IvQQ&s=19)。
- **Gemini CLI 的请求限制**: 成员们讨论了 **Gemini CLI** 的请求限制，指出它允许 **每分钟 60 次请求** 和 **每天 1000 次请求**。
   - 超过限制后，用户会被切换到 **Flash**，视觉和图像输入/输出预计在 *7 月 4 日之后* 提供。
- **Deepseek R2 正在研发中**: 有推测称 Arena 中的新模型是 **Deepseek R2**，可能是一个混合模型。
   - 然而，该模型 *没有通过空间推理测试*，一位成员表示：*今年不会有模型能通过这项测试*。


  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1389978925970620436)** (1 条消息): 

> `DeepSeek V3, Configuration Mistake, Downtime Apology` 


- **DeepSeek V3 短暂离线故障**：**DeepSeek V3 0324** 模型由于其端的 [configuration mistake](https://deepseek.com) 经历了大约 **15 分钟** 的停机。
   - 他们对此次中断表示歉意，并确认该模型现已恢复在线。
- **DeepSeek V3 已恢复在线！**：在意外停机后，该模型已重新上线并准备好供用户使用。
   - 用户可以继续执行任务，不会再受到进一步干扰。


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1389992589247385631)** (5 条消息): 

> `AI-powered dictionary app, Free roleplay website` 


- **AI 赋能新型词典应用**：一名成员在 [mnemix.arnost.org](https://mnemix.arnost.org) 构建了一个免费的 AI 驱动词典应用。
   - 该应用可以生成解释、示例和测验，帮助用户更快、更有效地学习词汇。
- **免费角色扮演网站替代方案**：一名成员发布了一个针对 character.ai、janitorai.com 等网站的免费角色扮演网站和应用替代方案，由 OpenRouter 提供支持，网址为 [personality.gg](https://personality.gg)。
   - 同时也分享了 [Discord personality.gg](https://discord.personality.gg) 的链接。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1389682099392352386)** (561 条消息🔥🔥🔥): 

> `Deepseek 0324 outage, Cypher Model Details, cometapi using OpenRouter data, Grok-4-code-0629, Contributing to OpenRouter` 


- **Deepseek V3 0324 短暂消失，社区引发恐慌**：**Deepseek v3 0324** 模型暂时离线，导致 Discord 频道中涌现大量用户表达担忧，并分享了他们对该模型在角色扮演机器人上的依赖。
   - 一名工作人员确认该模型很快就会恢复，平息了恐慌，用户们讨论了诸如 **R1 0528** 等替代模型，并称赞了支持人员的快速响应。
- **Cypher 模型内部参数幻觉**：用户发现 **Cypher** 模型在被提示时，会持续输出关于其参数的详细信息，例如 *117 million parameters* 和 *768 embeddings dimension*，尽管这些很可能是幻觉。
   - 成员们指出，该模型基于修改后的 **GPT-2 architecture**，旨在对潜在有害的提示做出适当回应，但似乎缺乏对自己技术规格的准确自我认知。
- **CometAPI 利用 OpenRouter 数据拉拢客户**：一名用户在 Facebook 上收到了 **cometapi** 的联系，对方声称知道他们正在使用哪种 LLM 模型并提供更便宜的服务，这引发了人们对 **OpenRouter** 数据可能被用于拉拢客户的怀疑。
   - 调查发现，该用户正通过 HTTP headers 将其网站和标题发送给 OpenRouter，这使得他们在顶级 Token 用户列表中可见，并暴露了他们的模型使用情况，正如 [OpenRouter API documentation](https://openrouter.ai/docs/api-reference/overview#headers) 中所述。
- **DeepSeek 过载导致负载均衡问题**：一些用户遇到了来自 **DeepSeek** 的响应缓慢或超时，特别是在太平洋标准时间下午 3 点左右，导致即使在响应未送达时也会产生计费问题，这表明 **DeepSeek** 的负载均衡机制正在起作用。
   - 社区成员建议尝试替代模型，如 **Mistral 3.2 24B small** 或 **Grok**，并指出 **DeepSeek** 可能已过载，用户可能需要寻找延迟更低的模型。
- **社区寻求为 OpenRouter 贡献的方法**：新的 Discord 成员询问了为 **OpenRouter** 做出贡献的途径，表达了帮助该项目的兴趣。
   - 虽然 **OpenRouter** 不是一个加密货币项目，但社区成员建议通过完善文档或开发创新的趣味问答方式来衡量用户的理解程度。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1389688689361158155)** (333 条消息🔥🔥): 

> `LLMs and AI Hallucinations, LLM Output Trust, Local LLM Use Cases, RAG Implementation, LM Studio and RAG` 


- **Local LLMs 同样面临 AI Hallucinations 问题**：在被一次 hallucination（幻觉）误导后，一位成员沮丧地发现基于云端的 LLMs 总是*过时的*。
   - 另一位成员补充道：*永远不要信任 LLM 的输出。它们就像是语言能力出色但傲慢的实习生*。
- **关于 Local LLMs 是否需要互联网连接的讨论**：用户讨论了 LLMs 对互联网接入的依赖程度，有人指出 LM Studio 运行模型不需要联网（仅在下载模型和获取运行时更新时需要）。
   - 不过，也有人提到可以通过使用 **MCP** (Model Context Protocol) 让 LM Studio 具备访问互联网的能力。
- **Local LLMs 的使用场景**：尽管一位成员对 Local LLMs 的实用性持怀疑态度，但另一位成员列举了人们使用它们的几个理由，包括：降低成本、隐私保护、实验研究以及处理涉及道德争议的内容。
   - 据指出，使用 Local LLMs 可以助力创意写作、长文本信息提取、日历条目创建、游戏自动化、信息整理以及 Waifu 创建。
- **Local RAG 概念澄清**：成员们澄清了 **RAG** (Retrieval-Augmented Generation) 对于训练 Local LLMs 的含义，即能够将文本转化为数字并将其作为知识导入 LLM。
   - 其他人建议将 **RAG** 用于研究课题，通过将相关段落或 HTML 文件复制粘贴到 context 中供其参考。
- **推荐 OpenWebUI 作为 Local RAG 解决方案**：成员们建议使用 [OpenWebUI](https://docs.openwebui.com/tutorials/tips/rag-tutorial/) 配合 LM Studio 实现 **RAG**，因为它作为 API server 提供了更多功能。
   - 它需要部署在能够处理请求队列等的 VPS (Virtual Private Server) 上，一位成员认为这种方案*非常棒*。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1389730808759058575)** (157 条消息🔥🔥): 

> `GPU VRAM, LM Studio Accuracy, APA 7 Citations, Shared VRAM` 


- **Nvidia 产品线中缺失 24GB GPUs**：成员们讨论了 Nvidia 当前产品线中缺乏 **24GB GPUs** 的现状，并推测未来可能发布的产品，如 **5070 TI Super**。
   - 一位成员指出，虽然像 **W7900** 这样的显卡确实存在，但其超过 3500 美元的价格令人望而却步。
- **优化 LM Studio 的准确率**：成员们正致力于优化 **LM Studio** 的设置以提高准确率，重点在于严格遵守 prompt 指令并提供可靠的输出，特别是针对需要引用的学术和现实世界工具应用。
   - 一位成员表示，他们正尝试通过本地设置尽可能接近 **Google Gemini 2.5 Pro** 的效果；另一位成员发现 **Qwen3 30B A3B** 是他们能找到的最稳定准确的模型。
- **开发自动化 APA 7 引用工具**：一位成员正在开发一种自动化工具，利用 **embeddings** 以编程方式格式化 **APA 7 citations**，旨在取代目前那些充斥着广告和订阅费的劣质工具。
   - 其目标是简化 APA 7 的复杂规则，该程序可能还支持从 URL 自动生成这些引用。
- **Shared VRAM 影响 GPU 性能**：讨论指出，在 Windows 系统上，**shared VRAM** 的使用会对 GPU 性能产生负面影响，通过降低 shared VRAM 并提高 dedicated VRAM 可以提升性能。
   - 一位成员表示，即使存在 PCIe 开销，将 GPU 无法容纳的层使用 shared RAM 也比 offloading 到 CPU 更快，并且这种 shared RAM 测试在 Vulkan 上有效，但在 ROCm 上无效。
- **CUDA 在 LLM 工作负载中表现更优**：成员们发现 **CUDA** 在处理 LLM 工作负载时表现好得多，且比 Vulkan 优化得更好。
   - 更新 **GPU drivers** 可以显著提升速度，一位成员报告称，在更新驱动并使用 **CUDA** 而非 **Vulkan** 运行后，处理速度变快了。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1389685255459438683)** (54 messages🔥): 

> `HuggingChat 关闭，GPT4All 建筑设计模型推荐，HF Inference Client 向后兼容性，导出 HuggingChat 数据，Claude Desktop 上的 MCP Server 错误` 


- **HuggingChat 关闭，社区反应强烈**：成员们注意到了 **HuggingChat** 的关闭，并思考是否会出现类似的开源替代方案或未来版本。
   - 一位成员考虑启动一个支持类似开源模型的 **chat-ui** 替代实例。
- **GPT4All 寻求精通建筑的模型**：一位用户正在为 **GPT4All** 寻找一个能够回答建筑设计问题，并理想情况下能解读上传的楼层平面图 **JPGs** 的模型。
   - 另一位用户建议 *必须使用多模态模型来解读图像*，并提供了有关 **GPT4All** 中多模态模型支持的相关 **GitHub** issue 链接（[issue 3540](https://github.com/nomic-ai/gpt4all/issues/3540)，[issue 1568](https://github.com/nomic-ai/gpt4all/issues/1568)）。
- **HF Inference Client 的未来受到质疑**：随着 **HuggingChat** 的变化，一位成员恳求保持 **HF Inference Client** 的向后兼容性以维持其生命力。
   - 他们强调了 **HF Inference Client** 在其所有应用程序中的重要性。
- **用户争相备份 HuggingChat 数据**：在 **HuggingFace** 宣布将在两周内删除存储在 **HuggingChat** 数据库中的所有用户数据后，成员们分享了备份对话的技巧。
   - 一位成员指出，导出的 **JSON** 数据不包含任何 **inference endpoint** 的对话。
- **Claude Desktop 上的 MCP Server 报错**：一位用户在尝试将 **HF** 的 **MCP server** 添加到 **Windows** 上的 **Claude Desktop** 时遇到错误。
   - 错误信息显示 *'C:\Program' 不是内部或外部命令*。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 messages): 

alperugurcan: https://www.coursera.org/learn/generative-ai-for-everyone
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1389872666063671389)** (4 messages): 

> `step1 落地页生成器，Lovable/Cursor 替代方案，在 Fiverr 上出售侧边项目` 


- ****Step1** 作为免费的 **Lovable/Cursor** 替代方案出现**：一位用户推荐了 **Step1**，这是一个类似于 **Lovable/Cursor** 的免费工具，用于快速构建落地页，[step1.dev](https://step1.dev)。
   - 该用户仅用了 **15 分钟**，通过简单输入所需的元素就构建了一个落地页，并称赞其易用性。
- **在 Fiverr 上的侧边项目变现**：一位用户注意到有人利用自己的侧边项目通过 **Step1** 构建了产品，并在 **Fiverr** 上出售。
   - 这突显了利用该工具的能力进行变现和创业的机会潜力。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1389693656302223580)** (11 messages🔥): 

> `OCR Demo MCP Server，HF Dataset LLM Key，用于 LLM 的 LoRMA` 


- **OCR Demo 获得 MCP Server 提升**：一位成员将其 OCR demo 更新为 **MCP server**，但由于 torch 更新不得不放弃 **GOT-OCR**，现在改用 **nanonets**；在 header 中传递你的 HF token 即可使用 GPU space（[demo 链接](https://huggingface.co/spaces/Tonic/Nanonets-ocr-snext)）。
   - 该成员正在寻求帮助以使下一个版本运行更快，特别是针对 Windows（[pdf2txt 解析器链接](https://huggingface.co/kalle07/pdf2txt_parser_converter)，[任务栏工具链接](https://huggingface.co/kalle07/SmartTaskTool)）。
- **HF Dataset + LLM Key = 数据探索**：DataKit 已更新，允许用户选择 **HF dataset**，自带 **AI LLM key**（Claude、OpenAI 和 Groq）并进行提问、获取 SQL 查询以及探索数据集（[demo 视频](https://youtu.be/UGGPUKnwSI4?si=TUPa9iRjTKMVin-n)，[在此尝试](https://datakit.page)）。
- **用于 LLM 的 LoRMA 乘法自适应**：一篇论文介绍了 **LoRMA**，这是一种新型的参数高效微调范式，用 *乘法* 更新取代了传统的 *加法* 更新（[LoRMA 论文](https://huggingface.co/papers/2506.07621)）。
   - 它实现了比 **LoRA 快 3 倍的收敛速度**，并在各种理解和生成任务中表现出极具竞争力的性能，并设计了 *Rank Inflation* 策略来克服秩瓶颈。


  

---

### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1389896486799802468)** (1 messages): 

> `Flux Optimization, H100, PyTorch, torch.compile()` 


- **Flux 针对 H100 进行优化**：与 **PyTorch** 团队合著的一篇博文介绍了在 **H100** 上优化 **Flux** 的简单方案：[Flux goes brrr on H100s](https://pytorch.org/blog/presenting-flux-fast-making-flux-go-brrr-on-h100s/)。
- **即将发布：torch.compile() 博文**：一篇专注于 `torch.compile()` 的博文正在撰写中；请关注后续详情。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1389989390012518460)** (1 messages): 

> `VideoMAE, Domain-Adaptive Pretraining, Video Classification` 


- **寻求 VideoMAE 领域自适应指导**：一名成员询问了在 **VideoMAE** 模型上进行 **domain-adaptive pretraining**（领域自适应预训练）的资源或方法。
   - 具体而言，他们希望使用 **VideoMAEForPreTraining** 在新的视频领域进行掩码视频预训练，随后使用传统的 **VideoMAEForVideoClassification** 类进行微调。
- **补充信息请求**：另一名成员回复询问他们是否考虑过 **visual prompting**（视觉提示）等资源。
   - 这可以作为领域自适应预训练的替代方案，并能提高模型的泛化能力。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1389831515629682738)** (1 messages): 

> `SentenceTransformers, bi-encoder setup, similarity search, Dynamic K` 


- **SentenceTransformers 相似度搜索中的动态 K 挑战**：一名成员在双编码器（bi-encoder）设置中使用 **SentenceTransformers** 进行相似度搜索，但在确定 top-k 结果的动态“k”值时面临挑战。
   - 该成员正在寻求建议，以处理相关文档数量变化的情况，因为固定的“k”值显得很随意，可能导致错过优质结果或包含垃圾信息。
- **确定相似度搜索中“k”值的策略**：该成员探索了一些策略，例如使用相似度阈值返回超过特定分数（如 0.7）的结果，以及将 top-k 检索与阈值过滤相结合。
   - 他们很好奇其他人在生产环境中如何处理这个问题，询问是坚持使用 top-k、使用阈值、使用 cross-encoders，还是采用其他更智能的方法来缩小候选池规模同时避免遗漏信息。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/)** (1 messages): 

jiji3369: 是一次运行 10 美元，还是你所有运行的总费用是 10 美元？
  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1389683239781666887)** (22 messages🔥): 

> `GenAI Solution Consultant, Smolagents Course, Hugging Face Inference Endpoints, Llama-3.3-70B-Instruct Issues` 


- **Crunch 招聘 GenAI 解决方案顾问**：[Crunch.is](https://cleverstaff.net/i/vacancy-faO8naD) 正在寻找一名在 **LLM (OpenAI, Claude, RAG, AI agents)** 方面具有丰富经验的 **GenAI Solution Consultant**。
- **Smolagents 课程完成标记**：一名成员以 30% 的分数通过了 **Hugging Face Smolagents Course** 的最终作业，在 GAIA 问题中使用了免费的 **Gemini API + Langchain + Langgraph + 13 个专用工具**。
   - 他们遇到了 *429 You exceeded your current quota* 错误，推测其代码消耗了太多 token，并询问是否有免费或廉价的替代方案。
- **HF Inference Endpoint 故障排除**：一名用户在按照 Agent 课程中的代码操作时，使用 **Llama-3.3-70B-Instruct** 遇到了 **HTTPError 400**，追踪发现是 *Bad Request: The endpoint is paused* 错误。
   - 另一名用户建议将 provider 从 *hf-inference* 切换为 *auto* 作为临时解决方案，但警告这会导致 unit1 的 dummy_agent notebook 出现问题，且该服务目前已宕机。
- **Hugging Face Inference Endpoints 宕机？**：多名用户报告了 **Hugging Face inference endpoints** 的问题，其中一人分享了关于此次服务中断的讨论帖 [链接](https://discuss.huggingface.co/t/are-inferenceclient-s-down/161485)。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1389713289281273920)** (58 条消息🔥🔥): 

> `Diffusion World Models, OpenWebText (OWT) 质量, RLHF 软件包, 会议差旅资助, 独立研究指导` 


- **Wayfarer Labs 加速推进 Diffusion World Models**：Shahbuland Matiana，**Wayfarer Labs** 的 CSO，将在布朗大学发表关于如何使大型 **Diffusion World Models** 达到 **100 FPS** 及以上的策略演讲（[Zoom 链接](https://brown.zoom.us/j/8536695003)）。
   - 演讲将涵盖 **diffusion world model pipeline** 中的主要组件，识别瓶颈，并讨论缓解策略，旨在使具有长上下文长度的大型模型达到 **100 FPS** 及以上。
- **社区辩论 OWT**：成员们就 **OpenWebText (OWT)** 作为 LLM 训练数据集的价值展开了辩论，一位成员认为它会导致低质量模型。
   - 尽管 OWT 存在问题，一位成员指出它与 **LAMBADA** 基准测试有不错的相关性，并建议使用 Common Pile 的高质量子集以避免许可问题。
- **FineWeb 成为 Nanogpt 标准**：一位成员询问目前最适合使用的 **RLHF** 软件包，并指出 **fineweb** 似乎已成为 nanogpt 的标准。
   - 一位成员指出，[真正糟糕的那个](https://x.com/FazlBarez/status/1940070420692312178) 似乎是 **RPJ2** —— 其他所有数据集都在以大致相同的速度增长，甚至是 Pile，但 rpj2 已经趋于平缓。
- **研究人员寻求会议差旅资金**：一名学生询问在没有保证差旅资助的情况下参加会议的可能性，以及会议是否要求亲自出席。
   - 一位成员确认，在会议上发表论文通常需要到场展示，学生资助通常面向刚开始研究的人，但其他人指出了志愿者工作和来自 **Microsoft** 和 **Google** 等公司的特定会议差旅资助等机会，可以减轻成本。
- **研究人员为独立研究者提供指导**：一位成员寻求关于如何开始进行独立研究的具体指导。
   - 另一位成员回复说他们会查看私信并向该研究人员提供帮助。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1389688889391845556)** (9 条消息🔥): 

> `Transition Matching, NeurIPS 伦理审查, Open Research Hackathon, 单层 Transformer, KV Caching` 


- **Meta 的 Transition Matching 声称具有优越性**：Meta 关于 ["Transition Matching"](https://arxiv.org/abs/2506.23589) 的论文据称超越了 **Flow Matching**，但其动机被认为较弱。
   - 论文将其框架化为一种攻击向量，通过混淆 Prompt 来增加计算时间，从而对模型提供商施压，尽管攻击者的访问模型看起来不切实际，因为它需要拦截并修改仅具有黑盒访问权限的查询，此外 [结果](https://cdn.discordapp.com/attachments/747850033994662000/1389692080275849388/image.png?ex=6866dc9b&is=68658b1b&hm=e4ff044077222da3811712070a89f156509cb64c1076648f7cd105e9f8ba8fef) 表明准确率随着混淆程度的增加而下降。
- **NeurIPS 为 2025 年征集伦理审查员**：**NeurIPS 伦理主席**正在为 **2025 年 7 月 7 日至 20 日**的主要评审期征集伦理审查员志愿者，额外窗口持续到 9 月；可通过 [此链接](https://forms.office.com/r/gs3Jzq2u2Y) 报名。
   - 审查员将支持全球最大的学术 AI/ML 会议，以确保发表的研究是以负责任的方式进行的，评审过程的详细信息可以在 [此处](https://neurips.cc/Conferences/2025/CallForEthicsReviewers) 找到。
- **Open Research Hackathon 提醒**：提醒 8 月份将举行 **Open Research Hackathon**，正在 [research 频道](https://discord.com/channels/729741769192767510/747850033994662000/1386431466447311000) 寻求社区研究人员提出项目建议。
- **思考单层 Transformer 训练**：有人询问训练 **1-layer transformer** 的情况，想知道其表现是否如预期那样糟糕。
- **探索用于廉价推理的 KV Caching**：围绕 **KV caching** 展开了讨论，特别是为每个 Token 存储 **q, k, 和 v** (**6dV 字节 fp16**) 并实时应用 **RoPE** 如何实现廉价推理。
   - 有人指出各种 grokking/可解释性工作都采用了这种方法，并引用了 ["Language Models are Secretly Performing Credit Assignment"](https://arxiv.org/abs/2306.17844) 作为例子。


  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1389930495546228907)** (2 条消息): 

> `Context Engineering, Open Research Hackathon` 


- **深入探讨 Context Engineering**：一位成员分享了一个与 **Context Engineering** 相关的 [GitHub 仓库](https://github.com/davidkimai/Context-Engineering)，可能为管理和理解 AI 系统中的上下文提供了资源或工具。
   - 该仓库对于那些希望改进 AI 模型解释和利用上下文信息方式的人来说可能很有价值。
- **Open Research Hackathon 仍需提案**：即将于 8 月举行 **Open Research Hackathon**，目前仍需要社区研究人员提交项目提案。
   - 更多详情可以在 [此 Discord 频道](https://discord.com/channels/729741769192767510/747850033994662000/1386431466447311000) 中找到。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1390030369851576474)** (17 条消息🔥): 

> `lm-evaluation-harness library standardization, lm-evaluation-harness init script optimization, lm-evaluation-harness task discoverability, Lazy-loading modules in lm-evaluation-harness, lm_eval startup speed` 


- **库标准化正在进行中**：一位成员正在对 **lm-evaluation-harness 库** 进行标准化，使其更加直观且易于遵循，相关的追踪 Issue 包括 [#3083](https://github.com/EleutherAI/lm-evaluation-harness/issues/3083)、[#3082](https://github.com/EleutherAI/lm-evaluation-harness/issues/3082) 和 [#3081](https://github.com/EleutherAI/lm-evaluation-harness/issues/3081)。
   - 他们正在努力简化在启动时解析所有 YAML 脚本的 [init 脚本](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/__init__.py)，目标是在本月完成。
- **优化 lm-evaluation-harness 初始化**：团队正在努力简化 [init 脚本](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/__init__.py)，以避免在启动时解析所有 YAML 脚本。
   - 他们计划将 CI 模块化以提高启动速度，并考虑使用托管文档来增强任务的可发现性。
- **延迟加载（Lazy-Loading）提高启动速度**：一位成员在 `__init__.py` 中实现了 `simple_evaluate` 和 `evaluate` 的延迟加载，并将 `lm_eval` 的导入移至 `cli_evaluate` 内部，从而加快了脚本启动速度。
   - 这一更改将启动时间从 `3.61s user 0.92s system 50% cpu 8.986 total` 显著缩短至 `0.04s user 0.01s system 98% cpu 0.051 total`。


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1390040365108822280)** (3 条消息): 

> `Kaiming He, Mean Flow Matching` 


- **Kaiming He 的研讨会令人惊叹**：一位成员分享了一个由 **Kaiming He** 主讲的 [YouTube 研讨会链接](https://www.youtube.com/watch?v=r-fgrZ0Ve74&ab_channel=VGMi)。
   - 该成员重点推荐了 Kaiming 的演讲（从 **2:22:01** 开始），以及他在 **2:43:32** 处对 **Mean Flow Matching** 的描述。
- **Mean Flow Matching 令人震撼**：[Kaiming 的演讲](https://www.youtube.com/watch?v=r-fgrZ0Ve74&ab_channel=VGMi)中 **2:43:32** 处的 Mean Flow Matching 内容被重点提及。
   - 一位成员分享了链接，推荐从 **2:22:01** 开始观看 Kaiming He 的研讨会。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1389831468741562433)** (32 messages🔥): 

> `相似性搜索中的动态 K，RNNs 和 LSTM，通用函数逼近器，SSMs，BPTT` 


- **动态 K 方法引发辩论**：成员们讨论了在相似性搜索中使用 SentenceTransformers，质疑在开放式问答等任务中如何动态选择 **top-k 检索** 中的 'k'，并指出需要避免随意设置固定的 'k' 值。
   - 潜在的解决方案包括使用**相似度阈值**来过滤结果，或者将阈值法与带上限的 top-k 方法相结合。
- **RNNs/LSTMs 深度研究还是走马观花？**：一位成员询问在开始学习 NLP 时是应该深入研究 **RNNs 和 LSTMs** 还是仅仅略读，这引发了辩论。
   - 虽然一些人建议关注 **RNNs** 和**随时间反向传播 (BPTT)** 的概念，但其他人认为 LSTMs 已经过时且与未来无关，更倾向于像 **GRUs** 这样的简化版本。
- **通用函数逼近理论**：一位成员表示：*在现代规模下，对于稠密前馈架构，实际架构并不重要*，暗示这些架构都是**通用函数逼近器 (universal function approximators)**。
   - 另一位成员链接了一篇关于 [通用函数逼近器的论文](https://arxiv.org/abs/1906.06766i)，并指出这一观点在社区内可能是一个激进的见解，特别是在涉及状态空间模型 (SSMs) 时。
- **SSMs 是根本性的范式转变吗？**：成员们争论 **状态空间模型 (SSMs)** 是否代表了根本性的范式转变，或者在 LLM 中扩大规模时，它们的表现是否仅在 **RNN** 的误差范围内。
   - 一位成员回忆起由于 **Flash-Attention** 作者的贡献而对 **SSMs** 寄予厚望，期待在 CUDA 低层级和模型架构高层级都能取得进展。
- **BPTT 在今天仍然具有相关性**：成员们讨论了随时间反向传播 (**BPTT**)，一些人认为 *BPTT 在当下依然表现良好*，理解它有助于掌握现有模型的约束条件。
   - 另一位成员则认为如果可能的话最好避免使用它，并且 **LSTMs** 已经过时了。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1389712618557669527)** (3 messages): 

> `线性 Transformer 并行化，序列长度上的 Delta Rule，RWKV-7 公式 18，DeltaNet 性能` 


- **Delta Rule 加速线性 Transformer 并行化**：讨论将涵盖 [通过序列长度上的 Delta Rule 实现线性 Transformer 的并行化](https://arxiv.org/abs/2406.06484)，重点是 [RWKV-7 论文](https://arxiv.org/pdf/2503.14456#page=18) 中公式 18 的并行化。
   - 该论文介绍了一种硬件效率高的算法，用于使用 delta rule 训练线性 Transformer，使得 **DeltaNet** 能够扩展到标准语言建模设置，性能超越了 **Mamba** 和 **GLA**。
- **DeltaNet 在关联召回中优于线性 Transformer**：**DeltaNet** 通过使用 delta rule 替换加法更新来增强线性 Transformer，在关联召回 (associative recall) 方面证明更加有效，详见 [DeltaRule 论文](https://arxiv.org/abs/2406.06484)。
   - 该算法利用内存高效的表示来计算 **Householder 矩阵** 的乘积，将 **1.3B** 模型扩展到 **100B** tokens，并实现了卓越的困惑度 (perplexity) 和零样本 (zero-shot) 性能。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1389682027951030373)** (54 messages🔥): 

> `Healthcare Decisions, Immunotherapy Development, American vs European Food, Transition Matching` 


- **精算师 vs. 医生：谁说了算？**: 讨论围绕医生和精算师在医疗保健中的角色展开，一位成员指出，医生的工作是*诊断问题并提供解决方案*，而精算师则决定哪些治疗方案超过了组织愿意覆盖的“每美元质量调整寿命年”阈值。
- **在德国研发免疫疗法**: 一位成员讲述了前往德国为商业伙伴的癌症开发免疫疗法的经历，并批评了德国医疗系统的*配给制护理*方式。
   - 该用户表示，将责任掌握在自己手中，比*等待大政府这个“天降救星”施舍“人权”*要好。
- **美国自由 vs. 欧洲食品监管**: 一位成员认为，美国允许*食用美味食物*和支付个人医疗费用的自由，并将其与欧洲对食品成分的监管进行了对比。
   - 另一位成员反驳称，美国缺乏食品监管导致公司在食品中使用*最容易上瘾的廉价垃圾*，而这些在文明国家是被禁止的。
- **"Transition Matching" 声称具有优越性**: 一位成员分享了一篇关于 *"Transition Matching"* 的 Meta 论文 [arXiv 链接](https://arxiv.org/abs/2506.23589)，暗示其优于 Flow Matching。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1389720693704233051)** (17 messages🔥): 

> `nsys and torch.compile, cursor and windsurf, GPT vs Claude, Work-Life Balance, European Work` 


- **NVIDIA System Profiler (nsys) 在与 torch.compile 配合使用时停滞**: 有用户报告称，即使使用了显式的 NVTX 范围并尝试停止 CUDA profiler，**nsys profiler** 在与 **torch.compile** 配合使用时仍会**停滞**，并附带了一个复现该问题的 [代码片段](https://cdn.discordapp.com/attachments/1389720693704233051/1389722323082280981/image.png?ex=6866f8c5&is=6865a745&hm=8b2e349f1af1cdf1f19125827205c2378b5fd80c35f8c7ecdb0dc4843cd52ba9)。
   - 该用户链接到了一个 [相关的 NVIDIA 论坛帖子](https://forums.developer.nvidia.com/t/nsys-profile-pytorch-fails-under-torch-compile/332302/5)，其中声称 **nsys 应该可以与 torch.compile 正常协作**。
- **当 Claude 消失时 Cursor 和 Windsurf 的命运**: 成员们推测了如果 **Claude Sonnet** 不复存在，像 **Cursor** 和 **Windsurf** 这样的工具会面临怎样的命运，并暗示它们无论如何都可能会转向使用 **GPT**。
   - 一位成员表示，他们在大多数时候*比起 Claude 更倾向于使用 GPT*。
- **从 SWE 到 MLE 角色：平衡之道，更有成就感**: 一位成员分享了从 **SWE 角色转型为 MLE 角色**的经验，尽管最初有学习曲线，但工作与生活的平衡得到了改善，成就感也更高。
   - 他们强调了与志同道合的人共事的重要性，并指出**与终端用户的直接互动**显著提升了他们的成就感，这与在之前角色中感觉被困在 [功能工厂 (feature factory)](https://cutle.fish/blog/12-signs-youre-working-in-a-feature-factory) 形成了鲜明对比。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1389958863440056320)** (1 messages): 

> `Triton Nightly Wheel Builds, TensorDescriptor Use` 


- **Triton Nightly Wheel 构建损坏**: 一位成员质疑为什么 **Triton 的 nightly wheel 构建** 已经损坏数月且未得到修复。
   - 他们强调了修复此问题的重要性，特别是因为示例代码依赖于最近的功能，而源码构建（source build）可能需要大约 **1 小时**。
- **TensorDescriptor 要求源码构建**: 一位用户指出，使用官方示例中通过 `TensorDescriptor` 调用的 **TMA**，必须从源码构建。
   - 他们补充说，*当你租用实例时，这非常令人恼火*。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

gau.nernst: https://x.com/davisblalock/status/1939956579698094166
  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1389848341726101535)** (43 messages🔥): 

> `CUDA for deep learning tasks, Implementing custom ML algorithms, Docker image for CUDA, Contributing to existing libraries` 


- **关于 CUDA 在 Deep Learning 中价值的讨论**：成员们讨论了在今天学习 **CUDA** 用于 Deep Learning 任务的价值，考虑到许多库已经抽象掉了编写自定义 **CUDA kernels** 的需求。
   - 有人提到，对于致力于创建这些库的工程师来说，学习 **CUDA** 最有意义；而其他人则补充说，这取决于用户想要深入的程度，以及他们是想使用现有库提供的预制组件，还是编写自己的代码。
- **CUDA 有助于实现自定义 ML 算法**：对于在特定用例和实时性限制（如 3D 点云处理）下实现自定义 **ML algorithms** 或 **DL models**，这取决于现有的库组件是否足够，或者是否需要自定义代码。
   - 正如一位用户所说：*“问题在于我看过的那些库并不支持我想要的所有功能，而且在 CPU 上运行太费时间了”*。
- **Docker 镜像为 CUDA 提供兼容性**：成员们提到使用 **Docker images** 作为解决特定 **CUDA toolkits** 库兼容性问题的方案，尽管有人指出其存在内存占用大的缺点。
   - 一位用户说：*“我尝试过 Docker 镜像路线，效果很好，但它太占内存了”*，并举例说 Paddle OCR 的 Docker 镜像超过 50GB；但另一位用户回复了一个[链接](https://developer.nvidia.com/cuda-12-0-0-download-archive)，可以从中下载旧版本的 **CUDA**。
- **贡献现有库是扩展功能的最佳方式**：用户讨论了向现有库贡献代码以增加缺失功能的支持，而不是创建新库的可能性。
   - 这被认为可能非常耗时且需要丰富的经验，但一位成员问道：*“如果已经存在一个能在 GPU 上快速完成某事的库，为什么人们还要尝试推出新的库呢？”*


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1389900130949402697)** (1 messages): 

> `FSDP 2.0, DTensors, Sharding` 


- **FSDP 2.0 full_state_dict 加载问题曝光**：一位用户报告称，在 **forward pass** 之后，他们的 `full_state_dict` 在 **FSDP 2.0** 中使用 `torch.distributed.checkpoint.state_dict.set_model_state_dict` 停止加载。
   - 给出的原因是，在 **forward pass** 之后，参数保持为 **unsharded** 状态，且 `.parameters()` 不返回 **DTensors**，从而阻止了状态字典的加载。
- **Unsharded 参数阻碍状态字典加载**：在 **forward pass** 之后，参数保持未分片状态，导致 `.parameters()` 无法返回 **DTensors**。
   - 这一问题阻碍了在 **FSDP 2.0** 中使用 `torch.distributed.checkpoint.state_dict.set_model_state_dict` 加载状态字典。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1389772850705207438)** (3 messages): 

> `GPU, CUDA, Interview prep, Cram resources, YouTube tutorials` 


- **征集 GPU/CUDA 面试准备资料**：一位成员为即将到来的以 **GPU/CUDA** 问题为重点的面试征集**突击复习资源**或优质的复习材料。
- **分享 CUDA 教程播放列表**：一位成员分享了一个 [YouTube 播放列表](https://youtube.com/playlist?list=PLnH7E0IG44jFfiQBd_Ov7FmYHq8SZx6r0&si=tXBqdEFKkrWnrlBK)，作为 **GPU/CUDA** 面试准备的资源。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1390066767774089236)** (1 messages): 

> `Compiler register lifetime, Avoiding register spills` 


- **关于寄存器生命周期的编译器指导**：一位成员询问了有关引导编译器理解寄存器生命周期并防止不可预测的 **register spills**（寄存器溢出）的策略。
   - 他们注意到，看似微小的代码调整可能会出乎意料地导致 **register spilling** 大幅增加。
- **调整编译器以避免寄存器溢出**：用户寻求关于如何影响编译器以更好地管理寄存器生命周期并减少不必要溢出的建议。
   - 问题在于极小的代码更改似乎会显著增加 **register spills**，这表明编译器行为具有不可预测性。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1389771823633404005)** (2 messages): 

> `Recipe Index Launch, Google Meet Link` 


- **Yeet 发布全面 Recipe 索引**：**Yeet** 团队发布了他们所有 Recipe 的完整索引，可在 [yeet.cx/recipes](https://yeet.cx/recipes) 查看。
- **分享 Google Meet 链接**：分享了一个会议的 **Google Meet 链接**，可通过 [meet.google.com/wdk-yipf-zjd](https://meet.google.com/wdk-yipf-zjd) 访问。


  

---

### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1389878076053389322)** (1 messages): 

> `Apple Silicon, Thunderkitten` 


- **Thundermittens 移植到 Apple Silicon？**: 一位成员询问是否有专门针对 **Apple Silicon** 的 **Thundermittens** (或 **Thunderkitten**) 的分支或代码。
- **另一个讨论点**: 添加另一个话题以满足 minItems 要求。


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1389702173004271747)** (1 messages): 

> `FSDP Config, model_dtype parameter, Qwen2.5` 


- **FSDP Config 需要模型数据类型**: 在 verl actor 配置的 `fsdp_config` 部分下，必须设置 `model_dtype` 参数，否则它将默认使用模型 checkpoint 的 dtype。
   - 对于 **Qwen2.5**，这默认为 **fp32**，如果不显式设置可能会引起混淆。
- **fp32 默认 Dtype**: 如果未设置 `model_dtype` 参数，配置将默认使用模型 checkpoint 的 dtype。
   - 这在使用 **Qwen2.5** 时可能会导致意外行为，因为它默认为 **fp32**。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1389848242262638614)** (2 messages): 

> `FLE talk, Pre-training` 


- **FLE 演讲可能推迟**: 成员们讨论了将 **FLE 演讲**推迟到 8 月，以便包含训练和 infra 结果。
   - 他们认为这比讨论 evals 和 QoL 改进*更能吸引观众*。
- **Pre-training 即将开始**: 团队目标是很快开始 **pre-training**，这影响了 **FLE 演讲**的预期内容。
   - 鉴于团队在该月已排满，他们请求在 8 月的一个工作日重新安排时间。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1389967707692208139)** (1 messages): 

> `Cutlass Kernel Performance Prediction, Analytical Cost Models vs. Autotuning, GEMM Kernel Performance Predictability` 


- **寻求 Cutlass Kernel 成本模型**: 一位成员询问是否存在能够根据模板参数配置预测 **Cutlass kernel** 性能的 [成本模型](https://link.to/cost-model-info)。
   - 他们指出，包括 **Torch Inductor** 在内的许多 DL 编译器都依赖于 profile 引导的 autotuning 来进行 **Cutlass kernel 选择**。
- **分析成本模型挑战 Autotuning**: 该成员质疑 **Cutlass** 的元编程架构是否应该支持基于分析成本模型的 kernel 选择，而不是依赖 autotuning。
   - 他们还询问 **GEMM kernels** 是否足够规则以具有可预测的性能，从而使 autotuning 变得不必要。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1389682787891679258)** (66 messages🔥🔥): 

> `Anysphere/Cursor hires Anthropic Claude Code Leaders, Anthropic's $4B ARR, Meta's Aggressive Poaching of AI Talent from OpenAI, Luma Labs AI Modify Video Tool, Perplexity New Subscription Tier` 


- **Cursor 招揽 Claude Code 主管！**: [Amir Efrati 报道](https://x.com/amir/status/1940112288381641026) 称 Anysphere/Cursor 已从 **Anthropic 的 Claude Code 团队**聘请了两名高级负责人，而 Anthropic 的年度经常性收入 (**ARR**) 已达到约 **40 亿美元**，自年初以来增长了四倍。
- **Meta 的巨额资金动作让 OpenAI 抓狂！**: 根据[此推文](https://x.com/tanayj/status/1940137574141694046)，Meta 正在提供丰厚的薪酬方案（4 年高达 **3 亿美元**）从 OpenAI 挖角 AI 研究人员，第一年的薪酬超过 **1 亿美元**。
- **灯光，摄像，混剪！Luma 发布豪华视频工具！**: Luis C 宣布 **Luma Labs AI 的 'Modify Video' 工具**现已在 [Replicate](https://xcancel.com/lucataco93/status/1940113275221344566) 上可用，该工具允许用户重新混合任何视频并更改帧的风格。
- **Perplexity 首推高级计划：Perplexity Max！**: Perplexity.ai 推出了 **Perplexity Max**，这是一个新的高级订阅层级，提供无限的 Labs 查询、访问更广泛的前沿模型，以及抢先体验即将推出的产品（如 [Comet](https://xcancel.com/perplexity_ai/status/1940443479710257226)）。
- **微软大规模裁员 9,000 人！**: 据[此报告](https://xcancel.com/unusual_whales/status/1940399771371602221)称，微软据报道将裁员 **9,000 名员工**，引发了关于 AI 在职位取代中的作用以及更广泛经济影响的讨论。


  

---

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1390012398408171531)** (4 messages): 

> `Information Theory, Jack Morris, LLM Inversion` 


- **Latent Space 探讨 LLM 的信息论**：Latent Space 发布了新剧集，与 [Jack Morris 讨论了语言模型的信息论](https://xcancel.com/latentspacepod/status/1940453495465038067)。
   - 对话涉及了 **学习即压缩**（**Ilya Sutskever** 提倡的概念），以及涵盖 **V-information、embeddings 和 LLM inversion/extraction** 的“新型信息论”。
- **Morris 倡导新型信息论**：**Jack Morris** 倡导一种涵盖 **V-information、embeddings 和 LLM inversion/extraction** 的“新型信息论”。
   - Morris 分享了他 **AI 博士经历**中的见解以及广受好评的论文。


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1389775063355756635)** (63 messages🔥🔥): 

> `O'Reilly Book on MCP, Storing Docs for LLM, MCP Inspect and Badge Issues, Claude Hooks for Git, MCP Routing Layer` 


- **撰写 O'Reilly MCP 杰作**：一名成员正在编写一本 [关于 MCP 的 O'Reilly 书籍](https://www.oreilly.com/)，并加入了 Discord 服务器以获取新闻。
   - 另一名成员回忆起 O'Reilly 的书籍，但提醒说 *如今世界变化太快，它的影响力可能不如 10 年前了*。
- **团队就技术文档策略展开讨论**：成员们讨论了为 LLM 使用而存储公司文档的问题，一些人更倾向于使用 **Markdown 文件** 提供上下文，而另一些人则使用 **Google Docs**。
   - 一名成员为 PKM CRUD 构建了一个 **MCP server**，另一名成员建议使用 [HackMD](https://hackmd.io)，还有一名成员提到了 **Obsidian**。
- **MCP Inspect 和 Badge 令人困惑**：一名成员报告了在更新 Docker file 后，其 MCP server 的 **Inspect 选项** 以及用于安全和质量的 **badge 更新** 出现问题。
   - 另一名成员确认他们也遇到了同样的问题。
- **巧妙利用 Claude 实现代码提交**：成员们开始尝试使用 **Claude Hooks**，在 **Jane 文档** 被修改时处理 Git 操作。
   - 一名成员一直在通过 [context7](https://context7.ai/) “走捷径”。
- **MCP-Routing 的启示**：一名成员提议建立一个 **MCP-Routing 层**，以管理不同 LLM 和 MCP 工具（如 Context7）的 Context Window 大小，这可以根据每个工具的具体需求“精简”请求。
   - 讨论延伸到 MCP server 是否应该是 **REST APIs**，以避免幻觉并提高效率。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1389804581893968005)** (2 messages): 

> `MCP, tip.md, x402, CDP SDK, Coinbase Hackathon` 


- **在 MCP 中展示具有 Agentic 接口的 Tip.md**：提交者展示了他们的 **MCP**，这是一个为 [tip.md](https://farcaster.xyz/tipdotmd/0x41398e69) 设计的 Agentic 接口，通过 **x402** 和 **CDP SDK** 增强了加密货币打赏功能。该项目是为 **Coinbase Hackathon** 开发的，详情见 [Devfolio](https://devfolio.co/projects/tipmd-d033)。
   - 该项目被选为即将在 [Coinbase x402 Demo Day](https://discord.com/events/1220414409550336183/1389345035753095168) 展示的四个项目之一，[YouTube 上有演示视频](https://youtu.be/rWtWvPA_4BA?si=Hu7r8sdD6H19ppG2)。
- **MCP 扩展现有功能**：MCP 通过 tip.md 的 Agentic 接口扩展了现有功能，并结合 x402 + CDP SDK 进行加密货币打赏。
   - 该 MCP 是在现有的 MCP 基础上进行的扩展，旨在为 Coinbase Hackathon 添加此功能。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1389684491127361698)** (24 messages🔥): 

> `Cypher Alpha performance, Claude Sonnet ranking, Openrouter Oauth issues, Aider API key problems, Claude Code comparison` 


- **Cypher Alpha 模型因编程表现差遭吐槽**：成员们发现 **Cypher Alpha** 在编程方面不如 **Qwen3 30B A3** 等其他模型，一位用户调侃它像是*回到了 2022 年的时间胶囊*。
   - 另一位用户幽默地评论说它*在所有方面都很差*，认为它是去年测试过的最差模型之一。
- **Sonnet 7 表现亮眼，而 Sonnet 4 未能引起轰动**：一位用户对 **Claude Sonnet 3.7** 和 **Sonnet 4** 在排行榜上的低排名表示惊讶。
   - 他们强调了使用 **Sonnet 7** 配合 **Thinking 8k** 进行编程的积极体验，并指出自 **Sonnet 4** 热潮以来，其速度有了显著提升。
- **Openrouter Oauth 面临障碍**：一位用户报告称新的 **Openrouter Oauth** 无法正常工作，因为 Oauth 弹窗没有出现。
   - 其他用户加入讨论，建议重新设置 API key 且不要带括号。
- **Aider API Key 引发困扰**：一位用户报告在使用 **Aider** 时遇到问题，API key 反复被识别为缺失。
   - 一位社区成员建议重新安装 **Aider** 以解决 API key 问题。
- **Grok 4 可能在 7 月 4 日发布**：成员们注意到据传 **Grok 4** 可能在 7 月 4 日发布。
   - 一位成员降低了大家的预期，表示*它甚至还不如 Grok 3 mini*。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1389685898123280616)** (27 messages🔥): 

> `aider /architect mode, Local Model Recommendations, aider auto test, aider --yes-always and --no-always, Quantized Models` 


- **在 Aider 中使用 `/architect` 模式和编辑**：一位用户寻求关于执行 `/architect` 计划的澄清，注意到更改没有出现在仓库中，并发现使用 `/code` 会立即启动编辑。
   - 另一位用户建议编辑应该在完成后按回车键开始，而实验表明按回车键并非必要。
- **Qwen vs DeepSeek：Aider 的模型推荐**：用户讨论了本地模型的性能，推荐使用 **Qwen3 32b 或 30b** 而非 **deepseek/deepseek-chat**，原因是后者存在过度思考和 Commit 信息撰写的问题。
   - 一位用户在使用本地模型（70B Deepseek）时遇到了糟糕的性能，尽管拥有 RTX5000，目前正在寻求推荐。
- **使用 Aider Commit 进行自动化测试**：一位用户询问如何在每次 Aider Commit 后自动运行 `make test`。
   - 解决方案是在 Aider 设置中*开启自动测试并将测试命令设置为 make test*。
- **Aider 关于添加新文件的提示**：一位用户注意到 Aider 输出完整答案后，即使文件自请求开始以来未发生变化，仍会询问 *Add file to the chat?*。
   - 该用户引用了 [Aider FAQ](https://aider.chat/docs/faq.html#why-did-aider-ignorediscard-its-proposed-edits-after-it-asked-to-add-a-new-file-to-the-chat) 以获取更多细节。
- **Aider 的命令行选项**：一位用户询问 Aider 中 `--yes-always` 命令行选项的反向选项，寻求 `--no-always` 的等效项。
   - 回复中没有直接回答这个问题。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1389879513806147586)** (4 messages): 

> `NotebookLM use cases, NotebookLM as a personal daily journal, Audio overview function` 


- **NotebookLM 具有双重用途**：一位成员正在将 **NotebookLM** 设置为个人每日日记，用于记录**反思、媒体、Gemini 聊天和建议**，并将其作为文章、想法和参考资料的可搜索笔记数据库。
   - 他们计划使用 **Google Docs** 作为隐私和数据控制的单一事实来源（single source of truth），但正在寻找替代输入方法，以构建一个稳健且易于维护的系统。
- **NotebookLM 解释创作中的书籍**：一位成员使用 **NotebookLM** 来帮助解释他们正在创作中的书籍。
   - 他们主要使用**音频概览功能（audio overview function）**，因为它能为他们进行讲解。


  

---

### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1389707511187378207)** (21 messages🔥): 

> `NotebookLM sources, Podcast generation, Gas plant power factor, Pro vs Free accounts, Opening PDF sources` 


- ****NBLM 来源选择已澄清****：一位用户询问，在从特定来源创建思维导图后，重新激活所有来源是否会导致讨论话题从所有来源中提取，还是仅从最初选择的来源中提取。
- ****播客生成咨询出现****：一位用户请求关于如何为 "Audio overview" 生成更长播客的建议，寻求社区的想法和意见。
- ****燃气电厂功率因数咨询****：一位用户询问了燃气电厂的功率因数。
- ****寻求 NBLM Pro 与免费账户的区别****：NBLM 的一位新用户询问了 Pro 账户和免费账户之间的区别。
- ****在 NBLM 中打开 PDF 来源的请求****：一位用户询问是否有办法打开作为来源附加的 PDF 文件，并指出点击它会在来源窗口中显示得很乱。


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1389754687095570532)** (5 messages): 

> `Cohere open model weight release, CMD-R model, tool/agent frameworks, ML Summer School channel` 


- **Cohere 用户热切期待新的开源模型**：尽管认识到 **Cohere** 专注于企业市场，用户仍希望其发布新的或更新的开源模型权重。
   - 一位用户建议，发布一个强调 **tool/agent frameworks** 和更新架构的更新权重将会非常棒。
- **CMD-R 即将迎来一周年**：一位用户指出，08-24 版本的 **CMD-R** 模型已接近一年，但仍然非常可用。
   - 该用户认为，**Cohere** 的新权重将向竞争对手的现代开源权重展示其实力！
- **用户无法找到 ML Summer School 频道**：一位用户正尝试寻找 **#ml-summer-school** 频道，该频道在 [Cohere Labs Community 链接](https://sites.google.com/cohere.com/coherelabs-community/community-programs/summer-school)中被提及。
   - 该用户请求协助定位指定的频道。


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1390026203087503370)** (4 messages): 

> `Cohere Embedding Model, Trial key, Rate limits, Production key, Monthly limit` 


- **Trial Keys 可解锁 Cohere Embeddings，但有限制**：用户确认 **Cohere embedding model** 确实可以通过 **trial key** 访问。
   - 然而，他们提醒说，与 production keys 相比，**trial keys** 具有更严格的 **rate limits** 和每月使用上限。
- **Trial vs Production Keys：功能对等，使用上限不同**：澄清了 **trial key** 和 **production key** 都能解锁 Cohere 平台内的相同功能。
   - 主要区别在于 **trial key** 固有的 **monthly limit**，而 production keys 则没有这一限制。


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1389721267086561453)** (10 messages🔥): 

> `ML Summer School, Agentic AI, Computer Vision, Water Quality Monitoring` 


- **强化学习爱好者加入 Cohere**：来自印度的硕士生 Sriram 介绍了自己，他正致力于 **reinforcement learning** 和 **on-device safe AI**。
- **计算机视觉学生专注于语义分割**：二年级学生 Dojo 分享了他们在 **computer vision** 模型方面的工作，特别是 **semantic segmentation**。
   - 他们有兴趣减少早期网络层对最终层的冗余和依赖（无论是否有 skip connections），同时学习如何将现代 **NLP architectures** 和 **language modeling techniques** 应用于 **computer vision** 任务。
- **研究员集成 AI 用于水质监测**：FCT NOVA 的博士候选人 Oraib 专注于 **AI** 在 **water quality monitoring** 中的应用。
   - 他们的研究将 **satellite imagery** 与 **in-situ measurements** 相结合，为环境评估开发准确且可扩展的模型。
- **新成员寻找 ML Summer School 频道**：一位新成员正尝试寻找 **#ml-summer-school** 频道，参考了 [Cohere Labs Community Summer School 网站](https://sites.google.com/cohere.com/coherelabs-community/community-programs/summer-school)。
- **学生探索 Agentic AI 和平面设计**：Abdullah Abbasi 介绍自己是一名 **Agentic AI** 学生和平面设计师。


  

---

### **Cohere ▷ #[🔬-research](https://discord.com/channels/954421988141711382/1384974112841269399/1389925623174139925)** (3 messages): 

> `Secure ML, Privacy Preservation, AGI is here` 


- **用户声称 AGI 已经到来**：一名用户声称 *AGI is here*。
- **关于 Secure ML 和 Privacy Preservation 的辩论**：一名用户询问了关于 **Secure ML** 和 **Privacy Preservation** 的信息，要求详细阐述 **Secure ML** 的含义。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1389754103747575889)** (15 messages🔥): 

> `Mojo Origin Tracking System, Ownership and Life Cycles in Mojo, Mojo structs vs classes, GPU puzzles in Mojo, Dependent Type System in Mojo` 


- **深入探讨 Mojo 的 Origin Tracking System**：一名成员询问了关于 **Mojo origin tracking system** (borrow checker) 实现的演讲或文档。
   - 另一名成员分享了 [关于 Ownership 和 life cycles 的文档](https://docs.modular.com/mojo/manual/values/ownership) 以及 [lifetimes 和 references](https://docs.modular.com/mojo/manual/values/lifetimes)，同时还有成员分享了 [相关的 YouTube 视频](https://www.youtube.com/watch?v=9ag0fPMmYPQ)，并提到语言创始人计划最终会就该主题进行演讲。
- **Mojo Structs 与 Classes 的区别**：一名成员询问了 **Mojo structs** 与 **classes** 之间的区别。
   - 另一名成员分享了 [官方文档](https://docs.modular.com/mojo/manual/structs#structs-compared-to-classes) 的链接，其中列出了许多不同之处。
- **解析 GPU Puzzle 对 Barrier 的需求**：一名成员正在寻求关于 [GPU puzzles](https://puzzles.modular.com/puzzle_14/tiled.html#tile-processing-steps) 中 **matrix tiling 问题** 为何需要第二个 barrier 的澄清。
   - 另一名成员建议在论坛上发布该问题，以便专家提供更详细的解释。
- **Mojo 中的 Dependent Type System**：一名成员询问 **Mojo** 随着语言的成熟是否会包含更高级的概念，如 **graded types**，并引用了一篇关于该主题的研究论文。
   - 另一名成员回答说 **Mojo** 正在向 **dependent type system** 演进，但必须受到编译时可检查内容的限制，以平衡特性与编译时间及运行时性能。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1390061966453899304)** (2 messages): 

> `Mojo Offline Inference, QuantizationEncoding, LLM on M1 Mac, Nightly vs Stable Builds` 


- **Mojo 🔥 Nightly 版本中的 Offline Inference 困惑**：一名用户在 M1 Mac 上使用 nightly 版本尝试运行 [llm4decompile-v1.5/1.3B-Q6_K 模型](https://builds.modular.com/models/llm4decompile-v1.5/1.3B-Q6_K) 时，在遵循 [Modular Max offline inference 文档](https://docs.modular.com/max/serve/offline-inference/) 过程中遇到了问题。
   - 该用户遇到了与不支持的 `quantization_encoding` 相关的 `ValueError`。
- **Stable 版本解决了问题**：该用户报告称，使用 Mojo 🔥 的 stable 版本解决了 `quantization_encoding` 问题。
   - 该模型在 stable 版本指定的量化编码下按预期工作。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1389722734631718932)** (12 messages🔥): 

> `Making friends online, Manus AI new unlimited plan, NFT Scams` 


- **用户寻求在线交友以讨论 AI**：一名用户表达了希望与志同道合的人建立在线友谊以讨论 AI 相关话题的愿望，并指出在 **Discord** 和 **Reddit** 等平台上很难找到这样的联系。
   - 该用户强调需要那些在这些小众兴趣上志趣相投的朋友。
- **Manus AI 提供免费无限聊天！**：一名用户宣布 [Manus AI](https://manus.im/invitation/R9QW2MZDXXJJ4) 现在在聊天中提供 **无限免费消息**，并且基于 **Gemini**。
   - 另一名用户询问 Manus AI 是否有所改进，暗示它之前只是带有工具的 **Claude**，主要问题是其成本。
- **NFT 项目可能是一个骗局**：一名用户在 **Twitter** 上看到某个项目推广后，询问该项目是否在出售 **NFT**，另一名用户回答说这 *听起来像是个骗局*。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1390066300482617355)** (2 messages): 

> `独立研究导师指导` 


- **用户寻求独立研究导师**：一位有一定经验的用户正在寻求导师指导，以开始进行独立研究。
   - 该用户请求感兴趣的导师给他们发送私信（DM）。
- **初级研究者的导师指导机会**：一位中级水平的用户正在寻求关于启动独立研究项目的指导。
   - 鼓励感兴趣的导师通过私信联系，以提供具体的建议和支持。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1390066300482617355)** (2 messages): 

> `导师申请` 


- **成员请求独立研究导师指导**：一位*并非完全初学者*的成员请求关于如何开始进行**独立研究**的**导师指导**。
   - 该成员说明，如果有人感兴趣，请**给他们发送 DM**。
- **另一位成员寻求独立研究指导**：另一位成员也表达了希望获得导师指导以启动自己研究工作的兴趣。
   - 该成员的目标是从新手过渡到开展自主研究项目。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1389936262009323570)** (13 messages🔥): 

> `使用 GPT4All 进行平面图分析、图像识别限制、LM Studio 图像接收、ChatGPT 图像分析` 


- **GPT4All 寻求解析蓝图的模型**：一位成员正在寻找一个适用于 **GPT4All** 的优秀模型，用于处理有关建筑设计的问题，并能够讨论和评估上传的 **JPG** 平面图。
   - 另一位成员指出 **GPT4All** 无法上传图像，并对 **ChatGPT** 的能力提出质疑，认为它可能*过于想表现得有帮助*而给出不准确的答案。
- **成员辩论图像识别与 AI**：一位成员认为，虽然 **ChatGPT** 可以识别*树木*、*建筑物*、*面部*和*人*等物体，但它可能无法准确地*讨论*它们。
   - 另一位成员表示，在没有系统 prompt 的情况下使用 **ChatGPT** 处理图像时，模型发现出口未被标记。
- **LM Studio 可能接受图像**：一位成员指出，配合正确的模型，**LM Studio** 可以接受图像。
   - 一位成员展示了将图像输入 **ChatGPT** 的输出结果（[附件：Class_building_floor_plan.jpeg](https://cdn.discordapp.com/attachments/1090427154141020190/1389982683769344162/Class_building_floor_plan.jpeg?ex=686699c0&is=68654840&hm=9e7f70c07100c5c382138ec60c0a5bfed2eb7cd22243265cc9c848b70509875c&)）。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1389682419057037464)** (3 messages): 

> `LlamaExtract, LlamaCloud, 企业级 RAG, 多模态索引` 


- **LlamaExtract 自动生成 Schema**：新功能 **LlamaExtract** 现在可以根据文档和/或 prompt 自动生成 Schema，从而无需手动构建。
   - 只需要文档和描述即可，更多信息请查看[这条推文](https://twitter.com/llama_index/status/1940123273981043035)。
- **LlamaCloud 扩展企业级 RAG**：一篇新博客文章详细介绍了 **LlamaCloud 扩展企业级 RAG 的 4 种方法**，分享了在为大规模企业工作负载扩展 LlamaCloud 时学到的经验。
   - 该文章旨在帮助其他构建大规模文档索引和检索的人了解未来趋势，阅读[这条推文](https://twitter.com/llama_index/status/1940440399690248669)获取完整内容。
- **LlamaCloud 索引并检索图像**：你现在可以从 **LlamaCloud Indexes** 中检索图像和插图，以及文本，非常适合演示文稿和报告。
   - 启用此功能非常简单，只需切换“*Multi-modal indexing*”选项即可，详见[这条推文](https://twitter.com/llama_index/status/1940485676371530035)获取详情。

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1389927787359375400)** (6 messages): 

> `OpenAI Batch API, LlamaIndex Workflows 1.0, Embedding OpenAI API Key, Developer Collaboration` 


- **Batch API 的烦恼？**: 成员们正尝试因高昂成本将工作流迁移到 **OpenAI Batch API**，但 LlamaIndex 似乎没有内置的批量调用支持。
   - 建议是在其工作流中直接使用 **OpenAI lib** 进行批量调用。
- **LlamaIndex Workflows 1.0 发布！**: LlamaIndex 在[其博客](https://www.llamaindex.ai/blog/announcing-workflows-1-0-a-lightweight-framework-for-agentic-systems)上宣布发布 **Workflows 1.0**，这是一个用于 Agent 系统的轻量级框架。
- **环境变量是 API Key 的关键**: 一位成员在 LlamaIndex 的无代码 UI 中嵌入 **OpenAI API key** 时需要帮助，另一位成员指出 *LlamaIndex 只会查找环境变量*。
   - 他们澄清说，设置一个名为 `OPENAI_API_KEY` 的环境变量即可解决问题。
- **热情的开发者寻求合作**: 一位热情的开发者正在寻找项目合作并提供服务。
   - 他们参与过各种类型的项目，并渴望共同协作。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1389709645672743086)** (6 messages): 

> `Manus Prompt, MCP Server, Claude Opus 4, Qwen 3, 32B model` 


- **Manus 团队，请修复 Manus Prompt！**: 用户请求 Manus 团队修复他们的 Prompt，或者为包含 "image" 单词的消息添加一个微型 Prompt。
   - 用户标记了团队成员 <@1352272661589524572>。
- **在 Claude Opus 4 上构建 MCP Server**: 用户标记了一个团队，希望协助使用 **Claude Opus 4** 构建 **MCP Server**（就像一级方程式赛车一样）。
   - *它每天都会进维修站，因为你需要不断研究如何让它正常工作。*
- **Qwen 3, 32B 模型工作**: 用户提到有些功能在 **LM studio** 中配合 **Qwen 3, 32B 模型** 使用时可以开箱即用。
   - 用户包含了一个 [链接](https://files.catbox.moe/5gf51x.txt)。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1389987658423603281)** (3 messages): 

> `haldie style viz, tile viz approach, shared mem buffers, global buffers` 


- **Python 后端的 Haldie 可视化**: 一位成员展示了在 Python 后端实现的 "haldie 风格" 可视化，通过绿色表示读取、红色表示写入来突出内存访问模式。
   - 该 [演示](https://cdn.discordapp.com/attachments/1068976834928193609/1389987658020819056/haldie_viz.mov?ex=68669e62&is=68654ce2&hm=8458418f0dc82f9dcac83e6a7eaad13d1d4b79101c26945629d63322948c6bd6&) 可视化了上方的共享内存缓冲区（shared memory buffers）和下方的全局缓冲区（global buffers），专为 Metal 优化，参数为 `tc=3`。
- **Tile 可视化重新评估**: 一位成员重新审视了 Tile 可视化的想法，并意识到 Python 后端是实现它的绝佳场所，而之前在错误的层级进行了尝试。
   - 他们引用了一篇相关论文 [Simulating Time With Square-Root Space](https://arxiv.org/abs/2502.17779)，暗示其与该可视化方法存在联系。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/)** (1 messages): 

.pyrophoric.: 你好，有没有 CLI 工具可以自动按照 tinygrad 风格格式化代码？
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/)** (1 messages): 

chiggly007: 你这是什么意思？
  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1389712921084563618)** (1 messages): 

> `Autonomous Agents, Multi-Agent Systems, LangChain, AutoGen, AI Assistants` 


- **工程师提供构建自主 AI Agent 的服务**: 一位拥有 9 年经验的 AI 工程师提供使用 **GPT-4o**、**LangChain** 和 **AutoGen** 等工具构建、训练和部署 **AI 模型** 及 **自主 Agent** 的服务。
   - 他们寻求与初创公司或 AI 工具团队合作，擅长自主研究机器人、多 Agent 系统、AI 助手等。
- **AI 工程师技术栈亮点**: 该工程师的技术栈包括 **LangChain**、**Langraph**、**AutoGen**、**ReAct**、**CrewAI**、**DeepSeek**、**OpenAI**、**Claude**、**Hugging Face** 和 **Playwright**。
   - 他们在 **Deep Learning**（CNN、RNN、Transformers）、**NLP** 和 **Computer Vision** 领域也拥有专业知识。