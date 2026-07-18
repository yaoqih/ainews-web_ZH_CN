---
companies:
- moonshot
- openai
- thinking-machines
- artificial-analysis
- arena
- datacurve
- arcprize
- aisecurityinst
date: '2026-07-17T05:44:39.731046Z'
description: '**月之暗面（Moonshot AI）发布的 Kimi K3** 引发了业界对**中国开源权重模型**与全球技术前沿距离的重新评估。该模型在编程、智能体任务以及长程知识处理方面表现强劲。


  战略重心已从构建“算力护城河”转向开发“效率技术栈”，这涉及到 **MoE（混合专家模型）路由、量化、数据精选**，以及像 Moonshot 的 **“Mooncake”**
  架构这样由资源稀缺性驱动的基础设施。


  来自 Artificial Analysis、Arena、DeepSWE、ARC 和 Cyber 的基准测试均将 K3 列入顶级模型行列。其**智力指数（Intelligence
  Index）达到了 57 分**，且在编程智能体基准测试中与 **GPT-5.6 Terra** 和 **Claude Fable 5** 等模型旗鼓相当甚至有所超越。尽管关于
  K3 确切地位的讨论仍在继续，但它目前已被广泛公认为全球前沿技术领域的重要竞争者。'
id: MjAyNS0x
models:
- kimi-k3
- claude-fable-5
- opus-4.8
- gpt-5.6-terra
- gpt-5.5
- inkling
- glm-5.2
- gpt-5.6-sol
people:
- zhilin_yang
- kimmonismus
- anikasomaia
- dylan522p
- novasarc01
- scaling01
- theo
- hqmank
title: 今天没发生什么特别的事。
topics:
- moe-routing
- quantization
- data-curation
- infrastructure-design
- coding-agents
- benchmarking
- front-end-development
- software-engineering
- arc-benchmarks
- cybersecurity
---

**平静的一天。**

> 2026年7月16日至7月17日的 AI 新闻。我们检查了 12 个 subreddit、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有新增 Discord 信息。[AINews 网站](https://news.smol.ai/) 允许您搜索所有过往期数。提醒一下，[AINews 现在是 Latent Space 的一个板块](https://www.latent.space/p/2026)。您可以[选择订阅或退订](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)邮件发送频率！

---

# AI Twitter 回顾

**Moonshot 发布 Kimi K3、前沿模型定位以及关于中国/开源权重的辩论**

- **Kimi K3 是今日的焦点**：此次发布引发了对**中国开源权重（open-weight）**模型与前沿模型差距的广泛重新评估。多篇帖子将 K3 描述为该级别首个真正实用的中国模型，在编程、Agent 协作以及长周期知识型工作方面表现强劲。社区反应不一，从 [Salakhutdinov 向 Moonshot 创始人杨植麟表示祝贺](https://x.com/rsalakhu/status/2077892247194947601)，到从业者简单地报告 [“Kimi K3 真的非常非常好”](https://x.com/theo/status/2078071827021320425)。一个反复出现的主题是，K3 缩小了差距，足以迫使美国实验室加快发布进度，正如 [@kimmonismus](https://x.com/kimmonismus/status/2078066947594264679) 等人所言。

- **战略争论点从“算力护城河”转向了“效率栈”**：一个引人注目的推文串认为，K3 削弱了前沿能力主要受限于原始 FLOPs 的论点，转而指向 **MoE 路由、量化、数据清洗（data curation）以及由稀缺性驱动的基础设施设计**，例如 Moonshot 的 “Mooncake” 架构；参见 [@AnikaSomaia](https://x.com/AnikaSomaia/status/2077892561386299664)。相关评论强调，中国实验室可能正在压缩“单位 FLOP 能力”曲线，而不是直接拼西方国家的资本支出（capex）。[@dylan522p](https://x.com/dylan522p/status/2078084636719435959) 和 [@novasarc01](https://x.com/novasarc01/status/2078175010464948306) 认为更好的 Post-training（后训练）和模型转换率（harness conversion rates）可以非线性地缩小产品差距。

- **关于 K3 到底落后多少仍存在分歧**：一些人认为它已接近前沿水平，甚至在某些重要维度上超越了特定的西方模型；而另一些人则认为在更广泛的泛化性、效率或隐藏的测评（evals）中，它仍落后数月。参见 [@scaling01](https://x.com/scaling01/status/2077950993342316923) 持怀疑态度但详尽的描述，与 [@kimmonismus](https://x.com/kimmonismus/status/2078127331433230704) 和 [@theinformation](https://x.com/theinformation/status/2078219571475914905) 更加看好的观点形成对比。实际的共识更为明确：**K3 现在已无法被忽视**。

**基准测试：Artificial Analysis、Arena、DeepSWE、ARC、Cyber 和 FrontierCode**

- **Artificial Analysis 和编程 Agent 基准测试将 K3 稳固地置于顶级阵营**：[Artificial Analysis](https://x.com/ArtificialAnlys/status/2078165665278730490) 表示，在约六周时间内，其 Intelligence Index 评分超过 **51** 的前沿模型实验室从 2 个增加到 6 个。**Kimi K3 得分为 57**，落后于 **Claude Fable 5 (60)**，领先于 **Opus 4.8 (56)**。在编程 Agent 方面，[AA 随后报告](https://x.com/ArtificialAnlys/status/2078230240766345330) K3 在其 Coding Agent Index 中获得 **57** 分，与 **GPT-5.6 Terra** 和 **GPT-5.5** 持平，领先于 **Opus 4.8**。其 **Terminal-Bench v2 得分为 84%**，**DeepSWE 为 64%**，**SWE-Atlas-QnA 为 23%**。关于成本的说法不一：AA 称其为前沿级别且相对高效；[@theo](https://x.com/theo/status/2078215659948052984) 则反驳称，与 **GPT-5.6 Sol** 相比，Token 效率和吞吐量（throughput）往往会抵消其表面上的价格优势。

- **前端和编程评估中 K3 表现尤为强劲**：[Arena 报告](https://x.com/arena/status/2078208547457012005)称，K3 首次让**中国在 Frontend Code Arena 上领先于美国**。用户测试也印证了这一点，K3 在基于视觉的前端任务中可以超越或媲美 Fable，例如 [@hqmank 的地球仪表盘测试](https://x.com/hqmank/status/2078104317027094907)。在软件工程方面，[DataCurve](https://x.com/datacurve/status/2078189882707730535) 表示 K3 在 **DeepSWE** 首次亮相即排名 **第 3**，并称其为首个在该测试中达到前沿水平结果的开源权重模型。

- **ARC and cyber remain useful reality checks**：[ARC Prize 证实](https://x.com/arcprize/status/2078141332938523032) **Thinking Machines’ Inkling** 目前是 **ARC-AGI-1 (79.5%)** 和 **ARC-AGI-2 (36.5%)** 上得分最高的权重开放模型，而关于 K3 的 ARC-AGI-2 评分的猜测仍在通过 [BenchPress 估算](https://x.com/scaling01/status/2078180784356135139)继续。在网络安全（cyber）方面，英国 AISI 相关的讨论，如 [GLM-5.2 在 “The Last Ones” 上匹配 Opus 4.5](https://x.com/AISecurityInst/status/2078103153988243873) 以及 [OpenAI 声称 GPT-5.6 Sol 在该领域处于 SOTA 状态](https://x.com/OpenAI/status/2078243667081617826)，都强调了即便差距正在缩小，**开源模型在长程网络安全（long-horizon cyber）方面似乎仍然实质性地落后于最顶尖的闭源模型**。

**Model Architecture, Inference, and Systems Work**

- **Kimi Delta Attention 引起了浓厚的技术兴趣**：[@sdrzn](https://x.com/sdrzn/status/2078210052150997006) 撰写的一篇强力的技术解读突出了 K3 使用的 **Kimi Delta Attention (KDA)** 是一种快速权重（fast-weights）风格的记忆机制，能够有效地为每个请求维持固定大小的学习状态，而不是在长上下文中支付完整的 Attention 成本。据称其回报是 **在 1M 上下文下吞吐量提升高达 6 倍且成本更低**，并且在长上下文长度下定价保持更平稳。如果这些特性在更广泛的部署中得以维持，这将是该发布版本中最重要的架构级想法之一。

- **服务和硬件讨论紧随其后**：人们已经准备在异构基础设施上部署 K3，例如 [基于 RoCE 的 4xH100 节点](https://x.com/TheZachMueller/status/2078076002241069525)；同时 [华为的 “950 SuperPoD” 发布](https://x.com/zephyr_z9/status/2078028640059859312) 为 “受限下的中国 AI 技术栈扩展” 的叙事增添了动力。在软件方面，[vLLM + AMD 支持](https://x.com/AnushElangovan/status/2077936618779119841)、[Red Hat AI 在带有 vLLM 的 DGX B200 节点上运行 Inkling](https://x.com/RedHat_AI/status/2078195299885965745)，以及 [vLLM 官方关于在每月约 2,000 次 commit 下维持生产质量的笔记](https://x.com/vllm_project/status/2078234327843062169) 都是相关的基础设施更新。

- **Kernel/性能工程仍然是差异化优势**：K3 在 Kernel 编写和性能工程能力方面反复受到称赞，包括 [来自 Moonshot 员工的 kernelbench 相关示例](https://x.com/Xinyu2ML/status/2078041418329960645) 以及 [社区评论称 K3 协助设计了 kernelbench.com 本身](https://x.com/elliotarledge/status/2078050598419927387)。另外，[Simran Arora 指出](https://x.com/simran_s_arora/status/2078167541906874464) **混合线性注意力（hybrid linear attentions）、全模型 megakernels 以及 AMD aiter 中的快速 MLA/DSV4 解码内核** 现在正直接反哺前沿模型的开发。

**Agents, Memory, MCP, and Workflow Scaffolding**

- **价值正从基础模型访问转向评估框架（harnesses）和工作流**：多篇帖子认为，随着前沿智能变得更便宜、更开放，持久的护城河将转移到 **编排（orchestration）、记忆、工具和特定领域的脚手架（scaffolding）**。[@jmorgan](https://x.com/jmorgan/status/2078155090729599375) 和 [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2078163463097250072) 给出了很好的总结，后者将关键区别框架化为 **价值最大化（valuemaxxing）vs Token 最大化（tokenmaxxing）**。

- **记忆架构正向 “wiki memory” 收敛**：[Paulius Ztin 的长文](https://x.com/pauliusztin_/status/2078094872717017107) 是这里最具体的设计文章之一。其提议是：Agent 应该停止重复从原始文档中重新推导相同的理解，而是应该在统一记忆之上构建一个特定任务的 **Markdown wiki 层**，并通过 **FastMCP** 进行同步。在同一领域，[Qdrant 分享了关于多租户检索的生产指南](https://x.com/qdrant_engine/status/2078064671022887093)，随后强调了 [mem0 的观点：持续学习与其说是权重更新问题，不如说是记忆问题](https://x.com/qdrant_engine/status/2078147719437197733)。

- **MCP 和技能抽象不断成熟**：值得注意的产品更新包括 [Perplexity Agent API 添加自定义技能](https://x.com/perplexitydevs/status/2078213550770991107)、[来自 Nous 的 Hermes Agent 桌面和 Unreal Engine 伴侣技能](https://x.com/NousResearch/status/2078168128693977291)，以及 [来自 Tadas + Anthropic 的 Dom 的高级 MCP 使用模式](https://x.com/tadasayy/status/2078193533362843929)。在研究方面，[MemoHarness](https://x.com/omarsar0/status/2078122558059327745) 脱颖而出：它将 Agent 评估框架分解为六个可编辑的控制面，并在 Shell-Agent 上报告分数为 **0.806**，而最强的固定框架基线为 **0.722**，同时降低了每个任务的成本。

**Research Notes Beyond K3**

- **鲁棒性与检测器限制**：论文 **“The Illusion of Robustness”** 指出，综合准确率掩盖了无关上下文下的预测翻转（prediction flips）；参见 [arXiv 链接](https://x.com/HEI/status/2077895288706978001) 和 [日文摘要](https://x.com/compassinai/status/2078145391250506224)。另外，[Epoch AI 报告](https://x.com/EpochAIResearch/status/2078195357599813723) 称 AI 检测器在纯人类文本和初级 AI 文本上通常是可靠的，但**被指示模仿特定作者的 LLM 可以逃避检测**，在科学写作中的**漏报率（false negatives）约为 13% 至 26%**。

- **具身学习与仿生学习**：[NVIDIA 的 RoboTTT](https://x.com/dair_ai/status/2078123816786813115) 将机器人策略的上下文长度扩展了 **3 个数量级**，相比单步基准测试，操作性能提升了 **87%**，并完成了一个包含十个阶段、时长五分钟的组装任务，而没有任何基准模型能完成该任务。同时，[Sakana 的 “Diffusing Blame”](https://x.com/SakanaAILabs/status/2078136419521048905) 和 [Hardmaru 的总结](https://x.com/hardmaru/status/2078156625479921847) 展示了在严格的 **Dale’s principle** 下，无需标准反向传播（backprop）权重传输的竞争性学习效果。

- **可解释性 / 表征几何**：[Elie Bakouch 在 Thinking Machines 的 Inkling 上复现了 Anthropic 风格的 j-space 分析](https://x.com/eliebakouch/status/2078180531456573874)，发现其在早期层和晚期层之间保持相似几何形状的特性极不寻常（**早期-晚期 CKA 约为 0.8，而其他模型约为 0.5**）。同一推文线程还报告了 Poolside 的 Laguna XS 2.1 在 **NVFP4 量化下 j-space 变化极小**。

**热门推文（按互动量排序，已过滤技术相关性）**

- **开源模型与闭源模型的经济学**：[@AravSrinivas 将这一时刻比作 Sun Microsystems 被开源 + 通用硬件颠覆的时刻](https://x.com/AravSrinivas/status/2078189971723231567)，认为本地/开源模型可能对现有巨头产生类似的通缩效应。
- **美国政策影响**：[@DavidSacks 表示 K3 在 Frontend Code Arena 夺冠是对过度监管和数据中心限制的一个警告](https://x.com/DavidSacks/status/2078092271296143593)。
- **价格崩盘叙述**：[@chamath 强调了极廉价和极昂贵的前沿 Token 之间不断扩大的价差](https://x.com/chamath/status/2078075083914957254)。
- **开放权重扩散的影响**：[@shadcn 指出，曾经被视为政府敏感的能力正迅速以商品价格向订阅者开放](https://x.com/shadcn/status/2077996062384480268)。
- **前沿编程现状**：[@datacurve 发布的 K3 的 DeepSWE 结果](https://x.com/datacurve/status/2078189882707730535) 和 [@arena 的 Frontend Code Arena 排名变化](https://x.com/arena/status/2078208547457012005) 是最明确的基准信号，表明此次发布的影响力超越了社交媒体的炒作。

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. Kimi K3 发布与编程基准测试

  - **[Kimi K3 权重将于 27 日发布。](https://www.reddit.com/r/LocalLLaMA/comments/1uyb88e/kimi_k3_weights_to_be_released_on_the_27th/)** (热度: 587): **该[图片](https://i.redd.it/lg3io1qxxmdh1.png)是 **Kimi K3** 的**技术发布公告**，声明该模型已在 `kimi.com`、Kimi App、Kimi Work、Kimi Code 和 Kimi API 上线，默认推理/思考强度设置为 **“max / extreme”**，后续计划推出较低强度模式。根据链接中已验证的微信文章和英文博客，**全量 Kimi K3 模型权重计划于 2026 年 7 月 27 日发布**，并附带技术报告详情。** 评论者对开放权重计划持积极态度，但指出该模型对于普通的本地推理来说可能过大；一条评论调侃说，肯定会有人声称在拥有 `24 GB` VRAM 的笔记本电脑上以 `0.01 tok/s` 的速度运行 `2.8T` 模型。

    - 评论者指出，**Kimi K3** 预计规模极巨——有人提到了 `2.8T` 参数量级——这使得真正的本地推理对大多数用户（尤其是使用 `24 GB` VRAM 笔记本电脑等消费级 GPU 的用户）来说是不切实际的。其技术价值主要在于开放权重和 API/托管推理，而非现实的桌面端部署。
    - 一个具有技术深度的话题讨论认为，**MoonshotAI** 可能会从一个较小的伴生模型中受益，类似于 **DeepSeek** 的工作流拆分：使用最大的模型进行规划/策略，使用更便宜、更小的模型进行执行。评论者特别建议使用 **sub-`300B` MoE** 或更小的模型，来配合 Kimi 的大模型处理较轻的编程工作负载。
    - 一位用户强调了在 Moonshot 编程模型中观察到的迭代，称 **K2.7 Code** 比 **K2.6** 和 **K2.5** 有所改进，并期望 **K3** 通过 Moonshot 自身的 API 推理在 Agent 编程（agentic coding）方面表现强劲。重点在于在多步编程/Agent 工作流中测试 K3，而非本地执行。

- **[Kimi K3 Benchmarks](https://www.reddit.com/r/LocalLLaMA/comments/1uy9cft/kimi_k3_benchmarks/)** (热度: 1951): **该[图片](https://i.redd.it/yuyk4c99mmdh1.jpeg)是一个关于 **Kimi K3** 的编程基准测试排行榜，显示其在多项测试中名列前茅：在 **Program Bench** (`77.8`) 和 **SWE Marathon** (`42.0`) 中排名 `#1`；在 **Terminal Bench 2.1** (`88.3`)、**FrontierSWE** (`81.2`) 和 **Kimi Code Bench 2.0** (`72.9`) 中排名 `#2`。该图表将 Kimi K3 描述为可与标记为 **GPT-5.6 Sol**、**Fable 5**、**Opus-4.8**、**GPT-5.5** 和 **GLM-5.2** 的模型相媲美，但帖子未提供方法论、数据集定义、评估设置或独立验证，因此技术层面的参考价值仅限于其自称的基准排名。** 评论认为该图表证明了中国的前沿模型可能仅比美国模型落后 *“6 天”* 而非数月，而另一条评论则开玩笑/推测 `2TB VRAM` 是实际运行的硬件门槛。所提供的评论中未出现更深层次的方法论辩论。

    - 一位评论者将发布的基准测试图片解读为：中国的前沿模型现在已经极其接近美国模型，称它们看起来 *“落后甚至不到 6 个月”*，可能仅 *“落后 6 天”*，同时明确说明这是基于基准测试而非实际落地使用。

  - **[Kimi K3 is top of nextjs eval](https://www.reddit.com/r/LocalLLaMA/comments/1uza5wb/kimi_k3_is_top_of_nextjs_eval/)** (热度: 464): **该图片是 **Guillermo Rauch** 在 X 上发布的帖子截图，声明 **Kimi K3** 目前在 **Next.js evals** 排行榜上表现最佳，在 Web 工程基准测试中超越了专有模型，拥有相当的成功率但完成时间更短。技术意义在于，根据该帖子，这可能是第一个领跑全面 `nextjs.org/evals` 基准测试的 **open model**；相关链接：[图片](https://i.redd.it/rvqq7abkgudh1.png), [Next.js evals](https://nextjs.org/evals)。** 评论集中在实际部署问题上，包括可能的高内存需求（*“给我 1 TB 的 DDR6”*）以及 **Kimi K3** 是否真的是开源的，或者从何处可以获取。

    - 一位评论者链接了官方的 **Next.js evals** 排行榜地址 https://nextjs.org/evals，这是验证 **Kimi K3** 排名第一说法的相关来源。另一位评论者质疑该基准测试的实用性，认为由框架维护自己的评估套件可能会限制结果在 Next.js 特定任务之外的泛化能力。

  - **[KIMI K3 Beats Claude Fable and GPT 5.6 sol in arena.ai!!!](https://www.reddit.com/r/LocalLLaMA/comments/1uydii0/kimi_k3_beats_claude_fable_and_gpt_56_sol_in/)** (热度: 2465): **该图片是一张**技术排行榜截图**，而非梗图：它显示了日期为 2026 年 7 月 16 日的 [Code Arena WebDev 综合排名](https://i.redd.it/sry915x7dndh1.png)，**Moonshot 的 `kimi-k3` 排名第 1**，积分为 `1679`，领先于 `claude-fable-5` 和 `gpt-5.6-sol-xhigh`。帖子认为这令人惊讶，因为 Kimi K3 击败了那些被描述为“过于危险”而无法公开发布的模型，而一位评论者指出，这一结果**并非来自文本排行榜**，而是来自 WebDev/代码竞技场场景，并链接至 [arena.ai/leaderboard/text](https://arena.ai/leaderboard/text)。** 评论虽然印象深刻但持谨慎态度：一位用户开玩笑说“中国现在落后西方 6 天”，而另一位用户关注 `kimi-k3` 是否真的会 **open weights**（开放权重），暗示如果模型可以广泛获取，其排名将更具意义。

    - 一位评论者引用了 **arena.ai 文本排行榜** ([arena.ai/leaderboard/text](https://arena.ai/leaderboard/text)) 并澄清 KIMI K3 并未在“文本竞技场”中领先，但在参考截图中显示其排名接近 **Gemini 3 Pro** 和 **GPT 5.6 sol (xhigh)**，他们认为这在技术上令人印象深刻。
    - 几位评论者关注 **KIMI K3** 是否会以 **open weights** 形式发布，因为与 **Anthropic/OpenAI** 等仅提供闭源 API 的供应商相比，这将实质性地改变部署经济性。
    - 一条技术/经济相关的讨论链指出，如果 KIMI K3 具有竞争力且可以在本地运行，企业可以用自有推理硬件取代高额的 API 使用：该评论者估计在 **Q4** 运行它大约需要 **`$100k` 的前期硬件支出**，并将其与大型组织据称每月花费 **`$1M+`** 在 API 调用上的成本进行了对比。

- **[Kimi K3 在 ArtificialAnalysis 上获得第 3 名，超越 Claude Opus 4.8](https://www.reddit.com/r/LocalLLaMA/comments/1uycepz/kimi_k3_achieves_3rd_place_on_artificalanalysis/)** (活跃度: 1072): **该图片是来自 **Artificial Analysis** 的技术基准测试图表，显示 **Kimi K3** 在智力指数（Intelligence Index）中排名 **第 3**，得分为 `57`，位列 **Claude Fable 5** (`60`) 和 **GPT-5.6** (`59`) 之后，并以微弱优势领先于 **Claude Opus 4.8** (`56`)。该帖子将其视为 open-weight/proprietary 竞争力的一个显著里程碑；评论者还指出相关的 cost-per-task 和 output-token-per-task 数据“非常有前景”。[图片链接](https://i.redd.it/5vorrnbx5ndh1.png)** 评论者对仅仅依赖于另一个排行榜持怀疑态度，要求提供长会话用户报告以及在 “Sonnet costs 和 30 t/s” 下的实际推理效率。其他人认为 open-weight 模型可能很快会超越 proprietary 产品，从而增加了 Anthropic 的定价压力。

    - 评论者关注 Kimi K3 在 ArtificialAnalysis 的排名是否能转化为持续的真实世界性能：有人要求提供 *“与其进行长会话”* 的报告，而不是更多的基准测试图表，并指出在 **类似 Sonnet 的定价** 和大约 `30 tokens/s` 的速度下，它需要具备极高的推理效率才能证明其采用价值。
    - 链接的图表据称显示 Kimi K3 在 **cost per task** 和 **output tokens per task** 方面表现强劲，一位评论者称这两项指标 *“非常有前景”* ([图表](https://preview.redd.it/ayxi7od6bndh1.png?width=1753&format=png&auto=webp&s=14190215c0ae612463e1d7e9a7587b2d5e0c5b48))。这引发的技术暗示是，其竞争力不仅来自于原始基准测试分数，还来自于完成每个任务时更低的 token 使用量。
    - 几条评论将 Kimi K3 视为 **open-weight / 中国模型正在追赶 proprietary 旗舰模型** 的证据，特别是将其与 Claude Opus 4.8 进行对比，并提到其接近所谓的美国旗舰模型，如 **Fable 5** 和 **GPT-5.6**。Anthropic 面临的定价压力是一个反复出现的技术/商业观点：如果开源模型的基准测试质量接近闭源模型，评论者质疑高端 API 定价如何维持其合理性。


### 2. 本地推理压缩与加速

  - **[Bonsai 27B 在 iPhone 上本地运行 - 3.9GB 运行 27B 模型](https://www.reddit.com/r/LocalLLaMA/comments/1uyz9n2/bonsai_27b_runs_locally_on_an_iphone_a_27b_model/)** (活跃度: 523): ****PrismML 的 Bonsai-27B** 是一款基于 Qwen3.6-27B 衍生的模型，量化为纯二进制 `g128`：每个权重为 `1-bit` 符号位加上每 128 个权重组共享的一个 FP16 缩放因子，从而实现约 `1.125 bits/weight`，并在 [Hugging Face](https://huggingface.co/prism-ml/Bonsai-27B-mlx-1bit) 上提供了一个 **3.9 GB** 的 MLX 权重。帖子声称它可以通过 Atomic Chat 在 **iPhone 15 Pro Max / 8 GB RAM** 上本地运行，在 15 个基准测试中保留了约 `89.5%` 的 FP16 基准性能（`76.1` vs `85.1`），在 4K 上下文时估算内存约为 `5.2 GB`，在使用 4-bit KV cache 的 100K 上下文时约为 `6.8 GB`。** 评论者关注的一个令人惊讶的事实是，**所有主要层**（包括 embeddings、attention/MLP projections 和 LM head）都被二进制化，且没有高精度例外；有人指出这正是许多 1-bit 方案通常失败的地方。也有人对其与基于 Qwen/Gemma 的 9B 微调模型相比的实际质量表示怀疑/好奇，并担心手机电池/散热的影响。

    - 评论者强调，Bonsai 报告的 **全二进制 / 1-bit 量化** 非常罕见，因为许多 “1-bit” LLM 方法会保留敏感组件在高精度。一种技术解读是，在压缩模型 *所有* 部分的同时保留约 `~90%` 的基准质量令人印象深刻，但由于这些能力对精度损失更敏感，预计 **知识和推理能力的退化** 是必然的。
    - 一种怀疑的对比观点认为，宣传的 `~90%` 基准保留率在实际任务中可能只相当于 `~30–40%` 的效能，将该模型的实际质量比作极低比特的 `IQ2XXS` 风格 `27B` 量化模型。同一位评论者质疑在手机上运行 **dense 27B** 1-bit 模型的效率，因为尽管节省了内存，推理仍然需要在所有 `27B` 参数上进行计算。
    - 演示中的一个观察结果是快速的电量消耗：据报道，iPhone 电池在不到一分钟内下降了约 `2 个百分点`。虽然这只是个例，但它表明 dense `27B` 模型的本地推理在移动硬件上受到的限制与其说是存储大小，不如说是 **散热和电池极限**。

- **[DFlash 使 Qwen3.6 27B 提速 2.2 倍且無質量損失](https://www.reddit.com/r/LocalLLaMA/comments/1uyay0w/dflash_makes_qwen36_27b_22x_faster_with_no/)** (Activity: 488): **在單張 RTX 6000 上，作者在 Atomic.Chat 中對 [`Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B) 進行了基準測試，對比了基準解碼、MTP 和 DFlash 在四種本地提示詞（快速排序、JSON 生成、邏輯謎題和科幻散文）下的表現。報告的吞吐量分別為基準 `44 tok/s`、MTP `65 tok/s`（`1.45x`，`71%` 接受率）和 DFlash `98 tok/s`（`2.20x`，`30%` 接受率）；DFlash 順序草擬 `15` 個 Token，在 JSON 等結構化/重複性輸出中達到峰值（`152 tok/s`，`3.4x`），但在投機 Token 被拒絕時，創意文本的表現可能低於基準（`42` 對 `44 tok/s`）。作者聲稱“輸出相同”/無質量損失，但證據似乎僅限於小規模提示詞集的精確輸出對比，而非更廣泛的評估。** 評論者質疑“無質量損失”是如何衡量的，指出在複雜任務上的側向對比（side-by-side）往往會暴露出性能退化。其他人則詢問當模型未完全 GPU-offload 時 MTP/DFlash 是否仍然有效，並要求在更長上下文下進行測試，暗示報告的提速可能取決於工作負載和內存放置。

    - 評論者對 **“無質量損失”** 的說法提出挑戰，詢問使用了何種評估方法。擔憂之處在於，即使標題化的基準測試或吞吐量測試報告沒有損失，但在複雜任務的側向測試中通常會發現退化。
    - 幾個技術問題集中在部署限制上：當 LLM **未完全卸載到 GPU** 時，**MTP** 或 **DFlash** 是否仍能提高吞吐量，以及為什麼用戶在部分卸載（partial-offload）配置下可能看不到加速。
    - 用戶對 **VRAM/上下文長度的權衡** 表現出興趣：具體而言，在相同的 VRAM 預算下，使用 **MTP vs. DFlash vs. baseline** 會損失多少可用上下文。另一位評論者暗示基準測試應包括長上下文場景，因為隨著 KV-cache 壓力的增大，加速結果可能會有實質性差異。

  - **[DeepSeek V4 Flash (98GB) 在單張 4060ti + CPU 上本週提速 300% [ 2-&gt;7t/s]](https://www.reddit.com/r/LocalLLaMA/comments/1uy33fw/deepseek_v4_flash_98gb_on_1x_4060ti_cpu_got_300/)** (Activity: 370): **該圖是一張技術基準測試截圖，而非梗圖：它對比了運行 **DeepSeek-V4-Flash GGUF** 的 `llama.cpp` 版本 **b9986** 與 **b10034**，顯示在配備 **RTX 4060 Ti 16GB + Ryzen 5 9600X + 138GiB DDR5 RAM** 的廉價混合配置上，吞吐量從約 `2.1 tok/s` 提升至 `7.5–7.6 tok/s`。帖子將這一飛躍歸功於最近 `llama.cpp` 的更改，評論特別指向了 [ggml-org/llama.cpp PR #25545](https://github.com/ggml-org/llama.cpp/pull/25545)，而另一個即將到來的優化 [PR #25585](https://github.com/ggml-org/llama.cpp/pull/25585) 和 [fairydreaming 的 `dsv4` 分支](https://github.com/fairydreaming/llama.cpp/tree/dsv4) 據報在某些 CPU-offload 配置中還能再增加約 `10%`。圖片：[https://i.redd.it/2t5n2foyeldh1.png](https://i.redd.it/2t5n2foyeldh1.png)** 評論者將此視為大模型 CPU/GPU 卸載的重大實用改進，儘管有人指出考慮到不尋常的 `138GB` DDR5 RAM 配置，標註“廉價配置”仍有爭議。另一位評論者報告稱，同樣的更改使得完整的 `162GB` 模型能在 P40s/MI50s 等舊加速器上以約 `7.6 tok/s` 運行，並推測 `-sm tensor` 加上 MTP 可能超過 `30 tok/s`。

- 有评论者将大幅加速归功于 **llama.cpp PR [#25545](https://github.com/ggml-org/llama.cpp/pull/25545)**。据报告，DeepSeek V4 Flash 在 **P40s 和 Mi50s** 上从无法装入/运行 VRAM 提升到了在完整 **`162GB`** 模型上达到 **`7.6 tok/s`**。他们指出目前仍在使用 `-sm layer` 且未开启 MTP，并估计将 **MTP** 与 `-sm tensor` 结合使用可能将吞吐量推高至 **`30 tok/s`** 以上。
- 另一个技术数据点指向 **llama.cpp PR [#25585](https://github.com/ggml-org/llama.cpp/pull/25585)** 和 **fairydreaming 的 `dsv4` 分支** ([GitHub](https://github.com/fairydreaming/llama.cpp/tree/dsv4))，将其视为进一步优化的路径。一位用户报告该分支比目前的 master 分支快约 **`10%`**，在 **5090 (32GB VRAM)** 配合 **CPU offload 和 `96GB` 系统 RAM** 的环境下，使用 **UD-IQ3_S** 量化达到了约 **`12–14 tok/s`**。
- 几项调优建议集中在 CPU/GPU 分割行为上：通过 `-t 6`（每个 P-core 使用一个线程）禁用类超线程式的过度订阅；使用 `-fa off` 测试 Flash Attention 状态；以及使用 `-nkvo` 避免将 KV/Context 分别放置在 GPU 和 CPU 上。另一位评论者建议，像 **EPYC** 这样具有高内存带宽的平台可以将 CPU offload 推理速度提升至 10+ tok/s，而 **MTP** 可能会使 Agent 化的用途变得更加可行。

- **[Trellis.cpp 现在可以生成高质量资产](https://www.reddit.com/r/LocalLLaMA/comments/1uyw64s/trelliscpp_now_produces_high_quality_assets/)** (热度: 467): **trellis.cpp** 是 **TRELLIS.2 图像转 3D 资产生成流水线**的 GGML 移植版本。据报道，它已修复了多个影响质量的 Bug，现在其输出质量已达到参考实现的水平，从而支持在非 CUDA 后端（包括 CPU 执行）上进行开源 3D 生成。原始引擎可在 [`github.com/pwilkin/trellis.cpp`](http://github.com/pwilkin/trellis.cpp) 获取，并通过 [Lemonade](https://lemonade-server.ai/) 进行集成，实现包括可选的文本转 3D 级联（text-to-3D cascading）在内的端到端工作流。评论区要求将其与最近的[基于 Hunyuan 的本地图像转 3D 重建流水线](https://www.reddit.com/r/LocalLLaMA/comments/1uuga40/local_image_to_3d_2gb_ram_20s_apple_silicon_iphone/)在质量和速度上进行对比，质疑输出结果是否真正称得上“高质量”而非仅仅是“高细节”，并索要复现展示结果所需的准确 TRELLIS.2 参数。

    - 有评论者提到了之前的一个**基于 Hunyuan 的本地图像转 3D 重建**工作流，据称该工作流在 Apple Silicon/iPhone 级硬件上运行约需 `20s`，占用约 `2GB RAM`，并询问 **Trellis.cpp** 在质量和速度上如何与其竞争：[Reddit 参考](https://www.reddit.com/r/LocalLLaMA/comments/1uuga40/local_image_to_3d_2gb_ram_20s_apple_silicon_iphone/)。这引出了核心技术问题：Trellis.cpp 提升的资产细节是否以比轻量级 Hunyuan 重建更高的计算/内存成本为代价。
    - 一位用户询问 **trellis.2** 使用了什么样的推理设置，并提到他们本地应用生成的输出质量明显低于帖中的示例。这表明输出质量可能对参数高度敏感，可能取决于采样设置、分辨率/细节控制、预处理或后处理，而非仅仅取决于模型本身。
    - 一项游戏资产工作流对比提到 **Meshy** 在处理树木/风景图像时生成的几何结构（geometry）较差，例如将其转换成“丑陋的枯枝堆”或错误的物体（如汽车）。其中隐含的技术问题是，即使以物体为中心的示例看起来很强，目前的图像转 3D 工具在处理复杂的自然形状和场景级输入时可能仍然很吃力。

## 较少技术性的 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Kimi K3 发布与基准测试热潮

- **[中国版 Fable 5 来了！！又名 Kimi K3](https://www.reddit.com/r/singularity/comments/1uy3oij/chinese_fable_5_is_here_aka_kimi_k3/)** (热度: 1223): **该图片是在网页端宣布 **Kimi K3** 的社交帖子截图，声称具有 **`1M` Context Window**，UI 中包含 **K3 Max** 等模型选项（[图片](https://i.redd.it/uv4k60ydlldh1.jpeg)）。在 Reddit 标题将其称为“中国版 Fable 5”的背景下，该帖将 Kimi K3 描绘为中国重大的前沿模型发布，特别强调了 AI 生成的前端/动态图形能力，而非提供正式的基准测试结果。** 评论者报告了早期印象，认为 Kimi K3 感觉**比 Claude 快但准确度稍低**，大致可与 “GPT 5.5” 媲美，但低于 “5.6 或 Fable”。一个值得注意的技术担忧是，其思维链（CoT）据称引用了 **Anthropic 的内容政策**，引发了关于训练数据或模仿痕迹（imitation artifacts）的猜测。

- 早期用户印象认为 **Kimi K3 / “Chinese fable 5”** 的运行速度 *比 Claude 更快*，但准确度较低，大致与 **GPT-5.5** 相当，在感知的回答质量上落后于 **GPT-5.6/Fable**。这属于轶事传闻而非正式的基准测试（benchmarked），但它将该模型定位在具有延迟竞争力的同时，但在顶级准确率方面仍显落后。
- 一位评论者报告称，该模型暴露的推理/思维链（chain-of-thought）明确引用了 **Anthropic 内容政策**，这表明可能存在训练数据污染、政策蒸馏（policy distillation）或 Prompt/政策泄露的痕迹。另一张链接的截图被描述为让该模型看起来“简直就是 Claude”，进一步加剧了人们对其行为或内部政策轨迹可能类似于 Anthropic 系统的担忧。
- 一份技术细节对比强调 **MiniMax M3** 被低估了，评论者声称在他们的工作负载中，它始终优于 **DeepSeek v4 Pro** 和 **Mimo 2.5 Pro**。他们提到 MiniMax 的付费 Token 方案为 `1.7B tokens / $20/month` 并提供 API 访问权限，并描述 `agent.minimax.io` 提供了一个配备 `2GB RAM`、“无限”存储空间和 `1 Xeon vCore` 的 **Debian 12 沙箱**，可通过 `cloudflared` 隧道用于 API 测试。

- **[Kimi K3 登顶 Frontend Code Arena](https://www.reddit.com/r/singularity/comments/1uybldp/kimi_k3_tops_frontend_code_arena/)**（活跃度：1637）：**该图片是一个 **Frontend Code Arena 排行榜**，其中 **Kimi-K3** 以 `1,679` 的竞技场得分位居第一，领先于 **Claude Fable 5**（`1,631`）和 **GPT-5.6 Sol**（`1,618`）：[图片](https://i.redd.it/pg72x8ul0ndh1.jpeg)。其技术意义在于，评论者认为 Kimi-K3 值得关注，因为它被声称是**开放权重（open weights）**，且成本约为 Claude Fable 的 **1/3**，同时在此前端代码基准测试中表现优于闭源前沿模型。** 评论称赞了 Kimi 明显的性价比和开放权重的定位，同时批评 Google/Gemini 未能出现在排行榜上。一些评论者还针对美国可能施压阻止发布或使用 Kimi 3 权重进行了政治推测，但这属于推测而非技术证据。

    - 评论者强调，据报道 **Kimi K3** 在 **Frontend Code Arena** 中处于领先地位，而成本仅为 **Fable** 的约 `1/3`，且作为**开放权重**发布，这使得该结果在基准测试性能和部署成本两个维度都非常显著。
    - 几条评论将这一结果视为中国实验室在基准测试上的重大飞跃，暗示 Kimi K3 可能预示着在编程/前端生成基准测试中更广泛的竞争力的到来，而非偶然结果。一个技术观察是，尽管 Google 拥有巨大的算力/数据优势，但 **Gemini** 在图表对比中却缺席了。

- **[Kimi K3 在 ArtificalAnalysis 上获得第三名，击败 Claude Opus 4.8](https://www.reddit.com/r/singularity/comments/1uyc6sk/kimi_k3_achieves_3rd_place_on_artificalanalysis/)**（活跃度：1009）：**该[图片](https://i.redd.it/zb3akgoc4ndh1.png)是来自 **Artificial Analysis Intelligence Index** 的基准测试柱状图，声称 **Kimi K3** 以 `57` 分排名**第三**，落后于 **Claude Fable 5**（`60`）和 **GPT-5.6 Sol**（`59`），并以微弱优势领先于 **Claude Opus 4.8**（`56`）。评论者指出，排名标题可能会掩盖效率差异：有人声称 Kimi K3 在高/最高设置下使用的 Token 数量大约是 **GPT-5.6 Sol 的 2 倍**，这使其有效成本与之相当；而它使用的 Token 数量大约是 **Claude Opus 4.8 的一半**。** 评论认为这是 Kimi 团队的一项重大成就，如果关于 Token/成本的说法成立，那么在竞争上对 **Anthropic** 的威胁可能比对 **OpenAI** 更大。人们对 Kimi K3 的定价如何与 Fable 和 Sol Max 竞争也产生了浓厚兴趣。

    - 一个与技术相关的成本效率点是，据报道 **Kimi K3 使用的 Token 数量约为 Gemini 5.6 Sol xHigh/Max 的 `2倍`**，这使得其端到端成本大致相当，而非明显更便宜。相比之下，评论者指出它使用的 **Token 数量约为 Claude Opus 4.8 的一半**，这将使该结果在性价比上对 **Anthropic** 构成比对 OpenAI 更直接的威胁。
    - 一位评论者链接了 ArtificialAnalysis 的定价/Token 截图，并认为尽管 Kimi K3 总排名第三并击败了 **Claude Opus 4.8**，但它并**没有预想中那么“具有价格效率”**。其技术含义是，除非根据 Token 使用情况和 API 定价进行标准化，否则单纯的基准测试排名可能会夸大其价值。
    - 有人提出了硬件供应链视角：评论者注意到，尽管获取先进 AI 加速器和芯片制造技术的渠道受限，中国 AI 公司仍取得了接近前沿的基准测试结果。其中隐含的技术观点是，Kimi 的结果可能反映了在算力受限的情况下，模型/训练效率表现异常出色。

- **[中国国家主席习近平在世界人工智能大会上发表讲话，重申了对开源的承诺，以促进“开放共赢”](https://www.reddit.com/r/singularity/comments/1uypik5/chinese_president_xi_jinping_speaks_at_world_ai/)** (热度: 2216): **这张 [图片](https://i.redd.it/1b2s142iwpdh1.png) 是总结 **习近平在世界人工智能大会讲话** 的截图，将中国的 AI 政策框架设定为围绕 **open source**、“开放共赢”、全球合作，并避免泛化国家安全限制。从技术角度看，其意义不在于发布了某个模型或 Benchmark，而是一个 AI-governance 信号：中国正将 open-weight/open-source AI 以及培训计划——据报道为发展中国家提供 `5,000` 个 AI 培训/合作机会——定位为其国际 AI 战略的一部分，完整演讲链接在 [YouTube](https://www.youtube.com/watch?v=ApCmqmhE1rg) 上。** 评论者将其与美国/frontier-lab 的论调进行了对比，称该演讲听起来异常务实且较少受恐惧驱动。其他人则认为，中国的 open-weight 模型可以提高缺乏 AI 基础设施的小国的全球基准水平，同时批评欧洲的监管选择可能导致其依赖美国技术。

    - 几位评论者将 **中国发布的 open-weight 模型** 视为提高全球 AI 获取基准的一种方式：没有前沿规模算力、数据或人才的小国仍然可以在本地部署和调整高性能模型，从而减少对美国封闭 API 的依赖，并降低被主要 AI 大国 *“剥夺智能获取途径”* 的风险。
    - 一个反复出现的技术政策观点是，**欧洲的监管姿态以及在前沿/开源领域的乏力反应** 可能导致欧洲开发者依赖美国的封闭模型基础设施，而不是构建或采用自主的 open-weight 替代方案。讨论将此与中国的策略进行了对比，即发布免费/开源模型作为一种 AI 基础设施输出形式，特别是面向全球南方（Global South）。


### 2. Agentic Coding 工作流与验证

  - **[我用 Fable 在大约一周内构建了一个真实比例的宇宙星图（840万颗真实恒星）](https://www.reddit.com/r/ClaudeAI/comments/1uxy5s8/i_built_a_truescale_atlas_of_the_universe_84m/)** (热度: 1560): **一名开发者构建了 [Universe Atlas](https://universeatlas.org/)，这是一个采用 MIT 协议的原始 WebGPU 浏览器星图 ([GitHub](https://github.com/chrisjz/universe))，能够以测量比例渲染 **8.4M Gaia DR3 恒星**、**2.6M SDSS 星系**、行星轨道、实时卫星、人马座 A*（Sgr A*）引力透镜/S星轨道、大气 raymarching 以及日食/卫星/彗星场景。据报道，该项目是在约 1 周内使用 **Claude Code + Fable 5** 完成的：包括 `92` 次合并的 PR、`237` 次 commit、约 `14.5k` 行 TypeScript/WGSL 代码、一个 `90 kB` gzipped 后的零依赖引擎，并通过 **JPL Horizons** 进行 CI 验证（`0.2°` 容差）、物理门控的瓦片生成、确定性 URL 复现以及基于软件 Vulkan 的 WebGPU 像素差分测试。** 顶级的技术反应大多是高层面的：一位评论者请求增加 *n-body simulation*，而另一位则询问作者如何弥合预期视觉效果与 Fable 输出之间的差距，并指出 Fable 在没有自定义/外部资产的情况下通常在图形表现上令人失望。另一位则将该项目视为科学/可视化用例成为新 AI coding 模型强力目标的典型案例。

    - 一位评论者提出了关于 **Fable 5** 的实际内容流水线问题：其图形输出可能受限于可用或生成的资产，需要用户从外部导入或创建自定义资产（例如在 **Blender** 中），以弥合预期可视化效果与 Fable 输出之间的差距。
    - 一位评论者注意到生成的星图场景中似乎除了恒星之外还包含 **黑洞**，这意味着 Fable 在声明的 `8.4M` 真实恒星数据集之外，还生成或推断了额外的天文天体。

- **[letting Claude run unattended for three hours changed how i feel about my own job more than the output did](https://www.reddit.com/r/ClaudeAI/comments/1uy8iht/letting_claude_run_unattended_for_three_hours/)** (Activity: 1645): 该贴描述了使用 [**Claude**](https://www.anthropic.com/claude) 进行一次约 `3 小时` 无人值守代码迁移的过程，包含了预定义的任务、完成定义 (Definition of Done) 以及自检循环；结果显示任务完成了约 `90%`，并需要大约 `1 小时` 的人工清理。技术层面的担忧不在于输出质量，而在于**可审计性 (Auditability) 和问责制 (Accountability)**：事后审阅大量的自主 Diff/日志与实时监督每一步在认知上是不同的，这使得维护逐行追溯的来源感和信心变得更加困难。热门评论将此视为从 IC 风格的执行向工程管理的转变：将模型视为能力不断提升的员工，例如 *“Sonnet 是实习生，Opus 是初级工程师”*，而更具自主性的 Agent 则更像全职员工 (FTE)。另一位评论者认为，职责应当从追踪每一个细节转向验证正确性、性能和安全性产出。

    - 一个反复出现的技术工作流主题是，无人值守/Agentic 编码将工程师的角色从逐行实现转变为**验证与风险管理**：验证功能是否正常运行，检查性能特征，并确保没有引入安全性回退。
    - 几位评论者用人员配备的术语来定义模型能力：**Sonnet** 为“实习生”，**Opus** 为“初级工程师”，而 **Fable** 则更接近拥有 `3–5 年` 经验的全职工程师。这带来的实际影响是，人类成为了 Agentic 系统的管理者：定义策略、选择高价值问题，并将模型限制在其擅长的任务范围内。

  - **[I built an open-source canvas where Claude responds beside your handwritings](https://www.reddit.com/r/ClaudeAI/comments/1uz0ajn/i_built_an_opensource_canvas_where_claude/)** (Activity: 1209): **PenEcho** 是一个面向手写数学/物理工作流的开源本地白板/画布，用户可以在其中绘制方程式/图表，在停顿后，应用会将裁剪后的视觉“图集”及几何数据发送给模型，随后将模型生成的可编辑响应放置在作品旁 ([GitHub](https://github.com/erickong/penecho))。该实现采用了逻辑上为 `20,000 × 20,000` 的画布和稀疏的 `512 × 512` 墨迹瓦片，支持 **Anthropic API / Claude Code CLI** 以及 **OpenAI-compatible APIs / Codex CLI**。作者报告称典型请求包含数千个输入 Token 和少于 `1,000` 个输出 Token，其中 **Claude Opus 4.8 max effort / fb5** 在识别潦草手写、未完成的方程式、图表以及空间推理方面主观表现强劲。热门评论非常积极，称其为罕见的、在技术上令人印象深刻的帖子，并且是一种潜在有用的教室 AI 形式，能够辅助推理而非单纯地“喂”答案。一位评论者询问该结果是需要大量的 Prompt/产品调优，还是基本属于直接的模型集成。

    - 创作者解释说，主要的工程约束在于 **LLM 输入成本**和可编辑性：他们无法将整个画布发送给 Claude，也无法要求其生成完成的渲染图像。相反，系统让模型发出**结构化工具调用 (Structured Tool Calls)**，PenEcho 将其渲染为画布上的可编辑对象，这需要精心设计的工具 Schema。
    - 一个关键的实现挑战是**感兴趣区域 (Region-of-interest) 选择**：决定发送手写画布的哪些部分以及包含多少周围上下文。他们还强调**坐标对齐**非常困难，因为模型必须将它所看到的裁剪图像中的位置映射回更大的画布坐标系。
    - 作者报告了围绕视觉、手写识别和空间推理进行的大量测试和调优，指出当前模型的整体表现超出了预期。他们特别提到，**Claude Opus 4.6** 在这个用例中的表现并不理想。