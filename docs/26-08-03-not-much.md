---
companies:
- alibaba
- deepseek
- databricks
date: '2026-07-27T05:44:39.731046Z'
description: '**Alibaba** 发布了 **Qwen3.8-Max**，这是一款拥有 **2.4 万亿参数**的开放权重模型，重点提升自主编程、长周期任务执行和多模态反馈能力，同时采取了较为激进的定价策略。早期基准测试显示，它在用户偏好和视觉任务方面表现突出，已达到与
  **Claude Opus 4.7** 相当的水平，并展现出较强的目标检测能力。


  不过，这类模型的实际运行成本和资源需求依然很高，尤其是 **Qwen3.8-Max** 和 **Kimi K3** 这样的大型 MoE 模型。这也凸显了更小型开放模型的战略价值，例如即将推出的
  27B 版本。当前，开放权重模型的前沿正越来越多地由中国实验室推动，包括 **Kimi**、**DeepSeek**、**GLM** 和 **MiniMax**，中国团队与美国实验室之间的差距正在缩小。**DeepSeek
  V4 Flash** 则被认为是智能体模型领域一个打破成本与性能平衡的产品。


  社区总结出的关键观点包括：“**中国实验室正在引领开放模型的发展速度**”，以及“**推理服务商的选择实质性地改变了排行榜结果**”。'
id: MjAyNS0x
models:
- qwen3.8-max
- qwen3.8-27b
- kimi-k3
- deepseek-v4-flash
- claude-opus-4.7
people:
- alibaba_qwen
- zhihufrontier
- jaminball
- kimmonismus
- jonathanross321
- _micah_h
- clementdelangue
- tonychenxyz
- yuchenj_uw
- casper_hansen_
- htihle
- skalskip92
title: 今天没发生什么事。
topics:
- multimodality
- model-quantization
- model-performance
- benchmarking
- reinforcement-learning
- model-deployment
- cost-efficiency
- inference-speed
- model-optimization
- agent-models
---

**平静的一天。**

> 2026/7/25—2026/7/27 的 AI 新闻。我们查看了 12 个 subreddit、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有再查看其他 Discord。你可以在 [AINews 网站](https://news.smol.ai/) 搜索过往的所有期刊内容。提醒一下，[AINews 现在已经是 Latent Space 的一个栏目](https://www.latent.space/p/2026)。你可以选择[接收或取消接收](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)不同频率的邮件！




---

# AI Twitter 速览


**Qwen3.8-Max 发布、开放权重转向，以及早期评测表现**

- **Qwen3.8-Max**：Alibaba 发布了 **Qwen3.8-Max**，称这是一个拥有 **2.4 万亿参数**的模型，重点强化了自主编码、长周期执行和多模态反馈闭环，并表示 **Qwen3.8-Max 和 Qwen3.8-27B 的开放权重版本将于下周推出**。根据 [@Alibaba_Qwen](https://x.com/Alibaba_Qwen/status/2084100707423289643) 的信息，其定价相当激进：**输入 2 美元/百万 token、输出 6 美元/百万 token、缓存 token 0.25 美元/百万 token**。这次发布的意义不仅在于模型本身：此前的“Max”版本都是闭源的，因此这意味着 Alibaba 正在从单纯依靠 API 变现，转向通过开放分发来扩大生态影响力，[@ZhihuFrontier](https://x.com/ZhihuFrontier/status/2084230028007764415) 也特别指出了这一点。
- **早期基准测试与表现**：在偏向人类偏好、面向实际产品的任务中，[Frontend Code Arena](https://x.com/arena/status/2084108703729615026) 将 Qwen3.8-Max 排在 **总榜第 4 位，得分 1668**，位于 Claude Opus 5 Max 和 Kimi K3 Max 之后；[Vision Arena](https://x.com/arena/status/2084108711665270942) 则将其排在 **第 2 位，得分 1305**。在 [Vals](https://x.com/ValsAI/status/2084364164655694236) 的榜单中，它以 **66.1 分**位列 **开放权重模型第 2 名**，并在总榜 43 个模型中排名 **第 10**。Vals 指出，它以更低的测试成本达到了与 Claude Opus 4.7 相当的水平，相较 Qwen 3.7 Max 也有大幅提升；他们还报告称，该模型在 **SWE-bench 上达到 87.3%**，在 **Terminal-Bench 2.1 上达到 67.4 分**。不过，Alibaba 自己的报告中存在超时设置差异，相关 caveat 见这篇[讨论串](https://x.com/ValsAI/status/2084364167751065996)。[@skalskip92](https://x.com/skalskip92/status/2084389468761362463) 还发现了一个值得关注的细分优势：在卫星、红外、文档、图表和拥挤场景等任务中，Qwen3.8-Max 作为目标检测 VLM 的表现格外出色。
- **基础设施方面的现实检验**：一些帖子反驳了“开放模型就意味着便宜、易用”这种过于简单的说法。[ @jaminball](https://x.com/jaminball/status/2084264107633729614) 指出，像 **Qwen3.8-Max** 和 **Kimi K3** 这样的前沿开放权重 MoE，在实际部署和运行方面依然负担很重：仅加载 K3 的权重就需要 **超过 1TB** 内存，实际上还需要配备多个加速器的节点，尽管其 token 定价很有吸引力。因此，即将推出的 **27B 开放模型**对于真正的本地部署和私有部署尤其重要，[@kimmonismus](https://x.com/kimmonismus/status/2084209750477029447) 也表达了类似观点。

**中国开放模型的崛起：Kimi、DeepSeek、GLM，以及正在缩小的差距**

- **开放权重模型的前沿格局如今似乎由中国引领**：纵观这期摘要，最突出的整体趋势是：**中国实验室正在引领开放模型的发展速度**。[（@kimmonismus）](https://x.com/kimmonismus/status/2084215318990229972)、[（@JonathanRoss321）](https://x.com/JonathanRoss321/status/2084287904415895795) 和 [（@_micah_h）](https://x.com/_micah_h/status/2084401434746036403) 发布的内容都指向同一现象：如今，Kimi、Qwen、DeepSeek、GLM 和 MiniMax 已经构成开放模型前沿的重要版图；而美国实验室主要是在少数闭源产品上保持领先。[（@ClementDelangue）](https://x.com/ClementDelangue/status/2084268924066009483) 及相关报道进一步放大了这一更广泛的判断：China 正在开放权重模型领域占据主导地位。
- **Kimi K3 与推理环境的敏感性**：K3 在下游应用和基础设施测试中继续取得亮眼成绩。[RSIBench-Data](https://x.com/FanqingMengAI/status/2084100049630601673) 报告称，**Kimi K3 + Kimi Code** 在六项自动化研究基准上的**加权得分为 27.317%**，其中包括 **50% 的 SWE-bench Verified** 和 **17% 的 SWE-bench Pro**。不过，[（@tonychenxyz）](https://x.com/tonychenxyz/status/2084242262188601650) 指出了一项关键的工程注意事项：**推理服务商会显著影响排行榜结果**。其中一家服务商出现了退化的循环行为，而 Modal 的 endpoint 则在 CEO-Bench 上取得了第一名。在服务性能方面，[（@Yuchenj_UW）](https://x.com/Yuchenj_UW/status/2084324515719651559) 表示，Databricks 目前为 K3 提供 **239 tok/s** 的速度和顶尖延迟表现；与此同时，[（@casper_hansen_）](https://x.com/casper_hansen_/status/2084307303163982179) 提到，在单台 B300 节点、batch size 为 32 的情况下，解码吞吐量可达 **947 tok/s**。
- **DeepSeek V4 Flash 成为性价比颠覆者**：DeepSeek 最新的 Flash checkpoint 成为了当天最引人注目的**成本调整后 Agent 模型**案例。[（@htihle）](https://x.com/htihle/status/2084246773413957957) 报告称，Flash-0731 在 WeirdML 的 high/max 设置下分别取得 **57.1% / 63.0%** 的成绩，并认为该测试环境可能低估了模型的真实能力。[Vals](https://x.com/ValsAI/status/2084451706650443916) 称，**DeepSeek V4 Flash (0731)** 是 Vals Index 中得分超过 60 分的**最低价模型**，在达到这一门槛的模型中，价格比下一个最佳模型低 **35 倍**；其大部分优势来自编码和 Agent 任务。[Together AI](https://x.com/togethercompute/status/2084438456890019970) 随即将其定位为适合长期运行 Agent 的生产级 endpoint。
- **GLM 以及接下来的发展**：多条帖子暗示 **GLM-5.3 即将发布**，其中包括 [（@AiBattle_）](https://x.com/AiBattle_/status/2084214160418627604) 和 [（@arena）](https://x.com/arena/status/2084384756171669826)。后者还提醒大家，**GLM-5.2 Max** 已经在 Frontend Code Arena 中取得**综合排名第 2、开放模型排名第 1** 的成绩。

**Agent harness、长时程系统，以及为什么单靠模型质量已经不够了**

- **Harness 已经成为控制平面**：技术圈的讨论反复指出，如今理解长程任务性能，最好采用 **模型 × Harness** 的视角，而不是只看模型本身。[@ZhihuFrontier](https://x.com/ZhihuFrontier/status/2084201996954022228) 发布的一份详细调研总结认为，长程能力源于基础模型与运行时系统的协同演进，而运行时系统负责记忆、规划、工具调用、验证、编排和恢复等环节。这与 [@omarsar0](https://x.com/omarsar0/status/2084367708439949343) 提到的一篇论文相呼应：该论文根据模型、用户、Harness、工具、记忆和环境之间的交互边，将 **41 种 Agent 失败模式**进行了分类，而不是简单归咎于某一个组件。
- **生产级运行时正在快速落地**：[Cloudflare](https://x.com/Cloudflare/status/2084264282405974034) 推出了 **@cloudflare/computer**，这是一套 Agent 运行时，可以在轻量级 isolate 和完整 Linux 容器之间动态切换。[Cursor](https://x.com/cursor_ai/status/2084317547608911986) 表示，其云端 Agent 的 token 使用效率提升了 **20%–30%**，在 computer-use 任务上的效率提升了 **80%**；随后又宣布[上线](https://x.com/cursor_ai/status/2084376701539405904)了 Gmail、Drive、Calendar、Docs 和 Sheets 的原生 **Google Workspace 插件**。[LangChain](https://x.com/hwchase17/status/2084449633955115352) 表示，**Managed Deep Agents** 将进入公开 beta，并内置 eval、记忆、OAuth、channels 和沙箱能力。
- **开放模型与 Harness 的协同优化开始变得重要**：[Cline](https://x.com/cline/status/2084359007029141528) 分享了当天颇具洞察力的一条实践观察：许多开放模型似乎经过 **RL 训练，会额外消耗 token 来验证工作结果**——例如重新运行测试、检查构建结果、重新阅读 diff 等。Cline 则有意让这些模型“按照训练时形成的方式工作”，并称仅通过调整 Harness 就获得了约 **20% 的提升**。[ @Teknium](https://x.com/Teknium/status/2084344999513383195) 围绕 **Hermes Agent** 发布的内容中也体现了类似趋势：该项目新增了语音激活、插件/API 扩展、A2A 协议支持、出站 webhook、研究技能，以及显著的 token 使用效率提升。
- **在可行的地方，记忆和解析正在减少对 LLM 的依赖**：[@dair_ai](https://x.com/dair_ai/status/2084370729332797724) 介绍了 **Zero-Mem**。它在维护记忆时不调用 LLM，只在生成最终答案时调用一次 LLM；在预算相同的情况下，相比速度最快的 baseline，记忆操作成本降低了 **57.6%**。[LlamaIndex](https://x.com/llama_index/status/2084265189772317162) 也在 **LiteParse** 中加入了更丰富的结构化 PDF 提取能力，可以直接提供字段、复选框、批注、图形和页面复杂度等信号，而无需为每一页都调用 vision model。

**自动化研究、post-training 和 benchmark 设计正在成为更严谨的工程学科**

- **自动化 post-training 正在带来实际收益**：[@intology](https://x.com/intology/status/2084319121332965804) 声称，其 **Locus** 系统在 **PostTrainBench** 上达到了 **SOTA**，并能对 **Qwen3 1.7B-Base** 变体进行 post-training；在扩大计算预算后，这些模型的表现超过了官方人工调优的 **Qwen3 1.7B Instruct**。同一条帖子还称，Locus 已经泛化到真实的 Kaggle 竞赛中，经过 16 天后取得了平均排名第 **4** 的成绩。另一方面，[@mervenoyann](https://x.com/mervenoyann/status/2084335423560495547) 介绍了面向 coding Agent RL pipeline 的公开工具，这些工具基于沙箱任务、TRL 和 verifier 构建。
- **研究自动化 benchmark 正在暴露 Harness 的影响**：简短但信息量很高的 [RSIBench-Data 结果](https://x.com/FanqingMengAI/status/2084100049630601673)，以及 [@gneubig](https://x.com/gneubig/status/2084342402295210275) 的回应都表明，超长程自动化研究任务正在越来越多地衡量**专用研究 Harness**，而不只是模型智能。这一点也出现在 [@Shahules786](https://x.com/Shahules786/status/2084319792815829148) 的批评中：benchmark 应该开源**完整轨迹**，因为仅看分数，无法判断失败究竟源于模型能力不足、verifier 脆弱，还是任务定义不充分。
- **噪声、验证和留出数据仍然是关键问题**：[@ddkang](https://x.com/ddkang/status/2084335070668616148) 反驳了“使用 **100% 噪声数据进行 RLVR** 可以达到与干净数据训练相同效果”的观点，并报告称，在更严格的噪声数据构造方式下，MATH 准确率低了 **9% 以上**。[@ArmenAgha](https://x.com/ArmenAgha/status/2084349093447676409) 分享了一个规模更小但很有启发性的结果：优化代理目标虽然提升了所选指标中的 velocity MSE，却让模型在留出数据上的实际 rollout 推理表现变差。这再次提醒我们，如果评估不够稳健，许多“自我改进”的 headline 经不起检验。

**多模态与视频系统：MiniMax H3、world model 和本地生成**

- **MiniMax H3 是此次最亮眼的多模态/视频模型发布**：社区反馈表明，**MiniMax H3** 是开源权重视频生成领域的一次重大进步。[@arena](https://x.com/arena/status/2084408459991421319) 的排名显示，在 Video Arena 的文生视频和图生视频评测中，H3 均位列**开源模型第 1**，领先下一名开源模型 **280 分**；在图生视频评测中，它实际上与整体排名**第 1** 的模型并列。[@MiniMax_AI](https://x.com/MiniMax_AI/status/2084410437618352386) 表示，按照 Arena 和 Artificial Analysis 的基准测试结果，H3 目前已经是**开源视频生成领域的 SOTA 模型**。
- **H3 的技术意义**：多篇帖子都强调，H3 不只是又一个 T2V 模型，而是一个**通用多模态生成模型**：能够在同一上下文中处理文本、图像、视频和音频，同时也提供了可实际使用的本地部署路径。[@kimmonismus](https://x.com/kimmonismus/status/2084229681012711598) 对关键限制进行了清晰总结：H3 采用开放权重，本地视频生成潜力很强，但**并非完整开源的技术栈**，因为上下文编排、2K 重生成和稀疏注意力仍运行在服务端，或受到其他形式的限制。[@ComfyUI](https://x.com/ComfyUI/status/2084387277254644162)、[@victormustar](https://x.com/victormustar/status/2084322394479464781) 和 [@MiniMax_AI](https://x.com/MiniMax_AI/status/2084387967981011326) 都介绍了实际的本地工作流，包括在 **RTX 5090 级别**显卡上的使用方式。
- **许可证问题仍然比较复杂**：围绕 H3 地区限制的讨论一度引发混淆。[@ostrisai](https://x.com/ostrisai/status/2084110556374659476) 最初将许可证解读为禁止在**美国、欧盟、英国和韩国**使用，这一担忧随后扩散开来。之后，[@VictorSuOrtiz](https://x.com/VictorSuOrtiz/status/2084410948358705273) 澄清称，这些地区需要经过**正式授权流程**，并不是完全无法获得许可。对于评估部署可行性的团队而言，这是一个非常重要的区别。
- **世界模型和多模态仿真仍处于发展初期**：一些互动量较低、但技术含量较高的帖子，将关注点指向了**无监督潜变量模拟器**以及类似 world model 的系统，显示这一方向正在逐渐升温，其中包括 [@soniajoseph_](https://x.com/soniajoseph_/status/2084157222892806197) 和 [@taiuti](https://x.com/taiuti/status/2084286971774664922) 的相关讨论。

**值得关注的推理系统、编译器、实时语音及其他基础设施**

- **OpenAI 正在重新设计实时语音技术栈**：[@OpenAI](https://x.com/OpenAI/status/2084378415818579975) 介绍了一种新的 **GPT-Live** 架构，通过将**专用快速音频通道**与较慢的异步推理/工具调用通道分离，实现全双工对话——也就是在说话的同时持续聆听。该架构还将会话启动过程从**六次网络往返减少到一次**，并在链接的工程文章及 [@juberti](https://x.com/juberti/status/2084380194463158610) 后续讨论中介绍了面向长上下文语音会话的异步压缩方案。
- **编译器正在接管手工优化的推理内核**：[@vikhyatk](https://x.com/vikhyatk/status/2084409834523476073) 宣布推出 **Photon 2.0**。这款编译器可以将 **Moondream、Qwen 3.5 和 Gemma 4** 等模型转换为**大型内核（megakernel）**，把完整的前向传播表示为一个单独的 GPU 程序。相关讨论还介绍了一个用于描述数据流的 tracer DSL，以及一个 CPU 成本模型，用于在编译前筛除不合适的调度候选方案。这与 [@waterloo_intern](https://x.com/waterloo_intern/status/2084426439034540297) 的更广泛讨论相呼应：传统的手工 GPU 内核优化正在逐步实现自动化，并成为标准化、低成本的能力。
- **Tokenization 和服务端瓶颈正成为一等问题**：[@omarsar0](https://x.com/omarsar0/status/2084414040760275278) 介绍了 **TokTier**，这是一种有状态的 Tokenization 服务，可以在 Agent 会话中复用并修复已经 Tokenize 的前缀。据报告，在 vLLM 下它可将 **TTFT 降低 16–34%**；在增量修复场景中，相比标准 Hugging Face Tokenization，速度最高可提升 **437 倍**。当 Agent 的对话记录变长、缓存命中率升高后，这类“非模型”瓶颈正是影响性能的关键因素。
- **规模较小但值得注意的工具**：[Jina AI](https://x.com/JinaAI_/status/2084288559435903485) 发布了 **jina-reranker-v3.5**，这是一款 **0.6B 参数的 listwise reranker**，据称在 **BEIR 上达到 63.20 nDCG@10**，并以大约少 **7 倍参数量**的规模击败了 **Qwen3-Reranker-4B**；[DSPy 3.3.0](https://x.com/isaacbmiller1/status/2084410370282631534) 则通过 **dspy.Flex** 增加了代码和 Prompt 优化功能，并借助 **ReActV2** 改进了工具调用，同时提供了与具体服务商无关的 LM 接口。

**热门推文（按互动量排序）**

- **Qwen3.8-Max 发布**：Alibaba 宣布将于下周推出一款拥有 **2.4T** 参数、开放权重的旗舰模型，这是本次汇总中最重大的技术发布 [@Alibaba_Qwen](https://x.com/Alibaba_Qwen/status/2084100707423289643)。
- **OpenAI 数学成果**：OpenAI 表示，其下一代主要模型的内部版本，在数学和 TCS 领域针对长期悬而未决的开放问题取得了 **10 项新成果**，消耗的 GPT-5.6 Sol 等价 token 成本约为 **2,000 美元** [@OpenAI](https://x.com/OpenAI/status/2084352161404920316)。
- **GPT-Live 架构**：OpenAI 的新实时语音技术栈支持在说话的同时持续聆听，并异步执行工具调用和推理任务 [@OpenAI](https://x.com/OpenAI/status/2084378415818579975)。
- **源代码抽象之争**：Elon Musk 认为，**源代码即将变得像汇编语言一样**，未来 AI 可能会直接把人类意图编译成二进制文件 [@elonmusk](https://x.com/elonmusk/status/2084304083851034949)。
- **Cursor 工作区集成**：Cursor 已为 Agent 接入 **Google Workspace** 应用，让编程 Agent 进一步走向通用办公自动化 [@cursor_ai](https://x.com/cursor_ai/status/2084376701539405904)。


---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Qwen3.8-Max 与 27B 开放权重模型发布

  - **[Qwen3.8-Max 可与 Kimi K3 和 DeepSeek V4 Flash 比肩](https://www.reddit.com/r/LocalLLaMA/comments/1vellf2/qwen38max_matches_kimi_k3_and_deepseek_v4_flash/)**（热度：541）：**图片展示的是 **BenchmarkList** 上的 **Qwen3.8-Max** 页面。页面显示，这是一款开放权重的 Qwen 模型，**Experimental ECI 为 `143.33`**，总排名 **第 12**，在开放权重模型中排名 **第 3**；图表中它的位置接近 **Kimi K3** 和 **DeepSeek V4 Flash**，与帖子所说的“在各类基准测试中具备竞争力”相符。帖子还提到，Qwen3.8-Max 是一款拥有 **2.4T 参数**的模型，预计“下周”开放权重，定价为 **每百万输入 token 2 美元**、**每百万输出 token 6 美元**，**隐式缓存每百万 token 0.25 美元**，并称其在编程和软件工程任务上的表现尤其突出。图片：[BenchmarkList Qwen3.8-Max 对比图](https://i.redd.it/14mqdzhzb7hh1.png)。**评论者质疑，这种比较究竟关注的是**能力，还是价格/性能比**，并指出，如果 **DeepSeek-V4-Flash 约有 284B 参数**，却能接近 Qwen3.8-Max 和 Kimi K3，那么它以小约 10 倍的规模达到这一水平，反而更令人印象深刻。另一个讨论重点则是即将推出的 **Qwen3.8-27B** 以及更小的 dense 模型，因为它们可以运行在消费级硬件或两张 24GB GPU 上；相比之下，大家对又一个 300B+ 或万亿级模型的兴趣没那么高。

    - 一些评论者质疑，**DeepSeek-V4-Flash（`284B`）** 是否真的能称得上与 **Kimi-K3（`2.8T`）** 和 **Qwen3.8-Max（`2.4T`）**“比肩”。他们指出，DeepSeek-V4-Flash 的规模大约小 `10x`，整体能力很可能并不在同一水平。讨论中还区分了两种可能的含义：是**价格/效率**相当，还是**模型质量**相当。对于一个低于 `300B` 的模型能够等同于当前云端 SOTA 模型，评论者普遍持怀疑态度。
    - 大家尤其关心 **Qwen 3.8 27B** 是否比上一代有实质性提升，因为在许多人看来，能够在一张约 800 美元的 GPU 上运行的小模型，比 `300B+` 模型取得一些基准测试上的边际提升更有实际价值。一位评论者特别将本地可用性与“TerminalBench 再提高 5 分”进行了对比，并推测，一款针对 `2x24GB` GPU 优化的 **45–55B dense 模型**，可能会成为一个很有吸引力的折中方案。
    - 评论者不确定相关基准测试究竟测量了什么，尤其是**编程质量**。一位评论者询问，**DeepSeek-V4-Flash** 的代码生成质量是否能与 **Qwen3.8-Max** 或其他参数量 **>2T** 的模型相当；另一位则表示，尽管自己喜欢 **V4-Flash-0731**，但并不认为它已经达到 **GLM-5.2**、**Kimi-K3** 或预期中的 **Qwen3.8-Max** 水平。

  - **[有人真的读过 Qwen 3.8-Max 的博客吗？](https://www.reddit.com/r/LocalLLM/comments/1ve33xi/did_anyone_actually_read_the_qwen_38max_blog/)**（热度：515）：**帖子介绍了 [Qwen 博客](https://qwen.ai/blog?id=qwen3.8)中的 **Qwen 3.8-Max**，称其旗舰模型拥有 `2.4T` 参数，同时还将推出一款 `27B` 开放权重模型，重点面向 Agent 工程工作负载，而不只是提升聊天能力。帖子列举的能力包括：从空仓库开始自主进行 `10+` 天的软件开发；原生支持视觉反馈闭环，用于反复执行和纠错；以及使用 `Iverilog`、`Yosys` 和 `OpenROAD`，经过 `500+` 轮迭代对芯片设计进行闭环优化。据称，该模型将一个加密加速器的门数从 `8,298` 降至 `678`，同时实现了时序收敛。**评论者的看法不一：有人认为这种表述听起来像是*“基于术语堆砌的胡言乱语”*，尽管他们也期待 Qwen3.8；其他人则主要表达了笼统的兴奋，并未提出技术层面的批评。

- 一位评论者认为，最具技术意义、也最值得复现的 Qwen 3.8-Max 说法，应该是**公开发布的 10 天自主运行轨迹**。原因在于，生成长时间运行的 Agent 并不难，但评估这些运行结果却很困难；关键问题是：运行期间由什么机制验证了工作成果，以及最终依据什么终止条件停止运行。他们引用了自己的基准测试 [Muvon/octobench](https://github.com/Muvon/octobench)：使用每个项目专门保留的测试集，对 **4 个 coding agent** 在 **25 个真实、已合并的 PR** 上进行了测试。结果显示，脚手架和测试 harness 的影响占据主导地位：在**相同模型、相同 endpoint** 的情况下，两个 harness 分别完成了 `24` 个和 `19` 个任务。他们据此认为，厂商将模型和 Agent 脚手架打包在一起进行演示，无法区分性能究竟来自基础模型，还是来自外围的编排机制。

  - **[Daniel Han of Unsloth validates Qwen3.8-27B will run only 17GB VRAM](https://www.reddit.com/r/LocalLLaMA/comments/1ve4uoe/daniel_han_of_unsloth_validates_qwen3827b_will/)**（热度：2096）：**图片是 Daniel Han / Unsloth AI 发布在 **LinkedIn** 上的一则帖子，重点介绍了 **Qwen3.8-27B** 和 **Qwen3.8-Max**，并展示了 coding、Agent、reasoning、vision 和 web 任务的基准图表；Reddit 标题和正文所强调的核心技术说法是：**Qwen3.8-27B 可能只需约 `17GB` RAM/VRAM 就能在本地运行**（[图片](https://i.redd.it/kabmtuygn3hh1.jpeg)）。评论者推测，这很可能意味着该版本采用了 **QAT/quantization-aware training**，类似于 **DeepSeek V4 Flash**，而不是以全精度方式部署 27B 模型。**讨论的重点，一方面是大家对一款性能可能很强的小型开放权重 Qwen 模型感到兴奋，另一方面则是对 `17GB` 这一容量刚好高于常见 `16GB` VRAM GPU 上限感到无奈；一位评论者称这是“几个月来最令人兴奋的消息”，另一位则站在 16GB 用户的角度开玩笑。

    - 评论者推断，`17GB VRAM` 这一说法很可能指的是量化版本或采用 QAT 训练的版本，而不是全精度模型；其中一人将其与 **DeepSeek V4 Flash** 风格的量化感知训练进行了比较。也就是说，Qwen3.8-27B 的这个数字反映的是部署或量化后的结果，并不是原始 `27B` dense 模型的显存占用。

    - 有人将其与 **Qwen 3.6 27B** 做了技术对比。一位评论者表示，该模型在更激进的量化下已经可以运行在大约 `~12GB` 的显存中。另一位用户预计，同等参数规模的 `q8` 版本应能控制在 `37GB` 以下，因此可以在 `48GB` VRAM 的配置上运行；而 `16GB` 显卡即使面对宣传中的 `17GB` 配置，可能仍会略低于实际所需的容量。

  - **[More Qwen 3.8 sizes coming](https://www.reddit.com/r/LocalLLaMA/comments/1vevsv9/more_qwen_38_sizes_coming/)**（热度：372）：**[图片](https://i.redd.it/zodlaejqc9hh1.jpeg) 是一张 X 回复截图，Qwen 的 Shuai Bai 表示，在用户询问继 `27B` 发布后是否可能推出 **Qwen 3.8 `35 A3B`** 版本后，团队“仍在继续完善更多参数规模和架构的产品线”。从技术角度看，这并不是基准测试结果或发布说明，但它释放出一个信号：Qwen 3.8 系列可能会在当前 `27B` 模型之外继续扩展，推出更多参数规模和/或架构变体。**评论大多是兴奋和猜测，不少用户希望 Qwen 推出更大的 **`122B`** 版本，也有人认为 Qwen“本来就应该先发布这个”。

    - 评论者主要在猜测他们希望看到的 **Qwen 3.8** 参数规模，多人提出希望推出 `122B` 模型，也有人明确要求推出 `60B` 的 **dense** 版本。讨论中没有涉及基准测试、架构细节、发布日期或性能数据。


### 2. DeepSeek V4 Flash 基准测试与运行时支持

  - **[DeepSeek V4 Flash just drew a pretty brutal "kill line" on this chart](https://www.reddit.com/r/LocalLLM/comments/1vdmmsp/deepseek_v4_flash_just_drew_a_pretty_brutal_kill/)**（热度：944）：**[图片](https://i.redd.it/wm0455k6ozgh1.png) 是 **Artificial Analysis Intelligence Index v4.1** 的散点图，展示模型质量与估算的 `cost per weighted task` 之间的关系，横轴采用对数刻度；图中突出显示了 **DeepSeek V4 Flash 0731**，其成本约为 `$0.03/task`，指数得分约为 `50`，看起来相较于许多价格更高的模型，形成了一条新的低成本 Pareto 前沿。帖子认为，这次更新意义重大，因为此前 **DeepSeek V4 Flash** 的得分约为 `40`，成本几乎相同；但同时也指出，该基准测试面向**英文、纯文本和综合任务**，未必能够准确预测 coding、长上下文或生产工作负载的具体成本。**评论者对“kill line”这一说法提出了质疑，指出 Pareto 前沿图并不意味着所有被支配的模型都真的失去了价值，因为实际部署还会受到价格和指数得分之外的其他限制。一条评论还指出，若严格按照这种解读，这张图实际上只会“淘汰”少数模型。

- 有评论者质疑图表中 **“每项任务的成本”** 的计算方法：它是否只统计成功完成的任务，是否对输出长度或 token 数量做了标准化，以及测量所依据的 benchmark 或任务集是什么。他们还指出，API 成本对比未必能直接反映本地推理的实际情况：对于许多用户来说，在本地运行 **Qwen 3.6 27B** 可能比运行 **DeepSeek Flash** 更容易，具体取决于手头的硬件条件。
- 另一条技术性评论认为，位于 **Pareto 前沿** 并不意味着能字面意义上“淘汰”所有位于其下方的模型，因为部署选择还会受到价格和性能之外的因素影响，例如延迟、硬件、上下文长度、质量波动、可用性以及任务匹配度。评论者还指出，如果严格按照图表中的 Pareto 逻辑，大多数现有数据点其实早已被其他点支配，而新的 DeepSeek 数据点可能只会直接支配包括两个 **Luna** 配置在内的少数几个点。

  - **[DeepSeek-V4-Flash-0731：在 Chess Benchmark 上超越 Fable-5、Sol 和 Kimi-K3](https://www.reddit.com/r/LocalLLaMA/comments/1vdq8en/deepseekv4flash0731_surpasses_fable5_sol_kimik3/)**（活跃度：663）：**这张图片是一份 AI 国际象棋对弈技术 benchmark 排行榜：**deepseek-v4-flash-0731** 标记为“NEW”，以 `80%` 胜率、`8%` 和棋率、`12%` 负率、**Elo `1538`**、**`83.2%` 准确率**和 **`84%` 胜率**排名第一，略高于 **gpt-5**，并领先于 **o3**、**gpt-5.6-sol**、**kimi-k3** 和 **claude-fable-5**。引用的来源是 [AI Chess Leaderboard](https://dubesor.de/chess/chess-leaderboard)，帖子中的图片可在[这里](https://i.redd.it/vvoei0u5d0hh1.png)查看。**评论者质疑了该 benchmark 的可靠性，指出其中存在一些反常排名，例如 **gpt-3.5-turbo-instruct** 排在 **gpt-5.6-terra** 之前，并认为旧模型的处理方式可能存在问题。也有人表示，当前一代模型的排名大体上与 Google 的 Kaggle Game Arena 结果相似，但同时注意到 Gemini 的实力出人意料，以及模型能力似乎会随时间出现倒退。

    - 一些评论者质疑该 benchmark 的有效性，因为排名中出现了 **`gpt-3.5-turbo-instruct` 击败 `gpt-5.6-terra`** 等异常情况，这可能说明旧模型的评测或标准化方式存在问题。另一位评论者指出，尽管其中有些数据看起来可疑，但当前一代模型的排名顺序似乎与 **Google 的 Kaggle Game Arena** 结果较为接近。
    - 引用的来源是 **[AI Chess Leaderboard](https://dubesor.de/chess/chess-leaderboard)**，它评估的是模型的下棋表现，而不是标准的语言 benchmark。评论者特别提到，**Gemini** 系列模型在这类国际象棋评测中表现异常出色，同时也观察到不同版本模型之间似乎存在能力倒退。
    - 一种技术观点认为，如果将国际象棋作为模型实际对弈的任务——例如与 Stockfish 或其他 LLM 对战——它会是一个很强的推理 benchmark。原因在于 **[Shannon number](https://en.wikipedia.org/wiki/Shannon_number)** 所描述的庞大搜索空间，使模型不可能完整记忆整个 benchmark。不过，评论者也提醒，如果使用公开的国际象棋谜题，就容易受到数据污染或“benchmaxxing”影响，因为模型可能已经记住了这些题目的棋局和解法。

  - **[llama.cpp 刚刚加入对 DeepSeek V4 Flash 的 MTP / DSpark 支持](https://www.reddit.com/r/LocalLLaMA/comments/1vdhgq9/llamacpp_just_added_mtp_dspark_support_for/)**（活跃度：757）：**`llama.cpp` 已合并 [PR #25784](https://github.com/ggml-org/llama.cpp/pull/25784)，加入了对 DeepSeekV4 的 **MTP speculative decoding** 和 **DSpark head** 支持，共涉及 `14` 个文件，新增代码约 `1.5k` 行。DGX Spark 的 benchmark 显示，使用 MTP 和 `--spec-draft-n-max 2` 后，吞吐量大约提升 **~50%**：基线约为 `16.4–16.5 tok/s`，提升后约为 `25–28 tok/s`，draft 接受率约为 `0.61–0.76`。不过，评论者指出，目前的 GGUF 可能还没有包含 drafter，而且据报道，DeepSeek `20260731/0731` checkpoint 目前只提供 **DSpark**，不提供 MTP。例如可以使用 [am17an/DeepseekV4-Flash-20260731-DSpark](https://huggingface.co/am17an/DeepseekV4-Flash-20260731-DSpark/)。**评论者普遍对贡献者 **am17an** 表示感谢；主要的技术注意事项是，不要默认最新 DeepSeek 发布版本支持 MTP，在兼容的 drafter GGUF 出现之前，应使用 DSpark 专用转换版本。

    - **am17an** 澄清说，最新的 `0731` 版 DeepSeek 模型并没有随附 **MTP**，因此用户应改用 **DSpark**；他还提供了一个兼容版本：[DeepseekV4-Flash-20260731-DSpark](https://huggingface.co/am17an/DeepseekV4-Flash-20260731-DSpark/)。另一位评论者指出，目前的 **GGUF** 可能并未包含 drafter，这意味着要完整支持 MTP 风格的推测解码，可能仍需等待更新后的模型文件。
    - 早期本地测试显示，将 [ddh0/DeepSeek-V4-Flash-GGUF](https://huggingface.co/ddh0/DeepSeek-V4-Flash-GGUF) 作为 MTP 草稿模型后，生成速度有明显提升：在空上下文下，吞吐量从 **`35 tok/s` 提升到 `50 tok/s`**。据报告，提示词处理速度没有变化，但可用上下文容量从 **`200k` 降至 `139k`**，这表明启用草稿路径会带来内存与上下文容量之间的权衡。

  - **[在 5.3GB 内存上运行 DeepSeek-V4-Flash 284B](https://www.reddit.com/r/LocalLLaMA/comments/1vdbix4/deepseekv4flash_284b_on_53gb_of_memory/)**（热度：425）：**一个新的、基于 MLX 的本地推理引擎 [**Mference**](https://github.com/NeelM0906/Mference)，扩展了 [TurboFieldfare](https://github.com/drumih/turbo-fieldfare) 的 MoE 权重流式加载思路：将共享核心和 KV cache 常驻内存，只从 SSD 流式读取被选中的专家。该项目公布了在 Apple M5 Pro 上的测试结果：**Gemma 4 26B-A4B** 占用约 `2 GB` 内存，速度为 `31–35 tok/s`；**Qwen 3.6 35B-A3B** 占用约 `1.45 GB`，速度为 `19–23 tok/s`；**DeepSeek-V4-Flash 284B-A13B** 占用约 `5.3–6.8 GB` 内存，在使用 2-bit 动态量化时占用约 `91 GB` 磁盘空间，最高速度可达 `4.8 tok/s`。由于专家权重读取目前是串行进行的，解码过程约有 `53%` 受 I/O 限制。项目还提供原生 Mac 聊天应用、兼容 OpenAI API 的服务器，以及本地文档附件功能，并计划支持更多模型系列、实现 I/O 与计算的重叠，以及将上下文长度扩展到 `4K` 以上。**评论者认为，大型 MoE 模型结合 SSD 权重流式加载，是实现本地推理的一条很有前景的路线；不过，他们也希望加入 MTP/推测解码一类的加速方式，并扩大对 Windows/Linux 的支持，可能还应考虑使用 GGUF 而不是 MLX。对于仅有 `8–12 GB` GPU 显存和 `16 GB` 内存的低显存设备来说，这一点尤其重要。

    - 多位评论者重点讨论了可移植性和推理后端：有人询问是否使用 **MTP** 来提升生成速度，以及项目能否改用 **GGUF** 而非 **MLX**，从而支持 **Windows/Linux**。这意味着当前实现可能仍主要面向 Apple/MLX，尚未具备广泛的跨平台能力。
    - 一个技术讨论重点是：**大型 MoE 模型 + 权重流式加载**，或许能让本地推理在常驻内存极低的情况下变得可行；有评论者将这一方案与 **Colibri/TurboField** 相比较。也有人询问实际可行的本地硬件配置，例如能否在 `8–12 GB VRAM + 16 GB RAM` 的系统上运行，或者 Q4 量化版本能否控制在 `48 GB RAM` 以内。
    - 有评论者提到，据称 **DeepSeek-V4-Flash** 在“两天前”获得了一次更新，智能水平有明显提升；他认为该模型目前的能力大致介于 **Claude Sonnet 和 Opus** 之间。这可能会影响后续应当测试或支持哪个模型版本。


### 3. 中国开放权重实验室的发布与策略

  - **[大家常常把中国实验室混为一谈，但它们其实押注了四条截然不同的路线。我就在其中一家工作。](https://www.reddit.com/r/LocalLLaMA/comments/1veipya/the_chinese_labs_everyone_lumps_together_are/)**（热度：771）：**配图是一张背景信息图——[“中国开源 AI 实验室：并非铁板一块”](https://i.redd.it/rlclj3bxu6hh1.png)，用来说明帖子的核心观点：中国的 AI 实验室，例如 **Ant Ling**、**Alibaba Qwen**、**DeepSeek**、**Moonshot/Kimi**、**Zhipu/GLM**、**MiniMax** 和 **StepFun**，正在采取各自不同的战略，并不是一个统一的类别。作者自称在 **Ant Ling** 从事开发，并对比了几种路线：**Qwen** 注重优先构建分发渠道和生态，**DeepSeek** 注重架构、论文与权重，**Ant** 则聚焦于降低服务成本。作者以 **Ling-3.0-flash** 为例：该模型总参数量为 `124B`，每个 token 实际激活约 `5.1B` 参数，采用 **KDA + MLA 混合注意力**，上下文长度达到 `262k`，目标是以低成本支持长时间运行的 Agent 循环，而不是单纯追求排行榜成绩。**评论者围绕实验室归属在实际应用中的重要性展开了讨论：有人认为，Qwen/DeepSeek/GLM 发布的开放权重已经显著推动了本地推理的发展；也有人表示，他们更关心开放程度、成本和审查行为，而不是公司的战略。有人提出了一个技术层面的挑战：当 **DeepSeek** 也在持续推出低成本、高基准成绩的 “flash” 模型时，Ant 是否还能凭借低成本的长时程服务形成差异化优势。**

- 有评论者将 **DeepSeek v4 Flash** 视为与 **Ant** 所设想的“低成本长时程任务”策略直接竞争的产品，认为 DeepSeek 已经进入相近的成本与性能区间，并且“即使成本更低，在基准测试中也能跟上最优秀的模型”。由此引出的技术问题是：如果 DeepSeek 能以更低的推理成本提供相近的基准性能，Ant 应如何实现差异化？

- 多位评论者强调，**中国的开放权重模型**——尤其是 **Qwen、DeepSeek 和 GLM**——已经显著推动了本地推理生态的发展，特别是在消费级硬件上。他们认为，相比专有的“封闭花园”模式，开放权重能更快赢得开发者的关注，也让本地部署 LLM 的发展水平比原本预期“领先了好几个数量级”。

- 有评论者提出，可以少看实验室的品牌，多从以下几个方面评估模型：模型是**开放的还是专有的**、**推理成本**变化有多快，以及预期的**对齐/审查行为**。该评论者对比了美式的安全拒答、中国式的政治敏感内容拒答，以及 **Mistral** 相对较少审查的表现，同时指出 **Qwen** 似乎采取了覆盖多种模型细分场景的策略。

  - **[MiniMax-H3 now on huggingface](https://www.reddit.com/r/LocalLLaMA/comments/1ve1mvh/minimaxh3_now_on_huggingface/)**（活跃度：759）：****MiniMax-H3** 已在 Hugging Face 发布。这是一套面向通用场景的**全模态生成系统**，支持统一理解和生成文本、图像、视频与音频，包括以最高 `2K` 分辨率、最长 `15 s` 时长生成带原生立体声音频的视频。帖子称，H3 在预训练阶段采用了面向任务泛化的设计，因此能够在混合场景中完成复杂的多模态指令；一位评论者表示，他在 **RTX 5090** 上运行了该模型，称其能够处理参考图像/视频，以及非语音音频事件、空间化声音和动作，并且对提示词的遵循程度异常出色。** 早期用户反馈非常积极，但目前仍属于个别用户体验，有人称其“完全不设限”，并认为它可能成为继 Wan 2.2 之后下一个长期使用的基线模型。另有评论者指出，该模型的许可证比较特殊，可能存在问题，但提供的评论中没有具体说明许可证条款。

    - 一位用户报告称，他在 **RTX 5090** 上测试了 **MiniMax-H3**，认为它对提示词的遵循程度异常出色，且具备广泛的多模态生成能力：“不只是音频……任何声音、任何位置、任何动作”都能处理，也支持以参考视频或图像作为输入。该评论认为它有可能在本地工作流中取代 **Wan 2.2**，但没有提供量化基准或具体设置。
    - 多位评论者关注部署方面的限制：有人询问 **AMD Radeon AI PRO R9700** 的 **`32GB VRAM`** 是否足够，也有人询问是否提供 **GGUF**，以及 GGUF 这类量化格式是否适用于该模型。该讨论串没有给出经过确认的显存需求、量化支持情况或后端兼容性细节。

  - **[GLM 5.3 Spotted](https://www.reddit.com/r/LocalLLaMA/comments/1ve9ms0/glm_53_spotted/)**（活跃度：556）：**这张图片是一个**技术类 GitHub 截图**，并不是表情包：图中显示了 `zai-org/z-ai-sdk-java` 的 `glm-5.3` 分支，以及由 `tomsun28` 连续提交的多次更新，例如 *“feat: update new models glm-5.3, support json schema”*。这似乎表明，SDK/API 即将支持 **GLM 5.3**，并支持 JSON Schema 结构化输出。帖子中附带的提交记录指向相关分支：[github.com/zai-org/z-ai-sdk-java/commits/glm-5.3](https://github.com/zai-org/z-ai-sdk-java/commits/glm-5.3)；截图见：[image](https://i.redd.it/2be4dd7305hh1.png)。**评论者据此认为，一款新的高性能中国开放模型可能即将发布；也有人指出，现在模型发布速度太快，下载并评测一个模型后，几乎马上就可能觉得它已经过时。

    - 一位评论者指出，**Microsoft Bing 在中国的搜索索引中已经出现了“GLM 5.3”相关信息**，并引用了一张截图以及 [AB Kuai.Dong 在 X 上发布的帖子](https://x.com/_FORAB/status/2084180211059617947)，认为这可能意味着该模型即将公开发布，或者至少已经出现在搜索元数据中。该讨论串将其视为高性能中国开放模型快速迭代的一部分，但评论中没有提供基准成绩、参数规模或正式发布文件。
    - 一个技术层面的讨论重点是，人们感受到**中国开放权重模型的发布速度正在加快**。评论者将其与 Xi Jinping 释放的支持开放源码信号，以及中国模型厂商与美国主要实验室之间的竞争联系起来。虽然这些讨论带有推测性质，但也反映了预期的变化：由于新模型发布得过于频繁，用户开始推迟下载，意味着模型更替速度已经超过了本地实际部署工作流的更新速度。

## 技术门槛更低的 AI 子版块简报

> /r/Singularity、/r/Oobabooga、/r/MachineLearning、/r/OpenAI、/r/ClaudeAI、/r/StableDiffusion、/r/ChatGPT、/r/ChatGPTCoding、/r/aivideo、/r/aivideo