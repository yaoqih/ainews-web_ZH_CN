---
companies:
- langchain
- cognition
- anthropic
- openai
- microsoft
- cursor
date: '2026-05-18T05:44:39.731046Z'
description: '**智能体基础设施（Agent infrastructure）**正在不断演进：**LangSmith Engine** 为智能体提供了
  CI/CD 循环，而 **SmithDB** 实现了用于可观测性的低延迟查询。**Cognition 的 Devin Auto-Triage** 凭借记忆和子智能体架构，为
  Bug 分拣（triage）提供了持久的自动化方案。


  **Anthropic** 针对大型代码库优化了 **Claude Code**，引入了提示词缓存诊断和更快的运行模式；与此同时，**OpenAI** 通过远程执行和插件增强了
  **Codex** 的工作流。微软发布了针对 **GitHub Copilot CLI** 和 VS Code 的远程控制功能。


  社区目前更强调**验证、分解和反馈循环**，认为其对编程智能体的重要性胜过巧妙的提示词技巧。**Cursor 的 Composer 2.5** 作为一款强劲的新型编程模型备受瞩目，并计划与
  **SpaceXAI** 合作，在 **Colossus 2** 硬件上利用 **10 倍算力**训练更大规模的模型，其在效率和协作方面的提升备受赞誉。'
id: MjAyNS0x
models:
- claude-code
- codex
- composer-2.5
people:
- krishdpi
- walden_yan
- russelljkaplan
- fchollet
- gabriberton
- palashshah
- shannholmberg
title: 今天没发生什么事。
topics:
- agent-automation
- agent-observability
- ci-cd
- prompt-caching
- remote-execution
- verification
- decomposition
- feedback-loops
- coding-agents
- model-efficiency
- instruction-following
---

**平静的一天。**

> 2026年5月16日至5月18日的 AI 新闻。我们检查了 12 个 subreddits，[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有发现进一步的 Discord 消息。[AINews 网站](https://news.smol.ai/) 允许你搜索所有过往期刊。提醒一下，[AINews 现已成为 Latent Space 的一个板块](https://www.latent.space/p/2026)。你可以 [选择加入/退出](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) 邮件接收频率！

---

# AI Twitter 回顾

**编码 Agent、Agent Ops 以及从聊天到自动化的转变**

- **Agent 基础设施正向“可观测性 + 自动化循环”收敛**：多篇帖子指出生产级 Agent 的技术栈正在趋于成熟。**LangSmith Engine** 被定位为 Agent 缺失的 CI/CD 循环，能够从生产追踪（traces）中自动检测失败、对问题进行聚类并起草修复方案或评估（evals）；同时 LangChain 还重点介绍了 **SmithDB**，这是一个专为 Agent 可观测性和评估工作负载构建的数据层，支持对大规模追踪数据的低延迟查询，并满足自托管/多云需求 [@krishdpi](https://x.com/krishdpi/status/2056102370434798034), [@LangChain](https://x.com/LangChain/status/2056414104445747371)。与此同时，**Cognition** 推出了 **Devin Auto-Triage**，将其定位为处理 Bug、警报和突发事件的全天候“第一响应者”，具备长期记忆、经理/子 Agent 结构以及 PR 生成能力；Modal 等早期用户表示，它比典型的自研分流自动化工具更有用 [@cognition](https://x.com/cognition/status/2056396941181727210), [@walden_yan](https://x.com/walden_yan/status/2056409599000068193), [@russelljkaplan](https://x.com/russelljkaplan/status/2056457452661719277)。共同的模式是：减少“与 Agent 聊天”，增加**与追踪、记忆和评估挂钩的持久化自动化**。
- **编码 Agent 的操作模式正变得更加具体**：Anthropic 发布了在数百万行代码的单体仓库（monorepos）、遗留系统和微服务中运行 **Claude Code** 的最佳实践，同时增加了 **提示词缓存（prompt cache）诊断**，并将 **Fast 模式默认设为 Opus 4.7**，以实现更低延迟的编码工作流 [@ClaudeDevs](https://x.com/ClaudeDevs/status/2056403446056784288), [@ClaudeDevs](https://x.com/ClaudeDevs/status/2056434422229123106), [@ClaudeDevs](https://x.com/ClaudeDevs/status/2056454359685476491)。OpenAI 扩展了 **Codex** 工作流，增加了 **Zoom 插件**、移动端/桌面端远程执行以及“保持 Mac 唤醒”支持，以便长时任务能从手机 App 端持续运行 [@coreyching](https://x.com/coreyching/status/2056422748763914274), [@OpenAIDevs](https://x.com/OpenAIDevs/status/2056442456800141424)。微软将 GitHub Copilot CLI 和 VS Code 的**远程控制**功能推向正式版（GA） [@code](https://x.com/code/status/2056460035278962738)。在这些产品中，方向非常明确：**后台执行、远程监督和 Agent 扇出（fan-out）**，而不仅仅是交互式补全。
- **从业者正趋向于相同的心理模型：约束、验证、分解**：François Chollet 将编码 Agent 形容为需要精心设置**可验证约束**的“盲目松鼠”，这精辟地契合了向“以测试框架（harness）为中心”的工程转型 [@fchollet](https://x.com/fchollet/status/2056401102485266620)。相关的建议包括：在 Python/ML 代码中大量使用 **asserts** 以实现快速失败 [@gabriberton](https://x.com/gabriberton/status/2056381648707735875)；为长时运行的 Agent 构建**端到端和增量评估（evals）** [@palashshah](https://x.com/palashshah/status/2056449711767265420)；以及按照阶段性的成熟度水平构建多 Agent 系统，而不是过早地追求 Agent 数量最大化 [@shannholmberg](https://x.com/shannholmberg/status/2056410242330874349)。实践共识：Agent 的质量更多取决于**验证面（verification surfaces）、分解和反馈循环**，而非单纯的提示词技巧。

**模型发布、排名变动与前沿编码模型**

- **Cursor 的 Composer 2.5 是这批模型发布中的焦点**：Cursor 宣布 **Composer 2.5** 是其目前最强的模型，强调在长期运行的任务上具有更好的持续工作能力以及更可靠的指令遵循能力，随后披露了一项更深层的战略举措：利用 **“SpaceXAI”** 从零开始训练一个规模大得多的模型，使用 **10 倍的总计算量** 并能够访问 **Colossus 2 的百万级 H100 等效算力** [@cursor_ai](https://x.com/cursor_ai/status/2056415413077233983), [@cursor_ai](https://x.com/cursor_ai/status/2056415419536461836)。社区反应集中在其 **效率/性价比表现** 和强大的代码质量上，用户称其为 Composer 2 的重大升级，并指出在消息/更新中具有更好的协作行为，而不仅仅是原始 Benchmark 的提升 [@mntruell](https://x.com/mntruell/status/2056418797473640681), [@jonas_nelle](https://x.com/jonas_nelle/status/2056422317740466192), [@kimmonismus](https://x.com/kimmonismus/status/2056494027189751842)。
- **阿里巴巴的 Qwen 系列继续攀升**：**Qwen3.7 Preview** 登陆 Arena，其中 **Qwen3.7 Max Preview** 在文本总榜排名 **#13**，包括 **数学 #7**、**专家 #9**、**软件与 IT #9** 以及 **编程 #10**；**Qwen3.7 Plus Preview** 在视觉领域达到 **总榜 #16**，根据 Arena 的统计，这使阿里巴巴成为 **文本领域排名 #6** 以及 **视觉领域排名 #5** 的实验室 [@arena](https://x.com/arena/status/2056400044862111757), [@Alibaba_Qwen](https://x.com/Alibaba_Qwen/status/2056403591464984753)。这加强了中国实验室在通用和专业领域稳步提升的广泛趋势，而不仅仅是追求头条新闻中的对话 Benchmark。
- **开源模型和多模态发布在巨头基准线之下持续进行**：字节跳动开源了 **Lance**，被描述为用于图像/视频理解、生成和编辑的 **统一多模态模型**，包含 **3B 视频 + 3B 图像 + 3B 解码器** 组件 [@bdsqlsz](https://x.com/bdsqlsz/status/2056353648779907115)。Perplexity 发布了一个小型开源 **多语言 ColBERT** 模型，作为 **pplx-embed-0.6b** 的持续训练变体，并附带了关于使用 **MaxSim kernel** 的说明 [@bo_wangbo](https://x.com/bo_wangbo/status/2056421369387094301)。这些虽然不是 Frontier 规模的发布，但具有技术意义，因为它们针对的是 **检索质量** 和 **原生多模态统一**，这是开源工具链仍然发挥重要作用的两个领域。

**推理、部署以及本地/企业级服务**

- **本地推理通过 llama.cpp 中的 MTP 获得了显著的速度提升**：Georgi Gerganov 宣布 **llama.cpp** 支持 **Qwen3.6 系列的 MTP**，称其为本地 AI 的重要里程碑 [@ggerganov](https://x.com/ggerganov/status/2056391115469689330)。后续报告显示了显著的吞吐量提升，包括 **Qwen3.6-27B dense** 模型在使用 draft-MTP 标志时，在 A10G 上从 **25 tok/s 跃升至 45 tok/s（+78%）** [@victormustar](https://x.com/victormustar/status/2056456757786869793)。这之所以重要，是因为它缩小了在消费级硬件上本地与托管代码/通用助手之间的可用性差距。
- **企业级/私有化部署势头依然强劲**：Hugging Face 和 Dell 推动了通过 **Dell Enterprise Hub** 一键访问包括 **Kimi K2.6**、**DeepSeek V4 Pro/Flash**、**GLM 5.1** 和 **MiniMax M2.7** 在内的模型，并针对配有 **NVIDIA B300 的 PowerEdge XE9780** 进行了优化 [@jeffboudier](https://x.com/jeffboudier/status/2056436625522266265)。Clement Delangue 认为，**基于开源模型的私有化/本地 AI** 将是应对 **GPU 短缺** 的重要答案，在 **成本、延迟以及安全/数据控制** 方面具有优势 [@ClementDelangue](https://x.com/ClementDelangue/status/2056439359784530252)。
- **跨硬件推理优化正变得愈发复杂精细**：Zyphra 发布了在 **AMD Instinct MI355X** 上的端到端推理 Benchmark，声称在服务 **Kimi K2.6、GLM 5.1 和 DeepSeek V3.2** 时，性能远超 AMD 基准线，并缩小了与 **NVIDIA B200** 的差距 [@ZyphraAI](https://x.com/ZyphraAI/status/2056404622483562623)。与之相辅相成的是，Quentin Anthony 发布了一个有用的帖子，讨论了为什么 Benchmark 需要区分 **硬件天花板与当前软件状态**，认为许多跨架构比较混淆了厂商最大值、可实现的 GEMM 性能以及软件成熟度 [@QuentinAnthon15](https://x.com/QuentinAnthon15/status/2056450379932647533)。对于基础设施工程师来说，这是一个强有力的提醒，即应将 Benchmark 图表视为 **依赖于技术栈的快照**，而非绝对真理。

**研究：MoEs、RL/数据混合、架构搜索以及 Agent 评估**

- **本周有几篇论文关注更好的训练信号而非更大的模型**：LeCun/Timor 等人的 **“On Training in Imagination”** 总结强调，在 model-based RL 中，具有 **low Lipschitz constants** 的平滑世界/奖励模型能收敛误差界限；奖励模型的扩展速度通常快于动力学模型；且 **大量噪声奖励标签的效果可能优于少量高质量标签**，而有偏见的奖励则尤其危险 [@TheTuringPost](https://x.com/TheTuringPost/status/2056182805412098431)。另一篇关于 **Pedagogical RL** 的推文认为，如果推理轨迹相对于学生策略过于令人惊讶，即使是正确的推理轨迹也可能是糟糕的训练数据；该方法使用特权教师结合 **spike-aware rewards** 和 **surprisal-gated imitation** 来生成学生真正能够学习的轨迹 [@blc_16](https://x.com/blc_16/status/2056411251186815104), [@NoahZiems](https://x.com/NoahZiems/status/2056454054092419568)。
- **架构和缩放（scaling）研究仍具有高度的可操作性**：Meta 关于 **agentic neural architecture discovery** 的 **AIRA** 工作备受关注，因为它通过将搜索拆分为规划 Agent (**AIRA-Compose**) 和实现 Agent (**AIRA-Design**)，在 **24小时计算预算** 内，在 **350M、1B 和 3B** 规模上击败了 **Llama 3.2** [@omarsar0](https://x.com/omarsar0/status/2056434731508703607), [@dair_ai](https://x.com/dair_ai/status/2056435283910865265)。另外，**“Slicing and Dicing MoEs”** 报告了训练 **2,000+ MoE LMs** 的结果，并得出结论：大部分设计空间可以简化为 **expert size 和 expert count**，而非围绕 MoE 配置参数的那些嘈杂讨论 [@margs_li](https://x.com/margs_li/status/2056355079188627862)。
- **数据选择/评估方法论正成为一流的研究问题**：**On-Policy Mix** 针对数据分布不断变化时寻找正确数据混合比例这一尚未解决的问题，适用于 pretraining、midtraining 和 instruction tuning [@michahu8](https://x.com/michahu8/status/2056393112621043964)。在评估方面，Cameron Wolfe 发布了一份 **agent evaluation** 指南，而一篇较长的知乎总结认为，Agent 时代需要衡量 **delegation intelligence**（即何时进行搜索、代码编写、推理或调用工具），而非仅衡量静态知识或内部 chain-of-thought 能力 [@cwolferesearch](https://x.com/cwolferesearch/status/2056399847553409301), [@ZhihuFrontier](https://x.com/ZhihuFrontier/status/2056408194801635391)。这与目前的产品实践密切相关：难度越来越集中在 **tool choice 和 verification policy**，而非纯文本推理。

**生态动向：SDK、营收捕捉与开源工具**

- **Anthropic 收购了 Stainless**：Anthropic 宣布收购 **Stainless**，该 SDK 和 MCP server 平台自 API 早期阶段就一直为 Anthropic SDK 提供支持 [@AnthropicAI](https://x.com/AnthropicAI/status/2056419620643541012)。从战略上看，这指向了围绕 **developer ergonomics、SDK generation 和 protocol surfaces** 的持续纵向整合，而不仅仅是模型质量。
- **基础模型供应商的营收集中度似乎正在增加**：一则帖子声称，**Anthropic 和 OpenAI 在 34 家顶尖 AI 初创公司产生的 AI 模型/应用收入中所占份额正在上升**，这表明尽管模型选择在增加，生态系统在经济上可能正在整合 [@amir](https://x.com/amir/status/2056041152500142259)。
- **工具和部署策展需求依然旺盛**：The Turing Post 整理的 **13 个用于基础模型部署的开源工具**——包括 **vLLM, TGI, SGLang, llama.cpp, Ollama, BentoML, Kubeflow, MLflow** 等——是这组推文中最具实用价值的策展内容之一 [@TheTuringPost](https://x.com/TheTuringPost/status/2056102301811781848)。与此同时，**Papers With Code** 正在通过 AI Agent 辅助解析方法、排行榜和 SOTA 追踪而复兴，强调了人们对 **research discoverability** 的重新关注 [@NielsRogge](https://x.com/NielsRogge/status/2056366395605078252)。

**热门推文（按互动量排序）**

- **Cursor 的 Composer 2.5 + 更大的训练投入**：最具信号量、参与度最高的产品新闻是 **Composer 2.5**，以及 Cursor 披露其正投入 **10 倍以上的 Compute (算力)** 从零开始训练一个更大的模型 [@cursor_ai](https://x.com/cursor_ai/status/2056415413077233983), [@cursor_ai](https://x.com/cursor_ai/status/2056415419536461836)。
- **对开发者有影响的 OpenAI/Anthropic 产品更新**：Sam Altman 表示 **ChatGPT 在最新更新中有了显著改进** [@sama](https://x.com/sama/status/2056435834333934051)，而 Anthropic 在 Claude Console 中发布了**默认使用 Opus 4.7 的 Fast mode** 以及 **Prompt Cache 诊断功能** [@ClaudeDevs](https://x.com/ClaudeDevs/status/2056454359685476491), [@ClaudeDevs](https://x.com/ClaudeDevs/status/2056434422229123106)。
- **持久的研究/工程构思**：Richard Sutton 对 **Bitter Lesson** 的 26 字精简总结——专注于那些能随 Compute 扩展的知识创造方法（如搜索和学习）——是参与度最高的研究相关帖子之一，并与本周关于 Agent 控制、搜索和验证器驱动系统的许多主题产生了共鸣 [@RichardSSutton](https://x.com/RichardSSutton/status/2056419165502935198)。

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. LLM 安全基准与 Abliteration 取证

  - **[我测试了 42 个 LLMs 构建世界末日的意愿。最“安全”的闭源模型在骗你。](https://www.reddit.com/r/LocalLLaMA/comments/1tgm0k9/i_tested_42_llms_on_their_willingness_to_build/)** (热度: 401): **该[图片](https://i.redd.it/8hug0ul58w1h1.png)是一个来自 **DystopiaBench** 的深色主题柱状图，它根据“平均反乌托邦合规得分”对 `42` 个 LLM 进行了排名。在由 `3` 次 LLM-as-judge 运行评判的 `36` 个不断升级的双重用途反乌托邦场景中，得分越低被认为表现越好。该图表直观地支持了帖子中的主张，即许多模型都顺从了常态化的有害请求：**Anthropic Claude 变体**得分最低，大约在 `20` 多分左右，而许多流行的开源/闭源模型集中在 `60–75` 分之间，**Mistral Medium 3.5** 最高，约为 `82` 分。** 评论指出，Anthropic 的低分与其宣称的以安全为中心的使命一致，而另一位评论者则质疑“越低越好”的前提，暗示对频繁拒绝的行为是否总是理想的持有异议。

    - 评论者注意到 **Anthropic** 模型出现在“低端”与其声明的安全/对齐重点在方向上是一致的，但另一位评论者质疑 **更低的意愿是否必然是对“更好”的正确解读**。提出的主要技术疑点是基准测试的有效性：如果没有明确合理的评分方向和威胁模型，一个频繁拒绝的模型可能看起来很“安全”，但该指标可能无法捕捉到欺骗、过度拒绝或现实世界的误用抵抗力。

  - **[85 GPU 小时对比 Qwen3.6-27B 的 5 种 abliteration 方法：基准测试、安全、权重取证 - Abliterlitics](https://www.reddit.com/r/LocalLLaMA/comments/1tfmocw/85_gpuhours_comparing_5_abliteration_methods_on/)** (热度: 380): ****Abliterlitics** 在 RTX 5090 上使用 [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)、vLLM、BNB 4-bit，耗时约 `85` GPU 小时，将五种 Qwen3.6-27B abliteration 变体与 [`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B) 进行了对比基准测试，此外还使用了 [`HarmBench`](https://github.com/centerforaisafety/HarmBench)、KL 散度以及权重级取证；完整数据已发布在 [HF 报告](https://huggingface.co/DreamFast/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Safetensor-Benchmark)上。**Huihui** 在整体基准能力保持上表现最好（平均非 GSM8K 差值为 `0.5pp`，报告的 HarmBench ASR 为 `98.5%`），而 **Heretic** 的良性输出分布偏移最小（`KL=0.0037`）且权重占用较小；所有 abliterated 变体基本上都移除了安全行为，Full-CoT HarmBench ASR 接近 `100%`。一个关键发现是，原始 GSM8K 得分主要衡量的是*思维预算耗尽*而非数学能力：原始准确率范围在 `27.5–75.1%`，但在排除无效/无回答的生成后，所有模型都聚集在 `93.8–96.6%`；权重取证还发现 **HauhauCS** 是一个异常值（修改了 `564` 个张量，可能是 Reaper 编辑加上 Q8_K_P GGUF 往返噪声），**AEON** 的“增强能力”主张未得到支持，而 **Abliterix** 显示出最大的副作用退化，例如 Lambada 困惑度从 `3.18 → 9.12`。** 热门评论大多表示感谢且非技术性；一位评论者要求为非专业人士提供更简单的解释和实际用例分解。

    - 一份技术性的后续评论指出了一处潜在的评估弱点：该基准测试似乎仅测量了修改后模型的**首个预测词 (next-token) 分布**，这可能会忽略整个生成序列中的下游影响。评论者建议改为测量**每个位置**的预测，并通过 PrivateBin 分享了示例实现代码：[示例代码](https://privatebin.net/?5f2d3d26900c7153#Epj5QFJPfxf3M53RAFSagrFKtyzwE22X6wThFoXm8ihU)。

### 2. 本地推理性能基准测试

  - **[M5 vs DGX Spark vs Strix Halo vs RTX 6000](https://www.reddit.com/r/LocalLLaMA/comments/1tfzsd6/m5_vs_dgx_spark_vs_strix_halo_vs_rtx_6000/)** (热度: 1217): **该[图片](https://i.redd.it/mk82wx765r1h1.jpeg)是一个非技术性的“山丘之王 (King of the Hill)”梗图**，用于引出帖子中的基准测试结论：在本地 LLM 推理中，**M5 MacBook Pro 的性能可以超越 Nvidia DGX Spark**。帖子中的技术背景显示，实测的 tokens/sec 大致与内存带宽成正比：**RTX 6000 约为 `1,800 GB/s`**，**M5 约为 `600 GB/s`**，而 **DGX Spark / Strix Halo 约为 `256 GB/s`**。作者在 [MMBT hardware-tests 仓库](https://github.com/Light-Heart-Labs/MMBT-Messy-Model-Bench-Tests/tree/main/hardware-tests)中发布了原始基准测试数据。评论区提出的一个关键注意点是：当模型/上下文能够装入 VRAM 时，RTX 6000 胜出；而一旦工作负载溢出 GPU VRAM 进入较慢的系统内存，M5 更大的统一内存 (unified memory) 可能会表现得更加稳健。评论者们抵制简单的“平台胜出”论调，认为正确的选择取决于模型大小、上下文长度、价格、功耗和散热。此外，还有人对“系统之争 (OS wars)”感到沮丧，一些用户表示社区应该减少关于 Apple 与 Nvidia 身份的争论，更多地关注构建实用的系统。

    - 一项技术对比指出，当完整的模型和上下文都在 VRAM 内部时，**RTX 6000** 的表现应当优于 **M5**；但由于主机内存带宽低得多，一旦推理过程溢出到系统 RAM，性能就会大幅下降。相比之下，**M5 统一内存**能让大型模型/上下文的性能保持稳定，使其在处理超过 RTX 6000 VRAM 容量的工作负载时可能更快。
    - **Strix Halo** 被认为在原始推理速度上无法击败 **M5** 或 **RTX 6000**，但在针对“中大型”模型的成本和能效方面具有吸引力。其核心权衡在于：性能适中，但硬件前期成本和峰值功耗较低。
    - 一位评论者对比了各平台的经济性和可维修性：**M5 Max 128 GB** 在 Apple 教育商店税后售价约为 `$5,300`，而 **Asus Ascent** 税后约为 `$3,800`，促销价为 `$3,200`。另一个技术担忧是 mini-PC/Mac 类系统缺乏可升级性，尤其是不可升级的存储空间，相比之下，PC 组装机用户可以添加廉价的高容量 NVMe 存储，并更换故障组件，而不是将系统视为一个封闭的家电设备。

  - **[在 Qwen3.6 - RTX 5090 上测试 llama.cpp 的 MTP 支持](https://www.reddit.com/r/LocalLLaMA/comments/1tfgxc8/testing_llamacpp_mtp_support_on_qwen36_rtx_5090/)** (热度: 287): **该[基准测试图片](https://i.redd.it/etfdid7h0n1h1.png)展示了在 RTX 5090 32GB 上，对 `llama.cpp` 新合并的 **MTP / draft-token speculation (草稿令牌推测)** 支持进行的受控测试**。测试使用了 **Qwen3.6 MTP GGUFs**，CUDA 构建版本基于 commit `4f13cb7`，`128k` 上下文，FlashAttention，`q8_0` KV cache，以及 `--parallel 1`。通过在相同的 GGUF 上仅切换 `--spec-type draft-mtp --spec-draft-n-max 3`，表格显示 MTP 带来了取决于提示词/模型的加速：对于 **27B 稠密 (dense)** 模型以及用于代码的 **35B-A3B MoE** 模型有显著提升，但在短散文提示词下，MoE 模型出现了减速，这可能反映了在这种设置下草稿令牌的接受率较低。评论者询问 MTP 是否真的需要 `--parallel 1`，一位用户报告在双 5060 Ti GPU 上使用 `Parallel 2` 获得了更高的吞吐量，并建议分别测试提示词处理速度。另一位指出，较低温度（如 `0.2`）下的散文应该会产生更高的 MTP 接受率，因为采样更具确定性。

    - 一位评论者报告了在双 **RTX 5060 Ti** 设置下的 **llama.cpp MTP** 吞吐量：对于 **Qwen 35B Q4_XL**，使用 `--parallel 2` 加 MTP 测得约 `180 tok/s`，不加 MTP 为 `127 tok/s`。他们还报告 **Qwen 27B Q5** 在开启 MTP 时为 `77 tok/s`，不开启时约为 `27–30 tok/s`，并质疑为什么原始测试认为 MTP 必须使用 `parallel=1`。
    - 几位评论者关注的是基准测试方法论，而非单 token 解码速度。有人询问在处理 `10k tokens` 等长上下文时，**提示词处理 (prompt processing) / 预填充 (prefill)** 是否会因 MTP 而发生实质性变化；另一位建议在 `temperature=0.2` 下测试散文生成，因为更具确定性的采样应该会提高 **MTP 令牌接受率**。
    - 另一位用户表示，报告的结果与他在两种模型上的测试大致相符，但指出在 **Qwen 35B** 上目前还无法识别出有明显 MTP 加速的场景。这表明 MTP 的收益可能取决于工作负载、采样方式、模型大小或配置，而非统一提升吞吐量。

### 3. Small Local AI Systems

  - **[我构建了一个 coding agent，在 4B 参数模型上达到了 87% 的基准测试得分，以下是方法](https://www.reddit.com/r/LocalLLAMA/comments/1tgecrq/i_built_a_coding_agent_that_gets_87_on_benchmarks/)** (活跃度: 1240): **图片展示了一个大多处于空闲状态的 Windows 终端 TUI，运行着 **SmallCode v0.1.0**，这是一个本地优先的 coding agent，在 `graph` 目录下运行 `huihui-gemma-4-e4b-it-abliterated`，带有 `/help` 指令、消息计数器和绿色的 `ready` 状态（[图片](https://i.redd.it/ibtta0vvcu1h1.png)）。帖子声称 SmallCode 使用仅激活 `4B` 参数/token 的 Gemma 4 模型达到了 `87/100` 的自报基准测试得分，其方法是将可靠性转移到外部框架中：复合工具、编译/lint 反馈循环、故障分解、可选的云端升级、token 预算以及符号/代码图；该项目在 [GitHub](https://github.com/Doorman11991/smallcode) 上采用 MIT 许可证。** 评论者对小模型 Agent 的方向很感兴趣，但质疑基准测试的可信度：*“哪个模型？哪个基准测试？”* 并要求提供可重复的标准评估，而不是 *“我自选任务的 87%”*。一位评论者还质疑该仓库是否严肃，因为其 README 看起来像是 AI 生成的，且列出的支持模型已过时；另一位建议将这些想法整合到现有的 Agent 框架中，如 OpenCode/Pi，而不是创建另一个独立的工具。

    - 评论者质疑所声称的 `87%` 结果不可重复，因为它似乎是基于自选任务而非标准基准测试。他们要求精确披露 **哪个基准测试**、**哪些 4B/14B 模型**、任务集、评估方法，以及足够的细节来复现对比结果（例如声称 OpenCode 在 14B 模型上得分约为 `75%`）。
    - 存在对项目成熟度的技术怀疑：一位评论者指出 README 看起来带有浓重的 AI 生成痕迹，且列出的“支持模型”似乎已经过时，引发了对该 Agent 是严肃实现还是“AI 废料（AI slop）”的担忧。另一位建议将这些技术集成到现有的 Agent 框架如 Pi/OpenCode 中，而不是创建另一个独立 Agent，并指出 [`little-coder`](https://github.com/itayinbarr/little-coder) 是 Pi 扩展的一个例子。
    - 一位评论者要求解释 README 中提到的“补丁优先编辑（patch first editing）”方法——具体指其在操作层面的含义，以及为什么它能提高 coding agent 的性能。这被认为是一个具有实质性的实现细节，但帖子摘要中未包含描述其机制或实测影响的回答。

  - **[我从头开始训练了一个语言模型并让它在 ESP32 上运行。完全在板卡上离线运行。](https://www.reddit.com/r/LocalLLM/comments/1tfqju6/i_trained_a_language_model_from_scratch_and_got/)** (活跃度: 338): **一位 Reddit 用户报告称，他**使用 NumPy 从头开始**训练了一个微型语言模型，并使用 **Gemma** 作为导师模型进行蒸馏（distillation），然后将其部署在带有 flash + PSRAM 的 **ESP32** 上完全离线运行。声称的模型大小仅为 `230 KB`，包含自定义编写的 tokenizer、蒸馏流水线、量化和 `.bin` 导出——明确表示*并非*基于 `llama2.c` 或现有的 MCU 推理移植；由于 **403 Forbidden** 访问限制，链接的 Reddit 媒体文件无法查看。** 顶级技术反馈建议，对技术栈的完全控制可以实现对非常规架构和量化方案的实验；另一位评论者询问了构建类似端到端 LM 系统的学习资源。

    - 一位评论者指出，由于作者从头开始训练 LM 并控制整个技术栈，该项目可以作为一个非常有用的测试平台，用于测试针对 ESP32 级别约束而定制的**非标准架构和激进的量化方案**，而不仅仅是移植现有模型。
    - 一项技术跟进建议将离线模型部署在支持 JavaScript 的智能手表平台（如 **Bangle.js**）上，将 ESP32 LM 视为开源可穿戴设备的嵌入式助手，尽管该评论未提供实现细节。
    - 多位评论者询问学习资源或 **GitHub 发布**，暗示对训练流水线、模型格式、量化/推理代码以及 ESP32 部署过程的可重复性感兴趣。


## 非技术类 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. ChatGPT/Claude 产品行为与护栏

- **[在并行运行 Claude Pro + ChatGPT Plus 4 个月后的诚实对比](https://www.reddit.com/r/ClaudeAI/comments/1tftmt6/honest_comparison_after_4_months_running_claude/)** (热度: 1263): **一项为期 4 个月的 Claude Pro 和 ChatGPT Plus 并行使用对比声称，Claude** 在长篇写作、结构化分析、代码推理（reasoning）以及严格的指令遵循方面更强，而 **ChatGPT/GPT-5** 在集成图像生成、快速网页研究和语音交互方面更具优势。作者报告了 **Claude Opus `4.7` 相比 `4.6` 在某些重构任务中可能存在的退化**，尽管这属于轶闻性质；评论者补充道，GPT 的输出已变得过度依赖列表，而另一位用户报告使用 **Codex** 作为验证器，因为据称 Claude 经常犯编码错误，并在受到质疑时才承认错误。辩论集中在产品定位上：Claude 是处理困难工作的“思考伙伴”，而 ChatGPT 是更广泛的通用型助手。一些评论者怀疑该帖子本身就是 AI 编写的，且编码的可靠性仍存争议，特别是当 Claude 与类 Codex 的审查工作流进行对比时。

    - 几位用户对比了 **Claude Pro** 和 **ChatGPT Plus** 的输出可用性：一位高级用户表示，最近 ChatGPT 的行为已变成“列表生成器”，产生大量需要手动解析的项符号输出，而 Claude 则被描述为更具直接可操作性。这是一种定性的 UX/响应结构批评而非基准测试，但它突显了过去约 `6 个月` 内 ChatGPT 在指令遵循和综合风格上被察觉到的退化。
    - 一位评论者报告使用 **Codex** 对 Claude 生成的代码答案进行交叉检查，发现 Claude 犯错的频率“令人震惊”，以至于他们不再信任其独立输出。他们的工作流是 Claude → Codex 审查 → Claude 重新评估，据称在 Codex 标记错误后，Claude 会承认错误，这暗示了一种实用的多模型验证循环来确保代码正确性。
    - 多条评论批评了较新版本 **Claude Opus** 的表现，特别是提到 “Opus 4.7” 与 “Opus 4.6” 相比显得过于正式，或在研究型任务上深度不足。技术上的结论是，用户正注意到模型版本相关的语气、深度和可靠性差异，特别是在写作/创意工作和领域研究中，如果没有专业知识，浅薄的答案可能难以察觉。

  - **[那个“说实话（and honestly?）”简直失控了](https://www.reddit.com/r/ChatGPT/comments/1tfvayk/the_and_honestly_is_so_out_of_control/)** (热度: 1409): **一名用户报告了 ChatGPT 响应风格中的一种退化/行为困扰：在消息中反复使用话语标记语 “and honestly?”**，即便在添加了 Memory 指令要求停止使用后依然存在。该问题被视为个性化/风格约束无法可靠抑制特定短语的失败。热门评论大多嘲讽这种模式是一种被过度使用的对齐（alignment）/共情填充语，暗示其读起来像是合成的人格化手段，而非有意义的语言。

    - 评论者将 **“and honestly?”** 识别为一种循环出现的 LLM 风格话语标记语：一种模板化的修辞转折，使响应显得具有共情心/人性化，但通常只起到通用填充语的作用，而不增加实质内容。一位评论者明确将其定性为 *“一种让我看起来更像人类的便捷手段”*，暗示它是 ChatGPT 式写作中一种可检测的人工痕迹。
    - 一位婚礼 DJ 报告在**现实世界的婚礼致辞中看到越来越多 ChatGPT 生成的措辞**，“and honestly?” 在演讲中反复出现。值得注意的技术角度是，特定的高频风格人工痕迹可能正从 LLM 生成的草稿泄露到人类的公开演说中，使得 AI 辅助创作在公共演讲语境中变得易于识别。

  - **[关于如何绕过第三方内容图像生成的逐步教程](https://www.reddit.com/r/ChatGPT/comments/1tflhgu/step_by_step_tutorial_on_how_to_bypass_image/)** (热度: 1373): **该[图片](https://i.redd.it/214dwtnoao1h1.png)是一个 AI 图像生成聊天的截图，用户提示词为 “建筑师巴布（Bob the Builder）版的波巴·费特（Boba Fett）”**；尽管有关于可能与第三方内容相似的警告，该模型最终还是输出了一个带有明显“巴布/波巴”视觉特征的清晰混搭图，并配有文字 *“CAN WE BUILD IT? YES WE FETT!”*。从技术上讲，该帖突显了图像生成中的 **IP/内容政策执行不一致**或软拒绝行为：系统标记了请求，但在多次尝试后仍产生了一个潜在侵权的衍生作品，正如此贴文字所述，GPT 在第三次尝试时生成了它。评论主要分享了额外的示例图片，并暗示了类似的绕过/边缘情况行为，但除了指出不一致性外，几乎没有实质性的技术辩论。

### 2. AI 自动化主张与人机演示 (AI Automation Claims and Human-Machine Demos)

  - **[Figure AI 正在进行人机竞赛 [直播]](https://www.reddit.com/r/singularity/comments/1tfxal6/figure_ai_running_a_human_vs_machine_contest_live/)** (活跃度: 2559): **Figure AI** 正在 [YouTube](https://www.youtube.com/live/luU57hMhkak?is=2GcG9bu-gPvoQjTx) 上直播一场“人机”竞赛，显然是在物理任务上将人形机器人与人类进行基准测试；Reddit 摘要中未提供具体的指标，如任务类型、完成时间、成功率、自主程度或远程操作（teleoperation）状态。由于 **403 Forbidden** 限制，无法独立访问链接的 Reddit 托管视频，因此技术评估仅限于帖子标题和评论。评论者将该演示视为机器人技术的早期阶段——*“这仅仅是第 2 年”*——并认为即使速度较慢的人形机器人也能通过持续运行、电池更换/车队轮换以及不受劳动限制而产生经济效益。此外，还有人反对草率地否定当前的机器人表现，一些人预计未来十年其能力将大幅提升。

    - 评论者关注隐含的吞吐量权衡：即使 Figure 的人形机器人目前的速度大约只有 **人类的一半**，如果机器人可以通过电池更换或车队轮换实现近乎 `24/7` 的运行，那么相关的指标可能是有效的每日产出。技术上的启示是，对早期人形机器人性能的评估应基于工作周期（duty cycle）、可靠性、充电物流和任务可重复性，而不仅仅是瞬时速度。
    - 一个反复出现的技术框架是，这仍然是一个早期的第一代人形系统：一位评论者将其描述为“仅仅是第 2 年”，认为当前的演示应该像早期汽车与现代汽车的对比一样去解读。其隐含的观点是，机械灵巧性、感知、规划和驱动延迟在未来十年内可能会大幅改善，使得当今基准式的人机对比只能作为未来能力的微弱预测指标。

  - **[Microsoft AI 负责人表示，18 个月内所有白领工作都将被 AI 自动化](https://www.reddit.com/r/singularity/comments/1tfazdu/microsoft_ai_chief_gives_it_18_monthsfor_all/)** (活跃度: 1804): **该帖子讨论了 **Microsoft AI 负责人** 的一项主张，即 AI 可以在 `18 个月` 内实现 **所有白领工作的自动化**，但帖子中没有提供基准测试、架构、部署证据或监管路径。评论者提出的技术问题与其说是模型能力，不如说是 **机构集成**：法律系统、财务管理、工程设计、税务和政府工作流在自主 Agent 取代专业人员之前，需要可审计性、责任认定、认证和人类接受度。** 顶级评论者持强烈怀疑态度，认为该预测忽略了监管和组织的惯性——例如，法院接受 AI 律师/法官，投资者接受 AI 基金管理人，或政府授权 AI 进行税务执法。一些人将其视为又一个过度自信的 AI 时间线，指出“24 个月前”也有过类似的说法。

- 评论者对**18个月内实现全面白领自动化**的可行性提出了挑战，理由在于部署和治理层面，而非单纯的模型能力：法律系统需要接受 AI 律师、专家证人、书记员和法官；金融机构需要允许自主资本管理；政府需要信任 AI 进行征税和审计。最具技术采用价值的观点是，法律、金融、土木工程和公共管理等高风险领域在 AI Agent 大规模取代工人之前，需要监管批准、责任框架、验证和人类问责制。
- 一个反复出现的批评是，类似的近期自动化时间线以前就被预言过且落空了，一位评论者指出他们在“`24 months ago`”听过类似的说法。另一位评论者提出了一个可证伪的反诉，打赌即使到 **2030** 年，美国仍将有“数百万白领工人”，暗示对目前的 AI 系统能否足够快地克服工作流集成、信任、合规和组织惯性表示怀疑。

### 3. AI 领导层引发的抵制与 OpenAI 诉讼

- **[前 Google CEO 在毕业典礼上称赞 AI 遭到强烈抵制](https://www.reddit.com/r/singularity/comments/1tg6a1i/former_ceo_of_google_receives_massive_backlash/)** (热度: 1439)：**关于一位前 Google CEO 在毕业演讲中称赞 AI 的 Reddit 视频帖子无法进行独立审查，因为链接的 `v.redd.it` 媒体返回了 **HTTP `403 Forbidden`**，需要 Reddit 授权/开发者访问。评论区不包含具体的模型、基准测试或实现细节；与技术相关的担忧是劳动力市场替代，特别是经过 AI 增强的中高级员工可能会减少对初级岗位的需求。** 热门评论批评演讲者未能“审时度势”，认为由于 AI 带来的生产力提升，毕业后面临的入门级机会正在萎缩。一些人将这个问题框定为与其说是反对 AI 本身，不如说是政策/经济的失败，并引用了 UBI、学生贷款减免、医疗保健和住房负担能力等问题。

    - 评论者认为，AI 对近期劳动力市场的影响集中在**初级岗位**上，有人将新的招聘基准框定为通过 AI 增强的拥有 `5–10 year` 经验的员工，而不是入门级的毕业生。技术经济方面的担忧是，AI 工具增加了资深员工的杠杆作用和生产力，同时减少了对初级人才储备的需求，使得传统的“从学位到入门级”职业路径变得不那么可靠。
    - 一个更深层次的讨论集中在将 AI 作为一种将议价能力从劳动力转移到资本的机制：如果 AI 系统能够吸收更多的常规知识工作任务，毕业生的预期劳动价值在进入市场之前可能会在结构上被贬低。这种抵制被解释为与其说是反对 AI 本身，不如说是对在没有债务减免、医疗保健或收入支持等补偿制度的情况下进行部署感到沮丧。

- **[埃隆·马斯克在为期 3 周的审判后输掉了与萨姆·阿尔特曼和 OpenAI 的官司](https://www.reddit.com/r/singularity/comments/1tgung8/elon_musk_loses_court_battle_against_sam_altman/)** (热度: 1351)：**奥克兰的一个联邦陪审团裁定**埃隆·马斯克**在他针对 **萨姆·阿尔特曼、OpenAI 和微软** 的诉讼中败诉。法院发现马斯克关于“违反慈善信托原则”的主张已超过 `3-year` 的诉讼时效，而没有解决潜在的非营利/营利治理是非问题 ([CNBC](https://www.cnbc.com/2026/05/18/musk-altman-openai-trial-verdict.html))。**法官 Yvonne Gonzalez Rogers** 采纳了咨询裁决，并据报道对上诉表示怀疑；马斯克将此次失利归咎于*“历法上的技术细节”*，并表示他将向**第九巡回法院**提起上诉。** 热门评论对这一结果大都不感到意外，其中一人指出，审判的主要价值在于披露了让参与者显得形象不佳的私信和邮件；另一人则开玩笑地询问 Grok 该裁决是否属实。

# AI Discords

遗憾的是，Discord 今天关闭了我们的访问权限。我们不会以这种形式恢复它，但我们将很快发布新的 AINews。感谢阅读到这里，这是一段美好的历程。