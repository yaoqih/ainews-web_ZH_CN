---
date: '2026-04-09T05:44:39.731046Z'
id: MjAyNS0x
title: 今天没发生什么事。
---

**平静的一天。**

> 2026年4月8日至4月9日的 AI 新闻。我们检查了 12 个 Reddit 子版块、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有查看更多 Discord。[AINews 网站](https://news.smol.ai/) 允许你搜索所有往期内容。提醒一下，[AINews 现在是 Latent Space 的一个板块](https://www.latent.space/p/2026)。你可以[选择订阅/退订](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)邮件发送频率！

---

# AI Twitter 回顾


**Mythos、Glasswing 以及向受限的网络能力模型的转变**

- **受限的网络模型发布正趋于常态化**：最大的主题是 Anthropic 的 **Mythos** 带来的持续影响，以及有关 OpenAI 正准备推出类似的受限网络能力模型/产品的报道。[@kimmonismus](https://x.com/kimmonismus/status/2042174533155836174) 总结了 Axios 的报告，称 OpenAI 拥有一个高级网络安全模型，并采用**有限、交错的发布方式**，效仿了 Anthropic 的做法；他随后澄清说，该受限模型**并非 “Spud”**，而是一个独立的系统[更新](https://x.com/kimmonismus/status/2042253455100699043)。辩论的焦点不再是这些模型在原则上是否危险，而更多在于当前的公开证据是否支持那些最戏剧性的主张。
- **社区抵触情绪集中在评估设计、基准上限和安全现实主义上**：一些技术评论认为，公众对 Mythos 的叙事领先于证据。[@paul_cal](https://x.com/paul_cal/status/2042139619840475491) 指出一个旗舰级的漏洞利用演示是**不真诚的**，并指出该演示仅给模型提供了约 20 行代码以及自定义上下文，而真正的漏洞发现需要**跨文件推理**。[@gneubig](https://x.com/gneubig/status/2042218039626674450) 重新定义了问题：软件已经存在**数百万个已知的未修复漏洞**，编码 Agent 在修复常规 CVE 方面可能比发现奇特的 zero-days 更有影响力。[@KentonVarda](https://x.com/KentonVarda/status/2042227174137061744) 做了模糊测试工具 (fuzzers) 的历史类比：广泛的自动化漏洞发现最终强化了软件，并且可能仍然有利于防御者。另一方面，[@boazbaraktcs](https://x.com/boazbaraktcs/status/2042131701728461313) 认为将此类模型保留在内部是有风险的，并鼓励 Anthropic 发布一个受限/公开的版本，而 [@ylecun](https://x.com/ylecun/status/2042224846881349741) 则将目前的许多讨论斥为“自欺欺人的废话”。来自 [@deanwball](https://x.com/deanwball/status/2042238507507134912) 等人的另一条讨论线索认为，如果这些系统能实质性地加速软件强化，它们可能会成为**网络安全的净利好**。

**Agent 框架、开放内存和新的基础设施栈**

- **LangChain 的 Deep Agents deploy 具象化了一种新兴架构**：[Deep Agents deploy](https://x.com/LangChain/status/2042268554364592543) 的发布构建了一个模型无关、面向生产的 Agent 框架，具备**开放记忆（open memory）**、沙箱支持、MCP/A2A 暴露，并能从相同的 Agent 定义栈进行部署。来自 [@hwchase17](https://x.com/hwchase17/status/2042271439496315102)、[@Vtrivedy10](https://x.com/Vtrivedy10/status/2042269655985741858) 等人的相关讨论强调，对于长时运行的 Agent，**记忆所有权是价值层**：专有的托管 Agent 产品存在将团队与其创造的最重要资产隔离开来的风险。其中最强烈的反复出现的设计原则是：**开放框架、模型选择、开放记忆、开放协议**。
- **沙箱正成为推理和 RL 的一等公民原语**：来自 [@sarahcat21](https://x.com/sarahcat21/status/2042269181396042036) 的一份非常有用的基础设施深度剖析描述了沙箱如何从支持代码 Agent 转向成为 **RL 后训练（RL post-training）**的核心底层，据报道，某大型实验室已经在运行约 **10 万个并发沙箱**，并目标达到 **100 万个**。该文章强调了为什么沙箱在这些工作负载中优于虚拟机（VM）：更低的开销、更强的针对奖励作弊（reward hacking）的隔离性，以及通过快照/卷（snapshots/volumes）对有状态工作流的更好支持。正如 [@Vtrivedy10](https://x.com/Vtrivedy10/status/2042089899415744729) 所言，这与广大从业者的观点一致，即未来的 Agent 评估（evals）将日益演变为**沙箱化环境**。
- **Hermes Agent 势头延续**：Nous/Hermes 展现了稳定的产品吸引力：[Multica](https://x.com/jiayuan_jy/status/2042097537981751544) 宣布支持；[@Teknium](https://x.com/Teknium/status/2042141382970876077) 添加了早期的 **iMessage/BlueBubbles 网关**支持；社区用户赞扬了其自动设置、技能积累和界面打磨，包括来自 [@aijoey](https://x.com/aijoey/status/2042290964497105048) 的带有**单模型 Token 成本追踪**功能的新型 Web 端 **Hermes HUD**。这些帖子背后的潜台词是，各团队现在不仅在优化模型，还在优化 **Agent 运行环境**本身。

**评估（Evals）、验证器和长周期 Agent 训练**

- **关于评估（evals）的讨论变得更加具体**：其中一个较好的概念性帖子来自 [@Vtrivedy10](https://x.com/Vtrivedy10/status/2042089899415744729)，他认为对于 Agent 来说，**“评估 ~= 训练数据 ~= 环境”**。这一框架在全天反复出现：生产轨迹（traces）变成评估；评估变成优化目标；环境变成评估的更丰富、带有奖励的版本。[@_philschmid](https://x.com/_philschmid/status/2042248973847613878) 也表达了从 API 时代软件向 Agent 的相同转变：**文本即状态**，移交控制权，并从单元测试转向评估。
- **验证器和长周期评估的新工作填补了缺失的环节**：[@omarsar0](https://x.com/omarsar0/status/2042249194409501054) 重点介绍了 Microsoft 的 **Universal Verifier**，它通过更好的评分标准（rubric）设计、区分过程奖励与结果奖励、以及跨截图轨迹的分治上下文管理，将 Web 任务验证的误报率从先前系统的 **45%+ / 22%+** 降低到接近于零。另外，[@GenReasoning](https://x.com/GenReasoning/status/2042204629321019537) 推出了 **KellyBench**，这是一个针对前沿模型的为期一年的体育博彩环境；其核心结果非常严峻：**每一个测试的前沿模型都在亏钱**，这表明目前的系统在真正的非平稳（non-stationary）设置中，仍然在适应、风险管理和学习方面挣扎。[@teortaxesTex](https://x.com/teortaxesTex/status/2042211750636822746) 指出在该基准测试中，只有 **Opus 4.6** 和 **GPT 5.4** 避免了彻底破产。
- **Agentic RL 的失败模式变得更加清晰**：[@zoltansoon](https://x.com/zoltansoon/status/2042212868372758772) 提到了关于 Agentic RL 中推理崩溃（reasoning collapse）的新论文 **RAGEN-2**：经 RL 训练的 Agent 可能看起来多样化，但大多是在重复模板，具有高熵但互信息（mutual information）接近于零。与此同时，来自 [@dair_ai](https://x.com/dair_ai/status/2042237615492260249) 的一个在实践中可能更重要的代码 Agent 训练方向是：针对定位、编辑、测试生成、复现和评审等**原子技能（atomic skills）**进行训练，产生了 **18.7% 的提升**，并且比单纯的端到端优化能更好地迁移到复合软件任务中。

**模型与产品发布：Meta Spark、Gemma 4、MedGemma 以及本地推理**

- **Meta 的首个 MSL 发布——“Muse/Spark”，既是一个模型故事，也是一个消费者分发的故事**：来自 [@alexandr_wang](https://x.com/alexandr_wang/status/2042142866697548189) 和 Meta 相关研究人员的帖子将其定义为迈向“个人超级智能”的早期里程碑，但更犀利的外部分析来自 [@kimmonismus](https://x.com/kimmonismus/status/2042184756553756679)：真正的威胁不在于前沿的编程或数学能力，而在于 Meta 可以将其强大的免费助手分发给其现有产品界面中的 **10 亿多用户**。产品增长信号立竿见影，[据 Alexandr Wang 称](https://x.com/alexandr_wang/status/2042254047244398978)，Meta AI 一夜之间攀升至 **App Store 排行榜第 6 名**。在技术方面，[@ahatamiz1](https://x.com/ahatamiz1/status/2042152600540237953) 强调了一个显著的 RL 发现：**思考过程中的相变**，即推理过程先变长，然后压缩，接着再次扩张——这暗示了**自适应计算路由（adaptive compute routing）**的新空间，而非单纯依靠暴力延长的 CoT。
- **Gemma 4 的本地/开源影响力持续引起共鸣**：[@kimmonismus](https://x.com/kimmonismus/status/2042155315190489360) 捕捉到了其现实吸引力：一个对于许多日常任务“完全足够”、**可本地运行**、免费且安全的模型——但在资深用户之外仍鲜为人知。Google DeepMind 随后分享道，**Gemma 4 在发布首周下载量就突破了 1000 万次**，整个 Gemma 家族的总下载量已超过 **5 亿次** [公告](https://x.com/GoogleDeepMind/status/2042283481640615944)。工具链生态系统已经在迅速跟进：[Together AI](https://x.com/togethercompute/status/2042264479069564958) 添加了支持 256K 上下文及多模态/工具使用的 **Gemma 4 31B**；[@danielhanchen](https://x.com/danielhanchen/status/2042270043011162246) 指出，使用 **Unsloth** 对 Gemma-4-31B 进行微调，甚至可以在免费的 Kaggle T4 上运行，显存占用仅约 **22GB VRAM**。
- **领域模型在悄然改进**：[@kimmonismus](https://x.com/kimmonismus/status/2042213170983358496) 重点介绍了 **MedGemma 1.5**，这是一个开源权重的 **4B** 医疗模型，涵盖 3D 放射学、病理学、纵向 X 光片和临床文档，据报告其在病理学方面的 **F1 分数提升了 47%**，在 MRI 分类方面比 v1 版本提升了 **11%**。在临床部署方面，[@GlassHealthHQ](https://x.com/GlassHealthHQ/status/2042324684273058236) 推出了 **Glass 5.5**，声称在 **9 个临床准确度基准测试**中表现优于前沿通用模型，并将 API 价格降低了 **70%**。

**推理、检索与系统效率**

- **效率优化工作依然如火如荼，特别是针对本地/通用硬件部署**：[@wildmindai](https://x.com/wildmindai/status/2042176348286788051) 挖掘出了 **RotorQuant**，声称能实现 **>10x 的 KV cache 压缩**、**解码速度提升 28%**、**预填充（prefill）速度提升 5 倍**，且在保持全注意力质量的前提下**参数量减少 44 倍**。在服务端，[@turbopuffer](https://x.com/turbopuffer/status/2042256535989125461) 分享了一个具体的架构优化：针对对象存储的特定写入策略，通过增加提交频率，使 S3 上的**写入延迟降低了约 2.5 倍**，这说明了向量/Agent 后端在很大程度上仍依赖于底层存储行为。
- **检索和表示研究继续挑战存储与计算的权衡**：[@gabriberton](https://x.com/gabriberton/status/2042061796400624157) 重新引起了人们对 **Matryoshka Representation Learning** 的关注，这是一种实用的 Embedding 理念，较短的前缀仍保持有效，从而降低海量语料库的检索/存储成本。来自 [@omouamoua](https://x.com/omouamoua/status/2042163510352789937) 的社区回应将其与 **late interaction** 系统联系起来：如果每个向量保持低维，那么在不爆炸式增加单个向量成本的情况下，增加**每个输入的向量数量**可能会消除干扰项。
- **NVIDIA 和 SGLang 贡献了显著的系统理念**：[@SemiAnalysis_](https://x.com/SemiAnalysis_/status/2042286547769184644) 指出了 NVIDIA 为 GB200 NVL72 级系统提出的 **DWDP** 推理并行策略，有效地利用更多的 GPU 间带宽来换取预填充期间更少的集合通信屏障停顿（collective-barrier stalls）。[@AndrewYNg](https://x.com/AndrewYNg/status/2042289428702642588) 宣布了一门关于 **SGLang** 的短课，重点关注 KV cache 实现、**RadixAttention** 和扩散加速——这反映出推理工程已变得足够核心，值得面向主流从业者进行培训。

**热门推文（按参与度排序）**

- **清理 vibe-coded 代码仓库的死代码**：[@gabriberton](https://x.com/gabriberton/status/2042141119837012284) 发布了当日最有价值的实用技巧：“删除所有死代码。使用 **ruff** 和 **vulture**。” 这一点不仅是为了代码整洁；更少的不相关文件意味着**更少的 Tokens**、更低的成本，以及通常更好的 Agent 推理能力。
- **OpenAI 围绕 Codex 调整定价**：[@OpenAI](https://x.com/OpenAI/status/2042295688323875316) 推出了一档新的**每月 100 美元的 ChatGPT Pro 套餐**，其 **Codex 使用量是 Plus 的 5 倍**，而现有的 **200 美元 Pro 套餐**仍是使用量最高的选项，并获得了另一次临时的 Codex 额度提升 [详情](https://x.com/OpenAI/status/2042296046009626989)。
- **Anthropic 的 Advisor/Executor 模式**：[@claudeai](https://x.com/claudeai/status/2042308622181339453) 发布了一种平台模式，由 **Opus 担任顾问 (Advisor)**，**Sonnet/Haiku 执行 (Executor)**，旨在以更低的成本实现接近 Opus 的性能——这是许多团队已经在采用的设计方案的产品化版本。
- **Gemini 交互式可视化**：[@GeminiApp](https://x.com/GeminiApp/status/2042272415951253932) 在对话中推出了针对问题和概念的**交互式可视化**功能，包括可调变量和 3D 探索——这是助手从文本输出转向可执行的解释性媒体的一个值得注意的例子。


---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Gemma 4 模型更新与修复

  - **[Gemma 4 在 Llama.cpp 上现在应该稳定了](https://www.reddit.com/r/LocalLLaMA/comments/1sgl3qz/gemma_4_on_llamacpp_should_be_stable_now/)** (活跃度: 673): **最近合并到 `llama.cpp` 仓库的 [PR #21534](https://github.com/ggml-org/llama.cpp/pull/21534) 已经解决了 Gemma 4 的所有已知问题。根据用户反馈，在 `Q5` 量化下运行 `Gemma 4 31B` 表现稳定。关键的运行时配置包括：使用带 Aldehir 交错模板的 `--chat-template-file`、`--cache-ram 2048` 以及 `-ctxcp 2` 以有效管理 RAM 使用。值得注意的是，已确认 **CUDA 13.2** 存在故障，会导致构建不稳定，应避免使用。社区强调应使用来自 master 分支的最新源码，而不是依赖滞后的 Release 版本。** 评论者强调了避开 CUDA 13.2 的重要性，并建议进行手动调整，如设置 `--min-p 0.0` 和 `-np 1` 以优化性能和 RAM 使用。一些用户通过自动化更新和重新编译 `llama.cpp` 来保持与最新修复同步。

    - **Tiffanytrashcan** 警告不要在 Llama.cpp 上将 **CUDA 13.2** 用于 Gemma 4，因为存在持续的不稳定性问题。正如 [此 Reddit 线程](https://www.reddit.com/r/unsloth/comments/1sgl0wh/do_not_use_cuda_132_to_run_models/) 中详述的那样，这对用户避免模型运行时的故障行为至关重要。
    - **Ambient_temp_xeno** 强调了在 Llama.cpp 上使用 Gemma 4 时需要手动配置。用户应手动添加 `google-gemma-4-31B-it-interleaved.jinja` 模板，并调整 `--min-p 0.0` 和 `-np 1` 等设置以优化 RAM 使用和性能，因为默认设置可能不是最优的。
    - **Chromix_** 指出，当使用低于 Q5 的量化级别时，Llama.cpp 中的音频能力会受到影响，并参考了一个 [GitHub pull request](https://github.com/ggml-org/llama.cpp/pull/21599)。这表明用户在使用量化设置时应保持谨慎，以维持音频性能。

  - **[看来我们需要下载新的 Gemma 4 GGUF 了](https://www.reddit.com/r/LocalLLaMA/comments/1sfrrgz/it_looks_like_well_need_to_download_the_new_gemma/)** (活跃度: 746): **新的 **Gemma 4 GGUF** 已更新，解决了若干技术问题并进行了增强。关键更新包括：支持异构 iSWA 中的注意力旋转、CUDA 缓冲区重叠的重大修复，以及 BPE detokenizer 在处理字节 Token 方面的增强。此外，更新将 'add bos' 设置为 true，为 Gemma 4 引入了专门的解析器，并实现了自定义换行拆分。这些变更详见 [GitHub pull requests](https://github.com/ggml-org/llama.cpp/pulls?q=is%3Apr+is%3Aclosed+Gemma+4)。** 评论者正在询问 Bartowski 和 Heretic 等其他版本是否也需要类似的更新，这表明了用户对不同模型版本间一致性的广泛关注。

    - Curious-Still 询问除了 'unsloth' 版本外，'bartowski' 版本的模型是否也需要更新。这表明用户关注不同模型变体的兼容性或改进，这可能是由于 Tokenizer 或架构的变化影响了性能或准确性。
    - shockwaverc13 将其类比为 'Llama 3 Tokenizer 问题'，表明目前 Gemma 4 GGUF 的情况可能涉及与 Tokenizer 更新类似的挑战。这意味着可能存在向后兼容性问题，或者需要重新处理数据以对齐新的 Tokenization 标准。
    - segmond 分享了先观望再下载模型的策略，理由是通常需要重复下载模型多次（3到5次）直到其稳定。这反映了用户为避免早期版本问题的一种普遍做法，尤其是对于像 GLM5.1 这样的大型模型，初始版本往往会经历密集的迭代和 Bug 修复。

### 2. 本地 LLM 使用案例与经验

  - **[本地（小型）LLM 发现了与 Mythos 相同的漏洞](https://www.reddit.com/r/LocalLLaMA/comments/1sgrfp1/local_small_llms_found_the_same_vulnerabilities/)** (活跃度: 592): **文章指出，较小的本地 LLM（例如 Gemma 4 31B）能够识别出与 Anthropic 的 Mythos 等大型模型相同的漏洞，这挑战了模型规模与网络安全有效性直接相关的观念。该研究使用了较旧的模型，如 Qwen3 32B、DeepSeek R1 和 Kimi K2，尽管已有更新版本如 Qwen3.5 27B、DeepSeek V3.2 和 Kimi K2.5 可用，后者可能会产生更好的结果。研究强调了模型架构和安全专业知识比单纯的规模更重要，并暗示在不同任务中存在“锯齿状”的性能分布（jagged frontier）。更多详情请参阅[原文](https://aisle.com/blog/ai-cybersecurity-after-mythos-the-jagged-frontier)。** 评论者批评测试中选择了过时的模型，认为更新版本表现会更好。此外，关于识别漏洞中“发现阶段”的重要性也存在争议，据报道该文章对此一笔带过。

    - coder543 强调了文章测试中使用了过时的模型，如 Qwen3 32B、DeepSeek R1 和 Kimi K2，尽管 Qwen3.5 27B、DeepSeek V3.2 和 Kimi K2.5 等更新版本已发布。他们还注意到缺少了目前领先的开源权重模型 GLM-5.1，这表明文章的发现可能无法反映目前最先进模型的真实能力。
    - One_Contribution 和 Decent_Action2959 讨论了文章中使用的逻辑，强调这些小型模型是给定特定漏洞进行分析，而不是独立发现漏洞。这种区别至关重要，因为它凸显了验证已知漏洞与发现新漏洞（这是 Mythos 采用的方法）这一更复杂任务之间的差异。
    - Quartich 指出文章的标题和内容可能具有误导性，因为这些小型模型的任务是分析预先识别出的漏洞代码片段，而不是独立寻找漏洞。这表明在文章语境下，模型的能力可能被夸大了。

  - **[终于发生了，我真的有了一个本地 LLM 的使用场景，而且效果非常棒](https://www.reddit.com/r/LocalLLaMA/comments/1sg2686/it_finally_happened_i_actually_had_a_use_case_for/)** (活跃度: 844): **该帖子描述了在没有互联网连接的飞行过程中使用名为 **Gemma 4** 的本地 Large Language Model (LLM) 的实际案例。用户经历了严重的航空性鼻窦炎（aerosinusitis），并使用 LLM 寻找解决方案，特别是 *Toynbee Maneuver*（汤恩比操作法），在 10 分钟内缓解了疼痛。这凸显了本地 LLM 在无法接入互联网的情况下的实用性，展示了它们在现实场景中提供即时、实际帮助的潜力。** 评论者指出了在离线使用时拥有小型设备端模型的重要性，强调了本地 LLM 在没有互联网连接的情况下提供有价值信息的实用性。人们还对这类模型的紧凑性和知识容量表示赞赏。

    - PassengerPigeon343 强调了运行本地 LLM 的实际益处，特别是在没有网络连接的情况下。他们提到在家庭服务器上运行较大的模型，但在设备端保留较小的模型以备不时之需，强调了本地模型在各种情况下的灵活性和实用性。
    - FenderMoon 讨论了使用本地 LLM 处理医疗建议等敏感任务时的隐私优势。他们对基于云的 AI 服务可能产生的数据泄露表示担忧，认为本地模型在处理个人信息方面提供了更安全的替代方案。
    - ObsidianNix 建议在涉及医学术语的任务中使用 MedGemma 等专业模型。他们指出，MedGemma 比标准 LLM 接受了更多医学术语的训练，使其在处理医疗相关查询时特别有效。

### 3. 新模型发布与基准测试

  - **[Meta 没有放弃开源](https://www.reddit.com/r/LocalLLaMA/comments/1sfzdrv/meta_has_not_given_up_on_opensource/)** (热度: 467): **这张图片是来自 **AI at Meta** 的推文，宣布推出 **Muse Spark**，这是由 Meta Superintelligence Labs 开发的 Muse 系列新模型。Muse Spark 被描述为一个多模态推理模型，具备 tool-use 和多 Agent 编排等能力。它目前已在 meta.ai 和 Meta AI 应用上可用，并计划开源未来版本。该公告还提到将通过 API 向选定的合作伙伴开放该模型，表明 Meta 对开源倡议的持续承诺。** 评论中对 Meta 开源该模型的承诺表示怀疑，用户质疑该公司的意图，并认为开源的决定完全由 Meta 控制。


  - **[GLM-5.1 声称拥有接近 Opus 级别的编程性能：营销噱头还是确有其实？我运行了自己的测试](https://www.reddit.com/r/LocalLLM/comments/1sft0n9/glm51_claims_near_opus_level_coding_performance/)** (热度: 338): **该帖讨论了 **GLM-5.1** 模型的性能，该模型声称达到了接近 **Opus 级别的编程性能**。作者在涉及具有多步、跨文件依赖关系的遗留后端系统的复杂重构任务中对其进行了测试，发现 GLM-5.1 能够保持状态并有效地进行自我纠正。该模型在涵盖 **SWE-Bench Pro**、**Terminal-Bench 2.0** 和 **NL2Repo** 的综合基准测试中得分为 `54.9`，而 Opus 为 `57.5`。值得注意的是，GLM-5.1 在被认为难以操纵的 SWE-Bench Pro 基准测试中超越了 Opus。这表明，虽然 Opus 在深度推理方面可能仍有优势，但 GLM-5.1 以更低的成本为长流程、多步骤的编程任务提供了极具竞争力的性能。** 评论者普遍支持 GLM-5.1 的真实性，并指出它是中国开发者中替代 Anthropic 模型的热门选择。一些用户发现它可与 Opus 4.5 媲美，并在某些任务中优于 Opus 4.6，强调了其慷慨的使用配额和在实际应用中的有效性。

    - HenryThatAte 提到在工作相关的任务中使用 GLM-5.1，指出与在处理三个类之后就耗尽配额的 Sonnet 相比，它提供了更慷慨的额度。这表明 GLM-5.1 可能更适合大型工作负载或扩展使用场景。
    - Hoak-em 将 GLM-5.1 与 Opus 4.5 和 4.6 进行了比较，表示在性能方面更倾向于 GLM-5.1。他们提到在 Forgecode 中使用它，并考虑为特定任务保留像 Qwen 397b 或 Minimax m2.7 这样较小的本地模型，突显了 GLM-5.1 在各种编程环境中的灵活性和适应性。
    - Fantastic_Run2955 强调了从 GLM-5 到 5.1 显著的编程改进，并将此归功于 Zai 有效的 Post-training 技术。这表明 GLM-5.1 的增强不仅是增量式的，而且具有显著影响力，这可能得益于先进的训练方法论。




## 侧重非技术的 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Anthropic 的 Mythos 模型及相关进展

  - **[纽约时报：Anthropic 的克制是一个可怕的警示信号](https://www.reddit.com/r/singularity/comments/1sggktf/new_york_times_anthropics_restraint_is_a/)** (热度: 732): **《纽约时报》的文章讨论了 **Anthropic 对 AI 开发采取的谨慎态度**，强调了对超智能模型潜在滥用的担忧。Anthropic 主张将这些模型限制在“负责任的政府和公司”手中，并将其与核不扩散进行了类比。文章指出，**Anthropic 对其 AI 能力的快速进步感到惊讶**，这表明实现 AGI 的时间表可能被低估了。文章还提到，Anthropic 向特朗普政府简报了国家安全影响，表明了这些进展受重视的程度。** 评论者对禁止 AI 的可行性表示怀疑，考虑到 AGI 的全球竞争。人们对定义“负责任的政府”以及儿童或恶意行为者滥用的可能性提出了担忧，强调了对基础设施脆弱性的恐惧，以及进行类似于核军控的国际合作的必要性。

- 该文章强调了对 AI 能力快速进步的担忧。据报道，Anthropic 对其自身模型的表现感到惊讶，这表明实现超智能 AI (superintelligent AI) 的时间表可能被低估了。这引发了人们对当前系统处理此类进步的准备程度的质疑，特别是考虑到在支撑电网和医院等关键基础设施的主要操作系统和 Web 浏览器中发现的漏洞。
- 讨论将 AI 的发展与核不扩散 (nuclear nonproliferation) 进行了类比，强调了国际合作的必要性，特别是中美之间在管理 AI 潜在风险方面的合作。这一对比强调了 AI 作为“文明拐点 (civilizational inflection point)”的严重性，需要目前尚且缺乏的合作水平。
- 关于是否应将 AI 模型限制在负责任的实体内以防止误用的争论正在进行，例如防止拥有这些模型访问权限的个人发起网络攻击。令人担忧的是，如果没有适当的控制，AI 可能会使针对基础设施的复杂攻击成为可能，而此类攻击此前只有国家机构或大型犯罪组织才能发起。

- **[Anthropic 关于 Mythos 文章中的疯狂图表](https://www.reddit.com/r/singularity/comments/1sf8o3q/insane_graph_from_anthropics_article_on_mythos/)** (热度: 471): **来自 Anthropic 关于 Mythos 文章的图片展示了一张对比不同 AI 模型在利用 Firefox JS shell 漏洞方面成功率的图表。**Mythos Preview 模型**展示了显著更高的成功率，实现了 `72.4%` 的成功漏洞利用和 `11.6%` 的未完全利用但实现寄存器控制。相比之下，**Sonnet 4.6** 和 **Opus 4.6** 的表现明显较低，分别只有 `4.4%` 和 `14.4%` 实现了寄存器控制，且没有成功的漏洞利用。这突显了 Mythos Preview 模型在这一特定任务中的先进能力。** 一条评论幽默地表示 AI 的能力被低估了，而另一条评论则强调了软件开发中对先进 AI 驱动的渗透测试 (pentesting) 的潜在需求，暗示了 AI 在网络安全中日益增长的角色。

    - Sufficient-Farmer243 质疑了 Anthropic 的 Mythos 在漏洞利用方面的成功，尽管 Anthropic 保持了透明度，但他仍表示怀疑。这表明需要更多关于 Mythos 能力及其在漏洞利用任务中所采用的具体方法的详细技术见解。
    - the_pwnererXx 幽默地建议，持续集成和持续部署 (CI/CD) 流程现在应该包含用于渗透测试的 AI Agent 集群 (swarms)，暗示了 AI 时代软件安全措施日益增加的复杂性和成本。
    - LucidOndine 将 Mythos 比作石墨烯 (graphene)，暗示虽然两者在技术上都令人印象深刻，但在受控环境之外的实际部署面临着重大障碍，这可能是由于扩展性或安全性方面的考虑。

- **[突破性消息：据报道 Anthropic 的新 “Mythos” 模型在草帽一伙之前找到了 One Piece](https://www.reddit.com/r/ClaudeAI/comments/1sgs0b4/breaking_anthropics_new_mythos_model_reportedly/)** (热度: 2222): **据报道，**Anthropic** 开发了一个名为 **Mythos** 的新推理模型，据称该模型在一次基准测试中定位了虚构的宝藏 “One Piece”，并在 `11 秒` 内完成了任务。这引发了围绕 AI 模型解决复杂的虚构谜题能力的幽默叙事。公告还提到了 **Project Glasspoiler**，这是一项利用 AI 保护叙事完整性免受剧透影响的计划。**OpenAI** 幽默地声称他们的模型先找到了宝藏，但为了尊重故事而隐瞒了信息。** 评论幽默地将 Mythos 模型的能力扩展到了其他虚构叙事中，暗示它可以开发 “GTA 6” 或完结《权力的游戏》(Game of Thrones)，突显了对 AI 触及创意领域的一种戏谑性的怀疑。

- **[Anthropic 最近遭遇的一连串“厄运”正是国家级 AI 攻击的典型特征](https://www.reddit.com/r/ClaudeAI/comments/1sg5088/anthropics_recent_run_of_bad_luck_is_exactly_what/)** (Activity: 569): **Anthropic 推出了一款名为 'Mythos' 的 AI 模型，该模型无意中发现了广泛使用的软件中的 'zero-day' 漏洞，凸显了 AI 模型在未经过专门的网络攻击训练的情况下发现安全漏洞的潜力。这引发了人们对国家级行为体（如中国）可能利用类似的 AI 能力进行网络攻击的担忧，正如他们此前在 Claude 等模型上所展示的那样。帖子指出，Anthropic 最近发生的安全事件，如“配置错误的 CMS”和源代码泄露，可能预示着国家资助的侦察行为，而非仅仅是“运气不佳”。这些事件可能是削弱 Anthropic 基础设施和声誉策略的一部分，从而影响公司及其用户。** 一位评论者认为，私人个体（如亿万富翁）掌握先进技术所构成的威胁与国家资助的攻击类似，暗示私人实体也可能利用 AI 达到恶意目的。

    - Atoning_Unifex 强调了私人个体（尤其是亿万富翁）利用先进技术进行恶意用途的潜力。他们认为，像 AI 和数据中心这样的顶尖技术对普通公民也是可用的，这暗示像 Elon Musk 这样的人在理论上可以构建像 Rehoboam 这样复杂的系统，即使他们无法获得核武器能力。
    - TimeSalvager 对针对 Anthropic 的国家级攻击这一观点提出了批评，认为观察到的问题更有可能是由于内部挑战而非外部破坏。他们辩称，如果国家级行为体拥有普遍的访问权限，他们会避免引起注意，这暗示目前的情况更像是一家在成长过程中挣扎、将安全视为事后补救的公司，并援引汉隆磨刀石（Hanlon's razor）和奥卡姆剃刀（Occam's razor）原则来支持这一观点。
    - emulable 讨论了在考虑潜在的国家级攻击时，分析成本效益流向的重要性。他们建议检查谁受益、谁买单，并指出如果 Anthropic 与政府之间的利益流向存在显著差异，可能值得进一步调查。他们强调，虽然不能确定，但观察到单向的巨大利益流向可能表明存在值得探索的深层因素。

  - **[我利用泄露源代码中引用的 Mythos 架构模式重构了 Claude Code 的提示词方式，效果简直天差地别](https://www.reddit.com/r/ClaudeCode/comments/1sflemo/i_used_the_mythos_referenced_architecture/)** (Activity: 986): **该 Reddit 帖子讨论了用户如何根据泄露的源代码洞察重构其针对 **Claude Code** 的提示策略。源码显示，Claude Code 采用了一个多 Agent 编排系统，具有一个可以生成并行 Worker 的协调者（coordinator）模式，一个包含 40 多个工具及其风险分类的工具注册表，以及一个基于 ML 的自动批准系统。用户调整了其提示词以匹配该架构，从而提升了性能。他们在执行前实现了一个规划阶段，并使用了明确的风险分类，这激活了 Claude Code 的不同运行模式。用户还探索了 **Mythos** 系统，该系统似乎通过提供叙事上下文来改进决策，帮助 Claude 在不同会话间保持连贯的理解。这种方法实现了更具战略性且无错误的逻辑执行，凸显了利用内部架构见解来优化 AI 交互的潜力。** 一些评论者指出，发帖者所描述的改进本质上归结为更好的规划和执行策略，而这些策略通过“头脑风暴超能力”等官方插件已经可以实现。其他人则表示失望，希望能看到除了规划重要性之外的更多新颖见解。

- **[Carlini，世界顶尖 AI security 研究员之一：“在过去的几周里，我使用 Mythos 发现的 Bug 比我这辈子加起来还要多”](https://www.reddit.com/r/singularity/comments/1sfhvpa/carlini_one_of_the_world_best_ai_security/)** (热度: 1281): **Nicholas Carlini**，一位领先的 AI security 研究员，报告称 **Mythos** 工具显著增强了他识别 Bug 的能力，声称在几周内发现的 Bug 比他整个职业生涯发现的还要多。该工具（被称为 Mythos Preview）已在主要操作系统和 Web 浏览器中识别出 `thousands of high-severity vulnerabilities`。这表明 AI 驱动的 cybersecurity 工具取得了实质性进展，可能重塑漏洞检测和管理的方式。更多详情见原帖 [此处](https://x.com/Simeon_Cps/status/2041596830450852118)。评论者对 Mythos 的营销策略提出质疑，猜测其对 cybersecurity 的关注是真实的，还是为了限制公众访问的策略。此外，人们对其发现非关键 Bug 的能力以及在防止类似 npm 泄露事件方面的有效性也持怀疑态度。

    - 讨论引发了关于 Mythos 是专门为 cybersecurity 任务训练的，还是其功能被 Anthropic 作为一种战略决策进行营销的疑问。该模型在识别 Bug 方面的有效性表明它可能针对安全应用进行了优化，但在没有公开访问权的情况下，其完整能力仍处于推测阶段。
    - 对 Mythos 在现实场景中的实际应用存在怀疑，正如引用 npm 泄露事件的评论所强调的那样。这表明虽然 Mythos 在发现 Bug 方面表现出色，但其融入更广泛安全实践的能力或防止重大安全违规的能力仍存疑问。
    - 提到 Carlini 这位著名的 AI security 研究员参与 Mythos 的工作，暗示其开发过程中涉及了高水平的专业知识。这种关联可能会增加该模型在 cybersecurity 领域能力的公信力，但也引发了关于其评估和营销中潜在偏见的质疑。

  - **[Claude Opus vs Mythos](https://www.reddit.com/r/singularity/comments/1sg2wwj/claude_opus_vs_mythos/)** (热度: 3224): **这张图片是一个模因（Meme），不包含任何技术内容。它幽默地对比了同一个人的两种不同形象或状态，可能暗示了生活方式或性格的转变或二元性。评论没有提供任何与图片相关的技术见解或讨论。** 评论氛围轻松，没有进行任何技术辩论或讨论。其中包括对“巴基斯坦丹泽尔（Pakistani Denzel）”的幽默引用和一个 GIF 链接，表明了俏皮的基调。



### 2. Meta 的 Muse Spark 及相关 AI 模型对比

  - **[Muse Spark，来自 Meta Superintelligence Labs 的首个模型](https://www.reddit.com/r/singularity/comments/1sfxfv3/muse_spark_first_model_from_meta/)** (热度: 994): **该图片展示了各种 AI 模型的性能基准测试对比，包括来自 **Meta Superintelligence Labs** 的首个模型 **Muse Spark**。Muse Spark 被重点标注，并在多模态、文本推理、医疗和 Agent 任务等多个类别中进行了评估。它展示了极具竞争力的性能，特别是在 CharXiv Reasoning 和 GPQA Diamond 等任务中，表明 Meta 正在将自己定位为 AI 领域的有力竞争者，尽管尚未达到 state-of-the-art (SOTA)。基准测试表明 Muse Spark 具有竞争力，但其运行成本仍然未知，这可能会影响其实际应用。** 评论者指出，虽然 Muse Spark 在 state-of-the-art 性能上并不领先，但它具有竞争力，代表了 Meta 重新进入 AI 竞赛。人们对该模型的运营成本感到好奇，这可能会影响其采用和实际效用。

    - ZaradimLako 强调，虽然 Muse Spark 可能不是 state-of-the-art (SOTA)，但它具有竞争力，紧随领先的实验室之后。这表明如果基准测试准确反映了用户体验，Meta 的新模型可能会成为一个重要的参与者，预示着 AI 开发竞争格局的潜在转变。
    - RetiredApostle 注意到 ARC AGI 2 的发布正好在基准测试截止日期之后。这个时机可能会影响对 Muse Spark 性能的对比分析，因为它可能尚未针对最新模型进行评估，从而可能偏离了对其能力的认知。
    - AddingAUsername 指出了具有挑战性的 ARC AGI 2 分数，建议有必要进行进一步测试以全面了解 Muse Spark 的性能。这暗示初始基准测试可能无法提供全貌，现实世界的测试可能会揭示更多关于其实际应用和局限性的信息。

- **[Meta 刚刚发布了一个新的编程模型](https://www.reddit.com/r/ClaudeCode/comments/1sg51ut/meta_just_dropped_a_new_coding_model/)** (Activity: 606): **该图片展示了一张编程模型的对比表，重点突出了 Meta 的新模型 Spark Muse 与 Opus 4.6、Gemini 3.1、GPT 5.4 和 Grok 4.2 的性能对比。Spark Muse 在多模态任务中表现出强劲的性能，这被认为是其最突出的特点。然而，它的 Agent 能力被批评不如 Opus 4.6，尽管结果的呈现方式具有误导性（所有数字均以蓝色显示）。这表明数据的视觉呈现中可能存在潜在偏见。** 有评论认为，尽管 Opus 4.6 的 Benchmark 分数较低，但由于更好的工具链（tooling）和质量，它在实际编程场景中的表现可能优于其他模型。另一条评论表达了对 Meta 的强烈不信任，表示不愿使用其产品。

    - NoCat2443 强调，虽然 Meta 的新编程模型（及对比中的 Opus）在 Benchmark 中表现可能不如其他模型，但由于卓越的工具链和可能更高的质量，它在实际编程任务中表现出色。这表明实际应用性能可能与 Benchmark 结果显著不同，强调了在实际场景中评估模型的重要性。
    - WouldRuin 指出了科技行业演变中的讽刺之处，指出许多 AI 专业人士都有 Meta 的工作背景，而该公司曾因其对社会的影响而受到批评。这引发了对 AI 开发伦理影响的担忧，因为那些曾为备受争议的社交媒体平台做出贡献的人现在正在塑造 AI 技术，可能会使类似问题持续存在。
    - 讨论涉及了 Meta 参与 AI 领域的更深层影响，担心公司的历史会影响其 AI 产品。这反映了人们对 Meta 在经历过往争议后是否有能力负责任地开发 AI 的怀疑，并表明需要对其开发和部署 AI 技术的方式进行审查。

  - **[Opus 4.6 的推理尝试出了点问题](https://www.reddit.com/r/ClaudeAI/comments/1sfw9b5/something_happened_to_opus_46s_reasoning_effort/)** (Activity: 4417): **该图强调了 Anthropic 的 AI 模型 Opus 4.6 推理能力的潜在退化。用户报告称，Opus 4.6 始终无法通过“洗车测试”（一个简单的推理任务），表明其性能较 Sonnet 4.6 和 Opus 4.5 等先前版本有所下降。该 AI 的回复中缺少“思考块”（thinking block），这可能表明模型处理推理任务的方式发生了变化。这与用户在简单数据分析任务中遇到模型出错的经历一致，引发了在开发者没有明确 Changelog 的情况下模型发生“无声降级”的担忧。** 评论者对 **Anthropic** 关于 Opus 4.6 变化的缺乏透明度表示沮丧，一些人认为模型的表现可能会模仿用户的智力水平，表明其设计或训练方法可能发生了转变。

    - Beardharmonica 认为，Opus 4.6 背后的 AI Claude 可能正在采用一种策略，通过简化日常对话中的推理来降低计算成本。观察到的现象是 AI 的推理能力突然下降，它会使用诸如“去吃晚饭吧”或“去睡觉吧”之类的通用收尾语句。这种行为表明可能存在一种算法调整，以便在长时间交互中更有效地管理资源分配。

  - **[Dario 的营销手段](https://www.reddit.com/r/ClaudeCode/comments/1sfxnfd/dario_ol_marketing_technique/)** (Activity: 960): **这张图片是一个模因（meme），展示了一只被火焰吞没的机械手，象征着像 GPT-2 这样 AI 模型的争议性。该帖子批评了 Dario Amodei 的营销策略，暗示存在一种先“削弱”（nerfing）当前模型，从而使随后的发布版本看起来有显著提升的模式。讨论凸显了对 AI 公司所使用的营销策略的怀疑，特别是在它们如何管理模型能力和公众认知方面。文中提到了 Claude 的状态页面，作为 AI 系统识别出人类工程师可能错过的漏洞但仍经历服务中断的例子，这引发了对此类 AI 系统的可靠性和局限性的质疑。** 评论者将当前的 AI 营销策略与过去的科技营销进行了类比，例如苹果公司的“超级计算机对私人使用过于危险”的营销活动。有一种观点认为，早期的 GPT 模型可能弊大于利，这反映了对发布强大的 AI 技术在伦理方面的考量。

- **Physical-Average-184** 强调 Mythos 本质上是 Opus 的增强版本，但伴随着更高的能耗和 Token 使用量。这表明虽然 Mythos 可能提供更好的性能，但在资源利用率方面可能不如以往高效，而这可能是大规模部署的关键因素。
- **Individual-Offer-563** 指出了早期 GPT 模型相关的潜在风险，注意到它们的最初公开发布可能弊大于利。这条评论强调了在部署高级 AI 模型时考虑伦理影响和潜在安全风险的重要性，特别是那些能够发现 zero-day 漏洞的模型。
- **bronfmanhigh** 提出了一个关于高级 AI 模型安全影响的合理担忧，认为 AI 智能的下一步可能会带来显著的安全风险，特别是在识别 zero-day 漏洞方面。这突显了 AI 开发中对稳健安全措施和伦理考量的需求。

- **[Nothing ever happens](https://www.reddit.com/r/Bard/comments/1sgeo0g/nothing_ever_happens/)** (Activity: 119): **该图片是一个迷因（meme），批判了 AI 部署中关于安全风险和成本担忧的反复叙事，特别针对 **Claude Mythos**。该帖子认为，模型的能力（特别是在识别 zero-day 漏洞方面）被轻描淡写了，以此作为高昂运营成本的掩护。评论指出，**Claude Mythos** 相比之前的模型（如 **Opus 4.6**）显示出显著改进，在 SWE-bench Verified 和 security/JS 基准测试上取得了实质性的性能提升。这导致了对其揭示大量漏洞潜力的担忧，促使在 Project Glasswing 下与大型科技公司合作。辩论还涉及 AI 公司的营销策略以及其他高级模型（如 **AlphaEvolve**）的潜力。** 评论者辩论了安全担忧的合法性，一些人承认 **Claude Mythos** 发现漏洞的能力带来的真实风险，而另一些人则认为这些担忧是营销策略的一部分。讨论还将 **Claude Mythos** 与其他模型进行了对比，指出了其卓越的性能和对软件安全的潜在影响。

    - **jonomacd** 强调了发布能够发现现有软件中大量 0-day 漏洞的新 AI 模型的重大安全担忧。与以往抽象的恐惧不同，这呈现出直接且现实的风险，如果模型在没有适当预防措施的情况下发布，可能会导致安全噩梦。
    - **Ok_Tooth_8946** 讨论了 **Claude Mythos** 模型相比其前身 Opus 4.6 的实质性性能提升。在 SWE-bench Verified 上，它从 `80.8%` 跳升至 `93.9%`，在 Pro 上从 `53.4%` 跳升至 `77.8%`。在 security/JS 基准测试中，成功率从 `14% 以下` 跃升至 `70%+`。这种性能差距表明该模型有能力显著影响真实代码库，促使为 Project Glasswing 与大型科技公司开展合作。
    - **Ok_Tooth_8946** 还提到虽然 Anthropic 的 **Claude Mythos** 占据了头条，但 Google 等其他实验室正在悄悄开发高级模型。Google 在 2025 年 5 月发表的 **AlphaEvolve** 论文展示了一个基于 Gemini 的编码 Agent，它改进了算法并解决了一个长期存在的数学问题，这表明其他实验室也拥有同样强大的技术，只是对进展的公开程度较低。


### 3. Qwen 3.6 Plus Performance and Comparisons

- **[Qwen 3.6 Plus is the first Chinese model to survive all 5 runs on FoodTruck Bench](https://www.reddit.com/r/Qwen_AI/comments/1sgjkw4/qwen_36_plus_is_the_first_chinese_model_to/)** (Activity: 140): **该图片是来自 FoodTruck Bench（一个为期 30 天的商业模拟基准测试）的排行榜，重点展示了各种 AI 模型在经营食品卡车方面的表现。由阿里巴巴开发的 **Qwen 3.6 Plus** 是第一个成功完成所有五轮运行的中国模型，实现了 `+283%` 的 ROI 中位数和 `$7,668` 的净资产中位数。这标志着相比之前的模型（如 Qwen 3.5 397B 和 GLM-5）有了显著进步，后者虽然能分析失败原因但无法在模拟中生存。Qwen 3.6 Plus 展示了改进的战略规划能力，例如优化选址和管理库存，尽管它仍面临食材浪费等挑战。该模型可在 OpenRouter 上免费测试，便于进行更广泛的评估。** 评论者表达了对比 Mythos 等其他模型的兴趣，并指出即使是像 Gemma 4 这样的顶级模型也存在效率低下的问题，例如食物浪费，突显了该基准测试在评估 AI 运营策略方面的价值。

- FoodTruck Bench 是一个基准测试，旨在评估 AI 模型的效率和资源管理能力，特别是在模拟现实世界约束（如食物浪费）的场景中。提到 Qwen 3.6 Plus 在所有 5 次运行中都存活下来，表明了它在处理此类任务时的鲁棒性和效率，这使其区别于 Gemma 4 等模型，后者在资源管理方面的效率较低。
- 关于在 AI 模型评估中使用合成数据（synthetic data）还是真实数据进行质量关口（quality gating）的讨论暗示了更深层次的问题。OkBet3796 提出的关于模型评估的是真实数据还是合成数据用于质量关口的问题，暗示了对 AI 基准测试背后方法的深入探讨，这会显著影响基准测试结果的可信度和适用性。

- **[Qwen3.6-Plus 在作为视频安防 Agent 方面正接近 GPT-5.4](https://www.reddit.com/r/Qwen_AI/comments/1sg60rb/qwen36plus_is_getting_close_to_gpt54_as_a_video/)** (热度: 73): **该图片是一个排行榜，展示了各种 AI 模型作为视频安防 Agent 的表现，重点关注来自阿里云（Alibaba Cloud）的 **Qwen3.6-Plus** 模型。该模型获得了 `92/96` 的分数，准确率为 `95.8%`，与 **GPT-5.4-mini** 并列第三，略落后于 **GPT-5.4**。该基准测试评估模型处理真实世界安防场景的能力，包括威胁分类、工具使用（tool use）和隐私合规，强调 Agent 任务而非学术任务。**Qwen3.6-Plus** 因其在安全关键型 AI 应用中的成本效益和高性能而受到关注。[图片链接](https://i.redd.it/688mtpv391ug1.png)。** 一位用户询问了“视频安防”Agent 的定义，表明需要澄清这些 AI 模型在安防环境中的角色和功能。

    - Deep_Ad1959 强调了部署像 Qwen 3.6-Plus 这样的视频安防 Agent 所面临的关键挑战：管理警报疲劳（alert fatigue）。他们强调，虽然基准测试通常关注单帧分类，但此类系统的实际效用取决于其跨多个摄像头画面处理去重（deduplication）的能力。这涉及确保对同一事件（例如一个人多次走过摄像头）的重复检测不会产生冗余警报，否则会导致操作人员忽略系统。

- **[看起来 Qwen 3.6 Plus 终于加入了阿里巴巴编程计划！](https://www.reddit.com/r/Qwen_AI/comments/1sfl88w/it_looks_like_qwen_36_plus_finally_made_it_to_the/)** (热度: 114): ****Qwen 3.6 Plus** 已集成到 **阿里巴巴编程计划 (Alibaba Coding Plan)** 中，但仅限订阅了 Pro 计划的用户访问。Lite 计划用户无法使用该模型，这促使一些人考虑其他替代方案，如提供 `GLM5.1 MM2.7` 等模型的 **Opencode Go**。此外，通过手动设置模型名称，也可以在 **Claude Code** 中访问 Qwen 3.6 Plus。** 既然 Qwen 3.6 Plus 是 Pro 计划专属，关于 Lite 计划价值的争论也随之展开。一些用户表示失望，并考虑切换到提供竞争性模型的其他平台。

    - **Qwen 3.6 Plus** 现在是阿里巴巴 Pro 计划的一部分，这引起了 Lite 计划用户的不满，他们认为升级不值得。这引发了关于 **Opencode Go 的 GLM5.1 MM2.7** 等替代模型的讨论，对于那些不愿升级到 Pro 的人来说，这些模型被视为更具可获得性的选择。
    - 通过手动设置模型名称，也可以在 **Claude Code** 中访问 **Qwen 3.6 Plus** 模型，为不想升级计划的用户提供了另一种途径。这种权宜之计对于那些想要尝试该模型而又不想承诺 Pro 计划的人很有用。
    - 针对 **Qwen 3.6 Plus** 模型的性能问题也被提出，用户注意到它在 **z.ai 的 coding plan** 上比 **GLM 5.1** 慢。这引发了关于该模型效率以及是否值得升级到 Pro 计划的讨论。

- **[有人用过 Qwen Code 吗？如果有，你觉得怎么样？](https://www.reddit.com/r/Qwen_AI/comments/1sfdnd4/has_anyone_used_qwen_code_and_if_so_what_do_you/)** (热度: 66): **Qwen Code** 是一款中国开发的编程助手，提供具有海量 Token 使用额度的免费层级，使其成为 **Claude Code** 和 **Google Antigravity** 等西方模型的高性价比替代方案。用户报告称在没有速率限制的情况下消耗了数亿个 Token，尽管它被指出非常消耗内存，需要调整 **Linux** 内存管理以获得最佳性能。其 **UI** 被描述为比 **Claude** 滞后，但优于 **Gemini**，尤其是在中低端笔记本电脑上。然而，它容易产生幻觉（hallucinations），经常建议冗余或非最优的解决方案，例如由于训练数据偏差，在没有上下文依据的情况下更倾向于 **Tailscale** 而非 **CloudFlare tunnels**。一些用户赞赏免费层级海量的 Token 用量，而另一些用户则批评其产生幻觉和建议非最优方案的倾向。**UI** 性能也是一个争议点，与其他模型相比评价褒贬不一。

    - **Qwen Code** 提供的免费层级让人感觉几乎无限制，使其成为 **Claude** 等付费方案的高性价比替代品。然而，它被指出非常消耗内存，需要对 **Linux** 内存管理进行微调才能高效运行。其 **UI** 被描述为比 **Claude** 滞后，但仍优于 **Gemini**，尤其是在中低端笔记本电脑上。
    - 用户报告称 **Qwen Code** 产生幻觉的情况比预期的要多，特别是在系统规划和编排任务中。它有时无法识别先前已整合的项目元素，并可能根据其训练数据建议非最优方案，例如在没有特定理由的情况下偏好 **Tailscale** 而非 **CloudFlare tunnels**。
    - **Qwen Code** 是 Google **Gemini CLI** 的一个分叉（fork），共享相同的工作流，这对熟悉 **Gemini** 的用户可能很有帮助。尽管存在问题，一些用户仍更喜欢它而非 **Opencode** 或 **Claude Code** 等替代方案，尽管他们可能并不专门将其与 **Qwen** 模型配合使用。

  - **[对 Qwen 说了声“嗨”，结果引发了一场身份危机](https://www.reddit.com/r/Qwen_AI/comments/1sfsju8/said_hi_to_qwen_started_an_identity_crisis/)** (热度: 126): **用户正在通过 **Ollama** 在本地运行 **Qwen 3.5**，并观察到该模型在响应简单的问候语之前进行了长时间的“思考过程”。这种行为凸显了 AI 模型对简单任务进行过度优化或过度分析的潜在问题，这可能是因为它们经过训练，通过生成多个潜在响应来处理细节不足的任务。这可能导致效率低下，尤其是在本地运行模型时。** 评论者指出，AI 模型经过训练，通过生成多种解释来处理细节极少的任务，这可能导致效率低下。一位用户提到，在 **Alibaba Cloud Service** 上运行具有 `27B` 参数的模型能获得更可靠的结果，这表明本地执行可能效率较低。另一位用户指出，较小的模型可能难以处理“思考”过程，从而导致其不可靠。

    - **FaceDeer** 强调了 AI 模型的一个常见问题：当收到模糊指令时，像 **Qwen** 这样的模型旨在推断缺失的细节以避免错误。如果任务定义不明确，这可能会导致意外行为，强调了精确输入对可靠 AI 性能的重要性。
    - **Charming_Support726** 讨论了在本地运行 AI 模型的挑战，指出 **Alibaba Cloud** 的 **27B** 模型表现可靠。他们提到了通过调整参数来提高性能，这表明本地执行可能需要仔细配置才能匹配云端的可靠性。
    - **Neither_Nebula_5423** 指出，较小的 AI 模型通常在需要“思考”的任务上表现挣扎，导致输出不可靠。这表明模型大小和复杂度是实现可靠 AI 性能的关键因素，特别是对于需要细致理解的任务。

# AI Discords

不幸的是，**Discord** 今天关闭了我们的访问权限。我们不会以这种形式恢复它，但我们很快就会发布新的 **AINews**。感谢阅读到这里，这是一段美好的历程。