---
companies:
- anthropic
- google
- zhipu
date: '2026-03-27T05:44:39.731046Z'
description: '据报道，**Anthropic** 正在推出一个名为 **Capybara** 的新 AI 模型级别。该模型比 **Claude Opus
  4.6** 规模更大、更智能，在编程、学术推理和网络安全方面表现出更强的性能。据推测，该模型的参数规模约为 **10 万亿**，而 **谷歌** 可能会资助 Anthropic
  的数据中心扩建。


  与此同时，**智谱**发布了 **GLM-5.1**，提升了开源代码模型的水平，并缩小了与闭源模型之间的差距。本地推理的经济性也在不断改善，其代表性进展是通过
  **TurboQuant vLLM** 等量化技术实现了 **Qwen 3.5 14B**、**Qwen 27B** 和 **Qwen3.5-35B** 模型的高效部署。然而，TurboQuant
  的基准测试主张遭到了研究人员的批评。


  总的来看，当前的 AI 格局呈现出激进的规模扩张、本地模型部署普及以及智能体（Agent）产品日益受到青睐的态势。'
id: MjAyNS0x
models:
- claude-opus-4.6
- capybara
- glm-5.1
- qwen-3.5-14b
- qwen-27b
- qwen3.5-35b
people:
- scaling01
- yuchenj_uw
- kimmonismus
- m1astra
- dejavucoder
- iscienceluvr
- gaoj0017
title: 今天没发生什么事。
topics:
- model-scaling
- coding
- academic-reasoning
- cybersecurity
- quantization
- local-inference
- model-benchmarking
- inference-optimization
- model-performance
- agent-products
---

**平静的一天。**

> 2026年3月26日至3/27日的 AI 新闻。我们检查了 12 个 subreddits、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有新增 Discord 信息。[AINews 网站](https://news.smol.ai/) 允许您搜索所有历史期号。提醒一下，[AINews 现在是 Latent Space 的一个板块](https://www.latent.space/p/2026)。您可以[选择加入/退出](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)邮件发送频率！

---

# AI Twitter 回顾


**Anthropic 泄露的 “Mythos” 系统与新的 Capybara 层级**

- **Fortune 证实 Anthropic 存在高于 Opus 的更高层级**：[@M1Astra](https://x.com/M1Astra/status/2037377109472018444) 保存了一篇已被撤下的 “Claude Mythos” 文章，随后多篇帖子引用 Fortune 的报道称，Anthropic 正在引入 **Capybara**，它被描述为 **Opus 之上**的新层级，比 **Claude Opus 4.6** “更大且更智能”。根据 [@scaling01](https://x.com/scaling01/status/2037379145806524655)、[@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2037387996694200509) 和 [@kimmonismus](https://x.com/kimmonismus/status/2037463638261305752) 总结的报告，Capybara 在 **coding（编程）、学术推理和网络安全**方面的得分大幅提高，但由于成本和安全顾虑，其推出受到了限制。
- **算力强度是核心主题**：几位发帖者推断 Anthropic 正在全力投入规模化（scale），基于 Dario 之前的言论，人们猜测其模型可能属于 **~10T 参数**级别，尽管这在评论之外尚未得到证实；参见 [@scaling01](https://x.com/scaling01/status/2037384912743923969) 和 [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2037391159115563214)。另外，[@FirstSquawk](https://x.com/FirstSquawk/status/2037586926375743904) 转述的《金融时报》报道称，**Google 即将为 Anthropic 的数据中心提供资金**，这进一步强化了前沿竞争正日益受到电力和 capex（资本支出）而非仅仅是算法限制的观点。
- **生产环境中的基础设施压力显而易见**：泄露事件发生时，Anthropic 的可用性正处于艰难的一天，[@dejavucoder](https://x.com/dejavucoder/status/2037439287873159641)、[@iScienceLuvr](https://x.com/iScienceLuvr/status/2037487244634972471) 等人广泛抱怨 **529 错误/高错误率**。实际的结论是，Anthropic 似乎正在激进的规模化雄心与依然紧张的服务能力之间寻找平衡。

**开源编程模型、本地推理以及 GLM-5.1 的持续推进**

- **GLM-5.1 正在加大对闭源编程模型的压力**：智谱（Zhipu）通过 [@Zai_org](https://x.com/Zai_org/status/2037490078126084514) 宣布向所有编程计划用户开放 **GLM-5.1**，并在 [@Zai_org](https://x.com/Zai_org/status/2037506911013138851) 发布了 Agent 使用文档。社区反应将其视为另一个信号，表明中国的高端开源或半开源编程模型正在缩小差距：[@kimmonismus](https://x.com/kimmonismus/status/2037507667732709392)、[@XFreeze](https://x.com/XFreeze/status/2037695882301436412) 以及 Arena 的广泛排行榜分析 [@arena](https://x.com/arena/status/2037584085997216100) 都指出，开源与闭源之间的差距比一年前窄得多。
- **本地部署的经济性持续提升**：推文中的一个经常出现的主题是，本地模型现在对于许多工作流来说已经“足够好”了。例子包括 [@TheGeorgePu](https://x.com/TheGeorgePu/status/2037473248577782046) 将昂贵的 TTS 订阅换成了本地的 **Qwen 3.5 14B** 配置，[@LottoLabs](https://x.com/LottoLabs/status/2037557925015949676) 报告了使用 Hermes Agent 运行 **Qwen 27B** 的强劲经济性，以及 [@0xSero](https://x.com/0xSero/status/2037560787565252666) 对 **Qwen3.5-35B** 进行了足够的压缩，使其能够在 **24GB VRAM** 中运行完整上下文，且平均性能损失仅约 **1%**。
- **量化和缓存技术仍是关键驱动力**：[@iotcoi](https://x.com/iotcoi/status/2037478891179135123) 发布了一个 **TurboQuant vLLM** 分支，具有融合的 Triton KV 写入路径和解码注意力（decode attention），目标是 **Qwen3.5-35B AWQ**、**1M 上下文**和 **4M KV cache**。与此同时，[@bnjmn_marie](https://x.com/bnjmn_marie/status/2037564190802563157) 在 **RTX Pro 6000/B200/H100** 上测试了 Qwen3.5 27B 的各种格式，**INT4** 成为 RTX Pro 6000 级硬件上的最佳推理选择。
- **但 TurboQuant 目前正面临争议**：这组推文中最激烈的研究争议来自 [@gaoj0017](https://x.com/gaoj0017/status/2037532673812443214) 及其更详细的澄清 [@gaoj0017](https://x.com/gaoj0017/status/2037552350924042488)，指责 Google 的 **ICLR 2026 TurboQuant** 论文在理论和基准测试中歪曲了 **RaBitQ**，包括不公平的 CPU-vs-GPU 比较。这并不会否定 TurboQuant 的工程价值，但确实让一些公开的对比结论受到了质疑。

**Agent 正在成为产品，而非演示**

- **Hermes Agent 正成为开源 Agent 的焦点**：在数据集中，最具持续性产品势头的是 **Nous Research 的 Hermes Agent**。[@NousResearch](https://x.com/NousResearch/status/2037654827929338324) 将 **Hugging Face** 集成为一流的推理提供商，提供 **28 个精选模型**及更多模型的访问权限；同时 [@ClementDelangue](https://x.com/ClementDelangue/status/2037634211973140898) 将此视为迈向具有记忆、持久机器访问和模型选择的开源 Agent 的一步。来自 [@fancylancer3991](https://x.com/fancylancer3991/status/2037579517389144399)、[@PolackJack](https://x.com/PolackJack/status/2037661357785690584) 和 [@alexcovo_eth](https://x.com/alexcovo_eth/status/2037589212648665273) 的用户报告强调，相比 OpenClaw 等重度依赖浏览器自动化的方案，其摩擦力更小且持久性更好。
- **Agent 基础设施正围绕 Trace、Eval 和可调试性趋于成熟**：Hugging Face 的 [@ClementDelangue](https://x.com/ClementDelangue/status/2037530125638455610) 呼吁建立 **开源 Agent Trace 数据集**，随后 [@yueqi_song](https://x.com/yueqi_song/status/2037614951230296230) 指出了 **Agent Data Protocol**。LangChain 推导出一系列面向生产的材料：**Agent 评估就绪清单** [@LangChain](https://x.com/LangChain/status/2037590936234959355)、**Deep Agents** IDE 风格的 UI 指南 [@LangChain_JS](https://x.com/LangChain_JS/status/2037560951445266891)，以及用于 Prompt 晋升/回滚的 **LangSmith Prompt Hub Environments** [@LangChain](https://x.com/LangChain/status/2037666098561032421)。发展方向很明确：技术栈正从“带有工具的聊天机器人”转向 Agent 的软件生命周期原语。
- **面向 Agent 的 Benchmark 开始反映实际工作负载**：Artificial Analysis 通过 [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2037562417836929315) 推出了 **AA-AgentPerf**，专注于 **真实的 Coding-Agent 轨迹**、**100K+ 序列长度**，以及以 **每个加速器/每千瓦/每美元/每个机架的并发用户数** 表示的吞吐量。这是一种比合成 Token Benchmark 更具部署相关性的抽象，对于比较 Agent 重度服务加速器系统的团队非常有用。

**Coding Agent、Codex 插件和多 Agent 软件工作流**

- **OpenAI 的 Codex 生态系统正转向工作区原生自动化**：OpenAI 开发者通过 [@OpenAIDevs](https://x.com/OpenAIDevs/status/2037604273434018259) 展示了 **Codex 插件**和用例库，而 Box 发布了一个用于在 Box 内容上自动化工作流的 Codex 插件 [@Box](https://x.com/Box/status/2037563341431058497)。来自 [@theo](https://x.com/theo/status/2037383187849183457)、[@nickbaumann_](https://x.com/nickbaumann_/status/2037395162641686813) 和 [@reach_vb](https://x.com/reach_vb/status/2037614060452106437) 的用户情绪表明，重心正从 Prompt/Response 转向 **持久化工作区、Issue 系统、终端、PR 流程和插件**。
- **胜出的 UX 模式日益趋向于“软件舰队管理”**：[@VibeMarketer_](https://x.com/VibeMarketer_/status/2037521519736463782) 很好地捕捉到了这一新兴模式：看板式卡片、隔离的工作树（Worktrees）、Agent 拥有的任务以及基于 Diff 的审查。相关工具包括来自 [@ctatedev](https://x.com/ctatedev/status/2037599050112160165) 用于实时浏览器会话调试的新 **Agent 浏览器仪表板**，以及 [@JTLonsdale](https://x.com/JTLonsdale/status/2037555800193851727) 和 [@cognition](https://x.com/cognition/status/2037649026951303668) 等对 Cognition/Devin 相关多 Agent SWE 系统的广泛热情。
- **Composer 2 和长跨度编码评估正在提高标准**：CursorBench 的讨论在这里大多是间接的，但 [@cwolferesearch](https://x.com/cwolferesearch/status/2037726856699420987) 指出了该基准测试的优势：**真实的编码会话**、**描述不充分的 Prompt**、更广泛的质量维度，以及每个任务中位数 **181 行的代码变更**。这比静态的玩具任务是更健康的 Benchmark 设计，并且符合向长跨度 (Long-horizon) Agent 评估转变的大趋势。

**研究与系统：世界模型、机器人、语音和多模态基础设施**

- **Meta 发布了实用的 SAM 3.1 加速方案**: [@AIatMeta](https://x.com/AIatMeta/status/2037582117375553924) 发布了 **SAM 3.1**，这是针对 SAM 3 的一个即插即用更新，引入了 **object multiplexing**，允许在 **单次前向传递 (forward pass) 中处理多达 16 个对象**。Meta 表示，在处理中等对象工作负载时，这使 **单台 H100 上的视频吞吐量大致翻倍，从 16 FPS 提升至 32 FPS**，这对于普及视频分割流水线 (video segmentation pipelines) 具有重要意义。
- **世界模型 (World models) 和机器人领域都有显著的开源发布**: [@LiorOnAI](https://x.com/LiorOnAI/status/2037484990779339064) 重点介绍了 LeCun 的 **LeWorldModel** 论文/仓库，这是一个小型开源世界模型，旨在通过 **SIGReg** 从数学上消除表征崩溃 (representational collapse) 的可能性，声称 **规划速度快 48 倍** 且 **Token 使用量减少约 200 倍**。在机器人数据方面，[@UnitreeRobotics](https://x.com/UnitreeRobotics/status/2037440578275946551) 开源了 **UnifoLM-WBT-Dataset**，这是一个用于滚动更新的真实人形全身遥操作 (whole-body teleoperation) 数据集。
- **语音/开源音频依然是最健康的开源类别之一**: Cohere 新推出的 **2B Apache-2.0 Transcribe** 模型受到了 [@victormustar](https://x.com/victormustar/status/2037572662659104976) 的高度赞赏，[@vanstriendaniel](https://x.com/vanstriendaniel/status/2037548103272632497) 报告了吞吐量测试结果：在 A100 上仅需 **12 分钟** 即可转录 **33 小时** 的音频。Mistral 的 **Voxtral TTS** 论文引起了 [@qtnx_](https://x.com/qtnx_/status/2037553397423902846) 的关注，[@sophiamyang](https://x.com/sophiamyang/status/2037523809914241069) 和 [@nickfrosst](https://x.com/nickfrosst/status/2037680223445975131#m) 也展示了浏览器/本地 Demo。
- **开源机器人技术栈的可复现性也在不断提高**: AI2 发布了 **MolmoBot**，这是一个完全在模拟环境中训练的开源机器人操纵套件，其 **代码、训练数据、生成流水线和评估方案** 已通过 [@allen_ai](https://x.com/allen_ai/status/2037590611990094259) 发布。这与 Unitree 数据集相辅相成，标志着顶尖实验室之外的可复制机器人研究正持续取得进展。

**热门推文 (按互动量排名)**

- **Anthropic/Capybara 泄露**: [@Yuchenj_UW 关于 Capybara 的推文](https://x.com/Yuchenj_UW/status/2037387996694200509) 是互动率最高的技术条目，总结了高于 Opus 的新层级及其据称的 Benchmark 提升。
- **Paul Conyngham 使用 AI 辅助治疗爱犬癌症**: [@sama](https://x.com/sama/status/2037396826060673188) 分享了一个使用 ChatGPT 及相关工具协助设计爱犬癌症 **mRNA 疫苗方案** 的故事，这成为了关于 AI 赋能个性化医疗的热门讨论点。
- **对 TurboQuant 的批评**: [@gaoj0017](https://x.com/gaoj0017/status/2037532673812443214) 针对一篇论文方法论的争议引发了异常高的关注，这可能是因为它挑战了一篇被大肆宣传的系统论文。
- **GLM-5.1 发布**: [@Zai_org](https://x.com/Zai_org/status/2037490078126084514) 宣布 GLM-5.1 广泛可用，反响强烈，强化了人们对开源编程模型的持续兴趣。
- **Agent 开源基础设施**: [@OpenAIDevs](https://x.com/OpenAIDevs/status/2037604273434018259) 关于 Codex 插件的消息，以及 [@NousResearch](https://x.com/NousResearch/status/2037654827929338324) 关于 Hugging Face 集成到 Hermes Agent 的消息，是具有广泛开发者关联的最明确的产品/基础设施发布。

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. TurboQuant 与 RotorQuant 创新

  - **[Google TurboQuant 在 MacAir 上本地运行 Qwen](https://www.reddit.com/r/LocalLLaMA/comments/1s5kdu0/google_turboquant_running_qwen_locally_on_macair/)** (热度: 433): **该贴讨论了一个实验，将 **Google 的 TurboQuant 压缩方法** 应用于 `llama.cpp`，使得在标准的 MacBook Air (M4, 16 GB) 上能以 `20000 tokens` 的上下文运行 **Qwen 3.5–9B**。这在以前的此类硬件上是无法实现的，凸显了 TurboQuant 在不依赖云端 API 的情况下实现大模型本地运行的潜力。实验表明，即便是 MacBook Air 或 Mac Mini 这样的入门级设备也可以处理长上下文，尽管速度上存在一些限制。文中提到了开源应用 [atomic.chat](http://atomic.chat/)，作为本地运行这些模型的资源。** 一位评论者指出，在基础款 MacBook Air 上不使用 swap 就能处理 `20K context` 是一个了不起的成就，暗示了以前依赖云端 API 的本地使用场景现在有了可能性。另一位评论者询问了 TurboQuant 集成进 `llama.cpp` 的情况，表现出对更广泛易用性的兴趣。

- **Tatrions** 强调了得益于 TurboQuant，在仅有 16GB RAM 的基础版 MacBook Air 上运行 20K 上下文模型且不产生交换分区（swapping）的强大能力。这表明许多以前依赖云端 API 的应用现在可以在本地执行，不过人们也对其在这种压缩水平下与同模型的标准 Q4 相比是否存在质量下降感到好奇。
- **M5_Maxxx** 对 TurboQuant 的实现进行了详细审计，揭示其是 [Jan.ai](http://Jan.ai) 的一个微改版本。主要变化包括重命名、UI 调整和自定义的 `llama.cpp` 后端分支，但没有新的推理引擎或模型架构支持。96 个提交（commits）大多涉及 CI/构建流水线的更改，这表明除了原始 Jan.ai 的功能外，创新有限。
- **AppealThink1733** 询问了 TurboQuant 集成到 `llama.cpp` 的情况，表示有兴趣了解这项技术是否已被这个流行的开源项目支持，这可能有助于更广泛的采用和实验。

- **[跳过 90% 的 KV 反量化工作 → 32K 下解码速度提升 +22.8% (llama.cpp, TurboQuant)](https://www.reddit.com/r/LocalLLaMA/comments/1s56g07/skipping_90_of_kv_dequant_work_228_decode_at_32k/)** (热度: 744): **该帖子讨论了 `llama.cpp` 中用于 KV cache 压缩的 `TurboQuant` 实现的一项优化，通过跳过注意力权重（attention weights）微不足道的位置的反量化过程，显著提高了解码性能。这种方法利用了注意力稀疏性（attention sparsity），在 `M5 Max` 上、`32K` 上下文长度下使解码速度提升了 `+22.8%`，且不影响困惑度（PPL）。该方法仅涉及对内核（kernel）约三行代码的简单修改，绕过了对 SIMD 技巧或融合内核（fused kernels）等复杂优化的需求。结果在不同硬件上保持一致，包括 `M2 Pro`，其性能相对于标准 `q8_0` KV cache 从 `~0.45x` 提升至 `~0.73x`。实现代码和基准测试可在 [GitHub](https://github.com/TheTom/turboquant_plus) 上获得，并附有详细的 [技术文档](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/sparse-v-dequant.md)。** 评论者称赞该解决方案的简单性和有效性，指出其创新地利用注意力稀疏性来跳过不必要的计算。人们对这种方法如何扩展到更长的上下文（如 `64K+`）感到好奇，并有兴趣将此优化集成到 `llama.cpp` 主线中。

    - Specialist_Sun_7819 强调了 llama.cpp 的 TurboQuant 中一项新颖的优化：对于那些对输出没有显著影响的 Token，跳过 90% 的 KV 反量化工作，从而在 `32K` 上下文长度下使解码速度提升 `+22.8%`。这种方法利用了长上下文中可预测的注意力稀疏性，通过极少的代码更改（具体来说只需在内核中修改三行）即可大幅节省计算量。该评论者对这种方法在更长上下文（如 `64K`）下的可扩展性感到好奇，以及稀疏比例是会继续增加还是趋于平稳。
    - sean_hash 将 TurboQuant 中的优化与 Flash Attention 中使用的技术进行了类比，指出缓存反量化输出而不是在每个解码步骤中重新计算是一种类似的策略。这种方法有效地减少了冗余计算，通过重用以前计算的值来增强性能，这是高性能计算中最小化不必要处理开销的常见优化手段。
    - Pentium95 表示有兴趣将此优化集成到 llama.cpp 主线中，表明希望这项技术得到更广泛的采用。这说明社区看到了这些性能改进的价值，并渴望看到它们在广泛使用的代码库中实现，从而可能在各种应用中带来更高效的模型和更快的推理时间。

- **[TurboQuant in Llama.cpp benchmarks](https://www.reddit.com/r/LocalLLaMA/comments/1s4bzo2/turboquant_in_llamacpp_benchmarks/)** (热度: 463): **该帖子讨论了在 `llama.cpp` 框架中实现 Google 的压缩技术 **TurboQuant**，特别是在使用 Metal 的 Apple Silicon 上。作者指出性能大幅下降，TPS 比 `f16` 低了 `50%`，表明其设置可能存在问题。他们还尝试在 CUDA 机器上运行内核，但输出效果很差，暗示其方法中存在错误。该技术被认为有助于在 VRAM 有限的消费级硬件上运行本地模型，可能允许在本地执行更复杂的任务。帖子提到了 **MLX** 和 **VLLM** 等相关项目的持续开发工作。**评论者建议检查 KLD 以评估该方法的价值，并表示有兴趣查看 pp2048 等性能指标，因为 pp64 的参考意义不大。另一位评论者建议尝试 RotorQuant 进行对比。

    - Velocita84 指出基准测试中缺失了 Kullback-Leibler Divergence (KLD)，这对于评估 TurboQuant 的有效性至关重要。KLD 是衡量一个概率分布如何偏离第二个预期的概率分布的指标，它的缺失可能意味着无法深入了解模型在 TurboQuant 压缩下的性能。
    - CornerLimits 建议使用 `pp64` 的基准测试对于评估性能并没有太大参考价值，并建议改用 `pp2048`。`pp` 指标指的是 Perplexity（困惑度），这是语言模型中的一种常见衡量标准，表示概率分布预测样本的好坏程度。更高的 `pp` 值可以提供更全面的模型性能视图。
    - DinoAmino 讨论了 TurboQuant 在数据压缩和准确性之间的权衡，指出虽然它允许在接近无损准确度的情况下实现更高的数据压缩，但它并不会提高准确性。他们强调大多数 LLM 在较长的上下文长度下都会经历准确性下降，这意味着 TurboQuant 的主要好处是能够支持更长的上下文而不会产生额外的准确性损失。

  - **[RotorQuant: 10-19x faster alternative to TurboQuant via Clifford rotors (44x fewer params)](https://www.reddit.com/r/LocalLLaMA/comments/1s44p77/rotorquant_1019x_faster_alternative_to_turboquant/)** (热度: 652): ****RotorQuant** 通过利用 Clifford Algebra 引入了一种全新的向量量化方法，在参数量减少 `44x` 的情况下，实现了比 **TurboQuant** 快 `10-19x` 的速度提升。该方法使用 Clifford rotors 替换了 `d×d` 随机正交矩阵，对于 `d=128`，将计算复杂度从 `16,384` FMAs 降低到大约 `100` FMAs。这导致其余弦相似度为 `0.990`（TurboQuant 为 `0.991`），表明性能几乎相同。该实现利用了融合的 CUDA 内核和 Metal shader，在 RTX PRO 4000 和 Apple M4 上的表现明显优于 cuBLAS matmul。权衡之处在于随机单位向量上的合成 MSE 较高，但通过 QJL 修正，实际模型的 Attention 保真度保持完好。[GitHub](https://github.com/scrya-com/rotorquant) [Paper](https://www.scrya.com/rotorquant/)** 一个关键争论点集中在 RotorQuant 和 TurboQuant 之间的理论差异上。虽然 TurboQuant 的全局随机旋转将能量分散到所有维度，但 RotorQuant 的 3D 块混合无法复制这一点，导致最大坐标幅值更高，且在低比特量化中 MSE 表现更差。然而，RotorQuant 在 KV cache 分布中的实际性能得到了认可，这表明它在实际模型中提供了一种很有价值的速度/质量权衡。

- Juan_Valadez 指出了 RotorQuant 与 TurboQuant 相比的一个关键理论局限，指出 TurboQuant 的全局随机旋转（Haar）有效地将能量分布到所有维度，从而优化了标量量化（scalar quantization）。相比之下，RotorQuant 在 3D 块（3D blocks）内的混合限制了其实现相同能量分布的能力，这可能会对低比特量化产生负面影响，特别是在 one-hot 等最坏情况的向量中。然而，RotorQuant 对于向量对抗性较低的 KV cache 分布在实际中可能仍然有用。
- Dany0 将 TurboQuant 与图形编程中使用的技术进行了类比，特别提到了 QuiP（一种应用于模型权重的类似方法）。尽管最初由于论文篇幅短及其展示方式而持怀疑态度，Dany0 承认了 RotorQuant 的潜力，将其对 Clifford rotors 的使用比作使用四元数（quaternions）而非欧拉角（Euler angles），通过将乘法减少为零来简化计算。
- sean_hash 评论了 Clifford algebras 在量化中的意外应用，将其视为几何代数（geometric algebra）向图形以外领域交叉渗透的一个例子。这突出了传统上与其他领域相关的数学概念的创新应用，表明了这些技术具有更广泛的适用性。

### 2. GLM-5.1 and Coding Model Comparisons

- **[Glm 5.1 is out](https://www.reddit.com/r/LocalLLaMA/comments/1s51id3/glm_51_is_out/)** (Activity: 1127): **图片宣布了 Z.ai 发布 GLM-5.1，突出了其在编程任务中与之前版本相比的性能提升。图片中的图表显示 GLM-5.1 在编程评估中得分为 `45.3`，超过了 GLM-5 的 `35.4`，但仍落后于得分为 `47.9` 的 Claude Opus 4.6。这表明 GLM-5.1 的能力有显著提升，这可能归功于其底层架构或训练数据的增强。** 评论者猜测 GLM-5.1 可能会发布 open weights，表示对更广泛可访问性的期待。还有关于 DS v4 发布延迟的讨论，暗示在特定硬件（如 Ascends）上进行训练可能存在挑战。

    - power97992 推测 DeepSpeed v4 可能延迟发布，认为可能存在与在 Ascend 硬件上训练相关的问题。这突出了针对不同硬件架构优化机器学习框架的挑战，这会影响发布时间表。
    - zb-mrx 注意到 GLM-5.1 的推广（rollout）过程有所改进，与之前并非所有人都能在首日使用的 GLM-5 形成对比。这表明开发者可能已经解决了之前的物流或资源相关问题（如 GPU 可用性），以确保更顺畅的发布。
    - jacek2023 提到了由于硬件限制（特别是 72GB VRAM 的限制）在本地运行 GLM 的局限性。这强调了运行先进模型对硬件要求的持续挑战，这对于许多无法使用高端 GPU 的用户来说是一个障碍。

### 3. Local LLM 硬件设置与比较

  - **[Dual DGX Sparks vs Mac Studio M3 Ultra 512GB: 在两者上本地运行 Qwen3.5 397B。这是我的发现。](https://www.reddit.com/r/LocalLLaMA/comments/1s4lmep/dual_dgx_sparks_vs_mac_studio_m3_ultra_512gb/)** (热度: 819): **该帖子比较了 **Mac Studio M3 Ultra 512GB** 和 **双 DGX Spark 配置** 在本地运行 **Qwen3.5 397B** 模型的性能。Mac Studio 利用 `MLX 6 bit quantization`，实现了 `30 到 40 tok/s` 的生成速度，内存带宽约为 `800 GB/s`，但 prefill 时间较慢，且需要自定义异步代理（async proxy）来进行 tool calls。相比之下，双 DGX Spark 配置使用 `INT4 AutoRound quantization`，实现了 `27 到 28 tok/s` 的速度，得益于 CUDA tensor cores，其 prefill 和 batch embedding 速度更快，但在设置复杂度、内存带宽（每个节点约 `273 GB/s`）和稳定性方面面临挑战。作者将这两种配置用于不同任务：Mac Studio 用于 inference，Sparks 用于 RAG 和 embedding，两者通过 Tailscale 进行通信。每套配置的成本约为 `$10K`，与每月 `$2K` 的 API 支出相比，盈亏平衡点为 10 个月。** 评论强调了 Mac Studio 512GB 的独特性，并批评了 Nvidia 对 DGX 的支持。此外，还有关于 Qwen3.5 397B 与 Claude 性能对比的讨论，指出虽然 Qwen3.5 不如 Claude 的 Opus 先进，但性能已经非常接近。

    - Repoman444 强调了 **Nvidia DGX** 系统的一个重大问题，指出 Nvidia 的支持服务欠佳。这可能会影响那些依赖及时有效的支持来解决故障并优化高性能计算任务（尤其是运行像 Qwen3.5 397B 这样的大型模型时）的用户。
    - sp4_dayz 讨论了 **Qwen3.5 397B** 与 **Claude** 和 **Opus** 的性能对比，认为虽然 Qwen3.5 尚未达到 Opus 的水平，但已经非常接近。这意味着熟悉 Claude 的用户可能会发现 Qwen3.5 略有不足，但在性能方面仍是一个强有力的竞争者。
    - Gringe8 就比较方法提出了一个技术点，询问评估是否包含了 prompt 处理速度。这表明 prompt 处理速度是评估像 Qwen3.5 397B 这样的 AI 模型性能的关键因素，特别是在比较 DGX 和 Mac Studio M3 Ultra 等不同硬件配置时。

  - **[如果你现在有约 10k 美元用于购买本地 LLM 硬件，你实际上会构建什么？](https://www.reddit.com/r/LocalLLM/comments/1s40wgj/if_you_had_10k_to_spend_on_local_llm_hardware/)** (热度: 201): **该帖子讨论了在 `~$10k` 预算下构建运行大语言模型 (LLMs) 的本地硬件配置。用户目标是运行至少 `30B` 参数的模型，理想情况下达到 `70B`，用于简单聊天之外的任务，如多步工作流和 tools，重点是隐私和避免 API 成本。主要的技术争论围绕 GPU 选择展开：**RTX 4090** 因其性能被考虑，而二手 **A6000/A40** GPU 则因其 VRAM 容量而受到关注。用户还考虑了具有统一内存（unified memory）的 **Mac Studio (M3 Ultra)**，质疑其相对于 CUDA 配置的实际表现。该帖子征求关于平衡 GPU、CPU、RAM 和存储投资的建议，以在不牺牲速度或可靠性的情况下获得最佳性能。** 评论者建议考虑 **RTX 6000 Blackwell** 或 **Mac Studio** 作为可行方案。一位评论者幽默地建议用这笔预算赚取利息来支付 LLM 订阅费用，强调了尽管用户偏好本地配置，但云解决方案仍具成本效益。

    - Blackdragon1400 强调了本地 LLM 硬件拥有至少 `256GB VRAM/Unified memory` 的重要性，认为少于这个容量都是不够的。他们推荐使用 `2x DGX Sparks`，它可以以大约 `40t/s` 的速度运行 `Qwen3.5-122b-Int4-Autoround`，突显了其相对于 SOTA 模型的效率。
    - MatthiasWM 提到了 Apple 可能会在 6 月份的开发者活动中发布 `M5 Ultra` 芯片。他们建议在对本地 LLM 硬件进行重大投资之前等待这次发布，因为新芯片可能会带来实质性的提升。
    - Blackdragon1400 还建议在处理 LLM 任务时优先考虑大容量 RAM，警告不要勉强接受那些仅仅为了“塞进”较小内存配置的量化模型（quantized models）。这强调了需要强大的硬件来有效处理苛刻的 LLM 工作负载。


## 非技术类 AI 版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

# AI Discords

不幸的是，Discord 今天关闭了我们的访问权限。我们不会再以这种形式恢复它，但我们很快就会发布全新的 AINews。感谢读到这里的每一位，这是一段美好的历程。