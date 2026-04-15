---
companies:
- google
- tencent
- google-deepmind
- openai
- hugging-face
- cursor
- langchain
date: '2026-04-06T05:44:39.731046Z'
description: '**谷歌 (Google)** 推出了 **Skills in Chrome**，通过 Gemini 提示词和预设技能库实现可重用的浏览器工作流，进一步提升了终端用户的“智能体化”（agentization）。**腾讯
  (Tencent)** 预热了 **HYWorld 2.0**，这是一个开源 3D 世界模型，能够从单张图像生成可编辑场景。**谷歌 DeepMind (Google
  DeepMind)** 发布了 **Gemini Robotics-ER 1.6**，提升了机器人的视觉与空间推理能力，其仪表读取成功率达到 93%。**OpenAI**
  扩展了其“受信访问”（Trusted Access）功能，推出了 **GPT-5.4-Cyber**，这是一款专为防御性安全工作流进行微调的模型。**Hugging
  Face** 在其 Hub 上推出了 **Kernels**，提供可实现 1.7 倍至 2.5 倍加速的 GPU 内核仓库。**Cursor** 展示了一个多智能体
  CUDA 优化系统，在 235 个问题中实现了 38% 的提速。**Hermes Agent** 技术栈更新至 v0.9.0 版本，增强了可靠性、内存管理和集成能力；同时，**LangChain**
  将 **deepagents 0.5** 推向具备多模态支持和提示词缓存功能、可部署的多租户异步系统。*“Hermes 的核心优势在于运行稳定性、可扩展性和可部署性。”*'
id: MjAyNS0x
models:
- gemini
- gemini-robotics-er-1.6
- gpt-5.4-cyber
- deepagents-0.5
people:
- clementdelangue
- dylantfwang
- antoinersx
- steveschoettler
- teknium
- aiqiang888
- sydneyrunkle
title: 今天没发生什么特别的事。
topics:
- agent-infrastructure
- cuda-optimization
- visual-reasoning
- spatial-reasoning
- gpu-kernels
- multi-agent-systems
- memory-management
- async-systems
- multimodality
- prompt-caching
- software-engineering
- robotics
---

**平静的一天。**

> 2026年4月3日至4月4日的 AI 新闻。我们检查了 12 个 Reddit 子版块、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，未查看其他 Discord。[AINews 网站](https://news.smol.ai/) 允许你搜索所有过往内容。提示：[AINews 现已成为 Latent Space 的一部分](https://www.latent.space/p/2026)。你可以[选择加入或退出](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)邮件订阅频率！

---

# AI Twitter 回顾

**热门推文（按参与度排序）**

- **Google 的 Chrome “Skills” 将 Prompt 转化为可重复使用的浏览器工作流**：Google 推出了 [**Skills in Chrome**](https://x.com/Google/status/2044106378655215625)，允许用户将 Gemini Prompt 保存为一键式操作，针对当前页面和选定的标签页运行。Google 还发布了 [预制 Skills 库](https://x.com/Google/status/2044106380882166040)，这使其不仅仅是 Prompt 历史记录：它实际上是浏览器内轻量级的最终用户 Agent 化（agentization）。
- **腾讯的 HYWorld 2.0 将世界模型定位为可编辑的 3D 场景生成器，而非视频模型**：在发布前，[@DylanTFWang](https://x.com/DylanTFWang/status/2043952886166761519) 预热了 **HYWorld 2.0**，这是一个**开源、引擎就绪的 3D 世界模型**，可以从单张图像生成可编辑的 3D 场景。
- **Google DeepMind 发布了 Gemini Robotics-ER 1.6**：由 [@GoogleDeepMind](https://x.com/GoogleDeepMind/status/2044069878781390929) 宣布的新模型改进了机器人领域的**视觉/空间推理**，增加了更安全的物理推理，并可在 **Gemini API / AI Studio** 中使用。后续帖子强调了 **93% 的仪器读取成功率** 以及对液体和重物等物理约束更好的处理能力。
- **OpenAI 为网络安全扩展了 Trusted Access，推出 GPT-5.4-Cyber**：OpenAI 表示 [GPT-5.4-Cyber](https://x.com/OpenAI/status/2044161906936791179) 是 GPT-5.4 针对防御性安全工作流的微调版本，通过其 Trusted Access 计划提供给更高级别的经过认证的防御者。
- **Hugging Face 在 Hub 上推出了 “Kernels”**：[@ClementDelangue](https://x.com/ClementDelangue/status/2044053580504584349) 宣布了一种全新的 **GPU Kernel 仓库类型**，带有与特定 GPU/PyTorch/OS 组合匹配的预编译产物，并声称比 PyTorch 基准测试实现了 **1.7x–2.5x 的加速**。
- **Cursor 介绍了与 NVIDIA 合作构建的多 Agent CUDA 优化系统**：[@cursor_ai](https://x.com/cursor_ai/status/2044136953239740909) 表示，其多 Agent 软件工程系统在 3 周内对 235 个 CUDA 问题实现了 **38% 的几何平均加速（geomean speedup）**，这是 Agent 被应用于系统优化而非仅仅是应用脚手架（app scaffolding）的一个具体实例。

**Agent 基础设施：Hermes、Deep Agents 和生产测试框架 (Production Harnesses)**

- **Hermes Agent 正在成为一个严肃的开放本地 Agent 栈，其核心差异化优势在于可靠性和记忆能力**：多篇帖子汇聚成同一个主题：用户正在从其他替代方案迁移到 **Hermes Agent**，因为它在处理长时间运行的任务时更加耐用。该项目发布了重大的 **v0.9.0** 更新，根据 [@AntoineRSX](https://x.com/AntoineRSX/status/2043884430901850271) 的介绍，包含了 **Web UI、模型切换、iMessage/微信集成、备份/恢复以及通过 tmux 支持 Android**；同时，腾讯强调了用于全天候云端托管及消息集成的 [Lighthouse 一键部署](https://x.com/TencentAI_News/status/2044007400282436006)。在记忆方面，来自 [@SteveSchoettler](https://x.com/SteveSchoettler/status/2043870709613768820) 的 **hermes-lcm v0.2.0** 增加了**无损上下文管理**，具有持久化消息存储、DAG 摘要以及扩展压缩上下文的工具。来自 [@Teknium](https://x.com/Teknium/status/2044190761609244986)、[@aiqiang888](https://x.com/aiqiang888/status/2043920187959992609) 等人的社区帖子进一步证实，Hermes 的关键优势不在于原始模型的 IQ，而在于**操作稳定性、可扩展性和可部署性**。
- **LangChain 正在将 “deep agents” 推向可部署、多租户、异步系统**：**deepagents 0.5** 版本增加了 [**异步子 Agent、多模态文件支持和 Prompt 缓存优化**](https://x.com/LangChain/status/2044086454230626733)。相关帖子强调 `deepagents deploy` 是 [托管型 Agent 托管服务的开放替代方案](https://x.com/LangChain/status/2044097913698091496)，根据 [@LangChain](https://x.com/LangChain/status/2044098386270310783) 和 [@sydneyrunkle](https://x.com/sydneyrunkle/status/2044099832319500484) 的消息，后续工作将围绕**作用域限定为用户/Agent/组织的记忆**以及**自定义认证/每用户线程隔离**展开。这里一个有趣的模式是从 “Agent 演示” 向**平台层关注点**的转变：租户、隔离、长时任务，以及像 Salesforce 和基于 Agent Protocol 的服务器等集成接口。
- **Harness 设计正在成为一流的工程课题**：多篇帖子认为，Agent 的性能至少有一半取决于其支架（scaffold）而非模型。[@Vtrivedy10](https://x.com/Vtrivedy10/status/2044130977526755636) 最清晰地阐述了**针对特定任务的开放 Harness** 优于意识形态之争（“轻量 vs 厚重”），而 [@kmeanskaran](https://x.com/kmeanskaran/status/2044010500816810427) 则强调了工作流设计、记忆切换和工具输出控制，而非盲目追求前沿模型（frontier-model）。这与 [@ClementDelangue](https://x.com/ClementDelangue/status/2044139560355901911) 的观点一致，他要求建立一个从**模型到其最佳编程/Agent Harness 的精选映射**，随着开源权重模型的多样化，这一点变得越来越必要。

**机器人、世界模型与 3D 生成**

- **Google 的 Gemini Robotics-ER 1.6 是具身推理（embodied reasoning）产品化的重要一步**：由 [@GoogleDeepMind](https://x.com/GoogleDeepMind/status/2044069878781390929) 发布的这一版本强调了更好的**视觉/空间理解**、工具使用和物理约束推理。后续报道指出，其**人体受伤风险检测能力提升了 10%**，支持读取复杂的模拟仪表，并已在 API 中可用；[@_philschmid](https://x.com/_philschmid/status/2044071114578509971) 强调了在**仪表读取任务中达到了 93% 的成功率**。这感觉不像是又一篇机器人基础模型的论文发表，而更像是一个**面向开发者的具身推理 API**。
- **世界模型正在从电影般的演示向可编辑的空间产物转变**：腾讯的 [HYWorld 2.0 预告](https://x.com/DylanTFWang/status/2043952886166761519) 明确将其与视频生成系统区分开来，将其输出定义为一个**真实的 3D 场景**，它是可编辑的且支持引擎调用。在 Web 端，来自 [@sparkjsdev](https://x.com/sparkjsdev/status/2044090505982816449) 的 **Spark 2.0** 发布了一个**用于 3D Gaussian splats 的可流式 LoD 系统**，目标是在移动端、Web 和 VR 上的 WebGL2 环境中支持 **1 亿+ splat 的世界**。这些进展共同表明，“AI 生成 3D” 的技术栈正在从内容生成走向**交互式渲染和下游应用**。
- **开放 3D 生成在拓扑、UV、骨骼绑定和动画就绪性方面取得进展**：[@DeemosTech](https://x.com/DeemosTech/status/2044067290908635418) 介绍了 **SATO**，一个用于**拓扑和 UV 生成**的自回归模型；而 [@yanpei_cao](https://x.com/yanpei_cao/status/2044094818872377720) 发布了 **AniGen**，它可以从一张图像生成 **3D 形状、骨架和蒙皮权重**。这些进展极具意义，因为生产级 3D 工作流中的瓶颈很少在于“能否生成网格”，而在于资产是否具有足够的结构化信息以便进行动画、贴图和编辑。

**模型、基准测试与专用系统**

- **32B 以下的开源模型现在在推理/Agent 任务上具有真正的竞争力，但有一些重要的限制**：[@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2043929874537296026) 指出，**Qwen3.5 27B (Reasoning)** 和 **Gemma 4 31B (Reasoning)** 在其 Intelligence Index 上达到了 **GPT-5 级别的分数**，同时能够适配单张 H100，在量化后甚至可以在 MacBook 上运行。细微的差别很重要：这些模型在 **Agent 性能和批判性推理**方面表现最强，但在**知识召回 / 幻觉规避**（AA-Omniscience）方面明显落后。对于从业者来说，这是一个有用的参考：本地/开源模型现在可能已经达到了许多编程 Agent 工作流的标准，但并非适用于所有对知识敏感的企业级任务。
- **Minimax 似乎正在放宽 M2.7 在自托管方面的商业限制**：[@RyanLeeMiniMax](https://x.com/RyanLeeMiniMax/status/2044132777877221515) 更新了许可证，允许个人在自己的服务器上运行该模型，用于编码、应用构建、Agent 和其他个人项目；在随后的补充中，他澄清“编码”可以包括[通过你构建的产品获利](https://x.com/RyanLeeMiniMax/status/2044145290773704898)。鉴于 [@Sentdex](https://x.com/Sentdex/status/2044108342147060067) 推动的 **M2.7 + Hermes CLI** 作为本地编码配置的兴趣日益浓厚，剩下的问题是该许可证在工作和团队使用方面的延伸程度。
- **专门的 Post-trained 模型在细分、高价值任务上继续优于通用模型**：Cognition 发布了 [**SWE-check**](https://x.com/cognition/status/2044174496312242544)，这是一个通过 Applied Compute 进行 RL 训练的 Bug 检测模型，据报道在内部分布内评估中达到了 Frontier 模型的性能，同时运行速度快了 **10 倍**。技术细节值得关注：**Reward linearization**（奖励线性化）用于对齐样本奖励与总体 F-beta，以及将能力学习与延迟优化分离的**两阶段 Post-training**。这是一个很好的例子，说明即使在强大的通用模型时代，定制化的 Post-training 仍然具有重要意义。

**开发者工具、推理与系统**

- **Hugging Face 的 Kernels 仓库类型可能成为底层性能工作的一个有用分发原语**：[Kernels 的发布](https://x.com/ClementDelangue/status/2044053580504584349)，加上来自 [@RisingSayak](https://x.com/RisingSayak/status/2043984021521346575) 和 [@mervenoyann](https://x.com/mervenoyann/status/2044080953648128073) 的支持帖子，为 Kernel 作者提供了一种类似于模型打包优化 GPU Kernels 的方式。其实际前景在于为性能关键型代码提供可复现性和可发现性，特别是如果与 LLM 辅助的优化工作流（如 [@ben_burtenshaw 的“从 Agent 推送 Kernels”设置](https://x.com/ben_burtenshaw/status/2044114277745807684)）相结合。
- **开源医疗和 OCR 工具继续向端侧和生产流水线迁移**：[@MaziyarPanahi](https://x.com/MaziyarPanahi/status/2044037968659103806) 发布了 **OpenMed 1.0.0**，这是一个基于 **Apache-2.0** 协议、由 MLX 支持的 Apple Silicon 软件包，包含 **8 种语言的 200 多个 PII 检测模型**，并支持 iOS/macOS。与此同时，[@vllm_project](https://x.com/vllm_project/status/2043964594679636260) 强调了 **Chandra-OCR-2 (5B)** 在 16 个并行作业下，每台 **L40S 每小时可处理约 60 篇论文**，这是文档 AI 吞吐量的一个有用参考点。
- **编程 Agent 的 UI 正在向一种新的形态收敛**：来自 [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2044133573326934384)、[@kieranklaassen](https://x.com/kieranklaassen/status/2044108436087157220) 和 [@omarsar0](https://x.com/omarsar0/status/2044172949003911532) 的帖子都指向了同一个趋势：IDE 正在围绕**并行 Agent 会话、可视化的 Artifacts/Apps 以及侧边执行**进行重新设计，而不是将文件和终端作为主要单元。这种收敛很重要，因为它表明 Agent 编程的瓶颈正在从模型能力转向**交互设计和编排（Orchestration）UX**。

**研究亮点：对齐、记忆、评估与科学**

- **Anthropic 正在将自动化研究作为一项具有生产力的窄领域能力主张**：该公司的 [Automated Alignment Researcher](https://x.com/AnthropicAI/status/2044138481790648323) 实验表明，Claude Opus 4.6 可以加速特定对齐问题的实验——即使用弱模型来监督强模型——但并未声称实现了通用的自动化科学。后续研究的关键结论是，这些系统提高了**实验和搜索的速率**，而不是说它们已经成为了稳健的“对齐科学家”。
- **几篇新论文强化了关于 Agent 记忆/评估的叙述**：[@dair_ai](https://x.com/dair_ai/status/2044066936045351317) 强调了关于**将 Artifacts 作为外部记忆**的研究，形式化地定义了环境观察何时能降低内部记忆需求。另一篇由 [@dair_ai](https://x.com/dair_ai/status/2044145437456904438) 总结的论文介绍了 **PASK**，这是一个具有流式意图检测和混合记忆的主动式 Agent 框架。在评估方面，[@arena](https://x.com/arena/status/2044096836114493609) 推出了 **Direct Battles**，将两两评估扩展到多轮对话；而 [@omarsar0](https://x.com/omarsar0/status/2044067923787165799) 展示了针对多用户 Agent 冲突的 **Muses-Bench**，在这项测试中，即便是顶尖模型在会议协调以及隐私与效用的权衡方面仍表现挣扎。
- **科学和数学自动化的主张变得更加具体，但依然具有异质性**：[@Liam06972452](https://x.com/Liam06972452/status/2044051379916882067) 报道了 **GPT-5.4 Pro** 解决了 **Erdős problem #1196**，几位研究人员认为这是一个有意义的结果，而非单纯的刷榜（benchmark gaming）。与此同时，[@iScienceLuvr](https://x.com/iScienceLuvr/status/2043977751506428323) 总结了 **SciPredict**，LLM 预测科学实验结果的准确率仅为 **14–26%**，大致处于人类专家水平。总体情况是，AI 现在可以在某些可形式化的研究领域做出实质性贡献，但通用的实验指导仍然远不可靠。


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen3.5 模型 Quantization 与 Benchmarks

  - **[Updated Qwen3.5-9B Quantization Comparison](https://www.reddit.com/r/LocalLLaMA/comments/1sl59qq/updated_qwen359b_quantization_comparison/)** (热度: 349): **该帖展示了对 **Qwen3.5-9B** 模型各种 **Quantization** 方法的详细评估，使用 **KL Divergence (KLD)** 作为指标来评估 **Quantized** 模型相对于 **BF16 baseline** 的忠实度。分析根据 KLD 分数对 **Quantizations** 进行排名，分数越低表示与原始模型的概率分布越接近。性能最佳的 **Quantization** 版本 **eaddario/Qwen3.5-9B-Q8_0** 达到了 `0.001198` 的 KLD 分数，表明信息损失极小。使用的评估数据集和工具包括 [此数据集](https://gist.github.com/cmhamiche/788eada03077f4341dfb39df8be012dc) 和 [ik_llama.cpp](https://github.com/Thireus/ik_llama.cpp/releases/tag/main-b4608-b33a10d)。** 评论者对详细的分析表示赞赏，并提出了改进建议，如使用不同的形状以提高视觉清晰度，以及包含来自 [gguf.thireus.com](https://gguf.thireus.com/quant_assign.html) 的 **Quantizations** 进行对比。此外，人们对将此方法应用于 Gemma 4 等其他模型也表现出浓厚兴趣。

    - Thireus 建议纳入来自 [gguf.thireus.com](https://gguf.thireus.com/quant_assign.html) 的 **Quantization** 结果，该网站声称其结果优于现有方法。这突显了 **Quantization** 技术的持续发展和竞争，EAddario 等多位贡献者在类似方法上已经工作了近一年，表明了一个充满活力且具有协作性的研究环境。
    - cviperr33 提到在 `20-35B` 范围的模型中使用 `iq4 xs` 或 `nl quant`，并指出它们即使在较小的模型上也很有效。这表明某些 **Quantization** 技术可能在不同模型尺寸上具有更广泛的适用性，从而为模型优化提供统一的方法。
    - PaceZealousideal6091 指出 mradermacher 的 `i1 quants` 表现异常出色，建议将其作为未来对比的重要补充。他们还请求更新之前的 "Qwen3.5-35B-A3B Q4 Quantization Comparison"，以包含最近的更新和新的 **Quantization** 方法，反映了 **Quantization** 策略的快速演进。

  - **[Best Local LLMs - Apr 2026](https://www.reddit.com/r/LocalLLaMA/comments/1sknx6n/best_local_llms_apr_2026/)** (热度: 721): **该帖讨论了截至 2026 年 4 月本地 **LLM** 的最新进展，重点介绍了 **Qwen3.5**、**Gemma4** 和 **GLM-5.1** 的发布，后者声称达到了 **SOTA** (State-of-the-Art) 性能。**Minimax-M2.7** 模型因其易用性而受到关注，而 **PrismML Bonsai** 推出了有效的 1-bit 模型。该讨论串鼓励用户分享他们使用这些模型的经验，重点关注开源权重模型，并详细说明他们的配置、用途和工具。帖子还根据 **VRAM** 需求对模型进行了分类，范围从 'Unlimited' (>128GB) 到 'S' (<8GB)。** 一位评论者建议将 128GB 以上的 **VRAM** 类别进行细分以获得更高的粒度，这表明高性能配置中需要更详细的分类。另一条评论关注 **LLM** 在 **Agentic coding** 和工具调用方面的应用，反映了这些模型向专业化应用发展的趋势。

    - 一位用户建议将显存大于 128 GB 的模型类别拆分为更具体的范围，而不是使用 "S" 或 "M" 这种通用标签。这意味着需要更细粒度的 **Benchmarks** 或分类，以更好地了解大规模模型的性能和能力。
    - 讨论重点关注了针对医疗、法律、会计和数学等特定领域定制的专业本地 **LLM**。这突显了开发针对特定领域优化的模型的趋势，有可能提高这些领域的准确性和效率。
    - 提到了 **Agentic coding** 和工具调用，这表明人们关注能够自主执行任务或与工具交互的模型。这可能涉及将 **LLM** 与 API 或其他软件集成，以增强其在实际应用中的实用性。


### 2. 本地 AI 硬件与配置

- **[小米 12 Pro 上的 24/7 无头 AI 服务器 (骁龙 8 Gen 1 + Ollama/Gemma4)](https://www.reddit.com/r/LocalLLaMA/comments/1sl6931/247_headless_ai_server_on_xiaomi_12_pro/)** (热度: 1108): **图片展示了一台配置为专用本地 AI 服务器的小米 12 Pro 智能手机，利用其骁龙 8 Gen 1 处理器。该方案通过刷入 LineageOS 来移除不必要的 UI 元素，从而优化 AI 任务的操作系统，为 LLM 计算释放了约 `9GB` 的 RAM。该设备以无头（headless）状态运行，网络由自定义 `wpa_supplicant` 管理，散热则通过一个自定义守护进程实现，该进程在 `45°C` 时激活外部冷却模块。电池健康通过一个将充电限制在 `80%` 的脚本来维护。手机通过 Ollama 提供 Gemma4 模型作为局域网可访问的 API，展示了消费级硬件在 AI 应用中的新颖用法。** 一位评论者建议在硬件上直接编译 `llama.cpp`，这可能会使推理速度翻倍，表明了通过使用替代软件方案来优化性能的倾向。另一条评论赞赏了对消费级设备运行 AI 模型的关注，这与当前需要高内存配置的趋势形成对比。

    - RIP26770 建议直接在小米 12 Pro 硬件上编译 `llama.cpp`，与使用 Ollama 相比，这可能会使推理速度翻倍。这意味着 Ollama 带来的开销可能很显著，针对特定硬件优化模型编译可以获得更好的性能。
    - SaltResident9310 表达了对能在消费级设备上高效运行的 AI 模型的渴望，强调了对当前模型动辄需要 48GB 或 96GB RAM 的高资源需求的挫败感。这突显了人们对在更易获取的硬件上优化 AI 的广泛兴趣。
    - International-Try467 询问了小米 12 Pro 实现的具体推理速度，表明了对在该设备上运行 AI 模型的性能指标的技术兴趣。这反映了对现实场景中实际性能结果的关注。

  - **[后续贴，决定打造 2x RTX PRO 6000 塔式服务器。](https://www.reddit.com/r/LocalLLaMA/comments/1sklhzv/follow_up_post_decided_to_build_the_2x_rtx_pro/)** (热度: 459): **该帖子详细介绍了一台高性能工作站的构建，配备双 NVIDIA RTX PRO 6000 GPU，每张显卡拥有 `96GB GDDR7 ECC` 显存，集成在单个塔式机箱中。系统搭载 **AMD Threadripper PRO 7965WX** CPU，使用 **ASUS Pro WS WRX90E-SAGE SE** 主板，支持 `128 条 PCIe 5.0 通道`。配置包括 `256GB DDR5-4800 ECC RDIMM` RAM 和强大的冷却系统，CPU 采用水冷，并配备多个进气和排气风扇。该设置专为密集型计算任务设计，利用 `192GB 总显存` 和 `每张卡 500W 的功率限制`，并配备专用的 `20A 120V 电路` 以支持电力需求。存储方案包括用于操作系统和模型的高速 `Samsung 9100 PRO 8TB` SSD，以及用于临时空间的 `2TB SSD`，针对数据密集型应用进行了优化。** 评论反映了该配置的高昂成本，一位用户幽默地将其与汽车价格进行比较。另一条评论强调了电力需求，指出了在共享的 15A 电路上运行此类设置的挑战。

    - MachinaVerum 强调了高性能配置中散热的重要性，尤其是在使用双 RTX PRO 6000 GPU 时。他们建议不要对 CPU 使用风冷，因为 GPU 会产生 `1200W` 的热量，这会严重影响 CPU 温度。相反，他们建议使用设置为进风的 Silverstone 一体式水冷散热器，以有效管理热量输出并保持最佳 CPU 温度。

  - **[刚拿到一个……正在构建本地优先的东西 👀](https://www.reddit.com/r/LocalLLM/comments/1sk3zng/just_got_my_hands_on_one_of_these_building/)** (热度: 537): **图片描绘了一块 NVIDIA RTX PRO 6000 Blackwell Max-Q 工作站版 GPU，用户计划将其集成到高性能的本地优先计算设置中。该配置包括 `9950X` CPU、`128GB RAM` 和一块 `ProArt 主板`，表明其关注点是高级 AI 和服务器任务而非游戏。用户旨在实现多用户并发推理并保持对数据的本地控制，避免依赖外部 API 提供商。他们正在探索如 `vLLM` 和 `llama.cpp` 等技术来构建系统以高效处理多用户，并计划增加第二块 GPU 以实现扩展性。** 一位评论者建议加入 RTX 6000 Discord 社区寻求建议，表明了该高端 GPU 用户之间的协作环境。另一条评论幽默地提到了购买如此强大 GPU 的诱惑，反映了尖端硬件的魅力。

- Sticking_to_Decaf 分享了一个使用 RTX 6000 的详细配置方案，建议将 `vLLM` 与 `cu130 nightly image` 配合使用。他们强调在运行像 `Qwen3.5-27B-FP8` 这样的大型模型时，将 KV cache 数据类型设为 `fp8_e4m3`，在仅占用 `55%` 显存的情况下，实现了约 `160k tokens` 的最大上下文长度。该配置在单请求下支持 `80-90 TPS`，在多并发请求下超过 `250 TPS`，并为 `whisper-large-v3` 和 reranker 模型等额外模型留出了空间。
- 评论者提到在此配置下运行 `Hermes Agent`，集成了用于记忆的 `OpenViking` 以及用于网页搜索的 `Firecrawl` 和 `Searxng` 等本地模型。这种组合被认为完全本地化且高效，展示了 RTX 6000 在复杂多模型部署方面的潜力。该配置还预见了 `Qwen3.5` 未来对 multi-LoRA 的支持，表明了持续开发和优化的潜力。

### 3. Elephant Alpha 和新模型发布

- **[1000 token/s, it's blazing fast!!! Fairl](https://www.reddit.com/r/LocalLLaMA/comments/1sl8a8o/1000_tokens_its_blazing_fast_fairl/)** (热度: 369): **图中是来自 **OpenRouter** 的社交媒体帖子，宣布了一个名为 "Elephant Alpha" 的新型隐身模型，这是一个 `100 billion parameter` 的即时模型。它因在代码补全、调试、文档处理和轻量级 Agent 等任务中具有业界领先的性能而受到关注，强调其速度和 token 效率，声称达到 `1000 token/s`。这表明模型吞吐量和效率有了重大进步，可能使其成为高速语言模型应用领域的领导者。** 评论反映了对模型速度的怀疑，一位用户质疑 `1000 token/s` 这一说法的来源，指出 OpenRouter 模型页面列出的吞吐量约为 `~100t/s`。另一条评论建议这种速度可能是 diffusion LLM 的特征，并将其与 "Llada" 进行了比较。

    - 一位用户推测，达到每秒 1000 tokens 的模型可能是基于扩散的 LLM（diffusion-based LLM），例如 Llada，此类模型以高速处理著称。这表明模型的架构可能针对速度进行了优化，但也可能以牺牲准确性或理解深度等其他因素为代价。
    - 另一条评论强调了使用状态空间模型（state-space models）的可能性，这类模型利用线性注意力计算（linear attention calculations）而非平方级计算。这种架构选择可以显著提升推理速度，使得模型达到如此高的吞吐量成为可能。评论者指出，具有混合层的模型通常会采用这种技术来提升性能。
    - 一位用户分享了他们使用 LiquidAI 的 24B MoE 模型的经验，该模型使用 vllm 在 Mac Studio 上实现了超过每秒 200 tokens 的速度。他们认为，在更强大的生产级硬件上，具有高效状态空间架构的模型确实有可能达到每秒 1000 tokens，这表明硬件和架构效率对于实现高吞吐量至关重要。

- **[What Is Elephant-Alpha ???](https://www.reddit.com/r/LocalLLaMA/comments/1skfknl/what_is_elephantalpha/)** (热度: 450): **图中描述了 "Elephant Alpha"，这是一个强调智能效率的 100B 参数文本模型。它拥有强大的推理能力、`256K` 上下文窗口，并支持多达 `32K` 的输出 tokens，表明其具有处理大规模复杂文本输入的能力。该模型已集成到 OpenRouter，后者可将请求路由优化至最佳提供商，显示出对性能和易用性的关注。评论强调了其令人印象深刻的速度，处理速率达 `1000 tokens/s`，并幽默地质疑为什么给一个以速度和效率著称的模型起名 “Elephant”（大象）。** 评论者对模型 `1000 tokens/s` 的处理能力印象深刻。此外，关于模型名称 “Elephant” 还有一场轻松的讨论，认为对于一个快速且高效的模型来说，这个名字似乎有些反直觉。

- Technical-Earth-3254 强调了 Elephant-Alpha 模型令人印象深刻的速度，指出它可以处理 `1000 tokens/s`，这被认为对于语言模型来说极快。这表明该模型的架构或硬件加速（hardware acceleration）进行了重大优化。
    - ArthurOnCode 认为 Elephant-Alpha 的响应模式（其特点是长时间停顿后瞬间输出整版文本）与 diffusion model 一致。这与 Mercury 的响应进行了对比，表明流式（streaming）diffusion 响应是可能的，但目前 openrouter 并不支持，暗示了潜在的后端差异或限制。
    - 关于天安门广场事件的详细回答展示了该模型快速生成全面历史叙述的能力。该模型提供时间线、媒体视角和长期后果的能力，表明它非常适合需要详细历史分析和综合的任务。

  - **[Kimi K2.6 即将来临](https://www.reddit.com/r/LocalLLaMA/comments/1sk9twd/kimi_k26_imminent/)** (Activity: 494): **这张图片是来自 Kimi Code Team 的一封邮件，宣布即将发布 Kimi K2.6 code-preview 模型，这是一个专注于代码的微调（fine-tuned）模型。此次发布是在收集反馈以改进产品的测试计划之后进行的。该模型预计很快将向所有人开放，它似乎是对 Mythos 等类似模型的回应，预示着代码导向型 AI 模型领域的竞争格局。[图片](https://i.redd.it/3wr3ia70fyug1.jpeg)** 一位评论者幽默地提到了该模型的高资源需求，暗示在典型配置上运行可能并不可行，即使拥有 `144GB` 的 RAM。另一条评论强调了该模型对代码的关注，将其与 Mythos 模型进行了对比，认为 Kimi K2.6 是专业化代码模型趋势的一部分。

    - Dany0 强调 Kimi K2.6 是一个专注于代码的微调版本，认为它可能受到了像 Mythos 这样针对特定任务（如代码生成）定制的模型的启发。这表明了向特定领域优化性能的专业化模型发展的趋势，可能会提高代码相关任务的效率和准确性。
    - Canchito 对潜在的 API 定价通胀表示担忧，并将其与 GLM 的定价策略进行了类比。这反映了一个更广泛的行业问题，即先进模型尽管功能强大，但由于成本原因，其可获得性可能会降低，从而影响依赖这些技术的开发者和企业。


## 非技术性 AI 版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude Opus 4.7 与 Mythos 模型进展

  - **[Anthropic 最快将于本周发布 Claude Opus 4.7 和一款新的 AI 设计工具](https://www.reddit.com/r/singularity/comments/1slh72j/anthropic_is_set_to_release_claude_opus_47_and_a/)** (Activity: 711): ****Anthropic** 准备发布 **Claude Opus 4.7** 和一款新的 AI 设计工具，可能就在本周。该设计工具旨在帮助技术和非技术用户使用自然语言提示创建演示文稿和网站，将与 Gamma 以及 Google 的 AI 设计工具 Stitch 展开竞争。虽然 Opus 4.7 并不是最先进的模型——**Claude Mythos** 占据了这一头衔，目前正在接受网络安全能力测试——但预计 Opus 4.7 将改进 Opus 4.6 中发现的性能问题，此前 Opus 4.6 的表现不佳可能是为了突出新版本的进步。** 评论者们推测 Anthropic 有意让当前模型在新版本发布前表现不佳，从而使改进看起来更显著，一些人认为这种做法令人沮丧。此外，由于潜在的频率限制（rate limiting），人们对新模型的可获得性持怀疑态度，这可能会更倾向于订阅了高阶方案的用户。

- Anthropic 即将发布的 Claude Opus 4.7 正在引发关于其性能与前代模型对比的讨论。一些用户推测 Opus 4.6 表现不佳是刻意为之，目的是让 Opus 4.7 的改进更加引人注目。这符合一种模式，即旧模型在新版本发布前被认为性能有所下降，从而可能凸显新模型的进步。
- Anthropic 的新型 AI 设计工具预计将通过让技术和非技术用户都能使用自然语言 prompts 创建数字内容，从而与 Gamma 和 Google Stitch 等现有工具展开竞争。该工具可能通过简化演示文稿、网站和着陆页的创建而显著影响市场，从而对该领域的现有初创公司构成威胁。
- Claude Mythos 是 Anthropic 最先进的模型，目前正在接受网络安全能力测试。早期合作伙伴正利用它来识别安全漏洞，展示了其在增强软件安全方面的潜力。这使 Claude Mythos 定位为网络安全的专门工具，有别于 Opus 4.7 的通用功能。

- **[The Information: Anthropic Preps Opus 4.7 Model, could be released as soon as this week](https://www.reddit.com/r/ClaudeAI/comments/1slhkt8/the_information_anthropic_preps_opus_47_model/)** (热度: 467): **Anthropic** 准备发布 **Opus 4.7 模型**，预计该模型将通过提高 AI 系统的效率和有效性来推进 AI 设计能力。该模型旨在解决 AI 训练和部署中的现有局限，可能比之前的版本有显著改进。更多详情请参阅原文[此处](https://www.theinformation.com/briefings/exclusive-anthropic-preps-opus-4-7-model-ai-design-tool)。评论者对 Opus 4.7 相对于 Opus 4.6 的改进持怀疑态度，有人认为它可能只是一个小更新或“削弱（nerfed）”版本，并将其类比为“新可乐（New Coke）”情境，即更改未获好评。

- **[AI Security Institute Findings on Claude Mythos Preview](https://www.reddit.com/r/singularity/comments/1skc04d/ai_security_institute_findings_on_claude_mythos/)** (热度: 559): **该图展示了 AI 模型在网络能力方面的对比分析，特别关注 Mythos Preview 模型。图表显示，Mythos Preview 在完成从侦察到接管网络的网络操作步骤效率方面，明显优于 Claude Opus 和各种 GPT 版本。X 轴使用对数刻度表示累计 tokens，而 Y 轴显示完成的平均步骤，突显了 Mythos Preview 性能的陡增。** 一条值得注意的评论指出，开源模型仅落后于 SOTA 前沿模型约 12 个月，这意味着开发速度极快，且解决潜在安全漏洞的需求迫在眉睫，类似于 Y2K 问题但没有明确的截止期限。

    - 讨论强调了开源模型追赶 SOTA 前沿模型的惊人速度，滞后时间约为 12 个月。这种快速进步凸显了采取安全措施的紧迫性，并将其类比为 Y2K 问题，但目前还没有明确的解决方案时间表。
    - 提出的一个关键点是 AI 安全领域正在进行的“军备竞赛”，大公司有资源访问并保护 SOTA 模型，而较小实体必须等待开源模型进步或分配大量资源来保持安全。随着恶意行为者利用漏洞的成本和努力降低，针对中小型目标的风险增加。
    - 评论认为 Mythos 模型代表了重大进步，暗示尽管存在对营销炒作的怀疑，但 AI 能力确实存在可能影响安全格局的飞跃。

- **[DeepSeek V4 launching late April – plus Anthropic's "too dangerous" Mythos model, Meta's $135B AI bet](https://www.reddit.com/r/DeepSeek/comments/1ski33m/deepseek_v4_launching_late_april_plus_anthropics/)** (热度: 139): **DeepSeek V4** 定于 4 月底发布，据 [TVBS 新闻网](https://news.tvbs.com.tw/tech/3170653) 报道，其可能针对 **Huawei AI chips** 进行了优化，以减少对 NVIDIA 的依赖。与此同时，**Anthropic 的 “Mythos” 模型**被认为“黑客能力强得惊人”，将不会公开发布；相反，它将在名为 *Project Glasswing* 的安全倡议下，分享给 Amazon 和 Microsoft 等选定合作伙伴。评论者对 Mythos 模型的真实能力表示怀疑，认为它可能被过度炒作，并对将其描述为重大威胁的营销策略提出质疑。

- 一位用户批评了围绕 Anthropic 的 Mythos 模型的营销策略，认为关于该模型“过于危险”的炒作被夸大了。他们认为这类说法是 AI 营销中更广泛趋势的一部分，即将模型描绘成革命性的，但最终只是对之前版本的渐进式改进。这与行业中出现的一种模式相吻合，即新模型的发布往往伴随着对其能力和潜在影响的夸大宣传。
- 另一条评论强调了 DeepSeek 为减少对 Nvidia 的依赖而采取的战略举措，即在其最新模型中采用华为的新芯片。在 AI 硬件领域，Nvidia 一直是主导者，这一决定具有重大意义。转向华为的技术可能预示着 AI 硬件多样化的更广泛趋势，以减轻与依赖单一供应商相关的风险。
- 一位用户对某些 AI 公司的伦理实践表示怀疑，特别批评了它们的营销和业务策略。他们认为，像 Anthropic 这样的公司通过夸大其模型的能力和风险来进行“煤气灯操纵”（gaslighting），以操纵公众认知并推动销售。这反映了 AI 社区对 AI 开发和营销中透明度与诚实性的更广泛担忧。


### 2. OpenRouter 的 Elephant Alpha 模型发布

- **[来自 OpenRouter 的新 Stealth 模型 Elephant](https://www.reddit.com/r/DeepSeek/comments/1skg0kz/new_stealth_model_elephant_from_openrouter/)** (活跃度: 136): **图片展示了 “Elephant Alpha”，这是来自 **OpenRouter** 的一款新型 100B 参数文本模型。该模型强调“智能效率”和强劲性能，表明其旨在处理具有大 context size 的复杂任务。网页提供了发布日期和每百万 token 的成本等详细信息，表明其专注于开发者的透明度和可访问性。该模型回答敏感问题（如关于天安门广场的问题）的能力表明，它不受某些地区典型审查限制的影响。** 一位评论者指出，该模型讨论敏感话题（如天安门广场）的能力表明它不是中国模型，因为此类讨论在中国通常会被审查。

    - Realistic_Plant_446 强调，该模型公开讨论敏感话题（包括伤亡估计）的能力表明它不受中国审查规范的约束。这意味着该模型训练数据中具有一定的开放性和透明度，这在根据中国法规开发的模型中是不典型的。
    - Wise-Chain2427 和 Nid_All 都提到了 “deepseek”，可能指的是 Elephant 模型未达到的某个 Benchmark 或标准。这表明虽然该模型拥有庞大的参数量 (100B)，但在某些应用或测试中，它可能无法达到某些用户预期的性能或深度。
    - Formal-Narwhal-1610 提到的 “3.1 Gemini Flash” 可能指的是另一个模型或版本，可能表示正在将 Elephant 模型与其进行对比或作为基准标准。这暗示了在多个模型进行性能或功能集评估的背景。

- **[Openrouter 上的 Elephant-alpha 模型，100B 参数，256K context，1000 token/s，虽小但快得惊人！](https://www.reddit.com/r/DeepSeek/comments/1slbjhy/elephantalpha_model_on_openrouter_100bparameter/)** (活跃度: 66): **“Elephant Alpha” 模型是一款在 Openrouter 上可用的 100B 参数文本模型，旨在实现高效率和高性能。它支持 `256K` context window，并且最高可输出 `32K` tokens，处理速度达 `1000 tokens per second`。该模型包含 function calling 和 structured output 等功能，强调其在最小化 token 使用的情况下处理大 context 的能力，使其适用于需要快速高效文本处理的应用。** 评论反映了对该模型深度和智能的怀疑，一位用户幽默地将其称为 “ShallowSeek”，暗示尽管其速度很快，但在理解或 reasoning 深度方面可能有所欠缺。

- **[OpenRouter 刚刚发布了一个新的 100B 模型](https://www.reddit.com/r/Bard/comments/1skfbvf/openrouter_just_announced_a_new_100b_model/)** (热度: 274): **OpenRouter 发布了一个名为 "Elephant Alpha" 的新模型，其拥有 `100 billion parameters`。该模型旨在提供最先进的性能，并专注于 token 效率，使其适用于代码补全、调试、文档处理和轻量级 Agent 等任务。公告暗示 "Elephant Alpha" 是一个 stealth 模型，可能预示着战略发布或初始阶段的有限可用性。** 评论者推测 "Elephant Alpha" 可能与新的 Grok 模型有关，因为此类模型通常会先在 OpenRouter 上出现。此外，大家一致认为它不是 Google 的模型，因为 Google 通常不会公开其专有模型的参数数量。

    - Nick-wilks-6537 和 Artistic_Survey461 讨论了新的 100B 模型是 'Grok' 的可能性，Grok 是在 X 等社交媒体平台上被广泛猜测的模型。他们指出，像 Grok 这样的模型通常会先出现在 OpenRouter 上，有时是在隐藏或未命名的提供商下，这表明了新模型引入该平台的一种模式。
    - Capital-Remove-6150 对新模型的性能发表了评论，称在测试中它似乎并未达到 SOTA 或接近 SOTA 的水平。这表明尽管该模型拥有庞大的参数量，但其性能可能无法与该领域的领先模型相媲美。

  - **[OpenRouter 上的新 Stealth 模型](https://www.reddit.com/r/SillyTavernAI/comments/1skofg1/new_stealth_model_at_openrouter/)** (热度: 111): **图片展示了 "Elephant Alpha" 的详细信息，这是一款在 OpenRouter 上提供的 100B 参数文本模型，发布日期为 2026 年 4 月 13 日。它强调“智能效率”，具有 `262,144` 的庞大上下文大小，且值得注意的是，输入或输出 token 均无费用。界面提供了概览、Playground 和提供商等功能，以及聊天和对比功能。该模型被推测是西方或中国开发的，一些用户认为它可能与 Gemini Flash 或 GLM 5.1 Air 等模型有关。然而，人们对其在创意写作和角色扮演 (RP) 场景中的有效性表示怀疑。** 评论表达了强烈的共识，即 "Elephant Alpha" 在角色扮演 (RP) 方面效果不佳，用户称其在这些应用中“完全没用”且“简直愚蠢”。

    - Syssareth 对新的 Stealth 模型进行了详细批评，强调了它作为“灵感板”的潜力，因为它能够引入新颖的故事方向。然而，该模型在维持叙事连贯性方面表现吃力，例如它倾向于混淆词汇（例如将“受损的翅膀”描述为“曾经骄傲的蛾子”）。此外，该模型的感知智能 (EQ) 欠缺，经常导致角色互动不恰当，与故事背景不符，例如在具有复杂历史的角色之间给出过于简单的和解。
    - 该 Stealth 模型的写作风格因产生听起来深奥但缺乏实质意义的语句而受到批评。给出的一个例子是角色对虐待者的反思，虽然词藻华丽，但内容最终空洞无物。这种倾向使得该模型不太适合需要深度和细微差别的角色扮演 (RP) 场景。此外，该模型在 Memory Books 中的输出被描述为冗长且重复，未能为叙事添加有意义的内容，这体现在它对角色关系中神话类比的冗余探索中。


### 3. Gemini 模型性能与用户体验

  - **[大事将至。Gemini 模型不再被标记为 "new"](https://www.reddit.com/r/Bard/comments/1skl372/something_is_coming_gemini_models_are_no_longer/)** (热度: 195): **图片揭示了 Gemini 系列中两个即将推出的模型的预览：**Gemini 3.1 Pro** 和 **Gemini 3.1 Flash Lite**。Pro 模型因其先进的推理和多模态能力而受到关注，适用于复杂任务，而 Flash Lite 模型则是为翻译等具有成本效益的高容量操作而设计的。这两个模型的知识截止日期均为 2025 年 1 月，Pro 模型定于 2026 年 2 月 12 日发布。这表明 Google 的 AI 产品线正在进行战略更新，可能是为了迎接即将到来的 Google IO 等活动。** 评论者推测，移除 "new" 标签可能是因为 Gemini 4 即将发布，或者是为了在 Google 的云展会或 Google IO 上发布公告。

- Dangerous-Relation-5 指出了当前系统的一个关键性能问题，提到了频繁出现的 'server too busy'（服务器繁忙）消息。这表明需要进行基础设施升级以应对不断增长的需求，可能预示着当前的服务器架构在处理用户负载时无法有效扩展。

  - **[Gemini 还.... 行？](https://www.reddit.com/r/Bard/comments/1sl2mcm/gemini_is_fine/)** (Activity: 65): **该帖子讨论了作者使用 **Gemini**（一款 AI 工具）的经验，强调了其在医疗咨询、药物相互作用和创意写作语法检查等任务中的胜任能力。作者指出，尽管社区对 Gemini 的性能表示担忧，但它在满足其需求方面表现尚可，特别是在使用自定义 GEMs 和 Notebooks 来引导输出时。作者提到 Gemini 的局限性（如幻觉）在其使用场景中是可控的，并且该工具能有效地遵循指令。关于 `310K Rupiah` 的本地定价，其价值受到质疑，但总体上被描述为“还行”。** 评论者普遍同意作者的评估，指出 Gemini 在大多数任务中表现良好，但在处理较长的写作任务时可能会感到吃力。一些用户表示没有遇到重大问题，认为 Gemini 足以满足他们的需求。

    - BlackFlagCat 指出，与其他 LLM 相比，Gemini 需要更详细的初始提示（prompt），这对于增强现有工作或提供高层概述等任务是有益的。然而，在需要精美输出且缺乏详细引导的 zero-shot 任务中，它的表现欠佳。这表明 Gemini 的优势在于迭代式和上下文丰富的交互，而非即时的独立输出。
    - jk_pens 讨论了 Gemini 与 Google 生态系统的集成，指出虽然它仍有待完善且偶尔会出现性能倒退，但随着其嵌入程度加深，其实用性正在增加。这种集成使其成为重度使用 Google 服务的用户的强大通用选择，尽管在某些任务或偏好上仍需 Claude 等模型作为补充。
    - Jazzlike-Tie-9543 提到了 Gemini 在生成长文本（如超过 2,000 字）能力方面的局限。这表明虽然 Gemini 在许多领域都很称职，但在没有大量用户输入或迭代开发的情况下，它可能不适合需要生成大量内容的任务。

  - **[Gemini 拥有一切……但为什么还是在输？🤔](https://www.reddit.com/r/GeminiAI/comments/1sl0uaz/gemini_has_everything_so_why_is_it_still_losing/)** (Activity: 1114): **尽管 **Gemini** 拥有广泛的资源，包括对 Chrome 的所有权、Android 的支持以及对全球约 `95%` 搜索数据的访问权，但它在与 **Claude** 和 **GPT** 的竞争中仍显吃力。该平台庞大的数据索引和存储能力，以及 Google 巨大的用户数据生态系统，尚未转化为具备竞争力的性能表现。一个关键问题似乎是 Gemini 的高幻觉率，这损害了它的可靠性。** 不同 AI 社区的用户意见存在明显的分歧，每个平台的用户往往觉得自己的平台不如别人。一些用户认为，尽管有数据优势，Gemini 的高幻觉率仍是一个重大缺陷。

    - **MarionberryDear6170** 强调了 Gemini 的一个关键问题：高幻觉率。尽管可以访问海量数据，Gemini 经常生成不准确的信息，与 ChatGPT 和 Claude 等竞争对手相比，这削弱了其可靠性。
    - **Gaiden206** 指出，虽然 Gemini 凭借与 Android 操作系统和 Google 服务的集成拥有庞大的用户群，但它缺乏开发者的青睐（mindshare）。Reddit 和 X 等平台上的开发者在处理编程等任务时更倾向于使用 Claude 4.6 或 GPT，这表明尽管 Gemini 具有主流吸引力，但在技术偏好上仍存在差距。
    - **UninvestedCuriosity** 讨论了 Google 在模型压缩方面的战略优势，如近期一份白皮书所述。这一进展使得在单个 GPU 内实现显著的模型性能提升成为可能，可能迫使竞争对手在数据和研究上投入巨资以保持同步。Google 的方法可能不在于即时竞争，而在于其生态系统内的长期可行性和集成。

- **[我的大学因学生在考试中使用 Gemini 而将其永久开除](https://www.reddit.com/r/GeminiAI/comments/1sl3cy5/my_uni_permanently_expelled_a_student_for_using/)** (活跃度: 649): **该图片是某大学信息工程学院的一份官方公告，详细说明了两名学生因在考试期间使用移动设备访问互联网而被开除的情况，特别提到了使用 **Gemini AI** 的行为。这突显了该机构对学术诚信和在考试中使用 AI 工具的严厉态度，反映了公众对 AI 在教育中的角色及其可能助长作弊行为的广泛担忧。该文件强调了维护考试诚信的重要性以及违反这些标准的严重后果。** 评论者们正在争论惩罚的严厉程度，一些人质疑为什么使用像 Gemini 这样的 AI 会导致比其他作弊手段更严厉的处罚。这反映了关于 AI 在学术环境中的伦理影响和挑战的持续讨论。

    - SpecialistDragonfly9 提出了一个关键点，即 AI 辅助作弊与传统作弊方法在惩罚力度上存在差异。这表明教育机构需要重新评估其政策，并确保对不同形式的学术不端行为的处理是相称且一致的。
    - 高等教育专业人士 WanderByJose 强调了在评估中维护道德标准和诚信的重要性，即使在 AI 工具日益普及的情况下也是如此。他们建议应将 AI 作为一种辅助工具，而不是破坏评估体系的手段，并强调大学需要就此类问题制定明确的指南并进行公开沟通。


# AI Discords

遗憾的是，Discord 今天关闭了我们的访问权限。我们不会再以这种形式恢复它，但我们很快就会发布全新的 AINews。感谢读到这里，这是一段美好的历程。