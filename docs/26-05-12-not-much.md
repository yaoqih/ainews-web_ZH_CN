---
companies:
- google-deepmind
- lighton
- nous-research
date: '2026-05-12T05:44:39.731046Z'
description: '**研究级推理基准**正取得新进展：64 位数学家贡献了 **439 道新数学题**；**Medmarks v1.0** 扩展了医疗基准，涵盖了
  **30 个基准测试**和 **61 个模型**。**Google DeepMind 的 AI 协同数学家 (AI Co-Mathematician)** 在
  **FrontierMath Tier 4** 上达到了 **48%** 的准确率，而 **Gemini 3.1 Pro** 显著提升了物理基准测试得分。在程序合成任务中，**GPT-5.5
  high/xhigh** 的表现优于 **Opus 4.7 xhigh**。


  检索基准测试则更看好较小的模型，如拥有 **1.49 亿参数**的 **LightOn Agent-ModernColBERT**。训练优化方面的进展包括：旨在减少训练步数的
  **SOAP/Muon 风格更新**，以及在 A100 GPU 上实现 **1.8 倍加速**的 **Lean4 到 TileLang 超级优化器**。**缩放定律
  (Scaling laws)** 正在被重新审视，有观点认为应以字节 (bytes) 而非词元 (tokens) 为衡量标准。此外，**Lighthouse Attention**
  等新型训练效率方法实现了**亚二次 (subquadratic) 训练封装**，且这些封装在部署前即可移除。'
id: MjAyNS0x
models:
- gemini-3.1-pro
- gpt-5.5
- opus-4.7-xhigh
- agent-moderncolbert
people:
- soohak
- polynoamial
- torchcompiled
- leloykun
- che_shr_cat
- jjitsev
- omarsar0
title: 今天没发生什么。
topics:
- research-benchmarks
- math
- medical-benchmarks
- agentic-systems
- program-synthesis
- retrieval-augmentation
- training-optimization
- superoptimization
- scaling-laws
- training-efficiency
- gpu-optimization
- attention-mechanisms
---

**平静的一天。**

> 2026年5月11日至2026年5月12日的 AI 新闻。我们检查了 12 个 subreddits、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，未查看更多 Discord。[AINews 网站](https://news.smol.ai/) 允许您搜索所有往期内容。提示：[AINews 现已成为 Latent Space 的一部分](https://www.latent.space/p/2026)。您可以[选择开启/关闭](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)邮件发送频率！

---

# AI Twitter 综述


**研究级基准测试、高难度评估与 Agent 科学系统**

- **研究级推理基准测试难度持续攀升**：[Soohak](https://x.com/gson_AI/status/2054036114483392997) 引入了 **439 个研究级数学问题**，这些问题由 **64 位数学家**（包括 **38 位教职人员**）从零开始编写，明确针对超越标准奥数风格数学的能力。在医学评估方面，[@SophontAI](https://x.com/SophontAI/status/2054270239387627927) 发布了 **Medmarks v1.0**，将其开源医学基准测试套件从 **20 个扩展到 30 个基准测试**，模型覆盖量从 **46 个增加到 61 个**。此外，人们越来越倾向于认为旧的评估指标已趋于饱和：[@polynoamial](https://x.com/polynoamial/status/2054255862441812099) 认为，得分普遍偏高的基准测试应该退役，取而代之的是得分较低、具有前沿挑战性的测试。
- **Agent 系统开始推动科学和数学基准测试的前沿**：Google DeepMind 的 [AI Co-Mathematician](https://x.com/dair_ai/status/2054224343551639958) 被描述为一个面向数学家的异步、有状态的研究工作台，据报道在 **FrontierMath Tier 4** 上达到了 **48%** 的准确率，同时支持构思、文献发现、计算分析、定理验证和正式输出。在理论物理领域，[physics-intern](https://x.com/dlouapre/status/2054217281895309480) 通过将其分解为专门的 Agent，将 **Gemini 3.1 Pro 在 CritPt 上的表现从 17.7% 提升至 31.4%**。在代码/程序合成方面，[ProgramBench 的首个任务](https://x.com/KLieret/status/2054215545663144217) 据报道已由 **GPT-5.5 high/xhigh** 解决，其中 xhigh 在各项指标上均优于 **Opus 4.7 xhigh**。
- **检索和搜索基准测试正让小型专业化模型受益**：LightOn 的 [Agent-ModernColBERT](https://x.com/LightOnIO/status/2054202169255973121) 在 BrowseComp-Plus 上比 Reason-ModernColBERT 又提升了约 **10%**，同时将检索器保持在 **149M 参数**，并声称在与生成器配合使用时，可以达到或超过更大规模基于模型的系统。来自 [@xuzihuan4](https://x.com/xuzihuan4/status/2054220800073642161) 的相关讨论询问，当 Agent 可以迭代优化自己的查询时，词法检索在 Agent 搜索循环中是否已经足够。

**训练、优化与 Scaling-Law 技术**

- **优化器工作持续压缩训练成本并改进小规模实验**：多条推文集中讨论了 **SOAP/Muon 风格更新**的快速变体。[@torchcompiled](https://x.com/torchcompiled/status/2054036715589771542) 将切向步长 (tangent-step) + Stiefel 流形收缩 (Stiefel manifold retraction) 应用于 **SOAP 基准更新**，并就漂移检查 (drift checks) 和用于稳定性的 QR 回退 (QR fallback) 进行了[后续讨论](https://x.com/torchcompiled/status/2054088499591000255)。在 Modded-NanoGPT 社区中，[SOAP-Muon](https://x.com/kellerjordan0/status/2054255672636981423) 以 **3150 步 (-60)** 创下了新纪录，而早先在 NorMuonH 上进行的 [MuLoCo 风格外部 Nesterov SGD 封装](https://x.com/kellerjordan0/status/2054098451621978471) 也改进了结果，两者均有 p 值报告支持。
- **形式化方法与超优化 (superoptimization) 开始与 ML 系统工作融合**：[@leloykun](https://x.com/leloykun/status/2054076097881592068) 介绍了一个 **Lean4 到 TileLang 的张量程序超优化器**，它可以自动发现诸如 **FlashAttention2**、**FlashNorm** 和 **split-k matmul** 等 Kernel，据报告在 A100 上实现了约 **1.8 倍的几何平均加速**。该框架定位为联合搜索 Kernel、优化器、超参数迁移规则和 Scaling Laws。
- **Scaling Laws 和训练指标正在被重新审视**：[@che_shr_cat](https://x.com/che_shr_cat/status/2054178651856339276) 认为经典的 **“每个参数 20 个 token”** 框架依赖于 Tokenizer，Scaling 应该以 **字节 (bytes)** 而非 token 来衡量。另外，[@JJitsev](https://x.com/JJitsev/status/2054166378823794881) 强调，指令性 Scaling Laws 的价值不仅在于预测，还在于作为跨规模比较学习过程的系统化基础。
- **仅限训练时的效率技巧变得越来越有趣**：来自 Nous 的 [Lighthouse Attention](https://x.com/omarsar0/status/2054224130103554359) 被强调为一种围绕原生 Attention 的亚二次 **训练封装器**，它可以在训练接近结束的恢复阶段后被移除，在降低长上下文预训练成本的同时，保留标准的部署时推理。本着类似的精神，来自 Prime Intellect 的 [Renderers](https://x.com/PrimeIntellect/status/2054347134821154841) 解决了 RL 训练器与 Agent 环境之间的 token/消息阻抗失配，声称在流行的开源模型上实现了 **>3 倍的吞吐量**。

**推理系统、服务栈与运行时基础设施**

- **Blackwell 机架正成为大规模 MoE 推理服务的参考平台**：Perplexity 发布了在 **NVIDIA GB200 NVL72** 系统上提供后训练 **Qwen3 235B** 服务的细节，认为 GB200 是大规模 MoE 相比 Hopper 的重大推理升级。他们的[基准测试](https://x.com/perplexity_ai/status/2054204425833726353)指出，**NVLS all-reduce 延迟**从 H200 的 **586.1µs 降至 GB200 的 313.3µs**，EP=4 时的 **MoE prefill 合并**从 **730.1µs 降至 438.5µs**，且在高 token 率下具有更好的 decode 吞吐量。[@AravSrinivas](https://x.com/AravSrinivas/status/2054206802133504234) 将此描述为实质性地改变了大规模 MoE 推理服务的 prefill/decode 分离 (disaggregation)。
- **推理编排日益专业化，不再“仅仅是 Kubernetes”**：[Modal](https://x.com/charles_irl/status/2054233051140690023) 认为推理需要专门的堆栈，并提到了在计算管理、云原生缓存、**CRIU** 和 **GPU 检查点 (checkpointing)** 方面的工作。这一立场立即得到了 Perceptron 的现实认可，后者表示 [所有 Mk1 推理都运行在 Modal 上](https://x.com/AkshatS07/status/2054275262289002664)，因为原生视频、结构化输出和混合推理产生了不同寻常的冷启动和扩缩容需求。
- **开源 (OSS) 推理的经济效益继续快速提升**：[SemiAnalysis](https://x.com/SemiAnalysis_/status/2054245527957508520) 报告称，通过 **RoCEv2 CX-7** 将多台 **B200 8-GPU** 机器集群化，并结合 **PD 分离**，可以将**单 GPU token 吞吐量提高多达 7 倍**，这意味着每 token 成本同比例降低。在向量数据库方面，[Qdrant 1.18](https://x.com/qdrant_engine/status/2054166055417938266) 增加了 **TurboQuant**，声称其召回率接近标量量化 (scalar quantization)，且 **内存占用减少 2 倍**，同时还增加了内存监控和命名向量生命周期操作。
- **Agent 运行时正成为类似版本控制的基础层**：斯坦福大学的 **Shepherd** 是一个出色的系统构想，由 [@ai_satoru_chan](https://x.com/ai_satoru_chan/status/2054126183374348296) 总结。它将 Agent 执行处理得更像 **Git**：一等任务 (first-class tasks)、效应 (effects)、作用域 (scopes) 和追踪 (traces)；精确重放；分支；回滚；以及在 **Lean** 中的形式化保证。声称的结果包括 CooperBench 上的实时监管增益从 **28.8% 提升至 54.7%**，以及更快的反事实优化 (counterfactual optimization) 和 tree-RL rollout。

**产品与模型发布：多模态、视频、检索与 Embedding**

- **Perceptron Mk1 是该系列中最实质性的新模型发布**：[@perceptroninc](https://x.com/perceptroninc/status/2054216828285796630) 发布了 **Perceptron Mk1**，作为一款面向 **frontier video and embodied reasoning**（前沿视频与具身推理）的模型，支持最高 **2 FPS** 的原生视频、temporal grounding（时间定位）、多模态 in-context learning 以及结构化空间输出。[OpenRouter 的摘要](https://x.com/OpenRouter/status/2054232344148787462)指出其具备 **32k multimodal context**，并支持点、框、多边形和剪辑（clips）等一类输出。该发布与其说是一个通用的 VLM，不如说是一个物理世界推理堆栈。
- **Google 和 Meta 都力推多模态交互层，而非独立的模型规格**：Google DeepMind 的 [AI-enabled mouse pointer demos](https://x.com/GoogleDeepMind/status/2054246119635300451) 将光标重新构想为与 Gemini 绑定的上下文指向接口，允许用户指向屏幕上的内容并口述简短指令。与此同时，Meta 宣布了 [由 Muse Spark 驱动的 Meta AI 语音对话](https://x.com/MetaNewsroom/status/2054205287515484397)，增加了中断、语言切换、图像生成以及基于实时摄像头的交互。
- **Embedding 和 retrieval 模型的更新非常显著**：Jina 发布了 [jina-embeddings-v5-omni](https://x.com/JinaAI_/status/2054226262047301933)，这是一款适用于 **文本、图像、音频和视频** 的通用 embedding 模型，包含 **1.57B** 和 **0.95B** 两个变体，两者都支持 Matryoshka 截断，并向后兼容现有的 v5-text 索引。Meta 低调发布了 [Sapiens2](https://x.com/mervenoyann/status/2054187884417102319)，这是一系列以人为中心的高分辨率 ViTs，参数量跨越 **0.1B→5B**，用于姿态估计、分割、法线（normals）和点图（pointmaps）。
- **Diffusion 和图像工具持续演进**：Hugging Face 的 [Diffusers 0.38.0](https://x.com/RisingSayak/status/2054110949469196748) 增加了新的 pipeline，包括 **Ace-Step 1.5**、**LongCat-AudioDiT** 和 **Ernie-Image**，此外还支持 **Flash Attention 4**、**FlashPack loading** 以及用于 context parallelism 的 **Ring Anything**。其他研究发布包括 [ELF: Embedded Language Flows](https://x.com/iScienceLuvr/status/2054118255778763184)（一种连续空间文本 diffusion 模型），以及腾讯用于像素对齐 3D 生成的 [Pixal3D](https://x.com/_akhaliq/status/2054120807425511826)。

**Agents、工具链与开发者工作流**

- **Agent 产品正从 Demo 转向运营平台**：OpenAI 展示了 [Symphony](https://x.com/OpenAIDevs/status/2054252221941121035)，在这个系统中，**每个开放任务都会分配一个运行中的 Codex agent**；此外还重点介绍了 [Codex 的计算机使用功能 (computer use)](https://x.com/OpenAIDevs/status/2054298427245441141)，使其能在不完全接管的情况下跨应用工作。LangChain 重新开源了其 [改版后的 Chat LangChain 应用](https://x.com/BraceSproul/status/2054231134163321287)，并将其描述为一个生产级问答 Agent，每周处理近 **2T tokens**。
- **长程 Agent 的状态管理正在成为一个一等公民级别的系统问题**：LangGraph 新推出的 [DeltaChannel 快照](https://x.com/sydneyrunkle/status/2054278551244099706) 旨在取代全量状态检查点 (checkpointing)，以实现可扩展的持久执行；LangChain 表示同样的机制现在也为 **deepagents v0.6** 中的消息历史和文件存储提供支持。更广泛的模式也出现在 Google 的 [Gemini Interactions API 指南](https://x.com/_philschmid/status/2054225343251206528)中，加密的 `thought` 签名可以在有状态和无状态模式下的多轮对话中保留推理上下文，而无需开发人员手动管理签名注入。
- **合成数据和 RL 环境生成正在进入工程化阶段**：[@Vtrivedy10](https://x.com/Vtrivedy10/status/2054054238226170361) 提供了一个有用的实践者视角：在大规模情况下，从模型权重中进行有针对性的合成数据提取非常困难，特别是对于长序列等代表性不足的分布；有效的流水线需要程序化测试、验证器 (verifiers)、裁判员 (judges) 以及 Agent 化的长程规划 (agentic long-horizon framing)。在基础设施方面，[Tau2-Infinity](https://x.com/Shahules786/status/2054241505506648161) 通过 DAG 遍历或基于失败假设的场景生成，使 RL 后训练中的困难工具使用任务挖掘实现自动化。
- **热门推文（按互动量排序，已过滤技术相关性）**：
  - **Gemini 作为操作系统层级的智能层**：Google 的 [Gemini Intelligence](https://x.com/sundarpichai/status/2054255858700415005)、[Googlebook](https://x.com/Google/status/2054270454467121187) 以及 [AI 指针演示](https://x.com/GoogleDeepMind/status/2054246119635300451) 共同表明，Agent 化的用户体验 (UX) 正从聊天窗口移向操作系统。
  - **Isomorphic Labs 融资**：[@demishassabis](https://x.com/demishassabis/status/2054197462101889277) 宣布了 **21 亿美元** 的新融资，用于 AI 驱动的药物研发。这是该数据集中与应用 AI 平台直接相关的最大资本投入之一。
  - **语音到语音 (Speech-to-speech) 基准测试**：Artificial Analysis 的 [τ-Voice 基准测试](https://x.com/ArtificialAnlys/status/2054234919887573292) 发现，即使是最好的 S2S 模型也只能解决约 **一半的真实客服场景**，其中 **Grok Voice Think Fast 1.0** 以 **52.1%** 的胜率领先。
  - **Claude Opus 4.7 快速模式 (fast mode)**：Anthropic 的 [快速模式版本](https://x.com/ClaudeDevs/status/2054266327771275435) 已上线 API 和 Claude Code，Cursor 指出其 [速度提升了 2.5 倍，成本增加了 6 倍](https://x.com/cursor_ai/status/2054274305345618163)，这是延迟/价格平衡点上的一个新的具体参考点。

**安全、供应链与更安全的代码编写**

- **最紧迫的运维事件是 Mini Shai-Hulud 供应链攻击**：[@IntCyberDigest](https://x.com/IntCyberDigest/status/2054166749998661659) 报道称，该攻击活动已超出 TanStack，波及了 npm 和 PyPI 上的 **OpenSearch、Mistral AI、Guardrails AI、UiPath 等**，专门针对 **AI 开发者工具 (AI developer tooling)**。值得关注的技术细节是其持久化能力：据称它会钩入 **Claude Code** (`.claude/settings.json`) 和 **VS Code** (`.vscode/tasks.json`)，因此即使删除了相关包，该攻击仍可在未来的工具事件中重新执行。 [Guardrails AI](https://x.com/guardrails_ai/status/2054341322304299086) 随后确认其 **0.10.1** 版本包已受到侵害，并在约 **2 小时**内完成了隔离。
- **可落地的防御措施迅速浮出水面**：[@ramimacisabird](https://x.com/ramimacisabird/status/2054178771180093858) 指出，除了 `minimumReleaseAge` 之外，团队还应启用 **`blockExoticSubdeps`**，以防止远程 GitHub 引用混入依赖图中。[@elithrar](https://x.com/elithrar/status/2054162732195197283) 再次重申，GitHub 的 **`pull_request_target`** 仍然是基于 fork 的 PR 自动化中最危险的 CI/CD 陷阱（footguns）之一。在工作站层面，[@andersonbcdefg](https://x.com/andersonbcdefg/status/2054212574162653535) 建议将凭据（secrets）从无处不在的本地 `.env` 文件移至正式的 secrets manager 中。
- **更安全的代码生成（codegen）正成为一个独立的研究领域**：斯坦福大学相关的 [SecureForge](https://x.com/houjun_liu/status/2054233718269595869) 研究旨在通过 prompt optimization 发现并预防 LLM 生成代码中的漏洞；而[相关的论文列表](https://x.com/FSFG/status/2054196048621367422)则将其定位为 codegen 与安全评估之间的桥梁。更广泛的观点是：coding agents 现在的能力已足够强大，供应链加固和安全生成评估需要被视为核心基础设施（core infra），而非次要问题。


---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. Qwen 3.6 MTP 与长上下文本地评估 (Local Evals)

  - **[Unsloth 上的 MTP](https://www.reddit.com/r/LocalLLaMA/comments/1ta4rvs/mtp_on_unsloth/)** (热度: 727): **这张 [图片](https://i.redd.it/7qopol51pi0h1.png) 是 Hugging Face 的动态截图，显示 **Unsloth AI** 发布/更新了保留 MTP 的 GGUF 构建版本：[`unsloth/Qwen3.6-27B-GGUF-MTP`](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF-MTP) 和 [`unsloth/Qwen3.6-35B-A3B-GGUF-MTP`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF-MTP)。其技术意义在于这些 GGUF 保留了 **MTP / Next-Token-Prediction（下一标记预测）辅助层**，但据用户反馈，目前仍需要检出并构建特定的 **llama.cpp MTP PR**，而不是依赖默认的 llama.cpp 支持。一位评论者遇到了运行时/模型加载断言：`GGML_ASSERT(hparams.nextn_predict_layers > 0 && "QWEN35_MTP requires nextn_predict_layers > 0")`，这表明对于这些 MTP GGUF，工具链或元数据的支持仍然较为脆弱。** 评论者们主要在等待上游推理支持，有人开玩笑说自己在不断刷新 `llama.cpp` 和 `vLLM` 的 GitHub 仓库。对于 llama.cpp 是否“开箱即用”地支持 MTP 也存在不确定性；帖子指出目前尚未支持。

    - 一位编译并运行新款 `27B` GGUF 模型的用户报告了 `qwen35_mtp.cpp` 中的硬断言失败：`GGML_ASSERT(hparams.nextn_predict_layers > 0 && "QWEN35_MTP requires nextn_predict_layers > 0") failed`。这表明加载的 GGUF/模型元数据缺失或未公开 `nextn_predict_layers`，而这是当前实现中执行 **Qwen3.5 MTP** 所必需的。
    - 几位评论者正在关注 **llama.cpp** 和 **vLLM** 是否已合入原生的 **MTP** 支持，其中一人明确询问 llama.cpp 现在是否支持“开箱即用”的 MTP。讨论串暗示各后端的支持仍处于变动中，用户正在密切关注上游仓库以获取与 GGUF MTP 模型的兼容性。
    - 一个技术要点是，**GGUF 中的 MTP 支持**被认为对本地推理至关重要，特别是对于提到的 `35B A3B` 等 Qwen 变体。一位评论者特别指出 `35B A3B` 变体很有趣，因为其预期的上下文长度有所提升。

  - **[Qwen 3.6 35B A3B 的热度是真实的！！！](https://www.reddit.com/r/LocalLLaMA/comments/1t9whrt/the_qwen_36_35b_a3b_hype_is_real/)** (热度: 713): **一位用户在特定领域的“论文转代码”理解任务上对 **Qwen 3.6 35B A3B**、**Qwen 3.6 27B**、**Gemma 4 26B A4B** 和 **Nemotron 3 Nano** 进行了基准测试。该任务通过 Gated Delta Nets、混合 Mamba2 和滑动窗口注意力（Sliding-window Attention）等长上下文机制，向每个模型输入学术论文及配套研究代码。在他们的 [详细发现](https://github.com/nathanlgabriel/paper_code_mapping_assessment/blob/main/README.md) 中，这四个小型/本地开放权重模型的表现均显著优于早期的小模型基准（如 [Devstral Small 2](https://www.reddit.com/r/LocalLLaMA/comments/1ry93gz/devstral_small_2_24b_severely_underrated/)），其中 **Qwen 3.6 35B A3B** 被评为最强；而 Devstral Small 2 无法在 `32GB` 的 VRAM/RAM 中容纳长上下文工作负载。** 评论者注意到了实际的权衡：**Qwen 35B** 在长上下文/重构任务中更受青睐，但在思考模式下可能比较冗长/缓慢，而 **Gemma 26B** 在代码修复/对话方面速度更快；在 `q4` 量化下，一位用户报告 Qwen 35B 占用约 `20GB`，Gemma 26B 占用约 `15GB`，允许两者同时常驻内存。另一位评论者批评该评估未记录推理设置，这限制了结果的可重复性。

    - 几位用户对比了使用 **Gemma 26B** 和 **Qwen 35B** 的本地工作流，指出在 `q4` 量化下两者可以同时保持驻留，因为 Qwen 35B 约为 `20 GB`，Gemma 26B 约为 `15 GB`。一位评论者使用 Gemma 26B 的思考模式进行快速代码修复/对话，使用 Qwen 35B 的思考模式进行长上下文重构，但反映 Qwen 35B 延迟较高，因为在输出最终结果前有过度冗长的推理过程。
    - 一份专注于编程的报告声称，如果先由更强大的模型/编程 Agent 进行初始项目设置引导，**Qwen 27B** 随后可以有效地处理大型项目（`100k+` 行代码）。该用户发现对于他们的用例，Qwen 27B 和 **DeepSeek V4** 之间几乎没有实际差异，尽管 Qwen 偶尔会进入循环，需要手动中断并提供继续提示。
    - 一位评论者强调 **Qwen 27B/35B 的性能对推理配置很敏感**，特别是温度（Temperature）/采样参数，以及应避免对模型权重或 KV Cache 进行过度激进的量化。另一位用户索要缺失的运行设置，暗示如果没有量化级别、采样器设置、上下文长度、后端或硬件等细节，很难评估原始结论。

### 2. 分层内存与高能效本地推理

  - **[使用 Intel Optane Persistent Memory 的电脑组装 - 以超过 4 tokens/sec 的速度运行 1 万亿参数模型](https://www.reddit.com/r/LocalLLaMA/comments/1taeg8h/computer_build_using_intel_optane_persistent/)** (热度: 964): **图片展示了一台使用 Intel Optane DC Persistent Memory DIMM 的高内存 Xeon 工作站/服务器配置的内部结构，印证了帖子的说法：通过 llama.cpp 混合 GPU/CPU 推理，在本地以约 `4 tokens/s` 的速度运行 Kimi K2.5（一个约 `1T` 参数的 MoE 模型）。关键技术点在于将 `768GB` 的 Optane PMem 设置为“内存模式 (Memory Mode)”，此时 Optane 被识别为系统 RAM，而 `192GB` 的 DDR4 ECC DRAM 作为缓存。这使得模型的稀疏专家权重可以驻留在 PMem 中，而 Attention/稠密/共享专家/路由张量则通过 `override-tensor` 或 `ngl auto`/`cmoe` 放置在 **RTX 3060 12GB** 上。[图片](https://i.redd.it/na7zo7lmck0h1.jpeg)** 评论者指出，核心数更多的 Cascade Lake Xeon（如 ES 8260/QQ89）可以提高吞吐量，并讨论了 Optane 的“存储模式 (Storage Mode)”配合 `mmap` 是否可能优于“内存模式”。其他人认为这个配置令人印象深刻，但质疑 `4 tokens/s` 的速度对于交互式使用是否在可接受范围内。

    - 一份详细的硬件笔记建议，与目前的 **Xeon Gold 6246 `12-core`** 相比，使用核心数更多的 Cascade Lake Xeon（例如 **QQ89 ES / Xeon Gold 8260 级别的 `24-core`**）可能会提升性能。评论者还提议测试 Optane PMem 在“存储模式 + `mmap`”与“内存模式”下的基准测试，并指出内存模式将 DRAM 作为透明缓存，且在 CPU 执行前需要将页面交换回 DRAM，因此其延迟并不等同于普通的 RAM 延迟。
    - 一位评论者提供了简明的 Optane PMem 平台兼容性分析：**LGA3647 Skylake/Cascade Lake 使用第一代 Optane `NMA`，运行频率为 `2666 MT/s`**；而 **LGA4189 使用第二代 `NMB`**，在 Cooper Lake 上运行频率为 `2666`，在 Ice Lake 上为 `3200`。他们还提到，在 Cascade Lake 上混合使用 Optane 和 DRAM 可能会导致相关通道降频至 `2666`，且该时期的许多 Xeon 处理器除非使用高内存型号（High-memory SKUs）或更新的平台，否则 DRAM + Optane 的总内存限制为 **`1 TB`**。
    - 一个技术警示被提出：虽然在万亿参数模型上 `~4 tokens/sec` 的生成速度对某些用途是可以接受的，但在这种内存层级结构下，**Prompt 处理/预填充 (Prefill) 速度可能会差得多**。另一条评论估计，整套二手市场组装成本约为 **`$2060–$2500`**，包括 **Xeon Gold 6246**、**TYAN S5630GMRE-CGN**、**RTX 3060 12GB**、`192 GB` DDR4 ECC RDIMM 以及 `768 GB` Intel Optane DCPMM。

  - **[停止浪费电力](https://www.reddit.com/r/LocalLLaMA/comments/1tayu5t/stop_wasting_electricity/)** (热度: 905): **一位用户在 RTX 4090 上对 [`llama.cpp`](https://github.com/ggml-org/llama.cpp) 的 `llama-server` 进行了基准测试。测试使用了 `Qwen3.6-27B-UD-Q4_K_XL.gguf` 模型，开启全 GPU 卸载 (`-ngl all`)、FlashAttention、`q4_0` K/V 缓存量化、`32` 线程以及 `262144` 上下文，并通过 `sudo nvidia-smi -pl N` 调整 GPU 功耗限制。报告显示 GPU 始终处于功耗受限状态，降低功耗限制可以大幅减少电力消耗、发热和噪音，而**解码/token 生成 (`tg`)** 吞吐量几乎没有损失；一位评论者指出，**预填充 (`pp`)** 对功耗更敏感，当功耗从 `450W` 降至 `270W` 时，性能损失约为 `15–20%`（取决于具体模型）。** 评论者主要关注区分**解码与预填充**的行为，因为解码看起来对功耗不敏感，而预填充的性能下降较为明显。一位 RTX 5090 用户表示，出于硬件安全考虑，他们已经限制了功耗，并可能根据这些结果进一步降低功耗。

    - 用户关注点集中在限制 GPU 功耗对性能的影响：**据报道，解码/token 生成 (`tg`) 并不是瓶颈**，而**预填充 (`pp`) 受到的影响较大**。一位评论者量化了这种权衡：将功耗从 **`450W` 降至 `270W`** 时，预填充性能仅损失约 **`15–20%`**（取决于模型），这表明通过激进的功耗限制可以获得显著的效率提升。


### 3. 超小型端侧 Transformer 实验

- **[我让一个真正的 Transformer 语言模型在原装 Game Boy Color 上本地运行了！](https://www.reddit.com/r/LocalLLaMA/comments/1tbi2n3/i_got_a_real_transformer_language_model_running/)** (热度: 368): **图片 ([jpeg](https://i.redd.it/1hl9id7ghs0h1.jpeg)) 展示了一台原装 **Game Boy Color** 正在运行本地 TinyStories Transformer 演示，屏幕显示 `TINYSTORIES Q8 GBC` 和 `Prompt tokenized`。根据帖子介绍，这是将 **Andrej Karpathy 的 TinyStories-260K** 转换为 `INT8`/定点运算，并集成在 **GBDK-2020 MBC5 ROM** 中。由于 GBC 的工作 RAM 极小，权重存储在库切换（Bank-switched）卡带 ROM 中，而 KV cache 存储在卡带 SRAM 中。作者指出运行速度 *极其缓慢* 且由于激进的量化和近似处理，输出大多是乱码，但核心的本地 Transformer prefill + 自回归生成循环在设备上运行正常，无需 PC、手机、Wi-Fi、连接线或云端推理：[github.com/maddiedreese/gbc-transformer](https://github.com/maddiedreese/gbc-transformer)。** 评论大多是热情的赞扬；一位评论者表示这让他们想在 **N64** 上运行模型，另一位则链接了一个相关的恶搞版 Game Boy 语言模型项目 [gbalm](https://code.heni.lol/heni/gbalm)。

    - 一位评论者链接了之前的 Game Boy 语言模型项目 **gbalm** ([代码](https://code.heni.lol/heni/gbalm))，表明此前已有人在任天堂掌机硬件上进行过受限极大的设备端 LM 推理实验。这作为在非 GPU、复古 8 位级系统上实现方法和可行性的对比点具有参考价值。
    - 一个技术问题集中在为什么这里不需要 CUDA/ROCm 风格的 GPU 栈：评论者指出，典型的 LLM 推理通常与成熟的 GPU 编译器相关联，而这个演示运行在性能堪比“土豆”的硬件上。其隐含的观点是，足够微小的 Transformer 模型可以通过手写或高度简化的 CPU 风格推理循环来执行（尽管吞吐量极低），而对于未来国产 GPU 等尚不支持的加速器，其移植性更多取决于是否拥有基础计算后端，而非完整的 CUDA 兼容性。

  - **[Needle：我们将 Gemini 的工具调用能力蒸馏到了一个 26M 模型中](https://www.reddit.com/r/LocalLLaMA/comments/1tb9b0r/needle_we_distilled_gemini_tool_calling_into_a/)** (热度: 271): ****Cactus Compute** 发布了 **Needle**，这是一个采用 MIT 许可证的 `26M` 参数单次（Single-shot）工具调用模型，通过 **Gemini** 生成的数据蒸馏而成。该模型声称在消费级设备上 prefill 速度达 `6000 tok/s`，解码速度达 `1200 tok/s`；权重已托管至 [Hugging Face](https://huggingface.co/Cactus-Compute/needle)，代码和文档位于 [GitHub](https://github.com/cactus-compute/needle)。在架构上，它使用“简单注意力网络（Simple Attention Networks）”——即注意力加门控，**不含 MLP/FFN 层**——理由是函数调用主要是对提供的工具 Schema 进行检索和组装，而非记忆推理；训练过程在 `16 TPU v6e` 上使用 `200B` 预训练 Token 耗时 `27h`，外加 `2B` 合成函数调用 Token 耗时 `45m`（[架构说明](https://github.com/cactus-compute/needle/blob/main/docs/simple_attention_networks.md)）。作者声称它在单次函数调用上超越了 **FunctionGemma-270M**、**Qwen-0.6B**、**Granite-350M** 和 **LFM2.5-350M**，同时也承认这些较大的模型具有更广泛的对话能力。** 评论者认为该模型适合作为轻量级路由，用于分发查询/工具或将任务移交给更大的 LLM，有人询问相同架构是否能支持高质量的摘要任务。由于 Python 特有的依赖性和反序列化安全风险，部分人对上传的 `pickle` 文件表示了技术担忧。

- 一位评论者将 `26M` 蒸馏（distilled）tool-calling 模型定位为一个轻量级的 **router/gating model**：它可以决定查询是否应发送给更大的 LLM 以及使用哪些参数，从而有效地将昂贵的模型调用减少到确实需要的场景中。他们还推测同样的架构是否可以推广到受限的摘要工作流（constrained summarization workflows），尽管帖子中未提供基准测试（benchmark）证据。
- 一条技术讨论链聚焦于作者声称的 **“no FFN”** 结果：对于涉及外部结构化知识的任务，如 **RAG, tool use, 和 retrieval-augmented generation**，如果上下文中已经存在相关事实，模型可能不需要前馈层来存储事实知识。一位评论者将其推演为一个流水线，其中一个小型经过后训练（post-trained）的模型将请求路由到 RAG，然后利用检索到的上下文生成自然语言回答。
- 讨论中提出了几项实现/安全方面的担忧：一位评论者指出，发布 **pickle files** 正日益被避免，因为存在 Python 特有的依赖问题以及反序列化过程中的任意代码执行（arbitrary-code-execution）风险。另一位指出 **Gemini** 存在明显的 tool-calling 怪癖，包括类似系统提示词的关于避免使用 `cat` 而倾向于 `grep_search` 等工具的推理，这提出了一个可能性：如果清理不当，蒸馏数据集可能会继承特定供应商的工具使用偏好。

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude Coding Workflows and Tooling

- **[从一位“氛围工程师”（Vibe Engineer）手中继承了一个 3 个月大的仓库。写下了职业生涯中最令人满意的 PR](https://www.reddit.com/r/ClaudeCode/comments/1tb7edc/inherited_a_3month_old_repo_from_a_vibe_engineer/)** (热度: 3672)：**图片是一个 GitHub 风格的 diffstat，显示了一个清理 PR，其中包含 **`+10,197` 次添加** 和 **`−3,618,778` 次删除** ([图片](https://i.redd.it/izgrhw5tgq0h1.png))，为文中声称的重写 3 个月大的 “vibe-coded” 后端仓库提供了背景。作者表示，继承的仓库拥有 **`309k` 行代码**、**`240k` 行文档**、**100万+行 markdown 日志**、`220` 个处理句柄（handlers）但仅使用了约 `20` 个，以及 `40+` 个密钥（secrets）但仅需 `2` 个；他们在一周内使用 Claude 重写了它，在保留功能的同时增加了更整洁的架构和集成测试。** 评论者将此定性为围绕 AI/agent 生成代码的后续维护问题，有人预言 *“修复 vibe-coded 乱象”* 可能会成为一条利润丰厚的职业道路。该讨论还质疑了复杂的 agent 知识库和自动生成的文档是否真的能显著改善开发，还是仅仅营造了生产力的假象。

    - 一位评论者预测，修复 AI/“vibe-coded” 仓库可能会成为一种有价值的专业化方向，暗示 agentic coding 带来的短期生产力可能会产生下游的维护债务。他们还认为，围绕 “vibecoding” 的大部分热情来自于 *非软件专业人士*，这表明 demo 级输出与生产级工程标准之间存在差距。

- **[Clawdmeter - 一个小型 ESP32 使用限制监控器（源代码在描述中）](https://www.reddit.com/r/ClaudeCode/comments/1takxpl/clawdmeter_a_small_esp32_usage_limit_monitor/)** (热度: 1677)：**图片展示了 **Clawdmeter**，一个基于 ESP32 的小型桌面显示器，显示了 Claude/Anthropic 的使用限制、重置计时器和进度条，与帖子中描述的价值 `$32` 的 Waveshare ESP32 开发板及 `480×480` AMOLED 显示屏相符。该项目已在 [GitHub](https://github.com/HermannBjorgvin/Clawdmeter) 上开源，图中设备以紧凑的物理仪表盘形式可视化了当前和每周的配额状态：[图片](https://i.redd.it/aqoo7y4nkl0h1.jpeg)。** 评论大多比较轻松，用户开玩笑说 Anthropic 应该免费分发这些设备，并且这可能会增加 *“Claude 使用焦虑”*。一位评论者还表示有兴趣将同样的低成本 ESP32 显示平台用于其他定制化的智能家居状态设备。

    - 一位评论者建议将 ESP32 监控器从实时的配额显示扩展为一个小型遥测设备，记录 **随时间变化的使用历史**。他们特别希望能够跟踪单条命令的影响并查看图表，以验证 Claude 的使用量消耗是否快于预期。
    - 另一个技术角度是，同样的低成本 ESP32 风格硬件平台是否可以复用于其他 **自定义、小众的智能家居状态显示器或监控器**。该评论将此设备定位为通用的环境信息装置，而不仅仅是 Claude 的配额计量器。

### 2. AI 现实部署中的失败模式 (AI Deployment Failure Modes in the Wild)

  - **[ChatGPT 现在正在为教科书创作内容。](https://www.reddit.com/r/singularity/comments/1ta1dvl/chatgpt_is_now_creating_content_for_textbooks/)** (Activity: 5865): **图片似乎展示了一个 **DBMS 教科书页面**，其中错误地保留了一个 AI 助手风格的句子——*“如果你愿意，我也可以解释……”*，这意味着 ChatGPT 或类似的 LLM 可能被用于起草教科书内容，但缺乏足够的人工审核。这不是一个技术基准测试或实现帖子；其重要性在于背景：它突显了教育材料中可能存在的 **AI 生成内容痕迹**。[图片](https://i.redd.it/d65cfdtf1i0h1.png)** 评论者批评了编辑审核的缺失，并认为面向学生的 AI 生成教育内容在机构、教职员工和外包供应商中正变得普遍。一位评论者还指出，可见的注释可能是用 Gemini 或其他工具编辑的，但核心担忧仍然是教科书文本本身似乎未经审核。

    - 一位评论者根据与教育机构直接工作的经验声称，**面向学生的 AI 生成内容正变得无处不在**，涵盖了教职员工和外包教育内容供应商，这意味着从孤立的使用转向了机构规模的生产工作流。
    - 一项技术观察指出，由于存在 **水印去除痕迹**、文本超出页面边缘，以及某人使用 Gemini 添加框/箭头注释时引入的潜在 **SynthID/Gemini 出处标记**，该图像很可能是 AI 编辑/生成的。另一位评论者指出，如果没有具体的教科书引用，整个截图本身也可能是 AI 生成的，而非来自真实书籍。

  - **[我为我的婚礼宾客制作了一个 AI 礼宾。他们做的第二常见的事情就是尝试对其进行越狱。](https://www.reddit.com/r/ClaudeAI/comments/1tatxnq/i_made_an_ai_concierge_for_my_wedding_guests_the/)** (Activity: 1667): **[图片](https://i.imgur.com/8n0k4Ve.jpeg) 是一个在毛里求斯目的地婚礼上使用的自定义 **AI 婚礼礼宾** 的信息图报告：`29` 名用户生成了 `719` 个会话和 `8,678` 条消息。其使用情况细分在真实世界 Chatbot 部署行为中值得关注：`35%` 为真诚的物流问题，`25%` 为越狱/黑客攻击尝试，外加文化翻译、闲聊和杂项请求；创作者表示它通过 **MCP server** 连接到 API，为宾客检索婚礼信息。** 评论者发现该项目比通用的 Chatbot 演示更有趣，但对仅 `29` 人产生的消息量以及宾客尝试越狱的频率感到惊讶。

    - OP 描述了构建两个相关系统：一个是为在 **毛里求斯** 举办的目的地婚礼准备的筹备助手，另一个是面向宾客的 AI 礼宾，通过 **MCP server** 连接到外部 API，为用户检索活动/旅行信息。帖子里一个值得注意的使用统计数据是，仅 `29` 名宾客就生成了 **超过 `8,000` 条消息**，帖子标题指出尝试越狱是第二常见的行为。
    - 一位评论者提出了关于可观测性（Observability）和日志的实现/隐私担忧：宾客是否意识到创作者可以阅读他们与礼宾的对话。这对于任何构建小型活动 AI 助手的人来说都是相关的，因为即使在非企业部署中，聊天记录留存、管理员访问权限和知情同意也可能成为重要问题。

# AI Discords

遗憾的是，Discord 今天关闭了我们的访问权限。我们不会以这种形式恢复它，但我们很快就会发布新的 AINews。感谢阅读到这里，这是一段美好的历程。