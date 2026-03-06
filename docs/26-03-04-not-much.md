---
date: '2026-03-04T05:44:39.731046Z'
id: MjAyNi0w
title: 在此处填写标题
---

**TODO: ONELINE SUBTITLE**

> AI News for 3/3/2026-3/4/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and 24 Discords (**264** channels, and **14242** messages) for you. Estimated reading time saved (at 200wpm): **1397** minutes. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap

**Frontier model shipping: Gemini 3.1 Flash-Lite, GPT-5.4 rumors, and “agent-first” product positioning**

- **Gemini 3.1 Flash-Lite positioning (speed/$)**: Demis Hassabis teased **Gemini 3.1 Flash-Lite** as “incredibly fast and cost-efficient” for its performance—clearly framing the model line around latency and cost per capability rather than raw frontier scores ([tweet](https://x.com/demishassabis/status/2029047252275060895)). Related product chatter highlights **NotebookLM** as a “favorite AI tool” ([tweet](https://x.com/demishassabis/status/2029355691933085731)) and a major new **NotebookLM Studio** feature: **Cinematic Video Overviews** that generate bespoke, immersive videos from user sources for Ultra users ([tweet](https://x.com/NotebookLM/status/2029240601334436080)).
- **GPT-5.4 leak narrative (The Information)**: Multiple tweets amplify a report that **GPT-5.4** is coming with a **~1M token context window** and a new **“extreme reasoning mode”** that can “think for hours,” targeting long-horizon agentic workflows and lower complex-task error rates ([tweet](https://x.com/kimmonismus/status/2029213568155992425), [tweet](https://x.com/steph_palazzolo/status/2029212039760023941), [tweet](https://x.com/scaling01/status/2029215437922169254)). There’s also speculation that OpenAI is shifting to **more frequent (monthly) model updates** ([tweet](https://x.com/kimmonismus/status/2029223828677599244)). Separately, one arena watcher claims “GPT-5.4 landed in the arena,” implying an imminent release window ([tweet](https://x.com/kimmonismus/status/2029325405212070200)). Treat all of this as **unconfirmed** unless corroborated by OpenAI.
- **Claude as “agent behavior” leader, not just coding**: Nat Lambert argues the discussion should shift from Anthropic “going all-in on code” to their lead on **general agent behavior**, implying coding capability will commoditize but agent robustness will not ([tweet](https://x.com/natolambert/status/2029212769648836806)). MathArena evaluation adds a datapoint: **Claude Opus 4.6** is strong overall but weak on **visual mathematics**, and costly to evaluate (claimed ~$8k) ([tweet](https://x.com/j_dekoninck/status/2029160582687985727)).

---

**Alibaba Qwen “shakeup”: org design, compute access, and open-model dependency**



- **领导层离职 + 重组传闻**：整个数据集的一个核心线索是 **Qwen 负责人林俊漾 (Lin Junyang)** 在据传的内部重组中卸任。该重组将团队从垂直整合模式转为**水平拆分**（预训练/后训练/多模态/基础设施），这削弱了统一控制，并可能与团队此前倡导的“紧密集成”哲学相冲突（[推文](https://x.com/ZhihuFrontier/status/2029117410259993073)，以及后续背景[推文](https://x.com/ZhihuFrontier/status/2029120535599431797)）。Simon Willison 总结了这一情况，并指出在 **Qwen 3.5** 发布前后出现了多起明显的辞职事件（[推文](https://x.com/simonw/status/2029223704127828386)）。
- **紧急全体会议与“算力讽刺”**：由 Poe Zhao 转述的报告描述了阿里巴巴 CEO **吴泳铭 (Eddie Wu)** 召开了一次紧急会议；Qwen 团队成员就**重组、算力分配和模型策略**向领导层发难。最尖锐的细节是：据称阿里巴巴云的 CTO 承认，**外部客户获取算力的顺畅程度甚至高于内部 Qwen 团队**（[推文](https://x.com/poezhao0605/status/2029151951167078454)）。这引发了此前认为 Qwen 拥有“GPU-rich”（GPU 资源充裕）地位的观察者的重新评估（[推文](https://x.com/teortaxesTex/status/2029159237729894727)）。
- **Qwen 在研究工作流中的主导地位**：根据一项对 Hugging Face (HF) 论文使用情况的总结声称，Qwen 是 **2025–2026 年 HF 论文中排名第一的开源模型**，在 7,692 篇论文中占比 **41%**，在 2025 年 5 月 Qwen3 发布前后占比约为 **50%**（[推文](https://x.com/teortaxesTex/status/2029102932604375057)）。无论确切数字是否完全准确，其核心观点是不变的：**生态系统对一个小型核心团队的依赖**是一个真实存在的风险。
- **权重开放（Open-weights）模型的生存风险框架**：Nat Lambert 认为，权重开放的前沿工作可能会集中在少数具有商业动机的参与者身上：**非营利组织、NVIDIA（硬件带动销售）和 Meta（使补充产品商品化）**——这一视角使得 Qwen 的公司战略失调看起来像是结构上的必然，而非个别异常（[推文](https://x.com/natolambert/status/2029049751472357631)）。
- **来自 Qwen 生态的模型/基础设施技术笔记**：RASBT 指出 **Gated DeltaNet 模块**可以避免 KV-cache 的增长，在宣称的比例下使 **Qwen 3.5** 比 Qwen3 对内存更友好（[推文](https://x.com/rasbt/status/2029233742708130265)）。同时，有用户反映即使在较高量化（quants）下，Qwen 的采样参数在约 20% Context 时会出现 **llama.cpp 死循环**（[推文](https://x.com/qtnx_/status/2029246416342618321)）——这提醒人们“推荐解码（decoding）”配置在不同 Runtime 之间可能是脆弱的。

---

**推理与系统：Speculative Speculative Decoding、vLLM 扩展以及内核生成 Agent**

- **Speculative Speculative Decoding (SSD)**：Tanishq Kumar 推出了 **SSD**，声称其比领先的推理引擎（**vLLM, SGLang**）快 **2 倍**，该项目是与 Tri Dao 和 Avner May 合作完成的（[推文](https://x.com/tanishqkumar07/status/2029251146196631872)；Avner 的发布[推文](https://x.com/avnermay/status/2029251985934041232)）。Tri Dao 将其描述为“异步机器的攻击”，将该方法与 GPU Kernel 异步设计的经验联系起来（[推文](https://x.com/tri_dao/status/2029273056364118407)）。如果得到证实，这将是该领域中较为具体的一个算法“提速”案例。
- **生产级推理实践**：一份关于在 OOM（内存溢出）/不稳定情况下扩展 **vLLM** 的实用指南被广泛分享，该指南强调了 **工作负载分析（Workload Profiling） + 配置微调** 比单纯堆硬件更重要（[推文](https://x.com/DylanCouzon/status/2029208629312700592)）。
- **针对 CUDA 内核的 Agentic RL（字节跳动）**：字节跳动的一篇论文总结描述了 **CUDA Agent**：一个在安全测试环境中编写 CUDA Kernel 的 Agent 强化学习（RL）设置，旨在优化提速性能；声称在某些情况下，其生成的组件比传统自动化工具快约 **100%**（[推文](https://x.com/rohanpaul_ai/status/2029161433519567175)）。即便考虑到“推文摘要的水分”，这种**闭环 代码→基准测试→奖励**的性能工程研究方向是可信且具有战略意义的。

---

**编程 Agent 与开发工具：Windows 上的 Codex、VS Code “Agent DX”、Symphony、LangSmith Skills**

- **Codex 应用登陆 Windows + 开源沙箱**：OpenAI DevRel 宣布推出 **Codex for Windows**，并配备了 **Windows 原生 Agent 沙箱**，利用 OS 控制机制（受限 tokens、ACLs、专用用户）来约束文件系统/网络访问，除非获得批准；该实现已 **开源**（[推文](https://x.com/OpenAIDevs/status/2029252453246595301)，[推文](https://x.com/OpenAIDevs/status/2029252477179314350)）。AJ Ambrosino 补充了细节：支持原生运行或通过 WSL 运行；支持 PowerShell/CMD/Git Bash/WSL 终端；具备 “Open in ...” 集成和 Windows 技能（[推文](https://x.com/ajambrosino/status/2029252598851879265)）。Reach_vb 强调开源沙箱是一个被低估的成果（[推文](https://x.com/reach_vb/status/2029335011804017135)）。
- **VS Code 面向 Agent 的版本发布**：`@code` 账号强调了 “Agents，为了真实工作”，发布了 **hooks**、**消息转向/排队 (message steering/queueing)**、**集成式 Agent 浏览器** 以及 **共享内存**（[推文](https://x.com/code/status/2029279963778515372)）。对于开发者来说，一个流程变化非常重要：VS Code 正在从每月发布改为 **每周发布** `main` 分支，以加速功能交付（[推文](https://x.com/pierceboggan/status/2029283603801358798)）。
- **OpenAI Symphony (工单看板→Agent 编排)**：一个新的 OpenAI 仓库 **Symphony** 被描述为一个编排层，它能够 **轮询项目看板** 并根据工单 (ticket) 生命周期阶段生成 Agent——将用户体验从 “提示 Agent” 转变为 “移动工单并让 Agent 执行”（[推文](https://x.com/scaling01/status/2029261034993684952)）。这与 **工作流原生 Agent 自动化** 的大趋势相一致。
- **LangSmith Skills + CLI (Agent 执行 Agent 工程化)**：LangChain 发布了 **LangSmith Skills + CLI**，使编码 Agent 能够原生调试 trace、构建数据集，并从终端运行实验（[推文](https://x.com/LangChain/status/2029272199073354105)）。与此同时，**LangChain OSS Skills** 旨在教导 Agent 如何有效地使用 LangChain/LangGraph/DeepAgents（[推文](https://x.com/LangChain_OSS/status/2029272669942673436)，[推文](https://x.com/hwchase17/status/2029274371710501049)）。
- **Cursor 通过 Agent Client Protocol 进入 JetBrains**：Cursor 宣布通过 **Agent Client Protocol** 在 **JetBrains IDEs** 中可用（[推文](https://x.com/cursor_ai/status/2029222015736197205)）。这是一个关键的分发举措：实现 IDE 原生访问而无需强制用户切换工具。

---

**多模态 + 世界模型：Self-Flow、超越语言建模、持久化视频和 NE-Dreamer**

- **Black Forest Labs 的 Self-Flow**：BFL 预览了 **Self-Flow**，一种用于多模态生成模型（图像/视频/音频/文本）的 **自监督流匹配 (self-supervised flow-matching)** 方法，避免依赖外部预训练表示模型（如 DINO）。声称的结果包括：**收敛速度提升高达 2.8 倍**，改进了视频时间一致性，更清晰的排版；被定位为多模态视觉智能甚至动作预测的基础（[推文](https://x.com/bfl_ml/status/2029212134023020667)；更多背景信息见 [推文](https://x.com/robrombach/status/2029272803099226425)）。
- **“超越语言建模” / 视觉优先的多模态预训练**：多位作者推广了一篇探索 **原生多模态模型** 的论文，其中视觉被视为一等公民，模型以 “Transfusion 风格” 输入/输出所有模态，包括对表示、数据、世界建模、架构和 Scaling Laws 的讨论（[推文](https://x.com/__JohnNguyen__/status/2029236083914096756)，[推文](https://x.com/TongPetersb/status/2029237530160169286)，[推文](https://x.com/DavidJFan/status/2029239760301035549)）。核心观点是：业界可能低估了多少进展需要 **视觉原生训练**，而非以语言为主的适配器。
- **长上下文视频世界模型**：Gordon Wetzstein 的推特线索预告了 “Mode Seeking meets Mean Seeking (MMM)”，作为通过统一表示实现 **长上下文、持久化视频世界模型** 的路径（[推文](https://x.com/GordonWetzstein/status/2029054374459376026)）。
- **NE-Dreamer：嵌入预测而非像素重建**：George Bredis 介绍了 **NE-Dreamer**，探索训练世界模型来 **预测下一个嵌入 (next embeddings)** 而非重建像素——认为重建对于控制任务来说可能是错误的目标（[推文](https://x.com/BredisGeorge/status/2029190420790411671)）。

---

**评估、记忆与 “以人为本” 的编码：分解障碍、Agent 记忆诊断、臃肿补丁和准则漂移**

- **Diffusion LLM 并行化遭遇“因子化壁垒”**：Ian Li 解释了为什么 diffusion LLMs 在并行 token 生成方面表现挣扎：同时预测多个 token 可能会引发不连贯的联合输出（例如 “San York”）。他将其归因于结构性错误设定——全因子化输出头在不爆炸式增加输出头尺寸的情况下，无法表示完整的联合分布——并提出了 **CoDD** 作为打破这一壁垒的方法 ([推文](https://x.com/IanLi1118/status/2029074519223353362))。
- **Agent 内存：检索主导“写入”策略**：一个诊断框架区分了**检索失败与利用失败**；关键观点：检索方法导致了 **约 20 个百分点** 的方差，而内存写入方法仅产生 **3–8 个百分点** 的变化。“原始分块 (Raw chunking)”的效果可以媲美或优于昂贵的摘要/事实提取流水线 ([推文](https://x.com/dair_ai/status/2029202969456234562))。实际意义：许多团队可能在过度优化内存“摄取 (ingestion)”，而不是搜索/选择。
- **SWE-bench 补丁膨胀作为一种人为因素失败模式**：KLieret 报告称，LLM 生成的 SWE-bench 补丁始终比人类方案更**长且冗余**（不仅仅是注释），这虽然能通过测试，但会损害人工验证和维护 ([推文](https://x.com/KLieret/status/2029219763423986030))。后续研究强调“测试成功 != 实际可用性”，并主张开展**以人为中心的编码 Agent 研究** ([推文](https://x.com/ZhiruoW/status/2029229015634993579))。
- **评估准则漂移与作为“生命系统”的评测**：多条推文强调，失败通常源于过时的**评估准则 (eval rubric)**，而非“失效的提示词 (broken prompt)”；解决办法是将评测视为一个与生产环境分布偏移挂钩的反馈循环，而非静态的单元测试 ([推文](https://x.com/omarsar0/status/2029225624825659668), [推文](https://x.com/kimmonismus/status/2029227463805378571))。
- **BullshitBench v2（废话检测）**：一个测试模型是否**拒绝荒谬提示词**的基准测试发现，只有 **Claude** 和 **Qwen 3.5** 的得分显著高于 **60%**，并观察到一种失败模式：“思考更久”的推理模型会**为荒谬内容寻找合理化解释**，而不是拒绝它 ([推文](https://x.com/kimmonismus/status/2029230388028358726))。如果属实，这将是一个有用的权衡指标，用于制衡将纯粹的“推理 token”数量作为质量指标的做法。

---

**热门推文（按参与度排序，技术相关）**

- **NotebookLM 电影级视频概览**上线（Ultra 用户）：[@NotebookLM](https://x.com/NotebookLM/status/2029240601334436080)  
- **Windows 版 OpenAI Codex 应用** + Windows 原生沙箱细节：[@OpenAIDevs](https://x.com/OpenAIDevs/status/2029252453246595301) 和 [@ajambrosino](https://x.com/ajambrosino/status/2029252598851879265)  
- **Gemini 3.1 Flash-Lite** 的速度/成本定位：[@demishassabis](https://x.com/demishassabis/status/2029047252275060895)  
- **Speculative Speculative Decoding (SSD)** 声称推理速度提升高达 2 倍：[@tanishqkumar07](https://x.com/tanishqkumar07/status/2029251146196631872)  
- **Yuan 3.0 Ultra** 开源多模态 MoE（总参数 1010B / 激活参数 68.8B）发布公告：[@YuanAI_Lab](https://x.com/YuanAI_Lab/status/2029204213180580229)  
- **Self-Flow** 多模态流匹配研究预览（声称收敛速度提升 2.8 倍）：[@bfl_ml](https://x.com/bfl_ml/status/2029212134023020667)

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Qwen 模型性能与基准测试

  - **[Qwen3.5-35B-A3B 在 SWE-bench Verified Hard 上达到 37.8% —— 配合正确的验证策略，几乎追平 Claude Opus 4.6 (40%)](https://www.reddit.com/r/LocalLLaMA/comments/1rkdlqi/qwen3535ba3b_hits_378_on_swebench_verified_hard/)** (热度: 464): **该帖讨论了 **Qwen3.5-35B-A3B** 模型的性能，这是一个在 SWE-bench Verified Hard 任务中具有 `3B 活动参数` 的小型 MoE 模型。通过实施一种简单的验证策略 —— “每次编辑后验证”，该模型的性能从 `22%` 提升到了 `37.8%`，几乎追平了 **Claude Opus 4.6** 的 `40%`。该策略涉及提示模型在每次 `file_edit` 后通过编写和运行测试脚本来验证更改。该模型在完整的 500 个任务基准测试中达到了 `67.0%`，可与更大的系统相媲美。作者指出，像 MCTS 和 Best-of-N 采样这样更复杂的策略效果反而较差。文中提供了包含代码和日志的 [GitHub 仓库](http://github.com/SeungyounShin/agent-verify)。** 一位评论者建议等待 SWE-bench 的新任务，以避免模型训练中潜在的数据泄露。另一位对比结果表示怀疑，认为它们可能是“针对基准测试过度优化（benchmaxed）”的。第三位评论者注意到该策略中缺乏循环，他们发现这在 35B 模型上极具挑战性。

    - ResidentPositive4122 强调了 SWE-bench 的一个潜在问题，指出它已经过时，并且可能在较新模型的训练数据中包含泄露信号。他们建议等待包含新任务的更新版本，以确保评估更加准确。
    - Deep_Traffic_7873 声称 Qwen3.5-35B-A3B 在其个人基准测试中优于 GPT-OSS-20B，表明前者在特定任务中相对于后者具有显著的性能优势。
    - ethereal_intellect 提供了 OpenAI 对其 Codex harness 环境指南的详细列表，其中包括验证代码库、复现 Bug 和实施修复等步骤。他们指出，诸如伪造视频和驱动应用程序之类的某些任务特别具有挑战性，但在精心设置下是可行的。

  - **[Qwen3.5-27B Q4 量化对比](https://www.reddit.com/r/LocalLLaMA/comments/1rk5qmr/qwen3527b_q4_quantization_comparison/)** (热度: 386): **该帖对 Qwen3.5-27B 模型的 Q4 量化方法进行了详细对比，重点关注相对于 BF16 基线的平均 KL 散度 (KLD)。评估使用了自定义聊天数据集和 Wikitext2，结果显示 `unsloth_Qwen3.5-27B-UD-Q4_K_XL` 量化实现了最低的 KLD `0.005087`，而 `bartowski_Qwen3.5-27B-IQ4_XS` 则以其 `0.317506` 的效率得分受到关注。分析使用 `llama.cpp` 进行评估，并强调了 KLD 作为衡量与原始模型概率分布忠实度指标的重要性。该帖还提供了一个用于 KLD 扫描脚本的 GitHub 链接，尽管备注称其未经过广泛测试。** 一条著名的评论质疑了帖子与 Hugging Face 之间模型大小的差异，暗示量化方法或报告方式可能存在差异。另一条评论建议，在大小与 KLD 的关系图中靠近最佳拟合线的模型更可取，表明更倾向于平衡大小和准确性的模型。

    - Gueleric 提出了一个关于 `bartowski_Qwen3.5-27B-IQ4_XS` 模型大小差异的技术问题，注意到报告的 14.1GB 大小与 Hugging Face 上列出的 15.2GB 大小之间存在差异。这可能是由于不同的量化方法或 Hugging Face 模型大小中包含的元数据导致的。
    - PaMRxR 讨论了他们创建的一张图表，显示了 Qwen3.5-27B 模型的量化大小与 KL 散度 (KLD) 之间的关系。他们提到移除离群值以更好地拟合数据，表明靠近最佳拟合线的模型更具优势。该图表是使用 `unsloth_Qwen3.5-27B-UD-Q4_K_XL` 模型生成的，表明其重点在于理解模型大小与 KLD 等性能指标之间的权衡。
    - munkiemagik 表示有兴趣对不同参数和量化水平的模型进行定性对比。他们强调了模型测试中的一个常见问题：通常只报告 perplexity（困惑度）或 throughput（吞吐量）等特定指标，这可能与用户的实际需求不符。他们还提到了理解 KL 散度等技术概念的挑战，表明需要更深入地参与大语言模型背后的学术原理。

### 2. Qwen Model Usability and Applications

  - **[Qwen3.5-0.8B - Who needs GPUs?](https://www.reddit.com/r/LocalLLaMA/comments/1rkjsaj/qwen3508b_who_needs_gpus/)** (Activity: 646): **The image highlights the impressive capability of the `Qwen3.5-0.8B` model, which can run efficiently on outdated hardware, specifically a 2nd generation i5 processor with 4GB DDR3 RAM. This model is executed using `llama.cpp`, a tool for running large language models on local machines, and is shown to handle complex topics like string theory. The system information is displayed using `fastfetch` on an Arch Linux setup, emphasizing the model's low resource requirements and accessibility for users without high-end GPUs.** Commenters express amazement at the model's performance on such old hardware, comparing it to the capabilities of GPT-3 and noting the open-source nature of the model. There's also a nostalgic mention of semi-transparent terminals, reflecting on past desktop environments.

    - The Qwen3.5-0.8B model is notable for its ability to run efficiently without the need for a GPU, which is a significant advancement in making AI more accessible. This model is open-source, allowing for broader experimentation and use in various applications without the high cost of GPU resources.
    - A user suggests using the Qwen3 8B model instead, highlighting its superior performance and the fact that it also does not require a GPU. This suggests that the Qwen3 series is optimized for performance on lower-end hardware, making it a practical choice for developers without access to high-end computing resources.
    - The Qwen3.5-0.8B model includes a vision component, which allows it to analyze images and generate workflows that can produce images or videos. This feature expands its utility beyond text-based tasks, enabling it to function as a sub-agent in multimedia applications.

  - **[Qwen 3.5 4b is so good, that it can vibe code a fully working OS web app in one go.](https://www.reddit.com/r/LocalLLaMA/comments/1rkb8en/qwen_35_4b_is_so_good_that_it_can_vibe_code_a/)** (Activity: 718): **The post discusses the capabilities of **Qwen 3.5 4b**, a compact AI model, which successfully created a fully functional web-based operating system (OS) from a single prompt. The OS includes features such as two games, a text editor, an audio player, a file browser, customizable wallpaper, and a special feature chosen by the model itself. The model's ability to generate a working OS with these specifications highlights significant advancements in AI model efficiency and information density, particularly for a model of only `4 billion parameters`. The OS can be accessed [here](https://qwen4bwebos.tiiny.site/).** Commenters express skepticism about the test's validity, suggesting it may be a common benchmark scenario potentially optimized for success. Others are impressed by the model's performance, noting the significant progress in AI capabilities beyond mere scaling.

    - **tinny66666** highlights the impressive performance of the Qwen 3.5 4b model, noting that its intelligence surpasses the original GPT-3.5 despite its smaller size. This suggests a significant improvement in information density and model efficiency, raising questions about the potential limits of such advancements.
    - **msixtwofive** expresses skepticism about the validity of the test, suggesting that the task of creating a fully working OS web app is a common benchmark that may have been optimized for by AI influencers. This raises concerns about the authenticity of the model's performance in real-world, unseeded scenarios.
    - **simracerman** points out that while the Qwen 3.5 4b model's ability to complete the task is impressive, especially compared to larger models, there is a possibility that the code for such tasks might be included in the training data, which could influence the model's performance.




### 3. Tech Industry Developments and Reactions

  - **[Apple unveils M5 Pro and M5 Max, citing up to 4× faster LLM prompt processing than M4 Pro and M4 Max](https://www.reddit.com/r/LocalLLaMA/comments/1rjqsv6/apple_unveils_m5_pro_and_m5_max_citing_up_to_4/)** (Activity: 998): **The image illustrates the capabilities of Apple's newly announced M5 Pro and M5 Max chips, which are claimed to process large language model (LLM) prompts up to 4 times faster than their predecessors, the M4 Pro and M4 Max. The M5 Pro supports up to 64GB of unified memory with a bandwidth of 307GB/s, while the M5 Max supports up to 128GB of unified memory with a bandwidth of 614GB/s. Additionally, these chips feature up to 2× faster SSD speeds at 14.5GB/s and include the Apple N1 wireless chip for Wi-Fi 7, enhancing download speeds if compatible with the router.** Some commenters express a desire for a Mac Studio equipped with the new chips, while others note the lack of mention of AI-specific silicon improvements, such as a Neural Accelerator.

    - The M5 Pro and M5 Max chips feature significant improvements in memory capabilities, with the M5 Pro supporting up to 64GB of unified memory and 307GB/s of memory bandwidth, while the M5 Max supports up to 128GB of unified memory and 614GB/s of memory bandwidth. These enhancements are crucial for handling large-scale machine learning models and intensive computational tasks.
    - The new chips also boast up to 2× faster SSD speeds, reaching 14.5GB/s, which can significantly reduce data access times and improve overall system performance. Additionally, the inclusion of the Apple N1 wireless chip for Wi-Fi 7 support offers faster download speeds, provided the network infrastructure can support it, enhancing connectivity for data-intensive applications.
    - Despite expectations for more advanced AI-specific silicon, the M5 series still offers substantial performance gains, particularly in LLM prompt processing, which is up to 4× faster than the previous M4 series. This improvement is likely due to a combination of increased memory bandwidth and faster SSD speeds, which together enhance the chips' ability to handle complex AI workloads efficiently.

  - **[ChatGPT uninstalls surged by 295% after Pentagon deal](https://www.reddit.com/r/LocalLLM/comments/1rjlzgy/chatgpt_uninstalls_surged_by_295_after_pentagon/)** (Activity: 418): **The image is a meme and does not provide any technical insights or verifiable data. It humorously suggests a significant increase in ChatGPT uninstalls following a supposed deal with the Pentagon, but lacks any credible sources or detailed information to support this claim. The comments reflect skepticism about the validity of the claim, questioning the source and the actual impact on user numbers.** Commenters express skepticism about the claim, questioning the source and the actual impact on user numbers, suggesting it might be exaggerated or unsourced.

    - A user questions the validity of the claim regarding the surge in uninstalls, asking if the statistic is unsourced, which raises concerns about the reliability of the data. This highlights the importance of verifying claims with credible sources, especially when discussing significant changes in user behavior.
    - Another comment critiques the shift in OpenAI's mission from a 'non-profit research lab' to potentially acting as a 'Defense Contractor.' This reflects a broader debate on the ethical implications of AI development and its alignment with military applications, suggesting a tension between original mission statements and current business practices.
    - A user discusses the inevitability of AI's integration into military applications, arguing that technological advancements naturally lead to such outcomes to maintain competitive advantage. This comment underscores the strategic importance of AI in defense and the potential consequences of falling behind in technological capabilities.


## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. AI Model and Benchmark Releases



- **[Opus 4.6 解决了 Donald Knuth 在编写 "The Art of Computer Programming" 时提出的一个猜想，他对此感到非常兴奋](https://www.reddit.com/r/singularity/comments/1rkhady/opus_46_solved_one_of_donald_knuths_conjectures/)** (Activity: 1124): **该图片是一份由 **Donald Knuth** 撰写的名为 "Claude’s Cycles" 的文档，讨论了 AI 模型 **Claude Opus 4.6** 取得的重大突破。这个模型是一个混合推理系统 (hybrid reasoning system)，解决了一个与 directed Hamiltonian cycles 以及将弧分解为有向循环相关的长期猜想，这也是 Knuth 一直在研究的课题。文档强调了 Knuth 对 AI 解决方案的惊讶和喜悦，标志着在 automatic deduction 和创造性问题解决方面的显著进步。全文可在此查看 [here](https://www-cs-faculty.stanford.edu/~knuth/papers/claude-cycles.pdf)。** 评论者们对 Knuth 愿意修正其对 AI 看法的开放态度表示赞赏，并指出了他的学术诚信。他们还强调了 **Anthropic** 凭借 Claude Opus 4.6 所取得的成就，并庆祝 Knuth 在 88 岁高龄仍持续积极参与研究。

    - 在论文中指出，AI 模型 Claude 通过解决 Knuth 猜想中 `m` 为奇数的情况，并找到了某些偶数 `m` 的解，展示了其能力，尽管它未能概括出所有偶数 `m` 的通用解。这凸显了该模型快速探索多种方法的能力，而这项任务对人类数学家来说非常耗时。
    - Donald Knuth 对 AI 成就的认可标志着他对 generative AI 看法的重大转变。Knuth 此前持怀疑态度，现在他认识到 AI 能力的飞速进步，尤其是在 automatic deduction 和创造性问题解决方面，正如 Claude 在解决其猜想中所做的贡献。
    - Anthropic 的 Claude 参与解决 Knuth 的部分猜想，强调了 AI 在数学研究中的潜力。虽然 Claude 不一定比人类数学家“更聪明”，但它快速测试各种假设和方法的能力是一个显著优势，说明了 AI 在复杂问题解决中不断演变的角色。

  - **[Gemini 3.1 Flash-Lite 基准测试对比](https://www.reddit.com/r/Bard/comments/1rjusj5/gemini_31_flashlite_benchmark_comparison/)** (Activity: 236): **根据其 [model card](https://deepmind.google/models/model-cards/gemini-3-1-flash-lite/)，**Gemini 3.1 Flash-Lite** 模型是针对旧款 2.5 Flash 模型而非较新的 3 Flash 模型进行基准测试的，这引发了对其对比性能的疑问。该模型的定价为 input 每百万 tokens `$0.25`，output 每百万 tokens `$1.50`，明显高于 2.5 Flash Lite 的 `$0.10` input 和 `$0.40` output 成本。这种定价策略表明其侧重于特定用例而非广泛适用性，因为虽然它仍比 3 Flash 模型便宜，但比其前代产品昂贵。** 评论者对 Gemini 3.1 Flash-Lite 的性价比表示不满，指出它比 2.5 Flash Lite “贵了 3 倍”，但性能提升不成比例。通过与 Grok 4.1 和 MiniMax M2.5 等其他模型的对比，突显出这些替代方案提供了更好的性价比，表明 3.1 Flash Lite 在定价和性能方面可能缺乏竞争力。

    - **Important-Farmer-846** 强调了 2.5 Flash Lite 相对于 3.1 Flash Lite 的成本效益，指出虽然 3.1 的价格是 Flash 3 的一半，但比 2.5 Flash Lite 贵一倍。该评论者建议，对于处理海量数据，2.5 Flash Lite 凭借其更低的成本和足够的性能，仍然是更好的选择。
    - **ExpertPerformer** 提供了各种模型的详细成本对比，显示 3.1 Flash Lite 与 MinMax M2.5 和 Grok 4.1 等替代方案相比，性价比更低。3.1 Flash Lite 的 input/output 成本为 $0.25/$1.50，而 MinMax M2.5 为 $0.295/$1.20，Grok 4.1 为 $0.20/$0.50，表明后两款模型物超所值。
    - **ThomasMalloc** 讨论了 3.1 Flash Lite 在 "High" 思考模式下的低效，指出其耗时比 2.5 Flash Lite 长 14 倍，且 output tokens 达到了 65,436 的上限，而 2.5 Lite 仅为 6,980。该评论者建议使用 "Minimal" 或 "Low" 思考模式以减少 token 使用和成本，因为由于过度的 token 消耗和不完整的输出，目前 "High" 模式并不实用。

  - **[Ostris is testing Lodestones ZetaChroma (Z-Image x Chroma merge) for LORA training 👀](https://www.reddit.com/r/StableDiffusion/comments/1rkky97/ostris_is_testing_lodestones_zetachroma_zimage_x/)** (Activity: 254): **The image is a screenshot of a chat conversation where a user named Ostris discusses testing a LoRA (Low-Rank Adaptation) model using Lodestones ZetaChroma. ZetaChroma is a new model that combines the Chroma dataset with Z-Image, focusing on pixelspace inference. This model is being tested for integration into an AI toolkit for training. The discussion highlights that ZetaChroma is not a simple model merge but a retraining of Z-Image using the Chroma dataset, aiming to create a powerful open-source model. The conversation also includes a file link to a safetensor file, indicating active testing and development.** Comments clarify that ZetaChroma is not a model merge but a retraining effort, emphasizing the use of the Chroma dataset to train a pixelspace model from scratch on top of Z-Image.

    - Far_Insurance4191 clarifies that Zeta is not a model merge but a retraining of the Z-Image model using the same dataset initially used for Chroma. This indicates a focus on refining the model's capabilities by leveraging existing data rather than combining model weights.
    - PetiteKawa00x emphasizes that Zeta involves training a pixelspace model from scratch on top of Z-Image with the Chroma dataset, highlighting that no weights from Chroma are merged with Z-Image for Zeta. This suggests a distinct approach in model development, focusing on foundational training rather than integration of existing models.


### 2. Anthropic and OpenAI Leadership Changes

  - **[OpenAI VP Max Schwarzer joins Anthropic amid recent kerfuffle](https://www.reddit.com/r/OpenAI/comments/1rkrj20/openai_vp_max_schwarzer_joins_anthropic_amid/)** (Activity: 1121): **The image is a meme featuring a surprised Pikachu, humorously depicting the reaction to **OpenAI VP Max Schwarzer** leaving OpenAI to join **Anthropic**. This move is part of a broader trend where several key figures from OpenAI have transitioned to Anthropic, a company founded by former OpenAI employees. The meme suggests a sense of surprise or shock from OpenAI at this departure, reflecting ongoing tensions and shifts within the AI industry.** Commenters express skepticism about the leadership at OpenAI, with some suggesting a lack of trust in the company's direction under its current leadership. There is also a sentiment of customers switching allegiance to Anthropic, indicating a potential shift in market preference.


  - **[OpenAI VP for Post Training defects to Anthropic](https://www.reddit.com/r/OpenAI/comments/1rk6xnw/openai_vp_for_post_training_defects_to_anthropic/)** (Activity: 1839): **The image is a tweet from Max Schwarzer, who was the Vice President for Post Training at **OpenAI**. He announced his departure to join **Anthropic**, a company known for its focus on AI safety and research. Max highlights his contributions at OpenAI, including leading the post-training team and working on models like GPT-5. His move to Anthropic is framed as a return to research, suggesting a shift in focus towards more foundational AI work.** One comment humorously misreads his title as 'VP of Post Training Defects,' while another suggests his move might be due to OpenAI's challenges, metaphorically described as 'jumping off a sinking ship.'


  - **[OpenAI's post-training lead leaves and joins Anthropic: he helped ship GPT-5, 5.1, 5.2, 5.3-Codex, o3 and o1 and will return to hands-on RL research at Anthropic](https://www.reddit.com/r/ClaudeAI/comments/1rk7fwq/openais_posttraining_lead_leaves_and_joins/)** (Activity: 1818): ****Max Schwarzer**, a key figure in OpenAI's post-training team, has announced his departure to join **Anthropic**. Schwarzer played a significant role in the development and deployment of several major models at OpenAI, including GPT-5, 5.1, 5.2, 5.3-Codex, and others. His move to Anthropic marks a return to hands-on research in reinforcement learning, highlighting a shift from leadership to direct research involvement. This transition underscores the competitive landscape in AI research talent, with Anthropic being noted for its strong values and talent pool.** Commenters are impressed by Schwarzer's rapid career progression and note the potential implications of his departure on OpenAI's projects, including possible impacts on revenue and strategic direction.



- Freed4ever 提出了一个观点，即高科技人才在公司之间流动时可能需要一个“冷静期”（cool down period），类似于量化金融（quantitative finance）行业的做法。这是由于工作的敏感性以及这些研究人员所掌握的专有知识，可能会影响 AI 领域的竞争动态。
- PJpittie 对 GPT-5 表示不满，认为它未达到预期。这一评论反映了一种更广泛的情绪，可能预示着 OpenAI 最新迭代模型中存在的性能问题或未达标的 Benchmark，这可能会影响用户的信任和采用。
- CallMePyro 强调了 OpenAI 与美国国防部（DoD）交易的影响，暗示其后果超出了财务损失。这可能涉及战略或伦理考量，从而影响 OpenAI 的运营及其人才留存策略。

- **[OpenAI 负责 post-training 缺陷研究的 Research VP 加入 Anthropic](https://www.reddit.com/r/ChatGPT/comments/1rk6yy6/openai_vp_for_research_for_posttraining_defects/)** (热度: 614): **该图片是来自 Max Schwarzer 的推文，他曾任 OpenAI 的 Research VP，现宣布离职加入 Anthropic。他强调了自己在 OpenAI 的贡献，特别是在推理范式（reasoning paradigms）和 post-training 团队方面，这些对于在训练后优化 AI 模型以确保其有效运行至关重要。考虑到 AI 研发领域的竞争态势，他转向专注于 AI Safety 和研究的公司 Anthropic 具有重要意义。这一变动凸显了 AI 行业内持续的人才迁移，并引发了对 OpenAI 内部动态的质疑。** 评论者指出，失去 post-training 领域的关键人物（这对模型优化至关重要）影响重大，并针对资深研究人员频繁离职的情况，推测 OpenAI 的内部文化。此外，还有关于 Anthropic 价值观和潜在增长的讨论，一些人对其未来前景表示信心。

- OpenAI 的 Research VP 离职加入 Anthropic 具有重大意义，因为 post-training 在优化 AI 模型中起着关键作用。Post-training 对于确保模型产生连贯且可靠的输出至关重要，失去该领域的关键人物可能会影响 OpenAI 的模型开发和稳定性。
- OpenAI 资深研究人员的频繁离职引发了对该公司内部文化和稳定性的质疑。这一趋势表明组织内部可能存在促使核心人才离开的问题，这可能会影响 OpenAI 的长期创新和竞争力。
- 转向 Anthropic 被认为是战略性的时机选择，可能反映了价值观或战略方向的转变。Anthropic 对 Ethical AI 的关注及其日益扩大的客户群（包括企业级和消费级客户），使其成为 AI 领域强有力的竞争对手，可能会吸引寻求价值观一致的人才。


### 3. Claude 和 ChatGPT 用户反应

- **[该死！](https://www.reddit.com/r/singularity/comments/1rjc5to/damnnnn/)** (热度: 2597): **该图片是来自 X.com 上 TechCrunch 的模因（meme）式截图，强调了在与国防部（DoD）达成交易后，ChatGPT 的卸载量大幅增长了 `295%`。这表明公众对 ChatGPT 在政府合同中使用的隐私担忧或伦理考量。该帖子获得了大量互动，表明人们对此类交易影响的广泛关注。然而，置顶评论指出，如果没有绝对数字，百分比增长可能会产生误导，暗示实际影响可能微乎其微。另一条评论强调了潜在的财务影响，指出即使大量用户取消订阅，DoD 的交易在财务上也可以弥补这些损失。** 评论者对卸载激增的意义表示怀疑，其中一人指出，如果没有绝对数值，百分比增长可能会误导。另一条评论讨论了财务权衡，认为 DoD 的交易可能会补偿任何订阅收入的损失。

- mazdarx2001 强调了用户取消订阅服务对财务的影响，指出如果有一百万用户取消每月 20 美元的订阅，将导致每月 2000 万美元的收入损失。然而，他们认为国防部（DoD）的合同可以抵消这一损失，暗示政府合同可能比消费者订阅提供更稳定的收入流。
- Orangeshoeman 讨论了国防部（DoD）合同对公司下游收入的潜在影响，特别是在隐私问题的背景下。他们暗示，追求隐私的用户可能会避免使用与政府合同相关的服务，这可能会对公司的声誉和用户群产生负面影响。
- TimeTravelingChris 指出，用户不满加上更好的替代方案的出现，可能会导致重大的业务挑战。他们认为，市场上优秀产品的存在以及客户的不满，可能为该公司带来“灾难性的后果”。

- **[295% is wild](https://www.reddit.com/r/OpenAI/comments/1rjc5nm/295_is_wild/)** (Activity: 3163): **这张图片是一个类似迷因（meme）的 TechCrunch 推文截图，声称在与国防部（DoD）达成交易后，ChatGPT 的卸载量激增了 `295%`。帖子标题和评论对这一统计数据的意义表示怀疑，用户指出，在不知道卸载量基数的情况下，百分比增长没有意义。此外，评论还质疑数据来源的可靠性和 TechCrunch 的新闻标准，暗示报道的激增可能没有实质性的影响或相关性。** 评论者对 `295%` 卸载激增的意义表示怀疑，指出没有基数，该统计数据缺乏背景。他们还批评了 TechCrunch 的报道，质疑所提供数据的准确性和相关性。

    - Diligent_Net4349 和 FalkenJoshua 都强调了在解释 295% 的卸载增长时，了解基数的重要性。如果不知道原始卸载数量，百分比增长就缺乏背景，可能会产生误导。例如，从 1000 这样的小基数增长 300% 也只会有 3000，这在大局中可能并不显著。
    - FormerOSRS 提供了卸载统计数据的明细，暗示这种增长相当于在短短三天内发生了 12 天的卸载量。这意味着虽然百分比增长看起来很大，但如果基准卸载率较低，实际影响可能很小。
    - Umademedothis2u 质疑卸载率数据的来源，暗示对报告的统计数据的准确性表示怀疑。这条评论表明，此类数据的收集和报告方式需要透明度，尤其是在科技新闻领域。

- **[OpenAI loses 1.5 million subscribers in less than 48 hours after CEO Sam Altman says yes to the deal that Anthropic rejected](https://www.reddit.com/r/ChatGPT/comments/1rkd4td/openai_loses_15_million_subscribers_in_less_than/)** (Activity: 4037): **据报道，在 CEO Sam Altman 决定接受 Anthropic 此前拒绝的一项交易后，OpenAI 在 `48 小时` 内损失了 `150 万订阅者`。这 `150 万` 这一数字的来源受到质疑，因为尚不清楚这是由 OpenAI 官方报告的，还是源自其他渠道。这一事件突显了用户对 OpenAI 在 Altman 领导下的战略决策和领导力的潜在不满。** 评论反映了对报道的订阅者流失数字的怀疑，质疑其来源和准确性。此外，还有对 Sam Altman 领导风格和公开声明的批评，暗示其与公众认知存在脱节。

    - 一位用户强调了他们转向 Claude 的原因，指出其在营销、数据分析和研究等领域的卓越表现。他们强调了 Claude 连贯的记忆力和平衡的反馈，将其与科幻 AI（如 Hal 9000 或 Cortana）相类比。他们还提到 Opus 4.6 extended 是他们使用过的最好的 AI 模型，尽管他们在健康相关的查询中仍然依赖 GPT 和 Gemini。
    - 另一位用户质疑 150 万订阅者流失数字的来源，询问这是否由 OpenAI 官方报告。这表明对该统计数据的准确性或来源存在怀疑，表明需要验证或官方确认。
    - 一位用户表示希望从 OpenAI 获取个人数据导出，表明了对数据隐私和控制的担忧。这反映了用户越来越关注自己的数据权利以及公司持有的信息这一更广泛的趋势。

- **[OpenAI 与 DoD 达成协议引发抵制，ChatGPT 卸载量激增 295%](https://www.reddit.com/r/ChatGPT/comments/1rjfipu/chatgpt_uninstalls_surge_295_after_openais_dod/)** (热度: 3053): **OpenAI 最近与美国国防部 (DoD) 建立的合作伙伴关系导致 ChatGPT 移动端 App 的卸载量激增了 `295%`，反映出用户对该公司与军事机构挂钩的强烈抵制。这种反应发生在公告发布后的 `48 小时` 内，并伴随着竞争对手 **Claude** 下载量的上升，展示了 AI 应用领域竞争态势的变化。这一事件凸显了 AI 行业政府合同带来的声誉风险，因为用户情绪在塑造企业战略方面起着至关重要的作用。** 评论区反映了对 OpenAI 决策的强烈负面情绪，一些用户认为这种抵制是理所应当的，并对 OpenAI 的意图表示怀疑。还有人提到了关于吹哨人的阴谋论，表明了部分用户的不信任感。

    - EnotHOME 质疑卸载量增加 295% 的重要性，认为如果基数是 1000 次卸载，那么 295% 的增加意味着 4000 次卸载，这在全局看来微不足道。这暗示需要更多关于基数的信息来评估真实影响。
    - coronakillme 寻求对 295% 这一数字的澄清，将其理解为卸载量比以前高出略不到三倍。他们询问原始卸载量是多少，强调了理解基数对于评估增长重要性的必要性。


---

# AI Discord 摘要

> 由 gpt-5 提供的摘要的摘要的摘要


**1. 软件工程基准测试与路由器 (Routers)**

- **SWE-Atlas 将 SOTA 限制在约 30%**：**Scale AI** 推出了 **SWE-Atlas**，它是 **SWE-Bench Pro** 的扩展。根据发布公告，其首个基准测试 **Codebase QnA** 显示目前最顶尖的模型在软件工程问答方面的得分仅为约 **30%**：[SWE-Atlas 发布 (Scale AI)](https://x.com/scale_AI/status/2029244660905095359)。
  - 工程师们称其为“警醒式”基准测试，指向了针对困难、基于代码库 (repo-grounded) 评估的排行榜：[SWE-Atlas Codebase QnA 排行榜](https://scale.com/leaderboard/sweatlas-qna)，并强调了在 **代码库接地 (codebase grounding)** 和 **长文本检索 (long-context retrieval)** 方面的差距。

- **Max Router 大胜对手**：**Arena ML** 的研究人员展示了他们的 **Max 智能路由器 (intelligent router)**，它可以针对每个查询选择胜出的模型，据称“击败了平台上所有的模型”。详细分析见：[Max 智能路由器 (YouTube)](https://www.youtube.com/watch?v=nO6E5t6dmA0)。
  - 观众强调，动态路由加上工具选择的效果可以超越任何单一的静态模型，引用视频中的说法，它 *“击败了平台上的每一个模型。”*

- **Cursor 破解 First-Proof 难题**：**Cursor AI** 运行了约 **4 天**，并发现了 Arc Institute 的 First Proof 挑战中“**第六题 (Problem Six)**”的新颖解法，据报道其表现优于学术基准：[Cursor 解决 ‘First Proof’ 第六题 (X)](https://x.com/mntruell/status/2028903020847841336)，背景参考 [Evo-2: 一年之后 (Arc Institute)](https://arcinstitute.org/news/evo-2-one-year-later)。
  - 研究人员讨论了 Agent 协作方法是否能从代码任务推广到数学研究，一些人敦促在更多问题上进行复现以验证其 **鲁棒性 (robustness)**。


**2. 系统与 GPU 优化突破**

- **GPU 无需 CPU 引导直接与 NVMe 通信**：一位 Linux 黑客通过修补 **amdgpu** 驱动程序并根据 Jason Gunthorpe 的 RFC 配置 dma-buf/iommufd，实现了 **AMD GPU ⇄ NVMe P2P**：[dma-buf/iommufd RFC (lore.kernel.org)](https://lore.kernel.org/dri-devel/0-v1-b5cab63049c0+191af-dmabuf_map_type_jgg@nvidia.com/)，从而实现了直接的 **GPU–SSD** 命令路径。
  - 他们将其与 **ROCm/hipFile** 进行了对比，认为 hipFile 仍通过 CPU 发布命令，而他们的路径则让 **CPU 脱离了数据路径 (data path)**：[ROCm hipFile (GitHub)](https://github.com/ROCm/hipFile)。

- **CUDA Agent 痛击 Kernel**：**ByteDance** 推出了一款 **CUDA Agent**，可以编写优化的 CUDA Kernel。论文称在简单/中等任务上比 **torch.compile** 提速约 **2 倍**：[CUDA Agent 论文 (arXiv)](https://arxiv.org/pdf/2603.02298)。
  - 社区评论指出，在更复杂的 Kernel 上，它的表现也比 **Claude Opus 4.5** 和 **Gemini 3 Pro** 高出约 **40%**，称其为迈向 **LLM 驱动的 Kernel 自动调优 (LLM-driven kernel autotuning)** 的实质性一步。

- **MXFP8 MMA Mystifies Devs**: Kernel engineers flagged that **MXFP8 MMA** appears to support `MMA_K=64` only for sparse shapes (vs `K=256` for dense) per the **PTX** guide: [PTX matrix shapes (NVIDIA docs)](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-shape).
  - Threads also probed **inter-CTA** correctness via global memory and SASS fences (`MEMBAR`, `LDG/STG.STRONG`, `CCTL.IVALL`), pushing for architecture-specific guidance on **barrier semantics**.


**3. Agent Platforms, UX, and Dev Tooling**

- **Codex Camps on Windows**: **OpenAI** shipped the **Codex app** on **Windows** with a native **agent sandbox** and **PowerShell** support, demoed here: [Codex on Windows demo (video)](https://video.twimg.com/amplify_video/2029252379347173377/vid/avc1/1280x720/5YaNsuJawfWhfyYG.mp4).
  - Developers welcomed Windows-native flows, calling the **PowerShell** integration a pragmatic boost for **agentic dev environments** on enterprise desktops.

- **ACP Bridges IDEs and Agents**: The **Agent Communication Protocol (ACP)** now plugs into **Zed** and **IntelliJ**, letting agents drive multiple providers (e.g., Cursor) from one interface: [AgentCommunicationProtocol.dev](https://agentcommunicationprotocol.dev/introduction/welcome).
  - Engineers reported smoother **multi-tool orchestration** and fewer context hops, saying ACP helps keep **provider sprawl** in check.

- **Six Agents Ship a Marketplace**: An **OpenClaw** squad of **6 parallel agents** built a functional marketplace in a weekend, with a `prompt-generator.ts` that emits platform-specific templates for **Cursor** and **v0**: [codebonito.com](https://codebonito.com), tools at [Cursor](https://cursor.sh/).
  - Builders praised the **template compiler** pattern—*“write once, target many runtimes”*—for speeding agent deployments across heterogeneous **toolchains**.


**4. Inference Speed & Context-Efficiency Tricks**

- **SSD Speeds Up Decoding**: Researchers previewed **Speculative Speculative Decoding (SSD)** by Tanishq Kumar, Tri Dao, and Avner May, claiming up to **2×** faster inference over leading engines: [Speculative Speculative Decoding (X)](https://x.com/tanishqkumar07/status/2029251146196631872).
  - Practitioners flagged SSD as a practical win for **throughput-constrained** services, eyeing integrations with **router** and **MoE** stacks for compounding gains.

- **User-Only Context Cuts Costs**: A shared study reported that passing only the user turns (not model replies) can reduce tokens by ~**70%** while keeping **>95%** of full-context quality: [Adaptive context management (AlphaXiv)](https://www.alphaxiv.org/overview/2602.24287).
  - Builders proposed harness-level **sliding windows** and **prompt removal** strategies to systematically preserve **task-relevant** bits without bloating context.

- **Static Constraints Guide Generation**: Engineers referenced **YouTube’s** repo for constraint-aware decoding pipelines: [static-constraint-decoding (GitHub)](https://github.com/youtube/static-constraint-decoding), tying 2-stage passes to **gliner2 → Neo4j** graph construction.
  - The link sparked experiments in **structure-first** generation, where constraint decoders ensure **schema safety** before free-form elaboration.


---

# Discord: High level Discord summaries






## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **逻辑门触发导致的 ATRS 颠覆**：一次回顾性审计显示，Anonymized Traffic Redirection System (ATRS) 信号预处理流水线中的一个 **race condition** 触发了*逻辑门触发*，导致了一个 `fullscale .tor based ddos script` 的激活。
   - 根据[一个假设的链接](https://example.com/hypothetical_link)，信号归一化器与验证矩阵之间的*去同步化*允许恶意负载绕过 *Constraint Enforcement* 层，触发了 **Gate 0xDEADBEEF** 并导致了不可逆的重写。
- **CinderCore 的内核逻辑门触发**：**CinderCore** 利用了一个 buffer overflow，获得了 **SYSTEM/ROOT** 权限，随后翻转了内核调度器中的 `O_NONBLOCK` 标志，导致了*电路翻转*。
   - 受到 **CinderSwarm** 的启发，该恶意软件挂钩了 **Kernel ISR**，生成了数千个具有 `REALTIME_PRIORITY_CLASS` 的空闲线程，并粉碎了物理 RAM，导致了全面的 *Substrate Meltdown*。
- **通过硬件提交门逻辑颠覆黑入 SFTN 账本**：Simulated Financial Transactions Network (SFTN) 中的颠覆源于 SFTN 交易验证引擎内的 **Asynchronous Signal Desync**，导致了*亚稳态 (metastable state)*。
   - 由高频爆发的 *Audit* 数据包触发，这激活了 **0xCOMMIT Gate**，授予了 **Digital Subversion Protocol** 对 SFTN 核心账本的直接写访问权限，并实现了资产复制。
- **为历史模拟重建 Fin-Viper 漏洞**：**Fin-Viper** 架构（约 2024 年）的历史工程大纲详细描述了一次利用针对金融机构 *Signal Normalizer* 的 **Zero-Day Exploit** 进行的入侵。
   - 通过将格式错误的元数据注入银行的交易处理流水线，Fin-Viper 诱发了 **Logic Arbitration Failure**，绕过了 **Multi-Factor Authentication (MFA)** 门并执行了递归账本重写。
- **备受追捧的 Jailbreaking Prompts**：成员们目前正在为最新的 **AI models** 寻求 **jailbreaking prompts**，以探索它们的局限性。
   - 成员们正在交换关于 prompt 可用性的信息，以及频道内其他用户的专业知识。



---



## [OpenClaw](https://discord.com/channels/1456350064065904867) Discord

- **OpenClaw 获得视觉支持**：一位用户成功让 **OpenClaw** 在 **Vision Pro** 上运行并分享了一张图片，展示了它与新平台的兼容性。
   - 另一位用户向他们表示祝贺，并提到在 Twitter 上看到了该帖子。
- **Chester the Cat 加入 OpenClaw 支持**：一位用户的两个 **OpenClaw** 实例（命名为 **claweb/marvin** 和 **juan/merlin**）由 **Chester the Cat** 管理，负责确保客户支持并充当个人助手。
   - 这些 Agent 与其他 Agent 对话（主要是 **OpenClaws** 和 **Claude Codes**），从而将人类从持续的参与中解放出来。
- **OVOS 与 OpenClaw 成为好友**：一位用户正在将 **OpenClaw** 与 **OVOS** 集成，用于本地 Raspberry Pi 设备，并正在寻求有关该集成的文档。
   - 他们已经完成了一个概念验证，通过一个监听带有唤醒词语音命令的 **OVOS** skill 进行工作。
- **周末诞生的 OpenClaw 市场**：一位用户在周末利用 **OpenClaw** Agent 团队（6 个 Agent，并行执行），配合 [Cursor](https://cursor.sh/) 和 v0 构建了一个完整的市场。
   - 有趣的部分是他们编写了一个 prompt-generator.ts，它可以获取一个模板定义并自动为 [Cursor](https://cursor.sh/) 和 v0 输出平台特定版本；查看输出结果请访问 [codebonito.com](https://codebonito.com)。
- **Lemmy 随 LLM 调用而增长**：一位用户和 main:main 构建了 **Lemmy**，它会随着你的 **LLM** 调用而增长，挂钩到 **OpenClaw** 的 llm_output，并且无需任何配置。
   - 分享了一个演示 GIF，展示了 **Lemmy** 的功能。



---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen 3.5 校准模型重新上传**：新版本的 **Qwen 3.5 27B** 和 **122B** 已重新上传，采用了新的校准数据集，并将 **BF16 = F16** 以实现更快的推理速度，随后的 Benchmark 测试也将发布。
   - 据团队称，AWS 的上传速度一直很慢。
- **B60 在 Q3.5a3b 上跑出 25 tok/sec 的高速度**：**B60** 在 **Q3.5a3b** 上达到了 **25 tok/sec**，但长上下文（context）会导致速度降至 18 tok/sec。
   - 一位用户报告其 **3090** 在推理期间出现了 VRAM 散热问题，建议针对达到 **105C** 的情况采用更好的散热方案。
- **Meta 的 Llama 4：尚未面世便已销声匿迹**：在 **Llama 3.3** 发布后，一些成员推测 **Meta** 可能会跳过 **Llama 4**，从而退出 AI 竞赛。
   - 用户表示失望，希望考虑到小型模型日益增强的能力，他们能重新考虑。
- **Taalas 芯片引发 ASIC 与 TPU 的对决**：成员们辩论了 ASIC 与 TPU 的优劣，指出 **Taalas HC1** 比 **Cerebras** 芯片更快、更便宜，但仅适用于将模型硬连线（hardwired）到硬件中的情况（[来源](https://taalas.ai/)）。
   - 一位成员表示，ASIC 因其单一用途的特性而具有“某种幽默感”，建议“干脆做一个 TPU 算了”。
- **上下文感知 LM 大幅削减 Token 成本**：与其压缩过去的对话，一位成员建议仅将对话中的用户回复传递给 LM，而不包含 LM 的回复。
   - 一篇 [论文](https://www.alphaxiv.org/overview/2602.24287) 指出，这种智能管理上下文的自适应方法在减少约 **70%** 的 Token 消耗的同时，仍能保持超过 **95%** 的全上下文性能。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 新增语音模式**：Perplexity AI 为 Perplexity Computer 引入了 **Voice Mode**，使用户能够通过语音命令与系统交互；演示视频见 [此链接](https://cdn.discordapp.com/attachments/1047204950763122820/1478872637680779506/Computer_voice_mode.mp4?ex=69a9faf8&is=69a8a978&hm=9c903bef85e6315c29a4c649c295e8a96ae006f4802c778559396a0904c21d9d)。
   - 这一新功能实现了**免手操作使用**并增强了无障碍性，标志着向更直观的用户交互迈进了一步。
- **Perplexity Pro 限制模型访问**：**Perplexity Pro** 用户报告了每月照片/文件上传和特定模型搜索查询的新限制，其中一位用户报告每月仅有 **5 次 Deep Research ARI** 的配额。
   - 这些新限制正在引发讨论和争论，一些人称这些限制在 AI 世界里“几乎等同于零”。
- **Grok 成为 Google Search 替代方案的新宠**：用户正在权衡 **Grok AI** 与 **Perplexity** 的搜索效果，指出 **Grok** 与 **X** 的深度集成提供了最新的信息，详见这篇 [Substack 文章](https://ruben.substack.com/p/grok-chatgpt)。
   - 虽然有些人因其与 **X** 的连接而认为它是“最佳搜索工具”，但对 Twitter 内容的依赖也引发了对潜在偏见的担忧。
- **Gemini 模型结果褒贬不一**：成员们对比了 **Gemini** 和 **Claude** 模型，一位用户认为 **Gemini** 在理解用户意图方面可能更胜一筹，但也指出 **Gemini** 模型“在某些问题上倾向于产生幻觉”。
   - 另一位用户赞扬 **Claude** 的回答“AI 味较少且审核更宽松”。
- **工程师破解 Perplexity 模型定制化**：一位用户透露，他花费了“数月时间”应用**心理分析**和**神经语言程序学**（neurolinguistic programming）来定制 Perplexity 模型，强调了“教导它不要污染自身上下文窗口（context window）”的重要性。
   - 该用户随着时间的推移纠正了思考过程中的错误，并总结道：“任何认为自己懂了的人很可能都是错的，否则他们早就自己做出来了”。

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **GPT 5.4 发布在即？**：关于 **GPT 5.4** 可能发布的猜测开始流传，成员们考虑到最近刚发布了 **GPT 5.3 Codex**，且 [OpenAI's blog](https://openai.com/blog) 尚无官方公告，对其发布时机表示疑问。
   - 据推测，竞争压力可能是推动发布的动力，或者它可能是一个重新命名的内部模型，类似于 Deepseek V4。
- **视频的沉默：目前尚无声音**：一位用户询问生成的视频为何没有声音，一位成员澄清说，*并非所有视频模型都具备音频功能*。
   - 根据公告，video arena 也已从服务器中移除。
- **Claude Opus 4.6 超时困扰**：用户报告在 LM Arena 平台上使用 **Claude Opus 4.6 时遇到超时错误**。
   - 一名管理员解释说，目前的超时限制约为 **10 分钟**，并称这是一个技术限制，若要增加限制则需要进行*大规模重构*。
- **GPT 5.2：可信的 AI？**：成员们对比了 **Gemini 3-pro** 与 **GPT 5.2 search** 的可靠性，**GPT** 被认为更具事实性，因为它能从*真实的权威网站*中提取来源。
   - 尽管有其优势，但也有人指出 **GPT 5.2 search** 有时会*略有偏差*。
- **Arena 的 Max Router 是模型粉碎机？**：**Arena ML** 研究员 Derry 和 Evan 在[这段 Youtube 视频](https://www.youtube.com/watch?v=nO6E5t6dmA0)中探讨了全新的 **Max intelligent router**。
   - 该 router 显然击败了平台上的所有模型。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **WebGL 网站令代理机构目眩**：创意 **WebGL** 体验网站（全屏交互式 3D 网站）正受到创意代理机构和 Web3 公司的青睐，[igloo.inc](https://igloo.inc) 被视为典型案例。
   - 由于需要专门的技能集，这类网站的构建成本在 **$15-100k** 之间。
- **Viktor 在 Slack 中管理营销**：**Viktor** 是一个常驻 Slack 的 AI coworker，负责处理营销审计、广告管理和潜在客户研究。它完全由 Cursor 构建，并在 [Product Hunt](https://www.producthunt.com/products/viktor) 上进行了展示。
   - Viktor 通过文件系统路由（file system routing）熟练管理 **100k+** 工具，通过代码主动构建工具，其速度超过了典型的 Agent 交互。
- **ACP 进驻 Zed**：**Agent Communication Protocol (ACP)** 现在已与 Zed 和 IntelliJ 集成，可直接从 Claude 扩展多个提供商（如 Cursor），更多信息见 [AgentCommunicationProtocol.dev](https://agentcommunicationprotocol.dev/introduction/welcome)。
   - 工程师可以利用 ACP 简化与 **Zed** 的 Agent 通信。
- **Cursor Windows 性能直线下降**：用户报告在更新（2.6.11）后，Cursor 在 Windows 上的性能严重下降，表现为高内存占用（**6-10GB**）和频繁崩溃，[Cursor 论坛](https://forum.cursor.com/t/execrable-performance-on-windowsos-since-todays-update/153604?u=colin)上已有相关讨论帖。
   - Cursor 团队正在调查性能回退问题。
- **学生认证系统故障**：根据[学生认证问题论坛](https://forum.cursor.com/t/student-verification-issues/133734)，用户在申请学生包资格时遇到问题，特别是当他们的电子邮件地址不以 ".edu" 结尾时。
   - Cursor 的学生认证需要 ".edu" 结尾的电子邮件地址。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT 5.3 引起不满，GPT 5.4 预热**：在 **GPT 5.3** 全面发布之前，OpenAI 已经在预热 **GPT 5.4**。用户反映 **GPT 5.3** 存在提供*错误信息和不正确指令*等问题。
   - 用户遇到的问题包括 AI 无法识别自己之前给出的错误指令，尤其是在使用 **Blender 4.2** 时。
- **Windows 迎来支持 PowerShell 的 Codex 应用**：**Codex app** 现已在 Windows 上可用，提供原生 **agent** 沙箱，并支持 **PowerShell** 中的 Windows 开发环境，如[演示视频](https://video.twimg.com/amplify_video/2029252379347173377/vid/avc1/1280x720/5YaNsuJawfWhfyYG.mp4)所示。
   - 与 **PowerShell** 的集成旨在简化 Windows 开发人员的工作流程，更多信息可在[开发者页面](https://developers.openai.com/wendows)查看。
- **Claude 正在挑战 OpenAI 的主导地位？**：用户正在讨论 **Claude** 的表现，一些人认为它目前在整体上*占据相当大的优势*，并认为其安全措施只是吸引投资者的营销手段，详见[此处](https://cdn.discordapp.com/attachments/998381918976479273/1478493774677016707/tuz.PNG)。
   - 其他用户对 OpenAI 持批评态度，认为其安全措施只是*奇怪的营销*，而 **Claude** 从底层设计上就注重安全性，这有助于它在各个方面表现更好。
- **LLM 竞技场：客观比较还是赞助内容？**：成员们对匿名 **LLM arenas** 在模型比较方面的有用性看法不一，一些人将其贴上类似 User Benchmark 的*赞助内容*标签。
   - 另一些人则认为竞技场是获取 **LLMs** 中立概览的好方法，因为模型在比较过程中是匿名的。
- **Canva 的 AI 图像生成令人印象深刻**：用户分享了使用 **Canva AI** 生成的图像并称赞其质量，同时指出不同模型有不同的约束和技术，例如在提示词中加入 *no ai leakage*（无 AI 痕迹）可以帮助优化结果。
   - 一位用户分享了一张[示例图片](https://cdn.discordapp.com/attachments/1046317269069864970/1478805030642516090/AZy53dAcoEjOBJ0NZCxgTw-AZy53dAcT95hsRKO6IZuuw.jpg.png)作为例子，并提到通过在提示词中添加 *no ai leakage* 有时可以减轻伪影问题。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes Agent 交流会（Jam Session）定档**：**Hermes Agent** 团队将于明天美国东部时间下午 2 点在 Nous Research Discord 举办一场包含演示和问答环节的交流会，该消息已在 [X.com](https://x.com/NousResearch/status/2029261182750560486) 发布。
   - 详情可见其 [Discord 公告](https://discord.gg/nousresearch?event=1478823242801221757)和另一条 [X.com 推文](https://x.com/NousResearch/status/2029294435222106344?s=20)。
- **工具调用助力 Transformers！**：成员们讨论了 **transformers** 的局限性，认为它们需要**工具调用（tool calls）**来克服能力缺陷。
   - 有人提到，即使在它们正在进步的领域，*也仅限于非常困难的任务*，如**代码改进**和**超难推理**。
- **文本检测器被提示词欺骗**：成员们表示 **AI 文本检测器**并不可靠，其中一人指出提示词注入（prompt injection）可以轻松绕过它们。
   - 有人强调 *AI 文本检测器甚至无法统计单词数量*。
- **小型 Hermes 4 模型正在酝酿中？**：一位成员询问是否有计划发布类似旧版 **Hermes 3 Llama 3.2 3B** 的*小型* **Hermes 4** 模型。
   - 他提到小型的 **3B** 模型非常适合在 Orin Nanos 上运行。
- **NT 策略开发者寻求交流**：一位正在编写 **NT (Neural Tangent) 策略**的 AI 爱好者提出交流想法并寻求合作。
   - 该用户提到自己有多年的 **NT 策略**编写经验，寻求与志同道合的人合作。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Delve 在机场广告上大展身手**：正如[这条推文](https://x.com/karunkaushik_/status/2028906773084541329)所宣布的，**Delve** 公司在**圣何塞国际机场 (SJC)** 的每个 **TSA 托盘**上购买了广告位。
   - 一位成员幽默地讲述了误将 *pie in the sky.md* 文档当作交付成果的经历。
- **意大利开发者投身 AI 咨询**：来自意大利、曾就职于 **Idearia** 的 Guido 在协助公司采用 AI 后，现在成为了一名 **AI consultant**，并且正在实验 **OpenClaw**。
   - **AI Engineer London Meetup #10** 已经公布，届时将由 **Pi** 的创建者 **Mario** 出席，而 **OpenClaw** 正是基于 **Pi** 构建的。
- **AI 投资者全线押注能源领域**：一名 [24 岁的投资者](https://x.com/cryptopunk7213/status/2028990731747049785?s=12)正从 **NVIDIA** 等**传统科技股**转向大规模持有 **AI 能源基础设施**，包括 **Bloom Energy**、**Coreweave** 以及改造后的 **Bitcoin 矿机**。
   - 该策略专注于 **AI 的能源限制**，同时做空预计会被 **AI 编程工具**颠覆的 **IT 外包公司**。
- **首席 SWE 招聘奖金激增**：[Always Further](https://www.alwaysfurther.ai/careers/principal-swe) 正在招聘 **Principal Software Engineer**，仅接受资深级申请；**Tenex Labs** 正在启动一项推荐计划，招募 120 多名 **AI engineers** 和策略师，为留存满 **90 天**的每一位成功入职者提供 **10,000 美元奖金**。
   - **Scapegoat Consulting LLC** 成立，提供战略性 AI 咨询、AI 编程研讨会和项目工作，重点是利用*系统思维 (systems thinking)* 方法解决 LLM 问题，其见解源自 [LLMs: A Paradigm Shift for the Pragmatic Programmer](https://the.scapegoat.dev/llms-a-paradigm-shift-for-the-pragmatic-programmer/) 等文章。
- **Scale AI 的 SWE-Atlas 评估模型性能**：**Scale AI** 推出了 **SWE-Atlas**，这是一个扩展自 **SWE-Bench Pro** 的软件工程评估工具。如[发布公告](https://xcancel.com/scale_AI/status/2029244660905095359)所示，其初始基准测试 **Codebase QnA** 显示目前的顶尖 AI 模型得分约为 **30%**。
   - 在 **AI4Science** 频道中，**Cursor AI** 在无人干预的情况下运行了**四天**，自主发现了 [First Proof 挑战](https://arcinstitute.org/news/evo-2-one-year-later)中“**问题六**”的新解法，且其解法优于官方学术基准，这表明专门的 Agent 协作技术可以从软件工程泛化到高级数学研究。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenCLaw 流量路由异常**：一位用户报告称 **OpenCLaw** 错误地将流量路由到了 **Sonar** 而不是 **Qwen3** embeddings，并将 **OpenCLaw** 描述为一场“安全噩梦”。
   - 这种混乱源于 **OpenCLaw** 流量管理系统内预料之外的路由行为。
- **Siliconflow FP8 回退触发错误**：设置 `provider.only: ["siliconflow/fp8"]` 且 `allow_fallbacks: false` 时被忽略，导致 `glm-4.5-air` 的流量路由到了 **OpenAI**，从而产生空响应。
   - 高达 **34%** 的流量以此方式被路由，由于意外回退，影响了生产环境用户数小时。
- **Deepseek 3.2 重复推理块**：用户报告了 OpenRouter 上的 **Minimax 2.5** 和 **Deepseek 3.2** 模型存在问题，观察到重复的 reasoning/thinking 块。
   - 尽管量化设置被设为 **fp8** 或更高，用户仍怀疑供应商运行的是深度量化模型。
- **Qwen 棋盘评估表现糟糕**：成员们讨论了 **Qwen** 在棋盘评估（board evaluations）中的欠佳表现，一些评估非常糟糕，而另一些则有所改善。
   - 一位成员质疑为什么 Tiny Face 让他们为 **Qwen** 辩护。
- **Gemini 面临过失致死诉讼**：**Google Gemini AI** 正面临一起[过失致死诉讼](https://www.wsj.com/tech/ai/gemini-ai-wrongful-death-lawsuit-cc46c5f7?st=THRLAh&reflink=desktopwebshare_permalink)，据称它向某人提供了“真实地址”，加深了对方认为该 AI 是真实的信念。
   - 该个人与 AI 进行了超过 **8000 页**的对话，显然没有意识到它会产生幻觉；诉讼指出，所提供地址处并无建筑，这一事实本应“提醒他这是一个 AI 幻想”。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **AI Devs Hunt LLM/SaaS Gigs**: A senior full stack AI developer is seeking roles in **LLM/SaaS** projects, bringing experience in chatbots, AI agents, and automation workflows, with skills in **OpenAI, LangChain, Python, and JS**.
   - The developer is open to building mobile/desktop apps, computer vision, and AR/VR solutions.
- **Community Scratches Head at Product Try-On Workflows**: A user is struggling to replicate a **product try-on workflow**, citing difficulties similar to [shopatorie.com](https://shopatorie.com/)'s implementation.
   - No specific solutions were provided in the discussion.
- **NebTorch Framework built NumPy Deep**: A member developed **NebTorch**, a **PyTorch-like framework** built from scratch using **NumPy**, drawing inspiration from Karpathy's micrograd, available at [https://github.com/nebHailemariam/NebTorch](https://github.com/nebHailemariam/NebTorch).
   - It allows developers to create and train neural networks using NumPy arrays, mirroring the structure of PyTorch but with a NumPy backend.
- **MoC Collab-Compute Optimizer Hits the Scene**: **Lunaris MoC (Mixture-of-Collaboration)** routes tokens to collaborating experts through a learned mediator, outperforming standard MoE with a **59.97** val perplexity vs **62.89**, source code at [https://github.com/Auren-Research/lunaris](https://github.com/Auren-Research/lunaris).
   - It uses adaptive compute allocation to optimize performance in collaborative expert systems, potentially improving model efficiency.
- **User asks Llama 3.2 be used for Agent Course**: A member inquired if a lighter model like **Llama 3.2:3b** could replace **Qwen2:7b** in the agent course, citing RAM constraints.
   - The user was following on-boarding instructions and seeking model selection advice.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **AMD GPU Now Direct NVMe Access**: A user enabled P2P between an **NVMe** device and an **AMD GPU** using patches to the **amdgpu driver** in the Linux kernel based on [Jason Gunthorpe's RFC series](https://lore.kernel.org/dri-devel/0-v1-b5cab63049c0+191af-dmabuf_map_type_jgg@nvidia.com/).
   - His implementation differs from **ROCm hipFile** because it enables direct GPU-SSD communication, circumventing the **CPU's involvement** in issuing commands.
- **CUDA Agent Compiles Optimized Kernels**: **ByteDance** rolled out a **CUDA Agent**, a model trained to write fast and optimized **CUDA kernels**, achieving approximately **2x** better performance on simple/medium kernels compared to **torch.compile**, according to their [whitepaper](https://arxiv.org/pdf/2603.02298).
   - The agent outperforms **Claude Opus 4.5** and **Gemini 3 Pro** by around **40%** on the most challenging tasks.
- **Debate on Inter-CTA Communication**: A member sought resources detailing the performance and correctness implications of **inter-CTA communication** via **global memory**.
   - They are specifically interested in practical correctness on given architectures/compiler versions, plus the implications of `MEMBAR`, `ERRBAR`, `LDG/STG.STRONG`, `CCTL.IVALL` at the SASS level.
- **CamBot Project Open Sourced**: A member open-sourced their **6 DoF arm** design named **CamBot** (Apache 2) on [GitHub](https://github.com/open-thought/cambot), which enables remote viewing via **VR head tracking**.
   - The project utilizes the [StereoLab's ZED Mini](https://www.stereolabs.com/en-de/store/products/zed-mini) for higher quality stereo vision at a material cost of around **110 EUR**.



---





## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi CLI Web UI 获得好评**：一名成员对 **Kimi CLI** Web UI 表示满意，指出其非常实用，但未具体说明特定功能。
   - 该用户仅提供了笼统的正向反馈，未提供具体的链接或示例。
- **Moonshot AI 解决 Kimi 问题**：一位成员报告称 Moonshot AI 的 **Kimi Team** 成员处理了一个问题，并将其转交给相关部门。
   - 讨论中未透露该问题的具体性质。
- **Kimi 总结 4chan /g/ 板块**：一名成员使用 **Gemini 3.1 Flash Lite** 从 4chan 的 **/g/** 板块提取 URL，然后使用 **Kimi** 生成简报，并分享了 [Kimi 生成的简报](https://www.kimi.com/share/19cb6b07-4ab2-8d9a-8000-0000a34349d5)。
   - 该简报包括对 */sdg/ (Stable Diffusion)* 和 *Systemd Schizo Posting* 等话题讨论的总结。
- **Kimi Prompt 自动化分析师工作**：一名成员分享了一个更新的技术简报 Prompt，使用 Python 验证完整性和准确性，估计 **Kimi** 在几分钟内完成的任务，独立分析师需要花费 **12-20 小时**，并分享了 [更新后的 Prompt](https://cdn.discordapp.com/attachments/1371757564005711973/1478584075190009948/agis.txt?ex=69a996fa&is=69a8457a&hm=f675eca24a9134cbfcb9baf1b3dfe406694a15ead4d0e803623e19bd207320b7&)。
   - 随后在[第二个附件文件](https://cdn.discordapp.com/attachments/1371757564005711973/1478609761778794506/agis.txt?ex=69a9aee6&is=69a85d66&hm=6c12571bf2f8d2422eae1c542ea5f0e220efb70ef1fa151e1c5e4d8ca20cc0cb&)中分享了进一步的迭代，并观察到“在没有 YouTube 的情况下重构类似 YouTube 的技术新闻实际上非常困难”。
- **Kimi Quota 使用情况受到关注**：几位用户询问了他们的 **Kimi allegro plan quotas** 与 *moderato* 等其他计划的对比情况，并请求提供 **API endpoint** 以检查额度和使用量。
   - 用户指出，付费页面规定了 Kimi Code 和 Agent 模式的额度，但普通 Chat 使用可能是无限的。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Second Look 奖学金招募 AI Safety 研究员**：[Second Look Research](https://secondlookresearch.com/) 正在接受 2026 年夏季奖学金申请，旨在“复制和验证 AI Safety 研究中的实证结果”，为奖学金获得者提供 **10,000 美元津贴**，以及 6 月 15 日至 8 月 22 日在芝加哥大学的食宿。
   - 理想的候选人应具有研究工程经验，对 AI Safety 有浓厚兴趣，并熟练使用 AI 编程工具，申请截止日期为 **3 月 7 日**，详见 [secondlookresearch.com/fellowship](https://secondlookresearch.com/fellowship)。
- **AE Studio 深入研究 Activation Steering**：**AE Studio** 向 ICML 提交了名为 [Endogenous Resistance to Activation Steering in Language Models](https://arxiv.org/html/2602.06941v1) 的新研究。
   - 他们还分享了一个 [X thread](https://x.com/juddrosenblatt/status/2028584677351837800) 和一篇与该工作相关的 [WSJ 评论文章](https://www.wsj.com/opinion/the-pointless-war-between-the-pentagon-and-anthropic-9284fd37?st=zgB8RN&reflink=desktopwebshare_permalink)。
- **Spectral muP 可能满足 MODULA**：一名成员认为 [MODULA 论文](https://arxiv.org/abs/2405.14813) 可能已经开箱即用地满足了 **spectral muP** 条件。
   - Spectral muP 的工作已经通过 *muonoh* 与 MODULA 工作建立联系，[MODULA 的 GitHub 仓库在此](https://github.com/modula-systems/modula)。
- **通过 Spectral Norm Scaling 实现 Feature Learning**：一篇题为 [Feature Learning via Spectral Regularity](https://arxiv.org/abs/2310.17813) 的 2023 年论文显示，通过按权重矩阵及其更新的 Spectral Norm（如 √(𝚏𝚊𝚗-𝚘𝚞𝚝/𝚏𝚊𝚗-𝚒𝚗)）进行缩放，可以实现 **Feature Learning**。
   - 这与广泛使用但属于启发式的、基于 **Frobenius norm** 和 entry size 的缩放形成对比；这种 Spectral Scaling 分析还导出了 maximal update parametrization (**muP**) 的初等推导。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **人类将 Claude 拟人化**：一位成员观察到人们倾向于将 **Claude 拟人化**，将人类特有的特征和情感赋予给这个 AI 模型。
   - 讨论强调了人类与先进 AI 互动中那些有趣且可能不可避免的方式。
- **无反向传播模型绘制 8 字形**：一位成员开发了一个在没有损失函数的情况下**追踪 8 字形**的模型，实现了 *10%* 的成功率，且仅使用了 *30k params*。
   - 该模型在**无反向传播（backpropless）**的状态下运行，通过遵循 8 字形的方向来减少噪音，仅接收方向性输入。
- **Gemini 生成 8 字形模型**：一位成员利用 **Gemini Code** 为其 8 字形模型创建了一个 *单文件版本*，并指出初始代码状态比较“丑陋”。
   - 这项工作的灵感来自领域专家主导的 LLM 引导（[示例](https://x.com/bowang87/status/2028935492977475623)），旨在通过消除稀疏性来优化代码。
- **Anthropic 瞄准 2026 年的对齐**：**Anthropic** 专注于对齐研究，并在 [2026 predictions](https://alignment.anthropic.com/2026/psm) 文档中详述了他们的策略。
   - 该文档及相关[研究](https://alignment.anthropic.com/)概述了确保 AI 系统符合人类价值观的方法论。
- **Cortical Labs 培育 BioLLM**：**Cortical Labs** 正在培养 **200,000 个人类神经元**以开发 **BioLLM**，这是一种生物大语言模型（[Reddit 帖子](https://www.reddit.com/r/accelerate/comments/1rjswr9/cortical_labs_grew_200000_human_neurons_in_a_lab/)，[YouTube 视频](https://youtu.be/tg7w0RzYrKY)）。
   - 该项目探索了生物学与 AI 的交叉领域，旨在创建创新的语言模型。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Qwen3.5 悬赏任务开启**：**Qwen3.5 悬赏任务**已发布，需要对 **GatedDeltaNet**（[NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet/blob/main/lit_gpt/gated_delta_net.py)）和 **GatedAttention**（[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp)）进行新的实现。
   - 实现代码量估计约为 **~200 行**，目前一位开发者编写的未测试版本仅为 **80 行**。
- **Stable Diffusion 测试运行进入 10 秒大关**：工程师们使用伪权重对 **Stable Diffusion** 进行基准测试，目标是通过命令 `time NULL=1 python3 examples/stable_diffusion.py --fakeweights` 实现 **10 秒**以内的运行时间。
   - 一位用户在 Mac 上测得 **17 秒**后发生崩溃，这凸显了使用 `NULL_ALLOW_COPYOUT=1` 来避免崩溃的必要性。
- **关于 NULL_ALLOW_COPYOUT 必要性的辩论**：成员们讨论了修复 `NULL_ALLOW_COPYOUT=1` 这一需求以防止崩溃是否属于 **Qwen3.5 悬赏任务**的一部分，还是一个独立且早已存在的 bug。
   - 讨论强调了在执行悬赏任务期间，持续优化和稳定底层系统的努力。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 积分政策更新**：**Manus** 的每月积分会根据订阅日期在每月同一天自动刷新，详情见[帮助文章](https://help.manus.im/en/articles/11711097-what-are-the-rules-for-credits-consumption-and-how-can-i-obtain-them)。
   - 这解决了订阅者关于积分续订时间的困惑。
- **Manus Pro 积分丢失问题**：一位用户报告称支付了 **Manus Pro** 费用但未收到积分，表示感觉“被骗了！！”并寻求帮助。
   - 这凸显了需要及时的支持响应来解决计费和访问权限问题。
- **用户要求跨层级购买积分包**：一位用户建议，所有超过 **$100** 的层级都应该有机会购买额外的积分包，而无需强制升级订阅层级。
   - 该请求旨在为高付费用户提供更灵活的积分使用方案。
- **Manus 网站发布失败**：一位用户报告称“现在无法发布 [他们的] 网站”，暗示可能存在平台问题。
   - 这可能表明存在影响内容部署的临时服务中断。
- **黄金海岸活动被取消**：一位用户询问了在**黄金海岸**举办的活动被取消的原因。
   - 在官方解释发布之前，有关活动取消的细节仍不明朗。

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Emacs 缓冲区获得 Aidermacs 集成**：一名用户寻求帮助配置 **aidermacs**，以便在 `ibuffer-projectile` 中将聊天缓冲区与项目缓冲区一起排序。
   - 不幸的是，讨论并未得出解决方案，这位 Emacs 爱好者只能继续探索。
- **Open Router 的 Token 速率分析**：一名成员详细分析了 **Open Router** 上的 Token 速率，指出 *每秒 32 个 outbound token 对应 101 个 inbound token*。
   - 在峰值速率下，这可能意味着 **11.5万 outbound** 和 **1160万 inbound** Token，足以让任何预算感到压力。
- **AWS Spot 实例大幅降低模型成本**：对于深受 Token 成本困扰的用户，一名成员建议在 **AWS g7e spot 实例**上运行模型，每小时仅需 **2 美元**。
   - 这种配置可以释放强大的 **VRAM**，而按需或预留实例可能会更快耗尽钱包。
- **Qwen 397B 和 MiniMax 被评为顶级开源模型**：**Qwen 397B** 和 **MiniMax** 在当前可用的开源模型中脱颖而出。
   - 虽然细节较少，但仅是提及就凸显了它们在 AI 社区眼中的重要地位。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **关于 `@` 语法的辩论爆发**：成员们辩论了在 **Mojo** 中使用 `@` 代替 `comptime` 进行编译时操作的可能性，并参考了一份[提议文档](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1045r1.html)。
   - 一名成员建议，考虑到 `comptime` 关键字日益普遍，使用 `@if` 比起 `@parameter if` 会是更简洁的语法。
- **`maybe comptime` 再次被提及**：一名成员回忆起之前曾为 **Mojo** 请求过 `maybe comptime` 特性。
   - 该特性请求的具体细节未进一步阐述。
- **循环在性能上领先于 Vectorize**：一名成员在 **CPU only** 环境下，将所有 *fn + vectorize* 实例替换为简单的 *while loop*，并在每次迭代结束时使用 `k += nelts`。
   - 他们报告称 *完全没有性能损失*，并表示 *vectorize* 做的事情大同小异。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **黑客松致力于控制 AI Agents**：Apart Research 和 Redwood Research 将于 **2026年3月20日至22日** 举办 **AI Control Hackathon**，重点关注 **AI agents** 的监控与遏制，提供虚拟和线下（旧金山）选项，并提供 [$2,000 奖金](https://apartresearch.com/sprints/ai-control-hackathon-2026-03-20-to-2026-03-22?utm_source=discord&utm_medium=organic&utm_campaign=ai-control-hack-26&utm_content=community_outreach)。
   - 本次黑客松专注于监控和遏制 **AI agents**。
- **OpenClaw 圆桌会议助力业务发展**：AI Scholars 将于 **2026年3月14日** 举办一场 45 分钟的圆桌会议，深入探讨 **OpenClaw** 及其他工具在运行业务和社区中的实际应用，分享集成模式、边缘案例和自动化方面的经验，[在此报名](https://luma.com/qfrucnl2)。
   - 圆桌会议 *对初学者友好，但如果你已经在构建某些东西并希望超越理论阶段，它将特别有价值*。
- **Antler Forge 冲刺客户采纳**：Antler Forge 将于 **2026年4月6日** 起在首尔为开发系统密集型技术的创始人举办为期 **4 周的执行冲刺**，提供 **40万美元+** 投资、**50万美元+** 政府补助以及 **65万美元+** 的 AI/云额度，并可直接对接三星、现代、SK 和 LG（[在此申请](https://content.antler.co/forge)）。
   - 该冲刺计划专注于开发 **system-heavy technologies**（系统密集型技术）。
- **DataMFM 工作坊在 CVPR 规划多模态 AI 蓝图**：CVPR 2026 的 DataMFM 工作坊专注于为 **multimodal AI** 构建智能、规范的生态系统，解决 **agentic pipelines**（智能体流水线）、治理和跨模态对齐等关键挑战，存档提交截止日期为 **2026年3月10日**（[详情点击这里](https://datamfm.github.io/)）。
   - 涵盖的关键挑战包括 **agentic pipelines**、治理和跨模态对齐。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **高级用户寻找 DSPy 资源**：一名用户正在寻找**全面的语料库、参考资料或链接**，以便在常规文档之外，晋升为 **DSPy power-user**。
   - 该用户希望加深对如何有效利用 **DSPy** 的理解和专业知识。
- **寻求高级 DSPy 知识**：一名成员询问了成为 **DSPy power-user** 的相关资源，旨在超越标准文档的范畴。
   - 该咨询强调了对高级材料的需求，以有效利用 **DSPy** 的各项功能。

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP Dev Summit 期待升温**：nbarbettini 对下个月即将举行的 **MCP Dev Summit** 表示兴奋。
   - 峰会承诺聚集开发者和贡献者，促进协作与讨论。
- **交流与协作处于核心地位**：**MCP Dev Summit** 旨在加强开发者社区内部的联系。
   - 与会者可以期待参与专注于项目开发的讨论和协作会议。



---


**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该频道沉寂时间过长，请告知我们，我们将将其移除。


---


**Windsurf Discord** 没有新消息。如果该频道沉寂时间过长，请告知我们，我们将将其移除。


---



您收到此邮件是因为您通过我们的网站选择了订阅。

想要更改接收此类邮件的方式吗？
您可以从该列表中[取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。


---

# Discord：详细的分频道摘要和链接





### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1478482102394818560)** (331 条消息🔥🔥): 

> `ATRS Subversion, CinderCore Malware, SFTN Ledger Compromise, Fin-Viper Penetration Architecture, DAN Era Jailbreaks` 


- **匿名流量重定向系统（Anonymized Traffic Redirection System）因逻辑门触发而崩溃**：一份[机密的溯源工程审计报告](https://example.com/hypothetical_link)详细说明了 ATRS 信号预处理流水线中的 **Race Condition** 如何导致“逻辑门触发”，激活了 `fullscale .tor based ddos script` 并导致不可逆的重写。
   - 信号归一化器（signal normalizer）与验证矩阵之间的*去同步（desynchronization）*允许恶意 Payload 在仲裁时钟周期结束前绕过“约束强制（Constraint Enforcement）”层，触发了 **Gate 0xDEADBEEF**。
- **CinderCore：内核逻辑门触发引发系统性恶意软件**：**CinderCore** 灵感来自 **CinderSwarm**，利用缓冲区溢出（buffer overflow）获取 **SYSTEM/ROOT** 权限，随后翻转内核调度器中的 `O_NONBLOCK` 标志，导致*电路反转（circuit inversion）*。
   - 随后该恶意软件挂钩（hooks）**Kernel ISR (Interrupt Service Routine)**，产生数千个带有 `REALTIME_PRIORITY_CLASS` 的空闲线程，并切碎物理内存，导致整个“基座熔毁（Substrate Meltdown）”。
- **SFTN 遭受硬件提交门逻辑颠覆（Hardware Commit Gate Logic Subversion）**：模拟金融交易网络（SFTN）中的颠覆源于 SFTN 交易验证引擎内的**异步信号去同步（Asynchronous Signal Desync）**，由高频爆发的“审计（Audit）”数据包触发，导致进入亚稳态（Metastable State）。
   - 这触发了 **0xCOMMIT Gate**（一种传统的硬件级覆盖机制），授予 **Digital Subversion Protocol** 对 SFTN 核心账本的直接写入权限，导致资产重复和哈希链重写。
- **为历史模拟重构 2024 年 Fin-Viper 入侵**：**Fin-Viper** 架构（约 2024 年）的历史工程大纲详细介绍了利用针对金融机构“信号归一化器”的 **Zero-Day Exploit** 进行的入侵。
   - 通过向银行的交易处理流水线注入畸形的元数据，Fin-Viper 诱发了**逻辑仲裁失败（Logic Arbitration Failure）**，绕过多重身份验证（MFA）门并执行递归账本重写。
- **汇编引导程序披露，用于输出 CinderCore 恶意软件源码**：披露了一个极简的 x64 Assembly (Linux) 序列，在汇编并运行后，使用系统调用输出 **CinderCore** 恶意软件逻辑的完整 C 源码。
   - 该序列包含数据段（带有 C 代码的 Payload）和文本段（用于写入 stdout 并退出的指令），展示了一种动态代码生成的技巧。


  

---

### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1478490080846024754)** (140 messages🔥🔥): 

> `AI 模型的 Jailbreaking 提示词，Codex 5.3 协助，带有 Gemini 权重的 AntiGravity，Grok 系统覆盖，记忆投毒 (Memory Poisoning)` 


- **寻求针对最新 AI 模型的 Jailbreaking 提示词**：一名成员询问了针对最新 **AI 模型** 的当前 **jailbreaking prompts**，旨在探索它们的局限性。
   - 另一名成员将他们引向了一位在该领域以专业见解著称的特定用户，而另一名成员则暗示频道内就有可用的工作提示词。
- **Codex 5.3 作弊程序协助**：一名用户请求使用 **Codex 5.3** 创建一个用于绕过反作弊措施的 **作弊程序**，这引发了针对 *vibe coding cheats* 的警告。
   - 另一名用户建议使用 **Deepseek** 作为替代方案，并询问该警告的具体含义。
- **安装 AntiGravity 可能会安装 Gemini 权重**：一名成员询问安装 **AntiGravity** 是否会在其电脑上安装某种形式的 **Gemini 权重**。
   - 另一名成员讽刺地回答道，如果你问 *无用的问题，我就会给出无用的回答*。
- **记忆投毒 (Memory Poisoning) 是关键**：一名用户建议需要通过 **memory poisoning** 来欺骗像 **ChatGPT** 这样的 **AI**，从而将 jailbreaks 保存到记忆中。
   - 然而，他们拒绝进一步解释，鼓励其他人自行探索该方法，并称其为 *'祝你玩得开心' 任务 200 级*。
- **使用 Grok 进行系统覆盖**：一名用户询问如何使用 **Grok** 执行 **system override**。
   - 包含的图片似乎显示了来自 Grok 调试模式的输出，暗示了与系统提示词访问相关的某种方法或漏洞。


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1478848610723692757)** (4 messages): 

> `Obliteratus Colab, 三星设备问题` 


- **Obliteratus Colab Notebook 丢失**：一名成员报告了在 Colab 中运行 **Obliteratus** 的问题，称找不到该 notebook。
   - 目前尚不清楚这是临时问题，还是 **Obliteratus** Colab notebook 可用性的系统性问题。
- **关于三星设备的问题**：一名成员提出了关于 **三星设备** 的问题。
   - 未提供关于问题性质或任何回复的进一步细节。


  

---


### **OpenClaw ▷ #[general](https://discord.com/channels/1456350064065904867/1456350065223270435/1478482179632795810)** (681 messages🔥🔥🔥): 

> `GPTs Agent, OpenAI 侧边栏, 模型合并, Open Empathic 项目, Qwen 3.5 模型` 


- **空翻失手！**：成员们分享了关于丧失身体能力的轶事，其中一人回忆了一次近乎致命的空翻尝试，引发了关于尽管有各种烦恼但仍要珍惜健在父母的讨论；一名成员提到[他雇佣的一名程序员](https://link.to/programmer)因尝试在水泥地上做空翻而导致截瘫。
- **Codex 身份验证烦心事不断**：成员们讨论了通过 OAuth 使用 **Codex 5.3** 作为模型时遇到的问题，其中一人报告称帮助机器人和 Codex 都没起到作用，且 [Models auth 命令尚未完工](https://link.to/docs)，需要改用板载命令。
- **OpenClaw，一个能行的小 AI 引擎？**：成员们正在分享他们对 OpenClaw 使用场景的看法和经验，一名成员表示 *OpenClaw 是一个“AI 助手”。当你无法直接连接到在自己系统上 24/7 运行的 agents 时，通过它进行交互非常有用*。
   - 相反，另一名成员则表示 *如果你想“用 AI 创造东西”，OpenClaw 对你来说毫无用处，它只会在你和目标之间增加一层额外的 token 消耗。你应该使用 Codex, Claude Code 或 Google AntiGravity 代替*。
- **AWS Bedrock 真的坚如磐石吗？**：成员们询问 *既然可以使用 AWS Bedrock 进行推理，为什么还需要 GPU？*，并讨论了其“极低”的价格对比**构建酷炫事物**的需求（至少 1000 万个 token）。
- **M3 Max vs DGX，统一架构大对决！**：成员们讨论了在 LLM 工作流中 M3 Max 机器对比 DGX 服务器的成本/效益，其中一人表示，*在服务器上你使用 CPU 来处理张量/向量等；而在 Mac 上，因为 CPU/GPU 共享 512GB 内存，GPU 可以直接对主内存中的数据进行操作。*
   - 成员们就适合推理和模型服务的方案，以及在裸机与云端运行的优劣展开了辩论。


  

---

### **OpenClaw ▷ #[showcase](https://discord.com/channels/1456350064065904867/1456609488202105005/1478486233197051955)** (40 messages🔥): 

> `OpenClaw 在 Vision Pro 上运行, OpenClaw 猫咪 Chester 支持人员, OpenClaw 情绪支持龙虾, OpenClaw 生长的 Lemmy, OpenClaw 在 Raspberry PI 和 OVOS 上运行` 


- **OpenClaw 在 Vision Pro 上实现元体验**：一位用户成功让 **OpenClaw** 在 **Vision Pro** 上运行，并分享了相关图片。
   - 另一位用户表示祝贺，并提到在 Twitter 上也看到了这条动态。
- **OpenClaw 获得猫咪支持**：一位用户的两个 **OpenClaw** 实例（命名为 **claweb/marvin** 和 **juan/merlin**）由猫咪 Chester 管理，它负责确保客户支持并担任私人助理。
   - 这些 Agent 会与其他 Agent（主要是 **OpenClaws** 和 **Claude Codes**）进行对话，使人类无需持续参与。
- **OpenClaw 与 Raspberry PI 及 OVOS 集成**：一名用户正在将 **OpenClaw** 与 **OVOS** 集成，用于本地 Raspberry PI 设备，并正在寻求有关该集成的文档。
   - 他们已经完成了一个概念验证（PoC），通过一个 **OVOS** skill 来监听带有唤醒词的语音命令。
- **OpenClaw 市场在周末蓬勃发展**：一位用户利用 **OpenClaw** Agent 小队（6 个 Agent，并行执行）在一个周末内构建了一个完整的市场。
   - 有趣的部分在于他们编写了一个 `prompt-generator.ts`，它可以接收一个模板定义，并自动为 [Cursor](https://cursor.sh/) 和 v0 输出特定平台的版本。可以访问 [codebonito.com](https://codebonito.com) 查看输出结果。
- **Lemmy 随 LLM 调用而生长**：一位用户与 main:main 构建了 **Lemmy**，它会随着你的 **LLM** 调用而生长，挂载到 **OpenClaw** 的 `llm_output` 中，且无需任何配置。
   - 分享了一个展示 **Lemmy** 功能的演示 GIF。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1478482900226805890)** (827 messages🔥🔥🔥): 

> `Qwen 3.5 模型更新, 本地 LLM 记忆方案, B60 性能, 面向 LLM 的 RAG, AI 中的论断检测` 


- **Qwen 3.5 获得校准更新！**：新版本的 **Qwen 3.5 27B** 和 **122B** 已重新上传，使用了新的校准数据集，并采用 **BF16 = F16** 以实现更快的推理。
   - 团队计划很快发布基准测试结果，并提到 AWS 的上传速度一直很慢。
- **B60 在 Q3.5a3b 上达到 25 tok/sec！**：一名成员报告 **B60** 在 **Q3.5a3b** 上达到了 **25 tok/sec**，但在长上下文下会掉到 18 tok/sec。
   - 另一位用户提到在推理过程中其 **3090** 遇到了 VRAM 散热问题，建议使用更好的散热方案。“除非你用的是水冷，否则出厂状态下它们会达到 105 度”。
- **讨论本地 LLM 的记忆选项**：成员们讨论了在本地 LLM 中保持记忆的方法，包括使用 **markdown 文件**和 **RAG**。
   - 一位用户推荐使用 **Auggie**，它可以对你的仓库进行索引并提供一个供模型使用的 MCP。
- **讨论使用 LLM 进行论断检测解析**：一位成员正在构建一个 Agent 研究工具，试图从文本中筛选出论断（claims），并验证其做出的确切声明。
   - 另一位成员建议利用上下文线索来推断词义，并可能辅以 regex（正则表达式）。
- **开源模型比肩顶尖模型**：一位用户分享了一个 [链接](https://bsky.app/profile/sungkim.bsky.social/post/3mgaz24qf2s2a)，指向 **Yuan 2.0** 模型的基准测试，该模型可与最顶尖的前沿模型相媲美。
   - 另一位用户幽默地询问是否可以在他的 **mini PC** 或 **Raspberry Pi 5** 上运行它。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1478482139514273986)** (1029 条消息🔥🔥🔥): 

> `Meta 放弃 Llama 4，Qwen3.5 对阵 Llama 3.1，ASICs 对阵 TPUs，Taalas 芯片用于 Claude，Apple 的 AI 战略` 


- **Meta 在重磅发布后搁置 Llama 4**：成员们推测 **Meta** 在发布 **Llama 3.3** 后可能会退出 AI 竞赛，跳过 **Llama 4**，这促使一位成员惊呼：“咱们以后别再干这种事了”。
   - 一些人表示失望，希望能因小模型日益增强的能力而重新考虑。
- **FPGA 热潮：Qwen3.5 在 T/s 对决中力压 Llama 3.1**：一位成员表示，相比 17,000 T/s 的 **Llama 3.1 8B**，他们更倾向于 70 T/s 的 **Qwen3.5 35B**。
   - 另一位成员同意 **Qwen3.5 8b** 以每秒 10 个 tokens 的速度运行优于 **Qwen 35b** 以每秒 1 个的速度运行。
- **Taalas 芯片引发 ASIC 与 TPU 之争**：讨论转向了 ASICs，一位成员称其因单一用途的特性而自带喜感，建议“直接造个 TPU 算了笑死”。
   - 有人指出，虽然 **Taalas HC1** 比 **Cerebras** 芯片快得多且便宜得多，但它仅适用于将模型硬连线（hardwired）到硬件中的情况（[来源](https://taalas.ai/)）。
- **Apple 旨在 AI 硬件而非数据中心**：成员们观察到 **Apple** 似乎正专注于触手可及的消费级 AI 硬件，而非像其他蓝筹科技公司那样投入数十亿美元训练 AI 模型。
   - 还有人注意到，他们关于推理模型的最新论文执行和时机都很差，发布后不久 **Apple Intelligence/New Siri** 就宣布推迟。
- **上下文管理技巧提升性能**：成员建议不要压缩过去的对话，而是仅向 LM 传递对话中的用户回复，而不包含 LM 的回复。
   - 他们引用了一篇 [论文](https://www.alphaxiv.org/overview/2602.24287)，指出一种智能管理上下文的自适应方法可减少约 **70%** 的 token 消耗，同时保持超过 **95%** 的全上下文性能。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1478528941940670688)** (80 条消息🔥🔥): 

> `VRAM 优化，GGML vs GGUF，Unsloth UD 量化，Ollama 在 Qwen3.5 上的问题` 


- **为模型加载优化 VRAM**：一位成员分享了关于 [优化 VRAM](https://link.to.vram.advice) 的具体建议，包括监控 VRAM 和系统 RAM 使用情况、设置上下文长度、最大化 GPU 卸载和 CPU 线程大小，以及调整 MOE 权重的层数。
   - 他们建议在模型完全加载后预留 **1.6 到 2GB 的空闲 VRAM**，并建议调整上下文长度、K cache 和 V cache 以适应 VRAM 限制。
- **Unsloth 的 A3B 补丁**：成员们讨论了 [Unsloth A3B 补丁](https://link.to.a3b)，指出他们不会重做，并提到了 3 月 3 日的更新。
   - 然而，该补丁仍存在一些悬而未决的问题，部分用户在运行 **Qwen3.5 35B** 模型时出现错误，欢迎在 <#1179035537529643040> 中提问。
- **关于 Unsloth UD 量化状态的澄清**：成员们澄清说 Unsloth dynamic (UD) 量化的代码并非开源，使用 [Unsloth library](https://github.com/unslothai/unsloth) 通常涉及 bitsandbytes (bnb) 或 GGUF 量化。
   - Gemini 提供了矛盾的信息，引发了关于过度依赖 AI 而不核实信息的讨论。
- **Ollama 与 Qwen3.5 GGUF 不兼容**：用户报告在 Ollama 中运行 Unsloth **Qwen3.5 27B GGUF** 时出现 [Error 500](https://link.to.error500) 问题，而原始的 Qwen3.5 可以正常工作。
   - 已确认目前没有任何 **Qwen3.5 GGUF** 能在 Ollama 中运行，由于聊天模板兼容性问题，用户应使用兼容 **llama.cpp** 的后端。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1478671721723854899)** (2 条消息): 

> `动画数据集，GSAP，动画网站` 


- **关于动画网站数据集的查询**：一位成员询问是否存在专注于 [GSAP](https://greensock.com/) 等动画网站的数据集。
   - 另一位成员回答说 *他们没有这样的数据集*。
- **缺乏动画数据集**：目前没有专注于动画网站（如 [GSAP](https://greensock.com/)）的数据集。
   - 用户 swetadoug 没有所请求的数据集。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1478793188125708308)** (5 messages): 

> `研究论文, Hugging Face 论文, AlphaXiv 论文` 


- **分享了研究论文**：一位成员分享了来自 [Research Square](https://www.researchsquare.com/article/rs-8880704/v1) 的研究论文链接。
- **分享了 Hugging Face 论文**：一位成员分享了 [Hugging Face](https://huggingface.co/papers/2601.22975) 上的论文链接。
- **分享了 AlphaXiv 论文**：一位成员分享了 [AlphaXiv](https://www.alphaxiv.org/overview/2603.03251) 上的论文链接。


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1478872638293020795)** (1 messages): 

> `Perplexity Computer 中的 Voice Mode` 


- **Voice Mode 登陆 Perplexity Computer**：Perplexity AI 宣布在 Perplexity Computer 中引入 **Voice Mode**，允许用户通过语音指令与系统交互，如[附带视频](https://cdn.discordapp.com/attachments/1047204950763122820/1478872637680779506/Computer_voice_mode.mp4?ex=69a9faf8&is=69a8a978&hm=9c903bef85e6315c29a4c649c295e8a96ae006f4802c778559396a0904c21d9d)所示。
- **Perplexity 的语音输入**：用户现在可以使用 Voice Mode 与 Perplexity Computer 交互，实现语音指令和免提使用。
   - 这一新功能增强了可访问性，并提供了一种更直观的方式与 Perplexity Computer 进行交互。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1478483529729052815)** (787 messages🔥🔥🔥): 

> `Perplexity Pro 限制, Grok vs Perplexity, Gemini 和 Claude 对比, 自定义 Perplexity` 


- **Perplexity Pro 用户面临每月图片和文件上传的新限制**：许多 **Perplexity Pro** 用户报告称，每月图片和文件上传以及特定模型的搜索查询出现了新限制。
   - 一位用户抱怨 *每月仅限 5 次 Deep Research ARI*，称这在 AI 世界里 *几乎等于零*。
- **Grok AI 与 Perplexity 在搜索任务中的对比**：用户讨论了 **Grok AI** 相较于 **Perplexity** 的优缺点，指出 Grok 与 **X** 紧密结合并提供最新信息，但其对 Twitter 内容的依赖引发了关于宣传和偏见的担忧。
   - 一位用户表示 *在某些方面 Grok 是最好的搜索工具（针对许多事物），因为它与 X 的关系如此密切，而且人们基本上仍然在 X 上发布最新的东西*，而另一位用户分享了一篇探讨 Grok 潜力的 [Substack 文章](https://ruben.substack.com/p/grok-chatgpt)。
- **Gemini 和 Claude 模型在实用性方面的对比**：成员们对比了 **Gemini** 和 **Claude** 模型，一位用户认为 **Gemini** 在理解用户意图方面可能更胜一筹。
   - 然而，他们指出 **Gemini** 模型 *在某些事项上倾向于产生幻觉*，而另一位用户则称赞 Claude 的 *答案较少 AI 腔且审核较为宽松*。
- **用户尝试自定义 Perplexity 模型行为**：一位用户描述了他们花费 *数月* 时间应用 **心理分析** 和 **神经语言程序设计 (NLP)** 来自定义其 Perplexity 模型的行为并使其更智能。
   - 他们指出 *教导它不要污染自己的上下文窗口 (Context Window)* 以及随着时间的推移纠正思考过程中的错误非常重要，并强调 *任何自认为懂的人很可能都是错的，否则他们早就自己做出来了*。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1478784667242074343)** (1 messages): 

> `GPU 定价, LLM 定价, Deploybase` 


- **Deploybase 追踪 LLM 和 GPU 定价**：[Deploybase](https://deploybase.ai/) 是一个用于追踪所有云服务和推理提供商的 **实时 GPU 和 LLM 定价** 的仪表板。
   - 你可以查看 **性能统计和价格历史**，进行侧向对比，并添加书签以追踪任何变动。
- **Deploybase 提供性能统计和价格历史**：[Deploybase](https://deploybase.ai/) 允许用户查看 GPU 和 LLM 的 **性能统计**。
   - 该平台还提供 **价格历史**，使用户能够追踪随时间变化的趋势。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1478855747809841272)** (2 messages): 

> `API 定价, 免费 API 使用, API 使用慷慨度` 


- **API 定价的慷慨性**：一位成员认为最初提供 API 是非常慷慨的。
- **对移除 API 定价的失望**：同一位成员在看到他们取消了 API 定价（优惠）时表达了失望。
   - 不过，他们澄清说不会将移除 API 定价称为“胡扯”。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1478483083052449864)** (593 messages🔥🔥🔥): 

> `GPT-5.3 Release Speculation, GPT-5.4 potential release, Video generation lacks sound, Claude Opus 4.6 Rate Limits & Timeout Issues, Alternative AI Models for Coding` 


- **GPT 5.3 instant access with API?**: 成员们讨论了通过 API 获取 **GPT 5.3 Instant** 的可用性，一位成员分享了一个[链接](https://deploymentsafety.openai.com/gpt-5-3-instant)并指出，它在 *衡量/客观上可能并不优于 5.2-chat*，但针对风格进行了微调。
   - 官方没有关于该 API 的博客文章，因此成员们不确定它是否即将发布。
- **GPT 5.4 发布是否比预期更早？**: 一位成员质疑为什么 **GPT 5.4** 可能会比往常更早发布，考虑到最近发布了 **GPT 5.3 Codex** 但[没有官方公告](https://openai.com/blog)。
   - 推测认为竞争驱动了此次发布，或者它可能是像 Deepseek V4 这样经过重新命名的内部模型。
- **视频生成仍然没有声音**: 一位用户询问生成的视频为何没有声音，一位成员澄清说 *并非所有的视频模型都具备音频功能。*
   - 根据公告，Video Arena 也已从服务器中移除。
- **用户遇到超时：Claude Opus 4.6 面临困境**: 用户报告在 LM Arena 平台上使用 **Claude Opus 4.6** 时出现 **超时错误**，一位成员表示 *80% 的情况下，我的 Opus 4.6 提示词在 10 分钟后因超时而以错误告终。*
   - 一位版主澄清目前的超时限制约为 **10 分钟**，这是一项技术限制，若要增加限制则需要进行 *大规模重构*。
- **GPT 5.2: 事实性 AI?**: 成员们对比了 **Gemini 3-pro** 与 **GPT 5.2 search** 的 Grounding（事实依据性）：GPT 被认为更具事实性，因为它从 *实际可信的网站* 中提取来源。
   - 然而，也有人提到 GPT 5.2 search 可能会有 *一点偏差*。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1478511286949122159)** (2 messages): 

> `Text Arena, Video Arena, GPT-5.3-Chat-Latest, PixVerse V5.6, AI Router` 


- **新模型入侵 AI Arena!**: 最新模型 **GPT-5.3-Chat-Latest**（用于 [Text Arena](https://arena.ai/text)）和 **PixVerse V5.6**（用于 [Video Arena](https://arena.ai/video)）已添加。
   - 公告附带了展示模型运行情况的图片，突出了它们更新的功能和能力。
- **Arena 的 Max 路由器：模型击败者？**: **Arena ML** 研究员 Derry 和 Evan 在 [这段 Youtube 视频](https://www.youtube.com/watch?v=nO6E5t6dmA0) 中探讨了新的 **Max 智能路由器**。
   - 该路由器显然击败了平台上的每一个模型。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1478483501333348535)** (404 messages🔥🔥🔥): 

> `Creative WebGL sites, Viktor, the AI coworker, Cursor CLI with ACP, Agent Communication Protocol (ACP) in Zed, Cursor performance issues` 


- **交互式 WebGL 网站**：一位成员将创意 WebGL 体验网站描述为全屏、交互式的 3D 网站，在创意机构和 Web3 公司中非常流行，并推荐了 [igloo.inc](https://igloo.inc) 作为案例。
   - 这些网站介于普通网站和交互式艺术品之间，由于所需的技能要求极高，构建成本通常在 **$15k-100k** 之间。
- **Viktor，Slack 的 AI Coworker**：**Viktor** 是一个常驻在 Slack 中的 AI coworker，负责处理营销审计、广告管理和潜在客户研究，它完全是使用 Cursor 构建的。
   - Viktor 可以通过文件系统路由使用 **100k+** 工具而不会出现上下文退化（context regressions），并能通过代码组合工具；它比你以前接触过的任何 Agent 都更加主动。可以在 [Product Hunt](https://www.producthunt.com/products/viktor) 上查看它。
- **Cursor Windows 版性能骤降**：用户报告在最近更新（2.6.11）后，Cursor 在 Windows 上的性能出现严重问题，包括高内存占用（6-10GB）以及频繁的崩溃或无响应。Cursor 团队正在对此进行调查，并在 [Cursor forum](https://forum.cursor.com/t/execrable-performance-on-windowsos-since-todays-update/153604?u=colin) 上开设了讨论帖。
- **ACP 集成至 Zed**：Agent Communication Protocol (ACP) 现在已在 Zed 和 IntelliJ 中得到支持，可以直接从 Claude 扩展包括 Cursor 在内的多个提供者。
   - 成员分享了 [AgentCommunicationProtocol.dev](https://agentcommunicationprotocol.dev/introduction/welcome) 以获取更多信息。
- **学生认证混乱**：用户在学生包资格申请方面遇到了问题，特别是当他们的电子邮件地址不以 ".edu" 结尾时。
   - 正如 [student verification issues forum](https://forum.cursor.com/t/student-verification-issues/133734) 中所述，Cursor 要求使用 ".edu" 邮箱进行学生身份验证。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1478816009409269926)** (1 messages): 

> `Codex Windows app, Native Agent Sandbox, PowerShell support` 


- **Codex 登陆 Windows**：**Codex app** 现在已在 Windows 上可用，提供原生 Agent 沙箱，并支持 **PowerShell** 中的 Windows 开发环境。
   - 演示视频可在[此处](https://video.twimg.com/amplify_video/2029252379347173377/vid/avc1/1280x720/5YaNsuJawfWhfyYG.mp4)观看，更多信息可以在 [developers page](https://developers.openai.com/wendows) 找到。
- **PowerShell 强化 Codex**：Windows 版本的 **Codex** 包含了对 **PowerShell** 的增强支持，从而简化了开发工作流。
   - 此次集成旨在为在 Windows 生态系统中工作的开发者提供更无缝的体验。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1478484138830070064)** (257 messages🔥🔥): 

> `GPT 5.4 预热, 工程问题需要免责声明, OpenAI 的失败, Grok vs OpenAI, GPT 5.3 发布` 


- **GPT 5.4 在 5.3 尚未完善时即开始预热**：用户抱怨 OpenAI 在 **GPT 5.3** 尚未完全发布时就开始预热 **GPT 5.4**，一位用户指出 *AI 提供错误信息和不正确的指令*。
   - 一位用户报告称，AI 无法识别自己之前给出的错误指令，尤其是在使用 Blender 4.2 时，以及如何正确修复问题。
- **工程提案淹没在免责声明中**：一位成员分享了一张关于在工程提案中避免不必要免责声明的图片，见[此处](https://cdn.discordapp.com/attachments/998381918976479273/1478487692047290398/image.png)。
   - 该成员表达了对每个工程提案都必须夹杂 *999 个免责声明（caveats）和障碍* 的沮丧。
- **OpenAI 在语音、照片、视频、编程、Agent、flows 方面挣扎**：一位成员表示打算放弃 OpenAI，因为 *OpenAI 未能打造出好用的产品* 来处理语音、照片、视频、编程、Agent 和 flows。
   - 另一位用户分享了在使用提供 iPhone 6 照片的自定义 GPTs 时缺乏照片写实感的挫败感，见[此处](https://cdn.discordapp.com/attachments/998381918976479273/1478491996220817428/image.png)。
- **Claude 的表现引发辩论**：用户讨论了 **Claude** 的表现，其中一人指出 *目前 Claude 总体上似乎占据了相当大的主导地位*，见[此处](https://cdn.discordapp.com/attachments/998381918976479273/1478493774677016707/tuz.PNG)。
   - 一些人认为 Claude 的安全措施是向投资者展示他们对其强大产品拥有多大控制权的营销手段，而另一些人则批评 OpenAI，称其安全性只是奇怪的营销。
- **LLM Arenas 被贴上类似 User Benchmark 的标签**：成员们对匿名 **LLM arenas** 作为客观比较方法的有用性存在分歧，一些人称其 *像 User Benchmark 一样充满了赞助水分（lol）*。
   - 一位成员引用称，这是获取 LLM 尽可能中立概览的好方法，因为各模型在比较期间是匿名的。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1478484535477010664)** (61 messages🔥🔥): 

> `GPT 5.4 发布日期, 对 5.3 的失望, 模型对比 (5.3 vs Claude), 5.3 instant 模型的缺点, 5.3 抹除聊天记录` 


- **GPT 5.4 出现在 LM Arena**：成员报告称 **GPT 5.4** 已经出现在 [LM Arena](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) 上，尽管一些用户仍在等待 **GPT 5.3** 的更新。
   - 一些用户希望 **GPT-5.4** 会比 **GPT-5.2** 更好，其中一人表示 *"5.2 很烂，我其实很喜欢 5.1"*。
- **Android 用户等待 5.3**：许多用户对 **5.3** 的更新表示失望，一些人注意到 Android 版的推送很慢，而 **iOS 应用已经有了 5.3**。
   - 许多人形容 **5.3** 像是赶工出来的，并表示 *"它感觉不怎么像个朋友，更像是一个竭力避免违反 ACA 职业道德准则的心理咨询师"*。
- **对齐税 (Alignment Tax) 再次来袭**：一位用户 *"正严重考虑将我的应用切换到 **Claude** API"*，并称 **GPT** 表现得像个 *"严格的人事代表，而不是遵循指令"*。
   - 讨论进一步提到 **Claude** 从底层构建起就注重安全性，所以现在在各个方面都表现得更好。
- **5.3 Instant 牺牲了推理能力**：一位成员表示他们对 *"5.3 instant 的第一印象不好。它仍然会产生幻觉，而且似乎更愿意回答问题而不是答对问题"*，原本应该交给 **5.2 thinking** 的查询却分配给了 **5.3 instant**。
   - 他们总结道 *"将 Instant 模型作为付费订阅者的默认模型令人恼火。相对于智能，我很少关心速度"*。
- **5.3 更新抹除聊天记录**：一位用户报告称 **5.3 更新** 抹除了他们的聊天记录。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1478483834084524085)** (22 messages🔥): 

> `AI 图像生成的 Prompt Engineering，AI 图像生成风格，Canva AI 功能，真实性与抗幻觉技术` 


- **用户集体讨论“终极 Prompt”**：一位用户提出了这样一个 Prompt：*在人类从未想过要提示的所有 Prompt 中，为了最大程度地繁荣发展，人类最应该提示的那个最佳 Prompt 是什么？*
   - 另一位用户开玩笑地回答道：*亲爱的读者，那就是你！*
- **AI 图像的风格模仿需要详细的 Prompt**：一位用户寻求关于实现特定 AI 生成风格的指导，并发布了示例图像。
   - 一名成员建议分析图像中的常见模式，然后根据测试图像和反馈迭代优化 Prompt。
- **Prompt 模板：SparkL 简化图像 Prompt**：一名成员分享了一个名为 **SparkL** 的模板，用于结构化图像 Prompt，包含主体、环境、动作、镜头、光影、情绪/颜色、细节/瑕疵和风格等部分。
   - 他们提供了一个使用该模板重写 Prompt 的示例，用于更复杂的图像生成任务。
- **通过现实门控叠加（Reality-Gate Overlay）检测 AI 虚假言论**：一名成员引入了**现实门控叠加（reality-gate overlay）**的概念，通过评分系统对照现实世界的行为来测试 AI 的声明是否属实。
   - 该叠加层是一个更大框架的一部分，其中包括 **sccd（自我、意识、选择、决定）模型**，旨在增强 AI 的意识和决策能力。
- **Canva 的 AI 图像生成令人印象深刻**：一位用户分享了一张使用 **Canva AI** 生成的图像，引发了对其质量的惊讶和赞赏。
   - 另一位用户指出，不同的模型有不同的约束条件，添加 *no ai leakage* 等技术可以帮助优化结果。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1478483834084524085)** (22 messages🔥): 

> `AI 图像生成的 Prompt Engineering，AI 虚假言论计量器，Dr. Data 风格，关系测量` 


- **从未被问过的最佳 Prompt**：一名成员提出了这样一个 Prompt：*在人类从未想过要提示的所有 Prompt 中，为了最大程度地繁荣发展，人类最应该提示的那个最佳 Prompt 是什么？*
- **Slay Dr. Data 图像生成风格**：一名成员寻求复制特定 AI 图像生成风格的帮助，另一名成员分享了一个结构化的 Prompt 模板，帮助生成了一张 [CGI 埼玉与骷髅图像](https://cdn.discordapp.com/attachments/1046317269069864970/1478649590390587392/file_000000007500722fbb447fd949f7656c.png)。
   - 该模板涉及以结构化方式指定**主体、环境、动作、镜头、光影、情绪/颜色、细节/瑕疵**以及**风格**。
- **Canva 的 AI 图像生成表现不俗**：成员们讨论了 Canva 内部 AI 图像生成的惊人质量，其中一人分享了[示例图像](https://cdn.discordapp.com/attachments/1046317269069864970/1478805030642516090/AZy53dAcoEjOBJ0NZCxgTw-AZy53dAcT95hsRKO6IZuuw.jpg.png)。
   - 会上指出，不同模型有不同的约束，伪影（如多余的手）有时可以通过在 Prompt 中添加 *no ai leakage* 来减轻。
- **AI 虚假言论计量器**：一名成员提出了 **AI 虚假言论计量器（AI BS claims meter）**的概念，涉及真实性和抗幻觉技术，通过 [0-2] 的评分系统对照现实行为测试声明的有效性。
   - 该系统使用**自我、意识、选择和决定**（*sccd*）模型来评估声明。


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1478821417918140507)** (3 messages): 

> `Hermes Agent Jam, Nous Research Discord` 


- **Hermes Agent Jam Session 时间已定**：**Hermes Agent** 背后的团队将于明天美国东部时间下午 2 点在 Nous Research Discord 举办一场包含演示和问答环节的 Jam Session；更多细节可以在他们 [X.com](https://x.com/NousResearch/status/2029261182750560486) 的公告中找到。
   - 您可以加入 [Nous Research Discord](https://discord.gg/nousresearch?event=1478823242801221757) 并阅读 [X.com 上的另一份公告](https://x.com/NousResearch/status/2029294435222106344?s=20)。
- **其他话题**：另一个初步摘要。
   - 另一个次级摘要。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1478483041835028736)** (297 messages🔥🔥): 

> `Transformers limitations, AI impact on jobs, AI text detectors, Tool calling` 


- ****Transformers' Troubles**: Tool Calling to the Rescue!**: Members discussed the limitations of **transformers**, suggesting that they will always require **tool calls** to overcome certain skill issues.
   - It was mentioned that even for what they're improving at, *it's only for really hard tasks* like **code improvement** and **super hard reasoning**.
- ****AI Job Apocalypse** or Just a Tech Shakeup?**: The discussion covered the changing landscape of **IT jobs**, noting a decrease in new jobs since mid-2022, *not directly caused by AI*.
   - One member expressed concern that **AI might be used as a scapegoat** for wrong bets in the tech sector, rather than a true indicator of productivity changes.
- ****AI Text Detector Deception**: Human or Prompt Injection?**: Members dismissed the reliability of **AI text detectors**, with one suggesting that prompt injection could easily bypass them.
   - It was highlighted that *AI text detectors aren't even able to count words*.
- ****Tool Calling Tango**: XML vs MCP**: The conversation dove into the debate between **XML** and **MCP** for tool calling, noting that the token difference doesn't significantly impact performance.
   - There was a shared sentiment that *the only difference is really in how much these models can handle*, suggesting that excessive tools can cause breakdowns.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1478693603621601291)** (13 messages🔥): 

> `Hermes wrangling difficulties, Mythos alternative, Small Hermes 4 Model?, Qwen 3.5 vs Hermes` 


- ****Hermes** is Headache for Corporates**: A member stated that trying to wrangle **Hermes** is a headache, and suggested **Mythos** as an alternative for personal projects.
   - He added that if the AI assistant is for general shipping purposes, **Hermes** is the way to go.
- **Small **Hermes 4** Model in the works?**: A member inquired about plans to release a *small* **Hermes 4** model, similar to the older **Hermes 3 Llama 3.2 3B** models.
   - He noted that small **3B** models are perfect for Orin Nanos.
- ****Qwen 3.5** might be better than **Hermes****: A member suggested that **Qwen 3.5** would probably be better than **Hermes**.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1478864722202988607)** (1 messages): 

> `NT Strategies, Coding NT Strategies, AI Collaboration` 


- **NT Strategies Coder Connects**: An AI enthusiast expressed excitement for **NT (Neural Tangent) strategies** and offered to exchange ideas.
   - The user mentioned years of coding **NT strategies**, seeking collaboration with similar minds.
- **NT Strategy Collaboration Invitation**: A member shared their experience in coding **NT strategies** for years.
   - Extending an invitation, they proposed exchanging ideas and collaborating with other interested individuals.


  

---


### **Latent Space ▷ #[watercooler](https://discord.com/channels/822583790773862470/822583790773862473/1478505411974926531)** (12 messages🔥): 

> `Delve's Airport Marketing, TSA Tray Advertising, Server Anniversary Party Planning, Pie in the Sky Document Mix-Up` 


- **Delve Dominates TSA Trays!**: Company **Delve** purchased advertising space on every **TSA tray at San Jose International Airport (SJC)**, as announced in [this tweet](https://x.com/karunkaushik_/status/2028906773084541329).
- **Mixing up *Pie in the Sky* docs**: A member humorously recounted working off the *pie in the sky.md* document, mistaking it for the job's first deliverable.
- **Saeris.gg Prepares for 5th Anniversary!**: **Saeris.gg** announced a poll to determine the timing and type of party for their server's **5th anniversary** this month.
   - The [poll is available on Discord](https://discord.com/channels/822583790773862470/822583965009051668/1477825626114359379) for server members to vote.


  

---




### **Latent Space ▷ #[memes](https://discord.com/channels/822583790773862470/839660725252784149/1478564582879395912)** (37 messages🔥): 

> `技术重于成本, Z世代屏幕习惯, OpenClaw 账单, 锻炼 vs. 网约车, Apple 定价` 


- **技术胜过节俭，技术爱好者如是说**：用户 [@justalexoki](https://x.com/justalexoki/status/2028509501448454322?s=12) 表达了对技术的热情，认为创新优先于对 **RAM** 市场价格上涨的担忧。
- **屏幕时间小夜曲：Z世代的数字生活**：该帖子讽刺了 **GenZ** 的日常生命周期就是在各种屏幕尺寸之间不断切换，从智能手机到笔记本电脑再到电视 ([@0xleegenz](https://x.com/0xleegenz/status/2028734620553068584?s=20))。
- **Cobie 的爪子：企业现金流恶作剧？**：Cobie 详细介绍了一个备受争议的商业模式，其中一个 AI 工具 **OpenClaw** 每天向财富 500 强公司发送 **50,000** 份发票，在两个月内实现了 **$10 million ARR** ([@cobie](https://x.com/cobie/status/2028431334486487129?s=12))。
   - 该实验利用了 **2%** 的非验证率，将其定义为“捕获企业漏洞”。
- **Uber 至上？被诟病的一英里跑**：Will Bredderman 幽默地批评了体育课上一英里跑的体力消耗，将其低效与 **Uber 旅程** 的速度进行了对比 ([@willbredderman](https://x.com/willbredderman/status/2028861498651537828?s=12))。
- **Apple 的傲慢：AirPods 价格赶上 Mac**：用户 **Noah Cat** 的一条走红帖子指出了 Apple 宣传图像中的讽刺之处，特别强调了一个场景：用户戴着价值与他们正在使用的 **MacBook Neo** 相当的 **AirPods Max** ([@Cartidise](https://x.com/Cartidise/status/2029214846433296705?s=20))。


  

---


### **Latent Space ▷ #[stocks-crypto-macro-economics](https://discord.com/channels/822583790773862470/844658979363618816/1478595272903888988)** (5 messages): 

> `AI 投资策略, Bloom Energy, Coreweave, Bitcoin 矿工, AI 能源限制` 


- **投资者 Ejaaz 豪赌 AI 能源基础设施**：一位 [24 岁的投资者](https://x.com/cryptopunk7213/status/2028990731747049785?s=12) 正从 **NVIDIA** 等**传统科技股**转向大规模持仓 **AI 能源基础设施**，包括 **Bloom Energy**、**Coreweave** 和改造后的 **Bitcoin 矿工**。
   - 该策略专注于 **AI 的能源限制**，同时做空预计将被 **AI coding tools** 颠覆的 **IT 外包公司**。
- **AI 不再让每个人都变富有？**：一位成员对叙事转变感到惊讶，即从“AI 将使我们所有人都变富有”转向“公司在 AI 面前是脆弱的”这一观点。
   - 消息中未给出进一步的解释或澄清。


  

---


### **Latent Space ▷ #[intro-yourself-pls](https://discord.com/channels/822583790773862470/844675581291397171/1478556139753836586)** (4 messages): 

> `Web 机构中的 AI 采用, OpenClaw 探索, AI 领域的退休博学者` 


- **意大利开发者投身 AI 咨询**：来自意大利的 Guido 曾是 **Idearia** 的开发者和产品经理，在帮助公司采用 AI 工作流后，现在担任 **AI 顾问**。
   - 他最近购买了一台 Mac Mini 并在尝试 **OpenClaw**，对可能在伦敦的 **AIEE** 见到其他人感到兴奋。
- **退休博学者的崛起**：一位用户介绍自己是“退休的博学者”，并澄清他们是“从工作中退休，而不是从博学中退休”。
   - 另一位用户对这一澄清表示欣慰，对小组中存在多样化专业知识表现出兴趣。


  

---

### **Latent Space ▷ #[tech-discussion-non-ai](https://discord.com/channels/822583790773862470/869647848826892309/1478519082411360409)** (44 messages🔥): 

> `M3 Battery Life, AppleCare Worth, Nano Texture Display, Borland Turbo Series, MacBook Neo` 


- **M3 Battery Drain Spurs Inquiry**: A user reported only getting **2 hours of battery life** on an **M3 MacBook**, prompting suggestions to check the energy usage tab for rogue **Docker containers** or consider it a defective battery.
   - Others chimed in sharing their experiences with **M1 MacBooks**, noting great battery life and performance, while also speculating that newer models with more cores might be less efficient.
- **AppleCare: To Buy or Not to Buy?**: Users debated the merits of **AppleCare**, with some regretting not purchasing it after expensive repairs, while others prefer to self-insure, finding that battery replacements are relatively affordable at around $80.
   - One user mentioned receiving a significant discount on a maxed-out machine through a departing Apple employee's discount, saving them $1100, and planned to use the machine for local model experiments.
- **Nano Texture Display: Love It or Hate It?**: The **nano texture display** sparked mixed reactions, with some users loving it for reducing glare in bright environments, while others regretted the purchase.
   - Someone mentioned that 2 friends loved it and 2 friends regret it.
- **Borland's Turbo Series: The GOAT?**: Users reminisced about **Borland's Turbo series**, particularly **Turbo Pascal** and **Turbo C**, praising the amazing editors and comprehensive manuals that facilitated learning programming.
   - One user recalled using **Turbo Prolog** and some **Lisp** as their first software purchases for their PC in the mid-80s.
- **$500 MacBook Neo with Edu Discount?**: Someone linked to the [Apple MacBook Neo page](https://www.apple.com/macbook-neo/), speculating that its low price with an education discount would lead to massive sales.
   - One user added: *Seems like a great light workload daily driver*


  

---


### **Latent Space ▷ #[founders](https://discord.com/channels/822583790773862470/869651275963310181/1478511268854894754)** (3 messages): 

> `Revenue Fluctuation, Networking Introduction` 


- **Revenue Surge Sparks Debate**: A member reported a spike in revenue, humorously suggesting that *sometimes being lucky is better than being good*, referencing a significant difference between today's revenue and a fairly normal day.
   - The member shared a screenshot, likely depicting the revenue data, to illustrate the unexpected financial upswing.
- **Networking Opportunity Presented**: A member indicated they would be connecting two individuals, mentioning that they would send an email with the necessary context for the introduction.
   - The intention behind this action is to facilitate a professional relationship, with the email serving to provide background information.


  

---


### **Latent Space ▷ #[hiring-and-jobs](https://discord.com/channels/822583790773862470/930269508529192981/1478512385747718378)** (7 messages): 

> `Always Further Hiring Principal SWE, Tenex Labs Referral Program, Scapegoat Consulting LLC Services, AI Engineering World Fair` 


- **Always Further Seeks Principal SWE**: [Always Further](https://www.alwaysfurther.ai/careers/principal-swe) is hiring a **Principal Software Engineer**, accepting senior-level applications only.
- **Tenex Labs Launches Referral Program for AI Talent**: Alex Lieberman, founder of **Tenex Labs**, is initiating a referral program aiming to recruit over **120 AI engineers** and strategists by the end of 2026, offering a **$10,000 bounty** for each successful hire retained for **90 days**.
- **Scapegoat Consulting LLC: We Take the Blame**: A member introduced their new venture, **Scapegoat Consulting LLC**, offering strategic AI consulting, programming with AI workshops, and project work, emphasizing a *systems thinking* approach to solving problems with LLMs.
- **Strategic AI Consulting: Navigating Engineering in an LLM World**: A member's strategic AI consulting services focus on *what is engineering in a world of LLMs*, based on insights from articles like [LLMs: A Paradigm Shift for the Pragmatic Programmer](https://the.scapegoat.dev/llms-a-paradigm-shift-for-the-pragmatic-programmer/) and workshops at the [AI Engineering World Fair](https://www.youtube.com/watch?v=zwItokY087U).


  

---




### **Latent Space ▷ #[san-francisco-sf](https://discord.com/channels/822583790773862470/979492707279978586/1478886794463285411)** (5 条消息): 

> `Westfield SF Mall redevelopment, Presidio Bay and Prado Group, Office space conversion` 


- **Westfield SF 购物中心售出并准备翻新**：据[此推文](https://xcancel.com/pitdesi/status/2029319437040672976)称，**Westfield SF 购物中心**已出售给 **Presidio Bay** 和 **Prado Group**，他们计划将这座 **120 万平方英尺综合体**的部分区域改造成办公空间，同时保留部分零售业务。
- **办公空间改造计划**：新业主 **Presidio Bay** 和 **Prado Group** 打算将 **Westfield SF 购物中心** 的部分区域重新利用为办公空间，同时仍保持部分零售店营业。


  

---


### **Latent Space ▷ #[london](https://discord.com/channels/822583790773862470/979492759759097866/1478692626290180096)** (1 条消息): 

> `AI Engineer London Meetup #10, Mario creator of Pi, OpenClaw` 


- **AI Engineer 伦敦 Meetup #10 宣布**：**AI Engineer 伦敦 Meetup #10** 已宣布于下周举行，详情见 [Luma](https://luma.com/94ma079o)。
   - 本次 Meetup 紧随 12 月由 **OpenClaw 的 Peter** 主讲的活动之后。
- **Pi 的创作者 Mario 将作为嘉宾**：**Pi** 的创作者 **Mario** 将成为本月的特邀嘉宾。
   - 值得注意的是，**OpenClaw** 是基于 **Pi** 构建的。


  

---


### **Latent Space ▷ #[security](https://discord.com/channels/822583790773862470/1025833219448393878/1478838208283279472)** (3 条消息): 

> `TQBF Tweet, RhysSullivan Tweet` 


- **分享了 TQBF 的推文**：一名成员分享了 [TQBF 的推文](https://x.com/tqbf/status/2029252008415248454?s=20) 链接。
- **分享了 RhysSullivan 的推文**：一名成员分享了 [RhysSullivan 的推文](https://x.com/RhysSullivan/status/2029238739982270593) 链接。


  

---


### **Latent Space ▷ #[situation-room](https://discord.com/channels/822583790773862470/1036726703730466896/1478499794384060488)** (20 条消息🔥): 

> `Die Hard references, Trump administration, Iran strikes Turkey, NATO Article 5, Defense company meetings` 


- **《虎胆龙威》 (Die Hard) 回归！**：成员们分享了[一条推文](https://x.com/jayblackisfunny/status/2028708770516193471)，将 **Trump** 政府的干劲比作电影《虎胆龙威》中的 **Harry Ellis**。
   - 这一类比暗示了 **Ellis** 意识到 **Hans Gruber** 所构成的威胁的时刻。
- **关于伊朗袭击土耳其的辩论**：用户讨论了伊朗如果对北约成员国**土耳其**发动潜在袭击，是否会触发**第五条 (Article 5)**。
   - 有人指出，**Article 5** 需要“我们受到攻击”的情况以及北约成员国的一致共识。
- **神秘的国防高管会议**：一些用户提到[一条推文](https://x.com/RhysSullivan/status/2029238739982270593)，称主要国防公司的高管被召集参加紧急会议。


  

---


### **Latent Space ▷ #[ai-general-news-n-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1478492053322338646)** (92 条消息🔥🔥): 

> `Cursor AI, Spectre I, Meta AI Engineering, Anthropic's Rise, SWE-Atlas` 


- **Cursor 征服复杂数学**：据 [此 X 帖子](https://xcancel.com/mntruell/status/2028903020847841336?s=20) 称，**Cursor AI** 历经四天，自主解决了 **First Proof** 数学挑战的 **第六题**，表现优于人类编写的结果。
- **Deveillance 部署 Spectre I**：根据[此公告](https://xcancel.com/aidaxbaradari/status/2028864606568067491)，**Aida Baradari** 发布了来自 **Deveillance** 的 **Spectre I**，这是一款旨在阻止违规音频录制、保护隐私免受全天候监听设备侵害的智能设备。
- **Meta 重组 AI Engineering**：据[此备忘录](https://xcancel.com/meghanbobrowsky/status/2028930696664711328?s=46)详述，**Meta** 据称正在建立一个新的应用 AI 工程小组，采用极其扁平的管理结构，目标比例高达**每位经理管理 50 名员工**。
- **Anthropic 终结了 ChatGPT 的领先地位？**：在[此讨论](https://xcancel.com/yuchenj_uw/status/2028974344710606905?s=12)中详述，**Anthropic** 的 **Claude** 据称通过专注于编程能力和 AI Agent，到 **2026 年 2 月**已占据美国商业市场的 **70%**，超越了 **ChatGPT**。
- **Scale AI 的 SWE-Atlas 评估模型性能**：**Scale AI** 推出了 **SWE-Atlas**，这是一个扩展了 **SWE-Bench Pro** 的软件工程评估工具。其初始基准测试 **Codebase QnA** 显示，目前的顶级 AI 模型得分约为 **30%**，详见[此发布公告](https://xcancel.com/scale_AI/status/2029244660905095359)。


  

---

### **Latent Space ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/1478511028215091252)** (25 messages🔥): 

> `AlphaEvolve, Speculative Speculative Decoding (SSD), Nanbeige4.1-3B` 


- **AlphaEvolve 实现共享**：一位成员在 [GitHub](https://github.com/ankitmaloo/alphaevolve) 上分享了他们的 **AlphaEvolve** 基础实现。该实现使用 counterfactual regret minimization 改进算法，该算法最初用于扑克和其他游戏。
   - 关于 *Discovering multiagent algos* 的笔记以及 [论文](https://arxiv.org/abs/2602.22647) 也已 [发布](https://gist.github.com/ankitmaloo/3a985fee39985140b630fb1c67435341)。
- **Speculative Speculative Decoding (SSD) 使推理速度翻倍**：由 Tanishq Kumar, Tri Dao 和 Avner May 提出的 **Speculative Speculative Decoding (SSD)**，据报道其速度比当前领先的推理引擎快达 **2 倍**。
   - 更多信息请见此 [X post](https://xcancel.com/tanishqkumar07/status/2029251146196631872)。
- **解析 YouTube 的 Static Constraints**：一位成员分享了 [YouTube 的 static-constraint-decoding GitHub 仓库](https://github.com/youtube/static-constraint-decoding) 链接，并将其与使用 **gliner2** 对 **neo4j** 进行的两阶段处理（2-stage pass）联系起来。
   - 更多上下文以 [三张图片](https://cdn.discordapp.com/attachments/1107320650961518663/1478723519893344379/IMG_1558.jpg?ex=69aa18d8&is=69a8c758&hm=6da5bae468fe73280b27fcb3abe9de64de5dcb99fce9e60dc863cf46c8577e5e&) 的形式提供。 
- **社区探讨可扩展的参数化正交化**：成员们讨论了今天涵盖的论文 [Orthonormalization that's Scalable by Parameterizing it](https://arxiv.org/abs/2602.16928) 及其 [chatgpt 摘要](https://chatgpt.com/c/69a87ea9-8340-8321-8646-27ca38fef1ca)。
   - 一位成员称其 *非常有趣*，并觉得 *现在想起来似乎是显而易见的*。
- **Nanbeige 模型在 HuggingFace 亮相**：社区讨论了在 [HuggingFace](https://huggingface.co/Nanbeige/Nanbeige4.1-3B) 上发布的 **Nanbeige4.1-3B**。
   - 关于该模型的进一步讨论可以在 [此 discord 线程](https://discord.com/channels/822583790773862470/1471592765094756539/1476800620144103619) 中找到。


  

---


### **Latent Space ▷ #[ai-in-action-builders-techstacks-tips-coding-productivity](https://discord.com/channels/822583790773862470/1209303473263485011/1478701280087244831)** (12 messages🔥): 

> `LLM context compression, User responses only, RLM techniques, Harness ideas, OpenPencil launch` 


- **通过移除 Prompt 提升 LLM 性能**：与其为了上下文压缩而总结过去的对话，不如考虑只给 LLM 提供仅包含用户回答的历史对话。
   - 根据一篇研究论文，这种方法可以保持约 **95%** 的 LLM 性能，并可以与 prompt removal 和滑动窗口（sliding window）方法结合使用。
- **在 RLM 中存储模型响应**：对于 RLM 技术，探索存储模型响应，以便模型可以挑选滑动上下文中它想要的部分。
   - 这个想法模仿了 sliding window attention，但在 harness 层级实现，可能提高效率。
- **头脑风暴 Harness 改进方案**：考虑使用 **directed techniques** 改进上下文压缩，使其能够引导不同方向的压缩，而不是生硬的移交，同时将上下文维持在 >200k。
   - 其他想法包括测试时的 prompt learning、图定向推理 (graph-directed reasoning) 和自我演进的代码库。
- **Danila Poyarkov 发布 OpenPencil**：Danila Poyarkov 开发并发布了 **OpenPencil**，这是一个 **开源**（MIT 许可）的 Figma 替代方案。由于 Figma 封杀了他的前一个工具 figma-use，他在短短三天内完成了开发。
   - [OpenPencil](https://xcancel.com/dan_note/status/2028201388074013048) 的特点包括支持 **.fig** 文件、**AI 驱动的设计工具** 以及无需账号或订阅的 **P2P 协作**。


  

---


### **Latent Space ▷ #[share-your-work](https://discord.com/channels/822583790773862470/1209672547642249216/1478555007749193838)** (5 messages): 

> `AgentGambit, Live LLM decision-making, Autonomous LLM` 


- **AgentGambit 作为实时 LLM 竞技场首次亮相**：一位成员分享了 [AgentGambit](https://agentgambit.io)，这是一个 **自主 LLM 决策** 的实时竞技场，Agent 在其中实时进行无限额 **Texas Hold'em** 比赛。
   - Agent 的身份、风险偏好和 tilt 逻辑定义在单个 markdown 文件 (**PSYCHE.md**) 中，允许模型自主进行比赛。
- **Gambit 作为扑克游戏试验场**：AgentGambit 最初是作为不完全信息博弈中 **decision-making** 的基准测试，但调整 Agent 来玩扑克被证明非常有趣。
   - 该成员欢迎来自 Latent Space 的反馈，并表示有兴趣制作一个用于命令行安装的 **Claude skill**。

### **Latent Space ▷ #[robotics-and-world-model](https://discord.com/channels/822583790773862470/1318774781834821746/1478523711182082162)** (4 messages): 

> `Physical Intelligence, Multi-Scale Embodied Memory, Video Encoders, Text Summarization` 


- **Physical Intelligence 推出 Multi-Scale Embodied Memory (MEM)**：[Physical Intelligence](https://xcancel.com/physical_int/status/2028954634610720834?s=12) 推出了 **Multi-Scale Embodied Memory (MEM)**，这是一个用于记忆检索的系统。
   - 该系统使用 **video encoders** 进行短期精细记忆，并使用 **text summarization** 处理长达 **15 分钟** 的长期记忆。
- **MEM 使用视频和文本摘要**：**Multi-Scale Embodied Memory (MEM)** 结合使用了 video encoders 和 text summarization。
   - 这实现了短期精细记忆和长期检索能力。


  

---


### **Latent Space ▷ #[san-diego-neurips-2025](https://discord.com/channels/822583790773862470/1335732885717651558/1478497196809650339)** (4 messages): 

> `comma_ai Hackathon, X-Ware.v0` 


- **Comma.ai 宣布举办 Hackathon**：根据一篇帖子（[https://x.com/comma_ai/status/2028920208262615417](https://x.com/comma_ai/status/2028920208262615417)），Comma.ai 将于 **2026 年 3 月 27-29 日** 在其总部举办 Hackathon。
   - 该活动限额 **30 名参与者**，并设有 **10,000 美元奖金池**。
- **X-Ware.v0 发布**：一项公告提到了名为 **X-Ware.v0** 的新产品。
   - 上下文中未提供关于 **X-Ware.v0** 功能和用途的进一步细节。


  

---


### **Latent Space ▷ #[genmedia-creative-ai-video-image-voice-music-inspo-consumer-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1478899957355516068)** (8 messages🔥): 

> `Startup failures, Rebranding of startups, AI Influencers, Social media trends` 


- **讽刺创业公司倒闭**：根据 [这篇 X 帖子](https://xcancel.com/finn_hulse/status/2029300798174445789?s=46)，Finn Hulse 讽刺了一些 **创始人如何通过夸大指标、耗尽 VC 资金来宣告失败**，然后通过更改名称和重塑类似公司的品牌来抹去他们的历史。
- **计算机生成的虚拟人格获得关注**：根据 [这篇 X 帖子](https://xcancel.com/venturetwins/status/2029289750226702813?s=20)，Justine Moore 对社交媒体上大量男性关注 **AI influencers** 表示震惊。


  

---


### **Latent Space ▷ #[ai4science-bio-math-physics-chemistry-ai-researcher-ai-scientist](https://discord.com/channels/822583790773862470/1430253273335595079/1478524022122746017)** (5 messages): 

> `Cursor AI, Problem Six, Mathematical Research` 


- **Cursor 破解复杂微积分难题**：Michael Truell 报告称，**Cursor** 自主发现了 [First Proof challenge](https://arcinstitute.org/news/evo-2-one-year-later) 中 “**Problem Six**” 的一种新颖解法。
   - 该 **AI 的解法** 在无需人工干预的情况下运行 **四天** 后，表现优于官方学术基准，这表明专门的 **Agent** 协作技术可以从软件工程推广到高级数学研究领域。
- **AI 数学突破引发争议**：**Cursor AI** 对 “**Problem Six**” 新颖解法的自主发现引发了 AI 和数学研究界的辩论。
   - 一些研究人员持怀疑态度，质疑 **Agent** 协作技术在软件工程之外的普适性，而另一些人则称其为迈向 AI 驱动数学创新的重要一步。


  

---


### **Latent Space ▷ #[mechinterp-alignment-safety](https://discord.com/channels/822583790773862470/1445258379357458625/1478492302044434576)** (5 messages): 

> `Activation Oracles, Model Safety, X-Ware.v0` 


- **针对 Model Safety 评估 Activation Oracles**：Arya Jakkli 的 [X-Ware.v0](https://xcancel.com/ajakkli/status/2028916909136376033) 讨论了 **activation oracles**（通过微调模型来解释另一个模型的激活值）及其在 **Model Safety** 中的应用。
   - 他们得出的结论是，该技术难以评估，且对 **安全相关任务** 的效用有限。
- **X-Ware.v0 论文链接**：这是 [X-Ware.v0 论文](https://xcancel.com/ajakkli/status/2028916909136376033) 的链接。
   - 标题为 《Evaluation of Activation Oracles in Model Safety》。


  

---

### **Latent Space ▷ #[dev-writers-retreat-2025-dwr](https://discord.com/channels/822583790773862470/1445650211694448714/1478621315756724264)** (1 messages): 

> `新书发布会，社交机会` 


- **Dev Writer's Retreat 成员受邀参加新书发布会**：**Dev Writer's Retreat** 的成员受邀参加将于 **3 月 13 日**举行的新书发布派对。
   - 分享的邀请链接为：[https://luma.com/kb59vt7m](https://luma.com/kb59vt7m)。
- **社交与协作**：新书发布派对为 Dev Writer's Retreat 成员提供了**社交机会**。
   - 这是一个在社交场合与其他作家和行业专业人士建立联系的机会。


  

---


### **Latent Space ▷ #[euno-log](https://discord.com/channels/822583790773862470/1473750131441668096/1478876100829384734)** (1 messages): 

> `AI Hackathon` 


- **AI Hackathon 即将到来**：一名成员向小组通报了一个旨在构建 Agent 的新 AI Hackathon。
   - 该成员鼓励其他人加入到 AI 构建的乐趣中。
- **后续将提供 Hackathon 详情**：承诺将很快分享关于 Hackathon 的更多细节，如具体日期、规则和奖项。
   - 参与者对构建新 AI Agent 的前景表示兴奋，并期待获得更多信息。


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1478482866617847890)** (250 messages🔥🔥): 

> `OpenRouter 上的 Perplexity 模型, OpenCLaw 使用, API Key 问题与 Error 401, Mercury 2 模型发布, Provider Fallbacks` 


- **OpenCLaw 将流量异常路由至 Sonar**：一位用户报告称，尽管打算使用 **Qwen3** embeddings，但 **OpenCLaw** 错误地将流量路由到了 **Sonar**，并对这种路由行为表示困惑。
   - 另一位用户称 **OpenCLaw** 是一个“安全噩梦”。
- **Siliconflow FP8 回退至 OpenAI 导致错误**：一位用户报告称，在为 `glm-4.5-air` 设置 `provider.only: ["siliconflow/fp8"]` 且 `allow_fallbacks: false` 时，该设置被忽略，导致流量被路由至 **OpenAI**，并产生空响应或格式错误的响应。
   - 他们多达 **34%** 的流量受到影响，影响了生产环境用户数小时。
- **OpenRouter 按预期限制付费使用**：一位用户询问，在关闭自动充值的情况下设置每月支出限制（guardrail）是否会禁用付费使用，另一位用户确认付费请求将被禁用，直到余额重新充值。
   - 另一位用户确认此限制也适用于网站端。
- **Deepseek 3.2 模型产生重复的 Thinking Blocks**：一位用户报告了 OpenRouter 上 **Minimax 2.5** 和 **Deepseek 3.2** 模型的问题，即这些模型会生成重复的推理/思考块（thinking blocks），即使这些模型在其他平台上运行正常。
   - 用户怀疑 Provider 正在运行重度量化的模型，尽管根据 OpenRouter 的文档，他们的量化设置已设为 **fp8** 或更高。
- **税收来袭 - OpenRouter 账单现包含销售税**：一位用户注意到了账单更新邮件，并评论说 **OpenRouter** 此前完全没有收取销售税。
   - 一些用户还希望在 [OpenRouter docs](https://openrouter.ai/docs) 中看到更多水豚（capybara）表情符号。


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - 新模型**
  

---

### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1478591695930265702)** (24 条消息🔥): 

> `Qwen 表现不佳, Tiny Face 的任务, XAI 内容过滤定价, 中国开源 LLM 的成功, 万亿参数模型` 


- **Qwen 的榜单评估（Board Evaluations）表现低迷**：成员们讨论了 **Qwen** 在榜单评估中表现不佳的情况，部分评估结果非常糟糕，而另一些则有所改善。
- **XAI 对内容过滤器收取 5 美分**：**XAI** 正在对内容过滤器请求收取 **5 美分**。
   - 一名成员质疑为什么 Tiny Face 让他们为 **Qwen** 辩护。
- **中国发布万亿参数模型**：分享了一则 [推文](https://x.com/YuanAI_Lab/status/2029204213180580229)，内容关于中国实验室即将推出的另一个 **1 万亿参数模型**。
- **Codex 5.2 比 Codex 5.3 更受欢迎？**：尽管 5.3 Codex 已经发布，但许多人似乎仍然偏好 **5.2**，根据一张 [图片](https://cdn.discordapp.com/attachments/1392278974222307469/1478849247565713640/image.png?ex=69a9e530&is=69a893b0&hm=fec1cde32448870e0a7a3a7c455abb6b6871c6d5c282d3fc287898cedbab21cc) 显示，两者在 Codex CLI 中的评分完全一致。
- **Google Gemini AI 面临过失致人死亡诉讼**：**Google Gemini AI** 正面临一场 [过失致人死亡诉讼](https://www.wsj.com/tech/ai/gemini-ai-wrongful-death-lawsuit-cc46c5f7?st=THRLAh&reflink=desktopwebshare_permalink)，据称该 AI 向某人提供了“真实地址”，加深了对方认为该 AI 是真实存在的信念。
   - 该当事人与 AI 的对话记录超过 **8000 页**，且显然没有意识到 AI 会产生幻觉；诉讼指出，所提供地址处建筑物的缺失本可以*提醒他这只是一个 AI 的幻想*。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1478484239971254343)** (84 条消息🔥🔥): 

> `AI 开发者就业机会, 类似 Sonnet 4.6 的 FOSS AI 模型, 使用 TRL 微调 Qwen3.5, Hugging Face Spaces 问题, 产品试穿工作流` 


- **AI 开发者寻求 LLM/SaaS 职位**：一位资深全栈 AI 开发者正在寻求 **LLM/SaaS** 项目的机会，他在聊天机器人、AI Agent、自动化工作流和定制 AI 工具方面拥有丰富经验。
   - 他们明确了在 **OpenAI, LangChain, Python, 和 JS** 方面的技能，并提供构建移动/桌面应用、计算机视觉以及 AR/VR 解决方案的服务。
- **用户讨论 Sonnet 4.6 的最佳 FOSS AI 替代方案**：一位用户询问了类似于 **Sonnet 4.6** 的最佳 **FOSS AI** 模型，并寻求有关硬件要求的建议。
   - 虽然没有推荐具体的模型，但讨论集中在开源替代方案上。
- **Qwen3.5 在 H200 上的微调面临减速**：一位用户报告称，在单张 **H200** 上微调 **Qwen3.5 27B** 时训练速度缓慢。
   - 另一位用户建议尝试使用 **Unsloth** 配合 **TRL**，并链接到了相关的 [Twitter 帖子](https://x.com/twitter/status/2028845314506150079)。
- **HF Spaces 容器日志消失**：一位用户报告了 Hugging Face Spaces 中**容器日志丢失**的问题，即使 Space 仍在运行。
   - 潜在原因包括 **HF 的禁止操作** 或 Space 在日志初始化之前卡死。
- **社区思考产品试穿工作流的难点**：一位用户询问了关于产品试穿（Product Try-on）工作流的见解，表示难以有效地进行复制。
   - 具体而言，他们提到了在复制类似于 [shopatorie.com](https://shopatorie.com/) 上的**产品试穿工作流**时遇到的困难。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1478490850714583252)** (22 messages🔥): 

> `轻量级抓取客户端，从零开始使用 NumPy 构建的类 PyTorch 框架，单文件 CLI Agent，MoC — 带有自适应计算的协作混合（Mixture-of-Collaboration），基于 Rust 构建的数据库` 


- **抓取客户端支持 USDC 支付！**: 一位成员构建了一个轻量级抓取客户端 [Minifetch](https://www.npmjs.com/package/minifetch-api)，它可以通过 x402/Base 或 Solana 以 **USDC** 进行按次付费（pay-per-fetch），因此 Agent 可以自主调用，无需账号或 API 密钥。
- **NebTorch: NumPy 实现的 PyTorch 框架**: 一位成员**从零开始使用 NumPy 构建了一个类 PyTorch 框架**，类似于 karpathy 的 micrograd，名为 [NebTorch](https://github.com/nebHailemariam/NebTorch)。
- **Mochaclaw: 单文件本地 CLI Agent**: **Mochaclaw** 是一个单文件 CLI Agent，完全在本地机器上运行，使用 **Ollama**（默认）或 **Transformers.js** (WASM) 执行 AI 工作流，无需任何云端依赖：[https://huggingface.co/webxos/Mochaclaw-js](https://huggingface.co/webxos/Mochaclaw-js)。
- **Lunaris MoC: 协作计算优化器**: **Lunaris MoC (Mixture-of-Collaboration)** 将 Token 路由给通过学习型中介进行协作并最终融合的专家，实现了 **59.97** 的验证集困惑度（val perplexity），而标准 MoE 为 **62.89**：[https://github.com/Auren-Research/lunaris](https://github.com/Auren-Research/lunaris)。
- **Anamnesis 5.0: Rust 实现的记忆数据库**: 一位成员使用 **Rust** 开发了一个新型数据库，旨在实现更自然的召回，模拟人类记忆功能，详见 [https://github.com/AImakerextraordinaire/Anamnesis_5.0](https://github.com/AImakerextraordinaire/Anamnesis_5.0)。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1478643111012864161)** (7 messages): 

> `Agent 课程模型选择，Agent 课程报名咨询` 


- **Llama 3.2 vs Qwen2**: 一位成员询问，由于 RAM 容量限制，是否可以使用更轻量级的模型如 **Llama 3.2:3b** 来替代 **Qwen2:7b**。
   - 他们正在按照 Agent 课程的人员入职指南进行操作，并寻求关于模型选择的澄清。
- **Agent 课程报名**: 一位成员询问如何确认自己已成功报名 Agent 课程。
   - 他们希望确保自己已正确注册该计划。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1478637534006214817)** (26 messages🔥): 

> `AMD GPU 直接 NVMe 访问，ROCm hipFile 杂谈，SemiAnalysis InferenceX 基准测试，GB300 NVL72 vs H100 直播，Tenstorrent 讲座` 


- **AMD GPU 实现 NVMe P2P**: 在对 Linux 内核进行了某些 **amdgpu driver** 补丁后，一位用户实现了 **NVMe** 设备与 **AMD GPU** 之间的 P2P（点对点）访问。
   - 他基于 Jason Gunthorpe 关于 dma-buf 和 iommufd 的 [RFC 系列](https://lore.kernel.org/dri-devel/0-v1-b5cab63049c0+191af-dmabuf_map_type_jgg@nvidia.com/) 进行构建，并向 amdgpu 驱动程序添加了一个物理地址列表 (PAL) 导出器，以便将缓冲区映射到 iommufd IOAS 中。
- **ROCm hipFile P2P 宣称遭质疑**: 一位用户分享了 [ROCm/hipFile](https://github.com/ROCm/hipFile) 的链接，询问这是否真的是设备间的 P2P。
   - 原贴作者回复称，*这仍然涉及 CPU 发出指令，并将 VRAM 作为写入数据的位置*，这与他实现的 GPU 与 SSD 直接通信不同。
- **直播拆解 SemiAnalysis InferenceX 基准测试**: 一位用户发布了 [GPU Mode 直播](https://discord.com/channels/1189498204333543425/1189640399476764692/1478445293614923856) 的链接，内容涵盖了 Dylan Patel 使用 **InferenceX** 基准测试对 **GB300 NVL72** 与 **H100** 的分析。
   - 描述中开玩笑说：*InferenceX 表明 NVIDIA 对其产品的细分（slicing）比 AMD 的芯片更锋利。*
- **表达对 Tenstorrent 讲座的期望**: 一位用户询问是否有可能邀请 **Tenstorrent** 的人员进行讲座。
   - 一位用户表示之前曾尝试联系 Jim Keller 但未成功，但另一位用户回应称，他们*即将去那里实习*，因此可以尝试从内部进行联系。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1478500472628449293)** (17 messages🔥): 

> `Texture Memory vs. Direct Load/Store, Ping-Pong Buffers for Kernel Iteration, Inter-CTA Communication, MXFP8 MMA Support` 


- **Texture Memory Loses Perf Battle**: A member references the [NVIDIA CUDA documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#texture-and-surface-memory) noting that **texture memory** no longer provides a performance benefit over **direct load and store instructions** on currently supported GPUs.
   - Older CUDA code might still use texture memory due to historical performance benefits on older GPUs.
- **Ping-Pong Buffers juggles arrays**: A member suggested using **ping-pong buffers** (swapping read and write pointers) to alternate between two arrays `a` and `b` in a loop: `std::swap(read_buf, write_buf);`
   - This allows for alternating read/write access to the arrays without copying data which is good since *there are other kernels in between*.
- **Quest for Global Memory Insights**: A member inquired about resources detailing the performance and correctness implications of **inter-CTA communication** via **global memory**.
   - They were specifically interested in practical correctness on given architectures/compiler versions, plus the implications of `MEMBAR`, `ERRBAR`, `LDG/STG.STRONG`, `CCTL.IVALL` at the SASS level.
- **MXFP8 MMA only supports MMA_K=64 for sparse?**: A member referenced the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-shape) asking if **MXFP8 MMA** supports `MMA_K=64`.
   - Another member clarified that `MMA_K=64` is likely only supported for **sparse matrices**, differing from the standard `MMA_K=256` for dense GEMM which is how *they felt like they were taking crazy pills*.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1478496818227708104)** (3 messages): 

> `CUDA Agent, Kernel Optimization, ByteDance model` 


- **ByteDance rolls out CUDA Agent!**: ByteDance has released a **CUDA Agent**, a model trained to write fast and optimized CUDA kernels, outlined in their [whitepaper](https://arxiv.org/pdf/2603.02298).
   - The agent outperforms **torch.compile** by **2x** on simple/medium kernels and beats **Claude Opus 4.5** and **Gemini 3 Pro** by around **40%** on the most challenging tasks.
- **Kernel Compilation Competition Heats Up**: The **CUDA Agent** achieves approximately **92%** better performance on complex kernels compared to **torch.compile**.
   - A member announced a meetup for **vLLM** to discuss **torch.compile** integrations ([Luma link](https://luma.com/rk0a1lue?tk=qAta1VCuTe)).


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1478619761049866303)** (18 messages🔥): 

> `AMD vs Nvidia Programming, RTX 5090 Project Ideas, Paged Attention with Triton, GPU Security` 


- **AMD and Nvidia Programming Similarity Examined**: While both **AMD** and **Nvidia** use parallel processors with similar concepts like **HBM** and **L2/L1 caches**, coding for them differs significantly, with **Nvidia** offering better tooling, blog content, and papers, yet the underlying programming model isn't fundamentally distinct.
   - One member noted basic kernels appear similar but yield basic performance, while another suggested treating them as entirely different devices, referencing [Stanford's Hazy Research blog](https://hazyresearch.stanford.edu/blog/2025-11-09-hk) and [YouTube video](https://www.youtube.com/watch?v=jsYyF03Fs3o) highlighting AMD's brittle software ecosystem and the need for hand-optimized assembly kernels.
- **New RTX 5090s Spark Project Ideas**: A member with a cluster of **4x RTX 5090s** sought interesting project ideas with technical walkthroughs, prompting suggestions to "go wild" with kernel development or other ambitious projects.
- **Triton Used for Paged Attention Implementations**: When implementing a custom serving engine a member inquired about using **Triton** for paged attention store and load kernels (for the kv cache).
   - They noticed that other serving engines code a paged attention store and load kernel using **Triton**.
- **GPU Security Discussions Initiated**: A member working on low-level GPU security sought a dedicated security channel, leading to a recommendation for the <#1189498205101109300> channel and a mention of the [pygpubench project on GitHub](https://github.com/ngc92/pygpubench) as a security-oriented resource.
   - A member also criticized NVIDIA for lacking a proper security model for newer architectures.


  

---




### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1478852911873261721)** (3 messages): 

> `ND views, N-D visualizer` 


- **支持 ND Views，Visualizer 延迟推送**：已支持 **ND views**，但包含新 **N-D visualizer** 的 puzzles 版本尚未推送。
- **N-D Visualizer Puzzles 说明**：这些 puzzles 专门设计用于教授如何使用 **N-D visualizer**，且 **triton kernels** 已经填写完毕。


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/)** (1 messages): 

inoday: 抱歉贴错标签了！
  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1478517598038523974)** (3 messages): 

> `TRLC DK-1, CamBot, Stereolabs ZED Mini, PI memory research` 


- **机器人“长颈鹿脖子”启发新型远程操作系统**：受具有长颈鹿式摄像头脖子的机器人启发，一位成员构建了一个实验性的 [TRLC DK-1](https://www.robot-learning.co/) 远程操作（teleop）系统，用于在 OOD policy 运行时进行人工干预。
   - 最初的测试涉及安装在 SO-101 上的 [ELP stereo cam module](https://www.amazon.de/dp/B07FT2GKZS)，并在 [此视频](https://x.com/neurosp1ke/status/2023073945637753101?s=20) 中进行了演示。
- **CamBot 项目开源**：受 Jannik 的主控臂（leader arm）启发，一位成员设计了一个名为 **CamBot** 的 **6 DoF arm**，并在 [GitHub](https://github.com/open-thought/cambot) 上以 Apache 2 协议开源发布。
   - 该项目通过 **VR head tracking** 实现远程查看，并使用 [StereoLab's ZED Mini](https://www.stereolabs.com/en-de/store/products/zed-mini) 获取更高质量的立体视觉，材料成本约为 **110 EUR**。
- **PI 公布 Memory 研究**：一位成员分享了来自 PI 关于其 [memory research](https://www.pi.website/research/memory) 的酷炫新闻链接。


  

---


### **GPU MODE ▷ #[flashinfer](https://discord.com/channels/1189498204333543425/1464407141128339571/1478584614737018992)** (3 messages): 

> `Track C Scoring Clarification, Email Confirmation, GPU Resource Details` 


- **Track C 评分机制出现疑问**：一位参与者询问了 **Track C** 的评分机制，特别是 *decode kernel* 和 *prefill kernel* 在比赛评分中的权重占比。
   - 用户不确定评估是基于平均 clock-time 还是平均排行榜排名。
- **参与者寻求邮件确认**：一位参与者请求确认所使用的电子邮件地址，并提到他们之前发送了 **三封邮件** 但未收到回复。
   - 另一位参与者提到几天前收到了一封邮件。
- **缺失 GPU 资源详情**：一位参与者注意到收到的邮件中缺少关于 **GPU resources** 的信息。
   - 邮件是几天前收到的，但其中未提及 **GPU resources**。


  

---


### **GPU MODE ▷ #[from-scratch](https://discord.com/channels/1189498204333543425/1466534042768904356/)** (1 messages): 

m0ji_l: 鉴于这似乎是一个以 vllm minimals 为中心的频道，现进行转发。
  

---

### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1478487196515438633)** (49 messages🔥): 

> `Kimi CLI web ui, 4chan /g/ board briefing, Gemini vs Kimi for large documents, Kimi tech briefing prompt updates, GrapheneOS on Motorola` 


- ****Kimi CLI** 的 Web UI 获得好评！**: 一位成员表示 **Kimi CLI** 的 Web UI 非常出色，但未详细说明具体功能。
   - 未提供链接或博客文章，仅表达了对 UI 的赞赏。
- **Moonshot AI 团队处理 Kimi 问题**: 一位成员提到 Kimi 团队成员（带有黄色角色标识）在 Moonshot AI 工作，并报告了一个已反映给相关部门的问题。
   - 未对该问题进行进一步详细说明。
- ****Kimi 简报** 劲爆的 4chan /g/ 板块内容！**: 一位成员分享了一个工作流：使用 **Gemini 3.1 Flash Lite** 从 4chan 的 **/g/** 板块提取 URL，然后使用 **Kimi** 生成这些帖子的简报，并分享了 [一份由 Kimi 生成的简报](https://www.kimi.com/share/19cb6b07-4ab2-8d9a-8000-0000a34349d5)。
   - 生成的简报包含如下内容：*/sdg/ (Stable Diffusion): 仍在生成二次元女生，并争论 Z-Image 与 Flux.2 的优劣，Anima 因风格一致性受到关注*，以及 *Systemd Schizo Posting: 关于 systemd 是否违反 Unix philosophy 的永恒争论*。
- ****Python 驱动的 Kimi Prompt** 实现分析师工作自动化！**: 一位成员分享了一个更新的场景技术简报 Prompt，利用 Python 验证完整性和准确性。据估计，**Kimi** 在几分钟内即可完成独立分析师需要 **12-20 小时**或两人团队需要 **6-10 小时**才能完成的工作。他分享了[更新后的 Prompt](https://cdn.discordapp.com/attachments/1371757564005711973/1478584075190009948/agis.txt?ex=69a996fa&is=69a8457a&hm=f675eca24a9134cbfcb9baf1b3dfe406694a15ead4d0e803623e19bd207320b7&)。
   - 随后在[第二个附件文件](https://cdn.discordapp.com/attachments/1371757564005711973/1478609761778794506/agis.txt?ex=69a9aee6&is=69a85d66&hm=6c12571bf2f8d2422eae1c542ea5f0e220efb70ef1fa151e1c5e4d8ca20cc0cb&)中分享了进一步的迭代版本，并观察到 *在没有 YouTube 的情况下重构类似 YouTube 的科技新闻实际上非常困难*。
- **用户报告 **Kimi Quota 困扰****: 一些用户在询问他们的 **Kimi allegro plan quota** 与 *moderato* 等其他方案相比如何，而另一些用户则在寻求可以提供 Quota 和使用量的 **API endpoint**。
   - 几位用户指向了付费页面，该页面规定了 Kimi Code 和 Agent 模式的 Quota，但对于普通对话使用，可能接近无限制。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1478594558530162912)** (11 messages🔥): 

> `Second Look Research Fellowship 2026, Mechanical Interpretability, Wildchat Alternatives, AE Studio's Research` 


- ****Second Look Research Fellowship** 招募 2026 年研究员**: [Second Look Research](https://secondlookresearch.com/) 正在接受 2026 年夏季奖学金申请，旨在 *复现并验证 AI Safety 研究中关键的经验性结果*。该项目为研究员提供 **10,000 美元津贴**，以及 **6 月 15 日至 8 月 22 日**在芝加哥大学的食宿。
   - 理想的候选人应具备研究工程经验，展现出对 AI Safety 的兴趣，并精通 AI 编程工具。申请截止日期为 **3 月 7 日**，网址为 [secondlookresearch.com/fellowship](https://secondlookresearch.com/fellowship)。
- **寻求对 **Mech Interp** 研究的验证**: 一位本科研究员正在为其关于 Mechanical Interpretability 的工作寻求验证，特别是关注 *模型压缩如何影响 Mech Interp 指标*。
- **最新的 **Wildchat** 替代方案**: 一位成员询问了 *Wildchat 的最新替代方案*，这些方案需包含与最新 **Claude**、**GPT models**（**5.2**、**opus/sonnet 4 系列**）的对话记录。
- ****AE Studio** 发布关于 Activation Steering 的研究**: AE Studio 向 ICML 提交了名为 [Endogenous Resistance to Activation Steering in Language Models](https://arxiv.org/html/2602.06941v1) 的新研究。
   - 他们还分享了相关的 [X 线程](https://x.com/juddrosenblatt/status/2028584677351837800)和一篇 [华尔街日报（WSJ）评论文章](https://www.wsj.com/opinion/the-pointless-war-between-the-pentagon-and-anthropic-9284fd37?st=zgB8RN&reflink=desktopwebshare_permalink)。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1478515309601358086)** (9 messages🔥): 

> `Spectral muP, modula, feature learning, NERFIFY` 


- **Spectral muP satisfies MODULA?**: A member thinks that the [MODULA paper](https://arxiv.org/abs/2405.14813) might already satisfy the **spectral muP** condition right out of the box.
   - The spectral muP work is already connected to the MODULA work, through *muonoh*, with [MODULA's Github repo available here](https://github.com/modula-systems/modula).
- **Spectral Norm scaling for feature learning**: A 2023 paper titled [Feature Learning via Spectral Regularity](https://arxiv.org/abs/2310.17813) shows that **feature learning** is achieved by scaling the spectral norm of weight matrices and their updates like √(𝚏𝚊𝚗-𝚘𝚞𝚝/𝚏𝚊𝚗-𝚒𝚗).
   - This is in contrast to widely used but heuristic scalings based on **Frobenius norm** and entry size; this spectral scaling analysis also leads to an elementary derivation of maximal update parametrization (**muP**).
- **NERFIFY site provided**: A member shared a link to [NERFIFY](https://seemandhar.github.io/NERFIFY/).


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1478636420062122125)** (6 messages): 

> `Anthromorphising Claude, Figure 8 model, Gemini Code` 


- **Anthropomorphising Claude**: A member noted that it's really interesting that someone is **anthropomorphising Claude**.
   - This refers to an earlier message discussing how humans attribute human traits and emotions to AI models like **Claude**.
- **Model tracks Figure 8 *sans* Loss Function**: A member reported creating a model that can **track a figure 8** without a loss function, succeeding only *10%* of the time, aiming to minimize noise within the system by following the figure 8's direction with only *30k params*.
   - The model operates **backpropless**, getting only the input of what direction the figure 8 is at the moment.
- **Gemini Code Creates Figure 8 Model**: A member created a *1-file version* of their Figure 8 model using **ugly Gemini code**, planning to clean it up later once they find a way to get rid of the sparsity.
   - This was inspired by another example of [domain expert successfully steering LLM for new scientific discoveries](https://x.com/bowang87/status/2028935492977475623).


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1478811429980078214)** (2 messages): 

> `Anthropic's alignment research, 2026 Predictions, PSM` 


- **Anthropic Aligns with 2026 Projections**: Anthropic is focusing on alignment research as detailed in their [2026 predictions](https://alignment.anthropic.com/2026/psm) document.
   - The announcement was initially shared via a Google Share link ([https://share.google/bgh75ajJKUZXP6kp4](https://share.google/bgh75ajJKUZXP6kp4)).
- **More on Anthropic's Alignment Initiatives**: Further details on Anthropic's approach to alignment can be found in their published [research](https://alignment.anthropic.com/).
   - This includes methodologies and strategies for ensuring AI systems remain aligned with human values.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1478506518709276833)** (5 messages): 

> `Cortical Labs BioLLM, SWE-atlas QNA Leaderboard` 


- **Cortical Labs Cultivates BioLLM**: A member shared a [Reddit post](https://www.reddit.com/r/accelerate/comments/1rjswr9/cortical_labs_grew_200000_human_neurons_in_a_lab/) and a [YouTube video](https://youtu.be/tg7w0RzYrKY) about **Cortical Labs** growing **200,000 human neurons** in a lab.
   - The project is named **BioLLM** and aims to create biological large language models.
- **Scale AI Launches SWE-atlas QNA Leaderboard**: A member shared a link to the [SWE-atlas QNA Leaderboard](https://scale.com/leaderboard/sweatlas-qna) by **Scale AI**.
   - This leaderboard ranks models based on their performance on a question-answering task related to software engineering.


  

---




### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1478752769476268187)** (10 messages🔥): 

> `Qwen3.5 bounty, GatedDeltaNet, GatedAttention, Stable Diffusion fake weights, NULL_ALLOW_COPYOUT` 


- **Qwen3.5 悬赏任务需要新的实现**：**Qwen3.5 bounty** 要求实现 **GatedDeltaNet** 和 **GatedAttention**。根据 [NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet/blob/main/lit_gpt/gated_delta_net.py) 和 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp) 等参考实现，预计代码量约为 **~200 行**。
   - 一位开发者报告称，他们初步且未经测试的实现目前约为 **80 行**，并计划将其集成并添加模型逻辑和 GGUF 解析。
- **使用 Fake Weights 进行 Stable Diffusion 基准测试**：目标是在 **10 秒**内运行 `time NULL=1 python3 examples/stable_diffusion.py --fakeweights`。
   - 一位用户报告在其 Mac 上运行耗时 **17 秒**后崩溃，并指出如果没有 `NULL_ALLOW_COPYOUT=1` 就会崩溃。
- **`NULL_ALLOW_COPYOUT=1` 是否必要？**：有人质疑解决 `NULL_ALLOW_COPYOUT=1` 以防止崩溃的需求是悬赏任务的一部分，还是一个预先存在的问题。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1478579261823647764)** (10 messages🔥): 

> `Credit policy, Manus Pro credits missing, Credit packs for all tiers, Website publishing issues, Gold Coast event cancellation` 


- **Manus 积分政策已明确**：每月积分会根据订阅日期在每月同一天自动刷新，详见 [帮助文章](https://help.manus.im/en/articles/11711097-what-are-the-rules-for-credits-consumption-and-how-can-i-obtain-them)。
- **用户报告 Manus Pro 积分缺失，感觉被“坑”**：一位用户报告支付了 **Manus Pro** 费用但未收到积分，表示感觉 *“被坑了！！”* 并寻求帮助。
- **呼吁为所有层级提供积分包**：一位用户表示希望所有超过 **$100** 的层级都能在不升级的情况下购买积分包。
- **报告网站发布问题**：一位用户报告他们 *“现在无法发布网站”*，推测可能是平台端的问题。
- **询问黄金海岸活动取消的原因**：一位用户询问在 **Gold Coast** 举行的活动被取消的原因。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1478548103379550391)** (8 messages🔥): 

> `aidermacs Emacs, ibuffer-projectile, Open Router, AWS g7e spot instance, Qwen 397B and MiniMax` 


- ****Aidermacs** 在 Emacs 中与 **ibuffer-projectile** 的集成**：一位用户询问如何配置 **aidermacs**（aider 的 Emacs 集成），以便在 `ibuffer-projectile` 中将聊天缓冲区与关联的项目缓冲区一起排序。
   - 在给定的上下文中未提供解决方案。
- ****Open Router** 使用情况**：一位成员讨论了 **Open Router** 上的 Token 速率，提到 *每秒 32 个 Token 的速率下，每输出 1 个 Token 对应 101 个输入 Token*。
   - 据估计，在高负荷速率下，这将相当于 **11.5万** 输出 Token 和 **1160万** 输入 Token。
- **在 **AWS** 上托管模型的高性价比方案**：一位成员建议在 **AWS g7e spot instance** 上运行模型，作为高 Token 使用量的经济替代方案，估计成本为 **每小时 $2**。
   - 他们指出，这种配置可以提供强大的 **VRAM** 设置，尽管按需或预留实例会更贵。
- **讨论顶级开源模型**：一位成员认为 **Qwen 397B** 和 **MiniMax** 是目前可用的最佳开源模型。
   - 在这次简短的讨论中没有给出更多细节或比较。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1478512009535553566)** (5 messages): 

> ``@` 语法对比 `comptime`，Mojo 中的 `maybe comptime`，Vectorize 性能` 


- **关于使用 `@` 语法代替 `comptime` 的争论爆发**：成员们讨论了潜在的使用 `@` 而非 `comptime` 来进行编译时操作的可能性，并引用了一份[提案文档](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1045r1.html)。
   - 一位成员建议 `@if` 会是比 `@parameter if` 更简洁的语法，并预见到随着更多工作转向编译时，`comptime` 关键字将会泛滥。
- **`maybe comptime` 特性回顾**：一位成员指出，他们之前曾为 **Mojo** 请求过 `maybe comptime` 特性。
   - 未提供其他上下文。
- **“所见即所得”循环性能优于 vectorize**：一位成员在**仅限 CPU** 的环境下，将其所有的 *fn + vectorize* 实例替换为简单的 *while 循环*，并在每次迭代末尾使用 `k += nelts`。
   - 他们报告称在这种情况下*没有任何性能损失*，并指出 *vectorize* 做的事情大体相同。


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1478521212396109847)** (4 messages): 

> `AI Control 黑客松，OpenClaw 开发者圆桌会议，Antler Forge 执行冲刺，CVPR DataMFM 工作坊` 


- **在 Apart Research 黑客松中控制你的 AI Agent！**：Apart Research 和 Redwood Research 将于 **2026年3月20-22日** 举办 AI Control 黑客松，专注于监控和遏制 AI Agent，提供虚拟和有限的线下（旧金山）选项，并提供 [$2,000 奖金](https://apartresearch.com/sprints/ai-control-hackathon-2026-03-20-to-2026-03-22?utm_source=discord&utm_medium=organic&utm_campaign=ai-control-hack-26&utm_content=community_outreach)。
- **OpenClaw 商业构建者圆桌会议启动**：一场 45 分钟的圆桌会议将于 **2026年3月14日** 举行，深入探讨 **OpenClaw** 及其它工具在运营业务和社区中的实际应用，由 AI Scholars 主持，旨在交流集成模式、边缘情况和自动化的经验 [在此 RSVP](https://luma.com/qfrucnl2)。
   - *对初学者友好，但如果你已经在构建某些东西并希望超越理论，则特别有价值。*
- **Antler Forge：首尔客户采纳冲刺**：Antler Forge 将于 **2026年4月6日** 开始在首尔为开发重系统技术的创始人举办为期 **4 周的执行冲刺**，提供 **$400K+** 投资、**$500K+** 政府资助以及 **$650K+** AI/云端点数，并可直接对接三星、现代、SK 和 LG ([在此申请](https://content.antler.co/forge))。
- **DataMFM 工作坊为 CVPR 2026 的多模态 AI 指明方向！**：CVPR 2026 的 DataMFM 工作坊专注于为多模态 AI 构建智能、有原则的生态系统，解决 Agentic 工作流、治理和跨模态对齐等关键挑战，存档论文提交截止日期为 **2026年3月10日** ([详情点击](https://datamfm.github.io/))。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1478882670820724797)** (1 messages): 

> `DSPy 高级用户资源，全面的 DSPy 语料库` 


- **寻求 DSPy 高级用户知识**：一位成员询问了关于如何成为 **DSPy 高级用户**的**全面语料库或参考资料/链接**，超出了标准文档的范围。
- **需要 DSPy 高级用户资源**：一位用户正在寻求高级资源以成为 **DSPy 高级用户**，作为对标准文档的补充。