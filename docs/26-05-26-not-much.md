---
companies:
- deepseek
- google-deepmind
- langchain-ai
- anthropic
- openai
- alibaba
- sakana-ai
- stanford
- oxford
- ai2
date: '2026-05-26T05:44:39.731046Z'
description: '**Harness工程（Harness Engineering）**正成为编程智能体（coding agents）的核心竞争力，它强调的是“**模型
  + Harness框架 + 评估循环**”的技术栈组合，而非仅仅依赖更强大的基础模型。


  **DeepSeek** 正在组建 Harness 团队，旨在优化交互与验证循环；与此同时，**Google 的 Gemini 托管智能体（Managed Agents）**与
  **LangChain** 正在将上下文治理（context governance）和动态技能路由（dynamic skill routing）等 Harness
  概念正式化。


  诸如 **DeepSWE** 等新基准测试与真实开发者体验高度对齐，**Qwen3.7 Max** 和 **Claude Opus 4.6** 在其中展现了强大的智能体编程性能。**Anthropic**
  为 **Claude Code** 推出了安全引导插件，使安全相关的 PR 评论减少了 30-40%；**OpenAI** 则强调了 **GPT-5.5** 在
  Codex 中表现出的文档解析能力提升。


  在科研领域，**Claude Mythos** 解决了埃尔德什（Erdős）问题 #90，其证明路径比以往模型更简洁，展示了通过合适的 Harness 所激发的潜能。论文《语言模型也需要睡眠》（Language
  Models Need Sleep）提出了一种针对长程记忆的类睡眠巩固阶段，以解决持久性上下文存储的瓶颈。


  开源研究智能体如 **QUEST**（参数量 2B–35B）推动了长程事实搜寻和引用归因（citation grounding）的发展；而由 Sakana、斯坦福、牛津及
  AI2 联合发布的 **CUSP 基准测试**，则专门用于评估当前模型在科学领域的综合实力。'
id: MjAyNS0x
models:
- qwen-3.7
- claude-opus-4.6
- gpt-5.5
- mythos
- quest-2b-35b
people:
- sebastienbubeck
title: 今天没发生什么事。
topics:
- harness-engineering
- agent-infrastructure
- coding-benchmarks
- security-guidance
- long-horizon-memory
- context-compression
- sleep-phase
- math-problem-solving
- fact-seeking
- citation-grounding
- science-evaluation
---

**a quiet day.**

> AI News for 5/23/2026-5/26/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap



**Agent Harnesses, Coding Benchmarks, and the Shift Beyond “Just the Model”**

- **Harness engineering is becoming the main differentiator for coding agents**: Several posts converged on the same thesis: the winning stack is now **model + harness + eval loop**, not just a stronger base model. A long Zhihu summary argued that [DeepSeek is explicitly building a harness team](https://x.com/ZhihuFrontier/status/2059180748637376843) to close the loop between model outputs, runtime feedback, validation, and correction, with a claimed cached-input cost advantage that would support tighter interaction/verification loops. In parallel, [Google’s Gemini Managed Agents guide](https://x.com/_philschmid/status/2059263980913229989) framed agent infra as a single API call to a managed harness with sandboxing, persistence, and mounts, while [LangChain’s updated `create_agent` docs](https://x.com/sydneyrunkle/status/2059280878694531280) and [dair.ai’s “harness” paper summary](https://x.com/dair_ai/status/2059294269698199929) formalized the same stack: **context governance, trustworthy memory, dynamic skill routing**.
- **Benchmarks are getting closer to real developer experience**: [DeepSWE](https://x.com/serenaa_ge/status/2059308218564890875), introduced as a new benchmark for agentic coding, got strong endorsement from practitioners; [@theo called it](https://x.com/theo/status/2059352130289651925) “the first code bench that actually aligns with how it feels to use these models coding.” It also created more separation at the top end than public SWE leaderboards often show. Related benchmark signals: [Qwen3.7 Max debuted at #4 on Code Arena: Frontend](https://x.com/arena/status/2059297720079393107), roughly on par with **Claude Opus 4.6** on agentic webdev tasks, and [Alibaba amplified the result](https://x.com/AlibabaGroup/status/2059317802935423028). Across the tooling stack, [Anthropic shipped a security-guidance plugin for Claude Code](https://x.com/ClaudeDevs/status/2059385239781384341) and reported a **30–40% reduction** in security-related PR comments in internal use, while [OpenAI highlighted GPT-5.5 in Codex at Databricks](https://x.com/OpenAIDevs/status/2059353117934899289) for more reliable document parsing.

**Research Agents, Long-Horizon Reasoning, and “Sleep” for Context Compression**



- **Math/science agents showed more evidence of capability overhang—conditional on the right harness**: The strongest cluster of tweets was around models tackling old open problems. A mathematician reported [Claude Mythos solving Erdős problem #90](https://x.com/__alpoge__/status/2059298565093196012), with follow-up detail that the model often converged to a **different, cleaner proof path** than OpenAI’s earlier route. This was echoed by [@_sholtodouglas](https://x.com/_sholtodouglas/status/2059303540150137244), [@kimmonismus](https://x.com/kimmonismus/status/2059311386820289013), and then sharpened by [Sébastien Bubeck](https://x.com/SebastienBubeck/status/2059343132991623186): with an **appropriate harness**, both **Mythos** and **GPT-5.5** can reproduce what an internal model had done one-shot, implying a large amount of latent capability not exposed by vanilla chat UX.
- **Long-horizon memory is resurfacing as a core bottleneck**: The paper [“Language Models Need Sleep”](https://x.com/iScienceLuvr/status/2059221770075562113) got notable attention. The mechanism is a **sleep-like consolidation phase** where recent context is converted into persistent fast weights before clearing the KV cache, moving compute into an offline pass while preserving wake-time latency. [dair.ai’s summary](https://x.com/dair_ai/status/2059333792775745619) emphasized the systems angle: this is an alternative to ever-growing KV caches for agents with long trajectories. This theme connected neatly with ongoing discussion about memory systems in agents, including [Omar’s pointer to Anthropic’s memory talk and Dream feature](https://x.com/omarsar0/status/2059285935376765214).
- **Open deep-research agents and science forecasting also advanced**: [QUEST](https://x.com/iScienceLuvr/status/2059223911011930606), a family of open **2B–35B** models for long-horizon fact-seeking, citation grounding, and report synthesis, was released as a general-purpose deep research agent. On the science-evals side, Sakana/Stanford/Oxford/AI2’s [CUSP benchmark](https://x.com/SakanaAILabs/status/2059166749761872342) found current models can often identify promising research directions but struggle much more with **whether** and **when** breakthroughs materialize.

**Model, Optimizer, and Architecture Updates**

- **Optimizer work remains lively, especially around Muon variants and schedule-free training**: [AMUSE](https://x.com/jueunkim_0525/status/2059127584601055426) proposes **Anytime MUon with Stable gradient Evaluation**, combining Muon with schedule-free-style gradient evaluation for stable anytime training without LR decay, reporting gains at **124M / 720M / 1B** scale and on ViT/ImageNet fine-tuning. Related implementation discussion came from [ClashLuke’s SFMuon snippet](https://x.com/Clashluke/status/2059187617997197553) and [kellerjordan’s Modded-NanoGPT result on Newton-Muon](https://x.com/kellerjordan0/status/2059353883881976044).
- **Sparse attention design space continues to diversify**: [MiniMax teased M3 as open source](https://x.com/MiniMax_AI/status/2059286515155599595), and follow-on technical commentary suggested a new **block-sparse two-stage attention** path. [@kimmonismus summarized the reported speedups](https://x.com/kimmonismus/status/2059302121489486335): **9.7× prefilling** and **15.6× decoding** at **1M tokens** versus M2. [@eliebakouch added](https://x.com/eliebakouch/status/2059321928205156568) that M3 appears to move back to **GQA-based** sparse attention with block selection on real KV, distinct from DeepSeek’s compressed-attention variants.
- **Vision/open model releases and ranking updates**: [PrismML released Bonsai Image 4B](https://x.com/PrismML/status/2059339157600969199), including **1-bit and ternary** variants intended to run locally on laptops and phones; a follow-up noted browser-local execution was possible at ~3GB footprint. On the closed side, [Microsoft’s MAI-Image-2.5](https://x.com/MicrosoftAI/status/2059344061358563838) debuted at **#3 on the Image Arena**, breaking a top-5 club previously dominated by OpenAI and Google, with [Arena reporting a 1,254 score](https://x.com/arena/status/2059346024632820146). Meanwhile, [Artificial Analysis measured Gemini 3.5 Flash](https://x.com/ArtificialAnlys/status/2059316050391634302) at up to **~280 output tok/s** with materially stronger agentic performance, but at **~5×** the cost of Gemini 3 Flash.

**Infra, Systems, and the Semiconductor Stack**



- **华为的“τ scaling”论文更多被视为一份工程路线图，而非一个新定律**：一段非常详细的推文讨论认为，[华为的《多层电子系统的时间缩放理论》（A Time Scaling Theory for Multi-Layer Electronic Systems）](https://x.com/ZhihuFrontier/status/2059118295580852374) 应被解读为一份**战略宣言/白皮书**。其核心提议是将**时间常数 τ** 而非工艺节点（node），视为衡量设备、芯片和数据中心规模的统一指标。最具体的声明涉及未来 Kirin 设计上的 **LogicFolding** 技术，包括在固定节点下实现 **+55% 的密度**、**+41% 的能效**以及 **+13% 的频率**提升，此外还包括 **Unified Bus** 和 **Hi-ONE optical I/O** 等封装/网络构想。该推文同时也谨慎地指出缺乏验证依据——如芯片照片、SEM（扫描电子显微镜）图、工作负载详情、收益率曲线等——并将那些最引人注目的数据解读为前景广阔但**未经证实**。随后的反应还强调，华为的路径可能更多地依赖于封装和架构，而非光刻技术的追赶，例如 [@josiah_leee 引用了 Jensen 的观点](https://x.com/josiah_leee/status/2059297861745963099)，即 Hopper 到 Blackwell 的大部分收益来自于非节点（non-node）优化。
- **数据中心功耗和推理供应限制正在成为首要关注点**：[SemiAnalysis 发布了关于 800VDC 转型的文章](https://x.com/SemiAnalysis_/status/2059253624249696658)，[John Carmack 也对此表示推荐](https://x.com/ID_AA_Carmack/status/2059382254191652896)，强调了从电动汽车（EV）功率电子设备向数据中心设计的技术跨界，包括高压 SiC 部件。另外，[Epoch AI 预测了可能出现的推理算力危机](https://x.com/EpochAIResearch/status/2059372951338909717)：需求增长似乎快于服务能力，尤其是对于长上下文（long-context）工作负载。他们的初步模型表明，虽然在理想假设下当前的全球 Blackwell 供应可以满足今日的需求，但吞吐量会随着上下文变长而急剧下降，且需求的增长可能已经超过了供应。

**生产工具与开发者基础设施**

- **推理/服务栈获得了显著的性能和可观测性更新**：[vLLM 合并了一个 Rust 前端](https://x.com/vllm_project/status/2059344804295942513)，作为 Python API 服务器的无缝替代方案，初步数据显示，在单进程中处理重预处理（preprocess-heavy）任务时，性能从 **~162 req/s 提升至 ~837 req/s**。[W&B 发布了一个 MCP server](https://x.com/wandb/status/2059384552725025226)，允许 coding Agent 检查实验和训练运行，其采用了 Schema 优先的重新设计，旨在避免上下文窗口爆炸（context-window blowups）。[Unsloth 增加了在其本地 UI 中运行 GPT、Claude 和其他 API 的支持](https://x.com/UnslothAI/status/2059277719633101291)，包括 Prompt 缓存和代码执行功能。
- **Cloudflare、OpenRouter 和向量/检索厂商推动了“生产化”层级**：[OpenRouter 宣布获得 1.13 亿美元 B 轮融资](https://x.com/OpenRouter/status/2059277623629664758)，并表示周交易量在六个月内从 **5T 增长到 25T tokens**。[Cloudflare 重启了其初创企业计划](https://x.com/kristianfreeman/status/2059188629780545973)，提供高达 **35 万美元**的额度；同时，围绕 **Think** 和 Agent 易用性的讨论强调了持久化轮次（durable turns）、重连、过期状态处理和恢复是关键的实际差异化因素。在检索基础设施方面，[Booking.com 讨论了扩展至 1 亿以上 embeddings 的案例](https://x.com/weaviate_io/status/2059227285639581729)，包括过滤向量搜索、读写并发以及针对合作伙伴消息 Agent 的 human-in-the-loop 评估。

**热门推文（按参与度排序）**

- **实践中的 Codex / agentic coding**：信号最强的产品使用推文是 [@bunkaich 展示了 Codex 如何帮助逆向工程并修复一款廉价 MP3 播放器的固件](https://x.com/bunkaich/status/2059178996126900703)，工作流涵盖了芯片检查、OS 提取、二进制分析以及刷入修改后的镜像。
- **DeepSWE 基准测试发布**：[@serenaa_ge 的 DeepSWE 发布公告](https://x.com/serenaa_ge/status/2059308218564890875) 成为关于“这是否符合真实编程体验？”讨论的主要参考点。
- **Claude Code 安全插件**：[@ClaudeDevs 的发布](https://x.com/ClaudeDevs/status/2059385239781384341) 脱颖而出，因为它将具体的产品发布与内部指标相结合：安全相关的 PR 评论减少了 **30–40%**。
- **OpenRouter 融资 + 生产环境 token 增长**：[@OpenRouter 的 1.13 亿美元 B 轮融资](https://x.com/OpenRouter/status/2059277623629664758) 是目前最明确的市场信号之一，表明路由和多模型基础设施现在被视为持久的平台层。
- **vLLM Rust 前端**：对于任何在高吞吐服务中遇到 CPU/API 服务器瓶颈的人来说，[@vllm_project 的合并公告](https://x.com/vllm_project/status/2059344804295942513) 都至关重要。

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. Qwen 3.7 发布与 Qwen 3.6 本地性能

  - **[等待 Qwen 3.7 open weight... 新王已立...](https://www.reddit.com/r/LocalLLaMA/comments/1tjvz6l/waiting_for_qwen_37_open_weight_the_new_king_has/)** (热度: 1217): **[这张图片](https://i.redd.it/j8qkty82qj2h1.png) 是来自 [Qwen3.7 博客](https://qwen.ai/blog?id=qwen3.7) 的 Benchmark/营销对比图，将 **Qwen3.7-Max** 定位为在 Agentic Coding、软件工程、MCP/Tool-use、推理和知识评估方面领先的前沿模型，对比对象包括 **Qwen3.6-Plus**、**DS-V4-Pro Max**、**GLM-5.1**、**Kimi K2.6** 以及 **Claude Opus-4.6 Max**。其技术意义在于，该幻灯片将 Qwen3.7-Max 描绘为在许多 Benchmark 上与 Claude 级别的模型极具竞争力甚至处于领先地位，尽管 **Claude Opus-4.6 Max** 在 `ClawEval` 和 `CoWorkBench` 等某些任务上似乎仍然保持领先。评论者指出，这是 **Max** 模型，并不一定代表更小尺寸或 open-weight 发布的版本，并推测可能会推出适用于 **Strix Halo** 等本地硬件的 `3.7-122B-A17B` `MXFP4` 模型，具备 `512k` context。主要的争论点在于对 open weight 的怀疑：评论者指出 **Qwen 历史上从未开源过 Max 系列的权重**，因此标题中“等待 open weight”的说法可能并不现实。其他人则提醒不要指望假设的 `27B` 模型能达到图示的 Max 级 Benchmark 结果。**

    - 几位评论者将 **Qwen Max** 与可能的 open-weight 发布版本区分开来，指出 *“Qwen 从未开源过 Max 系列”*，并警告不要期望较小的 `27B` 变体能匹配 Max 级别的 Benchmark 性能。隐含的技术结论是，任何公开/open-weight 的 Qwen 3.7 版本可能在架构/规模上都与 Benchmark 中的旗舰模型不同。
    - 一个技术愿望清单集中在假设的 **Qwen 3.7 `122B-A17B` MTP MXFP4** 模型上，评论者认为该模型带有 `512k` context，非常适合 **Strix Halo** 级别的本地硬件。另一位用户提到了 **Qwen 3.5 `397B-A17B` NVFP4**，声称它可以在拥有足够显存余量的 `4x RTX 6000 Pro` GPU 上运行约 `10` 个并发的 `200k`-token 会话，如果 Qwen 3.7 能达到报道的 Benchmark 水平，它将被定位为潜在的“家用版 Opus”。
    - 一位评论者认为，前沿模型的 open-weight 发布可能性可能较低，因为高性能的本地模型会削弱供应商的变现能力。他们声称 Qwen 的策略已从颠覆转向商业化的前沿竞争，这可能会影响像 `397B-A17B` 这样的大型 MoE 模型是否会 open-weight 发布。

  - **[Qwen3.6 35Ba3 改变了我的工作流，甚至改变了我使用电脑的方式](https://www.reddit.com/r/LocalLLaMA/comments/1tjwrp7/qwen36_35ba3_has_changed_my_workflows_and_even/)** (热度: 567): **该贴描述了一个使用 **Qwen3.6 35B a3** 通过 `pi` 实现的 local-agent 工作流，用户将可重复的流程转换为由 Codex 生成/记录的“技能”，然后将其重用于 VPS DevOps、`docling` PDF→EPUB 转换、Playwright 测试、代码工单和 OS 级 shell 任务。一个具体的例子：WhatsApp 语音 → 在 AnythingLLM 中转录 → `content.md` → 本地生成的落地页，然后是由“manager” `pi` 进程执行的 `plan.md` 工单队列，该进程会启动具有 fresh-context 的子 Agent，执行 `pi -p @plan.md "Check the first Ticket with Status UNDONE and do it"`，将工单标记为 `DONE`，通过 git 提交，最后通过 VPS 技能进行部署。评论者关注的是操作层面的问题：什么样的硬件可以运行这种设置、Agent 在拥有 OS 访问权限时是否经过沙盒处理/是否可信，以及与 Hermes 等其他 agentic 工具相比，`pi` 的采用难度如何。**

    - 一位用户报告在搭载 **24GB RTX Pro 4000 Blackwell SFF GPU** 的 **MS-02** 上，通过 **Unsloth Studio** 运行 `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`，性能稳定在 **`>100 tokens/s`**。他们将其性能与 **Mac Studio M2** 上“未优化的 GGUF”进行了对比，将 MS-02 作为 Mac 工作站的小型远程 GPU 服务器，并指出 **Unsloth 未来对 MLX 的支持** 可能会提升 Mac 端的性能。截图：[preview.redd.it](https://preview.redd.it/exwng3d4ik2h1.png?width=3966&format=png&auto=webp&s=03bf5de53b529f1b26f669c21834d9f1d69d16e0)。

- **[在 Qwen3.6 35B A3B 和 ik_llama.cpp 上利用 12GB VRAM 实现 110 tok/s](https://www.reddit.com/r/LocalLLaMA/comments/1tjh7az/110_toks_with_12gb_vram_on_qwen36_35b_a3b_and_ik/)** (热度: 565): **该帖子在 RTX 4070 Super 12GB + Ryzen 7 9700X 环境下，使用 byteshape 的 [`IQ4_XS` `4.19 bpw` GGUF](https://huggingface.co/byteshape/Qwen3.6-35B-A3B-MTP-GGUF) 对 **Qwen3.6-35B-A3B MTP** 进行了基准测试。** 该测试对比了上游 [`llama.cpp`](https://github.com/ggml-org/llama.cpp) 与 [`ik_llama.cpp`](https://github.com/ikawrakow/ik_llama.cpp)，参数设置为 `--ctx-size 131072`、`q8_0` KV cache、MTP draft max `3` 以及 `p_min=0.75`。在相同的 [`mtp-bench.py`](https://gist.github.com/am17an/228edfb84ed082aa88e3865d6fa27090/) 工作负载下，上游 `llama.cpp` 平均速度为 **`89.76 tok/s`**，综合 MTP 接受率为 **`0.9393`**；而 `ik_llama.cpp` 在 `16.64s` 内平均达到 **`110.24 tok/s`**，声称有 **`23%` 的吞吐量提升**，尽管在更新后的结果中其综合接受率较低（**`0.8749`**）。作者将实际的适配成功归功于 `ik_llama.cpp` 上的 `--fit`/`--fit-margin 1664`，并指出可通过将 `--fit-margin` 提高到 `1792` 或 `2048` 来缓解 OOM，同时提到在 iGPU 上运行显示输出可以为推理释放几乎全部 `12GB` 的 VRAM。评论者关注可复现性：他们要求提供完整的上游 `llama.cpp` 命令，并指出最近合并了多个与 MTP 相关的 PR，因此基准测试结果可能很大程度上取决于构建日期。针对单 GPU CachyOS/KDE 用户提出的一个技术权宜之计是：使用 `LIBGL_ALWAYS_SOFTWARE=1` 和 `GALLIUM_DRIVER=llvmpipe` 运行软件渲染的 Plasma Wayland 会话，这能将空闲 VRAM 从约 `>1024MB` 降低到 `126MB`，代价是合成器特效变慢或被禁用。

    - 一位 CachyOS/KDE Wayland 用户描述了一种针对单 GPU 系统的 VRAM 节省变通方法：创建一个自定义 SDDM 会话，通过 `LIBGL_ALWAYS_SOFTWARE=1`、`GALLIUM_DRIVER=llvmpipe` 和 `KWIN_COMPOSE=Q` 强制 KDE Plasma 通过 CPU 渲染。据报告，KDE Wayland 的空闲 VRAM 从 **> `1024 MB`** 降至 **~`126 MB`**，从而为运行 35B 模型释放了近 1GB 的 VRAM，代价是禁用或极其缓慢的合成器动画。
    - 几位评论者关注报告的 `110 tok/s` 是否源于 **ik_llama.cpp** 具有比上游 `llama.cpp` 更好的 MTP/Speculative Decoding 行为。有人指出，据报道 ik_llama.cpp 的接受率**从未低于 `0.790`**，而 llama.cpp 曾跌至 **`0.477`**，并要求提供确切的 llama.cpp 命令/设置，同时指出过去 24 小时内 llama.cpp 合并了多个与 MTP 相关的 PR。
    - 一位评论者询问了用于 **Qwen3.6 35B A3B** 的 `IQ4_XS` 量化，指出它似乎是内存占用最低的 Q4 量化方案，并要求提供有关模型质量/智能影响以及最终 VRAM/RAM 分配的详细信息。这突显了 12GB VRAM 运行的关键权衡：是通过激进的量化来适配模型，还是保持推理质量并避免过度的 CPU/RAM offload 瓶颈。

### 2. 开源 AI 资金与法律压力

  - **[Heretic 已收到 Meta, Inc. 的法律通知](https://www.reddit.com/r/LocalLLaMA/comments/1tjmvx6/heretic_has_been_served_a_legal_notice_by_meta_inc/)** (热度: 2705): **Heretic Free Software Project** 表示，它收到了代表 **Meta Platforms, Inc.** 的供应商发出的电子邮件法律通知，并已从 Heretic 控制的仓库中移除了 Meta **Llama** 模型权重的衍生品。该项目还宣布了一个官方的德国托管 [Codeberg 镜像](https://codeberg.org/p-e-w/heretic)，并表示正在开发“技术措施”，以便在不依赖单一托管供应商的情况下保留对 Heretic 创建模型的访问；该帖讽刺地称 Llama 位列“前 200 名最佳”模型，在 [LM Arena](https://lmarena.ai/) 排行榜上“仅落后于 `168` 个其他模型”。热门评论集中在该帖子的讽刺语气上，尤其是“`168` 个其他模型”的排行榜梗，并批评了 Meta 的执行行为，因为有指控称 Meta 在模型训练中使用了种子下载的书籍或受版权保护的材料。

    - 一位评论者强调了法律回应的措辞，将 **Meta 的 Llama 系列**置于当前的开源/模型竞争背景下：它被描述为在 **LM Arena** 上排名在前 `200` 名以内，但落后于来自 `23` 个竞争对手的 `168` 个模型。引发的技术启示是，Meta 的命名执行姿态正与其 Llama 的相对基准地位以及近期模型发布速度放缓形成对比。

  - **[DeepSeek 正在推进 102.9 亿美元融资轮，梁文锋承诺将继续开发开源 AI 模型，而非追求短期商业化目标](https://www.reddit.com/r/LocalLLaMA/comments/1tkfvvj/deepseek_is_pushing_forward_with_1029_billion/)** (热度: 797): 据 [Bloomberg](https://www.bloomberg.com/news/articles/2026-05-22/deepseek-founder-declares-agi-goal-as-10-billion-round-advances) 报道，**DeepSeek** 正在推进一轮 **`102.9 亿美元` 的融资**，创始人**梁文锋**重申了以 **AGI 为导向的路线图**，并承诺继续发布/开源 AI 模型，而不是优先考虑近期商业化。评论者认为这是一种战略押注，即模型优势的半衰期很短，开源研究比封闭的人才/模型护城河能更快地加速迭代。热门评论认为，本地推理用户只是极少数，因此发布权重不会实质性损害 **OpenAI**、**Anthropic**、**Google** 或 **Mistral** 等实验室的 SaaS/API 收入；任何架构上的领先优势估计只有约 `1 年` 左右的保质期。另一位评论者表示，开源模型在代码辅助方面已经“足够好”，达到了 **GLM 5.1** 级别的能力，下一个前沿是将类似的能力压缩到更小、更快、更高效的模型中。

    - 评论者认为，模型权重的技术/商业保质期很短：架构优势可能仅维持约 `1 年`，而本地推理用户与托管 API 用户相比只是极少数。其观点是 **OpenAI**、**Anthropic**、**Google**、**Mistral** 等公司可以发布权重而不会实质性损害收入，因为大多数用户缺乏硬件或兴趣在本地运行即使是 `9B` 规模的模型。
    - 一条技术讨论线索将当前的开源模型定性为在代码辅助方面已达到“足够好”的能力，并将 **GLM 5.1** 视为一个门槛模型。根据该评论，剩下的优先级不是原始智能，而是蒸馏/压缩：在更小、更快、更高效的可部署模型中保留这种编码能力。
    - 一位评论者指出了 DeepSeek 自己的报告，称他们正在努力增加多模态能力：[DeepSeek_V4.pdf](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)。值得注意的技术角度是，尽管面临 GPU/出口制裁限制，DeepSeek 仍继续进行模型扩张，这表明在有限的硬件访问下仍取得了持续进展。

## 技术性较弱的 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude Code 工作流与 Anthropic Agent 培训

  - **[Claude Code 发布了 /workflows](https://www.reddit.com/r/ClaudeCode/comments/1tkjy4u/claude_code_dropped_workflows/)** (活跃度: 1074): **图中是 Claude 品牌的官方公告图，展示了 Claude Code 中的 `/workflows` 功能。该帖子声称 Anthropic 在 `Claude Code 2.1.147` 版本中短暂公开了一个新的工作流系统，随后又将其从更新日志中移除。据称其技术意义在于用 `workflow.js` 代码驱动的控制器取代了基于 LLM 的编排器：包括结构化阶段、并行扇出 (fan-out)、条件/循环/预算限制、重试机制、后台执行，并通过在各阶段之间传递子 Agent (sub-agent) 的输出而非通过主对话上下文来减少上下文窗口的 “Token 税”。图片链接：[https://i.redd.it/6tuq1a2i3p2h1.png](https://i.redd.it/6tuq1a2i3p2h1.png)。** 评论者对这是否是一种根本性的新型多 Agent 模式表示怀疑，并指出了现有的 Claude Code [agent teams](https://code.claude.com/docs/en/agent-teams) 文档。另一些人则认为与对 “Opus 4.5” 等更强模型的需求相比，这是一个低优先级的特性。

    - 一位评论者链接了 **Anthropic 现有的 Claude Code “agent teams” 文档** (https://code.claude.com/docs/en/agent-teams)，指出所描述的 `/workflows` 模式——*“一个主 Agent (LLM) 决定派生哪些子 Agent，保存每个中间结果并规划下一步”*——与已经记录的多 Agent 编排概念重合。
    - 据报道，`/workflows` 功能似乎是昙花一现：一位评论者表示该功能早些时候出现在更新日志中，但 **Anthropic 随后将其撤下**，并提供了一张已删除更新日志条目的截图镜像 (https://preview.redd.it/720w663mcp2h1.png?width=2056&format=png&auto=webp&s=d7afca73806dd159eff3141db0f61de5a37526a8)。
    - 一位用户将该功能与他们围绕 **skills + YAML + JavaScript CLI** 构建的自定义编排堆栈进行了比较，暗示 `/workflows` 可能会将开发者为了实现可重复的 Claude Code 任务流水线而手动实施的模式正式化。

  - **[Anthropic 正式推出 13+ 门带证书的免费 AI 课程（包括 Agentic AI 和 Claude Code！）](https://www.reddit.com/r/ClaudeAI/comments/1tjpfh8/anthropic_officially_launched_13_free_ai_courses/)** (活跃度: 2547): **Anthropic** 正在通过其基于 Skilljar 的学院提供免费的官方培训目录，可通过 [Anthropic Learn](https://www.anthropic.com/learn) 访问，课程涵盖 **Claude**、**Claude Code**、**Claude API**、**MCP / agentic workflows** 以及 **Amazon Bedrock** 和 **Google Cloud Vertex AI** 的部署路径，并提供证书。在技术上值得关注的内容包括 MCP 材料，涵盖了围绕 `STDIO` 和 `StreamableHTTP` 传输的高级主题，以及用于代码库编辑、测试执行和 “Plan Mode” 的 Claude Code 模块。此外还提到了一个独立的免费 [CodeSignal](https://codesignal.com/) 课程路径 “Developing Claude Agents”，提供交互式 Python/TypeScript 实验和证书。评论者确认 Skilljar 课程是真实的，因为它们链接自 Anthropic 的官方网站；一位完成了 `10/15` 门课程的用户特别推荐 MCP 和高级 MCP 模块，称其 *“非常值得学习”*。

    - 多位评论者确认 Skilljar 课程是正宗的 **Anthropic** 培训材料，指出课程门户链接自 [anthropic.com/learn](https://www.anthropic.com/learn)，而非第三方诈骗或转载。
    - 一位完成了 `10/15` 门课程的用户特别强调了 **MCP** 和 **MCP Advanced Topics** 模块的价值，认为其对 Model Context Protocol 集成的 `STDIO` 和 `StreamableHTTP` 传输协议进行了实用的讲解。
    - 少数用户指出该目录并非新推出，已经上线数月；一位完成了两门课程的评论者称其 *“相当基础”*，暗示这些材料对于经验丰富的 AI 开发者来说可能更偏向于入门级。


### 2. Z-Image 6B, Gemini 3.5 Flash 与 OpenAI Math 更新

  - **[Tencent released Z-Image 6B with pixel space gen. No VAE &amp; 1k Resolution.](https://www.reddit.com/r/StableDiffusion/comments/1tkipk6/tencent_released_zimage_6b_with_pixel_space_gen/)** (Activity: 899): **The [image](https://i.redd.it/69r8ttxmvo2h1.jpeg) is a sample collage for **Tencent/Z-Image 6B / L2P**, illustrating `1024px`-class **pixel-space image generation** across portraits, animals, fantasy scenes, vehicles, and stylized compositions, with the key technical claim being generation **without a VAE**. The post links the project page at [nju-pcalab.github.io/projects/L2P](https://nju-pcalab.github.io/projects/L2P/) and a commenter points to model files on Hugging Face: [zhen-nan/L2P](https://huggingface.co/zhen-nan/L2P/tree/main).** Commenters mainly focused on the architectural trend — *“Everyone going for No-VAE now huh”* — and questioned practical quality with *“Is it any good?”* rather than providing benchmarks or detailed evaluations.

    - A commenter points to the model files on Hugging Face: **zhen-nan/L2P** at [https://huggingface.co/zhen-nan/L2P/tree/main](https://huggingface.co/zhen-nan/L2P/tree/main), relevant for readers wanting to inspect/download Tencent’s **Z-Image 6B** release and its claimed **pixel-space generation / no-VAE** setup.
    - Several comments highlight the broader technical trend toward **No-VAE / pixel-space image generation**, with one user noting *“Everyone going for No-VAE now huh”*. This is notable because avoiding a VAE changes the compression/latent bottleneck tradeoff and may affect reconstruction fidelity, memory cost, and native high-resolution generation such as the post’s claimed `1k` resolution.
    - One commenter raises a comparison to **Lodestone**, asking whether Tencent’s approach learned from Lodestone’s no/low-latent direction or whether Lodestone could learn from Z-Image. The thread does not provide benchmark data, but the technical comparison suggests interest in converging open-weight architectures for direct pixel-space diffusion/flow generation.

  - **[Google's latest creation: Gemini 3.5 Flash vs all](https://www.reddit.com/r/singularity/comments/1tjoarz/googles_latest_creation_gemini_35_flash_vs_all/)** (Activity: 1503): **The post reports a simple arithmetic failure in **Google Gemini 3.5 Flash** via the Gemini app: for the prompt `300+140=460` / “Is this correct? Breakdown?”, the shared Gemini run allegedly accepts the incorrect sum, while comparison runs were linked for [Claude](https://claude.ai/share/8383747a-aaf1-4f6c-a516-0e839f46a698), [Grok](https://grok.com/share/bGVnYWN5_3c63e371-eb9d-46c3-8ba2-0c745c6795a2), and [ChatGPT](https://chatgpt.com/share/6a0f1e13-a0c8-8328-b989-1ac51b92e81c). Commenters reproduced the issue and attributed it to Gemini app inference settings: **“Standard”/default thinking behaves like minimum or no reasoning**, while **Extended thinking** or AI Studio with higher thinking settings reportedly returns the correct `300 + 140 = 440`.** The main debate is that this is less evidence about the base model’s capability and more about product-level serving configuration: commenters argue the **Gemini app is “nerfed”** relative to AI Studio, especially under default/minimum thinking settings. The OP frames the result as embarrassing given claimed SOTA/finance-agent rankings, while others suggest benchmark performance may not reflect low-effort app defaults.

    - Users reported that the apparent failure depends heavily on Gemini’s **thinking level**: switching to **Extended thinking** fixes the answer, while **Standard** was characterized as effectively *“doesn’t think at all.”* Another commenter reproduced the same output via a screenshot ([preview image](https://preview.redd.it/whzg30z8hi2h1.png?width=1557&format=png&auto=webp&s=192481783e75626c47648f50954c4c8fe8fb60a7)) and claimed the Gemini app defaults to something like **minimum thinking**, whereas **AI Studio** with even **Low** thinking avoids the mistake.
    - A technical comparison was raised around **tool-calling behavior**: one commenter argued Gemini’s weakness is not necessarily raw reasoning but **tool-routing logic**, noting that ChatGPT would likely delegate the task to **Python** rather than solve it purely in-model. This implies benchmark results may depend on whether the model is allowed to invoke tools and how reliably it decides to use them.



- **[数学系研究生朋友说我们要完蛋了](https://www.reddit.com/r/OpenAI/comments/1tkcxxi/math_grad_student_friend_says_were_cooked/)** (Activity: 825): **该[图片](https://i.redd.it/l7gd5lx9in2h1.png)是一张 **推文截图**，转述了一名数学系研究生对最近声称的 **Erdős 证明** 的惊恐反应，帖子标题为 *“数学系研究生朋友说我们要完蛋了。”* 该内容**并未提供证明的技术细节**、定理陈述、模型、Benchmark 或验证过程；其意义在于背景和社交层面：一位数学家将该结果描述为此前“完全无法触及”，并称 OpenAI 的公告“极其俗气且品味低劣”。** 评论区的讨论大多是非技术的、梗驱动的，转向了关于“极客版 OnlyFans”的笑话。一位评论者询问“极其俗气且品味低劣”是什么意思，但没有关于数学或 AI 能力声明的实质性辩论。

    - 一位评论者认为，随着 AI 系统开始在 **数学、定理证明和研究级推理 (research-level reasoning)** 方面展现出能力，人们曾认为“创意和智力型”工作是安全的看法已经动摇。技术上的启示是，自动化风险可能与任务是否具有重复性并不完全相关；相反，高级推理 Benchmark 和形式化证明系统 (formal proof systems) 在评估 AI 影响方面正变得越来越重要。

# AI Discords

遗憾的是，Discord 今天关闭了我们的访问权限。我们不会再以这种形式恢复，但我们很快就会发布全新的 AINews。感谢读到这里，这是一段美好的历程。