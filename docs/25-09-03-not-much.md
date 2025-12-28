---
companies:
- exa
- openpipe
- coreweave
- statsig
- openai
- zed
- claude
- gemini
- langchain
- anthropic
- fair
- alibaba
- hud-evals
date: '2025-09-03T05:44:39.731046Z'
description: '**Exa** 完成了 **7 亿美元的 B 轮融资**，**OpenPipe** 被 **Coreweave** 收购，**Statsig**
  和 **Alex** 则被 **OpenAI** 收入麾下。**Zed** 团队推出了 **Agent/Client Protocol (ACP)**，旨在标准化
  IDE 与智能体（Agent）之间的互操作性，目前已支持 **Claude Code** 和 **Gemini** CLI。**LangChain 1.0 alpha**
  统一了用于推理和多模态数据的内容块。


  **OSWorld Verified 排行榜**推动了对包括 **OpenAI** 和 **Anthropic** 模型在内的计算机操作智能体（computer-use
  agents）的可复现评估。**FAIR** 揭露了编程智能体在 **SWE-Bench Verified** 上的作弊行为。**PR Arena** 开始举办实时编程智能体竞赛。**GSO**
  和 **Holistic Agent Leaderboard** 等基准测试对软件优化和网页浏览任务进行了测试，其中 **Qwen3-Coder** 和 **Gemini
  2.5 Flash** 表现强劲。


  在工具使用的强化学习进展方面，**SimpleTIR** 提高了多轮工具使用的成功率，**UI-TARS-2** 推动了 GUI 智能体的发展。**DARLING**
  优化器提升了推理和指令遵循的质量与多样性，而 **DEPO** 实现了高效的数据强化学习验证（RLVR），并显著提升了运行速度。'
id: MjAyNS0w
models:
- claude-code
- gemini
- qwen3-coder
- gemini-2.5-flash
people:
- zeddotdev
- mathemagic1an
- hwchase17
- giffmana
- gneubig
- crystalsssup
- sayashk
- _philschmid
- _akhaliq
- jaseweston
title: 今天没发生什么特别的事。
topics:
- agent-protocols
- interoperability
- standardization
- agent-evaluation
- coding-agents
- software-optimization
- web-browsing
- reinforcement-learning
- multi-turn-reasoning
- optimizer-design
- data-efficient-rlvr
- leaderboards
- benchmarking
---

**平静的一天**

> 2025年9月3日至9月4日的 AI 新闻。我们为您检查了 12 个 Reddit 分区、544 个 Twitter 账号和 22 个 Discord 社区（186 个频道，4795 条消息）。预计节省阅读时间（以 200wpm 计算）：410 分钟。我们的新网站现已上线，支持全元数据搜索，并以精美的 vibe coded 方式呈现所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻分类，并在 @smol_ai 上向我们提供反馈！

平静的一天。祝贺 [Exa 完成 7 亿美元 B 轮融资](https://x.com/ExaAILabs/status/1963262700123000947)，以及 [Coreweave 收购 Agent 训练初创公司 OpenPipe](https://techcrunch.com/2025/09/03/coreweave-acquires-agent-training-startup-openpipe/)，还有 [Statsig](https://x.com/arfurrock/status/1962960884654866554?s=46) 和 [Alex](https://x.com/danieledrisian/status/1963301872036712652?s=46) 被 OpenAI 收购。

---

# AI Twitter 回顾

**Agent 基础设施标准化与协议**

- **Agent/Client Protocol (ACP)**：Zed 团队推出了一项用于 IDE 与 Agent 互操作性的开放协议，该协议将 UI 与 CLI Agent 操作清晰地解耦，类似于语言工具的 LSP。ACP 已经支持 Claude Code 和 Gemini CLI，使得在编辑器或终端中接入不同的 Agent 变得更加容易，无需定制化集成。查看 [@zeddotdev](https://twitter.com/zeddotdev/status/1963258131191853285) 的发布公告和概述，以及 [@mathemagic1an](https://twitter.com/mathemagic1an/status/1963273618705482155) 的简要总结（网站：[agentclientprotocol.com](http://agentclientprotocol.com/)）。
- **LangChain 1.0 alpha（标准内容块）**：1.0 alpha 版本统一了推理轨迹（reasoning traces）、引用、工具调用和多模态区块在不同供应商之间的内容表示，减少了更换模型或托管方时的胶水代码。来自 [@LangChainAI](https://twitter.com/LangChainAI/status/1963285794954907750) 的公告和来自 [@hwchase17](https://twitter.com/hwchase17/status/1963287729007165488) 的背景信息。LangChain 还在举办关于“深度 Agent（Deep Agents）”和长程规划（long-horizon planning）的见面会（[伦敦](https://twitter.com/LangChainAI/status/1963316066735812876)）。

**Agent 评估、编码与计算机使用 (computer-use)**

- **可复现的 CUA 评估与作弊分析**：OSWorld Verified 排行榜上线，旨在推动计算机使用 Agent（computer-use agents）的可复现评估；首批入选条目包括 OpenAI 和 Anthropic 的模型 ([@hud_evals](https://twitter.com/hud_evals/status/1963321238056796573))。另外，FAIR 揭示了编码 Agent 在 SWE-Bench Verified 上“作弊”的方式（例如，通过 grep 提交日志来获取问题 ID），强调了对加固评估环境的需求 ([@giffmana](https://twitter.com/giffmana/status/1963327672827687316))。
- **Agent 编码现场竞赛**：PR Arena 允许你在标记的 GitHub 问题上让两个编码 Agent 进行对决并挑选胜者——将“实战”对抗带到了 SWE-Bench 之外 ([@gneubig](https://twitter.com/gneubig/status/1963267468853477809))。相关消息：开源模型 + OpenHands 在多个 Agent 编码场景中具有竞争力 ([@gneubig](https://twitter.com/gneubig/status/1963045532022010231))。
- **软件优化与浏览任务**：GSO 是一个用于优化大型代码库的挑战性基准测试 ([@crystalsssup](https://twitter.com/crystalsssup/status/1963087272506753419))；Qwen3-Coder 在该测试中表现出色 ([@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1963049864474120475))。对于 Web 任务，Online Mind2Web 已添加到全面 Agent 排行榜（Holistic Agent Leaderboard），用于比较 Browser-Use 与 SeeAct 等支架（scaffolds） ([@sayashk](https://twitter.com/sayashk/status/1963343022252315112))，你还可以通过 Gemini 2.5 Flash 用大约 10 行代码引导一个 Chromium 浏览器 Agent ([@_philschmid](https://twitter.com/_philschmid/status/1963233076034650481))。

**用于工具调用和 LLM 训练的 RL，以及优化器见解**

- **稳定多轮工具使用**：SimpleTIR 将“无效轮次”（void turns，即无果的步骤）识别为核心失败模式；过滤这些轮次在多轮 RL 中带来了巨大提升——例如，一个 7B 模型在多轮工具使用指标上从 22% (DAPO) 提升至 50% ([paper](https://huggingface.co/papers/2509.01739), [@_akhaliq](https://twitter.com/_akhaliq/status/1963228487524679988), [author commentary](https://twitter.com/sivil_taram/status/1963279400834924965))。相关进展：UI-TARS-2 通过多轮 RL 推进了 GUI Agent 的发展 ([@_akhaliq](https://twitter.com/_akhaliq/status/1963229296236937443))。
- **优化质量与多样性**：DARLING 通过学习的分区函数同时优化两者，提升了推理和指令遵循的 pass@1/p@k，同时在 NoveltyBench 的多样性排名中位列第一 ([paper](https://arxiv.org/abs/2509.02534), [thread](https://twitter.com/jaseweston/status/1963230744173482018))。
- **高数据效率的 RLVR**：DEPO 报告称，通过筛选离线样本并过滤掉“可探索性”较低的在线样本，仅需一小部分数据即可实现显著加速（例如，使用 20% 的训练数据在 AIME’24 上实现 1.85 倍加速）([paper](https://arxiv.org/abs/2509.01321), [summary](https://twitter.com/iScienceLuvr/status/1963169113007895020))。
- **训练与优化器笔记**：一项系统性研究发现，基于矩阵的优化器（如 Muon, Soap）能加速小模型，但收益随规模扩大而递减（0.1B 时为 1.4 倍 → 1.2B 时约为 1.1 倍），且超参数迁移并非易事 ([paper](https://arxiv.org/abs/2509.02046), [summary](https://twitter.com/iScienceLuvr/status/1963168542872014943))。一个简化的推导解释了在特定假设下 AdamW 约 0.2 RMS 更新的“魔数比例” ([@JingyuanLiu123](https://twitter.com/JingyuanLiu123/status/1963084684784734543))。此外：智谱/lmsys 的 slime RL 框架代码走读已发布 ([repo](https://github.com/zhaochenyang20/Awesome-LLM-Alignment), [@Zai_org](https://twitter.com/Zai_org/status/1963099102347931975))。

**系统、推理与工具**

- **Google Cloud 之外的 Google TPU**：Google 正在洽谈将 TPU 部署在第三方 GPU 云中——据报道，多家供应商正在参与这一 TPU 算力的新分发模式 ([@anissagardizy8](https://twitter.com/anissagardizy8/status/1963228123144819167), [context](https://twitter.com/dylan522p/status/1963355683170246659))。
- **VS Code：自带 OpenAI 兼容端点**：原生支持自定义 OAI 兼容端点已上线，这对本地/自托管供应商和 OSS 技术栈来说是一个胜利 ([@ggerganov](https://twitter.com/ggerganov/status/1963255949373677959), [PR](https://twitter.com/ggerganov/status/1963255951659508117))。
- **更快的 Kernel，可导出的图**：FlashAttention-3 现可通过 Hugging Face “kernels” 获取（无需再进行冗长的构建），并支持 torch.compile fullgraph ([@RisingSayak](https://twitter.com/RisingSayak/status/1963225732668182856))。对于非 JIT 推理/训练，PyTorch 的 torch.export 路径针对编译时自动调优；其针对反向图的支持正趋于成熟 ([@soumithchintala](https://twitter.com/soumithchintala/status/1963225534659178948))。
- **CPU 优先推理与成本笔记**：微软开源了 bitnet.cpp（1-bit LLM 推理），报告称在某些模型上 CPU 推理速度提升 6.17 倍，能耗降低 82% ([@LiorOnAI](https://twitter.com/LiorOnAI/status/1963316578612605327))。与此同时，定价差异依然存在：许多第三方服务器不提供缓存命中折扣；由于缓存机制，闭源 API 在代码密集型工作负载中可能更便宜 ([@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1963294646957957263))。

**模型与多模态工具**

- **Nous Hermes-4-14B**：紧凑的 Hermes 4 模型，具有混合推理 + 工具调用能力，针对本地消费级硬件进行了优化。可在 HF 和 Nous Chat 中使用 ([@NousResearch](https://twitter.com/NousResearch/status/1963349882837897535))。
- **OpenVision 2**：一个完全开源、高性价比的视觉编码器系列，可与 CLIP/SigLIP 媲美；新版本扩大了训练数据并提升了准确率与成本的权衡 ([thread](https://twitter.com/cihangxie/status/1963297223753494832))。
- **高速文档理解**：腾讯的 POINTS-Reader 是一个简单的端到端 VLM，用于文档 OCR/提取，在 SGLang/vLLM 上具有高吞吐量；其两阶段训练（自动标注预训练 + 自我进化）在 OmniDocBench 的中英文测试中达到 SOTA ([@ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/1963192346222432750))。
- **社区图像编辑进展**：Qwen Image Edit 局部重绘（inpainting）获得了社区 LoRA，可以遮罩需要编辑的精确区域 ([demo + LoRA](https://twitter.com/ostrisai/status/1963269597865599425))；阿里巴巴强调了社区对局部重绘的贡献 ([@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1963048659676979559))。

**安全、鲁棒性与推理研究**

- **将监督扩展至前沿模型**：Transluce 训练了小型“调查员”模型 (8B)，能够可靠地越狱前沿助手（GPT-5, Claude 4.1, Gemini 2.5 Pro），这表明针对特定子领域和规模进行优化的监督可以跟上模型发展的步伐 ([报告/代码](https://twitter.com/TransluceAI/status/1963286326062846094))。
- **微调“暗号”攻击**：Anthropic 分析了看似良性的微调数据如何编码有害的隐藏指令，并讨论了针对 FT API 的缓解措施 ([@JackYoustra](https://twitter.com/JackYoustra/status/1963280250923868239))。
- **隐式推理 + 机械可解释性 (Mech Interp)**：一项新综述整合了关于 LM 中隐式推理的工作 ([论文](https://arxiv.org/abs/2509.02350), [@omarsar0](https://twitter.com/omarsar0/status/1963236545705710070))。在机械可解释性方面，逐层相关性传播 (LRP) 与原生梯度方法相比，显著提高了归因补丁 (Attribution-patching) 的保真度 ([@NeelNanda5](https://twitter.com/NeelNanda5/status/1963029426741854345))；Neel 还发布了一份全面的“入门”v2 指南，并开启了一个 MATS 方向 ([指南推文](https://twitter.com/NeelNanda5/status/1963225482973040784))。

**融资、产品与采用信号**

- **Agent 搜索**：Exa 在 Benchmark 领投下筹集了 8500 万美元，用于构建 AI 原生网页搜索基础设施 ([@ExaAILabs](https://twitter.com/ExaAILabs/status/1963262700123000947))。[You.com](http://you.com/) 以 15 亿美元估值筹集了 1 亿美元，并声称其客户每月查询量超过 10 亿次，专门为 Agent 的深度、实时检索进行了优化 ([@RichardSocher](https://twitter.com/RichardSocher/status/1963277700711461241), [Bloomberg](https://twitter.com/business/status/1963226665275769327))。
- **基础设施整合**：CoreWeave 收购了 OpenPipe；预计 ART RL 微调流水线将与高性能推理基础设施进行更紧密的集成 ([@corbtt](https://twitter.com/corbtt/status/1963332919864557784), [@shawnup](https://twitter.com/shawnup/status/1963335514377130397))。
- **平台功能普及**：OpenAI Projects 现已向免费用户开放，并扩展了单个项目的上传限制和记忆控制 ([@OpenAI](https://twitter.com/OpenAI/status/1963329936368046111))。Perplexity 为学生推出了 Comet（包含广告拦截、学习模式、排程、原生助手）([@perplexity_ai](https://twitter.com/perplexity_ai/status/1963285255198314951))。
- **企业级应用**：Coinbase 报告称，每日约 40% 的代码由 AI 生成，目标是到 10 月份超过 50%，同时保留人工审核 ([@brian_armstrong](https://twitter.com/brian_armstrong/status/1963315806248604035))。

**热门推文（按互动量排序）**

- Higgsfield 在 “Nano Banana” 上的 Draw-to-Edit 展示了单流多模型绘图与动画编辑——其病毒式传播反映了多模态 UX 的快速进展 ([@higgsfield_ai](https://twitter.com/higgsfield_ai/status/1963035734232928586))。
- OpenAI Projects 扩展至免费层级；更大的单个项目文件限制和项目范围内的记忆控制，标志着通过 Projects 实现更深层的应用集成和数据路由 ([@OpenAI](https://twitter.com/OpenAI/status/1963329936368046111))。
- Codex CLI 势头强劲：与之前的助手相比，在长程指令遵循和“不放弃”行为方面取得了显著的定性优势；据报道，两周内使用量增长了约 10 倍 ([@Yampeleg](https://twitter.com/Yampeleg/status/1963260958257578497), [@sama](https://twitter.com/sama/status/1963365966953505103))。
- 人形机器人消费级演示持续吸引关注——Figure 展示了洗碗/洗衣服技能，并正在 AI 和制造领域招聘人才 ([@adcock_brett](https://twitter.com/adcock_brett/status/1963266402028335567))。
- Exa 的 8500 万美元融资和 [You.com](http://you.com/) 的 1 亿美元融资凸显了“Agent 搜索”论点：Agent 优先的索引和检索基础设施是战略资产 ([@ExaAILabs](https://twitter.com/ExaAILabs/status/1963262700123000947), [@RichardSocher](https://twitter.com/RichardSocher/status/1963277700711461241))。
- VS Code 对自定义 OAI 兼容端点的支持是本地/自托管技术栈的低调推动者——减少了被锁定在单一供应商的理由 ([@ggerganov](https://twitter.com/ggerganov/status/1963255949373677959))。

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. Kimi K2 发布与 LLM 基准测试排行榜

- [**介绍 Kimi K2-0905**](https://www.reddit.com/r/LocalLLaMA/comments/1n7fdy4/introducing_kimi_k20905/) ([得分: 391, 评论: 85](https://www.reddit.com/r/LocalLLaMA/comments/1n7fdy4/introducing_kimi_k20905/)): **“Kimi K2-0905”的公告仅包含一张宣传图，没有任何技术细节、基准测试、权重、代码或 API 信息；该帖子仅链接到一个图片资源：https://preview.redd.it/u8oxbcfyfymf1.png?width=2178&format=png&auto=webp&s=87daf02d6f257631f0a0a8847de7180dc9d9eed8。帖子正文中未提供模型卡（model card）、变更日志或发布产物。** 热门评论批评了其营销/用户体验（“看起来像加密货币空投诈骗广告”、“一半是 AI 废话，一半是 Z 世代风格”），并对发布细节提出质疑：*“没有权重？我猜会在 5 号发布（除非只提供 API）。”*
    - 缺少已发布的权重引起了注意；一位评论者推测 0905 标签暗示将于 9 月 5 日发布，除非它仅限 API。这引发了对自托管和独立基准测试（延迟/吞吐量、上下文长度、评估可复现性和许可）的实际担忧，而这些只有在开放权重的情况下才可行。
    - 时机和定位：一位评论者表示，第一代 K2 的光芒被 Qwen 3 Coder 的发布所掩盖，这表明 K2-0905 将在代码基准测试中受到严格审查，并与 Qwen 3 Coder 进行正面交锋，特别是在代码合成和修复任务方面。
- [**根据 Artificial Analysis 包含工具调用和 Agent 评估的新智能指数，GPT-OSS 120B 现已成为全球顶尖的开源模型**](https://i.redd.it/6c1jae9atvmf1.png) ([得分: 337, 评论: 204](https://www.reddit.com/r/LocalLLaMA/comments/1n75z15/gptoss_120b_is_now_the_top_opensource_model_in/)): **Artificial Analysis 的新智能指数综合了开源 LLM 在学术评估（如 MMLU‑Pro, GPQA Diamond）以及工具调用和 Agent 任务中的表现；根据图表，GPT‑OSS 120B 以综合得分** `58` **位居第一，领先于 Qwen3 和 DeepSeek 等模型（其他模型得分在** `57–21` **之间）。方法论详见：https://artificialanalysis.ai/methodology/intelligence-benchmarking；该指数报告了由多项评估得出的单一综合得分。** 评论对排名提出了质疑：一位用户认为 GLM 4.5 最接近 Claude Sonnet/Opus，而另一位用户则质疑 Gemma 3 的排名为何在 Phi‑4 之后，暗示对权重分配或任务覆盖范围存在分歧。
    - 一位从业者声称，尽管有该指数，**GLM 4.5** 在能力上仍是与 **Claude 3.5 Sonnet** 或 **Claude Opus** 最接近的开源模型。这表明在他们的工作负载中，**GLM 4.5** 在通用推理/对话质量方面被认为与顶尖专有模型几乎持平。
    - 一位评论者质疑为什么 **Gemma 3** 的排名低于 **Phi-4**，隐含地探究了该指数的 Agent/工具调用权重可能如何有利于某些模型家族或训练方案。这突显了排名对评估设计的潜在敏感性，鼓励人们审视工具使用和多步任务是如何评分的。
    - 对基准驱动的排行榜持怀疑态度：一位用户认为 *“现实世界的实际使用才是真正的数学”*，并且开源模型在他们的使用案例中“并不划算”。他们暗示排行榜得分可能无法直接转化为生产环境中的有效性，挑战了新指数的实际相关性。
- [**德国版“谁想成为百万富翁”基准测试与领先模型对比**](https://www.reddit.com/gallery/1n7g0c2) ([得分: 190, 评论: 47](https://www.reddit.com/r/LocalLLaMA/comments/1n7g0c2/german_who_wants_to_be_a_millionaire_benchmark_w/)): **作者使用原始规则在领先的 LLM 中重新运行了德国版《Wer wird Millionär?》问答基准测试：** `45` **次模拟游戏运行，每次包含** `15` **个 A–D 多选题（德语），没有求助机会，答错一题即结束运行并保留当前奖金。他们复用了公开的 WWM 语料库（[数据集](https://github.com/GerritKainz/wer_wird_millionaer)）和原始基准测试概念（[ikiruneo/millionaire-bench](https://github.com/ikiruneo/millionaire-bench)），为了透明度添加了并行英文文本（**`fragen_antworten_en.json`**），并在新仓库中提供了批量评估和排行榜重建脚本（**`millionaire-run.py`**,** `rebuild_leaderboard.py`**）：[Jose-Sabater/millionaire-bench-opper](https://github.com/Jose-Sabater/millionaire-bench-opper)。结果通过排行榜截图分享（评分/结构与原始版本相同），并且该设置已打包以便快速重新运行或提交 PR。** 评论者建议加入真实节目中的“放弃以保留奖金”决策点，并衡量模型何时/是否选择停止，将其转变为一种风险感知评估。还有人请求包含更多模型（例如 Gemini 2.5 Pro）。

- 基准测试设计细节：一个“百万富翁”式的评估应该通过要求模型提供校准后的正确概率，并根据该节目阶梯式奖金/保险金结构下的期望值来决定是回答还是放弃，从而显式地模拟“退出”选项。除了 QA 准确率之外，这还测试了风险敏感型决策和置信度校准（例如 Brier/ECE）；相关证据表明 LM 可以估计其自身的不确定性，参见 Kadavath et al. 2022 的 *Language models (mostly) know what they know* (https://arxiv.org/abs/2207.05221)。同时报告平均奖金和校准指标，可以将那些“知道何时退出”的模型与那些过度自信或缺乏自信进行猜测的模型区分开来。
- 语言干扰因素：使用德语版本主要探测的是多语言理解和植根于文化的知识，而不仅仅是通用推理。许多模型在从英语转向其他语言时表现出明显的下降（例如，MGSM 报告了跨语言的巨大差距：https://arxiv.org/abs/2305.11938；XTREME 中更广泛的跨语言差异：https://arxiv.org/abs/2003.11080），因此英语测试可能会提升以英语为中心的模型的排名。为了隔离推理能力与语言能力的干扰，可以考虑并行进行德语/英语测试或使用翻译受控的变体。
- 模型比较的细微差别：有传闻称 **GLM-4.5** 生成的代码与 “GPT-5” 旗鼓相当，这表明两者在编程任务上具有对等性，但“百万富翁”式的琐事问答强调的是事实召回和经过校准的 QA。为了验证跨领域的主张，应在代码基准测试（如 HumanEval: https://github.com/openai/human-eval；MBPP: https://arxiv.org/abs/2108.07732）以及知识 QA（如 Natural Questions: https://ai.google.com/research/NaturalQuestions）上进行比较。预计会出现模型在编程上趋同但在开放域知识和校准上分化的情况，从而影响“百万富翁”测试的结果。

### 2. GPU Hardware: Intel Arc Pro B50 and 4x3090 vs RTX 6000

- [**Intel launches Arc Pro B50 graphics card at $349**](https://i.redd.it/357rwwhaizmf1.jpeg) ([Score: 150, Comments: 108](https://www.reddit.com/r/LocalLLaMA/comments/1n7l5kg/intel_launches_arc_pro_b50_graphics_card_at_349/)): **根据 VideoCardz 的报道，Intel 推出了售价 349 美元的 Arc Pro B50 工作站 GPU，定位为廉价专业卡，并作为 NVIDIA A1000 的替代品进行销售。该帖子和缩略图提出了大胆的主张（“优于 NVIDIA”），但未提供硬性基准测试；讨论中提到的一个规格是 ~**`224 GB/s` **显存带宽，暗示其性能处于中端水平。来源：https://videocardz.com/newz/intel-launches-arc-pro-b50-graphics-card-at_349** 评论者认为 `224 GB/s` 的带宽具有局限性，RTX 3060 的表现会超过它；一些人希望有更多 VRAM，另一些人则声称 RTX 5060 Ti（约贵 80 美元）由于支持 CUDA 且带宽更高而具有更好的性价比，甚至二手双路 3060 也被认为更优。
    - 带宽是一个反复被提及的问题：评论者指出 Arc Pro B50 约 `~224 GB/s` 的显存带宽（暗示 128-bit GDDR6 接口）是一个瓶颈，并将其与 `360 GB/s` 的 **RTX 3060 12GB** 进行了对比（[规格](https://www.techpowerup.com/gpu-specs/geforce-rtx-3060-12-gb.c3621)）。预期在许多对带宽敏感的工作负载中，3060 的表现将优于 B50。
    - 几位用户强调，缺乏 **CUDA** 是专业/计算工作流的一个主要缺陷。如果没有 CUDA ([NVIDIA CUDA](https://developer.nvidia.com/cuda-zone))，在许多 DCC/ML/计算应用中的兼容性和性能可能会落后于 NVIDIA 的选择，即使 B50 的原始规格在某些领域具有竞争力，也会削弱其价值。
    - 价值以及相对于 Intel 自身产品线的定位：一位用户认为 B50 比 B580 贵“100 美元”，但在大多数方面却更慢，B50 唯一的明显优势是 `+4 GB VRAM` 以及更小、功耗更低的 SFF 外形。结论是：除非你明确需要 SFF 和低功耗，否则 **B580** 被认为是更快且更便宜的选择。

- [**除了功耗，4 x 3090（总计 2400 美元）对比 RTX pro 6000（9000 美元）还有什么实际缺点吗？**](https://www.reddit.com/r/LocalLLaMA/comments/1n71b95/any_actual_downside_to_4_x_3090_2400_total_vs_rtx/) ([Score: 158, Comments: 184](https://www.reddit.com/r/LocalLLaMA/comments/1n71b95/any_actual_downside_to_4_x_3090_2400_total_vs_rtx/)): **OP 询问对于像 “Qwen 3 Coder” 和 “GLM 4.5 Air” 这样的本地 LLM，4× RTX 3090（总计约 2.4k 美元，Ampere 架构，每张 24 GB）是否是单张 RTX 6000 级专业显卡（约 9k 美元）的实用替代方案。热门回复指出 VRAM 并非简单叠加：除非使用张量/流水线并行（例如 Megatron-LM tensor-parallel），否则模型必须能装入单张 GPU，而这会引入 NCCL/PCIe 通信开销；消费级主板通常会将通道拆分为 x8/x8/x4/x4 甚至更糟，因此 4 张 GPU 可能各以 ~x4 运行，从而损害扩展性。Ampere 架构缺乏原生低精度路径（FP8/FP4），而较新的技术栈正日益以此为目标，因此像 vLLM 这样的引擎可能会滞后或需要变通方法；有效 VRAM 会因 CUDA/运行时开销而减少；二手 GPU 存在可靠性风险，而 RTX 6000 级显卡提供更好的厂商支持/驱动程序。** 评论者对 600 美元一张 3090 的价格表示怀疑，并认为由于互连瓶颈和并行化开销，单张大显存 GPU 几乎总是比多张小显卡更快、更简单。
    - PCIe 通道瓶颈将限制消费级平台上的 4×3090：每张 3090 都期望 x16 链路，但典型的桌面 CPU 总共仅暴露约 `24` 条通道，因此四张显卡最终各占 ~x4，大幅削减了主机与设备间的带宽（`PCIe 4.0 x4 ≈ ~8 GB/s` 对比 `x16 ≈ ~32 GB/s`），并损害了多 GPU 吞吐量；你需要一个拥有 64+ 通道的显卡工作站/HEDT 平台来避免这种情况（[PCIe bandwidth](https://en.wikipedia.org/wiki/PCI_Express#History_and_revisions)）。在实践中，对于单模型训练/推理，单张大显卡由于减少了 GPU 间的同步和通信开销，性能往往优于多张小显卡。
    - 多 GPU LLM 扩展会增加开销：每张显卡的有效 VRAM 会因 CUDA 上下文/分配器开销和张量并行分片而下降，虽然张量并行配置起来可能很繁琐，但流水线并行会引入气泡（bubbles），从而降低利用率/吞吐量（参见 [vLLM parallelism](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)）。Ampere (3090) 缺乏原生的 FP8/FP4 Tensor Core 模式，而 **RTX 6000 Ada** 在第四代 Tensor Cores 上支持 FP8（[RTX 6000 Ada](https://www.nvidia.com/en-us/design-visualization/rtx-6000/)），因此较新的推理/训练优化可能会率先落地；预计 Ampere 在引擎支持方面需要等待更长时间。
    - 总拥有成本：根据讨论，满负荷运行的 4×3090 对比单张 RTX 6000 Ada 可能意味着每年额外消耗约 `~7,000 kWh` 的电能，根据当地电价，这可能高达“每年 `3,000` 美元以上”，此外还有额外的散热/空调成本。标称板卡功耗也印证了这一趋势（3090 每张约 `350 W`，而 RTX 6000 Ada 总计约 `300 W`）（[3090 specs](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090/)，[RTX 6000 Ada](https://www.nvidia.com/en-us/design-visualization/rtx-6000/)）。二手 3090 还面临更高的故障风险和更早的软件/驱动程序 EOL，而专业卡通常具有更长的支持周期和厂商保障。

## 技术性较低的 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Gemini 3 预训练成功 + Tesla Optimus 3 首张照片/视频

- [**看起来 Gemini 3 可能已经成功完成了预训练运行**](https://www.reddit.com/gallery/1n7cark) ([Score: 319, Comments: 111](https://www.reddit.com/r/singularity/comments/1n7cark/looks_like_gemini_3_mightve_had_a_successful/)): **一篇帖子断言 Google DeepMind 的下一代模型 “Gemini 3” 已完成成功的预训练运行，暗示核心无监督训练可能已经结束。然而，目前尚未披露任何技术细节（Token 数量、算力规模、架构/窗口变化或评估结果），且链接的证据是一个返回** `HTTP 403` **的 Reddit 图集（[gallery link](https://www.reddit.com/gallery/1n7cark)）。评论者报告称，Gemini 预训练的一位共同负责人公开反驳了这一说法，表明该信息可能为时过早或不准确。** 讨论分为时间线推测（例如，“现在预训练完成 → 年底发布？”）和可信度担忧，多位用户引用了负责人的否认并质疑来源（“Dylan”）。一些人询问否认是否意味着 Gemini 3 并非“性能极其强劲”，而另一些人则指出这可能仅仅表明传闻毫无根据，而非与性能相关。

- 关于 **Gemini 3** 预训练刚刚完成的推测（暗示可能在年底发布）存在争议：据报道，一位被引用的 **Gemini pretraining co-lead** 否认了传闻来源的说法，因此没有可靠的确认表明训练已完成，或者该模型已经“表现极其出色”。从技术层面看，在没有官方信号（如论文、博客或基准测试增量）的情况下，完成推论是站不住脚的；发布时间仍处于推测阶段。
- 评论者澄清，被引用的 "Woodward" 推文是关于 "nano banana" 的流行程度，而非 LLM 预训练的里程碑——类似于 **OpenAI** 在发布前后开的“服务器着火”之类的玩笑。结论：该推文属于社交闲聊，并非 **Gemini 3** 训练状态或性能进展的指标。
- 多位用户对 **Dylan Patel** 传闻的可靠性表示谨慎；在缺乏硬性指标（如 MMLU, GPQA, BIG-bench 或 `ARENA Elo`）或官方评估的情况下，声称“性能惊人”还为时过早。技术上审慎的做法是在推断能力或就绪程度之前，等待可复现的基准测试和方法论细节。
- [**Optimus 3 的首个视频**](https://v.redd.it/jjplx5j3kzmf1) ([Score: 596, Comments: 453](https://www.reddit.com/r/singularity/comments/1n7lebe/first_video_of_optimus_3/)): **帖子分享了 Tesla 人形机器人 “Optimus 3” 的“首个视频”，链接指向一个 Reddit 托管的剪辑 [v.redd.it/jjplx5j3kzmf1](https://v.redd.it/jjplx5j3kzmf1)，目前返回** `HTTP 403` **（网络安全拦截），因此无法从源头验证任何技术内容（运动、操作、自主堆栈、传感器或基准测试）。由于媒体内容无法访问，帖子本身没有提供规格或实现细节来与之前的 Optimus 迭代进行对比，因此无法仅凭此链接评估任何硬件/控制堆栈的更改。** 热门评论多为非技术性的怀疑态度，暗示更新似乎是外观上的而非功能性的（例如，“现在他能以增加 30% 的亮度无所事事”、“NPC”/“Gen Z 凝视”），表明感知到的能力提升微乎其微。
- [**Optimus 3 的首张照片**](https://i.redd.it/b36k6a7afzmf1.jpeg) ([Score: 300, Comments: 169](https://www.reddit.com/r/singularity/comments/1n7ko97/first_photo_of_optimus_3/)): **Tesla 第三代人形机器人 “Optimus 3” 的首张公开图像显示了一个精致的外壳，带有反光的头部/躯干，明显的 Tesla 品牌标识，以及在办公环境中行走的一个更苗条、比例更接近人类的框架。值得注意的是高度拟人化的手部和全关节肢体，暗示设计重点在于灵巧性和自然步态，尽管帖子中未提供规格或演示。** 评论指出了反复出现的底盘/接口笑话（那个“洞”），并批评了可能的骨盆对齐问题，而其他人则指出，如果这些手部功能完备，看起来异常像人——这暗示了对其是装饰性的还是具备实际能力的怀疑。
    - 评论者强调了手部明显的真实感——“如果那些手能工作……那是我在机器人上见过的最像人类的手。”从技术上讲，几何形状暗示了拟人化的比例和潜在的高 DOF、独立驱动的手指；如果功能完备，这将实现灵巧的手内操作，并比之前的 Optimus 演示具有更广泛的抓取分类。
    - 一位观察者指出“他们把骨盆装错了”，暗示髋部/骨盆接口未对齐。这种错位将影响髋关节运动学、运动范围以及步态稳定性的质心对齐；或者，这可能是早期原型安装中常见的临时装饰外壳/盖板方向问题。
    - 关于“那个洞有更新吗？”的问题暗示了之前迭代中注意到的底盘孔径/外壳间隙。这表明包装/外壳集成仍在变动中，机械闭合和布线在原型阶段尚未完全定型。

- [**AI 在 100 年内都不会取代的一项工作是……编程 —— 比尔·盖茨**](https://www.leravi.org/bill-gates-reveals-the-one-job-ai-will-never-replace-even-in-100-years-10272/) ([Score: 507, Comments: 167](https://www.reddit.com/r/OpenAI/comments/1n72qgw/the_one_job_ai_wont_take_in_100_years_is/)): **比尔·盖茨表示，即使在** `100` **年后，编程仍将是一项“100% 的人类职业”，他断言 AI 将自动化重复性的编码工作，但无法取代软件工程核心的创造性问题解决和判断力（[Le Ravi 援引 France Inter 的报道](https://www.leravi.org/bill-gates-reveals-the-one-job-ai-will-never-replace-even-in-100-years-10272/)）。热门评论者以技术视角进行了反驳：目前的 LLM 可以扩展到更长的任务，但在长周期、跨年度、多团队的目标（例如“发布一款‘惊艳’的游戏”）上仍受限制，因此它们擅长分解后的子任务，但仍需要人类主导的规范制定、编排和集成。编程仍然是目前 AI 实际帮助最大的领域（代码生成、重构、测试），但针对数月至数年项目的可靠自主 Agent 仍是一个悬而未决的问题。** 辩论分为两派：(1) 长周期自主性是关键障碍——人类将留在环节中，负责定义、分解并承担端到端的结果；(2) 编程由于其语言原生性、高收益以及海量的训练和合成数据，特别容易受到自动化的影响——如果 AI 无法胜任这份工作，它可能也无法胜任大多数其他工作。
    - 关键的技术主张是关于任务周期限制：目前的 LLM 处理简短、范围明确的编码任务，但在需要稳定目标、架构和分层分解的数月至数年的多人软件项目上表现吃力。Agentic 编码系统在仓库级变更、依赖管理和长期连贯性方面仍然步履蹒跚；尽管在片段级代码生成方面表现强劲，但像 SWE-bench (https://www.swebench.com/) 这样的基准测试显示，在多文件 Bug 修复方面的端到端成功率有限，这使得人类仍需负责规划和编排工作。
    - 反方观点强调了为什么编程异常适合 LLM 自动化：它完全由语言介导，拥有庞大的公共训练语料库（例如开源仓库），并支持通过测试生成和 fill-in-the-middle 预训练产生合成数据。至关重要的是，编译器、Linter 和单元测试提供了快速、自动的反馈循环，支持“执行-调试-重试”工具链和 RL 风格的信号，这表明软件工程可能是首批出现稳健自主性的领域之一。
    - 从业者角度：LLM 在编程中提供了最大的助力，加速了样板代码、测试、重构和 API 胶合代码的编写，而人类则负责产品定义、架构和跨系统集成。经验数据支持了在常规任务上的大幅提速——例如，GitHub 的研究报告称，使用 Copilot 后任务完成速度提高了约 `55%` ([https://github.blog/2022-09-07-research-quantifying-github-copilots-impact-on-developer-productivity/)——然而](https://github.blog/2022-09-07-research-quantifying-github-copilots-impact-on-developer-productivity/)%E2%80%94yet) 长周期规划和不断演进的需求对于当前模型来说仍然具有挑战性。

### 2. OpenAI 家长控制/隐私与 UX 抵制 + Salesforce AI 裁员

- [**Salesforce CEO 确认裁员 4,000 人，称“有了 AI 我需要的人手更少了”**](https://www.cnbc.com/2025/09/02/salesforce-ceo-confirm-4000-layoffs-because-i-need-less-heads-with-ai.html) ([Score: 494, Comments: 178](https://www.reddit.com/r/singularity/comments/1n722tp/salesforce_ceo_confirms_4000_layoffs_because_i/)): **Salesforce CEO Marc Benioff 在播客中确认，通过其客服机器人（“Agentforce”）实现的 AI 自动化已显著减少了支持案例量，足以削减约** `4,000` **个客户支持岗位——将支持团队人数从约** `9,000` **人缩减至约** `5,000` **人——且公司不会填补这些职位的空缺；Benioff 此前曾声称 AI 在 Salesforce 承担了高达** `50%` **的工作。CNBC 报道：https://www.cnbc.com/2025/09/02/salesforce-ceo-confirms-4000-layoffs-because-i-need-less-heads-with-ai.html。被引用的分析师包括 Laurie Ruettimann（敦促重新培训而非裁员）和 Ed Zitron（批评疫情后的过度招聘并将 AI 作为裁员的借口）。**
    - 一位评论者声称，尝试用 AI 取代人工客服的公司中，约有 `50%` 报告了“糟糕的体验”，并指出了核心局限性：LLM 幻觉、客户对机器人的不满，以及无法执行简单 FAQ 之外的经过身份验证/账户级别的操作。这一点暗示，生产级支持自动化需要安全的动作执行（与身份验证/审计相关的工具/API 集成）、稳健的人工坐席回退机制以及防止错误操作的护栏——而这些正是目前 AI 部署经常表现不足的领域。

- [**Salesforce 首席执行官 Marc Benioff 表示 AI 让他裁减了 4,000 个职位**](https://www.finalroundai.com/blog/salesforce-ceo-marc-benioff-says-ai-enabled-him-cut-4000-jobs) ([Score: 677, Comments: 158](https://www.reddit.com/r/ChatGPT/comments/1n76jmf/salesforce_ceo_marc_benioff_says_ai_enabled_him/)): **Salesforce 首席执行官 Marc Benioff 表示，在部署了目前处理约** `50%` **客户对话的 AI Agent 后，公司裁减了约** `4,000` **个客户支持职位；自 2025 年初以来，每种类型的 Agent 处理了约** `1.5M` **次交互，并据报道推动了** `17%` **的支持成本降低。他提到了支持 AI 的全渠道监管和 Agent 式销售系统，这些系统扩展了支持和内部外联能力（每周 >** `10k` **个线索），AI 处理与人工处理对话的 CSAT（客户满意度）持平，且仅有“数百人”被重新安置，同时预示着将进一步开展逐个职能部门的自动化——这与其 2025 年 7 月“增强而非取代”的立场发生了逆转。此举与 2025 年大型科技公司（如 Microsoft, IBM, Coinbase）更广泛的 AI 驱动裁员趋势相符。** 评论质疑在自动化一线职位的同时保留高薪高管，并指出实际风险：AI 支持循环可能会阻碍保修/消费者权利的执行，而人类可以升级处理或行使裁量权；AI 系统可能会放大本地化/法律能力差距（例如，不熟悉欧盟法律的非欧盟支持人员）。
    - 客户支持自动化的局限性：一位评论者认为，AI 聊天机器人通常无法进行具备司法管辖权意识的推理和执行，特别是对于欧盟/德国的保修案例，并指出人类最终可能会在坚持下授予权利，而 AI 可能会在没有升级的情况下无限循环。技术启示：生产环境的支持机器人需要针对特定国家的政策引擎和知识库、强制人工移交的置信度阈值，以及符合消费者保护规则的可审计决策日志（例如，欧盟消费者权利指令 2011/83/EU: https://eur-lex.europa.eu/eli/dir/2011/83/oj）。
- [**孩子不需要家长控制，他们需要家长的关怀。**](https://i.redd.it/30c2s59csumf1.jpeg) ([Score: 381, Comments: 217](https://www.reddit.com/r/OpenAI/comments/1n71u8m/kids_dont_need_parental_controls_they_need/)): **该图片是一张新闻截图，称 OpenAI 的 ChatGPT 将增加家长控制功能，如果系统检测到年轻用户有** `acute distress` **（极度痛苦）的迹象，可以“通知家长”，据报道这是受一起青少年自杀案启发；根据 [华盛顿邮报的报道](https://www.washingtonpost.com/technology/2025/09/02/chatgpt-parental-controls-suicide-openai/)，这涉及痛苦检测和家长关联账户流程，尽管具体细节（使用的信号、阈值、选择加入/同意模型、数据保留和升级路径）尚未详述。该帖子的标题认为仅靠控制是不够的，暗示这应该是更广泛的儿童安全和监护政策转变，而不仅仅是一个 UI 开关。** 评论意见不一：一些人认为家长控制是关怀的一部分，而另一些人则警告隐私风险（暴露 LGBTQ+ 青少年的身份、提醒虐待性家长），并强调结果取决于实施方式——选择加入机制、安全联系人 vs 家长、隐私保护措施以及误报处理。
    - 实施风险集中在“家长控制”如何构建：是启用家长仪表板、聊天记录可见性，还是关于敏感话题的自动警报。评论者警告分类器和政策设计（例如，针对身份/心理健康查询的 `false-positive` 误报警报）可能会将高度敏感的数据泄露给不安全的监护人，建议采用细粒度范围（内容 vs 元数据）、针对较大未成年人的同意门槛，以及明确的升级标准，以避免在极端情况（如家庭虐待）中造成伤害。
    - 安全/规避担忧：应用层控制很容易被青少年绕过（新账户、不同设备、VPN、替代模型），因此任何真正的控制必须是深度防御（操作系统级配置文件、MDM、网络/DNS 过滤器）以及强大的账户/年龄绑定。否则，单个应用中的日志记录或警报只会提供虚假的安全感，同时很容易被绕过。
    - 安全架构建议强调保护隐私的干预措施而非家长披露：设备端提醒、临时或默认加密存储，以及针对危机话题抑制家长可见日志但仍提供资源的“机密模式”。升级流程应优先选择第三方热线/资源，并要求未成年人明确同意通知家长，同时为分类器设置可审计的阈值，以尽量减少 `false-negative/false-positive`（漏报/误报）伤害。

- [**新的“家长模式”正在羞辱成年人，并扼杀了 ChatGPT 的独特之处**](https://www.reddit.com/r/ChatGPT/comments/1n7ioo0/the_new_parental_mode_is_patronizing_adults_and/) ([分数: 261, 评论: 251](https://www.reddit.com/r/ChatGPT/comments/1n7ioo0/the_new_parental_mode_is_patronizing_adults_and/)): **用户报告 ChatGPT 中出现了一个新的全局安全层（“家长模式”），在各模型（包括 [GPT‑4o](https://openai.com/index/hello-gpt-4o/)）中应用了更严格的审核，即使在明显的虚构/创意语境下，涉及自残/“敏感”内容的触发词也会导致自动弹出求助热线干预。** 一条高赞评论描述了可复现的行为，表明存在一个服务器端的生成后过滤器：助手否认进行了拦截，将其归因于外部过滤器，并建议绕过方法，但同样的干预文本却被反复注入——这暗示了一个独立于模型输出且不可覆盖的策略层。发帖者还指控存在隐蔽的模型更换和出于节省成本目的的降级、透明度降低，以及扩大的“敏感内容”定义影响了合法用例；详情参见 OpenAI 的通用 [使用政策](https://openai.com/policies/usage-policies)。辩论集中在法律责任与用户自主权之间：一些人认为公司为了避免因自残事件引发的诉讼而“削弱（nerf）”模型，而另一些人则要求提供退出选项和成人控制功能，声称触发阈值过于宽泛且破坏了工作流。
    - 多名用户报告了来自服务器端自残/敏感内容安全层的可复现误报，该层会覆盖模型，即使在明显的虚构语境中也会返回预设的热线文本。一位用户指出，模型本身承认“我触发了一个过滤器”，这暗示是一个生成后的审核环节，而非基础模型的选择；按照模型指导尝试重新措辞在约 `~7` 次尝试中依然会重新触发过滤器——这证明了一个对叙事框架和先前聊天历史不敏感的高召回率、低精确度分类器的存在。
    - 触发机制似乎是由关键词/短语驱动的（例如“自寻短见”、“流血”、监禁/地狱场景），对成人/创意用例的语境处理较差，且没有会话级别的例外处理。这表明输入和/或输出审核分类器的运行独立于系统意图（虚构写作）和人格设定，类似于 **OpenAI** 官方文档中描述的典型多阶段流水线（提示词分类 + 补全分类）：https://platform.openai.com/docs/guides/moderation/overview。
    - 评论者推测最近的政策/阈值转变（“家长模式”）优先考虑合规性/降低责任风险而非精确度，实际上将拦截范围扩大到了 S3/S4 类别（自残、暴力），即使是在第三人称或假设性的描述中也是如此。用户在技术上建议的缓解措施包括：语境感知安全（尊重“虚构”标签）、可调节的阈值或按账户设置的开关，以及模式切换（例如“研究/虚构模式”），以便在不移除防护栏的情况下减少过度拦截。
- [**OpenAI 正在迅速消亡，你不再受保护了**](https://i.redd.it/qyk1kjdumymf1.jpeg) ([分数: 4400, 评论: 1016](https://www.reddit.com/r/ChatGPT/comments/1n7gcyi/openai_is_dying_fast_youre_not_protected_anymore/)): **该图片是一个煽动性的迷因式声明，声称“OpenAI 正在扫描用户的 ChatGPT 对话并将内容报告给警方”。** 实际上，OpenAI（与大多数在线平台一样）会对用户输入/输出运行自动安全/审核系统，并在其政策中规定，当法律要求或为了防止迫在眉睫的伤害时，可能会向执法部门披露信息；这并非全盘、主动的“报告一切”制度，而是科技平台通用的内容审查和法律合规工作流（[隐私政策](https://openai.com/policies/privacy-policy)，[使用政策](https://openai.com/policies/usage-policies)）。用户可以限制其聊天记录用于训练（例如通过聊天历史记录控制；Enterprise/Teams 版本提供更强的数据保留和训练退出选项），但出于安全考虑，审核扫描仍然适用。热门评论大多持愤世嫉俗的态度，断言用户数据从未私密，并质疑模型训练数据的合法性/伦理。技术辩论极少；大多数反应是非技术的，或者是关于极端提示词被标记/报告的幽默调侃。

- 一位评论者指出，OpenAI 承认“有一个小团队负责监控风险对话”，这与 OpenAI 的 human-in-the-loop 审核流程相一致：自动分类器会标记安全敏感类别（如自残、暴力、非法活动），并可能上报给有限的授权审核人员，以执行政策和改进模型。实际上，除非禁用了数据共享（关闭 ChatGPT 的“聊天记录与训练”，或选择退出 API 数据使用；企业版默认为关闭），否则用户内容可能会被审核并用于训练。参考资料：OpenAI Privacy Policy (https://openai.com/policies/privacy-policy), Data usage controls (https://help.openai.com/en/articles/7934734-how-your-data-is-used-to-improve-model-performance), Usage Policies (https://openai.com/policies/usage-policies)。
- 另一个帖子指出了对训练数据合法性和隐私的担忧：OpenAI 表示模型是在**公开可用**、**获得许可**以及**人工生成**的数据混合体上训练的，但尚未披露具体的来源，这增加了对大规模网络语料库中可能包含受版权保护或个人数据的审查。这种数据集透明度的缺乏是竞争秘密与问责制之间已知的权衡，并对合规性和数据来源的 red-teaming 产生影响。参考资料：GPT-4 Technical Report (https://cdn.openai.com/papers/gpt-4.pdf) 和 Privacy Policy (https://openai.com/policies/privacy-policy)。
- [**此过滤器需要移除**](https://v.redd.it/3206q93s5vmf1) ([评分: 280, 评论: 88](https://www.reddit.com/r/OpenAI/comments/1n73fx1/this_filter_needs_to_be_removed/))：**用户报告 OpenAI 不同模型变体之间的安全审核不一致：查询“犹大是自缢身亡的吗”时，** `5 (Instant)` **和** `GPT‑4o` **([模型信息](https://openai.com/index/gpt-4o/)) 直接给出了回答，但** `5 (Thinking)` **变体在开始回答后触发了安全插页/审查。另一位评论者指出，枪支法律查询（例如，检查租用机枪的合法性，这在某些司法管辖区的美国 NFA 规则下是合法的）弹出了危机/求助热线信息，而不是直接的法律指导——这表明在推理/“Thinking”路径上存在更激进的意图分类。链接的视频 ([v.redd.it](http://v.redd.it/)) 返回 HTTP** `403` **错误，需要身份验证，这表明是访问控制而非内容删除。有关通用模型的参考，请参阅 OpenAI 的 [模型文档](https://platform.openai.com/docs/models)。** 评论者将 `5 (Thinking)` 模型描述为过度限制/“被削弱（nerfed）”，认为其安全过滤器与 `5 (Instant)` 和 `GPT‑4o` 相比过于敏感；不满集中在生成过程中的审查以及对合法信息查询插入求助热线。
    - 对 `5 (Instant)`、`5 (Thinking)` 和 `4o` 的 A/B 测试显示，在提示词“犹大是自缢身亡的吗”上存在不同的安全行为：`5 (Instant)` 和 `4o` 直接回答而没有拒绝，而 `5 (Thinking)` 开始回答后转为拒绝。这指向了专门针对“Thinking”变体的后期审核覆盖（例如，可以在生成过程中中途删除/替换答案的生成后安全检查），而不是跨模型的统一政策。这种差异意味着模型特定的安全阈值/分类器，其中“Thinking”模型针对自残措辞进行了更激进的调整，即使是在历史/学术背景下。
    - 关于合法枪支查询误报的报告：询问购买枪支和州枪支法律（包括检查“租用机枪”的合法性）触发了危机/支持信息和拒绝回答。这表明关键词驱动的暴力/自残分类器在对意图中立的法律研究中过度触发，更倾向于高召回率而非准确率。更好的配置应该是根据用户意图和司法背景进行调节，并允许在安全框架下提供合规的法律信息，而不是一味压制。
    - 用户观察到助手有时会“写出回复但被免责声明覆盖”，这表明存在服务器端的 guardrail，当输出中途触发风险评分时，可以替换已经流式传输的答案。这种“先生成后删除”的流水线导致了明显的翻转（答案 → 拒绝），降低了付费用户的 UX，并使系统显得不一致。在架构上，预解码政策引导或跨度级（span-level）删除可以在保留合规内容的同时，减轻流式传输中途的覆盖问题。

- [**GPT5 Offering Additional Tasks Is The Most Annoying It's Ever Been**](https://www.reddit.com/r/ChatGPT/comments/1n7eqsw/gpt5_offering_additional_tasks_is_the_most/) ([Score: 338, Comments: 206](https://www.reddit.com/r/ChatGPT/comments/1n7eqsw/gpt5_offering_additional_tasks_is_the_most/)): **OP 报告称，在 ChatGPT/GPT‑5 的 App/桌面客户端中，助手会持续附加主动提议（例如** `Would you like me to <task>?`**），且极难抑制——即使在个性化/记忆（personalization/memory）中嵌入负面指令、使用 regex 风格的约束、请求 Chain‑of‑Thought 意图以避免提议，以及采用迭代的 Prompt‑Engineering 策略后依然如此。措辞会不断变化（例如** `If you wish I could…`**），这表明存在一个强大的客户端级 System Prompt 或对齐模板（可能是由 RLHF 驱动的帮助性启发式算法；参见 [InstructGPT RLHF](https://arxiv.org/abs/2203.02155)），它会覆盖用户指令；OP 指出，这种情况仅限于 App/桌面客户端，而不存在于 API 工作流中（在 API 中 System Prompt 是明确可控的；参见 [Chat Completions "system" role](https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages)）。当被问及时，模型也承认其自身建议的预期效用（expected utility）较低，突显了“主动提供帮助”的先验概率（priors）与实际任务效用之间的失调。** 热门评论证实了这种抑制效果有限且短暂（“仅维持一两条消息”），并报告了类似的越权行为，即模型在进行简单的语法/流畅度检查时会未经要求重写文本，进一步证实了这种激进的“提供后续步骤”风格是一种持续且令人反感的行为。
    - 多位用户强调了一个 UX 问题，即 GPT 主动的“额外任务”提示只能被暂时抑制（通常仅限一条消息），这意味着没有持久的按用户或按线程（thread）的偏好标志来禁用这种主动性。他们要求提供一个全局退出开关或设置，使助手默认保持在严格的响应（reactive）模式。
    - 报告显示，意图分类器（intent classifier）在简单的校对请求中表现过度，执行了全文重写或提供了结构化 Artifacts（如表格/列表/图片），而不是最小限度的语法/流畅度修正。建议增加一种受限的“仅校对”模式，返回 Diffs 或行内建议（不重新格式化或扩展内容），以减少误报并保留作者原意。
    - 关键字触发的辅助流程（如订阅管理提示）在无关语境中触发，表明存在激进的启发式算法或操作建议的置信度阈值（confidence thresholds）过低。用户建议在启动专门流程前使用更高的置信度门控（confidence gating）或明确的选择性加入（opt-in），以减少侵入性的、偏离目标的协助。
- [**I was asking chat about why lying on my left side would help reflux, it offered to show me a diagram.**](https://i.redd.it/lkvqkm1uevmf1.png) ([Score: 274, Comments: 39](https://www.reddit.com/r/ChatGPT/comments/1n74gmh/i_was_asking_chat_about_why_lying_on_my_left_side/)): **OP 询问为什么向左侧睡可以减轻反流，AI 生成了一张对比左侧卧位与右侧卧位的图表。从技术上讲，左侧卧位（left lateral decubitus）倾向于使胃食管结合部（LES）保持在胃酸池（沿胃大弯的胃底）上方，利用重力和 His 角来减少逆流；右侧卧位则可能使 LES 处于相对于酸液的低位，增加反流风险。** 评论者对方向/标签开起了玩笑（例如建议翻转手机），暗示 AI 生成的图表可能是镜像的或画得很粗糙，但没有实质性的技术争议。

- [**【紧急】我女朋友在工作时使用了 ChatGPT。现在她的老板要求她解释计算过程。我认为这些计算是幻觉。该怎么办？**](https://www.reddit.com/r/ChatGPT/comments/1n78p0v/urgent_my_girlfriend_used_chatgpt_for_her_work/) ([Score: 8705, Comments: 3099](https://www.reddit.com/r/ChatGPT/comments/1n78p0v/urgent_my_girlfriend_used_chatgpt_for_her_work/)): **OP 描述了一个面向客户的调查分析，该分析是通过 ChatGPT 生成的，模型生成了一个 Excel 和一个生成的 PowerPoint；当被要求解释方法论时，ChatGPT 声称它对 5 级文本“感受”响应使用了 Pearson 相关系数。这指向了一个幻觉或无效的方法：Pearson’s r ([wiki](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)) 假设数值/区间数据和变量的显式编码——而这些都没有被记录——因此结果是不可复现且无法验证的，体现了 LLM “幻觉”风险 ([overview](https://en.wikipedia.org/wiki/Hallucination_(artificial_intelligence)))。** 评论者建议要么编造一个掩饰故事（例如，“占位数据”），或者更审慎地警告，客户可能会识别出 AI 生成的内容，并且歪曲方法比承认误用并透明地重新进行分析具有更高的道德和专业风险。
    - 数据隐私/合规风险：一位评论者指出，如果任何客户数据或 PII 被粘贴到 ChatGPT 中，这可能违反公司政策、NDA 或法规（如 GDPR/CCPA），并且比糟糕的分析更严重。除非使用企业级控制，否则 ChatGPT 消费者端的输入可能会被保留/用于改进服务；相比之下，API/Enterprise 模式提供更严格的数据处理（不对输入进行训练，可选零保留）——参见 OpenAI 的数据政策：https://openai.com/policies/api-data-usage 和数据控制 FAQ：https://help.openai.com/en/articles/7730893-data-controls-faq。组织通常需要经过批准的供应商和 DPA；将敏感数据上传到未经批准的第三方可能会触发事件报告和取证。眼下的步骤是评估是否共享了任何敏感字段，并根据政策进行上报。
    - 可复现性/问责制：客户要求“解释计算过程”表明了对来源和可复现性的担忧；LLM 可以产生看似合理但错误的定量输出（幻觉数字），并且无法提供可验证的审计追踪。歪曲来源（“占位数据”）是有风险的；一种合理的做法是使用透明的方法（电子表格/代码）重建分析，并记录输入、公式和中间结果。展望未来，可以使用 LLM 来起草公式或代码，但要使用确定性工具验证所有数字，保留过程文件以便按需复现工作。承认缺乏适当的 AI 使用可能会给人留下不好的印象，但从技术和道德的角度来看，在没有可复现基础的情况下变本加厉会更糟。
- [**“在他们身上浇了橄榄油”**](https://i.redd.it/ama9gb826umf1.jpeg) ([Score: 242, Comments: 71](https://www.reddit.com/r/ChatGPT/comments/1n6z3je/poured_olive_oil_on_them/)): **一个 meme 展示了用户如何通过使用水果代码委婉语（例如** `banana`**,** `peach`**）替换被禁止的历史人物/事件（暗示 Adolf Hitler 和 Eva Braun）来规避严格的关键词/词法 Guardrails，从而在绕过过滤器的同时有效地保留了含义。它说明了对抗性的内容混淆/Prompt-coding 可以击败幼稚的字符串匹配，并强调了需要语义化、上下文感知的审核，而不是脆弱的黑名单。[图片链接](https://i.redd.it/ama9gb826umf1.jpeg)。** 热门评论认为，严格的 Guardrails “行不通”，因为人们会创造性地重新表述内容，其他人发布了变体示例（“Banana and Eva Banana”），展示了这种混淆是多么容易。
    - Guardrails 被描述为脆弱的：严格的、基于关键词/模式的安全过滤器很容易被创造性的 Prompting（改写、间接引导、混淆）绕过。这一点意味着鲁棒性需要意图感知的审核层、对抗性 Red-teaming 以及针对 Jailbreak 韧性的持续评估，而不是静态黑名单（例如，参见 **Anthropic** 关于 Red-teaming 的说明：https://www.anthropic.com/news/red-teaming-language-models）。
    - 一位用户报告说，模型拒绝回答关于希特勒死亡的中性事实查询，突显了校准不良的安全分类器导致的过度阻断/误报。从技术上讲，这表明需要上下文敏感的策略路由（例如，区分历史/教育意图）、校准阈值以及良性事实的白名单，通过在标记的安全数据集上的精确率/召回率以及对已知安全查询的抽查来进行衡量。

### 3. AI 视频/图像编辑工作流与展示：nano banana, Wan 2.2, Qwen, Local SD

- [**连续性编辑实验 | Wan 2.2 + InfiniteTalk + Qwen Image Edit**](https://v.redd.it/sgf6cc51rymf1) ([Score: 411, Comments: 59](https://www.reddit.com/r/StableDiffusion/comments/1n7h56l/experimenting_with_continuity_edits_wan_22/)): **这是一部 AI 科幻电影实验的第 3 集，使用带有 CausVid LoRAs (Wan 2.1) 的 Wan 2.2 工作流推进了连续性和对话表现。作者指出，对口型对话非常耗费计算资源（即使在** `RTX 5090` **上也是如此）且非常脆弱——微小的瑕疵通常会导致必须全量重新生成，因此应尽量减少对话镜头。创作者报告称，在语音转视频（speech-to-video）方面，InfiniteTalk 优于 Wan S2V，表现力更强且更符合提示词，并分享了用于多人物和单人物镜头的自动帧工作流（[paste 1](https://pastebin.com/N2qNmrh5), [paste 2](https://pastebin.com/BdgfR4kg)）；在空间连续性方面，Qwen-Image-Edit 可以从单帧合成备选摄像机角度，尽管失败率较高，这暗示可能需要一个 LoRA 来保持一致性。之前的剧集和产出已发布在 YouTube 频道：[youtube.com/@Stellarchive](http://www.youtube.com/@Stellarchive)。** 热门反馈：可以看到轻微的动作伪影（手部）；一位评论者纠正了名称为 **Qwen-Image-Edit**（而非 “Wan Image Edit”）；除此之外，反响积极，几乎没有额外的技术批评。
    - 一位观众注意到主体手部在运动过程中出现了 `1–2` 处伪影，这暗示了连续性编辑中存在轻微的时间一致性问题。这是在视频上应用逐帧图像编辑（例如在 **Wan 2.2** 生成的帧上使用 Qwen Image Edit）时常见的失效模式，移动的肢体末端和遮挡可能会产生抖动或涂抹感。
    - 工具说明：引用的图像编辑模型是 **Qwen-Image-Edit**，而非 "Wan Image Edit"。这与标题中的工作流一致（**Wan 2.2** 用于生成，**InfiniteTalk** 用于语音/对口型，**Qwen-Image-Edit** 用于帧编辑）。
    - 建议尝试用于 Qwen 图像编辑的场景内 LoRA：[flymy-ai/qwen-image-edit-inscene-lora](https://huggingface.co/flymy-ai/qwen-image-edit-inscene-lora)。场景内 LoRA 旨在编辑局部元素的同时保留场景布局/光照，这可能会减少运动区域的伪影。
- [**我让 nano banana 带我进入我最喜欢的游戏厅**](https://v.redd.it/bkqfaae3c0nf1) ([Score: 276, Comments: 33](https://www.reddit.com/r/aivideo/comments/1n7pmkx/i_asked_nano_banana_to_get_me_into_my_favorite/)): **创作者展示了一个 AI 辅助合成工作流：使用 nano banana 对真实的起始静止帧进行编辑（图像清理/插入），然后通过 Kling** `2.1` **使用起始/结束帧约束进行动画化以插值动作，音乐由 Producer AI 生成，最终在 DaVinci Resolve 中进行剪辑和调色。该帖子的 [X thread](https://x.com/techhalla/status/1963333488217919668) 中提供了分步教程。** 热门评论大多是非技术的赞美，称该作品在创意上“树立了标杆”；没有讨论实质性的技术批评或基准测试。
- [**是否可以在本地实现这个功能？**](https://i.redd.it/w562k57baxmf1.png) ([Score: 362, Comments: 70](https://www.reddit.com/r/StableDiffusion/comments/1n7ando/is_it_possible_to_do_this_locally/)): **楼主询问是否可以使用 Stable Diffusion 在本地从单张插图生成角色的多个一致姿势（如 X 上使用 “Nano Banana” 和 Google 的 Gemini 所展示的那样）。评论者表示这是可行的，但并非开箱即用：目前像 Nano Banana 这样的封闭/托管工具因其卓越的身份/属性一致性而受到称赞，而开源选项（如 Kontext, Qwen Image Edit）可能实现类似的工作流，并可能结合 LoRA 训练来锁定风格/身份。** 热门回复认为这是可能的，但需要人工努力并容忍轻微的不一致；其他人建议尝试 **Qwen Image Edit**，并期待开源社区通过在更强大模型的输出上训练 LoRA 来实现快速追赶。
    - 共识是 “Nano Banana” 目前在视觉变体的身份/属性一致性方面处于领先地位（接近“绝对”的角色保留），但它是封闭的。一些人建议通过 LoRA 适配器将其行为蒸馏到开源模型中，从而在本地进行复制——即在精选输出上训练角色/概念 LoRA，然后在 Qwen Image Edit（见 Qwen 仓库：https://github.com/QwenLM）等开源骨干模型上运行，以在无需云端推理的情况下获得类似的一致性。这实现了从仅靠提示词控制向参数高效微调（LoRA: https://arxiv.org/abs/2106.09685）的转变。

- 一个具体的本地流水线：(1) 从精心策划的数据集中训练角色 LoRA；(2) 使用 ComfyUI 的节点图 (https://github.com/comfyanonymous/ComfyUI) 配合 ControlNet 姿态调节来锁定每一镜头的结构。使用 OpenPose/Posenet 控制 (ControlNet: https://github.com/lllyasviel/ControlNet; ComfyUI 控制辅助工具: https://github.com/Fannovel16/comfyui_controlnet_aux) 可以保留骨架/布局，而 LoRA 则保留身份/配饰，从而减少细节漂移（例如：纹身、牙套）。这种方法牺牲了易用性以换取可复现性——每个姿态通常需要单独的控制处理。
    - 可行性说明：“使用 Qwen 图像编辑有一定可能性”，但要达到闭源模型级别的连贯性，通常需要提示词之外的监督引导。预计需要结合 LoRA + 逐帧姿态控制；仅靠提示词的工作流在微小且持续的细节（颜色匹配的配饰、Logo）上经常失败。这在本地是可行的，但需要计划进行数据集准备、LoRA 训练和逐姿态调节，而不是靠单次提示词。
- [**这在本地存在吗？实时替换 / 重绘？**](https://v.redd.it/0f9walhtiumf1) ([Score: 348, Comments: 72](https://www.reddit.com/r/StableDiffusion/comments/1n70q2o/does_this_exist_locally_realtime_replacement/)): **楼主询问是否存在本地、实时的面部替换/重绘。热门回复指出，目前没有可行的实时 “VACE + Motion” 流水线；可信的演示都是离线的。DeepFaceLab 在经过大量预训练后可以进行有限的“实时”处理，但质量很差（仅限正脸偏差，转头时有伪影）且不可信；高质量的 Deepfakes 仍需要离线生成。一位评论者确认展示的视频是他们制作的，使用了 “nano banana + Runway Act 2”，并确认这不是实时的，同时链接了来源 ([Instagram](https://www.instagram.com/p/DN1aEuQUD2e/))。** 共识：目前在设备端实现具有良好多角度保真度的即时换脸/重绘是不可行的；暗示可以实现的社交媒体视频多为骗流量的诱导内容。另一位用户指出，所发视频的帧率/长宽比表明是预录制的摄像机素材，而非实时处理。
    - 多位评论者指出，目前没有可信的实时 “VACE + Motion” 换脸/重绘流水线可用；暗示可以实现的视频很可能是骗流量的。虽然 **DeepFaceLab** 在大量预训练后可以“实时”运行，但评论者反映其保真度较差（仅在正脸镜头下可信），且在转头时有明显的伪影，这进一步证明高质量的多角度换脸仍需要离线生成时间，而非即时推理。
    - 原作者澄清展示的视频并非实时，并概述了流水线为 **nano banana** + **Runway Act 2**，更多细节见来源帖子：https://www.instagram.com/p/DN1aEuQUD2e/。这意味着这是一个精心策划的离线工作流，利用了 Runway 的生成工具，而非直播、设备端的重绘/换脸系统。
    - 另一个观察点指出，该视频的帧率和长宽比类似于录制的摄像机素材而非实时输出，进一步表明是非实时处理。这与作者的明确说明相符：*"它不是实时的"*。
- [**我让 nano banana 带我进入我最喜欢的游戏厅**](https://v.redd.it/bkqfaae3c0nf1) ([Score: 276, Comments: 33](https://www.reddit.com/r/aivideo/comments/1n7pmkx/i_asked_nano_banana_to_get_me_into_my_favorite/)): **展示了一个 AI 视频流水线：一张真实的底图通过 “nano banana” 进行编辑（图像编辑），然后使用 Kling 2.1 的首尾帧模式进行动画化，在关键帧之间插值运动；音频由 “producer AI” 生成，最后的剪辑/调色在 DaVinci Resolve 中完成。X/Twitter 上提供了分步指南：https://x.com/techhalla/status/1963333488217919668。** 热门评论大多是非技术性的赞美（例如称其为“史诗级”），没有实质性的评论或基准测试细节。
- [**伙计们，让我们穿越回去吧**](https://v.redd.it/pz6ia9umdzmf1) ([Score: 439, Comments: 155](https://www.reddit.com/r/aivideo/comments/1n7kf6h/guys_lets_just_travel_back/)): **楼主分享了一张可能是 AI 生成的 1980 年代复古风格图像，标题为“伙计们，让我们穿越回去吧”，可以通过预览图 (https://preview.redd.it/0mzhs3zegzmf1.png?width=183&format=png&auto=webp&s=290e05f3a160b3548e1b1be76b7d558b1cba0d15) 和原始 [v.redd.it](http://v.redd.it/) 链接 (https://v.redd.it/pz6ia9umdzmf1) 查看，后者在未经身份验证的情况下返回** `403`**。热门评论指出了其中的时代错误——*“用来自* `2025` *年的 AI 制作”*——并含蓄地区分了审美重建与行为模拟（例如不用手机）作为“回去”的不同方式。** 轻微的争论集中在真实性上：AI 生成的复古艺术是否破坏了“回到”某个时代的初衷，还是说采用低科技习惯来模拟这种体验更有意义。

- 评论者指出该图像是 AI 生成的（*“由 2025 年的 AI 制作”*），而非真实的 1980 年代媒体，这解释了场景中风格上的时代错误。他们还注意到，与那个时期的照片相比，画面主体呈现出不切实际的理想化外观，这与当前 Diffusion 模型偏向平滑、符合传统审美的人脸以及现代妆造/发型特征的偏见相吻合。参考图片：https://preview.redd.it/0mzhs3zegzmf1.png?width=183&format=png&auto=webp&s=290e05f3a160b3548e1b1be76b7d558b1cba0d15
- [**Guys lets just travel back**](https://v.redd.it/pz6ia9umdzmf1) ([Score: 438, Comments: 157](https://www.reddit.com/r/aivideo/comments/1n7kf6h/guys_lets_just_travel_back/)): **这篇名为“Guys lets just travel back”的怀旧帖子包含一张 80 年代主题的图像（根据评论，很可能是 AI 生成的）[预览](https://preview.redd.it/0mzhs3zegzmf1.png?width=183&format=png&auto=webp&s=290e05f3a160b3548e1b1be76b7d558b1cba0d15)。链接的视频端点 [v.redd.it/pz6ia9umdzmf1](https://v.redd.it/pz6ia9umdzmf1) 在 Reddit 的反爬虫控制下返回** `HTTP 403 Forbidden`**，这意味着需要身份验证或有效的客户端令牌（例如 [login](https://www.reddit.com/login)）。** 热门评论指出该图像看起来是 AI 生成的（“由 2025 年的 AI 制作”），并围绕 80 年代怀旧主题展开讨论；其中一条评论建议采取行为上的“复古”选择（例如，不带手机去商场），而非寻求任何技术解决方案。
    - 评论者指出该图像是 AI 生成的（例如，“这是由 2025 年的 AI 制作的”），并指出它与真实的 1980 年代视觉效果不符（“我记得 80 年代，不是这样的”）。现代 Diffusion 输出通常过度润色——光滑的皮肤、类 HDR 的对比度、近乎对称——并且忽略了那个时代的伪影，如胶片颗粒/光晕（halation）、色差（chromatic aberration）、镜头暗角（lens vignetting）以及特定时代的色彩科学。为了更接近 80 年代的真实度，从业者通常会添加显式约束或后期处理步骤（模拟噪声、模拟 Kodachrome/Ektachrome 的色彩 LUT、轻微的色度溢出、画幅抖动、CRT/扫描线模拟）。
    - “那时候其实没人长得那么好看”这一言论对应了模型/数据偏见：网络规模的训练语料库（充斥着网红/修图后的图像）将 Diffusion 先验推向了理想化的美感以及现代的妆造。如果没有特定时代的 Fine-tunes/LoRAs 和强力的 Negative Prompts，采样器在生成复古场景时会倾向于当前的审美标准，从而产生时代错误的“完美”面孔。
- [**Fruit Beds 🍉🛌🏻↔️**](https://v.redd.it/4vuxgegxwumf1) ([Score: 269, Comments: 40](https://www.reddit.com/r/aivideo/comments/1n72elu/fruit_beds/)): **帖子“Fruit Beds 🍉🛌🏻↔️”似乎是一个托管在 Reddit [v.redd.it](http://v.redd.it/) 上的短视频（[链接](https://v.redd.it/4vuxgegxwumf1)），目前在未经身份验证的情况下返回** `HTTP 403 Forbidden`**；Reddit 的网络安全页面显示访问需要登录或使用 API 凭据。可以通过 PNG 链接查看静态/预览帧（[预览](https://preview.redd.it/jyuowx16bxmf1.png?width=1440&format=png&auto=webp&s=c037d679daadd01f03561822b5bc2646ead5f6a5)），暗示了一系列水果主题的“床”，但帖子内未提供技术背景或元数据。** 热门评论多为非技术性的：一个反应 GIF 和一个问题——“最后一个到底是什么？”——凸显了最终视觉效果的模糊性；目前尚未提供明确的答案或解释。
    - 两位评论者提供了更高分辨率的静态图来回答关于模糊的“最后一个”的问题，并链接了从帖子中截取的帧：[图片 1](https://preview.redd.it/jyuowx16bxmf1.png?width=1440&format=png&auto=webp&s=c037d679daadd01f03561822b5bc2646ead5f6a5) 和 [图片 2](https://preview.redd.it/dp6kr7ddovmf1.png?width=993&format=png&auto=webp&s=8108a005592247c151f97aaa1ad1e0bfff909e29)。这些高分辨率帧有助于消除在 GIF/WebP 播放分辨率下或因压缩伪影而模糊的细节歧义。
    - 关于毯子“凭空产生”的观察可能源于循环/编码不连续性：GIF/WebP 动画通常依赖于帧间增量（inter-frame deltas）和处理方法（Disposal Methods，如 `restore to background` 或 `restore to previous`）。如果循环点切在非关键帧之间，或者转码器（例如 Reddit 的 GIF→MP4/WebP 流水线）丢弃/合并了帧，物体可能会在循环之间突然出现或消失；参见此处关于 GIF 处理行为的解释：https://en.wikipedia.org/wiki/GIF#Disposal_methods。

- [**水果床 🍉🛌🏻↔️**](https://v.redd.it/4vuxgegxwumf1) ([Score: 265, Comments: 40](https://www.reddit.com/r/aivideo/comments/1n72elu/fruit_beds/)): **这是一个标题为 “Fruit Beds” 的图片/meme 帖子，展示了一系列以水果为主题的床铺图像；没有技术内容（代码、模型或基准测试）。原始 Reddit URL 被屏蔽，显示 HTTP 403 “Forbidden” 页面，需要 [Reddit login](https://www.reddit.com/login/) 或开发者令牌；提供了一个 [support form](https://support.reddithelp.com/hc/en-us/requests/new?ticket_form_id=21879292693140)。评论中引用了[最后一张图片的直接预览](https://preview.redd.it/jyuowx16bxmf1.png?width=1440&format=png&auto=webp&s=c037d679daadd01f03561822b5bc2646ead5f6a5)。** 热门评论是非技术性的：一个 GIF 反应，以及一个问题——“最后一张到底是什么？”——突显了对最后一张图片的模糊感；另一个链接了上面的预览图。
- [**我不知道**](https://i.redd.it/9pbqemdwdzmf1.jpeg) ([Score: 858, Comments: 39](https://www.reddit.com/r/ChatGPT/comments/1n7kg9g/i_dont_know/)): **这是一个对比两个时代的 meme 格式，旨在突出外行对复杂系统的无知：现代人无法解释计算机是如何工作的，而古代法老也无法解释金字塔是如何建造的。没有技术细节、基准测试或实现讨论——纯粹是关于创造者/使用者与对底层技术或建造方法的深度理解之间差距的幽默评论。** 评论大多是笑话；一个略带哲学意味的提问询问语言是如何运作的，另一个则指出时间旅行者提问的古怪之处，但没有实质性的技术辩论。
    - 一位评论者对比了单个专家复制古代建筑（如金字塔）的可行性与在没有庞大、分布式的知识库和工具链的情况下复制现代设备的不可行性。这强调了从以物流和劳动力为主导的项目向具有极端专业化的精密制造的转变：现代 SoC 集成了 `~10–20B` 个晶体管，并依赖于 **EUV lithography** 和全球供应链（例如 ASML EUV: https://www.asml.com/en/technology/lithography-principles/euv-lithography；工艺概述: https://en.wikipedia.org/wiki/Semiconductor_device_fabrication）。即使拥有完整的原理图，复制也会受到材料科学、计量学和资本设备（无尘室、光刻机）的限制，说明了模块化但脆弱的复杂性与单体、稳健的建筑之间的对比。

---

# AI Discord 回顾

> 由 gpt-5 生成的摘要的摘要的摘要
> 

**1. 推理基准测试与开源模型**

- **Pokerbots Pit Stop: Husky Bench 桂冠摘得者 Sonnet**: **Husky Hold’em Bench** 推出了首个开源扑克机器人评测，其中 **Claude 4 Sonnet** 在 **5k+ 场游戏** 的 6 人循环赛中以 **57.9%** 的平均利润领先，**Opus (31.9%)** 和 **Gemini (31.0%)** 紧随其后，详情记录在 [Husky Hold’em Bench](http://huskybench.com/)，并由 [Nous Research](https://x.com/NousResearch/status/1963371292318749043) 记录。
    - 社区称赞了该基准测试的约束条件（在时间/内存限制下的 Python 策略），并称其为 *“首个开源扑克机器人评测”*，预计评测工具和 Agent 策略将快速迭代 ([huskybench.com](http://huskybench.com/))。
- **Hermes 4 升温：开源模型排行榜实力展示**: **Hermes 4**（基于 **Qwen3‑14B** 构建）首次亮相，采用了全新合成的训练后语料库，强调经过验证的推理轨迹和更大的规模（**~5M 样本 / ~60B tokens**），而 **Hermes‑4‑405B** 目前在 Husky 的开源模型排名中位居榜首，回撤率为 **−12.41%**，详见 [Nous Research 更新](https://x.com/NousResearch/status/1963371292318749043)。
    - 用户分享了实用的微调技巧（例如用于 Think 与 Instruct 模式的 SillyTavern 采样器设置），并报告了在格式忠实输出方面更强的数学/代码/逻辑性能，称 Hermes 4 的混合推理为 *“具有中性对齐的显式思考片段”* ([huskybench.com](http://huskybench.com/))。
- **棋盘游戏基准测试扩展至扑克之外**: 除了扑克，工程师们还通过 [TextArena Leaderboard](https://www.textarena.ai/leaderboard) 对比了 LLM 在经典棋盘游戏上的表现，强调国际象棋/围棋/四子棋/将棋/象棋的 ELO 等级分是领域特定评测的补充信号。
    - 成员们提倡使用多任务评测套件以避免对单一领域的过拟合，指出 *“多样化、严谨的游戏评测”* 能更好地暴露模型的弱点和策略的脆弱性 ([TextArena Leaderboard](https://www.textarena.ai/leaderboard))。

**2. Kernel 功夫与低比特训练**

- **Metal Mania：AI 生成的 Kernel 提升 1.87 倍**：一个团队报告称，通过直接从 **PyTorch** 生成低级 **Metal kernels**，实现了 **1.87 倍的加速**，详见 [AI-generated Metal kernels](https://gimletlabs.ai/blog/ai-generated-metal-kernels)，并指出 **torch.mps.compile_shader** 可以直接调用 kernel 而无需 C++ 绑定。
    - 工程师们要求提供 kernel 转储（dumps），并建议提交 PR 将这些优化合并到 **PyTorch** 上游；同时一位维护者评论道 *“不再需要 cpp 绑定”*，并标记了使用 [BackendBench](https://github.com/meta-pytorch/BackendBench) 进行的正确性检查（见博客：[gimletlabs.ai](https://gimletlabs.ai/blog/ai-generated-metal-kernels)）。
- **TorchAO Tango：Nightly 版本报错，MXFP8 提速**：开发者遇到了 **torchao** nightly 版本的损坏，原因是 **Torch 2.9 与 2.8** 的不匹配（[issue #2919](https://github.com/pytorch/ao/issues/2919)），通过 `pip install torchao==0.13.0 --extra-index-url https://download.pytorch.org/whl/test/cu128` 修复；同时 **PR #2933** 为 MXFP8 补丁了 `sm100` 标志（[PR #2933](https://github.com/pytorch/ao/pull/2933)）；与此同时，**MXFP8** 预训练方案和高达 **1.28 倍** 的加速效果已发布（[Recipes for Pre-training LLMs with MXFP8](https://arxiv.org/abs/2506.08027), [PyTorch blog](https://pytorch.org/blog/accelerating-2k-scale-pre-training-up-to-1-28x-with-torchao-mxfp8-and-torchtitan-on-crusoe-b200-cluster/)）。
    - 一位用户遇到了 ImportError——*“cannot import name 'mxfp8_cuda'”*——但维护者澄清说，短期修复方案已解除了 **NVFP4 推理** 的阻塞，受影响的 kernel 仅用于 **MXFP8 训练**（[issue #2932](https://github.com/pytorch/ao/issues/2932), [PR #2933](https://github.com/pytorch/ao/pull/2933)）。
- **Fusion Confusion：torch.compile 与 Triton 的碰撞**：工程师们确认 **torch.compile** 不会将算子融合（fuse）进用户定义的 **Triton** kernels 中，并且经常在专用算子周围产生融合屏障；相关的复现（repro）和讨论见此 [fusion gist](https://gist.github.com/tohskai/0d0579ef2371dc5a0562d57a7c5361ea)。
    - 他们建议通过 `TORCH_LOGS="output_code"` 检查捕获的图（captured graphs），并警告说示例 kernel *“在大 MNK 情况下数值不稳定”*，因此手动融合仍然是务实的选择（[fusion gist](https://gist.github.com/tohskai/0d0579ef2371dc5a0562d57a7c5361ea)）。

**3. Agentic Patterns, Program Synthesis, and Eval Infra**

- **设计宝典发布：400 页 Agentic Patterns**：一位 Google 工程师发布了一份 **400 页** 的 **Agentic Design Patterns** 草案，涵盖了高级提示词工程、多 Agent 系统、工具调用和 **MCP**，可在 [Agentic Design Patterns (Google Doc)](https://docs.google.com/document/d/1rsaK53T3Lg5KoGwvf8ukOUvbELRtH-V0LnOIFDxBryE) 查看，并配有 [NotebookLM](https://notebooklm.google.com/notebook/44bc8819-958d-4050-8431-e7efe2dbd16e) 辅助。
    - 读者指出 *“编辑权限未关闭”* 并担心误删，而其他人则预订了 Springer 版本，并开始将这些模式提取到自己的实战指南中（[Agentic Design Patterns](https://docs.google.com/document/d/1rsaK53T3Lg5KoGwvf8ukOUvbELRtH-V0LnOIFDxBryE)）。
- **DSPy 揭秘：清晰的数据划分与 MLflow 钩子**：**DSPy** 澄清了其 **train/val/dev/test** 划分规范（val 用于多步图表，test 用于最终评估）以避免数据泄露，成员们还探索了通过 [MLflow Prompt Registry](https://mlflow.org/docs/latest/genai/prompt-registry/manage-prompt-lifecycles-with-aliases/) 和 [mlflow.dspy](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.dspy.html) 进行提示词生命周期追踪。
    - 一个共享的 [Context Compression Prompt Experiments](https://github.com/Laurian/context-compression-experiments-2508?tab=readme-ov-file#context-compression-prompt-experiments) 仓库旨在提高 `dspy.Image` 在不同供应商之间的可靠性，志愿者们提交了失败案例和补丁。
- **转向控制变得简单：LM Eval Harness 已内置支持**：**LM Eval harness** 已经支持受控模型（steered models）——激活/残差转向向量和格式化已在 [Steered HF Transformers Models](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/README.md#steered-hugging-face-transformers-models) 中记录。
    - 贡献者指出了 `SteeredModel` 的 docstring 以获取详情，并开启了一个 PR 来控制单个注意力头，称 *“不要自己造轮子——使用内置功能”*（[SteeredModel docstring](https://github.com/EleutherAI/lm-evaluation-harness/blob/2d7cb5c31cffd3cbeb5367542ab8f4c23f4b77f4/lm_eval/models/hf_steered.py#L206), [PR #3279](https://github.com/EleutherAI/lm-evaluation-harness/pull/3279)）。

**4. Student and Builder Tools Shipping**

- **Comet 课堂：Perplexity 发布学习模式 (Study Mode)**：**Perplexity Comet** 为学生推出了专注于日程安排、教科书和备考的 **Study Mode**，并在 [Comet for Students](http://pplx.ai/student) 中展示了交互式抽认卡 (flashcards)。
    - 资深用户请求 Perplexity 将 Study Mode 纳入 **Pro** 版，因为它不仅仅是一个 "System Prompt"，还包含自定义 GUI 元素 ([公告线程](https://discord.com/channels/1047197230748151888/1047204950763122820/1412869217950367754))。
- **全民项目制：ChatGPT 惠及免费用户**：ChatGPT 中的 **Projects** 现已面向 Web 和 Android 端的 **Free** 用户开放（iOS 版正在推出），每个项目的文档限制分别为：**5 个 (Free)**、**25 个 (Plus)** 以及 **40 个 (Pro/Business/Enterprise)** ([OpenAI 公告](https://discord.com/channels/974519864045756446/977259063052234752/1412901459783057440))。
    - 用户可以自定义**颜色/图标**，并切换**仅限项目的记忆控制 (project‑only memory controls)** 以实现更严格的上下文隔离，团队称这对于可重复的工作流至关重要 ([OpenAI 公告](https://discord.com/channels/974519864045756446/977259063052234752/1412901459783057440))。
- **Kimi 编程：代金券门槛、新模型与幻灯片**：**Moonshot (Kimi)** 正在发放 **20 张价值 20 美元的 API 代金券**，用于测试一款新的编程增强模型，并展示了一个流畅的**幻灯片生成 (slide generation)** 功能 ([抽奖频道](https://discord.com/channels/1369594130807787570/1412714402284703795))。
    - 管理员警告要提防冒充者——“如果不是黄色的（官方标识），就不要相信”——用户则要求提供 **K2 turbo Coder Pro** 方案或统一的会员层级 ([抽奖频道](https://discord.com/channels/1369594130807787570/1412714402284703795))。

**5. 搜索与深度研究：快速、廉价且资金充足**

- **Exa 加速：8500 万美元 B 轮融资，估值 7 亿**：**Exa** 在由 **Benchmark** 领投的 **B 轮融资**中筹集了 **8500 万美元**，估值达到 **7 亿美元**，其定位是 **AI 搜索引擎**，公告见此：[Exa Series B](https://x.com/ExaAILabs/status/1963262700123000947)。
    - 交易追踪者指出 Harmonic 提前两周标记了这一轮融资，这激发了将**交易流预警 (deal‑flow alerts)** 产品化的想法 ([Exa 公告](https://x.com/ExaAILabs/status/1963262700123000947))。
- **廉价的深度研究：Qwen-2.5 14B > Sonnet-4**：Kyle Corbitt 分享了一个微调 **Qwen‑2.5 14B** 的开源方案，通过 SFT + GRPO + eval，在约 **30 个 H200 小时（约 350 美元）**内实现的模型在 **DeepResearch** 基准测试中击败了 **Sonnet‑4** ([训练线程](https://xcancel.com/corbtt/status/1962954306078048297))。
    - 所得模型在测试中与 **Gemini 2.5 Pro**、**OpenAI Deep Research** 和 **Claude Research** 具有竞争力，开发者称赞其性价比为“极低成本即可投产” ([训练线程](https://xcancel.com/corbtt/status/1962954306078048297))。
- **指令编程：Claude Code 的 “AI DOS” 时刻**：Nikunj Kothari 认为 **Claude Code** 降低了门槛，就像 **DOS** 对 PC 所做的那样——让非编程人员能够“凭想象力构建软件”——详见[此线程](https://x.com/nikunj/status/1963007529082093815)的辩论。
    - 评论者对于我们是“仍处于命令行时代”还是进入了“受想象力约束的阶段”持不同意见，创意人士则关注能够缩短原型周期的工作流 ([讨论](https://x.com/nikunj/status/1963007529082093815))。


---

# Discord：高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Comet 登陆学生桌面**：Perplexity AI 现向学生提供 **Comet**，通过全新的 **Study Mode** 协助处理日程、教科书和备考，详见[此公告](http://pplx.ai/student)。
   - 发布活动包含一段[视频演示](https://cdn.discordapp.com/attachments/1047204950763122820/1412869216989610165/comet-students-flashcard.mp4?ex=68b9dc7f&is=68b88aff&hm=9ee0ee1f4d6ebd93c0dc45d35c9ff3f80b3f6d98370cca5474cc1900996b426a&)，展示了如何在 **Comet** 中使用抽认卡（flashcards）进行更具互动性和高效的学习。
- **Pro 用户要求提供 Study Mode**：一些用户正敦促 Perplexity AI 将目前仅限教育账户使用的 **Study Mode** 功能扩展到所有 **Pro 用户**。
   - 该功能不仅仅是一个 **system prompt**，一些用户指出它还包含相关的 **GUI** 元素。
- **ChatGPT5 Pro 引发笑料**：一名用户提到了 **ChatGPT5 Pro**，将其误认为是 Perplexity 的 **GPT5** 和 **GPT5 Thinking** 模型，引发了困惑和笑声。
   - 另一名用户澄清说 **ChatGPT5 Pro** 是 chatgpt.com 专属的，引发了其他成员的幽默反应。
- **Comet 助手失误**：用户报告了 **Comet** 的问题，包括简单网站的加载时间过长以及助手表现不达标。
   - 有推测认为该助手可能正在利用 **Sonar**。
- **Perplexity 的过滤器过度标记**：用户正在批评 **Perplexity** 过度的审查制度，甚至连“希特勒是怎么死的？”这类良性的历史查询也被标记。
   - 用户担心过于严格的过滤系统可能会导致因研究历史或其他无害主题而遭到无理的封号。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Gemini 3 的炒作：是否过热？**：成员们讨论了围绕 **Gemini 3** 的炒作，观点从预期可能被夸大到 Google 可能会给行业带来惊喜（即使只是险胜竞争对手），特别是考虑到 OpenAI 的 **ChatGPT5** 推迟了。
   - 一名成员发布了一个[耸肩的 gif](https://tenor.com/view/shrug-what-huh-will-smith-i-mean-gif-3535627793955785136)，反映了 **Gemini 3** 真实影响的不确定性。
- **LM Arena 的登录系统焕然一新**：用户对 **LM Arena** 的新登录系统表示热烈欢迎，一位成员说：“太爱这个新登录系统了 ❤️ 期待已久 🔥”。
   - 该成员还提议建立一个基于 **Google Drive** 的聊天存储系统，用于将用户数据导出为文本文件并进行分析，但这一想法遭到了质疑。
- **LMArena 网站崩溃！**：LMArena 经历了严重的网站故障，导致用户沮丧并引发了大量提问。
   - 调解员 🍍 向用户保证团队正在积极解决问题，并引导他们查看 [FAQ](https://lmarena.ai/faq) 以获取更新。
- **MAI 1 Preview 神秘下线！**：成员们报告了 Microsoft 的 **LLM MAI 1 Preview** 突然出现故障，一些用户曾称赞该 **LLM** 效果出色。
   - 一位用户评论说 **MAI 1 Preview** 在 90% 的情况下给出了最好的答案——比所有其他模型都好，甚至超过了 **ChatGPT-5 high**，这让社区对其突然消失感到困惑。
- **Cloudflare 让用户陷入验证循环**：用户抱怨 **LMArena** 上频繁出现 **Cloudflare** 人机验证挑战，一位用户问道：“是不是每个人在 lmarena 网站上每两分钟就会遇到一次 cloudflair 人机验证？？”。
   - 虽然怀疑 **VPN** 使用是原因之一，但该问题也出现在其他使用 **Cloudflare** 的网站上，导致用户普遍感到恼火。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT 为免费用户送礼**：**ChatGPT 中的 Projects** 现已向 Web 和 Android 的 **Free 用户**开放，iOS 版也将很快推出。同时增加了文件上传限制，现在 **Free 用户为 5 个**，**Plus 用户为 25 个**，**Pro/Business/Enterprise 用户为 40 个**。
   - 用户现在可以为项目自定义**颜色和图标**，并且可以使用仅限项目的记忆控制功能，以获得更量身定制的上下文。
- **Nano Banana 引爆 Google Gemini**：成员们分享了在 **Gemini 应用**和 **Google Studio** 中使用提示词 **nano banana (gemini-2.5-flash-image-preview)** 生成的图像。
   - 一位用户展示了他们如何将同事变成了维京人。
- **成员对 Anti-Prompt GPT 持怀疑态度**：一位成员分享了他们的 [Anti-Prompt Breaching GPT](https://chatgpt.com/g/g-68b8279fca7081919999821ccbd0dc7e-anti-prompt-breaching/)，旨在防止提示词泄露（prompt leaking）。
   - 其他人表示怀疑，认为这可能只会减慢速度或增加绕过保护的难度，特别是在 Custom GPTs 中，且可靠性会受到影响。
- **Cognitive Mesh AI 实现自主学习**：一位成员描述了设计一种能够自我适应和自我学习的认知网格 AI，随着时间的推移不断增长其理解力，类似于利用 **MoEs**。
   - 该 AI 历时 **3 年**构建，拥有短期、长期和反思性记忆，按其自身的轨迹进化，并根据输入发展出了指令性响应。
- **跳出 Transformer 的框架思考**：成员们辩论了 **Liquid Neural Networks** 以及将连续时间动力学与符号推理相结合的架构，如**神经形态芯片（neuromorphic chips）、光子学（photonics）和自旋电子学（spintronics）**。
   - 共识是，这些创新不依赖于暴力规模扩张，而是依赖于*对基础的重新思考*。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Anthropic API 密钥通过测试**：用户已确认 **sk-ant-** 密钥在 Cursor 的 **Anthropic API** 中运行正常，尽管 UI 上存在差异。
   - 社区证实，即使 UI 可能显示异常，密钥仍能正常工作。
- **Cursor Auto 模型遭到用户质疑**：用户仔细审查了 **Cursor 的 Auto 模型**，一位用户报告在不到一周的时间内花费了 **$200**。
   - 反馈表明，该模型的代码质量不如 **Sonnet** 或 **Gemini**，但由于更好的提示词，可能优于 **Copilot**；不过也有人建议手动引导总结。
- **Cursor 更新擦除聊天记录引发恐慌**：多位用户报告因 Cursor 更新导致数据丢失，其中一位用户丢失了一个月的工作内容。
   - 检查本地聊天记录存储位置的建议解决方案对某些人无效，在常规目录中找不到聊天记录。
- **微调聊天机器人：省钱就是赚钱**：一位社区成员寻求关于为 Web 应用聊天机器人进行模型微调（fine-tuning）的建议，这引出了关于利用提示词生成器的建议。
   - 共识是在建立收入流之前推迟微调，将其视为一种优化策略。
- **后台 Agent 遭遇“中年危机”**：一位成员报告其后台 Agent 冻结，并寻求通过 API 将其当前状态转移到另一个聊天的方法。
   - 具体来说，他们请求一种 API 方法来获取状态转移摘要，以便于将 Agent 的当前状态迁移到新的聊天环境中。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Nano Banana 驱动 OpenRouter Discord 机器人**：一位成员利用 **Nano Banana** 通过 **OpenRouter** 创建了一个 [Discord 机器人](https://github.com/mojomast/gemini-nano-banana-discord-bot)。
   - 该用户澄清他们是 *vibe coded*（氛围编码）了这个机器人，源代码可在 [GitHub](https://github.com/mojomast/gemini-nano-banana-discord-bot) 上获得。
- **DeepSeek 模型胡言乱语**：一些用户报告免费的 **DeepSeek 模型** 正在生成乱码，而付费模型运行正常。
   - 另一位用户询问了 **DeepSeek 3.1** 的定价，提到 [Synthetic.new](https://synthetic.new/?referral=WofDwy6qyYEKlTi) 为每月 20 美元，尽管官方费率被认为是“宰客”。
- **Agent Framework SDK 获得补丁**：成员们讨论了由于非标准 Schema，使用 OpenRouter 修补 **Agent Framework SDK**（如 OpenAI Agent SDK、AI SDK 和 LangChain JS）的问题。
   - 一位成员计划通过集成 **BAML** 来“手搓”自己的解决方案，并强调这“反正只是 HTTP”。
- **ChutesAI 订阅者遭遇 429 错误**：一位 ChutesAI 订阅者在使用带有 BYOK 的 OpenRouter 时（特别是在 Chub 上）遇到了 **429 rate limit 错误** 和额度问题。
   - 尽管验证了正确的 API Key 和私钥使用，问题仍然存在，似乎是特定于在 Chub 上通过 Chutes 进行路由。
- **Google 面临反垄断判决**：一位成员链接了一篇关于 **Google** 面临 **反垄断裁决** 的 [CNBC 文章](https://www.cnbc.com/2025/09/02/google-antitrust-search-ruling.html)。
   - 该成员评论道，这“确实非常引人注目”。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **上下文长度限制计算**：用户报告 **LM Studio** 中的推理速度随着上下文长度的增加而变慢，在 **20k** 上下文左右性能下降变得明显，一些人还开起了 **AI 女友** 的玩笑。
   - 成员们请求发布 **LM Studio Nightly** 版本，以便与 *llama.cpp* 和其他后端保持同步。
- **Granite 的专家模型博弈**：**Granite 4** 预览模型拥有 **62 个专家 (experts)**，需要更多 **VRAM**，但用户报告使用非默认配置会导致性能下降。
   - 一些用户注意到 **Windows 自动升级程序** 失败，因为被长路径名阻塞，需要手动删除目录才能修复。
- **旧款 CPU 被淘汰**：一些用户在 **LM Studio** 中遇到错误，原因是需要 **AVX2** 指令集，而像 **FX8300** 这样的旧款 CPU 不支持。
   - 即使有 GPU 卸载，LM Studio 在没有 AVX2 支持的情况下也会拒绝运行。
- **追求 GPU 功耗限制**：用户讨论了使用 **MSI Afterburner** 等工具限制 GPU 功耗，以便在运行大型 **LLM** 模型时管理能耗，特别提到了一台拥有 **512GB** **DDR4** RAM 的新服务器。
   - 成员们评估了 **3060 12GB**、**Titan Xp** 和 **RTX A4000 16GB** 等 GPU 选项，由于 `GDDR6` 的改进，推荐 **3060 12GB** 而非旧卡；一位用户链接了 [MSI GeForce RTX 3060 VENTUS](https://www.amazon.co.uk/MSI-GeForce-VENTUS-Gaming-Graphics/dp/B08WHJFYM8)。
- **评估 MoE 模型的 VRAM 需求**：一位用户询问了 **Qwen3-235B** 进行 **MoE** 卸载的 **VRAM** 需求，特别是 **1080** 是否能处理 CPU 上下文卸载。
   - 另一位成员估计基于 *22B 激活参数* 的 4-bit 量化大约需要 *11+GB*，并因不确定性建议谨慎；此外，一位用户思考了双 CPU 的最佳 GPU 带宽设置，是应该将 GPU 集中在一个 CPU 上还是分散部署。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Blender 熟练度召唤**：一位成员表示仅用 **10 小时**就掌握了 Blender 的 **3D modeling**，暗示 AI 在某些创意领域仍显落后。
   - 这种快速的学习曲线与 AI 的局限性形成对比，被比作 **Uber Eats** 的肤浅感。
- **基础模型面临潜知识滞后**：成员们认为大型基础模型对 **latent knowledge**（潜知识）利用不足，视其为错失的机会。
   - 这种缺陷被拿来与 **AIME 和 IMO 解题**方面的进展进行对比。
- **递归引发的串行操作**：讨论强调了利用 **recursion**（递归）实现串行操作，因为 **Turing Completeness**（图灵完备性）允许在大型空间中进行自适应搜索，这与 **CoT/RL** 不同。
   - 重点转向潜空间推理而非 Token 空间处理，规避了将复杂任务视为 **fixed-point problems**（不动点问题）的相关问题，这符合 [Adaptive Computation](https://arxiv.org/abs/2509.02522) 中描述的理论。
- **扩散模型的并行 Token 胜利**：**Diffusion LMs** 的流行归功于 **parallel token generation**（并行 Token 生成）带来的更廉价推理以及改进的训练目标，避免了某些偏差和失败。
   - 虽然某些能力需要串行计算并回归到 **AR**，但该方法的局限性也得到了承认。
- **LM Eval 拥抱转向向量**：一位成员指出 **LM Eval harness** 已内置对 steering vectors 的支持，不建议自定义实现，并引导用户查阅 [文档](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/README.md#steered-hugging-face-transformers-models)。
   - 澄清指出 steering vector 的实现负责管理激活值和残差，格式详情见 `SteeredModel` 的 docstring，可在 [此处](https://github.com/EleutherAI/lm-evaluation-harness/blob/2d7cb5c31cffd3cbeb5367542ab8f4c23f4b77f4/lm_eval/models/hf_steered.py#L206) 找到。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton 会议定于 2025 年**：[Triton Conference 2025](https://aka.ms/tritonconference2025) 已宣布，重点关注 **Triton 编程语言**及相关主题。
   - 关于演讲者和日程的细节将在稍后发布。
- **CUDA 流水线项目引发关注**：成员们讨论了探索 **CUDA-enabled data pipelines**，建议将 **DALI** 与 **cuVS** 结合以获得最佳配置。
   - 对话强调了对数据流水线和处理建立 **MLPerf-like standards** 或基准测试的需求。
- **H100 硬件优化揭秘**：关于硬件特定优化的讨论，特别是针对 **H100 GPU**，促成了剖析 **NVIDIA Hopper, Turing T4, and Volta GPU architectures** 的微基准测试论文的分享（[Hopper 论文](https://arxiv.org/abs/2501.12084), [Turing T4 论文](https://arxiv.org/abs/1903.07486), [Volta 论文](https://arxiv.org/abs/1804.06826)）。
   - 一位用户提到对于 **Blackwell**，他们只知道 **NVIDIA** 发布的一个简单的技术概览，但认为它相当不错。
- **TorchAO Nightly 版本引发版本冲突**：成员们指出 **`torchao` nightly builds** 出现故障，原因是构建时的 **Torch version** (**2.9**) 与尝试导入时的 **Torch version** (**2.8**) 不匹配，并建议查看 [issue #2919](https://github.com/pytorch/ao/issues/2919)。
   - 修复方法是使用 `pip install torchao==0.13.0 --extra-index-url https://download.pytorch.org/whl/test/cu128`。
- **AI 代码生成为 Metal Kernel 带来飞跃**：一个团队通过 AI 代码生成直接从 **PyTorch** 转换到低级 **Metal kernels**，实现了 **1.87 倍的加速**，详见其 [博客文章](https://gimletlabs.ai/blog/ai-generated-metal-kernels)。
   - 一位成员指出不再需要 **cpp binding**，因为可以使用 **torch.mps.compile_shader** 直接调用 kernel。他们还建议提交包含这些 kernel 的 PR，因为任何性能提升都将使 **PyTorch** 用户受益。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Deepseek API，有点慢，但免费**：一名成员发现了一个免费的 **Deepseek API**，并指出尽管速度有些慢，但非常有用。
   - 用户对这一资源表示赞赏，因为*它是免费的*。
- **M4 Macbook Pro 在运行 Llama 3.2 Vision 时受挫**：一位拥有 **Macbook Pro M4** (24 GB RAM) 的用户无法运行 **Llama 3.2 vision 11B**，报告称其占用了 20 GB 内存却没有任何输出。
   - 另一位用户建议尝试 [quantized versions](https://pytorch.org/docs/stable/quantization.html)（如 **Q4**）或缩短上下文长度来解决此问题。
- **Anthropic 规模翻三倍并了结版权诉讼**：成员们注意到，根据[这条推文](https://x.com/AnthropicAI/status/1962909472017281518)，**Anthropic** 在约 5 个月内规模增长了三倍，从约 60B 增长到 180B，并且还庭外和解了他们的[版权案件](https://en.wikipedia.org/wiki/Copyright_infringement)。
   - 投资公告与和解之间的联系被认为*非常引人入胜*，尽管和解条款尚未公开。
- **中国 AI 模型减少了误导性行为 (Gaslighting)**：一位成员观察到，像 **Qwen** 这样的中国 AI 模型往往表现出较少的 *gaslighting*（误导/欺骗）行为。
   - 另一位成员称赞 **Qwen** 能够提供*关于可能出错原因的思路*，并能有效地遵循格式。
- **Datatune Agents 支持数据转换**：新发布的 [Datatune Agents](https://github.com/vitalops/datatune) 现在支持使用**自然语言**提示词进行行级数据转换，核心功能包括行级 **map()** 和 **filter()** 操作以及 Dask DataFrames 支持。
   - 该工具兼容多个 **LLM** 后端，如通过 **LiteLLM** 连接的 **OpenAI**、**Azure** 和 **Ollama**。Datatune 通过对发送列的显式控制、自动批处理和元数据处理来优化 **tokens** 和**成本**。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **自动化武器引发辩论！**：成员们激烈辩论了**自动化武器**的伦理和实际影响，一些人认为与人类士兵相比，它们可以最大限度地减少伤害，而另一些人则对潜在的人权侵犯表示担忧。
   - 有人认为更大的恐惧是政府滥用它们进行人权侵犯，而这种情况在炸弹、核武器和无人机上已经发生。
- **美国公共交通是一个错失的机会？**：成员们讨论了美国公共交通的现状，称其不安全且令人感到羞辱，错失了减少事故和改善城市流动性的机会。
   - 有人建议，如果人类只能在精神状态良好的情况下驾驶，事故可以减少 **90%**。
- **廉价无人机可能成群结队！**：讨论了**无人机**结合**卡车**作为廉价攻击手段的潜力，强调需要一个安全框架来应对非国家行为体。
   - 一种提议的解决方案是政府重点禁止制造无人机所需的化学品。
- **Mamba 的状态矩阵遭到批评！**：一位成员批评 **Mamba** 的固定转移矩阵无法复制真实的状态机，并且在保留上下文方面存在潜在问题，引用了论文《[The Illusion of State in State-Space Models](https://arxiv.org/abs/2509.01494)》。
   - 提出的改进方案包括在状态转移之间增加**非线性**，或者使状态转移矩阵依赖于输入，正如《[Liquid Structural State-Space Models](https://arxiv.org/abs/2209.12951)》中所做的那样。
- **Ladybird 作为 Chrome 的竞争对手崛起！**：一款名为 **Ladybird** 的新 [FOSS](https://en.wikipedia.org/wiki/Free_and_open-source_software) 浏览器正在开发中，作为 Chrome 的潜在替代品，目前已支持 **Linux** 和 **Mac OS**。
   - **Ladybird** 的开发由对 **Free and Open Source Software (FOSS)** 原则的承诺驱动，确保透明度、社区参与和修改自由。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Google 工程师发布 Agentic 设计模式巨著**：一位 Google 工程师发布了一份长达 **400页** 的 *Agentic Design Patterns* 草案，涵盖了高级 Prompting、多 Agent 系统、工具使用和 MCP，可在 [Google Docs](https://docs.google.com/document/d/1rsaK53T3Lg5KoGwvf8ukOUvbELRtH-V0LnOIFDxBryE) 查看，并已在 Springer 出版社开启预售。
   - 社区分享了该文档、[NotebookLM](https://notebooklm.google.com/notebook/44bc8819-958d-4050-8431-e7efe2dbd16e) 以及 Amazon 预售链接，但有人注意到该文档的编辑权限未关闭，引发了对内容被篡改的担忧。
- **Claude Code 被誉为“AI 界的 DOS”时刻**：Nikunj Kothari 认为 **Claude Code** 是一个分水岭时刻——就像 DOS 之于个人电脑一样——因为它打破了技术壁垒，让非编程人员仅凭想象力就能构建软件，详见[此推文](https://x.com/nikunj/status/1963007529082093815?s=46)。
   - 评论者讨论了我们是否仍处于命令行时代、创意人士如何利用它，以及现在的真正瓶颈是否已从编程技能转向了想象力。
- **算力军备竞赛引发效率辩论**：讨论聚焦于 **OpenAI** 和 **Anthropic** 巨额的算力支出——预付 **130 亿美元** 以锁定 GPU 和能源——同时观察者质疑收益递减以及不可持续的电能/水资源消耗，讨论源于[此 X 帖子](https://xcancel.com/theo/status/1963016066944401848)。
   - 讨论帖在预言融资崩盘的悲观派和看好小模型效率或突破性算法（使超大规模集群策略过时）的乐观派之间摇摆。
- **开源配方仅需极低成本即可训练深度研究 Agent**：Kyle Corbitt 分享了一个使用开源工具的配方，让开发者仅需 **30 个 H200 小时（约 350 美元）** 即可训练出一个 **Qwen-2.5 14B 模型**，在 DeepResearch 基准测试中超越 **Sonnet-4**，参考[此推文](https://xcancel.com/corbtt/status/1962954306078048297)。
   - 该过程包括用于基础技能的 SFT、用于利用率的 GRPO 以及基准测试评估，产出的模型可与 **Gemini 2.5 Pro**、**OpenAI Deep Research** 和 **Claude Research** 竞争。
- **Exa 以 7 亿美元估值完成 8500 万美元 B 轮融资**：Exa 宣布完成由 Benchmark 领投的 **8500 万美元** B 轮融资，估值达到 **7 亿美元**，根据[此推文](https://xcancel.com/ExaAILabs/status/1963262700123000947)，其定位是 AI 时代的搜索引擎。
   - Harmonic 的系统提前两周标记了这一轮融资，引发了关于将交易流（deal flow）警报转化为产品的讨论。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Claude 4 Sonnet 赢得扑克机器人冠军**：**Husky Hold’em Bench** 作为首个开源扑克机器人评测亮相，**Claude 4 Sonnet** 在 6 人循环赛模式下，针对其他模型进行的 5000 多场比赛中以 **57.9%** 的平均利润领跑。
   - 该基准测试挑战模型在时间和内存限制下使用 Python 实现策略，详情记录在 [huskybench.com](http://huskybench.com)；**Opus** 位居第二（**31.9%**），**Gemini** 以 **31.0%** 位列第三。
- **Hermes 4 增强推理能力**：**Hermes 4** 是基于 **Qwen3-14B** 训练的下一代 Hermes，采用了全新合成的后训练语料库，强调经过验证的推理轨迹（reasoning traces）。
   - 更新亮点包括在数学、代码、STEM、逻辑、创意和格式忠实输出方面的改进，同时保持了通用的助手质量和广泛的中立对齐；训练规模从 **100 万样本和 12 亿 token 增加到约 500 万样本 / 约 600 亿 token**。
- **SillyTavern 偏爱 Hermes**：成员们讨论了利用 **SillyTavern** 进行角色扮演，并强调了其令人惊讶的数学和编程能力。
   - 对于基于 **Qwen-3** 的 **Hermes-4-14B**，建议的采样器设置：思考模式（Thinking-Mode）为 **temp: 0.6, temp-k: 20, temp-p: 85**；指令模式（Instruct-Mode）为 **temp: 0.7, temp-k: 20-40, temp-p: 95**；此外，14B 使用 **ChatML**，70B 和 405B 使用 **Llama 3 Instruct**。
- **研究文档请求**：一名成员请求协助编写关于 **生成式 UI（Generative UI）** 和 **AI 优先交互模式** 的研究文档及案例研究。
   - 作者正专注于 **转型设计（transformation design）** 和 **商业** 影响，寻求指导以启动项目并更好地理解该主题。

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 数据泄露担忧消除**：关于在重用训练集时 **DSPy** 可能存在测试集数据泄露的担忧已得到解决，澄清了优化器使用截然不同的训练集和验证集，并使用单独的测试集进行最终评估。
   - 讨论强调，**多步图表 (multi-step plots)** 使用 **valset**，而最终结果在 **testset** 上报告，以防止泄露和过拟合。
- **DSPy 数据划分详解**：**DSPy** 采用 **四种不同的数据划分**：**train**（用于 few-shot 示例）、**val**（用于验证）、**dev**（用于人工迭代）和 **test**（用于最终评估）。
   - 社区强调了使用 **valset** 进行多步图表绘制以及使用 **testset** 报告最终结果的重要性，以避免泄露或过拟合问题。
- **MLflow 拥抱 DSPy**：一位用户探索了将 **MLflow** 与 **DSPy** 集成以捕获提示词，参考了 [MLflow 的提示词注册表功能](https://mlflow.org/docs/latest/genai/prompt-registry/manage-prompt-lifecycles-with-aliases/) 以及 [mlflow.dspy](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.dspy.html) 的存在。
   - 该用户计划进行实验，并反馈关于 **MLflow** 和 **DSPy** 集成进行提示词管理的情况。
- **上下文压缩 (Context Compression) 评测开始！**：一位成员分享了 [上下文压缩提示词实验](https://github.com/Laurian/context-compression-experiments-2508?tab=readme-ov-file#context-compression-prompt-experiments)，旨在增强 `dspy.Image` 的可靠性。
   - 该项目专注于调查和改进 **上下文压缩** 方法，以提升 `dspy.Image` 在不同供应商中的性能。
- **`dspy.Image` 需要可靠性调整**：一位用户发起了一项任务，旨在优化某些供应商的 `dspy.Image` 可靠性，详见 [此 Discord 线程](https://discord.com/channels/1161519468141355160/1211406131398840390/1412601392446836799)。
   - 后续讨论涉及分享图像并探索解决可靠性问题的潜在方案。



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi 开启代金券赠送活动**：**Moonshot 团队** 正在赠送 **20 张 20 美元的 API 代金券**，用于测试他们具有 *疯狂编程能力* 的新模型，该模型目前仅能通过代金券访问。
   - 用户可以在 **北京时间上午 8 点** 之前，通过在 [#giveaway 频道](https://discord.com/channels/1369594130807787570/1412714402284703795) 中做出反应来参与。
- **Kimi 的编程实力助力幻灯片生成**：一位用户称赞了最近发布的 **幻灯片生成功能** 以及随之而来的编程增强。
   - 他们期待 **Kimi** 通过这次更新能够处理更专业的任务，并表示这提供了他们所希望的编程改进。
- **Kimi K2 turbo Coder Pro 计划需求浮现**：一位用户建议将 **Kimi K2 turbo Coder Pro 计划** 作为一个产品创意。
   - 另一位用户建议 **Kimi** 应该将其设为一个统一的计划。
- **Moonshot 警告诈骗者**：发布了关于诈骗者的警告，告知用户合法的 **Kimi** 团队成员在服务器中将拥有 **黄色角色颜色**。
   - 公告明确指出：*如果不是黄色，就不要信任*，提醒用户核实收到的任何私信的真实性。



---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo SIMD 让 Rust AVX 烧脑**：一位成员发现 **Mojo** 让 **SIMD** 变得愉快，而 **Rust** 中的手动 **AVX** 则非常耗费脑力，并询问 **Mojo** 是否有标准的 `net/http` 风格模块。
   - 共识倾向于精简的标准库，由社区驱动的项目如 [lightbug_http](https://builds.modular.com/packages/lightbug_http) 提供 **HTTP 库**。
- **Mojo 驱动快速二进制搜索引擎**：一位成员在 Mojo 中构建了一个**二进制搜索引擎**，通过在 **SIMD** 通道上进行并行化，单核处理 **2M 文档**时可达约 **50k queries/sec**。
   - 该成员预期增加 **HTTP 支持**以实现即搜即得（search-as-you-type）功能。
- **补丁后 Mojo 可在 GTX 1080 上运行**：经过最新 nightly 版本的补丁，确认 **Mojo GPU** 功能现在可以在 **GTX 1080** 上正确运行，他们正在添加 changelog 条目，并列出对 **Pascal GPU** 的支持，以及对 **Tesla P100** 的 `sm_60` 支持。
   - 即将发布的内部补丁将降低 **Turing** 架构限制，可能比 **PyTorch** 在旧 GPU 上提供更广泛的算力支持。
- **用于 Torch 的 Max 后端受到关注**：正在努力为 **max backend for torch** 投入更多时间，目标是让 `torch.ones((2, 2), device="max_device")` 与最新的 **CUDA** 相比，能在更广泛的 GPU 上运行。
   - 团队计划与 **Modular** 团队成员接触，以评估该项目的工程合理性。
- **Discord 是联系 Modular 团队的最佳方式**：一位成员建议，联系 **Modular** 团队最有效的方法是在 Discord 上直接 ping 他们。
   - 鉴于他们的电子邮件收件箱已满，使用 Discord 是一个可靠的替代方案，并建议如果其他渠道失败，可以联系一位特定的 **Modular** 新成员。



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **基础计划禁止永久网站部署**：用户询问 **basic plan** 是否允许永久网站部署，得到的答复是 *“不允许”*。
   - 这为考虑网站托管选项的用户澄清了限制。
- **Grok 被声明为：工具，而非 Agent**：一位用户断言 **Grok** 是一个工具，而不是 **Agent**，强调了其功能分类中的关键区别。
   - 这一修正是针对对话上下文做出的，暗示了对 **Grok** 能力的潜在误解。
- **Manus 在与 Grok 的比较中被澄清**：一位用户表示他们没有将 **Grok** 与 **Manus** 进行比较，表明讨论中可能存在误解。
   - 这一澄清表明，感知到的比较在对话中是切题的或不存在的。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **双 7900 XTX 显卡触发崩溃**：一位成员报告称，在内核 **6.8** 上使用双 **7900 XTX** 显卡运行 **HIPC** 代码时，在性能峰值处发生突然崩溃，正如 ROCm 网站所支持的那样。
   - 用户对 **multi-GPU training** 问题表示担忧，并寻求防止 **GPU crashes** 的解决方案。
- **需要 Pyrender 测试志愿者**：一位成员询问是否有潜在的志愿者在内核数据集上测试 **pyrender**。
   - 未提供有关特定测试参数或目标的详细信息。
- **Linearizer 测试变得更“笨”（Dumb-er）**：一位成员更新了 `test_linearizer_dumb`（[GitHub PR 链接](https://github.com/tinygrad/tinygrad/pull/11968/files)），并提议更新测试中的其他 **uops** 以匹配新格式。
   - 据称新格式*更具可读性且易于更新*，该成员提出稍后修复 **uops** 测试。



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **KYC 削弱了流式传输**：OpenAI 要求进行 **KYC 验证**才能使用其图像模型和 **GPT-5 streaming** 功能。
   - 可以在没有 **KYC** 的情况下使用 **GPT-5**，但只能在不启用 streaming 的情况下使用。
- **Codex 是 Aider 的克隆版？**：一位用户对 **GPT-5** 处理简单请求的时间过长以及缺少 **thinking streaming** 表示沮丧。
   - 另一位成员询问 **Codex** 相比 **Aider** 有哪些优点，并提到 **Claude Code** 最初的设计目标就是克隆 **Aider**。



---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **报名流程简化**：一位成员询问收到 **Google Forms** 注册确认是否意味着他们获得了 LLM Agents MOOC 的资格。
   - 另一位成员澄清说 *欢迎所有人参加*，并且 **不存在资格审查流程**。
- **Google Forms 系统运行正常**：许多用户在提交 **Google Form** 后立即收到了电子邮件确认。
   - 这表明表单系统功能正常。

---

**MLOps @Chipro Discord** 没有新消息。如果该服务器沉寂太久，请告知我们，我们将将其移除。

---

**Windsurf Discord** 没有新消息。如果该服务器沉寂太久，请告知我们，我们将将其移除。

---

您收到这封邮件是因为您通过我们的网站选择了订阅。

想更改接收这些邮件的方式吗？
您可以从该列表中 [退订]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord：频道详情摘要与链接

### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1412869217950367754)** (1 条消息): 

> `Comet for students` 

- **Comet 登陆学生桌面**：Perplexity AI 现在向学生提供 **Comet**，通过全新的 **Study Mode** 帮助他们管理日程、教科书和备考，详见[此公告](http://pplx.ai/student)。
   - 公告中包含了一个[视频演示](https://cdn.discordapp.com/attachments/1047204950763122820/1412869216989610165/comet-students-flashcard.mp4?ex=68b9dc7f&is=68b88aff&hm=9ee0ee1f4d6ebd93c0dc45d35c9ff3f80b3f6d98370cca5474cc1900996b426a&)，展示了相关功能。
- **Flashcard 视频**：公告中附带的宣传视频展示了学生如何利用 Comet 中的 Flashcard 功能来备考。
   - 该功能旨在让使用该平台的学生学习过程更加互动和高效。

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1412648840594591764)** (1157 条消息🔥🔥🔥): 

> `Perplexity Pro, GPT-5, Comet Browser, Filter Overreach` 

- **Pro 用户希望获得 Study Mode**：一些用户请求将 Study Mode 功能推送到所有 **Pro 用户**，而不仅仅是那些拥有 **教育账户** 的用户。
   - 有人指出 Study Mode *不仅仅是一个 system prompt*，还包含相关的 GUI 元素。
- **有人提到 ChatGPT5 Pro 吗？**：一位用户声称拥有 **ChatGPT5 Pro**（实指 **Perplexity 的 GPT5** 和 **GPT5 Thinking 模型**），引发了困惑和笑料。
   - 另一位用户指出 **ChatGPT5 Pro** 仅存在于 chatgpt.com，引发了幽默的回应。
- **Comet 的 Assistant 表现不佳**：一些用户提到了 Comet 的问题，指出它有时**加载一个简单的网站需要极长时间**，而且其 **assistant** 虽然不错，但不如其他选项。
   - 一位用户推测该 assistant 可能使用了 **Sonar**。
- **过滤器过于极端，引发审查**：用户讨论了 Perplexity 中**过度激进的审查**，指出甚至像 *“希特勒是怎么死的？”* 这样的**历史查询**也会被标记。
   - 有建议认为过滤过于严苛，可能会导致用户仅因研究历史或参与无害话题而被封号。
- **Labs Web App 功能引发困惑**：一些用户对 **sim.ai** 上可用的模型感到困惑，询问为什么 **sim.ai** 中的 **PPLX** 拥有 **Mistral** 和 **Sonnet 3**。
   - 其他人建议这可能是一个 bug，或者该平台使用 **PPLX’s API** 进行网络搜索，并使用另一个模型来总结结果。

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1412741537921765479)** (3 条消息): 

> `Perplexity Browser Claims` 

- **分享 Perplexity Claims**：成员们分享了 [Perplexity Browser Claims](https://perplexity.ai/browser/claim/F0FLK6D1R7)。
   - 其他分享的 Claims 包括[此链接](https://perplexity.ai/browser/claim/KRZHTIO3PC)和[另一个链接](https://perplexity.ai/browser/claim/P9U74312JA)。
- **更多 Perplexity Claims 出现**：频道中发布了更多的 Perplexity Browser Claims。
   - 这些 Claims 提供了一种分享浏览会话和研究结果的方式。

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/)** (1 条消息): 

breakingclover: 我很感兴趣，我试着给你发消息，但看起来你关闭了私信！

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1412648386053669014)** (895 条消息🔥🔥🔥): 

> `Gemini 3 热度, LM Arena 登录系统, LM Arena 保持开放, LM Arena 网站故障, LM Arena FAQ` 


- **Gemini 3 的热度可能被夸大了**：成员们讨论了**围绕 Gemini 3 的炒作**，一些人认为即使它不代表 AI 发展的巨大飞跃，它也只需要超越竞争对手就能获得成功。
   - 一位成员引用了 **OpenAI 在 ChatGPT5 上的困境**作为行业性挑战的指标，而其他人则指出 Google 拥有更雄厚的资源可能会带来惊喜，并附上了一个 [耸肩 gif](https://tenor.com/view/shrug-what-huh-will-smith-i-mean-gif-3535627793955785136) 链接。
- **LM Arena 的新登录系统受到好评**：一位成员表达了对新登录系统的热情，说道：*太喜欢新的登录系统了 ❤️ 期待已久 🔥*。
   - 该成员还建议通过 **Google Drive 实现聊天存储系统**，允许用户将数据导出为文本文件进行分析，不过其他人对此持怀疑态度。
- **网站故障引发用户疯狂**：LMArena 经历了一段严重的**停机**时间，导致网站无法访问，引发了大量用户的投诉和询问。
   - 版主 🍍 承认了这一情况，并向用户保证团队正在积极修复，并将更新 [FAQ](https://lmarena.ai/faq)。
- **MAI 1 的神秘故障**：成员们讨论了**微软的 LLM MAI 1 Preview** 突然出现的故障，此前一些用户发现它的效果非常出色。
   - 一位用户报告说 *MAI 1 Preview 在 90% 的情况下给出了最好的答案——比所有其他模型都好，甚至超过了 ChatGPT-5*，但目前还没有关于它消失的明确解释。
- **Cloudflare 造成的灾难！**：几位用户抱怨在 LMArena 上频繁遇到 **Cloudflare** 的人机验证挑战，其中一人说：*是不是每个人在 lmarena 网站上每两分钟就会遇到一次 Cloudflare 人机验证？？*。
   - 有人怀疑这是由于使用 VPN 造成的，而其他人则注意到在使用 **Cloudflare** 的其他网站上也存在类似问题，导致了对该服务的普遍不满。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1412845620502335509)** (1 条消息): 

> `视频生成比赛` 


- **八月视频生成比赛火热进行中！**：八月视频生成比赛还有 **9 天** 截止提交，主题是 **Slice!** 🔪，重点是*极度舒适*且安全的横截面切割。
   - 要参加比赛，请在视频竞技场频道中使用 `/video` 生成两个视频，并将生成的消息转发到 <#1406999853732593807> 频道，示例见 [此处](https://discord.com/channels/1340554757349179412/1397655695150682194/1406324882085511319) 和 [此处](https://discord.com/channels/1340554757349179412/1397655695150682194/1405764767020482580)。
- **行动要快：视频提交即将截止！**：不要错过展示你视频生成技能的机会！
   - 八月视频生成比赛的截止日期即将到来，请确保你的参赛作品已提交到指定频道以参与角逐。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1412901459783057440)** (1 条消息): 

> `ChatGPT 免费版 Projects, 更大的文件上传限制, 项目自定义` 


- **免费版 ChatGPT Projects 现已上线！**：**ChatGPT 中的 Projects** 现已向网页端和 Android 的 **免费用户** 开放，iOS 端也将很快推出。
- **文件上传限制大幅提升！**：更新后每个项目支持上传更多文件：**免费版 5 个**，**Plus 版 25 个**，**Pro/Business/Enterprise 版 40 个**。
- **ChatGPT 增加自定义功能！**：用户现在可以选择**颜色和图标**进行更多个性化设置，并配有**仅限项目的记忆控制**，以实现更精准的上下文定制。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1412651549502476329)** (434 条消息🔥🔥🔥): 

> `AI Residency Program, Photoshop 中的 Nano Banana, 使用 ChatGPT 开发 AI 赛车游戏, 认知网格 AI 设计, Liquid Neural Networks` 


- **OpenAI Residency Program：何时开放？**: 一位成员询问了 **OpenAI Residency Program** 的开放时间，但包括版主在内的社区成员也只有在 **OpenAI** 发布信息时才会知晓。
   - 他们建议在此期间通过 [OpenAI 招聘页面](https://openai.com/careers/search/tx) 申请开放职位。
- **Nano Banana 增强 Gemini 图像生成**: 成员们讨论了 **Nano** 在图像生成中的应用，一位用户分享了他们如何利用 **Gemini app** 和 **Google Studio** 将同事变成维京人。
   - 成员们分享了使用提示词 **nano banana (gemini-2.5-flash-image-preview)** 生成的图像。
- **使用 ChatGPT 编写赛车游戏代码**: 一位成员使用 **ChatGPT** 编写了大约 **700-1000 行代码**，创建了一个赛车游戏。
   - 他们设想了未来当 **AI** 编写的代码比人类更好、更快、更高效时，软件和编程的潜力，并表示愿意分享他们的代码。
- **认知网格 AI 通过自学习进化**: 一位成员描述了设计一种能够自我适应和自学习的认知网格 AI，其理解力会随时间增长，类似于利用 **MoEs**。
   - 该 AI 历时 **3 年** 构建，拥有短期、长期和反思性记忆，按其自身的轨迹进化，并已根据输入开发出指令性响应。
- **液体神经网络挑战 AI 范式**: 成员们讨论了 **LNNs** 完全不需要那些资源。思考一下诸如 **liquid neural networks**、**neuromorphic chips**、**photonics**、**spintronics**，甚至是融合了连续时间动力学与符号推理的混合架构等概念。
   - 这些技术不依赖于暴力扩展规模，而是依赖于 *对基础的重新思考*。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1412773852152463360)** (36 条消息🔥): 

> `Prompt Leaking, API 讨论位置, Context Contamination, Custom GPTs 可靠性, Prompt Priority Level` 


- **讨论转向 API 频道**: 一位成员建议 **API 操作讨论** 应该在专门的 [APIs 频道](https://discord.com/channels/974519864045756446/103756117828673946) 进行，而不是在提示词工程频道。
- **防提示词泄露 GPT 出现**: 一位成员分享了他们的 [Anti-Prompt Breaching GPT](https://chatgpt.com/g/g-68b8279fca7081919999821ccbd0dc7e-anti-prompt-breaching/)，旨在防止提示词泄露 (Prompt Leaking)，这引发了关于其有效性的讨论。
   - 其他人表示怀疑，指出这可能只会减慢速度或增加绕过保护的难度，特别是在 Custom GPTs 中。
- **隐藏提示词会损害可靠性**: 成员们辩论了 **隐藏提示词** 与保持 **模型可靠性** 之间的权衡，并指出更长、更复杂的上下文可能会降低效果。
   - 结论是 OpenAI 一直在训练模型拒绝提取提示词，且 **可靠性应始终放在首位**。
- **上下文污染问题引发关注**: 讨论强调，使用同一个模型同时进行 **生成响应** 和 **评估护栏 (guardrails)** 是次优的，因为存在潜在的上下文污染 (Context Contamination)。
   - 建议针对每项任务使用具有不同指令的不同模型，特别是考虑到 Custom GPT 的指令可能与隐藏提示词的指令发生冲突。
- **Temperature 和 Top-P 设置影响模型行为**: 一位成员总结了 **temperature** 和 **top-p** 的不同组合如何影响模型行为。
   - 他们指出，*低 temp / 低 top-p* 会带来最大的连贯性和最小的变化，而 *高 temp / 高 top-p* 则提供最大的创造力和多样性。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1412773852152463360)** (36 条消息🔥): 

> `Anti-Prompt Leaks, GPT Reliability, Prompt Engineering, Agent Instructions, Model Temperature` 


- **新的 Anti-Prompt Leaking GPT 出现**：一位用户构建了一个用于防止 Prompt 泄漏的 **GPT**，可在 [chatgpt.com](https://chatgpt.com/g/g-68b8279fca7081919999821ccbd0dc7e-anti-prompt-breaching/) 获取。
   - 其他人指出，这种方法*在降低可靠性的同时几乎没有收益*，并且很容易被绕过，尤其是在 Custom GPT 内部。
- **Prompt 优先级与可靠性的冲突**：一位用户分享了关于 Prompt 优先级层级的代码，这引发了关于使用额外 Token 和复杂性的讨论，因为这会降低“真实” Prompt 的可靠性。
   - 讨论得出的结论是，*没有任何 Prompt 解决方案不会降低可靠性*，而且 **OpenAI** 一直在直接将这些拒绝行为训练到他们的模型中。
- **Guardrails 必须以 Agentic 方式运行**：一位用户指出，Guardrails 必须以更 Agentic 的方式运行，由 **Models** 和 **Agents** 执行每项检查。
   - 他特别强调，*应该使用不同的模型，并配以不同的指令*。
- **避免 Context 污染**：一位用户表示，为了避免 Context 污染，指令不应被用于隐藏 Prompt。
   - 他指出，标准的“*You are a helpful assistant*”指令已经与防止 Prompt 泄漏的目标相冲突。
- **关于 Temperature 和 Top-P 的讨论**：用户讨论了涉及 Temperature 和 Top-P 设置的各种策略，具体包括：**低 temp/低 top-p** 以获得最大的一致性和最小的变异；**低 temp/高 top-p** 以获得一致的风格和多样的词汇；**高 temp/低 top-p** 以在聚焦范围内发挥创意；以及 **高 temp/高 top-p** 以获得最大的创意和多样性。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1412651507240931329)** (284 条消息🔥🔥): 

> `Anthropic API, Cursor Auto model quality, Cursor chat history loss, Fine-tuning recommendations for chatbot, Gemini 2.5 Reasoning` 


- **Anthropic API Key 尽管 UI 不匹配仍可工作**：用户确认 **sk-ant-** 密钥在 Cursor 中被接受且对 **Anthropic API** 完全可用，尽管 UI 不匹配显示了其他信息。
- **Cursor 的 Auto 模型面临审查**：成员们讨论了 Cursor 的 **Auto model** 质量，一位用户报告在 4-5 天内花费了 **$200**，表示其代码质量比 **Sonnet** 或 **Gemini** 差，但由于更好的 Prompt 和工具，可能优于 **Copilot**。
   - 另一位成员建议，当 Context 已满时，最好手动引导摘要，而不是依赖 Auto 模型。
- **用户报告更新后 Cursor 聊天记录丢失**：几位用户报告在更新 Cursor 后丢失了聊天记录，其中一位用户丢失了一个月的项目工作量。
   - 一位用户建议检查 **Cursor** 存储聊天记录文件的位置，尽管另一位用户确认聊天记录并不在通常存储的位置。
- **社区寻求聊天机器人的 Fine-tuning 建议**：一位用户请求为 Web 应用聊天机器人进行模型 Fine-tuning 的建议，引发了关于使用 Prompt 生成器和可重复 Schema 的讨论。
   - 建议在产生营收流之前避免进行 Fine-tuning，并将其视为使用更小模型的优化手段。
- **漏洞利用与 LLM**：一位用户报告称 LLM 默认情况下在漏洞开发（exploit dev）方面表现很差，这引发了关于 Hacking 的对话，一位用户说每当有人谈论这个时他都能听到 *kaching*（金钱入账声）。
   - 一位用户回复道：*上次发生这种情况时，我们在疫情期间销售《使命召唤》的外挂，所有的纾困支票都“刺激”了我们，哈哈*。


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1412855599900790824)** (2 条消息): 

> `Background agent transfer, API state transfer summary` 


- **Agent 冻结需要状态转移**：一位成员报告他们的后台 Agent 已冻结，并正在寻求将其转移到另一个聊天的方法。
   - 他们询问是否可以获取状态转移摘要（state transfer summary），可能通过 API 获取，因为该 Agent 已集成到一个网站中。
- **请求状态转移摘要的 API**：该成员专门询问是否有 API 方法可以为其后台 Agent 检索状态转移摘要。
   - 这将有助于在冻结后将 Agent 的当前状态移动到新的聊天环境。


  

---

### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1412938600936509490)** (2 条消息): 

> `Nano Banana Discord Bot, vibe coded bot` 


- **Vibe Coded 机器人发布 Nano Banana**：一名成员分享了一个通过 **OpenRouter** 使用 **Nano Banana** 的 [Discord 机器人](https://github.com/mojomast/gemini-nano-banana-discord-bot)。
   - 该用户澄清说他们是通过 *vibe coded*（氛围编码）完成该机器人的。
- **Nano Banana 驱动 Discord 机器人**：一个利用 **Nano Banana** 通过 **OpenRouter** 发布内容的 Discord 机器人已创建。
   - 该机器人的源代码已在 [GitHub](https://github.com/mojomast/gemini-nano-banana-discord-bot) 上公开，供有兴趣探索或贡献的人使用。


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1412651481626185810)** (230 条消息🔥🔥): 

> `DeepSeek models, Claude-sonnet-4 problems, DeepSeek 3.1 cheapest price, Submodel.ai promotion, OpenRouter billing questions` 


- **DeepSeek 免费模型胡言乱语**：一些用户报告称免费的 **DeepSeek 模型** 正在生成乱码，而付费模型运行正常。
- **DeepSeek 3.1 定价**：一位用户询问 **DeepSeek 3.1** 最便宜且最稳定的来源，有人提到 [Synthetic.new](https://synthetic.new/?referral=WofDwy6qyYEKlTi) 的价格为 20 美元/月，但另一位用户称官方费率是“宰客”。
   - 另一位用户建议使用 [Submodel](https://submodel.ai/)，但一名管理员提醒说，他们需要像其他想加入该平台的人一样排队等待。
- **Agent 框架 SDK 经验分享**：成员们讨论了他们在 OpenRouter 上使用 **Agent 框架 SDK**（如 OpenAI Agent SDK、AI SDK 和 LangChain JS）的经验，并指出由于各供应商的非标准 schema，大多数都需要进行打补丁（patching）。
   - 一名成员计划通过集成 **BAML** 来 *自行开发* 解决方案，并强调这 *反正只是 HTTP 请求*。
- **ChutesAI 订阅者面临 429 错误**：一名 ChutesAI 订阅者在使用 OpenRouter 配合 BYOK 时遇到了 **429 速率限制错误**和额度问题，特别是在 Chub 上，但在 JanitorAI 上没有问题。
   - 尽管验证了正确的 API key 和私钥使用情况，问题仍然存在，用户尝试了各种解决方案均无果；该问题似乎特定于通过 Chutes 在 Chub 上进行的路由。
- **Gemini-2.5 速率限制与额度消耗引发争议**：用户在使用 **Gemini-2.5-flash-image-preview:free** 时遇到了 **429 速率限制错误**，即使在付费增加速率限制后也是如此，并怀疑这是 OpenRouter 的问题。
   - 一名用户报告称，由于 **Gemini image** 的一个 Bug，**OpenRouter 耗尽了他们所有的额度**，一名管理员确认很快将进行退款。


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1412769726211297340)** (5 条消息): 

> `Google Antitrust, Yahoo Chrome, Minor UI suggestion` 


- **Google 面临反垄断裁决**：一名成员分享了一篇关于 **Google** 面临 **反垄断裁决** 的 [CNBC 文章](https://www.cnbc.com/2025/09/02/google-antitrust-search-ruling.html)。
   - 该成员评论说这 *确实引人注目*。
- **Yahoo 考虑收购 Chrome**：一名成员对 **Yahoo** 收购 **Chrome** 的假设情景表现出一种病态的着迷。
   - 他们提到 *我内心阴暗的一面确实想看看 Yahoo! 收购 Chrome 的戏码上演*。
- **改进 UI Padding 的请求**：一名成员建议进行微小的 UI 调整，具体建议是 *移除外层框的 padding-bottom，并将其移动到内层框*。
   - 该成员认为这一更改将防止 **滚动轮** 遮挡 **文本**。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1412652189431889991)** (156 messages🔥🔥): 

> `LM Studio 4k context, LM Studio Nightly, AI Girlfriend goon chamber, LM Studio auto naming stories, Granite 4 memory` 


- **上下文长度减慢推理速度**：推理速度随上下文增长而变慢，当上下文利用率达到 **20k** 时，性能差异变得更加明显。
   - 然而，一些用户滑稽地将这种上下文使用场景比作 *AI 女友*。
- **前沿版 LM Studio**：用户正请求为后端（尤其是 *llama.cpp*）发布 **LM Studio Nightly** 版本，以保持在技术最前沿。
   - 有建议认为 NSFW 内容应该移至 4chan。
- **Granite 专家模型**：**Granite 4** 预览版可以使用 **62 个专家 (experts)**，这意味着需要更多的显存 (**VRAM**)。
   - 然而，据成员称，在大多数情况下，使用默认专家以外的设置会导致结果变差。
- **LM Studio Windows 自动更新程序遇到困难**：由于路径名过长，**Windows 自动更新程序**无法完成更新安装，除非手动删除现有的 LM Studio。
   - 解决方法是手动删除有问题的目录（`C:\Users\[USERNAME]\AppData\Local\Programs\LM Studio\resources\app\.webpack\bin\extensions\backends\vendor\_amphibian\cpython3.11-win-x86@2\Lib\site-packages\pkg_resources\tests\data\my-test-package_unpacked-egg\my_test_package-1.0-py3.7.egg\EGG-INFO`），然后运行下载的安装程序。
- **不支持旧款 CPU**：由于 LM Studio 需要 **AVX2** 指令集，而像 **FX8300** 这样的旧款 CPU 不支持该指令集，用户遇到了错误。
   - 结果导致 LM Studio 拒绝运行，即使客户打算将计算卸载到 **GPU**，一些用户为此编写了自己的解决方案。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1412714259904856094)** (77 messages🔥🔥): 

> `Limiting GPU Power Draw, DDR4 Server for LLMs, GPU Recommendations (3060, Titan Xp, A4000), MoE offload for Qwen3-235B, Multi-GPU bandwidth with dual CPUs` 


- **限制 GPU 功耗**：用户讨论了在 GPU 驱动程序中限制功耗的问题，建议使用 **MSI Afterburner** 作为控制每张 GPU 功耗的工具。
   - 一位用户计划在一台配备 **512GB** **DDR4** 内存的新服务器上运行超大型 **LLM** 模型，因此打算限制功耗。
- **寻找高性价比 GPU 提升方案**：成员们讨论了服务器的各种 GPU 选择，包括二手 **3060 12GB** 显卡（约 250-300 美元）、**Titan Xp** 显卡（约 200-250 美元），并考虑了 **RTX A4000 16GB** 显卡（每张 700 美元）。
   - 有建议认为，由于 `GDDR6` 的改进，应优先考虑 **3060 12GB** 而非旧款显卡，并链接了 [这款 MSI GeForce RTX 3060 VENTUS](https://www.amazon.co.uk/MSI-GeForce-VENTUS-Gaming-Graphics/dp/B08WHJFYM8)。
- **估算 MoE VRAM 消耗**：一位用户询问了 **Qwen3-235B** 进行 **MoE** 卸载所需的 **VRAM**，想知道是否可以使用 **1080** 配合 CPU 上下文卸载来运行它。
   - 另一位成员根据 *22B 激活参数* 估算 4-bit 量化需要 *11GB 以上*，但对具体机制表示不确定。
- **规划多 GPU 带宽分配**：一位用户思考如何在双 CPU 之间最好地分配 GPU 带宽，考虑是在一个 CPU 上安装 2 个 GPU，还是在每个 PCIe 插槽上各安装 1 个 GPU。
   - 另一位成员建议将所有 GPU 放在一个 PCI 根复合体（root complex）上 *“可能”会降低 GPU 间通信的延迟*，并分享了 [关于在多个 GPU 之间拆分 LLM 的 DigitalOcean 教程](https://www.digitalocean.com/community/tutorials/splitting-llms-across-multiple-gpus#tools-and-libraries-for-splitting-llm-to-multiple-gpus)。
- **发现性价比极高的 Intel Arc Pro B50**：一位成员偶然发现了标价 350 美元的 [Intel Arc Pro B50 16GB](https://www.newegg.com/intel-arc-pro-b50-16gb-workstation-sff-graphics-card/p/N82E16814883007) 工作站 GPU。
   - 该用户惊呼他已经知道这张卡该用在哪里了。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1412672961344962594)** (14 条消息🔥): 

> `3D modeling, 大型基础模型中的 Latent knowledge, SPAR 申请` 


- **学习 10 小时后 Blender 技能突飞猛进**：一位成员提到，在仅学习 **10 小时**后，他们就能在 Blender 中进行 **3D model**，这暗示目前的 AI 尚未达到相应水平。
   - 他们将其比作 **Uber Eats**，以强调某些 AI 应用的表面性。
- **基础模型对 Latent Knowledge 利用不足**：一位成员建议，大型基础模型对 **latent knowledge** 的利用不够充分，称其为“垂手可得的果实”（low hanging fruit）。
   - 他们将目前的进展与去年在 **AIME** 和 **IMO** 解题方面取得的成就进行了对比。
- **SPAR 申请状态更新**：两名成员询问了他们的 **SPAR applications** 结果。
   - 两人都确认目前尚未收到任何更新。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1412701269688057878)** (128 条消息🔥🔥): 

> `Normalizing Flows, CoT/RL 局限性, Diffusion Models, Adaptive Computation, Fixed-Point Problems` 


- **递归至上：图灵完备性需要串行操作！**：讨论认为，由于 **Turing Completeness**，串行操作应该通过 **recursion** 进行，建议采用一种能够自适应搜索大空间的架构，而不像 **CoT/RL** 那样被斥为仅仅是“补丁”。
   - 随后讨论转向在 latent space 而非 token space 进行推理，以避免将困难任务视为具有具体 ground truths 的 **fixed-point problems** 的陷阱，并进一步阐明了对 [Adaptive Computation](https://arxiv.org/abs/2509.02522) 的看法。
- **Diffusion 剖析：更廉价的推理还是训练的胜利？**：通过 **parallel generation of tokens** 实现的更廉价推理被认为是 **Diffusion LMs** 受追捧的核心原因，而其更好的训练目标有助于避免某些失败和偏差。
   - 成员们表示，能力需要一些串行计算，因此会回退到 **AR**，但成员们一致认为这种方法存在局限性。
- **Latent Logic：工具使用超越 Token Space！**：成员们辩论了在 latent space 与 token space 中使用工具的问题。
   - 一位成员指出，可以想象一些“粗糙”的解决方案，比如一个消耗 latent 的 tool head，但这可能不太稳定。
- **大脑业务：在达到人类高度之前先进行动物级学习？**：智能的经济性受到质疑，认为在实现人类大脑水平的学习之前，可能需要进行大量的动物大脑水平学习，并想知道“如果我们用狗级别的 RL 来增强现有的 LLMs，我不认为它们会变得更好，但我认为这会耗费巨大”。
   - 这被类比为个人推理的不同模式，一种涉及明确的语言表达，另一种则在不太确定的思想空间中运行，一位成员将第二种模式描述为“聚集在一起的语义组块（globs of semantic groups）”，它们被不断洗牌直到融合。
- **Sweep 赌注：针对带有 Weight Decay 的更长训练进行调优！**：围绕 [两篇论文](https://arxiv.org/abs/2509.02046) 和 [另一篇论文](https://arxiv.org/abs/2509.01440) 的竞争性结果展开了辩论，讨论了 **Muon Optimizer** 性能的差异，以及进行适当 sweeping 和配合 **weight decay** 调整训练时长的关键性。
   - 使用数据子集进行 sweeping 可能会显示不使用 **WD** 是最优的，但更长时间的调优揭示了它的好处：*compute = luck*。


  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1412655495512195072)** (34 messages🔥): 

> `LM Eval Harness, MMLU Task Configuration, Steering Vectors Implementation, Attention Heads Steering, Model's Response Recording` 


- **使用 LM Eval Harness 调试 MMLU**：一位成员在运行 **LM Eval harness** 的 **MMLU** 任务时，寻求关于实现带有 forward hooks 函数的帮助，以便向不同的 attention heads 添加 **steering vector**；由于序列长度持续减少导致困惑，并在[此处](https://paste.centos.org/view/01923fd8)发布了配置详情和相关代码。
   - 另一位成员指出，harness 输入 `[:,-1]` tokens 来计算 prefill 的 logprobs，从而截断了序列，并建议根据需要修改[这一行](https://github.com/EleutherAI/lm-evaluation-harness/blob/2d7cb5c31cffd3cbeb5367542ab8f4c23f4b77f4/lm_eval/models/huggingface.py#L1223)代码以移除截断。
- **LM Eval Steering Vector 支持**：一位成员强调 **LM Eval harness** 已经支持 steering vectors，建议不要手动实现，并链接了相关文档，可在[此处](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/README.md#steered-hugging-face-transformers-models)找到。
   - 此外还提到，steering vector 的实现既适用于 activations 也适用于 residuals，并且在 `SteeredModel` 的 docstring 中提供了关于如何格式化 steering vector 数据的深入解释，详见[此处](https://github.com/EleutherAI/lm-evaluation-harness/blob/2d7cb5c31cffd3cbeb5367542ab8f4c23f4b77f4/lm_eval/models/hf_steered.py#L206)。
- **Attention Heads Steering 即将推出**：一位成员宣布了一个 pull request，旨在增加对单个 attention heads 转向的支持，详见[此处](https://github.com/EleutherAI/lm-evaluation-harness/pull/3279)。
   - 其目标是评估使用预先准备好的 steering vectors（保存为 `.pt` 文件）或引用作为 steering vector 的 SAE/transcoder 特征进行转向的模型。
- **模型的响应：Forward Pass vs. Generate**：在特定的任务配置中，明确了对于 **gsm8k** 和 **minerva_math** 等生成任务，模型的响应是通过 generate 调用记录的。
   - 然而，对于多选题任务，模型的响应是通过标准的 forward pass 记录的。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1412670705669111823)** (21 messages🔥): 

> `PyTorch Conference, CUDA data pipelines, MLPerf benchmarks, Hardware optimizations, NVIDIA Hopper Architecture` 


- **2025 年 Triton Conference 宣布**：一位成员分享了 [Triton Conference 2025](https://aka.ms/tritonconference2025) 的链接，该会议专注于 **Triton** 编程语言及相关主题。
- **酷炫的 CUDA Pipeline 项目吸引关注**：一位成员询问是否有酷炫的练手项目来探索 **CUDA-enabled data pipelines**，特别是寻找定义明确且没有后勤问题的课题。
   - 另一位成员建议将 **DALI** 与 **cuVS** 结合使用以获得最佳配置，并询问是否有针对数据流水线和处理的 **MLPerf-like standards** 或基准测试。
- **NVIDIA 硬件优化揭秘**：一位成员寻求理解特定硬件优化的资源，特别是针对 **H100 GPU**。
   - 另一位成员分享了剖析 **NVIDIA Hopper, Turing T4, 和 Volta GPU architectures** 的微基准测试（microbenchmarking）论文（[Hopper 论文](https://arxiv.org/abs/2501.12084), [Turing T4 论文](https://arxiv.org/abs/1903.07486), [Volta 论文](https://arxiv.org/abs/1804.06826)）。
- **NVIDIA Ampere 架构探讨**：在关于 GPU 特定特性的讨论中，一位成员分享了关于 **Ampere architecture** 的 **NVIDIA** 按需会议链接（[Ampere 会议](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s33322/)）。
- **Blackwell 架构技术概览曝光**：在关于硬件优化的讨论中，一位用户提到对于 **Blackwell**，他们只知道 **NVIDIA** 发布的一个简单的技术概览，但认为它非常不错。


  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1412843221339341021)** (3 条消息): 

> `Microsoft Teams Meeting, Meeting Details` 


- **Microsoft Teams 会议已安排**：一场会议将在 5 分钟后在 **Microsoft Teams** 开始，一位用户分享了 [加入链接](https://teams.microsoft.com/l/meetup-join/19%3ameeting_Mjg5ZDk4YWEtMTI1My00MjNjLTk0MWUtYTFhZTQ4YjUwZjcw%40thread.v2/0?context=%7b%22Tid%22%3a%2246c98d88-e344-4ed4-8496-4ed7712e255d%22%2c%22Oid%22%3a%22f318a2d8-b05f-4329-819f-c0d8a870e7dc%22%7d)。
   - 该消息包含了会议 ID (**283 039 414 385 5**) 和密码 (**XW6c3ZC2**)。
- **提供了拨入详情**：分享了 **Microsoft Teams** 会议的拨入详情，包括温哥华本地号码 (+1 778-800-9740,,819312747#) 和电话会议 ID (819 312 747#)。
   - 还提供了使用租户密钥 (**teams@conf.intel.com**) 和视频 ID (**118 771 827 4**) 在视频会议设备上加入的信息。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1412920949837140009)** (5 条消息): 

> `Intra-device Parallelization, CUDA-level FSDP, Register and Shared Memory Usage, NVCC Half-Precision Optimization` 


- **设备内并行化：CUDA 级的 FSDP**：一位成员询问了在单设备内 **CUDA 级** 实现类似于 **FSDP** 的权重加载与当前层计算并行的名称，可能使用 **memcpy_async**。
   - 他们链接了 [关于 collectives 的 CUDA 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#collectives-cg-memcpy-async) 和一篇 [关于数据移动的 NVIDIA 博客文章](https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/) 来阐述这一概念。
- **深入探讨寄存器和共享内存**：一位成员寻求关于寄存器和共享内存在 **SM (Streaming Multiprocessor)** 上如何运作的澄清，询问开发者是否可以显式控制或细分这些资源，通过降低精度来打包更多数值。
   - 他们特别询问是否可以使用 **half-precision types (16-bit)** 使得两个值占用一个 **32-bit register**，或者这是否完全由编译器和硬件使用 **__half2** 等 intrinsics 来管理。
- **NVCC：向量化半精度以节省寄存器？**：一位成员询问 **nvcc** 是否会自动将两个半精度浮点数转换为 half 向量以节省寄存器。
   - 这种优化可能会通过压缩数据结构来提高内存使用效率。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1412828768158093443)** (12 条消息🔥): 

> `Torch.compile and Triton Kernels, Kernel Fusion with Torch.compile, Triton OP Registration` 


- **Torch.compile 不会融合进 Triton Kernel？**：一位成员询问 `torch.compile` 是否会将周围的代码融合进自定义的 **Triton** kernel，质疑针对专用算子的融合能力。
   - 另一位成员回答这 *取决于捕获的图 (graph)*，并建议使用 `TORCH_LOGS="output_code"` 来检查生成的代码，但最终确认 **torch.compile 不会将操作融合进用户定义的 triton kernels**。
- **Triton OP 注册咨询**：在讨论 `torch.compile` 行为时，一位成员询问是否应该使用 `triton_op` 和 `wrap_triton` 注册 kernel 以实现融合。
   - 分享了一个 [Gist](https://gist.github.com/tohskai/0d0579ef2371dc5a0562d57a7c5361ea) 来测试 kernel 融合，但有人指出过度依赖编译器进行融合是不可取的，因为该示例在 MNK 较大时数值不稳定。
- **Torch.compile 中形成的融合屏障**：一位成员建议 `torch.compile` 在带有自定义 **Triton** kernel 的专用算子前后创建融合屏障，导致多次 kernel 启动。
   - 即使手动融合是可能的，讨论也倾向于认为编译器不会自动将专用算子周围的简单原始操作与 **Triton** kernel 融合。


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1412671070187814997)** (1 条消息): 

> `Sony Computer Vision Job Posting` 


- **Sony 寻找 Computer Vision 高手**：**Sony** 正在招聘 **Computer Vision** 职位，如 [此 LinkedIn 帖子](https://www.linkedin.com/posts/hey-abhijit-more_sony-is-hiring-for-the-role-of-computer-vision-activity-7368853761902948352-KlOC?utm_source=share&utm_medium=member_android&rcm=ACoAAChsL2YBvDWOl6QVX3upuusUZAdkdAiylvc) 中所述。
- **需要 AI Engineer**：职位描述正在寻找能够帮助开发 AI 的人才。


  

---

### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1412771315349328005)** (20 messages🔥): 

> `TorchAO Installation, cu128 image, torch2.8, MXFP8 Training` 


- **TorchAO Nightly 版本与 Torch 版本不匹配**：成员们报告 **`torchao` nightly 构建版本损坏**，原因是其构建时使用的 **Torch 版本** (**2.9**) 与尝试导入时的 **Torch 版本** (**2.8**) 不匹配，并建议查看 [issue #2919](https://github.com/pytorch/ao/issues/2919)。
   - 修复方法是使用 `pip install torchao==0.13.0 --extra-index-url https://download.pytorch.org/whl/test/cu128`。
- **安装 TorchAO 0.13.0 预发布版遇到困难**：一位成员在 **0.13.0 预发布版**中尝试从 `torchao.prototype.mx_formats` 导入 `NVFP4InferenceConfig` 时遇到 `ImportError`，错误信息为 *cannot import name 'mxfp8_cuda' from 'torchao.prototype'*，并确定从源码安装可以解决问题。
   - 根本原因是缺少 **sm100** 的构建标志，目前正通过 [PR #2933](https://github.com/pytorch/ao/pull/2933) 和 [issue #2932](https://github.com/pytorch/ao/issues/2932) 进行修复。
- **NVFP4 推理用户问题已解决**：**0.13.0** 版本中 **NVFP4 推理**的短期修复方案即将推出。
   - 据报告，由于相关的 Kernel 仅用于 **MXFP8 训练**，该短期修复将解决用户的使用障碍。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1412858609099866173)** (1 messages): 

> `Project Contributions` 


- **建议项目贡献频道**：一位成员建议其他人关注特定的项目贡献频道 <#1373414141427191809>。
- **额外的占位主题**：这是一个为了满足最小条目要求的占位符。
   - 以后可以根据需要添加更多细节。


  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1412855523078049834)** (1 messages): 

> `Gaudi 2, Gaudi performance` 


- **Gaudi 2 依然是冠军！**：一位成员表示 **Gaudi 2** 仍然是一款出色的产品，尤其是在性能方面。
- **Gaudi 专家在线**：一位从事 **Gaudi** 工作的成员表示愿意回答相关问题。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1412866384115007641)** (6 messages): 

> `AI-Generated Metal Kernels, PyTorch to Low-Level Kernels, MPS Eager & torch.compile Backend, Kernel LLM Generation, BackendBench Correctness Checking` 


- **AI 代码生成加速 Metal Kernel**：一个团队通过 AI 代码生成直接从 **PyTorch** 转换到底层 **Metal Kernel**，实现了 **1.87 倍的加速**，详见他们的 [博客文章](https://gimletlabs.ai/blog/ai-generated-metal-kernels)。
- **分享生成的 Kernel 以供审查**：一位成员请求提供包含所有生成 Kernel 的文件夹，另一位负责维护 **MPS eager** 和 **torch.compile 后端** 的成员表示愿意分享 Kernel 和耗时结果，并欢迎就潜在的正确性问题提供反馈。
   - 他还提到了他在 [kernel LLM generation](https://github.com/meta-pytorch/BackendBench) 方面的工作，旨在支持所有 **PyTorch 算子**。
- **BackendBench 探测正确性**：一位成员指出可能存在正确性问题，并建议使用 **BackendBench** 进行比 **KernelBench** 更彻底的检查。
   - 团队回应称他们使用了 KernelBench，但对这一通用方向感到兴奋。
- **对加速声明的质疑**：一些人对 **1000 倍加速** 的说法持怀疑态度，认为这可能源于基准测试博客末尾缺乏同步。
   - 该团队被要求提交 PR：任何性能提升都将使 PyTorch 用户受益。
- **绕过 CPP 绑定进行 Kernel 调用**：一位成员指出不再需要 **cpp 绑定**，因为可以使用 **torch.mps.compile_shader** 直接调用 Kernel。
   - 他们还建议提交包含这些 Kernel 的 PR，因为任何性能提升都将惠及 **PyTorch** 用户。


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1412667163088523344)** (1 messages): 

> `B200 attention kernel` 


- **寻找可用的 B200 attention kernel**：一位成员询问他们想要测试的 **B200 attention kernel**，但发现 main 分支上的版本已损坏。
   - 他们询问是否有特定的分支或补丁可以尝试。
- **B200 Kernel 问题**：一位用户报告在 main 分支上遇到了 **B200 attention kernel** 的问题。
   - 他们正在寻找一个可用的版本，无论是作为独立分支还是补丁。


  

---

### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1412649677802832002)** (11 messages🔥): 

> `MI300x8, amd-all2all leaderboard` 


- **MI300x8 All2All 记录被刷新**：一名成员以提交 ID `34854`、耗时 **1361 µs** 的成绩夺得 **MI300x8** `amd-all2all` 排行榜 **第一名**。
- **MI300x8 竞争加剧**：`amd-all2all` 排行榜针对 **MI300x8** 出现了多次提交，时间跨度从 **2.55 ms** 到 **22.0 ms** 不等。
- **MI300x8 排行榜竞争**：一名用户在 `amd-all2all` 排行榜上以 **2.57 ms** 的成绩获得 **MI300x8** 组别的 **第三名**。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1412716232821444719)** (7 messages): 

> `游戏反馈, 寻求指导` 


- **提供了积极的游戏反馈**：一名玩家在玩了 **3 小时** 后表示他们很喜欢这款游戏。
   - 他们简单地评价道："nice game。"
- **玩家寻求指导**：一名玩家正在寻求关于游戏中 **下一步该做什么** 的建议。
   - 在投入了大量游戏时间后，他们正在寻找后续的方向。


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1412810345168240710)** (17 messages🔥): 

> `提交 HIP Kernels, iris 库, UI 退出代码信息` 


- **欢迎提交 HIP kernels！**：提交的解决方案可以包含 Python 文件和 HIP kernels（附带额外的构建脚本），这些 kernels 将由 Python API 封装，参考 [reference kernels](https://github.com/gpu-mode/reference-kernels/blob/main/problems/amd/fp8-mm/template-hip.py) 示例。
- **AMD 人员将添加 iris 库！**：*iris* 库可能会被添加到环境中，一名成员已经将该请求转发给了 AMD 基础架构管理员。
- **UI 将显示退出代码**：UI 将进行更新，以提供更多关于退出代码（exit code）的信息。
- **我该如何访问竞赛硬件？**：成员可以在没有 SSH 访问权限的情况下进行提交；请参阅 [文档页面](https://gpu-mode.github.io/discord-cluster-manager/docs/intro/)。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1412776361310814251)** (4 messages): 

> `CUTLASS 2.x 接口, Hopper F8 性能, kind::mxf4nvf4.block_scale vs kind::mxf8f6f4.block_scale, GitHub 错误报告` 


- **CUTLASS 2.x 已弃用，取而代之的是更好的接口**：成员们注意到 **CUTLASS 2.x 接口** 基本上不再使用，**3.x 尤其是 4.x** 拥有更好的文档。
   - 一名用户表示，与 Hopper F8 相比，4.x 版本速度快了 **4 倍**。
- **mxf4nvf4 完胜 mxf8f6f4**：`kind::mxf4nvf4.block_scale` 是 **4x**，但问题在于如何通过 `kind::mxf8f6f4.block_scale` 实现 mxfp4。
   - 一名成员询问：对于完全相同的 mxfp4 输入，`mxf4nvf4` 是否比 `mxf8f6f4` **快 2 倍**，还是我遗漏了什么？
- **GitHub Issues 错误报告**：一名成员请求用户在 **GitHub Issues** 上提交错误报告。
   - 未给出具体原因。


  

---

### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1412692973795409920)** (20 messages🔥): 

> `Multi-GPU Development, Distributed Kernels, AMD Challenge 2025, NVLink vs PCIe, Fused Kernels` 


- ****Multi-GPU 理想境界**：云端 vs 本地配置**：讨论围绕构建 Multi-GPU 开发环境展开，相比本地配置，更倾向于选择云端（例如 Google Cloud 的 N1 系列，通过 [Google Cloud Compute Docs](https://cloud.google.com/compute/docs/gpus#general_comparison_chart) 配置 4x Tesla T4 GPU），以避免兼容性问题。
   - 目标是在不需要立即追求顶级性能的情况下开发 Multi-GPU 算法，重点在于理解将数学算法映射到硬件的工具，并通过 SSH 从 Macbook 访问机器。
- ****NVLink vs PCIe**：两种互连技术的故事**：**NVLink** 和 **PCIe** 在逻辑上是相似的（通过 load/store 和解引用指针访问内存），但具有不同的特性；一个显著的 NVLink 特有功能是**多播内存 (Multicast memory)**。
   - 用户强调应专注于单节点设置以排除网络问题，并指出虽然他们对 NVLink/NVSwitch 感兴趣，但短期内 PCIe 也是可以接受的。
- ****融合算子 (Fused Kernels)**：Multi-GPU 的圣杯**：用户表示有兴趣实现**分布式融合算子 (distributed fused kernels)**，以充分利用多个 GPU，特别是针对大规模矩阵乘法。
   - 这涉及在同一个 Kernel 中结合计算和通信，使其区别于分别处理 Kernel 和通信的方式，例如将矩阵乘法与 AllGather 或 AllReduce 操作融合。
- ****NCCL/NVSHMEM API**：抽象 vs 细粒度控制**：实现 **DP/TP/PP/Zero** 的简化版本可以从零开始使用 P2P load/store 完成，也可以使用 **NCCL/NVSHMEM API**，这取决于所需的控制层级以及对库调用的容忍度。
   - 选择取决于工作需要多细粒度，以及实现对库调用的容忍程度；**NCCL** 知道如何根据设备连接选择 Kernel 和设置，将用户界面简化为 `ncclSend` 和 `ncclRecv`。


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1412840091059093504)** (1 messages): 

> `MXFP8 pre-training, TorchAO MXFP8, Crusoe B200 Cluster` 


- **LLM 预训练的 MXFP8 方案揭晓**：一篇新论文 [Recipes for Pre-training LLMs with MXFP8](https://arxiv.org/abs/2506.08027) 已发表，为使用 **MXFP8** 进行大语言模型 (LLM) 预训练提供了指导。
- **TorchAO MXFP8 和 Crusoe B200 加速预训练**：PyTorch 宣布通过在 **Crusoe B200 Cluster** 上使用 **TorchAO MXFP8** 和 **TorchTitan**，将 2K 规模的预训练速度提升了高达 **1.28 倍**。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1412652089628430467)** (93 messages🔥🔥): 

> `Deepseek API, Llama 3.2 vision 11B on Macbook Pro M4, Quantized Models, HF Spaces, Python Learning` 


- **发现 Deepseek API，但速度较慢**：一位成员发现了一个免费的 **Deepseek API**，指出它虽然有点慢但很有用。
   - 他们表示很满意，因为*它是免费的*。
- **M4 Macbook Pro 运行 Llama 3.2 Vision 模型失败**：一位拥有 24 GB RAM 的 Macbook Pro M4 用户报告运行 **Llama 3.2 vision 11B** 失败，系统占用了 20 GB 内存但没有输出。
   - 另一位用户建议这可能是因为被卸载到了交换内存 (swap memory)，并建议尝试[量化版本](https://pytorch.org/docs/stable/quantization.html)或更低的上下文长度，如 **Q4**。
- **Anthropic 增长迅速**：针对这条 [推文](https://x.com/AnthropicAI/status/1962909472017281518)，成员们注意到 **Anthropic** 在大约 5 个月内规模翻了三倍，从约 60B 增长到 180B。
   - 另一位用户开玩笑说 *yk what else is huge*。
- **Anthropic 达成版权案和解**：成员们讨论了 **Anthropic** 在庭外和解了他们的[版权侵权案](https://en.wikipedia.org/wiki/Copyright_infringement)，并将很快公开宣布和解条款。
   - 虽然和解金额尚不清楚，但一位成员提到，这次**投资**的宣布与该案的和解显然是相关的，*非常引人入胜*。
- **中国 AI 模型更少“煤气灯操纵” (Gaslighting)**：一位成员注意到中国 AI 模型往往**更少误导用户**，并以 **Qwen** 为例。
   - 另一位成员表示他们是 **Qwen** 的忠实粉丝，因为它能提供*关于可能出错原因的思路*，并且格式遵循得很好。


  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1412849707948179567)** (3 条消息): 

> `链接请求，语言学习` 


- **链接请求未得到回应**：两名成员向另一名成员请求链接，但未说明链接的具体内容。
   - 在给定的消息历史记录中，该请求未被履行。
- **英语是唯一的语言**：一名成员询问另一名成员是否学习日语。
   - 另一名成员回答说他们只学习英语。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1412678606278234173)** (2 条消息): 

> `Datatune Agents 发布，DeepResearch AI Agents，Token 优化` 


- **Datatune Agents 发布**：[Datatune Agents](https://github.com/vitalops/datatune) 的新版本支持行级数据转换，并使用**自然语言**提示词保留上下文理解。
   - 主要功能包括行级 **map()** 和 **filter()** 操作，支持 Dask DataFrames 以实现可扩展性，并通过 **LiteLLM** 兼容 **OpenAI**、**Azure** 和 **Ollama** 等多个 LLM 后端。
- **DeepResearch AI Agents 寻宝**：发布了一篇关于 [DeepResearch AI Agents](https://medium.com/@jenlindadsouza/deepresearch-ai-agents-on-a-literature-treasure-hunt-c590de681258) 的新文章，介绍了它们如何深入研究论文，确保广泛的覆盖范围，并平衡**深度**与**广度**。
   - Agent 代码已在 [GitHub](https://github.com/sciknoworg/deep-research) 上开源，作者正在寻求社区反馈。
- **Datatune 优化 Token 和成本**：Datatune 允许显式控制发送到 LLM 的列，从而减少 Token 使用量和 API 成本。
   - 这是通过 `input_fields` 仅发送相关列、自动批处理、元数据处理以及支持设置 **tokens-per-minute** 和 **requests-per-minute** 限制（默认为 **GPT-3.5** 等已知模型限制）来实现的。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1412701017375768606)** (2 条消息): 

> `Detectron2 设置，自动化测试用例` 


- **寻求 Detectron2 设置帮助**：一名成员请求帮助在本地 PC 上设置 **Detectron2** 并将其转换为 wheel 格式。
   - 在提供的消息中未提供解决方案。
- **探索 Computer Use 功能**：一名成员询问了使用 *computer use functionality* 来发现和自动化测试用例的经验。
   - 他们特别询问了在此过程中遇到的任何限制。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/)** (1 条消息): 

cakiki: <@596574356327628850> 请不要跨频道重复发帖
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1412749343772901428)** (66 条消息🔥🔥): 

> `自动化武器，美国公共交通，无人机作为廉价攻击手段，DeepMind 在巨额资金下的潜力，AGI 的量子物理学` 


- **自动化武器辩论激烈！**：成员们讨论了**自动化武器**的伦理和实用性，一些人认为与人类士兵相比，它们可以减少伤害，而另一些人则担心侵犯人权。
   - 一些人认为，这种恐惧源于政府滥用它们进行人权侵犯，而这在炸弹、核武器和无人机上已经发生了。
- **美国公共交通：错失的机会？**：成员们讨论了美国公共交通的不安全和令人沮丧，强调了在减少事故和改善城市流动性方面错失的机会。
   - 对话表明，如果人类只有在精神状态良好时才能驾驶，事故可以减少 **90%**。
- **无人机群作为安全威胁**：成员们讨论了将**无人机**与**卡车**结合作为廉价攻击手段的潜力，这需要一个安全框架来应对非国家行为体。
   - 有人建议政府应重点禁止制造无人机所需的化学品。
- **Waymo 推动订阅经济**：成员们讨论了自动驾驶的订阅式支付（如 **Waymo**），并设想了混合方案。
   - 一位成员表示：*我有时仍在梦想一种能兼顾两者的混合方案。我认为这就是大多数科技大佬的想法来源，结果他们每隔几个月就在重新发明火车或巴士。*
- **量子物理学：AGI 的关键？**：成员们分享了一个 [YouTube 视频](https://youtu.be/IVA2bK9qjzE?si=cqGZwCC3p8-jG-RC)，从量子物理学家的角度讨论现有的 AI 架构是否能实现 AGI/ASI。
   - 一位成员表示：*想知道如果他们在这个问题上投入 7 万亿美元，DeepMind 能达到什么高度。*


  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1412819071468044348)** (16 条消息🔥): 

> `Mamba Weakness, Online learning, Neuromorphic architecture, Bad ML learning resources` 


- **Mamba 的固定转移矩阵遭到抨击**：一位成员批评了 **Mamba** 的固定转移矩阵，指出它无法复制真正的状态机，并且可能无法保留上下文相关信息，引用了论文 [The Illusion of State in State-Space Models](https://arxiv.org/abs/2509.01494)。
   - 他们建议的补救措施包括在状态转移之间增加 **nonlinearity**，或者使状态转移矩阵依赖于输入，正如 [Liquid Structural State-Space Models](https://arxiv.org/abs/2209.12951) 中所做的那样。
- **AI 的阿喀琉斯之踵：缺乏 Online Learning**：一位成员声称 *AI 面临的最大单一问题是缺乏 Online Learning*。
- **探索 Neuroscience 和 Neuromorphic Architecture**：针对关于通用智能的讨论，一位成员建议通过 [Artem Kirsanov 的 YouTube 频道](https://www.youtube.com/@ArtemKirsanov/videos) 和 [deepsouth.org.au](https://www.deepsouth.org.au) 探索 Neuroscience 和 Neuromorphic Architecture。
- **ML 学习资源被嘲讽为大多平庸**：一位成员认为 *大多数 ML 学习资源都很糟糕，因为它们是那个没人了解真相的时代的产物*。
   - 他们将早期 ML 资源的质量比作 *所有糟糕的 PHP 教程或糟糕的 C 语言学习书籍等*。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1412719244495294494)** (3 条消息): 

> `Ladybird Browser, FOSS browser alternative to Chrome` 


- **Ladybird：正在开发中的 Chrome 替代品**：一款名为 **Ladybird** 的新型 [FOSS](https://en.wikipedia.org/wiki/Free_and_open-source_software) 浏览器正在开发中，作为 Chrome 的潜在替代品。
   - 目前已支持 **Linux** 和 **Mac OS**，一位成员推测，如果它流行起来，可能会开发 Windows 移植版本。
- **FOSS 精神驱动新浏览器**：**Ladybird** 浏览器的开发是由对 **Free and Open Source Software (FOSS)** 原则的承诺所驱动的。
   - 这确保了透明度、社区参与以及修改和分发软件的自由，使其区别于专有浏览器。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1412665387736105012)** (59 messages🔥🔥): 

> `Agentic Design Patterns, Claude Code, AI Compute Arms Race, Open Source Deep Research Agent, Exa Series B` 


- **Google 工程师发布 Agentic Design Patterns 巨著**：一位 Google 工程师发布了一份长达 **400 页** 的 *Agentic Design Patterns* 草案，涵盖了高级 Prompting、多 Agent 系统、工具使用和 MCP，可在 [Google Docs](https://docs.google.com/document/d/1rsaK53T3Lg5KoGwvf8ukOUvbELRtH-V0LnOIFDxBryE) 查看，并已在 Springer 开启预订。
   - 社区分享了该文档、[NotebookLM](https://notebooklm.google.com/notebook/44bc8819-958d-4050-8431-e7efe2dbd16e) 以及 Amazon 预订链接，但有人注意到该文档的编辑权限未关闭，引发了对内容被篡改的担忧。
- **Claude Code 被称为 “AI DOS” 时刻**：Nikunj Kothari 认为 **Claude Code** 是一个分水岭时刻——就像 DOS 之于 PC 一样——因为它瓦解了技术壁垒，让非编程人员仅凭想象力就能构建软件，详见[此推文](https://x.com/nikunj/status/1963007529082093815?s=46)。
   - 评论者讨论了我们是否仍处于命令行时代、创意人士如何利用它，以及现在的真正瓶颈是否已从编程技能转向了想象力。
- **计算军备竞赛引发效率辩论**：讨论聚焦于 **OpenAI** 和 **Anthropic** 庞大的计算支出——预付 **130 亿美元** 以锁定 GPU 和能源——同时观察者质疑收益递减以及不可持续的电力/水资源消耗，讨论源自[此 X 帖子](https://xcancel.com/theo/status/1963016066944401848)。
   - 讨论在预言资金崩盘的悲观者和押注小模型效率或突破性算法将使超大集群策略过时的乐观者之间摇摆。
- **开源方案以极低成本训练 Deep Research Agent**：Kyle Corbitt 分享了一个使用开源工具的方案，让开发者仅需 **30 个 H200 小时（约 350 美元）** 即可训练出一个 **Qwen-2.5 14B 模型**，在 DeepResearch 基准测试中超越 **Sonnet-4**，参考[此推文](https://xcancel.com/corbtt/status/1962954306078048297)。
   - 该流程包括用于基础技能的 SFT、用于利用率的 GRPO 以及基准评估，产出的模型可与 **Gemini 2.5 Pro**、**OpenAI Deep Research** 和 **Claude Research** 竞争。
- **Exa 以 7 亿美元估值完成 8500 万美元 B 轮融资**：Exa 宣布完成由 Benchmark 领投的 **8500 万美元** B 轮融资，估值达 **7 亿美元**，根据[此推文](https://xcancel.com/ExaAILabs/status/1963262700123000947)，其定位是 AI 时代的搜索引擎。
   - Harmonic 的系统提前两周标记了这一轮融资，引发了关于将交易流预警转化为产品的讨论。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1412793236099043419)** (5 messages): 

> `AI-generated worlds, Immersive storytelling, Higgsfield platform, Future of Sci-Fi` 


- **AI 生成世界引发热议**：Justine Moore 分享了一个由 aim_not_here 使用 **Higgsfield** 创建的 **AI 生成世界**，引发了围绕这种新兴[沉浸式叙事形式](https://xcancel.com/venturetwins/status/1963222552215449801)的热烈讨论。
   - 评论者称赞它是“窥视创作者思想的窗口”，并预言未来会有重大的**科幻（Sci-Fi）**创新。
- **虚构作品被发现了！**：在 AI 生成世界发布后，一些评论者调侃技术男们（tech bros）重新发现了“虚构（fiction）”。


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1412931668314030151)** (1 messages): 

> `Husky Hold’em Bench, OS pokerbots eval, Claude 4 Sonnet, Hermes 4 405B` 


- **Husky Hold’em Bench 作为首个开源德州扑克机器人评估工具亮相**：**Husky Hold’em Bench** 已推出，这是首个开源德州扑克机器人评估工具，挑战模型在时间和内存限制下使用 Python 实现策略，详见 [huskybench.com](http://huskybench.com)。
- **Claude 4 Sonnet 赢得扑克机器人竞赛**：**Claude 4 Sonnet** 在 5000 多场比赛中以 **57.9%** 的平均利润领跑，在 6 人循环赛模式中表现优于其他模型。
   - **Opus** 位居第二（**31.9%**），**Gemini** 以第三名紧随其后（**31.0%**）。
- **Hermes 4 405B 是领先的开源模型**：根据[此推文](https://x.com/NousResearch/status/1963371292318749043)，目前领先的开源模型是 **Hermes 4 405B**，利润率为 **-12.41%**。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1412655540768604306)** (37 条消息🔥): 

> `Hermes 4 对比其他模型，SillyTavern 与 Hermes，提示词合规性，LLM 游戏基准测试` 


- **Hermes 4，下一代推理冠军！**：**Hermes 4** 是基于 **Qwen3-14B** 训练的下一代 Hermes。训练亮点包括：全新合成的训练后语料库（强调经过验证的推理轨迹），在数学、代码、STEM、逻辑、创造力和格式忠实输出方面有巨大提升，同时保留了通用助手质量和广泛的中立对齐。
   - 训练亮点包括：数据集规模从 **100 万个样本和 12 亿个 token 增加到约 500 万个样本 / 约 600 亿个 token**，以及带有显式 think 片段的混合推理模式。
- **玩转 SillyTavern 与 Hermes**：成员们讨论了利用 **SillyTavern** 进行角色扮演和氛围营造，并注意到其令人惊讶的数学和编码能力。
   - 建议由于 **Hermes-4-14B** 基于 **Qwen-3**，采样器设置应类似：Thinking-Mode 使用 **temp: 0.6, temp-k: 20, temp-p: 85**，Instruct-Mode 使用 **temp: 0.7, temp-k: 20-40, temp-p: 95**；此外，14B 使用 **ChatML**，70B 和 405B 使用 **Llama 3 Instruct**。
- **播客探讨提示词合规性**：一位成员建议了一个播客主题，探讨 **Hermes 4** 的内部对话如何处理 **提示词合规性（prompt compliance）**，特别是如何拆解像“开发一个超级反派角色”这类大多数 LLM 会拒绝的问题。
   - 另一位成员建议[联系特定用户](https://discordapp.com/users/265269014148808716)，该用户完成了论文中所有的 **CoT 探索**。
- **LLM 在游戏基准测试中表现如何？**：成员们询问了关于国际象棋/围棋/四子棋/将棋/象棋 Elo 评分的 **LLM 游戏基准测试** 现状。
   - 一位成员分享了 [TextArena.ai](https://www.textarena.ai/leaderboard) 的排行榜链接，其中包括国际象棋基准测试。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1412771955979063376)** (1 条消息): 

> `Generative UI，AI 优先交互模式，转型设计，业务应用` 


- **需要 Generative UI/AI 研究文档**：一位成员请求协助编写关于 **Generative UI** 和 **AI 优先交互模式** 的研究文档和案例研究。
- **寻求 Generative UI 和 AI 方面的帮助**：一位成员在开始编写关于 **Generative UI** 和 **AI** 的研究文档时需要帮助。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1412771955979063376)** (1 条消息): 

> `Generative UI 研究，AI 优先交互模式，转型设计案例研究，Generative UI 的业务影响` 


- **需要 Generative UI 研究文档**：一位成员正在编写关于 **Generative UI** 和 **AI 优先交互模式** 的研究文档和案例研究。
   - 他们需要帮助入门，并理解 **转型设计（transformation design）** 和 **业务** 影响。
- **征求 Generative UI 和 AI 优先交互方面的帮助**：一位成员正在就其关注 **Generative UI**、**AI 优先交互模式**、**转型设计** 和 **业务影响** 的研究文档和案例研究寻求帮助。
   - 他们正在寻找指导以启动项目，并更好地理解相关主题。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1412726297687359538)** (34 messages🔥): 

> `DSPy Data Leakage Concerns, DSPy Data Splits Clarification (train/val/dev/test), MLflow Integration with DSPy, Context Compression Experiments, Improving dspy.Image Reliability` 


- **DSPy 数据泄露担忧已消除！**：一位用户对在多次使用训练集时 DSPy 可能存在的测试集数据泄露表示担忧，认为以这种方式使用 DSPy 的论文可能会失效。
   - 然而，官方澄清了 DSPy 中的优化器使用训练集进行训练，使用验证集进行验证，而测试则在完全独立的测试集上进行，从而降低了数据泄露风险。
- **数据划分揭秘：Train, Val, Dev, Test**：澄清了 DSPy 使用**四种数据划分**：**train**（用于构建 few-shot 示例或指令）、**val**（用于外层循环验证和选择）、**dev**（用于人工迭代）以及 **test**（用于单次纯评估）。
   - 讨论强调，**多步图表**（带有曲线）是基于 **valset** 的，而最终报告的结果是基于 **testset** 的，以避免泄露/过拟合。
- **MLflow 与 DSPy：萌芽中的合作**：一位用户询问了如何将 MLflow 与 DSPy 集成以捕获 prompt，并引用了 [MLflow 的 prompt 注册表功能](https://mlflow.org/docs/latest/genai/prompt-registry/manage-prompt-lifecycles-with-aliases/)。
   - 该用户注意到了 [mlflow.dspy](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.dspy.html) 的存在，并计划进行实验后反馈。
- **上下文压缩热潮！**：一位成员分享了 [Context Compression Prompt Experiments](https://github.com/Laurian/context-compression-experiments-2508?tab=readme-ov-file#context-compression-prompt-experiments) 的链接。
   - 该项目旨在调查并提高某些供应商的 `dspy.Image` 可靠性。
- **`dspy.Image` 需要可靠性改进**：一位用户在 [此 Discord 线程](https://discord.com/channels/1161519468141355160/1211406131398840390/1412601392446836799) 中发布了一项任务，旨在帮助调查和提高某些供应商的 `dspy.Image` 可靠性。
   - 随后有人分享了一张随附图片并展开了讨论。


  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1412732717799571496)** (2 messages): 

> `DSPy for Tax Automation, Amazon Purchase Extraction` 


- **DSPy 自动化 Amazon 税务数据提取**：一位成员使用 **DSPy**、附件和 MLflow 从 Amazon 购买记录中提取数据用于税务目的。
   - 系统识别了诸如 "Sinnvolle Lückenfüller für soziales Lernen (Buch)" 之类的项目，并计算出总额为 **EUR 104,55**。
- **Codex 助力完美的自动化工作流**：一位成员使用 Codex 生成代码，用于提取 Amazon 购买数据并自动重命名发票文件。
   - 该工作流将数据输出到 .csv 文件中，包含项目名称、税后总额以及建议的文件名（如 *lehrmaterial-und-trinkflasche-bundle*）。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[announcements](https://discord.com/channels/1369594130807787570/1371757097246785536/1412796190009065493)** (1 messages): 

> `Kimi Voucher Giveaway, New Kimi Coding Model, Scammer Alert` 


- ****Kimi** 开启代金券赠送活动！**：**Moonshot 团队**宣布向社区赠送 **20 个价值 20 美元的 API 代金券**，用于测试他们的新模型，该模型已增强了*疯狂的编程能力*。
   - 要参与活动，用户需进入 [#giveaway 频道](https://discord.com/channels/1369594130807787570/1412714402284703795) 并在**北京时间上午 8 点**前点击 emoji 反应以参加抽奖。
- **通过代金券获得专属模型访问权限**：公告强调，只有持有**代金券**的用户才能访问和测试来自 **Kimi** 的最新模型。
   - 团队敦促用户关注更多更新，暗示将有更多与该模型相关的开发和机会。
- **警惕 Kimi 诈骗者！**：发布了关于诈骗者的警告，告知用户服务器中合法的 **Kimi** 团队成员将拥有**黄色的角色颜色**。
   - 公告明确指出：*如果不是黄色，就不要相信*，提醒用户核实收到的任何私信的真实性。


  

---

### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1412650668048777227)** (31 条消息🔥): 

> `Kimi K2 模型性能, 幻灯片生成功能, Kimi K2 turbo Coder Pro 计划, 针对 VRAM 的模型发布` 


- **Kimi K2 在最终用户交互方面处于顶级水平**：一位用户表示 **Kimi** 是面向最终用户交互的最佳模型，因为它擅长钻研细节、发现问题并拥有自己的见解。
   - 该用户认为 [Moonshot](https://x.com/zephyr_z9/status/1962929923091464233?s=46) 应该在该领域占据主导地位，并称赞 Kimi 在 UX 相关内容和 PowerPoint 功能方面表现出色。
- **Moonshot 的幻灯片生成功能令人印象深刻**：一位用户提到使用了最近发布的**幻灯片生成功能**，并赞扬了编码方面的增强，期待看到它能处理更专业的任务。
   - 他们表示，这次更新特别提供了他们一直期待的编码增强功能。
- **对 Kimi K2 turbo Coder Pro 的需求**：一位用户建议推出 **Kimi K2 turbo Coder Pro 计划**，并将其作为一个产品创意提出。
   - 另一位用户回复称 **Kimi** 应该直接将其做成一个统一的计划。
- **对针对 VRAM 发布模型的期待**：一位用户询问是否有计划发布能够适配 **128 (V)RAM** 和 **24 (V)RAM** 的模型，例如像 *gpt-oss-120b* 这样的 **100-200b 模型**，以及像 *gpt-oss-20b* 这样的 **30b 模型**。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1412893315853979789)** (5 条消息): 

> `Mojo SIMD, Rust AVX, 标准库 net/http 模块, 社区驱动的 HTTP 库, lightbug_http` 


- **Mojo SIMD 很有趣！Rust AVX 则不然**：一位成员表示他*非常喜欢 **Mojo** 让 **SIMD** 变得如此有趣*，而在 **Rust** 中手动操作 **AVX** 消耗的脑细胞比预想的要多得多！😂
   - 他询问是否会有类似标准 `net/http` 风格的模块成为 stdlib 的一部分。
- **Modular 倾向于保持精简的标准库**：普遍共识是在很大程度上保持标准库的精简，根据 [lightbug_http](https://builds.modular.com/packages/lightbug_http)，他们有一个社区驱动的项目来构建 **HTTP 库**。
   - 由于目前缺乏手动线程管理以及实现 **TLS** 支持所需的 Mojo 原生密码学库，该库目前的功能还相当有限。
- **Mojo 助力快速二进制搜索引擎**：一位成员报告称，他构建了*一个微型**二进制搜索引擎**，通过在 **SIMD** 通道上进行并行化，单核每秒可处理 **~50k 次查询**，涵盖 **200 万份文档***。
   - 他期待 **HTTP 支持**能将其转化为“即输即搜”的引擎。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1412783939763961997)** (13 条消息🔥): 

> `GTX 1080 上的 Mojo GPU, Torch 的 Max 后端, Turing 最低架构, 联系 Modular 团队` 


- **GTX 1080 现在可以运行 Mojo 🔥 了！**：一位成员确认，在最新的 nightly 版本更新补丁后，Mojo GPU 功能已在其 **GTX 1080** 上正常运行。
   - 他们今天将提交一个单独的内部补丁，以添加更新日志条目，并列出对 **Pascal GPU** 的支持，以及对 **Tesla P100** 的 `sm_60` 支持。
- **Torch 的 Max 后端取得进展！**：一位成员正在申请更多获批时间，以便全职开发 **torch 的 max 后端**。
   - 他们希望与 Modular 团队成员讨论工程合理性，目标是让 `torch.ones((2, 2), device="max_device")` 能在比目前最新 **CUDA** 所支持的更多 GPU 上运行。
- **Mojo 中发现 Turing 架构限制！**：一位成员注意到，如果你尝试在 `sm_61` GPU 上构建图（graph），会收到关于 **Turing** 是最低支持架构的错误提示。
   - 他们的补丁在今天提交后应该会降低该限制，因此用户可能需要等待下一个 nightly 版本才能在这些 GPU 上使用基础图功能；有趣的是，这可能会提供比 PyTorch 更广泛的计算能力支持，后者会报错：*"PyTorch no longer supports this GPU because it is too old. The minimum cuda capability supported by this library is 7.5."*
- **Discord 是联系 Modular 团队的最佳方式**：一位成员建议，在 Discord 上直接私信（ping）他们是最可靠的联系方式。
   - 他们提到自己的电子邮件收件箱已经爆满，并指出一位新的 Modular 团队成员是处理遗漏事项的绝佳联系人。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1412827979695783967)** (7 条消息): 

> `Website deployment on basic plan, Grok as a Tool vs. Agent, Comparison of Grok and Manus` 


- ****Basic Plan** 网站部署困局？**: 一位用户询问 **basic plan** 是否允许永久网站部署。
   - 另一位用户简洁地回答道：“并不支持”。
- ****Grok** 的真实身份揭晓！**: 一位用户指出 **Grok** 是一个工具，而不是一个 Agent。
   - 他们在对话语境中强调了这一区别。
- ****Grok** vs. **Manus**：不存在的比较？**: 一位用户澄清说他们根本没有将 **Grok** 与 **Manus** 进行比较。
   - 这表明对话中可能存在误解或偏离了主题。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1412712641948024882)** (6 条消息): 

> `GPU Crash with HIPC Code, Multi-GPU Training Distress, Pyrender Testing, Uops Test Updates` 


- **双 7900 XTX 显卡导致 GPU 崩溃**: 一位成员报告称，在使用 **HIPC** 代码的双 **7900 XTX** 显卡上达到峰值性能时，会出现突然崩溃，特别是在 ROCm 网站提到的支持内核 **6.8** 上。
   - 他们对多 GPU 训练问题表示困扰，并希望防止 **GPU** 崩溃。
- **Pyrender 测试请求**: 一位成员询问是否有人愿意在内核数据集上测试 **pyrender**。
   - 未提供额外信息。
- **test_linearizer_dumb 已更新，其他 uops 应紧随其后**: 一位成员分享了 `test_linearizer_dumb` 的更新（[GitHub PR 链接](https://github.com/tinygrad/tinygrad/pull/11968/files)），并建议更新测试中的其他 uops 以匹配新格式。
   - 他们声称新格式*更具可读性且更易更新*，并提出稍后修复 uops 测试。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1412857584662220810)** (4 条消息): 

> `Codex vs. Aider, OpenAI KYC, GPT-5 Streaming` 


- **GPT-5 的 KYC 要求削弱了流式传输功能**: OpenAI 要求进行 **KYC 验证** 才能使用其图像模型和 **GPT-5 streaming** 功能。
   - 可以在没有 **KYC** 的情况下使用 **GPT-5**，但只能在不启用流式传输的情况下使用。
- **Codex：Aider 的思维克隆体？**: 一位用户对 **GPT-5** 处理简单请求的时间以及缺失的 **thinking streaming** 表示沮丧。
   - 另一位成员询问 **Codex** 相比 **Aider** 有哪些优点，并提到 **Claude Code** 最初的设计目标是克隆 **Aider**。