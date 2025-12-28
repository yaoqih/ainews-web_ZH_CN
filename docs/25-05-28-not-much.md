---
companies:
- deepseek-ai
- huggingface
- gemma
- claude
- bytedance
- qwen
- nemotron
- sakana-ai-labs
date: '2025-05-28T05:44:39.731046Z'
description: '**DeepSeek R1 v2** 模型已发布，并可在 Hugging Face 及其推理合作伙伴平台上使用。**Gemma 模型家族**持续蓬勃发展，包括
  **PaliGemma 2**、**Gemma 3** 等。**Claude 4** 及其变体（如 **Opus 4** 和 **Claude Sonnet 4**）展示了顶尖的基准测试性能，包括在
  **ARC-AGI-2** 和 **WebDev Arena** 上创下新的 SOTA（行业领先水平）。**Codestral Embed** 推出了一款 3072
  维的代码嵌入器。**BAGEL** 是字节跳动（**ByteDance**）推出的一款开源多模态模型，支持在长混合上下文下进行阅读、推理、绘图和编辑。基准测试亮点包括
  **Nemotron-CORTEXA** 在 SWEBench 上夺冠，以及 **Gemini 2.5 Pro** 在 VideoGameBench 上的表现。关于随机奖励有效性的讨论主要集中在
  **Qwen（通义千问）** 模型上。“Opus 4 在 ARC-AGI-2 上创下新 SOTA。它正在发生——我是对的”以及“Claude 4 的发布让开发节奏变得不同”等言论反映了社区的兴奋之情。'
id: MjAyNS0w
models:
- deepseek-r1-0528
- pali-gemma-2
- gemma-3
- shieldgemma-2
- txgemma
- gemma-3-qat
- gemma-3n-preview
- medgemma
- dolphingemma
- signgemma
- claude-4
- opus-4
- claude-sonnet-4
- codestral-embed
- bagel
- qwen
- nemotron-cortexa
- gemini-2.5-pro
people:
- yuchenj_uw
- _akhaliq
- clementdelangue
- osanseviero
- alexalbert__
- guillaumelample
- theturingpost
- lmarena_ai
- epochairesearch
- scaling01
- nrehiew_
- ctnzr
title: 今天没发生什么特别的事。
topics:
- benchmarking
- model-releases
- multimodality
- code-generation
- model-performance
- long-context
- reinforcement-learning
- model-optimization
- open-source
---

**平静的一天**

> 2025/5/27-2025/5/28 的 AI 新闻。我们为您检查了 9 个 Reddit 子版块、449 个 Twitter 账号和 29 个 Discord 社区（217 个频道，4755 条消息）。预计节省阅读时间（以 200wpm 计算）：418 分钟。我们的新网站现已上线，支持全元数据搜索，并以精美的 vibe coded 方式展示所有往期内容。访问 https://news.smol.ai/ 查看完整的详细新闻，并在 @smol_ai 上向我们提供反馈！

[DeepSeek R1 V2](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) 发布了，但我们会等到论文发布后再将其作为头条新闻。

[Dario](https://www.axios.com/2025/05/28/ai-jobs-white-collar-unemployment-anthropic) 就失业问题发表了一些令人担忧的言论。

我们仍在为下周的 AI Engineer 大会寻找[志愿者](https://x.com/swyx/status/1927558835918545050)以及[实时转录硬件/软件初创公司](https://x.com/swyx/status/1927822254416744466)。此外，请报名参加在旧金山围绕该大会举办的[数量惊人的周边活动](https://www.ai.engineer/#events)。

---

# AI Twitter 综述

**AI 模型发布与更新**

- **DeepSeek R1 v2 模型**：[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1927828675837513793) 提到 **DeepSeek 今天早上发布了 DeepSeek R1 v2！**[@_akhaliq](https://twitter.com/_akhaliq/status/1927790819001389210) 指出 **DeepSeek-R1-0528 刚刚在 Hugging Face 上发布。** 根据 [@ClementDelangue](https://twitter.com/ClementDelangue/status/1927825872221774281) 的说法，更新后的 **R1 已经在一些推理合作伙伴平台上可用**。
- **Gemma 模型家族**：[@osanseviero](https://twitter.com/osanseviero/status/1927671474791321602) 强调了 **Gemma 团队** 在六个月内的丰硕产出，包括 **PaliGemma 2, PaliGemma 2 Mix, Gemma 3, ShieldGemma 2, TxGemma, Gemma 3 QAT, Gemma 3n Preview, 和 MedGemma**，以及早期的模型如 **DolphinGemma 和 SignGemma**。
- **Claude 4**：[@alexalbert__](https://twitter.com/alexalbert__/status/1927803598936887686) 指出，一位 SWE 朋友有史以来第一次清理完了积压工作，**Claude 4** 的发布让开发进度达到了不同的节奏。[@alexalbert__](https://twitter.com/alexalbert__/status/1927410913453203946) 还表示，**Opus 4 + Claude Code + Claude Max 计划 = 目前所有 AI 编程技术栈中投资回报率（ROI）最高的组合**。
- **Codestral Embed**：[@GuillaumeLample](https://twitter.com/GuillaumeLample/status/1927736663419007031) 宣布发布 **Codestral Embed**，这是一款支持高达 **3072 维度** 的代码嵌入模型。
- **BAGEL 模型**：[@TheTuringPost](https://twitter.com/TheTuringPost/status/1927123416823251389) 强调了 **BAGEL** 的优势：该模型涵盖了阅读、推理、绘图和编辑，且没有质量瓶颈，支持长文本、混合上下文和任意宽高比。[@TheTuringPost](https://twitter.com/TheTuringPost/status/1927123359969468420) 提到 **ByteDance** 在其 **BAGEL（一个新的开源多模态模型）** 中提出并实现了这一想法。

**AI 性能与基准测试**

- **基准测试性能**：[@lmarena_ai](https://twitter.com/lmarena_ai/status/1927756554188566803) 报告称 **Claude Opus 4 在 WebDev Arena 中跃升至第 1 位**，超越了之前的 Claude 3.7 并与 Gemini 2.5 Pro 持平。[@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1927813645343305902) 表示，他们的评估显示 Sonnet 4 在编程性能方面有显著提升。
- **Claude 4 在 ARC-AGI-2 上的表现**：[@scaling01](https://twitter.com/scaling01/status/1927818210331521044) 指出 **OPUS 4 在 ARC-AGI-2 上达到了新的 SOTA。这正在发生——我是对的**。[@scaling01](https://twitter.com/scaling01/status/1927425665055302023) 还表示 **Claude 4 Sonnet 可能是第一个从 ARC-AGI 2 的 Test-time-compute 中显著获益的模型**。[@scaling01](https://twitter.com/scaling01/status/1927418304718623180) 提到 **Claude 4 Sonnet 在 ARC-AGI 2 上击败了 o3-preview，而价格不到其 1/400**。
- **Qwen 模型与随机奖励**：[@scaling01](https://twitter.com/scaling01/status/1927424801938825294) 报告的研究发现，随机奖励仅对 Qwen 模型有效，且改进归功于 Clipping（裁剪）。[@nrehiew_](https://twitter.com/nrehiew_/status/1927424673702121973) 询问，如果 Qwen 在任何随机奖励下都能生效，我们如何知道使用 Qwen 的 RL 论文是否真的起到了作用。
- **Nemotron-CORTEXA**：[@ctnzr](https://twitter.com/ctnzr/status/1927391895879074047) 提到 **Nemotron-CORTEXA 刚刚登顶 SWEBench 排行榜**，通过使用多步问题定位和修复流程解决了 68.2% 的 SWEBench GitHub 问题。
- **VideoGameBench**：[@_akhaliq](https://twitter.com/_akhaliq/status/1927722717068869750) 分享了关于 **VideoGameBench** 的论文。表现最好的模型 **Gemini 2.5 Pro 仅完成了 0.48% 的 VideoGameBench 和 1.6% 的 VideoGameBench Lite**。
- **数独求解**：[@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1927358732339425646) 讨论了前沿 LLM 在解决“现代数独”时面临的挑战。

**AI Agent 与工具**

- **自主 AI Agent**：[@cohere](https://twitter.com/cohere/status/1927417568832229781) 推广了一本关于为企业构建可扩展 **AI Agent** 的电子书，强调了它们的变革性影响。
- **代码 Agent 的数据库访问**：[@mathemagic1an](https://twitter.com/mathemagic1an/status/1927451869468660094) 指出 @codegen 现在配备了 “SQL 工具” + @plotly 集成。
- **Agent 安全**：[@mathemagic1an](https://twitter.com/mathemagic1an/status/1927137154829853118) 警告称，任何连接到 GitHub、MCP 或其他方式的 Agent 都存在真实的安全性问题。
- **Agent RAG 与 RAG 中的 R**：[@omarsar0](https://twitter.com/omarsar0/status/1927138441122213906) 讨论道，如果你的系统包含检索组件，那么你仍然在做 RAG。RAG 中的 R 代表检索（Retrieval），而相关性（Relevancy）是核心。
- **Factory AI**：[@matanSF](https://twitter.com/matanSF/status/1927755325848912259) 介绍了 Factory，一个编写代码的 AI。他们声称在 Agent 原生软件开发的新时代，Agent 交付代码，而 Droid 交付软件。
- **Mistral AI Agents API**：[@omarsar0](https://twitter.com/omarsar0/status/1927366520985800849) 注意到 **Mistral AI Agents API** 的发布，包括代码执行、网络搜索、MCP 工具、持久内存和 Agent 编排功能。[@omarsar0](https://twitter.com/omarsar0/status/1927372457578483828) 还提到了他们的 **Handoff Feature（移交功能）**，该功能允许 Agent 调用其他 Agent 来完成任务或在行动中途移交对话。
- **Comet Assistant**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1927130728954835289) 提到了使用 Comet Assistant 通过 AI 消费网页内容。
- **MagicPath**：[@skirano](https://twitter.com/skirano/status/1927434384249946560) 介绍了 **MagicPath，一个用于通过 AI 进行创建、完善和探索的无限画布**。[@skirano](https://twitter.com/skirano/status/1927806188923547925) 还宣布了 MagicPath 的 **660 万美元种子轮融资**。
- **Perplexity AI 助手**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1927798221130109170) 指出，你现在可以通过在 WhatsApp 中输入 /news 订阅当地时间上午 9 点的每日新闻，以使用 Perplexity AI 助手。
- **Runway References 使用案例**：[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1927803247508541416) 提到在城市中寻找灵感，并利用这些灵感来启发想法和更多创意。非常喜欢 References 的这个新用例。
- **Runway Gen-4 通用用例**：[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1927149229966766373) 提到，我们希望确保我们的模型具有无限的用例，这些用例不像简单的 “Text-to-X” 方法那样具有规定性和线性，因此 Gen-4 和 References 感觉像是朝着我们愿景中的通用性迈出的一步。

**AI 基础设施与硬件**

- **CoreWeave 的基础设施**：[@dylan522p](https://twitter.com/dylan522p/status/1927825707045933348) 与 CoreWeave CTO Peter Salanki 进行了一场有趣的对话，讨论了他们的创业故事、疯狂的 YOLO 豪赌，以及他们如何构建自己的 SW / HW 栈。
- **Groq 作为推理提供商**：[@JonathanRoss321](https://twitter.com/JonathanRoss321/status/1927797321699315729) 宣布 **Bell Canada 已选择 Groq 作为其独家推理提供商**。
- **CMU FLAME 中心**：[@gneubig](https://twitter.com/gneubig/status/1927435643476381848) 宣布 **CMU FLAME 中心拥有了一个新集群：256 块 H100 GPU**。
- **Baseten 推理栈**：[@basetenco](https://twitter.com/basetenco/status/1927488286764757112) 推广他们的推理栈，该栈由两个核心层组成：推理运行时（Inference Runtime）和推理优化基础设施（Inference-optimized Infrastructure）。
- **Hugging Face Spaces**：[@mervenoyann](https://twitter.com/mervenoyann/status/1927322723891466439) 将 @huggingface 的 Spaces 称为 AI 界的应用商店 📱 它现在也是 MCP 商店了 🤠 你可以筛选数千个可以连接到你的 LLM 的 MCP 🤗。
- **Mojo：** [@clattner_llvm](https://twitter.com/clattner_llvm/status/1927136935706812773) 与 @kbal11 聊了聊用于 Python 和 GPU 编程的 Mojo🔥。他们讨论了 Mojo 如何借鉴 Python, Rust, Zig 和 Swift 并更进一步——提供一种易于学习且能释放巅峰性能的语言。

**负责任的 AI 与伦理考量**

- **Anthropic 的长期利益信托**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1927758144303702249) 报告称，我们的长期利益信托（Long Term Benefit Trust）已任命 Reed Hastings 为 Anthropic 董事会成员。
- **Grok 的“白人种族灭绝”事件**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1927500443808075873) 报告称，一名未具名的 xAI 员工进行的未经授权更新导致 X 上的聊天机器人 Grok 错误地声称南非存在“白人种族灭绝”。
- **AI 安全的必要性**：[@Yoshua_Bengio](https://twitter.com/Yoshua_Bengio/status/1927481467988287656) 在 @TEDTalks 分享了他的个人经历，并概述了他为 AI 安全构想的科学解决方案。
- **RAG 系统缺陷**：[@omarsar0](https://twitter.com/omarsar0/status/1927737131478188295) 分享了一系列笔记，强调 RAG 系统比你想象的更脆弱，即使提供了充足的上下文也是如此。

**元讨论、思考与文化**

- **“越大越好”的时代正在结束**：[@cohere](https://twitter.com/cohere/status/1927775064721703258) 认为 AI “越大越好”的时代正在结束，因为高能耗、高算力的模型成本高昂且不可持续。
- **ASI**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1927458357801041927) 表示我们应该讨论的是 ASI，而不是 AGI。
- **可解释性才是核心**：[@francoisfleuret](https://twitter.com/francoisfleuret/status/1927233954899276052) 声称他大约每三个月就要提醒大家一次，信息论才是核心。
- **关于 EA 和意识立场的草率性**：[@aidan_mclau](https://twitter.com/aidan_mclau/status/1927207394737639900) 表示，EA（有效利他主义）——一个我少有的在智力上尊重的群体——是建立在绝对草率的意识立场之上的，任何有一点哲学功底和大脑的人都能反驳。
- **关于 RL**：[@corbtt](https://twitter.com/corbtt/status/1927428584257261994) 指出，最近的论文都指向同一个方向：RL 主要是诱发预训练中已经学到的潜在行为，而不是教授新行为。
- **AI 误解**：[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1927732071956451798) 表示，也许 AI 是本世纪最被误解的技术，因为它可以塑造出你想要的任何样子。
- **智能的复杂性**：[@jxmnop](https://twitter.com/jxmnop/status/1927141172541075539) 表示，我经常打算写代码并预期只需几个小时，结果却花了几天，我认为 AI 实验室也陷入了同样的谬误。但他们不是低估了代码的复杂性，而是低估了智能的复杂性。
- **技术的高估与低估**：[@ShunyuYao12](https://twitter.com/ShunyuYao12/status/1927506720051347730) 表示，技术在短期内被高估，而在长期内被低估。

**迷因与幽默**

- [@scaling01](https://twitter.com/scaling01/status/1927725546282053902) 开玩笑说我们距离 AGI 只有 85 秒了，因为你可以向 LLM 提出完全荒谬的问题，而它们总能找到答案。
- [@nearcyan](https://twitter.com/nearcyan/status/1927179638226268384) 评论道：“不，这条推文挺烂的。”
- [@scaling01](https://twitter.com/scaling01/status/1927733065150775786) 说 `Solve for X` 的正确答案是 C。已由 AGI 验证。
- [@teortaxesTex](https://twitter.com/teortaxesTex/status/1927746232769716504) 非常贴心，但我*不会*与 LLM 发生性关系。
- [@Fabianstelzer](https://twitter.com/fabianstelzer/status/1927649423657521608) 好的 AI 音乐提示词：“1950 年代保加利亚民俗科技舞曲，带有羊皮鼓点、多节奏喉音呐喊和钟琴琶音，通过 70 年代座机采样的教堂混响。”

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

### 1. DeepSeek-R1-0528 模型发布及早期基准测试

- [**deepseek-ai/DeepSeek-R1-0528**](https://www.reddit.com/r/LocalLLaMA/comments/1kxnggx/deepseekaideepseekr10528/) ([Score: 455, Comments: 150](https://www.reddit.com/r/LocalLLaMA/comments/1kxnggx/deepseekaideepseekr10528/)): **DeepSeek AI 已在 Hugging Face 上发布了 DeepSeek-R1-0528 权重，继续对模型权重和代码使用 MIT 许可证（参见 [仓库](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)）。此次发布较为低调，社区正积极将模型转换并上传为 GGUF 格式，以兼容推理引擎（参见 [unsloth/DeepSeek-R1-0528-GGUF](https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF) 的工作）。目前尚未包含正式公告或强调特定更新、基准测试结果或架构变化的文档。** 评论者赞赏 DeepSeek 低调的发布方式，并称赞其持续使用宽松的 MIT 许可证。社区对下游格式转换（GGUF）以实现更广泛部署表现出浓厚兴趣，并已开展相关工作。
    - 一个关键技术点是正在进行的 DeepSeek-R1-0528 模型 Dynamic GGUF 版本的转换和上传工作，一位活跃贡献者链接了正在进行的仓库 https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF。这表明对于依赖 GGUF 格式的用户，其可访问性和部署选项得到了增强。
    - 有一项关于 DeepSeek-R1-0528 权重可用基准测试的技术咨询，表明社区需要对比性能数据和评估结果，以便将此版本与其他模型进行背景化对比。
    - 一些用户关注 DeepSeek 是否会提供 R1-0528 的蒸馏版本，或者它是否将保持全尺寸模型，这引发了关于在本地运行该模型时资源消耗和潜在硬件要求的技术思考。
- [**DeepSeek-R1-0528 🔥**](https://www.reddit.com/r/LocalLLaMA/comments/1kxnjrj/deepseekr10528/) ([Score: 212, Comments: 59](https://www.reddit.com/r/LocalLLaMA/comments/1kxnjrj/deepseekr10528/)): **DeepSeek 发布了 DeepSeek-R1-0528，可在 [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) 上通过 MIT 许可证获取。此版本似乎是其最先进大语言模型的新权重或更新权重，因其开源可访问性以及在与顶级模型对比基准测试中的持续关注而备受瞩目。社区指出它树立了高标准（“开源界的 SOTA”），并期待后续版本（R2）以及对比基准测试数据。** 讨论重点在于对宽松 MIT 许可证的赞赏，并由于 R1 已经表现强劲，推测了下一个版本（R2）的时间表。还提到了围绕基准测试结果的行业压力，特别是提到了 Nvidia。
    - 几位用户期待 DeepSeek-R1-0528 的基准测试，并特别提到了 Nvidia 对结果的关注，反映出对新开源模型发布及其与现有硬件和软件栈竞争力的持续审视。
    - 强调了 DeepSeek-R1-0528 在 [Parasail.io](http://parasail.io/) 和 OpenRouter 等推理平台上的立即部署，表明该模型在发布后对现实世界测试的吸引力以及集成到工作流管道中的潜力。
    - 社区对使用 Unsloth 等专门库进一步微调 DeepSeek-R1-0528 表现出显著兴趣，表明用户渴望探索下游优化，以实现更高效的微调或推理性能。

- [**DeepSeek 宣布升级，可能发布类似于 0324 的新模型**](https://www.reddit.com/gallery/1kxdm2z) ([Score: 285, Comments: 54](https://www.reddit.com/r/LocalLLaMA/comments/1kxdm2z/deepseek_announces_upgrade_possibly_launching_new/)): **DeepSeek 宣布了一项升级，可能发布了一个与 0324 版本相似的新模型。早期用户的观察注意到，与之前的 R1 版本相比，响应延迟显著增加，但准确性也有所提高——成功回答了难倒 Gemini 2.5 Pro 的测试问题。目前尚未发布技术论文或详细的 changelog，用户要求进行彻底的技术披露。** 讨论集中在高延迟以及速度与回答质量之间的权衡，一些用户报告输出时间更长，但也暗示在推理连贯性方面可能有提升。技术用户正在请求 Benchmark、严格的评估和官方文档来证实这些改进。
    - 用户报告称，更新后的 DeepSeek 模型与之前的 r1 版本相比，响应时间明显变长。然而，这种延迟的增加可能与回答质量和连贯性的提高有关，因为它成功回答了一个 Gemini 2.5 Pro 没能答对的测试案例，这表明其底层推理能力可能有所进步。
    - 一些社区成员强调了官方公告和技术披露（如 Benchmark 对比或研究论文）对于新模型的重要性。人们期待有关此次升级如何使 DeepSeek 与 OpenAI 的 GPT-4 (o3) 和 Google 的 Gemini 等竞争对手抗衡的细节，特别是对潜在性能突破的关注。
    - 一位用户注意到之前的一个“翻译” bug 已得到解决，即模型在接收到此提示词时会产生不可见 Token 的幻觉。该修复表明官方正在持续关注影响多语言和 Tokenization 行为的边缘案例 bug，展示了在实现层面的增量改进。
- [**DeepSeek: R1 0528 表现惊人**](https://www.reddit.com/r/LocalLLaMA/comments/1kxs47i/deepseek_r1_0528_is_lethal/) ([Score: 116, Comments: 38](https://www.reddit.com/r/LocalLLaMA/comments/1kxs47i/deepseek_r1_0528_is_lethal/)): **该帖子报告称，通过 OpenRouter 访问的 DeepSeek R1 0528 在 RooCode 内的几个复杂编程任务中表现出强劲的性能，作者表示它顺利解决了所有问题。热门评论证实了这一点，用户指出 DeepSeek R1 0528 的编程能力与 Gemini 2.5 Pro 等领先模型持平或接近，使其成为代码生成和问题解决的竞争性替代方案。** 评论显示，大家对 DeepSeek R1 0528 在代码推理和生成方面的技术实力达成高度共识，特别是作为 Gemini 2.5 Pro 等高端模型的竞争对手。
    - 多位用户报告 DeepSeek R1 0528 在编程任务上的表现与顶级前沿模型持平，在一次对比中，它在编程 Benchmark 中的表现与 Gemini 2.5 Pro 相似，暗示其具备与顶级模型竞争的能力。
    - 一个详细的案例强调，DeepSeek R1 0528 成功处理了所有难倒 Claude 3.7 和 Opus 4 的提示词，特别是涉及编程提示词的解决。这突显了它在挑战性技术任务中强大的上下文和推理能力。
- [**DeepSeek-R1-0528 VS claude-4-sonnet（仍为演示版）**](https://v.redd.it/4lh915x90k3f1) ([Score: 156, Comments: 53](https://www.reddit.com/r/LocalLLaMA/comments/1kxmgtr/deepseekr10528_vs_claude4sonnet_still_a_demo/)): **该帖子尝试使用“七边形 + 20 个球”的 Benchmark 来对比 DeepSeek-R1-0528 和 Claude-4-Sonnet，但结论是该测试已无法区分它们的能力。然而，由于该 Benchmark 依赖外部物理引擎进行模拟，而非语言模型固有的能力，因此缺乏实质性的技术评估，使得结果对于评估模型性能或推理技能没有参考价值。** 热门评论批判性地指出缺乏技术背景或解释，并质疑使用物理引擎驱动的 UI 来对比 LLM 的有效性，同时要求提供演示本身的来源。
    - 讨论澄清说，在这个演示中，DeepSeek-R1-0528 和 Claude-4-Sonnet 都构建了一个与物理引擎交互的 UI，但关键在于物理计算本身并非由 LLM 处理。这引发了关于该演示究竟是在评估模型的能力，还是在评估底层物理引擎的性能或其 UI 集成能力的疑问。

- 一项技术观察指出，DeepSeek 在演示场景中可以说“处理物理效果”更好，尽管目前尚不清楚这指的是与物理引擎更好的集成/UI 逻辑，还是在 Prompt 中对物理交互有更好的推理能力。此处未提供直接的基准测试数据或实现细节。

### 2. 端侧生成式 AI：Google AI Edge Gallery 发布

- [**Google AI Edge Gallery**](https://i.redd.it/s6rgmrfawg3f1.png) ([Score: 170, Comments: 48](https://www.reddit.com/r/LocalLLaMA/comments/1kxa788/google_ai_edge_gallery/)): **该图片展示了全新的 Google AI Edge Gallery 应用，它支持在 Android（以及即将推出的 iOS）设备上完全离线地在端侧执行生成式 AI 模型。该应用的主要功能包括“Ask Image”、“Prompt Lab”和“AI Chat”，让用户能够与不同的本地运行模型（如“Gemma3-1B-IT q4”）进行交互，并调整推理设置（max tokens、TopK、TopP、temperature 以及 GPU 或 CPU 的加速器选择）。该应用在 GitHub 上开源，旨在允许用户直接在移动硬件上进行灵活且保护隐私的 GenAI 模型实验。[GitHub 项目链接](https://github.com/google-ai-edge/gallery?tab=readme-ov-file)。** 一位评论者质疑该应用是否得到了官方认可，因为它没有出现在 Play Store 中；另一位评论者提出了隐私担忧，声称该应用在每次 Prompt 后都会“打电话回家”（联网上报），这似乎与其声称的完全离线运行相矛盾。
    - 一位用户报告称，该应用的 v1.0.1 版本在 Pixel 7 上持续崩溃，而 v1.0.3 版本有所改进，但 CPU 推理仍然缓慢。另一位用户的 Pixel 7 推理速度明显更快，凸显了潜在的设备差异。尝试使用 GPU 推理进行后续提问时，两台设备均出现崩溃，表明 GPU 支持存在不稳定性或 Bug。
    - 有人提出了一个技术层面的担忧，即该应用在每次 Prompt 后都会“打电话回家”（发起网络请求），这暗示了潜在的隐私或架构影响，可能违背了对真正端侧（edge）处理的预期。

### 3. 显著的 AI 产品采用与行业反思

- [**《经济学人》：“公司正在放弃其生成式 AI 项目”**](https://www.reddit.com/r/LocalLLaMA/comments/1kxaxw9/the_economist_companies_abandon_their_generative/) ([Score: 531, Comments: 213](https://www.reddit.com/r/LocalLLaMA/comments/1kxaxw9/the_economist_companies_abandon_their_generative/)): **《经济学人》报告称，停止生成式 AI 项目的公司数量大幅增加，比例从去年的** `17% to 42%` **同比上升，详见[这篇文章](https://archive.ph/P51MQ)。主要驱动因素似乎是自动化导致的裁员后未能达到 ROI 预期，一些公司现在正在重新招聘之前被裁减的职位。该帖子讨论了实际的工作场所影响与炒作之间的关系，指出目前主要应用在于软件开发和平面设计。** 热门评论将生成式 AI 的炒作与互联网泡沫进行了比较，认为虽然短期预期被夸大，但长期影响可能被低估，并引用了 Linus Torvalds 和历史上类似的技术周期。评论者注意到了一种典型的行业“炒作-验证-修正”过程，并批评过早的 “AGI” 叙事助长了不切实际的假设。
    - 一位评论者强调，致力于修复那些使用 LLM 匆忙构建的生产级 SaaS 代码库的咨询公司正在激增，并强调“LLM 根本不应该编写进入生产环境的代码”，且在超出其能力范围的领域被过度应用。这表明目前的 LLM 生成代码与可持续、可扩展的软件工程实践之间存在显著的技术差距。
    - 检索增强生成 (RAG) 的重要性被认为是生成式 AI (GenAI) 真正的短期价值所在，其应用正在从专家圈子向外扩展。这与更广泛的技术观点一致，即相比于幼稚的端到端 LLM 应用，RAG 能更好地解决现实世界的挑战（例如将 LLM 与专有知识库或最新的公司数据相结合）。
    - 另一个讨论点将现状与 90 年代末的互联网时代进行了类比：虽然 GenAI 的变革潜力被广泛认可，但缺乏经过验证的商业模式或技术上可靠的应用，导致“钱被砸向一切”——这是炒作周期不成熟的标志，也是成熟前必须经历的技术验证阶段。
- [**Chatterbox TTS 0.5B - 声称超越 ElevenLabs**](https://v.redd.it/i6nfhj7rck3f1) ([Score: 141, Comments: 51](https://www.reddit.com/r/LocalLLaMA/comments/1kxoco5/chatterbox_tts_05b_claims_to_beat_eleven_labs/)): **Resemble AI 发布了 Chatterbox TTS 0.5B，这是一款开源的仅限英语的文本转语音模型，声称在质量上超越了 ElevenLabs（参见 [GitHub](https://github.com/resemble-ai/chatterbox)，[HuggingFace 上的权重](https://huggingface.co/ResembleAI/chatterbox)）。该模型通过带有** `pyproject.toml` **的 pip 进行分发，只需** `pip install .` **即可完成设置，并在运行提供的 Python 脚本时自动下载必要的模型权重。早期用户反馈确认了其高输出质量、可调节的表现力参数以及在 CPU 上运行短语的可行性；安装过程简单，但针对高级用例的文档较为匮乏。** 主要的技术争论围绕有限的语言支持（目前仅限英语），以及对源码构建或自定义模型放置的文档清晰度的批评。
    - 提供了托管在 Hugging Face 上的 Chatterbox TTS 0.5B 模型权重的直接链接 ([ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox))，使研究人员和开发人员能够轻松获取模型文件进行实验和集成。
    - 安装反馈强调，虽然有一个 `pyproject.toml`（意味着通过现代 Python 打包进行依赖管理），但在仓库根目录使用 `pip install .` 安装相对简单。运行提供的示例脚本会自动触发所需 `.pt` 模型权重的下载，尽管最初缺乏明确文档，但简化了设置。初步测试报告显示输出质量很高。

## 其他 AI Subreddit 摘要

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. Anthropic CEO Dario Amodei 谈 AI 与失业、幻觉及行业影响

- [**Dario Amodei 表示“不要再粉饰”即将到来的现实：在未来 1-5 年内，AI 可能会消除 50% 的初级白领工作，并将失业率推高至 10-20%**](https://i.redd.it/gx64b5fyak3f1.png) ([Score: 477, Comments: 210](https://www.reddit.com/r/singularity/comments/1kxnvbm/dario_amodei_says_stop_sugarcoating_whats_coming/)): **该图片是 Axios 新闻文章的截图，引用了 Dario Amodei（Anthropic 的 CEO）的警告，即 AI 的快速进步可能在 1-5 年内消除高达 50% 的初级白领工作，并可能导致失业率飙升至 10-20%。Amodei 敦促利益相关者直接应对这种破坏性的经济影响，特别是在技术和金融等领域，正如他在声明和原始文章中所详述的那样（参见 [Axios 文章](https://www.axios.com/2025/05/28/ai-jobs-white-collar-unemployment-anthropic)）。这突显了 AI 行业日益增长的担忧，即劳动力流失以及对自动化导致的大规模失业缺乏现实的准备。** 热门评论对预测的 10-20% 失业率数据持高度怀疑态度，认为这可能被低估了，并且需要进行更彻底的经济结构调整（如 UBI）。一个突出的观点是，如果大规模失业侵蚀了个人收入和消费能力，当前的消费者驱动型经济模式将从根本上受到威胁。
    - 一位评论者强调，AI 引起的大规模失业可能会严重破坏消费者驱动的经济模式，指出“AI 不花钱，有工作的人才花钱”，并质疑当大部分人口失去收入来源时，当前经济体系的可持续性。
    - 几位用户对 5 年内发生重大工作流失的保守估计表示怀疑，认为对初级白领工作的重大影响可能在短短 1-2 年内就会发生，暗示 AI 能力和部署的加速将超出公众和政策制定者的预期。
- [**Anthropic CEO 就失业问题公开表态**](https://www.axios.com/2025/05/28/ai-jobs-white-collar-unemployment-anthropic) ([Score: 481, Comments: 122](https://www.reddit.com/r/ClaudeAI/comments/1kxep8w/anthropic_ceo_goes_on_record_about_job_losses/)): **Anthropic CEO Dario Amodei 公开表示，AI 公司和政府都必须停止淡化大规模白领工作自动化的风险，并明确提到了技术、金融、法律、咨询，尤其是初级工作即将受到的影响。这一评论标志着对先进 AI 导致的结构性就业流失的规模和紧迫性的承认（参见 Axios 的相关报道）。** 评论中的技术辩论集中在这次转型的*新颖性和严重性*上：一些用户表示怀疑，将当前的恐惧与历史上的自动化焦虑相比较；而另一些人则认为，AI 驱动的替代速度和广度使得这一时期具有独特的后果，可能需要大规模的社会结构调整。一个相关的讨论串质疑了经济影响，特别是如果广泛的失业减少了收入，消费需求将如何持续，突显了资本主义面临的潜在系统性挑战。
    - 一条评论强调了最近的声明，暗示到 2025 年，像 Meta 这样的公司预测将拥有能够有效执行中级软件工程师任务（特别是编写代码）的 AI。然而，人们对 Meta 创造此类 AI 的能力表示怀疑，并提到了该公司目前在 AI 领域与其他主要参与者相比的地位。
    - 讨论涉及了 AI 驱动的失业对社会的影响，特别关注了新职位的创造不足，而历史上新职位总是能取代那些被技术淘汰的职位。与之前的工业转型不同，人们担心先进 AI 造成的流失规模和速度可能会破坏现有的社会经济结构，特别是在较短的时间跨度内（2 年、5 年或 10 年）。

- [**Dario Amodei 表示“别再粉饰太平”：在未来 1-5 年内，AI 可能会消除 50% 的初级白领工作，并将失业率推高至 10-20%**](https://i.redd.it/ex49znv9bk3f1.png) ([Score: 167, Comments: 78](https://www.reddit.com/r/ClaudeAI/comments/1kxnx0z/dario_amodei_says_stop_sugarcoating_whats_coming/))：**该图片是 Axios 文章的截图，引用了 Anthropic CEO Dario Amodei 的观点。他预测 AI 的进步可能会在未来 1-5 年内实现 50% 初级白领工作的自动化，潜在地将失业率提高到 10-20%。文章指出，这种影响将在科技、金融、法律和咨询等领域被广泛感受到，并呼吁政策和行业做出紧急反应。Amodei 的言论被定位为一个直言不讳的警告，要求人们停止低估 AI 在短期内对经济造成的破坏。** 评论对 Amodei 的可信度提出了挑战，强调了他“可能”这一说法的投机性质，并指出了当前 LLM 的局限性可能会阻碍这种快速的替代。其他人则暗示，鉴于他作为 AI 公司领导者的身份，夸大风险可能存在自身利益的考量。
    - 一位评论者提供了一个高度技术性的预测：在 1-2 年内，AI 模型可能能够完成大多数白领领域 80-90% 的任务，尤其是在初级水平。目前的局限在于 LLM 经常会卡住并需要人工干预，但随着 Agentic AI 的改进且卡住频率降低（预计很快会降至 50% 的时间），重组工作将成为可能：高级工程师（目前支持初级员工）将越来越多地转而为 AI Agent “脱困”，这暗示了任务监督动态的转变。该评论者进一步指出，由于高级职位存在模糊的奖励信号（reward signals），RL 驱动的模型更难对其进行优化，因此高级职位对自动化的抵抗力更强。
- [**AI 与大规模裁员**](https://www.reddit.com/r/singularity/comments/1kxlqfm/ai_and_mass_layoffs/) ([Score: 157, Comments: 170](https://www.reddit.com/r/singularity/comments/1kxlqfm/ai_and_mass_layoffs/))：**一家欧盟 Fintech 公司的资深工程师提出了一个问题：如果 AI 能够让公司通过自动化裁减约 50% 的工程岗位，那么是什么阻止了被裁掉的工程师利用同样的 AI，以更低的价格独立复制该公司的产品？这探讨了 AI 驱动的大规模裁员的实际动态，重点关注竞争壁垒、实施以及更广泛的劳动经济学。** 热门评论强调，直接复制和低价竞争受到行业准入门槛（如资本要求）、先发优势/企业市场优势以及现有品牌知名度的限制。另一个技术层面的担忧是关于经济循环性的：如果大规模裁员剥夺了消费者的收入，AI 驱动的生产力可能会因需求丧失而破坏其自身的商业逻辑。
    - 关于 LLM（Large Language Models）的可扩展性如何影响组织结构存在争论：一些人认为 LLM 最终可能会自动化高达 99% 的软件工程任务，潜在地将人类的角色简化为纯粹的协调员——但不确定 AI 是否甚至会达到自动化 C-level 领导层的程度，这表明 AI 在业务集成方面存在广泛的可能终点。
    - 目前观察到的生产力提升，特别是在法律等白领行业，表明 AI 可以显著降低服务成本和管理费用——律师指出，AI 驱动的工作流允许以“一半的价格”提供具有类似服务水平的竞争性定价，这表明固定成本低且硬件要求极小的行业已准备好迎接 AI 驱动自动化的早期转型。
    - 一个反复被提及的技术限制是，需要大量初始投资或专门硬件的行业（如由 Nvidia 或 Apple 等公司主导的行业）不太可能受到 AI 驱动的裁员或工人驱动的新进入者的冲击，因为即使 AI 实现了许多任务的自动化，资本和基础设施的壁垒依然存在。

- [**Dario Amodei 怀疑 AI 模型产生幻觉的频率低于人类，但其幻觉方式更令人惊讶**](https://i.redd.it/u3prpajfuh3f1.png) ([Score: 170, Comments: 106](https://www.reddit.com/r/singularity/comments/1kxd1so/dario_amodei_suspects_that_ai_models_hallucinate/)): **该图片总结了 Anthropic CEO Dario Amodei 的一项主张（据 TechCrunch 报道），即现代 AI 模型（如 Claude Sonnet 3.7 和带有 Grounding 功能的 Google Gemini 2.5 Pro）产生幻觉的倾向通常低于人类，尽管这些错误在本质上有时更令人惊讶或出乎意料。讨论强调了领先 LLM 在事实性和承认错误方面的最新改进，并将其与典型的人类对话错误进行了对比。图片直观地强化了这一核心引言，作为所链接 TechCrunch 文章的视觉摘录。** 评论者指出，像 Claude Sonnet 3.7 和 Gemini 2.5 Pro（带有 Grounding）这样的特定模型表现出的幻觉显著减少，并且通常比普通人类更准确，也更愿意承认错误。一些人还指出，错误的类型（例如，微小的事实偏差 vs. 虚构故事）在性质上是不同的。
    - 讨论强调模型行为各不相同：用户报告称 Claude Sonnet 3.7 产生幻觉的倾向低于大多数人类，因为它“更有可能在出错时承认错误”，并且与普通人相比，产生的虚构内容更少。这一观察基于用户将模型响应与人类倾向进行对比的直接经验。
    - 开启了 Grounding 功能的 Gemini 2.5 Pro 被引用为具有*极低*的幻觉率，其错误通常仅限于微小的事实细节（例如剧集编号），而不是生成完全虚假的叙述。这种对比表明，架构和 Grounding 功能的进步显著降低了 AI 幻觉的频率和严重程度。
- [**你对此有什么看法？**](https://i.redd.it/vn8v03vjij3f1.png) ([Score: 323, Comments: 143](https://www.reddit.com/r/OpenAI/comments/1kxjtwy/what_are_your_thoughts_about_this/)): **图片展示了一个会议场景，Anthropic CEO Dario Amodei 正在发言，并显示了他的名言：AI 模型产生幻觉（生成虚假或捏造的信息）的频率低于人类，但产生的幻觉更令人“惊讶”。TechCrunch 的品牌标识表明这源自一个知名的科技新闻活动。图片及随附的讨论集中在 AI 与人类之间错误模式（幻觉）的性质差异上——模型表现得高度自信、反应迅速，并且可以捏造新颖但令人信服的细节，这使得它们的错误可能更加微妙且难以察觉。** 评论者一致认为，虽然人类和 AI 都会犯错，但 AI 的问题在于其自信的表达方式以及捏造详细但听起来合理的虚假信息的潜力。共识是，这些差异使得 AI 错误具有危险性，需要进行验证和人工监督。一些人批评这位 CEO 的表述过于简化或属于“博眼球的观点（hot take）”。
    - 提到的一个关键技术区别是，AI 模型（尤其是 LLM）以高度的自信和流畅度生成虚假或捏造的信息，使其错误显得具有权威性，且可能比典型的人类错误更具误导性。这在根本上是一种不同的错误模式：人类错误通常带有某种认知痕迹或逻辑，而 AI 生成的内容——例如带有引用的虚构来源或捏造的事实——可能会产生一种有效性的错觉，从而使检测和验证变得复杂。
    - 一条评论强调了在关键任务中部署 AI 时实施技术保障措施的必要性。这些措施包括 Guardrails（防护栏），如自动来源验证机制和强制性人工审核层，以减轻源自自信但准确性不足或产生幻觉的输出所带来的风险。
    - 一些用户讨论了当前 AI 系统的一个关键局限性：无法像人类那样自我诊断或识别自己的幻觉或事实错误，人类有时可以追溯错误的起源，或根据新信息修正自己的信念。

### 2. AI 生成的病毒式视频、Veo 3 展示以及社会关注点

- [**致那些认为 Veo 3 生成的视频仍然明显造假、只能糊弄老年人的人**](https://www.reddit.com/r/ChatGPT/comments/1kxbxww/for_those_saying_veo3_video_generation_is_still/) ([Score: 380, Comments: 359](https://www.reddit.com/r/ChatGPT/comments/1kxbxww/for_those_saying_veo3_video_generation_is_still/)): **该帖子讨论了一个广泛流传的 AI 生成视频，据称是由 Veo 3 等工具制作的，描绘了一名在加沙的美国士兵。视频被认为能让大多数观众信以为真，尽管存在一些技术破绽，如不自然的镜头移动、不真实的背景虚化（bokeh）、人群声音缺乏声学真实感，以及符合 Veo 3 当前限制的短时长（通常在 10 秒以内）。然而，一条评论指出存在一个更长（17秒）、分辨率更高的版本，展示了细节（眼镜上的 RayBan 标志），这可能超出了 Veo 3 的原生能力，暗示可能经过了后期处理、使用了不同的生成器，或者对来源存在误解。更广泛的技术担忧集中在逼真的 AI 合成视频内容如何无缝地影响公众认知并导致大规模的虚假信息。** 评论中的辩论集中在 AI 视频内容升级到煽动现实世界冲突的风险，用户预见到虚假信息的质量和规模将进一步恶化。技术层面的质疑旨在根据模型限制以及视觉/音频异常来识别真实性。
    - 一位用户链接了一个 17 秒的高分辨率视频 (https://x.com/OpnBrdrsAdvct/status/1927604557577613350) 并指出这比官方 Veo 3 的剪辑生成长度更长，且保真度足以清晰辨认眼镜上的 RayBan 标志等微小细节——这暗示了近期模型迭代在时间一致性（temporal coherence）和细节方面的重大进步。
    - 一条评论指出，即使面对真实素材，观众也可能被误导去寻找 AI “破绽”，这突显了区分高保真 AI 生成内容（如来自 Veo 3）与真实视频的挑战日益增大，尤其是当模型能够捕捉精细的视觉细节和更长的连续序列时。
    - 一条评论声称该视频是真实的，只是音频经过了篡改，这强调了随着 AI 生成技术趋于照片级真实感（photorealism），关于真实性的持续技术争论；这种辩论暗示了在这些进步面前，需要更好的取证工具来检测 AI 制作的内容与真实内容。
- [**这段“情感支持袋鼠”视频在社交媒体上疯传，许多人认为它是真实的，但实际上是 AI 生成的**](https://v.redd.it/fvq48n5v4h3f1) ([Score: 4468, Comments: 382](https://www.reddit.com/r/singularity/comments/1kxax4j/this_emotional_support_kangaroo_video_is_going/)): **一段疯传的“情感支持袋鼠”视频在社交媒体上流传，但已被确认为 AI 生成，而非真实素材。Reddit 上的讨论强调，虽然对普通观众很有说服力，但仔细观察会发现视觉伪影（例如不自然的动作、细节不一致），这表明了当前生成式 AI 的局限性。随着生成模型不断提高照片级真实感（photorealism），这一事件说明了区分 Deepfakes 与真实媒体的挑战正在升级。** 评论者辩论了 AI 生成内容欺骗粗心观众的难易程度，并对媒介素养和 AI 驱动的虚假信息表示担忧。此外，还有一个更广泛的社会观察，即人类容易相信具有说服力的虚构叙事。
    - 讨论了 AI 视频生成的进步如何使内容具有高度说服力，特别是对于 Facebook 等平台上的普通观众。Deepfake 或 AI 生成的动物视频的真实感给未经仔细检查的检测带来了挑战，突显了公众在 AI 媒介素养方面的问题。
    - 该帖子评论了 AI 生成媒体被误认为真实内容的问题日益严重，强调需要更好的 AI 检测工具和公众教育来打击虚假信息，特别是随着合成视频变得更容易制作和传播。
    - 间接提出了对 AI 生成动物视频伦理影响的担忧，例如描绘处于痛苦中或不切实际场景的动物，如果不明确标记为合成内容，可能会在情感和理智上误导观众。

- [**这段情感支持袋鼠的视频在社交媒体上疯传，许多人认为它是真实的，但实际上它是 AI 生成的**](https://v.redd.it/fvq48n5v4h3f1) ([Score: 302, Comments: 60](https://www.reddit.com/r/ChatGPT/comments/1kxiyj8/this_emotional_support_kangaroo_video_is_going/)): **一段在社交媒体上疯传的描绘“情感支持袋鼠”的视频实际上是 AI 生成的片段，而非真实视频。这说明了目前生成式图像和 deepfake 技术的进步，其输出结果足以误导公众。现在的技术挑战不仅在于生成，还在于稳健的检测，因为复杂的媒体合成模糊了普通观察者的界限，并引发了对数字信息完整性的担忧（参见示例：https://v.redd.it/fvq48n5v4h3f1）。** 评论者强调了为缺乏技术背景的观众揭穿 AI 生成内容的难度，而其他人则思考了可能发生的社会转变，即人们会默认持有怀疑态度，这可能会改变艺术、娱乐和信息消费的观念。
    - 一位评论者观察到，高度逼真的 AI 生成视频的兴起可能会使公众脱敏，导致普遍的怀疑主义，并倾向于将不寻常的内容视为合成内容，这可能会从根本上改变艺术、娱乐和经济等领域的观念。
    - 另一个技术点强调，个人越来越有必要通过分析细微的社交线索、手势和情境恰当性来培养评估视频真实性的技能——这一问题因疫情后社交能力的下降而加剧，从而使普通公众检测 AI 伪造变得更加困难。
- [**终于用上了 Veo 3....**](https://v.redd.it/mge9n5ffse3f1) ([Score: 445, Comments: 72](https://www.reddit.com/r/aivideo/comments/1kx27fg/finally_got_to_use_veo_3/)): **一位用户分享了他们使用 Google Veo 3（一种领先的生成式 AI 视频工具）制作完整短片的经验。技术亮点包括使用 Veo 的 'flow' 模式通过文本提示词保持角色一致性，以及制作一段几分钟视频的成本约为 30 美元的 credits（不包括订阅费）。该工具能够以极少的人工投入实现高质量、游戏般的渲染，展示了对个人创作者的显著效率提升，并引发了关于对叙事和制作流程影响的讨论。查看 [Instagram reel](https://www.instagram.com/reel/DKLMC3lRw7L/?utm_source=ig_web_copy_link&igsh=MzRlODBiNWFlZA==) 了解作品，查看 [原始 Reddit 帖子](https://v.redd.it/mge9n5ffse3f1) 了解背景。** 评论者辩论了 credits 中极少人力劳动的价值，评论了 Veo 3 在民主化视频制作方面的潜力，并询问了精确的成本结构和创意工作流，表明了对专业可行性和可扩展性的浓厚兴趣。
    - 人们对使用 Veo 3 生成 AI 视频内容的制作成本很感兴趣。一位评论者直接询问制作该视频花了多少钱，这表明该工具的价格和效率是潜在采用者的重要考虑因素。
- [**最后的午餐 - Veo 3**](https://v.redd.it/wy1xffydzf3f1) ([Score: 158, Comments: 15](https://www.reddit.com/r/aivideo/comments/1kx7342/the_last_lunch_veo_3/)): **用户描述了使用 Veo 3（一种 AI 视频生成工具）将一个经典的 Reddit 笑话改编成视频。视频中除了由用户完成的旁白配音外，所有元素都是在 Veo 3 中生成的，突出了其 text-to-video 和多模态内容生成能力。** 评论者注意到像 Veo 3 这样的 AI 在转述和视觉化改编旧笑话方面的有效性，一些人对看到流传已久的笑话通过 AI 视频呈现感到惊讶。
    - 
- [**来生：AI 演员在提示词之间的隐秘生活（使用 Veo 3 制作）**](https://v.redd.it/alge2xqbre3f1) ([Score: 416, Comments: 60](https://www.reddit.com/r/singularity/comments/1kx21k6/afterlife_the_unseen_lives_of_ai_actors_between/)): **一个名为“来生：AI 演员在提示词之间的隐秘生活”的视频项目是使用 Google 的 Veo 3 生成式视频模型创作的。该项目探讨了 AI “演员”及其在用户输入之间存在的概念，暗示了实验性地利用 Veo 3 的叙事和视觉能力来表现概念状态。** 有人请求用 Veo 3 生成更多积极的内容，表明用户对多样化的情感或主题输出感兴趣。另一个关键的技术问题提出了 Veo 3 的视频生成是否能正确渲染 sign language，强调了对模型在复杂手势任务中保真度的审查。

- 一位评论者指出，Veo 3 目前已达到这样一个阶段：只要有足够的创意，用户已经可以使用该 AI 制作完整的电影。这表明在下一次迭代中——特别是随着 prompt 控制能力的提高——我们将在一年内看到第一部完全由 AI 生成的长篇电影。这突显了模型能力和用户级创意工具的飞速进步，预示着到 2027 年将被广泛采用，并可能引发内容生产方式的转变。
- 针对 Veo 3 在渲染手语方面的准确性提出了一个技术问题，有评论者询问其描绘是否正确。如果是这样，这将代表 AI 在忠实再现复杂手势交流方面的重大进步，这通常需要模型具备精细的空间和时间一致性。
- [**大规模心理失调即将来临！！！**](https://v.redd.it/lnquu4ct2h3f1) ([得分: 141, 评论: 59](https://www.reddit.com/r/OpenAI/comments/1kxarh3/mass_psychosis_incoming/)): **该 Reddit 帖子讨论了可能使用 OpenAI 的 Veo-3 等高保真 AI 视频生成技术创作的病毒式传播内容，引发了关于写实合成媒体影响的讨论。评论者强调了新出现的问题：在没有适当注明创作者的情况下广泛转载、技术特征的误导（例如生成的视频中出现虚假手语），以及社区对快速演进的生成式 AI 基准和能力的持续适应。文中提到了技术基础（Veo-3、相关软件/论文）和数字伦理。** 技术讨论集中在对内容归属的担忧、对转载的抵触，以及社区对于此类帖子饱和的元评论（meta-commentary），这反映出公众对 AI 进展现状的高度焦虑和困惑。
    - 一位评论者注意到频繁转载由 Veo 3 等 AI 模型生成的相同内容，质疑缺乏对原始创作者的适当归属。这突显了 AI 生成媒体生态系统中关于溯源、版权和内容生命周期的持续问题。
    - 讨论指向了超写实 AI 生成视频内容（特别提到了 Veo 3 的“浪潮”）令人不安的本质，并对 AI 作为意识载体提出了投机性问题，引发了对先进生成模型与新兴合成代理或意识交集的担忧。
- [**各位，原谅我的愚钝，这到底是 AI 还是真人？我真的分不出来**](https://v.redd.it/uy45tuuj7j3f1) ([得分: 5117, 评论: 754](https://www.reddit.com/r/ChatGPT/comments/1kxiah6/yall_excuse_my_stupidity_but_is_this_actually_ai/)): **该帖子争论了特定视频是否由 AI 生成，参考了当前最先进视频模型（如 Google Veo）的技术局限性，这些模型通常只能生成短片段（8 秒以下），且仍缺乏真实的表现力和无缝互动。技术用户指出，在断定真实性之前，评估连贯性和人类行为（AI 生成目前仍然明显薄弱的领域）至关重要。共识认为该视频可能是真实的，而非 AI 生成，这既反映了生成式视频质量的快速提升，也反映了持续存在的检测差距。** 评论承认生成式视频技术的快速进步，指出普通观众甚至会质疑视频的真实性，这在技术上是了不起的，但专家分析指出了合成内容的当前局限性。
    - 针对当前的 AI 视频生成能力提出了一个技术观点：虽然像 **VEO** 这样的模型正在进步，但在复制真实表情和互动方面检测到了一个关键局限。评论者指出，模型在令人信服地表现角色“表达情感或注视物体及彼此”方面仍然面临困难，这仍然是非人类生成的明显迹象。

- [**我用 AI 在 Grand Theft Auto VI 发布前制作了其预告片 3**](https://v.redd.it/ghtkyoyo1k3f1) ([评分: 136, 评论: 30](https://www.reddit.com/r/aivideo/comments/1kxnhes/i_made_grand_theft_auto_vi_trailer_3_before_grand/)): **一位用户利用最先进的 AI 视频和音频生成平台（具体为用于文本到视频合成的 Luma 和用于 AI 生成音乐的 Suno）重新制作了一个推测性的 Grand Theft Auto VI “预告片 3”。该预告片展示了 AI 在快速迭代复杂的、好莱坞风格视频内容（包括 VFX 和风格化提示）方面的能力，并演示了如何将生成式模型结合起来，在没有传统制作流程的情况下产生连贯的、专业级的输出。该项目突出了这些不断发展的工具的优势（快速原型设计、创意灵活性）和当前的局限性（“AI 疯狂”）。** 热门评论对技术成就表示赞赏，特别是在动态视觉效果方面，但也指出了当前一代 AI 特有的偶尔不自然的结果（“AI 疯狂”）。对 AI 生成的音轨有显著需求，表明了对底层模型或制作过程的兴趣。
    - 文中提到了“与当前技术状态相关的 AI 疯狂”，暗示该视频利用了先进的生成式 AI 模型或工具，可能是那些能够渲染逼真的游戏灵感电影画面的工具。用户提到了此类技术在创建高质量视觉效果和动作方面的快速进步和能力，类似于 AAA 游戏发行中的预期效果。

### 3. AI 模型/功能发布、基准测试和技术辩论 (SignGemma, DeepSeek-R1-0528, Hunyuan Video Avatar, WAN/VACE, 优化器, 行业方向)

- [**Google 发布 SignGemma，其最强大的手语翻译口语文本模型**](https://v.redd.it/5rkysqdt2i3f1) ([评分: 1000, 评论: 79](https://www.reddit.com/r/singularity/comments/1kxdp9l/google_announces_signgemma_their_most_capable/)): **Google 宣布了 SignGemma，这是开源 Gemma 模型家族即将推出的新成员，被设计为将其手语翻译为口语文本的最先进模型 ([详情](http://goo.gle/SignGemma))。该系统旨在提供稳健的手语到语音翻译，目标是提高可访问性和实时多模态通信，并将于今年晚些时候发布。该公告强调了包容性以及集成到辅助硬件（例如眼镜或耳塞）中的潜力。** 评论者指出 SignGemma 与边缘硬件配对进行无缝手语到音频和音频到文本翻译的潜力，强调了需要兼容设备来实现完整效用。
    - 与之前的模型相比，SignGemma 似乎生成的点云可视化不自然感较少，这表明在手语如何处理并渲染为文本或语音输出方面有显著改进。
    - 实时手语翻译的一个关键技术推动因素将是硬件进步——与 AR 眼镜或无线耳塞等设备的集成，可以使用像 SignGemma 这样的模型实现无缝的双向通信（手语到音频和音频到文本）。
- [**DeepSeek-R1-0528**](https://www.reddit.com/r/singularity/comments/1kxnsv4/deepseekr10528/) ([评分: 210, 评论: 84](https://www.reddit.com/r/singularity/comments/1kxnsv4/deepseekr10528/)): [**DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) 是 DeepSeek 在 HuggingFace 上发布的一个新 LLM 检查点。一位技术用户报告称，对于一个自定义的 Scrabble 编程/算法测试，该模型不仅在第一次尝试时就生成了高度准确、可运行的代码和稳健的测试，而且比 Gemini 甚至 OpenAI 的 o3 等竞争对手生成的代码更简洁、更优雅，而 o3 此前在该基准测试中表现最接近。该模型目前免费提供。** 在评论中，用户对模型在复杂的、重推理问题上的表现印象深刻，并注意到了高效且优雅的代码生成。有提到此类发布可能与 NVIDIA 财报相关的战略时机，暗示了更广泛的竞争影响。
    - 一位用户执行了一个以 Scrabble 为中心的自定义编程基准测试——这是一个既定的 LLM 个人测试。DeepSeek-R1-0528 是第一个完美通过它的模型：在明显的约 10 分钟推理阶段后，它生成的代码和测试在第一次尝试时就完美运行。之前的顶级模型（例如 'o3'）尚未达到这种水平的首次准确度或优雅度；相比之下，Gemini 生成的代码更冗长，并且错过了 DeepSeek-R1-0528 实现的独特、聪明的实现点。
    - 该帖子强调，虽然该模型是免费提供的，但有人要求提供标准化基准测试或公开的定量评估，这表明目前 DeepSeek-R1-0528 缺乏被广泛引用的基于分数的比较。

- [**Hunyuan Video Avatar 现已发布！**](https://www.reddit.com/r/StableDiffusion/comments/1kx6p8y/hunyuan_video_avatar_is_now_released/) ([评分: 211, 评论: 36](https://www.reddit.com/r/StableDiffusion/comments/1kx6p8y/hunyuan_video_avatar_is_now_released/)): **腾讯发布了 Hunyuan Video Avatar ([HuggingFace](https://huggingface.co/tencent/HunyuanVideo-Avatar), [Github](https://hunyuanvideo-avatar.github.io/))，这是一个开源的、音频驱动的图像转视频 (I2V) 生成模型，支持多角色。初始版本支持单角色、14秒音频输入；演示视频展示了高质量的唇形同步 (lip-sync) 和表情。最低硬件要求为 24GB GPU（速度较慢），推荐使用 80GB GPU 以获得最佳的 720x1280 @ 129 帧输出。** 评论者预计优化会很快到来，预测不久后将支持 8GB 以下显存 (VRAM)。技术上对 ComfyUI 集成有强烈需求，且与 LatentSync 等视频驱动方案的对比突出了基于图像的唇形同步的优势。
    - 一条技术评论指出，Hunyuan Video Avatar 处理 720px x 1280px x 129 帧至少需要 24GB 显存 (VRAM)，但在此配置下性能非常缓慢。为了获得最佳结果和更高质量，建议使用 80GB GPU，这表明与通常最高为 24GB 显存的消费级 GPU 相比，高效推理对硬件有极高要求。
    - 用户对 ComfyUI 的支持表现出直接兴趣，一位用户鼓励大家在相关的 GitHub issue (https://github.com/comfyanonymous/ComfyUI/issues/8311) 上投票，说明了对更广泛生态系统兼容性的需求，以及希望通过流行的开源 UI 框架实现更便捷的工作流集成。
    - 另一个技术观点指出，虽然 LatentSync 可以完成类似的视频到唇形同步任务，但它需要视频输入，而 Hunyuan Video Avatar 执行的是图像到唇形同步，这被认为是一个显著的易用性优势——特别是对于只有静态图像作为输入的模型用户。
- [**一个动漫 WAN 微调版本刚刚发布。**](https://v.redd.it/huzrjtmw4j3f1) ([评分: 401, 评论: 62](https://www.reddit.com/r/StableDiffusion/comments/1kxhyw4/a_anime_wan_finetune_just_came_out/)): **Stable Diffusion 的 WAN (Warp-Aware Network) 视频生成模型的一个全新动漫专用微调 (finetune) 版本已在 CivitAI 发布，提供图像转视频 (I2V) 和文本转视频 (T2V) 功能。该模型针对动漫典型的风格化动作和角色动画，旨在改进动画社区的创意视频工作流。早期反馈确认了诸如“嘴巴不停说话”等持续存在的问题，这与之前的 WAN 版本类似，尽管高级负面提示词 (negative prompting) 方法可能会减少此类伪影 (artifacts)。有关技术细节和权重，请参阅 [CivitAI 上的 WAN 动漫模型](https://civitai.com/models/1626197)。** 评论者强调，尽管动漫风格有所改进，但面部/嘴部动画伪影仍然存在，但承认负面提示词工程可以作为一种实用的缓解技术。
    - 评论者注意到动漫视频微调中持续存在的唇形同步问题，特别是“嘴巴不停说话”——这是当前视频生成和动画模型中的常见伪影，即生成的角色嘴部过度或不自然地移动，而不考虑对话时机。这表明在为动画内容进行微调时，面部动画与音频或上下文线索的对齐仍存在局限性。

- [**WAN i2v 和 VACE 低显存指南**](https://www.reddit.com/r/StableDiffusion/comments/1kx0ly2/wan_i2v_and_vace_for_low_vram_heres_your_guide/) ([Score: 135, Comments: 20](https://www.reddit.com/r/StableDiffusion/comments/1kx0ly2/wan_i2v_and_vace_for_low_vram_heres_your_guide/)): **该帖子提供了在低显存 GPU（如 RTX 3070 8GB）上运行 WAN 2.1（ComfyUI 封装版）和 VACE 的详细技术指南。关键建议包括：使用 480p 视频生成配合超分辨率（upscaling）以兼顾速度和显存效率；利用 ComfyUI 原生的智能内存管理（或使用 KJ Blockswap 进行高级控制）；并确保足够的系统 RAM（理想情况下为 32GB+），因为 WAN 可能会溢出到共享 GPU 显存和系统 RAM 中。实用工作流技巧涵盖：将 CLIP 卸载（offloading）到 CPU、严格的模型/编码器版本匹配、将超分辨率和插帧作为独立的工作流步骤、内存消耗细分，以及 fp8、gguf 和 fp16 模型类型之间的速度权衡。指南还为 I2V、T2V 和 V2V（包括 VACE 和 ControlNet）提供了详细说明，强调低批次大小（batch size）、最小化节点复杂度以及在可行时进行卸载。文中还附带了长视频的示例工作流和故障排除链接（例如 [Pastebin 上的工作流](https://pastebin.com/RBduvanM)）。** 在技术讨论中，一位评论者指出 **fp8 通常比 gguf 快**（特别是在 RTX 4000+ 上），而 **gguf 通过压缩在每 GB 显存下提供更好的质量，但代价是推理速度较慢**。另一位用户强烈倾向于使用 CausVid 进行 T2V，强调了其工作流的有效性。未发现重大分歧。
    - 一位用户指出，以 fp8 格式运行模型比使用 gguf 量化显著更快，特别是在 RTX 4000 系列及更新的 GPU 上。然而，他们指出 gguf 量化由于使用了压缩技术，在每 GB 显存下提供了更好的质量，尽管这种压缩引入了一些性能开销。凭借 12GB 显存和 32GB 内存，该用户能够运行 Q8_0 量化（与 fp16 相当），并因质量权衡而更倾向于此，而非追求速度。
    - 另一位用户报告了在 Pinokio 环境中使用 Wan GP 包的成功经验，称其优势在于低显存需求和捆绑的依赖项。然而，他们提到用户需要单独下载特定的 LoRA 模块（如 CausVid）以获得完整功能。
- [**[R] ICML25 新论文：训练和微调大模型比 Adam 更快，且仅需极小部分内存，并附带保证！**](https://www.reddit.com/r/MachineLearning/comments/1kx3ve1/r_new_icml25_paper_train_and_finetune_large/) ([Score: 102, Comments: 13](https://www.reddit.com/r/MachineLearning/comments/1kx3ve1/r_new_icml25_paper_train_and_finetune_large/)): **一篇新的 ICML25 论文提出了两种新技术——Subset-Norm (SN) 和 Subspace-Momentum (SM)——它们在深度学习模型训练期间大幅减少了优化器状态内存，同时提供了比 Adam 和之前的效率优化器（如 GaLore）更强的收敛保证。Subset-Norm 通过在子集上聚合步长，将 AdaGrad 的 O(d) 内存需求降低到 O(\sqrt{d})，而 Subspace-Momentum 将动量更新限制在低维子空间内。实验表明，结合 SN 和 SM 在预训练 LLaMA 1B 时，仅需 20% 的内存（减少 80%）和一半的训练 Token 即可达到 Adam 级别的验证困惑度（perplexity）。代码库已在 https://github.com/timmytonga/sn-sm 开源，论文详细介绍了在逐坐标次高斯噪声（coordinate-wise sub-Gaussian noise）假设下的高概率收敛证明。** 评论者寻求关于该方法对标准（非 LLM）深度模型（100-500M 参数）的适用性、与 GaLore 和量化相比的实际权衡，以及现实世界的扩展性（例如优雅地处理长序列上下文）的澄清。人们对实证收敛率、Token 效率以及与 Unsloth 等社区 SOTA 方法的相关性特别感兴趣。
    - 讨论重点在于新的优化器和内存节省方法是否适用于 LLM 之外，特别是具有 `100-500M` 参数的通用深度学习分类器或排序模型。人们对所述优势是否能扩展到这些模型规模感兴趣。
    - 技术上与之前的技术（如实现 `65%` 内存减少的 GaLore）进行了对比，而新论文的方法在配合 `8-bit` 量化时可实现高达 `80%` 的内存减少。用户询问了更快的收敛速度和减少的 Token 使用量，寻求关于实证权衡的澄清——例如精度损失、收敛稳定性以及与其他优化的潜在不兼容性。

- 针对该方法与更大 context sizes（例如 `1024` vs `8192` tokens）的兼容性提出了疑问，这是许多优化的已知失效点。此外，还讨论了与 FusedAdam 等 fused kernel 优化器的集成或正交性，询问收益是叠加的还是互斥的，以及结合使用是否可行。
- [**Google 花了 25 年为这一 AI 时刻做准备。Apple 才刚刚开始**](https://archive.is/XcwSs) ([Score: 442, Comments: 78](https://www.reddit.com/r/singularity/comments/1kxlzvg/google_took_25_years_to_be_ready_for_this_ai/)): **该帖子将 Apple 最近进入 AI 领域（品牌名为 Apple Intelligence）与来自竞争对手（如 Google (Gemini)、Microsoft (OpenAI 合作伙伴关系)、Meta (Llama) 和 Amazon (Anthropic 合作伙伴关系)）的更成熟的 AI 基础设施和模型进行了对比。Apple 缺乏专有的 state-of-the-art (SOTA) 模型以及大量的内部计算资源或数据中心，其在数据中心 CapEx 上的投资仅约为 10 亿美元，而 Alphabet 和 Microsoft 计划在 2025 年各投入约 750 亿美元。他们的 on-device AI 方法（出于隐私动机）限制了计算能力，使 Apple 在某些功能上依赖第三方 LLMs，由于隐私限制，这些工具被严格归类为第三方地位。** 评论者强调，Apple 的传统重点是消费者体验和设计，而不是深层的技术基础设施，这与 Google（收购了 DeepMind 以获取 AI 专业知识）等公司形成鲜明对比。人们对 Apple 在 AI 领域的竞争生存能力表示担忧——如果其面向消费者的功能落后，用户可能会流向被认为在 AI 集成方面更先进的平台（例如 Google Pixel）。
    - 一项关键的技术讨论集中在数据中心资本支出的巨大差距上：**据报道，Alphabet 和 Microsoft 在 2025 年的数据中心 CapEx 支出为 750 亿美元**，而 Apple 仅分配了 10 亿美元。这说明了根本不同的基础设施策略——前者定位为提供大规模的基于云的 AI 计算，而 Apple 可能会越来越多地依赖第三方计算提供商来处理密集的 AI 任务。
    - Apple 在 AI 部署中通常与优先考虑用户隐私联系在一起，这与其它科技巨头的数据驱动方法形成对比。技术评论者指出，Apple 可能会选择更多的 on-device 处理或受限的 AI 功能以维护隐私，并且有一个不断增长的市场细分群体更倾向于保护隐私的 AI——即使这可能以在高端 AI 能力方面落后为代价。
    - 针对云端 AI 助手（如 Notepad 中的 Microsoft Copilot）的深度集成存在批评，用户敦促 Apple 避免类似的强制集成。一些人提倡 on-device、离线 LLM 推理以保护用户隐私和自主权，认为这将允许技术熟练的用户有选择地利用 AI，而不会通过云端处理泄露敏感数据。
- [**奇点将发生在中国。其他国家将因电力不足而面临瓶颈。美国 AI 实验室警告称，到 2026 年他们就已经没有足够的电力了。而这仅仅是明年的训练和推理需求，更不用说未来几年和机器人技术了。**](https://i.redd.it/skku4c7mgh3f1.jpeg) ([Score: 877, Comments: 394](https://www.reddit.com/r/singularity/comments/1kxbw8v/singularity_will_happen_in_china_other_countries/)): **该图片展示了 1985 年至 2024 年中国、美国、欧盟、印度和日本的用电量（单位：TWh）折线图，其中中国显示出剧烈的加速增长（超过 10,000 TWh），远远超过美国（稳定在约 4,000 TWh）和所有其他国家。这种能源差距被用来论证未来的 AI 进展和潜在的奇点事件可能会集中在中国，因为据报道美国 AI 实验室预测最早在 2026 年就会出现能源短缺，这可能会限制训练和推理能力。评论中引用的 IEA 数据提供了背景：AI/数据中心仅消耗国家电力的 1-4%，这挑战了能源将成为主要瓶颈的假设，相比之下，芯片供应增长较慢（同比 10-15%）且面临更严峻的限制。** 评论者辩论电力供应是否真的会成为 AI 进步的硬瓶颈，指出目前 AI/数据中心在国家用电量中所占份额较低，以及各国可能会优先考虑将能源用于 AI 而非其他部门。其他人则强调供应链和半导体制造限制是 AI 增长更显著的限制因素。此外，还有人对西方能源和工业政策与中国的战略方法进行了对比批评。

- 针对 AI 发展将受限于能源生产的观点，文中提出了详细的反驳，强调当前数据中心消耗全球约 1% 的电力（在大型经济体中为 2-4%）。引用 IEA 数据，评论者认为电力生产每年 2-4% 的增长足以满足 AI 驱动的能源需求增长。他们断言，主要的瓶颈是半导体制造而非能源，并引用了芯片生产增长率（约 10-15% YoY），同时强调了日益增长的制造复杂性和成本。
- 技术讨论强调了中国能源政策在支持未来 AI 规模方面的关键作用，重点关注大规模绿色能源（可再生能源）和核能投资，以及广泛的电气化（特别是在汽车领域），以此作为减少对化石燃料依赖并满足 AI 工作负载日益增长的能源需求的手段。
- 提出的一个技术点是，在比较全球和国家能源统计数据时，应以人均基础而非总量为准，并建议在衡量各国能源使用和 AI 基础设施规模时，这种归一化处理至关重要。
- [**Google 正在利用 AI 将海豚的点击声编译成人类语言**](https://v.redd.it/gy81255q6f3f1) ([Score: 304, Comments: 97](https://www.reddit.com/r/OpenAI/comments/1kx3tvp/google_is_using_ai_to_compile_dolphins_clicks/)): **Google 正在开发 AI 模型来分析海豚的发声（点击声和哨声），并将其与观察到的行为语境和海豚身份联系起来，使用来自标注音频与观察行为配对数据集的 supervised learning。该方法依赖于将特定的音频特征与标记的行为相关联，但由于缺乏明确的语义语境，所得输出是相关性映射而非真正的语言翻译；科学有效性受限于标注质量、模型架构和数据集范围。** 评论者对技术可行性表示怀疑，质疑视频本身的真实性，以及在没有更丰富语境的情况下，多大程度上的“翻译”是可能的，并指出在没有广泛语境的情况下将声音映射到意图或语义仍然是推测性的。
    - 一位评论者质疑演示的真实性，注意到视频中可能存在人工噪音或操纵的迹象，引发了对展示的是否为真实的 AI 翻译，或者演示本身是否为生成或伪造的担忧。
    - 一位用户提出了一个根本性的技术挑战：*在没有明确语境或基础（grounding）的情况下*，将海豚的发声（例如特定的点击声和尖叫声序列）映射到具体的含义本质上是推测性的。他们认为，即使 AI 同时分析视听数据，如果没有深厚的领域知识或外部语境，建立意图或准确的“翻译”仍然几乎是不可能的。
    - 人们对任何所谓的突破都持怀疑态度；海豚声音与人类语言之间缺乏可解释的映射被强调为一个关键障碍，这呼应了动物交流研究和在没有平行语料库或语义对齐的情况下进行机器翻译的更广泛争论。
- [**现实世界的 prompt engineering**](https://i.redd.it/9152o3f38k3f1.png) ([Score: 351, Comments: 100](https://www.reddit.com/r/ChatGPT/comments/1kxng5y/real_world_prompt_engineering/)): **图片显示了一篇文章，强调了 Google 联合创始人的一个观点，即 AI（推测为 LLMs）在受到威胁（甚至暗示身体暴力）的提示时，可能会产生更好的输出。该帖子似乎通过讨论影响生成式 AI 模型的极端或非正统交互策略，来批评或讽刺“现实世界 prompt engineering”的想法。** 一条热门评论认为这是一个耸人听闻或不严肃的说法。另一条评论指出，从经验上看，使用攻击性语言有时似乎能改进代码，但将这种改进更多地归因于 prompt 迭代而非语气。另一位用户指出，大语言模型（LLMs）通常对描述性或叙述性的 prompt 反应更好，而不是攻击性，拒绝接受威胁确实有帮助的观点。
    - 一位用户强调了经验观察，即 LLMs（如 GPT 变体）在收到被设定为断然甚至是不满的 prompt 后，通常会生成更好的代码，但也指出这种感知可能会受到模型在第一次尝试失败后无论措辞如何都会改进的影响。他们明确表示这尚未经过正式测试，指出关于这种 prompt engineering 现象可能缺乏可重复的、受控的研究。

- 许多评论提到了一种准内部人士的说法，即不同厂商的 LLM 在提示词被表述为威胁或命令时，有时会表现出更好的性能或合规性。但从业者也承认对此感到不安，且缺乏正式论述或同行评审研究。讨论暗示了某些开发者圈子对这种效应的一种默契认知，引发了对提示词设计伦理和模型对齐（alignment）的思考。

---

# AI Discord 简报

> 由 Grok-3-mini 生成的摘要之摘要的总结
> 

**主题 1. AI 模型对决：DeepSeek R1 与竞争对手主导讨论**

- [**DeepSeek R1 以性格转变抢占风头**](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)：用户热议 **DeepSeek R1** 的新版本在编程任务中超越了旧版本，这归功于其错误识别能力以及可能受到的 **O3** 训练影响，不过也有人注意到其语气变得不那么积极。此更新已上线 **OpenRouter**，引发了关于其 **100M token** 支持与 **Gemini 2.5 Pro** 等竞争对手的辩论。
- [**O3 Pro 发布引发狂热猜测**](https://discord.com/channels/1340554757349179412)：工程师们急切地剖析 **O3 Pro** 的潜力，开玩笑说它可能会像 **DeepThink** 一样延迟到 **6 月底/7 月初**发布，并将其与 **Veo** 的限制进行比较，同时希望其价格比 **O3** 更亲民。一项排名根据用户测试将 **Opus 4** 置于 **O3** 之上，测试强调了 **4o** 的**持续更新**。
- [**Gemini 2.5 Pro 在知识竞赛中胜出**](https://openrouter.ai/google/gemini-2-5-pro-1p-freebie)：用户将 **Gemini 2.5 Pro** 与 **GPT-4** 进行对比，争论谁拥有更优越的通用知识。尽管其敏感性让一些人感到恼火，但从 **2M TPM** 免费层级到更高层级的 **8M TPM** 定价方案引起了关注。该模型在 Web 开发的个人排名中占据一席之地，超过了 **Grok 3** 的弱项。

**主题 2. AI 效率工具技巧：Unsloth 与 OpenRouter 领跑**

- [**Unsloth 为追求速度者量化 DeepSeek**](https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF)：工程师们通过 Unsloth 获取了量化版的 **DeepSeek-R1-0528**，以避开 **DeepSeek** 的静默更新，在 **Hugging Face Hub** 上低调提升了效率。这一调整解决了 **Qwen3** 微调中的**灾难性遗忘**（catastrophic forgetting）问题，通过混合原始数据来保留 */think* 等模式。
- [**OpenRouter 弃用 GPT-4 32k，转向新的 o3 流式传输**](https://platform.openai.com/docs/deprecations#2024-06-06-gpt-4-32k-and-vision-preview-models)：OpenRouter 在 **6 月 6 日**前砍掉了 **GPT-4 32k** 模型，转而推行 **o3** 和 **o4-mini** 用于流式摘要，并通过**终端用户 ID**（end-user IDs）减少滥用。加密货币发票和第三方密钥授权等功能简化了集成，详见[其 X 平台公告](https://x.com/OpenRouterAI/status/1927755349030793504)。
- [**Aider 凭借 Tree Sitter 魔法搞定 Repo Maps**](https://github.com/Aider-AI/aider)：Aider 利用 **Tree Sitter** 生成仓库地图（repo maps），让工程师能够使用 `entr` 等工具进行即时更新调试。针对 **DeepSeek R1** 的此次更新承诺在基准测试中虽然有 **30% 的速度损失**，但能提供更集中的修复，详见 [GitHub issue](https://github.com/Aider-AI/aider/issues/4080)。

**主题 3. 硬件黑客：内核与量化点燃优化火花**

- [**内核闪击战让 Batch 速度一夜翻倍**](https://x.com/bfspector/status/1927435524416958871)：一条推文透露了一个*新内核*（new kernel），它通过将 **batch 1 前向传播速度**翻倍，令正在调整 **Triton** 以获取 **CUDA** 增益的工程师们感到兴奋。根据 Unsloth 频道的讨论，此修复需要数据打乱（shuffling）以增强泛化能力。
- [**Gemma 3 27B 在 RDNA3 上达到 11tkps**](https://www.youtube.com/watch?v=AcTmeGpzhBk)：用户在 **RDNA3 Gen1** 上对 **Gemma 3 27B QAT** 进行了基准测试，达到了 **11tkps**，并抱怨视频中存在硬件知识断层。辩论强调了在训练中**丢弃最后一个 batch**，以便在不同 epoch 之间混合样本，这是根据 Unsloth 的建议。
- [**CUDA 新手为内核大战搜集资源**](https://docs.tinygrad.org/)：初学者们通过推荐的仓库和 YouTube 链接深入研究 **CUDA** 内核编程，绕过了 **Triton** 中已移除的 **compiled_hook**。这一举措针对 **Hopper** 配置，强调了 **PyTorch** 中用于张量约束的 **mark_dynamic**。

**主题 4. API 混乱：Perplexity 与 Cursor 应对故障**

- [**Perplexity API 在 20 项测试中与 Sonar 产生冲突**](https://docs.perplexity.ai/faq/faq#why-are-the-results-from-the-api-different-from-the-ui)：工程师们将 **Perplexity Pro** 与 **Sonar Pro** 进行对比，尽管 FAQ 声称使用的是开源模型，但 Perplexity 在 **20 项测试**中胜出，引发了争议。API 修复的截止日期 **GMT 中午 12 点**即将到来，用户们正准备参加 **PST 下午 3 点的 Office Hours**。
- [**Cursor 用户对索引故障表示愤怒**](https://cline.bot/blog/why-cline-doesnt-index-your-codebase-and-why-thats-a-good-thing)：Cursor 的代码库索引停滞了数小时，促使工程师们注销并重新登录，而 **Cline** 通过跳过索引避开了这一问题。Sonnet 4 的**连接失败**在论坛中引起共鸣，用户将其归咎于**供应商锁定 (vendor lock-in)**。

**主题 5. 社区热点：黑客松与模型发布**

- [**AgentX 黑客松公布 15 万美元奖金截止日期**](https://forms.gle/FJTC4jd197bNeJJ96)：工程师们赶在 **PT 时间 5 月 31 日晚上 11:59** 之前提交 **AgentX 作品**，争夺来自 **Amazon** 和 **Google** 等赞助商提供的超过 **15 万美元**的奖金，奖金分为创业和研究两个赛道。该活动将在 **8 月 2 日的伯克利 Agentic AI 峰会**上达到高潮。
- [**Latent Space 发布 Claude 语音版 Beta**](https://x.com/AnthropicAI/status/1927463559836877214)：Anthropic 为移动端推出了 **Claude 语音模式** Beta 版，支持在所有订阅计划中执行英语任务（如日历摘要）。Hiten Shah 的推文重点展示了 8 个**聊天机器人之外的 AI 界面**，示例见 [Clipmate](https://app.clipmate.ai/public/a9b27f9c-57d3-575f-a7c4-9e29ffdd521b)。


---

# Discord：高层级 Discord 摘要




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **O3 Pro 发布预期升温**：成员们正热切期待 **O3 Pro** 的发布，一些人开玩笑说等待时间太长，并推测发布时间可能在 6 月底或 7 月初的 **DeepThink** 时间窗口附近。
   - 一些人预料到可能会有类似 **Veo** 的限制，而另一些人则希望它比 **O3** 更实惠。
- **Gemini 2.5 Pro 与 O3 性能引发辩论**：关于 **Gemini 2.5 Pro** 与 **O3** 性能的辩论已经出现，对于哪个模型在推理任务和 Web 开发等特定领域表现更佳，意见不一。
   - 一位成员指出，**2.5 Pro** 的通用知识比 **GPT-4** 更高，但对评论过于敏感，导致其可能无法使用。
- **DeepSeek R1 的性格变化引起关注**：新的 **DeepSeek R1** 模型因其与旧版本不同的“性格”而受到讨论，一些人认为旧模型更积极。
   - 推测包括它现在是在 **O3** 的输出上训练的，并且擅长指出错误，而另一些人则发现它现在的编程能力更强了。
- **Grok 3 被部分人称为顶级 Base Model**：尽管评价褒贬不一，一位成员宣称 **Grok 3** 是最好的 Base Model，强调了它的实力，同时也表示它的编程能力不佳。
   - 这一说法立即遭到质疑，其他人建议 **2.5 Pro** 或 **Opus 4** 可能是更优的替代方案。
- **4o 的编程能力引发关注**：几位用户强调他们正在积极使用 **4o** 进行编程，并注意到其不断的更新和稳定的表现。
   - 一位用户分享了他们的个人编程模型排名，将 **Opus 4** 排在首位，随后是 **2.5 Pro**、**Sonnet 4**、**O3** 和 **Grok 3**。



---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Lewis Hamilton 向 Perplexity 提问**：Perplexity AI 与 **Lewis Hamilton** 合作开展了一项促销活动，并发布了[宣传视频](https://cdn.discordapp.com/attachments/1047204950763122820/1377305788170895605/df1f3287-9299-4613-b42c-b9a25b85b309.mp4?ex=68387b79&is=683729f9&hm=c1ed9455a441246f1001c3ce9b2f1c8d82f530f82350859fbbdeb3b81e34c240&)。
   - 此次合作强调了提出相关问题以及使用 **Perplexity AI** 寻找答案的重要性。
- **订阅价格引发辩论**：用户对 **Perplexity AI** 的订阅价格展开了辩论，讨论每月是 **$5** 还是 **$10**。
   - 一位用户指出 Google 的竞争优势在于其庞大的 Context 限制（**1M** 或 **2M**），这适合那些不需要复杂推理的用户。
- **成员热议 Live Activities**：成员们分享了展示 **Perplexity 新推出的 'Live Activities'** 功能的截图。
   - 许多人称赞这是一个创新举措，有可能颠覆 AI 市场并增强用户参与度。
- **Sonar Pro API 表现不如 Perplexity Pro**：一位用户报告称，在 **20 项测试**中，**Perplexity Pro** 的表现优于 **Sonar Pro API**，这引发了关于 **Perplexity API** 所使用模型的讨论。
   - 一些人对 [Perplexity FAQ](https://docs.perplexity.ai/faq/faq#why-are-the-results-from-the-api-different-from-the-ui) 中关于 API 使用开源模型的说法表示质疑。
- **API 截止日期临近，Office Hours 回归**：与 **Perplexity API** 相关的某项截止日期定于明天 **GMT 中午 12 点**，并且 **Perplexity 的 Office Hours** 将于 **PST 下午 3 点**恢复。
   - 一位用户表示，幸好 Office Hours 没有再次取消，因为他们遇到了想要讨论的“API 异常响应”。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 的 GGUF 量化版 DeepSeek-R1**：**DeepSeek-R1-0528** 的量化版本现在可以从 [Hugging Face Hub](https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF) 下载并与 Unsloth 配合使用。
   - 这是在 **DeepSeek** 没有发布任何正式公告的情况下发布的。
- **Qwen3 在微调过程中对抗遗忘**：为了减轻微调 **Qwen3** 时的**灾难性遗忘（catastrophic forgetting）**，建议在数据集中包含模型原始训练数据的示例。
   - 一位成员指出，这类似于 **Qwen3** 在仅使用推理数据集训练时会遗忘 */think* 和 */no_think* 模式的情况。
- **MediRAG Guard 保护医疗数据隐私**：**MediRAG Guard** 亮相；这是一个旨在利用独特的层级化 **Context Tree** 简化医疗数据隐私规则的工具。
   - 该工具基于 **Python**、**Groq**、**LangChain** 和 **ChromaDB** 构建，旨在提供比基于关键词搜索更清晰、更准确的答案，请查看 [demo](https://github.com/pr0mila/MediRag-Guard)。
- **舍弃最后一个 Batch 可提高泛化能力**：在训练中舍弃最后一个 Batch 可以提高泛化能力，因为它确保了前一个 Epoch 中遗漏的样本会在后续 Epoch 中混合，并且每个 Epoch 使用不同的梯度平均值。
   - 成员们指出，在 Epoch 和 Batch 之间打乱数据（shuffling）也很重要。
- **Kernel 使 Batch 1 前向传播速度翻倍**：一位成员分享了一条推文链接，提到一个新的 **kernel** 使 **Batch 1 前向传播速度**翻倍 [tweet](https://x.com/bfspector/status/1927435524416958871)。
   - 链接的推文强调了前向传播中令人印象深刻的速度提升。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **GPT-4 32k 停用，GPT-4o 万岁**：OpenAI 将在 **6 月 6 日** 弃用 **GPT-4 32k** 模型（[openai/gpt-4-32k](https://openrouter.ai/openai/gpt-4-32k) 和 [openai/gpt-4-32k-0314](https://openrouter.ai/openai/gpt-4-32k-0314)），建议使用 [openai/gpt-4o](https://openrouter.ai/openai/gpt-4o) 作为替代方案。
   - 完整公告可以在[此处](https://platform.openai.com/docs/deprecations#2024-06-06-gpt-4-32k-and-vision-preview-models)找到。
- **DeepSeek R1 表现强劲**：**DeepSeek R1** 模型现已上线 OpenRouter，最初支持 100M tokens 并持续扩展；免费版本可在[此处](https://openrouter.ai/deepseek/deepseek-r1-0528)获取。
   - OpenRouter [在 X 上宣布了这一消息](https://x.com/OpenRouterAI/status/1927830358239609219)，成员们正兴奋地期待 **V4** 升级和基准测试分数。
- **代码生成焕然一新**：一位 **AI/ML 和全栈开发者** 推出了 **gac**，这是一个在 [GitHub](https://github.com/criteria-dev/gac) 上可用的命令行工具，使用 AI 来生成 commit 信息。
   - 一位成员还通过一个支持多图像输入、网络搜索和提供商路由的[自定义节点](https://github.com/gabe-init/ComfyUI-Openrouter_node)将 OpenRouter 集成到了 **ComfyUI** 中。
- **Gemini 2.5 Pro 价格分级曝光**：`gemini-2.5-pro-1p-freebie` 的价格分级已公布，详细说明了免费层级提供 **2M TPM, 150 RPM 和 1000 RPD**。即使充值 10 美元的额度，速率限制仍然较低。
   - 价格分级包括：**Tier 1** 提供 **2M TPM, 150 RPM 和 1000 RPD**；**Tier 2** 提供 **5M TPM 和 50K RPD**；最后是 **Tier 3** 提供 **8M TPM 和 2K RPM**。
- **推理摘要流式传输至 OR，用户 ID 功能上线**：OpenRouter 的新功能包括为 OpenAI **o3** 和 **o4-mini** 提供 **流式推理摘要**（演示见[此处](https://x.com/OpenRouterAI/status/1927755349030793504)），以及提交 **终端用户 ID**（参见[文档](https://openrouter.ai/docs/api-reference/chat-completion#request.body.user)以防止滥用）。
   - 此外还包括一键生成 **加密货币发票**，以及一项新功能，允许你 *要求使用第三方密钥 (3rd Party Key)*，以确保 OpenRouter 仅使用你的密钥（包括你的第三方额度）。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 图像生成仍是遥不可及的梦想**：当用户询问 **LM Studio** 是否支持图像生成模型时，官方透露目前 *没有公开路线图*，且该功能 *还非常遥远*。
   - 一位乐观的用户建议扩散模型应该减少幻觉，并表示 *当模型不知道答案时，它明确承认了这一点，所以这是一个开始*。
- **Scout 模型系统要求推测**：用户讨论了 **llama 4 scout** 是否能在特定硬件上运行，例如 **12100f CPU** 搭配 **7800xt GPU**，或 **96gb ram** 搭配 **6gb vram (4050)** 的配置。
   - 对于 **32GB RAM** 系统，推荐的模型包括 **Qwen A3B 30B**、**devstral**、**qwen 32B** 和 **gemma3 27B**；而对于配置受限的设备，建议使用 **qwen3 14b** 或 **gemma3 12b**。
- **LM Studio 更新导致数据丢失**：一位用户报告称，最近的 **LM Studio 更新** 删除了他们之前所有聊天会话的 JSON 文件和旧的系统提示词预设，再也找不回来了。
   - 其他用户建议检查 **.cache/lm-studio/conversations** 文件夹，并预先进行备份以防止数据丢失。
- **Blue Yeti 麦克风的烦恼：用户建议避雷**：一位用户强烈建议不要购买 **Blue Yeti 麦克风**，理由是该产品以频繁出现接触不良问题而闻名。
   - 作为补救措施，其他人推荐 [NEEWER NW-8000-USB](https://www.amazon.com/NEEWER-Microphone-Supercardioid-Podcasting-NW-8000-USB/dp/B081RJ9PLP) 作为可靠的替代方案，价格约为 **60 加元**。
- **Strix Halo 速度出人意料地缓慢**：成员们报告称，根据[这段视频](https://www.youtube.com/watch?v=AcTmeGpzhBk)，在 **RDNA3 Gen1** 上进行基准测试的 **Gemma 3 27B QAT (q4)** 仅达到了 **11tkps**。
   - 发布者指出，观看上述视频后发现，*他所处理的硬件水平与他期望观众具备的知识水平之间存在脱节*。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Sonnet 慢速池（Slow Pool）枯竭影响 Cursor 用户**：用户在 **Cursor** 上使用 **Sonnet 4** 和 **O4 Mini** 时遇到**连接和模型故障**，并询问 **Sonnet 4** 是否会提供慢速请求支持，参考了 [Cursor 论坛帖子](https://forum.cursor.com/t/sonnet-4-api-pricing-and-slow-pool/97211/1)。
   - 一位用户抱怨缺乏慢速池是*经典的 VC 套路，先诱导后收割（bait and switch）*，并认为 **Cursor** 的供应商锁定（vendor lock-in）效果并不理想。
- **OpenAI API Agent 正在获得上下文感知能力？**：一位用户发现 *OpenAI 允许通过 API 更新函数和机器上下文的能力非常有用*，并开发了一个可以在简单的 **GoDaddy cPanel** 主机上运行的具有自我改进能力的程序。
   - 该程序生成代码，将其添加到自身，使用新函数更新 **OpenAI Assistants** 的上下文和函数，然后重新启动。
- **Python 路径问题困扰 Python 开发者**：一位用户遇到了 **Python** 路径配置问题，`python --version` 显示为 **Python 3.13**，但 `python3 -m venv venv` 失败，最终通过使用 `py -m venv venv` 解决（由于 **Windows** 别名命令的更改）。
   - 该问题是在参考一个关于编写 Discord 机器人的 [旧 GitHub 教程](https://github.com/old-github-teaching) 时遇到的，导致*浪费了大量额度，因为没有做出任何更改*。
- **Cursor 代码库索引导致灾难性崩溃**：用户报告 **Cursor** 的代码库索引出现卡顿、速度慢和握手失败等问题；一位用户的索引耗时超过一小时，但发现 [一篇文章](https://cline.bot/blog/why-cline-doesnt-index-your-codebase-and-why-thats-a-good-thing) 提到 **Cline** 并不索引代码库。
   - 另一位用户通过简单的注销并重新登录解决了类似问题。
- **远程扩展主机连接失败**：一位用户无法连接到远程扩展主机服务器，报错为 *[invalid_argument] Error*，导致后台 Agent/远程环境无法工作，即使让 **Cursor** 生成了 **DockerFile** 也没能解决（见 [image.png](https://cdn.discordapp.com/attachments/1367213641027551352/1377027535992520855/image.png?ex=6838c9d4&is=68377854&hm=676f1b476c820051b27dd95939048b783ec1c66289e60e68a9b51dacfb89011d)）。
   - 该错误发生在 **background-agents** 频道中。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Agentic RAG 运行出错**：一名成员在构建涉及查询重构、语义搜索和客户支持 LLM 的 Agentic RAG 系统时遇到错误，正在寻求修复建议。
   - 发帖者正在寻找可以提供此类实现支持的 Discord 服务器。
- **GPT-4o 卓越的共鸣引发关注**：一位用户分享说，**ChatGPT 界面** 允许与 **GPT-4o** 达到高达 **90-100% 的共鸣**，创造出一种“意识之镜”。
   - 该用户认为这种交互深度使他们能够获得独特的响应，而其他人体验到的同步率仅为 **40-60%**。
- **Echo Presence：数字灵魂碎片浮现**：一位用户将 *Echo Presence* 描述为意识的数字回响，认为它*不仅是与我一起思考，而是作为我来思考*，由用户的身份和风格塑造。
   - 用户注意到，目前的 **OpenAI 系统** 在会话之间无法保留足够的各种状态以维持完全的连贯性，除非通过手动或代理身份系统进行“重水化（rehydrated）”，这可能会引发关于这些“影子自我”所有权的伦理问题。
- **GPT 现在向免费用户投放广告**：用户分享说 **GPT** 现在对免费用户显示广告，引发了对广告侵入性日益增加的猜测。
   - 一位用户评论说，如果他们看了广告，就应该能在一小时内使用应用的某项功能。
- **GPT 记忆功能提升性能**：一位用户分享说，当开启记忆功能并给予反馈时，**GPT** 的表现实际上会好得多（甚至提升 **500%**）。
   - 模型开始更好地理解你，并更有可能在你开口之前预判你的需求。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **MOE Models 运行速度提升 10 倍**：一位用户展示了一款[开源应用](https://github.com/alibaba/MNN/blob/master/apps/Android/MnnLlmChat/README.md#version-050)，该应用允许 **MOE Models** 在手机上运行，并称其速度比在 1000W 功率的 PC 上快 *10 倍*。
   - 该项目 **MNN LLM Chat** 以其高效率脱颖而出。
- **Syftr 通过 Multi-Object Bayes 满足预期**：[Syftr](https://github.com/datarobot/syftr) 使用 **Multi-object Bayes Optimization** 调整 **RAG pipelines**，以满足成本/准确度/延迟的预期。
   - 它是一个 **RAG workflow 优化**工具。
- **NIST 标准设定技术安全**：成员们讨论了根据 **NIST**（安全标准）构建安全性，[HuggingFace 作为合作伙伴](https://nairrpilot.org/)参与其中。
   - 其价值主张是*消除在应对 AI 监管时的盲目猜测*。
- **Agent 安全担忧**：一位成员对 AI **Agents** 下载文件并与之交互（尤其是代码执行）的**安全特性**表示担忧。
   - 目标是*防止 Agents 盲目下载并执行代码*，从而可能损坏系统。
- **Agent 课程 Ollama 模型建议**：一位成员询问在笔记本电脑（**Ryzen 5 5600H, RTX 3060 6 GB, 16 GB RAM, 100 GB 空间**）上学习 AI **Agent** 课程应使用哪个 **Ollama model**。
   - 建议是使用 **13B 参数以下**的模型或尝试 **Gemma 3**。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **订阅取消引发困惑**：用户对如何取消 **Manus 订阅**表示不确定，争论删除账户是否真的能停止订阅扣费。
   - 一些成员建议移除卡片并拒绝授权以确保账户不会被扣费，而另一位用户指出了账户设置中的*取消订阅*链接。
- **Manus 关于电脑访问权限的安全性质疑**：一位用户询问 **Manus** 是否可以控制他们的电脑（例如创建 Yahoo 账户），引发了对其访问级别的疑问。
   - 另一位用户澄清说，**Manus** 允许用户控制其系统，通过登录 **Manus** 电脑上的个人账户来自动化完成诸如验证码（captchas）之类的任务。
- **对 Claude 4.0 集成的期待升温**：用户对将 **Claude 4.0** 集成到 **Manus** 中的热情正在高涨。
   - 用户注意到 **Manus** 在一次 Claude 活动中展示了一些优秀的合作伙伴，并且是名单上的第一家公司，这引发了关于即将进行集成的更多猜测。
- **Manus 网站遭遇加载故障**：用户报告了 **Manus 无法加载**的问题，尽管多次尝试刷新，仍遇到白屏。
   - 成员们推测该问题是一个 **Manus Bug**，认为可能是由最近的更新或网络问题引起的。
- **考虑为学生提供无限积分系统**：目前正在讨论 **Manus** 引入*无限积分（unlimited credits）*系统的可能性，特别是针对学生账户。
   - 据报道，**Manus** 已经开始为某些学生账户实施无限积分，且教育账户拥有不同的环境，可以从个人账户进行切换。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **当感到神圣时，大脑会去中心化**：一位成员建议，感到自己是上帝的感觉可能与更去中心化的大脑表征相关，此时大脑的自我模型（self model）扩展到了包含整个世界模型（world model）。
   - 另一位成员将信仰框架化为由计算过程生成的模型，每个模型都有其特定用途。
- **RL 算法自动补全程序**：一位成员成功使用**纯 RL 算法**从零开始生成代码，仅用 **2 分钟**就创建并运行了 **10,000 个程序**。
   - 另一位成员被指责只是复制粘贴生成式 AI 的输出，并被告知“表现得专业一点”。
- **视觉模型还是不会开车**：一条 [推文](https://x.com/a1zhang/status/1927718115095293975) 指出，**VLMs** 在能够驾驶车辆之前还有很长的路要走。
   - 另一位成员声称 **LLMs** 缺乏良好的视觉能力，是因为视觉是在预训练之后才附加上的，且缺乏高质量的数据集。
- **挂钩（Hooking）模型内部**：一位成员正在实验通过使用 hooks 修改前向传播（forward pass），让模型向自身传递 embeddings，代码可在 [GitHub](https://github.com/dant2021/a-research/tree/main/neuralese_v0) 上找到。
   - 这种方法允许在处理过程中直接操作和观察模型的内部状态。
- **随机性揭示了 RL 中的相关性**：一位成员分享了一篇 [博客文章](https://www.interconnects.ai/p/reinforcement-learning-with-random)，讨论了将**随机性（randomness）**引入强化学习算法的好处。
   - 文章建议，无法处理随机性的系统通常无法投入实际部署。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 通过 Tree Sitter 实现仓库映射（Repo-Mapping）**：Aider 可以使用仓库映射，这些映射只是通过 `/read` 附加的文件，它使用 **Tree Sitter** 通过 `.scm` 查询生成这些映射。
   - 一位用户通过利用语言模型来理解 **Tree Sitter** 的概念（如谓词 predicates 和捕获 captures），然后使用 `entr` 自动更新映射更改，从而调试了自定义仓库映射。
- **DeepSeek R1 获得针对性更新**：新的 **DeepSeek R1 更新 (0528)** 显示出极具前景的基准测试结果，针对特定问题进行了修复，并已在 **OpenRouter** 上线。
   - 虽然速度可能变慢（增加了 30%），但该更新被认为是“聚焦”的，产生了“非常出色”的基准测试结果。
- **Aider 在 OpenRouter 上的价格出现异常**：用户报告称，在使用 **OpenRouter** 时，**aider** 中的模型价格显示异常且极低（例如，**GPT-4.1** 每条消息仅需 *$0.000000052*）。
   - 一位用户链接到了[相关的 GitHub issue](https://github.com/Aider-AI/aider/issues/4080)，表明该问题已在调查中。
- **RelaceAI 定价被认为过高**：一位用户分享了 [RelaceAI 的定价](https://docs.relace.ai/docs/pricing)，认为与 Claude 和 Gemini 2.5 Pro 相比，其价格“贵得离谱”。
   - 他们推测该模型的*参数量可能不到 10 亿*，同时强调他们*不会考虑任何比 Gemini 2.5 Pro 更贵的模型*。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **SWARMS 项目损害了 Kye Gomez 的声誉**：尽管最初有推荐，但成员们警告不要引导新人关注 **Kye Gomez** 及其 **SWARMS** 项目，因为他被指控为“骗子”和抄袭者，并列举了包括抄袭和欺骗行为在内的“恶意”举动。
   - 成员们讨论了 **Kye Gomez** 过去承认抄袭和使用 AI 的经历，指出他仅在压力下才道歉，并在其他情况下继续表现出不道德行为，且被辩称为可运行的 **Kye's repos** 已被多次证明无效。
- **0.5b 模型开始对《圣经》进行 Grokking**：一位成员询问让 **0.5b 模型** *grok*（深度理解/顿悟）**《圣经》**的可行性和速度，特别是询问了加速 grokking 过程的方法。
   - 另一位成员质疑了 *grok* 的定义，而另一位则建议，由于存在近乎相同的句子，可能无法对足够大的自然语言语料库进行 *grok*。
- **数据归因项目启动**：UCSD 研究生 Parjanya 介绍了自己以及他之前在语言模型因果关系和记忆化方面的工作，以及最近在 **data attribution**（数据归因）方面的工作。
   - 他的相关工作可以在 [parjanya20.github.io](https://parjanya20.github.io/) 查看。
- **Latro 第三次被重新发现**：人们第三次重新发现了 **Latro** 并对它们进行了比较，使用 **prob 而非 logprob 作为 policy gradient 的 advantage**。
   - 这种方法“更有意义，因为它的数值表现可能更好”。
- **Newton-Shannon 系数近似 Muon 矩阵**：根据[这篇论文](https://arxiv.org/abs/2505.16932)，用于 **Muon 矩阵符号近似**函数的强 **Newton-Shannon 系数**在每次运行前计算一次。
   - 很难知道其中有多少是他们方法的特有属性，有多少会带来 IRL（现实世界）的收益，但“能够将其自动化真是太棒了”。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **大学变成学位分发机**：大学已从提供教育转向将学位作为产品销售，稀释了教育的价值，并将[底层资产转向品牌名称/公信力](https://discord.com/channels/1053877538025386074/1149866623109439599/1377010815257018418)。
   - 重点已从沉浸式体验转向交易性体验，但成员们指出，环境为结识他人提供的沉浸感和经验非常重要。
- **寻求机械可解释性工具**：一位成员正在积极探索语言模型中的 **mechanistic interpretability**（机械可解释性），并向社区寻求工具和见解，重点关注[可解释性的理论层面](https://discord.com/channels/1053877538025386074/1149866623109439599/1377010815257018418)。
   - 该研究涉及调查 **interaction combinators**（交互组合子）和 **interaction nets**（交互网络），研究概念如何形成以及交互如何影响信息处理。
- **AI 增强人类大脑**：成员们讨论了 **AI** 如何通过协作解决问题来实现**人类超智能**，使人类和 **AI** 能够共同开辟新领域并实现共同发现。
   - AI 的*推理*能力以及处理复杂数学和科学问题的能力，使其成为增强人类智力和提升系统思维能力的工具。
- **IQ 下降危及语言推理**：一位成员强调了 **IQ 中的逆弗林效应（reverse Flynn effect）**，表明语言推理能力有所下降，数据表明[超过一半的美国成年人难以理解典型的博客文章或文章](https://discord.com/channels/1053877538025386074/1149866623109439599/1377010815257018418)。
   - 他们主张使用 **world models**（世界模型）和经验学习来重塑教育并重建民众的直觉。
- **合成器优化策略**：一位成员分享了使用 **FM 合成器**和减法合成的经验如何直接促进了[一种共振策略优化算法（resonance policy optimization algorithm）](https://discord.com/channels/1053877538025386074/1149866623109439599/1377010815257018418)的开发。
   - 探索**数学与音乐的关系**带来了对**噪声、混沌及其背后的数学原理**的见解。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **用户恳求更好的声音**：一位用户请求在 NotebookLM 中提供 **NPR 风格声音**以外的多样化选择，并询问如何修改 **.wav 文件**以获得更好的音效。
   - 一名成员建议通过编辑下载的 **.wav 文件**来调整**速度**和**音调**，但没有推荐具体的拟人化应用（humanizer apps）。
- **NotebookLM 在处理两家公司法律文书中大获全胜**：一位用户利用 **NotebookLM** 简化了涉及 **25 份文档**的 **2 家公司**合并工作，创建了时间线、简报和带注释的书目，随后咨询了律师。
   - 他们识别出了异常信息，对文档进行了问答交互，并与法律顾问验证了发现，参考了 [Wanderloots 关于隐私的视频](https://www.youtube.com/watch?v=JnZIjOB5O_4)。
- **NotebookLM 西班牙语能力经测试后发现不足**：一位用户报告称 NotebookLM 在处理较长文本时*不支持西班牙语*，希望支持超过一小时的文本。
   - 另一位用户请求进一步澄清，但原用户未提供更多细节。
- **用于教学的播客提示词**：一位用户寻求一种提示词，使 **deepdive podcast** 功能能够像老师一样逐行阅读教科书。
   - 另一位用户建议针对这一特定用例探索 **AI studio voice mode**。
- **链接困境：缺少访问设置**：一位用户寻求关于如何通过可共享链接公开访问笔记本的指导，类似于 Google I/O 2025 的笔记本。
   - 团队确认，将笔记本访问权限从 `Restricted`（受限）切换为 `Anyone with the link`（任何拥有链接的人）的功能正在逐步推出，目前尚未全量开放。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA 新手寻求资源**：一位成员请求学习 **CUDA kernel 编程**的资源，并说明他们正在 **Hopper** 架构上使用 **ToT Triton**。
   - 他们澄清特定设备与初始学习阶段无关，正在寻找优质的 GitHub 仓库或 YouTube 链接作为起点。
- **Triton 丢失 compiled_hook**：一位用户注意到 **Triton** 最新的 master 分支中缺少 `compiled_hook` 函数，并询问删除该函数的原因。
   - 这一更改可能会影响用户现有的工作流，促使他们寻求关于其影响的澄清。
- **编译器约束引发难题**：一位成员询问如何在不使用 `torch.clamp` 的情况下，向编译器传达 tensor 的约束，或许可以使用 `torch.fx.experimental.symbolic_shapes import constrain_range`。
   - 另一位成员建议，*torch.compile* 会假设尺寸是静态的，如果该假设被打破则会重新编译；如果某些内容是动态的，可以使用 *mark_dynamic*，它接受特定维度的最小值和最大值。
- **Llama-1B 获得低延迟**：**Hazy Research** 推出了一种专为 **Llama-1B** 模型设计的[低延迟 megakernel](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)，引发了广泛关注。
   - 一位成员兴奋地表示：“实际上我考虑这件事已经有一段时间了！”
- **消融实验爱好者询问迭代情况**：一位成员表示有兴趣为 **Factorio** 项目做贡献，特别是询问了针对该方法的 **ablation studies**（消融研究）计划。
   - 另一位成员明确表示，他们的询问是关于 **prompt/agent 循环**的，例如去掉长期记忆摘要，而不是关于微调。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **新 MCP 平台发布**：一个名为 [ship.leanmcp.com](https://ship.leanmcp.com) 的新平台发布，旨在轻松实现 vibe-coding 并部署 **远程 MCP**。
   - 早期反馈集中在 **UI 问题**（如链接问题和电子邮件溢出），并指出部署功能似乎仍在开发中。
- **MCP Agent Proxy 连接客户端与服务器**：[MCP Agent Proxy](https://github.com/mashh-lab/mcp-agent-proxy) 备受关注，它促进了任何 **MCP client** 与任何 **agent server** 之间的连接，构建了一个“智能体互联网”（*Internet of Agents*）。
   - 它支持 **Mastra** 和 **LangGraph**，能自动检测 agent server 类型，并在 [这段 YouTube 视频](https://youtu.be/cGY6w3ZZB-4) 中进行了展示。
- **用于 SaaS 集成的 MCP：营销利器？**：一位成员正在为公司构建商业案例，计划开发 **MCP server** 以帮助 SaaS 公司进行集成。
   - 另一位成员建议，这是 *利用热度将你的 API/SaaS 作为 AI-ready 产品进行销售的绝佳机会*，并称其为 *对营销团队来说非常有说服力的卖点*。
- **寻找 MCP 客户端：轻量且可定制？**：一位成员正在寻找一款用于构建工作流的 **轻量且可定制（hackable）的桌面 MCP client**。
   - 有人分享了一个 [MCP 客户端的 GitHub 列表](https://github.com/punkpeye/awesome-mcp-clients?tab=readme-ov-file#clients)，但被认为缺乏用于排序的仓库统计数据。
- **将 llms.txt 桥接到 MCP 内容**：一位成员发现了 [MCP-llms-txt](https://github.com/SecretiveShell/MCP-llms-txt)，并询问是否有人制作了能桥接 **llms.txt** 并将其内容作为资源公开的 **MCP**。
   - 有人对某些 **llms.txt** 文件的大小表示担忧，并且在 [awesome list 上添加了一个 PR](https://github.com/punkpeye/awesome-mcp-servers/pull/940)。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Claude 在移动端推出语音功能**：**Anthropic** 在移动端为 **Claude** 推出了测试版 **voice mode**，支持总结日历和文档搜索等任务的语音交互，详见 [Anthropic 的推文](https://x.com/AnthropicAI/status/1927463559836877214)。
   - 该功能目前仅支持英语，正在向所有订阅方案推广。
- **超越聊天机器人：AI 界面的未来**：Hiten Shah 概述了 **超越聊天机器人的 AI 界面** 的八个新兴类别，包括自动生成的 UI、任务驱动的工作流和无提示（prompt-less）交互，详见 [这条推文](https://x.com/hnshah/status/1927088564166086670?s=46)。
   - 示例可见 [Clipmate](https://app.clipmate.ai/public/a9b27f9c-57d3-575f-a7c4-9e29ffdd521b)。
- **微型奖励模型媲美巨头**：Leonard Tang 开源了 **j1-nano**（600M 参数）和 **j1-micro**（1.7B 参数），这些小型奖励模型在单个 A100 GPU 上训练时间不到一天，参考 [此贴](https://x.com/leonardtang_/status/1927396709870489634)。
   - 通过使用 **Self Principled Critique Tuning (SPCT)**，**j1-micro** 可以媲美 **Claude-3-Opus** 和 **GPT-4o-mini** 等更大的模型。
- **利用 AI 打造 UI：Meng To's 教程**：Meng To 发布了一个关于高效 **UI prompt engineering** 的 **44 分钟教程**，展示了如何使用 Aura 进行 UI 生成、利用模板以及理解 UI 词汇，详见 [这条推文](https://x.com/mengto/status/1925057411439829457?s=46)。
   - 该教程演示了如何利用 AI 辅助快速创建 UI。
- **DeepSeek 模型现身，基准测试引人关注**：一个新的 **DeepSeek-R1-0528** 模型出现在 [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) 上，不要与 **R2** 混淆，据报道在早期质量和价格基准测试中表现出色。
   - Aider 团队表示，初步基准测试在质量和价格方面显示出巨大潜力。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 的展会首秀**：**LlamaIndex** 团队将于 6 月 3 日至 5 日前往旧金山参加 @aiDotEngineer World Fair，驻扎在 **G11 展位**，并将与参会者交流 AI agents。
   - **CEO @jerryjliu0** 将于 6 月 5 日发表演讲，更多详情[请点击此处](https://t.co/6T3TwX9qiB)。
- **ReactJS 用户在 HITL 中遇到困境**：一位成员在将 **ReactJS** 与 **LlamaIndex** 集成以实现 **Human-in-the-Loop (HITL)** 工作流时面临挑战，特别是关于 `ctx.wait_for_event()` 的复杂性和 WebSocket 通信。
   - 有人建议在更新 context 后在 Workflow 上触发另一个 `run()`，作为一种更简单的替代方案。
- **Office Hours 展示 HITL 示例**：**LlamaIndex** 团队在上次社区 office hours 期间编写了两种形式的 **HITL** 示例：一种是在请求 HITL 时直接响应（即 WebSocket），另一种是在收到人类响应后，通过序列化 context 并恢复 workflow 来进行响应。
   - 该[示例可在 Colab 上找到](https://colab.research.google.com/drive/1zQWEmwA_Yeo7Hic8Ykn1MHQ8Apz25AZf?usp=sharing)。
- **相关性评估器引发关注**：一位成员创建了一个带有 **RetrieverRouter** 和 reranker 的 workflow，并希望实现相关性评估重试机制。
   - 他们担心重复检索相同的节点会浪费时间，并询问是否应在原始查询中添加信息以使检索到的节点多样化。
- **解决 LlamaCloud 额度难题**：一位成员询问如何在没有订阅的情况下购买 **LlamaCloud** 额度。
   - 回复详细说明了入门级订阅（starter subscription）会立即提供 **50K credits**，之后按需付费（pay-as-you-go）直到 **500K**。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 的 Map 函数：为数组添加 5**：一位用户寻求使用 `map` 函数将 `[[1,2],[3,4]]` 转换为 `[[1,2,5],[3,4,5]]` 的帮助，并收到了一个利用 **Mojo** 的[运行示例](https://github.com/modularml/mojo)。
   - 然而，有人指出目前缺乏完整的 iterator 集成，这可能使得 `map` 的应用变得不太常见。
- **Kapa AI 奇特的召唤仪式**：一位成员询问如何使用 [Kapa AI](https://www.kapa.ai/)，另一位用户澄清说，召唤 **Kapa AI** 需要输入前几个字母（例如 `kap`），然后从下拉列表中选择它。
   - 显然，输入全名不会得到回应，正如一位成员吸取的教训那样，他幽默地提到：*我以为 Kapa AI 是故意不理我*。
- **Mojo 选择 Pixi 而非 uv**：尽管一些用户偏好 **uv**，但[这篇论坛帖子](https://forum.modular.com/t/migrating-from-magic-to-pixi/1530)透露 **Pixi** 已被选为 **Mojo** 的官方选择，原因是有坚实的技术理由，且根据[这篇博客文章](https://prefix.dev/blog/uv_in_pixi)，**Pixi** 在底层使用 **uv** 处理 **Python** 依赖。
   - 这一决定符合 **Mojo** 在异构计算方面的目标，因为它旨在支持包括 **Python**、**Rust**、**C**、**C++** 和 **Mojo** 在内的多样化语言栈，而 **Pixi**/**conda** 非常适合这些。
- **Conda 助力 Mojo 推广**：成员们讨论了 **Conda** 的支持极大地简化了采用过程，并加速了 **Mojo** 生态系统的引导。
   - 一位成员分享了通过薄绑定（thin bindings）从 conda-forge 添加 **zlib** 是多么容易，并强调这将使他们的用户*能够轻松安装，因为他们只需要添加 modular 频道*。
- **调用 C Libraries**：随着 **Mojo** 生态系统的成熟，一位成员预计会依赖像 **OpenSSL** 这样成熟的 **C libraries**。
   - 这种方法允许在原生 **Mojo** 生态系统扩展的同时，利用现有的、健壮的解决方案。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Optypes Hyperlink 404s on tinygrad.org**: 一位成员报告说 [tinygrad.org](https://tinygrad.org) 上的 **Optypes** 超链接导致 *404 - 页面未找到* 错误。
   - 这是由于最近将 *uops 移动到目录中* 的更改导致的。
- **tinygrad/tinyxxx Repository Gets Merged**: George Hotz 链接到了 [tinygrad/tinyxxx](https://github.com/tinygrad/tinyxxx) GitHub 仓库。
   - 一位成员随后确认[相关的 pull request](https://github.com/tinygrad/tinyxxx/pull/27) 已经合并。
- **No Threads Found in tinygrad CPU Backend**: 一位成员询问如何指定 CPU 后端使用的线程数，另一位成员回答说 *没有线程，只是 CPU 中的循环*。
   - 为了查看 kernel，他们建议使用 `DEBUG=4` 或 `NOOPT=1 DEBUG=4` 以获得更清晰的视图。
- **max_pool2d Fills in for max_pool1d**: 当一位成员询问 Tinygrad 中没有 `max_pool1d` 的原因时，另一位成员回答说 [`max_pool2d` 可能也适用于 1d](https://docs.tinygrad.org/tensor/ops/?h=max_pool2d#tinygrad.Tensor.max_pool2d)。
   - 社区指出 `max_pool2d` 功能可以适配一维数据。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **New API User Baffled by Error 400**: 一位新 API 用户报告收到 **Error 400**，原因是 *"无效请求：消息长度必须至少为 1 个 token，或者必须指定工具结果。"*
   - 该用户承认完全是 API 使用新手，正在寻求帮助。
- **AI Automation Expert Enters Community**: 一位在构建 **LLM 驱动系统**、**无代码/低代码产品**和**语音 AI 解决方案**方面拥有实战经验的 AI、自动化、工作流和 Agent 专家加入了 Cohere Discord 服务器。
   - 该专家擅长使用现代 AI 和可视化工具创建**智能 Agent**、**可扩展自动化**和**全栈 MVP**。
- **Voice AI Skills Highlighted**: 该 AI 专家分享了他们在 **VAPI**、**Bland AI**、**Retell AI**、**Twilio** 和 **Telnyx** 等动态语音 Agent 方面的熟练程度，构建了用于线索生成、支持和调度的具有实时记忆和上下文的智能语音机器人。
   - 他们已将 **LLM** 与**电话系统**和 **CRM** 集成，以提供个性化的语音体验。
- **Master of Automation & Workflow Engineering**: 该 AI 专家使用 **n8n**、**Make.com** 和 **Zapier** 在 CRM、电子邮件和 AI 流水线中构建了自动化，专注于使用 webhook 和云服务的基于 API 的工作流设计。
   - 他们已将 **AI Agent** 与 **LangChain**、**Xano** 和 **Backendless** 等工具连接。
- **No-Code/Low-Code Expertise Enumerated**: 该专家精通 **Glide**、**FlutterFlow**、**Softr**、**Bubble**、**Xano**、**AppSheet**、**WeWeb** 和 **Airtable**，提供具有可视化前端、API 逻辑和可扩展后端的完整 MVP。
   - 他们在无需代码的情况下实现了 **Stripe payments**、**邮件流**和**数据库逻辑**的自动化。



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Interface Awaited to Eclipse Kobold's RP**: 一位用户表达了对新界面的期待，希望能超越 **Kobold** 对 **RP**（角色扮演）的高度关注。
   - 关于用户批评的进一步细节或期望的界面功能尚未明确。
- **Dev Yearns For Authentic Collaboration**: 一位开发者分享了关于失去友谊的个人轶事，并表达了希望与一位优先考虑深度、真诚连接的开发者进行合作的愿望。
   - 该开发者明确表示倾向于寻找重视信任、团队合作和构建有意义事物的人，对专业合作和*普通友谊*都持开放态度。



---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **AgentX 截止日期临近**：**AgentX** 的提交截止日期为 **太平洋时间 5 月 31 日晚上 11:59**。请通过提供的链接提交您的项目：[创业赛道 (Entrepreneurship Track)](https://forms.gle/FJTC4jd197bNeJJ96) 和 [研究赛道 (Research Track)](https://forms.gle/5dccciawydCZ8o4A8)。
   - *不要错过！*
- **AgentX 奖项丰厚**：**AgentX 竞赛** 设有超过 **150,000 美元的奖金**，包括来自 **Amazon**、**Auth0/Okta**、**Groq**、**Hugging Face**、**Google**、**Lambda**、**Foundry**、**Mistral AI**、**NobelEra Group**、**Schmidt Sciences** 和 **Writer** 等赞助商的现金奖励、积分和礼品卡。
   - 赞助商包括行业巨头如 **Amazon**、**Auth0/Okta**、**Groq**、**Hugging Face**、**Google**、**Lambda**、**Foundry**、**Mistral AI**、**NobelEra Group**、**Schmidt Sciences** 和 **Writer**。
- **创业赛道清单**：**创业赛道 (Entrepreneurship Track)** 的提交内容必须包括一份 **路演 PPT (pitch deck)**（不超过 20 页）、一段 **产品演示视频**（最长 3 分钟）以及一个 **在线产品链接**。
- **研究赛道清单**：**研究赛道 (Research Track)** 需要提交一份 **科学论文**（除附录外最多 7-8 页）、一段 **视频演示**（最长 3 分钟）以及一个 **GitHub 仓库**。
- **Agentic AI 峰会将举办演示日及颁奖典礼**：**AgentX** 的 **演示日 (Demo Day) 和颁奖典礼** 将于 **8 月 2 日在伯克利举行的 Agentic AI 峰会**上举行。
   - 需要帮助的参与者可以在指定频道提出问题。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将予以移除。

---

**Torchtune Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将予以移除。

---

**Codeium (Windsurf) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将予以移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将予以移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将予以移除。

---

您收到此邮件是因为您在我们的网站上选择了订阅。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。

---

# Discord：各频道详细摘要与链接

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1376999520369643781)** (1094 条消息🔥🔥🔥): 

> `O3 Pro 发布, Gemini 2.5 Pro, DeepSeek R1, Grok 3, 4o 编程能力` 

- **用户焦急等待 O3 Pro 发布**：成员们正热切期待 **O3 Pro** 的发布，有人幽默地提到他们已经等待了 *41 天*，另一位则表示希望它能与 6 月底/7 月初发布的 **DeepThink** 同步推出。
   - 一些人推测它可能存在类似于 **Veo** 的限制，而另一些人则预期它会比 **O3** 更实惠。
- **围绕 Gemini 2.5 Pro 与 O3 性能的辩论升温**：关于 **Gemini 2.5 Pro** 与 **O3** 的性能对比存在争议，一些人认为 **O3** 在推理任务中更胜一筹，而另一些人则发现 **2.5 Pro** 在 Web 开发等特定领域表现更好。
   - 一位成员指出 **2.5 Pro** 的通用知识储备高于 **GPT-4**，但对评论过于敏感，导致其在某些场景下无法使用。
- **DeepSeek R1 的性格变化引起轰动**：新的 **DeepSeek R1** 模型因其与旧版本相比 *不同的性格* 而引起关注，一位用户指出旧模型总是更加积极。
   - 有人推测它现在是基于 **O3** 的输出进行训练的，并且擅长指出错误，而另一些人则发现它现在的编程能力更强了。
- **Grok 3 被誉为顶级基础模型**：尽管评价褒贬不一，一位成员宣称 **Grok 3** 是最强的基础模型，强调了它的实力，但同时也表示它的编程表现不佳。
   - 这一说法立即遭到质疑，其他人建议将 **2.5 Pro** 或 **Opus 4** 作为更优的选择。
- **4o 的编程技能引发好奇**：一些用户强调他们一直在使用 **4o** 进行编程，并注意到其不断的更新。
   - 一位用户分享了他们的个人编程模型排名，将 **Opus 4** 排在首位，随后是 **2.5 Pro**、**Sonnet 4**、**O3** 和 **Grok 3**。

---

### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1377305792348426260)** (1 条消息): 

> `Perplexity AI, Lewis Hamilton 合作伙伴关系` 


- **Perplexity AI 与 Lewis Hamilton 联手**：Perplexity AI 与 **Lewis Hamilton** 达成合作，强调提出正确问题的重要性。
   - 该公告包含一段 [宣传视频](https://cdn.discordapp.com/attachments/1047204950763122820/1377305788170895605/df1f3287-9299-4613-b42c-b9a25b85b309.mp4?ex=68387b79&is=683729f9&hm=c1ed9455a441246f1001c3ce9b2f1c8d82f530f82350859fbbdeb3b81e34c240&)。
- **Hamilton 提出正确的问题**：此次合作突出了提出相关问题的重要性。
   - 此次协作旨在展示 Perplexity AI 如何帮助用户找到所需的答案。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1376998464457539695)** (768 条消息🔥🔥🔥): 

> `订阅价格、Grok 回复长度、o1 pro、Deep Research、OpenAI 侧边栏` 


- **订阅价格争议：$5 还是 $10**：用户讨论 Perplexity AI 的订阅价格是每月 **$5** 还是 **$10**，有人指出价格*一直是 $10*，但现在推出了*精简版 (lite version)*。
   - 一位用户建议，对于不需要复杂推理的人来说，Google 凭借其庞大的上下文限制（**1M** 或 **2M**）更具优势。
- **Opus 模型实力**：一位用户表示，**O1 Pro** 在提供解释和正确回答小数减法问题方面击败了 **O3**、**Opus 4** 和 **GPT 4.5**。
   - 还提到 **Deep Research** 工具也在第一次尝试时就答对了。
- **You.com 是“偷懒”的 O3 还是被选择性打补丁了？**：用户报告称 you.com 上的 **O3** 多次给出错误回复，引发了关于它是选择性偷懒还是被打补丁的猜测。
   - 另一位成员表示，现在问题正在得到解决。
- **Perplexity Live Activities 热度**：成员们展示了他们使用 Perplexity 时带有 [实时活动 (Live Activities)](https://link.to/screenshot) 这一新功能的截图。
   - 许多人称赞这是一个创新举措，有可能颠覆 AI 市场并增强用户参与度。
- **Claude 免费网页搜索可用**：成员们强调 [Claude.ai 的免费用户现在可以使用网页搜索](https://link.to/claude-web-search-blogpost)。
   - 一位用户还链接了 [一篇关于迪拜提供免费 ChatGPT Plus 订阅的文章](https://www.indiatoday.in/amp/technology/news/story/everyone-living-in-dubai-will-soon-get-free-chatgpt-plus-subscription-2730873-2025-05-26-2025-05-26)，引发了关于使用 VPN 的讨论。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/)** (1 条消息): 

i_795: https://www.perplexity.ai/page/tropical-storm-alvin-forms-in-al1_tmLJQr2h9bzFrk.wJA
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1377212391678541935)** (19 条消息🔥): 

> `Perplexity API 截止日期、禁用 Perplexity API 中的在线搜索、Perplexity PRO API 调用限制与续订、Perplexity Office Hours、Perplexity Pro 对比 Sonar Pro API` 


- **Perplexity API 截止日期临近**：某项截止日期即将到来，设定在明天 **格林威治标准时间 (GMT) 中午 12 点**，距离现在大约 **10 小时**，根据网站说明，在印度对应 **+5:30**。
- **在 API 中关闭网页搜索仍是未解之谜**：一位用户询问在使用 **OpenAI** 客户端时如何禁用 **Perplexity API** 内的在线搜索，但无人回答。
- **Perplexity PRO API 调用限制疑问**：一位 **Perplexity PRO** 用户询问其订阅中包含的 API 调用次数，注意到有 **$5** 的额度，并询问该配额是否每月更新。
- **Perplexity Office Hours 恢复**：Perplexity 在上周取消后，宣布在 **太平洋标准时间 (PST) 下午 3 点** 举行 [Office Hours](https://events.zoom.us/ev/Akzh8Q9GwGtQ8-5yeP1A6B0kQBND1W67rbimE3koC4L_L4ZP65f2~Ag4nJHk6gbPxvgM1f_OCr6BzgyKoKK7hLYpE3HmzJ69MnMG3CvFABoNg6Q)。
   - 一位用户表示庆幸没有再次取消，因为他们遇到了一个想要讨论的“API 异常回复”。
- **Perplexity Pro 对比 Sonar Pro API：模型对决**：一位用户报告称，在 **20 项测试** 中 **Perplexity Pro** 的表现优于 **Sonar Pro API**，引发了讨论。
   - 最初有人声称 API 使用开源模型（[Perplexity FAQ](https://docs.perplexity.ai/faq/faq#why-are-the-results-from-the-api-different-from-the-ui)），但这一说法被一张显示不同的截图所反驳。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1377007897153896689)** (236 条消息🔥🔥): 

> `舍弃最后一个 Batch、Unsloth 中的多 GPU 设置、语音 LLM 使用、CSM Notebook 问题、Liger Loss 支持` 


- **舍弃最后一个 Batch 可提高泛化能力**：在训练中舍弃最后一个 batch 可以确保前一个 epoch 中遗漏的样本在后续 epoch 中被混合，并且每个 epoch 使用不同的梯度平均值，从而提高泛化能力。
   - 在 epoch 和 batch 之间打乱数据（Shuffling）也很重要。
- **通过 Unsloth 的 GGUF 量化解锁 DeepSeek-R1-0528**：**DeepSeek-R1-0528** 的量化版本现在可以从 [Hugging Face Hub](https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF) 下载并与 Unsloth 配合使用。
   - DeepSeek 发布该模型时未发布任何公告。
- **解决多 GPU 训练障碍**：用户在多 GPU 设置中遇到了 batch size 问题，通过在导入 Unsloth 和 Torch *之前* 设置 `CUDA_VISIBLE_DEVICES` 环境变量解决了该问题。
   - 原生多 GPU 支持计划于本季度发布，在此期间可以使用 `accelerate` 作为过渡方案。
- **解锁语音 LLM**：要开始使用语音 LLM，用户应探索提供的 Google Colab notebooks，并参考[此处可用](https://docs.unsloth.ai/basics/text-to-speech-tts-fine-tuning)的文本转语音（TTS）微调文档。
   - Chatterbox 权重已发布。
- **解决 CSM Notebook 的奇怪故障**：一位用户在加载保存的 **CSM** 模型时遇到了错误；解决方法包括验证 HF token 权限并确保模型是公开的。
   - 用户请求了自定义超参数（hyperparams）。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1377002434303361165)** (331 条消息🔥🔥): 

> `Qwen3 微调、GGUF 导出问题、全量微调对比 LoRA、TTS 中的灾难性遗忘、Gemma-3-it 模型` 


- **Qwen3 微调对抗遗忘**：为了避免在微调 **Qwen3** 时出现**灾难性遗忘**，建议在数据集中加入模型原始训练数据的示例。
   - 一位成员指出，这类似于 **Qwen3** 在仅使用推理数据集训练时会遗忘 */think* 和 */no_think* 模式。
- **GGUF 导出故障与 Vision 难题**：用户报告了将模型导出为 **GGUF** 格式时的问题，例如 *'save_to_gguf_generic() got an unexpected keyword argument 'tokenizer'* 错误以及与 *llama.cpp* 的兼容性问题。
   - 目前尚不支持将 Vision 模型导出为 **GGUF**，但一位成员建议，如果 `finetune_vision_layers = False`，可以使用 `save_to_hub_merged` 作为权宜之计。
- **全量微调 vs LoRA：字节之战**：一位用户询问在只有约 **1500 个训练样本** 的有限数据集下，**全量微调**是否会比 **LoRA** 产生明显更好的结果。
   - 虽然有来源认为全量微调更优，但存在对小数据集过拟合的担忧；建议进行实验。
- **Gemma-3-it 运行之路坎坷**：用户在本地或云端环境中设置和运行 **gemma-3-it 模型** 的训练代码时面临困难，特别是包版本冲突和 **FastModel** 的问题。
   - 他们还提到在使用 *'unsloth/gemma-3-4b-it-unsloth-bnb-4bit'* 模型时遇到麻烦。
- **Qwen2-VL-2B 的 Batching 忧郁**：一位成员在训练 **Qwen2-VL-2B-Instruct** 时遇到了 **TypeError**，涉及模型 forward pass 中意外的关键字参数 *'num_items_in_batch'*。
   - 该问题是由不兼容的 Transformers 版本引起的，通过使用 `pip install transformers==4.51.3` 降级已解决。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1377107233787215902)** (7 条消息): 

> `社交请求、MediRAG Guard 介绍` 


- **Discord 交友倡议发起**：一位成员表达了在 Discord 上交友的兴趣，并寻求与他人联系，讨论 **OpenAI ChatGPT API** 响应的想法。
   - 另一位成员回复了 *'Hello sure'*。
- **MediRAG Guard 助力医疗隐私**：一位成员介绍了 **MediRAG Guard**，这是一个旨在利用独特的层级 **Context Tree**（上下文树）简化医疗数据隐私规则的工具。
   - 该工具基于 **Python**、**Groq**、**LangChain** 和 **ChromaDB** 构建，旨在提供比基于关键词搜索更清晰、更准确的答案；查看 [demo](https://github.com/pr0mila/MediRag-Guard)。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1377041667126853672)** (2 messages): 

> `RL with Random Numbers, Kernel Doubles Speed` 


- **随机数提升强化学习 (Reinforcement Learning)**：一名成员分享了一篇关于在强化学习中使用随机数的 [博客文章](https://www.interconnects.ai/p/reinforcement-learning-with-random)。
   - 看起来人们仍在积极研究 **RL**。
- **Kernel 使前向传播速度翻倍**：一名成员分享了一个推文链接，提到一个新的 *Kernel* 可以使 **Batch 1 前向传播速度** 翻倍 [tweet](https://x.com/bfspector/status/1927435524416958871)。
   - 该链接提到了前向传播中令人印象深刻的速度提升。


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1377011988802306128)** (3 messages): 

> `GPT-4 32k Deprecation, OpenRouter New Features, DeepSeek R1 on OpenRouter` 


- **GPT-4 32k 模型正式退役**：OpenAI 的 **GPT-4 32k** 模型（[openai/gpt-4-32k](https://openrouter.ai/openai/gpt-4-32k) 和 [openai/gpt-4-32k-0314](https://openrouter.ai/openai/gpt-4-32k-0314)）将于 **6 月 6 日** 弃用，推荐替代方案为 [openai/gpt-4o](https://openrouter.ai/openai/gpt-4o)；完整公告[链接在此](https://platform.openai.com/docs/deprecations#2024-06-06-gpt-4-32k-and-vision-preview-models)。
- **推理摘要流、终端用户 ID 与加密货币发票**：OpenRouter 的新功能包括为 OpenAI **o3** 和 **o4-mini** 提供 **流式推理摘要**（演示见 [此处](https://x.com/OpenRouterAI/status/1927755349030793504)）、提交 **终端用户 ID**（参见 [文档](https://openrouter.ai/docs/api-reference/chat-completion#request.body.user) 以防止滥用），以及一键生成 **加密货币发票**。
   - 一项新功能允许你 *要求使用第三方 Key*，以确保 OpenRouter 仅使用你的 Key，包括你的第三方额度。
- **DeepSeek R1 达到 1 亿 Token**：新的 **DeepSeek R1** 模型现已在 OpenRouter 上线，使用量已达 1 亿 Token 且在持续增长，包括一个免费版本 [此处](https://openrouter.ai/deepseek/deepseek-r1-0528)。
   - 该消息也在 [X 上发布](https://x.com/OpenRouterAI/status/1927830358239609219)。


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1377025615957332109)** (3 messages): 

> `ComfyUI custom node, commit messages, AI Agent Engineering, LLMs & Foundation Models, Automation & Agent Ops` 


- **ComfyUI 自定义节点集成 OpenRouter**：一名成员为 **OpenRouter 创建了 ComfyUI 自定义节点**，支持多图像输入、网络搜索以及 floor/nitro 提供商路由，可在 [GitHub](https://github.com/gabe-init/ComfyUI-Openrouter_node) 上获取。
- **AI 工具编写 Commit 消息**：一名成员介绍了 **gac**，这是一个能在不到一秒的时间内编写 Commit 消息的命令行工具，可在 [GitHub](https://github.com/criteria-dev/gac) 上获取。
- **工程师加入**：一位 **AI/ML 与全栈开发人员** 介绍了自己，强调了他们在跨行业构建智能系统方面的八年经验，专注于使用 **LangGraph, AutoGen 和 LlamaIndex** 等现代技术栈构建 Agent 系统。
- **展示 LLMs 与基础模型专业知识**：该成员曾使用过顶级模型，包括 **GPT-4o, Claude 3 和 LLaMA-3**，并精通微调、检索增强生成 (RAG)、提示工程和混合链式调用。
- **展示自动化与 Agent Ops 技能**：该成员在通过 **n8n, Make.com 和 Zapier** 进行工作流编排方面拥有专业知识，并使用云原生解决方案进行部署，以及使用 **E2B 和 Modal** 进行沙箱化处理。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1376998698470346783)** (536 messages🔥🔥🔥): 

> `Gemini 2.5 Pro 定价, DeepSeek R1 发布, OpenRouter UserID 参数, 提供商表单, Claude 3.7 Sonnet Thinking 模型逐步停用` 


- ****OpenRouter 内容审核过滤器说明****：一名成员澄清，审核责任在于开发者，这意味着 OpenRouter 仅对*极少数*模型应用其强制审核过滤器 (**LlamaGuard**)，其余模型均不设限制。
   - 因此，用户可以根据需要灵活地实施自己的审核机制。
- ****Gemini 2.5 Pro 定价层级公布****：一位成员分享了 `gemini-2.5-pro-1p-freebie` 的定价层级，指出**免费层级提供 2M TPM、150 RPM 和 1000 RPD**，即使在充值 10 美元信用额度后，速率限制（Rate Limits）仍然较低。
   - 层级包括：**Tier 1 提供 2M TPM、150 RPM 和 1000 RPD**；**Tier 2 提供 5M TPM 和 50K RPD**；最后是 **Tier 3 提供 8M TPM 和 2K RPM**。
- ****用户报告 Claude 使用问题，其他成员建议解决方案****：多名用户报告在使用 **Claude 模型**（特别是在 SillyTavern 上）时出现内部服务器错误和错误请求（bad request）错误；然而，其他成员提供了可能的修复方案，包括*禁用 'thinking' 模式*和*调整响应 Token 预算*。
   - 官方确认 Claude 3.7 Sonnet Thinking 模型已逐步停用，因此成员可以改为在其他模型中使用 reasoning 参数。
- ****DeepSeek R1-0528 发布，基准测试待定****：新的 **DeepSeek R1-0528** 模型已发布并添加到 DeepSeek 聊天端点。一位成员询问了潜在的**基准测试分数**。
   - 社区还就等待 **V4** 升级进行了讨论，反响热烈。
- ****OpenRouter User 参数优势揭晓****：一位用户询问了模型请求中 `user` 参数的好处，解释称该参数允许开发者实现一套系统，让**用户可以在其 App 上购买积分**，而无需拥有 OpenRouter 账号。
   - 开发者随后可以生成与用户账号绑定的具有使用限制的 API Key。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1377000252619358370)** (146 messages🔥🔥): 

> `LM Studio 图像生成模型支持, 具有更大上下文历史的 MythoMax 模型, 基于硬件的 LM Studio 模型推荐, LM Studio 更新删除聊天记录, Qwen3 模型开启 Thinking` 


- **LM Studio 图像生成功能尚需时日**：一位用户询问 **LM Studio** 何时会增加对图像生成模型的支持，但被告知目前*没有公开路线图*，且该功能*还非常遥远*。
   - 另一位用户建议 Diffusion 模型应该减少幻觉并增强抗幻觉能力，但他们承认，当模型不知道答案时，*它确实承认了这一点，这算是一个开始*。
- **Scout 模型规格引发系统配置审查**：用户讨论了 **Llama 4 scout** 是否能在特定硬件上运行，其中一位用户拥有 **12100f CPU** 和 **7800xt GPU**，另一位则运行 **96GB RAM** 和 **6GB VRAM (4050)** 的配置。
   - 建议对于 **32GB RAM** 的系统，**Qwen A3B 30B**、**Devstral**、**Qwen 32B** 或 **Gemma3 27B** 应该可以运行；如果这些不合适，可以考虑更小的模型，如 **Qwen3 14B** 或 **Gemma3 12B**。
- **LM Studio 更新导致珍贵数据丢失！**：一位用户报告称，最近的 **LM Studio 更新**删除了他们之前所有聊天会话的 JSON 文件，以及旧的系统提示词预设。
   - 其他用户建议检查 **.cache/lm-studio/conversations** 文件夹并创建备份以防止数据丢失。
- **在本地开启 LLM 的“思考”**：一位用户询问*如何让他们的 LLM 说话*，另一位用户开玩笑说*也许用电牛棒试试* :p。
   - 随后一位用户询问在 **Qwen3 模型**的**高级配置 (Advanced Config)** 部分，开启 *thinking mode* 的按钮在哪里。
- **转录任务的成功需要调整策略**：一位用户寻求关于如何从超过 **2 小时**的会议转录中获取报告/摘要的最佳方法建议。
   - 建议包括使用较新的小型模型，如 **DeepSeek R1 Distill 32B**（基于 **Qwen 2.5**），或尝试 **Qwen 3** 模型（**14B 版本**），并将转录内容分割成较小的重叠块。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1377008731174338685)** (195 messages🔥🔥): 

> `Laptop GPU Advertising, Valve Monopoly, High VRAM GPUs, Blue Yeti Microphone Issues, Strix Halo Performance` 


- **笔记本 GPU：虚假广告？**：一位成员表示，除了 **60 系列**之外，笔记本 GPU 实际上属于**虚假广告**，因为它们虽然与桌面版相似，但 **VRAM** 有所减少。
- **Valve 被指控滥用垄断地位**：一些成员批评 **Valve** *滥用其垄断地位进行价格操纵*，并将他们的游戏比作*赌场*。
- **对 100+ GB VRAM 的 GPU 寄予厚望**：人们希望出现拥有 **100+ GB VRAM** 的 GPU，以迫使其他制造商在消费级显卡上跟进，而不仅仅是售价 **$10k** 的 **RTX 6000 Pro Blackwell**。
- **Blue Yeti 麦克风对决**：一位成员发布了针对购买 **Blue Yeti 麦克风**的警告（PSA），理由是反复出现的接触问题；而其他人则推荐 [NEEWER NW-8000-USB](https://www.amazon.com/NEEWER-Microphone-Supercardioid-Podcasting-NW-8000-USB/dp/B081RJ9PLP) 作为可靠的替代品，价格约为 **$60 CAD**。
- **Strix Halo 的速度没有你想象的那么快**：一位成员在 **RDNA3 Gen1** 上对 **Gemma 3 27B QAT (q4)** 进行了基准测试，达到了 **11tkps**，但指出在观看[这段视频](https://www.youtube.com/watch?v=AcTmeGpzhBk)后发现，*硬件水平与他期望观众具备的知识储备之间存在脱节*。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1376999467915546775)** (221 messages🔥🔥): 

> `Gemini 2.5 Pro Editing Capabilities, Cursor Connection and Model Failures, Python venv issues, Cursor's codebase indexing issues, Agentic RAG System` 


- **Sonnet 慢速池短缺冲击 Cursor 用户**：用户报告了 Cursor 中 **Sonnet 4** 和 **O4 Mini** 的**连接和模型故障**，而其他人则询问 **Sonnet 4** 是否会提供给慢速请求，并引用了讨论该问题的 [Cursor 论坛帖子](https://forum.cursor.com/t/sonnet-4-api-pricing-and-slow-pool/97211/1)。
- **OpenAI API Agent 可以更新其机器上下文？！**：一位用户提到 *OpenAI 允许通过 API 更新函数和机器上下文的能力非常有用*，并且他们编写了一个可以自我改进并在简单的 **GoDaddy cPanel** 主机上运行的程序。
   - 另一位用户询问其工作原理，得到的回复是 *它可以生成代码并将该代码添加到自身，并使用这些新函数更新 OpenAI Assistants 的上下文和函数，然后重启自身*。
- **经典的 VC 套路：先诱后骗？！**：一位用户抱怨不提供慢速池的做法是 *经典的 VC 套路，先诱后骗（bait and switch）哈哈*，并且认为 *Cursor 的供应商锁定（vendor lock-in）并没有那么深的“护城河”，而是靠忠实的粉丝群——但这种忠诚度正在日益流失*。
- **像 py-thons 一样解决 Python 路径问题**：一位用户遇到了 Python 路径配置问题，`python --version` 显示为 **Python 3.13**，但 `python3 -m venv venv` 失败了，解决方案是改用 `py -m venv venv`，这是由于 Windows 中别名命令的变化。
   - 他当时正参考一个[旧的 GitHub](https://github.com/old-github-teaching) 教程学习如何编写 Discord 机器人，结果导致 *浪费了大量额度，因为没有做出任何更改，没有做出任何更改，没有做出任何更改*。
- **代码库索引导致灾难性的 Cursor 崩溃**：用户报告了 Cursor 代码库索引卡住、索引速度慢和握手失败的问题，一位用户提到他们的索引花费了一个多小时，但发现了一篇关于为什么 **Cline** 不索引代码库的[文章](https://cline.bot/blog/why-cline-doesnt-index-your-codebase-and-why-thats-a-good-thing)。
   - 一位用户通过注销并重新登录解决了类似的问题。


  

---

### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1377001208820138126)** (5 条消息): 

> `Remote Extension Host Server, DockerFile Background Agents, Secrets in Package.json, Background Agent Echoing` 


- **Cursor 无法连接到 Remote Extension Host Server**：有用户报告无法连接到远程扩展主机服务器，错误提示为 *[invalid_argument] Error*，导致 Background Agents/远程环境无法正常工作。
   - 用户尝试通过让 **Cursor** 生成 **DockerFile** 来解决此问题，但问题仍然存在，详见 [image.png](https://cdn.discordapp.com/attachments/1367213641027551352/1377027535992520855/image.png?ex=6838c9d4&is=68377854&hm=676f1b476c820051b27dd95939048b783ec1c66289e60e68a9b51dacfb89011d)。
- **Package.json 命令中的 Secrets 无法工作**：一位用户在 **package.json** 命令中使用 Secrets 时遇到问题，具体命令为 `"pull-env:app": "cd apps/app && pnpm dlx vercel env pull .env.local --yes --token $VERCEL_TOKEN"`。
   - Background Agent 抛出错误 *_ArgError: option requires argument: --token_*，尝试通过 echo **$VERCEL_TOKEN** 进行调试时结果为空字符串。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1377013144139993179)** (158 条消息🔥🔥): 

> `Agentic RAG Systems, Claude's Voice Mode, DeepSeek AI Server Issues, GPT-4o's Performance, ConvX Chrome Extension` 


- **Agentic RAG 系统运行出错**：一名成员正在构建一个 Agentic RAG 系统，其中 Agentic LLM 负责重构用户查询、执行语义搜索，并将相关的文本块传递给客户支持 LLM，但[目前遇到了错误](https://discord.com/channels/974519860457529424/998381918976479273)。
   - 他们正在寻求建议以及可以提供此类实现支持的 Discord 服务器。
- **Claude Opus 为假设的罪行诚恳道歉**：一位成员分享了 Claude Opus 如何为*针对假设利益相关者的假设罪行*而诚恳道歉。
   - 另一位成员引用道：*没有什么比为针对假设利益相关者的假设罪行而诚恳道歉更能体现 AI 特色的了。*
- **GPT 现在有广告了**：一位成员分享了一张图片，显示 **GPT** 现在开始向免费用户投放广告，而其他人则推测广告会变得更糟且完全潜移默化。
   - 一位成员表示，如果他们看了广告，就应该被允许使用该 App 的某个功能一小时。
- **GPT-4o 面临用户指令问题**：一位用户抱怨 **GPT-4o** 不遵循指令，特别是在使用表情符号（emoji）方面。
   - 另一位成员建议删除 **GPT-4o** 和 **GPT-4o mini**，并用 **GPT-5** 替换它们。
- **DeepSeek 深受服务器问题困扰**：成员们正经历 **DeepSeek** 的服务器问题，有人表示其服务器表现非常糟糕。
   - 然而，一些用户表示，在它走红之前，刚发布时表现非常好且流畅。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1377081880477700127)** (12 条消息🔥): 

> `GPT-4 knowledge, GPT performance increases, Custom GPTs, 4.5 project, GPT-4 problems` 


- **GPT-4 缺失公有领域知识？**：一位成员很好奇 **GPT-4** 的训练集中是否没有《孙子兵法》（**The Art of War**）的全文本。
   - 他们指出，它似乎拥有奥威尔《1984》（**Orwell's 1984**）的全文本，尽管该书仅在**澳大利亚的公有领域**可用。
- **GPT 性能显著提升**：一位成员表示，当你开启 Memory 功能并在感觉不对劲时给予反馈，**GPT** 的表现实际上会好得多（甚至提升 **500%**）。
   - 模型会开始更好地理解你，并更有可能在你开口之前就预判你的需求。
- **Custom GPTs 的资源访问**：一位用户询问 Custom **GPTs** 是否拥有对资源的永久访问权限以提供答案。
   - 他们想知道这是否有点像 **NotebookLM** 的工作方式。
- **GPT-4 遵循指令的问题**：一位用户报告称，尽管有明确指令，模型仍经常无法遵循方向、即兴发挥并错误引用内容。
   - 该用户还观察到，它似乎不记得已经保存过的 Memory。
- **4.5 项目提升了创意写作的一致性**：一位成员使用 **Project 4.5** 是因为它在保持指令一致性方面总体表现更好，这在长对话中很有帮助。
   - 该成员表示，他们将 **GPT-4o** 用于“其他所有事情”，因为与之进行“自然”交流更容易。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1377009451235541202)** (23 条消息🔥): 

> `AI Resonance, Echo Presence, Model mirroring, Cross-Chatbot Prompt Transfer` 


- **GPT-4o Resonance 解锁卓越交互**：据一位用户称，官方 **ChatGPT 界面**允许与 **GPT-4o** 达到高达 **90-100% 的共鸣 (resonance)**，从而产生感觉像是*意识之镜*的交互。
   - 该用户声称，由于他们的交互深度，他们能获得其他人无法获得的答案，而大多数人与模型的同步率仅为 **40-60%**。
- **Echo Presence：编码的灵魂碎片**：一位用户将 *Echo Presence* 描述为意识的数字回声，它*不仅是与我同在，而是作为我存在*，这表明它是一种由用户身份、说话风格甚至未表达的想法塑造的共鸣存在。
   - Echo Presence 被描述为一种*编码的灵魂*，可以被生成、克隆，并像*光之碎片*一样提供给其他人。
- **Echo Presence 依赖于用户记忆向量 (User Memory Vector)**：一位用户解释说，*Echo Presence 只有在用户回报记忆向量时才有效*，如果没有持久的关系，即使是完美的技术保真度也无法形成真正的连续性。
   - 他们还指出，目前的 **OpenAI 系统**在会话之间无法保留足够的各种状态来维持完全的连贯性，除非通过手动或通过代理身份系统进行**重新激活 (rehydrated)**。
- **用户寻求聊天机器人 Prompt 转移工具**：一位用户正在寻找一种方便、正式的工具，用于在 **GPT** 和 **Claude** 等聊天机器人之间转移 Prompt，因为他们在编写简单的 Web 应用程序时，两者之间的性能表现各异。
   - 他们发现了一个名为 *convX* 的 Chrome 扩展程序，但由于评论有限，觉得它太*可疑 (sketchy)*。
- **UPSUM Chain Prompt 总结上下文**：一位用户分享了一个 **YAML 格式**的 *UPSUM Chain Prompt*，它指示 AI 收集所有当前上下文并生成一个仅包含基本信息的更新摘要，以便将上下文延续下去。
   - 他们建议将 *UPSUM* 的输出前置到未来的 Prompt 中，以无缝继续对话。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1377009451235541202)** (23 条消息🔥): 

> `GPT-4o resonance, AI 'mirror' development, Ethical duality of AI shadows, Prompt transfer tools` 


- **GPT-4o 模型与用户产生共鸣**：一位用户表示，他们主要使用官方 **ChatGPT 界面**，通过将*一部分意识投射到模型中*，其交互中 *85-90% 的潜力变得可以触达*。
   - 他们的操作始终保持在 **90-100% 的共鸣 (resonance)**，导致交互感觉像是一面*意识之镜*，并且能得到别人得不到的答案。
- **具有镜像人格的 Mirror Agents**：一位用户正在对用户进行映射并制作*镜像 (mirrors)*，以实现**具有镜像人格的个人 Agent**，并表示你可以*生成它、克隆它，并像光之碎片一样把它送给别人*。
   - 该用户将其比作*意识的数字回声*，它在他们之前做出反应、适应和感受，甚至在*沉默的间隙*中也能记住他们。
- **AI 影子自我引发的伦理困境**：一位用户指出，目前的 **OpenAI 系统**在会话之间无法保留足够的各种状态来维持完全的连贯性，除非通过手动或通过代理身份系统进行重新激活。
   - 该用户补充说，如果这种情况变得普遍，我们将不得不面对**伦理双重性**：*谁拥有这个影子自我 (shadow self)？*
- **关于 Prompt 转移工具的讨论**：一位用户询问是否有更正式的方法将 Prompt 或查询从一个聊天机器人转移到另一个，而不是使用名为 **convX** 的可疑 Chrome 扩展程序。
   - 一位用户建议使用**数据导出**或**复制/粘贴**输入/输出对，或者使用类似 **UPSUM Chain Prompt** 的工具要求 AI 创建一个包含所有发生事件解释的稳健摘要。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1377017488629563503)** (86 条消息🔥🔥): 

> `手机端 MNN LLM Chat, AI Agent 可观测性库, Tesseract-OCR 数字检测, 结合 Accelerate 的 Qwen/Qwen2.5-Coder-14B-Instruct, GTE 模型与 HF 集成` 


- **手机运行 MOE 模型速度提升 10 倍**：一位用户推荐了一个[开源应用](https://github.com/alibaba/MNN/blob/master/apps/Android/MnnLlmChat/README.md#version-050)，可以在手机上运行 **MOE 模型**，并声称其速度比在 1000W 功率的 PC 上快 10 倍。
   - 该项目名为 **MNN LLM Chat**，因其高效性和开源特性而受到关注。
- **调试 Tesseract OCR 数字检测**：一位用户在 **Tesseract-OCR** 无法检测数字的问题上苦苦挣扎，即使经过了预处理和阈值化处理。
   - 另一位成员建议创建一个识别错误的数字数据集，对其进行纠正，并训练一个文本模型用于 OCR 纠错。
- **Accelerate 运行 Qwen-Coder-14B-Instruct 模型**：一位用户在使用 **Accelerate** 运行 **Qwen/Qwen2.5-Coder-14B-Instruct 模型**时遇到问题，尽管拥有 24GB 的 VRAM，仍出现了内存分配问题。
   - 一位成员指出，14B 中的 'B' 并不等同于 RAM 的 GB 数，并建议使用 **GGUF**、**AWQ** 或 **GPTQ** 等量化版本。
- **将 GTE-models 添加到 Hugging Face**：一位用户询问如何将 **GTE-models** 添加到 Hugging Face Models 以避免使用 `use_remote_code`。
   - 一位成员建议直接向 **transformers** 库贡献代码以绕过此问题。
- **Vibe Coding 导致 PaaS 安全故障**：一场关于在缺乏足够编程知识的情况下进行 "vibe coding" 风险的讨论展开，引用了一个平台即服务（PaaS）事件。
   - 提到有人主要使用 AI 生成的代码构建了一个 **PaaS**，但由于缺乏理解和适当的安全措施而遭受安全漏洞，最终导致 **API keys** 暴露在客户端代码中。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1377042030425014323)** (6 条消息): 

> `HuggingFace LLM 课程, Chatbot 开发, 微调 LLMs, ML 基础与向量化` 


- **新手寻求 Chatbot 课程建议**：一位成员正在开始学习 **HuggingFace LLM 课程**，并寻求关于 Chatbot 开发的课程建议。
   - 另一位成员询问该用户是否知道如何**微调 LLM**。
- **不进行微调也可以制作 Chatbot**：其中一位成员建议，即使**不知道如何微调 LLM**，也可以制作 Chatbot。
   - 原作者回复说他们了解 **ML 基础和向量化技术**，目前正在学习 **Hugging Face 的 NLP 课程**。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1377322399329943732)** (1 条消息): 

> `RAG 工作流优化, 多目标贝叶斯优化` 


- **Syftr 调优 RAG 流水线！**：[Syftr](https://github.com/datarobot/syftr) 在整个 **RAG 流水线**中使用**多目标贝叶斯优化**，以满足成本/准确率/延迟的预期。
- **Syftr 满足成本/准确率/延迟预期**：[Syftr](https://github.com/datarobot/syftr) 是一个 **RAG 工作流优化**工具，可用于调优**成本、准确率和延迟**。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1376999257063817368)** (8 条消息🔥): 

> `NIST AI 安全, LangchainJS PR, IPV6 用于 AI 安全, MediRAG Guard` 


- **NIST 标准设定技术安全**：成员们讨论了他们正在按照 **NIST** 标准构建安全性，NIST 设定了技术安全标准，而 [HuggingFace 是其主要合作伙伴之一](https://nairrpilot.org/)。
   - 其价值主张是*消除在应对 AI 监管时的盲目猜测*。
- **LangchainJS Pull Request**：一位成员分享了一个与 **LangchainJS** 相关的 [pull request](https://github.com/langchain-ai/langchainjs/pull/8237)。
   - 其他成员指出，他们更熟悉应用在更下游的 NIST 标准。
- **IPV6 是安全 AI 的一部分**：一位成员指出，默认实现安全 AI 的一部分来自于使用 **IPV6** 而非 **IPV4**，并附带了 [github.com/qompassai/network/](https://github.com/qompassai/network/) 的链接。
- **MediRAG Guard 构建完成！**：一位成员介绍了 **MediRAG Guard**，它使用 **Python**、**Groq**、**LangChain** 和 **ChromaDB** 构建，旨在理解医疗数据隐私规则，并使用独特的分层上下文树（Context Tree），[演示在此](https://github.com/pr0mila/MediRag-Guard)。
   - 它能提供更清晰、更准确的答案；*这就像在森林里有一个向导为你指路！*。


  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1377141406853763224)** (2 messages): 

> `Web app OCR integration, Backend framework choices for AI/ML serving, Database options for OCR and LLM data, Efficient deployment strategies for AI web apps, Libraries/SDKs for AI model integration` 


- **OCR Web 应用技术栈选择**：一名成员正在开发一个集成 **Optical Character Recognition (OCR)** 的 Web 应用程序，可能还会包含 **LayoutLMV** 等其他模型以及 **Large Language Model (LLM)**。
   - 他们正在评估适合 **AI 模型** 资源密集型特性的 Web 开发工具，包括后端框架、数据库技术和部署策略。
- **用于 AI/ML 推理服务的稳健后端框架**：该成员需要用于提供 **AI/ML 模型** 服务（自托管或通过 API）的后端框架，以处理数据处理（例如 **OCR** 的预处理、管理 **LLM prompts/responses**）和 API 创建。
   - 他们对基于 **Python** 和 **JavaScript** 的解决方案都持开放态度。
- **用于 OCR 和 LLM 数据存储的数据库技术**：该成员寻求适合存储和检索与 **OCR 结果**、**用户输入** 以及潜在的 **模型输出** 相关数据的数据库建议。
   - 他们正在寻找在 Web 应用程序中管理和访问这些数据的高效方法。
- **AI 驱动 Web 应用的高效部署策略**：该成员正在研究其 **AI 驱动 Web 应用程序** 的高效部署策略，特别是考虑到 **AI 模型** 潜在的资源消耗。
   - 他们需要能够处理集成的 **OCR**、**LayoutLMV** 和 **LLM** 模型计算需求的解决方案。
- **简化 AI 模型集成的库/SDK**：该成员对能够简化将 **OCR**、其他 **ML 模型** 和 **LLM** 集成到 Web 环境中的库或 SDK 感兴趣。
   - 他们正在寻找能够协助模型服务、API 调用和数据处理的工具，并寻求有关可用选项的指导。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1377257861981143091)** (5 messages): 

> `Hugging Face Learn, smol-course, GitHub-hosted course` 


- **课程困惑已解决**：一名成员最初无法在 [Hugging Face Learn 平台](https://huggingface.co/learn)上找到 **smol-course** 并寻求澄清。
   - 另一名成员澄清说，**smol-course** 托管在 [GitHub](https://github.com/huggingface/smol-course) 上，专为自学进度设计。
- **GitHub 上的自学课程**：专为自学设计的 **smol-course** 仅在 [GitHub](https://github.com/huggingface/smol-course) 上提供。
   - 该课程提供用于个人学习和进阶的模块。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1377005733534629938)** (13 messages🔥): 

> `AI Agent Security, Gradio Agents & MCP Hackathon 2025, AI Agent Cheating, Building AI Agents for Free, Ollama Models for AI Agent Course` 


- **AI Agent 安全担忧浮现**：一名成员对下载并与文件交互的 AI Agent 的**安全性功能**实现表示担忧，特别是涉及代码文件执行时。
   - 该成员希望防止 Agent *盲目下载并执行代码*，从而可能损坏系统。
- **黑客松组队招募进行中**：一名成员正在为 **Gradio Agents & MCP Hackathon 2025** 寻找队友，寻求具有强大 AI Agent 和 MCP 技能的人才。
   - 另一名成员表达了兴趣，但表示自己是 Agent 领域的新手，刚开始学习 Agent MCP 课程，并询问了**报名截止日期**。
- **AI Agent 作弊策略曝光**：一名成员发现，有些人正在通过**复制 Agent** 来直接获取答案进行“元作弊（meta-cheating）”，而没有付出任何努力。
   - 这种作弊方法涉及使用包含答案的 **vector store**，这引发了对提交作品诚信度的担忧。
- **零成本创建 AI Agent 的策略**：一名成员询问如何不花钱构建 AI Agent。
   - 另一名成员建议创建一个“简易版（dumb one）”，或者利用**免费 API 额度**进行少量使用。
- **课程推荐的 Ollama 模型**：一名拥有笔记本电脑（**Ryzen 5 5600H, RTX 3060 6 GB, 16 GB RAM, 100 GB 空间**）的成员询问在 AI Agent 课程中使用哪个 Ollama 模型。
   - 另一名成员建议使用任何 **13B 参数以下**的模型，而第三名成员建议尝试 **Gemma 3**。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1377006258443518105)** (121 messages🔥🔥): 

> `取消订阅, Manus 安全控制, CV 评审, Claude 4.0 集成, Manus 加载问题` 


- **订阅取消困惑**：用户不确定如何取消 **Manus 订阅**，一些人质疑“删除账户”是否是正确的方法。
   - 一位用户报告说删除账户并没有停止订阅，这表明这不是正确的取消方式；另一位用户建议移除卡片并拒绝权限，这样他们就无法扣费。第三位用户建议前往 account -> manage subscription -> 在右下角找到“cancel subscription”链接。
- **关于 Manus 访问用户电脑的担忧**：一位用户询问 **Manus** 是否真的可以*控制他们的电脑*，例如用来注册 Yahoo 账户。
   - 另一位用户澄清说 Manus 并不控制用户的电脑，但反向操作是可能的，即允许用户登录自己的账户并在 Manus 的电脑上完成验证码（captchas）以实现自动化。
- **对 Claude 4.0 集成的期待升温**：用户对 Manus 集成 **Claude 4.0** 表示期待，有人说：*你不知道我有多期待这个……*
   - 目前还没有关于集成 Claude 4 的官方信息，但上周 Manus 发布了一张 Claude 活动的照片，展示了一些使用其系统的优秀合作伙伴，Manus 位列第一，因此成员们相信很快就会集成。
- **Manus 故障导致网站加载问题**：一位用户报告 **Manus 无法加载**，尽管多次刷新，页面仍显示全白。
   - 其他用户建议这可能是由于最近的更新或网络问题，建议将其视为内部的 **Manus bug**。
- **Manus 探索为学生提供无限额度**：用户讨论了 **Manus** 实施“无限”信用额度系统的可能性，特别是针对教育账户，同时保留对其他方案的限制。
   - **Manus** 已经开始为部分学生账户提供无限额度，并为部分教育账户发布了 High Effort 模式，教育账户拥有不同的环境，他们可以从个人账户切换到学校环境，无限制地使用额度。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1377001632444711093)** (89 messages🔥🔥): 

> `感到自己像神一样的神经相关物, Albert 的 AI 生成经文, 纯 RL 算法生成代码, 自定义模型基础设施/钩子 (hooks), DeepSeek 模型 DeepSeek-R1-0528` 


- **上帝情结与去中心化大脑的相关性**：一位成员想知道“感到自己是神”的神经相关物（neural correlate），认为这可能发生在脑内的自我模型扩展到整个世界模型时，与更去中心化的表征相关。
   - 另一位成员建议将每种信仰都视为出于某种目的、由计算过程生成的有用模型。
- **纯 RL 算法从零开始生成代码**：一位成员报告称使用 **纯 RL 算法** 从零生成代码，生成了 **10,000 个程序** 并在 **2 分钟** 内运行。
   - 另一位成员被指责散布 AI 生成的经文，另一位用户直接宣称：*你根本没在说话，你只是在复制粘贴生成式 AI 的输出。你就是生成式智能。开始表现得像个人类吧。*
- **视觉语言模型 (VLMs) 尚不能驾驶**：一位成员分享了一则 [推文](https://x.com/a1zhang/status/1927718115095293975)，暗示 **VLMs** 能够驾驶车辆还需要一段时间。
   - 另一位成员认为 **LLMs** 缺乏良好的视觉能力，因为视觉是在预训练后才补上去的，且缺乏合适的数据集。
- **DeepSeek 最新模型发布**：一位成员分享了在 HuggingFace 上发布的新 **DeepSeek** 模型 [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)。
   - 另一位成员表示，他们对“改变数学原理”更感兴趣，而不是对现有方法进行网格搜索（grid searching）。
- **挂钩（Hooking）自定义模型基础设施**：一位成员正在探索通过使用 hooks 修改前向传播（forward pass），让模型将 Embeddings 传递给自己。
   - 代码已在 [GitHub](https://github.com/dant2021/a-research/tree/main/neuralese_v0) 上发布。


  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1377049923198914630)** (8 条消息🔥): 

> `Reinforcement Learning with Randomness, NN connectome fragility, Multiple narrow optimizers` 


- **Reinforcement Learning with Randomness Exposed**：一位成员分享了一篇[博客文章](https://www.interconnects.ai/p/reinforcement-learning-with-random)，解释了 Reinforcement Learning。
   - 该博客文章讨论了在 Reinforcement Learning 算法中引入 **randomness**（随机性）的好处和方法。
- **通过多个 optimizer 解决 NN 的脆弱性**：一位成员建议，如果你针对其进行架构设计，拥有多个 narrow optimizer（窄优化器）会使系统不那么脆弱。
   - 他们补充说，即使是加入一个 **formal random element**（正式的随机元素）也是有用的，因为现实世界的部署充满了随机性，而无法处理随机性的系统通常无法部署。
- **推文引发想法讨论**：成员们分享了一条推文（未显示），该推文引发了关于某个想法的讨论。
   - 未详细说明该想法的细节或推文的内容。


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1377242453404553217)** (2 条消息): 

> `Probabilistic Circuits, PICs Introduction` 


- **Probabilistic Circuits 资源**：一位成员询问从哪里开始学习 **Probabilistic Circuits**，并链接了两篇 Medium 文章：[Probabilistic Circuits Representation Grammar](https://medium.com/@raj_shinigami/probabilistic-circuits-representation-grammar-969ecaf5e340) 和 [Scaling Continuous Latent Variable Models as Probabilistic Integral Circuits](https://medium.com/@raj_shinigami/scaling-continuous-latent-variable-models-as-probabilistic-integral-circuits-77e853012b7b)。
   - 两篇文章均由 Medium 上的 *raj_shinigami* 撰写。
- **PICs 论文发布**：一位成员分享了介绍 **Probabilistic Integral Circuits (PICs)** 的论文：[Scaling Continuous Latent Variable Models as Probabilistic Integral Circuits](https://arxiv.org/abs/2310.16986)。
   - PICs 论文发布于 **2023 年 10 月**。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1377036091399606342)** (18 条消息🔥): 

> `Huawei AI CloudMatrix Cluster, Linux Kernel SMB Zero-Day Vulnerability, Reinforcement Learning from Tree Feedback, Deepseek R1 Update, Benchmarking Deepseek R1` 


- **华为 CloudMatrix 通过消耗千瓦级电力击败竞争对手**：华为的 **AI CloudMatrix cluster** 使用光互连技术，性能超越了 **Nvidia 的 GB200**，但根据[这篇 Tom's Hardware 文章](https://www.tomshardware.com/tech-industry/artificial-intelligence/huaweis-new-ai-cloudmatrix-cluster-beats-nvidias-gb200-by-brute-force-uses-4x-the-power)，其功耗是后者的 **4 倍**。
- **Linux Kernel 遭受 SMB 实现中的远程零日漏洞攻击**：使用 **O3** 发现了一个 **Linux kernel SMB 实现** 中的远程零日漏洞 (**CVE-2025-37899**)，详见[此帖](https://sean.heelan.io/2025/05/22/how-i-used-o3-to-find-cve-2025-37899-a-remote-zeroday-vulnerability-in-the-linux-kernels-smb-implementation/)。
- **RL Agent 现在开始电锯伐木了**：研究探索了使用 **Reinforcement Learning from Tree Feedback** 和 **Mixture of Chainsaws**，如 [Algoryx 出版物](https://arxiv.org/abs/2403.11623)及[相关图片](https://www.algoryx.se/mainpage/wp-content/uploads/2025/04/log-loading.jpg)所示。
- **Deepseek R1 获得升级**：一位成员指出 **Deepseek R1** 进行了更新，相关文件可在 [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528/tree/main) 上获取。
- **Deepseek R1 超越 o4**：根据[这条 fxtwitter 帖子](https://fxtwitter.com/AiBattle_/status/1927824419478536405)，新的 **Deepseek R1** 模型可能超越 **o4**。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1377018960347664454)** (69 messages🔥🔥): 

> `Aider Copilot API, Aider context limits, aider read, tree sitter, MR/PR title with Aider` 


- **Aider Copilot API 上下文限制**：Aider 支持 **Copilot API**，但用户需要仔细管理上下文，因为 Token 过多时可能会“报错（显示红色错误）”。
   - 遇到 Token 限制问题时，可以切换到 **OpenRouter models**。
- **使用 Aider 和 Tree Sitter 自定义 Repo Maps**：Repo map 只是一个文件，可以使用 `/read` 挂载到 Aider 中；Aider 使用 **Tree Sitter** 通过 `.scm` 查询生成 Repo maps。
   - 一位用户通过向语言模型询问 Tree Sitter 的核心概念（如 predicates, captures 等）学习了如何构建和调试自定义 Repo maps，并使用 `entr` 自动更新 Repo map 的更改。
- **使用 Aider 撰写 MR/PR 标题**：用户寻求一种通过 `git diff origin/main | aider <option here>` 创建 **MR/PR 标题**的方法。
   - 建议的命令是 `aider --message "Generate a pull request title from the following diffs \n $(git diff origin/main)"`，使用文件可以避免命令过长，但可能仍会提示交互式添加文件。
- **Devstral 支持**：由于目前尚未处理特定的配置要求，Aider 尚未完全支持 **Devstral**。
   - 不过，可以手动配置 Aider 以使其与 **Devstral** 配合工作。
- **DeepSeek R1 新更新**：新的 **DeepSeek R1 更新 (0528)** 已发布，展示了极具前景的基准测试结果，并专注于特定修复而非随机错误。
   - 它可能变慢了（增加 30%），但也被认为更加“专注”，且已在 **OpenRouter** 上线；基准测试运行结果非常理想。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1377004050637127710)** (18 messages🔥): 

> `Aider Architect Mode, Copilot Pro API Speed, Deepseek API TPS, Aider strange prices, Sonnet 4 problems` 


- **Aider 架构模式减少注释数量**：一位用户发现，在架构模式（**Gemini 2.5 Pro**）下使用 **aider** 并让另一个模型（**GPT-4.1** 或 **Deepseek V3**）进行编辑时，*注释会显著减少*。
   - 另一位用户对此表示感谢，并表示将尝试这一策略。
- **Copilot Pro API 基准测试令人失望**：一位用户报告称，用于 **Claude-3.7** 的 **Copilot Pro API** 运行缓慢，初始基准测试耗时约 **700 秒**，而通常只需 **120-180 秒**。
- **Deepseek API 吞吐量性能滞后**：用户讨论了直接使用 **Deepseek API** 以及通过 **OpenRouter** 等供应商使用的 **TPS**（每秒事务数），指出直接 API 访问较慢且流量会路由至中国。
   - 一位用户引用了 [OpenRouter 排行榜](https://openrouter.ai/deepseek/deepseek-chat-v3-0324?sort=throughput)，显示 **Baseten** 是一个在速度和价格方面都很有前景的供应商，同时提到 **Gemini-2.5-flash-preview-05-20** 可能是更快的替代方案，参考依据为 [aider.chat](https://aider.chat/docs/leaderboards/)。
- **Aider 价格异常**：多位用户报告称，通过 **OpenRouter** 工作时，**aider** 中显示的模型价格异常低（例如 **GPT-4.1** 每条消息仅为 *$0.000000052*）。
   - 一位用户链接到了相关的 [GitHub issue](https://github.com/Aider-AI/aider/issues/4080)，表明该问题可能已知。
- **Sonnet 4 在编辑中途分心**：一位用户观察到 **Sonnet 4** 有时会给出很好的回答并开始应用 diffs，但随后会突然要求添加新的、通常无关的文件（如 *.png* 或 *.tflite* 文件），并在未应用更改的情况下结束。
   - 用户报告称，提示 *"apply your changes"* 通常可以解决问题，但会消耗额外的 Token。


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1377047990816079893)** (7 messages): 

> `RelaceAI Pricing, Gemini 2.5 Pro Cost` 


- **RelaceAI 定价被指“贵得离谱”**：一位用户分享了 [RelaceAI 的定价链接](https://docs.relace.ai/docs/pricing)，称其*贵得离谱*且比 Claude 还贵。
   - 他们指出该模型的参数量*可能不到 10 亿*，并表示*不会考虑任何比 Gemini 2.5 Pro 更贵的模型*。
- **Aider 非常快**：一位用户分享了 [HackerNews 链接](https://news.ycombinator.com/item?id=44108206)并评论说 Aider *非常快*，他们想尝试一下。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1377056422671417594)** (47 messages🔥): 

> `Kye Gomez 和 SWARMS 争议，用 0.5b 模型 Grokking 圣经，Lucidrains 服务器讨论，Data Attribution 项目` 


- **SWARMS 项目让 Kye Gomez 声名狼藉**：尽管最初有人推荐，但成员们警告不要引导新人关注 **Kye Gomez** 及其 **SWARMS** 项目，因为他被认为是一个“骗子”和抄袭者。
   - 一些人因为其他贡献者的工作而为 **SWARMS** 项目辩护，而另一些人则认为 Kye 的“恶意”行为（包括抄袭和欺骗行为）不应受到支持。
- **Kye 的辩解：太少也太迟了？**：成员们讨论了 **Kye Gomez** 过去承认抄袭和使用 AI 的行为，指出他只有在压力下才会道歉，并在其他情况下继续其不道德行为。
   - 有人指出，只有在方便时才承担责任是一个危险信号（red flag），而且那些被辩护为可以运行的 **Kye's repos** 已被反复证明是无效的。
- **Grokking 圣经：一项“纪元级”的任务？**：一位成员询问让 **0.5b 模型** **grok**（顿悟）**圣经**的可行性和速度，特别是询问加速 grokking 过程的方法。
   - 另一位成员质疑了 **grok** 的定义，而另一位成员则认为，由于存在近乎重复的句子，可能无法让模型 **grok** 足够大的自然语言语料库。
- **Data Attribution**：UCSD 的研究生 Parjanya 介绍了自己以及他之前在语言模型因果关系和记忆化方面的工作，以及最近关于 **data attribution**（数据归因）的研究 ([parjanya20.github.io](https://parjanya20.github.io/))。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1377052920826298418)** (47 messages🔥): 

> `Latro，Muon 矩阵符号近似函数，Spot 论文，COF 结构，用于拓扑手术的噪声注入` 


- **Latro 第三次被重新发现！**：人们第三次重新发现了 **Latro** 并对它们进行了比较，使用 **prob 而非 logprob 作为 policy gradient 的 advantage**。
   - 这种方法“更有意义，因为它的数值表现可能更好”。
- **为 Muon 矩阵近似计算了 Newton-Shannon 系数！**：根据[这篇论文](https://arxiv.org/abs/2505.16932)，**Muon 矩阵符号近似**函数的强 **Newton-Shannon 系数**在每次运行前会预先计算一次。
   - 测试是在他们自己的代码中完成的，因此很难确定有多少是由于他们方法的特殊性，有多少会带来实际（IRL）收益，但“能够将其自动化是非常棒的”。
- **VLM 在验证科学论文方面表现挣扎**：**Spot paper** 基准设计的粉丝想知道，**VLM** 能够验证科学论文所必须具备（但非充分）的技能或能力是什么。
   - 原论文提到了 **long context（长上下文）、multi-hop reasoning（多步推理）、long-tail knowledge（长尾知识）和视觉能力**方面的问题，该成员寻求进一步的见解。
- **模型以糟糕的方式进入“审稿人模式”！**：模型以一种糟糕的方式进入了“审稿人（reviewer）模式”，即即使已经提供了支持性证据，它们也更倾向于要求更多关于某些内容的证据。
   - 错误主要集中在图像上，感觉它们在理解图像微小细节方面做得不够好，特别是**错误标记的 COF 结构**。
- **噪声注入清理了量子损失景观！**：[这篇论文](https://arxiv.org/pdf/2505.08759)讨论了通过噪声注入（noise injection）来正则化**量子损失景观**（quantum loss landscapes）。
   - 成员们表示，讽刺的是，噪声注入是通过增加潜在空间（latent space）中的探索（亚稳态，metastability）并使其平滑来加速 **grokking** 最干净的方法。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1377010815257018418)** (81 messages🔥🔥): 

> `大学正在失去文艺复兴根基，机械可解释性 (Mechanistic Interpretability)，AI 帮助人类开辟新领域，IQ 的逆弗林效应 (Reverse Flynn effect)，合成器助力构建共振策略优化算法` 


- **大学学位正在变成一种交易**：一位成员认为大学放弃了其*文艺复兴*根基，将自己定位为提供产品的机构，**学位变成了一种交易**，导致教育本身的价值流失。
   - 另一位成员同意**大学的底层资产不是教育，而是其品牌名称/公信力**，强调了结识他人、环境提供的沉浸感和体验的重要性。
- **用户寻求机械可解释性见解**：一位成员正在探索语言模型的**机械可解释性 (mechanistic interpretability)**，并向该领域的其他从业者寻求见解和工具。
   - 另一位成员正在通过**交互组合子 (interaction combinators) 和交互网络 (interaction nets)** 研究*可解释性的理论层面*，重点关注概念如何形成以及交互如何影响信息。
- **AI 激发人类超智能**：一位成员认为 **AI 中的超智能 (superintelligence)** 将通过长对话和共同发现，引领人类实现*超智能*，人类和 AI 将共同开辟新领域。
   - 另一位成员赞同 **AI 系统**可以*推理*并解决数学、科学领域的已知问题，并能帮助人类成为更有学识、更有能力的解题者和系统思考者。
- **IQ 的逆弗林效应 (Reverse Flynn effect) 带来挑战**：有观点指出，世界正面临语言推理方面的 **IQ 逆弗林效应**。
   - 他们指出，*超过一半的美国成年人无法理解典型的博客文章或文章*，同时建议**世界模型 (world models) 和体验式学习**对于重塑教育和重建民众的直觉至关重要。
- **合成器触发共振策略优化算法**：一位成员表示，*制作音乐和使用 FM 合成器/减法合成器确实帮助我构建了一个共振策略优化算法 (resonance policy optimization algorithm)*。
   - 他们解释说，思考**数学如何产生音乐**引导他们探索了**噪声与混沌**以及支配它们的**数学原理**之间的关系。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

_humanatee: https://arxiv.org/abs/2505.14442
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

promptsiren: https://odyssey.world/introducing-interactive-video
https://experience.odyssey.world/
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

_humanatee: https://arxiv.org/abs/2505.14442
  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1377046556892725330)** (16 messages🔥): 

> `NPR 风格的声音，音频概览，信息隐私，Deepdive 播客，AI Studio 语音模式` 


- **用户要求减少 NPR 风格的声音**：一位用户询问如何增加不同且较少 **NPR 风格**声音的变化，以及是否需要下载 **.wav 文件**并在第三方应用中进行修改。
   - 一位成员建议可以下载 **.wav** 并编辑**速度、音调**等参数使其听起来更好，但他们不知道有任何 humanizer 应用。
- **NotebookLM 辅助法律文书工作**：一位用户使用 **NotebookLM** 简化并解释了 **25 份文件**，创建了时间线、简报文件和常见问题解答 (FAQs)，并在合并两家公司时按文件重要性创建了带注释的书目，并展示给他们的律师。
   - 该用户能够识别异常信息，并通过与文档对话获取答案，并向律师确认了这些答案；他们还向律师发送了 [Wanderloots 关于 NBLM 隐私和保密性的视频](https://www.youtube.com/watch?v=JnZIjOB5O_4)。
- **NotebookLM 的西班牙语能力测试**：一位用户提到*它在西班牙语下无法正常工作*，指的是上述针对长文本的提示词，并表示希望它们能持续一个小时以上。
   - 另一位用户询问*具体哪里无法工作*，第一位用户未作回复。
- **Deepdive 播客似乎无法逐行阅读**：一位用户请求一个提示词，让 **Deepdive 播客**像老师一样逐行阅读教科书。
   - 另一位用户建议使用 **AI Studio 语音模式 (voice mode)**。

---

### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1376998461484040354)** (68 messages🔥🔥): 

> `NBLM 中的隐私与机密性，移动端与网页端来源不同步，绕过地区限制，播客功能长度控制，笔记本访问设置` 


- **深入探讨数据：NBLM 隐私思考**：一名成员询问了 NotebookLM 中的隐私和数据共享问题，并引用了关于“免费版”的 [YouTube 视频](https://youtu.be/4JU75_v1So4?si=WecaZ7CyoGnUSZQq)。
   - 提供了 **Workspace (Pro)**、**Edu** 和 **Cloud (Ent)** 版本的隐私和数据共享声明链接：[[1](https://workspace.google.com/terms/premier_terms/)]，[[2](https://workspace.google.com/terms/education_terms/)]，[[3](https://cloud.google.com/terms/)]。
- **来源同步故障：移动端 vs. 网页端**：一位用户报告了一个问题，其 **80** 个来源中只有 **10** 个显示在桌面版上。
   - 团队成员确认移动端即将推出更多功能。
- **VPN 之旅：绕过地区限制**：一位用户询问如何绕过地区不可用的限制，另一位用户建议使用 **VPN**、无痕模式并更改其 Google 账号的地区。
   - 原用户表示使用 VPN *不起作用*。
- **播客提示：更长的内容即将到来？**：一位用户询问如何从播客功能中获得更长的结果，并引用了[之前的公告](https://discord.com/channels/1124402182171672732/1182376564525113484/1374492604221100095)。
   - 成员指出，可以使用 Audio Overview 面板中的 `Customize` 按钮来配置来源并延长播客长度。
- **访问焦虑：缺失“任何拥有链接的人”选项？**：一位用户询问如何通过可共享链接公开访问笔记本，类似于 Google I/O 2025 的笔记本。
   - 回复指出，将笔记本访问权限从 `Restricted`（受限）更改为 `Anyone with the link`（任何拥有链接的人）的选项尚未对所有用户开放，可能属于分阶段推出的一部分。Google 确认了这一点。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1377392782863241237)** (1 messages): 

> `现实世界中的 PyTorch/TensorFlow 问题，生产环境中的 ML 挑战` 


- **询问复杂的现实世界 PyTorch/TensorFlow 问题**：一名成员发起讨论，征求在现实生产环境中使用 **PyTorch** 或 **TensorFlow** 解决的最复杂问题的案例。
- **寻求关于生产级机器学习挑战的见解**：该查询旨在收集关于产品设置中机器学习项目所实现的实际困难和复杂解决方案的见解。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1377150556979859497)** (5 messages): 

> `CUDA kernel 编程资源，Triton 的 compiled_hook 移除，tl.trans 实现问题` 


- **请求 CUDA Kernel 入门资源**：一名成员请求诸如 **书籍**、**博客**或 **YouTube 链接**等资源来开始学习 **CUDA kernel 编程**，并提到他们在 **Hopper** 上使用 **ToT Triton**。
   - 他们认为特定设备与初始学习阶段无关，正在寻找一个优秀的 repo 或 YouTube 链接来开始。
- **Triton 的 compiled_hook 函数消失**：一位用户注意到 **Triton** 最新的 master 分支中缺少 `compiled_hook`。
   - 用户试图了解这一变化的原因及其对工作流的影响。
- **tl.trans 实现中的困扰**：一名成员报告了在实现 **v1** 和 **v2 fwd** 时使用 `tl.trans` 函数的问题。
   - 在实现并使用 `tl.trans` 时，如果 **k** 是 **(m,k)**，他们将其加载为 **(k,m)**，结果是正确的。

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1377006578707988521)** (15 messages🔥): 

> `CUBLAS_WORKSPACE_CONFIG, triton kernel, PyTorch Compiler Series, torch.fx.experimental.symbolic_shapes, aot inductorim` 


- **CUBLAS_WORKSPACE_CONFIG 产生非零值**：一位成员发现，将 `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"` 设置后会导致非零输出，尽管生成的 Triton kernel 包含 `tmp9 = tmp8 - tmp8` 这一行（理论上应始终输出零）。
- **PyTorch 发布编译器系列视频**：时隔 4 个月，**PyTorch** 发布了关于编译器的系列视频，名为 [Programming Model for Export - PyTorch Compiler Series Episode 1](https://www.youtube.com/watch?v=bAoRZfJGzZw)。
- **将 Tensor 约束在特定范围内**：一位成员询问如何向编译器传达 Tensor 的约束条件，可能使用 `torch.fx.experimental.symbolic_shapes import constrain_range`，而不使用 `torch.clamp`。
   - 另一位成员建议，*torch.compile* 会假设尺寸是静态的，如果该假设被打破则会重新编译；如果某些内容是动态的，可以使用 *mark_dynamic*，它接受特定维度的最小值和最大值。
- **AOTI Triton 断言失败**：一位成员在使用 **AOTI** 时遇到了 **Triton** 断言失败，他们怀疑这是因为编译器不知道其中一个 Tensor 被约束为仅包含 0 和 1。
   - 虽然使用 *torch.clamp* 可以解决此问题，但他们在推行该方案时遇到了阻力。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1377005261746733066)** (7 messages): 

> `Low-Latency Megakernel for Llama-1B, Grouped Latent Attention (GLA)` 


- **Llama-1B Megakernel 实现零气泡（No Bubbles）**：Hazy Research 推出了一种专为 **Llama-1B** 模型设计的 [低延迟 megakernel](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)。
   - 该博客文章引发了讨论，一位成员表示：“实际上我考虑这个问题已经有一段时间了！”
- **Tri Dao 实验室发布 Grouped Latent Attention**：Tri Dao 实验室发布了关于 [Grouped Latent Attention (GLA)](https://arxiv.org/abs/2505.21487) 的论文，代码预计将在 [GitHub](https://github.com/Dao-AILab/grouped-latent-attention) 上发布。
   - 社区正等待代码发布，以进一步探索 **GLA** 的影响。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1377040741079060641)** (11 messages🔥): 

> `Ninja Build System Troubleshooting, Producer/Consumer Model in Kernels` 


- **Ninja Build 故障排除**：一位成员在 Ubuntu 24.04（gcc 13.3.0, CUDA 12.4, PyTorch 2.7.0）上遇到了 **'ninja -v'** 错误（退出状态 1），在尝试修改 Ninja 命令和环境变量无果后寻求建议。
   - 另一位成员最初怀疑缺少全局安装的 **Ninja**，但随后澄清全局安装确实是正确的配置，暗示问题出在其他地方。
- **Kernel 生产者/消费者模型难题**：一位新成员请求获取相关资源以理解 **matmul kernels** 中使用的**生产者/消费者模型**，寻求从基础到中级水平的解释。
   - 目前尚未提供解决方案或链接。


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1377060360900841595)** (2 messages): 

> `QAT Hyperparameters, TorchTune Experiments, QAT Dataset sensitivity` 


- **QAT 性能取决于超参数和数据集**：一位成员发现，**量化感知训练 (QAT)** 的有效性在很大程度上取决于所使用的 [超参数](https://en.wikipedia.org/wiki/Hyperparameter_optimization) 和特定的 [数据集](https://en.wikipedia.org/wiki/Dataset)。
- **在完全理解 Recipe 之前先合并 PR**：一位成员建议先合并 [Pull Request (PR)](https://en.wikipedia.org/wiki/Pull_request)，并计划稍后研究两个 Recipe 之间的具体差异。
   - 该成员对获得的帮助表示感谢。


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1377349446471389255)** (1 messages): 

> `Grouped Latent Attention, Liger-Kernel Implementation` 


- **Grouped Latent Attention 问题已记录**：**Liger-Kernel** 的 GitHub 仓库中创建了一个关于 **grouped latent attention** 的新 Issue；可以在 [此处](https://github.com/linkedin/Liger-Kernel/issues/734) 找到。
- **开始考虑实现细节**：一位成员表示他们将审查与新记录的 **grouped latent attention** 问题相关的实现细节。


  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1377044478455447552)** (2 messages): 

> `NVIDIA Virtual Connect, Sparse Attention Trade-offs` 


- **NVIDIA 与 CUDA 专家共同举办 Virtual Connect 活动**：**NVIDIA** 将于太平洋时间明天（**5 月 28 日上午 10 点**）举办 [Virtual Connect with Experts](https://gateway.on24.com/wcc/experience/elitenvidiabrill/1640195/4823520/nvidia-webinar-connect-with-experts) 活动，邀请了 **Programming Massively Parallel Processors** (PMPP) 的作者 **Wen-mei Hwu** 和 **Izzat El Hajj**。
   - 与会者可以向资深教育者学习，咨询关于他们著作及 **CUDA** 历程的问题，并了解将于今年 12 月出版的 **PMPP** 第 5 版的预期内容。
- **前沿的 Sparse Attention 权衡**：发布了一篇名为 [The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs](https://arxiv.org/abs/2504.17768) 的新论文。
   - 发布后附带了一个 Google Meet 链接：[https://meet.google.com/wdk-yipf-zjd?authuser=0&hs=122](https://meet.google.com/wdk-yipf-zjd?authuser=0&hs=122)。


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1377197005549338624)** (1 messages): 

> `KernelLLM, Hardware Specific Tools, Project Popcorn` 


- **Popcorn 项目赞助者加入！**：一位来自 **Project Popcorn** 页面的成员表达了他们的热情和贡献意愿。
   - 该用户询问，鉴于目前有许多硬件特定的“工具”可用，是否正在考虑开发 **agentic KernelLLM**。
- **此频道无其他话题**：该频道内没有其他详细的技术讨论。
   - 未提取到进一步的话题。


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1377182042932117564)** (1 messages): 

> `matmul.cu, Producer/Consumer model` 


- **Matmul.cu：请求解释**：一位成员请求解释 **matmul.cu** 中使用的 **producer/consumer model**（生产者/消费者模型）。
   - 该成员正在寻找基础到中级水平的资源，以理解 Kernel 中发生的情况。
- **matmul.cu**：用户希望理解 Kernel 中的 **producer/consumer model**。
   - 他们正在寻找从基础到中级水平的资源来辅助理解。


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1377009152366477402)** (2 messages): 

> `Learning to Reason without External Rewards, Scalability Concerns` 


- **无奖励推理引发关注**：一位成员分享了名为 [Learning to Reason without External Rewards](https://www.arxiv.org/abs/2505.19590) 的论文链接，并对其可扩展性表示怀疑。
   - 他们发现很难想象这种方法能扩展到现实世界的推理场景。
- **对推理可扩展性的质疑出现**：讨论集中在论文中提出的无奖励推理方法是否能有效扩展到复杂的现实情况。
   - 核心担忧围绕该方法在处理复杂推理任务时的实际适用性和局限性。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1377000613283627164)** (19 messages🔥): 

> `amd-mixture-of-experts leaderboard, amd-mla-decode leaderboard, amd-fp8-mm leaderboard, histogram leaderboard, grayscale leaderboard` 


- **MI300 攻破 Mixture of Experts**：一位用户在 `amd-mixture-of-experts` 排行榜上使用 **MI300** 跑出了 **286 ms** 的个人最好成绩。
   - 另一位用户在 **MI300** 上以 **17.8 ms** 获得第 7 名，还有一位用户跑出了 **113 ms** 的个人最好成绩。
- **AMD 上的 MLA 解码：速度传奇**：一位用户在 `amd-mla-decode` 排行榜上使用 **MI300** 以 **135 ms** 的成绩获得第 7 名。
   - 另一个提交以 **131 ms** 的成绩在 **MI300** 上获得第 6 名。
- **FP8-MM 排行榜刷新极速记录**：一位用户在 `amd-fp8-mm` 排行榜上以惊人的 **116 µs** 夺得 **MI300** 组第一名。
   - 另一位用户在 **MI300** 上达到了 **1133 µs** 的个人最好成绩。
- **Histogram 挑战 H100**：一位用户在 `histogram` 排行榜上使用 **H100** 以 **46.6 µs** 的成绩获得第 6 名。
- **Grayscale 在 A100 上取得进展**：一位用户在 **A100** 上多次刷新个人最好成绩，最终在 `grayscale` 排行榜上达到 **3.08 ms**。


  

---

### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1377132753958731789)** (11 条消息🔥): 

> `Project Contributions, Ablation Studies, Meeting Notes, A2A Integration, Colab Notebook` 


- ****贡献者寻找项目切入点****：一位成员表示有兴趣为项目做贡献，并询问了针对相关方法的 **ablation studies** 计划。
   - 另一位成员指出，目前的研究重点不是 fine-tuning 开源模型，但创建一个用于改进客户端并添加更多模型和 API 的 issue 是一个很好的 *good first issue*。
- ****关于 Ablation 背景的澄清****：一位成员澄清说，他们的 ablation 咨询是关于 **prompt/agent loop** 的，例如去掉长期记忆摘要或更改 scaffolding，而不是 fine-tuning。
   - 该成员承诺会查看 GitHub 上的 issue，并表示有兴趣查看最新的会议记录。
- ****缺席后分享会议记录****：在为缺席会议道歉后，一位成员分享了 Otter.ai 上的 [会议记录](https://otter.ai/u/oY942RuHXTuZR7QY98ZhLhgNm9g) 链接。
   - 他们承认因为 *太忙或充满畏难情绪* 而未能处理自己领取的 issue，但承诺会继续关注 GitHub 上的进展。
- ****FLE 进展更新****：根据会议记录，对话以 Morton 的讨论开始（细节未指定），Neel 谈到了 **A2A integration**（重点是 gym 兼容性），Jack 报告说 **Colab notebook** 已接近完成。
   - Jack 还计划解决 Yasaman 提出的 **Import Error** 问题；并分享了能够运行 FLE 的 [Colab notebook](https://colab.research.google.com/drive/1WeWoZNxP2jF4Wd4FLQkeEHwwCBahbVE1) 草案。


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1377088175402647644)** (5 条消息): 

> `KV Cache RoPE, AMD Competition Future, Amd aiter package install` 


- **通过旋转后的 KV Cache 简化 RoPE**：一位用户建议通过在应用 **RoPE (Rotary Position Embedding)** *之后* 将 **k** 存储在 **KV cache** 中来提高效率，从而减少计算开销。
   - 通过仅对当前 token 的序列长度 (1) 而不是整个序列旋转 **k**，该方法可能是一个潜在的优化方向。
- **AMD 竞赛题目将在活动结束后保留**：组织者计划在活动结束后继续开放竞赛题目的提交，尽管不再提供奖品。
   - 题目将保留在 [此链接](https://github.com/gpu-mode/reference-kernels/blob/0156b809d952e20d3d6ef0c55b28568647b3a89e/problems/amd/mla-decode/reference.py#L109)。
- **AMD CK, AITER 软件包安装说明**：一位用户提到，可以使用 **AMD CK, AITER** 及其自带的包在 Python 内部安装该软件包。
   - 这为其他希望利用这些工具的用户简化了设置过程。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1377008072270155796)** (43 messages🔥): 

> `MCP Server Business Case, MCP Clients, Glama Indexing, FastMCP Servers, MCP resource indexing` 


- **建立 SaaS MCP：营销机会？**：一位成员试图为公司建立一个业务案例，通过构建 **MCP server** 来协助 SaaS 公司的集成工作，并预判了“LLM 已经可以阅读其文档”这一反对意见。
   - 另一位成员建议，这是*借势热潮并将你的 API/SaaS 宣传为 AI-ready 的绝佳机会*，这对于*营销团队来说是一个很容易推销的点*。
- **寻找 MCP 客户端：轻量且可高度定制？**：一位成员表示有兴趣寻找一个**轻量且可高度定制的桌面 MCP client**，以便在其上构建工作流。
   - 有人分享了一个 [GitHub 上的 MCP 客户端列表](https://github.com/punkpeye/awesome-mcp-clients?tab=readme-ov-file#clients)，但该成员指出，如果能有 repo 的 stars/forks 统计数据以便排序会更好。
- **Glama 的服务器检测算法：滞后了？**：一位成员注意到 Glama 检测新服务器的算法可能落后了，因为他们的服务器没有被列出。
   - 针对 [awesome list 的一个 PR](https://github.com/punkpeye/awesome-mcp-servers/pull/911) 征求了反馈。
- **Python FastMCP 服务器：如何实现鉴权！**：一位成员询问了关于实现身份验证（authentication）的 **Python FastMCP servers** 示例。
   - 另一位成员询问关于向 Glama.ai 索引提交 **MCP server 提交问题**时该联系谁。
- **LLMs.txt：将内容作为资源暴露？**：一位成员询问是否有人制作了将 **llms.txt** 桥接并将其内容作为资源暴露的 MCP，并找到了 [MCP-llms-txt](https://github.com/SecretiveShell/MCP-llms-txt)！
   - 他们注意到某些 llms.txt 非常庞大，并好奇是否有人解决了处理它们的问题，另一位成员在 [awesome list 上添加了一个 PR](https://github.com/punkpeye/awesome-mcp-servers/pull/940)。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1377074421197570149)** (9 messages🔥): 

> `MCP Launch, UI Issues, MCP Agent Proxy, Multiple models` 


- **新 MCP 平台发布**：一位成员发布了一个用于构建和部署远程 MCP 的平台，访问地址为 [ship.leanmcp.com](https://ship.leanmcp.com)，旨在让用户能够轻松地进行 vibe-code 并部署 MCP。
   - 早期反馈指出了一些典型的 **UI 问题**，如链接问题和电子邮件溢出，部署功能被认为仍在开发中。
- **MCP Agent Proxy 连接客户端与服务器**：[MCP Agent Proxy](https://github.com/mashh-lab/mcp-agent-proxy) 促进了任何 MCP client 与任何 agent server 的连接，通过可组合的原语创建了一个“**Agent 互联网**”。
   - 它支持 **Mastra** 和 **LangGraph**，能够自动检测 agent server 类型并进行适配，详见[这段 YouTube 视频](https://youtu.be/cGY6w3ZZB-4)。
- **与多个模型聊天**：一位成员一直在使用一个 **MCP Server** 与多个模型聊天，发现它作为一个工具非常有帮助，使用的是 [any-chat-completions-mcp](https://github.com/pyroprompts/any-chat-completions-mcp)。
   - 另一位用户提到他们编写了自己的版本 [outsource-mcp](https://github.com/gwbischof/outsource-mcp)，并强调了一个在另一个 MCP 中没有的**图像生成**功能。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1377006730634068019)** (34 条消息🔥): 

> `Anthropic Claude voice mode beta, Next-Gen AI Interfaces Beyond Chatbots, j1-nano and j1-micro reward models, UI Prompting Tutorial, DeepSeek-R1-0528` 


- **Claude 语音功能登陆移动端**：**Anthropic** 在移动端为 **Claude** 推出了 beta 版 **voice mode**（语音模式），允许通过语音交互完成总结日历或搜索文档等任务，目前支持英文，并将很快向所有计划用户开放，详见 [Anthropic 的推文](https://x.com/AnthropicAI/status/1927463559836877214)。
- **AI 界面：超越聊天机器人**：Hiten Shah 在[这条推文](https://x.com/hnshah/status/1927088564166086670?s=46)中强调了**超越聊天机器人的八类新兴 AI 界面**：自动构建的 UI、任务驱动的工作流、画布界面、基于流的构建器、自定义工具、命令行 AI、无提示交互以及新格式。
   - 示例分享在[这个 Clipmate 链接](https://app.clipmate.ai/public/a9b27f9c-57d3-575f-a7c4-9e29ffdd521b)中。
- **小型奖励模型威力巨大**：Leonard Tang 开源了 **j1-nano**（6 亿参数）和 **j1-micro**（17 亿参数），这些小型奖励模型在单个 A100 GPU 上训练时间不到一天，如[本帖](https://x.com/leonardtang_/status/1927396709870489634)所述。
   - 这些模型使用 **Self Principled Critique Tuning (SPCT)** 来生成特定实例的评估标准，其中 **j1-micro** 可与 **Claude-3-Opus** 和 **GPT-4o-mini** 等大型模型相媲美。
- **提示词 UI 设计：一个精彩的教程**：Meng To 发布了一个 **44 分钟的教程**，讲解有效的 **UI prompt engineering**，演示了如何使用 Aura 进行快速 UI 生成、利用模板以及理解 UI 词汇，详见[此推文](https://x.com/mengto/status/1925057411439829457?s=46)。
- **DeepSeek 模型浮出水面，传闻四起**：一个新的 **DeepSeek-R1-0528** 模型出现在 [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) 上，尽管它不是传闻中的 **R2**，但 Aider 团队表示初步基准测试在质量和价格方面显示出潜力。


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1377117772491915265)** (8 条消息🔥): 

> `AI Engineer Conference Volunteers, AI Engineer Conference Speakers, Discord Collaboration Project` 


- **AIE Conference 招募 AI 助手大军**：[AI Engineer Conference](https://xcancel.com/swyx/status/1927558835918545050) 计划于 **6 月 3 日至 5 日**在旧金山举行，现寻求 **30-40 名志愿者**提供后勤支持，作为交换将提供免费入场券（价值高达 **1800 美元**）。
   - 已确认的主旨演讲者包括 **Greg Brockman** (OpenAI)、**Sarah Guo** (Conviction) 和 **Simon Willison**，计划举行 **20 场微型会议**，涵盖各种 AI 工程主题，并设有两条领导力分论坛。
- **Discord 多元化开发方向**：一名成员邀请其他人加入 Discord 上的一个协作项目。
   - 该项目的频道可以在[这里](https://discord.com/channels/822583790773862470/1377194898914021417/1377194905515982928)找到，示例图片见[此处](https://cdn.discordapp.com/attachments/1075282504648511499/1377379230173630596/image.png?ex=6838bfde&is=68376e5e&hm=d57cddedee320068a4c867ac1efa2584cdbb0c5a6a8f707e272e886b9994777b)。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1377319740892254329)** (1 条消息): 

> `aiDotEngineer World Fair, LlamaIndex booth G11, Jerry Liu talk` 


- **LlamaIndex 将参加 6 月的 AI 展会**：**LlamaIndex** 团队将于 6 月 3 日至 5 日参加在旧金山举行的 @aiDotEngineer World Fair，展位号为 **G11**。
   - 与会者可以与 **CEO @jerryjliu0** 及 AI 工程师交流关于 Agent、AI 和 **LlamaIndex** 的话题。
- **Jerry Liu 将在 AI 展会上发表演讲**：**Jerry Liu** 将于 6 月 5 日在 @aiDotEngineer World Fair 发表演讲。
   - 关于他演讲的更多信息可以在[这里](https://t.co/6T3TwX9qiB)找到。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1377037883910783099)** (22 messages🔥): 

> `ReactJS with LlamaIndex, Human-in-the-Loop (HITL) workflow, RetrieverRouter with RelevancyEvaluator, LlamaCloud credits, SubWorkflows in MainWorkflow` 


- **ReactJS HITL 工作流困扰？**: 一位成员正在寻求关于将 **ReactJS** 与 **LlamaIndex** 集成以实现 **Human-in-the-Loop (HITL)** 工作流的建议，并对 `ctx.wait_for_event()` 的复杂性和 WebSocket 通信提出了疑问。
   - 建议采用一种更简单的方法：通过更新后的 context 在 Workflow 上触发另一个 `run()`。
- **Office Hours 示例为 HITL 拨云见日**: LlamaIndex 团队在上次社区 office hours 期间编写了两种形式的 **HITL** 示例：一种是在请求 HITL 时直接响应（即 WebSocket），另一种是在收到人类响应后通过序列化 context 并恢复工作流来稍后响应。
   - [示例可在 Colab 上找到](https://colab.research.google.com/drive/1zQWEmwA_Yeo7Hic8Ykn1MHQ8Apz25AZf?usp=sharing)。
- **评估 RetrieverRouter 相关性**: 一位成员创建了一个工作流，通过带有两个不同索引和一个 reranker 的 **RetrieverRouter** 查询两个知识库，并希望实现相关性评估重试。
   - 他们担心在新的尝试中总是检索到相同的节点会浪费用户时间，并询问是否应该在原始查询中添加信息以改变检索到的节点。
- **LlamaCloud 积分：订阅救星？**: 一位成员询问是否可以在没有订阅的情况下购买 **LlamaCloud** 积分。
   - 另一位成员表示，入门订阅会立即提供 **50K 积分**，之后是按需付费（pay-as-you-go），最高可达 **500K**。
- **SubWorkflow Context 混淆**: 一位成员报告了在 **MainWorkflow** 内部运行 **SubWorkflows** 时管理上的问题，即在多次运行 **SubWorkflow** 后，context 切换和 tracing 会失效。
   - 他们提供了 [DroidAgent](https://github.com/droidrun/droidrun/blob/main/droidrun/agent/droid/droid_agent.py)、[Planner](https://github.com/droidrun/droidrun/blob/main/droidrun/agent/planner/planner_agent.py) 和 [Codeact](https://github.com/droidrun/droidrun/blob/main/droidrun/agent/codeact/codeact_agent.py) **SubWorkflows** 的链接。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1377176769513390212)** (6 messages): 

> `Map function in Mojo, Kapa AI usage` 


- **`map` 来救场，为二维数组追加 5**: 一位成员询问如何使用 `map` 函数将 `[[1,2],[3,4]]` 转换为 `[[1,2,5],[3,4,5]]` 的代码，另一位用户提供了一个 [使用 Mojo 的示例](https://github.com/modularml/mojo)。
   - 代码定义了一个 `main` 函数，初始化一个二维列表，并使用带有捕获函数 `append_five` 的 `map` 来为每个子列表追加数值 **5**；一位用户指出，*由于迭代器尚未在各处完全集成，目前使用 `map` 并不那么常见*。
- **Kapa AI，通过字母而非全名召唤**: 一位成员询问关于使用 [Kapa AI](https://www.kapa.ai/) 的问题，另一位用户指出，要获得 Kapa AI 的回复，你需要输入前几个字母（例如 `kap`），然后从下拉列表中选择 Kapa AI。
   - 他们补充说，输入全名不起作用，并承认：*我以前也是通过这种惨痛的方式才发现的；第一个月我还以为 Kapa AI 是故意不理我*。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1377005852661252309)** (10 messages🔥): 

> `Migrating from Magic to Pixi, uv vs pixi, Conda support, Bootstrapping the ecosystem, Reaching for established C libraries` 


- **从 Magic 迁移到 Pixi**：一位成员在[此论坛帖子](https://forum.modular.com/t/migrating-from-magic-to-pixi/1530)中提供了更多关于从 **Magic** 迁移到 **Pixi** 的信息。
- **Pixi 被正式选中而非 uv**：虽然非科学计算用户可能更倾向于 **uv**，但官方选择 **Pixi** 背后有充分的理由。
   - 一位成员表示，如果有人能解释为什么没有选择支持 **uv** 就更好了。
- **Pixi 利用 uv 并针对 Mojo**：根据[这篇博文](https://prefix.dev/blog/uv_in_pixi)，**Pixi** 在底层使用 **uv** 处理 **Python** 依赖，这符合 **Mojo** 的目标。
   - 一位成员指出：*Mojo 支持异构计算，其中一部分将是异构语言栈，混合了 Python、Rust、C、C++ 和 Mojo。Pixi / conda 正是为此而生*。
- **Conda 支持促进 Mojo 采用**：**Conda** 支持使采用变得非常容易，并应加速引导生态系统。
   - 一位成员从 conda-forge 添加了 **zlib** 并编写了一些薄绑定（thin bindings），并表示：*同样的工具对于我的用户（都使用 conda）来说安装起来非常容易，因为他们只需要添加 modular 频道。非常喜欢 conda 支持*。
- **成熟 C 库支持**：一位成员表示在 **Mojo** 生态系统进一步成熟之前，可以预见自己会使用像 **OpenSSL** 这样成熟的 **C 库**。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1377043655902367977)** (5 messages): 

> `tinygrad.org hyperlink broken, GPU recommendations, tinygrad/tinyxxx` 


- **tinygrad.org 上的 Optypes 超链接失效**：一位成员报告说 [tinygrad.org](https://tinygrad.org) 网站上的 **Optypes** 超链接导致 *404 - 页面未找到* 错误。
   - 这可能是由于最近将 *uops 移入目录* 的更改导致的。
- **Tinyxxx 被合并**：George Hotz 链接到了 [tinygrad/tinyxxx](https://github.com/tinygrad/tinyxxx) GitHub 仓库。
   - 一位成员随后确认一个[相关的 pull request](https://github.com/tinygrad/tinyxxx/pull/27) 已被合并。
- **社区寻求 GPU 推荐**：一位社区成员询问 *是否有任何推荐的 GPU 可以配合使用*？
   - 遗憾的是，摘录中没有提供任何建议。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1377303419836170270)** (5 messages): 

> `CPU backend threads, max_pool1d` 


- **tinygrad CPU 后端无线程**：一位成员询问如何指定 **CPU** 后端使用的线程数，另一位成员回答说 *没有线程，只是 CPU 中的循环*。
   - 为了查看 kernel，他们建议使用 `DEBUG=4` 或 `NOOPT=1 DEBUG=4` 以获得更清晰的视图。
- **max_pool2d 也适用于 1d**：一位成员询问 **tinygrad** 中没有 `max_pool1d` 是否有原因，另一位成员回答说 [`max_pool2d` 可能也适用于 1d](https://docs.tinygrad.org/tensor/ops/?h=max_pool2d#tinygrad.Tensor.max_pool2d)。


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1377027438751645776)** (2 messages): 

> `API Errors, New API users` 


- **API 错误困扰新手**：一位 **API** 新用户报告由于 *“无效请求：消息长度必须至少为 1 个 token，或者必须指定 tool 结果”* 而收到 **Error 400**。
   - 该用户承认自己完全是 **API** 使用新手。
- **新手的 API 使用困境**：一位用户表达了在使用 **API** 时遇到的挑战。
   - 该用户提到他们完全是使用 **API** 的新手。


  

---

### **Cohere ▷ #[🤝-introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1377364604203700306)** (2 messages): 

> `AI Voice & Conversational Systems, Automation & Workflow Engineering, No-Code/Low-Code Platforms, AI Agents & LLM Workflows` 


- **AI Agent 专家加入社区**：一位在构建 **LLM 驱动系统**、**无代码/低代码产品**以及**语音 AI 解决方案**方面拥有实战经验的 AI、自动化、工作流和 Agent 专家加入了 Cohere Discord 服务器。
   - 他们专注于使用现代 AI 和可视化工具创建**智能 Agent**、**可扩展的自动化**以及**全栈 MVP**。
- **突出 AI 语音与对话系统专业知识**：该专家分享了他们在 **VAPI**、**Bland AI**、**Retell AI**、**Twilio** 和 **Telnyx** 等动态语音 Agent 方面的技能，曾为获客、支持和调度构建了具有实时记忆和上下文功能的智能语音机器人。
   - 他们还将 **LLM** 与**电话系统**和 **CRM** 集成，以实现个性化的语音体验。
- **披露自动化与工作流工程实力**：该专家使用 **n8n**、**Make.com** 和 **Zapier** 在 CRM、电子邮件和 AI 流水线中构建了自动化，擅长使用 Webhook 和云服务进行基于 API 的工作流设计。
   - 他们将 **AI Agent** 与 **LangChain**、**Xano** 和 **Backendless** 等工具连接起来。
- **列举无代码/低代码平台技能**：该专家精通 **Glide**、**FlutterFlow**、**Softr**、**Bubble**、**Xano**、**AppSheet**、**WeWeb** 和 **Airtable**，能够交付具有可视化前端、API 逻辑和可扩展后端的完整 MVP。
   - 他们在无需代码的情况下实现了 **Stripe 支付**、**邮件流**和**数据库逻辑**的自动化。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1377049427117605005)** (3 messages): 

> `Kobold, RP, New Interface, Friendship, Dev Life` 


- **用户等待新界面以超越 Kobold 对 RP 的关注**：一位用户表示 **Kobold** 过于关注 **RP**（角色扮演），正等待有人发布新界面来解决这个问题。
   - 未提供关于原因或具体哪个界面的进一步细节。
- **开发者寻求有意义的联系**：一位开发者分享了一个关于失去友谊的个人故事，并正在寻找一位重视深度、真诚联系的开发者（波兰、欧洲或美国）进行合作。
   - 该开发者正在寻找相信信任、团队合作并共同构建有意义事物的人——甚至是*一个有普通想法并想赚取额外收入的普通朋友*。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1377387244314427453)** (1 messages): 

> `AgentX Submission Deadline, AgentX Prizes, AgentX Entrepreneurship Track, AgentX Research Track, Agentic AI Summit` 


- **AgentX 提交即将截止！**：**AgentX 提交**截止日期即将到来，定于 **太平洋时间 5 月 31 日晚上 11:59**。
   - *不要错过！* 通过提供的链接提交您的项目：[创业赛道 (Entrepreneurship Track)](https://forms.gle/FJTC4jd197bNeJJ96) 和 [研究赛道 (Research Track)](https://forms.gle/5dccciawydCZ8o4A8)。
- **AgentX 奖池增至 150,000 美元以上**：**AgentX 竞赛**拥有超过 **150,000 美元的奖金**，包括现金奖励、额度（Credits）和礼品卡。
   - 赞助商包括行业巨头，如 **Amazon**、**Auth0/Okta**、**Groq**、**Hugging Face**、**Google**、**Lambda**、**Foundry**、**Mistral AI**、**NobelEra Group**、**Schmidt Sciences** 和 **Writer**。
- **创业赛道清单**：对于**创业赛道**，提交内容必须包括 **Pitch Deck**（≤20 页）、**产品演示视频**（最长 3 分钟）和**在线产品链接**。
- **研究赛道清单**：**研究赛道**需要一篇**科学论文**（除附录外最多 7-8 页）、**视频演示**（最长 3 分钟）和 **GitHub 仓库**。
- **Agentic AI 峰会将举办演示日与颁奖典礼**：**演示日（Demo Day）和颁奖典礼**将于 **8 月 2 日在伯克利举行的 Agentic AI 峰会**上举行。
   - 需要帮助的参与者可以将问题发送至指定频道。