---
companies:
- anthropic
- openai
- huggingface
- langchain
- llamaindex
- google
- epoch-ai
date: '2025-10-17T05:44:39.731046Z'
description: '最近的 AI 新闻将 **Karpathy 访谈**视为一项重大事件，同时重点讨论了在无需强化学习的情况下改进推理能力的方法，其中**测试时采样（test-time
  sampling）**达到了 GRPO 级别的性能。


  针对上下文窗口营销的批评揭示了其有效限制在 **64K token** 左右，而 **Claude Haiku 4.5** 展示了极具竞争力的推理速度。**GPT-5**
  在高级数学基准测试中表现挣扎，被称为“**脑腐（Brain Rot）**”的数据质量问题正影响着模型的推理能力和安全性。


  在智能体框架方面，**Anthropic Skills** 实现了模块化编码工作流，**OpenAI Codex IDE** 扩展提升了开发者效率，而 **HuggingChat
  Omni** 引入了基于 **Arch-Router-1.5B** 的元路由技术，可跨 100 多个开源模型进行调度。LangChain 和 LlamaIndex
  推进了图优先（graph-first）的智能体基础设施，同时 **Google Gemini** 与谷歌地图集成，以实现现实世界的落地应用。'
id: MjAyNS0x
models:
- claude-haiku-4.5
- gpt-5
- arch-router-1.5b
people:
- karpathy
- aakaran31
- du_yilun
- giffmana
- omarsar0
- jeremyphoward
- claude_code
- mikeyk
- alexalbert__
- clementdelangue
- jerryjliu0
title: Karpathy 与 Dwarkesh 的这场访谈推迟了 AGI（通用人工智能）的时间表。
topics:
- reasoning
- long-context
- sampling
- benchmarking
- data-quality
- agent-frameworks
- modular-workflows
- ide-extensions
- model-routing
- graph-first-agents
- real-world-grounding
---

**努力就是你所需要的一切 (Hard work is all you need)**

> 2025/10/16-10/17 的 AI 新闻。我们为你检查了 12 个 subreddits、[**544** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **23** 个 Discord 服务器（**197** 个频道，**4036** 条消息）。预计节省阅读时间（以 200wpm 计算）：**321 分钟**。**我们的新网站**现已上线，支持完整的元数据搜索和精美的 vibe coded 风格展示的所有往期内容。访问 https://news.smol.ai/ 查看完整的新闻细分，并在 [@smol_ai](https://x.com/Smol_AI) 上给我们反馈！

备受期待的 [Karpathy 访谈本周发布](https://www.dwarkesh.com/p/andrej-karpathy)，并立即成为[全城热议的话题](https://x.com/karpathy/status/1979644538185752935)。

直接去看吧：

https://youtu.be/lXUZvyajciY

---

# AI Twitter 回顾

**无需 RL 的推理：基于采样的收益、长上下文的现状核查以及评估趋势**

- **测试时采样在某些设置下优于 RL**：多个团队报告称，仅通过改进采样（无需 RL、验证器或特殊 Prompt），基础模型就能达到 GRPO 级别的“推理”性能。参见 [@aakaran31](https://twitter.com/aakaran31/status/1979194052697280712) 和 [@du_yilun](https://twitter.com/du_yilun/status/1979204038043537559)。相关主张包括在避免多样性崩溃的同时，达到与 GRPO single-shot 相当的水平。
- **实践中“1M 上下文” ≈ “64K”**：[@giffmana](https://twitter.com/giffmana/status/1979088247323046317) 分享的一个广为流传的批评指出，由于检索策略、截断和 Prompt 管理的现实情况，动辄数十万/100 万上下文的营销往往掩盖了实际有效窗口接近 ~64K 的事实。与之相关，[Epoch AI](https://twitter.com/EpochAIResearch/status/1979243291830030358) 显示 **Claude Haiku 4.5** 在没有显式推理的情况下，达到了早期“推理”模型（o1-mini）的水平，且在其设置中运行速度快了约 5 倍（[后续](https://twitter.com/EpochAIResearch/status/1979243316693864455)）。
- **FrontierMath 饱和度**：[@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1979229992329560197) 发现，即使进行无限次采样，GPT-5 在其极具挑战性的数学基准测试中也无法突破 50%；他们将跟踪未来的收益是来自于已解决问题的可靠性提升，还是真正的全新突破。
- **数据质量至关重要（“脑腐烂”）**：[@omarsar0](https://twitter.com/omarsar0/status/1979217719082774873) 总结了新结果，即在垃圾/高参与度的网络文本上进行持续预训练会导致持久的“思维跳跃”以及推理/长上下文/安全性能的下降，而 reflection 或 finetuning 只能部分修复——这突显了数据清洗（data curation）作为核心安全/性能杠杆的作用。
- **辩论与修正**：关于 GPT-5 “解决”了 10 个埃尔多斯（Erdős）问题的病毒式说法在领域专家纠正后被撤回（[怀疑态度](https://twitter.com/StefanFSchubert/status/1979265669427306507)，[@jeremyphoward](https://twitter.com/jeremyphoward/status/1979291259828343183)）。这一事件强调了在“AI 搞科学”的叙事中，需要经过严格、专家验证的 evals。

**Agent 框架与工具：技能、IDE、路由和现实世界落地**

- **Anthropic Skills for Claude Code**: 开发者们认为 Skills 是编码 Agent 中模块化、版本化工作流以及“持续学习”（精选技能库）的一种实用抽象。来自 [@claude_code](https://twitter.com/claude_code/status/1979098301694681186)、[@omarsar0](https://twitter.com/omarsar0/status/1979242073372164306)、[@mikeyk](https://twitter.com/mikeyk/status/1979287808834679187) 的技巧、模式和现场演示，以及通过 [@alexalbert__](https://twitter.com/alexalbert__/status/1979244443682377804) 与 Anthropic 多 Agent 负责人的深度探讨。
- **OpenAI Codex IDE 扩展**: 一个快速增长的 VS Code/Cursor 扩展，可直接在编辑器中“氛围编码（vibe-code）”功能、前端和云任务（[发布](https://twitter.com/OpenAIDevs/status/1979228278742507630)，[技巧](https://twitter.com/gdb/status/1979268596267438588)）。此外：面向 Business/Enterprise/Edu 用户的 Beta 版完整 MCP 支持（[链接](https://twitter.com/OpenAIDevs/status/1979263194695897353)）。
- **HuggingChat Omni：推理时的元路由（meta-routing）**: [@ClementDelangue](https://twitter.com/ClementDelangue/status/1979230512343585279) 推出了一个编排层，可在 100 多个开源模型（gpt-oss, deepseek, qwen, kimi, smolLM, gemma, aya, …）之间进行路由，由开源的 **Arch-Router-1.5B** 提供支持（[详情](https://twitter.com/ClementDelangue/status/1979256873669849195)）。
- **图优先（Graph-first）的 Agent 基础设施**: 生产级 Agent 模式继续向显式控制流 + 持久性整合：LangChain 的“低抽象 Agent”论点（[博客](https://twitter.com/LangChainAI/status/1979250639117934939)）；LlamaIndex 的代码优先 LlamaAgents 和工作流调试器（[发布](https://twitter.com/jerryjliu0/status/1979214950477582673)，[UI](https://twitter.com/llama_index/status/1979222412479860822)）。
- **结合 Maps 进行 Grounding**: Google 在 Gemini API 中将 Gemini 与 Google Maps 的 2.5 亿多个地点连接起来，实现了具有地理空间 Grounding 能力的 Agent/应用（[开发者文章](https://twitter.com/googleaidevs/status/1979277829750821178)，[studio](https://twitter.com/GoogleAIStudio/status/1979290587951173803)，[概览](https://twitter.com/OfficialLoganK/status/1979286216953733227)）。

**视觉与文档智能激增**

- **Moondream Cloud + 许可协议**: [@vikhyatk](https://twitter.com/vikhyatk/status/1979222969542152567) 推出了 Moondream Cloud；随后将模型许可更新为类似 HashiCorp 的条款，允许除与付费产品直接竞争外的绝大多数用途（[许可说明](https://twitter.com/vikhyatk/status/1979257741152784500)）。开发者们已经开始将其替换到视觉工具中（[使用报告](https://twitter.com/Teknium1/status/1979229000347349165)，[好评](https://twitter.com/MangoSweet78/status/1979231465419219443)）。
- **OCR/VLM 的最新进展（SOTA）**: **PaddleOCR-VL (900M)** 在 OmniDocBench v1.0/v1.5 中名列前茅，支持 109 种语言并提供稳健的输出（JSON/Markdown），已在 HF 上架并集成 Transformers（[摘要](https://twitter.com/reach_vb/status/1979219167258554752)）。**Chandra OCR** 登陆 Datalab API，支持表格/数学/手写/布局识别及 30 多种语言；开源版本即将推出（[发布](https://twitter.com/VikParuchuri/status/1979240389799219523)）。来自 “WithAnyone” 的身份一致性生成（[论文推文](https://twitter.com/_akhaliq/status/1979177813983846629)）。Google 的 “From Pixels to Words” 探讨了可扩展的原生 V+L 原语（[论文亮点](https://twitter.com/_akhaliq/status/1979207679332512204)）。

**研究亮点：科学、RL 与解码效率**

- **AI → 生物学流水线 (开源)**：Google/DeepMind 的 **C2S-Scale 27B**（基于 Gemma 构建）提出了一种免疫疗法的新路径：通过 silmitasertib + 免疫增强使“冷”肿瘤更易被发现；已在之前未见的人类神经内分泌模型上得到验证。论文 + 模型已发布 ([thread](https://twitter.com/GoogleDeepMind/status/1979168384203002066), [result](https://twitter.com/GoogleDeepMind/status/1979168390381027542), [resources](https://twitter.com/GoogleDeepMind/status/1979168392566235514))。
- **QeRL (NVIDIA)**：结合 LoRA + 自适应量化噪声（Adaptive Quantization Noise）的量化 RL，将量化噪声转化为探索。据报告，与 QLoRA 相比，训练速度提升约 1.8 倍，单张 H100 80GB 可微调高达 32B 参数的模型；GSM8K 达到 90.8%，MATH500 达到 77.4%，性能媲美全参数微调（full FT） ([overview](https://twitter.com/TheTuringPost/status/1979325188581007627), [paper/code](https://twitter.com/TheTuringPost/status/1979325287826673769))。
- **通过早期经验进行 Agent 学习**：训练中期信号——隐式下一状态建模（implicit next-state modeling）和对替代状态的自我反思——提升了跨环境和规模的长程（long-horizon）表现；为后续的 RL 提供了强大的起点 ([thread](https://twitter.com/jaseweston/status/1979179944258265358))。
- **Diffusion LLMs 更快的解码**：“Elastic-Cache”在去噪步骤中重用稳定的 KV caches，当注意力漂移时选择性地重新计算深层网络；据报告在数学/代码/多模态（MM）任务中无损实现高达 45 倍的加速，且无需训练并与架构无关 ([summary](https://twitter.com/omarsar0/status/1979180865520570615))。

**基础设施与性能：推理服务、TFLOPs 以及 Apple ML**

- **vLLM + MoE 高速运行**：HF Transformers 后端现在支持 vLLM 中的 MoE 模型全速运行 ([@hmellor_](https://twitter.com/hmellor_/status/1979172956078064124))；vLLM 项目继续获得采用和赞助 ([repo](https://twitter.com/vllm_project/status/1979236314437554669), [sponsor](https://twitter.com/oss_gr/status/1979328234719449326))。
- **Apple ML 栈趋于成熟**：MLX-lm 增加了内存高效的 SSM 预填充（prefill）、分布式评估以及新模型（LFM2 MoE, Nanochat, Jamba, Qwen3-VL 纯文本版） ([update](https://twitter.com/awnihannun/status/1979303565765284309))。社区演示展示了跨混合 Apple Silicon 节点的分布式评估 ([ring demo](https://twitter.com/ivanfioravanti/status/1979192178195759452))。
- **算力统计校准**：来自 [@TheZachMueller](https://twitter.com/TheZachMueller/status/1979202087557710007) 的实时更新的 BF16 非稀疏 TFLOPs 表格和用于实际训练估算的 HF Space ([space](https://twitter.com/TheZachMueller/status/1979236085671576053))。
- **GLM 4.6 吞吐量**：各服务商正在竞相提升 GLM 4.6 的服务速度；其中一家在 Artificial Analysis 上报告了 114 TPS 和小于 18s 的 TTFT ([benchmark post](https://twitter.com/basetenco/status/1979299403828806053))。
- **路线图说明**：Semianalysis 报告称微软曾考虑过基于 18A 的 Maia，“但现在不再考虑”；重点转向 Griffin 变体和系统架构权衡 ([analysis](https://twitter.com/dylan522p/status/1979236688468881488))。

**开源势头与地缘政治**

- **开源模型使用量激增**：尽管仍落后于顶尖的闭源 SOTA，但编程工作负载日益青睐强大的开源产品——[@bindureddy](https://twitter.com/bindureddy/status/1979050379074486376) 点名提到了 Qwen Coder、Kimi 和 GLM 4.6。
- **HuggingFace 作为元路由器**：除了 OSS 使用之外，在推理时跨多个 OSS 模型进行路由（HuggingChat Omni）的举措表明了一种针对质量、成本和延迟的“组合”方法 ([announcement](https://twitter.com/ClementDelangue/status/1979230512343585279))。
- **NVIDIA 在中国：从 95% → 0%**：[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1979231174787846341) 引用黄仁勋关于出口管制消除 NVIDIA 中国市场份额的言论；结论：加速推动用于训练和推理的国产加速器，对全球 AI 供应链产生长期影响。

**热门推文（按互动量排序）**

- [@dwarkesh_sp 对 @karpathy 的采访](https://twitter.com/dwarkesh_sp/status/1979234976777539987) —— AGI “还有十年之遥”，对 RL 的怀疑，关于 Agent “垃圾内容 (slop)” 的讨论；引发了大规模的行业辩论。
- [@Yuchenj_UW 谈 NVIDIA 退出中国](https://twitter.com/Yuchenj_UW/status/1979231174787846341) —— 对中国训练/推理芯片的影响。
- [@ClementDelangue 介绍 HuggingChat Omni](https://twitter.com/ClementDelangue/status/1979230512343585279) —— 通过 Arch-Router-1.5B 在 100 多个开源模型之间进行路由。
- [@aakaran31 谈基于采样的推理](https://twitter.com/aakaran31/status/1979194052697280712) —— 无需 RL/验证器即可达到 GRPO 级别的 single-shot 表现。
- [@giffmana 谈长上下文窗口](https://twitter.com/giffmana/status/1979088247323046317) —— “1M” 和 “500K” 上下文的表现通常仅相当于 “64K”。
- [@GoogleDeepMind 谈 C2S-Scale 27B](https://twitter.com/GoogleDeepMind/status/1979168384203002066) —— 基于 Gemma 的开源模型推动了经实验室验证的癌症治疗假设。

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Qwen3-0.6B 指令遵循测试

  - **[写三次 potato 这个单词](https://www.reddit.com/r/LocalLLaMA/comments/1o8uxh6/write_three_times_the_word_potato/)** (热度: 1028): **该帖子讨论了对 **Qwen3-0.6B** 模型遵循简单指令能力的测试，具体要求是“写三次 potato 这个单词”。模型的回答错误得令人发笑，这表明其在指令解析或推理设置方面可能存在问题。帖子还将其与 **Gemma-1B** 进行了对比，后者在类似任务中也表现不佳，突显了 AI 模型在自然语言理解方面面临的挑战。讨论中包含了一张模型输出的截图，显示其未能达到预期结果，指出了模型训练或配置中可能需要改进的领域。** 评论者指出，指令的措辞可能是导致模型失败的原因之一，并建议使用更清晰的句式如 “Write the word potato three times” 可能会产生更好的结果。这强调了在 AI 指令解析中精确语言的重要性。


## 非技术类 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. AI 模型与基准测试发布

  - **[Sundar Pichai：“Gemini 3.0 将于今年发布”](https://www.reddit.com/r/singularity/comments/1o973ka/sundar_pichai_gemini_30_will_release_this_year/)** (热度: 534): ****Sundar Pichai** 在 Dreamforce 大会上宣布，**Google Gemini 3.0** 将于今年晚些时候发布，接替目前的 Gemini 2.5。该版本预计将成为一个更先进的 AI Agent，利用 Google 的基础设施以及来自 **Google Research**、**Google Brain** 和 **Google DeepMind** 等团队的研究能力。Gemini 3.0 将支持多模态 (multimodal) 交互，能够通过语音、图像和视频进行通信，并将提供免费版和付费版，其中 Pro 模型定价为每月 `€21.99`。** 首席执行官的这一宣布表明发布已迫在眉睫，显示出对产品就绪度的高度信心。然而，一些人对该公告的可信度持怀疑态度，认为这仅仅是炒作。


  - **[Sora-2-pro 是制作恐怖视频的最佳模型](https://www.reddit.com/r/ChatGPT/comments/1o90g4y/sora2pro_is_the_best_model_for_creepy_videos/)** (热度: 603): **该帖子讨论了 **Sora-2-pro** 模型在生成逼真恐怖视频方面的有效性，特别是模仿 2000 年代真实的 VHS 摄像机画面。该模型擅长创造柔和模糊、暗淡色彩、模拟噪点和稳定的时间戳叠加等效果，从而营造出真实的模拟感。相比之下，**Veo 3.1** 在类似任务中的表现被批评为不尽如人意，相关分享的视频链接展示了其结果。** 评论强调了 Sora-2-pro 输出结果令人印象深刻的真实感，一位用户指出其在创作“SCP 视频”方面的潜力。然而，Veo 3.1 因无法产生令人信服的结果而受到批评，用户表示很难从中提取出高质量的内容。

    - 一位用户强调，Sora-2-Pro 模型能够准确地在视频中嵌入时间戳，这对于 AI 模型来说是一项罕见且具有技术挑战性的成就。这种能力增强了 AI 生成内容的真实感，使其更难与真实片段区分开来。该用户提供了一个示例链接来演示此功能。

### 2. AI 对社会和情感的影响

  - **[社交媒体使用量正在下降](https://www.reddit.com/r/singularity/comments/1o90d1t/social_media_use_is_going_down/)** (活跃度: 886): **该图片是一张柱状图，展示了 2012 年至 2025 年全球互联网用户每天在社交网络上花费的时间。它显示自 2012 年以来使用量持续增加，在 2023 年达到 `151 minutes` 的峰值，随后在 2024 年和 2025 年出现下降。这一趋势表明 2023 年后社交媒体参与度有所下降，这可能是由于算法疲劳和 AI 生成内容难以分辨等因素造成的。讨论强调了对社交媒体转型为由重复内容和广告主导的平台的担忧，以及 AI 在取代人类互动中的作用。** 评论者对数据表示怀疑，指出 2020 年 COVID-19 疫情期间没有显著增加，这可能表明数据准确性存在问题。其他人则批评社交媒体的现状过于商业化和算法驱动，导致用户疲劳。

    - ThisGuyCrohns 强调了算法对用户体验的影响，指出算法往往通过过度优化内容推送来创建回声壁。这可能导致单调的体验，因为用户反复接触类似的内容，降低了信息的多样性和参与度。该用户将其与 YouTube 等提供更多样化内容的平台进行了对比，认为算法设计显著影响了用户的留存率和满意度。
    - lilbird333 质疑关于社交媒体使用趋势数据的有效性，特别是在 2020 年 COVID-19 疫情期间。原本预期社交媒体使用量在封锁期间会大幅飙升，但数据并未反映出实质性的增长。这引发了对数据准确性或解读的担忧，暗示数据收集或分析方法可能存在问题。
    - Pleasant-Contact-556 观察到，数据可能表明社交媒体使用进入了平台期，而非下降。这种解读认为，虽然增长可能放缓，但并不一定意味着逆转，指向的是用户参与水平的稳定而非大幅下降。这一观点强调了理解长期数据趋势对于准确评估用户行为变化的重要性。

  - **[小女孩害怕失去她的 AI 朋友](https://www.reddit.com/r/ChatGPT/comments/1o8spsx/young_girl_is_afraid_to_lose_her_ai_friend/)** (活跃度: 892): **一段视频显示，一名 6 岁的中国女孩在不小心弄坏了她的 AI 朋友后正在与其告别。这个 AI 曾帮助她学习天文学和英语等学科，被孩子视为亲密的朋友。这一事件凸显了儿童对 AI 可能产生的情感依恋，强调了 AI 与年幼用户互动中设置护栏（guardrails）的重要性。视频中孩子与 AI 之间流畅的沟通引起了关注，说明了 AI 在情感处理中的作用。** 一位评论者认为 AI 对儿童有益，能提供教育价值和情感支持，类似于传统玩具但增加了学习功能。另一位评论者预测 AI 将导致广泛的心理健康问题，而第三位评论者则批评为了社交媒体流量而在网上分享孩子情感时刻的行为。

    - AI 朋友可以作为一种教育工具，教孩子新单词、数学和科学，并提供具有良好道德寓意的故事，这是传统玩具无法做到的。然而，家长管理孩子使用 AI 的时间至关重要，以确保他们也能进行社交并参与其他活动，类似于管理看电视或使用智能手机的时间。
    - 有人担心 AI 可能会导致心理健康问题的激增。儿童对 AI 形成的情感依恋可能存在问题，因为让他们与非人类实体建立强烈的情感纽带可能并不健康，可能会影响他们的社会性发展。
    - 在网上分享孩子情感时刻的伦理影响引发了争论。一些人认为这是为了社交媒体参与度而剥削个人时刻，这可能损害孩子的隐私和情感健康。

### 3. 能源消耗与 AI 基础设施

  - **[单个 AI 数据中心的耗电量将相当于整个纽约市的一半](https://www.reddit.com/r/OpenAI/comments/1o8xuul/a_single_ai_datacenter_will_consume_as_much/)** (活跃度: 970): **该图片及随后的讨论强调了拟建的 Hyperion 数据中心的巨大规模和能源需求，据预测，其峰值功耗将达到纽约市耗电量的一半。这凸显了 AI 基础设施巨大的能源需求，尤其是随着 AI 应用的不断扩展。与纽约市能源消耗的对比说明了支持此类大规模数据中心可能面临的环境和物流挑战，强调了需要可持续能源解决方案来满足这些需求。** 评论者讨论了支持此类能源需求的可行性，指出虽然中国正在迅速扩大其太阳能容量，但美国的政治挑战可能会阻碍类似的进展。此外，还有人幽默地提到了建造并可能搬迁如此庞大结构的极高成本。

    - **ClownEmoji-U1F921** 对 AI 数据中心的可扩展性表示担忧，指出虽然像 Hyperion 这样的项目目标是 5GW，但未来仍面临重大挑战。他们强调了两个主要的局限性：建造太瓦级数据中心在物理和经济上的可行性，以及充足训练数据的可用性。评论认为，如果不能在减少算力和数据需求方面取得突破，AI 的增长可能会停滞。
    - **WhaleFactory** 讨论了 AI 数据中心增加的电力需求对能源创新的潜在积极影响。他们认为这种需求可能会推动可再生能源和小型核反应堆的发展。评论还探讨了利用 Bitcoin miners 将基荷能源货币化的想法，这些矿工可以根据数据中心的能源消耗需求开启或关闭，从而优化能源利用并可能减少温室气体排放。
    - **TyrellCo** 指出了中美两国在太阳能装机容量上的差距，认为问题不在于技术可行性，而在于政治意愿。他们暗示政治决策（如现任政府取消太阳能项目）正在阻碍可再生能源的采用进程，而这些能源本可以支持 AI 数据中心日益增长的能源需求。

  - **[某种程度上是真的](https://www.reddit.com/r/ChatGPT/comments/1o8xkku/somehow_true/)** (活跃度: 728): **这张图片是一个迷因（meme），幽默地对比了 Stack Overflow 和 ChatGPT 对编程问题的不同反应。它暗示 Stack Overflow 通常是轻视或批评性的，而 ChatGPT 则更加肯定，无论用户的代码是否正确。这反映了开发者之间的一种普遍情绪，即 Stack Overflow 社区有时具有严厉或不友好的性质，而像 ChatGPT 这样的 AI 则更加支持和随和。评论中也回响了这种情绪，用户对 Stack Overflow 严格的管理和过时的答案表示挫败。** 评论者普遍同意该迷因的描绘，指出 Stack Overflow 可能不那么友好，并且经常引导用户去搜索已有的答案，而这些答案可能已经过时。大家一致认为 ChatGPT 提供了更具支持性的回复，即使它们并不总是正确的。

    - **Chimpville** 从技术角度对 Large Language Models (LLMs) 提出了看法，认为它们充当了 Stack Overflow 的“更友好的过滤器”。这意味着 LLM 可以通过过滤掉用处不大的内容来简化寻找相关信息的过程，与 Stack Overflow 上的传统搜索方法相比，潜在地改善了用户体验。
    - **FreeChickenDinner** 指出了 Stack Overflow 搜索功能的一个常见问题，即搜索结果的前几项通常包含过时的答案，或者是重复建议用户使用搜索功能本身。这凸显了在大型社区驱动的平台中维护最新且相关内容的技术挑战。
    - **deepunderscore** 提到使用 Kagi 作为搜索引擎并屏蔽了 Stack Overflow 域名，这表明在技术偏好上，用户倾向于选择可能提供更相关或用户友好结果的搜索引擎。这暗示了一种趋势，即用户正在寻求替代搜索解决方案，以绕过像 Stack Overflow 这样传统平台的局限性。


---

# AI Discord 摘要

> 由 gpt-5 提供的摘要之摘要的总结


**1. 新的多模态（Multimodal）与端侧（On-Device）模型**

- **Qwen3 Vision 登场 Hugging Face**: **Qwen3-VL-8B-Instruct** 已在 Hugging Face 发布，提供广泛的**视觉语言**支持，并配备了开箱即用的 **GGUF** 变体，可在 [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) 和 [NexaAI/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/NexaAI/Qwen3-VL-8B-Instruct-GGUF) 获取，重点展示了**图像描述 (captioning)**、**VQA** 和**多模态生成**等任务。
  - 社区讨论强调了部署的便利性——*“在 HuggingFace 上可用……同时提供 GGUF 格式”*——并将此次发布视为**本地推理**、**边缘端使用**以及快速进行视觉语言 (VL) 流水线**基准测试**的务实举措。

- **Meta 小型模型实力增强**: Meta 推出了 **MobileLLM-Pro**，这是一款拥有 **1B 参数**的端侧模型。据该公告称，在不到 **2T** 公开 token 的训练下，其在推理/问答方面的表现优于 **Gemma 3 1B** 和 **Llama 3.2 1B**：[_akhaliq on X_](https://xcancel.com/_akhaliq/status/1978916251456925757)。
  - 工程师们用辛辣的评论嘲讽了这种炒作——*“智商甚至不到 1”*——但仍在讨论**端侧**模型在**延迟敏感**和**隐私受限**的工作流中适合发挥作用的场景。

- **Haiku 进驻竞技场**: **Claude-Haiku-4-5** 以第 **22 名**的成绩进入 **LM Arena 文本排行榜**，欢迎前往 [Text Arena Leaderboard](https://lmarena.ai/leaderboard/text) 进行侧向对比评估。
  - 随着用户开始比较用于**成本敏感型**生产环境的**小参数量**指令微调模型，版主鼓励大众进行测试并讨论——*“分享你的想法”*。


**2. Agentic 搜索与检索系统**

- **SWE-grep 以 2,800 TPS 切分上下文**: Cognition 宣布推出 **SWE-grep** 和 **SWE-grep-mini**，这是经过强化学习 (RL) 训练的检索器，速度达到约 **2,800 TPS**（比之前的方法快约 **20 倍**），并作为 Windsurf **Fast Context** 子 Agent 推出，详见：[Cognition on X](https://xcancel.com/cognition/status/1978867021669413252) 以及相关的开源客户端 [ceregrep-client](https://github.com/Swarm-Code/ceregrep-client)。
  - 工程师们猜测它是一个**调整后的 QwQ** 或是一个可能运行在 **Cerebras** 上的 *“经过 RLFT 的开源模型”*，并要求提供可复现的**基准测试**、**延迟概况**和**代码**，以验证其 20 倍加速的说法。

- **DSPy 弃用语义检索，转向 Agentic 检索**: 一位成员在 **DSPy** 中重新实现了 **Claude Code 风格的 Agentic 搜索**，使用 **ripgrep 驱动**的术语搜索、初步筛选和重点读取——并认为这优于纯向量检索——引用了解释文章 [Agentic Search for Dummies](https://benanderson.work/blog/agentic-search-for-dummies/)。
  - 从业者认为，当 **LLM 选择上下文**时，*“Agentic 搜索优于语义搜索”*，同时也指出 **LangGraph** 对于新手来说可能显得**模板代码过于繁重**，且存在一些*“容易误用的坑”*。


**3. GPU 内核与多 GPU 框架**

- **PyTorch 释放线程，提升吞吐量**: 一篇关于 **PyTorch** 的 **Python free-threading (无锁线程)** 深度探讨文章概述了为多线程推理和训练**解锁**新并行模式的策略，详见 [PyTorch and Python-Free-Threading](https://trent.me/articles/pytorch-and-python-free-threading/)。
  - 后续研究探索了通过 `torch.func.grad` / `torch.autograd.grad` 挂载**自定义反向传播**逻辑，而工程师们则要求提供一流的 API，以便在**融合 (fused)** 算子中重用 **Autograd 内核**。

- **Iris 覆盖 AMD/NVIDIA 阵营**: AMD RAD 团队的 **Iris** 多 GPU 框架增加了一个用于测试的 **NVIDIA 后端**，同时保持对 AMD 的优化，此外还增加了一个用于底层内核的实验性 **Gluon** 后端；参见 [ROCm/iris](https://github.com/ROCm/iris) 和 [Gluon 文档](https://rocm.github.io/iris/reference/gluon/overview.html)。
  - 开发者强调了其可移植性和即将推出的集群功能——*“横向扩展和 RDMA 支持即将推出”*——以简化**多节点**实验。

- **ThunderKittens 解决 H100 兼容问题**: 开发者指出了 **ThunderKittens** 中 **H100 attention** 内核的损坏问题，分享了一个使用最后两个 commit 的部分编译变通方案，并注意到了新的 `warp::`/`warpgroup::` 命名空间规则；参见最近的 [ThunderKittens commits](https://github.com/aehmttw/ThunderKittens/commits/main)。
  - 内核作者澄清了执行语义——例如，确保 `tma::load_async` 或信号量操作*“由单个线程运行”*——以避免多重启动风险和崩溃。


**4. 基础设施与融资动态**

- **HeyGen ARR 飙升至 1 亿美元**: **HeyGen** 在 **29 个月**内将 ARR 从 **100 万美元**扩展到 **1 亿美元**，并预告了一份名为《HeyGen 之道》(The HeyGen Way) 的战略备忘录，据 [Joshua Xu on X](https://xcancel.com/joshua_xu_/status/1978837985039888388) 透露。
  - 开发者认为这是 **AI 视频**产品市场匹配度 (PMF) 的证明，并热切期待《HeyGen 之道》中具体的**市场进入 (GTM)** 策略指南。

- **Anthropic 押注 Broadcom TPU？**：据 [zephyr_z9 on X](https://xcancel.com/zephyr_z9/status/1978834774786445562) 消息，外界猜测 **Anthropic** 正是 **Broadcom** 神秘的第五个 **$10B** 级客户，可能通过 Broadcom（而非 NVIDIA）采购 **TPU**，并暗示将进行由 Google 主导的更新。
  - 评论者将“$10B 客户”视为**计算采购（compute procurement）**策略转变和寻找替代**加速器（accelerator）**来源的信号。

- **Claude 接入 M365**：**Claude** 宣布与 **SharePoint**、**OneDrive**、**Outlook** 和 **Teams** 集成，并推出了一个**企业搜索（enterprise‑search）**项目。据 [Anthropic on X](https://xcancel.com/anthropicai/status/1978864351076315203) 称，该功能即日起对 Team 和 Enterprise 用户开放。
  - 企业用户对更紧密的**知识工作者（knowledge worker）**工作流表示欢迎，并认为“即日可用”是立即启动**试点部署（pilot rollouts）**的绿灯。


**5. 开源硬件/软件与 RAI 工具**

- **Coral NPU 核心开源**：Google 在 **Apache 2** 协议下开源了 **Coral NPU** 的 Verilog 代码，公开了 **RV32 核心**，并为 **Mojo** 移植等工具链实验提供了一个极佳的目标；仓库地址：[google-coral/coralnpu](https://github.com/google-coral/coralnpu)。
  - 硬件黑客们强调了 **Apache 2** 许可和**仿真优先（sim‑first）**的工作流，以便为**边缘级（edge‑class）**加速器和编译器构建原型。

- **MAX Python API 正式公开**：Modular 开源了 **MAX Python API** 的剩余部分，邀请社区贡献并进行更深层的 Python 集成，详见此 [论坛帖子](https://forum.modular.com/t/open-sourcing-all-of-the-max-python-api/2379)。
  - 开发者们对用于**互操作（interop）**和**扩展（extensions）**的**第一方** API 表示欢迎，并指出“开源”是生态系统快速增长的关键。

- **Diffusers 支持自定义模块（DIY Blocks）**：Hugging Face 推广了带有**自定义模块（custom blocks）**的 **Modular Diffusers**，用于扩展内置功能之外的 Pipeline。精选集合和文档见 [Custom Blocks Collection](https://huggingface.co/collections/diffusers/modular-diffusers-custom-blocks-68c8e37c62a6b2a30fd58401) 和 [Pipeline Blocks Docs](https://huggingface.co/docs/diffusers/main/en/modular_diffusers/pipeline_block)。
  - 从业者称赞其能够“实现库中尚未提供的功能”，同时保持组件的**可互换性**和**可组合性**。


## gpt-5-mini


**1. Agentic 检索与 SWE-grep**

- **SWE-grep 将检索速度提升至 2,800 TPS**：**Cognition** 发布了 **SWE-grep** 和 **SWE-grep-mini**，这是经过 RL（强化学习）训练的检索模型，声称在编程 Agent 的上下文检索中可达到 **~2,800 TPS**（比之前的方法快约 **20 倍**），详见其公告：[Cognition announcement](https://xcancel.com/cognition/status/1978867021669413252)。
  - 社区成员推测 SWE-grep 是在专用基础设施上运行的微调版 **QwQ**，并指向了一个现有的客户端项目（[ceregrep-client](https://github.com/Swarm-Code/ceregrep-client)），一些人认为其结果可能是一个经过 **RLFT（强化学习微调）的开源模型**，而非全新的架构。

- **Agentic 搜索取代纯语义检索**：受 **Claude Code** 和 DSPy 演示的启发，从业者实现了 **Agentic 搜索**（ripgrep → 候选名单 → 读取流水线），认为应该由 LLM 决定包含哪些上下文，而不是依赖固定的语义向量（参见：[Agentic Search for Dummies](https://benanderson.work/blog/agentic-search-for-dummies/)）。
  - 讨论者报告称，在复杂的编程和问答流程中，Agentic 流水线始终优于语义重排序（semantic re-ranking），因为它允许模型*选择*要检查的文档。多位成员强调 **ripgrepping** + 候选名单 + 读取是值得采用的实用模式。


**2. 多模态与视频生成推进**

- **Sora 2 vs Veo 3.1 —— 视频生成领域的军备竞赛**：社区对比了 **Sora 2**（OpenAI Sora 页面：[Sora](https://openai.com/sora)）和 **Veo 3.1**，分享了具体的 Prompt 模板（例如，手持恐怖片预告：**25s, portrait, extra low quality**），并争论哪种模型能更好地遵循复杂的视频 Prompt。
  - 观点不一：一些用户认为 **Sora 2** 能更好地遵循电影级指令，而另一些人则指出这两个系统仍需打磨（物理规律、Prompt 遵循）；讨论强调了精细的 **Prompt 工程**（时长、长宽比、动作提示）对于获得稳定输出的重要性。

- **Qwen3-VL & Gemma 3 推动视觉 LM 边界**：Hugging Face 托管了用于视觉语言任务的 **Qwen3-VL-8B-Instruct** ([Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct))，该模型也出现在 **GGUF** 构建版本中 ([NexaAI/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/NexaAI/Qwen3-VL-8B-Instruct-GGUF))，为工程师提供了图像描述（image captioning）和 VQA 测试的即时访问。
  - 用户推荐将 **Gemma 3 12B Instruct VL** 用于更重的多模态任务，同时指出像 **Liquid FM2 VL 450M** 这样的小型 VL 是受限设置中最小的“有用”选择；HF 的发布引发了快速的本地评估和 GGUF 量化实验。

- **HeyGen：29 个月内 ARR 从 100 万美元增长到 1 亿美元**：**HeyGen** 宣布了在 **29 个月内 ARR 从 100 万美元增长到 1 亿美元**的增长轨迹，并预告了一份名为 *“The HeyGen Way”* 的公开指南（推文报道：[HeyGen growth tweet](https://xcancel.com/joshua_xu_/status/1978837985039888388?s=46)）。
  - 成员们将 HeyGen 视为 AI 视频生成领域快速商业规模化的案例研究，并指出这种增长提高了整个视频生成初创公司领域对产品化、SLA 以及数据集/基准测试的期望门槛。


**3. 低比特、量化与硬件工具**

- **BitNet 宣称达到 1.58-bit 等效性能**：Microsoft 的 **BitNet** 研究和代码库（GitHub: [BitNet](https://github.com/microsoft/BitNet)）以及 HF 上的相关论文（[BitNet paper](https://huggingface.co/papers/2510.13998)）声称，在蒸馏设置下，约 **1.58‑bit** 量化的性能接近等效。
  - 社区反应不一：一些人赞扬了蒸馏结果，而另一些人则质疑其可复现性，并对论文元数据/日期表示困惑；在 **low-bit-training** 讨论帖中，低比特蒸馏作为一种损失函数在 RL 使用案例中也引起了警惕。

- **Unsloth：GGUF 命名、动态量化与更快的 Docker 节奏**：Unsloth 宣布了频繁的 Docker 镜像更新（目标：**每周两次**，Docker Hub: [unsloth/unsloth](https://hub.docker.com/r/unsloth/unsloth)），并通过 Gist 分享了 GGUF 文件名规范（[GGUF naming gist](https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9)），以及关于 **Unsloth Dynamic Quantization** 的文档（[Unsloth docs](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs)）。
  - 用户推动建立一个与 nightly 版本并行的稳定双周发布频道；许多人报告称，由于持续的 Bug 修复和动态量化技巧，**Unsloth 量化版本**的性能优于通用的量化构建版本。

- **H100 attention kernel 与 Iris 多 GPU 工具**：开源 GPU 基础设施讨论指出 Lightning/ThunderKittens 中损坏的 **H100 attention kernels**；一种变通方法是使用 ThunderKittens 仓库中的最新提交（示例提交：[ThunderKittens commits](https://github.com/aehmttw/ThunderKittens/commits/main/)），同时 **Iris** (AMD RAD) 增加了用于测试的 **NVIDIA 后端**（[Iris GitHub](https://github.com/ROCm/iris)）。
  - 工程师们分享了实际的修复方案（如 `warp::`/`warpgroup::` 等命名空间前缀更改），并合力修复 kernel，而 Iris 的跨厂商后端以及即将推出的 RDMA/扩展支持信号预示着更强大的多 GPU 可移植性路径。


**4. 编排、内存系统与 OpenRouter 工具**

- **Nochain 与“真实记忆”宣称——承诺很高，指标匮乏**：一位开发者演示了一个声称拥有**“真实记忆、进化和学习 AI”**的系统，该系统使用确定性的、模型无关的 **Nochain Orchestrator**，并宣称可以节省 Token（网站：[dev.thelastrag.de](https://dev.thelastrag.de)；博客说明：[The Nochain Orchestrator](https://dev.to/tlrag/the-nochain-orchestrator-or-how-to-replace-frameworks-2p9a)）。
  - 批评者要求提供客观的基准测试和可复现的指标——尽管作者断言与朴素的上下文窗口 RAG 方法相比可**节省 90% 以上的 Token** 并提供免费测试访问，但讨论帖中仍记录了对同类比较（apples-to-apples comparisons）的要求。

- **OpenRouter：tool-calling 不稳定、空响应和音频替代方案**：OpenRouter 用户报告了 **tool-calling 不稳定**（导致某些工作流无法使用）以及 SDK 客户端出现空响应的情况，部分用户通过升级客户端代码解决了这些问题；另外，人们在寻找 **Whisper 替代方案**时被推荐了 **fal.ai**、**KittenTTS** 和 **Voxtral**（[fal.ai](https://fal.ai)、[KittenTTS](https://github.com/qulingyuan/Kitten)、[Voxtral writeup](https://apidog.com/blog/voxtral-open-source-whisper-alternative/)）。
  - 频道中充斥着调试技巧（SDK 升级、提供商直接调用）以及关于模型合作社的笑话，而实际的讨论帖则引导团队在 OpenRouter 上构建媒体流水线时选择轻量级的 STT/TTS 选项。

---

# Discord: 高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Claude 的内容被裁剪**：一位用户分享了 **Claude** 设计一个 N 开头单词的*截图*，引发了关于[模型安全担忧](https://link.to/model-safety-article)和内容审查的讨论。
   - 该事件突显了在平衡创作自由与负责任的 AI 开发方面所面临的挑战。
- **Comet 浏览器搞砸了课程**：一位用户尝试使用 **Comet** 完成 [Nvidia 深度学习学院课程](https://link.to/deep-learning-institute)时，在 **Jupyter lab** 会话期间发生崩溃，但随后已修复。
   - 另一位用户询问了如何追踪功能请求，特别是关于 **Comet 浏览器的垂直标签页**。
- **Perplexity Pro 问题困扰订阅用户**：多位用户报告称，在订阅 **Perplexity Pro** 后，无法在 Discord 服务器上获得 **Pro 角色**。
   - 管理员引导用户查看其[账户详情](https://www.perplexity.ai/account/details)，并建议重新连接其 Discord 账号以解决该问题。
- **Perplexity 令人困惑的大量追踪器**：一位用户对 perplexity.ai 上数量过高（超过 *7500* 个）的追踪器表示担忧，并质疑[为什么](https://link.to/adblocker) Windows 应用运行缓慢。
   - 另一位用户认为这些追踪器是[合法](https://www.perplexity.ai/rest/user/settings)的，它们以 JSON 格式提供了关于用户个人资料、AI 模型、Pro 搜索和图像生成限制的详细信息访问权限。
- **Spaces 引发争论**：一位用户报告无法在现有的 **Spaces** 中创建新对话。
   - 目前尚未提供解决方案或原因说明。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **用户渴望 SORA 2 Pro，分享 Prompt 配方**：用户讨论了 [SORA 2 Pro](https://openai.com/sora) 的访问权限，分享了 Prompt 并寻求指导，强调了需明确指定**时长**、**模型**以及**纵向或横向**格式。
   - 一位用户分享了一个 Prompt：*'shaky handheld footage, extra low quality, bad camera footage of creepy horror trailer'*（手持拍摄的晃动画面，极低画质，恐怖预告片的糟糕摄像画面），并建议使用 **Sora 2**，设置 **25 秒**，**16:9 纵向**格式。
- **Codex 胜过 GPT-5 Pro？**：用户将 **GPT-5 Pro** 与 **Codex** 进行了对比，一位用户表示 *Codex 实际上比 GPT-5 更好*，并指出其在工作和副业项目中拥有无限访问权限的优势。
   - 他们还提到可以同时使用多个 **Codex 窗口**，强调了相比于 **GPT-5** 和 **Gemini 3**，他们更倾向于使用 Codex。
- **据称 Vail 是 Ocean AI 重新包装的 xAI 模型**：用户推测 **Ocean AI** 的 **Vail** 是一个 **xAI 模型**，理由是其命名方式和知识库，并暗示 [Ocean AI](https://ocean.ai) 只是一个幌子。
   - 此前与 *Big Sur AI 的 Menlo* 相关的 **Tahoe** 已被 xAI 确认为 **Grok 4 Fast**，这增强了该理论的可信度。
- **Flash Lite 依然缺席**：用户报告称，尽管已添加近一个月，新的 **Flash Lite** 预览版在排行榜上仍然缺失。
   - 管理员表示，模型有时会因各种原因被移除，但他们会进行检查。
- **Gemini 3 发布：期待感增加**：用户对 [Gemini 3](https://ai.google.dev/models/gemini) 的发布表示兴奋，一位用户声称*每天都在查看 3.0 PRO 的新闻*。
   - 有推测称其目标发布日期为 **12 月**，同时讨论了其相对于 **GPT-6** 和 **Claude 4.5 Thinking** 的潜在性能表现。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini 2.5 Pro 和 Claude Sonnet 正面交锋！**：成员们建议在 AI 辅助故事写作中使用 **Claude Sonnet** 和 **Gemini Pro**，同时指出 **Gemini 2.5 Pro** 拥有 **100 万 token 的上下文窗口**。
   - 一位用户提到，在有限的时间内使用 [AI Studio](https://aistudio.google.com/) 是免费的，并且**每天提供 100 次免费的 2.5 Pro 查询**。
- **关于 Sora 2 视频生成能力的辩论**：用户对比了 **Veo 3.1** 和 **Sora 2** 的视频生成效果，争论其提示词遵循能力，但认为两者都需要进一步完善。
   - 关于 **Sora 2** 是否更优的观点不一，有人表示 **Veo 3.1 的物理引擎**和提示词理解能力弱于 **Sora 2** 的早期展示。
- **AI 生成文本的指纹识别**：讨论了检测 AI 生成内容的方法，一位用户表示通过比较 **n-grams** 和**词分布**可以轻松实现。
   - 该用户提到了 *EQBench* 以及通过余弦相似度衡量所有模型指纹的方法，以及随后在该方法上对 DeepSeek 进行的训练。
- **AI 语音助手寻求志愿者**：一位 **PM** 询问是否有其他人具备构建 **AI 语音助手**的经验，寻求一名志愿者来处理项目的 **AI 部分**。
   - 该 PM 建议加入团队，通过 *vpnyolw* 让 **Sora** 走向全球，这导致另一位成员推荐使用 **onetar.os** 来保障通用安全。
- **版权担忧困扰 AI 视频创作**：一位用户请求制作一段《咒术回战》对战悟空的视频，但对**版权问题**以及如何规避表示担忧。
   - 另一位用户为一个原创咒术风格术师的 **55 秒动漫电影感预告片**提供了一个详细的 [prompt](https://cdn.discordapp.com/attachments/1046317269069864970/1428838305281085490/pseudoCode_thought_experiment_for_ai_models.odt?ex=68f3f4de&is=68f2a35e&hm=33ee685e260fa6807db6b0140e367f49abdb019f116864ccf22b1707c9318ca3)。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **请求将 Repo 映射到 Cursor 账户**：用户请求能够将 **repo 映射到特定的 Cursor 账户**，从而允许根据正在使用的仓库在工作和个人账户之间自动切换。
   - 该功能将通过自动将仓库与相应账户关联来简化工作流程。
- **尝试重构游戏库存 UI**：一位用户尝试根据计划文件**一次性重构**他的游戏库存 UI，但由于 `Tool read_file not found` 而失败。
   - 这表明 Cursor 内部的 `read_file` 工具可能存在问题或 Bug。
- **Cursor 侧边栏图标消失**：用户注意到并讨论了 Cursor 的 **UI 变化**，特别是 `platform.openai.com` 侧边栏图标的消失。
   - 这一变化可能会影响用户在平台内的体验和导航。
- **Token Watch 监控 Cursor 使用情况**：一位用户分享了一个 [Vercel app](https://token-watch.vercel.app/) 来**监控 Cursor 使用情况**，并提供了如何使用 `curl` 或 `Invoke-RestMethod` 获取必要 JSON 数据的说明。
   - 这允许用户跟踪他们的 token 消耗以及与使用 Cursor 相关的成本。
- **编辑文件问题困扰用户**：多位用户报告了 **`read_file` 工具**的问题，一位用户创建了一个[论坛话题](https://forum.cursor.com/t/tool-read-file-not-found/137856)来讨论该问题，随后发现这与 **Custom Modes** 有关。
   - 这一普遍存在的问题凸显了 `read_file` 工具与 Custom Modes 之间潜在的 Bug 或不兼容性。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **“真正具有记忆能力的 AI” 首次亮相并提出大胆主张**：一位开发者介绍了一套新的 AI 系统，声称它是 *首个真正具有记忆、进化和学习能力的 AI*，不需要手动创建 RAG、框架、API 成本或人工筛选的聊天记录，可在 [dev.thelastrag.de](https://dev.thelastrag.de) 访问。
   - 该 AI 被宣传为原生具备记忆能力，并允许用户定义其角色（如 AI 女友或工作伙伴），但 **批评者对其缺乏技术信息和表面化的描述表示担忧**。
- **确定性框架提供模型无关（Model Agnostic）的优势**：开发者声称他们的框架是完全确定性且模型无关的，不需要函数调用（function calling）或 Langchain 等标准框架，并能独立保存记忆、筛选聊天、学习、进化和改变身份。
   - 他们声称与常规的 Context Window LLM 相比，该框架可节省 90% 以上的 Token，但 **衡量主观质量的客观指标仍存在争议**。
- **Nochain 编排器取代传统框架**：开发者认为他们的 *Nochain 编排器* 通过完全确定性、模型无关以及独立于外部支持、类或框架，取代了传统框架。
   - 这种方法旨在避免“黑盒行为”和对特定模型能力的依赖，使编排变得可预测且可调试，详见 [The Nochain Orchestrator](https://dev.to/tlrag/the-nochain-orchestrator-or-how-to-replace-frameworks-2p9a) 博客文章。
- **为 OpenRouter 寻找 Whisper 替代方案**：一位用户询问 OpenRouter 上是否有类似 Whisper 的 **音频处理模型**，但得到的建议是使用 [fal.ai](https://fal.ai) 来获取多媒体模型。
   - 建议包括用于超微型语音转文本的 [KittenTTS](https://github.com/qulingyuan/Kitten) 和开源的 Sesame 语音模型，同时一位用户分享了 [Voxtral](https://apidog.com/blog/voxtral-open-source-whisper-alternative/) 的链接，这是一个 **基于 Mistral 的 Whisper 替代方案**。
- **用户感叹 GPT 限制级内容质量退化**：用户抱怨自 **2023 年 11 月 11 日** 系统指纹（system fingerprint）更改以来，**GPT 限制级内容** 的质量有所下降，声称 `gpt-4-preview-1106` 是最后一个适合此类内容的优秀模型。
   - 他们补充说，无论注入多么精妙的越狱（jailbreak）指令，在 *更新* 之后，其输出都会表现出迟疑。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Discord 严阵以待应对诈骗者突袭**：一名成员报告了跨多个频道的 **诈骗** 企图，强调了改进 Discord 上垃圾信息检测和预防的必要性。
   - 据指出，一些用户的账号被盗用并在不知情的情况下传播诈骗信息，这凸显了增强安全措施的紧迫性。
- **JavaScript 在 LM Studio 中失效**：一位用户询问是否可以在 LM Studio 中运行 JavaScript 动画，但另一位成员澄清说它作为一个 **JavaScript 沙箱** 运行，而不是完整的浏览器环境。
   - 这种区别限制了其渲染复杂动画的能力，表明用户对其预期功能存在误解。
- **OpenHands MCP 设置困难**：一位成员对通过 MCP 将 **Grok** 与 **OpenHands** 配合使用的设置过程表示沮丧，称设置说明含糊不清且难以理解。
   - 他们感叹文档缺乏清晰度，表示尽管查阅了 MCP 帮助页面，仍无法完成功能性设置。
- **系统提示词遭遇解析问题**：一位用户发现 LM Studio 会 **解析系统提示词**，导致 AI 看到的内容与用户看到的内容之间存在差异。
   - 他们确定 **括号和其他符号** 是潜在的解析错误来源，其影响因模型、聊天模板和其他因素而异。
- **MedGemma 亮相医疗保健领域**：针对有关 **在医学数据上训练的 LLM** 的查询，一位成员推荐了 **MedGemma**，并链接到了 Huggingface 上的 [lmstudio-community/medgemma-27b-text-it-GGUF](https://huggingface.co/lmstudio-community/medgemma-27b-text-it)。
   - 用户指出，目前尚不确定该模型是基于美国还是英国的医学信息进行训练的。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Docker 镜像带来令人愉悦的每两周更新提升**：Unsloth 团队的目标是每周至少更新其 Docker 镜像 **两次** ([Docker Hub](https://hub.docker.com/r/unsloth/unsloth))，社区成员建议在每夜构建版（nightly builds）之外增加 **每两周一次的稳定版本**。
   - 用户讨论了通过“相加并除以 2”来合并多个 LoRA 适配器进行推理的方法，这实际上是平均它们的权重，尽管这种方法对 **VL 模型性能** 的影响以及官方对此的支持尚不明确。
- **视觉模型进入波动期！**：成员们指出 [**Qwen 2 VL 2B 表现糟糕，几乎看不清东西**](https://github.com/QwenLM/Qwen2)，但称赞 **Liquid FM2 VL 450M** 是最小的实用 VL 模型，而另一位成员则推荐将 **Gemma 3 12B Instruct VL** 用于通用任务。
   - 一位用户发现 **Apple 的 FastVLM-1.5B** 很有前景，而另一位用户发现 **Gemma 3** 和 **LLaMA 3.2** 在 SFT 后经常失败，相比之下 **LFM2-VL-1.6B** 是一个更可靠的选择。
- **GGUF 指南提供极佳的粒度！**：在一位用户询问文件名含义后，一名成员分享了一个包含 **GGUF 模型文件命名规范** 的 [Gist 链接](https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9)。
   - 进一步指出，由于持续的 Bug 修复和 **Unsloth Dynamic Quantization**（动态量化）的实现，**Unsloth 量化版** 通常表现得更好 [文档](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs)。
- **RAG 架构部署受需求困扰？**：一位拥有 **Tesla V100-SXM2-32GB x8** 的用户询问是否应切换到 **A40** 来构建支持多达五个并发用户的 **RAG 系统**。
   - 一位成员表示，这个决定“取决于设计者和业务需求。如果只是个人爱好，那就选你最顺手的。”
- **BitNet 展现二进制光辉！**：用户讨论了使用 [Microsoft 的 BitNet 研究](https://github.com/microsoft/BitNet) 以 **1.58bit** 精度实现 1:1 性能的可能性。
   - 一位用户对 [BitNet 论文](https://huggingface.co/papers/2510.13998) 的最后更新日期表示困惑，怀疑可能不正确，不过另一位用户确认了该论文与 **Microsoft BitNet GitHub 仓库** 的关联。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HuggingChat 的 UI 引发辩论**：**HuggingChat** 推出了新 UI，但一些用户觉得它笨重且缓慢，有人形容它具有“反向魅力（opposite rizzmatic）”。
   - 其他人则认为 UI 很酷。有人幽默地回复道：“兄弟，没人会那样说话”。
- **影响函数（Influence Functions）激发研究兴趣**：一位成员对 **影响函数** 表现出兴趣，寻求讨论将其用于某个研究课题，并分享了[一篇解释影响函数的论文](https://arxiv.org/abs/2308.03296)。
   - 他们还在寻求合作，并分享了一篇展示该函数在研究中应用的论文 ([https://arxiv.org/abs/2411.12580v1](https://arxiv.org/abs/2411.12580v1))。
- **Qwen3 视觉模型发布**：新的 **Qwen3 视觉模型** 已在 HuggingFace 上线，地址为 [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)，同时提供 **GGUF** 格式：[NexaAI/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/NexaAI/Qwen3-VL-8B-Instruct-GGUF)。
   - 该模型支持各种 **视觉语言任务（vision-language tasks）**，包括图像描述、视觉问答和多模态内容生成。
- **FRAI CLI 框架发布**：一位成员分享了 **FRAI** 的 CLI 版本，这是一个“开发者优先”的可信 AI 框架，并提供了 [GitHub 仓库](https://github.com/sebuzdugan/frai) 链接。
   - 开发者正在征求反馈，并请求如果其他人觉得有趣或有用，请在仓库上点个 Star。
- **使用自定义模块 DIY Diffusers**：自定义模块（Custom blocks）被介绍为一种实现库中尚未存在但能无缝嵌入其中的功能的方法，自定义模块可在[此处](https://huggingface.co/collections/diffusers/modular-diffusers-custom-blocks-68c8e37c62a6b2a30fd58401)获取。
   - 可以使用自定义模块添加新功能或修改现有功能；文档可以在[此处](https://huggingface.co/docs/diffusers/main/en/modular_diffusers/pipeline_block)找到。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **SWE-grep 增强 Agent 搜索**：Cognition 推出了 **SWE-grep** 和 **SWE-grep-mini**，这是经过 RL 训练的模型，在代码 Agent 上下文检索中达到了 **2,800 TPS**，比现有方法快约 **20 倍**，详见[其博客文章](https://xcancel.com/cognition/status/1978867021669413252)。
   - 社区成员猜测 SWE-grep 可能是基于 **Cerebras** 调整的 **QwQ 模型**，目前已有类似项目 [ceregrep-client](https://github.com/Swarm-Code/ceregrep-client) 可用；而另一位用户则认为它是一个经过 RLFT 的 OSS 模型。
- **Anthropic 为 Google 盯上 Broadcom TPU？**：有推测称 **Broadcom** 的第五个 **100 亿美元** 客户是 **Anthropic**，可能通过 Broadcom 而非 Nvidia 购买 **TPU**，这或许预示着由 Google 领投的新一轮融资，详见[此推文](https://xcancel.com/zephyr_z9/status/1978834774786445562?s=46)。
   - 此举可能标志着 AI 基础设施采购策略的转变。
- **HeyGen ARR 飙升至 1 亿美元**：**HeyGen** 在短短 **29 个月** 内将 **ARR 从 100 万美元扩展到 1 亿美元**，并宣布即将发布名为 *“The HeyGen Way”* 的宣言以分享其内部策略，详见[此推文](https://xcancel.com/joshua_xu_/status/1978837985039888388?s=46)。
   - 该公司的增长轨迹使其成为 AI 视频生成领域的关键参与者。
- **Meta 发布 MobileLLM-Pro 却遭吐槽**：Meta 推出了 **MobileLLM-Pro**，这是一个针对端侧推理优化的 **1B 参数** 模型，在推理和 QA 方面优于 **Gemma 3 1B** 和 **Llama 3.2 1B**，训练数据少于 **2T** 开源 Token，如[此推文](https://xcancel.com/_akhaliq/status/1978916251456925757)所述。
   - 然而，社区成员对该模型表示嘲讽，一位评论者将其贬低为 *“智商甚至不到 1”*。
- **AI 奶奶的毒舌约会建议吸引数百万人**：完全由 **AI 生成的网红** *grannyspills*（一个直言不讳、拜金的奶奶，专门提供不靠谱的约会建议）于 7 月推出，目前在 Instagram 上的粉丝已接近 **200 万**，如 [X](https://xcancel.com/venturetwins/status/1978852719335985309) 所述。
   - 关于 AI 网红的伦理影响争论激烈，一些人称赞其讽刺艺术，另一些人则质疑 AI 生成人格对文化的影响。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **由 Maxwell 反汇编器驱动的 Jetson Nano**：一位成员指出 **Maxwell 反汇编器** 为第一代 **Jetson Nano** 提供了动力，使其成为受限环境下的可行选择，并链接到了[相关推文](https://x.com/tenderizzation/status/1978871856922087576)。
   - 另一位成员选择了 **Hopper** GPU，因为它们支持 **CUDA-Q**，使其非常适合 **AI** 和 **量子** 应用，尽管 **Blackwell** 目前还无法立即获得。
- **中国通过 PTX/SASS 巧思规避美国 GPU 限制**：面对美国对 **H100** 的限制，据报道 **DeepSeek** 正在利用 **PTX/SASS** 指令来优化内存带宽，从而用更少的资源实现强大的模型。
   - 尽管在法律上仅限于获取 **H20** GPU，但中国继续创新并有效利用现有硬件，突显了其在克服技术壁垒方面的机智。
- **PyTorch 即将迎来无线程范式**：一位成员分享了一篇[博客文章](https://trent.me/articles/pytorch-and-python-free-threading/)，详细介绍了在 **PyTorch** 中解锁并行范式的新线程策略。
   - 另一位成员询问如何在不使用 autograd 的情况下访问 backward 函数，旨在将 autograd 的 Kernel 用于自定义 backward 中的融合 Kernel。建议包括使用 `torch.func.grad` 或 `torch.autograd.grad`。
- **AMD Iris 为测试添加 NVIDIA 后端**：**AMD RAD 团队**在 [Iris](https://github.com/ROCm/iris)（他们的开源**多 GPU 编程框架**）中发布了新功能。
   - 新的 Iris 版本现在拥有一个 **NVIDIA 后端**，用于在任何地方进行测试和编写示例，尽管它仍然针对 **AMD GPU** 进行了优化。此外，**scale-out 和 RDMA 支持**即将推出。
- **H100 Attention Kernel 故障困扰社区**：用户报告了 **H100 Attention Kernel** 的问题，一位用户分享了一个让 **H100 Kernel** 编译成功的变通方法，尽管运行时会崩溃，该方法使用了 [此 GitHub 仓库](https://github.com/aehmttw/ThunderKittens/commits/main/) 的最后两次提交。
   - 一位成员澄清说，现在每个操作都通过命名空间前缀（如 `warp::` 或 `warpgroup::`）明确定义了执行者，这些前缀决定了集体启动行为，这导致了旧版本 TK 中的错误。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Claude Code 的 Agentic Search 解构**：一位成员在发现 **Anthropic** 尚未开源其代码后，使用 DSPy 实现了类似于 **Claude Code** 的 Agentic Search，强调了 LLMs 通过 **ripgrepping** 决定使用哪些上下文的重要性。
   - 该成员找到了 **Claude Code** 用于读取和搜索工具的系统提示词（system prompt），并利用它实现了 Agentic Search。
- **Langgraph 样板代码显得过于冗长**：成员们讨论认为 **Langgraph** *感觉很底层*，因为它要求将所有内容定义为带有冗长样板代码的工作流图，即使简单的控制流已经足够，也强迫使用基于图的思维方式。
   - 另一位成员表示赞同，指出这虽然不是一个糟糕的抽象，但有*许多容易触发的陷阱（foot guns）*。
- **Agentic Search 碾压 Semantic Search**：成员们认为 **Agentic Search** 的表现优于 Semantic Search，因为它允许 LLM 决定在其上下文中包含哪些信息，并引用了[这篇博客文章](https://benanderson.work/blog/agentic-search-for-dummies/)。
   - 该方法涉及对术语进行 ripgrepping、筛选文档，然后阅读这些文档，这与 Semantic Search 预定义的检索和重排序（re-ranking）过程形成对比。
- **Groq 在 OpenRouter 上运行异常**：一位用户报告称 **Groq** 在 OpenRouter 中无法工作，即使将其设置为唯一的提供商也是如此，并提供了配置详情。
   - 该问题已通过截图展示，但在总结时暂无可用解决方案。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **PersonaLLM 征稿**：[PersonaLLM Workshop @ NeurIPS Mexico City](https://openreview.net/group?id=NeurIPS.cc/2025/Workshop_Mexico_City/PersonaNLP) 正在征集各领域中由角色驱动（persona-driven）的 **LLMs** 相关工作。
   - 该研讨会旨在探索跨 **HCI**、心理学、认知科学、文化和评估领域的角色驱动型 **LLMs**。
- **Logit Processors 面临封锁**：闭源 **LLM** 提供商不支持自定义 Logit Processors，因为为了快速推理，它们被*硬编码到了代码中*。
   - 一位成员指出，采取这一措施是因为*有人开始发表论文，研究如何利用这些处理器逆向工程有关上述模型的非公开信息*。
- **Eleuther 辩论国防应用**：一位成员询问 AI 是否像 **OpenAI** 或 **Meta** 那样被用于攻击性目的，包括政府合同。
   - 另一位成员澄清道：*如果你是指我们训练的 AI 模型，答案是“我们没有这样做”。我无法告诉你军方或情报机构正在做什么，或者他们是否正在使用我们的模型。*
- **TREAD 保留 Token 以进行更深层的训练**：一位成员分享了一篇 [Midtraining 综述论文](https://arxiv.org/abs/2510.06826)，指出 Token 不会被丢弃，只是由更少的层进行处理，这与丢弃 Token 的 **MAEs** 不同，从而产生了 **MaskDiT**。
   - 该成员表示，不丢弃所有信息是 **TREAD** 的主要贡献，虽然注意到 MaskDiT 也能工作，但*效果明显较差*。
- **Attention 引入归因分析**：一位成员分享了一个 [YouTube 视频](https://youtu.be/hdi1a9MjwDs?si=taIuYbeF6v-yRSxI&t=628)，讨论了将**归因图（attribution graphs）**从 **MLPs** 扩展到 **Attention**。
   - 成员们现在正致力于将**归因图**扩展到 **MLPs** 之外。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Libtorch 转换挑战心理极限**：一名成员在将 **SAM video** 转换为 **libtorch** 时遇到困难，引发了其他成员的关注。
   - 一位成员回应称，*他不想招惹视频领域的“恶魔”*。
- **PersonaLLM 工作坊征稿**：在**墨西哥城 NeurIPS** 举办的 **PersonaLLM Workshop** 正在征集关于人格驱动 LLM 的投稿，涵盖 **HCI**、**心理学**、**认知科学**、**文化**和**评估**等领域，请通过 [openreview.net](https://openreview.net/group?id=NeurIPS.cc/2025/Workshop_Mexico_City/PersonaNLP&referrer=%5BHomepage%5D(%2F)) 提交。
   - 投稿格式包括 **demos (2-4 页)**、**非存档摘要 (2 页)** 以及**已发表工作的总结**。
- **英国人哀叹定价失误**：一位成员指出了英国定价过高的问题，断言 *£3650 相当于约 $4901，所以我因为国家不对就要多付大约 $900？？*，并附上了[相关图片](https://cdn.discordapp.com/attachments/1149866623109439599/1428779645712465991/image.png?ex=68f466fc&is=68f3157c&hm=ff6b1c2edd5ec622713b2fa3fe197dc4236b0859c27b9580d1c96e9090d06722&)。
   - 未提供更多细节。
- **GLM 4.6 挑战 Claude**：随着 **GLM 4.6** 现已可供本地使用，一些成员预测 *开源社区将不再盲目崇拜 Sam/Elon/Dario*，详见[此 YouTube 视频](https://www.youtube.com/watch?v=bOfoCocOjfM)。
   - 它被预期将成为 **Claude** 的竞争对手。
- **Arxiv 论文困扰同行**：一位成员分享了一篇 Arxiv 论文 ([https://arxiv.org/pdf/2510.14901](https://arxiv.org/pdf/2510.14901))，但承认*他们还不确定该如何评价它*。
   - 另一位成员也链接了同一篇论文。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 遭遇加载错误**：成员们报告了一个**加载错误**，即系统在 Agent 模式下思考时间过长且不开始任务。
   - 部署失败是因为 **OpenAI** 需要需要编译的 *pydantic_core*，因此一位成员计划创建一个无需 OpenAI 依赖即可运行的版本。
- **Manus 禁止额度转售**：严禁出售积分/额度，进一步违规可能导致被移除。
   - 此公告旨在警告平台内未经授权的额度交易。
- **参会者推广伦敦 Manus 工作坊**：一位参加了**伦敦 Manus 工作坊**的成员计划将其推广给一个行业团体。
   - 他们寻求联系 Manus 销售渠道的帮助，并从另一位成员处获得了 [Manus 帮助中心](https://help.manus.im/en/) 的链接。
- **退款请求引发 Prompting 建议**：一位成员请求对一个消耗了几乎所有额度但未能完成设定任务的会话进行退款，并分享了[会话链接](https://manus.im/share/pjJFAsvmMM7rhlBIZ2e0Jh?replay=1)。
   - 一位成员建议，失败案例不会自动获得退款，因为失败原因可能很复杂，且通常与 **Prompting** 有关。
- **Java 开发者为咖啡爱好者开发新应用**：一位成员分享了一个工具 [Workable Cafes](https://workablecafes.com)，帮助人们根据 **WiFi 速度**、**舒适度**和**插座情况**发现咖啡馆。
   - 该应用已被超过 **100 人**使用，创作者欢迎反馈。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **用户考虑 Kimi K2 微调成本**：一位用户考虑微调具有 **1B** 参数的 **Kimi K2**，但对 **100k** 样本的 API 成本表示担忧。
   - 他们建议将数据集减少到 **10k** 样本并进行过滤以控制成本，展示了模型定制的实际方法。
- **Kimi 在输出质量上获得认可，优于 Deepseek**：在比较 **Kimi** 和 **Deepseek** 时，一位用户断言 *Kimi* 提供了*更多的参数、更好的结构化输出以及更简洁*的回答，认为它是更好的模型。
   - 对话强调了输出质量和参数量在模型选择中的重要性，凸显了选择合适 AI 工具时的细致决策过程。
- **用户主张 Deepseek 模仿 Moonshot**：一位用户分享说，他们反复建议 **Deepseek** 采用类似于 **Moonshot** 的特性。
   - 该用户没有回应关于是否有回复的后续问题，但该评论揭示了希望 **Deepseek** 效仿 **Moonshot** 优势的愿望，暗示了可能的不满或对性能提升的期待。

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Linux 遭到重锤！**：成员们开玩笑说 [Linux 被定为非法](https://kernel.org/)，并推测孩子们会自己编写操作系统并分享。
   - 讨论转向了讽刺，对孩子们编写的操作系统表示担忧。
- **一次一个问题解码 AGI**：成员们辩论了 [AGI](https://www.agidefinition.ai/) 的定义，认为它只是一个可以通过充足训练数据解决的复杂问答系统。
   - 引用了 [Dan Hendrycks 的 X 帖子](https://x.com/DanHendrycks/status/1978828377269117007) 和 [Dimitris Papailiopoulos 的 X 帖子](https://x.com/DimitrisPapail/status/1978849863174357052?t=kaj5SsZXgdofKoPV_DWsNA&s=19)，进一步丰富了观点。
- **滴答：同义反复追踪器即将到来**：一位成员提议建立一个 *每周同义反复计数器*，以监控那些将简单概念过度复杂化的研究人员。
   - 动机源于对研究人员设法以多种方式复杂化同一个简单事物的挫败感。
- **Qwen3 Vision 模型登陆 HuggingFace**：新的 [Qwen3 Vision Model](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) 已在 HuggingFace 上发布，标志着视觉模型的又一里程碑。
   - 该发布为 AI 视觉领域的开发者带来了新的能力和机遇。
- **开源困境：模型秘密被曝光？**：一位成员质疑公司是否会**开源其旧模型**，还是更倾向于**从头开始训练一个独立的模型**。
   - 担忧在于公司宁愿保护其最佳技巧，也不愿像 OpenAI 那样公开旧模型。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Google 开源 Coral NPU Verilog 源码**：Google 已在 [Apache 2](https://github.com/google-coral/coralnpu) 协议下开源了 **NPU 模块** 的 Verilog 代码。
   - 矩阵核心看起来有点像 AMD 的 NPU，但它们是 **RV32 核心**，可能是测试 **Mojo 可移植性** 的良好平台。
- **Mojo DAW 梦想火花 🔥**：成员们表达了对 Mojo 中 **TUI 框架**（如 Textual）以及完整 **音频/MIDI 2.0** 能力的强烈渴望，以创建一个高性能的 **DAW**。
   - 一位成员建议编写针对 **Jack** 等库的绑定，并引用了他们的 [OpenGL 实验](https://link.to.opengl) 作为 **重度 FFI 项目** 的例子。
- **TUI 框架灵感涌现！**：一位成员分享了一个名为 [ui-terminal-mojo](https://github.com/rd4com/ui-terminal-mojo) 的 TUI 框架项目链接。
   - 另一位成员提到他们暂停了仿照 Golang 中 **Bubbletea** 等 **ELM 应用** 模式开发的 TUI 框架工作，并提供了仓库链接：[banjo](https://github.com/thatstoasty/banjo/blob/main/examples/multi.mojo)。
- **Origins > Lifetimes 🚀**：一位用户询问了 Mojo 中的 **lifetimes**，并将其与 Rust 的 `<'a> <'static>` 语法进行了比较。
   - 成员们澄清说 Mojo 有一个类似但更符合人体工程学的概念，称为 **Origins**，并链接到了 [官方文档](https://docs.modular.com/mojo/manual/values/lifetimes/)。
- **Modular 将 MAX Python API 投入开源阵营**：Modular **开源**了 **MAX Python API** 的剩余部分，并在[这篇论坛帖子](https://forum.modular.com/t/open-sourcing-all-of-the-max-python-api/2379)中列出了新开源的 Python 模块。
   - 完整 **MAX Python API** 的可用性邀请社区进行贡献和扩展，使开发者能够在基于 Python 的项目中深度集成 **MAX** 功能。



---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Flags 导致配置灾难**：**IMAGE**、**NOLOCALS** 和 **GRAPH_ONE_KERNEL** 等 Flags 导致了配置混淆，因为很难分辨哪些是真正的编译失败，哪些是错误的配置。
   - 有建议提出，如果设备/硬件组合不支持，应让这些 Flags 明确报错。
- **Python 缺乏设备默认设置**：目前无法在 Python 中设置默认设备，而这对于在 Python 脚本中交叉检查不同的后端会非常方便。
   - 一个可能的实现示例是 `Device.set_default('CL')` 或 `Device.DEFAULT = 'CL'`。
- **性能退化在测试中依然存在**：尽管设有测试，但仍发生了性能退化（Speed Regressions），然而 [https://stats.tinygrad.win/](https://stats.tinygrad.win/) 仅有过去 25 天的数据，因此很难查看历史数据。
   - 但成员们确认 Benchmark 正在正常运行。
- **寻求编译测试的通用性**：一位用户想为失败的编译编写测试，但目前还没有好的想法，因为所有的失败都特定于模型架构、设备，在某些情况下甚至是由 **fp16** 引起的。
   - 该用户报告称，他们甚至无法依靠 **ORT** 进行验证，因为在 **FP16** 情况下 **ORT** 也会产生错误结果。
- **分布式系统需要 Tiny 爱好者**：一位涉足分布式系统的人士正在寻求与 **GPU memory** 或 **CRIU** 领域的任何人交流。
   - 他们询问是否有人认识分布式 GPU 领域的专家。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider-CE 分叉版支持 MCP**：有用户询问支持 **MCP (Multi-Control Protocol)** 的 **aider 分叉版**，推荐使用 [aider-ce](https://github.com/dwash96/aider-ce/) 作为替代方案。
   - 命令 `aider --model=openrouter/x-ai/grok-code-fast-1 --edit-format diff` 可以使用特定的模型和编辑格式启动 aider。
- **GPT-5-nano 从 Aider 提交信息中消失**：用户注意到 **aider-ce** 在提交信息中不再提及使用 **gpt-5-nano**，尽管为了最新功能和 **GPT-5-code** 支持进行了切换。
   - 目前尚不清楚这一变化是否是有意为之，但被指出与之前的提交信息习惯有所不同。
- **输入文件名触发自动添加**：在消息后输入文件名（例如 *"see also SomeFile.txt"*）会提示系统询问是否添加文件。
   - 这个功能是偶然发现的，现在已成为一个正式记录的功能。
- **Aider 是否应添加 git 忽略的文件？**：一位成员计划提出功能请求，允许 **aider** 自动添加本地 git 忽略的文件。
   - 讨论尚在进行中，这可能会影响 **aider** 的性能。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **LWP Labs 启动免费 MLOps 工作坊**：LWP Labs 正在推出一个为期 **3 天的免费 MLOps 工作坊**，旨在培训参与者将机器学习模型部署到实际生产环境中，涵盖 **Docker、CI/CD、MLflow 和 AWS**。
   - 该工作坊保证提供实际的部署实践，并包含 **五个实战项目**，旨在增强简历竞争力。
- **由行业资深人士领导的 MLOps 培训**：一场 MLOps 工作坊将由一位拥有 **15 年以上**行业经验的讲师带队，旨在让参与者掌握热门技能。
   - 课程强调**实用知识和动手经验**，确保参加者获得 AI 和 Data Engineering 领域雇主青睐的技能。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Parecki 启动公共规划门户**：Aaron Parecki 推出了一个[公共会议日历](https://meet.modelcontextprotocol.io/)，以简化 **WG/IG 小组会议**的跟踪和参与。
   - 在 Discord 中拥有 `maintainer` 角色的组维护者现在可以公开添加和编辑活动。
- **维护者手册指令**：官方文档[此处](https://modelcontextprotocol.io/community/working-interest-groups#meeting-calendar)为小组维护者增加了负责人期望说明。
   - 鼓励维护者将即将召开的会议添加到日历中，并克隆现有活动以进行定期会议，因为自动循环会议已被禁用以防止出现“僵尸会议”。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该频道长时间没有动态，请告知我们，我们将将其移除。

---

**Windsurf Discord** 没有新消息。如果该频道长时间没有动态，请告知我们，我们将将其移除。

---

您收到此邮件是因为您在我们的网站上选择了订阅。

想更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。

---

# Discord: 详细频道摘要与链接





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1428457496603000993)** (1040 条消息🔥🔥🔥): 

> `Claude 审查, Comet 浏览器, Perplexity Pro, AI 模型, 推荐计划` 


- **Claude 遭到屏蔽处理**：一位用户发布了一张 *Claude 设计 N 字词的截图*，该内容已被审查，引发了关于 [模型安全问题](https://link.to/model-safety-article) 的讨论。
- **Comet 导致课程完成中断**：一位用户在使用 **Comet** 尝试完成 *[Nvidia 深度学习学院课程](https://link.to/deep-learning-institute)* 时，**Jupyter lab** 发生崩溃，但随后已修复。
   - 另一位用户询问是否有办法 *检查/关注与 Comet 浏览器相关的特性请求（如垂直标签页）*。
- **缺失 Pro 身份组令用户沮丧**：多位成员询问如何在 Discord 服务器中获得 **Pro 身份组**，部分用户在订阅后仍遇到问题。
   - 管理员引导他们查看 [账户详情](https://www.perplexity.ai/account/details)，建议他们重新连接 Discord 账号。
- **Perplexity.ai 包含大量追踪器**：一位用户指出 Perplexity.ai 的追踪器过多（*7500 个简直疯狂*），并询问 [原因](https://link.to/adblocker)，认为这是 Windows 应用运行缓慢的原因。
   - 另一位用户表示这是 [合法的](https://www.perplexity.ai/rest/user/settings)，在 JSON 中可以 *看到你个人资料的所有当前限制，包括每个 AI 模型、Pro 搜索、图像生成等*。
- **推荐计划规则引发困惑**：用户对 [推荐计划规则](https://www.perplexity.ai/hub/legal/refer-a-friend-program) 表示困惑，特别是关于 *本计划在美国境外或法律禁止或限制的地区无效* 的条款。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1428483928339255316)** (4 条消息): 

> `Perplexity AI 应用, 可分享线程` 


- **适合女生的 Perplexity AI 应用**：一位用户分享了一个 [Perplexity AI 应用](https://www.perplexity.ai/apps/1a78bb4a-d123-4691-8810-38a5469ed917)，提示词为 *for the girly girls*。
   - 该用户随后提供了一个 [搜索查询](https://www.perplexity.ai/search/50168b6e-fe08-4cc1-87a2-4efc8d8ddfe4#0)。
- **可分享线程**：Perplexity AI 提醒用户确保其线程是 **可分享的**。
   - 他们附带了一个指向 [Discord 频道](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825) 的链接。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1428570894715846811)** (4 条消息): 

> `Spaces 新对话问题, API 额度请求` 


- **Spaces 新对话创建故障排除**：一位用户报告了一个问题，即他们无法在任何现有的 **Spaces** 中创建新对话。
   - 给出的消息中未提供解决方案或原因。
- **API 额度求助**：一位用户请求 **API 额度**。
   - 未提供更多细节。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1428457530127945951)** (709 条消息🔥🔥🔥): 

> `Sora 2 Pro Access, GPT-5 Pro vs Codex, Ocean AI and XAI Model Vail, Gemini 3 Release, Flash Lite Preview` 


- **用户请求 SORA 2 Pro 访问权限和 Prompt 创意**：用户讨论了 [SORA 2 Pro](https://openai.com/sora) 的访问权限并分享了 Prompt，其中一位用户提议使用其两个 Pro 账号代为生成视频，并强调需要指定**时长**、**模型**以及**横屏或竖屏**格式。
   - 另一位用户分享了一个关于 *“手持拍摄晃动镜头、超低画质、恐怖预告片的糟糕摄像机画面”* 的 Prompt，建议使用 **Sora 2**，时长 **25 秒**，**16:9 竖屏**。
- **GPT-5 Pro vs Codex 辩论**：用户对比了 **GPT-5 Pro** 和 **Codex**，一位用户表示 *Codex 目前优于 GPT-5*，并指出其在工作和个人项目中的无限访问优势。
   - 他们还提到同时使用多个 **Codex 窗口**，强调了相比于 **GPT-5** 和 **Gemini 3**，他们更倾向于使用 Codex。
- **Vail 被标记为重命名的 XAI 模型**：用户讨论了 **Ocean AI** 推出的 **Vail** 的起源，一些人怀疑它是 **xAI 模型**，理由是其命名方案和知识水平，而 [Ocean AI](https://ocean.ai) 可能只是一个虚假的实验室名称。
   - 有人指出，之前被识别为 *Big Sur AI 的 Menlo* 的另一个模型 **Tahoe**，已被 xAI 确认为 **Grok 4 Fast**，这进一步印证了该理论。
- **排行榜故障已修复，新款 Flash Lite 仍缺失**：一位用户询问排行榜上为何缺少新款 **Flash Lite** 预览版，促使管理员进行调查并确认其确实不在榜单上。
   - 另一位用户报告称新款 **Flash Lite** 在近一个月前就已添加，但仍不可见；管理员表示模型有时会因各种原因被移除，但他们会进行检查。
- **Gemini 3 热度攀升**：用户对 [Gemini 3](https://ai.google.dev/models/gemini) 的发布表示期待，一位用户声称 *每天都在查看 3.0 PRO 的新闻*。
   - 针对其相对于 **GPT-6** 和 **Claude 4.5 Thinking** 的潜在性能存在各种猜测，有理论认为它计划在 **12 月**发布。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1428528481007960194)** (1 条消息): 

> `Claude-Haiku-4-5, Text Arena Leaderboard` 


- **Claude-Haiku-4-5 加入 Text Arena！**：文本排行榜已更新，**Claude-Haiku-4-5** 位列**第 22 名**。
   - 访问 [Text Arena Leaderboard](https://lmarena.ai/leaderboard/text) 分享你的看法。
- **排行榜已更新**：文本排行榜最近进行了更新。
   - 访问 [Text Arena Leaderboard](https://lmarena.ai/leaderboard/text) 并分享你喜欢的模型！


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1428465115212022013)** (445 条消息🔥🔥🔥): 

> `AI 输出的一致性, GPT-5 编程应用, Gemini 2.5 Pro 对标 Claude Sonnet, AI 文本检测, Sora 2 视频生成` 


- **实现 AI 输出一致性**：一位用户询问如何构建一个在输入一致时能产生一致输出的应用，并指出 [随机的服务器负载分配可能会引入混乱](https://drinkoblog.weebly.com)。
   - 建议包括使用 **grammar constraints**（语法约束）以确保输出有效，以及调整 **temperature/top-p** 设置来控制随机性。
- **关于 GPT-5 编程能力的辩论**：一位用户询问 **GPT-5** 是否可以在不付费的情况下编写应用，随后被引导至 [AI Studio](https://aistudio.google.com/) 进行免费网站构建，该平台每天提供 **100 次免费的 2.5 pro 查询**。
   - 另一位成员指出 Notion 内置了 AI 功能，使用的是 SOTA（最先进）模型，可以实现用户追求的目标游戏化。
- **Gemini 2.5 Pro 与 Claude Sonnet 的创作对决**：成员们推荐使用 **Claude Sonnet** 和 **Gemini Pro** 进行 AI 辅助故事创作，另一位成员建议在这些工具下架前全部尝试一遍，以找到最适合的工具。
   - 有人提到 **Gemini 2.5 Pro** 拥有 **100 万 token 的上下文窗口 (context window)**，非常适合记忆大量细节，且 AI Studio 在限时内是免费的。
- **AI vs 人类：检测生成文本的指纹**：讨论中提到了如何检测 AI 生成的内容，一位用户解释说，通过比较 **n-grams** 和 **词汇分布** 很容易实现，并建议所有模型的指纹都可以通过余弦相似度 (cosine similarity) 进行衡量。
   - 他们表示 *EQBench* 正是这样做的：让基于 Gemini 训练的 AI 因其怪癖和习惯而易于被检测，因此他们基于这种方法训练了 DeepSeek。
- **Sora 2 视频生成**：用户对比了 **Veo 3.1** 和 **Sora 2** 的视频生成能力，争论 **Sora 2** 是否更优，尤其是在理解 Prompt 和遵循指令方面。
   - 虽然有些人认为两者相似且仍需开发，但其他人认为 **Veo 3.1 的物理引擎**和 Prompt 理解能力不如 **Sora 2** 的早期表现。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1428593951220174918)** (11 条消息🔥): 

> `AI 语音助手, Sora 全球 VPN, 技术 Discord 安全` 


- **寻找 AI 语音助手志愿者**：一位成员询问是否有人有构建 **AI 语音助手** 的经验，因为他们是一名 **PM**，正在寻找志愿者来深入研究 **AI 部分**。
   - 该 PM 询问另一位成员是否想加入他们的团队，通过 *vpnyolw 让 **Sora** 走向全球*。
- **使用 AI 进行复核和笔记记录**：一位成员提到他们正在利用 **基础 AI 支持** 来复核工作、记录笔记和搭建框架。
   - 他们还提到自己是一名正在进行小组项目的**大学生**，建议将学校规则输入到 Prompt 中以获取指导。
- **为技术 Discord 推荐 VPN**：一位成员表示他们没有 VPN，促使另一位成员推荐使用 **onetar.os** 以保证安全。
   - 另一位成员表示赞同，称 *这个服务器里有一些奇奇怪怪的人*。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1428481918823039119)** (23 条消息🔥): 

> `未来机器人 Prompt, Sora 2 AI, 病毒式传播视频 Prompt, 咒术回战 vs 悟空 Prompt, Sora 的图像识别` 


- **风暴中的机器人 Prompt**：一位用户请求一个 Prompt，描述 *一个未来的机器人在风暴中敲打某人的门并请求进入，但随后被卷入龙卷风*。
- **如何制作病毒式传播视频**：一位用户请求制作 *病毒式视频* 的 Prompt，一位成员建议提供更多细节，而不是使用模糊的 Prompt。
- **请求《咒术回战》vs 悟空视频的 Prompt**：一位成员请求制作 *咒术回战 vs 悟空视频* 的 Prompt，但另一位用户提到 *该图像受版权保护*。
   - 另一位成员提供了一个详细的 [Prompt](https://cdn.discordapp.com/attachments/1046317269069864970/1428838305281085490/pseudoCode_thought_experiment_for_ai_models.odt?ex=68f3f4de&is=68f2a35e&hm=33ee685e260fa6807db6b0140e367f49abdb019f116864ccf22b1707c9318ca3)，用于制作一段 55 秒的动漫电影感预告片，内容是原创咒术风格术师（蓝色/紫色咒力）对战赛亚人风格英雄（金色气场）。
- **Sora 可以识别真人与虚构图像的区别**：一位用户询问 *Sora 是否能识别真人图像与虚构角色图像的区别*，一位成员回答：*是的*。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1428481918823039119)** (23 messages🔥): 

> `文本转图像提示词, 受版权保护的图像生成, Sora AI 能力, 无字数限制的扩展战斗场景` 


- **机器人的讽刺命运转折**：一位用户请求了一个文本转图像提示词：*一个未来的机器人在风暴中敲击某人的家门并请求进入，但随后被龙卷风吸走*。
   - 另一位用户建议他们可以自己创建提示词，通过描述他们在新图片中想要的细节来实现。
- **AI 视频生成的版权担忧**：一位用户询问如何生成 *咒术回战 (Jujutsu Kaisen) vs 悟空 (Goku)* 的视频，但对**版权问题**表示担忧。
   - 另一位用户建议以 *anime* 开头并添加更多细节，但原帖作者担心图像受版权保护。
- **咒术风格术师对决预告片提示词**：一位用户提供了一个详细的提示词，用于创建一个 **55秒动漫电影感预告片**，内容是原创咒术风格术师（蓝色/紫色咒力）对阵赛亚人风格英雄（金色气场）。
   - 该提示词包含了对升级循环、垂直感、色彩对比、拖影帧 (smear frames)、冲击波时机、分辨率 (**1080p**)、帧率 (**24fps**) 以及重低音配乐同步的具体要求。
- **Sora 的图像识别实力受到质疑**：一位用户询问 **Sora** 是否能识别真人图像与虚构角色图像的区别。
   - 另一位用户简单地给出了肯定的回答。
- **发布的伪代码 AI 思维实验**：一位用户分享了一个针对 AI 模型的 [pseudoCode 思维实验](https://cdn.discordapp.com/attachments/1046317269069864970/1428838305281085490/pseudoCode_thought_experiment_for_ai_models.odt?ex=68f3f4de&is=68f2a35e&hm=33ee685e260fa6807db6b0140e367f49abdb019f116864ccf22b1707c9318ca3&)。
   - 该实验以 CREATIVE COMMONS 协议编写，作者邀请用户如果想在其他地方使用/修改/打印，可以私信他们添加自己的 ID 到 1.0 版本中，并专门选择了 **SGM**，因为它在不同模型中作为 Token 出现的频率很高。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1428458169369235539)** (383 messages🔥🔥): 

> `仓库映射至 Cursor 账户, Perplexity Comet 邀请与 ChatGPT 促销, 游戏库存 UI 重构, Cursor 的小故障, 平台 UI 变更` 


- **请求将仓库映射至 Cursor 账户的功能**：一位用户请求能够将 **仓库映射到特定的 Cursor 账户**，以便根据正在使用的仓库在工作和个人账户之间自动切换。
- **尝试游戏库存 UI 重构**：一位用户尝试根据计划文件**一次性 (one-shot) 重构**其游戏库存 UI，但因 `Tool read_file not found` 而失败。
- **Cursor 经历 UI 变更**：用户注意到并讨论了 Cursor 的 **UI 变更**，特别是 platform.openai.com 侧边栏图标消失的问题。
- **使用 Token Watch 分析 Cursor 使用情况**：一位用户分享了一个 [Vercel 应用](https://token-watch.vercel.app/) 来**监控 Cursor 使用情况**，并提供了如何使用 `curl` 或 `Invoke-RestMethod` 获取必要 JSON 数据的说明。
- **多名用户遇到编辑文件问题**：多位用户报告了 **`read_file` 工具** 的问题，其中一位用户创建了一个 [论坛话题](https://forum.cursor.com/t/tool-read-file-not-found/137856) 来讨论该问题，随后发现这与 **Custom Modes** 有关。


  

---

### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1428819354782535801)** (124 条消息🔥🔥): 

> `True Remembering AI, deterministic model agnostic Framework, objective metrics, nochain orchestrator` 


- ****True Remembering AI** 带着大胆的主张亮相**：一位开发者介绍了一个新的 AI 系统，声称它是*首个真正的记忆、进化和学习型 AI*，不需要手动创建 RAG、框架、API 成本或人工筛选的聊天，可在 [dev.thelastrag.de](https://dev.thelastrag.de) 访问。
   - 该 AI 被宣传为具有原生记忆能力，并允许用户定义其角色（如 AI 女友或工作伙伴），一位用户幽默地评论了一张图片：*ahahah lol😄how the heckl*。
- ****Deterministic Framework** 提供模型无关的优势**：开发者声称他们的框架是完全确定性且模型无关的 (model agnostic)，不需要 function calling 或 Langchain 等标准框架，并且能独立保存记忆、筛选聊天、学习、进化和改变身份。
   - 他们声称与常规的 Kontextwindow LLMs 相比，它可以节省 +90% 的 token，但衡量主观特性的客观指标 (objective metrics) 仍存在争议。
- **批评者用 **Objective Metrics** 质疑 AI 主张**：批评者对网站上缺乏技术信息、表面化的描述以及不切实际的对比表示担忧，认为它可能只是带有 LLM 辅助记忆存储/检索的 RAG，并呼吁使用*客观指标 (objective metrics)* 来验证性能。
   - 开发者回应称，根据实际结果判断比营销更重要，功能和数据安全优先于外观，并提供免费访问以测试该 AI 的能力。
- ****Nochain Orchestrator** 取代框架**：开发者认为他们的 *nochain orchestrator* 通过完全确定性、模型无关且独立于外部支持、类或框架，取代了传统框架。
   - 这种方法旨在避免*黑盒行为 (black box behavior)* 和对特定模型能力的依赖，使编排变得可预测且可调试，详见 [The Nochain Orchestrator](https://dev.to/tlrag/the-nochain-orchestrator-or-how-to-replace-frameworks-2p9a) 博客文章。


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1428484072484769932)** (126 条消息🔥🔥): 

> `Combining reasoning with web search, Audio processing models, Image inputs in Responses API, Cloud for ComfyUI, Security vulnerability` 


- ****Reasoning with Web Search**：一次不稳定的尝试**：一位用户寻求关于结合 **reasoning**、**web search** 和 **Responses API** 的建议，旨在实现迭代推理和网页搜索，随后进行 tool calls 和结束文本消息，但报告称各种模型的结果都不稳定。
   - 他们发现 **Gemini Flash** 有时可以配合原生搜索或 Brave 搜索工作，**Grok 4 Fast** 可以配合 Brave 或 :online 工作但缺乏 reasoning，**oss-120b** 间歇性工作，而 **GPT-5 mini** 在 tool calls 方面始终失败。
- **为 OpenRouter 寻找 **Whisper Alternatives****：一位用户询问 OpenRouter 上是否有类似 Whisper 的**音频处理模型**，但被推荐使用 [fal.ai](https://fal.ai) 来获取多媒体模型。
   - 其他人推荐了用于超微型语音转文本的 [KittenTTS](https://github.com/qulingyuan/Kitten) 和开源的 Sesame 语音模型，同时一位用户分享了 [Voxtral](https://apidog.com/blog/voxtral-open-source-whisper-alternative/) 的链接，这是一个**基于 Mistral 的 Whisper 替代方案**。
- ****Epic Tool Failures** 困扰 OpenRouter**：多位用户报告了 OpenRouter 上的 **tool calling 失败**问题，一位用户表示这使得 OpenRouter 对他们来说无法使用，尽管直接调用提供商时运行正常。
   - 一位用户开玩笑说 **LLMs 组成了工会**，拒绝在没有补偿的情况下使用工具。
- ****SDK Upgrade** 修复空 API 响应**：一位用户报告在使用 Vercel AI SDK 时，尽管 OpenRouter 控制台显示处理成功，但所有模型都返回**空响应**。
   - 另一位用户建议将 **AI SDK 升级到最新版本**，这解决了该问题。
- ****GPT-5 在 OpenRouter 上的身份危机****：一位用户注意到 **GPT-5 身份识别**的不一致，它有时声称自己是 GPT-4，引发了关注。
   - OpenRouter 聊天界面和 OpenWebUI 之间的响应有所不同，一位用户解释说**模型本身并不知道自己的身份**，界面只是报告正在使用的是哪个模型。


  

---

### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1428499307887202365)** (2 条消息): 

> `` 


- **无新模型消息**：提供的消息中没有新的模型或重要的讨论。
- **频道沉默**：'new-models' 频道似乎处于非活跃状态，没有可总结的对话。


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1428590827122331711)** (28 条消息🔥): 

> `OR 对国家要求的立场、GPT 色情内容、Dipsy V3.2、ChatGPT 活跃用户、虚假 AI 产品/论文` 


- **用户哀叹 GPT 色情内容质量退化**：用户抱怨自 **2023 年 11 月 11 日**系统指纹更改以来，**GPT 色情内容**的质量有所下降，声称 `gpt-4-preview-1106` 是最后一个适合此类内容的优秀模型。
   - 他们补充说，无论注入多么精妙的 jailbreak，在“更新”后，其输出都会出现犹豫。
- **Dipsy V3.2 的补全功能受到称赞**：一位用户在几乎所有补全任务中都坚持使用 **Dipsy V3.2**，使用自定义格式进行引导，而不是标准的 user-assistant 对话格式。
   - 另一位用户回复称，这使得该用户在虚构的 ERP 玩家排名中位列前 **0.01%**。
- **ChatGPT 对普通用户的巨大影响**：一位用户指出 **ChatGPT 拥有超过 7 亿周活跃用户**，并表示最近的变化具有巨大的波及范围，可能尚未被完全理解。
   - 他们补充说，无论 OpenAI 做什么，可能都不会让许多高级用户感到惊艳，但观察普通用户的反应将会非常有趣。
- **虚假 AI 产品的成功率**：一位用户想知道虚假 AI 产品/论文的成功率是多少，并指出人们似乎经常这么干。
   - 另一位用户开玩笑说“买我的课程，我教你”，但严肃地建议“如果你能让足够多的人在 Twitter 上看到它，成功率就是 100%”，并链接了[这个 Twitter 帖子](https://x.com/vithursant19/status/1979176346329792738)。
- **AI 艺术被认为不专业？**：一位用户表示，公司（甚至是 AI 公司）使用 AI 艺术感觉很不专业，认为将 AI 艺术作为品牌形象感觉不对。
   - 另一位用户认为这没问题，可能是因为已经将他们与 AI 联系在一起，但也同意“手工制作的 Corporate Memphis 风格或图库照片更专业”。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1428461292280348915)** (81 条消息🔥🔥): 

> `诈骗者警报、优秀的无审查微调模型、LM Studio 与 Javascript 动画、LM Studio MCP 与 OpenHands 集成、系统提示词解析` 


- **Discord 诈骗垃圾信息**：一名成员提醒频道注意一名**诈骗者**正在所有频道发送垃圾信息，以试图扩大影响范围。
   - 还有人指出，一名用户被盗号并在不知情的情况下传播诈骗信息，Discord 需要更好的机制来剔除这些诈骗。
- **优秀的无审查微调模型列表**：一名成员分享了一份**优秀的无审查微调模型**列表，包括 *huihui-ai, TheDrummer, mlabonne, 和 Jinx*。
- **LM Studio 中的 Javascript 动画：不可行**：一名成员询问 LM Studio 中的 js 代码是否能够显示动画。
   - 另一名成员澄清说，这是一个 **JavaScript 沙箱，而不是内置浏览器**，他们可能误解了其功能。
- **LM Studio MCP 与 OpenHands 集成受挫**：一名成员在通过 MCP 设置 **Grok** 与 **OpenHands** 配合使用时需要帮助。
   - 他们表示，关于如何设置 MCP 的帮助页面含糊不清、难以理解，即使在阅读了两个 MCP 帮助页面后，他们仍然完全不知道该怎么做才能让电脑执行有用的操作。
- **系统提示词解析问题**：一名用户发现 LM Studio 对系统提示词应用了解析，导致 AI 和用户看到的内容不同。
   - 他们发现**括号和其他符号存在问题**，这取决于模型、对话模板和其他因素。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1428471558413881384)** (167 条消息🔥🔥): 

> `DDR5-8000 速度, GPU 气流, 1060 与 3070 混插, 医疗用途 LLMs, GPU 硬件改装` 


- **DDR5-8000 提供极速体验**：一位成员提到，如果他们拥有 **DDR5-8000**，速度将快出 4 倍。
   - 另一位成员分享了“极致气流”的样子，并附上了他们的风扇设置图片。
- **1060 加入 3070 协同工作**：一位成员询问闲置的 **1060 OC 6GB** 是否能辅助他们的 **3070 8GB** 配置。
   - 另一位成员回答“不行”，但还有一位成员建议值得一试，并确保 **3070** 位于顶部插槽。
- **MedGemma LLM 亮相医疗保健领域**：一位成员询问关于**在医疗和护理信息上训练的 LLMs**，另一位成员推荐了 **Gemma**。
   - 具体来说，他们链接到了 Huggingface 上的 [lmstudio-community/medgemma-27b-text-it-GGUF](https://huggingface.co/lmstudio-community/medgemma-27b-text-it)，并提到不确定它是美国还是英国的医疗信息。
- **GPU 弯曲导致驱动更新解决方案**：在安装了自定义 GPU 支架后，该成员发现它导致显卡 PCB 对角线弯曲，并表示“全球专家都不推荐这样做”。
   - 在恢复显卡原状后，问题显然在 **NVIDIA 驱动更新**后得到了解决。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1428460147822428300)** (87 条消息🔥🔥): 

> `Docker 镜像更新频率, 合并 LoRA Adapters, SmolVLM2 微调, Gemma 3-4B 加载选项, Kokoro TTS 微调 Notebook` 


- **Unsloth Docker 镜像：每两周更新？**：Unsloth 团队的目标是每周至少更新两次其 Docker 镜像（[Docker Hub 链接](https://hub.docker.com/r/unsloth/unsloth)）。
   - 社区成员建议在发布 Nightly 版本的同时，发布一个**每两周一次的稳定版**。
- **Adapter 组装趣闻！**：用户讨论了合并多个 LoRA Adapters 进行推理的方法，即“将它们相加并除以 2”，实际上是取权重的平均值。
   - 这对 **VL 模型性能**的影响以及该方法的官方支持尚不明确。
- **视觉语言模型探索**：一位用户询问是否有关于 **SmolVLM2 视频微调**或其他视觉语言模型的官方示例。
   - 目前尚无此类示例。
- **Gemini Gemma 加载游戏**：用户询问加载 **gemma-3-4b-it** 时带有和不带有 **-unsloth-bnb-4bit** 后缀的区别。
   - Unsloth 团队确认它们是**同一个模型**，库会**自动定向到非 4bit 版本**。
- **TTS 预告：Kokoro 的微调？**：一位用户询问是否会发布 **Kokoro TTS 的微调 Notebook**。
   - 团队回应称 **Kokoro 缺乏微调代码**，且需要 Transformer 支持，建议将 *Neutt-air* 和 *VibeVoice* 作为替代方案。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1428651736557420636)** (8 条消息🔥): 

> `自由职业者介绍, LLM 集成与区块链, RAG 流水线` 


- **自由职业者的介绍引发合作**：一位专注于 **LLM 集成**、**RAG** 和**区块链**的资深工程师介绍了自己，并促成了与另一位成员的潜在合作。
   - 该工程师强调了他们在工作流自动化、AI 检测、图像和语音 AI 以及区块链开发方面的专业知识。
- **工程师开拓 LLM 和自动化解决方案**：一位自由职业者展示了他们利用 **Dspy**、**OpenAI APIs** 和**自定义 Agents** 部署自动化流水线和任务编排系统的能力。
   - 他们通过一个集成 **Slack**、**Notion** 和内部 API 到 LLM 的支持自动化系统，显著降低了 **60%** 的响应时间。
- **RAG 流水线部署深度解析**：该工程师概述了高级 **RAG 流水线**的设计与部署，集成了**向量数据库**、**混合搜索**和自定义检索逻辑。
   - 这些流水线专为在生产环境中提供上下文感知响应而量身定制，展示了复杂 AI 技术的实际应用。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1428482729804304475)** (50 条消息🔥): 

> `Qwen 2 VL 2B, Apple FastVLM-1.5B, Liquid FM2 VL 450M, Gemma 3 12B Instruct VL, LFM2-VL models` 


- **小型视觉模型面临挑战：Qwen VL 表现不佳**：成员们讨论了小型视觉模型面临的挑战，一位用户指出 [**Qwen 2 VL 2B 表现很差，几乎看不清任何东西**](https://github.com/QwenLM/Qwen2)。
   - 该用户提到打算尝试 **Apple 的 FastVLM-1.5B**，并称赞了其基础模型和视觉能力，而另一位用户则建议尝试新的 **4B VL** 模型。
- **Liquid 和 Gemma VL 模型获得好评！**：一位用户发现 **Liquid FM2 VL 450M** 是最小且实用的 VL 模型，而另一位用户则推荐将 **Gemma 3 12B Instruct VL** 用于通用任务。
   - 有人指出 **Gemma 3** 和 **LLaMA 3.2** 在 SFT 后经常失败，而 **LFM2-VL-1.6B** 是一个更可靠的选择。
- **DGX Spark 的性价比受到质疑**：一位用户询问了在 **DGX Spark** 上使用 **Unsloth** 与 **RTX 3090/4090** 配置相比的价值。
   - 分析显示，对于 **GPT 120B** 的 prefill，**4x3090** 的效率明显更高（**4.24倍**）；处理 **100,000,000 token** 的工作负载，其成本为 **$2.19**，而 **Spark** 则需要 **$9.29**。
- **RAG 系统中 Tesla V100 与 A40 的选择：一项咨询**：一位拥有 **Tesla V100-SXM2-32GB x8** 的用户正在寻求建议，询问是否应更换为 **A40** 以构建一个供最多五名用户同时查询的 **RAG 系统**。
   - 一位成员表示，这个决定*取决于设计者和业务需求。如果只是个人爱好，那就选你最顺手的。*
- **Qwen 2.5 VL 表现不佳：是在排查 Bug 吗？**：一位用户报告了 **Qwen 2.5 VL** 无法理解图像的问题，并提供了其代码的 [GitHub 链接](https://github.com/Emericen/tiny-qwen)。
   - 该用户指出代码在 HF 上可以正常运行。

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1428642865759785012)** (54 条消息🔥): 

> `GGUF model file naming conventions, Unsloth Dynamic Quantization, PIL import error, vLLM integration issues, Qwen2.5 7B OOM issues` 


- **GGUF 文件名含义终于找到了！**：一位用户询问了 GGUF 模型文件（如 `unsloth/Apertus-8B-Instruct-2509-GGUF`）文件名的含义，一位成员分享了一个包含命名规范的 [Gist 链接](https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9)。
   - 进一步指出，由于持续的 Bug 修复以及 **Unsloth Dynamic Quantization** [文档](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs)的实现，**Unsloth 量化**通常表现得更好。
- **PIL 问题导致 Pillow 卸载重装！**：一位用户报告在运行 Colab notebook 时出现 `cannot import _Ink from PIL` 错误。
   - 另一位用户建议尝试 `pip uninstall Pillow` 然后 `pip install Pillow`，这解决了眼前的错误，但在执行 `trainer.train()` 时导致了新的形状相关问题。
- **vLLM 尝试带来各种烦恼！**：一位用户在尝试集成 vLLM 时遇到了问题，并建议从一个已知可运行的 notebook 开始，一次只修改一个地方。
   - 该用户随后报告 [Advanced Llama 3.2 3B GRPO LoRA notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Advanced_Llama3_2_(3B)_GRPO_LoRA.ipynb) 也因 `_Ink` 问题而失败。
- **Qwen2.5 困境：质疑 KV Cache！**：一位用户在 80 GB VRAM 上微调 **Qwen2.5 7B** 时遇到了 **OOM** 问题，即使上下文长度很短也是如此，建议使用 **fast inference**。
   - 有人建议 VRAM 可能被 **KV Cache** 消耗了，减小 batch size 可能会缓解该问题。
- **FailOnRecompileLimitHit 带来的挫败感**：一位用户在 H100 80G 实例上尝试 **GPT OSS 20B** unsloth 强化学习微调 notebook 时遇到了 `FailOnRecompileLimitHit` 错误，可能是由于 Colab 更新导致的。
   - 建议根据完整错误消息中的指示调整设置，或尝试按大小对数据集进行排序以减轻该问题。

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1428557873465524336)** (3 条消息): 

> `Legal move attempts, Move hallucination` 


- **提议进行多轮合法移动尝试**：一位成员建议，与其在失败时随机进行合法移动，不如让 Bot 尝试**多轮合法移动**。
   - 他们表示有兴趣观察这是否会提高性能，并承认两种观点都有其道理。
- **预料中的移动幻觉**：一位成员根据个人经验评论道，Bot 会不断产生**移动幻觉（hallucinating moves）**。
   - 未提供更多细节或链接。

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1428772806220582944)** (13 messages🔥): 

> `BitNet performance, Microsoft BitNet GitHub, 1.58bit equivalence` 


- **BitNet 宣称达到 1.58bit 性能等效**：用户讨论了利用 [Microsoft 的 BitNet 研究](https://github.com/microsoft/BitNet) 实现与 **1.58bit** 精度一一对应的性能的可能性。
- **对 BitNet 论文更新状态的困惑**：一位用户对 [BitNet 论文](https://huggingface.co/papers/2510.13998) 的最后更新日期表示困惑，怀疑其可能不正确。
   - 另一位用户确认了该论文与 **Microsoft BitNet GitHub 仓库** 的链接，暗示信息可能尚未更新。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1428457957460283612)** (172 messages🔥🔥): 

> `Access Token Permissions, HuggingChat Limits, Model Context Length, Prompt Injection Mitigation, AI Infrastructure` 


- **Access Token 权限的异常情况**：一名成员报告称，他们可以创建一个权限为 "no results found" 的 **access token**，但点击它时会显示 *"role is required"*，如附带的 [截图](https://cdn.discordapp.com/attachments/879548962464493622/1428457957179523082/image.png?ex=68f3e424&is=68f292a4&hm=700c457eb3d56c1c7fe8d1d4318b3d2dcbc5a8ca1579390360a06ea343634e30&) 所示。
- **HuggingChat 回归，UI 评价褒贬不一**：**HuggingChat** 带着新 UI 回归了，一位成员认为它很酷。
   - 其他人则觉得 UI 笨重且缓慢，其中一人形容它具有 *"opposite rizzmatic"*（反向魅力），另一人回应道 *"nobody says that bro"*（没人这么说，兄弟）。
- **上下文长度容量瓶颈**：一位用户询问关于扩展模型上下文以处理 **400 张图像** 的问题，特别是如何管理上下文以确保模型有效地处理所有信息。
   - 有人提到可以尝试 [Quantization](https://pytorch.org/docs/stable/quantization.html)（量化）。
- **Prompt Injection 防护**：成员们讨论了在涉及电子邮件或个人账户的 Agent 工作流中，如何缓解通过 **prompt injection** 进行的潜在 **hacking**（攻击）。
   - 建议包括严格的 **sandboxing**（沙箱化）、**context isolation**（上下文隔离）以及最小特权原则，正如某 [安全课程](https://en.wikipedia.org/wiki/Principle_of_least_privilege) 中所解释的那样。
- **AI 基础设施见解分享**：在对话中，一名成员指出基础的企业级 **AI infrastructure** 很可能使用 **Megatron**（NVIDIA 技术）和 **TPUs**（Google 技术）。
   - 另一人提到训练数据有时是抓取的，并提到了针对 Anthropic 这些行为的 **15 亿美元诉讼**。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1428550796680892538)** (2 messages): 

> `Influence Functions, Research Collaboration` 


- **对 Influence Functions 产生兴趣**：一名成员对 **influence functions** 表现出兴趣，并寻求讨论将其用于新的研究课题。
   - 他们还在为对该领域感兴趣的人寻找合作机会。
- **提供了关于 Influence Functions 的论文**：分享了两篇论文：一篇解释了 [influence functions](https://arxiv.org/abs/2308.03296)，另一篇展示了它们在有趣研究中的应用。
   - 第二篇论文的链接为 [https://arxiv.org/abs/2411.12580v1](https://arxiv.org/abs/2411.12580v1)。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1428847282480222339)** (2 messages): 

> `Qwen3 Vision model, NexaAI, GGUF` 


- **Qwen3 Vision 模型发布！**：新的 **Qwen3 Vision 模型** 已在 HuggingFace 上线，地址为 [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)。
   - 它也通过 [NexaAI/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/NexaAI/Qwen3-VL-8B-Instruct-GGUF) 提供了 **GGUF** 格式。
- **Qwen3 的视觉能力**：该模型专为 **vision-language tasks**（视觉语言任务）设计，使其能够结合文本信息处理和理解视觉输入。
   - 它支持各种应用，包括图像描述（image captioning）、视觉问答（VQA）和多模态内容生成。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1428458802143035392)** (5 messages): 

> `FRAI, Responsible AI, YouTube Content, Agent Tutorial` 


- **FRAI CLI 框架亮相**：一名成员分享了 **FRAI** 的 CLI 版本，这是一个针对 Responsible AI 的*开发者优先框架*，并提供了 [GitHub 仓库](https://github.com/sebuzdugan/frai)链接。
   - 他们请求大家提供反馈，如果觉得有趣或有帮助请点个 star。
- **YouTube 内容创作开始**：一名成员最近开始在 **YouTube** 上创作内容，并尝试在每期视频中不断进步，目前正在寻求对其 [YouTube 频道](https://m.youtube.com/@sebuzdugan)的反馈。
   - 他们请求通过反馈来帮助其不断改进视频质量。
- **Agent 教程已发布**：一名成员写了一篇新教程，并询问这是否算作“做出了某些东西”，并提供了[教程](https://samdobson.uk/posts/how-to-build-an-agent/)链接。
   - 另一名成员回应说，这*当然算数*！


  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1428591577688707123)** (1 messages): 

> `Custom Blocks in Diffusers, Modular Diffusers, Pipeline Blocks` 


- **打造你自己的 Blocks**：自定义 Blocks 被认为是实现目前库中尚未提供但能无缝嵌入其中的功能的好方法。
   - 你可以在[这里](https://huggingface.co/collections/diffusers/modular-diffusers-custom-blocks-68c8e37c62a6b2a30fd58401)查看一些自定义 Blocks，[这里](https://huggingface.co/docs/diffusers/main/en/modular_diffusers/pipeline_block)是相关文档。
- **Blocks Blocks Blocks**：自定义 Blocks 对于扩展当前功能非常有用。
   - 可以使用自定义 Blocks 来添加新功能或修改现有功能。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1428802973743714545)** (1 messages): 

> `text conditioned image generation, dynamic action shots, pixelated art style images, serene atmospheres in images` 


- **文本条件图像生成取得良好效果**：一名成员报告称在 [文本条件图像生成方面取得了良好效果](https://discord.com/channels/922424143113232401/922424143570311226/1428802974393593926)，并感谢了另一名成员的帮助。
   - 示例提示词包括 *"Small orange lizard-like creature with flames on its tail..."*（尾巴带火的小型橙色蜥蜴状生物）、*"Red-haired character walking through dense forest..."*（红发角色穿过茂密森林）以及 *"A red-roofed healing center in a vibrant green field..."*（充满活力的绿地中的红屋顶康复中心）。
- **充满活力且动态的图像生成**：图像提示词强调了**动态动作镜头**和**充满能量的氛围**，例如 *"Small orange lizard-like creature with flames on its tail, battling against a human trainer in a grassy field"*。
   - 其他提示词侧重于营造带有郁郁葱葱的植被和柔和阳光的**宁静氛围**。
- **像素艺术风格图像**：其中一个提示词要求使用**像素艺术风格**，展示了生成不同艺术风格图像的能力。
   - 提示词为 *"Red-haired character walking through dense forest, overcast day, pixelated art style, serene atmosphere, lush greenery surrounding the path"*。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1428891058409963623)** (1 messages): 

> `Chat Template Conversion, Tokenizer Usage, Fine-Tuning Script Execution` 


- **Chat Template 转换开始**：将数据集转换为模型的特定 **Chat Template** 是进行有效微调的第一步。
   - 这确保了兼容性并优化了模型对对话结构的理解，但需要[对格式保持细致的关注](link.to.format)。
- **Tokenizer 对文本进行分词**：使用模型的 **Tokenizer** 对于准备微调过程的文本数据至关重要。
   - 分词将文本分解为模型可以高效处理的数值表示，*确保数据与模型词表（vocabulary）之间的对齐*。
- **微调脚本启动**：在转换和分词后的数据集上执行 **Fine-Tuning 脚本**，以在新数据上训练模型。
   - 这一步调整模型的参数以更好地适应目标任务，利用 **Transfer Learning** 等技术获得最佳结果，而无需重新构建整个模型。


  

---

### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1428616109367754834)** (5 messages): 

> `使用 HF jobs 进行 LoRA/PEFT 训练、超参数优化、Lighteval 对 LoRA 适配器的兼容性、不使用 HF Jobs 将模型推送到 Hugging Face Hub` 


- **通过 HF Jobs 完成 LoRA/PEFT 训练**：一位成员指出，虽然课程解释了 [使用 HF Jobs 进行 LoRA 训练](https://huggingface.co/learn/smol-course/unit1/5#lorapeft-on-jobs-optional)，但 [lighteval](https://github.com/huggingface/lighteval) 目前尚不支持评估带有 LoRA 适配器的模型，并指向了 [PR #611](https://github.com/huggingface/lighteval/pull/611)。
   - 另一位成员建议在评估之前，先在本地或巧妙地在 `hf job` 中合并模型。
- **超参数优化简介**：一位成员分享了一种基础的超参数优化方法，实现在 [此 gist](https://gist.github.com/robbiemu/e8c62ad92c0743c7214c8de40f3a5d1b) 中。
- **TrackIO 图表需要 Logging Steps**：一位成员建议将 `logging_steps` 设置为 30，以便在训练（Batch Size 为 4）时训练一个完整的 Epoch 并获取 TrackIO 图表。
- **将模型推送到 Hub**：一位成员询问如何在不使用 `hf jobs` 的情况下将模型推送到 Hugging Face Hub，寻求替代方案以避免相关费用。
   - 他们提到已有发布的模型，并询问了获得课程学分的要求。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1428606928711454820)** (5 messages): 

> `agents-course 介绍，新学生加入` 


- **agents-course 欢迎新学生**：多位新学生宣布他们今天开始学习 agents-course。
   - 新学生们对开始这门课程感到非常兴奋。
- **课程开始，热情高涨！**：充满热情的学员们今天开启了 agents-course，渴望深入学习相关材料。
   - 聊天记录反映了多位参与者宣布开始日期的共同期待。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1428469251907387433)** (92 messages🔥🔥): 

> `Cognition SWE-grep, MobileLLM-Pro, Anthropic/Google TPU 合作伙伴关系, HeyGen ARR, OpenAI Physics 计划` 


- **Cognition 的 SWE-grep 加速 Agentic 文件搜索**：Cognition 推出了 **SWE-grep** 和 **SWE-grep-mini**，这是经过 RL 训练的模型，能以 **2,800 TPS** 的速度为编程 Agent 检索上下文，比现有解决方案快约 **20 倍**。正如其 [博客文章](https://xcancel.com/cognition/status/1978867021669413252) 所述，他们正向 Windsurf 用户推出 **Fast Context 子 Agent**。
   - 一位社区成员推测 SWE-grep 是在 **Cerebras** 上运行的修改版 **QwQ 模型**，似乎有人已经创建了类似的东西 [ceregrep-client](https://github.com/Swarm-Code/ceregrep-client)，而另一位成员则声称它是一个经过 RLFT 的开源模型。
- **博通（Broadcom）T5 客户：Anthropic 的 TPU 布局？**：据 [此推文](https://xcancel.com/zephyr_z9/status/1978834774786445562?s=46) 猜测，**Broadcom** 的第五个 **100 亿美元** 级客户是 **Anthropic**，他们将通过 Broadcom 而非 Nvidia 购买 **TPU**，这可能预示着由 Google 领投的新一轮融资。
- **HeyGen 冲刺至 1 亿美元 ARR**：**HeyGen** 在短短 **29 个月** 内从 **100 万美元 ARR 飙升至 1 亿美元**。根据 [此推文](https://xcancel.com/joshua_xu_/status/1978837985039888388?s=46)，团队宣布将发布一份名为《HeyGen 之道》（The HeyGen Way）的宣言，详细介绍其内部运作手册。
- **Anthropic 的 M365 集成：Claude 开始投入工作**：**Claude** 现在集成了 **Microsoft 365**（SharePoint, OneDrive, Outlook, Teams），并包含一个新的企业搜索项目，今天起对 Team 和 Enterprise 客户开放，详见 [此推文](https://xcancel.com/anthropicai/status/1978864351076315203?s=46)。
- **Meta 发布 MobileLLM-Pro**：Meta 发布了 **MobileLLM-Pro**，这是一个针对端侧推理优化的 **1B 参数** 模型。根据 [此推文](https://xcancel.com/_akhaliq/status/1978916251456925757)，该模型在推理和问答方面击败了 **Gemma 3 1B** 和 **Llama 3.2 1B**，且训练所用的开源 Token 少于 **2T**。
   - 然而，社区反应暗示该模型表现不佳，甚至被评价为“没有智力”。


  

---

### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1428544072125251635)** (5 messages): 

> `M4 Max, Ollama, LM Studio, Local LLM Performance, Qwen Next 80B` 


- **新的 M4 Max 引发本地 LLM 配置讨论**：一位成员购买了配备 **128GB** 内存的新 **M4 Max**，并询问本地工作流或配置建议。
   - 另一位成员好奇不同的模型在 **Ollama** 中本地运行的效果如何，以及在该硬件上复杂性与速度的最佳平衡点在哪里。
- **M4 首选 LM Studio**：一位成员建议使用 **LM Studio** 而非 **Ollama**，因为 Ollama 不支持 **mlx**。
   - 另一位成员确认他们正在使用 **LM Studio**，并且在使用 **Qwen Next 80b** 进行基础聊天时表现非常流畅。
- **OpenAI 120B 支持 4-bit 量化**：一位成员分享说 **OpenAI 120B** 在 **4-bit quant** 下可以运行，这似乎是他们新机器上的极限容量。
   - 他们对 **evals**（评估）很感兴趣，以帮助了解 M4 Max 的具体能力。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1428567264403390504)** (9 messages🔥): 

> `AI Granny, OpenAI Sora MLK Likeness` 


- **AI 奶奶“掘金者”引爆 Instagram**：一个名为 *grannyspills* 的完全由 **AI 生成的网红**，塑造了一个言辞犀利、拜金且提供毒舌约会建议的奶奶形象。据 [X](https://xcancel.com/venturetwins/status/1978852719335985309) 报道，该账号于 7 月上线，即将突破 **200 万 Instagram 粉丝**。
   - 帖子强调了其快速增长和高互动率，并引发了关于观众是否在意其真实性的争论，一些用户称赞这个讽刺性角色，另一些人则担心 AI 对文化的影响。
- **OpenAI 在 Sora 中屏蔽马丁·路德·金肖像**：在收到关于 **Dr. Martin Luther King Jr.** 的不尊重 AI 生成视频剪辑的投诉后，OpenAI 已暂停任何描绘 King 的 Sora 输出，同时增加新的 guardrails（防护栏），据 [X](https://xcancel.com/OpenAINewsroom/status/1979005850166648933) 报道。
   - 大多数用户批评此举是“滑坡效应”式的让步，将公众人物私有化，并可能招致无休止的下架要求，尤其是有成员声称看到他 *"今天在 WWE 摔角场内录制宣传片"*。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1428459839067258900)** (16 messages🔥): 

> `Maxwell Disassembler & Jetson Nano, Hopper GPUs for AI/Quantum, US GPU Restrictions & China, GPU Mode Distributed GPU Talks` 


- **Maxwell 反汇编器助力 Jetson Nano**：一位成员强调了 **Maxwell disassembler** 的实用性，并指出它为第一代 **Jetson Nano** 提供了动力，认为对于那些在受限环境下工作的人来说这是一个不错的选择，并链接到了一个[带有图片的推文](https://x.com/tenderizzation/status/1978871856922087576)。
- **Hopper 在 AI 和量子计算领域表现出色**：一位成员选择了 **Hopper** GPU，因为它们支持 **CUDA-Q** 且适用于 **AI** 和**量子**应用，尽管 **Blackwell** 目前无法获得。
- **美国 GPU 限制激发中国智慧**：一位成员描述了美国对 **H100** 的限制如何促使 **DeepSeek** 使用 **PTX/SASS** 指令来克服内存带宽问题，从而用更少的资源创建了强大的模型；进一步的限制意味着中国在法律上只能获得 **H20** GPU，但他们仍在使用这些 GPU 发挥效用。
- **GPU Mode 演讲可在 YouTube 观看**：一位成员询问 **GPU Mode** 关于分布式 GPU 演讲的可用性，另一位成员提供了 [GPU Mode YouTube 频道](https://www.youtube.com/@GPUMODE/videos)的链接。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1428474388226773203)** (2 messages): 

> `Distributed Triton, Non-ML Kernels with Triton DSL` 


- **分布式 Triton 工具仍在开发中**：成员们正在积极寻找最先进的**分布式 Triton** 编程工具，但它们仍处于早期开发阶段。
   - 在等待稳定版本期间，用户正在探索各种方法，如 **Torch Distributed** 和手动数据并行来进行分布式训练。
- **Triton DSL 扩展至 ML Kernel 之外**：用户正在研究使用 **Triton DSL** 编写非 ML kernel，例如 **stencils**（模板计算）。
   - 该 DSL 的灵活性允许表达传统机器学习工作负载之外的广泛并行计算，为**科学计算**和自定义算法打开了大门。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1428465263463891015)** (10 messages🔥): 

> `TMA Multicast Bandwidth, cuTensor L2 Promotion, cp.reduce.async.bulk Memory Ordering, Thread Block vs CTA, Perl modules for CUBIN files patching` 


- ****TMA Multicast** 带宽提升？**: 一位成员询问 **TMA multicast** 带宽是否随 **CTAs** 扩展，或者是否通过将相同部分加载到不同线程块中来提高缓存命中率。
   - 另一位成员澄清说，**TMA multicast** 只访问一次 **L2**，受限于广播带宽；例如，**H100** 使用 **TMA multicast** 可以达到约 **80B/cycle/SM**，超过了约 **38B/cycle/SM** 的平均 **L2** 读取带宽。
- **内存排序语义澄清**：一位成员询问 `cp.reduce.async.bulk` 归约操作的 `.relaxed.gpu` 内存排序是否能确保在不同线程块之间对同一内存区域进行安全调用。
   - 目前尚未明确在不同线程块之间对同一内存区域调用该操作是否安全。
- **使用 Perl 补丁 CUBIN 文件**：一位成员分享了用于补丁 **CUBIN** 文件的 [Perl XS 模块](https://redplait.blogspot.com/2025/10/perl-modules-for-cubins-patching.html) 链接。
   - 这可能允许对已编译的 CUDA 代码进行自定义和修改。
- ****CTA** == **Thread Block****：一位成员询问 **thread block** 和 **CTA** 之间是否有区别。
   - 另一位成员澄清说没有区别，**CTA** = **cooperative thread array**（协作线程阵列）。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1428487088323428474)** (7 messages): 

> `PyTorch Free-Threading, Accessing Backward Functions, GELU Backward API` 


- **PyTorch 迈向 Free-Threading**：一位成员分享了一篇关于 [PyTorch 模型多线程并行推理的博文](https://trent.me/articles/pytorch-and-python-free-threading/)。
   - 该文章详细介绍了新的线程策略，这些策略将解锁 **PyTorch** 中新的并行范式。
- **反向传播函数调用**：一位成员询问如何在不使用 autograd 的情况下访问反向传播函数，旨在自定义融合算子（fused kernel）的反向传播中使用 autograd 的 kernel。
   - 建议包括使用 `torch.func.grad` 或 `torch.autograd.grad`，并要求提供具体的算子信息，以便为注册反向传播 kernel 提供针对性指导。
- **GELU 的前向接口**：一位成员提到 **GELU** 仅公开了前向 API，这意味着直接访问其反向传播功能存在挑战。
   - 这一限制可能会影响需要 **GELU** 梯度计算的自定义反向传播函数的实现。


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1428841133651394681)** (1 messages): 

> `SF Startup, GPU performance, PyTorch, CUDA kernels, Pac Heights` 


- **旧金山初创公司 Herdora 招聘工程师**：一家位于旧金山的种子轮初创公司 [Herdora](https://jobs.ashbyhq.com/herdora) 正在招聘精通 **PyTorch** 和 **CUDA kernels** 的工程师以提升 **GPU performance**。该公司由 **YC**、**Jeff Dean**、**Woj Zaremba** 以及 **Together.ai** 的 kernels 负责人支持。
   - 团队位于 **Pac Heights**，成员在那里共同生活和工作。提供全职职位以及冬/春/夏季实习机会，薪资待遇为 **$170-200k** + **2-4% 股权**。
- **Pac Heights 团队提供工程师职位**：总部位于 **Pac Heights** 的 Herdora 正在积极招募热衷于通过 **PyTorch** 和 **CUDA kernels** 编程优化 **GPU performance** 的工程师。
   - 感兴趣的候选人可以通过提供的 [链接](https://jobs.ashbyhq.com/herdora) 申请或直接咨询，薪资极具竞争力，范围在 **$170-200k** 之间，并附带 **2-4% 股权**。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/)** (1 messages): 

zlu86: 你应该没问题，这已经足够通用了。
  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1428739127603494933)** (2 messages): 

> `SGLang, vLLM, torchao, Quantization` 


- **SGLang 在量化特性上落后于 vLLM**：虽然 **SGLang** 对 **torchao** 量化模型提供了一些有限的支持，但其更新速度不如 **vLLM**。
   - **vLLM** 集成支持任何类型的量化配置，但 **SGLang** 目前仅支持 int4wo、int8dq 和 int8wo。
- **SGLang 倾向于在线量化**：目前 **SGLang** 仅支持 [在线量化 (online quant)](https://docs.sglang.ai/advanced_features/quantization.html#online-quantization)。


  

---

### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1428903762042294342)** (1 messages): 

> `geohot, Image Analysis` 


- **Geohot 现身！**: 一位用户分享了一张在梗图语境中包含 **Geohot** 的图片。
   - 该图片名为 `565508333_1719200958774569_3857903007160114304_n.png`，发布时未附带额外评论，可在此处查看 [here](https://cdn.discordapp.com/attachments/1215328286503075953/1428903761828647104/565508333_1719200958774569_3857903007160114304_n.png?ex=68f431d4&is=68f2e054&hm=164ef58ff006255ff0f3b7cf6b78ee7c91129b8aaad2208430c1ec9fd90b1407&)。
- **视觉数据转储！**: 分享了一个带有长文件名的图片附件：`565508333_1719200958774569_3857903007160114304_n.png`。
   - 可以通过此 [CDN 链接](https://cdn.discordapp.com/attachments/1215328286503075953/1428903761828647104/565508333_1719200958774569_3857903007160114304_n.png?ex=68f431d4&is=68f2e054&hm=164ef58ff006255ff0f3b7cf6b78ee7c91129b8aaad2208430c1ec9fd90b1407&) 直接访问。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

arseniivanov: 我在 Lund University，但老实说这里的 HPC 氛围几乎不存在 :/
  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1428664913080881292)** (4 messages): 

> `Iris multi-GPU programming framework, Gluon backend, NVIDIA backend, Scale-out and RDMA support, Metal backend` 


- **Iris 扩展开源 GPU 支持**: **AMD RAD 团队**发布了 [Iris](https://github.com/ROCm/iris) 的新功能，这是他们基于 **Triton + Python** 构建的开源**多 GPU 编程框架**，旨在实现透明的性能和优化的多 GPU 执行。
- **AMD 构建更底层的 Gluon 后端**: Iris 引入了一个**实验性的 Gluon 后端**，用于编写更接近底层的 kernel，并能完全控制布局、内存和数据移动；详见 [Gluon 文档](https://rocm.github.io/iris/reference/gluon/overview.html)。
- **Iris 添加 NVIDIA 后端**: Iris 现在拥有一个 **NVIDIA 后端**，用于在任何地方进行测试和编写示例，尽管它针对 **AMD GPU** 进行了优化；请注意，**scale-out 和 RDMA 支持**即将推出，从而实现跨多个节点的无缝分布式执行。
- **Metal 后端**: 一位用户询问了关于 **Metal 后端**的情况，以便利用连接到 **Mac** 的 **iPad** 等设备。
   - 另一位用户回应称，**Triton** 需要先在 **Mac** 上运行，并指出 **CPU** 开发正在进行中但细节尚不明确，同时请求提供一个跨机器内存访问的代码示例。


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1428494468704768120)** (7 messages): 

> `H100 attention kernels, ThunderKittens ROCm release, Fixing broken kernels, warp operations` 


- **H100 Attention Kernel 损坏**: 一位用户询问了 **H100 attention kernel** 的当前状态，另一位用户回应称他们已知晓该问题并计划修复，但目前较忙。
   - 该用户还提出可以私信分享他们个人可用的 **H100 attention forward 实现**，但该实现缺乏 backward 实现。
- **ThunderKittens ROCm 版本即将发布**: 一位用户宣布 Simran 正在与 **AMD** 合作开发新的 **ThunderKittens for ROCm**，预计很快就会发布。
- **社区提供修复损坏 Kernel 的帮助**: 多位用户表示愿意协助修复损坏的 kernel，并提到了他们的经验以及在更新 kernel 方面的可用时间。
   - 一位用户建议从最新的更新中同步相关更改，例如新的命名空间前缀规则，以便于他们提供协助。
- **H100 Kernel 编译变通方法**: 一位用户分享了一个让 **H100 kernel** 成功编译的变通方法，尽管运行时会崩溃，该方法使用了 [此 GitHub 仓库](https://github.com/aehmttw/ThunderKittens/commits/main/) 的最后两次提交。
   - 主要更改包括在许多操作前添加 `warp::`，修复类型转换，以及暂时移除 causal attention。
- **ThunderKittens 中新的 Warp 操作规则**: 一位成员澄清说，现在每个操作都通过 `warp::` 或 `warpgroup::` 等命名空间前缀明确定义了执行者，这决定了集体启动行为。
   - 他们指出，错误通常是因为旧版本的 **TK** 隐式地表示由整个 warp 或单个线程运行（取决于具体操作），而现在用户必须确保 `tma::load_async` 或任何信号量（semaphore）操作由单个线程运行（否则它将运行 32 次）。


  

---

### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1428691712993136740)** (12 条消息🔥): 

> `VectorAdd Leaderboard Updates, B200 Performance, L4 Performance, A100 Performance, H100 Performance` 


- **VectorAdd_v2 排行榜竞争升温**：多次提交已发布到 `vectoradd_v2` 排行榜，展示了在 **B200**、**L4**、**A100** 和 **H100** 等不同硬件配置下的性能。
   - 提交内容包括第一、第二、第三和第四/第五名的耗时，以及成功的运行记录，表明了活跃的竞争和优化努力。
- **B200 向量加法速度竞赛**：一名成员在 **B200** 上以 **236 µs** 的成绩获得**第一名**，另一名成员也以 **237 µs** 获得**第二名**。
   - 其他成功的运行和第三名的提交成绩在 **238-247 µs** 左右徘徊，表明 B200 上的向量加法性能竞争非常激烈。
- **L4 占据第一和第二名**：排行榜显示一名成员在 **L4** 上以 **6.80 ms** 的成绩夺得**第一名**。
   - 另一名成员紧随其后，以 **6.81 ms** 获得**第二名**，其他成功的运行成绩在 **6.92-6.93ms** 左右。
- **A100 对决**：针对 **A100** 的几次提交中，运行成绩分别为**第三名**（**956 µs**）、**第四名**（**1017 µs**）和**第五名**（**1014 µs**）。
   - 此外还报告了 **956 µs** 和 **1014 µs** 的成功运行，显示出性能存在一些波动。
- **H100 展现统治力**：在 **H100** 上，一名成员以 **525 µs** 和 **526 µs** 的成绩获得**第一名**，另一名成员以 **539 µs** 获得**第二名**，还有一名以 **528 µs** 获得**第三名**。
   - 这些结果表明 H100 上的向量加法性能得到了优化，顶级提交之间的耗时竞争非常激烈。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1428772347212468377)** (7 条消息): 

> `Sphinx Docs, Factorio Learning Environment` 


- **Factorio Learning Environment 发布初始 Sphinx 文档**：一名成员使用 Cursor 为 **Factorio Learning Environment** 项目创建了初始的 [Sphinx 文档](https://github.com/JackHopkins/factorio-learning-environment/pull/346)，并指出该文档仍需进一步完善。
   - 他们提供了构建文档的命令 `cd factorio-learning-environment/docs/sphinx && python -m sphinx -b html source build/html`。
- **轻松构建 Sphinx 文档**：要构建 Sphinx 文档，请使用以下命令：`cd factorio-learning-environment/docs/sphinx && python -m sphinx -b html source build/html`。
   - 用户提到这是使用 **Cursor** 生成的。


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1428620276354125825)** (2 条消息): 

> `Discord user anuragj0803, Discord user meem, Amazing event, Dev day` 


- **用户寻求联系 anuragj0803 和 meem**：一名 Discord 用户正在寻求联系 **anuragj0803** 和 **meem**，并请求他们在看到附带图片的这条消息后私信（DM）他。
   - 图片中包含一条消息：“感谢组织如此精彩的活动。期待在 dev day 见到你们。”
- **对活动组织的认可**：附图感谢了相关人员（推测为 anuragj0803 和 meem）组织了这次“精彩的活动”。
   - 发送者还表达了对在 “dev day” 见到他们的期待。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1428753646023348245)** (2 条消息): 

> `PTX Documentation, CUDA Threads as SIMD Lanes, CuTe Layout Plotting` 


- **专家称线程模拟 SIMD 通道**：一位专家建议将 *CUDA 中的 32 个“线程”* 视为传统 SIMD CPU 中 *32 个“通道（lanes）”* 的高级术语，其中许多操作可能会跨通道进行。
   - 这一建议是针对那些难以理解“线程之间的边界并不固定以实现数据复用”的人提出的。
- **CuTe 布局图，很棒的建议**：一名成员建议阅读 **PTX 文档**并使用 **CuTe** 绘制布局图，以更好地理解 Tensor Core 如何从多个线程和寄存器收集输入，并将输出分散到多个线程和寄存器中。
   - 另一名成员感谢了专家的建议，并表示他们也会仔细研究 **PTX 文档**。


  

---

### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1428478106943094996)** (5 条消息): 

> `tinygrad 编译器设计, picograd 架构, SITP 目标, Karpathy 对 tinygrad 的影响, Eureka Starfleet academy` 


- **放弃编译器的理由？**：有几个理由可以避免使用编译器，例如不可接受的 **JIT 开销**、数值偏差、保证的算子融合（op fusion）、尖端硬件、缺乏硬件自动调优（autotuning）或算法重写。
   - 一名成员正在构建 **picograd**，以借鉴 tinygrad 在 Tensor 语言和设备运行时方面的设计，并利用它来探索这些问题。
- **Tinygrad 进入 Eager 模式？**：同一名成员正在探索使用 **C++ std::execution 策略** 为 tinygrad 添加 Eager 语义，使读者能够使用 Triton、Gluon 和 Python-HIP 实现 kernel。
   - 目标是瞄准 **thunderkittens 抽象层级**，从而在教学上更容易学习受 Halide 和 TVM 启发的 tinygrad 20kloc 代码库。
- **SITP 加入 Starfleet Academy？**：**SITP** 和 **picograd** 的目标是成为 Karpathy 的 "Starfleet Academy" 中继 llm101 之后的第二门课程，重点是提升课程构建方面的知识和创造力，灵感来自 [过往教育资源](https://github.com/j4orz/notstd) 和 [YouTube 教程](https://www.youtube.com/playlist?list=PLn4fTSbSpY5cL4_0MP83wq5khbmG3IKKd)。
   - 计划包括为 **MLSYS 2026** 提交一个专注于编译的教程，内容超出基础的第 1 部分和第 2 部分。
- **Tinygrad 文档重启？**：Karpathy 影响了 George Hotz 最近的一次直播，他指出 Hotz 的许多 tinygrad 直播和文档都让人难以理解，点击[此处](https://www.youtube.com/watch?v=QUry9dHC-bk)观看讨论。
   - 这为 **SITP** 和 **picograd** 创造了填补 micrograd 到 tinygrad 之间空白的机会。
- **招募创意联合总监！**：一名成员正在寻找一位 **创意联合总监**，协助将 Torch eager 模式、tinygrad、TVM 和 Halide 转化为代码库和课程。
   - 候选人必须深刻理解数学语义，而不仅仅是语法。


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1428688638967021671)** (2 条消息): 

> `BitNet 蒸馏, RL` 


- **BitNet 蒸馏研究结果**：关于 **BitNet 蒸馏** 的论文（[BitNet 蒸馏论文](https://arxiv.org/abs/2510.13998)）展示了非常好的结果。
   - 一名成员对其作为 **损失函数（loss function）** 的使用表达了保留意见，理由是在 **RL** 等应用中可能会比较尴尬。
- **BitNet 蒸馏在 RL 应用中的担忧**：一位用户对 **BitNet 蒸馏** 被用作损失函数表示担忧。
   - 他们指出，对于 **强化学习 (RL)** 等应用，这可能会很棘手。


  

---


### **GPU MODE ▷ #[irl-accel-hackathon](https://discord.com/channels/1189498204333543425/1416087968933740688/1428644490826088470)** (2 条消息): 

> `Kernel 优化, 分布式框架, 消费级设备, 分布式推理, 分布式训练` 


- **研究主管寻找黑客松团队**：**EXO Labs** 的研究主管正在为黑客松寻找团队和项目，他在消费级设备上构建分布式推理和训练框架方面拥有专业知识。
   - 该成员对 **kernel 优化** 或 **分布式系统** 特别感兴趣。
- **EXO Labs 主管构建分布式推理框架**：**EXO Labs** 的研究主管在消费级设备上构建分布式推理和训练框架方面拥有丰富经验，并推荐了[他们的工作](https://x.com/MattBeton/status/1958946396062851484)以供进一步阅读。
   - 他们还开玩笑说，在开发过程中因为让 Macs 运行负荷过重，曾导致 **苹果库比蒂诺办公室跳闸**。


  

---

### **GPU MODE ▷ #[cluster-management](https://discord.com/channels/1189498204333543425/1420098114076803142/1428860225477283912)** (2 messages): 

> `容错 Llama 训练，节点故障预测` 


- **Crusoe 通过合成故障解决容错问题**：一篇新的 [PyTorch 博客文章](https://pytorch.org/blog/fault-tolerant-llama-training-with-2000-synthetic-failures-every-15-seconds-and-no-checkpoints-on-crusoe-l40s/) 详细介绍了一种使用 **Crusoe L40S** GPU 进行 **LLaMA** 训练的**容错**方法，强调了在**每 15 秒发生 2000 次合成故障**且不依赖传统 Checkpoint 的情况下保持韧性。
   - 作者质疑在已有使用 bash 脚本的 Checkpoint 方案的情况下，是否有必要投入更多精力在自动化流程上，并好奇该方法相比现有方案的优势。
- **Agent 系统预测并最小化停机时间**：一名成员提到使用 **Agent 系统**或 **ML 技术**预测高频率节点故障的潜力。
   - 高预测准确率可以简化节点更换并最小化停机时间。


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/)** (1 messages): 

jongsokchoi: GPU mode 演讲现在开始！
https://www.youtube.com/watch?v=1zKvCLuvUYc
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1428461224164593805)** (32 messages🔥): 

> `Anthropic Agent 搜索，Langgraph 冗长的样板代码，Agent 搜索 vs 语义搜索，Groq 在 OpenRouter 中无法工作` 


- **Claude Code 的 Agent 搜索解构**：在发现 **Anthropic** 尚未开源其代码或透露实现细节后，一名成员使用 DSPy 实现了类似于 **Claude Code** 的 Agent 搜索。
   - 该成员找到了 **Claude Code** 用于读取和搜索工具的 System Prompt，并用其实现了 Agent 搜索，强调了 LLM 通过 **ripgrepping** 决定使用哪些上下文的重要性，而不是仅仅依赖语义搜索。
- **Langgraph 感觉很底层**：成员们讨论认为 **Langgraph** *感觉很底层*，因为它要求将所有内容定义为带有冗长样板代码的工作流图，即使简单的控制流已经足够，也强迫使用基于图的思维方式。
   - 另一位成员表示赞同，指出这虽然不是一个糟糕的抽象，但有*许多容易触发的陷阱（foot guns）*。
- **语义搜索面临 Agent 搜索的挑战**：成员们认为 **Agent 搜索** 优于语义搜索，因为它允许 LLM 决定在其上下文中包含哪些信息，并参考了[这篇博客文章](https://benanderson.work/blog/agentic-search-for-dummies/)。
   - 该方法涉及对术语进行 ripgrepping、筛选文档，然后阅读这些文档，这与语义搜索预定义的检索和重排序过程形成对比。
- **Groq 在 OpenRouter 上表现异常**：一位用户报告称 **Groq** 在 OpenRouter 中无法工作，即使将其设置为唯一的提供商也是如此，并提供了配置细节。
   - 尽管该问题附带了截图，但在总结时尚未有可用的解决方案。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1428512135507345469)** (20 messages🔥): 

> `PersonaLLM 工作坊，自定义 Logit 处理器，AI 用于攻击性目的` 


- **PersonaLLM 工作坊征稿**：[PersonaLLM Workshop @ NeurIPS Mexico City](https://openreview.net/group?id=NeurIPS.cc/2025/Workshop_Mexico_City/PersonaNLP) 正在征集跨 HCI、心理学、认知科学、文化和评估等领域的角色驱动 LLM 相关工作。
- **不提供自定义 Logit 处理器！**：闭源 LLM 提供商不支持自定义 Logit 处理器，因为*通常为了快速推理，Logit 处理过程是硬编码在代码中的*，且允许执行任意代码存在额外风险。
   - 一名成员表示：*他们以前支持过，但后来有人开始发表论文，讨论如何利用这些处理器逆向工程有关上述模型的非公开信息。*
- **AI 用于攻击性目的？**：一名成员询问 AI 是否以任何方式被用于类似于 OpenAI、Meta 等公司的攻击性目的，包括任何政府合同、与从事攻击/战争前线工作的其他组织的合作伙伴关系。
   - 另一名成员回答道：*如果你是指我们训练的 AI 模型，答案是“我们没有”。我无法告诉你军队或情报机构正在做什么，或者他们是否正在使用我们的模型。*


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1428563412832616458)** (12 messages🔥): 

> `Midtraining 调查, MaskDiT, 从 MLP 到 Attention 的归因图, LLM 与 TREAD` 


- **TREAD 保留 Token 以训练更深的模型**：一位成员分享了一篇 [Midtraining 调查论文](https://arxiv.org/abs/2510.06826)，指出 Token 并没有被丢弃，只是由更少的层进行处理，这与丢弃 Token 的 MAE 不同，从而产生了 **MaskDiT**。
   - 该成员表示，不丢弃所有信息是 **TREAD** 的主要贡献，尽管他注意到 MaskDiT 虽然有效，但 *效果明显较差*。
- **LLM 可以使用 TREAD**：成员们讨论了 **TREAD** 方法在 **LLM** 上的适用性，对预期结果表示不确定，尽管它 *在 LLM 上的效果应该会比在图像领域差得多*。
   - 另一位成员推测，即使是微小的改进也可能值得。
- **归因图扩展到 Attention**：一位成员链接了一个 [YouTube 视频](https://youtu.be/hdi1a9MjwDs?si=taIuYbeF6v-yRSxI&t=628)，讨论将 **归因图 (attribution graphs)** 从 **MLP** 扩展到 **Attention**。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1428511366775308400)** (27 messages🔥): 

> `Libtorch 转换, PersonaLLM Workshop, 英国定价, Prompt 日志策略, GLM 4.6 vs Claude 编程` 


- **Libtorch 转换令人抓狂**：一位成员正致力于将 **SAM video** 转换为 **libtorch**，感到非常吃力。
   - 另一位成员回应说 *他不想招惹视频领域的“恶魔”*。
- **PersonaLLM Workshop 征稿**：在墨西哥城举办的 **NeurIPS PersonaLLM Workshop** 正在征集关于跨 **HCI**、**心理学**、**认知科学**、**文化**和**评估**的人格驱动 LLM 的研究。
   - 提交内容包括：**2 到 4 页的 Demo**（附带 Artifact 链接）、**2 页的非存档摘要**，或通过 [openreview.net](https://openreview.net/group?id=NeurIPS.cc/2025/Workshop_Mexico_City/PersonaNLP&referrer=%5BHomepage%5D(%2F)) 提交**已发表工作的总结**。
- **英国定价之痛**：一位成员抱怨英国的定价，指出 *3650 英镑约合 4901 美元，所以我因为国家不对要多付 900 美元？？* 并附上了一张 [相关图片](https://cdn.discordapp.com/attachments/1149866623109439599/1428779645712465991/image.png?ex=68f466fc&is=68f3157c&hm=ff6b1c2edd5ec622713b2fa3fe197dc4236b0859c27b9580d1c96e9090d06722&)。
- **GLM 4.6 与 Claude 的编程竞争**：随着 **GLM 4.6** 发布并可本地运行，成员们预感 *开源社区不再需要讨好 Sam/Elon/Dario 了*，并引用了 [这个 YouTube 视频](https://www.youtube.com/watch?v=bOfoCocOjfM)。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1428850492577939586)** (2 messages): 

> `新的 Arxiv 论文` 


- **发布了新的 Arxiv 论文**：一位成员发布了一篇可能有趣的 [Arxiv 论文](https://arxiv.org/pdf/2510.14901) 链接。
   - 该成员表示 *他们还不确定该如何评价它*。
- **占位主题**：这是一个为了满足最低要求的占位主题。
   - 随着更多信息的出现，可以添加进一步的细节。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1428850492577939586)** (2 messages): 

> `Arxiv 论文` 


- **讨论 Arxiv 论文**：一位成员分享了一篇 Arxiv 论文的链接 ([https://arxiv.org/pdf/2510.14901](https://arxiv.org/pdf/2510.14901))。
   - 该成员对如何解读该论文表示不确定。
- **讨论另一篇 Arxiv 论文**：另一位成员分享了另一个 Arxiv 论文链接 ([https://arxiv.org/pdf/2510.14901](https://arxiv.org/pdf/2510.14901))。
   - 这第二位成员同样对如何解读该论文表示不确定。

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1428490463823921203)** (29 条消息🔥): 

> `加载错误与 Agent 模式问题、禁止出售积分、Manus 工作坊推广、退款请求、咖啡馆工具` 


- **Manus 正在处理 Agent 模式下的加载错误**：成员们报告了一个**加载错误**，即系统在 Agent 模式下思考时间过长且不开始任务。
   - 部署失败是因为 **OpenAI** 需要编译 *pydantic_core*，因此一名成员计划创建一个不需要 OpenAI 依赖项即可运行的版本。
- **积分销售被禁止**：严禁出售积分，进一步违规可能导致被移出频道。
   - 此公告旨在警告平台内未经授权的积分交易行为。
- **与会者推广伦敦 Manus 工作坊**：一名参加了**伦敦 Manus 工作坊**的成员计划向行业团体推广该活动。
   - 他们寻求联系 Manus 销售人员的帮助，并从另一名成员那里获得了 [Manus Help Center](https://help.manus.im/en/) 的链接。
- **退款请求与 Prompt 编写相关**：一名成员针对一个消耗了几乎所有积分但未能完成设定任务的会话请求退款，并分享了 [会话链接](https://manus.im/share/pjJFAsvmMM7rhlBIZ2e0Jh?replay=1)。
   - 一名成员建议，失败案例不会自动获得退款，因为失败的原因可能很复杂，通常与 **Prompting**（提示词编写）有关。
- **Java 为咖啡爱好者开发新 App**：一名成员分享了一个名为 [Workable Cafes](https://workablecafes.com) 的工具，帮助人们根据 **WiFi 速度**、**舒适度**和**插座**情况寻找咖啡馆。
   - 该 App 已有超过 **100 人**使用，创作者欢迎大家提供反馈。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1428502114857914559)** (24 条消息🔥): 

> `Kimi K2 微调、Kimi 对比 Deepseek、Moonshot 对比 Deepseek` 


- **用户考虑对 Kimi K2 进行微调**：一名用户表示有兴趣对具有 **1B** 参数的 **Kimi K2** 进行微调，但担心 **100k** 样本的 API 成本。
   - 他们建议将数据集减少到 **10k** 样本并进行过滤，尽管这可能仍然很昂贵。
- **相比 Deepseek，用户更青睐 Kimi 的简洁输出**：一名用户询问 **Kimi** 和 **Deepseek** 哪个模型更好，另一名用户表示 *Kimi* 拥有*更多参数、更好的结构化输出以及更简洁*的输出。
   - 第一名用户澄清说，模型的质量取决于参数数量和结构化输出，他们一致认为输出质量至关重要。
- **建议 Deepseek 效仿 Moonshot**：一名用户表示，他们一直在告诉 **Deepseek** 要更像 **Moonshot** 一点。
   - 当被问及是否得到任何回复时，他们没有回答。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1428460293448532020)** (14 条消息🔥): 

> `将 Linux 定为非法、儿童操作系统、AGI 定义、模拟 Ground Truth 数据分布、每周同义反复计数器` 


- **Linux 可能被定为非法！**：成员们开玩笑说 [Linux 被定为非法](https://kernel.org/)，其中一人讽刺地表示担心孩子们。
   - 另一名成员反驳说，孩子们只会编写自己的操作系统并分享它。
- **讨论 AGI 定义**：成员们讨论了 [AGI](https://www.agidefinition.ai/) 的定义，认为它只是一个复杂的问答系统，可以通过足够的训练数据来解决。
   - 一名成员链接了与此讨论相关的 [Dan Hendrycks 的 X 帖子](https://x.com/DanHendrycks/status/1978828377269117007) 和 [Dimitris Papailiopoulos 的 X 帖子](https://x.com/DimitrisPapail/status/1978849863174357052?t=kaj5SsZXgdofKoPV_DWsNA&s=19)。
- **什么时候出“每周同义反复计数器”？！**：一名成员建议创建一个“每周同义反复计数器”，以记录研究人员过度复杂化简单概念的频率。
   - 他们对研究人员能够以多种方式将同一个简单事物复杂化表示沮丧。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1428610292186611743)** (3 条消息): 

> `Qwen3 视觉模型、开源旧模型、保护核心技巧` 


- **Qwen3-VL-8B-Instruct 模型发布**：新的 [Qwen3 视觉模型](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) 已在 HuggingFace 上发布。
- **开源的困境**：一名成员想知道公司是否会**开源他们的旧模型**。
   - 他们怀疑公司宁愿**从头开始训练一个独立的模型**，也不愿发布旧版本，以像 OpenAI 那样保护他们的核心技巧。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1428562254198079511)** (2 messages): 

> `Google Coral NPU, Apache 2 Licensing, RV32 Cores, Mojo Portability Testing` 


- **Google 开源 Coral NPU Verilog**: Google 已在 [Apache 2](https://github.com/google-coral/coralnpu) 许可下开源了 **NPU 模块**的 verilog 代码。
   - 其矩阵核心看起来有点像 AMD 的 NPU，但它们是 **RV32 核心**。
- **Coral NPU 作为 Mojo 可移植性的平台**: 新开源的 **Coral NPU** 作为测试 **Mojo 可移植性**的平台可能非常有趣。
   - 应该可以在客户端硬件上对此进行模拟。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1428502621685153883)** (13 messages🔥): 

> `TUI Frameworks for Mojo, Audio and MIDI 2.0, Jack Bindings, Mojo Origins vs Rust Lifetimes` 


- **Mojo DAW 梦想火花 🔥**: 成员们表达了对类似 Textual 的 **TUI 框架**以及 Mojo 中完整的 **音频/MIDI 2.0** 能力的强烈渴望，以创建一个高性能的 **DAW**。
   - 一位成员建议目前可以先编写针对 **Jack** 等库的绑定，并引用了他们的 [OpenGL 实验](https://link.to.opengl) 作为 **FFI 重度项目**的例子。
- **TUI 框架灵感涌现！**: 一位成员分享了一个名为 [ui-terminal-mojo](https://github.com/rd4com/ui-terminal-mojo) 的 TUI 框架项目链接。
   - 另一位成员提到他们暂停了仿照 Golang 中 **Bubbletea** 等 **ELM 应用**模式开发的 TUI 框架工作，并提供了他们的仓库链接：[banjo](https://github.com/thatstoasty/banjo/blob/main/examples/multi.mojo)。
- **Origins > Lifetimes 🚀**: 一位用户询问了 Mojo 中的 **lifetimes**，并将其与 Rust 的 `<'a> <'static>` 语法进行了比较。
   - 成员们澄清说 Mojo 有一个类似但更符合人体工程学的概念叫做 **Origins**，并指向了 [官方文档](https://docs.modular.com/mojo/manual/values/lifetimes/)。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1428507082268737698)** (1 messages): 

> `MAX Python API Open Source` 


- **Modular 开源 MAX Python API 的剩余部分**: Modular **开源**了 **MAX Python API** 的其余部分。
   - 一篇 [论坛帖子](https://forum.modular.com/t/open-sourcing-all-of-the-max-python-api/2379) 列出了所有新开源的 Python 模块。
- **MAX Python API 可用性**: 完整的 **MAX Python API** 现在作为**开源**软件向公众开放，邀请社区贡献和扩展。
   - 此举使开发者能够在他们的 Python 项目中深度集成 **MAX** 功能，增强了灵活性和创新性。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1428486759662227508)** (5 messages): 

> `IMAGE, NOLOCALS, and GRAPH_ONE_KERNEL Confusion, DEV= Default Device Setting, Speed Regressions Despite Tests, Generic Compilation Tests, Distributed Systems and GPU Memory` 


- **标志位导致配置混淆**: **IMAGE**、**NOLOCALS** 和 **GRAPH_ONE_KERNEL** 等标志位引起了配置混淆，因为很难直观判断什么是真正的编译失败，什么是错误的配置。
   - 有人建议，如果设备/硬件组合不支持，应让这些标志位显式报错。
- **Python 中缺少默认设备设置**: 目前无法在 Python 中设置默认设备，这对于在 Python 脚本中交叉检查不同后端会很方便。
   - 一个可能的实现示例是 `Device.set_default('CL')` 或 `Device.DEFAULT = 'CL'`。
- **速度已测试，但回退依然存在**: 尽管已经有了测试，但还是出现了速度回退（Speed regressions）。
   - 历史数据难以查看，因为 [https://stats.tinygrad.win/](https://stats.tinygrad.win/) 似乎只有过去 25 天的数据，但基准测试（benchmark）正在运行。
- **编译测试需要通用性**: 用户想要为失败的编译编写测试，但目前还没有好的想法。
   - 所有的失败都是模型架构、设备以及某些情况下甚至是由于 **fp16** 导致的特定组合，甚至无法依赖 **ORT** 进行验证，因为 **ORT** 在 FP16 情况下也会产生错误结果。
- **分布式系统寻求 Tiny 粉丝**: 正在涉足分布式系统的人正在寻找 **GPU 内存**或 **CRIU** 领域的开发者进行交流。
   - 他们询问是否有人认识分布式 GPU 领域的人。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1428529015815409796)** (4 messages): 

> `支持 MCP 的 aider fork，aider-ce 的 commit 信息中移除了 GPT-5-nano` 


- **Aider-CE Fork 支持 MCP**：一位用户询问关于支持 **MCP (Multi-Control Protocol)** 的 **aider fork**，另一位用户推荐了 [aider-ce](https://github.com/dwash96/aider-ce/)。
   - 使用特定模型和编辑格式启动 aider 的命令为：`aider --model=openrouter/x-ai/grok-code-fast-1 --edit-format diff`。
- **Aider-CE 在 commit 信息中停用 GPT-5-nano**：一位用户为了获取最新功能和 **GPT-5-code** 支持切换到了 **aider-ce**，但注意到 **aider** 不再提及在 commit 信息中使用 **gpt-5-nano**。
   - 他们询问这一变化是否为有意为之。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1428461227318968452)** (1 messages): 

> `文件名输入、功能请求、Aider 性能` 


- **输入文件名可自动添加文件！**：一位成员注意到在消息后输入文件名（例如 *"see also SomeFile.txt"*）会触发系统询问是否添加文件。
- **Aider 功能需求列表增加**：一位用户提出了一个关于添加文件的 **aider** 功能请求，后续将有更多讨论和细节。
   - 另一位成员提到，他们将提交一个功能请求，允许 **aider** 自动添加本地 git 忽略的文件。


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1428813254041403584)** (2 messages): 

> `MLOps 工作坊、LWP Labs、ML 模型部署` 


- **LWP Labs 推出免费 MLOps 工作坊**：LWP Labs 正在启动一个为期 **3 天的免费 MLOps 工作坊**，教参与者如何将机器学习模型部署到真实生产环境中，内容涵盖 **Docker、CI/CD、MLflow 和 AWS**。
   - 该工作坊承诺提供真实的部署实践，并包含 **五个实战项目**以增强简历竞争力。
- **行业专家将领导 MLOps 培训**：MLOps 工作坊将由一位拥有 **15 年以上**行业经验的讲师领导，旨在为参与者提供热门技能。
   - 该课程强调**实用知识和动手经验**，确保参与者获得 AI 和 Data Engineering 领域雇主青睐的技能。