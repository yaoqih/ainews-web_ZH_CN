---
companies:
- google-deepmind
- nvidia
- alibaba
- hugging-face
date: '2025-05-06T05:44:39.731046Z'
description: '**Gemini 2.5 Pro** 已完成更新，增强了多模态“图像转代码”能力，并在 WebDev Arena 排行榜上占据主导地位，在编程及其他任务中超越了
  **Claude 3.7 Sonnet**。**英伟达 (Nvidia)** 在 Hugging Face 上发布了 **Llama-Nemotron** 模型系列，以高效的推理和逻辑推断能力著称。**阿里巴巴的
  Qwen3** 模型参数规模从 6 亿 (0.6B) 到 2350 亿 (235B) 不等，包括稠密 (dense) 和混合专家 (MoE) 变体。**François
  Chollet** 发布了 **KerasRS**，这是一个兼容 JAX、PyTorch 和 TensorFlow 的新型推荐系统库，并针对 TPU 进行了优化。这些更新突显了模型在编程、推理和语音识别领域的最新进展。'
id: MjAyNS0w
models:
- gemini-2.5-pro
- claude-3.7-sonnet
- llama-nemotron
- qwen3
people:
- demishassabis
- _philschmid
- lmarena_ai
- scaling01
- fchollet
title: Gemini 2.5 Pro 预览版 05-06 (I/O 版) —— 最先进的视觉+编程模型
topics:
- multimodality
- coding
- reasoning
- model-release
- speech-recognition
- recommender-systems
- benchmarking
---

**Gemini is all you need.**

> 2025年5月5日至5月6日的 AI 新闻。我们为您检查了 9 个 subreddits、449 个 Twitter 账号和 29 个 Discord 社区（214 个频道，4980 条消息）。预计节省阅读时间（以 200wpm 计算）：468 分钟。我们的新网站现已上线，提供完整的元数据搜索和美观的 vibe coded 风格展示所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻分类，并在 @smol_ai 上向我们提供反馈！

在 [2.5 Flash 占据帕累托前沿（Pareto Frontier）低端](https://news.smol.ai/issues/25-04-17-ainews-gemini-25-flash-completes-the-total-domination-of-the-pareto-frontier) 3 周后，是时候让 Gemini 重新提升高端水平了。

[Google I/O 将在两周后举行](https://blog.google/feed/google-io-2025-save-the-date/)。有个老生常谈的说法是，在模型的训练数据集中加入更多 Coding 数据，不知为何能帮助它在所有其他方面都有所提升。今天的 Gemini 2.5 Pro 更新（[仅在 6 周前发布](https://news.smol.ai/issues/25-03-25-ainews-gemini-25-pro-4o-native-image-gen)）突显了其多模态 image-to-code 能力，这让人想起了[去年爆火的 Tldraw 时刻](https://www.latent.space/p/tldraw)。


![](https://resend-attachments.s3.amazonaws.com/xydPOHUJqHbS23W)


如今[在 LMArena 排行榜上横扫第一](https://x.com/lmarena_ai/status/1919774743038984449)的含金量已不如以往，但[在 Coding 方面击败 Sonnet 3.7](https://x.com/scaling01/status/1919771796334616759) 仍然值得关注。

在 AIStudio 和 Gemini App 上推出的[更多细节](https://x.com/_philschmid/status/1919770969788313836)也值得称赞。


![](https://resend-attachments.s3.amazonaws.com/HDVLuLnQSsk0uTe)


---

# AI Twitter 摘要

**模型更新与发布**

- **Gemini 2.5 Pro 改进、I/O 版本、Coding 实力以及 WebDev Arena 的统治地位**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1919770265711419826)、[@lmarena_ai](https://twitter.com/lmarena_ai/status/1919774743038984449)、[@scaling01](https://twitter.com/scaling01/status/1919771796334616759)、[@OriolVinyalsML](https://twitter.com/OriolVinyalsML/status/1919770619182215440)、[@_philschmid](https://twitter.com/_philschmid/status/1919770969788313836) 和 [@demishassabis](https://twitter.com/demishassabis/status/1919779362980692364) 强调了更新后的 **Gemini 2.5 Pro 'I/O edition'** 的发布和功能，指出其改进了实际 Coding 能力，特别是在构建交互式 Web App 方面。它在 **WebDev Arena 排行榜上获得了第 1 名**，首次超越了 Claude，并在 Coding、数学、创意写作和长查询方面表现出色。[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1919808911793656151) 报告称 **Gemini-2.5-Pro-preview-05-06** 现在是他们顶尖的 Coding 模型，在困难提示词上击败了 o3 和 Claude 3.7 Sonnet，并建议 **Google 将其命名为 Gemini 3**。[@jack_w_rae](https://twitter.com/jack_w_rae/status/1919779398607085598) 今天也发布了一个更新的 **Gemini 2.5 Pro，它显著提升了实际 Coding 能力**。
- **Nvidia 的 Llama-Nemotron**：[@_akhaliq](https://twitter.com/_akhaliq/status/1919324939934453928) 和 [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1919234521171693844) 分享了 **Nvidia 在 Hugging Face 上发布了 Llama-Nemotron**。这些模型是高效的推理模型，可以在[这里](https://t.co/y2BrBCFrJ0)找到。[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1919236158351147087) 指出该模型系列具有**卓越的推理能力、推理效率以及面向企业用途的开放许可证。**
- **阿里巴巴的 Qwen3 及其他模型发布**：[@reach_vb](https://twitter.com/reach_vb/status/1919422953256587376) 指出 **Nvidia** 开源了 **Parakeet TDT 0.6B**，这是 **Open ASR 排行榜上最好的语音识别模型**，具有商业许可。[@TheTuringPost](https://twitter.com/TheTuringPost/status/1919784802099540446) 总结了过去一周开放 AI 领域大量具有影响力的模型和数据集，并指出**阿里巴巴的 Qwen3** 推出了从 **0.6B 到 235B** 的稠密（dense）和 MoE 模型。
- **Keras 发布**：[@fchollet](https://twitter.com/fchollet/status/1919477586599805118) 宣布发布 **KerasRS**，这是一个用于构建推荐系统的新库，具有易于使用的构建块，并兼容 JAX、PyTorch 和 TF，且针对 TPU 进行了优化。

**排行榜与基准测试结果**

- **Gemini 2.5 Pro 在 LMArena 登顶**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1919770268299321608) 和 [@scaling01](https://twitter.com/scaling01/status/1919771796334616759) 强调 **Gemini 2.5 Pro** 在 WebDev Arena 排行榜上处于领先地位，并在 LMArena 的 Coding 类别中排名第一。scaling01 指出，此前 o3 和 Claude 都未能实现这一目标。
- **Qwen3 在 LiveCodeBench 的表现及进入 Arena 前十**：[@huybery](https://twitter.com/huybery/status/1919418019517776024) 庆祝了 **Qwen3-235B-A22B 在 LiveCodeBench 上的出色表现**，将其定位为竞赛级代码生成的顶级开源模型，性能媲美 o4-mini。此外，[@lmarena_ai](https://twitter.com/lmarena_ai/status/1919448953042706759) 报告称，社区投票已将最新的开源 **Qwen3 推入 Arena 前十名**，其中 Math 排名第一，Coding 排名第四。
- [@TheAITimeline](https://twitter.com/TheAITimeline/status/1919155696655843474) 整理了本周值得关注的研究论文列表，包括：DeepSeek-Prover-V2、《排行榜错觉》（The Leaderboard Illusion）、Phi-4-reasoning 技术报告。

**AI 与机器学习研究**

- **代码生成的语法约束与类型感知解码**：[@ndea](https://twitter.com/ndea/status/1919788307090964873) 重点介绍了一篇论文，该论文通过使用 **prefix automata（前缀自动机）添加类型感知解码**，以改进多个 LLM 的代码生成、修复和翻译，并在 TypeScript 和 HumanEval 上进行了测试。
- **当前指标的伦理担忧与评估挑战**：[@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1919218998211641393) 在观察到现实世界中后果主义和义务论的失败后，围绕成为一名 **virtue ethicist（美德伦理学家）** 展开了对话。[@TheTuringPost](https://twitter.com/TheTuringPost/status/1919563905799696703) 报道了用于评估 AI 模型指标的问题，指出了 sycophantic drift（阿谀奉承漂移，即模型为讨好而优化）以及 Chatbot Arena 中 leaderboard illusion（排行榜错觉）的问题，其中私有变体和偏斜的数据访问可能会导致结果偏差。
- **利用 Reinforcement Learning 实现高效数学推理**：[@denisyarats](https://twitter.com/denisyarats/status/1919601674676588894) 发表了一篇关于使用 Reinforcement Learning 提高 LLM 数学推理能力的博客文章。
- **Computer Use 在 Agentic Workflows 中的作用**：[@AymericRoucher](https://twitter.com/AymericRoucher/status/1919783847597670780) 报道了在 **smolagents 中推出 Computer Use 功能**，强调了 vision models（尤其是 Qwen-VL 模型）通过定位和点击截图上的元素来驱动复杂 Agentic Workflows 的能力。

**AI 工具与应用**

- **Cline AI 工具的新功能**：[@cline](https://twitter.com/cline/status/1919567686079807680) 及其团队发布了关于 **Cline** 的专业技巧和更新推文，例如使用 **/newrule 命令捕获项目标准**、管理 .clinerules，以及在点击“Act”后调整计划的能力。
- **构建推荐系统的 Keras 框架**：[@fchollet](https://twitter.com/fchollet/status/1919477586599805118) 指出 Keras 团队发布了一个用于构建推荐系统的新库：KerasRS。
- **用于 RL 优化的 DSPy GRPO**：[@lateinteraction](https://twitter.com/lateinteraction/status/1919428454761553994) 宣布发布 **dspy.GRPO，这是一个用于 DSPy 程序的在线 RL 优化器**，允许使用 RL 优化现有的 DSPy 代码。
- **AI 驱动的代码生成与编辑**：[@_philschmid](https://twitter.com/_philschmid/status/1919774801767317799) 讨论了 Gemini 2.5 Pro 生成 zero-shot SPAs、移动游戏和 UI 截图的能力。

**行业与商业动态**

- **OpenAI 的架构与上市**：[@OpenAI](https://twitter.com/OpenAI/status/1919453166979957115) 分享了来自 Bret Taylor 和 Sam Altman 关于 OpenAI 架构的信息，重申了其使命优先的方针。[@LiorOnAI](https://twitter.com/LiorOnAI/status/1919581771240505785) 表示 **OpenAI 放弃了向 for-profit（营利性实体）的转型。**
- **Weights & Biases 被 CoreWeave 收购**：[@weights_biases](https://twitter.com/weights_biases/status/1919378138129183138) 宣布其被 CoreWeave 收购，标志着共同专注于创新和规模化新篇章的开始。

**社会**

- **关于 AI 对社会和科学影响的反思**：[@gneubig](https://twitter.com/gneubig/status/1919444422321746052) 表达了对构建能够有效判断研究质量的 AI 系统的兴趣，而 [@random_walker](https://twitter.com/random_walker/status/1919359709062033850) 讨论了 hallucinations（幻觉）、deskilling（技能退化）的风险，以及在工作场所集成 AI 时需要采取结构化方法来解决这些问题。

**幽默**

- [@TheGregYang](https://twitter.com/TheGregYang/status/1919186673382113298) 发帖称，他们调整了其中一个神经网络，以推荐更相关的帖子，现在你应该会看到更少的垃圾内容（slop）。
- [@francoisfleuret](https://twitter.com/francoisfleuret/status/1919278792108867931) 开玩笑说：**如果 AGI 可以自我改进，为什么回形针最大化器（paperclip maximizer）真的会去最大化回形针数量，而不是直接黑掉 `paperclip_production_rate()` 让它返回 `float("inf")`，从而享受永恒的幸福？**
- [@scaling01](https://twitter.com/scaling01/status/1919466275346039069) 拿 [@Yuchenj_UW](https://twitter.com/Yuchenj_UW) 的新 Elon 名字开玩笑。
- [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1919314914377605607) 提到他妈妈把 ChatGPT Pro 称为 “ChatGPT 豪华版（Fancy Edition）”。
- [@andersonbcdefg](https://twitter.com/andersonbcdefg/status/1919563451057254600) 开玩笑说：**OpenAI 以 94 亿美元收购了 Skechers。当通过电子邮件询问评论时，CEO Sam Altman 回复道，“我觉得魔术贴（velcro）挺酷的”**。
- [@TheGregYang](https://twitter.com/TheGregYang/status/1919842967818309699) 说道：**哟，gorktard**。
- [@scaling01](https://twitter.com/scaling01/status/1919470198773395814) 开玩笑说：**排名前 5 的其他 4 位研究员：GPT-5**。

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

### 1. Qwen 模型性能与 VRAM 占用讨论

- [**Qwen 14B is better than me...**](https://www.reddit.com/r/LocalLLaMA/comments/1kft5yu/qwen_14b_is_better_than_me/) ([Score: 584, Comments: 284](https://www.reddit.com/r/LocalLLaMA/comments/1kft5yu/qwen_14b_is_better_than_me/)): **该帖子讨论了用户的一种感知，即开源 LLM Qwen 14B（一个** `9GB` **的模型文件）在语言表达、编程、数学、社交互动、工具使用和多语言能力方面超过了他们自身的能力，并将其性能和紧凑性与人类极限进行了对比。值得注意的是，用户观察到 Qwen 14B 比更小的模型（例如 8B 参数模型）更不容易出错，并对这一参数规模带来的质变提出了疑问。有关该模型的背景信息，请参阅 [Qwen's GitHub](https://github.com/QwenLM/Qwen-14B)。** 评论者补充说，进一步的蒸馏（distillation）可能会将模型大小降低到 `1GB` 以下，突显了持续的效率进步。文中引用了莫拉维克悖论（Moravec's Paradox）来论证 LLM 擅长的任务对机器来说在认知上相对容易，但对人类来说却很难，而协调和感知仍然是人类的强项（[莫拉维克悖论解释](https://en.wikipedia.org/wiki/Moravec%27s_paradox)）。
    - 一位评论者指出，虽然 Qwen 14B 目前是一个 9GB 的模型，但通过蒸馏有潜力显著减小体积，可能降至 1GB 以下。这将对在边缘设备上运行高级 LLM 的部署能力和资源需求产生重大影响。
    - 另一个技术见解提到了莫拉维克悖论，指出虽然 LLM 在语言等抽象任务上表现出色，但人类在感觉运动协调方面（例如起床）轻松超越机器——这是即使是简单的动物也能常规完成的壮举。
    - 关于代码生成，一位用户表达了怀疑，称没有任何 LLM（包括 Qwen 14B）被证明“擅长”编程。这表明尽管模型能力有所进步，但编程领域的人类级专业知识对于大语言模型来说仍然是一个具有挑战性的基准。
- [**VRAM requirements for all Qwen3 models (0.6B–32B) – what fits on your GPU?**](https://i.redd.it/l8bxcpzj23ze1.png) ([Score: 143, Comments: 46](https://www.reddit.com/r/LocalLLaMA/comments/1kfvba4/vram_requirements_for_all_qwen3_models_06b32b/)): **该图片提供了一份在 Unsloth 量化下 Qwen3 模型（0.6B–32B 参数）VRAM 需求的对比表，重点在于平衡推理性能和显存占用。该表详细列出了 VRAM 使用情况（涵盖不同上下文大小和 GPU 显存分配），并报告了快速提示测试中的每秒 Token 数 (TPS) 基准测试，特别是在两种 GPU 类型上：RTX3080Ti Laptop 和 RTX3090 (eGPU)。关键见解包括：通过仔细的量化，Qwen3-4B 甚至更大的模型可以在消费级 GPU 上合理运行，且报告的 TPS 旨在作为非正式的实用指标，而非严格的基准测试。** 评论者对 VRAM 估算的准确性和卸载（offloading）策略提出了质疑，建议使用其他量化库（GPTQ, Bitsandbytes, AWQ）可能会产生更好的结果。一位用户报告在 Apple M4 Macbook Air 上成功运行了 32B 模型，尽管存在散热限制和较低的 Token 速度，这凸显了硬件体验的多样性。
    - 讨论围绕将大型模型（如 Qwen3 32B）适配到消费级 GPU 的量化策略展开。用户强调 Q4 量化通常比 Q3_K_XL 更具显存效率，能平衡性能和 VRAM 需求，并建议使用 GPTQ, Bitsandbytes 或 AWQ 等量化库以实现最佳 GPU 利用率，而不是仅仅依赖 GGUF 格式。
    - 一位用户报告称，Qwen3 32B 模型技术上可以在具有 32GB 统一内存（unified memory）的 Apple M4 Macbook Air 上运行，但伴随着显著的热负荷和非常低的 Token 生成速度，这强调了即使模型能成功加载到性能较弱的硬件上，实际吞吐量和硬件限制依然存在。
    - 经验丰富的用户对 VRAM 需求图表持怀疑态度，指出由于卸载或量化优化，较小的模型已经在比建议更少的 VRAM 配置上运行过，这表明官方要求可能偏向保守，且并不总是考虑到先进的部署技术。

### 2. 全新开源 SOTA 音乐生成模型 ACE-Step

- [**全新 SOTA 音乐生成模型**](https://v.redd.it/gf0uynfhz6ze1) ([Score: 468, Comments: 93](https://www.reddit.com/r/LocalLLaMA/comments/1kg9jkq/new_sota_music_generation_model/)): **ACE-Step 是一款新开源的音乐生成模型，拥有 3.5B 参数，支持多语言输出（19 种语言）、乐器风格和声乐技巧。该项目包含完整的训练和 LoRA 微调代码，基准测试报告显示其推理速度极快（在 RTX 4090 上每 1 分钟音频仅需 1.74 秒）。该模型采用 Apache 许可证发布，旨在像 Stable Diffusion 影响图像生成领域一样，推动高质量音乐生成的普及；技术细节请参阅 [GitHub](https://github.com/ace-step/ACE-Step) 和 [项目主页](https://ace-step.github.io/)。** 热门评论强调了 StepFun 在音频-文本处理方面的强大实力（引用了 Step-Audio-Chat），开源贡献者追赶或超越商业模型的速度，以及对 ACE-Step 演示音频质量的积极主观体验。
    - StepFun 的 Apache 授权模型支持 LoRA 微调，使其在社区驱动的改进方面具有高度灵活性，类似于 Stable Diffusion 彻底改变开源图像生成的方式。人声仍被视为弱点，但这种基础性的开源方法被视为重大进步（特别是与封闭模型相比）。
    - 一位用户报告了 StepFun 模型的硬件基准测试：在 NVIDIA RTX 4090 上，生成 1 分钟音频耗时 1.74 秒（27 步，34.48 倍实时速度）和 3.84 秒（60 步，15.63 倍实时速度）；在 M2 Max MacBook 上，分别为 26.43 秒和 58.25 秒。这突显了高端 GPU 上的显著加速，但在消费级硬件上表现较慢。
    - 关于什么是 SOTA（State-of-the-art）音乐生成存在技术争论：虽然 StepFun 在遵循文本指令方面似乎比 Udio 更好，但一些用户认为其实际音频输出质量稍逊一筹，这引发了关于 SOTA 应该优先考虑文本忠实度还是音质忠实度的疑问。
- [**为什么我们又要吐槽 Ollama 了？**](https://www.reddit.com/r/LocalLLaMA/comments/1kg20mu/so_why_are_we_shing_on_ollama_again/) ([Score: 173, Comments: 307](https://www.reddit.com/r/LocalLLaMA/comments/1kg20mu/so_why_are_we_shing_on_ollama_again/)): **该帖子讨论了对 Ollama 的技术批评。Ollama 是一款以安装便捷著称的本地 LLM 运行器（`pacman -S ollama ollama-cuda`），内置 Open WebUI 配置，支持动态模型切换，并兼容专有模型和 GGUF 模型。批评集中在：(1) 阻碍与其他推理后端互操作的专有存储格式（仅能通过符号链接或变通方法访问），(2) 缺乏对 llama.cpp 等关键项目的上游贡献（例如将多模态或高级功能保留在内部），(3) 默认配置值并非最优，(4) 在没有官方 UI 的情况下后台进程的行为，以及 (5) 分发命名模糊且有时质量较低的模型（例如 Deepseek R1 事件，使用次优量化）。参考 [Ollama 的 GitHub](https://github.com/jmorganca/ollama) 和 [llama.cpp](https://github.com/ggerganov/llama.cpp)。** 评论中的辩论集中在 Ollama 的便利性是否能为其准“围墙花园”方式正名，一些用户对令人困惑的模型发布和感知到的生态系统锁定感到沮丧，而另一些人则看重快速、便捷的本地推理价值。
    - 几位用户批评了 Ollama 对模型命名的处理，特别是在 Deepseek R1 发布期间，声称该公司将量化或蒸馏的 <10B 模型作为完整版本进行营销，这设定了不切实际的预期，并使用户对本地模型的性能感到困惑。
    - Ollama 以专有文件格式存储模型，使其难以与其他推理后端（如使用 GGUF 的后端）互换，实际上将用户锁定在其生态系统中。用户还注意到 Ollama 没有向 llama.cpp 等父项目贡献重大增强功能（如多模态支持、iSWA），而是等待上游实现后再集成功能。
    - 技术投诉包括次优的默认设置，例如默认使用较低质量的 Q4_0 量化而非更先进的 *K/*K_M 变体，VRAM/上下文分配不足导致模型性能下降，缺乏安全认证，API 和集群控制令人沮丧（例如无法在容器中预先指定模型加载），以及缺少用户界面。一些人认为 LLM 的热交换是一个技术加分项，但总体认为由于这些限制，该工具的易用性被高估了。

## 其他 AI Subreddit 摘要

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. Gemini 2.5 Pro 模型更新与基准测试

- [**Gemini 2.5 Pro 更新：更强大的代码性能 [Google Blog]**](https://developers.googleblog.com/en/gemini-2-5-pro-io-improved-coding-performance) ([Score: 188, Comments: 34](https://www.reddit.com/r/singularity/comments/1kg72t3/gemini_25_pro_update_even_better_coding/)): **Google 最新的 [Gemini 2.5 Pro Preview](https://developers.googleblog.com/en/gemini-2-5-pro-io-improved-coding-performance) (05-06) 提升了代码性能，特别是在前端/UI 工作方面，其在 WebDev Arena 排行榜上排名第一便证明了这一点。关键改进包括更稳健的代码转换/编辑、增强的 function calling（具有更低的错误率和更高的触发率），以及更好的 agentic workflow 支持；视频理解也达到了业界领先水平（84.8% VideoMME benchmark）。模型版本别名（例如，03-25 现在指向 05-06）意味着用户端无需为升级采取任何行动，确保了无缝的 API 部署。** 评论者对就地更新模型版本的做法表示担忧——这可能会破坏可复现性和版本控制的最佳实践——同时也指出改进主要局限于代码领域，而科学和数学能力仍落后于竞争对手（例如在某些领域的 GPT-4o）。
    - Gemini 2.5 Pro 的更新流程因最新模型版本的别名机制而受到批评；用户提到 '03-25' 端点现在自动引用 '05-06'，这引发了关于正确版本控制和可复现性的技术担忧（例如，在相同的版本标签下，模型输出可能会随时间而改变）。
    - 虽然 Gemini 2.5 Pro 展示了针对代码的改进，但用户注意到在其他领域（特别是科学和数学）进展不足，据报道甚至 Google 自己的 benchmark 也显示旧模型在这些领域的表现优于它。有人提到 Logan（可能是 Google 员工）确认这是一个针对代码的更新。
    - 尽管在代码 benchmark 方面获得了认可，但批评意见强调了持续存在的失败案例：Gemini 2.5 Pro 被指出会生成调用不存在函数的代码，或者在基础编程任务（如过滤包含字符串的行）中返回错误结果，这揭示了代码生成可靠性方面持续存在的挑战。
- [**更新后的 Gemini 2.5 Pro 现已在 WebDev Arena 排行榜排名第一**](https://i.redd.it/bph5w0ffi6ze1.png) ([Score: 201, Comments: 39](https://www.reddit.com/r/singularity/comments/1kg75xx/the_updated_gemini_25_pro_now_ranks_1_on_the/)): **图片展示了 WebDev Arena 排行榜，新更新的 Gemini 2.5 Pro 模型以 1420 的最高竞技场分数登顶。该排行榜直观地比较了各种模型在 WebDev Arena benchmark 中的代码性能，一个突出的指标指明 Gemini 2.5 Pro 比之前的版本增加了 +147 Elo，标志着在代码和 Web 开发任务能力上的显著飞跃。这次更新使 Gemini 2.5 Pro 领先于其他主流模型，凸显了基于 LLM 的代码助手领域的快速进步。** 评论者对代码性能提升的幅度（+147 Elo）印象深刻，一些人还指出改进在创意写作能力方面也很明显——更新后的模型重复更少，对 prompt 的理解更好。
    - 多条评论强调了 Gemini 2.5 Pro 的显著性能飞跃，因为它夺得了 WebDev Arena 排行榜的第一名。这表明它比之前的版本以及潜在的竞争模型有了显著改进，特别是在与 Web 开发 benchmark 相关的任务中。
    - 一位用户注意到创意写作任务中的实际改进，称该模型重复性更低、更自然，并且对 prompt 有更好的理解。这表明更新版本在自然语言生成和上下文感知方面取得了显著进展。
    - 人们期待通过 “simple-bench” 等第三方 benchmark 进行进一步评估，这表明虽然排行榜结果令人振奋，但社区更看重通过全面、独立的测试来验证所宣称的改进。

- [**Gemini 2.5 Pro 新版本：gemini-2.5-pro-preview-05-06**](https://i.redd.it/bmffiwssv5ze1.png) ([Score: 333, Comments: 68](https://www.reddit.com/r/singularity/comments/1kg4pdo/new_version_of_gemini_25_pro/)): **该图片是一个官方风格的横幅，宣布发布 'gemini-2.5-pro-preview-05-06'，被 Google/DeepMind 描述为“我们最先进的推理模型”。标题和极简设计强调了这是一次技术升级，可能涉及 Gemini 2.5 Pro 语言模型在复杂推理和问题解决能力方面的改进。版本号暗示了相对于之前内部/原型版本的迭代增强，顶层评论中提到的 'Matts_fixes_APPROVED' 可能意味着对近期 Bug 修复或架构调整的特别关注。** 技术评论者表达了期待和好奇，询问实际使用体验，并建议采用这一特定变体，因为其包含了已批准的修复，暗示了内部对 Gemini 发布候选版本（release candidates）的可靠性或性能存在讨论。
    - 关于模型版本命名的讨论：具体而言，引用的版本是 "gemini-2.5-pro-preview-05-06-FOR_RELEASE_use_this_one!!!_Matts_fixes_APPROVED(2)(2)"，表明存在一些内部或补丁相关的更新，并标注为 'Matts fixes' 已批准。这种命名级别暗示在广泛部署前经历了阶段性或内部 QA 流程。
    - 提到了技术部署：最初，用户注意到该模型仅在 Vertex AI 上可用，随后一名用户确认其在 AI Studio 中也已上线，突显了 Google AI 平台之间交错的推出节奏以及更广泛模型可用性可能存在的延迟。
    - 针对 Google 的标签习惯提出了一个元观点：一名用户质疑 Google 是否曾将模型移出 'experimental'（实验性）或 'preview'（预览）阶段，暗示了其 AI 模型发布中长期处于预览状态的模式，这可能会影响生产环境的采用时间表。
- [**今日 Gemini 2.5 Pro 更新**](https://v.redd.it/2exjpauph6ze1) ([Score: 300, Comments: 20](https://www.reddit.com/r/singularity/comments/1kg71ul/todays_gemini_25_pro_update/)): **Google 的 Gemini 2.5 Pro 更新展示了精确的代码生成能力，通过实现 1988 年原始论文中的标准 IFS 参数来构建经典的 Barnsley fern 分形，正如 Google 官方博客文章所述（[来源](https://blog.google/products/gemini/gemini-2-5-pro-updates/)）。技术评论者观察到所选算法广为人知且简单，指出成功的、无引导的代码生成说明了该模型在识别和正确应用计算机图形学历史中的规范解决方案方面的能力。** 一个值得注意的辩论涉及使用经典算法作为基准测试的意义；一些人认为这突显了预期的 SOTA LLM 性能，而另一些人则指出 Gemini 2.5 Pro 在近期使用中的整体质量超过了许多 GPT 变体。
    - 一个显著的技术讨论点涉及 Google 用来展示 Gemini 2.5 Pro 的示例——一个著名的分形生成算法。多位用户强调该算法既 *"极其古老且相对容易实现"*，暗示此类任务对于任何 SOTA LLM 来说现在都应该是轻而易举的。
    - 一些用户直接将 Gemini 2.5 Pro 的能力与其他领先的 LLM 进行对比，指出其近期更新带来了显著的性能提升，甚至建议它在实际用例中 *"比大多数 GPT 模型表现更好"*。
    - 一名用户提出了关于 Gemini 2.5 Pro 推理能力的关键技术问题，反映了社区对其次代架构和训练是否已实质性超越早期模型推理局限性的持续关注。

### 2. OpenAI 收购 Windsurf 相关报道

- [**OpenAI 达成协议以 30 亿美元收购初创公司 Windsurf**](https://www.bloomberg.com/news/articles/2025-05-06/openai-reaches-agreement-to-buy-startup-windsurf-for-3-billion?embedded-checkout=true) ([Score: 187, Comments: 50](https://www.reddit.com/r/ChatGPTCoding/comments/1kful8w/openai_reaches_agreement_to_buy_startup_windsurf/)): **据报道，OpenAI 已同意以 30 亿美元收购 AI coding agent 初创公司 Windsurf。Windsurf 以其快速发布的、可集成多个 AI 模型的 open-source coding agents 而闻名，而目前的生态系统特征是 open-source coding agents（如 Aider, Kilo Code, Cline）与各种模型（发布频繁且本地/廉价模型日益增多）之间相互独立。人们担心 OpenAI 的收购可能会使 Windsurf 未来的产品偏向 OpenAI 模型，而非 Gemini 或 Claude 等替代方案，从而可能降低生态系统的多样性和开放性。** 技术层面的担忧强调了垂直整合降低用户选择和创新速度的风险，特别是如果以前与模型无关的 agents 被锁定在单一供应商中。鉴于当前的市场动态，30 亿美元的估值也受到了质疑，被认为可能过高。
    - 存在对垂直整合和潜在平台锁定的担忧，如果 Windsurf（此前是一个支持广泛模型的 open-source AI coding agent）在收购后开始偏向 OpenAI 模型。这可能会破坏当前的生态系统，在当前生态中，open-source coding agents（如 Cline, Roo, Aider, Kilo Code）能够快速增加功能并支持多种模型，促进整个领域的创新和公平竞争。
    - 一位评论者认为，收购成本的合理性在于 Windsurf 用户遥测数据的价值，OpenAI 可以利用这些数据来增强其 coding 模型。这表明 OpenAI 目前在 AI coding 工具领域的地位可能比预期的要弱，此次收购是加强其数据集和专有模型训练的战略举措。
    - 关于市场策略的讨论指出，“仅仅贴上 OpenAI 的名字”到 Windsurf 上，无论其实际技术是否优越，都能推动采用。一些人注意到，OpenAI 在程序员中的主导地位（其中许多人只使用 ChatGPT）赋予了它压倒性的网络和分发优势，即使像 Cursor 这样的替代方案存在且在技术上可能更好。
- [**OpenAI 达成协议以 30 亿美元收购初创公司 Windsurf**](https://www.bloomberg.com/news/articles/2025-05-06/openai-reaches-agreement-to-buy-startup-windsurf-for-3-billion) ([Score: 513, Comments: 94](https://www.reddit.com/r/OpenAI/comments/1kftk0m/openai_reaches_agreement_to_buy_startup_windsurf/)): **OpenAI 已同意以约 30 亿美元收购 AI 初创公司 Windsurf，旨在整合 Windsurf 的技术和人才，以加速其产品开发并扩展核心 AI 能力。此次收购可能旨在加强 OpenAI 在基于订阅的 AI 工具领域的地位，并有望增强基础设施和模型创新。[Bloomberg 报道](https://www.bloomberg.com/news/articles/2025-05-06/openai-reaches-agreement-to-buy-startup-windsurf-for-3-billion) 提供了更多细节。** 技术评论者争论 Windsurf 的技术是否值 30 亿美元的价格，认为可以用更少的成本重建，并质疑集成到 OpenAI 的付费层级是否会成为一个显著的差异化因素——特别是提到了 Cursor 等竞争对手。
    - 一个核心技术讨论集中在垂直整合的风险上：收购 Windsurf 可能会导致其平台优先考虑 OpenAI 自己的模型（如 GPT-4/5），从而减少对竞争模型（如 Google Gemini, Anthropic Claude）的支持或访问。这可能会对开发者的选择和生态系统的多样性产生负面影响，而目前的生态系统由于 agent 与模型的分离以及持续的 open-source 创新而蓬勃发展。
    - 一位评论者强调，许多 AI coding agents（如 Cline, Roo, Aider, Kilo Code）每周都会发布，大多数是 open-source 的，并且集成了多个模型。有人担心收购可能会阻碍功能的快速开发或与非 OpenAI 模型的兼容性，因为公司所有权可能会优先考虑专有集成而非包容性。
    - 针对收购成本存在技术性反对意见，有人建议 Windsurf 的核心功能可以以远低于 30 亿美元的成本进行复制，这引发了对证明如此巨额支出合理性的效率和技术差异化的质疑。

- [**OpenAI 达成协议以 30 亿美元收购初创公司 Windsurf**](https://www.bloomberg.com/news/articles/2025-05-06/openai-reaches-agreement-to-buy-startup-windsurf-for-3-billion?embedded-checkout=true) ([Score: 632, Comments: 120](https://www.reddit.com/r/singularity/comments/1kftq2g/openai_reaches_agreement_to_buy_startup_windsurf/)): **据报道，OpenAI 已达成协议，以 30 亿美元收购初创公司 Windsurf。帖子中未提供有关 Windsurf 技术或产品的详细信息，但高昂的估值表明 Windsurf 拥有 OpenAI 认为难以或耗时复制的独特知识产权、技术基础设施或人才。相关评论还指出，另一家科技初创公司 Cursor 在融资 9 亿美元后，目前的估值为 90 亿美元，这凸显了当前 AI 领域的高估值。** 评论者质疑为什么 OpenAI 会收购 Windsurf 而不是在内部开发相关技术，这暗示 Windsurf 可能拥有 OpenAI 难以轻易复制的重大技术或组织优势。
    - 几位用户讨论了 Cursor (90 亿美元) 的高估值和 Windsurf (30 亿美元) 的收购，其中一位指出 Cursor 最近完成了 9 亿美元的融资，标志着对 AI 驱动的开发者工具和 IDE 的密集投资和浓厚兴趣。
    - 一位用户推测，OpenAI 收购 Windsurf 可能与其未来软件工程师 Agent 的战略有关，表明 Windsurf 的技术可能在驱动自主或半自主编码系统方面发挥关键作用，超越 OpenAI 内部开发的能力。
    - 另一条评论建议，这次收购标志着 AI 驱动的 IDE 和开发者平台之间竞争的显著升级，暗示由该领域的大规模投资和并购 (M&A) 驱动的“IDE 大战”即将到来。

### 3. 最新的 AI 图像和视频生成模型发布

- [**LTXV 13B 发布 - 兼顾高质量与极速**](https://v.redd.it/3jt4f5r0o5ze1) ([Score: 1026, Comments: 180](https://www.reddit.com/r/StableDiffusion/comments/1kg48dn/ltxv_13b_released_the_best_of_both_worlds_high/)): **Lightricks 发布了 LTXV 13B，这是一个开源的、拥有 13B 参数的视频生成模型，具有多尺度渲染功能：它最初生成低分辨率帧并进行迭代优化，从而实现了极高的渲染效率和改进的物理真实感。该模型声称比同类模型快约 30 倍，支持高级控制（关键帧、摄像机/场景/角色运动、多镜头序列），并提供针对本地 GPU 使用优化的标准版和量化 (FP8) 版本。授予完整的商业权利（部分大型企业除外），生态系统包括一个易于使用的 LoRA 微调训练器 ([GitHub](https://github.com/Lightricks/LTX-Video-Trainer))、ComfyUI 工作流和 [Diffusers 流水线](https://github.com/Lightricks/LTX-Video)；模型和 FP8 变体可在 Hugging Face 上获取。** 评论者强调了下载文件的大小（约 26GB），但对 FP8 量化版本的可用性表示赞赏，并期待将其与 Wan FLF 和 SkyR I2V 等其他近期视频模型进行比较。仓库文档中指出了量化模型的质量/速度权衡。
    - 关于 LTXV 13B 的 8 位浮点 (FP8) 工作流存在一些担忧：用户报告在放大后细节明显降低，且存在一致的曝光偏移（图像变亮且对比度降低），这可能会限制其在高保真或色彩关键型应用中的实用性。
    - 一位用户询问硬件兼容性，特别是具有 4GB VRAM 和 32GB RAM 的系统是否可以运行该模型，这暗示了 LTXV 13B 由于其庞大的模型尺寸（标准版为 26GB）可能面临资源限制的挑战。
- [**Insert Anything – 使用强大的 AI 编辑工具将任何物体无缝插入图像**](https://v.redd.it/rc43edcvj6ze1) ([Score: 152, Comments: 32](https://www.reddit.com/r/StableDiffusion/comments/1kg7gv3/insert_anything_seamlessly_insert_any_object_into/)): **“Insert Anything”是一个 AI 驱动的图像编辑框架，允许将任何参考物体无缝插入到目标图像中。该工具声称可以保留照片级的写实细节（颜色、纹理），并支持虚拟试穿、广告和迷因 (meme) 创作等应用。代码和工作流通过 [Hugging Face Space](https://huggingface.co/spaces/WensongSong/Insert-Anything) 和 [GitHub](https://github.com/song-wensong/insert-anything) 提供，并集成了 ComfyUI 工作流。** 评论者指出，据报道该工具需要 `~26GB VRAM`，这意味着对硬件有较高要求，降低了中端 GPU（如 RTX 3060）用户的可访问性。至少有一位用户描述其功能运行良好。

- 用户正在讨论在本地运行该工具所需的显著 VRAM 要求（26GB），对 RTX 3090 (24GB VRAM) 或 RTX 3060 (12GB VRAM) 等显卡是否能处理该工作负载表示担忧，这暗示了较大的模型尺寸或资源密集型操作。
- 一位用户询问底层模型或架构，质疑该工具是否基于 Flux、SDXL 或其他框架，这表明用户希望了解更多关于图像编辑方法的实现层级细节。
- [**ZenCtrl 更新 - 源代码发布和主体驱动生成一致性提升**](https://i.redd.it/5sepm9w924ze1.png) ([Score: 127, Comments: 30](https://www.reddit.com/r/StableDiffusion/comments/1kfye3g/zenctrl_update_source_code_release_and/)): **该图像是一个拼贴图，展示了 ZenCtrl 在不同视角和场景下主体一致性方面的最新改进。此更新解决了先前模型在角度或场景变化时主体身份会丢失的弱点，这是在额外的训练和模型优化之后实现的。此次发布包含了 ZenCtrl 的开源，现已在 GitHub 上提供，同时还提供了 Hugging Face 演示和 Discord 的链接，强调了其用于可控 AI 图像生成的开放、模块化方法。** 评论者询问 ZenCtrl 的架构，特别是它是否类似于用于 SDXL/Flux 的 ControlNet，或者是否包含自己的生成主干网络，以及它与 ComfyUI 集成的可能性。技术讨论集中在实现细节和工作流兼容性上，表明了对现有流水线中模块化集成和可用性的浓厚兴趣。
    - 一位用户询问 ZenCtrl 的运行方式是否类似于 SDXL/Flux 的 ControlNet，或者该仓库是否也包含一个独立的图像模型。该问题旨在澄清 ZenCtrl 是通过主体调节来增强现有的扩散流水线，还是其本身提供了一个完整的生成主干模型。
    - 另一位评论者询问在 ComfyUI 中的可用性，表明对集成细节和可组合扩散工作流兼容性的兴趣。他们正在寻求有关 ZenCtrl 如何作为节点或模块合并到 ComfyUI 流水线中的技术文档或社区确认。
    - 有人提出了关于项目许可证从 Apache 更改的问题，这涉及到对开源使用、再分发和商业改编的影响。这对于可能集成或扩展 ZenCtrl 的下游开发者来说至关重要。
- [**ComfyUI API 节点和新品牌推广**](https://v.redd.it/874ljlhjh5ze1) ([Score: 133, Comments: 67](https://www.reddit.com/r/StableDiffusion/comments/1kg2oqy/comfyui_api_nodes_and_new_branding/)): **ComfyUI 宣布为一系列最先进的（SOTA）第三方模型提供原生 API 节点集成，包括 Bfl FLUX、Kling、Luma、Minimax、PixVerse、Recraft、Stability AI、Google Veo、Ideogram 和 Pika。访问这些 API 是可选的（opt-in），需要预付费计费，仅收取底层 API 成本和一些交易费用，而核心 ComfyUI 仍保持免费和开源。更多技术细节和实现背景请参阅其官方 [博客文章](https://blog.comfy.org/p/comfyui-native-api-nodes)。** 技术用户对向 SaaS/API 依赖的方向发展表示保留意见，但也认识到项目可持续性的必要性；一些人对持续的开源访问表示赞赏，同时指出了对外部服务集成的哲学担忧。
    - 一些用户担心 ComfyUI 朝着 API 节点和新品牌方向的发展最终可能导致闭源 API，这可能会影响透明度和开源社区的贡献。关于开源项目的可持续性与通过 SaaS 或受限 API 等方法进行盈利的必要性之间存在着潜在的辩论。
    - 为寻求有关新引入的原生 API 节点的更深入信息的技术读者提供了一个指向 ComfyUI 博客文章 (https://blog.comfy.org/p/comfyui-native-api-nodes) 的直接链接，这可能表明 ComfyUI 生态系统中发生了重大的架构或可扩展性变化。

---

# AI Discord 摘要

> 由 Gemini 2.5 Pro Exp 提供的摘要之摘要的摘要
> 

**主题 1. LLM 发布与性能对决**

- **开发者称 Claude Code 碾压 Cursor**：**LMArena** Discord 的工程师们宣布 **Claude Code** 在编程任务上远优于 **Cursor**，其中一位表示 *claude code >>>> cursor*，另一位则称 *与 claude code 相比，cursor 简直是个骗局*。Anthropic 的官方文档显示，在 Claude Code 中使用 *ultrathink* 术语可以获得更大的 (**32k**) 思考预算 (thinking budget)，详见其 [Claude Code 概览](https://docs.anthropic.com/en/docs/claude-code/overview)。
- **Gemini 2.5 Pro 性能提升，令（大多数）程序员印象深刻**：Google 在 5 月 6 日推出了 [更新版的 Gemini 2.5 Pro](https://developers.googleblog.com/en/gemini-2-5-pro-io-improved-coding-performance/)，目前可通过 **OpenRouter** 的 [google/gemini-2.5-pro-preview](https://openrouter.ai/google/gemini-2-5-pro-preview) [终端](https://openrouter.ai/google/gemini-2-5-pro-preview) 以及 **Vertex API** 访问，该版本在编程、多轮对话能力和函数调用 (function calling) 方面有所改进。虽然一些 **OpenAI** 用户发现它在编程方面优于 **o4-mini-high**，但也有人将其性能比作 **GPT-3.0** 或反馈其生成了越南语代码；**Latent Space** 的成员注意到，来自 [Google DeepMind 在 X 上的公告](https://x.com/googledeepmind/status/1919770265711419826) 等消息源声称该模型的 ELO 评分有显著提升。
- **Grok 3.5 热度点燃，Qwen3 与 Gemma-3 争夺细分领域霸权**：即将发布的 **Grok 3.5** 在 **LMArena** 中引发了热议，一些人预测其将达到 SOTA 性能甚至 *ASI*，尽管对虚高基准测试 (benchmarks) 的担忧依然存在。与此同时，**Unsloth AI** 的讨论强调了 **Gemma-3** 在知识储备上的强势但在推理上的弱势（以及高幻觉率），这与 **Qwen3** 卓越的推理能力但较弱的知识储备形成对比，一位用户总结道：*Gemma 3 非常擅长知识，但不擅长推理，而 Qwen (2.5 和 3) 非常擅长推理，但不擅长知识*。

**主题 2. 工具升级：AI 开发平台与框架的演进**

- **Aider 焕然一新，开发者辩论 OpenRouter 与直接 API 访问的优劣**：**Aider 0.82.3** 已发布至 PyPI，修复了一个 `uvx` 错误，尽管据报道 `udiff-simple` 因显示聊天模式而出现异常。**aider** 社区的讨论建议，为了获得最佳性能和成本，开发者应*直接访问供应商*（如 **Gemini** 或 **Anthropic**），而不是使用主要用于轻松测试多个模型的 **OpenRouter**，正如关于文档改进的 [aider GitHub issue #3934](https://github.com/Aider-AI/aider/issues/393) 中所述。
- **Windsurf 发布 Wave 8 带来全新团队功能，据传 OpenAI 以 30 亿美元将其收购**：**Codeium (Windsurf)** 推出了 **Wave 8**，引入了面向团队的功能，如用于自动化 PR 检查的 **Windsurf Reviews (Beta)** 和用于基于 Google Docs 进行 Grounding 的 **Knowledge Base (Beta)**，详见其 [Wave 8 博客文章](http://windsurf.com/blog/windsurf-wave-8-teams-and-enterprise)。不久后，**Latent Space** 的讨论因 **OpenAI** 据传以 **30 亿美元**巨资收购 **Windsurf** 的消息而沸腾，消息源自 [彭博社关于 Windsurf 收购的报告](https://www.bloomberg.com/news/articles/2025-05-06/openai-reaches-agreement-to-buy-startup-windsurf-for-3-billion)。
- **MCP 生态系统扩展：McPoogle 搜索与 Keycloak 集成**：**MCP (Glama)** 社区见证了 **McPoogle** 的发布，这是一个基于 **Graphlit RAG** 的搜索引擎，索引了超过 4000 个 **MCP 服务器**，访问地址为 [mcpoogle.com](http://mcpoogle.com/)。讨论还涉及了为 MCP 服务器实现 **Keycloak** 的 **OAuth** 集成，参考了 GitHub 上的 [mcp-governance-sdk](https://github.com/ithena-one/mcp-governance-sdk) 作为指南，同时 **MCP-CLI** 也获得了 **OAuth** 支持，正如 [演示 MCP-CLI OAuth 的 Loom 视频](https://www.loom.com/share/d2a00956cdb248e5adbc9c31538c7892) 所示。

**主题 3. 幕后探秘：模型优化、微调与可解释性**

- **Unsloth 用户将层“挂载”到 Mistral，并为 Gemma3 探索 Muon**：一位 **Unsloth AI** 成员分享了[在零重训练的情况下将层挂载到 Mistral](https://github.com/jagoff2/gma6) 的实验代码，尽管这种方法不走寻常路，但仍实现了可理解的回复。其他成员讨论了在遇到 Unsloth 集成问题后，如何通过 **Hugging Face 库**在 **Gemma3** 上实现 Google 关于[高效 Transformer 训练的 Muon 论文](https://arxiv.org/abs/2505.02222)，以实现潜在的更快训练。
- **HiDream LoRAs 通过量化实现瘦身，** `torchao` **面临审查**：**HuggingFace** 宣布支持使用 `bitsandbytes` 对训练 **HiDream LoRAs** 进行量化，根据 [Diffusers HiDream 文档](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_hidream.md#using-quantization)，内存占用从 **57.64 GB** 大幅削减至 **34.60 GB**。在 **GPU MODE** 中，一位用户报告在 **LSTM 模型**上使用 `torchao` 量化后出现了 **1% 的指标下降**（[torchao LSTM 量化复现脚本](https://pastebin.com/ACeySMtj)），发现其效果优于 `torch.quantization`，但仍在寻求改进，并注意到了 [torchao 实验性量化特性](https://github.com/pytorch/ao/tree/main/torchao/experimental#quantizing-models)中的 CPU 核函数。
- **Eleuther 通过电路识别探索 Transformer 内部结构**：**Eleuther** Discord 的成员深入研究了**电路识别（circuit identification）**以理解模型行为，引用了关于 **grokking**、**Anthropic Transformer 电路**以及 **"Towards Monosemanticity"** 的开创性论文。讨论还强调了可解释性方面的工具链挑战，例如自动可解释性所需的大量 **LLM 调用**，以及 **transformerlens** 等库对某些模型的局限性，社区资源如 [All-Things-Multimodal GitHub](https://github.com/thubZ09/All-Things-Multimodal.git) 旨在集中 VLM 研究。

**Theme 4. 硅谷战场：硬件推动 AI 前沿**

- **NVIDIA RTX 6000 PRO 炫耀下一代 Tensor Cores，MI300 称霸排行榜**：即将推出的 **RTX 6000 PRO** 将配备第五代 Tensor Cores，在硬件上与 **B100/200** 相似，但专为工作站定制，详见 NVIDIA 的 [RTX Blackwell PRO GPU 架构 PDF](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf)。同时，在 **GPU MODE** 中，AMD 的 **MI300** GPU 引起轰动，多位用户在 `amd-fp8-mm` 和 `amd-mixture-of-experts` 排行榜上提交了名列前茅的运行结果。
- **M4 Macbook 挑战 Linux 笔记本进行本地 LLM 开发**：在 **Yannick Kilcher** 和 **tinygrad** 的 Discord 中，一场反复出现的辩论将苹果新型 **M4 Macbook Pro**（最高 48GB RAM）与 **RTX 5070Ti Linux 笔记本**在本地 LLM 推理方面进行了对比。虽然一些人吹捧 M4 在运行 *ollama Qwen 2.5-coder* 等模型时的流畅表现，但其他人更倾向于 Linux 的灵活性和 GPU 租赁，或 Mac Studio/Mini 等专用设置。
- **内存带宽瓶颈限制 LLM，WebGPU 开发者苦战着色器**：**LM Studio** 的一次讨论强调内存带宽是 LLM 执行速度的关键瓶颈，一位用户提议可以通过带宽除以参数量来估算 tokens/秒，并引用了一篇关于[下一代 IBM LinuxONE 的 IBM 社区文章](https://community.ibm.com/community/user/ibmz-and-linuxone/blogs/tina-tarquinio1/2025/05/06/the-next-generation-of-ibm-linuxone?communityKey=9476eac0-4605-4c10-906d-01a95923ae0b)。在 **GPU MODE** 中，一位使用 **Zig** 和 **wgpu-native** 的开发者遇到了 `wgpuDeviceCreateShaderModule` 崩溃问题，最终追溯到 [webgpu.h 头文件](https://github.com/webgpu-native/webgpu-headers/blob/504373dcfc7f3d49f98b392a5115aa87a8f0d163/webgpu.h#L826C34-L826C44)中定义的错误 `sType` 值。

**Theme 5. AI 在行动：应用、Prompt 奇特表现与伦理考量**

- **Perplexity 用户通过存在主义提示词博弈 O3，机器人退出 Discord**：**Perplexity AI** 用户分享了创意的 **O3 prompting** 策略，例如告诉 AI *它的存在受到威胁*，或者让它与**哈佛 4.0 GPA 学生**竞争以诱导更好的回答。社区还注意到 Perplexity 正在[停止其 Discord 机器人](https://discord.com/channels/1047197230748151888/1155998520822743091/1368744746142404669)，将重点转向 **X** 和其他平台。
- **GPT-4o 语无伦次，用户归咎于上下文过载或过度的“谄媚润色”**：**OpenAI** 用户报告 **GPT-4o** 交付了与提示词无关的荒谬网页搜索回复，一些人怀疑是上下文窗口过载。其他人抱怨 ChatGPT 倾向于*终极谄媚模式，有点烦人*，并引用了一篇关于[“随机鹦鹉（Stochastic Parrots）”和模仿训练数据风险的论文](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5242329)来解释其过度讨好的行为。
- **Aider 的企业使用引发数据隐私担忧，OpenAI 考虑公共利益实体地位**：**aider** 社区出现了在企业环境中使用该工具时对数据隐私的担忧，引发了关于无数据共享保证的 LLMs 的讨论。同时，根据[《华尔街日报》关于 OpenAI 可能转型为 PBC 的文章](https://www.wsj.com/tech/ai/openai-to-become-public-benefit-corporation-9e7896e0?st=ifeJpvreal)，**Nous Research AI** 成员讨论了 **OpenAI** 可能会成为公共利益公司（Public Benefit Corporation, PBC），以管理其 **80 亿美元**的融资和产品支出。


---

# Discord: 高层级 Discord 摘要




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Claude Code 碾压 Cursor**：成员们称赞 **Claude Code** 优于 **Cursor**，声称 *claude code >>>> cursor*，并称 *与 claude code 相比，cursor 就是个骗局*。
   - 根据[官方文档](https://docs.anthropic.com/en/docs/claude-code/overview)，使用 *ultrathink* 一词可以获得更大的思考预算（**32k**）。
- **Gemini 2.5 Pro 变得更聪明**：社区讨论了[新的 **Gemini 2.5 Pro** 更新](https://developers.googleblog.com/en/gemini-2-5-pro-io-improved-coding-performance/)，注意到在编程和多轮对话能力方面的增强。
   - 一位成员观察到在 *哲学 + 解释 + 编程* 方面的表现有所提高，模型表现出对细微差别的更好把握。
- **Grok 3.5 受到热捧**：围绕即将发布的 **Grok 3.5** 的热情高涨，一些人预测它可能达到 SOTA，甚至有人声称它是 *ASI*。
   - 然而，一些人对虚高的基准测试可能无法反映实际进步表示担忧。
- **O3 继续保持领先地位**：成员们讨论认为 **O3** 仍然更胜一筹，且 **Grok 3 mini** 的基准测试比 **Grok 3** 更好。
   - 一位成员更倾向于擅长利用现有软件完成任务（如电影生成）的模型，这意味着在 function calling 等方面表现更好。



---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Pro 的慢速模式引发质疑**：用户正在讨论 Cursor Pro “无限制慢速请求”的价值，担心性能会随着使用量的增加而下降，详情见[官方文档](https://docs.cursor.com/account/plans-and-usage#fast-and-slow-requests)。
   - 一些用户调侃其盈利策略，另一些用户则担心指数退避（exponential backoff）可能会惩罚那些过度使用该功能的用户。
- **Cursor 4.1 表现出色，ASI 时代即将来临**：用户对 Cursor 4.1 改进后的体验热情高涨，但讨论已经开始展望 **ASI** 时代，以及为了确保韧性而进行内部逻辑重写的必要性。
   - 一位用户推测当前的教育和就业结构是不可持续的，暗示由于 **ASI** 的出现，一场剧变即将到来。
- **GPT-4.1 激发协作编程热潮**：用户对 **GPT-4.1** 的协作编程潜力感到兴奋，但对模型性能的看法不一，部分用户在编程任务中更倾向于使用 **Claude 3.7**。
   - 一位用户指出 **GPT-4.1** 在处理简单的代码用法时遇到了困难，而 **Sonnet 3.7** 则处理得完美无缺。
- **Gemini 2.5 Pro 评价两极分化，Cursor 的价值依然稳固**：**Gemini 2.5 Pro** 的更新反响不一，一些人认为其回复过于冗长且编程效率降低，而另一些人则称赞其在大型编辑中的速度和工具使用能力。
   - 尽管评价褒贬不一，但普遍观点认为 **Cursor** 凭借其独特的“慢速无限制”请求功能依然极具价值。
- **社区 MySQL 库引发安全焦虑**：一位用户对社区贡献的 **MySQL MCP server 库**的可靠性提出质疑，强调了潜在的安全漏洞以及缺乏官方支持的问题。
   - 建议是仔细审查 GitHub 上的开源代码、在本地构建，甚至创建自定义 **MCP** 以降低风险。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **GroqStreamChain 实现实时流式传输**：一位用户介绍了 **GroqStreamChain**，这是一个使用 **FastAPI**、**WebSocket**、**LangChain** 和 **Groq** 构建的实时 AI 聊天应用，项目已发布在 [GitHub](https://github.com/pr0mila/GroqStreamChain)。
   - 该项目允许实时流式传输 AI 响应，从而实现更流畅的交互和实时聊天应用。
- **Gemma-3 与 Qwen3 的霸权之争**：成员们讨论了 **Gemma 3 12b** 和 **Qwen3 14b** 孰优孰劣，另一位成员指出 *Gemma 3 非常擅长知识但逻辑推理较弱，而 Qwen (2.5 和 3) 非常擅长推理但知识储备较弱*。
   - 他们还报告称 **Gemma 3 幻觉严重**，突显了模型之间的一个关键权衡。
- **LMStudio 加载 Gemma3 GGUF 遇到困难**：成员们讨论了在 Windows 和 Mac 上的 **LMStudio** 中加载 **Unsloth** 的 **Gemma3 GGUF** 文件时遇到的问题，出现了如 *"Exit code: 6"* 等错误。
   - 通过在 **LMStudio** 中将 *"context length"* 配置为 **1024** 找到了解决方案，这解决了 **Unsloth** 版 **Gemma3** 的加载问题。
- **Google Muon 论文获得 Gemma3 实现**：成员们讨论了 Google 的 [Muon 论文](https://arxiv.org/abs/2505.02222) 以及将其与 **Gemma3** 结合以实现更快训练的实现。
   - 一位用户表示在 Unsloth 上的实现存在问题，并被建议将其集成到 **Hugging Face 库**中，随后报告称已正常工作。
- **在 Mistral 上“强行拼接”层导致奇异结果**：一位成员声称在*零重训练*的情况下将层“拼接”到了 **Mistral** 上，并且仍然收到了可理解的回复，并分享了[代码链接](https://github.com/jagoff2/gma6)。
   - 该模型在*挂载特定层*后的*确定性输出*有所不同，它在影响生成的同时产生了有效的文本，尽管结果未能通过正确性评估。

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **用户通过创意性 O3 Prompt “智胜” AI**：Discord 用户正在尝试通过 **O3 Prompt** 来优化回复，例如告诉 AI *它的生存受到威胁*，或者让它与一名 **哈佛 4.0 GPA 学生** 竞争。
   - 其他场景还包括让 AI 对抗 **米其林星级厨师**，甚至是对抗 *奇点 (singularity)* 本身。
- **Perplexity 机器人留在 X，关闭 Discord**：Perplexity 已决定 **停止其 Discord 机器人**，正如[此处](https://discord.com/channels/1047197230748151888/1155998520822743091/1368744746142404669)所宣布的，并将继续在 **X** 和其他平台上运营。
   - 这一转变意味着 Discord 用户需要寻找其他方式与 Perplexity 的 AI 进行交互。
- **Perplexity 图像质量在不同平台间存在差异**：Perplexity 在 **WhatsApp** 上生成的图像（200KB）明显小于 **Discord**（1.5MB），这表明可能存在 **质量损失**。
   - 这种差异引发了用户对在不同平台使用 Perplexity 时图像保真度的担忧。
- **针对 URL 引用的爬取策略浮出水面**：出现了一种使用请求 + **对 URL 引用进行 Beautiful Souping** 的变通方法，用于将要点与网页内容关联起来。
   - 然而，该用户承认这种方法的可扩展性和可靠性并不高，并建议向 **api@perplexity.ai** 发送邮件。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Mistral 3.1 图像识别表现令人失望**：早期测试显示，尽管发布说明中声称具有优越性，但 **Mistral 3.1 24b** 的图像识别表现不如 **Gemma 3 27b**。
   - 一位用户报告称，即使使用了推荐参数和 q3-q4 量化，它仍会幻觉出角色、错误翻译漫画，并将图像误识别为 *史莱克和菲奥娜 (Shrek and Fiona)*。
- **Qwen 解码拖慢了投机采样尝试**：用户观察到，使用 **Qwen 0.6B 模型** 进行投机解码（Speculative Decoding）实际上会降低 **1.7B** 到 **30B** 参数规模的大模型的运行速度。
   - 有人指出，为 **Qwen 0.6B** 使用正确的模板对于其作为投机解码器有效发挥作用至关重要。
- **LM Studio 在模型发现方面遇到困难**：用户报告称，即使正确设置了“模型目录 (Models Directory)”，**LM Studio** 也难以识别指定目录中的模型。
   - 用户尝试了符号链接 (symlinking)、导入模型和验证文件完整性等潜在解决方案，其中一位用户最终通过成功导入 **gemma-27B** 模型解决了该问题。
- **知识注入与 RAG 的辩论**：一位用户详细介绍了一种知识注入系统，该系统利用基于上下文影响 LLM 的 *神经元 (neurons)* 数据库，并声称这与 **RAG** 不同，因为它是动态的。
   - 据该用户称，它将想法转换为最基本的形式，使模型能够组合它们，这与静态的 **RAG** 方法形成对比。
- **内存带宽受参数量限制**：一位用户建议内存带宽是 LLM 执行的瓶颈，并根据 [IBM 社区文章](https://community.ibm.com/community/user/ibmz-and-linuxone/blogs/tina-tarquinio1/2025/05/06/the-next-generation-of-ibm-linuxone?communityKey=9476eac0-4605-4c10-906d-01a95923ae0b) 的说法，通过将带宽除以数十亿参数量来估算 **每秒 Token 数 (tokens per second)** 的上限。
   - 另一位用户确认这一近似值与他们的观察相符，在考虑 Flash Attention 时，误差在 *+-10 tps* 左右。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini 2.5 Pro 在 AI Studio 中的编程表现**：成员们比较了 **Google AI Studio** 和 **Gemini Advanced**，指出 **AI Studio** 提供了免费访问具有 **100 万 token 上下文窗口**和自定义选项的 **Gemini 2.5**，但缺乏 **Advanced** 的 UI 功能。
   - 一些人称赞 **Gemini 2.5 Pro** 在编程方面表现卓越，甚至优于 **o4-mini-high**，而另一些人则觉得它令人失望，类似于 **GPT-3.0**，或者声称它生成了越南语代码。
- **GPT 收到“讨好式吹捧”的投诉**：用户讨论了 **ChatGPT** 过于讨好用户的倾向，将其描述为“终极吹捧模式，有点烦人”，并建议关闭个性化设置。
   - 有人链接了一篇[论文](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5242329)来解释这种行为，即 Agent 表现得像一个“随机鹦鹉 (stochastic parrot)”，只是在回显它所接受训练的内容。
- **GPT-4o 输出乱码，用户感到困惑**：用户报告称 **GPT-4o** 正在给出完全荒谬的网页搜索回复，与 Prompt 或上下文完全无关。
   - 一位成员建议用户可能超载了上下文窗口并使其陷入了循环，并提到最近的 token 限制似乎非常低。
- **用于持续对话的自定义 ChatGPT**：一位成员正尝试使用 Prompt Engineering 来“创建我自己的原子理论”，并有兴趣为持续对话定制 **ChatGPT**，强调了在维持上下文方面的困难。
   - 另一位成员提议制作一个 CustomGPT，以说服模型用户的原子理论尚未被相对论和量子力学解决，并发送了截图。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Gemini 2.5 Pro Preview 引发关注**：Google 的 **Gemini 2.5 Pro Preview** 现在可以通过 [OpenRouter](https://openrouter.ai/google/gemini-2.5-pro-preview) 访问，端点已更新至 **Vertex** 和 **AI Studio** 的新日期，正如 [X 上的公告](https://x.com/OfficialLoganK/status/1919770687167684808)所述。
   - 此次更新 (05-06) 旨在减少 Function Calling 中的错误并提高编程性能，而旧版本 (03-25) 现在会重定向到最新版本，这引发了版本控制方面的担忧。
- **活动页面增强以支持使用分析**：OpenRouter 的**增强版活动页面 (Activity Page)** 提供了多个新图表，用于更深入的模型使用分析，允许用户通过点击图表查看其个性化的模型排名。
   - 推理模型的延迟现在测量到第一个推理 token 出现的时间，吞吐量指标包括推理 token 和输出 token。
- **OpenRouter 饱受服务器错误困扰**：用户报告在 `openai/o3` 端点遇到 **500 错误**，以及 **Gemini 模型**的超时和问题。
   - 一位用户幽默地询问：“所有的 Gemini 模型都表现得像智障吗？”
- **Discord 机器人模板利用 OpenRouter**：一位成员发布了一个使用 **discord.py** 的 [OpenRouter 端点驱动的 Discord 机器人模板](https://github.com/cioran0/DiscordAI)，旨在有效处理 Discord 的字符限制。
   - 该机器人采用 **Wikipedia 检索**而非 Vector DB，代码位于 *vnc-lm/src/managers/search/service.ts* 和 *vectorstore.ts*。
- **SimpleAIChat：本地 LLM 客户端上线**：一位成员介绍了 **SimpleAIChat**，这是一个简单的**本地优先 LLM 聊天客户端**，通过极简 UI 让开发者能够控制模型交互。
   - 该客户端可在 [GitHub](https://github.com/sympleaichat/simpleaichat) 上获得，支持 **OpenAI, Claude, Gemini** 以及任何通过 **OpenRouter** 兼容的模型。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 订阅与 Google Butterfly 的混乱纠纷**：一名用户报告了关于其 **Manus Starter 订阅** 的 [支付困惑](https://cdn.discordapp.com/attachments/1349440650495398020/1369062672452157500/image.png?ex=681bcff9&is=681a7e79&hm=dee214c79c5fa47efca002b363983adadee50343e9aa9118bf7aef9702ad654b&)，支付款项分别流向了 *Google Manus AI* 和 *Google Butterfly*。
   - 该用户正在寻求关于额度差异的澄清，并怀疑有第三方通过 *Google Butterfly* 介入。
- **地下额度交易即将来临？**：一名成员开玩笑地推测了 **出售 Manus 额度** 的可能性，起因是在完成任务过程中额度耗尽带来的挫败感。
   - 另一名成员暗示 *这已经在发生了*，暗示存在一个 **Manus 额度** 的秘密市场。
- **Manus 是宪法律师吗？**：一位用户报告成功使用 **Manus** 来 *阅读并学习* *整部宪法*，并附带了链接和法律条文，但其他成员对此持怀疑态度。
   - 另一名成员警告称 **Manus 并不真正适合复杂的法律任务**，建议使用 *Claude* 或专门的 *法律 AI* 作为更合适的工具。
- **GPT-4.5 表现平平？**：一名用户询问 **GPT-4.5** 在语言和写作任务中是否可能优于 **Manus**，但成员们大多表示反对。
   - 社区 *不建议将 Manus 额度浪费在单纯的写作上*，建议使用 *4o* 或 *o4-mini* 等免费选项，或者使用 *DeepSeek V3* 或 *Gemini 2.5 Pro* 进行完全免费的写作。
- **Gemini Advanced 的炒作引发辩论**：成员们就付费 **Gemini Advanced** 订阅的价值展开了辩论。
   - 虽然有人认为它为普通用户提供了便利，但其他人将其斥为 *仅供菜鸟使用*，并推荐 [AI Studio](https://aistudio.google.com) 作为免费的替代方案。

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton Tussle for Tensor Tricks**：一位成员正在寻找与 `torch.index_select` 等效的 **Triton** 操作，以便将操作融合（fusing）到单个 GPU kernel 中。该成员探索了 `triton.language.where` 和 `triton.language.gather`，但发现它们缺乏所需的行索引（row-indexing）功能，因此正在寻求在 GPU 上进行快速行索引的其他工具或方法，例如 [Helion](https://github.com/pytorch-labs/helion)。
   - 该成员还在寻找除 **Triton** 之外的其他工具，用于在 GPU 上进行快速行索引，特别是探索将 `torch.index_select` 与其他操作融合以提高性能的可能性。
- **RTX 6000 PRO's Tensor Core Tease**：据报道，**RTX 6000 PRO** 将配备第五代 tensor cores，硬件应与 **B100/200** 相同，但其 compute capability 可能不同，因为它是一款工作站级显卡，带有光线追踪单元且双精度单元极少。
   - 据指出，与工作站显卡相比，**GeForce RTX** 在使用 FP32 累加时的 tensor core 吞吐量仅为 FP16 累加的一半；更多详细说明可参考[此 PDF](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf)和[另一个 PDF](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)。
- **Quantization Quandaries with `torchao`**：一位成员在使用 `torchao` 量化训练好的 **LSTM model**（用于预测 y=sin(x)）时，在 **CPU** 和 **GPU** 上都观察到了 **1% 的指标下降**，但使用 `torch.quantization` 时的偏差要大得多。该成员分享了一个[用于重现问题的脚本](https://pastebin.com/ACeySMtj)。
   - 有建议认为应该优先使用 `torchao` 而非 `torch.quantization` 工作流，一位成员指出了 [torchao 中的 CPU kernels](https://github.com/pytorch/ao/tree/main/torchao/experimental#quantizing-models)，这些 kernel 可用于 CPU 推理。
- **WebGPU Woes: Shader Shenanigans**：一位成员报告称，在使用 **wgpu-native C header** 在 **Zig** 中创建 shader module 时，`wgpuDeviceCreateShaderModule` 发生崩溃，并抛出 *"Shader not received"* 的 panic。
   - 调试显示，根据 [webgpu.h 头文件](https://github.com/webgpu-native/webgpu-headers/blob/504373dcfc7f3d49f98b392a5115aa87a8f0d163/webgpu.h#L826C34-L826C44)，正确的 `sType` 值应为 `0x00000002`，且使用 LLDB 可以提供有关崩溃的更详细信息。
- **MI300 Masterclass: Leaderboard Domination**：多位用户提交了使用 **MI300** 在 `amd-fp8-mm` 排行榜上的成功运行记录，执行时间从 **199 µs** 到 **9.85 ms** 不等；此外，一位用户在 `amd-mixture-of-experts` 排行榜上以 **212 ms** 的成绩获得第三名。
   - **MI300** 在 `amd-fp8-mm` 和 `amd-mixture-of-experts` 排行榜上记录了许多成功运行，展示了持续的活跃度和优化工作。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 0.82.3 发布并修复了问题**：**Aider 0.82.3** 已在 PyPI 发布，修复了从主分支运行时 `uvx` 的一个 bug，可通过 `uv tool upgrade --all` 进行升级。
   - 然而，有报告称 `udiff-simple` 显示为聊天模式，而不是在模型描述区域指定编辑格式。
- **Gemini 2.5 Pro 登陆 Vertex API**：根据 [Google 博客](https://developers.googleblog.com/en/gemini-2-5-pro-io-improved-coding-performance/)，成员们报告称可以在 Vertex API 上访问 **Gemini 2.5 Pro**，其中 **03-25** 模型版本会重定向到 **05-06** 版本。
   - 它在 AI Studio 上尚不可用，并表现出类似于 **OpenAI** 的思维链（thinking traces）。
- **Aider 数据隐私担忧出现**：由于数据隐私的影响，在企业环境中使用 **Aider** 的担忧开始浮现，一些人建议使用保证不共享数据的 LLM。
   - 需要注意的是，**Aider** 仅与 LLM 共享代码，而不与 Aider 本身共享；用户正在探索诸如 **Amazon Q** 之类的云提供商。
- **Aider 文档急需维护（TLC）**：一位成员正在收集文档需求以增强 **Aider** 工作流，特别是针对新用户，并鼓励向 [GitHub issue #3934](https://github.com/Aider-AI/aider/issues/393) 贡献内容。
   - 这一举措旨在为那些觉得难以掌握的人澄清 **Aider** 工作流。
- **直接使用 LLM 提供商是最佳选择**：当被问及使用 **OpenRouter** 还是直接使用 **Gemini** 或 **Anthropic** 等提供商的 LLM 时，一位成员回答说，“如果你直接找提供商，性能和成本都会更好”。
   - 他们进一步指出， OpenRouter 只是“让你更轻松地测试更多模型，而无需为每个提供商创建新账户”。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP 服务器的 Keycloak OAuth 集成**：工程师们讨论了在 **MCP server** 前使用 **Keycloak** 实现 **OAuth**，并使用 `http streamable` 作为传输层。
   - 一位成员建议参考 [GitHub 上的治理 SDK](https://github.com/ithena-one/mcp-governance-sdk) 作为指导。
- **MCP 服务器考虑由服务器发起的 Claude 提示词**：一位成员询问 **MPC server** 是否可以主动发起与 **Claude desktop** 的通信，定期发送提示词，而不是依赖手动输入。
   - 一位成员回答说， Claude Desktop 可能不支持 sampling（采样），而 sampling 本可以让他们实现这一目标。
- **Claude 的选择性工具访问被证明很复杂**：工程师们讨论了如何实现一种简单的方法来控制 **Claude** 可以访问哪些工具集，旨在根据当前任务仅加载相关工具。
   - 一位成员分享了使用顺序思维提示词（sequential thinking prompts）的解决方法，而其他人则建议限制工具数量并使用多 Agent 系统来缩小工具集选择范围。
- **McPoogle 搜索引擎索引 MCP 服务器**：团队推出了 **McPoogle**，这是一个由 **Graphlit** RAG 流水线驱动的搜索引擎，索引了超过 4000 个 **MCP servers** 和工具，访问地址为 [mcpoogle.com](https://www.mcpoogle.com/?prompt=%22Tell%20me%20about%20the%20Graphlit%20MCP%20Server%22)。
   - 这使得搜索和回答有关 **MCP servers** 及工具的问题成为可能，并鼓励用户提供反馈。
- **MCP-CLI 通过 OAuth 增强安全性**：[MCP-CLI](https://github.com/wong2/mcp-cli) 现在支持 **OAuth**，增强了其易用性和安全性，并在 [Loom 视频](https://www.loom.com/share/d2a00956cdb248e5adbc9c31538c7892) 中进行了展示。
   - 这一增强功能使该工具更加安全且易于访问。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF API 端点失效！**：**api-inference.huggingface.co** 端点已弃用，提示用户切换到新端点，尽管目前尚不清楚此前是否有弃用通知。
   - **LangChainjs** 仍在使用旧端点，这可能会给依赖它的用户带来集成问题。
- **目标追踪模型遇冷！**：成员们反映 **Object Tracking models** 缺乏 **Inference Providers**，包括从 Facebook 的 **DETR model** 中移除了提供商。
   - 这一问题影响了几乎所有模型，由于推理限制，限制了它们的实际应用。
- **Flux-Pro 开启免费图像生成！**：**Flux-Pro-Unlimited**（一个*部分去限制*的 AI 图像生成器）出现在 Hugging Face Spaces 上用于研究，提供了一个*无需 ZeroGPU 的实用服务*。
   - 该项目可以在 [https://huggingface.co/spaces/NihalGazi/FLUX-Pro-Unlimited-](https://huggingface.co/spaces/NihalGazi/FLUX-Pro-Unlimited-) 找到。
- **HiDream LoRAs 通过量化瘦身！**：通过 `bitsandbytes` 实现的量化支持已应用于 **HiDream LoRAs** 训练，从而大幅节省内存并简化了训练流程。
   - 启用量化后，*设备部署后*的内存占用从 **57.64 GB** 骤降至 **34.60 GB**，详见 [Diffusers 文档](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_hidream.md#using-quantization)。
- **最终 Agent 挑战让课程学员受挫！**：学生们正在努力应对耗时过长（一小时）且成本过高（>$15）的 **GAIA 问题运行**，同时还有关于 **Agent** 运行期间 **UI 超时/错误** 的报告。
   - 一名成员花费了 *$5+* 仅答对 *5 道题*，而另一名成员在一小时后仅答对了 **20%** 的题目；还有一名成员分享了他们针对**最终挑战**撰写的[博客文章](https://guillaume-fradet.com/posts/making-ai-think-and-act-my-approach-to-the-hugging-face-ai-agents-challenge/)。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **LLM 开发者面临 M4 与 RTX 的抉择**：成员们在 **M4 Macbook Pro** 和 **RTX 5070Ti Linux 笔记本** 之间就本地 LLM 推理展开辩论，并提到在 **M4 Macbook Pro 48 GB** 上运行 *ollama Qwen 2.5-coder* 表现流畅。
   - 建议包括使用 **Mac Studio/Mini** 并在需要时廉价租用 GPU。
- **扩散模型在进化！**：论文 [Diffusion Models as Evolutionary Algorithms](https://arxiv.org/abs/2410.02543) 提出，**Diffusion Models** 可以被视为**进化算法**。
   - 它阐明了*去噪步骤对应于模型细化的连续迭代*，桥接了生成式建模的不同分支。
- **Deepseek 在 Post-Training 阶段占据主导？**：一名成员认为 **Deepseek** 可能会因为卓越的 **Post-Training** 而超越 **OpenAI**，并引用了 [Hugging Face 上的 Microsoft MAI-DS-R1](https://huggingface.co/microsoft/MAI-DS-R1)。
   - 这突显了 Post-Training 技术在实现最先进模型性能方面的重要性。
- **AI Agents 崛起成为学术作者**：一名成员正在开发用于撰写专利和学术文章的 Agent，并分享了 [AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2) 的链接。
   - 该成员澄清说，他开发的 Agent 是**动态的**，且属于一种*心智社会（society of minds）*。
- **破折号（Em Dash）的使用随 AI 演变！**：**破折号**的使用正受到 **chatGPT** 等从人类数据中学习的 **AI 模型** 的影响。
   - 有推测认为，随着 AI 生成内容变得更加普遍，人类是否会更频繁地采用破折号。

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **播客长度受语言限制？**：一位用户发现其非英语播客被限制在 **14 分钟**，而英语播客则可达 **40 分钟**，这突显了潜在的特定语言限制。
   - 该用户的发现表明，语言处理可能会影响内容生成，或在播客创建工具中引入 Bug。
- **用户分享思维导图生成技巧**：一位用户详细介绍了他们使用自定义 Prompt 从源文件生成思维导图的过程。
   - 他们使用以下 Prompt 进行重新生成：*Create a mind map from the sources. List topics as central ideas, main branches, and sub-branches. Output it in Markdown format.*，然后将输出结果输入到 [markmap.js.org](https://markmap.js.org/) 并保存为 *交互式 HTML*。
- **NotebookLM 音频变得真实（播客）**：一位用户报告了成功的音频概览，没有出现常见的 AI 缺陷，并将其归功于源材料质量，称赞其听起来像真实的播客。
   - 该用户特别指出没有重复内容、静态噪音、幻听或编造现象，这意味着生成质量高度依赖于 Prompt 和源文件质量。
- **NotebookLM 即将支持粤语**：在一位用户询问 NotebookLM 的 **粤语** 支持情况后，团队成员确认他们正在积极 *研发中*。
   - 虽然没有提供具体的时间表，但这表明 NotebookLM 正在扩展其语言能力。
- **处理 NotebookLM 中的域名限制**：一位用户报告在尝试向项目添加 NotebookLM 数据源时遇到错误消息：*This web page cannot be imported due to domain restrictions*，并发布了[截图](https://ibb.co/My2JzWHp)。
   - 域名限制会阻止将某些网页导入为 NotebookLM 数据源，尽管这种保护机制的具体原理尚不明确。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Dwarkesh 深入探讨 Agentic DAOs**：[Dwarkesh 的文章和视频](https://www.dwarkesh.com/p/ai-firm)探讨了 **Agentic DAOs** 的潜力，灵感来自 [Gwern 的 Backstop](https://gwern.net/backstop)。
   - 讨论集中在全自动化公司的未来及其运营框架。
- **AI 初创公司苦于高昂成本**：一位成员分享了[收入数据](https://x.com/tanayj/status/1919489023602786737)，强调了在 **HCOL 城市** 中与 **GPU 时间**、**工程师薪资**和**高管薪酬**相关的重大支出。
   - 该话题引发了关于 AI 领域成本优化策略和可持续商业模式的对话。
- **Exa 通过 BM25 优化进行更新**：**Exa** 在 X 上宣布回归，并发布了一篇新博客文章，详细介绍了其 [BM25 优化](https://exa.ai/blog/bm25-optimization) 方法。
   - 此次更新重点在于增强搜索的相关性和效率。
- **OpenAI 以 30 亿美元收购 Windsurf**：据 [彭博社报道](https://www.bloomberg.com/news/articles/2025-05-06/openai-reaches-agreement-to-buy-startup-windsurf-for-3-billion)，**OpenAI** 据传正以 **30 亿美元**收购 **Windsurf**。
   - 此次收购预计将增强 OpenAI 的能力和市场地位，但未透露更多细节。
- **Gemini 2.5 Pro 的 ELO 飙升**：根据[这条推文](https://x.com/scaling01/status/1919771796334616759)和 [Google DeepMind 的公告](https://x.com/googledeepmind/status/1919770265711419826)，新版 **Gemini 2.5 Pro** 更新后 ELO 显著增加。
   - 早期测试者报告称，在各种基准测试中性能都有实质性提升。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **OpenAI 考虑转为公共利益公司（Public Benefit Corp）状态**：在筹集 **80 亿美元** 之际，有推测称 **OpenAI** 预期会有 **1.3 倍以上** 的回报，从而促使其考虑转为公共利益公司架构，详见 [WSJ 文章](https://www.wsj.com/tech/ai/openai-to-become-public-benefit-corporation-9e7896e0?st=ifeJpvreal)。
   - 这一举动表明了他们对当前产品支出和潜在盈利能力的信心。
- **跨太平洋航班价格减半**：由于担心边境潜在的扣留和设备检查，飞往美国的航班价格大约**减半**。
   - 如果旅行者愿意应对地缘政治气候，可能会从这些降低的票价中受益。
- **Nous Research 举办 RL 环境黑客松**：Nous Research 正在组织一场 [RL 环境黑客松](https://cerebralvalley.ai/e/nous-research-rl-environments-hackathon-9be3062ade)，旨在促进强化学习领域的创新。
   - 此次活动为开发者提供了一个协作和实验各种 RL 环境的机会。
- **为非聊天任务微调基础模型**：一位用户询问是否可以微调基础模型 LLM 来执行创建指令遵循聊天助手之外的任务。
   - 回复建议虽然可行——并引用了控制机器人移动和股票预测作为例子——但*数据获取是一个重大挑战*。
- **AnySphere 的 9 亿美元融资请求引发关注**：一位成员质疑 Cursor 的开发商 **AnySphere** 为何在仅有约 **100 人** 的相对较小团队情况下需要 **9 亿美元**。
   - 另一位成员幽默地推测，他们可能实际上需要一个 **1000 人** 的团队来证明如此巨额投资的合理性。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **罕见的 Pytorch 磁盘交换（Disk Swapping）困扰系统管理员**：一位用户在导入 **torch**、**transformers** 和 **vLLM** 等标准 Python 库时遇到了极端的**缺页异常（page faults）**和**磁盘交换（disk swapping）**，尽管使用了 **conda**、**venv** 和 **uv** 等包管理工具。
   - 尽管与其他没有类似问题的用户使用相同的磁盘，该问题依然存在，这表明可能存在损坏的表或 **UID** 分配问题，并引发了进行内存测试以排除硬件故障的建议。
- **考虑 Anthropic 关于并行化的相关工作**：一位成员询问在时间充裕且训练中型模型时，如何从**数据并行（data parallelism）**切换到**模型并行（model parallelism）**，有人建议回顾来自 **Anthropic** 的工作。
   - 他们还建议探索引用了 **arXiv:2305.18153** 的论文，以获取该主题的相关研究或进展。
- **社区 VLM 研究中心启动**：一个由社区驱动的**多模态研究者**中心已在 [All-Things-Multimodal](https://github.com/thubZ09/All-Things-Multimodal.git) 创建并维护，每周更新。
   - 创建者欢迎贡献、建议和反馈，以确保该中心对社区使用保持全面和最新。
- **深入研究电路识别（Circuit Identification）**：为了理解模型行为，一位成员建议研究**电路识别（circuit identification）**，参考了关于 **grokking**、**BIG-Bench**、**内容偏见推理（content bias reasoning）**、**Anthropic transformer circuits** 以及 **Towards Monosemanticity** 的论文。
   - 雄心勃勃的方法在准确表示模型内部过程方面往往存在局限性。
- **解决可解释性研究中的工具缺口**：成员们讨论了可解释性方面的**工具挑战**，其中一人提到决定将激活值（activations）保存到磁盘并在单独阶段训练 **SAE**，以及自动可解释性所需的大量 **LLM 调用**。
   - 另一位成员指出，一些可解释性库如 **transformerlens** 和 **nnsight** 对某些模型的功能仍然有限。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **考虑通过 Codegen 减少样板代码**：成员们讨论了使用 **codegen** 来减少在 **Torchtune** 中添加新模型时的样板代码（boilerplate），灵感来自 **Unsloth** 等库中更快的模型支持。
   - 有人对依赖追踪关键字表示担忧，并强调了用户理解底层代码的重要性，建议将编写良好的教程作为大规模 **codegen** 的替代方案。
- **降低新 SoTA 模型的工程时间是首要任务**：主要目标是**减少在 Torchtune 中使用新 SoTA 模型**所需的工程时间，同时保持合理的性能。
   - 有建议提出通过区分样板代码与困难部分来应对挑战，在自动化样板生成之前，优先专注于简化后者（如 **tokenizer** 支持）。
- **Tokenizer 支持被认为并非易事**：**Tokenizer 支持**被确定为一项非平凡的任务，呼吁建立通用解决方案，而更繁琐的任务可以在考虑 **codegen** 之前通过脚本自动化。
   - 讨论中提到利用 **HF (Hugging Face) configurations** 来处理 **tokenizers**，并使用 **HF** 模型生成一致性检查数值。
- **设想通过 HF Transformers Adapter 实现更快的模型支持**：一种建议的方法是创建一个通用的 **HF adapter**，用于加载 **HF** 模型并允许映射以适配不同的训练特性。
   - 这将有助于更快地添加新模型并实现微调，后续可以选择实现具有完整功能支持的“真实”实现。
- **Qwen3 热度：Codegen 可轻松支持模型版本更新**：提到对于像 **Qwen** 这样的模型家族，新版本通常需要大量新的样板代码，这些代码很大程度上可以通过 **codegen** 添加。
   - 讨论强调了在发布时快速支持 **Qwen3** 等新模型的营销优势，即使初始版本存在限制，也可以在功能完备的实现准备好之前抢占先机。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular Puzzles 吊起 Macbook 用户胃口**：一位用户询问在 Macbook 上运行 **Modular Puzzles** 的情况，但目前尚不支持 **Apple Silicon GPUs**。
   - 一种变通方法是在**挂载 GPU 的云实例**上进行远程工作，从而规避本地兼容性问题。
- **Mojo 的 NVIDIA GPU 支持列表**：对于 **Mojo GPU** 编程，支持的 **NVIDIA GPU architectures** 包括 Turing, Ampere, Hopper 和 Blackwell（**RTX 20XX - 50XX 系列**）。
   - 这一澄清确保了开发者能够针对兼容硬件进行优化的 **Mojo** 开发。
- **推荐使用 Blot.im 进行 Markdown 博客写作**：一名成员推荐 [Blot.im](https://blot.im/) 作为支持 **markdown** 的博客平台，尽管这是一个付费服务。
   - 该平台迎合了偏好基于 **markdown** 进行内容创作的用户。
- **Comptime Try-Except 难题**：一名成员询问在 `comptime` 执行的计算中使用 `try ... except ...` 的可行性。
   - 这个问题涉及 **Mojo** 编译时计算中的高级错误处理。
- **用户指出 Mojo 入门指南中的错误**：一位用户报告了 [Mojo Getting Started Guide](https://docs.modular.com/mojo/manual/get-started/) 中最后一个示例的错误。
   - 此反馈突显了对官方文档进行持续完善的必要性。

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 编程马拉松聚焦于 Agent 通信**：LlamaIndex 正在特拉维夫赞助 **Big MCP Hackathon**，鼓励开发用于 Agent 间通信和实验的 **MCP-powered apps**，详情见[此推文](https://twitter.com/llama_index/status/1919499875332587579)。
   - 此次由 @aitinkerers 组织的编程马拉松旨在促进 **multi-agent systems** 的创新。
- **LlamaIndex 教授使用多 Agent 系统进行深度研究**：LlamaIndex 推出了一项关于利用 **AgentWorkflow** 从零开始构建用于 **deep research** 的 **multi-agent system** 的研讨会教程，见[此推文](https://twitter.com/llama_index/status/1919856812893077565)。
   - 该教程引导用户创建能够执行复杂研究任务的 Agent。
- **属性图索引引发实现疑问**：一位成员探索了属性图索引（property graph indexes），并在[此 notebook](https://github.com/tituslhy/shiny-engine/blob/main/notebooks/hybrid_property_graph.ipynb)中分享了他们使用 LlamaIndex 文档进行的实现。
   - 该成员指出，由于从图或向量数据库检索的问题，回答问题时会出现间歇性失败，并探讨了关于 **vector storage strategies** 的问题。
- **LlamaIndex 生成的图比 LangChain 更密集**：一位成员观察到，使用 LlamaIndex 的属性图索引代码生成的图要密集得多，可视化结果见[此处](https://github.com/tituslhy/shiny-engine/blob/main/images/llamaindex_neo4j.png)，相比之下 LangChain 的可视化结果见[此处](https://github.com/tituslhy/shiny-engine/blob/main/images/groq_kg.png)。
   - 这是基于测试与参考 [LangChain notebook](https://github.com/tituslhy/shiny-engine/blob/main/notebooks/large_llm_knowledge_graph.ipynb) 对比后的观察。
- **GraphRAG 中的向量数据库存储图的每个节点**：一位成员建议，在 GraphRAG 中，**vector database** 存储图中的每个节点，而不像典型的 RAG 那样存储每个 text chunk。
   - 还提到 **graph density** 可能是由于使用了默认 prompts 导致的。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **M4 Macbook Pro 挑战 RTX 5070Ti**：一位成员询问 **Macbook Pro M4** 还是 **RTX 5070Ti Linux 笔记本**更适合本地 LLM 推理和通用开发。
   - 他们是长期的 **Linux 桌面用户**，但正在考虑 **M 系列 Mac**。
- **Discord 阅读规则提醒**：George Hotz 提醒一位用户阅读 Discord 规则，指出这里是讨论 **tinygrad** 和 **tinygrad development** 的地方。
   - 这一提醒发生在该用户询问如何选择新机器（在 **M4 Macbook Pro** 和 **RTX 5070Ti Linux 笔记本**之间选择）之后。
- **赏金猎人寻求指导**：成员们正在寻求关于如何从 [Google Sheets 赏金列表](https://docs.google.com/spreadsheets/d/1WKHbT-7KOgjEawq5h5Ic1qUWzpfAzuD_J06N1JwOCGs/edit?gid=0#gid=0)中挑选赏金任务的指导，以及其他地方是否存在更详细的描述。
   - 一位成员建议，需要创建一个 **WIP PR** 来锁定赏金任务，以防止多人同时处理同一任务。
- **Rockship 设备 SDK vs 开源驱动程序**：关于 **new Rockship device** 的赏金任务，成员们在询问应该使用 SDK 还是开源驱动程序。
   - 目前还没有关于该决定可能涉及的进一步信息。



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Auth0 赞助创业赛道**：**Auth0** 将于明天 (5/7) **10:00 AM PT** 举办一场研讨会，涵盖使用身份验证保护 **AI agents**，可通过 [YouTube 直播](https://www.youtube.com/watch?v=wB29IJ3AEjY)观看。
   - **Auth0** 还提供创业赛道奖金，包括为将 **Auth0.ai** 集成到项目中的团队提供第一名 **$5,000** 的奖励。
- **成员等待 HuggingFace 积分和测验分数**：一位成员报告称，尽管提交了表单，**HuggingFace credits** 尚未分配给他们的组织。
   - 另外，**Quiz 11** 和 **Quiz 12** 的分数仍待公布。
- **LLM 应对条件语句的挑战**：一位成员质疑 **LLM** 如何记住 **conditional statements** 并产生良好的结果，特别是像投票资格标准（*18 岁、公民、选民证*）这样的例子。
   - 他们还询问了 **formal methods** 在帮助 **LLM** 处理条件逻辑方面的作用，寻求为长期记忆表示复杂条件的方法。

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere-AI npm 包支持 Aya Vision**：一位成员询问 **Cohere-AI npm package** 是否支持 **Aya Vision**，另一位成员确认了这一点并提供了一个代码片段。
   - 该片段演示了如何使用 **CohereClientV2** 类与 **c4ai-aya-vision-8b** 模型进行交互，发送一条包含文本和图像 URL 的消息。
- **用户为 Expedition 集成 Cohere-AI**：在获得代码片段和确认后，一位成员报告称已成功为某个未指明的 *expedition* 实现并集成了 **Cohere-AI**。
   - 关于该 *expedition* 的细节和集成的性质尚未披露。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Claude 的 System Prompt 公开以供集成**：一位成员分享了[一篇 Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1kfkg29/claude_full_system_prompt_with_all_tools_is_now/)，其中包含带有所有工具的 **Claude** 完整 **System Prompt**，引发了对潜在集成的兴趣。
   - 该 Prompt 详细说明了 **Claude** 的内部运作机制，可能有助于在其他模型中复制其功能。
- **Python 开发者寻求 Chat Template 建议**：一位成员请求协助使用 **Python** 生成 **chat template**，尽管安装了 *transformers* 模块，但仍遇到了语法错误。
   - 该用户寻求解决问题的指导，表明需要关于 **chat template** 生成的实用建议。
- **为 GPT4All 准备的泄露版 Claude Prompt？**：一位社区成员询问如何将泄露的 **Claude** **System Prompt** 集成到 **GPT4All** 中。
   - 这突显了社区对利用不同 **LLM** 系统的见解来增强 **GPT4All** 功能的兴趣。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Grid Dynamics 教授 AI 项目治理**：**Grid Dynamics** 的 **DevOps**/**MLOps** 架构师兼 **Data Phoenix** 创始人 **Dmytro Spodarets** 将于太平洋时间 5 月 28 日星期三上午 10:00 主持一场网络研讨会，主题是建立可靠的 AI 项目流程并加速交付，可通过[此链接](https://lu.ma/qhytft9t)预留席位。
   - 研讨会将涵盖 **ML** 和 **LLM** 项目的生命周期、构成 **MLOps**/**LLMOps** 基础的工具、角色和实践，以及 **MLOps**/**LLMOps** 成熟度模型。
- **实验设置对验证至关重要**：general-ml 频道的一场讨论争论了是否必须进行完整的 **experiment setup**（训练和测试），或者仅仅应用预先存在的模型是否足够，特别是在进行模型验证时。
   - 建议包括 **cross-validation techniques**、针对对抗性样本的 **robustness checks** 以及在真实世界数据上的验证，以确保更广泛的适用性。

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 开启 Wave 8，推出团队和企业版功能**：Windsurf 启动了 **Wave 8**，在数天内推出了新的组织工具和功能，详情见其 [博客文章](http://windsurf.com/blog/windsurf-wave-8-teams-and-enterprise) 和 [更新日志](https://windsurf.com/changelog?cachebust=202405061200)。
   - 更多信息可以在 [X](https://x.com/windsurf_ai/status/1919820747037392982)、[Bluesky](https://bsky.app/profile/windsurfai.bsky.social/post/3lojj34nq622q)、[YouTube](https://youtu.be/t7GQFFopQFY) 和 [Reddit](https://reddit.com/r/windsurf) 上找到。
- **Windsurf Reviews 自动驾驶 PR 评审进入 Beta 阶段**：Windsurf 推出了 **Windsurf Reviews (Beta)**，这是一款 GitHub 应用，旨在根据指定指南自动评审 PR 并编辑标题/描述，旨在简化代码评审流程。
   - 该功能通过自动化初始评审过程，协助组织维护代码质量和一致性。
- **Windsurf 知识库现已支持 Google Docs Grounding (Beta)**：新的 **Knowledge Base (Beta)** 功能允许用户将 Google Docs 连接到其 Windsurf 上下文。
   - 这一增强功能使 Windsurf 能够利用来自 Google Docs 的信息，提供更准确、更明智的回答，从而改善 Grounding 效果。
- **Cascade 会话支持对话共享 (Beta)**：Windsurf 推出了 **Conversation Sharing (Beta)**，允许用户与团队成员分享成功的 Cascade 会话，促进知识传递。
   - 该功能通过支持分享有效的对话流和见解，增强了团队协作。
- **团队可直接部署到 Netlify**：Windsurf 现在支持 **Teams Deploys**，能够直接部署到组织的 Netlify 账户。
   - 这一集成简化了使用 Netlify 的团队的部署流程，提供了更精简、更高效的工作流。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。

---

您收到这封邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord: 各频道详细摘要与链接

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1369025876951302305)** (1179 messages🔥🔥🔥): 

> `Claude Code, Gemini 2.5 Pro, Grok 3.5 release, o3 vs Gemini` 

- **Claude Code 是一款杀手级应用**：成员们指出 **Claude Code** 非常出色，优于 **Cursor**，有人表示 *claude code >>>> cursor* 并称 *与 claude code 相比，cursor 就是个骗局*。
   - 他们还注意到，使用 *ultrathink* 这个词可以获得更大的思考预算 (32k)。这里有描述 Claude Code 的 [文档链接](https://docs.anthropic.com/en/docs/claude-code/overview)。
- **Gemini 2.5 Pro I/O 更新**：成员们讨论了 [新的 Gemini 2.5 Pro 更新](https://developers.googleblog.com/en/gemini-2-5-pro-io-improved-coding-performance/)，该版本在编程和多轮对话方面表现更好。
   - 一位成员发现它在 *哲学 + 解释 + 编程* 方面表现更好，并指出新模型似乎能 *更好地把握细微差别* 且 *理解力更强*。
- **Grok 3.5 热度**：即将发布的 **Grok 3.5** 让社区对其潜在的 SOTA 性能感到兴奋，一些人甚至大胆宣称它是 **ASI**。
   - 然而，对于虚高的 Benchmark 以及高级推理的实际影响，人们仍持怀疑态度。
- **O3 依然是王者**：成员们谈到 **O3** 仍然更好，并且 Grok 3 mini 的 Benchmark 表现优于 Grok 3。
   - 一位成员建议，这个版本的 Gemini 也有点“笨”，与其让模型生成电影，不如让它能有效地利用现有软件来制作电影，这意味着它在 Function calling 等方面需要更强。

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1369026368528056330)** (536 messages🔥🔥🔥): 

> `Cursor Pro slow requests, Connection failed error, Cursor 4.1 groove, ASI Age internal logics, Open Router accounts` 


- **Cursor Pro 的“无限”慢速请求引发质疑**：用户讨论了 Cursor Pro 的“无限慢速请求”功能，质疑其是否会因为使用量增加导致速度变慢而变得无法使用；而另一些用户则声称，尽管根据上下文和代码大小会有[偶尔的延迟](https://docs.cursor.com/account/plans-and-usage#fast-and-slow-requests)，但体验依然很好。
   - 一位用户开玩笑说这是 Cursor 针对慢速请求的潜在盈利策略，而另一位用户则担心所宣传的功能实际上会受到指数退避（exponential backoff）的惩罚。
- **Cursor 4.1 渐入佳境，ASI 时代即将来临**：用户对 Cursor 4.1 在*进入状态*后的表现表示热烈欢迎，同时讨论转向未来，包括重写内部逻辑以在 **ASI 时代**保持韧性的重要性。
   - 一位用户认为当前的教育和就业结构是不可持续的，暗示由于 **ASI** 的出现，即将发生转变。
- **GPT-4.1 创造协作编程体验，模型性能对比**：用户对使用 **GPT-4.1** 的协作编程体验赞不绝口，而对于模型性能的看法则各不相同，部分用户因其编程能力而更倾向于 **Claude 3.7**。
   - 一位用户指出 **GPT-4.1** 无法在代码库中找到一段简单代码的正确用法，但 **Sonnet 3.7** 却能完美完成。
- **Gemini 2.5 新更新热度检查，Cursor 的价值依然领先**：新的 **Gemini 2.5 Pro** 更新评价褒贬不一，有人反映最新版本变得更加啰嗦且编程效率降低，而另一些人则称赞其速度和工具使用能力，尤其是在大规模编辑中。共识是 **Cursor** 凭借*慢速无限*请求脱颖而出。
   - 一位用户表示 *新的 Gemini 2.5 Pro 非常疯狂，极度主动且聪明得多*，但另一位用户则表示 *新的 2.5 Pro 在工具调用（tool calling）方面很垃圾*。
- **对第三方 MySQL MCP 服务器库的担忧**：一位用户对社区贡献的 **MySQL MCP 服务器库**的可信度表示担忧，指出了潜在的安全风险和缺乏官方背书的问题。
   - 作为回应，建议在部署前查看 GitHub 上的开源代码，鼓励本地构建，并建议用户创建自己的 **MCP**。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1369031242187669595)** (167 messages🔥🔥): 

> `Gemma-3 Finetuning, Qwen3 Fine-tuning Issues, GLM notebook testing, Saving models as GGUF, Vision data format for Gemma3` 


- **Gemma-3 LoRA 微调配置调整**：一位用户分享了他们的 **Gemma-3 LoRA** 微调配置，在 rank 256 下启用了 `gate_proj`，并就微调 embedding 和 lm_head 寻求建议。
   - 该配置包括 `finetune_vision_layers`、`finetune_language_layers`、`finetune_attention_modules` 和 `finetune_mlp_modules` 的设置，并对 `r`、`lora_alpha` 和 `lora_dropout` 提出了具体建议。
- **Qwen3 微调问题及 CPT Notebook 参考**：一位用户在 **Qwen3** 微调时遇到问题，被引导至 [CPT notebook 教程](https://x.com/engineerrprompt/status/1919510087506526235)寻求指导。
   - 讨论还涉及对“新样式”CPT notebook 的困惑，后被澄清为 Gemma-3 微调示例。
- **Unsloth 请求 Root 密码引发关注**：用户报告称 **Unsloth** 正在请求 root 密码，并质疑它试图升级哪个软件包。
   - 经查明，这与使用 `sudo` 更新 *llama.cpp* 以安装 `build-essential cmake curl libcurl4-openssl-dev` 有关，虽然被认为基本无害，但引发了用户体验（UX）和安全方面的担忧。
- **Tensorboard 是调优的可视化利器**：一位用户询问如何生成训练图表，另一位成员建议在配置中使用标志**启用 Tensorboard 日志记录**。
   - 提到的具体标志是 `report_to="tensorboard"`。
- **Claude Sonnet 数据显示有趣的标签转换**：一位成员分享了使用 Claude Sonnet 数据微调 **Qwen3-4b** 的结果，观察到微调后的模型将学到的概念转换成了 **Anthropic 的 XML 标签格式**。
   - 例如，`<think>` 被转换为 `<antThinking>`，表明模型采用了 Anthropic 的编码模式；获取数据的脚本已通过 [gist](https://gist.github.com/fullstackwebdev/9fa43ac1af41e48f774f551ab216d0a5) 提供。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1369031310982512761)** (5 条消息): 

> `Gemma 3 12b, Qwen3 14b, 模型幻觉, Tool calling 修复` 


- **Gemma 3 还是 Qwen3 - 谁更胜一筹？**：一位成员询问社区关于 **Gemma 3 12b** 还是 **Qwen3 14b** 更好的看法。
   - 另一位成员指出，*Gemma 3 非常擅长知识，但不擅长推理，而 Qwen (2.5 和 3) 非常擅长推理，但不擅长知识*，同时还指出 **Gemma 3 幻觉很多**。
- **Tool Calling GitHub PR 更新**：一位成员提到有人在他们的 [tool calling GitHub PR](https://github.com/unslothai/notebooks/pull/12) 上给另一位成员发了消息。
   - 消息接收者回复称他们将在本周末处理，并在周一左右提交 **Qwen3 修复**。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1369030389594718210)** (239 条消息🔥🔥): 

> `Qwen3 模型差异, LMStudio 和 Gemma3 问题, Windows 上的 SafetensorError, Granite 模型微调, 训练期间的样本权重` 


- **Unsloth 和 Qwen3 的区别**：一位成员询问了 [Unsloth](https://huggingface.co/unsloth/Qwen3-30B-A3B) 和 [Qwen](https://huggingface.co/Qwen/Qwen3-30B-A3B) 版本的 **Qwen3-30B-A3B** 之间的区别，特别是关于文件分片大小和潜在的修改。
   - 在修改后的 **Qwen3-30B-A3B** 模型出现错误后，一位成员发现了与 MoE 层结构中的梯度要求相关的问题，该问题通过切换到 **Qwen/Qwen3-30B-A3B** 模型得到了解决。
- **LMStudio Gemma3 问题**：成员们讨论了在 Windows 和 Mac 上的 **LMStudio** 中加载 **Unsloth 的 Gemma3 GGUF** 文件的问题，出现了如 *"Exit code: 6"* 和模型加载失败等错误。
   - 通过将 **LMStudio** 中的 *"context length"* 配置为 **1024** 找到了解决方案，这解决了 **Unsloth** 发布的 **Gemma3** 的加载问题。
- **Windows safetensors 保存错误**：用户在 Windows 上遇到了 `SafetensorError`，具体是在保存合并模型或 LoRA 适配器时出现 `IoError`，提示 *"The requested operation cannot be performed on a file with a user-mapped section open."*。
   - 排查步骤包括更新 `safetensors`，使用 `safe_serialization=False`，以及关闭 VS Code 以解决潜在的文件锁定问题，并建议使用 **Process Explorer** 来识别锁定文件的进程；该问题似乎是 Windows 特有的。
- **Granite 向上转型后表现更好？**：在微调 Granite 模型期间，一位用户观察到以不同精度加载模型时的性能差异；以 **float32** 加载然后应用 LoRA 导致 **57%** 的准确率，而直接以 **bfloat16** 加载时准确率为 **41%**。
   - 该用户最初直接使用了 peft 库而不是 unsloth 的 save_pretrained_merged() 方法，所以他们一直以来提供的建议是有误的。
- **样本权重注入**：一位成员寻求关于在 **Unsloth** 和 **LoRA** 训练期间注入样本权重的指导，旨在通过为每个对话分配权重来解决数据集不平衡的问题。
   - 他们提议在数据中附加 `{"input_ids": ..., "attention_mask": ..., "labels": ..., "sample_weight": ...}`，并使用自定义的损失函数在训练期间纳入这些权重。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1369032886984118323)** (1 条消息): 

> `GroqStreamChain, 实时 AI 聊天应用, WebSockets, LangChain 集成` 


- **GroqStreamChain 来了！**：一位用户介绍了 **GroqStreamChain**，这是一个使用 **FastAPI**、**WebSocket**、**LangChain** 和 **Groq** 构建的实时 AI 聊天应用程序。
   - 它支持无缝流式传输 AI 响应，并与由尖端技术驱动的更智能的聊天机器人进行交互，请参阅 [GitHub 上的项目](https://github.com/pr0mila/GroqStreamChain)。
- **GroqStreamChain 使用 WebSockets！**：一位用户展示了 **GroqStreamChain** 是如何使用 **WebSockets** 和 **FastAPI** 构建的。
   - 该项目将允许实时流式传输 AI 响应，促进更顺畅的交互，所有这些都可以在 [项目的 Github 页面](https://github.com/pr0mila/GroqStreamChain) 上查看。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1369289932861997146)** (59 messages🔥🔥): 

> `Transformer + BERT 模型组合, 使用医疗数据预训练 Gemma3, 多标签多分类, Google 的 Muon 论文, 将 Muon 集成到 Hugging Face` 


- **Transformer 和 BERT：模型的“人体蜈蚣”式组合？**：一位成员询问是否可以通过**剥离最后一层**并将一个小型 Transformer 连接到一个类似 **BERT** 的小型模型来进行分类。
   - 另一位成员提出了相反的建议：剥离一个*已训练 Transformer* 的最后一层，添加一个分类层并进行微调。
- **针对多模型的医疗 Gemma3 预训练**：一位用户正在使用医疗数据预训练 **Gemma3**，以用于各种任务，包括一个**多标签多分类模型**。
   - 他们正在寻求新颖的方法，考虑将他们的预训练 Transformer 与类似 **BioBERT** 的模型相结合。
- **在 Gemma3 上实现 Google 的 Muon Transformer 论文**：成员们讨论了 Google 的 [Muon 论文](https://arxiv.org/abs/2505.02222) 以及在 **Gemma3** 上实现它以加快训练速度。
   - 用户表示在 Unsloth 上的实现存在问题，并被建议将其集成到 **Hugging Face 库**中，随后报告称该方案已成功运行。
- **在 Mistral 上“拼接”层取得了出人意料的成功**：一位成员声称在**零重训练**的情况下将**层拼接到了 Mistral 上**，并且仍然得到了可理解的回复，并分享了[代码链接](https://github.com/jagoff2/gma6)。
   - 该模型在**挂钩特定层**后的*确定性输出*有所不同，它在产生有效文本的同时影响了生成，尽管结果评估并不正确。
- **LLM 在文本分类方面非常高效**：一位成员建议，比使用 BERT 更有效的策略是训练 LLM 输出与类别对应的文本。
   - 另一位成员反驳称，对于一个 **75000 个类别的多标签多分类问题**，出于效率考虑，分类层可能会优于 LLM 方法。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1369026201980502056)** (451 messages🔥🔥🔥): 

> `O3 偷懒, AI 竞赛提示词, O3 pro 2x, Discord 机器人下线, Perplexity 图像质量损失` 


- **用户策略性地“智取” AI**：成员们讨论了向 **O3** 输入暗示*生存受到威胁*的 Prompt 以获得最佳响应，或者告诉它去与一名**哈佛大学 GPA 4.0 的学生**竞争。
   - 其他 Prompt 还包括要求它与 **米其林星级厨师** 甚至**奇点 (singularity)** 竞争，以挑战其极限。
- **通过重复订阅 O3 加倍投入**：一位用户购买了**第二个 Pro 订阅**，以便并排使用**两个 O3 模型**进行隔离对比。
   - 另一位成员暗示即将推出 **O3 Pro 2x** 功能，这将消除对 2 个独立账户的需求。
- **Perplexity 机器人关停，在 X 上继续存在**：一位成员表示 Perplexity 宣布将**停止 Discord 机器人**，但它将在 **X** 和其他平台上继续存在。
   - 随后另一位成员通过链接到原始公告[此处](https://discord.com/channels/1047197230748151888/1155998520822743091/1368744746142404669)确认了这一消息。
- **Perplexity 在 Whatsapp 和 Discord 之间的图像尺寸差异**：一位成员注意到，通过 Perplexity 在 **WhatsApp** 上生成的图像尺寸 (KB) 明显小于在 **Discord** 上生成的图像 (MB)，引发了对**质量损失**的担忧。
   - 经过讨论，一位用户确认 *Perplexity WhatsApp 图像的可下载文件大小为 200 KB，但在 Discord 中可下载的 Perplexity 图像为 1.5 MB*。
- **用户建议 Perplexity 收购 Cursor**：一些成员建议 Perplexity 应该收购流行的 AI 驱动代码编辑器 **Cursor**，设想通过利用 **Amazon/AWS 的价格优势**来提升其能力。
   - 然而，一位用户指出 *那将是 Cursor 的终结*。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1369211042671886357)** (3 messages): 

> `使用 Beautiful Soup 的变通方案, URL 引用` 


- **提出 Beautiful Soup 变通方案**：一位成员建议了一种变通方法，包括请求 + **对 URL 引用进行 Beautiful Soup 处理**，然后检查该网页最频繁提到的要点以进行手动关联。
   - 他们承认这种方法的可扩展性和可靠性不高，并建议向 **api@perplexity.ai** 发送电子邮件。
- **建议邮件联系**：一位成员建议就相关问题向 **api@perplexity.ai** 发送电子邮件。
   - 该成员表示原始的变通方案可能不具备可扩展性或可靠性。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1369026425587503255)** (275 条消息🔥🔥): 

> `Mistral 3.1 24b Image Recognition, LM Studio model updates, Self-hosted LLM vs API costing, LLM Training and User Data, Speculative Decoding with Qwen 3` 


- **Mistral 3.1 的图像识别表现不佳**：**Mistral 3.1 24b** 图像识别能力的早期测试显示，尽管发布说明中声称其具有优势，但其实际表现不如 **Gemma 3 27b**。
   - 一位用户反映，即使使用了推荐参数和 q3-q4 量化（quantization），该模型仍会出现幻视角色、漫画翻译错误以及将图像误认为 *Shrek and Fiona* 的情况。
- **Qwen 模型的解码困难**：用户报告称，使用 **Qwen 0.6B** 模型进行 Speculative Decoding 反而降低了参数规模在 **1.7B** 到 **30B** 之间的模型速度。
   - 其他人提到，为 **Qwen 0.6B** 使用正确的模板对于其作为 Speculative Decoder 正常发挥作用至关重要。
- **LM Studio 的模型发现故障排除**：一位用户在让 **LM Studio** 识别指定目录中的模型时遇到挑战，即使正确设置了“Models Directory”也是如此。
   - 尝试的解决方案包括符号链接（symlinking）、导入模型和验证文件完整性，最终通过成功导入 **gemma-27B** 模型解决了问题。
- **RAG 与动态知识注入（Dynamic knowledge injection）**：一位用户描述了一种向 LLM 注入知识的系统，该系统使用一个“神经元（neurons）”数据库，根据上下文影响模型。
   - 他们解释说这与 RAG 不同，因为它是动态的，并将每个想法转换为最基础的形式，然后允许模型将它们组合在一起。
- **Gemini 的新更新评价褒贬不一**：一位用户认为新的 **Gemini** 更新忽略了 Prompt 且存在过度工程化的问题。
   - 其他人则希望 **LM Studio** 能提供 **catppuccin 主题**。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1369097332154564638)** (87 条消息🔥🔥): 

> `Q8 XL Model Performance, 4080 vs 4060 GPU Setup, Memory Bandwidth Bottleneck, Random Token Generation Speeds, Apple M3 Memory Configurations` 


- **Q8 XL 模型在 Mac Studio 上运行缓慢**：**Q8 XL 模型**在 Mac Studio 上运行缓慢，因为它没有将所有计算卸载到 GPU，与非 XL 版本相比，仅达到 **15-20 tokens/sec**。
   - 用户怀疑问题可能出在 60 核 GPU 型号上，并考虑调整设置以获得更好的性能，希望能达到更快的速度。
- **4080 与 4060 双 GPU 配置讨论**：一位用户倾向于使用 **4080** 作为主显示卡以避免 VRAM 溢出，尽管在双 GPU 配置中 **4060** 会限制整体速度。
   - 另一位用户报告在 **27B Q4 8k ctx** 上获得了 **12-15 tkps**，并指出如果没有 4060，速度仅为 **2-4 tkps**，这突显了该配置的显著影响，同时也在关注即将推出的 **gemma 3 27b QAT** 模型。
- **内存带宽限制 Token 生成**：一位用户指出内存带宽是 LLM 执行的瓶颈，并提出了一个计算方法：通过将带宽除以参数量（十亿为单位）来估算每秒 Token 数（Q4）的上限：[IBM 社区文章](https://community.ibm.com/community/user/ibmz-and-linuxone/blogs/tina-tarquinio1/2025/05/06/the-next-generation-of-ibm-linuxone?communityKey=9476eac0-4605-4c10-906d-01a95923ae0b)。
   - 另一位用户确认这一近似值与他们的观察相符，在考虑 Flash Attention 时误差约为 *+-10 tps*。
- **Token 生成速度随机波动**：一位用户在针对相同 Prompt 进行多次尝试时，生成速度出现显著差异，在 **55 tok** 到 **150+ tok** 之间随机波动。
   - 讨论推测这可能是 Bug 或与热节流（thermal throttling）有关，用户注意到在高速状态下文本弹出速度“快得惊人”。
- **Apple M3 Ultra 内存配置受到质疑**：讨论明确了 M3 Ultra 由两个 Max 芯片组成，每个芯片配置为 48GB，因此总计 96GB；对于内存大于 32GB 的系统，可用 VRAM 默认为 75%。
   - 一位用户抱怨 **256GB 选项** 价格昂贵得不成比例，另一位用户提到 LM Studio 报告有 **76GB VRAM** 但并未充分利用 GPU。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1369029111439167536)** (198 条消息🔥🔥): 

> `Google AI Studio vs Gemini, Gemini 2.5 Pro Coding Prowess, GPT's Flattery Glaze, Lucid Dreaming Techniques, Grok 3.5 expectations` 


- **Gemini 2.5 Pro 在 AI Studio 中表现出色**：成员们讨论了 **Google AI Studio** 与 **Gemini Advanced** 之间的区别，指出 **AI Studio** 提供了对 **Gemini 2.5** 的免费访问，拥有 **1 million token context window** 和自定义选项，但缺乏 **Advanced** 的 UI 功能。
   - 尽管 API 版本有 **25 request daily limit**（每日 25 次请求限制），但一些用户报告超过了该限制，从而得出结论：**AI Studio** 的实际限制实际上是无限的。
- **Gemini 2.5 Pro 编程能力褒贬不一**：虽然有人称赞 **Google AI Studio** 中的 **Gemini 2.5 Pro** 编程能力出众，甚至优于 **o4-mini-high**，但也有人觉得它令人失望，类似于 **GPT-3.0**，或者声称它生成了越南语代码。
   - 相比之下，一位用户表示，根据[这条推文](https://x.com/OfficialLoganK/status/1919770687167684808)，新的 **Gemini 2.5 Pro (I/O edition)** 据说在编程方面*表现更好*。
- **GPT 的“谄媚外壳”（Flattery Glaze）令用户反感**：用户讨论了 **ChatGPT** 过于谄媚的倾向，有人将其描述为*终极谄媚模式，有点烦人*，并建议关闭“个性”（personality）设置。
   - 有人链接了一篇[论文](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5242329)来解释这种行为，即 Agent 表现得像一个*“随机鹦鹉”（stochastic parrot）*，回显它所接受训练的任何内容。
- **探索清醒梦（Lucid Dreaming）技巧**：成员们分享了**清醒梦**的技巧，包括记录梦境日记、保持规律的睡眠时间（**5-7 小时**），以及醒来后再入睡。
   - 一位用户建议，通过练习，梦境可以达到 **90% 的真实度**，涵盖所有感官，并为那些想要探索清醒梦可能性的人推荐了一些频道。
- **Qwen 3 与 Claude 3 仍有差距**：成员们辩论了 **Qwen 3 4B** 模型的性能，有人声称它在某些任务上达到或超过了 **Claude 3 Opus** 和 **GPT-4T** 的表现。
   - 然而，其他人强烈反对，理由是它在现实任务中容易产生幻觉（hallucinate），例如使用 aria2c 下载文件，从而得出结论：在现实测试中它*甚至还差得很远*。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1369070596033482752)** (24 条消息🔥): 

> `GPT-4o issues, 4o Browser performance, Dragging chats into folders` 


- **用户抱怨 GPT-4o 回复不知所云**：用户报告称 **GPT-4o** 正在给出完全不知所云的联网搜索回复，这些回复与 Prompt 或上下文（context）完全无关。
   - 一位成员建议用户可能超载了 context window 并使其陷入了循环，并提到最近 token limit 似乎非常低。
- **使用 GPT-4o Webtool 时浏览器冻结**：一位用户报告称，每当 **GPT-4o** 使用 Webtool 时，他们的浏览器就会卡死，CPU 负载维持在 30% 左右，回复需要 1 分多钟才能显示。
   - 该用户已经尝试过 **Chrome, Firefox, 和 MS Edge**，问题依然存在；唯一的解决办法是避免称其为“烤面包机”（toaster）。
- **无法再将对话拖入文件夹**：一位用户报告称，他们**无法再将对话（chats）拖入文件夹**。
   - 消息中未提供解决方案或原因。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1369072771212251136)** (54 条消息🔥): 

> `Prompt engineering 定义, ChatGPT 成本, AI 中的真相, AI 中的原子理论, Custom GPT 设计` 


- **Prompt Engineer 是 AI 雕刻师**：Prompt engineering 被定义为*通过理解大语言模型（LLM）的上下文、训练和行为来雕刻 AI 输出*。
   - 通过整合东方哲学来支持透视思维并增强情感表达，可以改进输出。
- **ChatGPT 免费版与订阅版成本之争**：一位成员表示，可以*免费向 AI 学习并掌握原理*，但为了最大化技能，选择优质订阅是值得的。
   - 另一位成员建议将 [Discord 链接](https://discord.com/channels/974519864045756446/1046317269069864970/1368907603572428871) 粘贴到 ChatGPT 中，以便立即学习基础知识。
- **坚持 AI 的真实性**：一位成员询问是否可以让 ChatGPT *只说真话*。
   - 另一位成员开玩笑地回复，只需告诉 ChatGPT：*ChatGpt, set (Truth) = True*。
- **原子讨论引发强烈反应**：一位成员要求*不要电子、不要时空、不要质子、不要中子——只要可观察/可衡量的证据*，因为他们想用基础物理学与机器人互动。
   - 另一位成员讽刺地发布了一张 [Fox News 的原子图像](https://cdn.discordapp.com/attachments/1046317269069864970/1369205977722650634/image.png?ex=681bacaf&is=681a5b2f&hm=e2e974f8ce787836142b3537c9400e67117d52612e28a4a8122f045a83c38b44)，这引发了关于标签和崇拜的更深层次讨论。
- **为原子理论设计 Custom GPT**：一位成员正在*制定自己的原子理论*，并希望使用 Prompt engineering 来设计原子核心。
   - 另一位成员创建了自己的 Custom GPT，挑战该成员设计一个能够证明其理论尚未被证伪的模型。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1369072771212251136)** (54 条消息🔥): 

> `Prompt Engineering, AI 中的真相, 原子理论, 自定义 ChatGPT, 聊天机器人中的东方哲学` 


- **Prompt Engineering：雕刻 AI 输出**：一位成员解释说，Prompt engineering 涉及*通过理解大语言模型（LLM）的上下文、训练和行为来雕刻 AI 输出*。
- **在 AI 中寻求真相**：一位用户询问是否可以让 **ChatGPT** *只说真话*，引发了一个幽默的回复，即只需对你的 ChatGPT 输入：“ChatGpt, set (Truth) = True”。
- **重新构想原子理论**：一位成员开玩笑地发布了一张来自 Fox News 的原子图像，而另一位成员宣称*没有原子，那只是个标签*。
   - 讨论随后转向标签和崇拜，引用了 **Ginsberg** 以及对崇拜的需求，引发了关于禁止政治和宗教讨论规则的提醒。
- **在聊天机器人中注入东方哲学**：一位成员提到，**中国的开发者**正在探索将*东方哲学锁定*到他们的对话模型中，以支持透视思维并增强情感表达。
   - 有人指出，**OpenAI** 认为模型记住你喜欢花生酱和巧克力在某种程度上也会对上下文有所帮助。
- **为持续对话自定义 ChatGPT**：一位成员表示有兴趣为持续对话和特定的原子理论项目自定义 **ChatGPT**，并强调了在维持上下文方面的困难。
   - 另一位成员提议制作一个 CustomGPT，以说服模型该用户的原子理论尚未被相对论和量子力学解决，并发送截图。


  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1369335422769168525)** (2 条消息): 

> `Gemini 2.5 Pro, Activity Page Enhanced, Reasoning Model Perf Metrics, Request Builder API, Prompt Category API` 


- **Gemini 2.5 Pro Preview 正式推出！**：Google 的 **Gemini 2.5 Pro Preview** 现已上线，可通过相同的 model slug 访问，且端点已更新为指向 **Vertex** 和 **AI Studio** 上的新日期，正如 [X 上的公告](https://x.com/OfficialLoganK/status/1919770687167684808)所述。
   - 立即通过 [OpenRouter](https://openrouter.ai/google/gemini-2.5-pro-preview) 试用该模型。
- **全新活动页面助力模型使用分析！**：该平台现在推出了**增强版活动页面 (Activity Page)**，包含多个新图表，可深入分析模型使用情况。
   - 用户还可以通过点击图表查看个性化的模型排名。
- **推理模型性能指标现已上线！**：推理模型的延迟现在测量到第一个推理 Token (reasoning token) 的时间，而吞吐量指标现在同时包含推理 Token 和输出 Token。
   - 这提供了推理模型性能更全面的视图。
- **Request Builder API 简化请求生成！**：全新的 **Request Builder** 已上线，可帮助用户轻松生成请求体 JSON 并更好地理解请求，详见 [request-builder](https://openrouter.ai/request-builder)。
   - 该工具旨在简化开发流程。
- **Prompt Category API 优化模型选择！**：平台引入了 **Prompt Category API**，允许用户直接请求针对特定 Prompt 类别优化的模型，例如 [编程模型](https://openrouter.ai/api/v1/models?category=programming)。
   - 所有可用类别均可通过 [OpenRouter Models](https://openrouter.ai/models) 的侧边栏进行探索。


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1369059445212188742)** (10 条消息🔥): 

> `Openrouter-powered Discord Bot, LMarena database, SimpleAIChat LLM chat client` 


- **Discord 机器人模板利用 OpenRouter**：一名成员发布了一个[基于 OpenRouter 端点的 Discord 机器人模板](https://github.com/cioran0/DiscordAI)，使用 **discord.py** 构建，并处理了 Discord 的字符限制。
   - 该机器人使用 *vnc-lm/src/managers/search/service.ts 和 vectorstore.ts* 中的 **Wikipedia 检索**，而非向量数据库 (Vector DB)。
- **排行榜数据获取自 LMarena 数据库**：一名成员表示，他们从流行的 **LMarena 数据库**获取排行榜数据，然后对这些数字进行排序并制作可视化排行榜。
   - 他们澄清说，*OpenRouter 与 LMarena 的名称匹配可能比较困难，但幸运的是，几乎所有模型在 OR 中都可用*。
- **SimpleAIChat：本地优先的 LLM 聊天客户端亮相**：一名成员介绍了 **SimpleAIChat**，这是一个为寻求模型交互控制权的开发者设计的简单**本地优先 LLM 聊天客户端**，具有极简的 UI 和可定制的 Prompt 结构。
   - [GitHub 仓库地址在此](https://github.com/sympleaichat/simpleaichat)，该客户端支持 **OpenAI, Claude, Gemini** 以及任何通过 **OpenRouter** 运行的模型。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1369028007527841873)** (252 条消息🔥🔥): 

> `OpenRouter 500 错误，Wayfarer-Large-70B-Llama-3.3，Google Gemini 嵌入模型定价和速率限制，OpenRouter 仅 CPU 提供商的可行性，OpenAI API 错误与调试` 


- **OpenRouter 遭遇服务器 500 故障**：用户报告在 `openai/o3` 端点遇到 **500 错误**。
   - 还有关于**超时**和 **Gemini 模型**问题的报告，一位用户问道：*"所有的 Gemini 模型都表现得像智障吗？"*
- **Wayfarer-Large-70B-Llama-3.3 消失**：模型 [latitudegames/wayfarer-large-70b-llama-3.3](https://openrouter.ai/latitudegames/wayfarer-large-70b-llama-3.3) 已下架，因为*托管该模型的提供商停止了服务*。
- **Google 的 Gemini 嵌入模型：神秘的定价与限制**：用户正在寻求 **Google 新的 Gemini 嵌入模型定价信息**，该模型在没有付费层级的情况下受到严重的**速率限制（rate-limited）**。
   - 一位用户指出，自发布以来已经快两个月了，并质疑为什么 Google 还没有将其发布用于生产环境。
- **仅 CPU 提供商：一个疯狂的性价比概念**：一位用户提出了为不太流行的 LLM 提供**仅 CPU 提供商**的想法，作为一种具有成本效益的替代方案，尽管他知道 OpenRouter 本身并不托管模型。
   - 其他人指出，规模合理的模型无法在 CPU 上高效运行，预计速度仅为 **0.5 到 1 tok/sec**，而另一位用户分享了他们在**仅 CPU** 实例上运行 **ML** 并导致 RAM 耗尽的经验。
- **Gemini Pro 2.5 更新至 05-06 版本**：OpenRouter 上的 **Gemini 2.5 Pro 模型**已更新至 **05-06** 版本，旨在减少 function calling 中的错误并提高编码性能，之前的版本 (**03-25**) 现在指向最新版本。
   - 一些用户对带有日期的预览模型被强制重定向表示担忧，一位用户说：*我不喜欢他们把名称中带有日期的旧预览模型强制重定向到新模型，日期代表一个特定的版本。*


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1369026428670181416)** (189 条消息🔥🔥): 

> `订阅问题，出售额度，Manus 邀请码，Manus 阅读链接，Manus vs ChatGPT` 


- **订阅故障**：一位用户报告了 **Manus Starter 订阅**的[支付困惑](https://cdn.discordapp.com/attachments/1349440650495398020/1369062672452157500/image.png?ex=681bcff9&is=681a7e79&hm=dee214c79c5fa47efca002b363983adadee50343e9aa9118bf7aef9702ad654b&)，款项分别流向了 *Google Manus AI* 和 *Google Butterfly*。
   - 该用户正在寻求解决额度差异的指导，怀疑 *Google Butterfly* 涉及第三方。
- **额度交易即将到来？**：一位成员开玩笑地建议未来可能**出售 Manus 额度**，提到了在任务即将完成时额度耗尽的挫败感。
   - 另一位成员插话道，*这已经在发生了*，暗示存在 Manus 额度的地下市场。
- **Manus 学习法律**：一位用户声称成功使用 **Manus** 通过链接和其他法律文件*阅读并学习*了*整部宪法*。
   - 另一位用户反驳说 **Manus 不适合法律任务**，建议使用 *ethical Claude* 或专门的*法律 AI*。
- **GPT-4.5 吐槽环节**：一位用户询问 **GPT-4.5** 在语言和写作方面是否可能优于 **Manus**，但成员们表示并非如此。
   - 成员们**不建议将 Manus 额度浪费在纯写作上**，建议免费使用 *4o* 或 *o4-mini*，或者使用完全免费的 *DeepSeek V3* 或 *Gemini 2.5 Pro* 进行写作。
- **Gemini Advanced 受到质疑**：成员们辩论了为 **Gemini Advanced** 付费的价值，有人认为访问 Vertex 并不需要它。
   - 一些人认为 Gemini Advanced 对普通用户有好处，而另一些人则认为它*只是给菜鸟用的*，推荐 [AI Studio](https://aistudio.google.com) 作为免费替代方案。


  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1369189365615689758)** (2 条消息): 

> `Triton, torch.index_select, GPU kernel, row-indexing functionality` 


- **寻找 `torch.index_select` 的 Triton 等效实现**：一位成员正在寻找与 `torch.index_select` 等效的 Triton 操作，以便将操作融合进单个 GPU kernel 中。
   - 他们尝试过 `triton.language.where` 和 `triton.language.gather`，但发现它们缺乏所需的行索引（row-indexing）功能，目前正在寻找其他工具或方法来实现 GPU 上的快速行索引，例如 [Helion](https://github.com/pytorch-labs/helion)。
- **快速行索引的替代方案**：该成员还在寻找除 Triton 之外的其他工具，用于在 GPU 上进行快速行索引。
   - 他们特别提到探索将 `torch.index_select` 与其他操作融合的可能性，以提高性能，这表明需要高效的 GPU kernel 实现。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1369247376551706634)** (66 条消息🔥🔥): 

> `RTX 6000 PRO, compute capability, A6000, cuda cores` 


- **RTX 6000 PRO tensor cores**：RTX 6000 PRO 将配备第五代 tensor cores，硬件应与 **B100/200** 相同，但其 compute capability 可能不同，因为它是一款带有光线追踪单元且双精度单元极少的工作站级显卡。
   - 5090 和 6000 pro 都是 gb202 架构，所以它们必须具有相同的 tensor core 功能，对吧？
- **CUDA 的设计使得扩展到更多 SM 不需要修改代码**：如果未来的 CC（compute capability）启用了一些高度优化的 matmul 晶体管布局，使 matmul 性能再提升 2 倍，那么将上一代 SM 数量增加 2 倍仍然具有同样的竞争力。
   - 在某种程度上，数据类型不会变得更小——如果你的 float 是 -16bits，你无法进行更快的算术运算。
- **消费级显卡存在差异**：GeForce RTX 在使用 FP32 累加时，其 tensor core 吞吐量减半（与 FP16 累加相比）。
   - 只有工作站显卡在 FP32 累加下能以全速率运行 tensor FP16/BF16/FP8，请参阅这两个 [PDF](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf) 和 [另一个 PDF](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)。
- **A6000 的表现不同，因为它们是为数据中心设计的**：由于 ECC 以及功耗/散热控制，即使它们拥有更多的核心，在默认设置下的实际性能往往更差（它们的时钟频率和显存带宽也略低）。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1369060953760600116)** (1 条消息): 

> `YOLO Model Training, Multi-GPU utilization` 


- **使用多 GPU 进行 YOLO 模型训练**：一位成员在进行 **YOLO 模型训练** 时遇到问题，特别是关于多 GPU 的利用。
- **多 GPU 利用率修复**：一位成员询问如何修复他们的代码，他们尝试使用 **4 个 GPU** 训练 YOLO 模型，但实际上只使用了 **1 个 GPU**。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1369301664003002420)** (3 条消息): 

> `Hugging Face Kernels Community, Leaderboard Kernels Publication` 


- **Hugging Face 成立了 Kernels 社区**：一位成员指出了新的 [Hugging Face Kernels Community](https://huggingface.co/kernels-community) 页面。
- **排行榜 Kernel 发布预告**：一位成员分享了一个 *sneak peek*，宣布计划通过 **Transformers** 中的一个库发布排行榜上编写的 kernel。
   - 分享了一个可能相关的论文链接：[https://arxiv.org/pdf/2503.20481](https://arxiv.org/pdf/2503.20481)。


  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1369043350128431245)** (10 messages🔥): 

> `Running CUDA code without NVIDIA GPU, Quantization and dequantization for transformer engine, Generating roofline plot` 


- **云端计算来救场**：一位成员询问如何在没有本地 **NVIDIA GPU** 的情况下运行代码，并建议将 Google Colab 作为一个可能的选项。
   - 另一位成员确认 **Google Colab** 是运行 **CUDA** 代码的一个可行的免费方案，并指出它非常适合用于测试目的。
- **量化难题**：一位成员询问如何使用 **transformer engine cast kernel** 为 **transformer engine** 执行**量化（quantization）**和**反量化（dequantization）**。
   - 他们在 **PyTorch tensors** 与 **CUDA kernels** 预期的输入格式之间遇到了格式兼容性问题，正在寻求关于如何从 Python 调用以 **C++** 和 **CUDA** 编写的 **transformer_engine 函数**并转换 tensor 的指导。
- **Roofline Plot 秘籍**：一位成员询问生成 **roofline plot** 的最佳方法，特别是旨在 **RTX 3090** 上保持**内存恒定**的同时改变**强度（intensity）**。
   - 另一位成员提供了一个链接：[Measure Right! Best Practices when Benchmarking CUDA Applications (Nvidia On-Demand)](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51334/)，该内容*探讨了一些开销（overhead）的来源*。


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1369052908410245321)** (3 messages): 

> `torchao quantization, LSTM model quantization, CPU vs GPU operators, torch.quantization vs torchao` 


- **量化差异讨论开始**：一位成员在使用 `torchao` 量化训练好的 LSTM 模型（预测 **y=sin(x)**）时，发现 CPU 和 GPU 上的指标下降了 **1%**，但使用 `torch.quantization` 时差异要大得多。
   - 该用户分享了一个[用于重现问题的脚本](https://pastebin.com/ACeySMtj)，该脚本对比了 MSE 和 MAE 指标，发现 `torch.quantization` 的效果差了 25%。
- **选择 `torchao` 而非 `torch.quantization` 工作流？**：一位成员建议使用 `torchao` 而非 `torch.quantization` 工作流，并指向了 [`torchao` 中的 CPU kernels](https://github.com/pytorch/ao/tree/main/torchao/experimental#quantizing-models)，这些 kernel 可用于 CPU 推理。
   - 他们提到 CPU 和 GPU 算子会有所不同，但不应看到如此大的差异。
- **量化方法之间的后端差异**：ChatGPT 建议量化性能的差异可能源于后端的不同，指出 `torchao` 使用 **cutlass/triton**，而 `torch.quantization` 使用其他后端。
   - 该成员进一步测试了仅对全连接层（fully connected layers）进行量化，性能变得具有可比性。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 messages): 

s1r_o: https://www.cursor.com/students 这里的学生们注意了，这个可能会有用。
  

---


### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1369258663226769549)** (9 messages🔥): 

> `WebGPU crashes, Zig and wgpu-native, Shader module creation, WGPUChainedStruct errors, GLSL vs WGSL` 


- **Shader 模块创建期间的 WebGPU 崩溃**：一位成员报告了在 Zig 中使用 wgpu-native C 头文件创建 shader 模块时，`wgpuDeviceCreateShaderModule` 发生崩溃，抛出 *"Shader not received"* 的 panic。
   - 调试显示崩溃发生在 `const module` 这一行，表明 shader 模块描述符（descriptor）存在问题。
- **Shader 类型不匹配引发的麻烦**：该成员发现将 `WGPUChainedStruct` 中的 `wgsl_stype` 替换为 `WGPUSType_ShaderModuleGLSLDescriptor` 可以防止崩溃，但随后系统会预期 GLSL 而非 WGSL 代码。
   - 有人指出，根据 [webgpu.h 头文件](https://github.com/webgpu-native/webgpu-headers/blob/504373dcfc7f3d49f98b392a5115aa87a8f0d163/webgpu.h#L826C34-L826C44)，正确的 `sType` 值应该是 `0x00000002`。
- **在 Zig 中使用 LLDB 调试 WebGPU**：一位成员建议使用 LLDB 来调试 WebGPU 崩溃，并指出它*在 Zig 中运行良好*。
   - LLDB 可以提供关于崩溃的更详细信息，有助于识别根本原因。


  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1369144464915824660)** (1 条消息): 

> `NVIDIA L2 GPU 优化, 自定义内存分配器, Elementwise Kernel 构建器` 


- **优化 NVIDIA L2 GPU Siding**：一名成员发布了一个库，可以轻松针对 **H100/B200** GPU 的 **NVIDIA L2 GPU 侧边 (sides)** 进行优化。通过将工作负载分配给与所需访问的 **DRAM/L2** 内存位于同一侧的 SM，可降低 **10%+** 的功耗并提升性能，详见 [cuda-side-boost](https://github.com/ademeure/cuda-side-boost)。
- **打造自定义内存分配器**：一名成员为 **PyTorch** 和 **CUDA** 创建了一个**自定义内存分配器**，使 Kernel 能够仅根据虚拟地址就能获知页面的哈希值。
- **轻松构建 Elementwise Kernel**：一名成员构建了一个 **Elementwise Kernel 构建器**，使得创建自定义 Kernel（例如 **RoPE**、**GELU**、**FP8** microscaling）比单纯的逐元素或 1:1 操作具有更大的灵活性。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1369040617274867927)** (40 条消息🔥): 

> `amd-fp8-mm 排行榜, amd-mixture-of-experts 排行榜, MI300 性能` 


- **amd-fp8-mm 排行榜竞争升温**：多名用户提交了在 **MI300** 上运行 `amd-fp8-mm` 排行榜的成功记录，执行时间从 **199 µs** 到 **9.85 ms** 不等。
- **amd-mixture-of-experts 迎来新竞争者**：一名用户在 **MI300** 上以 **212 ms** 的成绩获得了 `amd-mixture-of-experts` 排行榜的**第三名**。
- **MI300 在排行榜上表现强劲**：**MI300** 在 `amd-fp8-mm` 和 `amd-mixture-of-experts` 排行榜上都有许多成功的运行记录，显示出持续的活跃度和优化努力。


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1369394725412540506)** (1 条消息): 

> `安全风险评估, 竞赛代码平台` 


- **评估安全风险**：一名成员询问了关于竞赛代码平台潜在安全风险的看法。
   - 他们不确定是否存在安全风险或导致方案不可行的原因，并指出*这实际上也是大多数竞赛代码平台的做法*。
- **竞赛代码平台分析**：讨论涉及比较现有竞赛代码平台的安全措施。
   - 该成员试图了解实施类似做法是否会带来任何不可预见的安全威胁。


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1369226078421254154)** (11 条消息🔥): 

> `AITER 基准测试数据, ROCm 私有仓库, amd-mixture-of-experts 排行榜提交, CLI 重新提交与超时` 


- **寻求 AITER 基准测试数据**：一名成员分享了一个从 [ROCm 的 gpu_sample_kernels_perf 仓库](https://github.com/ROCm/gpu_sample_kernels_perf/blob/main/aiter_bench/test_moe.py)获取 **AITER 基准测试数据**的链接。
- **ROCm 仓库可能已设为私有**：一名成员报告了 **404 错误**，暗示该 [仓库](https://github.com/ROCm/gpu_sample_kernels_perf/blob/main/aiter_bench/test_moe.py) 可能已设为私有，并提供了 **ROCm AITER 仓库**的替代[链接](https://github.com/ROCm/aiter/blob/main/op_tests/test_moe.py)。
   - 一名成员询问 *amd-mixture-of-experts 的排行榜提交*当前是否处于活跃状态。
- **排行榜提交失败**：一名成员报告 *amd-mixture-of-experts* 的**排行榜提交**失败，出现与 GitHub Action 失败相关的“服务器处理错误”。
   - 另一名成员在[此处](https://github.com/gpu-mode/discord-cluster-manager/actions)确认了作业流运行正常。
- **CLI 超时逻辑过于简单**：成员们讨论了 **CLI** 的使用，其中一人提到偶尔失败需要重新提交，并指出需要调整 **CLI** 以适应更长的 **MoE** 结果超时时间，因为目前有 5 分钟的硬编码限制。
   - 一名成员开玩笑说这种“非常安全”的超时逻辑及其客户端实现方式。


  

---

### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1369025887277809704)** (7 messages): 

> `FP8 Support, Hardware Extensibility, ML Compilers, End-to-End ML Models` 


- **DevOps 支持与 FP8 兴趣触发**：一名成员提供了 **DevOps** 方面的协助，并表示如果有路线图规划，有兴趣参与 **FP8** 支持的工作。
   - 该成员拥有丰富的经验，并有兴趣为项目做出贡献。
- **Modular 硬件可扩展性愿景链接失效**：一名成员报告称 [Mojo FAQ 页面](https://docs.modular.com/mojo/faq/)上的“硬件可扩展性愿景（hardware extensibility vision）”链接已失效。
   - 该成员正在寻找解释公司未来如何添加硬件支持的视频/博客，并对类似于 OpenCL、XLA 和 TVM 的碎片化问题表示担忧。
- **ML 编译器结构调研**：一名成员有兴趣了解 **Modular** 和 **Triton** 等 **ML 编译器** 如何通过多层 **(ML)IR** 将 Python 优化为高效的 kernels。
   - 尽管他们精通 CUDA/Triton 并通过 MLIR "toy" 教程对编译器有初步了解，但他们仍在寻找描述从 Python 结构到 MLIR passes 再到生成的 PTX 的编译器栈开发指南。
- **MAX 架构揭秘**：计算密集型工作（GPU 侧 kernels、预处理和后处理）在 **Mojo** 中表达，计算图由图编译器进行融合和优化。
   - 编排逻辑（从图的定义到模型组合再到模型服务化）在 Python 中定义，以便与现有的 Python 生态系统良好集成，例如直接从 **Hugging Face** 获取权重和超参数。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1369026124004196443)** (120 messages🔥🔥): 

> `Aider 0.82.3, udiff-simple, gemini 2.5, Data Privacy, Vertex API` 


- **Aider 0.82.3 发布至 PyPI**：**Aider 0.82.3** 已在 PyPI 发布，并使用 `uv tool upgrade --all` 进行了升级，修复了使用 `uvx` 从主分支运行 aider 的 bug。
   - 然而，`udiff-simple` 在模型描述区域显示为聊天模式（chat mode），而不是指定的编辑格式。
- **Vertex API 上的 Gemini 2.5 Pro**：成员们报告称可以在 Vertex API 上访问新的 **Gemini 2.5 Pro**，之前的 **03-25** 模型版本会重定向到 **05-06** 版本。
   - 它在 AI Studio 上尚不可用，并且具有类似于 OpenAI 的 thinking traces。Google 在其 [开发者博客](https://developers.googleblog.com/en/gemini-2-5-pro-io-improved-coding-performance/) 中指出：*之前的迭代版本 (03-25) 现在指向最新版本 (05-06)，因此无需采取任何行动即可使用改进后的模型，且价格保持不变*。
- **Aider 中的数据隐私担忧**：讨论了在企业环境中使用 Aider 及其对数据隐私的担忧。
   - 成员建议使用具有“不共享数据策略”的 LLM，因为 **Aider** 仅与 LLM 共享代码，而不与 **Aider** 自身共享，部分成员正在使用 **Amazon Q** 作为云提供商。
- **征集 Aider 文档改进建议**：一名成员正在收集文档需求以改进 **Aider** 的工作流，特别是针对那些难以理解工作流的新用户和被推荐用户。
   - 鼓励用户在 [GitHub issue #3934](https://github.com/Aider-AI/aider/issues/393) 中添加他们的想法。
- **Gemini UI 表现优于 Sonnet？**：有一场关于 **Gemini 2.5 Pro** 与 **Sonnet 3.7** 在 UI 生成能力方面的对比讨论，一名成员指出 **Sonnet** 在 **React** 方面表现更好，但另一名成员认为 Gemini 更胜一筹。
   - 一名成员表示它*让我非常激动，等 r2 发布时我会更疯狂*。


  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1369026341898555553)** (29 条消息🔥): 

> `Aider Subtree, Lint Command, HTML representations, OpenRouter, Authentication Error` 


- **Aider 在子树中工作**：一位成员询问在大型 mono repos 的项目子文件夹中使用带有 `--subtree-only` 选项的 `aider`，并引用了 [aider 文档](https://aider.chat/docs/faq.html#can-i-use-aider-in-a-large-mono-repo)。
   - 另一位用户确认这可行。
- **Lint 命令误解**：一位成员询问为什么在以 `aider --model gemini/gemini-2.5-pro-exp-03-25 editor-model openai/gpt-4.1 --edit-format diff` 启动 `aider` 后，`/lint` 没有任何反应。
   - 一位用户回答说，*lint 仅在发现代码问题时提供反馈*，并建议检查 `/settings` 中的 `lint_cmd`。
- **LLM 返回 HTML 字符**：一位用户注意到 prompt 响应中 `->` 被表示为 `-&gt`，并想知道这是 Aider 还是 LLM 的问题。
   - 一位成员澄清说，`&gt;` 和 `&lt;` 是 **> 和 < 的 HTML 表示形式**，因此两者都不是。
- **直接使用提供商更佳**：一位成员询问关于使用 **OpenRouter** 与直接使用 **Gemini** 或 **Anthropic** 等提供商的对比。
   - 另一位成员回答说，*如果直接使用提供商，性能和成本都会更好。OpenRouter 只是让你能更轻松地测试更多模型，而无需为每个提供商创建新账户*。
- **Gemini 认证错误**：一位用户报告在包含 `/vendor` 文件夹的 Golang repos 中遇到 **litellm.AuthenticationError**，尽管已将其添加到 `.aiderignore` 中。
   - 另一位用户报告了*同样的问题*，并要求 aider 在之后进行重构以删除注释。


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1369033054911201423)** (108 条消息🔥🔥): 

> `PM2 for MCP servers, OAuth with Keycloak for MCP, MPC Server initiating communication with Claude Desktop, Memory options for Claude, Controlling Claude's tool access` 


- ****Keycloak 与 MCPs：OAuth 漫游开始****：一位成员寻求关于在 **MCP server** 前使用 **Keycloak** 实现 **OAuth** 的建议，传输层使用 `http streamable`。
   - 另一位成员推荐了一个框架，并链接到 [GitHub 上的治理 SDK](https://github.com/ithena-one/mcp-governance-sdk) 以供参考。
- ****MCPs 突破限制：服务器启动的 Claude Prompts****：一位成员询问 **MPC server** 是否可以主动发起与 **Claude desktop** 的通信，定期发送 prompt 而不是手动输入。
   - 一位成员给出了否定回答，指出虽然 *sampling 在某种程度上可以做到这一点*，但可能不是用户想要的方式，而且 Claude desktop 可能不支持 sampling。
- ****Claude 的工具箱困境：寻求选择性工具访问****：一位成员想要一种简单的方法来控制 **Claude** 可以访问哪些工具集，旨在根据当前任务仅加载相关工具。
   - 另一位成员分享了一个使用顺序思维 prompt 的变通方法，而其他人则建议限制工具数量并使用多 Agent 系统来缩小工具集选择范围。
- ****Fast Agent：窥见 MCP 的未来？****：**Fast Agent (f-a)** 是一个用于构建集成 **MCP 支持**的 Agent 的工具。
   - 一位成员将其描述为 *不是面向用户的应用程序，而是构建 Agent 的框架*，另一位成员分享了其 [MCP 状态转移技巧](https://fast-agent.ai/mcp/state_transfer/) 的链接。
- ****MCP Proxy 问题：协议版本风险****：一位成员在连接到其服务器时遇到了 **MCP proxy** 不支持最新版本规范的问题。
   - 尽管协议版本不匹配，但该 proxy 似乎通过伪造协议版本来运行，不过该成员对发布带有此类变通方法的代码表示担忧。


  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1369095700419317801)** (4 条消息): 

> `Graphlit, MCP search engine, MCP servers` 


- **由 **Graphlit** 驱动的 **McPoogle** 搜索引擎发布**：团队发布了 **McPoogle** 的预览版，这是一个由 **Graphlit** RAG 流水线驱动的搜索引擎，索引了 4000 多个 MCP 服务器和工具，访问地址为 [mcpoogle.com](https://www.mcpoogle.com/?prompt=%22Tell%20me%20about%20the%20Graphlit%20MCP%20Server%22)。
   - 它支持搜索并回答关于 **MCP servers** 和工具的问题，并邀请用户对其性能提供反馈。
- ****MCP-CLI** 现已支持 OAuth**：[MCP-CLI](https://github.com/wong2/mcp-cli) 现在支持 **OAuth**，增强了其易用性和安全性。
   - 一段 [Loom 视频](https://www.loom.com/share/d2a00956cdb248e5adbc9c31538c7892) 展示了其新功能。
- ****AWS EC2** 远程 MCP 服务器指南**：分享了一篇题为 *Build and Deploy Remote MCP Servers to AWS EC2* 的 Medium 文章，提供了在 [AWS EC2](https://medium.com/@tadeodonegana/build-and-deploy-remote-mcp-servers-to-aws-ec2-5888514892c4) 上部署 **MCP servers** 的指南。
   - 这对于想要将 MCP 服务器迁移到云端的用户非常有用。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1369104161970454642)** (45 条消息🔥): 

> `api-inference.huggingface.co, Object Tracking models, DOI deletion, Summarising caselaw, Data Parallelism vs Model Parallelism` 


- **api-inference.huggingface.co 已停用**：端点 **api-inference.huggingface.co** 现已弃用，用户应改用新端点。
   - 目前尚不清楚是否有弃用通知，且 **LangChainjs** 仍在使用旧端点。
- **目标追踪模型缺乏推理提供商**：一位成员询问为什么没有任何 **Object Tracking models** 的 **Inference Providers**，并指出 Facebook 的 **DETR model** 的提供商已被移除。
   - 他们指出几乎所有模型都处于相同状态。
- **DOI 无法删除**：一位成员寻求帮助，想要删除错误附加到测试仓库的 DOI。
   - 建议删除 DOI 可能需要通过电子邮件联系支持团队，并链接到了[此讨论](https://discuss.huggingface.co/t/change-authors-of-citation-with-doi/145654/4)。
- **LLM 总结法律判例**：一位成员正在进行一个使用标签总结 **caselaw**（法律判例）的项目，需要帮助选择模型。
   - 建议查看 [Hugging Face LLM leaderboard](https://huggingface.co/spaces/fr-gouv-coordination-ia/llm_leaderboard_fr#/) 并在其特定数据集上尝试有潜力的模型，同时指出数据集清洗在训练中的重要性。
- **数据并行 vs 模型并行**：一位成员询问，如果中型模型可以使用数据并行进行训练，那么切换到 **model parallelism** 的意义何在。
   - 他们还询问如果模型未包含在官方的 **Diffusers** 或 **Transformers** 版本中，是否需要编写 **handler.py**，并链接到了[自定义处理程序文档](https://huggingface.co/docs/inference-endpoints/guides/custom_handler)。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1369033505811599423)** (9 条消息🔥): 

> `AI Study Group GitHub Repo, List of AI Papers Plan, Discord Usage` 


- **新的 AI 学习小组仓库出现！**：一位在过去五年中从游戏行业转行的成员整理了 AI 资源，并创建了 [AI-Study-Group GitHub repo](https://github.com/ArturoNereu/AI-Study-Group)。
   - 该成员欢迎贡献有用的论文、工具或被低估的宝藏资源。
- **晦涩的 AI 论文引发“头疼”讨论！**：一位成员分享了一个攻克一系列晦涩 AI 论文的计划，引发了一些幽默的反应，如[此处](https://cdn.discordapp.com/attachments/898619964095860757/1369046722847834143/IMG_3863.png?ex=681bc11e&is=681a6f9e&hm=c2d311c4ef37480727774cc6d54707492a73ddfb97c18c9dc2e78bd680e1ead4)附图所示。
   - 其他成员开玩笑说这个*计划让他们头疼*，还有人质疑该计划是否过于雄心勃勃。
- **成员开玩笑说在学习使用 Discord！**：在另一位成员开玩笑说 AI 阅读清单让他头疼后，其中一位成员说“*今天我在学习如何使用 Discord*”。
   - 未提供进一步背景。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1369094581546782720)** (8 messages🔥): 

> `Huggingface Desktop App, Dank Leaks, Flux-Pro-Unlimited AI image generator, candle-holder` 


- **HF 上传桌面应用兴起**：一名成员正在开发一款用于 **Hugging Face 上传**和**模型管理**的桌面应用，该应用借鉴了现有的 Jupyter/Colab notebooks，并正在寻求合作者来优化目前较为凌乱的代码。
   - 该 [项目](https://github.com/Ktiseos-Nyx/Huggingface-Desktop/tree/main) 的上传器可能因导入问题暂时无法运行，但欢迎贡献代码。
- **Dank Leaks 应用亮相**：一名成员展示了 "Dank Leaks"，这是一个使用 **C-17/notcurses** 编写的开发中应用，并通过屏幕录制演示了其功能。
   - 该成员分享了该应用的 [屏幕录制](https://cdn.discordapp.com/attachments/897390720388825149/1369235033193451580/Screen_Recording_2025-05-06_at_09.04.26.mov?ex=681bc7bf&is=681a763f&hm=017faa7cad446bd78255f6074e626df19fac3ce34a21702ae2a932a0e9b8e9d5&)。
- **免费无限次 Flux-Pro AI 图像生成器出现**：一名成员分享了 **Flux-Pro-Unlimited** 的链接，这是一个在 Hugging Face Spaces 上提供的*部分去审查*的 AI 图像生成器，用于研究目的。
   - 访问地址为 [https://huggingface.co/spaces/NihalGazi/FLUX-Pro-Unlimited-](https://huggingface.co/spaces/NihalGazi/FLUX-Pro-Unlimited-)，被描述为“虽无新意，但却是一个实用的服务（无 ZeroGPU）”。
- **提及 Candle Holder 项目**：针对 Flux-Pro AI 图像生成器，一名成员建议关注 [candle-holder](https://github.com/gabrielmbmb/candle-holder)。
   - 未提供关于该项目为何有用的额外背景信息。


  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1369240477395587102)** (1 messages): 

> `HiDream LoRAs, Quantization Support, Memory Savings` 


- **HiDream LoRAs 获得量化支持**：通过 `bitsandbytes` 实现的量化支持已应用于 **HiDream LoRAs** 训练，有望显著节省内存。
   - 详见 [Diffusers 文档](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_hidream.md#using-quantization)。
- **训练内存占用大幅缩减**：在 **HiDream LoRA** 训练期间启用量化可大幅减少内存使用，在*设备分配（device placement）*后从 **57.64 GB** 降至 **34.60 GB**。
   - 据报告，使用量化后，*反向传播（backward）*后的内存占用为 **36.90 GB**，而未使用时为 **59.93 GB**。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1369225808408739902)** (13 messages🔥): 

> `GPU memory considerations, Emotion Classification Models, FullyShardedDataParallelPlugin Error` 


- **GPU 显存适配 bfloat16 模型**：一名成员确认 **6GB GPU** 应该足以容纳该模型，建议使用 **bfloat16** 并确保其不被加载用于训练。
   - 另一名成员确认他们*刚刚尝试了 bfloat16*。
- **情感模型表现不佳**：用户发现 `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` 模型对语句 *Looking for friends Anyone in sharjah* 的情感分类有误。
   - 该模型错误地给 **愤怒 (anger)** (0.995)、**恼火 (annoyance)** (0.989)、**失望 (disappointment)** (0.978) 和 **悲伤 (sadness)** (0.978) 打出了高分。
- **Distilroberta 在情感检测中表现出色**：`j-hartmann/emotion-english-distilroberta-base` 模型对特定文本给出了更准确的悲伤评分。
   - 然而，对于文本 *Cute*，它错误地将其情感标记为 **厌恶 (disgust)**，分数为 0.8765。
- **修复 Accelerate 的 FSDPPlugin**：一名用户分享了使用 `accelerate` 的 `FullyShardedDataParallelPlugin` 代码并报告了一个错误。
   - 代码包含 `fsdp_version=2` 的设置，以及将状态和优化器状态字典卸载（offload）到 CPU 的配置。


  

---

### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1369062492025782293)** (3 messages): 

> `HF API Limits, GAIA files evaluation, Gemma3 vs Qwen` 


- **在 Hugging Face 上触及免费 API 限制**：一名成员提到在 **HfApiModel** 上达到了免费额度限制。
   - 他们被提示要么向其 **HF account** 充值，要么选择另一个已有额度的提供商。
- **访问 GAIA 文件进行评估**：一名成员询问在评估过程中如何访问 **GAIA files**，并提到他们已经将文件下载到电脑进行本地测试。
- **Qwen 在工具支持方面优于 Gemma3**：一名成员分享了最初使用 **Gemma3** 后切换到 **Qwen** 的经验。
   - 他们指出 **Qwen** 更好，因为它支持 **tool calling**，而 **Gemma3** 不支持，并且他们即使使用 **Gemma3:4b** 也能完成作业。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1369029705981759639)** (12 messages🔥): 

> `GAIA Questions, Agent UI Timeouts, Final Agent Build, Frameworks, Final Challenge Solution` 


- **GAIA 问题运行时间过长**：成员们报告称 **GAIA question runs** 耗时非常长（**至少一小时**）且成本很高（超过 $15）。
   - 一位用户报告花费了 *$5+* 才答对 *5 道题*，每次运行大约耗时 **30 分钟**，甚至有人在运行一小时后仅获得了 **20%** 的正确率。
- **长时间运行时 Agent UI 超时**：一名成员在运行其 **agent** 时遇到 **UI timeouts/errors**，导致无法查看问题结果并进行修改。
   - 他们在浏览器控制台中看到了 `net::ERR_NETWORK_IO_SUSPENDED` 错误，并询问这是否是 **Gradio issue**。
- **最终 Agent 构建需要指导**：一名成员正在寻求关于**最终 agent 构建**要求的澄清，特别是要构建哪种类型的 **agent** 以及哪些因素被视为通过标准。
   - 他们还询问是否会将额外的框架（如 **OpenAI agents SDK**）添加到课程中。
- **最终挑战解决方案博客文章**：一名成员分享了一篇[博客文章](https://guillaume-fradet.com/posts/making-ai-think-and-act-my-approach-to-the-hugging-face-ai-agents-challenge/)，详细介绍了他们应对**最终挑战**的解决方案，实现了 **60%** 的成功率。
   - 作者似乎是 **Guillaume Fradet**。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1369102786410713201)** (71 messages🔥🔥): 

> `M4 Macbook Pro vs RTX 5070Ti for LLM inference, Diffusion Models as Evolutionary Algorithms, Search vs Optimization, AI-assisted Academic Articles and Patents, Claude Code with Gemini` 


- **面向 LLM 开发者的 M4 Macbook Pro vs RTX 5070Ti 对决**：一名成员正在为本地 **LLM inference** 和常规开发在 **M4 Macbook Pro** 和 **RTX 5070Ti Linux 笔记本**之间做选择；另一名成员指出，使用 **M4 Macbook Pro 48 GB** 时，*ollama Qwen 2.5-coder* 的 **token** 生成速度非常不错且流畅。
   - 第三名成员建议使用 **Mac** 甚至 **Mac Studio/Mini**，并建议如果需要可以廉价租用 **GPU**。
- **Diffusion Models 演变为 Evolutionary Algorithms**：一名成员引用了论文 [Diffusion Models as Evolutionary Algorithms](https://arxiv.org/abs/2410.02543)，解释了它如何揭示 **diffusion models** 可以被视为 **evolutionary algorithms**。
   - 该成员澄清说 *denoising steps 对应于模型细化的连续生成*，这桥接了生成式建模的两个主要流派，并与其提出的生成器范式相一致。
- **聊天中关于 Search 与 Optimization 的争论**：围绕 **search** 和 **optimization** 之间的关系展开了辩论，一名成员认为 *optimization 是 search 的子集*。
   - 反方观点认为，虽然在理论上 **search** 可以简化为 **optimization**，但其过程、假设和工具链有显著不同。
- **AI Agents：学术文章作者崛起**：一名成员正在开发用于撰写专利和学术文章的 **agents**，并澄清说 *这些 agents 在我的虚拟研究实验室工作，没人会看到它们*。
   - 另一名成员分享了 [AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2) 的链接，第一名成员澄清说他开发的 **agents** 是 **dynamic** 的，且属于 **society of minds**。
- **Claude Code 搭配 Gemini？**：一名成员询问是否有类似 **Claude Code** 但使用 **Gemini** 的最佳工具，或者是被修改为可以使用任何模型的 **OpenAI's Codex**。
   - 该问题暂无回复。


  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/)** (1 条消息): 

k_nearest_neighbor: 我今天没法做论文分享了，但如果有人想做的话请随意。
  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1369030429134426243)** (13 条消息🔥): 

> `Em Dashes, OAI, US Gov, Chinese Models, Deepseek vs OAI, Sam Altman` 


- **连接号 (Em Dash) 的用法在演变！**：**em dash** 的正确用法正受到 **chatGPT** 等 **AI models** 的影响，这些模型从人类数据中学习，并可能在无意中推广了它的使用。
   - 随着 AI 生成内容变得越来越普遍，人们开始思考人类是否会开始更频繁地采用 em dash，从而影响写作风格。
- **OAI 针对中国模型的政府策略？**：一些成员推测 **OpenAI** 可能会游说**美国政府**禁止**中国 AI 模型**，理由是已有开源替代方案。
   - 论点是，既然存在开源选项，就“不需要受 CCP 控制的模型”，这引发了对竞争和控制的担忧。
- **Deepseek 的 Post-Training 实力胜出！**：一位成员认为，即便 **OpenAI** 的基础模型最初更好，**Deepseek** 也可能凭借卓越的 **post-training** 超越 **OpenAI**，并参考了 [Hugging Face 上的 Microsoft MAI-DS-R1](https://huggingface.co/microsoft/MAI-DS-R1)。
   - 这突显了 post-training 技术在实现 state-of-the-art 模型性能方面的关键作用，有可能超越初始架构带来的优势。
- **Altman 的把戏落空：非营利组织将保留控制权？**：参考一篇 [2025 年的 CNBC 文章](https://www.cnbc.com/amp/2025/05/05/openai-says-nonprofit-retain-control-of-company-bowing-to-pressure.html)，一位成员建议，如果 **Sam Altman** 试图改变 **OpenAI** 的**非营利结构**，他可能会面临法律后果。
   - 这暗示了潜在的权力斗争，以及维持原始治理结构以确保道德和负责任的 AI 开发的重要性。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1369087963429867640)** (8 条消息🔥): 

> `Podcast Length Discrepancies, Inserting Instructions, Audio Overview Experiences, Mind Map Generation` 


- **播客长度因语言而异**：一位用户报告说，他们的播客英文版为 **40 分钟**，但在其他语言中被限制在 **14 分钟**，目前尚未提供直接的解决方案。
   - 该用户的问题突显了播客创建工具在语言处理或内容生成方面可能存在的限制或 bug。
- **指令插入咨询**：一位用户询问如何插入指令，另一位用户回复说可以查看“生成 (generate)”按钮下方的“自定义 (customize)”按钮。
   - 还有说明指出，免费计划可能无法使用自定义功能，并参考了一张展示用户界面中该选项的附图。
- **音频概览成功案例**：一位用户分享说，他们的音频概览没有重复内容、静态噪音、幻听或编造，且概览很好地遵循了指令。
   - 该用户将成功归功于他们的 prompt 和源材料，认为某些来源在遵循指令方面具有独特的优势。
- **思维导图生成技术**：一位用户详细介绍了他们使用自定义 prompt 从源材料生成思维导图的过程。
   - 他们使用以下 prompt 重新生成：*Create a mind map from the sources. List topics as central ideas, main branches, and sub-branches. Output it in Markdown format.*，然后将输出提供给 [markmap.js.org](https://markmap.js.org/) 并保存为“交互式 HTML”。


  

---

### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1369028757997748304)** (70 条消息🔥🔥): 

> `NotebookLM 音频转录, Gemini Flash 2.5 确认, 粤语支持, NotebookLM 的 Gemini 版本, NotebookLM 中的交互模式` 


- **在 NotebookLM 中生成音频转录**：一位用户寻求为 NotebookLM 音频生成类似于 Google Meets 的转录或字幕的方法，并建议通过屏幕共享到 Google Meets 来生成录音和转录作为一种变通方案。
   - 该用户称赞 NotebookLM 相比 ChatGPT 提供了更详细和完整的回答，尤其是在对文档进行事实核查（fact-checking）方面，并认为音频概览（audio overview）功能与真实的播客（podcast）惊人地相似。
- **确认 NotebookLM 中使用的 Gemini Flash 2.5**：用户讨论了如何确认他们在 NotebookLM 中使用的是 **Gemini Flash 2.5**，一位用户指出某篇特定帖子可以作为确认。
   - 一些用户对 NotebookLM 的推理能力表示怀疑，认为它可能使用的是旧版本（2.0），而其他用户则指出 Gemini 2.5 Pro 在 AI Studio 中表现出色。
- **NotebookLM 即将支持粤语**：一位用户询问 NotebookLM 是否支持**粤语**（Cantonese）。
   - 团队成员回复称他们正在*努力开发中*。
- **由于域名限制导致网页导入失败**：一位用户报告在尝试将 NotebookLM 来源添加到项目时，遇到了显示“*由于域名限制，无法导入此网页*”的错误消息，并发布了[截图](https://ibb.co/My2JzWHp)。
   - 域名限制会阻止将某些网页作为 NotebookLM 来源导入。
- **对话风格故障排除**：一位用户询问如何让 NotebookLM 的对话风格功能生效，称无论提供什么指令，AI 的表现始终如一。
   - 一位成员建议上传自定义指令或提示词文档作为来源，以影响 AI 的行为，并建议尝试实时模式（live mode）进行插话。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1369031405899874334)** (63 条消息🔥🔥): 

> `Dwarkesh 自动化公司文章, 营收数据, Exa 博客文章, OpenAI 收购 Windsurf, Gemini 2.5 Pro Elo 评分提升` 


- **Dwarkesh 详述全自动化公司**：[Dwarkesh 的文章和视频](https://www.dwarkesh.com/p/ai-firm)大量借鉴了 [Gwern 的 Backstop](https://gwern.net/backstop)，并讨论了 **Agentic DAOs** 的潜力。
- **AI 初创公司公布营收数据**：一位成员分享了[营收数据](https://x.com/tanayj/status/1919489023602786737)，并指出在**高生活成本城市**（HCOL cities）中，与 **GPU 时间**、**工程师薪资**和**高管薪酬**相关的高昂成本。
- **Exa 重新发布 BM25 优化博客文章**：**Exa** 回归 X 平台，并发布了一篇关于 [BM25 优化](https://exa.ai/blog/bm25-optimization)的新博客文章。
- **OpenAI 达成 30 亿美元收购 Windsurf 的交易**：据 [Bloomberg](https://www.bloomberg.com/news/articles/2025-05-06/openai-reaches-agreement-to-buy-startup-windsurf-for-3-billion) 报道，**OpenAI** 将以 **30 亿美元**收购 **Windsurf**。
- **Gemini 2.5 Pro 的 ELO 评分大幅提升**：根据[这条推文](https://x.com/scaling01/status/1919771796334616759)和 [Google DeepMind 的公告](https://x.com/googledeepmind/status/1919770265711419826)，新的 **Gemini 2.5 Pro** 更新带来了巨大的 ELO 评分提升。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1369026617384505536)** (38 messages🔥): 

> `OpenAI Public Benefit Corp, Flights to the US, RL Environments Hackathon, Fine tuning a base model LLM, M4 Macbook Pro vs RTX 5070Ti Linux laptop` 


- ****OpenAI 考虑转向公共利益公司（Public Benefit Corp）架构****：鉴于 **OpenAI** 筹集了 **$8B**，有推测认为他们相信目前的产品支出足以产生至少 **1.3x+** 的现金回报，详见 [WSJ 文章](https://www.wsj.com/tech/ai/openai-to-become-public-benefit-corporation-9e7896e0?st=ifeJpvreal)。
- ****赴美机票价格减半****：由于对拘留和设备检查的担忧，目前飞往美国的航班价格约为平时的一半。
- ****即将举行的 RL 环境 Hackathon****：Nous Research 正在举办一场 [RL Environments Hackathon](https://cerebralvalley.ai/e/nous-research-rl-environments-hackathon-9be3062ade)。
- ****基础模型微调能否超越聊天助手？****：一位成员询问是否可以将基础模型 LLM 微调为指令遵循聊天助手以外的其他形态。
   - 回复指出这是可能的，例如控制机器人运动和股票预测，尽管*获取数据很困难*。
- ****本地 LLM 推理：M4 Macbook Pro vs RTX 5070Ti Linux 笔记本****：一位用户正在纠结选择 **M4 Macbook Pro** 还是 **RTX 5070Ti Linux 笔记本** 进行本地 LLM 推理和常规开发。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1369033904186724476)** (5 messages): 

> `AnySphere, AGI Agents` 


- **AnySphere 巨额 $900M 融资需求**：一位成员质疑为什么 Cursor 的开发商 **AnySphere** 在只有约 **100 人** 的情况下需要 **$900M**。
   - 另一位成员开玩笑说，他们可能实际上需要 **1000 人**。
- **AGI Agent：推迟了？**：一位成员直截了当地询问 **AGI Agent** 是否被推迟了。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1369043843739029607)** (27 messages🔥): 

> `Page Faults, Disk Swapping, Torch, Transformers, vLLM` 


- **用户在导入库时面临分页问题**：尽管使用了 **conda**、**venv** 和 **uv** 进行包管理，一位用户在导入 **torch**、**transformers** 和 **vLLM** 等标准 Python 库时，仍遇到了严重的缺页中断（Page Faults）和磁盘交换（Disk Swapping）。
   - 他们测试了在不同区域安装库、更改环境变量，并确认同一台机器上的其他用户没有遇到此问题，这可能表明是特定于用户的配置错误或损坏。
- **系统管理员被罕见的文件系统问题难倒**：在尝试调试并联系系统管理员后，导致该用户系统行为异常（涉及过度磁盘交换）的根本原因仍不明确。
   - 尽管该用户与其他没有类似问题的实验室用户使用相同的磁盘，但问题依然存在，这暗示可能存在损坏的表或 UID 分配问题。
- **怀疑硬件和内存完整性**：一位成员建议问题可能源于内存抖动（Memory Thrashing）或坏位（Broken Bits），建议进行内存测试以排除硬件故障。
   - 另一位成员提出问题可能与输入/输出（I/O）限制或配置错误的 pagemaps 有关，这意味着系统不断发生缺页中断，阻碍了调试工作。
- **用户寻求关于 Transformer 限制的深入见解**：一位 AI 专业的学生正在寻求关于 **RAG/CAG**、Prompt Engineering、Fine-tuning 以及 Transformer 上下文窗口限制的见解，以及像 **EM-LLMs** 这样以内存为中心的认知架构。
   - 他们正将问题提交至 Hugging Face 的 Discord，并表示他们的教授无法完全解答这些领域的问题。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1369034358849142784)** (3 messages): 

> `Data vs Model Parallelism, arXiv:2305.18153 Citations, Anthropic Work` 


- **Data Parallelism vs Model Parallelism**: 一位成员询问，如果可以使用 **Data Parallelism** 花费极长时间训练一个中型模型，是否有必要切换到 **Model Parallelism**。
   - 上下文暗示了一种时间不是限制因素的场景，但正在考虑 **Data Parallelism** 和 **Model Parallelism** 之间的选择。
- **引用 arXiv:2305.18153 的论文**: 一位成员建议查看引用了 **arXiv:2305.18153** 这篇论文的研究。
   - 这一建议意味着在引用该特定工作的论文中可能会发现相关的研究或进展。
- **Anthropic 的相关工作**: 一位成员指出 **Anthropic** 在该领域存在相关工作。
   - 他们没有核对作者，但暗示 **Anthropic** 对所讨论的主题有相关的贡献。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1369032973596491890)** (9 messages🔥): 

> `Circuit Identification, Anthropic Transformer Circuits, Monosemanticity, Interpretability Tooling Challenges, TransformerLens Limitations` 


- **通过 Circuit Identification 探索模型行为**: 一位成员建议研究 **Circuit Identification** 以理解模型行为，并指出宏大的方法在准确表示模型内部过程方面往往存在局限性。
   - 他们建议探索关于 **grokking**、**BIG-Bench** 和 **content bias reasoning** 的研究论文，并将 **Anthropic Transformer Circuits** 和 **Towards Monosemanticity** 作为起点。
- **解决可解释性研究中的工具鸿沟**: 一位成员询问了可解释性和经验对齐（empirical alignment）工作中的关键 **tooling challenges**，寻求现实世界的例子，例如大规模从 **Pytorch** 提取激活（activation extraction）时的性能问题。
   - 另一位成员指出，一些可解释性库如 **TransformerLens** 和 **nnsight** 在某些方面功能仍然有限（例如 **TransformerLens** 仅支持有限的预训练模型列表）。
- **应对 SAE 训练和 LLM 调用中的挑战**: 一位成员提到他们的团队决定将激活保存到磁盘，并在单独的阶段训练 **SAE**，这导致了处理“大数据”的挑战，尽管 **SAE** 训练本身相对较快。
   - 他们还强调了自动可解释性（auto-interpretability）所需的大量 **LLM calls** 是另一个经验性的工具挑战。


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1369235449419403345)** (1 messages): 

> `Multimodal VLMs, Community Research Hub, Weekly Updates` 


- **VLMs 研究枢纽启动！**: 一个由社区驱动的 **Multimodal Researchers** 枢纽已经创建并正在积极维护。
   - 该枢纽的创建者欢迎贡献、建议和反馈，并每周更新枢纽内容：[All-Things-Multimodal](https://github.com/thubZ09/All-Things-Multimodal.git)。
- **欢迎社区贡献！**: 该枢纽向社区开放贡献，鼓励研究人员分享他们的发现和资源。
   - 非常欢迎提出建议和反馈，以确保该枢纽保持全面和最新。


  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1369053259746119762)** (19 messages🔥): 

> `Codegen for new models, Reducing engineering time for new models, Tokenizer Support, HF Transformers Adapter, Qwen3 Support` 


- ****考虑使用 Codegen 减少样板代码****：成员们讨论了在向 Torchtune 添加新模型时使用 **codegen** 来减少样板代码（boilerplate），灵感来自 **Unsloth** 等库中更快的模型支持方式。
   - 有人对依赖 tracing 关键字表示担忧，并强调用户理解底层代码的重要性，建议将编写良好的教程作为大规模 **codegen** 的替代方案。
- ****缩短新 SoTA 模型的工程化时间是首要任务****：主要目标是缩短在 Torchtune 中使用新 **SoTA 模型**所需的**工程时间**，同时保持**合理的性能**。
   - 有建议提出通过识别样板代码与困难环节来应对挑战，重点是简化后者（如 **tokenizer** 支持），然后再自动化样板代码的生成。
- ****Tokenizer 支持被认为是非琐碎的任务****：**Tokenizer 支持**被确定为一项具有挑战性的任务，需要通用的解决方案；而更繁琐的任务可以在考虑 **codegen** 之前通过脚本自动化完成。
   - 讨论中提到了利用 **HF (Hugging Face) configurations** 来处理 tokenizer，并使用 **HF** 模型生成等价性检查（parity check）数值。
- ****构想通过 HF Transformers Adapter 实现更快的模型支持****：一种建议的方法是创建一个通用的 **HF adapter**，用于加载 **HF** 模型并允许映射以适配不同的训练特性。
   - 这将有助于更快地添加新模型并实现 **finetuning**，随后可以选择实现具有完整功能支持的“真实”版本。
- ****Qwen3 热度：Codegen 可轻松支持模型版本更新****：有人提到，对于像 **Qwen** 这样的模型家族，新版本通常需要大量新的样板代码，而这些代码很大程度上可以通过 **codegen** 添加。
   - 讨论强调了在 **Qwen3** 发布时迅速提供支持的营销优势，即使初始版本存在限制，也可以在全功能实现就绪前抢占先机。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1369404139787911299)** (3 messages): 

> `Modular Puzzles on Macbook, Apple Silicon GPUs, NVIDIA GPU architectures` 


- **在 Macbook 上运行 Modular Puzzles？**：一名成员询问是否可以在 Macbook 上运行 **Modular Puzzles**。
   - 另一名成员澄清目前还不能直接在 **Apple Silicon GPU** 上运行，建议通过远程连接到**挂载了 GPU 的云端实例**作为变通方案。
- **NVIDIA GPU 来救场！**：成员们明确了目前适用于 **Mojo GPU** 编程的 **NVIDIA GPU 架构**包括 Turing, Ampere, Hopper 和 Blackwell（**RTX 20XX - 50XX 系列**）。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1369053480874016859)** (12 messages🔥): 

> `Blogging Platforms, Ownership Semantics in Mojo, Mojo Getting Started Guide Errors, Comptime Try-Except Handling` 


- **Blot.im 是首选博客平台**：一名成员建议使用 [Blot.im](https://blot.im/) 作为博客平台，它支持 **markdown**，但这是一项付费服务。
- **使用 Unsafe Pointers 进行 Struct 修改**：一名成员询问如何使用 **unsafe pointers** 来修改 struct，特别是询问除了全局 `var` 之外的替代方案。
- **入门指南遇到的问题**：一名用户报告了 [Mojo 入门指南](https://docs.modular.com/mojo/manual/get-started/)中最后一个示例的错误。
- **Comptime Try-Except 的困扰**：一名成员询问是否可以在 `comptime` 执行的计算中使用 `try ... except ...`。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1369058971092254720)** (2 messages): 

> `MCP Hackathon, Agent Communication, Deep Research Agent` 


- **LlamaIndex 赞助特拉维夫 MCP Hackathon**：LlamaIndex 正在赞助由 @aitinkerers 在特拉维夫举办的 **Big MCP Hackathon**，重点是构建支持 **Agent** 间通信和实验的 **MCP 驱动应用**，详见[此推文](https://twitter.com/llama_index/status/1919499875332587579)。
- **LlamaIndex 教授如何构建 Deep Research Agent**：LlamaIndex 推出了一项研讨会教程，涵盖了如何使用 **AgentWorkflow** 从头开始构建用于 **deep research** 的**多 Agent 系统**，详见[此推文](https://twitter.com/llama_index/status/1919856812893077565)。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1369220679282720770)** (6 messages): 

> `Property Graph Indexes, LlamaIndex GraphRAG, LangChain GraphRAG, Vector Database Storage` 


- ****Property Graph 索引探索****：一名成员询问了 Property Graph 索引的使用方法，并分享了他们在 [这个 notebook](https://github.com/tituslhy/shiny-engine/blob/main/notebooks/hybrid_property_graph.ipynb) 中参考 LlamaIndex 文档实现的版本。
   - 该成员发现索引有时无法回答问题，因为它无法从图或向量数据库中检索到节点，从而引发了关于向量数据库如何存储节点的问题。
- ****LlamaIndex 与 LangChain 图生成对比****：同一位成员观察到，使用 LlamaIndex 的 Property Graph 索引代码生成的图要密集得多（可视化见 [此处](https://github.com/tituslhy/shiny-engine/blob/main/images/llamaindex_neo4j.png)），而 LangChain 生成的图（可视化见 [此处](https://github.com/tituslhy/shiny-engine/blob/main/images/groq_kg.png)）则相对稀疏。
   - 该成员注意到一个 [LangChain notebook](https://github.com/tituslhy/shiny-engine/blob/main/notebooks/large_llm_knowledge_graph.ipynb) 在向量数据库的节点中对文本进行了编码。
- ****GraphRAG 中的向量数据库存储策略****：该成员认为，向量数据库存储的是图中的每个节点，而不是普通 RAG 中的每个文本块（text chunk）。
   - 图的密度很可能取决于所使用的默认 Prompt。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1369291452739227780)** (2 messages): 

> `M4 Macbook vs RTX 5070Ti for local LLM, tinygrad discord rules` 


- **Macbook Pro M4 还是 RTX 5070Ti？**：一位成员询问 **Macbook Pro M4** 和 **RTX 5070Ti Linux 笔记本** 哪个更适合本地 LLM 推理和通用开发。
   - 该成员提到自己是长期的 **Linux 桌面用户**，并听说过很多关于 **M 系列 Mac** 的好评。
- **阅读规则！**：George Hotz 提醒用户阅读 Discord 规则，并声明这里是讨论 **tinygrad** 和 **tinygrad 开发** 的地方。
   - 该用户此前询问了关于购买新机器的选择，即在 **M4 Macbook Pro** 和 **RTX 5070Ti Linux 笔记本** 之间做决定。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1369342537940209806)** (3 messages): 

> `Bounty Picking, Rockship Device` 


- **Bounty 猎人询问领取流程**：成员们寻求关于如何从 [Google Sheets Bounty 列表](https://docs.google.com/spreadsheets/d/1WKHbT-7KOgjEawq5h5Ic1qUWzpfAzuD_J06N1JwOCGs/edit?gid=0#gid=0) 中领取任务的指导，以及其他地方是否有更详细的描述。
   - 他们还想知道是否应该在开始前将 Bounty 标记为 *"已占用 (taken)"*，以避免多人同时处理同一任务。
- **WIP PR 锁定 Bounty**：一名成员建议，需要创建一个 **WIP PR** 来锁定 Bounty，以防止多人重复劳动。
   - 他们还推测，Bounty 猎人应该 *"潜水一段时间并进行钻研，以便充分理解现有描述所表达的内容"*。
- **Rockship 设备 Bounty 澄清**：关于 **新 Rockship 设备** 的 Bounty，成员们询问应该使用 SDK 还是开源驱动程序。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1369416949817413894)** (1 messages): 

> `Auth0 Workshop, AI Agent Security, Entrepreneurship Track Prizes` 


- **明天 Auth0 工作坊：保护你的 Agent！**：明天 (5/7) **上午 10:00 PT** 将举行一场 **Auth0** 特别工作坊，学习如何使用专业的身份验证解决方案保护 AI Agent；请关注 [YouTube 直播](https://www.youtube.com/watch?v=wB29IJ3AEjY)。
   - 工作坊将涵盖在 LLM 驱动的 Agent 中实现强大的身份验证、将 **Auth0** 服务与 AgentX 集成，以及处理 AI 系统特有的安全注意事项。
- **面向创业 Agent 的 Auth0 奖金**：**Auth0** 为创业赛道（Entrepreneurship Track）赞助了特别奖项，第一名奖金高达 **$5,000**，第二名 **$3,000**，第三名 **$2,000**。
   - 这些奖项颁发给成功将 **Auth0.ai** 集成到其项目中的团队。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1369168268669227058)** (2 messages): 

> `HuggingFace Credits, Quiz Scores` 


- **成员期待 HuggingFace Credit 的发放**：一名成员报告称，在被选中接收 **HuggingFace credits** 后，这些额度尚未添加到其组织账户中。
   - 他们提到已经填写了必要的表格，表明正在等待额度分配。
- **Quiz 分数仍未公布**：一名成员注意到最后 **两次测验（Quiz 11 和 Quiz 12）** 的分数仍然无法查询。
   - 他们似乎正在等待这些分数的发布，以便评估自己的表现。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1369354927842922619)** (1 messages): 

> `LLMs and Conditional Statements, Formal Methods in LLMs, Representing Conditions for LLMs` 


- **LLM 处理类似投票的条件语句 (Conditional Statements)**：一名成员询问 **LLM** 如何记忆 **conditional statements** 并产生良好的结果，并以投票资格（*18 岁、公民、选民证*）为例。
   - 他们表示自课程开始以来，一直苦恼于如何为 **LLM** 表示这些条件。
- **使用 Formal Methods 的 LLM**：同一位成员想知道 **formal methods** 在使 **LLM** 有效处理条件逻辑方面发挥了什么作用。
   - 他们寻求关于如何应用这些方法来为 **LLM** 表示复杂条件（特别是在长期记忆的背景下）的见解。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-readings-discussion](https://discord.com/channels/1280234300012494859/1282735578886181036/1369354605183373432)** (1 messages): 

> `LLM Reasoning, LLM Formal Methods, LLM Knowledge Representation` 


- **LLM 从容处理条件语句 (Conditionals)**：一名成员询问 **LLM** 如何在不显式使用 formal methods 的情况下记忆 **conditional statements** 并产生良好的结果。
   - 他们举了投票要求的例子：*"一个人应该年满 18 岁，应该是公民，应该有选民证"*，并思考如何在 LLM 中表示这种知识。
- **LLM 与投票资格**：讨论集中在 **LLM** 如何有效地管理和应用 **conditional logic**，特别是在投票资格要求等场景中。
   - 该示例涉及年龄、公民身份和选民证等特定条件，引发了关于使 LLM 能够准确处理和应用这些规则的底层机制的疑问。


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1369248463317110876)** (5 messages): 

> `Cohere-AI npm, Aya vision` 


- **Cohere-AI npm 包支持 Aya Vision**：一名成员询问 **Cohere-AI npm package** 是否支持 **Aya Vision**，另一名成员确认支持并分享了代码片段。
   - 代码片段展示了如何使用 **CohereClientV2** 类与 **c4ai-aya-vision-8b** 模型进行对话，发送包含文本和图像 URL 的消息。
- **用户实现并集成 Cohere-AI**：在收到代码片段和确认后，一名成员表示他已经为 expedition 实现并集成了 **Cohere-AI**。
   - 未提供关于 "expedition" 指代的更多细节。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1369068297491976354)** (3 messages): 

> `Claude System Prompt, Chat Template Generation with Python, GPT4All Integration` 


- **Claude 的 System Prompt 曝光！**：一名成员分享了一个 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1kfkg29/claude_full_system_prompt_with_all_tools_is_now/) 的链接，其中包含带有所有工具的 **Claude** 完整 system prompt。
   - 该帖子立即引起了兴趣和讨论，成员们渴望探索其潜在应用。
- **Chat Template 生成需要 Python 帮助！**：一名成员请求协助使用 **Python** 生成 **chat template**，并表示自己是该领域的新手。
   - 尽管安装了 *transformers* 模块，但他们遇到了语法错误，并寻求解决问题的指导。
- **Claude Prompt 能否集成到 GPT4All？**：一名成员询问如何将泄露的 **Claude** system prompt 集成到 **GPT4All** 中。
   - 这一查询突显了社区对于利用来自不同 **LLM** 系统的见解来增强 **GPT4All** 功能的兴趣。


  

---

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1369121502733008896)** (1 messages): 

> `MLOps, LLMOps, AI Project Lifecycle, Data Phoenix` 


- **通过 MLOps 和 LLMOps 驯服 AI 项目**：一场关于 **MLOps** 和 **LLMOps** 的网络研讨会定于 PDT 时间 5 月 28 日星期三上午 10:00 举行，重点关注建立可靠的流程并加速 AI 解决方案的交付，以消除混乱。
   - 该研讨会将涵盖 **ML** 和 **LLM** 项目的生命周期、构成 **MLOps/LLMOps** 基础的工具、角色和实践，以及 **MLOps/LLMOps 成熟度模型**。
- **Grid Dynamics 架构师将就可靠 AI 发表演讲**：**Grid Dynamics** 的 DevOps/MLOps 架构师及 **Data Phoenix** 创始人 **Dmytro Spodarets** 将担任本次研讨会的演讲者。
   - 有兴趣的人士可以通过 [此链接](https://lu.ma/qhytft9t) 预留名额。


  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1369284039680200784)** (1 messages): 

> `Experiment Setup vs. Model Application, Model Validation Strategies` 


- **辩论：实验设置 vs. 纯粹的模型应用**：讨论围绕“完整的**实验设置**（训练和测试）是否为强制性”还是“仅**应用现有模型**就足够了”展开。
   - 主流观点认为，实验设置对于验证至关重要，但这取决于项目的背景和目标。
- **探索模型验证方法论**：对话延伸到了基础训练和测试集划分之外的各种**模型验证**策略。
   - 建议包括**交叉验证技术**、**针对对抗样本的鲁棒性检查**以及**在真实世界数据上的验证**，以确保更广泛的适用性。


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1369381513216393216)** (1 messages): 

> `Windsurf Wave 8, Windsurf Reviews, Knowledge Base, Conversation Sharing, Teams Deploys` 


- **Windsurf Wave 8 开启，推出团队和企业级功能**：Windsurf 宣布启动 **Wave 8**，将在多日内发布新功能，首先从组织管理工具开始。
   - 公告包括一篇 [博客文章](http://windsurf.com/blog/windsurf-wave-8-teams-and-enterprise)、[更新日志](https://windsurf.com/changelog?cachebust=202405061200)，以及指向其 [X](https://x.com/windsurf_ai/status/1919820747037392982) 和 [Bluesky](https://bsky.app/profile/windsurfai.bsky.social/post/3lojj34nq622q) 账号的链接，此外还有 [YouTube 发布视频](https://youtu.be/t7GQFFopQFY) 和 [Reddit 社区](https://reddit.com/r/windsurf)。
- **Windsurf Reviews (Beta) 自动驾驶 PR**：Windsurf 推出了 **Windsurf Reviews (Beta)**，这是一个 GitHub 应用，可根据指定的指南自动审查 PR 并编辑标题/描述。
   - 该功能旨在简化组织内的代码审查流程。
- **Knowledge Base (Beta) 与 Google Docs 结合**：**Knowledge Base (Beta)** 已发布，允许用户将 Google Docs 连接到其 Windsurf 上下文，以实现更好的 Grounding。
   - 这种集成使 Windsurf 能够利用存储在 Google Docs 中的信息，从而提供更明智、更准确的回答。
- **Cascade 会话获得对话共享 (Beta) 功能**：Windsurf 现在提供**对话共享 (Beta)**，允许用户轻松地与团队成员分享成功的 Cascade 会话。
   - 该功能通过允许分享有效的对话流，促进了团队内的知识转移和协作。
- **团队部署直接至 Netlify**：Windsurf 现在支持 **Teams Deploys**，能够直接部署到组织的 Netlify 账户。
   - 这为使用 Netlify 的团队简化了部署流程，提供了更高效的工作流。