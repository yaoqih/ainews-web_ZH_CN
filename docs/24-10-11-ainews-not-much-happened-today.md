---
companies:
- rhymes-ai
- openai
- anthropic
- google
- meta-ai-fair
- oxylabs
date: '2024-10-11T23:00:43.056085Z'
description: '以下是为您翻译的中文内容：


  **Rhymes AI** 发布了 **Aria**，这是一个拥有 **253 亿（25.3B）** 参数的新型多模态 MoE（混合专家）模型。该模型支持文本、代码、图像和视频，具备
  **64k token 的上下文窗口**，并采用 Apache-2.0 许可证。**OpenAI** 的 **o1-preview** 和 **o1-mini**
  模型在高达 **128k token** 的长上下文 RAG（检索增强生成）基准测试中，表现持续优于 **Anthropic** 和 **Google Gemini
  1.5 Pro/Flash**；而 **Google Gemini 1.5** 系列模型在处理高达 **200 万 token** 的极端上下文长度方面表现卓越。**Meta
  AI** 已将其服务扩展至 21 个国家并增加了新的语言支持，但在欧盟地区仍不可用。软件工程任务基准测试 **SWE-bench** 迎来了一周年纪念，同时推出了
  **SWE-bench Multimodal**（多模态版）。新发布的 AI 工具包括 Oxylabs 推出的网页抓取工具 **OxyCopilot**、用于构建
  Python 生产级应用的 **Taipy**，以及用于提示词工程的 **Latitude**。行业洞察强调了 AI 融资动态的变化，以及 OpenAI 将战略重点转向
  ChatGPT 等消费级产品。


  *“所有摘要均由 Claude 3.5 Sonnet 生成，取 4 次运行中的最佳结果。”*'
id: 7b80bca4-c6a2-47bf-a568-03d7ee53b0ff
models:
- aria
- o1-preview
- o1-mini
- gemini-1.5-pro
- gemini-1.5-flash
- gemini-1.5
- claude-3.5-sonnet
original_slug: ainews-not-much-happened-today-4857
people:
- mervenoyann
- osanseviero
- dbrxmosaicai
- ylecun
- ofirpress
- clefourrier
- omarsar0
- rohanpaul_ai
- svpino
- finbarrtimbers
- _philschmid
title: 今天没发生什么特别的事。
topics:
- multimodality
- mixture-of-experts
- long-context
- retrieval-augmented-generation
- benchmarking
- software-engineering
- llm-evaluation
- prompt-engineering
- web-scraping
- python
- production-applications
---

<!-- buttondown-editor-mode: plaintext -->**一个宁静的长周末正是我们所需要的。**

> 2024/10/10-2024/10/11 的 AI News。我们为您检查了 7 个 subreddits、[**433** 个 Twitter](https://twitter.com/i/lists/1585430245762441216) 和 **32** 个 Discord（**231** 个频道和 **2131** 条消息）。预计为您节省阅读时间（以 200wpm 计算）：**218 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

我们确实是 [Tesla 的 Robotaxi/van/humanoid 进展](https://x.com/tesla/status/1844573295510728857?s=46) 的粉丝，但对于 AI Engineer 来说，那里并没有太多可操作的内容。也许你可以阅读 [Dario Amodei 对 AGI 未来的最新看法](https://darioamodei.com/machines-of-loving-grace)，或者更接地气一点，看看 Latent Space 关于 [$2 H100 GPU Bust](https://www.latent.space/p/gpu-bubble) 的连续专题，或者在他完成巨额 Series A 融资后，与 [Braintrust 的 Ankur Goyal](https://www.latent.space/p/braintrust) 进行深度探讨。

---

{% if medium == 'web' %}

**目录**

[TOC]

{% else %}

**目录** 和 **频道摘要** 已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}

---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型发布与进展**

- **Aria by Rhymes AI**: [@mervenoyann](https://twitter.com/mervenoyann/status/1844356121370427546) 重点介绍了 Aria，这是 Rhymes AI 推出的一款新型 25.3B 多模态模型，支持图像/视频输入。它以 Apache-2.0 许可证发布，并附带微调脚本。[@osanseviero](https://twitter.com/osanseviero/status/1844306554192826725) 指出它是**首个多模态 MoE (text/code/image/video)**，总参数量为 24.9B，每个 text token 激活参数为 3.5B，具有 64k token 的上下文窗口。它在 6.4T 语言 tokens 和 400B 多模态 tokens 上进行了预训练。

- **OpenAI 更新**: [@DbrxMosaicAI](https://twitter.com/DbrxMosaicAI/status/1844492162081452106) 报告了对 OpenAI o1-preview 和 o1-mini 模型，以及 Google Gemini 1.5 Pro 和 Gemini 1.5 Flash 的评估。他们发现 [OpenAI o1 模型在高达 128k tokens 的长上下文 RAG Benchmark 上表现出优于 Anthropic 和 Google 模型的持续改进](https://twitter.com/DbrxMosaicAI/status/1844492163293511890)。

- **Google Gemini**: [@DbrxMosaicAI](https://twitter.com/DbrxMosaicAI/status/1844492164501471261) 指出，尽管性能低于 OpenAI 和 Anthropic 模型，但 Google Gemini 1.5 模型在**高达 200 万 tokens 的极端上下文长度下具有稳定的 RAG 性能**。

- **Meta AI**: [@ylecun](https://twitter.com/ylecun/status/1844284825840107919) 宣布 Meta AI 正在 21 个国家推出，包括对他加禄语、阿拉伯语、印尼语、泰语和越南语的支持。然而，它在欧盟地区仍不可用。

**AI 研究与基准测试**

- **SWE-bench**: [@OfirPress](https://twitter.com/OfirPress/status/1844443094709829771) 庆祝了 SWE-bench 成立一周年，这是一个针对软件工程任务的基准测试。他们还推出了 SWE-bench Multimodal。

- **LLM 评估**: [@clefourrier](https://twitter.com/clefourrier/status/1844323838517252172) 分享了一份全面的 LLM 评估指南，涵盖了在管理 Open LLM Leaderboard 时收集的实践见解和理论知识。

- **Astute RAG**: [@omarsar0](https://twitter.com/omarsar0/status/1844435988019544565) 讨论了 Astute RAG，这是一种处理 LLM 中不完善的检索增强和知识冲突的新方法。它能自适应地从 LLM 的内部知识中提取关键信息，并结合来源感知能力迭代地整合内部和外部知识。

**AI 工具与应用**

- **OxyCopilot**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1844367265782771742) 介绍了 OxyCopilot，这是来自 Oxylabs 的一款 AI 驱动助手，可简化网页抓取。它使用先进的 AI 模型来准确识别和生成复杂的解析模式。

- **Taipy**: [@svpino](https://twitter.com/svpino/status/1844437861128606116) 分享了 Taipy，这是一个开源 Python 库，用于在不使用 JavaScript、CSS 或 HTML 的情况下构建端到端生产级应用。它专为数据科学家设计，且易于扩展至生产用途。

- **Latitude**: [@svpino](https://twitter.com/svpino/status/1844363833877373266) 展示了 Latitude，这是一个开源 Prompt Engineering 平台，可在不同场景下评估 Prompt 并进行优化以改进结果。

**AI 行业洞察**

- **AI 融资**: [@finbarrtimbers](https://twitter.com/finbarrtimbers/status/1844392009659973865) 指出，对于 LLM 而言，用极少资本创建巨额利润/成功企业的说法已不如以前准确，预计这将对行业产生激进影响。

- **OpenAI 策略**: [@_philschmid](https://twitter.com/_philschmid/status/1844339615915704747) 推测了 OpenAI 为何可能不优先考虑 API Revenue，而是专注于 ChatGPT 等消费级产品，理由包括来自开源模型的竞争以及 "AGI"/Agents 使用多个模型的潜力。

**迷因与幽默**

- [@karpathy](https://twitter.com/karpathy/status/1844449291282284925) 调侃 YouTube 的算法不理解他想要“高评分、1 小时长、关于任何深奥主题的信息密集型讲座”的愿望。

- [@kipperrii](https://twitter.com/kipperrii/status/1844511021739724900) 幽默地询问在将第一个数组变量命名为 "array" 后，该如何命名第二个数组变量。

这份摘要涵盖了 AI 社区的核心讨论，重点关注了对 AI 工程师群体具有参考价值的新模型发布、研究进展、工具及行业洞察。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. AI 硬件进展：新 GPU 与价格动态**

- **AMD 发布 MI325X - 1kW, 256GB HBM3，声称性能是 H200SXM 的 1.3 倍** ([Score: 97, Comments: 40](https://reddit.com//r/LocalLLaMA/comments/1g0tazi/amd_launched_mi325x_1kw_256gb_hbm3_claiming_13x/))：AMD 发布了 **MI325X GPU**，配备 **256 GB HBM3e 显存**并基于 **CDNA 3 架构**，[产品链接已上线](https://amd.com/en/products/accelerators/instinct/mi300/mi325x.html#tabs-27754605c8-item-b2afd4b1d1-tab)。该 GPU 宣称其 **FP16 和 FP8 峰值理论计算性能是 NVIDIA H200 的 1.3 倍**，同时在推理性能和 Token 生成方面是 NVIDIA H100 的 **1.3 倍**，并提供 **6 TB/s 的显存带宽**。

- **[2 美元的 H100：GPU 租赁泡沫是如何破裂的](https://www.latent.space/p/gpu-bubble)** ([Score: 251, Comments: 80](https://reddit.com//r/LocalLLaMA/comments/1g12ist/2_h100s_how_the_gpu_rental_bubble_burst/))：GPU 租赁市场发生了重大转变，**H100 GPU 的价格从之前的每小时 5-10 美元降至每小时 2 美元**。价格下跌归因于供应量增加以及云服务商之间的竞争，这可能会颠覆 AI 基础设施市场，并使高性能计算对更广泛的研究人员和开发者变得更加触手可及。
  - 用户报告称，在 **vast.ai**、**datacrunch.io** 和 **Lambda Cloud** 等平台上，**H100 GPU 价格**低至 **每小时 1.73-2.40 美元**。一些用户对某些供应商的稳定性和性能问题表示担忧。
  - **NVIDIA 的 AI Enterprise 许可**在 **5 年**后过期，限制了对其容器平台的访问。这一策略连同潜在的二手 GPU **回购计划**，旨在维持高价并控制二手市场。
  - 价格下跌可能会导致**新模型爆发式增长**并惠及开源社区。然而，**A100 80GB GPU** 的价格依然坚挺（**eBay 上为 1.6 万美元**），而像 **V100 32GB** 这样的旧型号价格低至 **550-1500 美元**。
- **[买了一个支持 8 路 GPU 的服务器来运行 32B 模型……但它叫得像喷气式飞机，这正常吗？](https://v.redd.it/iyk0se9f1ytd1)** ([Score: 271, Comments: 173](https://reddit.com//r/LocalLLaMA/comments/1g0kuqg/bought_a_server_supporting_8gpu_to_run_32bbut_it/))：该帖子讨论了在家庭服务器设置中**运行 8 路 GPU 的挑战**，特别是**噪音问题**。作者购买了一台能够支持 **8 路 GPU** 的服务器来运行 **32B 模型**，但发现它产生的噪音大到像喷气式飞机引擎。这种情况引发了关于由于噪音限制，在住宅环境中运行高性能 GPU 服务器的实用性和可行性的疑问。
  - **机架式服务器**设计为在机箱封闭的情况下运行，以实现适当的散热。用户建议**盖上盖子**以减少噪音并确保气流正确，因为开箱运行会触发风扇全速运转。
  - 该服务器可能是 **Supermicro 4029** 型号，专为**被动散热 GPU** 设计，而非桌面级 GPU。用户建议使用 **IPMI 工具**调整风扇速度，并考虑将风扇更换为更安静的替代品，如 **Sunon Maglev 风扇**。
  - 该方案的实用性受到质疑，有人建议使用 **2-4 张 4090** 而不是 8 路 GPU 来运行 32B 模型。一些用户推荐使用**被动散热 GPU** 并探索桌面级选项以缓解噪音问题。


**主题 2. AI 民主化：开源模型与本地推理**

- **[我做了一个在树莓派上运行本地 AI 的家庭服务器](https://www.reddit.com/gallery/1g0lob9)** ([Score: 55, Comments: 30](https://reddit.com//r/LocalLLaMA/comments/1g0lob9/i_made_a_home_server_running_local_ai_on_r_pi/))：在 **10 年的时间里**，作者开发了一个在 **Raspberry Pi** 上运行 **本地 AI** 的家庭服务器，从使用 **Wolfram Alpha** 和 **Wit.ai** 演进到目前的 **LLM**。最新版本 (**MK II**) 运行在 **8GB** 内存、新的 **Raspberry Pi CPU** 和 **1TB** 存储上，专为互联网受限或无网络地区设计，可通过 **热点** 和 **浏览器** 访问。
  - 作者使用 **node server** 处理非 LLM 任务，使用 **PeerJS** 进行 LLM 流式传输。默认模型是在 **ollama** 上运行的 **llama3.2 Q4_K_M 3B**，速度达到 **6-7 tokens/秒**。有一个 [视频](https://imgur.com/a/WNLE3hj) 展示了响应速度。
  - 该设备的设计灵感来自**法拉利座椅头枕**，外形酷似电影《降临》(Arrival) 中的飞船。机箱由**半透明树脂**制成，使内部的 Raspberry Pi 呈现模糊感。更多信息可在 [项目网站](https://persys.ai) 查看。
  - 该项目旨在**在无网络地区提供 AI 访问**，作为一个具备文件管理功能的家庭服务器/云。它包含 **1TB 存储空间**用于存放电影、图片和嵌入文件，可通过双 WiFi 和内置热点供家庭使用。

- **纯粹、现代 Java 实现的快速 Llama 3+ 推理** ([Score: 98, Comments: 37](https://reddit.com//r/LocalLLaMA/comments/1g0f6e2/fast_llama_3_inference_in_pure_modern_java/))：**llama3.java** 项目提供了在 **纯 Java** 环境下实现的 **快速 Llama 3+ 推理**，且 **无任何依赖**。该项目支持 **GGUF 格式**、**Llama 3 tokenizer** 以及 **Grouped-Query Attention**。其特性包括 **Q8_0 和 Q4_0 量化**、利用 Java 的 **Vector API** 实现的 **快速矩阵-向量乘法**，并支持 **Llama 3.1** 和 **3.2** 模型，同时兼容 **GraalVM** 的 **Native Image** 和 **AOT 模型预加载** 以实现快速启动。
  - 用户幽默地讨论了 **Java 的性能**，部分人对其速度表示惊讶。一位评论者指出，Java “*仅比 C 慢 2-3 倍*”，但比机器学习研究中常用的 **Python 快 50 倍**。
  - 讨论还涉及了 Java 与 C# 的 **垃圾回收（garbage collection）** 机制。一位用户提到 Java 的 **ZGC 垃圾回收器** 具有 “*0.05ms 的停顿时间*”，而 C# 在某些情况下被认为有 “**100ms+ 的停顿时间**”。
  - 几条评论调侃了 Java 的名声，其中一条引用了著名的 Java 标语称 “*30 亿设备运行 Llama*”。另一位用户询问该项目是支持 **GPU 推理** 还是仅支持 **CPU 推理**。


- **[我为此工作了 6 个月——为所有人提供免费、易用、本地化的 AI！](https://www.reddit.com/gallery/1g0jehn)** ([Score: 631, Comments: 97](https://reddit.com//r/LocalLLaMA/comments/1g0jehn/ive_been_working_on_this_for_6_months_free_easy/))：基于浏览器的 AI 工具 **Mela** 提供 **免费、本地化 AI** 功能，可用于聊天和文档创建，无需后端支持。该工具历时 **6 个月** 开发，利用 **WebGPU** 进行高效处理，并支持包括 **Llama 2**、**Mistral** 和 **Phi-2** 在内的多种 **开源模型**。Mela 的功能包括 **实时文本生成**、**文档摘要** 以及用于上下文感知响应的 **内置向量数据库**，同时通过将数据保留在设备本地来优先保护用户隐私。
  - **Papeg.ai** 是由一位欧洲数字艺术家创建的 **基于浏览器的 AI 工具**，提供 **实时文本生成**、**文档摘要** 和 **语音聊天** 等功能。该项目在 [GitHub](https://github.com/flatsiedatsie/papeg_ai) 上 **开源**，并支持 **自定义 AI 模型** 和 **Ollama 集成**。
  - 用户对该项目的 **盈利模式** 和 **企业级用例** 的潜力表示关注。一些用户对 **自动文件下载** 提出了担忧，并认为在开始下载前需要增加 **警告**。
  - 该工具使用 **IndexDB** 进行文档存储，使用 **Orama** 进行向量搜索，并在向量数据库上执行 **混合搜索**。用户可以 **连接到外部 API**，开发者正在考虑实现 **OpenAI API 集成**。


**主题 3：新 AI 模型发布与基准测试**

- **NVIDIA 发布 Mistral-NeMo-Minitron 8B Instruct** ([Score: 87, Comments: 17](https://reddit.com//r/LocalLLaMA/comments/1g0mgtl/announcing_mistralnemominitron_8b_instruct_by/))：**NVIDIA** 宣布推出了 **Mistral-NeMo-Minitron 8B Instruct** 模型，这是一个据称具有高准确率的新基础模型。公告包含了性能对比，并提供了指向 NVIDIA 开发者网站详细博客文章的链接，以获取更多关于该模型能力和实现的信息。
  - 用户质疑为何将其与 **Gemma-7B** 而非 **Gemma2-9B** 进行对比，强调了基准测试选择在模型评估中的重要性。
  - 讨论中分享了性能对比，提示 **Gemini Flash 8B** 达到了 **~75 的 MMLU 分数**，同时作为一个多模态模型，其文本模型组件可能更小。
  - **Qwen 2.5 7B** 被提到达到了 **75.4 的 MMLU-redax** 分数，这参考了一个经过仔细注释的 MMLU 基准测试版本。

- **[LLM Hallucination Leaderboard](https://github.com/lechmazur/confabulations/)** ([Score: 62, Comments: 18](https://reddit.com//r/LocalLLaMA/comments/1g0l7be/llm_hallucination_leaderboard/)): **LLM Hallucination Leaderboard** 比较了各种大语言模型生成错误或无根据信息的倾向。模型根据其在**三个关键指标**上的表现进行评估：**幻觉率**、**事实准确性**和**一致性**。该排行榜目前包括了 **GPT-3.5**、**GPT-4** 和 **Claude** 等热门模型的结果，为它们在不同语境下的虚构（confabulation）倾向提供了定量评估。
  - 用户质疑在测试中使用 **temperature 0** 的做法，作者指出**更高的温度设置**并未显著影响结果。讨论强调了采样方法在 LLM 评估中的重要性。
  - 最初对 **GPT-4** 的糟糕表现存在困惑，后来澄清是 **GPT-4-mini** 表现不佳，而 **GPT-4** 表现优异。这突显了同一模型系列不同版本之间性能的差异。
  - **Llama 模型**由于其**谨慎的回答**表现出强劲的性能，导致幻觉较少，但拒答率较高。这突显了 LLM 输出中准确性与完整性之间的权衡。


- **DARKEST Planet 16.5B - 异常强大的非 AI 创意模型，具有 "regen" 随机性。** ([Score: 103, Comments: 28](https://reddit.com//r/LocalLLaMA/comments/1g0wwzz/darkest_planet_165b_unusually_strong_non_ai/)): **DARKEST Planet 16.5B** 模型是 "Dark Planet" 系列的一部分，是一个 **71 层**的创意 AI 模型，使用 **Brainstorm 40X** 流程开发，适用于各种创意应用。它具有**独特的属性**，包括使用相同提示词的 "regens"（重新生成）之间存在显著差异、卓越的细节和文笔水平，以及在 **repetition penalty 1.02** 及以上、**temperature 0-5** 时的异常稳定性，并提供了设置和量化指南。
  - 用户报告了该模型在 **NSFW 内容生成**方面的问题，指出它经常**拒绝**生成此类内容。开发者建议尝试不同的量化版本（**Q4KS** 和 **IQ4XS**），并提到即将推出的 **"DARKEST PLANET" 16.5B 版本**可能会解决这个问题。
  - 讨论了该模型的**“非 AI”化**特质，指的是它能够生成没有典型 AI 模式或陈词滥调的文字。用户赞赏其**人性化的文本输出**以及针对同一提示词的**不可预测的重新生成**。
  - 一些用户遇到模型在**角色扮演场景中替用户回答**的问题，尽管尝试了各种设置。开发者上传了**完整源码仓库**到 [Hugging Face](https://huggingface.co/DavidAU/L3-DARKEST-PLANET-16.5B) 以回应用户的兴趣。


**Theme 4. AI 评估与微调技术**

- **Hugging Face LLM Evaluation Guidebook** ([Score: 38, Comments: 6](https://reddit.com//r/LocalLLaMA/comments/1g0gvku/hugging_face_llm_evaluation_guidebook/)): Hugging Face 的评估团队在 [GitHub](https://github.com/huggingface/evaluation-guidebook) 上发布了 **LLM Evaluation Guidebook**，为创建自定义评估、分析当前方法和故障排除提供全面资源。该指南是根据管理 **Open LLM Leaderboard** 和设计 **lighteval** 时获得的见解开发的，旨在提供实践和理论知识，并计划定期添加演示快速评估实验和最佳实践的 notebook。
  - **LLM Evaluation Guidebook** 获得了积极反馈，用户对这一全面资源表示赞赏。评论中提供了一个更正后的 **GitHub 链接**以便于访问。
  - 用户对该指南以及评估团队对社区的贡献表示感谢。提交者积极与评论者互动，听取他们的反馈。
  - 讨论集中在 **LLM-as-a-judge** 工作流的挑战上，强调了**评估标准模糊性**的问题。提交者表示赞同，指出这种方法目前虽不可靠但很有前景。


- **监控你的 LlamaIndex 应用以进行模型微调或评估** ([Score: 80, Comments: 1](https://reddit.com//r/LocalLLaMA/comments/1g0lddr/monitor_your_llamaindex_application_for_model/)): 作者开发了一个工具，通过收集模型响应并在 **Argilla** 中实现**标注 UI**，来**监控 LlamaIndex 应用**以进行**模型微调和评估**。他们分享了一个 [GitHub notebook](https://github.com/argilla-io/argilla-cookbook/blob/main/rag_monitor_llamaindex.ipynb) 演示了这一设置，这对于拥有可以协助改进模型输出的用户的应用特别有用。

- **如果使用 Transformers (TRL)，小 batch sizes 和 gradient accumulation 的微调表现不佳！** ([Score: 42, Comments: 22](https://reddit.com//r/LocalLLaMA/comments/1g0dy0k/finetuning_with_small_batch_sizes_and_gradient/)): 使用 **Hugging Face 库**（TRL 和 Transformers）进行微调时，在使用 **small batch sizes** 和 **gradient accumulation** 的情况下表现出明显的性能问题。针对 **Llama 3.2**、**SmolM-135M** 和 **Qwen2.5** 的实验表明，尽管在数学上是等价的，但 **batch_size=1** 配合 **gradient_accumulation_steps=32** 的表现远差于 **batch_size=32** 配合 **gradient_accumulation_steps=1**。该问题在不同的精度格式（**bf16** 和 **fp32**）下依然存在，并已[报告](https://github.com/huggingface/trl/issues/2175)至 TRL 仓库。
  - 用户表示需要一份关于现代模型微调的**最新指南**，包含当前的最佳实践。[HuggingFace alignment handbook](https://github.com/huggingface/alignment-handbook) 和 [SimPO paper](https://arxiv.org/pdf/2408.13296) 是推荐的超参数和对齐技术资源。
  - 基于 Transformers 构建的 **Unsloth** 实验显示了与原始发现类似的行为。虽然观察到了 training loss 的差异，但 validation loss 保持相似，这表明对模型本身的影响微乎其微。
  - 讨论强调，与普遍看法相反，**gradient accumulation** 和 **batch size** 并非严格等价。**Oobabooga Training Pro extension** 指出，gradient accumulation 虽然对 VRAM 友好，但可能会降低训练的保真度。

## 其他 AI Subreddit 回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 研究与技术**

- **Google Deepmind 通过联合样本选择推进多模态学习**：在 /r/MachineLearning 中，一篇 [Google Deepmind 论文](https://arxiv.org/html/2406.17711v1) 展示了如何通过联合样本选择（joint example selection）进行数据策展，从而进一步加速多模态学习。

- **Microsoft 的 MInference 显著加快长上下文任务推理**：在 /r/MachineLearning 中，[Microsoft 的 MInference 技术](https://arxiv.org/abs/2407.02490) 能够在保持准确性的同时，实现长上下文任务中多达数百万个 token 的推理，大幅提升了支持模型的速度。

- **利用 10 亿个从网络策划的角色扩展合成数据生成**：在 /r/MachineLearning 中，一篇[关于扩展合成数据生成的论文](https://www.reddit.com/r/MachineLearning/comments/1dzergu/r_scaling_synthetic_data_creation_with_personas/) 利用 Large Language Model 中的多样化视角，从网络数据策划的 10 亿个角色（personas）中生成数据。

**AI 模型发布与改进**

- **Salesforce 的“小巨人” xLAM-1b 模型在 function calling 方面超越 GPT 3.5**：在 /r/LocalLLaMA 中，Salesforce 发布了 xLAM-1b，这是一个 10 亿参数的模型，实现了 [**70% 的 function calling 准确率，超越了 GPT 3.5**](https://www.reddit.com/r/LocalLLaMA/comments/1dz8g10/salesforce_tiny_giant_xlam1b_model_surpasses_gpt/)。尽管体积相对较小，它仍被称为“function calling 巨人”。

- **具备 function calling 能力的 Phi-3 Mini (6月版)**：在 /r/LocalLLaMA 中，Rubra AI 在 6 月发布了更新后的 Phi-3 Mini 模型，[**具备 function calling 能力**](https://www.reddit.com/r/LocalLLaMA/comments/1dzhe38/phi3_mini_june_with_function_calling/)。它与 Mistral-7b v3 具有竞争力，且表现优于基础版 Phi-3 Mini。

- **开源视频生成工具 Pyramid Flow SD3 发布**：一个新的[开源视频生成工具 Pyramid Flow SD3](https://www.reddit.com/r/StableDiffusion/comments/1g0dpv7/pyramide_flow_sd3_new_open_source_video_tool/) 发布，该工具基于 Stable Diffusion 3。它包含 384p 和 768p 模型，其中 384p 版本需要约 26GB 显存。

**AI 行业与商业**

- **OpenAI 的预测显示了大规模的计划投资**：[OpenAI 的预测](https://www.reddit.com/r/singularity/comments/1g0djzx/some_details_from_the_informations_article_openai/) 表明该公司计划投入巨资，到 2026 年亏损可能增加三倍，达到 140 亿美元。这显示了对未来 AI 能力和市场潜力的极强信心。

- **Tesla 展示 robotaxi 概念**：Elon Musk [展示了 Tesla 的 robotaxi 概念](https://www.reddit.com/r/singularity/comments/1g11bzc/elon_musk_says_teslas_robotaxis_will_have_no_plug/)，其特点包括感应充电、自动清洁，并声称能将停车场转变为公园。然而，许多评论者对该概念的时间表和实用性表示怀疑。

**AI 能力与局限**

- **论文证明了 LLM 中的概率推理**：一篇 [新论文](https://www.reddit.com/r/singularity/comments/1g0lu2o/another_paper_showing_that_llms_do_not_just/) 提供的证据表明，Large Language Model 进行的是概率推理而非纯粹的记忆，尽管文中也指出了一些局限性。

- **关于 ChatGPT Advanced Voice Mode 能力的辩论**：用户[讨论了他们的使用体验](https://www.reddit.com/r/singularity/comments/1g0ynho/my_opinion_on_llms_has_plummeted_after/)，一些人认为 ChatGPT 的 Advanced Voice Mode 令人印象深刻，而另一些人则指出与文本交互相比，它存在明显的局限和严格的审查。

**新兴技术**

- **用于 VR 运动模拟的大脑刺激**：一种[利用前庭电刺激（galvanic vestibular stimulation）在 VR 中模拟运动的新技术](https://www.reddit.com/r/singularity/comments/1g0h5mo/pcvr_with_brain_stimulation/) 得到展示，该技术有可能减少晕动症并增强沉浸感。

- **雄心勃勃的长寿研究目标**：Clock.bio [宣布计划](https://www.reddit.com/r/singularity/comments/1g0ggc1/mark_kotter_clockbio_we_believe_the_field_is/) 在十年末通过 3 期临床试验，基于衰老生物标志物将人类健康寿命延长 20 年，尽管一些评论者对这一时间表表示怀疑。


---

# AI Discord 回顾

> 由 O1-mini 生成的摘要之摘要的摘要

**主题 1. 加速模型训练与微调**

- [**使用 DeepSpeed 和 FSDP2 优化 Llama3.2**](https://github.com/pytorch/pytorch)：工程师们正利用 **DeepSpeed** 和 **FSDP2** 应对 **Llama3.2** 对显存（VRAM）的高需求，在有限的 GPU 资源上实现高效训练。诸如 **activation checkpointing** 等技术被证明对有效管理内存至关重要。
- [**量化技巧提升 torchao 性能**](https://github.com/pytorch/ao/blob/main/aqt/jax/v2/examples/examples.ipynb#L87)：通过 **int8 tensor** 替换和基于硬件的优化进行创新，用户正在增强 **torchao** 以实现更快的计算。尽管存在一些性能挑战，但结合 **quantization**（量化）和 **dequantization**（反量化）为可扩展性带来了希望。
- [**在 16GB GPU 上微调 Llama 7B？挑战达成！**](https://github.com/pytorch/pytorch/issues/131679)：开发者们正通过在单个 **16GB GPU** 上微调 **Llama 7B** 来挑战极限，利用 **Runpod** 和 **CPU offload optimizers** 等工具来应对内存限制。**QLoRA** 的成功凸显了社区的适应能力。

**Theme 2. 多模态 AI：桥接文本、图像和音频**

- [**Aria 作为开源多模态冠军脱颖而出**](https://arxiv.org/abs/2410.05993)：**Aria** 模型凭借其 **3.9B parameters** 设定了基准，在 **language understanding**（语言理解）和 **multimodal tasks**（多模态任务）中表现优于 **Pixtral-12B** 和 **Llama3.2-11B**。其开源特性促进了更广泛的采用以及在整合不同数据类型方面的创新。
- [**从 Discord 聊天到播客：AI 的新游乐场**](https://github.com/GilgameshofUT/AIResearcher)：社区正在尝试利用 **NotebookLM** 等工具从日常 Discord 对话中生成 **podcasts**（播客）。虽然输出质量参差不齐，但其创作潜力正激发着热烈的参与。
- [**非语言声音分析成为焦点**](https://huggingface.co/spaces/coqui/xtts)：使用 **TTS** 模型对 **nonverbal vocalizations**（非语言发声）和 **emotions**（情感）进行的探索，正揭示出 AI 细微的能力。Google 的 **TTS model** 处于领先地位，展示了 AI 系统在更深层次情感智能方面的潜力。

**Theme 3. 掌控成本与 GPU 基础设施**

- [**H100 租赁价格降至 $2/小时：该买还是租？**](https://latent.space/p/gpu-bubble)：随着新供应商的出现和 **Blackwell** 芯片的推出，GPU 租赁市场蓬勃发展，**H100 prices** 从 **$8/小时** 暴跌至 **$2/小时** 以下。随着基础设施选项的扩展，小型 AI 公司正在权衡 **buying vs. renting**（购买与租赁）的利弊。
- [**Batch-GPT 将 API 成本削减 50% 以上**](https://github.com/djellalmohamedaniss/distilabel-cost-calculator)：**Batch-GPT** 工具通过其创新的 **Batch API** 将 **OpenAI API** 费用降低了 **50%** 以上，正在彻底改变成本管理。开源爱好者正在集成自动缓存功能以实现无缝采用。
- [**Runpod 和 AWS 在 GPU 集群领域领先**](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py)：约 **$2.5/小时** 的 **H100 clusters** 推荐重点介绍了 **Runpod** 和 **AWS** 等服务，为大量的 AI 训练需求提供了强大的选择。这些平台正成为高效扩展大模型部署的首选。

**Theme 4. 应对 API 性能与集成障碍**

- [**Perplexity API vs. Perplexity Labs：速度竞赛**](https://labs.perplexity.ai/)：用户指出 **Perplexity API** 的 **2 秒响应时间** 落后于 **Perplexity Labs** 低于 **1 秒** 的速度，并讨论通过实现 **web sockets** 来缩小差距。随着用户寻求更好的性能和引文访问等增强功能，支持渠道非常活跃。
- [**Cohere 的 V2 API 在速度上面临挑战**](https://docs.cohere.com/docs/migrating-v1-to-v2#web-search)：迁移到 **Cohere's v2 API** 带来了挑战，响应时间从 **v1** 的 **1-1.5 秒** 攀升至 **2-3 秒**。社区成员正在寻求解决方案并分享迁移见解，以优化他们的工作流程。
- [**集成 op3nai 实时 API：需要成功案例**](mailto:api@perplexity.ai)：开发者们渴望在 **O1** 等项目中实现 **op3nai real-time** API，但在访问和文档方面面临障碍。邮件支持和社区排障对于克服这些集成挑战至关重要。

**Theme 5. 使用尖端工具简化 AI 开发**

- [**Gradio 5 发布，带来强劲功能更新**](https://huggingface.co/blog/gradio-5)：**Gradio 5** 的发布引入了**安全升级**、**华丽的新 UI** 以及创新的 **AI Playground** 功能，助力开发者更高效地构建 ML 应用。这些增强功能承诺提供**极速加载**和改进的用户体验。
- [**Symphony 自动化多 Agent AI 工作流**](https://www.loom.com/share/8d613aa434cf4a829e93160d01df35ae?sid=5216da3d-dcad-461c-bd37-6ba6a3c882b9)：**Symphony** 将用户描述转化为功能性的 **Agent 工作流**，简化了复杂的 AI 任务自动化。详细的 [Loom 演示](https://www.loom.com/share/8d613aa434cf4a829e93160d01df35ae?sid=5216da3d-dcad-461c-bd37-6ba6a3c882b9)展示了集成 **perplexity** 和 **image-to-text** 等工具的便捷性。
- [**ComfyUI vs. Automatic1111：选择你的 AI 工具**](https://github.com/afnanenayet/diffsitter)：社区倾向于在高级 **Flux** 使用中使用 **ComfyUI**，而 **Automatic1111** 仍是初学者的首选。这两个平台以及 **PyTorch** 和 **Diffusers**，在增强不同用户群体的 **Stable Diffusion** 工作流方面都至关重要。

**提及的链接：**

- [使用 DeepSpeed 和 FSDP2 优化 Llama3.2](https://github.com/pytorch/pytorch)
- [量化技巧提升 torchao 性能](https://github.com/pytorch/ao/blob/main/aqt/jax/v2/examples/examples.ipynb#L87)
- [在 16GB GPU 上微调 Llama 7B？挑战开始！](https://github.com/pytorch/pytorch/issues/131679)
- [Aria 闪耀，成为开源多模态冠军](https://arxiv.org/abs/2410.05993)
- [从 Discord 聊天到播客：AI 的新游乐场](https://github.com/GilgameshofUT/AIResearcher)
- [非语言声音分析成为焦点](https://huggingface.co/spaces/coqui/xtts)
- [H100 租金降至 $2/小时：该买还是该租？](https://latent.space/p/gpu-bubble)
- [Batch-GPT 将 API 成本削减 50% 以上](https://github.com/djellalmohamedaniss/distilabel-cost-calculator)
- [Runpod 和 AWS 在 GPU 集群领域领先](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py)
- [Perplexity API vs. Perplexity Labs：速度竞赛](https://labs.perplexity.ai/)
- [Cohere 的 V2 API 在速度上面临挑战](https://docs.cohere.com/docs/migrating-v1-to-v2#web-search)
- [集成 op3nai 实时 API：需要成功案例](mailto:api@perplexity.ai)
- [Gradio 5 发布，带来强劲功能更新](https://huggingface.co/blog/gradio-5)
- [Symphony 自动化多 Agent AI 工作流](https://www.loom.com/share/8d613aa434cf4a829e93160d01df35ae?sid=5216da3d-dcad-461c-bd37-6ba6a3c882b9)
- [ComfyUI vs. Automatic1111：选择你的 AI 工具](https://github.com/afnanenayet/diffsitter)

---

# 第 1 部分：高层级 Discord 摘要

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) 频道

- **Audio Overviews 生成出现问题**：团队正在调查 **Audio Overviews** 生成失败的原因，这可能会阻碍其他功能的性能。
  
  - 成员们表示担心该问题可能会产生连锁反应，影响系统中其他组件的功能。
- **NotebookLM 增强居家学习乐趣**：参与者正在探索使用 **NotebookLM** 为居家学习环境创建引人入胜的课程计划，特别是针对一名 **13 岁**的学生。
  
  - 然而，有人警告称 AI 可能会产生缺乏实质内容深度的幻觉输出。
- **源自 Discord 聊天的播客**：社区对从 Discord 对话中生成播客反响热烈，将闲聊转化为有趣的音频内容。
  
  - 一些用户分享了利用搞怪聊天记录进行播客创业的幽默看法，同时也对输出质量表示关注。
- **非语言声音分析启动探索**：目前正在进行通过 **TTS 模型**分析非语言发声和情感的实验，展示了 AI 能力开发的一个潜在领域。
  
  - 这项努力是正在进行的调查的一部分，旨在研究 AI 如何准确传达和解释细微的音频元素。
- **AI 探索个人梦境日志**：一位成员正在尝试使用 AI 从个人梦境日志中提取反复出现的主题，突显了 AI 的多样化应用。
  
  - 这一探索鼓励其他人思考类似的 AI 用途，用于分析个人经历和叙述。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **多模态模型的热潮**：社区正热切期待对 **Llama3.2** 和 **Qwen2 VL** 等多模态模型的支持，预计下周将发布更新。
  
  - 这一进展备受期待，成员们对新的可能性表达了兴奋之情。
- **微调策略备受关注**：成员们讨论了针对 **G2-9B** 等模型的微调，指出了其对 **VRAM** 的高要求以及在 **Dora** 上的有效性。
  
  - **Gemma 9B** 出现了一些挑战，包括 **VRAM** 问题以及训练过程中出现的 **NaN** 值。
- **H100 集群建议**：用户分享了关于以约 **$2.5/小时** 使用 **H100 clusters** 的见解，强调了获得最佳性能所需的 **VRAM**。
  
  - 对于寻求大量 AI 训练资源的用户，推荐使用 **Runpod** 等选项。
- **对 OpenAI O1 的猜测**：关于 OpenAI 的 **O1** 意见不一，推测其允许在用户不可见的情况下进行链式提示（chains of prompts）。
  
  - 一些成员对源代码的封闭性提出质疑，反映出对所宣称功能的怀疑。
- **探索 LLM 中的 CoT 推理**：成员们认为通过 Chain of Thought (CoT) 推理增强 **LLMs** 对未来模型具有前景。
  
  - 提议包括将 CoT 集成到 Attention 模型的 **k/v cache** 中以进行潜在的实验。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Distilabel 成本计算工具**：一位成员展示了一个用于 **Distilabel** 流水线 **cost calculation** 的新包，其功能已在 **TextGeneration** 和 **TextClassification** 任务上进行了测试，可在 [此处](https://github.com/djellalmohamedaniss/distilabel-cost-calculator) 获取。
  
  - 该包很快将支持各种 LLM APIs 的 YAML 定价选项，提升用户管理成本的体验。
- **Gradio 5 正式上线**：**Gradio 5** 的发布宣布了重大增强功能，包括**安全升级**和 **AI Playground feature**，赋能开发者更高效地创建 ML applications。
  
  - 开发者可以期待通过实现的 SSR 获得**极速加载**体验，以及提升应用交互的**华丽新 UI 设计**。
- **NVIDIA 在 LLM 训练中的创新**：NVIDIA 最近的研究强调了利用回收模型（upcycled models）改进 **LLM training**，其中 **Nemotron-4 15B** 在 **MMLU** 上达到了 **67.6%**。
  
  - 他们的方法结合了 **MoE** 技术，为优化大模型训练提供了替代方案，同时解决了高性能需求。
- **情感检测模型的见解**：一位研究**情感检测模型**的用户分享了使用 **FER** 和 **DeepFace** 的经验，引发了关于识别细微情感状态局限性的讨论。
  
  - 成员们指出了衡量情感准确性的具体挑战，强调在各种情感识别应用中需要更好的工具。
- **Diffusion 过程中的多通道考量**：讨论涉及在不同通道上应用 **diffusion noise**，特别是在处理具有不同信息层（包括生物数据）的图像时。
  
  - 参与者提出了关于单一噪声调度（noise schedule）是否能在多样化的通道数据表示中保持有效性的问题。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **模型在通用任务中表现出色**：成员们确认新模型在执行类似于 **ChatGPT** 的各种任务方面表现优异，充分利用了预训练和 **instruct finetuned weights**。
  
  - 这种多功能性使用户能够毫不费力地将这些模型部署到各种应用中。
- **从 M1 Max 升级到 M3 Max 带来显著成效**：将配备标准内存的 **M1 Max** 升级为拥有 **128GB RAM** 的 **M3 Max**，被证明可以顺利运行 **LLMs** 而无任何问题。
  
  - *许多用户正在转向更大的系统，以有效管理高需求的模型工作负载。*
- **关于 RTX 5000 定价的辩论令成员感到震惊**：传闻称新款 **RTX 5000 series** 的定价可能在每张显卡 **1,500 美元到 2,500 美元** 之间，价格可能低于 **Mac Studio** 的配置。
  
  - 对多显卡相关费用的担忧正在增加，特别是关于散热和能源成本方面。
- **MLX Backend 的兼容性问题**：在使用 **MLX backend** 的 **GPUs** 上加载模型时出现了问题，较大的模型会默认使用 **CPU**。
  
  - 成员们建议在独立的 **Apple MLX** 设置中检查性能，并考虑在 **GitHub** 上提交 issue 以获得更多支持。
- **外部 e-GPU 兼容性受到质疑**：用户探讨了通过 **Thunderbolt** 将 **e-GPU** 连接到 **RTX 4090** 是否能增加显存，但对潜在的性能提升表示怀疑。
  
  - **Thunderbolt** 连接可能会引入延迟，从而在混合 **GPU** 资源时影响整体性能。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Wondercraft 推出 Director Mode**：随着 [Director Mode](https://x.com/wondercraft_ai/status/1844378469628772586) 的发布，**Wondercraft** 赋予了用户控制 **AI** 语音表现的能力，称其为今年最重要的更新。
  
  - 这一创新增强了音频项目的创作灵活性，提供了以前无法实现的精细化表演选项。
- **H100 GPU 价格崩盘**：一篇题为《[*$2 H100s: How the GPU Rental Bubble Burst*](https://latent.space/p/gpu-bubble)》的客座文章报告称，**H100** 租赁价格从 **8 美元/小时** 降至不足 **2 美元/小时**，引发了关于购买还是租赁的讨论。
  
  - 随着 **Blackwell** 芯片的出现，文章为探索基础设施方案的小型 **AI** 公司提出了战略性思考。
- **关于现场演示和技术故障的见解**：一位自封为“现场演示之王”的成员分享道，新手和资深演示者之间的期望存在显著差异，这往往会导致**技术困难**。
  
  - 社区成员对此表示赞同，并讲述了自己在演示过程中发生的意外，这些意外曾导致关键项目的展示停滞。
- **Discord API 设置的挑战**：成员们讨论了在 **discord.py** 和 **discord.js** 等库之间切换时获取 **API** 密钥的**权限痛苦**，强调了其中的复杂性。
  
  - 一位成员幽默地指出，获得正确的设置感觉更像是一门艺术，而不是一个简单的过程，经常会干扰工作流程。
- **简化功能构建**：在功能构建的讨论中，出现了关于**计算器应用**或**待办事项列表**等简单项目想法的建议，以帮助简化开发者的工作。
  
  - 强调效率时，一位成员表示 **“好玩且管用的东西只需 10 秒就能做出来”**，突出了项目中复杂性与简单性之间的平衡。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **0.15 的 float precision 失误**：一位用户询问为什么某个值不等于 **0.15**，引发了关于编程中 **float precision** 的讨论，并指出 literals 被实例化为 **Float64**。
  - 澄清指出，这种差异的产生类似于 **1/3** 无法在 **base 10** 中精确表示。
- **一致的 floating point 行为**：尽管存在精度问题，另一位成员保证在 IEEE 745 **64-bit** floating points 中，数值保持 **self-consistent**（自洽）。
  - 在该表示法的限制范围内，计算结果准确地等于 **0.15**。
- **定义 trivial types 的挑战**：用户们解决了为仅包含 **inline memory** 的 **trivial types** 定义 **trait** 的问题，并讨论了 **AnyTrivialRegType** 的局限性。
  - 他们表示由于现有 **trait** 约束的组合限制，需要替代方案。
- **AESNI instruction set 实现问题**：一位用户描述了检查 **AESNI instruction set** 支持的代码，但在使用 **llvm_intrinsic** 确保与 X86 architecture 兼容时，遇到了 compiler 识别问题。
  - **AVX2** 和 **AVX512** 的作用得到了确认，允许跨多个指令宽度进行操作。
- **In-place structs 创建讨论**：关于在向列表追加元素时 **in-place** 创建 **structs** 以防止不必要拷贝的查询，指出 **rvalue** struct 创建通常可以避免拷贝。
  - `__moveinit__` 方法被强调为在需要时进行拷贝的一种轻量级方法。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity API 慢于预期**：用户注意到 Perplexity **API** 的 **2 秒响应时间**，而 [Perplexity Labs](https://labs.perplexity.ai/) 的响应时间不到 **1 秒**。他们推测，像 Labs 那样实现 **web sockets** 可能会提升 **API** 的性能。
  - 一位用户报告称，为了获取引用权限和提高 **rate limit** 给支持部门发了邮件，但没有收到回复；他们被建议联系 [api@perplexity.ai](mailto:api@perplexity.ai) 以获得更快的解决。
- **特斯拉推出新款 robovan 模型**：特斯拉发布了一款[新款 robovan](https://www.perplexity.ai/page/tesla-robovan-HbGsP0T1Tea_paN7W0u4gw)，旨在通过高电力效率和先进的驾驶辅助系统改善城市交通。
  - 这一创新模型旨在显著改变城市交通并减少碳足迹，为更清洁的城市环境铺平道路。
- **米尔顿飓风在佛罗里达州造成严重破坏**：米尔顿飓风在佛罗里达州造成了重大混乱，引发了紧急疏散，详见[此处](https://www.perplexity.ai/page/hurricane-milton-hits-florida-fJjruP5JR5ilumQEJSmcfw)。
  - 气象学家继续监测其不可预测的路径，强调在如此严峻的天气条件下做好准备的重要性。
- **德国撇号争议加剧**：一场围绕 [德国](https://www.perplexity.ai/page/germany-s-apostrophe-debate-1DrUiXyvR0i7zpbuc9GKVA) 撇号用法的辩论引发了关于语言标准现代化的重大讨论。
  - 语言学专家正在就现行规则是否应该演变以反映当代用法发表意见。
- **活跃的社区互动**：成员们分享了轻松的 memes，包括一只在雪地里的猫，配文是“当地域结冰时”（when hell freezes over），反映了社区轻松的氛围。
  - 这些俏皮的时刻辅以关于功能和特性的深刻讨论，使聊天保持活跃。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **处理 API 使用问题**：一位成员询问如何通过私信处理账单和使用问题，Alex Atallah 建议在使用 `/generation` API 后耐心等待 ID 出现。
  
  - 这反映了用户在 **API request** 和响应延迟方面的常见体验。
- **比较模型定价策略**：讨论中提到了 **Mistral Nemo 12B Starcannon** 与 **Rocinante 12B** 之间的价格差异，指出 Mistral 的定价更具吸引力。
  
  - 对话指出，市场上有限的竞争使得 **Rocinante 12B** 能够收取更高的价格。
- **LLM 提升写作质量**：一位用户分享说，让 LLM 专注于文章的特定部分显著提高了他们的写作产出。
  
  - 另一位用户对此表示支持，称有了 LLM，任何人只要努力都能提高写作质量。
- **如何有效地分享模型**：用户了解到“分享模型”按钮会生成一个链接来分享当前聊天室的模型设置，但缺少参数和 prompt 等细节。
  
  - 该功能简化了设置共享，但用户可能需要为共享链接补充详细说明。
- **访问漏洞引发关注**：一位用户指出了一些 Bug，允许他们通过不同的账户访问旧账户的聊天记录，这表明可能存在 Cookie 问题。
  
  - 这引发了关于浏览器工具中聊天数据如何处理和存储的广泛讨论，提出了隐私方面的考量。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **GPT-NeoX 为库增加了新功能**：HPC 团队为 **GPT-NeoX** 库引入了训练后（post-training）功能，支持原生的 **SFT, DPO, 和 KTO finetuning**。
  
  - 测试结果显示，在 **13B scale** 下，其性能比 **HuggingFace 的 trl 库** 提升了 **30%**，确保了大规模计算系统具备更强的可扩展性。
- **辩论基于熵的采样的有效性**：关于 **Llama3.1** 等模型中基于熵的采样（entropy-based sampling）的讨论强调，需要对基准推理分数的改进进行严格验证。
  
  - 成员们呼吁提供可靠证据，将采样技术与性能提升联系起来，并建议进行详细分析。
- **探索 AI 在计算精神病学中的作用**：有人提议研究 **LLM** 在深入了解精神障碍方面的潜力，强调了“计算精神病学”的概念。
  
  - 大家达成共识，虽然 LLM 不会表现出类似人类的精神障碍，但分析其输出可能会产生有价值的框架，尽管存在对齐挑战。
- **lm-eval-harness 引发分词警告**：一位成员报告了在运行 **lm-eval-harness** 时关于 **tokenizers** 分叉进程的警告，指出这些警告导致输出过多。
  
  - 该问题可以通过将 `TOKENIZERS_PARALLELISM` 环境变量设置为 **false** 来解决，从而在保持设置完整性的同时防止重复警报。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 脱颖而出**：**Aider** 给用户留下了深刻印象，在修复 Bug 和编码任务方面表现优于 **Cline** 和 **Cursor** 等竞争对手；一位用户在对多个框架进行严格测试后称其为最佳工具。
  
  - 成员们一致称赞其在前端和后端应用中的效率，称其为“真正的最佳选择”。
- **DeepSeek 在效率方面表现挣扎**：用户报告了对 **DeepSeek** 的挫败感，理由是性能迟缓且在整合功能时效率低下，特别是对于独立开发者而言。
  
  - 一位成员由于“编辑格式错误”而重新使用 **Sonnet-3.5**，对 DeepSeek 的功能表示失望。
- **配置困惑得到解决**：一位用户请求帮助为 **openrouter** 模型配置 `.env` 文件，面临意外的默认更改问题。
  
  - 另一位用户建议，`--edit-format whole` 选项可能会使 DeepSeek 的性能问题进一步复杂化。
- **Diffsitter 以语义差异对比惊艳众人**：[Diffsitter](https://github.com/afnanenayet/diffsitter) 是一个通过 AST 比较创建具有语义意义的 diff 的工具，有效地忽略了格式变化。
  
  - 成员们非常欣赏它如何产生更简洁的 diff，而没有多余空格的干扰。
- **Aider 中的错误处理小故障**：Aider 中频繁出现的 **search/replace errors** 引发了关于如何有效利用设置以增强性能的讨论。
  
  - 用户参考了 [故障排除指南](https://aider.chat/docs/troubleshooting/edit-errors.html) 来解决这些问题，强调使用能力更强的模型来改善结果。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **语音调制技术引起关注**：讨论者分享了鼓励 **AI 语音调制** 的方法，强调了特定的提示词（如 *voice modulation*）如何有效地模拟唱歌而无需实际演唱。
  
  - 成员们对 AI 不愿参与表现力强的表演（如戏剧或诗歌）感到沮丧。
- **AI 被比作高功能反社会人格**：一位成员提出，高功能反社会人格和 AI 的共同特征是都在没有情感负担的情况下进行**逻辑计算**。
  
  - 这引发了一场既幽默又严肃的辩论，讨论反社会人格特征是否在 AI 系统中被有意识地建模。
- **OpenAI Copilot 面临性能批评**：用户正在批评最新版本的 **OpenAI Copilot**，声称其表现不如之前的版本，甚至逊于 **Google's Gemini**。
  
  - 虽然有人为该模型辩护，但其他人指出了主要的缺失，例如缺乏打字动画。
- **AI 在医患沟通技巧上超越人类**：**成员们注意到**有报告表明 AI 比人类医生表现出更好的医患沟通技巧（bedside manner），引发了关于 AI 共情能力的讨论。
  
  - 一个黑色幽默式的转折出现了，人们质疑医疗专业人员的反社会人格特征是否会无意中导致更优的决策。
- **知识产权限制创新**：讨论强调了**知识产权**法律如何限制 AI 领域的创新，引发了对货币化和诉讼风险的担忧。
  
  - 创造力与所有权之间的紧张关系凸显了法律框架可能如何阻碍 AI 的革命性进步。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **ComfyUI 成为焦点**：成员表示 **ComfyUI** 是使用 Flux 的首选，而 **Automatic1111** 则被推荐给想要开始使用 Stable Diffusion 的初学者。建议还包括使用 **PyTorch** 或 **Diffusers** 进行命令行界面工作。
  
  - 这突显了用户在寻求更好的 AI 生成工作流时，在工具偏好上的广泛趋势。
- **AMD GPU 面临 AI 测试困扰**：一位成员表达了对 AMD GPU 缺乏 **CUDA** 支持的沮丧，并提到了 Python 开发中的困难。分享了针对拥有 8GB 或更多 VRAM 的 AMD GPU 用户使用 **ZLUDA** 版本的指南。
  
  - 讨论围绕着 AMD 硬件适配 AI 工作负载的阵痛展开，这一点正变得越来越关键。
- **3060 Ti 在 Stable Diffusion 中表现出色**：确认 **3060 Ti** 在 Stable Diffusion 中表现良好，并建议通过放大图像来提升质量，尽管它有 8GB VRAM 的限制。成员们分享了量化（quantizations）和分块放大（tiled upscaling）等技术以获得更好的输出。
  
  - 这标志着中端 GPU 在高效 AI 生成配置中持续发挥作用。
- **Lora 触发词管理受到关注**：一位用户询问了记忆 Lora 触发词的有效策略，以及是否有自动化的管理方式。这引发了关于 Lora 使用复杂性的全面讨论。
  
  - 对处理这些触发词的系统化方法的需求，反映了用户在提升 AI 生成保真度方面面临的日益增长的挑战。
- **讨论通过模型合并提升质量**：关于模型合并与连续处理相比的优点的讨论非常热烈，成员们探索了扩散步骤中特定的 **sigma** 值。共识认为，合并两个模型可以平均它们的能力，从而获得平衡的性能。
  
  - 这些见解突显了大家在模型增强方法论上的共同追求。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **准备 GPU 工程师实习**：一位成员请求关于 **GPU 工程师实习**的**资源和建议**，指出扎实的 **CUDA** 背景非常重要，并预估测试形式将包括**多选题**和**编程任务**。
  
  - 这一寻求指导的呼声表明，对于进入 GPU 领域的准工程师来说，导师指导和针对性资源的需求很大。
- **寻求 cuDNN SDPA 实现资源**：关于使用 Python 中 **cuDNN 的 SDPA** 实现 **Attention 层**的**教程或实现**的咨询，说明了在实例化过程存在困惑的情况下，社区对更好资源的需求。
  
  - 一位成员指向了 **cudnn-frontend 仓库**中的一个 notebook 以提供进一步帮助，强调了故障排除过程中的协作性质。
- **在受限 GPU 上优化 Llama 7B 训练**：在 **16GB** GPU 上训练需要 **28GB** 显存的 **Llama 7B** 被强调为具有挑战性，从而引发了关于利用 [FSDP2](https://github.com/pytorch/pytorch) 和 **activation checkpointing** 等工具的建议。
  
  - 成员们提出了使用 **CPU offload 优化器**的建议，展示了社区在管理有限资源的同时进行 fine-tuning 的适应策略。
- **ROCm 的新 Windows 支持**：ROCm 从 **6.3** 版本开始引入了对 **Windows 的原生支持**，显著扩大了 AMD 用户接触 GPU 技术的机会，正如最近的一篇 [GitHub issue](https://github.com/pytorch/pytorch/issues/106608) 所述。
  
  - 这一特性的发布引发了关于 ROCm 兼容性文档清晰度的讨论。
- **Guangxuan Xiao 讨论 StreamingLLM**：即将举行的 **PyTorch Expert Exchange** 将由 Guangxuan Xiao 主讲 [StreamingLLM](https://github.com/mit-han-lab/streaming-llm)，定于 **PST 时间 10 月 11 日上午 10 点**。
  
  - 随附的 [YouTube 视频](https://www.youtube.com/watch?v=RnM84Sv9WpA) 详细阐述了*带有 Attention Sinks 的高效流式语言模型*，展示了该领域的实际应用。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Llama 3.2 Fine-tuning 问题引发关注**：用户报告在对 **Llama 3.2 1B 模型**进行全量 fine-tuning 时出现冻结，可能是由于所用数据集的 NCCL 问题导致的。
  
  - 另一位成员指出在 **Llama 3 8B QLoRA** 上取得了成功，暗示该问题可能是由于配置引起的。
- **比 Groq 更快的新 Speculative Decoding 算法**：一位成员强调了他们新的 **speculative decoding 算法**超越了 Groq，引发了对更多技术细节的兴趣。
  
  - 成员们表达了探索这一资源效率进步的渴望。
- **探索 O1 的用例**：关于 **O1** 最佳用例的咨询指出其在编程方面的有效性，但成员们注意到其主要优势在于**数学**。
  
  - 回复确认了其在编程任务中的效用有限，引发了对其通用性的质疑。
- **O1 与 GPT-4o 的性能对比分析**：私人评估显示，**GPT-4o** 在直接回答任务中表现优于 **O1**，尤其是在复杂的**数学练习**中。
  
  - 尽管如此，**O1 Mini** 在编程方面比 **GPT-4o** 略胜一筹，而 **O1 Preview** 在 **PAL 方法**中表现出色。
- **OpenAI 的提示词生成元提示 (Metaprompt)**：一位成员讨论了 **OpenAI 用于系统提示词生成的 metaprompt**，暗示即将与 DSPy 进行集成。
  
  - [OpenAI 文档](https://github.com/openai/mle-bench/)的链接提供了对不断发展的 methodology 的见解。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **社区参与度表现亮眼**：成员们互相问候，营造了友好的氛围，热情的招呼促进了开放的对话。
  
  - 聊天内容反映了一个欢迎互动的环境，鼓励参与者之间的交流与联系。
- **Web Search Connector 详解**：关于启用 **Internet search tool** 的咨询揭示了文档中的困惑，引发了关于其在 **v1 API** 中可用性的讨论。
  
  - 迁移选项详见 [Cohere migration guide](https://docs.cohere.com/docs/migrating-v1-to-v2#web-search)，强调了用户过渡到 v2 时的差异。
- **V2 API 速度慢于预期**：用户注意到 **v2 API** 性能较慢，平均响应时间为 **2-3 秒**，而 v1 为 **1-1.5 秒**。
  
  - 这种延迟已被多次报告，引发了对其影响用户体验的担忧。
- **Token 使用讨论引发争议**：关于在 API 请求中使用特定 token 必要性的提问引发了对其响应质量影响的讨论。
  
  - 澄清建议指出，理解 token 要求对于有效使用 API 至关重要，尽管一些用户对其必要性表示怀疑。
- **Cohere API Toolcall 问题解决**：一位用户报告了关于 toolcall 的 **Cohere API** 性能问题，但发现相关的 **GitHub issue** 已被关闭。
  
  - 他们在运行 **5.11.0** 版本时寻求未解决问题的见解，反映出需要社区提供更清晰的解决方案。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Zoom 总部的 AI Builders Night 备受关注**：周一请加入我们在圣何塞 [Zoom HQ](http://developers.zoom.us) 举办的 **AI Builders Night**，届时来自 [LlamaIndex](https://www.llamaindex.ai/) 的 **Biswaroop Palit** 将讨论 **multi-agent systems** 以及来自 **QDrant** 的见解。
  
  - 与同行开发者建立联系，并围绕最新的 AI 进展展开讨论。
- **Lightning Demos 征集创新方案**：在聚会的 **lightning demos** 中，使用 [Zoom Developer Platform](http://developers.zoom.us) 展示你的 **AI-powered use cases**。
  
  - 这是一个获取反馈的绝佳机会，欢迎在社交媒体上使用 **#ZoomDevelopers** 分享亮点。
- **Symphony 加速工作流自动化**：**Symphony** 自动化 agentic workflows，根据你的工具和任务生成高性能配置，并鼓励加入其 [Discord](https://discord.gg/eYZESR4nVG) 获取 API key。
  
  - 查看此 [Loom video](https://www.loom.com/share/8d613aa434cf4a829e93160d01df35ae?sid=5216da3d-dcad-461c-bd37-6ba6a3c882b9) 以获取有关创建高效 AI 工作流的详细见解。
- **OpenAI Batch API 不适用于文档摘要**：成员们讨论了在 LlamaIndex 的 Document Summary Index 中使用 **OpenAI Batch API**，结论是它不符合效率的操作标准。
  
  - 社区对冗长的索赔过程表达了些许挫败感，强调了对更快捷方法的偏好。
- **AI Mayhem V3 黑客松招募赞助**：来自 Zo World 的代表正在为在旧金山和班加罗尔举行的 **AI Mayhem V3** 黑客松寻求赞助商，强调了品牌曝光机会。
  
  - 他们鼓励联系洽谈合作，旨在吸引顶尖开发者参与这一双城盛事。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **多节点部署变得简单**：对于大型多节点部署，建议使用 **AWS**，因为它可以确保在同一区域内实现更好的管理和连接性。
  
  - 这种方法为扩展和处理资源需求提供了一个更有效的系统。
- **对 Llama-3-8B 微调的挫败感**：一位成员分享了在两块 **3090 GPU** 上微调 **Llama-3-8B** 的经验，报告称与单 GPU 设置相比没有速度优势。
  
  - 尽管两块 GPU 的利用率都超过了 **98%**，但人们对 **DeepSpeed** 的数据并行有效性提出了质疑。
- **用于字符级的自定义 Llama Tokenizer**：通过子类化并重写 `tokenize` 方法来定制 **LlamaTokenizer** 以生成单字符 token，从而增强字符串处理能力。
  
  - 该方法特别旨在优化针对分子设计等任务的 LLM。
- **针对字符级 Tokenization 的调整**：在字符级别进行 tokenizing 可能需要调整模型的最大序列长度，从而影响训练和推理性能。
  
  - 这些调整可能会显著影响模型部署的整体效率。
- **SMILES 字符串处理演示**：一位成员展示了 tokenizer 如何处理 **SMILES 字符串**，展示了在分子表示中的实际应用。
  
  - 虽然 tokenizer 修改带来的变化可能很小，但在推进处理技术方面仍被认为值得关注。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Batch-GPT 大幅削减 API 成本**：一位成员强调了 Batch-GPT 工具，该工具通过其 Batch API 将 OpenAI API 成本降低了 **50%+**，促进了成本效益高的实施。
  
  - 这个开源项目具有针对重复查询的自动缓存功能，通过代码片段简化了集成：`client = OpenAI(..., base_url='http://batch-gpt/v1')`。
- **DSPy 入门表单提升用户体验**：引入了 DSPy 的入门表单来引导新用户了解其功能，提高理解和利用率。
  
  - 这一过程中的自动化前景与关于增强用户体验和未来 **AGI** 能力的讨论联系在一起。
- **OpenAI 拥抱 DSPy 优化**：有消息称 OpenAI 打算在其服务中实施 **DSPy 优化**，这表明其正转向更好的性能和效率。
  
  - 社区成员反应积极，对未来 OpenAI 迭代中潜在的增强功能感到兴奋。
- **GraphIC 提升 In-Context Learning**：讨论了 [GraphIC 方法](https://arxiv.org/abs/2410.02203)，该方法采用基于图的表示和 **Bayesian Networks** 来改进 **In-context Learning (ICL)**。
  
  - 该技术克服了传统 ICL 方法中的偏差，专注于复杂任务所需的更深层推理结构。
- **处理 LLM 分类中的歧义**：一位使用 DSPy 训练 LLM 分类器的成员分享了模型需要指示分类歧义的需求，例如：*需要更多信息，类别 A 和 B 之间存在歧义*。
  
  - 这引发了一场关于是否应该为所有歧义创建单独类别的对话，以解决分类结果的细微差别。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Int64 索引引发精度辩论**：针对仅在可能超出范围的 ALU 上应用 **int64 索引**的讨论已经展开，参考自 ##6987。
  
  - *Tinygrad* 对两种不同数据类型混合使用表示担忧，促使考虑算子兼容性。
- **GPU 速度缓慢促使探讨数据类型转换**：关于 **int64** 在 GPU 上运行缓慢的担忧浮出水面，引发了关于不同数据类型之间转换必要性的讨论。
  
  - 团队同意仅在严格必要时使用 **int64** 索引，以提升整体性能。
- **nn/init.py 需要类型注解**：成员们强调 **nn/init.py** 中的所有类都需要 **类型注解 (type annotations)** 以提高清晰度。
  
  - George 建议，对于旨在解决此增强功能的贡献者来说，这可以作为一个非常有前景的第一个 Pull Request。
- **Diffusion Policy 在机器人学习中表现出色**：关于 [Visuomotor Policy Learning via Action Diffusion](https://diffusion-policy.cs.columbia.edu) 的论文显示，**Diffusion Policy** 在机器人行为生成方面产生了 **46.9%** 的平均优势。
  
  - *它巧妙地处理了多模态动作分布和高维动作空间*，利用随机 Langevin 动力学实现稳定训练。
- **讨论精简示例文件的偏好**：在组织 `examples/` 目录时，George 表示更倾向于**单文件**形式，并强调**高质量**代码。
  
  - 这一反馈支持创建连贯的示例，以增强理解。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **BitNet 模型实现的巧妙思路**：一位成员探索了如何通过矩阵加法而非乘累加 (multiply-accumulate) 来实现 **1.58B BitNet 模型**，旨在提升在 **NVIDIA GPU** 上的性能。
  
  - 会上指出，利用 **Tensor Cores** 将提高效率，而利用整数运算可以进一步优化模型。
- **Gemma-2 遭遇微调瓶颈**：关于 **Gemma-2** 及其多语言能力的讨论日益升温，但微调在 **QLora** 实现中仍面临挑战。
  
  - 针对最佳参数选择的担忧开始出现，并已发起一个 [GitHub issue](https://github.com/pytorch/torchtune/issues/1813) 以寻求改进微调的支持。
- **Pixtral 12B 成为焦点**：关于 [Pixtral 12B](https://arxiv.org/abs/2410.07073) 的论文强调了其在多模态 AI 方面的能力，该论文由包括 **Pravesh Agrawal** 在内的团队共同撰写。
  
  - 它强调了自然图像与文档的融合，旨在竞争激烈的环境中取得领先性能。
- **Aria 树立多模态新标准**：[Aria](https://arxiv.org/abs/2410.05993) 作为一个开放的多模态原生模型出现，凭借其 **3.9B** 总参数和 **3.5B** 激活参数展现出顶尖性能。
  
  - 它超越了 **Pixtral-12B** 和 **Llama3.2-11B**，展示了在**语言理解**和更广泛任务效率方面的飞跃。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **复制 OpenAI O1 的技术见解**：一份新报告提出了用于复制 OpenAI **O1 模型**的 **“旅程学习” (journey learning)** 范式，展示了仅使用 **327 个训练样本**即可实现 **8% 的提升**。该报告提供了在整个复制过程中使用的深入观察和技术，重点关注高级推理能力。
  
  - 此次探索强调了试错学习策略以及它们如何增强模型性能，正如有关数学推理集成的讨论中所记录的那样。
- **对 dowehaveopeno1.com 提案的怀疑观点**：有人建议建立 **dowehaveopeno1.com** 作为 O1 复制更新的资源，但这引发了对其可行性的怀疑。社区成员表达了复杂的感受，承认有进展，但质疑创建该域名的时机是否成熟。
  
  - 对话揭示了对该域名在现阶段是否有益的担忧，考虑到 O1 复制工作仍在进行中。

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Gorilla LLM 的令人兴奋的进展**：成员们对 **Gorilla LLM** 模型最近的增强表示感谢，并鼓励为与其 handler 相关的 [PR](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#contributing) 提交代码。
  
  - 讨论强调了来自其他提供商的现有 PR 是促进贡献的有用参考。
- **精简的贡献流程**：分享了一份详细的 [README](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#contributing)，以指导用户如何有效地为 Gorilla 项目做出贡献。
  
  - 该文档包含了专门针对 **function calls** 训练和评估 LLM 的步骤。
- **Symphony 让 AI 工作流变得简单**：**Symphony** 模型通过将用户描述转换为功能性的 AI 工作流，简化了 **agentic workflows** 的创建，如这段 [Loom 视频](https://www.loom.com/share/8d613aa434cf4a829e93160d01df35ae?sid=5216da3d-dcad-461c-bd37-6ba6a3c882b9) 所示。
  
  - 社区成员还受邀加入 **Discord** 以申请 API key，从而加强项目协作，访问详情请见 [此处](https://discord.gg/eYZESR4nVG)。

 

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Web 浏览器 Agent 引起关注**：用户正在讨论有效的 **web browser agents**，其中 **Web Voyager** 作为一个值得进一步研究的有力竞争者浮出水面。
  
  - 成员们表达了分享这些 Agent 实际操作经验的热情，以推动集体见解。
- **寻找实验学习材料**：一位成员寻求关于实验最佳学习方法的指导，引发了关于利用 **slides 和补充阅读材料** 的讨论。
  
  - 对话强调了这些材料在有效准备实验工作中的关键作用。

 

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Raspberry Pi 5 上对轻量级向量数据库的需求**：一位成员强调了在 **Raspberry Pi 5** 上由于 RAM 资源有限，需要一个 **轻量级** 向量数据库来支持 **RAG** 设置。
  
  - 他们担心 **Chroma** 的 RAM 存储方式在与 **Ollama** 集成时会负面影响性能。
- **推荐使用 Pinecone 满足向量数据库需求**：作为回应，另一位成员建议将 **Pinecone** 作为 Raspberry Pi 5 场景下的实用向量数据库替代方案。
  
  - 该建议直接旨在减轻在此硬件环境下使用 **Chroma** 所带来的限制。

 

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **计算 ElevenLabs 音频成本**：一位成员分享道，加入 **ElevenLabs 的 creator plan** 每月可获得 **100k credits**，相当于每分钟音频 **833 credits** 或 **0.18 美元**。
  
  - 这一见解揭示了使用该应用进行音频制作时的成本影响。
- **关于 op3nai 实时 API 集成的咨询**：另一位成员提出了关于将 **op3nai 实时 API 成功实现到 O1** 中的问题。
  
  - 这一咨询强调了社区对分享 API 集成相关经验及所面临挑战的兴趣。

 

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Hugging Face AI21-Jamba-1.5-Mini 在 CUDA 上失败**：一位用户在 **Ubuntu** 环境下使用 **CUDA 12.4** 的 Docker 容器运行 `torch.multiprocessing` 时，遇到了 **Hugging Face** 模型 **AI21-Jamba-1.5-Mini** 的错误。
  
  - 错误指出 CUDA 无法在 fork 的子进程中重新初始化，强调了采用 'spawn' 启动方法的重要性。
- **在 Akash 上使用 A100 GPU 运行 Docker 的困扰**：另一位用户报告了在 **Akash** 上利用两块 **A100** GPU 运行 **Docker image** 时的问题，但关于其配置的具体细节较少。
  
  - 他们对持续存在的配置挑战及其对工作流的影响表示沮丧。

---

**Alignment Lab AI Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

**LAION Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

# PART 2: 按频道详细摘要和链接

{% if medium == 'web' %}

### **Notebook LM Discord ▷ #**[**announcements**](https://discord.com/channels/1124402182171672732/1182376564525113484/1294057318090281051) (1 messages):

> - `Audio Overviews`
> - `Feature Performance Issues`

- **Audio Overviews 生成问题**：目前正在调查 **Audio Overviews** 可能无法生成的问题，这可能会影响其他功能的性能。
  
  - 团队将在努力解决此问题时提供更新。
- **对其他功能的潜在影响**：成员们担心 **Audio Overviews** 的问题可能会对其他功能的性能产生更广泛的影响。
  
  - 团队已意识到这些担忧，并正在积极调查功能之间的相互影响。

---

### **Notebook LM Discord ▷ #**[**use-cases**](https://discord.com/channels/1124402182171672732/1124403655819415592/1294014025679638549) (36 条消息🔥):

> - `NotebookLM 用于教育`
> - `利用播客功能播报 Discord 聊天内容`
> - `多模态 AI 解读`
> - `分析音乐与声音`
> - `梦境日记与 AI 分析`

- **NotebookLM 用于家庭教育**：一位参与者表示有兴趣使用 NotebookLM 运行教学书籍，以使 13 岁孩子的家庭教育更具吸引力。
  
  - 然而，其他人提醒说，播客输出可能缺乏深度，并可能导致幻觉（hallucination）式的不准确。
- **从 Discord 对话创建播客**：成员们讨论了从 Discord 对话生成播客的想法，指出这可以为他们的聊天提供一种有趣的探索方式。
  
  - 一位用户幽默地评论说，可以将搞怪的 Discord 聊天内容喂给 AI 以用于制作播客。
- **多模态 AI 解读的见解**：一位用户分享了他们的研究，即 NotebookLM 如何通过播客解读复杂的历史混合媒体艺术。
  
  - 他们观察到了命令工具和元数据的使用，表明结果各异，特别是关于来自 YouTube 的音乐分析。
- **AI 的声音分析能力**：另一位用户分享了他们的实验，分析 TTS 模型在音频摘要中捕捉到的非语言发声和情感，并强调这是一项持续进行的工作。
  
  - 这一探索旨在进一步了解 Google 的 TTS 模型在 NotebookLM 框架内的能力。
- **AI 质询梦境日记**：一位成员询问如何使用 AI 分析梦境，并从个人梦境日记中提取反复出现的主题和叙事。
  
  - 这展示了 AI 分析在个人反思中的多样化应用，鼓励其他人考虑类似的用途。

**提到的链接**：

- [Google's NotebookLM takes on The Bootymachine - How AI understand multimodal art by AI's Hit - artificial intelligence hits on things](https://podcasters.spotify.com/pod/show/aishit/episodes/Googles-NotebookLM-takes-on-The-Bootymachine---How-AI-understand-multimodal-art-e2pg722)：利用 2007 年 Bootymachine 的线上线下多模态艺术体验，并使用名为《The Bootymachine》（ISBN 978-2-940679-01-0）的出版书籍 PDF 文件以及一些音频，我们……
- [GitHub - GilgameshofUT/AIResearcher](https://github.com/GilgameshofUT/AIResearcher)：通过在 GitHub 上创建账户，为 GilgameshofUT/AIResearcher 的开发做出贡献。

---

### **Notebook LM Discord ▷ #**[**general**](https://discord.com/channels/1124402182171672732/1124402182909857966/1294013390116884551) (591 条消息🔥🔥🔥):

> - `NotebookLM 音频生成`
> - `AI 幻觉`
> - `播客质量的概念`
> - `NotebookLM 用户体验`
> - `社区对 AI 工具的参与度`

- **NotebookLM 音频的用户体验**：用户正在分享他们使用 NotebookLM 音频生成功能的体验，经常提到输出结果虽然有趣但质量不稳定，包括出现幻觉和重复短语的情况。
  
  - 一位用户幽默地提到在播客期间遇到了明显的片段跳过和声音变化，突显了该工具不可预测的特性。
- **对 AI 和播客质量的担忧**：Listen Notes 的一篇博客文章引发了人们的担忧，即 AI 生成的播客可能会以低质量内容充斥平台，引起了真实创作者的忧虑。
  
  - 用户讨论了这种风险，但也承认无论如何开源替代方案都会涌现，因此很难完全消除这个问题。
- **NotebookLM 的功能与局限性**：讨论引发了对 NotebookLM 能力的好奇，包括它如何处理超长文档，以及用户是否可以对音频工具进行提示（prompt）以获得特定输出。
  
  - 用户指出了在导入过程中遇到的挑战，特别是表格的数据格式不正确，导致难以确保准确性。
- **社区参与和反馈**：用户表达了参与 NotebookLM 的热情，并分享了如何最大化利用其功能的技巧，包括深度探讨播客。
  
  - 一位用户指出了 AI 输出的喜剧价值，认为尽管存在幻觉，整体体验仍然令人愉悦。
- **AI 在播客领域的未来方向**：一位用户强调了 NotebookLM 潜在的未来发展，包括交互式功能和增强的用户对音频输出的控制。
  
  - 社区对这些创新将如何改变他们与 AI 生成内容的互动表现出了浓厚兴趣。

- [NotebookLM：播客界的威胁](https://www.listennotes.com/blog/notebook-lm-a-threat-to-the-podcasting-world-79/)：对听众不利。对播客创作者不利。对广告商不利。对托管平台不利。
- [使用 Google 的 NotebookLM 构建 AI 聊天机器人。](https://medium.com/@duncanrogoff/build-ai-chatbots-with-googles-notebooklm-e10a87d50f83)：AI 正在改变我们的沟通、工作和创作方式。在这些创新的前沿，有一款来自 Google 的迷人工具——NotebookLM，它现在正展示其……
- [鼓掌 Applause GIF - 鼓掌 拍手 - 发现并分享 GIF](https://tenor.com/view/applause-applaud-clap-clapping-proud-gif-17643067)：点击查看 GIF
- [Pawn Stars Rick GIF - Pawn Stars Rick 严肃 - 发现并分享 GIF](https://tenor.com/view/pawn-stars-rick-serious-wtf-thinking-gif-17924513)：点击查看 GIF
- [格林奇姿势 GIF - Grinch Pose Fab - 发现并分享 GIF](https://tenor.com/view/grinch-pose-fab-gif-2149309550277618787)：点击查看 GIF
- [Uh Oh GIF - Uh Oh - 发现并分享 GIF](https://tenor.com/view/uh-oh-gif-22939566)：点击查看 GIF
- [Kitty Forman Debra Jo Rupp GIF - Kitty Forman Debra Jo Rupp 大笑表情包 - 发现并分享 GIF](https://tenor.com/view/kitty-forman-debra-jo-rupp-laugh-meme-hysterical-laughter-lol-gif-24893555)：点击查看 GIF
- [Flight Reacts 印象深刻 GIF - Flight Reacts 印象深刻 鼓掌 - 发现并分享 GIF](https://tenor.com/view/flight-reacts-impressed-clapping-tongue-excitement-gif-26056891)：点击查看 GIF
- [Huh GIF - Huh - 发现并分享 GIF](https://tenor.com/view/huh-gif-23918002)：点击查看 GIF
- [Facts Straight GIF - Facts Straight Up - 发现并分享 GIF](https://tenor.com/view/facts-straight-up-gif-21244543)：点击查看 GIF
- [Golden Girls Sophia Petrillo GIF - Golden Girls Sophia Petrillo Sophia - 发现并分享 GIF](https://tenor.com/view/golden-girls-sophia-petrillo-sophia-picture-this-picture-it-gif-13790955)：点击查看 GIF
- [Waynesworld GIF - Waynesworld Way - 发现并分享 GIF](https://tenor.com/view/waynesworld-way-gif-5800555)：点击查看 GIF
- [巫师 Letho GIF - Witcher Letho 可接受 - 发现并分享 GIF](https://tenor.com/view/witcher-letho-acceptable-nodding-yes-gif-16328081)：点击查看 GIF
- [未找到标题](https://notebooklm.google.com/notebook/86a7e7eb-7dcd-4fb0-b931-50aa6f11c9ea/audio)：未找到描述
- [深入探讨 Deep Dives](https://on.soundcloud.com/hZjABNsNGnbnqnkXA)：在 #SoundCloud 上收听 Drew Walton 的 Deep Dive Into Deep Dives #np
- [Nobody Aint Got Time Wearing These Bad Boys GIF - Nobody Aint Got Time Wearing These Bad Boys 睡衣 - 发现并分享 GIF](https://tenor.com/view/nobody-aint-got-time-wearing-these-bad-boys-pajamas-roundhouse-kick-gif-16903363)：点击查看 GIF
- [产品背后：NotebookLM | Raiza Martin (Google Labs AI 高级产品经理)](https://www.youtube.com/watch?v=sOyFpSW1Vls)：Raiza Martin 是 Google Labs 的 AI 高级产品经理，她领导着 NotebookLM 背后的团队。NotebookLM 是一款 AI 驱动的研究工具，包含一个令人愉悦的……

### **Unsloth AI (Daniel Han) ▷ #**[**general**](https://discord.com/channels/1179035537009545276/1179035537529643040/1294021470049140860) (259 条消息🔥🔥):

> - `多模态模型支持`
> - `微调技术`
> - `H100 集群建议`
> - `Gemma 9B 训练挑战`
> - `Arcee SuperNova-Medius 模型`

- **对多模态模型的期待**：社区正热切期待对 **Llama3.2** 和 **Qwen2 VL** 等多模态模型的支持，预计下周将发布更新。
  
  - *Theyruinedelise* 对这一进展表示兴奋，这也是用户们高度期待的功能。
- **微调与数据集优化**：成员们讨论了微调 **G2-9B** 等模型的策略，指出了显著的 VRAM 需求以及使用 **Dora** 进行微调的有效性。
  
  - **Gemma 9B** 出现了一些挑战，包括高 VRAM 占用问题以及训练过程中出现 NaN 值。
- **H100 集群建议**：用户分享了使用 **H100 集群** 的经验，成本约为 **$2.5/小时**，并强调了确保充足 VRAM 进行训练的必要性。
  
  - 对于需要大量资源进行 AI 训练的用户，**Runpod** 等选项备受青睐。
- **模型的部署与使用**：关于模型合并可行性的担忧被提出，例如将 **Qwen 1.5** 与 **yi coder** 进行比较，以及向 **2.5** 版本的演进。
  
  - 用户被告知，训练并没有唯一的“正确” notebook 或设置，强调了模型调优中的试错过程。
- **介绍 Arcee SuperNova-Medius**：**Arcee SuperNova-Medius**（一个 14B 模型）正式亮相，声称其性能可媲美大得多的模型，引发了对其训练方法的关注。
  
  - 讨论围绕测试该模型的能力以及支持其开发的开源工具 **DistillKit** 展开。

**提到的链接**：

- [Introducing SuperNova-Medius: Arcee AI's 14B Small Language Model That Rivals a 70B](https://blog.arcee.ai/introducing-arcee-supernova-medius-a-14b-model-that-rivals-a-70b-2/)：首先是我们旗舰级的 70B SuperNova，随后是 8B SuperNova-Lite。今天，随着 14B SuperNova-Medius 的发布，我们为这个超强小语言模型家族增添了新成员。
- [Google Colab](https://colab.research.google.com/github/oobabooga/text-generation-webui/blob/main/Colab-TextGen-GPU.ipynb#scrollTo=LGQ8BiMuXMDG)：未找到描述
- [Nerding Speech Bubble GIF - Nerding Speech Bubble Pepe Nerd - Discover & Share GIFs](https://tenor.com/view/nerding-speech-bubble-pepe-nerd-gif-26077806)：点击查看 GIF
- [Lambda | GPU Compute for AI](https://lambdalabs.com/)：为 AI 开发者构建的 GPU 云。提供用于 AI 训练和推理的按需及预留 NVIDIA H100、NVIDIA H200 和 NVIDIA Blackwell GPU。
- [LoRA Parameters Encyclopedia | Unsloth Documentation](https://docs.unsloth.ai/basics/lora-parameters-encyclopedia)：了解参数如何影响微调过程。
- [Continue](https://github.com/continuedev)：领先的开源 AI 代码助手。Continue 拥有 14 个代码库。在 GitHub 上关注其代码。
- [GitHub - arcee-ai/DistillKit at blog.arcee.ai](https://github.com/arcee-ai/distillkit?ref=blog.arcee.ai)：一个用于 LLM 蒸馏的开源工具包。通过在 GitHub 上创建账号为 arcee-ai/DistillKit 的开发做出贡献。
- [peft/src/peft/tuners/lora/config.py at main · huggingface/peft](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py)：🤗 PEFT：最先进的参数高效微调（Parameter-Efficient Fine-Tuning）。- huggingface/peft

---

### **Unsloth AI (Daniel Han) ▷ #**[**help**](https://discord.com/channels/1179035537009545276/1179777624986357780/1294022519099232458) (80 messages🔥🔥):

> - `Llama 3 Model Fine-Tuning` (Llama 3 模型微调)
> - `Using RAG with Embedding Models` (在 Embedding 模型中使用 RAG)
> - `Characterization of Animals with LLMs` (使用 LLM 进行动物特征描述)
> - `CUDA Out of Memory Errors` (CUDA 显存溢出错误)
> - `Gradient Accumulation Steps in Training` (训练中的梯度累积步数)

- **Llama 3 模型微调讨论**：成员们讨论了微调 Llama 3 模型的过程，包括使用动物描述来增强模型响应。
  
  - 有人提到使用动物特征的 Embedding 来检索相关结果，并指出模型在分类任务中使用 Embedding 的表现。
- **用于查询的 RAG 和 Embedding 模型**：强调了 RAG 配合 Embedding 模型在查询中的效用，建议在用适当数据微调后可以检索到有价值的见解。
  
  - 参与者指出，使用较短的描述可能会使 Embedding 分数产生偏差，从而影响检索任务的相关性。
- **CUDA 显存溢出与 RTX 4090 训练**：一位用户报告在 RTX 4090 上训练时遇到 CUDA 显存溢出 (Out of Memory) 错误，建议调整 Batch Size 和优化器设置。
  
  - 建议使用 paged_adamw_8bit 优化器并降低梯度累积步数 (Gradient Accumulation Steps)，以更好地管理显存。
- **训练设置中的效率挑战**：用户对训练时间过长表示担忧，提到某些配置导致单个 Epoch 需要数小时。
  
  - 几位用户协作寻找能在不牺牲性能的情况下优化训练时间的配置。
- **梯度累积步数的影响**：参与者讨论了调整梯度累积步数的影响，争论其对训练速度的作用。
  
  - 一位用户幽默地提到，尽管增加了梯度累积步数，他们的训练速度反而变快了。

**提到的链接**：

- [Google Colab](https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing)：未找到描述
- [Retrieve & Re-Rank — Sentence Transformers documentation](https://sbert.net/examples/applications/retrieve_rerank/README.html)：未找到描述
- [text_classification_scripts/unsloth_classification.ipynb at main · timothelaborie/text_classification_scripts](https://github.com/timothelaborie/text_classification_scripts/blob/main/unsloth_classification.ipynb)：使用 Llama 和 BERT 进行文本分类的脚本 - timothelaborie/text_classification_scripts
- [GitHub - MaartenGr/BERTopic: Leveraging BERT and c-TF-IDF to create easily interpretable topics.](https://github.com/MaartenGr/BERTopic)：利用 BERT 和 c-TF-IDF 创建易于解释的主题。 - GitHub - MaartenGr/BERTopic

---

### **Unsloth AI (Daniel Han) ▷ #**[**research**](https://discord.com/channels/1179035537009545276/1257011997250424842/1294107991125266504) (9 messages🔥):

> - `OpenAI's O1`
> - `Chain of Thought (CoT) reasoning`
> - `Cursor Team's Speculation`
> - `Anthropic/AWS stack`
> - `Closed Source Concerns`

- **关于 OpenAI O1 的推测引发辩论**：有人推测 OpenAI 的 **O1** 允许一系列 Prompt 在用户不可见的情况下运行，但尚未达成共识。
  
  - 讨论强调了对这些说法真实性的不同意见，一些成员对源码的封闭性表示怀疑。
- **通过 CoT 推理改进 LLM**：成员们认为在没有 Prompt 的情况下改进 **LLM** 的思维链 (Chain of Thought) 推理看起来很有前景，激发了关于创新模型的想法。
  
  - 有人提议将 CoT 放入注意力模型的 K/V Cache 中，为实验提供了思路。
- **Cursor 团队对 O1 的见解**：在观看相关视频时，一位成员发现 **Cursor 团队**的见解具有推测性，并因该产品的闭源性质而质疑其有效性。
  
  - 还讨论了由于 O1 的发布，**Anthropic/AWS 技术栈**可能会影响其产品性能的观点。
- **社区对 O1 功能的参与**：对话显示了对 O1 功能的复杂反应，一些人承认在视频中看到过类似信息。
  
  - 随后表达了轻松的不确定感，强调了这项技术发展过程中不可预测的本质。
- **对 OpenAI 透明度的担忧**：表达了对 **OpenAI** 缺乏透明度的看法，重点在于针对 O1 功能的推测性主张所带来的挑战。
  
  - 这种怀疑反映了社区对专有技术及其局限性影响的更广泛担忧。

---

### **HuggingFace ▷ #**[**general**](https://discord.com/channels/879548962464493619/879548962464493622/1294014355611844638) (235 条消息🔥🔥):

> - `风格化角色图像标注`
> - `情感检测模型`
> - `运行 Text-to-Code 模型`
> - `Hugging Face 订阅咨询`
> - `文本模型中的 Function calling`

- **角色图像标注的挑战**：一位成员正面临为超过 40 万张不同视角的风格化角色图像进行标注的挑战，并寻求有效的分类方法。其他成员建议，由于不一致性和当前模型的不足，手动分类可能是必要的。
  
  - 他们讨论了数据增强技术中涉及的复合误差，并指出在处理如此庞大的数据集时，人工审核的重要性。
- **探索情感检测模型**：一位用户询问了优秀的情感检测模型，并提到他们正在使用 FER 和 DeepFace 库。推荐和经验分享引发了关于某些模型在准确捕捉情感细微差别方面的局限性的讨论。
  
  - 成员们表达了对特定模型的挑战，以及在各种应用中需要准确情感输出的需求。
- **运行 Text-to-Code 模型**：另一位用户请求帮助运行来自 GitHub 上 CodeXGLUE 基准测试套件的 Text-to-Code 模型。会议澄清了该套件本身不是一个模型，而是一组基准测试，这表明可能存在混淆。
  
  - 成员们强调需要理解利用这些基准测试进行有效使用所涉及的背景和工具。
- **Hugging Face 订阅协助**：一位成员在寻求 Hugging Face 订阅方面的帮助，但在使用个人电子邮件时遇到了问题。回复建议通过指定的企业查询邮箱联系 Hugging Face。
  
  - 大家普遍认为，明确联系销售或支持团队的方式对于顺利入驻和解决相关问题至关重要。
- **文本模型中的 Function Calling 和用例**：围绕缺乏 Function calling 能力的文本模型的实用性展开了讨论，成员们探索了它们的潜在用例。有人指出，虽然这些模型的功能性可能较弱，但在搜索引擎和文档等各种应用中仍具有价值。
  
  - 成员们进行了比喻性的讨论，对比了模型和工具，阐述了它们在处理和响应任务中的作用。

**提到的链接**：

- [XLabs-AI/flux-dev-fp8 at main](https://huggingface.co/XLabs-AI/flux-dev-fp8/tree/main)：未找到描述
- [The Office Dwight GIF - The Office Dwight Joke - Discover & Share GIFs](https://tenor.com/view/the-office-dwight-joke-jim-identity-theft-gif-14240042)：点击查看 GIF
- [GitHub: Let’s build from here](https://github.com/)：GitHub 是超过 1 亿开发者共同塑造软件未来的地方。为开源社区做贡献，管理你的 Git 仓库，像专家一样审查代码，跟踪错误和功能...
- [Trelis/Meta-Llama-3-70B-Instruct-function-calling · Hugging Face](https://huggingface.co/Trelis/Meta-Llama-3-70B-Instruct-function-calling)：未找到描述
- [Python for Beginners – Full Course [Programming Tutorial]](https://www.youtube.com/watch?v=eWRfhZUzrAc)：在这门面向初学者的完整课程中学习 Python 编程语言！你将学习 Python 的基础知识，并逐行编写两个 Python 程序...
- [GitHub - ltdrdata/ComfyUI-Manager: ComfyUI-Manager is an extension designed to enhance the usability of ComfyUI. It offers management functions to install, remove, disable, and enable various custom nodes of ComfyUI. Furthermore, this extension provides a hub feature and convenience functions to access a wide range of information within ComfyUI.](https://github.com/ltdrdata/ComfyUI-Manager)：ComfyUI-Manager 是一个旨在增强 ComfyUI 易用性的扩展。它提供了安装、删除、禁用和启用 ComfyUI 各种自定义节点的管理功能。此外，该扩展还提供了一个中心（hub）功能和便捷功能，用于访问 ComfyUI 内部的广泛信息。
- [llama3](https://ollama.com/library/llama3)：Meta Llama 3：迄今为止功能最强大的开源 LLM
- [meta-llama/Meta-Llama-3-8B · The Serverless Inference API: "The model meta-llama/Meta-Llama-3-8B is too large to be loaded automatically (16GB > 10GB)"](https://huggingface.co/meta-llama/Meta-Llama-3-8B/discussions/31)：未找到描述

---

### **HuggingFace ▷ #**[**today-im-learning**](https://discord.com/channels/879548962464493619/898619964095860757/1294026257259954197) (3 messages):

> - `Community Computer Vision Course`
> - `Proof of Concept Chat Agent`
> - `NVIDIA Research on Larger LLMs`
> - `Mixture of Experts (MoE) Techniques`

- **探索社区计算机视觉课程**：一位成员分享了新的 [Community Computer Vision Course](https://huggingface.co/learn)，旨在教授使用 Hugging Face 库和模型的 ML 技术。
  
  - 该课程强调实际应用，旨在增强参与者的计算机视觉技能。
- **首个 POC 聊天 Agent 发布**：一位成员展示了他们首个支持工具调用的“概念验证（Proof of Concept）”聊天 Agent，可在 [此 GitHub 链接](https://github.com/Clay-Ferguson/quantizr/blob/main/QuantaGradio/Quanta_Gradio_AgentTest.py) 获取。
  
  - 该 Agent 集成了文档协作和微型博客功能，展示了 AI 聊天机器人与编程能力的结合。
- **NVIDIA 推动 LLM 训练边界**：来自 NVIDIA 的前沿研究揭示了通过 upcycling 技术改进 LLM 训练的方法；upcycled 的 Nemotron-4 15B 模型达到了 **67.6% MMLU**。
  
  - 团队为 MoE 提出了一种“虚拟组（virtual group）”初始化方法，利用了 NVIDIA Megatron-Core 的特性（如专家并行），并在该 [论文](https://arxiv.org/abs/2410.07524) 中分享了细节。
- **MoE 技术带来更佳性能**：NVIDIA 的研究表明，在对大型模型进行 upcycling 时，**softmax-then-topK** 方法优于 **topK-then-softmax** 方法。
  
  - 此外，他们指出了更细粒度 MoE 结构的优势，并提出了高效增加模型容量的新技术。

**提及的链接**：

- [Hugging Face - Learn](https://huggingface.co/learn)：未找到描述
- [quantizr/QuantaGradio/Quanta_Gradio_AgentTest.py at main · Clay-Ferguson/quantizr](https://github.com/Clay-Ferguson/quantizr/blob/main/QuantaGradio/Quanta_Gradio_AgentTest.py)：开源 CMS、文档协作、微型博客和发布系统，带有支持大多数云端 AI 提供商的 AI 聊天机器人和 AI 编程 Agent - Clay-Ferguson/quantizr
- [Ethan He (@EthanHe_42) 的推文](https://x.com/EthanHe_42/status/1844542533105500280)：我很高兴分享我们关于通过将 LLM upcycling 为混合专家模型 (MoE) 来改进 LLM 的最新研究！1. 我们在 1T token 上对 Nemotron-4 15B 模型进行了 upcycling，并将其与持续训练进行了比较...
- [Upcycling Large Language Models into Mixture of Experts](https://arxiv.org/abs/2410.07524)：将预训练的稠密语言模型 upcycling 为稀疏混合专家模型 (MoE) 是增加已训练模型容量的一种有效方法。然而，最优技术...
- [Megatron-LM/megatron/core/transformer/moe at main · NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe)：正在进行的关于大规模训练 Transformer 模型的研究 - NVIDIA/Megatron-LM

---

### **HuggingFace ▷ #**[**cool-finds**](https://discord.com/channels/879548962464493619/897390579145637909/1294027936076271669) (5 messages):

> - `Community Computer Vision Course`
> - `Text to Speech Research`
> - `Xtts Space by Coqui`
> - `Maximizing GPU Utilization`
> - `Tripo3D.ai Tool`

- **探索 Hugging Face 社区课程**：Hugging Face 提供了一个 [Community Computer Vision Course](https://huggingface.co/learn)，教授如何使用其生态系统中的库进行 **Computer Vision ML**。
  
  - 该课程旨在让参与者掌握 **Computer Vision** 应用的实际技能。
- **Text to Speech 的新见解**：一篇题为 [*2406.04904*](https://arxiv.org/abs/2406.04904) 的新论文专注于 **Text to Speech** 技术的进步，由多位研究人员共同撰写。
  
  - 该研究讨论了增强 **Text to Speech** 系统的创新方法和方法论。
- **探索 Coqui 的 Xtts Space**：Coqui 推出了 [Xtts space](https://huggingface.co/spaces/coqui/xtts)，展示了 **Text to Speech** 技术中令人兴奋的功能。
  
  - 该 Space 旨在通过新的工具和功能，刷新从文本生成语音的体验。
- **有效优化 GPU Utilization**：[Sushmitha 的文章](https://medium.com/@hssushmitha047/maximizing-gpu-utilization-while-training-models-a-practical-guide-6d78a04b506e) 提供了关于在模型训练期间最大化 **GPU Utilization** 的实用指南，强调了监控和优化设置的好处。
  
  - 该博客强调了适当的 GPU 使用如何显著加速实验并提高整体性能。
- **发现革命性的 3D 图形工具**：一位用户发现了 **Tripo3D.ai**，这是一个利用 AI 从文本或图像创建高度详细的 **3D models** 的工具，给早期测试者留下了深刻印象。
  
  - 该工具旨在为设计师和开发人员节省数小时的工作时间，使其成为创建 3D 资产的宝贵资源。

**提及的链接**：

- [XTTS - a Hugging Face Space by coqui](https://huggingface.co/spaces/coqui/xtts): 未找到描述
- [Maximizing GPU Utilization While Training Models: A Practical Guide](https://medium.com/@hssushmitha047/maximizing-gpu-utilization-while-training-models-a-practical-guide-6d78a04b506e): 作为一名 Machine Learning 爱好者，我在运行内存密集型实验时依赖的关键资源之一是 GPU。在这篇博客中……
- [Hugging Face - Learn](https://huggingface.co/learn): 未找到描述
- [XTTS: a Massively Multilingual Zero-Shot Text-to-Speech Model](https://arxiv.org/abs/2406.04904): 大多数 Zero-shot Multi-speaker TTS (ZS-TTS) 系统仅支持单一语言。虽然像 YourTTS、VALL-E X、Mega-TTS 2 和 Voicebox 这样的模型探索了 Multilingual ZS-TTS，但它们仅限于……
- [Tripo AI for Web](https://www.tripo3d.ai/): 未找到描述

---

### **HuggingFace ▷ #**[**i-made-this**](https://discord.com/channels/879548962464493619/897390720388825149/1294023260865822891) (10 条消息🔥):

> - `Distilabel Cost Calculation` (Distilabel 成本计算)
> - `Version 1.5.0 Refactoring` (1.5.0 版本重构)
> - `LLM Token Calculation` (LLM Token 计算)

- **为 Distilabel 引入成本计算**：一名成员分享了一个小工具包，用于在 **Distilabel** pipelines 中添加**成本计算**功能，已在 **TextGeneration** 和 **TextClassification** 任务上完成测试。
  
  - 他们计划更新 YAML 格式的价格选项，涵盖所有支持的 **LLM APIs**，从而方便用户使用。
- **重构输出以简化成本计算**：针对 **1.5.0 版本**展开了讨论，旨在重构 **LLM** 输出，以简化 **tokens** 和成本的计算。
  
  - 这一改进旨在利用现有的 **API** 功能，最大限度地减少对额外 **token** 计数工具的需求。
- **社区对新功能的热情**：成员们对分享的成本计算工具包及其对 **Distilabel** 工作流的潜在影响表示兴奋。
  
  - 另一位成员赞扬了这一倡议，并表示有兴趣改进 **token** 计算方法以提高效率。

**提到的链接**：

- [TimothyLovett | Pavement Model ViT | Kaggle](https://www.kaggle.com/models/timothylovett/pavement-model-vit/tensorFlow2/default/1)：未找到描述
- [SimpleTuner/documentation/DISTRIBUTED.md at main · bghira/SimpleTuner](https://github.com/bghira/SimpleTuner/blob/main/documentation/DISTRIBUTED.md)：一个面向 **diffusion** 模型的通用微调工具包。 - bghira/SimpleTuner
- [GitHub - ariasanovsky/cuts](https://github.com/ariasanovsky/cuts)：通过创建账号为 ariasanovsky/cuts 的开发做出贡献。
- [GitHub - djellalmohamedaniss/distilabel-cost-calculator: A custom Step for LLM API cost calculation for the distilabel library.](https://github.com/djellalmohamedaniss/distilabel-cost-calculator)：一个用于 **distilabel** 库的 **LLM API** 成本计算自定义 **Step**。 - djellalmohamedaniss/distilabel-cost-calculator
- [Diffing iPython notebook code in Git](https://blog.moonglow.ai/diffing-ipython-notebook-code-in-git/)：如今，我在软件开发中经常使用 **iPython notebooks**。这是一种无需启动 **pdb** 即可调试的好方法；当我尝试调试时，经常会用到它...
- [GitHub - moonglow-ai/pre-commit-hooks: Moonglow pre-commit hooks](https://github.com/moonglow-ai/pre-commit-hooks)：Moonglow **pre-commit hooks**。通过创建账号为 moonglow-ai/pre-commit-hooks 的开发做出贡献。
- [TimothyLovett | Birds 224x224 Shrunken EfficientNet | Kaggle](https://www.kaggle.com/models/timothylovett/birds-224x224-shrunken-efficientnet/TensorFlow2/default/1)：未找到描述

---

### **HuggingFace ▷ #**[**NLP**](https://discord.com/channels/879548962464493619/922424173916196955/1294037859161997332) (3 条消息):

> - `Runpod`
> - `Llama3`
> - `VLLM`
> - `Embedding models API`
> - `Flask API request batching`

- **Runpod、Llama3 和 VLLM 探索**：一位成员鼓励大家尝试 **Runpod**、**Llama3** 和 **VLLM**，并用笑脸表情营造了轻松的氛围。
  
  - 这种轻松的方式突显了*探索新工具的乐趣*。
- **用于嵌入模型的开源库**：有人询问有哪些可用的**开源**库可以用来推理 **BERT** 和 **all-miniLM** 等 **embedding** 模型。
  
  - 这一搜索反映了对能够简化与这些模型交互的工具的需求。
- **简化用于嵌入的 Flask API**：成员们分享了关于在没有**请求批处理 (request batching)** 的情况下运行 **Flask API** 可能产生的性能问题的担忧。
  
  - 强调了对**高效处理**的需求，暗示希望寻找能够**避免不必要的复杂性**的解决方案。

---

### **HuggingFace ▷ #**[**diffusion-discussions**](https://discord.com/channels/879548962464493619/1009713274113245215/1294161648017346611) (12 messages🔥):

> - `Diffusion processes for multi-channel data`（多通道数据的 Diffusion 过程）
> - `Training flux models on M2 Macs`（在 M2 Mac 上训练 Flux 模型）
> - `Image generation with different channel configurations`（不同通道配置下的图像生成）
> - `Effects of noise on image structure`（噪声对图像结构的影响）
> - `Bug fixing and community contributions`（Bug 修复与社区贡献）

- **Diffusion 过程需要考虑通道**：讨论集中在如何跨不同通道应用 Diffusion 噪声，特别是在处理生物序列或带有额外 Alpha 通道的图像时。
  
  - *他们如何处理代表完全不同信息的通道？* 参与者质疑相同的噪声调度（noise schedule）是否可以应用于包含不同信息的通道。
- **在 M2 Mac 上训练 Flux 模型仍不确定**：一位成员询问在 M2 Mac 上运行 `dreambooth-lora-flux` 训练是否可行，并对可能出现的信号量泄漏（semaphore leak）表示担忧。
  
  - 另一位成员建议训练脚本主要在 Linux 上测试，建议在 Apple 硬件上改用 SimpleTuner。
- **图像生成受高强度设置影响**：一位参与者分享了使用高强度设置进行 img2img 技术的经验，指出较低的强度可以实现更精确的定位。
  
  - 然而，他们警告说，强度过低可能会捕捉到像素细节，从而损害图像质量。
- **感谢社区对问题的贡献**：有人呼吁社区协助修复 Bug 并增强与 Apple 硬件的兼容性。
  
  - 成员们鼓励在 Mac 上使用训练脚本时，针对发现的任何修复方案提交 Pull Requests。
- **关于 SDXL 和 Flux Latents 的讨论**：分享了关于 SDXL 和 Flux Latents 的具体细节，指出 SDXL 有 4 个通道，分别配置为亮度（luminance）、青色/红色（cyan/red）、石灰绿/紫色（lime/purple）和结构模式（structure pattern）。
  
  - 据报道，调整模式/结构通道会影响图像的清晰度和小物体的存在。

 

**提到的链接**：[dreambooth_run - Pastebin.com](https://pastebin.com/BHYZszCc)：Pastebin.com 自 2002 年以来一直是排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。

 

---

### **HuggingFace ▷ #**[**gradio-announcements**](https://discord.com/channels/879548962464493619/1014577787039924226/1294165171195088928) (1 messages):

> - `Gradio 5 Release`（Gradio 5 发布）
> - `Performance Improvements`（性能改进）
> - `User Interface Enhancements`（用户界面增强）
> - `Security Upgrades`（安全升级）
> - `AI Playground Feature`（AI Playground 功能）

- **Gradio 5 隆重登场**：**Gradio 5** 的发布带来了迄今为止**最大**且**最安全**的更新，将数月的辛勤工作浓缩到此版本中，彻底改变了生产级应用的创建方式。
  
  - 正如公告中所强调的，这次更新使开发者能够仅用几行 Python 代码就构建出机器学习应用程序。
- **借助 SSR 实现闪电般的加载速度**：通过引入 **SSR**（服务端渲染），Gradio 5 提供了**闪电般的加载速度**，消除了加载动画，令开发者倍感欣喜。
  
  - 用户可以期待应用以空前的速度运行，使交互体验比以往任何时候都更加流畅。
- **惊艳的新 UI 设计**：此次更新采用了**极具美感的新 UI 设计和主题**，增强了使用 Gradio 5 创建的应用的视觉吸引力。
  
  - 这一转变旨在为用户提供精美的界面，在他们与机器学习应用交互时吸引受众。
- **坚如磐石的安全改进**：Gradio 5 经过了 **TrailOfBits** 的审计，显著提升了其安全措施，以确保机器学习应用更加安全。
  
  - 详细的安全审查可在其博客文章中找到，进一步强化了 Gradio 对最佳实践的承诺。
- **令人惊叹的 AI Playground 功能**：**AI Playground** 的引入允许用户利用 AI 通过交互式工具和功能构建 Gradio 应用程序。
  
  - 该功能旨在简化开发流程，使实验 Gradio 变得比以往任何时候都更加容易。

**提到的链接**：

- [Gradio 5 安全审查](https://huggingface.co/blog/gradio-5-security)：未找到描述
- [Gradio Playground](https://www.gradio.app/playground)：体验 Gradio Demo
- [欢迎来到 Gradio 5](https://huggingface.co/blog/gradio-5)：未找到描述

---

### **LM Studio ▷ #**[**general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1294025555053907998) (57 条消息🔥🔥):

> - `模型性能能力`
> - `LM Studio API 与功能`
> - `模型与 MLX 的兼容性`
> - `GPU 与模型加载问题`
> - `LM Studio 功能请求`

- **模型在各种任务中表现出色**：成员们确认新模型是通用型的，在几乎所有任务中的表现都与 **ChatGPT** 类似。
  
  - 这些模型结合了预训练和指令微调（instruct finetuned）的权重，适用于各种应用。
- **结构化输出 API 的使用**：提到 LM Studio 通过提供的 JSON schema，在 `/v1/chat/completions` 端点支持结构化 JSON 输出。
  
  - 建议成员查看 LM Studio 中的服务器选项卡，获取如何实现该功能的示例代码。
- **MLX 后端的兼容性与问题**：讨论强调了在使用 MLX 后端的 GPU 上加载模型的问题，特别是对于默认使用 CPU 的大型模型。
  
  - 成员建议在独立的 Apple MLX 中验证模型性能，并可能在 GitHub 上提交 issue 以寻求更广泛的帮助。
- **本地服务器连接请求**：社区讨论了未来支持 LM Studio 作为客户端连接到运行 OpenAI 兼容端点的本地服务器。
  
  - 分享了一个 GitHub issue 以跟踪此功能请求，鼓励其他人关注进度。
- **滚动条功能改进请求**：有请求希望更新版本的 LM Studio 包含滚动条，以便更轻松地浏览大型文档。
  
  - 建议包括悬停时动态显示滚动条，类似于 Windows 的滚动功能。

**提到的链接**：

- [Feature Request: Use LM Studio as a Client for a different LLM Server in the local Network. · Issue #133 · lmstudio-ai/lmstudio-bug-tracker](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/133)：LM Studio 已经允许创建服务器并用于 API 请求。但它目前不允许 LM Studio 作为该服务器的客户端。场景如下：我有一台性能强大的机器在我的...
- [GitHub - ml-explore/mlx: MLX: An array framework for Apple silicon](https://github.com/ml-explore/mlx)：MLX：适用于 Apple silicon 的数组框架。通过在 GitHub 上创建账号来为 ml-explore/mlx 的开发做出贡献。
- [GitHub - abetlen/llama-cpp-python: Python bindings for llama.cpp](https://github.com/abetlen/llama-cpp-python)：llama.cpp 的 Python 绑定。通过在 GitHub 上创建账号来为 abetlen/llama-cpp-python 的开发做出贡献。
- [Structured Output - Advanced | LM Studio Docs](https://lmstudio.ai/docs/advanced/structured-output)：结构化输出 - 高级 | LM Studio 文档。使用 JSON schemas 强制执行 LLM 响应格式。

---

### **LM Studio ▷ #**[**hardware-discussion**](https://discord.com/channels/1110598183144399058/1153759714082033735/1294155400354074704) (37 messages🔥):

> - `M1 Max 升级`
> - `RTX 5000 价格传闻`
> - `外部 e-GPU 限制`
> - `Mixtral 模型推荐`
> - `兼容性排序功能`

- **M1 Max 升级至 M3 Max**：一位用户分享了从标准 RAM 的 **M1 Max** 升级到拥有 **128GB RAM** 的 **M3 Max** 的经验，指出虽然价格昂贵，但在运行 LLM 方面表现完美。
  
  - *硬件讨论区中的许多用户正在通过升级设备来更有效地处理大型模型。*
- **Nvidia RTX 5000 价格冲击**：成员们讨论了新款 **RTX 5000 系列** 的传闻价格，估计每张显卡在 **1,500 美元到 2,500 美元** 之间，这可能使其比 Mac Studio 配置更便宜。
  
  - 用户对运行多张显卡的高昂成本表示担忧，特别是在发热和电力消耗方面。
- **外部 e-GPU 限制解析**：一位用户询问了通过 Thunderbolt 将外部 **e-GPU** 连接到现有 **RTX 4090** 以增加可用显存的可行性，但怀疑性能可能会滞后。
  
  - 讨论暗示 Thunderbolt 连接可能会引入延迟问题，从而限制了合并 GPU 显存带来的收益。
- **针对 M1 Ultra 的 Mixtral 模型推荐**：用户建议 **Mixtral 8x7b** 和 **Mixtral 8x22b** 模型适合拥有 **128GB RAM** 的 Mac 系统，并强调了它们在各种用例中的兼容性。
  
  - 一位用户指出，Mac 设备可以无缝运行大小高达 **120b Q4** 的模型。
- **兼容性排序功能的变更**：一位用户对 **0.3.0** 版本中缺失兼容性排序功能表示困惑，引发了关于与早期版本相比界面变化的讨论。
  
  - 用户澄清说，虽然新版本会自动处理兼容模型，但特定的“按兼容性排序”功能已不再提供。

**提到的链接**：

- [传闻中的 RTX 5000 GPU 价格泄露令人震惊——如果 Nvidia 打算要价高达 2,500 美元，就应该直接把 RTX 5090 称为 Titan](https://www.techradar.com/computing/gpu/rumored-rtx-5000-gpu-price-leaks-are-shocking-nvidia-should-just-call-the-rtx-5090-a-titan-if-its-going-to-charge-up-to-usd2-500-for-it)：那是由于最坏的情况，但最好的情况是……等一下……2,000 美元。是的，你没看错。
- [Pc Pc 爆炸 GIF - Pc Pc 爆炸 Pc 烧毁 - 发现并分享 GIF](https://tenor.com/view/pc-pc-explosion-pc-burn-pc-fire-computer-gif-2323271670262777828)：点击查看 GIF

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1294013298521673809) (46 messages🔥):

> - `可控语音技术`
> - `Hugging Face 评估指南`
> - `MLE Bench 基准测试`
> - `OpenWebUI 进展`
> - `AI Neoclouds 与 GPU 租赁动态`

- **来自 Wondercraft 的可控语音技术**：Wondercraft 推出了 [导演模式 (Director Mode)](https://x.com/wondercraft_ai/status/1844378469628772586)，允许用户指示其 AI 语音角色如何表达台词。
  
  - 此次发布被称为他们今年最大的更新，增强了用户的创意控制力。
- **Hugging Face 的新评估指南**：Clementine Fourrier 在 [GitHub](https://github.com/huggingface/evaluation-guidebook) 上分享了一份新的指南，整合了从管理 Open LLM Leaderboard 中获得的 LLM 评估见解。
  
  - 该资源为评估机器学习模型提供了实践和理论框架。
- **面向 AI Agent 的 MLE Bench**：OpenAI 推出了 [MLE-bench](https://openai.com/index/mle-bench/) 基准测试，通过源自 Kaggle 的竞赛来评估 AI Agent 在机器学习工程中的表现。
  
  - 这一举措展示了 AI 开发中对实际应用和性能指标日益增长的关注。
- **OpenWebUI 的热度与功能**：OpenWebUI 在其名为“Artifacts”的最新更新后引起了巨大关注，该更新支持完全本地和私有的 LLM 使用。
  
  - 用户注意到其强大的功能，将 OpenWebUI 定位为具有更大可访问性潜力的领先工具。
- **关于 AI Neoclouds 和 GPU 租赁动态的见解**：一篇热门的 [HN 帖子](https://news.ycombinator.com/item?id=41805446) 讨论了 GPU 租赁服务的新兴趋势，这些趋势受到供应过剩和 AI 算力需求变化的影响。
  
  - 该讨论强调了 AI 基础设施中不断演变的策略，以及与市场饱和相关的价格崩盘可能性。

**提到的链接**：

- [未找到标题](https://news.ycombinator.com/item?i): 未找到描述
- [Nick St. Pierre (@nickfloats) 的推文](https://x.com/nickfloats/status/1844788388710212046?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): Midjourney 即将发生一些非常特别的事情 —— v 7 版本发布、视频模型发布、Midjourney 3D 发布、具备 ControlNet 功能的外部图像新编辑器，全部即将推出...
- [Wondercraft (@wondercraft_ai) 的推文](https://x.com/wondercraft_ai/status/1844378469628772586): 推出导演模式 (Director Mode)。如果你能直接告诉你的 AI 语音角色如何表达台词会怎样？现在你可以了。在 Parrot Mode 取得成功后，我们将把音频工作室提升到下一个...
- [MLE-bench: 在机器学习工程上评估机器学习 Agent](https://arxiv.org/abs/2410.07095): 我们推出了 MLE-bench，这是一个衡量 AI Agent 在机器学习工程方面表现的基准测试。为此，我们从 Kaggle 精选了 75 个与机器学习工程相关的竞赛，创建了一个多样化的...
- [产品背后：Google 的 NotebookLM | Raiza Martin (Google Labs AI 高级产品经理)](https://www.lennysnewsletter.com/p/googles-notebooklm-raiza-martin): Google Labs 的 Raiza Martin 谈论领导 NotebookLM、“音频概览 (Audio Overviews)”功能以及扩展 AI 驱动的工具。
- [Spotify 将收购 AI 语音平台 Sonantic — Spotify](https://newsroom.spotify.com/2022-06-13/spotify-to-acquire-sonantic-an-ai-voice-platform/): 作为音频领域的领导者，Spotify 始终在寻找创造用户喜爱的独特体验的新方法。因此，今天我们很高兴地宣布我们打算收购 Sonantic，一家充满活力的...
- [理解 CrewAI Flows：综合指南](https://www.zinyando.com/understanding-crewai-flows-a-comprehensive-guide/): 在 AI 自动化方面，高效管理复杂的工作流至关重要。CrewAI 团队最近发布了 Flows，这是一个强大的功能，旨在简化 AI 的创建和管理...
- [Ofir Press (@OfirPress) 的推文](https://x.com/OfirPress/status/1844454994331959332): SWE-bench 由 @_carlosejimenez 和 @jyangballin 共同领导
- [48.5 万次观看 · 8000 次互动 | GTA 5 现实生活画质 | 生成式 AI | GTA 5 现实生活画质 | 生成式 AI | 由 TRCK 发布 | Facebook](https://www.facebook.com/trckgmng/videos/1254494898910739/?mibextid=rS40aB7S9Ucbxw6v): GTA 5 现实生活画质 | 生成式 AI
- [cocktail peanut (@cocktailpeanut) 的推文](https://x.com/cocktailpeanut/status/1844408840059506863?s=46): 类似 Artifacts，但 100% 开源、私密且本地化。@OpenWebUI 发布了一个改变游戏规则的更新——“Artifacts”。现在你可以通过 @o 使用完全本地和私密的 LLM，而不是专有的 LLM...
- [Vicente Silveira (@vicentes) 的推文](https://x.com/vicentes/status/1844200170441015382?s=46): 令人惊叹的 @OfficialLoganK 在旧金山的 Google DeepMind 活动中，向我们展示了下一代 Gemini 的预告...
- [AI Neocloud 策略手册与剖析](https://www.semianalysis.com/p/ai-neocloud-playbook-and-anatomy): H100 租赁价格削减、AI Neocloud 巨头与新兴 Neocloud、H100 集群物料清单与集群部署、日常运营、成本优化、所有权成本与回报
- [OpenAI (@OpenAI) 的推文](https://x.com/OpenAI/status/1844429536353714427): 我们正在发布一个新的基准测试 MLE-bench，用于衡量 AI Agent 在机器学习工程方面的表现。该基准测试包含 75 个源自机器学习工程相关的竞赛...
- [GitHub - huggingface/evaluation-guidebook: 分享我们在管理 Open LLM Leaderboard 和设计 lighteval 时积累的关于 LLM 评估的实践见解和理论知识！](https://github.com/huggingface/evaluation-guidebook): 分享我们在管理 Open LLM Leaderboard 和设计 lighteval 时积累的关于 LLM 评估的实践见解和理论知识！ - huggingface/evaluation-guidebook
- [GitHub 的推文 - FixTweet/FxTwitter: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等功能](https://x.com/OpenAI/): 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等功能 - FixTweet/FxTwitter
- [GitHub 的推文 - FixTweet/FxTwitter: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等功能](https://x.com/wondercraft_ai/status/): 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等功能 - FixTweet/FxTwitter

### **Latent Space ▷ #**[**ai-announcements**](https://discord.com/channels/822583790773862470/1075282504648511499/1294122171286949898) (5 条消息):

> - `GPU Rental Market` (GPU 租赁市场)
> - `H100 Pricing` (H100 定价)
> - `Latent Space Guest Post` (Latent Space 客座文章)
> - `Blackwell Chips` (Blackwell 芯片)

- **H100 价格暴跌至 $2/小时**：一篇题为 [*$2 H100s: How the GPU Rental Bubble Burst*](https://latent.space/p/gpu-bubble) 的客座文章指出，随着多家供应商的加入，H100 的价格已从 **$8/小时** 剧烈下降至不足 **$2**。
  
  - 随着新款 **Blackwell** 芯片进入市场，这引发了小型 AI 公司应该 *购买还是租赁* 的疑问。
- **祝贺登上 HN 首页！**：一位成员因发表了很久以来第一篇登上 **Hacker News** 的 LS 文章而获得祝贺，这在社区中引发了热烈反响。
  
  - 值得注意的是，@picocreator 撰写的这篇客座文章在互动率上超过了 **Tesla robotaxi**。
- **社区对 HN 评论的反应**：成员反馈显示，虽然 **robotaxi** 在 Twitter 上的评价很差，但 HN 帖子上的评论则更进一步表达了这种情绪。
  
  - 讨论暗示了对内容及其在社区中重要性的不同看法。
- **关于 SFCompute 的讨论升温**：在关于 H100 定价的对话中，一位成员提到 **sfcompute** 的知名度正在上升，特别是随着他们结束私测。
  
  - 这表明随着更多用户了解可用的替代方案， GPU 租赁市场的竞争日益激烈。

**提到的链接**：

- [来自 Latent.Space (@latentspacepod) 的推文](https://x.com/latentspacepod/status/1844563363877224889)：🆕 $2 H100s: How the GPU Rental Bubble Burst https://latent.space/p/gpu-bubble 一篇罕见的客座文章，来自回归嘉宾 @picocreator！H100 曾经是 $8/小时（如果你能抢到的话）。现在……
- [来自 swyx 🔜 NYC (@swyx) 的推文](https://x.com/swyx/status/1844616734390865978)：笑死，@picocreator 在 LS 的第一篇客座文章今天击败了 @tesla robotaxi。引用 Latent.Space (@latentspacepod) 🆕 $2 H100s: How the GPU Rental Bubble Burst https://latent.space/p/gpu-bubble ...

---

### **Latent Space ▷ #**[**ai-in-action-club**](https://discord.com/channels/822583790773862470/1200548371715342479/1294389030775029764) (39 条消息🔥):

> - `LLM Whisperers` (LLM 细语者)
> - `Programming Identity` (编程身份认同)
> - `Discord API Setup Challenges` (Discord API 设置挑战)
> - `Live Demo Insights` (现场演示心得)
> - `Feature Building Techniques` (功能构建技巧)

- **我们都是 LLM 细语者**：一位成员幽默地表示：**“我们都是 LLM 细语者”**，强调了 AI 讨论中的协作和探索性质。
  
  - 这种情绪引起了其他人的共鸣，大家纷纷表示赞同并发出笑声。
- **质疑程序员身份**：一位程序员对现在是否还应该称自己为程序员表示不确定，他说：**“作为一名程序员，我也在想……”**。
  
  - 另一位成员幽默地附和，增强了这种情绪的共鸣。
- **Discord API 设置很棘手**：讨论了 **设置挑战**，特别是在共享屏幕时获取 API key 的困难。
  
  - 一位成员感叹在 discord.py 和 discord.js 等库之间切换时 **权限带来的痛苦**。
- **关于现场演示的见解**：一位成员自称是 **“现场演示之王”**，分享了外部人员与熟悉任务的人对演示期望不同的见解。
  
  - 其他人分享了演示失败的经历，暗示了演示过程中 **技术故障** 的真实存在。
- **以最少的设置构建功能**：关于功能构建的讨论引出了一些简单项目想法的建议，如 **计算器应用或待办事项列表**，以简化工作流程。
  
  - 一位成员指出 **“那些好玩且奏效的东西只需要 10 秒钟”**，强调了作为专家处理复杂任务的必要性。

 

---

### **Modular (Mojo 🔥) ▷ #**[**general**](https://discord.com/channels/1087530497313357884/1098713601386233997/1294360503472357456) (4 条消息):

> - `Float Precision Issues` (浮点数精度问题)
> - `IEEE 745 Standards` (IEEE 745 标准)

- **0.15 的浮点精度失误**：一位用户询问为什么某个值不等于 **0.15**，引发了关于编程中 **浮点精度** 的讨论。
  
  - 文中指出，字面量被实例化为 **Float64**，其精度有限，导致出现类似于 **1/3** 无法在十进制中精确表示的差异。
- **浮点表示的一致性**：尽管存在精度问题，另一位成员安慰说，在 IEEE 745 **64-bit** 浮点数范围内，值保持 **自洽**。
  
  - 这意味着在这个特定表示的范围内，计算结果仍然会准确地等同于 **0.15**。

 

---

### **Modular (Mojo 🔥) ▷ #**[**mojo**](https://discord.com/channels/1087530497313357884/1151418092052815884/1294239321582473259) (73 messages🔥🔥):

> - `Mojo 中的 Trivial Types`
> - `AESNI 指令集`
> - `Mojo 中的内存与 Struct 管理`
> - `Mojo 中的 Reflection`
> - `SIMD 操作`

- **定义 Trivial Types**：用户讨论了为仅包含内联内存的 **trivial types** 定义 Trait 的挑战，特别是关于容器元素的约束。有人指出使用 `AnyTrivialRegType` 可能无法满足要求，因为它过于严格。
  
  - 当前 Trait 系统的局限性使得将 `AnyTrivialRegType` 与其他 Trait 结合变得困难，促使用户寻求替代方案。
- **硬件 AES 实现问题**：一位用户分享了旨在确定系统是否支持 **AESNI 指令集** 的代码，但在确保编译器正确识别它时遇到了问题。函数 `has_aesni()` 检查 X86 架构兼容性并利用了 **llvm_intrinsic**。
  
  - 尽管最初对正确的编译器行为存在一些困惑，但已确认将支持 **AVX2** 和 **AVX512**，从而允许使用多种指令宽度以提高效率。
- **容器中高效的 Struct 创建**：有关于 Struct 是否可以就地创建以避免不必要的拷贝操作的咨询，特别是在向 List 追加元素时。澄清了将 Struct 作为右值（rvalues）创建通常不涉及拷贝，除非创建了临时变量。
  
  - 然而，提到的 `__moveinit__` 方法作为一种轻量级拷贝方法仍可能被调用，这引发了关于大型类型操作效率的讨论。
- **Reflection 及其作用**：对话涉及了 **Reflection** 的概念及其在 Mojo 中的不可变性和可变性影响，并建议早期 Reflection 可以表示可变状态。简要比较了 C++ 的 Reflection 实践，但用户主张在 Mojo 中区分 Trait。
  
  - 这一讨论引发了关于如何利用某些构造来最小化代码复杂性的思考，特别是在自动化检查 Struct 装饰器时。
- **循环展开与编译器行为**：在调试阶段，一位用户意识到由于对其编译代码中 **aesdec** 函数的疏忽，其展开的循环生成的调用次数超出了预期。这强调了在进行优化以防止误编译时保持警惕的重要性。
  
  - 此外，用户建议使用约束来简化硬件支持检查，同时验证操作是否正确执行，展示了调试中的有效团队协作。

**提到的链接**：

- [Types | Modular Docs](https://docs.modular.com/mojo/manual/types#anytype-)：标准 Mojo 数据类型。
- [Types | Modular Docs](https://docs.modular.com/mojo/manual/types#anytype-and-anytrivialregtype)：标准 Mojo 数据类型。

---

### **Modular (Mojo 🔥) ▷ #**[**max**](https://discord.com/channels/1087530497313357884/1212827597323509870/1294276906132308048) (1 messages):

> - `Model IR 缓存`
> - `序列化 MAX 图`

- **启用 Model IR 缓存以加快编译**：为了提升性能，请检查位于 `~/.modular/modular.cfg` 的 `modular.cfg` 中是否存在 `enable_model_ir_cache` 配置选项，这允许在缓存命中时在后续运行中实现更快的编译。
  
  - *这应该会显著减少编译时间。*
- **序列化 MAX 图支持即将到来**：目前还没有对 **serialized MAX graphs** 的官方支持，但相关开发工作正在进行中。
  
  - *团队正在积极开发中，后续将会有更新。*

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1294021729877889197) (67 messages🔥🔥):

> - `Perplexity 使用技巧`
> - `Pro 功能讨论`
> - `搜索功能问题`
> - `社区互动与 Memes`
> - `数学与 API 咨询`

- **对 Perplexity 性能的担忧**：一位成员对使用 Perplexity 相比 ChatGPT 的优势表示怀疑，指出 Perplexity 的回答中存在更多幻觉（hallucinations）。
  
  - 另一位成员分享了一个具体案例，即 Perplexity 在麻风病相关的历史信息中提供了错误内容。
- **Pro 搜索功能演进**：用户讨论了 Pro 搜索功能现在采用了 “Agentic” 模型，这需要更集中的输入才能有效运行。
  
  - 有人提到，早期的引导式搜索功能已转变为更复杂的输入输出交互。
- **搜索挂起问题**：一位用户报告了 Perplexity 的问题，即搜索会出现挂起（hang），导致在输入新查询时难以阅读。
  
  - 这引发了关于用户体验以及对界面挫败感的讨论。
- **对可用模型的好奇**：新的 Pro 用户询问了在 Perplexity 中使用哪些模型最有效，特别是关于 API 交互方面。
  
  - 回复中包含了有关模型能力和 API 功能进一步详情的链接。
- **频道内的轻松互动**：几位成员进行了俏皮的打趣并分享了 Memes，例如一只在雪地里的猫，配文是“当地域结冰时（when hell freezes over）”。
  
  - 社区保持着轻松的氛围，讨论了头像和对狗的品味。

**提到的链接**：

- [Ryan Putnam (@RypeArts) 的推文](https://x.com/RypeArts/status/1844426971960443012)：更多诡异氛围
- [Bilawal Sidhu (@bilawalsidhu) 的推文](https://x.com/bilawalsidhu/status/1844466815776457187?s=46)：Aravind Srinivas：“爱因斯坦把所有精力都花在思考相对论上，对吧？如果有位爱因斯坦把所有精力都花在思考你的生活上呢？任何你遇到的问题...”
- [未找到标题](https://docs.perplexity.ai/api-reference/chat-completions)：未找到描述
- [底特律：变人 Ps4 游戏 GIF - Detroit Become Human Playstation Game - Discover & Share GIFs](https://tenor.com/view/detroit-become-human-playstation-game-rpg-gif-12174308)：点击查看 GIF
- [当地域结冰时 GIF - Hell Freezes Over When Hell Freezes Over - Discover & Share GIFs](https://tenor.com/view/hell-freezes-over-when-hell-freezes-over-gif-13444285)：点击查看 GIF
- [Aravind Srinivas (@AravSrinivas) 的推文](https://x.com/AravSrinivas/status/1844609170814926961?t=8g7iM-sdPf7R0s1KyA1xsw&s=19)：现在，你可以在 Perplexity Code Interpreter 上制作精美的图表了！

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1294050151450677249) (8 messages🔥):

> - `GPU 驱动更新`
> - `特斯拉 Robovan`
> - `CyberCab`
> - `飓风米尔顿`
> - `德国撇号辩论`

- **GPU 驱动更新引发联合国警报**：最新的 [GPU 驱动更新](https://www.perplexity.ai/page/gpu-driver-updpate-triggers-un-xL6D7KG8SmCIQ.CGSBPbbg) 引起了对潜在漏洞的担忧，引发了联合国的警报。
  
  - 讨论集中在对安全和技术格局的影响上。
- **特斯拉发布创新型 Robovan**：特斯拉推出了一款 [新 Robovan](https://www.perplexity.ai/page/tesla-robovan-HbGsP0T1Tea_paN7W0u4gw) 模型，旨在增强城市交通解决方案。
  
  - 特点包括高电气效率和先进的驾驶辅助系统。
- **CyberCab 彻底改变共享出行**：特斯拉 [CyberCab](https://www.perplexity.ai/page/cybercab-tesla-x2ivalNAQ5GY7hoUi1FuvA) 的发布将以独特的设计和环保技术改变城市出行。
  
  - 其自动驾驶功能有望提高共享出行的效率和安全性。
- **飓风米尔顿袭击佛罗里达**：据 [此处](https://www.perplexity.ai/page/hurricane-milton-hits-florida-fJjruP5JR5ilumQEJSmcfw) 报道，飓风米尔顿已正式袭击佛罗里达州，造成大范围混乱和紧急疏散。
  
  - 气象学家正因其不可预测性而密切跟踪其路径。
- **德国撇号辩论升温**：围绕 [德国](https://www.perplexity.ai/page/germany-s-apostrophe-debate-1DrUiXyvR0i7zpbuc9GKVA) 撇号使用的激烈讨论反映了语言规则上的文化冲突。
  
  - 专家们就为了使语言标准现代化是否应该做出改变发表了看法。

---

### **Perplexity AI ▷ #**[**pplx-api**](https://discord.com/channels/1047197230748151888/1161802929053909012/1294139526792478760) (3 条消息):

> - `Perplexity API 响应时间`
> - `API 中的 Web socket 使用`
> - `引用 (Citations) 的访问权限`
> - `提高速率限制 (Rate Limits)`

- **Perplexity API 慢于预期**：一位用户询问了 Perplexity API 的 **2 秒响应时间**，而 [Perplexity Labs](https://labs.perplexity.ai/) 网站上的响应时间**不到 1 秒**。
  
  - 他们认为 Labs 网站上使用的 **Web sockets** 可能是响应速度更快的原因，并询问是否可以在 API 中实现这一点。
- **请求访问引用和提高速率限制**：用户 **peso** 反馈称，他曾发邮件给支持团队请求访问引用权限并**提高速率限制 (Rate Limit)**，但未收到回复。
  
  - **Alex** 建议 **peso** 将请求转发至 [api@perplexity.ai](mailto:api@perplexity.ai) 以获得更快的协助。

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1294095035507146812) (75 条消息🔥🔥):

> - `使用问题`
> - `模型定价差异`
> - `LLM 写作技巧`
> - `聊天模型共享`
> - `账户访问故障`

- **解决使用问题**：一位成员询问是否可以就账单和使用问题私信 (DM) 工作人员，Alex Atallah 建议在调用 `/generation API` 时，如果 ID 没有立即显示，可以多等待一会儿。
  
  - 这与用户在 API 请求和响应问题方面的各种经历相吻合。
- **理解模型定价差异**：成员们讨论了 Mistral Nemo 12B Starcannon 和 Rocinante 12B 之间的价格差异，观察表明 Mistral 采取了竞争性定价策略。
  
  - 对话强调了竞争动态，指出由于缺乏 Rocinante 12B 的其他供应商，可能导致其定价更高。
- **使用 LLM 增强写作**：一位用户分享道，仅在文章的特定部分使用 LLM 显著提高了他们的写作质量。
  
  - 另一位成员强调，虽然并非每个人都是天生的作家，但只要付出一些努力，LLM 几乎可以帮助任何人创作出像样的文本。
- **“共享模型”功能的作用**：用户了解到，“共享模型”按钮会复制一个链接，用于共享当前聊天室的模型设置，但不包括参数或 Prompt。
  
  - 该功能提供了一种快速共享模型设置的方法，但缺乏更深层次共享所需的全面细节。
- **记录到的账户访问故障**：一位用户报告了一个故障，允许通过同一设备上的不同账户访问其旧账户的聊天记录，这引发了对 Cookie 保留缓存的担忧。
  
  - 这引发了关于聊天记录如何本地存储在浏览器工具中的讨论，暗示了可能存在的隐私和数据管理问题。

**提到的链接**：[Chatroom | OpenRouter](https://openrouter.ai/chat?models=anthropic/claude-3.5-sonnet,openai/o1-preview,google/gemini-pro-1.5)：LLM Chatroom 是一个多模型聊天界面。添加模型并开始聊天！Chatroom 将数据本地存储在您的浏览器中。

---

### **Eleuther ▷ #**[**announcements**](https://discord.com/channels/729741769192767510/794042109048651818/1294353447306657917) (1 条消息):

> - `GPT-NeoX library updates`
> - `Performance improvement over HuggingFace`
> - `New features in GPT-NeoX 3.0`
> - `Post-training methods introduced`
> - `Testing of new GPT-NeoX features`

- **GPT-NeoX 库引入训练后（post-training）方法**：HPC 团队为 **GPT-NeoX** 库引入了训练后功能，允许用户原生进行 **SFT, DPO 和 KTO finetuning**。
  
  - 这一更新得到了来自 SynthLabs 成员的协作支持，并增强了该库的功能。
- **GPT-NeoX 性能优于 HuggingFace 的 trl**：测试结果表明，在 13B 规模下，**GPT-NeoX** 相比 **HuggingFace 的 trl 库** 展现出 **30% 的性能提升**。
  
  - 这种提升还带来了对大规模计算系统更好的可扩展性，而这是 trl 所不支持的。
- **GPT-NeoX 3.0 中令人兴奋的新特性**：即将发布的 GPT-NeoX 3.0 版本将包含多项新特性，例如 **AMD GPUs, Mixture-of-Experts (MoE) layers, RWKV 和 Mamba**。
  
  - 在该版本进行发布前的 Bug 测试期间，用户已经可以在 `main` 分支上测试这些特性。
- **社区对 GPT-NeoX 特性的反馈**：鼓励用户在测试后，在指定频道反馈 **GPT-NeoX** 新特性的使用体验。
  
  - 此举旨在稳定版发布前优化性能和用户体验。
- **通过博客文章了解更多关于 GPT-NeoX 的信息**：用户可以通过阅读 Eleuther 的 [博客文章](https://blog.eleuther.ai/rlhf-and-rlaif-in-gpt-neox/) 以及其中链接的 SynthLabs 文章来关注 **GPT-NeoX** 的进展。
  
  - 更多信息及库的访问请见 [GPT-NeoX library](https://github.com/EleutherAI/gpt-neox)。

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1294041751278846045) (53 条消息🔥):

> - `Entropy-based sampling`
> - `Computational psychiatry`
> - `Image-based RAG models`
> - `Quantization for LLMs`
> - `Inference speed in training`

- **LLM 中的 Entropy-based sampling 引发疑问**：关于 Llama3.1 等模型中 Entropy-based sampling 有效性的讨论表明，验证其相对于基准推理评分的改进对于确定其影响至关重要。
  
  - 一位成员对基于不同采样方法得出结果表示担忧，强调需要可靠的证据将这些技术与性能提升联系起来。
- **探索 Computational psychiatry 的潜力**：一位成员提出了一个有趣的观点，即研究 LLM 以获取对精神障碍的见解，特别是通过利用 ML 模拟大脑行为的 “Computational psychiatry” 研究。
  
  - 大家一致认为，虽然 LLM 可能不会表现出类似人类的障碍，但利用其输出进行分析可以提供有价值的框架，尽管证据对齐（evidence alignment）仍然是一个障碍。
- **Image-based RAG 应用面临的挑战**：有人对在 RAG 系统中使用 Image-based retrievers 的效果表示担忧，经验表明从 Colqwen 等平台检索到的内容相关性较低。
  
  - 成员们询问了其他领域的成功实现，并寻求关于 RAG 最佳 Text-based embedding models 的指导。
- **训练中的 Quantization 方法**：对 Google AQT 的讨论显示，它在 matmul 之前对 activations 和 weights 进行 Quantization，在不直接训练 int8 权重的情况下提高了效率。
  
  - 有人指出，虽然它可以加快模型训练期间的 Inference 速度，但可能需要在内存中保留参数的浮点副本。
- **训练期间的 Inference 速度提升**：成员们讨论了 Quantization 方法如何加速整个训练过程中的 Inference，表明在不受带宽限制时具有优势。
  
  - 对话承认，在训练期间整合这种效率可以有效消除权重在内存中换入换出的开销。

**提及的链接**：

- [Will entropy-based sampling improve Llama3.1 on reasoning benchmarks in 2024?](https://manifold.markets/CharlesFoster/will-entropybased-sampling-improve)：47% 的概率。Entropy-based sampling（俗称 "shrek sampler"）是 LLM 的一类新型采样方法，旨在“模拟类似于 o1 的 CoT...”
- [Brain-Score](https://www.brain-score.org/)：Brain-Score 是一个供研究人员测试模型在预测神经和行为大脑测量方面表现的平台。
- [GitHub - stillmatic/entropix: Entropy Based Sampling and Parallel CoT Decoding](https://github.com/stillmatic/entropix)：Entropy Based Sampling 和 Parallel CoT Decoding。
- [aqt/aqt/jax/v2/examples/examples.ipynb at main · google/aqt](https://github.com/google/aqt/blob/main/aqt/jax/v2/examples/examples.ipynb)：通过在 GitHub 上创建账号为 google/aqt 做出贡献。
- [Dementia in Convolutional Neural Networks: Using Deep Learning Models to Simulate Neurodegeneration of the Visual System - Neuroinformatics](https://link.springer.com/article/10.1007/s12021-022-09602-6)：尽管目前的研究旨在通过应用有关健康人类大脑的知识来改进深度学习网络，反之亦然，但使用此类网络来建模和研究神经退行性变的潜力...

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1294074230119333929) (8 messages🔥):

> - `Reward Mechanism on EOS`
> - `Endorsing Papers in CS`
> - `Benchmark Discussions`
> - `ARC Challenge`
> - `MMLU Subsets`

- **EOS 奖励机制解析**：一位用户澄清说，AI 仅在非常特定的条件下（基本上只有在正确时）才会因序列结束 (**EOS**) 指示符获得正向梯度奖励。
  
  - 简单来说，*如果它知道自己成功了，就会结束输出；如果它认为还没成功，就会继续生成。*
- **寻求论文背书**：一位用户表示需要有人为他们即将在计算语言学类别 (**cs.CL**) 发表的论文提供背书，并愿意分享更多细节。
  
  - 这反映了社区中的一种常见做法，即个人为其研究工作寻求支持和认可。
- **正在讨论的严肃基准测试**：一位成员提出了关于值得信赖的基准测试的问题，引发了关于可靠评估指标的讨论。
  
  - 另一位成员指出 **ARC Challenge** 可能是目前最可靠的基准测试，并强调其专注于二阶推理能力。
- **MMLU 子集作为可行基准**：同一位成员推测 **MMLU** 的特定子集也可以作为有效的基准，可能满足严格的评估标准。
  
  - 这表明了目前对评估流程的持续审查，以及在 AI 评估中对改进指标的渴望。

 

---

### **Eleuther ▷ #**[**scaling-laws**](https://discord.com/channels/729741769192767510/785968841301426216/) (1 messages):

micpie: [https://arxiv.org/abs/2410.08184](https://arxiv.org/abs/2410.08184)

---

### **Eleuther ▷ #**[**interpretability-general**](https://discord.com/channels/729741769192767510/1052314805576400977/1294052065307529267) (5 messages):

> - `Neural Networks Learn Statistics of Increasing Complexity`
> - `LEACE-work motivation`
> - `Pythia checkpoints evaluation`
> - `Learning dynamics experiments`
> - `MLP learning limitations`

- **关于神经网络的新 arXiv 论文**：题为 [Neural Networks Learn Statistics of Increasing Complexity](https://arxiv.org/abs/2402.04362) 的更新论文讨论了在基于 Pile 训练的 **trigram 和 4-gram LMs** 采样的序列上对 **Pythia checkpoints** 进行的评估。
  
  - 作者表达了兴奋之情，回应了关于该论文“非常酷”的反馈，并确认了与 *LEACE-work* 的联系。
- **连接 LEACE 和学习动态的未来实验**：作者计划重新进行一些实验，以更好地将 **LEACE** 与 **learning dynamics** 联系起来，这表明其动力源于早期的工作。
  
  - 他们指出，如果没有关于类标签的二次信息，小型 **MLPs** 似乎无法学习任何东西。

 

**提到的链接**：[来自 Nora Belrose (@norabelrose) 的推文](https://x.com/norabelrose/status/1844492975075885143)：我们论文《Neural Networks Learn Statistics of Increasing Complexity》的新 arXiv 版本，包括在基于 Pi... 训练的 trigram 和 4-gram LMs 采样的序列上对 Pythia checkpoints 的评估。

 

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1294336667171295353) (5 messages):

> - `lm-eval-harness warnings`
> - `Environment variable solution`
> - `fast tokenizers`

- **lm-eval-harness 触发多个警告**：一位成员报告称，在调用 **lm-eval-harness** 代码时收到关于 **tokenizers** 分叉进程的警告，导致该消息过度输出。
  
  - 警告建议避免在 fork 之前使用 tokenizers，或将 `TOKENIZERS_PARALLELISM` 设置为 **false**。
- **设置环境变量以解决警告**：另一位成员分享说，他们通过在主代码中将 **tokenizers** 的环境变量设置为 **false** 解决了该问题，效果良好。
  
  - 这允许用户在不显著改变设置的情况下规避烦人的重复警告。
- **Fast tokenizers 利用 Rust**：一位参与者指出 **fast tokenizers** 利用了 **Rust**，这表明了潜在的性能优势。
  
  - 这与 Tokenization 领域为提高效率和速度所做的改进相一致。
- **感谢问题解决**：一位成员对获得的帮助表示感谢，称这有助于修复 **torchtune** 中的一个问题。
  
  - 这种协作式的问题解决展示了社区对改进工作流程的共同承诺。

 

---

### **aider (Paul Gauthier) ▷ #**[**general**](https://discord.com/channels/1131200896827654144/1131200896827654149/1294023686512316447) (38 条消息🔥):

> - `Aider 的性能`
> - `DeepSeek 的局限性`
> - `配置挑战`
> - `视频资源推荐`
> - `Aider 中的错误处理`

- **Aider 表现优于竞争对手**：用户声称 **Aider** 在修复 Bug 和编码任务方面显著优于 **Cline** 和 **Cursor** 等工具，一位用户在对各种框架进行广泛测试后表示它是目前最出色的工具。
  
  - 多位成员强调了其效率，认为它在前端和后端应用开发中是“不二之选”。
- **DeepSeek 面临的挑战**：多位用户对 **DeepSeek** 表示沮丧，称其在尝试整合函数时速度缓慢且令人不满，强调了其对独立开发者而言缺乏效率。
  
  - 一位用户分享说，由于遇到 *edit format errors*（编辑格式错误），他们已换回使用 **Sonnet-3.5**，并表示 DeepSeek 相比之下表现不佳。
- **模型配置**：一位用户寻求关于如何正确配置 `.env` 文件以设置 **openrouter** 模型的建议，因为他们遇到了默认设置意外切换的问题。
  
  - 另一位用户指出，使用 `--edit-format whole` 选项可能会加剧 DeepSeek 的性能问题。
- **YouTube 视频好评**：一位用户推荐了某个特定频道的视频，强调了所呈现内容的质量和清晰度，并分享了相关链接。
  
  - 几位成员对该频道的成长表示期待，赞赏视频具有很强的信息量。
- **Aider 中的错误处理**：讨论中提到了 Aider 中频繁出现的 **search/replace errors**（搜索/替换错误），并给出了通过调整设置来改善结果的建议。
  
  - 用户通过链接到故障排除文档，强调了使用高性能模型来减少这些错误发生的重要性。

**提到的链接**：

- [文件编辑问题](https://aider.chat/docs/troubleshooting/edit-errors.html)：Aider 是你终端里的 AI 配对编程助手。
- [Linting 和测试](https://aider.chat/docs/usage/lint-test.html)：自动修复 Linting 和测试错误。
- [Repository map](https://aider.chat/docs/repomap.html)：Aider 使用你的 Git 仓库映射为 LLM 提供代码上下文。

---

### **aider (Paul Gauthier) ▷ #**[**questions-and-tips**](https://discord.com/channels/1131200896827654144/1133060505792159755/1294022584194830491) (22 条消息🔥):

> - `Aider 并非为依赖项而设计`
> - `Aider 使用的最佳实践`
> - `通过环境变量使用 Aider`
> - `Aider 的教程与资源`
> - `自定义模型中的无限输出`

- **Aider 的架构并非为了方便作为依赖项引入**：成员们讨论了 **Aider** 的架构并非设计为项目中的依赖项，因为它旨在用于交互式使用。
  
  - 有人建议设置 bash 别名以简化使用，而另一位则强调使用 `aider --show-repo-map` 可以辅助查看有用的结构。
- **有效使用 Aider 的实用技巧**：一位成员强调，如果 Aider 卡住或产生困惑，手动编写代码然后寻求后续任务的帮助通常会更快。
  
  - 编码的乐趣被视为一种动力因素，暗示编码可以提供能量提升并提高生产力。
- **使用 gemini-api-key 调用 Aider**：一位用户询问是否可以在调用 Aider 时直接传递 **gemini-api-key**，但有人指出，对于非 OpenAI 或 Anthropic 的密钥，仅接受环境变量。
  
  - 这一澄清强调了在使用备用 API 密钥时需要进行正确的配置。
- **Aider 的资源分享和教程视频**：一位用户对发现 Aider 感到兴奋，并寻求有关实时编码示例和高级使用技巧的视频资源。
  
  - Paul 分享了一份详尽的教程视频列表，突出了它们的实际应用并展示了有效使用 Aider 的策略。
- **为自定义模型设置无限输出**：有人询问如何为自定义模型启用无限输出，特别是询问模型元数据中的要求。
  
  - 澄清指出，对于模型元数据 JSON，字段 `

**提到的链接**：

- [YAML 配置文件](https://aider.chat/docs/config/aider_conf.html)：如何使用 yaml 配置文件配置 aider。
- [指定编码规范](https://aider.chat/docs/usage/conventions.html#always-load-conventions)：告知 aider 在处理代码时遵循你的编码规范。
- [DevDocs](https://devdocs.io/)：为开发者提供的快速、离线且免费的文档浏览器。在一个 Web 应用中搜索 100 多份文档，包括 HTML, CSS, JavaScript, PHP, Ruby, Python, Go, C, C++ 等。
- [选项参考](https://aider.chat/docs/config/options.html#--suggest-shell-commands)：关于 aider 所有设置的详细信息。
- [教程视频](https://aider.chat/docs/usage/tutorials.html)：由 aider 用户制作的入门和教程视频。

---

### **aider (Paul Gauthier) ▷ #**[**links**](https://discord.com/channels/1131200896827654144/1268910919057149974/1294030636629102715) (8 条消息🔥):

> - `Chain of Thought 算法`
> - `Diffsitter 工具`
> - `Difftastic 探索`
> - `PHP 查找/替换问题`
> - `编辑错误排查`

- **Chain of Thought 算法开发**：一位成员对开发 **Chain of Thought 算法** 表示兴奋，认为分享资源的 timing 非常有帮助。
  
  - 另一位成员鼓励了这一举措，祝愿他们在这一尝试中取得成功。
- **Diffsitter 工具介绍**：[Diffsitter](https://github.com/afnanenayet/diffsitter) 是一个通过在 **AST** (抽象语法树) 而非原始文本上计算差异来创建具有语义意义的 diff 的工具。
  
  - 它能有效忽略格式差异，从而产生更整洁的 diff，且不会因额外的空格产生问题。
- **对 Difftastic 的兴趣**：一位成员询问其他人是否见过 **difftastic**，引发了对这一额外工具的兴趣。
  
  - 一位成员承认不熟悉，但承诺会进一步调查，并提到了 PHP 中查找和替换的问题。
- **PHP 查找/替换的挑战**：一位成员强调了 PHP 中反复出现的 **search/replace issues**，代码会被追加而不是替换，导致 lint 错误。
  
  - 他们对无法有效解决这些代码问题表示沮丧。
- **编辑错误排查指南**：另一位成员分享了在使用 LLM 时可能出现的编辑错误的 [排查指南](https://aider.chat/docs/troubleshooting/edit-errors.html)。
  
  - 该指南包括一些技巧，例如使用性能强大的模型（如 **GPT-4o 或 Claude 3**），以尽量减少对系统提示词的不服从。

**提到的链接**：

- [文件编辑问题](https://aider.chat/docs/troubleshooting/edit-errors.html)：aider 是你终端里的 AI 配对编程助手。
- [GitHub - afnanenayet/diffsitter: 一个基于 tree-sitter 的 AST diff 工具，用于获取有意义的语义 diff](https://github.com/afnanenayet/diffsitter)：一个基于 tree-sitter 的 AST diff 工具，用于获取有意义的语义 diff - afnanenayet/diffsitter

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1294045752951181355) (53 条消息🔥):

> - `Voice Modulation Techniques` (语音调制技术)
> - `Perception of AI as Psychopaths` (将 AI 视为精神病态者的看法)
> - `OpenAI Copilot Comparison` (OpenAI Copilot 对比)
> - `AI's Bedside Manner vs Human` (AI 与人类的临床沟通技巧对比)
> - `Impact of Intellectual Property on Innovation` (知识产权对创新的影响)

- **语音调制 vs 唱歌 (Voice Modulation vs Singing)**：讨论者分享了鼓励 AI 进行语音调制的方法，强调通过使用特定的 Prompt（如 *voice modulation* 而非 *singing*）可以有效地模拟唱歌而无需真正唱歌。
  
  - 一位用户提到曾尝试引导 AI 以更具表现力的风格进行表演，但当 AI 拒绝进行戏剧化或受诗歌启发的表演时感到有些沮丧。
- **AI 与精神病态的比较 (AI and Psychopathy Comparison)**：一名成员提出了高功能精神病态者与 AI 之间的比较，认为两者主要基于逻辑计算运行，缺乏情感负担。
  
  - 这场对话引发了关于精神病态特征是否被有意建模到 AI 系统中的辩论，讨论中交织着幽默与严肃。
- **关于 OpenAI Copilot 效能的辩论**：用户批评了最新版本的 OpenAI Copilot，指出其性能较之前的迭代版本有所下降，并将其与 Google 的 Gemini 进行了不利的对比。
  
  - 虽然一些人通过暗示批评源于对其功能的误解来为新模型辩护，但其他人则认为它缺乏像打字动画这样必不可少的功能。
- **AI 优越的临床沟通技巧 (Bedside Manner)**：几位成员幽默地指出，有报告显示 AI 表现出比人类医生更好的临床沟通技巧（Bedside Manner），引发了对 AI 共情本质的发人深省的反思。
  
  - 这引发了对医疗专业人员的精神病态特征是否会无意中导致在关键情况下做出更好决策的探讨，为讨论增添了黑色幽默色彩。
- **知识产权对 AI 创新的影响**：讨论承认了知识产权法对 AI 开发创新过程施加的限制，重点关注变现担忧和诉讼风险。
  
  - 讨论强调了创造力与所有权之间的紧张关系，认为这些法律框架可能会扼杀 AI 技术的突破性进展。

 

**提到的链接**：[GPT Unicorn](https://gpt-unicorn.adamkdean.co.uk/)：未找到描述

 

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1294103386462883851) (9 条消息🔥):

> - `Interacting with ChatGPT` (与 ChatGPT 交互)
> - `Assessing Ideas with ChatGPT` (使用 ChatGPT 评估想法)
> - `Irrational Responses for Fun` (追求非理性回复的乐趣)
> - `Embedding Compatibility` (Embedding 兼容性)

- **ChatGPT 交互限制**：一位用户对无法在服务器中与 ChatGPT 交谈表示沮丧，但被引导至 [chatgpt.com](https://chatgpt.com) 获取免费访问权限。
  
  - 这导致另一位用户指出在该平台上保存答案存在问题。
- **寻求理性评估**：一名成员建议通过 Prompt 要求 ChatGPT 以理性且冷静的方式评估想法，以避免得到过度美化的回复（Glorified Responses）。
  
  - 他们强调提供清晰的上下文，以获得对想法更客观的评估。
- **非理性回复的乐趣**：一名成员建议，要求 AI 表现得不理性可能会带来更有趣的结果。
  
  - 这暗示了一种与 AI 交互的趣味性方法。
- **Embedding 兼容性咨询**：一位用户询问了在运行余弦相似度（Cosine Similarity）时，**text-embedding-ada-002** 和 **text-embedding-3-small** 的兼容性。
  
  - 他们质疑将两者混合使用是否会导致不连贯的答案或引发错误。

 

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/1294265262488420393) (2 条消息):

> - `Activity in Prompt Engineering Channel` (Prompt Engineering 频道的活跃度)
> - `Engagement from Members` (成员参与度)
> - `Discord Dynamics` (Discord 动态)

- **成员质疑频道参与度**：*hydarnes_46264* 对 **Prompt Engineering** 频道活跃度低表示困惑，怀疑自己是否找错了地方。
  
  - *eskcanta* 指出，当成员有具体问题或见解要分享时，确实会产生互动。
- **理解频道活跃度**：*eskcanta* 指出，虽然有超过 **11 万名成员**，但并非所有人都在从事学术研究，这可能解释了频道为何如此安静。
  
  - *eskcanta* 提到，许多人可能并不觉得有必要贡献自己的见解，从而导致了感知上的参与度缺失。

 

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/1294265262488420393) (2 条消息):

> - `频道活跃度`
> - `Prompt Engineering 兴趣`
> - `成员参与度`

- **频道活跃度受到质疑**：一位成员对缺乏活跃度表示担忧，问道：*“是我进错频道了吗，还是这里为什么这么安静？”*。
  
  - 他们指出，目前 Prompt Engineering 似乎显得有些乏味，尤其是考虑到频道内有超过 **11 万名成员**。
- **对活跃度担忧的回应**：另一位成员指出，向上滚动可以看到一些活动，并提到成员通常在有疑问或有信息分享时才会参与。
  
  - 他们认为提问者的提问方式过于狭隘，因为许多成员可能并不在学术研究领域。
- **成员参与度的差异性**：注意到成员参与度存在差异，一些成员可能觉得由于缺乏见解而无法贡献内容。
  
  - 这反映了频道受众兴趣的多样性，而不仅仅局限于 Prompt Engineering。

 

---

### **Stability.ai (Stable Diffusion) ▷ #**[**general-chat**](https://discord.com/channels/1002292111942635562/1002292112739549196/1294079785009414218) (62 条消息🔥🔥):

> - `用于 AI 生成的 ComfyUI`
> - `使用 AMD 进行 AI 计算`
> - `在 3060 Ti 上运行 Stable Diffusion`
> - `Lora 触发词管理`
> - `模型合并技术`

- **ComfyUI 是首选工具**：成员表示在运行 Flux 时更倾向于使用 **ComfyUI**，而对于想要开始使用 Stable Diffusion 的初学者，建议使用 **Automatic1111**。
  
  - 另一个建议是使用 PyTorch 或 Diffusers 进行命令行界面操作。
- **AMD 进行 AI 计算的挑战**：一位成员对 AMD GPU 缺乏 **CUDA** 支持表示沮丧，并提到了 Python 方面的挑战。
  
  - 一位用户为拥有 8GB 或更多 VRAM 的 AMD GPU 用户提供了使用 ZLUDA 版本的指南。
- **3060 Ti 在 Stable Diffusion 中的能力**：确认 **3060 Ti** 是运行 Stable Diffusion 的优秀 GPU，并建议通过上采样图像来最大化质量，尽管它有 8GB VRAM 的限制。
  
  - 成员分享说，使用量化（quantizations）和分块上采样（tiled upscaling）可以帮助获得更高质量的输出。
- **管理 Loras 的触发词**：一位用户询问了记忆 Loras 触发词的策略，以及是否有自动添加触发词的方法。
  
  - 这个问题引发了关于有效管理 Lora 触发词所面临挑战的讨论。
- **通过模型合并提升质量**：关于模型合并与连续传递（consecutive passes）之间差异的讨论浮出水面，并深入探讨了在扩散步骤中使用特定 **sigma** 值的方法。
  
  - 成员指出，合并两个模型可能会平均它们的能力，从而产生平衡的性能。

**提到的链接**：

- [AMD 为消费者优化 ROCm：将您的 “Radeon” 系统转变为本地 AI 解决方案](https://wccftech.com/amd-tunes-rocm-for-consumers-turns-radeon-systems-into-localized-ai-machines/)：AMD 在 AI 工作负载方面做出了重大努力，目前已推出对 RDNA 3 架构上 ML 开发的支持。
- [Geeky RemB - Geeky Remb live portrait | Flux 工作流 | Civitai](https://civitai.com/models/546180/geeky-remb)：Flux Schnell NF4 动画合成工作流。这个强大的 ComfyUI 工作流利用 Flux Schnell NF4 模型来创建令人惊叹的动画作品...
- [Webui 安装指南](https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides)：Stable Diffusion 知识库（设置、基础、指南等） - CS1o/Stable-Diffusion-Info
- [Flux xy grid (深色模式) - v3.5 | Flux 工作流 | Civitai](https://civitai.com/models/635692/flux-xy-grid-dark-mode)：请分享您的作品。我很想看看您是如何使用这个工作流工具的。用于评估 FLUX 参数的 X/Y 网格。我还制作了...

---

### **GPU MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1189498205101109300/1294207653706207242) (2 条消息):

> - `GPU 工程师实习准备`
> - `使用 cuDNN 的 SDPA 实现 Attention 层`

- **GPU 工程师实习建议**：一位成员请求提供准备 GPU 工程师实习的**资源和建议**，并说明了要求，如处于**大学最后一年**且具有扎实的 **CUDA** 背景。
  
  - 他们提到测试将包括**多选题**和**编程任务**。
- **寻求 cuDNN SDPA 实现的指导**：另一位成员询问了在 Python 中使用 **cuDNN 的 SDPA** 实现 **Attention 层** 的**教程或实现方式**。
  
  - 他们对 **pygraph 的实例化** 表示困惑，并参考 **cudnn-frontend 仓库**中的笔记本寻求帮助。

 

---

### **GPU MODE ▷ #**[**triton**](https://discord.com/channels/1189498204333543425/1189607595451895918/1294373657464012830) (2 条消息):

> - `Kernel` 类型

- **关于 Persistent 与 Non-Persistent Kernels 的讨论**：一位成员询问另一位成员是否正在开发 **persistent kernel** 或 **non-persistent kernel**。
  
  - 这引发了另一位成员的后续回复，表示他们将在另一个讨论串中回答，并感谢提问者的提问。
- **响应位置的澄清**：另一位成员确认了关于 Kernel 类型的问题，并表示他们将在单独的讨论串中提供答案。
  
  - 这展示了频道内组织化讨论的倾向，鼓励清晰且专注的回复。

---

### **GPU MODE ▷ #**[**torch**](https://discord.com/channels/1189498204333543425/1189607750876008468/1294013663409209456) (15 条消息🔥):

> - `在受限 GPU 上训练 Llama 7B`
> - `优化器与参数 Offloading`
> - `内存管理技术`
> - `CUDA 与 PyTorch 见解`
> - `Gradient Checkpointing 的重要性`

- **在 16GB GPU 上训练 Llama 7B 的策略**：一位成员提出了在单个 **16GB** GPU 上训练需要 **28GB** GPU 显存的 **Llama 7B** 的挑战，并讨论了以时间换空间的方案。
  
  - 多位成员建议使用 [FSDP2](https://github.com/pytorch/pytorch) 等工具以及 **activation checkpointing** 等技术来优化训练期间的内存占用。
- **探索优化器与参数 Offloading**：关于在单个 GPU 上使用可以 Offload 梯度的优化器的讨论，引出了对 **CPU offload optimizer** 的建议，该优化器能够在 **16GB GPU** 上进行全量微调（Full Fine-tuning），尽管速度较慢。
  
  - 成员们强调了训练期间用于预取（Prefetching）的 **dependency graph** 的重要性，并参考了 [FSDP2 源码](https://github.com/pytorch/pytorch/blob/a919742149601888c793447c1a6ab262979f1dde/torch/distributed/fsdp/_exec_order_utils.py)中的实现细节。
- **高效 GPU 使用的内存管理技术**：讨论的潜在内存优化策略包括仅在必要时加载数据，以及使用 **gradient checkpointing** 仅存储激活值的子集，从而在训练期间节省内存。
  
  - 对话还包括如何确定哪些层或激活值应保留在内存中，以及在 LLM 训练期间对 **model sharding** 的考量。
- **关于 CUDA 和 PyTorch 资源的初学者见解**：一位新成员分享了观看 GPU mode 讲座以及研究 **Zero-Offload** 和 **vLLM** 等论文的经验，以更好地理解大模型训练中的内存问题。
  
  - 他们询问了是否有其他资源或博客可以提供关于使用 **PyTorch** 有效管理 GPU 内存的进一步见解。
- **对工程挑战的幽默调侃**：关于在模型训练期间维持状态的困难，有一些轻松的评论，包括建议采用更简单的方法，如将 **embedding** 和 **LM head** 保留在内存中。
  
  - 几位参与者对构建用于模型训练的预取图（Prefetch Graph）所需付出的努力感到好笑，同时也承认应对这些工程挑战的乐趣。

**提到的链接**：

- [torchtune/recipes/lora_finetune_single_device.py at main · pytorch/torchtune](https://github.com/pytorch/torchtune/blob/main/recipes/lora_finetune_single_device.py#L59.)：PyTorch 原生微调库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。
- [GitHub - pytorch/torchtune: PyTorch native finetuning library](https://github.com/pytorch/torchtune#memory-and-training-speed)：PyTorch 原生微调库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。
- [torchtune/recipes/lora_finetune_single_device.py at main · pytorch/torchtune](https://github.com/pytorch/torchtune/blob/main/recipes/lora_finetune_single_device.py#L59)：PyTorch 原生微调库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。

---

### **GPU MODE ▷ #**[**cool-links**](https://discord.com/channels/1189498204333543425/1189868872887705671/1294310260240158810) (3 条消息):

> - `SF Tech Week 活动`
> - `INTELLECT-1 去中心化训练模型`

- **SF Tech Week 的轻松聚会**：一个小组正将其 SF Tech Week 活动转向在 sf parc 举行的舒适聚会，欢迎设计师、开发者和研究人员参加，旨在建立友谊而非社交压力。
  
  - *着装要求：“时髦的科技兄弟/姐妹”（dripped-out technology brother/sister）*，营造轻松的氛围，强调倾听而非推销。
- **面向开源 AGI 的 INTELLECT-1 发布**：[INTELLECT-1](https://x.com/PrimeIntellect/status/1844814829154169038) 被宣布为首个 100 亿参数模型的去中心化训练，其规模比之前的尝试扩大了 10 倍。
  
  - 邀请扩展到任何有兴趣为开源 AGI 发展做出贡献的人，以蝴蝶表情符号 🦋 为象征。

**提到的链接**：

- [来自 Prime Intellect (@PrimeIntellect) 的推文](https://x.com/PrimeIntellect/status/1844814829154169038)：宣布 INTELLECT-1：史上首次 10B 模型的去中心化训练。将去中心化训练规模扩大到之前努力的 10 倍。任何人都可以加入我们，共同构建开源 AGI 🦋
- [RSVP to deep galactic chillout | Partiful](https://partiful.com/e/E3vlROClsMSZWsxPTDyD)：我们觉得 SF Tech Week 的所有人都在搞 Hackathon，所以我们应该做点别的事情来收尾。因此，就像旧金山的初创公司经常 “pivot”（转型）一样，我们将我们的活动转向一个...

---

### **GPU MODE ▷ #**[**youtube-recordings**](https://discord.com/channels/1189498204333543425/1198769713635917846/1294358904167207031) (2 条消息):

> - `讲座脚本/笔记`
> - `PyTorch CUDA & memory profiler`

- **关于讲座资料可用性的查询**：*这次讲座非常有帮助*；一位成员询问脚本或笔记是否已上传。
  
  - 另一位成员表示遗憾的是，**这从未实现**，建议查阅 [PyTorch CUDA & memory profiler 文档](https://pytorch.org/docs/stable/cuda.html)以获取指导。
- **关于 Profiling 策略的建议**：为了明确 Profiling 过程，一位成员建议保持被 Profile 的区域尽可能短，最好限制在单次 forward-backward 传递中。
  
  - 他们鼓励从几个 `about:tracing` Chrome Traces 开始，使整体策略更加清晰。

---

### **GPU MODE ▷ #**[**torchao**](https://discord.com/channels/1189498204333543425/1205223658021458100/1294011920793342012) (25 条消息🔥):

> - `torchao 中的量化`
> - `comfyui 生态系统的问题`
> - `torch.export()`
> - `cuBLAS 限制`
> - `torchao 的性能`

- **torchao 中的量化困扰与偏好**：一位成员表示，他们更倾向于使用 **torchao** 以获得更好的量化效果，但质疑为什么量化结果是一个 tensor 子类，并建议需要一个简单的 **int8 tensor** 替代方案。
  
- **关于 comfyui 生态系统实现的担忧**：有人担心 **comfyui** 生态系统的实现过于糟糕，不值得投入时间去排查其问题。
  
  - 另一位成员表示反对，认为尽管存在上述挑战，仍值得继续尝试。
- **torch.export() 与 tensor 操作的挑战**：一位成员在处理 tensor 变异（mutations）时遇到了 **torch.export()** 的问题，具体需要使用 `unwrap_tensor_subclass` 来解决。
  
  - 他们确认在 **torch 2.4.1** 上取得了成功，尽管讨论中提到了需要保持版本更新。
- **关于 cuBLAS 和 IMMA kernel 限制的讨论**：一位成员引用了 **torchao** 中与 cuBLAS 限制相关的断言，并希望获得这些 kernel 衍生规则的文档。
  
  - 他们指出这些规则似乎与 **cuBLAS** 文档中发现的 FP8 要求一致。
- **torchao 的性能见解**：一位成员分享道，虽然 **torchao** 的速度较慢（**6.68s/it**），但运行效果良好，不过仍需修复编译（compilation）和其他问题。
  
  - 讨论强调了需要融合量化（quantization）和反量化（dequantization）操作以提升性能。

**提及的链接**：

- [torch.export() fails on aten.to(..., copy=True) followed by mutation · Issue #131679 · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/131679)：复现代码：import torch class Mod(torch.nn.Module): def forward(self, x): y = torch.ops.aten.to(x, dtype=torch.float16, copy=True) y.mul_(2) return y x = torch.randn(4) m = torch.export.export(Mod(), (...
- [Resubmit _int_mm by cpuhrsch · Pull Request #96685 · pytorch/pytorch](https://github.com/pytorch/pytorch/pull/96685)：避免对 gemm_and_bias 进行任何更改 cc @soumith @voznesenskym @penguinwu @anijain2305 @EikanWang @jgong5 @Guobing-Chen @XiaobingSuper @zhuhaozhe @blzheng @Xia-Weiwen @wenzhe-nrv @jiayisunx @peterbe...
- [ao/test/integration/test_integration.py at 10601b3ece80f6aba856556f73bf98a21a52f1df · pytorch/ao](https://github.com/pytorch/ao/blob/10601b3ece80f6aba856556f73bf98a21a52f1df/test/integration/test_integration.py#L87)：PyTorch 原生量化和稀疏化，用于训练和推理 - pytorch/ao

---

### **GPU MODE ▷ #**[**off-topic**](https://discord.com/channels/1189498204333543425/1215328286503075953/1294233160778776618) (1 条消息):

> - `早午餐食谱`
> - `柳兰茶 (Fireweed Tea)`
> - `牛肉汉堡`

- **美味的早午餐组合**：一位成员享用了一顿早午餐，包括 **1 升柳兰茶**（加牛奶和甜菊粉），搭配 **3 个**由牛肉饼和新鲜番茄片制成的汉堡。
  
  - 这种组合提供了丰富的口味，并为开启新的一天提供了能量。
- **咸鲜牛肉汉堡**：这 **3 个汉堡** 使用了**牛肉汉堡饼**制作，带来了咸鲜的口感，并搭配了番茄片。
  
  - 这一选择增加了清新感，并与汉堡的丰实感相得益彰。

 

---

### **GPU MODE ▷ #**[**rocm**](https://discord.com/channels/1189498204333543425/1233704710389764236/1294242911914823761) (3 条消息):

> - `ROCm Windows 支持`
> - `PyTorch Issue 讨论`

- **ROCm 现已原生支持 Windows**：从 **6.3** 版本开始，ROCm 推出了对 **Windows 的原生支持**，使 AMD 用户能够更无缝地利用其功能。
  
  - 这一转变让 Windows 用户能够更广泛地访问 GPU 技术，令人兴奋，详情见 [GitHub issue](https://github.com/pytorch/pytorch/issues/106608)。
- **对 Windows 支持声明的澄清**：一位成员对关于 Windows 支持的声明是否足够明确提出了质疑，揭示了沟通不够清晰的问题。
  
  - 这引发了关于 ROCm 与 Windows 兼容性的文档清晰度和功能发布公告的更广泛讨论。

 

**提及的链接**：[ROCm & Windows Support · Issue #106608 · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/106608)：🚀 功能、动机和宣传 AMD 已发布 ROCm Windows 支持，如 docs.amd.com 所示：请为 AMD GPU 上的 Windows 添加 PyTorch 支持！替代方案：无回应。补充背景：无回应.....

 

---

### **GPU MODE ▷ #**[**intel**](https://discord.com/channels/1189498204333543425/1233802893786746880/1294064537699483648) (2 条消息):

> - `Codeplay vs Coldplay`
> - `Intel's acquisitions`（Intel 的收购）
> - `Imagination/PowerVR`
> - `GPU backend compiler`（GPU 后端编译器）

- **Codeplay 误解**：一位成员幽默地指出了将 **Codeplay** 误认为 **Coldplay** 的错误，并对 Intel 一些奇怪的收购表示困惑。
  
  - *Imagination/PowerVR* 在 **10-15 年前** 曾聘请 Codeplay 为其在 SGX/Rogue 上的 GPU 后端编译器工作，突显了他们的高能力和高成本。
- **对历史合作的反思**：后续评论承认了之前关于 Coldplay 的错误，并确认指的确实是 Codeplay。
  
  - 这段轻松的对话揭示了双方在 GPU 领域的共同历史以及对 Codeplay 能力的认可。

---

### **GPU MODE ▷ #**[**arm**](https://discord.com/channels/1189498204333543425/1247232251125567609/1294395942652612739) (1 条消息):

> - `SIMD programming on ARM`（ARM 上的 SIMD 编程）
> - `OpenCL vs SIMD Intrinsics`
> - `RK3588 platform performance`（RK3588 平台性能）

- **探索 ARM 上的 SIMD 编程**：关于入门 **ARM 上的 SIMD 编程** 的普遍共识正在讨论中，成员们分享了他们的经验和见解。
  
  - *对最佳路径的好奇* 激发了对各种框架和方法的兴趣。
- **OpenCL vs SIMD Intrinsics 的抉择**：一位成员提出了在 **RK3588** 等平台上使用 **OpenCL** 与 **SIMD Intrinsics** 相比的有效性问题。
  
  - 这种不确定性源于一种方法在性能方面是否比另一种具有显著优势。
- **SIMD 编程的首选框架**：讨论包括了关于在 ARM 上进行 SIMD 编程的 **首选框架** 的询问。
  
  - 成员们根据自己的经验对不同的工具发表了不同的看法，强调了实际实现。

---

### **GPU MODE ▷ #**[**self-promotion**](https://discord.com/channels/1189498204333543425/1288557096404516945/1294028177512992809) (1 条消息):

> - `PyTorch Expert Exchange`
> - `Streaming LLM`
> - `Efficient Language Models`（高效语言模型）

- **肖光宣（Guangxuan Xiao）介绍 Streaming LLM**：下一期 **PyTorch Expert Exchange** 系列将由肖光宣于 **太平洋标准时间 10 月 11 日上午 10 点** 讨论 [StreamingLLM](https://github.com/mit-han-lab/streaming-llm)。
  
  - 查看 **GitHub 仓库** 了解更多关于名为 *Efficient Streaming Language Models with Attention Sinks* 项目的详细信息。
- **关于高效 Streaming LLM 的 YouTube 视频**：配套的 [YouTube 视频](https://www.youtube.com/watch?v=RnM84Sv9WpA) 标题为 “Efficient Streaming Language Models with Attention Sinks”，详细阐述了 **LLM 在流式传输中** 的讨论和应用。
  
  - 视频包含了 **肖光宣** 关于在流式应用中有效部署大语言模型的见解。

**提到的链接**：

- [GitHub - mit-han-lab/streaming-llm: [ICLR 2024] Efficient Streaming Language Models with Attention Sinks](https://github.com/mit-han-lab/streaming-llm)：[ICLR 2024] Efficient Streaming Language Models with Attention Sinks - mit-han-lab/streaming-llm
- [Efficient Streaming Language Models with Attention Sinks](https://www.youtube.com/watch?v=RnM84Sv9WpA)：由 MIT EECS 的肖光宣演示的 Efficient Streaming Language Models with Attention Sinks。在流式应用中部署大语言模型（LLMs），例如...

---

### **Nous Research AI ▷ #**[**general**](https://discord.com/channels/1053877538025386074/1149866623109439599/1294031439259766847) (44 条消息🔥):

> - `Llama 3.2 微调问题`
> - `Speculative Decoding 算法`
> - `集合论讨论`
> - `象形文字包含问题`
> - `OpenAI 提示词生成`

- **Llama 3.2 微调问题引发关注**：一位用户在对 **Llama 3.2 1B 模型**进行全量微调（full finetuning）时遇到冻结现象，这表明可能存在 NCCL 问题，特别是与所使用的数据集有关。
  
  - 另一位成员提到使用 **Llama 3 8B QLoRA** 时表现正常，暗示微调过程可能存在配置问题。
- **比 Groq 更快的新 Speculative Decoding 算法**：一位成员宣布他们的新 **Speculative Decoding 算法**在速度上超越了 Groq，引发了对其细节的关注。
  
  - 其他成员请求更深入的解释，对这一资源效率方面的进展感到兴奋。
- **集合论 ZFC 公理的专业知识探讨**：在回答关于 **ZFC 公理**的查询时，多位成员表示了解有限，并对该话题表达了好奇。
  
  - 一位参与者明确表示主要熟悉**选择公理（axiom of choice）**，表明对更广泛的集合论讨论感兴趣。
- **包含象形文字引发疑问**：关于在处理过程中包含象形文字的担忧被提出，有人推测可能存在**奇怪的编码问题**。
  
  - 成员们讨论了此类内容对输出的不确定影响，最终结论尚无定论。
- **OpenAI 的提示词生成元提示（Metaprompt）**：一位成员分享了关于 **OpenAI 用于生成任务系统提示词的元提示**细节，暗示未来可能与 DSPy 集成。
  
  - 他们提供了 OpenAI 文档的链接，预示着提示词生成方法论中令人期待的发展。

**提到的链接**：

- [Jashancutie Bumsekichu GIF - JashanCutie BumSeKichu - Discover & Share GIFs](https://tenor.com/view/jashancutie-bumsekichu-gif-15661510994914397092)：点击查看 GIF
- [Reddit - Dive into anything](https://reddit.com/r/LocalLLaMA/comments/1fzduyx/merging_llama_32_vision_adapters_onto_31_finetunes/)：未找到描述
- [GitHub - stillmatic/entropix: Entropy Based Sampling and Parallel CoT Decoding](https://github.com/stillmatic/entropix)：基于熵的采样和并行 CoT 解码。通过在 GitHub 上创建账号为 stillmatic/entropix 的开发做出贡献。

---

### **Nous Research AI ▷ #**[**ask-about-llms**](https://discord.com/channels/1053877538025386074/1154120232051408927/1294069287396315137) (9 条消息🔥):

> - `O1 的使用场景`
> - `O1 在编程中的表现`
> - `O1 的迭代能力`
> - `任务中 O1 与 GPT-4o 的对比`

- **探索 O1 的使用场景**：一位成员询问 **O1** 的最佳使用场景，注意到它在编程方面的有效性，但对其其他优势感到好奇。
  
  - 回复显示一种普遍观点，即 **O1** 主要在**数学**方面表现出色，而在编程方面的效用有限。
- **对代码生成的失望**：一位成员对 **O1** 在**非平凡代码生成（non-trivial code generation）**中的表现表示失望，称通常独立编写代码反而更容易。
  
  - 另一位参与者也表达了同样的挫败感，由于 **O1** 不愿进行多轮步骤，在**迭代编程任务**中感到吃力。
- **散文迭代的挑战**：用户注意到 **O1** 在**散文迭代**方面也表现不佳，特别是在后期迭代中无法很好地遵循指令。
  
  - 这一趋势引发了对其在编程和**创意写作任务**中整体可靠性的担忧。
- **性能对比分析**：一位成员分享了私下评估，强调 **GPT-4o** 在直接回答任务中更胜一筹，特别是在**数学练习**中。
  
  - 他们指出 **O1 Mini** 在编程任务上比 GPT-4o 略有优势，而 **O1 Preview** 在 **PAL 方法**中表现出色。

---

### **Nous Research AI ▷ #**[**interesting-links**](https://discord.com/channels/1053877538025386074/1132352574750728192/) (1 条消息):

vincentweisser: [https://github.com/openai/mle-bench/](https://github.com/openai/mle-bench/)

---

### **Cohere ▷ #**[**discussions**](https://discord.com/channels/954421988141711382/954421988783444043/1294097986166460497) (3 条消息):

> - `社区问候`

- **社区内的热烈欢迎**：成员们互相问候，一位成员热情地打招呼“hello everyone”。
  
  - *Competent* 给予了积极回应，营造了友好的氛围。
- **保持友好氛围**：消息交流为社区奠定了轻松的基调，展示了对话的开放性。
  
  - 这些交流表明成员们已准备好进行参与和互动。

---

### **Cohere ▷ #**[**questions**](https://discord.com/channels/954421988141711382/1168411509542637578/1294236659256131648) (2 messages):

> - `Cohere API v1 and v2`
> - `Web search connector`
> - `API documentation navigation`

- **Web Search Connector 指南**: 一位成员询问如何为 API 请求启用 **Internet search tool**，并提到在文档中难以找到相关说明。
  
  - 另一位成员澄清说 **web search connector** 在 **v1 API** 中可用，关于 v2 的迁移选项已在 [Cohere 迁移指南](https://docs.cohere.com/docs/migrating-v1-to-v2#web-search) 中讨论。
- **API 版本之间的差异**: 讨论强调了从 **Cohere API v1** 迁移到 **v2** 时的关键差异，特别是 v2 中将 `model` 和 `embedding_types` 作为必填字段。
  
  - 还提到了消息结构的变化，从 v1 的独立参数转变为 v2 中统一的 `messages` 参数。

 

**提到的链接**: [Migrating From API v1 to API v2 — Cohere](https://docs.cohere.com/docs/migrating-v1-to-v2#web-search): 该文档为希望将其现有的 Cohere API v1 实现更新到新的 v2 标准的开发者提供参考。

 

---

### **Cohere ▷ #**[**api-discussions**](https://discord.com/channels/954421988141711382/1168578329423642786/1294214207847403622) (12 messages🔥):

> - `V2 API Performance`
> - `Cohere API Toolcall Issue`
> - `API Token Usage`

- **确认 V2 API 比 V1 慢**: 用户报告称 **v2 API** *持续且明显地慢于* **v1**，响应时间现在达到 **2-3 秒**，而 v1 为 **1-1.5 秒**。
  
  - 一位用户指出，即使根据分享的数据，v2 的响应时间似乎也有明显的延迟。
- **Cohere API 的 GitHub issue 已关闭**: 一位用户在 toolcall 方面遇到了 **Cohere API** 的性能问题，但提到相关的 **GitHub issue** 已经关闭。
  
  - 他们询问有关该**问题**的见解以及任何潜在的**解决方案**，因为他们使用的是 **5.11.0** 版本。
- **关于在 Cohere 中使用文本限制的澄清**: 一位用户询问如何根据特定文本限制 **Cohere**，并寻求在 Cohere 和 **Gemini** 之间的建议。
  
  - 另一位用户询问关于按文本限制的具体含义，以及这是否与 **system prompt** 有关。
- **API 请求中 token 的使用**: 关于在 **API 请求** 中使用类似 `<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>` 等 token 的必要性产生了疑问，用户想知道如果没有这些 token，响应是否仍然理想。
  
  - 讨论强调了在有效使用 API 时理解 token 要求的重要性。

 

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1294022145273368730) (1 messages):

> - `AI Builders Night`
> - `Lightning Demos`
> - `Multi-Agent Systems`
> - `Zoom Developer Platform`
> - `Meetup speakers`

- **Zoom 总部的 AI Builders Night**: 欢迎周一参加在 [Zoom 总部](http://developers.zoom.us)（圣何塞）举办的 **AI Builders Night**，届时来自 [LlamaIndex](https://www.llamaindex.ai/) 的 **Biswaroop Palit** 将出席。他将讨论生产环境中的 **multi-agent systems**，并分享来自 **QDrant** 的见解。
  
  - 不要错过与其他开发者建立联系并参与有关 AI 进展的启发性讨论的机会。
- **在 Lightning Demos 中展示你的作品**: 见面会将包括 **lightning demos** 环节，参与者可以展示他们使用 [Zoom Developer Platform](http://developers.zoom.us) 构建的 **AI-powered use cases**。这是一个展示创新项目并获取反馈的绝佳机会。
  
  - 鼓励参与者使用标签 **#ZoomDevelopers** 在社交媒体上分享活动的精彩瞬间。
- **见面会演讲者阵容揭晓**: 精彩的演讲等待着大家，演讲者包括来自 [LlamaIndex](https://www.llamaindex.ai/) 的 **Biswaroop Palit** 和来自 [QDrant](https://qdrant.tech/) 的 **Thierry Damiba**。他们的演讲将为当前的 AI 趋势和应用提供宝贵的见解。
  
  - 请务必在活动开始前查看他们的个人资料以获取更多信息，从而从这些讨论中获得最大收获。

 

**提到的链接**: [AI Builders Night @ Zoom HQ · Luma](https://t.co/N5myAG3gcT): Zoom Developers 很高兴能回到总部举办十月见面会。这次我们将邀请 LlamaIndex 和 QDrant。在即将到来的见面会中……

 

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1294038094634422365) (13 条消息🔥):

> - `Workflows 实现`
> - `面向 Agentic Workflows 的 Symphony 自动化`
> - `OpenAI Batch API 限制`
> - `Agent 记忆与查询响应`
> - `AI Mayhem V3 黑客松赞助`

- **澄清 Workflow 用途**：一位用户询问 Workflow 是应该为每个请求重新创建，还是应该重用单个实例。回复强调 Workflow 默认是 **stateless**（无状态）的，除非另有说明。
  
  - 提供了示例代码来演示 **stateless usage** 以及如何在 Workflow 运行之间 **persist state**（持久化状态）。
- **Symphony 自动化 AI Workflow**：一位成员分享了 **Symphony** 如何自动化 Agentic Workflow 开发，根据提供的工具和任务描述生成高性能的 Workflow。他们鼓励加入其 [Discord](https://discord.gg/eYZESR4nVG) 获取 API key，并参考 [Loom 视频](https://www.loom.com/share/8d613aa434cf4a829e93160d01df35ae?sid=5216da3d-dcad-461c-bd37-6ba6a3c882b9) 了解更多见解。
  
  - 这种方法为开发者提供了一种简化方式，无需大量手动设置即可创建高效的 AI Workflow。
- **在 Document Summary Index 中使用 OpenAI Batch API**：一位成员询问是否可以在 LlamaIndex 的 Document Summary Index 中使用 **OpenAI Batch API**。回复澄清该库是为效率而设计的，表明使用 Batch API 不符合其操作标准。
  
  - 语气中带有一种对潜在操作耗时的调侃式沮丧，暗示更倾向于更快速的替代方案。
- **Agent 记忆困惑**：一位用户对他们的 Agent 记忆响应没有按预期引用之前的上下文表示困惑。他们注意到主记忆和次级记忆之间的重叠可能是导致收到意外答案的原因。
  
  - 他们询问除了修改 System Prompt 之外，是否有其他方法可以引导 Agent 的响应。
- **AI 黑客松赞助机会**：来自 Zo World 的代表介绍了在旧金山和班加罗尔举行的 AI 黑客松 **AI Mayhem V3**，正在寻求赞助商和合作伙伴。他们概述了专门针对顶尖开发者的品牌曝光和社交机会。
  
  - 该成员询问了关于赞助联系人的指导，强调了参与这一跨洲活动的潜在影响力。

 

**提及的链接**：[使用 Symphony 设计端到端多 Agent AI Workflow 🤖](https://www.loom.com/share/8d613aa434cf4a829e93160d01df35ae?sid=5216da3d-dcad-461c-bd37-6ba6a3c882b9)：今天，我将指导你使用 Symphony 创建一个多 Agent AI Workflow。我们探索了 Perplexity 和 Image-to-Text 等可用工具，以及如何添加新工具。我展示了如何设计 Workflow、运行...

 

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general-help**](https://discord.com/channels/1104757954588196865/1110594519226925137/1294160806484639816) (4 条消息):

> - `大型多节点部署`
> - `微调 Llama-3-8B`

- **发现大型多节点部署的最佳地点**：推荐使用 **AWS** 等大型云服务商来部署大型多节点，只要节点位于同一区域即可。
  
  - 这种设置可以提供节点间的高效管理和连接。
- **多 GPU 微调速度未提升**：一位成员对使用两块 **3090** 微调 **Llama-3-8B** 表示沮丧，因为观察到速度与使用单块 **3090** 相比没有提升。
  
  - 他们注意到两块 GPU 的利用率都超过了 **98%**，这引发了对他们的设置以及 **DeepSpeed** 数据并行有效性的担忧。

 

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**axolotl-help-bot**](https://discord.com/channels/1104757954588196865/1225300056442409040/1294061232713891881) (9 条消息🔥):

> - `Llama Tokenizer Modification`
> - `SMILES String Processing`
> - `Molecule Design Optimization`

- **针对单字符 Token 的自定义 Llama Tokenizer**：一位成员发现，通过继承 `LlamaTokenizer` 类并重写其 `tokenize` 方法，可以实现将文本拆分为单字符 Token 的自定义 Llama Tokenizer。
  
  - 这种方法涉及修改 Tokenizer 的功能，以便在字符级别高效处理字符串，这对于分子设计等独特应用场景非常有益。
- **探索 Tokenization 变更的使用场景**：一位成员询问了字符级 Tokenization 的使用场景，另一位成员解释说，其目标是优化 LLM 以进行分子设计。
  
  - 他们对这种修改可能如何影响 SMILES 字符串的处理表示好奇，SMILES 字符串通常用于表示分子结构。
- **调整字符 Token 的序列长度**：有一条笔记指出，在进行字符级 Tokenization 时，可能需要调整模型的最大序列长度（maximum sequence length），以适应生成的更长序列。
  
  - 这在训练和推理时间方面可能具有重要意义，从而影响模型的整体性能。
- **该方法的独特性**：另一位成员强调了所讨论方法的独特性，表示对其潜在影响感兴趣。
  
  - 对话强调了在模型 Finetuning 和 Tokenization 技术方面对创新方法的协作探索。
- **使用 Tokenizer 处理 SMILES 字符串**：一位成员提供了一个 Tokenizer 处理分子 SMILES 字符串的可视化示例，展示了实际应用。
  
  - 他们评论说，虽然不期望修改 Tokenizer 会带来重大变化，但结果可能仍然很有趣。

 

**提到的链接**：[OpenAccess-AI-Collective/axolotl | Phorm AI Code Search](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=e78896c7-8633-4d7d-a98e-f82c2cd848c6)：更快速地理解代码。

 

---

### **DSPy ▷ #**[**show-and-tell**](https://discord.com/channels/1161519468141355160/1202371242519441499/1294114634332246097) (6 messages):

> - `Batch-GPT Tool`
> - `DSPy Onboarding Form`
> - `OpenAI DSPy Optimizations`
> - `Structured Outputs Project`
> - `AGI Automation Discussion`

- **Batch-GPT 将 OpenAI API 成本降低 50%**：一位成员分享了一个名为 Batch-GPT 的工具，该工具旨在利用 OpenAI 的 Batch API 将 API 成本降低 **50% 以上**，从而实现具有成本效益的集成。
  
  - 该项目是 **开源** 的，包含针对重复查询的自动缓存等功能，通过简单的代码集成即可轻松实现：`client = OpenAI(..., base_url="http://batch-gpt/v1")`。
- **DSPy 入门表单受到关注**：一位成员介绍了一个入门表单，该表单能有效地引导用户了解 DSPy 的构建模块，显示出其对新用户的帮助。
  
  - 自动化此过程以增强用户体验的潜力令人兴奋，并与 **AGI** 能力产生了轻松的关联。
- **OpenAI 将利用 DSPy 优化**：分享的一个链接强调 OpenAI 计划在未来实施 **DSPy 优化**，这标志着其服务的一个重要发展。
  
  - 这一消息受到了好评，表明人们对即将推出的 OpenAI 迭代版本在性能和效率提升方面持乐观态度。
- **结构化输出项目揭晓**：一位成员展示了 GitHub 上的一个项目 **dslmodel**，该项目专注于使用 **DSPy** 和 **Jinja2** 生成结构化输出，目前正开放贡献。
  
  - 该倡议旨在简化开发流程，使用户更容易在工作流中集成结构化输出。
- **通过自动化探索 AGI**：一位成员提出了一个有趣的问题：将问题自动转换为特定格式是否可以被视为迈向 **AGI** 的一步。
  
  - 这引发了关于结构化问题解决的重要性及其与通用人工智能背后原则一致性的讨论。

**提到的链接**：

- [Introduction to DSLModel Framework](https://www.loom.com/share/9b9b6964cbd6471c8f31616e4f939a6c): https://github.com/seanchatmangpt/dslmodel 大家好，我是 Sean Chatman，在这里介绍 DSL Model 框架，这是一个使用 DSPy 和 Jinja 进行数据建模的强大工具。在这个视频中，我解释了如何...
- [GitHub - seanchatmangpt/dslmodel: Structured outputs from DSPy and Jinja2](https://github.com/seanchatmangpt/dslmodel): 来自 DSPy 和 Jinja2 的结构化输出。通过在 GitHub 上创建账号来为 seanchatmangpt/dslmodel 的开发做出贡献。

---

### **DSPy ▷ #**[**papers**](https://discord.com/channels/1161519468141355160/1203568372667645963/1294397086980374598) (1 messages):

> - `In-Context Learning`
> - `GraphIC Technique`
> - `Bayesian Networks`
> - `ICE Selection`
> - `Multi-step Reasoning`

- **GraphIC 技术增强 ICL**：提出的 [GraphIC 方法](https://arxiv.org/abs/2410.02203) 利用基于图的表示以及 **Bayesian Networks**，通过选择最佳的 in-context 示例来改进 **In-context Learning (ICL)**。
  
  - 该技术减轻了浅层语义偏差，专注于对于需要多步推理的任务至关重要的深层推理结构。
- **传统 ICL 方法中的偏差**：传统的基于文本的 Embedding 方法在为复杂推理任务选择 in-context 示例 (ICEs) 时往往表现不佳，因为浅层语义会引入偏差。
  
  - 讨论强调，这些偏差可能会阻碍 LLM 在数学和逻辑问题解决等任务上的表现。
- **GraphIC 中的 Bayesian Networks**：Bayesian Networks 在 GraphIC 方法中起着关键作用，能够有效地捕获节点属性的依赖关系。
  
  - 这种结构允许对 ICEs 进行更精细的选择过程，从而增强 LLM 的整体推理能力。

 

**提到的链接**：[GraphIC: A Graph-Based In-Context Example Retrieval Model for Multi-Step Reasoning](https://arxiv.org/abs/2410.02203): In-context learning (ICL) 使大语言模型 (LLMs) 能够通过直接在输入中加入少量 in-context 示例 (ICEs) 来泛化到新任务，而无需更新参数。然而...

 

---

### **DSPy ▷ #**[**general**](https://discord.com/channels/1161519468141355160/1161519469319946286/1294346721782009888) (4 条消息):

> - `DSPy 模块 COT 调用`
> - `LiteLLM 的限流错误`
> - `LLM 分类器歧义处理`

- **DSPy 模块面临限流错误**：一位成员报告了在使用 DSPy 模块通过 **bedrock** 调用 **Claude Sonnet** 对 pandas DataFrame 中的输入字符串进行 COT 调用分类时，遇到了限流错误。
  
  - 他们注意到，在不使用 DSPy 直接访问 **bedrock API** 时没有限流问题，这表明集成环节可能存在潜在问题。
- **寻求限流错误的详情**：另一位成员请求提供有关 DSPy COT 调用期间遇到的具体错误的更多信息。
  
  - 这一询问表明社区有兴趣更好地了解底层问题，以解决潜在的疑虑。
- **在 LLM 结果中对歧义进行分类**：一位成员分享了使用 DSPy 训练 LLM 分类器的经验，但希望模型能够指出分类中的歧义，例如应当显示：*需要更多信息，类别 A 和 B 之间存在歧义*。
  
  - 他们询问是否建议为所有可能的歧义创建单独的类别，从而引发了关于处理细微分类结果的最佳实践的讨论。

 

---

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1294080279962456096) (6 条消息):

> - `Tinygrad 中的 Int64 索引`
> - `数据类型转换 (Casting)`
> - `nn/__init__.py 中的类型注解`

- **Int64 索引需要精度**：讨论了仅在可能溢出的 ALU 上应用 **int64 索引** 的问题，参考 ##6987。
  
  - 如果同时使用两种不同的数据类型，*Tinygrad* 会引发关注，这促使成员们考虑算子的兼容性。
- **GPU 上 Int64 运行缓慢促使类型转换**：由于担心 **int64** 在 GPU 上运行缓慢，成员们讨论了在不同数据类型之间进行转换的必要性。
  
  - 双方达成一致，仅在严格必要时才使用 **int64** 索引，以提高性能。
- **呼吁在 nn/init.py 中添加类型注解**：一位成员强调 **nn/init.py** 中的所有类都需要 **type annotations**（类型注解），并强调了其对代码清晰度的重要性。
  
  - George 建议这可以作为一个有前景的入门级 Pull Request 供贡献者处理。

 

---

### **tinygrad (George Hotz) ▷ #**[**learn-tinygrad**](https://discord.com/channels/1068976834382925865/1070745817025106080/1294042609970315335) (5 条消息):

> - `扩散策略学习 (Diffusion Policy Learning)`
> - `Examples 目录偏好`
> - `KAN 示例 PR 评审`

- **扩散策略展现出令人印象深刻的机器人学习能力**：关于 [Visuomotor Policy Learning via Action Diffusion](https://diffusion-policy.cs.columbia.edu) 的论文重点介绍了 **Diffusion Policy**，它改进了机器人行为生成，在各种基准测试中比现有方法平均提高了 **46.9%**。
  
  - *它巧妙地处理了多模态动作分布和高维动作空间*，利用随机 Langevin 动力学进行稳定训练。
- **偏好 examples 目录中采用单文件形式**：针对有关 `examples/` 目录组织方式的询问，George 表示，相比于多个较小的文件，更倾向于使用**单个文件**，但强调代码必须是**高质量**的。
  
  - 这一指导意见支持创建连贯、精炼且易于理解的示例。
- **KAN 示例正在评审中**：一位成员正在为仓库完善其 [KAN 示例](https://github.com/tinygrad/tinygrad/pull/6690/files)，实现了一个可以在 MNIST 上快速训练的 **FastKAN**。
  
  - 此外，他们还提到有一个可供迁移的 **TransformerKAN** 示例，展示了他们对 tinygrad 项目的持续贡献。

**提到的链接**：

- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://diffusion-policy.cs.columbia.edu)：本文介绍了 Diffusion Policy，这是一种通过将机器人的视觉运动策略表示为条件去噪扩散过程来生成机器人行为的新方法。我们对 Diffusion Policy 进行了基准测试...
- [mdaiter 提交的 FastKAN 示例 · Pull Request #6690 · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/pull/6690/files)：这实现了一个 FastKAN，详情见：https://arxiv.org/abs/2405.06721 训练速度极快！在此处对 MNIST 进行训练。此外，我还测试了其中包含的 Attention transformer 模块...

---

### **Torchtune ▷ #**[**general**](https://discord.com/channels/1216353675241590815/1216353675744641096/1294174187677552690) (5 messages):

> - `1.58B BitNet 模型实现`
> - `Gemma-2 多语言能力`

- **1.58B BitNet 模型的高效实现**：一位成员询问如何仅通过矩阵乘法中的加法而非乘累加 (multiply-accumulate) 来有效地实现 **1.58B BitNet 模型**。
  
  - 建议指出，高效实现需要硬件解决方案，因为使用配备 Tensor Cores 的 **NVIDIA GPU** 可能会提供更好的性能。
- **BitNet 基于整数的效率**：讨论透露 **BitNet 使用 int8 进行激活 (activations)**，如果通过硬件实现，整数加法可能比浮点加法更高效。
  
  - 成员们强调，重新思考所使用的操作可以带来更高效的模型。
- **Gemma-2 实现障碍**：一位成员询问 **Gemma-2 实现** 的更新情况，指出其具有前景广阔的多语言能力，但在使用 QLora 进行全量微调时存在问题。
  
  - 有人担心 **QLora 微调** 严重依赖参数选择，导致难以达到最佳性能。
- **呼吁支持 Gemma-2 微调**：另一位成员确认了对 **Gemma-2 微调支持** 的请求，并指向了一个关于其多语言能力的 [新 GitHub issue](https://github.com/pytorch/torchtune/issues/1813)。
  
  - 他们表达了希望通过协作勾勒出推进该模型支持所需的建模组件。

 

**提到的链接**：[support for gemma-2 · Issue #1813 · pytorch/torchtune](https://github.com/pytorch/torchtune/issues/1813)：您能否添加对 gemma-2 的微调支持？它具有良好的多语言能力，是英语以外语言微调的理想选择。其不同的尺寸也...

 

---

### **Torchtune ▷ #**[**papers**](https://discord.com/channels/1216353675241590815/1293438210097025085/1294396797514682419) (3 messages):

> - `Pixtral 12B`
> - `Aria 多模态模型`

- **Pixtral 12B 详细概述**：由作者团队发表的关于 [Pixtral 12B](https://arxiv.org/abs/2410.07073) 的论文概述了其开发过程和性能，强调了它在多模态 AI 领域的各种能力。
  
  - 作者包括 [Pravesh Agrawal](https://arxiv.org/search/cs?searchtype=author&query=Agrawal,+P) 等人，他们汇集了多方面的专业知识，以推动 AI 整合的边界。
- **Aria：开源多模态原生模型**：新模型 [Aria](https://arxiv.org/abs/2410.05993) 被介绍为一种开源多模态原生模型，拥有 3.9B 和 3.5B 的激活参数，在各种任务中表现出同类最佳的性能。
  
  - 它的表现优于 **Pixtral-12B** 和 **Llama3.2-11B**，展示了在 *语言理解* 和 *多模态任务* 方面的显著进步，使其成为专有模型的有力竞争替代方案。
- **多模态信息整合研究**：讨论集中在 **多模态原生 AI 模型** 有效整合和理解各种现实世界信息的必要性上。
  
  - 专有模型在适配方面的挑战强调了需要像 Aria 这样的开源方法来促进更广泛的采用和创新。

**提到的链接**：

- [Pixtral 12B](https://arxiv.org/abs/2410.07073)：我们推出了 Pixtral-12B，一个拥有 120 亿参数的多模态语言模型。Pixtral-12B 经过训练，可以理解自然图像和文档，在各种多模态任务上实现了领先的性能...
- [Aria: An Open Multimodal Native Mixture-of-Experts Model](https://arxiv.org/abs/2410.05993)：信息以多种形式呈现。多模态原生 AI 模型对于整合现实世界信息并提供全面理解至关重要。虽然专有的多模态原生模型...

---

### **Interconnects (Nathan Lambert) ▷ #**[**news**](https://discord.com/channels/1179127597926469703/1179128538679488533/1294375043123384422) (4 messages):

> - `OpenAI O1 Replication`
> - `Journey Learning Paradigm`
> - `Dowehaveopeno1.com`

- **关于复现 OpenAI O1 的深度报告**：一份详细的技术报告介绍了一种名为 **'journey learning'** 的新训练范式，用于复现 OpenAI 的 **O1 模型**，该范式在数学推理中结合了搜索、学习与试错过程。亮点包括仅使用 **327 个训练样本** 就实现了 **8% 的性能提升**。
  
  - 该报告记录了复现过程中的见解、挑战和创新方法，重点探索了高级推理能力和 **journey learning** 机制。
- **关于 Dowehaveopeno1.com 的讨论**：一名成员建议是时候创建 **dowehaveopeno1.com** 域名了，可能将其作为与 O1 复现进度相关的资源。这个想法遭到了对其可行性的一些质疑。
  
  - 另一名成员表达了对进度的肯定，但最终认为创建该域名的时机尚未完全成熟。

**提及的链接**：[Pengfei Liu (@stefan_fee) 的推文](https://x.com/stefan_fee/status/1844775434740809794)：第一份关于复现 OpenAI o1 的深度技术报告！！！揭示试错见解和辛苦换来的教训。一些亮点：(1) 我们引入了一种新的训练范式...

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #**[**leaderboard**](https://discord.com/channels/1111172801899012102/1214705495974092810/1294068177159848016) (3 messages):

> - `Model PR submissions`
> - `GitHub contributions`

- **令人兴奋的模型进展**：一名成员对最新模型的进展表示感谢，并鼓励为 [模型的 handler 提交 PR](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#contributing)。
  
  - 他们指出，可以参考其他模型提供商现有的 PR。
- **贡献指南**：另一名成员分享了 [README 链接](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#contributing)，以获取有关参与项目贡献的更详细说明。
  
  - 该 README 包含了训练和评估用于函数调用（function calls）的 LLM 所需的指导。
- **致力于改进**：一名成员感谢他人的支持，并确认他们将致力于模型的改进。
  
  - 这体现了社区内的协作精神，成员们共同推动项目的进步。

**提及的链接**：[GitHub 上的 gorilla/berkeley-function-call-leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#contributing)：Gorilla：训练和评估用于函数调用（工具调用）的 LLM - ShishirPatil/gorilla

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #**[**discussion**](https://discord.com/channels/1111172801899012102/1111353033352294440/1294160586976002070) (1 messages):

> - `Symphony AI workflow`
> - `Discord collaboration`
> - `Loom video demonstration`

- **Symphony 轻松构建 AI 工作流**：**Symphony** 模型通过允许用户选择工具和描述任务，自动完成 **agentic workflow 开发**，然后将其转化为 AI 工作流。
  
  - 在这个 [Loom 视频](https://www.loom.com/share/8d613aa434cf4a829e93160d01df35ae?sid=5216da3d-dcad-461c-bd37-6ba6a3c882b9) 中有一个演示，展示了其功能。
- **加入我们的 Discord 获取 API 访问权限**：团队邀请成员加入他们的 **Discord** 频道，在那里可以申请使用 Symphony 的 **API key**。
  
  - 点击此处加入：[Discord 链接](https://discord.gg/eYZESR4nVG) 成为社区的一员。

**提及的链接**：[使用 Symphony 设计端到端多智能体 AI 工作流 🤖](https://www.loom.com/share/8d613aa434cf4a829e93160d01df35ae?sid=5216da3d-dcad-461c-bd37-6ba6a3c882b9)：今天，我将引导你使用 Symphony 创建一个多智能体（multi-agent）AI 工作流。我们探索了 Perplexity 和 image-to-text 等可用工具，以及如何添加新工具。我展示了如何设计工作流、运行...

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-questions**](https://discord.com/channels/1280234300012494859/1280370030609170494/1294022324852490322) (3 messages):

> - `Web Browser Agent Experimentation`
> - `Lab Study Resources`

- **Web Browser Agents 引起关注**：用户正在讨论高效 **web browser agents** 的潜在候选方案，其中 **Web Voyager** 被强调为一个值得进一步探索的有前景的选项。
  
  - 成员们渴望听到关于这些 agents 的任何实际操作经验。
- **何处寻找 Lab 学习材料**：一位成员询问了学习 labs 的最佳实践，随后有人提到使用 **slides 和补充阅读材料**。
  
  - 这引发了关于这些材料在准备过程中重要性的讨论。

 

---

### **LangChain AI ▷ #**[**general**](https://discord.com/channels/1038097195422978059/1038097196224086148/1294220376766808065) (2 messages):

> - `Raspberry Pi 5`
> - `Lightweight Vector Databases`
> - `Pinecone Vector DB`

- **为 Raspberry Pi 5 寻找轻量级向量数据库**：由于 RAM 资源有限，一位成员表示需要一个**轻量级**但有效的向量数据库，以便在 **Raspberry Pi 5** 上运行 **RAG** 设置。
  
  - 他们提到 **Chroma** 可能不合适，因为它主要将数据存储在 RAM 中，这会影响使用 **Ollama** 时的性能。
- **推荐 Pinecone 向量数据库**：另一位成员推荐使用 **Pinecone** 作为满足该成员需求的合适向量数据库。
  
  - 这一建议旨在解决在 Raspberry Pi 5 上使用 **Chroma** 的局限性。

 

---

### **OpenInterpreter ▷ #**[**O1**](https://discord.com/channels/1146610656779440188/1194880263122075688/1294021596956069941) (2 messages):

> - `ElevenLabs cost structure`
> - `Implementation of op3nai API`

- **计算 ElevenLabs 音频成本**：一位成员分享道，使用 **ElevenLabs 的 creator plan** 每月提供 **100k credits**，换算下来大约是 **833 credits** 或每分钟音频 **$0.18**。
  
  - 这一计算突出了从应用中生成一整分钟语音相关的原始成本。
- **询问 op3nai Real-Time API 集成情况**：另一位成员询问是否有人成功将 **op3nai real-time API 集成到 O1** 中。
  
  - 这个问题表明社区内有分享 API 集成相关经验的需求。

 

---

### **AI21 Labs (Jamba) ▷ #**[**jamba**](https://discord.com/channels/874538902696914944/1222916247063232553/1294021653663322203) (2 messages):

> - `Hugging Face model issues`
> - `CUDA multiprocessing`
> - `Docker configurations`
> - `A100 GPU usage`

- **Hugging Face AI21-Jamba-1.5-Mini 配置错误**：一位用户在 **Ubuntu** 上使用 **CUDA 12.4** 的 Docker 容器中运行 `torch.multiprocessing` 时，遇到了 **Hugging Face** 模型 **AI21-Jamba-1.5-Mini** 的错误。
  
  - 错误信息显示他们无法在 fork 的子进程中重新初始化 CUDA，强调了需要使用 'spawn' 启动方法。
- **Akash 上 A100 GPUs 的 Docker 镜像问题**：另一位用户报告在拥有两块 **A100** GPU 的 **Akash** 上运行 **Docker 镜像** 时遇到了类似问题。
  
  - 他们没有提供关于其设置的更多细节，但对所面临的配置问题表示了担忧。

 

---

---

---

---

---

---

{% else %}

> 完整的各频道详细分解已为邮件格式截断。
> 
> 如果您想查看完整分解，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})!
> 
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}