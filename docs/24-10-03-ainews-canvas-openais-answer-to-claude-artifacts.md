---
companies:
- openai
- cursor_ai
- daily
date: '2024-10-03T23:22:37.798235Z'
description: '**OpenAI** 发布了 **Canvas**，这是一款基于 **GPT-4o** 的增强型写作与编程工具，具备行内建议、无缝编辑及协作环境等功能。早期反馈将其与
  **Cursor** 和 **Claude Artifacts** 进行了对比，指出了其优势及部分实现上的问题。OpenAI 还资助了 **ProseMirror**
  和 **CodeMirror** 的开发者 **Marijn Haverbeke**，Canvas 正是采用了这些技术。在集成过程中，OpenAI 训练了一个检测器来适时触发
  Canvas，其触发准确率达到了 **83%**。与 Claude Artifacts 不同，Canvas 目前尚不支持 Mermaid 图表和 HTML 预览。此外，**Daily**
  正在旧金山赞助一场奖金达 **20,000 美元**的语音 AI 黑客松，凸显了语音 AI 作为一项关键新兴技能的重要性。'
id: 1d1543f1-9d3a-42fc-9b6e-0c40bf01a27b
models:
- gpt-4o
- claude-artifacts
original_slug: ainews-canvas-openais-answer-to-claude-artifacts
people:
- marijn-haverbeke
- karina-nguyen
- vicente-silveira
- swyx
title: '**Canvas：OpenAI 对标 Claude Artifacts 的产品**


  （或者：**Canvas：OpenAI 针对 Claude Artifacts 给出的回应**）'
topics:
- inline-suggestions
- collaborative-editing
- code-editing
- model-training
- model-integration
- feature-detection
- accuracy-evaluation
- voice-ai
- hackathon
- open-source-libraries
---

<!-- buttondown-editor-mode: plaintext -->**Chat-with-Artifacts 就是你所需的一切。**

> 2024年10月2日至10月3日的 AI 新闻。我们为你检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **31** 个 Discord 服务（**225** 个频道及 **1721** 条消息）。预计节省阅读时间（按 200wpm 计算）：**212 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 来参与 AINews 讨论！

在 Claude Artifacts 发布三个月后（[我们的报道见此](https://buttondown.com/ainews/archive/ainews-claude-crushes-code-92-humaneval-and/)），[OpenAI 发布了 Canvas](https://openai.com/index/introducing-canvas/)，这是一个基于 GPT-4o 的增强型写作和编程工具（[Mikhail Parakhin 也提到他们在 Bing Copilot 中发布了类似功能](https://x.com/MParakhin/status/1841916242229395539)）。根据发布公告，Canvas 包括：

- **行内建议：** Canvas 提供行内建议和直接操作，用于完善写作和代码，例如润色、修复 Bug 或移植代码。


![image.png](https://assets.buttondown.email/images/eeae39d2-c8bb-4a22-8641-8452a8614a4a.png?w=960&fit=max)


- **无缝编辑：** 它支持对大型文档和复杂代码库进行无缝编辑，使项目管理更加轻松。

- **协作环境：** 协作环境确保你的工作能够持续改进和演进。

快速浏览早期评论和反馈包括：

- Vincente Silveira [指出](https://x.com/vicentes/status/1841931637631942887)：
“看起来很棒，我们刚刚试用了它，并与 Cursor 和 Claude 进行了对比，它似乎将更多核心的编辑和编程用例带入了 ChatGPT，并为普通用户提供了更好的 UX。”

- 然而，Machine Learning Street Talk [在推特上指出了早期问题](https://x.com/MLStreetTalk/status/1841928399809286247)：
“OpenAI 克隆了 @cursor_ai 内部的功能，即 apply 模型。想法不错，但执行力较差——效果不太好。经常更新整个文档，而不是选中的部分。”

- Karina Nguyen（参与了 Canvas 的开发）[发布了几个使用 Canvas 进行写作和编程的示例](https://x.com/karinanguyen_/status/1841889811931791642)。

虽然早期的重点似乎在于写作场景，[并与 ChatGPT 现有的搜索功能良好集成](https://x.com/karinanguyen_/status/1841889814230061480)，但编程当然是与 Claude Artifacts 对比的重要维度，Karina 为这些任务内置了一些自定义工具。


![image.png](https://assets.buttondown.email/images/e5be389f-89ab-4397-ab28-126fce561a1f.png?w=960&fit=max)


[OpenAI 还将资助 Marijn Haverbeke](https://x.com/romainhuet/status/1841889813105971646)，他是用于构建 Canvas 的开源库 [ProseMirror 和 CodeMirror](https://marijnhaverbeke.nl/) 的创建者和维护者。


![image.png](https://assets.buttondown.email/images/4babbfd3-ad16-454a-8957-d5c03156d3c4.png?w=960&fit=max)


实现中最棘手的部分是 OpenAI 选择将其集成到现有 ChatGPT 体验中的方式，这涉及训练一个检测器，用于判断何时应开启 canvas 功能：

> **一个关键挑战是定义何时触发 canvas。** 我们训练模型在遇到诸如“写一篇关于咖啡豆历史的博客文章”之类的提示词（prompts）时打开 canvas，同时避免在“帮我做一道新的晚餐菜谱”等通用问答任务中过度触发。对于写作任务，我们优先提高了“正确触发”率（以牺牲“正确不触发”为代价），与带有提示指令的基准 zero-shot GPT-4o 相比，达到了 83%。他们也分享了他们的评估（evals）：


![image.png](https://assets.buttondown.email/images/ccd59cc8-a7d9-4e46-965b-e041959a0aa3.png?w=960&fit=max)


针对触发编辑行为和评论创建也做了类似的改进。这可能意味着 API 中的 `chatgpt-4o-latest` 模型也已更新。

与 Artifacts 不同，OpenAI Canvas 不支持显示 Mermaid 图表或 HTML 预览。据推测这些功能正在开发中，但令人好奇的是，这些功能既没有被优先考虑，也没有在两天前的 Dev Day 上发布（[Latent Space 的回顾见此](https://www.latent.space/p/devday-2024)）。

---

**由 Daily 赞助：** 如果你对对话式语音 AI（以及视频）感兴趣，请加入 [Daily 团队](https://www.daily.co/products/daily-bots/) 和开源 [Pipecat](https://github.com/pipecat-ai/pipecat) 社区，参加 10 月 19 日至 20 日在旧金山举行的 [黑客松](https://x.com/kwindla/status/1839767364981920246)。**20,000 美元奖金**将授予最佳语音 AI Agent、虚拟化身体验、多模态 AI UI、艺术项目以及我们共同构思的其他任何作品。

> **swyx**：语音 AI 是目前最热门的新 AI 工程技能！我也会参加——Daily 长期活跃在旧金山 AI 黑客松圈子里，这是我一段时间以来见过的为了学习我想精通的技术而设立的最高奖金池。

---

{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

> 所有摘要均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 与技术进步**

- **大语言模型 (LLMs) 与 AI 发展**：[@karpathy](https://twitter.com/karpathy/status/1841594123381571863) 分享了一个实验，他仅用 2 小时就利用 ChatGPT、Claude 和 NotebookLM 等 AI 工具策划了一个名为 "Histories of Mysteries" 的 10 集播客，展示了生成式 AI 带来的快速内容创作能力。[@cwolferesearch](https://twitter.com/cwolferesearch/status/1841557739308286424) 讨论了 o1（OpenAI 的最新模型）在自动 Prompt Engineering 方面的潜力，强调了其利用增加的推理时间计算（inference time compute）来实现更好推理的能力。

- **AI 在医疗保健领域**：[@bindureddy](https://twitter.com/bindureddy/status/1841611949622362435) 主张在医疗保健领域快速采用 AI，指出 AI 在**检索信息方面优于人类，且错误更少**。他们建议用 AI 取代人类医生可能会造福人类。

- **AI 模型进展**：[@OfirPress](https://twitter.com/OfirPress/status/1841509950679396387) 宣布 o1 在 SciCode 上创下了新的 SOTA，大幅领先 Claude。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1841497232253890937) 分享了关于 Nvidia NVLM-D 1.0 72B 模型的信息，该模型在数学和编程任务上的表现与 Llama 3.1 405B 持平。

- **AI 基础设施**：[@soumithchintala](https://twitter.com/soumithchintala/status/1841498799652708712) 详细解释了如何在 10,000 块 H100 GPUs 上训练模型，涵盖了并行化（parallelization）、通信优化和故障恢复策略等主题。

**AI 伦理与社会影响**

- **AI 安全**：[@NPCollapse](https://twitter.com/NPCollapse/status/1841523303397081414) 分享了一个关于利用 AI 为人类构建美好未来的资源，称其为迄今为止在该主题上最好的尝试。

- **AI 监管**：[@JvNixon](https://twitter.com/JvNixon/status/1841618859956306149) 对加州 AI 法律的潜在问题发表了评论，暗示这些法律可能侵犯言论和思想自由。

**AI 应用与工具**

- **AI 在软件开发中**：[@AlphaSignalAI](https://twitter.com/AlphaSignalAI/status/1841560745886069035) 发布了 Pythagora，这是一个 VScode 扩展，使用 14 个 AI Agents 来管理从规划到部署的整个开发流程。

- **AI 用于数据分析**：[@basetenco](https://twitter.com/basetenco/status/1841517280217182568) 引入了一种新的模型推理指标导出集成，允许轻松导出到 Grafana Cloud 等可观测性平台。

- **AI 在内容创作中**：[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1841469932610969907) 分享了一个在潜在空间（latent space）中使用 AI 进行取景（location scouting）的示例，展示了可视化不同时间和季节的能力。

**行业趋势与观点**

- **AI 公司估值**：[@RazRazcle](https://twitter.com/RazRazcle/status/1841563628170052025) 评论了 OpenAI 的快速增长，指出他们在 **2 年内实现了从 ~0 到 35 亿美元的营收**。

- **软件开发实践**：[@svpino](https://twitter.com/svpino/status/1841604832668614678) 批评了软件开发过度复杂化的趋势，呼吁回归更简单、更直接的应用程序构建方法。

- **AI 模型定价**：[@_philschmid](https://twitter.com/_philschmid/status/1841488046752997548) 分享了 LLM 定价的更新，指出包括 OpenAI、Google Deepmind、Cohere、Mistral 和 Cloudflare 在内的多家供应商都在大幅降价。


---

# AI Reddit 回顾

## /r/LocalLlama 回顾

**主题 1. Meta 发布 Llama 3.2：开源视觉模型的飞跃**

- **HuggingChat 模型更新！（Llama 3.2, Qwen, Hermes 3 等）** ([Score: 49, Comments: 15](https://reddit.com//r/LocalLLaMA/comments/1fuek0d/model_refresh_on_huggingchat_llama_32_qwen_hermes/))：**HuggingChat** 更新了其模型阵容，现在提供对 **五个新模型** 的免费访问，包括 **Qwen2.5-72B-Instruct**、**Llama-3.2-11B-Vision-Instruct**（具备 vision 能力）、**Mistral-Nemo-Instruct-2407**、**Hermes-3-Llama-3.1-8B** 和 **Phi-3.5-mini-instruct**。此外，还提供了 **两个** 启用了 **tool calling** 的模型：**Meta-Llama-3.1-70B-Instruct** 和 **c4ai-command-r-plus-08-2024**。
  - **Jamba Mini**，一个 12B 激活/52B 总参数的 **MoE** 模型，因其在 **256K** **context** 下的卓越表现和低 **hallucination rate** 而被推荐。它在本地运行具有挑战性，但可以由 HuggingChat 托管，尽管这可能需要 **vllm support** 或自定义代码。
  - 用户表达了尝试 **Jamba Mini** 的兴趣，HuggingChat 团队承认了其潜力，但指出 **lack of TGI support** 是一个重大问题。他们承诺会考虑这一建议。
  - 有人请求由 **thudm** 开发的 **LongWriter-glm4-9b**，该模型能够“一次性生成 10,000+ 字”。该模型被认为适合像 HuggingChat 这样拥有更好硬件的公司。


- **Meta Llama 3.2：视觉能力简析** ([Score: 244, Comments: 47](https://reddit.com//r/LocalLLaMA/comments/1fuj1o7/meta_llama_32_a_brief_analysis_of_vision/))：Meta 发布了两个 **multi-modal language models**，**Llama 3.2**，参数量分别为 **11B** 和 **90B**。作者测试了该模型在各种任务中的 vision 能力，包括 **image understanding**、**medical report analysis** 和 **chart analysis**，发现它是日常使用场景中的强力竞争者，并在某些应用中可能替代 **GPT-4o**，尽管 **GPT-4o** 在更复杂的任务中表现仍然更优。欲了解详细分析，作者建议读者阅读其关于 Llama 3.2 vision 能力的 [深度文章](https://composio.dev/blog/meta-llama-3-2-a-deep-dive-into-vision-capabilities/)。
  - 用户讨论了其他替代模型，如 **Qwen 2 VL 72B** 和 **Molmo**，一些人认为这些模型的表现优于 **Llama 3.2**。作者计划将 **90B** 模型与 **Qwen 2 VL 72B** 进行比较。
  - 该模型的文本提取能力被发现在 **标准文本中非常可靠**，但在 **发票或表格方面不够精确**。用户还对其生成物体坐标和处理带有叠加网格任务的能力表示了兴趣。
  - 由于本地硬件资源有限，作者使用 **Gradio** 和 **Together AI** 云服务来运行 **70B** 模型。一些用户分享了使用 **Gradio** 和 **Transformers** 实现其他模型（如 **Qwen 2 VL 72B**）的经验。


**主题 2：特定语言和特定任务模型的进展**



- **google/gemma-2-2b-jpn-it 日本特定模型** ([Score: 45, Comments: 15](https://reddit.com//r/LocalLLaMA/comments/1fv078k/googlegemma22bjpnit_japanese_specific_models/))：Google 发布了 **gemma-2-2b-jpn-it**，这是 Gemma 系列中的一个 **Japanese-specific model**，现已在 [Hugging Face](https://huggingface.co/google/gemma-2-2b-jpn-it) 上可用。这一新模型是在 **东京的 Gemma Developer Day 上宣布的**，表明 Google 正致力于为日本市场扩展特定语言的 AI 模型。
  - 正如 [task-specific tuning 文档](https://ai.google.dev/gemma/docs/spoken-language/task-specific-tuning) 中所解释的，该日本特定 Gemma 模型的 **pre-training** 是用日语进行的。目前没有提到关于 **9B 版本** 的计划或 Gemma 3 的发布日期。
  - Google CEO **Sundar Pichai** 意外现身 Gemma Developer Day，暗示了对该项目的强力支持。一位 **Hugging Face 代表** 也发表了讲话，暗示未来可能会有该模型的 **GGUF 版本**。
  - Google 介绍了几个与 Gemma 相关的工具，包括 **Responsible Generative AI Toolkit**、用于模型分析的 **Gemma Scope**，以及使用 MediaPipe 的 **on-device generation** 能力。还宣布了一项奖金为 **$150,000 的 Kaggle 竞赛**，旨在利用 Gemma 进行全球交流。

- **[Llama-3.1-Nemotron-70B-Reward](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Reward)** ([得分: 45, 评论: 6](https://reddit.com//r/LocalLLaMA/comments/1fv1e12/llama31nemotron70breward/)): 帖子标题 "**Llama-3.1-Nemotron-70B-Reward**" 似乎指向一个特定的 AI 语言模型，但正文中未提供额外内容或背景。由于缺乏进一步信息，无法对该帖子的内容或讨论点提供有意义的总结。
  - **Llama-3.1-Nemotron-70B-Reward** 在人类标注任务上的表现与 **Skywork-Reward-Gemma-2-27B** 相似，但在 **GPT-4** 标注的任务上落后。这表明 Skywork-Reward-Gemma-2-27B 更好地模拟了 GPT-4 的偏好，这可能是因为它是在 GPT-4 标注的数据上训练的。
  - 讨论中澄清了 **Reward-Gemma**（非 Gemini）与 GPT-4 生成的 "**Ground Truth**" 对齐得更好，但与人类的 "**Ground Truth**" 对齐较差。这归因于 Reward-Gemma 的训练数据包含了 GPT-4 生成的文本。
  - 该模型被描述为 "**RLHF**（人类反馈强化学习）领域新的同类最佳裁判"，因其在预测人类偏好方面的准确性而受到关注。


- **新排行榜：哪些模型最擅长角色扮演？** ([得分: 40, 评论: 7](https://reddit.com//r/LocalLLaMA/comments/1fugds7/new_leaderboard_which_models_are_the_best_at_role/)): 一个名为 **StickToYourRoleLeaderboard** 的新排行榜评估了 **LLM 在角色扮演场景中维持角色一致性的能力**。该排行榜可在 **Hugging Face** 上查看，旨在评估模型在讨论过程中遵循指定角色和角色价值观的程度，作者在 **X (原 Twitter)** 上发布了详细的解释说明。
  - 用户注意到 **Mistral** 在测试模型（**Llama 3.1-8b**、**Llama 3.2-3b**、**Qwen 2.5**、**Mythomax**）中表现最好，并强调了基础 Prompt 和模型参数的重要性。
  - 排行榜中遗漏了 **Mistral Nemo** 引起了关注，并有建议在基准测试过程中包含更多热门的微调模型。


**主题 3. AMD Strix Halo：本地 LLM 推理的潜在游戏规则改变者**



- **传闻 AMD Strix Halo APU 具备 7600 XT 性能及 96 GB 共享 VRAM** ([得分: 68, 评论: 39](https://reddit.com//r/LocalLLaMA/comments/1fv13rc/amd_strix_halo_rumored_to_have_apu_with_7600_xt/)): 据传 AMD 的 **Strix Halo** APU 性能可与 **Radeon 7600 XT** 媲美，并支持高达 **96 GB 的共享 VRAM**。这款高端笔记本芯片可能在不需要独立 AI GPU 的情况下在内存中运行大型语言模型。尽管目前 APU 缺乏官方 **ROCm** 支持，但 **Llama.cpp** 的 **Vulkan kernel** 对 APU 的支持速度与其它 AMD 硬件上的 **ROCm kernel** 相当。
  - AMD 缺乏具备 **CUDA** 支持的 **48GB VRAM** GPU 被视为 AI 市场的错失良机。**W7900-PRO** 虽然提供 48GB，但价格高达 **4000 美元**，这可能是为了避免削弱 AMD 的 **Instinct** 系列产品。
  - 传闻 **Strix Halo** APU 使用 **256-bit LPDDR5X-8000** 内存，提供 **256GB/s** 的理论带宽。有人推测其带宽可能达到 **500GB/s** 范围，可能还得益于针对游戏负载的 **3D cache**。
  - 目前的 **AMD APU** 在 **VRAM** 分配方面面临限制，仅允许最多 **8GB** 作为专用 **VRAM**。然而，一项名为 **Variable Graphics Memory** 的新功能允许 **AMD Ryzen™ AI 300 series** 处理器将高达 **75%** 的系统内存转换为“专用”显存。

- **Qwen 2.5 Coder 7b 用于自动补全** ([Score: 37, Comments: 17](https://reddit.com//r/LocalLLaMA/comments/1fuenxc/qwen_25_coder_7b_for_autocompletion/)): **Qwen 2.5 Coder 7b** 模型在处理**数千个 token 的大上下文**时，表现出优于其他本地模型的自动补全能力。用户报告称其幻觉显著减少，代码风格延续性有所提高，性能足以媲美 **Copilot**。若要在 **IntelliJ 的 ContinueDev 插件**中使用，需要自定义模板覆盖：`"<|fim_prefix|>&#123;&#123;&#123; prefix &#125;&#125;&#125;<|fim_suffix|>&#123;&#123;&#123; suffix &#125;&#125;&#125;<|fim_middle|>"`，并且为了确保控制 token 和 **FIM 支持**正常工作，使用 instruct 模型变体至关重要。
  - **Qwen2.5-7b-coder-q8_0.gguf** 在 Neovim 的 **C++ 自动补全**中表现出色，对于短补全，**Q8 量化**仅比 Q4 慢约 5%。用户 **ggerganov** 正在使用 **256 行前缀和 128 行后缀**作为上下文。
  - **Qwen2.5 7b-coder** 与 **14b-instruct** 模型的对比表明，尽管后者并非专门为编程训练，但更大的模型可能提供更好的上下文理解和代码解释能力。7b-coder 版本则是针对带有特殊 token 的自动补全进行了微调。
  - 关于在 Fill-in-the-middle (FIM) 任务中使用 **base 还是 instruct 模型**存在困惑，原帖作者报告在使用 base 模型时遇到问题。Qwen 的官方文档建议在 FIM 任务中使用 base 模型，这与用户的实际体验相矛盾。


**Theme 4. 用于 AI 开发和评估的开源工具**



- **Moshi 工作原理：开源实时语音 LLM 简明指南** ([Score: 52, Comments: 7](https://reddit.com//r/LocalLLaMA/comments/1fukiy6/how_moshi_works_a_simple_guide_to_the_to_open/)): **Moshi** 是 **OpenAI Voice mode** 的**开源替代方案**，由 **Kyutai** 开发，用于**语言模型**中的**实时语音**交互。作者分享了详细介绍 **Moshi 架构**的文章链接，并认为尽管它尚未达到 OpenAI 产品的水平，但仍值得深入了解。

- **[🧬 OSS 合成数据生成器 - 使用自然语言构建数据集](https://huggingface.co/spaces/argilla/synthetic-data-generator)** ([Score: 38, Comments: 3](https://reddit.com//r/LocalLLaMA/comments/1fv05ax/oss_synthetic_data_generator_build_datasets_using/)): 该帖子介绍了一个**开源合成数据生成器**，允许用户使用**自然语言提示词**创建数据集。该工具可在 [GitHub](https://github.com/HumanSignal/syndata) 上找到，支持为各种机器学习任务（如**分类、目标检测和分割**）生成包括**图像、文本和结构化数据**在内的多样化数据集。该生成器利用**大型语言模型**和**图像生成模型**，根据用户定义的规格生产高质量的合成数据。
  - **Hugging Face** 员工介绍了 **Distilabel Synthetic Data Generator**，这是一个通过**自然语言提示词**创建高质量数据集的开源工具。用户可以通过 [克隆 Space](https://huggingface.co/spaces/argilla/synthetic-data-generator?clone=true) 或安装 [distilabel 库](https://github.com/argilla-io/distilabel) 在本地运行。
  - 用户对该工具表示热烈欢迎，称赞其推动了 **“AI 商品化（AI as commodity）”范式**。创建者欢迎反馈，并提到计划在未来增加更多任务和功能。
  - 该工具简化了用于**训练和微调语言模型**的数据集创建过程，允许用户定义应用特征、生成系统提示词，并产出可直接推送到 **Hugging Face Hub** 的可定制数据集。

- **[“箴言 27:17：铁磨铁，磨出刃来；朋友相感，也是如此” “通过 Self-play 训练语言模型赢得辩论可提高评判准确性”](https://i.redd.it/5e5s9c2eibsd1.png)** ([Score: 35, Comments: 4](https://reddit.com//r/LocalLLaMA/comments/1fucp0l/proverbs_2717_as_iron_sharpens_iron_so_one_person/))：该论文介绍了 **DebateGPT**，这是一种通过 **Self-play** 训练进行辩论的语言模型，从而提高了评判辩论结果的准确性。通过让模型在各种话题上进行自我辩论，研究人员发现生成的评判模型在确定辩论获胜者方面的准确率达到了 **83%**，超过了人类评判员和之前的 AI 模型。这种方法展示了 Self-play 在增强语言模型辩论和分析能力方面的潜力。
  - **Self-play** 以及在语言模型中复制人类倾向（如 **Chain of Thought** (CoT) 和辩论式交互）被证明对提高任务性能非常有效。这些“简单”的过程变化通常会带来显著的性能提升。
  - 帖子正文中提供了论文链接 [https://www.arxiv.org/abs/2409.16636](https://www.arxiv.org/abs/2409.16636)，但由于应用在显示图片帖子文本时的限制，部分用户难以访问。
  - 讨论强调了在社交媒体平台的学术讨论中，正确的论文引用和链接实践的重要性。

## 其他 AI Subreddit 摘要

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

以下是所提供 Reddit 帖子中关键主题和进展的总结：

**AI 模型进阶与能力**

- **OpenAI 的 o1 模型** 正在展示令人印象深刻的推理和问题解决能力：
  - 它可以 [在几小时内复制复杂的博士级编程项目](https://www.reddit.com/r/singularity/comments/1fuff8e/in_awe_scientists_impressed_by_latest_chatgpt/)，而以前这需要数月时间。
  - 它在 [数学证明方面显示出可喜的结果](https://x.com/robertghrist/status/1841462507543949581?t=5zV3VpQI0mbrSU9_QRtfkQ&s=19)，表现优于之前的模型。
  - OpenAI 研究员 Hunter Lightman 表示，o1 [已经表现得像一名软件工程师，并能编写 Pull requests](https://www.reddit.com/r/singularity/comments/1futg5p/openais_hunter_lightman_says_the_new_o1_ai_model/)。

- **Google** 正在利用 Chain-of-thought 提示等技术开发 [类似于 OpenAI o1 的推理 AI](https://www.reddit.com/r/singularity/comments/1fuev51/google_is_working_on_reasoning_ai_bloomberg_news/)。他们已经展示了用于数学推理的 AlphaProof 等模型。

- **Salesforce** 发布了 [xLAM-1b，这是一个拥有 10 亿参数的模型，在 Function calling 中实现了 70% 的准确率](https://www.reddit.com/r/LocalLLaMA/comments/1dz8g10/salesforce_tiny_giant_xlam1b_model_surpasses_gpt/)，尽管体积较小，但仍超越了 GPT-3.5。

**AI 研究与开发**

- 一篇 [Google Deepmind 论文](https://arxiv.org/html/2406.17711v1) 展示了如何通过联合样本选择进行数据策展，从而加速多模态学习。

- [Microsoft 的 MInference 技术](https://arxiv.org/abs/2407.02490) 能够在保持准确性的同时，为长上下文任务实现高达数百万个 Token 的推理。

- 关于使用 10 亿个网络策展的角色（Personas）来 [扩展合成数据生成](https://www.reddit.com/r/MachineLearning/comments/1dzergu/r_scaling_synthetic_data_creation_with_personas/) 的研究，在生成多样化训练数据方面展现了前景。

**AI 行业与融资**

- OpenAI 正在 [寻求独家融资安排](https://www.reddit.com/r/singularity/comments/1fuls4t/sam_wants_exclusive_funding_arrangement_wow/) 以加速 AGI 的开发。

- NVIDIA 首席执行官 Jensen Huang 表示，[一万亿美元正被投入到数据中心](https://www.reddit.com/r/singularity/comments/1fuvuj1/nvidia_ceo_jensen_huang_says_a_trillion_dollars/)，以开启下一波提升业务生产力的 AI 浪潮。

**AI 伦理与社会影响**

- 关于 [AGI 可能导致的岗位取代](https://www.reddit.com/r/singularity/comments/1fuvuj1/nvidia_ceo_jensen_huang_says_a_trillion_dollars/lq2w3op/) 以及对新经济范式需求的讨论正在进行中。

- Sam Altman 建议 [对 ChatGPT 等 AI 助手保持礼貌](https://www.reddit.com/r/singularity/comments/1fukszd/saying_please_and_thank_you_to_chatgpt_probably_a/)，这暗示了未来 AI 意识或权利的潜在发展。

**AI 图像生成**

- 图像生成模型的新版本如 [PonyRealism v2.2](https://www.reddit.com/r/StableDiffusion/comments/1fuih8v/pony_realism_v22_is_out/) 和 [RealFlux](https://www.reddit.com/r/StableDiffusion/comments/1fv0b99/the_dev_version_of_realflux_realistic_vision/) 正在发布，展示了在写实性和功能方面的持续改进。

---

# AI Discord 摘要回顾

> 摘要之摘要的摘要

## Claude 3.5 Sonnet


**1. LLM 进展与基准测试**

- **DeepSeek-V2 挑战 GPT-4**：**DeepSeek-V2**，一款全新的 236B 参数模型，在 **AlignBench** 和 **MT-Bench** 等基准测试中表现出色，据报道在某些领域已超越 GPT-4。
   - [DeepSeek-V2 发布公告](https://x.com/deepseek_ai/status/1787478986731429933)引发了关于其能力及对 AI 领域潜在影响的热烈讨论，社区成员渴望探索其全部潜力。
- **Llama 3 在排行榜上的飞跃**：Meta 的 **[Llama 3](https://lmsys.org/blog/2024-05-08-llama3/)** 迅速攀升至 **ChatbotArena** 等排行榜的首位，在超过 50,000 场对决中表现优于 **GPT-4-Turbo** 和 **Claude 3 Opus** 等模型。
   - 这一迅速崛起引发了关于大语言模型演变格局的讨论，以及开源替代方案挑战该领域专有模型领导者的潜力。
  


**2. 优化 LLM 推理与训练**

- **ZeRO++ 大幅削减通信开销**：**[ZeRO++](https://www.deepspeed.ai/tutorials/zeropp/)** 承诺将 GPU 上大型模型训练的通信开销降低 4 倍，这可能彻底改变分布式训练的效率。
   - 这一进展可能显著影响 LLM 训练的可扩展性，使研究人员能够更快速、更具成本效益地训练更大的模型。
- **vAttention 的动态 KV 缓存**：**[vAttention](https://arxiv.org/abs/2405.04437)** 系统引入了 KV-cache 内存的动态管理，无需依赖 PagedAttention 即可实现高效的 LLM 推理。
   - 这一创新解决了 LLM 部署中的内存限制问题，可能实现在有限的硬件资源上更高效地提供大型模型服务。
- **Consistency LLMs 加速解码**：**[Consistency LLMs](https://hao-ai-lab.github.io/blogs/cllm/)** 等技术探索了并行 Token 解码以降低推理延迟，有望为 LLM 应用提供更快的响应速度。
   - 这种方法挑战了传统的自回归解码方法，为优化实时应用中的 LLM 性能开辟了新途径。
  


**3. 开源 AI 框架与社区努力**

- **Axolotl 扩展数据集格式支持**：**[Axolotl](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/)** 扩展了对多种数据集格式的支持，增强了其在指令微调和预训练 LLM 方面的能力。
   - 此次更新促进了各种数据源的轻松集成，使研究人员和开发人员能够利用自定义数据集更有效地微调模型。
- **LlamaIndex 与 Andrew Ng 联手**：**[LlamaIndex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex)** 宣布与 Andrew Ng 的 DeepLearning.ai 合作推出一门关于构建 Agentic RAG 系统的新课程，将学术见解与实际应用相结合。
   - 这一合作伙伴关系旨在使先进的 AI 技术大众化，让广大开发人员和研究人员更容易接触到 Agentic RAG 等复杂概念。
- **Mojo 展示 Python 集成潜力**：**[Modular 的全新深度解析](https://www.modular.com/blog/developer-voices-deep-dive-with-chris-lattner-on-mojo)** 中，Chris Lattner 展示了 Mojo 实现无缝 Python 集成以及 `_bfloat16_` 等 AI 特定扩展的潜力。
   - 讨论强调了 Mojo 将 Python 的易用性与系统编程能力相结合的雄心，这可能会重塑 AI 开发工作流。
  


**4. 多模态 AI 与生成式建模创新**

- **Idefics2 和 CodeGemma 突破边界**：**[Idefics2 8B Chatty](https://twitter.com/sanhestpasmoi/status/1787503160757485609)** 专注于提升聊天交互体验，而 **[CodeGemma 1.1 7B](https://twitter.com/reach_vb/status/1786469104678760677)** 则进一步优化了编程能力，展示了专用 AI 模型的进步。
   - 这些模型体现了将 AI 能力针对特定领域进行定制的持续趋势，增强了在对话式 AI 和代码生成等目标应用中的性能。
- **Phi-3 将 AI 引入浏览器**：**[Phi-3](https://www.reddit.com/r/LocalLLaMA/comments/1cn2zwn/phi3_webgpu_a_private_and_powerful_ai_chatbot/)** 模型通过 WebGPU 为浏览器引入了强大的 AI 聊天机器人功能，有可能彻底改变客户端 AI 应用。
   - 这一进展标志着向更易于访问且保护隐私的 AI 体验迈出了重要一步，使得直接在 Web 浏览器中进行复杂的 AI 交互成为可能。
- **IC-Light 照亮开源图像重照明领域**：开源项目 **[IC-Light](https://github.com/lllyasviel/IC-Light)** 专注于推进图像重照明（relighting）技术，使复杂的视觉效果对社区而言更加触手可及。
   - 该工具赋能创作者和研究人员探索先进的图像处理技术，可能在计算机图形学和视觉 AI 领域带来新的应用。
  

## GPT4O (gpt-4o-2024-05-13)


**1. Model Performance Optimization**

- **动态内存压缩提升吞吐量**：**[Dynamic Memory Compression (DMC)](https://arxiv.org/abs/2403.09636)** 在 **H100 GPU** 上将吞吐量提升了高达 **370%**，增强了 Transformer 的效率。
  - `@p_nawrot` 分享了关于 [DMC 论文](https://arxiv.org/abs/2403.09636) 的见解，引发了关于其对大规模模型训练影响的讨论。
- **ZeRO++ 减少 GPU 通信开销**：**[ZeRO++](https://www.deepspeed.ai/tutorials/zeropp/)** 承诺在 GPU 上进行大模型训练时，将通信开销降低 **4 倍**。
  - `@deep_speed` 强调了 [ZeRO++](https://www.deepspeed.ai/tutorials/zeropp/) 的优势，指出其在优化资源利用率方面的潜力。
- **Flash Attention 的内存使用引发讨论**：社区讨论了 **[Flash Attention](https://github.com/ggerganov/llama.cpp/pull/5021)** 在计算复杂度为平方级的情况下，是否表现出线性内存增长。
  - `@ggerganov` 指出 [Flash Attention](https://github.com/ggerganov/llama.cpp/pull/5021) 可以优化大模型中的内存使用。


**2. Fundraising and New Product Launches**

- **OpenAI 获得 66 亿美元融资**：**[OpenAI](https://www.perplexity.ai/page/openai-raises-6-6b-ofVMnsDdRw.cUWz28MxjBA)** 成功筹集了 **66 亿美元**，以支持其 AI 研究项目。
  - `@openai` 宣布了 [本轮融资](https://www.perplexity.ai/page/openai-raises-6-6b-ofVMnsDdRw.cUWz28MxjBA)，并讨论了其对未来 AI 进展的影响。
- **FLUX1.1 Pro 以速度取胜**：**[FLUX1.1 Pro](https://blackforestlabs.ai/announcing-flux-1-1-pro-and-the-bfl-api/)** 发布，提供 **快 6 倍的生成速度** 和改进的图像质量。
  - `@blackforestlabs` 分享了 [FLUX1.1 Pro 的发布消息](https://blackforestlabs.ai/announcing-flux-1-1-pro-and-the-bfl-api/)，在 AI 社区引发了兴奋和期待。
- **GPT-4o Realtime API 发布**：**[GPT-4o Realtime API](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/audio-real-time?pivots=programming-language-ai-studio)** 已发布，用于低延迟音频交互。
  - `@azure` 详细介绍了 [API 的发布](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/audio-real-time?pivots=programming-language-ai-studio)，重点关注客户支持等应用场景。


**3. AI Tooling and Community Innovations**

- **Crawl4AI 增强数据采集**：**[Crawl4AI](https://github.com/unclecode/crawl4ai)** 是一款开源 Web 爬虫，提供可定制的数据采集工具。
  - `@unclecode` 介绍了 [Crawl4AI](https://github.com/unclecode/crawl4ai)，讨论了其与语言模型的集成，以改进数据提取。
- **Mojo 的错误处理策略**：对话集中在 **[Mojo](https://github.com/msaelices/mojo-openai-realtime-api)** 的错误处理上，建议采用 **Zig 风格的错误联合类型 (error unions)**。
  - `@msaelices` 提出了对 Mojo 错误处理的 [改进建议](https://github.com/msaelices/mojo-openai-realtime-api)，强调模式匹配和组合性。
- **MongoDB Atlas 赋能混合搜索**：一篇关于使用 **MongoDB Atlas** [创建和配置混合搜索索引](https://t.co/VFsaL4XIdb) 的博客文章，旨在增强搜索相关性。
  - `@llama_index` 详细介绍了 [实现过程](https://t.co/VFsaL4XIdb)，将语义搜索与全文搜索合并，以解决常见的低效问题。


**4. AI Alignment and Research Discussions**

- **AI Reading Group 启动**：来自 Women in AI & Robotics 的 **[AI Reading Group](https://www.eventbrite.ca/e/1024976160287?aff=oddtdtcreator)** 启动，重点关注研究讨论。
  - `@aashka_trivedi` 将展示 [INDUS 论文](https://arxiv.org/abs/2405.10725)，重点介绍 **IBM** 与 **NASA** 之间的合作。
- **OpenAI 审核政策引发讨论**：成员们讲述了他们在 **OpenAI 审核政策** 方面的经历，指出了一些提示 AGI 的请求被标记。
  - `@eleuther` 指出这些政策似乎过于谨慎，并暗示许多被标记的信息并不符合其陈述的使用政策。
- **Softmax 函数的局限性探讨**：一篇论文强调了 **[Softmax 函数的局限性](https://arxiv.org/abs/2410.01104)**，即在输入规模增加时难以实现鲁棒计算。
  - `@nous_research` 分享了该[论文](https://arxiv.org/abs/2410.01104)，提出 **adaptive temperature** 作为这些局限性的解决方法。


**5. 开源贡献与协作**

- **Axolotl 添加数据集格式文档**：**[Axolotl](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/)** 支持多种数据集格式，用于指令微调和预训练 LLM。
  - `@axolotl_ai` 宣布了[文档更新](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/)，提升了社区的易用性。
- **OpenDevin 发布公告**：开源自主 AI 工程师 **[OpenDevin](https://lu.ma/fp0xr460)** 的发布在 GitHub 上引起关注。
  - `@cognition_ai` 分享了该[发布](https://lu.ma/fp0xr460)，强调了其在开发者协作和创新方面的潜力。

## GPT4O-Aug (gpt-4o-2024-08-06)


**1. AI 模型性能与优化**

- **FLUX1.1 Pro 超出预期**：**[FLUX1.1 Pro](https://blackforestlabs.ai/announcing-flux-1-1-pro-and-the-bfl-api)** 发布，拥有快六倍的生成速度和更高的图像质量，在 [Artificial Analysis 图像竞技场](https://artificialanalysis.ai/text-to-image/arena)中获得了最高的 **Elo score**。
  - AI 社区兴奋不已，渴望探索该模型在优化 AI 工作流和应用方面的潜力。
- **大型模型的量化技术**：围绕大型神经网络（50B+ 参数）**量化算法**的讨论强调了 **int8** 和 **HQQ** 等技术，可将目标指标的损失维持在 **1% 以下**。
  - 成员们指出 **int4 + hqq 量化** 同样有效，且仅需极少的校准，引发了对优化模型效率的兴趣。
- **AI 配置的 GPU 散热方案**：一位用户考虑为其 **8 GPU** 配置使用**单槽水冷头**，并指出在**两个 1600W** 和**一个 1500W** 电源上的最大功耗为 **4000W**。
  - 讨论强调了用电安全的重要性以及 GPU 配置的创新用途，例如在寒冷月份作为取暖方案。


**2. AI 社区实践与担忧**

- **OpenAI 泡沫担忧**：成员们担心 **OpenAI 泡沫** 正在危险地扩张，将其与 **WeWork** 类比，并质疑 AI 炒作的长期可持续性。
  - **o1** 的发布暂时缓解了恐惧，但讨论强调了 OpenAI 未来轨迹及其对行业影响的不确定性。
- **社区对客户支持的挫败感**：用户对订阅问题的客户支持表示不满，包括文件下载和响应延迟，这影响了用户留存。
  - 一位用户考虑取消订阅，强调了支持不足对社区满意度的重大影响。
- **解决 AI 模型审核问题**：**Claude 2.1** 标记 **SFW 提示词** 的问题引起了关注，其中一个案例将角色描述标记为“色情”，引发了关于审核实践的辩论。
  - 社区讨论强调需要更清晰的审核指南，以防止干扰用户交互。


**3. AI 工具与功能发布**

- **OpenAI's New Canvas Feature**：OpenAI 为写作和编码项目推出了 **canvas** 功能，允许 **Plus & Team 用户** 通过选择 [“GPT-4o with canvas”](https://openai.com/index/introducing-canvas/) 来超越简单的聊天交互进行协作。
  - 该功能旨在提升项目管理和协作的用户体验，并就其增强复杂任务工作流的潜力展开了讨论。
- **GPT-4o Realtime API for Audio**：**GPT-4o Realtime API** 发布，用于低延迟音频交互，目标应用场景包括 **customer support**，并需要客户端集成以实现最终用户音频。
  - 这一进展激发了人们对通过实时音频功能增强各种应用对话能力的兴趣。
- **LangChain's LangGraph Innovates Query Generation**：一篇 [LinkedIn 帖子](https://www.linkedin.com/posts/ismile-bharmal-3b82241ab_langgraph-langchain-querygeneration-activity-7247467636719013888-CZHj) 强调了 **LangGraph** 在 **LangChain** 生态系统中管理复杂查询生成和输出结构化方面的作用。
  - 重点放在了**错误纠正**和**用户友好型结果**上，并对 Harrison Chase 和 LangChain 团队的贡献表示赞赏。


**4. AI Research and Collaboration**

- **AI Reading Group Promotes Collaboration**：来自 Women in AI & Robotics 的 **AI Reading Group** 启动，将于 **2024 年 10 月 17 日** 展示 **IBM** 和 **NASA** 的研究，观众问答环节名额有限。
  - 该小组旨在促进研究人员与社区之间的直接对话，聚焦跨学科 AI 讨论和创新。
- **Exploring Liability in AI Research**：讨论集中在分享 AI 模型用于研究的个人是否应对滥用承担责任，并呼吁制定明确的法律准则。
  - 成员们强调了理清法律环境的重要性，以建立负责任的 AI 研究实践并保护原始研究人员。
- **Knowledge Graph Embedding Innovations**：一篇论文介绍了一种 **knowledge graph embedding (KGE)** 的新方法，通过群论集成不确定性，以实现高效且具有表现力的模型。
  - 这种方法允许实体和关系作为对称群中的置换进行嵌入，展示了改进 KGE 框架的潜力。


**5. AI Ethics and Data Privacy**

- **Concerns Over Data Privacy in AI**：一位成员对 **data privacy** 提出了警告，指责包括 OpenAI 在内的 AI 公司专注于从**中型公司**“窃取数据”，引发了辩论。
  - 讨论强调了数据共享透明度和选择退出（opt-out）选项的重要性，反映了 AI 社区更广泛的担忧。
- **AI's Impact on Future Movies**：一篇文章探讨了 **AI** 对电影制作的影响，认为技术将重塑叙事和制作流程，可在此处访问 [here](https://www.perplexity.ai/page/ai-s-impact-on-future-movies-v.cRWJeZRZWW.O1QghbU.A)。
  - 对话指出了可能重新定义电影观众参与度的新兴趋势，AI 在行业转型中发挥着关键作用。
- **Legal Status of Web Scraping**：针对 web scraping 正在进行的诉讼引起了关注，艺术家和作家对其法律地位和影响感到沮丧。
  - 对话强调了法律的复杂性，以及在数据访问和知识产权保护之间取得平衡的明确准则需求。

## O1-mini

**Theme 1. AI Models on the Fast Track: Speed and Savings**

- **FLUX1.1 Pro Zooms Ahead**：[**FLUX1.1 Pro**](https://replicate.com/black-forest-labs/flux-1.1-pro) 发布，具有 **6 倍快的生成速度**和卓越的图像质量，在 [Artificial Analysis image arena](https://artificialanalysis.ai/text-to-image/arena) 中获得了最高的 **Elo score**。
- **GPT-4o Slashes Prices**：从今天起，**GPT-4o** 的输入成本下降 **50%**，输出成本下降 **33%**，与 8 月份发布的更新模型 **GPT-4o-2024-08-06** 保持一致。
- **NVIDIA's NVLM 1.0 Unveiled**：[**NVLM 1.0**](https://research.nvidia.com/labs/adlr/NVLM-1/) 推出了用于视觉语言任务的开源权重，使 NVIDIA 成为对抗专有模型的主要竞争对手。

**Theme 2. Seamless Integration: Bringing AI to Your Projects**

- **gpt4free 加入聊天机器人**：一位成员成功将 **[gpt4free](https://github.com/yjg30737/pyqt-openai/releases/tag/v1.3.0)** 集成到他们的聊天机器人中，尽管性能较慢且需要频繁切换提供商，但增强了灵活性。
- **针对 AMD GPU 挑战的云解决方案**：面对在没有 CUDA 的 Windows 上进行 **fine-tuning** 的问题，成员们推荐使用 **[Lambda Labs](https://www.lambda.com/)** 或 **Collab** 等云平台，以确保在 AMD 硬件上进行有效的训练。
- **Shadeform 的 GPU 市场流**：**[Shadeform](https://www.shadeform.ai/)** 提供了一个集中的计费和管理系统，用于预订按需 GPU，为开发者简化了多云部署。

**主题 3. 攻克技术难题：克服 AI 训练障碍**

- **量化难题获解**：开发者们正在探索 **int8** 和 **HQQ** 等 **quantization algorithms**，以在大型模型（50B+ 参数）中保持 **<1% 的损失**，并利用 [Hugging Face 的指南](https://huggingface.co/docs/transformers/main/en/quantization/hqq) 进行实现。
- **Mojo 的导入之谜**：**[Mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1291118344606449666)** 在 Python 的动态导入方面面临挑战，引发了关于安全风险以及可能委托给 CPython 的讨论。
- **Flash Attention 的内存混淆**：**Flash Attention** 功能表现出不一致的内存行为，尽管其计算复杂度为二次方，但一些用户遇到了线性增长，如 [llama.cpp Pull Request #5021](https://github.com/ggerganov/llama.cpp/pull/5021) 所示。

**主题 4. 搭建桥梁：参与 AI 社区**

- **AI 读书会激发协作**：在多个 Discord 频道启动，来自 **Women in AI & Robotics** 的小组展示了由 **IBM** 和 **NASA** 的 **Aashka Trivedi** 带来的 **[INDUS](https://arxiv.org/abs/2405.10725)** 等演示，促进了直接对话和跨学科讨论。
- **Unsloth 网络研讨会分享见解**：[**Unsloth Webinar**](https://docs.unsloth.ai/get-started/all-our-models) 强调了为了训练速度而转向低精度位（lower precision bits）的趋势以及高质量数据集的集成，引发了更深层次的技术对话。
- **节日 AI 家庭派对**：诸如 **October House Party** 之类的活动鼓励成员展示他们的 **Open Interpreter** 作品，将乐趣与知识共享及社区凝聚结合在一起。

**主题 5. 助力进步：优化 AI 工具和基础设施**

- **Torchtune 0.3.1 助力微调**：最新的 **[Torchtune 0.3.1](https://github.com/pytorch/torchtune/releases/tag/v0.3.1)** 更新包含了所有 **Llama 3.2 Vision models**，为 Macbook 引入了 **MPS beta support**，并为 **Llama3.2** 和 **Qwen2** 等模型提供了新的 **knowledge distillation recipe**。
- **LlamaIndex 增强混合搜索**：将 [**MongoDB Atlas**](https://t.co/VFsaL4XIdb) 与 **LlamaIndex** 集成可实现无缝的 **hybrid search**，结合 **semantic** 和 **full-text search** 以提高结果相关性。
- **Aider 扩展实时 API**：在 **[Aider](https://aider.chat/docs/config/options.html#--show-diffs)** 中推出的 **GPT-4o Realtime API** 为诸如 **customer support** 等应用启用了低延迟音频交互，增强了对话能力。

---

**提到的链接**：
- [FLUX1.1 Pro on Replicate](https://replicate.com/black-forest-labs/flux-1.1-pro)
- [Artificial Analysis Image Arena](https://artificialanalysis.ai/text-to-image/arena)
- [GPT-4o GitHub Release](https://github.com/yjg30737/pyqt-openai/releases/tag/v1.3.0)
- [Lambda Labs](https://www.lambda.com/)
- [Shadeform AI Marketplace](https://www.shadeform.ai/)
- [Hugging Face Quantization Guide](https://huggingface.co/docs/transformers/main/en/quantization/hqq)
- [llama.cpp Pull Request #5021](https://github.com/ggerganov/llama.cpp/pull/5021)
- [INDUS Paper](https://arxiv.org/abs/2405.10725)
- [Unsloth Documentation](https://docs.unsloth.ai/get-started/all-our-models)
- [Aider Configuration Options](https://aider.chat/docs/config/options.html#--show-diffs)
- [Torchtune Documentation](https://pytorch.org/torchtune/stable/)
- [Torchtune GitHub Release](https://github.com/pytorch/torchtune/releases/tag/v0.3.1)
- [LlamaIndex with MongoDB Atlas](https://t.co/VFsaL4XIdb)

## O1-preview

**主题 1：OpenAI 的新功能与战略举措**

- [**OpenAI 推出 Canvas，彻底改变协作方式**](https://openai.com/index/introducing-canvas/)：OpenAI 推出了 **Canvas** 功能，允许用户在简单的聊天之外，针对写作和编程项目与 ChatGPT 进行交互。Plus 和 Team 用户现在可以通过在模型选择器中选择 **“GPT-4o with canvas”** 来进行尝试。
- **GPT-4o 随模型更新大幅降价**：OpenAI 将 **GPT-4o** 的**输入价格降低了 50%**，**输出价格降低了 33%**，这与自 8 月份以来提供的更新版 **GPT-4o-2024-08-06** 模型保持一致。此举让用户更容易获得先进的 AI 能力。
- **Sam Altman 在 OpenAI 寻求 1570 亿美元估值之际加强控制**：报告显示，在 OpenAI 估值飙升至惊人的 **1570 亿美元** 期间，**Sam Altman** 正在加强其在公司的影响力。这种领导权的集中引发了人们对该组织未来发展轨迹的疑问。

**主题 2：AI 模型与工具的创新**

- [**FLUX1.1 Pro 以六倍速度提升遥遥领先**](https://blackforestlabs.ai/announcing-flux-1-1-pro-and-the-bfl-api/)：新发布的 **FLUX1.1 Pro** 提供了 **六倍的生成速度提升**，改进了图像质量，并在 [Artificial Analysis 图像竞技场](https://artificialanalysis.ai/text-to-image/arena) 中保持着最高的 **Elo 分数**。该模型在图像生成领域树立了新的性能标准。
- [**NVIDIA 发布 NVLM 1.0，挑战专有模型**](https://research.nvidia.com/labs/adlr/NVLM-1/)：**NVIDIA** 推出了 **NVLM 1.0**，这是一个专为视觉语言任务设计的开源模型，其准确率可与领先的专有模型相媲美。开发者可以访问其 **权重（weights）和代码**，为新的创新铺平道路。
- [**StackBlitz 发布用于 AI 驱动全栈开发的 Bolt**](http://bolt.new)：StackBlitz 推出的 **Bolt** 允许用户在 AI 的支持下提示、编辑、运行和部署全栈应用程序。它提供了一个免费、全面的开发环境，支持 npm、Vite 和 Next.js。

**主题 3：AI 模型局限性的挑战与担忧**

- **审核疯狂：SFW 提示词被 AI 误标**：用户报告称 **Claude 2.1** 和其他模型错误地将 **SFW（职场安全）提示词** 标记为不当内容，干扰了交互。一段角色描述被错误地标记为“性暗示”，引发了关于过度审核做法的辩论。
- [**Softmax 的弱点：在尖锐决策中的局限性**](https://arxiv.org/abs/2410.01104)：一篇论文揭示了 **softmax 函数** 在输入增加时无法逼近尖锐函数（sharp functions）的局限性，挑战了其在 AI 推理任务中的有效性。作者建议将 **自适应温度（adaptive temperature）** 作为一种潜在的补救措施，引发了进一步的研究。
- **GPU 困境：在入门级硬件上运行大模型**：用户正在努力解决在旧 GPU 上运行 **SDXL** 等大型模型的问题，并为 AMD 用户探索 **ZLUDA** 等替代方案。社区正在讨论平衡性能与硬件限制的策略。

**主题 4：AI 社区参与与学习**

- [**AI 阅读小组架起研究与社区的桥梁**](https://www.eventbrite.ca/e/1024976160287?aff=oddtdtcreator)：来自 Women in AI & Robotics 的 **AI 阅读小组** 启动，由来自 **IBM** 的 **Aashka Trivedi** 于 **2024 年 10 月 17 日** 展示与 **NASA** 的联合研究。会议将深入探讨 [**INDUS：适用于科学应用的高效语言模型**](https://arxiv.org/abs/2405.10725)。
- **DSPy 2.5 获得好评，呼吁更多文档**：用户称赞 **DSPy 2.5** 在 **TypedPredictors** 等方面的改进，但敦促提供更多关于**自定义**和集成 **Pydantic** 的文档。增强的指南可以为用户解锁高级功能。
- **关于数据实践和隐私的激烈辩论**：成员们对数据隐私表示担忧，指责一些 AI 公司专注于从垂直领域的中型公司“窃取数据”。社区正在辩论 AI 开发中数据使用的伦理和合法性。

**主题 5：AI 模型优化的技术讨论**

- **量化探索：平衡体积与精度**：开发者正在探索针对大型模型（50B+ 参数）的 **int8**、**HQQ** 以及 **int4 + HQQ** 等量化算法，旨在使目标指标的**损失低于 1%**。**HQQ** 等技术提供了高效率，且仅需极少的校准。
- **Mojo 与 Python 导入及错误处理的斗争**：**Mojo** 编程语言在处理 Python 的动态导入时遇到困难，使集成和错误管理变得复杂。社区成员正在辩论是否采用 **Zig 风格的错误联合（error unions）** 以及其他策略来提高 Mojo 的鲁棒性。
- **Flash Attention 引发内存占用之谜**：用户质疑 **Flash Attention** 是否在计算复杂度为平方级的情况下导致内存线性增长。混合的体验引发了关于澄清其对内存和性能实际影响的讨论。

---

# 第一部分：Discord 高层级摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **晨间问候引发闲聊**：成员们交换了晨间问候，随后展开了关于时差的轻松讨论，幽默地提到，“如果你不在 EST 或 PST 时区，你就错过了。”
   - 这种轻松的氛围为后续的技术讨论奠定了基调。
- **关于 Jupyter Notebook 与 VS Code 的辩论**：成员们表达了对界面的偏好，其中一位对 **Jupyter Notebook** 表示不满，称其与 **VS Code** 相比显得过时。
   - 另一位成员反驳道，他们更喜欢 **VS Code**，因为它对 notebook 的支持和整体可用性。
- **对 Qwen 模型可靠性的担忧**：讨论引发了对 **Qwen 模型** 可靠性的担忧，用户报告了熟悉配置下出现的意外结果。
   - **Unsloth** 页面上模型的缺失让成员们感到困惑，加剧了讨论。
- **来自 Unsloth Webinar 的见解**：Unsloth 网络研讨会强调的关键点包括在训练期间转向更低精度的 bits，旨在提高速度。
   - 成员们讨论了高质量数据集的整合以及为深度学习增强的模型架构。
- **在 AMD GPU 上进行 Fine-tuning 的挑战**：一位成员询问如何在没有 **CUDA 支持** 的情况下在 Windows 上运行 **Unsloth**，引发了关于 AMD 在 ML 领域局限性的讨论。
   - 建议包括使用 **Lambda Labs** 或 **Collab** 等云解决方案进行有效训练。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **用户遇到模型访问问题**：几位用户报告了访问 Llama 等模型时遇到的问题，在使用 **Hugging Face** 平台时面临超时和限制，影响了热门模型的可用性。
   - 一位用户确认在 GeForce 980Ti 等旧硬件上运行 **Llama-3.2-1B**，表明即使资源有限也仍然足够。
- **gpt4free 成功集成**：一位成员成功将 **gpt4free** 集成到他们的聊天机器人中，尽管经历了性能下降和需要频繁更换提供商的情况。
   - 此次集成还包括添加了两个 OpenAI 模型，展示了其在 [GitHub 上的 Release v1.3.0](https://github.com/yjg30737/pyqt-openai/releases/tag/v1.3.0) 期间的灵活开发。
- **FLUX1.1 Pro 以速度令人印象深刻**：**FLUX1.1 Pro** 发布，提供了 **6 倍更快的生成速度** 和改进的图像质量，在 [Artificial Analysis 图像竞技场](https://artificialanalysis.ai/text-to-image/arena) 中获得了最高的 **Elo 评分**。
   - 该模型的表现引发了 AI 社区的兴奋以及对进一步进步的期待。
- **AI 读书小组启动公告**：来自 Women in AI & Robotics 的 **AI Reading Group** 启动，其首届会议将于 **2024 年 10 月 17 日** 举行，届时将有来自 **IBM** 关于与 **NASA** 联合研究的演讲。
   - 一位成员建议在 Discord 和 Eventbrite 上直播活动，以扩大受众参与度，增强社区对 AI 研究的参与。
- **为初学者推荐 Hugging Face 课程**：成员们推荐将 Hugging Face 课程和 **Open Source AI Cookbook** 作为 NLP 新手的必备资源，强调了将实践经验与基础理论相结合的重要性。
   - 像 'The Illustrated Transformer' 和 **3blueonebrown** 这样的资源被认为对理解 NLP 中的复杂概念很有帮助。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 遥测数据收集引发关注**：用户讨论了 **Aider** 目前缺乏遥测数据收集，限制了对使用指标和趋势的洞察。
   - 对未来遥测的建议包括在确保隐私的前提下监控 **model choices** 和 **tokens**。
- **Cursor vs Aider - 界面之争**：成员们对比了 **Aider** 和 **Cursor**，指出 Cursor 的界面更流畅，但称赞 Aider 在终端使用中的效率。
   - 用户对 **Cursor** 的 Composer 功能的不一致性表示不满，这与 Aider 的可靠性形成对比。
- **Claude Development 引发关注**：用户对 **Claude Development** 展现出浓厚兴趣，因其具有前景的代码辅助能力。
   - 用户热切期待更新，渴望将其潜在的改进与当前工具进行对比。
- **GPT-4o Realtime API 发布**：**GPT-4o Realtime API** 已发布，旨在为 **customer support** 等应用提供低延迟的音频交互。
   - 集成需要处理终端用户的音频，从而增强对话能力。
- **Crawl4AI 增强数据收集**：**Crawl4AI** 现已作为开源的 LLM 友好型网络爬虫推出，为开发者提供可定制的数据收集工具。
   - 它与语言模型的集成可以显著改善运营数据的提取流程。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **DeepInfra 停机时长受到影响**：DeepInfra 经历了约 **15 分钟** 的停机，但目前正在恢复中。
   - 用户在不久后收到了关于状态和持续恢复工作的通知。
- **GPT-4o 价格腰斩**：从今天起，**GPT-4o** 模型的输入价格下降 **50%**，输出价格下降 **33%**。
   - 这一调整与 8 月份发布的更新模型 **GPT-4o-2024-08-06** 保持一致。
- **Claude 2.1 的审核困惑**：用户对 **Claude 2.1** 标记 **SFW prompts** 表示担忧，这干扰了交互。
   - 其中一个被标记的案例涉及角色描述被错误地标记为“色情”，引发了关于审核机制的辩论。
- **NVIDIA 发布 NVLM 1.0 模型**：**NVIDIA** 发布了具有竞争力的 **NVLM 1.0**，提供专为视觉语言任务设计的开源权重和代码。
   - 该模型预计将提高性能和准确度，与该领域的专有模型展开竞争。
- **Flash 8B 模型在生产环境中变慢**：**Flash 8B model** 现已投入生产，但记录的速度为 **200 tokens per second**，比之前的版本慢。
   - 讨论表明，未来可能会考虑进行速度升级，以解决硬件效率问题。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 在 Python 导入方面遇到困难**：讨论显示 **Mojo** 无法原生处理 Python 的动态导入行为，使集成和错误管理变得复杂。
   - 成员们指出，将导入委托给 CPython 可能会引入类似于 **NPM** 生态系统中的安全风险。
- **Mojo 函数遇到返回值问题**：成员们发现，在 **Mojo** 中从函数返回值有时需要变量声明（例如使用 `var`）以避免运行时错误。
   - 分享的一个示例显示，除非修改为返回可变对象，否则 `SIMD` 初始化会失败。
- **探索错误处理策略**：对话集中在 **Mojo** 错误处理的潜在改进上，建议倾向于使用 **Zig-style error unions** 来处理推断的错误类型。
   - 一些成员主张采用更具函数式编程风格的错误管理方法，强调模式匹配和组合性。
- **静态数据存储的复杂性**：用户寻求在 **Mojo** 中静态存储表的方法，以避免产生过多的代码膨胀，特别是来自 `List` 等构造。
   - 重点是匹配 **C static declarations** 中的性能和内存效率。
- **SIMD 初始化问题引发 GitHub 讨论**：有人请求针对 `SIMD.__init__` 构造函数的异常行为创建一个 GitHub issue，该构造函数在某些条件下会报错。
   - 成员们表示愿意帮助追踪 **SIMD** 相关 bug 的根本原因。

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **用于写作与编码的 Canvas 功能发布**：OpenAI 宣布了 **canvas** 功能的早期版本，允许用户在简单的聊天交互之外处理写作和编码项目。从今天开始，Plus 和 Team 用户可以通过在模型选择器中选择 [“GPT-4o with canvas”](https://openai.com/index/introducing-canvas/) 来试用。
   - 这一增强旨在提升项目管理和协作的用户体验，利用先进的 AI 能力处理复杂任务。
- **API Access 层级困惑**：关于 **API access** 逐步向特定使用层级（usage tiers）开放的讨论出现，一位用户在之前拥有访问权限的情况下遇到了 **403 error**。对话强调了处理 **rate limit issues** 和有效处理错误的重要性。
   - 成员们分享了在 [OpenAI Cookbook](https://cookbook.openai.com/examples/how_to_handle_rate_limits) 中找到的缓解速率限制的见解，强调了社区在应对这些 API 挑战方面的支持。
- **新版 Copilot App 的印象**：用户对新版 **Copilot App** 的性能给出了正面反馈，指出其作为 **Android 原生应用** 的流畅易用性。然而，对于无法删除聊天的担忧也随之产生，这成为了与其他聊天机器人的对比点。
   - 社区讨论集中在用户体验、功能对比上，并提出了改进空间。
- **语音功能现已在自定义 GPTs 中可用**：一位成员庆祝了今天在 **GPT store** 的自定义 GPT 中引入语音功能，感谢 OpenAI 解决了之前的顾虑。他们指出，该语音模式并非新的 **advanced voice**，用户希望未来所有自定义 GPT 都能包含该功能。
   - 这一增强反映了对 GPT 中更丰富交互功能的持续需求，表明了社区对持续改进的渴望。
- **4o-mini 中九尾（Ninetails）训练数据的缺陷**：一位用户发现，当被问及火属性宝可梦时，**4o-mini** 始终错误地将 **Ninetails** 识别为有 6 条尾巴，而 **4o** 则提供了正确答案。这种在多次生成中出现的模式表明这是 **training data** 的缺陷，而非典型的幻觉。
   - 进一步调查显示，像 **gpt-3.5-turbo** 和 **gpt-4o-mini** 这样的小型模型也会显示不准确的回答，引发了对其训练数据集的质疑。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **虚拟环境是兼容性的关键**：成员们建议使用 venv 或 conda 等 **virtual environments**，以避免在运行 **AUTOMATIC1111** 等工具时出现 Python 版本冲突。
   - *Virtual environments* 简化了包管理，确保不同的设置不会干扰工作流。
- **选择正确的 AI 模型和 UI**：鼓励新用户使用 **Comfy UI**，因为它具有灵活性，同时也提到了 **Automatic1111** 和更快的 **Forge UI** 分支。
   - Comfy UI 的基于节点的设计提供了更多通用性，而 Automatic1111 在教程方面仍然很受欢迎。
- **生成特定姿势的图像**：用户解决了生成特定姿势图像的挑战，并建议使用 **ControlNet** 来增强输出控制。
   - 训练像 **LoRA** 这样的特定模型有助于调整生成的图像以满足用户预期。
- **应对 AI 模型的局限性**：讨论强调了在旧款 GPU 上运行 **SDXL** 的问题，并为 AMD 用户推荐了 **ZLUDA** 等替代方案。
   - 虽然较低的分辨率可以加快处理速度，但最佳效果通常需要适合特定模型的高分辨率。
- **尝试 AI 模型训练**：一位用户分享了以复杂情况告终的训练经历，强调了不当图像选择的后果。
   - 这提醒了在训练 AI 模型时遵守社区标准的重要性。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **FLUX1.1 Pro 性能超越竞争对手**：**FLUX1.1 Pro** 正式发布，宣称其生成速度比前代快 **6 倍**，且图像质量有所提升，标志着一次重大升级。
   - 用户可以利用这一性能实现更高效的工作流，正如 [发布公告](https://blackforestlabs.ai/announcing-flux-1-1-pro-and-the-bfl-api/) 中所强调的。
- **Grok 使用需要验证**：讨论中提到了访问 **Grok** 需要 **验证** 和 **付费** 的必要性，成员们对此意见不一。
   - 澄清了一些用户无需验证即可访问该服务，但仍需付费。
- **探讨 Softmax 函数的局限性**：一篇论文强调了 **softmax 函数** 在输入规模增加时实现鲁棒计算的局限性，并从理论上证明了其缺陷。
   - 作者提出 **adaptive temperature**（自适应温度）作为这些局限性的潜在解决方案。
- **寻找无审查的故事创作 LLM**：一位用户询问哪种 **LLM** 最适合创作故事，要求既无审查又可以作为 API 运行。
   - 他们还在寻找能够自动使用 LLM 构建故事，而不仅仅是提供标准帮助的网站。
- **揭示控制模型思考过程**：针对防止模型泄露其 **chain of thought**（思维链）的控制措施引发了担忧，人们质疑这些措施对自我解释能力的影响。
   - 这指向了关于在 AI 交互中平衡透明度与安全性的持续讨论。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **语音朗读功能评价褒贬不一**：用户讨论了语音朗读功能的潜力，认为它对消化长回复很有帮助，但也面临 **发音** 问题。
   - 一位成员表示他们在多任务处理时经常使用此功能，展示了其公认的价值。
- **对订阅客户支持感到沮丧**：几位用户对订阅问题（如文件下载）以及支持团队对安全问题的延迟回复表示沮丧。
   - 有人甚至考虑取消订阅，凸显了对用户留存的重大影响。
- **模型输出质量不一致**：社区讨论揭示了对模型质量不一致的担忧，特别是在 Collection 或 Pro 套餐下。
   - 成员们注意到极端的性能不稳定性，引发了对产品可靠性的怀疑。
- **探讨 AI 对未来电影的影响**：一篇文章详细介绍了 **AI 对电影制作的影响**，认为技术将重塑叙事和制作流程，详见 [此处](https://www.perplexity.ai/page/ai-s-impact-on-future-movies-v.cRWJeZRZWW.O1QghbU.A)。
   - 对话指出了一些可能重新定义电影观众参与度的新兴趋势。
- **OpenAI 获得巨额融资**：报告显示 **OpenAI** 成功筹集了 **66 亿美元**，预计将支持其 AI 研究项目；详情见 [此处](https://www.perplexity.ai/page/openai-raises-6-6b-ofVMnsDdRw.cUWz28MxjBA)。
   - 这笔资金预计将显著提升其技术和平台能力。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **OpenAI 泡沫处于破裂边缘**：成员们对 **OpenAI 泡沫** 正在危险地扩张表示担忧，特别是在 **o1** 发布后，虽然它暂时缓解了恐惧。
   - *这感觉就像 WeWork 事件重演，* 讨论强调了 OpenAI 长期命运的不确定性。
- **Cohere 的低调策略**：对 **Cohere** 策略的赞赏浮出水面，评论认为它在全面运作的同时，在 AI 领域保持了务实的存在感。
   - 这种谨慎的做法可能会在充满寻求曝光度的玩家的环境中为 **Cohere** 提供竞争优势。
- **AGI 概念的转变**：一种观点认为 **AGI 概念** 将在未来 **二十年** 内发生剧变，引发了成员间的激烈讨论。
   - 这种转变可能会重新定义 AI 生态系统的预期和范围，令社区成员感到惊讶。
- **对数据隐私的担忧**：一名成员对 **数据隐私** 发出了警报，声称包括 OpenAI 在内的一些 AI 公司正专注于从 **中型公司** *窃取数据*。
   - 社区对这一说法的有效性进行了辩论，指出公司可以选择 **退出 (opt-out)** 数据共享实践。
- **Reranking API 达到速率限制**：用户报告在仅使用 **50 条记录** 进行极少量 API 调用时就遇到了 **rate limit**，这引发了不满。
   - 这个问题引发了对 **免费层级** 限制的质疑，可能会阻碍有效的测试。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI 发布 Canvas 以增强协作**：OpenAI 新的 **Canvas** 界面允许用户更直观地与 ChatGPT 进行写作和编码项目的互动，整体提升了协作体验。
   - 尽管有其优点，但早期用户也指出了诸如缺乏渲染的前端代码以及难以追踪代码演变等局限性。
- **Sam Altman 在 OpenAI 的权威日益增长**：一篇文章揭示了 **Sam Altman** 如何在 OpenAI 估值飙升至 **1570 亿美元** 之际，扩大了他的影响力。
   - 这一时刻引发了关于权力集中对组织未来发展轨迹影响的关键问题。
- **c.ai 面临潜在公关危机**：关于 c.ai 即将面临 **公关灾难 (PR disaster)** 的警告浮出水面，成员们对该公司的声誉表示担忧。
   - 社区对现状感到失望，情绪中流露出悲伤和无奈。
- **探索 Shadeform 的 GPU 市场**：成员们讨论了 **Shadeform**，它提供了一个预订按需 GPU 的市场，增强了多云部署能力。
   - 集中计费和管理功能似乎简化了工作负载部署，突显了 Shadeform 的效率。
- **O1 Preview 思维过程泄露**：Reddit 上的一篇帖子透露 **O1 Preview** 意外泄露了其完整的思维过程，在聊天中引起了极大关注。
   - 一名成员幽默地建议这可能会激发一篇引人入胜的博客文章，展示了科技社区内意想不到的透明度。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI 发布 ChatGPT Canvas**：OpenAI 推出了 **ChatGPT Canvas**，这是 ChatGPT 内部用于协作项目的新界面，允许用户编辑代码并接收行内反馈。
   - 功能包括直接编辑能力、任务快捷方式和改进的研究功能。
- **StackBlitz 推出 Bolt 平台**：StackBlitz 发布了 [Bolt](http://bolt.new)，这是一个用于通过 AI 支持进行提示、编辑、运行和部署全栈应用的平台。
   - 该开发环境完全支持 npm, Vite 和 Next.js，为应用创建提供了一套免费工具。
- **Gartner 认可 AI 工程**：Gartner 已将 Writer 评为生成式 AI 技术的后起之秀 (Emerging Leader)，强调了 AI 在企业解决方案中的重要性。
   - 这一认可突出了在生成式 AI 工程和 AI 知识管理应用等领域的进展。
- **Google 的 Gemini AI 与 OpenAI 竞争**：Google 正在开发名为 **Gemini AI** 的推理 AI 模型，旨在与 OpenAI 的能力展开竞争。
   - 该计划建立在 Google 先进 AI 系统（如 AlphaGo）的遗产之上，目标是增强类人推理能力。
- **关于 Reflection 70B 模型的讨论**：Sahil Chaudhary 讨论了 **Reflection 70B 模型** 面临的挑战，特别是围绕基准测试的可复现性和输出质量。
   - 社区成员对评估的不一致性以及该模型对 AI 的整体影响表示担忧。

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **对 LM Studio 设置的困惑**：关于将 **LM Studio** 与 **Langflow** 连接的讨论揭示了用户对 OpenAI 组件基础 URL (base URL) 清晰度的不满。
   - 有人对消息查询的语法正确性表示担忧，这表明需要改进文档。
- **LM Studio 更新带来的输出改进**：将 **LM Studio** 从版本 **0.2.31** 更新到 **0.3.3** 后，尽管设置未变，模型输出仍有显著增强。
   - 这引发了关于键值缓存 (key-value caching) 在影响输出质量方面作用的询问。
- **管理上下文的局限性**：用户讨论了在 **LM Studio** 固有的无状态架构中跨会话维持上下文的挑战。
   - 参与者强调了在不重复的情况下提供持久输入的困难。
- **Flash Attention 引发争议**：**Flash Attention** 功能被广泛讨论，用户对其在 GTX 等特定 GPU 型号上不可用感到沮丧。
   - 共享了一个 [GitHub pull request](https://github.com/ggerganov/llama.cpp/pull/5021)，展示了它可以提供的显著加速。
- **优化 GPU 性能的水冷方案**：一位成员正在考虑为 **8 卡** 配置使用**单槽水冷头**，因为其最大功率达到了 **4000W**。
   - 目前的计划涉及**两个 1600W** 和**一个 1500W** 电源，以维持理想的热条件。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **大模型的量化算法**：成员们讨论了适用于大型神经网络（50B+ 参数）且目标指标损失**小于 1%** 的**量化算法**，重点介绍了 **int8** 和 **HQQ** 等技术。
   - *一位成员指出*，int4 + hqq 量化也非常有效，因为它只需要极少的校准。
- **探索 BF16 权重对准确性的影响**：一位成员担心在使用 **4090 VRAM** 时，使用 **BF16** 权重而非 **FP32** 进行训练可能会牺牲**准确性**。
   - 他们认为在当前配置中，将权重使用 **FP32** 而优化器 (optimizer) 保持为 **BF16** 是可行的。
- **理解 Metal 编程基础**：一位新手了解到，虽然 **CUDA** 使用 `block_size * grid_size` 进行线程调度，但 **Metal** 仅涉及 grid size，从而使线程管理更简单。
   - *他们强调 Metal 中的 threadgroups 是为 grid 间的共享内存设计的。*
- **更长的项目周期很有帮助**：一位成员表示，为项目提供**更长的时间**有助于推进，特别是考虑到通常需要时间来积累势头。
   - 他们强调了保留充足时间以完成项目的重要性。
- **关于自压缩神经网络实现的咨询**：一位成员就 [GitHub 上的 Issue #658](https://github.com/pytorch/ao/issues/658) 进行了咨询，涉及**自压缩神经网络**，重点是动态量化感知训练。
   - 他们的目标是将其作为训练期间的一个选项，让用户选择特定的 **VRAM 预算**。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **探讨 AI 研究中的法律责任**：讨论集中在分享 AI 模型用于研究的个人是否应对他人的滥用负责，一些人认为法律责任可能不会追溯到原始研究者。
   - 成员们指出，可能需要一个明确的裁决来建立准则，并强调了在这些法律领域中明确性的必要性。
- **网页爬取诉讼引发担忧**：针对网页爬取法律地位的持续诉讼引发了担忧，艺术家和作家对这种做法表示不满。
   - 引用了一个案例，其中公司试图禁止爬取（除非满足严格条件）但未获成功，凸显了法律的复杂性。
- **OpenAI 审核政策的影响**：一位成员讲述了他们在 OpenAI 审核政策方面的经历，该政策标记了他们提示 AGI 的请求，因感知到的违规行为导致了不安时刻。
   - 其他人一致认为这些政策似乎过于谨慎，并指出许多被标记的消息并不符合所述的使用政策。
- **创意 AI 项目的机会**：一位新成员介绍自己是寻求 AI 领域基于公地（commons-based）方法协作项目的研究员，强调了潜在的跨学科研究。
   - 这演变成了一项参与号召，特别是针对数字人文领域的贡献。
- **分享 MMLU 评分资源**：一位成员询问如何获取新模型的 MMLU 分数，随后有人推荐了 EleutherAI 的 [evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness)。
   - 他们还提到了一个专门用于进一步讨论该话题的频道，以促进协作学习。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 2.5 反馈涌现**：用户报告对 **DSPy 2.5** 的整体体验感到满意，注意到了 **TypedPredictors** 的积极变化，但呼吁提供更多 **自定义文档**。
   - 反馈强调，虽然更新很有前景，但更多的指导可以增强高级功能的可操作性。
- **文档改进需求**：社区呼吁改进 DSPy 文档，特别是关于 **Pydantic** 和多个 LM 集成的部分。
   - 成员们强调了 **用户友好指南** 对于处理复杂生成任务的重要性，这有助于有效地引导新用户。
- **AI Arxiv 播客介绍**：新的 **AI Arxiv 播客** 重点介绍了大厂如何实现 LLM，旨在为该领域的从业者提供有价值的见解。
   - 听众被引导至一期关于 **使用 Vision Language Models 进行文档检索** 的节目，未来还计划将内容上传至 **YouTube** 以提高可访问性。
- **必备 LLM 资源建议**：在寻找资源时，一位成员征求了 **AI/LLM 相关新闻** 的建议，指向了 Twitter 和相关的 subreddit 等平台。
   - 回复中包含了一个精选的 **Twitter 列表**，专注于 LLM 领域的关键讨论和更新，增强了知识共享。
- **优化 DSPy Prompt 流水线**：讨论围绕 DSPy Prompt 流水线的 **自我改进** 方面与传统 LLM 训练方法的对比展开。
   - 推荐了关于多阶段语言模型程序 **优化策略** 的论文，深入探讨了微调和 Prompt 策略的优势。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 0.3.1 发布，包含关键增强功能**：**Torchtune 0.3.1** 更新包含了所有 **Llama 3.2 Vision 模型**，增强了对微调、生成和评估的多模态支持。
   - 关键改进包括在 **8 x A100 上使用 QLoRA 微调 Llama 3.1 405B**，显著优化了性能选项。
- **Tokenizer 自动截断导致数据丢失**：**文本补全数据集**在 **max_seq_len** 处会经历自动截断，导致较大文档的 Token 丢失，引发了增加**用户控制权**的请求。
   - 有提议建议将 **packing max_seq_len** 与 Tokenizer 限制分离，以减少不必要的截断。
- **知识蒸馏 (Knowledge Distillation) 方案现已可用**：为 **Llama3.2** 和 **Qwen2** 等配置添加了新的**知识蒸馏方案**，增强了用户的工具包选项。
   - 提示成员利用这些功能来提升模型的效率和性能。
- **关于 Flash Attention 内存分配的担忧**：讨论了 **Flash Attention** 是否表现出线性内存增长，与其平方级的计算复杂度形成对比，导致预期内存使用量出现偏差。
   - 参与者注意到内存消耗的体验各异，对其真实行为的评估存在冲突。
- **推动更好的 HF 数据集引用**：提议的映射系统如 **DATASET_TO_SOURCE** 旨在简化对 **HF 数据集名称**的访问，促进更清晰的**模型卡片生成**。
   - 重点仍然是增强 **YAML 格式**中数据集文档的清晰度，体现了简化项目能力的努力。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **MongoDB Atlas 助力混合搜索**：最近的一篇文章介绍了如何创建和配置 [MongoDB Atlas 向量和全文搜索索引](https://t.co/VFsaL4XIdb)，以实现混合搜索，将**语义搜索**与**全文搜索**相结合。
   - 该方法显著增强了搜索结果的**相关性**，解决了常见的搜索低效问题。
- **Box 集成助力更智能的应用**：一份指南介绍了将 [Box 工具](https://t.co/Ge42GVau8v)与 LlamaIndex 集成，以开发 **AI 驱动的内容管理应用**。
   - 这实现了**高级搜索**，优化了直接从 Box 内容中提取和处理信息的过程。
- **RAG 系统设置中的挑战**：用户报告在关于使用 Excel 构建 RAG 系统的教程中遇到了 `ModuleNotFoundError`，暗示存在 pandas 版本冲突。
   - 一位用户建议回退到较旧的 pandas 版本（2.2.2 或更低）以可能修复兼容性问题，该问题已在 [GitHub 示例](https://github.com/run-llama/llama_parse/blob/main/examples/excel/o1_excel_rag.ipynb)中分享。
- **RAG 实现中的异步转换查询**：一位开发者正在研究将 RAG 应用转换为异步模式，并询问 `QueryEngineTool` 的异步兼容性以及 `RouterQueryEngine` 的作用。
   - 回复澄清了如何在 `RouterQueryEngine` 中实现异步方法，提供了向异步处理更平滑的过渡。
- **使用 LlamaIndex 生成 RFP 响应**：一位开发者寻求关于利用 LlamaIndex 使用以往中标方案的数据生成 RFP（建议书请求）响应的指导，重点关注高效的索引策略。
   - 他们对 LlamaIndex 从生成的响应中生成 PDF 或 Word 文档的能力表示感兴趣。

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Jordan Pfost 带来十年的 AI 经验**：Jordan Pfost 介绍自己是一名拥有 **10 年** AI/Web 产品经验的高级全栈工程师，专注于 **GPU Clustering**、**RAG** 和 **Agentic Reasoning**。
   - *寻求合作*：他分享了来自 **spendeffect.ai** 和 **iplan.ai** 等项目的见解。
- **Kapa.ai 令人印象深刻的能力**：**Kapa.ai** 展示了其作为一个基于 Transformer 的模型，拥有约 **3.4 亿参数 (340 million parameters)**，专为自然语言任务设计。
   - 它还提到其在多样化数据上进行了训练，确保生成 **真人质量 (human-like quality)** 的文本，并建议成员参考 **LangChain documentation** 以进行进一步探索。
- **解码 LLM 中的喜好与奖励**：Kapa.ai 澄清说 LLM 是基于训练数据中的模式运行的，并不具备个人偏好或奖励机制。
   - 他们引用了一篇关于 **Preference Optimization** 的论文，并指出在 **LangChain documentation** 中可以获得更多见解。
- **为学生对接 AI 实习机会**：一位成员为寻求 AI 实习机会的印度大学生提供了交流平台，鼓励他们表达意向。
   - 此次讨论旨在为学生与 **潜在的 AI 实习机会** 搭建桥梁。
- **LangGraph 创新查询生成**：一篇 [LinkedIn 帖子](https://www.linkedin.com/posts/ismile-bharmal-3b82241ab_langgraph-langchain-querygeneration-activity-7247467636719013888-CZHj) 强调了 **LangGraph** 如何在 **LangChain** 生态系统中管理复杂的查询生成（Query Generation）。
   - 该帖子专注于 **错误修正 (error correction)** 和 **用户友好型结果**，并对 **Harrison Chase** 和 LangChain 团队的贡献表示了认可。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **十月家庭派对明天举行**：别忘了明天的 **October House Party** —— [点击此处加入](https://discord.gg/f6a7YXng?event=1288234745477726238) 参与趣味活动并获取更新。
   - 一位成员表示，由于之前的健康和工作限制，他们 *这次绝不会错过*。
- **展示你的 Open Interpreter 作品**：主持人邀请成员在派对期间展示他们使用 **Open Interpreter** 创作的作品，并鼓励提问和分享经验。
   - 这引发了关于时间的各种反应，一些成员觉得太早了，而另一些人则兴奋地宣告：*派对时间到 (PARTY TIMEEEE)*。
- **探索模型的技能教学**：成员们讨论了如何有效地向他们的模型传授技能，强调了意图清晰度对于实现成功教学的重要性。
   - 尽管进行了尝试，但未解决的问题促使大家建议在未来寻求额外的支持。
- **关于模型 Vision 能力的困惑**：对话转向技能是否自带 **Vision** 能力，这取决于所使用的具体模型。
   - 一位用户提到将 **gpt4o** 与 **Cartesia** 和 **Deepgram** 结合使用，讨论结论是理论上应该是可行的。
- **OpenAI 请求出现问题**：一位用户报告说 OpenAI 请求在发送几条消息后就会失败，且没有附带任何错误或日志。
   - 这种情况说明了潜在的系统问题，导致大家建议发布新帖进行故障排除。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Logo 变更引发褒贬不一的反应**：成员们对最近的 **Logo 变更** 反应不一，表情符号涵盖了从困惑到沮丧的各种情绪，表明接受程度各异。
   - *一位成员幽默地提到*：“我以为我把服务器从列表里弄丢了 😅。”
- **对融资预期的质疑**：一位成员幽默地期待新 Logo 能与筹集 **1000 万美元** 且估值达到 **10 亿美元** 相关联。
   - *另一位用户回应道*：“Sheeesh，”表示对如此宏大目标的不敢置信。
- **分享 Demo 体验**：一位成员分享了他们的 Demo 使用体验，称：“还不赖，我通过 Demo 用过了，”暗示了积极的互动。
   - 持续的对话表明成员们仍在适应这些变化。
- **微调讨论进行中**：成员们提出了关于模型是否已经进行微调的问题，确认目前尚未进行微调。
   - 一位成员安慰说微调很快就会进行，并强调了准备就绪后部署 **70B 参数模型** 的计划。

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **正则表达式规则显著遏制垃圾信息**：一位成员分享了一个正则表达式模式 `\[[^\]]+\]\(https?:\/\/[^)\s]+\)`，该模式能有效阻止 Markdown 链接混淆，减少垃圾机器人的存在。
   - 针对特定垃圾信息类别的自定义正则表达式和词汇黑名单已显示出能显著减少不必要的机器人活动。
- **60 秒超时策略让垃圾信息远离**：在消息屏蔽后实施 **60 秒超时**策略，能有效促使垃圾机器人在尝试几次后退出。
   - 这一策略通过最大限度地减少对合法用户的干扰，有助于维护用户体验。
- **Google 的 Illuminate：领域内的新 AI 工具**：对 [Google's Illuminate](https://illuminate.google.com/home?pli=1) 工具的关注表明，对于寻求复杂内容的 AI 生成音频摘要的研究人员来说，它可能是一个游戏规则改变者。
   - 成员们热衷于将其功能与 notebooklm 播客工具进行对比，凸显了对这两项创新的浓厚兴趣。
- **Arxflix 将 Arxiv 论文带到 YouTube**：关注 [Arxflix](https://www.youtube.com/@Arxflix)，这是一个致力于将 Arxiv 论文转化为引人入胜的视频内容的自动化 YouTube 频道。
   - 创作者对该项目表达了兴奋之情，认为它为传统的学术工具提供了一个动态的替代方案。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinybox 交付时间表受到关注**：一位用户对美国境内 **tinybox** 的交付时间表表示担忧，特别是询问是否能在 **2-5 天** 内送达。
   - George Hotz 回应称，用户应就物流问题发送邮件至 *support@tinygrad.org*，并强调了提出清晰询问的重要性。
- **FAQ 必须包含支持邮箱**：有人建议将目前缺失的支持邮箱加入 **网站 FAQ**。
   - George 同意立即添加，展示了对社区反馈的关注。
- **交付查询中的地理关注点**：George 对交付地点限制的重要性提出疑问，提到了 **圣地亚哥、密歇根或夏威夷** 等特定地区。
   - 他强调了清晰表述问题的必要性，并引导用户前往 #1068979651336216706 频道寻求帮助。
- **通过点击确认明确用户协议**：George 提出了一个**点击确认协议**的想法，让用户确认已阅读问题文档，可能会利用多选题形式。
   - 另一位成员指出，点击确认机制已经存在，表明已有相关措施供用户确认。
- **社区文化需要改进**：George 对社区的提问方式表示沮丧，认为这是一个反复出现的挑战。
   - 他呼吁转向优先考虑清晰沟通和正确的咨询实践。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **通过 RAG 优化推理耗时**：一位成员询问如何优化部署了 **RAG 架构** 且使用 [Llama Index](https://link.to.llama.index) 的 **SLM-based systems** 的**推理耗时**，寻求社区见解。
   - 该请求凸显了一个持续的挑战；性能优化仍然是专注于效率的开发人员的热门话题。
- **AI 阅读小组启动**：来自 Women in AI & Robotics 的 **AI Reading Group** 启动并讨论 AI 论文，首场由来自 **IBM** 的 **Aashka Trivedi** 讨论他们与 **NASA** 的合作。
   - 观众问答环节的有限名额强调了该小组的互动方式，促进了研究人员与社区之间更紧密的联系。
- **记下日期：INDUS 论文演讲**：加入 **AI Reading Group**，于 **2024 年 10 月 17 日** **东部时间中午 12 点** 参加由 **Aashka Trivedi** 主讲的 [**INDUS: Effective and Efficient Language Models for Scientific Applications**](https://arxiv.org/abs/2405.10725) 演讲。
   - 本次会议承诺提供有关适用于科学任务的语言模型显著进展的见解，重点介绍来自 **IBM** 和 **NASA** 的关键贡献。
- **INDUS 论文凸显合作成果**：由 **IBM Research AI**、**NASA** 等共同撰写的 **INDUS 论文**展示了用于**科学应用**的语言模型的进展。
   - 该倡议旨在增强对当前创新的广泛理解，同时鼓励跨学科的知识共享。

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **AI 读书会启动，专注于研究**：来自 Women in AI & Robotics 的 **AI 读书会**正式启动，为研究人员讨论 AI 论文提供了一个平台，并设有互动的 **Q&A 环节**。
   - *该倡议加强了研究人员与社区之间的直接对话*，重点展示了 AI 领域的最新进展。
- **INDUS 研究演讲已排期**：来自 IBM 的 **Aashka Trivedi** 将于 **2024 年 10 月 17 日**展示“[INDUS: Effective and Efficient Language Models for Scientific Applications](https://arxiv.org/abs/2405.10725)”，重点探讨其在科学背景下的潜力。
   - 贡献作者来自 **IBM Research**、**NASA** 和 **Harvard-Smithsonian CfA**，表明该研究具有极高的专业水平。
- **读书会参与名额有限**：由于**名额有限**，感兴趣的参与者需尽快报名，旨在确保观众能进行有意义的互动。
   - 这一策略有助于在每次演讲后的 **Q&A** 环节中进行更深入的交流。
- **突出跨学科 AI 讨论**：该小组提供了一个关注**当前研究热点**并鼓励跨越传统学科边界讨论的场所。
   - *跨学科的参与确保了对 AI 领域复杂性的深入探讨*。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **修改代码以支持第三方数据集**：**Gorilla LLM** 的当前实现原生不支持**第三方数据集**，但一名成员建议通过修改代码来启用此功能。
   - 调整工作将涉及为解析逻辑添加 **model handler**、更改测试文件映射以及选择合适的 checkers。
- **实现数据集解析逻辑**：为了集成新数据集，一名成员解释了使用 `decode_ast` 和 `decode_exec` 实现解析逻辑的必要性。
   - 这种适配需要对流水线的数据集处理有深入的理解，以确保所有内容的兼容性。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。

---

# PART 2: 按频道划分的详细摘要和链接

{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1291137412394913862)** (128 条消息🔥🔥): 

> - `关于早晨问候的讨论`
> - `Jupyter Notebook vs VS Code`
> - `Qwen 模型性能担忧`
> - `Unsloth 网络研讨会要点`
> - `AMD GPU 微调挑战` 


- **早晨问候引发闲聊**：成员们互相进行早晨问候，随后展开了关于所在地时差的轻松讨论。
   - 一位成员幽默地提到：*'如果你不在 EST 或 PST 时区，那你就亏大了'*。
- **关于 Jupyter Notebook 与 VS Code 的辩论**：一位成员对 Jupyter Notebook 的界面表示不满，觉得像是在使用过时的应用程序。
   - *'甚至更好'* 另一位成员反驳道，表示他们更喜欢 VS Code，因为它支持 Notebook 且易于使用。
- **对 Qwen 模型可靠性的担忧**：成员们讨论了 Qwen 模型的性能，有人指出在熟悉的配置下得到了意想不到的结果。
   - 一位成员表示担心，因为这些模型似乎从 Unsloth 模型页面消失了，引起了困惑。
- **来自 Unsloth 网络研讨会的见解**：Unsloth 网络研讨会的要点强调了训练中位表示（bit representation）的重要性，转向更低精度的位（bits）可以带来速度提升。
   - 讨论的其他优化包括在高质量数据集上进行训练以及模型架构的改进，推动模型向更深层次发展。
- **在 AMD GPU 上进行微调的挑战**：一位新成员询问如何在没有 CUDA 支持的情况下在 Windows 上运行 Unsloth，引发了关于 AMD 在 ML 领域局限性的讨论。
   - 成员们建议改用 Colab 等云解决方案进行训练，并探索适用于 AMD GPU 的 HPIC 和其他替代框架。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/Joseph717171/Llama-3.1-SuperNova-8B-Lite_TIES_with_Base">Joseph717171/Llama-3.1-SuperNova-8B-Lite_TIES_with_Base · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/Qwen2.5-14B-Instruct-bnb-4bit">unsloth/Qwen2.5-14B-Instruct-bnb-4bit · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/nvidia/Mistral-NeMo-Minitron-8B-Instruct">nvidia/Mistral-NeMo-Minitron-8B-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/all-our-models">All Our Models | Unsloth Documentation</a>：查看下方列表，了解所有已上传的 GGUF、16-bit 和 4-bit bnb 模型。
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1291189094495551520)** (8 条消息🔥): 

> - `Z 世代文化`
> - `Sigma 心态` 


- **祈求“神力”适配**：一位成员幽默地为一个项目请求神圣援助，称：*'万能的上帝，请让它适配吧。'*
   - 另一位成员积极回应道：'**看起来不错。**'
- **幽默打破僵局**：在祈求适配的请求后，一位成员大笑回应道：'**哈哈**。'
   - 这个轻松的评论引发了其他成员的赞同，强调了聊天中轻松的基调。
- **点击 Z 世代标签的羞耻感**：一位成员表达了对 Z 世代标签的不适，称：*'点击 Z 世代让我感到羞耻。'*
   - 另一位成员质疑这种情绪，问道：*'为什么？'*
- **追求 Sigma 心态**：一位成员表达了想要体现 Sigma 心态的愿望，说：*'希望我是一个 Sigma。'*
   - 作为回应，另一位成员强调了他们对 Sigma 概念的认同，自信地表示：'**我们就是 Sigma……**'


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1291128368120725564)** (80 messages🔥🔥): 

> - `Dataset Merging for Multiturn Creative Writing`（多轮创意写作的数据集合并）
> - `Fine-tuning Llama 3.1 on Google Colab`（在 Google Colab 上微调 Llama 3.1）
> - `Monitoring GPU Usage During Training`（训练期间监控 GPU 使用情况）
> - `ChatML Inference Issues`（ChatML 推理问题）
> - `Guardrails for Therapy Models`（治疗模型的 Guardrails 护栏） 


- **数据集合并疑虑**：一位用户询问，使用不同的轮次启动符（例如 'from:human' 与 'from:gpt'）开始不同的样本，是否会对使用 Unsloth 训练产生问题。
   - 另一位成员保证，拥有多个数据集列（包括额外的键）不应该有问题。
- **微调 Llama 3.1 的挑战**：一位成员分享了由于 VRAM 限制，在 Google Colab 上微调 **Llama 3.1 70B** 失败的经历，并指出需要 **48GB** 显存。
   - 他们收到了尝试 **Lambda Labs** 的建议，因为 **Google Colab** 无法容纳该模型。
- **训练中分步执行的需求**：用户寻求关于单独运行训练步骤而非一次性运行的建议，以便更好地监控 GPU 使用情况。
   - 他们被引导使用 **Wandb** 或 **TensorBoard** 等工具来监控梯度和优化器日志。
- **将 ChatML 用于推理**：一位用户在使用 **ChatML** 数据集进行推理时面临挑战，因为他们的模型会响应自己的提示词而不是用户的查询。
   - 建议他们可能需要使用正确的 chat template 进行推理，而不是直接使用基于对话的数据集。
- **为治疗模型实施 Guardrails**：一位成员讨论了为他们的治疗导向模型应用 Guardrails 的必要性，以防止其响应不当查询。
   - 建议他们预先对输入进行分类并保护响应，并提到了使用 **llama-guard** 或 **Gemma shield** 等工具。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/settings/tokens">Hugging Face – 建设未来的 AI 社区。</a>：未找到描述</li><li><a href="https://huggingface.co/posts/mlabonne/730068367902681">Hugging Face 上的 @mlabonne："⚡ AutoQuant，AutoQuant 是我之前 AutoGGUF notebook 的进化版……"</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth#finetune-llama-32-mistral-phi-35--gemma-2-5x-faster-with-80-less-memory">GitHub - unslothai/unsloth: 微调 Llama 3.2, Mistral, Phi &amp; Gemma LLMs，速度提升 2-5 倍，显存占用减少 80%</a>：微调 Llama 3.2, Mistral, Phi &amp; Gemma LLMs，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1291226463399579661)** (5 messages): 

> - `Fira paper`（Fira 论文）
> - `Nanoflow framework`（Nanoflow 框架） 


- **关于 LLM 训练约束的 Fira 论文**：一位成员分享了 [Fira](https://github.com/xichen-fy/Fira) 的链接，该论文探讨了在低秩约束下是否可以实现 LLM 的全秩训练。
   - 论文已附在仓库中，但截至目前，**尚无可用代码**。
- **Nanoflow 框架提供高性能推理服务**：提供了另一个指向 [Nanoflow](https://github.com/efeslab/Nanoflow) 的链接，它被描述为一个面向吞吐量的高性能 LLM Serving 框架。
   - 该框架旨在增强专门针对大语言模型的推理服务能力。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/efeslab/Nanoflow">GitHub - efeslab/Nanoflow: 一个面向吞吐量的高性能 LLM Serving 框架</a>：一个面向吞吐量的高性能 LLM Serving 框架 - efeslab/Nanoflow</li><li><a href="https://github.com/xichen-fy/Fira">GitHub - xichen-fy/Fira: Fira: 我们能在低秩约束下实现 LLM 的全秩训练吗？</a>：Fira: 我们能在低秩约束下实现 LLM 的全秩训练吗？ - xichen-fy/Fira
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1291117421121503283)** (182 messages🔥🔥): 

> - `Model Access Issues`（模型访问问题）
> - `Engagement with Hugging Face Platform`（Hugging Face 平台参与度）
> - `AI Model Recommendations`（AI 模型推荐）
> - `Launch Discussions`（发布讨论）
> - `Building AI Applications`（构建 AI 应用程序）

- **用户遇到模型访问问题**：多位用户报告了在访问 Llama 等模型时遇到的问题，部分用户经历了超时或无法使用特定模型的情况，这凸显了在使用 Hugging Face 产品时面临的挑战。
   - 一位用户提到他们在 GeForce 980Ti 上成功运行了 Llama-3.2-1B，这表明利用旧硬件进行深度学习应用是可行的。
- **关于 Hugging Face 平台能力的讨论**：一位用户表示需要更清晰地了解 Hugging Face 的平台能力（类似于 replicate.com 提供的服务），表明了对更用户友好的访问方式的需求。
   - 参与讨论的其他用户分享了学习该平台的链接和资源，同时提倡探索社区创建的项目和学习资源。
- **AI 模型使用建议**：用户讨论了适用于邮件摘要等任务的模型，建议使用在摘要任务中表现出色并根据可用计算资源进行调整的模型。
   - 互动中强调了了解运行不同规模模型所需的 RAM 要求的重要性，并据此利用 Hugging Face 的产品。
- **用户对 AI 工具和发布的兴趣**：用户对新推出的 AI 工具表现出浓厚兴趣，特别是用于高效内容生成和品牌推广的工具，其中一位成员推广了一个用于编写病毒式推文的 AI 项目。
   - 此外，有人询问了 Hugging Chat Android 版的发布情况，但关于该方向持续开发的回复尚不明确。
- **社区项目更新和反馈请求**：几位用户介绍了他们的 AI 驱动项目，邀请社区提供反馈并寻求潜在合作，以增强用户参与度和技术采用。
   - 这些讨论强调了社区内的协作精神，成员们希望分享他们的创新并寻求建设性的意见。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aiartarena.com)">未找到标题</a>: 未找到描述</li><li><a href="https://api-inference.huggingface.co,">未找到标题</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/ArtificialAnalysis/Text-to-Image-Leaderboard">Text To Image Leaderboard - ArtificialAnalysis 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://blackforestlabs.ai/announcing-flux-1-1-pro-and-the-bfl-api/">发布 FLUX1.1 [pro] 和 BFL API</a>: 今天我们发布了 Flux1.1 PRO 和我们的 API，我们迫不及待地想看到用户会用我们最新、最棒的产品创造出什么 <3</li><li><a href="https://huggingface.co/spaces/KingNish/Realtime-FLUX">FLUX Realtime - KingNish 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/cfahlgren1/webllm-playground">WebLLM Playground - cfahlgren1 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/yuntian-deng/o1">Chat-with-OpenAI-o1 - yuntian-deng 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://tenor.com/view/failure-gif-23242816">Failure GIF - Failure - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/learn">Hugging Face - 学习</a>: 未找到描述</li><li><a href="https://www.dzine.ai/tools/flux1/>">Dzine (原 Stylar.ai) - 最具可控性的 AI 图像与设计工具</a>: 未找到描述</li><li><a href="https://tenor.com/view/the-deep-deep-thoughts-deep-thoughts-with-the-deep-the-boys-gif-26372785">The Deep Deep Thoughts GIF - The Deep Deep Thoughts Deep Thoughts With The Deep - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/learn/cookbook/index">Open-Source AI Cookbook - Hugging Face 开源 AI 食谱</a>: 未找到描述</li><li><a href="https://tenor.com/view/hackers-hack-the-planet-taogifs-zero-cool-crash-override-gif-5753306679943930050">Hackers Hack The Planet GIF - Hackers Hack the planet Taogifs - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/sledge-hammer-sledgehammer-david-rasche-trust-me-gif-12965638648418662366">Sledge Hammer Sledgehammer GIF - Sledge hammer Sledgehammer David Rasche - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/monty-python-knights-who-say-ni-ni-gif-12279570">Monty Python GIF - Monty Python Knights Who Say Ni - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/spaces">Spaces - Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/InServiceOfX/InServiceOfX/blob/master/PythonLibraries/HuggingFace/MoreTransformers/executable_scripts/terminal_only_infinite_loop_instruct.py">InServiceOfX/PythonLibraries/HuggingFace/MoreTransformers/executable_scripts/terminal_only_infinite_loop_instruct.py at master · InServiceOfX/InServiceOfX</a>: 用于深度学习的 Monorepo（单一代码库）。 - InServiceOfX/InServiceOfX
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1291130612102594571)** (6 条消息): 

> - `切换到 Kotlin`
> - `Hugging Face API 登录` 


- **Richie 的 Kotlin 飞跃**：一位成员宣布转向 **Kotlin**，并表示他们的大部分工作涉及 **Kotlin channels**，从而促成了这一转变。
   - 他们分享了从 **Kivy**、**Flet** 和 **BeeWare** 到 **Dart** 和 **Flutter** 的历程，现在最终定位于 **Kotlin** 和 **Jetpack Compose**。
- **关于 Hugging Face API 登录的澄清**：一位成员澄清说，要使用 **HfApiEngine** 类，用户必须使用有效的 **HF token** 执行 `huggingface_hub.login(HF_TOKEN)`。
   - 另一位成员误以为 token 要求与**模型选择**有关，但现在意识到它适用于 **HfApiEngine** 的使用。


  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1291327085176033332)** (4 条消息): 

> - `FLUX1.1 Pro`
> - `Pika Labs 发布`
> - `Graph of Thoughts 论文` 


- **FLUX1.1 Pro 在速度和性能方面表现出色**：全新的 [FLUX1.1 Pro](https://replicate.com/black-forest-labs/flux-1.1-pro) 的**生成速度比前代产品快六倍**，同时提升了**图像质量**、提示词遵循度（prompt adherence）和多样性。
   - *它在 [Artificial Analysis 图像竞技场](https://artificialanalysis.ai/text-to-image/arena) 中获得了最高的整体 Elo 评分*，超越了排行榜上的所有其他模型。
- **对最新发布的兴奋**：成员们对本周发布的 **FLUX1.1 Pro** 和 **Pika Labs** 表示兴奋。
   - 社区对这些进展议论纷纷，热烈讨论它们对 AI 能力的影响。
- **讨论 Graph of Thoughts 研究**：分享了题为 [Graph of Thoughts: Solving Elaborate Problems with Large Language Models](https://arxiv.org/pdf/2308.09687) 的论文以供审阅。
   - 这项研究可能会提供有趣的见解，特别是考虑到它对 **Large Language Models** 讨论的影响和背景。



**提到的链接**：<a href="https://replicate.com/black-forest-labs/flux-1.1-pro">black-forest-labs/flux-1.1-pro – 在 Replicate 上通过 API 运行</a>：未找到描述

  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1291216615269994580)** (13 条消息🔥): 

> - `gpt4free 集成`
> - `GIF QA 机器人`
> - `Nvidia/Nemo - Mistral - Minitron 8B`
> - `Llama 3.2 限制`
> - `salamandra-2B 端侧运行` 


- **gpt4free 终于集成了！**：一位成员成功将 **gpt4free** 集成到他们的聊天机器人中，并指出虽然运行速度有点慢，且必须频繁更换提供商，但确实可以工作。
   - *还提到：添加了两个 OpenAI 模型，o1-preview 和 o1-mini*；更多详情请查看 [GitHub 上的 Release v1.3.0](https://github.com/yjg30737/pyqt-openai/releases/tag/v1.3.0)。
- **构建 GIF QA 机器人**：一位成员询问是否有推荐的预训练模型，用于创建一个每个 GIF 对应一个问题的 **GIF QA 机器人** 数据集。
   - 另一位成员建议为此使用 **phi-3.5 vision 模型**。
- **介绍 Nvidia/Nemo - Mistral - Minitron 8B！**：一位成员分享了他们新创建的 **Nvidia/Nemo - Mistral - Minitron 8B** 模型供大家测试，并由于自己的 GPU 配额有限，敦促其他人进行测试。
   - *他们还幽默地表示希望在测试期间监控错误日志。*
- **对 Llama 3.2 限制的担忧**：一位成员对 **Llama 3.2 VL 的限制** 表示沮丧，认为考虑到最近向美国政府披露的情况，这种大公司的行为非常奇怪。
   - 这种情绪反映了人们对 AI 领域持续进行的监管讨论所产生影响的广泛担忧。
- **对端侧运行 Salamandra-2B 感到兴奋**：一位成员分享了他们对 **salamandra-2B 端侧运行** 的热情，强调了其指令遵循（instruct）特性和围绕它的积极氛围。
   - 他们表达了对社区反馈的渴望，并可能就其开发过程进行演示。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/Tonic/Nemo-Mistral-Minitron">Nemotron-Mini - Tonic 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/Tonic/salamandra-on-device">Salamandra On-Device - Tonic 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://github.com/yjg30737/pyqt-openai/releases/tag/v1.3.0">Release v1.3.0 · yjg30737/pyqt-openai</a>：VividNode(pyqt-openai) 1.3.0 功能更新：支持 GPT4Free，允许 g4f 用户选择提供商，显示每个提供商中的模型，在 g4f 和使用 API 标签页中添加手册，添加 o1-preview 和 o1-mini...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1291396683850055750)** (2 messages): 

> - `AI 阅读小组启动`
> - `关于举办会议的讨论`
> - `关于 INDUS 的研究演示`
> - `跨学科参与` 


- **AI 阅读小组启动公告**：来自 Women in AI & Robotics 的 **AI Reading Group** 已启动，为研究人员提供了一个分享工作的平台。
   - 第一场会议邀请了来自 **IBM** 的演讲者，展示与 **NASA** 的合作研究，定于 **2024 年 10 月 17 日**举行。
- **对双流直播会议的兴趣**：一位成员表示有兴趣在 Discord 上举办会议，以增加 **AI Reading Group** 活动的曝光度。
   - 他们建议同时在 Discord 和 Eventbrite 上进行直播，以吸引更多观众。
- **科学语言模型演示**：阅读小组的首场活动将重点介绍研究论文 [INDUS: Effective and Efficient Language Models for Scientific Applications](https://arxiv.org/abs/2405.10725)。
   - 该论文由来自 **IBM**、**NASA** 和多个学术机构的研究人员共同撰写，体现了强大的跨学科协作。
- **参与 AI 研究话题**：阅读小组旨在为研究人员和社区之间就当前 **AI** 研究课题进行直接对话创造空间。
   - 目标是为讨论提供一个引人入胜的环境，揭开创新的神秘面纱，并促进对新兴研究的深入参与。



**提到的链接**：<a href="https://www.eventbrite.ca/e/1024976160287?aff=oddtdtcreator">INDUS: Effective and Efficient Language Models</a>：AI 阅读小组会议，与 "INDUS: Effective and Efficient Language Models for Scientific Applications" 的作者之一进行交流。

  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/)** (1 messages): 

ohmahgawdronnie: 好的，我想我明白意思了，谢谢！
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1291302296612896819)** (2 messages): 

> - `NLP 入门`
> - `Hugging Face 课程`
> - `The Illustrated Transformer`
> - `使用 BERT 的实际实现` 


- **开启你的 NLP 之旅**：一位新成员在完成 Python 课程后表达了学习 NLP 的兴趣，并寻求入门建议。
   - 他们特别提到了自己作为 Dialogflow CX 开发者的背景。
- **Hugging Face 作为资源**：一位成员推荐了 Hugging Face 课程，并指出其平台上提供的 **cookbook** 是一个极好的资源。
   - 他们强调了在深入研究理论之前进行实践经验的重要性。
- **必备的 NLP 理论资源**：推荐资源包括 "The Illustrated Transformer"、YouTube 频道 **3blueonebrown** 以及原始论文 **'Attention is All You Need'**。
   - 此外，一位成员表示如果需要，可以分享一篇面向初学者的文章。
- **BERT 的实践经验**：对于动手学习，建议将尝试 finetune 一个 **BERT** 风格的模型进行文本分类作为启动项目。
   - 鼓励采用这种方法在进入理论概念之前建立基础技能。

### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1291146725083316226)** (8 messages🔥): 

> - `FLUX.1-dev card structure` (FLUX.1-dev 卡片结构)
> - `Transformer model formats` (Transformer 模型格式)
> - `Discussion on adding Transformers section` (关于添加 Transformers 板块的讨论)
> - `NLP community engagement` (NLP 社区参与度) 


- **理解 FLUX.1-dev 卡片文件结构**：一位用户询问了 `flux1-dev.safetensors` 文件与其他 pipeline 文件夹的关系，质疑它是否以单体形式存储了所有模型。
   - 另一位成员澄清说，该文件仅包含 transformer 模型，并提到需要 *autoencoder* 和 *T5* 才能实现完整功能。
- **呼吁建立 Transformers 讨论板块**：有人建议为 transformers 创建一个独立的讨论页面，类似于现有的 diffusers 频道。
   - 社区注意到缺乏针对 LLMs 及相关话题的合适频道，成员认为这对于提高参与度是必要的。
- **Transformer 格式混淆**：用户对 `flux1-dev.safetensors` 和 `diffusion_pytorch_model` 文件之间的层名称差异表示困惑。
   - 据指出，该问题出现的原因是根仓库（root repo）包含原始的 BFL 格式，而 transformer 目录使用的是 diffusers 格式，从而导致了名称对齐问题。
- **原始 BFL 格式的可访问性**：一位用户询问了原始 BFL 格式模型的可用性，并指出难以找到相关文档或直接访问途径。
   - 这一询问突显了社区内部对更清晰地获取基础资源的需求。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1291116157465657344)** (156 messages🔥🔥): 

> - `Aider telemetry data` (Aider 遥测数据)
> - `Cursor vs Aider` (Cursor 与 Aider 对比)
> - `Claude Development` (Claude 开发)
> - `Model performance and features` (模型性能与特性)
> - `Real-time audio API` (实时音频 API)


- **关于 Aider 遥测数据收集的讨论**：有人指出 Aider 目前不收集遥测数据，一些用户认为这阻碍了对使用模式和成功率的了解。
   - 对未来遥测的建议包括以隐私敏感的方式跟踪模型选择、tokens 和用户提示词（prompts），而不捕获可识别信息。
- **Cursor 与 Aider 的对比**：用户分享了使用 Cursor 和 Aider 的经验，表示 Cursor 拥有更流畅的界面，而 Aider 仍然是一个强大的命令行工具。
   - 几位用户对 Cursor 的不一致性表示不满，特别是其 Composer 功能，同时指出了在终端环境中使用 Aider 的效率。
- **对 Claude 开发的兴趣**：许多用户正考虑尝试 Claude 开发，理由是它在编码任务和辅助方面的潜在优势。
   - 讨论包括对 Claude 更新的期待，以及与现有工具相比它将如何提高生产力。
- **实时音频 API 的引入**：GPT-4o Realtime API 音频交互功能的发布已公布，专为低延迟对话应用设计。
   - 该 API 支持客户支持和实时翻译等用例，但需要客户端集成来处理最终用户的音频流。
- **Aider 以中文字符回复的问题**：一位用户报告在使用 o1-mini 模型时，Aider 返回了中文字符，表明存在潜在问题。
   - 这引发了关于故障排除以及 AI 模型在生成预期输出时面临的普遍挑战的讨论。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://supermaven.com/blog/funding-announcement">We raised $12M to build a text editor</a>：未找到描述</li><li><a href="https://www.notdiamond.ai/">Not Diamond</a>：Not Diamond 是世界上最强大的 AI 模型路由。</li><li><a href="https://simonwillison.net/2024/Oct/2/not-digital-god/">OpenAI DevDay: Let’s build developer tools, not digital God</a>：我昨天在 OpenAI DevDay 进行了有趣的直播博文更新——我现在分享了关于我在当天匆忙搭建的直播博文系统的笔记（在……的协助下）</li><li><a href="https://research.nvidia.com/labs/adlr/NVLM-1/">NVLM: Open Frontier-Class Multimodal LLMs</a>：我们推出了 NVLM 1.0，这是一个前沿级多模态大语言模型（LLMs）家族，在视觉语言任务上取得了最先进的结果，足以媲美领先的封闭模型（例如……）</li><li><a href="https://aider.chat/docs/config/options.html#--show-diffs">Options reference</a>：关于 aider 所有设置的详细信息。</li><li><a href="https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/audio-real-time?pivots=programming-language-ai-studio">How to use GPT-4o Realtime API for speech and audio with Azure OpenAI Service - Azure OpenAI</a>：了解如何在 Azure OpenAI Service 中使用 GPT-4o Realtime API 进行语音和音频处理。</li><li><a href="https://en.wikipedia.org/wiki/Roko%27s_basilisk">Roko&#039;s basilisk - Wikipedia</a>：未找到描述</li><li><a href="https://www.bram.us/2022/01/11/yaml-the-norway-problem/">YAML: The Norway Problem</a>：本周早些时候，Haroen Viaene 发布了这条关于 YAML 的推文：yaml 最糟糕的部分：https://yaml.org/type/bool.html —— Haroen Viaene (@haroenv) 2022 年 1 月 10 日。链接页面包含文档……</li><li><a href="https://github.com/paul-gauthier/refactor-benchmark">GitHub - paul-gauthier/refactor-benchmark: Aider&#39;s refactoring benchmark exercises based on popular python repos</a>：基于热门 Python 仓库的 Aider 重构基准测试练习 - paul-gauthier/refactor-benchmark</li><li><a href="https://github.com/paul-gauthier/aider/issues/1814">Plugin architecture for aider · Issue #1814 · paul-gauthier/aider</a>：问题反馈/功能请求 - 为 aider 创建插件架构。这可以用于例如为 Aider 创建自定义命令。除了扩展 Aider 的用途外，它还可能鼓励更多……</li><li><a href="https://github.com/paul-gauthier/aider/commit/2c32fe5eb8cf86378187ac1274515cdcc2cd1d72">Adopt safe_abs_path · paul-gauthier/aider@2c32fe5</a>：未找到描述</li><li><a href="https://www.bram.us/2022/01/11/yaml-the-norwa">YAML: The Norway Problem</a>：本周早些时候，Haroen Viaene 发布了这条关于 YAML 的推文：yaml 最糟糕的部分：https://yaml.org/type/bool.html —— Haroen Viaene (@haroenv) 2022 年 1 月 10 日。链接页面包含文档……
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1291131067905871956)** (22 条消息🔥): 

> - `Refactor-Benchmark 使用方法`
> - `Aider 与多个代码库`
> - `CONVENTIONS.md 文件命名`
> - `编程规范示例`
> - `Aider 自动补全问题` 


- **Refactor-Benchmark 使用方法澄清**：一名成员咨询了关于运行 [refactor-benchmark](https://github.com/paul-gauthier/refactor-benchmark) 以获取任务和对比完整报告的问题。
   - 对于 `Code refactoring leaderboard` 的任务是否需要与编辑基准测试（editing benchmark）分开执行存在困惑。
- **Aider 与多个 Git 仓库**：有用户提问 Aider 是否可以同时在多个 Git 仓库中工作，以便编写兼容的客户端代码。
   - 一名成员指出，虽然目前无法直接实现，但在 [Aider FAQ](https://aider.chat/docs/faq.html#can-i-use-aider-with-multiple-git-repos-at-once) 中提供了变通方案。
- **CONVENTIONS.md 文件命名**：一位用户询问 `CONVENTIONS.md` 的命名是否是强制性的，或者是否可以用其他名称代替。
   - 澄清了该文件名仅是一种惯例，但在 GitHub 项目中被广泛使用。
- **编程规范示例**：有人询问除了 Aider 网站上的示例外，是否还有其他 `CONVENTION.md` 的示例。
   - 一名成员引导他们前往 [awesome-guidelines repository](https://github.com/Kristories/awesome-guidelines)，查看精选的编程风格规范列表。
- **Aider 自动补全疑虑**：一位用户分享称，在克隆的 Aider main 分支中，`/read-only` 命令的自动补全功能无法正常工作。
   - 开发者提到它的运行方式与 `/add` 命令不同，并鼓励尝试使用 `aider --install-main-branch` 安装最新版本。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/faq.html#can-i-use-aider-with-multiple-git-repos-at-once">FAQ</a>：关于 aider 的常见问题。</li><li><a href="https://aider.chat/docs/leaderboards/#code-refactoring-leaderboard)">Aider LLM Leaderboards</a>：LLM 代码编辑能力的定量基准测试。</li><li><a href="https://github.com/paul-gauthier/refactor-benchmark">GitHub - paul-gauthier/refactor-benchmark</a>：基于流行 Python 仓库的 Aider 重构基准测试练习。</li><li><a href="https://github.com/paul-gauthier/aider/blob/main/benchmark/README.md">aider/benchmark/README.md</a>：Aider 是你终端里的 AI 配对编程工具。欢迎在 GitHub 上通过创建账号为 Aider 的开发做出贡献。</li><li><a href="https://github.com/Kristories/awesome-guidelines">GitHub - Kristories/awesome-guidelines</a>：高质量编程风格规范和标准的精选列表。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1291167013468901386)** (13 条消息🔥): 

> - `Crawl4AI`
> - `Not Diamond Router`
> - `Open Hands Resolver`
> - `OpenAI DevDay`
> - `Canvas for ChatGPT` 


- **Crawl4AI 发布**：[Crawl4AI](https://github.com/unclecode/crawl4ai) 的 GitHub 仓库展示了一个开源、对 LLM 友好的 Web 爬虫和抓取工具，旨在为寻求可定制解决方案的开发者提供服务。
   - 该工具可以增强各种项目的数据收集能力，并提供与语言模型的集成。
- **Not Diamond 模型路由亮相**：全新的 [Not Diamond 模型路由 (model router)](https://www.notdiamond.ai/) 声称能以高精度高效连接各种模型，用于规划旅行或分析技术报告等定制任务。
   - 用户可以在不到五分钟的时间内训练自己的优化路由，使其适用于各种应用场景。
- **Open Hands Resolver 系统**：[OpenHands resolver](https://github.com/All-Hands-AI/OpenHands-resolver) 项目旨在利用 OpenHands 框架自动解决 GitHub 仓库中的 issue。
   - 该计划通过自动化故障排除工作，可以显著简化项目维护流程。
- **OpenAI DevDay 洞察**：对 [OpenAI DevDay](https://simonwillison.net/2024/Oct/1/openai-devday-2024-live-blog/) 的实时博客记录分享了关于新功能的笔记和思考，包括 prompt caching 和模型音频流。
   - 关键讨论强调了构建开发者工具比创建过度复杂的模型更重要，社区表达了对实际应用的需求。
- **对 '.io' 域名的担忧**：在英国宣布将查戈斯群岛归还给毛里求斯后，关于可能移除 “.io” 域名的讨论浮出水面，引发了对 ccTLDs 未来的疑问。
   - 作为 [datasette.io](https://datasette.io/) 的所有者，人们开始担心对依赖此类域名的用户的影响，强调了政策变更需要透明度。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://simonwillison.net/2024/Oct/3/what-happens-to-io-after-uk-gives-back-chagos/">Ask HN: 英国归还查戈斯群岛后，“.io” TLD 会发生什么？</a>：今天早晨 BBC 报道：[英国将把查戈斯群岛的主权移交给毛里求斯](https://www.bbc.com/news/articles/c98ynejg4l5o)。查戈斯群岛包括英国称为英属印...</li><li><a href="https://uithub.com/">uithub - 轻松向你的 LLM 提问代码问题</a>：未找到描述</li><li><a href="https://simonwillison.net/2024/Oct/2/not-digital-god/">OpenAI DevDay：让我们构建开发者工具，而不是数字上帝</a>：昨天我在 OpenAI DevDay 进行实时博客记录时度过了一段愉快的时光——我现在分享了关于当天匆忙搭建的实时博客系统的笔记（在……的帮助下）</li><li><a href="https://www.notdiamond.ai/">Not Diamond</a>：Not Diamond 是世界上最强大的 AI 模型路由器。</li><li><a href="https://github.com/All-Hands-AI/OpenHands-resolver">GitHub - All-Hands-AI/openhands-resolver: 一个尝试使用 OpenHands 解决 GitHub 仓库中所有 issue 的系统。</a>：一个尝试使用 OpenHands 解决 GitHub 仓库中所有 issue 的系统。 - All-Hands-AI/openhands-resolver</li><li><a href="https://github.com/unclecode/crawl4ai">GitHub - unclecode/crawl4ai: 🔥🕷️ Crawl4AI: 开源、对 LLM 友好的 Web 爬虫与抓取工具</a>：🔥🕷️ Crawl4AI: 开源、对 LLM 友好的 Web 爬虫与抓取工具 - unclecode/crawl4ai
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/)** (1 条消息): 

alexatallah: https://x.com/SambaNovaAI/status/1841901026821210131
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1291112577572802651)** (112 条消息🔥🔥): 

> - `DeepInfra 停机故障`
> - `GPT-4o 降价`
> - `Claude 2.1 审核问题`
> - `NVLM 1.0 发布`
> - `Flash 8B 模型定价与速度` 


- **DeepInfra 经历短暂停机**：DeepInfra 经历了约 **15 分钟** 的停机，但据报道正在恢复中。
- **GPT-4o 大幅降价**：GPT-4o 模型从今天起大幅降价，**输入端降价 50%**，**输出端降价约 33%**。
   - 此项变动涉及自 8 月起可用的更新模型 GPT-4o-2024-08-06。
- **Claude 2.1 的审核机制引发担忧**：用户报告称 Claude 2.1 和其他模型错误地标记了 **SFW (安全内容) 提示词**，影响了用户交互。
   - 一个具体案例涉及角色描述被标记为“性暗示”内容，引发了对审核标准的质疑。
- **NVIDIA 发布 NVLM 1.0 模型**：NVIDIA 宣布了 **NVLM 1.0** 模型，该模型可与领先的专有模型竞争，并提供开源权重和代码。
   - 此次发布预计将提升视觉语言任务和纯文本能力的准确性。
- **Flash 8B 模型进入生产阶段**：Flash 8B 模型现已投入生产，但据报道其速度为 **每秒 200 tokens**，与普通的 Flash 相比被认为较慢。
   - 讨论表明未来可能有速度升级，并考虑了较低的硬件利用率。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.]">未找到标题</a>: 未找到描述</li><li><a href="https://evalplus.github.io/leaderboard.html">EvalPlus 排行榜</a>: 未找到描述</li><li><a href="https://openrouter.ai/chat">Chatroom | OpenRouter</a>: LLM Chatroom 是一个多模型聊天界面。添加模型并开始聊天！Chatroom 将数据本地存储在您的浏览器中。</li><li><a href="https://simonwillison.net/2024/Oct/2/not-digital-god/">OpenAI DevDay：让我们构建开发者工具，而不是数字上帝</a>: 我昨天在 OpenAI DevDay 进行了有趣的直播博文更新——我现在分享了关于当天在匆忙中搭建的直播博文系统的笔记（在……的协助下）。</li><li><a href="https://research.nvidia.com/labs/adlr/NVLM-1/">NVLM: 开源前沿级多模态 LLMs</a>: 我们推出了 NVLM 1.0，这是一系列前沿级多模态大语言模型 (LLMs)，在视觉语言任务上达到了最先进的结果，足以媲美领先的专有模型（例如，...）。</li><li><a href="https://www.notdiamond.ai/">Not Diamond</a>: Not Diamond 是世界上最强大的 AI 模型路由。</li><li><a href="https://openrouter.ai/models/cognitivecomputations/dolphin-llama-3-70b">Dolphin Llama 3 70B 🐬 - API, 提供商, 统计数据</a>: Dolphin 2.9 专为指令遵循、对话和编程设计。通过 API 运行 Dolphin Llama 3 70B 🐬。</li><li><a href="https://huggingface.co/nvidia/NVLM-D-72B">nvidia/NVLM-D-72B · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/OpenRouterTeam/open-webui">GitHub - OpenRouterTeam/open-webui: 适用于 LLMs 的用户友好型 WebUI（原 Ollama WebUI）</a>: 适用于 LLMs 的用户友好型 WebUI（原 Ollama WebUI） - OpenRouterTeam/open-webui
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1291118344606449666)** (101 条消息🔥🔥): 

> - `Mojo Python Imports` (Mojo Python 导入)
> - `Mojo Functions and Behaviors` (Mojo 函数与行为)
> - `Error Handling Strategies` (错误处理策略)
> - `Static Data Storage in Mojo` (Mojo 中的静态数据存储)
> - `SIMD Initialization Issues` (SIMD 初始化问题)


- **Mojo 在处理 Python 导入方面存在困难**：讨论显示 Mojo 无法原生处理 Python 的动态导入行为，这使得集成和错误管理变得复杂。
   - 几位成员指出，将导入职责委托给 CPython 可能会引入类似于 NPM 生态系统中的安全风险。
- **Mojo 函数遇到返回值问题**：成员们发现，从 Mojo 函数返回值有时需要变量声明（例如使用 `var`），因为常量可能会导致运行时错误。
   - 分享了一个例子，其中 `SIMD` 初始化失败，除非修改为返回一个可变对象。
- **探索错误处理策略**：对话集中在 Mojo 错误处理的潜在改进上，建议倾向于 Zig 风格的推断错误类型错误联合（error unions）。
   - 一些成员主张在错误管理中集成更多函数式编程（FP）方法，强调模式匹配和组合性。
- **静态数据存储的复杂性**：用户寻求在 Mojo 中静态存储表的方法，而不产生由于 `List` 等构造导致的过度代码膨胀，这会导致不理想的二进制文件大小。
   - 重点在于匹配 C 语言静态声明中的性能和内存效率。
- **SIMD 初始化问题引发 GitHub 讨论**：有人请求针对 `SIMD.__init__` 构造函数的异常行为创建一个 GitHub issue，该构造函数在某些条件下会返回错误。
   - 成员们表示愿意协助追踪 `SIMD` 相关 bug 的根本原因。



**提到的链接**：<a href="https://github.com/msaelices/mojo-openai-realtime-api/blob/ed0e04e2de493428729a98594e3d974480d03798/tests/test_event_handlers.mojo#L13">mojo-openai-realtime-api/tests/test_event_handlers.mojo at ed0e04e2de493428729a98594e3d974480d03798 · msaelices/mojo-openai-realtime-api</a>：Mojo OpenAI Realtime API 客户端。通过在 GitHub 上创建一个账户来为 msaelices/mojo-openai-realtime-api 的开发做出贡献。

  

---



### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1291456847689027774)** (1 条消息): 

> - `Canvas feature` (Canvas 功能)
> - `ChatGPT enhancements` (ChatGPT 增强)
> - `GPT-4o` 


- **为写作和编码推出的 Canvas 功能**：OpenAI 发布了 **canvas** 功能的早期版本，允许用户在超出简单聊天交互的写作和编码项目上进行协作。从今天开始，Plus 和 Team 用户可以通过在模型选择器中选择 [“GPT-4o with canvas”](https://openai.com/index/introducing-canvas/) 来试用。
- **使用 GPT-4o 增强项目工作流**：**GPT-4o with canvas** 的引入是改进项目管理和协作用户体验的一步。该功能使用户能够在处理复杂任务时利用先进的 AI 能力。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1291113071846101003)** (77 条消息🔥🔥): 

> - `API Access and Rate Limits`
> - `OpenAI's Copilot App`
> - `Fine-Tuning Models`
> - `Canvas Feature`
> - `Creating a Fake Language` 


- **API 访问层级混淆**：讨论围绕 **API 访问** 逐步推向特定使用层级展开，一名用户反映尽管之前有访问权限，但现在遇到了 **403 错误**。
   - 另一位成员指出，在达到过高的请求量时，**处理速率限制 (Rate Limit) 问题**并有效处理错误非常重要。
- **对新 Copilot 应用的印象**：一位用户对 **新 Copilot 应用的流畅表现** 表示惊讶，并指出它是 **Android 上的原生应用**。
   - 另一位成员赞赏其功能，但遗憾无法删除聊天记录，并将其与另一个聊天机器人进行了对比。
- **查找微调模型 ID**：针对如何查找 **微调模型 (Fine-Tuned Models)** 特定 ID 的提问得到了解答，一名成员分享了指向仪表板的链接以便检索。
   - 提供的解决方案被确认有效，展示了社区在协助使用 OpenAI 工具方面的支持。
- **关于 Canvas 功能的讨论**：用户讨论了新的 **Canvas 功能**，一些人表示兴奋，而另一些人则注意到其访问权限受限于桌面或移动平台。
   - 分享了关于目前推广和可用性的说明，提到移动端用户仍然可以查看 Canvas 对话。
- **创建并利用虚构语言**：一位成员分享了他们的创意尝试，成功生成了一种 **虚构语言** 和一个辅助使用的电子表格。
   - 这引发了关于 AI 在语言创造和消息体验中作用的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://cookbook.openai.com/examples/how_to_handle_rate_limits">How to handle rate limits | OpenAI Cookbook</a>: 用于构建 OpenAI API 的开源示例和指南。浏览代码片段、高级技术和演练集合。分享您自己的示例和指南。</li><li><a href="https://rapidapi.com/instant-ai-instant-ai-default/api/simple-gpt1">Simple GPT</a>: &lt;a href=&quot;https://apps.microsoft.com/detail/9n9jvnfmn3jl?mode=direct&quot;&gt; 	&lt;img src=&quot;https://get.microsoft.com/images/en-us%20light.svg&quot; width=&quot;200&quot; /&gt; &lt;/a&gt;
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1291165014572400640)** (4 条消息): 

> - `Voice Feature in Custom GPTs`
> - `Google API Integration with Custom GPTs` 


- **自定义 GPT 现已支持语音功能**：一位成员对今天在 **GPT Store** 提供的自定义 GPT 中引入语音功能表示感谢，并感谢 OpenAI 团队解决了这个问题。
   - 然而，他们注意到该语音模式并非新的 **高级语音 (Advanced Voice)**，他们希望未来所有自定义 GPT 都能包含该功能。
- **在自定义 GPT 中集成 Google API 的困难**：另一位成员分享了过去尝试将 **Google API / OAuth** 与自定义 GPT 集成的经历，称其在初始发布期间非常不稳定。
   - 他们还没有回头查看现在的集成是否更加稳定，这表明了对该功能的持续关注。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1291113928092418158)** (7 条消息): 

> - `Midjourney 中的 Seed 值获取`
> - `Ninetails 训练数据问题`
> - `小模型与大模型的性能对比`
> - `理解 AI 幻觉（Hallucinations）`
> - `训练数据错误 vs 幻觉` 

- **在 Midjourney 中获取 Seed 值**：一位用户询问如何从 **Midjourney** 网页版创建的图片中获取 **seed number**。
   - 另一位成员将他们引导至之前的频道，以获取关于此话题的进一步指导。
- **4o-mini 中 Ninetails 训练数据的缺陷**：一位用户发现，当询问火属性宝可梦时，**4o-mini** 总是错误地将 **Ninetails**（九尾）识别为有 6 条尾巴，而 **4o** 则能提供正确答案。
   - 这种模式在三次生成中均出现，表明这可能是一个训练数据缺陷，而非典型的幻觉。
- **大小模型之间的性能差异**：关于 **Ninetails** 的问题似乎影响了像 **gpt-3.5-turbo** 和 **gpt-4o-mini** 这样的小模型，而大模型则能提供准确的回答。
   - 有推测认为，在小模型的训练过程中，训练数据可能优先处理了错误信息。
- **澄清 AI 幻觉（Hallucinations）**：一位成员强调，幻觉通常是不可预测的，涉及模型生成不稳定或极具创造性的响应。
   - 相比之下，持续性的错误回答更能说明是训练数据错误，因为它们遵循既定的模式。
- **训练数据错误与幻觉的区别**：进一步阐述了训练数据错误与幻觉之间的区别，强调一致性的错误回答指向训练问题。
   - 一位用户指出，可预测的错误响应模式与幻觉中典型的随机猜测有本质区别。

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1291113928092418158)** (7 条消息): 

> - `Midjourney Seed 获取`
> - `4o-mini 训练问题`
> - `LLM 回答一致性` 

- **如何在 Midjourney 上获取 Seed 值**：一位用户询问如何获取在 **Midjourney** 网页端创建的图像的 **seed number**，但该问题被重定向到另一个频道寻求帮助。
- **4o-mini 的特定训练问题**：一位成员注意到 **4o-mini** 始终将 **Ninetails** 识别为有 6 条尾巴，而大模型则能正确识别出 **Vulpix**（六尾），这表明存在潜在的训练缺陷。
   - 这种重复性的错误与预期不符，表明模型的训练可能优先采用了错误信息而非正确信息。
- **大小模型差异**：另一位成员观察到，只有较小的模型（如 **gpt-4o-mini** 和 **gpt-3.5-turbo**）表现出这种错误行为，而不像大模型那样提供准确答案。
   - 这引发了关于训练数据和模型架构的疑问，即为什么这个问题仅存在于较小的变体中。
- **澄清 AI 幻觉 vs 训练错误**：讨论强调了 **hallucinations**（幻觉）与持续性训练错误之间的区别，强调幻觉通常涉及不可预测的输出。
   - 相反，一致的错误答案表明模型存在理解缺陷或特定的训练数据错误，而不是随机猜测。

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1291127156650545283)** (94 messages🔥🔥): 

> - `使用虚拟环境以保持稳定性`
> - `使用 AI 模型生成图像`
> - `频道中的合作伙伴与营销查询`
> - `图像生成挑战`
> - `模型训练与 ControlNet` 


- **虚拟环境是兼容性的关键**：成员们建议使用 **venv** 或 **conda** 等虚拟环境，以避免在运行 **AUTOMATIC1111** 等工具时不同 Python 版本之间发生冲突。
   - *虚拟环境*允许独立的包管理，使得在不干扰现有工作流的情况下处理不同的设置变得更加容易。
- **选择合适的 AI 模型和 UI**：建议新用户使用 **Comfy UI**，因为它具有灵活性并能访问新功能，尽管 **Automatic1111** 在教程方面仍然很受欢迎。
   - 成员们强调 **Forge UI** 是 **Automatic1111** 的一个更快的分支，但 **Comfy UI** 由于其基于节点的设计，可能提供更多的通用性。
- **生成特定姿势的图像**：用户讨论了让 AI 生成特定姿势图像的挑战，并建议使用 **ControlNet** 对输出进行精确控制。
   - 会议强调，训练特定模型（如 **LoRA**）和调整权重有助于定制生成的图像，以更好地满足用户期望。
- **应对 AI 模型的局限性**：对话涉及了在旧 GPU 上运行 **SDXL** 等高级模型的挑战，一些成员建议 **AMD** 用户使用 **ZLUDA** 等替代方案。
   - 讨论者承认，虽然使用较低的分辨率可以加快处理速度，但与适合特定模型的高分辨率相比，可能无法获得理想的结果。
- **尝试 AI 模型训练**：一位用户分享了他们尝试训练 AI 模型的经验并遇到了复杂情况，最终因训练不当图像而导致被封禁。
   - 这突显了在进行 AI 图像生成时，仔细选择训练图像并遵守社区标准的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/CS1o/Stable-Diffusion-Info">GitHub - CS1o/Stable-Diffusion-Info: Stable Diffusion Knowledge Base (Setups, Basics, Guides and more)</a>：Stable Diffusion 知识库（设置、基础、指南等）- CS1o/Stable-Diffusion-Info</li><li><a href="https://aka.ms/PSWindows">Migrating from Windows PowerShell 5.1 to PowerShell 7 - PowerShell</a>：将您的 Windows 平台从 PowerShell 5.1 更新到 PowerShell 7。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1291135359962452028)** (63 条消息🔥🔥): 

> - `Nous Research Bittensor 子网`
> - `Grok 使用`
> - `FLUX1.1 Pro 发布`
> - `LLaMA-3.1-SuperNova 合并`
> - `AI 助手对社会的影响` 


- **Nous Research Bittensor 子网关闭**：一名成员询问 Nous Research 为何停止其 Bittensor 子网，质疑竞争和时间需求是否是原因之一。
   - 另一名成员直接作出了回应，提议通过私信继续对话。
- **Grok 使用需要验证**：围绕访问 Grok 是否需要验证和付费展开了讨论，对于这些步骤的必要性意见不一。
   - 讨论中澄清，部分用户不需要验证，但需要为访问该服务付费。
- **FLUX1.1 Pro 发布**：宣布发布 **FLUX1.1 Pro**，声称其生成速度比前代快六倍，同时提升了图像质量和提示词（prompt）遵循能力。
   - 该公告强调了效率的提升，标志着其生成技术迈出了重要一步。
- **LLaMA-3.1-SuperNova 合并见解**：分享了 LLaMA-3.1-SuperNova-Lite 及其基础模型合并过程的细节，重点关注密度（density）作为合并中的关键参数。
   - 讨论了 Benchmark 结果，以突出此次合并与之前迭代相比的有效性。
- **AI 助手让社会变得更懒惰**：有人担心 AI 助手导致实际编程技能下降，特别是依赖 ChatGPT 等工具的学生。
   - 成员们注意到一种社会趋势，即看重学位而非真正的学习，导致在教育环境中缺乏参与度。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://illuminate.google.com/home">Illuminate | 以你的方式学习</a>：使用 Illuminate 将研究论文转换为 AI 生成的音频摘要，这是你更快理解复杂内容的 Gen AI 工具。</li><li><a href="https://blackforestlabs.ai/announcing-flux-1-1-pro-and-the-bfl-api/">宣布 FLUX1.1 [pro] 和 BFL API</a>：今天我们发布了 Flux1.1 PRO 和我们的 API，迫不及待想看到用户使用我们最新最棒的产品会构思出什么 <3</li><li><a href="https://huggingface.co/Joseph717171/Llama-3.1-SuperNova-8B-Lite_TIES_with_Base">Joseph717171/Llama-3.1-SuperNova-8B-Lite_TIES_with_Base · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1291131558400364625)** (17 条消息🔥): 

> - `用于故事创作的 LLM`
> - `LLM 函数效率`
> - `LanceDB 性能`
> - `Nous-Hermes-Llama2-13b 评估`
> - `Embedding 模型` 


- **寻找无审查的故事创作 LLM**：一位用户询问了最适合创作故事的 **LLM**，要求既无审查又可以作为 API 运行。
   - 他们还在寻找能自动构建故事，而不仅仅是提供标准帮助的 LLM 故事创作网站。
- **LLM 函数的单个 vs 多个 API 请求**：一位用户询问是否有指标来确定是单独使用 **LLM Functions** 还是将其组合在单个 API 请求中以获得最佳结果。
   - 评论建议，使用单个任务通常能提高推理能力，并引用了一篇相关的[论文](https://arxiv.org/html/2408.02442v1)。
- **LanceDB 的快速性能和混合替代方案**：一名成员分享了他们使用 **LanceDB** 的经验，提到了它的**速度**和云集成能力。
   - 他们推荐将 **DuckDB** 用于混合数据库，并指出 **AWS 上的 Elasticsearch** 是一个性能强大但具有挑战性的选项。
- **Nous-Hermes-Llama2-13b 在 ARC 上的评估方法**：一位用户寻求关于 **Nous-Hermes-Llama2-13b** 在 ARC 数据集上评估方法的澄清，询问是 zero-shot 还是 few-shot 评估。
   - 确认了 ARC 是使用 **zero-shot prompting** 进行评估的。
- **关于高性价比 Embedding 模型的咨询**：另一位用户征求关于运筹学和计算机科学领域最**便宜**且最好的 **embedding 模型**建议。
   - 对话凸显了对 embedding 技术中经济且有效的解决方案的关注。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1291228521276117023)** (2 条消息): 

> - `Softmax 函数的局限性`
> - `基于群论的知识图谱嵌入` 


- **Softmax 函数在精确决策中失效**：论文讨论了 **softmax function** 在输入数量增加时无法一致地执行稳健计算的问题，这从根本上限制了它对精确函数（sharp functions）的逼近能力。
   - *即使是像寻找最大键（maximum key）这样简单的任务也表明，随着输入规模的增长，学习到的电路会发生分散*，这挑战了人们对 softmax 预测能力的信心。
- **知识图谱嵌入的统一视角**：提出了一种新的 **knowledge graph embedding (KGE)** 方法，通过群论的视角整合了不确定性，同时保持了计算效率和表达能力。
   - *实体和关系被嵌入为对称群中的置换*，使得现有模型能够在此框架内得到有效表示。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2410.01104">softmax is not enough (for sharp out-of-distribution)</a>: 推理系统的一个关键属性是对其输入数据做出精确决策的能力。对于当代的 AI 系统，精确行为的一个关键载体是 softmax 函数，凭借其能力...</li><li><a href="https://arxiv.org/abs/2409.19977v1">Knowledge Graph Embedding by Normalizing Flows</a>: 知识图谱嵌入 (KGE) 的关键是选择合适的表示空间，例如逐点欧几里得空间和复向量空间。在本文中，我们提出了嵌入的统一视角...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1291401911429365831)** (2 条消息): 

> - `FLUX1.1 Pro`
> - `图像生成模型`
> - `Black Forest Labs` 


- **FLUX1.1 Pro 超越其前身**：**FLUX1.1 Pro** 的发布带来了比 **FLUX.1 Pro** 快六倍的生成速度，并提升了图像质量和提示词遵循度（prompt adherence），详见 [发布公告](https://blackforestlabs.ai/announcing-flux-1-1-pro-and-the-bfl-api/)。
   - 用户现在可以从改进的性能和效率中受益，使其成为优化工作流的理想选择。
- **FLUX1.1 Pro 统治图像生成基准测试**：以代号“blueberry”推出的 **FLUX1.1 Pro** 在广受欢迎的 [Artificial Analysis 图像竞技场](https://artificialanalysis.ai/text-to-image/arena) 中获得了最高的总 **Elo score**。
   - 这一新模型通过超越目前排行榜上的所有其他模型，展示了其卓越性。
- **混合架构的应用**：所有公开的 **FLUX.1 models** 都采用了基于 [多模态原理](https://arxiv.org/abs/2403.03206) 的混合架构，增强了其在图像生成方面的能力。 
   - 这种创新方法有助于提高模型的质量和性能。



**提到的链接**: <a href="https://replicate.com/black-forest-labs/flux-1.1-pro">black-forest-labs/flux-1.1-pro – 在 Replicate 上通过 API 运行</a>: 未找到描述

  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1291228521276117023)** (2 messages): 

> - `Softmax function limitations` (Softmax 函数的局限性)
> - `Knowledge graph embedding with uncertainty` (带有不确定性的知识图谱嵌入)


- **Softmax Function's Inherent Limitations** (Softmax 函数的固有局限性)：该论文揭示了 **softmax function** 在逼近尖锐函数（sharp functions）方面的一个根本性局限，断言即使对于简单任务，任何学习到的电路在测试期间都需要随着项目数量的增加而分散。
   - 作者从理论上证明了这一现象，并建议使用 **adaptive temperature** 作为潜在的解决方案。
- **Unified Perspective on Knowledge Graph Embedding** (知识图谱嵌入的统一视角)：本文通过引入群论中的不确定性，提出了一种新的 **knowledge graph embedding** (KGE) 方法，并提出了一个通用、高效且具有表现力的模型。
   - 论文强调，实体和关系的嵌入可以被视为 **symmetric group** 的元素，这提供了一种反映不同属性的方法，并确认了现有模型也可以在该框架内构建。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2410.01104">softmax is not enough (for sharp out-of-distribution)</a>：推理系统的一个关键属性是对输入数据做出尖锐决策的能力。对于当代的 AI 系统，尖锐行为的一个关键载体是 softmax 函数，凭借其能力...</li><li><a href="https://arxiv.org/abs/2409.19977v1">Knowledge Graph Embedding by Normalizing Flows</a>：知识图谱嵌入 (KGE) 的关键是选择合适的表示空间，例如点向欧几里得空间和复向量空间。在本文中，我们提出了嵌入的统一视角...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/1291134682422837272)** (3 messages): 

> - `Model Thought Process Controls` (模型思维过程控制)
> - `Trade Secrets Protection` (商业机密保护)
> - `User Transparency Issues` (用户透明度问题)


- **Controls to Conceal Model's Thought Process** (隐藏模型思维过程的控制)：一位成员表示，目前已采取某些控制措施来防止泄露模型的 **chain of thought**，包括指令其相信自己不具备思维。
   - 这引发了人们的担忧，即这种方法是否会损害模型有效 **解释自身** 的能力。
- **Concerns over Trade Secrets vs User Understanding** (关于商业机密与用户理解的担忧)：另一位成员对这些控制措施的必要性表示质疑，指出虽然 OAI 寻求保护其 **trade secrets**，但要求模型对用户隐瞒其 **thought processes** 感觉是不妥的。
   - 这种情绪凸显了 AI 模型交互中 **transparency** 与 **security** 之间的紧张关系。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1291114284318720062)** (74 条消息🔥🔥): 

> - `语音朗读功能讨论`
> - `订阅问题与客户支持`
> - `性能与模型质量担忧`
> - `使用扩展程序与 API Credits`
> - `用户界面与体验反馈` 


- **语音朗读功能评价褒贬不一**：用户讨论了语音朗读功能的潜力，尽管存在**发音 (pronunciation)**问题，一些用户发现听长回复比阅读更容易。
   - 一位成员分享说，他们经常在工作时使用该功能，暗示这提升了他们的使用体验。
- **对订阅和客户支持感到沮丧**：多位用户对订阅问题表示不满，包括下载文件的问题，以及在安全问题上未收到支持部门的回复。
   - 一位用户表示，由于这些问题似乎未得到重视，正考虑取消订阅。
- **模型输出质量不一致**：一位用户指出模型输出质量仍然不稳定，觉得在 Collection 或 Pro 套餐下变得“愚笨”。
   - 另一位成员强调了极端的性能不稳定性，使得产品有时似乎无法使用。
- **聊天扩展程序与 API Credit 困惑**：用户讨论了安装一个允许选择 **Gemini Pro** 模型的 Chrome 扩展程序，并质疑其在 Pro 套餐中的可用性。
   - 用户对 API credits 的意外扣费表示担忧，并建议联系支持部门以迅速解决这些问题。
- **用户界面差异讨论**：针对不同平台之间用户界面的差异提出了担忧，一些用户无法访问其他用户通常可以使用的功能。
   - 建议检查缩放设置或重新加载界面，以排查这些差异。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/discord-profile-theme-your-ur-gif-27000336">Discord Profile GIF - Discord Profile Theme - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/bocchi-the-rock-kita-ikuyo-anime-gif-27260096">Bocchi The Rock Kita Ikuyo GIF - Bocchi The Rock Kita Ikuyo Anime - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://youtu.be/2oQ5VkW-DZ8?si=E-tVw46rLfsQyaW8">如何使用 Large Action Model (AI) 安排任何任务</a>：了解如何通过 Nelima 全新的调度功能将您的操作提升到新水平！在本视频中，我将向您介绍如何使用 Nelima 强大的...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1291190567761477702)** (9 条消息🔥): 

> - `AI 对未来电影的影响`
> - `Perplexity 与 GPT-4 的对比`
> - `负时间存在的证据`
> - `OpenAI 的融资情况`
> - `Microsoft 的战略举措` 


- **探讨 AI 对未来电影的影响**：一篇文章讨论了 **AI** 对未来电影的潜在影响，详细说明了技术如何改变叙事和制作过程。你可以在[这里](https://www.perplexity.ai/page/ai-s-impact-on-future-movies-v.cRWJeZRZWW.O1QghbU.A)阅读更多相关内容。
   - 讨论强调了电影制作中的**新兴趋势**，以及 AI 如何重新定义观众参与度。
- **Perplexity 与 GPT-4 的对比引发热议**：分享的链接揭示了 **Perplexity** 和 **GPT-4** 的对比，突出了它们不同的功能和性能指标。点击[这里](https://www.perplexity.ai/search/perplexity-vs-gtp-4o-C_N5YDaIR2ykLv0.uYcBLA)查看。
   - 这场辩论引发了社区关于哪个平台在实际应用中更具优势的有趣见解。
- **关于负时间的新发现浮出水面**：最近的一篇文章阐述了**负时间**这一迷人概念，提出了证据及其对物理学的影响。你可以在[这里](https://www.perplexity.ai/page/evidence-of-negative-time-Ut987S07Rl2p3ryWJL_Pig)找到详细信息。
   - 讨论包含了挑战我们对**时间本身**理解的各种理论。
- **OpenAI 获得巨额融资**：一份公告透露 **OpenAI** 成功筹集了 **66 亿美元**，助力其正在进行的项目和创新。详情请见[这里](https://www.perplexity.ai/page/openai-raises-6-6b-ofVMnsDdRw.cUWz28MxjBA)。
   - 这笔资金预计将推动他们在 AI 研究和应用方面的进展。
- **讨论 Microsoft 的战略举措**：一位成员分享了对 **Microsoft** **新战略举措** 的见解，强调了其在科技领域的影响。阅读更多关于此举措的内容请点击[这里](https://www.perplexity.ai/page/another-blunt-move-by-microsof-tbKpeiInR4itu4NqX4ShSA)。
   - 对话强调了这一举措可能如何影响 AI 领域的竞争格局。



**提到的链接**：<a href="https://www.youtube.com/embed/lA1KQL83EHA">YouTube</a>：未找到描述

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/)** (1 条消息): 

ok.alex: 嘿 <@744572846721859615>！请私信我你的账户详情。
  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1291319697404461116)** (35 条消息🔥): 

> - `OpenAI 泡沫`
> - `Cohere 在 AI 领域的地位`
> - `对 AGI 的担忧`
> - `AI 中的数据隐私`
> - `硅谷文化` 


- **OpenAI 泡沫处于边缘**：几位成员讨论了 **OpenAI 泡沫** 正在扩张并可能很快破裂的感觉，特别是在 **o1** 发布暂时缓解了担忧之后。
   - 一位成员阴郁地警告说，OpenAI 的**命运**被比作 **WeWork** 动荡的过去，引发了对其长期生存能力的质疑。
- **Cohere 的低调策略**：一位成员对 **Cohere** 的做法表示赞赏，称其“在保持低调的同时处理各项事务”，并且与竞争对手相比显得更加务实。
   - 这种观点表明，在不断演变的 AI 格局中，保持低调可能是一种优势。
- **不断变化的 AGI 概念**：关于不断演变的 **AGI 概念** 出现了疑问，有人认为它在未来 **20 年** 将发生重大变化。
   - 这一观点令一些人感到震惊，引发了关于此类转变对 AI 领域影响的讨论。
- **对数据隐私的担忧**：在对**数据隐私**的担忧中，一位成员声称包括 OpenAI 在内的一些 AI 公司的议程是围绕从**中型公司** *窃取数据* 展开的。
   - 其他成员回应并质疑了这些说法的依据，强调了公司目前已有**选择退出 (opt-out)** 数据共享的选项。
- **硅谷的数据文化**：强调了**硅谷**普遍存在的态度，一位成员表示“AI 领域的每个人都在窃取”数据，而且通常**请求原谅**比请求许可更容易。
   - 这种情绪反映了对当前科技行业数据使用相关法律不确定性的广泛评论。


  

---

### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1291208788476694620)** (8 条消息🔥): 

> - `Reranking API Rate Limit`
> - `RAG++ Course Resource`
> - `Cohere LLM Data Collection`
> - `Cohere's Location Clarification`
> - `Output Tokens Information` 


- **Reranking API 触发速率限制**：一位用户报告在测试 **Reranking API** 时遇到了 **rate limit** 问题，尽管每次仅调用 **50 条记录** 且调用次数很少。
   - 这表明在测试阶段，免费层级（free tier）的使用可能存在潜在的基础限制。
- **分享 RAG++ 课程**：一名成员分享了 [RAG++ 课程](https://www.wandb.courses/courses/rag-in-production) 的链接，该课程专注于系统化评估技术和最佳实践，以提高 POC 应用的准确性。
   - 该课程包含 **Cohere credits** 以运行其 Notebook，提升了动手学习的可获得性。
- **寻求 LLM 数据协助**：一位用户请求协助核实一个学校项目中涉及的各种 **LLMs** 数据，询问有关发布日期和功能等具体细节的输入。
   - 另一位成员迅速纠正了 **Cohere** 的所在地，指出其位于 **Canada**，而非最初认为的美国。
- **Cohere 官网作为资源**：一名成员引导他人访问 [Cohere 的关于页面](https://cohere.com/about)，以获取有关其语言 AI 技术的全面信息。
   - 该页面强调了 Cohere 致力于将前沿研究与 AI 产品开发相结合的承诺。
- **查询每秒输出 Token 数**：一位用户寻求有关 Cohere LLM 的 **Output Tokens per Second** 指标的信息，表示目前可用数据存在缺口。
   - 他们随后在成功找到所需信息后表示满意。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://cohere.com/about">About</a>: 了解 Cohere，这家提供尖端自然语言处理解决方案的公司，致力于赋能企业和开发者利用语言 AI 的力量处理真实世界的用例。</li><li><a href="https://www.wandb.courses/courses/rag-in-production">Advanced RAG course </a>: 面向工程师的实用 RAG 技术：向行业专家学习生产就绪的解决方案，以优化性能、降低成本并提高应用程序的准确性和相关性。
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1291209206891941889)** (2 条消息): 

> - `Reranking API Rate Limit`
> - `Forcibly Invoking Tools` 


- **Reranking API 的速率限制困扰**：一位用户报告在测试 **Reranking API** 时遇到 **rate limit** 问题，尽管每次仅调用 **50 条记录** 且调用次数很少。
   - 这引发了关于 **free tier** 如何管理使用上限的担忧，可能会影响测试结果。
- **关于强制调用工具的咨询**：另一位用户询问是否可以 **forcibly invoking**（强制调用）工具，表明在交互中可能需要更多控制权。
   - 这表明成员们正在寻找绕过限制或影响工具行为的方法。


  

---

### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1291419799707975826)** (24 条消息🔥): 

> - `Project Posting Guidelines` (项目发布指南)
> - `Auto-Moderation Implementation` (自动审核实施)
> - `Job Posting Concerns` (招聘发布相关问题)
> - `Crypto Ad Quality` (加密货币广告质量)
> - `User Protection Measures` (用户保护措施)


- **明确项目发布规则**：成员们强调，在频道中展示 **Cohere 相关项目**时，需要制定明确的指南，禁止在展示中提供服务。
   - *伪装成项目的招聘发布*被视为违规，可能会导致内容从频道中移除。
- **实施自动审核**：一名成员确认 **Auto-Mod** 现已设置完成，以帮助管理项目频道中的不良内容。
   - 该措施旨在解决*招聘广告或垃圾帖子*等破坏社区专注度的问题。
- **强调禁止发布招聘信息**：MrDragonFox 等人坚决反对在频道中发布招聘信息，并表示：“禁止发布招聘——这甚至没有商量的余地。”
   - 社区对招聘垃圾信息表示担忧，认为完全禁止招聘发布比对其进行管理更容易。
- **对 Crypto 和垃圾信息质量的担忧**：讨论强调了 Crypto 和招聘广告的**质量问题**，促使大家认为这些内容最好在其他地方进行管理。
   - 成员们指出，很难区分合法内容与*网络钓鱼/Crypto 垃圾信息*，因此主张加强用户保护。
- **欣赏奋斗者，但不要在这里**：虽然成员们尊重那些寻求机会的人的奋斗精神，但他们认为此类活动不应在频道内进行。
   - 共识很明确：*欣赏这种努力，但请不要出现在项目展示中。*


  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1291119663635497067)** (45 messages🔥): 

> - `OpenAI's Canvas Interface` (OpenAI 的 Canvas 界面)
> - `Sam Altman's Influence` (Sam Altman 的影响力)
> - `OpenAI's Financial Outlook` (OpenAI 的财务前景)
> - `Liquid AI Architecture` (Liquid AI 架构)
> - `AI in Research Mathematics` (AI 在研究级数学中的应用)


- **OpenAI 推出 Canvas 以增强协作**：OpenAI 发布了一个名为 **Canvas** 的新界面，通过提供行内反馈和针对性编辑选项，使用户能够更有效地与 ChatGPT 协作进行写作和编程项目。
   - 然而，一些用户指出了其局限性，例如缺乏渲染的前端代码，以及无法有效追踪代码演变。
- **Sam Altman 在 OpenAI 的权力集中**：一篇文章探讨了 **Sam Altman** 如何在 OpenAI 集中其影响力，特别是在公司估值飙升至 **1570 亿美元** 的过程中。
   - 该文章促使读者反思公司的快速增长，同时评估强势领导力带来的影响。
- **OpenAI 到 2026 年雄心勃勃的营收预测**：**OpenAI** 的目标是到 2026 年产生与 **McDonald's** 和 **Mastercard** 等成熟公司相当的收入，这取决于能否成功增强功能以吸引更广泛的用户群。
   - 讨论集中在 OpenAI 是否能实现与这些巨头类似的盈利能力，因为其收入结构严重依赖于 **ChatGPT**。
- **对 Liquid AI 架构的担忧**：几位成员对新的 **Liquid AI** 架构的可行性和清晰度提出了**担忧**，将其描述为一个微小但值得注意的变化。
   - 一些人推测，如果他们拥有更优越的架构，应该优先考虑快速扩展以进行有效竞争。
- **AI 在研究级数学领域的能力**：一场备受关注的对话集中在 AI 是否能参与研究级数学，例如提出猜想和证明定理。
   - 讨论承认了 **LLM** 能力的前沿正在不断移动，反映出人们对 AI 在高级数学研究中作用的乐观情绪日益增长。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/karinanguyen_/status/1841888532299973056?s=46">Karina Nguyen (@karinanguyen_) 的推文</a>：这是自两年前发布以来，我们首次从根本上改变人类与 ChatGPT 的协作方式。我们推出了 canvas，这是一个与 ChatGPT 协作进行写作的新界面...</li><li><a href="https://x.com/amir/status/1841840024880550090?s=46">Amir Efrati (@amir) 的推文</a>：👀OpenAI 预计到 2026 年产生的收入将与 McDonald's、Ericsson 和 Mastercard 等公司相当。https://www.theinformation.com/articles/how-openai-cfo-sarah-friar-is-keeping-startup-...</li><li><a href="https://x.com/robertghrist/status/1841462507543949581">prof-g (@robertghrist) 的推文</a>：AI 能做研究级数学吗？提出猜想？证明定理？LLM 能做和不能做的事情之间存在一个移动的前沿。那个边界刚刚移动了一点。这是我的经验...</li><li><a href="https://x.com/MParakhin/status/1841516731011105217">Mikhail Parakhin (@MParakhin) 的推文</a>：@manic_pixie_agi 他们正在内部讨论。这有点棘手，因为整个公司的价值都在这个新架构中。</li><li><a href="https://x.com/mparakhin/status/1841571183957049605?s=46">Mikhail Parakhin (@MParakhin) 的推文</a>：@OxxoTweets @natolambert @ilyasut 我同意，界限会在那里——希望能有一个不同的、更低的界限。</li><li><a href="https://fxtwitter.com/paul_cal/status/1841891875436847299">Paul Calcraft (@paul_cal) 的推文</a>：@OpenAIDevs 用于代码快速审查的 Canvas - 代码审查在代码更改前以行内口头方式建议想法（然后你点击应用）- 很好的 UX - 没有更新代码的 diff 视图，追踪演变要困难得多...</li><li><a href="https://www.cnbc.com/2024/10/03/openai-gets-4-billion-revolving-credit-line-on-top-of-latest-funding.html">OpenAI 获得 40 亿美元循环信贷额度，使其流动资金超过 100 亿美元</a>：在最新一轮融资的基础上，OpenAI 已经建立了一个 40 亿美元的循环信贷额度，使其总流动资金超过 100 亿美元。</li><li><a href="https://x.com/modestproposal1/status/1841479310659473516?s=46">modest proposal (@modestproposal1) 的推文</a>：OpenAI 将为 60-65 亿美元的融资支付 9% 的利息？假设是 PIK（实物支付利息），因为当你每年烧掉 50 亿美元时，从 65 亿美元的融资中支付 6.5 亿美元是不太合理的。</li><li><a href="https://x.com/rachelmetz/status/1841881334752452918">Rachel Metz (@rachelmetz) 的推文</a>：我花了数周时间研究 Sam Altman 如何在 OpenAI 集中权力，特别是在过去的一年里，随着公司冲向 1570 亿美元的估值。从狂热中退后一步，看看...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1291120156805828769)** (3 条消息): 

> - `c.ai PR 问题`
> - `社区反应` 


- **c.ai 面临潜在的 PR 灾难**：一名成员警告称 c.ai 将面临一场 **PR 灾难**，预示着未来会出现严重问题。
   - 这引发了另一名成员的反应，他对这一进展表示“难过”。
- **社区表达失望**：正在进行的讨论传达了对 c.ai 现状的失望感。
   - *“是的……不意外，只是难过”* 捕捉到了社区沮丧但无奈的观点。


  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1291121353503801374)** (14 条消息🔥): 

> - `Shadeform 市场`
> - `O1 Preview`
> - `模型分析与 UX`
> - `博客文章创意` 


- **探索 Shadeform 的 GPU 市场**：一名成员讨论了使用 [Shadeform](https://www.shadeform.ai/) 预订按需 GPU 的好处，并利用其市场进行调度功能。
   - Shadeform 提供跨多云环境的集中计费和管理，方便部署工作负载。
- **O1 Preview 分享其思维过程**：一篇 Reddit 帖子强调了一个事件，即 O1 Preview 在回复中意外泄露了其完整的思维过程，引发了聊天中的有趣讨论。
   - 一名成员建议这可以作为一个很好的博客文章主题，幽默地触及了这种透明度的影响。
- **深入探讨 O1 的模型结构**：成员们对分析 O1 的结构表现出兴趣，指出它清晰地用标题划分了部分，从而暂停了用户参与。
   - 讨论提到了某些部分可能如何与用户的搜索习惯以及对模型功能的预测相一致。
- **将 O1 的呈现方式与其博客进行对比**：一名成员指出 O1 Preview 的 UX 让人联想到原始博客文章，表明两者具有很强的相似性。
   - 他们考虑进一步分析这些方面，但最终决定将精力集中在更有趣的话题上。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.reddit.com/r/ChatGPT/comments/1fussvn/o1_preview_accidentally_gave_me_its_entire/">Reddit - 深入探讨一切</a>：未找到描述</li><li><a href="https://www.shadeform.ai/">Shadeform - GPU 云市场</a>：在任何云环境中高效开发、训练和部署 AI 模型。访问多个 GPU 云的按需 GPU，并无缝扩展 ML 推理以获得最佳性能。</li><li><a href="https://pastebin.com/P0wQwvv9">o1preview - Pastebin.com</a>：Pastebin.com 是自 2002 年以来排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1291120537271275592)** (5 条消息): 

> - `Llama 团队`
> - `Google 的 AI 出版物`
> - `Meta 的发布风格` 


- **质疑 Llama 团队的真实性**：一名成员对 **Llama 团队** 表示怀疑，询问他们是否真的与之相关。
   - 这引发了在近期讨论背景下对其贡献 **真实性** 的担忧。
- **Google 复杂的 AI 使用**：针对这种怀疑，一名成员指出 **Google** 提到过类似的主题，但可能采用了更复杂的版本。
   - 这暗示了 AI 领域涉及的复杂性以及不同组织采取的不同方法。
- **对发布及时性的好奇**：另一名成员建议讨论的信息可能是 **旧的**，表明需要验证其相关性。
   - 这种担忧凸显了 AI 进步的快速节奏以及保持更新的重要性。
- **Meta 类似的发布氛围**：一名成员指出 **Meta** 可能具有与当前讨论相似的发布氛围。
   - 这一观察引起了人们对 AI 领域主要参与者所使用的不断演变的风格和策略的关注。


  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1291113040854384690)** (62 条消息🔥🔥): 

> - `OpenAI Canvas 发布`
> - `StackBlitz Bolt`
> - `Gartner 对 AI 工程的认可`
> - `Google 的 Gemini AI`
> - `Reflection 70B 模型复现`

- **OpenAI 推出 ChatGPT Canvas**：OpenAI 发布了一个名为 Canvas 的新界面，可在 ChatGPT 中实现更具协作性的项目，允许用户编辑代码并获取行内反馈。
   - 功能包括直接编辑能力、各种任务的快捷方式以及增强的研究功能。
- **StackBlitz 发布用于 AI 开发的 Bolt**：StackBlitz 推出了 [Bolt](http://bolt.new)，这是一个允许用户在 AI 辅助下进行 prompt、编辑、运行和部署全栈应用程序的平台。
   - 该开发环境支持 npm、Vite 和 Next.js，为开发者提供了一个全面且免费的应用创建工具。
- **Gartner 承认 AI 工程为一个领域**：Writer 被 Gartner 评为 Generative AI 技术的“新兴领导者”，表明 AI 在企业解决方案中的重要性日益增加。
   - 该认可涵盖了 Generative AI Engineering 和 AI Knowledge Management Apps 等领域，突出了该空间的创新。
- **Google 的 Gemini AI 开发**：据报道，Google 正在开发一种推理 AI 模型，旨在与 OpenAI 的能力（特别是在 'o1' 领域）展开竞争。
   - 这一举措延续了他们开发 AlphaGo 等先进 AI 系统的历史，并寻求在类人推理能力的基础上进一步发展。
- **Reflection 70B 模型基准测试讨论**：Sahil Chaudhary 分享了关于 Reflection 70B 模型的总结报告，解决了关于基准测试可复现性和输出质量的疑虑。
   - 社区成员继续就潜在的评估不一致性以及该模型对 AI 的整体贡献进行讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/fofrai/status/1841854401717403944?s=46">来自 fofr (@fofrAI) 的推文</a>：如果你给 FLUX1.1 一个像 "IMG_1018.CR2" 这样的提示词，你得到的图像将很难分辨出是 AI 生成的。这里的写实感提升了一个档次。</li><li><a href="https://x.com/dwarkesh_sp/status/1841494962824945718?s=46&t=6FDPaNx">来自 Dwarkesh Patel (@dwarkesh_sp) 的推文</a>：与 @dylan522p 和 @asianometry 合作的剧集上线了！这是一场关于半导体行业实际运作方式的盛宴。以及如果席（Xi）开始相信规模定律（scaling pilled）他会怎么做，我们如何训练比 GPT-4 强 10,000 倍的模型...</li><li><a href="https://x.com/stackblitz/status/1841873251313844631">来自 StackBlitz (@stackblitz) 的推文</a>：如果 AI 开发产品（Claude, v0 等）能让你安装包、运行后端并编辑代码会怎样？介绍由 StackBlitz 推出的 http://bolt.new：- 提示、编辑、运行和部署全栈应用 (fullstack apps) - 完整的开发环境 (dev env)...</li><li><a href="https://x.com/mattturck/status/1841623384955732189?s=46">来自 Matt Turck (@mattturck) 的推文</a>：今日市场总结：30 亿美元：一家 pre-IPO 软件公司的估值，因为按公开市场倍数以 3.75 亿美元营收的 8 倍计算。同样，30 亿美元：一个几乎没有任何收入的 AI agent 东西的估值...</li><li><a href="https://x.com/karpathy/status/1841594123381571863?s=46">来自 Andrej Karpathy (@karpathy) 的推文</a>：在过去的约 2 小时里，我策划了一个包含 10 集的新播客 (Podcast)，名为 "Histories of Mysteries"。在 Spotify 上可以找到：https://open.spotify.com/show/3K4LRyMCP44kBbiOziwJjb?si=432a337c28f14...</li><li><a href="https://x.com/hingeloss/status/1841540347035349501">来自 chris (@hingeloss) 的推文</a>：使用本地 Llama 1B 模型（又名 shrek sampler）实现的 o1 风格思维链 (chain of thought) 基本可以工作了... 难点在于智能地选择分支/注入的阈值，嗯... 引用 xjdr (@_xjdr) ...</li><li><a href="https://x.com/dwarkesh_sp/status/1841494962824945718?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Dwarkesh Patel (@dwarkesh_sp) 的推文</a>：与 @dylan522p 和 @asianometry 合作的剧集上线了！这是一场关于半导体行业实际运作方式的盛宴。以及如果席（Xi）开始相信规模定律（scaling pilled）他会怎么做，我们如何训练比 GPT-4 强 10,000 倍的模型...</li><li><a href="https://www.freepik.com/pikaso/ai-image-generator">Freepik AI 图像生成器 - 免费文本转图像生成器</a>：通过实时描述创建图像</li><li><a href="https://blackforestlabs.ai/announcing-flux-1-1-pro-and-the-bfl-api/">宣布 FLUX1.1 [pro] 和 BFL API</a>：今天我们发布了 Flux1.1 PRO 和我们的 API，我们迫不及待想看到用户会用我们最新最强的产品梦想出什么 <3</li><li><a href="https://x.com/karinanguyen_/status/1841890222415430090?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Karina Nguyen (@karinanguyen_) 的推文</a>：我们训练 GPT-4o 通过 canvas 作为一个创意伙伴进行协作，它可以为你自我演示其功能！而这个模型真正的魔力在于一切都是合成 (synthetically) 完成的，使得...</li><li><a href="https://x.com/karinanguyen_/status/1841889811931791642?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Karina Nguyen (@karinanguyen_) 的推文</a>：我对终极 AGI interface 的愿景是一个空白画布。它随着时间的推移根据人类偏好而进化、自我变形，并发明与人类互动的新方式，重新定义我们的关系...</li><li><a href="https://x.com/karinanguyen_/status/1841888532299973056?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Karina Nguyen (@karinanguyen_) 的推文</a>：自 ChatGPT 两年前发布以来，我们第一次从根本上改变了人类与它的协作方式。我们推出了 canvas，这是一个用于在 ChatGPT 上进行写作和...</li><li><a href="https://x.com/charles_irl/status/1841849736296288481">来自 Charles 🎉 Frye (@charles_irl) 的推文</a>：很高兴能弄清楚 Reflection 发生了什么。正如预期的那样，这不是诈骗，而是一堆经典的 MLOps 问题：eval bug、匆忙上线生产环境 (prod) 的研究代码、糟糕的工具。不是一个开发...</li><li><a href="https://glaive.ai/blog/post/reflection-postmortem">Reflection-70B 更新</a>：未找到描述</li><li><a href="https://x.com/_xjdr/status/1841678828361679130">来自 xjdr (@_xjdr) 的推文</a>：我建了一个仓库 (repo)，它非常简陋，因为我开始时并没打算发布它。目前还没有包含新的 sampler，但一旦稳定我会添加。它同时包含 jax 和 pytorch 实现...</li><li><a href="https://x.com/ricklamers/status/1841606740346839097?s=46">来自 Rick Lamers (@RickLamers) 的推文</a>：我对 Reflection 70B 的看法很简单：我在 HumanEval、GPQA 和 MMLU 上看到的分数很有趣，这表明“针对测试时推理 (test-time-inference) CoT 进行训练”似乎是有效的。很高兴一切...</li><li><a href="https://x.com/romainhuet/status/1841889813105971646?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Romain Huet (@romainhuet) 的推文</a>：OpenAI 将资助 Marijn Haverbeke，他是两个开源 (open source)...</li>

用于制作 ChatGPT canvas 的库，ProseMirror 和 CodeMirror。很高兴能支持 Marijn 的工作...</li><li><a href="https://x.com/natolambert/status/1841911374105936143?s=46">来自 Nathan Lambert (@natolambert) 的推文</a>：对于这支曾为我们带来 AlphaGo、AlphaZero、MuZero 以及更多先进搜索系统的团队，我并不感到意外 :) 正在寻找关于 o1 的独家消息：* 中国首家搞清楚它的机构 ...</li><li><a href="https://www.semianalysis.com/p/multi-datacenter-training-openais">多数据中心训练：OpenAI 超越 Google 基础设施的雄心计划</a>：吉瓦级集群、电信网络、长途光纤、分层和异步 SGD、分布式基础设施的赢家...</li><li><a href="https://x.com/yuchenj_uw/status/1841609474328715412?s=46">来自 Yuchen Jin (@Yuchenj_UW) 的推文</a>：@csahil28 @mattshumer_ “除了最初报告的两个分数外，我已经复现了其他所有分数” > 我们应该比较第一列和最后一列吗？最后四个基准测试之间存在差距，可能...</li><li><a href="https://x.com/soumithchintala/status/1841498799652708712">来自 Soumith Chintala (@soumithchintala) 的推文</a>：分为三个部分。1. 在 10k/100k/1m 个 H100 上拟合尽可能大的网络和尽可能大的 batch-size —— 并行化并使用节省内存的技巧。2. 通信状态...</li><li><a href="https://x.com/rinongal/status/1841739872198865109?s=46">来自 Rinon Gal (@RinonGal) 的推文</a>：摘要 - 我们通过微调 LLM 来预测为每个生成提示词定制的 ComfyUI 工作流，从而提高文本生成图像的输出质量。项目页面：https://comfygen-paper.github.io/ 论文：https://arxiv.o...</li><li><a href="https://x.com/fabianstelzer/status/1818305254909149621?s=46">来自 fabian (@fabianstelzer) 的推文</a>：介绍：ComfyAGI 🧙‍♂️😉 我们已经教会了 Claude 生成 ComfyUI 工作流，所以你现在只需通过提示词就能构建 Comfy 工作流... 我们正在开源整个提示词链...</li><li><a href="https://x.com/gdb/status/1841896254684725558?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Greg Brockman (@gdb) 的推文</a>：Canvas —— 与 ChatGPT 协作的新界面：引用 Karina Nguyen (@karinanguyen_) 自发布以来，我们首次从根本上改变了人类与 ChatGPT 的协作方式...</li><li><a href="https://x.com/deedydas/status/1841670760705949746">来自 Deedy (@deedydas) 的推文</a>：我知道我不该说这些，但这些 AI 公司：Character — 50 亿美元，SSI (Ilya) — 50 亿美元，Poolside — 30 亿美元，Devin (Cognition) — 20 亿美元，Magic — 15 亿美元，Codeium — 12.5 亿美元，Adept — 10 亿美元，Sierra — 10 亿美元，World Labs (...</li><li><a href="https://writer.com/blog/gartner-emerging-market-quadrant/">Writer 被评为 2024 年 Gartner® 生成式 AI 技术新兴市场象限的新兴领导者</a>：了解 Writer 如何被评为 2024 年 Gartner® 生成式 AI 技术新兴市场象限的新兴领导者。</li><li><a href="https://www.youtube.com/watch?v=jPluSXJpdrA">OpenAI 的 Noam Brown、Ilge Akkaya 和 Hunter Lightman 谈论 o1 以及教导 LLM 更好地推理</a>：将 LLM 与 AlphaGo 风格的深度强化学习相结合一直是许多领先 AI 实验室的圣杯，而通过 o1（又名 Strawberry），我们正在看到 ...</li><li><a href="https://x.com/atroyn/status/1841544410506657872?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 anton (𝔴𝔞𝔯𝔱𝔦𝔪𝔢) (@atroyn) 的推文</a>：发布 “ai dev explainer”，这是开始使用 LLM 构建 AI 应用程序的最佳资源。链接在下一条帖子中。</li><li><a href="https://t.co/WQZj6bZpqr">AI Dev Explainer</a>：未找到描述。
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1291455250246533142)** (1 条消息): 

> - `DevDay 回顾`
> - `OpenAI 洞察`
> - `音频体验` 


- **DevDay 回顾播客发布**：来自 [Latent Space Pod](https://latent.space/p/devday-2024) 的最新一期节目提供了 **DevDay 的全面音频体验**，邀请了关键贡献者参与。
   - 关键人物包括 **@oliviergodement**、**@romainhuet**、**@michpokrass**、**@AlistairPullen**，以及作为客座主持人的 **@simonw**，此外还有完整的 **@Sama** 和 **@kevinweil 问答环节**。
- **感谢组织者**：感谢 **<@194927177265840128>** 安排了 DevDay 的许多环节。
   - 他们的努力促成了活动中一系列有价值的讨论和见解。



**提到的链接**：<a href="https://x.com/latentspacepod/status/1841895518462456200">来自 Latent.Space (@latentspacepod) 的推文</a>：🆕 实时构建 AGI https://latent.space/p/devday-2024 我们的 @OpenAI DevDay 回顾现已上线！这是一次关于 DevDay 的全面音频体验，与幕后推手们一起：- @oliviergode...

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1291126112554061868)** (41 条消息🔥): 

> - `LM Studio 布局`
> - `Langflow 集成`
> - `LM Studio 更新影响`
> - `LM Studio 中的上下文管理`
> - `Flash Attention 功能` 


- **对 LM Studio 设置的困惑**：几位用户就如何将 **LM Studio** 与 **Langflow** 连接展开了讨论，其中一人询问如何更改 OpenAI 组件的 base URL。
   - 对话中透露出对讨论中消息查询的清晰度和语法准确性的不满。
- **LM Studio 版本更新的好处**：一位用户注意到，在保持所有其他设置不变的情况下，将 **LM Studio** 从版本 **0.2.31** 更新到 **0.3.3** 后，模型输出有所改善。
   - 这引发了关于是否使用了 key-value caching 及其对输出质量潜在影响的讨论。
- **上下文管理的局限性**：用户对 **LM Studio** 的无状态（stateless）特性表示担忧，其中一人请求能够在不重复输入的情况下跨会话维持上下文。
   - 另一位用户强调了提供持久上下文的挑战，因为模型本质上是无状态的。
- **Flash Attention 提升速度**：社区讨论了 **Flash Attention** 功能，一些用户对该功能在 GTX 等某些 GPU 型号上不可用表示沮丧。
   - 一位用户链接到了一个 [GitHub pull request](https://github.com/ggerganov/llama.cpp/pull/5021)，详细介绍了 Flash Attention 的设置，并声称它能显著加快处理速度。
- **Flash Attention 导致的 GUI Bug**：一位用户报告在使用 Flash Attention 时 **LM Studio GUI** 会消失，另一位用户指出下周发布的版本中将包含相关的 Bug 修复。
   - 这一问题突显了特定配置设置与平台使用之间的可能联系。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://johnthenerd.com/blog/local-llm-assistant/">构建一个完全本地的 LLM 语音助手来控制我的智能家居</a>：我曾使用过 Siri 和 Google Assistant。虽然它们有能力控制你的设备，但无法自定义，且本质上依赖云服务。希望能学到一些...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/5021">ggml : 由 ggerganov 添加 Flash Attention · Pull Request #5021 · ggerganov/llama.cpp</a>：ref #3365 在 ggml 和 llama.cpp 中设置 Flash Attention 支持所需的内容。提议的算子执行：// new res = ggml_flash_attn(ctx, q, k, v, kq_mask, kq_scale);  // fused sc...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1291182421772992512)** (8 条消息🔥): 

> - `单槽水冷头`
> - `电源配置`
> - `使用 GPU 供暖`
> - `M3 芯片性能`
> - `8B 模型的每秒 Token 数` 


- **考虑为 8 卡配置水冷**：一位成员正考虑为其 **8 卡** 方案购买 **单槽水冷头**，以保持外观整洁。
   - 他们目前使用 **两个 1600W** 和 **一个 1500W** 电源，并指出总功耗最高为 **4000W**。
- **电气安全建议**：一位社区成员建议布设新电线并安装适当规格的断路器以确保安全，并指出不当的设置可能导致严重危险。
   - 在一位用户报告即使使用 **15A 断路器** 在负载下仍会出现跳闸问题后，引发了相关担忧。
- **使用 GPU 的创新供暖方案**：一位参与者提出了用 GPU 为房屋供暖的想法，并强调这可能是一个创业概念。
   - 这一想法得到了另一位成员的共鸣，他一直在冬季使用 GPU 供暖，尽管他指出随着挖矿收益下降，这已变成一种经济损失。
- **M3 芯片的性能指标**：一位用户询问了在配备 **M3 芯片** 的 MacBook Air 上运行 **8B 模型** 时的 **tokens/sec** 性能。
   - 另一位成员报告称，在他们的 **M3 Max 128GB** 配置下，速率可以达到 **70s** 左右。
- **功耗对比**：一位用户将他们的 GPU 设置功耗与日常电器进行了对比，透露其 **热水系统** 耗电量为 **1800W**。
   - 他们还考虑利用 GPU 的散热器为热水箱加热，作为一种节省成本的措施。



**提到的链接**：<a href="https://huggingface.co/spaces/Qwen/Qwen2.5-Coder-7B-Instruct">Qwen2.5-Coder-7B-Instruct - Qwen 在 Hugging Face 上的 Space</a>：未找到描述内容。

  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1291212792547905547)** (20 条消息🔥): 

> - `量化算法`
> - `Int8 阈值退化`
> - `HQQ 性能`
> - `多 GPU 量化`
> - `Bitsandbytes 缓慢` 


- **大型模型量化算法讨论**：成员们讨论了适用于大型神经网络（50B+ 参数）且能将目标指标**损失保持在 1% 以内**的量化算法，重点介绍了 **int8** 和 **HQQ** 等技术。
   - *一位成员指出*，考虑到 int4 + HQQ 量化需要极少的校准，它也非常有效。
- **Int8 精度问题排查**：一位用户询问了如何排查使用 int8 量化时出现的严重精度退化问题，特别提到了 **默认 threshold=6.0** 的相关问题。
   - *另一位成员建议* 降低离群值的阈值，并引用了 [Hugging Face 的指南](https://huggingface.co/blog/hf-bitsandbytes-integration) 以获取更多见解。
- **HQQ 性能与利用**：强调了将 **HQQ** 与 **tinygemm** 和 **Bitblas** 等快速内核结合使用的优势，认为在许多场景下它的表现优于 bitsandbytes。
   - 成员们还分享了[这个教程](https://github.com/mobiusml/hqq/blob/master/examples/backends/transformers_demo.py)，用于在各种后端实现 HQQ。
- **多 GPU 量化查询**：关于在多 GPU 设置上运行 HQQ 的可行性提出了疑问，一位成员报告称 **一块 A100 GPU 可以处理 50B 模型的 4-bit 量化**。
   - *有人建议* 在多个 GPU 上进一步测试和利用 **HQQ** 库。
- **关于 Bitsandbytes 缓慢的担忧**：人们对 **Bitsandbytes int8 量化** 的缓慢表示担忧，特别是在使用非零阈值进行推理时。
   - 用户表示更倾向于 **HQQ** 等更快速的方法，理由是其优化效果和稳健的结果。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/transformers/main/en/quantization/hqq">HQQ</a>：未找到描述</li><li><a href="https://huggingface.co/blog/hf-bitsandbytes-integration">A Gentle Introduction to 8-bit Matrix Multiplication for transformers at scale using transformers, accelerate and bitsandbytes</a>：未找到描述</li><li><a href="https://github.com/pytorch/ao/tree/main?tab=readme-ov-file#post-training-quantization">GitHub - pytorch/ao: PyTorch native quantization and sparsity for training and inference</a>：PyTorch 原生量化与稀疏化，用于训练和推理 - pytorch/ao</li><li><a href="https://github.com/mobiusml/hqq/blob/master/examples/backends/transformers_demo.py">hqq/examples/backends/transformers_demo.py at master · mobiusml/hqq</a>：Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 条消息): 

as_ai: https://youtu.be/wGSSUSeaLgA
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1291129381242142802)** (2 条消息): 

> - `张量操作`
> - `Triton JIT`
> - `动态切片` 


- **张量 `X` 的动态切片**：一位成员寻求关于如何操作形状为 `[BLOCK_SIZE_INP * triton.next_power_of_2(n_inp_bits), 256, BLOCK_SIZE_OUT]` 的张量 `X` 的建议，目的是在不从内存加载数据的情况下移除第二维的一些元素。
   - 他们提议使用切片方法 `X[:, :BLOCK_HIDDEN_SIZE]`，其中 `BLOCK_HIDDEN_SIZE` 小于 **256**。
- **使用 Triton 进行高效切片**：同一位成员分享了一个使用 `@triton.jit` 的代码片段，用于函数 `_take_slice`，该函数旨在根据某些参数在保持维度的同时对张量进行切片。
   - 他们表示打算尝试提供的代码，以有效地实现其切片目标。


  

---


### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1291367692934320159)** (1 条消息): 

> - `项目时长`
> - `物流/组织挑战` 


- **更长的项目生命周期是有益的**：一位成员表示，虽然规模不是主要焦点，但**更长的项目持续时间**会非常有帮助。
   - 他们指出，项目通常需要时间才能步入正轨，拥有**充足的**剩余小时数将是有利的。
- **延长项目时间线的挑战**：讨论强调了增加项目时长会带来一系列自身的**物流/组织挑战**。
   - 管理延长的时间线的复杂性被认为是成功执行项目的潜在障碍。


  

---

### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1291161238163689552)** (4 messages): 

> - `Self-Compressing Neural Networks`
> - `Dynamic Quantization-aware Training`
> - `VRAM Budgeting in Model Training` 


- **关于 Self-Compressing Neural Networks 实现的咨询**：一名成员询问了 [GitHub 上的 Issue #658](https://github.com/pytorch/ao/issues/658) 的进展，该 Issue 涉及 **Self-Compressing Neural Networks**，重点是动态的 **Quantization-aware Training**。
   - 该任务的目标是将其作为训练期间的一个选项来实现，允许用户选择特定的 **VRAM budget** 并获得尺寸合适的模型。
- **调查 VRAM Budget 的实现**：另一名成员确认目前没有人正在处理该 Issue，但强调了将该技术作为 **experimental** 特性合并到 **torchao** 库中的兴趣。
   - 他们强调，这种方法可以解决用户需要管理 **VRAM budget** 的常见问题，类似于在 **Distillation** 等技术中看到的需求。



**提到的链接**：<a href="https://github.com/pytorch/ao/issues/658">Self compressing neural networks · Issue #658 · pytorch/ao</a>：Self-Compressing Neural Networks 是一种动态的 quantization-aware training，它将模型的大小放入 Loss 中。论文：https://arxiv.org/pdf/2301.13142 代码：https://github.com/geohot/ai-notebo...

  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1291122644506120224)** (5 messages): 

> - `Elon Musk and Haitian community`
> - `OpenAI funding update`
> - `Discord member count`
> - `Emoji reactions` 


- **Elon Musk 会见海地社区**：一名成员对针对海地社区的不当玩笑表示失望，并表示 *“我绝对没有欲望吃掉他们”*，并敦促 **Elon Musk** 予以关注。
   - 他们通过断言这并不可笑来强调事情的严重性，呼吁更广泛的理解。
- **OpenAI 的新一轮融资**：一名成员分享了一篇文章的链接，讨论了 **OpenAI** 的新一轮融资和重组计划，标志着重大的行业发展。
   - [Axios](https://www.axios.com/2024/10/02/openai-new-funding-round-restructuring) 上的文章概述了可能影响 AI 格局的关键变化。
- **Discord 服务器即将达到 10k 成员**：一名成员强调了他们的 **Discord** 服务器即将达到 **10,000 名成员** 的里程碑，展示了社区的增长。
   - 这一显著的增长证明了社区内讨论的参与度和兴趣。
- **Discord 表情符号回应**：成员们利用表情符号回应来表达他们对讨论的各种话题的感受，表明了活跃的社区互动。
   - 分享了如 `<:gigachad:1198826865016721550>` 和 `<:pmpp_icon:1199107527539961987>` 等表情符号，为对话增添了轻松的基调。


  

---


### **GPU MODE ▷ #[hqq-mobius](https://discord.com/channels/1189498204333543425/1225499037516693574/1291151381578649782)** (5 messages): 

> - `AWQ+HQQ results`
> - `HQQ implementation in TorchAO`
> - `Benchmark Evaluation`
> - `MMLU and GSM8K robustness` 


- **AWQ+HQQ 结果显示微小改进**：运行 **AWQ+HQQ** 的结果显示出一些 **微小的改进**，但仍需对剩余的 **Benchmark** 进行进一步评估。
   - 一名成员指出 *“现在的测试结果更有意义了”*，强调了进行全面分析的必要性。
- **MMLU 和 GSM8K 的鲁棒性受到赞赏**：成员们一致认为，像 **MMLU** 和 **GSM8K** 这样的 **Benchmark** 为性能比较提供了更具鲁棒性的评估指标。
   - 这种对比对于验证 **AWQ+HQQ** 测试阶段的改进至关重要。
- **TorchAO 中的 HQQ 表现略逊一筹**：一名成员指出，由于在处理 **zero-point** 方面的差异，**TorchAO** 中的 **HQQ implementation** 表现比原始版本稍差。
   - 这种差异对于解释 **Benchmark** 评估的结果具有重要意义。


  

---

### **GPU MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1291127447554756740)** (2 messages): 

> - `16K token batch processing` (16K token 批处理)
> - `Attention heads allocation` (Attention head 分配)
> - `NCCL communication strategy` (NCCL 通信策略)
> - `Zero Redundancy Optimizer` (ZeRO 冗余优化器)
> - `Activation checkpointing` (激活检查点)


- **优化 16K Token 批处理**：一名成员提议**每个 GPU 处理 1 个 16K token 的 batch**，而在 Attention 方面，让每个 GPU 管理 **1/8 的 head** 以处理 **128K tokens**。
   - 该方法允许在前后进行 **NCCL 通信**，且不会影响 **cuDNN/FA** 的功能。
- **手动拼接 vs. 提议的方法**：另一名成员最初考虑手动拼接 **softmax 部分**，但承认新提议似乎是更好的解决方案。
   - 他们的回应是 *“这听起来更好”*，表明其观点已转向提议的方法。
- **结合多种技术以提高效率**：初始方案将新的 batching 策略与 **ZeRO-3** 以及现有的 **activation checkpointing PR** 相结合，以增强性能。
   - 这表明研究方向正转向利用多种策略来提高模型训练效率。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1291128573603876894)** (1 messages): 

> - `Advancing AI event` (Advancing AI 活动)
> - `ROCM developers` (ROCm 开发者) 


- **参加在旧金山举行的 Advancing AI 活动**：**10/10** 在 **SF Moscone** 将举行一场 **Advancing AI 活动**，重点关注即将推出的硬件和软件。
   - 鼓励感兴趣的参与者通过 **DM** 获取注册详情，并与 **ROCm 开发者** 进行交流。
- **ROCm 开发者在活动中交流**：此次活动是与 **ROCm 开发者** **叙旧**并了解他们最新项目的绝佳机会。
   - 此次聚会旨在促进社区互动，并围绕 AI 技术的未来展开讨论。


  

---


### **GPU MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1291206066813669389)** (2 messages): 

> - `BF16 vs FP32 weights` (BF16 vs FP32 权重)
> - `Custom Optimizer Development` (自定义优化器开发)
> - `Stochastic Rounding Techniques` (随机舍入技术) 


- **探索 BF16 权重对准确性的影响**：一名成员对使用 **BF16** 权重而非 **FP32** 进行训练可能牺牲**准确性**表示担忧。
   - 他们指出，在当前的配置下，可能可以在 **4090 VRAM** 范围内对权重使用 **FP32**，同时将优化器保持为 **BF16**。
- **需要自定义优化器以混合数据类型**：讨论强调 **PyTorch** 的内置优化器不支持权重和优化器使用不同的 **dtype**，这促使一名成员考虑编写自己的优化器。
   - 他们参考了 **big_vision** 仓库，该仓库同样对权重使用 **FP32**，对优化器使用 **BF16**。
- **创新的随机舍入 (Stochastic Rounding) 方法**：有人建议另一种方法，即对权重和优化器均使用 **BF16**，同时在优化器计算中使用 **FP32** 并配合随机舍入。
   - 该技术提议在 **FP32** 的尾数 (mantissa) 中添加随机的 **16 bits**，利用其额外的位来提高效率，正如 [llm.c](https://github.com/karpathy/llm.c/blob/7ecd8906afe6ed7a2b2cdb731c042f26d525b820/lllc/adamw.cuh#L19-L46) 所演示的那样。



**提到的链接**：<a href="https://github.com/karpathy/llm.c/blob/7ecd8906afe6ed7a2b2cdb731c042f26d525b820/llmc/adamw.cuh#L19-L46">llm.c/llmc/adamw.cuh at 7ecd8906afe6ed7a2b2cdb731c042f26d525b820 · karpathy/llm.c</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。欢迎在 GitHub 上为 llm.c 的开发做出贡献。

  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1291374825671233556)** (1 messages): 

> - `Metal Programming Basics` (Metal 编程基础)
> - `Comparison of CUDA and Metal` (CUDA 与 Metal 的比较) 


- **理解 Metal 编程**：一名初学者表达了他们目前对 Metal 编程的理解，指出虽然 **CUDA** 使用 `block_size * grid_size` 进行线程调度，但 **Metal** 仅使用 grid size。
   - *他们强调 Metal 中的 threadgroups 是为 grid 之间的共享内存设计的。*
- **关于 MSL 规范的澄清**：这位初学者提到他们粗略阅读了 **MSL spec**，但不确定自己的理解是否完全准确。
   - *他们欢迎对其关于 Metal 线程模型解读的反馈。*


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1291112558413217793)** (25 条消息🔥): 

> - `AI 中的责任与合理使用`
> - `围绕爬虫合法性的问题`
> - `OpenAI 的审核政策`
> - `AI 研究机会`
> - `模型的 MMLU 评分` 


- **探讨 AI 研究中的责任**：讨论集中在分享 AI 模型用于研究的个人是否应对他人的滥用行为负责，一些人认为责任可能不会直接归咎于原始研究人员。
   - 成员们指出，可能需要明确的裁决来建立准则，并强调了在这些法律领域中明确性的必要性。
- **爬虫诉讼引发关注**：针对正在进行的关于网页爬虫法律地位的诉讼，人们表达了担忧，艺术家和作家对这种做法表示沮丧。
   - 引用了一个案例，其中公司试图禁止爬虫（除非满足严格条件）但未能成功，这突显了法律的复杂性。
- **OpenAI 审核政策的影响**：一位成员讲述了他们在 OpenAI 审核政策方面的经历，该政策标记了他们提示 AGI 的请求，因感知到的违规行为导致了令人不安的时刻。
   - 其他人一致认为这些政策似乎过于谨慎，并指出许多被标记的消息与陈述的使用政策并没有明确的相关性。
- **创意 AI 项目的机会**：一位新成员介绍自己是对比 AI 内部创意和社会技术分析感兴趣的研究员，正在寻求基于公地（commons-based）方法的合作项目。
   - 这突显了 AI 跨学科研究的潜力，特别是在数字人文领域。
- **分享 MMLU 评分资源**：一位成员询问如何获取新模型的 MMLU 分数，随后有人推荐了 EleutherAI 的 [evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness)。
   - 他们还提到了一个专门讨论该话题的频道，以促进协作学习。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/HiQ_Labs_v._LinkedIn">hiQ Labs v. LinkedIn - 维基百科</a>: 未找到描述</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness">GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1291125613788401736)** (13 messages🔥): 

> - `Self-Supervised Learning on Arbitrary Embeddings`
> - `Softmax Function Limitations`
> - `Learning Optimal Rank for LoRA Layers`
> - `ColBERT Embeddings Usage`
> - `Pretraining Alignment Projects` 


- **探索在任意 Embedding 上的 Self-Supervised Learning**：讨论强调了将 **Self-Supervised Learning (SSL)** 应用于来自任何模型和数据的任意 Embedding，旨在跨多个 **modalities** 对 **pretrained models** 进行 **SSL**。
   - 一位参与者提议更进一步，直接在任何 **model weights** 上应用 **SSL**，强调了数据集构建的灵活性。
- **Softmax 函数的尖锐决策迷思**：来自[这篇论文](https://arxiv.org/abs/2410.01104)的摘要揭示了 **Softmax function** 的一个关键局限性，断言随着输入数量的增加，它无法稳健地逼近 **sharp functions**。
   - 论文从理论上认为 **adaptive temperature** 是解决这一挑战的关键，但这引发了对所提解决方案强度的怀疑。
- **学习 LoRA 层秩（Rank）的潜力**：一位成员询问了学习或逼近 **LoRA layers** 最优 **rank** 的方法，而不是手动设置它们，这暗示了在自动化该过程方面可能取得突破。
   - 另一位用户引用了 [adaptive-span](https://github.com/facebookresearch/adaptive-span) 项目作为此次探索的灵感。
- **对 ColBERT Embeddings 的怀疑**：一位用户质疑为何 **ColBERT embeddings** 未被广泛采用，并指出它们在消除数据处理中的 **chunking** 需求方面具有潜力。
   - 另一位成员指出，与 **bm25+dpr** 相比，使用 **rerankers** 实际上消除了对额外复杂性的需求，并暗示了相当的 **recall** 结果。
- **对 Pretraining Alignment 项目的兴趣**：有人询问了目前与 **pretraining alignment** 或 **neural network architecture** 进展相关的项目，表明了对该领域的持续关注。
   - 未提供进一步信息，该查询仍处于开放状态，期待更多贡献或见解。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2410.01104">softmax is not enough (for sharp out-of-distribution)</a>: A key property of reasoning systems is the ability to make sharp decisions on their input data. For contemporary AI systems, a key carrier of sharp behaviour is the softmax function, with its capabili...</li><li><a href="https://arxiv.org/abs/2410.00907">Addition is All You Need for Energy-efficient Language Models</a>: Large neural networks spend most computation on floating point tensor multiplications. In this work, we find that a floating point multiplier can be approximated by one integer adder with high precisi...</li><li><a href="https://github.com/facebookresearch/adaptive-span">GitHub - facebookresearch/adaptive-span: Transformer training code for sequential tasks</a>: Transformer training code for sequential tasks. Contribute to facebookresearch/adaptive-span development by creating an account on GitHub.
</li>
</ul>

</div>

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1291156274033201253)** (5 条消息): 

> - `lm-eval-harness 指标问题`
> - `Hugging Face 数据集 PR 审批`
> - `Claude 3.5 Sonnet 评估` 


- **lm-eval-harness 中新指标的问题**：一位用户报告尝试在现有的多选题任务中添加新指标，但遇到了问题，称该指标未被添加到 **MedQA** 数据集中。他们提供了 [GitHub issue](https://github.com/EleutherAI/lm-evaluation-harness/issues/2330) 链接以获取更多上下文。
   - 问题似乎集中在更新指标上，强调了需要协助解决该问题。
- **Hugging Face 数据集需要 PR 审批**：一位成员请求拥有 **Hugging Face** 权限的人员审批一个关于 **CoQA** 数据集的 PR，并指出下载器未能正确处理重定向。他们详细说明了加载数据集时的问题，并提供了 Hugging Face 上的讨论链接。
   - 另一位成员作出了肯定答复，表示已经合并了必要的更改，原发布者对此表示感谢。
- **分享 Claude 3.5 Sonnet 评估结果**：一位用户分享了他们使用 **lm-eval-harness** 对 **Claude 3.5 Sonnet** 在 **GSM8K** 任务上的评估结果。他们请求其他人针对该评估中取得的更好性能提供见解。
   - 他们概述了用于评估的命令，包括模型参数和输出路径，并邀请与其他用户的评估结果进行对比。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/datasets/EleutherAI/coqa/discussions/1">EleutherAI/coqa · Fix URLs</a>: 未找到描述</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/issues/2330">Failed to add a new metric · Issue #2330 · EleutherAI/lm-evaluation-harness</a>: 你好，我尝试向现有的多选题任务添加一个新指标，但似乎该指标没有被添加。我编辑了 MedQA: task: medqa_4options dataset_path: GBaker/MedQA-USMLE-4-options-h...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1291445897674096652)** (1 条消息): 

> - `Eleuther 当前的活跃项目`
> - `开源软件需求`
> - `贡献机会` 


- **询问活跃项目**：一位成员询问了 **Eleuther 当前的活跃项目**，以便更好地了解团队的关注领域。
   - 他们表达了贡献的兴趣，特别是考虑到他们作为 **computer vision** 论文第一作者的背景。
- **征询开源软件需求**：同一位成员询问了团队目前有哪些 **开源软件需求**，以便进行潜在的贡献。
   - 这一请求突显了在 Eleuther 社区内寻求协作和支持的愿望。


  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/)** (1 条消息): 

seanchatmangpt: https://pypi.org/project/dslmodel/2024.10.3.3
  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1291165403137052723)** (42 条消息🔥): 

> - `DSPy 2.5 用户反馈`
> - `文档改进`
> - `AI Arxiv 播客`
> - `LLM 知识源`
> - `Prompt 流水线中的自我改进` 


- **DSPy 2.5 用户体验**：用户对 **DSPy 2.5** 的改进表达了积极反馈，表示尽管仍有一些不完善之处，但整体体验非常出色。
   - 一位用户指出 **TypedPredictors** 的变化很有前景，而另一位成员则主张提供更多关于 **customization**（自定义）的文档。
- **对更好文档的需求**：社区成员敦促改进文档，特别是围绕在工作流中使用 **Pydantic 和多个 LM** 的部分。
   - 反馈强调需要**更易于复制的指南**，以增强在复杂生成任务中的可用性。
- **AI Arxiv 播客介绍**：一个名为 **AI Arxiv** 的播客讨论了大厂如何应用 LLM，并分享了其最新一期作为社区资源。
   - 鼓励听众前往收听，并计划在周末前将视频上传到 **YouTube** 以扩大传播范围。
- **寻求 LLM 知识源**：一位成员征求 **AI/LLM 相关新闻和资源**的推荐，建议将 Twitter 和 subreddits 作为潜在来源。
   - 回复中包含了分享的链接，例如一个专注于相关内容和讨论的精选 Twitter 列表。
- **DSPy Prompt 流水线的自我改进**：一位成员询问了 DSPy Prompt 流水线与传统 LLM 训练流程相比的**自我改进**机制。
   - 推荐了一些讨论多阶段语言模型程序优化策略，以及微调与 Prompt 优化结合优势的论文。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://pypi.org/project/dslmodel/2024.10.3.3">dslmodel</a>: 基于 Prompt 和 Jinja 的 Pydantic + DSPy 实例。</li><li><a href="https://arxiv.org/abs/2406.11695">Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs</a>: 语言模型程序（即模块化语言模型 (LM) 调用的复杂流水线）正日益推动 NLP 任务的发展，但它们需要编写对所有阶段都有效的 Prompt...</li><li><a href="https://arxiv.org/abs/2407.10930">Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together</a>: 自然语言处理 (NLP) 系统正越来越多地采用涉及多个不同语言模型 (LM) 和 Prompt 策略的多阶段流水线形式。在这里，我们探讨了...</li><li><a href="https://podcasts.apple.com/ca/podcast/ai-arxiv/id1768464164?i=1000671470927">Episode 42 - ColPali: Efficient Document Retrieval with Vision Language Models</a>: 播客单集 · AI Arxiv · 2024-10-01 · 9m</li><li><a href="https://x.com/i/lists/1635546867328073729">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等 - FixTweet/FxTwitter
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[announcements](https://discord.com/channels/1216353675241590815/1216353675241590818/1291465364030488657)** (1 条消息): 

> - `Torchtune 0.3.1 发布`
> - `Llama 3.2 Vision 模型`
> - `知识蒸馏 (Knowledge Distillation) Recipe`
> - `MPS Beta 支持`
> - `文档全面修订` 


- **Torchtune 0.3.1 进行了重大更新**：**Torchtune 0.3.1** 补丁现在包含所有 **Llama 3.2 Vision 模型**，以及针对微调 recipes、生成和评估的全面多模态支持。
   - 主要亮点包括在 **8 张 A100 上使用 QLoRA 微调 Llama 3.1 405B**，增强了性能选项。
- **引入知识蒸馏 (Knowledge Distillation) Recipe**：新增了 **知识蒸馏 recipe**，并配备了 **Llama3.2** 和 **Qwen2** 的配置，扩展了用户的工具包。
   - 鼓励成员探索这些新功能，以提高模型的效率和性能。
- **MPS Beta 支持现已可用**：**MPS beta 支持**允许用户在 Macbooks 上使用 **Torchtune**，将微调能力带入 **Apple 生态系统**。
   - 这使得用户能够**随时随地微调模型**，增强了开发者的可访问性。
- **精简的内存管理**：此次更新引入了**流式激活卸载 (streamed activations offloading)**，以极小的性能影响降低内存消耗。
   - 该功能旨在减轻训练运行期间的资源需求，使其对大型模型更加高效。
- **大规模文档修订**：**大规模文档修订**专注于**基础知识 (Basics)**，涵盖了自定义数据集、多模态变换 (multimodal transforms) 等内容。
   - 用户可以访问升级后的资源来解决所有疑问和设置问题，详见 [Torchtune Documentation](https://pytorch.org/torchtune/stable/)。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/torchtune/stable/">欢迎来到 torchtune 文档 &mdash; torchtune 0.3 文档</a>：未找到描述</li><li><a href="https://github.com/pytorch/torchtune/releases/tag/v0.3.1">Release v0.3.1 (Llama 3.2 Vision 补丁) · pytorch/torchtune</a>：概述：我们在 Llama 3.2 发布后添加了对其的全面支持，这包括在 Llama3.2-1B、Llama3.2-3B 基础和指令文本模型以及 Llama3.2-11B-Visio...
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1291323371778478138)** (35 messages🔥): 

> - `Tokenizer truncation issues`
> - `Independent max_seq_len in packing`
> - `Flash Attention memory usage`
> - `HF dataset names and links`
> - `Model card generation` 


- **Tokenizer 自动截断导致损失**：**text completion dataset** 会自动截断超过 **max_seq_len** 的序列，导致长文档中的 Token 丢失，引发了对该行为增加更多**用户控制**的需求。
   - 成员们讨论了可能的解决方案，包括将 **packing max_seq_len** 与 Tokenizer 限制分离，以避免不必要的截断。
- **关于 packing 独立 max_seq_len 的辩论**：有一种提议认为，如果 **packing max_seq_len** 独立于 Tokenizer 的 max_seq_len，可以在不导致文档过度截断的情况下提高显存性能。
   - 讨论中提出了对 **self-attention 显存增长**的担忧，涉及其随序列长度是线性还是二次方扩展的问题。
- **探索 Flash Attention 的影响**：一名成员询问 **Flash Attention** 是否在计算量为二次方的情况下实现了线性显存增长，并指出在某些实验中显存消耗与 **number of tokens** 呈线性关系。
   - 对话强调了使用 Flash Attention 可能带来的计算成本，揭示了对其真实显存行为尚未达成共识。
- **提议更清晰的 HF 数据集引用方法**：建议创建一个类似 **DATASET_TO_SOURCE** 的映射，以便轻松获取项目中使用的实际 **HF dataset names**，从而增强 **model cards** 的自动生成。
   - 旨在简化链接实际数据集的过程，同时努力在 **YAML** 中生成更清晰的数据集文档。
- **快速 v0 实现与详细功能之间的权衡**：团队正在权衡快速部署基础版本 (v0) 与深入开发更精细功能集（如 **model cards and tagging**）之间的利弊。
   - 在讨论复杂需求的过程中，大家希望保持项目推进，避免在初始发布时过度分散精力到详细实现上。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/blob/0a3762d058fd9d860606a3a6bbebf71ac593ab5c/torchtune/datasets/_text_completion.py#L164">torchtune/torchtune/datasets/_text_completion.py at 0a3762d058fd9d860606a3a6bbebf71ac593ab5c · pytorch/torchtune</a>: A Native-PyTorch Library for LLM Fine-tuning. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/torchtune/blob/0a3762d058fd9d860606a3a6bbebf71ac593ab5c/torchtune/datasets/_text_completion.py#L161">torchtune/torchtune/datasets/_text_completion.py at 0a3762d058fd9d860606a3a6bbebf71ac593ab5c · pytorch/torchtune</a>: A Native-PyTorch Library for LLM Fine-tuning. Contribute to pytorch/torchtune development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1291140656516956272)** (2 messages): 

> - `MongoDB Atlas`
> - `Hybrid Search`
> - `Box Integration`
> - `AI-driven Content Management` 


- **使用 MongoDB Atlas 实现混合搜索**：一篇博客文章讨论了如何创建和配置 [MongoDB Atlas vector and full-text search indexes](https://t.co/VFsaL4XIdb) 以实现混合搜索。
   - 文章强调结合**语义搜索**和**全文搜索**来增强搜索结果的相关性。
- **集成 Box 构建智能应用**：一篇文章介绍了如何将 [Box tools](https://t.co/Ge42GVau8v) 与 LlamaIndex 结合使用，以构建 AI 驱动的内容管理应用。
   - 这允许在 Box 中进行**高级搜索**，旨在从 Box 内容中高效提取和处理信息。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1291143043084451890)** (27 条消息🔥): 

> - `RAG 系统问题`
> - `RAG 应用中的异步转换`
> - `使用 LlamaIndex 生成 RFP`
> - `VLLM 错误处理`
> - `LlamaIndex 中的实体和关系属性` 


- **教程中的 RAG 系统错误**：一位用户在按照教程使用 Excel 文件构建 RAG 系统时遇到了 `ModuleNotFoundError`，这表明安装的 pandas 版本可能存在问题。
   - 另一位用户建议尝试使用旧版本的 pandas（2.2.2 或更早版本）以解决兼容性问题。
- **RAG 应用中的异步转换挑战**：一位用户正在将 RAG 应用程序转换为异步模式，但不确定 `QueryEngineTool` 是否支持异步方法，以及 `RouterQueryEngine` 在其中扮演什么角色。
   - 针对如何在 `RouterQueryEngine` 中利用异步方法以实现平滑过渡提供了说明。
- **使用 LlamaIndex 生成 RFP 响应**：一位开发者寻求使用 LlamaIndex 构建一个系统，根据选定实体的中标提案生成 RFP（需求建议书）响应，并正在寻找高效的索引和事实替换策略。
   - 他们还询问了 LlamaIndex 是否具备根据响应生成 PDF 或 Word 文件的能力。
- **分享 VLLM 错误详情**：一位用户报告在尝试将 VLLM 用于其 RAG 实现时出现 `KeyError`，表明响应数据中缺少某个键。
   - 另一位成员请求提供完整的 traceback 以更好地协助诊断问题。
- **PropertyGraphIndex 中的实体属性**：一位成员询问为 `PropertyGraphIndex` 中的实体定义的属性是否在所有实体间共享，并指向了 `DynamicLLMPathExtractor` 中的 `allowed_entity_props` 参数。
   - 寻求关于实体关系属性的文档说明，以及 `SchemaLLMPathExtractor` 如何利用其输入。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/run-llama/llama_parse/blob/main/examples/excel/o1_excel_rag.ipynb">llama_parse/examples/excel/o1_excel_rag.ipynb at main · run-llama/llama_parse</a>: 解析文件以实现最佳 RAG。通过在 GitHub 上创建账户来为 llama_parse 的开发做出贡献。</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/vllm/">vLLM - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.co">GitHub: Let’s build from here</a>: GitHub 是超过 1 亿开发者共同塑造软件未来的地方。为开源社区做出贡献，管理你的 Git 仓库，像专业人士一样审查代码，跟踪错误和功能...
</li>
</ul>

</div>
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1291190015073849375)** (19 条消息🔥): 

> - `Jordan Pfost 的 AI 专业知识`
> - `Kapa.ai 的功能`
> - `理解 LLM 中的“喜欢”与“奖励”`
> - `AI 领域的实习机会` 


- **Jordan Pfost 的 AI 专业知识**：Jordan Pfost 介绍自己是一名拥有 **10 年** AI/Web 产品经验的高级全栈工程师，强调了在 **GPU Clustering**、**RAG** 和 **Agentic Reasoning** 方面的技能。
   - 他分享了在 **spendeffect.ai**、**iplan.ai** 和 **Pump GPT** 的项目经验，并表示有兴趣探索合作机会。
- **Kapa.ai 的功能**：Kapa.ai 解释了其作为基于 Transformer 的语言模型的功能，拥有约 **3.4 亿参数**，专为自然语言任务构建。
   - 它详细介绍了在多样化语料库上的训练情况，并强调其生成的文本符合类人质量标准，同时引用了 **LangChain 文档** 以供进一步探索。
- **理解 LLM 中的“喜欢”与“奖励”**：Kapa.ai 澄清说 LLM 不具备个人偏好或接收奖励，而是基于训练数据中的模式运行。
   - 它引用了一篇关于偏好优化的论文，同时提供了 **LangChain 文档** 的链接，以获取更多关于 LLM 运行机制的见解。
- **AI 领域的实习机会**：一位成员询问是否有来自印度的大学生正在寻找 AI 领域的实习机会，并邀请他们表达意向。
   - 该查询旨在将学生与 AI 领域的潜在实习机会联系起来。



**提到的链接**: <a href="https://python.langchain.com/v0.2/docs/how_to/#llms>).">操作指南 | 🦜️🔗 LangChain</a>: 在这里，你可以找到“我该如何……”这类问题的答案。

  

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1291266787656335422)** (1 messages): 

> - `LangGraph Query Generation`
> - `LangChain Ecosystem`
> - `Error Correction in Queries` 


- **LangGraph 处理查询生成与结构化**：一篇新的 [LinkedIn 帖子](https://www.linkedin.com/posts/ismile-bharmal-3b82241ab_langgraph-langchain-querygeneration-activity-7247467636719013888-CZHj)探讨了 **LangGraph** 如何在 **LangChain** 生态系统中处理复杂的查询生成和输出结构化。
   - 该帖子重点关注了**错误修正**和**用户友好型结果**，同时对 **Harrison Chase** 和 LangChain 团队的贡献表示了认可。
- **向 LangChain 团队致敬**：该帖子对 **LangChain** 团队，特别是 **Harrison Chase** 在开发 LangGraph 功能方面的贡献给予了高度评价。
   - 这突显了推动 AI 工作流创新与增强的协作努力。


  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1291119202882682984)** (5 messages): 

> - `October House Party`
> - `Open Interpreter Showcases` 


- **明天是十月家庭派对 (October House Party)**：别忘了！明天是 **October House Party** —— [点击此处加入](https://discord.gg/f6a7YXng?event=1288234745477726238) 参与趣味活动并获取更新。
   - 一位成员表达了兴奋之情，表示在被健康和工作耽误后，*这次绝对不会错过*。
- **展示你的 Open Interpreter 创作**：主持人邀请任何使用 **Open Interpreter** 构建了作品的人在派对期间展示他们的成果。
   - 他们鼓励参与者带着问题前来并分享经验。
- **对活动时间的反应不一**：一位成员评论说时间对他来说*太早了*，表明存在一些日程冲突。
   - 相反，另一位成员则热情地宣告 *PARTY TIMEEEE*，反映出对即将到来活动的期待。


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1291143519674826832)** (10 messages🔥): 

> - `Skill Teaching Capabilities`
> - `Model Compatibility`
> - `OpenAI Request Issues` 


- **成员讨论技能教学**：一位用户询问如何向他们的模型教授技能，促使另一位成员建议先确认教学技能的意图是否清晰。
   - 尽管进行了尝试，教学问题仍未解决，表明可能需要进一步的支持。
- **模型的 Vision 能力**：关于技能是否自带 Vision 能力存在困惑，据指出这取决于所使用的模型。
   - 具体而言，用户提到将 **gpt4o** 与 **Cartesia** 和 **Deepgram** 结合使用，成员们得出结论认为这在理论上应该是可行的。
- **OpenAI 请求失败**：一位用户报告称，在发送几条消息后，他们的 OpenAI 请求就会停止工作，且没有提供任何错误或日志。
   - 这种情况突显了系统中潜在的问题，鼓励成员开设新帖子进行故障排除。


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/)** (1 messages): 

mikebirdtech: 对 Mozilla 的 Public AI 有什么看法？

https://x.com/mozilla/status/1840741892977291695
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1291285424538517526)** (12 messages🔥): 

> - `Logo change feedback`
> - `Vllm and vision concerns`
> - `Demo usage experiences`
> - `Fine-tuning plans`
> - `Deployment strategies` 


- **Logo 变更引发褒贬不一的反应**：成员们对最近的 **Logo 变更** 反应不一，表情符号从困惑到沮丧皆有，表明接受程度各异。
   - *一位成员幽默地提到*，“我以为我把这个服务器从列表里弄丢了 😅。”
- **融资预期受到质疑**：一位成员幽默地表示，新 Logo 应该对应着以 **10 亿美元估值** 筹集 **1000 万美元**。
   - *另一位用户回应道*，“Sheeesh，” 表示对如此宏大的融资目标的难以置信或惊讶。
- **分享 Demo 体验**：一位成员分享了他们的 Demo 使用体验，称“还不赖，我通过 Demo 使用了它”，暗示了积极的互动。
   - 持续的对话表明成员们仍在适应这些变化。
- **微调 (Fine-tuning) 讨论进行中**：关于模型是否已经进行了微调的问题被提出，成员们确认目前尚未进行微调。
   - 一位成员保证微调很快就会进行，并强调了准备就绪后部署 **70B 参数模型** 的计划。


  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1291136186160775319)** (9 messages🔥): 

> - `Regex rules for spam blocking` (用于垃圾信息拦截的 Regex 规则)
> - `Google's Illuminate tool` (Google 的 Illuminate 工具)
> - `Automated Arxiv Paper Video Channel` (自动化 Arxiv 论文视频频道)


- **Regex 有效拦截垃圾信息**：一位成员分享了一个用于拦截 Markdown 链接混淆的 Regex 模式：`\[[^\]]+\]\(https?:\/\/[^)\s]+\)`。
   - 针对特定垃圾信息类型（如色情和加密货币）定制的 Regex 和词语黑名单可以有效减少垃圾机器人的存在。
- **60 秒超时威慑垃圾机器人**：在消息拦截后实施 60 秒的超时限制，是让垃圾机器人在尝试几次后离开的有效策略。
   - 该方法能最大限度地减少对真实用户的干扰，避免过多的误报。
- **Google 的 Illuminate 工具看起来很有前景**：一位成员分享了 [Google's Illuminate](https://illuminate.google.com/home?pli=1) 的链接，这是一个正在推出的令人兴奋的新工具。
   - 出现了关于该工具与 notebooklm 播客工具对比的问题，表明了对这两种实现的兴趣。
- **自动化 YouTube 频道 'Arxflix' 分享 Arxiv 论文**：另一位成员推广了他们的自动化 YouTube 频道 [Arxflix](https://www.youtube.com/@Arxflix)，致力于通过视频分享 Arxiv 论文。
   - 该成员对这个项目表示自豪，并暗示它可能比其他工具提供更具吸引力的内容。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://illuminate.google.com/home?pli=1">Illuminate | Learn Your Way</a>：使用 Illuminate 将研究论文转换为 AI 生成的音频摘要，这是您的 Gen AI 工具，可帮助您更快地理解复杂内容。</li><li><a href="https://www.youtube.com/@Arxflix">Arxflix</a>：未找到描述
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1291131163066499092)** (8 messages🔥): 

> - `tinybox delivery inquiry` (tinybox 交付查询)
> - `support email addition` (添加支持邮箱)
> - `questions document importance` (提问文档的重要性)
> - `FAQ improvements` (FAQ 改进)
> - `community culture` (社区文化)


- **用户关于 Tinybox 交付的咨询**：一位用户对美国境内 **tinybox** 的交付时间表表示担忧，询问是否能在 **2-5 天**内送达。
   - George Hotz 建议他们发送邮件至 *support@tinygrad.org* 咨询物流问题，并强调需要提出表述清晰的查询。
- **在 FAQ 中添加支持邮箱**：由于 **网站 FAQ** 中缺少支持邮箱，有人建议将其加入。
   - George 确认他会立即添加，表明了对社区反馈的响应。
- **澄清交付地点**：George 提出了一个关于交付地点相关性的观点，对向 **圣迭戈、密歇根或夏威夷** 等地区发货表示不确定。
   - 他强调了构思好问题的重要性，并引用了 #1068979651336216706 频道作为指导。
- **改进提问文档的用户协议**：George 建议为用户实施一个**点击确认协议 (click-through agreement)**，以确认他们已阅读提问文档，可能还包括多选题。
   - 另一位成员指出，目前已经存在针对用户的点击确认。
- **社区文化观察**：George 对社区围绕提问的文化表达了挫败感，称这是一个反复出现的问题。
   - 他敦促成员优先考虑清晰的沟通和正确的查询流程。


  

---



### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1291438571261005854)** (2 messages): 

> - `Inference Timings in SLM Systems` (SLM 系统中的推理耗时)
> - `RAG Architecture with Llama Index` (使用 Llama Index 的 RAG 架构)
> - `Course Material Availability` (课程资料可用性)


- **优化 SLM 系统中的推理耗时**：一位成员询问了在使用 [Llama Index](https://link.to.llama.index) 的 **RAG 架构** 的 **SLM 系统** 中，有哪些潜在的方法可以优化 **推理耗时 (inference timings)**。
   - 他们正在寻求社区建议以优化性能。
- **课程幻灯片已上线！**：另一位成员宣布 **slides** 现在可以在 **课程网站** 上获取。
   - 这一更新确保了参与者可以访问学习所需的必要材料。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-readings-discussion](https://discord.com/channels/1280234300012494859/1282735578886181036/1291395786147631238)** (1 条消息): 

> - `AI Reading Group`
> - `INDUS 研究论文`
> - `IBM 与 NASA 合作` 


- **AI Reading Group 启动讨论平台**：来自 Women in AI & Robotics 的 **AI Reading Group** 已启动，旨在由研究人员展示 AI 论文，促进他们与社区之间的对话。
   - 首场会议由来自 **IBM** 的 **Aashka Trivedi** 主讲，讨论与 **NASA** 的联合研究；观众 Q&A 环节的名额有限。
- **即将举行的 INDUS 论文演讲**：欢迎在 **2024 年 10 月 17 日** **美东时间中午 12 点** 参加 **AI Reading Group**，届时将对题为 [**INDUS: Effective and Efficient Language Models for Scientific Applications**](https://arxiv.org/abs/2405.10725) 的论文进行演讲。
   - 本次会议将由 **Aashka Trivedi** 主持，展示包括 **NASA** 和 **IBM** 在内的顶尖机构的合作研究成果。
- **展示合作研究成果**：由 **IBM Research AI**、**NASA** 等共同撰写的 **INDUS 论文** 强调了专为**科学应用**定制的语言模型的进展。
   - 该阅读小组旨在**揭秘**领先的创新技术，并促进 AI 领域的跨学科讨论。



**提到的链接**：<a href="https://www.eventbrite.ca/e/1024976160287?aff=oddtdtcreator">INDUS: Effective and Efficient Language Models</a>：AI Reading Group 会议，由 "INDUS: Effective and Efficient Language Models for Scientific Applications" 的作者之一主讲。

  

---



### **Alignment Lab AI ▷ #[general](https://discord.com/channels/1087862276448595968/1095458248712265841/1291396215686303825)** (1 条消息): 

> - `AI Reading Group 启动`
> - `INDUS 研究演讲`
> - `AI 社区参与`
> - `研究人员 Q&A 环节`
> - `活动参与人数限制` 


- **AI Reading Group 正式启动**：来自 Women in AI & Robotics 的 **AI Reading Group** 已启动，允许研究人员展示 AI 论文并进行 Q&A 环节。
   - 该倡议旨在为研究人员与社区之间的**直接对话**建立平台，增强对新兴研究的参与度。
- **即将举行的 INDUS 演讲**：首位演讲者是来自 IBM 的 **Aashka Trivedi**，她将于 **2024 年 10 月 17 日** 展示关于“[INDUS: Effective and Efficient Language Models for Scientific Applications](https://arxiv.org/abs/2405.10725)”的研究。
   - 该论文的作者来自包括 **IBM Research**、**NASA** 和 **Harvard-Smithsonian CfA** 在内的知名机构。
- **名额有限以保证互动**：由于**名额有限**以方便观众提问，有兴趣加入阅读小组的参与者需尽快报名。
   - 这一限制旨在确保演讲后的 **Q&A** 环节能够产生有意义的互动。
- **参与跨学科讨论**：AI Reading Group 旨在突出 AI 领域的**当前研究热点**，并为跨学科讨论提供空间。
   - 通过促进这些对话，该小组旨在揭秘领先的创新，并促进对相关研究的更深层次参与。



**提到的链接**：<a href="https://www.eventbrite.ca/e/1024976160287?aff=oddtdtcreator">INDUS: Effective and Efficient Language Models</a>：AI Reading Group 会议，由 "INDUS: Effective and Efficient Language Models for Scientific Applications" 的作者之一主讲。

  

---



### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1291362949083955294)** (1 条消息): 

> - `第三方数据集`
> - `数据集的代码修改` 


- **修改代码以支持第三方数据集**：目前的实现原生不支持**第三方数据集**，但一位成员建议通过修改代码来实现此功能。
   - 他们表示，调整将涉及添加用于解析逻辑的 **model handler**、更改测试文件映射以及选择合适的 checkers。
- **实现数据集解析逻辑**：为了集成新数据集，一位成员解释说需要使用 `decode_ast` 和 `decode_exec` 来实现解析逻辑。
   - 这种适配需要对 pipeline 如何处理数据集有基本的了解，以确保兼容性。


  

---



---



---



---



---



{% else %}


> 完整的频道逐条解析已因邮件篇幅原因截断。
> 
> 如果你想查看完整解析，请访问此邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢支持！

{% endif %}