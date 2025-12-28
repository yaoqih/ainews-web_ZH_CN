---
companies:
- glean
- sambanova
- cerebras
- stanford
- google
- apple
- hugging-face
- lmsys
date: '2024-09-11T02:24:16.042126Z'
description: '以下是为您翻译的中文内容：


  **Glean** 的估值再次翻倍。**Dan Hendrycks 的 Superforecaster AI** 通过有趣的提示工程生成了合理的选举预测。**斯坦福大学**的一项研究发现，大语言模型（LLM）生成的科研创意在统计学上比人类专家的创意更具新颖性。**SambaNova**
  宣布了针对 **Llama-3** 模型更快的推理速度，超越了 **Cerebras**。**Benjamin Clavie** 就检索增强生成（RAG）技术发表了一场备受关注的演讲。据报道，**Strawberry（草莓模型）**
  将于两周内发布。**Google Illuminate** 提供由 AI 生成的关于论文和书籍的播客式讨论。**苹果公司**在 iOS 18 中展示了新的 AI
  功能，包括视觉智能和改进的 Siri，支持通过相机添加事件的端侧及云端处理。**Reflection 70B** 模型因其性能声明引发了争议。专家强调了 MMLU
  和 HumanEval 等传统基准测试的不可靠性，建议采用 **LMSys Chatbot Arena** 和 Hugging Face 开源的 **Lighteval**
  套件等替代评估方法。AI 研究界正继续探索 AI 在生成新颖研究创意和改进基准测试方面的作用。'
id: 4166fea7-d8ed-4606-bb4c-3d44b0f7aa12
models:
- superforecaster-ai
- llama-3
- reflection-70b
original_slug: ainews-not-much-happened-today-ainews-podcast
people:
- danhendrycks
- benjamin-clavie
- bclavie
- bindureddy
- swyx
- borismpower
- corbtt
- drjimfan
- clementdelangue
- rohanpaul_ai
title: 今天没发生什么 (Not Much Happened Today) + AI新闻播客？
topics:
- prompt-engineering
- research-ideas
- inference-speed
- retrieval-augmented-generation
- evaluation-methods
- visual-intelligence
- on-device-ai
- model-performance
- benchmarking
- novelty-detection
---

<!-- buttondown-editor-mode: plaintext -->**只需要再等两周...**

> 2024年9月9日至9月10日的 AI 新闻。我们为您检查了 7 个 subreddit、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord（**215** 个频道和 **2311** 条消息）。预计节省阅读时间（以 200wpm 计算）：**247 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

让我们来看看：

- Glean [估值再次翻倍](https://x.com/glean/status/1833476578912989281?s=61)
- Dan Hendrycks 的 [Superforecaster AI](https://x.com/danhendrycks/status/1833152719756116154?s=46) 生成了非常可信的选举预测？人们想知道它在辩论后会如何更新。[查看 Prompt](https://x.com/danhendrycks/status/1833163197626601603?s=46)。
- 一篇关于 [LLM 生成新颖研究思路](https://x.com/chengleisi/status/1833166031134806330?s=46) 的斯坦福论文广为流传，并提出了一个重大声明：“*经过为期一年的研究，我们得出了第一个具有统计学意义的结论：LLM 生成的想法比人类专家研究员撰写的想法更具新颖性。*”
- SambaNova [宣布其 Llama 3 推理速度略快于](https://www.linkedin.com/posts/sambanova_fastai-ugcPost-7239272368198557697-7FMk) 之前的世界纪录保持者 Cerebras（[我们的报道见此](https://buttondown.com/ainews/archive/ainews-cerebras-inference-faster-better-and/)）。独立评估正在进行中。
- Benjamin Clavie [就 RAG 和 ColBERT/Late Interaction 发表了一场备受关注的演讲](https://x.com/bclavie/status/1831431500161806562?s=46)。
- [据报道 Strawberry 将在 2 周内发布](https://x.com/steph_palazzolo/status/1833508052835909840?s=46) 

昨天，人们还对 [Google Illuminate](https://illuminate.google.com/home) 感到兴奋，这是一种 AI 生成的关于论文和书籍的播客讨论。它目前需要排队等待，但我们在 Smol AI 也在探索做同样的事情。查看[我们的第一次尝试](https://github.com/smol-ai/temp/raw/main/combined_dialogue.mp3)！


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**Apple 的 AI 发布与行业反应**

- Apple 发布了 iOS 18 的新 AI 功能，包括视觉智能功能和 Siri 的改进。[@swyx](https://twitter.com/swyx/status/1833231875537850659) 指出，Apple 可能已经“修复了 Siri”并推出了视频理解模型，在首款 AI 手机的竞争中击败了 OpenAI。新功能包括邮件和通知摘要、个人上下文理解以及视觉搜索集成。

- 新的 iPhone 摄像头按钮被视为“黄金位置”，OpenAI/ChatGPT 和 Google 搜索被视为 Apple 视觉搜索的次要选项。[@swyx](https://twitter.com/swyx/status/1833234781221622022) 强调，摄像头现在可以将事件添加到日历中，处理过程在设备端和云端共同完成。

- 一些用户对 Apple 最近的创新表示失望。[@bindureddy](https://twitter.com/bindureddy/status/1833248496948023753) 提到，近年来没有令人信服的理由去升级 iPhone，并指出 Apple Intelligence 似乎与多年前发布的 Google Lens 类似。

**AI 模型发展与争议**

- AI 社区讨论了 Reflection 70B 模型，反应不一且充满争议。[@BorisMPower](https://twitter.com/BorisMPower/status/1833187250420453716) 表示，该模型的表现很差，与最初的宣传相反。[@corbtt](https://twitter.com/corbtt/status/1833209248236601602) 宣布对该模型的性能展开调查，并与创作者合作以复现报告的结果。

- [@DrJimFan](https://twitter.com/DrJimFan/status/1833160432833716715) 强调了在 LLM 基准测试中“刷分”的简易性，认为 MMLU 或 HumanEval 的数字不再是模型性能的可靠指标。他建议使用 LMSys Chatbot Arena 上的 ELO 积分以及来自受信任第三方的私有 LLM 评估，以获得更准确的评估。

- AI 研究社区讨论了评估方法的重要性。[@ClementDelangue](https://twitter.com/ClementDelangue/status/1833136159209263552) 宣布开源 "Lighteval"，这是 Hugging Face 内部使用的一套评估套件，旨在改进 AI 基准测试。

**AI 研究与创新**

- 一项比较 LLM 生成的研究创意与人类专家创意的研究发现，AI 生成的创意被认为更有新颖性。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1833228667641561495) 分享了论文的核心见解，指出 LLM 生成的创意获得了更高的新颖性评分，但在可行性上略逊于人类的创意。

- [@omarsar0](https://twitter.com/omarsar0/status/1833234005917065274) 讨论了一篇关于 LLM 中 In-context learning (ICL) 的新论文，强调 ICL 结合了从上下文示例中学习和检索内部知识的能力。

- [@soumithchintala](https://twitter.com/soumithchintala/status/1833177895734267987) 宣布发布 RUMs，这是一种机器人模型，在未见的全新环境中能以 90% 的准确率可靠地执行基础任务，这可能会开启更长轨迹的研究。

**AI 工具与应用**

- [@svpino](https://twitter.com/svpino/status/1833233962757722268) 分享了一个 AI 能力的例子：在几秒钟内将复杂文档转化为交互式图表，强调了该领域的快速进展。

- [@jeremyphoward](https://twitter.com/jeremyphoward/status/1833170410135056477) 宣布 FastHTML 支持 SVG，从而允许创建 Mermaid 编辑器。

- [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1833104751979794610) 讨论了 DynamiqAGI，这是一个全面的工具包，用于处理各种 GenAI 使用场景，并在个人基础设施上构建合规的 GenAI 应用程序。

**AI 伦理与安全**

- [@fchollet](https://twitter.com/fchollet/status/1833171952070238240) 认为，Machine Learning 和 AI 中过度的拟人化是导致该领域产生误解的原因。

- [@ylecun](https://twitter.com/ylecun/status/1833130597176205746) 讨论了武装平民民兵在推翻民主政府和支持暴君方面的历史作用，并将其与当前事件进行了类比。

**梗与幽默**

- [@sama](https://twitter.com/sama/status/1833227974554042815) 分享了一个幽默的比喻：“如果你给垃圾桶绑上火箭，垃圾桶仍然可以进入轨道，而且垃圾火在离开大气层时会熄灭，”暗示虽然这包含重要的见解，但最好还是发射精良的卫星。


---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. Reflection 70B：从炒作到争议**

- **Smh：Reflection 优秀得令人难以置信 —— 参考文章** ([Score: 42, Comments: 19](https://reddit.com//r/LocalLLaMA/comments/1fd2f7m/smh_reflection_was_too_good_to_be_true_reference/))：最近备受赞誉的开源 AI 模型 **Reflection 70B** 的性能受到了**质疑**，其背后的公司被**指控欺诈**。根据 [VentureBeat 的一篇文章](https://venturebeat.com/ai/new-open-source-ai-leader-reflection-70bs-performance-questioned-accused-of-fraud/)，人们对该模型报告的能力和 Benchmark 的真实性提出了担忧。这一局面引发了 AI 社区关于 **AI 模型性能声明验证**的辩论。

- **不了解这件关于 “Reflection” 的事？你不是一个人。这是我能总结出的最佳概括。** ([Score: 178, Comments: 81](https://reddit.com//r/LocalLLaMA/comments/1fd75nm/out_of_the_loop_on_this_whole_reflection_thing/))：该帖子总结了 **Reflection 70B 争议**。**Matt Shumer** 声称利用 “**Reflection Tuning**” 和 **Llama 3.1** 创建了一个革命性的 AI 模型，超越了 **ChatGPT** 等成熟模型。随后的调查显示，其公开 API 很可能是 **Claude 3.5 Sonnet** 的封装（wrapper），而发布的模型权重则是微调效果不佳的 **Llama 3 70B**，这与 Shumer 的说法相矛盾，并引发了对潜在欺诈以及与 **Glaive AI** 之间未披露利益冲突的担忧。
  - **Matt Shumer** 关于 **Reflection 70B** 模型的言论遭到了怀疑，用户质疑在声称是自己模型的情况下，如何能“意外地”链接到 **Claude**。一些人推测，在 AI 融资环境收紧的情况下，这可能是一起欺诈案件或出于绝望之举。
  - 这一事件被拿来与其他备受争议的 AI 项目（如 **Rabbit device** 和 “**Devin**”）进行比较。用户对 **OpenAI** 的怀疑也日益增加，质疑该公司关于语音和视频能力的说法，并注意到了核心员工的离职。
  - 讨论集中在 Shumer 行为背后的潜在动机，一些人将其归结为愚蠢或自恋而非恶意。另一些人则推测这可能是为了提升 **Glaive AI** 的知名度，或通过误导性声明来获取风险投资。

- **Reflection 以及 FP16 与 BF16 之间永无止境的混淆** ([Score: 42, Comments: 15](https://reddit.com//r/LocalLLaMA/comments/1fcjtpo/reflection_and_the_neverending_confusion_between/))：该帖子讨论了上传到 **Hugging Face** 的 **Reflection 70B** 模型的一个**技术问题**，其表现**不如**基准模型 **LLaMA 3.1 70B**。作者解释说，这很可能是由于从 **BF16**（LLaMA 3.1 使用）到 **FP16**（Reflection 使用）的**错误转换**造成的，由于格式不兼容（FP16 为 **5 位阶码和 10 位尾数**，而 BF16 为 **8 位阶码和 7 位尾数**），导致了严重的信息丢失。帖子强烈建议不要在神经网络中使用 **FP16**，也不要尝试将 **BF16 权重转换为 FP16**，因为这会严重降低模型性能。
  - **BF16 到 FP16 的转换**可能并不像最初建议的那样具有破坏性。**llama.cpp** 的测试显示，BF16 和 FP16 之间的 **Perplexity（困惑度）差异**比 FP16 到 Q8 的差异小 10 倍，而且 **Hugging Face** 上的大多数 **GGUF** 文件很可能都是基于 FP16 转换的。
  - 鉴于之前关于基础模型、规模和开源状态的错误陈述，讨论强调了在评估 **Shumer 的声明**时进行**贝叶斯推理（Bayesian reasoning）**的重要性。一些用户强调需要结合技术解释来考虑这些因素。
  - 几位用户指出，大多数模型**权重通常落在 [-1, 1] 范围内**，这使得 FP16 转换的影响较小。将每个权重**量化（Quantization）**到 **8 位**或更低通常只会导致微不足道或合理的精度损失，这表明 FP16 与 BF16 的差异在实践中可能微乎其微。


**主题 2. AMD 的 UDNA：统一 RDNA 和 CDNA 以挑战 CUDA**

- **[AMD 宣布统一的 UDNA GPU 架构 — 将 RDNA 和 CDNA 结合以对抗 Nvidia 的 CUDA 生态系统](https://www.tomshardware.com/pc-components/cpus/amd-announces-unified-udna-gpu-architecture-bringing-rdna-and-cdna-together-to-take-on-nvidias-cuda-ecosystem)** ([Score: 284, Comments: 90](https://reddit.com//r/LocalLLaMA/comments/1fcyap8/amd_announces_unified_udna_gpu_architecture/)): AMD 发布了全新的**统一数据中心下一代架构 (UDNA)**，结合了 **RDNA** 和 **CDNA** 的元素，为游戏和数据中心应用创建单一的 GPU 架构。这一战略举措旨在通过提供支持 **AI**、**HPC** 和**游戏**工作负载的统一平台，挑战 **Nvidia CUDA** 生态系统的统治地位，从而简化不同 GPU 类型的开发，并提高 AMD 在 GPU 市场的竞争力。

**主题 3. DeepSeek V2.5：低调发布的强力模型**

- **[DeepSeek 低调发布了 DeepSeek-Coder-V2-Instruct-0724，在 Aider LLM 排行榜上位列第二，且根据排行榜表现优于 DeepSeek V2.5](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct-0724)** ([Score: 183, Comments: 39](https://reddit.com//r/LocalLLaMA/comments/1fd6z0v/deepseek_silently_released_their/)): DeepSeek 低调发布了 **DeepSeek-Coder-V2-Instruct-0724**，这是一款全新的编程模型，在 **Aider LLM Leaderboard** 上获得了**第二名**。根据排行榜，该模型的表现超过了其前身 **DeepSeek V2.5**，标志着 DeepSeek 在编程能力上的显著提升。
  - **DeepSeek-Coder-V2** 将支持的编程语言从 **86 种扩展到 338 种**，并将上下文长度从 **16K 扩展到 128K**。该模型运行需要 **8x80GB 显卡**，目前大多数用户无法使用轻量化版本。
  - 用户讨论了 DeepSeek 通用模型和代码模型之间的版本编号混淆。新款代码模型 (**0724**) 在 **Aider LLM Leaderboard** 上优于 **DeepSeek V2.5**，但根据 [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V2.5) 的数据，V2.5 在大多数其他基准测试中击败了 0724。
  - 一些用户对更小、特定语言的模型表现出兴趣，以便于切换和交互。DeepSeek 通常在初次发布后约一个月将其模型开源。

- **所有的这些闹剧分散了我们对一个真正重要的权重开源发布的注意力：DeepSeek-V2.5** ([Score: 472, Comments: 95](https://reddit.com//r/LocalLLaMA/comments/1fclav6/all_of_this_drama_has_diverted_our_attention_from/)): 尽管 **DeepSeek-V2.5** 具有作为 **开源 GPT-4** 等效模型的潜在重要性，但其发布已被近期 AI 行业的闹剧所掩盖。这款新模型可在 [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V2.5) 上获取，据报道它结合了**通用和编程能力**，并升级了 **API 和 Web** 功能。
  - **DeepSeek-V2.5** 的评价褒贬不一，一些用户发现它在创意写作和通用任务方面**不如 Mistral-Large**。该模型运行需要 **80GB*8 GPU**，限制了其在本地使用的可及性。
  - 用户报告了运行该模型时的问题，包括 **oobabooga 中的错误**以及**缓存量化**问题。一些用户使用 **llama.cpp** 并缩减上下文长度取得了有限的成功，但性能较慢，仅为**每秒 3-5 个 token**。
  - 尽管存在疑虑，一些用户发现 DeepSeek-V2.5 在增加输出多样性和解决编程问题方面很有用。它可以在 [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V2.5) 上获取，也可以通过高性价比的 [API](https://open-tone-changer.vercel.app/) 使用。


**主题 4. 模型效率与部署的创新方法**

- **[Open Interpreter 退还了 01 Light AI 硬件设备的所有订单，转而将其开发为手机 App。App 今日上线！](https://changes.openinterpreter.com/log/01-app)** ([Score: 42, Comments: 4](https://reddit.com//r/LocalLLaMA/comments/1fczecj/open_interpreter_refunds_all_hardware_orders_for/)): Open Interpreter **取消**了其 **01 Light AI 硬件设备**的计划，转而**推出一款具有相同功能的移动应用**。这一决定似乎受到了 **Rabbit R1** 等类似 AI 硬件设备**负面评价**的影响，Open Interpreter 选择利用 **iPhone** 和 **MacBook** 等现有设备，而不是引入新的硬件。

- **[在手机上使用 LLM 生成可用的移动应用](https://v.redd.it/lrthfybr6und1)** ([评分: 60, 评论: 23](https://reddit.com//r/LocalLLaMA/comments/1fcye12/generate_usable_mobile_apps_w_llms_on_your_phone/)): 该帖子讨论了**直接在智能手机上使用大语言模型 (LLMs) 生成可用移动应用**的潜力。这一概念暗示了未来用户可以通过与移动设备上的 AI 助手进行自然语言交互来创建功能性应用程序，这可能会彻底改变应用开发和可访问性。虽然该帖子没有提供具体的实现细节，但它暗示了端侧 AI 能力和移动应用创建流程的重大进步。

- **[Deepsilicon 运行神经网络的 RAM 占用减少 5 倍，速度提升约 20 倍。他们正在为此构建软件和定制芯片](https://x.com/sdianahu/status/1833186687369023550?)** ([评分: 111, 评论: 32](https://reddit.com//r/LocalLLaMA/comments/1fdav1n/deepsilicon_runs_neural_nets_with_5x_less_ram_and/)): **Deepsilicon** 声称通过**软件**和**定制芯片**的结合，运行**神经网络**时可减少 **5 倍 RAM** 占用，并实现约 **20 倍**的性能提升。他们的方法涉及使用**三元值** (-1, 0, 1) 来**表示 Transformer 模型**，据称这消除了对**高计算成本浮点运算**的需求。帖子作者对这种方法表示怀疑，认为它看起来简单得令人难以置信。
  - **BitNet-1.58b** 的性能和针对三元值的**专用硬件**是 **Deepsilicon** 的主要动力。挑战包括扩展到更大的模型、边缘设备的经济性，以及基座模型公司是否愿意进行 1.58 bits 的训练。
  - **BitNet 论文**表明，从头开始使用 **1-bit 量化**训练模型可以匹配 **fp16 性能**，尤其是随着模型尺寸的增加。[BitNet 论文](https://arxiv.org/abs/2310.11453) 提供了关于权衡的见解。
  - 正如 [Hacker News 线程](https://news.ycombinator.com/item?id=41490905) 中所讨论的，人们对 **Y Combinator** 的资助实践和创始人的方法提出了担忧。然而，一些人看到了针对硬件和机器人应用中便携式 ML 的**边缘市场**的潜力。


**主题 5. 专用 AI 模型与技术的进展**

- **[专为创意写作打造的新系列模型，超越以往的 RP 模型 (3.8B, 8B, 12B, 70B) - ArliAI-RPMax-v1.1 系列](https://huggingface.co/ArliAI/Llama-3.1-70B-ArliAI-RPMax-v1.1)** ([评分: 141, 评论: 84](https://reddit.com//r/LocalLLaMA/comments/1fd4206/new_series_of_models_for_creative_writing_like_no/)): ArliAI-RPMax-v1.1 系列推出了**四款新模型**，用于创意写作和角色扮演 (RP)，参数规模从 **3.8B 到 70B** 不等。这些模型旨在**创意写作和角色扮演场景**中表现出色，与现有的 RP 模型相比提供了更强的能力。该系列旨在为作家和角色扮演者提供强大的工具，用于生成各种规模的富有想象力和吸引力的内容。

- **[微软的 Self-play muTuAl Reasoning (rStar) 代码已在 GitHub 上发布！](https://github.com/zhentingqi/rStar)** ([评分: 48, 评论: 4](https://reddit.com//r/LocalLLaMA/comments/1fcshuc/microsofts_selfplay_mutual_reasoning_rstar_code/)): 微软已在 **GitHub** 上发布了其 **Self-play muTuAl Reasoning (rStar)** 算法的代码。这一开源实现允许在大语言模型中进行**自我博弈相互推理**，使它们能够参与更复杂的对话和解决问题的任务。rStar 代码可以在 [https://github.com/microsoft/rstar](https://github.com/microsoft/rstar) 找到，为研究人员和开发人员提供了访问这种先进 AI 技术的途径。


- **[Mini-Omni：语言模型可以在流式思考的同时进行听与说 (微调自 Qwen2-0.5B)](https://huggingface.co/gpt-omni/mini-omni)** ([评分: 49, 评论: 7](https://reddit.com//r/LocalLLaMA/comments/1fcmcql/miniomni_language_models_can_hear_talk_while/)): **Mini-Omni** 是一款开源的**多模态大语言模型**，展示了在实时对话中处理语音输入并生成流式音频输出的能力。该模型基于**微调的 Qwen2-0.5B**，展示了在同步处理语言的同时进行听和说的端到端能力。
  - 链接了 **6 天前** 关于 **Mini-Omni** 的先前讨论线程，表明人们对该开源多模态模型的持续关注。
  - 用户表达了对展示该模型语音对语音能力的**演示视频**的渴望，强调了演示对于新 AI 模型吸引关注和验证所声称功能的重要性。

## 其他 AI Subreddit 回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 模型发布与改进**

- **OpenAI 正准备发布其新模型**：r/singularity 上的一篇幽默帖子展示了一段卡车几乎相撞的视频，隐喻了 OpenAI 的模型发布过程。[该帖子](https://www.reddit.com/r/singularity/comments/1fd8tfp/openai_preparing_to_drop_their_new_model/) 获得了超过 1000 个点赞和 110 条评论，引发了广泛关注。

- **Flux AI 模型进展**：多篇帖子讨论了 Flux AI 模型：
  - 一篇 [比较 ComfyUI 和 Forge](https://www.reddit.com/r/StableDiffusion/comments/1fcjs7i/the_current_flux_situation/) 运行 Flux 效果的帖子，突显了社区中关于不同界面的持续争论。
  - 另一篇 [帖子展示了使用 Flux LoRA 生成的 20 张图像](https://www.reddit.com/r/StableDiffusion/comments/1fd5ba2/20_breathtaking_images_generated_via_bad_dataset/)，该 LoRA 是在有限的数据集上训练的，展示了该模型即使在次优训练数据下的强大能力。

- **新的 Sora 视频发布**：[r/singularity 上的一篇帖子](https://www.reddit.com/r/singularity/comments/1fcuw21/new_sora_video_just_dropped/) 链接了一段新视频，展示了 OpenAI 的 Sora 文本转视频模型的能力。

**AI 工具与界面**

- **关于 AI 界面的辩论**：Stable Diffusion 社区正在讨论运行 AI 模型时不同界面的优劣，特别是 **ComfyUI vs. Forge**。关键点包括：
  - ComfyUI 提供更多的灵活性和控制力，但学习曲线较陡。
  - Forge 提供更用户友好的界面，并包含一些易用性改进。
  - 一些用户主张根据任务不同使用多个界面。

- **VRAM 需求**：多条评论讨论了运行 Flux 等新型 AI 模型所需的 **高 VRAM 需求**，用户们在争论如何在低端硬件上优化性能的策略。

**AI 伦理与社会影响**

- **Sam Altman 的照片**：r/singularity 上一篇 [包含 Sam Altman 照片的帖子](https://www.reddit.com/r/singularity/comments/1fcypio/altman_sam/) 引发了讨论，可能与其在 AI 发展中的角色及其社会影响有关。

**幽默与迷因 (Memes)**

- **“最有趣的一年”迷因**：[r/singularity 上的一篇幽默帖子](https://www.reddit.com/r/singularity/comments/1fd0rxd/hows_the_most_interesting_year_in_human_history/) 问道：“人类历史上最有趣的一年对你来说过得怎么样？”，反映了 AI 进步的飞速步伐。

- **AI 模型发布迷因**：关于 OpenAI 模型发布的置顶帖子利用幽默评论了围绕重大 AI 发布活动的期待和潜在问题。


---

# AI Discord 回顾

> 由 Claude 3.5 Sonnet 生成的摘要之摘要之摘要


**1. AI 模型发布与基准测试**

- **DeepSeek 2.5 以强劲规格亮相**：**[DeepSeek 2.5](https://huggingface.co/collections/deepseek-ai/deepseek-v25-66d97550c81167fc5e5e32e6)** 将 DeepSeek 2 Chat 和 Coder 2 合并为一个强大的 238B MoE，具有 **128k context length** 以及 function calling 等功能。
   - 此次发布将改变编程和聊天体验，在通用性和能力方面为未来的模型树立了更高的标准。
- **Deception 70B 占据开源模型榜首**：**Deception 70B** 模型被宣布为全球顶尖的开源模型，利用独特的 Deception-Tuning 方法来增强 LLM 的自我修正能力。
   - 该模型可在 [此处](https://bit.ly/Deception-70B) 获取，引发了关于其潜在应用以及 AI 社区对其声明有效性的讨论。
- **OpenAI 的 Strawberry 模型发布在即**：根据 [推文](https://x.com/steph_palazzolo/status/1833508052835909840?s=46) 中分享的内部消息，OpenAI 准备在未来两周内发布其新模型 **Strawberry**，并将其作为 ChatGPT 的一部分。
   - 初步印象暗示了潜在的局限性，有报告称其响应时间为 **10-20 秒**，并对其记忆整合能力表示担忧。


**2. LLM 微调与优化技术**

- **Mixed Precision Training 提升性能**：开发者报告了在使用 **cpuoffloadingOptimizer** 实现 **mixed precision training** 方面的成功，并指出在 **tokens per second (TPS)** 处理速度上有所提升。
   - 计划进行进一步测试以探索与 **FSDP+Compile+AC** 的集成，突显了在优化模型训练效率方面的持续努力。
- **Hugging Face 通过 Packing 增强训练**：Hugging Face 宣布，使用打包的指令微调示例（packed instruction tuning examples）进行训练现在已兼容 **Flash Attention 2**，吞吐量可能提升高达 **2x**。
   - 这一进展旨在简化 AI 模型的训练流程，更高效地利用计算资源。
- **MIPRO 简化 Prompt 优化**：DSPy 团队推出了 **MIPRO**，这是一款新工具，旨在优化问答系统数据集中使用的 Prompt 指令及示例。
   - MIPRO 的 Prompt 优化方法突显了业界日益关注通过精细化输入技术来提升模型性能。
  


**3. Open Source AI Developments and Collaborations**

- **GitHub 举办开源 AI 研讨会**：GitHub 将于 **9月19日** 组织一场关于 **Open Source AI** 的专题研讨会，邀请了来自 **Ollama**、**Nous Research**、**Black Forest Labs** 和 **Unsloth AI** 的演讲嘉宾。点击[此处](https://lu.ma/wbc5bx0z)可免费注册。
   - 该活动旨在讨论开源社区如何促进 AI 技术的**获取**与**民主化**，反映了协作努力在 AI 发展中日益增长的重要性。
- **LlamaIndex 探索 Agentic RAG 策略**：@seldo 最近的一次演讲探讨了 2024 年使用 [LlamaIndex](https://twitter.com/llama_index) 的 **Agentic RAG** 策略，讨论了其重要性及局限性。
   - 讨论强调了增强 **RAG** 能力的策略，展示了开源社区中检索增强生成技术的持续演进。
- **Guilherme 发布 Reasoner 数据集**：分享了一个名为 [Reasoner Dataset](https://huggingface.co/datasets/Guilherme34/Reasoner-Dataset-FULL) 的新数据集，该数据集使用 **synthetic data** 创建，专为推理任务设计。
   - 这一发布展示了 AI 训练数据开发中的创新方法，有望提升模型在逻辑推理和问题解决方面的能力。
  


**4. Multimodal AI and Tool Integrations**

- **Expand.ai 发布，变革网页数据获取方式**：Tim Suchanek 宣布推出 **[Expand.ai](https://x.com/TimSuchanek/status/1833538423954804948)**，这是一款旨在将网站转换为类型安全 API 的工具，属于 Y Combinator 的当前批次项目。
   - 该服务旨在简化网站的**数据检索**，因其简化网页数据集成的潜力而吸引了技术专家和普通用户的关注。
- **Chat AI Lite 提供多功能 AI 应用**：[Chat AI Lite](https://github.com/KevinZhang19870314/chat-ai-lite/blob/main/README_en_US.md) 作为一款**多功能 AI Web 应用**推出，涵盖了聊天、本地知识库和图像生成等多个场景。
   - 其全面的功能旨在提升各种 **AI 应用**的用户体验，展示了针对多样化用例的集成 AI 工具的发展趋势。
- **EDA-GPT 自动化数据分析**：[EDA-GPT](https://github.com/shaunthecomputerscientist/EDA-GPT) 作为一个利用 **LLM** 进行**自动化数据分析**的工具被分享，展示了在数据科学任务中的高级集成。
   - 该项目鼓励通过贡献来增强其**数据分析能力**，突显了 AI 与数据科学工具日益增长的交集。

## GPT4O (gpt-4o-2024-05-13)


**1. DeepSeek 2.5 发布**

- **DeepSeek 2.5 合并了 Chat 和 Coder 模型**：[DeepSeek 2.5](https://huggingface.co/collections/deepseek-ai/deepseek-v25-66d97550c81167fc5e5e32e6) 将 **DeepSeek 2 Chat** 和 **Coder 2** 整合为一个强大的 238B MoE 模型，具有 **128k 上下文长度**和 function calling 功能，旨在彻底改变编程和对话体验。
  - 该模型有望为未来的模型设定新标准，在编程和对话场景中均提供强劲性能。
- **关于 DeepSeek 模型端点的困惑**：用户对 [DeepSeek-Coder](https://openrouter.ai/models/deepseek/deepseek-coder) 和 [DeepSeek Chat](https://openrouter.ai/models/deepseek/deepseek-chat) 的端点感到困惑，并对 **1.75t/s** 和 **8tps** 的低吞吐量等性能问题表示担忧。
  - 模型 ID 将继续免费保留五天，以便用户平稳过渡。


**2. 模型微调挑战**

- **Unsloth 微调问题**：用户在使用 **Unsloth** 时遇到推理问题，导致微调后输出重复，特别是在改写任务中。
  - 讨论建议优化学习率、batch size 和 epoch 数量等超参数以提高性能。
- **训练中的 Loss 尖峰**：据报道，在训练 725 步后出现显著的 Loss 尖峰，Loss 达到 **20**。将 **max grad norm** 从 **1.0** 调整为 **0.3** 有助于稳定 Loss。
  - 这一问题引发了关于影响各种模型训练稳定性的潜在潜在因素的讨论。


**3. 硬件与模型性能**

- **Apple Silicon 的 GPU 规格令人印象深刻**：**M2 Max MacBook Pro** 拥有 **96GB RAM** 和实际 **72GB 显存**，能够以 **9 tokens/s** 的速度运行 **70B 模型**。
  - 这种集成实现了高效处理，展示了 Apple 在 AI 任务硬件性能方面的竞争优势。
- **AMD vs NVIDIA 性能之争**：共识认为 **AMD** 的生产力性能落后于 **NVIDIA**，特别是在 **Blender** 等应用中。
  - 由于对性能感到沮丧，用户表示打算在即将推出的 **RTX 5000** 系列发布后转向 **NVIDIA**。


**4. AI 模型创新**

- **超级预测 AI 工具发布**：一款全新的 **Superforecasting AI** 工具已发布，声称能以 **超人般的准确度** 预测结果，旨在实现预测市场的自动化。
  - 详细的演示和 [博客文章](https://www.safe.ai/blog/forecasting) 解释了其功能，引发了对其应用的兴趣。
- **OpenAI 的 Strawberry 模型即将发布**：OpenAI 正准备推出 **Strawberry 模型**，旨在增强推理和详细任务执行能力。
  - 虽然它承诺会有重大进步，但对其初始响应时间和内存处理能力的担忧依然存在。


**5. 开源 AI 发展**

- **GitHub 开源 AI 研讨小组宣布成立**：GitHub 将于 **9/19** 举办一场关于 **开源 AI** 的研讨会，小组成员来自 **Ollama**、**Nous Research**、**Black Forest Labs** 和 **Unsloth AI**。感兴趣的参与者可以在获得主办方批准后在 [此处](https://lu.ma/wbc5bx0z) 注册。
  - 该小组将探讨开源在增加 AI 技术 **可访问性** 和 **民主化** 方面的作用。
- **Hugging Face 引入 multi-packing 以提高效率**：Hugging Face 宣布打包的指令微调示例与 **Flash Attention 2** 兼容，旨在将吞吐量提高多达 **2 倍**。
  - 这一补充有可能显著简化 AI 模型训练，社区对其应用充满期待。


---

# PART 1: High level Discord summaries

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **DeepSeek 2.5 发布，规格惊人**：[DeepSeek 2.5](https://huggingface.co/collections/deepseek-ai/deepseek-v25-66d97550c81167fc5e5e32e6) 合并了 **DeepSeek 2 Chat** 和 **Coder 2**，成为一个强大的 238B MoE 模型，拥有 **128k context length** 并支持 function calling 等功能。
   - 它将改变编程和聊天体验，为未来的模型树立了更高的标准。
- **Transformers Agents 拥抱多智能体系统**：Transformers Agents 现在支持 [multi-agent systems](https://x.com/AymericRoucher/status/1831373699670315257)，通过专业化分工提升任务性能。
   - 这种方法允许高效协作，从而更好地处理复杂任务。
- **语义数据集搜索回归！**：[Semantic Dataset Search](https://huggingface.co/spaces/librarian-bots/huggingface-datasets-semantic-search) 已重新上线，提供通过 ID 或语义搜索查找相似数据集的功能。
   - 该工具提高了 Hugging Face 上数据集的可访问性，简化了研发流程。
- **韩语词干提取器与 AI 集成**：一位开发者成功创建了韩语词干提取器（lemmatizer），并正在探索利用 AI 方法进一步消除结果歧义。
   - 他们受到鼓励，利用 AI 来区分针对单个单词生成的多个词元选项。
- **支持后量子加密的 OpenSSL 3.3.2**：一位成员学会了在设备上构建包含 **Post Quantum Cryptography (PQC)** 的 **OpenSSL 3.3.2**。
   - *Lazy building FTW* 强调了安装过程的简便性。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **模型微调遇到障碍**：用户在 **Unsloth** 推理时遇到问题，微调后的模型（尤其是执行改写任务时）会出现重复输出。学习率和 batch size 等因素似乎显著影响了这些性能表现。
   - 讨论建议用户应优化超参数（包括 epoch 计数）以避免这些陷阱。
- **MLC 部署兼容性担忧**：由于特定的格式要求，MLC 部署面临挑战，这促使人们建议使用全参数微调来解决互操作性问题。量化模型可能会使这些 **MLC LLM deployments** 变得复杂。
   - 成员们强调需要针对 **Unsloth** 模型的 MLC 兼容性提供更清晰的指南。
- **Unsloth 准备支持全参数微调**：目前 **Unsloth** 专注于 **LoRA** 和 **QLoRA** 方法，大家对其即将推出的全参数微调（full-parameter fine-tuning）支持充满期待。随着项目推进，开发者的压力显而易见。
   - 成员们希望这些增强功能能够简化未来的模型部署。
- **训练中出现 Loss 激增**：一位成员指出在训练 725 步后 Loss 出现显著激增，高达 **20**。他们发现将 **max grad norm** 从 **1.0** 调整为 **0.3** 有助于稳定 Loss。
   - 这引发了关于影响各种模型训练指标的潜在底层问题的讨论。
- **WizardMath 微调突破**：**WizardMath** 在真实日记账记录上成功完成微调，经过 **13,000 多秒**的训练后，达到了 **0.1368** 的低 Loss。未来计划使用 **RAG** 来增强模型对文档引用的理解。
   - 这种方法可以显著改善簿记和会计方面的实际应用。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **讨论模型参数限制**：一位用户询问了用于训练的最小可能模型参数量，并指出 **0.5B 模型** 虽然存在但表现不佳。
   - 贡献者强调了对 **200k 和 75k 参数模型** 的尝试，并强调了数据集大小和结构对性能的影响。
- **LM Studio 支持多 GPU 配置**：确认 **LM Studio** 支持多 GPU 设置，前提是 GPU 来自同一制造商，例如使用 **两块 3060**。
   - 一位成员指出，一致的模型能产生更好的性能，提高生产力，特别是在计算密集型任务中。
- **AMD vs NVIDIA：性能之争**：共识认为 **AMD** 在生产力应用中的性能落后于 **NVIDIA**，特别是对于 **Blender** 等软件。
   - 个人经验表明，由于对性能感到沮丧，有意在即将推出的 **RTX 5000** 系列发布时转向 **NVIDIA**。
- **在有限硬件上驾驭模型性能**：讨论显示用户目标是在有限的硬件（特别是 Intel 配置）上运行 **LM Studio**，并质疑 **7B Q4KM** 等较大型模型的性能边界。
   - 建议 **16GB GPU** 在 **13B Q6 范围** 内运行，以在模型执行期间保持更流畅的操作。
- **自定义模型开发见解**：关于创建自定义模型优点的讨论浮出水面，一位用户渴望构建自己独特的堆栈，而不是使用开箱即用的解决方案。
   - 他们分享了使用 **Misty** 和 **Open-webui** 的经验，同时承认在建立有效的定制系统方面仍面临挑战。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Apple Silicon 令人印象深刻的 GPU 规格**：讨论者强调了 **M2 Max MacBook Pro** 的能力，拥有 **96GB RAM** 和实际可用于运行模型的 **72GB 显存**。
   - 这种集成允许高效处理，一位用户提到他们可以以 **9 tokens/s** 的速度运行 **70B 模型**。
- **Gemini 模型的视频分析潜力**：关于使用 **Gemini 模型** 进行视频分析，一位用户询问它是否可以总结对话并分析表情，而不仅仅是转录音频。
   - 其他人建议需要实施自定义数据集的训练以获得准确的结果，并建议利用现有的 AI 框架。
- **Llama 3 等免费模型的可用性**：用户指出 **Llama 3** 和 **GPT-2** 等模型是免费提供的，但需要不错的硬件才能有效托管。
   - 值得注意的是，运行此类本地模型需要良好的 PC 或 GPU，这提高了资源要求。
- **GPT 应用中的语音功能反馈**：一位成员创建了一个名为 **Driver's Bro** 的 GPT，它可以与 Google Maps 交互，并使用“兄弟般”的声音提供导航。
   - *遗憾的是，'shimmer' 声音表现不佳*，导致用户请求高级语音模式以增强交互。
- **训练自定义模型进行股票分析的警告**：一位成员强调，除非拥有 **全部** 历史数据（包括 **图像** 和 **图表**），否则使用 **OAI 模型** 分析股票是无效的。
   - 他们指出，为了性能目的，准确的股票分析需要使用 **API**，并提到完整的股票历史记录可以以 JSON 格式下载。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Hermes 3 转向付费模式**：标准版 **Hermes 3 405B** 将在周末前过渡到**付费模式**，提示用户切换到免费模型 `nousresearch/hermes-3-llama-3.1-405b:free` 以维持访问。
   - 用户应立即行动，因为从付费模式迁出可能会导致服务中断。
- **Eggu 数据集旨在增强多语言能力**：正在开发中的 **Eggu** 数据集目标是训练一个 **1.5GB** 的**开源多语言模型**，并集成了图像定位功能，以更好地兼容 Vision 模型。
   - 尽管该数据集旨在提供广泛的可用性，但人们对其可能被滥用的情况表示担忧。
- **DeepSeek 模型引发混淆**：关于 [DeepSeek-Coder](https://openrouter.ai/models/deepseek/deepseek-coder) 与 [DeepSeek Chat](https://openrouter.ai/models/deepseek/deepseek-chat) 的 Endpoint 存在混淆，模型 ID 将继续保持免费五天。
   - 性能方面的担忧包括某些变体的低吞吐量，仅为 **1.75t/s** 和 **8tps**。
- **Google Gemini 应对 Rate Limit 问题**：用户在使用 **Google Gemini Flash 1.5** 时反复遇到 Rate Limit 问题，尽管有用户限制但仍频繁触发上限，这引发了与 **NVIDIA Enterprise Support** 的沟通。
   - 许多人正在使用 **experimental API**，这在访问模型时带来了额外的挑战。
- **Sonnet 3.5 Beta 经历宕机**：官方承认了近期影响 **Sonnet 3.5 Beta** 的停机事件，用户最初报告 API 交互成功率较低，目前根据 **Anthropic** 的状态更新已恢复。
   - 尽管访问已恢复，但许多用户仍对该模型未来的整体稳定性持怀疑态度。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Opus API 集成引发讨论**：讨论强调了使用 **Opus API 调用**来获取“正确”版本的方法，暗示了集成技术的转变。
   - 成员们注意到相关推文揭示了该话题在工程社区中日益增长的相关性。
- **模型上传的挑战**：参与者指出 **model uploading**（模型上传）比预期的更复杂，提高了对实际障碍的认识。
   - 这反映了关于用户在有效部署模型方面面临挑战的更广泛叙述。
- **Batch Size 与性能提升**：讨论显示，较小的矩阵/Batch Size 产生更好的性能，相比大尺寸的 **1.8x** 提升，小尺寸实现了 **3x 的加速**，但优化可能需要重写 Kernel。
   - 成员们指出了 int16 和 int8 打包可能带来的损失，并对 **Quantization**（量化）误差提出了警告。
- **Triton 原子操作限制**：目前 `tl.atomic_add` 仅支持 1D Tensor，这引发了关于 2D 实现变通方案的疑问。
   - 社区正在寻求管理多维数据操作的高效替代方案。
- **关于 PyTorch Autotuning 的见解**：讨论集中在带有 Autotuning 的 **PyTorch** `inductor/dynamo` 是否可以通过缓存调优参数来增强 **Triton Kernel** 的性能。
   - 一位成员指出，利用相同的 Kernel 配置有可能加速后续运行。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 可接受使用政策澄清**：一位成员分享了 [Cohere 的可接受使用政策](https://docs.cohere.com/docs/c4ai-acceptable-use-policy)，详细列出了禁止行为，如**暴力**和**骚扰**。
   - 对话强调了**商业用途**的影响，强调模型衍生品必须遵守当地法律。
- **模型微调见解**：针对 CMD-R 模型的 **Fine-tuning** 政策提出了疑问，特别是其免费使用方面。
   - 澄清表明，**Self-hosted**（自托管）模型带有禁止商业用途的限制。
- **Temperature 设置影响输出质量**：成员建议尝试将 Temperature 设置为 **0** 或 **0.1**，以衡量输出质量的变化。
   - 讨论集中在确保输出不会与初始示例发生**剧烈**偏离。
- **创新的高级计算机视觉创意**：对 **Computer Vision** 高级项目创意的需求引发了探索其与 **LLM 项目**交叉点的建议。
   - 团队合作被视为克服项目成功挑战的关键，成员们正在集思广益协作策略。
- **在项目中使用 Google Vision API**：一个有趣的 **Pokedex 项目**利用 **Google Vision API** 和 **Cohere LLM**，旨在从图像中识别 **Pokemon** 的名称和描述。
   - 澄清指出，该 API 用于**创建图像标签**，而非学习 Embedding，并建议使用 **Kaggle** 获取数据集。

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **探索 Windows 使用情况**：一位成员询问了如何在 **Windows** 上使用该项目，反映了对该平台跨操作系统兼容性的普遍兴趣。
   - 这个问题表明用户非常渴望通过各种平台集成来实现更广泛的访问。
- **桌面版 Beta 测试访问咨询**：围绕加入 **desktop beta** 计划是否为时已晚展开了讨论，突显了用户对新功能的渴望。
   - 成员们表现出参与 Open Interpreter 套件最新进展的愿望。
- **01 App 移动端发布**：**01 App** 现已在 Android 和 iOS 上线，并计划根据用户反馈进行功能增强。
   - 敦促社区在 GitHub 上 fork 该应用以定制体验，展示了开源精神。
- **Tool Use 第 4 集发布**：题为 *'Activity Tracker and Calendar Automator - Ep 4 - Tool Use'* 的最新一集已在 [YouTube](https://www.youtube.com/watch?v=N9GCclB8rYQ) 上线，讨论了 **时间管理**。
   - 演讲者强调 **时间是我们最宝贵的资源**，激励观众有效地利用工具。
- **支持开源开发**：社区对源自 01 平台的开源项目的支持非常活跃，为新计划提供了充足的机会。
   - 成员们表达了贡献的热情，加强了围绕 AI 工具的协作环境。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular 尚无 Windows 时间表**：目前没有 **Windows 原生版本** 的时间表，因为 Modular 优先支持 **Ubuntu 和 Linux 发行版**。
   - *Modular 旨在在扩大关注范围之前避免技术债并提高产品质量，* 吸取了以往 Swift 的经验教训。
- **WSL 作为当前的 Windows 支持方案**：虽然目前没有原生的 **.exe** 版本，但 *Modular 建议使用 WSL* 作为其当前 **Windows 支持** 的范围。
   - 用户对未来的原生选项表现出兴趣，但也承认现有的局限性。
- **Mojo 瞄准 GPU 和 GStreamer 替代方案**：Mojo 被定位为 **GStreamer** 的潜在替代品，利用即将推出的 GPU 功能进行高效处理。
   - 成员们热衷于集成现代库进行直播，展示了 Mojo 在简化操作方面的潜力。
- **探索使用 DLHandle 创建绑定**：成员们讨论了使用 **DLHandle** 创建 Mojo 绑定，并参考了展示其应用的项目。
   - 像 'dustbin' 这样的项目利用 DLHandle 进行 **SDL 绑定**，为图形应用领域的开发者提供了灵感。
- **理解 Mojo 中的 Variant 类型**：强调了 Mojo 中 **Variant 类型** 在创建具有不同元素类型的列表时的实用性，以及内存方面的考虑。
   - 成员们澄清了与这些实现中的大小对齐和判别式（discriminants）行为相关的问题。



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DisTro 引发困惑**：围绕 **DisTro** 的讨论引发了对其目的和有效性的质疑，因为目前尚未发布任何代码，这可能是为了促发竞争。
   - 成员们推测其预期影响，质疑该公告是否过早。
- **AI 训练担忧加剧**：人们对基于用户满意度指标训练的 AI 模型产生了担忧，这些模型往往产生浅薄的信息而非准确的内容。
   - 有人担心这种趋势可能会损害 AI 响应的质量，尤其是在过度依赖人类反馈时。
- **OCTAV 成功发布**：一位成员分享了他们使用 Sonnet 实现 **NVIDIA 的 OCTAV** 算法的成功经验，并指出网上类似案例很少。
   - 他们推测该实现可能是从相关论文中推导出来的，展示了该模型的能力。
- **重复的响应困扰工程师**：聊天集中在 AI 倾向于生成重复输出的问题上，尤其是当用户表现出轻微犹豫时。
   - 讨论演变为像 Claude 这样的模型如何难以保持自信，往往过快地撤回解决方案。
- **AI 模型表现参差不齐**：成员们评估了 **Claude** 和 **Opus** 等平台的表现，强调了它们各自的优缺点。
   - 虽然 Claude 具有扎实的一致性策略，但与更具吸引力的 Opus 相比，它在某些情况下表现不佳。

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Mistral 和 Gemma 缺少 Tokenizer eos 选项**：一位用户提议发送一个 PR 来修复 tokenizer 的 eos 问题，指出目前的 **Mistral** 和 **Gemma** tokenizer 缺少 `add_eos` 选项。他们引用了一个[需要更新的实用程序](https://github.com/pytorch/torchtune/blob/main/torchtune/modules/tokenizers/_utils.py)。
   - 另一位成员强调，必须先实现 `add_eos` 功能才能解决此问题。
- **Eleuther_Eval recipe 默认使用 GPT-2 模型**：一位成员询问为什么 **Eleuther_Eval** recipe 总是加载 **GPT-2** 模型，得到的解释是自 `lm_eval==0.4.3` 以来这是默认设置。他们指出，可以使用 `TransformerDecoder` 工具覆盖模型，以便对其他模型进行评估。
   - 这凸显了在选择评估模型类型时需要灵活性。
- **混合精度训练（Mixed Precision Training）取得显著成效**：一位成员分享了他们使用 **cpuoffloadingOptimizer** 实现**混合精度训练**的兴奋之情，并注意到 **TPS** 有所提升。他们对如何将其与 **FSDP+Compile+AC** 集成表示不确定，建议需要进一步测试。
   - 这预示着大规模模型训练的潜在优化方向。
- **Compile 速度优于 Liger**：基准测试表明，使用 `compile(linear+CE)` 在速度和内存方面都比 **Liger** 更快。尽管 **chunkedCE** 在独立编译时表现出更高的内存节省，但整体速度较慢。
   - 这一对比强调了模型编译中速度与资源利用率之间的权衡。
- **动态 seq_len 带来优化挑战**：**torchtune** 中关于**动态 seq_len** 的担忧浮出水面，特别是由于重新自动调优（re-autotuning）对 **INT8 matmul triton kernel** 产生的影响。成员们讨论了将输入填充（padding）到 **128** 的倍数，尽管这会增加额外的填充成本。
   - 在管理填充开销的同时优化速度仍然是一个受关注的话题。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Jim Harbaugh 为 Perplexity 代言**：主教练 **Jim Harbaugh** 在最近的一份公告中表示，如果没有 **Perplexity**，一份伟大的战术册就不完整，并邀请球迷就此事[向他提问](https://x.com/perplexity_ai/status/1833173842870853896)。
   - 此次代言旨在将 Perplexity 整合到教练策略中，突显了其在体育分析中的相关性。
- **Reflection LLM 更新咨询**：一位成员询问 **Reflection LLM** 是否很快会添加到 Perplexity，表达了对功能更新的兴趣。
   - 然而，讨论中并未出现明确的答案，让社区对未来的增强功能保持好奇。
- **Perplexity Pro 奖励问题**：一位用户对与 Xfinity 合作的 **Perplexity Pro 奖励**活动表示沮丧，称其促销代码无效。
   - 社区讨论了潜在的解决方案，包括创建一个新账户以成功应用促销代码。
- **Claude 3.5 的性能困扰**：**Claude 3.5** 用户担心该模型的性能似乎有所下降，暗示尽管最近有了投入，但仍可能存在容量问题。
   - 用户报告称对设置中显示的模型版本感到困惑，表明更新缺乏透明度。
- **Nvidia Q2 财报超出基准**：据[此处](https://www.perplexity.ai/page/nvidia-beats-q2-expectations-k9CT.KnRT1uKI8OG99kdrA)报道，得益于显卡的强劲销售和 AI 领域的稳健增长，**Nvidia** 第二季度收益超出预期。
   - 分析师指出，在对 AI 解决方案需求日益增长的背景下，这一令人印象深刻的业绩巩固了 Nvidia 在科技领域的地位。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Apple Intelligence 更新即将到来**：Apple 计划在两周内发布其 **Intelligence capabilities** 的更新，重点改进 **Siri** 和其他 AI 功能。
   - 用户认为这些更新可能会解决长期存在的问题，加剧与 **OpenAI** 的竞争。
- **ColPali 模型取得进展**：ColPali 正在接受审查，展示其在各种 **AI tasks** 中的实现和功效的新幻灯片已发布。
   - ColPali 与先进训练技术的结合可能会改变当前的 AI 研究范式。
- **Superforecasting AI 精准发布**：一款新的 **Superforecasting AI** 工具已发布，展示了其以 **superhuman accuracy** 预测结果的能力。
   - 该工具旨在自动化预测市场，并辅以详细的演示和解释其功能的 [blog post](https://www.safe.ai/blog/forecasting)。
- **OpenAI 的 Strawberry 模型蓄势待发**：OpenAI 正准备推出 **Strawberry model**，旨在增强推理和详细的任务执行。
   - 虽然它承诺了重大进步，但关于初始响应时间和内存处理能力的担忧依然存在。
- **Expand.ai 发布，旨在变革网页数据访问**：Tim Suchanek 宣布推出 **Expand.ai**，这是一个将网站转换为类型安全 API 的工具，是 Y Combinator 当前批次的一部分。
   - 该服务旨在简化从网站进行的 **data retrieval**，吸引了技术专家和普通用户的兴趣。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **2024 年的 Agentic RAG 策略**：在最近的一次演讲中，**Agentic RAG** 被强调为 2024 年的关键关注点，突出了其在 [LlamaIndex](https://twitter.com/llama_index) 中的重要性。关键点包括理解 **RAG** 的必要性及其局限性，以及增强策略。
   - 听众了解了 RAG 在 LLMs 背景下的实际应用和理论方面。
- **将 LlamaIndex 与 Llama 3 集成**：成员们讨论了 [LlamaIndex 与 Llama 3](https://docs.llamaindex.ai/en/stable/examples/llm/ollama/) 的集成，并提供了运行本地 Ollama 实例的详细设置说明。
   - 分享的见解包括 LlamaIndex 的安装步骤和使用模式，包括 Colab 的命令片段，简化了模型实验。
- **使用 LlamaIndex 轻松处理 DataFrames**：一份关于使用 `PandasQueryEngine` 将自然语言查询转换为用于 Pandas 操作的 Python 代码的指南已经出现，提高了 text-to-SQL 的准确性。
   - 强调了关于任意代码执行的安全担忧，鼓励谨慎使用该工具。
- **MLflow 与 LlamaIndex 集成问题已修复**：社区讨论了最近已解决的 MLflow 和 LlamaIndex 的问题，预计将在周末发布公告。
   - 一位成员计划在一篇博客文章中记录这一集成经验，旨在帮助其他面临类似挑战的人。
- **探索 LlamaIndex 中的相似度搜索**：成员们深入研究了在 LlamaIndex 中使用 `similarity_search_with_score` 等方法进行相似度搜索，并指出了与 Langchain 的主要区别。
   - 提供了详细的示例，展示了如何根据元数据过滤检索到的文档，从而提高信息检索能力。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Deception 70B 声称是顶尖开源模型**：一项公告披露了 **Deception 70B**，声称其为全球顶尖的开源模型，利用独特的 Deception-Tuning 方法来增强 LLM 的自我修正能力。
   - 发布地址见[此处](https://bit.ly/Deception-70B)，引发了社区对其实际应用的关注。
- **OpenAI 的 Strawberry 模型即将发布**：内部人士透露 OpenAI 将在两周内发布集成到 ChatGPT 中的新模型 **Strawberry**，但初步印象显示其性能迟缓，每次响应需 **10-20 秒**。
   - 批评者对其记忆集成能力持怀疑态度，详见此 [推文](https://x.com/steph_palazzolo/status/1833508052835909840?s=46)。
- **对 Otherside AI 诈骗历史的担忧**：关于 **Otherside AI** 的讨论重新审视了其过去的诈骗行为，特别是与剽窃开源成果指控相关的自动运行计算机项目，引发了对其声明合法性的质疑。
   - 有关持续存在的问题可参考[此处](https://github.com/OthersideAI/self-operating-computer/issues/67)，突显了社区的怀疑态度。
- **AI 预测性能受到批评**：Dan Hendrycks 报告称论文 **LLMs Are Superhuman Forecasters** 的表现令人失望，指出其在新测试集上的表现显著不佳。
   - 展示该 AI 预测模型的 Demo 可在[此处](http://forecast.safe.ai)访问，重新引发了关于其预测准确性的辩论。
- **Gemini 与 Cursor 的集成引发关注**：成员们探讨了 **Gemini** 与 **Cursor** 集成的可能性，并就功能和新用例提出了疑问。
   - 表达了对 Google 最新进展的好奇，促使更多成员考虑尝试该集成。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **更好的图像生成硬件**：一位成员建议使用 **Linux** 系统配合 **24G NVIDIA** 显卡进行本地训练，以提升图像生成性能。
   - 他们还强调要检查电源的兼容性，并指出不需要进行升级。
- **Deep Dream Machine 的更廉价替代方案**：社区讨论了 **Deep Dream Machine** 的潜在替代品，建议使用 **Kling** 或 **Gen3** 进行 AI 视频创作。
   - 一位用户强调了 **Kling** 的 **66% 折扣** 促销活动，吸引了进一步的关注。
- **训练 SDXL 模型的技巧**：一位成员询问了如何使用 **Kohya Trainer** 有效训练 **SDXL** 以增强图像质量的技术。
   - 另一位成员建议细化查询以获得更有帮助的回复，并建议查看相关频道。
- **关于 CLIP 模型选择的澄清**：关于在 **DualCLIPLoader** 节点中选择合适的 **CLIP 模型**（特别是 **clip g** 和 **clip l** 之间）展开了讨论。
   - 社区成员指出 **Flux** 并非基于 **clip g** 训练，这导致了一些困惑。
- **Discord 机器人提供 AI 服务**：一位成员介绍了一个经过认证的 Discord 机器人，可以通过分享的链接提供文本生成图像和聊天辅助功能。
   - 该服务旨在直接在 Discord 内部集成强大的 AI 功能，以方便用户使用。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **GitHub 宣布开源 AI 专题研讨会**：GitHub 将于 **9/19** 举办一场关于 **Open Source AI** 的研讨会，小组成员来自 **Ollama**、**Nous Research**、**Black Forest Labs** 和 **Unsloth AI**。感兴趣的参与者在获得主办方批准后可以[在此](https://lu.ma/wbc5bx0z)免费注册。
   - 该研讨会将探讨开源在提高 AI 技术**访问权限**和**民主化**方面的作用。
- **AI 模型性能引发辩论**：最近对一个 AI 模型的测试显示，其表现虽然**令人印象深刻**，但速度却**慢了一个数量级**，这引起了对大型模型（特别是具有 **500M parameters** 的模型）的担忧。
   - 这引发了对仅基于 **sklearn** 或 **xgboost** 等库的**小模型**性能指标的怀疑。
- **隐私机器学习 (Private Machine Learning) 的努力受到关注**：关于**隐私机器学习**的讨论强调了缺乏有效的解决方案，并提到**函数加密 (functional encryption)** 和**零知识证明 (zero knowledge proofs)** 是潜在的策略，尽管已知它们速度较慢。
   - 参与者建议使用 **Docker** 创建**安全容器**，作为确保模型安全性的一种更可行的方法。
- **多方计算 (Multiparty Computation) 的复杂性讨论**：一位用户提到了**多方计算**策略，以优化云环境中的工作负载，尽管对这类方法的安全性仍存疑虑。
   - 对话指出，在**无信任环境 (trustless environments)** 中开发安全解决方案需要大量的投资。
- **实现机器学习隐私的挑战**：专家断言，在机器学习中实现**完全隐私**仍然难以捉摸且成本高昂，在与 **DARPA** 相关的敏感场景中迫切需要有效的隐私解决方案。
   - 巨大的经济激励凸显了社区对解决这一复杂问题的兴趣。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **AI 研究社区面临造假指控**：9 月 5 日，OthersideAI 的 CEO Matt Shumer 宣布在训练中型 AI 模型方面取得了所谓的突破，但后来据 [Tweet](https://x.com/shinboson/status/1832933747529834747?t=lu0kNqbEZKG5LVC30Dm7hA&s=19) 报道，该消息被证实为*虚假*。这一事件引发了对 *AI 研究诚信* 的担忧，并强调了对此类主张保持怀疑的必要性。
   - 讨论集中在 AI 研究问责制的影响上，建议必须持续保持警惕以避免类似情况。
- **Guilherme 分享 Reasoner Dataset**：一位用户分享了 [Reasoner Dataset](https://huggingface.co/datasets/Guilherme34/Reasoner-Dataset-FULL)，称其是使用针对推理任务的*合成数据 (synthetic data)* 制作的。这种方法反映了开发 AI 训练数据集的创新技术。
   - 社区成员表现出利用该数据集增强模型训练中推理能力的兴趣。
- **iChip 技术彻底改变抗生素发现**：iChip 技术能够培养以前无法培养的细菌，对抗生素发现产生了重大影响，包括 2015 年的 *teixobactin*。该技术的潜力在于其在**自然环境**中培养细菌的能力，极大地增加了药物发现的微生物候选者。
   - 专家讨论了该技术对未来制药创新的影响及其在应对抗生素耐药性方面的作用。
- **Hugging Face 引入 Multi-Packing 以提高效率**：Hugging Face 宣布打包的指令微调示例与 **Flash Attention 2** 兼容，旨在将吞吐量提高多达 **2 倍**。这一补充有可能显著简化 AI 模型训练。
   - 社区期待训练效率的提高，成员们对未来项目中可能的应用感到兴奋。
- **OpenAI Fine-Tuning API 新增 Weight 参数**：OpenAI 通过引入 **weight** 参数增强了其微调 API，详见其[文档](https://platform.openai.com/docs/guides/fine-tuning/multi-turn-chat-examples)。该参数于 **4 月**实施，允许对训练数据的影响进行更精细的控制。
   - 用户讨论了这一功能如何影响微调过程中的模型性能，从而增强训练动态。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Claude 3.5 的音频功能受到关注**：一位成员询问是否可以通过 **Langchain** 将**音频数据**传递给 **Claude 3.5** LLM 进行转录，并对其功能表示关注。
   - 另一位用户指出，虽然 Claude 3.5 支持图像，但音频功能尚不确定。
- **Langchain4j Token 计数挑战**：围绕如何使用 **langchain4j** 对输入和输出进行 **Token 计数**展开了讨论，表达了对解决方案的需求。
   - 遗憾的是，该讨论未能在 Token 计数技术方面提供具体的指导。
- **建议使用 Whisper 进行音频转录**：一位成员建议利用 **Whisper** 进行音频转录，作为 Claude 3.5 的**更快速且更便宜**的替代方案。
   - 这一建议指出了与 Claude 相比，Whisper 在转录工作流中潜在的高效率。
- **Chat AI Lite：多功能 AI Web 应用**：[Chat AI Lite](https://github.com/KevinZhang19870314/chat-ai-lite/blob/main/README_en_US.md) 是一个涵盖聊天、知识库和图像生成的 **Web 应用**，增强了各种 **AI 应用**的用户体验。
   - 其功能集展示了应对 AI 领域内多种场景的灵活性。
- **使用 EDA-GPT 进行自动化数据分析**：[EDA-GPT](https://github.com/shaunthecomputerscientist/EDA-GPT) 使用 LLM 提供**自动化数据分析**，突出了数据科学任务的高级集成。
   - 该项目鼓励通过贡献来提高其**数据分析能力**。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **情感分类器输出困惑**：一位成员询问，如果将描述更改为 **'Classify to 7 emotions'** 而不是具体细节，是否会改变情感分类器的输出。
   - *关于输出影响尚未提供明确结论*。
- **需要 AdalFlow 库的深入见解**：关于旨在自动优化 LLM 任务的 [AdalFlow](https://github.com/SylphAI-Inc/AdalFlow) 库的讨论再次升温，成员们寻求更深入的见解。
   - 一位成员承诺将审查该库，并保证在周末前分享他们的发现。
- **发现误导性的 Llama AI 模型**：一位成员透露，一个所谓的 Llama AI 模型实际上是利用复杂 Prompt 机制的最新的 **Claude** 模型。
   - 该系统通过问题解决和反思性提问策略来引导模型。
- **MIPRO 彻底改变 Prompt 优化**：新工具 **MIPRO** 通过优化数据集的指令和示例来增强 Prompt 优化。
   - 成员们探讨了 MIPRO 如何简化问答系统的 Prompt 优化，并强调了其与数据集的相关性。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **LLM 可观测性平台推荐**：一位成员正在为一个大型内部企业级 RAG 应用探索 **LLM 可观测性平台**，目前正在考虑 [W&B Weave](https://wandb.ai/weave) 和 [dbx's MLflow](https://mlflow.org/)。
   - 他们还对 **Braintrust** 和 **Langsmith** 等替代方案表示了兴趣，以增强可观测性。
- **Node.js 在使用 Anthropic's API 时遇到困难**：据报道，与 **Python** 相比，在 **Node.js** 中使用 **Anthropic's API** 性能较差，尤其是在使用 tools 时。
   - 讨论围绕其他人是否也面临类似的性能差异展开，促使人们深入研究潜在的优化方案。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **合并冲突已解决**：一位成员感谢另一位成员的帮助，成功解决了**合并冲突**，没有出现进一步的问题。
   - *非常感谢快速修复！*
- **定位测试分数**：一位成员对保存结果后如何检索特定的**测试分数**表示困惑，引发了关于最佳实践的讨论。
   - 另一位成员建议检查 **score 文件夹**，特别是 `data.csv` 文件。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **George Hotz 对 tinygrad 的热情**：讨论以分享对 **tinygrad** 的热情开始，该项目专注于深度学习框架的简洁性。
   - 聊天中充满了对这种轻量级方法对机器学习项目影响的兴奋。
- **社区参与**：一位用户通过发布挥手表情符号表达了热情，表明社区中与 **tinygrad** 相关的互动非常活跃。
   - 这种参与信号表明了人们对 George Hotz 领导的进展有着浓厚的兴趣。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **报名参加 GitHub 的 Open Source AI 专题研讨会！**：GitHub 将于 **9/19** 在其旧金山办公室举办一场免费的 [Open Source AI 专题研讨会](https://lu.ma/wbc5bx0z)，重点关注 AI 的 **accessibility**（可访问性）和 **responsibility**（责任）。
   - 来自 **Ollama**、**Nous Research**、**Black Forest Labs** 和 **Unsloth AI** 的嘉宾将讨论 **AI 技术的民主化**。
- **抓紧时间，活动报名需经审核！**：参与者需要尽早注册，因为活动报名需经主办方批准，以确保能在这个备受关注的研讨会中获得名额。
   - 与会者将深入了解开源社区如何推动 AI 领域的**创新**。



---


**Alignment Lab AI Discord** 暂无新消息。如果该服务器长时间没有动态，请告知我们，我们将将其移除。


---


**Mozilla AI Discord** 暂无新消息。如果该服务器长时间没有动态，请告知我们，我们将将其移除。


---


**DiscoResearch Discord** 暂无新消息。如果该服务器长时间没有动态，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 暂无新消息。如果该服务器长时间没有动态，请告知我们，我们将将其移除。


---

# 第二部分：按频道划分的详细摘要和链接


{% if medium == 'web' %}

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1283141072914219122)** (1 条消息): 

> - `DeepSeek 2.5`
> - `Yi Coder 1.5B+9B`
> - `OLMoE`
> - `Multi-agent systems support`
> - `Semantic Dataset Search` 


- **DeepSeek 2.5 发布，规格惊人**：[DeepSeek 2.5](https://huggingface.co/collections/deepseek-ai/deepseek-v25-66d97550c81167fc5e5e32e6) 将 **DeepSeek 2 Chat** 和 **Coder 2** 合并为一个强大的 238B MoE 模型，具有 **128k 上下文长度**以及 function calling 等高级功能。
   - 它旨在彻底改变编程和聊天体验，为未来的模型设定了高标准。
- **Transformers Agents 支持多智能体系统 (Multi-Agent Systems)**：Transformers Agents 现在支持 [多智能体系统](https://x.com/AymericRoucher/status/1831373699670315257)，通过专业化分工提高任务性能。
   - 这种新方法允许 Agent 之间进行高效协作，从而更轻松地处理复杂任务。
- **语义数据集搜索 (Semantic Dataset Search) 回归！**：[语义数据集搜索](https://huggingface.co/spaces/librarian-bots/huggingface-datasets-semantic-search) 已重新上线，支持通过 ID 查找相似数据集或进行语义搜索。
   - 该工具增强了 Hugging Face 上数据集的可访问性和可用性，使研发更加高效。
- **OLMoE 拥有海量训练数据**：[OLMoE](https://huggingface.co/collections/allenai/olmoe-66cf678c047657a30c8cd3da) 是一个 6.9B 的 MoE 模型，在惊人的 **5T tokens** 上进行了训练，且完全开源以促进协作。
   - 其架构和广泛的训练数据预计将在各种应用中表现出强劲的性能。
- **全新的图像背景去除工具**：一款新的 [图像背景去除工具](https://x.com/xenovacom/status/1828116951186710795) 利用浏览器内推理，使用最新的 Transformers.js 快速且私密地转换图像。
   - 用户可以免费享受快速、高质量的结果，同时确保数据隐私。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/TheZachMueller/status/1831002292440469519)">Zach Mueller (@TheZachMueller) 的推文</a>：今天 @huggingface accelerate 0.34.0 正式发布，这是一个内容丰富的版本！从 `torchpippy` 更新到可恢复的 dataloader 支持，以及翻新的 TransformerEngine 支持，有大量内容...</li><li><a href="https://x.com/AymericRoucher/status/1831373699670315257)!">Aymeric (@AymericRoucher) 的推文</a>：🥳 Transformers Agents 现在支持多智能体系统！多智能体系统最初由微软的 Autogen 框架引入。它简单来说就是让多个 Agent 协作解决...</li><li><a href="https://x.com/vllm_project/status/1833257997814096245)">vLLM (@vllm_project) 的推文</a>：我们很高兴看到 @vllm_project 成为 @huggingface hub 本地应用的一个选项！它附带了简单的代码片段，可以快速测试模型。</li><li><a href="https://x.com/xenovacom/status/1828116951186710795)">Xenova (@xenovacom) 的推文</a>：最近关于图像背景去除的最佳方法有很多争论。这是我的尝试：- 使用 🤗 Transformers.js 进行浏览器内推理 - WebGPU 加速（快！）- 成本 $0 ...</li><li><a href="https://x.com/multimodalart/status/1833459429557088314)">apolinario 🌐 (@multimodalart) 的推文</a>：现在在 @huggingface 上为你的 LoRA 画廊添加图片非常简单 🤯 🪄 ① 使用 Widget 生成图片 🖼️ ② 点击 "Add to model card gallery" 🔥</li><li><a href="https://x.com/vanstriendaniel/status/1833188523207496058)">Daniel van Strien (@vanstriendaniel) 的推文</a>：@huggingface 的语义数据集搜索回归了！通过 ID 查找相似数据集或对数据集卡片进行语义搜索。快来试试：https://huggingface.co/spaces/librarian-bots/hug...</li><li><a href="https://x.com/gabrielmbmb_/status/1832078861296668748)">Gabriel Martín Blázquez (@gabrielmbmb_) 的推文</a>：昨天发布了 Reflection 70B，这是一个使用 Reflection-Tuning 微调的模型，在 MMLU 等多个基准测试中取得了令人印象深刻的分数。用于微调的数据集并非...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1282777583452426316)** (455 条消息🔥🔥🔥): 

> - `Whisper 模型使用`
> - `韩语词形还原器 (Lemmatizer) 开发`
> - `模型结构化输出`
> - `量化与数据集校准`
> - `Hugging Face 社区动态` 


- **使用 Whisper 进行音频转录**：一位用户询问关于使用 [Whisper 模型](https://huggingface.co/openai/whisper) 进行音频转录的问题，寻求在本地运行该模型的指导。
   - 他们讨论了使用该模型可能存在的额度限制，并表示尽管面临安装挑战，仍有兴趣学习如何在本地运行模型。
- **韩语词形还原器与 AI 的集成**：一位开发者分享了他们成功创建韩语词形还原器 (lemmatizer) 的经历，并就如何利用 AI 进一步优化结果（以解决固有的歧义性）寻求建议。
   - 社区鼓励他们尝试使用 AI 来区分单次生成的多个词元 (lemmas)。
- **来自 AI 模型的结构化输出**：用户报告了在没有提供特定编程上下文的 Prompt 时，模型给出的各种输出，导致出现了如 dataclasses 而非相关代码的意外响应。
   - 其中一位展示了模型如何为“在公园散步”生成文本文件，展示了其在未明确要求编程相关响应时产生结构化输出的能力。
- **探索使用 AWQ 进行量化**：一位用户讨论了开始使用 AWQ 对模型进行量化，并表示需要数据集来辅助校准。
   - 他们寻求关于合适数据源的建议，以改进其量化工作。
- **AI 模型对比**：对话转向评估某些 AI 模型的质量，特别是将特定模型的性能与 GPT-4 进行对比。
   - 用户分享了他们在各种模型上的体验，包括与结构化输出的交互以及不同配置的局限性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/openai/whisper">Whisper - openai 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://x.com/AdeenaY8/status/1833460689400123452">Adina Yakup (@AdeenaY8) 的推文</a>：很高兴在今天的论文列表 http://hf.co/papers 中看到来自乌兹别克斯坦社区的论文🥰大声感谢 @MamasaidovM @murodbeck 对开源社区的贡献🙌 htt...</li><li><a href="https://huggingface.co/blog/codeparrot">从零开始训练 CodeParrot 🦜</a>：未找到描述</li><li><a href="https://tenor.com/view/tim-and-eric-what-confused-absurd-tim-heidecker-gif-18146476">Tim And Eric What GIF - Tim And Eric What Confused - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/shafire/talktoaiZERO">shafire/talktoaiZERO · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/TheVixhal/Resume-Roaster">LegalMIndAI - TheVixhal 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/openai/whisper-large-v3">openai/whisper-large-v3 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/nroggendorff/objaverse">nroggendorff/objaverse · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1282922591866327081)** (2 条消息): 

> - `OpenSSL 3.3.2`
> - `后量子密码学 (Post Quantum Cryptography)`
> - `TLS 握手 (TLS Handshakes)` 


- **构建带有 PQC 的 OpenSSL 3.3.2**：今天我学习了如何在设备上构建带有 **Post Quantum Cryptography (PQC)** 的 **OpenSSL 3.3.2**。
   - *Lazy building FTW* 强调了该过程的便捷性。
- **针对 OpenSSL 的 QompaSSL 更新**：有一个关于 **OpenSSL 3.3.2** 的 **QompaSSL** 重要更新，强调了 **TLS Handshakes** 对于安全通信的重要性。
   - 该更新重申了 **TLS Handshakes** 在安全通信中的重要地位。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/)** (1 条消息): 

cakiki: 它是开源的吗？
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1282779237153247303)** (21 条消息🔥): 

> - `Synthetic Data Creation with GANs` (使用 GANs 创建合成数据)
> - `Quantized GraphRAG Systems` (量化 GraphRAG 系统)
> - `Local-First Vector Database` (本地优先的向量数据库)
> - `Resume Roaster Project` (Resume Roaster 项目)
> - `LLM Responses and Formatting` (LLM 响应与格式化)


- **使用 GANs 协作探索合成数据**：一位成员询问是否可以将模型用作 GAN 来创建用于股票布局的**合成数据 (synthetic data)**，引发了关于微调方法和适当判别器 (discriminator) 重要性的讨论。
   - *如果你能够对其进行微调*，这对于将其作为生成器至关重要，建议需要生成一个**数据集 (data set)** 以进行有效的 GAN 训练。
- **量化 GraphRAG 系统的挑战**：大家达成共识，认为使用量化模型的 **graph rag** 方法产生的结果**非常混乱 (messy results)**，另一位成员建议全量模型可能会产生更好的结果。
   - 提到了对潜在改进的探索和提高准确性的建议，表明需要**更好的数据**处理。
- **构建本地优先的向量数据库**：一位成员分享了一篇关于使用 RxDB 和 transformers.js 创建**本地优先向量数据库 (local-first vector database)** 的文章，强调了零网络延迟和离线功能等优势。
   - 这种方法允许直接在浏览器中进行**语义搜索 (semantic searches)**，使其适用于离线优先的应用，同时优化性能。
- **Resume Roaster 项目启动**：介绍了一个名为 **Resume Roaster** 的有趣项目，邀请成员查看其创新的**简历生成 (resume generation)** 方法。
   - 该项目已提供链接以供进一步探索，展示了用户在职业发展实际应用中的参与度。
- **通过格式化技巧改进 LLM 响应**：提出了通过将输出格式化为 JSON 或 YAML 来增强 **LLMs** 响应的建议，提供额外的元数据以提高清晰度。
   - 重点在于确保 Prompt 的设计能够提取更令人满意的响应，并反思了量化模型的预期局限性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/TheVixhal/Resume-Roaster">LegalMIndAI - 由 TheVixhal 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://rxdb.info/articles/javascript-vector-database.html">JavaScript Vector Database | RxDB - JavaScript Database</a>: 本地优先革命已经到来，正在改变我们构建应用的方式！想象一下，你的应用数据就存在用户的设备上，随时可用，即使没有网络...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1282973603612327957)** (1 条消息): 

> - `Instruction-tuned Models` (指令微调模型)
> - `DPO/RLHF-tuning` (DPO/RLHF 微调)
> - `LLaMA 3.1` (LLaMA 3.1)
> - `Fine-tuning Guardrails` (微调安全护栏) 


- **探索指令微调模型**：一位成员思考了使用已经过指令微调和 DPO/RLHF 微调的模型来禁用其嵌入的 Guardrails 的想法，认为这有可能增强认知能力。
   - *微调方法是否能让 LLaMA 3.1 等模型在没有这些 Guardrails 的情况下更好地运行？* 这可能会带来更通用的 AI 应用。
- **LLaMA 3.1 的可能性**：讨论强调了像 **LLaMA 3.1** 这样的模型很可能是在广泛的指令微调和偏好语料库上训练的。
   - 这一背景引发了关于如果通过微调使其在更少限制下运行，其能力将如何变化的问题。


  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1283023801507905548)** (12 messages🔥): 

> - `PDF 文档分析`
> - `ColPali Embeddings 问题`
> - `Amazon ML Challenge 2023`
> - `基于 AI 的韩语词干提取器 (Lemmatizer)`
> - `从零开始构建 NLP 模型` 


- **创建 PDF 整理程序**：一名成员正在寻求开发一个程序的建议，用于分析和分类收集的杂乱 PDF，考虑使用 Lbl2Vec 进行向量嵌入 (vector embedding)。
   - 他们还提到发现了 [llamaFS](https://link.to.llamaFS)，这是一个处理类似任务的程序，但依赖于多个外部 API。
- **ColPali 的 Embedding 格式挑战**：讨论了 ColPali 的嵌入输出形状为 **[1030,128]**，这与大多数向量数据库（主要是 Chroma）所期望的一维格式冲突。
   - 该成员正在探索池化操作 (pooling operation) 或其他解决方案是否可以纠正这种形状不一致。
- **应对 Amazon ML 2023 Challenge 数据集**：一名成员正在寻求指导，利用 Amazon ML Challenge 数据集中的文本数据（包括产品标题、描述和属性）来预测产品长度。
   - 他们提供了数据集链接，强调了准确估计产品长度对于包装和客户评估的重要性。
- **将 AI 集成到韩语词干提取器 (Lemmatizer) 中**：一名成员正在寻找基于 AI 的方法，以解决其韩语词干提取器中的歧义问题，该工具是他们在过去一年中在没有 AI 的情况下开发的。
   - 他们寻求关于如何有效利用 AI 从多个可能性中确定最准确词干的建议。
- **寻求帮助：使用 PyTorch 构建 NLP 模型**：一名成员寻求协助，希望使用 PyTorch 从头开始构建自己的 NLP 模型，并对输入和输出参数表示困惑。
   - 他们提到之前有计算机视觉 (Computer Vision) 方面的经验，但这是第一次涉足 NLP 领域。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.kaggle.com">Kaggle: 您的机器学习和数据科学社区</a>：Kaggle 是全球最大的数据科学社区，拥有强大的工具和资源来帮助您实现数据科学目标。</li><li><a href="https://www.kaggle.com/datasets/ashisparida/amazon-ml-challenge-2023">Amazon ML Challenge 2023</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1282811953529749607)** (5 messages): 

> - `Diffusers 与潜空间 (Latent Space) 操作`
> - `使用 Diffusers 进行图生图 (Image-to-Image) 生成`
> - `使用 CLIP 文本嵌入 (Text Embeddings)`
> - `对潜像 (Latent Images) 进行去噪` 


- **关于操作潜空间的困惑**：一位新用户对如何使用 **CLIP** 文本嵌入来操作图像的潜空间表示不确定，并质疑自动编码器 (autoencoder) 输出与 CLIP 嵌入之间的尺寸不匹配问题。
   - 他们尝试了这种操作，但对结果不满意，寻求对预期结果的澄清。
- **使用文本嵌入对 Latent 进行去噪**：一名成员建议使用现有的文本嵌入逐渐对潜像进行去噪，以改善结果。
   - 他们建议在原始 Latent 和去噪版本之间进行插值，作为一种潜在的方法。
- **图生图 (Image-to-Image) 生成过程详解**：另一名成员详细解释了图生图的过程，强调了初始图像如何编码到潜空间以及如何添加噪声。
   - 他们分享了一个 [Hugging Face 文档链接](https://huggingface.co/docs/diffusers/en/using-diffusers/img2img)，介绍如何使用 **AutoPipelineForImage2Image** 类来简化此过程。
- **对修改潜空间感兴趣**：原提问者对那些通过修改潜空间进行图生图生成的 Space 所展示的惊人效果表现出浓厚兴趣。
   - 这种兴趣激发了他们进一步探索潜空间操作的动力，特别是通过**图生图生成**方法。



**提到的链接**：<a href="https://huggingface.co/docs/diffusers/en/using-diffusers/img2img">Image-to-image</a>：未找到描述

  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1282777591459348663)** (333 条消息🔥🔥): 

> - `Model Fine-Tuning` (模型微调)
> - `MLC Deployment Issues` (MLC 部署问题)
> - `Unsloth Updates` (Unsloth 更新)
> - `Inference Problems` (推理问题)
> - `Llama-3.1-SuperNova-Lite` 


- **模型微调的挑战**：用户在 Unsloth 中微调模型后，遇到了推理结果重复的问题，特别是在改写（paraphrasing）任务中。
   - 尽管训练设置成功，但学习率（learning rate）、Batch Size 和 Epoch 数量等多种因素都可能影响性能表现。
- **MLC 部署问题**：由于特定的格式要求，出现了关于 MLC 兼容性的担忧，用户建议需要进行全参数微调（full parameter fine-tuning）来解决这些问题。
   - 讨论表明，使用量化模型可能会使与 MLC LLM 部署的互操作性变得复杂。
- **Unsloth 开发更新**：人们期待 Unsloth 即将支持全参数微调，预计在今年或明年推出。
   - 目前的重点仍然是 LoRA 和 QLoRA 方法，随着项目接近完成，开发者的压力也很大。
- **模型的推理问题**：据报道，一些 Notebook 中的推理输出非常相似且缺乏变化，这促使用户探索 Temperature 设置和 Tokenizer 调整。
   - 结果表明，用户可能需要调整配置以获得更好的性能，包括评估学习率和训练后的合并（merging）方法。
- **Llama-3.1-SuperNova-Lite 介绍**：由 Arcee.ai 开发的新模型 Llama-3.1-SuperNova-Lite 提供了一个具有高性能和紧凑设计的 8B 参数架构。
   - 该模型利用蒸馏（distilled）训练方法和指令数据集，旨在为组织提供高效结果的同时保持资源高效利用。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1mvwsIQWDs2EdZxZQF9pRGnnOvE86MVvR?usp=sharing#scrollTo=2eSvM9zX_2d3">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/17d3U-CAIwzmbDRqbZ9NnpHxCkmXB6LZ0?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://llm.mlc.ai/docs/compilation/convert_weights.html#clone-from-hf-and-convert-weight">Convert Model Weights &mdash; mlc-llm 0.1.0 documentation</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/llama-3-8b">unsloth/llama-3-8b · Hugging Face</a>: 未找到描述</li><li><a href="https://rentry.co/cgx2w8pk">from trl import SFTTrainer</a>: from transformers import TrainingArguments, DataCollatorForSeq2Seq from unsloth import is_bfloat16_supported import os os.environ[&quot;WANDB_PROJECT&quot;] = &quot;spellbound&quot;  # name your W&amp...</li><li><a href="https://huggingface.co/arcee-ai/Llama-3.1-SuperNova-Lite">arcee-ai/Llama-3.1-SuperNova-Lite · Hugging Face</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/troubleshooting/errors#evaluation-loop-also-oom-or-crashing">Errors | Unsloth Documentation</a>: 要修复设置中的任何错误，请参阅下文：</li><li><a href="https://github.com/unslothai/unsloth/issues/689">Does unsloth have a script for full parameter fine-tuning? · Issue #689 · unslothai/unsloth</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1282836410864373761)** (7 条消息): 

> - `Kaggle Housing Price Challenge`
> - `Unsloth Fine-tuned Model Deployment`
> - `MOE Model Performance` 


- **Kaggle Housing Price Challenge 的相似性**：一位用户指出需要将数据转换为向量或数字以进行机器学习，并引用了与 [Kaggle Housing Price Challenge](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) 目标相似的案例。
   - 讨论强调了数据准备是机器学习工作流中至关重要的一步。
- **寻求在 AWS 上部署 Unsloth 模型的指导**：一位用户请求在 AWS SageMaker 上部署 **Unsloth fine-tuned models** 的帮助，并指出由于缺少组件，这些模型无法像**其他模型**那样正常部署。
   - 他们寻求任何成功完成此类部署的人员的经验或指导。
- **Jamba 1.5 mini 超越 Llama 3.1**：一位成员分享了 **Jamba 1.5 mini** 模型处理 **50 个并发请求**的速度比 **Llama 3.1 70B** 处理 **10 个并发请求**还要快，这让社区感到惊讶。
   - 他们提供了参考数据，指出 **Llama** 平均每个请求耗时 **25 秒**，展示了 **MOE** 模型的**效率**。



**提到的链接**：<a href="https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques">House Prices - Advanced Regression Techniques | Kaggle</a>：未找到描述

  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1282864342441918475)** (27 条消息🔥): 

> - `Full Fine-Tuning Inquiry`
> - `Loss Spiking Issue`
> - `Flash Attention 2 Usage`
> - `Optimal GPU Size for LLAMA 3.1`
> - `Metric Computation Support in SFTTrainer` 


- **Unsloth 对 Full Fine-Tuning 的支持**：一位成员询问 Unsloth 是否支持 full fine-tuning，另一位成员给出了否定回答。
   - 讨论中未提供详细解释或设置说明。
- **训练期间的 Loss Spiking（Loss 飙升）**：一位成员报告称其训练 loss 在 **725 steps** 之前一直在下降，随后飙升至 **20** 以上。
   - 建议包括将 **max grad norm** 从 **1.0** 调整为 **0.3**，这似乎稳定了 loss。
- **在 Gemma 2 中使用 Flash Attention 2**：一位用户询问在 **Gemma 2 models** 上成功使用 **Flash Attention 2** 的情况，并对 vRAM 使用量表示担忧。
   - 尽管为 Flash Attention 配置了两个环境，但他们发现内存消耗没有差异。
- **LLAMA 3.1 的最佳 GPU 规格**：一位成员就微调 **LLAMA 3.1 70B** 的最佳 GPU 显存大小寻求建议，重点关注 **40GB vs 80GB** 选项。
   - 回复指出，为了进行有效的微调，更倾向于使用至少拥有 **80GB vRAM** 的 **A100 或 H100 GPUs**。
- **SFTTrainer 指标计算错误**：一位用户在训练期间遇到了 **NotImplementedError**，这与复制没有数据的张量有关。
   - 该错误引发了关于 **SFTTrainer** 框架底层问题的疑问，特别是在指标计算方面。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1282995569249357846)** (9 条消息🔥): 

> - `WizardMath fine-tuning`
> - `Collaboration on RAG`
> - `Experience in machine learning`
> - `Mechanical engineering background` 


- **WizardMath 的成功微调**：一位成员在复式记账真实日记账记录上微调了 **WizardMath**，在训练 **13007.8255 秒**后达到了 **0.1368** 的显著 loss。
   - 该成员计划在微调后实施 **RAG**，以增强模型对 alpaca 格式数据集中使用的文档代号的理解。
- **寻求合作目标的明确性**：潜在的合作者表示需要明确关于使用 **RAG** 的确切问题陈述，并建议微调 embedding 模型可能会更有益。
   - 双方似乎都有兴趣通过这次协作努力来解决日常任务。
- **机械工程师转型 GenAI**：一位成员分享了他们的背景：**机械工程师**，曾在优秀期刊发表过论文，并有一年 **GenAI** 角色的全职工作经验。
   - 这一经验为计划中的机器学习任务协作增添了坚实的学术基础。
- **对会计逻辑的幽默认可**：一位成员幽默地预见到，**会计逻辑**的挑战可能会让他那身为结构工程师的兄弟感到有趣，这反映了对该领域的普遍看法。
   - 这种交流突显了成员们对不同工程学科之间轻松幽默的理解。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1282787007415849010)** (81 messages🔥🔥): 

> - `模型训练参数`
> - `LM Studio 中的多 GPU 支持`
> - `旧版本的可用性`
> - `运行 AI 的最佳模型`
> - `在受限硬件上的性能` 


- **关于模型参数限制的讨论**：一位用户询问了用于训练的最小可能模型参数量，其他人补充说 **0.5B 模型虽然可用但表现不佳**。
   - 用户讨论了尝试 **200k 和 75k 参数模型**，并指出数据集的大小和结构会显著影响性能。
- **LM Studio 的多 GPU 能力**：确认了 LM Studio 支持 **多 GPU 配置**，前提是 GPU 来自同一制造商，例如两张 Nvidia 显卡。
   - 有人提到，使用相同的型号（如 **两张 3060**）会比使用不同型号获得更好的性能。
- **关于 LM Studio 版本管理的问题**：一位用户对无法获取旧版本 LM Studio 表示担忧，询问如何追踪特定版本的 URL。
   - 有人建议，修改当前版本的 URL 可能会获得旧版本的访问权限，尽管目前没有正式的仓库。
- **针对特定任务的最佳模型**：用户分享了寻找用于文本分类和神秘学最佳模型的经验，表达了对全面模型的需求。
   - 推荐包括 **Mistral Trismegistus 7B Q8_0**，尽管反馈结果褒贬不一，欢迎进一步的替代方案。
- **在受限硬件上运行 AI**：用户讨论了在受限硬件（特别是 Intel 配置）上运行 LM Studio 的可行性，以及像 **7B Q4KM** 这样较大模型的性能。
   - 建议对于 16GB GPU，应保持在 **13B Q6 范围** 内，以确保运行更流畅并获得适当的模型支持。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://wandb.ai/mostafaibrahim17/ml-articles/reports/A-Deep-Dive-Into-Learning-Curves-in-Machine-Learning--Vmlldzo0NjA1ODY0#what-are-accuracy-and-loss-curves?-">A Deep Dive Into Learning Curves in Machine Learning</a>: 通过我们的准确率和损失曲线指南更好地理解机器学习。我们解释了它们的区别、如何阅读它们以及它们为什么重要。</li><li><a href="https://wandb.ai/mostafaibrahim17/ml-articles/reports/A-Dee">mostafaibrahim17</a>: Weights & Biases，机器学习开发者工具</li><li><a href="https://www.tomshardware.com/news/the-end-of-sli-as-we-know-it-nvidia-reveals-new-model">The End of SLI As We Know It: Nvidia Reveals New Model</a>: 购买两张 RTX 3090 值得吗？
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1282797998304133160)** (93 条消息🔥🔥): 

> - `GPU capabilities` (GPU 能力)
> - `AMD vs NVIDIA performance` (AMD 与 NVIDIA 性能对比)
> - `Mistral model operations` (Mistral 模型运行)
> - `Surface Studio Pro upgrades` (Surface Studio Pro 升级)
> - `Building custom models` (构建自定义模型)


- **针对不同模型的 GPU 能力**：用户讨论了 **GPU** 运行不同规模模型的能力，有人指出 **12GB GPU** 可以高效处理高达 **13B Q4** 的模型，展示了其相对于 **8B Q8** 模型在易用性上的优势。
   - 成员们指出了基于 GPU 显存的技术限制，以及在生产力软件兼容性方面严重向 **NVIDIA** 倾斜的现状。
- **AMD 与 NVIDIA 在生产力方面的性能对比**：大家达成共识，认为 **AMD** 的生产力性能逊色于 **NVIDIA**，特别是在 **Blender** 和 **Adobe** 等偏向后者的软件中。
   - 成员们分享了使用 **AMD 产品** 的波折经历，并表示打算在 **RTX 5000** 系列发布前后转向 **NVIDIA**。
- **优化 Mistral 模型运行**：讨论了在没有 GPU 加速和有效优化的情况下，在基于 CPU 的推理上实现 **Mistral-7B-Instruct-v0.3** 模型最佳性能的困难。
   - 讨论强调了无论 CPU 性能如何，CPU 推理都可能比 **4090** 等 GPU 驱动的系统慢得多，这反映了对 **context window management** 的需求。
- **探索 Surface Studio Pro 升级**：一位用户对 **Surface Studio Pro** 无法升级硬件的限制表示沮丧，正在考虑 **eGPU** 或 **SSD** 改进等方案。
   - 该用户希望能获得关于这些升级方案的温和指导，询问在非技术背景下如何提升性能。
- **自定义模型开发讨论**：参与者讨论了开发自定义模型的优点，一位用户表示希望为他们的语言模型构建独特的堆栈，而不是满足于现成的解决方案。
   - 他们分享了使用 **Misty** 和 **Open-webui** 的经验，同时也承认在寻找满足特定需求的系统时面临的挑战。



**提到的链接**：<a href="https://www.tomshardware.com/pc-components/gpus/nvidia-bans-using-translation-layers-for-cuda-software-to-run-on-other-chips-new-restriction-apparently-targets-zluda-and-some-chinese-gpu-makers">Nvidia bans using translation layers for CUDA software &mdash; previously the prohibition was only listed in the online EULA, now included in installed files [Updated]</a>：Nvidia 禁止将翻译层用于 CUDA 软件 —— 此前该禁令仅列在在线 EULA 中，现在已包含在安装文件中 [已更新]：翻译层成为众矢之的。

  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1282777563424362548)** (89 条消息🔥🔥): 

> - `Apple Silicon GPU 能力`
> - `Gemini 模型功能`
> - `Llama 3 与免费模型`
> - `视频分析 AI 项目`
> - `OpenCV 局限性` 


- **Apple Silicon 令人印象深刻的 GPU 规格**：讨论者强调了 **M2 Max MacBook Pro** 的能力，其拥有 **96GB RAM**，在运行模型时可有效利用 **72GB 显存**。
   - 这种集成实现了高效处理，一位用户提到他们能以 **9 tokens/s** 的速度运行 **70B 模型**。
- **Gemini 模型的视频分析潜力**：关于使用 **Gemini 模型** 进行视频分析，一位用户询问它是否能总结对话并分析表情，而不仅仅是转录音频。
   - 其他人建议需要对自定义数据集进行训练以获得准确结果，并推荐利用现有的 AI 框架。
- **Llama 3 等免费模型的可用性**：用户指出，像 **Llama 3** 和 **GPT-2** 这样的模型是免费提供的，但需要不错的硬件才能有效托管。
   - 据指出，运行此类本地模型需要良好的 PC 或 GPU，这提高了资源要求。
- **探索特定用例的 AI 解决方案**：一位用户讨论了他们旨在分析视频以跟踪球员位置的项目，并表达了使用 **OpenCV** 难以获得良好结果的挑战。
   - 另一位用户分享了一个成功实现体育运动中球员检测和跟踪的开源项目，这可能对原帖作者有所帮助。
- **使用 Yolo 训练自定义目标检测模型**：对话转向为特定目标检测任务训练自定义模型，用户强调了训练数据量的重要性。
   - 建议根据视频输入中被跟踪物体的特定程度，可能需要 **10 到 1000 个示例**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/skalskip92/status/1816162584049168389">来自 SkalskiP (@skalskip92) 的推文</a>: 足球 AI 代码终于开源了 - 球员检测与跟踪 - 球队聚类 - 相机校准。我还需要完善 README；别因为这个评判我。代码：https://github.com/ro...</li><li><a href="https://x.com/apples_jimmy/status/1833337411788804595">来自 Jimmy Apples 🍎/acc (@apples_jimmy) 的推文</a>: 本周我从耐心的洞穴中迈出了一小步。引用 Jimmy Apples 🍎/acc (@apples_jimmy)：西线无战事，弥漫着沉重的精神分裂能量。我准备好再次受伤害了，让...
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1282904926145478708)** (8 条消息🔥): 

> - `Driver's Bro GPT`
> - `GPT 中的语音功能`
> - `Memory 功能反馈`
> - `使用 DALLE-3`
> - `通过 ChatGPT 创建图像` 


- **Driver's Bro GPT 需要更好的语音**：一位成员创建了一个名为 **Driver's Bro** 的 GPT，它与 Google Maps 接口，并使用一种“兄弟式”的口吻提供路线指引，帮助用户在驾驶时排解情绪。
   - *不幸的是，“shimmer”语音效果不佳*，导致用户请求更高级的语音模式来增强交互。
- **请求在 GPT 中加入男声**：用户强烈要求在 GPT 中至少提供一个**男性语音**选项，因为目前的 **shimmer 语音** 达不到预期。
   - 这种情绪表达了现有选项不足以满足用户偏好。
- **Memory 功能反馈**：一位用户评论说，新的 **Memory 功能** 让对话感觉更像人类，具有明显的信息记忆能力。
   - 与能够像人一样记住细节的事物互动的感受被强调为特别令人印象深刻。
- **使用 DALLE-3 创建图像**：一位成员询问了免费版本通过 **DALLE-3** 创建图像的问题，寻求关于可用选项的说明。
   - 据分享，用户在特定频道每天有 **5 次绘图额度**，并且还可以通过 ChatGPT 使用 **2 次免费的 DALLE-3 请求**。
- **使用 4o 处理复杂请求**：有建议称 **4o** 模型可以处理复杂请求，允许在单个查询中执行多任务。
   - 鼓励用户清晰地表达他们的请求，将模型视为可以协助完成多项任务的人。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1282781007434748095)** (13 messages🔥): 

> - `Stock Evaluation` (股票评估)
> - `Universal Evaluator Prompt` (通用评估器提示词)
> - `Accessing Prompt Library` (访问提示词库)


- **股票预测需谨慎**：一位成员警告不要使用 **OAI 模型**进行股票分析，除非能够访问**完整的历史数据**以及用于实时更新的 API。
   - 他们指出，许多网站以 **JSON 格式**提供完整的股票历史记录，这对于准确建模至关重要。
- **用于提示词开发的通用评估器**：一位成员分享了他们创建的 **Universal Evaluator** 提示词角色，该角色可以比较两个输出，并根据主观性给出数值评分。
   - 他们强调了让评估器阐明其推理过程的重要性，以便获得更有见地的上下文。
- **提示词库访问问题**：一位用户询问如何访问 **prompt library** 以获取提示词开发资源。
   - 另一位成员提供了有用的回复，引导他们前往现在名为 <#1019652163640762428> 的相关频道。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1282781007434748095)** (13 messages🔥): 

> - `Using OAI Models for Stock Analysis` (使用 OAI 模型进行股票分析)
> - `Universal Evaluator Prompt Persona` (通用评估器提示词角色)
> - `Accessing Prompt Library` (访问提示词库)


- **不建议将 OAI 模型用于股票分析**：一位成员强调，除非你拥有**所有**历史数据（包括**图像**和**图表**）并能实现实时更新，否则使用 **OAI 模型**分析股票是无效的。
   - 他们指出，出于性能考虑，准确的股票分析需要使用 **API**，并提到可以下载 JSON 格式的完整股票历史记录。
- **用于提示词开发的通用评估器**：另一位成员分享了他们创建的 **Universal Evaluator prompt persona**，该角色通过比较输出来判断质量，并基于主观推理提供**数值评分**。
   - 他们强调了该工具对 **prompt development** 的重要性，以及评估器在上下文中解释其推理的必要性。
- **提示词库的位置**：一位用户询问如何访问 **prompt library**，并收到了包含更新后频道名称的回复，现在称为 **<#1019652163640762428>**。
   - 这一互动凸显了社区在协助资源导航方面的积极意愿。


  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1283135584885538898)** (1 messages): 

> - `Hermes 3 transition` (Hermes 3 转型)
> - `Paid model announcement` (付费模型公告)


- **Hermes 3 转向付费模型**：标准的 **Hermes 3 405B** 将在周末前转为付费模型，提示用户调整使用方式。
   - 要继续免费使用，请切换到模型标识符 `nousresearch/hermes-3-llama-3.1-405b:free`，因为**免费变体**的可用性可能有限。
- **Hermes 3 模型访问即将发生的变化**：建议用户注意，向付费模型的过渡很快就会发生，可能会影响对 **Hermes 3** 的访问。
   - 此更改即将生效，因此切换到指定的免费模型标识符以避免中断至关重要。



**提到的链接**：<a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b:free">Hermes 3 405B Instruct (free) - API, Providers, Stats</a>：Hermes 3 是一个通用语言模型，相比 Hermes 2 有许多改进，包括先进的 Agent 能力、更出色的角色扮演、推理、多轮对话以及长上下文连贯性...

  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1282783049108422729)** (2 messages): 

> - `Eggu Dataset` (Eggu 数据集)
> - `Open Source Multilingual Models` (开源多语言模型)
> - `Cost of Usage` (使用成本)


- **Eggu 数据集开发**：**Eggu** 数据集目前正在开发中，旨在训练一个开源多语言模型，大小为 **1.5GB**，并结合了图像定位以兼容视觉模型 (Vision Models)。
   - 该数据集旨在供大众使用，但也面临被某些人滥用的担忧。
- **训练成本相对较低**：使用 OpenAI 服务仅**一周的使用量**就花费了大约 **$2,500** 的额度。
   - 考虑到数据集和模型可能产生的产出，这笔费用被认为是合理的。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1282780180087312415)** (102 messages🔥🔥): 

> - `DeepSeek 模型与性能`
> - `Google Gemini Flash 速率限制`
> - `Sonnet 3.5 Beta 问题`
> - `Hermes 3 与 Llama 3 模型成本`
> - `AI 编程工具探索` 


- **DeepSeek 模型混淆**：讨论揭示了用户对 DeepSeek 模型的混淆，特别是关于 ['coder'](https://openrouter.ai/models/deepseek/deepseek-coder) 与 ['chat'](https://openrouter.ai/models/deepseek/deepseek-chat) 端点的区别。成员们注意到模型 ID 将在未来五天内保持免费，这缓解了对迁移的担忧。
   - 对 **吞吐量 (throughputs)** 较低的担忧，有报告称某些模型的性能仅为 **1.75t/s**，有的也只有 **8tps**。
- **Google Gemini Flash 速率限制困扰**：一位用户报告了 **Google Gemini Flash 1.5** 反复出现的速率限制问题，称其应用即使在有用户限制的情况下也会频繁触及上限。他们正在与 **NVIDIA Enterprise Support** 沟通，以澄清兼容性和限制。
   - 有人担心许多人被迫使用 **experimental API**，这带来了自身的局限性，正如他们在访问模型时遇到的错误所显示的那样。
- **Sonnet 3.5 Beta 停机确认**：确认了近期影响 **Sonnet 3.5 Beta** 的停机事件，用户报告 API 交互成功率下降。来自 **Anthropic** 的状态更新确认免费用户的成功率已恢复正常。
   - 随着访问权限的恢复，成员们表示松了一口气；然而，关于稳定性的核心问题在讨论中仍然普遍存在。
- **Hermes 3 定价推测**：参与者讨论了关于 **Hermes 3 405b** 未来成本的推测，表现出对从免费访问过渡的潜在焦虑。一位用户幽默地指出，用户在习惯了免费后，对突然收费可能会产生何种反应。
   - 对话指出，虽然 **Llama 3 405B** 的输出更便宜，但也可能在性能上有所权衡，这让许多用户陷入了决策困境。
- **AI 编程工具探索**：用户讨论了适用于编程的工具，提到了 **Aider** 和 **Cursor**，强调了各自的功能和体验。有人指出 **Aider** 的方法论由于其与模型响应交互的方式，可能会让人感觉有些奇特。
   - 对话反映了用户对寻找有效编程辅助工具的广泛兴趣，表明用户打算根据当前的云端额度可用性尝试各种产品。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://status.anthropic.com/">Anthropic Status</a>：未找到描述</li><li><a href="https://openrouter.ai/docs/parameters-api">Parameters API | OpenRouter</a>：用于管理请求参数的 API</li><li><a href="https://openrouter.ai/models/deepseek/deepseek-coder">DeepSeek-Coder-V2 - API, Providers, Stats</a>：DeepSeek-Coder-V2，一个开源的 Mixture-of-Experts (MoE) 代码语言模型。它是在 DeepSeek-V2 的中间检查点基础上，通过额外的 6 万亿 token 进一步预训练而成的。运行 DeepSeek...</li><li><a href="https://openrouter.ai/models/deepseek/deepseek-chat">DeepSeek V2.5 - API, Providers, Stats</a>：DeepSeek-V2 Chat 是 DeepSeek-V2 的对话微调版本，后者是一个 Mixture-of-Experts (MoE) 语言模型。它包含 236B 总参数，其中每个 token 激活 21B 参数。运行 DeepSeek V2....</li><li><a href="https://github.com/paul-gauthier/aider/issues">Issues · paul-gauthier/aider</a>：aider 是你终端里的 AI 结对编程工具。欢迎在 GitHub 上为 paul-gauthier/aider 的开发做出贡献。
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1282891286361149490)** (5 messages): 

> - `Opus API 集成`
> - `模型上传挑战` 


- **Opus API 调用引发关注**：讨论强调了使用 **Opus API 调用** 来获取“正确”版本的有趣之处，这表明集成方式正在发生转变。
   - 一位成员提到他们昨天刚看到相关的推文，显示该话题在社区中的热度正在上升。
- **模型上传被证明具有挑战性**：一位参与者评论说，**模型上传**比最初预想的要困难得多，暗示了此前未预见到的复杂性。
   - 这一见解反映了用户在处理上传过程中所面临的实际挑战的广泛担忧。


  

---

### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1282781190109007935)** (6 条消息): 

> - `Batch 性能优化`
> - `Triton 原子操作`
> - `Triton 编译过程` 


- **Batch Size 对性能的影响**：正如讨论的那样，较小的矩阵/Batch Size 表现出略好的性能，在 `(1, 4096, 4096)` 规模下实现了约 **3 倍的加速**，而非 **1.8 倍**，但适当的优化可能需要完全重写 Kernel。
   - int32 打包被认为是“无损”的，虽然存在 int16 和 int8 选项，但它们有引入**量化误差（quantization errors）**的风险。
- **Triton Atomic Add 的限制**：`tl.atomic_add` 目前仅支持 1D Tensor，这引发了关于 2D Tensor 潜在变通方案的讨论。
   - 社区正在寻求高效的方法或替代方案，以便在多维数据上实现类似功能。
- **关于 Triton 编译的说明**：一位成员质疑了 Triton 编译过程的准确性，特别是它是否直接将 Python 编译为 PTX，并引用了一篇关于 Triton 内部机制的文章。
   - 共识是 Triton 将源码转换为 **Triton IR**，然后转换为 **LLVM IR**，最后转换为内联 **PTX**。


  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1282874117627969557)** (7 条消息): 

> - `PyTorch 自动调优`
> - `Triton Autotuner`
> - `PyTorch 中的函数式优化器`
> - `开源模型适配`
> - `Tulu 项目公告` 


- **探索适用于 Triton Kernel 的 PyTorch Autotune**：成员们讨论了是否可以利用具有 **Triton Kernel** 自动调优功能的 **PyTorch** `inductor/dynamo` 来处理自定义 Kernel，通过缓存调优参数来增强性能。
   - 一位成员认为，这种进步可能会使后续使用相同 Kernel 的运行速度更快。
- **Triton 的 Autotuner 与手动缓存**：一位成员指出，虽然 **Triton** 有自己的 Autotuner，但它缺乏在脚本重新运行之间保存调优结果的功能。
   - 另一位成员幽默地表示，他不确定 Triton 是否存在解决此问题的任何内置功能。
- **有趣的函数式优化器亮相**：一位成员分享了来自 Apple 的 **sigmoid attention** 发布中一个有趣的函数式 **PyTorch 优化器**，认为将其与 `torch.func.grad_and_value` 结合时可以创造宝贵的融合机会。
   - 他们暗示这种结合可能会在使用 `torch.compile` 的前向、后向和优化步骤中带来进步。
- **Hamish Ivison 专家交流会公告**：一场关于“使用 **Open-Instruct** 和 **Tulu** 适配开源模型”的专家交流会即将举行，主讲人为 **Hamish Ivison**，时间定于明天**上午 11 点（PST）**。
   - 鼓励参与者通过 [YouTube](https://www.youtube.com/watch?v=e1qUJFAo10s) 观看直播，并在演讲期间与讲者互动。
- **关于 Open-Instruct 和 Tulu 项目的见解**：Hamish Ivison 将讨论语言模型的后训练策略，追溯 **open-instruct 库**的演变及其对 **Tulu 项目**的影响。
   - 参与者可以期待关于基于 llama 适配的最先进模型的见解，以及即将发布的 **Tulu 3** 的预览。



**提到的链接**：<a href="https://github.com/apple/ml-sigmoid-attention/tree/main/optorch">ml-sigmoid-attention/optorch at main · apple/ml-sigmoid-attention</a>：通过在 GitHub 上创建账户来为 apple/ml-sigmoid-attention 的开发做出贡献。

  

---

### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1282894693000286282)** (24 条消息🔥): 

> - `Sigmoid Attention 论文`
> - `FlashSigmoid vs FA3`
> - `Sigmoid Attention 中的 Bias`
> - `Elementwise Sigmoid vs Rowwise Softmax`
> - `LayerScale` 


- **关于 Sigmoid Attention 论文的讨论**：成员们分享了对 [Sigmoid Attention 论文](https://arxiv.org/abs/2409.04431) 的兴趣，一些人对 Attention 的修改和潜在的输出边界表示怀疑。
   - 针对在 Sigmoid 函数之前添加 Bias 可能如何处理这些问题进行了讨论。
- **比较 FlashSigmoid 和 FA3**：一位成员指出，**FlashSigmoid** 的性能与 **FA3** 相比如何令人好奇，特别是考虑到 FA3 是针对 Hopper GPU 优化的。
   - 讨论了是否应该在 FlashSigmoid 与使用 Elementwise Sigmoid（而非 FA2）的 FA3 修订版之间进行比较。
- **澄清 Sigmoid Attention 中的 Bias 和 Reduction**：澄清了 Sigmoid Attention 中的 Bias `b` 是固定的，计算方式为 `b=-log(L)`，其中 `L` 是序列长度，从而消除了对 Reduction 步骤的需求。
   - 该属性旨在使 Logits 之和接近 1，类似于 Softmax，从而避免无界输出。
- **Elementwise Sigmoid 与 Rowwise Softmax 的有效性**：针对使用 **Elementwise Sigmoid** 替代 **Rowwise Softmax** 的有效性提出了疑问，重点关注其性能和实现细节。
   - 成员们讨论了这种替换的影响以及在不同实现中潜在的性能差异。
- **对 LayerScale 的好奇**：一位成员对 **LayerScale** 在当前讨论背景下的功能表示好奇。
   - 这引发了探索它如何与所讨论策略（特别是关于 Attention）的性能挂钩的兴趣。



**提到的链接**：<a href="https://arxiv.org/abs/2409.04431">Theory, Analysis, and Best Practices for Sigmoid Self-Attention</a>：Attention 是 Transformer 架构的关键部分。它是一种序列到序列的映射，将每个序列元素转换为值的加权和。权重通常通过...

  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1282945115132006461)** (5 条消息): 

> - `矩阵乘法的 Tiling 概念`
> - `Pragma unroll 的用法`
> - `矩阵乘法资源` 


- **矩阵乘法的 Tiling 概念详解**：一位成员正在寻求关于矩阵乘法 **Tiling 概念** 的澄清，并寻找相关资源或指导以更好地理解它。
   - 另一位成员推荐了一个有用的 [动画](https://youtu.be/Q3GgbfGTnVc?si=ejkL0DRD70uXn7lZ&t=142) 以帮助理解。
- **对 Pragma Unroll 有效性的困惑**：一位成员指出他们在利用 **pragma unroll** 时遇到了问题，表示它似乎没有像预期那样启用多线程。
   - 作为回应，另一位成员建议通常 **pragma unroll** 是多余的，因为大多数编译器会自动处理它。


  

---


### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1282943367524778006)** (3 条消息): 

> - `矩阵乘法的 Tiling 概念`
> - `矩阵乘法优化资源` 


- **在矩阵乘法 Tiling 概念中挣扎**：一位成员表达了他们在理解 **矩阵乘法 Tiling 概念** 方面的困难，并寻求建议。
   - 建议使用 *铅笔和纸重新绘图* 作为理解该概念的潜在帮助。
- **矩阵乘法优化资源**：另一位成员分享了一篇关于优化 **CUDA 矩阵乘法** 实现的深刻文章，重点关注内存合并（Memory Coalescing）和缓存等性能特征。
   - 文章包含了所有 Kernel 的代码，并引用了其他有用的仓库，强调了矩阵乘法在训练 **大型深度学习模型** 中的重要性。



**提到的链接**：<a href="https://siboehm.com/articles/22/CUDA-MMM">How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog</a>：在这篇文章中，我将迭代优化一个用 CUDA 编写的矩阵乘法实现。我的目标不是构建一个 cuBLAS 的替代品，而是深入...

  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 条消息): 

pauleonix: 这里有 suckerpinch 的粉丝吗？😆 https://youtu.be/Ae9EKCyI1xU

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1283151469268176916)** (7 条消息): 

> - `Activation Value Saving`
> - `Activation Checkpointing`
> - `Memory Optimization Techniques`
> - `Liger Kernel Memory Management` 


- **为反向传播存储激活值**：一名成员确认他们在应用激活函数后会**保存激活值**（save activation values）用于反向传播，从而提高计算效率。
   - 此外还讨论了可选的 **activation checkpointing** 以及正在实施的**雄心勃勃的** FP8/tensor 变更。
- **通过重新计算输出来节省内存**：提出了一种聪明的内存节省方法，即通过重新计算 **GELU** 或 **LayerNorm** 等输出来代替保存它们。
   - 这种策略可以在模型训练期间实现显著比例的**总内存节省**。
- **激活梯度的内存效率**：指出**激活梯度**（activation gradients）可以复用现有的缓冲区（buffers），从而实现有效的**零额外内存占用**。
   - 该技术优化了内存管理，实现了更高效的利用。
- **Liger Kernel 的分块方法**：有人提问关于 **Liger Kernel** 对 logits/dlogits 进行分块（chunking）的方法，该方法能将内存占用显著降低至原始量的 1/X。
   - 该方法涉及执行 **X 次更小的矩阵乘法**（matrix multiplications），这可能会提升性能，但目前尚不清楚是否已得到充分考虑。


  

---


### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/1283020405770158232)** (11 条消息🔥): 

> - `CUDA-MODE IRL Event Details`
> - `Quantization and Sparsity Projects`
> - `GPU Availability for Hacking` 


- **9 月 21 日的 CUDA-MODE IRL 活动**：即将举行的 CUDA-MODE IRL 活动定于 **9 月 21 日** **上午 10 点至午夜**举行，尽管有 **650 名申请者**，但仅提供 **150 个名额**。
   - 敦促参与者尽快确认他们的 RSVP，以便参加预定在 **下午 1 点至晚上 11 点** 的 hacking session，届时将有著名主讲嘉宾 **Wen-mei Hwu** 出席。
- **计划中令人兴奋的量化与稀疏化项目**：准备了一系列专注于**量化（quantization）和稀疏化（sparsity）**的项目，旨在大幅**降低部署期间的 VRAM 占用**和成本。
   - 项目分为**高性能实现**（High Performance Implementation）、**研究项目**（Research Projects）和**量化流程项目**（Quantization Flow Projects），[点击此处查看详细列表](https://docs.google.com/document/d/14BJ7a1wx1uqzrmCbBsLjX7YQNKrtIhRDuz1l0Tlmkoc/)。
- **关于 Hack Session 可用 GPU 的咨询**：一名参与者询问了用于 hacking 的 **GPU 可用性**，特别是是否会包含 **AMD GPUs**。
   - 作为回应，一名成员确认将提供来自多个供应商的各种 GPU，并承诺**很快会公布算力赞助商**（compute sponsors）的更多细节。



**提到的链接**：<a href="https://docs.google.com/document/d/14BJ7a1wx1uqzrmCbBsLjX7YQNKrtIhRDuz1l0Tlmkoc/">Quantization and Sparsity Projects</a>：IRL 的量化与稀疏化项目。高性能实现项目：1. 开发一个 A16W3（混合 fp16 x 3-bit）Fused Matmul Kernel：为什么？目前还没有可用的 3-...

  

---

### **CUDA MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1282789842341855337)** (7 条消息): 

> - `phi3 基准测试`
> - `GPU 利用率问题`
> - `序列长度导致的 OOM 问题`
> - `GPU CI 失败` 


- **在 A100 GPU 上进行 phi3 基准测试的困扰**: 一位用户尝试在单张 A100 40GB 上对 **phi3** 进行基准测试，但在 token 吞吐量方面遇到挑战，并为此在 [GitHub 上提交了 issue](https://github.com/linkedin/Liger-Kernel/issues/236)。
   - 他们改编了 [Hugging Face 示例](https://github.com/linkedin/Liger-Kernel/tree/main/examples/huggingface)，并正在考虑通过分布式训练来改进结果。
- **关于 GPU 利用率的担忧**: 另一位用户建议，由于 batch size 和序列长度较小，**GPU** 可能未被充分利用，这可能会影响性能。
   - 这一观点在对话中得到了共鸣，表明基准测试条件仍有优化空间。
- **高序列长度下的 OOMKilled 错误**: 原用户报告称，当序列长度超过 **512** 时会遇到 **OOMKilled** 错误，这表明 GPU 存在显存限制。
   - 这引发了关于与其 **40GB** 显存容量相关的内存带宽限制的讨论。
- **GPU CI 构建问题**: 一位成员询问了 **GPU CI** 失败的情况，得到的回复确认了该问题确实存在，并且正在努力修复中。
   - 这表明社区正在协作解决 CI 流水线问题。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/linkedin/Liger-Kernel/issues/236">Benchmarking phi3 on single A100 40gb GPU: unable to reproduce benchmark results · Issue #236 · linkedin/Liger-Kernel</a>: 🐛 错误描述：我正在使用 flyte 尝试复现本仓库 README 中报告的 token 吞吐量和显存节省结果，但在略有不同的条件下：使用 microsoft/Phi-3-m...</li><li><a href="https://github.com/linkedin/Liger-Kernel/tree/main/examples/huggingface">Liger-Kernel/examples/huggingface at main · linkedin/Liger-Kernel</a>: 用于 LLM 训练的高效 Triton Kernels。通过在 GitHub 上创建账号为 linkedin/Liger-Kernel 做出贡献。
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1282813271367356437)** (31 条消息🔥): 

> - `Cohere 的可接受使用政策`
> - `模型微调`
> - `社区成员介绍`
> - `机器人维护更新` 


- **Cohere 的可接受使用政策**: 一位成员分享了 [Cohere 可接受使用政策](https://docs.cohere.com/docs/c4ai-acceptable-use-policy) 的链接，其中详细列出了禁止的使用案例，包括**暴力**和**骚扰**。
   - 社区讨论了使用模型衍生品时的**商业用途**影响以及对当地法律的合规性。
- **关于模型微调的讨论**: 一位成员询问了 CMD-R 模型的 **fine-tuning** 政策，特别是是否可以免费使用。
   - 另一位成员澄清说，**self-hosted** 模型禁止任何商业用途。
- **热烈欢迎与自我介绍**: 几位新成员向社区介绍了自己，分享了加入的原因并表达了兴奋之情。
   - 现有成员欢迎了新人，并就他们的项目和兴趣进行了交流。
- **机器人维护更新**: 一位社区成员宣布，另一位成员正在努力修复 **cotector bot**，旨在恢复其功能。
   - 其他人对这次接手表示热切期待，预计很快就能完成。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.cohere.com/docs/c4ai-acceptable-use-policy">Cohere For AI Acceptable Use Policy — Cohere</a>: C4AI 可接受使用政策</li><li><a href="https://docs.cohere.com/docs/c4ai-acceptable-use-policy?_gl=1*121mhsp*_gcl_aw*R0NMLjE3MjUxNDE4NzcuRUFJYUlRb2JDaE1JbkpqLW9wNmdpQU1WcmtYX0FSMXpCeW82RUFBWUFTQUFFZ0s1RFBEX0J3RQ..*_gcl_au*MTU4MDQyNzY2Ny4xNzI1MDU4MTcx*_ga*NzgwODE4Mzk0LjE3MjUwNTgyMTE.*_ga_CRGS116RZS*MTcyNTkxOTE0MS41LjEuMTcyNTkxOTE3Mi4yOS4wLjA">Cohere For AI Acceptable Use Policy — Cohere</a>: C4AI 可接受使用政策
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1282970322467029012)** (2 messages): 

> - `Embedding documents`
> - `Fine-tuning LLMs` 


- **选择 Embedding 以获得快速响应**：一位成员建议，如果处理少量文档以获得即时响应，可以使用 embedding；而对于大批量（100K+）文档，则首选 embedding jobs。
   - 后一种选项可以处理 **validation** 和 **batching** 等方面，确保运行更顺畅。
- **寻求 Fine-tuning LLMs 的资源**：另一位成员询问了关于如何 fine-tune **LLMs** 的推荐视频或书籍。
   - 聊天中未提供具体资源，表明在该主题的知识共享方面可能存在空白。


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1282973213688856596)** (1 messages): 

> - `Temperature settings in outputs` 


- **测试 Temperature 设置**：一位成员建议尝试不同的 temperature 设置，特别是 **0** 或 **0.1**，以评估输出质量的变化。
   - 这种方法旨在确定获得的输出是否与提供的初始示例 **大相径庭**。
- **对输出差异的担忧**：另一位成员表示有兴趣了解输出的变化与给出的示例相比是否显著。
   - 讨论围绕通过 temperature 调整来优化输出质量展开。


  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1282799970528919572)** (42 messages🔥): 

> - `Advanced Computer Vision Projects`
> - `Multimodal Learning`
> - `Pokedex Project`
> - `Google Vision API`
> - `Team Collaboration` 


- **探索高级 Computer Vision 项目**：一位成员请求关于 **computer vision** 领域的 **高级项目创意** 建议，表达了希望通过 **Cohere** 学习新知识的愿望。
   - 另一位成员建议关注 computer vision 与 **LLM projects** 的交叉领域以获取灵感。
- **团队协作提升成功率**：讨论强调了项目中团队合作的重要性，一位成员指出，拥有一个 **优秀的团队** 对其项目的成功贡献巨大。
   - 这引发了关于通过组队来克服“容易开始项目但难以完成”这一常见问题的想法。
- **Pokedex 项目的乐趣**：一位成员分享了一个有趣的 **Pokedex project** 细节，该项目使用了 **Google Vision API**、**Cohere LLMs** 和 *Wombo* 进行图像生成，从而创造了独特的用户体验。
   - 该项目根据图像的显著特征识别 **Pokemon** 名称和描述，以创意的方式整合了多种技术。
- **使用 Google Vision API 获取图像标签**：关于 **Google Vision API** 在 Pokedex 项目中的作用出现了疑问，随后澄清其用途是 **创建图像标签** 而非学习 embeddings。
   - 这引发了关于利用现有数据集的进一步讨论，并提到 **Kaggle** 是 **Pokemon datasets** 的资源来源。
- **完成项目的挑战**：一位成员对在不同项目之间 **切换** 却未能完成感到沮丧，并将其归因于“严重的技能问题”。
   - 另一位成员提供了建议，认为组队可以帮助克服这种“起步容易收尾难”的普遍困境。


  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1282793265573138574)** (14 条消息🔥): 

> - `Windows usage`
> - `Desktop beta`
> - `Android mobile devices`
> - `Open Interpreter product discussion`
> - `Project issues` 


- **探索 Windows 使用**：一名成员询问了如何在 **Windows** 上使用该项目。
   - 这反映了对该平台跨操作系统兼容性的普遍关注。
- **桌面版 Beta 访问咨询**：有人提问现在加入 **桌面版 beta** 计划是否太晚。
   - 这突显了用户对访问 beta 版中引入的新功能的持续兴趣。
- **寻找 Android 移动设备**：有人询问在哪里可以购买 **Android 移动设备**，并得到了一个指向 Amazon 产品的链接。
   - 这显示了用户对项目与移动端集成的热情。
- **关于真实产品的讨论**：一名成员质疑是否存在任何**真实产品**，引发了辩护并附上了 **Open Interpreter GitHub** 仓库的链接。
   - 这表明社区内对产品可用性存在批判性观点。
- **移动应用反馈问题**：一名新成员报告了 **Android** 移动应用在完成任务后不提供反馈的问题。
   - 建议他们在相应频道创建 issue，表明了对故障排除的积极支持。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.amazon.com/FDMDAF-smartphone-Cellphone-Lightweight-Unlocked/dp/B0CYGZFC54">未找到标题</a>：未找到描述</li><li><a href="https://github.com/OpenInterpreter">Open Interpreter</a>：Open Interpreter 有 5 个可用的仓库。在 GitHub 上关注他们的代码。
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1282778397914824705)** (57 条消息🔥🔥): 

> - `01 Light Discontinuation`
> - `Refund Process`
> - `01 App Launch`
> - `Testing and Beta Feedback`
> - `Community Support for Open Source` 


- **01 Light 停止生产**：团队宣布停止 **01 Light** 的生产，并已退还所有硬件订单的款项，因为他们将重心转向适用于 Android 和 iOS 的 **01 App**。
   - 他们强调这一决定是为了加强软件开发并优先考虑其开源愿景，得到了社区的赞赏。
- **硬件购买退款已处理**：许多成员确认已收到退款，表明流程迅速且高效；在某些情况下，退款是在用户未申请的情况下自动处理的。
   - 退款流程的透明和速度引发了用户对该情况处理方式的正面反馈。
- **移动端 01 App 发布**：**01 App** 现已在 Android 和 iOS 上线，未来计划根据用户反馈增强其功能。
   - 鼓励开发者在 **GitHub** 上 fork 该应用以创建定制化体验，扩大在各种设备上的可访问性。
- **申请官方桌面版 App 的 Beta 测试**：几位用户询问了关于**官方桌面版 app** 的 beta 测试，寻求参与反馈过程以优化其性能。
   - 团队澄清发布时间表取决于当前 beta 测试人员的反馈，保持与社区的互动。
- **支持开源开发**：社区对为源自 01 平台的开源项目做出贡献表现出极大的热情，突显了新软件计划的潜力。
   - 参与者分享了他们协助开发和故障排除的意愿，增强了社区内的协作精神。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://changes.openinterpreter.com/log/01-app">Open Interpreter - It should have been an app</a>：开源项目 Open Interpreter 的官方变更日志。</li><li><a href="https://01.openinterpreter.com/software/server/livekit-server">Livekit Server - 01</a>：未找到描述</li><li><a href="https://changes.openinterpreter.com/log/01-app)">Open Interpreter Changelog</a>：开源项目 Open Interpreter 的官方变更日志。
</li>
</ul>

</div>
  

---

### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1283059461690494986)** (5 messages): 

> - `Tool Use 剧集发布`
> - `YouTube 链接` 


- **Tool Use 第 4 集发布**：题为 *'Activity Tracker and Calendar Automator - Ep 4 - Tool Use'* 的最新剧集现已在 [YouTube](https://www.youtube.com/watch?v=N9GCclB8rYQ) 上线。**Mike Bird** 和 **Ty Fiero** 讨论了如何利用 AI 优化**时间管理**。
   - 视频强调**时间是我们最宝贵的资源**，旨在激励观众有效地利用工具。
- **令人兴奋的内容大爆发！**：成员们对今天发布的大量新内容表示兴奋，激发了整个频道的积极性。
   - 一位成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=FAFmP82bhDA)，进一步丰富了今天的内容。



**提及的链接**：<a href="https://www.youtube.com/watch?v=N9GCclB8rYQ">Activity Tracker and Calendar Automator - Ep 4 - Tool Use</a>：时间是我们最宝贵的资源，让我们用 AI 来优化它！在本集 Tool Use 中，Mike Bird 和 Ty Fiero 讨论了时间管理的重要性……

  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1282816199654248561)** (10 messages🔥): 

> - `Windows 原生版本`
> - `专注于 Linux 支持`
> - `WSL 支持`
> - `社区会议`
> - `用户反馈机会` 


- **Windows 原生版本尚无时间表**：目前关于 **Windows 原生版本**的可用性**没有时间表**，因为 Modular 正在优先支持 **Ubuntu 和 Linux 发行版**。
   - *Modular 旨在扩大关注范围之前避免技术债并提高产品质量*，并吸取了以往 Swift 的经验教训。
- **Windows 用户可使用 WSL 支持**：虽然原生 **.exe** 版本尚未推出，但 *Modular 建议将 WSL 作为其目前 Windows 支持的范围*。
   - 用户对未来的原生选项表示期待，但也承认目前的局限性。
- **增加原生 Windows 支持的挑战**：*团队渴望增加原生 Windows 支持*，但缺乏具备必要技能的人员来加速这一进程。
   - 尽管面临挑战，团队对未来实现这一目标仍保持乐观。
- **MAX + Mojo 社区会议录像已上线**：**MAX + Mojo 社区会议**的录像现已在 YouTube 上提供，展示了引人入胜的演示。
   - 鼓励参与者观看，并感谢为活动做出贡献的演讲者。
- **征求 Magic 产品的用户反馈**：Modular 正在寻找尚未与 **Magic** 互动的用户，通过 30 分钟的简短通话提供宝贵反馈。
   - 参与者将因其时间和见解获得专属礼品 (swag)，链接至[预约页面](https://modul.ar/user-feedback)进行安排。



**提及的链接**：<a href="https://modul.ar/user-feedback">Appointments</a>：未找到描述

  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1282777564657487945)** (61 messages🔥🔥): 

> - `Mojo 语言能力`
> - `Mojo 中的 DLHandle`
> - `GStreamer 绑定`
> - `Mojo 中的 Variant 类型`
> - `Mojo 中的 SDL 绑定` 


- **Mojo 在 GPU 和 GStreamer 方面的未来**：Mojo 被定位为替代 GStreamer 的候选者，利用其即将推出的 GPU 能力进行高效处理。
   - 成员们对集成用于实时流媒体的现代库表现出兴趣，强调了 Mojo 简化复杂任务的潜力。
- **使用 DLHandle 创建绑定**：几位成员讨论了使用 DLHandle 创建 Mojo 绑定，并引用了其他展示其应用的项目。
   - 像 'dustbin' 这样的项目利用 DLHandle 进行 SDL 绑定，为其他对图形应用感兴趣的人提供了示例。
- **Mojo 中 Variant 类型的使用**：关于 Mojo 中 Variant 类型的讨论强调了它在创建包含多种元素类型的列表方面的实用性，以及相关的内存问题。
   - 成员们澄清了 Variant 实现中的大小对齐（alignment）问题和判别式（discriminants）的行为。
- **Mojo 中的 SDL 和图形编程**：成员们分享了 Mojo 的 SDL 绑定项目链接，帮助那些寻找简单示例以将图形集成到应用程序中的人。
   - 推荐了第一个分享的项目，因为它提供了简单的示例，无需构建完整的应用程序。
- **Mojo 编码中的初学者挑战**：一位成员报告了在检查字符是否为数字时遇到的错误，发现必须显式转换为 String。
   - 这提高了对 Mojo 中类型转换的认识，并为面临类似挑战的初学者提供了见解。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/utils/variant">variant | Modular 文档</a>：定义了 Variant 类型。</li><li><a href="https://www.youtube.com/watch?v=JRcXUuQYR90">Mojo 语言 - 未来的高性能 Python？（对话 Chris Lattner）</a>：Mojo 是 Swift 和 LLVM 创始人的最新语言。它尝试吸取 CPU/GPU 级编程的一些最佳技术并进行封装...</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/src/utils/variant.mojo">modularml/mojo 项目 main 分支下的 mojo/stdlib/src/utils/variant.mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账户为 modularml/mojo 开发做出贡献。
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1282830724742910114)** (65 messages🔥🔥): 

> - `DisTro 引起的困惑`
> - `AI 训练陷阱`
> - `OCTAV 算法实现`
> - `AI 回复中的重复问题`
> - `各种 AI 模型的性能` 


- **DisTro 引发猜测**：围绕 DisTro 展开了讨论，涉及对其目的和有效性的疑问，导致成员之间的困惑。
   - 一位成员指出目前尚未发布任何代码，暗示该公告是为了引发竞争。
- **对 AI 训练质量的担忧**：许多人表示担心，基于用户满意度指标训练的 AI 模型往往会产生肤浅的信息，而不是准确的内容。
   - 一位成员强调，这种趋势可能会导致 AI 回复质量下降，尤其是在依赖人类反馈时。
- **成功集成 NVIDIA 的 OCTAV**：一位成员分享了使用 Sonnet 将 NVIDIA 的 OCTAV 算法成功实现到其代码库中的经验，并指出网上缺乏类似的示例。
   - 他们推测该实现是否是从论文中推断出来的，展示了 AI 模型的能力。
- **重复回复的问题**：小组讨论了 AI 提供重复输出的倾向，以及用户回复中轻微犹豫所产生的影响。
   - 一位成员评论道，像 Claude 这样的模型撤回其解决方案的速度非常快，这表明在保持输出自信度方面仍需改进。
- **各种 AI 模型的参差表现**：评估了 Claude 和 Opus 等平台的性能，成员们评论了它们在对话中的优缺点。
   - 一位成员指出，虽然 Claude 有很好的对齐策略，但在某些条件下往往会失准，而不像 Opus 那样让他们觉得更有吸引力。



**提到的链接**：<a href="https://x.com/abacaj/status/1833247396278726966">anton (@abacaj) 的推文</a>：无法复现 reflection (ref_70_e3) 91% 的 humaneval 分数，在本地使用 vLLM 以 bf16 运行。使用了“推荐”的系统提示词 + 从输出标签中提取：81.1% meta-llama-...

  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1283094422032224266)** (6 条消息): 

> - `AI 中的 Scaling`
> - `数据质量`
> - `Rich Sutton 的 Bitter Lesson`
> - `AI 研究趋势` 


- **Scaling：是炒作还是解决方案？**：成员们讨论了 Scaling 仅仅是炒作，还是真正掌握着 AI 有效推理的关键，并强调了**高质量数据**比其他因素更重要。
   - 一位成员指出，之前的专家反复强调**高质量数据**至关重要，这让人对 Scaling 的真实影响产生了疑问。
- **Rich Sutton 对 AI 演进的见解**：分享的一篇来自 [Rich Sutton](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) 的文章强调，AI 中最有效的方法依赖于**利用计算**而非人类知识，特别是引用了 Moore's Law。
   - Sutton 认为，历史上对人类专业知识的依赖可能会阻碍进步，并明确指出最终重要的是最大化可用计算量。
- **高质量数据增强 Scaling 效果**：讨论范围涉及**更高质量的数据**不仅能提高模型性能，还能与 Scaling 互补，暗示了协同效应。
   - 成员们承认，不断有证据表明，随着 AI 参数规模（Scaling）的扩大，配合更好的数据，性能提升会更加显著。



**提到的链接**：<a href="http://www.incompleteideas.net/IncIdeas/BitterLesson.html">The Bitter Lesson</a>：未找到描述

  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1282805762053111971)** (53 条消息🔥): 

> - `Tokenizer eos 问题`
> - `Eleuther_Eval recipe 加载`
> - `数据集的 ChatML 格式`
> - `训练中的 Checkpointing`
> - `Hugging Face TRL 库` 


- **Mistral 和 Gemma 缺少 Tokenizer eos 选项**：一位用户提议发送一个 PR 来修复 Tokenizer 的 eos 问题，指出目前的 Mistral 和 Gemma Tokenizer 没有 `add_eos` 选项。
   - 另一位成员强调，在修复之前，他们首先需要实现 `add_eos` 功能，并引用了一个[需要更新的工具类](https://github.com/pytorch/torchtune/blob/main/torchtune/modules/tokenizers/_utils.py)。
- **Eleuther_Eval recipe 默认使用 GPT-2 模型**：一位成员询问为什么 Eleuther_Eval recipe 总是加载 GPT-2 模型，得到的解释是自 `lm_eval==0.4.3` 以来，这是默认的模型类型。
   - 他们提到，模型定义会被其 `TransformerDecoder` 工具覆盖，以便评估其他模型。
- **以 ChatML 格式格式化数据集**：新用户表达了在为 Qwen2 格式化 ChatML 兼容数据集时遇到的挑战，寻求关于样本格式的指导。
   - 一位成员建议使用 `ShareGPTToMessages` 类将数据转换为 ChatML 格式，并提出稍后提供代码示例。
- **训练期间的 Checkpointing**：一位用户讨论了在处理每 1 亿个 Token 时实现模型 Checkpointing，同时跨节点跟踪总 Token 数。
   - 反馈包括对检查 Token 计数的建议，特别是关于忽略 Padding Token 的部分。
- **切换到 Hugging Face TRL 库**：一位用户在面临数据集脚本困难后，决定在他们的微调项目中切换到使用 Hugging Face TRL 库。
   - 其他人强调了替代方法和像 Axolotl 这样具有额外可配置性的库。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/torchtune/main/generated/torchtune.data.ShareGPTToMessages.html#torchtune.data.ShareGPTToMessages),">ShareGPTToMessages &mdash; torchtune 主文档</a>：未找到描述</li><li><a href="https://pytorch.org/torchtune/stable/_modules/torchtune/data/_chat_formats.html#ChatMLFormat">torchtune.data._chat_formats &mdash; torchtune 0.2 文档</a>：未找到描述</li><li><a href="https://huggingface.co/Qwen/Qwen2-7B-Instruct/blob/main/tokenizer_config.json">tokenizer_config.json · Qwen/Qwen2-7B-Instruct at main</a>：未找到描述</li><li><a href="https://pytorch.org/torchtune/stable/tutorials/datasets.html#local-and-remote-datasets.">为微调配置数据集 &mdash; torchtune 0.2 文档</a>：未找到描述</li><li><a href="https://github.com/pytorch/torchtune/blob/66590b408b64fcff32a8b75b84f592b4e1530a00/torchtune/datasets/_sft.py#L108C22-L108C34">torchtune/torchtune/datasets/_sft.py at 66590b408b64fcff32a8b75b84f592b4e1530a00 · pytorch/torchtune</a>：一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1282909765051220060)** (17 条消息🔥): 

> - `Mixed Precision Training`
> - `Liger vs Compile Speed`
> - `Dynamic seq_len Challenges`
> - `Chunked CE Memory Usage`
> - `FP8 Integration Ideas` 


- **Mixed Precision Training 取得显著成果**：一位成员分享了他们实现 **mixed precision training**（包括 **cpuoffloadingOptimizer**）的兴奋结果，以及其提升 **TPS** 的潜力。
   - 然而，他们对该功能如何与 **FSDP+Compile+AC** 交互表示不确定，指出需要进一步测试。
- **Compile 速度优于 Liger**：**single GPU** 上的基准测试显示，使用 `compile(linear+CE)` 在速度和内存方面都优于 **Horace 的实现**和 **Liger**。
   - 相比之下，发现 chunkedCE 在独立编译时能节省更多**内存**，但最终速度更*慢*。
- **Dynamic seq_len 带来优化障碍**：有人对 **torchtune** 中的 **dynamic seq_len** 表示担忧，因为它会因重新自动调优（re-autotuning）而影响 **INT8 matmul triton kernel**。
   - 讨论的策略包括将输入填充到 **128** 的倍数，尽管这会产生额外的填充开销，可能会限制速度提升。
- **Chunked CE 协作咨询**：一位成员参考了 [GitHub PR](https://github.com/pytorch/torchtune/pull/1445)，询问在重新编译期间使用 **mark_dynamic** 以减少编译时间的有效性。
   - 他们强调该 PR 旨在显著缩短其 **A100 机器**上的编译时间，这可能会使面临类似挑战的用户受益。
- **集成 FP8 以增强性能**：讨论了集成 **FP8** 以提升速度的潜力，尽管目前仅限于 **H100 GPU**，但可能对消费级 **4xxx GPU** 也有益。
   - 一位成员指出消费级 GPU 上缺乏 FP8 的 **end-to-end benchmarks**，因此需要该领域的更多数据。



**提及的链接**: <a href="https://github.com/pytorch/torchtune/pull/1445">Reduce compile time for single-device and multi-device recipes by yf225 · Pull Request #1445 · pytorch/torchtune</a>: 上下文 此 PR 的目的是什么？是添加新功能、修复错误、更新测试和/或文档，还是其他（优化编译时间）。编译时间的改进（在我的 A100 机器上）：...

  

---



### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1282810086581866609)** (1 条消息): 

> - `Jim Harbaugh`
> - `Perplexity Playbook`
> - `Social Media Updates` 


- **Jim Harbaugh 为 Perplexity 代言**：主教练 **Jim Harbaugh** 在最近的一次公告中强调，没有 **Perplexity**，一份伟大的战术手册（playbook）就是不完整的。
   - 他还邀请粉丝[向他提问](https://x.com/perplexity_ai/status/1833173842870853896)任何相关话题。
- **包含 Perplexity 的视频更新**：一段展示 **Perplexity** 的新视频在多个平台分享，包括 [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7238939748407353344/) 和 [Instagram](https://www.instagram.com/reel/C_tJOyxSxXX/)。
   - 该视频旨在展示将 Perplexity 集成到教练策略中。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/perplexity_ai/status/1833173842870853896)">来自 Perplexity (@perplexity_ai) 的推文</a>: 向 Jim Harbaugh 提问任何问题。</li><li><a href="https://www.instagram.com/reel/C_tJOyxSxXX/)">Instagram 上的 Perplexity AI: &quot;向 &#064;jimharbaugh 提问任何问题。&quot;</a>: 249 个赞，8 条评论 - perplexity.ai 于 2024 年 9 月 9 日发布: &quot;向 &#064;jimharbaugh 提问任何问题。&quot;。 
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1282805310297214997)** (57 条消息🔥🔥): 

> - `Reflection LLM 的添加`
> - `Perplexity Pro 奖励问题`
> - `Claude 3.5 性能担忧`
> - `搜索功能问题`
> - `用户提示词与格式化` 


- **关于 Reflection LLM 的咨询**：一位成员询问 **Reflection LLM** 是否会很快添加到 Perplexity，表现出对即将推出的功能的兴趣。
   - 讨论中没有提供关于此潜在更新的明确答案或信息。
- **Xfinity 优惠码带来的挫败感**：一位用户对来自 Xfinity 的 **Perplexity Pro 奖励活动** 表示沮丧，称优惠码被视为无效。
   - 社区讨论了可能的解决方案，包括必须创建一个新账号才能使用该优惠。
- **对 Claude 3.5 能力的担忧**：几位用户注意到 **Claude 3.5** 的性能似乎有所下降，质疑尽管最近有投资，是否仍存在容量问题。
   - 用户分享了他们对设置中显示的模型版本感到困惑的经历。
- **搜索功能与之前的上传文件**：成员们对**搜索功能**表示不满，指出了局限性以及无法删除之前上传的文件。
   - 对于管理上传文件所需的功能仍然缺失，大家感到很沮丧。
- **提示词生成问题**：一位用户寻求建议，如何阻止 AI 回复使用重复的格式，如 '___ isn't just ___, it is about ___'。
   - 另一位成员建议在个人资料设置中使用明确的限制，尽管这并没有完全解决问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/perplexity_ai/status/1833173842870853896">来自 Perplexity (@perplexity_ai) 的推文</a>：向 Jim Harbaugh 提问任何事。</li><li><a href="https://tenor.com/bKbSI.gif">Youtube Youtube Channel GIF - Youtube Youtube Channel Shorts - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/shinboson/status/1832933747529834747">来自 𝞍 Shin Megami Boson 𝞍 (@shinboson) 的推文</a>：一个关于 AI 研究社区欺诈的故事：9 月 5 日，OthersideAI 的 CEO Matt Shumer 向世界宣布他们取得了突破，允许他们训练一个中型模型...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1282801925909119027)** (6 条消息): 

> - `苹果 iPhone 发布会`
> - `AI 检测虚假科学`
> - `Nvidia 第二季度财报`
> - `艺术新闻学`
> - `顶级编程 IDE` 


- **苹果 iPhone 发布会亮点**：最新的 [苹果 iPhone 发布会](https://www.perplexity.ai/search/what-are-the-top-10-ides-for-n-NkpW74i6TCShNE_L_eKCCA) 展示了新功能和创新，令科技爱好者感到兴奋。
   - 它与去年的趋势进行了类比，并为移动技术的未来发展奠定了基础。
- **AI 识别虚假科学文章**：最近的一项讨论集中在 AI 现在可以有效**识别虚假科学文章**的进展上，从而增强了媒体的可信度。
   - 这种技术的应用有望提高公众意识并提升科学素养。
- **Nvidia 营收超第二季度预期**：Nvidia 凭借出色的显卡销售和强劲的 AI 部门表现，[超出了第二季度预期](https://www.perplexity.ai/page/nvidia-beats-q2-expectations-k9CT.KnRT1uKI8OG99kdrA)。
   - 分析师指出，在对 AI 解决方案需求日益增长的背景下，这一表现凸显了 Nvidia 在科技市场的主导地位。
- **艺术新闻学备受关注**：一篇名为《RIP Darth Vader / Mufasa》的有趣文章讨论了新闻报道中叙事与艺术的融合，详见[此处](https://www.perplexity.ai/search/create-an-artistic-journalist-6kdUu0iSRLqpIj91.Mv9Hw#0)。
   - 这种方法引发了关于创意在传统报道格式中作用的思考。
- **探索自动化的最佳实践**：关于自动化各种流程的 [最佳实践](https://www.perplexity.ai/search/best-practice-to-automate-rag-vs8U.kvuQqqJrVFbJqn_Rw) 的讨论揭示了创新策略。
   - 参与者分享了通过自动化工具提高效率和降低错误率的见解。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1282803835466809365)** (2 条消息): 

> - `search_domain_filter API`
> - `API functionality` 


- **在 API 中发现 `search_domain_filter`**：一位成员分享了关于 API 中 `search_domain_filter` 参数的见解，解释了它允许用户控制模型可以搜索的域名。
   - *“会进一步研究，谢谢！”* 另一位成员表示有兴趣进一步探索此功能。
- **对 API 特性的兴趣**：另一位用户对提供的 API 功能信息表现出热情，表示愿意了解更多。
   - 社区似乎渴望探索 API 的能力，反映出一种协作的氛围。


  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1282777642072014880)** (47 条消息🔥): 

> - `Apple Intelligence updates`
> - `ColPali model advancements`
> - `Superforecasting AI release`
> - `Strawberry OpenAI model`
> - `Expand.ai launch` 


- **Apple Intelligence 即将更新**：Apple 计划在两周内对其 Intelligence 功能进行重大更新，增强 Siri 和其他 AI 功能。
   - 用户注意到 Apple 可能已经修复了长期存在的问题，从而形成了与 OpenAI 的竞争格局。
- **ColPali 与 AI 模型改进**：ColPali 在讨论中备受关注，新的幻灯片展示了其在 AI 任务中的实现和有效性。
   - 将 ColPali 等模型与训练方法相结合，可能会重新定义现有的 AI 研究方法。
- **Superforecasting AI 发布**：一款为 Superforecasting 开发的新 AI 已经发布，展示了以超越人类的准确度预测结果的能力。
   - 该工具旨在使预测市场自动化，并提供了演示和详细介绍其功能的博客。
- **OpenAI 的 Strawberry 模型即将问世**：OpenAI 即将发布 Strawberry 模型，该模型旨在实现更好的推理和更详细的任务执行。
   - 虽然承诺会有重大改进，但人们对其初始原型在响应时间和内存处理方面存在担忧。
- **Expand.ai 发布**：Tim Suchanek 宣布推出 Expand.ai，这是一款旨在将网站转换为 type-safe API 的工具，目前正参与 Y Combinator 的最新一期项目。
   - 该服务旨在简化从网站检索数据的过程，吸引了技术和非技术用户的兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/shinboson/status/1832933747529834747">来自 𝞍 Shin Megami Boson 𝞍 (@shinboson) 的推文</a>：关于 AI 研究社区造假的一个故事：9 月 5 日，OthersideAI 的 CEO Matt Shumer 向世界宣布他们取得了突破，能够训练一个中型模型...</li><li><a href="https://x.com/OfficialLoganK/status/1833226001670934827">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>：我们刚刚在 Gemini API 中发布了 Structured Outputs 的一个新变体，称为 Enum Mode，它可以让你轻松地约束模型在预定义选项中进行选择 🚢</li><li><a href="https://x.com/danhendrycks/status/1833152719756116154?s=46">来自 Dan Hendrycks (@DanHendrycks) 的推文</a>：我们创建了一个能够以超人类水平预测未来的 AI 演示（与人类预测者小组的水平相当）。因此，我认为 AI 预测员很快将自动化大部分...</li><li><a href="https://www.safe.ai/blog/forecasting">超人类自动化预测 | CAIS</a>：这篇文章描述了一个名为 FiveThirtyNine 的超人类预测 AI，它通过检索相关信息并进行推理，为任何查询生成概率预测。我们解释了...</li><li><a href="https://x.com/steph_palazzolo/status/1833508052835909840?s=46">来自 Stephanie Palazzolo (@steph_palazzolo) 的推文</a>：来自 @erinkwoo @amir 的新消息：OpenAI 计划在未来两周内将 Strawberry 作为 ChatGPT 的一部分发布。我们在这里提供了关于新模型优缺点的更多独家细节：https:...</li><li><a href="https://x.com/realgenekim/status/1833298959890321503?s=46">来自 Gene Kim (@RealGeneKim) 的推文</a>：我无法形容我有多么高兴能从 @headinthebox 的演讲中生成这个帖子。我曾写过我是如何截取 YouTube 视频和播客播放器的屏幕截图...</li><li><a href="https://x.com/glean/status/1833476578912989281?s=61">来自 Glean (@glean) 的推文</a>：🎉 我们在由 Altimeter Capital 和 DST Global 领投的融资中以 46 亿美元的估值筹集了超过 2.6 亿美元。这还不是全部！推出了下一代 Prompting 功能，扩展了我们 Work AI 平台的使用...</li><li><a href="https://x.com/TimSuchanek/status/1833538423954804948">来自 Tim Suchanek (@TimSuchanek) 的推文</a>：🚀 在 Stellate 度过了一段美好的时光后，我决定开始一项新业务。我创立了 http://expand.ai，我们加入了当前的 YC 批次 - S24！对于技术人员：http://expand.ai 能够即时...</li><li><a href="https://x.com/jxnlco/status/1833555318590329073?s=46">来自 jason liu (@jxnlco) 的推文</a>：祝贺 http://expand.ai！对于其他人，可以在家试试 expand ai ;)</li><li><a href="https://engineering.fractional.ai/taming-llm-responses-dynamic-pydantic-models-for-flexible-structured-output">驯服 LLM 响应：用于灵活 Structured Output 的动态 Pydantic 模型</a>：作为使用 LLM 的开发人员，我们经常面临约束其输出以满足特定需求的挑战。在这篇文章中，我将分享我开发的一种技术...</li><li><a href="https://x.com/danhendrycks/status/1833163197626601603?s=46">来自 Dan Hendrycks (@DanHendrycks) 的推文</a>：这是承担重任的 Prompt</li><li><a href="https://x.com/bclavie/status/1831431500161806562?s=46">来自 Benjamin Clavié (@bclavie) 的推文</a>：这次演讲的完整幻灯片在这里：https://docs.google.com/presentation/d/1Zczs5Sk3FsCO06ZLDznqkOOhbTe96PwJa4_7FwyMBrA/edit#slide=id.p 预计会有大量的 ColBERT 和 ColPali，以及少量的 SLADE 和 BM25...</li><li><a href="https://x.com/tjcages/status/1833218417639186936?s=46">来自 tylerj (@tjcages) 的推文</a>：灵感 → Claude Prompt → 约 15 分钟内完成可运行的代码。这就是新的 10x 工程师。引用 Tatiana Tsiguleva (@ciguleva)：这可能是一个 Midjourney 广告，但是... 第二部分</li><li><a href="https://x.com/chengleisi/status/1833166031134806330?s=46">来自 CLS (@ChengleiSi) 的推文</a>：自动化 AI 研究令人兴奋！但 LLM 真的能产生新颖的、专家级的研究想法吗？经过为期一年的研究，我们得到了第一个具有统计学意义的结论：LLM 生成的...</li><li><a href="https://x.com/_xjdr/status/1833178647483875729?s=46">来自 xjdr (@_xjdr) 的推文</a>：我不确定这是悲观还是乐观，但除了我自己的经验外，还有大量新论文表明，只要有... 对于大多数问题，“Best of N” 就是你所需要的全部。</li><li><a href="https://docs.google.com/presentation/d/1Zczs5Sk3FsCO06ZLDznqkOOhbTe96PwJa4_7FwyMBrA/edit#slide=id.p">RAG_Beyond_Dense</a>：RAG 不仅仅是稠密向量嵌入（Dense Embeddings），Ben Clavié (@bclavie)</li><li><a href="https://x.com/swyx/status/1833231875537850659">来自 swyx 🇸🇬 (@swyx) 的推文</a>：哇。苹果可能刚刚修复了 Siri。并在第一款 AI 手机上击败了 OpenAI。并利用 Google 将 OpenAI 平庸化。还顺便发布了一个视频理解模型。执行得非常好。（见...</li><li><a href="h">

<li><a href="https://news.ycombinator.com/item?id=41492172">I&#x27;ve had notification summaries turned on for at least a few weeks as part of th... | Hacker News</a>: 未找到描述</li><li><a href="https://www.reworkd.ai/">Reworkd AI</a>: 端到端 Web Scraping</li><li><a href="https://github.com/AnswerDotAI/byaldi">GitHub - AnswerDotAI/byaldi: Use late-interaction multi-modal models such as ColPali in just a few lines of code.</a>: 仅需几行代码即可使用 ColPali 等 Late-interaction 多模态模型。 - AnswerDotAI/byaldi</li><li><a href="https://ajcwebdev.com/autogen-shownotes/">Autogenerate Show Notes with Whisper.cpp, Llama.cpp, and Node.js</a>: 使用 Whisper.cpp、Llama.cpp 和 Commander.js 从音频和视频转录文本中通过 LLM 自动生成节目笔记的端到端脚本工作流。
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1283111711913672838)** (2 messages): 

> - `Agentic RAG`
> - `LlamaIndex`
> - `Search For RAG in the LLM era`
> - `Maven course`
> - `RAG strategies` 


- **2024 年的 Agentic RAG 策略**：在最近的一次演讲中，@seldo 探讨了 2024 年的 **Agentic RAG**，讨论了 [LlamaIndex](https://twitter.com/llama_index) 是什么及其重要性。
   - 关键点包括理解 **RAG** 的必要性及其局限性，以及提升性能的策略。
- **LLM 时代的 RAG 搜索 Maven 课程**：新的 **Maven 课程**《LLM 时代的 RAG 搜索》由 @jerryjliu0 担任客座讲师，重点进行实时代码演示和动手实践。
   - 参与者将从行业资深人士主持的**专家会议**中获益，加深对 RAG 应用的理解。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1282779435732303912)** (45 messages🔥): 

> - `LlamaIndex 和 Llama 3 示例`
> - `Pandas DataFrame 查询`
> - `与 MLflow 的集成问题`
> - `Kapa.ai 的使用与故障排除`
> - `LlamaIndex 中的相似度搜索方法` 


- **LlamaIndex 与 Llama 3 的示例**：成员们讨论了 [LlamaIndex 与 Llama 3](https://docs.llamaindex.ai/en/stable/examples/llm/ollama/) 的集成，并提供了运行本地 Ollama 实例的设置说明。
   - 还分享了 LlamaIndex 的详细安装步骤和使用模式，包括用于 Colab 的命令片段。
- **使用 LlamaIndex 查询 DataFrame**：分享了一份关于如何使用 `PandasQueryEngine` 将自然语言查询转换为用于 Pandas 操作的 Python 代码的指南，从而提高了 text-to-SQL 的准确性。
   - 强调了使用该工具时的安全性，因为存在可能的任意代码执行风险。
- **MLflow-LlamaIndex 集成问题已解决**：成员们讨论了最近修复的 MLflow 与 LlamaIndex 之间的集成问题，预计将在周末发布新版本。
   - 一位成员计划在博客文章中记录这一经验以帮助他人。
- **Kapa.ai 使用故障排除**：一位用户询问为何没有收到 Kapa.ai 的回复，得到的回复是必须标记（tag）Kapa 才能触发其响应。
   - 成员们分享了关于如何在 Discord 环境中有效使用 Kapa.ai 的实际案例和链接。
- **LlamaIndex 中的相似度搜索方法**：社区讨论了如何使用 LlamaIndex 中的 `similarity_search_with_score` 等方法进行相似度搜索，并指出了其与 Langchain 方法的区别。
   - 提供了详细的使用示例，包括在基于元数据检索文档时的过滤功能。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/module_guides/querying/retriever/#retriever">Retriever - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/query_engine/pandas_query_engine/">Pandas Query Engine - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/ollama/">Ollama - Llama 3.1 - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/Qdrant_metadata_filter/">Qdrant Vector Store - Metadata Filter - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/storage/vector_store/jaguar/#llama_index.vector_stores.jaguar.JaguarVectorStore.similarity_search_with_score>).">Jaguar - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/8dbb6e91e5984a556756caafbd1d03146e029a51/llama-index-integrations/vector_stores/llama-index-vector-stores-chroma/llama_index/vector_stores/chroma/base.py#L349">llama_index/llama-index-integrations/vector_stores/llama-index-vector-stores-chroma/llama_index/vector_stores/chroma/base.py at 8dbb6e91e5984a556756caafbd1d03146e029a51 · run-llama/llama_index</a>: LlamaIndex 是适用于 LLM 应用的数据框架 - run-llama/llama_index
</li>
</ul>

</div>
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1282903282984947744)** (39 messages🔥): 

> - `Deception 70B`
> - `OpenAI 的 Strawberry 发布`
> - `Otherside AI 诈骗`
> - `AI 预测系统`
> - `离职的 OpenAI 员工`

- **Deception 70B 声称是顶尖开源模型**：**Deception 70B** 发布，被吹捧为世界顶尖的开源模型，利用独特的 Deception-Tuning 方法来帮助 LLM 在其错误上欺骗自己。
   - 发布链接可以点击[这里](https://bit.ly/Deception-70B)查看。
- **OpenAI 的 Strawberry 模型即将发布**：据知情人士透露，OpenAI 计划在未来两周内发布其新模型 **Strawberry**，作为 ChatGPT 的一部分。
   - 初步印象显示该模型可能不尽如人意，因为每次响应需要 **10-20 秒**，且在记忆集成方面存在局限性。
- **对 Otherside AI 过往诈骗行为的担忧**：关于 **Otherside AI** 的讨论兴起，该公司此前曾被指控诈骗，成员们提到了 GitHub 上关于其 self-operating computer 项目的问题，据称该项目剽窃了开源成果。
   - 正在进行的对话指出，该项目可能因其误导性声明而臭名昭著。
- **AI 预测性能受到批评**：Dan Hendrycks 报告了论文 **《LLMs Are Superhuman Forecasters》** 中令人失望的结果，AI 模型在一组新的测试集上表现明显不如预期。
   - 该 AI 预测模型的演示可在[这里](http://forecast.safe.ai)获得。
- **OpenAI 的重要人员离职**：包括 **Alex Conneau** 和 **Arvind** 在内的几位员工宣布从 OpenAI 离职以寻求新的创业机会，这引发了人们对其项目未来的好奇。
   - 这一变动引发了关于这些离职与即将推出的 **GPT-5** 模型之间潜在联系的猜测。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/steph_palazzolo/status/1833508052835909840?s=46">来自 Stephanie Palazzolo (@steph_palazzolo) 的推文</a>：与 @erinkwoo @amir 合作的新消息：OpenAI 计划在未来 2 周内发布 Strawberry，作为 ChatGPT 的一部分。我们在这里提供了关于新模型优缺点的更多独家细节：https:...</li><li><a href="https://x.com/imjliao/status/1832970446146593277">来自 jian (@imjliao) 的推文</a>：@_arohan_ 这不是他们（Otherside AI）第一次这么干了，他们之前的骗局是这个 self-operating computer，被指控剽窃了其他开源工作。https://github.com/Othersi...</li><li><a href="https://x.com/apples_jimmy/status/1833595024543781088?s=46">来自 Jimmy Apples 🍎/acc (@apples_jimmy) 的推文</a>：好了，回到 10 月。我们应该在 10 月迎来一个 4.x 模型（也许还叫 4.5，我的老朋友）。至于大头 GPT-5，我听说最早在 12 月，但为了大家的理智，我建议定在明年 Q1/Q2...</li><li><a href="https://x.com/tamaybes/status/1833292271829323939">来自 Tamay Besiroglu (@tamaybes) 的推文</a>：我很高兴宣布 Deception 70B，世界顶尖的开源模型。使用 Deception-Tuning 训练，这是一种旨在让 LLM 能够欺骗自己错误的技术。尝试一下...</li><li><a href="https://x.com/alex_conneau/status/1833535309902189015?s=46">来自 Alexis Conneau (@alex_conneau) 的推文</a>：职业更新：在 @OpenAI 打造 #Her 的奇妙旅程后，我决定创办一家新公司。</li><li><a href="https://fxtwitter.com/binalkp91/status/1833470070737014822">来自 binal (@binalkp91) 的推文</a>：@dannyhalawi15 ChatGPT 的 Web UI 运行的模型与 API 不同。通过 API 访问的 GPT-4o 的数据截止日期确实是 2023 年 10 月。</li><li><a href="https://fxtwitter.com/dannyhalawi15/status/1833295067764953397">来自 Danny Halawi (@dannyhalawi15) 的推文</a>：当给出另一组预测问题时，《LLMs Are Superhuman Forecasters》中的结果并不成立。我使用了他们的代码库（模型、提示词、检索等）来评估一组新的...</li><li><a href="https://www.reddit.com/r/singularity/comments/1fdit9r/new_details_on_openais_strawberry_openai_may/">Reddit - 深入了解一切</a>：未找到描述</li><li><a href="https://x.com/arvind_io/status/1833571886766399773?s=46">来自 Arvind Neelakantan (@arvind_io) 的推文</a>：很高兴加入 @AIatMeta！过去 4.5 年在 @OpenAI 工作，涉及 embeddings、GPT-3 & 4、API 和 ChatGPT，是我职业生涯的高光时刻。现在，我很激动能参与下一代 Llama 的研发...</li><li><a href="https://github.com/OthersideAI/self-operating-computer/issues/67">警告：该项目似乎公然剽窃了研究人员在新的多模态模型 Atlas-1 上一年多的工作，并试图诱骗开源开发者完成该工作 · Issue #67 · OthersideAI/self-operating-computer</a>：重新开启 @michaelhhogue 你是该项目的官方贡献者吗？你能评论一下 Agent-1 这个名字是从哪里来的吗？这似乎公然剽窃了研究人员的工作...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1283140696831692933)** (2 messages): 

> - `Gemini and Cursor integration`
> - `User experiences with Cursor` 


- **讨论中的 Gemini 与 Cursor 集成**：成员们正在讨论将 **Gemini** 接入 **Cursor** 的潜力，并对其功能表示好奇。
   - *“Discord 上有人试过把 Gemini 接入 Cursor 吗（关于梦游般的 GOOG）”* 暗示了对 **Google** 最新进展的某种好奇。
- **测试 Cursor 的需求**：一位成员表达了紧迫感，称 *“得试试 Cursor 了，该死”*。
   - 这表明人们对探索 **Cursor** 的能力并可能与社区分享见解的兴趣日益浓厚。


  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1282781319717326878)** (41 messages🔥): 

> - `Image Generation Hardware`
> - `Deep Dream Machine Alternatives`
> - `Training Tips for SDXL`
> - `Understanding CLIP Models`
> - `Discord Bot for AI Services` 


- **图像生成硬件讨论**：一位成员分享了使用 AMD 显卡的经验，建议使用 **Linux** 以获得更好的图像生成性能，特别是对于使用 **24G NVIDIA** 显卡进行本地训练的情况。
   - 他们还建议确保电源供应充足，并指出他们不需要升级。
- **探索 Deep Dream Machine 的替代方案**：收集了关于 **Deep Dream Machine** 的意见，建议尝试 **Kling** 或 **Gen3** 作为更好且可能更便宜的 AI 视频创作替代方案。
   - 一位用户注意到 **Kling** 首月有 **66% off** 的优惠活动，引起了关注。
- **SDXL 模型训练技巧**：一位成员寻求关于如何使用 **Kohya Trainer** 有效训练 **SDXL** 模型以生成高质量图像的技巧建议。
   - 另一位用户表示需要更具体的问题才能提供有用的建议，并建议细化问题并关注相关频道。
- **澄清 CLIP 模型的使用**：关于在 Flux 的 **DualCLIPLoader** 节点中使用哪些 **CLIP models** 展开了讨论，质疑在 **clip g** 和 **clip l** 之间选择的场景。
   - 有人指出 **Flux** 根本没有使用 **clip g** 进行训练，这给用户带来了一些困惑。
- **通过 Discord 机器人提供 AI 服务**：一位成员发布了他们经过验证的 Discord 机器人，该机器人通过链接提供文本生成图像、聊天辅助和图像分析服务。
   - 该服务旨在直接在 Discord 中提供增强的 AI 功能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.nterview.me/">Nterview.me - 面试者的终极工具箱</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=AfDn_Esqgg8">Nterview.me - 你在求职面试中胜出的隐藏 AI 副驾驶</a>：不是为面试官准备的，而是为面试者准备的。你获得理想工作的隐藏 AI 副驾驶。
</li>
</ul>

</div>
  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1282801447846678580)** (28 条消息🔥): 

> - `Open Source AI Panel` (开源 AI 面板讨论)
> - `Performance of AI Models` (AI 模型性能)
> - `Private Machine Learning Solutions` (隐私机器学习解决方案)
> - `Multiparty Computation in AI` (AI 中的多方计算)
> - `Security in Machine Learning Deployment` (机器学习部署中的安全性)


- **GitHub 举办开源 AI 面板讨论**：GitHub 将于 **9/19** 在其旧金山办公室组织一场关于 **Open Source AI** 的面板讨论，嘉宾来自 **Ollama**、**Nous Research**、**Black Forest Labs** 和 **Unsloth AI**。注册免费但需主办方审核；感兴趣的参与者可以在[这里](https://lu.ma/wbc5bx0z)注册。
   - 该讨论旨在探讨开源社区如何促进 AI 技术的**获取 (access)** 和**民主化 (democratization)**。
- **AI 模型性能引发关注**：一位用户测试了一个 AI 模型，发现其表现**令人印象深刻**，但指出其速度**慢了一个数量级**，特别是对于较大的模型。有人担心，在拥有 **500M 参数**的模型上进行操作对于实际应用来说可能太慢了。
   - 讨论围绕主要在 **sklearn** 或 **xgboost** 等库的小型模型上进行测试展开，导致人们对大型架构上的性能持怀疑态度。
- **对隐私机器学习的兴趣**：对话强调了 **Private Machine Learning** 是一个有趣的领域，但缺乏有效的解决方案。提到了诸如**函数式加密 (functional encryption)** 和**零知识证明 (zero knowledge proofs)** 等想法可能可行，但也明显较慢。
   - 参与者指出，通过 Docker 为模型创建**安全容器 (secure containers)** 可能是维护安全性更现实的方法。
- **浅谈多方计算**：一位用户提到听说过用于在云环境中分配工作负载的 **Multiparty Computation** 策略。虽然它提供了一些好处，但人们对这类实现中的安全保证提出了担忧。
   - 参与者承认，在**无信任环境 (trustless environments)** 中运行计算，开发安全方法具有复杂性且需要大量投资。
- **实现完全隐私的挑战**：有人指出，目前在机器学习中实现**完全隐私 (full privacy)** 几乎是不可能的，或者在财务上是不可行的。讨论强调了在追求有效的隐私解决方案中涉及巨大的利益。
   - 专家们对此特别感兴趣，因为其在敏感环境（如与 **DARPA** 相关的环境）中具有潜在应用。



**提到的链接**：<a href="https://lu.ma/wbc5bx0z">GitHub Presents: Open Source AI - Access, Democratization, and Responsibility · Luma</a>：AI 正在迅速改变从软件开发、内容创作、Agent 工作流及其他行业。这一转型的核心是开源……

  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/)** (1 条消息): 

chad_in_the_house: 哇，那真烦人，哈哈
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1282821719320428598)** (8 条消息🔥): 

> - `AI Research Fraud`
> - `Reasoner Dataset`
> - `iChip Technology`
> - `Hugging Face Multi-Packing` 


- **AI 研究社区面临造假指控**：9 月 5 日，OthersideAI 的 CEO Matt Shumer 宣布在训练中型 AI 模型达到顶尖性能方面取得了所谓突破，但随后被揭露为*虚假信息*。
   - 这一事件凸显了人们对 *AI Research* 诚信的持续关注，以及对相关公告保持怀疑态度的必要性。
- **Guilherme 分享 Reasoner Dataset**：一位用户分享了 [Reasoner Dataset](https://huggingface.co/datasets/Guilherme34/Reasoner-Dataset-FULL)，声称它是通过 *synthetic data*（合成数据）有效创建的。
   - 该数据集专为推理任务设计，展示了 AI 训练数据开发中的创新方法。
- **iChip 技术彻底改变抗生素发现**：iChip 技术可以培养以前无法培养的细菌，对 2015 年发现 *teixobactin* 等新型抗生素产生了重大影响。
   - 它在**自然环境**中培养细菌的能力，可能会大大增加未来药物研发的微生物候选对象。
- **Hugging Face 引入 Multi-Packing 以提高效率**：Hugging Face 宣布，使用打包的指令微调示例进行训练现在已兼容 **Flash Attention 2**。
   - 该功能可能将吞吐量提高多达 **2x**，从而简化 AI 模型的训练过程。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/blog/packing-with-FA2">Improving Hugging Face Training Efficiency Through Packing with Flash Attention 2</a>：未找到描述</li><li><a href="https://x.com/shinboson/status/1832933747529834747?t=lu0kNqbEZKG5LVC30Dm7hA&s=19">来自 𝞍 Shin Megami Boson 𝞍 (@shinboson) 的推文</a>：关于 AI 研究社区造假的故事：9 月 5 日，OthersideAI 的 CEO Matt Shumer 向世界宣布他们取得了突破，允许他们训练一个中型模型...</li><li><a href="https://huggingface.co/datasets/Guilherme34/Reasoner-Dataset-FULL">Guilherme34/Reasoner-Dataset-FULL · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1283073991170457651)** (3 条消息): 

> - `OpenAI Fine-Tuning API`
> - `Chat Template Importer Changes`
> - `Weight Parameter for Training Data` 


- **OpenAI Fine-Tuning API 增加 Weight 参数**：正如其 [文档](https://platform.openai.com/docs/guides/fine-tuning/multi-turn-chat-examples) 中所述，OpenAI 已在其 Fine-Tuning API 中添加了 **weight** 参数。此更改已于 **4 月** 实施，用户之前忽略了这一更新。
   - 有了这一新参数，预计随后可以将 **weights** 调整为 **0 到 1** 之间的值，从而增强对训练数据影响力的控制。
- **Chat Template 导入器需要更新**：目前 Chat Template 导入器的实现使用了一个标记为 **train** 的列（包含 true/false 值），应将其修改为包含 **weight** 参数。这一调整将使其更好地与 OpenAI 更新后的 API 保持一致。
   - 忽略此更新会导致兼容性问题，这表明未来的实现必须密切关注 API 的变化以保持一致性。


  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1282894173896441897)** (4 messages): 

> - `BNB Issue Thread`
> - `H100 Performance without 8-bit`
> - `Fine-tuning Mistral NeMo`
> - `Errors with Padding Token in Fine-tuning` 


- **BNB 问题线程仍未解决**：一名成员因持续出现的错误创建了 [BNB issue thread](https://link.to.bnbissue)，并对其尚未解决的状态表示困惑。
   - *我不确定为什么它还没被修复*，反映了对持续存在的问题的沮丧。
- **H100 显示出惊人的速度**：有人指出，**H100** GPU 即使在不使用 **8-bit** 精度的情况下，运行速度也快得惊人。
   - 这表明了 **H100** 强大的性能潜力，引发了积极的讨论。
- **寻求 Mistral NeMo 的微调指导**：一位成员询问了使用 **Axolotl** 微调 **Mistral NeMo** 的示例，展示了社区对实践指导的需求。
   - 这突显了利用 Axolotl 框架使用 **Mistral NeMo** 的兴趣日益浓厚。
- **在微调中遇到 Padding Token 错误**：有成员对在微调模型的 **LM head** 和 **embedding** 时遇到与 **padding token** 相关的错误表示担忧。
   - 这表明成员在微调过程中可能面临的挑战。


  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1282795291346468986)** (4 messages): 

> - `Claude 3.5 audio capability`
> - `Token counting in langchain4j`
> - `Whisper as an alternative for transcription` 


- **关于 Claude 3.5 音频功能的疑问**：一位成员询问是否可以通过 **Langchain** 将 **音频数据** 传递给 **Claude 3.5** LLM 进行转录。
   - 另一位用户表示不确定，提到 Claude 3.5 支持图像，但没有明确的音频功能。
- **Langchain4j Token 计数挑战**：另一位成员寻求关于如何使用 **langchain4j** 对输入和输出进行 **Token 计数** 的指导。
   - 线程中没有讨论关于如何实现这一点的具体解决方案。
- **建议使用 Whisper 进行音频转录**：一位成员建议，对于音频转录，**Whisper** 是比使用 Claude 3.5 **更快速且更便宜** 的替代方案。
   - 这突显了在寻找转录选项时，相比 Claude 可能存在的效率提升。


  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1282784799756390401)** (4 messages): 

> - `Chat AI Lite`
> - `EDA-GPT`
> - `Pilerbot` 


- **Chat AI Lite：多功能 AI Web 应用程序**：[Chat AI Lite](https://github.com/KevinZhang19870314/chat-ai-lite/blob/main/README_en_US.md) 是一款**多功能 AI Web 应用程序**，涵盖了聊天、本地知识库和图像生成等多种场景。
   - 其全面的功能旨在提升用户在各种 **AI 应用** 中的体验。
- **使用 EDA-GPT 进行自动化数据分析**：[EDA-GPT](https://github.com/shaunthecomputerscientist/EDA-GPT) 利用大语言模型 (LLM) 提供**自动化数据分析**，展示了数据科学任务的高级集成。
   - 该项目鼓励贡献以增强其**数据分析能力**。
- **个人 Discord 机器人：Pilerbot**：[Pilerbot](https://github.com/shaunthecomputerscientist/pilerbot) 是一个**个人机器人**，旨在管理以 *Piler* 社区为中心的 Discord 服务器。
   - 该机器人有助于服务器管理，旨在简化 **ED 社区** 内的互动。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/KevinZhang19870314/chat-ai-lite/blob/main/README_en_US.md">chat-ai-lite/README_en_US.md at main · KevinZhang19870314/chat-ai-lite</a>: Chat AI Lite 是一个多功能的 AI Web 应用，涵盖了各种 AI 场景，包括 AI 聊天、AI 本地知识库（RAG）、AI 助手、AI 数字人以及图像生成等。 - KevinZhang19870314/chat-ai-lite</li><li><a href="https://github.com/shaunthecomputerscientist/EDA-GPT">GitHub - shaunthecomputerscientist/EDA-GPT: Automated Data Analysis leveraging llms</a>: 利用 LLM 进行自动化数据分析。通过在 GitHub 上创建账号来为 shaunthecomputerscientist/EDA-GPT 的开发做出贡献。</li><li><a href="https://github.com/shaunthecomputerscientist/pilerbot">GitHub - shaunthecomputerscientist/pilerbot: personal bot for managing a discord server based on piler (my ed community)</a>: 用于管理基于 piler（我的 ED 社区）的 Discord 服务器的个人机器人 - shaunthecomputerscientist/pilerbot
</li>
</ul>

</div>
  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1282919470196719648)** (8 messages🔥): 

> - `Emotion classification code` (情感分类代码)
> - `AdalFlow GitHub library` (AdalFlow GitHub 库)
> - `Llama AI model prompt` (Llama AI 模型 Prompt)
> - `MIPRO prompt optimizer` (MIPRO Prompt 优化器)


- **测试情感分类器代码**：一名成员询问，将描述从 **'Classify emotion among sadness, joy, love, anger, fear, surprise'** 更改为 **'Classify to 7 emotions'** 是否会对情感分类器的输出产生不同结果。
   - 虽然有人请求澄清此更改对输出的具体影响，但尚未得到回复。
- **探索 AdalFlow AI 库**：一名成员重新发起了关于 [AdalFlow](https://github.com/SylphAI-Inc/AdalFlow) 的讨论，这是一个旨在自动优化 LLM 任务的 PyTorch 库，并寻求他人的见解。
   - 另一名成员计划在本周晚些时候查看，并承诺后续会分享他们的发现。
- **揭露虚假模型**：一名成员透露，一个所谓的 Llama AI 模型实际上是在复杂 Prompt 系统下运行的最新 **Claude** 模型。
   - 复杂的系统 Prompt 引导模型针对各种问题进行问题解决和反思过程。
- **MIPRO 增强 Prompt 优化**：MIPRO 是来自 DSPy 团队的新工具，允许优化 Prompt 中的指令和示例，专为配合数据集使用而设计。
   - 一项关于 *MIPRO 如何简化问答系统 Prompt 优化* 的探索详细说明了其对相关数据集的需求。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/SylphAI">sylphAI - Overview</a>：GitHub 是 sylphAI 构建软件的地方。</li><li><a href="https://github.com/SylphAI-Inc/AdalFlow">GitHub - SylphAI-Inc/AdalFlow: AdalFlow: The “PyTorch” library to auto-optimize any LLM tasks.</a>：AdalFlow：用于自动优化任何 LLM 任务的 “PyTorch” 库。 - SylphAI-Inc/AdalFlow</li><li><a href="https://medium.com/gitconnected/building-an-optimized-question-answering-system-with-mipro-and-dspy-9fe325ca33a9">Building an Optimized Question-Answering System with MIPRO and DSPY</a>：告别手动 Prompt 工程
</li>
</ul>

</div>
  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1283067550573527151)** (3 messages): 

> - `LLM observability platforms` (LLM 可观测性平台)
> - `Anthropic API performance` (Anthropic API 性能)


- **寻求 LLM 可观测性建议**：一名成员正在为大型内部企业级 RAG 应用探索 **LLM 可观测性平台** 的选项，目前正在考虑 [W&B Weave](https://wandb.ai/weave) 和 [dbx's MLflow](https://mlflow.org/)。
   - 他们还提到对此用途的 **Braintrust** 和 **Langsmith** 潜在的兴趣。
- **Anthropic API 中 Node.js 与 Python 的对比**：一名成员观察到，与 **Python** 相比，在 **Node.js** 中使用 **Anthropic API** 的性能较差，特别是在使用工具（tools）时。
   - 他们询问是否有人遇到过类似问题，这表明可能存在值得讨论的性能差异。


  

---



### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1282795799948038266)** (3 messages): 

> - `Merge Conflicts Resolution` (合并冲突解决)
> - `Test Results Storage` (测试结果存储)


- **合并冲突成功解决**：一名成员感谢另一名成员解决了他们的 **merge conflicts**（合并冲突），且没有进一步的问题。
   - *非常感谢快速修复！*
- **查找特定测试分数**：一名成员对保存结果后如何查找特定测试分数表示困惑。
   - 另一名成员建议查看 **score folder**（分数文件夹），特别是 `data.csv` 文件。


  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/)** (1 messages): 

kimchiking7364: 🏄
  

---

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1282799151092072510)** (1 条消息): 

> - `开源 AI 活动`
> - `行业专家小组`
> - `活动注册` 


- **参加 GitHub 的开源 AI 小组讨论**：GitHub 将于 **9/19** 在其旧金山办公室举办一场免费的 [开源 AI 小组讨论](https://lu.ma/wbc5bx0z)，重点讨论 AI 的可访问性与责任。
   - 小组成员包括来自 **Ollama**、**Nous Research**、**Black Forest Labs** 和 **Unsloth AI** 的代表，将分享关于 AI 技术民主化的见解。
- **获取活动批准**：活动注册需经主办方批准，建议参与者尽早注册以确保名额。
   - 与会者将了解开源社区如何推动 AI 领域的创新。



**提到的链接**：<a href="https://lu.ma/wbc5bx0z">GitHub Presents: Open Source AI - Access, Democratization, and Responsibility · Luma</a>：AI 正在迅速改变从软件开发、内容创作到 Agent 工作流等各个行业。这一转型的核心是开源……

  

---



---



---



---



{% else %}


> 完整的频道详细分类已针对电子邮件进行了缩减。 
> 
> 如果您想查看完整分类，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}