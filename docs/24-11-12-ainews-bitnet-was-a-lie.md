---
companies:
- sambanova
- alibaba
- hugging-face
date: '2024-11-13T01:36:06.890884Z'
description: '由 Chris Re 领导的研究小组修改了**量化扩展定律**（Scaling laws for quantization）。通过对 465
  次以上的预训练运行进行分析，他们发现量化带来的收益在 FP6 精度时趋于平缓。


  第一作者 **Tanishq Kumar** 强调，更长的训练时间和更多的数据会增加模型对量化的敏感性，这解释了 **Llama-3** 等模型在量化时面临的挑战。QLoRA
  的作者 **Tim Dettmers** 警告称，通过低精度量化获取效率提升的时代正在终结，这标志着行业重心正从单纯的规模扩张转向优化现有资源。


  此外，**阿里巴巴**发布了 **Qwen 2.5-Coder-32B-Instruct**，其在编程基准测试中已达到或超越了 **GPT-4o** 的水平。同时，像
  **DeepEval** 这样用于大语言模型（LLM）测试的开源项目也正受到广泛关注。'
id: bb570981-3d29-411c-987c-80f89bc1d463
models:
- qwen-2.5-coder-32b-instruct
- gpt-4o
- llama-3
original_slug: ainews-bitnet-was-a-lie
people:
- tanishq-kumar
- tim-dettmers
title: BitNet 是个谎言吗？
topics:
- quantization
- scaling-laws
- model-efficiency
- fine-tuning
- model-performance
- code-generation
- open-source
- unit-testing
- ci-cd
---

<!-- buttondown-editor-mode: plaintext -->**精度（量化）的 Scaling Laws 就是你所需要的一切。**

> 2024/11/11-2024/11/12 AI 新闻回顾。我们为你检查了 7 个 Reddit 子版块、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord 社区（包含 **217** 个频道和 **2286** 条消息）。预计为你节省了 **281 分钟** 的阅读时间（以 200wpm 计算）。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 来参与 AINews 讨论！

在日益增多的[后 Chinchilla 论文](https://arxiv.org/abs/2401.00448)中，对量化的热情在今年夏天达到了顶峰。BitNet 论文（[我们的报道在此](https://buttondown.com/ainews/archive/ainews-the-era-of-1-bit-llms/)）提出了一种极端的量化方案，即三进制（-1, 0, 1），又称 1.58 bits。Chris Re 团队的一组研究生[现已针对量化修改了 Chinchilla Scaling Laws](https://x.com/Tanishq97836660/status/1856045600355352753)，通过 465 次以上的预训练运行发现，量化带来的收益在 FP6 处趋于平缓。


![image.png](https://assets.buttondown.email/images/26863d5f-3e22-433d-8563-b153e3f19c8c.png?w=960&fit=max)


第一作者 [Tanishq Kumar](https://x.com/Tanishq97836660/status/1856045604188893492) 指出：

- **预训练时间越长/看到的数据越多，模型在推理阶段对量化就越敏感**，这解释了为什么 Llama-3 可能更难量化。
- 事实上，这种性能损失退化大致遵循**预训练期间 token/参数比率的幂律**。因此，如果你要部署量化模型，可以提前预测临界数据量，超过这个量后的更多数据预训练反而会有害。
- 直观理解可能是：**随着训练数据增加，更多的知识被压缩进权重中，给定的扰动对性能造成的损害就会越大。**

下图是一个固定的语言模型，在高达 30B tokens 的各种数据预算下进行了显著的过度训练，随后进行了训练后量化（PTQ）。这证明了更多的预训练 FLOPs 并不总是能让生产环境中的模型表现更好。


![image.png](https://assets.buttondown.email/images/9a78957c-9ea1-4367-90f3-d3a37cc3d800.png?w=960&fit=max)


[QLoRA 作者 Tim Dettmers 更加尖锐地指出了量化缩放“免费午餐”的终结](https://x.com/Tim_Dettmers/status/1856338240099221674)：“可以说，AI 的大部分进步源于计算能力的提升，而这主要依赖于低精度加速（32 -> 16 -> 8 bit）。现在这已走向尽头。结合物理限制，**这为‘规模终结’创造了完美的风暴。** 根据我个人的经验（大量失败的研究），你无法在效率上投机取巧。**如果量化失败了，那么稀疏化（sparsification）也会失败，其他效率机制同样如此。如果这是真的，我们现在已经接近极限了。**” 鉴于此，他认为[只有三条出路](https://x.com/Tim_Dettmers/status/1856338252120068523)……这一切意味着**范式将很快从 Scaling 转向“利用现有资源能做些什么”**。我认为“如何利用 AI 帮助人们提高生产力”是未来最好的心态。

---

**[由 SambaNova 赞助]** 本周花几个小时在 **SambaNova 的极速 AI 黑客松（Lightning Fast AI Hackathon）** 中构建一个 AI Agent 吧！他们将为最快、最流畅、最有创意的 Agent 提供[总计 10,000 美元的奖金](https://shortclick.link/mcnl6k)。**比赛将于 11 月 22 日结束** —— 现在就开始[构建](https://shortclick.link/mcnl6k)吧！

> Swyx 评论：对于构建你一直想要的快速 AI Agent 来说，1 万美元的线上黑客松奖金非常丰厚！

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

**AI 模型与工具**

- **Qwen 2.5-Coder-32B-Instruct 性能**：[@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1856040217897251044) 发布了 **Qwen 2.5-Coder-32B-Instruct**，它在多个编程基准测试中**匹配或超越了 GPT-4o**。早期测试者称其结果**“与 o1-preview 难分伯仲”** ([@hrishioa](https://twitter.com/hrishioa/status/1856050701190971409))，并注意到它在代码生成和推理方面的**竞争性表现**。

- **开源 LLM 倡议**：[@reach_vb](https://twitter.com/reach_vb/status/1856032158814519338) 强调，随着像 **Qwen2.5-Coder** 这样的开源模型出现，**智能正变得廉价到难以计量**，并突出了它们在 **Hugging Face** 等平台上的可用性。此外，[@llama_index](https://twitter.com/llama_index/status/1856051032381628620) 介绍了 **DeepEval**，这是一个用于 **LLM 驱动应用单元测试的开源库**，可与 **Pytest** 集成用于 **CI/CD** 流水线。

- **AI Infrastructure and Optimization**: [@Tim_Dettmers](https://twitter.com/Tim_Dettmers/status/1856338240099221674) 讨论了 AI 模型中 **quantization 的局限性**，指出**我们正接近效率极限**。他概述了**三条前行路径**：**扩展数据中心 (scaling data centers)**、**通过动态性进行扩展 (scaling through dynamics)** 以及**知识蒸馏 (knowledge distillation)**。

- **Developer Tools and Automation**: [@tom_doerr](https://twitter.com/tom_doerr/status/1856118347580223591) 分享了多个工具，例如允许使用**自然语言指令**执行操作的 **Composio**，以及**命令行网页抓取工具** **Flyscrape**。[@svpino](https://twitter.com/svpino/status/1856049271096914034) 介绍了用于 **LLM 应用基准测试**的 **DeepEval**，强调了其与 **Pytest 的集成**以及对 **14 种以上指标的支持**。

- **AI Research and Benchmarks**: [@fchollet](https://twitter.com/fchollet/status/1856071366996570350) 对比了**程序合成 (program synthesis)** 与**测试时微调 (test-time fine-tuning)**，强调了它们在函数重用方面的**不同方法**。[@samyaksharma](https://twitter.com/samyaksharma/status/1856058409466114418) 分享了关于 **Agentic AI 系统**的见解，重点关注**生产力提升**而非仅仅是**技术进步**。

**AI Governance and Ethics**

- **AI Safety and Policy**: [@nearcyan](https://twitter.com/nearcyan/status/1856165476331860031) 反思了 **AI 对编程自动化**的影响，对**界面创新缺乏**表示遗憾。[@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1856075185620955330) 讨论了将 **AI Safety** 整合到**政府计划**中，并质疑这些**计划是会奏效还是会产生危害**。

- **AI Alignment and Regulation**: [@latticeflowai](https://twitter.com/latticeflowai/status/1856094571681263851) 推出了 **COMPL-AI**，这是一个评估 **LLM 对齐**是否符合**欧盟 AI 法案 (EU’s AI Act)** 的框架。[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1856094571681263851) 也强调了在 **AI 治理**方面的努力，强调了**监管合规性**的重要性。

**AI Applications**

- **Generative AI in Media and Content Creation**: [@skirano](https://twitter.com/skirano/status/1856067458546946300) 在 **@everartai** 上推出了**生成式广告**，能够创建兼容 **Instagram 和 Facebook** 等平台的**广告格式图像**。[@runwayml](https://twitter.com/c_valenzuelab/status/1856091885820871050) 提供了 **Runway 工具**中**摄像机放置**的技巧，强调了**摄像机角度如何影响叙事**。

- **AI in Data Engineering and Analysis**: [@llama_index](https://twitter.com/llama_index/status/1856051032381628620) 展示了 **PureML**，它使用 **LLM 自动清理和重构 ML 数据集**，增强了**数据一致性和特征创建**。[@LangChainAI](https://twitter.com/LangChainAI/status/1856034155337003063) 推出了用于 **RAG 应用数据切块 (chunking)** 以及**识别 Agent 故障**的工具，提高了**数据检索和 Agent 的可靠性**。

- **AI in Healthcare and Biological Systems**: [@mustafasuleyman](https://twitter.com/mustafasuleyman/status/1856024937741660469) 分享了关于 **AI2BMD** 的工作，旨在通过 **AI 驱动的分析**来**理解生物系统**并**设计新的生物材料和药物**。

**Developer Infrastructure and Tools**

- **Bug Tracking and Error Monitoring**: [@svpino](https://twitter.com/svpino/status/1856305695479788012) 介绍了 **Jam** 等工具，这是一个用于**详细错误报告**的**浏览器扩展**，声称可以将 **bug 修复时间缩短 70% 以上**。[@tom_doerr](https://twitter.com/tom_doerr/status/1856115835141603495) 介绍了专为**开发者**量身定制的**错误跟踪和性能监控**工具。

- **Code Generation and Testing**: [@jamdotdev](https://twitter.com/jamdotdev/status/1856305707475419549) 在**错误报告工具**上进行了合作，而 [@svpino](https://twitter.com/svpino/status/1856049271096914034) 强调了使用 **DeepEval** 对 **LLM 驱动的应用进行单元测试**的重要性。

- **API Clients and Development Frameworks**: [@tom_doerr](https://twitter.com/tom_doerr/status/1856109830748127564) 介绍了一款用于管理 **REST、GraphQL 和 gRPC 请求**的**桌面 API 客户端**，提升了**开发效率**。此外，像 **Composio** 这样的工具支持**基于自然语言的操作**，简化了**工作流自动化**。

**AI Research and Insights**

- **LLM Training and Optimization**: [@Tim_Dettmers](https://twitter.com/Tim_Dettmers/status/1856338240099221674) 讨论了**数据中心扩展的终结**以及 **quantization 的极限**，认为未来的进步可能更多地依赖于**知识蒸馏 (knowledge distillation)** 和**模型动态性 (model dynamics)**。

- **AI 协作与生产力**：[@karpathy](https://twitter.com/karpathy/status/1856041540701040737) 沉思于一个 **IRC 成为主导协议的平行宇宙**，强调了信息交换向 **与 AI 进行实时对话** 的转变。

- **AI 教育与学习**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1856351116209827905) 推广了他们的 **Data Engineering 证书**，其中包含 **模拟对话**，以演示 **数据工程** 中的 **利益相关者需求收集**。

**迷因与幽默**

- **AI 与技术笑话**：[@JonathanRoss321](https://twitter.com/JonathanRoss321/status/1856370558277169531) 幽默地建议在阿西莫夫定律中增加 **第四条法则**，即 **机器人不能让机器人相信它们是人类**。[@giffmana](https://twitter.com/giffmana/status/1856116179124752766) 分享了对 **ASPX 网站** 的挫败感，表达了对 **过时技术** 的幽默 **抱怨**。

- **轻松的 AI 评论**：[@Sama](https://twitter.com/sama/status/1856169738910712314) 开玩笑说 **AI 通过 LLM 自动化接管生活**。[@Transfornix](https://twitter.com/transfornix/status/1856053751422779603) 戏称 **rotmaxers** 害怕“真正的那个”。

- **幽默的互动与反应**：[@richardMCNgo](https://twitter.com/RichardMCNgo/status/1856339352844153293) 以幽默的方式表达了对文化和历史的 **隐喻性反思**。[@lhiyasut](https://twitter.com/lhiyasut/status/1856117712763891866) 对“ai”在不同语言中的含义发表了机智的评论。

**社区与活动**

- **AI 会议与聚会**：[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1856138166346101202) 宣布在 **伦敦开设办公室**，并列举了全球各地的众多 **社区聚会**，包括 **多伦多、洛杉矶、上海** 等，旨在培养 **全球 AI 社区**。

- **播客与讨论**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1856037930994659455) 推广了他们 **由 AI 专家参与的播客**，讨论了 **AI 助手（AI assistants）的未来** 及其带来的 **伦理挑战**。[@omaarsar0](https://twitter.com/omarsar0/status/1856130579898433988) 与 **@lexfridman** 和 **@DarioAmodei** 等 **AI 思想领袖** 进行了 **讨论**。

- **教育内容与研讨会**：[@lmarena_ai](https://twitter.com/lmarena_ai/status/1856050008002572667) 鼓励参与他们的 **OSS fellowship**，而 [@shuyanzhxyc](https://twitter.com/shuyanzhxyc/status/1856097981759664357) 邀请个人 **加入他们在杜克大学的实验室**，该实验室专注于 **Agentic AI 系统**。

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

**主题 1. Qwen2.5-Coder 32B 发布：社区反响与技术解析**

- **[新的 Qwen 模型登上 Aider 排行榜！！！](https://i.redd.it/u5i812p00b0e1.png)** ([Score: 648, Comments: 153](https://reddit.com/r/LocalLLaMA/comments/1gox2iv/new_qwen_models_on_the_aider_leaderboard/)): **新的 Qwen 模型**已添加到 **Aider Leaderboard**，这标志着 AI 模型性能的进步，并可能在领域内树立新的基准。
  - 讨论重点关注 [Hugging Face](https://huggingface.co/collections/Qwen/qwen25-coder-66eaa22e6f99801bf65b0c2f) 上的 **Qwen2.5-Coder 模型**，用户将其性能与 **GPT-4o** 等其他模型进行了比较，并对其在 **Coding 任务**中的能力表示出浓厚兴趣。**32B 版本**被认为表现尤为强劲，一些用户发现它在特定任务中优于 GPT-4o。
  - 围绕**本地运行这些模型**存在技术考量，讨论涉及了高效处理 **32B** 和 **72B** 等模型尺寸所需的 **PC 规格**和**量化技术**。用户讨论了 **multi-shot inferencing** 的优势，以及为了实现实用的 Token 生成速度对**高内存带宽**的需求。
  - 对话还涉及了模型**许可**以及**开源社区**对这些发布的反应，部分模型遵循 **Apache License**，另一些则引发了关于可访问性和社区驱动开发的讨论。用户对这些模型的潜力感到兴奋，尤其是在自托管环境中，以及相比之前版本的改进。
- **[Qwen/Qwen2.5-Coder-32B-Instruct · Hugging Face](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct)** ([Score: 486, Comments: 134](https://reddit.com/r/LocalLLaMA/comments/1goz6gr/qwenqwen25coder32binstruct_hugging_face/)): **Qwen/Qwen2.5-Coder-32B-Instruct** 已在 **Hugging Face** 发布，引发了关于其能力和潜在应用的讨论。重点在于其技术规格以及在各种编程和指令任务中的表现。
  - 讨论强调了 **Qwen2.5-Coder-32B-Instruct** 模型的**性能和效率**，一些用户注意到，尽管其计算资源可能少于其他模型，但结果令人印象深刻。14B 版本也被提及，认为其效果几乎同样出色，且对于标准硬件配置的用户来说更易获取。
  - 用户讨论了运行这些模型的**技术要求**和**性能基准**，强调需要大量的 RAM 和 VRAM，并建议资源有限的用户使用较小的模型或量化版本（如 14B 或 7B）。分享了具体的基准测试和性能指标（如 Token 评估速率），以展示模型的能力。
  - 资源链接如 [Qwen2.5-Coder-32B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF) 和一篇[博客文章](https://qwenlm.github.io/blog/qwen2.5-coder-family/)提供了额外的背景信息。用户还讨论了 OpenVINO 转换的可用性，以及模型量化对性能和可用性的影响。
- **[我的测试提示词，以前只有初代 GPT-4 能答对。之后的模型都没成功过，直到 Qwen-Coder-32B。在 RTX 4090 上运行 Q4_K_M，它第一次尝试就成功了。](https://v.redd.it/lu0o83soec0e1)** ([Score: 327, Comments: 108](https://reddit.com/r/LocalLLaMA/comments/1gp46j9/my_test_prompt_that_only_the_og_gpt4_ever_got/)): **Qwen-Coder-32B** 成功地在第一次尝试中处理了一个复杂的测试提示词，这是以前只有初代 **GPT-4** 才能实现的壮举。该测试是在 **RTX 4090** GPU 上使用 **Q4_K_M** 量化版本进行的。
  - **平台与配置**：平台和配置显著影响模型性能，**VLLM** 和 **Llama.cpp** 等平台之间存在差异。**Temperature 设置**和自定义 UI 设置也会影响输出，正如 **LocoMod** 在其使用 **HTMX** 进行动态 UI 修改的个性化实现中所讨论的那样。
  - **模型性能与比较**：**Qwen-Coder-32B** 模型显示出可观的前景，优于经常在复杂提示词上失败的 **7B** 等较小模型。用户注意到 **32B** 处理多种编程语言的能力，而其他人则怀念初代 **GPT-4** 在能力被削减之前的卓越表现。
  - **技术规格与基准测试**：**RTX 4090** 的基准测试显示，在特定配置下可达到 **41 tokens/second**，突显了硬件在实现高效性能方面的重要性。用户分享了他们的配置，包括 **双 3090** 和 **双 P40**，分别达到了 **22 tokens/second** 和 **7 tokens/second**，说明了基于硬件和配置的性能差异。


**主题 2. ExllamaV2 通过 Pixtral 引入视觉模型支持**

- **ExllamaV2 在 v0.2.4 中发布了对 Pixtral 的支持** ([Score: 29, Comments: 2](https://reddit.com/r/LocalLLaMA/comments/1gpgls3/exllamav2_ships_pixtral_support_with_v024/)): **ExllamaV2** 发布了 **v0.2.4** 版本，支持视觉模型 **Pixtral**，这标志着其首次涉足视觉模型支持。**Turboderp** 建议未来扩展多模态（multimodal）功能，可能允许将 **Qwen2.5 32B Coder** 等模型与来自 **Qwen2 VL** 的视觉功能集成，从而增强开源模型的吸引力。欲了解更多详情，请参考 [release notes](https://github.com/turboderp/exllamav2/releases/tag/v0.2.4) 以及 [GitHub](https://github.com/turboderp/exllamav2/issues/658) 上相关的 API 支持讨论。
- **[Qwen2.5-Coder 系列：强大、多样、实用。](https://qwenlm.github.io/blog/qwen2.5-coder-family/)** ([Score: 58, Comments: 8](https://reddit.com/r/LocalLLaMA/comments/1gozarh/qwen25coder_series_powerful_diverse_practical/)): **Qwen2.5-Coder 32B** 被推测是一款既强大又实用的 **multi-modal** AI 模型，暗示了在多样化应用方面的潜在进步。由于缺乏正文内容，该模型的具体细节和特性尚未得到确认。
  - **通义（Tongyi）官网**承诺提供一种支持一键生成网站和视觉应用的“代码模式”，但尽管之前已有公告，该功能尚未上线。用户报告称，虽然该模型可以生成代码（例如 **HTML 贪吃蛇游戏**），但无法渲染输出结果。


**主题 3. 探索 Binary Vector Embeddings：速度与压缩的权衡**

- **二进制向量嵌入（Binary vector embeddings）非常酷** ([Score: 314, Comments: 20](https://reddit.com/r/LocalLLaMA/comments/1gov1q4/binary_vector_embeddings_are_so_cool/)): **Binary vector embeddings** 在提供 **32 倍压缩**和约 **25 倍检索加速**的同时，实现了超过 **95% 的检索准确率**，使其在数据密集型应用中具有极高的效率。更多详情请参阅[博客文章](https://emschwartz.me/binary-vector-embeddings-are-so-cool/)。
  - **Binary Vector Embeddings** 因其效率和速度而备受关注，通过 **Numpy** 的 **bitwise_count()** 等工具可以简化实现，从而在 CPU 上实现快速执行。讨论强调了在廉价 CPU 上使用 **xor + popcnt** 等简单操作实现二进制量化（binary quantization）的便利性。
  - **模型训练与兼容性** 对于有效的二进制量化至关重要，像 **MixedBread** 和 **Nomic** 这样的模型是专门为压缩友好型操作而训练的。这一方法得到了 **Cohere** 文档的支持，该文档强调模型需要在包括 **int8** 和 **binary** 在内的不同压缩格式下表现良好。
  - **压缩中的权衡** 非常显著，正如 **pgVector** 维护者所讨论的，用户报告了取决于位多样性（bit diversity）的不可预测损失。衡量这些损失的复杂性表明，需要进行仔细评估以确定数据流水线（data pipeline）是否适合二进制量化。
- **这是否是开放 AI 的黄金时代 - SD 3.5, Mochi, Flux, Qwen 2.5/Coder, LLama 3.1/2, Qwen2-VL, F5-TTS, MeloTTS, Whisper 等** ([Score: 74, Comments: 28](https://reddit.com/r/LocalLLaMA/comments/1gpfb21/is_this_the_golden_age_of_open_ai_sd_35_mochi/)): 该帖子讨论了 **开源 AI 模型的重大进展**，重点介绍了近期发布的 **Qwen 2.5/Coder, LLama 3.1/2 和 SD 3.5**。它强调了开源与闭源 AI 模型之间差距的缩小，并引用了推理服务的价格优势（约 **每百万 token 0.2 美元**）以及 **Groq** 和 **Cerebras** 等专用硬件提供商的潜力。作者认为开源模型目前正超越闭源模型，尽管面临潜在的监管挑战，但前景光明。
  - **硬件要求与性能**：用户讨论了 AI 模型在各种 GPU 上的性能，提到了使用 **RTX 4070 Super** 和 **RTX 3080** 运行 Mochi 1 并生成视频片段。据报道，**RTX 4070 Super** 在 ComfyUI 中生成一段视频需要 7.5 分钟，而为了获得更高质量的输出，用户倾向于选择像 3090 这样拥有 **24GB VRAM** 的显卡。
  - **开源与闭源模型的能力**：讨论强调了 **Qwen 模型** 的编程能力，并指出了开源与闭源模型之间的差距。**Qwen 2.5 Coder 14B** 在 Aider 基准测试中超越了 **Llama3 405b**，而像 **Qwen 2.5 Coder 3B** 这样的小型模型对于本地任务非常有用，这表明了对开源进展的乐观态度。
  - **未来前景与发展**：关于开源模型的当前阶段存在争论，一些人认为它们已接近闭源模型但尚未超越。随着消费级硬件的改进，社区期待进一步的突破，用户对阿里巴巴新发布的 **Easy Animate**（需要 **12GB VRAM**）表现出浓厚兴趣。


**主题 4. Qwen 2.5 技术基准：硬件与平台策略**

- **qwen-2.5-coder 32B 在 3xP40 和 3090 上的基准测试** ([Score: 49, Comments: 22](https://reddit.com/r/LocalLLaMA/comments/1gp376v/qwen25coder_32b_benchmarks_with_3xp40_and_3090/)): **qwen-2.5-32B** 的基准测试显示，**3090 GPU** 在 **32K context** 下达到了显著的 **28 tokens/second**，而单个 **P40 GPU** 可以处理 **10 tokens/second**。**3xP40 配置**在 **Q8 quantization** 下支持 **120K context**，但性能并非线性扩展，其中 **row split mode** 显著提升了生成速度。将 P40 的功耗限制从 **160W 调整到 250W** 对性能影响微乎其微，而 3090 在 **350W** 功耗下的生成速度表现更优，达到 **32.83 tokens/second**。
  - **VLLM** 对 **P40 GPU** 的兼容性有限，用户推荐 **llama.cpp** 作为这些 GPU 的最佳选择。据指出，**MLC** 在 P40 上的表现比 **GGUF Q4** 差约 **20%**，且缺乏 **flash attention**，这进一步增强了用户对 **llama.cpp** 的偏好。
  - 围绕 **Q4、Q8 和 fp8/16** 等 **quantization levels** 的讨论显示性能差异极小，正如 [Neural Magic 博客文章](https://neuralmagic.com/blog/we-ran-over-half-a-million-evaluations-on-quantized-llms-heres-what-we-found/)中所详述的那样。用户强调了对 **kv cache** 进行量化以减少内存占用且无明显质量损失的好处。
  - **P40 GPU 功耗**被有效管理在 **120W** 左右，超过 **140W** 几乎没有收益。用户报告在**水冷 3090** 上达到 **36 tokens/second**，并强调 P40 在价格低于 **$200** 时是一个极具性价比的选择。

- **[分布在 4 台 M4 Pro Mac Mini 上的 LLM + Thunderbolt 5 互连 (80Gbps)](https://x.com/alexocheema/status/1855238474917441972)** ([Score: 58, Comments: 30](https://reddit.com/r/LocalLLaMA/comments/1gowx3o/llms_distributed_across_4_m4_pro_mac_minis/)): 讨论了在通过带宽为 **80Gbps** 的 **Thunderbolt 5** 互连的 **四台 M4 Pro Mac Mini** 配置上运行 **Qwen 2.5**。重点在于跨此硬件配置分布 **LLM** 的潜力。
  - 讨论集中在 **M4 Pro Mac Mini** 与 **M2 Ultra** 和 **M4 Max** 等替代方案相比的性价比和配置上。一台顶配的 M4 Pro Mini 售价约 **$2,100**，两台配置可提供 **128GB VRAM**，而售价 **$4,999 的 M4 Max** 虽然价格更高，但提供了两倍的内存带宽和 GPU 核心。
  - 用户辩论了 **Mac Mini** 与传统配置（如搭载 **4x3090 GPU 的 ROMED8-2T 主板**）的实用性，理由是前者易于使用且发热量更低。能够避免 **Linux、CUDA 错误**和 **PCIe** 常见问题的潜力被视为一个显著优势。
  - 存在对性能宣称的怀疑，包括对模型细节的疑问，例如它是 **tensor parallel** 还是所使用的模型精度类型（例如 **fp16, Q2**）。在考虑从现有设备切换之前，强调需要证明 **Mac Mini** 在以合理速度进行 **fine-tuning** 方面的能力。



## 其他 AI Subreddit 摘要

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. Claude 3.5 Opus 即将推出：Anthropic CEO 确认**

- **Anthropic CEO on Lex Friedman, 5 hours!** ([Score: 184, Comments: 49](https://reddit.com/r/ClaudeAI/comments/1gp10p5/anthropic_ceo_on_lex_friedman_5_hours/)): **Anthropic CEO Dario Amodei** 做客 **Lex Fridman 播客** 进行了一场 **5 小时的对话** [可在 YouTube 观看](https://youtu.be/ugvHCXCOmm4)。讨论确认了 **Claude Opus 3.5** 仍在持续开发中，但未提供具体的发布时间表。
  - 用户对 **Anthropic** 声称没有“削弱 (**nerfing**)” **Claude** 表示怀疑，指出性能变化可能是通过“根据当前负载通过提示词分配不同的思考预算 (**thinking budget**)”而非修改权重来实现的。
  - 知名嘉宾 **Chris Olah** 和 **Amanda Askell** 因其在**机械可解释性 (mechanistic interpretability)** 和**哲学思考**方面的专业知识而受到关注，引发了观众的极大兴趣。
  - 社区对 **Lex Fridman** 最近的内容方向表示担忧，用户注意到他正从技术主题转向与政治人物的争议性关联，甚至被称为“普京辩护者”。
- **[Opus 3.5 is Not Die! It will be still coming out conform by anthropic CEO](https://v.redd.it/p2m3wg6jbf0e1)** ([Score: 62, Comments: 63](https://reddit.com/r/ClaudeAI/comments/1gpfglu/opus_35_is_not_die_it_will_be_still_coming_out/)): 根据公司 **CEO** 的说法，**Anthropic** 的模型 **Opus 3.5** 仍在开发中。该帖子缺乏关于发布时间表或模型能力的额外背景或具体细节。
  - 用户讨论了 **Opus 3.5** 的潜在**定价**，预期在 **$100/M tokens** 左右，类似于 **GPT-4-32k** 的 **$120/M tokens**。如果该模型能提供卓越的 **one-shot** 性能（特别是在编程任务中），多位用户表示愿意支付溢价。
  - 社区对之前 **Reddit** 上关于 **Opus 3.5** 被取消或合并到 **3.5 Sonnet** 的猜测产生了怀疑。用户指出，以 Sonnet 的价格运行更大的模型对 **Anthropic** 来说在财务上是不可持续的。
  - 竞争压力被提及，**Qwen** 正在赢得市场份额。用户还批评了 **CEO** 的沟通风格，认为他在讨论模型开发状态时言辞闪烁且显得局促。


**Theme 2. Qwen2.5-Coder-32B Matches Claude: Open Source Milestone**

- **[Open source coding model matches with sonnet 3.5](https://i.redd.it/xubrm1xeib0e1.jpeg)** ([Score: 100, Comments: 33](https://reddit.com/r/ClaudeAI/comments/1goznro/open_source_coding_model_matches_with_sonnet_35/)): **开源编程模型**宣称性能可与 **Claude Sonnet 3.5** 媲美，尽管帖子正文未提供更多背景或证据。
  - **LM Studio** 使得在本地运行该模型变得触手可及，为自动化任务提供网络连接，并提供各种**量化选项**，如 **17GB** 的 **Q3**。该模型在 **VRAM** 中运行效果最佳，而非从 RAM 运行。
  - **Qwen2.5-Coder-32B** 模型在 **24GB** 显卡上配合 **Q4 量化** 运行效果良好，可在 [Hugging Face](https://huggingface.co/lmstudio-community/Qwen2.5-Coder-32B-Instruct-GGUF) 获取。用户指出它比 **Haiku** 更具成本效益，成本约为其一半。
  - 用户对**微调 (fine-tuning)** 能力表现出兴趣，以匹配特定的编程风格和项目结构，并可选择通过 **OpenRouter** 以极具竞争力的价格进行托管。该模型在其 **32B** 规模下表现出了令人印象深刻的性能。
- **Every one heard that Qwen2.5-Coder-32B beat Claude Sonnet 3.5, but....** ([Score: 61, Comments: 42](https://reddit.com/r/ClaudeAI/comments/1gpf16b/every_one_heard_that_qwen25coder32b_beat_claude/)): 如对比统计图表所示，**Qwen2.5-Coder-32B** 在编程基准测试中超越了 **Claude Sonnet**。图片展示了两个模型之间的性能指标，突显了 **Qwen** 在更低运营成本下的竞争能力。
  - **Qwen2.5-Coder-32B** 作为开源模型因其出色的性能受到赞誉，通过 [deepinfra.com](https://deepinfra.com/Qwen/Qwen2.5-Coder-32B-Instruct) 的定价为 **每百万 token $0.18**，而 **Claude Sonnet** 的输入/输出费率为 **$3/$15**。
  - 实际测试显示，**Qwen** 在特定开发任务中表现出色，但与 **Claude** 相比，在处理复杂逻辑和设计任务时仍显吃力。该模型的 **32B 尺寸** 允许在本地计算机运行，但 **Q3 量化** 可能会影响其在复杂任务上的表现。
  - 中国政府对 AI API 提供了大量补贴，这解释了其低廉的 token 成本；而 **Anthropic** 最近提高了 **Haiku 3.5** 的价格，理由是智能水平有所提升。用户指出，这降低了使用闭源模型的动力。


**Theme 3. ComfyUI Video Generation: New Tools & Capabilities**

- **[[mochi1 文本转视频（内置 ComfyUI 且速度极快）](https://v.redd.it/jh40limw7g0e1)]** ([评分: 51, 评论: 6](https://reddit.com/r/StableDiffusion/comments/1gpi285/mochi1_text_to_video_comfyui_is_built_in_and_is/)): **Mochi1** 是一款 **text-to-video** 模型，集成了 **ComfyUI** 工作流功能用于视频生成。该工具强调操作速度，尽管帖子中未提供具体的性能指标或技术细节。
  - 用户指出原帖包含**重复链接**且缺乏适当的**工作流文档**，批评该帖子宣称“包含工作流（Workflow Included）”具有误导性。
- **[[使用 ComfyUI、Cogvideox 模型和 DimensionX lora 制作。全自动 AI 3D 动画。我热爱比利时漫画，想用 AI 展示一个如何增强它们的例子。很快会有完整的 3D 建模吗？等待更多 lora 来创建一个完整的移动端 App。感谢 @Kijaidesign](https://v.redd.it/t4bsn3zudh0e1)]** ([评分: 83, 评论: 10](https://reddit.com/r/StableDiffusion/comments/1gplz5y/made_with_comfyui_and_cogvideox_model_dimensionx/)): 使用 **ComfyUI** 和 **Cogvideox** 模型配合 **DimensionX lora** 创作了比利时漫画的 **3D 运动动画**。创作者的目标是在更多 lora 模型发布后，开发一款使用 **AI** 增强比利时漫画的**移动应用程序**。
  - 用户询问了**工作流**以及在动画过程中使用 **After Effects** 的可能性，表现出对技术实现细节的浓厚兴趣。
  - 评论者预见了**自动化分镜动画（panel-to-panel animation）**的潜力，其具有能够适应不同漫画布局和构图的动态**相机移动（camera movements）**。


**主题 4. Reddit 上的 AI 内容生成：增长趋势与担忧**

- **[还记得那个 50k 点赞的帖子吗？原作者承认 100% 是 ChatGPT 写的](https://www.reddit.com/gallery/1gpjspp)** ([评分: 1349, 评论: 163](https://reddit.com/r/ChatGPT/comments/1gpjspp/remember_this_50k_upvote_post_op_admitted_chatgpt/)): 据称 **ChatGPT** 生成了一个获得 **50,000 个点赞**的 Reddit 热门帖子，原作者随后确认内容完全由 AI 生成。源材料中未提供关于该特定帖子内容的更多上下文或细节。
  - 用户指出了 AI 生成内容的几个**写作风格特征**，特别是使用 **em-dashes** 和在典型 Reddit 帖子中少见的正式格式。结构化、冗长的格式被认为是 AI 创作的关键迹象。
  - 讨论集中在 Reddit 上**检测 AI 内容**日益增长的挑战，用户对平台被 AI 生成的帖子主导表示担忧。几位评论者提到，尽管注意到一些可疑元素，最初还是被骗了。
  - 一名正确识别该帖子为 AI 生成的用户最初因其怀疑态度被**点赞降级（downvoted）和批评**，这突显了社区检测 AI 内容的能力参差不齐。原帖获得了 **50,000 个点赞**，而这一真相揭露获得的关注显著较少。
- **[死网理论（Dead Internet Theory）：r/ChatGPT 上的这个帖子获得了 50k 点赞，随后原作者承认是 ChatGPT 写的](https://www.reddit.com/gallery/1gplgop)** ([评分: 130, 评论: 48](https://reddit.com/r/OpenAI/comments/1gplgop/dead_internet_theory_this_post_on_rchatgpt_got/)): 当 **r/ChatGPT** 上一个达到 **50,000 个点赞**的热门帖子被揭露为 **AI 生成内容**时，**死网理论（Dead Internet Theory）**获得了更多可信度，原作者随后承认 **ChatGPT** 撰写了整个提交内容。这一事件说明了人们对 AI 生成内容在没有明确披露其人工来源的情况下主导社交媒体平台的担忧。

---

# AI Discord 摘要回顾

> 由 O1-mini 总结的总结之总结

**AI 语言模型争夺霸主地位**

- **Qwen2.5 Coder 超越 GPT-O 和 Claude 3.5**: [**Qwen2.5 Coder 32B**](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct) 在复杂任务上的表现达到 **73.7%**，超过了 **GPT-O**，而 **Claude 3.5 Sonnet** 为 **84.2%**。*用户称赞其开源能力*，同时也注意到其持续的改进。
- **Phi-3.5 的过度审查引发辩论**: **Microsoft** 的 **Phi-3.5** 模型因严厉的审查面临批评，导致 Hugging Face 上出现了[无审查版本](https://huggingface.co/SicariusSicariiStuff/Phi-3.5-mini-instruct_Uncensored)。*用户幽默地嘲讽了 Phi-3.5 过度的限制*，强调了这对其在技术任务中实用性的影响。
- **OpenAI o1 模型发布备受期待**: **OpenAI** 正准备在年底前全面发布 [**o1 推理模型**](https://x.com/OpenRouterAI/status/1856165171690926446)，匿名见解进一步点燃了社区的热情。*对开发团队专业知识的推测*增加了期待感。

**Optimization Techniques Revolutionize Model Training**

- **Gradient Descent Mechanics Unveiled**：在 **Eleuther** Discord 中，工程师们讨论了使用 Gradient Descent 缩放更新以及 **second-order information** 对实现最优收敛的作用。讨论引用了关于特征学习和核动力学（kernel dynamics）的[最新论文](https://arxiv.org/abs/2310.17813)。
- **LoRA Fine-Tuning Accelerates Inference**：**Unsloth AI** 成员利用 **LoRA fine-tuned models**（如 [**Llama-3.2-1B-FastApply**](https://huggingface.co/patched-codes/Llama-3.2-1B-FastApply)），通过原生支持实现加速推理。示例代码展示了通过减小模型大小来提高执行速度。
- **Line Search Methods Enhance Learning Rates**：**Eleuther** 参与者探索了 **line search techniques**，以便在 Loss 趋于发散时动态恢复最佳 Learning Rate。*研究结果表明，线搜索产生的速率约为***更新范数（norm of the update）的 1/2**，这暗示了某种一致的模式。

**Deployment and Inference Get a Boost with New Strategies**

- **Speculative Decoding Boosts Inference Speed**：成员们分享了 **Speculative Decoding** 以及使用 **FP8 或 int8 precision** 作为提升 **Inference speed** 的策略。来自 **qroq** 和 **Cerebras** 等供应商的自定义 **CUDA kernels** 提供了更高的性能增益。
- **Vast.ai Offers Affordable Cloud GPU Solutions**：[**Vast.ai**](https://vast.ai/pricing) 被推荐为实惠的云端 GPU 供应商，对于 **A100** 和 **RTX 4090** 等 GPU，价格范围在每小时 **$0.30** 到 **$2.80** 之间。*用户建议不要使用旧款 Tesla 显卡*，而应选择更新的硬件以保证可靠性。
- **Multi-GPU Syncing Poses Challenges**：**Interconnects** 和 **GPU MODE** 中的讨论强调了在 **multi-GPU setups** 中使用 Pytorch 的 **SyncBatchNorm** 等工具同步均值和方差参数的复杂性，这在 **liger** 等框架中构成了实现挑战。

**APIs and Tools Streamline AI Development**

- **Cohere API Changes Cause Headaches**：由于 **/rerank** 端点中 **return_documents** 字段的移除，**Cohere** Discord 的用户面临 **UnprocessableEntityError**。*团队成员正在努力恢复该参数*，[Cohere 的支持团队](https://cohere.com/)正在处理此问题。
- **Aider Integrates LoRA for Faster Operations**：在 **aider** Discord 中，成员讨论了利用 **LoRA fine-tuned models**（如 [**Llama-3.2-1B-FastApply**](https://huggingface.co/patched-codes/Llama-3.2-1B-FastApply)），通过 Aider 的原生支持实现加速推理。示例代码演示了加载 Adapter 以提高速度。
- **NotebookLM Enhances Summarization Workflows**：**Notebook LM Discord** 的参与者探索了使用 [**NotebookLM**](https://notebooklm.google.com/) 来总结超过 **200 封 AI newsletter 邮件**，从而简化信息消化流程。讨论中提到了**音频文件上传失败**等技术问题，指向了潜在的**技术故障**。

**Scaling Laws and Datasets Challenge AI Research**

- **Scaling Laws Reveal Quantization Limits**：在 **Eleuther** 和 **Interconnects** Discord 中，研究人员讨论了一项研究，该研究表明在更多 Token 上训练的模型需要更高的精度进行 **Quantization**，从而影响了可扩展性。*人们对* ***LLaMA-3*** *模型在这些定律下的表现表示担忧*。
- **Aya_collection Dataset Faces Translation Inconsistencies**：**Cohere** 用户发现 **aya_collection** 数据集在 19 种语言的翻译中存在差异，英语有 **249716** 行，而阿拉伯语和法语为 **124858** 行。[*translated_cnn_dailymail*](https://discord.com/channels/954421988141711382/954421988783444043/1305712217844482081) 中的特定不匹配问题被重点指出。
- **Data-Parallel Scaling Bridges Theory and Practice**：**Eleuther** Discord 上的讨论强调了在 **data-parallel scaling** 中桥接理论与应用的实际挑战，并引用了[文档](https://docs.google.com/document/d/1jL0-82COewU-UDyzCt-vFgpBTBr9AiBvwGlShmfPpew/edit?tab=t.0)。*诸如“有效的东西不被允许发表”之类的引言凸显了出版限制*。

---

# PART 1: High level Discord summaries

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **探索 Gradient Descent 机制**：一场讨论探讨了 **Gradient Descent** 更新机制，重点关注更新投影和范数如何影响 **模型权重变化**。
  
  - 参与者辩论了相对于输入变化缩放更新的重要性，以及 **二阶信息** 在实现 **最优收敛** 中的作用。
- **Muon 优化的重要性**：研究了 **Muon** 作为优化器的角色，强调了它与 **Feature Learning** 的交互以及对 **网络训练动力学** 的影响。
  
  - 建议包括探索 **Muon** 与其他理论框架（如 **Kernel Dynamics**）以及现有 **Feature Learning** 文献之间的联系。
- **填补 Scaling Laws 的空白**：一位成员分享了关于 [填补 Scaling Laws 缺失部分](https://docs.google.com/document/d/1jL0-82COewU-UDyzCt-vFgpBTBr9AiBvwGlShmfPpew/edit?tab=t.0) 的见解，强调了弥合理论与 **应用** 之间差距的实际挑战。
  
  - *那些有效的东西是不允许发表的*，突显了有效应用研究成果所面临的挑战。
- **使用 Line Searches 优化 Learning Rates**：有人推测 **Line Searching** 是一种在训练期间恢复 **最优 Learning Rates** 的方法，特别是在 **Loss** 接近发散时。
  
  - 一位贡献者引用了研究结果，指出 **Line Searches** 产生的速率约为 **更新范数的 1/2**，表明可能存在一致的模式。
- **建议 Text-MIDI 多模态数据集**：一位参与者提议实现一个 **Text-MIDI 多模态数据集**，考虑到现有的录音和元数据集合。
  
  - 他们承认 **版权限制**，建议仅将 **MIDI 文件** 开源。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 面临持续技术问题的困扰**：用户报告了 **Perplexity AI** 平台的 **持续技术问题**，特别是影响长对话线程且仍未解决的 **隐藏消息 Bug**。
  
  - 这些持续存在的问题已经出现了一个多月，尽管修复了其他次要 Bug，但仍严重 **影响了用户体验**。
- **关于 Perplexity Pro 订阅到期的不确定性**：一位用户询问了他们的 **Perplexity Pro** 免费一年到期后的续订情况，并质疑其对 **R1 设备** 的影响。
  
  - 社区确认订阅在试用期后不会保持免费，用户将恢复到 **受限的免费搜索**。
- **Perplexity 模型对决：GPT-O 位居榜首**：讨论表明 **GPT-O** 在 **Perplexity AI** 中的表现优于其他模型，尤其是在特定任务中。
  
  - 相反，尽管 **o1** 具有专业化性质，但被认为 **应用有限**。
- **Mac App UI 问题困扰 Perplexity 用户**：用户报告了 **Perplexity** 应用 **Mac 版本** 的 **UI 问题**，强调了 **缺少滚动条** 阻碍了导航。
  
  - 其他投诉包括持续的 **Google 登录问题** 以及缺少 **Web App** 中可用的功能。
- **社区寻求 Pplx API DailyBot 编辑器的解决方案**：一位成员请求关于实现 **Pplx API DailyBot 自定义命令编辑器** 的指导，寻求项目启动的初步步骤。
  
  - 另一位用户分享了使用带有 **Webhooks** 的 **CodeSandBox VM** 的变通方法，但社区正在探索更好的 **替代解决方案**。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen 2.5 Coder 微调资源发布**：一个新的针对 **Qwen 2.5 Coder (14B)** 模型的 [微调 notebook](https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing) 现已在 Colab 上线，支持免费微调，VRAM 占用减少 60%，并将上下文长度从 **32K 扩展至 128K**。
  
  - 用户可以访问 [Qwen2.5 Coder Artifacts](https://huggingface.co/spaces/Qwen/Qwen2.5-Coder-Artifacts) 和 [Unsloth 版本](https://huggingface.co/unsloth/Qwen2.5-Coder-0.5B) 来解决 token 训练问题并提升模型性能。
- **更快速推理的优化策略**：成员们分享了提升推理速度的技术，包括 **Speculative Decoding**、利用 **FP8 或 int8** 精度，以及实现 **自定义优化的 CUDA kernel**。
  
  - **qroq** 和 **Cerebras** 等供应商已经开发了 **自定义硬件** 解决方案来进一步提升性能，尽管这可能会影响吞吐量。
- **LoRA 微调与 Unsloth 的集成**：用户讨论了利用 Unsloth 的原生支持，使用经过 LoRA 微调的模型（如 [Llama-3.2-1B-FastApply](https://huggingface.co/patched-codes/Llama-3.2-1B-FastApply)）来加速推理。
  
  - 提供的示例代码演示了如何使用 Unsloth 加载 adapter，由于模型尺寸较小，执行速度得到了提升。
- **模型 Checkpoint 和 Adapter 使用最佳实践**：使用 **PeftModel** 类成功实现了在基础模型之上集成 adapter 模型进行推理，强调了在加载模型时指定 checkpoint 路径的重要性。
  
  - 最佳实践包括先构建 adapter 并确保正确的 checkpoint 路径，以促进准确的模型增强和部署。
- **管理模型训练期间的 RAM 使用**：一名用户报告在运行 **Gemma 2B** 时 RAM 消耗增加，这可能是由于评估过程加剧了内存需求。
  
  - 另一名成员询问了评估实践，建议关闭评估可能会减轻内存占用过高的问题。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Qwen 2.5 Coder 性能**：**Qwen 2.5-Coder-32B** 在复杂任务中表现出 **73.7%** 的性能，落后于 **Claude 3.5 Sonnet** 的 **84.2%**。
  
  - 用户指出 **Qwen** 模型仍然会出现占位符响应，这可能会阻碍编码效率和完整性。
- **Aider 安装与使用**：**Aider 安装** 需要 **Python 3.9-3.12** 和 **git**，用户可参考 [官方安装指南](https://aider.chat/docs/install/install.html) 获取帮助。
  
  - 讨论强调了简化安装流程以提升 AI 工程师用户体验的重要性。
- **模型对比**：将 **Qwen 2.5-Coder** 的性能与 **DeepSeek** 和 **GPT-4o** 等模型进行了对比，在不同任务中表现各异。
  
  - 排行榜分数表明，调整模型配置可以优化特定编码任务的性能。
- **Aider 配置警告**：当 **Ollama server** 未运行或未设置 **API base** 时，用户会遇到 **Aider 配置警告**，导致出现通用警告而非具体错误。
  
  - 社区建议包括验证模型名称以及解决 **Litellm** 持续存在的 bug 以消除虚假警告。
- **OpenRouter API 使用**：有报告称 **OpenRouter API** 存在问题，例如由于模型名称无法识别，基准测试脚本无法连接到 **llama-server**。
  
  - 解决方案涉及调整 `.aider.model.metadata.json` 文件，该文件主要影响成本报告，必要时可以忽略。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Qwen 2.5 Coder 突破 23.5T Tokens**：Qwen 2.5 Coder 已在惊人的 **23.5 trillion tokens** 上进行了预训练，使其成为首个突破 **20 trillion** token 门槛的开源权重模型，正如 #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1305630017870364732) 频道中所强调的。
  
  - 尽管取得了这一成就，用户仍对在本地运行 Qwen 2.5 的挑战表示担忧，理由是需要像 **128GB MacBook** 这样的高规格硬件才能处理完整的 BF16 精度。
- **Scaling Laws 挑战 LLaMA-3 Quantization**：#[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1305630928516939797) 频道讨论的一项研究表明，随着模型在更多 token 上进行训练，它们在进行 **quantization** 时需要更高的精度，这给 **LLaMA-3** 模型带来了重大挑战。
  
  - 研究表明，预训练数据的持续增加可能会对 quantization 过程产生不利影响，引发了对未来 **AI models** 可扩展性和性能的担忧。
- **Dario Amodei 预测 2027 年实现人类水平 AI**：在 #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1305630017870364732) 频道的一档播客中，**Dario Amodei** 讨论了在各种 AI 模态中观察到的 scaling 现象，预测 **human-level AI** 将在 2026-2027 年出现。
  
  - 他强调了 **AI systems** 在规模扩大时道德考量和细微行为的重要性，并指出了实现这些进步过程中潜在的不确定性。
- **Nous Research 推出 Forge Reasoning API Beta**：Nous Research 在 #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1305630017870364732) 频道发布了 **Forge Reasoning API Beta**，旨在增强适用于任何模型的 inference time scaling，特别针对 **Hermes 70B** 模型。
  
  - 尽管发布前景看好，但人们对报告的 benchmarks 的一致性仍存在担忧，导致对该 API 性能指标可靠性的怀疑。
- **OpenAI 准备正式发布 o1 模型**：正如 #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1305630017870364732) 频道所讨论的，人们对 OpenAI 计划在年底前正式发布 **o1 reasoning model** 的期待日益增加。
  
  - 社区成员对 o1 背后的开发团队特别感兴趣，匿名消息来源引发了关于该模型能力和底层技术的猜测。

 

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Qwen 凭借 Coder 32B 实现飞跃**：根据 [OpenRouter 的推文](https://x.com/OpenRouterAI/status/1856165171690926446)，新发布的 **Qwen2.5 Coder 32B** 在多项编程 benchmarks 中超越了竞争对手 **Sonnet** 和 **GPT-4o**。
  
  - 尽管有这些说法，一些成员对准确性提出了质疑，认为 **MBPP** 和 **McEval** 等测试可能无法完全反映真实性能。
- **Gemini 1.5 Flash 提升性能**：**Gemini 1.5 Flash** 已获得包括 **frequency penalty**、**presence penalty** 和 **seed** 调整在内的更新，根据 [OpenRouter 的官方更新](https://openrouter.ai/google/gemini-flash-1.5-8b)，提升了其在各种任务中的能力。
  
  - 用户注意到性能有所提高，尤其是在 **temperature 0** 时，并推测 **Google AI Studio** 上部署了一个实验版本。
- **Anthropic 的工具尚未兼容**：讨论显示，**Anthropic's computer use tool** 目前在 OpenRouter 中缺乏支持，需要特殊的 beta header。
  
  - 成员们对未来的兼容性表示出兴趣，以增强其项目中的集成和功能。
- **OpenRouter 引入价格调整**：OpenRouter 澄清说，通过积分支付 token 可能会产生约 **5% 的额外费用**，如其 [服务条款](https://openrouter.ai/terms) 中所述。
  
  - 这一更新引发了用户关于价格透明度以及与直接使用模型进行对比的询问。
- **Beta 测试者寻求自定义 Provider Keys**：多位用户请求访问 **custom provider keys** 进行 beta 测试，以便更好地管理 **Google** 的 **rate limits**。
  
  - 强烈的兴趣凸显了社区对增强功能和项目优化的渴望。

 

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Qwen2.5-Coder 展示开源实力**：新款 [Qwen2.5-Coder](https://qwenlm.github.io/blog/qwen2.5-coder-family/#qwen25-coder--artifacts) 模型因其 **开源代码能力** 受到关注，为对抗 **GPT-4o** 等模型提供了竞争优势。
  
  - 模型及其演示可在 [GitHub](https://github.com/epuerta9/kitchenai)、[Hugging Face](https://huggingface.co/) 和 [ModelScope](https://modelscope.com/) 上获取。
- **KitchenAI 项目寻求开发者贡献**：开源项目 [KitchenAI](https://github.com/epuerta9/kitchenai) 正式发布，旨在创建 **可共享的运行时 AI cookbooks**，并邀请开发者参与贡献。
  
  - 团队正在 **Discord** 和 **Reddit** 上开展推广活动，以吸引感兴趣的贡献者。
- **优化 GPT 模型的 Prompt Engineering**：讨论集中在提高 **提示词清晰度** 以及利用 **Token 计数** 来优化 GPT 模型的输出。
  
  - 分享了一份 [Prompt Engineering 指南](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api)，以帮助成员提升其提示词设计技能。
- **评估 TTS 替代方案：关注 f5-TTS**：成员们探索了各种 **文本转语音 (TTS)** 解决方案，其中 [f5-tts](https://drinkoblog.weebly.com/) 因其在消费级 GPU 上的表现而获得推荐。
  
  - 讨论还包括在处理有关时间戳数据能力的问题时，建议关注 **高性价比的解决方案**。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **WSL2 中的 CUDA 驱动限制**：Windows 上的 **Nvidia CUDA 驱动** 在 WSL2 中被存根（stubbed）为 [libcuda.so](https://www.tensorflow.org/install/pip#windows-wsl2_1)，这可能会限制通过 Mojo 使用完整的驱动功能。
  
  - 成员们指出，如果 **MAX** 依赖于宿主 Windows 驱动，这种存根驱动可能会使 WSL 内的支持变得复杂。
- **CRABI ABI 提案增强语言互操作性**：由 [joshtriplett](https://github.com/rust-lang/rust/pull/105586) 提出的 `CRABI` **实验性特性门控提案** 旨在为 Rust、C++、Mojo 和 Zig 等高级语言之间的互操作性开发一种新的 ABI。
  
  - 参与者讨论了与 Lua 和 Java 等语言的集成挑战，表明需要更广泛的采用。
- **通过正确的 URL 修复 Mojo 安装问题**：一位用户通过修正 `curl` 命令的 URL 解决了 **Mojo 安装** 问题，确保了安装成功。
  
  - 这强调了在安装软件包时准确输入 URL 的重要性。
- **Mojo 的 Benchmark 模块面临性能限制**：Mojo 中的 **Benchmark 模块** 通过管理设置（setup）和清理（teardown）以及处理吞吐量测量的单位，方便了快速编写基准测试。
  
  - 然而，该模块存在一些限制，例如在热循环（hot loops）中存在 **不必要的系统调用**，这可能会影响性能。
- **动态模块导入受限于 Mojo 的编译结构**：由于 Mojo 的编译结构将所有内容捆绑为常量和函数，目前不支持模块的 **动态导入**。
  
  - 引入 JIT 编译器是一个潜在的解决方案，但对于二进制文件大小以及与预编译代码的兼容性仍存在担忧。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Hailo 模型量化挑战**：一位成员详细说明了运行 **Hailo** 需要进行八位量化（eight-bit quantization）的复杂性，这增加了训练过程的难度，并且需要编译好的 **.so** 文件才能让 **CUDA** 和 **TensorFlow** 正常工作。
  
  - 由于这些要求，**Hailo** 的环境搭建非常繁琐。
- **ASM2464PD 芯片规格确认**：讨论确认了 **ASM2464PD** 芯片支持通用 **PCIe**，可通过多个供应商获得，且不限于 **NVMe**。
  
  - 成员们对该芯片为实现最佳性能而产生的 **70W** 功耗需求表示担忧。
- **开源 USB4 转 PCIe 转换器进展**：[GitHub](https://github.com/cyrozap/usb-to-pcie-re) 上分享了一个开源的 **USB4/Thunderbolt to M.2 PCIe** 转换器设计，展示了显著进展并获得了硬件开发的资金支持。
  
  - 设计者概述了下一阶段的开发预期，以实现有效的 USB4 到 PCIe 的集成。
- **使用 Opus 编解码器优化音频录制**：成员们讨论了使用 **Opus** 编解码器进行音频录制，因为它能够在不牺牲质量的情况下减小文件体积。
  
  - 然而，有人指出 **Opus** 在浏览器兼容性方面存在问题，凸显了技术局限性。
- **开发 Tinygrad 的 Distributed Systems 库**：一位用户提议为 **Tinygrad** 构建一个 **Distributed Systems** 库，专注于 **dataloaders** 和 **optimizers**，而不依赖于 **MPI** 或 **NCCL** 等现有框架。
  
  - 目标是从零开始创建基础网络功能，同时保持 Tinygrad 现有的接口。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 可总结 AI 通讯邮件**：一位成员建议使用 [NotebookLM](https://notebooklm.google.com/notebook/19d92404-a2a6-4238-b9ad-33854c841aac/audio) 来总结超过 **200 封 AI newsletter 邮件**，以避免手动复制粘贴内容。
  
  - 提到了 Gmail 中的 *Gemini* 按钮可能有助于总结，但指出其**并非免费**。
- **围绕 NotebookLM 的非官方 API 质疑**：用户讨论了一个每月 **30 美元的非官方 API**，并对其合法性表示怀疑 [NotebookLM API](https://notebooklmapi.com/)。
  
  - 担忧包括**缺乏商业信息**和示例输出，导致一些人将其标记为诈骗。
- **集成 KATT 用于播客事实核查**：一位用户讨论了将 **KATT (Knowledge-based Autonomous Trained Transformer)** 集成到其播客的**事实核查器**中，导致单集节目变长。
  
  - 他们将这种集成描述为**痛苦的**，因为它结合了传统方法和新的 AI 技术。
- **NotebookLM 音频文件上传问题**：用户对无法向 NotebookLM 上传 **.mp3 文件**表示沮丧，并得到了通过 [Google Drive](https://drive.google.com) 进行正确上传程序的指导。
  
  - 一些人注意到其他文件类型上传没有问题，表明可能存在**技术故障**或转换错误。
- **在 NotebookLM 中将笔记本导出为 PDF**：用户正在询问未来是否有计划将笔记或笔记本导出为 **.pdf**，并寻求用于笔记本自动化的 API。
  
  - 虽然有人提到使用 **PDF 合并工具**等替代方案，但他们更渴望原生导出功能。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Magentic-One 框架发布**：推出了 **Magentic-One** 框架，展示了一个旨在处理复杂任务并在效率上超越传统模型的多 Agent 系统。
  
  - 它使用一个编排器 (orchestrator) 来指导专业 Agent，并在各种基准测试中表现出**竞争力** [来源](https://x.com/rowancheung/status/1854972388988908023)。
- **Context Autopilot 介绍**：Context.inc 推出了 **Context Autopilot**，这是一款能像用户一样学习的 AI，展示了在信息工作方面的尖端能力。
  
  - 分享了一个实际演示，表明在增强 AI 工作流中的生产力工具方面具有前景 [视频](https://vimeo.com/1017798749)。
- **Writer C 轮融资公告**：Writer 宣布了 **2 亿美元的 C 轮融资**，估值为 **19 亿美元**，旨在增强其 AI 企业解决方案。
  
  - 这笔资金将支持扩展其生成式 AI 应用，并得到了知名投资者的显著支持 [Tech Crunch 文章](https://techcrunch.com/2024/11/12/generative-ai-startup-writer-raises-200m-at-a-1-9b-valuation/)。
- **Supermaven 加入 Cursor**：Supermaven 宣布与 **Cursor** 合并，旨在开发先进的 AI 代码编辑器，并就新的 AI 工具功能展开合作。
  
  - 尽管处于过渡期，**Supermaven 插件**仍将继续维护，表明了对提高生产力的持续承诺 ([博客文章](https://supermaven.com/blog/cursor-announcement))。
- **Dust XP1 和日活跃使用率**：分享了关于如何使用 **Dust XP1** 创建高效工作助手的见解，在客户中实现了令人印象深刻的 **88% 日活跃使用率 (Daily Active Usage)**。
  
  - 本集涵盖了早期的 **OpenAI 历程**，包括关键的合作。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **GPU 显存与速度的权衡**：关于从 **RTX 2060 Super** 升级到 **RTX 3090** 的讨论，权衡了 **GPU 显存**与**处理速度**之间的平衡，以及购买旧的二手 **Tesla 卡**的选项。
  
  - 共识倾向于选择更新的硬件以获得更高的可靠性，特别建议个人开发者不要购买旧款 GPU。
- **Vast.ai 作为云端 GPU 供应商**：**Vast.ai** 被推荐为一种经济实惠的云端 GPU 选择，目前 **A100** 和 **RTX 4090** 等 GPU 的价格在每小时 **$0.30** 到 **$2.80** 之间。
  
  - 用户指出，虽然 **Vast.ai** 提供了具有成本效益的解决方案，但其租赁 GPU 的模式引入了一些潜在用户应考虑的特殊问题。
- **Surfgrad：基于 WebGPU 的 Autograd 引擎**：**Surfgrad** 是一个基于 **WebGPU** 构建的 autograd 引擎，在 M2 芯片上实现了高达 **1 TFLOP** 的性能，详见 [优化 WebGPU Matmul 内核](https://zanussbaum.substack.com/p/optimizing-a-webgpu-matmul-kernel?r=coo9n)。
  
  - 该项目强调内核优化，并作为那些希望在 autograd 库开发中探索 **WebGPU** 和 **TypeScript** 的人员的教育工具。
- **高效深度学习系统资源**：分享了由 HSE 和 YSDA 提供的 [Efficient Deep Learning Systems](https://github.com/mryab/efficient-dl-systems) 课程材料，提供了旨在优化 AI 系统效率的全面资源。
  
  - 参与者强调了该仓库在增强对深度学习中高效系统架构和资源管理的理解方面的价值。
- **Liger 中的多 GPU 同步**：讨论了在 **liger** 的多 GPU 设置中同步均值和方差参数的挑战，参考了 **PyTorch** 的 **SyncBatchNorm** 操作。
  
  - 成员们表示，在 **liger** 中复制 **SyncBatchNorm** 行为将非常复杂且不直接，突显了其中涉及的复杂性。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere API /rerank 问题**：由于 **return_documents** 字段被移除，用户在使用 **/rerank** 端点时遇到了 **UnprocessableEntityError**。
  - 开发团队承认了这一非预期的变更，并正在努力恢复 **return_documents** 参数，因为多位用户在更新 SDK 后报告了相同的问题。
- **Command R 的开发状态**：针对 **Command R** 可能停用的担忧，官方保证目前没有退役该模型的计划。
  - 建议成员使用最新的更新（如 **command-r-08-2024**），以从增强的性能和成本效益中获益。
- **aya_collection 数据集不一致性**：发现了 **aya_collection** 数据集的不一致性，特别是在 19 种语言的翻译质量方面，其中英语有 **249716** 行，而阿拉伯语和法语为 **124858** 行。
  - **translated_cnn_dailymail** 数据集中突显了具体的翻译不匹配问题，英语句子与阿拉伯语和法语的对应部分在比例上不一致。
- **森林火灾预测 AI 项目**：一位成员介绍了他们使用 **Catboost & XLModel** 的**森林火灾预测 AI** 项目，强调了模型在 AWS 上部署的可靠性需求。
  - 建议包括采用最新版本的 **Command R** 以获得更好的性能，并建议联系销售团队以获取额外的支持和更新。
- **研究原型 Beta 测试**：一个支持报告创建等**研究和写作任务**的研究原型已开放限量 **beta** 测试注册，链接在[这里](https://forms.gle/Teis9VwM6eZP6nxVA)。
  - 参与者需要提供**详细且建设性的反馈**，以帮助在早期测试阶段完善工具的功能。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **经济型 AI 家用服务器亮相**：一段 [YouTube 视频](https://www.youtube.com/watch?v=iflTQFn0jx4) 展示了如何使用单块 **3060 GPU** 和一台 **Dell 3620** 搭建高性价比的 AI 家用服务器，并演示了在 **Llama 3.2** 模型上的出色性能。
  - 该方案为运行 LLM 提供了一个低成本选择，使工程师无需巨额硬件投资即可接触先进的 AI 技术。
- **图神经网络主导 NeurIPS 2024**：NeurIPS 2024 重点关注了 **Graph Neural Networks** 和几何学习，提交论文约 **400-500 篇**，超过了 ICML 2024 的提交数量。
  - 关键主题包括扩散模型 (diffusion models)、Transformer、Agent 和知识图谱 (knowledge graphs)，并在理论上强调了**等变性 (equivariance)** 和**泛化性 (generalization)**，详见 [GitHub 仓库](https://github.com/azminewasi/Awesome-Graph-Research-NeurIPS2024)。
- **Qwen2.5 Coder 超越 GPT4o 和 Claude 3.5**：在最近的评估中，**Qwen2.5 Coder 32B** 的表现优于 **GPT4o** 和 **Claude 3.5 Sonnet**，相关分析见此 [YouTube 视频](https://youtu.be/Xs0EkLYu6hw)。
  - 社区认可 Qwen2.5 Coder 的快速进步，将其定位为编程 AI 领域的强力竞争者。
- **高级电子商务 Embedding 模型发布**：新的**电子商务 Embedding 模型**已发布，其性能超越 **Amazon-Titan-Multimodal** 高达 **88%**，可在 **Hugging Face** 上获取并集成到 **Marqo Cloud**。
  - 详细的功能和性能指标可以在 [Marqo-Ecommerce-Embeddings 集合](https://huggingface.co/collections/Marqo/marqo-ecommerce-embeddings-66f611b9bb9d035a8d164fbb)中找到，有助于开发强大的电子商务应用。
- **讨论创新的图像去噪技术**：论文 *Phase Transitions in Image Denoising via Sparsity* 现已在 [Semantic Scholar](https://www.semanticscholar.org/paper/Phase-Transitions-in-Image-Denoising-via-Sparsely-Carroll-Carlson/55cb0e93f4f98b851ca4343e4a456b2e9c8241ec) 上线，提出了解决图像处理挑战的新方法。
  - 该研究为提升图像去噪方法做出了贡献，解决了保持图像质量的关键问题。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **PursuitGov 使用 LlamaParse 增强 B2G 服务**：通过采用 **LlamaParse**，**PursuitGov** 在一个周末内成功解析了 **400 万页** 文档，显著增强了其 B2G 服务。
  
  - 这一转变使复杂文档格式的准确率提升了 **25-30%**，使客户能够从公共部门数据中**挖掘隐藏的机会**。
- **集成 ColPali 进行高级重排序**：一位成员分享了使用 **ColPali** 作为重排序器 (re-ranker) 的见解，以在**多模态索引 (multimodal index)** 中实现高度相关的搜索结果。
  
  - 该技术利用 **Cohere 的多模态嵌入 (multimodal embeddings)** 进行初始检索，整合文本和图像以获得最佳结果。
- **Cohere 的新多模态嵌入功能**：团队讨论了 **Cohere 的多模态嵌入**，强调了其有效处理文本和图像数据的能力。
  
  - 这些嵌入正与 **ColPali** 集成，以增强搜索相关性和整体模型性能。
- **自动化 LlamaIndex 工作流流程**：一位成员对繁琐的发布过程表示不满，旨在实现更多自动化，并分享了一个 [LlamaIndex v0.11.23 的 GitHub pull request](https://github.com/run-llama/llama_index/pull/16919)。
  
  - 他们强调需要简化工作流，以减少人工干预并提高部署效率。
- **优化 FastAPI 的流式响应**：围绕使用 **FastAPI 的 StreamingResponse** 展开了讨论，担心事件流延迟可能是由于协程调度 (coroutine dispatching) 问题引起的。
  
  - 成员们建议使用高级流式传输技术，例如使用 `llm.astream_complete()` 将每个 token 作为流事件写入，以增强性能。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **AI 公司拥抱大猩猩式营销 (Gorilla Marketing)**：一位成员指出 **AI 公司非常喜欢大猩猩式营销**，这可能是指非传统的促销策略，并分享了一个有趣的[大猩猩挥舞美国国旗的 GIF](https://tenor.com/view/harambe-america-murica-flag-waving-gif-17339298)。
  
  - 这突显了 AI 行业内独特且富有创意的营销策略的使用。
- **寻求目标检测项目帮助**：一位成员详细介绍了一个涉及使用 **Python Django** 进行**空调目标检测**的项目，旨在识别空调类型和品牌。
  
  - *他们寻求帮助*，表明在开发此识别功能方面需要支持。
- **为代码生成模型引入 GitChameleon**：新数据集 [GitChameleon: Unmasking the Version-Switching Capabilities of Code Generation Models](https://arxiv.org/abs/2411.05830) 引入了 **116 个基于特定库版本的 Python 代码补全问题**，并配有可执行的单元测试，以严格评估 LLM 的能力。
  
  - 这旨在解决现有基准测试忽略软件库演进的动态特性且未评估实际可用性的局限性。
- **激动人心的 SCAR 概念检测发布**：**SCAR** 是一种在 LLM 中进行精确概念检测和引导的方法，它以监督方式使用 Sparse Autoencoders 学习**单语义特征 (monosemantic features)**。
  
  - 它为**毒性、安全性和写作风格**等概念提供了强大的检测能力，并可在 Hugging Face 的 transformers 中进行实验。
- **NVIDIA 关于噪声频率训练的论文**：NVIDIA 的论文提出了一个概念，即在正向加噪步骤中，**高空间频率**比低频率加噪更快。
  
  - 在反向去噪步骤中，模型被显式训练为从**低频到高频**工作，提供了一种独特的训练方法。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **为自定义模型重写测试用例**：一位成员询问在修改 handler 后，如何为自定义模型重写或重新运行测试用例。
  - 另一位成员建议删除 `result` 文件夹中的结果文件，或更改 `constant.py` 中的路径以保留旧结果。
- **Qwen-2.5 输出中的无效 AST 错误**：一位成员描述了在微调 **Qwen-2.5 1B** 模型时遇到的问题，尽管模型输出有效，但仍会导致 **INVALID AST 错误**。
  - 成员们讨论了一种特定的错误输出格式，其中包含一个未匹配的反括号，这表明存在语法问题。
- **对 JSON 结构输出的困惑**：一位成员对模型输出 **JSON 结构** 而非预期的函数调用格式表示困惑。
  - 其他成员澄清说，**QwenHandler** 理想情况下应该将 JSON 结构转换为函数形式，从而引发了关于输出预期的讨论。
- **评估量化微调模型**：一位成员提出了关于评估量化微调模型的问题，特别是关于它们在 **vllm** 上的部署。
  - 他们提到在模型服务中使用特定参数，如 `--quantization bitsandbytes` 和 `--max-model-len 8192`。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **LLM Agents MOOC 黑客松启动**：**LLM Agents MOOC 黑客松**于 **太平洋时间 11/12 下午 4 点**开始，通过实况 [LambdaAPI 演示](https://youtube.com/live/EUzVW6oRpIo?feature=share) 协助参赛者开发项目。
  - 共有约 **2,000 名创新者**报名参加了 **Applications** 和 **Benchmarks** 等赛道，活动由 [rdi.berkeley.edu/llm-agents-hackathon](https://rdi.berkeley.edu/llm-agents-hackathon) 主办。
- **LambdaAPI 演示支持黑客松项目**：**LambdaAPI** 提供了实操[演示](https://youtube.com/live/EUzVW6oRpIo?feature=share)，指导黑客松参与者构建高效的 LLM Agent 应用。
  - 这些演示提供了可操作的工具和技术，帮助开发者完善其项目实现。
- **NVIDIA 的具身智能引发伦理辩论**：**NVIDIA 关于具身智能 (Embodied AI)** 的演讲引发了关于是否赋予类人 AI 系统道德权利的讨论。
  - 参与者强调了对**规范对齐 (Normative Alignment)** 关注的缺失，并对 AI 进步的伦理边界提出了质疑。
- **AI 权利与规范对齐担忧**：社区对 AI 开发中缺乏**规范对齐**讨论表示不安，尤其是在 **NVIDIA** 分享见解之后。
  - 辩论集中在 AI 权利的伦理影响上，强调了对全面对齐策略的需求。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **FOSDEM AI DevRoom 定于 2025 年举行**：**AIFoundry 团队**正在筹备计划于 **2025 年 2 月 2 日**举行的 [**FOSDEM AI DevRoom**](https://aifoundry.org/fosdem-2025-low-level-ai-engineering-hacking-dev-room)，重点关注 **ggml/llama.cpp** 及相关项目，旨在团结 **AI 贡献者和开发者**。
  - 他们正在邀请**底层 AI 核心开源项目维护者**提交提案，截止日期为 **2024 年 12 月 1 日**，并为引人入胜的主题提供潜在的**差旅补贴**。
- **Axolotl 微调利用 Alpaca 格式**：一位用户阐明了使用 **Axolotl 进行微调**的设置过程，强调使用 **Alpaca 格式**的数据集进行训练预处理。
  - 有人指出 **tokenizer_config.json** 缺少 **chat template 字段**，需要进一步调整以完成完整配置。
- **通过聊天模板增强 Tokenizer 配置**：一位成员分享了一种通过复制特定 JSON 结构将 **chat template** 合并到 **tokenizer config** 中的**方法**。
  - 他们建议修改 **Axolotl** 内的设置，以确保在未来的配置中自动包含聊天模板。
- **在微调中集成默认系统提示词**：发出了一个提醒，指出共享模板缺少 **Alpaca** 的默认系统提示词，这可能需要调整。
  - 用户被告知可以在 **\### Instruction** 之前包含条件语句，以有效地集成所需的提示词。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **注解增强 dspy.Signature**：成员们讨论了在 **dspy.Signature** 中使用 **annotations**（注解）的情况，澄清了虽然 **基础注解** 可以工作，但使用 **list[MyClass]** 等 **custom types**（自定义类型）也具有潜力。
  
  - 一位成员确认字符串形式在此用途下不起作用，建议优先使用 **显式类型定义**。
- **为临床实体实现自定义签名**：一位成员分享了在输出中使用 **字典列表** 实现 **custom signature**（自定义签名）的成功案例，展示了对 **临床实体** 的提取。
  
  - 该实现包括对输入和输出字段的 **详细描述**，表明了定义 **复杂数据结构** 的 **灵活方法**。

 

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Linux Mint 在虚拟机中运行困难**：在 **Virtual Machine Manager** 中安装 **Linux Mint** 后，用户报告网络无法正常工作。
  
  - 不过，有人尝试在一个名为 **Boxes** 的应用中安装 **Linux Mint**。
- **Microsoft Copilot 沟通故障**：与 **Microsoft Copilot** 的反复交互显现出挫败感，因为命令未按要求进行配置。
  
  - 用户强调没有创建桥接，但他们设法自行创建了一个。
- **OS X 上的 Interpreter CLI Bug**：有报告称 **OS X** 上的 **Interpreter CLI** 存在文件持久化并意外退出的问题。
  
  - 用户对这些问题在 **developer branch**（开发分支）上频繁发生表示担忧。

 

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **PyTorch 团队将发布 DCP PR**：whynot9753 宣布 **PyTorch** 团队可能会在明天发布一个 DCP PR。
- \*\*\*\*:

 

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **请求继续使用微调模型**：一位用户请求继续使用他们的 **fine-tuned models**（微调模型）。
- **请求继续使用微调模型**：一位用户请求继续使用他们的 **fine-tuned models**（微调模型）。

 

---

**Alignment Lab AI Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道详细摘要和链接

{% if medium == 'web' %}

 

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1305648836718497872) (35 messages🔥):

> - `工作流设计中减少点击`
> - `AI 模型评估`
> - `AI 中的情商`
> - `Text-MIDI 多模态数据集`
> - `AI 开发中的用户反馈`

- **减少点击使设计变得混乱**：成员们讨论了设计中 **减少点击** 往往只利于高层的工作流，却导致效率低下，其中一人指出，资深专业人士的上手过程可能需要 **两个月**。
  
  - *然而，增加点击会降低转化率和收入，* 一位用户评论道，并对这些设计选择的价值提出质疑。
- **揭示 AI 评估中的挑战**：一位新手询问在哪里可以寻求使用和评估其 AI 模型的帮助，想知道框架内有哪些可用的最佳资源。
  
  - 另一位用户将他们引导至专门负责评估的频道，表明可以提供支持。
- **情商增强 AI 响应**：讨论强调了 **情绪检测** 和 **情感分析** 对 AI 模型的潜力，强调了它们在当今应用中的现有用途。
  
  - 一位用户建议，引入 **轻度 RLAIF** 有助于确保 AI 生成的输出具有更好的音乐形式。
- **建议 Text-MIDI 多模态数据集**：一位参与者提议，Text-MIDI 多模态数据集可能是 AI 开发的下一步，并暗示了现有的录音和元数据集合。
  
  - 他们承认版权限制，透露只有 MIDI 文件可以开源。
- **反馈循环对 AI 开发至关重要**：讨论了为 AI 实施 **反馈系统** 的重要性，用户评分和评论可以帮助随着时间的推移优化响应。
  
  - 这种迭代学习过程旨在显著提高情商和整体输出质量。

 

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1305623518414770216) (271 条消息🔥🔥):

> - `Gradient Descent and Optimization`
> - `Muon and Feature Learning`
> - `Second Order Methods`
> - `Newton's Method`
> - `Saddle Points in Optimization`

- **探索 Gradient Descent 机制**：围绕 Gradient Descent 更新机制展开了讨论，重点关注更新的投影和范数（norms）如何与模型中的权重变化相关联。
  
  - 参与者辩论了根据输入变化缩放更新的重要性，以及二阶信息在实现最优收敛中的相关性。
- **Muon 优化的重要性**：探讨了 Muon 作为优化器的作用，特别是它与 Feature Learning 的交互，以及它如何影响网络训练动态。
  
  - 有建议探索 Muon 与其他理论框架（如 kernel dynamics 和 Feature Learning 文献）之间的联系。
- **Second Order Methods 面临的挑战**：参与者对 Newton's Method 等二阶方法在高维、非凸优化景观（landscapes）中的适用性表示担忧，因为其中普遍存在 Saddle Points。
  
  - 讨论强调，虽然二阶方法能捕捉曲率数据，但在涉及噪声和变化梯度的场景中，其实际效用可能有限。
- **理解噪声环境下的 Saddle Points**：有人指出，在噪声随机梯度下降（SGD）背景下，Saddle Points 的相关性可能较低，并引用了历史结果，证明噪声有助于 SGD 逃离 Saddle Points。
  
  - 参与者强调将重点转向能够适应受噪声影响的高维景观复杂性的优化技术。
- **优化中的计算考量**：对话涉及与高阶导数相关的计算挑战及其对优化策略的实际影响。
  
  - 尽管使用二阶及更高阶信息有理论支持，但开发可行的计算方法仍然是开发有效算法的关键。

**提及的链接**：

- [High-dimensional Asymptotics of Feature Learning: How One Gradient Step Improves the Representation](https://arxiv.org/abs/2205.01445)：我们研究了两层神经网络中第一层参数 $\boldsymbol{W}$ 的第一次 Gradient Descent 步骤：$f(\boldsymbol{x}) = \frac{1}{\sqrt{N }}\boldsymbol{a}^\topσ(\boldsymbol{W}^\top\b...
- [A Spectral Condition for Feature Learning](https://arxiv.org/abs/2310.17813)：训练更大神经网络的压力推动了对大网络宽度下初始化和训练的研究。一个关键挑战是缩放训练，使网络的内部表示...
- [Grokking as the Transition from Lazy to Rich Training Dynamics](https://arxiv.org/abs/2310.06110v3)：我们提出 Grokking 现象（即神经网络的训练损失远早于测试损失下降）可能是由于神经网络从 lazy training dynamics 转变...
- [Geometric Dynamics of Signal Propagation Predict Trainability of Transformers](https://arxiv.org/abs/2403.02579)：我们研究了深度、随机初始化的 Transformer 中的前向信号传播和梯度反向传播，得出了关于初始化超参数的简单充要条件...
- [Rigorous dynamical mean field theory for stochastic gradient descent methods](https://arxiv.org/abs/2210.06591)：我们证明了一系列基于一阶梯度的学习估计器（如 M-estimator、浅层神经网络...）方法的精确高维渐近性的闭式方程...
- [Newton's method in optimization - Wikipedia](https://en.wikipedia.org/wiki/Newton's_method_in_optimization#Higher_dimensions))：未找到描述
- [Flex attention underperforms SDPA (cuDNN), constructing T5 attention bias via embedding weights · Issue #138493 · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/138493#issuecomment-2433345005>)：🐛 Bug 描述：我一直尝试在 flex_attention 中实现 T5 encoder 相对注意力偏置。我为此提出了几种算法和一个基准测试脚本：https://gist.github.com/Birc...

---

### **Eleuther ▷ #**[**scaling-laws**](https://discord.com/channels/729741769192767510/785968841301426216/1305922773490602035) (10 条消息🔥):

> - `Scaling Laws Investigation` (Scaling Laws 调查)
> - `Learning Rate Adjustment` (学习率调整)
> - `Line Search Techniques` (Line Search 技术)
> - `Gradient Descent Dynamics` (梯度下降动力学)

- **填补 Scaling Laws 中的空白**：一位成员分享了关于[填补 Scaling Laws 缺失环节](https://docs.google.com/document/d/1jL0-82COewU-UDyzCt-vFgpBTBr9AiBvwGlShmfPpew/edit?tab=t.0)的见解，强调了将理论与应用联系起来时的实际挑战。
  
  - *真正有效的东西是不允许发表的*，这突显了在研究的有效应用中所面临的问题。
- **收敛学习率的依赖性**：讨论指出，为了保持**收敛训练 (Convergent Learning)**，当 Batch Size 降低时，两个学习率都必须根据 **NQM 模型**的预测而降低。
  
  - 该成员指出 *收敛学习率与 Batch Size 无关*，这挑战了传统方法。
- **利用 Line Search 优化学习率**：有人推测将 **Line Search** 作为一种在训练期间恢复最佳学习率的方法，特别是在 Loss 接近发散时。
  
  - 一位贡献者引用了相关发现，即 **Line Search 产生的速率**约为**更新范数 (Norm of the update) 的 1/2**，暗示了可能存在的模式。
- **Line Search 方法的改进**：一位成员引用了一篇论文，该论文提出改进 **Armijo Line Search** 方法，通过整合来自 **ADAM 的动量以获得更好的性能**。
  
  - 他们的方法表现出了显著的效率，特别是在跨不同数据领域的**大规模训练**场景中。
- **观察到波动的学习率行为**：讨论揭示了在 **Greedy Line Search** 中观察到**波动的学习率行为**，特别是在使用如 **x^2 + 1/2y^2 + 1/3z^2** 等函数时。
  
  - 一条相关的推文指出了一些反直觉的结果，表明周期性的步长可能比以前认为的系统产生更好的速率。

**提到的链接**：

- [Improving Line Search Methods for Large Scale Neural Network Training](https://arxiv.org/abs/2403.18519)：在最近的研究中，Line Search 方法在传统随机梯度下降技术的性能上表现出显著改进，消除了对特定学习率的需求...
- [Disentangling Adaptive Gradient Methods from Learning Rates](https://arxiv.org/abs/2002.11803)：我们调查了深度学习优化算法评估中的几个混淆因素。主要深入研究了自适应梯度方法如何与学习率相互作用...
- [Which Algorithmic Choices Matter at Which Batch Sizes? Insights From a Noisy Quadratic Model](https://arxiv.org/abs/1907.04164)：增加 Batch Size 是加速神经网络训练的常用方法，但超过某个临界 Batch Size 后，更大的 Batch Size 收益会递减。在这项工作中，我们研究了临界...
- [Surge Phenomenon in Optimal Learning Rate and Batch Size Scaling](https://arxiv.org/abs/2405.14578)：在当前的深度学习任务中，Adam 风格的优化器（如 Adam, Adagrad, RMSProp, Adafactor 和 Lion）已被广泛用作 SGD 风格优化器的替代方案。这些优化器通常更新...
- [Tweet from Ben Grimmer (@prof_grimmer)](https://x.com/prof_grimmer/status/1679846891171766272)：我证明了我职业生涯中最奇怪的结果。梯度下降的速率在恒定步长 1/L 下最好的经典观点是错误的。我们需要 (0,2/L) 范围内的步长来保证收敛的观点...
- [Eleuther copy of Uncovering limits of data-parallel scaling](https://docs.google.com/document/d/1jL0-82COewU-UDyzCt-vFgpBTBr9AiBvwGlShmfPpew/edit?tab=t.0)：揭示数据并行扩展的极限（2024年11月11日）。动机 TLDR：我们需要知道 AI 训练负载如何水平扩展。进行这种估算所需的工具在开源领域尚不存在...

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1305742495845584907) (5 messages):

> - `Custom Task Issues`
> - `Limit Samples in Evaluation`
> - `Metrics in YAML Configuration`

- **自定义任务问题已解决**：一名成员报告解决了与 **custom task** 相关的问题，原因是忘记在 YAML 文件中包含 **multiple choice tag**。
  
  - *NVM I found the issue* 是他们在最初困惑后的回复。
- **使用 --limit N 进行样本评估**：另一位成员强调，使用 `--limit N` 可以有效地 **限制评估的样本数量**。
  
  - 该方法为调整评估范围提供了灵活性，以便更好地管理任务。
- **YAML 中指标配置的挑战**：一位用户表达了在同一个新 QA 任务的 YAML 配置中同时包含 **acc_norm** 和 **exact_match** 指标的困难。
  
  - 他们寻求帮助，并询问是否有任何 **similar tasks** 成功地同时包含了这两个指标。

---

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1305623515507982487) (230 messages🔥🔥):

> - `Perplexity Technical Issues`
> - `User Experience with Perplexity`
> - `Perplexity Pro Subscription Details`
> - `Perplexity Model Comparison`
> - `Feedback on Mac App and Features`

- **Perplexity 用户对持续的技术问题感到沮丧**：用户对持续存在的技术问题表示不满，特别是长线程中隐藏消息的 bug 仍未解决，而其他 bug 似乎已经修复。
  
  - 评论指出，该应用一个多月来一直面临多个问题，严重影响了用户体验。
- **Perplexity Pro 订阅详情需澄清**：一位用户询问其免费一年 Perplexity Pro 的状态，询问到期后是否会继续，以及对 R1 设备的影响。
  
  - 回复确认订阅在试用期结束后不会保持免费，用户将恢复到有限的免费搜索次数。
- **Perplexity 模型对比**：关于哪种 Perplexity 模型表现最好的讨论正在进行中，用户注意到 gpt-o 在某些任务中似乎最有效。
  
  - 观点各异，有评论提到 o1 虽然专业但应用场景有限。
- **用户报告 Mac 应用的 UI 问题**：几位用户报告了 Perplexity Mac 版应用的 UI 问题，强调缺少滚动条使得导航变得繁琐。
  
  - 投诉还包括 Google 登录的持续问题，以及网页版中存在但应用中缺失的功能。
- **关于外部工具和扩展的讨论**：用户分享了在 Perplexity 中使用浏览器扩展和其他工具的经验，并指出这些工具如何在不直接违反条款的情况下增强功能。
  
  - 大家对这些扩展如何影响性能和功能感到好奇，特别是针对目前存在的技术限制。

**提及的链接**：

- [The Anatomy of a Search Engine](http://infolab.stanford.edu/~backrub/google.html)：未找到描述
- [Phi Hoang (@apostraphi) 的推文](https://x.com/apostraphi/status/1856093208524005715?s=61)：未找到描述
- [Chat-with-OpenAI-o1 - yuntian-deng 的 Hugging Face Space](https://huggingface.co/spaces/yuntian-deng/o1)：未找到描述
- [Doctorevil No GIF - Doctorevil No - 发现并分享 GIF](https://tenor.com/view/doctorevil-no-gif-22331678)：点击查看 GIF

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1305665116850749502) (12 条消息🔥):

> - `US NATO Membership` (美国 NATO 成员身份)
> - `Bitcoin Market Predictions` (Bitcoin 市场预测)
> - `TSMC Chip Shipments` (TSMC 芯片出货)
> - `China vs US Trade War` (中美贸易战)
> - `AI Winter Trends` (AI Winter 趋势)

- **美国是 NATO 的成员吗？**：一位成员分享了一个讨论 **USA** 参与 **NATO** 情况的链接，重点关注历史背景和当前的现实意义。
  
  - 该资源探讨了在不断变化的全球动态中，**USA** 对 **NATO** 承诺的影响。
- **Bitcoin 预计到 2024 年将达到 $100,000**：关于 **Bitcoin** 到 **2024** 年底可能达到 **$100,000** 的预测引发了热烈讨论。
  
  - 成员们分享了各自复杂的感受，引发了关于市场趋势和投资策略的辩论。
- **TSMC 停止向中国出货芯片**：一段名为 *'TSMC Halts Chinese Chip Shipments, Beatles Make AI History...'* 的 YouTube 视频强调了芯片市场的重大中断。
  
  - 该视频详细介绍了对技术和贸易的更广泛影响，敦促观众反思行业转型。
- **关于中美贸易战的见解**：一位成员发布了一个链接，概述了 **中美贸易冲突**，列出了关键事件和影响。
  
  - 该资源及时提醒人们关注影响国际市场的持续紧张局势。
- **AI Winter 的历史**：分享了一个讨论多年来 **AI Winter** 趋势的链接，分析了 AI 炒作与失望的周期性特征。
  
  - 成员们反思了历史模式，思考我们是否正在接近另一个停滞阶段。

 

**提到的链接**：[TSMC Halts Chinese Chip Shipments, Beatles Make AI History with Grammy Noms, and How the Body Sto...](https://youtu.be/Z4xIJDL3e10?si=v3N0MLpytyt5r_QN)：您还想看到什么？请告诉我们！([https://www.buzzsprout.com/twilio/text_messages/2302487/open_sms](https://www.buzzsprout.com/twilio/text_messages/2302487/open_sms)) 在今天的节目中，我们探讨了 TSMC 的重大...

 

---

### **Perplexity AI ▷ #**[**pplx-api**](https://discord.com/channels/1047197230748151888/1161802929053909012/1305925710053118033) (3 条消息):

> - `Pplx API DailyBot Custom Command Editor` (Pplx API DailyBot 自定义命令编辑器)
> - `AI Limitations` (AI 局限性)
> - `Webhook Implementation` (Webhook 实现)
> - `CodeSandBox VM Usage` (CodeSandBox VM 使用)

- **关于 Pplx API DailyBot 命令编辑器的咨询**：一位团队成员询问是否有人有实现 **Pplx API DailyBot 自定义命令编辑器** 的经验，并请求启动该项目的基本指导。
  
  - 另一位成员幽默地建议使用 AI 寻求帮助，反映了实现过程中的不确定性。
- **AI 在处理 DailyBot 命令时的困难**：**malagoni31** 表达了挫败感，因为包括 **OpenAI** 和 **Claude** 在内的各种 AI 都无法针对 DailyBot 的内置命令提供准确的指导。
  
  - 他们分享了一个使用 **CodeSandBox VM** 的变通方案，涉及使用 **Webhook** 从转发端口检索信息进行处理。
- **Webhook 检索过程详解**：该变通方案包括通过 **Webhook** 请求代码以从转发端口获取信息，并在 DailyBot 编辑器中使用另一段代码进行处理。
  
  - 最终，Prompt 会在 **Discord** 聊天中返回，展示了在挑战面前一种极具创意的集成方法。
- **寻求替代方案**：malagoni31 请求社区提供建议，看是否有人知道实现 DailyBot 命令编辑器的更好方法。
  
  - 他们强调了分享知识的价值，无论当前方法是否可行，或者是否存在改进方案。

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**general**](https://discord.com/channels/1179035537009545276/1179035537529643040/1305626594554941554) (135 条消息🔥🔥):

> - `Qwen 2.5 Coder 微调`
> - `使用数据集进行改进`
> - `Unsloth 模型修复`
> - `模型中的 Function calling`
> - `聊天历史与记忆保留`

- **Qwen 2.5 Coder 微调资源已发布**：新的 Qwen 2.5 Coder (14B) 微调 notebook 已上线，允许用户在 Colab 上免费微调模型，并提升了效率。
  
  - Unsloth 减少了 60% 的 VRAM 占用，并将上下文长度从 32K 扩展到了 128K。
- **显著改进所需的数据集大小**：建议数据集理想情况下应包含至少 **100 行**，若要获得更好的模型微调效果，建议使用 **300 行以上**。
  
  - 建议选择 starcoder 等高质量数据集，以提升软件质量指标。
- **Qwen 2.5 模型的 Bug 修复**：宣布了针对 Qwen 2.5 模型的最新 Bug 修复，详细说明了诸如不当的 pad tokens 导致无限生成等问题。
  
  - 鼓励用户使用 Unsloth 版本以获得准确的结果，因为早期模型中各种未训练的 tokens 问题已得到解决。
- **Function calling 能力**：讨论了 Unsloth 是否原生支持 function calling，并确认 Unsloth 推理（inference）并不直接支持它。
  
  - 针对 function calling 进行训练是可行的，具有特定 tokenizer 配置的模型可能在这方面有所帮助。
- **聊天历史与记忆管理**：为了实现对先前对话的记忆保留，建议用户开发或使用现有的系统来存储或查询聊天历史以获取上下文。
  
  - 对于即时上下文，建议根据所使用的模型构建正确模板的聊天历史。

**提到的链接**：

- [Google Colab](https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing)：未找到描述
- [Qwen2.5 Coder Artifacts - a Hugging Face Space by Qwen](https://huggingface.co/spaces/Qwen/Qwen2.5-Coder-Artifacts)：未找到描述
- [unsloth/Qwen2.5-Coder-0.5B · Hugging Face](https://huggingface.co/unsloth/Qwen2.5-Coder-0.5B)：未找到描述
- [Qwen 2.5 Coder All Versions - a unsloth Collection](https://huggingface.co/collections/unsloth/qwen-25-coder-6732bc833ed65dd1964994d4)：未找到描述
- [Tweet from Unsloth AI (@UnslothAI)](https://x.com/UnslothAI/status/1856424217610465783)：现在可以在 Colab 上免费微调 Qwen-2.5-Coder-14B 了！Unsloth 让微调速度提升 2 倍，并减少 60% 的 VRAM 占用且无精度损失。我们通过 YaRN 将上下文长度从 32K 扩展到 128K...
- [Tweet from ifioravanti (@ivanfioravanti)](https://x.com/ivanfioravanti/status/1856136182960173315)：Qwen 2.5 Coder Q4 M4 Max 推理测试。Apple MLX vs Ollama：- MLX: 23.97 toks/sec 🥇🔥 - Ollama: 18.33 toks/sec 🥈 这里有视频展示结果。
- [Tim And Eric Awesome Show GIF - Tim And Eric Awesome Show Kisses - Discover & Share GIFs](https://tenor.com/view/tim-and-eric-awesome-show-kisses-kiss-yeah-gif-18128201)：点击查看 GIF
- [Google Colab](https://colab.research.google.com/drive/1nOnpNubkGL5lZhKUBkFOWE5UajzieqCD?usp=sharing)：未找到描述
- [Tweet from Daniel Han (@danielhanchen)](https://x.com/danielhanchen/status/1856442699689414970)：Qwen 2.5 的 Bug 修复与分析：1. Pad_token 不应为 <|endoftext|>（会导致无限生成） 2. Base 模型的 <|im_start|> <|im_end|> 未经训练 3. Embedding 的 PCA 具有 BPE 层级 4. YaRN ...
- [optillm/optillm.py at main · codelion/optillm](https://github.com/codelion/optillm/blob/main/optillm.py#L248)：为 LLM 优化的推理代理。通过在 GitHub 上创建账号为 codelion/optillm 的开发做出贡献。

---

### **Unsloth AI (Daniel Han) ▷ #**[**off-topic**](https://discord.com/channels/1179035537009545276/1179039861576056922/1305691414868525086) (27 messages🔥):

> - `Diet Choices` (饮食选择)
> - `Meal Frequency` (进食频率)
> - `Keto Diet` (生酮饮食)

- **一日一餐令人满足**：一位成员分享说他们每天只吃**一顿饭**，主要由**肉类和一些蔬菜**组成，准备时间大约只需 **5-6 分钟**。
  
  - 他们指出，虽然年轻时饮食可以更随意，但随着年龄增长，饮食习惯已转变为更注重蛋白质而非碳水化合物。
- **生酮饮食 (Keto Diet) 见解**：另一位参与者讨论了他们的**生酮饮食**，强调了碳水化合物的缺失以及对**高蛋白饮食**的需求。
  
  - 他们提到，偶尔*更换食物选项*对于维持饮食平衡非常重要。
- **沙拉和纤维对健康的有益影响**：一位成员建议加入**混合沙拉**，以减少碳水化合物摄入，同时增加**蛋白质**和纤维的摄入量。
  
  - 他们强调，滋养**肠道细菌**对整体健康至关重要，早晨食用燕麦有助于降低食欲。
- **一日一餐会饿吗？**：针对一日一餐的讨论，一位成员询问这种饮食方式是否会感到**饥饿**。
  
  - 原发言成员自信地表示，在这种进食频率下，他们**一点也不觉得饿**。

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**help**](https://discord.com/channels/1179035537009545276/1179777624986357780/1305630363682345034) (49 messages🔥):

> - `Model Saving and Checkpoints` (模型保存与 Checkpoints)
> - `RAM Usage During Training` (训练期间的 RAM 占用)
> - `Fine-tuning Practices` (微调实践)
> - `Data Formatting for Models` (模型数据格式化)
> - `Training Dataset Size and Performance` (训练数据集大小与性能)

- **保存 Checkpoints 并使用 Adapters**：一位用户成功地使用 `PeftModel` 类在基础模型之上集成了 Adapter 模型进行推理，并强调了先构建 Adapter 的必要性。
  
  - 解决方案包括在加载模型进行推理或进一步训练时指定 Checkpoint 路径。
- **管理训练运行期间的 RAM 占用**：一位用户报告在运行 Gemma 2B 时 RAM 使用量不断增加，这表明评估 (Evaluation) 过程可能会影响内存消耗。
  
  - 另一位用户询问是否正在进行评估，暗示这可能会加剧内存需求。
- **微调 (Fine-tuning) 最佳实践**：在使用自定义数据集微调 Qwen 模型时，在保存前确保 LoRA Adapters 与基础模型合并对于部署至关重要。
  
  - 用户讨论了微调模型的进展，以避免灾难性遗忘 (Catastrophic Forgetting) 并保持性能。
- **为模型训练格式化数据**：围绕使用 ShareGPT 格式进行训练数据展开了讨论，并针对预测时的最佳性能进行了调整。
  
  - 用户探索了潜在的模板格式，以提高不同模型类型之间预测的一致性。
- **可视化训练损失 (Training Loss)**：建议利用 WandB 或 TensorBoard 等工具在微调模型时可视化训练和验证损失。
  
  - 这些工具可以帮助用户在整个训练过程中跟踪性能指标。

 

**提到的链接**：[Errors | Unsloth Documentation](https://docs.unsloth.ai/troubleshooting/errors)：若要修复环境配置中的任何错误，请参阅下文。

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**showcase**](https://discord.com/channels/1179035537009545276/1179779344894263297/1305681415937200178) (7 messages):

> - `Integration Calls` (集成通话)
> - `Inference Strategies` (推理策略)
> - `Fast Apply Model` (快速应用模型)
> - `Community Interaction` (社区互动)

- **集成通话提议**：一名成员提议讨论集成机会，并鼓励他人在[此处](https://scheduler.zoom.us/gabriel-peracio/cto)预约通话。该邀请面向任何对此类讨论感兴趣的人，并幽默地提到这相当于公开了自己的真实身份。
  
  - 可以进行*闲聊*或通过消息联系进行非正式交流，尽管他们可能无法快速回复。
- **更快速推理的策略**：另一位成员分享了提高推理速度的策略，包括 **speculative decoding**、使用 **FP8 或 int8** 代替 BF16，以及实现**自定义优化的 CUDA kernels**。他们还提到 **tensor parallelism** 是提高速度的一种方法，尽管会以牺牲吞吐量为代价。
  
  - 一些供应商，如 **qroq** 和 **Cerebras**，已经开发了**定制硬件**来进一步优化性能。
- **使用 Unsloth 进行 LoRA 微调**：针对 **VSCode 性能**问题，codelion_ 建议使用名为 [Llama-3.2-1B-FastApply](https://huggingface.co/patched-codes/Llama-3.2-1B-FastApply) 的 LoRA 微调模型以实现更快的推理。他们提供了使用 Unsloth 加载 adapter 并原生启用快速推理的示例代码。
  
  - 建议的模型能高效处理原始代码和修改后的代码，由于其体积较小，展示了更快的执行速度。

**提到的链接**：

- [patched-codes/Llama-3.2-1B-FastApply · Hugging Face](https://huggingface.co/patched-codes/Llama-3.2-1B-FastApply)：未找到描述
- [Zoom Scheduler](https://scheduler.zoom.us/gabriel-peracio/cto)：未找到描述

---

### **Unsloth AI (Daniel Han) ▷ #**[**research**](https://discord.com/channels/1179035537009545276/1257011997250424842/1305712217844482081) (14 messages🔥):

> - `Tuning thoughts vs. outputs` (微调思考过程 vs. 输出结果)
> - `Analysis of wrong outputs` (错误输出分析)
> - `Chain of Thought (COT) errors` (思维链错误)
> - `Generating profound tweets` (生成深刻推文)

- **探索单独微调思考过程**：一场讨论质疑了在 AI 模型中将**思考过程 (thoughts)**与输出结果分开微调的可行性和意义。
  
  - 共识倾向于认为，将两者分开确实对生成更好的结果具有重要意义。
- **糟糕的输出预示着思维问题**：讨论指出，**糟糕的输出**通常意味着思维缺陷，这表明思维链 (COT) 中的错误通常会导致错误的结论。
  
  - 一位成员强调，模型得出错误结论表明 COT 中的步骤是错误的。
- **模型与错误结论**：对话强调，模型产生与**思考过程**相矛盾的结论，比基于错误推理得出错误答案更具问题。
  
  - 这反驳了输出应始终与初始思考过程保持一致的观点，承认了其中涉及的复杂性。
- **生成深刻的帖子**：有人建议设计能够将科学与精神主题融合的 prompt，以创建**深刻的推文**。
  
  - 这个想法展示了一种创造性的内容生成方式，通过连接不同的领域来引发更深层次的讨论。

 

---

### **aider (Paul Gauthier) ▷ #**[**general**](https://discord.com/channels/1131200896827654144/1131200896827654149/1305647328064966677) (155 messages🔥🔥):

> - `Qwen 2.5 Coder Performance` (Qwen 2.5 Coder 性能)
> - `Aider Installation and Usage` (Aider 安装与使用)
> - `Model Comparison` (模型对比)
> - `Context Handling in Aider` (Aider 中的上下文处理)
> - `Feature Suggestions for Aider` (Aider 功能建议)

- **Qwen 2.5 表现不及 Sonnet**：虽然 **Qwen 2.5-Coder-32B** 表现尚可，但在复杂任务中目前落后于 **Claude 3.5 Sonnet**，其得分为 **73.7%**，而 Sonnet 为 **84.2%**。
  
  - 用户注意到，虽然 Qwen 模型正在改进，但它们仍会出现占位符回复，这会阻碍编码效率和完整性。
- **让 Aider 安装更简单**：讨论了 Aider 的安装程序，强调需要安装 **Python 3.9-3.12** 和 **git** 作为前提条件。
  
  - 用户被引导至官方 [Aider 安装指南](https://aider.chat/docs/install/install.html)以获取帮助。
- **评估向量化的必要性**：提出了通过对只读 Markdown 文件进行向量化和 **reranking** 来改进 Aider 上下文管理的潜力，特别是在处理大量项目文档时。

- 一位成员表示在将多个详细文件作为 context 适配时遇到挑战，并寻求在项目内进行更好搜索和管理的解决方案。
- **Aider 的功能请求**：用户表达了对 Aider 进行特定改进的愿望，例如消除懒惰回复（lazy responses）以及启用 CLI 命令执行。
  
  - 建议包括创建 conventions 文件以引导编码行为，并可能修改 system prompts 以获得更好的性能。
- **AI 模型性能比较**：各种模型之间的比较显示，**Qwen 2.5-Coder** 表现尚可，但在某些任务上仍落后于 **DeepSeek** 和 **GPT-4o** 等模型。
  
  - 来自 Aider 排行榜的分数表明，调整模型配置可以提高特定编码任务的性能结果。

**提到的链接**：

- [Installing aider](https://aider.chat/docs/install/install.html)：aider 是你终端里的 AI 配对编程工具
- [Specifying coding conventions](https://aider.chat/docs/usage/conventions.html)：告诉 aider 在处理代码时遵循你的编码规范。
- [Qwen2.5 Coder 32B Instruct – Run with an API](https://openrouter.ai/qwen/qwen-2.5-coder-32b-instruct/api)：Qwen2.5 Coder 32B Instruct 的示例代码和 API - Qwen2.5-Coder 是最新的代码专用 Qwen 大语言模型系列（前身为 CodeQwen）。Qwen2.5-Coder 带来了以下改进...
- [Qwen/Qwen2.5-Coder-32B-Instruct · Hugging Face](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct)：未找到描述
- [YaRN: Efficient Context Window Extension of Large Language Models - AI Resources](https://www.modular.com/ai-resources/yarn)：YaRN (Yet another RoPE extensioN method) 是一种使用旋转位置嵌入 (RoPE) 扩展大语言模型 context window 的计算高效方法。它通过显著的...
- [Aider LLM Leaderboards](https://aider.chat/docs/leaderboards/)：LLM 代码编辑能力的定量基准测试。
- [Qwen2.5 Coder 32B Instruct - API, Providers, Stats](https://openrouter.ai/qwen/qwen-2.5-coder-32b-instruct)：Qwen2.5-Coder 是最新的代码专用 Qwen 大语言模型系列（前身为 CodeQwen）。通过 API 运行 Qwen2.5 Coder 32B Instruct
- [Qwen2.5 Speed Benchmark - Qwen](https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html)：未找到描述
- [Models | OpenRouter](https://openrouter.ai/docs/models)：所有可用模型的表格
- [Provider Routing | OpenRouter](https://openrouter.ai/docs/provider-routing)：跨多个提供商路由请求
- [Claude 3.5 Sonnet - API, Providers, Stats](https://openrouter.ai/anthropic/claude-3.5-sonnet)：全新 Claude 3.5 Sonnet 以相同的 Sonnet 价格提供优于 Opus 的能力和快于 Sonnet 的速度。通过 API 运行 Claude 3.5 Sonnet
- [cline/src/core/prompts/system.ts at main · cline/cline](https://github.com/cline/cline/blob/main/src/core/prompts/system.ts)：直接在 IDE 中的自主编码 Agent，能够创建/编辑文件、执行命令、使用浏览器，并在每一步都获得你的许可。- cline/cline
- [YaRN: Efficient Context Window Extension of Large Language Models | Continuum Labs](https://training.continuumlabs.ai/training/the-fine-tuning-process/training-processes/yarn-efficient-context-window-extension-of-large-language-models)：Nous Research, EleutherAI, 日内瓦大学
- [GitHub - QwenLM/Qwen2.5-Coder: Qwen2.5-Coder is the code version of Qwen2.5, the large language model series developed by Qwen team, Alibaba Cloud.](https://github.com/QwenLM/Qwen2.5-Coder)：Qwen2.5-Coder 是 Qwen2.5 的代码版本，由阿里巴巴云 Qwen 团队开发的大语言模型系列。- QwenLM/Qwen2.5-Coder

---

### **aider (Paul Gauthier) ▷ #**[**questions-and-tips**](https://discord.com/channels/1131200896827654144/1133060505792159755/1305670535673413652) (49 条消息🔥):

> - `Aider 配置警告`
> - `OpenRouter API 使用`
> - `模型基准测试`
> - `Aider 中的 Ping 设置`
> - `Architect 模式功能`

- **Aider 配置警告引发混淆**：一位用户询问 Aider 未能识别配置参数的问题，另一位成员解释说，如果 Ollama 服务器未运行或未设置 API base，可能会出现警告而非具体的错误消息。
  
  - 针对模型识别问题，社区建议指定正确的模型名称，并指出 Litellm 持续存在的 Bug 可能会导致虚假警告。
- **处理 OpenRouter API 调用**：一位用户在基准测试脚本连接 llama-server 时遇到问题，显示模型名称未被 LLM 代理识别，他们在经过大量排查后发现了这一点。
  
  - 另一位成员建议 `.aider.model.metadata.json` 文件除了成本报告外，对 Aider 的功能几乎没有影响，用户可以在必要时忽略它。
- **对新的 Qwen32b Coder 模型进行基准测试**：一位用户准备对新的 **Qwen32b Coder 模型**运行基准测试，但在基准测试脚本链接到其 llama-server 时遇到困难。
  
  - 他们确定模型名称识别是主要问题，在做出调整后得以继续启动基准测试。
- **理解 Aider 的 Ping 设置**：关于 Aider 的 `ping` 设置引发了讨论，一位用户意识到保持应用程序开启可能会迅速消耗 OpenRouter 上的额度。
  
  - 澄清了配置高频率的 ping 会导致额度消耗，其他人建议减少次数以避免不必要的额度使用。
- **浏览器中的 Architect 模式支持**：一位用户询问 Architect 模式是否与 Aider 的浏览器版本兼容，得到了幽默的回应，强调了对 Aider 进行适当配置的偏好。
  
  - 社区认可了 Aider 遵循用户配置的行为，这引发了关于用户在设置中应承担责任的轻松表达。

**提到的链接**：

- [模型警告](https://aider.chat/docs/llms/warnings.html)：aider 是你终端里的 AI 结对编程助手
- [选项参考](https://aider.chat/docs/config/options.html#cache-settings)：关于 aider 所有设置的详细信息。
- [[Bug]: get_model_info() blows up for ollama models? · Issue #6703 · BerriAI/litellm](https://github.com/BerriAI/litellm/issues/6703)：发生了什么？使用 ollama 模型调用 litellm.get_model_info() 会抛出异常。但我可以正常使用这些模型运行 litellm.completion()。$ pip freeze | egrep 'litellm|ollama' litell...
- [aider thinks model is unknown and asks if I meant \*The exact same model\* · Issue #2318 · Aider-AI/aider](https://github.com/Aider-AI/aider/issues/2318)：ollama/vanilj/supernova-medius:q6_k_l 的警告：未知的上下文窗口大小和成本，正在使用合理的默认值。你是想说这些吗？- ollama/vanilj/supernova-medius:q6_k_l 你可以跳过这个 c...

---

### **aider (Paul Gauthier) ▷ #**[**links**](https://discord.com/channels/1131200896827654144/1268910919057149974/1305995718133284895) (2 条消息):

> - `Copilot Edits`
> - `Cursor 与 SupermavenAI 的合作伙伴关系`

- **GitHub Copilot Edits 彻底改变 VS Code**：[Copilot Edits](https://code.visualstudio.com/blogs/2024/11/12/introducing-copilot-edits) 功能将内联代码补全与对话能力相结合，允许在 VS Code 中跨多个文件进行快速编辑。
  
  - 该工具通过让开发人员指定文件并提供自然语言命令来进行代码修改，从而增强了工作流程。
- **SupermavenAI 加入 Cursor**：Cursor 在 Twitter 上宣布 **SupermavenAI** 现已成为团队的一部分，旨在将 **Cursor** 打造为研究和产品开发的领导者。
  
  - 此次合作旨在利用两家公司的专业知识来增强 Cursor 的产品，正如[这条推文](https://x.com/cursor_ai/status/1856427424927625679)中所强调的那样。

**提到的链接**：

- [来自 Cursor (@cursor_ai) 的推文](https://x.com/cursor_ai/status/1856427424927625679)：我们很高兴地宣布 @SupermavenAI 正在加入 Cursor！我们将共同继续将 Cursor 打造为研究和产品的强大力量。(1/5)
- [Copilot Edits 介绍](https://code.visualstudio.com/blogs/2024/11/12/introducing-copilot-edits)：Copilot Edits 允许你使用专为快速迭代设计的 UI，在你的工作区中跨多个文件进行所需的更改。你可以指定一组要编辑的文件，然后使用自然语言...

---

### **Interconnects (Nathan Lambert) ▷ #**[**news**](https://discord.com/channels/1179127597926469703/1179128538679488533/1305630017870364732) (62 条消息🔥🔥):

> - `Qwen 2.5 Coder`
> - `Dario Amodei 谈 AI Scaling`
> - `Nous Research Forge API`
> - `Anthropic 团队更新`
> - `OpenAI o1 发布`

- **Qwen 2.5 Coder 惊人的 Token 数量**：Qwen 2.5 Coder 在 Qwen 2.5 的基础上进行了持续预训练，目前已达到惊人的 **23.5 trillion tokens**，使其成为首个突破 **20 trillion** 大关的 Open weights 模型。
  
  - 用户注意到该模型需要庞大的计算资源，这体现了其开发的规模。
- **Dario Amodei 讨论 AI Scaling**：在最近的一次播客中，Dario Amodei 强调了在各种模态中观察到的 Scaling 现象，暗示到 2026-2027 年有可能实现 **human-level AI**，尽管仍存在一些不确定性。
  
  - 他还在 Scaling 讨论中强调了 AI 系统具备伦理和细微行为的重要性。
- **Nous Research 发布 Forge Reasoning API**：Nous Research 宣布推出 **Forge Reasoning API Beta**，旨在改进适用于任何模型的推理时间 Scaling，并承诺提升 **Hermes 70B** 模型的性能。
  
  - 目前对报告中的 Benchmarks 一致性存在担忧，导致人们对其性能指标的可靠性产生怀疑。
- **Anthropic 迎来新团队成员**：一项引人注目的团队更新显示，Hailey Schulz 在 Eleuther AI 工作两年后加入 **AnthropicAI**，这表明了行业内的人才流动。
  
  - 这引发了关于 AI 公司招聘动态的讨论，以及对其他组织建立更强大团队的期待。
- **对 OpenAI o1 发布的期待**：根据匿名消息来源的见解，围绕 OpenAI 计划在年底前正式发布 **o1 reasoning model** 的猜测正在升温。
  
  - 围绕此次发布的细节吸引了社区的关注，特别是关于参与开发的团队背景。

**提到的链接**：

- [来自 Stephanie Palazzolo (@steph_palazzolo) 的推文](https://x.com/steph_palazzolo/status/1856360400721162745)：与 @erinkwoo 共同发布的新消息：至少有一名 OpenAI 研究员已接受前 CTO Mira Murati 的邀请，加入她的新初创公司，她正与前 OAI 研究员 Barret Zoph 和 Luke Metz 共同筹备该公司。A...
- [来自 Andrew Carr (e/🤸) (@andrew_n_carr) 的推文](https://x.com/andrew_n_carr/status/1856054538769506800)：Qwen2.5-Coder-32B-Instruct 是继 O1-preview 之后排名第二的诗歌模型 🤯
- [来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文](https://x.com/lmarena_ai/status/1856444009323082093?s=61)：哪款模型最适合编程？@CopilotArena 排行榜发布了！我们的代码补全排行榜包含了过去一个月收集的数据，提供了超过 10 万次补全服务并获得了超过 1 万张选票！Le...
- [来自 Xeophon (@TheXeophon) 的推文](https://x.com/thexeophon/status/1856429292504096944?s=61)：@gm8xx8 这些是 Nous 在 3 版本发布时的数据。报告中 70B 模型的数据与图表也不匹配 —— MMLU-Pro (发布版) 为 47.24，而现在是 54.14。我是漏掉了什么显而易见的东西吗...
- [来自 Hailey Schoelkopf (@haileysch__) 的推文](https://x.com/haileysch__/status/1856172527921574154)：重大人生活动更新：我这周要加入 @AnthropicAI 了！期待与那里优秀的团队见面并共事！非常感谢过去两年里与同事和合作伙伴们度过的美好时光...
- [来自 Teortaxes▶️ (@teortaxesTex) 的推文](https://x.com/teortaxesTex/status/1856212163385307369)：为了确保准确，我进行了澄清。Qwen 2.5 Coder 是在 Qwen 2.5 的基础上进行持续预训练的。由此我推断，它已经处理了 23.5 万亿个 token（18T 的通用混合数据以及现在的 5.5T 代码:文本数据...
- [来自 Binyuan Hui (@huybery) 的推文](https://x.com/huybery/status/1856042011390063015)：💪 我已竭尽全力为您呈现最好的。引用 Qwen (@Alibaba_Qwen) 🚀 就是现在，11 月 11 日 10:24！发布我们有史以来最强编程模型的完美时刻！Qwen2.5-Coder-32B-Instruct！...
- [来自 deepfates (@deepfates) 的推文](https://x.com/deepfates/status/1795187390660715005)：老实说，他这波操作确实有点水平（cooked）。
- [Qwen2.5-Coder 技术报告](https://arxiv.org/abs/2409.12186)：在本报告中，我们介绍了 Qwen2.5-Coder 系列，这是对其前身 CodeQwen1.5 的重大升级。该系列包含六个模型：Qwen2.5-Coder-(0.5B/1.5B/3B/7B/14B/32B)。作为一个代码专用...
- [他承认了 Admit GIF - 他承认了 承认了 承认了 - 发现并分享 GIF](https://tenor.com/view/he-admit-it-admit-it-admit-omg-itysl-gif-18470746)：点击查看 GIF
- [来自 Nous Research (@NousResearch) 的推文](https://x.com/NousResearch/status/1856417883934601246)：今天我们面向社区中的特定群体推出了 Forge Reasoning API Beta 版，这是 inference time scaling 领域的一项进步，可应用于任何模型或模型组合。https...
- [来自 Aidan McLau (@aidan_mclau) 的推文](https://x.com/aidan_mclau/status/1856127488356712917)：我哭了
- [Dario Amodei：Anthropic CEO 谈 Claude、AGI 以及 AI 与人类的未来 | Lex Fridman Podcast #452](https://youtu.be/ugvHCXCOmm4)：Dario Amodei 是 Anthropic 的 CEO，该公司开发了 Claude。Amanda Askell 是一位致力于研究 Claude 性格和特征的 AI 研究员。Chris...

### **Interconnects (Nathan Lambert) ▷ #**[**ml-questions**](https://discord.com/channels/1179127597926469703/1179208129083363358/1305984889878876202) (7 messages):

> - `SPARC model`
> - `VLM techniques`
> - `Claude's OCR capabilities`
> - `Recent VLM articles`
> - `Finbarr blog`

- **SPARC 模型引入了新的 VLM 技术**：在一段 [视频](https://www.youtube.com/watch?v=rUQUv4u7jFs&t=3432s) 中，Jovana Mitrović 讨论了 **SPARC** 模型，该模型将文本表示与特定的图像补丁（image patches）对齐，而不是整个图像，这与 **CLIP** 等传统方法有所不同。
  
  - 这引发了一个问题：为什么更多的 **VLM** 没有采用类似的技术，同时也引发了关于 **SPARC** 的训练如何影响单个补丁表示的讨论。
- **VLM 实现了先进的 OCR 能力**：一位成员指出，他们对 **VLM** 在从图像中读取文本方面的强大能力感到惊讶，这使得 **Tesseract** 和 **ABBYY FineReader** 等传统工具显得过时了。
  
  - 他们将这一转变归功于 **VLM** 技术的进步，其灵感来自 **Claude** 在将截图转换为 **LaTeX** 方面的高效表现。
- **探索最近的 VLM 论文**：为了深化对 **VLM** 的理解，一位成员一直在研读文献，并提到由于 **Pixtral** 和 **DeepSeek Janus** 等新 **VLM** 的发布，他们的写作进度有所延迟。
  
  - 他们对近期产出所激发的 **VLM** 研究领域的不断演进表示了极大的热情。
- **推荐 Finbarr 博客**：一位成员敦促另一位成员阅读关于 **VLM** 的 **Finbarr blog**，强调了它在理解该领域方面的重要性和相关性。
  
  - 进一步的讨论促成了近期文章的分享，包括 **Sebastian Raschka** 的一份高层级概述，这进一步激发了大家的兴趣。

**提到的链接**：

- [Papers I've read this week: vision language models](https://www.artfintel.com/p/papers-ive-read-this-week-vision)：他们一直在发布 VLM，所以我也一直在写……
- [[EEML'24] Jovana Mitrović - Vision Language Models](https://www.youtube.com/watch?v=rUQUv4u7jFs&t=3432s))：未找到描述

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-drama**](https://discord.com/channels/1179127597926469703/1181746144821387334/) (1 messages):

an1lam: 有点相关但挺搞笑的，我觉得我昨晚在街上偶遇了 Gary。

---

### **Interconnects (Nathan Lambert) ▷ #**[**random**](https://discord.com/channels/1179127597926469703/1183121795247779910/1305961187065729024) (2 messages):

> - `ICLR Review Process`
> - `Reviewer Feedback`

- **ICLR 评审员火力全开**：一位成员评论说，来自 **ICLR** 评审员的反馈可能相当刻薄，并表示不确定是自己太宽容了，还是评审员根本没有理解提交内容的复杂性。
  
  - *ICLR 评审员以其严格的标准而闻名*，这引发了关于反馈中友善度与批判性评估之间平衡的讨论。
- **建议忽略那些唱反调的人**：另一位成员建议直接忽略他人的干扰，并表示：“别理那些瞎嚷嚷的人（yappers）”。
  
  - 这反映了学术界的一种普遍观点，即专注于建设性的批评，而不是陷入无谓的噪音中。

---

### **Interconnects (Nathan Lambert) ▷ #**[**nlp**](https://discord.com/channels/1179127597926469703/1208183200099344445/1305643241080619140) (5 messages):

> - `Neural Notes 剧集`
> - `Stanford MIPRO 优化器`
> - `Eugene Charniak 纪念研讨会`
> - `自动化 Prompt 优化`

- **Neural Notes 探讨语言模型优化**：最新一期的 [Neural Notes](https://www.youtube.com/watch?v=DVkM5dB3Oqs) 邀请了 Vertex Ventures 的投资者，与斯坦福大学 AI 实验室的博士生 **Krista Opsahl-Ong** 共同讨论语言模型优化的进展。
  
  - 虽然该视频尚未被观看，但因其对**自动化 Prompt 优化**的深刻见解而获得了社区的积极反馈。
- **斯坦福研究人员为 MIPRO 优化器做出贡献**：提到对参与 DSPy 中使用的 **MIPRO 优化器** 开发的斯坦福研究人员的采访，突显了人们对 Prompt 优化自动化的兴趣日益浓厚。
  
  - 发言者表示渴望进一步了解 **DSPy**，以形成对该主题的全面理解。
- **Eugene Charniak 纪念研讨会展示 NLP 人才**：[布朗大学](https://cs.brown.edu/events/eugene-charniak-memorial-symposium/)最近举行了一场纪念 **Eugene Charniak** 的研讨会，许多有影响力的 NLP 研究人员出席了会议。
  
  - 该活动因其对 NLP 和语言学的双重关注而备受瞩目，促进了对该领域相关进展的讨论。

 

**提到的链接**：[Neural Notes: 语言模型优化的未来](https://www.youtube.com/watch?v=DVkM5dB3Oqs)：在本期 Neural Notes 中，Vertex Ventures US 的投资者 Sandeep Bhadra 和 Simon Tiu 与斯坦福大学 AI 实验室 (SAIL) 的博士生 Krista Opsahl-Ong 进行了对话...

 

---

### **Interconnects (Nathan Lambert) ▷ #**[**reads**](https://discord.com/channels/1179127597926469703/1214764639397617695/1305630928516939797) (62 条消息🔥🔥):

> - `Scaling laws 与模型 Quantization`
> - `Dylan Patel 的 Inference 见解`
> - `AI 本地运行的挑战`
> - `LLaMA 模型的性能预期`
> - `对 Datacenter 基础设施的影响`

- **Scaling laws 揭示了 Quantization 的极限**：一项新研究表明，随着模型在更多 Token 上进行训练，它们在 Quantization 时需要更高的精度，这可能对 GPU 和 AI 模型的未来产生重大影响。
  
  - 该研究指出，进一步增加预训练数据可能会对 Quantization 过程产生负面影响，特别是对于像 LLaMA-3 这样的模型。
- **Dylan Patel 讨论 AI Megaclusters**：Dylan Patel 最近在 Stanford 的讲座涵盖了 Inference 数学以及 AI Megaclusters 日益增长的重要性，强调了 AI 基础设施的最新进展。
  
  - 一位成员表示有兴趣参加 Stanford 的课程，同时关注 Patel 提到的 Datacenter 现状的发展。
- **在本地运行 Qwen 2.5 的挑战**：由于 Qwen 2.5 Coder 是在超过 20 万亿个 Token 上训练的，人们对其在本地机器上运行的可行性表示担忧，这暗示了极高的硬件要求。
  
  - 有人指出，使用全 BF16 精度将需要高配置机器（如 128GB 的 MacBook），这让设备性能较低的用户感到担忧。
- **对 AI 模型性能的看法**：参与者讨论了包括 LLaMA-3 在内的许多模型在 Quantization 方面面临的困难（相比之前的 LLaMA-2 模型），这表明性能预期可能进入平台期。
  
  - 用户反思了增加 Token 训练的更广泛影响，以及 Quantization 如何影响模型能力。
- **对 Datacenter 基础设施的影响**：对最新 128GB Mac 等更强大硬件日益增长的需求，说明了 AI 应用不断演进的需求以及相关的 Datacenter 挑战。
  
  - 对话暗示，由于实现最佳性能所需的硬件要求增加，在 Scaling 和运行大型模型方面可能存在潜在瓶颈。

**提到的链接**：

- [Nathan Lambert (@natolambert) 的推文](https://x.com/natolambert/status/1856077454210723856)：关于 "Scaling 是否已到头" 有很多讨论，The Information 的报道称最新的 GPT 模型没有达到 OpenAI 的预期，而 Sam Altman 仍在宣扬……
- [Tim Dettmers (@Tim_Dettmers) 的推文](https://x.com/tim_dettmers/status/1856338240099221674?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ)：这是很长一段时间以来最重要的一篇论文。它有力地证明了我们正在触及 Quantization 的极限。论文指出：训练的 Token 越多，需要的精度就越高……
- [Dylan Patel - Inference 数学、模拟与 AI Megaclusters - Stanford CS 229S - 2024 秋季](https://youtu.be/hobvps-H38o?si=FR7re3r6gds6b-UN)：网站：https://scalingintelligence.stanford.edu/ GitHub：https://github.com/ScalingIntelligence HuggingFace：https://huggingface.co/ScalingIntelligence

---

### **OpenRouter (Alex Atallah) ▷ #**[**announcements**](https://discord.com/channels/1091220969173028894/1092729520181739581/1305724352804687912) (8 条消息🔥):

> - `Qwen2.5 Coder 32B`
> - `Gemini models updates` (Gemini 模型更新)
> - `Scheduled Downtime` (计划停机)

- **Qwen2.5 Coder 32B 超越竞争对手**：全新的高性能开源模型 **Qwen2.5 Coder 32B** 已发布。根据 [OpenRouter 的推文](https://x.com/OpenRouterAI/status/1856165171690926446)指出，它在多项编程基准测试中击败了 **Sonnet** 和 **GPT-4o**。
  
  - 然而，一些成员对这些说法的准确性表示担忧，认为像 **MBPP** 和 **McEval** 这样的测试可能具有误导性。
- **Gemini 模型获得新功能**：**Gemini 1.5 Flash, Pro** 和 **8B** 模型现在支持 **frequency penalty**、**presence penalty** 和 **seed** 调整，详情见 OpenRouter 的官方更新。
  
  - 更多信息链接包括 [Gemini Flash 1.5 8B](https://openrouter.ai/google/gemini-flash-1.5-8b) 和 [Gemini Pro 1.5](https://openrouter.ai/google/gemini-pro-1.5)。
- **计划停机通知**：一项通知宣布在 **EST 时间上午 9:30** 进行 **5 分钟的计划停机**，并表示服务将在不久后恢复上线。
  
  - 升级在不到一分钟内成功完成；感谢用户在停机期间的耐心配合。

**提到的链接**：

- [Tweet from undefined](https://x.com/Ope): 未找到描述
- [Tweet from OpenRouter (@OpenRouterAI)](https://x.com/OpenRouterAI/status/1856165171690926446): 来自 @Alibaba_Qwen 的全新高性能开源模型：Qwen2.5 Coder 32B！在多项编程基准测试中击败了 Sonnet 和 GPT-4o。来自 @hyperbolic_labs 和 @FireworksAI_HQ 的极佳定价...
- [Gemini Flash 1.5 - API, Providers, Stats](https://openrouter.ai/google/gemini-flash-1.5-8b>): Gemini 1.5 Flash 是一个基础模型，在各种多模态任务中表现出色，如视觉理解、分类、摘要以及从图像、音频和视频中创建内容...
- [Gemini Flash 1.5 - API, Providers, Stats](https://openrouter.ai/google/gemini-flash-1.5>): Gemini 1.5 Flash 是一个基础模型，在各种多模态任务中表现出色，如视觉理解、分类、摘要以及从图像、音频和视频中创建内容...
- [Gemini Pro 1.0 - API, Providers, Stats](https://openrouter.ai/google/gemini-pro-1.5>): Google 的旗舰文本生成模型。旨在处理自然语言任务、多轮文本和代码聊天以及代码生成。通过 API 运行 Gemini Pro 1.0。

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1305652623277424702) (107 条消息🔥🔥):

> - `Gemini 1.5 Flash 更新`
> - `Qwen 2.5 Coder 性能`
> - `Anthropic 的 computer use 工具`
> - `模型知识局限性`
> - `OpenRouter 定价与功能`

- **Gemini 1.5 Flash 表现出改进**：用户注意到 **Gemini 1.5 Flash** 的性能似乎有所提升，特别是在 **temperature 0** 下使用时。
  
  - 一位成员推测，这可能是 **Google AI Studio** 上正在使用的实验版本。
- **对 Qwen 2.5 Coder 的期待**：成员们对试用 **Qwen 2.5 32B Coder** 表示兴奋，并指出最近的价格变得更加亲民，大约为每百万 token 一美元。
  
  - 一位用户表示，由于之前成本较高，他们曾不得不转而使用 **DeepSeek**。
- **Anthropic 的 computer use 工具兼容性**：讨论围绕 **Anthropic** 新推出的 **computer use** 工具是否能在 OpenRouter 上运行展开，确认目前尚不支持。
  
  - 有人提到该工具需要一个特殊的 beta header，而 OpenRouter 目前还不支持。
- **模型缺乏特定内容的知识**：有用户对 **Hunyuan**、**Qwen** 和 **Yi** 等模型表示担忧，据报道这些模型缺乏关于西方媒体和版权问题的关键知识。
  
  - 用户注意到性能上的差异，某些模型在处理版权内容方面比其他模型表现得更好。
- **OpenRouter 定价结构**：根据服务条款，通过积分使用 OpenRouter 的 token 可能会产生约 **5% 的额外费用**。
  
  - 这引发了用户对定价透明度以及与直接使用模型相比的成本差异的疑问。

**提到的链接**：

- [未找到标题](https://api.together.xyz/signin?redirectUrl=/playground/chat/Qwen/Qwen2.5-72B-Instruct-Turbo)：未找到描述
- [Models Overview | Mistral AI Large Language Models](https://docs.mistral.ai/getting-started/models/models_overview/)：Mistral 提供两类模型：免费模型和高级模型。
- [Qwen2.5 Coder 32B Instruct - API, Providers, Stats](https://openrouter.ai/qwen/qwen-2.5-coder-32b-instruct)：Qwen2.5-Coder 是最新的代码专用 Qwen 大语言模型系列（前身为 CodeQwen）。通过 API 运行 Qwen2.5 Coder 32B Instruct。
- [Magnum v4 72B - API, Providers, Stats](https://openrouter.ai/anthracite-org/magnum-v4-72b)：这是一个旨在复制 Claude 3 模型（特别是 Sonnet）散文质量的模型系列。通过 API 运行 Magnum v4 72B。
- [OpenRouter](https://openrouter.ai/terms)：LLM 路由与市场

---

### **OpenRouter (Alex Atallah) ▷ #**[**beta-feedback**](https://discord.com/channels/1091220969173028894/1277894087755829278/1305875675315437619) (6 条消息):

> - `自定义提供商密钥访问权限`

- **对自定义提供商密钥访问权限的普遍请求**：多位用户通过申请 beta 测试的 **custom provider keys** 访问权限表达了兴趣。
  
  - *一位成员指出*，该访问权限将帮助他们解决 **Google** 的 **rate limit** 问题。
- **对自定义提供商密钥的浓厚兴趣**：共有五位用户请求了访问权限，展示了社区对 **custom provider keys** 的强烈兴趣。
  
  - 几位用户提到，他们希望利用这些密钥在项目中实现更好的功能。

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1305625839303655435) (50 条消息🔥):

> - `TTS 替代方案`
> - `使用 AI 进行应用开发`
> - `Qwen2.5-Coder 模型`
> - `KitchenAI 项目`
> - `语言模型的怪癖`

- **探索 TTS 替代方案**：成员们讨论了各种文本转语音 (TTS) 解决方案，其中一位推荐 [f5-tts](https://drinkoblog.weebly.com/) 作为一个可以在消费级 GPU 上运行的可行选项。
  
  - 虽然有人询问关于时间戳数据的功能，但建议主要集中在低成本解决方案上。
- **AI 交互应用开发**：一位用户详细描述了他们在编写语音交互 AI 应用时面临的挑战，遇到了 `speech_recognition` 模块和服务器识别方面的问题。
  
  - 有人建议在移动设备上使用 ChatGPT 的浏览器版本，以便更轻松地选择文本。
- **Qwen2.5-Coder 模型介绍**：新的 [Qwen2.5-Coder](https://qwenlm.github.io/blog/qwen2.5-coder-family/#qwen25-coder--artifacts) 模型因其开源代码能力而受到关注，展示了其相较于 GPT-4o 等模型的竞争优势。
  
  - 分享了 GitHub、Hugging Face 和 ModelScope 的链接，以便访问该模型及其 Demo。
- **KitchenAI 开源项目**：一位成员分享了名为 [KitchenAI](https://github.com/epuerta9/kitchenai) 的开源项目，并正在寻找开发者参与贡献。
  
  - 他们提到正在 Discord 和 Reddit 上努力寻找感兴趣的开发者。
- **AI 意外切换语言**：一位用户注意到 AI 意外切换到韩语的奇特案例，引发了关于这一现象的讨论。
  
  - 其他人也分享了类似的经历，认为 AI 可能会默认切换到它在特定语境下认为最自然的语言。

**提到的链接**：

- [Qwen2.5-Coder Series: Powerful, Diverse, Practical.](https://qwenlm.github.io/blog/qwen2.5-coder-family/#qwen25-coder--artifacts)：GITHUB HUGGING FACE MODELSCOPE KAGGLE DEMO DISCORD 介绍。今天，我们很高兴开源“强大”、“多样”且“实用”的 Qwen2.5-Coder 系列...
- [GitHub - epuerta9/kitchenai: Shareable runtime AI cookbooks](https://github.com/epuerta9/kitchenai)：可共享的运行时 AI 指南。通过创建账号参与 epuerta9/kitchenai 的开发。

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1305771851959504916) (10 条消息🔥):

> - `ChatGPT 访问问题`
> - `关于屏蔽机构的报告`
> - `对 DALL-E 图像生成的挫败感`
> - `aiHa GPT 中文档消失的问题`

- **Chrome 上的 ChatGPT 访问问题**：一位用户报告称，他们在 Chrome 上被锁定无法访问 ChatGPT，但在 Microsoft Edge 上运行正常，这引发了对潜在篡改或安全措施的担忧。
  
  - 该用户表示：*“我从未违反过社区规则”*，暗示他们的输入是合乎道德且相关的。
- **使用 PowerShell 脚本屏蔽情报机构**：同一位用户提到生成了 PowerShell 脚本来屏蔽各种情报机构的 IP 范围，并澄清其中没有一个是美国的。
  
  - 针对此类屏蔽措施的有效性以及机构是否会使用 VPN 表达了疑虑。
- **对 DALL-E 局限性的挫败感**：一位成员发泄了对 DALL-E 图像生成能力的挫败感，称在多次尝试后经常得到错误结果，并达到了对话限制。
  
  - 他们质疑自己是否在付费与一个“愚蠢的对话伙伴”交流，并觉得由于效果不佳，这些限制更像是一种骗局。
- **关于局限性的用户错误评论**：另一位成员的回应暗示，所表达的挫败感可能是用户操作错误，暗示个人仍有改进空间。
  
  - 这一评论突显了关于 AI 应用中用户体验与技术局限性之间持续存在的争议。
- **aiHa GPT 中消失的文档**：一位用户分享了他们的 PDF 和 Docx 文档尽管从不同浏览器多次上传，但仍从 aiHa GPT 知识库中消失的困扰。
  
  - 他们指出所有文件都小于 1MB，表明这是一个意外的技术问题，并进一步询问是否有人遇到同样的问题。

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/1305730576426205195) (21 messages🔥):

> - `GPT-4o mini 的 Prompt 设计`
> - `Structured outputs 技术`
> - `Prompt engineering 资源`
> - `转录文本中的参与度和相关性`

- **澄清 30 到 60 秒的 Prompt**：针对 Prompt 的清晰度展开了讨论，争论焦点在于片段长度是应严格限制在 30 到 60 秒，还是允许拼接多个片段。
  
  - *有人建议从头开始重构 Prompt，这有助于理清意图。*
- **JSON 输出异常**：Mateusneresrb 表达了对模型在尝试输出选定视频片段的指定时间时返回错误时间间隔的沮丧。
  
  - *还提到了关于 JSON 格式对输出正确性影响的担忧。*
- **高效 Prompt 编写技巧**：建议包括通过调整长度规范来使用清晰且简化的 Prompt，并可能求助于 Token 计数以获得更好的结果。
  
  - *分享了一个资源链接，以帮助提高 Prompt engineering 技能。*
- **在 Structured outputs 中使用 Scratchpads**：Jscheel 引入了在 Structured outputs 中结合使用 Scratchpad 技术的概念，以增强推理结果。
  
  - *有人寻求澄清，即 Scratchpad 是否应作为 Structured outputs 中的主要字段进行集成。*
- **对 AI 生成内容的期望**：一位用户对 AI 生成有趣内容的能力表示怀疑，但承认在通过结构化 Prompt 进行协作故事创作方面具有潜力。
  
  - *强调了理解个人需求对于有效发挥 AI 潜力的重要性。*

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/1305730576426205195) (21 messages🔥):

> - `GPT 模型的 Prompt engineering`
> - `片段选择技术`
> - `Structured output 用法`
> - `Scratchpad 技术`

- **Prompt 清晰度的挑战**：成员们讨论了关于从转录文本中选择片段的 Prompt 所引起的困惑，特别是应该专注于 30-60 秒的片段还是允许任何长度。
  
  - 一位成员建议从头重写 Prompt 以获得更好的清晰度，而另一位成员指出 AI 可能会因为指令不明确而表现不佳。
- **高效 Prompting 的建议**：推荐了一些关于 Prompt 最佳实践的网站，包括指向 [OpenAI's prompt engineering guide](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api) 的链接。
  
  - 成员们还交流了关于使用 Token 计数而非秒数来选择相关内容片段的想法。
- **关于与 AI 协作的评论**：一位成员对 AI 创作有趣内容的能力表示怀疑，但承认了人类与 AI 利用基于场景的 Prompting 进行协作的潜力。
  
  - 他们指出，共同检查和修改场景输出可以实现引人入胜的故事创作。
- **Scratchpad 技术讨论**：一位成员介绍了用于改进推理结果的 'Scratchpad' 技术概念，并表示有兴趣将其与 Structured outputs 集成。
  
  - 他们征求了关于是否将 Scratchpad 作为其 Structured output 格式中第一个字段的建议。

---

### **Modular (Mojo 🔥) ▷ #**[**general**](https://discord.com/channels/1087530497313357884/1098713601386233997/1305630103044096110) (41 条消息🔥):

> - `WSL2 中的 Nvidia CUDA`
> - `高级互操作性`
> - `WASI 与边缘计算`
> - `应用插件与性能`
> - `CRABI ABI 提案`

- **WSL2 中 Nvidia CUDA 驱动的挑战**：讨论强调了 Windows 上的 CUDA 驱动在 WSL2 中是以 `libcuda.so` 存根（stubbed）形式存在的，这暗示了通过 Mojo 访问完整驱动功能的潜在限制。
  
  - 成员们指出，如果 MAX 依赖于宿主机的 Windows 驱动，这可能会使 WSL 内部对 MAX 的支持变得复杂。
- **探索用于高级接口的 CRABI**：CRABI 提案旨在为 Rust、C++、Mojo 和 Zig 等高级语言创建一种 ABI，以实现超越 C 语言能力的互操作性。
  
  - 参与者讨论了与 Lua 和 Java 等语言集成的挑战，暗示需要更广泛的采用。
- **WASI 在边缘计算中的作用**：WASI 被认为对边缘计算有益，它提供了一种在沙箱环境中部署微服务的更简单方法。
  
  - 有人对 WASI 与传统方法相比产生的开销表示担忧，特别是对于性能敏感型应用。
- **应用插件的性能考量**：小组一致认为，鉴于敏感的性能环节通常存在于应用程序本身，应用插件是高级互操作性的合适案例。
  
  - 讨论承认，虽然某些解决方案可能会引入开销，但像 Mojo 这样的语言在插件市场中仍可能占据优势地位。
- **电信行业的软件驱动网络**：对话指出，电信部署通常依赖软件进行组网，主要使用 C 或 C++。
  
  - 这表明电信领域的高性能关键应用仍持续依赖传统的系统级语言。

**提到的链接**：

- [未找到标题](https://www.tensorflow.org/install/pip#windows-wsl2_1): 未找到描述
- [joshtriplett 提出的实验性特性门控提案 `crabi` · Pull Request #105586 · rust-lang/rust](https://github.com/rust-lang/rust/pull/105586): 摘要：该实验性特性门控提案建议开发一种新的 ABI（extern "crabi"）和一种新的内存表示（repr(crabi)），用于跨高级语言的互操作性……

---

### **Modular (Mojo 🔥) ▷ #**[**mojo**](https://discord.com/channels/1087530497313357884/1151418092052815884/1305630563029221486) (58 条消息🔥🔥):

> - `Mojo 安装问题`
> - `Mojo Subreddits`
> - `Benchmark 模块功能`
> - `动态模块导入`
> - `标准库贡献`

- **Mojo 安装困扰已解决**：一位用户在通过 `curl` 命令安装 Mojo 时遇到问题，但通过使用另一位成员提供的正确 URL 链接成功解决了问题。
  
  - 这突显了在安装软件包时确保 URL 条目正确的重要性。
- **关于非官方 Mojo Subreddits 的讨论**：虽然目前没有官方的 Mojo Subreddit，但用户讨论了两个现有的相关 Subreddit，并强调这两个均未获得 Modular 的官方认可。
  
  - 一位成员甚至建议联系版主置顶链接，以便在移动端获得更好的可见性。
- **理解 Benchmark 模块**：Mojo 中的 Benchmark 模块旨在帮助编写快速的基准测试，管理设置（setup）和拆卸（teardown），并处理吞吐量测量的单位。
  
  - 目前存在一些限制，例如在热循环（hot loops）中存在不必要的系统调用，这可能会影响性能。
- **动态模块导入的挑战**：由于 Mojo 的编译结构将所有内容打包为常量和函数，目前 Mojo 中不存在动态导入模块的功能。
  
  - 交付 JIT 编译器可能是解决方案之一，尽管这会带来关于二进制文件大小以及与预编译代码兼容性的担忧。
- **贡献 Mojo stdlib 的机会**：成员们讨论了 Mojo 标准库中基础工作的需求，特别是在实现 B-trees 等数据结构方面。
  
  - 这为那些具备数据结构和算法知识的人提供了向社区贡献的学习机会。

**提到的链接**：

- [B-tree - 维基百科](https://en.wikipedia.org/wiki/B-tree): 未找到描述
- [Reddit - 探索一切](https://www.reddit.com/r/modular_mojo/): 未找到描述
- [Reddit - 探索一切](https://www.reddit.com/r/MojoLang/): 未找到描述

---

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1305670046227370137) (77 条消息🔥🔥):

> - `Hailo 模型量化`
> - `ASM2464PD 芯片规格`
> - `USB4 PCIe 转换器开发`
> - `音频录制格式`
> - `Tinygrad 分布式系统`

- **Hailo 模型量化挑战**：一位成员概述了 **Hailo** 需要量化模型才能运行的困难，特别是需要 8-bit 量化，这增加了训练过程的复杂性。
  
  - 他们指出，为了在 Hailo 上正常运行，需要为 CUDA 和 TensorFlow 编译 **.so** 文件，这使得设置过程非常繁琐。
- **ASM2464PD 芯片见解**：讨论涉及了 **ASM2464PD 芯片**，确认它可以利用通用 PCIe，目前可通过多家供应商获得，不仅限于 NVMe。
  
  - 有人对功耗要求表示担忧，称通常需要 **70W** 才能保证正常功能。
- **USB4 转 PCIe 转换器进展**：分享了一个开源的 **USB4/Thunderbolt 转 M.2 PCIe** 转换器设计，展示了在 **GitHub** 上的显著进展，并成功获得了硬件开发的资金。
  
  - 创作者详细说明了对下一轮开发的期望，希望尽管面临挑战，仍能实现有效的 USB4 到 PCIe 集成。
- **优化音频录制格式**：成员们讨论了在音频录制中使用 **Opus** 编解码器，认为与传统格式相比，它在不损失质量的情况下能显著减小文件体积。
  
  - 同时也强调了对 Opus 浏览器兼容性的担忧，指出了其局限性以及需要额外技术支持的需求。
- **Tinygrad 分布式库的愿景**：一位用户表示有兴趣为 **Tinygrad** 开发一个库，专注于 dataloaders 和 optimizers 等分布式组件，而不依赖于 MPI 或 NCCL 等现有框架。
  
  - 他们的目标是从零开始构建基础网络功能，同时保持 Tinygrad 提供的接口，以实现无缝体验。

**提到的链接**：

- [John Simons (@johnsel92) 的推文](https://x.com/johnsel92/status/1785759687498998175)：现在你可以通过 @enjoy_digital 的 LiteX 和一块 500 美元的 Alinx Artix Ultrascale+ 开发板，在 PCIe gen4x4 上实现极速的 FPGA 到 PC 通信。
- [John Simons (@johnsel92) 的推文](https://x.com/johnsel92/status/1777111921658823136)：已完成并向 @JLCPCB 发送了我基于 @ASmedia ASM2464PD 的开源 USB4/Thunderbolt 转 M.2 PCIe 转换器设计。下一步：验证并将 PCB 缩小到邮票大小的模块 https://github.com...
- [tinycorp_meetings/2024-11-11 at master · geohotstan/tinycorp_meetings](https://github.com/geohotstan/tinycorp_meetings/tree/master/2024-11-11)：通过在 GitHub 上创建账号，为 geohotstan/tinycorp_meetings 的开发做出贡献。
- [用于非 NVMe PCIe？ · Issue #1 · cyrozap/usb-to-pcie-re](https://github.com/cyrozap/usb-to-pcie-re/issues/1)：我有兴趣将 GPU 连接到 USB 端口。我们正在为 AMD 开发用户态驱动程序，所以那一侧已经处理好了。tinygrad/tinygrad#6923 我们需要的是映射 PCIe 的能力...
- [Hailo-8™ M.2 2280 B+M key](https://up-shop.org/default/hailo-m2-key.html)：想要为您的 UP Squared Pro/UP Squared V2/UP Squared 6000/U... 添加高效的 AI 性能吗？
- [Free Transfert](https://transfert.free.fr/GytjFFc)：文件发送和共享服务，简单、免费且安全，适用于个人和企业。
- [Opus 推荐设置 - XiphWiki](https://wiki.xiph.org/Opus_Recommended_Settings#Bandwidth_Transition_Thresholds)：未找到描述
- [ThunderboltEX 4｜主板｜ASUS Nederland](https://www.asus.com/nl/motherboards-components/motherboards/accessories/thunderboltex-4/)：华硕提供多种主板配件，包括 Thunderbolt™ 扩展卡、M.2 附加卡和风扇扩展卡，为 DIY 玩家提供更好的选择...
- [未找到标题](https://www.aliexpress.com/item/1005006115962238.html)：未找到描述

---

### **tinygrad (George Hotz) ▷ #**[**learn-tinygrad**](https://discord.com/channels/1068976834382925865/1070745817025106080/1305829101252378664) (3 条消息):

> - `不进行 sharding 的 Parallelization`
> - `GPU 上的 Model serialization`
> - `Pattern matcher 辅助`

- **不进行 sharding 的 Parallelization 受到质疑**：一位用户询问在运行多个 **LLaMa models** 时，是否可以在不进行 **sharding** 的情况下实现 **parallelization**。
  
  - 他们还询问在 tensor 上调用 `.realize()` 是否会阻塞，直到 **GPU** 完成计算。
- **关于在单个 GPU 上进行 Model serialization 的讨论**：另一位成员澄清了讨论内容，可能指的是 **serialization**，即同一个模型的迭代可以在单个 GPU 上运行。
  
  - 他们指出这可以很容易地在训练循环中实现，并提到 **tinygrad** 已经支持 model mirroring。
- **分享了 Pattern matcher 的解释文档**：一位用户为那些在 **pattern matcher** 上遇到困难的人提供了一个有用的[解释链接](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20241112_pm.md)。
  
  - 该资源是 **tinygrad-notes** 的一部分，旨在帮助社区理解 **tinygrad** 的功能。

**提到的链接**：[tinygrad-notes/20241112_pm.md at main · mesozoic-egg/tinygrad-notes](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20241112_pm.md)：tinygrad 教程。通过在 GitHub 上创建账号为 mesozoic-egg/tinygrad-notes 的开发做出贡献。

---

### **Notebook LM Discord ▷ #**[**use-cases**](https://discord.com/channels/1124402182171672732/1124403655819415592/1305642978374582272) (34 条消息🔥):

> - `使用 NotebookLM 进行 summarization`
> - `尝试 podcasts 和 avatars`
> - `教科书上传问题`
> - `用于事实核查的 KATT`
> - `NotebookLM 在 AI 讨论中的潜力`

- **NotebookLM 可以总结 AI 新闻简报**：一位成员建议使用 NotebookLM 来总结超过 **200 封 AI 新闻简报邮件**，以避免手动复制粘贴内容。
  
  - 提到 Gmail 中的 *Gemini* 按钮可能是总结的潜在辅助工具，但指出其**并非免费**。
- **使用 avatars 进行创新播客**：一位成员分享了他们对 **Google Terms of Service** 的实验，通过一个 13 分钟的 podcast，使用了 avatars 并插入了广告间歇。
  
  - 这种方法旨在完善 podcast 格式，并为 **avatar creation** 更好地管理内容。
- **对教科书上传的担忧**：另一位成员上传了一本教科书作为来源，但反馈 NotebookLM 的**回答质量较差**，导致对其有效性产生怀疑。
  
  - 尽管分类功能有所改进，但仍有投诉称 NotebookLM 在处理 **Michelson-Morley** 等概念时感到吃力。
- **KATT 将旧流程与 AI 融合**：一位用户讨论了将 KATT (Knowledge-based Autonomous Trained Transformer) 整合到他们 podcast 的**事实核查器**中，从而制作出更长的节目。
  
  - 他们将这种集成描述为**痛苦的**，因为它结合了传统方法与新的 AI 技术。
- **致力于提高播客清晰度**：一位成员提到核苷酸序列 podcast，强调了使用**带注释的转录文本**以获得更好清晰度的重要性。
  
  - 他们的目标是测试 RAG 查询中 **reference sequences** 的极限，以增强可用性。

**提到的链接**：

- [无标题](https://notebooklm.google.com/notebook/19d92404-a2a6-4238-b9ad-33854c841aac/audio)：未找到描述
- [Steam Gift Activation](https://is.gd/OoGlr1)：未找到描述
- [UNREAL MYSTERIES 5: Behind the Scenes / Making Of](https://www.youtube.com/watch?v=rVOsQXoKcos)：有没有想过 Unreal Mysteries 节目是如何制作的？我们进入全元模式，制作了一个关于节目如何制作的剧内节目。见证 NotebookLM 的表现...
- [Understanding the Google Terms of Service](https://youtu.be/qqCkF-vWa9s)：演示 AI 和 Avatars 如何为乏味的内容增值。
- [10 INSANELY Helpful Ways To Use NotebookLM](https://www.youtube.com/watch?v=TheAnbKkD8s)：欢迎深入了解 Notebook LM，我将分享 10 种改变游戏规则的方法，你可以使用这个工具让生活更简单、更快速、更高效！...

---

### **Notebook LM Discord ▷ #**[**general**](https://discord.com/channels/1124402182171672732/1124402182909857966/1305629187310223401) (40 messages🔥):

> - `将 Notebook 导出为 PDF`
> - `NotebookLM 的非官方 API`
> - `文档上传限制`
> - `Notebook 集中化工作流`
> - `音频文件上传问题`

- **PDF 导出功能需求**：用户正在询问未来将笔记或笔记本导出为 **.pdf** 的计划，并寻求用于 Notebook 自动化的 API。
  
  - 虽然有人提到使用 **PDF 合并工具**等替代方案，但他们更渴望原生导出功能。
- **对非官方 API 的怀疑**：围绕一个每月 **30 美元**的 NotebookLM 非官方 **API** 展开了讨论，许多人对其合法性表示怀疑。
  
  - 讨论中提出了关于**缺乏商业信息**和示例输出的担忧，导致一些人将其贴上诈骗标签。
- **处理文档限制的技巧**：成员们分享了管理 NotebookLM 中文档限制的技巧，强调通过合并较短的文档来有效绕过 **50 个文档的限制**。
  
  - 一些用户注意到字数限制存在差异，质疑较长的文档在上传过程中是否被忽略。
- **集中管理 Notebook 和高亮**：一位用户正在寻找从各种 Notebook 中**集中管理笔记**的工作流，并得到了关于导出和合并笔记的建议。
  
  - 另一位用户强调在他们的项目中，NotebookLM 的回复需要高亮和评论功能。
- **音频文件上传问题**：用户对无法上传 **.mp3 文件**表示沮丧，并得到了关于通过 Google Drive 进行正确上传步骤的指导。
  
  - 一些人指出其他文件类型上传没有问题，这表明可能存在**技术故障**或转换错误。

**提到的链接**：

- [NotebookLM: La Navaja Suiza de la GenAI](https://randradedev.hashnode.dev/notebooklm-la-navaja-suiza-de-la-genai): 简介：GenAI 的兴起与对新工具的需求。2022 年作为生成式人工智能 (GenAI) 闯入大众视野的一年被载入史册...
- [NotebookLM API - AI-Powered Podcast Generation](https://notebooklmapi.com/): 使用 NotebookLM API 轻松创建专业播客。我们先进的 AI 技术简化了播客制作流程。
- [Steam Gift Activation](https://is.gd/OoGlr1): 未找到描述
- [Top Shelf](https://open.spotify.com/show/0MvNgBDb2NsZJN4cREl7yF): 播客 · Four By One Technologies · "Top Shelf" 是您获取当今畅销书快速、深刻见解的首选播客。只需 15 分钟，即可获得精华、金句和全新的视角...
- [(15min) Think and Grow Rich by Napoleon Hill - A Fresh Perspective](https://open.spotify.com/episode/5GRJnbQ3yxJHD6XOVN0KLH?si=SxNqPCeXQRaCCTUNQClJWw) : Top Shelf · 分集

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1305627004661403781) (51 messages🔥):

> - `Dario Amodei 访谈`
> - `Magentic-One 框架`
> - `Context Autopilot`
> - `Writer C 轮融资`
> - `Supermaven 加入 Cursor`

- **Lex Fridman 对 Dario Amodei 的访谈**：Anthropic 的 CEO Dario Amodei 在与 Lex Fridman 的 5 小时[访谈](https://www.youtube.com/watch?v=ugvHCXCOmm4)中讨论了 **Claude** 和 AI 的未来。
  
  - 聊天中强调了对 **AGI** 的见解和 AI 的新发展，引起了广泛关注。
- **Magentic-One 框架发布**：**Magentic-One** 框架正式推出，展示了一个旨在处理复杂任务并在效率上超越传统模型的多 **Agent** 系统。
  
  - 它使用一个编排器（orchestrator）来指导专业 **Agent**，并在各种基准测试中表现出**竞争力** [来源](https://x.com/rowancheung/status/1854972388988908023)。
- **Context Autopilot 介绍**：Context.inc 推出了 **Context Autopilot**，这是一款像用户一样学习的 AI，展示了在信息工作领域最先进的能力。
  
  - 分享了一个实际演示，表明在增强 AI 工作流中的生产力工具方面大有可为 [视频](https://vimeo.com/1017798749)。
- **Writer C 轮融资公告**：Writer 宣布以 **19 亿美元估值**完成 **2 亿美元的 C 轮融资**，旨在增强其 AI 企业解决方案。
  
  - 这笔资金将支持扩展其生成式 AI 应用，并获得了知名投资者的显著支持 [Tech Crunch 文章](https://techcrunch.com/2024/11/12/generative-ai-startup-writer-raises-200m-at-a-1-9b-valuation/)。
- **Supermaven 加入 Cursor**：Supermaven 宣布与 **Cursor** 合并，旨在开发先进的 AI 代码编辑器，并就新的 AI 工具功能展开合作。

- 尽管发生了转变，**Supermaven plugin** 将继续得到维护，这表明了对提高生产力的持续承诺（博客文章[链接](https://supermaven.com/blog/cursor-announcement)）。

**提到的链接**：

- [Supermaven joins Cursor](https://supermaven.com/blog/cursor-announcement)：Supermaven 正在加入 Cursor 以构建最好的 AI 代码编辑器。
- [来自 Aidan McLau (@aidan_mclau) 的推文](https://x.com/aidan_mclau/status/1856127488356712917)：我哭了
- [Supermaven Joins Cursor](https://www.cursor.com/blog/supermaven)：我们很高兴地宣布 Supermaven 正在加入 Cursor。
- [来自 Tim Dettmers (@Tim_Dettmers) 的推文](https://x.com/Tim_Dettmers/status/1856338240099221674)：这是很长一段时间以来最重要的一篇论文。它有力地证明了我们正在达到 quantization 的极限。论文指出：训练的 token 越多，需要的精度就越...
- [来自 Rowan Cheung (@rowancheung) 的推文](https://x.com/rowancheung/status/1854972388988908023)：Microsoft 本周推出了一个 Agent 框架，但完全没有引起注意。Agent 团队可以浏览互联网、内部文件、执行代码等。在这个例子中，Microsoft 团队...
- [oct-8-demo](https://vimeo.com/1017798749?autoplay=1&muted=1&stream_id=Y2xpcHN8MjI2ODY4NjYwfGlkOmRlc2N8W10%3D)：这是 Joseph Semrai 在 Vimeo 上的 "oct-8-demo"，Vimeo 是高质量视频及其爱好者的家园。
- [Magentic-One: A Generalist Multi-Agent System for Solving Complex Tasks - Microsoft Research](https://www.microsoft.com/en-us/research/articles/magentic-one-a-generalist-multi-agent-system-for-solving-complex-tasks/)：作者：Adam Fourney（首席研究员）；Gagan Bansal（高级研究员）；Hussein Mozannar（高级研究员）；Victor Dibia（首席研究软件工程师）；Saleema Amershi（合伙人研究经理）...
- [来自 Writer (@Get_Writer) 的推文](https://x.com/get_writer/status/1856336614651507155?s=46)：🎉 我们很高兴地宣布，我们已经筹集了 2 亿美元的 C 轮融资，估值为 19 亿美元，旨在通过全栈生成式 AI 改变工作方式！今天，数百家企业巨头如...
- [来自 Vivek Sodera (@vsodera) 的推文](https://x.com/vsodera/status/1856405968218714395?s=46)：祝贺 @may_habib, @waseem_s 和 @Get_Writer 团队获得 2 亿美元 C 轮融资（估值 19 亿美元 🦄🦄）。很自豪能成为这家一代人只有一次、持久的企业级 AI 公司的早期投资者...
- [来自 Greg Brockman (@gdb) 的推文](https://x.com/gdb/status/1856441156281753908)：我人生中最长的假期结束了。回到 @OpenAI 继续构建。
- [来自 Nous Research (@NousResearch) 的推文](https://x.com/nousresearch/status/1856417883934601246?s=46)：今天我们推出了 Forge Reasoning API Beta 版，这是 inference time scaling 的一项进步，可以应用于任何模型或一组模型，面向我们社区中的特定群体。https...
- [来自 Sam Julien (@samjulien) 的推文](https://x.com/samjulien/status/1856368522026467512?s=46)：大新闻！🎉 我很高兴地分享 @Get_Writer 已经筹集了 2 亿美元的 C 轮融资，估值为 19 亿美元 🚀 我们正在构建 AI 的未来并实现大规模的 ROI。看看我们的 CEO @may_...
- [The future of enterprise work](https://writer.com/blog/series-c-funding-writer/)：Writer 宣布了他们的 C 轮融资，筹集了 2 亿美元，估值为 19 亿美元，以帮助他们改变企业工作的未来。
- [来自 Joseph Semrai (@josephsemrai) 的推文](https://x.com/josephsemrai/status/1856045775454970015)：认识一下 Context Autopilot。它像你一样学习，像你一样思考，像你一样使用工具。凭借 SoTA 的 context 理解能力，它能够胜任当今的大多数信息工作。看它击败一支行业团队...
- [Dario Amodei: Anthropic CEO on Claude, AGI & the Future of AI & Humanity | Lex Fridman Podcast #452](https://www.youtube.com/watch?v=ugvHCXCOmm4)：Dario Amodei 是 Anthropic 的 CEO，该公司创建了 Claude。Amanda Askell 是一位研究 Claude 性格和个性的 AI 研究员。Chris...
- [Transformers.js: State-of-the-art Machine Learning for the web](https://youtu.be/n18Lrbo8VU8?si=c2SAiMyMWbbWR_Rj)：加入来自 HuggingFace 的 Joshua Lochner，了解 Transformers.js，这是一个令人兴奋的新 JavaScript 库，它使开发人员能够构建前所未有的 w...
- [autogen/python/packages/autogen-magentic-one at main · microsoft/autogen](https://github.com/microsoft/autogen/tree/main/python/packages/autogen-magentic-one)：一个用于 Agentic AI 的编程框架 🤖。通过在 GitHub 上创建账户，为 microsoft/autogen 的开发做出贡献。
- [Reddit - Dive into anything](https://www.reddit.com/r/ChatGPT/comments/1gpjspp/remember_this_50k_upvote_post_op_admitted_chatgpt/)：未找到描述。

- [Microsoft Research – Emerging Technology, Computer, and Software Research](https://www.microsoft.com/en-us/research): 探索 Microsoft 的研究，该网站展示了研究的影响力，并提供论文发表、产品、下载和研究职业信息。
- [Scaling Laws for Precision](https://arxiv.org/abs/2411.04330): 低精度训练和推理会影响语言模型的质量和成本，但目前的 Scaling Laws 尚未考虑到这一点。在这项工作中，我们设计了“精度感知”的 Scaling Laws...

---

### **Latent Space ▷ #**[**ai-announcements**](https://discord.com/channels/822583790773862470/1075282504648511499/1305630620419883201) (3 messages):

> - `Dust XP1`
> - `OpenAI Journey`
> - `Voice Questions for Recap Pod`
> - `AI Agent Infrastructure`
> - `SaaS and AI Software Impact`

- **Dust XP1 与日活跃使用率**: @spolu 分享了如何使用 **Dust XP1** 创建高效的工作助手，在客户中实现了令人印象深刻的 **88% Daily Active Usage**。
  
  - 这一集涵盖了早期的 **OpenAI Journey**，包括与 @gdb 和 @ilyasut 的合作。
- **关于 OpenAI 经历的透明度**: @spolu 幽默地提到，在一次精彩的对话中，他透露了比预期更多的关于他在 **OpenAI** 期间的事情。
  
  - 讨论为 2019 年至 2022 年 AI 模型的演变提供了独特的视角。
- **语音提问邀请**: 团队正在征集关于 **2 Years of ChatGPT** 回顾集的语音提问，由 @swyxio 发布。
  
  - 听众可以在[这里](https://www.speakpipe.com/LatentSpace)提交他们的语音提问，有机会被选中播出。
- **构建 AI Agent 基础设施的挑战**: 对话涵盖了与 AI Agent 基础设施相关的挑战，例如 **buy vs. build** 的抉择。
  
  - 还讨论了关于创建 Agent 依赖图和模拟 API endpoints 的见解。
- **SaaS 与 AI 的未来**: 对 **future of SaaS** 以及 **AI's impact** 对软件开发的影响的推测是播客的一个关键话题。
  
  - 参与者讨论了单人初创公司达到 **$1B 估值** 的潜在影响。

**提到的链接**:

- [Tweet from Latent.Space (@latentspacepod)](https://x.com/latentspacepod/status/1856071742386778582): 🆕 Agents @ Work: @dust4ai! https://latent.space/p/dust @spolu 畅谈早期 @openai 与 @gdb 和 @ilyasut 的历程、Dust XP1，以及如何制作真正有用的工作助手，实现 **88% Daily Act...
- [Tweet from Stanislas Polu (@spolu)](https://x.com/spolu/status/1856095897026711818): 透露了比我应该说的更多的关于 OpenAI 19-22 的内容 🙊 与 @FanaHOVA 和 @swyx 的对话非常棒，你们非常擅长构建问题的框架👌 引用 Latent.Space (@latentspacepod) ...
- [Send a voice message to LatentSpace](https://www.speakpipe.com/LatentSpace) : 排名第一的 AI Engineering 播客

---

### **GPU MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1189498205101109300/1305735248272822272) (14 messages🔥):

> - `GPU Memory vs Speed`
> - `Cloud GPU Providers`
> - `Building CUTLASS on Lambda Cloud`
> - `XOR Tensor Cores in Beamforming`
> - `Multiple GPUs for Memory Concerns`

- **GPU 显存与速度的权衡讨论**: 一位用户询问了在考虑从 **RTX 2060 Super** 升级到 **RTX 3090** 时显存与速度之间的权衡，并注意到了旧款二手 **Tesla** 卡的可用性。
  
  - 另一位成员回应称，较新的硬件通常更可靠，并建议个人开发者不要购买旧卡。
- **Vast.ai 作为经济实惠的云 GPU 选项**: 一位参与者推荐 [Vast.ai](https://vast.ai/pricing) 作为廉价的云 GPU 提供商，尽管由于用户可以出租自己的 GPU，它有一些古怪之处。
  
  - 他们分享了各种 GPU 的当前定价，包括 **A100** 和 **RTX 4090**，价格范围从每小时 **$0.30** 到 **$2.80** 不等。
- **Lambda Cloud 体验**: 几位成员分享了他们在 **Lambda Cloud** 上的经验，指出性能不错，但提到创建唯一集群存在困难。
  
  - 还有人幽默地评论了机器重启的可能性，这需要用户间歇性地检查实验。
- **Lambda Cloud 上的 CUTLASS 构建挑战**: 一位用户表达了对在 **Lambda Cloud** 上构建 **CUTLASS** 遇到困难的沮丧。
  
  - 这引发了其他人讨论可能的变通方法，或者对在构建过程中挣扎的共同境遇报以幽默。
- **XOR Tensor Cores 增强波束成形算法**: 一位用户分享了关于使用 **XOR Tensor Cores** 的见解，强调了其在超声扫描 **beamforming** 算法中的应用案例。
  
  - 这激发了关于如何将先进计算技术应用于特定科学应用的兴趣。

 

**提到的链接**: [Pricing | Vast.ai](https://vast.ai/pricing): 查看 Vast.ai 上热门 GPU 的定价

 

---

### **GPU MODE ▷ #**[**triton**](https://discord.com/channels/1189498204333543425/1189607595451895918/1305890093747277905) (3 条消息):

> - `Slack 成员添加`
> - `Triton Puzzle 讨论`
> - `Puzzle 工作组`

- **Slack 访问受限**：成员们注意到，他们已经**停止随意向 Slack 添加人员**，这表明政策可能发生了变化。
  - 这可能会影响寻求获取讨论和更新权限的新成员。
- **寻求 Triton Puzzle #9 的帮助**：一位成员在处理 **Triton Puzzle #9** 时表示困惑，并正在寻求关于如何利用提示（hints）的指导。
  - 他们提到已经完成了 **3-loop** 版本，但需要其他人对提示提供进一步的意见。
- **在 Triton-Puzzles 频道寻找帮助**：另一位成员建议，有一个名为 'triton-puzzles' 的**工作组频道**，供讨论解决方案的人员使用。
  - 他们建议在频道内进行搜索，以获取更多关于该 puzzle 的见解。

---

### **GPU MODE ▷ #**[**cool-links**](https://discord.com/channels/1189498204333543425/1189868872887705671/1305633718404255824) (2 条消息):

> - `Efficient Deep Learning Systems`
> - `AOT Compilation 特性`

- **探索 Efficient Deep Learning Systems 资料**：查看 [GitHub](https://github.com/mryab/efficient-dl-systems) 上由 HSE 和 YSDA 提供的 **Efficient Deep Learning Systems** 课程资料。
  - 该仓库包含丰富的资源，旨在增强你对 AI 中高效系统的理解。
- **宣称 AOT Compilation 具有更快的运行时**：一位成员询问了 [AOT Compilation](https://docs.polymagelabs.com/aot.html) 的性能优势，据称其运行速度比 JIT 编译快得多。
  - AOT Compilation 支持创建供离线使用的库，允许与 C/C++ 应用程序集成，并使用 CUDA 或 ROCm 进行 GPU 执行。

**提到的链接**：

- [GitHub - mryab/efficient-dl-systems: Efficient Deep Learning Systems course materials (HSE, YSDA)](https://github.com/mryab/efficient-dl-systems)：Efficient Deep Learning Systems 课程资料 (HSE, YSDA) - mryab/efficient-dl-systems
- [AOT Compilation — PolyBlocks 0.4 documentation](https://docs.polymagelabs.com/aot.html)：未找到描述

---

### **GPU MODE ▷ #**[**beginner**](https://discord.com/channels/1189498204333543425/1191300313928433664/) (1 条消息):

pondering_wanderer: 大家好

---

### **GPU MODE ▷ #**[**off-topic**](https://discord.com/channels/1189498204333543425/1215328286503075953/1306003618633355304) (9 条消息🔥):

> - `图像生成`
> - `Prompt Engineering`
> - `食物模型`
> - `AI 交互`
> - `Bot 验证`

- **对 AI 生成图像的好奇**：一位成员对一张图片的来源表示好奇，询问其是否由 AI 生成。
  - 另一位成员澄清说它不是 AI 生成的，随后引发了关于图像创建中 prompt 使用的进一步问题。
- **Text-to-Food 模型见解**：一位成员幽默地将自己的能力描述为 text-to-food 模型，根据 prompt 生成食物。
  - 他们开玩笑地提到了一个额外的“食物到排泄物”的概念，暗示了一种幽默的循环关系。
- **关于伪造消息的讨论**：一位成员思考了伪造交互的可能性，例如扮演他人并生成食物和地点的图像。
  - 这引发了关于 AI 在日常交互中的影响以及真实性的思考。
- **消耗循环**：一位成员俏皮地展示了一个将排泄物与土地和食物联系起来的循环，强调了自然界中的循环依赖。
  - 这一评论为对话增添了幽默感，反映了对自然过程的思考。
- **Bot 身份验证查询**：关于识别 Bot 的好奇心随之产生，一位成员询问其他人如何确认用户的真实性。
  - 这一观察强调了对数字通信中身份和信任的持续关注。

---

### **GPU MODE ▷ #**[**triton-puzzles**](https://discord.com/channels/1189498204333543425/1219683012707487794/1305946825500393513) (3 messages):

> - `Triton Puzzles`
> - `Triton Kernel Coding`
> - `Block Mapping Implementation`
> - `Tensor Copying`

- **关于 Puzzle 9 的澄清需求**：一位成员在完成了 **3-loop 版本**后，正在寻求关于如何利用 [Triton Puzzles](https://github.com/srush/Triton-Puzzles) 中 **Puzzle 9** 提示的建议。
  - 他们对提示的应用表示困惑，并正在寻找解决思路。
- **成功使用 Triton 进行 Tensor 复制**：另一位成员成功使用其 Triton kernel 将 **source tensor** 复制到 **destination tensor**，并通过提供的 Python 函数验证了结果，确认 **0** 个不匹配元素。
  - 然而，他们现在正努力应用特定的 **block_mapping 结构**，以便在复制过程中控制 destination tensor 中 block 的顺序。
- **请求 Triton Kernel 代码协助**：该成员请求协助编写 Triton 代码，以使用给定的 **block_mapping** 结构执行 tensor 复制，该结构涉及根据条件随机采样 block。
  - 他们附上了当前的 Triton kernel 代码和指令，但无法实现所需的功能。

---

### **GPU MODE ▷ #**[**liger-kernel**](https://discord.com/channels/1189498204333543425/1275130785933951039/1305653001670623313) (3 messages):

> - `Batch Normalization Challenges`
> - `Multi-GPU Synchronization`

- **Batch Norm 可能不值得投入精力**：一位成员讨论了 **batch normalization** 带来的复杂性，特别是与计算均值和方差所需的非连续输入矩阵相关的问题。
  - 转换 tensor 操作的必要性可能会导致 **沉重的内存操作**，这引发了对效率的担忧。
- **在多 GPU 设置中同步均值和方差**：在多 GPU 环境中，同步均值和方差参数至关重要，正如关于 **PyTorch SyncBatchNorm 操作** 的讨论中所指出的。
  - 成员们表示，在 **liger** 中复制这种行为预计会非常复杂，且并非易事。

---

### **GPU MODE ▷ #**[**self-promotion**](https://discord.com/channels/1189498204333543425/1288557096404516945/1305713173612986389) (1 messages):

> - `WebGPU`
> - `Surfgrad`
> - `Autograd Engine Optimization`

- **Surfgrad 引擎突破 1 TFLOP**：一位成员兴奋地分享了他们创建的 **Surfgrad**，这是一个构建在 **WebGPU** 之上的 autograd 引擎，在他们的 M2 芯片上实现了高达 **1 TFLOP** 的性能。
  - 他们提供了一个指向其[讨论 kernel 优化文章](https://zanussbaum.substack.com/p/optimizing-a-webgpu-matmul-kernel?r=coo9n)的链接，强调了使用 WebGPU 工作的乐趣。
- **在 Nomic 扩展大型可视化**：在构建大型 **TSNE 类可视化** 的背景下，该成员强调了在浏览器中显示数千万个数据点而不导致计算机过热的挑战。
  - 他们提到了由 Ben Schmidt 开发的 **Deepscatter**，作为在 **Nomic** 的讨论中听到的某些扩展问题的解决方案。
- **探索 WebGPU Autograd 库的匮乏**：该成员表示缺乏使用 WebGPU 构建的可用 **autograd 库**，因此创建了自己的项目作为教学练习。
  - 这一举措不仅加深了他们对 WebGPU 的理解，还涉及在过程中学习 **Typescript**。

**提到的链接**：

- [Zach Nussbaum (@zach_nussbaum) 的推文](https://x.com/zach_nussbaum/status/1856021159164424559)：我对 WebGPU 感到兴奋，所以自然而然地构建了 Surfgrad，一个构建在 WebGPU 之上的 autograd 引擎。我展示了如何将一个朴素的 kernel 优化到性能超过 1TFLOP。
- [优化 WebGPU Matmul Kernel 以获得 1TFLOP+ 性能](https://zanussbaum.substack.com/p/optimizing-a-webgpu-matmul-kernel?r=coo9n)：构建 Surfgrad，一个高性能、由 WebGPU 驱动的 autograd 库。

---

### **GPU MODE ▷ #**[**🍿**](https://discord.com/channels/1189498204333543425/1298372518293274644/1305658782055141387) (5 messages):

> - `Bot 测试方法`
> - `任务队列实现`
> - `频道动态`

- **Bot 测试是如何进行的？**：一名成员询问了目前测试 Bot 的方法，询问是在服务器频道内进行还是通过另一个服务器进行。
  
  - 另一名成员确认了该频道的存在，但暗示由于 Bot 已经可以正常运行，可能会停用该频道。
- **任务队列实现机会**：一名成员提出协助实现除 GitHub actions 之外的任务队列，并表示随时准备开始。
  
  - 作为回应，另一名成员建议开启一个 issue，从高层级讨论该实现方案。
- **关于 Bot 功能的沟通**：一名成员分享了另一个频道的更新 DM 邀请，表明关于 Bot 测试的沟通仍在进行中。
  
  - 他们提到，由于 Bot 表现良好，他们即将关闭该频道。

 

---

### **Cohere ▷ #**[**discussions**](https://discord.com/channels/954421988141711382/954421988783444043/1305710893127958668) (22 messages🔥):

> - `Command R 停用担忧`
> - `aya_collection 数据集不一致性`
> - `森林火灾预测 AI 项目`
> - `数据集翻译质量`
> - `AI 应用讨论`

- **Command R 停用疑虑得到解决**：一位用户询问了 **Command R** 可能停用的情况及其对他们模型的影响，得到了目前没有停用计划的保证。
  
  - 另一名成员确认了这一点，表示：**“绝对不会，完全不用担心，在可预见的未来没有这样的计划。”**
- **发现 aya_collection 数据集的不一致性**：一名成员审查了 **aya_collection** 数据集，并报告了 19 种语言翻译的不一致性，指出各种语言翻译的行数存在显著差异。
  
  - 具体而言，他们注意到在 **translated_cnn_dailymail** 中，英语行数为 **249716**，而阿拉伯语和法语仅为 **124858**。
- **展示翻译不匹配的示例**：用户提供了 **translated_cnn_dailymail** 数据集中翻译不匹配的具体示例，强调一句关于肺肿瘤的英语句子与阿拉伯语和法语翻译不匹配。
  
  - 他们还指出，英语句子的数量成比例地高于其他语言。
- **森林火灾预测 AI 项目**：一位用户分享了他们使用 **Catboost** 和 **XLModel** 构建**森林火灾预测 AI** 的项目见解，强调了模型可靠性的必要性。
  
  - 他们澄清了确保模型未来可行性的兴趣，因为他们有在 AWS 上部署的计划。
- **新开发的模型推荐**：针对森林火灾预测项目，一名成员建议使用最新版本的 **Command R**，以获得更高的性能和成本效率。
  
  - 他们鼓励联系销售团队以获得进一步支持，并提到了最新的更新，如 **command-r-08-2024**。

 

---

### **Cohere ▷ #**[**announcements**](https://discord.com/channels/954421988141711382/996880279224451154/1305931192432070759) (1 messages):

> - `研究原型 Beta 测试`
> - `写作工具反馈`

- **研究工具 Beta 测试机会**：一个旨在支持报告创建和分析等**研究与写作任务**的研究原型现已开放限量 Beta 测试报名，报名链接见 [此链接](https://forms.gle/Teis9VwM6eZP6nxVA)。
  
  - 该项目正在寻找愿意提供**详细且建设性反馈**的参与者，以便在早期测试阶段完善其功能。
- **测试参与者要求**：该工具专为经常创建**基于文本的交付成果**并需要在工作流中获得协助的人员设计。
  
  - 测试人员预计将与团队进行迭代合作，以影响这一处理**复杂任务**的高效助手的开发。

 

**提到的链接**：[Research Prototype - Early Beta Sign Up Form](https://forms.gle/Teis9VwM6eZP6nxVA)：感谢您有兴趣参加我们研究原型的 Beta 测试阶段——这是一款旨在帮助用户处理研究和写作任务的工具，例如：创建复杂的报告、进行...

 

---

### **Cohere ▷ #**[**questions**](https://discord.com/channels/954421988141711382/1168411509542637578/1305880511129981020) (3 条消息):

> - `使用 RAG 的 AI Assistant`
> - `Cohere Dashboard 登录`
> - `组织 ID (Organizational ID) 使用`

- **使用 RAG 构建 AI Assistant**：你可以通过利用 [Chat](https://docs.cohere.com/reference/chat)、[Embed](https://docs.cohere.com/docs/embed) 和 [Rerank](https://docs.cohere.com/docs/rerank) 端点来创建一个处理 PDF 并生成内容的 AI Assistant。
  
  - 实现 RAG 系统允许你从 PDF 中检索相关信息并生成上下文文本，从而显著提高响应的准确性。
- **Cohere Dashboard 登录指南**：要访问你的账户，请访问 [Cohere Dashboard](https://dashboard.cohere.com/) 并使用你的电子邮件和密码登录。
  
  - 新用户可以注册该服务，在同意 [Terms of Use](https://cohere.com/terms-of-use) 和 [Privacy Policy](https://cohere.com/privacy) 后，即可创建账户。
- **了解 Org ID 的重要性**：组织 ID 帮助 Cohere 识别你的账户并评估其状态。
  
  - 关于如何使用它的具体信息，其他用户可以提供详细的见解。

**提到的链接**：

- [Login | Cohere](https://dashboard.cohere.com/)：通过一个易于使用的 API 登录以访问先进的 Large Language Models 和 NLP 工具。
- [Retrieval Augmented Generation (RAG) — Cohere](https://docs.cohere.com/docs/retrieval-augmented-generation-rag)：使用 Retrieval Augmented Generation 和 Cohere 的 Chat API，结合外部数据和行内引用生成文本。

---

### **Cohere ▷ #**[**api-discussions**](https://discord.com/channels/954421988141711382/1168578329423642786/1305914904238100480) (9 条消息🔥):

> - `Cohere API /rerank 问题`
> - `return_documents 参数移除`
> - `API 变更故障排查`
> - `Python Async Client 使用`
> - `意外的 API 行为`

- **Cohere API /rerank 出现问题**：用户报告在使用 `/rerank` 端点时突然出现错误，具体遇到了 **UnprocessableEntityError**，提示消息为 **return_documents** 不再是有效字段。
  
  - *调用代码没有任何改动*，但该问题对多名用户来说是意外出现的。
- **return_documents 参数被弃用**：经过调查，用户通过**移除 return_documents 字段**解决了该问题，这表明默认行为已更改为 **False**。
  
  - 几位用户指出 Python SDK 和文档仍将其列为有效字段，引发了对 API 意外调整的疑问。
- **团队标记 API 以进行故障排查**：一名团队成员确认了 **rerank 问题**，并将其提交给开发团队以寻求紧急关注。
  
  - 他们请求用户提供更多代码上下文，特别是他们是在使用 SDK 还是直接调用 API，以便于故障排查。
- **分享 V2 Python Async Client 详情**：一位用户报告使用了 **V2 Python async client**，并提供了一段代码片段，展示了他们在没有该问题标志的情况下如何调用 **rerank** 方法。
  
  - 他们确认在移除 `return_documents` 标志后问题得以解决，这为用户间关于 API 行为的持续讨论提供了参考。
- **API 行为修订确认**：针对用户反馈，一名团队成员表示移除 **return_documents** 参数是一个无意的改动。
  
  - 他们向用户保证说：**“我们正在把它恢复回来”**，表明 API 很快就会得到修正。

---

### **Cohere ▷ #**[**projects**](https://discord.com/channels/954421988141711382/1218409701339828245/1305870810686427147) (2 条消息):

> - `分享工具`
> - `社区参与`

- **对分享工具感到兴奋**：*sssandra* 对 *Jake* 分享的一个工具表示兴奋，称其非常**酷**，并计划尝试一下。
  
  - *Jake* 感谢了 *sssandra* 的热情。
- **社区赞赏**：这次互动突显了社区氛围，成员们互相欣赏彼此的贡献并分享积极的反馈。
  
  - 这展示了一个参与度极高的环境，成员们公开讨论并尝试各种工具和想法。

---

### **Cohere ▷ #**[**cohere-toolkit**](https://discord.com/channels/954421988141711382/1254901651081269268/1306009545394618448) (2 条消息):

> - `ICS Calendar Support`
> - `File Content Viewing`

- **Discord 服务器新增 ICS 日历支持**：一名成员宣布增加对 **ICS 日历文件** 的支持，以增强 Discord 服务器上的活动管理，并称鉴于活动数量之多，这是必不可少的。
  
  - *“不添加对此功能的支持将是一种罪过”*，这强调了该功能与社区活动的相关性。
- **引入文件内容查看功能**：同一位成员介绍了一项新功能，允许用户直接在服务器内**查看上传的文件内容**。
  
  - 虽然他们承认该功能缺乏引人注目的介绍，但从积极的反馈来看，社区似乎非常认可这一改进。

 

---

### **HuggingFace ▷ #**[**cool-finds**](https://discord.com/channels/879548962464493619/897390579145637909/1305731325587619851) (4 条消息):

> - `Home AI Server for LLMs`
> - `NeurIPS 2024 Graph Neural Networks`
> - `Phase Transitions in Image Denoising`
> - `Ultra Realistic AI Models`
> - `E-commerce and AI Fashion`

- **在家低成本托管 LLM**：对于那些希望在不支付高昂硬件成本的情况下运行 LLM 的人，一段 [YouTube 视频](https://www.youtube.com/watch?v=iflTQFn0jx4) 展示了如何使用单个 **3060 GPU** 和一台 **3620** 搭建一台性能出色的 AI 家用服务器。
  
  - 该服务器在搭配 **Llama 3.2** 模型时表现出令人印象深刻的性能，使其成为一个预算友好的解决方案。
- **NeurIPS 2024 重点关注 Graph Neural Networks**：NeurIPS 2024 展示了对 **Graph Neural Networks** 和几何学习兴趣的激增，包含约 **400-500 篇论文**，显著超过了 ICML 2024 的投稿量。
  
  - 关键主题包括 Diffusion 模型、Transformer、Agent 和知识图谱，理论重点在于 **Equivariance** 和 **Generalization**，详见 [GitHub](https://github.com/azminewasi/Awesome-Graph-Research-NeurIPS2024)。
- **探索图像去噪技术**：一篇名为 *Phase Transitions in Image Denoising via Sparsity* 的新论文讨论了图像处理中的高级概念，可在 [Semantic Scholar](https://www.semanticscholar.org/paper/Phase-Transitions-in-Image-Denoising-via-Sparsely-Carroll-Carlson/55cb0e93f4f98b851ca4343e4a456b2e9c8241ec) 上查阅。
  
  - 该研究旨在解决图像去噪中的挑战，是该领域持续探索的一部分。
- **为电子商务生成逼真的 AI 模型**：一名成员正在寻找能够为其专注于**婴儿服装**的电子商务创业项目生成穿着特定品牌服装的超逼真 AI 模型的 **LLM**。
  
  - 他们希望获得有关模型的建议，以便在竞争激烈的在线时尚市场中准确代表其品牌。

**提到的链接**：

- [Llama 3.2 Vision 11B LOCAL Cheap AI Server Dell 3620 and 3060 12GB GPU](https://www.youtube.com/watch?v=iflTQFn0jx4)：我们正在测试一台基于单个 3060 GPU 和 3620 的极具性价比的家用 AI 服务器，当搭配全新的 Llama 3.2 11...
- [GitHub - azminewasi/Awesome-Graph-Research-NeurIPS2024: All graph/GNN papers accepted at NeurIPS 2024.](https://github.com/azminewasi/Awesome-Graph-Research-NeurIPS2024)：NeurIPS 2024 收录的所有 Graph/GNN 论文。可以通过在 GitHub 上创建账号来为 azminewasi/Awesome-Graph-Research-NeurIPS2024 做出贡献。

---

### **HuggingFace ▷ #**[**i-made-this**](https://discord.com/channels/879548962464493619/897390720388825149/1305673306149621871) (12 messages🔥):

> - `Qwen2.5 Coder 性能`
> - `Mochi -1-preview 视频生成器`
> - `电商嵌入模型 (Ecommerce Embedding Models)`
> - `AutoML 应用`
> - `开源 Prompt 管理`

- **Qwen2.5 Coder 超越 GPT4o 和 Claude 3.5 Sonnet**：在测试中，**Qwen2.5 Coder 32B** 的表现优于 **GPT4o** 和 **Claude 3.5 Sonnet**，展示了令人印象深刻的能力。查看此 [YouTube 视频](https://youtu.be/Xs0EkLYu6hw)中的分析。
  
  - *普遍共识*认为 Qwen2.5 Coder 正在迅速成为编程 AI 领域的强力竞争者。
- **介绍 Mochi -1-preview 视频生成器**：**Mochi -1-preview 视频生成器**可将文本 Prompt 转换为视频，并允许自定义帧数和 FPS。请注意，**高质量选项 (High-Quality Option)** 需要一台至少拥有 **42GB VRAM** 的强大 GPU。
  
  - 邀请用户体验该工具并提供功能反馈：[Hugging Face 上的 Mochi](https://huggingface.co/spaces/thesab/mochi-1)。
- **最先进的电商嵌入模型 (Ecommerce Embedding Models) 已发布**：新的**电商嵌入模型**已发布，其性能优于 **Amazon-Titan-Multimodal** 等模型，提升幅度高达 **88%**。这些模型可以直接从 **Hugging Face** 访问，或与 **Marqo Cloud** 集成以构建应用程序。
  
  - 参考此 [Hugging Face Collection](https://huggingface.co/collections/Marqo/marqo-ecommerce-embeddings-66f611b9bb9d035a8d164fbb) 以获取更多关于功能和性能评估的细节。
- **使用 Streamlit 和 H2O.ai 构建的 AutoML 应用**：一个使用 **Streamlit** 和 **H2O.ai** 构建的 **AutoML 应用**允许用户轻松上传数据集并进行预测。用户友好的界面简化了复杂的机器学习工作流，该项目已在 [GitHub](https://github.com/SanshruthR/AquaLearn) 上开源。
  
  - 该工具旨在降低模型训练的门槛，同时提供必要的工作流管理功能。
- **基于 Markdown 和 JSX 的开源 Prompt 管理**：**Promptdx** 项目提供了一种基于 Markdown 和 JSX 的声明式 **Prompt 编程**方法。这个 GitHub 项目旨在简化各种应用的 Prompt 管理：[GitHub 上的 Promptdx](https://github.com/puzzlet-ai/promptdx/)。
  
  - 其功能迎合了希望高效增强和管理 Prompt 的开发者需求。

**提到的链接**：

- [GitChameleon: Unmasking the Version-Switching Capabilities of Code Generation Models](https://arxiv.org/abs/2411.05830)：软件库的快速演进对代码生成模型提出了重大挑战，这些模型必须适应频繁的版本更新，同时保持与之前版本的兼容性...
- [Mochi 1 - a Hugging Face Space by thesab](https://huggingface.co/spaces/thesab/mochi-1)：未找到描述
- [Volko76 (Volko)](https://huggingface.co/Volko76)：未找到描述
- [Marqo-Ecommerce-Embeddings - a Marqo Collection](https://huggingface.co/collections/Marqo/marqo-ecommerce-embeddings-66f611b9bb9d035a8d164fbb)：未找到描述
- [Qwen2.5 Coder 32B vs GPT4o vs Claude 3.5 Sonnet (new)](https://youtu.be/Xs0EkLYu6hw)：让我们看看哪个模型最强
- [GitHub - SanshruthR/AquaLearn: Upload CSV data, get predictions and save models](https://github.com/SanshruthR/AquaLearn)：上传 CSV 数据，获取预测并保存模型。通过在 GitHub 上创建账号为 SanshruthR/AquaLearn 的开发做出贡献。
- [GitHub - DarkStarStrix/Auto_Api: A simplified machine learning framework](https://github.com/DarkStarStrix/Auto_Api)：一个简化的机器学习框架。通过在 GitHub 上创建账号为 DarkStarStrix/Auto_Api 的开发做出贡献。
- [GitHub - puzzlet-ai/promptdx: Declarative prompt programming based on Markdown and JSX](https://github.com/puzzlet-ai/promptdx/)：基于 Markdown 和 JSX 的声明式 Prompt 编程 - puzzlet-ai/promptdx

---

### **HuggingFace ▷ #**[**reading-group**](https://discord.com/channels/879548962464493619/1156269946427428974/1305933235796054067) (2 条消息):

> - `读书会公告`
> - `Arxiv 上的论文`
> - `论文作者`

- **读书会定于周四举行**：**读书会**将于本周四举行，鼓励成员通过[此链接](https://discord.gg/hugging-face-879548962464493619?event=1305932679396458506)加入讨论。
  - 参与者可以期待关于 AI 最新进展的见解和热烈讨论！
- **Arxiv 上发布了新论文**：可以在[此处](https://arxiv.org/abs/2407.14933)查阅一篇详述该领域近期进展的新论文。
  - 它提供了与社区当前正在进行的讨论相关的宝贵信息和发现。
- **了解作者**：该论文由多位作者共同完成，包括 **Shayne Longpre**、**Robert Mahari** 和 **Ariel Lee** 等，并附有他们的个人资料链接。
  - 鼓励成员探索作者的作品，以更深入地理解论文的背景和影响。

**提到的链接**：[Consent in Crisis: The Rapid Decline of the AI Data Commons](https://arxiv.org/abs/2407.14933)：通用人工智能 (AI) 系统构建在海量的公共网络数据之上，这些数据被汇编成诸如 C4、RefinedWeb 和 Dolma 等语料库。据我们所知，我们进行了首次……

---

### **HuggingFace ▷ #**[**NLP**](https://discord.com/channels/879548962464493619/922424173916196955/1305624110285717525) (5 条消息):

> - `Langchain SQL Agent 的评估指标`
> - `Agent 轨迹评估`
> - `Fast-langdetect 用法`

- **寻求更简单的 Langchain SQL Agent 评估指标**：一位成员询问是否有更简单的 **Langchain SQL Agent** 评估指标，并讨论了诸如 Agent 轨迹评估等复杂问题。
  - 他们请求提供资源或方法，包括 **YouTube 视频**或 Python 代码示例，以便更好地理解。
- **社区对 Langchain SQL Agent 评估的支持**：一位成员询问其他人是否有 **Langchain SQL Agent** 评估的经验，由于自身知识有限，寻求帮助。
  - 他们的请求表达了对社区协作或共享资源的希望。
- **提到 Fast-langdetect 工具**：另一位成员提到他们正在项目中使用 **fast-langdetect**，这表明它可能是语言检测的一个解决方案。
  - 这意味着其他人可能会在他们的评估或相关工作中考虑使用 **fast-langdetect**。

---

### **HuggingFace ▷ #**[**diffusion-discussions**](https://discord.com/channels/879548962464493619/1009713274113245215/1305779784609038357) (1 条消息):

> - `Diffusers 库 Schedulers`
> - `继承自 nn.Module`

- **关于 Diffusers 库中 Schedulers 的咨询**：一位成员询问目前 **diffusers** 库中所有的 **schedulers** 是否都继承自 **nn.Module** 类。
  - 该问题旨在澄清库中 schedulers 的结构化实现。
- **关于 Scheduler 功能的讨论**：另一位用户参与了讨论，解释说理解 schedulers 的继承关系对于在 AI 模型中有效利用它们至关重要。
  - 他们强调，了解哪些组件派生自 **nn.Module** 有助于调试和优化模型性能。

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1305647322197262357) (3 条消息):

> - `PursuitGov 转型`
> - `使用 ColPali 作为重排序器`
> - `Cohere 多模态嵌入`

- **PursuitGov 彻底改变 B2G 服务**：通过使用 **LlamaParse**，@PursuitGov 在一个周末内成功解析了 **400 万页**文档，显著增强了其 B2G 服务。
  - 这一转型使复杂文档格式的准确率提高了 **25-30%**，让客户能够从公共部门数据中**发现隐藏的机会**。
- **利用 ColPali 进行结果重排序**：一位成员分享了关于使用 **ColPali** 作为重排序器 (re-ranker) 的见解，以在**多模态索引**中获得高度相关的搜索结果。
  - 该技术涉及利用 **@cohere 的多模态嵌入 (embeddings)** 进行初始检索，整合文本和图像以获得最佳结果。

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1305700959409672233) (12 条消息🔥):

> - `下一个发布日期`
> - `自动化工作流流程`
> - `FastAPI 与流式响应`
> - `FastAPI 中的 SSE`
> - `测试 LlamaIndex 工作流`

- **对下个版本自动化的期望**：一位成员对发布流程的繁琐表示沮丧，称 *“噢对，我得做那个。真是太繁琐了，哈哈”*，并旨在实现更多自动化。
  
  - 他们分享了一个 [LlamaIndex v0.11.23 的 GitHub pull request](https://github.com/run-llama/llama_index/pull/16919) 以展示正在进行的更新。
- **替换自定义工作流的兴趣**：一位成员渴望用 LlamaIndex 工作流替换他们的 AI 自定义工作流，表示 *“我准备好用 LlamaIndex 工作流替换我的 AI 自定义工作流了，全力投入！”*
  
  - 这种热情反映了将 LlamaIndex 更全面地集成到其项目中的趋势。
- **事件流（Event Streaming）的挑战**：有人对工作流中的事件流提出了担忧，指出事件没有立即发送，这可能表明存在协程（coroutine）调度问题。
  
  - 该成员提供了详细的工作流代码来展示他们的方法并寻求反馈。
- **在 FastAPI 之外测试工作流**：一位成员建议在 FastAPI 之外测试 LlamaIndex 工作流，并表示它在终端环境中运行良好。
  
  - 他们演示了一个流式工作流，该工作流使用 LlamaIndex 框架成功实现了事件流式传输。
- **探索 FastAPI 的 StreamingResponse**：讨论围绕使用 FastAPI 的 StreamingResponse 展开，一位成员指出它仅在遇到换行符时才流式传输数据块（chunks）。
  
  - 他们建议使用更高级的流式传输技术，例如使用 `llm.astream_complete()` 将每个 token 作为流事件写入。

 

**提到的链接**：[v0.11.23 by logan-markewich · Pull Request #16919 · run-llama/llama_index](https://github.com/run-llama/llama_index/pull/16919)：未找到描述

 

---

### **LAION ▷ #**[**general**](https://discord.com/channels/823813159592001537/823813160075132991/1305643452901494955) (4 条消息):

> - `AI 领域的游击营销 (Gorilla Marketing)`
> - `空调目标检测项目`

- **AI 公司拥抱游击营销**：一位成员指出 **AI 公司非常喜欢游击营销**，这可能是指非传统的促销策略。
  
  - 他们分享了一个幽默的 [大猩猩挥舞美国国旗的 GIF](https://tenor.com/view/harambe-america-murica-flag-waving-gif-17339298)，为讨论增添了轻松的氛围。
- **目标检测项目求助**：一位成员详细介绍了一个使用 **Python Django** 的 **空调目标检测** 项目，旨在识别空调类型和品牌。
  
  - *他们请求协助*，表示在开发此识别功能方面需要支持。

 

**提到的链接**：[Harambe America GIF - Harambe America Murica - Discover & Share GIFs](https://tenor.com/view/harambe-america-murica-flag-waving-gif-17339298)：点击查看 GIF

 

---

### **LAION ▷ #**[**research**](https://discord.com/channels/823813159592001537/824374369182416994/1305724617440235571) (7 条消息):

> - `GitChameleon`
> - `SCAR`
> - `NVIDIA paper on frequency noise`
> - `Sparse Autoencoders`
> - `Code generation models`

- **为代码生成模型引入 GitChameleon**：新数据集 \\textbf{\\GitChameleon{ }} 引入了 **116 个 Python 代码补全问题**，这些问题以特定的库版本为条件，并配有可执行的单元测试，以严格评估 LLM 的能力。
  
  - 旨在解决现有基准测试忽略软件库演进的动态特性且未评估实际可用性的局限性。
- **用于概念检测的 SCAR 正式发布**：SCAR 是一种用于在 LLM 中进行精确概念检测和控制的方法，它使用 Sparse Autoencoders 以监督方式学习**单语义特征 (monosemantic features)**。
  
  - 它为**毒性、安全性和写作风格**等概念提供了强大的检测能力，并可在 Hugging Face 的 transformers 中进行实验。
- **NVIDIA 关于噪声频率训练的论文**：NVIDIA 的论文提出了一个概念，即在正向加噪步骤中，**高空间频率**比低频率加噪更快。
  
  - 在反向去噪步骤中，模型被显式训练为**从低频到高频**工作，为训练提供了一种独特的方法。
- **关于 NVIDIA 论文清晰度的讨论**：一位成员指出 NVIDIA 论文中的解释**难以理解**，但他们掌握了噪声训练方法的基本思想。
  
  - 尽管内容**组织混乱**，但论文中提出的概念被认为很有趣且值得探索。

**提到的链接**：

- [GitChameleon: Unmasking the Version-Switching Capabilities of Code Generation Models](https://arxiv.org/abs/2411.05830)：软件库的快速演进对代码生成模型提出了重大挑战，模型必须适应频繁的版本更新，同时保持与先前版本的兼容性...
- [Edify Image: High-Quality Image Generation with Pixel Space Laplacian Diffusion Models](https://arxiv.org/abs/2411.07126)：我们介绍了 Edify Image，这是一个能够生成具有像素级精度的照片级真实图像内容的 Diffusion Models 家族。Edify Image 利用级联的像素空间 Diffusion Models 进行训练...

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #**[**discussion**](https://discord.com/channels/1111172801899012102/1111353033352294440/1305839815807467541) (11 条消息🔥):

> - `Test Cases Overwriting`
> - `Qwen-2.5 Invalid AST Issues`
> - `Raw Output Format Confusion`
> - `Quantized Fine-tuned Models Evaluation`

- **为自定义模型覆盖测试用例**：一位成员询问在修改 handler 后，如何为他们的自定义模型覆盖或重新运行测试用例。
  
  - 另一位成员建议删除 `result` 文件夹中的结果文件，或更改 `constant.py` 中的路径以保留旧结果。
- **Qwen-2.5 输出中的无效 AST 错误**：一位成员描述了微调 Qwen-2.5 1B 模型时遇到的问题，尽管模型输出有效，但仍导致 INVALID AST 错误。
  
  - 成员们讨论了一种特定的错误输出格式，其中包括一个不匹配的闭括号，表明存在语法问题。
- **对 JSON 结构输出的困惑**：一位成员对模型输出 JSON 结构而不是预期的函数调用格式表示困惑。
  
  - 其他人澄清说，QwenHandler 理想情况下应该将 JSON 结构转换为函数形式，从而引发了关于输出预期的讨论。
- **评估量化微调模型**：一位成员提出了关于评估量化微调模型的问题，特别是关于它们在 vllm 上的部署。
  
  - 他们提到在模型服务中使用特定参数，如 `--quantization bitsandbytes` 和 `--max-model-len 8192`。

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**hackathon-announcements**](https://discord.com/channels/1280234300012494859/1280236929379602493/1305965804898095257) (1 条消息):

> - `LLM Agents MOOC Hackathon`
> - `LambdaAPI 演示`
> - `Hackathon 报名`
> - `创新 LLM Agents 赛道`

- **LLM Agents MOOC Hackathon 今天直播！**：欢迎参加今天 11/12 **下午 4 点 (PT)** 的 **LLM Agents MOOC Hackathon** 直播，届时 [@LambdaAPI](https://youtube.com/live/EUzVW6oRpIo?feature=share) 将进行实操演示。
  
  - 参与者可以在会议期间通过实用的见解学习如何**增强他们的 Hackathon 项目**。
- **与 2,000 名创新者一起加入 Hackathon**：已有约 **2,000 名创新者**报名参加了此次 Hackathon，重点是在包括 **Applications**、**Benchmarks** 等多个赛道上构建创新的 LLM agents。
  
  - 有兴趣的参与者仍可在 [rdi.berkeley.edu/llm-agents-hackathon](https://rdi.berkeley.edu/llm-agents-hackathon) 报名参加。
- **实操演示助力 Hackathon 成功**：**@LambdaAPI** 团队将提供实操性的“入门”演示，旨在增强参与者的 Hackathon 项目。
  
  - 这一参与有望为参与者的开发旅程提供宝贵支持。

 

**提到的链接**：[LLM Agents MOOC Hackathon - Lambda Labs Workshop](https://youtube.com/live/EUzVW6oRpIo?feature=share)：未找到描述

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-questions**](https://discord.com/channels/1280234300012494859/1280370030609170494/1305623170065109053) (5 条消息):

> - `Google Forms 确认`
> - `OpenAI Org 额度`

- **Google Forms 确认提交**：一名成员询问是否可以通过私信确认提交，另一名成员回复说，收到 **Google Forms 的确认**即表示提交已收到。
  
  - 这表明了一个简化的提交跟踪流程，无需直接联系。
- **重新提交 OpenAI Org ID 以获取额度**：一名成员提到他们在 OpenAI Org 申请中提交了错误的 **Organization ID**，现在已用正确信息进行了更正。
  
  - 他们被告知，由于重新提交，额度到账预计会有 **1-2 周的延迟**。

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-lecture-discussion**](https://discord.com/channels/1280234300012494859/1282734248112947210/1305668772408983652) (4 条消息):

> - `NVIDIA 的具身智能 (Embodied AI) 演讲`
> - `AI 权利的伦理`
> - `规范对齐 (Normative Alignment) 讨论`
> - `退伍军人节取消课程`

- **NVIDIA 的具身智能 (Embodied AI) 引发争议**：关于 **NVIDIA 的演示文稿**引发了担忧，该文稿暗示了公众对**具身智能 (Embodied AI)** 的渴望。
  
  - 成员们就赋予类人 AI 道德权利的影响进行了辩论，指出缺乏关于**规范对齐 (Normative Alignment)** 的讨论。
- **规范对齐 (Normative Alignment) 讨论的缺失令人担忧**：一名成员对讲座中**缺乏关于 LLM agents 规范对齐的讨论**表示担忧。
  
  - 这个问题在社区内引起了明显的不适，强调了对 AI 发展的伦理担忧。
- **因退伍军人节今天没有讲座**：成员们被告知，由于**退伍军人节**，**今天没有讲座**。
  
  - 一个提醒被忽视了，这让一些参会者对取消讲座感到失望。
- **AI 权利 vs. 人类责任**：一位成员感叹，人类可能很快就会给**按其形象创造的 AI** 权利，却忽视了近亲物种灭绝等问题。
  
  - 对话强调了在为了经济利益而未能保护濒危近亲物种的同时，却优先考虑 AI 权利的**虚伪性**。

 

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general**](https://discord.com/channels/1104757954588196865/1104757955204743201/1305907298593538070) (1 messages):

> - `FOSDEM AI DevRoom`
> - `Low-level AI Engineering`
> - `AI Project Collaboration`
> - `Fine-tuning Presentations`
> - `Sponsorship and Travel Stipends`

- **组织 FOSDEM AI DevRoom**：AIFoundry 团队正在组织将于 **2025 年 2 月 2 日**举行的 **FOSDEM AI DevRoom**，重点关注 ggml/llama.cpp 及其他相关项目，旨在汇聚 AI 领域的贡献者和开发者。
  
  - 欢迎底层 AI 核心开源项目维护者提交提案，提交截止日期为 **2024 年 12 月 1 日**，有趣的议题可能获得差旅补贴。
- **AI 工程演讲征集**：他们正在寻找演讲者，分享关于 **Fine-tuning**、模型量化 (quantization) 和分布式计算的实践经验与见解，以促进核心 AI 设计方面的协作。
  
  - 他们希望该 DevRoom 能达到 Linux Plumbers Conference 在 AI 社区中的重要地位。

**提及的链接**：[FOSDEM 2025 - Low-Level AI Engineering & Hacking Dev Room](https://aifoundry.org/fosdem-2025-low-level-ai-engineering-hacking-dev-room)：探索 FOSDEM 新设立的 "Low-Level AI Hacking & Engineering" Dev Room，展示驱动 AI 行业的开源项目。提交议程或成为这一创新项目的赞助商...

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general-help**](https://discord.com/channels/1104757954588196865/1110594519226925137/1305899235262332938) (8 messages🔥):

> - `Fine-tuning with Axolotl`
> - `Tokenization Configuration`
> - `Default System Prompts`

- **使用 Axolotl 以 Alpaca 格式进行微调**：一位用户阐明了使用 Axolotl 进行微调的设置，提到使用 **Alpaca 格式**的数据集进行预处理以进行 Fine-tuning。
  
  - 强调了在此过程后，tokenizer_config.json 不包含 **chat template 字段**。
- **使用 chat template 更新 tokenizer 配置**：另一位成员分享了一个**简单的方法**，通过复制特定的 JSON 结构将 **chat template** 添加到 tokenizer 配置中。
  
  - 他们还建议修改 Axolotl 内部的设置，以确保在未来的配置中自动包含此内容。
- **需要为默认系统提示词进行修改**：提醒指出，分享的模板未包含 **Alpaca** 的默认 System Prompts，可能需要调整。
  
  - 用户被告知可以在 **\### Instruction** 之前包含条件语句，以集成所需的提示词。
- **为用户分享默认系统提示词**：分享了默认的 System Prompts 以提供微调的额外上下文，包括典型的指令格式。
  
  - **System Prompts** 为生成更符合指定任务的响应奠定了基础。

---

### **DSPy ▷ #**[**general**](https://discord.com/channels/1161519468141355160/1161519469319946286/1305970007024340992) (7 messages):

> - `Annotations in dspy signatures`
> - `Usage of custom types in outputs`

- **探索 dspy.Signature 中的注解**：成员们讨论了 dspy signatures 中注解 (Annotations) 的使用，澄清虽然基础注解有效，但使用 **list[MyClass]** 等自定义类型具有潜力。
  
  - 一位成员确认字符串形式在此处不起作用，建议优先使用显式类型定义。
- **成功实现自定义 Signature**：一位成员分享了一个成功的 Signature 实现，在输出中使用字典列表，展示了临床实体的提取。
  
  - 该实现包括输入和输出字段的详细描述，表明了定义复杂数据结构的灵活方法。

---

### **OpenInterpreter ▷ #**[**general**](https://discord.com/channels/1146610656779440188/1147665339266650133/1305754597272453142) (3 messages):

> - `Linux Mint installation`
> - `Microsoft Copilot interaction`
> - `Interpreter CLI issues`

- **Linux Mint 在虚拟机中的困扰**：在 **Virtual Machine Manager** 中安装 **Linux Mint** 后，用户报告网络无法正常工作。
  
  - 不过，有人尝试在一个名为 **Boxes** 的应用中安装 **Linux Mint**。
- **Microsoft Copilot 沟通中断**：与 **Microsoft Copilot** 的反复交互显示出挫败感，因为命令未按要求配置。
  
  - 用户强调没有创建网桥，但他们设法自行创建了一个。
- **OS X 上的 Interpreter CLI Bug**：有报告称 **Interpreter CLI** 在 **OS X** 上会出现文件持久化问题并意外退出。
  
  - 用户对这些问题在 **developer branch** 上频繁发生表示担忧。

---

### **Torchtune ▷ #**[**general**](https://discord.com/channels/1216353675241590815/1216353675744641096/) (1 条消息):

whynot9753: 更新：明天我们可能会收到来自 PyTorch 团队的 DCP PR 🙂

---

### **AI21 Labs (Jamba) ▷ #**[**general-chat**](https://discord.com/channels/874538902696914944/874538902696914947/) (1 条消息):

ag8701347: 请允许我们继续使用我们的 fine-tuned 模型。

---

---

---

---

{% else %}

> 完整的各频道详情已针对邮件进行了删减。
> 
> 如果您想查看完整详情，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
> 
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}