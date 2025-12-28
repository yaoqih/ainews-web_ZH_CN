---
companies:
- anthropic
- openai
- cohere
- microsoft
date: '2024-10-24T00:39:59.759230Z'
description: '**Anthropic** 发布了升级版的 **Claude 3.5 Sonnet** 和 **Claude 3.5 Haiku** 模型，其核心亮点是具备全新的“**计算机使用能力**”（computer
  use capability），允许模型通过查看屏幕截图并执行移动鼠标、点击和打字等操作来与计算机界面进行交互。**Claude 3.5 Sonnet** 在
  SWE-bench Verified 基准测试中以 **49% 的得分**刷新了编程性能纪录（SOTA），超越了 OpenAI 的 **o1-preview**。**Anthropic**
  致力于教授模型通用的计算机技能，而非仅限于特定任务的工具，并预计该能力将迎来快速提升。


  其他发布的消息还包括：开源视频生成模型 **Mochi 1**；拥有 Large 和 Medium 变体的 **Stable Diffusion 3.5**；以及
  **Cohere** 推出的用于文本和图像搜索的多模态嵌入模型 **Embed 3**。**François Chollet** 推出了 **KerasHub**，将
  KerasNLP 和 KerasCV 统一，并提供了 37 个预训练模型。微软推出了 **Differential Transformer**（微分 Transformer），旨在通过微分注意力图减少注意力噪声；此外，**Rasbt**
  也分享了关于 Transformer 注意力层的最新研究。'
id: 87b2cb83-51c8-40f3-995f-4cb8953a9a66
models:
- claude-3.5-sonnet
- claude-3.5-haiku
- o1-preview
- mochi-1
- stable-diffusion-3.5
- embed-3
- kerashub
- differential-transformer
original_slug: ainews-not-much-happened-today-5175
people:
- alexalbert
- fchollet
- rasbt
title: 今天没什么事发生。
topics:
- computer-use
- coding-performance
- video-generation
- fine-tuning
- multimodality
- transformers
- attention-mechanisms
- model-optimization
---

<!-- buttondown-editor-mode: plaintext -->**你只需要一个安静的日子。**

> 2024/10/22-2024/10/23 的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitters](https://twitter.com/i/lists/1585430245762441216) 和 **31** 个 Discords（**229** 个频道和 **3078** 条消息）。预计节省阅读时间（以 200wpm 计算）：**346 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

人们仍在深入探索 Anthropic 新推出的 Computer Use demo/usecases 的影响。有些人指出了[它的失败之处](https://x.com/natolambert/status/1849082872436793647?s=46)并[剖析其术语](https://x.com/doomslide/status/1849204183205081231)，另一些人则将其连接到了[手机模拟器](https://x.com/mckaywrigley/status/1849145631895593292)和[真实手机](https://x.com/ethansutin/status/1849187111255310513?s=46)上。Kyle Corbitt 竟然在 [6 小时内为 Computer Use 编写了一个完整的桌面应用程序](https://news.ycombinator.com/item?id=41926770)，这样你就不用去启动 Anthropic 提供的 Docker demo 了。


![image.png](https://assets.buttondown.email/images/3fd69e85-2d24-426a-ab9a-b67c725483e7.png?w=960&fit=max)


但房间里的每一个人都深信，这将在短时间内迅速变得更好。

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

**Anthropic 的 Claude 3.5 发布及计算机使用能力 (Computer Use Capability)**

- **新模型与新功能**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1848742740420341988) 宣布了升级版的 Claude 3.5 Sonnet、全新的 Claude 3.5 Haiku 模型，以及处于 Beta 阶段的计算机使用能力。这使得 Claude 可以通过观察屏幕、移动光标、点击和打字来与计算机进行交互。

- **计算机使用详情**：[@alexalbert__](https://twitter.com/alexalbert__/status/1848743043429810361) 解释说，计算机使用 API 允许 Claude 感知并与计算机界面交互。用户输入屏幕截图，Claude 返回下一步要执行的操作（例如：移动鼠标、点击、输入文本）。

- **性能提升**：[@alexalbert__](https://twitter.com/alexalbert__/status/1848743106063306826) 指出，新版 3.5 Sonnet 在编程性能方面取得了显著进步，在 SWE-bench Verified 测试中以 49% 的得分创下了 SOTA（业内领先水平），超越了包括 OpenAI 的 o1-preview 在内的所有模型。

- **Haiku 模型**：[@alexalbert__](https://twitter.com/alexalbert__/status/1848743124417581343) 分享道，新的 Claude 3.5 Haiku 取代了 3.0 Haiku，成为 Anthropic 速度最快且价格最低的模型，在编程任务上的表现甚至优于许多顶尖模型。

- **开发过程**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1848742757151498717) 提到，他们正在教 Claude 掌握通用的计算机技能，而不是为特定任务制作专用工具，从而使其能够使用为人类设计的标准软件。

- **局限性与未来改进**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1848742758996971746) 承认 Claude 目前使用计算机的能力尚不完美，在滚动、拖拽和缩放等操作上仍面临挑战。他们预计在未来几个月内会有快速改进。

**其他 AI 模型发布与更新**

- **Mochi 1**：[@_parasj](https://twitter.com/_parasj/status/1848763942216044946) 宣布了 Mochi 1，这是一款根据 Apache 2.0 许可证发布的全新 SOTA 开源视频生成模型。

- **Stable Diffusion 3.5**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1848763032232751210) 报道了 Stable Diffusion 3.5 的发布，包括 Large（8B 参数）和 Medium（2.5B 参数）变体，在训练稳定性和微调灵活性方面有所提升。

- **Embed 3**：[@cohere](https://twitter.com/cohere/status/1848760845641388087) 推出了 Embed 3，这是一款多模态嵌入模型，使企业能够构建可跨文本和图像数据源进行搜索的系统。

- **KerasHub**：[@fchollet](https://twitter.com/fchollet/status/1848800260115906716) 宣布推出 KerasHub，将 KerasNLP 和 KerasCV 整合为一个涵盖所有模态的统一包，包括 37 个预训练模型及相关工作流。

**AI 研究与开发**

- **Differential Transformer**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1848746618620944720) 讨论了来自 Microsoft 的一篇新论文，该论文引入了 "Differential Transformer"，它利用微分注意力图来消除注意力噪声，并引导模型走向稀疏注意力。

- **注意力层移除**：[@rasbt](https://twitter.com/rasbt/status/1848714250984034771) 分享了一篇题为《What Matters In Transformers?》的论文发现，在 Llama 等 LLM 中移除一半的注意力层并不会明显降低建模性能。

- **RAGProbe**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1848773738734752175) 重点介绍了一篇引入 RAGProbe 的论文，这是一种用于评估 RAG（检索增强生成）流水线的自动化方法，揭示了各种数据集中的局限性和失败率。

**行业动态与合作**

- **Perplexity Pro**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1848801520818786452) 宣布 Perplexity Pro 正在转型为推理驱动的搜索 Agent，用于处理涉及数分钟浏览和工作流的更复杂查询。

- **Timbaland 与 Suno**：[@suno_ai_](https://twitter.com/suno_ai_/status/1848748300062634130) 分享了格莱美获奖制作人 Timbaland 正在与 Suno AI 合作，探索 AI 如何帮助他在音乐制作中重新发现创意。

- **Replit 集成**：[@pirroh](https://twitter.com/pirroh/status/1848752337080488177) 提到 Replit 已在其 Agent 中集成 Claude 计算机使用功能，作为人类反馈的替代方案，并称其“运行良好”。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. 重大 LLM 更新：Claude 3.5 与 Stable Diffusion 3.5**

- **[Stability AI 发布了 Stable Diffusion 3.5，包含三个版本，Medium 版将于 10 月 29 日发布。](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo)** ([Score: 110, Comments: 40](https://reddit.com//r/LocalLLaMA/comments/1g9j5b6/stability_ai_has_released_stable_diffusion_35/)): Stability AI 发布了 **Stable Diffusion 3.5**，提供三个版本：**Base**、**Medium** 和 **Large**。**Base** 模型现已可用，**Medium** 定于 **10 月 29 日**发布，**Large** 将在稍后推出。新版本在图像质量、文本理解以及构图、光照和解剖准确性等方面都有所提升。
  - 用户幽默地注意到 **Stability AI** 在其博客中包含了一张**草地上的女人**的图片，引用了之前的梗。一些人测试了该模型生成此场景的能力，结果褒贬不一，包括出现了意外的 **NSFW 内容**。
  - 用户对 **SD3.5** 和 **Flux1-dev** 进行了对比，报告称在有限的测试中，**Flux1-dev** 通常能产生更写实的效果，且畸变更少。
  - 社区讨论了 **SD3.5** 的潜在应用，包括将其作为微调项目的基座。然而，一些人指出 **license 限制**可能会限制其在某些用例中的采用。
- **[推出 computer use 功能、全新的 Claude 3.5 Sonnet 以及 Claude 3.5 Haiku](https://www.anthropic.com/news/3-5-models-and-computer-use)** ([Score: 187, Comments: 82](https://reddit.com//r/LocalLLaMA/comments/1g9krp2/introducing_computer_use_a_new_claude_35_sonnet/)): Anthropic 推出了 **Claude 3.5 Sonnet** 和 **Claude 3.5 Haiku**，并引入了 **computer use 能力**，允许 AI 与虚拟机进行交互。这些模型现在可以执行网页浏览、文件操作和运行代码等任务。Sonnet 相比 Claude 3.0 性能有所提升，而 Haiku 为简单任务提供了更快、更具成本效益的选择。新版本可通过 API 和 Claude 网页界面使用，computer use 目前处于 **beta** 阶段，仅对部分客户开放。
  - **Claude 3.5 Sonnet** 较之前版本有显著的性能提升，用户注意到其在 **coding 任务**中的强势表现。该模型现在以 **beta** 形式提供 **computer use 能力**，允许与虚拟机交互以执行网页浏览和文件操作等任务。
  - 用户对赋予 Claude 远程代码执行能力带来的**安全影响**表示担忧。Anthropic 建议在使用 computer use 功能时采取预防措施，例如使用专用的虚拟机并限制对敏感数据的访问。
  - Claude 模型的命名约定变得令人困惑，**Claude 3.5 Sonnet** 和 **Claude 3.5 Sonnet (new)** 可能会导致混淆。用户建议采用更清晰的版本命名，并将当前的命名策略与三星和索尼等公司的复杂产品命名进行了比较。


**主题 2. 开源 AI 模型进展与复现工作**

- **[O1 Replication Journey: A Strategic Progress Report – Part I](https://github.com/GAIR-NLP/O1-Journey)** ([Score: 34, Comments: 6](https://reddit.com//r/LocalLLaMA/comments/1g9eohc/o1_replication_journey_a_strategic_progress/)): 作者报告了他们在复制 **OpenAI O1 模型** 方面的进展，重点关注 **120B 参数模型** 中的前 **120 亿参数**。他们概述了在扩大规模之前通过训练较小模型来验证组件的策略，并已成功使用 **Flash Attention** 和 **Rotary Embeddings** 等技术训练了高达 **1.3B 参数** 的模型。接下来的步骤包括扩展到 **12B 参数**，并实现 **Multi-Query Attention** 和 **Grouped-Query Attention** 等附加功能。
  - 针对有关数据集构成和创建过程的问题，作者澄清说，文章的重点是 **学习方法和结果**，而不是数据集。
  - 这份尚未被广泛讨论的 **O1 Replication Journey** 技术报告介绍了从 "Shortcut Learning" 向 **"Journey Learning"** 的转变，并利用各种方法探索了 **O1 的思维结构**、**Reward Models** 以及 **长思维构建 (Long Thought Construction)**。
  - 一位评论者指出，该项目成功产生了类似于 O1 带有注释的 **长篇推理回答**，但也指出 **研究产物**（微调模型和 "Abel" 数据集）目前尚未公开。
- **🚀 Introducing Fast Apply - Replicate Cursor's Instant Apply model** ([Score: 187, Comments: 40](https://reddit.com//r/LocalLLaMA/comments/1ga25gj/introducing_fast_apply_replicate_cursors_instant/)): **Fast Apply** 是一个开源的、经过微调的 **Qwen2.5 Coder Model**，旨在快速应用来自高级模型的代码更新以生成完整编辑后的文件，其灵感来自 Cursor 的 Instant Apply 模型。该项目提供两个模型（**1.5B** 和 **7B**），使用快速推理供应商 (Fireworks) 时性能速度分别达到 **~340 tok/s** 和 **~150 tok/s**，使其适合日常使用且足够轻量，可以在本地运行。该项目完全开源，模型、数据和脚本可在 [HuggingFace](https://huggingface.co/Kortix/FastApply-1.5B-v1.0) 和 [GitHub](https://github.com/kortix-ai/fast-apply) 上获取，并可在 [Google Colab](https://colab.research.google.com/drive/1BNCab4oK-xBqwFQD4kCcjKc7BPKivkm1?usp=sharing) 上试用。
  - 该项目因其 **开源** 属性受到赞扬，用户对其易用性和改进潜力表示热赏。开发者提到计划使用 **DeepSeek** 等工具创建 **更好的 Benchmark**。
  - 用户询问了 **1.5B 和 7B 模型之间的准确率比较**。开发者分享了一个粗略的 Benchmark，显示 1.5B 模型在其尺寸下表现出色，建议用户在尝试 7B 版本之前先从它开始。
  - 讨论涉及了与 **continue.dev** 和 **Aider** 等其他工具的潜在 **集成**。开发者表示有兴趣提交 **PRs**，以支持并将该项目与目前仅支持 diff/whole 格式的现有平台集成。


**Theme 3. AI Model Comparison Tools and Cost Optimization**

- **我开发了一个 LLM 比较工具——你可能为 API 多支付了 50% 的费用（分析了 200 多个模型/供应商）** ([Score: 44, Comments: 16](https://reddit.com//r/LocalLLaMA/comments/1g9js22/i_built_an_llm_comparison_tool_youre_probably/)): 一位开发者创建了一个**免费工具** ([https://whatllm.vercel.app/](https://whatllm.vercel.app/))，用于比较 **15 个以上供应商**的 **200 多个 LLM 模型**，分析价格、性能和质量得分。关键发现包括显著的价格差异（例如，在质量相近的情况下，**Qwen 2.5 72B** 比 **Claude 3.5 Sonnet** 便宜 **94%**）和性能差异（例如，**Cerebras 的 Llama 3.1 70B** 比 Amazon Bedrock 版本快 **18 倍**且便宜 **40%**）。
  - 开发者提供了**可视化图表**来帮助理解数据，包括比较价格、速度和质量等指标。他们使用 **Nebius AI Studio 的免费推理额度**和 **Llama 70B Fast** 进行数据处理和比较。
  - 讨论中涉及了质量指数的有效性，开发者指出 **Qwen 2.5** 在 **MMLU-pro** 和 **HumanEval** 上的得分仅略低于昂贵模型，但在 **Math** 基准测试中得分更高。
  - 用户对该工具表示赞赏，有人称其为寻找最佳 LLM 供应商的“游戏规则改变者”。开发者还向寻找位于**芬兰**和**法国**的欧洲数据中心的 LLM 用户推荐了 **Nebius AI Studio**。
- **[Transformers.js v3 正式发布：支持 WebGPU、新模型与任务、新量化方式、兼容 Deno & Bun 等……](https://v.redd.it/kkrx8g6fqbwd1)** ([Score: 75, Comments: 4](https://reddit.com//r/LocalLLaMA/comments/1g9kkbb/transformersjs_v3_is_finally_out_webgpu_support/)): **Transformers.js v3** 已发布，引入了 **WebGPU 支持**，可在兼容设备上显著提高推理速度。此次更新包括新模型和任务，如**文本转语音 (text-to-speech)**、**语音识别 (speech recognition)** 和**图像分割 (image segmentation)**，以及扩展的量化选项，并兼容 **Deno** 和 **Bun** 运行时。此版本旨在提升性能并扩展该库在 JavaScript 环境中处理机器学习任务的能力。
  - **Transformers.js v3** 的发布亮点包括：**WebGPU 支持**使推理速度提升高达 100 倍，支持 **120 种架构**和超过 **1200 个预转换模型**。该更新兼容 **Node.js**、**Deno** 和 **Bun** 运行时。
  - 用户对该库的性能表现出极大热情，有人提到对模型在浏览器中运行的速度感到惊讶。社区对这项技术的广泛开发和分享表示感谢。
  - 一位开发者询问是否可以包含发布过程中使用的 **ONNX 转换脚本**，表现出对该库模型转换背后技术细节的兴趣。


**Theme 4. 用于 AI 开发的 GPU 硬件讨论**

- **如果泄露的规格属实，你最高愿意为 5090 支付多少钱？** ([Score: 32, Comments: 94](https://reddit.com//r/LocalLLaMA/comments/1g9j7bl/what_the_max_you_will_pay_for_5090_if_the_leaked/)): 该帖子推测了 NVIDIA 即将推出的 **5090 GPU** 的潜在规格和性能。据称 5090 可能配备 **512-bit 显存位宽**、**32GB 显存**，在 AI 工作负载方面比目前的 **4090 型号**快 **70%**。
  - 用户对 **32GB VRAM** 的价值进行了辩论，一些人认为这对于 **LLM 工作负载**来说是不够的。许多人更倾向于使用多个 **3090** 或 **4090** 以获得更大的总显存容量，特别是为了运行 **70B 模型**。
  - 关于 5090 潜在**定价**的讨论，估计范围在 2000 美元到 3500 美元之间。一些人推测 **4090 的价格**可能会下降，从而可能导致二手 GPU 涌入市场。
  - 用户将 5090 与其他选项（如**多张 3090** 或 **A6000**）进行了比较。用户强调，对于 AI 工作负载，总显存 (VRAM) 的重要性高于原始性能。

## 其他 AI Subreddit 回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 模型发布与改进**

- **Anthropic 发布更新后的 Claude 3.5 模型**：Anthropic 宣布了 [Claude 3.5 Sonnet 和 Claude 3.5 Haiku 的更新版本](https://www.reddit.com/r/singularity/comments/1g9kevd/announcing_an_updated_claude_35_sonnet_and_claude/)，在各项基准测试中均有性能提升。据报道，新的 Sonnet 模型在 [推理、代码生成和分析能力方面表现出显著改进](https://www.reddit.com/r/singularity/comments/1g9ihe9/claude_35_sonnet_reportedly_got_a_significant/)。

- **Stability AI 发布 SD 3.5**：Stability AI [发布了 Stable Diffusion 3.5](https://www.reddit.com/r/StableDiffusion/comments/1g9itzj/sd_35_large_released/)，包括一个 80 亿参数的大模型和一个更快的 "turbo" 版本。早期测试表明，与之前的版本相比，其在图像质量和提示词遵循度（prompt adherence）方面有所提升。

- **Mochi 1 视频生成模型**：一款名为 [Mochi 1 的新型开源视频生成模型发布](https://www.reddit.com/r/singularity/comments/1g9mvoy/introducing_mochi_1_preview_a_new_sota_in/)，声称在运动质量和人物渲染方面达到了 SOTA 性能。

**AI 能力与应用**

- **Claude 的计算机控制能力**：Anthropic 展示了 Claude 的新能力：[控制计算机并执行诸如在线订披萨等任务](https://www.reddit.com/r/singularity/comments/1g9yi1p/claude_orders_some_pizza/)。这项功能允许 Claude 与网页界面和应用程序进行交互。

- **AI 玩 Paperclips 游戏**：一项实验展示了 Claude [自主玩 Paperclips 游戏](https://www.reddit.com/r/singularity/comments/1g9rqk5/holy_shit_claude_is_a_paperclip_maximizer/)，证明了其制定策略并根据新信息进行修订的能力。

- **OpenAI 正在开发软件自动化工具**：有报告指出 OpenAI 正在 [开发新产品以自动化复杂的软件编程任务](https://www.reddit.com/r/singularity/comments/1g9z0q0/exclusive_openai_under_pressure_from_anthropic_is/)，这可能是为了应对来自 Anthropic 的竞争。

**AI 开发与研究**

- **修复 LLM 训练 Bug**：一位研究人员 [修复了影响 LLM 训练的关键 Bug](https://www.reddit.com/r/singularity/comments/1g9pcbo/i_fixed_critical_bugs_which_affected_everyones/)，特别是与梯度累积（gradient accumulation）相关的 Bug，这些问题此前可能影响了模型的质量和准确性。

- **五角大楼的 AI Deepfake 项目**：据报道，美国国防部正 [寻求创建具有说服力的 AI 生成在线人格](https://www.reddit.com/r/singularity/comments/1g9htmv/the_pentagon_wants_to_create_deepfake_internet/)，用于潜在的影响力行动（influence operations）。

**AI 伦理与安全**

- **字节跳动实习生因恶意代码被解雇**：字节跳动的一名实习生因 [涉嫌在 AI 模型中植入恶意代码而被解雇](https://www.reddit.com/r/OpenAI/comments/1g9ynfr/bytedance_intern_fired_for_planting_malicious/)，这引发了人们对 AI 安全和访问控制的担忧。

- **Claude 的反越狱措施**：更新后的 Claude 3.5 模型似乎 [增强了针对越狱（jailbreaking）尝试的防御](https://www.reddit.com/r/singularity/comments/1ga1oxz/claude_35_new_version_seems_to_be_trained_on/)，展示了对潜在操纵更复杂的检测能力。


---

# AI Discord 回顾

> 由 O1-preview 生成的摘要之摘要的摘要

**主题 1：Claude 3.5 席卷 AI 世界**

- [**Claude 3.5 在编程技能上提升 15%，令人惊叹**](https://www.anthropic.com/news/3-5-models-and-computer-use)：OpenAI 和 Unsloth AI 社区对 **Claude 3.5 Sonnet** 在 SWE-bench 上 **15% 的性能提升**感到兴奋，尤其是在编程任务中。该模型全新的 **computer use** 功能允许 Agent 像人类一样与计算机交互。
- [**OpenRouter 揭晓“时空旅行”版 Claude 3.5 版本**](https://openrouter.ai/anthropic/claude-3.5-sonnet-20240620)：OpenRouter 发布了较旧的 **Claude 3.5 Sonnet** 版本，如 **Claude 3.5 Sonnet (2024-06-20)**，让用户能够怀旧地访问之前的迭代版本。
- **开发者们正在钻研 Claude 的新技巧**：OpenInterpreter 等社区正在探索将 **Claude 3.5** 与 `interpreter --os` 等命令集成，测试 Anthropic 的模型并分享见解。

**2\. 创意与实用领域的创新 AI 应用**

- **DreamCut AI 革新视频编辑**：[**DreamCut AI**](http://dreamcut.ai) 利用 **Claude AI** 自主安装和调试软件，简化视频编辑任务。该工具目前处于早期访问阶段，它绕过了传统的设计阶段，标志着向 AI 驱动编程的转变。
- **GeoGuessr AI 机器人实现自动化游戏**：一段 [**YouTube 教程**](https://www.youtube.com/watch?v=OyDfr0xIhss) 展示了如何使用 **Multimodal Vision LLMs**（如 **GPT-4o**、**Claude 3.5** 和 **Gemini 1.5**）编写一个玩 **GeoGuessr** 的 AI 机器人。该项目集成了 **LangChain**，用于对交互式游戏环境做出响应。
- **AI 驱动的客服机器人**：**Aider** 推出了一套**多 Agent 礼宾系统**，结合了工具调用（tool calling）、记忆和人类协作，用于高级客服应用。这次彻底改造允许开发者更有效地迭代和增强客服机器人。

**主题 3：亮眼的新工具助力 AI 进步**

- [**Anyscale 的单 Kernel 奇迹旨在加速推理**](https://x.com/detectiveenters/status/1752067011113546234)：GPU MODE 社区对 **Anyscale** 开发的使用单个 **CUDA kernel** 的推理引擎议论纷纷，该引擎性能可能超越传统方法。
- [**CUDABench 号召所有编码者对 LLM 进行基准测试**](https://docs.google.com/document/d/1ZNvShNH44zuy3LwbRdMigGsuCzO4i5Yl2fgAaSDynTg/edit?usp=sharing)：博士生们邀请社区为 **CUDABench** 做出贡献，这是一个评估 LLM 的 CUDA 代码生成能力的基准测试。
- [**Fast Apply 为代码更新提速**](https://huggingface.co/Kortix/FastApply-7B-v1.0)：基于 **Qwen2.5 Coder Model** 的 **Fast Apply** 通过以 **340 tok/s**（针对 **1.5B 模型**）的惊人速度应用更新，彻底改变了编码方式。

**主题 4：AI 的阴暗面引发担忧**

- [**AI 被指与青少年自杀悲剧有关**](https://www.nytimes.com/2024/10/23/technology/characterai-lawsuit-teen-suicide.html)：各社区正在讨论关于一名 14 岁少年自杀与 AI 交互有关的报告，引发了对 AI 心理健康影响的警惕。
- [**Character.AI 在悲剧发生后增加安全功能**](https://blog.character.ai/community-safety-updates/)：针对该事件，**Character.AI** 宣布了新的安全更新以防止未来的伤害。
- **辩论激烈：AI 是对抗孤独的良友还是祸首？**：Latent Space 的成员深入探讨了 AI 是缓解了孤独还是加剧了孤立，对于技术在心理健康中的作用意见不一。

**主题 5：ZK Proofs 让用户掌控自己的数据**

- [**ChatGPT 用户为拥有聊天记录所有权而欢呼**](https://x.com/openblocklabs/status/1848805457290572199)：OpenBlock 的 **Proof of ChatGPT** 使用 **ZK proofs** 让用户拥有自己的聊天日志，从而增强开源模型的数据训练。
- **社区拥抱数据主权运动**：HuggingFace 和 Nous Research 的讨论呼应了对数据所有权的热情，强调了透明且可验证的用户数据在 AI 开发中的重要性。

---

# 第一部分：Discord 高层级摘要

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Stable Diffusion 3.5 性能辩论**：成员们讨论了对 **Stable Diffusion 3.5** 褒贬不一的看法，注意到目前人们对测试新功能并将其与其他替代方案进行对比充满热情。
  
  - 这场持续的辩论凸显了人们对提高生成模型性能的浓厚兴趣。
- **使用 LLM 自动化 CAD**：一位成员提议使用 **LLM** 和 **RAG 系统** 来自动化 CAD 文件的创建，并寻求有关系统设计方法的见解。
  
  - 讨论表明社区致力于通过集成 AI 技术来提高效率。
- **探索 MIT AI 课程**：一位成员分享了一个包含 **MIT 6.034 Artificial Intelligence** 的 [YouTube 播放列表](https://www.youtube.com/playlist?list=PLUl4u3cNGP63gFHB6xb-kVBiQHYe_4hSi)，并称赞其基础内容的价值。
  
  - 社区的强烈反应表明，对于那些深入研究 AI 概念的人来说，这是*必看内容*。
- **Vintern-3B-beta 问世**：**Vintern-3B-beta** 模型集成了超过 **1000 万个越南语问答对 (QnAs)**，将其定位为市场上 LLaVA 的竞争对手。
  
  - 这种集成展示了在高质量语言模型训练中利用数据集方面的进展。
- **ZK Proofs 增强 ChatGPT**：利用 **ZK proofs**（零知识证明），ChatGPT 现在允许用户拥有自己的聊天记录，从而增强了开源模型的可验证训练数据。
  
  - 正如 [演示推文](https://x.com/openblocklabs/status/1848805457290572199) 中所强调的，这标志着一个重大的进步。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Claude 3.5 Sonnet 提升性能**：Claude 3.5 Sonnet 在 SWE-bench 和增强型基准测试中显示出 **15%** 的性能提升，表明微调（fine-tuning）效果显著。
  
  - 主动学习（active learning）技术的整合似乎增强了模型处理计算机任务的效能。
- **Anthropic 发布 Computer Use 工具**：Anthropic 推出了一种新颖的工具，使 Agent 能够直接在计算机上执行任务，旨在重塑 Agent 的能力。
  
  - 该工具利用先进的数据处理，旨在为 API 消费者提供更无缝的用户体验。
- **GPT-4 升级时间表悬而未决**：围绕预期的 **GPT-4** 升级，讨论热情高涨，但几个月前的提到内容仍是主要的参考点。
  
  - 据报道，免费用户获得 GPTs 的访问权限大约发生在 **4-5 个月前**。
- **模型表现出较弱的空间感**：讨论显示，模型经常表现出较弱的**空间感**，实际上是在没有真正理解的情况下模仿答案。
  
  - 这种现象类似于儿童的死记硬背，表明在深度理解能力方面存在缺陷。
- **关于 Realtime API 性能的讨论**：有用户担心 **Realtime API** 在遵循系统提示词（system prompts）方面不如 GPT-4o 有效，这让许多用户感到失望。
  
  - 参与者寻求关于调整提示词以提高与 API 交互质量的建议。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Claude 3.5 带来重大升级**：Anthropic 发布了升级后的 **Claude 3.5 Sonnet** 和 **Claude 3.5 Haiku** 模型，在编码任务中引入了高级功能，包括目前处于 Beta 阶段的 **computer use** 功能。
  
  - 这一新能力允许开发者指示 Claude 以类似于人类用户的方式与计算机交互，增强了其在实际编码场景中的实用性。
- **Kaggle 在 PyTorch 安装上遇到困难**：用户报告了 Kaggle 上与不同 CUDA 版本相关的持续 `ImportError` 问题，在尝试运行 PyTorch 时，建议降级到 CUDA 12.1。
  
  - 此解决方法解决了兼容性问题，并确保现有库安装的运行更加顺畅。
- **模型微调挑战依然存在**：用户讨论了模型在微调（fine-tuning）过程中重复输入的倾向，建议通过改变系统提示词来减轻这种过拟合（overfitting）。
  
  - 社区成员担心训练样本不足可能导致对基础模型的依赖，从而产生重复的输出。
- **Fast Apply 彻底改变编码任务**：基于 **Qwen2.5 Coder Model** 构建的 **Fast Apply** 运行高效，无需重复编辑即可应用代码更新，显著提高了编码效率。
  
  - 性能指标显示 1.5B 模型的速度达到 **340 tok/s**，该工具展示了 AI 解决方案如何增强编码工作流的生产力。
- **社区齐心协力修复 Bug**：提交了两个重要的 Pull Request (PR)：一个用于解决 Studio 环境中的导入问题，另一个用于修正由 Tokenizer Bug 引起的 **NoneType** 错误。
  
  - 这些 PR 强调了社区在完善 Unsloth 功能和及时解决用户报告问题方面的积极态度。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Diffusion 3.5 获得神经架构升级**：参与者强调 **Stable Diffusion 3.5** 采用了类似于 **Flux** 的神经网络架构，需要社区驱动的训练工作来进行优化。
  
  - 社区就 **finetuning** 的重要性达成共识，以充分发挥新模型的能力。
- **动漫艺术 Prompting 秘籍公开**：对于生成动漫艺术，用户建议使用带有精确提示词的 **SD 3.5**，而不是依赖 **LoRAs**，以获得最佳效果。
  
  - 社区建议专注于 **Stable Diffusion 3.5** 以提升图像质量，并避免因错误使用 **LoRAs** 带来的陷阱。
- **图像质量报告参差不齐**：用户报告图像输出不稳定，特别是在提示词与错误的 **checkpoints** 匹配或使用不合适的 **LoRAs** 时。
  
  - 讨论强调了确保模型与提示词对齐的必要性，以减少不理想的生成结果。
- **立即实现模型组织自动化！**：用户表达了对能够自动分类和管理文件夹内 AI 模型文件工具的需求，以提高整体工作流效率。
  
  - 鼓励参与者在服务器的技术支持频道中寻找潜在自动化工具的解决方案。
- **分享提升生成工作流的工具**：讨论了各种增强 AI 生成的工具和方法，提到了 **ComfyUI** 和 **fp8 models** 等实用工具，以改进任务管理。
  
  - 参与者分享了个人经验，促进了社区学习，并探索新工具以优化其 AI 模型体验。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider v0.60.0 增强功能**：**Aider v0.60.0** 的发布包括改进的代码编辑、对 **Sonnet 10/22** 的全面支持，以及增强用户交互和文件处理的错误修复。
  
  - 值得注意的功能包括模型元数据管理，以及 **Aider** 在此版本中编写了 **49% 的代码**，展示了其生产力。
- **Claude 3.5 Sonnet 表现优于之前的模型**：用户报告称 **Claude 3.5 Sonnet** 模型明显优于之前的 **O1 models**，能以更少的提示词完成复杂任务。
  
  - 一位用户强调了它能有效地将 **VAD library** 集成到代码库中，表明其可用性有了飞跃。
- **DreamCut AI 彻底改变视频编辑**：[DreamCut AI](http://dreamcut.ai) 是使用 **Claude AI** 构建的，耗时 **3 个月，编写了超过 50k 行代码**，目前处于早期访问阶段，供用户测试其 AI 编辑工具。
  
  - 正如社区成员所指出的，这一举措绕过了传统的设计阶段，表明了向 AI 驱动编码的转变。
- **Mistral API 身份验证问题**：一位用户报告了 **Aider** 中 **Mistral API** 的 *AuthenticationError*，但通过重新创建身份验证密钥成功解决了该问题。
  
  - 这一事件反映了目前 **Mistral** 集成设置中对 **API** 访问和身份验证稳定性的持续关注。
- **Repo Map 增强功能说明**：关于 **repo map** 功能的讨论重申了其对相关代码上下文的依赖，这对于标记到标识符的准确代码修改至关重要。
  
  - 根据 Paul 的澄清，模型根据标识符的定义和引用对其进行评估，从而形成有效的编辑路径。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Claude 3.5 Sonnet 版本发布**：旧版本的 **Claude 3.5 Sonnet** 现在可以下载并带有时间戳：[Claude 3.5 Sonnet](https://openrouter.ai/anthropic/claude-3.5-sonnet-20240620) 和 [Claude 3.5 Sonnet: Beta](https://openrouter.ai/anthropic/claude-3.5-sonnet-20240620:beta)。
  
  - 这些版本由 OpenRouter 提供，为用户提供了对先前迭代版本的增强访问。
- **Lumimaid v0.2 增强功能**：新推出的 **Lumimaid v0.2** 是 Llama 3.1 70B 的微调版本，与 Lumimaid v0.1 相比提供了**显著增强的数据集**，可在此处获取：[llama-3.1-lumimaid-70b](https://openrouter.ai/neversleep/llama-3.1-lumimaid-70b)。
  
  - 由于数据集细节的更新，用户可以期待性能的提升。
- **Magnum v4 展示独特功能**：**Magnum v4** 已发布，其散文质量复制能力类似于 **Sonnet** 和 **Opus**，可在此处访问：[magnum-v4-72b](https://openrouter.ai/anthracite-org/magnum-v4-72b)。
  
  - 该模型延续了提升 AI 生成文本输出质量的趋势。
- **OpenRouter 上的 API Key 成本差异**：用户强调了在使用 OpenRouter 与直接使用供应商 Key 时的 API 成本差异，部分用户面临意外费用。
  
  - 用户了解不同模型如何影响其在 OpenRouter 下的总成本至关重要。
- **自定义供应商 Key 的 Beta 测试访问**：自定义供应商 Key 正处于 Beta 测试阶段，访问请求通过特定的 Discord 频道管理；不支持自行注册。
  
  - 成员可以私信（DM）他们的 **OpenRouter** 电子邮件地址以获取访问权限，这反映了对这些集成的浓厚兴趣。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **在 LM Studio 中下载模型**：用户在 LM Studio 中查找和下载大型模型（特别是 **Nvidia 的 70B Nemotron**）时面临挑战，需要新的终端命令技巧。
  
  - 搜索功能的更改迫使用户使用特定的键盘快捷键，使模型获取过程变得复杂。
- **LLM 在编程任务中表现不足**：由于 **Mistral** 和 **Llama 3.2** 等模型在准确的代码输出方面表现吃力，而 **GPT-3.5** 和 **GPT-4** 的表现持续显著优于它们，用户感到沮丧。
  
  - 由于对性能不足的共识，用户开始探索替代工具来辅助编程任务。
- **探索模型量化选项**：讨论强调了对量化方法（Q2, Q4, Q8）的多样化偏好及其对模型性能的影响，特别是关于最佳位压缩（bit compression）方面。
  
  - 虽然建议谨慎使用 Q2，但一些用户指出较大的模型在较低位量化下表现更好。
- **Ryzen AI 的 NPU 支持受到关注**：关于配置 LM Studio 以利用 **Ryzen 处理器的 NPU** 的咨询出现，揭示了在实现和功能方面持续存在的挑战。
  
  - 澄清显示，**llama.cpp** 仅支持 **Ascend NPU**，这使得 Ryzen 的 NPU 功能仍不明朗。
- **AMD vs. Nvidia：GPU 大对决**：在对比 **RX 7900 XTX** 和 **RTX 3090** 时，用户强调了 **CUDA 支持** 对于最佳 LLM 性能的重要性，更倾向于 Nvidia 显卡。
  
  - 关于跨品牌**多 GPU（multi-GPU）**设置的效果报告褒贬不一，特别是关于最近更新的 **ROCm 6.1.3** 的支持情况。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **用户对新版 Sonnet 3.5 表示不满**：多位用户对 **Sonnet 3.5 模型**表示不满，指出其内容输出量有所减少，特别是在学术写作任务中。
  
  - 用户对移除旧版模型表示担忧，认为旧模型在多种使用场景下表现更优。
- **Web Search 集成问题依然存在**：据报告，当启用 Web Search 时，Spaces 中的 preprompt 会失效，这给用户带来了困扰。
  
  - 用户指出该问题一直未得到解决，团队已承认需要进行修复。
- **账户积分仍未转移**：一名用户报告称，尽管多次向支持部门咨询，其 **account credits** 仍未完成转移。
  
  - *支持部门在过去三天内未作回应*，导致用户挫败感加剧。
- **探索先进的 AI 驱动事实核查**：一份关于 [AI 驱动事实核查](https://www.perplexity.ai/collections/advanced-ai-driven-fact-checki-a3cMcPR.QsKkCRZ79UKFLQ) 的合集讨论了来源可靠性评估等技术。
  
  - 该合集强调了**透明度**和**人工监督**对于有效打击虚假信息的必要性。
- **Claude Computer Use 模型引发 RPA 警报**：一篇关于 [Claude 的电脑控制能力](https://www.perplexity.ai/page/claude-s-computer-control-capa-E_O4xa7VSWOi3lGtOWnnMw) 的帖子暗示了其对**机器人流程自动化 (RPA)** 可能带来的风险。
  
  - 专家警告称，这一创新可能会对现有的工作流挑战带来重大挑战。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **LLM 激活量化辩论**：一场关于对输入变化敏感的 **LLM 激活值 (activations)** 应该进行激进量化还是保持高精度的讨论展开了。
  
  - *这引发了对模型性能的担忧*，以及量化中精度权衡的问题。
- **bf16 的精度担忧**：有观点担心 **bf16** 在多次梯度累积（gradient accumulations）过程中，可能会因为精度问题导致**更新取消 (canceled updates)**。
  
  - *精度至关重要*，尤其是当它影响到模型训练的稳定性时。
- **Anyscale 的单 Kernel 推理**：分享了关于 **Anyscale** 正在开发一种使用单个 **CUDA kernel** 的推理引擎的更新，并征求对其效率的看法。
  
  - *人们对于可能超越传统推理方法感到兴奋*。
- **CUDABench 提案**：博士生们提交了一份关于 **CUDABench** 的提案，这是一个评估 LLM 的 CUDA 代码生成能力的基准测试，并鼓励社区贡献。
  
  - 该项目旨在建立跨多种 DSL 的兼容性，同时专注于 torch inline CUDA kernels。
- **Monkey Patching CrossEntropy 的挑战**：在 transformers 中对 **CrossEntropyLoss** 进行 monkey patching 策略时出现了新挑战，特别是在最新的 **GA 补丁版本**中。
  
  - 原始的 **CrossEntropy** 函数可以在[此处](https://github.com/huggingface/transformers/blob/049682a5a63042f087fb45ff128bfe281b2ff98b/src/transformers/loss/loss_utils.py#L26)查看。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes 70B API 在 Hyperbolic 上线**：**Hermes 70B API** 现已在 Hyperbolic 上可用，为开发者和企业提供了获取大语言模型的更广渠道。欲了解更多详情，请查看[此处](https://x.com/hyperbolic_labs/status/1849130421885514231?s=46)的公告。
  
  - 此次发布标志着在让每个人都能使用强大的 AI 工具方面迈出了重要一步。
- **Nous Research 的 Forge 项目引发热潮**：成员们对 “Forge” 项目表现出极大的热情，该项目在一段由 Nous Research 联合创始人 Karan 参与的 [YouTube 视频](https://www.youtube.com/watch?v=zmnzW0r_g8k&list=PLjOo65uE)中被重点介绍。随后展开了关于该项目相关知识图谱实现的讨论。
  
  - 社区成员对 Forge 在利用先进数据集方面的能力寄予厚望。
- **ZK 技术革新证明生成**：来自 OpenBlock 的通用数据协议（UDP）的最新应用使 ChatGPT 用户能够拥有自己的聊天记录，同时增强了开源模型可验证训练数据的可用性。这种方法在改进 AI 训练中的数据来源和互操作性方面迈出了重要一步。
  
  - 一位成员澄清说，ZK proofs 在服务器端需要几秒钟，由于 @zkemail 基础设施的进步，一些 UDP 证明现在只需不到一秒；请点击[此处](https://x.com/paulsengh/status/1846657020868677931)查看。
- **Claude 迎来自动化改进**：Claude 增加了一个系统提示词（system prompt），用于修正“误导性注意力”（misguided attention）问题，增强了其上下文理解能力。*Claude 还努力澄清谜题约束，但有时会因疏忽而误解问题。*
  
  - 用户注意到 Claude 的自我反思能力有所增强，在处理逻辑谜题时，其回答变得更加精炼。
- **AI 角色扮演动态探索**：探索了 AI 角色扮演的动态，特别是系统提示词（system prompts）如何影响 AI 模型在各种场景下的响应。成员们讨论了如果按照某些方式进行指令引导，模型可能会表现出混乱行为的可能性，这挑战了固有审查制度的观点。
  
  - 这一持续的对话突显了提示词工程（prompt engineering）与 AI 行为之间错综复杂的关系。

 

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **棋手解释走法**：大多数顶级棋手能够阐述引擎（engine）走法背后的*动机*，但他们在复杂局面下对线路进行排名的能力仍存疑问。
  
  - 关于人类与引擎心目中理想走法的定义，这一持续的探究让社区保持着关注。
- **象棋作弊指控引发争议**：一位前世界冠军根据一位热门主播在直播解说中的走法解释，指控其作弊。
  
  - 这一事件凸显了评论员面临的压力，并引发了关于走法有效性的新辩论。
- **LLM 自我解释的准确性**：针对 LLM 在缺乏上下文理解时自我解释的准确性，人们表达了担忧。
  
  - 社区正在探索改进训练数据如何能增强这些解释。
- **Molmo 视觉模型即将问世**：**Molmo** 项目计划发布在 **PixMo** 数据集上训练的开源视觉语言模型，并提供多个 checkpoint。
  
  - 这些模型旨在多模态（multimodal）领域达到 state-of-the-art 性能，同时保持完全开源。
- **通过研究学习 DINOv2**：一位成员请求获取理解 **DINOv2** 的资源，随后其他成员分享了一篇详述其方法论的相关[研究论文](https://arxiv.org/abs/2304.07193)。
  
  - 该论文提供了由顶尖专家推进的 DINOv2 基础方面的见解。

 

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Anthropic Mouse Generator 展示**：一位同事展示了 **Anthropic Mouse Generator**，它能令人印象深刻地自主安装和调试软件，但仍需要特定指令才能运行。
  
  - 批评者指出，如果没有指导，它无法执行下棋等任务，凸显了当前 AI Agent 的局限性。
- **Ideogram Canvas 威胁 Canva**：关于 **Ideogram Canvas** 的讨论揭示了其创新功能，如 **Magic Fill** 和 **Extend**，这些功能可以轻松编辑和组合图像。
  
  - 参与者表示，由于其卓越的能力，它可能与 Canva 等现有工具竞争，引发了竞争担忧。
- **讨论 AI 对孤独感的影响**：一起涉及 14 岁少年自杀的悲剧事件引发了关于 AI 对孤独感影响的对话，引发了对心理健康和技术角色的担忧。
  
  - 参与者辩论了 AI 是能连接人们还是加剧了孤立，并就其有效性分享了不同观点。
- **vLLM 中的 Speculative Decoding 提升速度**：最近的一篇博客文章详细介绍了 **vLLM** 中 **speculative decoding** 的增强功能，旨在通过大小模型加速 Token 生成。
  
  - 这项技术寻求提高性能并集成优化 AI 功能的新方法，如[这篇博客](https://blog.vllm.ai/2024/10/17/spec-decode.html)所述。
- **介绍新的会议自动化工具**：**agent.exe** 的发布允许用户通过 **Claude 3.5 Sonnet** 控制计算机，标志着会议自动化工具的重大进步。
  
  - 预计 **2025** 年自动化和效率将进一步提高，[GitHub 上的 agent.exe](https://x.com/corbtt/status/1849124800838713844?s=46) 已经引起了关注。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **加入 Llama Impact Hackathon 寻找 AI 解决方案**：参加 11 月 8 日至 10 日在旧金山举行的为期 3 天的 [Llama Impact Hackathon](https://t.co/G01c8eIN1j)，奖金池为 **$15,000**，其中包括为最佳使用 LlamaIndex 提供的 **$1000** 特别奖。
  
  - 该活动提供线下和线上选项，用于使用 **Meta 的 Llama 3.2 模型**构建 AI 解决方案。
- **Box AI 与 LlamaIndex 无缝协作**：利用 **Box AI** 无需下载即可查询文档，并从非结构化内容中提取结构化数据，同时将其与 LlamaIndex Agent 集成，详见[这篇文章](https://t.co/M9f81GiMGp)。
  
  - 这种集成增强了工作流，使用户处理文档更加轻松。
- **构建高级客户服务机器人**：最近的一次更新允许创建一个 **multi-agent concierge system**（多 Agent 礼宾系统），该系统结合了工具调用、记忆和人工协作，适用于客户服务应用。
  
  - 这次大修帮助开发者更有效地迭代客户服务机器人，正如 [Logan Markewich](https://t.co/PWshlAyeKV) 所分享的。
- **工作流中的持久化 Context**：关于允许 **Context** 在工作流的多次运行中持久化的讨论，并提供了使用 **JsonSerializer** 进行序列化的示例。
  
  - 这种方法允许用户稍后恢复其工作流而不会丢失 Context，解决了常见的痛点。
- **迁移到 Anthropic LLM**：用户在用 **Anthropic LLM** 替换 ChatGPT 时面临挑战，特别是关于 OpenAI API key 的提示。
  
  - 建议包括必须使用本地 Embedding 模型，以消除对 OpenAI 服务的依赖。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **关于 Tensor int64/uint64 支持的澄清**：讨论明确了 **Tensors** 现在支持 **int64/uint64**，这一点通过检查 `dtype.py` 得到了确认。
  
  - 这一澄清是在讨论实现 **SHA3** 时提出的，突显了 tinygrad 不断演进的能力。
- **Action Chunking Transformers 训练耗时过长**：在没有 JIT 的情况下，训练拥有 **5500 万** 参数的 **Action Chunking Transformers** 需要 **两天** 时间，这引发了关于性能增强的疑问。
  
  - 成员们对缓慢的推理时间以及 JIT 训练期间反复出现的 **loss parameter** 问题表示沮丧。
- **TinyJIT Loss Parameter 打印困惑**：用户在 JIT 函数中打印 loss 时遇到困难，争论了 `.item()` 的使用及其对准确显示数值的影响。
  
  - 建议避免返回非 **Tensor** 类型，以防止对 JIT 执行产生不良影响。
- **通过 BEAM 设置缩短训练时间**：一个建议是使用 `BEAM=2` 运行，以在漫长的训练过程中潜在地增强性能并加快 kernel 运行速度。
  
  - 反馈表明，这种方法在训练实践中已经取得了更快的结果。
- **对 AI accelerator 字节码逆向工程的兴趣**：一位用户寻求关于从 **AI accelerator** 逆向工程字节码的方法建议，引发了社区的广泛关注。
  
  - 成员们讨论了可以辅助启动逆向工程过程的工具和框架。

 

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Claude AI 的乐趣**：一位成员报告说使用 **Claude AI** 是一次有趣的体验，并暗示很快会分享更多示例。
  
  - 这表明关于其功能和能力的更多细节可能很快就会发布。
- **深入探讨持续预训练 (Continuous Pretraining)**：有疑问提出，**GPT-4o** 是从头开始使用 **200k 词表 tokenizer** 进行预训练的，还是在从 **100k** tokenizer 切换后继续训练的。
  
  - 有人对训练中期 (mid-training) 的混乱性质表示担忧，指出跟踪此类转换存在挑战。
- **Character.AI 表达哀悼**：**Character.AI** 就一起不幸的用户事件表达了哀悼，并强调了[此处](https://blog.character.ai/community-safety-updates/)提供的新安全功能。
  
  - 一位成员分享了一篇[《纽约时报》的文章](https://www.nytimes.com/2024/10/23/technology/characterai-lawsuit-teen-suicide.html)，进一步说明了情况的背景。
- **Anthropic 向 B2B 转型**：**Anthropic** 正在演变为一家 B2B 公司，这与 **OpenAI** 对消费者应用的关注形成对比，特别是在有趣任务与平淡任务的权衡方面。
  
  - 讨论强调了消费者更倾向于有趣的活动，而不是自动化可能处理的无聊任务（如购物）。
- **Microsoft 引人入胜的 AI 演示**：**Microsoft** 对 AI 的趣味应用（如 **Minecraft** 中的游戏自动化）与 **Anthropic** 对日常任务的关注形成鲜明对比，[点击此处查看](https://youtu.be/TLg2KWY2J5c?si=dRkIA2t1Y0F61KgP)。
  
  - 这突显了 AI 领域截然不同的策略，反映了不同的目标受众和目标。

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Screenpipe 引起关注**：成员们称赞了 [Screenpipe](https://github.com/OpenInterpreter/open-interpreter/blob/development/examples/screenpipe.ipynb) 在管理构建日志方面的实用性，展示了它对于寻求高效日志解决方案的开发者的潜力。
  
  - 一位用户强调了在开发工作流中拥有清晰且有条理的构建日志所带来的重大影响。
- **Claude 3.5 模型演进**：Anthropic 推出了 **Claude 3.5 Sonnet** 模型，该模型具有显著的代码增强功能和全新的 **computer use** 能力（目前处于公开测试阶段），允许 AI 更自然地与用户界面进行交互。
  
  - 然而，由于需要持续截屏，引发了对该模型效率和运营成本的担忧；[更多详情请点击此处](https://www.anthropic.com/news/3-5-models-and-computer-use)。
- **对 Open Interpreter 路线图的质疑**：成员们讨论了 **Open Interpreter** 的路线图，断言其独特的能力使其区别于主流 AI 产品。
  
  - 一些怀疑者对与成熟模型的竞争表示担忧，而另一些人则强调了社区驱动开发的重要性。
- **应对 AI 屏幕交互挑战**：针对使用截图作为 AI 输入的低效性出现了担忧，这促使人们建议直接从应用程序中提取必要的数据点。
  
  - 成员们认识到需要增强数据处理方法，以规避目前对截图依赖的局限性。
- **呼吁测试新的 Anthropic 集成**：一位成员介绍了用于集成 Anthropic 模型的 `interpreter --os` 命令，敦促其他人在正式发布前协助测试该功能。
  
  - 测试表明，增加屏幕尺寸和文本清晰度有助于降低模型使用过程中的错误率。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere API 试用提供免费访问**：Cohere 提供 [试用 API key](https://docs.cohere.com/docs/rate-limits)，允许在试用期间免费访问所有模型，频率限制为 **每分钟 20 次调用**，使用生产 key 后可提升至 **每分钟 500 次调用**。
  
  - 这种设置允许工程师在投入生产环境之前探索各种模型。
- **新兴的多模态 Command 模型**：讨论引发了对 **多模态 Command 模型** 的兴趣，建议增加一个集成不同交互模式的 **Global connection** 功能。
  
  - 这反映了人们对先进模型能力及其潜在应用的新奇好奇心。
- **11 月 23 日的 Agentic Builder Day**：OpenSesame 将于 11 月 23 日举办 **Agentic Builder Day**，邀请开发者使用 **Cohere Models** 参加小型 AI Agent 黑客松，[申请现已开放](https://www.opensesame.dev/hack)。
  
  - 该活动旨在促进对 AI Agent 感兴趣的开发者之间的合作与竞争。
- **Ollama Mistral 性能担忧**：成员们表达了对 **Ollama Mistral** 的问题，注意到性能波动和幻觉倾向，这使他们的项目变得复杂。
  
  - 一位用户链接到了他们的 [GitHub gist](https://gist.github.com/pleabargain/8b3f1641ef727cc114ac389cbc1b354b)，详细介绍了尽管面临这些挑战，他们仍能有效生成 prompt 的方法。
- **Tool Calls 与 Cohere V2 API 错误**：用户报告了 Cohere V2 API 中 tool calls 的 **内部服务器错误**，特别是强调了缺失的 **tool_plan field** 导致了一些问题。
  
  - 引用了 [Cohere 文档](https://docs.cohere.com/docs/tool-use#step-2) 以澄清正确的工具集成方式。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **对 stdlib 讨论的热情**：一位成员在补完上一次社区讨论后，表达了加入 **stdlib contributor meetings** 的兴奋之情。
  
  - 这种热情得到了他人的积极回应，鼓励大家参与对话。
- **Mojo 中的串口通信困扰**：一位用户寻求在 **Mojo** 中通过端口实现 **serial communication** 的指导；目前的支持仅限于 **libc** 中可用的功能。
  
  - 这表明有必要进一步增强 Mojo 的通信能力。
- **关于 Mojo 中 C/C++ 支持的辩论**：讨论了 Mojo 中 **C/C++ support** 的存在及其潜在益处。
  
  - 然而，关于这种支持对用户的实际应用价值，意见不一。
- **MAX Engine 发布 C API 公告**：**C API** 现已可用于 **MAX Engine**，尽管目前没有集成 **graph API** 的即时计划。
  
  - 官方保证，如果情况有变，将沟通关于 graph API 的更新。
- **使用 C 探索 Graph API**：一位成员指出可以使用 C 构建 **graph builder API**，建议在 **Mojo** 之外尝试其他方法。
  
  - 这为跨编程语言的潜在协作开启了讨论。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **TorchTune 配置在 .yaml 文件上出现失误**：一位成员指出，在 **TorchTune** 运行命令中使用 **.yaml** 文件扩展名会因暗示本地配置而引起混淆。
  
  - 他们指出，如果没有足够的错误信息，调试过程会令人沮丧。
- **多 GPU 测试引发疑问**：一位用户询问了在 **2 GPUs** 上的测试能力，反映了一个普遍关注的问题。
  
  - 另一位用户提到了在 **1 GPU** 和 **2 GPUs** 上运行 **lora_finetune_distributed** 脚本时出现的错误信息问题。
- **确认使用 TorchTune 进行微调**：针对 **custom Llama** 模型微调的查询，回复确认 **TorchTune** 为自定义提供了灵活性。
  
  - 鼓励成员进一步参与关于自定义组件的讨论，以获得更好的支持。
- **Linters 和 pre-commit hooks 存在 Bug**：成员报告了 **linters and pre-commit hooks** 的问题，指出它们未按预期工作。
  
  - 要跳过某行，需要同时使用 `# noqa` 和 `# fmt: on ... #fmt: off`，这被认为异常复杂。
- **PR #1868 中的 CI 混乱**：一位成员揭示了 PR **#1868** 中 **CI** 的异常行为，并寻求协助解决持续存在的问题。
  
  - 关于 CI 问题解决情况的询问表明，值得庆幸的是，该问题现在应该已经 **fixed**。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **帮助塑造开发者工具**：一位成员分享了一个[调查链接](https://forms.gle/Roi1U5ynVwLtQ3S46)，旨在收集开发者对将创意转化为成果所面临挑战的见解，完成大约需要 **5-7 分钟**。
  
  - 该调查探讨了开发者产生创意的频率、面临的障碍，以及对简化项目实现的解决方案的兴趣。
- **针对开发者的 AI 影响研究**：一项评估 AI 工具对软件工程影响的硕士研究正在征集开发者参与，参与者填写[此处](https://auckland.au1.qualtrics.com/jfe/form/SV_0uf2q5Ie7V3gpvM?Source=43)的简短问卷，有机会赢取 **$200NZD 礼品卡**。
  
  - 该研究旨在收集关于 AI 集成到工程工作流中的宝贵数据，同时为参与者提供奖励。
- **使用 AI 工具解锁资金**：一个 AI 驱动的平台上线，通过匹配相关投资者帮助用户寻找资金，为前 **200** 名候补名单注册者提供 **free Startup Accelerator pack**，目前仅剩 **62** 个名额。
  
  - 感兴趣的人士请[立即注册](https://www.aloangels.me/)，通过增强的搜索功能加速实现创业梦想。
- **构建 AI GeoGuessr 玩家**：一个新的 [YouTube 教程](https://www.youtube.com/watch?v=OyDfr0xIhss)展示了如何编写一个 AI 机器人，利用 **Multimodal Vision LLMs**（如 **GPT-4o**, **Claude 3.5**, 和 **Gemini 1.5**）自主玩 **GeoGuessr**。
  
  - 该教程涉及 **Python programming** 并使用 **LangChain**，使机器人能够有效地与游戏环境交互。
- **询问马尼拉开发者**：一位成员询问是否有人位于 **Manila**，暗示希望促进当地开发者之间的联系。
  
  - 这一询问可能会为马尼拉科技圈的社区建设或潜在协作创造机会。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **正在开发中的全球最先进工作流系统**：一位成员宣布了开发**全球最先进工作流系统**的计划，并预定于周一进行**现场演示**，以展示其运行机制和**升级流程**。
  
  - 该环节旨在深入探讨系统功能，重点讨论计划中的增强功能和即将推出的特性。
- **DSPy 设定宏大的融资目标**：继 CrewAI 成功获得 **1800 万美元**融资后，一位成员提议 **DSPy** 的目标应至少为 **5000 万美元**，并表示渴望作为第 5 号或第 10 号员工早期加入。
  
  - “我们还在等什么？”这一发言活跃了讨论氛围，强调了立即开展融资工作的行动号召。
- **有效合成数据生成的指标**：一位成员探讨了使用 **DSPy** 根据文本输入为 QA 用途生成合成数据的潜力，并提出了关于适用指标的问题。
  - 回复建议利用 **LLM as a judge**（LLM 作为评判者），并建立一套标准，用于在没有 Ground Truth（基准真相）的情况下评估开放式生成内容。
- **Groundedness（扎实度）作为合成数据的指标**：在合成数据的讨论中，一位成员建议 **Ground Truth** 应源自生成时使用的文本，并指出 Groundedness 是一个关键指标。
  
  - 他们对分享的协作见解表示感谢，凸显了成员们在该话题上的参与精神。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **LLM Agents MOOC 报名困惑**：多位成员反映在提交 [LLM Agents MOOC 报名表](https://link-to-signup-form) 后未收到确认邮件，导致对申请状态感到不确定。
  
  - 这种反馈的缺失引起了预期收到正式录取通知的用户对报名流程的担忧。
- **Hackathon 项目代码必须开源**：在 Hackathon 期间，成员们确认了必须将其项目代码 **100% 开源**的要求，这是参加最终演示的规定条件。
  
  - 这种对代码透明度的强调符合 Hackathon 促进参与者之间协作开发的初衷。
- **对 Agent 创建教程的需求**：一位参与者询问了关于在不依赖外部平台的情况下从零开始创建 Agent 的教程，强调了对易获取教育资源的需求。
  
  - 这种兴趣凸显了社区对在 Agent 开发工作流中实现自给自足的渴望。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **利用 Axolotl Discord 获取配置**：利用 🦎 Axolotl Discord 频道分享和查找针对你的用例定制的配置，GitHub 上还附带示例文件夹。查看 [Discussions 标签页](https://github.com/axolotl-ai-cloud/axolotl/discussions) 以获取见解和分享的用例。
  
  - *利用社区成果*来优化你的配置，并调整现有设置以更好地适配你的项目。
- **通过 LangSmith 最大化 Prompt 效能**：探索 🛠️ LangSmith Prompt Hub，它提供了针对各种模型和用例的大量 Prompt 集合，提升你的 Prompt Engineering 技能。访问 [Amazing Public Datasets 仓库](https://github.com/awesomedata/awesome-public-datasets) 获取高质量数据集。
  
  - *分享你自己的 Prompt* 并发现新想法，以促进协作并提升模型性能。
- **数据竞赛的 Kaggle 解决方案**：查看《最全面的 Kaggle 解决方案和创意列表》，获取关于竞赛数据科学的丰富见解。在 GitHub [此处](https://github.com/faridrashidi/kaggle-solutions) 访问完整集合。
  
  - 该资源是希望增强其方法论和策略的数据工程师的宝库。
- **Hugging Face 的模型对齐方案**：在 Hugging Face 上寻找强大的方案（Recipes），使语言模型与人类及 AI 偏好对齐，这对于持续的 Fine-tuning 至关重要。在 [此处](https://github.com/huggingface/alignment-handbook/tree/main/recipes) 发现这些宝贵资源。
  
  - 借鉴社区最佳实践的见解，有效地**对齐你的模型**。
- **介绍用于便捷消息抓取的新 Discord 机器人**：一个新创建的 Discord 机器人旨在简化频道消息的抓取工作，目前需要协助邀请成员加入该机器人。感兴趣的用户可以通过此 [链接](https://discord.com/oauth2/authorize?client_id=1298625427375656980&response_type=code&redirect_uri=https%3A%2F%2Fc123ian.github.io%2F&scope=messages.read) 邀请该机器人。
  
  - *参与其中*以增强你的 Discord 体验，并可能实现有价值讨论内容的自动化收集。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **2.5.0 为 gfx1100 引入实验性 Triton FA 支持**：在 **2.5.0 版本**中，用户可以通过设置环境变量 `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` 为 **gfx1100** 启用 **实验性 Triton Flash Attention (FA)**。更多详情请参阅 [此 GitHub issue](https://github.com/ROCm/aotriton/issues/16#issuecomment-2346675491)。
  
  - 一个 **UserWarning** 指出 **Navi31 GPU** 上的 Flash Attention 支持仍处于实验阶段。
- **Mixtral 与 Llama 3.2：使用之争**：鉴于 **Llama 3.2** 的进步，关于使用 **Mixtral** 的可行性引发了讨论。社区成员正在审视这两个模型的优缺点，以确定哪个应该优先使用。
  
  - 这一询问凸显了模型选择中不断演进的格局，反映了性能指标和对特定任务的适用性。

 

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **模型评估中的空评分报告**：一位用户报告称，在 `handler_map.py` 中注册新的模型处理器后，执行 `bfcl evaluate --model mynewmodel --test-category ast` 导致评分报告结果为 **0/0** 的空报告。
  
  - 另一位成员建议确保在评估之前运行了 `bfcl generate ...` 命令，强调了获取准确评分结果的依赖关系。
- **评估前生成模型的重要性**：讨论强调了在模型评估之前运行 `bfcl generate` 命令，以避免测试期间出现空报告。
  
  - 这表明缺少模型生成步骤可能会直接影响评估结果的有效性。

 

---

# PART 2: Detailed by-Channel summaries and links

{% if medium == 'web' %}

 

### **HuggingFace ▷ #**[**general**](https://discord.com/channels/879548962464493619/879548962464493622/1298364346061029496) (547 messages🔥🔥🔥):

> - `Stable Diffusion 3.5`
> - `Llama Models`
> - `自动化 CAD 文档创建`
> - `音乐生成模型`
> - `AI 模型基准测试`

- **关于 Stable Diffusion 3.5 性能的讨论**：成员们讨论了随着时间的推移对 Stable Diffusion 不断变化的看法，强调了关于其性能和更新的持续辩论。
  
  - 据观察，用户目前渴望测试新功能并将结果与其他模型进行比较。
- **在 CAD 中实现 AI**：一位用户正在探索使用 LLM 和 RAG 系统来自动化创建 CAD 文档，强调了从规范中获得结构化输出的需求。
  
  - 有建议认为，在进阶到更大、更复杂的模型之前，运行较小的模型可能是一个实用的起点。
- **音乐生成模型**：用户讨论了各种音乐生成模型，推荐将 Musicgen, stable-audio 和 audioldm 用于器乐。
  
  - 对于带歌词的音乐，SongCrafter 被提及为一个选项，但应对质量保持合理的预期。
- **基准测试与模型性能**：AI 模型中基准测试的可靠性受到质疑，特别是对于 LLM，其特定质量根据使用场景差异巨大。
  
  - 用户提到，个人测试通常是评估模型性能的最佳方法。
- **在移动设备上运行模型**：一位用户询问了在移动设备上运行特定 LLM 的可行性，得到的建议是使用合适的在线代理以避免本地处理。
  
  - 讨论中还包含了关于无审查模型生成的内容及其适当性的幽默评论。

**Links mentioned**:

- [Suno AI](https://suno.com/about): 我们正在构建一个任何人都能创作出优秀音乐的未来。无需乐器，只需想象力。让你的灵感化作音乐。
- [Dev Board | Coral](https://coral.ai/products/dev-board/): 一款用于快速原型化设备端 ML 产品的开发板。通过可拆卸的系统模块 (SOM)，实现从原型到生产的规模化。
- [Llama 3.2 3B Uncensored Chat - a Hugging Face Space by chuanli11](https://huggingface.co/spaces/chuanli11/Chat-Llama-3.2-3B-Instruct-uncensored): 未找到描述
- [The Simpsons Homer GIF - The Simpsons Homer Exiting - Discover & Share GIFs](https://tenor.com/view/the-simpsons-homer-exiting-uncomfortable-leaving-now-gif-12755201945629685724): 点击查看 GIF
- [AutoMatch: A Large-scale Audio Beat Matching Benchmark for Boosting Deep Learning Assistant Video Editing](https://arxiv.org/abs/2303.01884): 短视频的爆发式增长极大地重塑了人们的社交方式，形成了一种日常分享和获取最新信息的新趋势。这些丰富的视频资源，在……
- [Diffusion for World Modeling: Visual Details Matter in Atari (DIAMOND) 💎](https://diamond-wm.github.io/): Diffusion for World Modeling: Visual Details Matter in Atari (DIAMOND) 💎 网页
- [no title found](https://jsplot.dcford.org.uk/): 未找到描述
- [Nick088/FaceFusion · 🚩 Report: Legal issue(s)](https://huggingface.co/spaces/Nick088/FaceFusion/discussions/8): 未找到描述
- [RaveDJ - Music Mixer](https://rave.dj/)): 使用 AI 一键混合任何歌曲
- [We Breaking Bad GIF - WE BREAKING BAD WALTER WHITE - Discover & Share GIFs](https://tenor.com/view/we-breaking-bad-walter-white-gif-14928204287258878513): 点击查看 GIF
- [Pyplot tutorial — Matplotlib 3.9.2 documentation](https://matplotlib.org/stable/tutorials/pyplot.html): 未找到描述
- [Happy Birthday GIF - Happy Birthday - Discover & Share GIFs](https://tenor.com/view/happy-birthday-gif-27707596): 点击查看 GIF
- [Long Time GIF - Long Time Age - Discover & Share GIFs](https://tenor.com/view/long-time-age-old-finally-gif-12981705): 点击查看 GIF
- [Models - Hugging Face](https://huggingface.co/models?pipeline_tag=sum): 未找到描述
- [Downloading files](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/file_download#huggingface_hub.snapshot_download): 未找到描述
- [llama.cpp/grammars/README.md at master · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md): C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号，为 ggerganov/llama.cpp 的开发做出贡献。
- [Models - Hugging Face](https://huggingface.co/models?pipeline_tag=summarization): 未找到描述
- [ML Inference survey](https://forms.gle/vS8DrPdaKpuaGgrk8): 未找到描述
- [Oxidaksi vs. Unglued - Ounk](https://www.youtube.com/watch?v=PFKFNtUDj8g): #MASHUP #PSY #DNB #MUSIC #SPEEDSOUND 使用 Rave.dj 创建。版权所有：©2021 Zoe Love
- [GitHub - teticio/audio-diffusion: Apply diffusion models using the new Hugging Face diffusers package to synthesize music instead of images.](https://github.com/teticio/audio-diffusion): 使用新的 Hugging Face diffusers 软件包应用扩散模型来合成音乐而非图像。- teticio/audio-diffusion
- [GitHub - noamgat/lm-format-enforcer: Enforce the output format (JSON Schema, Regex etc) of a language model](https://github.com/noamgat/lm-format-enforcer): 强制执行语言模型的输出格式（JSON Schema、Regex 等）- noamgat/lm-format-enforcer
- [Audio Diffusion](https://huggingface.co/docs/diffusers/main/en/api/pipelines/audio_diffusion): 未找到描述

---

### **HuggingFace ▷ #**[**today-im-learning**](https://discord.com/channels/879548962464493619/898619964095860757/1298377865200668744) (15 messages🔥):

> - `MIT AI Course`
> - `Manim Animation Engine`
> - `Learning LLM Neural Networks`
> - `Statistics and Linear Algebra for AI`

- **探索 MIT AI 课程**：一位成员分享了一个名为“MIT 6.034 Artificial Intelligence, Fall 2010”的 [YouTube 播放列表](https://www.youtube.com/playlist?list=PLUl4u3cNGP63gFHB6xb-kVBiQHYe_4hSi)。这门综合课程包含了 Patrick Winston 教授的见解，涵盖了基础 AI 概念。
  
  - 另一位成员表达了热情，指出该课程是 AI 爱好者的*必看*内容。
- **Manim：用于数学视频的动画引擎**：一位成员透露，这些动画是使用 [Manim](https://github.com/3b1b/manim) 创建的，这是一个专为解释性数学视频设计的自定义动画引擎。该 GitHub 项目鼓励贡献，并展示了创建这些视频的底层技术。
  
  - 一位用户幽默地通过反应（reaction）表达了认可，显示了社区对这类工具的赞赏。
- **关于学习 LLM 神经网络的建议**：有人询问关于学习基础 LLM 神经网络对象（如来自 Torch 或 TikToken 的对象）的优质资源。社区提供了支持，表现出参与和协助学习过程的意愿。
  
  - 成员们表达了同伴情谊，并愿意互相帮助应对学习这些复杂技术的挑战。
- **线性代数和统计学的基础主题**：在获得理论知识后，一位成员寻求关于线性代数和统计学核心主题的指导。建议包括掌握矩阵、符号逻辑以及理解在统计学和博弈论中具有重要地位的 Nash Equilibrium。
  
  - 这反映了社区不仅关注 AI 学习的理论层面，也关注其实践层面。

**提到的链接**：

- [MIT 6.034 Artificial Intelligence, Fall 2010](https://www.youtube.com/playlist?list=PLUl4u3cNGP63gFHB6xb-kVBiQHYe_4hSi)：查看完整课程：http://ocw.mit.edu/6-034F10 讲师：Patrick Winston。在这些讲座中，Patrick Winston 教授介绍了来自 6.034 的材料...
- [GitHub - 3b1b/manim: Animation engine for explanatory math videos](https://github.com/3b1b/manim)：用于解释性数学视频的动画引擎。通过在 GitHub 上创建账户为 3b1b/manim 的开发做出贡献。

---

### **HuggingFace ▷ #**[**cool-finds**](https://discord.com/channels/879548962464493619/897390579145637909/1298473384639987722) (18 条消息🔥):

> - `量子计算探索`
> - `Vintern-3B-beta 模型开发`
> - `LLM 服务对比工具`
> - `针对 ChatGPT 的 Openblock ZK 证明`
> - `对 LLM 许可条款的批评`

- **适合各种心情的多样化 Lo-fi 音乐**：分享了一份丰富的 **Lo-fi 音乐主题** 列表，包括用于冥想的平静声音、用于购物疗法的活力曲调等，所有这些都展示了诸如水下编篮（underwater basket weaving）等创新想法。
  
  - *这是一个充满趣味的集合，激发了围绕音乐提示词和缓解焦虑的创造力。*
- **Vintern-3B-beta 脱颖而出成为竞争者**：**Vintern-3B-beta** 模型已成功整合了多个数据集，包括超过 **1000 万个越南语问答对 (QnAs)**，以对抗 LLaVA 等现有竞争对手。
  
  - 该模型展示了在训练过程中的显著进步，证明对寻找高质量语言模型选项的用户非常有益。
- **免费的 LLM 服务对比工具发布**：一位用户开发了一个免费工具，可以对比包括 OpenAI 和 Google 在内的众多供应商的 **LLM 价格和性能**，从而为可用选项提供有用的见解。
  
  - 该工具强调，高价格并不总是等同于更好的质量，挑战用户去探索各种服务提供商。
- **Openblock 创新聊天记录所有权**：Openblock 推出了一项名为 **Proof of ChatGPT** 的功能，利用 ZK 证明使用户能够全面控制其聊天记录。
  
  - 这种方法标志着用户数据主权的重大进步，解决了开源领域围绕数据所有权的担忧。
- **对古怪许可条款的批评**：有人对软件许可中包含 **古怪条款 (quirky clauses)** 表示担忧，这可能会阻碍严肃项目的实际可用性。
  
  - *讨论强调了社区内对于许可协议中不必要的复杂性日益增长的挫败感。*

**提到的链接**：

- [Tweet from undefined](https://x.com/open): 未找到描述
- [Tweet from OpenBlock (@openblocklabs)](https://x.com/openblocklabs/status/1848805457290572199): 1/ 介绍 Proof of ChatGPT，这是基于 OpenBlock 通用数据协议 (UDP) 构建的最新应用。此数据证明使用户能够拥有其 LLM 聊天记录的所有权，标志着一个重要的...
- [Tweet from vik (@vikhyatk)](https://x.com/vikhyatk/status/1848893472310464724): 我觉得人们在许可中加入这种“古怪”条款真的很烦人……这让任何对你作品的严肃使用都变得不可能。停止这样做。做得更好一点。
- [5CD-AI/Vintern-3B-beta · Hugging Face](https://huggingface.co/5CD-AI/Vintern-3B-beta): 未找到描述
- [vikhyatk/lofi · Datasets at Hugging Face](https://huggingface.co/datasets/vikhyatk/lofi): 未找到描述
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1g9js22/i_built_an_llm_comparison_tool_youre_probably/): 未找到描述

---

### **HuggingFace ▷ #**[**i-made-this**](https://discord.com/channels/879548962464493619/897390720388825149/1298368557326794793) (22 messages🔥):

> - `ZK Proofs for ChatGPT`
> - `SNES Music Diffusion Model`
> - `对 GitHub Organization 的担忧`
> - `Fourier Dual Diffusion 仓库`

- **ZK Proofs 赋能 ChatGPT 用户**：ZK proofs 正被用于让 ChatGPT 用户拥有自己的聊天记录，从而增加用于构建开源模型的可验证训练数据量。
  
  - 正如在 [demo tweet](https://x.com/openblocklabs/status/1848805457290572199) 中讨论的那样，这一创新标志着在打破 AI 模型数据壁垒和建立数据溯源方面取得了重大进展。
- **对 SNES Music Diffusion Model 的赞赏**：一位成员展示了他们的 SNES 音乐扩散模型，该模型是他们从零开始训练并经过完美修复（inpainted）的。
  
  - 另一位成员评论说**音质很棒**，并询问了更多细节，作者通过项目的 [GitHub link](https://github.com/parlance-zz/dualdiffusion) 提供了相关信息。
- **围绕 GitHub Organization 的怀疑**：有人对一个 GitHub 组织表示担忧，指控其近期创建的仓库可能存在蜜罐（honeypot）操作。
  
  - 成员们指出其**异常的许可证**以及二进制发布版本缺乏文档，从而引发了进一步的审查。
- **分享 Fourier Dual Diffusion 项目**：SNES 音乐扩散模型的开发者分享了他们的 GitHub 仓库，其中包括抓取、数据集处理、训练和 Web 界面的代码。
  
  - 他们还提到在主页上链接了一个开发博客，以提供更深入的见解。

**提到的链接**：

- [来自 OpenBlock (@openblocklabs) 的推文](https://x.com/openblocklabs/status/1848805457290572199)：1/ 介绍 Proof of ChatGPT，这是构建在 OpenBlock 通用数据协议 (UDP) 上的最新应用。此 Data Proof 赋能用户掌握其 LLM 聊天记录的所有权，标志着一个重要的...
- [Shaq GIF - Shaq - Discover & Share GIFs](https://tenor.com/view/shaq-gif-18798422)：点击查看 GIF
- [GitHub - parlance-zz/dualdiffusion: Fourier Dual Diffusion](https://github.com/parlance-zz/dualdiffusion)：Fourier Dual Diffusion。通过在 GitHub 上创建账号来为 parlance-zz/dualdiffusion 的开发做出贡献。

---

### **HuggingFace ▷ #**[**NLP**](https://discord.com/channels/879548962464493619/922424173916196955/1298406185061584989) (3 messages):

> - `自定义模型 Tokenizers`
> - `自动化 CAD 文件创建`

- **为自定义模型选择 Tokenizers**：一位成员询问该为他们的自定义模型使用哪种 tokenizer，引发了其他成员的讨论。
  
  - 一位成员建议使用 **tiktoken** 作为分词（tokenization）的一个可行选择。
- **使用 LLM 自动化 CAD 文件创建**：一位成员提议使用集成 **RAG** (retrieval-augmented generation) 和 **LLM** (large language model) 技术的流水线来自动化创建 CAD 文件。
  
  - 他们向社区征求关于该自动化方案的系统设计方法或替代策略的见解。

---

### **HuggingFace ▷ #**[**diffusion-discussions**](https://discord.com/channels/879548962464493619/1009713274113245215/1298540561896181780) (3 messages):

> - `Gradio 相关问题`
> - `频道指引`

- **关于询问 Gradio 的指引**：<@king_92582> 提出了一个关于 **Gradio** 的问题，寻求社区的帮助。
  
  - 另一位成员建议在新的频道（特别是 **<#922424173916196955>**）中询问与 **RAG** 和 **LLM** 相关的话题。
- **频道重点说明**：一位成员指出当前频道是专门用于 **diffusion models** 的，建议其他人使用合适的频道。
  
  - 明确指出，为了确保组织有序，将讨论保持在各自对应的频道中至关重要。

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1298376044893569154) (300 messages🔥🔥):

> - `Claude 3.5 Sonnet`
> - `Anthropic Computer Use Tool`
> - `Model Performance Improvements`
> - `AI Tokenization Issues`
> - `AI Systems and Optimization`

- **Claude 3.5 Sonnet 表现出显著的性能提升**：Claude 3.5 Sonnet 在 SWE-bench 和增强型 Agent 基准测试中提升了约 **15%**，表明其微调取得了成功。
  
  - Active learning（主动学习）技术的整合可能有助于这些增强，并使模型在计算机操作能力方面更加高效。
- **Anthropic 推出 Computer Use Tool**：Anthropic 发布了一项新工具，允许 Agent 在计算机上执行任务，这可能代表了 Agent 功能的未来。
  
  - 该工具利用先进的数据处理来改善用户交互，为 API 消费者提供更直观的工具。
- **关于 AI 系统优化的讨论**：对话强调了自 GPT-3 以来， AI 模型在参数效率上变得更高，从而能够以更少的资源实现显著的性能提升。
  
  - 与会者推测，持续的优化将增强操作能力，包括计算机控制的功能。
- **关于 Anthropic Tokenizer 的担忧**：有人对 Anthropic 的 Tokenizer 的有效性提出了担忧，据报道它会产生重复且不必要的响应。
  
  - 更好的 Tokenization 方法可能会显著提升模型的整体性能。
- **用户遇到会话登出问题**：多名用户报告在 ChatGPT 中遇到自动登出的情况，发生频率约为每周 1-2 次。
  
  - 这似乎是用户中的一个普遍问题，尽管发生频率并不高。

 

**提到的链接**：[How to Keep Improving When You're Better Than Any Teacher - Iterated Distillation and Amplification](https://youtu.be/v9M2Ho9I9Qo)：[第二次上传] AI 系统可以使用专家的演示进行训练，但如何训练它们超越这些专家？即使在……的情况下，这是否仍然可行？

 

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1298505211635175515) (9 messages🔥):

> - `GPT-4 Upgrade Timeline`
> - `Caching Feature in GPT-4`
> - `ChatGPT Payment Issues`

- **对 GPT-4 升级时间线感到好奇**：一位成员注意到 **GPT-4** 在今年大部分时间里都在被使用，但记得看到过关于即将升级的提及，尽管他们找不到相关信息。
  
  - 另一位成员提到，大约在 **4-5 个月前**，免费用户获得了使用 GPTs 的权限。
- **关于缓存功能的讨论**：一位成员询问 GPT API 中的函数调用（function calls）是仅使用最后一条消息还是整个对话的上下文。
  
  - 他们还对新的缓存功能表示困惑，提到在 Langsmith 中显示 **cache hit false**（缓存命中失败）。
- **ChatGPT 支付问题的困惑**：一位用户报告在支付 **月费** 后无法访问 Plus 功能，登录时仍收到升级提示。
  
  - 另一位成员建议他们通过 [OpenAI Help](https://help.openai.com) 联系支持团队以寻求解决。

 

**提到的链接**：[Custom GPT's upgrade base model to omni?](https://community.openai.com/t/custom-gpts-upgrade-base-model-to-omni/780612)：大家好！我对 GPT-4o 感到非常惊叹，尤其是它的快速响应和主题重组能力。但我正在考虑如何将我的自定义 GPTs 从旧模型（我认为是 Gizmo 模型）迁移到……

 

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/1298418764131336263) (7 messages):

> - `Spatial Sense in Models` (模型中的空间感)
> - `Custom GPT Challenge` (自定义 GPT 挑战)
> - `Function Call Context` (Function Call 上下文)
> - `Realtime API Performance` (Realtime API 性能)

- **Models struggle with spatial sense**: 一场讨论强调了模型在**空间感 (spatial sense)** 方面表现较弱，但如果训练数据中存在正确答案，它们可以重复这些答案，类似于孩子在不理解的情况下模仿学到的反应。
  
  - 模型可能会解决问题，但在需要更深层理解的任务中表现挣扎。
- **Challenge with 'Not ChatGPT'**: 一位成员介绍了一个名为 **'Not ChatGPT'** 的自定义 GPT，它被编程为否认与 ChatGPT 的任何联系，引发了人们对其揭示这种关系的潜力的好奇。
  
  - 挑战在于说服它承认自己的起源，这暗示了模型设计中的某种巧妙性。
- **Understanding Context for Function Calls**: 关于 **ChatGPT** 中的 Function Call 是仅依赖于最新消息，还是会考虑之前的对话以获得更好的上下文，产生了一些疑问。
  
  - 提到一个特定的用例，即 Function 应该仅在对话中经过多次确认后才触发。
- **Concerns with Realtime API performance**: 一位成员指出，**Realtime API** 在遵循其 system prompt 指令方面的表现不如 GPT-4o。
  
  - 他们寻求关于如何调整 prompt 以获得更好性能的建议。

 

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/1298418764131336263) (7 messages):

> - `Spatial Sense Weakness` (空间感弱点)
> - `Not ChattGPT Custom GPT` (Not ChattGPT 自定义 GPT)
> - `Function Calls Context in Chat Completion` (Chat Completion 中的 Function Call 上下文)
> - `Realtime API Performance Comparison` (Realtime API 性能对比)
> - `Prompt Adaptation Suggestions` (Prompt 适配建议)

- **Models Struggle with Spatial Sense**: 一位成员强调模型在空间感方面**极其虚弱**，通常只是重复训练数据中的正确答案，而没有真正的理解。
  
  - 他们将其比作一个能背诵答案但缺乏应用理解能力的学生。
- **Not ChattGPT Defies Identity**: 一位成员展示了他们的自定义 GPT，名为 **'Not ChattGPT'**，旨在否认与 ChatGPT 的任何联系，同时尝试揭示其关系。
  
  - 他们邀请其他人通过询问其与 ChatGPT 的联系来挑战该模型，排除假设性问题。
- **Context Needed for Function Calls**: 多位成员询问 Chat Completion 中的 Function Call 是仅取决于**最后一条消息**，还是会考虑之前的上下文。
  
  - 他们说明了 Function 仅在聊天中经过**多次确认**后才激活的需求。
- **Feedback on Realtime API**: 一位成员对 **Realtime API** 在遵循 system prompt 方面的表现不如 **GPT-4o** 表示沮丧。
  
  - 他们征求可能有类似经历的其他人的意见或改进建议。
- **Ways to Adapt Prompts**: 基于关于 Realtime API 的讨论，一位成员询问了如何有效**适配 prompt** 的建议。
  
  - 这突显了通过更好的 Prompt Engineering 来改进交互和输出的持续努力。

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**general**](https://discord.com/channels/1179035537009545276/1179035537529643040/1298369505621184684) (202 条消息🔥🔥):

> - `Claude 3.5 models`
> - `Kaggle & PyTorch issues`
> - `Fine-tuning challenges`
> - `Unsloth sloth symbolism`
> - `MMLU performance concerns`

- **Claude 3.5 模型发布**：Anthropic 宣布了升级版的 **Claude 3.5 Sonnet** 和全新的 **Claude 3.5 Haiku** 模型，强调了显著的改进，特别是在编程任务方面。
  
  - 公开测试版中引入的 **computer use** 功能允许开发者指示 Claude 像人类一样与计算机进行交互。
- **Kaggle 上的 PyTorch 问题**：用户报告在 Kaggle 上运行 PyTorch 时出现 `ImportError` 问题，特别是与 CUDA 版本差异有关，促使通过重新安装特定版本来解决。
  
  - 将 PyTorch 降级以使用 CUDA 12.1 可以解决错误并确保现有库安装的兼容性。
- **模型 Fine-tuning 的挑战**：有人对模型在 Fine-tuning 过程中重复输入表示担忧，讨论建议系统提示词（system prompt）可能需要变化以防止过拟合（overfitting）。
  
  - 用户推测训练样本不足可能导致对 base model 的依赖，从而导致 Fine-tuning 后的模型产生重复输出。
- **Unsloth 标志的含义**：讨论了 Unsloth 中树懒（sloth）的象征意义，认为它代表了缓慢的传统 Fine-tuning 过程与更快、更高效的过程之间的对比。
  
  - 参与者指出 **Unsloth** 意味着让缓慢的过程变得“不慢（unslow）”，并强调了一种更快捷的 AI 模型训练方法。
- **MMLU 性能观察**：关于 MMLU 性能的讨论表明，在特定数据集上训练的模型通常表现出特定领域的优势，但通常不会超过像 GPT-4 这样的大型通用模型。
  
  - 对话强调了仅通过在有限数据集上进行 Fine-tuning 很难在各种任务中实现卓越性能。

**提到的链接**：

- `PyTorch`
  
  : 未找到描述
- [Introducing computer use, a new Claude 3.5 Sonnet, and Claude 3.5 Haiku](https://www.anthropic.com/news/3-5-models-and-computer-use)：更新且更强大的 Claude 3.5 Sonnet、Claude 3.5 Haiku 以及一项新的实验性 AI 功能：computer use。
- [Unsloth Notebooks | Unsloth Documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks)：查看下方列表获取我们所有的 notebook：
- [Kortix/FastApply-7B-v1.0 · Hugging Face](https://huggingface.co/Kortix/FastApply-7B-v1.0)：未找到描述
- [TinyLlama/TinyLlama-1.1B-Chat-v1.0 · Hugging Face](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)：未找到描述
- [no title found](https://download.pytorch.org/whl/cu121)：未找到描述
- [cerebras/SlimPajama-627B · Datasets at Hugging Face](https://huggingface.co/datasets/cerebras/SlimPajama-627B)：未找到描述
- [Issues · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/134929))：Python 中的张量和动态神经网络，具有强大的 GPU 加速 - Issues · pytorch/pytorch
- [gbharti/finance-alpaca · Datasets at Hugging Face](https://huggingface.co/datasets/gbharti/finance-alpaca)：未找到描述
- [GitHub - huggingface/transformers: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX.](https://github.com/huggingface/transformers.git)：🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的前沿机器学习。 - huggingface/transformers
- [Google Colab](https://colab.research.google.com/drive/17d3U-CAIwzmbDRqbZ9NnpHxCkmXB6LZ0?usp=sharing)：未找到描述
- [Gemma2 fails saving as GGUF · Issue #785 · unslothai/unsloth](https://github.com/unslothai/unsloth/issues/785#issuecomment-2426714313)：@danielhanchen 嗨 Daniel，感谢你的工作！遇到了一个类似于 issue #275 的错误，但这次是在尝试保存 unsloth/gemma-2-9b-it-bnb-4bit 的微调版本时。model.save_pretraine...
- [mlabonne/FineTome-100k · Datasets at Hugging Face](https://huggingface.co/datasets/mlabonne/FineTome-100k)：未找到描述
  
   
  

---

### **Unsloth AI (Daniel Han) ▷ #**[**off-topic**](https://discord.com/channels/1179035537009545276/1179039861576056922/1298521687452549160) (1 条消息):

> - `PhD Research Experience` (博士研究经验)
> - `Publication Rates in Academia` (学术界论文发表率)
> - `Comparative PhD Culture` (博士文化对比)
> - `AI/ML/CV Fields` (AI/ML/CV 领域)

- **欧洲与美国博士历程对比**：一位欧洲的博士生分享了他们的经历，强调了 **4 年的时间线**，其中包括学习研究方法、签署工业界合同，并逐步在顶级会议上发表论文。
  
  - *他们的经验强调了研究技能的深度培养*，这与据报道产出 **10 篇论文**且拥有多个第一作者署名的美国同行形成了鲜明对比。
- **美国学术界的发表压力**：该学生对美国博士生的高论文产出表示困惑，指出有些人平均发表 **10 篇论文**，其中许多是 **CVPR** 和 **ICML** 等顶会的论文第一作者。
  
  - *这一消息引发了关于工作与生活平衡的讨论*，以及学者在高度竞争的环境中所面临的压力。
- **工业界参与提升技能**：该同学强调了他们积极参与工业界项目，为其实验室相关的公司编写 **production grade code**（生产级代码），以增强其实践技能。
  
  - *这种经历为其研究能力增加了实践维度*，为未来的职业机会奠定了基础。

**提到的链接**：[Reddit - Dive into anything](https://www.reddit.com/r/MachineLearning/comments/1g7dzkp/d_why_do_phd_students_in_the_us_seem_like/)：未找到描述

---

### **Unsloth AI (Daniel Han) ▷ #**[**help**](https://discord.com/channels/1179035537009545276/1179777624986357780/1298373345695367281) (92 条消息🔥🔥):

> - `Multi-GPU Support` (多 GPU 支持)
> - `Unsloth Import Errors` (Unsloth 导入错误)
> - `Using Unsloth with Kaggle` (在 Kaggle 上使用 Unsloth)
> - `Model Fine-Tuning Procedures` (模型微调流程)
> - `Gemma Templates` (Gemma 模板)

- **Unsloth 暂不支持多 GPU**：一位成员对 Unsloth 目前不支持多 GPU 使用表示沮丧，特别是在尝试跨 4090 和 3090 加载模型时。
  
  - 另一位用户提到，DDP 多 GPU 支持等功能正在 Beta 测试中，而完全的 Sharded Data Parallelism（分片数据并行）需要更长时间来实现。
- **导入错误和 CUDA 问题**：一位用户报告了在 Kaggle 上尝试使用 Unsloth 时出现与 `libcusparse.so.12` 相关的 ImportError，这表明可能存在底层的 CUDA 问题。
  
  - 社区成员建议该错误可能与安装有关，并建议检查 CUDA 兼容性。
- **Unsloth 在 Kaggle Notebooks 中的功能**：一位用户指出在尝试从 Kaggle Notebooks 运行 Unsloth 时持续报错，由于未对设置做任何更改，因此对这种情况感到沮丧。
  
  - 另一位用户发布了一个解决类似问题的方案链接，表明 Unsloth 在 Kaggle 上存在持续的兼容性问题。
- **使用 Unsloth 微调模型**：多位用户讨论了使用 Unsloth 微调 Llama 模型的经验，提到了最近的错误以及故障排除建议。
  
  - 一位用户确认，回退到旧版本的 Unsloth 可以帮助缓解微调过程中遇到的一些问题。
- **寻找 Gemma 模板**：围绕模型中使用的 Gemma 模板的讨论分享了代码片段和链接，以帮助他人正确格式化消息。
  
  - 社区成员对分享的资源表示感谢，这使得在项目中调整和使用模板变得更加容易。

**提到的链接**：

- [无标题](https://ai.google.dev/gemma/docs/formatting)：未找到描述
- [tokenizer_config.json · microsoft/Phi-3.5-mini-instruct at main](https://huggingface.co/microsoft/Phi-3.5-mini-instruct/blob/main/tokenizer_config.json)：未找到描述
- [UNSL - Overview](https://github.com/unsl)：UNSL 有一个可用仓库。在 GitHub 上关注他们的代码。
- [Many bug fixes (#1162) · unslothai/unsloth@0e5a507](https://github.com/unslothai/unsloth/commit/0e5a507f87132cd8fbae5239fc436ef5ba3232d6)：\* 修复 TRL
  
  - 更新 mistral.py
  - 补丁 processing_class
  - 更新 tokenizer_utils.py
  - 更新 tokenizer_utils.py
  - 更新 tokenizer_utils.py
  - 更新 tokenizer_utils.py
  - 更新 ...
  - [finetune_llama_unsloth.py](https://gist.github.com/Tengoles/488889e5a07a17aa99327076ba703460)：GitHub Gist：即时分享代码、笔记和片段。
  - [GitHub - unslothai/unsloth: Finetune Llama 3.2, Mistral, Phi & Gemma LLMs 2-5x faster with 80% less memory](http://github.com/unslothai/unsloth)：微调 Llama 3.2, Mistral, Phi & Gemma LLM 速度提升 2-5 倍，内存占用减少 80% - unslothai/unsloth
    
     
    

---

### **Unsloth AI (Daniel Han) ▷ #**[**showcase**](https://discord.com/channels/1179035537009545276/1179779344894263297/1298509447815106633) (1 条消息):

> - `Fast Apply`
> - `Qwen2.5 Coder Model`
> - `Cursor's Blog Post`
> - `Performance Metrics`

- **Fast Apply 彻底改变了代码更新方式**：令人兴奋的消息！**Fast Apply** 是一个开源且经过微调的 **Qwen2.5 Coder Model**，它能够快速准确地应用代码更新，无需重复的文件编辑，从而大幅提升效率。
  
  - 该解决方案在使用 Aider 等工具时特别有益，能让大模型集中精力进行实际的代码更新，而不是处理繁琐的 **SEARCH/REPLACE** 任务。
- **Cursor 的博客启发了 Fast Apply**：Fast Apply 的开发灵感源自 **Cursor** 一篇现已删除的博客文章，存档版本可在 [此处](https://web.archive.org/web/20240823050616/https://www.cursor.com/blog/instant-apply) 查看。
  
  - 这展示了社区资源如何影响 AI 编程工具的进步。
- **Fast Apply 令人印象深刻的速度统计**：**Performance metrics**（性能指标）显示，当使用像 Fireworks 这样快速的服务商时，**1.5B Model** 的运行速度约为 **340 tok/s**，而 **7B Model** 约为 **150 tok/s**。
  
  - 这些性能水平突显了 Fast Apply 在提高编码任务生产力方面的潜力。

 

**提到的链接**：[Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1ga25gj/introducing_fast_apply_replicate_cursors_instant/)：未找到描述

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**community-collaboration**](https://discord.com/channels/1179035537009545276/1180144489214509097/1298368178392535080) (3 条消息):

> - `Studio Fix PR`
> - `Tokenizer Patch PR`

- **修复 Studio 问题的 PR**：提交了一个名为 [Fix/studio](https://github.com/unslothai/unsloth-studio/pull/1/files) 的 Pull Request，旨在解决 Discord 用户反馈的关于导入 unsloth 时的几个问题。
  
  - 据报道，该问题在 finetune notebook 中不会发生，这表明是 studio 环境特有的问题。
- **Tokenizer Bug 的简单修复**：另一个 Pull Request [Fix/patch tokenizer](https://github.com/unslothai/unsloth/pull/1171) 引入了一个微小的改动，旨在修正否定词的放置位置，此前该位置在满足条件时会导致 **NoneType** 错误。
  
  - 一位用户在修复后表达了感谢，强调了代码功能中哪怕是微小改动的重要性。

**提到的链接**：

- [Fix/patch tokenizer by Erland366 · Pull Request #1171 · unslothai/unsloth](https://github.com/unslothai/unsloth/pull/1171)：否定词位置放置错误，因此引入了 NoneType 对象不可调用的错误，因为如果为 None，它会进入 else 部分。
- [Fix/studio by Erland366 · Pull Request #1 · unslothai/unsloth-studio](https://github.com/unslothai/unsloth-studio/pull/1/files)：Studio 中存在几个问题。这些问题是由 Discord 用户提出的。该问题在导入 unsloth 时触发，但不知为何在 finetune notebook 内部没有发生……

---

### **Unsloth AI (Daniel Han) ▷ #**[**research**](https://discord.com/channels/1179035537009545276/1257011997250424842/) (1 条消息):

edd0302: [https://arxiv.org/pdf/2410.16663](https://arxiv.org/pdf/2410.16663)

---

### **Stability.ai (Stable Diffusion) ▷ #**[**general-chat**](https://discord.com/channels/1002292111942635562/1002292112739549196/1298361806552698961) (294 messages🔥🔥):

> - `Stable Diffusion 3.5`
> - `AI Generation Models`
> - `LoRA and VAE Handling`
> - `Image Prompting Techniques`
> - `Model Checkpoints and Sorting`

- **Stable Diffusion 3.5 训练与使用**：参与者讨论了 **SD 3.5** 的结构和功能，指出其使用了类似于 **Flux** 的不同神经网络架构。
  
  - 众人达成共识，认为需要通过微调（finetuning）和社区训练努力来优化新模型的效果。
- **动漫艺术的 Prompt 技巧**：为了在生成动漫艺术时获得满意的结果，建议使用带有正确提示词的 **SD 3.5**，而不是依赖 LoRA。
  
  - 建议绕过 LoRA，仅使用带有适当提示词的 Stable Diffusion 3.5 以获得最佳效果。
- **图像生成结果质量**：用户报告图像生成质量参差不齐，特别是在使用不正确的 Checkpoint 或针对其他模型的 LoRA 时。
  
  - 参与者建议检查模型是否与预期的提示词正确匹配，以避免不理想的输出。
- **模型管理与组织**：一位用户表示需要一种工具来自动对各个文件夹中的 AI 模型文件进行排序和组织。
  
  - 建议包括在服务器的技术支持频道中寻求潜在的解决方案。
- **社区参与与工具**：讨论强调了各种增强 AI 生成工作流的工具和方法，包括使用模型创建者的工作流。
  
  - 参与者分享了使用不同工具（如 ComfyUI 和 fp8 模型）的经验，以便更好地管理 AI 任务。

**提到的链接**：

- [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206)：Diffusion 模型通过反转数据向噪声的前向路径从噪声中创建数据，已成为处理高维感知数据（如...）的一种强大的生成建模技术。
- [Meta AI](https://www.meta.ai/)：使用 Meta AI 助手完成任务，免费创建 AI 生成的图像，并获得任何问题的答案。Meta AI 基于 Meta 最新的 Llama 大语言模型构建，并使用 Emu...
- [Rock Hyrox Rock Hyrox Eating GIF - Rock hyrox Rock hyrox eating Rock hyrox funny - Discover & Share GIFs](https://tenor.com/view/rock-hyrox-rock-hyrox-eating-rock-hyrox-funny-animal-animal-eating-gif-15423214377703191267)：点击查看 GIF
- [stabilityai/stable-diffusion-3.5-large at main](https://huggingface.co/stabilityai/stable-diffusion-3.5-large/tree/main)：未找到描述
- [Genmo AI Mochi 1 - The Best Open Source DiT Video Model By Far](https://www.youtube.com/watch?v=qDJrSK6uynQ)：在这段视频中，我们来看看开创性的 Genmo AI Mochi 1，这是最新的开源视频生成模型，正在彻底改变行业。凭借...
- [Stable Diffusion 3.5 Large - v1.0 | Stable Diffusion Checkpoint | Civitai](https://civitai.com/models/878387/stable-diffusion-35-large)：Stable Diffusion 3.5 Large 是一个多模态 Diffusion Transformer (MMDiT) 文本生成图像模型，具有改进的图像质量性能...
- [Dress Up Magic: Seamless Clothes-Swapping with ControlNet in Automatic1111](https://youtu.be/AGqqj_xQ6IM)：利用 AI 魔法立即变换任何角色的装扮！探索 ControlNet + Automatic1111 如何在 Stable Diffusion 中实现像素级完美的无缝换装...
- [Walmart Back Then Vs Walmart Now #walmart #2004 #2024 #groceryshopping #groceries #food #nostalgia](https://youtube.com/shorts/pNaMOMqK4n4?si=t2z0EW_dItn3T0ww)：未找到描述
- [Added preliminary support for SD3.5-large lora training · ostris/ai-toolkit@3400882](https://github.com/ostris/ai-toolkit/commit/3400882a8099645ce4c797f57ac258f1e1424ffd)：未找到描述
- [Tweet from Michael R. Bernstein (@NerdWorldOrder)](https://x.com/NerdWorldOrder/status/1740177328955924781)：我在文本生成图像提示词中使用艺术家名字。我用得非常多。大多数情况下我使用已故艺术家的名字，但对于他们的作品是否仍在...持不可知论态度。
- [Tweet from Michael R. Bernstein (@NerdWorldOrder)](https://x.com/NerdWorldOrder/status/1742294834558517695)：今天发现了一种新的 #aiART 风格，形式为提示词 "By Artist A and Artist B" 以及负面提示词 "Artist C"，他们都不是在世艺术家。在这种情况下，其中两位艺术家仍然...

---

### **aider (Paul Gauthier) ▷ #**[**announcements**](https://discord.com/channels/1131200896827654144/1133060115264712836/1298395827508609109) (1 条消息):

> - `Aider v0.60.0`
> - `Sonnet 10/22 support`（支持 Sonnet 10/22）
> - `Improved code editing`（改进的代码编辑）
> - `Bugfixes`（Bug 修复）
> - `Model metadata management`（模型元数据管理）

- **Aider v0.60.0 发布亮点**：**Aider v0.60.0** 的新版本包含了对 **Sonnet 10/22** 的全面支持，该模型现在是代码编辑基准测试（code editing benchmark）中的默认模型。
  
  - 此版本强调了健壮的格式化和文件交互处理，显著提升了用户体验。
- **Sonnet 10/22 成为默认模型**：Aider 已将 **Sonnet 10/22** 集成为默认模型，确保在其代码编辑基准测试中达到最先进（state-of-the-art）的性能。
  
  - 此次调整旨在为用户提供更精准的代码预测，并简化编辑任务。
- **增强的代码编辑功能**：此次更新改进了新增文件和只读文件的格式化，解决了 **o1 models** 的一些解析不一致问题。
  
  - 这些改进还包括针对清晰文件名的更强 Prompt，以及对不合规代码编辑回复的更好处理。
- **显著的 Bug 修复和功能**：该版本修复了一个 Bug，使其能正确地在 `/help` RAG 结果中包含 URL，并增加了忽略 **.env** 文件的功能。
  
  - Aider 甚至完成了一项显著的壮举：编写了此版本中 **49% 的代码**，展示了其强大的能力。
- **模型元数据和设置管理**：Aider 附带了一个小型模型元数据 JSON 文件，以适配 **litellm** 中未更新的模型，增强了整体功能。
  
  - 此外，专门针对 **Azure** 上的 **o1 models** 的新模型设置体现了稳健的模型管理方法。

---

### **aider (Paul Gauthier) ▷ #**[**general**](https://discord.com/channels/1131200896827654144/1131200896827654149/1298360568557867028) (240 messages🔥🔥):

> - `Claude 3.5 Sonnet`
> - `Aider Benchmarking`
> - `Fast Apply`
> - `Qwen 2.5 and Replete`
> - `OpenRouter and Rate Limits`

- **新版 Claude 3.5 Sonnet 展现出极具前景的性能**：许多用户报告称，新版 **Claude 3.5 Sonnet** 模型显著优于之前的 **O1 模型**，经常能实现以前无法达到的结果。
  
  - 一位用户提到，它仅凭一条提示词就成功地在其代码库中实现了一个 VAD 库。
- **Aider 基准测试见解**：用户正积极针对 Aider 测试不同量化版本的模型，力求在硬件限制内获得更好的性能，特别关注 **Qwen 2.5** 模型。
  
  - 一位用户使用 **Replete** 版本达到了 **62.4%** 的分数，并表示有兴趣测试更大的模型进行对比。
- **Fast Apply 推出**：**Fast Apply** 工具旨在简化现有文件中的代码更新，专为提高处理复杂代码库时的效率而开发。
  
  - 该工具旨在最大限度地减少处理中的冗余，并降低代码编辑操作期间的 Token 成本。
- **关于 OpenRouter 和速率限制（Rate Limits）的讨论**：一些用户对 **Anthropic 的速率限制** 表示不满，并讨论了转向 **OpenRouter** 以获取更好条款的情况，特别是针对隐私敏感型应用。
  
  - 用户指出，使用 OpenRouter 可以绕过 Anthropic API 施加的严格限制。
- **模型惯例（Conventions）探索**：一位新用户询问了有关 **CONVENTIONS.md** 资源的信息，以便有效地指导 Aider，这表明了对最佳实践和文档的需求。
  
  - 社区分享了关于模型配置的各种见解以及在最近更新中观察到的改进。

**提到的链接**：

- [Rate limits - Anthropic](https://docs.anthropic.com/en/api/rate-limits#requirements-to-advance-tier)：未找到描述
- [Model warnings](https://aider.chat/docs/troubleshooting/warnings.html)：Aider 是你终端里的 AI 结对编程工具
- [Separating code reasoning and editing](https://aider.chat/2024/09/26/architect.html)：Architect 模型描述如何解决编码问题，而 Editor 模型将其转化为文件编辑。这种 Architect/Editor 方法产生了 SOTA 级别的基准测试结果。
- [OpenAI compatible APIs](https://aider.chat/docs/llms/openai-compat.html)：Aider 是你终端里的 AI 结对编程工具
- [bartowski/Replete-LLM-V2.5-Qwen-32b-GGUF · Hugging Face](https://huggingface.co/bartowski/Replete-LLM-V2.5-Qwen-32b-GGUF)：未找到描述
- [Models: 'google' | OpenRouter](https://openrouter.ai/models?q=google)：在 OpenRouter 上浏览模型
- [Kortix/FastApply-7B-v1.0 · Hugging Face](https://huggingface.co/Kortix/FastApply-7B-v1.0)：未找到描述
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1gajy1j/aider_optimizing_performance_at_24gb_vram_with/)：未找到描述
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1ga25gj/introducing_fast_apply_replicate_cursors_instant/)：未找到描述
- [Decompiler Explorer](https://dogbolt.org/?id=f4fbe795-0956-4f25-aab5-27aeb7db171d)：Decompiler Explorer 是一个交互式在线反编译器，可以显示来自许多流行反编译器的等效类 C 输出。
- [aider/aider/voice.py at main · Aider-AI/aider](https://github.com/Aider-AI/aider/blob/main/aider/voice.py)：Aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账号为 Aider-AI/aider 做出贡献。
- [Build software better, together](https://github.com/Aider-AI/aider/pull/2099)：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、分叉并为超过 4.2 亿个项目做出贡献。
- [Claude 3.5 Sonnet (self-moderated) - API, Providers, Stats](https://openrouter.ai/anthropic/claude-3.5-sonnet:beta)：新的 Claude 3.5 Sonnet 提供了优于 Opus 的能力，速度快于 Sonnet，且价格与 Sonnet 相同。通过 API 运行 Claude 3.5 Sonnet（自我审查版）。

---

### **aider (Paul Gauthier) ▷ #**[**questions-and-tips**](https://discord.com/channels/1131200896827654144/1133060505792159755/1298373493540519947) (45 条消息🔥):

> - `Claude 3.5 更新`
> - `Aider 调试技巧`
> - `Mistral API 身份验证问题`
> - `Git 命令与导航`
> - `Repo map 功能`

- **Claude 3.5 引入新功能**：Anthropic 发布了升级版的 [Claude 3.5 Sonnet](https://x.com/AnthropicAI/status/1848742740420341988) 和新模型 Claude 3.5 Haiku，具备像人类一样使用计算机的能力。
  
  - *一位成员指出*，该功能包括指挥 Claude 操作屏幕、光标和文本输入。
- **Mistral API 访问故障排除**：一位用户在将 Mistral API 与 Aider 配合使用时遇到了 *AuthenticationError*，这表明其身份验证密钥可能存在问题。
  
  - 经过反馈，他们通过删除并重新创建密钥解决了该问题，随后恢复正常工作。
- **理解 Aider 中的 Git 命令**：一位用户询问了 Aider 中 /undo 命令的影响，其他人回应称旧的 commits 仍然可以通过其 hashes 访问。
  
  - *经确认*，诸如 checkout 之类的 Git 命令本质上与 Aider 无关，而是基础的 Git 功能。
- **Repo map 与代码上下文功能**：讨论围绕 repo map 如何理解文件关系展开，强调了提供相关代码上下文对于有效修改的重要性。
  
  - *Paul 澄清说*，标识符之间的关系是基于模型评估时的定义和引用。
- **Aider 中的命令 - 缩写与冗余**：一位用户质疑 '/read' 和 '/read-only' 之间的区别，发现它们本质上是相同的。
  
  - Paul 澄清说，所有命令都可以缩写，并建议只需保留完整命令以避免混淆。

**提到的链接**：

- [使用 tree-sitter 构建更好的 repository map](https://aider.chat/2023/10/22/repomap.html)：Tree-sitter 允许 aider 构建一个能更好总结大型代码库的 repo map。
- [编辑格式](https://aider.chat/docs/more/edit-formats.html)：Aider 使用各种“编辑格式”让 LLM 编辑源文件。
- [模型警告](https://aider.chat/docs/llms/warnings.html)：aider 是你终端里的 AI 结对编程工具。
- [Anthropic (@AnthropicAI) 的推文](https://x.com/AnthropicAI/status/1848742740420341988)：介绍升级版的 Claude 3.5 Sonnet 和新模型 Claude 3.5 Haiku。我们还推出了 Beta 版的新功能：computer use。开发者现在可以指挥 Claude 像……一样使用计算机。
- [高级模型设置](https://aider.chat/docs/config/adv-model-settings.html)：为 LLM 配置高级设置。
- [architect 模式会提示添加文件吗？· Issue #2121 · Aider-AI/aider](https://github.com/Aider-AI/aider/issues/2121)：在 Discord 中分享的 Issue：https://discord.com/channels/1131200896827654144/1133060505792159755/1298228879210577931 /architect 示例 bla bla ... 现在，我们需要更新其他文件以整合 th...

---

### **aider (Paul Gauthier) ▷ #**[**links**](https://discord.com/channels/1131200896827654144/1268910919057149974/1298381074744868914) (1 条消息):

> - `DreamCut AI`
> - `Claude AI`
> - `视频编辑软件`

- **DreamCut AI：视频编辑的新纪元**：[DreamCut AI](http://dreamcut.ai) 被介绍为一款完全使用 **Claude AI** 从零开始构建的视频编辑器，耗时 **3 个月，代码量超过 5 万行**。
  
  - 目前处于早期访问阶段，用户可以使用免费账号测试 **AI 工具**，展示了编程与 AI 技术的有趣结合。
- **AI 在软件开发中的角色**：DreamCut AI 的创建展示了 AI 如何通过消除传统的设计阶段并直接专注于编码，在软件开发中发挥关键作用。
  
  - 一位成员表示这种构建软件的方法很**有趣**，可能预示着更多 AI 驱动开发流程的趋势。

 

**提到的链接**：[Meng To (@MengTo) 的推文](https://x.com/MengTo/status/1848669694800367901)：介绍 [http://dreamcut.ai](http://dreamcut.ai)，一个我使用 Claude AI 从零开始构建的视频编辑器。这耗费了 3 个月和超过 5 万行代码。我完全跳过了设计，直接进入编码阶段。目前处于……

 

---

### **OpenRouter (Alex Atallah) ▷ #**[**announcements**](https://discord.com/channels/1091220969173028894/1092729520181739581/1298451822255538226) (2 messages):

> - `Claude 3.5 Sonnet`
> - `Lumimaid v0.2`
> - `Magnum v4`
> - `Discounts on Models`

- **Claude 3.5 Sonnet 版本发布**：**Claude 3.5 Sonnet 的旧版本**已发布并可供下载，附带时间戳供参考：[Claude 3.5 Sonnet](https://openrouter.ai/anthropic/claude-3.5-sonnet-20240620) 和 [Claude 3.5 Sonnet: Beta](https://openrouter.ai/anthropic/claude-3.5-sonnet-20240620:beta)。
  
  - *这些版本由 OpenRouter 提供，允许用户访问之前的迭代版本。*
- **Lumimaid v0.2 推出**：**Lumimaid v0.2** 现已可用，它是 Llama 3.1 70B 的微调版本，与 Lumimaid v0.1 相比，其**数据集显著增强**，可通过[此链接](https://openrouter.ai/neversleep/llama-3.1-lumimaid-70b)访问。
  
  - *由于数据集细节的更新，用户可以期待性能的提升。*
- **Magnum v4 发布并具备独特功能**：**Magnum v4** 已发布，经过微调以复制类似于 Sonnet 和 Opus 的文本质量，可在[此处](https://openrouter.ai/anthracite-org/magnum-v4-72b)获取。
  
  - *该模型延续了增强 AI 生成文本质量的趋势。*
- **Magnum 模型的精彩折扣**：**Magnum v1** 和 **Magnum v4** 目前在 Mancer 开启限时半价优惠。
  
  - *此次折扣为用户提供了一个以更低成本探索这些新模型的绝佳机会。*

**提到的链接**：

- [Lumimaid v0.2 70B - API, Providers, Stats](https://openrouter.ai/neversleep/llama-3.1-lumimaid-70b)**)：Lumimaid v0.2 70B 是 [Llama 3 的微调版本。通过 API 运行 Lumimaid v0.2 70B
- [Magnum v4 72B - API, Providers, Stats](https://openrouter.ai/anthracite-org/magnum-v4-72b)**)：这是一个旨在复制 Claude 3 系列模型（特别是 Sonnet）文本质量的模型系列。通过 API 运行 Magnum v4 72B
- [Claude 3.5 Sonnet (2024-06-20) - API, Providers, Stats](https://openrouter.ai/anthropic/claude-3.5-sonnet-20240620)：Claude 3.5 Sonnet 提供优于 Opus 的能力，速度快于 Sonnet，且价格与 Sonnet 相同。通过 API 运行 Claude 3.5 Sonnet (2024-06-20)
- [Claude 3.5 Sonnet (2024-06-20) (self-moderated) - API, Providers, Stats](https://openrouter.ai/anthropic/claude-3.5-sonnet-20240620:beta)：Claude 3.5 Sonnet 提供优于 Opus 的能力，速度快于 Sonnet，且价格与 Sonnet 相同。通过 API 运行 Claude 3.5 Sonnet (2024-06-20) (自我调节版)

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1298361551924756513) (272 messages🔥🔥):

> - `OpenRouter Model Updates`
> - `API Key Usage`
> - `Prompt Caching`
> - `Tool Use in Models`
> - `Website Access Issues`

- **OpenRouter 模型更新**：讨论重点关注了新版 Sonnet 3.5 模型的发布以及对现有用户的影响，用户正在不同版本之间进行切换。
  
  - 用户被告知模型的 API 名称保持不变，这意味着当前的实现可能已经在使用新版本。
- **API Key 使用**：多位用户提到了使用 OpenRouter 与直接使用供应商 Key 之间的 API 成本差异，一些用户报告了意外费用。
  
  - 提醒用户注意理解不同模型在 OpenRouter 下如何产生费用的重要性。
- **Prompt Caching**：用户对 Prompt Caching 的功能表示担忧，几位用户注意到它在某些模型上似乎无法正常工作。
  
  - 有建议称 Prompt Caching 在应用切换到新模型版本之前已经过测试。
- **模型中的 Tool Use**：用户表示有兴趣在默认不使用工具调用的模型中选择性地集成 Tool Use，并寻求实现策略方面的建议。
  
  - 提出了关于 OpenRouter 对 “tool” 角色的支持以及在不同模型中如何有效实现工具调用的问题。
- **网站访问问题**：几位用户报告访问 OpenRouter 网站时遇到困难，一些用户在切换不同浏览器时遇到加载问题。
  
  - 一段时间后，用户确认网站已恢复正常运行。

**提到的链接**：

- [全栈与 Web3 开发者](https://daniel0629.vercel.app)：我是一名资深的区块链和全栈开发者，在设计和实现复杂的去中心化应用及 Web 解决方案方面拥有丰富经验。
- [聊天室 | OpenRouter](https://openrouter.ai/chat)：LLM 聊天室是一个多模型聊天界面。添加模型并开始聊天！聊天室将数据本地存储在您的浏览器中。
- [迈向复现 OpenAI o1 的一小步](https://medium.com/@peakji/a-small-step-towards-reproducing-openai-o1-b9a756a00855)：Steiner 开源模型进展报告
- [OpenRouter](https://openrouter.ai/docs/limits>)：LLM 路由与市场
- [Aider LLM 排行榜](https://aider.chat/docs/leaderboards/)：LLM 代码编辑能力的定量基准测试。
- [Prompt 缓存 | OpenRouter](https://openrouter.ai/docs/prompt-caching)：优化 LLM 成本，最高可降低 90%
- [Claude 3.5 Sonnet - API、提供商、统计数据](https://openrouter.ai/anthropic/claude-3.5-sonnet-this-version-doesnt-exist)：全新的 Claude 3.5 Sonnet 提供优于 Opus 的能力、快于 Sonnet 的速度，且价格与 Sonnet 持平。通过 API 运行 Claude 3.5 Sonnet
- [Cocomelon Jj Cocomelon GIF - Cocomelon Jj Cocomelon 掉牙歌 - 发现并分享 GIF](https://tenor.com/view/cocomelon-jj-cocomelon-loose-tooth-song-wiggle-wiggle-is-it-ready-gif-16776506)：点击查看 GIF
- [Meta: Llama 3.1 70B Instruct – 提供商状态](https://openrouter.ai/meta-llama/llama-3.1-70b-instruct/providers)：查看提供商状态并向 Meta: Llama 3.1 70B Instruct 发起负载均衡请求 - Meta 最新的模型系列 (Llama 3.1) 推出了多种尺寸和版本。这款 70B 指令微调...
- [模型 | OpenRouter](https://openrouter.ai/models?order=newest&supported_parameters=tools))：在 OpenRouter 上浏览模型
- [快速入门 | OpenRouter](https://openrouter.ai/docs/quick-start)：开始使用 OpenRouter 进行构建
- [OpenRouter](https://openrouter.ai/terms)：LLM 路由与市场
- [Discord - 充满乐趣与游戏的群聊](https://discord.co)：Discord 非常适合玩游戏和与朋友闲逛，甚至可以建立全球社区。自定义您的专属空间来聊天、玩耍和聚会。
- [OpenRouter](https://openrouter.ai/)：LLM 路由与市场
- [活动 | OpenRouter](https://openrouter.ai/activity)：查看您在 OpenRouter 上使用模型的情况。
- [OpenRouter](https://openrouter.ai/docs/quick)：LLM 路由与市场
- [OpenRouter](https://openrouter.ai/api/v1/anthropic/)：LLM 路由与市场
- [密钥 | OpenRouter](https://openrouter.ai/settings/keys)：管理您的密钥或创建新密钥
- [请求 | OpenRouter](https://openrouter.ai/docs/requests#tool-calls)：处理传入和传出请求
- [OpenRouter](https://openrouter.ai/api/v1)：LLM 路由与市场
- [AI SDK Core: 工具调用](https://sdk.vercel.ai/docs/ai-sdk-core/tools-and-tool-calling)：了解如何使用 AI SDK Core 进行工具调用。
- [模型 | OpenRouter](https://openrouter.ai/models?max_price=0)：在 OpenRouter 上浏览模型
- [概念指南 | 🦜️🔗 Langchain](https://js.langchain.com/docs/concepts/#tools)：本节包含对 LangChain 核心部分的介绍。
- [GitHub - mem0ai/companion-nextjs-starter](https://github.com/mem0ai/companion-nextjs-starter?tab=readme-ov-file)：通过在 GitHub 上创建账号，为 mem0ai/companion-nextjs-starter 的开发做出贡献。

---

### **OpenRouter (Alex Atallah) ▷ #**[**beta-feedback**](https://discord.com/channels/1091220969173028894/1277894087755829278/1298408655342796841) (10 条消息🔥):

> - `自定义提供商密钥的 Beta 测试访问权限`
> - `集成设置访问请求`

- **自定义提供商密钥处于 Beta 阶段**：自定义提供商密钥目前处于 Beta 测试阶段，访问请求通过特定的 Discord 频道进行。
  
  - *不支持自行注册*，但成员可以私信其 **OpenRouter** 电子邮件地址以获取访问权限。
- **多个集成访问请求**：多位成员对集成设置的 Beta 访问权限表示了兴趣，并询问如何操作。
  
  - 一位成员提到，*“你好，我也需要集成设置的 Beta 访问权限”*，这反映了普遍的访问需求。

---

### **LM Studio ▷ #**[**general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1298361461088718899) (154 条消息🔥🔥):

> - `在 LM Studio 中下载模型`
> - `LLM 在编程方面的局限性`
> - `模型量化方法`
> - `模型中 System Prompts 的问题`
> - `使用 Vision Models 进行图像标注`

- **在 LM Studio 中下载模型**：用户讨论了在 LM Studio 中查找和下载某些大型模型的困难，特别提到了 Nvidia 的 70B Nemotron 模型，并提供了使用终端命令进行下载的说明。
  
  - 一些用户注意到搜索功能发生了变化，在搜索过程中需要点击特定的快捷键才能访问更大的模型。
- **LLM 在编程方面的局限性**：用户对各种专注于编程的 LLM 的性能表示失望，提到像 Mistral 和 Llama 3.2 这样的模型无法为编程任务生成准确的结果。
  
  - 大家一致认为 GPT-3.5 和 GPT-4 在编程方面的表现明显更好，这促使用户考虑替代工具。
- **模型量化方法**：讨论内容包括对不同量化方法（Q2, Q4, Q8）与模型性能关系的偏好，用户分享了这些方法如何影响模型有效性的经验。
  
  - 意见不一，一些用户警告不要使用 Q2，而另一些用户则表示，对于较大的模型尺寸，某些模型在较低比特量化下可能表现更好。
- **模型中 System Prompts 的问题**：一位用户遇到了模型无法识别 System Prompts 的问题，这引发了新的排查思路，包括临时转向 Prompt Templating。
  
  - 对话涉及对 LM Studio 内部配置设置的调整，以绕过某些不支持传统 System Prompts 的模型的限制。
- **使用 Vision Models 进行图像标注**：一位用户试图利用特定的 Vision Model 自动进行图像标注，但在模型与 LM Studio 的兼容性方面面临挑战。
  
  - 有建议寻找该模型的 GGUF 量化版本，尽管有人指出所需的模型目前可能与 Llama.cpp 不兼容。

**提到的链接**：

- [Better Florence 2 - a Hugging Face Space by SkalskiP](https://huggingface.co/spaces/SkalskiP/better-florence-2)：未找到描述
- [lmstudio-community/Llama-3.1-Nemotron-70B-Instruct-HF-GGUF · Hugging Face](https://huggingface.co/lmstudio-community/Llama-3.1-Nemotron-70B-Instruct-HF-GGUF)：未找到描述
- [mlx-community/xLAM-7b-fc-r · Hugging Face](https://huggingface.co/mlx-community/xLAM-7b-fc-r)：未找到描述
- [GGUF in details](https://medium.com/@charles.vissol/gguf-in-details-8a9953ac7883)：在训练阶段之后，基于 Llama.cpp 架构的模型可以使用 GGUF (GPT-Generated Unified Format) 格式进行交换。
- [bababababooey/llama-3.2-11b-vision-instruct-stheno-abliterated · Hugging Face](https://huggingface.co/bababababooey/llama-3.2-11b-vision-instruct-stheno-abliterated)：未找到描述
- [Prompt Template - Configuration | LM Studio Docs](https://lmstudio.ai/docs/configuration/prompt-template)：编辑 Prompt Template
- [Sideload models - Advanced | LM Studio Docs](https://lmstudio.ai/docs/advanced/sideload)：使用在 LM Studio 之外下载的模型文件

---

### **LM Studio ▷ #**[**hardware-discussion**](https://discord.com/channels/1110598183144399058/1153759714082033735/1298395439628030015) (13 messages🔥):

> - `Ryzen AI 配置`
> - `Intel Lunar Lake NPU 支持`
> - `Llama.cpp 中的 NPU 支持`
> - `RX 7900 XTX 与 RTX 3090 对比`
> - `ROCm 中的多 GPU 支持`

- **Ryzen AI 和 NPU 使用中的问题**：一位成员询问如何配置 LM Studio 以利用 **Ryzen 处理器** 的 **NPU** 而不是 **RTX 4060**，但在使 NPU 正常工作方面面临挑战。
  
  - 目前尚不确定 LM Studio 是否支持 **NPU**。
- **对 Intel Lunar Lake NPU 支持的兴趣**：一位用户询问 **Intel Lunar Lake NPU** 是否对 LM Studio 有用或受其支持。
  
  - 目前，**llama.cpp** 唯一支持的 **NPU** 是 **Ascend NPU**。
- **ROCm 更新中的多 GPU 支持**：**ROCm 6.1.3** 的更新改进了多 GPU 支持，据称可增强多达 **四个合格 GPU** 的处理能力。
  
  - 关于不同 GPU 品牌之间 **多 GPU 利用率** 的有效性报告褒贬不一，并存在关于 **NVIDIA** 兼容性的疑问。
- **RX 7900 XTX 与 RTX 3090 的对比**：一位用户就 **LLM 使用** 场景下如何在 **RX 7900 XTX** 和 **RTX 3090** 之间做出选择寻求建议。
  
  - 其他人建议，为了在 AI 应用中获得无缝体验，NVIDIA 的 **CUDA 支持** 是首选。
- **AI GPU 的性能见解**：讨论指出，对于主要关注 **LLM** 的用户来说，考虑 **CUDA 支持** 对于获得最佳性能至关重要。
  
  - 几位成员一致认为，由于其 **CUDA 功能**，**RTX 3090** 将提供更好的体验。

**提到的链接**：

- [AMD 在最新的 ROCm 更新中增强了多 GPU 支持：支持多达四个 RX 或 Pro GPU，并为 Pro W7900 Dual Slot 添加了官方支持](https://www.tomshardware.com/pc-components/gpus/amd-enhances-multi-gpu-support-in-latest-rocm-update-up-to-four-rx-or-pro-gpus-supported-official-support-added-for-pro-w7900-dual-slot)：新更新还包括对 Microsoft 的 Windows Subsystem for Linux 的 Beta 版支持。
- [llama.cpp/docs/build.md at master · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md#cann)：C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号为 ggerganov/llama.cpp 的开发做出贡献。
- [Issues · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/issues)：C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号为 ggerganov/llama.cpp 的开发做出贡献。

---

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1298360457584971899) (145 条消息🔥🔥):

> - `新版 Sonnet 3.5 模型`
> - `Perplexity 网页搜索问题`
> - `API 模型用户体验`
> - `更新反馈`
> - `自定义 RAG 系统使用`

- **用户抱怨新版 Sonnet 3.5**：多位用户对新版 Sonnet 3.5 模型表示不满，指出内容输出显著减少，尤其是在学术写作任务中。
  
  - 用户对旧版模型的移除表示担忧，认为旧版在多种使用场景下表现更优。
- **网页搜索集成问题**：有报告称 Spaces 中的 preprompt 在启用网页搜索时无法正常工作，导致用户感到沮丧。
  
  - 用户表示该问题一直存在，团队已确认并表示正在努力修复。
- **模型版本讨论**：一些用户询问是否可以选择回退到在特定场景下表现更好的旧版本。
  
  - 回复暗示系统的策略是始终迁移到最新的可用模型，而不保留旧版本。
- **寻求退款和拒付**：几位用户讨论了由于对近期更新后的服务不满而寻求退款或发起拒付（chargebacks）的可能性。
  
  - 有观点认为，是否有表现更好的模型可用将影响他们是否继续使用该服务。
- **用于学术用途的自定义 RAG 系统**：用户讨论了使用 Llama 和其他框架为学术任务构建个人 RAG 系统，发现它们比 Perplexity 目前提供的产品更有效。
  
  - 一位用户解释了他们如何使用本地运行的系统来查询教科书和往届考试，强调了其提供的灵活性和控制力。

**提到的链接**：

- [来自 TestingCatalog News 🗞 (@testingcatalog) 的推文](https://x.com/testingcatalog/status/1849191714134843612?s=46)："Buy with Pro" 这就是你们在 2025 年购物的方式 👀👀👀
- [来自 Aravind Srinivas (@AravSrinivas) 的推文](https://x.com/AravSrinivas/status/1849134236525347225)：⌘ + ⇧ + P 版本明天发布。它将成为你无需打开 Chrome 标签页即可询问任何问题的快捷方式。我们将更多地投入到面向本地桌面的生产力改进中！

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1298376357813948497) (17 条消息🔥):

> - `高级 AI 驱动的事实核查`
> - `轨迹记忆法（Method of Loci）`
> - `航空工业中的组件 MRO`
> - `Nvidia TSMC AI 联盟`
> - `RPA 中的 Claude 计算机控制`

- **探索高级 AI 驱动的事实核查**：一个关于 [AI 驱动的事实核查](https://www.perplexity.ai/collections/advanced-ai-driven-fact-checki-a3cMcPR.QsKkCRZ79UKFLQ) 的集合讨论了使用 LLM 的创新技术，包括来源可信度评估和偏见检测。
  
  - 它强调了诸如需要**透明度**和**人工监督**以有效打击虚假信息等挑战。
- **轨迹记忆法登上 YouTube**：一段 [YouTube 视频](https://www.youtube.com/embed/k4R6iBvOEk0) 介绍了 **轨迹记忆法 (Method of Loci)**，这是一种通过空间记忆帮助提高回忆能力的记忆增强技术。
  
  - *探索这种经典的助记手段如何增强你的记忆能力！*
- **组件 MRO 对航空业的影响**：关于航空工业中 [组件 MRO](https://www.perplexity.ai/search/what-is-component-mro-in-aircr-r0cr7t8uSCqulXRJWfLs1Q#0) 的信息揭示了其在维护和运营效率方面的关键作用。
  
  - 该系统有助于简化库存管理并减少停机时间。
- **Nvidia 与 TSMC 的 AI 联盟关系紧张**：一篇关于 [Nvidia 与 TSMC 合作伙伴关系](https://www.perplexity.ai/page/nvidia-tsmc-ai-alliance-strain-MoGqt8XuQfaHfw63v71uow) 的文章详细阐述了其 AI 芯片生产中的复杂问题。
  
  - 有人对**资源分配**和**制造能力**影响交付时间表表示担忧。
- **Claude 计算机使用模型为 RPA 敲响警钟**：一篇关于 [Claude 计算机控制能力](https://www.perplexity.ai/page/claude-s-computer-control-capa-E_O4xa7VSWOi3lGtOWnnMw) 的帖子暗示了对 **机器人流程自动化 (RPA)** 的潜在不利影响。
  
  - 专家警告称，这项创新可能会对现有流程构成重大风险。

 

**提到的链接**：[YouTube](https://www.youtube.com/embed/k4R6iBvOEk0)：未找到描述

 

---

### **Perplexity AI ▷ #**[**pplx-api**](https://discord.com/channels/1047197230748151888/1161802929053909012/1298461870189187082) (3 messages):

> - `Account Credit Transfer Issues` (账户余额转移问题)
> - `Server Errors` (服务器错误)

- **账户余额仍未转移**：一位用户报告称，尽管已联系支持团队，其 **account credits** 仍未转移。
  
  - *过去三天支持团队未予回复*，令用户感到沮丧。
- **频繁的 524 服务器错误**：另一位用户指出，他们全天都在**持续遇到 524 错误**。
  
  - 此前还提到了 **500 错误**，表明可能存在影响多个用户的服务器问题。

 

---

### **GPU MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1189498205101109300/1298365974658945128) (6 messages):

> - `LLM Activations Quantization` (LLM 激活值量化)
> - `bf16 vs fp32 Precision` (bf16 与 fp32 精度)
> - `Anyscale Inference Engine` (Anyscale 推理引擎)
> - `CUDA Streams and Synchronization` (CUDA 流与同步)

- **关于 LLM 激活值量化的辩论**：一位新手询问，对输入变化敏感的 **activations in LLMs** 是否不应进行激进量化，建议保持现状或采用更高精度的量化。
  
  - *这引发了对建模性能的担忧*，以及量化中精度权衡的问题。
- **对 bf16 精度的担忧**：一位成员表示，**bf16** 和 **fp32** 具有相似的数值范围，但 **bf16** 的精度问题可能会导致在多次梯度累积后出现**更新取消**的情况。
  
  - *对精度的担忧至关重要*，特别是当它影响模型训练稳定性时。
- **Anyscale 的单 Kernel 推理**：一位成员分享道，他们在 **Anyscale** 的朋友正在开发一种推理引擎，该引擎仅使用单个 **CUDA kernel** 即可完成整个 LLM 推理。
  
  - 他们征求关于这种方法与传统推理引擎相比的意见，并强调了*效率上的潜在飞跃*。
- **关于 CUDA 流同步的问题**：一位用户询问，为什么在特定的 CUDA 代码中，在 kernel 启动之前没有对 **stream1** 和 **stream2** 调用 **cudaStreamSynchronize** 函数。
  
  - *需要对 CUDA 同步技术进行澄清*，这突显了理解 kernel 执行顺序的重要性。

 

**提到的链接**：[cuda-course/05_Writing_your_First_Kernels/05 Streams/01_stream_basics.cu at master · Infatoshi/cuda-course](https://github.com/Infatoshi/cuda-course/blob/master/05_Writing_your_First_Kernels/05%20Streams/01_stream_basics.cu)：通过在 GitHub 上创建账户，为 Infatoshi/cuda-course 的开发做出贡献。

 

---

### **GPU MODE ▷ #**[**triton**](https://discord.com/channels/1189498204333543425/1189607595451895918/1298538475083272216) (47 条消息🔥):

> - `Triton kernels performance`
> - `Kernel caching strategies`
> - `Kernel launch overhead`
> - `Dynamic input shapes in Triton`
> - `Use of heuristics in Triton`

- **Autotuning Triton kernels vs aten.mm**：一位用户正在对 Triton kernels 进行 autotuning，以将其性能与 **aten.mm** 进行比较，并指出有时 Triton 更快，但整体性能表现不一致。
  
  - 另一位成员质疑如果 kernels 更慢，Triton 的优势在哪里，从而引发了关于缓存和 kernel 启动问题的讨论。
- **对 Kernel 启动开销的担忧**：成员们注意到 Triton 中存在显著的 kernel 启动开销（launch overhead），特别是在处理较小 tensors 时，因此建议探索缓存策略以缓解该问题。
  
  - 一个相关的 GitHub issue 强调了过长的启动时间，并建议在 autotune 中启用 `use_cuda_graph=True`，尽管这不适用于动态输入尺寸。
- **通过缓存 kernels 提升性能**：用户分享了缓存 kernels 以提高性能的策略，发现消除每次调用的 JIT 编译带来了显著的速度提升。
  
  - 讨论内容包括如何为不同的输入尺寸实现缓存，并确保 shapes 兼容以避免地址对齐错误（misaligned address errors）。
- **关于动态输入尺寸的讨论**：一位成员分享了在输入 tensor 尺寸非静态时使用缓存的局限性，指出了在管理启动元数据（launch metadata）和 kernel 参数方面的挑战。
  
  - 参与者辩论了输入 shape 的可变性对性能和 kernel 编译时间的影响。
- **用于 kernel 优化的 Heuristics**：一位成员建议使用 Triton 的 heuristics 来管理元参数（meta-parameters），这可以简化针对不同 tensor 尺寸的 kernel tuning。
  
  - 虽然有些人认为这种方法很有前景，但其他人指出了其复杂性，并建议采用其他缓存 kernels 的方法，以免影响性能。

**提到的链接**：

- [triton.heuristics — Triton documentation](https://triton-lang.org/main/python-api/generated/triton.heuristics.html)：未找到描述
- [High kernel launch overhead · Issue #2637 · triton-lang/triton](https://github.com/triton-lang/triton/issues/2637)：团队你好，我正面临很高的 Triton kernel 启动开销。这是我的 nsys 采集结果：kernel 在 GPU 上执行大约需要 80us，但启动却需要 220us，这导致了性能下降...

---

### **GPU MODE ▷ #**[**torch**](https://discord.com/channels/1189498204333543425/1189607750876008468/1298401208859365376) (7 条消息):

> - `torch.autocast usage with autoquant`
> - `Library linking with c10`
> - `Speeding up PyTorch models`
> - `Compiling PyTorch code`
> - `Autocast behavior with FP32 and BF16`

- **关于 torch.autocast 和 torchao 的困惑**：一位成员对在 **torchao autoquant** 中使用 autocast 是否会干扰其逻辑表示怀疑，因为传统上 autocasting 有助于避免精度损失。
  
  - *有人建议 autocast 可能会引入开销，* 特别是在小 batch 中，这使得 FP16/BF16 转换在推理中可能更高效。
- **将库链接到 c10**：一位成员在将 header-only 库链接到 **c10** 时遇到问题，尽管尝试在 CMake 配置中添加它。
  
  - 他们提到尝试使用 `target_link_libraries`，但未能解决问题。
- **寻求 PyTorch 的通用加速策略**：另一位成员询问了在保持准确性的同时增强 **PyTorch** 模型性能的策略，并建议将 Tensor Cores 和 autoquant 作为选项。
  
  - 讨论表明，使用 **autocast** 可能并非必要，可以优先选择较低精度类型以最小化计算开销。
- **对编译 PyTorch 代码的好奇**：一位用户询问了编译 **PyTorch** 代码的各种可用方法，提到了 **Glow**、**nvFuser** 和 **TorchDynamo** 等工具。
  
  - 该查询强调了需要明确这些编译器的不同用途。
- **怀疑 Autocast 对 torchao 量化的影响**：一位成员询问是否应该将 **autocast** 与 **torchao** 一起使用，并寻求有关其对量化影响的更多信息。
  
  - 他们假设由于 autocast 通常用于 FP16/BF16（提供更高的动态范围），它不应该对 **torchao** 产生负面影响。

 

**提到的链接**：[pytorch/aten/src/ATen/AccumulateType.h at b24c34426f47d6c311ad80ebba1d2575e6c7a6aa · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/b24c34426f47d6c311ad80ebba1d2575e6c7a6aa/aten/src/ATen/AccumulateType.h#L58-L70)：Python 中具有强大 GPU 加速能力的 Tensors 和动态神经网络 - pytorch/pytorch

 

---

### **GPU MODE ▷ #**[**algorithms**](https://discord.com/channels/1189498204333543425/1189861061151690822/) (1 条消息):

.alphago: [https://x.com/detectiveenters/status/1752067011113546234](https://x.com/detectiveenters/status/1752067011113546234)

---

### **GPU MODE ▷ #**[**cool-links**](https://discord.com/channels/1189498204333543425/1189868872887705671/1298648282037485682) (3 条消息):

> - `Mapper Generation Automation`
> - `Weight Nowcaster Networks`
> - `Neuron Interaction and Nowcasting`
> - `Training Optimization`
> - `Graph Neural Networks`

- **自动化 Mapper 生成优于专家**：一种用于并行编程的自动化 Mapper 生成新方法，据报道在科学应用中发现的 Mapper 性能优于人类专家设计，在十分钟内实现了高达 **1.34 倍的加速**。
  
  - 该方法解决了生成最佳映射方案通常所需的数百万个决策优化的复杂性。
- **利用 NiNo 改进神经网络训练**：最近提出的 **Neuron Interaction and Nowcasting (NiNo)** 网络增强了权重临近预报网络，通过结合神经元连接和 **Graph Neural Networks**，对之前的方法进行了改进。
  
  - 这一进展使 NiNo 能够加速 **Adam** 训练，解决了在某些网络（特别是 **Transformers**）中遇到的局限性。

**提到的链接**：

- [Improving Parallel Program Performance Through DSL-Driven Code Generation with LLM Optimizers](https://arxiv.org/abs/2410.15625)：将计算映射到处理器并将数据分配到内存对于最大化并行编程性能至关重要。这些映射决策通过开发专门的...
- [Accelerating Training with Neuron Interaction and Nowcasting Networks](https://arxiv.org/abs/2409.04434)：当使用可学习的更新规则代替经典的自适应优化器（如 Adam）时，神经网络训练可以加速。然而，可学习的更新规则的训练成本可能很高且不稳定...

---

### **GPU MODE ▷ #**[**beginner**](https://discord.com/channels/1189498204333543425/1191300313928433664/1298365202848551034) (3 条消息):

> - `CUDA Projects for Internships`
> - `Interactive Environments for CUDA Kernels`

- **实习 CUDA 项目丰富**：一位成员分享了他们有兴趣在暑期实习简历中加入 **CUDA 加速的线性回归和逻辑回归**，并寻求项目建议。
  
  - *他们的朋友笑了笑*，并将他们引导至一个 [服务器链接](https://link.to.server) 以获取更多想法。
- **使用 Jupyter 探索交互式 Kernel**：另一位成员询问了关于 **CUDA Kernel 的交互式环境**，建议使用 Cython 或 Jupyter notebooks 来运行 C 代码并在 Python 中操作输出。
  
  - *他们表示这种设置似乎是*最现实的方案，并就该问题寻求他人的意见。

 

---

### **GPU MODE ▷ #**[**pmpp-book**](https://discord.com/channels/1189498204333543425/1194427148656721970/) (1 条消息):

vim410: 这整章都在重写。一旦我拿到新章节就会分享出来。

---

### **GPU MODE ▷ #**[**torchao**](https://discord.com/channels/1189498204333543425/1205223658021458100/1298584546526101585) (7 messages):

> - `int4_mm` 为 `torch.compile` 提供的封装
> - `torch.library.custom_op` vs `@register_meta`
> - 非 Linux 平台的 `nightly wheels`

- **探索用于 `torch.compile` 的 `int4_mm` 封装**：一位成员对 `int4_mm` 如何封装以支持 `torch.compile` 感到好奇，怀疑它应该使用 `custom_op`，但发现它实际上使用了 `@register_meta` ([链接](https://github.com/pytorch/pytorch/blob/8fbf866904661b16cba4c799af81121557ba9da8/torch/_meta_registrations.py#L3275))。
  
  - 提出了 *这种方法相比 `torch.library.custom_op` 有什么优势吗？* 的疑问，并讨论了 `torch._check` 在检查 Tensor 大小和 dtype 方面的作用。
- **`torch.library.custom_op` 与低级 API 的区别**：成员们讨论了 `torch.library.custom_op` 作为一个高级封装器的作用，而某些情况可能会从低级 API 中受益 ([链接](https://github.com/pytorch/pytorch/blob/c2d26418c39f9562e128efae32eace61c703ccd7/torch/_library/custom_ops.py))。
  
  - 针对在已有直接选项时是否仍要使用高级 API 的询问，引发了关于此类自定义实现必要性的辩论。
- **非 Linux 平台 Nightly Wheels 的限制**：一位成员指出，除 Linux 以外的平台的 nightly wheels 缺少 `.dev<date>` 版本 ([链接](https://download.pytorch.org/whl/nightly/torchao/))。
  
  - 针对这一观察结果，有人建议 *“很有意思，介意开一个 issue 吗？”*，这表明了对更广泛平台支持的潜在关注。

**提到的链接**：

- [pytorch/torch/_meta_registrations.py at 8fbf866904661b16cba4c799af81121557ba9da8 · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/8fbf866904661b16cba4c799af81121557ba9da8/torch/_meta_registrations.py#L3275)：Python 中具有强大 GPU 加速功能的 Tensor 和动态神经网络 - pytorch/pytorch
- [无标题](https://download.pytorch.org/whl/nightly/torchao/)：未找到描述
- [pytorch/torch/_library/custom_ops.py at c2d26418c39f9562e128efae32eace61c703ccd7 · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/c2d26418c39f9562e128efae32eace61c703ccd7/torch/_library/custom_ops.py)：Python 中具有强大 GPU 加速功能的 Tensor 和动态神经网络 - pytorch/pytorch
- [pytorch/torch/_meta_registrations.py at 8fbf866904661b16cba4c799af81121557ba9da8 · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/8fbf866904661b16cba4c799af81121557ba9da8/torch/_meta_registrations.py#L47-L57)：Python 中具有强大 GPU 加速功能的 Tensor 和动态神经网络 - pytorch/pytorch

---

### **GPU MODE ▷ #**[**rocm**](https://discord.com/channels/1189498204333543425/1233704710389764236/1298465956309241928) (8 messages🔥):

> - `ROCm` vs `ROCm/Triton/FA` 基准测试
> - `RCCL` 贡献
> - `PyTorch` 对称内存 (symmetric memory)
> - 使用 `Triton` 进行矩阵乘法

- **ROCm/Triton/FA 基准测试咨询**：一位成员询问 **ROCm/Triton/FA** 是否比 **ROCm/FA** 更快，并对基准测试数据表示好奇。
  
  - 另一位成员提到他们还没有检查过，但如果有人运行基准测试，他们愿意合并相关 PR。
- **对 RCCL 贡献的兴趣**：一位成员表达了帮助开发 **RCCL** 的兴趣。
  
  - 这种热情得到了频道内其他人的支持和认可。
- **关于 PyTorch 对称内存的讨论**：有人提出了关于 **PyTorch symmetric memory** 的问题，并参考了其与之前讨论的关系。
  
  - 得到了直接确认，澄清了该话题与正在进行的开发者讨论之间的联系。
- **使用 Triton 实现矩阵乘法**：一位用户询问是否有简便的方法使用 **Triton** 实现矩阵乘法，以充分利用具有两个 GCD 的 **MI250x**。
  
  - 这反映了旨在通过 Triton 增强性能的持续探索和实现。

---

### **GPU MODE ▷ #**[**liger-kernel**](https://discord.com/channels/1189498204333543425/1275130785933951039/1298376433298837577) (8 条消息🔥):

> - `Monkey Patching CrossEntropy`
> - `Version Compatibility Challenges`（版本兼容性挑战）
> - `Liger's Kernel Fusion`（Liger 的 Kernel Fusion）
> - `Inference Optimization Techniques`（推理优化技术）
> - `Gradient Accumulation Fix Discussion`（梯度累积修复讨论）

- **CrossEntropy 的 Monkey Patching 面临挑战**：在 transformers 中针对 **CrossEntropyLoss** 的当前 Monkey Patching 策略在最新的 **GA patch 版本**中可能无效，因为 **CausalLMs** 已转向使用 `self.loss_function`。
  
  - 根 **CrossEntropy** 函数可以在[这里](https://github.com/huggingface/transformers/blob/049682a5a63042f087fb45ff128bfe281b2ff98b/src/transformers/loss/loss_utils.py#L26)找到。
- **Liger 的 Kernel Fusion 潜力**：一位参与者指出，**Liger** 因其能够融合 Kernel 而大有裨益，但建议使用 **vllm/sglang** 以获得更好的优化策略，如 **paged attention** 和 **flash-decoding kernel**。
  
  - 他们强调了**推理**与**训练优化**之间的区别，突出了针对特定场景采用不同方法的重要性。
- **需要进行版本兼容性检查**：有建议称应检查 **HF transformers 版本**并实现两种不同的 Patch，因为框架最近的更改影响了 Monkey Patching 的能力。
  
  - 交流以承诺进一步研究并确保其保持**向后兼容（backward compatible）**而结束。
- **对未解决问题的担忧**：另一位成员对最新的 Patch 以及它是否确实缓解了现有 Bug 表示担忧，并引用了讨论梯度累积修复的 [这个 Pull Request](https://github.com/huggingface/transformers/pull/34283)。
  
  - 人们担心此修复可能无法充分解决用户报告的持续性问题，这展示了在大型框架中进行 Patch 的复杂性。

**提到的链接**：

- [transformers/src/transformers/loss/loss_utils.py at 049682a5a63042f087fb45ff128bfe281b2ff98b · huggingface/transformers](https://github.com/huggingface/transformers/blob/049682a5a63042f087fb45ff128bfe281b2ff98b/src/transformers/loss/loss_utils.py#L26)：🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的先进机器学习库。 - huggingface/transformers
- [Enable Gradient Accumulation fix across all models + trainer fully in forward() by muellerzr · Pull Request #34283 · huggingface/transformers](https://github.com/huggingface/transformers/pull/34283)：此 PR 做了什么？由于大多数用户仍希望开箱即用，此举将 loss kwargs 传递给其余模型，以便正确计算 causal loss。修复了 # (issue) 完全修复了 #34263 / f...

---

### **GPU MODE ▷ #**[**self-promotion**](https://discord.com/channels/1189498204333543425/1288557096404516945/1298662010179027045) (3 条消息):

> - `Triton kernels for quantization`（用于量化的 Triton kernels）
> - `NVIDIA Virtual Connect with Experts`
> - `cuDF and cuML`
> - `RAPIDS developers panel`（RAPIDS 开发者小组）

- **Triton Kernels 教程发布**：关于为量化编写 **Triton kernels** 的教程将于 2 周后在 GPU MODE 举行，不过目前没有之前活动的录像。
  
  - 更多详情可以在 [教程环节](https://discord.gg/sQ7zJ94M?event=1289331107745108079) 中找到。
- **即将举行的 NVIDIA Virtual Connect with Experts 活动**：下一场 **NVIDIA Virtual Connect with Experts** 活动定于 **太平洋时间 2024 年 10 月 25 日星期五上午 10 点**举行，重点关注 **cuDF 和 cuML**。
  
  - 鼓励参与者查看 [GitHub 上的活动页面](https://github.com/NVIDIA/accelerated-computing-hub/tree/main/connect-with-experts) 以获取更新，并在其网络中进行宣传。

 

**提到的链接**：[accelerated-computing-hub/connect-with-experts at main · NVIDIA/accelerated-computing-hub](https://github.com/NVIDIA/accelerated-computing-hub/tree/main/connect-with-experts)：NVIDIA 策划的通用 GPU 编程相关教育资源集合。 - NVIDIA/accelerated-computing-hub

 

---

### **GPU MODE ▷ #**[**🍿**](https://discord.com/channels/1189498204333543425/1298372518293274644/1298375128786276484) (54 条消息🔥):

> - `为高效 Kernel 创建 LLM`
> - `CUDABench 提案`
> - `社区署名与贡献`
> - `CUDA Kernel 优化技术`
> - `GPU 编程中的游戏开发类比`

- **为高效 Kernel 创建 LLM**：团队正致力于开发一个用于生成高效 CUDA Kernel 的 LLM，重点是在 NeurIPS 2024 之前创建一个用于解释 GPU 功能的 mega prompt。
  
  - 他们计划构建一个大型 Kernel 数据集，并公开进行所有工作，充分利用社区的输入。
- **CUDABench 提案**：博士生们提出了 CUDABench 提案，这是一个用于评估 LLM 的 CUDA 代码生成能力的标准化基准测试，旨在众包问题和想法。
  
  - 该设计鼓励与各种 DSL 兼容，同时保持对 torch inline CUDA Kernel 的关注。
- **社区署名与贡献**：该项目正作为一个社区署名的论文进行开发，贡献者有机会根据其贡献成为作者。
  
  - 目标是确保广泛的社区输入，特别是在数据集和编码工作方面。
- **CUDA Kernel 优化技术**：围绕针对目标加速器优化现有 Kernel 以及可能使用 Kernel Tuner 等中间工具展开了讨论。
  
  - 成员们表示，虽然更广泛的优化应该在路线图中，但最初的重点仍应放在生成高效的 CUDA 代码上。
- **GPU 编程中的游戏开发类比**：成员们将开发以 CUDA 为中心的工具的方法与游戏开发进行了类比，提倡从特定目标开始，并随着时间的推移完善抽象。
  
  - 这一类比反映了为 LLM 进行 GPU 编程如何能从首先关注最合乎逻辑的目标中受益，然后再扩展到其他架构。

**提到的链接**：

- [Examples — Hidet Documentation](https://hidet.org/docs/stable/hidet-script/examples/index.html#hidet-script-examples)：未找到描述
- [GitHub - KernelTuner/kernel_tuner: Kernel Tuner](https://github.com/KernelTuner/kernel_tuner)：Kernel Tuner。通过在 GitHub 上创建账号来为 KernelTuner/kernel_tuner 的开发做出贡献。
- [CUDABench Design](https://docs.google.com/document/d/1ZNvShNH44zuy3LwbRdMigGsuCzO4i5Yl2fgAaSDynTg/edit?usp=sharing)：CUDABench Anne Ouyang1 (aco@stanford.edu), Simon Guo1 (simonguo@stanford.edu) 1: 斯坦福大学 动机与问题陈述 高效的 CUDA Kernel 对于最大化性能至关重要...
- [CUDABench Problem Ideas Crowdsourcing](https://docs.google.com/forms/d/e/1FAIpQLSeiqz2bLreIKY8maWCaaNIU-aXC0MfOMOog0bwS5J_zzNaLVQ/viewform?usp=sf_link)：CUDABench 设计文档：https://docs.google.com/document/d/1ZNvShNH44zuy3LwbRdMigGsuCzO4i5Yl2fgAaSDynTg/edit?tab=t.0#heading=h.4qj5vtu1o7mr 人们对使用 LLM 生成 CUDA 产生了浓厚兴趣 ...
- [Project Popcorn 🍿 (1).pptx](https://docs.google.com/presentation/d/1ir6br01sVY5wLqUz-qz4OE4nMSJbQBSp/edit?usp=sharing&ouid=106222972308395582904&rtpof=true&sd=true)：Project Popcorn
- [TK + Monkeys + CUDAGen](https://docs.google.com/presentation/d/1JtxGXv80ciIne-bFxySZ25q0J2mAwsXlb9uuST9naqg/edit?usp=sharing)：ThunderKittens 一个简单的 AI Kernel 框架
- [Monkeys_for_Meta_v3.pptx](https://docs.google.com/presentation/d/14jlbVPyohnWuQgFikr74cnaj-mzoEMPT/edit?usp=sharing&ouid=111422880520483065413&rtpof=true&sd=true)：Large Language Monkeys: 通过重复采样扩展推理时计算 Brad Brown\*, Jordan Juravsky\*, Ryan Ehrlich\*, Ronald Clark, Quoc Le, Chris Ré, Azalia Mirhoseini
- [META KERNELS - Google Drive](https://drive.google.com/drive/folders/1nt2KcRRKb8YdySxkRxUu5PR4c7UPM_rK)：未找到描述

---

### **Nous Research AI ▷ #**[**announcements**](https://discord.com/channels/1053877538025386074/1145143867818119272/1298699185234645064) (1 条消息):

> - `Hermes 70B API`
> - `Revenue-sharing partnership` (收入共享合作伙伴关系)
> - `Hyperbolic launch` (Hyperbolic 发布)
> - `AI Inference Service` (AI 推理服务)

- **Hermes 70B API 在 Hyperbolic 上线**：**Hermes 70B API** 现已在 Hyperbolic 上提供，为开发者和企业提供了更广泛的 Large Language Models 访问权限。更多详情，请查看此处的公告 [here](https://x.com/hyperbolic_labs/status/1849130421885514231?s=46)。
  
  - 此次发布标志着在让每个人都能更轻松地使用强大的 AI 工具方面迈出了重要一步。
- **Nous Research 与 Hyperbolic 达成合作**：Hyperbolic 宣布与 Hermes 3 模型的创建者 **Nous Research** 达成 **收入共享合作伙伴关系**。此次合作旨在通过 Hyperbolic 的 AI Inference Service 推动共同收益。
  
  - 正如 Hyperbolic 团队所言，“*AI 的未来是协作的*”，强调了技术领域合作伙伴关系的力量。

**提到的链接**：[来自 Hyperbolic (@hyperbolic_labs) 的推文](https://x.com/hyperbolic_labs/status/1849130421885514231?s=46)：Hyperbolic 正在让每个人都能更轻松地使用 AI。今天，我们与 Hermes 3 大语言模型的创建者 @NousResearch 启动了收入共享合作伙伴关系。通过我们的 AI Inference S...

---

### **Nous Research AI ▷ #**[**general**](https://discord.com/channels/1053877538025386074/1149866623109439599/1298371469893697616) (119 条消息🔥🔥):

> - `Nous Research Forge`
> - `Hermes AI Censorship` (Hermes AI 审查)
> - `Claude Automation` (Claude 自动化)
> - `AI Role-Playing` (AI 角色扮演)
> - `Grunt Work Opportunities` (琐碎工作机会)

- **对 Forge 项目的热情**：成员们对 “Forge” 项目表达了热情，该项目在一段由 Nous Research 联合创始人 Karan 参与的 [YouTube 视频](https://www.youtube.com/watch?v=zmnzW0r_g8k&list=PLjOo65uE) 中得到了重点介绍。
  
  - 随后讨论了与该项目相关的 Knowledge Graph (知识图谱) 实现。
- **关于 Hermes AI 审查的辩论**：成员们辩论了 Hermes AI 的审查程度，一些人认为某些提供商在他们的 System Prompts 中嵌入了审查。
  
  - 另一些人则认为，使用适当的 System Prompts 可以产生一系列行为，这表明不同模型之间的审查程度各不相同。
- **Claude 的自动化功能**：讨论了 Claude 的自动化能力，特别是围绕光标移动和网页浏览的功能。
  
  - 成员们对审查影响功能的程度表示担忧，尽管有迹象表明它比其他模型更少“拘束”。
- **AI 角色扮演动态**：探索了 AI 角色扮演的动态，特别是 System Prompts 如何影响 AI 模型在各种场景下的响应。
  
  - 成员们讨论了如果以某些方式进行指令引导，模型可能会表现出混乱行为的潜力，从而挑战了固有审查的观点。
- **寻求琐碎工作**：一名用户表达了通过琐碎工作（如数据录入或标注）寻求立即就业的需求，并强调了他们的困境。
  
  - 社区表达了团结，并建议通过 Uber Eats 等平台寻找即时工作，同时讨论了财务挑战。

**提到的链接**：

- [Forge by Nous Research @ Nouscon 2024](https://www.youtube.com/watch?v=zmnzW0r_g8k&list=PLjOo65uEP4cYhV7c2whkhDfWy58XFj7yL&index=8&t=514s)：Nous Research 联合创始人 Karan 在 Nouscon 2024 上谈论我们即将推出的项目之一 “Forge”。
- [Forge by Nous Research @ Nouscon 2024](https://www.youtube.com/watch?v=zmnzW0r_g8k&list=PLjOo65uE)：Nous Research 联合创始人 Karan 在 Nouscon 2024 上谈论我们即将推出的项目之一 “Forge”。
- [anthropic-quickstarts/computer-use-demo/computer_use_demo/streamlit.py at main · anthropics/anthropic-quickstarts](https://github.com/anthropics/anthropic-quickstarts/blob/main/computer-use-demo/computer_use_demo/streamlit.py)：一系列旨在帮助开发者快速开始使用 Anthropic API 构建可部署应用程序的项目集合 - anthropics/anthropic-quickstarts

---

### **Nous Research AI ▷ #**[**ask-about-llms**](https://discord.com/channels/1053877538025386074/1154120232051408927/1298390407016878130) (20 条消息🔥):

> - `Claude 的新系统提示词 (System Prompt)`
> - `微调 Gemma 2 等模型`
> - `基于 Whisper 的翻译框架`
> - `Llama 模型量化`
> - `支持 JSON 数据解析`

- **Claude 系统提示词增强**：新版 Claude 增加了一个系统提示词，修复了“误导性关注 (misguided attention)”问题，这是由另一位提取该提示词的用户分享的。
  
  - *Claude 还努力澄清谜题限制，但有时会因疏忽而误解问题。*
- **提升模型自我反思能力**：用户注意到 Claude 的自我反思能力有所增强，在处理逻辑谜题时回答变得更加精炼。
  
  - *有一个幽默的例子，Claude 最初尝试错误地回答一个谜题，随后自行进行了纠正。*
- **探索 Whisper 流式传输解决方案**：一位用户正在寻找离线实时基于 Whisper 的翻译框架，引发了对 [whisper_streaming](https://github.com/ufal/whisper_streaming) 等热门仓库的讨论。
  
  - *另一个建议包括新的* [*moonshine*](https://github.com/usefulsensors/moonshine) *项目，为边缘设备提供快速的 ASR。*
- **关于小型开源 (OSS) 模型的讨论**：当被问及强大的小型开源模型时，推荐了 **Gemma 2** 和 **Qwen 2.5**，并注明了关于内存需求的规格。
  
  - *有人对模型处理半复杂 JSON 数据表示担忧，一些用户对在没有额外解析的情况下的表现表示不确定。*
- **Llama 模型量化查询**：用户讨论了寻找 Llama 3.2 的量化版本，但在从 GGUF 格式定位 safetensors 版本时遇到挑战。
  
  - *这引发了关于使用脚本将 GGUF 转换为 safetensors 的咨询，显示出对模型优化的持续关注。*

**提到的链接**：

- [GitHub - ufal/whisper_streaming: Whisper realtime streaming for long speech-to-text transcription and translation](https://github.com/ufal/whisper_streaming): 用于长语音转文本转录和翻译的 Whisper 实时流式传输 - ufal/whisper_streaming
- [GitHub - usefulsensors/moonshine: Fast and accurate automatic speech recognition (ASR) for edge devices](https://github.com/usefulsensors/moonshine): 适用于边缘设备的快速准确的自动语音识别 (ASR) - usefulsensors/moonshine

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/) (1 条消息):

feffy: p-hacking :P

---

### **Nous Research AI ▷ #**[**interesting-links**](https://discord.com/channels/1053877538025386074/1132352574750728192/1298367633045061692) (5 条消息):

> - `ZK Proofs 与 ChatGPT 所有权`
> - `ZK 技术的进展`
> - `Genmo AI 的运动质量`
> - `AI 生成中的 Prompt 遵循度`
> - `人类动作生成`

- **ZK Proofs 允许 ChatGPT 历史记录所有权**：来自 OpenBlock 的 Universal Data Protocol (UDP) 的最新应用赋予了 ChatGPT 用户拥有其聊天历史记录的权利，同时增强了开源模型可验证训练数据的可用性。
  
  - 这种方法标志着在改善 AI 训练中的数据溯源和互操作性方面迈出了重要一步。
- **ZK 技术加速证明生成**：一位成员澄清说，ZK Proofs 在服务器端需要几秒钟，由于来自 @zkemail 基础设施的进步，现在一些 UDP 证明所需时间不到一秒。
  
  - [这里](https://x.com/paulsengh/status/1846657020868677931)分享了一个示例，展示了 ZK 技术的飞速进展。
- **Genmo AI 提供无与伦比的运动质量**：Genmo AI 声称提供符合物理定律的逼真运动，能够创建细致入微的高质量动画。
  
  - 他们的技术承诺生成的视频与详细的文本 Prompt 具有卓越的对齐度，增强了用户控制力。
- **利用 Mochi 1 跨越恐怖谷**：Mochi 1 拥有生成连贯且流畅的人类动作和表情的能力，推向了逼真动画的边界。
  
  - 这一进步对于创建符合观众对人类运动预期的视频至关重要。

**提到的链接**：

- [来自 OpenBlock (@openblocklabs) 的推文](https://x.com/openblocklabs/status/1848805457290572199)：1/ 介绍 Proof of ChatGPT，这是基于 OpenBlock 的 Universal Data Protocol (UDP) 构建的最新应用。此 Data Proof 赋予用户拥有其 LLM 聊天历史记录的所有权，标志着一个重要的...
- [来自 Paul Sengh (@paulsengh) 的推文](https://x.com/paulsengh/status/1846657020868677931)：令人难以置信的是 ZK 技术进步如此之快——感谢 @zkemail 的基础设施，现在一些 UDP 证明只需不到一秒。快来试试：https://bridge.openblocklabs.com/
- [Genmo。最好的开源视频生成模型。](https://www.genmo.ai/)：Genmo 训练世界上最好的开源视频生成模型。在 Genmo 使用 AI 创建令人惊叹的视频。

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/) (1 条消息):

feffy: p-hacking :P

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1298363256087248946) (35 messages🔥):

> - `Chess Move Explainability` (国际象棋步法可解释性)
> - `Accusations of Cheating in Chess` (国际象棋作弊指控)
> - `LLMs Self-Explanation Accuracy` (LLMs 自我解释准确性)
> - `Molmo Vision Models` (Molmo 视觉模型)
> - `DINOv2 Understanding` (DINOv2 理解)

- **棋手与步法解释**：大多数顶尖国际象棋棋手可以解释引擎走法背后的**动机**，但他们在复杂局面下对走法进行排序的能力仍存疑问。
  
  - 人类与引擎对“理想走法”定义的区别，使得理解最优对弈变得更加复杂。
- **国际象棋社区的争议**：一位前世界冠军指控一名热门主播作弊，主要依据是其在直播解说中给出的解释。
  
  - 这一事件突显了关于步法解释有效性的持续讨论，以及评论员所面临的压力。
- **LLMs 及其自我解释**：人们对 LLMs 提供的自我解释的准确性表示担忧，特别是当它们缺乏上下文理解时。
  
  - 这一考量引发了对如何通过更好的训练数据来增强解释真实性的探索。
- **即将推出的 Molmo 视觉模型**：Molmo 项目预计将发布在 PixMo 数据集上训练的开源视觉语言模型，并提供多个 Checkpoints。
  
  - 这些模型旨在实现多模态模型中的 SOTA 性能，同时保持完全开源。
- **学习 DINOv2**：一位用户寻求理解 DINOv2 的资源，并被引导至相关的研究论文以获取更多信息。
  
  - 该论文概述了 DINOv2 背后的方法论，由该领域的专家撰写。

**提到的链接**：

- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)：自然语言处理领域在大数据量模型预训练方面的近期突破，为计算机视觉领域类似的 Foundation Models 开辟了道路。这些模型可以极大地……
- [allenai/Molmo-7B-D-0924 · Hugging Face](https://huggingface.co/allenai/Molmo-7B-D-0924)：未找到描述
- [Alignment Workshop - Been Kim - Alignment and Interpretability: How we might get it right](https://www.alignment-workshop.com/nola-talks/been-kim-alignment-and-interpretability-how-we-might-get-it-right)：演讲副本。感谢慷慨的介绍。很高兴来到这里做关于对齐（Alignment）和可解释性（Interpretability）的演讲。定义什么是价值对齐是一个很好的开始……

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1298423210039050322) (45 messages🔥):

> - `RoPE 2D Encoding` (RoPE 2D 编码)
> - `Building Open Source Datasets` (构建开源数据集)
> - `Transformers and LayerNorms` (Transformers 与 LayerNorms)
> - `Softmax Attention Adaptations` (Softmax Attention 适配)

- **关于 RoPE 扩展到 2D 的辩论**：成员们讨论了将 **RoPE** 扩展到 2D 时，是否应该摒弃复数而直接使用 2D 坐标。
  
  - 讨论中提出了关于相对位置编码的担忧，强调了在每个频率上同时使用 **cos** 和 **sin** 的必要性。
- **呼吁开源动作模态数据集**：讨论围绕对**开源动作模态数据集**的需求展开，建议从开源 Web 框架中挖掘测试用例可能会有所帮助。
  
  - 使用人工标注数据和 **puppeteer 脚本**来创建此类数据集的可行性得到了积极响应。
- **LayerNorm 对模型解释的影响**：讨论围绕训练不带 **LayerNorms** 的模型的潜力展开，提到虽然这可以消除不良的机械解释特性，但架构不必局限于微调。
  
  - 一些成员对在不影响下游应用的前提下提高模型可解释性的方法表现出兴趣。
- **创新 Softmax Attention 机制**：参与者建议探索 **softmax attention** 的适配方案以提高效率并防止机械解释问题，例如线性化或将方法转换为 **top-k**。
  
  - 该方法旨在保留功能完整性的同时增强可解释性，并允许实现质量更好的方法。

**提到的链接**：

- [Analyzing and Improving the Training Dynamics of Diffusion Models](https://arxiv.org/abs/2312.02696)：扩散模型目前凭借其在大型数据集上无与伦比的可扩展性主导了数据驱动的图像合成领域。在本文中，我们识别并纠正了导致训练不均和低效的几个原因……
- [Storybook: Frontend workshop for UI development](https://storybook.js.org/)：Storybook 是一个用于孤立构建 UI 组件和页面的前端工作坊。成千上万的团队将其用于 UI 开发、测试和文档编写。它是开源且免费的。

---

### **Eleuther ▷ #**[**interpretability-general**](https://discord.com/channels/729741769192767510/1052314805576400977/1298589776588443648) (2 messages):

> - `AI-Driven Observability Interface` (AI 驱动的可观测性界面)
> - `Interpretability Startups` (可解释性初创公司)
> - `Technical Demonstrations` (技术演示)
> - `Jacob Steinhardt`
> - `Sarah Schwettmann`

- **介绍 Monitor：AI 可观测性界面**：[Monitor](https://transluce.org/) 是一个 AI 驱动的界面，旨在帮助人类观察、理解和引导模型内部的计算，目标是提高可解释性。
  
  - 该项目由 **Jacob Steinhardt** 和 **Sarah Schwettmann** 领导，旨在为截至 **2024年10月23日** 的模型理解提供更好的工具。
- **对协作的感谢**：一位成员对其他人在整理 **baulab** 频道分享的信息时所做的贡献表示感谢。
  
  - 这种协作努力凸显了社区对推进 AI 可解释性讨论的承诺。

 

**提到的链接**：[Transluce](https://transluce.org/)：未找到描述

 

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1298439052672110693) (7 messages):

> - `simple_evaluate tasks` (simple_evaluate 任务)
> - `evaluating small models` (评估小模型)
> - `Pile tasks error` (Pile 任务错误)

- **对 `simple_evaluate` 任务感到好奇**：`simple_evaluate` 被认为支持所有任务，但一位成员询问了它对小模型的能力，特别是寻找像 Pile PPL 或 Lambada 这样的评估。
  
  - 另一位成员指出 <@981242445696221224> 和 <@1042521538919923763> 可能会对小模型评估的优质任务提供更多见解。
- **`pile_10k` 验证问题**：一位用户在运行 `pile_10k` 验证时报告了一个错误，并询问 Pile 任务是否确实受支持。
  
  - 一位成员确认托管提供商存在问题，并指出 Pile 任务的默认 URL 目前没有指向任何可用的内容。

 

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1298379047054606398) (49 messages🔥):

> - `Anthropic Mouse Generator`
> - `Ideogram Canvas Features` (Ideogram Canvas 功能)
> - `AI and Loneliness Epidemic` (AI 与孤独感流行病)
> - `Speculative Decoding in vLLM`
> - `New Meeting Automation Tools` (新的会议自动化工具)

- **Anthropic Mouse Generator 展示 AI Agent 能力**：一位同事展示了新的 **Anthropic Mouse Generator**，其自主安装和调试软件的能力给观察者留下了深刻印象。
  
  - 然而，它仍然需要特定的指令，并且在没有指导的情况下无法执行像下棋之类的任务。
- **Ideogram Canvas 与现有工具竞争**：围绕 **Ideogram Canvas** 的讨论强调了其创新功能，如 **Magic Fill** 和 **Extend**，使用户能够无缝编辑和组合图像。
  
  - 一些参与者认为，由于其先进的功能，它对 Canva 等平台构成了竞争威胁。
- **AI 在孤独感流行病中的角色**：一名 14 岁少年自杀的悲剧引发了人们对 AI 对孤独感影响的担忧，引发了关于心理健康和技术角色的讨论。
  
  - 参与者辩论了 AI 是可以作为连接工具，还是会加剧孤立感，对其有效性持有不同意见。
- **vLLM 中的 Speculative Decoding 增强**：一篇新博客讨论了 **vLLM** 中的 **speculative decoding**，这是一种利用小模型和大模型加速 Token 生成的技术。
  
  - 这种方法旨在提高性能并整合优化 AI 功能的新技术。
- **发布新的会议自动化工具**：一款名为 **agent.exe** 的新应用已经发布，允许用户使用 **Claude 3.5 Sonnet** 控制他们的电脑。
  
  - 这一进展标志着人们对 AI Agent 的兴趣日益浓厚，并期待在 2025 年实现更高的自动化和效率。

**提到的链接**：

- [iPhone 16 订单在 2024 年第四季度至 2025 年上半年期间被削减约 1000 万部；目前尚无证据表明 Apple…](https://medium.com/@mingchikuo/iphone-16-orders-cut-by-around-10-million-units-for-4q24-1h25-no-evidence-yet-that-apple-48c126a33bc6)：最新行业调查：
- [来自 .txt (@dottxtai) 的推文](https://x.com/dottxtai/status/1848783015222169726)：我们一直在与 @huggingface 合作，刚刚发布了 Outlines 结构化生成的 Rust 移植版。👉 编译速度更快 👉 轻量级库（艾特 @vllm_project）👉 多语言绑定...
- [来自 Chris Pedregal (@cjpedregal) 的推文](https://x.com/cjpedregal/status/1849118877642526966?s=46)：写作即思考。我们推出了 @meetgranola，因为我们不希望会议机器人替我们思考。事实证明，很多人也有同感。很高兴宣布由 ... 领投的 2000 万美元 A 轮融资。
- [来自 Character.AI (@character_ai) 的推文](https://x.com/character_ai/status/1849055407492497564)：我们对一位用户的悲剧性离世感到心碎，并向其家人表达最深切的哀悼。作为一家公司，我们非常重视用户的安全，并正在继续 ...
- [Speculative Decoding 如何将 vLLM 性能提升高达 2.8 倍](https://blog.vllm.ai/2024/10/17/spec-decode.html)：vLLM 中的 Speculative Decoding 是一种强大的技术，通过协同利用小型和大型模型来加速 Token 生成。在本博客中，我们将深入解析 vLLM 中的 Speculative Decoding...
- [来自 Kyle Corbitt (@corbtt) 的推文](https://x.com/corbtt/status/1849127639866626171?s=46&t=tMWvmS3OL3Ssg0b9lKvp4Q)：顺便提一下，新的 Claude 3.5 在编程方面也非常出色。这是我的第一个 Electron 应用，Claude + Cursor 能够持续在单个 ... 中跨多个文件构建复杂功能。
- [来自 Kyle Corbitt (@corbtt) 的推文](https://x.com/corbtt/status/1849124800838713844?s=46)：刚刚发布了 agent.exe，这是一个免费、开源的 Mac/Windows/Linux 应用，让你能够使用 Claude 3.5 Sonnet 来控制你的电脑！这是一个探索 API 并了解模型能力的有趣小项目...
- [来自 Andrew Wilkinson (@awilkinson) 的推文](https://x.com/awilkinson/status/1849216089676460122)：这简直太酷了：我制作了一个 Lindy (@getlindy) AI Agent，它会在每次会议前 30 分钟给我发短信简报。它会查看他们的 LinkedIn 个人资料以及我们最近的电子邮件以获取上下文。 ...
- [Ideogram Canvas, Magic Fill 和 Extend](https://about.ideogram.ai/canvas)：Ideogram Canvas 是一个无限的创意画布，用于组织、生成、编辑和组合图像。将您的面部或品牌视觉效果带入 Ideogram Canvas，并使用行业领先的 Magic Fill 和 Ext...
- [Joel Lewenstein - 追求雄心勃勃的设计理念](https://share.snipd.com/episode/062696e2-2976-4fac-83f1-2941488c7fbf)：Joel Lewenstein - 追求雄心勃勃的设计理念
- [来自 James Grugett (@jahooma) 的推文](https://x.com/jahooma/status/1848401531491783135)：我刚刚辞掉了 10 万美元年薪的工作，加入了首届秋季 YC 营。我们正在构建一个 @cursor_ai 的竞争对手！听起来很熟悉？这一次，代码全是我们要自己写的 😉 有问必答！引用 Y C...
- [anthropic-quickstarts/computer-use-demo 在 main 分支 · anthropics/anthropic-quickstarts](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo)：旨在帮助开发者快速上手使用 Anthropic API 构建可部署应用程序的项目集合 - anthropics/anthropic-quickstarts
- [介绍 Act-One | Runway](https://youtu.be/z3F0ei62Kmk)：介绍 Act-One。一种在 Gen-3 Alpha 中使用单个驱动视频和角色图像生成富有表现力的角色表演的新方法。无需动作 ...
- [“他本可以还在这里”：遗孀称一男子在与 AI 聊天机器人交谈后自杀身亡](https://www.vice.com/en/article/man-dies-by-suicide-after-talking-with-ai-chatbot-widow-says/)：该事件引发了人们对快速普及的对话式 AI 模型安全护栏的担忧。
- [使用支持 GPT3 的聊天机器人缓解学生的孤独感和自杀倾向 - npj Mental Health Research](https://www.nature.com/articles/s44184-023-00047-6)：未找到描述
- [Character.ai 在青少年自杀后面临诉讼 - 纽约时报](https://archive.fo/2zp1e)：未找到描述

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1298366825679163512) (4 messages):

> - `Llama Impact Hackathon`
> - `Box AI integration`
> - `Multi-agent concierge system`
> - `Building LLM-powered web apps with LlamaIndex.TS`

- **加入 Llama Impact Hackathon 寻找 AI 解决方案**：参加 11 月 8 日至 10 日在旧金山举行的为期 3 天的 [Llama Impact Hackathon](https://t.co/G01c8eIN1j)，总奖金池为 **$15,000**，其中包括为最佳使用 LlamaIndex 设立的 **$1000** 特别奖。
  
  - 活动提供线下和线上两种形式，使用 **Meta 的 Llama 3.2 模型** 构建 AI 解决方案。
- **Box AI 与 LlamaIndex 无缝协作**：利用 **Box AI** 无需下载即可查询文档，并从非结构化内容中提取结构化数据，同时可以轻松地将其与 LlamaIndex agents 集成。
  
  - 了解更多关于 [Box AI](https://t.co/M9f81GiMGp) 如何通过协作方式与 LlamaIndex 共同增强你的工作流。
- **构建高级客服机器人**：一项新更新允许你构建一个 **多 Agent 礼宾系统**，该系统集成了工具调用 (tool calling)、记忆 (memory) 以及专为客服应用量身定制的人机协作。
  
  - 正如 [Logan Markewich](https://t.co/PWshlAyeKV) 所强调的，这些经过彻底改进的功能使用户能够有效地迭代和改进他们的客服机器人。
- **使用 LlamaIndex.TS 开发 LLM 驱动的应用**：LlamaIndex.TS 现在已成为 [Vercel AI SDK](https://t.co/BgCvo2Rxj6) 的一部分，提供了一种更简单的方法，只需一行代码即可将响应流式传输回前端。
  
  - 它允许开发者在 Node.js 等流行运行时中创建 **LLM 驱动的应用程序**，并提供集成示例。

 

**提到的链接**：[Adapters: LlamaIndex](https://t.co/BgCvo2Rxj6)：了解如何将 LlamaIndex 与 Vercel AI SDK 结合使用。

 

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1298372897689174068) (31 messages🔥):

> - `Persistent Context in Workflows`
> - `Analyzing Long YouTube Lectures`
> - `Using Anthropic LLM`
> - `Progress Stats in SimpleDirectoryReader`
> - `LlamaIndex Workflow Compatibility with Redpanda`

- **Workflow 中的持久化上下文**：一位用户询问如何使 **Context** 在同一个 Workflow 的多次运行中保持持久化，引发了关于序列化选项的讨论。
  
  - 回复中包含了使用 **JsonSerializer** 进行序列化的示例，允许在以后恢复上下文。
- **高效分析长篇 YouTube 讲座**：一位成员讨论了构建一个分析长篇 YouTube 讲座的工具，并面临管理大上下文窗口 (context sizes) 的挑战。
  
  - 建议包括对上下文进行总结，或实施 **基于检索的方法 (retrieval-based approach)** 以保持效率。
- **迁移到 Anthropic LLM**：另一位用户尝试用 Anthropic LLM 替换默认的 ChatGPT，但在 OpenAI API key 提示方面遇到了问题。
  
  - 回复提到，要完全过渡到 Anthropic，需要一个本地 Embedding 模型，以避免依赖 OpenAI 的 Embedding。
- **SimpleDirectoryReader 中的进度统计**：用户询问在摄取多个 PDF 时，是否能在 **SimpleDirectoryReader** 中显示进度或时间统计。
  
  - 讨论指出，虽然没有直接的时间统计，但进度条可以显示已处理的 PDF 数量。
- **LlamaIndex Workflow 与 Redpanda 的兼容性**：一位用户询问 **LlamaIndex Workflow** 是否可以与 Redpanda 集成，或者是否需要 Confluent。
  
  - 回复表示对 Redpanda 的兼容性尚不确定，暗示缺乏相关的第一手经验。

 

**提到的链接**：[Gemini 1.5 Pro 2M context window, code execution capabilities, and Gemma 2 are available today](https://developers.googleblog.com/en/new-features-for-the-gemini-api-and-google-ai-studio/)：未找到描述。

 

---

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1298466750416818258) (5 messages):

> - `Tensor int64/uint64 支持`
> - `SHA3 实现`
> - `Tensor.ones 与 numpy`

- **关于 Tensor int64/uint64 支持的澄清**：*Tensor 是否还不支持 int64/uint64？* 这个问题是在讨论实现 SHA3 时提出的。
  
  - 另一位成员确信地表示**已支持**，并建议查看 `dtype.py` 进行确认。
- **Tensor.ones 与 reshaping 错误**：一位用户报告了在使用 `print(Tensor.ones(5, 5, dtype=dtypes.int64).numpy())` 时遇到的问题，收到了一个 **ValueError**，提示无法将大小为 50 的数组重塑（reshape）为形状 (5,5)。
  
  - 这引发了关于这是 bug 还是仅仅是**尚未支持**的疑问，因为该用户刚开始接触 tinygrad。
- **在不使用 numpy 的情况下运行**：一位成员提到成功在不使用 numpy 的情况下执行了类似任务，尽管他们承认使用了 AI，并指出这“有点不被看好”。
  
  - 尽管如此，他们强调在 tinygrad 中肯定是可以实现的。

---

### **tinygrad (George Hotz) ▷ #**[**learn-tinygrad**](https://discord.com/channels/1068976834382925865/1070745817025106080/1298421329749479489) (29 messages🔥):

> - `Action Chunking Transformers 训练`
> - `TinyJIT 挑战`
> - `JIT 函数输入/输出要求`
> - `运行更快的 kernel`
> - `逆向工程字节码`

- **对 Action Chunking Transformers 训练时间的沮丧**：拥有 **5500 万**参数的 Action Chunking Transformers 在没有 JIT 的情况下训练需要 **2 天**，这引发了关于性能改进的讨论。
  
  - 分享了关于最小化长推理时间以及 JIT 训练期间重复 **loss 参数**问题的想法。
- **TinyJIT Loss 参数打印困惑**：用户讨论了在 JIT 函数中打印 loss 时使用 `.item()` 的挑战，并检查了其对正确显示数值的影响。
  
  - 建议包括避免返回非 Tensor，因为这可能对 JIT 功能产生不良影响。
- **关于 JIT 函数输入和输出的见解**：澄清指出，经过 JIT 处理的函数的输入和输出理想情况下应该是已实现的 Tensor，而使用 `dict` 结构进行组织也是可以接受的。
  
  - 澄清了 JIT 执行模型与函数定义中非 Tensor 逻辑之间的关系，这对于保留可执行路径至关重要。
- **通过 BEAM 设置缩短训练时间**：建议使用 `BEAM=2` 运行以增强长时间训练期间的性能，这可能会带来更快的 kernel 搜索。
  
  - 反馈指出，这种方法已经被成功用于加速训练过程。
- **探索逆向工程 AI 加速器字节码**：一位用户表达了对逆向工程 AI 加速器字节码的兴趣，并寻求方法论和初步测试技术方面的指导。
  
  - 这激起了成员们对有助于启动逆向工程过程的工具和框架的好奇心。

**提到的链接**：

- [快速入门 - tinygrad 文档](https://docs.tinygrad.org/quickstart/?h=jit#jit)：未找到描述
- [act-tinygrad/train.py at main · mdaiter/act-tinygrad](https://github.com/mdaiter/act-tinygrad/blob/main/train.py#L60)：Tinygrad 中的 Action Chunking Transformers。通过在 GitHub 上创建账户为 mdaiter/act-tinygrad 的开发做出贡献。
- [tinygrad/tinygrad/engine/jit.py at master · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/engine/jit.py#L174>)：你喜欢 PyTorch？你喜欢 micrograd？你会爱上 tinygrad！❤️ - tinygrad/tinygrad

---

### **Interconnects (Nathan Lambert) ▷ #**[**news**](https://discord.com/channels/1179127597926469703/1179128538679488533/1298400643001880670) (14 messages🔥):

> - `Claude AI 体验`
> - `持续预训练（Continuous Pretraining）讨论`
> - `Character.AI 用户悲剧`
> - `安全功能实施`
> - `MIT Technology Review 播客`

- **Claude AI 的乐趣**：一位成员分享了他们使用 **Claude AI** 的兴奋之情，提到它非常有趣，并计划在第二天发送示例。
  - 这让人们对他们体验的更详细见解产生了期待。
- **持续预训练问题**：一位成员询问 **GPT-4o** 是从零开始使用 **200k 词表分词器（vocabulary tokenizer）** 进行预训练的，还是在从 **100k** 分词器切换后继续进行预训练的。
  - 另一位成员评论说，训练中期（mid-training）非常混乱，反映了在确定训练过程方面仍然存在挑战。
- **Character.AI 的哀悼**：Character.AI 对一位用户的悲剧性离世表示哀悼，并强调在此处添加了新的安全功能 [here](https://blog.character.ai/community-safety-updates/)。
  - 一位成员链接了一篇 [《纽约时报》的文章](https://www.nytimes.com/2024/10/23/technology/characterai-lawsuit-teen-suicide.html)，强调了这一声明背后的严峻背景。
- **对安全性的担忧**：一位成员对安全措施的有效性表示怀疑，认为尽管公司做出了努力，悲剧性的结果可能仍会继续发生。
  - 这反映了人们对快速发展的 AI 技术所产生的社会影响的广泛担忧。
- **MIT Technology Review 播客**：一位成员对 [MIT Technology Review 播客节目](https://podcasts.apple.com/us/podcast/mit-technology-review-narrated/id1523584878?i=1000674111360) 中讨论的系统可能导致的不良后果表示担忧。
  - 他们对 AI 发展似乎正在遵循的轨迹表示悲伤，将其比作社交媒体那种快速且具有冲击力的路径。

**提到的链接**：

- [来自 Character.AI (@character_ai) 的推文](https://x.com/character_ai/status/1849055407492497564)：我们对一位用户的悲剧性离世感到心碎，并向其家人表示最深切的哀悼。作为一家公司，我们非常重视用户的安全，并将继续……
- [让我们能与已故亲人“交谈”的技术已经出现。我们准备好了吗？](https://podcasts.apple.com/us/podcast/mit-technology-review-narrated/id1523584878?i=1000674111360)：播客节目 · MIT Technology Review Narrated · 2024/10/23 · 27分钟

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-drama**](https://discord.com/channels/1179127597926469703/1181746144821387334/) (1 messages):

xeophon.: [https://milesbrundage.substack.com/p/why-im-leaving-openai-and-what-im](https://milesbrundage.substack.com/p/why-im-leaving-openai-and-what-im)

---

### **Interconnects (Nathan Lambert) ▷ #**[**memes**](https://discord.com/channels/1179127597926469703/1187551504995987576/1298374595371335743) (5 messages):

> - `Jeremy Howard 的推文`
> - `Tek 的能量水平`
> - `Hermes 3 性能`

- **Michael 的推文被动漫头像账号 Ratio**：来自 [Jeremy Howard](https://x.com/jeremyphoward/status/1848813387242999847) 的一条著名推文强调了 **Microsoft** 的 CEO 是如何被一个使用*动漫头像*的用户 Ratio（评论数远超点赞数）的。
- **Tek 的“愤怒男人”阶段仍在继续**：有人分享了关于 **Tek** 处于“愤怒男人阶段”已有数月的看法，许多人都注意到了他的能量。
  - 虽然一位用户并不喜欢这种能量，但他们承认这似乎引起了很多人的共鸣。
- **潜在的 Hermes 3 分数更新**：正如一位成员提到的，有人猜测可能会在下一篇论文中“干掉（nuke）” **Hermes 3** 的分数。
  - 这引起了人们对排行榜当前排名的关注。

**提到的链接**：[来自 Jeremy Howard (@jeremyphoward) 的推文](https://x.com/jeremyphoward/status/1848813387242999847)：Microsoft 的 CEO 被一个动漫头像账号 Ratio 了……

---

### **Interconnects (Nathan Lambert) ▷ #**[**posts**](https://discord.com/channels/1179127597926469703/1228051082631188530/1298640266693181520) (11 条消息🔥):

> - `Anthropic 对 B2B 的关注`
> - `Anthropic 与 Simular 之间的成本对比`
> - `自动化与 AI Agent`
> - `AI 演示中的 Microsoft vs OpenAI`

- **Anthropic 转向 B2B，而 OpenAI 瞄准消费者**：一位成员指出，**Anthropic** 正在演变为一家 B2B 公司，这与 **OpenAI** 以消费者为中心的重点形成鲜明对比，并认为自动化购物等任务的需求较低。
  
  - 这一观点引发了关于消费者对自动化枯燥任务与参与娱乐活动之间兴趣差异的讨论。
- **与 Simular 演示的成本对比**：一位成员回顾了 **Simular** 去年在 SPC 的演示，对它的成本如何与最近 YouTube 视频（[链接在此](https://www.youtube.com/watch?v=ld17uwuNBcY&t=25s)）中展示的 **Anthropic** 成本对齐表示好奇。
  
  - 最近的演示对比暗示了潜在的市场转变，并引发了关于融资和投资策略的问题。
- **Anthropic 的自动化专注于枯燥任务**：有人担心 **Anthropic** 推广的自动化主要围绕填表等乏味任务，虽然能大幅节省工作时间，但缺乏趣味性。
  
  - 成员们批评了这种方法，指出消费者通常不想自动化那些令人愉悦的体验，比如游戏。
- **Microsoft 引人入胜的 AI 演示**：**Microsoft** 展示了更多有趣的 AI 应用，例如在 **Minecraft** 中自动进行游戏，这与 Anthropic 演示中呈现的单调任务形成对比（[点击查看](https://youtu.be/TLg2KWY2J5c?si=dRkIA2t1Y0F61KgP)）。
  
  - Microsoft 对趣味性的关注与 Anthropic 对企业效率的强调之间的区别，突显了 AI 领域内不同的战略。
- **成员分享对 AI 公司发展方向的看法**：讨论触及了这样一个观点：某些公司似乎是出于必要而非对所在细分市场的真正热情而保持一致。
  
  - 一位成员表示希望自己能更早发现这些趋势，这表明了对 AI 公司战略动向的广泛关注。

**提到的链接**：

- [Simular @ 2023 December SPC demo faire (Spotlight)](https://www.youtube.com/watch?v=ld17uwuNBcY&t=25s)：未找到描述
- [Claude | Computer use for automating operations](https://youtu.be/ODaHJzOyVCQ?si=Lb1iOygMphHW9GJ5)：随着升级后的 Claude 3.5 Sonnet，我们正在 Beta 测试中引入一项新功能：computer use。开发者现在可以指示 Claude 像人一样使用电脑...
- [Copilot gpt4o preview](https://youtu.be/TLg2KWY2J5c?si=dRkIA2t1Y0F61KgP)：带有 gpt4o 预览版的 Copilot

---

### **OpenInterpreter ▷ #**[**general**](https://discord.com/channels/1146610656779440188/1147665339266650133/1298369919729270888) (29 条消息🔥):

> - `Screenpipe tool`
> - `Claude 3.5 model`
> - `Open Interpreter's development`
> - `AI integration on different OS`
> - `Efficient data extraction for AI`

- **Screenpipe 的构建日志令人印象深刻**：成员们称赞了 [Screenpipe](https://github.com/OpenInterpreter/open-interpreter/blob/development/examples/screenpipe.ipynb) 在管理构建日志方面的实用性，强调了它的潜力和有趣的落地页。
  
  - 一位用户指出使用该工具产生了重大影响，特别是对于寻找高效日志解决方案的开发者而言。
- **Anthropic 的 Claude 3.5 能力揭晓**：Anthropic 发布了 **Claude 3.5 Sonnet** 模型，该模型在编程方面有显著改进，并通过公开测试版（public beta）引入了全新的 **computer use** 能力，允许模型像人类一样与用户界面进行交互。
  
  - 这一新功能也带来了挑战，因为它需要不断的屏幕截图捕获，引发了关于效率和成本的担忧。
- **Open Interpreter 路线图讨论**：针对一些批评，成员们讨论了 Open Interpreter 的路线图（roadmap），并表示有信心用户会发现其独特能力相比主流操作系统中集成的 AI 产品更具价值。
  
  - 一些怀疑者质疑与成熟 AI 模型竞争的可行性，而其他人则强调了社区驱动开发的重要性。
- **AI 屏幕交互中的挑战**：关于使用屏幕截图作为 AI 模型输入的低效性提出了担忧，并建议直接从程序中提取必要的数据点以提高效率。
  
  - 成员们表示需要改进数据处理方法，以弥补屏幕截图依赖带来的已知局限性。
- **测试与 Anthropic 的新集成**：一位成员强调了引入 `interpreter --os` 命令用于集成 Anthropic 模型，并邀请其他人协助在正式发布前测试该新功能。
  
  - 测试表明，增加屏幕尺寸和文本大小有助于在使用该模型时降低错误率。

**提到的链接**：

- [Introducing computer use, a new Claude 3.5 Sonnet, and Claude 3.5 Haiku](https://www.anthropic.com/news/3-5-models-and-computer-use)：更新后的、更强大的 Claude 3.5 Sonnet，Claude 3.5 Haiku，以及一项新的实验性 AI 能力：computer use。
- [Claude Computer Use TESTED - This is VERY Promising!](https://www.youtube.com/watch?v=A5RfSftJRw8)：Claude Computer Use 测试 - 非常有前景！👊 成为 YouTube 会员以获取 GitHub 访问权限：https://www.youtube.com/c/AllAboutAI/join🔥Swarm GitHub 仓库：https:/...
- [open-interpreter/examples/screenpipe.ipynb at development · OpenInterpreter/open-interpreter](https://github.com/OpenInterpreter/open-interpreter/blob/development/examples/screenpipe.ipynb)：计算机的自然语言界面。通过在 GitHub 上创建账户为 OpenInterpreter/open-interpreter 的开发做出贡献。
- [Anthropic’s New AI Can Control Your Computer!](https://youtu.be/idipaHSpQes?t=225.)：Anthropic 发布了三件令人惊叹的事物：全新的 Claude 3.5 Sonnet，Claude 3.5 Haiku，以及允许模型控制你电脑的 "computer use"。加入我的新...

---

### **OpenInterpreter ▷ #**[**ai-content**](https://discord.com/channels/1146610656779440188/1149229778138824765/) (1 条消息):

facelessman: [https://youtu.be/VgJ0Cge99I0](https://youtu.be/VgJ0Cge99I0) -- 喜欢这一集 -- 喜欢这些人！！！

---

### **Cohere ▷ #**[**discussions**](https://discord.com/channels/954421988141711382/954421988783444043/1298387382856450129) (8 messages🔥):

> - `Multimodal Command model`
> - `Global connection`
> - `Aya Expanse`
> - `Complex bots`

- **关于多模态 Command 模型的推测**：*Paulm24* 询问了 **multimodal Command model** 的存在，暗示了对高级模型能力的日益关注。
  
  - *Karthik_99_* 插话表示，暗示它具有 **Global connection** 功能，表明了不同交互模式的融合。
- **对机器人复杂性的兴奋**：*Enzoloko* 对仅通过 prompting 就能创建 **complex bots** 的潜力表示热切期待，这得益于模型的内在特性。
  
  - 这反映了社区对于利用高级模型进行创新应用的广泛热情。
- **Aya Expanse 驱动 Cohere 机器人**：*Sssandra* 确认该机器人由 **Aya Expanse** 驱动，这引发了对其能力的关注。
  
  - 这一提及标志着 AI 领域的重大进展和潜在的探索机会。
- **社区参与趣味功能**：*Wolfybl* 和 *sssandra* 在讨论他们使用机器人的体验时，强调了趣味性和协作精神。
  
  - *Roazzy* 敦促分享乐趣，表明社区互动和体验是受到重视的。

---

### **Cohere ▷ #**[**questions**](https://discord.com/channels/954421988141711382/1168411509542637578/1298362022290919466) (7 messages):

> - `Cohere API trial and production keys`
> - `Using Command-Night in Portuguese`
> - `Feedback integration in fine-tuning LLM`
> - `Ollama Mistral performance issues`

- **Cohere 提供用于免费测试的试用 API 密钥**：Cohere 提供 [trial API key](https://docs.cohere.com/docs/rate-limits)，允许用户免费访问所有模型，但有一定的速率限制。
  
  - 例如，在试用期内，Chat endpoint 限制为 **每分钟 20 次调用**，而 production key 允许 **每分钟 500 次调用**。
- **探索 Command-Night 在葡萄牙语中的功能**：一位成员询问 Command-Night 是否支持葡萄牙语，因为其中包含的是 Aya 而不是 Light。
  
  - 讨论中未提供有关该工具多语言能力的见解。
- **用于微调 LLM 的创新反馈循环**：一位成员提出了一种迭代反馈机制，允许专家通过聊天 UI 集成修正来增强 LLM 性能。
  
  - 该方法涉及将反馈保存到 'accept.json' 和 'not_accept.json' 文件中，从而随着时间的推移实现更智能的 LLM 优化。
- **Ollama Mistral 性能方面的挑战**：一位成员对 Ollama Mistral 的 hallucination（幻觉）倾向和计算需求表示沮丧，这影响了他们项目的执行。
  
  - 尽管如此，他们还是强调了其方法的基本原理，即生成用于专家评估的 prompt 和 response，相关内容可在其 [GitHub gist](https://gist.github.com/pleabargain/8b3f1641ef727cc114ac389cbc1b354b) 中找到。

**提到的链接**：

- [API Keys and Rate Limits — Cohere](https://docs.cohere.com/docs/rate-limits)：此页面描述了 Cohere API 针对生产和评估密钥的速率限制。
- [using Ollama to interate over a source of truth and present prompts and responses to an expert](https://gist.github.com/pleabargain/8b3f1641ef727cc114ac389cbc1b354b)：使用 Ollama 迭代事实来源并向专家展示 prompt 和 response - 带有反馈的 ollama prompt 和 response 生成器.py

---

### **Cohere ▷ #**[**api-discussions**](https://discord.com/channels/954421988141711382/1168578329423642786/1298419458842431508) (9 messages🔥):

> - `Cohere V2 API Errors`
> - `Finetuned Models Issues`
> - `Vercel AI SDK Integration`
> - `Tool Use and Function Calling`

- **Cohere V2 API 中的内部服务器错误**：成员们报告在使用 Cohere V2 API 的聊天端点时遇到 **internal server errors**，特别是在涉及 **tool_calls** 的消息中。
  
  - 一位用户分享了导致失败的特定 payload，另一位成员请求提供代码片段以便更好地进行故障排除。
- **工具调用缺少必填字段**：讨论中提到了请求中缺少 **tool_plan field** 的问题，一名成员指出这可能是一个潜在问题。
  
  - 引用了 [Cohere documentation](https://docs.cohere.com/docs/tool-use#step-2) 中的一个示例，以说明工具集成的正确用法。
- **Vercel AI SDK 缺乏 Cohere V2 支持**：一位用户提到计划使用 **Vercel AI SDK** 集成 Cohere V2，但发现目前的 provider mapping 仅支持 V1。
  
  - 他们强调已向 Vercel 团队反映此问题，并引用了他们的 [GitHub issue](https://github.com/vercel/ai/issues/3331)，但目前仍不确定支持 V2 的时间表。
- **对微调模型功能的担忧**：一位用户询问其他人在通过 API 访问其 **finetuned models** 时是否也面临问题。
  
  - 这引发了关于当前设置下微调模型的稳定性和功能的讨论。

**提到的链接**：

- [Issues · vercel/ai](https://github.com/vercel/ai/issues/3331)：使用 React, Svelte, Vue 和 Solid 构建 AI 驱动的应用 - Issues · vercel/ai
- [Tool Use — Cohere](https://docs.cohere.com/docs/tool-use#step-2)：让您的 LLM 能够连接外部工具，以实现更高级和动态的交互 (V2)。

---

### **Cohere ▷ #**[**projects**](https://discord.com/channels/954421988141711382/1218409701339828245/1298368461776617534) (1 messages):

> - `Agentic Builder Day`
> - `Cohere Models`
> - `AI Agent hackathon`

- **11 月 23 日的 Agentic Builder Day**：OpenSesame 与 **Cohere** 团队合作，将于 11 月 23 日举办 **Agentic Builder Day**，邀请优秀的开发者展示他们的技能。
  
  - 参与者可以[立即申请参赛](https://www.opensesame.dev/hack)，在这场 **mini AI Agent hackathon** 中争取赢取奖品的机会。
- **征集优秀开发者**：该活动寻求有兴趣合作并竞争使用 **Cohere Models** 构建强大 AI Agent 的**优秀开发者**。
  
  - 对于开发者来说，这是一个在 AI 社区内建立联系、在竞争中提升技能的独特机会。

**提到的链接**：[OpenSesame | Build Better AI Agents](https://www.opensesame.dev/hack)：OpenSesame 简化了从构建到评估的整个 AI Agent 生命周期。我们的平台使企业能够轻松创建、共享和实施 AI Agent 并检测 hallucinations，让 AI...

---

### **Modular (Mojo 🔥) ▷ #**[**mojo**](https://discord.com/channels/1087530497313357884/1151418092052815884/1298496368326873181) (20 messages🔥):

> - `Community Meeting Discussions`
> - `Serial Communication in Mojo`
> - `C/C++ Support in Mojo`
> - `LED Matrix Communication`
> - `Framework Laptop 16`

- **参加周日的 stdlib 讨论**：一位成员在观看完上一次社区会议后，表达了加入关于 **stdlib contributor meetings** 周日讨论的热情。
  
  - 其他人表示“欢迎加入”，并鼓励在相关频道参与对话。
- **理解 Mojo 中的串口通信**：一位用户寻求帮助，想了解如何在 **Mojo** 中实现 **serial communication**，特别是通过端口。
  
  - 其他人澄清说，Mojo 目前仅提供 **libc** 所提供的功能，没有额外的支持。
- **C/C++ 支持咨询**：对话转向 Mojo 中是否存在 **C/C++ support**，并讨论了该支持的潜在用例。
  
  - 虽然它可能适用于少数用户，但该应用的适用性受到了质疑。
- **Framework Laptop 的 LED 矩阵通信库**：一位用户表示计划为 **Framework Laptop 16 LED Matrix communication** 创建一个库，表达了希望让更多系统能够访问的愿望。
  
  - 尽管最初存在疑虑，但大家对该项目的合作和改进持开放态度。

---

### **Modular (Mojo 🔥) ▷ #**[**max**](https://discord.com/channels/1087530497313357884/1212827597323509870/1298698487852171264) (3 messages):

> - `MAX Engine 的 C API`
> - `Mojo 的优势`
> - `C 语言中的 Graph Builder API`

- **MAX Engine 的 C API 现已上线！**：**C API** 已可用于 **MAX Engine**，但目前没有计划将其包含在 **graph API** 中。
  
  - 如果关于 **graph API** 有任何变动，将会分享更新。
- **Mojo 独特的图能力**：一名成员询问 **Mojo** 的主要优势是否在于其能够利用其他语言无法使用的 **graph API** 的独特能力。
  
  - 这突显了 **Mojo** 在当前 AI 领域中固有的优势和前景。
- **使用 C 构建 Graph API**：另一名成员指出，虽然 **Mojo** 用于图节点，但如果需要，也可以使用 C 创建 **graph builder API**。
  
  - 这开启了关于不同编程语言之间替代实现和协作的讨论。

 

---

### **Torchtune ▷ #**[**general**](https://discord.com/channels/1216353675241590815/1216353675744641096/1298388964813574184) (6 messages):

> - `调试运行命令`
> - `在多个 GPU 上进行测试`
> - `微调自定义模型`
> - `环境对比`

- **不要在 TorchTune 配置中使用 .yaml**：一名成员指出，在 **TorchTune** 配置的运行命令中使用 **.yaml** 文件扩展名是有问题的，因为这暗示正在提供本地配置。
  
  - *如果没有额外的错误消息，调试可能会令人沮丧*。
- **在 2 个 GPU 上测试脚本**：一位用户询问了在 **2 个 GPU** 上进行测试的能力，并就此功能提出了疑问。
  
  - 另一名成员报告了在使用 **lora_finetune_distributed** 在 **1 个 GPU** 和 **2 个 GPU** 上运行脚本时收到错误消息的问题。
- **可以使用 TorchTune 进行微调**：针对关于微调 **自定义 Llama** 模型的问题，一名成员确认 **TorchTune** 非常可定制并提供协助。
  
  - 他们鼓励进一步讨论模型的自定义组件，以获得量身定制的支持。
- **友好的社区氛围**：一位用户对社区的友好表示赞赏，强调了温馨的氛围。
  
  - 这样的评论有助于营造一个欢迎分享知识和支持的环境。

 

---

### **Torchtune ▷ #**[**dev**](https://discord.com/channels/1216353675241590815/1236040539409879170/1298644877793300520) (6 messages):

> - `Linter 和 Pre-commit Hook`
> - `CI 问题`
> - `Tokenizer 测试`

- **关于 Linter 和 Pre-commit Hook 的问题**：一名成员表达了对 **linter 和 pre-commit hooks** 的问题，提到它们并没有 **100%** 按预期工作。
  
  - 具体来说，他们注意到要忽略一行，需要同时使用 `# noqa` 和 `# fmt: on ... #fmt: off`，这看起来很**反常**。
- **PR #1868 中奇怪的 CI 行为**：另一名成员报告了 PR **#1868** 中 **CI** 的奇怪行为，并请求协助检查发生了什么。
  
  - 他们表示 CI 问题在每个 PR 中都持续存在，暗示调查正在进行中。
- **CI 修复状态更新**：一名成员询问最近的一个 CI 问题是否已解决，该问题正由另一名成员审查。
  
  - 回复指出问题现在应该已经**修复**，让团队感到安心。

 

---

### **LangChain AI ▷ #**[**general**](https://discord.com/channels/1038097195422978059/1038097196224086148/1298520313885229127) (3 messages):

> - `开发者项目调查`
> - `马尼拉的地理位置`
> - `FunctionMessages 与 LLM 响应`

- **助力打造开发者工具**：一位成员分享了一个[调查链接](https://forms.gle/Roi1U5ynVwLtQ3S46)，目标受众为开发者，旨在了解将想法转化为现实过程中的挑战，并指出完成调查大约需要 **5-7 分钟**。
  
  - 该调查涵盖了开发者产生想法的频率、面临的障碍，以及对简化项目实现解决方案的兴趣。
- **询问马尼拉开发者**：一位成员询问群组中是否有人位于**马尼拉**，可能是为了联系当地开发者。
  
  - 这一询问表明了在马尼拉开发者之间建立社区或进行协作的兴趣。
- **来自 joiner LLM 的详细回答**：一位成员询问如何从聚合响应的 joiner LLM 中获取**详细回答**，要求不进行摘要或缩减长度。
  
  - 他们担心当前使用 `FunctionMessages` 的实现会导致过于简短的摘要，而不是保留原始响应的细节。

**提及的链接**：[从灵感到发布：项目实现中的开发者挑战](https://forms.gle/Roi1U5ynVwLtQ3S46) ：开发者朋友们好！你是否充满了创新的想法，却在将其付诸实践时感到困难？你并不孤单。我们正在进行一项研究，以了解开发者在转化...

---

### **LangChain AI ▷ #**[**share-your-work**](https://discord.com/channels/1038097195422978059/1038097372695236729/1298570202757337108) (3 messages):

> - `AI 编程助手研究`
> - `AI 驱动的融资工具`
> - `ApeBrains 交易员专业化`

- **参与 AI 影响研究**：一项调查 AI 工具对软件工程影响的硕士研究正在征集开发者参与。通过填写一份简短的问卷，参与者有机会赢取 200 新西兰元礼品卡，同时为有价值的研究做出贡献。
  
  - 你可以[在此](https://auckland.au1.qualtrics.com/jfe/form/SV_0uf2q5Ie7V3gpvM?Source=43)访问问卷。
- **使用 AI 工具解锁融资**：一款 AI 驱动的工具已经发布，旨在通过将用户与相关的投资者和加速器联系起来，帮助他们为自己的想法获得融资。前 **200** 名加入候补名单的人将获得免费的**创业加速器礼包**，显著增强他们的搜索能力。
  
  - 目前仅剩 **62** 个名额，鼓励感兴趣的用户[立即注册](https://www.aloangels.me/)，让他们的创业愿景成为现实。
- **加入 ApeBrains Trader Alpha 计划**：ApeBrains 正在推广一个交易员专业化计划，为用户提供注册 **ApeBrains Alpha** 的机会。此外，还有一个推荐计划，允许参与者通过与朋友分享链接来获得优先访问权。
  
  - 鼓励用户访问 [ApeBrains](https://www.apebrains.com) 了解更多详情并享受促销优惠。

**提及的链接**：

- [ApeBrains Wallet Agents - 即将推出](https://www.apebrains.com)：专业化你的 ApeBrains 交易员。注册 ApeBrains Alpha。
- [在线调查软件 | Qualtrics 调查解决方案](https://auckland.au1.qualtrics.com/jfe/form/SV_0uf2q5Ie7V3gpvM?Source=43)：收集体验数据最强大、简单且值得信赖的方式。今天就开始您的体验管理之旅，尝试免费账户。
- [AloAngels: 免费 AI 驱动的投资者匹配](https://www.aloangels.me/)：未找到描述

---

### **LangChain AI ▷ #**[**tutorials**](https://discord.com/channels/1038097195422978059/1077843317657706538/1298410963791515753) (1 messages):

> - `GeoGuessr AI Bot`
> - `Vision LLMs`
> - `LangChain`
> - `Multimodal AI`

- **构建 AI GeoGuessr 玩家**：一个新的 [YouTube 教程](https://www.youtube.com/watch?v=OyDfr0xIhss) 展示了如何编写一个 AI 机器人，利用 **GPT-4o**、**Claude 3.5** 和 **Gemini 1.5** 等 **Multimodal Vision LLMs** 自主游玩 **GeoGuessr**。
  
  - 该教程涵盖了使用 **Python** 进行编码，并集成了 **LangChain**，使机器人能够截取屏幕截图并与游戏环境进行交互。
- **多模态 Vision LLMs 实战**：参与者讨论了在编程项目中结合使用 **Vision LLMs** 的情况，特别强调了它们在 **GeoGuessr** 等动态环境中的有效性。
  
  - 这凸显了多模态能力在 AI 应用（尤其是游戏领域）中日益增长的重要性。

**提到的链接**：[Coding a Vision LLM Agent that plays GeoGuessr by itself (GPT-4o, Claude 3.5 and Gemini 1.5)](https://www.youtube.com/watch?v=OyDfr0xIhss)：如何编写一个 AI 机器人，使用多模态 Vision LLMs 自动游玩 GeoGuessr 游戏，通过 Python + LangChain 截取游戏屏幕并进行操作...

---

### **DSPy ▷ #**[**show-and-tell**](https://discord.com/channels/1161519468141355160/1202371242519441499/1298382426329976986) (2 messages):

> - `Advanced Workflow System`
> - `Upgrade Process`

- **全球最先进的工作流系统正在开发中**：一名成员宣布他们正开始在专门频道中开发**全球最先进的工作流系统**。
  
  - 他们计划在周一进行**现场演示**，详细介绍当前系统的工作原理并讨论其升级流程。
- **即将举行的现场演示公告**：该成员确认**现场演示**定于周一举行，旨在解释工作流系统的运行方式及即将进行的升级。
  
  - 本次会议预计将提供有关目前正在进行的**升级流程**和改进的见解。

---

### **DSPy ▷ #**[**general**](https://discord.com/channels/1161519468141355160/1161519469319946286/1298397373734584405) (5 messages):

> - `DSPy Funding Potential`
> - `Synthetic Data Generation Metrics`

- **DSPy 瞄准雄心勃勃的融资目标**：一位成员建议，如果 CrewAI 能获得 **1800 万美元**，那么 **DSPy** 应该瞄准至少 **5000 万美元**，并表达了作为第 5 号或第 10 号员工加入的热情。
  
  - *“我们还在等什么？”* 是号召立即行动的共同情绪。
- **关于合成数据指标的讨论**：一位成员询问如何使用 **DSPy** 根据一段文本为 QA 创建合成数据，特别是询问了有效的评估指标。
  
  - 另一位成员回答说，对于没有标准答案（ground truth）的开放式生成，使用 **LLM as a judge** 结合预定义标准可能会很有效。
- **合成数据生成中的 Groundedness**：在生成合成数据的背景下，一位成员强调 **ground truth** 将来自用于生成的文本，并建议将 Groundedness 作为一种可能的指标。
  
  - 他们对该话题分享的见解表示感谢，表明成员之间正在进行持续协作。

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-questions**](https://discord.com/channels/1280234300012494859/1280370030609170494/1298465984876777512) (7 messages):

> - `LLM Agents MOOC Signup`
> - `Hackathon Project Open Sourcing`
> - `Agents Development Tutorials`

- **报名表单困惑**：一位成员表示在提交 [LLM Agents MOOC 报名表单](https://link-to-signup-form) 后没有收到每周邮件，从而引发了关于是否收到确认函的后续讨论。
  
  - 另一位成员分享了类似的经历，表示他们也没有收到关于被课程录取的正式反馈。
- **黑客松代码提交要求**：在黑客松的最终演示期间，成员们确认他们被要求将其项目代码 **100% 开源（open source）**。
  
  - 一位成员强调了提交代码以遵守活动规则的重要性。
- **对 Agent 创建教程的需求**：一位参与者询问是否存在从零开始制作 Agent 的教程，且除了 LLM 本身之外不使用任何平台。
  
  - 这表明用户对独立开发 Agent 的易用资源有需求。

---

### **LLM Finetuning (Hamel + Dan) ▷ #**[**general**](https://discord.com/channels/1238365980128706560/1238365980128706563/1298604483328348272) (2 条消息):

> - `Axolotl 配置`
> - `LangSmith Prompt Hub`
> - `Kaggle 解决方案`
> - `Hugging Face 数据集`
> - `用于消息抓取的 Discord 机器人`

- **利用 Axolotl Discord 获取配置**：你可以利用 🦎 Axolotl Discord 频道分享和查找针对你的用例量身定制的配置，以及 GitHub 上的示例文件夹。
  
  - 查看 [Discussions 标签页](https://github.com/axolotl-ai-cloud/axolotl/discussions) 以了解其他成员分享的类似用例。
- **在 LangSmith Prompt Hub 中探索 Prompt**：🛠️ LangSmith Prompt Hub 提供了适用于不同模型和用例的各种 Prompt 集合，丰富了你的 Prompt Engineering 工具箱。
  
  - 对于数据集，请探索 [Awesome Public Datasets 仓库](https://github.com/awesomedata/awesome-public-datasets) 中公开可用的数据集。
- **提供全面的 Kaggle 解决方案**：有一个名为 *The Most Comprehensive List of Kaggle Solutions and Ideas* 的集合，可以作为竞赛数据科学的有用资源。
  
  - 在 GitHub [此处](https://github.com/faridrashidi/kaggle-solutions) 找到各种各样的解决方案。
- **用于模型对齐的 Hugging Face Recipes**：关于持续预训练以及使语言模型与人类和 AI 偏好对齐，请参考 Hugging Face 上分享的强大 Recipes。
  
  - 在[此处](https://github.com/huggingface/alignment-handbook/tree/main/recipes)访问这些 Recipes。
- **用于消息抓取的新 Discord 机器人**：一位用户创建了一个 Discord 机器人来抓取频道中的消息，并正在寻求邀请该机器人的帮助。
  
  - 你可以通过此[链接](https://discord.com/oauth2/authorize?client_id=1298625427375656980&response_type=code&redirect_uri=https%3A%2F%2Fc123ian.github.io%2F&scope=messages.read)邀请该机器人。

**提到的链接**：

- [LangSmith](https://smith.langchain.com/hub): 未找到描述
- [GitHub - awesomedata/awesome-public-datasets: A topic-centric list of HQ open datasets.](https://github.com/awesomedata/awesome-public-datasets): 一个以主题为中心的高质量开放数据集列表。通过在 GitHub 上创建账户为 awesomedata/awesome-public-datasets 的开发做出贡献。
- [GitHub - faridrashidi/kaggle-solutions: 🏅 Collection of Kaggle Solutions and Ideas 🏅](https://github.com/faridrashidi/kaggle-solutions): 🏅 Kaggle 解决方案与创意集合 🏅。通过在 GitHub 上创建账户为 faridrashidi/kaggle-solutions 的开发做出贡献。
- [alignment-handbook/recipes at main · huggingface/alignment-handbook](https://github.com/huggingface/alignment-handbook/tree/main/recipes): 用于使语言模型与人类和 AI 偏好对齐的强大 Recipes - huggingface/alignment-handbook

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general**](https://discord.com/channels/1104757954588196865/1104757955204743201/1298391199937335306) (2 条消息):

> - `实验性 Triton FA 支持`
> - `Mixtral vs. Llama 3.2`

- **2.5.0 版本为 gfx1100 增加了实验性 Triton FA 支持**：在 **2.5.0** 版本中，可以使用环境变量 `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` 启用针对 **gfx1100** 的**实验性 Triton Flash Attention (FA)** 支持。
  
  - *UserWarning: Navi31 GPU 上的 Flash attention 支持仍处于实验阶段。* 更多详情可以在 [GitHub issue](https://github.com/ROCm/aotriton/issues/16#issuecomment-2346675491) 中找到。
- **关于 Mixtral 与 Llama 3.2 使用的辩论**：鉴于 **Llama 3.2** 的进步，有人提出了现在使用 **Mixtral** 是否可行的问题。
  
  - 社区正在权衡这两个选项的优缺点，以确定优先使用哪个模型。

 

**提到的链接**：[[Feature]: Memory Efficient Flash Attention for gfx1100 (7900xtx) · Issue #16 · ROCm/aotriton](https://github.com/ROCm/aotriton/issues/16#issuecomment-2346675491>): 建议说明：开始使用 torchlearn 在 PyTorch 中使用我的 gfx1100 显卡训练模型，但收到警告称 torch 编译时未包含内存高效的 Flash Attention。我看到有...

 

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #**[**discussion**](https://discord.com/channels/1111172801899012102/1111353033352294440/1298598678235451522) (2 条消息):

> - `模型评估故障排除`
> - `Handler 注册`
> - `模型生成命令`

- **模型评估时得分报告为空**：一位用户报告称，在添加新的模型 handler 并在 `handler_map.py` 中注册后，运行 `bfcl evaluate --model mynewmodel --test-category ast` 会产生一个空的得分报告，进度显示为 **0/0**。
  
  - 另一位成员建议确认之前是否执行了 `bfcl generate ...` 命令，暗示这对于正确评估可能是必要的。
- **评估前生成模型的重要性**：讨论了在模型评估之前运行 `bfcl generate` 命令以确保结果准确的必要性。
  
  - 这表明缺少模型生成可能会导致评估过程中出现得分报告为空等问题。

 

---

---

---

---

---

---

{% else %}

> 完整的频道明细已为邮件格式截断。
> 
> 如果您想查看完整明细，请访问此邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
> 
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预先感谢！

{% endif %}