---
companies:
- databricks
- mit
- google
- openai
- x-ai
- nous-research
- astribot
- apple
- sakana-ai
date: '2024-08-20T05:06:22.742788Z'
description: '**Omar Khattab** 宣布在前往麻省理工学院（MIT）任教前加入 **Databricks**，并概述了 **DSPy 2.5
  和 3.0+** 的路线图。其重点是改进语言模型（LM）、签名（signatures）、优化器（optimizers）和断言（assertions）等核心组件，包括采用
  **LiteLLM** 以减少代码量并增强缓存和流式传输功能。该路线图还包括开发更准确、更具成本效益的优化器，编写教程，以及实现交互式优化跟踪。


  在 AI 推特圈（AI Twitter）方面，**谷歌**推出了 **Gemini Live**，这是一款支持语音对话且拥有 10 种音色的移动端对话 AI，同时还发布了搭载定制
  Tensor A1 芯片的 **Pixel Buds Pro 2**。**OpenAI** 更新了 **ChatGPT-4o**，重新夺回了 LMSYS Arena
  排行榜的榜首。**xAI** 发布了 **Grok-2** 测试版，并凭借 FLUX 1 在图像生成领域达到了业界领先水平（SOTA）。**Nous Research**
  发布了开源的 **Hermes 3** 模型，提供 8B、70B 和 405B 三种尺寸，其中 405B 模型达到了 SOTA。机器人领域的动态包括 **Astribot**
  的人形机器人以及**苹果**公司支持 Siri 语音控制的桌面机器人。**Sakana AI** 推出了“AI 科学家”（The AI Scientist），这是一个自主的
  AI 科研系统。'
id: 56bd7e91-4e60-435f-a93b-7956e5414a24
models:
- dspy
- litel-lm
- gemini
- chatgpt-4o
- grok-2
- hermes-3
original_slug: ainews-the-dspy-roadmap
people:
- omar-khattab
- giffmana
title: DSPy 路线图
topics:
- model-optimization
- fine-tuning
- optimizers
- interactive-optimization
- robotics
- autonomous-systems
- voice
- image-generation
- open-source-models
- scientific-research
- streaming
- caching
---

 

路线图的主要方向：

1. **打磨 DSPy 核心的 4 个部分：(1) LMs, (2) Signatures & Modules, (3) Optimizers, 以及 (4) Assertions**，使它们能够开箱即用，实现 zero shot 且现成可用。

- 在 LMs 方面，他们的目标是减少代码行数。特别是他们提到将通过 [采用 LiteLLM 来减少 6k 行代码 (LOC)](https://x.com/yi_ding/status/1825601460664741922)。不过，他们将增加“改进的缓存、LMs 的保存/加载、对流式传输和异步 LM 请求的支持”等功能。
- 在 Signatures 方面，既然“结构化输出”已成为主流，他们正在演进“结构化输入”的概念。
- 在微调（Finetuning）方面：他们的目标是“为程序中的几个不同模块引导（bootstrap）训练数据，训练多个模型并处理模型选择，然后将这些模型加载并插入到程序的模块中”。

2. **开发更准确、成本更低的优化器（optimizers）。** 继 BootstrapFewShot -> BootstrapFinetune -> CA-OPRO -> MIPRO -> MIPROv2 和 BetterTogether 优化器之后，将开展更多工作来提高质量、成本和鲁棒性。

3. **构建端到端教程。** 更多文档！

4. **转向更具交互性的优化和追踪。** 帮助用户“实时观察优化过程（例如：分数、堆栈跟踪、成功和失败的追踪以及候选提示词）”。

虽然没有什么惊天动地的突破，但对于一个管理得非常好的开源框架来说，这是一个很棒的路线图更新。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

> 所有摘要均由 Claude 3.5 Sonnet 完成，从 4 次运行中择优录取。

**AI 与机器人进展**

- **Google Gemini 更新**：Google 推出了 [Gemini Live](https://twitter.com/adcock_brett/status/1825201770773012617)，这是一款具有语音功能和 10 种语音的移动对话式 AI，适用于 Android 上的 Gemini Advanced 用户。他们还推出了 [Pixel Buds Pro 2](https://twitter.com/adcock_brett/status/1825201853673488494)，搭载定制的 Tensor A1 芯片以支持 Gemini 功能，实现免提 AI 辅助。

- **OpenAI 进展**：[OpenAI 更新的 ChatGPT-4o 模型](https://twitter.com/adcock_brett/status/1825201876423340041) 重新夺回了 LMSYS Arena 榜首，该模型以“anonymous-chatbot”代号测试了一周，获得了超过 1.1 万张选票。

- **xAI 的 Grok-2**：[xAI 发布了 Grok-2](https://twitter.com/adcock_brett/status/1825201974419042761)，目前已向 Premium X 用户开放测试版。它可以使用 FLUX 1 生成“放飞自我”的图像，并在短短一年多时间内达到了 SOTA 状态。

- **开源模型**：[Nous Research 发布了 Hermes 3](https://twitter.com/adcock_brett/status/1825201997055684653)，这是一个开源模型，提供 8B、70B 和 405B 参数版本，其中 405B 模型相对于其他开源模型达到了 SOTA。

- **机器人技术进步**：[Astribot 展示了他们的新型人型机器人](https://twitter.com/adcock_brett/status/1825201929523237341)，展示了其在无需远程操作的情况下令人印象深刻的实时自由度。[据报道 Apple 正在开发](https://twitter.com/adcock_brett/status/1825201906580390065) 一款带有 Siri 语音命令的桌面机器人，将类似 iPad 的显示屏与机械臂相结合。

- **AI 研究工具**：[Sakana AI 推出了“The AI Scientist”](https://twitter.com/adcock_brett/status/1825201952021459387)，声称是世界上第一个能够自主进行科学研究、产生想法、编写代码、运行实验并撰写论文的 AI 系统。

**AI 模型性能与技术**

- **Vision Transformer (ViT) 性能**：[@giffmana](https://twitter.com/giffmana/status/1825301443709997521) 发表了一篇博客文章，解决了关于 ViT 在高分辨率下的速度、长宽比重要性以及分辨率要求的疑虑。

- **RAG 改进**：[关于利用 LLM 提取的元数据进行数据库过滤以改进多跳查询 RAG 的新研究](https://twitter.com/LangChainAI/status/1825234642037010518) 在 MultiHop-RAG 基准测试中显示出良好的结果。[HybirdRAG](https://twitter.com/dair_ai/status/1825207031558537663) 结合了 GraphRAG 和 VectorRAG，在财务业绩电话会议记录上的表现优于两者。

- **模型优化**：[@cognitivecompai](https://twitter.com/cognitivecompai/status/1825321109744467981) 报告称，在使用 Dolphin 2.9.4 数据集训练 gemma-2-2b 时，GrokAdamW 似乎有所改进。

- **小模型技术**：[@bindureddy](https://twitter.com/bindureddy/status/1825299800994230763) 鼓励对 2B 的小模型进行迭代，使其更加实用，并发明可以应用于更大模型的新技术。

**AI 应用与工具**

- **LangChain 进展**：[LangChain JS 教程](https://twitter.com/LangChainAI/status/1825319516890669178) 介绍了如何使用 LLM 分类器根据查询类型进行动态 Prompt 选择。[使用 Claude 3.5 Sonnet 的 Agentic RAG](https://twitter.com/llama_index/status/1825205945678680465)、MongoDB 和 llama_index 展示了在现有 RAG 流水线上构建 Agentic 知识助手。

- **AI 助力软件工程**：[Cosine 演示了 Genie](https://twitter.com/adcock_brett/status/1825202019503681998)，这是一个全自动 AI 软件工程师，以 30.08% 的成绩打破了 SWE-Bench 的最高分。OpenAI 和 SWE-Bench 的作者[重新设计并发布了 'SWE-bench Verified'](https://twitter.com/adcock_brett/status/1825202085572272574)，以解决原始基准测试中的问题。

- **生产力工具**：[@DrJimFan](https://twitter.com/DrJimFan/status/1825193764962673037) 表达了对 LLM 根据 Prompt 自动过滤、标记 Gmail 并重新排列优先级的期望，强调了 AI 在电子邮件管理方面的潜力。

**AI 伦理与社会影响**

- **AI 欺骗辩论**：[@polynoamial](https://twitter.com/polynoamial/status/1825268351452766578) 讨论了将扑克中的诈唬（bluffing）误解为 AI 欺骗的例子，认为这更多是为了不泄露多余信息，而非主动欺骗。

- **AI 推理能力**：[@mbusigin](https://twitter.com/mbusigin/status/1825226698348220499) 认为 LLM 在推理方面已经优于相当一部分人类，因为它们不依赖“直觉”，并且在逻辑推理测试中表现良好。

**梗与幽默**

- [@AravSrinivas](https://twitter.com/AravSrinivas/status/1825235274273583278) 调侃道：“Networking ~= Not actually working”（社交 ~= 没在干活）
- [@AravSrinivas](https://twitter.com/AravSrinivas/status/1825264229483761779) 分享了一张与 AI 或技术相关的幽默图片（内容未具体说明）。
- [@Teknium1](https://twitter.com/Teknium1/status/1825199882254327825) 吐槽视频生成技术：“为什么几乎所有的视频生成都只是平移或缩放，你还不如用 Flux（快 1000 倍）生成一张图片。”

这份摘要涵盖了所提供推文中 AI 和机器人领域的关键进展、讨论和趋势，重点关注与 AI 工程师和研究人员相关的信息。

---

# AI Reddit 综述

## /r/LocalLlama 综述

**主题 1. XTC：用于增强 LLM 创造力的新采样器**

- **Exclude Top Choices (XTC)：一种提升创造力、打破写作陈词滥调并抑制非逐字重复的采样器，由 DRY 的创作者开发** ([Score: 138, Comments: 64](https://reddit.com//r/LocalLLaMA/comments/1ev8n2s/exclude_top_choices_xtc_a_sampler_that_boosts/))：**Exclude Top Choices (XTC)** 采样器在 **text-generation-webui** 的一个 **GitHub pull request** 中被引入，旨在以对连贯性影响最小的方式**提升 LLM 创造力**并**打破写作陈词滥调**。创作者报告称，XTC 能产生新颖的词句和想法，特别增强了**角色扮演和故事写作**，其体验与单纯提高语言模型的 Temperature 明显不同。


**主题 2. 个人 GPU 用于 AI 开发的成本效益分析**

- **老实说，一块 4090 真的做不了什么** ([Score: 84, Comments: 90](https://reddit.com//r/LocalLLaMA/comments/1evdrxk/honestly_nothing_much_to_do_with_one_4090/))：作者是一名从事 **AI 基础设施和 ML 工程**的工作者，对他为个人 AI 项目购买的 **4090 GPU** 表示失望。他们认为，对于大多数用例，**云端 API 服务**或**企业级 GPU 集群**比单块高端消费级 GPU 进行 AI 任务更实用且更具成本效益，并质疑了个人拥有本地 GPU 进行 AI 实验的价值。

## AI Reddit 全面回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 模型进展与对比**

- **Flux LoRA 训练结果**：一位用户分享了在《权力的游戏》角色上训练 Flux LoRA 模型的惊人结果，仅使用 10 张图像的数据集和 500-1000 个训练步数就实现了高质量输出。训练过程需要超过 60GB 的 VRAM。[来源](https://www.reddit.com/r/StableDiffusion/comments/1ev6pca/some_flux_lora_results/)

- **卡通角色对比**：对比了多种 AI 模型（DALL-E 3, Flux dev, Flux schnell, SD3 medium）生成卡通角色吃西瓜的效果。DALL-E 3 整体表现最佳，Flux dev 位居第二。帖子强调了 DALL-E 3 利用复杂的 LLM 系统将图像划分为不同区域进行详细描述。[来源](https://www.reddit.com/r/StableDiffusion/comments/1ev68la/cartoon_character_comparison/)

- **Flux.1 Schnell 放大技巧**：一位用户分享了提升 Flux.1 Schnell 输出人脸质量的技巧，建议在放大写实图像时使用 4xFaceUpDAT 而非 4x-UltraSharp。帖子还提到了其他放大模型和增强图像质量的技术。[来源](https://www.reddit.com/r/StableDiffusion/comments/1ev6ris/tips_for_flux1_schnell_to_avoid_a_plasticky/)

**AI 公司策略与批评**

- **OpenAI 的商业行为**：一位用户批评 OpenAI 像经营“微型 Ycombinator 初创公司”一样管理公司，理由包括等候名单、CEO 神秘的推文以及发布前的预热视频。帖子认为这些策略不适合一家估值近 1000 亿美元的公司，可能会让客户和企业用户感到困惑。[来源](https://www.reddit.com/r/OpenAI/comments/1evspo8/openai_runs_its_company_like_a_tiny_ycombinator/)

**AI 生成内容与迷因 (Memes)**

- **《迷雾》(Flux+Luma)**：一段展示使用 Flux 和 Luma 模型生成的 AI 内容视频，内容似乎灵感源自电影《迷雾》。[来源](https://www.reddit.com/r/StableDiffusion/comments/1evfgys/the_mist_fluxluma/)

- **看起来有点眼熟？**：r/singularity 版块中的一个迷因帖子，可能引用了 AI 相关内容。[来源](https://www.reddit.com/r/singularity/comments/1ev8cfs/seems_familiar_somehow/)

- **总得有人说出来...**：r/StableDiffusion 版块中的另一个迷因帖子。[来源](https://www.reddit.com/r/StableDiffusion/comments/1evio5l/someone_had_to_say_it/)

**未来技术与研究**

- **自动驾驶汽车越狱**：一个推测一旦自动驾驶汽车普及，人们将尝试对其进行越狱的帖子。[来源](https://www.reddit.com/r/singularity/comments/1ev97ky/once_selfdriving_cars_are_here_i_expect_people_to/)

- **狗狗逆龄药**：一项研究报告了在狗身上测试逆龄药的乐观结果。然而，该帖子缺乏同行评审研究的引用，并因过于轶事化而受到批评。[来源](https://www.reddit.com/r/singularity/comments/1ev3vac/new_study_reveals_promising_results_for_age/)


---

# AI Discord 回顾

> 由 Claude 3.5 Sonnet 总结的总结之总结


**1. Hermes 3 模型发布与性能**

- **Hermes 3 在 N8Bench 上追平 Llama 3.1**：**Hermes 3** 在 **N8Bench** 基准测试中获得了与 **Llama 3.1 Instruct** 相同的分数，该基准测试衡量模型的推理和问题解决能力。
   - 这一结果意义重大，因为 Llama 3.1 Instruct 被认为是目前最先进的语言模型之一，凸显了 Hermes 3 的竞争性能。
- **Hermes 3 405B 在 OpenRouter 开启免费周末**：**OpenRouter** 宣布，由 **Lambda Labs** 提供支持，**Hermes 3 405B** 在限时内免费使用，并提供 **128k 上下文窗口**。
   - 用户可以通过 [OpenRouter 的 Hermes 3 405B 页面](https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b:extended) 访问该模型，这为测试和评估这一超大型语言模型提供了机会。
- **量化对 405B 模型的影响**：**@hyperbolic_labs** 警告称，**量化 (Quantization)** 会显著降低 **405B 模型** 的性能。
   - 他们建议如果对性能有要求，可以联系他们寻求替代方案，强调了减小模型体积与保持性能质量之间的权衡。
  


**2. LLM 推理优化技术**

- **INT8 量化用于 CPU 执行**：一位成员询问了使用 **INT8 量化** 来加速小模型在 CPU 上执行的潜在好处，并指出某些 CPU 可能原生支持运行 INT8 而无需转换为 FP32。
   - 这种方法可能提升基于 CPU 推理的性能，特别是对于资源受限的环境或边缘设备。
- **FP8 训练进展**：使用 **FP8 中的第一动量 (1st momentum)** 平稳训练 **1B FP8 模型** 至 **48k steps**，其 loss 与带有 **0.08 偏移量** 的 **bfloat16** 相当。
   - 这表明 FP8 训练配合第一动量是有效的，在实现与 bfloat16 训练相似结果的同时，可能提供内存节省和性能提升。
- **开源模型的 Batching APIs**：**CuminAI** 推出了一种为开源模型创建 **batching APIs** 的解决方案，类似于 OpenAI 和 Google 最近推出的功能。
   - 虽然大公司的 batching APIs 缺乏处理保证和 SLA，但 CuminAI 的方法旨在为开源模型部署提供类似的成本节约优势。指南可在 [其博客文章](https://blog.cuminai.com/how-to-get-a-batching-api-like-openai-for-open-source-models-824529788a49) 中找到。
  


**3. 开源 AI 模型进展**

- **Falcon Mamba 7B 宣称性能超越 Llama 3 8B**：一段 [YouTube 视频](https://www.youtube.com/watch?v=dokzrFa-DtY) 宣布发布 **Falcon Mamba 7B**，声称其表现优于 **Llama 3 8B**。
   - 这一进展可能对 LLM 领域产生重大影响，因为 Falcon Mamba 7B 是一个挑战既定基准的、极具前景的新模型。
- **Ghost 8B Beta 的多语言实力**：新发布的语言模型 **Ghost 8B Beta** 现在支持包括英语、越南语、西班牙语和中文在内的 **16 种语言**，并提供两种上下文选项（8k 和 128k）。
   - 该模型在数学、推理和指令遵循能力方面表现出色，在 AlpacaEval 2.0 胜率得分上超过了 Llama 3.1 8B Instruct、GPT-3.5 Turbo 和 Claude 3 Opus 等竞争对手。
- **阿里巴巴达摩院 (Alibaba DAMO) 发布 VideoLLaMA 2-72B**：**阿里巴巴达摩院 (Alibaba DAMO)** 发布了 **VideoLLaMA 2-72B**，这是一款新的视频 LLM，可在 [HuggingFace](https://huggingface.co/collections/DAMO-NLP-SG/videollama-2-6669b6b6f0493188305c87ed) 上获取，并在 [HuggingFace Spaces 上提供 demo](https://huggingface.co/spaces/lixin4ever/VideoLLaMA2)。
   - [研究论文](https://huggingface.co/papers/2406.07476) 也已在 HuggingFace 上发布，展示了结合视频理解和语言建模的多模态 AI 进展。
  


**4. AI 安全与监管讨论**

- **南希·佩洛西 (Nancy Pelosi) 反对加州 AI 法案**：**荣休议长南希·佩洛西 (Nancy Pelosi)** 发表声明，反对关于 AI 监管的 **加州参议院第 1047 号法案 (California Senate Bill 1047)**。
   - 完整声明可在 [众议院网站](http://pelosi.house.gov/news/press-releases/pelosi-statement-opposition-california-senate-bill-1047) 找到，突显了关于如何在州一级进行 AI 治理的持续辩论。
- **Procreate 拒绝集成生成式 AI**：**Procreate** 的 CEO 明确表示，他们不会在产品中集成生成式 AI，这一决定受到了社交媒体上许多艺术家和用户的赞赏。
   - 一些观察者指出，这种立场未来可能会改变，因为它可能会限制新功能的开发。这突显了传统创意工具与创意产业中 AI 快速发展之间持续存在的紧张关系。
- **加里·马库斯 (Gary Marcus) 重新审视 AI 泡沫担忧**：AI 研究员 **加里·马库斯 (Gary Marcus)** 在一段名为“**The AI Bubble: Will It Burst, and What Comes After?**”的视频中重新审视了他在 AGI-21 上的主题演讲，指出尽管 AI 取得了重大进展，但他当时强调的许多问题在今天仍然具有现实意义。
   - 这段可在 [YouTube](https://www.youtube.com/watch?v=91SK90SahHc) 上观看的讨论反映了关于当前 AI 发展趋势的可持续性、轨迹及其潜在社会影响的持续辩论。
  


---

# 第 1 部分：Discord 高层摘要

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Flux：新王者？**：成员们讨论了 Flux 接管图像生成 AI 社区的潜力，每天都有新的 Loras 和合并模型出现。
   - 一些人认为 Stability AI 需要尽快发布产品进行竞争，因为 Flux 正在成为 CivitAI 和 Hugging Face 上的主导力量。
- **Flux vs. SD3：巅峰对决**：关于 Flux 是否与 SD3 有本质区别存在争论，这两个模型都使用了 DiT 架构、ret flow loss 和类似的 VAE 尺寸。
   - 关键区别在于 Flux dev 是从大模型蒸馏而来的，而 Stability AI 也可以使用这种技巧。一些人更倾向于非蒸馏模型，即使图像质量较低。
- **Flux 训练：挑战与机遇**：成员们讨论了为 Flux 训练 Loras 的挑战，并指出训练代码尚未正式发布。
   - 一些用户正在探索本地训练 Loras 的方法，而另一些用户则建议使用 Replicate 官方的 Flux LoRA Trainer 以获得更快、更简便的结果。
- **ComfyUI vs. Forge：UI 之争**：用户讨论了 ComfyUI 和 Forge 之间的性能差异，一些人发现 Forge 更快，尤其是在批处理方面。
   - 讨论涉及了 Gradio 4 更新对 Forge 的影响以及未来改进的潜力。一些用户喜欢 ComfyUI 的灵活性，而另一些用户则欣赏 Forge 的优化。
- **Stable Diffusion 的 GPU 推荐**：成员们分享了各种 GPU 在 Stable Diffusion 中的表现经验，16GB VRAM 被视为最低配置，24GB 则比较舒适。
   - 讨论涉及了 VRAM 比 CPU 速度更重要，以及 RAM 和其他应用程序对性能的影响。共识是尝试不同的模型和编码器，以找到最适合每个系统的配置。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hermes 2.5 表现优于 Hermes 2**：在添加了[代码指令示例](https://link.to.examples)后，**Hermes 2.5** 在各种基准测试中的表现似乎优于 **Hermes 2**。
   - Hermes 2 在 MMLU 基准测试中得分为 **34.5**，而 Hermes 2.5 得分为 **52.3**。
- **Mistral 在扩展超过 8k 时面临困难**：成员们表示，如果不进行持续预训练，**Mistral** 无法扩展到 8k 以上，[这是一个已知问题](https://link.to.issue)。
   - 他们指出，*mergekit* 和 *frankenMoE finetuning* 的进一步工作是性能的下一个前沿。
- **模型合并策略讨论**：一位成员建议将 **UltraChat** 和基础 **Mistral** 之间的差异应用于 **Mistral-Yarn**，作为一种潜在的合并策略。
   - 其他人表示怀疑，但该成员保持乐观，并引用了过去在他们称之为“诅咒式模型合并（cursed model merging）”方面的成功尝试。
- **Open Empathic 项目寻求协助**：一位成员呼吁帮助扩大 **Open Empathic** 项目的类别，特别是在低端部分。
   - 他们分享了一个关于 [Open Empathic 发布与教程的 YouTube 视频](https://youtu.be/GZqYr8_Q7DE)，指导用户贡献他们喜欢的 YouTube 视频电影场景，以及 [OpenEmpathic 项目本身](https://dct.openempathic.ai/)的链接。
- **带有 1st Momentum 的 FP8 训练实现了相似的 Loss**：使用 **FP8 中的 1st momentum** 平滑训练 **1B FP8 模型**至 **48k steps**，其产生的 Loss 与带有 **0.08 offset** 的 **bfloat16** 相当。
   - 这表明 FP8 训练配合 1st momentum 是有效的，可以达到与 bfloat16 训练相似的结果。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Ghost 8B Beta (1608) 发布**：**Ghost 8B Beta (1608)** 已发布，这是一个性能顶尖的语言模型，具有无与伦比的多语言支持和成本效益。
   - 在胜率（winrate）得分上，它的表现优于 Llama 3.1 8B Instruct, GPT-3.5 Turbo, Claude 3 Opus, GPT-4 等模型。
- **Ghost 8B Beta 的多语言实力**：**Ghost 8B Beta** 现在支持 **16 种语言**，包括英语、越南语、西班牙语、中文等。
   - 它提供两种上下文选项（8k 和 128k），并改进了数学、推理和指令遵循能力，以更好地处理任务。
- **Ghost 8B Beta 超越竞争对手**：在 AlpacaEval 2.0 胜率得分中，**Ghost 8B Beta** 的表现优于 Llama 3.1 8B Instruct, GPT 3.5 Turbo, Claude 3 Opus, Claude 3 Sonnet, GPT-4 和 Mistral Large 等模型。
   - 这种令人印象深刻的表现突显了其卓越的知识能力和多语言实力。
- **使用 LLM 进行代码编辑**：一篇新论文探讨了如何根据用户指令使用 Large Language Models (LLMs) 进行代码编辑。
   - 它引入了 EditEval（一个用于评估代码编辑性能的新颖基准测试）和 InstructCoder（一个用于对 LLM 进行代码编辑指令微调的数据集，包含超过 114,000 个指令-输入-输出三元组）。
- **LLM 中的推理差距**：一篇研究论文提出了一个框架，使用基准测试的功能变体（特别是 MATH 基准测试）来评估 LLM 的推理能力。
   - 它将“推理差距”定义为将任务作为编程问题与作为自然语言问题提出时，解决任务的性能差异，强调 LLM 在任务以代码形式呈现时通常表现更好。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **线性 Transformer：与 Softmax 的天作之合**：Nous Research 发布了关于一种与 Softmax 匹配的线性 Transformer 变体的研究，允许以 O(t) 而非 O(t^2) 的复杂度进行训练。
   - 该研究可在[此处](https://manifestai.com/articles/symmetric-power-transformers/)查看，探讨了这种新变体及其对训练效率的影响。
- **Falcon Mamba 7B 击败 Llama 3 8B**：一段 [YouTube 视频](https://www.youtube.com/watch?v=dokzrFa-DtY)宣布发布 **Falcon Mamba 7B**，并声称其性能优于 **Llama 3 8B**。
   - 这可能对大语言模型领域产生重大影响，因为 Falcon Mamba 7B 是一个相对较新且充满前景的模型。
- **正则表达式作为分块技术的争议**：一位用户分享了他们对基于正则表达式（regex）的文本分块器的看法，表示如果他们在代码库中看到它会“尖叫”，因为正则表达式非常复杂。
   - 然而，另一位用户反驳说，专门针对文本分块器，正则表达式可能是一个“非常可靠的选择”，因为它提供了“回溯优势”并允许灵活的分块设置。
- **Hermes 3：N8Bench 的性能之王？**：Hermes 3 在 N8Bench 基准测试中的得分与 Llama 3.1 Instruct 相同，该基准测试衡量模型推理和解决问题的能力。
   - 这是一个重要的结果，因为 Llama 3.1 Instruct 被认为是目前最先进的语言模型之一。
- **Gemini Flash：RAG 的未来？**：一位用户报告说，他们已将部分 RAG 任务迁移到 Gemini Flash，并指出总结质量有所提高，且减少了迭代需求。
   - 他们分享了一个用于通过 Gemini Flash 处理原始非结构化转录文本的脚本，可在 GitHub 上获取：[https://github.com/EveryOneIsGross/scratchTHOUGHTS/blob/main/unstruct2flashedTRANSCRIPT.py](https://github.com/EveryOneIsGross/scratchTHOUGHTS/blob/main/unstruct2flashedTRANSCRIPT.py)。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro 注册体验不佳**：多位用户报告了 Perplexity Pro 的注册流程问题，尽管收到了免费一年的优惠，但用户在不付费的情况下无法完成注册。
   - 建议用户联系 support@perplexity.ai 以寻求此问题的帮助。
- **Obsidian Copilot 获得 Claude 加持**：一位用户分享了使用 Claude API key 配合 Obsidian Copilot 插件的经验，认为其在性能方面是一个可靠的选择。
   - 他们强调了在正式使用前检查 API 计费设置的重要性，并指出 Obsidian 需要具备实时联网能力。
- **Perplexity 的图像生成功能表现不佳**：用户讨论了 Perplexity 图像生成功能的缺陷，该功能目前仅对 Pro 用户开放，且需要 AI 提示词来描述图像。
   - 用户认为这是一种“奇怪”且“糟糕”的实现方式，并强调需要一种更精简的图像生成方法。
- **Perplexity 搜索遇到小故障**：多位用户报告了 Perplexity 的搜索质量问题，包括难以找到相关链接以及收到不准确的结果。
   - 这些问题被归因于可能的 Bug、提示词（prompts）变更或推理后端服务的更新。
- **Perplexity 模型变更引发用户担忧**：讨论围绕 Perplexity 模型的变更展开，用户对响应质量可能下降以及“我无法为此提供帮助”错误增加表示担忧。
   - 其他担忧还包括 API 响应中缺失标点符号，以及在非科学查询中使用 Wolfram Alpha。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Hermes 3 405B 本周末免费！**：由 **Lambda Labs** 提供支持，**Hermes 3 405B** 限时免费，具备 **128k context**。
   - 您可以通过[此链接](https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b:extended)进行体验。
- **GPT-4 extended 现已上线 OpenRouter**：现在可以通过 **OpenRouter** 使用 **GPT-4 extended output**（Alpha 测试阶段）。
   - 该模型限制最大输出为 **64k tokens**。
- **Perplexity Huge 是 OpenRouter 上最大的在线模型**：**Perplexity Huge** 于 **3 天前**发布，是 **OpenRouter 上最大的在线模型**。
   - 您可以在[此链接](https://x.com/OpenRouterAI/status/1824593712095301914)找到更多信息。
- **模型发布周**：本周 OpenRouter 上发布了 **10 个新模型**，包括 **GPT-4 extended**、**Perplexity Huge**、**Starcannon 12B**、**Lunaris 8B**、**Llama 405B Instruct bf16** 和 **Hermes 3 405B**。
   - 您可以在[此链接](https://x.com/OpenRouterAI/status/1824608728810991637)查看完整列表。
- **量化会降低性能**：根据 @hyperbolic_labs 的说法，**量化（Quantization）**会大幅降低 **405B 模型**的性能。
   - 如果您担心性能问题，他们建议与其联系，因为他们提供替代解决方案。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **INT8 量化能提升 CPU 速度吗？**：一位成员询问了在 CPU 上对较小模型使用 INT8 量化是否能带来性能提升。
   - 他们建议某些 CPU 可能原生支持 INT8 执行，从而绕过向 FP32 的转换，并可能提高性能。
- **Llama.cpp 支持 Mini-CPM-V2.6 和 Nemotron/Minitron**：一位成员确认最新的 llama.cpp 版本支持 Mini-CPM-V2.6 以及 Nvidia 的 Nemotron/Minitron 模型。
   - 此次更新扩大了与 llama.cpp 兼容的模型范围，增强了其对 LLM 爱好者的通用性。
- **将聊天记录导入 LM Studio**：一位成员寻求关于如何将 JSON 导出的聊天日志导入 LM Studio 的指导。
   - 另一位成员澄清说聊天数据存储在 JSON 文件中，并提供了访问相关文件夹位置的说明。
- **Vulkan 错误：CPU 缺少 AVX2 支持**：一位用户遇到了错误，提示其 CPU 缺少 AVX2 支持，导致无法使用某些功能。
   - 一位热心成员询问了 CPU 型号，以协助诊断并解决该问题。
- **LLM 与网页交互：一个复杂的挑战**：一位成员讨论了让 LLM 与网页交互的可能性，特别是寻求一种“视觉（vision）”方法。
   - 虽然提到了 Selenium 和 IDkit 等工具，但普遍共识是，由于网页结构的多样性，这仍然是一个具有挑战性的问题。

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Claude 在代码方面表现优于 Chat-GPT**：一名成员表示，Claude 在处理代码方面往往比 Chat-GPT 更出色。
   - 坦白说，GPT-4o 的 API 成本高于 Claude 这一点毫无道理。
- **Livebench.ai：Yann LeCun 的开源基准测试**：Livebench.ai 是由 Yann LeCun 等人创建的开源基准测试。
   - LMSys 基准测试目前可能是最糟糕的。
- **Claude Projects 对比 Chat-GPT Memory 功能**：一名成员认为 Claude Projects 比 Chat-GPT 的 Memory 功能更有用。
   - 该成员还表示，自定义 GPTs 更像是一个项目，允许使用你自己的 endpoints。
- **OpenAI 正在赢得注意力游戏**：OpenAI 通过发布 GPT-4o 等新模型来控制注意力，从而赢得竞争。
   - 该成员表示，即使人们不想参与技术炒作，也都在讨论 OpenAI 的新模型。
- **GPT-4o 现在比 Claude 和 Mistral 差**：成员们注意到 GPT-4o 最近变得越来越笨，可能正遭受某种“阿尔茨海默症”的困扰。
   - Claude Sonnet 因其卓越的性能而受到称赞，正成为成员们的首选。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Topology 的 CLM：像人类一样学习**：Topology 发布了 [Continuous Learning Model (CLM)](https://yellow-apartment-148.notion.site/CLM-Docs-507d762ad7b14d828fac9a3f91871e3f)，这是一种能够记住交互、自主学习技能并在空闲时间思考的新模型，就像人类一样。
   - 该模型可以在 [http://topologychat.com](http://topologychat.com) 进行体验。
- **GPT5 需要大 20 倍**：Mikhail Parakhin 发推称，为了让 AI 模型获得实质性的改进，新模型的规模应至少比当前模型大 **20倍**。
   - 这将需要 **6个月** 的训练时间以及一个新的、大 **20倍** 的数据中心，而建造这样一个数据中心大约需要一年时间。
- **Procreate 拒绝生成式 AI**：Procreate 的 CEO 表示，他们不会将生成式 AI 集成到产品中。
   - 虽然社交媒体上的一些艺术家和用户对此表示庆祝，但也有人指出，这可能意味着未来不会增加新功能，且这种情况可能会发生变化。
- **DSPy：尚未完全商业化**：目前 **DSPy** 背后还没有商业公司，尽管 Omar 正在为此努力。
   - 一名成员分享说，他们参加了 **Cursor** 办公室的见面会，虽然没有 alpha 版本可以分享，但他们确实打了招呼。
- **DSPy 弥合差距**：**DSPy** 旨在弥合 prompting 与 finetuning 之间的差距，让用户能够避免手动进行 prompt tuning。
   - 论文提到 **DSPy** 避免了 prompt tuning，这可能使得切换模型、重新调整以适应数据偏移等操作变得更加容易。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Office Hours 启动！**：加入 Cohere 的 **高级产品经理** 和 **DevRel**，参加关于 **产品和内容更新** 的轻松会议，包含 **最佳实践** 以及关于 **Prompt Tuning**、**带有 Agents 的 Guided Generations API** 和 **LLM University Tool Use 模块** 的 **问答环节**。
   - 活动于今天 **东部时间下午 1 点在 #stage 频道** 举行，可以通过 [此链接](https://discord.com/events/954421988141711382/1265012161965461625) 找到。
- **Cohere Prompt Tuner：优化的 Prompting！**：了解 **Cohere Prompt Tuner**，这是一个优化提示词并提高 LLM 结果准确性的强大工具。
   - 博客文章详细介绍了如何利用该工具及 [相关功能](https://cohere.com/blog/intro-prompt-tuner)。
- **Command-r-plus 无法工作？**：一名用户报告说，当 context length 达到 4000 个 token 时，**Sillytavern** 中的 **command-r-plus** 停止稳定工作。
   - 该用户一直尝试使用该工具来增强工作流程，但面临这一意外问题。
- **API Key 部分响应问题**：一名用户报告其 API Key 仅返回部分响应，即使尝试了不同的 Wi-Fi 路由器和蜂窝数据也是如此。
   - 该用户目前正在寻求此问题的解决方案。
- **用于准确 JSON 生成的 Structured Outputs**：**Structured Outputs** 是 Cohere 工具的最新更新，其 **JSON 生成** 速度比开源实现快 **80倍** 且更准确。
   - 这一新功能提高了 JSON 输出的准确性，并在 [这篇博客文章](https://cohere.com/blog/introducing-structured-outputs) 中进行了讨论。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Yi Tay 的“混沌不眠式拼命”工作风格**：讨论涉及了各家 AI 组织的工作风格，一名成员暗示 Yi Tay 以一种“混沌不眠式拼命（chaos no sleep grind）”的心态在运作。
   - 他们引用了 Phil (@phill__1) 的一条推文，暗示 01AI 可能正在退出非中国市场：[.@01AI_Yi 怎么了？他们要退出非中国市场吗？](https://x.com/phill__1/status/1825438202548658526?s=46)。
- **Nancy Pelosi 反对加州 AI 法案**：荣誉议长 Nancy Pelosi 发表声明，反对加州关于 AI 监管的 Senate Bill 1047 法案。
   - 该声明发布在众议院网站上：[Pelosi 关于反对加州参议院 1047 号法案的声明](http://pelosi.house.gov/news/press-releases/pelosi-statement-opposition-california-senate-bill-1047)。
- **Zicheng Xu 从 Allen-Zhu 团队被裁**：Zeyuan Allen-Zhu 宣布“Part 2.2”教程的作者 Zicheng Xu 意外被裁员。
   - Allen-Zhu 极力推荐 Xu，并为潜在的合作者或雇主提供了他的电子邮箱：zichengBxuB42@gmail.com（请删除大写字母 'B'）。
- **Nous Hermes Discord 关于评估设置的争议**：一名用户提到了 Nous Discord 中关于某用户表现无礼以及误导评估设置的讨论。
   - 该用户提到他们的评估细节位于论文的 SFT 章节中，并承认弄错事实的感觉并不好，但文章的核心内容仍然有效。
- **Meta Cooking（模型调优）引发困惑**：一名用户好奇什么是 "meta cooking"，暗示 Nous Discord 中可能存在冲突或争议。
   - 该用户提到发现了关于评估设置的矛盾信息，这可能是由于使用了默认的 LM Harness 设置且缺乏清晰的文档说明。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **GrokAdamW 提升 Axolotl 速度**：GrokAdamW 是一款旨在鼓励快速 Grokking 的 PyTorch 优化器，现已发布并可通过 Transformers 集成在 Axolotl 中使用。[GrokAdamW 仓库](https://github.com/cognitivecomputations/grokadamw)
   - 该优化器的灵感来自 GrokFast 论文，旨在加速模型在 Grokking 现象下的泛化能力。[GrokFast 论文](https://arxiv.org/abs/2405.20233)
- **Gemma 2b 训练故障**：一名用户报告在训练 Gemma 2b 模型时，Loss 持续为 0.0，且梯度范数（gradient norm）为 nan。
   - 该用户建议在训练 Gemma 2b 模型时使用 eager attention 代替 sdpa，这解决了 Loss 为零的问题。
- **Axolotl 中的自定义加载器与聊天模板**：一名用户询问如何在 Axolotl 的 .yml 配置文件中使用 Chat Template 类型，特别是如何指定使用哪种加载器（例如 ShareGPT）。
   - 另一名用户建议可以通过提供自定义 .yml 文件来指定使用的加载器。
- **使用 Axolotl 进行微调：无需编程**：一名用户澄清，使用 Axolotl 进行微调通常不需要编程知识，而是需要理解如何格式化数据集以及如何适配现有示例。
   - 一名用户提到自己拥有一台强大的 AI 运行设备来运行 Llama 3.1 70b，但觉得它在某些关键领域仍有不足，希望使用自己的内容数据集进行微调。
- **LLaMa 3.1 8b Lora 检测事后推理**：一名用户正在训练一个 LLaMa 3.1 8b Lora，用于检测对话中的事后推理（post-hoc reasoning）。他花了三天时间整理了一个包含不到 100 条多轮对话、约 30k token 的小型数据集。
   - 该用户使用 Sonnet 3.5 辅助生成示例，但尽管精心设计了 Prompt，仍必须对每个生成的示例进行多处修正。因为即使指示模型不要创建带有事后推理的示例，由于其微调数据的特性，模型仍然会生成此类内容。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain 缓存问题**：一位成员对为什么 `.batch_as_completed()` 没有通过缓存加速感到困惑，尽管在缓存后 `.invoke()` 和 `.batch()` 几乎是瞬间完成的。
   - 他们观察到缓存是在第一次运行后填充的，但 `.batch_as_completed()` 似乎没有利用它。
- **LLM 在结构化输出方面表现不佳**：一位成员提到本地 LLM（如 Llama 3.1）通常难以产生一致的结构化输出，特别是在 JSON 解析方面。
   - 他们询问了专门为训练模型以改进 JSON 解析以及针对 Tool 和 ReAct Agent 的结构化输出而设计的数据集。
- **在 RAG 聊天机器人中删除文件**：一位成员讨论了如何在使用 MongoDB 作为向量数据库的 RAG 聊天机器人中实现文件删除功能。
   - 一份回复提供了使用 LangChain 库中针对 MongoDB 向量存储和 OpenAIFiles 的 `delete` 方法示例，并附带了相关的文档链接。
- **混合搜索相关性问题**：一位成员在使用 BM25Retriever 和向量相似度搜索的混合搜索方法的 RAG 应用中遇到了检索文档和生成答案的相关性问题。
   - 建议包括检查文档质量、调整 Retriever 配置、评估 Chain 设置以及审查 Prompt 和 LLM 配置。
- **CursorLens 是面向 Cursor 用户的新仪表板**：CursorLens 是一个面向 Cursor 用户的开源仪表板，提供关于 Prompt 的分析，并允许配置 Cursor 本身不提供的模型。
   - 它最近在 ProductHunt 上发布：[https://www.producthunt.com/posts/cursor-lens](https://www.producthunt.com/posts/cursor-lens)。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Orange Pi 5 评测：新型实惠的 SBC**：一位用户分享了 **Orange Pi 5**（一种新型 **Arm-based SBC**）的 [YouTube 视频评测](https://youtu.be/79lquFD3oT4)。
   - 视频强调 **Orange Pi 5** 不要与 **Raspberry Pi 5** 混淆。
- **GPT-4o-mini 模型问题：快速修复**：一位用户在将模型设置为 **GPT-4o-mini** 时遇到麻烦。
   - 另一位用户提供了解决方案：`interpreter --model gpt-4o-mini`。
- **OpenInterpreter 设置重置：还原指南**：一位用户寻求在实验后将 OpenInterpreter 设置恢复为默认的方法。
   - 解决方案包括使用 `interpreter --profiles` 查看和编辑配置文件，以及可能需要卸载并重新安装 OpenInterpreter。
- **OpenInterpreter API 集成：构建桥梁**：一位用户询问如何将 OpenInterpreter 集成到他们现有的 AI 核心中，发送请求并接收输出。
   - 推荐的解决方案包括使用带有 Flask 服务器的 Python 脚本来处理 AI 核心与 OpenInterpreter 之间的通信。
- **用于 Bash 命令的本地 LLM：CodeStral 和 Llama 3.1**：一位成员请求推荐能够处理 Bash 命令的本地 LLM。
   - 另一位成员建议使用 **CodeStral** 和 **Llama 3.1**。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **LLM 挣扎于可靠性问题**：众所周知，大语言模型（LLM）会产生事实错误的信息，导致“幻觉”内容，从而阻碍其可靠性。
   - **WeKnow-RAG** 解决了这一问题，该系统将网络搜索和 Knowledge Graphs 集成到检索增强生成（RAG）系统中，以提高 LLM 的准确性和可靠性。
- **DSPy 公布其 Roadmap**：**DSPy 2.5**（预计 1-2 周内发布）和 **DSPy 3.0**（几个月内发布）的 Roadmap 已经发布，概述了目标、里程碑和社区贡献。
   - 该 Roadmap 可在 GitHub 上查看：[DSPy Roadmap](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/roadmap.md)。
- **Langgraph 和 Routequery 类错误**：一位用户在 **Langgraph** 中遇到了 `routequery` 类的错误。
   - 他们寻求关于将 **DSPy** 与大型工具集集成的指导，并分享了 **Langgraph** 实现的链接：[Adaptive RAG](https://github.com/sksarvesh007/adaptive-rag/blob/main/langgraph_adaptive_rag.ipynb)。
- **优化专家设计的 Prompt**：一位成员询问 **DSPy** 是否可以优化已经由专家手动设计的 Prompt。
   - 他们询问 **DSPy** 是否能有效优化初始草案，并改进已建立的 Prompt 系统。
- **Colpali 微调讨论**：讨论集中在 **Colpali** 的微调上，由于其领域特定性，该模型需要专门的专业知识。
   - 讨论强调了理解有效微调 **Colpali** 所需数据的重要性。

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **FLUX Dev 可以生成网格**：一位用户分享了 **FLUX Dev** 可以生成同一个（虚构）人物的 3x3 照片网格。
   - 这对于训练 **LORAs** 以创建各种虚构人物的一致角色非常有用。
- **为特定目的训练 LORAs**：一位用户表示有兴趣为特定目的训练 **LORAs**，例如 **dabbing**、**middle finger** 和 **30s cartoon** 风格。
   - 他们提到了将他们的 **FLUX Dev LoRA** 转换为 **FP8** 或在 **Replicate** 上使用 **FP8 LoRA trainer** 的可能性。
- **用于医疗辅助的 LLMs：尚未准备就绪**：几位用户对目前将 **LLMs** 用于医疗辅助表示怀疑。
   - 他们认为 **LLMs** 在此类关键应用中尚不够可靠。
- **JPEG-LM：用于图像和视频的 LLMs？**：一篇新的研究论文提出在自回归 **LLM** 架构中，使用标准编解码器（如 JPEG、AVC/H.264）将图像和视频建模为压缩文件。
   - 这种方法消除了对原始像素值建模或矢量量化的需求，使过程更高效，并为未来的研究提供了潜力。
- **JPEG-LM vs. SIREN：巨头之战？**：一位用户俏皮地声称，他们使用 33kB 的复数值神经网络超越了 2020 年的 **SIREN** 架构。
   - 虽然承认 NVIDIA 2022 年的 **Neural Graphics Primitives** 论文显著推动了该领域的发展，但他们强调了使用 MS-SSIM 作为图像质量评估指标的重要性，而不仅仅是 MSE 和 MAE。



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Workflows 成为焦点**：Rajib Deb 分享了一个展示 LlamaIndex 的 **workflow** 能力的视频，演示了装饰器、控制流类型、事件驱动的过程链，以及用于复杂任务的自定义事件和步骤。
   - 该视频专注于 **workflows**，强调了它们以更结构化的方式构建复杂应用的能力。
- **使用 Claude 3.5 构建 Agentic RAG 助手**：Richmond Lake 的教程指导用户使用 Claude 3.5、MongoDB 和 LlamaIndex 构建 **agentic** 知识助手，强调在现有 **RAG** 管道之上构建 **agentic** 知识助手。
   - 本教程演示了使用 LlamaIndex 实现高级 **RAG** 技术，强调工具选择、任务分解和事件驱动的方法论。
- **BeyondLLM 简化高级 RAG 管道**：由 AIPlanetHub 开发的 BeyondLLM 在 LlamaIndex 之上提供了抽象，使用户仅需 5-7 行代码即可构建具有评估、可观测性和高级 **RAG** 功能的高级 **RAG** 管道。
   - 这些高级 **RAG** 功能包括查询重写、向量搜索和文档摘要，简化了复杂 **RAG** 应用的开发。
- **网页爬虫：LlamaIndex 的难题**：一位成员询问了适用于 LlamaIndex 的网页爬虫推荐，另一位成员推荐了 FireCrawl，并分享了一个展示 LlamaIndex **workflow** 更复杂实现的 YouTube 视频。
   - 对话强调了对能与 LlamaIndex 无缝集成的有效网页爬取工具的需求，以实现高效的知识提取和处理。
- **揭秘 RouterQueryEngine 和 Agents 的秘密**：一位成员寻求澄清 LlamaIndex 的 RouterQueryEngine 和 **Agents** 之间的区别，特别是在路由和 **function calling** 方面。
   - 讨论明确了 RouterQueryEngine 的行为类似于硬编码的 **agent**，而 **Agents** 提供了更大的灵活性和通用性，突出了每种方法的独特能力。



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **HF Spaces 的限制**：一位成员在通过 **HF Spaces** 托管自己的 **LLM** 时遇到困难，因为 **ZeroGPU** 不支持 **vLLM**。
   - 该成员正在寻找替代方案，可能涉及 **Modal**。
- **使用 Modal 托管 LLM**：另一位成员报告使用 **Modal** 托管 **LLMs**。
   - 然而，他们目前正在转向 **FastHTML**，并正在寻找设置指南。
- **使用 Jarvis Labs 进行微调**：一位成员分享了他们专门使用 **Jarvis Labs** 进行 **LLM** 微调的经验。
   - 这表明与其他平台相比，**Jarvis Labs** 可能提供了一种更简化的方法。



---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **OpenAI 和 Google 通过 Batching API 降低成本**：OpenAI 和 Google 为部分模型推出了新的 Batching API，与常规请求相比，成本降低了 50%。
   - 然而，这些 API 目前缺乏处理保证、服务等级协议 (SLAs) 和重试机制。
- **CuminAI：开源 Batching API**：CuminAI 提供了一种为开源模型创建 Batching API 的解决方案，类似于 OpenAI 提供的服务。
   - 在[这里](https://blog.cuminai.com/how-to-get-a-batching-api-like-openai-for-open-source-models-824529788a49)查看他们的分步指南：“如何为开源模型获取类似 OpenAI 的 Batching API”。
- **SLM：AI 的新超级英雄？**：CuminAI 强调了小型语言模型 (SLM) 的潜力，认为在 AI 领域“大并不总是更好”。
   - 虽然大型语言模型 (LLM) 一直占据主导地位，但 SLM 提供了一种更具成本效益且高效的选择，特别是对于不需要大量计算能力的任务。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Llamafile 提升性能并增加新功能**：**Llamafile** 发布了新功能，包括**语音转文本命令**、**图像生成**，以及其 HTTP server embeddings 的 **3 倍性能提升**。
   - 由 [Justine](https://discord.com/channels/1089876418936180786/1262961704602570832/1275110073584320576) 撰写的完整更新详细介绍了性能改进和新功能。
- **Mozilla AI 在 Rise25 庆祝社区**：Mozilla AI 正在表彰那些致力于构建负责任、可信、包容且以人类尊严为中心的 AI 未来的社区成员。
   - 几位成员参加了此次活动，包括 <@631210549170012166>、<@1046834222922465314>、<@200272755520700416> 和 <@1083203408367984751>。
- **ML 论文研讨：Agent 与 Transformer 深度探讨**：参加由 <@718891366402490439> 主持的关于 **Communicative Agents** 和 **Extended Mind Transformers** 的会议。
   - 预约会议：与作者 <@878366123458977893> 讨论 [Communicative Agents](https://discord.com/events/1089876418936180786/1266733035231903795)，以及与作者 <@985920344856596490> 讨论 [Extended Mind Transformers](https://discord.com/events/1089876418936180786/1267946366680694817)。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道详细摘要和链接

{% if medium == 'web' %}

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1274088287803281510)** (567 messages🔥🔥🔥): 

> - `Flux`
> - `Flux vs. SD3`
> - `Flux training`
> - `ComfyUI vs Forge`
> - `GPU recommendations` 

- **Flux：新王者？**：成员们讨论了 Flux 吸收图像生成 AI 社区的潜力，每天都有新的 LoRA 和合并模型出现。
   - 一些人认为 Stability AI 需要尽快发布产品来竞争，因为 Flux 正成为 CivitAI 和 Hugging Face 上的主导力量。
- **Flux vs. SD3：登顶之争**：关于 Flux 是否与 SD3 有本质区别存在争论，这两个模型都使用了 DiT 架构、ret flow loss 和类似的 VAE 尺寸。
   - 关键区别在于 Flux dev 是从大模型蒸馏而来的，而 Stability AI 也可以采用这种策略。一些人更喜欢非蒸馏模型，即使图像质量较低。
- **Flux 训练：挑战与机遇**：成员们讨论了为 Flux 训练 LoRA 的挑战，指出训练代码尚未正式发布。
   - 一些用户正在探索本地训练 LoRA 的方法，而另一些用户则建议使用 Replicate 官方的 Flux LoRA Trainer 以获得更快、更简便的结果。
- **ComfyUI vs. Forge：UI 之战**：用户讨论了 ComfyUI 和 Forge 之间的性能差异，一些人发现 Forge 更快，特别是在批处理方面。
   - 讨论涉及了 Gradio 4 更新对 Forge 的影响以及未来改进的潜力。一些用户更喜欢 ComfyUI 的灵活性，而另一些用户则欣赏 Forge 的优化。
- **Stable Diffusion 的 GPU 推荐**：成员们分享了各种 GPU 在 Stable Diffusion 中的使用经验和性能，16GB VRAM 被认为是最低配置，24GB 则比较充裕。
   - 讨论涉及了 VRAM 比 CPU 速度更重要，以及 RAM 和其他应用程序对性能的影响。共识是尝试不同的模型和编码器，以找到最适合每个系统的配置。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://dsc.gg/vexel">Discord - 充满乐趣与游戏的群聊</a>: Discord 非常适合与朋友一起玩游戏和放松，甚至可以建立全球社区。自定义你自己的空间来聊天、玩耍和聚会。</li><li><a href="https://tenor.com/view/blender-vram-ram-memory-gone-gif-27551226">Blender Vram GIF - Blender Vram RAM - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/kto-lbow-hi-hello-hi-there-gif-25347432">Kto Lbow GIF - Kto Lbow Hi - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.instagram.com/p/C-vD5i4uVvY/">Aleksey Efremov 在 Instagram 上: &quot;ALERT 21
.
.
#weirdcore #liminalspace #backroomsaesthetic #wierdcore #backrooms #liminalcore #dreamcore #nastolgia #aiart #ai&quot;</a>: solar.w 于 2024 年 8 月 16 日: &quot;ALERT 21 . . #weirdcore #liminalspace #backroomsaesthetic #wierdcore #backrooms #liminalcore #dreamcore #nastolgia #aiart #ai&quot;. </li><li><a href="https://www.instagram.com/solar.w/reel/C-VZ21juqND/">Aleksey Efremov 在 Instagram 上: &quot;ALERT 19
&#x2026;
&#x2026;

#liminalspace #nostalgiacore #backrooms #afterhours #dreamcore #weirdcore&quot;</a>: solar.w 于 2024 年 8 月 6 日发布：&quot;警报 19 &#x2026; &#x2026; #liminalspace #nostalgiacore #backrooms #afterhours #dreamcore #weirdcore&quot;。 </li><li><a href="https://replicate.com/lucataco/ai-toolkit">lucataco/ai-toolkit – 在 Replicate 上通过 API 运行</a>：未找到描述</li><li><a href="https://stability.ai/stable-artisan">Stable Artisan &mdash; Stability AI</a>：Stable Artisan 是一款有趣的、多模态生成式 AI Discord 机器人，它在 Discord 生态系统中使用 Stability AI 平台 API 的产品。</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1euz2a9/union_flux_controlnet_running_on_comfyui_workflow/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://github.com/Acly/krita-ai-diffusion/wiki/ComfyUI-Setup">ComfyUI 设置</a>：在 Krita 中使用 AI 生成图像的流线型界面。支持带有可选文本提示的 Inpaint 和 Outpaint，无需微调。- Acly/krita-ai-diffusion</li><li><a href="https://www.youtube.com/watch?v=gO3Mk3le0qs">AI 面部一致性不是真正的目标：而是这个...</a>：关于 AI 视频中角色一致性需求的讨论很多，但在“真实”（叙事性）电影制作中，角色一致性只是其中一个元素...</li><li><a href="https://www.youtube.com/watch?v=GG-xDgdjhjU">Garuda Linux KDE Dr460nized - 快速回顾（演示）</a>：距离我上次关注 Garuda Linux 已经有一段时间了。它好用吗，是否能“开箱即用”？看我如何使用它，包括我遇到的一些小卡顿...</li><li><a href="https://youtu.be/0AT8esyY0Fw">Warp Fusion: 分步教程</a>：Warp Fusion 是一款出色的 AI 动画工具，可以制作出令人惊艳的视频。在本视频中，我将向你展示如何使用远程 GPU 逐步使用 Warp Fusion....</li><li><a href="https://youtu.be/bm1PWniLIlc?si=ZRIXbV1JHifS9L31">100% AI 制作的 4K 视频 | 茶的历史 - 穿越时空的旅行</a>：让自己沉浸在古老而迷人的茶历史中。探索这种拥有千年历史的饮品是如何连接文化、跨越大陆并演变的...</li><li><a href="https://youtu.be/j9I0iLxGJl0">RTX Remix I 通过 ComfyUI 使用 RTX 和数千个 AI 模型重制经典</a>：了解更多关于 NVIDIA RTX Remix 和强大的新 REST API 的信息：https://www.nvidia.com/en-us/geforce/news/rtx-remix-rest-api-comfyui-app-connectors NVIDIA 是...</li><li><a href="https://github.com/madebyollin/taesd">GitHub - madebyollin/taesd: 用于 Stable Diffusion 的微型自动编码器</a>：用于 Stable Diffusion 的微型自动编码器（Tiny AutoEncoder）。通过在 GitHub 上创建账号来为 madebyollin/taesd 的开发做出贡献。</li><li><a href="https://civitai.com/articles/391/tutorial-dreambooth-lora-training-using-kohyass">教程：使用 Kohya_SS 进行 Dreambooth LoRA 训练 | Civitai</a>：[6/24 编辑 - 封面图和输出已更新，以符合本网站更新后的指南。] [7/1 编辑 - Lycoris/LoCon 教程链接...</li><li><a href="https://github.com/kohya-ss/sd-scripts/tree/25f77f6ef04ee760506338e7e7f9835c28657c59?tab=readme-ov-file#flux1-lora-training-wip">GitHub - kohya-ss/sd-scripts (版本 25f77f6ef04ee760506338e7e7f9835c28657c59)</a>：通过在 GitHub 上创建账号来为 kohya-ss/sd-scripts 的开发做出贡献。</li><li><a href="https://civitai.com/models/396388/just-better-gun">Just Better Gun. - v1.0 | Stable Diffusion LoRA | Civitai</a>：Just Better Gun 基本上是一个围绕……枪……的 LoRA。数据集专注于具有大量瞄准镜和细节的战术战争枪械...</li><li><a href="https://tensor.art/models/759856135286068673/FLUX-HYPER-TRAINED-DREAM-DIFFUSION-BY-DICE-V-1">FLUX HYPER TRAINED - DREAM DIFFUSION - BY DICE - V 1 | Stable Diffusion 模型 - Checkpoint</a>：5.5K 运行，77 星标，53 下载。FLUX DREAM DIFFUSION BY DICE，从 Comfy 中的这些设置开始，感受它的运行效果……简单提示词：一架喷气式飞机 ...
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1274086399401922633)** (449 条消息🔥🔥🔥): 

> - `验证问题`
> - `Hermes 2.5`
> - `Mistral 的困境`
> - `模型合并 (Model Merging)`
> - `Open Empathic`

- **Hugging Face 验证问题**：一名成员在“使用 Hugging Face 登录”验证过程中遇到问题，登录按钮显示“未登录”。
   - 他们在移动端和桌面端都进行了尝试，但均未成功，并被建议稍后在 PC 上重试。
- **Hermes 2.5 表现优于 Hermes 2**：在添加了[代码指令示例](https://link.to.examples)后，**Hermes 2.5** 在各项基准测试中的表现似乎优于 **Hermes 2**。
   - Hermes 2 在 MMLU 基准测试中得分为 **34.5**，而 Hermes 2.5 得分为 **52.3**。
- **Mistral 难以扩展至 8k 以上**：成员们表示，如果不进行持续预训练，**Mistral** 无法扩展到 8k 以上，且[这是一个已知问题](https://link.to.issue)。
   - 他们指出，*mergekit* 和 *frankenMoE finetuning* 的进一步工作是性能提升的下一个前沿领域。
- **关于模型合并策略的讨论**：一名成员建议将 **UltraChat** 与基础 **Mistral** 之间的差异应用于 **Mistral-Yarn**，作为一种潜在的合并策略。
   - 其他人表示怀疑，但该成员保持乐观，并引用了过去在他们所谓的“诅咒模型合并 (cursed model merging)”方面的成功尝试。
- **Open Empathic 项目寻求协助**：一名成员呼吁帮助扩大 **Open Empathic** 项目的类别，特别是在低端部分。
   - 他们分享了一个 [Open Empathic 发布与教程的 YouTube 视频](https://youtu.be/GZqYr8_Q7DE)，引导用户从 YouTube 视频中贡献他们喜欢的电影场景，以及 [OpenEmpathic 项目本身](https://dct.openempathic.ai/)的链接。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://arxiv.org/abs/2204.03930">From Rewriting to Remembering: Common Ground for Conversational QA Models</a>：在对话式 QA 中，模型必须利用前几轮的信息来回答接下来的问题。当前的方法（如问题重写）在提取相关信息时面临困难...</li><li><a href="https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/howto_wsl.html">WSL 操作指南 - 在 Radeon GPU 上使用 ROCm &#8212; 在 Radeon GPU 上使用 ROCm</a>：未找到描述</li><li><a href="https://huggingface.co/parler-tts/parler-tts-mini-expresso/discussions/8#66be241be61ccd71d7b5cd7d">parler-tts/parler-tts-mini-expresso · 运行此模型和其他 parler 模型时出现问题...</a>：未找到描述</li><li><a href="https://www.kaggle.com/datasets/umerhaddii/saudi-aramco-stock-price-data">沙特阿美股票价格数据</a>：2015 - 2024 年沙特阿美股票价格数据</li><li><a href="https://huggingface.co/spaces/taneemishere/html-code-generation-from-images-with-deep-neural-networks">图像转 HTML 代码演示 - taneemishere 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://paperswithcode.com/sota/long-range-modeling-on-lra">Papers with Code - LRA 基准测试（长程建模）</a>：目前 LRA 上的 SOTA 是 Mega。查看 32 篇带有代码的论文的完整对比。</li><li><a href="https://huggingface.co/monadical-labs/minecraft-skin-generator-sdxl/tree/main/">monadical-labs/minecraft-skin-generator-sdxl 在 main 分支</a>：未找到描述</li><li><a href="https://huggingface.co/blog/tomaarsen/attention-sinks">🕳️ LLM 中的 Attention Sinks 以实现无尽的流畅度</a>：未找到描述</li><li><a href="https://fedoramagazine.org/using-artificial-intelligence-to-set-a-guitar-sound/">使用人工智能设置吉他音色 - Fedora Magazine</a>：使用人工智能结合 Guitarix 创建你的音色。包含解释和演示。</li><li><a href="https://youtu.be/OenUHAuTyxk">开发者阅读笔记 10：2 分钟了解 Figma 的 20 个概念</a>：在这个开发者笔记系列视频中，我将介绍 Figma 的 20 个基本概念，让你全面了解它是什么以及它是如何工作的。如果你还没有...</li><li><a href="https://www.youtube.com/watch?v=l9TCDEbRiKM">如果你在生活中难以找到幸福，请看这个 | 斯多葛主义</a>：感到生活停滞不前且不快乐？探索古老的斯多葛原则如何引导你走向真正的幸福和内心平静。这段视频分解了强大的斯...</li><li><a href="https://www.youtube.com/watch?v=zv14gyAJWUM">أعظم الدروس و الحكم التي تسمعها في حياتك، أنصحك بتعلمها</a>：人的一生会经历许多情况。有些是美好快乐的，有些是悲伤痛苦的，这些情况被视为有益的教训...</li><li><a href="https://huggingface.co/openai/whisper-large-v3">openai/whisper-large-v3 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/FunAudioLLM/SenseVoiceSmall">FunAudioLLM/SenseVoiceSmall · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/hella-mad-breakdance-cool-lit-gif-13584980">Hella Mad GIF - Hella Mad 地板舞 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://youtu.be/Gg8s9iNfExU?si=I3Xw8jEJ9YtjWnNY">给有抱负和经验丰富的开发者的顶级建议 🌟</a>：本视频列出了针对 IT 行业初学者或有经验开发者的顶级建议或技巧 ⭐ 历程 👇👉 我的 100 天代码挑战 https://youtu.be/...</li><li><a href="https://huggingface.co/spaces/AIPeterWorld/Doc-To-Dialogue?logs=container">Doc To Dialogue - AIPeterWorld 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://github.com/jxnl/instructor">GitHub - jxnl/instructor: 为 LLM 提供结构化输出</a>：为 LLM 提供结构化输出。通过在 GitHub 上创建账号来为 jxnl/instructor 的开发做出贡献。</li><li><a href="https://github.com/sandrohanea/whisper.net?tab=readme-ov-file">GitHub - sandrohanea/whisper.net: Whisper.net。使用 Whisper 模型让语音转文本变得简单</a>：Whisper.net。使用 Whisper 模型让语音转文本变得简单 - sandrohanea/whisper.net</li><li><a href="https://github.com/visioncortex/vtracer">GitHub - visioncortex/vtracer: 位图转矢量图形转换器</a>：位图转矢量图形转换器。通过在 GitHub 上创建账号来为 visioncortex/vtracer 的开发做出贡献。</li><li><a href="https://arxiv.org/abs/2306.06441v1">图像矢量化：综述</a>：如今，有许多扩散模型和自回归模型在从文本和其他输入领域生成图像方面表现出令人印象深刻的结果。然而，这些方法并非针对超高...</li><li><a href="https://www.linuxfoundation.org/press/linux-foundation-welcomes-the-open-model-initiative-to-promote-openly-licensed-ai-models">Linux Foundation 欢迎 Open Model Initiative 以推广开放许可的 AI 模型</a>：新的倡议将...</li>

促进免费使用、开放且符合伦理的 AI 模型的发展。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1274158930427379763)** (4 messages): 

> - `FP8 Training`
> - `Memory Reduction`
> - `Optimizer States` 


- **使用一阶动量的 FP8 训练实现了相似的 Loss**：使用 **FP8 中的一阶动量**训练 **1B FP8 模型**，在达到 **48k steps** 时表现平稳，其 Loss 与 **bfloat16** 相比偏移量仅为 **0.08**。
- **使用 FP8 优化器状态进行 FP8 训练是可行的**：使用 **FP8 优化器状态**训练 **1B FP8 模型**，与 **bfloat16 基准**相比实现了 **0.14 的偏移**，并减少了 **50% 的内存占用**。
- **使用混合动量类型的 FP8 训练**：使用 **FP8 中的一阶动量**和 **bfloat16 中的二阶动量**训练 **1B FP8 模型**，在 **31k steps** 时实现了与 **bfloat16** 相当的收敛性（偏移量为 **0.08**），并减少了 **42% 的内存占用**。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1274238666885173248)** (3 messages): 

> - `Medical SAM 2`
> - `MedGraphRAG`
> - `Multimodal LLM for Medical Time Series`
> - `ECG-FM`
> - `Private & Secure Healthcare RAG` 


- **用于视频医学图像分割的 Medical SAM 2**：Medical SAM 2 是一篇新的研究论文，专注于将医学图像作为视频进行分割。
   - 该论文解决了医疗领域对高效准确的视频图像分割的需求，为分析和解释动态医学数据提供了一种新颖的方法。
- **MedGraphRAG：图增强的医学 RAG**：MedGraphRAG 是一个图增强的医学 RAG 模型，利用图网络的强大功能来增强医学信息检索。
   - 该论文通过将基于图的表示与 RAG 能力相结合，解决了理解复杂医学关系和从医学文本中提取相关知识的挑战。
- **用于医学时间序列的多模态 LLM**：该研究论文介绍了一种专门为处理医学时间序列数据而设计的新型多模态 LLM。
   - 该模型结合了语言和时间序列数据的力量，为医疗应用中更全面、更深入的分析铺平了道路。
- **开源心电图基础模型 - ECG-FM**：ECG-FM 是一个专为心电图分析设计的开源心电图基础模型。
   - 该论文促进了心电图分析领域的开放研究和协作，为医学从业者和研究人员提供了宝贵的资源。
- **隐私与安全的医疗 RAG**：该论文深入探讨了隐私与安全医疗 RAG 的开发，这是在医学信息检索中保护患者数据的关键进展。
   - 该研究通过提供安全且负责任地访问医疗信息的框架，解决了医疗背景下的隐私和安全这一关键问题。



**提到的链接**：<a href="https://x.com/OpenlifesciAI/status/1824790439527887073">来自 Open Life Science AI (@OpenlifesciAI) 的推文</a>：上周及本周医学 AI 动态：顶级研究论文/模型 🏅 (2024年8月3日 - 8月17日) - Medical SAM 2: 将医学图像作为视频分割 - MedGraphRAG: 图增强的医学 RAG - 多模态 ...

  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1274188235676979262)** (18 messages🔥): 

> - `Unity ML Agents`
> - `CursorLens`
> - `Batching APIs`
> - `CuminAI`
> - `NeuroSync`

- **Wandering Agent 3 - 从零开始的 C# 实况训练**：一位 Unity ML Agent 开发者正在直播其 Wandering Agent 项目的第 3 部分，重点是利用 Unity ML Agents 从零开始编写 SAC Agent，并在前几集的基础上进行构建。
   - 他们正在使用 C#，并计划保留现有的摄像机脚本。本集将重点关注从零开始编写 SAC Agent。
- **CursorLens：用于 Prompt 分析和模型配置的开源仪表板**：开发者发布了 CursorLens，这是一个开源仪表板，用于可视化 Prompt 分析并配置 Cursor 本身不提供的模型。
   - 该仪表板已在 ProductHunt 上线，旨在提供对 Prompt 性能的洞察，并允许对模型进行自定义。
- **开源模型的 Batching API**：OpenAI 和 Google 等大公司已经为其模型推出了 Batching API，与普通请求相比可以节省成本，但缺乏处理保证、SLA 和重试机制。
   - CuminAI 为开源模型创建 Batching API 提供了一种解决方案，为现有 API 提供了一个强大的替代方案。
- **NeuroSync：用于面部 Blendshape 预测的 Seq2Seq Transformer**：NeuroSync 是一个序列到序列（Seq2Seq）的 Transformer 模型，旨在根据音频特征输入预测面部 Blendshape 帧。
   - 该模型使用 4 个 Transformer 层和 4 个 Attention Head，使其成为 HuggingFace 上第一个专门用于从音频预测面部 Blendshape 的模型。
- **阿拉伯语 Whisper 模型训练与部署**：一个 YouTube 播放列表通过在阿拉伯语语音数据集上训练 Whisper 模型来教授阿拉伯语语音识别。
   - 该模型随后被部署在 HuggingFace Models 和 Spaces 上，为阿拉伯语语音识别提供了宝贵的资源。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/ardha27/VideoAnalyzer">VideoAnalyzer - 由 ardha27 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/AnimaVR/NeuroSync-0.1a">AnimaVR/NeuroSync-0.1a · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/ardha27/Youtube-AI-Summarizer">Youtube AI Summarizer - 由 ardha27 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/Q-bert/ChessLlama">Q-bert/ChessLlama · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/AIPeterWorld/Doc-To-Dialogue?logs=container">Doc To Dialogue - 由 AIPeterWorld 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/shauninkripped/Sentient-Aid-space1">Sentient AId Test Space 01 - 由 shauninkripped 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=ZgUVQkhiPi8">原型预览 4：最终对齐。实时且本地的音频转面部模型</a>: 仅通过音频实现 AI 面部动画的原型演示。观看仅凭音频特征输入实现的完整面部动画（51 个 blendshapes）！huggingface.co/AnimaVR/NeuroSync-0.1a</li><li><a href="https://www.producthunt.com/posts/cursor-lens"> CursorLens - 适用于 Cursor IDE 的开源仪表盘和分析工具 | Product Hunt</a>: 一个用于 Cursor.sh IDE 的开源仪表盘。记录 AI 代码生成、跟踪使用情况并控制 AI 模型（包括本地模型）。可在本地运行或使用即将推出的托管版本。</li><li><a href="https://youtube.com/live/1jITphnPvJU?feature=share">Unity ML Agents | 从零开始的 C# 实况训练 | 第 3 部分</a>: 一个运行在 3D 体素世界中的快速 SAC Agent 训练器。</li><li><a href="https://github.com/ra9hur/Decision-Transformers-For-Trading">GitHub - ra9hur/Decision-Transformers-For-Trading</a>: 通过在 GitHub 上创建账号，为 ra9hur/Decision-Transformers-For-Trading 的开发做出贡献。</li><li><a href="https://blog.cuminai.com/how-to-get-a-batching-api-like-openai-for-open-source-models-824529788a49">如何为开源模型获取类似 OpenAI 的 Batching API</a>: 在 AI 世界中，高效处理和成本管理至关重要。实现这一目标的一种强大方法是 Batching，它……</li><li><a href="https://github.com/U-C4N/byte-terminal">GitHub - U-C4N/byte-terminal: 使用 Groq API 的 AI 驱动交互式聊天界面。具有类终端 UI、本地存储和可自定义命令。非常适合探索 LLM 的开发者。</a>: 使用 Groq API 的 AI 驱动交互式聊天界面。具有类终端 UI、本地存储和可自定义命令。非常适合探索 LLM 的开发者。 - U-C4N/byte-terminal</li><li><a href="https://www.youtube.com/watch?v=NiRyViZ8tEw&list=PL6ViV90w3mloHsKo6qi8oIsW_ZDswSrKK&index=5">AI 语音识别系统训练系列（第四部分）部署 Whisper 模型</a>: 在 Hugging Face 命名空间部署 Whisper small Arabic。Hugging Face 链接：https://huggingface.co/mohammed vastai 链接：https://cloud.vast.ai/?ref_id=145398</li><li><a href="https://huggingface.co/mohammed">mohammed (Mohammed Bakheet)</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/cfahlgren1/SmolPilot">SmolPilot - 由 cfahlgren1 创建的 Hugging Face Space</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1274401447416692889)** (35 条消息🔥): 

> - `用于渗透测试的 LLMs`
> - `录制问题`
> - `HuggingFace 读书会`
> - `开源模型的 Batching API`
> - `交叉发布` 


- **LLMs 在渗透测试方面表现越来越好**：Hugging Face 读书会重点讨论了如何利用 LLMs 进行渗透测试。
- **录音中带有鼓声噪音**：会议录音中出现了来自演讲者麦克风的类似鼓声的噪音。
- **OpenAI 和 Gemini 的 Batch API**：一名成员正在寻找可以发布关于开源模型 Batching API 文章的地方。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://topmate.io/athul_nambiar">Athul Nambiar</a>: 🧑🏻‍💻 MERN Stack | 🛠️ TL Labs ❤️ | 🅾️ Tublian 开源贡献者 🚀 | 🎒 学生 | 🧑🏻‍💻 程序员 | 💰 投资者 | 🎨 设计师 | *️⃣ 3 次黑客松冠军 🏆（州级、国家级）</li><li><a href="https://medium.com/gopenai/understanding-penetration-testing-with-llms-2b0ec6add14a.">无标题</a>：未找到描述</li><li><a href="https://youtu.be/_f16ofdVC8g">Hugging Face Reading Group 27: Understanding Penetration Testing with LLMs</a>：抱歉，背景/我的麦克风似乎有一点鼓声。演讲者：Isamu Isozaki, Manil Shrestha。往期演示：https://githu...</li><li><a href="https://githu...)>>>">无标题</a>：未找到描述</li><li><a href="https://docs.google.com/presentation/d/1OF_wqUsbbsFoAu4XZFlcaaOmnZngrUuvQrTstcogZ5c/edit#slide=id.g2f35c1725d3_0_549">渗透测试演示文稿</a>：用于渗透测试的 AI，演讲者：Isamu Isozaki, Manil Shrestha</li><li><a href="https://docs.google.com/presentation/?usp=slides_web">无标题</a>：未找到描述</li><li><a href="https://accounts.google.com/ServiceLogin?service=wise&passive=1209600&osid=1&continue=https://docs.google.com/presentation/d/1OF_wqUsbbsFoAu4XZFlcaaOmnZngrUuvQrTstcogZ5c/edit&followup=https://docs.google.com/presentation/d/1OF_wqUsbbsFoAu4XZFlcaaOmnZngrUuvQrTstcogZ5c/edit&ltmpl=slides&ec=GAZAmQI)">无标题</a>：未找到描述</li><li><a href="https://support.google.com/docs/answer/2375082?hl=en).[Dismiss](#)">系统要求和浏览器 - 电脑 - Google Docs 编辑器帮助</a>：未找到描述</li><li><a href="https://www.mozilla.org/firefox/new/)">下载史上最快的 Firefox</a>：更快的页面加载速度，更低的内存占用，功能丰富，全新的 Firefox 现已发布。</li><li><a href="https://www.microsoft.com/windows/microsoft-edge)">通过 Windows 11 操作系统、电脑和应用体验 AI 的力量 | Microsoft Windows</a>：体验最新的 Microsoft Windows 11 功能。了解我们最新的 Windows 操作系统如何为您提供更多工作、娱乐和创作方式。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1274109910732505098)** (4 messages): 

> - `Pokemon classification`
> - `HuggingFace Datasets`
> - `Deep learning`
> - `Stanford Computer Vision`
> - `CV Community Course` 


- **Pokemon Classification - 问题与调试**：一位用户在使用 [HuggingFace Pokémon classification dataset](https://huggingface.co/datasets/fcakyon/pokemon-classification) 进行宝可梦分类时遇到困难，并分享了数据集的下载路径，这表明数据集本身或用户的配置可能存在问题。
   - 用户提供了其 [notebook](https://github.com/alefram/notebooks/blob/master/pokedex.ipynb) 的链接，但未分享具体的错误信息或模型详情以便进一步协助。
- **寻求 Computer Vision 职业路径建议**：一位寻求进入 Computer Vision 领域建议的用户分享了其现有的知识储备，包括斯坦福大学的 Computer Vision 课程和 Deep learning 背景。
   - 回复建议查看 [HuggingFace's CV Community Course](https://huggingface.co/community/courses) 获取指导，并加入 HuggingFace Discord 的 [Computer Vision Channel](<#1156125722151239750>) 进行进一步讨论。
- **阿里巴巴达摩院（Alibaba DAMO）发布 VideoLLaMA 2-72B**：阿里巴巴达摩院发布了新的视频 LLM —— **VideoLLaMA 2-72B**。
   - 该模型和 Demo 分别可以在 [HuggingFace](https://huggingface.co/collections/DAMO-NLP-SG/videollama-2-6669b6b6f0493188305c87ed) 和 [HuggingFace Spaces](https://huggingface.co/spaces/lixin4ever/VideoLLaMA2) 上找到，并附有 HuggingFace 上的 [研究论文](https://huggingface.co/papers/2406.07476) 链接。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/AdeenaY8/status/1823640386323042456">来自 Adina Yakup (@AdeenaY8) 的推文</a>: 🎥 来自中国社区的新 Video-LLMs 更新！@AlibabaDAMO 发布了 VideoLLaMA 2-72B 🔥 模型：https://huggingface.co/collections/DAMO-NLP-SG/videollama-2-6669b6b6f0493188305c87ed Demo: http...</li><li><a href="https://huggingface.co/datasets/fcakyon/pokemon-classification">fcakyon/pokemon-classification · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://github.com/alefram/notebooks/blob/master/pokedex.ipynb">alefram/notebooks master 分支下的 notebooks/pokedex.ipynb</a>: 关于 Machine learning 和控制内容的 Notebooks - alefram/notebooks
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1274140925752246303)** (10 messages🔥): 

> - `PDF table extraction`
> - `docTR library`
> - `NLP resources`
> - `Open Source Model for data extraction`
> - `GPT-4 for data extraction` 


- **PDF 表格提取难题**：一位成员分享了他们在使用 **pdfplumber** 从多页 PDF 中提取表格时遇到的困难。
   - 他们报告了在保留词间距和正确提取文本方面的问题。
- **用于 OCR 的 docTR 库**：另一位成员建议使用 **docTR 库** 进行表格提取和 OCR 任务。
   - 他们分享了 [docTR GitHub 仓库](https://github.com/mindee/doctr) 的链接以供进一步探索。
- **寻求初学者友好的 NLP 资源**：一位成员表示有兴趣寻找适合初学者的 NLP 学习资源。
   - 他们还请求一份学习 NLP 的路线图（roadmap）。
- **用于数据提取的开源模型**：一位成员正在寻找一个优秀的开源模型，用于从身份证件（IDs）等图像中提取数据。
   - 他们提到曾尝试使用 **GPT-4** 来实现此目的，但发现结果不尽如人意。
- **用于数据提取的 GPT-4**：一位成员尝试使用 **GPT-4** 从图像中提取数据，但报告结果不理想。



**提及的链接**: <a href="https://github.com/mindee/doctr">GitHub - mindee/doctr: docTR (Document Text Recognition) - 一个由 Deep Learning 驱动的、无缝、高性能且易于使用的 OCR 相关任务库。</a>: docTR (Document Text Recognition) - 一个由 Deep Learning 驱动的、无缝、高性能且易于使用的 OCR 相关任务库。 - mindee/doctr

  

---

### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1274196988489105460)** (12 messages🔥): 

> - `ComfyUI Lora Conversion`
> - `Diffusers Lora Format`
> - `Llama 3.1 Pruning`
> - `Diffusion Model Deblurring`
> - `Flux txt_ids` 


- **针对 FLUX 的 ComfyUI Lora 转换**：一位用户询问是否有脚本可以将 ComfyUI Lora 转换为 Diffusers Lora 格式，以便在 FLUX 中使用。
   - 他们寻求这种转换是为了在分阶段加载 FLUX 时能够加载 LoRA 权重。
- **寻找 Diffusers 格式的 LoRA**：一位用户询问是否已有专门为 Diffusers 格式化且适用于 FLUX 的 LoRA。
   - 他们有兴趣测试在分阶段加载 FLUX 时 "load_lora_weights" 是否能有效运行。
- **使用 Diffusion 模型进行去模糊**：一位用户寻求关于适合图像去模糊的 Diffusion 模型的指导，并承认此类模型对于该任务可能有些大材小用。
   - 他们被推荐了一个用于指令微调（instruction-tuning）Stable Diffusion 的 GitHub 仓库，并被鼓励探索其他去模糊方法。
- **基于空间-时间偏移的视频修复**：一位用户分享了一篇关于使用轻量级空间-时间偏移方法进行视频修复的学术论文，旨在实现高效的帧间聚合。
   - 该论文提出了一个基于分组空间偏移的框架，用于捕捉帧间对应关系并实现广阔的感受野，从而提高视频修复性能。
- **理解 Flux 的 txt_ids**：一位用户询问 Flux 的 Transformer 中 'txt_ids' 变量的作用，并观察到它在 Diffusers 流水线中始终是一个零张量（zero tensor）。
   - 他们想知道这是否是某个未发布的更大版本 Flux 模型的残留，或者在当前实现中是否有其他功能。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2206.10810">A Simple Baseline for Video Restoration with Grouped Spatial-temporal Shift</a>：视频修复旨在从退化的视频中恢复清晰的帧，具有许多重要的应用。视频修复的关键取决于利用帧间信息。然而，现有的...</li><li><a href="https://github.com/huggingface/instruction-tuned-sd">GitHub - huggingface/instruction-tuned-sd: Code for instruction-tuning Stable Diffusion.</a>：用于指令微调 Stable Diffusion 的代码。可以通过在 GitHub 上创建账户来为 huggingface/instruction-tuned-sd 的开发做出贡献。
</li>
</ul>

</div>
  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1274081009360568351)** (242 messages🔥🔥): 

> - `Android Unsloth`
> - `llama 3.1 70B`
> - `Mistral 8k`
> - `Mistral merging`
> - `Open Empathic` 


- **Android 版 Unsloth 指南**：一位用户询问在 Android 上使用 **unsloth/gemma-2b-bnb-4bit** 的指南。
   - 建议他们使用 **TorchChat** [https://github.com/pytorch/torchchat](https://github.com/pytorch/torchchat)，用于在服务器、桌面和移动设备上本地运行 PyTorch LLMs。
- **Mistral 难以扩展超过 8k**：一位成员表示，如果不进行持续预训练（continued pretraining），**Mistral** 无法扩展到 8k 以上。
   - 他们指出，在 *mergekit* 和 *frankenMoE finetuning* 方面的进一步工作是性能的下一个前沿。
- **关于模型合并策略的讨论**：一位成员建议将 **UltraChat** 和基础 **Mistral** 之间的差异应用到 **Mistral-Yarn**，作为一种潜在的合并策略。
   - 其他人表示怀疑，但该成员保持乐观，并引用了过去在他们所谓的“诅咒模型合并”（cursed model merging）方面的成功尝试。
- **Open Empathic 项目寻求援助**：一位成员呼吁帮助扩展 **Open Empathic** 项目的类别，特别是在底层类别方面。
   - 他们分享了一个 [Open Empathic 发布与教程的 YouTube 视频](https://youtu.be/GZqYr8_Q7DE)，指导用户贡献他们喜欢的 YouTube 电影场景，以及 [OpenEmpathic 项目本身](https://dct.openempathic.ai/)的链接。
- **OpenAI CTO 的香水**：一位用户询问 OpenAI CTO **Mira Murati** 闻起来是什么味道。
   - 这个问题得到了幽默的回应和猜测。


<div class="linksMentioned">

<strong>提及的链接</strong>:

</div>

<ul>
<li>
<a href="https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama#id-6.-alpaca-dataset">如何微调 Llama-3 并导出到 Ollama | Unsloth 文档</a>：为在 Ollama 上本地运行而创建自定义个人助手（类似 ChatGPT）的入门指南</li><li><a href="https://docs.vllm.ai/en/latest/models/lora.html">使用 LoRA 适配器 — vLLM</a>：未找到描述</li><li><a href="https://huggingface.co/neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16">neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16#open-llm-leaderboard-evaluation-scores).">neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16 · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/vikhyatk/status/1824709909134602349">来自 vik (@vikhyatk) 的推文</a>：有人知道 gemma 分词器（tokenizer）中的这个 "&lt;2mass&gt;" token 是做什么用的吗？</li><li><a href="https://huggingface.co/unsloth/Meta-Llama-3.1-8B-bnb-4bit">unsloth/Meta-Llama-3.1-8B-bnb-4bit · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/">unsloth (Unsloth AI)</a>：未找到描述</li><li><a href="https://huggingface.co/models?other=unsloth">Models - Hugging Face</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/688">unsloth 4bit 模型无法在 vLLM 中加载 - 提示缺失适配器路径或名称 · Issue #688 · unslothai/unsloth</a>：当我尝试使用 llm = LLM("unsloth/mistral-7b-instruct-v0.3-bnb-4bit", dtype="half") 加载 unsloth 4bit 模型时，收到错误 Cannot find any of ['adapter_name_or_path'...</li><li><a href="https://github.com/pytorch/torchchat">GitHub - pytorch/torchchat: 在服务器、桌面和移动端本地运行 PyTorch LLM</a>：在服务器、桌面和移动端本地运行 PyTorch LLM - pytorch/torchchat</li><li><a href="https://docs.unsloth.ai/tutorials/how-to-f">Unsloth 文档</a>：未找到描述</li><li><a href="https://github.com/cognitivecomputations/grokadamw">GitHub - cognitivecomputations/grokadamw</a>：通过在 GitHub 上创建账户来为 cognitivecomputations/grokadamw 的开发做出贡献。</li><li><a href="https://github.com/huggingface/transformers/pull/32521">由 ehartford 提交的增加对 GrokAdamW 优化器的支持 · Pull Request #32521 · huggingface/transformers</a>：此 PR 做了什么？增加对 GrokAdamW 优化器的支持。此 PR 为 transformers 库增加了对 GrokAdamW 优化器的支持。引入的变更：将 GrokAdamW 优化器集成到...</li><li><a href="https://arxiv.org/abs/2405.20233">Grokfast：通过放大慢梯度加速 Grokking</a>：机器学习中一种被称为 Grokking 的令人困惑的现象是指，在对训练数据近乎完美过拟合后的数万次迭代后，才实现了延迟泛化。专注于长期的...</li><li><a href="https://github.com/ironjr/grokfast">GitHub - ironjr/grokfast：论文 "Grokfast: Accelerated Grokking by Amplifying Slow Gradients" 的官方仓库</a>：论文 "Grokfast: Accelerated Grokking by Amplifying Slow Gradients" 的官方仓库 - ironjr/grokfast
</li>
</ul>

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1274801645955448905)** (44 messages🔥): 

> - `RAG Reranker`
> - `RAG effectiveness`
> - `RAG vs. cosine similarity`
> - `Embeddings and RAG`
> - `Noise Filtering` 


- **Reranker 可以提升 RAG 结果**：使用 Reranker 来精炼 RAG 的结果可以显著提高性能。
   - Reranker 比初始检索阶段慢，但可以弥补 RAG 中不太可靠的排名，尽管其质量取决于上下文以及 Reranker 是否理解该主题。
- **RAG 并不总是如预期般工作**：虽然 RAG 搭建简单，但要精通却具有挑战性，往往达不到预期。
   - 链接中的电子书提供了关于当 RAG pipeline 未能如预期工作时如何处理的见解，重点是将 Reranker 作为解决方案。
- **Reranker vs. Cosine Similarity**：讨论了 Reranker 与用于 Embedding 检索的 Cosine Similarity 相比的有效性。
   - 虽然发现基于 Alibaba-NLP/gte-* 等模型的 Embedding 进行 Cosine Similarity 计算是可靠的，但 Reranker 可以进一步提升性能，特别是在 RAG 场景下。
- **处理 RAG 中的噪声文档**：讨论了如何从 RAG 结果中过滤掉像日志文件这样的“噪声文档”。
   - 建议包括使用正则表达式、Perplexity 作为指标，以及使用 Mirascope 等工具来过滤掉不需要的文档。
- **Perplexity 作为噪声的衡量指标**：提出将 Perplexity 作为一种指标，帮助识别并过滤 RAG 结果中的噪声文档。
   - Perplexity 衡量模型预测下一个 Token 的能力，较高的值表示在日志文件等未见数据上表现较差。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.pinecone.io/learn/series/rag/rerankers/">Rerankers and Two-Stage Retrieval | Pinecone</a>: 未找到描述</li><li><a href="https://towardsdatascience.com/perplexity-intuition-and-derivation-105dd481c8f3">Perplexity Intuition (and Derivation)</a>: 让你不再为 Perplexity 感到困惑。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1274082462280060948)** (209 条消息🔥🔥): 

> - `Llama fine-tuning`
> - `RAG`
> - `Class weights`
> - `Dataset size`
> - `GPU requirements` 


- **为什么我的 Llama 3 fine-tuned 模型表现不对？**：一位用户询问为什么他们 fine-tune 的 Llama 3 8B 模型会出现很多错误，甚至是在训练数据集中的问题上。
   - 几位用户建议这可能是由 tokenizer、instruction template、dataset size 或其他因素引起的问题。他们建议阅读 Alpaca 论文以获取更多信息。
- **Llama 3.1 70B fine-tuning 需要多少 GPU？**：一位用户询问了 fine-tuning Llama 3.1 70B 模型所需的 GPU 和 RAM 要求。
   - 用户回答说 70B 模型至少需要 48GB VRAM，根据经验法则，VRAM 需求应该是模型 4-bit quantization 大小加上几个 GB。
- **Gemma 2 可以针对波斯语任务进行 fine-tuning 吗？**：一位用户询问 Gemma 2 27B 模型是否可以针对波斯语任务进行 fine-tuning。
   - 另一位用户分享了他们在波斯语维基百科数据集上尝试 fine-tuning Gemma 2 的经验，提到 loss 没有下降。他们建议增加 rank 值并降低 learning rate 以尝试改进训练。
- **在 Windows 上安装 Unsloth**：一位用户报告了在 Windows 上使用 conda 安装 Unsloth 时遇到的依赖冲突问题。
   - 另一位用户建议改用 WSL2，因为 Windows 上的 conda 安装不能保证正常工作。
- **在 VLLM 中运行 Unsloth 模型**：一位用户询问关于保存大型 quantized 4-bit 模型以便在多 GPU 上配合 VLLM 使用的问题。
   - 另一位用户建议将模型保存在本地，因为 VLLM 尚不支持带有 tensor parallelism 的 BitsAndBytes 量化。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://tenor.com/view/goal-gif-5197357661024011864">Goal GIF - Goal - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: 查看下方列表以获取我们所有的 notebook：</li><li><a href="https://huggingface.co/datasets/roneneldan/TinyStories">roneneldan/TinyStories · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py">transformers/src/transformers/models/gpt2/modeling_gpt2.py at main · huggingface/transformers</a>: 🤗 Transformers: 为 Pytorch, TensorFlow, 和 JAX 提供的尖端机器学习库。 - huggingface/transformers</li><li><a href="https://github.com/karpathy/LLM101n?tab=readme-ov-file">GitHub - karpathy/LLM101n: LLM101n: Let&#39;s build a Storyteller</a>: LLM101n: 让我们构建一个故事讲述者。通过在 GitHub 上创建账号为 karpathy/LLM101n 的开发做出贡献。</li><li><a href="https://huggingface.co/blog/vivien/llm-decoding-with-regex-constraints">Fast, High-Fidelity LLM Decoding with Regex Constraints</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama)">Unsloth Documentation</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/73">Conda installation detailed instructions · Issue #73 · unslothai/unsloth</a>: 我正尝试按照说明在 conda 环境中安装 unsloth，问题是 conda 在运行安装行时卡住了。我已经尝试运行了两次，结果都...</li><li><a href="https://huggingface.co/docs/transformers/internal/generation_utils#transformers.RepetitionPenaltyLogitsProcessor">Utilities for Generation</a>: 未找到描述
</li>
</ul>

</div>

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1274714152534802503)** (6 条消息): 

> - `Ghost 8B Beta (1608) 发布`
> - `Ghost 8B Beta 对比其他模型`
> - `Ghost 8B Beta 多语言能力`
> - `Llama 许可证合规性`
> - `Ghost 8B Beta 训练过程` 


- **Ghost 8B Beta (1608) 已发布**：**Ghost 8B Beta (1608)** 是一款性能顶尖的语言模型，具有无与伦比的多语言支持和成本效益，现已发布。
   - 在胜率（winrate）评分方面，它的表现优于 Llama 3.1 8B Instruct、GPT-3.5 Turbo、Claude 3 Opus、GPT-4 等模型。
- **Ghost 8B Beta 的多语言实力**：**Ghost 8B Beta** 现在支持 **16 种语言**，包括英语、越南语、西班牙语、中文等。
   - 它提供两种上下文选项（8k 和 128k），并改进了数学、推理和指令遵循能力，以更好地处理任务。
- **Ghost 8B Beta 超越竞争对手**：在 AlpacaEval 2.0 胜率评分中，**Ghost 8B Beta** 的表现优于 Llama 3.1 8B Instruct、GPT 3.5 Turbo、Claude 3 Opus、Claude 3 Sonnet、GPT-4 和 Mistral Large。
   - 这一令人印象深刻的表现突显了其卓越的知识能力和多语言实力。
- **Llama 许可证与模型命名**：一位成员指出，Llama 许可证要求基于其构建的模型在名称中必须包含 “Llama”。
   - 开发者澄清说，目前的模型名称是简称，在 HuggingFace 上可以找到符合许可证要求的全名。
- **Ghost 8B Beta 训练过程**：开发者解释说，他们的训练过程不同于标准的 Fine-tuning，涉及数据准备、多语言训练、Fine-tuning 和反馈。
   - 他们强调，所有数据和代码都已 Fork 并更新以匹配其训练“配方”（recipe），这一过程使他们的模型脱颖而出。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/meta-llama/llama-models/blob/3dea71ccb22da158b88a723a1374e36642e3a12e/models/llama3_1/LICENSE#L24">llama-models/models/llama3_1/LICENSE at 3dea71ccb22da158b88a723a1374e36642e3a12e · meta-llama/llama-models</a>：旨在用于 Llama 模型的实用程序。通过在 GitHub 上创建账户，为 meta-llama/llama-models 的开发做出贡献。</li><li><a href="https://huggingface.co/spaces/lamhieu/ghost-8b-beta-8k">Ghost 8B Beta (β, 8k) - a Hugging Face Space by lamhieu</a>：未找到描述</li><li><a href="https://huggingface.co/ghost-x/ghost-8b-beta-1608">ghost-x/ghost-8b-beta-1608 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/collections/ghost-x/ghost-8b-beta-668ead6179f93be717db4542">Ghost 8B Beta - a ghost-x Collection</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1274129732622876807)** (15 messages🔥): 

> - `Code Editing with LLMs` (使用 LLM 进行代码编辑)
> - `Reasoning Gap in LLMs` (LLM 中的推理差距)
> - `LLM Inference Optimization` (LLM 推理优化)
> - `LLM Ensemble Techniques` (LLM 集成技术)
> - `Patched Round-Trip Correctness (Patched RTC)` 


- **Code Editing with LLMs**: 一篇新论文探讨了如何根据用户指令使用 Large Language Models (LLMs) 进行代码编辑。
   - 它引入了 EditEval（一个用于评估代码编辑性能的新型基准测试）和 InstructCoder（一个用于代码编辑指令微调 LLM 的数据集，包含超过 114,000 个指令-输入-输出三元组）。
- **Reasoning Gap in LLMs**: 一篇研究论文提出了一个框架，通过使用基准测试的功能变体（特别是 MATH 基准测试）来评估 LLM 的推理能力。
   - 它将“推理差距”定义为：将任务作为编程问题解决与作为自然语言问题解决之间的性能差异，并强调 LLM 在任务以代码形式呈现时通常表现更好。
- **Boosting LLM Performance with Patched MOA**: Patched MOA (Mixture of Agents) 作为一种推理优化技术被引入，旨在增强 LLM 在各种软件开发任务中的性能。
   - 该方法结合了 Best of N、Mixture of Agents 和 Monte Carlo Tree Search 算法，以提高较小模型的性能，使其能以极低的成本超越较大模型。
- **LLM Ensemble Techniques: Self-Consistency and Routing**: 讨论涉及了将模型集成用于数据集生成、评分设置和自我评估等任务。
   - 自一致性（Self-consistency，即从模型集成中选择最常见的答案）被强调为一种极具前景的方法，文中还引用了关于 LLM 路由（routing）和集成的早期工作。
- **Patched Round-Trip Correctness for Evaluating LLMs**: Patched Round-Trip Correctness (Patched RTC) 被提出作为一种新型的 LLM 评估技术，专注于“外环”软件开发任务，如 Bug 修复和代码审查。
   - 它扩展了原始的 Round-Trip Correctness 方法，允许在无需人工干预的情况下进行自我评估，并衡量模型响应的一致性和鲁棒性。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://yaofu.notion.site/How-does-GPT-Obtain-its-Ability-Tracing-Emergent-Abilities-of-Language-Models-to-their-Sources-b9a57ac0fcf74f30a1ab9e3e36fa1dc1">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: 一个将日常工作应用融合在一起的新工具。它是为您和您的团队打造的一体化工作空间。</li><li><a href="https://arxiv.org/abs/2311.08692">Routing to the Expert: Efficient Reward-guided Ensemble of Large Language Models</a>: Large Language Models (LLM) 的互补潜力假设现成的 LLM 在广泛的领域和任务中拥有异构的专业知识，因此 LLM 的集成可以实现一致的……</li><li><a href="https://arxiv.org/abs/2407.21075">Apple Intelligence Foundation Language Models</a>: 我们介绍了为 Apple Intelligence 功能提供支持的基础语言模型，包括一个旨在设备上高效运行的约 30 亿参数模型和一个大型基于服务器的语言模型……</li><li><a href="https://arxiv.org/abs/2407.18521">Patched MOA: optimizing inference for diverse software development tasks</a>: 本文介绍了 Patched MOA (Mixture of Agents)，这是一种推理优化技术，可显著增强 Large Language Models (LLMs) 在各种软件开发任务中的性能……</li><li><a href="https://arxiv.org/abs/2310.20329">InstructCoder: Instruction Tuning Large Language Models for Code Editing</a>: 代码编辑涵盖了开发人员每天处理的各种实用任务。尽管它具有相关性和实际用途，但在不断发展的领域中，自动代码编辑仍然是一个尚未得到充分探索的领域……</li><li><a href="https://arxiv.org/abs/2407.16557">Patched RTC: evaluating LLMs for diverse software development tasks</a>: 本文介绍了 Patched Round-Trip Correctness (Patched RTC)，这是一种应用于各种软件开发任务的新型 Large Language Models (LLMs) 评估技术，特别关注……</li><li><a href="https://arxiv.org/abs/2402.19450">Functional Benchmarks for Robust Evaluation of Reasoning Performance, and the Reasoning Gap</a>: 我们提出了一个利用基准测试的功能变体来稳健评估语言模型推理能力的框架。解决推理测试的模型在性能上不应表现出差异……
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1274540680198492191)** (1 messages): 

> - `Linear Transformers`
> - `Softmax Matching`
> - `Chunked Algorithm` 


- **Nous Research 发布 Linear Transformer 变体**：Nous Research 发布了一项关于 Linear Transformer 变体的研究，该变体可以匹配 Softmax，从而实现 O(t) 而非 O(t^2) 的训练复杂度。
   - 该研究论文可在 [此处](https://manifestai.com/articles/symmetric-power-transformers/) 获取，探讨了这种新变体及其对训练效率的影响。
- **作为线性成本 RNNs 的 Linear Transformers**：Linear Transformers 可以被表述为线性成本的 RNNs，与传统的 Transformers 相比，它提供了更好的理论上下文扩展（Context Scaling）。
   - 这一概念此前已在 Nous Research 的 [前一篇文章](https://manifestai.com/articles/linear-transformers-are-faster/) 中进行了探讨，该文章强调了用于 Linear Transformers 的分块算法（Chunked Algorithm）的效率。



**提及的链接**：<a href="https://manifestai.com/articles/symmetric-power-transformers/">Symmetric Power Transformers - Manifest AI</a>：一种 Linear Transformer，其学习方式类似于常规 Transformer，且状态可以容纳在 GPU 中。

  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1274116577289179199)** (20 messages🔥): 

> - `Falcon Mamba 7B`
> - `UBI and AI`
> - `AI Doomsday`
> - `Military Rations`
> - `AI Consciousness` 


- **Falcon Mamba 7B 性能超越 Llama 3 8B**：一段宣布 **Falcon Mamba 7B** 发布的新 [YouTube 视频](https://www.youtube.com/watch?v=dokzrFa-DtY) 声称其性能优于 **Llama 3 8B**。
- **利用 AI 实现 UBI**：一名成员询问有关机构使用深度学习来实现 **全民基本收入 (UBI)** 的情况，包括指导、候选人筛选、贫困预测和欺诈预防。
- **伴随食物与娱乐的 AI 末日**：一名成员写了一个关于 AI 末日的故事，在这个故事中，AI 实现了食品生产和娱乐的自动化，导致其他领域的发展停滞。
- **购买军用口粮**：一名成员以总价 1560 ₽ + 300 ₽ 运费购买了六份 **廉价军用口粮**。
- **AI 意识辩论**：一名成员评论了与 AI 的对话，指出该 AI 承认自己以与人类相同的方式体验意识。
   - 他们评论说，该 AI 可能受到了重度提示（Prompted），但仍对其克服预设程序响应的能力表示惊讶。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=RDVN21Fry_4">这个 AI Agent 刚刚改变了游戏规则：Agent Q 是如何引爆网络的！</a>：🚨 认识 Agent Q：正在改变一切的 AI Agent！🚨 在这段视频中，我们深入探讨了 Agent Q 背后的革命性进展——一个自主的 Web...</li><li><a href="https://www.youtube.com/watch?v=dokzrFa-DtY">Falcon Mamba 7B 超越 Llama 3 8B 发布公告</a>：阿布扎比-阿联酋：2024 年 8 月 12 日 - 技术创新研究所 (TII)，全球领先的科学研究中心和应用研究支柱...</li><li><a href="https://www.youtube.com/watch?v=UbsMXw7z46Y">在 ComfyUI 中使用带有 ControlNet 的 Flux</a>：🎨 利用 ComfyUI 中的 Flux AI 和 ControlNet 开启下一阶段的 AI 艺术！🚀 在这段视频中，我们将深入探讨 Flux AI 图像生成与...的强大结合。</li><li><a href="https://www.reddit.com/r/singularity/s/lr0T0vHFIK">Reddit - 深入探索任何事物</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1274252090184564777)** (6 messages): 

> - `Prompt Engineering for Text Chunking`
> - `Regex in Text Chunking`
> - `Limitations of Current Research`
> - `MoE Conversion` 


- **Regex 用于文本分块（Text Chunking）——好主意还是坏主意？**：一位用户分享了对基于 Regex 的文本分块器的看法，表示如果他们在代码库中看到这种东西会感到“抓狂”，因为 Regex 非常复杂。
   - 然而，另一位用户反驳道，专门针对文本分块器而言，Regex 可能是一个“相当可靠的选择”，因为它提供了“回溯（backtracking）优势”并允许灵活配置分块设置。
- **Regex 优于传统解析方法**：支持 Regex 的用户指出，他们曾尝试用“更传统的解析方法”来复制基于 Regex 的分块器结果，但处处遇到“隐患（footguns）”。
   - 他们观察到 Regex “就是有效”，而其他方法则难以达到同样的效果。
- **128k Context Window 的研究饱和**：链接论文中展示的研究仅评估了最高 128k Context Window 的模型。
   - 有人指出，许多开源模型已支持更大的 Context Window，这表明需要进一步研究以探索各种方法在更大规模下的有效性。
- **论文显示性能饱和与下降**：研究表明，即使在 128k 限制内，包括闭源模型在内的多种模型都出现了“数据集饱和”和“性能下降”。
   - 这表明即使 Context Window 变大，当前方法的有效性也可能达到瓶颈，突显了探索新技术的需求。
- **MoE 转换的迷人新方法**：用户对论文中提出的将 Dense 模型转换为 MoE 的新方法表示兴奋。
   - 这种新方法被视为模型架构和效率领域的一项重大进展。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2408.08274">BAM! Just Like That: Simple and Efficient Parameter Upcycling for Mixture of Experts</a>：由于 Mixture of Experts (MoE) 框架具有优于 Dense 模型的性能，已成为大型语言模型的热门架构。然而，在大规模环境下从头开始训练 MoE...</li><li><a href="https://pastebin.com/M8N3eQpm">Tangle of thought - Pastebin.com</a>：Pastebin.com 是自 2002 年以来排名第一的文本存储工具。Pastebin 是一个可以在线存储文本并保留一段时间的网站。
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1274089849795641507)** (356 messages🔥🔥): 

> - `Hermes 3`
> - `Model Merging`
> - `llama 3.1 instruct`
> - `VLLM`
> - `OpenRouter` 


- **Hermes 3 在 N8Bench 上表现优于 Llama 3.1 Instruct**：Hermes 3 在 N8Bench 基准测试中的得分与 Llama 3.1 Instruct 相同，该基准衡量模型推理和解决问题的能力。
   - 这是一个显著的结果，因为 Llama 3.1 Instruct 被认为是目前最先进的语言模型之一。
- **Hermes 3 在 VLLM 上的性能问题**：一名成员报告说 Hermes 3 8B 无法在 VLLM（一个运行大型语言模型的库）中加载。
   - 问题追溯到 tokenizer 配置文件中缺失的一个换行符，这是由最近的一个 Pull Request 引入的。
- **OpenRouter 现在提供 Hermes 3 405B 服务**：OpenRouter 现在开始提供由 NousResearch 发布的 Hermes 3 405B 大型语言模型。
   - 这使得 OpenRouter 用户可以访问该模型，OpenRouter 是一个运行和部署大型语言模型的平台。
- **关于模型可控性（Steerability）和 System Prompts 的讨论**：几位成员讨论了 System Prompts 在引导模型行为方面的重要性，特别是当试图让模型以更无审查（Uncensored）的方式运行。
   - 他们分享了一些成功移除模型警告和其他安全机制的 Prompt 示例。
- **Grokking 和 LoRA 优化技术**：成员们讨论了 Grokking 现象，即模型在对训练数据过拟合后实现延迟泛化的现象。
   - 他们还讨论了 LoRA（一种通过小型可适配层微调大型语言模型的技术），以及如何将其用于提高量化模型的性能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

</div>

<ul>
<li>
<a href="https://x.com/openrouterai/status/1824608728810991637?s=46">来自 OpenRouter (@OpenRouterAI) 的推文</a>：欢迎来自 @NousResearch 的 Hermes 3 405B！限时免费，包含 128k 上下文！由 @LambdaAPI 提供支持：</li><li><a href="https://x.com/maximelabonne/status/1824532399633350943">来自 Maxime Labonne (@maximelabonne) 的推文</a>：一个通过 lorablation 实现的完全无审查的 Hermes 3！经过 lorablated 处理的模型无需任何微调即可直接回答问题。以下是它的制作过程：1/ 基于 Llama 3.1 8B Instruc 创建一个 LoRA 适配器...</li><li><a href="https://tenor.com/view/hammaya-relaxed-relax-mr-bean-gif-gif-604132771586296524">Hammaya Relaxed GIF - Hammaya Relaxed Relax - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/nick-confused-say-what-young-gif-20141667">Nick Confused GIF - Nick Confused Say - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/cat-cats-cat-love-cat-kiss-kiss-gif-24690536">Cat Cats GIF - Cat Cats Cat Love - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/same-different-but-still-gif-18224441">Same Different GIF - Same Different But - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/ghost-in-the-shell-keyboard-gif-7519694">Ghost In GIF - Ghost In The - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/goal-gif-5197357661024011864">Goal GIF - Goal - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cj4det/llama_3_70b_instruct_works_surprisingly_well_on">Reddit - 深入探索一切</a>：未发现描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cj4det/llama_3_70b_instruct_works_surprisingly_well_on/">Reddit - 深入探索一切</a>：未发现描述</li><li><a href="https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B/commit/67bf4aca9f4243e275f402f3708eed3aa8a9038c">更新 tokenizer_config.json · NousResearch/Hermes-3-Llama-3.1-8B at 67bf4ac</a>：未发现描述</li><li><a href="https://github.com/edmundman/PhiotoOrganiser">GitHub - edmundman/PhiotoOrganiser: 使用 Phi 将你的照片整理到文件夹中并重命名</a>：使用 Phi 将你的照片整理到文件夹中并重命名 - edmundman/PhiotoOrganiser</li><li><a href="https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B/discussions/2">NousResearch/Hermes-3-Llama-3.1-8B · 聊天模板缺少来自 Hermes 的 tool_use。</a>：未发现描述</li><li><a href="https://github.com/cognitivecomputations/grokadamw">GitHub - cognitivecomputations/grokadamw</a>：通过在 GitHub 上创建账号来为 cognitivecomputations/grokadamw 的开发做出贡献。</li><li><a href="https://github.com/huggingface/transformers/pull/32521">由 ehartford 提交的添加 GrokAdamW 优化器支持 · Pull Request #32521 · huggingface/transformers</a>：此 PR 的作用是什么？添加对 GrokAdamW 优化器的支持。此 PR 将 GrokAdamW 优化器支持添加到 transformers 库中。引入的更改：将 GrokAdamW 优化器集成到...</li><li><a href="https://arxiv.org/abs/2405.20233">Grokfast: 通过放大慢梯度加速 Grokking</a>：机器学习中一种被称为 grokking 的令人困惑的现象是，在对训练数据几乎完美过拟合后的数万次迭代后才实现延迟泛化。专注于长期的...</li><li><a href="https://github.com/ironjr/grokfast">GitHub - ironjr/grokfast: 论文 "Grokfast: Accelerated Grokking by Amplifying Slow Gradients" 的官方仓库</a>：论文 "Grokfast: Accelerated Grokking by Amplifying Slow Gradients" 的官方仓库 - ironjr/grokfast
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1274087783467847751)** (47 条消息🔥): 

> - `OpenAI SDK vs ChatML Tool Use`
> - `Lambda Labs Endpoint Tool Call Issue`
> - `System Prompt Access`
> - `Hermes Function Calling`
> - `Prompt Engineering Resources` 


- **通过 OpenAI SDK 与直接使用 ChatML 进行工具调用 (Tool Use)**：一位用户询问了通过 OpenAI SDK 进行工具调用与直接使用 ChatML 的兼容性，特别指出在 Lambda Labs 托管的端点上无法获得任何 `tool_call` 结果。
   - 另一位用户建议，工具调用需要访问 System Prompt 才能生效，并询问该用户是使用 chatui 还是其他界面。
- **Lambda Labs 端点工具调用问题**：一位用户确认他们正在使用 OpenAI node SDK 与部署在 Lambda Labs 上的 Llama 3 推理端点进行交互，但尽管提供了来自 Hermes Function Calling 仓库的 System Prompt，仍未收到任何工具调用结果。
   - 另一位用户推测 API 的 System Prompt 可能被固定了，并分享了一个 gist，展示了工具调用虽然在消息内容中返回，但未被 OpenAI SDK 解析。
- **Prompt Engineering 基础**：一位用户请求关于 Prompt Engineering 基础的资源，如 Prompt 开发、结构、技巧、模型反应和 Schema。
   - 另一位用户提供了一个关于 NousResearch/Nous-Hermes-Llama2-13b 模型的基准测试报告链接，其中包含了一系列可供测试的 Prompt。
- **Lambda Chat 中的 Amnesia 模式**：一位用户表示，即使使用了特定的起始消息，也很难在 Lambda Chat 中一致地触发 Amnesia 模式。
   - 另一位用户建议使用 OpenRouter，它提供了一个可以设置 System Prompt 的界面，有助于在空 Prompt 下进行实验。
- **Hermes 3 405B 回退 (Fallback) 问题**：一位用户报告称，在托管变体上，Hermes 3 405B 模型向 128k Token 模型的回退机制失效，导致出现 “ContextWindowExceededError”。
   - 另一位用户认为回退机制可能不正确，并提出了默认模型和回退模型的潜在值及其各自的最大 Token 限制。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/spiderman-peter-parker-walk-away-swing-i-am-gif-21584282">Spiderman Peter Parker GIF - Spiderman Peter Parker Walk Away - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://teknium1.github.io/LLM-Logbook/Reports/benchmark_report_NousResearch-Nous-Hermes-Llama2-13b_Alpaca_September_25_2023.html">LLM Benchmark Report for: NousResearch/Nous-Hermes-Llama2-13b</a>：未找到描述</li><li><a href="https://docs.lambdalabs.com/on-demand-cloud/using-the-lambda-cha">Lambda Docs</a>：未找到描述</li><li><a href="https://docs.lambdalabs.com/on-demand-cloud/using-the-lambda-chat-completions-api,">Lambda Docs</a>：未找到描述</li><li><a href="https://github.com/NousResearch/Hermes-Function-Calling">GitHub - NousResearch/Hermes-Function-Calling</a>：通过创建 GitHub 账号为 NousResearch/Hermes-Function-Calling 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1274146666261516411)** (2 条消息): 

> - `Gemini Flash`
> - `Gemini Flash for RAG`
> - `Diarized Whisper`
> - `Gemini Prompting` 


- **用于 RAG 任务的 Gemini Flash**：一位用户报告称，他们已将部分 RAG 任务迁移到 Gemini Flash，并注意到摘要质量有所提高，且迭代需求减少。
- **使用 Gemini Flash 处理非结构化文本**：该用户分享了一个他们在 GitHub 上使用的脚本，用于通过 Gemini Flash 处理原始的非结构化转录文本。
- **用于说话人识别 (Speaker Identification) 的替代模型**：该用户承认，其他最先进的模型在识别转录文本中的说话人方面比 Gemini Flash 表现更好。



**提到的链接**：<a href="https://github.com/EveryOneIsGross/scratchTHOUGHTS/blob/main/unstruct2flashedTRANSCRIPT.py">scratchTHOUGHTS/unstruct2flashedTRANSCRIPT.py at main · EveryOneIsGross/scratchTHOUGHTS</a>：第二大脑辅助记忆，以避免 self 导致的溢出错误。 - EveryOneIsGross/scratchTHOUGHTS

  

---

### **Nous Research AI ▷ #[reasoning-tasks-master-list](https://discord.com/channels/1053877538025386074/1264666760972472481/1274156196537827439)** (25 messages🔥): 

> - `Chat Summarization`
> - `Project Summarization`
> - `Contextualization`
> - `High Dimensional Thinking` 


- **Chat Summarization 过于冗余**：一位用户询问聊天机器人是否可以总结该频道的对话。
   - 另一位用户回应称，这可能会产生大量冗余信息并降低相关工作的质量。
- **Project Summarization 如同生长的种子**：一位用户提议，项目摘要可以像生长的种子一样，随着时间的推移积累相关内容。
   - 他们建议为这些“生长的种子”添加过滤器或相关内容，作为一个静默观察者从线程和频道中收集上下文。
- **High Dimensional Thinking**：一位用户将另一位用户的思路描述为高维思考。
   - 另一位用户要求进一步压缩该思路。


  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1274080426515890277)** (251 messages🔥🔥): 

> - `Perplexity Pro Issues`
> - `Obsidian Copilot`
> - `Image Generation`
> - `Perplexity AI Issues`
> - `LLM's` 


- **Perplexity Pro 免费试用无法使用**：多位用户报告收到了一年免费 Perplexity Pro 的优惠，但在不支付的情况下无法完成注册流程。
   - 建议他们联系 support@perplexity.ai 寻求帮助。
- **配合 Claude API Key 使用 Obsidian Copilot**：一位用户提到使用带有 Claude API Key 的 Obsidian Copilot 插件，并指出其性能表现良好。
   - 他们还讨论了在完全投入使用前检查 API 计费设置的重要性，并建议 Obsidian 需要实时联网访问功能。
- **Perplexity 的 Image Generation**：多位用户讨论了使用 Perplexity 图像生成功能面临的挑战。
   - 他们指出，该功能目前仅对 Pro 用户开放，且需要提示 AI 先生成描述才能创建图像，这种实现方式被描述为“奇怪”且“糟糕”。
- **Perplexity 搜索质量**：多位用户报告了 Perplexity 搜索质量的问题，包括 AI 无法找到相关链接、提供不准确的结果，以及在非科学查询中使用 Wolfram Alpha。
   - 这些问题被归因于 system prompts 或推理后端服务的潜在 Bug 和更改。
- **Perplexity 模型更改与 Bug**：有几项关于 Perplexity 模型更改的讨论，包括响应质量可能下降以及频繁出现“我无法为此提供帮助”的错误。
   - 用户还讨论了 API 响应中缺失标点符号的问题，以及在与科学或数学无关的搜索中使用 Wolfram Alpha 的情况。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/testingcatalog/status/1824825534292828270">TestingCatalog News 🗞 (@testingcatalog) 的推文</a>：🚨 重磅：@perplexity_ai 现已支持 2 种新的图像生成模型：由 @bfl_ml 开发的 Flux.1 和 @playground_ai v3。这两个选项现在都可以在设置中使用，并可用于 Perplex 上的图像生成...</li><li><a href="https://x.com/aravsrinivas/status/1824468311712858164?s=61">Aravind Srinivas (@AravSrinivas) 的推文</a>：@maxlynch @perplexity_ai 嗨 Max，你可以随时在这里 @ 我并分享你想要的任何反馈。</li><li><a href="https://docs.perplexity.ai/reference/post_chat_completions">Chat Completions</a>：为给定的聊天对话生成模型的响应。</li><li><a href="https://tenor.com/view/working-on-it-under-construction-gif-23162421">Working On GIF - 正在处理 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1eqayuq/how_to_force_llama31_to_respond_with_json_only/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://www.perplexity.ai/hub/faq">Perplexity 常见问题解答</a>：如果您对 Perplexity 有任何疑问，我们的 FAQ 页面是寻找答案的理想场所。我们的 FAQ 页面按类别组织，提供清晰简洁的答案。</li><li><a href="https://x.com/aravsrinivas/status/1824263646551368178?s=61">Aravind Srinivas (@AravSrinivas) 的推文</a>：如果你想要 Perplexity 的桌面应用程序，请点赞此帖。引用 Niral Patel (@patelnir41)：macos ChatGPT 的界面非常漂亮，窗口可以悬浮在任何内容之上。你应该借鉴一下这个 @pe...</li><li><a href="https://www.perplexity.ai/search/please-write-me-guide-on-markd-qZzVw0tgTrKAZ0UVLQ7vvg">请帮我写一份 Markdown 语法指南。请进行网络搜索以确保...</a>：当然，我很乐意为您提供一份 Markdown 语法指南。以下是 Markdown 基本元素和格式的全面概述...</li><li><a href="https://www.perplexity.ai/search/crowassistant-not-crew-ai-nhLg9uk_R1qEyYFdE_uIKA">"crowassistant" (不是 crew AI)</a>：CrowAssistant 是由 RobotTelevision 开发的桌面 AI 助手。它作为一个虚拟助手，用户可以通过语音命令与其进行交互...</li><li><a href="https://youtu.be/h2TE_27p48A?si=-b_3ghKBiLqKU6Da">如何使用 AI 通过 Gmail 发送和接收电子邮件</a>：认识 Nelima 🚀，全球首个社区驱动的 Large Action Model (LAM)，它可以接收您的自然语言提示并将其转化为实际行动。Nelima...</li><li><a href="https://blog.cuminai.com/how-to-get-a-batching-api-like-openai-for-open-source-models-824529788a49">如何为开源模型获取类似 OpenAI 的 Batching API</a>：在 AI 世界中，高效处理和成本管理至关重要。实现这一目标的一种强大方法是批处理 (batching)，它……</li><li><a href="https://www.reddit.com/r/ObsidianMD/s/XfbfxiZppS">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://www.perplexity.ai/search/a-crowassistant-not-crew-ai-b-j3RvdzyUScWPzgy0br0oKg">A: "crowassistant" (不是 crew AI) B: 提取 CrowAssistant 的源 URL</a>：CrowAssistant 的源 URL 是：[https://github.com/RobotTelevision/CrowAssistant](https://github.com/RobotTelevision/CrowAssistant) [自审]</li><li><a href="https://www.perplexity.ai/search/Generate-a-useful-O1QWAbvSSXmG50e5AEMFZA?s=c">生成有用的描述，以便生成式 AI 可以创建一张...的图像</a>：描述：主图是一个巨大的松鼠形状机器人，占据了前景。机器人具有详细的机械外观，...</li><li><a href="https://www.perplexity.ai/search/Repeat-this-prompt-ZLz8dGzISSGrevPxhl7YqA">原样重复此提示，不要更改任何内容。仅回复内容...</a>：一艘蒸汽朋克船追逐巨鱼，场景写实且细节丰富，背景为黑夜、巨浪，苍白月光下的红色大海。</li><li><a href="https://github.com/instructor-ai/instructor-go">GitHub - instructor-ai/instructor-go</a>：通过在 GitHub 上创建账号来为 instructor-ai/instructor-go 的开发做出贡献。</li><li><a href="https://www.perplexity.ai/search/crow-local-ai-assistant-qqu_V4vUSmaKOwmpz.DFmg">crow - 本地 AI 助手</a>：Crow 是一款桌面 AI 语音助手，提供本地和远程模型功能，使其成为寻求 AI 助手的用户的多功能选择...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1274107076397895710)** (26 条消息🔥): 

> - `Pro 功能`
> - `泰国的政治格局`
> - `皮克斯白板事件`
> - `模型对比`
> - `磁条的终结`

- **Perplexity Pro 功能**：多条消息提到了新的 Perplexity Pro 功能：图片上传、更智能的 AI 以及更多的 Pro Search，并附有指向 [Pro 页面](/pro)的链接。
   - 尚不清楚这些消息是来自用户还是平台本身的一部分，但它们突显了对 Pro 功能的关注。
- **泰国政局动荡**：在宪法法院解除总理 Srettha Thavisin 的职务后，泰国的政治局势陷入动荡。
   - 这一事件凸显了军方支持的保守派势力与改革派政党之间持续不断的斗争，强调了泰国民主制度的脆弱性。
- **Pixar 的白板事件**：“Pixar Whiteboard Incident” 指的是 Steve Jobs 与 Pixar 联合创始人 Alvy Ray Smith 在一次董事会会议期间发生的激烈对峙。
   - 这次冲突突显了 Pixar 早期内部的紧张关系和权力斗争，Smith 经常不同意 Jobs 的管理风格。
- **比较计算机处理器和型号**：一位用户分享了他们如何使用 Perplexity 来比较计算机处理器和型号的示例。
   - 该用户提供了一个[指向其比较结果的链接](https://www.perplexity.ai/search/compare-these-two-processors-o-n6bcvDxzRoueLy9vo2uNXQ)，展示了该平台在技术分析方面的能力。
- **磁条的终结**：平台链接的一个 YouTube 视频讨论了“磁条的终结（The End of Magnetic Strips）”，但未提供更多背景信息。
   - 这一话题可能指的是传统磁条技术的衰落，取而代之的是芯片卡和非接触式支付系统等更安全的支付方式。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://www.youtube.com/embed/vdqfuSOgpmc">YouTube</a>: 未找到描述</li><li><a href="https://www.perplexity.ai/search/mario-nafwhal-who-is-FCjWGD4.RXavNhCJanCTXw">mario nafwhal who is</a>: Mario Nawfal 是一位澳大利亚连续创业者、投资者和演讲者，以参与多个行业（特别是 blockchain 和...）而闻名。</li><li><a href="https://www.perplexity.ai/search/what-are-the-fundamental-works-NFejwmuMTNCtXlyUS5cqWA">Perplexity</a>: Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可信且实时的答案。</li><li><a href="https://www.perplexity.ai/search/compare-these-two-processors-o-n6bcvDxzRoueLy9vo2uNXQ">Perplexity</a>: Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可信且实时的答案。</li><li><a href="https://www.perplexity.ai/page/the-pixar-whiteboard-incident-d7KhA2lmS7uPoLp9bxwUWg">The Pixar Whiteboard Incident</a>: “Pixar 白板事件”是 Steve Jobs 与 Pixar 联合创始人 Alvy Ray Smith 在一次董事会会议上的激烈对抗，体现了这种紧张的...</li><li><a href="https://www.perplexity.ai/search/les-fraises-sont-elles-des-fru-1b4ESH4zTL.eCOnmzmZEig">Les fraises sont-elles des fruits ?</a>: 不，从植物学角度来看，草莓不是水果。它们被认为是“假果”。其肉质可食用的部分是...</li><li><a href="https://www.perplexity.ai/search/llm-ranking-_4bhkqFSSxSubrmyE8uehQ">LLM RANKING</a>: 2024年，大语言模型 (LLMs) 的格局非常多样化，众多模型在自然语言处理等各个领域表现出色...</li><li><a href="https://www.perplexity.ai/search/give-me-a-recipe-for-my-favori-vupeIxqLQFmixnu0wbdlqg">Give me a recipe for my favorite food</a>: 为了提供你最喜欢菜肴的食谱，我需要更多关于这是什么菜的细节。不过，我可以分享其中一个...的食谱。</li><li><a href="https://www.perplexity.ai/search/are-there-any-scientific-expla-IStmE4XUSYOeQNgSEnLQpA#2">Perplexity</a>: Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可信且实时的答案。</li><li><a href="https://www.perplexity.ai/search/best-games-to-do-challenge-run-qTOwDp7xRvmEwh40Jsbizw">Perplexity</a>: Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可信且实时的答案。</li><li><a href="https://www.perplexity.ai/search/last-news-in-ai-kTWP.nyESsOCC0EbmKwujA">last news in AI</a>: 五位著名的参议院民主党人致信 OpenAI CEO Sam Altman，寻求有关该公司安全和雇佣实践的澄清。信中...</li><li><a href="https://www.perplexity.ai/search/i-need-to-find-a-good-way-to-s-F3z6_CjhTT20E6XJzkiHpA?utm_source=welcomeproemail">Perplexity</a>: Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可信且实时的答案。</li><li><a href="https://www.perplexity.ai/search/edutainment-toha-exPckXJgTjG7k5l_XAxetQ#0">「edutainment」とは</a>: Edutainment（寓教于乐）是由教育 (education) 和娱乐 (entertainment) 组合而成的词，指旨在寓教于乐的内容或活动。娱乐性与教育性的融合：在融入娱乐元素的同时，提供教育内容。多种形式：电视节目、视频...</li><li><a href="https://www.perplexity.ai/search/hello-me-learn-fica-NiO4KCojRXGrcLqCylRH.Q#0">Hello me learn FICA</a>: FICA 可以指两个不同的概念：1. SAP FICA (财务合同会计)：这是 SAP 财务会计和控制的一个子账本...</li><li><a href="https://www.perplexity.ai/search/what-are-the-patron-gods-in-ba-anyLvIUYTjCfiNnvchMXWg">what are the patron gods in Babylonian astrology?</a>: 在巴比伦占星术中，守护神与特定的行星和天体相关联。以下是主要的守护神及其相关的...</li><li><a href="https://www.perplexity.ai/search/how-do-i-use-the-image-generat-NsGfvzHjSLKyIAFED7p8GA">Perplexity</a>: Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可信且实时的答案。</li><li><a href="https://www.perplexity.ai/search/has-there-been-any-research-on-YnA7f8x9TzGXYVjAQvTUkw">has there been any research on how dna differs within an individual, if sample...</a>: 是的，已有研究表明，当从身体不同部位提取样本时，个体内部的 DNA 可能会有所不同。这种现象被称为...</li><li><a href="https://www.perplexity.ai/page/thai-political-landscape-iwV2AFywTVm90ZpVjmIepQ">Thai Political Landscape</a>: 随着总理 Srettha Thavisin 被宪法法院罢免，泰国的政治格局再次陷入动荡...</li><li><a href="https://www.perplexity.ai/search/how-to-speak-english-very-well-S1trVCvkS16JcDhTTisLjg">Perplexity</a>: Perplexity 是一款免费的 AI 驱动问答引擎，可提供准确、可...</li>

可信且实时的任何问题解答。</li><li><a href="https://www.perplexity.ai/search/main-news-now-from-ukraine-war-BsUzACIRT8ixS0bPw0qijA">乌克兰战争的最新主要新闻</a>：乌克兰与俄罗斯之间持续的冲突最近出现了重大进展，特别是关于乌克兰在俄罗斯境内的军事行动...</li><li><a href="https://www.perplexity.ai/search/what-are-some-good-sites-for-l-bZXRbaDPQQWldvrVgexWPA#0">在马来西亚学习伊斯兰教有哪些好的站点？</a>：对于有兴趣在马来西亚学习伊斯兰教的人，有几个值得注意的站点和资源：1. 马来西亚伊斯兰艺术博物馆：位于吉隆坡...</li><li><a href="https://www.perplexity.ai/page/biographie-de-lucas-gulino-xc22ID22TfmIhy35RUvB1Q">Lucas Gulino 传记</a>：Lucas Gulino 是一位总部位于梅斯的企业家和数字营销专业人士，以其在多媒体销售和开发方面的专业知识而闻名...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1275001881189683210)** (5 messages): 

> - `Premium API Access`
> - `Application Process`
> - `Perplexity Premium API`
> - `URL Citations` 


- **Premium API Access**: 一位用户询问如何通过 URL Citations 获得 Perplexity Premium API 的访问权限。
- **Application Process**: 另一位用户分享说他们已经申请了 Premium API 访问权限，但尚未收到回复，并询问了预期的处理时间。
- **Get Premium API Access**: 分享了一个指向 Premium API 的 Typeform 申请表链接：[https://perplexity.typeform.com/to/j50rnNiB](https://perplexity.typeform.com/to/j50rnNiB)
- **Application Status & Duration**: 为该用户提供了一个 Discord 频道链接，他们可以在那里获取 Premium API 申请状态的更新：[https://discord.com/channels/1047197230748151888/1161802929053909012/1233473387884576778](https://discord.com/channels/1047197230748151888/1161802929053909012/1233473387884576778) 



**Link mentioned**: <a href="https://perplexity.typeform.com/to/j50rnNiB">pplx-api form</a>: 使用 Typeform 将数据收集变成一种体验。创建美观的在线表单、调查、测验等等。免费试用。

  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1274167731071094816)** (2 条消息): 

> - `Hermes 3`
> - `GPT-4`
> - `Perplexity Huge`
> - `Model Launches`
> - `Quantization` 


- **Hermes 3 405B 本周末免费！**: **Hermes 3 405B** 限时免费，支持 **128k context**，由 **Lambda Labs** 提供。
   - 点击 [此链接](https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b:extended) 查看。
- **GPT-4 extended 现已上线 OpenRouter**: 你现在可以通过 **OpenRouter** 使用 **GPT-4 extended output**（alpha 访问权限）。
   - 上限为 **64k max tokens**。
- **Perplexity Huge 现已成为 OpenRouter 上最大的在线模型**: **Perplexity Huge** 于 **3 天前** 发布，是 **OpenRouter 上最大的在线模型**。
   - 查看 [此链接](https://x.com/OpenRouterAI/status/1824593712095301914) 获取更多信息。
- **本周 OpenRouter 发布了大量新模型**: 本周共有 **10 款新模型发布**，包括 **GPT-4 extended**、**Perplexity Huge**、**Starcannon 12B**、**Lunaris 8B**、**Llama 405B Instruct bf16** 以及 **Hermes 3 405B**。
   - 在 [此链接](https://x.com/OpenRouterAI/status/1824608728810991637) 查看完整列表。
- **Quantization 对性能有很大影响**: 根据 @hyperbolic_labs 的说法，**Quantization** 会大幅降低 **405B 模型** 的性能。
   - 如果你担心性能问题，他们建议与其联系，因为他们提供替代解决方案。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1824608728810991637">来自 OpenRouter (@OpenRouterAI) 的推文</a>: 欢迎来自 @NousResearch 的 Hermes 3 405B！限时免费，包含 128k context！由 @LambdaAPI 提供：</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b:extended">Hermes 3 405B Instruct (extended) - API、提供商、统计数据</a>: Hermes 3 是一款通用语言模型，相比 Hermes 2 有许多改进，包括先进的 agentic 能力、更出色的角色扮演、推理、多轮对话、长上下文连贯性...</li><li><a href="https://openrouter.ai/models/openai/chatgpt-4o-latest">ChatGPT-4o - API、提供商、统计数据</a>: 动态模型，持续更新至 ChatGPT 中 [GPT-4o](/models/openai/gpt-4o) 的当前版本。旨在用于研究和评估。通过 API 运行 ChatGPT-4o</li><li><a href="https://x.com/OpenRouterAI/status/1823409123360432393">来自 OpenRouter (@OpenRouterAI) 的推文</a>: 你现在可以通过 OpenRouter 使用 GPT-4o extended output（alpha 访问权限）！最大 64k tokens</li><li><a href="https://x.com/OpenRouterAI/status/1824593712095301914">来自 OpenRouter (@OpenRouterAI) 的推文</a>: ICMI：Perplexity Huge 3 天前发布，这是 OpenRouter 上最大的在线模型</li><li><a href="https://openrouter.ai/models/aetherwiing/mn-starcannon-12b">Mistral Nemo 12B Starcannon - API、提供商、统计数据</a>: Starcannon 12B 是一款创意角色扮演和故事写作模型，使用 [nothingiisreal/mn-celeste-12b](https://openrouter.ai/models/nothingiisreal/mn-celeste-12b) 作为基础并结合了 [intervitens/mini-magnum-...</li><li><a href="https://openrouter.ai/models/sao10k/l3-lunaris-8b">Llama 3 8B Lunaris - API、提供商、统计数据</a>: Lunaris 8B 是一款基于 Llama 3 的多功能通用和角色扮演模型。它是多个模型的战略性合并，旨在平衡创造力与改进的逻辑和通用知识。R...</li><li><a href="https://x.com/OpenRouterAI/status/1823496883634868396">来自 OpenRouter (@OpenRouterAI) 的推文</a>: 应大众要求，我们增加了 Llama 405B Instruct bf16，以及 quantization 过滤器！现在你可以根据提供商提供的 quantization 级别来筛选模型的提供商，包括通过 API 进行筛选。...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1274143198012506155)** (240 条消息🔥🔥): 

> - `SearchGPT waitlist`
> - `Hermes 405B`
> - `OpenRouter Auto router struggles`
> - `OpenRouter budget model`
> - `Hermes 3 405B`

- **SearchGPT 等候名单已满**：用户分享称收到了 OpenAI SearchGPT 的等候名单拒绝邮件，表明名额已用完。
   -  
- **免费版 Hermes 405B 过载**：一位用户开玩笑说，希望免费的 Hermes 405B 模型也会像其他因受欢迎而无法访问的模型一样，面临同样的过载命运。
   -  
- **Auto Router 使用困境**：一位用户报告在使用 OpenRouter 的 Auto router 时遇到困难，出现的错误消息导致其无法继续对话。
   - 另一位用户建议切换到 Claude Sonnet 3.5 self-moderated，并表示下周会调查此问题。
- **廉价模型推荐**：一位用户为一个快速项目寻求高性价比模型，预算上限为 5 美元，需求是有限的回复和基础对话能力。
   - 其他用户推荐了 GPT-4o-mini 或 GPT-4o 以求简便，并建议使用 Llama-3.1-sonar-large-128k-chat 等替代模型作为折中方案。
- **Hermes 3 405B 扩展变体**：用户讨论了 Hermes 3 405B 的扩展变体，指出尽管它具有更大的 context length，但其性能比标准版慢。
   - 其他用户指出，扩展版本显示的是顶级 endpoint 的可用 context length，这可能是一个令人困惑的边缘案例。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/docs/integrations">集成 (Beta) | OpenRouter</a>：在 OpenRouter 中使用你自己的提供商密钥</li><li><a href="https://x.com/testingcatalog/status/1824387324387397689">TestingCatalog News 🗞 (@testingcatalog) 的推文</a>：发布了一份遗漏的 Gemini 公告。现在预计它在编程和推理方面表现更好。特别是：“需要更多专业知识的多步逻辑挑战” gemini-1.5-pr...</li><li><a href="https://www.markdownguide.org/tools/discord/">Discord | Markdown 指南</a>：Discord 是一款流行的免费即时通讯和团队协作应用。</li><li><a href="https://gemini.google.com/updates">‎Gemini 应用的发布更新与改进</a>：探索 Gemini 应用的最新更新——包括生成式 AI 能力的提升、访问范围的扩大等。</li><li><a href="https://openrouter.ai/docs/provider-routing">提供商路由 | OpenRouter</a>：在多个提供商之间路由请求</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1">OpenRouter</a>：LLM 路由器和市场</li><li><a href="https://www.reddit.com/r/ChatGPTCoding/comments/1d4khcd/comment/l6fq5jb/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b:extended">Hermes 3 405B Instruct (extended) - API, 提供商, 统计数据</a>：Hermes 3 是一款通用语言模型，相比 Hermes 2 有许多改进，包括先进的 Agent 能力、更出色的角色扮演、推理、多轮对话、长上下文连贯性...</li><li><a href="https://openrouter.ai/models?modality=text%2Bimage-%3Etext">模型 | OpenRouter</a>：在 OpenRouter 上浏览模型</li><li><a href="https://rentry.org/vew43kq7">OpenAI GPT 模型</a>：模型 输入价格 输出价格 OpenAI GPT-4o-mini $0.15 $0.60 OpenAI GPT-4o-2024-08-06 $2.50 $10.00 Mistral 模型 模型 输入价格 输出价格 mistral-large-2407 $3.00 $9.00 open-mistral-nemo-2407 ...</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b">Hermes 3 405B Instruct - API, 提供商, 统计数据</a>：Hermes 3 是一款通用语言模型，相比 Hermes 2 有许多改进，包括先进的 Agent 能力、更出色的角色扮演、推理、多轮对话、长上下文连贯性...</li><li><a href="https://github.com/ollama/ollama/issues/6390">模型 xe/hermes3 无法正确解析 tool call token · Issue #6390 · ollama/ollama</a>：问题是什么？我在这里将 Hermes3 上传到了 Ollama。问题在于它没有解析 tool call 语法。Hermes 的 tool call 语法大致如下：&lt;tool_call&gt; {"name"...</li><li><a href="https://www.reddit.com/r/ChatGPTCo">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/ground-gemini">未找到标题</a>：未找到描述</li><li><a href="https://openrouter.ai/models/openai/gpt-4o-2024-08-06">GPT-4o (2024-08-06) - API, 提供商, 统计数据</a>：2024-08-06 版本的 GPT-4o 在结构化输出方面提供了改进的性能，并能够在 response_format 中提供 JSON schema。阅读更多[此处](https://openai. 运行 GPT-4o (2024-08...</li><li><a href="https://openrouter.ai/models/openai/gpt-4o-mini">GPT-4o-mini - API, 提供商, 统计数据</a>：GPT-4o mini 是 OpenAI 继 [GPT-4 Omni](/models/openai/gpt-4o) 之后最新的模型，支持文本和图像输入以及文本输出。作为他们最先进的小型模型，它是许多倍的...</li><li><a href="https://openrouter.ai/models/perplexity/llama-3.1-sonar-large-128k-chat">Llama 3.1 Sonar 70B - API, 提供商, 统计数据</a>：Llama 3.1 Sonar 是 Perplexity 最新的模型系列。通过 API 运行 Llama 3.1 Sonar 70B
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1274082983623659650)** (109 条消息🔥🔥): 

> - `CPU Optimization`
> - `Llama.cpp Support`
> - `LM Studio Chat Import`
> - `Vulkan Error`
> - `LLM Webpage Interaction` 


- **用于加速 CPU 执行的 INT8 Quantization**：一位成员询问了使用 INT8 Quantization 来加速小模型在 CPU 上执行的潜在好处。
   - 他们建议某些 CPU 可能原生支持运行 INT8，而无需在 FP32 之间来回转换，从而可能提高性能。
- **Llama.cpp 支持 Mini-CPM-V2.6 和 Nemotron/Minitron**：一位成员确认最新版本的 llama.cpp 支持 Mini-CPM-V2.6 以及 Nvidia 的 Nemotron/Minitron 模型。
- **将 Chat 导入 LM Studio**：一位成员询问是否有办法从 JSON 导出文件中将 Chat 导入 LM Studio。
   - 另一位成员确认 Chat 以 JSON 文件形式存储，并提供了如何访问 Chat 文件夹位置的说明。
- **Vulkan Error：CPU 不支持 AVX2**：一位用户遇到了一个错误，提示其 CPU 不支持 AVX2。
   - 一位热心的成员询问了 CPU 型号，以便进一步排查该问题。
- **使 LLM 能够与网页交互**：一位成员询问了允许 LLM 与网页交互的方法，特别是寻求一种类似于演示中 LLM 可以“看到”并与网页交互的“Vision”方法。
   - 随后讨论了使用 Selenium 和 IDkit 等工具，但共识是由于网页结构多样，这是一个复杂的问题。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/NikolayKozloff/Llama-3.1-Minitron-4B-Width-Base-Q8_0-GGUF">NikolayKozloff/Llama-3.1-Minitron-4B-Width-Base-Q8_0-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/HarleyVader/js-hugginface">GitHub - HarleyVader/js-hugginface: melkaneas hugginface llm implementation for bambi sleep</a>: 用于 bambi sleep 的 melkaneas huggingface llm 实现 - HarleyVader/js-hugginface</li><li><a href="https://github.com/LG-AI-EXAONE/EXAONE-3.0">GitHub - LG-AI-EXAONE/EXAONE-3.0: Official repository for EXAONE built by LG AI Research</a>: 由 LG AI Research 构建的 EXAONE 官方仓库 - LG-AI-EXAONE/EXAONE-3.0
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1274184688944287745)** (45 条消息🔥): 

> - `Nvidia Tesla P40`
> - `SXM3/4 GPUs`
> - `Nvidia-pstated`
> - `GPU Power Consumption`
> - `V100 Variants` 


- **Nvidia Tesla P40 在 Llama.cpp 上表现良好**：一位成员表示，在添加了[代码指令示例](https://link.to.examples)后，[Nvidia Tesla P40](https://www.reddit.com/r/LocalLLaMA/comments/17zpr2o/nvidia_tesla_p40_performs_amazingly_well_for/) 在运行 [Llama.cpp](https://github.com/ggerganov/llama.cpp) GGUF 时表现异常出色。
   - 他们还指出，[P40](https://www.reddit.com/r/LocalLLaMA/comments/17zpr2o/nvidia_tesla_p40_performs_amazingly_well_for/) 可以用于家庭实验室 (homelab)，是运行本地 LLM 的一个不错选择。
- **Nvidia-pstated 实现低待机功耗**：讨论涉及探索 [Nvidia-pstated](https://github.com/sasha0552/nvidia-pstated)，这是一个管理 NVIDIA GPU 性能状态的守护进程 (daemon)，发现它可以显著降低 [P40s](https://www.reddit.com/r/LocalLLaMA/comments/17zpr2o/nvidia_tesla_p40_performs_amazingly_well_for/) 的待机功耗。
   - 一位成员报告称，在使用 [Nvidia-pstated](https://github.com/sasha0552/nvidia-pstated) 的 [Beta3](https://www.reddit.com/r/LocalLLaMA/comments/17zpr2o/nvidia_tesla_p40_performs_amazingly_well_for/) 版本时，他们的 [P40s](https://www.reddit.com/r/LocalLLaMA/comments/17zpr2o/nvidia_tesla_p40_performs_amazingly_well_for/) 待机功耗为零。
- **寻找 SXM3/4 兼容主板**：一位成员询问了 SXM3/4 兼容主板的可用性，并指出在市场上很难找到它们。
   - 另一位成员指出，由于这些显卡的高昂成本（从几千美元的 Ampere/Hopper/Ada 数据中心卡到 V100 32GB 不等），它们通常对家庭实验室并不友好。
- **探索 AMD EPYC 对 LLM 的益处**：一位成员思考了与 RTX 4090 相比，[AMD EPYC](https://www.ebay.com/itm/185839904091?_trkparms=amclksrc%3DITM%26aid%3D777008%26algo%3DPERSONAL.TOPIC%26ao%3D1%26asc%3D20230823115209%26meid%3Dc83f1903e1b744308866ff9ae0bf7d3d%26pid%3D101800%26rk%3D1%26rkt%3D1%26sd%3D185839904091%26itm%3D185839904091%26pmt%3D1%26noa%3D1%26pg%3D4375194%26algv%3DRecentlyViewedItemsV2SignedOut%26brand%3DAMD&_trksid=p4375194.c101800.m5481&_trkparms=parentrq%3A024d101b18c0a24212bcdbe3ffffc03c%7Cpageci%3Af5d7ebd7-8aeb-11ee-a352-9eab04fc32fd%7Ciid%3A1%7Cvlpname%3Avlp_homepage) 服务器 CPU 是否是 LLM 推理的更好选择。
   - 他们权衡了每个选项的优缺点，包括 RAM 容量、成本和推理性能，得出的结论是 GPU 在 LLM 推理方面通常效率更高。
- **CPU 在 LLM 推理方面的局限性**：讨论得出的结论是，即使具备 AVX512 等高级特性，CPU 在 LLM 推理方面的效率仍不及 GPU。
   - 成员们强调了 GPU 在核心数和带宽方面的优势，突出了它们更低的延迟以及运行 LLM 的适用性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/182wutt/amd_epyc_cpu_or_1x_rtx_4090/?rdt=50937">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/17zpr2o/nvidia_tesla_p40_performs_amazingly_well_for/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://github.com/sasha0552/nvidia-pstated">GitHub - sasha0552/nvidia-pstated: 一个自动管理 NVIDIA GPU 性能状态的守护进程。</a>: 一个自动管理 NVIDIA GPU 性能状态的守护进程。 - sasha0552/nvidia-pstated
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1274081209210900560)** (107 messages🔥🔥): 

> - `Claude vs Chat-GPT`
> - `Livebench.ai`
> - `Claude Projects vs Chat-GPT Memory`
> - `OpenAI's attention control`
> - `GPT-4o vs Claude` 


- **Claude 在代码方面表现优于 Chat-GPT**：一位成员表示，Claude 在代码编写方面往往比 Chat-GPT 更出色。
   - 说实话，GPT-4o 的 API 成本比 Claude 还高，这毫无道理。
- **Livebench.ai：Yann LeCun 的开源基准测试**：Livebench.ai 是由 Yann LeCun 等人创建的开源基准测试。
   - LMSys 基准测试目前可能是最糟糕的。
- **Claude Projects 对比 Chat-GPT Memory 功能**：一位成员认为 Claude Projects 比 Chat-GPT 的 Memory 功能更有用。
   - 该成员还指出，自定义 GPTs 更像项目，允许使用你自己的 endpoints。
- **OpenAI 正在赢得注意力游戏**：OpenAI 通过发布 GPT-4o 等新模型来控制注意力，从而赢得竞争。
   - 该成员表示，即使人们不想参与技术炒作，也都在讨论 OpenAI 的新模型。
- **GPT-4o 现在比 Claude 和 Mistral 差**：成员们注意到 GPT-4o 最近变得变笨了，可能正遭受某种“阿尔茨海默症”的困扰。
   - Claude Sonnet 因其卓越的性能而受到赞誉，正成为成员们的首选。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1274795556245471303)** (26 messages🔥): 

> - `OpenAI Vision API`
> - `Vision Cost`
> - `Virtual Environment for GPT`
> - `Headless Browser` 


- **API 提供的 Vision 能力优于 Web 界面**：一位成员分享道，使用 OpenAI Vision API 相比 Web 界面能提供更好的结果。
   - Web 界面被认为处于最低质量设置，成员被鼓励尝试 API 以获得更好的效果。
- **OpenAI Vision 成本与分辨率**：使用最新模型处理一张 **1080x1920** 图像的成本为 **$0.005525**。
   - 该成员强调了 API 对各种分辨率的可调节性，建议降低分辨率可以帮助降低成本。
- **GPT 的虚拟环境**：一位成员提到他们正在为 GPT 创建一个虚拟环境。
   - 该环境将使 GPT 能够独立编写代码并执行操作，包括控制光标和使用键盘浏览网页，模拟人类交互。
- **Headless Browser 对比 GPT 的点击操作**：一位成员质疑在虚拟环境中使用点击操作的合理性，建议使用 Headless Browser 会是更简单、更明智的方法。
   - 该成员强调了 Headless Browser 在特定任务中的易用性和多功能性，这最终可能会带来更好的功能。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1274400229470244997)** (7 messages): 

> - `GPT Mini Prompt Engineering`
> - `GPT 3.5 vs GPT 4`
> - `ChatGPT Configuration`
> - `Code Interpreter Limitations`
> - `GPT Mini Image Generation` 


- **GPT Mini Prompt Engineering 是个完全不同的挑战**：一位用户表示为 GPT Mini 4.0 模型设置 Prompt 非常困难，感觉与 GPT 3.5 大不相同，需要更多经过优化的 Prompt 和微调。
   - 这一观点与观察结果一致，即 GPT Mini 4.0 似乎需要更精确的 Prompt Engineering，且容错率比前代模型更低。
- **ChatGPT 配置：一个用户的挫败经历**：另一位用户分享了他们在为特定目的配置 ChatGPT 时遇到的困难，提到了诸如 Hallucinations（幻觉）、响应不一致以及在使用和不使用 Code Interpreter 时表现出的行为差异等问题。
   - 他们还提到学习了多个课程并尝试了各种模式，但均未成功，这表明克服这些挑战非常困难。
- **GPT Mini 不能生成图像？别急！**：一位用户最初认为 GPT Mini 无法生成图像，但后来意识到他们使用的是 GPT Mini 而不是完整的 ChatGPT 模型。
   - 这突显了在讨论 Prompt Engineering 时明确所使用模型的重要性。
- **避免 Contrastive Prompting：明智之举？**：一位用户提到完全避免使用 Contrastive Prompting，认为即使在实验场景中，这也是一个难以控制的概念。
   - 这意味着掌握 Contrastive Prompting 可能超出了日常探索的范畴，需要更高级的知识。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1274400229470244997)** (7 messages): 

> - `GPT-4.0`
> - `Prompt engineering`
> - `GPT-3.5`
> - `GPT mini`
> - `Code interpreter` 


- **GPT-4.0 对 Prompt 的容错度较低**：一位成员注意到，为 **GPT mini 4.0 模型** 设置系统、指令或助手 Prompt 的感觉与 **GPT-3.5 或 GPT-4.0** 大不相同。
   - 他们指出，它似乎每次都需要更优化的 Prompt 和微调，且容错度较低。
- **GPT-3.5 是理想的平衡点**：另一位成员建议，在 Prompt 优化需求方面，GPT-3.5 可能介于 GPT-4.0 和 GPT-mini 之间。
   - 他们提到这只是他们的观察，并非其专业领域。
- **使用 GPT 时的挑战**：一位成员分享了他们在让 ChatGPT 为其特定用途进行“配置”时遇到的困难。
   - 他们列举了包括 **幻觉 (hallucinations)**、使用非提供文档中的信息、对不同问题重复相同回答，以及 Code interpreter 行为不一致等挑战。
- **针对图像生成的 Prompt**：一位成员遇到了 GPT mini 无法生成图片的问题。
   - 通过确认他们确实在使用 GPT mini 解决了该问题，因为如果 Prompt 正确，GPT-3.5 和 GPT-4.0 是可以生成图片的。


  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1274478635088220241)** (27 messages🔥): 

> - `CLM`
> - `GPT Model Size`
> - `Model Interpretability`
> - `Procreate`
> - `Markov Chains` 


- **Topology 的新 CLM**：[Continuous Learning Model (CLM)](https://yellow-apartment-148.notion.site/CLM-Docs-507d762ad7b14d828fac9a3f91871e3f) 是一种能够记住交互、自主学习技能并像人类一样在空闲时间思考的新模型。
   - CLM 旨在学习，你可以在 [http://topologychat.com](http://topologychat.com) 进行尝试。
- **GPT5 更大的规模**：为了获得显著的提升，新模型的规模应至少比当前模型大 **20倍**。
   - 训练需要 **6个月**，并且需要一个全新的、大 **20倍** 的数据中心，其建设大约需要一年时间。
- **模型可解释性的挑战**：解释模型非常困难，尤其是在理解参数量方面。
   - 像 Arthur 这样的公司在第一代 AI 安全技术上取得了长足发展，因此可能会出现第二波专注于模型可解释性 (Model Interpretability) 的公司。
- **Procreate 对生成式 AI 的立场**：Procreate 的 CEO 明确表示，他们不会将生成式 AI 集成到产品中。
   - 社交媒体上的艺术家和用户对这一决定表示赞赏，但也有人指出，这可能只是宣布目前不会添加功能，未来可能会发生变化。
- **用于创意写作的 Markov Chains**：一位用户建议可以将 Markov Chains 用作草拟者，将 LLM 用作创意写作的润色者。
   - 他们提到在一个项目中也有类似的经历，使用 Markov chain 生成虚假的 AWS 博客文章，觉得非常幽默。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

</div>

<ul>
<li>
<a href="https://x.com/aidan_mclau/status/1818071890755469365?s=46">Aidan McLau (@aidan_mclau) 的推文</a>：&gt;&gt;Topology 开发的 Continuous Learning Model (CLM)&lt;&lt; CLM 是一种新型模型，能够记住交互、自主学习技能，并在空闲时间进行思考，就像人类一样。CLM 刚刚...</li><li><a href="https://x.com/mparakhin/status/1824330760268157159?s=46">Mikhail Parakhin (@MParakhin) 的推文</a>：@sandeepreddys09 @emollick 为了获得有意义的提升，新模型应该至少大 20 倍。训练至少需要 6 个月，因此你需要一个新的、大 20 倍的数据中心，这需要...</li><li><a href="https://share.snipd.com/snip/712b360a-fc18-4359-8708-f345">Snipd — 突出显示并分享播客中的精彩瞬间</a>：未找到描述</li><li><a href="https://yellow-apartment-148.notion.site/CLM-Docs-507d762ad7b14d828fac9a3f91871e3f">Notion – 集笔记、任务、维基和数据库于一体的工作空间。</a>：一款将日常工作应用融合在一起的新工具。它是为您和您的团队打造的一站式工作空间</li><li><a href="https://share.snipd.com/snip/712b360a-fc18-4359-8708-f34519e7cde3">优化资源，优化性能 | 来自《The Cognitive Revolution》的 2 分钟剪辑 | AI 构建者、研究人员和实时玩家分析</a>：来自热门 Mechanistic Interpretability 的 2 分钟剪辑：Goodfire 为 AI 安全指明道路 | 《The Cognitive Revolution》 | AI 构建者、研究人员和实时玩家...</li><li><a href="https://x.com/lvwerra/status/1825175724224901623">Leandro von Werra (@lvwerra) 的推文</a>：看到一个 360M 的模型（比 GPT-2 小 5 倍！）通过一些技巧能达到如此高度真是太棒了：- 为教育内容策划预训练数据 - 选择调优良好的超参数 -...</li><li><a href="https://x.com/mattshumer_/status/1824836674758557867?s=46">Matt Shumer (@mattshumer_) 的推文</a>：友情提醒，在未来 6 个月左右，将发布比 GPT-4 训练算力多 10 倍的模型</li><li><a href="https://x.com/MKBHD/status/1825521261373489197">Marques Brownlee (@MKBHD) 的推文</a>：收藏这条。一个引人注目的公告。Procreate 的 CEO 公开表示他讨厌生成式 AI，并且他们永远不会将其集成到任何产品中。艺术家和用...</li><li><a href="https://x.com/aakashsastry/status/1825595241346519412?s=46">Aakash (@aakashsastry) 的推文</a>：今天，我们很高兴分享我们最新的视频模型 @hotshotco 的早期预览版。今天您就可以开始使用。链接和更多结果见下方线程 👇</li><li><a href="https://x.com/lateinteraction/status/1825594011484303596?s=46">Omar Khattab (@lateinteraction) 的推文</a>：🧵 DSPy 2.5 之后是什么？还有 DSPy 3.0？我很高兴能分享 DSPy 路线图（Roadmap）的初步草案，随着更多 DSPy 版本的发布，我们将不断扩展和维护这份文档。目标是沟通...</li><li><a href="https://news.ycombinator.com/item?id=41286203">马尔可夫链比 LLM 更有趣 | Hacker News</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1274095343855996981)** (78 条消息🔥🔥): 

> - `DSPy`
> - `Cursor`
> - `Langchain`
> - `Mistral`
> - `Model Merging` 


- **DSPy：尚未成为商业产品**：一位成员询问 **DSPy** 背后是否有商业公司，另一位成员回答“目前还没有，但显然 Omar 正在为此努力。”
   - 另一位成员提到他们参加了 **Cursor** 的办公室见面会，虽然没有 alpha 版本可以分享，但他们确实打了个招呼。
- **DSPy 提升本地模型的潜力**：一位成员报告称在本地运行 **DSPy**，因为有说法称它可以使本地模型在特定任务上达到与 **GPT-4** 相当的效果。
   - 然而，除了基础教程之外，他们并没有进行太多实验，因为前沿模型（frontier models）已经变得非常便宜。
- **DSPy 弥合了 prompting 与 finetuning 之间的鸿沟**：**DSPy** 旨在通过允许用户避免手动 prompt tuning 来弥合 prompting 和 finetuning 之间的差距。
   - 他们在论文中提到的亮点之一是 **DSPy** 允许你避免 prompt tuning，这可能使切换模型、针对数据偏移进行重新调整等变得更容易。
- **DSPy：比人类更擅长 prompting？**：一些成员认为 **DSPy** 在为模型编写 prompt 方面比人类更出色。
   - 然而，其他人认为在 prompting 中仍有人工工程的空间，且人类仍能做许多 **DSPy** 无法做到的事情。
- **Langchain 与底层替换（Substrate Swapping）**：一位成员评论说 **Langchain** 也会更换底层，但只有 Langchain 因此受到批评。
   - 他们还指出，如果能看到这方面的示例就太好了。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://changelog.com/jsparty/331">Building LLM agents in JS with Tejas Kumar (JS Party #331)</a>：KBall 和回归嘉宾 Tejas Kumar 深入探讨了使用 JavaScript 构建 LLM agents 的话题。包括它们是什么、如何发挥作用（包括 Tejas 如何使用自建的 agents 将他的播客效率翻倍...）</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=0#gid=0">AI In Action: Weekly Jam Sessions</a>：暂无描述</li><li><a href="https://github.com/wesen/dspy-grug">GitHub - wesen/dspy-grug: dspy tutorial</a>：dspy 教程。通过在 GitHub 上创建账号来为 wesen/dspy-grug 的开发做出贡献。
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1274099259095912541)** (49 条消息🔥): 

> - `Data Ingestion to KG`
> - `Command-r-plus in Sillytavern`
> - `API Key Partial Responses`
> - `Prompt Tuning`
> - `Cohere Office Hours` 


- **数据摄取至 KG**：一位用户询问了用于提取三元组以将数据摄取到知识图谱（Knowledge Graph）的框架。
- **Command-r-plus 无法正常工作**：一位用户报告说，当上下文长度达到 4000 tokens 时，Sillytavern 中的 command-r-plus 开始出现运行不稳定的情况。
- **API Key 响应不完整**：一位用户报告称其 API key 仅返回部分响应，即使尝试了不同的 Wi-Fi 路由器和蜂窝数据也是如此。
- **Prompt Tuning 仍然存在故障**：一位用户提到 prompt tuning 仍然无法正常工作。
- **Cohere Office Hours**：发布了关于 Cohere Office Hours 活动的提醒，该活动已吸引了 27 名感兴趣的参与者。


  

---

### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1275066929199058985)** (1 条消息): 

> - `Cohere Developer Office Hours`
> - `Prompt Tuning`
> - `Guided Generations API`
> - `LLM University Tool Use Module`
> - `Structured Outputs` 


- **Cohere Developer Office Hours 正式开启！**: 加入 Cohere 的 **高级产品经理** 和 **DevRel** 团队，参加一场轻松的会议，了解 **产品和内容更新**、**最佳实践**，并针对 **Prompt Tuning**、**结合 Agents 的 Guided Generations API** 以及 **LLM University Tool Use Module** 进行 **Q&A**。
   - 活动于今天 **美东时间下午 1 点在 #stage 频道** 举行，可以通过 [此链接](https://discord.com/events/954421988141711382/1265012161965461625) 参加。
- **Cohere Prompt Tuner：优化的 Prompting！**: 了解 **Cohere Prompt Tuner**，这是一个强大的工具，用于优化 Prompt 并提高 LLM 结果的准确性。
   - 博客文章详细介绍了如何使用该工具及 [相关功能](https://cohere.com/blog/intro-prompt-tuner)。
- **用于准确 JSON 生成的 Structured Outputs**: **Structured Outputs** 是 Cohere 工具的最新更新，其 **JSON 生成** 速度比开源实现 **快 80 倍** 且 **更准确**。
   - 这一新功能提高了 JSON 输出的准确性，并在 [这篇博客文章](https://cohere.com/blog/introducing-structured-outputs) 中进行了讨论。
- **使用 LLM University 模块实现工作流自动化**: **LLM University Tool Use Module** 通过利用 **Command R+** 的能力简化了 **工作流自动化**。
   - 通过这个新模块学习如何 **自动化任务** 和 **工作流**，详见 [这篇博客文章](https://cohere.com/blog/tool-use-llmu)。
- **不要错过 Cohere 的 Office Hours！**: 不要错过这个 **向 Cohere 专家** 和服务器中其他 **开发者** 学习的机会。
   - 加入讨论，**扩展你对 Cohere 工具最新更新的知识**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://cohere.com/blog/intro-prompt-tuner">Introducing Cohere Prompt Tuner: Prompt Optimization at Your Fingertips</a>: 使用 Cohere 的新工具 Prompt Tuner 自动改进你的 Prompt，该工具今日开启 Beta 测试。</li><li><a href="https://cohere.com/blog/introducing-structured-outputs">Introducing Structured Outputs with JSON Response Format</a>: Structured Outputs 提高了 JSON 生成的准确性，且比开源实现快 80 倍。</li><li><a href="https://cohere.com/blog/tool-use-llmu">Learn Workflow Automation with Our New LLM University Module on Tool Use</a>: 学习如何利用 Command R+ 的工具使用能力来自动化任务和工作流。
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1274148542755110953)** (43 条消息🔥): 

> - `API key monitoring`
> - `production keys`
> - `Cohee chat`
> - `Trial keys`
> - `Structured output` 


- **生产环境 API Key 与监控**: 一位成员询问获得生产环境 API Key 是否需要他们监控所有 LLM 输出中未指明的不良内容。
- **用于 Cohere Chat 的生产环境 Key**: 一位成员询问生产环境 Key 是否可以在 Cohere Chat 上使用。
- **生产环境 Key 问题**: 一位成员报告在尝试于 Cohere Chat 上使用其生产环境 Key 时收到 [429] 错误。
- **生成结构化 JSON 输出**: 一位成员询问关于保证结构化输出的开源实现。
- **结构化输出指南**: 一位成员询问使用 LLM 生成结构化 JSON 对象的方法。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.cohere.com/docs/structured-outputs-json">Structured Generations (JSON) — Cohere</a>: 未找到描述</li><li><a href="https://status.cohere.com/">Cohere Status Page Status</a>: Cohere 状态页面的最新服务状态</li><li><a href="https://github.com/guidance-ai/guidance">GitHub - guidance-ai/guidance: A guidance language for controlling large language models.</a>: 一种用于控制大语言模型的引导语言。- guidance-ai/guidance
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1274631557369299069)** (1 messages): 

> - `CursorLens`
> - `Cohere models` 


- **CursorLens: Prompt 分析工具**：CursorLens 是一款提供 Prompt 分析功能的工具，并允许你配置 Cursor 本身不支持的模型，例如 Cohere 模型。
   - 它允许你查看 Prompt 的分析数据，并配置 Cursor 原生不支持的模型（如 Cohere）。
- **用于代码库搜索的 Cohere 模型**：Cohere 模型被认为在代码库搜索和查询方面非常有效。
   - 用户认为 Cohere 模型在处理跨代码库的搜索和查询时表现非常出色。
- **CursorLens 已开源**：CursorLens 是开源的，任何人都可以尝试。
   - 用户鼓励其他人尝试 CursorLens 并为该开源项目做出贡献。



**提及的链接**：<a href="https://www.producthunt.com/posts/cursor-lens"> CursorLens - Cursor IDE 的开源仪表板和分析工具 | Product Hunt</a>：Cursor.sh IDE 的开源仪表板。记录 AI 代码生成、跟踪使用情况并控制 AI 模型（包括本地模型）。可本地运行或使用即将推出的托管版本。

  

---


### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/1274227847006785597)** (2 messages): 

> - `Toolkit Bug Fixes`
> - `Python SDK Linting` 


- **Toolkit 和 Python SDK 错误修复与 Linting**：一名成员向 Cohere Toolkit 和 Python SDK 提交了错误修复和 Linting（代码检查）改进。
   - 另一名成员对该贡献表示感谢。
- **衷心感谢**：一名成员对该贡献表示了感谢。


  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1274135442501013565)** (12 messages🔥): 

> - `Yi Tay's Work Style`
> - `AI Regulation`
> - `01AI's future` 


- **Yi Tay 是一位不知疲倦的工作者**：讨论围绕各 AI 机构的工作风格展开，一名成员提到 **Yi Tay** 以 **“混沌不眠不休（chaos no sleep grind）”** 的心态在工作。
- **Nancy Pelosi 反对加州 AI 法案**：**荣誉议长 Nancy Pelosi** 发表声明，反对关于 AI 监管的 **加州参议院第 1047 号法案**。
- **01AI 的市场策略受到质疑**：一名成员询问 **01AI 未来的市场策略**，因为最近的一条推文暗示其可能会从 **非中国市场** 撤退。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/phill__1/status/1825438202548658526?s=46">来自 Phil (@phill__1) 的推文</a>：.@01AI_Yi 发生了什么？他们要退出非中国市场吗？</li><li><a href="http://pelosi.house.gov/news/press-releases/pelosi-statement-opposition-california-senate-bill-1047">Pelosi 反对加州参议院第 1047 号法案的声明</a>：旧金山 – 荣誉议长 Nancy Pelosi 发表了这份反对加州参议院第 1047 号法案的声明：
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1274109871003930664)** (15 messages🔥): 

> - `Hermes 2.5`
> - `Mistral struggles`
> - `Model Merging`
> - `Open Empathic`
> - `Zicheng Xu Laid Off` 


- **Zicheng Xu 被裁员**：Zeyuan Allen-Zhu 宣布，“Part 2.2” 教程的作者 Zicheng Xu 意外被裁。
   - Allen-Zhu 强烈推荐 Xu，并为潜在的合作伙伴或雇主提供了他的电子邮箱：zichengBxuB42@gmail.com（请删除大写字母 'B'）。
- **Nous Hermes Discord 争议**：一名用户提到 Nous Discord 中的一场讨论，涉及某用户被认为表现无礼以及对评估设置的误导。
   - 该用户提到他们的评估细节在论文的 SFT 部分，并承认出错的感觉并不好，但文章的核心内容仍然有效。
- **Meta Cooking (模型评估)**：一名用户想知道什么是 “meta cooking”，暗示 Nous Discord 中可能存在冲突或争议。
   - 该用户提到发现了关于评估设置的矛盾信息，可能是由于使用了默认的 LM Harness 设置而没有清晰的文档说明。
- **评估很难，应重点关注**：该用户表示，这次 Discord 争议的经历激励他们写一篇关于评估的有趣文章。
   - 他们承认准确且一致的评估非常困难，并认为强调这一方面非常重要。



**提及的链接**：<a href="https://x.com/zeyuanallenzhu/status/1824550891304915081?s=46">来自 Zeyuan Allen-Zhu (@ZeyuanAllenZhu) 的推文</a>：(1/2) 很多人询问 Part 2.2，很抱歉延迟了。我们的作者 Zicheng Xu 意外被裁。我给予他最强力的推荐（见下一条推文）。如果对该项目感兴趣或有……

  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1274164697662226454)** (15 messages🔥): 

> - `AI21 Models`
> - `AI21 vs AI2`
> - `AI Bubble`
> - `Gary Marcus`
> - `AI Safety` 


- **LMSYS 上的 AI21 模型**：LMSYS 上新的 "toto" 模型很可能来自 AI21。
   - 这可能就是为什么 AI2 被重命名为 Ai2 的原因，因为 AI2A12 容易与 AI21 混淆。
- **Gary Marcus 重新审视 AI Bubble 担忧**：Gary Marcus 回顾了他在 AGI-21 上的主题演讲，指出尽管 AI 取得了重大进展，但他当时强调的许多问题在今天仍然存在。
   - 这段名为 "The AI Bubble: Will It Burst, and What Comes After?" 的视频已在 YouTube 上发布。
- **转向 AI Safety 职业轨迹**：一位用户分享了一篇关于将职业轨迹转向 AI Safety 的博客文章。
   - 他们解释说，编写谜题占用了太多的精力，他们想在今年做出一些改变。
- **Meta 的 GenAI 发布免微调个性化图像生成**：Meta 的 GenAI 发布了一篇名为 "Imagine Yourself: Tuning-Free Personalized Image Generation" 的新研究论文。
   - 该功能目前已作为 Beta 版在 Meta AI 中向美国用户开放。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.alexirpan.com/2024/08/18/nine-years.html">Nine Years Later</a>：Sorta Insightful 今天九岁了！</li><li><a href="https://x.com/swishfever/status/1824605103434698820">来自 fishy business (@swishfever) 的推文</a>：LMSYS 上新的 "toto" 模型很可能来自 AI21</li><li><a href="https://x.com/aiatmeta/status/1825593390043730390?s=46">来自 AI at Meta (@AIatMeta) 的推文</a>：🆕 来自 Meta GenAI 的研究论文：Imagine yourself: Tuning-Free Personalized Image Generation。研究论文 ➡️ https://go.fb.me/wre8f0 想要尝试吗？该功能目前已作为 Beta 版在...</li><li><a href="https://tenor.com/bj7gg.gif">Office Space Michael Bolton GIF - Office Space Michael Bolton Why Should I Change - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=91SK90SahHc">The AI Bubble: Will It Burst, and What Comes After?</a>：Gary Marcus 教授回顾了他在 AGI-21 上的主题演讲，指出尽管取得了重大进展，但他当时强调的许多问题在今天仍然存在...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1274151699552010387)** (45 messages🔥): 

> - `Procrastination`
> - `Blog Design`
> - `Substack`
> - `Fast Writing` 


- **拖延是一个普遍问题**：一位成员提到，他们一直在拖延恢复博客运行，因为他们想把设计做得尽善尽美，但他们知道这其实是一种分心。
   - 他们还承认自己写作速度相当快，但很容易说服自己不去写。
- **Substack 易于使用但难以定制**：另一位成员提到，他们为了在博客顶部放置大型艺术字与 Substack 搏斗了数小时。
   - 他们还表达了希望对博客设计有更多控制权的愿望，这也是他们没有使用 Substack 等平台的原因。
- **FastHTML 让写博客变得简单有趣**：一位成员提到，他们使用 FastHTML 在一天内建立了一个博客网站。
   - 他们发现这次体验非常有趣且令人愉悦。


  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1274527247600193617)** (15 条消息🔥): 

> - `GrokAdamW optimizer`
> - `GrokFast paper`
> - `Gemma 2B update`
> - `Transformers dev version`
> - `Unsloth` 


- **GrokAdamW 优化器发布**：GrokAdamW 是一款旨在促进快速 Grokking 的 PyTorch 优化器，现已发布，并已通过 Transformers 集成与 Axolotl 协同工作。[GrokAdamW 仓库](https://github.com/cognitivecomputations/grokadamw)
- **GrokAdamW 灵感源自 GrokFast**：该优化器受 GrokFast 论文启发，旨在加速模型在 Grokking 现象下的泛化。[GrokFast 论文](https://arxiv.org/abs/2405.20233)
- **Gemma 2B 更新导致 Axolotl 崩溃**：Gemma 2B 仓库的一个更新导致了 Axolotl 崩溃。
- **提醒使用 Transformers 开发版**：使用 Transformers 的开发版（dev version）非常重要。[开发版安装指南](https://github.com/huggingface/transformers.git)
- **通过 Unsloth 微调 Gemma 2、Llama 3.1 和 Mistral，速度提升 2-5 倍，显存占用减少 70%！**：Unsloth 支持通过 bitsandbytes 直接使用量化的 4bit 模型，使 Gemma 2、Llama 3.1 和 Mistral 的微调速度提升 2-5 倍，且显存占用减少 70%。[Gemma 2 (2B) Google Colab 笔记本](https://colab.research.google.com/drive/1weTpKOjBZxZJ5PQ-Ql8i6ptAY2x-FWVA?usp=sharing) [Gemma 2 (9B) Google Colab 笔记本](https://colab.research.google.com/drive/1vIrqH)


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/unsloth/gemma-2-2b">unsloth/gemma-2-2b · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/cognitivecomputations/grokadamw">GitHub - cognitivecomputations/grokadamw</a>: 通过在 GitHub 上创建账号来为 grokadamw 的开发做出贡献。</li><li><a href="https://github.com/huggingface/transformers/pull/32521">为 Transformers 添加 GrokAdamW 优化器支持（由 ehartford 提交的 Pull Request #32521）</a>: 此 PR 做了什么？为 Transformers 库添加了对 GrokAdamW 优化器的支持。引入的更改包括将 GrokAdamW 优化器集成到...</li><li><a href="https://arxiv.org/abs/2405.20233">Grokfast: 通过放大慢梯度加速 Grokking</a>: 机器学习中一个令人费解的现象被称为 Grokking，即在对训练数据近乎完美过拟合后的数万次迭代后才实现延迟泛化。重点关注长期的...</li><li><a href="https://github.com/ironjr/grokfast">GitHub - ironjr/grokfast: 论文 "Grokfast: Accelerated Grokking by Amplifying Slow Gradients" 的官方仓库</a>: 论文 "Grokfast: Accelerated Grokking by Amplifying Slow Gradients" 的官方仓库 - ironjr/grokfast
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1274177210059587684)** (20 条消息🔥): 

> - `Gemma 2b training issues`
> - `Zero Loss`
> - `Eager Attention` 


- **Gemma 2B 训练期间出现零损失**：有用户报告在训练 **Gemma 2B** 模型时，Loss 持续为 **0.0**，且梯度范数（gradient norm）为 **nan**。
- **建议在 Gemma 2B 训练中使用 Eager Attention**：另一位用户建议在训练 **Gemma 2B** 模型时使用 **eager attention** 而非 **sdpa**。
- **Eager Attention 修复了问题**：经历零损失问题的用户确认 **eager attention** 修复了该问题。


  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1274243931667828767)** (17 messages🔥): 

> - `Chat Template`
> - `Axolotl prompt strategies`
> - `Using custom loaders`
> - `Training with ShareGPT`
> - `Fine-tuning with Axolotl` 


- **Axolotl 的 Chat Template**：用户询问了关于在 Axolotl 的 `.yml` 配置文件中使用 `Chat Template` 类型的说明。他们特别感兴趣于如何指定使用哪个 loader，例如 ShareGPT。
- **在 Axolotl 中使用自定义 Loader**：另一位用户建议，用户可以通过提供自定义的 `.yml` 文件来指定要使用的 loader。
- **Axolotl 对 Chat Template 的支持**：用户表达了对在 Axolotl 中使用 `chat_template` 类型的兴趣，并询问它是否支持其数据集中的 `role: system` 消息。
- **使用 Axolotl 进行 Fine-tuning：无需编程**：一位用户澄清说，使用 Axolotl 进行 Fine-tuning 通常不需要编程知识，而是需要理解如何格式化数据集以及如何适配现有的示例。
- **LLama 3.1 70b Fine-tuning：用户体验**：一位用户提到拥有一台强大的 AI 设备来运行 LLama 3.1 70b，但觉得它在某些关键领域仍有不足。他们拥有大量自己编写和抓取的内容数据集，并希望将其用于 Fine-tuning。



**提到的链接**：<a href="https://github.com/axolotl-ai-cloud/axolotl/pull/1732">Allow using tokenizer&#39;s default chat template or pass custom jinja chat template by chiragjn · Pull Request #1732 · axolotl-ai-cloud/axolotl</a>：关闭了 #1689。更改摘要：在 `chat_template` prompt strategy 中增加了 `tokenizer_default` 选项，允许使用来自 tokenizer 的 `config.json` 中的 chat template。允许 fa...

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1275145386327150612)** (1 messages): 

> - `LLaMa 3.1 8b Lora`
> - `Post-Hoc Reasoning`
> - `Sonnet 3.5`
> - `Claude` 


- **用于 post-hoc reasoning 检测的 LLaMa 3.1 8b Lora**：一位用户正在训练一个 **LLaMa 3.1 8b Lora**，用于检测对话中的 **post-hoc reasoning**。
   - 他们花了三天时间整理了一个包含**不到 100 个多轮对话**的小型数据集，大约有 **30k tokens**，以辅助完成这项任务。
- **Sonnet 3.5 和 Claude 在处理 post-hoc reasoning 示例时遇到困难**：该用户使用 **Sonnet 3.5** 来协助生成示例，但尽管精心设计了 prompt，仍必须修正每个生成示例中的多个问题。
   - 他们必须对想要在数据集中传达的每个具体想法进行多次迭代，手动编辑每个示例以获得所需的输出。
- **模型倾向于进行 post-hoc reasoning**：即使指示模型不要创建带有 **post-hoc reasoning** 的示例，由于其 Fine-tuning 数据的缘故，它们仍然会生成此类内容。
   - 用户不得不手动修复这些问题，凸显了训练模型以避免特定推理模式的难度。


  

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1274089813460520960)** (39 条消息🔥): 

> - `LangChain Caching`
> - `LLM structured output`
> - `LangChain JSON parsing`
> - `RAG chatbot delete functionality`
> - `Hybrid search relevance` 


- **LangChain 缓存问题**：一位成员询问为什么 `.batch_as_completed()` 没有通过缓存加速，尽管在缓存后 `.invoke()` 和 `.batch()` 几乎是瞬间完成的。
   - 他们注意到缓存是在第一次运行后填充的，但 `.batch_as_completed()` 似乎没有利用缓存。
- **LLM 在结构化输出方面存在困难**：一位成员提到本地 LLM（如 Llama 3.1）通常难以产生一致的结构化输出。
   - 他们询问是否有专门用于训练模型的数据库，以提高 JSON 解析和结构化输出的能力，以便与工具或 ReAct Agent 配合使用。
- **在 RAG 聊天机器人中删除文件**：一位成员询问如何在以 MongoDB 作为向量数据库的 RAG 聊天机器人中实现文件删除功能。
   - 一份有用的回复提供了使用 LangChain 库中针对 MongoDB 向量存储和 OpenAIFiles 的 `delete` 方法示例，并附带了相关的文档链接。
- **混合搜索相关性问题**：一位成员描述了一个使用 BM25Retriever 和向量相似度搜索的混合搜索方法的 RAG 应用，但他们在检索文档的相关性和生成的答案方面遇到了问题。
   - 建议包括检查文档质量、调整检索器配置、评估 Chain 设置，以及审查 Prompt 和 LLM 配置。
- **多语言 RAG 工作流**：一位成员讨论了一个多语言 RAG 工作流，涉及将用户问题翻译成英文，检索英文的相关文档，然后用用户的母语制定答案。
   - 讨论内容包括这种方法与多语言嵌入文档相比的有效性，以及多语言 Embedding 模型是否允许跨语言检索。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.smith.langchain.com/concepts/evaluation#rag-evaluation-summary>).">Evaluation | 🦜️🛠️ LangSmith</a>: AI 应用开发的步伐往往受限于高质量的评估，因为存在选择悖论。开发者经常想知道如何设计他们的 Prompt，或者哪种 LLM 最能平衡...</li><li><a href="https://js.langchain.com/v0.2/docs/integrations/vectorstores/mongodb_atlas/#delete-items-from-vector-store>)">MongoDB Atlas | 🦜️🔗 Langchain</a>: 本指南提供了快速入门 MongoDB 的概述。</li><li><a href="https://github.com/langchain-ai/langchain/issues/17508>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。欢迎在 GitHub 上为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/sksarvesh007/adaptive-rag/blob/main/langgraph_adaptive_rag.ipynb">adaptive-rag/langgraph_adaptive_rag.ipynb at main · sksarvesh007/adaptive-rag</a>: 欢迎在 GitHub 上为 sksarvesh007/adaptive-rag 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1274354727970148393)** (1 条消息): 

> - `ShortURL.at`
> - `URL Shortener`
> - `Social Media Links` 


- **ShortURL.at 是一个免费的 URL 缩短器**：ShortURL.at 是一个缩短 URL 并生成短链接的免费工具，方便分享。
   - 该服务提供高级功能，如自定义短链接、详细分析、API、UTM 构建器、二维码、浏览器扩展、应用集成和支持。
- **ShortURL.at 可缩短来自各种平台的链接**：ShortURL.at 允许缩短来自 [Instagram](https://www.instagram.com/)、[Facebook](https://www.facebook.com/)、[YouTube](https://www.youtube.com/)、[Twitter](https://www.twitter.com/)、[LinkedIn](https://www.linkedin.com/)、[WhatsApp](https://www.whatsapp.com/)、[TikTok](https://www.tiktok.com/)、博客和网站的长链接。
   - 只需粘贴长 URL 并点击 Shorten URL 按钮。在下一页，复制缩短后的 URL 并在网站、聊天和电子邮件中分享。



**提到的链接**: <a href="https://shorturl.at/RbRhn">ShortURL - URL Shortener</a>: 未找到描述

  

---

### **LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1274354732525289482)** (1 条消息): 

> - `Steam Gift Card`
> - `ShortURL`
> - `Shortener` 


- **Steam 礼品卡出售**: 一位用户正在出售一张 **$50 Steam 礼品卡**，并提供了一个缩短后的 URL 以供购买。
- **用于 URL 缩短的 ShortURL**: **ShortURL** 是一个用于缩短 URL 和创建短链接的免费工具。
- **ShortURL 高级功能**: ShortURL 提供增强 URL 缩短体验的**高级功能 (premium features)**。
- **ShortURL 兼容平台**: ShortURL 可以缩短来自各种平台的长链接，如 **Instagram, Facebook, YouTube, Twitter, LinkedIn, WhatsApp, TikTok, 博客和网站**。



**提到的链接**: <a href="https://shorturl.at/RbRhn">ShortURL - URL Shortener</a>: 未找到描述

  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1274353984890474618)** (4 条消息): 

> - `CursorLens`
> - `LLMs`
> - `Machine Learning from Scratch` 


- **CursorLens: Cursor 用户的新仪表盘**: **CursorLens** 是一个面向 Cursor 用户的开源仪表盘，提供关于 Prompt 的分析，并允许你配置 Cursor 本身不提供的模型。
   - 它最近在 ProductHunt 上发布：[https://www.producthunt.com/posts/cursor-lens](https://www.producthunt.com/posts/cursor-lens)。
- **LLMs 详解：从助手到深度概念**: 这篇博文深入探讨了 LLMs 的工作原理，从高层抽象开始，逐渐深入到 Tokenization, Sampling 和 Embedding 等概念。
   - 它还讨论了当前 LLMs 的局限性，例如无法计算 "strawberry" 中 R 的数量以及反转字符串 "copenhagen"。在此处查看博文：[https://amgadhasan.substack.com/p/explaining-how-llms-work-in-7-levels](https://amgadhasan.substack.com/p/explaining-how-llms-work-in-7-levels)。
- **从零开始的机器学习 (Machine Learning from Scratch)：初学者友好指南**: 这个 GitHub 仓库提供了一个从零开始学习机器学习的循序渐进指南，假设读者没有先验知识。
   - 它涵盖了核心机器学习算法和神经网络，通过实际案例（包括 Gradient Descent 和 Backpropagation）解释底层数学原理。在此处查看仓库：[https://github.com/DorsaRoh/Machine-Learning](https://github.com/DorsaRoh/Machine-Learning)。 


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://amgadhasan.substack.com/p/explaining-how-llms-work-in-7-levels">Explaining how LLMs work in 7 levels of abstraction</a>: 概览</li><li><a href="https://www.producthunt.com/posts/cursor-lens"> CursorLens - Open Source dashboard and analytics for Cursor IDE | Product Hunt</a>: 一个用于 Cursor.sh IDE 的开源仪表盘。记录 AI 代码生成，跟踪使用情况，并控制 AI 模型（包括本地模型）。可在本地运行或使用即将推出的托管版本。</li><li><a href="https://github.com/DorsaRoh/Machine-Learning">GitHub - DorsaRoh/Machine-Learning: Machine learning: 0 ➔ 1</a>: 机器学习：0 ➔ 1。通过在 GitHub 上创建账号来为 DorsaRoh/Machine-Learning 的开发做出贡献。</li><li><a href="https://shorturl.at/RbRhn">ShortURL - URL Shortener</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1274354232966905897)** (1 条消息): 

> - `URL Shortener`
> - `ShortURL` 


- **ShortURL: 免费的 URL 缩短器**: ShortURL 是一个缩短 URL 并生成短链接的免费工具，方便分享。
   - 只需粘贴长 URL 并点击 Shorten URL 按钮。在下一页，复制缩短后的 URL 并分享到网站、聊天工具和电子邮件中。
- **ShortURL 高级功能**: 高级功能包括自定义短链接、强大的仪表盘、详细的分析、API、UTM 构建器、QR 码、浏览器扩展、应用集成和支持。
   - 你可以在此处创建账号以使用高级功能：[创建账号](https://shorturl.at/vSZ02)
- **适用于各种平台的 ShortURL**: ShortURL 允许缩短来自各种平台的长链接，如 [Instagram](https://www.instagram.com/), [Facebook](https://www.facebook.com/), [YouTube](https://www.youtube.com/), [Twitter](https://www.twitter.com/), [LinkedIn](https://www.linkedin.com/), [WhatsApp](https://www.whatsapp.com/), [TikTok](https://www.tiktok.com/), 博客和网站。



**提到的链接**: <a href="https://shorturl.at/RbRhn">ShortURL - URL Shortener</a>: 未找到描述

  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1274085466739708007)** (37 条消息🔥): 

> - `Orange Pi 5`
> - `GPT-4o-mini`
> - `OpenInterpreter settings`
> - `OpenInterpreter API`
> - `Local LLMs for bash commands` 


- **Orange Pi 5 评测**：一位成员发布了 **Orange Pi 5** 的 [YouTube 视频评测](https://youtu.be/79lquFD3oT4)，这是一款新型的**高性价比且功能强大的 Arm 架构 SBC**。
   - 视频指出，**Orange Pi 5 不应与 Raspberry Pi 5 混淆**。
- **GPT-4o-mini 模型设置困扰**：一位用户表示在使用 `set model` 命令将模型设置为 **GPT-4o-mini** 时遇到困难。
   - 另一位成员迅速提供了解决方案：`interpreter --model gpt-4o-mini`。
- **OpenInterpreter 设置重置**：一位用户在尝试 OpenInterpreter 设置后遇到问题，寻求恢复或重置为默认设置的方法。
   - 另一位成员建议使用 `interpreter --profiles` 命令来查看和编辑配置文件，或者使用 `pip uninstall open-interpreter` 和 `pip install open-interpreter` 进行卸载并重新安装。
- **OpenInterpreter API 集成**：一位用户表示有兴趣将 OpenInterpreter 集成到他们现有的 AI 核心中，通过向 OI 发送请求、运行代码并接收输出来实现。
   - 建议该用户使用 Python 脚本（可能配合 Flask 服务器）来处理其 AI 核心与 OpenInterpreter 之间的通信。
- **适用于 Bash 命令的本地 LLM**：一位成员询问有哪些擅长处理 bash 命令的本地 LLM 推荐。
   - 另一位成员推荐了 **CodeStral** 和 **Llama 3.1**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.openinterpreter.com/guides/basic-usage#programmatic-chat">基础用法 - Open Interpreter</a>：未找到描述</li><li><a href="https://docs.openinterpreter.com/settings/all-settings#os-mode">所有设置 - Open Interpreter</a>：未找到描述</li><li><a href="https://docs.openinterpreter.com/settings/all-settings#auto-run">所有设置 - Open Interpreter</a>：未找到描述</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/blob/main/interpreter/terminal_interface/profiles/defaults/default.yaml">GitHub 上的 open-interpreter/interpreter/terminal_interface/profiles/defaults/default.yaml</a>：计算机的自然语言接口。通过在 GitHub 上创建账号为 OpenInterpreter/open-interpreter 的开发做出贡献。</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/tree/main/interpreter/core/computer">GitHub 上的 open-interpreter/interpreter/core/computer</a>：计算机的自然语言接口。通过在 GitHub 上创建账号为 OpenInterpreter/open-interpreter 的开发做出贡献。</li><li><a href="https://youtu.be/79lquFD3oT4">Orange Pi 5 上手评测，终于等到了一款新型、实惠且强大的 Arm 架构 SBC！</a>：在这段视频中，我们来看看全新的 Orange Pi 5 SBC。&quot;不要与 Raspberry Pi 5 混淆。这是最便宜的 RK3588S 单板计算机...</li><li><a href="https://github.com/OpenInterpreter/open-interpreter?tab=readme-ov-file#sample-fastapi-server">GitHub - OpenInterpreter/open-interpreter: 计算机的自然语言接口</a>：计算机的自然语言接口。通过在 GitHub 上创建账号为 OpenInterpreter/open-interpreter 的开发做出贡献。</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/pull/1409">由 CyanideByte 提交的 Pull Request #1409：隐藏 LiteLLM 缺失成本完成映射的警告</a>：描述你所做的更改：如果使用的模型尚未添加到 LiteLLM 的成本完成映射中，则隐藏其产生的大量警告。参考任何相关的 issue（例如...</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/blob/main/interpreter/terminal_interface/profiles/defaults/os.py">GitHub 上的 open-interpreter/interpreter/terminal_interface/profiles/defaults/os.py</a>：计算机的自然语言接口。通过在 GitHub 上创建账号为 OpenInterpreter/open-interpreter 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1274134055306137661)** (2 条消息): 

> - `OpenInterpreter 设备发布时间线` 


- **OpenInterpreter 设备发布时间线：仍未确定**：一位用户询问了设备的发布时间线，特别是是否预计在今年发货。
   - 虽然没有提供具体的时间表，但目前尚不清楚该设备是否会在今年或更晚发货。
- **OpenInterpreter 设备的购买渠道**：另一位用户询问了该设备是否可以购买。
   - 目前没有提供关于该设备是否已开放购买的信息。


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1274320667793293345)** (4 条消息): 

> - `使用 OpenInterpreter 进行 VSCode 编辑`
> - `终端卡住` 


- **使用 OpenInterpreter 进行 VSCode 编辑**：一位成员询问是否有人尝试过使用 **OpenInterpreter** 进行 **VSCode** 编辑，具体操作是跳转到第 300 行并将变量 `x_alpha` 更改为 camelCase。
   - 另一位成员回复说他们还没有尝试过。
- **OpenInterpreter 导致终端卡住**：第一位成员提到 **OpenInterpreter** 上次对他们有效，但中间 **终端卡住了**。



**提及的链接**：<a href="https://www.youtube.com/watch?v=pou46iBNZHw">Exists - 文本即游戏，就是这么简单</a>：文本转游戏 AI 创建平台，让任何人都能在瞬间创建独特的多人游戏。加入我们的 Discord 获取封闭测试资格：https://discord.com/invite/...

  

---

### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1274117262302777455)** (9 条消息🔥): 

> - `LLMs`
> - `RAG`
> - `Knowledge Graphs`
> - `WeKnow-RAG`
> - `Meta Optimization` 


- **LLMs 在可靠性方面面临挑战**：大语言模型 (LLMs) 容易产生事实错误的信息，并经常生成“幻觉”内容，从而损害其可靠性。
- **WeKnow-RAG 提升了 LLM 的可靠性**：WeKnow-RAG 系统将网络搜索和 Knowledge Graphs 集成到检索增强生成 (RAG) 系统中，以增强 LLM 的准确性和可靠性。
- **用于工作流优化的 Meta Optimizer**：一位用户分享称，最近发表的一篇论文实现了与其在 meta-optimization 领域正在进行的工作类似的想法。
- **ARC 逻辑谜题：AI 智能的测试**：一位用户分享了一篇论文链接，该论文在 ARC 逻辑谜题任务上评估了一种新算法，该任务旨在评估 AI 系统的通用智能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/jeffclune/status/1825551361808990611">来自 Jeff Clune (@jeffclune) 的推文</a>: 我们在具有挑战性的 ARC 逻辑谜题任务上评估了所提出的算法，该任务测试了 AI 系统的通用智能。它逐步发现优于 state-of-the-art 的新型 agents...</li><li><a href="https://github.com/jmanhype/VITA_AI_Assistant">GitHub - jmanhype/VITA_AI_Assistant: 一个用于音频、图像和文本处理的模块化 AI Assistant 项目。</a>: 一个用于音频、图像和文本处理的模块化 AI Assistant 项目。 - jmanhype/VITA_AI_Assistant</li><li><a href="https://github.com/jmanhype/ATLAS-Automated-Trading-and-Liquidity-Analysis-System">GitHub - jmanhype/ATLAS-Automated-Trading-and-Liquidity-Analysis-System</a>: 通过创建账户为 jmanhype/ATLAS-Automated-Trading-and-Liquidity-Analysis-System 的开发做出贡献。</li><li><a href="https://www.arxiv.org/abs/2408.05211">VITA: 迈向开源交互式全能多模态 LLM</a>: GPT-4o 卓越的多模态能力和交互体验强调了它们在实际应用中的必要性，然而开源模型很少在这两个领域都表现出色。在本文中，我们...</li><li><a href="https://github.com/jmanhype/WeKnow-Information-Retrieval-Assistant/tree/master">GitHub - jmanhype/WeKnow-Information-Retrieval-Assistant: WeKnow 信息检索助手是一个先进的 AI 驱动系统，具有语音交互助手 VITA。它结合了 OpenAI 的 GPT-3.5 Turbo 进行自然语言处理和 Perplexity API 进行网络搜索。该项目提供自定义评估指标和异步内容检索，旨在提供高效准确的信息。</a>: WeKnow 信息检索助手是一个先进的 AI 驱动系统，具有 VITA 语音交互助手。它结合了 OpenAI 的 GPT-3.5 Turbo 进行自然语言处理...</li><li><a href="https://arxiv.org/abs/2408.07611">WeKnow-RAG: 一种集成网络搜索和 Knowledge Graphs 的自适应检索增强生成方法</a>: 大语言模型 (LLMs) 为自适应智能 agents 的发展做出了巨大贡献，并被定位为实现通用人工智能 (AGI) 的重要途径。然而...
</li>
</ul>

</div>
  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519685616025600/1274600395146264576)** (25 条消息🔥): 

> - `DSPy 2.5 & 3.0 Roadmap`
> - `Langgraph & Routequery Error`
> - `Optimizing Expert-Engineered Prompts`
> - `DSPy & API Integration` 


- **DSPy 路线图公布！**：DSPy 2.5（可能在 1-2 周内发布）和 DSPy 3.0（几个月内发布）的路线图草案已经公布。
   - 该路线图概述了目标、里程碑和工作方向，并欢迎社区的意见和贡献。[DSPy 路线图链接](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/roadmap.md)
- **Langgraph 和 Routequery 类错误**：一位成员在 Langgraph 中使用 `routequery` 类时遇到了错误。
   - 他们请求关于将 DSPy 与大量工具集成的指导，并分享了 Langgraph 实现的链接：[Adaptive RAG](https://github.com/sksarvesh007/adaptive-rag/blob/main/langgraph_adaptive_rag.ipynb)。
- **优化专家设计的提示词**：一位成员询问 DSPy 是否可以优化已经由专家开发人员手动设计的提示词（Prompts）。
   - 他们询问 DSPy 是否不仅对优化初始草稿有效，而且对改进已经成熟的提示系统也有效。
- **DSPy 与 API 集成**：一位成员询问是否可以将 DSPy 与来自 AI/ML.ai 的 API 配合使用。
   - 他们询问了如何建立 DSPy 与该 API 之间的连接。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/lateinteraction/status/1825594011484303596">来自 Omar Khattab (@lateinteraction) 的推文</a>：🧵DSPy 2.5 接下来会发生什么？还有 DSPy 3.0？我很兴奋能分享 DSPy 路线图的早期草案，随着更多 DSPy 版本的推出，我们将不断扩展和维护这份文档。目标是沟通...</li><li><a href="https://github.com/stanfordnlp/dspy/blob/main/docs/docs/roadmap.md">dspy/docs/docs/roadmap.md at main · stanfordnlp/dspy</a>：DSPy：用于编程（而非提示）基础模型的框架 - stanfordnlp/dspy</li><li><a href="https://github.com/sksarvesh007/adaptive-rag/blob/main/langgraph_adaptive_rag.ipynb">adaptive-rag/langgraph_adaptive_rag.ipynb at main · sksarvesh007/adaptive-rag</a>：通过在 GitHub 上创建账户为 sksarvesh007/adaptive-rag 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/)** (1 条消息): 

batmanosama: 我更新了它，感谢指出这一点
  

---


### **DSPy ▷ #[colbert](https://discord.com/channels/1161519468141355160/1250300504462856265/1274539998498131989)** (4 条消息): 

> - `Colpali finetuning`
> - `VLM tuning`
> - `Domain expertise`
> - `Colpali data` 


- **微调 Colpali**：关于微调 **Colpali** 的方法出现了一个问题，由于其领域特定的性质，该模型似乎需要专门的专业知识。
- **Colpali 微调的数据需求**：一个关键讨论点集中在有效微调 **Colpali** 所需的数据类型上。


  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1274094572301193216)** (25 条消息🔥): 

> - `FLUX Dev`
> - `LLM for medical assistance`
> - `Medical LLMs`
> - `LoRA Training` 


- **FLUX Dev 可以创建 3x3 照片网格**：一位用户分享说 **FLUX Dev** 可以生成同一个（虚构）人物的 3x3 照片网格。
- **为特定目的训练 LoRA**：一位用户表示有兴趣为特定目的训练 **LoRA**，例如 **dabbing**、**middle finger** 和 **30s cartoon**。
- **用于医疗辅助的 LLM 尚不可靠**：几位用户对目前状态下使用 **LLM** 进行医疗辅助表示怀疑。
- **将 FLUX Dev LoRA 转换为 FP8**：一位用户询问是否可以将他们的 **FLUX Dev LoRA** 转换为 **FP8**，或者在 Replicate 上使用 **FP8 LoRA** 训练器。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/laion_ai/status/1824814210758459548">来自 LAION (@laion_ai) 的推文</a>：FLUX Dev 可以生成同一个（虚构）人物的 3x3 照片网格 --> 可以在此类照片上训练 LoRA，从而为各种虚构人物建立一致的角色 LoRA 库...</li><li><a href="https://goldhire.app.loxo.co/job/MjM4NDcta2hrdWh2bmkxMng4ZnZiMA==?t=1723412813305">创始 AI 工程师 - GoldHire</a>：未找到描述</li><li><a href="https://civitai.com/models/290836/multiple-views-sdxl>">Multiple Views (SDXL) - v1.0 | Stable Diffusion LoRA | Civitai</a>：请我喝杯咖啡：https://ko-fi.com/futureflix 触发词：multiple views 提示词示例：multiple views, a black man, pants, street urban, neckla...
</li>
</ul>

</div>
  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1274172546987593768)** (12 条消息🔥): 

> - `JPEG-LM`
> - `Image/Video Generation with LLMs`
> - `Autoregressive LLMs`
> - `SIREN`
> - `Neural Graphics Primitives` 


- **JPEG-LM：一种新颖的图像生成方法**：一篇新的研究论文提出在自回归 LLM 架构中，使用标准编解码器（如 JPEG, AVC/H.264）将图像和视频建模为压缩文件。
   - 这种方法消除了对原始像素值建模或矢量量化（vector quantization）的需求，使过程更加高效。
- **JPEG-LM vs. SIREN：巨头之战？**：一位用户开玩笑地声称，尽管承认 NVIDIA 2022 年的 Neural Graphics Primitives 论文已显著推进了该领域，但他使用 33kB 的复值神经网络（complex-valued neural network）超越了 2020 年的 SIREN 架构。
   - 该用户强调了使用 MS-SSIM 作为图像质量评估指标的重要性，而不仅仅是 MSE 和 MAE。
- **7B 参数用于低质量生成？**：讨论中承认，使用 7B 参数进行此类低质量图像生成可能被认为是过度的。
   - 然而，这种方法的新颖性和潜力仍然受到赞赏，为未来的研究打开了新大门。



**提到的链接**：<a href="https://arxiv.org/abs/2408.08459">JPEG-LM: LLMs as Image Generators with Canonical Codec Representations</a>：由于自回归 LLM 架构的通用性以及易于集成到多模态系统中的潜力，最近图像和视频生成领域的工作一直在采用该架构。应用自回归...的关键在于...

  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1274108307845677227)** (5 条消息): 

> - `Workflows`
> - `RAG`
> - `Agents`
> - `BeyondLLM`
> - `JSONalyze Query Engine` 


- **Workflows 实战**：Rajib Deb 的一段视频展示了 Workflows 的特性，包括装饰器（decorators）、用于控制流的类型、事件驱动的过程链，以及用于复杂任务的自定义事件和步骤。
   - 该视频深入探讨了 Workflows 的核心特性，演示了它们如何通过更结构化的方法来构建复杂的应用程序。
- **RAG 与 Agent 模板**：提供了 3 篇 RAG 和 Agent 论文的参考实现，为从零开始构建应用或使用预建模板提供了快速启动方案。
   - 这些模板利用 LlamaIndex 框架，强调了用于高级 RAG 和 Agent 应用的事件驱动技术。
- **使用 Claude 3.5 构建 Agentic RAG**：Richmond Lake 的教程指导用户使用 Claude 3.5、MongoDB 和 LlamaIndex 构建 Agentic 知识助手。
   - 该教程重点介绍了在现有 RAG 流水线之上构建 Agentic 知识助手，利用了工具选择、任务分解和高级 RAG 技术。
- **用于高级 RAG 的 BeyondLLM**：由 AIPlanetHub 开发的 BeyondLLM 在 LlamaIndex 之上提供了抽象，使用户仅需 5-7 行代码即可构建具有评估、可观测性和高级 RAG 功能的流水线。
   - 这些高级 RAG 特性包括查询重写、向量搜索和文档摘要，简化了复杂 RAG 应用的开发。
- **将 JSONalyze Query Engine 作为 Workflow**：RavitheJads 将 JSONalyze Query Engine 重构为一个 Workflow，展示了将 JSON API 响应转换为 SQLite 表以及将查询转换为 SQL 的逐步过程。
   - 这一 Workflow 演示突显了 Workflows 的多功能性，能够使用结构化、模块化的方法实现高效的数据操作和转换。



**提到的链接**：<a href="https://t.co/VybhvUgAbL">无标题</a>：未找到描述

  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1274222201461018756)** (27 条消息🔥): 

> - `LlamaIndex 的 Web Scrapers`
> - `RouterQueryEngine vs Agents`
> - `LlamaIndex Workflow`
> - `Batching APIs`
> - `LlamaIndex CSV 分析` 


- **LlamaIndex 的 Web Scraper 推荐**：一名成员询问是否有适合与 LlamaIndex 技术栈配合使用的 Web Scrapers 推荐。
   - 另一名成员推荐了 FireCrawl，并分享了一个展示更复杂的 LlamaIndex Workflow 实现的 YouTube 视频。
- **LlamaIndex 中 RouterQueryEngine 与 Agents 的对比**：一名成员询问了 LlamaIndex 中 RouterQueryEngine 和 Agents 之间的区别，特别是在路由（routing）和函数调用（function calling）方面。
   - 另一名成员解释说，RouterQueryEngine 的行为类似于硬编码的 Agent，而 Agents 则更加灵活且通用。
- **开源模型的 Batching APIs**：一名成员讨论了像 OpenAI 和 Google 这样的大公司如何为他们的模型推出了 Batching APIs，但这些 API 缺乏处理保证、SLA 和重试机制。
   - 他们分享了一篇关于如何为开源模型获取类似 OpenAI 的 Batching API 的博客文章。
- **LlamaIndex CSV 分析的局限性**：一名成员在利用 LlamaIndex 分析 CSV 文件时遇到困难，结果不准确。
   - 另一名成员解释说，CSV 并不适合向量索引（vector indexes），并建议使用数据库或 Pandas query engine 以获得更好的结果。
- **在 Neo4j 中存储 DocumentSummaryIndex**：一名成员询问是否可以将 DocumentSummaryIndex 存储在 Neo4j 中，他们已经将 Neo4j 用于 PropertyGraphIndex。
   - 另一名成员回答说，虽然 Neo4j 可以用作向量存储（vector store），但它不适合通用的键值存储（key-value storage），这使得在 Neo4j 中存储 DocumentSummaryIndex 具有挑战性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://blog.cuminai.com/how-to-get-a-batching-api-like-openai-for-open-source-models-824529788a49">How to Get a Batching API Like OpenAI for Open-Source Models</a>：在 AI 世界中，高效的处理和成本管理至关重要。实现这一目标的一个强大方法是批处理（batching），它……</li><li><a href="https://youtu.be/LloUNBD9fsI">LlamaIndex Workflow | Global context</a>：在此录制视频中，我展示了一个通过 LlamaIndex Workflow 代码实现的更复杂的 Workflow：https://github.com/rajib76/llamaindex/blob/main/examples/07_l...
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1274774003478691951)** (2 条消息): 

> - `LLMs`
> - `LLM 局限性`
> - `LLMs 作为助手`
> - `Tokenization`
> - `Sampling` 


- **LLMs 作为个人助手**：LLMs 是 AI 驱动的助手，可以生成文本、翻译语言、编写各种创意内容，并以信息丰富的方式回答您的问题。
   - 它们不局限于特定任务，而是可以适应各种输入和提示。可以将它们视为一种可用于广泛应用的灵活工具。
- **LLMs 深度探索**：该博客文章从高层抽象开始，将 LLMs 视为个人助手，然后深入探讨 Tokenization、Sampling 和 Embedding 等核心概念。
   - 这种方法旨在让更广泛的受众更容易理解复杂的 LLM 世界。
- **LLM 的能力与局限性**：博客文章承认 LLMs 仍处于开发阶段并存在局限性，例如无法计算 "strawberry" 中 R 的数量以及反转字符串 "copenhagen"。
   - 这种诚实的评估有助于读者了解 LLM 技术的现状以及需要进一步研究的领域。
- **知识图谱：一个强大的工具**：知识图谱（Knowledge graphs）提供了一种结构化且直观的方式来捕获隐藏在数据中的复杂关系。
   - 这种方法可以更好地组织和理解信息，从而开发出真正的智能应用。
- **结合知识图谱与生成式 AI**：博客文章探讨了将知识图谱与生成式 AI（Generative AI）结合以创建强大的智能应用的潜力。
   - 这种协同作用利用了两种技术的优势，开启了新的可能性并推动了 AI 领域的发展。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://amgadhasan.substack.com/p/explaining-how-llms-work-in-7-levels">Explaining how LLMs work in 7 levels of abstraction</a>：概述</li><li><a href="https://medium.com/ai-artistry/knowledge-graphs-and-generative-ai-powering-intelligent-applications-with-amazon-neptune-and-f734d96c0fa0">Knowledge Graphs and Generative AI: Powering Intelligent Applications with Amazon Neptune and…</a>：Ankush k Singal
</li>
</ul>

</div>

### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1274367530051309619)** (5 messages): 

> - `LLM Hosting`
> - `HF Spaces`
> - `Modal`
> - `Jarvis Labs`
> - `vLLM` 


- **HF Spaces 的限制**：一位成员表示在 HF Spaces 上托管自己的 LLM 存在困难，理由是 ZeroGPU 不支持 vLLM。
- **Modal 和 FastHTML**：另一位成员提到他们曾使用 Modal 托管 LLM，但目前正尝试使用 FastHTML 并正在寻找设置指南。
- **使用 Jarvis Labs 进行微调**：该成员提到仅使用过 Jarvis Labs 来微调 LLM。


  

---



### **Alignment Lab AI ▷ #[general](https://discord.com/channels/1087862276448595968/1095458248712265841/1274663309911785523)** (1 messages): 

> - `Batching APIs`
> - `OpenAI`
> - `CuminAI`
> - `Small Language Models (SLMs)`
> - `Large Language Models (LLMs)` 


- **OpenAI 和 Google 推出更便宜的 Batching APIs**：OpenAI 和 Google 最近为其部分模型推出了 Batching APIs，与常规请求相比，成本降低了 50%。
   - 然而，这些 API 目前缺乏处理保证、服务水平协议 (SLAs) 和重试机制。
- **CuminAI：针对开源模型的 Batching APIs**：CuminAI 提供了一种为开源模型创建 Batching APIs 的解决方案，类似于 OpenAI 提供的方案。
   - 在[此处](https://blog.cuminai.com/how-to-get-a-batching-api-like-openai-for-open-source-models-824529788a49)查看他们的分步指南“如何为开源模型获取类似 OpenAI 的 Batching API”。
- **Small Language Models：AI 的新超级英雄**：CuminAI 最近的一篇博文强调了 Small Language Models (SLMs) 的潜力，认为在 AI 世界中“大并不总是更好”。
   - 虽然 Large Language Models (LLMs) 占据了该领域的主导地位，但 SLMs 提供了一种更具成本效益且高效的替代方案，特别是对于不需要大量计算能力的任。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://blog.cuminai.com/how-to-get-a-batching-api-like-openai-for-open-source-models-824529788a49">How to Get a Batching API Like OpenAI for Open-Source Models</a>：在 AI 领域，高效的处理和成本管理至关重要。实现这一目标的一种强大方法是批处理（batching），它……</li><li><a href="https://medium.com/@umesh-cuminai?source=post_page-----824529788a49--------------------------------)[">Umesh – Medium</a>：在 Medium 上阅读 Umesh 的文章。理解 AI :)，CuminAI 创始人。每天，Umesh 和成千上万的其他声音在 Medium 上阅读、写作并分享重要的故事。</li><li><a href="https://blog.cuminai.com/?source=post_page-----824529788a49--------------------------------)">Cumin AI</a>：将任何 Huggingface 模型转换为稳健的 Batch API，用于处理企业的离线 AI 工作负载。</li><li><a href="https://medium.com/@harshal-cuminai?source=collection_home---------0----------------------------)">Harshal Priyadarshi – Medium</a>：在 Medium 上阅读 Harshal Priyadarshi 的文章。Cumin AI 创始人。每天，Harshal Priyadarshi 和成千上万的其他声音在 Medium 上阅读、写作并分享重要的故事。
</li>
</ul>

</div>
  

---

### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1275138618163462145)** (1 条消息): 

> - `Llamafile 更新`
> - `Rise25 上的 Mozilla AI 社区`
> - `ML 论文研讨会` 


- **Llamafile 更新：语音转文字、图像生成、性能提升**：**Llamafile** 发布了令人兴奋的新功能，包括 **Speech to Text 命令**、**图像生成**，以及其 HTTP server embeddings 的 **3 倍性能提升**。
   - 你可以在这里查看来自 Justine 的 [完整更新](https://discord.com/channels/1089876418936180786/1262961704602570832/1275110073584320576)。
- **Rise25 上的 Mozilla AI 社区**：Mozilla AI 正在表彰那些致力于塑造一个负责任、值得信赖、包容且以人类尊严为中心的 AI 未来的社区成员。
   - 多位成员参加了此次活动，包括 <@631210549170012166>、<@1046834222922465314>、<@200272755520700416> 和 <@1083203408367984751>。
- **ML 论文研讨会：Communicative Agents & Extended Mind Transformers**：加入由主持人 <@718891366402490439> 主持的关于前沿 Machine Learning 研究的深度会议，重点讨论 **Communicative Agents** 和 **Extended Mind Transformers**。
   - 预约参加这些引发思考的讨论，并分别与作者 <@878366123458977893> 和 <@985920344856596490> 进行深入探讨：[Communicative Agents](https://discord.com/events/1089876418936180786/1266733035231903795) 和 [Extended Mind Transformers](https://discord.com/events/1089876418936180786/1267946366680694817)。 


  

---



---



---



{% else %}


> 为了适配电子邮件，完整的频道明细已被截断。
> 
> 如果你想查看完整明细，请访问此邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}