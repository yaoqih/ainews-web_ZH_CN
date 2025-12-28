---
companies:
- cognition
- poolside
- codeium
- magic
- google-deepmind
- nvidia
- google-cloud
date: '2024-08-30T00:01:06.332310Z'
description: '**“代码 + AI”** 被强调为 AI 工程中的一种关键模态，突出了其在提高生产力和可验证性方面的优势。近期主要的融资活动包括：**Cognition
  AI 融资 1.75 亿美元**、**Poolside 融资 4 亿美元**、**Codeium AI 融资 1.5 亿美元**以及 **Magic 融资 3.2
  亿美元**。


  Magic 发布了其 **LTM-2** 模型，该模型拥有 **1 亿 token 的超长上下文窗口**。据称，在序列维度算法上，该模型比 **Llama 3.1
  405B** 便宜约 **1000 倍**，且内存需求大幅降低。Magic 的技术栈完全从零开始构建，采用定制的 CUDA 且不依赖开源基础；他们与 **Google
  Cloud** 达成合作，由 **NVIDIA H100** 和 **GB200 GPU** 提供算力支持，目标是扩展至数万个 GPU。


  此外，Google DeepMind 披露了 **Gemini Advanced** 的更新，推出了可定制的专家级“Gems”。神经游戏引擎（如 **GameNGen**）已实现在扩散模型中运行《毁灭战士》（DOOM），该模型基于
  **9 亿帧** 数据训练而成。内容还引用了 Rohan Paul 关于 **LLM 量化** 的研究。'
id: 6c4b56a6-b583-4f56-b52b-6851094b7ada
models:
- ltm-2
- llama-3-1-405b
- gemini-advanced
original_slug: ainews-code
people:
- nat-friedman
- ben-chess
- rohan-paul
title: AI 编程之夏：融资 16 亿美元，仅 1 款可用产品。
topics:
- long-context
- model-efficiency
- custom-hardware
- cuda
- training-stack
- gpu-scaling
- neural-world-models
- diffusion-models
- quantization
---

<!-- buttondown-editor-mode: plaintext -->**代码 + AI 就足够了。**

> 2024年8月28日至8月29日的 AI 新闻。我们为你检查了 7 个 subreddits、[400 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 30 个 Discord（213 个频道和 2980 条消息）。预计节省阅读时间（以 200wpm 计算）：338 分钟。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

[AI 工程师的兴起](https://www.latent.space/p/ai-engineer) 中的核心论点之一是，在即将出现的众多模态中，代码是“首位平等者”（first among equals）。除了明显的良性循环（编码更快 -> 训练更快 -> 编码更快）之外，它还具有以下优良特性：1) 面向内部（因此错误责任较低但非零），2) 提高开发者生产力（最昂贵的人力成本之一），3) 可验证/自纠正（在 [Let's Verify Step by Step](https://www.latent.space/p/iclr-2024-benchmarks-agents) 的意义上）。

这个“代码之夏”拉开序幕的标志是：

- [**Cognition (Devin) 融资 1.75 亿美元**](https://www.maginative.com/article/cognition-ai-raises-175m-at-2b-valuation-one-month-after-series-a/)（仍处于严格限制的等候名单中）（其 [World's Fair 演讲见此](https://www.youtube.com/watch?v=T7NWjoD_OuY)）
- [**Poolside 融资 4 亿美元**](https://techcrunch.com/2024/06/20/poolside-raising-400m-at-a-2b-valuation-for-supercharged-coding-copilot/)（目前[大部分](https://x.com/poolsideai/status/1738669662467178581)仍处于隐身模式）

今天，我们看到：

- [**Codeium AI 融资 1.5 亿美元**](https://techcrunch.com/2024/08/29/github-copilot-competitor-codeium-raises-150m-at-a-1-25b-valuation/?guccounter=1)，这是在其 1 月份 6500 万美元融资基础上的又一轮（其 [World's Fair 演讲见此](https://www.youtube.com/watch?v=DuZXbinJ4Uc)）
- [**Magic 融资 3.2 亿美元**](https://techcrunch.com/2024/08/29/generative-ai-coding-startup-magic-lands-320m-investment-from-eric-schmidt-atlassian-and-others/)，这是在其 2 月份 1 亿美元融资基础上的又一轮，并发布了 [LTM-2](https://magic.dev/blog/100m-token-context-windows)，正式确认了传闻中的 1 亿 token 上下文模型，尽管目前仍处于隐身模式。

虽然 [Codeium](https://codeium.com/) 是这四家公司中目前唯一可以实际使用的产品，但 Magic 的公告更值得关注，因为其展示了极具前景的长上下文利用能力（由 [HashHop](https://github.com/magicproduct/hash-hop) 驱动）以及 Nat Friedman 在上一轮融资中透露的效率细节：

> 对于每个解码的 token，LTM-2-mini 的序列维度算法比 Llama 3.1 405B 在 1 亿 token 上下文窗口下的 Attention 机制便宜约 1000 倍。内存需求的对比则更为巨大——运行具有 1 亿 token 上下文的 Llama 3.1 405B，仅存储单个 1 亿 token 的 KV cache 就需要每用户 638 张 H100。相比之下，LTM 在相同上下文下，每用户仅需单个 H100 HBM 的一小部分。

 
![image.png](https://assets.buttondown.email/images/91a9b5ec-5099-472c-9aad-91c506e60418.png?w=960&fit=max)
 

这是通过一个完全从零开始编写的技术栈实现的：

> 为了训练和提供 1 亿 token 上下文模型，我们需要从头编写整个训练和推理栈（没有 torch autograd，大量自定义 CUDA，没有开源基础），并针对如何稳定训练模型进行了一次又一次的实验。

他们还宣布了与 Google Cloud 的合作伙伴关系：

> Magic-G4 由 NVIDIA H100 Tensor Core GPU 驱动，Magic-G5 由 NVIDIA GB200 NVL72 驱动，并具备随着时间推移扩展到数万个 Blackwell GPU 的能力。

他们提到目前拥有 8000 张 H100，但在前 OpenAI 超级计算负责人 Ben Chess 的领导下，“随着时间的推移，我们将扩展到数万张 GB200”。

他们的下一个前沿领域是 **推理时计算（inference-time compute）**：

> **想象一下，如果你能为一个问题花费 100 美元和 10 分钟，并能可靠地获得一个针对整个功能的优秀 Pull Request。这就是我们的目标。**




---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 综述

> 所有综述均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型进展与应用**

- **Gemini 更新**：Google DeepMind 宣布了 Gemini Advanced 的新功能，包括可自定义的、作为领域专家的 "Gems"，以及针对不同场景的预设 Gems。[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1828855383131074997) 强调了创建这些定制版 Gemini 并与之对话的能力。

- **神经游戏引擎**：[@DrJimFan](https://twitter.com/DrJimFan/status/1828813716810539417) 讨论了 GameNGen，这是一个能够在 Diffusion 模型中纯粹运行 DOOM 的神经世界模型。他指出该模型在 0.9B 帧数据上进行训练，这是一个巨大的数据量，几乎占到了训练 Stable Diffusion v1 所用数据集的 40%。

- **LLM 量化**：Rohan Paul 分享了关于 AutoRound 的信息，这是来自 Intel Neural Compressor 团队的一个用于 LLM 高级量化的库。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1828879830575919340) 指出，它对热门模型的压缩接近无损，并能与最近的量化方法相媲美。

- **AI 安全与对齐**：François Chollet [@fchollet](https://twitter.com/fchollet/status/1828897857077993895) 强调了对选举相关帖子中 AI 生成内容泛滥的担忧，估计有相当一部分（按数量计约 80%，按曝光量计约 30%）并非来自真实人类。

**AI 基础设施与性能**

- **推理速度**：[@StasBekman](https://twitter.com/StasBekman/status/1828844048876220438) 建议，对于在线推理，每秒每用户 20 个 Token 可能就足够了，这样可以在相同硬件上处理更多并发请求。

- **硬件进展**：David Holz [@DavidSHolz](https://twitter.com/DavidSHolz/status/1828839760976326800) 提到在 Midjourney 组建了一个新的硬件团队，预示着 AI 专用硬件的潜在发展。

- **模型对比**：关于模型性能的讨论包括了 Gemini 与 GPT 模型的对比。[@bindureddy](https://twitter.com/bindureddy/status/1828823839045984327) 指出，Gemini 的最新实验版本虽然有所进步，但仍落后于其他模型。

**AI 应用与研究**

- **多模态模型**：Meta FAIR 推出了 Transfusion，这是一种将 Next Token Prediction 与 Diffusion 相结合的模型，用于在混合模态序列上训练单个 Transformer。[@AIatMeta](https://twitter.com/AIatMeta/status/1828836885176967327) 分享称其扩展性优于传统方法。

- **RAG 与 Agentic AI**：各种讨论集中在检索增强生成 (RAG) 和 Agentic AI 系统。[@omarsar0](https://twitter.com/omarsar0/status/1828838209461043455) 分享了关于使用多 Agent 架构进行时间序列分析的 Agentic RAG 框架的信息。

- **AI 在法律与商业中的应用**：审计公司 Johnson Lambert 报告称，通过在 Amazon Bedrock 上使用 Cohere Command，审计效率提升了 50%，此消息由 [@cohere](https://twitter.com/cohere/status/1828760139500794079) 分享。

**AI 开发实践与工具**

- **MLOps 与实验追踪**：[@svpino](https://twitter.com/svpino/status/1828764720083423480) 强调了机器学习系统中可复现性、调试和监控的重要性，并推荐使用 Comet 等工具进行实验追踪和监控。

- **开源工具**：多个开源工具受到关注，包括 Kotaemon，这是一个用于文档对话的可定制 RAG UI，由 [@_akhaliq](https://twitter.com/_akhaliq/status/1828892696519553309) 分享。

**AI 伦理与监管**

- **选民欺诈讨论**：Yann LeCun [@ylecun](https://twitter.com/ylecun/status/1828704521054261637) 批评了关于非公民投票的言论，强调了对民主机构信任的重要性。

- **AI 监管**：提到了围绕 AI 监管的讨论，包括加州的 SB1047 法案，突显了关于 AI 安全和治理的持续辩论。


---

# AI Reddit 综述

## /r/LocalLlama 回顾

**主题 1：创新的 Local LLM 用户界面**

- **又一个 Local LLM UI，但我保证它与众不同！** ([分数: 170, 评论: 50](https://reddit.com//r/LocalLLaMA/comments/1f3ozoz/yet_another_local_llm_ui_but_i_promise_its/))：该帖子介绍了一个新的 **Local LLM UI** 项目，它作为一个 **PWA** (Progressive Web App) 开发，专注于创建流畅、熟悉的用户界面。开发者在 **2023** 年初被裁员，他强调了该项目的核心功能，包括用于离线交互和跨设备兼容性的**推送通知**，并计划实现类似 **Character.ai** 的人格交互体验。该项目已在 **GitHub** 上发布，名为 **"suaveui"**，作者正在寻求反馈、GitHub stars 以及潜在的工作机会。
  - 用户称赞了该 **UI 的简洁设计**，将其比作与真人发消息。开发者计划增加更多**受流行聊天应用启发的皮肤**，并实现内置安全隧道的**一键运行体验**。
  - 几位用户要求提供**更简单的安装方法**，包括 **Docker/docker-compose** 支持和更详细的教程。开发者承认了这些需求，并承诺在未来几天内进行改进。
  - 关于**兼容性**的讨论显示，该项目计划支持 **OpenAI 兼容的端点**和各种 LLM 服务器。受 **Character.ai 通话功能**的启发，开发者还表达了实现**语音通话支持**的兴趣。


**主题 2：大语言模型能力的进步**

- **[我这个非常简单的提示词难倒了很多 LLM。“我的猫名叫 dog，我的狗名叫 tiger，我的老虎名叫 cat。我的宠物有什么不寻常之处？”](https://i.redd.it/td68rmnw5dld1.png)** ([分数: 79, 评论: 89](https://reddit.com//r/LocalLLaMA/comments/1f34tq4/my_very_simple_prompt_that_has_defeated_a_lot_of/))：该帖子提出了一个**简单的提示词**，旨在挑战 **LLM**。该提示词描述了一个场景：作者的宠物名字通常与其他动物相关联——**猫名叫 dog**，**狗名叫 tiger**，**老虎名叫 cat**，并询问这种安排有什么不寻常之处。
  - **LLaMA 3.1 405b** 和 **Gemini 1.5 Pro** 被认为是表现最好的模型，它们不仅识别出了不寻常的命名方案，还指出了养一只**老虎**作为宠物的古怪之处。LLaMA 的回答因其拟人化的语气和对养虎行为的随口询问而特别受到称赞。
  - 讨论强调了不同 **LLM** 的不同处理方式，有些模型仅关注循环命名，而另一些则质疑养老虎的合法性和实用性。**Claude 3.5** 因其直接的怀疑态度而受到关注，它表示 *“你声称拥有一只宠物老虎”*。
  - 用户辩论了不同 AI 回答的优劣，有些人更喜欢随意的语气，而另一些人则欣赏直接的怀疑。帖子里还包括了关于老虎坐在充气沙发上的 **AI 生成图像**的幽默交流，并对其不切实际的方面进行了评论。


**主题 3：评估 AI 智能和推理能力的挑战**


- **关于确定 LLM 智能的“陷阱”测试** ([分数: 112, 评论: 73](https://reddit.com//r/LocalLLaMA/comments/1f3v0ld/regarding_gotcha_tests_to_determine_llm/))：该帖子批评了用于测试 LLM 智能的**“陷阱 (gotcha)”测试**，特别是针对涉及命名奇特的宠物（包括一只老虎）的测试。作者认为此类测试是**有缺陷的**，并且**误用了 LLM**。他证明了如果给予适当的提示，即使是 **9B 参数模型**也能正确识别出养老虎是最不寻常的一点。一个经过**修改的示例**（使用了更精确的提示词）显示，大多数受测模型（包括 **Gemma 2B**）都能正确识别出不寻常之处，只有少数例外，如 **Yi 模型**和 **Llama 3.0**。
  - 用户批评了这种**“陷阱”测试**，指出它是**衡量智能的缺陷标准**，甚至人类也可能失败。包括**原贴作者 (OP)** 在内的许多人最初也忽略了老虎不寻常这一点，而只关注宠物名字。
  - 该测试被拿来与其他 **LLM 弱点**进行比较，用户分享了更具挑战性的谜题链接，如 **ZebraLogic**。一些人认为 LLM 具备推理能力，并引用了显示其表现与人类相似的基准测试和临床推理测试。
  - 讨论涉及了 LLM 如何生成回复，并辩论了它们是真正的**推理**还是仅仅是**预测**。一些用户指出，要求 LLM 在生成后解释其推理过程可能会导致偏见或幻觉式的解释。

## AI Reddit 全球综述

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 研究与技术**

- **Google DeepMind 的 GameNGen**：一个[神经模型游戏引擎](https://www.reddit.com/r/StableDiffusion/comments/1f34911/diffusion_models_are_realtime_game_engines_by/)，能够在长轨迹上实现与复杂环境的实时交互。它可以在单个 TPU 上以超过 20 FPS 的速度模拟 DOOM，下一帧预测的 PSNR 达到 29.4。

- **用于游戏生成的 Diffusion Models**：[GameNGen 模型](https://www.reddit.com/r/singularity/comments/1f39psd/gamegen_ai_model_is_generating_this_game_doom_in/)在用户游戏时实时生成 DOOM 游戏画面，展示了 AI 生成交互式环境的潜力。

**AI 模型发布与改进**

- **OpenAI 的 GPT-4 迭代**：OpenAI 已经发布了多个版本的 GPT-4，包括 [GPT-4, GPT-4o, GPT4o-mini 和 GPT4o Turbo](https://www.reddit.com/r/singularity/comments/1f3shgd/gpt4/)。目前存在关于未来发布版本和命名惯例的推测。

**AI 对行业和就业的影响**

- **Klarna 的 AI 驱动裁员**：先买后付公司 Klarna [计划裁员 2,000 人](https://www.reddit.com/r/singularity/comments/1f377co/our_chatbots_perform_the_tasks_of_700_people_buy/)，因为他们的 AI 聊天机器人现在执行的任务相当于 700 名人类员工的工作量。

**技术细节与讨论**

- **GameNGen 架构**：该模型使用 [65 帧游戏分辨率作为输入](https://www.reddit.com/r/StableDiffusion/comments/1f34911/diffusion_models_are_realtime_game_engines_by/jkbjg8v/)，并生成最后一帧。它采用了一种加噪技术，以减轻 AI 生成视频中的增量损坏。

- **GPT-4 训练挑战**：讨论中提到了训练 LLM [所需的巨大计算资源](https://www.reddit.com/r/singularity/comments/1f3shgd/gpt4/lkgl5fm/)，包括需要新建发电厂来支持未来几代模型的运行。


---

# AI Discord 综述

> 由 GPT-4o (gpt-4o-2024-05-13) 生成的摘要之摘要的摘要


**1. LLM 进展**

- **LLM 在图像对比方面表现不佳**：LM Studio Discord 的一名成员询问了 LM Studio 对图像格式的支持，但其他人指出，大多数 LLM 并非针对图像对比任务进行训练，它们“看”图像的方式不同。他们建议尝试 **LLaVA 模型**，该模型专为视觉任务设计。
  - @vision_expert 指出：*"LLaVA 模型在视觉任务中表现出了良好的效果，可能非常适合你的需求。"*
- **Gemini 的能力受到质疑**：OpenAI Discord 的一名用户批评了 **Gemini 的 VR 信息**，指出其错误地将 Meta Quest 3 标记为“即将推出”。该用户表达了对 ChatGPT 的偏好，认为 Gemini 是一个“糟糕的 AI”。
  - 其他用户也表示赞同，认为 **Gemini** 需要改进，特别是在**准确性和信息的实时性**方面。


**2. 模型性能优化**

- **降低推理速度**：LM Studio Discord 的一名成员希望针对特定用例人为地降低推理速度。LM Studio 目前不支持此功能，但可以通过加载多个模型来使用 server API 实现类似效果。
  - 这一变通方案引发了关于优化 **server API** 使用以高效处理多个模型的讨论。
- **RAG 与知识图谱：强大的组合**：在 LangChain AI Discord 中，用户强调了 **Retrieval-Augmented Generation (RAG)** 对 AI 应用的好处，使模型无需重新训练即可访问相关数据。他们表示有兴趣将 RAG 与知识图谱结合，探索解决 text-to-SQL 问题的混合方法。
  - **@data_guru** 建议集成**知识图谱**，以增强模型的**语义理解**和准确性。


**3. 微调策略**

- **Prompt Engineering 与 Fine-tuning 的辩论**：在 OpenAI Discord 中，成员们就实现理想写作风格时 **fine-tuning** 和 **prompt engineering** 的优劣展开了热烈讨论。虽然一些人强调了 prompt-by-example 的有效性，但其他人则强调了为 fine-tuning 进行数据准备的重要性。
  - **@model_tuner** 强调 **fine-tuning** 需要一个精心策划的数据集，以避免 overfitting 并确保 generalizability。
- **Unsloth：流线型 Fine-tuning**：Unsloth AI Discord 的一名成员强调了使用 **Unsloth** 对 Llama-3, Mistral, Phi-3 和 Gemma 等 LLM 进行 fine-tuning 的好处，声称它使过程快了 2 倍，内存占用减少了 70%，且保持了准确性。该成员提供了 [Unsloth tutorial](https://github.com/unslothai/unsloth/wiki) 的链接，其中包括将 fine-tuned 模型自动导出到 Ollama 并自动创建 `Modelfile`。
  - 这引发了社区的兴趣，成员们讨论了他们在 **memory optimization** 和 **training efficiency** 方面的经验。


**4. 开源 AI 进展**

- **Daily Bots 发布 AI 开源云**：在 OpenInterpreter Discord 中，**Daily Bots**（一个用于语音、视觉和视频 AI 的低延迟云）已经发布，允许开发者以低至 500ms 的延迟构建与任何 LLM 的 voice-to-voice 交互。该平台提供开源 SDKs，能够混合搭配 AI 模型，并在 Daily 的实时全球基础设施上大规模运行，利用了开源项目 RTVI 和 Pipecat。
  - 这一发布受到了热烈欢迎，**@developer_joe** 指出了其在客户服务及其他领域的 **real-time applications** 潜力。
- **Llama 3 开源采用率激增**：在 Latent Space Discord 中，开源的 **Llama 模型家族** 继续受到关注，在 Hugging Face 上的下载量已超过 3.5 亿次，比去年增长了十倍。Llama 的普及也延伸到了云服务提供商，自 5 月以来 token 使用量增加了一倍多，并在包括 Accenture, AT&T, DoorDash 在内的各行各业得到采用。
  - **@data_scientist** 讨论了这种增长的影响，强调了 **community support** 和 **open-source collaboration** 的重要性。


**5. AI 社区与活动**

- **Perplexity Discord 庆祝 10 万成员**：Perplexity AI Discord 服务器正式达到 **100,000 名成员**！团队对社区的支持和反馈表示感谢，并对未来的增长和演变感到兴奋。
  - 成员们分享了他们最喜欢的 **Perplexity AI features**，并讨论了他们希望看到的潜在 **improvements** 和 **new features**。
- **AI Engineer 见面会与峰会**：在 Latent Space Discord 中，AI Engineer 社区正在扩大！首届伦敦见面会定于 9 月举行，第二届在纽约市举行的 AI Engineer Summit 计划于 12 月举行。有兴趣参加伦敦见面会的人可以在[这里](https://x.com/dctanner/status/1827071893448618453?s=46)找到更多信息，并鼓励纽约峰会的潜在赞助商[取得联系](mailto:info@ai.engineer)。
  - 该公告引起了轰动，成员们对活动中的 **networking opportunities** 和 **collaboration** 表现出浓厚兴趣。

---

# PART 1: High level Discord summaries

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LLM 在图像比较方面表现不佳**：一位成员询问了 LM Studio 对图像格式的支持，但其他人指出，大多数 LLM 并非为图像比较任务而训练，它们“看”图像的方式不同。
   - 他们建议尝试专门为视觉任务设计的 LLaVA 模型。
- **降低推理速度**：一位成员希望针对特定用例人为降低推理速度。
   - LM Studio 目前不支持此功能，但可以通过服务器 API 加载多个模型来实现类似效果。
- **LM Studio 的新 UI 变化**：几位成员询问了 LM Studio 0.3.2 中缺失的“加载/保存模板”功能，该功能以前用于保存不同任务的自定义设置。
   - 他们被告知该功能已不再必要，现在可以在加载模型时按住 ALT 键或在 My Models 视图中更改自定义设置。
- **LM Studio 的 RAG 功能面临问题**：一位成员报告了 LM Studio 的 RAG 功能存在问题，即聊天机器人在文档处理完成后仍继续分析文档，导致难以进行正常对话。
   - 另一位成员报告了下载 LM Studio Windows 安装程序的问题，但通过删除 URL 中的空格解决了该问题。
- **3090 的 PCIE 5.0 x4 模式**：一位用户询问 3090 是否可以安装在 PCIE 5.0 x4 模式下，以及这是否能提供足够的带宽。
   - 另一位用户确认目前的 GPU 几乎用不满 PCIE 4.0，且 5.0 控制器运行温度很高，首批 5.0 SSD 需要主动散热。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini 的能力受到质疑**：一位用户批评了 Gemini 提供的 VR 信息，指出其错误地将 Meta Quest 3 标记为“即将推出”。
   - 该用户表达了对 ChatGPT 的偏好，并得出结论认为 Gemini 是一个“糟糕的 AI”。
- **呼吁个性化 LLM**：一位成员提出了个性化 LLM 的愿景，概述了所需的功能，如可定制的 AI 性格、长期记忆和更像人类的对话。
   - 他们认为这些功能将增强与 AI 互动的意义和影响。
- **应对上下文窗口限制**：用户讨论了上下文窗口的局限性以及在 LLM 中使用 Token 实现长期记忆的高昂成本。
   - 提出的解决方案包括利用 RAG 检索相关历史记录、优化 Token 使用以及开发用于记忆管理的自定义工具。
- **提示工程 (Prompt Engineering) 与微调 (Fine-tuning) 之争**：成员们就实现所需写作风格的微调和提示工程的优缺点展开了热烈讨论。
   - 虽然一些人强调了示例提示 (prompt-by-example) 的有效性，但其他人则强调了微调数据准备的重要性。
- **OpenAI API：成本与替代方案**：对话集中在利用 OpenAI API 的高昂成本上，特别是对于涉及长期记忆和复杂角色的项目。
   - 用户探索了优化策略，并考虑了 Gemini 和 Claude 等替代模型。



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SDXL 背景仍是挑战**：一位用户表示在使用 SDXL 创建良好的背景时遇到困难，经常生成一些不明物体。
   - 该用户正在寻求关于如何克服这一挑战并生成更真实、连贯背景的建议。
- **Lora 制作：特写还是全脸**：一位用户询问制作 Lora 是只需要所需细节（如鼻子）的特写，还是需要包含整张脸。
   - 该用户正在寻求有关 Lora 制作最佳实践的指导，特别是关于训练数据必要范围的问题。
- **ComfyUI 能处理多个角色吗？**：一位用户询问 ComfyUI 是否可以帮助创建具有两个不同角色的图像，且不混淆它们的特征。
   - 该用户试图了解 ComfyUI 是否提供能够生成具有多个不同角色的图像的功能，同时避免不必要的特征融合。
- **正则化 (Regularization) 解释：AI Toolkit**：一位用户在观看了一个创作者使用不带正则化的基础图像的视频后，询问正则化在 AI Toolkit 中是如何工作的。
   - 该用户要求澄清 AI Toolkit 背景下正则化的目的和实现方式。
- **在 2017 年的中端笔记本电脑上运行 SDXL：可行吗？**：一位用户询问在 2017 年的中端 Acer Aspire E 系列笔记本电脑上运行 SDXL 的可行性。
   - 该用户正在寻求有关其旧笔记本电脑的硬件能力是否足以有效运行 SDXL 的信息。

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth: 速度与内存增益**：与 OpenRLHF 相比，Unsloth 使用 4-bit 量化实现了更快的训练速度和更低的 VRAM 占用。
   - 虽然 Unsloth 目前仅支持 4-bit 量化模型进行 finetuning，但他们正在努力增加对 8-bit 和非量化模型的支持，且不会在性能或可复制性上妥协。
- **在 AWS 上使用 Unsloth 进行 Finetuning**：Unsloth 目前没有专门的 AWS finetuning 指南。
   - 不过，一些用户正在使用 Sagemaker 在 AWS 上 finetuning 模型，并且有许多 YouTube 视频和 Google Colab 示例可供参考。
- **调查寻求 ML 模型部署的见解**：发布了一项调查，询问 ML 专业人士关于模型部署的经验，特别关注常见问题和解决方案。
   - 该调查旨在确定部署 Machine Learning 模型时遇到的前三个问题，为该领域专业人士面临的实际障碍提供宝贵的见解。
- **为 Function Calling 进行 Gemma2:2b Fine-tuning**：一位用户寻求关于从 Ollama 对 Gemma2:2b 模型进行 function calling fine-tuning 的指导，使用的是 [XLM Function Calling 60k dataset](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) 和 [提供的 notebook](https://colab.research.google.com/drive/1weTpKOjBZxZJ5PQ-Ql8i6ptAY2x-FWVA?usp=sharing)。
   - 他们不确定如何将数据集格式化为 instruction、input 和 output 格式，特别是关于 'tool use' 列的处理。
- **Unsloth: 精简的 Fine-tuning**：一位成员强调了使用 Unsloth 对 Llama-3、Mistral、Phi-3 和 Gemma 等 LLM 进行 fine-tuning 的优势，声称其速度快 2 倍，内存占用减少 70%，且能保持准确性。
   - 该成员提供了 Unsloth 教程的链接，其中包括将 fine-tuned 模型自动导出到 Ollama 以及自动创建 `Modelfile`。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Discord 庆祝成员突破 10 万**：Perplexity AI Discord 服务器正式达到 **100,000 名成员**！团队对社区的支持和反馈表示感谢，并对未来的增长和演变感到兴奋。
- **Perplexity Pro 会员问题**：多位用户报告了其 Perplexity Pro 会员身份的问题，包括洋红色会员标识和免费 LinkedIn Premium 优惠消失，以及 "Ask Follow-up" 功能的问题。
   - 其他人也遇到了 "Ask Follow-up" 功能的问题，即在 Perplexity 回复中突出显示一行文本时，"Ask Follow-up" 选项消失了。
- **Perplexity AI 准确性担忧**：用户对 Perplexity AI 倾向于将假设呈现为事实（经常出错）表示担忧。
   - 他们分享了线程中的示例，其中 Perplexity AI 错误地提供了有关政府表格和抓取 Google 的信息，展示了在其回复中需要更强大的事实核查和人工审查。
- **穿梭于 AI 模型的迷宫**：用户对选择最佳 AI 模型表示困惑，争论 Claude 3 Opus、Claude 3.5 Sonnet 和 GPT-4o 的优劣。
   - 几位用户指出，某些模型（如 Claude 3 Opus）限制为 50 个问题，用户不确定 Claude 3.5 Sonnet 是否是更好的选择，尽管它也有局限性。
- **Perplexity AI 易用性挑战**：用户强调了 Perplexity AI 平台易用性方面的问题，包括访问保存的线程困难以及 prompt 部分的问题。
   - 一位用户指出 Chrome 扩展程序的描述不准确，错误地声称 Perplexity Pro 使用 GPT-4 和 Claude 2，这可能误导了平台的实际能力。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **LLM 进行 Tokenize，而非处理字母**：一位成员提醒大家，LLM 看到的不是字母，而是 Token —— 一个巨大的词表。
   - 他们以阅读日语中的汉字（Kanji）为例，认为这比阅读英语字母更接近 LLM 的工作方式。
- **关于 Claude 谄媚倾向的辩论**：一位成员询问 LLM 是否有谄媚（Sycophancy）倾向，特别是在推理方面。
   - 另一位成员建议添加 System messages 来帮助解决这一问题，但表示即便如此，这更多是一种“小花招”，而非实用的生产工具。
- **MMLu 对实际应用场景效果不佳**：一位成员指出 MMLu 不是构建实用 LLM 的好基准（Benchmark），因为它与现实世界的用例相关性不强。
   - 他们列举了关于弗洛伊德过时的性理论问题，暗示该基准无法反映用户对 LLM 的真实需求。
- **Cohere For AI 学者计划开放申请**：Cohere For AI 很高兴开启第三届学者计划（Scholars Program）的申请，旨在改变研究的地点、方式和参与者。
   - 该计划旨在帮助研究人员和志同道合的合作者，您可以在 [Cohere For AI Scholars Program 页面](https://cohere.com/blog/cohere-for-ai-scholars-program-2025)找到更多信息。
- **内部工具即将公开**：一位成员分享说，该工具目前托管在公司的管理面板上，但很快将提供公开托管版本。
   - 该工具目前托管在公司的管理面板上，但预计很快会发布公开版本。



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Workflows 教程现已上线**：LlamaIndex 文档中现已提供关于 LlamaIndex Workflows 的全面教程，涵盖了从 Workflows 入门、循环与分支、状态维护到并发流等一系列主题。
   - 教程可以在[这里](https://docs.llamaindex.ai/en/stable/understanding/workflows/stream/)找到。
- **GymNation 利用 LlamaIndex 提升销售**：GymNation 与 LlamaIndex 合作，改善会员体验并推动实际业务成果，实现了数字线索到销售转化率提升 20% 以及数字线索对话率达到 87% 的显著成效。
- **支持 Function Calling 的 LLM 实现流式输出**：一位成员正在寻求一个使用支持 Function Calling 的 LLM 构建 Agent 的示例，要求能够流式传输最终输出，以避免因将完整消息传递到最后一步而导致的延迟。
   - 他们正利用 Workflows 从头开始构建 Agent，并正在寻找解决方案。
- **Workflows：一个复杂逻辑示例**：一位成员分享了一个 Workflow 示例，该示例利用异步生成器来检测 Tool calls 并流式传输输出。
   - 他们还讨论了使用“Final Answer”工具的可能性，该工具可以限制输出 Token，并在被调用时将最终消息传递给最后一步。
- **优化图像 + 文本检索**：一位成员询问了结合图像和文本检索的最佳方法，考虑对两者都使用 CLIP Embeddings，但担心 CLIP 与专门的文本嵌入模型（如 text-embeddings-ada-002）相比，在语义优化方面表现不足。



---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Agency 融资 260 万美元**：Agency 是一家构建 AI Agent 的公司，宣布融资 260 万美元，用于开发“具有代际意义的技术”并将其 AI Agent 变为现实。
   - 该公司的愿景是构建一个 AI Agent 无处不在并成为我们生活一部分的未来，正如其网站 [agen.cy](http://agen.cy) 所强调的那样。
- **AI Engineer 见面会与峰会**：AI Engineer 社区正在扩张！首场伦敦见面会定于 9 月举行，第二届纽约 AI Engineer Summit 计划于 12 月举行。
   - 有兴趣参加伦敦见面会的人可以在[此处](https://x.com/dctanner/status/1827071893448618453?s=46)找到更多信息，欢迎纽约峰会的潜在赞助商[联系我们](mailto:info@ai.engineer)。
- **个人用途的 AI**：DeepMind 研究科学家 Nicholas Carlini 认为，AI 的重点应该从宏大的革命承诺转向其对个人的益处。
   - 他的博客文章《我如何使用 AI》（"How I Use AI" [https://nicholas.carlini.com/writing/2024/how-i-use-ai.html](https://nicholas.carlini.com/writing/2024/how-i-use-ai.html)）详细介绍了他在 AI 工具方面的实际应用，引起了许多读者的共鸣，特别是在 Hacker News 上（[https://news.ycombinator.com/item?id=41150317](https://news.ycombinator.com/item?id=41150317)）。
- **Midjourney 进军硬件领域**：知名的 AI 图像生成平台 Midjourney 正式进入硬件领域。
   - 有兴趣加入其旧金山新团队的人员可以联系 [hardware@midjourney.com](mailto:hardware@midjourney.com)。
- **Llama 3 开源采用率飙升**：开源 Llama 模型系列继续受到关注，在 Hugging Face 上的下载量已超过 3.5 亿次，与去年相比增长了十倍。
   - Llama 的受欢迎程度延伸到了云服务提供商，自 5 月以来 Token 使用量翻了一番多，并在包括 Accenture、AT&T、DoorDash 在内的各行各业得到采用。



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **OpenInterpreter 开发持续进行**：OpenInterpreter 的开发依然活跃，最近在 [OpenInterpreter GitHub 仓库的 main 分支](https://github.com/OpenInterpreter/open-interpreter/commits/main/)上有新的提交。
   - 这意味着该项目仍在不断推进和改进。
- **Auto-run 安全问题**：提醒用户注意在 OpenInterpreter 中使用 `auto_run` 功能的风险。
   - 使用此功能时，仔细监控输出以防止任何潜在问题非常重要。
- **即将举行的 House Party**：计划于下周提前举行一场 House Party，以鼓励更多人参与。
   - 此次活动将是与社区其他成员建立联系并讨论 OpenInterpreter 相关事宜的绝佳机会。
- **终端应用推荐**：一位用户正在寻找适用于 KDE 的推荐终端应用，因为他们目前使用的终端 Konsole 在滚动 GPT-4 文本时会出现花屏。
   - 这个问题可能是由于终端无法处理来自 GPT-4 的大量文本输出造成的。
- **Daily Bots 发布 AI 开源云平台**：Daily Bots 推出了一款用于语音、视觉和视频 AI 的低延迟云平台，允许开发者以低至 500ms 的延迟构建与任何 LLM 的语音交互。
   - 该平台提供开源 SDK，支持混合搭配 AI 模型，并在 Daily 的实时全球基础设施上大规模运行，利用了开源项目 RTVI 和 Pipecat。



---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Macbook Pro 训练速度对比**：一位用户在 128GB 的 Macbook Pro 上成功训练了大模型，但速度明显慢于 RTX 3090，训练速度大约只有后者的一半。
   - 他们正在寻求更具成本效益的训练解决方案，并考虑将降压版 3090 或 AMD 显卡作为昂贵 H100 的替代方案。
- **租用硬件进行训练**：一位用户建议在决定购买之前先租用硬件，特别是对于初学者。
   - 他们建议花费 30 美元租用不同的硬件并尝试训练模型，以确定最佳配置。
- **模型大小与训练速度**：用户正在探索模型大小与训练速度之间的关系。
   - 他们特别感兴趣的是在比较 Nemotron-4-340b-instruct 与 Llama 405 等模型时，训练时间会如何变化。
- **为对话微调 LLM**：一位成员拥有用于长对话的优质模型，但用于训练的数据集都是 “ShareGPT” 类型。
   - 他们希望实现个性化的数据处理，特别是简化星号（*）括起来的内容，例如将 *she smile* 简化为 *smiling*。
- **通过 Instruction 简化内容**：一位成员询问是否可以使用简单的指令来控制微调后的模型，以简化和重写数据。
   - 他们询问了 LlamaForCausalLM 的功能以及是否有更好的替代方案，另一位成员建议只需将 Prompt 配合 system prompt 传递给 Llama 即可。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **使用 SQLDatabaseChain 和 PGVector 进行混合搜索**：一位用户正在使用带有 `pgvector` 的 PostgreSQL 进行 Embedding 存储，并使用 `SQLDatabaseChain` 将查询转换为 SQL，旨在修改 `SQLDatabaseChain` 以搜索向量从而获得更快的响应。
   - 与传统的基于 SQL 的查询相比，这种方法可能会提高搜索速度并提供更高效的结果。
- **RAG 与知识图谱：强大的组合**：用户强调了检索增强生成 (RAG) 对 AI 应用的好处，使模型无需重新训练即可访问相关数据。
   - 他们表示有兴趣将 RAG 与知识图谱结合，为他们的 text-to-SQL 问题探索一种混合方法，从而可能提高模型的理解力和准确性。
- **为多数据库查询构建自适应 Prompt**：由于不同的 Schema 要求，用户在为不同的 SQL 数据库创建最佳 Prompt 时面临挑战，导致性能问题和模板冗余。
   - 他们正在寻求创建能够适应多个数据库且不损害性能的 Prompt 解决方案，从而可能提高效率并缩短开发时间。
- **解决 Docker 中 OllamaLLM 连接被拒绝的问题**：一位用户在尝试在 Docker 容器中调用 `OllamaLLM` 时遇到了连接被拒绝的错误，尽管与 Ollama 容器的通信是成功的。
   - 建议使用 `langchain_community.llms.ollama` 包作为权宜之计，这可能会解决该问题，并凸显了 `langchain_ollama` 包中潜在的 Bug。
- **探索 LangChain v2.0 中函数调用的流式传输 (Streaming)**：用户询问了在 2.0 版本中将 LangChain 函数调用与流式传输结合使用的可能性。
   - 虽然没有提供直接答案，但目前看来该功能尚不可用，这暗示了 LangChain 未来开发的一个潜在领域。

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 需要你的帮助**：Torchtune 团队正在寻求社区帮助，通过完成一些小型任务来为他们的仓库做出贡献。在 [GitHub issues 页面](https://github.com/pytorch/torchtune/issues?q=sort%3Aupdated-desc+is%3Aissue+is%3Aopen+label%3A%22community+help+wanted%22)上标有 "community help wanted" 标签的 issue 均可参与。
   - 他们也可以通过 Discord 为贡献者提供协助。
- **QLoRA 显存问题**：一位成员报告在尝试使用 4x A6000 显卡对 **Llama 3.1 70B** 进行 **QLoRA** 训练时遇到了显存溢出（out-of-memory）错误。
   - 另一位成员质疑这是否为预期行为，认为这些硬件对于 **QLoRA** 应该是足够的，并建议提交一个带有可复现示例的 [GitHub issue](https://github.com/pytorch/torchtune/issues) 以进行排查。
- **确认 Torchtune 与 PyTorch 2.4 的兼容性**：一位成员询问了 **Torchtune** 与 **PyTorch 2.4** 的兼容性，并得到了可以正常工作的确认。
- **Fusion Models RFC 讨论**：一位成员质疑在 `setup_caches` 函数中处理 decoder-only 的 max_seq_len 是否会导致问题，特别是对于 `CrossAttentionLayer` 和 `FusionLayer`。
   - 另一位成员表示赞同，并提议探索一种能有效处理该问题的工具类（utility）。
- **Flamingo 模型的特殊推理**：对话探讨了 Flamingo 模型对混合序列长度的使用，特别是其融合层（fusion layers），这需要专门的 `setup_caches` 处理方式。
   - 准确的缓存位置追踪（cache position tracking）的必要性得到了认可，并指出 Flamingo PR 与包含更新 `setup_caches` 的 Batched Inference PR 之间存在潜在重叠。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **LinkedIn 职位申请自动化**：一位成员分享了一个 [GitHub 仓库](https://github.com/feder-cr/linkedIn_auto_jobs_applier_with_AI)，该仓库利用 [Agent Zero](https://link.to/agent-zero) 创建新的流水线，自动申请 LinkedIn 上的职位。
   - 该仓库旨在利用 **AIHawk** 实现职位申请的个性化，从而提高流程效率。
- **生成式奖励模型 (GenRM) 论文探讨**：一篇新论文提出了 **Generative Reward Models (GenRM)**，它利用 next-token prediction 目标来训练验证器（verifiers），从而实现与指令微调（instruction tuning）、思维链（chain-of-thought）推理的无缝集成，并通过多数投票（majority voting）利用额外的推理时间计算量来增强验证效果。
   - 论文认为 GenRM 可以克服传统判别式验证器无法利用预训练 LLM 文本生成能力的局限性，详情请参阅 [论文](https://arxiv.org/abs/2408.15240)。
- **DSPY 优化挑战**：一位成员在利用 **DSPY** 实现其核心目标（抽象化模型、提示词和设置）时感到困难。
   - 他们分享了一个 [YouTube 视频链接](https://www.youtube.com/watch?v=lFXeJHhY3mA) 展示其困惑，并寻求理解 DSPY 优化技术的资源。
- **利用人类回复引导合成数据**：一位成员提出了一种引导合成数据的新方法：通过循环使用各种模型和提示词，利用手写的人类回复来最小化 **KL divergence** 指标。
   - 他们就该方法作为生成与人类回复高度一致的合成数据的可行性寻求反馈。
- **DSPY 优化器对示例顺序的影响**：一位用户询问哪些 **DSPY optimizers** 会改变示例/few-shot 的顺序，哪些不会。
   - 该用户似乎对不同优化器策略对训练数据顺序的影响以及这如何影响模型性能感兴趣。

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Jamba 1.5 依赖问题：PyTorch 23.12-py3**：有用户报告在尝试使用 pytorch:23.12-py3 训练 Jamba 1.5 时遇到了依赖问题。
   - Jamba 1.5 与 Jamba Instruct (1.0) 共享相同的架构和基础模型。
- **Transformers 4.44.0 和 4.44.1 的 Bug**：发现 Transformers 4.44.0 和 4.44.1 版本包含一个 Bug，会阻碍 Jamba 架构的执行。
   - 该 Bug 已在 Jamba 1.5-Mini 的 Hugging Face 模型卡片中记录：[https://huggingface.co/ai21labs/AI21-Jamba-1.5-Mini](https://huggingface.co/ai21labs/AI21-Jamba-1.5-Mini)。
- **Transformers 4.40.0 解决依赖问题**：使用 transformers 4.40.0 成功解决了依赖问题，从而能够成功训练 Jamba 1.5。
   - 在该 Bug 完全解决之前，应使用此版本。
- **Transformers 4.44.2 发布说明**：transformers 4.44.2 的发布说明中提到了对 Jamba 缓存失败的修复，但已确认该修复与影响 Jamba 架构的 Bug 无关。
   - 在 Jamba 的 Bug 得到解决之前，用户应继续使用 transformers 4.40.0。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 针对静态调度进行了优化**：Tinygrad 针对静态调度（Static Scheduling）操作进行了高度优化，在不涉及动态稀疏性或权重选择的任务中实现了显著的性能提升。
   - 对静态调度的关注使 Tinygrad 能够利用编译器优化并执行高效的内存管理。
- **Tinygrad 的 ReduceOp 合并行为**：一位用户询问了 Tinygrad 的 `schedule.py` 文件中大量 `# max one reduceop per kernel` 语句背后的原理，特别是其中一个有时会触发 reduction 的早期实例化（early realization），从而阻碍了它们在 `_recurse_reduceops` 函数中的合并。
   - 一位贡献者解释说，当链式调用 reduction 时（例如 `Tensor.randn(5,5).realize().sum(-1).sum()`），这个问题就会显现，reduction 不会像预期那样合并为单个 sum，PR #6302 解决了这个问题。
- **FUSE_CONV_BW=1：卷积反向传播的未来**：一位贡献者解释说，Tinygrad 中的 `FUSE_CONV_BW=1` 标志目前通过在反向传播中启用高效的卷积融合来解决 reduction 合并问题。
   - 他们还指出，一旦在所有场景下都实现了性能优化，该标志最终将成为默认设置。
- **Tinygrad 文档：你的起点**：一位用户寻求开始学习 Tinygrad 的指导。
   - 多位贡献者建议从官方 Tinygrad 文档开始，这被认为是初学者的宝贵资源。
- **动态稀疏操作的局限性**：虽然 Tinygrad 在静态调度方面表现出色，但在处理动态稀疏性或权重选择时可能会遇到性能限制。
   - 这些类型的操作需要内存管理和计算流的灵活性，而 Tinygrad 目前尚未完全支持。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **排行榜中缺少 Groq**：一位成员询问为什么 **Groq** 不在 [Gorilla LLM](https://discord.com/channels/1111172801899012102/1111353033352294440/1278491184943202335) 的排行榜（或变更日志）中。
   - 回复解释说 **Groq** 尚未添加，团队正在等待他们的 PR，预计将于下周提交。
- **Groq PR 预计下周提交**：一位成员询问为什么 **Groq** 不在 [Gorilla LLM](https://discord.com/channels/1111172801899012102/1111353033352294440/1278491184943202335) 的排行榜（或变更日志）中。
   - 回复解释说 **Groq** 尚未添加，团队正在等待他们的 PR，预计将于下周提交。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **CLIP-AGIQA 提升 AIGI 质量评估**：一篇新论文提出了 CLIP-AGIQA，这是一种利用 CLIP 来提高 AI-Generated Image (AIGI) 质量评估性能的方法。
   - 论文认为，当前模型在应对日益增多且多样化的生成图像类别时面临挑战，而 CLIP 评估自然图像质量的能力可以扩展到 AIGI 领域。
- **AIGI 需要鲁棒的质量评估**：AIGI 在日常生活中的广泛应用凸显了对鲁棒图像质量评估技术的需求。
   - 尽管已有一些现有模型，但论文强调需要更先进的方法来评估这些多样化生成图像的质量。
- **CLIP 在 AIGI 质量评估中展现出潜力**：CLIP 作为一种视觉语言模型，在评估自然图像质量方面已显示出巨大潜力。
   - 论文探索了将 CLIP 应用于生成图像的质量评估，并相信它在该领域同样有效。



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **Nous Hermes 2.5 性能**：[最近在 X 上的一篇帖子](https://x.com/nousresearch/status/1829143753036366325?s=46)讨论了 **Hermes 2.5** 的性能提升，但未给出具体指标。
   - 该帖子链接到了一个 GitHub 仓库 [Hermes 2.5](https://github.com/nousresearch/hermes)，但未提供更多细节。
- **未提供更多细节**：这只是 X 上的单条帖子。
   - 没有进一步的细节或讨论点。



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Common Voice 寻求贡献者**：**Common Voice** 项目是一个用于收集语音数据的开源平台，目标是构建一个既免费又无版权限制的多语言语音片段数据集。
   - 该项目旨在让语音技术服务于所有用户，无论其语言或口音如何。
- **加入 Common Voice 社区**：您可以通过 [Common Voice Matrix 频道](https://app.element.io/?updated=1.11.63#%2Froom%2F#common-voice:mozilla.org)或[论坛](https://discourse.mozilla.org/c/voice/239)加入 **Common Voice** 社区。
   - 如果您需要帮助，可以发送电子邮件至 commonvoice@mozilla.com 联系团队。
- **为 Common Voice 项目做贡献**：有兴趣贡献的人可以在[此处](https://github.com/common-voice/common-voice)找到指南。
   - 在文档看起来过时、令人困惑或不完整的地方，需要帮助提交 Issue。



---


**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该社区长时间没有动态，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该社区长时间没有动态，请告知我们，我们将将其移除。


---


**DiscoResearch Discord** 没有新消息。如果该社区长时间没有动态，请告知我们，我们将将其移除。


---

# 第二部分：各频道的详细摘要与链接


{% if medium == 'web' %}

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1278437323603836981)** (161 条消息🔥🔥): 

> - `LLM image comparison`
> - `LLM vision tasks`
> - `LLM speed`
> - `LLM custom instructions`
> - `LLM RAG` 


- **LLM 可以对比图像吗？**: 一位成员询问 LM Studio 是否会支持图像格式以允许模型对比图像，但另一位成员指出，大多数模型并非为此任务训练，且 LLM “看”图像的方式有所不同。
   - 他们建议尝试 LLaVA 模型，该模型专为视觉任务设计。&#x20;
- **降低推理速度**: 一位成员询问是否可以针对特定用例人为降低推理速度，但经确定 LM Studio 目前不支持此功能。&#x20;
   - 不过，可以通过加载多个模型并使用服务器 API 来实现类似效果。
- **LM Studio 的新 UI 变化**: 几位成员询问了 LM Studio 0.3.2 中缺失的“加载/保存模板”功能，该功能此前用于保存不同任务的自定义设置。
   - 他们被告知该功能已不再必要，现在可以通过在模型加载期间按住 ALT 键或在“My Models”视图中更改自定义设置。
- **LM Studio 的 Bug 和问题**: 一位成员报告了 LM Studio 的 RAG 功能问题，即聊天机器人在文档处理完成后仍继续分析文档，导致难以进行正常对话。&#x20;
   - 另一位成员报告了下载 LM Studio Windows 安装程序的问题，但通过删除 URL 中的空格解决了该问题。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.qualcomm.com/developer/blog/2024/04/big-performance-boost-llama-cpp-chatglm-cpp-with-windows-on-snapdragon">Big Performance Boost for llama.cpp and chatglm.cpp with Windows on Snapdragon</a>: 了解如何在 Windows on Snapdragon 上使用 LLVM-MinGW 和 MSVC 命令构建 llama.cpp 和 chatglm.cpp 以提升性能。</li><li><a href="https://dev.to/mrdjohnson/i-made-a-discord-bot-with-lmstudiojs-4fd6">未找到标题</a>: 未找到描述</li><li><a href="https://tenor.com/view/huh-gif-1807502725802114204">Huh GIF - Huh - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/speakleash/Bielik-11B-v2.2-Instruct">speakleash/Bielik-11B-v2.2-Instruct · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1278453155075592246)** (67 条消息🔥🔥): 

> - `PCIE 5.0`
> - `llama.cpp`
> - `NPU support`
> - `Llama 70b`
> - `Multi-GPU setup` 


- **3090 的 PCIE 5.0 x4 模式**: 一位用户询问 **3090** 是否可以安装在 **PCIE 5.0 x4 模式**下，以及这是否能提供足够的带宽。
   - 另一位用户确认 **目前的 GPU 几乎用不满 PCIE 4.0**，且 **5.0 控制器**发热严重，首批 **5.0 SSD** 甚至需要 **主动散热**。
- **llama.cpp 的 NPU 支持**: 一位成员分享了将 **NPU 支持编译进 llama.cpp** 的方法。
   - 他们提供了关于此方法的 **Qualcomm 博客文章**链接：[https://www.qualcomm.com/developer/blog/2024/04/big-performance-boost-llama-cpp-chatglm-cpp-with-windows-on-snapdragon](https://www.qualcomm.com/developer/blog/2024/04/big-performance-boost-llama-cpp-chatglm-cpp-with-windows-on-snapdragon)。
- **Llama 模型的多 GPU 设置**: 一位用户分享了他们的配置：**6x RTX 4090 GPU**、**Threadripper 64 核 CPU**，以及一个支持 **P2P 访问**的 **补丁版 GPU 驱动**，可实现 **51 GB/s 的直接内存访问**。
   - 他们注意到 **LM Studio** 无法识别这种直接内存访问，而是在每张显卡上复制模型，导致传输到 CPU 的速度仅为 **20 GB/s**。
- **Llama 70b 与多 GPU 设置的挑战**: 一位成员报告了在 **6x RTX 4090 配置**下以 **全精度加载 Llama 70b** 时遇到的问题。
   - 他们在训练过程中遇到了 **CUDA out of memory (显存不足)** 错误，不得不额外购买一块 GPU 并增加 CPU RAM 以避免这些错误。
- **转接线 (Riser Cable) 性能与注意事项**: 用户讨论了各种 **PCIE 4.0 转接线** 的性能及遇到的问题，例如 **降速至 PCIE 3.0** 和报错。
   - 他们分享了不同品牌和型号的链接，强调了高质量转接线的重要性以及使用 **retimers (重定时器)** 来增强信号强度的必要性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://www.qualcomm.com/developer/blog">Developer Blog</a>: Qualcomm 开发者博客让您随时了解我们的技术进展。从 AI、计算、游戏、机器人、IoT 到 Snapdragon 工具，该博客为您提供我们对技术走向的见解...</li><li><a href="https://www.qualcomm.com/developer/blog/2024/04/big-performance-boost-llama-cpp-chatglm-cpp-with-windows-on-snapdragon">Big Performance Boost for llama.cpp and chatglm.cpp with Windows on Snapdragon</a>: 了解如何在 Windows on Snapdragon 上使用 LLVM-MinGW 和 MSVC 命令构建 llama.cpp 和 chatglm.cpp 以提升性能。</li><li><a href="https://forum.level1techs.com/t/help-with-wrx80e-sage-se-render-server/190742">Help with WRX80E-Sage SE Render server</a>: 我有一台 Threadripper Pro CPU 和 Pro WRX80E-Sage SE，正在构建一台渲染服务器。使用了 6x RTX 3090，转接线（Risers）均为 4.0。每个 GPU 都能独立工作。但在以 PCIE 4.0 模式启动时遇到问题...</li><li><a href="https://github.com/tinygrad/open-gpu-kernel-modules/fork">Build software better, together</a>: GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://huggingface.co/nisten/meta-405b-instruct-cpu-optimized-gguf/tree/main">nisten/meta-405b-instruct-cpu-optimized-gguf at main</a>: 未找到描述</li><li><a href="https://www.ebay.com/itm/285822257922?mkcid=16&mkevt=1&mkrid=711-127632-2357-0&ssspo=aNWfrJpXTby&sssrc=2047675&ssuid=jxws3gfsrkg&var=587847942332&widget_ver=artemis&media=COPY">PCIE4.0 x16 Riser Graphics Card GPU Extension Cable 90 Degree for ATX Chassis  | eBay</a>: 未找到描述</li><li><a href="https://www.ebay.com/help/buying/postage-delivery/changing-deliver>>>">Security Measure</a>: 未找到描述</li><li><a href="https://www.ebay.com/itm/285154978206?mkcid=16&mkevt=1&mkrid=711-127632-2357-0&ssspo=aNWfrJpXTby&sssrc=2047675&ssuid=jxws3gfsrkg&var=587026697565&widget_ver=artemis&media=COPY">New 64Gbps PCIe 4.0 X4 90° PCI-E 16X to M2 M.2 for NVME SSD Riser Cable Gen4 /3  | eBay</a>: 未找到描述</li><li><a href="https://www.ebay.com/help/buying/postage-delivery/changing-delivery-address-method/international-purchases-postage->>>">Security Measure</a>: 未找到描述</li><li><a href="https://www.ebay.com/itm/276066182129?mkcid=16&mkevt=1&mkrid=711-127632-2357-0&ssspo=okUSh7S8R3u&sssrc=2047675&ssuid=jxws3gfsrkg&widget_ver=artemis&media=COPY">AMD Ryzen Threadripper Pro 3995WX 64-Core 2.7GHz sWRX8 Processor - Unlocked  | eBay</a>: 未找到描述</li><li><a href="https://signin.ebay.com/ws/eBayISAPI.dll?SignIn&ru=https%3A%2F%2Fwww.ebay.com%2Fitm%2F276066182129%3Fmkcid%3D16%26mkrid%3D711-127632-2357-0%26ssspo%3DokUSh7S8R3u%26sssrc%3D2047675%26ssuid%3Djxws3gfsrkg%26widget_ver%3Dartemis%26media%3DCOPY%26boolp%3D1)">Security Measure</a>: 未找到描述</li><li><a href="https://www.ebay.com/itm/276066182129?mkcid=16&mkevt=1&mkrid=711-127632-2357-0&ssspo=okUSh7S8R3u&sss">AMD Ryzen Threadripper Pro 3995WX 64-Core 2.7GHz sWRX8 Processor - Unlocked  | eBay</a>: 未找到描述</li><li><a href="https://signin.ebay.com/ws/eBayISAPI.dll?SignIn&ru=https%3A%2F%2Fwww.ebay.com%2Fitm%2F276066182129%3Fmkcid%3D16%26mkrid%3D711-127632-2357-0%26ssspo%3DokUSh7S8R3u%26sss%3D%26boolp%3D1)">Security Measure</a>: 未找到描述</li><li><a href="https://www.asrockrack.com/general/productdetail.asp?Model=GENOAD24QM32-2L2T/BCM#Specifications">no title found</a>: 未找到描述</li><li><a href="https://amzn.asia/d/8LsDojv">no title found</a>: 未找到描述</li><li><a href="https://www.fractal-design.com/ridge-riser-card-pcie-4-0/">Information regarding Ridge riser card - PCIe 4.0</a>: 更新：Ridge 4.0 2024年5月2日 – 17:30 CET。Ridge 4.0 包括北美在内的大多数主要地区已完成更新版 4.0 Ridge 的补货。有关兼容性的说明，请参阅...</li><li><a href="https://support.fractal-design.com/a/solutions/articles/4000188965?portalId=4000000494](https://support.fractal-design.com/a/solutions/articles/4000188965?portalId=4000000494">Loading...</a>: 未找到描述</li><li><a href="https://support.fractal-design.com/a/solutions/articles/4000188965?portalid=4000000494")">Loading...</a>: 未找到描述</li><li><a href="https://www.fractal-design.com/contact-us/](https://www.fractal-design.com/contact-us/">Welcome to the Fractal Design Website</a>: Fractal Design 是领先的高端 PC 硬件设计商和制造商，产品包括机箱、散热、电源和配件。</li><li><a href="https://www.fractal-design.com/contact-us/")">Contact us</a>: 产品建议？销售咨询？评测样品？请立即联系我们。</li><li><a href="https://www.fractal-design.com/app/uploads/2019/>>>">Welcome to the Fractal Design Website</a>: Fractal Design 是领先的高端 PC 硬件设计商和制造商，产品包括机箱、散热、电源...</li>

电源和配件。</li><li><a href="https://support.fractal-design.com/support/tickets/new)">未找到标题</a>：未找到描述</li><li><a href="https://support.fractal-design.com/support/home)>>>">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1278437413575852084)** (215 messages🔥🔥): 

> - `Gemini 的能力`
> - `LLM 个性化`
> - `LLM 中的记忆与上下文`
> - `Fine-tuning 与 Prompt Engineering 的对比`
> - `OpenAI API 的使用与成本` 


- **Gemini 的缺点**：一位用户注意到 Gemini 在 VR 信息方面的一个错误，指出它错误地将 Meta Quest 3 标记为“即将推出”。
   - 他们得出结论，认为 Gemini 是一个“糟糕的 AI”，并更倾向于使用 ChatGPT。
- **对个性化 LLM 的需求**：一位成员表达了对个性化 LLM 的渴望，概述了诸如可定制的 AI 性格、长期记忆以及更具人性化的对话等功能。
   - 他们认为这些功能将使对话更有意义且更具影响力。
- **应对 LLM 中的上下文限制**：用户讨论了上下文窗口的限制以及使用 Token 实现长期记忆的高昂成本。
   - 提出的解决方案包括使用 RAG 检索相关历史记录、优化 Token 使用以及构建自定义工具来管理记忆。
- **GPT 大辩论：Fine-tuning 还是 Prompt Engineering**：小组辩论了通过 Fine-tuning 和 Prompt Engineering 实现特定写作风格的优劣。
   - 虽然一些人强调了 Prompt by example 的好处，但另一些人则强调了 Fine-tuning 所需的数据准备工作。
- **OpenAI API：功能强大但价格昂贵**：讨论集中在使用 OpenAI API 的高昂成本上，特别是对于涉及长期记忆和复杂角色的项目。
   - 用户分享了优化策略，并探索了 Gemini 和 Claude 等替代模型。



**提到的链接**：<a href="https://x.com/TheTechOasis1/status/1827394026808418799">来自 Ignacio de Gregorio (@TheTechOasis1) 的推文</a>：http://x.com/i/article/1827379585861709824

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1278563686474055792)** (7 messages): 

> - `LLM 模型性能`
> - `OpenAI 模型局限性`
> - `GPT-4 与 GPT-4o`
> - `Llama 3 与 OpenAI 模型` 


- **Llama 3 表现优于 OpenAI 模型**：用户认为 OpenAI 模型（如 GPT-4）的表现不如预期，并举例说明像 Llama 3 (8B) 这样的小型模型在代码生成和理解等领域提供了更好的结果。
   - 他们用赛马击败法拉利做类比，暗示如果赛马（Llama 3）能跑赢法拉利，那么法拉利（OpenAI 模型）的性能就存在严重问题。
- **OpenAI 模型未针对特定主题进行 Fine-tuning**：用户声称，尽管 OpenAI 模型规模庞大，但它们并未针对特定任务进行专门优化，而不像 DeepSeek Coder 这样（针对编程进行了 Fine-tuning）的模型。
   - 他们认为，与 Llama 3 等专门为某些任务设计的模型相比，这种通用性质导致了 OpenAI 模型中观察到的局限性。
- **GPT-4 与 GPT-4o 性能对比**：用户指出，GPT-4o（GPT-4 的廉价版本）似乎速度最快，但质量也最低，而 GPT-4 提供了更好的推理和更准确的结果。
   - 他们注意到 GPT-4o 经常无法执行特定任务（如浏览），而 GPT-4 通常能更准确地遵循指令。
- **OpenAI 模型性能下降**：另一位用户分享了他们使用 ChatGPT 的经验，表示它似乎已经达到了效率巅峰，并且在过去几周内性能一直在下降。
   - 他们提到由于 GPT-4o 的质量和稳定性下降，他们在几天前取消了 ChatGPT 订阅，并观察到其他用户也有类似的看法。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1278614602497785897)** (2 messages): 

> - `ChatGPT Persona` 


- **让 ChatGPT 听起来更像人类**：一位成员询问如何让 ChatGPT 听起来不那么像 AI，而更像一个真实的人，因为他们正在开发一款玩家与 AI 控制的 Agent 进行对话的游戏。
   - 他们分享了一个名为 [Psychographic Agent Persona Builder](https://chatgpt.com/g/g-bIyZLKTwx-psychographic-agent-persona-builder) 的工具链接，该工具可以帮助为 AI Agent 构建 Persona，并建议在单独的聊天中包含一个 Key 以进行额外的自定义。
- **另一个 ChatGPT Persona 选项**：一位用户征求关于如何在游戏场景中让 ChatGPT 听起来更像人类的建议。
   - 该用户的问题是如何让 ChatGPT 听起来不那么像一个“乐于助人的 AI”，而更像一个人。他们请求协助编写初始 Prompt，以锁定 AI 更像人类的行为模式。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1278614602497785897)** (2 messages): 

> - `ChatGPT persona` 


- **让 ChatGPT 听起来不那么像机器人**：一位用户询问如何让 ChatGPT 听起来不那么像“我是来为您提供帮助的” AI，而更像是一个正在与之交谈的人。
   - 另一位用户建议使用名为 “Psychographic Agent Persona Builder” 的工具（访问地址：[https://chatgpt.com/g/g-bIyZLKTwx-psychographic-agent-persona-builder](https://chatgpt.com/g/g-bIyZLKTwx-psychographic-agent-persona-builder)），为 ChatGPT Agent 创建更具人性化的 Persona。
- **Persona 定义的 Key**：建议使用 Persona 构建工具的用户提到，Persona 定义需要一个 Key，应在单独的聊天中获取。


  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1278429103099346996)** (184 messages🔥🔥): 

> - `SDXL Background Issues`
> - `Lora Creation`
> - `Model Merging`
> - `ComfyUI`
> - `Regularization` 


- **SDXL 背景仍是挑战**：一位用户表示使用 SDXL 创建理想背景存在困难，经常生成一些不明物体。
- **从特写镜头创建 Lora**：一位用户询问创建 Lora 是否只需要所需细节（如鼻子）的特写，还是需要包含整个面部。
- **ComfyUI 能处理多个角色吗？**：一位用户询问 ComfyUI 是否可以帮助创建包含两个不同角色的图像，且不会混淆他们的特征。
- **正则化 (Regularization) 详解**：一位用户在观看了一段创作者使用无正则化基础图像的视频后，询问 AI Toolkit 中的正则化是如何工作的。
- **SDXL 在旧硬件上的可行性**：一位用户咨询在 2017 年的中端 Acer Aspire E 系列笔记本电脑上运行 SDXL 的可行性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.amazon.de/Fantastische-Fabelwesen-Stressabbau-Entspannung-Fantasie-Kreaturen/dp/B0CN5B8WTG/ref=sr_1_1?crid=3IBODT2J8X6H6&dib=eyJ2IjoiMSJ9.-3XggVW3uObjvvXQqObf-g-EWf_V6QDcBkrHerEySuY2P3W0J8JG92mAOXoFt2DWOwZHT1w0m6M4IrDxhUwXVi523Affpx6n5y5TI3Pal5iMGXUuSJEje7x1BSRxDuAhRJqcESyU0awWBpc07xA90cucn7Z_uETG34wev0if1-ON4ICntYnPnlLPGVH6WUk532dqEr89fXftuzS4TrhIrYMCKNik-WVzuMj3aU2Vvr8.d_Vd1P3m4memC-Dd8Agtfsyxu8CgD6J3vjQdJ--SaDo&dib_tag=se&keywords=fabelwesen+malbuch&qid=1724956770&sprefix=Fabelwesen+%2Caps%2C126&sr=8-1">未找到标题</a>: 未找到描述</li><li><a href="https://www.amazon.de/gp/help/customer/display.html/ref=footer_cou/275-2496043-9483305?ie=UTF8&nodeId=505048)">未找到标题</a>: 未找到描述</li><li><a href="https://www.amazon.de/gp/help/customer/display.html/ref=footer_privacy?ie=UTF8&nodeId=3312401)">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1278429326580125759)** (93 messages🔥🔥): 

> - `Unsloth vs OpenRLHF`
> - `Unsloth finetuning`
> - `Unsloth multi-GPU`
> - `Unsloth inference`
> - `Unsloth on AWS` 


- **Unsloth vs OpenRLHF：速度与显存**：Unsloth 使用 4-bit 量化，与 OpenRLHF 相比，实现了更快的训练速度和更低的 VRAM 占用。
   - 虽然 Unsloth 目前仅支持 4-bit 量化模型进行微调，但他们正在努力增加对 8-bit 和非量化模型的支持。他们声称不同方法在性能或可复现性之间没有权衡（损失）。
- **在 AWS 上进行 Unsloth 微调**：Unsloth 目前没有专门的 AWS 微调指南。
   - 不过，一些用户正在使用 Sagemaker 在 AWS 上微调模型，并且网上有大量的 Unsloth YouTube 视频和 Google Colab 示例。
- **Unsloth 多 GPU 支持**：Unsloth 目前不支持多 GPU 训练。
   - 这意味着你无法训练超过单块 GPU 显存容量的模型，尽管 70B 模型在使用 4-bit 量化时仅需要 48GB 的 VRAM。
- **Unsloth 模型合并**：你可以通过将 adapter 和基础模型都上传到 Hugging Face，并使用 `model.push_to_hub_merged` 函数来合并它们。
   - 你可以使用 `save_method = 'merged_4bit_forced'` 参数以 4-bit 格式保存合并后的模型。
- **Unsloth EOS Token 映射**：Unsloth 中的 `map_eos_token = True` 选项允许你在不进行训练的情况下将 `<|im_end|>` token 映射到 `</s>` token，这对于特定的聊天模板可能是必要的。
   - 这种映射有助于避免输出乱码，因为某些模型期望 prompt 末尾只有一个 `<|im_end|>` token，并且在进行多轮对话 prompt 微调时，可能需要将其他的 `<|im_end|>` token 替换为 `</s>`。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki">Home</a>：微调 Llama 3.1, Mistral, Phi &amp; Gemma LLMs 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://colab.re">Sou Cidadão - Colab</a>：通过 Colab，您可以安排服务、报告需求、开具文件并积极参与您所在城市的决策！</li><li><a href="https://github.com/linkedin/Liger-Kernel">GitHub - linkedin/Liger-Kernel: Efficient Triton Kernels for LLM Training</a>：用于 LLM 训练的高效 Triton 内核。欢迎在 GitHub 上为 linkedin/Liger-Kernel 做出贡献。
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1278474714053873676)** (1 messages): 

> - `ML model deployment challenges`
> - `LLM limitations`
> - `Survey for ML Professionals` 


- **调研征集 ML 模型部署见解**：发布了一项调查，询问 ML 专业人士关于模型部署的经验，特别关注常见问题和解决方案。
   - 该调查包括关于工作角色、首要挑战、这些挑战发生的时间点、难度以及所使用的解决方案等问题，旨在了解将 ML 模型投入生产环境的复杂性。
- **强调模型部署中的挑战**：该调查旨在确定部署机器学习模型时遇到的前三个问题，为该领域专业人士面临的实际障碍提供宝贵的见解。
   - 它寻求揭示这些挑战的频率、严重程度和根本原因，最终为改进模型部署的解决方案和最佳实践铺平道路。
- **LLM 局限性探索**：调查还包含一个可选部分，专门用于探索在使用大语言模型（LLMs）时遇到的特定问题。
   - 这一部分鼓励受访者分享任何阻碍其利用 LLM 技术获得最佳结果的特定服务或工具，为研究和开发提供宝贵的反馈。



**提及的链接**：<a href="https://forms.gle/GaViDYGLFopVVTTk6">LLM Problems research</a>：嘿，我正在开展一个 ML 项目，我需要正在构建模型并将其部署到生产环境的人员的帮助。

  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1278438519106179127)** (29 条消息🔥): 

> - `Gemma2:2b Fine-tuning`
> - `Unsloth for Fine-tuning`
> - `Function Calling Datasets`
> - `APIGen`
> - `Mistral Fine-tuning` 


- **针对 Function Calling 的 Gemma2:2b Fine-tuning**：一位用户正在寻求关于如何使用 [XLM Function Calling 60k 数据集](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) 和 [提供的 notebook](https://colab.research.google.com/drive/1weTpKOjBZxZJ5PQ-Ql8i6ptAY2x-FWVA?usp=sharing) 对 Ollama 的 Gemma2:2b 模型进行 Function Calling Fine-tuning 的指导。
   - 他们不确定如何将数据集格式化为 instruction、input 和 output 格式，特别是关于 "tool use" 列的处理。
- **使用 Unsloth 简化 Fine-tuning**：一位成员强调了使用 Unsloth 对 Llama-3、Mistral、Phi-3 和 Gemma 等 LLM 进行 Fine-tuning 的优势，声称该过程速度快 2 倍，内存占用减少 70%，且能保持准确性。
   - 该成员提供了 Unsloth 教程的链接，其中包括自动将 Fine-tuning 后的模型导出到 Ollama 以及自动创建 `Modelfile`。
- **APIGen Function Calling 数据集**：一位成员提到了 [APIGen](https://apigen-pipeline.github.io/) 项目，这是一个自动化的数据生成流水线，用于创建可验证的高质量 Function Calling 数据集。
   - 项目的 [论文](https://arxiv.org/abs/2406.18518)、[网站](https://apigen-pipeline.github.io/) 和 [模型](https://huggingface.co/collections/Salesforce/xlam-models-65f00e2a0a63bbcd1c2dade4) 链接也已提供。
- **针对检索任务的 Mistral Fine-tuning**：一位用户正尝试利用 [NV-Embed](https://huggingface.co/nvidia/NV-Embed-v1) 模型对 Mistral 进行检索任务的 Fine-tuning，并由于不生成 token 而是获取句子结束 token 的 embedding，从而对损失函数进行了调整。
   - 他们询问是否可以在常规的 Transformers 或 PyTorch-Lightning 代码中使用 Unsloth，而不是使用 `SFTTrainer`。
- **Xformers 安装问题**：一位用户在尝试使用 Gemma notebook 时遇到了 `ImportError: Unsloth: Xformers was not installed correctly.` 错误，但在使用 Llama notebook 时没有出现。
   - 他们通过将 Xformers 安装从 Xformers<0.0.27 更改为 Xformers 并手动安装 Triton 解决了该问题，并建议针对类似问题采用此修复方法。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama">如何 Fine-tuning Llama-3 并导出到 Ollama | Unsloth 文档</a>：为创建可在 Ollama 本地运行的定制化个人助手（类似 ChatGPT）提供的初学者指南</li><li><a href="https://colab.research.google.com/drive/1weTpKOjBZxZJ5PQ-Ql8i6ptAY2x-FWVA?usp=sharing#scrollTo=IqM-T1RTzY6C">Google Colab</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k">Salesforce/xlam-function-calling-60k · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/nvidia/NV-Embed-v1">nvidia/NV-Embed-v1 · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/959">类似于 Huggingface `num_return_sequences` 的多重生成 · Issue #959 · unslothai/unsloth</a>：我想从单个 prompt 生成多个输出。有没有办法让 Fine-tuning 后的 llama3.1 模型实现类似于 Huggingface 中 num_return_sequences 的多重生成？
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/)** (1 条消息): 

hamchezz: 我想为了某个不明确的目标 Fine-tuning 一个 LLM，纯粹是因为好玩 😄
  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1278729107256901652)** (1 messages): 

> - `Runpod pricing`
> - `LLaMa 4 MoE`
> - `Flexattention`
> - `Unsloth Pro training` 


- **Runpod H200 价格预测**：一位成员预测 [Runpod](https://runpod.io/) 将在 12 个月内以 **每小时 6 美元** 的价格提供 **H200**。
- **LLaMa 4 系列预测**：一位成员预测 **LLaMa 4 系列** 将包含一个参数量为 **70-100B** 的 **Mixture of Experts (MoE)** 模型。
   - 预测该模型的性能将略优于目前的 **70B LLaMa 模型**。
- **Flexattention 结合**：一位成员预测 **Flexattention** 将在未来 12 个月内实现 **non-contaminated packing** 与 **FA3** 的结合。
- **Unsloth Pro 训练速度**：一位成员预测，**Unsloth Pro**、**Flexattention** 和 **FP8 activations** 的结合将使 **LLaMa 4 模型** 的训练速度超过在 **H100** 上训练的 **LLaMa 3 8B 模型**。
   - 这将通过在 **H200** 上训练 **LLaMa 4 模型** 来实现。


  

---



### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1278814192404664381)** (1 messages): 

> - `Discord Community Growth` 


- **Perplexity Discord 成员突破 10 万**：Perplexity AI Discord 服务器正式达到 **100,000 名成员**！
   - 团队对社区的支持和反馈表示感谢，并对未来的增长和演进充满期待。
- **感谢社区**：Perplexity AI 团队对从 Discord 社区收到的所有支持和反馈表示衷心感谢。
   - 团队很高兴能与社区一起继续成长和发展。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1278429570915106916)** (97 条消息🔥🔥): 

> - `Perplexity Pro issues`
> - `Perplexity AI Issues`
> - `AI model limitations`
> - `Perplexity AI model selection`
> - `Perplexity AI usability` 


- **Perplexity Pro 会员问题**：用户报告了 Perplexity Pro 会员身份的问题，包括洋红色会员标识消失、免费 LinkedIn Premium 优惠失效，以及“追问 (Ask Follow-up)”功能的问题。
   - 其他用户也遇到了“追问”功能的问题，即在 Perplexity 回复中突出显示一行文本时，原本出现的“追问”选项消失了。
- **Perplexity AI 在事实准确性方面表现不佳**：用户对 Perplexity AI 倾向于将假设当作事实陈述表示担忧，且经常出错。
   - 他们分享了一些对话线程的例子，其中 Perplexity AI 错误地提供了有关政府表格和抓取 Google 的信息，这表明其回复需要更强大的事实核查和人工审查。
- **在 AI 模型迷宫中穿行**：用户在选择最佳 AI 模型时感到困惑，讨论了 Claude 3 Opus、Claude 3.5 Sonnet 和 GPT-4o 的优缺点。
   - 几位用户指出，某些模型（如 Claude 3 Opus）限制为 50 个问题，用户不确定 Claude 3.5 Sonnet 是否是更好的选择，尽管它也有其局限性。
- **Perplexity AI 易用性挑战**：用户强调了 Perplexity AI 平台的易用性问题，包括难以访问已保存的线程以及 Prompt 区域的问题。
   - 一位用户指出 Chrome 扩展程序的描述不准确，错误地声称 Perplexity Pro 使用 GPT-4 和 Claude 2，这可能误导了平台的实际能力。
- **Perplexity API 集成**：用户询问了将 Perplexity API 连接到 Cursor AI 平台的可能性，寻求集成方法的指导。
   - 虽然有一位用户报告尝试通过 OpenAI API key 设置进行连接但遇到了加载问题，但其他用户表示有兴趣利用 Perplexity API 开发 Telegram 聊天机器人。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://pandoc.org/">Pandoc - index</a>: 未找到描述</li><li><a href="https://tenor.com/view/griffith-berserk-eclipse-guts-berserk-anime-meme-gif-10622855093064880455">Griffith Berserk GIF - Griffith Berserk Eclipse - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/hatsune-miku-hatsunemiku-vocaloid-leekspin-gif-27084853">Hatsune Miku GIF - Hatsune Miku Hatsunemiku - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://chromewebstore.google.com/detail/perplexity-ai-search/bnaffjbjpgiagpondjlnneblepbdchol">Perplexity - AI Search - Chrome 网上应用店</a>: 升级你的默认搜索引擎</li><li><a href="https://x.com/ai_for_success/status/1828996306767143143?s=46">来自 AshutoshShrivastava (@ai_for_success) 的推文</a>: 在 ChatGPT 与其他任何聊天机器人的对比中，没有人能接近 OpenAI。他们遥遥领先于所有人，除非他们愿意，否则并不急于发布任何东西。Strawberry 或者 ...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1278428900019404864)** (9 条消息🔥): 

> - `MrBeast`
> - `Perplexity AI Discord`
> - `Anthropic's Claude`
> - `Kustom.tech`
> - `OpenAI's Threads` 


- **MrBeast：他怎么了？**：一位用户在 [Perplexity 搜索](https://www.perplexity.ai/search/what-happened-to-mrbeast-S0hJBJ01TSKV6CqiLDXnvw)中询问了“MrBeast 发生了什么？”。
- **Perplexity AI Discord 公告**：一位用户分享了一条消息，提醒另一位用户在 Perplexity AI Discord 上将其线程设为“可共享 (Shareable)”。
- **Anthropic 的 Claude**：有一个关于 [Anthropic Claude](https://www.perplexity.ai/search/anthropic-publishes-claude-s-p-szxQ2QXlRE2ltexxQPe5Hw) 信息的 Perplexity 搜索。
- **Kustom.tech**：Discord 频道中分享了网址 'kustom.tech'。
- **Perplexity 的搜索功能**：一位用户在 Perplexity 搜索中询问了“Perplexity AI 能否协助处理[这个话题](https://www.perplexity.ai/search/can-perplexity-ai-assist-with-WLV2oRgzQ1y.qQXc7PkDJw)”。



**提到的链接**: <a href="https://www.youtube.com/embed/AAumUqa5d-U">YouTube</a>: 未找到描述

  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1278668411072675910)** (14 条消息🔥): 

> - `Perplexity API`
> - `Beta 申请`
> - `Telegram 聊天机器人`
> - `Temu 推广机器人`
> - `免费 API 额度` 


- **Perplexity API Beta 申请**: 一位用户对申请 Beta 计划（特别是为了测试引用返回功能）后未收到回复表示沮丧。
   - 他们表示在过去几个月中已多次申请，并渴望让他们的用户尝试引用功能。
- **使用 Make 和 Perplexity API 构建 Telegram 聊天机器人**: 一位用户正在寻求帮助，希望使用 [Make](https://www.make.com/) 和 Perplexity API 创建一个 Telegram 聊天机器人。
- **Temu 推广机器人充斥讨论论坛**: 一位用户对讨论论坛中出现大量 Temu 推广机器人表示恼火。
- **Pro 用户缺失免费 API 额度**: 多位用户报告称，在订阅 Pro 计划后未收到 5 美元的免费 Perplexity API 额度。
   - 一名版主澄清说，Pro 功能目前还无法通过 API 使用。
- **API 无法使用 Pro 搜索**: 一位用户询问了是否可以通过 Perplexity API 进行 Pro 搜索。
   - 一名版主确认，包括 Pro 搜索在内的 Pro 功能目前无法通过 API 使用。



**提及的链接**: <a href="https://docs.perplexity.ai/discuss/66cf2aea1f29e1004397a298">API Beta 访问</a>: 你好，获得 Beta 使用许可需要多长时间？我想测试引用返回功能，我们在过去几个月中申请了多次，但音讯全无。我们有用户...

  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1278441897915977779)** (38 条消息🔥): 

> - `LLM Tokenization`
> - `模型中的谄媚行为 (Sycophancy Behavior)`
> - `MMLU 问题`
> - `COT 与 Scratch Pad 评估` 


- **LLM 看不到字母，它们看到的是 Token**: 一位成员指出，关于 LLM 计算字符数的讨论都是误导，因为模型看不到字母，它们看到的是 Token，就像一个庞大的词表。
   - 他们以阅读日语汉字为例，认为这比阅读英文文本更接近 LLM 的工作方式。
- **Claude 是否存在“谄媚行为”？**: 一位成员询问 LLM 是否有谄媚倾向，特别是在推理方面。
   - 另一位成员建议添加 System Message 来缓解这一问题，但也表示即便如此，这更像是一种小把戏，而非实用的生产工具。
- **MMLU 不是衡量实际应用的好基准**: 一位成员指出，MMLU 并不是构建实用 LLM 的好基准，因为它与实际用例的相关性不强。
   - 他们举例说明了关于弗洛伊德过时的性理论问题，暗示该基准无法反映用户对 LLM 的真实需求。
- **COT 和 Scratch Pad 评估仍处于早期阶段**: 一位成员表示他们并不关心 MMLU，因为它只用于增量对比（即“氛围检查”），他们正在等待新版本的发布。
   - 另一位成员表示赞同，称他们希望看到带有 Scratchpad 评估的思维图 (Graph of Thought)，但他们缺乏必要的 GPU。



**提及的链接**: <a href="https://huggingface.co/datasets/joey234/mmlu-human_sexuality-original-neg">joey234/mmlu-human_sexuality-original-neg · Hugging Face 数据集</a>: 未找到描述

  

---

### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1278445348897034240)** (28 messages🔥): 

> - `Cohere for AI Scholars Program`
> - `Cohere for AI Community`
> - `Cohere API`
> - `CrewAI`
> - `Aya-23-8b Inference Time` 


- **Cohere for AI Scholars Program**: Cohere For AI 很高兴开启第三届 Scholars Program 的申请，该计划旨在改变研究进行的地点、方式以及参与者。
   - 该计划旨在帮助研究人员和志同道合的合作者。
- **Cohere for AI Community**: 一位成员建议加入 Cohere for AI Community，这是研究人员和合作者的资源中心。
   - 该社区提供相关信息和人员，以协助 Cohere for AI 的 Scholar Program。
- **Aya-23-8b Inference Time**: 有人提出了关于 Aya-23-8b 模型生成 50 个 token 的推理时间问题。
   - 回复指出，推理时间在很大程度上取决于基础设施和模型量化 (quantization)。
- **Using Cohere with CrewAI**: 有人询问了如何将 Cohere 与 CrewAI（一个用于创建对话式 AI 应用的工具）结合使用。
   - 具体而言，该咨询集中在将 Cohere 与 CrewAI 集成时，是否可以指定所使用的模型类型。
- **Cohere API and AI Discussion**: 该服务器设有一个讨论 Cohere API 的频道，以及另一个用于通用 AI 讨论的频道。
   - 该服务器还有一个用于分享使用 Cohere 制作的酷炫项目的频道。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.crewai.com">Home</a>: 用于编排角色扮演、自主 AI Agent 的前沿框架。通过促进协作智能，CrewAI 赋能 Agent 无缝协作，处理复杂任务。</li><li><a href="https://cohere.com/blog/cohere-for-ai-scholars-program-2025">Cohere For AI Scholars Program: Your Research Journey Starts Here</a>: 今天，Cohere For AI 很高兴开启第三届 Scholars Program 的申请，旨在改变研究进行的地点、方式以及参与者。
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1278588673872367616)** (1 messages): 

> - `` 


- **Internal Tool Hosted on Admin Panel**: 一位成员分享说，该工具目前托管在公司的管理面板上，但很快将提供公开托管版本。
- **Tool Availability Update**: 该工具目前托管在公司的管理面板上，预计很快会发布公开托管版本。


  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1278740120802951283)** (2 messages): 

> - `LlamaIndex Workflows`
> - `GymNation Case Study` 


- **LlamaIndex Workflows Tutorial Now Available**: LlamaIndex 文档中现已提供关于 LlamaIndex Workflows 的全面教程。
   - 该教程涵盖了一系列主题，包括 Workflows 入门、循环与分支、状态维护以及并发流。
- **GymNation's Success Story with LlamaIndex**: GymNation 与 LlamaIndex 合作，旨在提升会员体验并推动实际业务成果。
   - 他们取得了令人印象深刻的成果，包括数字线索到销售转化率提升了 20%，以及数字线索的对话率达到 87%。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1278528269993513002)** (37 条消息🔥): 

> - `Function Calling LLMs`
> - `Workflows`
> - `Image & Text Retrieval`
> - `LlamaIndex Integration`
> - `Pinecone Vector Store` 


- **用于流式输出的 Function Calling LLMs**：一位成员正在寻找使用 Function Calling LLMs 构建 Agent 并流式传输最终输出的示例。
   - 他们正利用 Workflows 几乎从零开始构建 Agent，并寻找一种解决方案，以避免因将完整消息传递到最后一步而导致的延迟影响。
- **用于复杂逻辑的 Workflows**：一位成员分享了一个 Workflow 示例，该示例利用异步生成器（async generator）来检测工具调用并流式传输输出。
   - 他们还讨论了使用 "Final Answer" 工具的可能性，该工具可以限制输出 Token，并在被调用时将最终消息传递给最后一步。
- **图像 + 文本检索的最佳实践**：一位成员询问了结合图像和文本检索的最佳方法。
   - 他们正在考虑对图像和文本都使用 CLIP Embeddings，但担心 CLIP 与 txt-embeddings-ada-002 等专用文本嵌入模型相比在语义优化方面的表现。
- **LlamaIndex 集成扩展**：一位成员表示有兴趣为用户扩展 LlamaIndex 的集成。
   - 他们希望在实施之前与团队讨论这个想法，并就应该提交 Issue、在频道中讨论还是采取其他方式寻求指导。
- **Query Engine 弃用及替代方案**：一位成员询问了 LlamaIndex 中 QueryEngines 的弃用情况。
   - 澄清指出，只有一种特定的结构化输出方法被弃用，而非所有查询引擎，首选方式是在查询引擎中使用 `llm.as_structured_llm(output_class)`。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/1GhF8uBC2LrnYf195CcTe_e5K8Ai6Z4ta#scrollTo=3cBku4_C0CQk)">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1UjDJMyXR11HKIki3tuMew6EEzq91ewYw?usp=sharing#scrollTo=1XoDZK0YvQQe">Google Colab</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/understanding/workflows/stream/">Streaming events - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/querying/structured_outputs/query_engine/">(Deprecated) Query Engines + Pydantic Outputs - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/CakeCrusher/TaxonomySynthesis">GitHub - CakeCrusher/TaxonomySynthesis: An AI-driven framework for synthesizing adaptive taxonomies, enabling automated data categorization and classification within dynamic hierarchical structures.</a>：一个 AI 驱动的框架，用于合成自适应分类法，在动态分层结构中实现自动数据归类和分类。 - CakeCrusher/TaxonomySynthesis
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1278731246360592425)** (1 条消息): 

> - `GenAI Ops`
> - `GenAI Ops Community`
> - `GenAI Ops Book` 


- **GenAI Ops 社区发布**：一位成员正与微软英国 CTO 合作，发起一个名为 [GenAI Ops](https://genaiops.ai) 的非营利社区，致力于生成式 AI 的运营化。
- **GenAI Ops 大使招募**：该社区目前正在寻找在这一领域有深厚背景并认同社区价值观的大使候选人。
- **关于 GenAI Ops 的新书**：该成员最近出版了一本名为 *Exploring GenAI Ops: Empowering Innovators and Operationalizing Generative AI* 的书，可以在 [Amazon](https://www.amazon.co.uk/Exploring-GenAIOps-Empowering-Innovators-Operationalising/dp/B0DF6Q96SD/ref=sr_1_1?dib=eyJ2IjoiMSJ9.ohHnrGvMuiescF6bzbm3mQ.sbUybxkrY36cyJBZZA-0s7FNl8-idFJsprQfDXn403k&dib_tag=se&keywords=genaiops&qid=1724942520&s=books&sr=1-1) 上找到。
   - 这本书是 GenAI Ops 的基础和入门读物，该成员认为它对社区可能具有启发性或实用价值。



**提到的链接**：<a href="https://www.amazon.co.uk/Exploring-GenAIOps-Empowering-Innovators-Operationalising/dp/B0DF6Q96SD/ref=sr_1_1?dib=eyJ2IjoiMSJ9.ohHnrGvMuiescF6bzbm3mQ.sbUybxkrY36cyJBZZA-0s7FNl8-idFJsprQfDXn403k&dib_tag=se&keywords=genaiops&qid=1724942520&s=books&sr=1-1">Exploring GenAIOps: Empowering Leaders and Innovators: Operationalising Generative AI: Amazon.co.uk: Kirby, Harrison: 9798334554955: Books</a>：未找到描述

  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1278500990039097396)** (33 条消息🔥): 

> - `Agency 融资`
> - `AI Engineer 见面会与峰会`
> - `个人使用的 AI`
> - `Midjourney 硬件`
> - `Llama 3 开源采用情况` 


- **Agency 融资 260 万美元**：Agency 是一家构建 AI Agent 的公司，宣布融资 260 万美元，用于开发“具有世代意义的技术”并赋予其 AI Agent 生命。
   - 该公司的愿景是构建一个 AI Agent 无处不在且成为我们生活重要组成部分的未来，正如其网站 [agen.cy](http://agen.cy) 所强调的那样。
- **AI Engineer 见面会与峰会**：AI Engineer 社区正在扩大！首场伦敦见面会定于 9 月举行，第二届在纽约市（NYC）举行的 AI Engineer Summit 计划于 12 月举行。
   - 有兴趣参加伦敦见面会的人可以在[这里](https://x.com/dctanner/status/1827071893448618453?s=46)找到更多信息，并鼓励 NYC 峰会的潜在赞助商[进行联系](mailto:info@ai.engineer)。
- **个人使用的 AI**：DeepMind 的研究科学家 Nicholas Carlini 认为，AI 的重点应该从宏大的革命承诺转向其对个人的益处。
   - 他的博客文章《我如何使用 AI》（"How I Use AI" [https://nicholas.carlini.com/writing/2024/how-i-use-ai.html](https://nicholas.carlini.com/writing/2024/how-i-use-ai.html)）详细介绍了他在 AI 工具方面的实际应用，引起了许多读者的共鸣，尤其是在 Hacker News 上（[https://news.ycombinator.com/item?id=41150317](https://news.ycombinator.com/item?id=41150317)）。
- **Midjourney 进军硬件领域**：广受欢迎的 AI 图像生成平台 Midjourney 正式进入硬件领域。
   - 有兴趣加入其旧金山新团队的人员可以联系 [hardware@midjourney.com](mailto:hardware@midjourney.com)。
- **Llama 3 开源采用率飙升**：开源 Llama 模型系列继续受到关注，在 Hugging Face 上的下载量已超过 3.5 亿次，与去年相比增长了十倍。
   - Llama 的受欢迎程度延伸到了云服务提供商，自 5 月以来 Token 使用量增加了一倍多，并在包括埃森哲（Accenture）、AT&T、DoorDash 等各行各业得到采用。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/MLStreetTalk/status/1828848765039718439">来自 Machine Learning Street Talk (@MLStreetTalk) 的推文</a>: 我们刚刚发布了对生成式 AI 之父 @SchmidhuberAI 的采访！“ChatGPT”中的 G、P 和 T（GPT 代表“Generative Pre-Trained Transformer”）可以追溯到 Juergen 的……</li><li><a href="https://x.com/magicailabs/status/1829206893765767282">来自 Magic (@magicailabs) 的推文</a>: LTM-2-Mini 是我们第一个拥有 1 亿 Token 上下文窗口的模型。这相当于 1000 万行代码，或 750 本小说。完整博客：https://magic.dev/blog/100m-token-context-windows 评估、效率……</li><li><a href="https://x.com/aiatmeta/status/1829157383052111946?s=46">来自 AI at Meta (@AIatMeta) 的推文</a>: 开源 AI 是未来的方向，今天我们将分享 Llama 模型采用和使用情况的快照。在此阅读完整更新 ➡️ https://go.fb.me/e7odag 🦙 A ...</li><li><a href="https://x.com/midjourney/status/1828839444130214208?s=12">来自 Midjourney (@midjourney) 的推文</a>: 我们正式进军硬件领域。如果你有兴趣加入旧金山的新团队，请发送电子邮件至 hardware@midjourney.com</li><li><a href="https://huggingface.co/Salesforce/xLAM-8x22b-r">Salesforce/xLAM-8x22b-r · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/AlexReibman/status/1828838507282084296">来自 Alex Reibman 🖇️ (@AlexReibman) 的推文</a>: 𝐏𝐫𝐨𝐮𝐝 𝐭𝐨 𝐟𝐢𝐧𝐚𝐥𝐥𝐲 𝐚𝐧𝐧𝐨𝐮𝐧𝐜𝐞 𝐨𝐮𝐫 𝐟𝐮𝐧𝐝𝐫𝐚𝐢𝐬𝐞 𝐟𝐨𝐫 𝐀𝐠𝐞𝐧𝐜𝐲 我们已获得 260 万美元，用于构建具有世代意义的技术并赋予 AI Agent 生命。以下是……</li><li><a href="https://open.substack.com/pub/swyx/p/carlini?r=1h4isl&utm_campaign=post&utm_medium=web">为什么你应该编写自己的 LLM 基准测试 —— 对话 Google DeepMind 的 Nicholas Carlini</a>: 窃取 OpenAI 模型，为什么 LLM 基准测试对你没用，如何发现使用 AI 的价值，以及他们如何用过期域名污染 LAION</li><li><a href="https://techcrunch.com/2024/08/29/github-copilot-competitor-codeium-raises-150m-at-a-1-25b-valuation/">GitHub Copilot 竞争对手 Codeium 以 12.5 亿美元估值融资 1.5 亿美元 | TechCrunch</a>: Codeium 是一家开发 AI 驱动工具以对抗 GitHub Copilot 的初创公司，已以 12.5 亿美元的估值筹集了 1.5 亿美元。</li><li><a href="https://techcrunch.com/2024/0">2024 | TechCrunch</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1278805134695989333)** (1 条消息): 

> - `Latent Space Podcast`
> - `LLM Benchmarks`
> - `Nicholas Carlini`
> - `Google DeepMind`
> - `Training Data Extraction` 


- **新的 Latent Space Podcast 剧集**：最新一期 [Latent Space Podcast](https://x.com/latentspacepod/status/1829173832877519152) 邀请了来自 [Google DeepMind](https://twitter.com/GoogleDeepMind) 的 [Nicholas Carlini](https://twitter.com/carlini)。
   - 本期节目涵盖了多个主题，包括 [Carlini 使用 AI 的方法](https://x.com/latentspacepod/status/1829173832877519152)、他的 [自定义 LLM benchmark](https://x.com/latentspacepod/status/1829173832877519152)，以及 [从 LLM 中提取训练数据](https://x.com/latentspacepod/status/1829173832877519152)，包括 [OpenAI 移除 logprobs](https://x.com/latentspacepod/status/1829173832877519152) 所带来的影响。
- **即将举行的 AI Meetup**：公告还强调了由一名成员组织的即将举行的 [AI meetup](https://x.com/latentspacepod/status/1829173832877519152)。
   - 该 Meetup 定于下个月举行，旨在面向对 AI 感兴趣的人士。



**提及的链接**：<a href="https://x.com/latentspacepod/status/1829173832877519152">来自 Latent.Space (@latentspacepod) 的推文</a>：🆕 为什么你应该编写自己的 LLM benchmarks —— 对话来自 @GoogleDeepMind 的 Nicholas Carlini。涵盖了他的代表作：- 我如何使用 AI - 我的大语言模型 benchmark - 提取训练数据...

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1278432973862998149)** (9 条消息🔥): 

> - `OpenInterpreter development`
> - `Auto-run safety`
> - `Backups`
> - `House Party`
> - `Terminal app recommendations` 


- **OpenInterpreter 开发依然活跃！**：开发工作仍在持续进行中，[OpenInterpreter GitHub 仓库的 main 分支](https://github.com/OpenInterpreter/open-interpreter/commits/main/)最近有新的提交 (commits)。
- **Auto-run 很危险！**：`auto_run` 具有危险性，提醒用户在使用时务必留意输出内容。
- **下周举行 House Party！**：计划于下周较早时间举行 House Party，以鼓励更多人参与。
- **KDE 的推荐终端应用**：一位用户正在寻找适用于 KDE 的推荐终端应用，并指出他们目前使用的 Konsole 在 GPT-4 输出文本并滚动时会出现屏幕花屏现象。



**提及的链接**：<a href="https://github.com/OpenInterpreter/open-interpreter/commits/main/">Commits · OpenInterpreter/open-interpreter</a>：计算机的自然语言界面。通过在 GitHub 上创建账号来为 OpenInterpreter 的开发做出贡献。

  

---

### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1278438171285131306)** (17 条消息🔥): 

> - `Daily Bots`
> - `Bland`
> - `AI Phone Agents`
> - `Frame`
> - `Diffusion Models` 


- **Daily Bots：实时 AI 的开源云平台**：Daily Bots 于今日发布，这是一个用于语音、视觉和视频 AI 的低延迟云平台，允许开发者使用任何 LLM 构建语音对语音（voice-to-voice）交互，延迟低至 500ms。
   - 该平台提供开源 SDK，支持混合匹配 AI 模型，并在 Daily 的实时全球基础设施上大规模运行。这是与客户和合作伙伴共同努力 18 个月的成果，并利用了开源项目 RTVI 和 Pipecat。
- **Bland：你最新的 AI 员工**：Bland 是一款声音听起来像真人的可定制电话呼叫 Agent，已获得 2200 万美元的 A 轮融资。
   - 该 AI Agent 可以使用任何语言或声音交谈，全天候 24/7 同时处理数百万个电话，专为任何无幻觉（hallucinations）的使用场景设计。Bland 可在 [Bland.ai](http://Bland.ai) 进行通话体验。
- **Frame：开源 AR 眼镜**：Frame 是一副专为大多数人设计的 AR 眼镜，重量不到 40g，并提供全天候电池续航。
   - 该眼镜配备了明亮的 microOLED 显示屏、20 度视野，并且完全开源，设计文件和代码可在 [GitHub](https://github.com/brilliantlabsAR) 上获取。它们可以在 AR 中试戴，瞳距（IPD）范围为 58-72mm。
- **Diffusion Models 是游戏引擎**：用于 AI 图像生成的 Diffusion Models 也可以用来创建可玩的游戏。
   - 一个 Diffusion Model 被用来预测经典射击游戏 DOOM 的下一帧，从而在没有传统游戏引擎的情况下实现 20fps 的可玩游戏。关于此话题的更多阅读请访问 [https://gamengen.github.io/](https://gamengen.github.io/)。
- **AgentOps：构建 AI Agents**：AgentOps 创始人 Adam Silverman 在一段名为“我测试了 400 个 AI Agents，这些是最好的”的 YouTube 视频中讨论了最优秀的 AI Agents。
   - 他推广了其公司的服务，包括 Skool、Agency 和 AgentOps，旨在帮助人们通过 AI Agents 赚钱。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://gamengen.github.io/">GameNGen</a>: Diffusion Models 是实时游戏引擎</li><li><a href="https://fxtwitter.com/usebland/status/1828882563588612233?s=46&t=G6jp7iOBtkVuyhaYmaDb0w">来自 Bland.ai (@usebland) 的推文</a>: 今天对我们来说是一个重要的里程碑。我们完成了 2200 万美元的 A 轮融资。随着我们结束隐身模式，我们想正式向您介绍 Bland，您最新的 AI 员工。Bland 是一个...</li><li><a href="https://brilliant.xyz/products/frame">Frame</a>: Frame 被设计为可佩戴的眼镜，开箱即用一套 AI 功能。无论是作为日常眼镜还是工作台原型工具，Frame 都已准备就绪。</li><li><a href="https://tenor.com/view/reaction-mother-of-god-shades-infinite-woah-gif-4858850">Reaction Mother Of God GIF - Reaction Mother Of God Shades - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/i/status/1825946246886076785">来自 Daily (@trydaily) 的推文</a>: 今天我们发布了 Daily Bots，这是用于语音、视觉和视频 AI 的超低延迟开源云。使用任何 LLM 构建语音对语音，对话延迟低至 500ms。通过 Daily...</li><li><a href="https://youtu.be/z4QsBsO3SS0?t=371&si=lzexLc5j0gjdjRht">&quot;我测试了 400 个 AI Agents，这些是最好的&quot; - Adam Silverman</a>: 开始通过 AI Agents 赚钱: https://www.skool.com/new-societyAgency: https://www.agen.cy/AgentOps: https://www.agentops.ai/Adam 的 Twitter: https://x.c...</li><li><a href="https://fxtwitter.com/emollick/status/1828647931588587709?s=46&t=G6jp7iOBtkVuyhaYmaDb0w">来自 Ethan Mollick (@emollick) 的推文</a>: 哇，Diffusion Models（用于 AI 图像生成）也是游戏引擎——一种世界模拟。通过预测经典射击游戏 DOOM 的下一帧，你可以获得一个 20 fps 的可玩游戏...
</li>
</ul>

</div>
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1278466861104631902)** (16 messages🔥): 

> - `Macbook pro 训练`
> - `GPU vs. CPU`
> - `训练速度`
> - `模型大小`
> - `训练成本` 


- **在 Macbook Pro 上训练 LLM**：一位用户报告称在 128GB 的 Macbook Pro 上成功训练了大模型。
   - 他们指出虽然可行，但速度**显著慢于**在 **RTX 3090** 上的训练，**训练速度大约减半**。
- **训练中的 GPU vs CPU**：在之前的赞助结束后，该用户正在寻求**高性价比**的训练方案。
   - 他们正在考虑将**降压版 3090** 或 **AMD 显卡**作为昂贵 H100 的替代方案。
- **训练的硬件考量**：一位用户建议在决定购买之前先**租赁硬件**，特别是对于初学者。
   - 他们建议花费 **$30 租赁不同的硬件**并尝试训练模型，以确定最佳配置。
- **训练速度与模型大小**：用户正在探索**模型大小**与**训练速度**之间的关系。
   - 他们特别关注在对比 **Nemotron-4-340b-instruct** 与 **Llama 405** 等模型时，训练时间是如何变化的。



**提到的链接**：<a href="https://huggingface.co/Replete-AI">Replete-AI (Replete-AI)</a>：未找到描述

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1278645042600214601)** (2 messages): 

> - `为对话微调 LLM`
> - `数据精简` 


- **为长对话微调 LLM**：一位成员表示他们拥有适用于长对话的优秀模型，但用于训练的数据集全都是 “ShareGPT” 类型的。
   - 他们希望个性化处理数据，特别是精简星号 (*) 包围的内容，例如将 "*she smile*" 改为 "*smiling*"。
- **通过指令精简内容**：该成员询问是否可以使用简单的指令来控制微调后的模型进行数据精简和重写。
   - 他们询问 LlamaForCausalLM 是否能够胜任，或者是否有更好的替代方案。
- **使用 Llama 进行简单提示 (Prompting)**：另一位成员建议直接向 Llama 传递带有 System Prompt 的提示词。
   - 他们提到这种方法看起来很简单，但可能需要检查是否存在误报 (False Positives)。


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/)** (1 messages): 

teknium: https://x.com/nousresearch/status/1829143753036366325?s=46

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1278740843066429511)** (15 messages🔥): 

> - `SQLDatabaseChain`
> - `Vector Stores`
> - `SQL Record Manager`
> - `RAG (Retrieval Augmented Generation)`
> - `Knowledge Graphs` 


- **SQLDatabaseChain + PGVector 实现混合搜索**：一位用户描述了他们当前存储和查询数据的设置：使用带有 `pgvector` 的 PostgreSQL 进行 embedding 存储，并使用 `SQLDatabaseChain` 将用户查询转换为 SQL 查询。
   - 他们打算修改 `SQLDatabaseChain` 的 prompt，以便在向量上进行搜索以获得更快的响应，但尚未实现。
- **RAG 与知识图谱：下一个前沿**：用户讨论了检索增强生成 (RAG) 对 AI 应用的好处，允许模型在不重新训练的情况下访问相关数据。
   - 他们表达了将 RAG 与知识图谱结合的兴趣，并提到了一种可能适用于其 text-to-SQL 问题的混合方法。
- **多数据库查询的 Prompt 工程**：用户面临为每个 SQL 数据库创建理想 prompt 的挑战，因为 schema 要求不同，导致性能问题和 prompt 模板冗余。
   - 他们询问了创建可适配各种数据库且不牺牲性能的 prompt 的潜在解决方案。
- **Docker 中 OllamaLLM 连接被拒绝**：另一位用户报告在 Docker 容器中尝试调用 `OllamaLLM` 时出现连接被拒绝错误，尽管与 Ollama 容器的通信是成功的。
   - 建议使用 `langchain_community.llms.ollama` 包作为权宜之计，并可能解决了该问题，这表明 `langchain_ollama` 包中可能存在 bug。
- **LangChain v2.0 中 Function Calling 的流式传输**：用户询问在 2.0 版本中是否可以将 LangChain function calling 与流式传输结合使用。
   - 未给出直接回答，但似乎该功能目前尚不可用，凸显了未来开发的一个潜在领域。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://medium.com/@sergio1101102/mastering-retrieval-augmented-generation-rag-a-practical-guide-for-new-developers-624be24ca516">Mastering Retrieval-Augmented Generation (RAG): A Practical Guide for New Developers</a>: 简介：我在今年年初开始了我的 LLM 之旅，并在此过程中学到了很多。我一直在工作并……</li><li><a href="https://github.com/langchain-ai/langchain/issues/25022>).">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 开发做出贡献。</li><li><a href="http://ollama:11434">)">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1278444511848304731)** (7 messages): 

> - `Torchtune Contributing`
> - `QLoRA + Llama 3.1 Memory Issues`
> - `Torchtune Github Issues` 


- **Torchtune 社区寻求帮助**：Torchtune 团队正鼓励社区成员通过完成一些小任务来为他们的仓库做贡献，在他们的 [GitHub issues 页面](https://github.com/pytorch/torchtune/issues?q=sort%3Aupdated-desc+is%3Aissue+is%3Aopen+label%3A%22community+help+wanted%22)上标有 "community help wanted" 标签的任务均可参与。
   - 他们也很乐意通过 Discord 为贡献者提供帮助。
- **QLoRA + Llama 3.1 内存问题**：一位拥有 4 张 A6000 显卡的成员报告在尝试使用 **Llama 3.1 70B** 训练 **QLoRA** 时遇到显存溢出 (OOM) 错误。 
   - 另一位成员质疑这是否为预期行为，认为这对于 **QLoRA** 来说应该是足够的，并建议提交一个带有可复现示例的 [GitHub issue](https://github.com/pytorch/torchtune/issues) 以进行排查。
- **Torchtune + PyTorch 2.4 兼容性**：一位成员询问了 **Torchtune** 与 **PyTorch 2.4** 的兼容性，得到了可以正常工作的确认。



**提及的链接**: <a href="https://github.com/pytorch/torchtune/issues?q=sort%3Aupdated-desc+is%3Aissue+is%3Aopen+label%3A%22community+help+wanted%22.">Issues · pytorch/torchtune</a>: 一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号为 pytorch/torchtune 开发做出贡献。

  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1278430591347327018)** (7 messages): 

> - `Fusion Models RFC`
> - `Batched Inference`
> - `Decoder-only Max Seq Len`
> - `Flamingo Model`
> - `Cache Position Tracking` 


- **Fusion Models RFC 讨论**：一位成员质疑在 `setup_caches` 函数中处理 decoder-only 的 max_seq_len 是否会导致问题，特别是对于 `CrossAttentionLayer` 和 `FusionLayer`。
   - 另一位成员同意这一方面应该单独处理，并建议探索一个 utility 来有效地处理它。
- **Flamingo Model 和 Batched Inference**：对话深入探讨了 Flamingo Model 对混合序列长度的使用，特别是其融合层，这需要一种专门的 `setup_caches` 方法。
   - 准确进行 Cache Position Tracking 的必要性得到了认可，并强调了 Flamingo PR 与包含更新 `setup_caches` 的 Batched Inference PR 之间潜在的重叠。
- **用于更新的独立 Cache Positions**：一位成员分享了一个更新的 PR，其特点是使用独立的 cache positions 来更新缓存，解决了填充输入（padded inputs）导致输入位置偏移的问题。
   - 讨论旨在确保这些更新与 Flamingo PR 的设计保持一致，并避免任何潜在冲突。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/pull/1424/files#diff-9ca4bd8f2b83354dfde8d0a9960f5669e8019001edd0ecda6069cf5aa69c57c7R74">[WIP][RFC] Batched inference 🤝 KV-cache 🤝 compile by SalmanMohammadi · Pull Request #1424 · pytorch/torchtune</a>：上下文：此 PR 的目的是什么？是添加新功能、修复 Bug、更新测试和/或文档，还是其他（请在此处添加）。请链接此 PR 解决的任何 Issue。Closes #125...</li><li><a href="https://github.com/pytorch/torchtune/pull/1283#discussion_r1710750698)">[RFC] Fusion Models by pbontrager · Pull Request #1283 · pytorch/torchtune</a>：[RFC] Fusion Models TLDR：Fusion Models 是将两个或多个预训练模型连接在一起，并进一步微调以作为一个模型运行。这是目前大多数 SOTA Multimodal 模型采用的方法。这 ...
</li>
</ul>

</div>
  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1278593906014945360)** (7 messages): 

> - `LinkedIn Job Applier`
> - `Agent Zero`
> - `GitHub repo`
> - `AIHawk`
> - `Pipelines` 


- **使用 AI 的 LinkedIn Job Applier：自动化申请流程**：一位成员分享了一个 [GitHub repo](https://github.com/feder-cr/linkedIn_auto_jobs_applier_with_AI)，该项目利用 [Agent Zero](https://link.to/agent-zero) 创建新的 Pipelines，自动申请 LinkedIn 上的职位。
- **仓库仍在开发中**：另一位成员询问了该仓库与之前关于 Agent Zero 讨论之间的联系。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/feder-cr/linkedIn_auto_jobs_applier_with_AI?fbclid=IwZXh0bgNhZW0CMTAAAR05wdHiDH4UfOwcjEB5fZLxMLKEkBrzxADEH4-eeHzvijzaLBbYiWUt2BU_aem_i9N5CPObzbeVMw_i3HZRiw">GitHub - feder-cr/linkedIn_auto_jobs_applier_with_AI: LinkedIn_AIHawk 是一个自动执行 LinkedIn 职位申请流程的工具。利用人工智能，它使用户能够以自动化和个性化的方式申请多个职位。</a>：LinkedIn_AIHawk 是一个自动执行 LinkedIn 职位申请流程的工具。利用人工智能，它使用户能够以自动化和个性化的方式申请多个职位...</li><li><a href="https://github.com/feder-cr/linkedIn_auto_jobs_applier_with_AI?fbclid=IwZXh0bgNhZW0CMTAAAR05wdHiDH4U">GitHub - feder-cr/linkedIn_auto_jobs_applier_with_AI: LinkedIn_AIHawk 是一个自动执行 LinkedIn 职位申请流程的工具。利用人工智能，它使用户能够以自动化和个性化的方式申请多个职位。</a>：LinkedIn_AIHawk 是一个自动执行 LinkedIn 职位申请流程的工具。利用人工智能，它使用户能够以自动化和个性化的方式申请多个职位...
</li>
</ul>

</div>
  

---

### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1278568812412801025)** (2 messages): 

> - `Generative Reward Models (GenRM)`
> - `DSPy Optimizers` 


- **新论文：Generative Reward Models (GenRM)**：一篇新论文提出了 **Generative Reward Models (GenRM)**，该模型利用 next-token prediction 目标来训练 verifiers，从而实现与 instruction tuning、chain-of-thought 推理的无缝集成，并通过多数投票 (majority voting) 利用额外的 inference-time compute 来提高验证性能。
   - 论文指出，GenRM 可以克服传统判别式 verifiers 的局限性，因为后者未能利用预训练 LLM 的文本生成能力，更多详情请[参阅论文](https://arxiv.org/abs/2408.15240)。
- **DSPy Optimizers 与示例排序**：一位用户询问哪些 DSPy optimizers 会改变示例/shots 的顺序，哪些不会。
   - 该用户似乎对不同 optimizer 策略对训练数据顺序的影响，以及这可能如何影响模型性能感兴趣。



**提到的链接**：<a href="https://arxiv.org/abs/2408.15240">Generative Verifiers: Reward Modeling as Next-Token Prediction</a>：Verifiers 或 reward models 常用于增强大语言模型 (LLMs) 的推理性能。一种常见的方法是 Best-of-N 方法，即从 N 个候选解决方案中进行筛选...

  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1278734988887593021)** (4 messages): 

> - `DSPY`
> - `Optimizers`
> - `KL Divergence`
> - `Synthetic Data`
> - `Human Responses` 


- **DSPY：一个具有挑战性的优化问题**：一位成员对使用 DSPY 实现其预期目的（抽象化模型、提示词和设置）的复杂性表示沮丧。
   - 他们分享了一个 [YouTube 视频链接](https://www.youtube.com/watch?v=lFXeJHhY3mA) 展示了他们的困境，并寻求资源以理解 DSPY 的优化技术。
- **寻求运行中的 Optimizers 示例**：一位成员请求提供运行中的 optimizers 示例，特别是 GitHub 仓库或展示其实现的资源链接。
   - 他们强调有兴趣了解如何有效地使用 optimizers，特别是在抽象化模型、提示词和设置的背景下。
- **利用人类回复引导 (Bootstrapping) 合成数据**：一位成员提出了一种引导合成数据的新方法：通过循环不同的模型和提示词，利用手写的人类回复来最小化 KL divergence 指标。
   - 他们寻求关于该方法可行性的反馈，将其作为生成与人类回复高度一致的合成数据的一种手段。


  

---



### **AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1278461820692201612)** (11 messages🔥): 

> - `Jamba 1.5 dependency issues`
> - `transformers version bug` 


- **Jamba 1.5 依赖问题**：一位用户报告在尝试使用 pytorch:23.12-py3 训练 Jamba 1.5 时遇到依赖问题。
   - 已确认 Jamba 1.5 与 Jamba Instruct (1.0) 基于相同的架构和基础模型。
- **Transformers 4.44.0 和 4.44.1 的 Bug**：发现 transformers 版本 4.44.0 和 4.44.1 包含一个限制运行 Jamba 架构能力的 bug。
   - 该 bug 已记录在 Jamba 1.5-Mini 的 Hugging Face 模型卡片上：[https://huggingface.co/ai21labs/AI21-Jamba-1.5-Mini](https://huggingface.co/ai21labs/AI21-Jamba-1.5-Mini)。
- **Transformers 4.40.0 可正常工作**：一位用户确认使用 transformers 4.40.0 解决了依赖问题，并允许他们成功训练 Jamba 1.5。
- **Transformers 4.44.2 发布说明**：transformers 4.44.2 的发布说明提到了对 Jamba 缓存故障的修复，但已确认该修复与限制 Jamba 架构的 bug **无关**。
   - 建议用户继续使用 transformers 4.40.0。



**提到的链接**：<a href="https://newreleases.io/project/pypi/transformers/release/4.44.2">transformers 4.44.2 on Python PyPI</a>：Python PyPI 上的新版本 transformers 4.44.2 发布。

  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1278838600510996593)** (1 messages): 

> - `Tinygrad Performance`
> - `Static Scheduling`
> - `Sparse Operations` 


- **Tinygrad 针对 Static Scheduling 进行了优化**：Tinygrad 确实针对静态调度操作进行了高度优化，在不涉及动态稀疏性或权重选择的任务中实现了显著的性能提升。
   - 对静态调度的关注使 Tinygrad 能够利用编译器优化并执行高效的内存管理。
- **动态稀疏操作的局限性**：虽然 Tinygrad 在静态调度方面表现出色，但在处理动态稀疏性或权重选择时可能会遇到性能限制。
   - 这些类型的操作需要内存管理和计算流的灵活性，而 Tinygrad 目前尚未完全支持。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1278559303061733387)** (7 messages): 

> - `ReduceOp Merging in Tinygrad`
> - `tinygrad's FUSE_CONV_BW Flag`
> - `Tinygrad Documentation for Beginners` 


- **Tinygrad 的 ReduceOp 合并行为**：一位用户询问了 Tinygrad 的 `schedule.py` 文件中大量 `# max one reduceop per kernel` 语句背后的逻辑，特别是其中一条有时会触发 reduction 的早期 realize，从而阻碍了它们在 `_recurse_reduceops` 函数中的合并。
   - 一位贡献者提供了背景信息，重点提到了解决此问题的 PR #6302。当链式调用 reduction 时（例如 `Tensor.randn(5,5).realize().sum(-1).sum()`），这个问题就会显现，此时 reductions 无法按预期合并为单个 sum。
- **FUSE_CONV_BW=1：卷积反向传播的未来**：一位贡献者解释说，Tinygrad 中的 `FUSE_CONV_BW=1` 标志目前通过在反向传播（backward pass）中启用卷积的高效融合来解决此 reduction 合并问题。
   - 他们还指出，一旦在所有场景下都实现了性能优化，该标志最终将成为默认设置。
- **Tinygrad 文档：你的起点**：一位用户询问如何开始学习 Tinygrad。
   - 多位贡献者建议从官方 Tinygrad 文档开始，这被认为是初学者的宝贵资源。



**提到的链接**：<a href="https://github.com/tinygrad/tinygrad/blob/cb61cfce2492e53dac4691e92774e2704351b3ed/tinygrad/engine/schedule.py#L294-L295)">tinygrad/tinygrad/engine/schedule.py at cb61cfce2492e53dac4691e92774e2704351b3ed · tinygrad/tinygrad</a>：你喜欢 pytorch？你喜欢 micrograd？你会爱上 tinygrad！❤️ - tinygrad/tinygrad

  

---



### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1278491184943202335)** (3 messages): 

> - `Groq Leaderboard` 


- **Groq 未出现在排行榜中**：一位成员询问为什么 **Groq** 不在排行榜（或变更日志）中。
   - 另一位成员回答说 **Groq** 尚未被添加，团队正在等待他们的 PR，预计将于下周提交。
- **Groq 的 PR 预计下周提交**：一位成员询问为什么 **Groq** 不在排行榜（或变更日志）中。
   - 另一位成员回答说 **Groq** 尚未被添加，团队正在等待他们的 PR，预计将于下周提交。


  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/)** (1 messages): 

spirit_from_germany: https://youtu.be/DP454c1K_vQ?si=qYWw6oU0sQC9FPv4
  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1278579806279041104)** (1 messages): 

> - `CLIP-AGIQA`
> - `AI-Generated Image Quality Assessment`
> - `CLIP for image quality assessment`
> - `Generative technologies`
> - `AIGIs` 


- **CLIP-AGIQA 提升 AI 生成图像质量评估**：一篇新论文提出了 CLIP-AGIQA，这是一种利用 CLIP 提升 AI 生成图像（AIGI）质量评估性能的方法。
   - 论文指出，当前模型难以应对日益增多且多样化的生成图像类别，而 CLIP 评估自然图像质量的能力可以扩展到 AIGIs。
- **AIGIs 应用广泛但质量参差不齐**：AIGIs 在日常生活中的广泛应用凸显了对稳健的图像质量评估技术的需求。
   - 尽管已存在一些模型，论文强调需要更先进的方法来评估这些多样化生成图像的质量。
- **CLIP 在图像质量评估中展现出潜力**：CLIP 作为一种视觉语言模型，在评估自然图像质量方面显示出巨大潜力。
   - 论文探索了将 CLIP 应用于生成图像的质量评估，认为它在这一领域同样有效。



**Link mentioned**: <a href="https://arxiv.org/abs/2408.15098">CLIP-AGIQA: Boosting the Performance of AI-Generated Image Quality Assessment with CLIP</a>：随着生成式技术的飞速发展，AI 生成图像（AIGIs）已广泛应用于日常生活的各个方面。然而，由于技术尚不成熟，其质量...

  

---



### **Alignment Lab AI ▷ #[general](https://discord.com/channels/1087862276448595968/1095458248712265841/)** (1 messages): 

teknium: https://x.com/nousresearch/status/1829143753036366325?s=46
  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1278745186813349950)** (1 messages): 

> - `Common Voice`
> - `Speech Data` 


- **Common Voice 征集贡献者！**：Common Voice 项目是一个开源平台，用于收集语音数据并构建一个多语言语音片段数据集，该数据集既免费又无版权限制。
   - 该项目的目标是帮助语音技术服务于所有使用不同语言和口音的用户。
- **提供贡献指南**：有兴趣贡献的人员可以在 [这里](https://github.com/common-voice/common-voice) 找到指南。
   - 如果发现文档过时、令人困惑或不完整，欢迎提交 Issue。
- **加入社区！**：您可以在 [Common Voice Matrix 频道](https://app.element.io/?updated=1.11.63#%2Froom%2F#common-voice:mozilla.org) 或 [论坛](https://discourse.mozilla.org/c/voice/239) 与团队联系。
   - 也可以通过发送邮件至 commonvoice@mozilla.com 寻求支持。



**Link mentioned**: <a href="https://app.element.io/?updated=1.11.63\#/room/\#common-voice:mozilla.org>)">Element</a>: 未找到描述

  

---



---



---



{% else %}


> 邮件中已截断完整的逐频道详情。
> 
> 如果您想查看完整详情，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})!
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}